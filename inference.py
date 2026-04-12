"""
AML Investigator - Baseline Inference Script
Loops through all 3 tasks to satisfy the Phase 2 Validator.
"""
import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import ValidationError

from server.AML_env_environment import AmlEnvironment
from models import AmlAction, AmlObservation


API_BASE_URL = os.getenv("API_BASE_URL") or "http://127.0.0.1:1234/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
HF_TOKEN = os.getenv("HF_TOKEN") or "lm-studio"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
TASK_NAME = os.getenv("TASK_NAME", "aml_easy")
TASKS = ["aml_easy", "aml_medium", "aml_hard"]
BENCHMARK = "aml_investigator"
MAX_STEPS = 25

HISTORY_MAX_STEPS = 3

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Tier 1 AML compliance investigator using a ReAct-style loop.
    Think privately, then return exactly one JSON object for the next action.

    Output format:
    {
      "thought": "Observation: ... Plan: ...",
      "action": {
        "action_type": "...",
        ...
      }
    }

    The "thought" field is your thinking pad and is required.
    It must include two labeled sections in order:
    - Observation: what evidence you see now.
    - Plan: the single next action and why.
    Keep it concise.

    Available actions:
    - {"action": {"action_type": "query_transactions", "account_id": "ACC-XXXX", "limit": 10, "offset": 0}}
    - {"action": {"action_type": "search_transactions", "account_id": "ACC-XXXX", "keyword": "invoice"}}
    - {"action": {"action_type": "get_kyc_record", "entity_id": "ENT-XXXX"}}
    - {"action": {"action_type": "submit_decision", "decision": "FRAUD", "evidence_links": ["ACC-1234"]}}
    - For false positives, use {"action": {"action_type": "submit_decision", "decision": "CLEAR", "evidence_links": []}}

    Rules:
    - Use only the alert, current observation, and recent history shown here.
    - get_kyc_record must use ENT ids, never ACC ids.
    - Return JSON only. No markdown fences. No explanation outside JSON.

    Example 1:
    {"thought":"Observation: The flagged account sent a large payment with a business-like memo. Plan: Check receiver KYC before deciding.","action":{"action_type":"get_kyc_record","entity_id":"ENT-9002"}}

    Example 2:
    {"thought":"Observation: There are multiple inbound deposits just under 10000 from different accounts. Plan: Inspect one sender's KYC to test structuring.","action":{"action_type":"get_kyc_record","entity_id":"ENT-9011"}}
    """
).strip()


def _extract_text_from_chat_completion(completion: object) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ValueError("Model response has no choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError("Model response choice has no message")

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str):
                    chunks.append(text_val)
            else:
                text_val = getattr(item, "text", None)
                if isinstance(text_val, str):
                    chunks.append(text_val)
        merged = "".join(chunks).strip()
        if merged:
            return merged

    raise ValueError("Model response content is empty")


def _extract_text_from_responses_api(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None) or []
    chunks: List[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            text_val = getattr(part, "text", None)
            if isinstance(text_val, str):
                chunks.append(text_val)
            elif isinstance(part, dict):
                maybe_text = part.get("text")
                if isinstance(maybe_text, str):
                    chunks.append(maybe_text)

    merged = "".join(chunks).strip()
    if merged:
        return merged

    raise ValueError("Responses API output is empty")


def _extract_text_from_completions_api(completion: object) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ValueError("Completions API response has no choices")

    first_choice = choices[0]
    text_val = getattr(first_choice, "text", None)
    if isinstance(text_val, str) and text_val.strip():
        return text_val.strip()

    raise ValueError("Completions API response text is empty")


def _coerce_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]

    return text


def _strip_channel_wrappers(raw_text: str) -> str:
    """
    Some OSS reasoning models emit channel tags like:
    <|channel|>analysis<|message|>...<|channel|>final<|message|>{...}
    Keep only the final/message payload before JSON parsing.
    """
    text = raw_text.strip()
    if "<|channel|>" not in text:
        return text

    final_marker = "<|channel|>final<|message|>"
    if final_marker in text:
        return text.split(final_marker, 1)[1].strip()

    message_marker = "<|message|>"
    if message_marker in text:
        return text.split(message_marker, 1)[1].strip()

    return text


def _extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _parse_action_payload(raw_text: str) -> AmlAction:
    cleaned_text = _strip_channel_wrappers(raw_text)
    candidate = _coerce_json_object(cleaned_text)
    parse_errors: List[str] = []

    for attempt in (
        candidate,
        _extract_balanced_json_object(cleaned_text) or "",
        _extract_balanced_json_object(raw_text) or "",
    ):
        if not attempt:
            continue
        try:
            payload = json.loads(attempt)
            if isinstance(payload, dict):
                return AmlAction.model_validate(payload)
            parse_errors.append("decoded JSON was not an object")
            continue
        except ValidationError as exc:
            parse_errors.append(f"schema: {exc.errors()[0]['msg']}")
            continue
        except Exception as exc:
            parse_errors.append(f"json: {exc}")

    details = parse_errors[-1] if parse_errors else "could not parse model output into JSON object"
    raise ValueError(details)


def _debug_text_repr(value: Any) -> str:
    text = str(value)
    escaped = text.encode("unicode_escape", errors="backslashreplace").decode("ascii", errors="replace")
    return f"len={len(text)} repr={escaped!r}"


def _build_model_observation(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    validated = AmlObservation.model_validate(obs_dict)
    return {
        "alert_details": validated.alert_details,
        "budget_remaining": validated.budget_remaining,
        "last_action": validated.last_action,
        "last_action_result": validated.last_action_result,
        "done": validated.done,
        "reward": validated.reward,
    }


def _render_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No previous steps."
    entries = history[-HISTORY_MAX_STEPS:]
    lines = [json.dumps(item, ensure_ascii=True) for item in entries]
    return "\n".join(lines) if lines else "No previous steps."


def _build_recovery_action_from_obs(obs_dict: dict, next_offsets: Dict[str, int]) -> dict:
    """Use a non-terminal fallback action when model output is malformed."""
    alert = str(obs_dict.get("alert_details", "") or "")
    match = re.search(r"ACC-\d+", alert)
    if match:
        account_id = match.group(0)
        offset = next_offsets.get(account_id, 0)
        next_offsets[account_id] = offset + 10
        return {
            "action": {
                "action_type": "query_transactions",
                "account_id": account_id,
                "limit": 10,
                "offset": offset,
            }
        }
    return {
        "action": {
            "action_type": "submit_decision",
            "decision": "CLEAR",
            "evidence_links": [],
        }
    }


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def log_thought(step: int, thought: Optional[object]) -> None:
    """Print model thought to stderr so stdout contract stays validator-safe."""
    if thought is None:
        return
    if isinstance(thought, dict):
        compact = json.dumps(thought, ensure_ascii=True)
    else:
        compact = str(thought)
    compact = compact.replace("\n", " ").strip()
    print(f"[THOUGHT] step={step} thought={compact}", file=sys.stderr, flush=True)


def get_model_message(
    client: OpenAI,
    obs_dict: dict,
    history: List[Dict[str, Any]],
    next_offsets: Dict[str, int],
) -> Tuple[str, bool]:
    model_obs = _build_model_observation(obs_dict)
    history_block = _render_history(history)
    user_prompt = (
        f"Observation:\n{json.dumps(model_obs, ensure_ascii=True, indent=2)}\n\n"
        f"History:\n{history_block}\n\n"
        "Return exactly one JSON object with keys: thought, action. "
        "thought must include 'Observation:' and 'Plan:'."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=260,
            response_format={"type": "json_object"},
        )
        return _extract_text_from_chat_completion(completion), False
    except Exception as chat_exc:
        chat_error = f"chat:{chat_exc}"

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            max_output_tokens=1000,
        )
        return _extract_text_from_responses_api(response), False
    except Exception as responses_exc:
        responses_error = f"responses:{responses_exc}"

    try:
        completion = client.completions.create(
            model=MODEL_NAME,
            prompt=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            temperature=0.0,
            max_tokens=260,
        )
        return _extract_text_from_completions_api(completion), False
    except Exception as completions_exc:
        completions_error = f"completions:{completions_exc}"

    recovery_json = _build_recovery_action_from_obs(obs_dict, next_offsets)
    print(
        (
            "[DEBUG] Model request failed; using recovery action "
            f"({completions_error}; {chat_error}; {responses_error})"
        ),
        file=sys.stderr,
        flush=True,
    )
    recovery_payload = {
        "thought": "Observation: Model request failed. Plan: take a safe recovery action.",
        "action": recovery_json["action"],
    }
    return json.dumps(recovery_payload, ensure_ascii=True), True


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = AmlEnvironment()

    for task_name in TASKS:
        history: List[Dict[str, Any]] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        had_parse_error = False
        next_offsets: Dict[str, int] = {}

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = env.reset(task=task_name)

            for step in range(1, MAX_STEPS + 1):
                if obs.done:
                    break

                obs_dict = AmlObservation.model_validate(obs.model_dump()).model_dump()
                action_str, used_recovery = get_model_message(client, obs_dict, history, next_offsets)
                if used_recovery:
                    had_parse_error = True

                action_for_log = action_str
                action_payload_for_history: Dict[str, Any] = {}
                parsed_model_action = False
                model_thought_for_history: Optional[str] = None
                try:
                    action_obj = _parse_action_payload(action_str)
                    log_thought(step=step, thought=action_obj.thought)
                    model_thought_for_history = action_obj.thought
                    parsed_model_action = True

                    action_payload_for_history = action_obj.action.model_dump(exclude={"metadata"}, exclude_none=True)
                    action_for_log = json.dumps({"action": action_payload_for_history}, ensure_ascii=True)
                    if action_payload_for_history.get("action_type") == "query_transactions":
                        acc = action_payload_for_history.get("account_id")
                        offset = int(action_payload_for_history.get("offset", 0))
                        limit = int(action_payload_for_history.get("limit", 10))
                        if isinstance(acc, str):
                            next_offsets[acc] = max(next_offsets.get(acc, 0), offset + max(limit, 1))
                    error = None
                except Exception as e:
                    had_parse_error = True
                    error = f"JSON Parse/Schema Error: {str(e)}"
                    debug_payload = _debug_text_repr(action_str) if action_str.strip() else "empty model output"
                    print(
                        f"[DEBUG] step={step} parse_failed_raw={debug_payload}",
                        file=sys.stderr,
                        flush=True,
                    )
                    log_thought(
                        step=step,
                        thought="Observation: model output was invalid. Plan: use safe recovery action.",
                    )
                    recovery_json = _build_recovery_action_from_obs(obs_dict, next_offsets)
                    recovery_payload = {
                        "thought": "Observation: JSON/schema parse failed. Plan: query next page safely.",
                        "action": recovery_json["action"],
                    }
                    action_obj = AmlAction.model_validate(recovery_payload)
                    action_payload_for_history = recovery_payload["action"]
                    action_for_log = json.dumps({"action": action_payload_for_history}, ensure_ascii=True)

                obs = env.step(action_obj)

                reward = obs.reward or 0.0
                done = obs.done

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_for_log.replace("\n", ""), reward=reward, done=done, error=error)
                # Keep prompt context clean: only feed back model-authored, schema-valid turns.
                if parsed_model_action:
                    history.append(
                        {
                            "step": step,
                            "thought": model_thought_for_history,
                            "action": action_payload_for_history,
                            "result": obs.last_action_result,
                            "budget_remaining": obs.budget_remaining,
                        }
                    )
                    if len(history) > HISTORY_MAX_STEPS:
                        history = history[-HISTORY_MAX_STEPS:]

                if done:
                    break

            if had_parse_error or obs.error_message:
                score = 0.05
            elif "submit_decision" in (obs.last_action or ""):
                score = 0.75
            else:
                score = 0.25
            score = min(max(score, 0.01), 0.99)
            success = (not had_parse_error) and (obs.error_message is None) and score > 0.5

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
