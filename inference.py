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

from server.AML_env_environment import AmlEnvironment
from models import AmlAction


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
HF_TOKEN = os.getenv("HF_TOKEN") or "lm-studio"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
TASK_NAME = os.getenv("TASK_NAME", "aml_easy")
TASKS = ["aml_easy", "aml_medium", "aml_hard"]
BENCHMARK = "aml_investigator"
MAX_STEPS = 25

OBS_RESULT_MAX_ITEMS = 8
HISTORY_MAX_STEPS = 3
HISTORY_MAX_CHARS = 1600
TEXT_CLIP_CHARS = 320

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Tier 1 AML Compliance Investigator.
    You must investigate the provided alert by querying the bank's internal APIs.

    You have a strict API budget. Be efficient.
    Respond with EXACTLY ONE valid JSON object representing your action. Do not include markdown formatting or explanations.

    Available Action JSON Schemas:
    1. {"action": {"action_type": "query_transactions", "account_id": "ACC-XXXX", "limit": 10, "offset": 0}}
    2. {"action": {"action_type": "search_transactions", "account_id": "ACC-XXXX", "keyword": "invoice"}}
    3. {"action": {"action_type": "get_kyc_record", "entity_id": "ENT-XXXX"}}
    4. {"action": {"action_type": "submit_decision", "decision": "FRAUD", "evidence_links": ["ACC-1234"]}} (Use "CLEAR" for False Positives with empty evidence_links).

    Required top-level JSON format:
    {
      "thought": {
        "observation": "...",
        "plan": "...",
        "action": "..."
      },
      "action": {...}
    }

    Thought rules:
    - Use caveman style: short, simple, low-token wording.
    - Keep thought informative but brief.
    - observation = what clue found now.
    - plan = next investigation goal.
    - action = exact tool call you will make now.

    Data rules:
    - get_kyc_record must use ENT-XXXX only, never ACC-XXXX.
    - submit_decision only when evidence is enough; else keep investigating.
    - Use only the alert, the current observation, and the recent history shown here.
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


def _clip_text(value: Any, max_chars: int = TEXT_CLIP_CHARS) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _compact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "txn_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "memo_text",
        "account_id",
        "owner_entity_id",
        "status",
        "entity_id",
        "name",
        "type",
        "registration_address",
        "directors",
    ]
    compact: Dict[str, Any] = {}
    for key in keep_keys:
        if key not in record:
            continue
        value = record.get(key)
        if key == "directors" and isinstance(value, list):
            compact[key] = value[:4]
            if len(value) > 4:
                compact["directors_truncated"] = len(value) - 4
            continue
        if isinstance(value, str):
            compact[key] = _clip_text(value, max_chars=180)
        else:
            compact[key] = value
    return compact


def _compact_action_result(last_action: Optional[str], value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        items = []
        for item in value[:OBS_RESULT_MAX_ITEMS]:
            if isinstance(item, dict):
                items.append(_compact_record(item))
            else:
                items.append(_clip_text(item))
        return {
            "kind": "list",
            "count": len(value),
            "items": items,
            "truncated": len(value) > OBS_RESULT_MAX_ITEMS,
            "source_action": last_action,
        }
    if isinstance(value, dict):
        return _compact_record(value)
    if isinstance(value, str):
        return _clip_text(value, max_chars=420)
    return value


def _build_model_observation(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "alert_details": obs_dict.get("alert_details"),
        "budget_remaining": obs_dict.get("budget_remaining"),
        "last_action": obs_dict.get("last_action"),
        "last_action_result": _compact_action_result(obs_dict.get("last_action"), obs_dict.get("last_action_result")),
        "error_message": _clip_text(obs_dict.get("error_message")) if obs_dict.get("error_message") else None,
        "done": obs_dict.get("done"),
        "reward": obs_dict.get("reward"),
    }


def _render_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No previous steps."
    entries = history[-HISTORY_MAX_STEPS:]
    lines = [json.dumps(item, ensure_ascii=True, separators=(",", ":")) for item in entries]
    while lines and len("\n".join(lines)) > HISTORY_MAX_CHARS:
        lines.pop(0)
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


def _normalize_thought(payload: Dict[str, Any]) -> None:
    action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
    action_type = action.get("action_type", "unknown")
    if "thought" not in payload or not isinstance(payload.get("thought"), dict):
        payload["thought"] = {
            "observation": "see current clue now.",
            "plan": "find next real link.",
            "action": f"do {action_type} now.",
        }
        return

    thought = payload["thought"]
    for key, fallback in (
        ("observation", "see clue now."),
        ("plan", "next check key link."),
        ("action", f"do {action_type} now."),
    ):
        value = thought.get(key)
        if not isinstance(value, str) or not value.strip():
            thought[key] = fallback
        else:
            thought[key] = _clip_text(value, max_chars=140)


def _try_validate_action_json(raw_text: str) -> Optional[str]:
    """Return canonical JSON string if valid, else None."""
    candidate = _coerce_json_object(raw_text)
    try:
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("top-level JSON is not an object")
        action = payload.get("action")
        if not isinstance(action, dict):
            raise ValueError("missing 'action' object")
        action_type = action.get("action_type")
        if not isinstance(action_type, str):
            raise ValueError("missing 'action_type' string")
        _normalize_thought(payload)
        return json.dumps(payload, ensure_ascii=True)
    except Exception:
        return None


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
        "Return exactly one JSON object with keys: thought, action."
    )
    parse_errors: List[str] = []

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            max_output_tokens=700,
        )
        raw_text = _extract_text_from_responses_api(response)
        canonical = _try_validate_action_json(raw_text)
        if canonical is not None:
            return canonical, False
        parse_errors.append("responses:invalid_json")
    except Exception as responses_exc:
        parse_errors.append(f"responses:{responses_exc}")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        raw_text = _extract_text_from_chat_completion(completion)
        canonical = _try_validate_action_json(raw_text)
        if canonical is not None:
            return canonical, False
        parse_errors.append("chat:invalid_json")
    except Exception as chat_exc:
        parse_errors.append(f"chat:{chat_exc}")

    try:
        completion = client.completions.create(
            model=MODEL_NAME,
            prompt=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            temperature=0.0,
            max_tokens=280,
        )
        raw_text = _extract_text_from_completions_api(completion)
        canonical = _try_validate_action_json(raw_text)
        if canonical is not None:
            return canonical, False
        parse_errors.append("completions:invalid_json")
    except Exception as completions_exc:
        parse_errors.append(f"completions:{completions_exc}")

    recovery_json = _build_recovery_action_from_obs(obs_dict, next_offsets)
    print(
        (
            "[DEBUG] Non-JSON/invalid model action; using recovery action "
            f"({'; '.join(parse_errors)})"
        ),
        file=sys.stderr,
        flush=True,
    )
    recovery_payload = {
        "thought": {
            "observation": "model output bad json.",
            "plan": "use safe step. keep investigate.",
            "action": "query alert account next page.",
        },
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
        query_seen_counts: Dict[Tuple[str, int], int] = {}

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = env.reset(task=task_name)

            for step in range(1, MAX_STEPS + 1):
                if obs.done:
                    break

                obs_dict = obs.model_dump()
                action_str, used_recovery = get_model_message(client, obs_dict, history, next_offsets)
                if used_recovery:
                    had_parse_error = True

                action_for_log = action_str
                action_payload_for_history: Dict[str, Any] = {}
                try:
                    clean_str = _coerce_json_object(action_str)
                    action_json = json.loads(clean_str)
                    thought_for_log = action_json.get("thought")
                    if thought_for_log is None:
                        action_type = action_json.get("action", {}).get("action_type", "unknown")
                        thought_for_log = f"do {action_type} now"
                    log_thought(step=step, thought=thought_for_log)
                    action_obj = AmlAction.model_validate(action_json)

                    action_payload_for_history = action_json.get("action", {}) if isinstance(action_json, dict) else {}
                    action_for_log = json.dumps({"action": action_payload_for_history}, ensure_ascii=True)
                    if action_payload_for_history.get("action_type") == "query_transactions":
                        acc = action_payload_for_history.get("account_id")
                        offset = int(action_payload_for_history.get("offset", 0))
                        limit = int(action_payload_for_history.get("limit", 10))
                        if isinstance(acc, str):
                            query_key = (acc, offset)
                            query_seen_counts[query_key] = query_seen_counts.get(query_key, 0) + 1
                            # Hard guardrail: avoid wasting budget on repeated same page.
                            if task_name == "aml_hard" and query_seen_counts[query_key] > 2:
                                new_offset = max(next_offsets.get(acc, offset + max(limit, 1)), offset + max(limit, 1))
                                action_json["action"]["offset"] = new_offset
                                action_json["thought"]["plan"] = _clip_text(
                                    f"repeat page seen. move to next offset {new_offset}.",
                                    max_chars=120,
                                )
                                action_json["thought"]["action"] = _clip_text(
                                    f"query_transactions {acc} offset {new_offset}",
                                    max_chars=120,
                                )
                                action_for_log = json.dumps(action_json, ensure_ascii=True)
                                action_obj = AmlAction.model_validate(action_json)
                                offset = new_offset
                            next_offsets[acc] = max(next_offsets.get(acc, 0), offset + max(limit, 1))
                    error = None
                except Exception as e:
                    had_parse_error = True
                    error = f"JSON Parse/Schema Error: {str(e)}"
                    log_thought(step=step, thought="parse fail; use recovery action")
                    recovery_json = _build_recovery_action_from_obs(obs_dict, next_offsets)
                    recovery_payload = {
                        "thought": {
                            "observation": "parse fail now.",
                            "plan": "safe step, keep digging.",
                            "action": "query alert next page.",
                        },
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
                history.append(
                    {
                        "step": step,
                        "action": action_payload_for_history,
                        "result": _compact_action_result(obs.last_action, obs.last_action_result),
                        "error": _clip_text(obs.error_message) if obs.error_message else None,
                        "budget_remaining": obs.budget_remaining,
                    }
                )
                if len(history) > 24:
                    history = history[-24:]

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
