"""Baseline inference runner for AML_env.

The script supports local LM Studio via an OpenAI-compatible base URL and keeps
the multi-task loop expected by the project validator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

try:
    from AML_env.client import AmlEnv
    from AML_env.models import AmlAction
except Exception:
    ROOT_DIR = Path(__file__).resolve().parent
    if str(ROOT_DIR) not in os.sys.path:
        os.sys.path.insert(0, str(ROOT_DIR))
    from client import AmlEnv
    from models import AmlAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1") or "http://127.0.0.1:1234"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
HF_TOKEN = os.getenv("HF_TOKEN") or "lm-studio"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
TASK_NAME = os.getenv("TASK_NAME", "aml_easy")
BENCHMARK = os.getenv("BENCHMARK", "AML_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))
TASKS = ["aml_easy", "aml_medium", "aml_hard"]

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

SYSTEM_PROMPT = (
    "You are a Tier 1 AML Compliance Investigator. "
    "Return exactly one JSON object with the nested shape {\"action\": {...}}. "
    "Allowed action types: query_transactions, search_transactions, get_kyc_record, submit_decision. "
    "Do not output markdown, code fences, or explanations."
)


def _clean_text(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").strip()


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _format_action(action: AmlAction) -> str:
    return json.dumps(action.model_dump(), separators=(",", ":"), ensure_ascii=True)


def _format_error(error: Optional[str]) -> str:
    return error if error else "null"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={_format_reward(reward)} done={str(done).lower()} error={_format_error(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(_format_reward(r) for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def _build_env() -> AmlEnv:
    if LOCAL_IMAGE_NAME:
        return AmlEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return AmlEnv(base_url=ENV_BASE_URL)


def _fallback_action() -> AmlAction:
    return AmlAction.model_validate(
        {
            "action": {
                "action_type": "submit_decision",
                "decision": "CLEAR",
                "evidence_links": [],
            }
        }
    )


def _model_action(client: OpenAI, observation: Any, history: list[str]) -> AmlAction:
    history_block = "\n".join(history[-5:]) if history else "No prior steps."
    user_prompt = (
        f"Alert:\n{observation.alert_details}\n\n"
        f"Observation:\n{json.dumps(observation.model_dump(), separators=(",", ":"), ensure_ascii=True)}\n\n"
        f"History:\n{history_block}\n\n"
        "Return the next JSON action."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = (completion.choices[0].message.content or "").strip()
        if not content:
            return _fallback_action()
        try:
            return AmlAction.model_validate_json(content)
        except Exception:
            return AmlAction.model_validate(json.loads(content))
    except Exception:
        return _fallback_action()


def run_episode(client: OpenAI, env: AmlEnv, task_name: str) -> tuple[bool, int, float, list[float]]:
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    observation = env.reset(task=task_name)
    final_done = False
    final_error: Optional[str] = None

    for step in range(1, MAX_STEPS + 1):
        if observation.done:
            break

        action = _model_action(client, observation, history)
        result = env.step(action)
        observation = result.observation

        reward = float(result.reward or 0.0)
        final_done = bool(result.done)
        final_error = observation.error_message
        steps_taken = step
        rewards.append(reward)

        action_text = _clean_text(_format_action(action))
        log_step(step=step, action=action_text, reward=reward, done=final_done, error=final_error)

        history.append(
            f"step={step} action={action_text} reward={_format_reward(reward)} done={str(final_done).lower()} "
            f"error={_format_error(final_error)} result={_clean_text(str(observation.last_action_result))}"
        )

        if final_done:
            break

    score = max(0.0, min(1.0, sum(rewards)))
    success = bool(final_done and final_error is None and score > 0.0)
    return success, steps_taken, score, rewards


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = _build_env()

    try:
        for task_name in TASKS:
            success, steps_taken, score, rewards = run_episode(client, env, task_name)
            _ = score
            log_end(success=success, steps=steps_taken, rewards=rewards)
    finally:
        env.close()


if __name__ == "__main__":
    main()