"""
AML Investigator - Baseline Inference Script
Loops through all 3 tasks to satisfy the Phase 2 Validator.
"""
import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# Adjust the import based on your openenv server setup
from openenv.core.env_server.interfaces import Environment
# If running locally without docker wrapper for validation, you might need to import your Env directly
from server.AML_env_environment import AmlEnvironment
from models import AmlAction

API_KEY = os.getenv("HF_TOKEN") or "lm-studio"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1" or "http://localhost:1234/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-oss-20b"

# Must match openenv.yaml EXACTLY
TASKS = ["aml_easy", "aml_medium", "aml_hard"]
BENCHMARK = "aml_investigator"
MAX_STEPS = 25

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
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_message(client: OpenAI, obs_dict: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-5:]) if history else "No previous steps."
    user_prompt = f"Observation:\n{json.dumps(obs_dict, indent=2)}\n\nHistory:\n{history_block}\n\nProvide your next JSON action:"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback to prevent crash
        return '{"action": {"action_type": "submit_decision", "decision": "CLEAR", "evidence_links": []}}'

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize your environment natively for the baseline script
    env = AmlEnvironment()

    for task_name in TASKS:
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = env.reset(task=task_name)
            
            for step in range(1, MAX_STEPS + 1):
                if obs.done:
                    break

                obs_dict = obs.model_dump()
                action_str = get_model_message(client, obs_dict, history)
                
                # Parse LLM string to Pydantic Model
                try:
                    # Strip possible markdown backticks
                    clean_str = action_str.replace("```json", "").replace("```", "").strip()
                    action_json = json.loads(clean_str)
                    action_obj = AmlAction.model_validate(action_json)
                    error = None
                except Exception as e:
                    # Errors are data! If the LLM writes bad JSON, we catch it and force a dummy action 
                    # so the environment can return a schema error to the LLM.
                    error = f"JSON Parse/Schema Error: {str(e)}"
                    action_obj = AmlAction(action={"action_type": "submit_decision", "decision": "CLEAR", "evidence_links": []})

                obs = env.step(action_obj)
                
                reward = obs.reward or 0.0
                done = obs.done

                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_str.replace('\n', ''), reward=reward, done=done, error=error)
                history.append(f"Step {step}: Action: {action_str} -> Result: {obs.last_action_result} | Error: {obs.error_message}")

                if done:
                    break

            # Calculate a baseline score for the stdout logs (Graders handle real scoring)
            score = sum(rewards) + 1.0 if "submit_decision" in (obs.last_action or "") else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score > 0.5

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())