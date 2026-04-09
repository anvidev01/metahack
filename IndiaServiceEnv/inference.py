"""
Baseline inference script for IndiaServiceEnv.
Uses OpenAI client. Reads API_BASE_URL, MODEL_NAME, HF_TOKEN
from environment variables, with safe defaults.
"""
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Logging helpers ──────────────────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
          flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
          flush=True)

# ── Environment variables with safe defaults ─────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "https://dobie17-indiaserviceenv.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["classify_and_route", "multi_turn_resolution", "policy_conflict_escalation"]

SYSTEM_PROMPT = """You are a customer service agent for Indian
utility and telecom companies.

You must respond with a JSON action in this exact format:
{
  "action_type": "classify"|"call_tool"|"respond"|"escalate"|"resolve",
  "content": "your response to customer",
  "tool_name": "tool_name_if_calling_tool or null",
  "tool_params": {"param": "value"} or null
}

Available tools: check_refund_status, check_complaint_history,
get_policy, escalate_to_supervisor

Always call required tools before resolving. Never hallucinate
tool results."""

# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    rewards = []
    steps_taken = 0
    last_score = 0.01   # absolute score from last step (server now returns absolute not incremental)
    obs = {}

    log_start(task_id, "IndiaServiceEnv", MODEL_NAME)

    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        obs = reset_resp.json()
    except Exception as e:
        print(f"[DEBUG] Reset failed: {e}", file=sys.stderr, flush=True)
        log_end(False, 0, 0.01, [])
        return 0.01

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False

    for step_num in range(1, 11):
        if done:
            break

        user_msg = f"""Ticket: {obs.get('customer_message', '')}
History: {json.dumps(obs.get('conversation_history', []))}
Available tools: {obs.get('available_tools', [])}
Step: {obs.get('current_step', 0)}/{obs.get('max_steps', 10)}"""

        messages.append({"role": "user", "content": user_msg})

        # LLM call — wrapped so it NEVER crashes
        raw = None
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=500,
                temperature=0.0
            )
            raw = response.choices[0].message.content
        except Exception as e:
            print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr, flush=True)
            raw = '{"action_type":"resolve","content":"fallback","tool_name":null,"tool_params":null}'

        messages.append({"role": "assistant", "content": raw})

        # Parse action — wrapped so it NEVER crashes
        try:
            action = json.loads(raw)
        except Exception:
            action = {
                "action_type": "respond",
                "content": raw or "fallback",
                "tool_name": None,
                "tool_params": None
            }

        # Step environment — wrapped so it NEVER crashes
        try:
            result = requests.post(
                f"{ENV_URL}/step",
                json=action,
                timeout=30
            ).json()
            obs = result["observation"]
            reward = float(result["reward"]["value"])  # absolute score from server
            done = result["done"]
        except Exception as e:
            print(f"[DEBUG] Step failed: {e}", file=sys.stderr, flush=True)
            reward = last_score   # keep previous score on error
            done = True

        # Server returns absolute score per step, always in (0.01, 0.99)
        reward = min(max(reward, 0.01), 0.99)  # safety clamp
        rewards.append(reward)
        last_score = reward   # track latest absolute score
        steps_taken = step_num

        log_step(
            step=step_num,
            action=action.get("action_type", "unknown"),
            reward=reward,
            done=done,
            error=None
        )

        if done:
            break

    # final_score = last absolute score from server, already in (0.01, 0.99)
    final_score = min(max(last_score, 0.01), 0.99)
    log_end(
        success=final_score >= 0.1,
        steps=steps_taken,
        score=final_score,
        rewards=rewards
    )
    return final_score


# ── Main entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    scores = {}
    try:
        for task in TASKS:
            try:
                scores[task] = run_task(task)
            except Exception as e:
                print(f"[DEBUG] Task {task} failed: {e}", file=sys.stderr, flush=True)
                scores[task] = 0.01
    except Exception as e:
        print(f"[DEBUG] Fatal: {e}", file=sys.stderr, flush=True)
