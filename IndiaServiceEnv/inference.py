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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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
    rewards      = []
    steps_taken  = 0
    total_reward = 0.0

    log_start(task_id, "IndiaServiceEnv", MODEL_NAME)

    try:
        # Reset environment
        try:
            obs = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": task_id},
                timeout=30
            ).json()
        except Exception as e:
            print(f"[DEBUG] Reset failed: {e}", flush=True)
            return 0.0

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False

        while not done and steps_taken < 10:
            # Build user message from observation
            user_msg = (
                f"Ticket: {obs['customer_message']}\n"
                f"History: {json.dumps(obs['conversation_history'])}\n"
                f"Available tools: {obs['available_tools']}\n"
                f"Step: {obs['current_step']}/{obs['max_steps']}"
            )
            messages.append({"role": "user", "content": user_msg})

            # LLM call – wrapped fully so it never crashes
            raw = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=500,
                        temperature=0.0
                    )
                    raw = response.choices[0].message.content
                    break
                except Exception as e:
                    print(f"[DEBUG] LLM call failed (attempt {attempt+1}): {e}", flush=True)
                    if "429" in str(e):
                        time.sleep(15)
                    else:
                        break   # non-rate-limit error – use fallback immediately

            if raw is None:
                raw = '{"action_type":"resolve","content":"error fallback","tool_name":null,"tool_params":null}'

            messages.append({"role": "assistant", "content": raw})

            # Parse action
            try:
                action = json.loads(raw)
            except Exception:
                action = {"action_type": "respond", "content": raw,
                          "tool_name": None, "tool_params": None}

            # Step environment
            try:
                result = requests.post(
                    f"{ENV_URL}/step",
                    json=action,
                    timeout=30
                ).json()
            except Exception as e:
                print(f"[DEBUG] Step failed: {e}", flush=True)
                break

            obs         = result["observation"]
            reward      = result["reward"]["value"]
            done        = result["done"]

            total_reward += reward
            steps_taken  += 1
            rewards.append(reward)

            action_type = action.get("action_type", "respond")
            error_val   = result.get("error", None)

            log_step(steps_taken, action_type, reward, done, error=error_val)

    except Exception as e:
        print(f"[DEBUG] Task failed: {e}", flush=True)

    finally:
        score   = min(max(total_reward, 0.0), 1.0)
        success = score >= 0.1
        log_end(success, steps_taken, score, rewards)

    return total_reward


# ── Main entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    scores = {}
    try:
        for task in TASKS:
            scores[task] = run_task(task)
    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)
    finally:
        print(json.dumps({
            "type": "SUMMARY",
            "scores": scores if "scores" in dir() else {}
        }), flush=True)
