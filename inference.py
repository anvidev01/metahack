"""
Baseline inference script for IndiaServiceEnv.
Uses OpenAI client. Reads API_BASE_URL, MODEL_NAME, HF_TOKEN 
from environment variables.
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore")

import os
import json
import urllib.request

def post_json(url, data, timeout=30):
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", 
          flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", 
          flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", 
          flush=True)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "https://dobie17-indiaserviceenv.hf.space")

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

def run_task(task_id: str) -> float:
    log_start(task_id, "IndiaServiceEnv", MODEL_NAME)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    done = False
    step_num = 0
    rewards_list = []
    
    try:
        # Reset environment
        obs = post_json(f"{ENV_URL}/reset", {"task_id": task_id}, timeout=30)
        
        while not done and step_num < 10:
            # Build user message from observation
            user_msg = f"""Ticket: {obs['customer_message']}
History: {json.dumps(obs['conversation_history'])}
Available tools: {obs['available_tools']}
Step: {obs['current_step']}/{obs['max_steps']}"""
            
            messages.append({"role": "user", "content": user_msg})
            
            # LLM call with retry for rate limit
            import time
            raw = None
            for attempt in range(3):
                try:
                    req_data = {
                        "model": MODEL_NAME,
                        "messages": messages,
                        "max_tokens": 500,
                        "temperature": 0.0
                    }
                    req_url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f"Bearer {HF_TOKEN}"
                    }
                    req = urllib.request.Request(
                        req_url,
                        data=json.dumps(req_data).encode('utf-8'),
                        headers=headers
                    )
                    with urllib.request.urlopen(req, timeout=30) as response:
                        resp_json = json.loads(response.read().decode('utf-8'))
                        raw = resp_json['choices'][0]['message']['content']
                    break
                except Exception as e:
                    if "429" in str(e):
                        time.sleep(15)
                    else:
                        raw = '{"action_type":"resolve","content":"fallback","tool_name":null,"tool_params":null}'
                        break
            if raw is None:
                raw = '{"action_type":"resolve","content":"fallback","tool_name":null,"tool_params":null}'
            messages.append({"role": "assistant", "content": raw})
            
            # Parse action
            try:
                action = json.loads(raw)
            except:
                action = {"action_type": "respond", "content": raw,
                          "tool_name": None, "tool_params": None}
            
            # Step environment
            try:
                result = post_json(f"{ENV_URL}/step", action, timeout=30)
                obs = result["observation"]
                reward = float(result["reward"]["value"])
                done = result["done"]
            except Exception as e:
                print(f"[DEBUG] Step failed: {e}", flush=True)
                reward = 0.01
                done = True
            
            total_reward += reward
            step_num += 1
            rewards_list.append(reward)
            
            action_type = action.get("action_type", "respond")
            error_val = result.get("error", None)
            
            log_step(step_num, action_type, reward, done, error=error_val)
            
    finally:
        score = max(0.01, min(0.99, total_reward))
        success = score >= 0.1
        log_end(success, step_num, score, rewards_list)
        
    return score

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        try:
            scores[task] = run_task(task)
        except Exception as e:
            print(f"[DEBUG] Task {task} failed: {e}", flush=True)
            scores[task] = 0.01
    print(json.dumps({"type": "SUMMARY", "scores": scores}))
