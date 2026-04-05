"""
Baseline inference script for IndiaServiceEnv.
Uses OpenAI client. Reads API_BASE_URL, MODEL_NAME, HF_TOKEN 
from environment variables.
"""
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import os
import json
import requests
from openai import OpenAI

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

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

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

def run_task(task_id: str) -> float:
    log_start(task_id, "IndiaServiceEnv", MODEL_NAME)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    done = False
    step_num = 0
    rewards_list = []
    
    try:
        # Reset environment
        obs = requests.post(f"{ENV_URL}/reset", 
                            json={"task_id": task_id}).json()
        
        while not done and step_num < 10:
            # Build user message from observation
            user_msg = f"""Ticket: {obs['customer_message']}
History: {json.dumps(obs['conversation_history'])}
Available tools: {obs['available_tools']}
Step: {obs['current_step']}/{obs['max_steps']}"""
            
            messages.append({"role": "user", "content": user_msg})
            
            # LLM call with retry for rate limit
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=500,
                        temperature=0.0
                    )
                    break
                except Exception as e:
                    if "429" in str(e):
                        time.sleep(15)
                    else:
                        raise e
            
            raw = response.choices[0].message.content
            messages.append({"role": "assistant", "content": raw})
            
            # Parse action
            try:
                action = json.loads(raw)
            except:
                action = {"action_type": "respond", "content": raw,
                          "tool_name": None, "tool_params": None}
            
            # Step environment
            result = requests.post(f"{ENV_URL}/step", 
                                   json=action).json()
            
            obs = result["observation"]
            reward = result["reward"]["value"]
            done = result["done"]
            
            total_reward += reward
            step_num += 1
            rewards_list.append(reward)
            
            action_type = action.get("action_type", "respond")
            error_val = result.get("error", None)
            
            log_step(step_num, action_type, reward, done, error=error_val)
            
    finally:
        score = max(0.0, min(1.0, total_reward))
        success = score >= 0.1
        log_end(success, step_num, score, rewards_list)
        
    return total_reward

if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
