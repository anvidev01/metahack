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
    # Reset environment
    obs = requests.post(f"{ENV_URL}/reset", 
                        json={"task_id": task_id}).json()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    done = False
    step_num = 0
    
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "observation": obs
    }))
    
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
        
        print(json.dumps({
            "type": "STEP",
            "task_id": task_id,
            "step": step_num,
            "action": action,
            "reward": reward,
            "done": done,
            "breakdown": result["reward"]["breakdown"]
        }))
    
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "total_reward": total_reward,
        "steps_taken": step_num
    }))
    
    return total_reward

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)
    print(json.dumps({"type": "SUMMARY", "scores": scores}))
