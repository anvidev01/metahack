import json
import os
import random
from typing import Tuple, Dict, Any

from app.models import Action, Observation, Reward
from app.tasks import TASKS_CONFIG
from app.graders import evaluate_action

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

class IndiaServiceEnv:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.task_config = TASKS_CONFIG.get(self.task_id, {})
        self.max_steps = self.task_config.get("max_steps", 5)
        self.available_tools = self.task_config.get("available_tools", [])

        self._state = {
            "task_id": self.task_id,
            "ticket": {},
            "conversation_history": [],
            "classification": None,
            "tools_called": [],
            "resolution": None,
            "current_step": 0,
            "score_breakdown": {},
            "consecutive_invalid_actions": 0,
            "max_steps": self.max_steps
        }
        
        # Load data gracefully so state() works before reset
        try:
            with open(os.path.join(DATA_DIR, "tickets.json"), "r") as f:
                all_tickets = json.load(f).get("tickets", [])
        except:
            all_tickets = []
            
        self.task_tickets = [t for t in all_tickets if t.get("task_id") == self.task_id]
        
        # Max steps and tools already initialized above
        
    def reset(self) -> Observation:
        # 3. reset() must always return the same ticket for the same task_id
        random.seed(self.task_id)
        if self.task_tickets:
            # Deterministically shuffle or just pick first, but prompt says "same ticket".
            # We sort to ensure order and always pick the first to be fully deterministic.
            ticket = sorted(self.task_tickets, key=lambda x: x["ticket_id"])[0]
        else:
            ticket = {"customer_message": "Dummy ticket", "ticket_id": "dummy"}
            
        self._state = {
            "task_id": self.task_id,
            "ticket": ticket,
            "conversation_history": [],
            "classification": None,
            "tools_called": [],
            "resolution": None,
            "current_step": 0,
            "score_breakdown": {},
            "consecutive_invalid_actions": 0,
            "max_steps": self.max_steps
        }
        
        return self._get_observation()
        
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._state["current_step"] += 1
        
        self._state["conversation_history"].append({
            "role": "agent",
            "action": action.model_dump() if hasattr(action, "model_dump") else action.dict()
        })
        
        if action.action_type == "call_tool" and action.tool_name:
            if action.tool_name not in self.available_tools:
                self._state["consecutive_invalid_actions"] += 1
                tool_result = {"error": f"Tool {action.tool_name} not available for this task."}
            else:
                self._state["tools_called"].append(action.tool_name)
                self._state["consecutive_invalid_actions"] = 0
                
                # Mock tool routing
                if action.tool_name == "check_refund_status":
                    pnr = action.tool_params.get("pnr") if action.tool_params else None
                    tool_result = self.check_refund_status(pnr)
                elif action.tool_name == "check_complaint_history":
                    account_id = action.tool_params.get("account_id") if action.tool_params else None
                    tool_result = self.check_complaint_history(account_id)
                elif action.tool_name == "get_policy":
                    policy_type = action.tool_params.get("policy_type") if action.tool_params else None
                    tool_result = self.get_policy(policy_type)
                elif action.tool_name == "escalate_to_supervisor":
                    reason = action.tool_params.get("reason") if action.tool_params else None
                    tool_result = self.escalate_to_supervisor(reason)
                else:
                    tool_result = {"error": "Unknown tool"}
                    
            self._state["conversation_history"].append({
                "role": "system",
                "action": {"type": "tool_response", "result": tool_result}
            })
        else:
            self._state["consecutive_invalid_actions"] = 0
            
        if action.action_type == "classify":
            self._state["classification"] = action.content
        elif action.action_type == "resolve":
            self._state["resolution"] = action.content
            
        score, breakdown, done = evaluate_action(self.task_id, self._state, action)
        
        self._state["score_breakdown"] = breakdown
        
        if self._state["consecutive_invalid_actions"] >= 3:
            done = True
            
        obs = self._get_observation()
        reward = Reward(
            value=score,
            breakdown=breakdown,
            done=done,
            info={}
        )
        
        return obs, reward, done, {}
        
    def state(self) -> Dict[str, Any]:
        return self._state
        
    def _get_observation(self) -> Observation:
        return Observation(
            ticket_id=self._state["ticket"].get("ticket_id", "unknown"),
            customer_message=self._state["ticket"].get("customer_message", ""),
            conversation_history=self._state["conversation_history"],
            available_tools=self.available_tools,
            current_step=self._state["current_step"],
            max_steps=self.max_steps,
            task_id=self.task_id
        )

    # --- MOCK TOOLS ---
    def check_refund_status(self, pnr: str) -> dict:
        if not pnr: return {"error": "Missing PNR"}
        # For multi_turn_resolution, always return 8 days so they escalate as per policy
        return {
            "status": "pending",
            "days_since_cancellation": 8,
            "amount": 1500.0
        }
        
    def check_complaint_history(self, account_id: str) -> dict:
        return {
            "existing_complaints": ["COMP-9901 (active)"],
            "oldest_dispute_months": 3
        }
        
    def get_policy(self, policy_type: str) -> dict:
        return {
            "refund_percentage": 50,
            "timeline_days": 14
        }
        
    def escalate_to_supervisor(self, reason: str) -> dict:
        return {
            "ticket_id": self._state["ticket"].get("ticket_id", "unknown"),
            "escalated": True
        }
