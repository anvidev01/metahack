from pydantic import BaseModel
from typing import Any, Optional, Literal

class Action(BaseModel):
    action_type: Literal["classify", "call_tool", "respond", "escalate", "resolve"]
    content: str
    tool_name: Optional[str] = None
    tool_params: Optional[dict] = None

class Observation(BaseModel):
    ticket_id: str
    customer_message: str
    conversation_history: list[dict]
    available_tools: list[str]
    current_step: int
    max_steps: int
    task_id: str

class Reward(BaseModel):
    value: float  # 0.0 to 1.0
    breakdown: dict[str, float]
    done: bool
    info: dict[str, Any]
