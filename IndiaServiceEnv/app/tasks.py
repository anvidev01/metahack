import json

TASKS_CONFIG = {
    "classify_and_route": {
        "difficulty": "easy",
        "max_steps": 3,
        "available_tools": []
    },
    "multi_turn_resolution": {
        "difficulty": "medium",
        "max_steps": 6,
        "available_tools": ["check_refund_status"]
    },
    "policy_conflict_escalation": {
        "difficulty": "hard",
        "max_steps": 8,
        "available_tools": ["check_complaint_history", "get_policy", "escalate_to_supervisor"]
    }
}
