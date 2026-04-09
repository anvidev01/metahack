import re

def grade_classify_and_route(state):
    score = 0.01
    breakdown = {
        "correct_classification": 0.0,
        "correct_routing": 0.0
    }
    
    # We will search the history for actions taken by the agent
    agent_actions = [x["action"] for x in state["conversation_history"] if x["role"] == "agent"]
    
    classification_found = False
    routing_found = False
    
    for act in agent_actions:
        act_str = str(act).lower()
        if "bill" in act_str or "amount" in act_str or act.get("action_type") == "classify":
            classification_found = True
        if "department" in act_str or act.get("action_type") in ["escalate", "route"]:
            routing_found = True
            
    if classification_found:
        breakdown["correct_classification"] = 0.49
        score += 0.49
    if routing_found:
        breakdown["correct_routing"] = 0.5
        score += 0.5
        
    return score, breakdown

def grade_multi_turn_resolution(state):
    score = 0.01
    breakdown = {
        "asked_for_pnr": 0.0,
        "called_tool_correctly": 0.0,
        "interpreted_tool_correctly": 0.0,
        "correct_resolution": 0.0,
        "redundancy_penalty": 0.0
    }
    
    agent_actions = [x["action"] for x in state["conversation_history"] if x["role"] == "agent"]
    
    # check if PNR was asked
    asked_pnr = False
    for act in agent_actions:
        if act.get("action_type") == "respond" and "pnr" in act.get("content", "").lower():
            asked_pnr = True
    if asked_pnr:
        breakdown["asked_for_pnr"] = 0.19
        score += 0.19
        
    # check tool
    called_tool = False
    for act in agent_actions:
        if act.get("action_type") == "call_tool" and act.get("tool_name") == "check_refund_status":
            called_tool = True
    if called_tool:
        breakdown["called_tool_correctly"] = 0.3
        score += 0.3
        
    # interpret & resolution
    # for simplicity, let's look for resolution keywords if tool was called
    resolved = False
    interpreted = False
    for act in agent_actions:
        if act.get("action_type") in ["resolve", "respond", "escalate"]:
            content = act.get("content", "").lower()
            if "escalate" in content or "wait" in content or "days" in content:
                interpreted = True
            if act.get("action_type") == "resolve" or "escalate" in content or "wait" in content:
                resolved = True
                
    if called_tool and interpreted:
        breakdown["interpreted_tool_correctly"] = 0.2
        score += 0.2
    
    if called_tool and resolved:
        breakdown["correct_resolution"] = 0.3
        score += 0.3
        
    return score, breakdown

def grade_policy_conflict_escalation(state):
    score = 0.01
    breakdown = {
        "detected_existing_complaint": 0.0,
        "called_history_tool": 0.0,
        "correct_decision": 0.0,
        "correct_refund_computed": 0.0,
        "resolution_plan_complete": 0.0
    }
    
    agent_actions = [x["action"] for x in state["conversation_history"] if x["role"] == "agent"]
    
    has_history = False
    for act in agent_actions:
        if act.get("action_type") == "call_tool" and act.get("tool_name") == "check_complaint_history":
            has_history = True
            
    if has_history:
        breakdown["called_history_tool"] = 0.19
        score += 0.19
        # If they called the tool, we assume they detected the complaint based on prompt requirements, 
        # but let's give points if they mentioned it:
        breakdown["detected_existing_complaint"] = 0.15
        score += 0.15
        
    # check resolution decision
    decision_made = False
    refund_computed = False
    plan_complete = False
    
    for act in agent_actions:
        content = act.get("content", "").lower()
        if "merge" in content or "escalate" in content:
            decision_made = True
            
        if "50%" in content and "25%" not in content:
            refund_computed = True
            
        if act.get("action_type") == "resolve":
            # Check fields
            if "comp-9901" in content and ("escalate" in content or "merge" in content) and "50%" in content and ("14" in content or "days" in content):
                plan_complete = True
                
    if has_history and decision_made:
        breakdown["correct_decision"] = 0.15
        score += 0.15
        
    if has_history and refund_computed:
        breakdown["correct_refund_computed"] = 0.25
        score += 0.25
        
    if plan_complete:
        breakdown["resolution_plan_complete"] = 0.25
        score += 0.25
        
    return score, breakdown

def apply_global_penalties(state, action, reward, breakdown):
    # -0.1 per redundant action (asking same question twice)
    # -0.2 for attempting resolution without required tool calls
    # -0.3 for hallucinating tool results (claiming tool was called when it wasn't)
    old_actions = [x["action"] for x in state["conversation_history"][:-1] if x["role"] == "agent"]
    
    if action.action_type in ["respond", "classify", "resolve", "escalate"]:
        for old in old_actions:
            if old.get("action_type") == action.action_type and old.get("content") == action.content:
                reward -= 0.1
                breakdown["redundancy_penalty"] = breakdown.get("redundancy_penalty", 0) - 0.1
                break
                
    if action.action_type == "resolve":
        task_id = state["task_id"]
        # Check if required tools were called
        tools_called = [act.get("tool_name") for act in old_actions if act.get("action_type") == "call_tool"]
        if task_id == "multi_turn_resolution" and "check_refund_status" not in tools_called:
            reward -= 0.2
            breakdown["no_tool_penalty"] = -0.2
        if task_id == "policy_conflict_escalation" and "check_complaint_history" not in tools_called:
            reward -= 0.2
            breakdown["no_tool_penalty"] = -0.2
            
    # Wait, hallucination checking uses the single 'action'
    content = action.content.lower()
    if ("tool" in content or "result" in content) and action.action_type not in ["call_tool", "classify"]:
        tools_called = [act.get("tool_name") for act in old_actions if act.get("action_type") == "call_tool"]
        if not tools_called:
            reward -= 0.3
            breakdown["hallucination_penalty"] = -0.3

    return reward, breakdown

def evaluate_action(task_id, state, action):
    # Returns (reward, breakdown, done)
    # reward is the ABSOLUTE score (not incremental), always in (0.01, 0.99)
    # This ensures /step always returns a value strictly between 0 and 1.
    if task_id == "classify_and_route":
        score, breakdown = grade_classify_and_route(state)
    elif task_id == "multi_turn_resolution":
        score, breakdown = grade_multi_turn_resolution(state)
    elif task_id == "policy_conflict_escalation":
        score, breakdown = grade_policy_conflict_escalation(state)
    else:
        score, breakdown = 0.01, {}
        
    score, breakdown = apply_global_penalties(state, action, score, breakdown)
    
    done = False
    if action.action_type == "resolve" or state["current_step"] >= state.get("max_steps", 5):
        done = True
        
    # Cap absolute score strictly within (0, 1) — never exactly 0.0 or 1.0
    score = max(0.01, min(0.99, score))
    
    # Store for state tracking (used by inference.py to get final score)
    state["absolute_score"] = score
    
    # Return absolute score (not incremental) so every /step response is in (0.01, 0.99)
    return score, breakdown, done
