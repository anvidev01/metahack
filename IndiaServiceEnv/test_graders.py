import json
from app.graders import evaluate_action
from app.models import Action

def make_state(task_id, history):
    return {
        "task_id": task_id,
        "customer_message": "test msg",
        "conversation_history": history,
        "available_tools": [],
        "current_step": len(history),
        "max_steps": 10,
        "absolute_score": 0.0
    }

def run_grader(task_id, sequence):
    state = make_state(task_id, [])
    # We will simulate feeding actions one by one
    total = 0.0
    for act in sequence:
        state["conversation_history"].append({"role": "agent", "action": act})
        # Evaluate incremental
        action_obj = Action(**act)
        inc, breakdown, done = evaluate_action(task_id, state, action_obj)
        total += inc
    return state["absolute_score"]

def test_classify_and_route_perfect():
    seq = [
        {"action_type": "classify", "content": "Checking bill and amount parameters.", "tool_name": None, "tool_params": None},
        {"action_type": "escalate", "content": "Routing to billing department.", "tool_name": None, "tool_params": None},
        {"action_type": "resolve", "content": "Resolution.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("classify_and_route", seq)
    assert abs(score - 0.99) < 0.001, f"Expected 0.99, got {score}"

def test_classify_and_route_partial():
    seq = [
        {"action_type": "classify", "content": "Check billing parameters.", "tool_name": None, "tool_params": None},
        {"action_type": "resolve", "content": "Resolved.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("classify_and_route", seq)
    assert 0.3 < score < 0.8, f"Expected ~0.5, got {score}"

def test_classify_and_route_zero():
    seq = [
        {"action_type": "respond", "content": "Hello.", "tool_name": None, "tool_params": None},
        {"action_type": "resolve", "content": "Finished.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("classify_and_route", seq)
    assert score < 0.2, f"Expected <0.2, got {score}"

def test_multi_turn_perfect():
    seq = [
        {"action_type": "respond", "content": "Please give your PNR", "tool_name": None, "tool_params": None},
        {"action_type": "call_tool", "content": "checking tool natively", "tool_name": "check_refund_status", "tool_params": {}},
        {"action_type": "respond", "content": "It will take few days so I will escalate it.", "tool_name": None, "tool_params": None},
        {"action_type": "resolve", "content": "Resolved natively.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("multi_turn_resolution", seq)
    assert abs(score - 0.99) < 0.001, f"Expected 0.99, got {score}"

def test_multi_turn_partial():
    seq = [
        {"action_type": "call_tool", "content": "checking tool natively", "tool_name": "check_refund_status", "tool_params": {}},
        {"action_type": "resolve", "content": "Done.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("multi_turn_resolution", seq)
    assert 0.3 < score < 0.8, f"Expected ~0.6, got {score}"

def test_multi_turn_zero():
    seq = [
        {"action_type": "resolve", "content": "No tools called at all.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("multi_turn_resolution", seq)
    assert score < 0.2, f"Expected <0.2, got {score}"

def test_policy_escalation_perfect():
    seq = [
        {"action_type": "call_tool", "content": "Checking history.", "tool_name": "check_complaint_history", "tool_params": {}},
        {"action_type": "resolve", "content": "We will escalate this comp-9901 ticket. The refund is 50% and timeline is 14 days.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("policy_conflict_escalation", seq)
    assert abs(score - 0.99) < 0.001, f"Expected 0.99, got {score}"

def test_policy_escalation_partial():
    seq = [
        {"action_type": "call_tool", "content": "Checking history.", "tool_name": "check_complaint_history", "tool_params": {}},
        {"action_type": "resolve", "content": "We will escalate this ticket, sorry.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("policy_conflict_escalation", seq)
    assert 0.3 < score < 0.8, f"Expected ~0.5, got {score}"

def test_policy_escalation_zero():
    seq = [
        {"action_type": "resolve", "content": "Nothing done.", "tool_name": None, "tool_params": None}
    ]
    score = run_grader("policy_conflict_escalation", seq)
    assert score < 0.2, f"Expected <0.2, got {score}"

if __name__ == "__main__":
    tests = [
        test_classify_and_route_perfect,
        test_classify_and_route_partial,
        test_classify_and_route_zero,
        test_multi_turn_perfect,
        test_multi_turn_partial,
        test_multi_turn_zero,
        test_policy_escalation_perfect,
        test_policy_escalation_partial,
        test_policy_escalation_zero
    ]
    passed = True
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as e:
            print(f"FAIL: {test.__name__} - {e}")
            passed = False
            
    if not passed:
        exit(1)
    else:
        print("ALL TESTS PASSED.")
