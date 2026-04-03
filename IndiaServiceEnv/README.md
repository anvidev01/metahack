---
title: IndiaServiceEnv
emoji: ☎️
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-service
---

# IndiaServiceEnv

This is an **OpenEnv** wrapper environment engineered to simulate real-world Indian utility and telecom grievance resolution workflows.

## Environment Space

### Observation Space
At each step, agents are provided with:
- **`ticket_id`**: Associated complaint ticket reference.
- **`customer_message`**: Raw message from the simulated customer.
- **`conversation_history`**: Persistent multi-turn dialog chain (system and agent roles).
- **`available_tools`**: Mapped mock endpoints the agent can execute.
- **`current_step` / `max_steps`**: Environment boundaries.

### Action Space
Agents must output strict JSON resolving to:
- **`action_type`**: `classify`, `call_tool`, `respond`, `escalate`, or `resolve`.
- **`content`**: Natural language thought or response.
- **`tool_name` / `tool_params`**: Mock function triggers (if calling a tool).

## Task Descriptions
1. **`classify_and_route`**: Intercept simple Jio billing complaints and categorize them accurately without requiring external system lookups.
2. **`multi_turn_resolution`**: Ask the user for their IRCTC details (PNR), execute the `check_refund_status` tool, interpret timeline logic iteratively, and formulate a responsive resolution sequence.
3. **`policy_conflict_escalation`**: Unpack deep conflicts. Call `check_complaint_history` to locate preexisting complaints, identify intersecting rules using `get_policy`, compute the 25% vs 50% refund dynamic, and finalize the ticket exactly.

## Baseline Result Scores

During localized benchmarking, `meta-llama/Llama-3.3-70B-Instruct` scored sequentially across the 3 environment configurations:

| Task ID | Metric (Reward) |
| --- | --- |
| `classify_and_route` | 0.50 |
| `multi_turn_resolution` | 0.80 |
| `policy_conflict_escalation` | 0.75 |

Raw `SUMMARY` Output:
```json
{"type": "SUMMARY", "scores": {"classify_and_route": 0.5, "multi_turn_resolution": 0.8, "policy_conflict_escalation": 0.75}}
```
