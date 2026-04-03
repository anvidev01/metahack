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

This is an **OpenEnv** wrapper environment engineered to simulate real-world Indian utility and telecom grievance resolution workflows. Customer service agents must address mock tickets (Jio, IRCTC, BESCOM), effectively utilize designated APIs to lookup or interpret records, and issue resolutions.

## Environment Space

### Action Space
Agents must output strict JSON resolving to exactly 5 potential `action_type` keys:
1. `classify`: Determine generic problem parameters.
2. `call_tool`: Execute dynamic external functions.
3. `respond`: Give standard replies asking for customer parameters (PNR, User IDs).
4. `escalate`: Defer the context to a human.
5. `resolve`: Finalize the ticket output safely.

Additional fields in the Action model must include:
- `content`: Natural language thought or response.
- `tool_name`: Mock function triggers (if calling a tool).
- `tool_params`: The parameters passed to the function if triggered.

### Observation Space
At each step, agents receive the following 7 explicit state mapping components:
1. `ticket_id`: Associated complaint ticket reference.
2. `customer_message`: Explicit string from the simulated customer.
3. `conversation_history`: Log progression of internal tool chains and user dialogs.
4. `available_tools`: Contextually available APIs for the agent to inject into tool calls.
5. `current_step`: Tracks sequential boundary index natively counting upward.
6. `max_steps`: Task cutoff constraints forcing a resolution or termination penalty.
7. `task_id`: Identifier mapping.

## Configured Tasks
- **`classify_and_route`** (Difficulty: **Easy** | Max Steps: 3)
  Intercept simple Jio billing complaints and categorize them accurately without requiring external system lookups.
- **`multi_turn_resolution`** (Difficulty: **Medium** | Max Steps: 6)
  Ask the user for their IRCTC details (PNR), execute the `check_refund_status` tool, interpret timeline logic iteratively, and formulate a responsive resolution sequence.
- **`policy_conflict_escalation`** (Difficulty: **Hard** | Max Steps: 8)
  Unpack deep conflicts. Call `check_complaint_history` to locate preexisting complaints, identify intersecting rules using `get_policy`, compute the customized 25% vs 50% refund dynamics over time constraints, and issue escalation explicitly.

## Setup & Running

**Docker:**
```bash
docker build -t indiaserviceenv .
docker run -p 7860:7860 indiaserviceenv
# Access locally via http://localhost:7860
```

**Local Dev:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

**Groq Inference Configuration (.env)**
```text
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
HF_TOKEN=<groq_api_key_starting_with_gsk_>
ENV_URL=http://localhost:7860  # Or point directly to Hugging Face URL
```

## Live Model Baseline Scores

| Task                       | Difficulty | Score |
|----------------------------|------------|-------|
| classify_and_route         | Easy       | 0.50  |
| multi_turn_resolution      | Medium     | 0.80  |
| policy_conflict_escalation | Hard       | 0.75  |
