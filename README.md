---
title: IndiaServiceEnv
emoji: 🇮🇳
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

IndiaServiceEnv grew out of a simple frustration — watching 
people around us struggle to get a Jio billing error corrected, 
chase an IRCTC refund for weeks, or dispute an electricity bill 
with no clear path forward. This environment was built by Anvi 
Singh, solo founder of InBridge (inbridge.in), together with 
teammates Abhishek Rai and Abhyuday Agnihotri, who brought the 
same energy to this project that goes into building civic tech 
that actually works for real people.

InBridge is a live assistant helping Indian citizens navigate 
government schemes like PM-KISAN and Ayushman Bharat. The users 
we built it for face the exact same friction when dealing with 
telecom and utility grievances — they just have no one to help 
them through it. IndiaServiceEnv brings that real-world 
complexity into a trainable RL setting for the first time, so 
agents can learn to do what a good customer service 
representative should: listen, look up the right information, 
apply the correct policy, and actually resolve the problem.

This is an **OpenEnv** wrapper environment engineered to 
simulate real-world Indian utility and telecom grievance 
resolution workflows. Customer service agents must address 
mock tickets (Jio, IRCTC, BESCOM), effectively utilize 
designated APIs to lookup or interpret records, and issue 
resolutions.

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
  Intercept simple Jio billing complaints and categorize them 
  accurately without requiring external system lookups.
- **`multi_turn_resolution`** (Difficulty: **Medium** | Max Steps: 6)
  Ask the user for their IRCTC details (PNR), execute the 
  `check_refund_status` tool, interpret timeline logic 
  iteratively, and formulate a responsive resolution sequence.
- **`policy_conflict_escalation`** (Difficulty: **Hard** | Max Steps: 8)
  Unpack deep conflicts. Call `check_complaint_history` to 
  locate preexisting complaints, identify intersecting rules 
  using `get_policy`, compute the customized 25% vs 50% refund 
  dynamics over time constraints, and issue escalation 
  explicitly.
  *Note: This task features a "hidden state" mechanic — the 
  agent must discover the existing complaint via an explicit 
  tool call as it is NOT shown in the initial observation 
  context, enforcing genuine exploratory behaviors.*

## Design Decisions
This environment features deliberately engineered mechanics 
to train realistic exploration bounding:
- **Hidden State / Exploration Requirement**: In the 
  `policy_conflict_escalation` task, pre-existing complaints 
  are *not* surfaced inside initial tickets. The agent must 
  systematically discover hidden parameters using exploration 
  functions (`check_complaint_history`), mirroring real CRM 
  architecture.
- **Incremental Partial Rewards**: Partial sub-scores exist 
  to guide agents sequentially (e.g., getting +0.3 for mapping 
  API context instead of receiving a binary win scalar), 
  increasing sub-step signal density natively!
- **Penalty Systems (Supervisor Escalation)**: Models lose 
  score limits (`-0.1` per step) if they loop context or ask 
  redundant questions, mimicking customer frustration. 
  Crucially, a `-0.3` deduction penalty natively controls 
  hallucinations, matching strict telecommunications compliance 
  guidelines organically!
- **Policy Conflict Parameters**: Intersecting rule policies 
  explicitly enforce LLM mathematical derivations (such as 
  comparing age variables mathematically against raw policy 
  data), rather than permitting generic boolean extraction 
  techniques!

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
ENV_URL=https://huggingface.co/spaces/dobie17/IndiaServiceEnv
```

## Live Model Baseline Scores

| Task                       | Difficulty | Score |
|----------------------------|------------|-------|
| classify_and_route         | Easy       | 0.50  |
| multi_turn_resolution      | Medium     | 0.80  |
| policy_conflict_escalation | Hard       | 0.75  |
