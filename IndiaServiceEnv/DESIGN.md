# Design Decisions — IndiaServiceEnv

## Why India-Specific Grievance Resolution?

India has 1B+ telecom subscribers and 300M+ 
electricity consumers. Grievance resolution is a 
daily friction point with no AI training benchmark 
existing for this domain. IndiaServiceEnv fills 
this gap directly.

Built by the team behind InBridge (inbridge.in) — 
a live civic assistant already serving Indian 
citizens navigating government schemes. Armaan Soni,
Abhishek Rai, and Abhyuday Agnihotri.

## Reward Design Philosophy

Most RL environments use sparse rewards — binary 
win/lose at episode end. IndiaServiceEnv uses dense 
partial rewards at every dialogue turn:

| Sub-step | Reward |
|---|---|
| Correct issue classification | +0.2 to +0.5 |
| Correct tool called | +0.2 to +0.3 |
| Tool result interpreted correctly | +0.2 |
| Correct resolution path | +0.3 |
| Structured resolution plan | +0.25 |

Penalties mirror real supervisor escalation triggers:
- Redundant question: -0.1
- Resolve without required tools: -0.2
- Hallucinated tool result: -0.3

## Hidden State Mechanic

The hard task (policy_conflict_escalation) does NOT 
surface the existing complaint in the initial 
observation. The agent must discover it by calling 
check_complaint_history() — mirroring real CRM 
architecture where agents must actively query 
systems rather than having all context handed to them.

This creates genuine exploration incentive rarely 
seen in existing OpenEnv environments.

## Difficulty Progression

| Task | Mechanic | Why Hard |
|---|---|---|
| classify_and_route | Single-step classification | Baseline capability test |
| multi_turn_resolution | Multi-turn + 1 tool | Requires dialogue management |
| policy_conflict_escalation | Hidden state + 2 tools + math | Requires planning and computation |

## Why Useful for the RL Community

1. Tests tool-use in realistic settings
2. Tests multi-turn dialogue management
3. Tests policy reasoning under conflict
4. India-specific data resists memorization
5. Deterministic graders ensure reproducible evals
