# Production Agent Scenario

## The Hidden Assumption in Agentic RL

Almost every Agentic RL framework is built on an implicit assumption:

> **Training phase ≠ Deployment phase**

The standard workflow: train on offline/simulated data → deploy a fixed model → retrain periodically.

This works in research settings, but in production it hits fundamental walls:

| Problem | Manifestation |
|---|---|
| **Distribution shift** | Training data is synthetic; real user requests have a different distribution → capability degradation after deployment |
| **Cold start** | A newly deployed model knows nothing about a specific user's habits, tools, or workflows → long "warmup" period |
| **Long-tail tasks** | Benchmarks cover common tasks; real users' niche needs cannot be covered by offline training |
| **Environment drift** | Tool APIs update, user behavior changes → static models cannot self-adapt |

## Claw-R1's Core Scenario: Personal Agent Self-Evolution

Claw-R1's first validation scenario is the **OpenClaw personal assistant**:

```
Setup:
  User deploys OpenClaw on a Mac Mini, connected to Slack / WeChat / email.
  Every day they interact with OpenClaw via messages:
  scheduling, information retrieval, code assistance, etc.

Traditional approach:
  OpenClaw uses a fixed GPT-4o / Claude 3.5.
  Capabilities do not grow with usage.

Claw-R1 approach:
  1. User message → OpenClaw → Gateway (intercepts LLM call)
  2. Gateway logs each interaction step → DataPool (local)
  3. Reward Model scores each interaction (user satisfaction signals, task completion, etc.)
  4. Training Engine on a remote server continuously consumes DataPool, updates model weights
  5. Updated weights are pushed back to the Gateway; the next call uses the improved model

Result:
  The OpenClaw running on the user's Mac Mini becomes progressively better
  at understanding this specific user over time.
```

## Three Requirements Traditional RL Frameworks Cannot Meet

This scenario requires three capabilities that traditional frameworks lack:

### ① Service continuity

Model weight updates must not interrupt Gateway request handling. In Claw-R1:

- The Trainer directly manages the lifecycle of Rollout Engine and Reward Model (`wake_up` / `sleep` / weight sync)
- The Gateway is a **pure HTTP proxy** — it only forwards requests and submits steps; it does not manage any engine lifecycle
- This guarantees continuous request forwarding and data collection even during weight updates

### ② No preset data

Traditional frameworks require a pre-collected dataset (SFT corpus or RL environment). Claw-R1's training data comes **entirely from live user interactions**:

- What questions the user asked, how the agent answered, which tools were called — all of this becomes training data automatically
- Zero data engineering; data accumulates naturally as the service runs

### ③ Reward signals from the real environment

Traditional RLVR rewards come from verifiable task outcomes (does the code run? is the math answer correct?). Production rewards are more nuanced:

- User follows up → implicit positive signal
- User corrects the agent → negative feedback
- Task completed with no follow-up → Reward Model estimates quality of intermediate steps

Claw-R1 uses a **Reward Model** to convert these **soft rewards** into trainable process rewards, bridging the gap between "verifiable tasks" and "real conversational tasks".

## Three Operating Modes

| Mode | Agent type | Data source | Notes |
|---|---|---|---|
| **White-box offline** | AgentFlow (Python) | Synthetic dataset or pre-collected trajectories | Fully implemented; recommended for research |
| **Black-box offline** | Any HTTP agent | Pre-collected logs | Gateway endpoint reserved |
| **Black-box online** | Any HTTP agent | Live user interactions | Gateway endpoint reserved; target production mode |

!!! info "Current Status"
    White-box offline mode is fully implemented. The black-box online endpoints (`/complete_trajectory`, `/{traj_uid}/{prompt_uid}/v1/chat/completions`) are designed and stubbed, actively being developed.

## Deployment = Training

Claw-R1 introduces a new paradigm:

```
┌─────────────────────────────────────────────────────┐
│         Traditional: Train → Deploy (fixed)          │
│                                                      │
│  [Synthetic data] → [Train] → [Fixed model] → Users  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│         Claw-R1: Deploy = Train (continuous)         │
│                                                      │
│  Users ──► Agent ──► [Live data] ──► Train ──► Agent │
│            ▲___________________________________|      │
└─────────────────────────────────────────────────────┘
```

In this paradigm:

- Every user interaction is a training sample
- Every model update improves the agent's real-world performance
- The longer the agent runs, the better it becomes for its specific users and environment
