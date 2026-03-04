# Reward System

The `RewardLoopWorker` is a Ray Actor responsible for assigning reward scores to trajectory steps. It bridges the gap between raw agent interactions and trainable reward signals.

## Three Reward Sources

Claw-R1 supports three types of reward computation, which can be combined:

| Type | Description | Best For |
|---|---|---|
| **Rule-based** | Deterministic function of step output | Verifiable tasks (math, code execution) |
| **Discriminative RM** | Binary classifier reward model | Preference learning, safety evaluation |
| **Generative RM** | LLM-based evaluator via custom scoring function | Complex quality assessment, nuanced feedback |

## Reward in Production vs. Research Settings

In **research settings** (white-box offline mode), rewards are computed from known ground truth:

```
Trajectory:   [user msg] → [agent think] → [tool call] → [tool result] → [final reply]
Reward:            0.0            0.3            0.7            0.9            0.8
```

- Rule-based: is the final answer correct? does the code pass tests?
- Model-based: is each step logically sound? is the tool use appropriate?

In **production settings** (online mode), rewards come from real user signals:

| Signal | Type | Interpretation |
|---|---|---|
| User sends follow-up | Implicit positive | Agent answer was relevant but incomplete |
| User corrects the agent | Negative feedback | Factual or task error |
| User says "thanks" | Positive signal | Task completed satisfactorily |
| No follow-up after task | Neutral / estimated | Reward Model estimates step quality |

Claw-R1 uses a Reward Model to convert these **soft signals** into scalar process rewards, filling the gap between verifiable task rewards and open-ended conversational rewards.

## RewardLoopWorker API

### `compute_score_batch(steps: list[Step]) → list[float]`

Computes rewards for a batch of steps. This is the primary interface used by the Trainer.

```python
# In AsyncTrainer
rewards = await reward_worker.compute_score_batch.remote(batch_steps)
for step, reward in zip(batch_steps, rewards):
    step.reward = reward
```

### Custom Reward Function

Register a custom generative reward model by implementing the `reward_loop_manager` interface:

```python
# custom_reward.py
def compute_reward(step: dict, model, tokenizer) -> float:
    """
    Args:
        step: dict with keys 'messages', 'response', 'metadata'
        model: loaded reward model
        tokenizer: model tokenizer
    Returns:
        scalar reward in [0.0, 1.0]
    """
    prompt = build_evaluation_prompt(step)
    score = model.score(prompt)
    return score
```

Then register it in the configuration:

```yaml
reward:
  type: genrm
  reward_loop_manager: path.to.custom_reward.compute_reward
  model_path: /path/to/reward/model
```

## Reward in the Training Loop

Reward computation is **decoupled from the agent service**:

1. The Gateway does **not** compute rewards before submitting steps to DataPool
2. DataPool stores steps with `reward=0.0` initially
3. The Trainer calls `RewardLoopWorker.compute_score_batch()` before the PPO update
4. Updated rewards are used for advantage computation

This ensures that even slow generative reward models (which may call an external LLM) do not affect agent service latency.

!!! tip "Reward Design"
    For new tasks, start with simple rule-based rewards (e.g., exact match, code execution pass rate). Generative reward models are more expressive but introduce variance and computational cost. Use discriminative models as a middle ground.
