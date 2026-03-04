# Async Training

Claw-R1's async training architecture separates rollout (trajectory generation) and training (weight updates) into two independent Ray Actors that run on different GPU pools simultaneously.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Rollout GPU Pool                                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  AsyncRollouter (Ray Actor)                      │   │
│  │  ├── AgentFlowManager (generates trajectories)   │   │
│  │  ├── vLLM servers (inference)                    │   │
│  │  └── RewardLoopWorker                            │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │  submit_step (via Gateway → DataPool)
                       ▼
              ┌─────────────────┐
              │   DataPool       │   ← shared Ray Actor
              └────────┬────────┘
                       │  fetch_batch()
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Training GPU Pool                                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  AsyncTrainer (Ray Actor)                        │   │
│  │  ├── Actor worker group (policy model)           │   │
│  │  ├── Critic worker group (value model)           │   │
│  │  └── RefPolicy worker group (KL baseline)        │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │  NCCL weight broadcast
                       ▼
              AsyncRollouter.update_weights()
```

## AsyncTrainer

`AsyncTrainer` is a Ray Actor that runs a continuous PPO training loop:

1. `fetch_batch()` from DataPool — blocking wait for a complete `prompt_uid` group
2. Compute rewards for the batch via `RewardLoopWorker`
3. Compute advantages (GAE or GRPO)
4. Run PPO Actor + Critic update steps
5. Every `trigger_parameter_sync_step` steps, trigger weight synchronization

### Key Configuration

```yaml
# async_ppo_trainer.yaml
async_trainer:
  staleness_threshold: 0.1         # off-policy tolerance (policy version lag)
  trigger_parameter_sync_step: 4   # sync weights every N training steps
  use_rollout_log_probs: true       # use log probs from rollout for IS ratio
  max_queue_size: null              # DataPool queue size (null = unlimited)
```

## AsyncRollouter

`AsyncRollouter` is a Ray Actor that runs on the rollout GPU pool and owns:

- **DataLoader**: Iterates over the dataset (for white-box offline mode)
- **vLLM replicas**: High-throughput inference servers
- **AgentFlowManager**: Manages `AgentFlowBase` workers
- **Gateway**: The FastAPI HTTP server (started as a subprocess)
- **RewardLoopWorker**: Computes rewards during rollout

The rollouter continuously generates trajectories and submits them to DataPool via the Gateway.

### Pause / Resume for Weight Sync

During weight synchronization, the rollouter temporarily pauses generation:

```python
# ParameterSynchronizer flow
rollouter.pause()           # stop new generation, drain in-flight requests
# NCCL broadcast: Actor weights → vLLM
rollouter.update_param_version(new_version)
rollouter.resume()          # restart generation with updated weights
```

This ensures that the policy version recorded in each `Step` accurately reflects which model generated it, enabling correct staleness detection and importance-sampling correction in the Trainer.

## ParameterSynchronizer

A lightweight Ray Actor that coordinates the weight sync handshake between `AsyncTrainer` and `AsyncRollouter`:

```python
class ParameterSynchronizer:
    def sync(self, trainer_actor, rollouter_actor, policy_version: int):
        # 1. Pause rollout
        rollouter_actor.pause.remote()
        # 2. NCCL broadcast weights from trainer Actor to vLLM
        trainer_actor.broadcast_weights.remote()
        # 3. Update version in rollouter
        rollouter_actor.update_param_version.remote(policy_version)
        # 4. Resume rollout
        rollouter_actor.resume.remote()
```

## Advantage Computation

Claw-R1 supports two advantage estimation algorithms:

### GAE (Generalized Advantage Estimation)

For trajectory-level value baselines. Computes advantages at the **step level** (one advantage per step in the trajectory), then broadcasts to the **token level** (all response tokens in a step share the same advantage).

```python
# core_algos.py
advantages = compute_gae_advantage_return(
    token_level_rewards,
    token_level_values,
    response_mask,
    gamma=0.99,
    lam=0.95,
)
```

### GRPO (Group Relative Policy Optimization)

For prompt-level baselines. Groups multiple rollouts from the same `prompt_uid` and normalizes advantages within the group:

```python
advantages = compute_grpo_outcome_advantage(
    token_level_rewards,
    response_mask,
    index=prompt_uid_tensor,
)
```

GRPO does not require a separate Critic model, making it more memory-efficient and suitable for larger models.

## Resource Pools

Trainer and Rollouter run on separate GPU pools to prevent resource contention:

```yaml
# async_ppo_trainer.yaml
rollout:
  tensor_model_parallel_size: 4   # rollout GPU pool: 4 GPUs per node

trainer:
  tensor_model_parallel_size: 8   # training GPU pool: 8 GPUs per node
```
