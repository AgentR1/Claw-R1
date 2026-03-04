# Configuration Reference

Claw-R1 uses [Hydra](https://hydra.cc/) for hierarchical configuration management. All YAML configs are located in `claw_r1/config/`.

## Config Files

| File | Purpose |
|---|---|
| `agent_ppo_trainer.yaml` | Base PPO trainer config (extends veRL's ppo_trainer) |
| `async_ppo_trainer.yaml` | Async-specific overrides for fully-asynchronous training |
| `overrides/rollout.yaml` | Rollout worker settings (async mode, agent flow) |

---

## `async_ppo_trainer.yaml`

```yaml
defaults:
  - agent_ppo_trainer

async_trainer:
  # Off-policy tolerance: steps where policy_version lag > threshold
  # are treated as off-policy and corrected with importance sampling
  staleness_threshold: 0.1

  # Sync model weights from Trainer to Rollouter every N training steps
  trigger_parameter_sync_step: 4

  # Whether to use log-probs collected during rollout for IS ratio computation
  use_rollout_log_probs: true

  # Maximum number of prompt_uid groups held in DataPool
  # null = unlimited (use with caution on memory-constrained machines)
  max_queue_size: null
```

---

## `overrides/rollout.yaml`

```yaml
rollout:
  mode: async    # "sync" or "async"

agent_flow:
  num_workers: 8
  default_agent_flow: single_step_single_turn_agent

# Custom async server configuration (optional)
async_server:
  host: "0.0.0.0"
  port: 8000
```

---

## Model Configuration (`BaseModelConfig`)

```python
@dataclass
class BaseModelConfig:
    path: str           # HuggingFace model path or local directory
    trust_remote_code: bool = False
```

Set via Hydra:

```bash
python -m claw_r1.async_main \
    trainer.model.path=/path/to/model \
    trainer.model.trust_remote_code=true
```

---

## Checkpoint Configuration (`CheckpointConfig`)

```python
@dataclass
class CheckpointConfig:
    save_freq: int = 100         # save every N training steps
    save_path: str = "./checkpoints"
    load_path: str | None = None # resume from checkpoint
```

---

## Reward Configuration

```yaml
reward:
  type: rule   # "rule", "disrm", or "genrm"

  # For rule-based rewards:
  # rule_fn: path.to.reward_function

  # For discriminative reward model:
  # model_path: /path/to/reward/model
  # batch_size: 32

  # For generative reward model:
  # reward_loop_manager: path.to.custom_reward.compute_reward
  # model_path: /path/to/eval/model
```

---

## Gateway Command-line Arguments

The Gateway is configured entirely via CLI arguments (not Hydra), since it runs as an independent process:

```bash
python -m claw_r1.gateway.gateway \
    --data-pool-name   data_pool \          # Ray actor name for DataPool
    --vllm-addresses   host1:8001,host2:8001 \  # comma-separated, load-balanced
    --tokenizer-path   /path/to/model \
    --prompt-length    4096 \
    --response-length  1024
```

---

## Multi-GPU Setup

```yaml
# Separate GPU pools for trainer and rollouter
trainer:
  tensor_model_parallel_size: 8    # 8 GPUs for training
  pipeline_model_parallel_size: 1

rollout:
  tensor_model_parallel_size: 4    # 4 GPUs for inference (vLLM)
  n_gpus_per_node: 4
```

!!! note "Resource Pool Separation"
    Claw-R1 uses Ray's resource group mechanism to ensure Trainer and Rollouter GPUs never overlap. This is configured automatically when using `async_ppo_trainer.yaml`. See [Async Training](../components/async-training.md) for details.
