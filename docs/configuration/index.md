# Configuration Reference

Claw-R1 uses Hydra for training configuration and regular CLI flags for standalone services such as Gateway and Dashboard.

## Training Config Files

| File | Purpose |
|---|---|
| `claw_r1/config/agent_ppo_trainer.yaml` | Base PPO trainer configuration inherited from `verl`. |
| `claw_r1/config/async_ppo_trainer.yaml` | Asynchronous training configuration. |
| `claw_r1/config/overrides/rollout.yaml` | Rollout worker mode and Agent Flow settings. |

## Async Training

```yaml
async_training:
  staleness_threshold: 0.1
  trigger_parameter_sync_step: 4
  require_batches: 1
  partial_rollout: false
  use_rollout_log_probs: true
  max_queue_size: null

trainer:
  nnodes: 1
  n_gpus_per_node: 4

rollout:
  nnodes: 1
  n_gpus_per_node: 4
```

Both `trainer` and `rollout` need GPU allocations. Total GPUs are:

```text
trainer.nnodes * trainer.n_gpus_per_node + rollout.nnodes * rollout.n_gpus_per_node
```

## Agent Flow

White-box flows can be selected directly:

```yaml
agent:
  default_agent_flow: single_step_single_turn_agent
```

Black-box flows are registered through a YAML file:

```yaml
- name: blackbox_gsm8k_agent
  _target_: claw_r1.blackbox_agent.gsm8k_agent_flow.BlackBoxGSM8KAgentFlow
```

Training script override:

```bash
actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_gsm8k_agent \
actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml
```

## Gateway

The Gateway is started as a process by the rollouter, but it can also be run directly:

```bash
python -m claw_r1.gateway.gateway \
  --data-pool-name data_pool \
  --vllm-addresses host1:8001,host2:8001 \
  --tokenizer-path /path/to/model \
  --prompt-length 4096 \
  --response-length 1024 \
  --reward-worker-name reward_loop_worker \
  --ray-address auto \
  --ray-namespace default \
  --host 0.0.0.0 \
  --port 8100
```

Gateway startup timeout is configured through Hydra:

```yaml
trainer:
  gateway_startup_timeout: 300
```

## Dashboard

The dashboard reads live DataPool and parameter synchronization state:

```bash
sh example/start_dashboard.sh \
  --ray-address auto \
  --ray-namespace claw_r1_async \
  --actor-name data_pool \
  --sync-actor-name parameter_synchronizer \
  --channel train,val \
  --port 8120
```

Equivalent YAML:

```yaml
ray_address: auto
ray_namespace: claw_r1_async
actor_name: data_pool
sync_actor_name: parameter_synchronizer
channel: train
refresh_interval_ms: 2000
host: 0.0.0.0
port: 8120
```

The dashboard intentionally has no mock mode. It fails visibly when the Ray actors are unavailable.

## Example Training Command

```bash
python3 -m claw_r1.async_main \
  algorithm.adv_estimator=grpo \
  data.train_files=$TRAIN_FILE \
  data.val_files=$VAL_FILE \
  data.train_batch_size=128 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_gsm8k_agent \
  actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  rollout.n_gpus_per_node=1 \
  rollout.nnodes=1 \
  async_training.trigger_parameter_sync_step=1 \
  async_training.use_rollout_log_probs=true
```
