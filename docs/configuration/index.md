# Configuration Reference

Claw-R1 使用 [Hydra](https://hydra.cc/) 进行层次化配置管理。所有 YAML 配置位于 `claw_r1/config/`。

## 配置文件

| 文件 | 用途 |
|---|---|
| `agent_ppo_trainer.yaml` | 基础 PPO trainer 配置（继承 veRL 的 ppo_trainer） |
| `async_ppo_trainer.yaml` | 异步训练专用配置 |
| `overrides/rollout.yaml` | Rollout worker 设置（异步模式、Agent Flow） |

---

## `async_ppo_trainer.yaml`

异步训练的核心配置文件：

```yaml
defaults:
  - ppo_trainer
  - /overrides/rollout@actor_rollout_ref.rollout
  - _self_

# -- 异步训练设置 --
async_training:
  staleness_threshold: 0.1           # off-policy 容忍度
  trigger_parameter_sync_step: 4     # 每 N 步同步权重到 Rollouter
  require_batches: 1                 # 每次从 DataPool 取的 batch 数
  partial_rollout: false             # 同步时是否中断进行中的 rollout
  use_rollout_log_probs: true        # 使用 rollout 时收集的 log_probs
  max_queue_size: null               # DataPool 队列大小（null = 无限）

  checkpoint_engine:
    enable: true
    device_buffer_size_M: 4096

# -- Training GPU Pool --
trainer:
  nnodes: 1
  n_gpus_per_node: 4

# -- Rollout GPU Pool --
rollout:
  nnodes: 1
  n_gpus_per_node: 4
  total_epochs: 10
  test_freq: 1

# -- Actor 配置 --
actor_rollout_ref:
  hybrid_engine: false
  actor:
    use_rollout_log_probs: ${oc.select:async_training.use_rollout_log_probs, true}
  checkpoint_engine: ${oc.select:async_training.checkpoint_engine, null}
```

!!! warning "GPU 分配"
    `trainer` 和 `rollout` 都必须分配 GPU。总 GPU 数 = `trainer.nnodes × trainer.n_gpus_per_node + rollout.nnodes × rollout.n_gpus_per_node`。

---

## `overrides/rollout.yaml`

Rollout worker 的配置覆盖：

```yaml
name: vllm
mode: async

agent:
  default_agent_flow: single_step_single_turn_agent
  agent_flow_config_path: null
```

---

## Gateway 配置

Gateway 作为独立进程运行，通过 CLI 参数配置（非 Hydra）：

```bash
python -m claw_r1.gateway.gateway \
    --data-pool-name   data_pool \
    --vllm-addresses   host1:8001,host2:8001 \
    --tokenizer-path   /path/to/model \
    --prompt-length    4096 \
    --response-length  1024 \
    --reward-worker-name reward_loop_worker \
    --ray-address      auto \
    --ray-namespace    default \
    --host             0.0.0.0 \
    --port             8100
```

Gateway 启动超时可通过 Hydra 配置：

```yaml
trainer:
  gateway_startup_timeout: 300   # 秒，默认 300
```

---

## Agent Flow 配置

### 白盒 Agent Flow

在 `overrides/rollout.yaml` 中指定：

```yaml
agent:
  default_agent_flow: single_step_single_turn_agent
```

### 黑盒 Agent Flow

通过外部 YAML 文件注册：

```yaml
# claw_r1/blackbox_agent/agent_flow_config.yaml
- name: blackbox_gsm8k_agent
  _target_: claw_r1.blackbox_agent.gsm8k_agent_flow.BlackBoxGSM8KAgentFlow
```

在训练脚本中引用：

```bash
actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_gsm8k_agent \
actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml
```

---

## 多 GPU 配置

```yaml
# 独立的 GPU 池
trainer:
  nnodes: 1
  n_gpus_per_node: 2    # 2 GPU 用于训练（Actor + Critic）

rollout:
  nnodes: 1
  n_gpus_per_node: 1    # 1 GPU 用于推理（vLLM）
```

!!! note "资源池隔离"
    Claw-R1 使用 Ray 的资源组机制确保 Trainer 和 Rollouter 的 GPU 不重叠。使用 `async_ppo_trainer.yaml` 时自动配置。详见 [Async Training](../components/async-training.md)。

---

## 完整训练脚本示例

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

更多示例见 `example/` 目录。
