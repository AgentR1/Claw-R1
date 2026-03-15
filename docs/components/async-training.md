# Async Training

Claw-R1 的异步训练架构将 rollout（trajectory 生成）和 training（权重更新）分离为两个独立的 Ray Actor，运行在不同的 GPU 池上。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│  Rollout GPU Pool                                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  AsyncRollouter (Ray Actor)                      │   │
│  │  ├── DataLoader (遍历数据集)                      │   │
│  │  ├── vLLM replicas (推理引擎)                     │   │
│  │  ├── AgentFlowManager (管理 Agent 执行)           │   │
│  │  ├── Gateway (FastAPI 子进程, 端口 8100)          │   │
│  │  └── RewardLoopWorker (计算 reward)               │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │  submit_step (via Gateway → DataPool)
                       ▼
              ┌─────────────────┐
              │   DataPool       │   ← 共享 Ray Actor
              └────────┬────────┘
                       │  fetch_batch()
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Training GPU Pool                                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  AsyncTrainer (Ray Actor)                        │   │
│  │  ├── Actor worker group (策略模型)                │   │
│  │  ├── Critic worker group (价值模型)               │   │
│  │  └── RefPolicy worker group (KL baseline)        │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │  NCCL weight broadcast
                       ▼
              AsyncRollouter.update_weights()
```

## AsyncTrainer

`AsyncTrainer` 是运行在 Training GPU Pool 上的 Ray Actor，执行持续的 PPO 训练循环：

1. 从 DataPool `fetch_batch()` — 阻塞等待完整的 `prompt_uid` 组
2. 通过 `RewardLoopWorker` 计算 batch 的 reward
3. 计算 advantage（GAE 或 GRPO）
4. 执行 PPO Actor + Critic 更新
5. 每 `trigger_parameter_sync_step` 步触发权重同步

### Worker 初始化

AsyncTrainer 在 `init_workers()` 中创建 Actor、Critic、RefPolicy 的 worker group，并将它们部署到 Training GPU Pool：

```python
# 创建顺序：Critic → RefPolicy → Actor（最后创建 Actor 以免影响 vLLM 内存估算）
self.critic_wg.init_model()
self.ref_policy_wg.init_model()
self.actor_wg.init_model()
```

## AsyncRollouter

`AsyncRollouter` 运行在 Rollout GPU Pool 上，持有：

- **DataLoader**：遍历训练数据集
- **vLLM replicas**：高吞吐推理服务器
- **AgentFlowManager**：管理 `AgentFlowBase` worker
- **Gateway**：FastAPI HTTP 服务器（作为子进程启动）
- **RewardLoopWorker**：在 rollout 期间计算 reward

### Gateway 启动流程

Rollouter 将 Gateway 作为子进程启动：

1. 快速初始化（Ray 连接、DataPool、vLLM 地址）→ HTTP 立即可用
2. Tokenizer 在后台线程加载
3. Rollouter 轮询 `GET /ready` 等待 Gateway 完全就绪
4. 超时时间可通过 `trainer.gateway_startup_timeout` 配置（默认 300 秒）

### 暂停/恢复（权重同步）

权重同步期间，Rollouter 暂停生成：

```python
rollouter.pause()                          # 停止新生成，等待进行中的请求完成
# NCCL broadcast: Actor weights → vLLM
rollouter.update_param_version(new_version)
rollouter.resume()                         # 使用更新后的权重恢复生成
```

## ParameterSynchronizer

轻量级 Ray Actor，协调 AsyncTrainer 和 AsyncRollouter 之间的权重同步：

```python
class ParameterSynchronizer:
    def sync_weights(self, version, validate=False):
        # 1. 暂停 rollout
        # 2. NCCL broadcast: trainer Actor → vLLM
        # 3. 更新 rollouter 的 param_version
        # 4. 可选：运行验证
        # 5. 恢复 rollout
```

## Advantage 计算

### GAE (Generalized Advantage Estimation)

用于 trajectory 级别的 value baseline。在 **step 级别** 计算 advantage，然后广播到 **token 级别**（同一 step 内所有 response token 共享相同的 advantage）。

### GRPO (Group Relative Policy Optimization)

用于 prompt 级别的 baseline。将来自同一 `prompt_uid` 的多个 rollout 分组，在组内归一化 advantage。不需要单独的 Critic 模型，更节省内存。

## 资源池配置

Trainer 和 Rollouter 运行在独立的 GPU 池上，防止资源竞争：

```yaml
# async_ppo_trainer.yaml

# Training GPU Pool (Actor, Critic, RefPolicy)
trainer:
  nnodes: 1
  n_gpus_per_node: 2

# Rollout GPU Pool (vLLM)
rollout:
  nnodes: 1
  n_gpus_per_node: 1
```

总 GPU 数 = `trainer.nnodes × trainer.n_gpus_per_node + rollout.nnodes × rollout.n_gpus_per_node`。

!!! warning "GPU 分配"
    必须同时为 trainer 和 rollout 配置 GPU。如果 trainer 没有分配 GPU，训练参数（Actor、Critic）将无法部署到 GPU 上。

## 关键配置

```yaml
# async_ppo_trainer.yaml
async_training:
  staleness_threshold: 0.1           # off-policy 容忍度
  trigger_parameter_sync_step: 4     # 每 N 步同步权重
  require_batches: 1                 # 每次从 DataPool 取多少个 batch
  use_rollout_log_probs: true        # 使用 rollout 时的 log_probs
  max_queue_size: null               # DataPool 队列大小（null = 无限）
  partial_rollout: false             # 同步时是否中断 rollout

  checkpoint_engine:
    enable: true
    device_buffer_size_M: 4096
```

## 入口

```bash
python3 -m claw_r1.async_main \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    rollout.n_gpus_per_node=1 \
    rollout.nnodes=1 \
    async_training.trigger_parameter_sync_step=1 \
    ...
```

完整示例见 `example/test_async_blackbox.sh`。
