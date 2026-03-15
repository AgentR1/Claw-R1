# Middleware Layer

## 同步训练的问题

传统 RL 训练是同步的：生成一个 batch → 训练一步 → 生成下一个 batch。这导致：

- GPU 利用率低（训练时推理空闲，推理时训练空闲）
- 无法支持实时服务（训练期间无法响应用户请求）
- 扩展困难（推理和训练的资源需求不同）

## Gateway + DataPool 架构

Claw-R1 通过 **Middleware Layer**（Gateway + DataPool）将 Agent 侧和 Training 侧完全解耦：

```
Agent 侧                    Middleware                    Training 侧
┌──────────┐           ┌──────────────────┐           ┌──────────────┐
│ Agent    │──HTTP──►  │  Gateway         │──Ray RPC──►│  DataPool    │
│ (任意)   │◄──HTTP──  │  (FastAPI, 8100) │           │  (Ray Actor) │
└──────────┘           └──────────────────┘           └──────┬───────┘
                                                             │ fetch_batch()
                                                             ▼
                                                      ┌──────────────┐
                                                      │  Trainer     │
                                                      │  (Ray Actor) │
                                                      └──────────────┘
```

## Gateway 的设计

Gateway 是一个**独立进程**（FastAPI），具有以下特性：

- **纯代理**：不管理任何引擎生命周期，只转发请求和收集数据
- **OpenAI 兼容**：黑盒 Agent 通过 `base_url` 透明接入
- **延迟初始化**：HTTP 服务立即可用，tokenizer 在后台加载

Gateway 支持两种工作模式：

| 模式 | 端点 | 数据收集方式 |
|---|---|---|
| 白盒 | `/generate`, `/submit_steps` | Agent 自行构建 Step 并提交 |
| 黑盒 | `{base_url}/v1/chat/completions` | Gateway 自动 tokenize 并构建 Step |

详见 [Gateway Server](../components/gateway.md)。

## DataPool 的设计

DataPool 是一个 **Ray Actor**，作为 trajectory 缓冲区：

| 特性 | 说明 |
|---|---|
| **异步写入** | Gateway 通过 fire-and-forget 的 Ray RPC 提交 Step |
| **阻塞读取** | Trainer 调用 `fetch_batch()` 等待完整的 prompt 组 |
| **Channel 分区** | `"train"` 和 `"val"` 数据隔离 |
| **FIFO 队列** | 按 `prompt_uid` 分组，完整组按先进先出顺序消费 |
| **容量管理** | 可配置 `max_queue_size` 防止内存溢出 |

详见 [DataPool](../components/datapool.md)。

## Step 数据模型

```python
@dataclass
class Step:
    prompt_ids:     list[int]   # state: 完整上下文 token IDs
    response_ids:   list[int]   # action: LLM 生成的 token IDs
    reward:         float       # 即时 reward
    trajectory_uid: str         # 同一对话的 step 共享此 ID
    prompt_uid:     str         # 同一 prompt 的 rollout 共享此 ID
    step_index:     int         # trajectory 内的位置
    policy_version: int         # 生成时的策略版本
    is_last:        bool        # 是否为最后一个 step
    metadata:       dict        # 辅助数据
```

## Reward 标注

Reward 计算与 Agent 服务**解耦**：

1. Gateway 提交 Step 时 `reward=0.0`
2. DataPool 存储原始 Step
3. Trainer 在 PPO 更新前通过 `RewardLoopWorker` 计算 reward
4. 更新后的 reward 用于 advantage 计算

这确保了即使是慢速的 generative reward model 也不会影响 Agent 服务延迟。
