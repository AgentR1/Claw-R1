# Middleware Layer

## 为什么需要数据中间件？

Agentic RL 中，Agent 产生交互数据，Trainer 消费数据进行训练。然而在实际场景中，两者之间存在显著的不对称：

- **数据来源多样**：白盒 Agent、黑盒 Agent、在线服务 Agent，产出的数据格式和频率各不相同
- **数据质量参差**：并非所有交互都有训练价值，需要评估和筛选
- **产消速率不匹配**：Agent 侧的数据产生速率与 Trainer 侧的消费速率往往不同步
- **数据需要管理**：分区、索引、背压控制、统计监控 — 这些不是简单的队列能解决的

Claw-R1 通过 **Middleware Layer**（Gateway + DataPool）在 Agent 侧和 Training 侧之间建立一层**数据基础设施**，统一解决数据的采集、管理和供给问题。

## Gateway + DataPool 架构

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

## Gateway：数据采集入口

Gateway 是一个**独立进程**（FastAPI），负责从 Agent 交互中采集训练数据：

- **纯代理**：不管理任何引擎生命周期，只转发请求和采集数据
- **OpenAI 兼容**：黑盒 Agent 通过 `base_url` 透明接入，Gateway 自动从对话中构建 Step
- **延迟初始化**：HTTP 服务立即可用，tokenizer 在后台加载

Gateway 支持两种数据采集模式：

| 模式 | 端点 | 数据采集方式 |
|---|---|---|
| 白盒 | `/generate`, `/submit_steps` | Agent 自行构建 Step 并提交 |
| 黑盒 | `{base_url}/v1/chat/completions` | Gateway 自动 tokenize 并构建 Step |

详见 [Gateway Server](../components/gateway.md)。

## DataPool：数据管理核心

DataPool 是一个 **Ray Actor**，不仅是 trajectory 缓冲区，更是 Claw-R1 的数据管理中枢：

| 能力 | 说明 |
|---|---|
| **数据存储** | 以 Step 粒度存储交互数据，支持多维索引 |
| **质量追踪** | 每个 Step 记录 `policy_version`，支持新鲜度检测 |
| **Channel 分区** | `"train"` 和 `"val"` 数据隔离，互不干扰 |
| **GRPO 分组** | 按 `prompt_uid` 分组，凑齐所有 rollout 后才供给训练 |
| **容量管理** | 可配置 `max_queue_size`，超限自动丢弃最旧数据 |
| **统计监控** | 实时提供队列深度、produce/consume/drop 速率等指标 |

详见 [DataPool](../components/datapool.md)。

## Step 数据模型

Step 是数据管理的原子单位，记录了一次 Agent 交互的完整信息：

```python
@dataclass
class Step:
    prompt_ids:     list[int]   # state: 完整上下文 token IDs
    response_ids:   list[int]   # action: LLM 生成的 token IDs
    reward:         float       # 即时 reward（质量评分）
    trajectory_uid: str         # 同一对话的 step 共享此 ID
    prompt_uid:     str         # 同一 prompt 的 rollout 共享此 ID
    step_index:     int         # trajectory 内的位置
    policy_version: int         # 生成时的策略版本（新鲜度追踪）
    is_last:        bool        # 是否为最后一个 step
    metadata:       dict        # 辅助数据（来源、数据集字段等）
```

## Reward 标注与数据质量评估

Reward 计算与 Agent 服务**解耦**，确保数据质量评估不影响 Agent 服务延迟：

1. Gateway 采集 Step 时 `reward=0.0`（原始数据）
2. DataPool 存储原始 Step
3. Trainer 在消费数据前通过 `RewardLoopWorker` 评估数据质量（计算 reward）
4. 评估后的 reward 用于 advantage 计算和数据筛选

这种设计使得即使是慢速的 generative reward model 或人类反馈管线也不会影响 Agent 的正常服务。
