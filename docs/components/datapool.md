# DataPool

DataPool 是一个 **Ray Actor**，作为 Agent 侧（Gateway）和 Training 侧（Trainer）之间的中央 trajectory 缓冲区。

## 在架构中的角色

```
Gateway ──► DataPool.submit_step()      (异步，fire-and-forget)
Trainer ◄── DataPool.fetch_batch()      (阻塞拉取，batch-ready 的组)
```

DataPool 完全解耦了写入速度（由 Agent 请求频率驱动）和读取速度（由训练吞吐量驱动）。双方互不等待。

## Channel 系统

DataPool 通过 **channel** 对数据进行分区。默认 channel 为 `"train"`，验证流程使用 `"val"` channel 以隔离数据。

```python
# 训练数据
data_pool.submit_step(step, channel="train")

# 验证数据
data_pool.submit_step(step, channel="val")
```

每个 channel 拥有独立的存储、索引和 FIFO 队列。

## 数据模型

DataPool 以 **step 粒度** 存储 trajectory。每个 step 是一个 `(s, a, r)` 元组：

```python
@dataclass
class Step:
    prompt_ids:     list[int]   # state: 完整上下文 token IDs
    response_ids:   list[int]   # action: LLM 生成的 token IDs
    reward:         float       # 该 step 的即时 reward
    trajectory_uid: str         # 同一对话中的 step 共享此 ID
    prompt_uid:     str         # 同一 prompt 的 rollout 共享此 ID（用于 GRPO）
    step_index:     int         # trajectory 内的位置（0-indexed）
    policy_version: int         # 生成该 step 时的策略版本
    is_last:        bool        # 是否为 trajectory 的最后一个 step
    metadata:       dict        # 辅助数据（数据集字段、来源信息等）
```

### 内部索引

| 索引 | 类型 | 用途 |
|---|---|---|
| `trajectory_index` | `dict[str, list[int]]` | `trajectory_uid` → step 索引列表 |
| `trajectory_complete` | `dict[str, bool]` | 追踪 trajectory 是否已收到 `is_last` step |
| `prompt_groups` | `dict[str, PromptGroup]` | `prompt_uid` → trajectory 列表和完成状态 |

## Producer API

### `submit_step(step: Step, channel="train")`

添加单个 step 到指定 channel。由 Gateway 通过 Ray RPC 调用。

### `submit_steps(steps: list[Step], channel="train")`

批量提交多个 step。比循环调用 `submit_step` 更高效。

### `complete_trajectory(trajectory_uid, reward=None, channel="train")`

标记一条 trajectory 完成。用于黑盒模式，Agent 通过 Gateway 的 `v1/complete_trajectory` 端点触发。

## Consumer API

### `fetch_batch(n_rollouts, channel="train") → list[Step] | None`

FIFO 拉取下一个就绪的 `prompt_uid` 组。一个组在所有 trajectory 都收到 `is_last` step 后变为"就绪"。

当没有完整组可用时返回 `None`。

```python
# Trainer 侧
while True:
    batch = await data_pool.fetch_batch.remote(n_rollouts=5)
    if batch is not None:
        train_on_batch(batch)
```

## 容量管理

当设置 `max_queue_size` 时，DataPool 在队列满时丢弃最旧的就绪组，防止 Trainer 较慢时内存无限增长：

```yaml
async_training:
  max_queue_size: null   # null = 无限
```

## Training Backend

DataPool 使用 `TrainingBackend` 将 `list[Step]` 转换为训练引擎的原生格式：

```python
class VerlBackend(TrainingBackend):
    """将 Step 列表转换为 verl DataProto。"""

    def convert(self, steps: list[Step]) -> DataProto:
        # prompt_ids: 左填充到 prompt_length
        # response_ids: 右填充到 response_length
        # input_ids: [prompt_ids | response_ids]
        # attention_mask, position_ids, response_mask 等
        ...
```

## Off-policy 支持

Trainer 可以通过 staleness threshold 配置来处理历史（off-policy）数据：

```yaml
async_training:
  staleness_threshold: 0.1   # policy_version 滞后 > threshold 的 step 为 off-policy
```

Off-policy step 仍包含在 batch 中，但在 loss 计算时通过 importance sampling 进行降权。
