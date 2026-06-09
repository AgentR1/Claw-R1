# 架构与组件

Claw-R1 的组件围绕一条数据流组织：Agent 产生交互，Gateway 采集并规范化 step，DataPool 存储与筛选数据，TrainingBackend 将可训练数据供给 RL 训练后端。

```text
Black-box Agent         White-box Agent
      |                       |
      v                       v
   Gateway Server  <---- explicit Step APIs
      |
      v
   DataPool  <---- Dashboard 读取统计、step、事件、筛选状态和 prefix-tree 预览
      |
      v
   Async Trainer ---- Parameter Synchronizer ---- Async Rollouter / vLLM
```

## Gateway Server

Gateway 是面向 Agent 的 HTTP 入口，支持两种接入方式：

- 白盒 Agent：显式构造 `Step`，通过 API 提交到 Gateway。
- 黑盒 Agent：使用 OpenAI 兼容接口，只修改 `base_url` 即可通过 Gateway 转发请求并采集数据。

Gateway 会把交互整理成统一的 step-level 记录，再提交给 DataPool。

## DataPool

DataPool 是 Ray Actor，也是 Claw-R1 的数据管理核心。它负责：

- 保存 prompt IDs、response IDs、reward、metadata 等 step-level 字段。
- 维护 trajectory ID、step index、prompt group、policy version 和 channel。
- 记录 reward、curation label、trainability、tag 和 note 等筛选信息。
- 按训练后端需要返回 batch，并跟踪数据消费状态。

## Agent Flow

Agent Flow 管理 Agent 执行生命周期。白盒 flow 适合直接控制推理与工具调用逻辑的场景；黑盒 flow 适合已有 Agent 系统，只需通过 Gateway 捕获 OpenAI 兼容请求。

## Reward System

Reward System 用于把不同来源的质量信号转为训练可用的 reward 或筛选依据：

- rule-based reward
- discriminative reward model
- generative judge
- explicit 或 implicit human feedback
- policy version freshness

## Async Training

异步训练将 rollout 和 training 拆成不同 Ray actor，分别使用不同 GPU 池。DataPool 作为中间层缓冲并供给训练数据，Parameter Synchronizer 负责将训练后的权重同步给 rollout 侧。

## Prefix Tree Merge

Prefix Tree Merge 面向 multi-step agent 训练中的共享前缀问题。它通过 prefix-tree packing 减少重复 prefix 计算，并可在 Dashboard 中预览真实 DataPool step 构成的 prefix tree。

## 生命周期映射

| 阶段 | 主要组件 | Dashboard 视图 |
|---|---|---|
| 采集交互 | Gateway, Agent Flow | Collection |
| 存储表示 | DataPool | Representation |
| 评估质量 | Reward System | Curation Signals |
| 筛选样本 | DataPool curation APIs | Curation |
| 优化共享上下文 | Prefix Tree Merge preview | Optimization |
| 供给训练 | DataPool, Async Trainer | Consumption |
