# Core Concepts

Claw-R1 的设计围绕三个核心概念展开：**通用数据采集**、**数据中间件管理**和**数据驱动的持续进化**。它们共同构成一个从采集到训练的数据飞轮。

<div class="grid cards" markdown>

-   **Base URL Integration** · 通用数据采集

    ---

    零代码侵入的 Agent 数据采集机制。任何使用 OpenAI 兼容 API 的 Agent 只需修改 `base_url`，Gateway 即可自动采集其交互数据。

    [:octicons-arrow-right-24: Base URL Integration](base-url-integration.md)

-   **Middleware Layer** · 数据中间件

    ---

    Gateway + DataPool 数据基础设施。统一解决数据的采集入口、质量管理、分区缓冲和按需供给。

    [:octicons-arrow-right-24: Middleware Layer](middleware-layer.md)

-   **Production Scenario** · 数据驱动进化

    ---

    "部署 = 训练" 范式。Agent 在服务用户的同时持续采集交互数据，用户行为天然成为数据质量信号，驱动模型持续进化。

    [:octicons-arrow-right-24: Production Scenario](production-scenario.md)

</div>

## 数据飞轮

```
                    base_url
                 ┌────────────┐
                 │ 任意 Agent  │
                 │ (白盒/黑盒) │
                 └──────┬─────┘
                        │ OpenAI API
                        ▼
              ┌──────────────────┐
              │    Gateway       │ ← 数据采集入口
              │  (自动采集 Step)  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    DataPool      │ ← 数据管理核心
              │  (评估·筛选·供给) │    (质量评估 + 分区管理)
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    Trainer       │ ← 数据消费
              │  (持续训练)       │
              └────────┬─────────┘
                       │ 权重同步
                       ▼
              ┌──────────────────┐
              │    vLLM          │
              │  (更好的模型)     │
              └──────────────────┘
```

三个概念的协同：

1. **Base URL** 让任何 Agent 的交互数据零成本被采集
2. **Middleware** 管理数据的质量、分区和供给
3. **Production Scenario** 让人类反馈信号自然融入数据，驱动模型持续进化
