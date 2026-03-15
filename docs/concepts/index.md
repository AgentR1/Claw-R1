# Core Concepts

Claw-R1 的设计围绕三个核心概念展开，它们共同构成一个闭环飞轮。

<div class="grid cards" markdown>

-   **Base URL Integration**

    ---

    零代码侵入的黑盒 Agent 接入机制。任何使用 OpenAI 兼容 API 的 Agent 只需修改 `base_url` 即可透明接入训练系统。

    [:octicons-arrow-right-24: Base URL Integration](base-url-integration.md)

-   **Middleware Layer**

    ---

    Gateway + DataPool 中间件架构。完全解耦 Agent 侧和 Training 侧，支持异步数据收集和训练。

    [:octicons-arrow-right-24: Middleware Layer](middleware-layer.md)

-   **Production Scenario**

    ---

    "部署 = 训练" 范式。Agent 在服务用户的同时持续收集数据和改进，无需离线重训。

    [:octicons-arrow-right-24: Production Scenario](production-scenario.md)

</div>

## 闭环飞轮

```
                    base_url
                 ┌────────────┐
                 │ 黑盒 Agent  │
                 │ (任意框架)   │
                 └──────┬─────┘
                        │ OpenAI API
                        ▼
              ┌──────────────────┐
              │    Gateway       │ ← Middleware Layer
              │  (自动收集数据)   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    DataPool      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    Trainer       │ ← Production Scenario
              │  (持续训练)       │    (部署 = 训练)
              └────────┬─────────┘
                       │ 权重同步
                       ▼
              ┌──────────────────┐
              │    vLLM          │
              │  (更好的模型)     │
              └──────────────────┘
```

三个概念的协同：

1. **Base URL** 让任何 Agent 零成本接入
2. **Middleware** 异步收集和缓冲训练数据
3. **Production Scenario** 让模型在服务中持续进化
