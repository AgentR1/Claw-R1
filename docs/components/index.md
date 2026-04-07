# Components

Claw-R1 的组件围绕**数据流**组织：从 Agent 交互的采集，到数据的管理与质量评估，再到向训练引擎的供给。各组件通过 HTTP 和 Ray RPC 通信。

<div class="grid cards" markdown>

-   **Gateway Server** · 数据采集入口

    ---

    FastAPI HTTP 服务。所有 Agent LLM 调用的统一入口，自动从交互中采集训练数据（Step）并提交到 DataPool。支持白盒显式提交和黑盒自动采集两种模式。

    [:octicons-arrow-right-24: Gateway Server](gateway.md)

-   **DataPool** · 数据管理核心

    ---

    Ray Actor。Claw-R1 的数据管理中枢 — 存储、索引、分区和供给交互数据。支持 Channel 隔离、GRPO 分组、容量背压控制和实时统计监控。

    [:octicons-arrow-right-24: DataPool](datapool.md)

-   **Reward System** · 数据质量评估

    ---

    `RewardLoopWorker` Ray Actor。多维度数据质量评估：rule-based、discriminative RM、generative RM，以及人类反馈信号的整合。

    [:octicons-arrow-right-24: Reward System](reward-system.md)

-   **Agent Flow** · 白盒数据采集

    ---

    Agent 执行生命周期管理。白盒 Agent 通过 Python API 显式提交 Step，完整控制数据采集过程。

    [:octicons-arrow-right-24: Agent Flow](agent-flow.md)

-   **Black-box Agent** · 黑盒数据采集

    ---

    零代码侵入的黑盒 Agent 接入。任何使用 OpenAI 兼容 API 的 Agent 通过 `base_url` 透明接入，Gateway 自动采集交互数据。

    [:octicons-arrow-right-24: Black-box Agent](blackbox-agent.md)

-   **Async Training** · 数据消费与训练

    ---

    `AsyncTrainer` 和 `AsyncRollouter` Ray Actor。持续从 DataPool 消费高质量数据进行训练，带参数同步。

    [:octicons-arrow-right-24: Async Training](async-training.md)

-   **Prefix Tree Merge** · 前缀去重优化

    ---

    将共享前缀的多条序列合并为一次前向计算，消除 multi-step agent 训练中的冗余 prefix 计算。目前在 [`prefix-tree-merge`](https://github.com/AgentR1/Claw-R1/tree/prefix-tree-merge) 分支测试中。

    [:octicons-arrow-right-24: Prefix Tree Merge](prefix-tree-merge.md)

</div>

## 数据流全景

```
                        数据采集层
                      ┌─────────────────────────────────────────┐
  黑盒 Agent ────────►│                                         │
  (base_url)          │         GATEWAY SERVER                  │
                      │         (FastAPI, 端口 8100)             │
  白盒 Agent ────────►│         自动采集交互 Step                 │
  (AgentFlow)         └────────────┬────────────────────────────┘
                                   │ Ray RPC (submit_steps)
                                   ▼
                        数据管理层
                      ┌─────────────────────────────────────────┐
                      │         DATAPOOL                         │
                      │         (Ray Actor)                      │
                      │                                          │
                      │  • 存储与索引    • Channel 分区            │
                      │  • GRPO 分组     • 容量背压控制            │
                      │  • 质量评估      • 实时统计监控            │
                      └──────────────────┬──────────────────────┘
                                         │ fetch_batch()
                                         ▼
                        数据消费层
                      ┌─────────────────────────────────────────┐
                      │         ASYNC TRAINER                    │
                      │         (Ray Actor, Training GPU Pool)   │
                      │   ┌─────────────────────────────────┐   │
                      │   │  Actor │ Critic │ RefPolicy      │   │
                      │   └─────────────────────────────────┘   │
                      └────────────────┬────────────────────────┘
                                       │ NCCL weight sync
                                       ▼
                      ┌─────────────────────────────────────────┐
                      │         ASYNC ROLLOUTER                  │
                      │         (Ray Actor, Rollout GPU Pool)    │
                      │         vLLM servers                     │
                      └─────────────────────────────────────────┘
```
