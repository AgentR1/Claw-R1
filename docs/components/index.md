# Components

Claw-R1 由六个独立可运行的组件组成，通过 HTTP 和 Ray RPC 通信。

<div class="grid cards" markdown>

-   **Gateway Server**

    ---

    FastAPI HTTP 服务。所有 Agent LLM 调用的网络层入口。管理 vLLM 负载均衡，自动收集训练数据并提交到 DataPool。

    [:octicons-arrow-right-24: Gateway Server](gateway.md)

-   **DataPool**

    ---

    Ray Actor。Agent 侧和 Training 侧之间的中央 trajectory 缓冲区。支持 Gateway 的异步写入和 Trainer 的批量读取。

    [:octicons-arrow-right-24: DataPool](datapool.md)

-   **Agent Flow**

    ---

    Agent 执行生命周期管理框架。支持白盒（Python）和黑盒（OpenAI API）两种模式。

    [:octicons-arrow-right-24: Agent Flow](agent-flow.md)

-   **Black-box Agent**

    ---

    黑盒 Agent 系统。任何使用 OpenAI 兼容 API 的 Agent 通过 `base_url` 透明接入训练循环。

    [:octicons-arrow-right-24: Black-box Agent](blackbox-agent.md)

-   **Async Training**

    ---

    `AsyncTrainer` 和 `AsyncRollouter` Ray Actor。持续、非阻塞的训练循环，带参数同步。

    [:octicons-arrow-right-24: Async Training](async-training.md)

-   **Reward System**

    ---

    `RewardLoopWorker` Ray Actor。从 rule-based、discriminative 或 generative reward model 计算 step 级别的 reward。

    [:octicons-arrow-right-24: Reward System](reward-system.md)

-   **Prefix Tree Merge**

    ---

    前缀树合并优化。将共享前缀的多条序列合并为一次前向计算，消除 multi-step agent 训练中的冗余 prefix 计算。

    [:octicons-arrow-right-24: Prefix Tree Merge](prefix-tree-merge.md)

</div>

## 组件交互图

```
                      ┌─────────────────────────────────────────┐
  黑盒 Agent ────────►│                                         │
  (base_url)          │         GATEWAY SERVER                  │
                      │         (FastAPI, 端口 8100)             │
  白盒 Agent ────────►│                                         │
  (AgentFlow)         └────────────┬────────────────────────────┘
                                   │ Ray RPC (submit_step)
                                   ▼
                      ┌─────────────────────────────────────────┐
                      │         DATAPOOL                         │
                      │         (Ray Actor)                      │
                      │         Channel: train / val              │
                      └──────────────────┬──────────────────────┘
                                         │ fetch_batch()
                                         ▼
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
