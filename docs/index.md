# Claw-R1

**Empowering OpenClaw with Advanced Agentic RL**

Claw-R1 是一个基于中间件的 Agentic RL 训练框架，让任何 Agent 都能通过强化学习持续进化。

---

<div class="grid cards" markdown>

-   :material-link-variant: **Zero-Code Integration**

    ---

    通过 `base_url` 机制，任何使用 OpenAI 兼容 API 的 Agent 只需修改一个参数即可接入训练系统。无需修改 Agent 代码。

    [:octicons-arrow-right-24: Base URL Integration](concepts/base-url-integration.md)

-   :material-layers-outline: **Middleware Layer**

    ---

    Gateway + DataPool 中间件完全解耦 Agent 侧和 Training 侧。Gateway 自动收集训练数据，DataPool 异步缓冲，Trainer 持续消费。

    [:octicons-arrow-right-24: Middleware Layer](concepts/middleware-layer.md)

-   :material-robot: **Production Agent Scenario**

    ---

    "部署 = 训练" 范式。Agent 在服务用户的同时持续收集数据和改进，支持白盒离线、黑盒离线和黑盒在线三种模式。

    [:octicons-arrow-right-24: Production Scenario](concepts/production-scenario.md)

-   :material-sync: **Async Training & Rollout**

    ---

    Rollout 和 Training 分离到独立 GPU 池，通过 DataPool 异步通信。支持 GRPO 和 GAE advantage 计算。

    [:octicons-arrow-right-24: Async Training](components/async-training.md)

</div>

## Why Claw-R1?

| 特性 | 传统 Agentic RL | Claw-R1 |
|---|---|---|
| Agent 接入 | 需要用框架 API 重写 | 只改 `base_url` |
| 训练数据 | 预收集的离线数据 | 实时交互自动收集 |
| 训练模式 | 同步（生成 → 训练交替） | 异步（生成和训练并行） |
| 部署方式 | 训练完成后部署固定模型 | 部署即训练，持续进化 |
| Agent 类型 | 仅支持框架内 Agent | 任何 OpenAI 兼容 Agent |

## 快速开始

```bash
# 安装
pip install -e .

# 运行黑盒 GSM8K 训练
export CUDA_VISIBLE_DEVICES=0,1,2
sh example/test_async_blackbox.sh
```

[:octicons-arrow-right-24: 完整安装指南](getting-started/installation.md) · [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

## 项目状态

| 模式 | 状态 |
|---|---|
| 白盒离线训练 | :material-check-circle: 已实现 |
| 黑盒离线训练 | :material-check-circle: 已实现 |
| 黑盒在线训练 | :material-progress-wrench: 开发中 |
| 异步训练 | :material-check-circle: 已实现 |

## Team

State Key Laboratory of Cognitive Intelligence, USTC

## Citation

```bibtex
@misc{claw-r1,
  title={Claw-R1: Empowering OpenClaw with Advanced Agentic RL},
  author={Claw-R1 Team},
  year={2025},
  url={https://github.com/AgentR1/Claw-R1}
}
```
