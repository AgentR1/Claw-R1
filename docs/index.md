# Claw-R1

**The Data Foundation for Agentic Reinforcement Learning**

Claw-R1 是 Agentic RL 的数据基础设施 — 专注于从任意 Agent 交互中采集、评估和筛选高质量训练数据，并支持人类反馈信号的整合。

---

<div class="grid cards" markdown>

-   :material-database-arrow-right: **Universal Data Collection**

    ---

    从白盒、黑盒到在线服务 Agent，通过 `base_url` 机制零代码接入，自动采集交互数据。支持 LangChain、AutoGen、CrewAI 等任意 OpenAI 兼容 Agent。

    [:octicons-arrow-right-24: Base URL Integration](concepts/base-url-integration.md)

-   :material-layers-outline: **Data Middleware Layer**

    ---

    Gateway + DataPool 数据中间件：Gateway 采集交互数据，DataPool 管理数据质量、分区缓冲、按需供给训练引擎。

    [:octicons-arrow-right-24: Middleware Layer](concepts/middleware-layer.md)

-   :material-chart-bar: **Data Evaluation & Curation**

    ---

    多维 Reward 系统（规则/判别式 RM/生成式 RM）+ 人类反馈信号整合 + 策略版本追踪，系统性评估和筛选数据质量。

    [:octicons-arrow-right-24: Reward System](components/reward-system.md)

-   :material-robot: **Production Agent Scenario**

    ---

    "部署 = 训练" 范式。Agent 在服务用户的同时持续采集数据，用户行为（采纳、修改、追问）天然成为数据质量信号。

    [:octicons-arrow-right-24: Production Scenario](concepts/production-scenario.md)

</div>

## Why Claw-R1?

Agentic RL 生态正蓬勃发展 — verl、Agent-R1、Forge 等优秀框架在 Runtime 和训练算法方面持续推进。然而，随着 Agent 从简单 ReAct 演进到 Claude Code、OpenClaw 等通用架构，一个相对欠缺、值得深耕的方向逐渐浮现：**如何从多样的 Agent 交互中系统性地采集、评估和筛选高质量训练数据？**

Claw-R1 聚焦于这一方向，提供 Agent 与 Trainer 之间的**数据基础设施**。

| 维度 | 传统 Agentic RL 框架 | Claw-R1 |
|---|---|---|
| 核心关注 | 训练算法与 Runtime | **数据的采集、评估与筛选** |
| Agent 接入 | 需要用框架 API 重写 | 只改 `base_url`，零代码侵入 |
| 数据来源 | 预收集的离线数据 | 实时交互自动采集 + 离线数据集 |
| 数据质量管控 | 较少关注 | 多维 Reward + 人类反馈 + 新鲜度检测 |
| 训练引擎 | 内置绑定 | 可插拔 TrainingBackend，对接任意引擎 |

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

| 能力 | 状态 |
|---|---|
| 白盒 Agent 数据采集 | :material-check-circle: 已实现 |
| 黑盒 Agent 数据采集 | :material-check-circle: 已实现 |
| 在线服务数据采集 | :material-progress-wrench: 开发中 |
| 异步训练供给 | :material-check-circle: 已实现 |
| 人类反馈管线 | :material-progress-wrench: 规划中 |
| 数据质量 Dashboard | :material-progress-wrench: 规划中 |

## Team

State Key Laboratory of Cognitive Intelligence, USTC

## Citation

```bibtex
@misc{clawr1-2026,
  title={Claw-R1: The Data Foundation for Agentic Reinforcement Learning},
  author={Wang, Daoyu and Ouyang, Jie and Yu, Shuo and Cheng, Mingyue and Liu, Qi},
  year={2025},
  howpublished={\url{https://github.com/AgentR1/Claw-R1}},
  note={GitHub repository}
}
```
