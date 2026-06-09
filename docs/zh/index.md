# Claw-R1 中文文档

Claw-R1 是面向 Agentic Reinforcement Learning 的 step-level 数据中间件系统。它在 Agent 侧和训练侧之间提供一层可部署的数据基础设施，用于采集真实交互、评估数据质量、筛选可训练样本，并将整理后的数据供给 RL 训练后端。

当前中文文档覆盖项目概览、快速开始、核心架构和 Dashboard 使用说明。更细的 API 与算法细节仍以英文文档为主，可通过页面顶部语言切换返回英文版本。

## 最新动态

- **2026.06**：Claw-R1 Demo 技术报告已更新：[https://arxiv.org/abs/2606.09138](https://arxiv.org/abs/2606.09138)。
- **2026.06**：Dashboard 已接入真实 Ray DataPool 和参数同步 actor，用于 Agentic RL data lifestyle management。
- **2026.04**：Prefix Tree Merge 开始用于减少 multi-step agent 训练中的共享前缀重复计算。

## Claw-R1 解决什么问题？

Agentic RL 的训练栈通常重点关注 rollout runtime 和优化算法，但真实 Agent 交互数据本身也需要系统化管理：

- 如何让白盒 Agent、黑盒 Agent 和在线服务都能持续产出训练数据？
- 如何保留 step-level 表示、trajectory 归属、reward、policy version 和元数据？
- 如何将 rule reward、reward model、人类反馈和 freshness 信号统一为可筛选的数据质量信息？
- 如何把已筛选的数据按 channel 和 prompt group 稳定供给训练后端？

Claw-R1 的答案是 **Gateway + DataPool + TrainingBackend** 这一层数据中间件。

## 核心能力

| 能力 | 说明 |
|---|---|
| 通用数据采集 | 白盒 Agent 直接提交 `Step`；黑盒 Agent 通过 OpenAI 兼容 `base_url` 接入 Gateway。 |
| Step-level 表示 | 保存 prompt IDs、response IDs、reward、metadata、trajectory ID 和 step index。 |
| 数据评估与筛选 | 支持 rule-based reward、reward model、人类反馈、policy freshness 和 channel 分区。 |
| 训练数据供给 | 通过可插拔 `TrainingBackend` 将 DataPool 中的数据转换为训练引擎需要的 batch。 |
| 在线可观测性 | Dashboard 展示采集、表示、筛选、优化预览和训练消费状态。 |

## 快速入口

- [快速开始](getting-started.md)
- [架构与组件](components.md)
- [Dashboard 使用说明](dashboard.md)
- [英文文档](../index.md)
