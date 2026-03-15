# Quick Start

本指南展示如何快速运行 Claw-R1 的异步训练。

## 前置条件

- 已完成 [安装](installation.md)
- 至少 3 张 GPU（2 张训练 + 1 张推理）
- 训练数据（parquet 格式）

## Black-box 模式（推荐入门）

黑盒模式下，Agent 使用标准 OpenAI API 与 Gateway 交互，无需修改 Agent 代码。以 GSM8K 数学题为例：

### 1. 准备数据

```bash
# 下载 GSM8K 数据集（parquet 格式）
# 确保 train.parquet 和 test.parquet 在 ~/data/gsm8k/ 下
```

### 2. 运行训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

sh example/test_async_blackbox.sh
```

该脚本会：

1. 启动 Ray 集群
2. 创建 DataPool（Ray Actor）
3. 在 GPU 0-1 上部署 Actor + Critic（训练）
4. 在 GPU 2 上部署 vLLM（推理）
5. 启动 Gateway（端口 8100）
6. 运行 `BlackBoxGSM8KAgentFlow`：
    - 为每个样本调用 `init_trajectory` 获取 `base_url`
    - 创建 `GSM8KAgent`，使用 `base_url` 作为 OpenAI API 的 endpoint
    - Agent 通过多轮 tool calling 解题
    - Gateway 自动收集每轮对话为 Step 并提交到 DataPool
7. AsyncTrainer 从 DataPool 拉取 batch 进行 PPO 训练
8. 定期同步权重到 vLLM

### 3. 关键配置参数

```bash
# GPU 分配
trainer.n_gpus_per_node=2        # 训练用 2 张 GPU
rollout.n_gpus_per_node=1        # 推理用 1 张 GPU

# Agent Flow
actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_gsm8k_agent
actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml

# 异步训练
async_training.trigger_parameter_sync_step=1   # 每步同步权重
actor_rollout_ref.rollout.n=5                  # 每个 prompt 生成 5 条 trajectory
```

## White-box 模式

白盒模式下，Agent 逻辑用 Python 编写，直接通过 Gateway 的 `/generate` 和 `/submit_steps` 端点交互。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

sh example/test_async.sh
```

白盒模式使用 `MultiStepAgentFlow` 或 `SingleStepSingleTurnAgentFlow`，Agent 自行管理 tokenize 和 Step 构建。

## 自定义 Agent

### 添加黑盒 Agent

1. 实现 Agent 类（只需 `base_url` 和 OpenAI API）
2. 实现 `BlackBoxAgentFlowBase` 子类
3. 在 `agent_flow_config.yaml` 中注册
4. 在训练脚本中指定

详细步骤见 [Black-box Agent](../components/blackbox-agent.md)。

### 添加白盒 Agent

1. 继承 `AgentFlowBase`（或 `MultiStepAgentFlow`）
2. 实现 `run()` 方法
3. 使用 `@register("name")` 注册

详细步骤见 [Agent Flow](../components/agent-flow.md)。

## 监控训练

训练日志默认输出到控制台。可配置 SwanLab 等日志后端：

```bash
trainer.logger='["console","swanlab"]'
trainer.project_name='my_project'
trainer.experiment_name='my_experiment'
```

## 下一步

- [Components](../components/index.md) — 了解各组件的详细设计
- [Configuration](../configuration/index.md) — 完整配置参考
- [Gateway API](../api/gateway.md) — HTTP 端点文档
