# 快速开始

本页说明如何运行 Claw-R1 的异步训练示例，并启动 Dashboard 查看数据流状态。

## 环境准备

Claw-R1 复用 `verl` 的训练环境。请先参考英文安装页完成基础环境，并确保环境中包含 `verl==0.7.0`。

```bash
conda activate steppo
git clone https://github.com/AgentR1/Claw-R1
cd Claw-R1
```

示例默认需要至少 3 张 GPU，并需要将 GSM8K parquet 文件放到脚本期望的位置。

## 运行黑盒 Agent 训练

黑盒模式适合已经使用 OpenAI 兼容 API 的 Agent。Agent 不需要改内部逻辑，只需要把请求指向 Gateway，Claw-R1 就能采集交互数据。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
sh example/test_async_blackbox.sh
```

脚本会启动 Ray、创建 DataPool actor、启动 trainer 与 rollout GPU 池、在 `8100` 端口启动 Gateway，并运行注册好的 GSM8K black-box agent flow。

## 启动 Dashboard

训练 actor 已启动后，在第二个终端运行：

```bash
conda activate steppo
sh example/start_dashboard.sh
```

默认访问地址为 `http://127.0.0.1:8120`。Dashboard 会读取真实 Ray actor，而不是展示 mock 数据。

## 运行白盒 Agent 示例

白盒模式中，Agent 代码直接构造并提交 `Step` 对象：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
sh example/test_async.sh
```

## 常用配置项

```bash
trainer.n_gpus_per_node=2
rollout.n_gpus_per_node=1
actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_gsm8k_agent
actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml
async_training.trigger_parameter_sync_step=1
actor_rollout_ref.rollout.n=5
```

## 下一步

- [架构与组件](components.md)
- [Dashboard 使用说明](dashboard.md)
- [英文配置文档](../configuration/index.md)
- [英文 Gateway API](../api/gateway.md)
