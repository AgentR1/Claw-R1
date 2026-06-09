# Dashboard

Claw-R1 Dashboard 是 Agentic RL 数据生命周期的在线控制台。它连接真实 Ray DataPool actor 和参数同步 actor，用于观察当前训练系统中的采集、表示、筛选、优化预览和训练消费状态。

## 能看到什么？

- **Overview**：端到端展示采集、reward、筛选、优化和消费状态。
- **Collection**：查看 DataPool 事件流，以及不同 channel 的数据来源分布。
- **Representation**：查看 step-level prompt/action token IDs、trajectory ID、reward 状态、policy version 和 metadata。
- **Curation**：查看并维护 batch quality label、trainability、tag 和 note。
- **Optimization**：基于真实 DataPool step 为某个 prompt group 生成 prefix-tree 预览。
- **Consumption**：查看 fetch-batch 计数、已消费 prompt group、policy sync version 和最近同步状态。

## 启动方式

训练任务创建 Ray actors 后，在仓库根目录运行：

```bash
conda activate steppo
sh example/start_dashboard.sh
```

默认服务地址为 `http://127.0.0.1:8120`。

也可以显式指定配置：

```bash
sh example/start_dashboard.sh \
  --ray-address auto \
  --ray-namespace claw_r1_async \
  --actor-name data_pool \
  --sync-actor-name parameter_synchronizer \
  --channel train,val \
  --port 8120
```

## 配置来源

Dashboard 支持三种配置方式：

- CLI 参数
- 环境变量
- `dashboard/config.example.yaml`

| 字段 | 默认值 | 作用 |
|---|---:|---|
| `ray_address` | `auto` | Ray cluster 地址。 |
| `ray_namespace` | `null` | 训练任务使用的 Ray namespace。 |
| `actor_name` | `data_pool` | DataPool Ray actor 名称。 |
| `sync_actor_name` | `parameter_synchronizer` | 参数同步 actor 名称。 |
| `channel` | `train` | 一个或多个 DataPool channel，多个值用逗号分隔。 |
| `refresh_interval_ms` | `2000` | 前端轮询间隔。 |
| `host` / `port` | `0.0.0.0` / `8120` | Dashboard 绑定地址。 |

如果 DataPool actor 不可用，Dashboard API 会返回 service-unavailable 错误，方便直接暴露部署问题。
