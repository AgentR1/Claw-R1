# Gateway Server

Gateway Server 是一个 **FastAPI HTTP 服务**，作为 Agent 与 Claw-R1 训练基础设施之间的网络层代理。

## 设计原则

- **独立进程**：Gateway 作为普通 OS 进程运行（非 Ray Actor），可以独立于 Ray 集群重启。
- **纯代理**：Gateway 不管理任何引擎生命周期，只负责转发请求、收集 Step、提交到 DataPool。
- **OpenAI 兼容**：黑盒端点实现与 OpenAI chat completions API 相同的接口，可作为 drop-in 替换。
- **延迟初始化**：启动时先快速初始化 Ray 连接和配置，HTTP 服务立即可用；tokenizer 在后台线程加载，通过 `/ready` 端点报告就绪状态。

## 启动方式

Gateway 通常由 `AsyncRollouter` 作为子进程自动启动。也可手动启动：

```bash
python -m claw_r1.gateway.gateway \
    --data-pool-name  data_pool \
    --vllm-addresses  http://host1:8001,http://host2:8001 \
    --tokenizer-path  /path/to/model \
    --prompt-length   4096 \
    --response-length 1024 \
    --reward-worker-name reward_loop_worker \
    --ray-address     auto \
    --ray-namespace   default \
    --host            0.0.0.0 \
    --port            8100
```

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--data-pool-name` | 是 | DataPool 的 Ray Actor 名称 |
| `--vllm-addresses` | 是 | 逗号分隔的 vLLM 服务器地址列表（轮询负载均衡） |
| `--tokenizer-path` | 是 | HuggingFace tokenizer 路径 |
| `--prompt-length` | 是 | 最大 prompt token 长度（用于 padding） |
| `--response-length` | 是 | 最大 response token 长度（用于 padding） |
| `--reward-worker-name` | 否 | RewardLoopWorker 的 Ray Actor 名称 |
| `--ray-address` | 否 | Ray GCS 地址（默认 `auto`） |
| `--ray-namespace` | 否 | Ray namespace |
| `--host` | 否 | 监听地址（默认 `0.0.0.0`） |
| `--port` | 否 | 监听端口（默认 `8100`） |

## 两种工作模式

### White-box 模式

白盒 Agent（`AgentFlowBase` 子类）通过 Gateway 根路径端点交互：

```
AgentFlow → POST /generate        → vLLM → 返回 token IDs
AgentFlow → POST /submit_steps    → DataPool
AgentFlow → POST /compute_reward  → RewardLoopWorker
```

Agent 自己管理 tokenize、Step 构建和提交。

### Black-box 模式

黑盒 Agent 只需要一个 `base_url`，通过标准 OpenAI 接口交互：

```
1. BlackBoxAgentFlow → POST /init_trajectory           → 获取 base_url
2. BlackBoxAgentFlow → POST {base_url}/v1/register_trajectory  → 注册 channel/metadata
3. Agent             → POST {base_url}/v1/chat/completions     → 标准 OpenAI 调用（可多轮）
4. BlackBoxAgentFlow → POST {base_url}/v1/complete_trajectory  → 标记完成
```

Gateway 在 `v1/chat/completions` 内部自动完成 tokenize、Step 构建和 DataPool 提交，Agent 完全无感知。

## base_url 机制

`base_url` 的格式为：

```
http://<host>:<port>/<trajectory_uid>/<prompt_uid>
```

`trajectory_uid` 和 `prompt_uid` 编码在 URL path 中，使得 Gateway 能将请求关联到正确的 trajectory，而 Agent 端只需修改 `base_url` 即可接入训练系统。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://gateway:8100/abc123/1",  # base_url 由 init_trajectory 返回
    api_key="not-needed",
)
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## 内部状态管理

Gateway 为每条 trajectory 维护以下状态：

| 状态 | 说明 |
|---|---|
| `_trajectory_step_counter` | 每条 trajectory 的下一个 step_index |
| `_trajectory_channel` | trajectory 对应的 DataPool channel（默认 `"train"`） |
| `_trajectory_metadata` | trajectory 关联的 metadata（如 reward_model、data_source 等） |

这些状态在 `register_trajectory` 时设置，在 `complete_trajectory` 时清理。

## 负载均衡

当提供多个 `--vllm-addresses` 时，Gateway 使用 **round-robin** 轮询分发请求：

```python
_vllm_cycle = itertools.cycle(vllm_addresses)
vllm_url = next(_vllm_cycle)
```

## API 参考

完整的端点文档见 [Gateway API](../api/gateway.md)。
