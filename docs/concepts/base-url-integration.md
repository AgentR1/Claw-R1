# Base URL Integration

## 问题：如何拦截黑盒 Agent 的 LLM 调用？

在 Agentic RL 中，训练系统需要拦截 Agent 与 LLM 之间的每次交互，以收集 `(state, action, reward)` 数据。对于白盒 Agent（源码可控），这很简单。但对于黑盒 Agent（如第三方服务、编译后的二进制文件），如何在不修改 Agent 代码的情况下拦截？

## 方案对比

| 方案 | 侵入性 | 可靠性 | 适用范围 |
|---|---|---|---|
| SDK monkey-patch | 中 | 低（版本更新易失效） | 仅限特定 SDK |
| 代理层（Proxy） | 高 | 中（需配置网络） | 通用 |
| **base_url 替换** | **极低** | **高** | **所有 OpenAI 兼容 SDK** |

## base_url 机制

几乎所有 OpenAI 兼容的 SDK 都支持自定义 `base_url`。Claw-R1 利用这一点：

1. Gateway 暴露 `POST {base_url}/v1/chat/completions` 端点
2. Agent 只需将 `base_url` 从 `https://api.openai.com` 改为 Gateway 的地址
3. Gateway 透明地转发请求到 vLLM，同时自动收集训练数据

```python
from openai import OpenAI

# 原始代码
client = OpenAI(base_url="https://api.openai.com/v1")

# 接入 Claw-R1：只改一行
client = OpenAI(
    base_url="http://gateway:8100/traj123/prompt1",
    api_key="not-needed",
)

# 后续代码完全不变
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## base_url 的结构

```
http://<host>:<port>/<trajectory_uid>/<prompt_uid>
```

- `trajectory_uid`：标识一条完整的对话轨迹
- `prompt_uid`：标识同一 prompt 的多次 rollout（用于 GRPO 分组）

这两个 ID 编码在 URL path 中，Gateway 从 path 中提取，Agent 完全无感知。

## 在 Claw-R1 中的使用

### 黑盒离线模式

`BlackBoxAgentFlowBase` 自动管理 `base_url` 的生命周期：

```
1. POST /init_trajectory              → 获取 base_url
2. POST {base_url}/v1/register_trajectory  → 注册 channel/metadata
3. Agent 使用 base_url 进行多轮对话     → Gateway 自动收集 Step
4. POST {base_url}/v1/complete_trajectory  → 标记完成
```

Agent 只需要接收 `base_url` 参数，其余由训练框架处理。

### 黑盒在线模式

在线模式下，外部服务直接调用 Gateway 的 `init_trajectory` 获取 `base_url`，然后将其传递给 Agent。Agent 的每次 LLM 调用都自动被 Gateway 记录。

## 为什么优于 SDK Hook

| 维度 | SDK Hook | base_url |
|---|---|---|
| Agent 代码修改 | 需要注入 hook 代码 | 只改一个参数 |
| 多语言支持 | 每种语言需要单独实现 | 所有语言通用 |
| 版本兼容性 | SDK 更新可能破坏 hook | HTTP 协议稳定 |
| 调试难度 | Hook 层增加调试复杂度 | 标准 HTTP 请求，易于调试 |
| 生产可靠性 | 中等 | 高 |

## 支持的 SDK 和框架

任何支持自定义 `base_url` 的 OpenAI 兼容 SDK 都可以直接使用：

- **Python**: `openai`, `httpx`, `requests`
- **JavaScript/TypeScript**: `openai-node`
- **Go**: `go-openai`
- **框架**: LangChain, LlamaIndex, AutoGen, CrewAI 等
