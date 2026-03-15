# Gateway API

Gateway 默认监听端口 **8100**（通过 `--port` 配置）。所有端点均接受和返回 JSON。

## Base URL

```
http://<gateway-host>:8100
```

---

## White-box 端点

这些端点由 `AgentFlowBase` 的白盒 Agent 调用。

### `POST /generate`

将生成请求转发到 vLLM 并返回带 token ID 的响应。

**调用方**: `AgentFlowBase.gateway_generate()`

#### Request

```json
{
  "trajectory_uid": "string",
  "prompt_uid": "string",
  "messages": [
    { "role": "user", "content": "string" }
  ],
  "max_tokens": 1024,
  "temperature": 1.0,
  "top_p": 1.0
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `trajectory_uid` | string | 是 | 当前对话的唯一 ID |
| `prompt_uid` | string | 是 | Prompt 组 ID（用于 GRPO 分组） |
| `messages` | array | 是 | OpenAI 格式的聊天消息 |
| `max_tokens` | int | 否 | 最大响应长度（默认取 `--response-length`） |
| `temperature` | float | 否 | 采样温度（默认 1.0） |
| `top_p` | float | 否 | Top-p 采样（默认 1.0） |

#### Response

```json
{
  "response_text": "string",
  "response_ids": [101, 202, 303],
  "prompt_ids": [50, 60, 70, 80]
}
```

---

### `POST /submit_steps`

提交一个或多个 `Step` 对象到 DataPool。

**调用方**: `AgentFlowBase.gateway_submit_steps()`

#### Request

```json
{
  "steps": [
    {
      "trajectory_uid": "string",
      "prompt_uid": "string",
      "prompt_ids": [50, 60, 70],
      "response_ids": [101, 202],
      "reward": 0.0,
      "step_index": 0,
      "policy_version": 42,
      "is_last": true,
      "metadata": {}
    }
  ]
}
```

#### Response

```json
{
  "accepted": 1
}
```

---

### `POST /compute_reward`

为一个 step 计算 reward（由 Trainer 调用，不由 Agent 调用）。

#### Request

```json
{
  "trajectory_uid": "string",
  "messages": [...],
  "dataset_fields": {
    "ground_truth": "string",
    "task_type": "string"
  }
}
```

#### Response

```json
{
  "reward": 0.85
}
```

---

## Black-box 端点

这些端点供黑盒 Agent 使用。黑盒 Agent 只需要知道一个 `base_url`，所有交互都通过该 URL 完成。

`base_url` 的格式为 `http://<host>:<port>/<trajectory_uid>/<prompt_uid>`，由 `POST /init_trajectory` 返回。

### `POST /init_trajectory`

分配一条新的 trajectory 并返回 `base_url`。

#### Request

无请求体。

#### Response

```json
{
  "trajectory_uid": "a1b2c3d4e5f6...",
  "base_url": "http://0.0.0.0:8100/a1b2c3d4e5f6.../1"
}
```

---

### `POST {base_url}/v1/register_trajectory`

注册 trajectory 的 channel 和 metadata。在 Agent 开始交互之前调用。

`trajectory_uid` 从 URL path 中提取，无需在 body 中传递。

#### Request

```json
{
  "channel": "train",
  "metadata": {
    "data_source": "gsm8k",
    "ground_truth": "42"
  }
}
```

所有字段均为可选。`channel` 默认为 `"train"`。

#### Response

```json
{ "status": "ok" }
```

---

### `POST {base_url}/v1/chat/completions`

OpenAI 兼容的聊天补全端点。黑盒 Agent 只需将 `base_url` 设为 OpenAI SDK 的 `base_url`，即可透明接入训练系统。

Gateway 会：

1. 将请求转发到 vLLM 服务器
2. 对 prompt 和 response 进行 tokenize
3. 自动构建 `Step` 并提交到 DataPool
4. 返回标准 OpenAI 格式的响应

#### Request

标准 OpenAI `chat/completions` 请求体。

```json
{
  "model": "qwen",
  "messages": [
    { "role": "user", "content": "What is 2+2?" }
  ],
  "temperature": 0.7
}
```

#### Response

标准 OpenAI `chat/completions` 响应体。

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "4"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 1,
    "total_tokens": 11
  }
}
```

---

### `POST {base_url}/v1/complete_trajectory`

标记一条 trajectory 完成。Agent 完成所有交互后调用。

#### Request

无请求体。

#### Response

```json
{ "status": "ok" }
```

---

### `POST /complete_trajectory/{trajectory_uid}`

内部端点，通过 trajectory_uid 直接标记完成。可选传入 reward 和 channel。

#### Request

```json
{
  "channel": "train",
  "reward": 0.9
}
```

#### Response

```json
{ "status": "ok" }
```

---

## 就绪检查

### `GET /ready`

当 Gateway 完全初始化（包括 tokenizer 加载完成）后返回 200。用于 Rollouter 启动时的健康检查。

#### Response (200)

```json
{ "status": "ready" }
```

#### Response (503)

```json
{ "detail": "Gateway not ready (tokenizer still loading)" }
```

---

## `GET /docs`

FastAPI 自动生成的 Swagger UI 文档页面。
