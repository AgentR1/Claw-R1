# Gateway API

The Gateway exposes an HTTP API on port `8000` (default). All endpoints accept and return JSON.

## Base URL

```
http://<gateway-host>:8000
```

---

## `POST /generate`

Forward a generation request to vLLM and return the response with token IDs.

**Used by**: `AgentFlowBase.gateway_generate()`

### Request

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

| Field | Type | Required | Description |
|---|---|---|---|
| `trajectory_uid` | string | Yes | Unique ID for this conversation |
| `prompt_uid` | string | Yes | Unique ID for the prompt group (GRPO grouping) |
| `messages` | array | Yes | OpenAI-format chat messages |
| `max_tokens` | int | No | Max response length (default: from `--response-length`) |
| `temperature` | float | No | Sampling temperature (default: 1.0) |
| `top_p` | float | No | Top-p sampling (default: 1.0) |

### Response

```json
{
  "response_text": "string",
  "response_ids": [101, 202, 303],
  "prompt_ids": [50, 60, 70, 80]
}
```

| Field | Type | Description |
|---|---|---|
| `response_text` | string | Generated text |
| `response_ids` | int[] | Token IDs of the generated response |
| `prompt_ids` | int[] | Full context token IDs (prompt + prior turns) |

---

## `POST /submit_steps`

Submit one or more `Step` objects to the DataPool.

**Used by**: `AgentFlowBase.gateway_submit_steps()`

### Request

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

### Response

```json
{
  "submitted": 1,
  "status": "ok"
}
```

---

## `POST /compute_reward`

Compute a reward score for a step (called by Trainer, not by agents).

### Request

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

### Response

```json
{
  "reward": 0.85
}
```

---

## `POST /{trajectory_uid}/{prompt_uid}/v1/chat/completions` *(reserved)*

OpenAI-compatible endpoint for black-box agents. The `trajectory_uid` and `prompt_uid` are embedded in the URL path.

### Request

Standard OpenAI `chat/completions` request body.

### Response

Standard OpenAI `chat/completions` response.

!!! info "Status: Reserved"
    This endpoint is designed and stubbed. It will be the primary integration point for black-box online agents in a future release.

---

## `POST /complete_trajectory/{trajectory_uid}` *(reserved)*

Signal that a trajectory is complete. Used in black-box mode when the agent manages conversation boundaries.

### Request

```json
{
  "trajectory_uid": "string",
  "final_reward": 0.9
}
```

### Response

```json
{ "status": "ok" }
```

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```
