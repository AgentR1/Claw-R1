# Gateway Server

The Gateway Server is a **FastAPI HTTP service** that acts as the network-layer proxy between agents and the Claw-R1 training infrastructure.

## Design Principles

- **Independent process**: The Gateway runs as a plain OS process, not a Ray actor. This means it can be restarted independently of the Ray cluster.
- **Pure proxy**: The Gateway does not manage any engine lifecycle. It only forwards requests, collects steps, and submits to DataPool.
- **OpenAI-compatible**: Implements the same interface as OpenAI's chat completions API, making it a drop-in replacement.

## Starting the Gateway

```bash
python -m claw_r1.gateway.gateway \
    --data-pool-name  data_pool \
    --vllm-addresses  http://host1:8001,http://host2:8001 \
    --tokenizer-path  /path/to/model \
    --prompt-length   4096 \
    --response-length 1024
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--data-pool-name` | Yes | Ray actor name of the DataPool to connect to |
| `--vllm-addresses` | Yes | Comma-separated list of vLLM server addresses (load-balanced round-robin) |
| `--tokenizer-path` | Yes | Path to the HuggingFace tokenizer |
| `--prompt-length` | Yes | Maximum prompt token length (for padding) |
| `--response-length` | Yes | Maximum response token length (for padding) |

## Endpoints

### `POST /generate` (white-box mode)

Called by `AgentFlowBase.gateway_generate()`. Forwards the generation request to a vLLM server, tokenizes the response, and returns token IDs.

```python
# Request
{
    "trajectory_uid": "string",
    "prompt_uid": "string",
    "messages": [...],         # OpenAI chat messages
    "max_tokens": 1024,
    "temperature": 1.0
}

# Response
{
    "response_text": "string",
    "response_ids": [101, 202, ...],   # token IDs
    "prompt_ids": [50, 60, ...]        # full context token IDs
}
```

### `POST /submit_steps` (white-box mode)

Called by `AgentFlowBase.gateway_submit_steps()`. Submits one or more `Step` objects to the DataPool.

```python
# Request
{
    "steps": [
        {
            "trajectory_uid": "string",
            "prompt_uid": "string",
            "prompt_ids": [...],
            "response_ids": [...],
            "reward": 0.0,
            "step_index": 0,
            "policy_version": 42,
            "is_last": false,
            "metadata": {}
        }
    ]
}
```

### `POST /compute_reward`

Computes a reward score for a completed trajectory step.

```python
# Request
{
    "trajectory_uid": "string",
    "messages": [...],
    "dataset_fields": {}    # task-specific fields (ground truth, etc.)
}

# Response
{
    "reward": 0.85
}
```

### `POST /{trajectory_uid}/{prompt_uid}/v1/chat/completions` *(reserved)*

OpenAI-compatible endpoint for black-box agents. The `trajectory_uid` and `prompt_uid` are encoded in the URL path, allowing the Gateway to associate incoming requests with the correct trajectory without any client-side changes beyond `base_url`.

!!! info "Status"
    This endpoint is designed and stubbed. Full black-box online integration is under active development.

### `POST /complete_trajectory/{trajectory_uid}` *(reserved)*

Called by black-box agents to mark the end of a trajectory and optionally provide a final reward.

## Load Balancing

When multiple `--vllm-addresses` are provided, the Gateway distributes requests across them using **round-robin**:

```python
# Internal: cycle through vLLM addresses
self.vllm_address_cycle = itertools.cycle(vllm_addresses)
vllm_url = next(self.vllm_address_cycle)
```

This provides basic load balancing without requiring an external proxy.
