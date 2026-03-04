# Quick Start

This guide walks you through two minimal examples: a **white-box agent** (AgentFlow managed by Claw-R1) and a **black-box agent** (any HTTP-based agent, zero modification required).

## White-box Mode

In white-box mode, Claw-R1 owns the agent loop via `AgentFlowBase`. The framework can observe token IDs, compute rewards internally, and submit steps directly.

### 1. Start the Gateway

```bash
python -m claw_r1.gateway.gateway \
    --data-pool-name data_pool \
    --vllm-addresses http://localhost:8001 \
    --tokenizer-path /path/to/your/model \
    --prompt-length 4096 \
    --response-length 1024
```

### 2. Launch Training

Use the provided Hydra configuration:

```bash
python -m claw_r1.main_agent_ppo \
    --config-name agent_ppo_trainer \
    trainer.model.path=/path/to/your/model \
    trainer.project_name=claw_r1_quickstart
```

### 3. Monitor

Ray Dashboard is available at `http://localhost:8265` by default once the Ray cluster is initialized.

---

## Black-box Mode

In black-box mode, your existing agent needs **only one change**: redirect its `base_url` to the Gateway.

### Any Python agent (OpenAI SDK)

```python
from openai import OpenAI
import uuid

traj_uid  = str(uuid.uuid4())  # unique per conversation
prompt_uid = str(uuid.uuid4()) # unique per prompt group (for GRPO)

# Before
# client = OpenAI(base_url="https://api.openai.com/v1")

# After — single line change
client = OpenAI(
    base_url=f"http://localhost:8000/{traj_uid}/{prompt_uid}",
    api_key="not-used",
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello, what can you do?"}],
)
print(response.choices[0].message.content)
```

### OpenClaw

In your OpenClaw configuration file, change:

```yaml
# Before
LLM_API_BASE: "https://api.openai.com/v1"

# After
LLM_API_BASE: "http://gateway-host:8000"
```

That's it. Every LLM call OpenClaw makes will be transparently intercepted, logged to DataPool, and used for training — with no interruption to service.

---

## Async Training Mode

For production scenarios where the agent serves requests continuously, use the fully-async entry point:

```bash
python -m claw_r1.async_main \
    --config-name async_ppo_trainer \
    trainer.model.path=/path/to/your/model
```

The async runner automatically:

1. Initializes the `DataPool` Ray actor
2. Starts the `AsyncRollouter` on the rollout GPU pool
3. Starts the `AsyncTrainer` on the training GPU pool
4. Starts a `ParameterSynchronizer` that periodically pushes updated weights to the rollout vLLM servers
5. Starts the `Gateway` as an independent FastAPI process

!!! info "No dataset required"
    In async online mode, the training data comes entirely from live agent interactions. No pre-collected dataset is needed — the DataPool fills automatically as the agent handles real requests.

---

## Next Steps

- [Concepts: Base URL Integration](../concepts/base-url-integration.md)
- [Components: Gateway Server](../components/gateway.md)
- [Configuration Reference](../configuration/index.md)
