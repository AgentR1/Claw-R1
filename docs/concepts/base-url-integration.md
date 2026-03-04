# Base URL Integration

## The Problem

Every Agentic RL framework faces the same tension:

- The framework must **intercept LLM calls** to collect trajectory data for training
- But production agents (OpenClaw, AutoGen, CrewAI, LangChain) are often **black boxes** — you cannot modify their internal logic

Existing approaches each have significant costs:

| Approach | Examples | Integration Method | Drawback |
|---|---|---|---|
| Modify agent source | verl, RL-Factory | Embed rollout interfaces in agent code | High maintenance cost; impossible for true black-box agents |
| Python class wrapper | OpenRLHF | Inherit `AgentInstanceBase`, override execution | Requires understanding framework API; not portable |
| SDK hook patching | Agent Lightning, ART | Replace LangChain/OpenAI SDK HTTP layer | SDK-specific; breaks when the agent switches frameworks |
| **Redirect `base_url`** | **Claw-R1** | **Redirect LLM calls to Gateway** | **Zero changes; works with any HTTP client** |

## How It Works

All major LLM clients — OpenAI SDK, LangChain, LiteLLM, custom HTTP clients — follow the same protocol:

```
POST {base_url}/v1/chat/completions
Authorization: Bearer {api_key}
Content-Type: application/json
Body: { "model": "...", "messages": [...], ... }
```

Claw-R1's Gateway Server is a standard FastAPI HTTP service (an independent process, not a Ray actor) that implements a complete **OpenAI-compatible proxy**:

```
Before:  Agent ──► OpenAI/vLLM endpoint ──► model response

After:   Agent ──► Gateway (HTTP) ──► model response
                        │
                  [async, non-blocking]
                        │
                        ▼
                   DataPool (Ray Actor)
                   trajectory buffer
```

From the agent's perspective, it is talking to a slightly slower OpenAI API. It has no knowledge that every conversation is being recorded and fed into a training pipeline.

## Integration Example

### Black-box agent (OpenAI SDK)

```python
from openai import OpenAI
import uuid

# One-line change: redirect base_url to Gateway
# The URL path encodes trajectory_uid and prompt_uid
client = OpenAI(
    base_url=f"http://gateway-host:8000/{uuid.uuid4()}/{uuid.uuid4()}",
    api_key="unused",
)
```

### OpenClaw (TypeScript)

```yaml
# config.yaml
LLM_API_BASE: "http://gateway-host:8000"
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://gateway-host:8000/traj-123/prompt-456",
    api_key="unused",
)
```

## Why This Is Better Than SDK Hook Patching

Frameworks like Agent Lightning and ART intercept at the SDK layer:

```python
# SDK hook approach (illustrative)
import agent_lightning
agent_lightning.patch_openai()  # replaces openai.ChatCompletion.create
```

This approach has three fundamental limitations:

1. **Requires agent-side execution** — `patch_openai()` must run inside the agent's process, which still requires modifying the agent's startup code
2. **SDK-specific** — fails for agents using non-standard HTTP libraries
3. **Process-level coupling** — the agent and training framework share a process boundary

Claw-R1's Gateway is a **network-layer proxy**. It requires no shared process, no specific SDK, and no language dependency. This makes it work natively with:

- TypeScript/JavaScript agents (e.g., OpenClaw)
- Agents running in separate Docker containers or remote machines
- Agents using custom-built HTTP clients
- Any future agent framework, without framework updates

!!! tip "Trajectory and Prompt UIDs"
    The Gateway uses the URL path to extract `trajectory_uid` (identifies a conversation) and `prompt_uid` (groups rollouts for GRPO-style advantage computation). In white-box mode, these are managed automatically by `AgentFlowBase`. In black-box mode, the agent generates them as UUIDs and encodes them in the URL.
