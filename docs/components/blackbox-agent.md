# Black-box Agent

Black-box Agent 系统允许任何使用 OpenAI 兼容 API 的 Agent 接入 Claw-R1 的训练循环，无需修改 Agent 内部逻辑。Agent 只需将 `base_url` 指向 Gateway，即可透明地收集训练数据。

## 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│  BlackBoxAgentFlowBase (训练侧编排)                           │
│                                                               │
│  1. POST /init_trajectory          → 获取 base_url            │
│  2. POST {base_url}/v1/register_trajectory → 注册 metadata    │
│  3. 调用 _run_agent(base_url, kwargs)                         │
│     │                                                         │
│     │  ┌─────────────────────────────────┐                    │
│     └──│  具体 Agent (如 GSM8KAgent)      │                    │
│        │  只知道 base_url，使用 OpenAI API │                    │
│        │  POST {base_url}/v1/chat/completions (多轮)          │
│        └─────────────────────────────────┘                    │
│  4. POST {base_url}/v1/complete_trajectory → 标记完成          │
└──────────────────────────────────────────────────────────────┘
```

## 核心设计

### 关注点分离

- **BlackBoxAgentFlowBase**：处理与 Gateway 的完整协议（init → register → complete），是训练侧的编排层。
- **具体 Agent**（如 `GSM8KAgent`）：只接收 `base_url` 和任务参数，使用标准 OpenAI API 完成任务。Agent 完全不知道训练系统的存在。

这种分离使得：

- 同一个 Agent 可以在训练模式和独立服务模式下复用
- 新增任务只需实现 Agent + 对应的 Flow 子类
- Agent 可以用任何语言/框架实现，只要支持 OpenAI API

### 注册机制

Agent Flow 通过 `@register("name")` 装饰器注册，并在 YAML 配置中引用：

```yaml
# agent_flow_config.yaml
- name: blackbox_gsm8k_agent
  _target_: claw_r1.blackbox_agent.gsm8k_agent_flow.BlackBoxGSM8KAgentFlow
```

## 类层次

```
AgentFlowBase                         (agent_flow/agent_flow.py)
    │
    └── BlackBoxAgentFlowBase          (blackbox_agent/blackbox_agent_flow.py)
            │
            └── BlackBoxGSM8KAgentFlow (blackbox_agent/gsm8k_agent_flow.py)
```

### BlackBoxAgentFlowBase

所有黑盒 Agent Flow 的基类，实现了完整的 Gateway 协议：

```python
class BlackBoxAgentFlowBase(AgentFlowBase):

    async def run(self, sampling_params, **kwargs) -> int:
        # 1. 提取 channel、prompt_uid、metadata
        channel, prompt_uid, metadata = self._prepare_params(kwargs)

        # 2. init_trajectory → 获取 base_url
        init_resp = await http.post(f"{self.gateway_url}/init_trajectory")
        base_url = ...

        # 3. register_trajectory → 注册 channel 和 metadata
        await http.post(f"{base_url}/v1/register_trajectory", json={...})

        # 4. 调用子类实现的 _run_agent
        num_turns = await self._run_agent(base_url, kwargs)

        # 5. complete_trajectory → 标记完成
        await http.post(f"{base_url}/v1/complete_trajectory")

        return num_turns

    @abstractmethod
    async def _run_agent(self, base_url: str, kwargs: dict) -> int:
        """子类实现：创建并运行具体 Agent。"""
        ...
```

子类只需实现 `_run_agent`：从 `kwargs` 中提取任务参数，创建 Agent 实例，调用 Agent 的执行方法。

### BlackBoxGSM8KAgentFlow

GSM8K 数学题的具体实现：

```python
@register("blackbox_gsm8k_agent")
class BlackBoxGSM8KAgentFlow(BlackBoxAgentFlowBase):

    async def _run_agent(self, base_url: str, kwargs: dict) -> int:
        from claw_r1.blackbox_agent.gsm8k_agent import GSM8KAgent

        question = ...   # 从 kwargs["raw_prompt"] 提取
        ground_truth = ...  # 从 kwargs["reward_model"] 提取
        max_turns = self.config.actor_rollout_ref.rollout.get("max_turns", 3)

        agent = GSM8KAgent(base_url=base_url)
        return await agent.solve(
            question=question,
            ground_truth=ground_truth,
            max_turns=max_turns,
        )
```

## GSM8KAgent

一个训练无关的 Agent，使用 OpenAI 兼容 API 解决 GSM8K 数学题：

- 接收 `base_url`（指向 Gateway）和任务参数
- 使用 tool calling（`check_answer` 工具）进行多轮推理
- 支持 Qwen 风格的 tool call 解析（`✿FUNCTION✿` 格式）
- 返回使用的轮次数

```python
agent = GSM8KAgent(base_url="http://gateway:8100/traj123/1")
num_turns = await agent.solve(
    question="What is 15 * 23?",
    ground_truth="345",
    max_turns=3,
)
```

## 添加新的黑盒 Agent

1. **实现 Agent 类**（训练无关）：

```python
# claw_r1/blackbox_agent/my_agent.py
class MyAgent:
    def __init__(self, base_url: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key="x")

    async def solve(self, task: str, **kwargs) -> int:
        # 使用 self.client 进行多轮对话
        # 返回使用的轮次数
        ...
```

2. **实现 Flow 子类**：

```python
# claw_r1/blackbox_agent/my_agent_flow.py
from claw_r1.agent_flow.agent_flow import register
from claw_r1.blackbox_agent.blackbox_agent_flow import BlackBoxAgentFlowBase

@register("blackbox_my_agent")
class BlackBoxMyAgentFlow(BlackBoxAgentFlowBase):
    async def _run_agent(self, base_url, kwargs):
        from claw_r1.blackbox_agent.my_agent import MyAgent
        task = kwargs.get("raw_prompt", "")
        agent = MyAgent(base_url=base_url)
        return await agent.solve(task=task)
```

3. **注册到配置**：

```yaml
# agent_flow_config.yaml
- name: blackbox_my_agent
  _target_: claw_r1.blackbox_agent.my_agent_flow.BlackBoxMyAgentFlow
```

4. **在训练脚本中使用**：

```bash
python3 -m claw_r1.async_main \
    actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_my_agent \
    actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml \
    ...
```

## 文件结构

```
claw_r1/blackbox_agent/
├── blackbox_agent_flow.py      # BlackBoxAgentFlowBase 基类
├── gsm8k_agent_flow.py         # GSM8K Flow 子类
├── gsm8k_agent.py              # GSM8K Agent（训练无关）
└── agent_flow_config.yaml      # Agent Flow 注册配置
```
