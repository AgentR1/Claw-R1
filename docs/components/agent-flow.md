# Agent Flow

Agent Flow 是 Claw-R1 中管理 Agent 执行生命周期的框架。它分为两大类：

- **白盒 Agent Flow**：Agent 逻辑用 Python 编写，直接通过 Gateway 的 `/generate`、`/submit_steps` 等端点交互，自行管理 tokenize 和 Step 构建。
- **黑盒 Agent Flow**：Agent 使用标准 OpenAI API，通过 `base_url` 透明接入，Gateway 自动处理 tokenize 和 Step 提交。

## 类层次

```
AgentFlowBase                              (abstract base)
    │
    ├── SingleStepSingleTurnAgentFlow      (白盒：单轮问答)
    ├── MultiStepAgentFlow                 (白盒：多轮工具调用)
    │
    └── BlackBoxAgentFlowBase              (黑盒基类)
            └── BlackBoxGSM8KAgentFlow     (黑盒：GSM8K 数学题)
```

## AgentFlowBase

所有 Agent Flow 的抽象基类，提供：

- Gateway URL 管理
- 配置访问（`self.config`）
- 抽象方法 `run(sampling_params, **kwargs) -> int`

### 白盒辅助方法

白盒 Agent Flow 可使用以下方法与 Gateway 交互：

#### `gateway_generate(trajectory_uid, prompt_uid, messages, **kwargs)`

向 Gateway `/generate` 发送异步 HTTP POST，返回生成文本和 token IDs。

```python
text, response_ids, prompt_ids = await self.gateway_generate(
    trajectory_uid="traj-abc",
    prompt_uid="prompt-xyz",
    messages=[{"role": "user", "content": "Summarize this document."}],
    max_tokens=512,
    temperature=0.8,
)
```

#### `gateway_submit_steps(steps, channel="train")`

向 Gateway `/submit_steps` 提交 Step 列表。

#### `gateway_compute_reward(trajectory_uid, messages, dataset_fields)`

向 Gateway `/compute_reward` 请求 reward 计算。

## SingleStepSingleTurnAgentFlow

最简单的白盒实现：单个 prompt 产生单个 response。适用于每个样本都是独立问答对的数据集。

```python
class MyAgentFlow(SingleStepSingleTurnAgentFlow):
    async def run(self, sampling_params, **kwargs) -> int:
        messages = [{"role": "user", "content": kwargs["raw_prompt"]}]
        text, response_ids, prompt_ids = await self.gateway_generate(
            trajectory_uid=kwargs["trajectory_uid"],
            prompt_uid=kwargs["prompt_uid"],
            messages=messages,
        )
        step = Step(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            reward=0.0,
            trajectory_uid=kwargs["trajectory_uid"],
            prompt_uid=kwargs["prompt_uid"],
            step_index=0,
            is_last=True,
        )
        await self.gateway_submit_steps([step])
        return 1
```

## MultiStepAgentFlow

多轮 Agent Flow，支持工具调用、规划等场景。每轮产生一个 Step，通过 `trajectory_uid` 串联。

```python
class ToolAgentFlow(MultiStepAgentFlow):
    async def run(self, sampling_params, **kwargs) -> int:
        messages = [{"role": "user", "content": kwargs["task"]}]
        step_index = 0

        while True:
            text, response_ids, prompt_ids = await self.gateway_generate(...)
            is_last = self.is_terminal(text)

            step = Step(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                step_index=step_index,
                is_last=is_last,
                ...
            )
            await self.gateway_submit_steps([step])

            if is_last:
                break

            messages.append({"role": "assistant", "content": text})
            tool_result = await self.execute_tool(text)
            messages.append({"role": "tool", "content": tool_result})
            step_index += 1

        return step_index + 1
```

## BlackBoxAgentFlowBase

黑盒 Agent Flow 的基类。处理与 Gateway 的完整协议（init → register → complete），将 Agent 执行委托给子类的 `_run_agent` 方法。

详细文档见 [Black-box Agent](blackbox-agent.md)。

## 注册机制

Agent Flow 通过 `@register("name")` 装饰器注册到全局注册表：

```python
from claw_r1.agent_flow.agent_flow import register

@register("my_agent_flow")
class MyAgentFlow(AgentFlowBase):
    ...
```

也可通过 YAML 配置文件注册（用于黑盒 Agent）：

```yaml
# agent_flow_config.yaml
- name: blackbox_gsm8k_agent
  _target_: claw_r1.blackbox_agent.gsm8k_agent_flow.BlackBoxGSM8KAgentFlow
```

## AgentFlowManager 和 AgentFlowWorker

- **AgentFlowManager**：管理多个 `AgentFlowWorker`，将 batch 中的每个样本分发给对应的 Agent Flow 执行。
- **AgentFlowWorker**：Ray Actor，持有 tokenizer 和配置，执行具体的 Agent Flow。

```
AsyncRollouter
    └── AgentFlowManager
            └── AgentFlowWorker (Ray Actor, 可多个)
                    └── AgentFlowBase 子类实例
```

## 配置

在训练脚本中指定 Agent Flow：

```bash
python3 -m claw_r1.async_main \
    actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_gsm8k_agent \
    actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml \
    ...
```
