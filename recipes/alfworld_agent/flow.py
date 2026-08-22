"""TextWorld ALFWorld agent built directly on the Claw-R1 Gateway."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from claw_r1.agent_flow.agent_flow import AgentFlowBase, register
from claw_r1.blackbox_runtime.protocol import parse_tool_calls
from claw_r1.data_pool.data_model import Step

SYSTEM = "You are an ALFWorld agent. Call env_step with exactly one admissible command."
TOOLS = [{"type": "function", "function": {"name": "env_step", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def _env(game_file: str, max_steps: int):
    import textworld
    import textworld.gym
    infos = textworld.EnvInfos(won=True, admissible_commands=True)
    return textworld.gym.make(textworld.gym.register_game(game_file, infos, max_episode_steps=max_steps))


@register("alfworld_agent")
class ALFWorldAgentFlow(AgentFlowBase):
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> int:
        metadata = {k: v for k, v in kwargs.items() if k not in {"raw_prompt", "agent_name", "channel"}}
        game_file = metadata.get("game_file")
        if not game_file: raise ValueError("ALFWorld recipe requires sample metadata.game_file")
        max_steps = int(metadata.get("max_steps", 50))
        environment = _env(str(Path(game_file)), max_steps)
        trajectory, prompt_uid = uuid4().hex, str(kwargs.get("uid", uuid4().hex))
        channel = kwargs.get("channel")
        steps: list[Step] = []
        success = False; invalid = 0
        try:
            observation, infos = environment.reset()
            admissible = list(infos.get("admissible_commands", []) or [])
            messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _state(observation, admissible, kwargs.get("raw_prompt"))}]
            for index in range(max_steps):
                prompt_ids = await self.apply_chat_template(messages, tools=TOOLS)
                generated = await self.gateway_generate(prompt_ids, sampling_params)
                response_ids = generated["token_ids"][: self.config.actor_rollout_ref.rollout.response_length]
                text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                content, calls = parse_tool_calls(text)
                step = Step(prompt_ids=prompt_ids, response_ids=response_ids, rollout_log_probs=(generated.get("log_probs") or [])[: len(response_ids)], trajectory_uid=trajectory, prompt_uid=prompt_uid, step_index=index, policy_version=int(kwargs.get("policy_version", 0) or 0), is_last=False, metadata=dict(metadata))
                steps.append(step); messages.append({"role": "assistant", "content": content, "tool_calls": calls or None})
                command = _command(calls)
                if command not in admissible:
                    invalid += 1; messages.append({"role": "tool", "tool_call_id": calls[0]["id"] if calls else "invalid", "content": f"Invalid command. Admissible commands:\n{chr(10).join(admissible)}"}); continue
                observation, _reward, done, infos = environment.step(command)
                admissible = list(infos.get("admissible_commands", []) or [])
                messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": _state(observation, admissible, None)})
                if infos.get("won", False): success = True; break
                if done: break
        finally:
            environment.close()
        if not steps: return 0
        final_metadata = {**metadata, "task_success": success, "num_steps": len(steps), "invalid_action_count": invalid}
        for step in steps: step.metadata = final_metadata
        steps[-1].is_last = True; steps[-1].reward = 1.0 if success else 0.0
        await self.gateway_submit_steps(steps, channel=channel)
        return len(steps)


def _command(calls: list[dict[str, Any]]) -> str | None:
    if not calls: return None
    try: return json.loads(calls[0]["function"]["arguments"]).get("command")
    except (KeyError, TypeError, json.JSONDecodeError): return None


def _state(observation: str, commands: list[str], fallback: Any) -> str:
    task = fallback if isinstance(fallback, str) else ""
    return f"Task: {task}\nObservation: {observation}\nAdmissible commands:\n" + "\n".join(f"- {c}" for c in commands)
