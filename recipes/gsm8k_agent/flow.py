from typing import Any

from claw_r1.agent_flow.agent_flow import register
from claw_r1.blackbox_runtime.flow import BlackBoxAgentFlowBase
from .agent import GSM8KAgent


@register("gsm8k_agent")
class GSM8KAgentFlow(BlackBoxAgentFlowBase):
    async def _run_agent(self, base_url: str, kwargs: dict[str, Any], sampling_params: dict[str, Any], trajectory_uid: str) -> int:
        del sampling_params, trajectory_uid
        prompt = kwargs.get("raw_prompt", [])
        question = next((m.get("content", "") for m in reversed(prompt) if isinstance(m, dict) and m.get("role") == "user"), str(prompt)) if isinstance(prompt, list) else str(prompt)
        rm = kwargs.get("reward_model", {})
        truth = str(rm.get("ground_truth", "")) if isinstance(rm, dict) else str(getattr(rm, "ground_truth", ""))
        return await GSM8KAgent(base_url).solve(question, truth, int(self.config.actor_rollout_ref.rollout.get("max_turns", 3)))
