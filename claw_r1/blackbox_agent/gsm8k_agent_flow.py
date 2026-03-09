"""GSM8K black-box agent flow — concrete strategy for GSM8K."""

from typing import Any

from claw_r1.agent_flow.agent_flow import register

from claw_r1.blackbox_agent.blackbox_agent_flow import BlackBoxAgentFlowBase

from claw_r1.blackbox_agent.gsm8k_agent import GSM8KAgent


@register("blackbox_gsm8k_agent")
class BlackBoxGSM8KAgentFlow(BlackBoxAgentFlowBase):
    """Black-box flow that delegates to :class:`GSM8KAgent`."""

    async def _run_agent(self, base_url: str, kwargs: dict[str, Any]) -> int:
        raw_prompt = kwargs.get("raw_prompt", [])
        if isinstance(raw_prompt, list) and raw_prompt:
            question = next(
                (m.get("content", "") for m in reversed(raw_prompt) if m.get("role") == "user"),
                str(raw_prompt),
            ) or str(raw_prompt)
        elif isinstance(raw_prompt, str):
            question = raw_prompt
        else:
            question = str(raw_prompt)

        reward_model = kwargs.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = str(reward_model.get("ground_truth", ""))
        else:
            ground_truth = str(getattr(reward_model, "ground_truth", ""))

        max_turns = self.config.actor_rollout_ref.rollout.get("max_turns", 3)
        agent = GSM8KAgent(base_url=base_url)
        return await agent.solve(question=question, ground_truth=ground_truth, max_turns=max_turns)
