from claw_r1.agent_flow.agent_flow import register
from claw_r1.blackbox_runtime.flow import BlackBoxCliAgentFlow

@register("openclaw")
class OpenClawFlow(BlackBoxCliAgentFlow):
    adapter_path = "recipes.openclaw.adapter:OpenClawAdapter"
