from claw_r1.agent_flow.agent_flow import register
from claw_r1.blackbox_runtime.flow import BlackBoxCliAgentFlow

@register("claude_code")
class ClaudeCodeFlow(BlackBoxCliAgentFlow):
    adapter_path = "recipes.claude_code.adapter:ClaudeCodeAdapter"
