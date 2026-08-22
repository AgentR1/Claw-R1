# OpenClaw

Use `default_agent_flow=openclaw`. Set `OPENCLAW_BIN` when necessary. Inject
task services, MCP configuration, workspace files, and a system-prompt setup
through `before_agent_commands` and `files`; the recipe deliberately has no
dataset dependency.
