# Black-box agents and recipes

Claw-R1 records every model request through Gateway and writes the resulting
token-level `Step` objects into DataPool. Recipes select an agent integration,
not a dataset:

```text
recipes/gsm8k_agent     OpenAI-compatible example
recipes/claude_code     Claude Code CLI in an isolated sandbox
recipes/codex           Codex CLI in an isolated sandbox
recipes/openclaw        OpenClaw CLI in an isolated sandbox
```

For GSM8K:

```bash
actor_rollout_ref.rollout.agent.default_agent_flow=gsm8k_agent
actor_rollout_ref.rollout.agent.agent_flow_config_path=recipes/gsm8k_agent/agent_flow_config.yaml
```

CLI recipes use `CLAW_R1_SANDBOX_PROVIDER=local_bwrap` or `e2b`. Samples may
provide `files`, `before_agent_commands`, `after_agent_commands`,
`grader_command`, and `backend_options`. The grader prints JSON with an
`overall_score` field. See [`recipes/README.md`](../../recipes/README.md) for
the metadata contract.
