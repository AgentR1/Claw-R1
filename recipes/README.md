# Agent recipes

Recipes are agent integrations, not datasets. Point
`actor_rollout_ref.rollout.agent.agent_flow_config_path` at the recipe's
`agent_flow_config.yaml` and provide its metadata contract in the RL dataset.

| Recipe | Flow | Required metadata |
| --- | --- | --- |
| `gsm8k_agent` | `gsm8k_agent` | `reward_model.ground_truth` |
| `alfworld_agent` | `alfworld_agent` | `game_file` |
| `claude_code` | `claude_code` | optional `files`, commands, `backend_options` |
| `codex` | `codex` | optional `workspace`, `files`, commands, `backend_options` |
| `openclaw` | `openclaw` | optional `files`, MCP/task-service commands, `backend_options` |

All CLI recipes accept `before_agent_commands`, `after_agent_commands`, and a
`grader_command` that prints `{"overall_score": <float>}`. Select
`CLAW_R1_SANDBOX_PROVIDER=local_bwrap` or `e2b`.
