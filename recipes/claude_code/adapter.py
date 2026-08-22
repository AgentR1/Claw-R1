"""Claude Code's Anthropic-compatible CLI adapter."""
from __future__ import annotations

import asyncio, os, shutil
from pathlib import Path
from typing import Any

from claw_r1.blackbox_runtime.adapter import BlackBoxAdapter


class ClaudeCodeAdapter(BlackBoxAdapter):
    async def run(self, state: dict[str, Any], messages: list[dict[str, Any]], turn_id: str) -> dict[str, Any]:
        executable = os.getenv("CLAUDE_CODE_BIN", "claude")
        if shutil.which(executable) is None: raise RuntimeError(f"Claude Code executable not found: {executable}")
        runtime = Path(state["workspace"]) / ".claw-r1" / "claude_code"; home = runtime / "home"; home.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ); env.update({"HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"), "ANTHROPIC_BASE_URL": state["proxy_base"], "ANTHROPIC_AUTH_TOKEN": "claw-r1-local", "ANTHROPIC_CUSTOM_MODEL_OPTION": "proxy-model"})
        for key in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"): env.pop(key, None)
        prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        opts = state["options"]; args = [executable, "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", "proxy-model", "--max-turns", str(opts.get("max_turns", 20)), "--permission-mode", str(opts.get("permission_mode", "bypassPermissions"))]
        return await _run(args, state, env, turn_id)


async def _run(args, state, env, turn_id):
    proc = await asyncio.create_subprocess_exec(*args, cwd=state["workspace"], env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(state["options"].get("timeout", 1800)))
    if proc.returncode: raise RuntimeError(stderr.decode(errors="replace")[-4000:])
    return {"turn_id": turn_id, "status": "completed", "response": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}
