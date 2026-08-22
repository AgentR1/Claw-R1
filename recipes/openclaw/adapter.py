"""OpenClaw CLI adapter with isolated workspace and MCP configuration."""
from __future__ import annotations

import asyncio, os, shutil
from pathlib import Path
from typing import Any

from claw_r1.blackbox_runtime.adapter import BlackBoxAdapter


class OpenClawAdapter(BlackBoxAdapter):
    async def run(self, state: dict[str, Any], messages: list[dict[str, Any]], turn_id: str) -> dict[str, Any]:
        executable = os.getenv("OPENCLAW_BIN", "openclaw")
        if shutil.which(executable) is None: raise RuntimeError(f"OpenClaw executable not found: {executable}")
        runtime = Path(state["workspace"]) / ".claw-r1" / "openclaw"; home = runtime / "home"; home.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ); env.update({"HOME": str(home), "OPENAI_BASE_URL": state["proxy_base"], "OPENAI_API_KEY": "claw-r1-local"})
        # MCP/task services are injected by the dataset through files and
        # before_agent_commands, keeping this recipe dataset independent.
        prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        args = [executable, "agent", "--prompt", prompt]
        proc = await asyncio.create_subprocess_exec(*args, cwd=state["workspace"], env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(state["options"].get("timeout", 1800)))
        if proc.returncode: raise RuntimeError(stderr.decode(errors="replace")[-4000:])
        return {"turn_id": turn_id, "status": "completed", "response": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}
