"""Codex Responses-API adapter with per-rollout Codex state."""
from __future__ import annotations

import asyncio, os, shutil
from pathlib import Path
from typing import Any

from claw_r1.blackbox_runtime.adapter import BlackBoxAdapter


class CodexAdapter(BlackBoxAdapter):
    async def run(self, state: dict[str, Any], messages: list[dict[str, Any]], turn_id: str) -> dict[str, Any]:
        executable = os.getenv("CODEX_BIN", "codex")
        if shutil.which(executable) is None: raise RuntimeError(f"Codex executable not found: {executable}")
        runtime = Path(state["workspace"]) / ".claw-r1" / "codex"; home = runtime / "home"; codex_home = home / ".codex"; codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "config.toml").write_text("model = \"proxy-model\"\nmodel_provider = \"claw_r1\"\napproval_policy = \"never\"\n[model_providers.claw_r1]\nname = \"Claw-R1 Gateway\"\nbase_url = \"" + state["proxy_base"] + "\"\nwire_api = \"responses\"\n", encoding="utf-8")
        env = dict(os.environ); env.update({"HOME": str(home), "CODEX_HOME": str(codex_home), "CODEX_SQLITE_HOME": str(home / ".codex-sqlite")})
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"): env.pop(key, None)
        prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        opts = state["options"]; args = [executable, "exec", "--json", "--model", "proxy-model", "--skip-git-repo-check", "--sandbox", str(opts.get("sandbox_mode", "danger-full-access")), prompt]
        proc = await asyncio.create_subprocess_exec(*args, cwd=state["workspace"], env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(opts.get("timeout", 1800)))
        if proc.returncode: raise RuntimeError(stderr.decode(errors="replace")[-4000:])
        return {"turn_id": turn_id, "status": "completed", "response": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}
