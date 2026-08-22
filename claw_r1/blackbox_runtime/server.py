"""HTTP control plane for one isolated CLI agent backend.

The process lives inside a sandbox.  Its local proxy translates each CLI's
model protocol to OpenAI Chat Completions and forwards it to the trajectory
specific Claw-R1 Gateway URL supplied at registration time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .protocol import chat_to_responses, messages_from_anthropic, messages_from_responses

app = FastAPI(title="Claw-R1 Blackbox Runtime")
_state: dict[str, Any] = {"sessions": {}, "workspace": None}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/rollout/register")
async def register(body: dict[str, Any], request: Request) -> dict[str, Any]:
    session_id = str(body["session_id"])
    gateway_url = str(body["gateway_url"]).rstrip("/")
    adapter_path = str(body["adapter"])
    options = body.get("backend_options") or {}
    workspace = Path(str(body.get("workspace") or _state["workspace"])).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    proxy_base = str(request.base_url).rstrip("/") + f"/v1/proxy/{session_id}"
    _state["sessions"][session_id] = {"gateway_url": gateway_url, "adapter": _load_adapter(adapter_path), "options": options, "workspace": workspace, "proxy_base": proxy_base, "turns": {}}
    return {"status": "registered", "session_id": session_id}


@app.post("/v1/sessions/{session_id}/files")
async def write_files(session_id: str, body: dict[str, Any]) -> dict[str, int]:
    state = _session(session_id)
    written = 0
    for item in body.get("files", []):
        rel = Path(str(item["path"]))
        if rel.is_absolute() or ".." in rel.parts: raise HTTPException(400, "file path escapes workspace")
        target = state["workspace"] / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(item.get("content", "")), encoding="utf-8")
        written += 1
    return {"written": written}


@app.post("/v1/sessions/{session_id}/messages")
async def messages(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
    state = _session(session_id)
    turn_id = str(body.get("turn_id") or len(state["turns"]))
    canonical = json.dumps(body.get("messages", []), sort_keys=True, ensure_ascii=False)
    if turn_id in state["turns"]:
        old, result = state["turns"][turn_id]
        if old != canonical: raise HTTPException(409, "turn_id reused with different messages")
        return result
    result = await _run_backend(state, body.get("messages", []), turn_id)
    state["turns"][turn_id] = (canonical, result)
    return result


@app.post("/v1/sessions/{session_id}/execute_cmd")
async def execute_cmd(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
    state = _session(session_id)
    proc = await asyncio.create_subprocess_shell(str(body["cmd"]), cwd=state["workspace"], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(body.get("timeout", 600)))
    except asyncio.TimeoutError:
        proc.kill(); await proc.wait(); raise HTTPException(408, "command timed out")
    return {"returncode": proc.returncode, "stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}


@app.post("/v1/sessions/{session_id}/abort")
async def abort(session_id: str) -> dict[str, str]:
    _state["sessions"].pop(session_id, None)
    return {"status": "aborted"}


@app.api_route("/v1/proxy/{session_id}/{path:path}", methods=["POST"])
async def model_proxy(session_id: str, path: str, request: Request):
    """Protocol bridge used by Codex and Claude Code local configuration."""
    state = _session(session_id)
    payload = await request.json()
    if path.endswith("responses"):
        body = {"model": payload.get("model", "default"), "messages": messages_from_responses(payload), "tools": payload.get("tools")}
        response_kind = "responses"
    elif path.endswith("messages"):
        body = {"model": payload.get("model", "default"), "messages": messages_from_anthropic(payload), "tools": payload.get("tools")}
        response_kind = "anthropic"
    else:
        body, response_kind = payload, "chat"
    result = await _gateway_chat(state, body)
    if response_kind == "responses": return JSONResponse(chat_to_responses(result["choices"][0], body.get("model", "default")))
    if response_kind == "anthropic":
        msg = result["choices"][0]["message"]
        return JSONResponse({"id": "msg_claw_r1", "type": "message", "role": "assistant", "stop_reason": "end_turn", "content": [{"type": "text", "text": msg.get("content", "")} ]})
    return JSONResponse(result)


async def _run_backend(state: dict[str, Any], messages_: list[dict[str, Any]], turn_id: str) -> dict[str, Any]:
    return await state["adapter"].run(state, messages_, turn_id)


def _load_adapter(path: str):
    from importlib import import_module
    from .adapter import BlackBoxAdapter
    module_name, _, symbol = path.partition(":")
    if not module_name or not symbol: raise HTTPException(400, "adapter must be module:Class")
    adapter = getattr(import_module(module_name), symbol)()
    if not isinstance(adapter, BlackBoxAdapter): raise HTTPException(400, "adapter must implement BlackBoxAdapter")
    return adapter


def _session(session_id: str) -> dict[str, Any]:
    if session_id not in _state["sessions"]: raise HTTPException(404, "unknown session")
    return _state["sessions"][session_id]


async def _gateway_chat(state: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(f"{state['gateway_url']}/chat/completions", json=body)
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, required=True); parser.add_argument("--workspace", required=True)
    args = parser.parse_args(); _state["workspace"] = args.workspace
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__": main()
