"""Protocol normalisation shared by Gateway and CLI black-box adapters."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse the Qwen/Hermes tool-call envelope into OpenAI tool calls.

    Unparseable blocks deliberately remain in the assistant content; losing
    generated tokens would make the stored trajectory differ from the model
    output used by the agent.
    """
    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL.findall(content or ""):
        try:
            call = json.loads(raw)
            name, arguments = call["name"], call["arguments"]
            if not isinstance(name, str):
                continue
            calls.append(
                {
                    "id": f"call_{uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments) if not isinstance(arguments, str) else arguments,
                    },
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return _TOOL_CALL.sub("", content).strip() if calls else content, calls


def messages_from_responses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Losslessly reduce the Responses API input shape to chat messages."""
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": str(instructions)})
    source = payload.get("input", [])
    if isinstance(source, str):
        return messages + [{"role": "user", "content": source}]
    for item in source if isinstance(source, list) else []:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif isinstance(item, dict) and item.get("type") == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""), "content": str(item.get("output", ""))})
        elif isinstance(item, dict):
            messages.append({"role": item.get("role", "user"), "content": _content_text(item.get("content", ""))})
    return messages


def messages_from_anthropic(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if system:
        messages.append({"role": "system", "content": _content_text(system)})
    for item in payload.get("messages", []):
        if isinstance(item, dict):
            messages.append({"role": item.get("role", "user"), "content": _content_text(item.get("content", ""))})
    return messages


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in value)
    return str(value)


def chat_to_responses(choice: dict[str, Any], model: str) -> dict[str, Any]:
    message = choice.get("message", {})
    output: list[dict[str, Any]] = []
    if message.get("content"):
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": message["content"]}]})
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        output.append({"type": "function_call", "call_id": call.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments", "{}")})
    return {"id": f"resp_{uuid4().hex}", "object": "response", "model": model, "status": "completed", "output": output}
