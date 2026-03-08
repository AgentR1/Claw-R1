"""GSM8K black-box agent — fully independent of training internals.

This agent uses a standard OpenAI-compatible API to interact with the LLM,
parses tool calls from raw text output (Qwen-style ``<tool_call>`` tags),
and executes a local ``check_answer`` tool.

It knows nothing about trajectory UIDs, Steps, DataPool, or reward — all of
those are transparently handled by the Gateway.
"""

import json
import logging

import httpx
import regex

logger = logging.getLogger(__name__)

CHECK_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "check_answer",
        "description": "Check if your answer to the math problem is correct.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your final numerical answer",
                }
            },
            "required": ["answer"],
        },
    },
}

TOOL_CALL_REGEX = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)


def parse_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Extract ``<tool_call>`` blocks from raw LLM output.

    Mirrors the parsing logic of verl's ``HermesToolParser``.

    Returns:
        (remaining_text, list_of_tool_calls) where each tool call is a dict
        with ``name`` and ``arguments`` keys.
    """
    if "<tool_call>" not in content:
        return content, []

    matches = TOOL_CALL_REGEX.findall(content)
    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match)
            if not isinstance(parsed, dict):
                continue
            tool_calls.append({"name": parsed["name"], "arguments": parsed["arguments"]})
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    remaining = TOOL_CALL_REGEX.sub("", content).strip()
    return remaining, tool_calls


def check_answer(answer: str, ground_truth: str) -> str:
    """Run local answer verification, returning textual feedback only."""
    from verl.utils.reward_score.gsm8k import compute_score

    score = compute_score(
        f"#### {answer}",
        ground_truth,
        method="flexible",
        format_score=0.0,
        score=1.0,
    )
    if score > 0:
        return "Correct! Your answer is right."
    return "Incorrect. Your answer is wrong, please try again."


class GSM8KAgent:
    """Stateless GSM8K solving agent that talks to an OpenAI-compatible API.

    The agent is completely unaware of training-side concepts such as
    ``trajectory_uid``, ``Step``, or ``DataPool``.  All it needs is a
    ``base_url`` pointing to an OpenAI-compatible endpoint.

    Args:
        base_url: Root URL for the API, e.g. ``http://host:port/{traj}/{prompt}``.
            The OpenAI SDK client will use ``{base_url}/v1`` as its base,
            while trajectory completion hits ``{base_url}/v1/trajectory/complete``.
    """

    def __init__(self, base_url: str):
        import openai

        self.base_url = base_url.rstrip("/")
        self.client = openai.AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="not-needed",
            timeout=600.0,
        )

    async def solve(self, question: str, ground_truth: str, max_turns: int = 3) -> int:
        """Attempt to solve *question* in up to *max_turns* LLM interactions.

        After all turns (or an early stop), the agent signals trajectory
        completion via ``POST {base_url}/v1/trajectory/complete``.

        Returns the number of turns actually used.
        """
        messages: list[dict] = [{"role": "user", "content": question}]

        turns_used = 0
        for turn in range(max_turns):
            turns_used = turn + 1

            resp = await self.client.chat.completions.create(
                model="default",
                messages=messages,
                tools=[CHECK_ANSWER_TOOL],
            )
            content = resp.choices[0].message.content or ""
            _, tool_calls = parse_tool_calls(content)

            if tool_calls:
                messages.append({"role": "assistant", "content": content})
                for tc in tool_calls:
                    if tc["name"] == "check_answer":
                        answer = tc["arguments"].get("answer", "")
                        result = check_answer(answer, ground_truth)
                        messages.append({"role": "tool", "content": result})
            else:
                messages.append({"role": "assistant", "content": content})
                break

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as http:
            await http.post(f"{self.base_url}/v1/trajectory/complete")

        return turns_used
