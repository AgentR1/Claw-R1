"""OpenAI-compatible GSM8K agent recipe."""
import json
import re

from claw_r1.blackbox_runtime.protocol import parse_tool_calls

CHECK_ANSWER_TOOL = {"type": "function", "function": {"name": "check_answer", "description": "Check a numerical answer.", "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}}


def check_answer(answer: str, ground_truth: str) -> str:
    from verl.utils.reward_score.gsm8k import compute_score
    return "Correct! Your answer is right." if compute_score(f"#### {answer}", ground_truth, method="flexible", format_score=0.0, score=1.0) > 0 else "Incorrect. Your answer is wrong, please try again."


class GSM8KAgent:
    def __init__(self, base_url: str):
        import openai
        self.client = openai.AsyncOpenAI(base_url=base_url.rstrip("/"), api_key="not-needed", timeout=600)

    async def solve(self, question: str, ground_truth: str, max_turns: int = 3) -> int:
        messages: list[dict] = [{"role": "user", "content": question}]
        for turn in range(max_turns):
            response = await self.client.chat.completions.create(model="default", messages=messages, tools=[CHECK_ANSWER_TOOL])
            content = response.choices[0].message.content or ""
            _, calls = parse_tool_calls(content)
            messages.append({"role": "assistant", "content": content})
            if not calls: return turn + 1
            for call in calls:
                fn = call["function"]
                if fn["name"] == "check_answer":
                    try: answer = json.loads(fn["arguments"]).get("answer", "")
                    except json.JSONDecodeError: answer = ""
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": check_answer(str(answer), ground_truth)})
        return max_turns
