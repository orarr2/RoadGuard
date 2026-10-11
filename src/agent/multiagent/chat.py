"""Single chat wrapper that hides the provider and runs an offline stub.

Per DS23 S42 slide 8: all agent nodes call chat() rather than the raw
provider SDK. In class USE_LIVE=True hits Groq; in CI and eval the stub
plays back a scripted policy so results are reproducible and free.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from .guards import PRICE_IN, PRICE_OUT

USE_LIVE = os.environ.get("ROADGUARD_USE_LIVE", "0") == "1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


@dataclass
class ChatMessage:
    """A normalized model reply. One of content or tool_calls is set."""

    finish_reason: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Rough estimate used by the offline stub for cost accounting."""
    return max(1, math.ceil(len(str(text)) / 4))


def _chat_live(
    messages: list[dict],
    tools: list | None,
    temperature: float,
) -> tuple[ChatMessage, int, int]:
    """Hits Groq. Only called when USE_LIVE=1."""
    from groq import Groq  # type: ignore

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=tools,
        temperature=temperature,
        tool_choice="auto",
    )
    m = resp.choices[0].message
    raw_tool_calls = getattr(m, "tool_calls", None) or []
    tool_calls = [
        {
            "id": tc.id,
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in raw_tool_calls
    ]
    return (
        ChatMessage(
            finish_reason=resp.choices[0].finish_reason,
            content=m.content,
            tool_calls=tool_calls,
        ),
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


def _chat_offline(
    messages: list[dict],
    tools: list | None,
) -> tuple[ChatMessage, int, int]:
    """Deterministic stub for eval and CI."""
    last = messages[-1] if messages else {}
    text = json.dumps(messages).lower()

    if last.get("role") == "tool":
        msg = ChatMessage(finish_reason="stop", content="offline stub: produced final after tool.")
    elif "decision" in text and "procedures" in text:
        msg = ChatMessage(
            finish_reason="stop",
            content=json.dumps(
                {
                    "action": "standard",
                    "rationale": "offline stub decision",
                    "priority": 40,
                    "cited_sources": [],
                }
            ),
        )
    elif tools:
        first = tools[0]["function"]
        msg = ChatMessage(
            finish_reason="tool_calls",
            content=None,
            tool_calls=[
                {
                    "id": "stub_" + first["name"],
                    "function": {"name": first["name"], "arguments": "{}"},
                }
            ],
        )
    else:
        msg = ChatMessage(finish_reason="stop", content="offline stub: no tools")

    in_tok = estimate_tokens(json.dumps(messages))
    out_tok = estimate_tokens(msg.content or json.dumps(msg.tool_calls))
    return msg, in_tok, out_tok


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
) -> tuple[ChatMessage, int, int]:
    """Return (message, tokens_in, tokens_out). Transparent to live/offline."""
    if USE_LIVE:
        return _chat_live(messages, tools, temperature)
    return _chat_offline(messages, tools)


def cost_of(tokens_in: int, tokens_out: int) -> float:
    return tokens_in * PRICE_IN + tokens_out * PRICE_OUT
