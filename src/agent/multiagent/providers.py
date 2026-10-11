"""Multi-provider chat backends for the agent layer.

The agent nodes call `chat()` from `.chat`, which routes through here.
Switching provider is an env var change, not a code edit, so the same
code runs against Groq, Anthropic, Ollama (local), or any
OpenAI-compatible endpoint (LM Studio, vLLM, llamafile, Together, ...).

Models and API keys stay in env vars. Nothing is persisted to disk by
this module.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Normalized reply shape
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """Provider-agnostic reply used by every node."""

    finish_reason: str
    content: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer for the offline stub and for providers that do not
    return a usage object."""
    return max(1, math.ceil(len(str(text)) / 4))


# ---------------------------------------------------------------------------
# Groq (OpenAI-compatible via the groq SDK)
# ---------------------------------------------------------------------------


def _chat_groq(
    messages: list[dict],
    tools: Optional[list],
    temperature: float,
    model: str,
) -> tuple[ChatMessage, int, int]:
    from groq import Groq  # lazy import; installed on the VM only

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("groq provider: GROQ_API_KEY not set")
    client = Groq(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    m = choice.message
    raw = getattr(m, "tool_calls", None) or []
    tool_calls = [
        {
            "id": tc.id,
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in raw
    ]
    return (
        ChatMessage(
            finish_reason=choice.finish_reason or "stop",
            content=m.content,
            tool_calls=tool_calls,
        ),
        getattr(resp.usage, "prompt_tokens", 0) or 0,
        getattr(resp.usage, "completion_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Anthropic (claude)
# ---------------------------------------------------------------------------


def _to_anthropic_tools(tools: Optional[list]) -> list[dict]:
    """Translate OpenAI function-tool schemas to Anthropic's shape."""
    if not tools:
        return []
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _split_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic separates the system prompt from the user/assistant turns."""
    system_parts = []
    turns = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else json.dumps(content))
            continue
        if role == "tool":
            turns.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id") or m.get("id") or "unknown",
                            "content": content if isinstance(content, str) else json.dumps(content),
                        }
                    ],
                }
            )
            continue
        if role == "assistant" and m.get("tool_calls"):
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for call in m["tool_calls"]:
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"__raw__": args}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or "call",
                        "name": fn.get("name", ""),
                        "input": args or {},
                    }
                )
            turns.append({"role": "assistant", "content": blocks})
            continue
        turns.append(
            {
                "role": role or "user",
                "content": content if isinstance(content, str) else json.dumps(content),
            }
        )
    return "\n\n".join(system_parts), turns


def _chat_anthropic(
    messages: list[dict],
    tools: Optional[list],
    temperature: float,
    model: str,
) -> tuple[ChatMessage, int, int]:
    try:
        from anthropic import Anthropic  # lazy
    except ImportError as exc:
        raise RuntimeError(
            "anthropic provider: `pip install anthropic` and set ANTHROPIC_API_KEY"
        ) from exc
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("anthropic provider: ANTHROPIC_API_KEY not set")
    system, turns = _split_anthropic_messages(messages)
    client = Anthropic(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "temperature": temperature,
        "messages": turns,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = _to_anthropic_tools(tools)
    resp = client.messages.create(**kwargs)
    content_text = ""
    tool_calls: list[dict] = []
    for block in resp.content or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            content_text += getattr(block, "text", "")
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(block, "id", ""),
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": json.dumps(getattr(block, "input", {})),
                    },
                }
            )
    finish = "tool_calls" if tool_calls else (resp.stop_reason or "stop")
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) if usage else estimate_tokens(json.dumps(messages))
    out_tok = getattr(usage, "output_tokens", 0) if usage else estimate_tokens(content_text or json.dumps(tool_calls))
    return (
        ChatMessage(finish_reason=finish, content=content_text or None, tool_calls=tool_calls),
        int(in_tok or 0),
        int(out_tok or 0),
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints (LM Studio, vLLM, Together, OpenRouter, ...)
# ---------------------------------------------------------------------------


def _chat_openai_compat(
    messages: list[dict],
    tools: Optional[list],
    temperature: float,
    model: str,
) -> tuple[ChatMessage, int, int]:
    try:
        from openai import OpenAI  # lazy
    except ImportError as exc:
        raise RuntimeError(
            "openai_compat provider: `pip install openai` first"
        ) from exc
    base_url = os.environ.get("OPENAI_COMPAT_BASE_URL") or "http://localhost:1234/v1"
    api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "not-needed")
    client = OpenAI(base_url=base_url, api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    m = choice.message
    raw = getattr(m, "tool_calls", None) or []
    tool_calls = [
        {
            "id": getattr(tc, "id", ""),
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in raw
    ]
    usage = getattr(resp, "usage", None)
    return (
        ChatMessage(
            finish_reason=choice.finish_reason or "stop",
            content=m.content,
            tool_calls=tool_calls,
        ),
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Ollama (local, free)
# ---------------------------------------------------------------------------


def _chat_ollama(
    messages: list[dict],
    tools: Optional[list],
    temperature: float,
    model: str,
) -> tuple[ChatMessage, int, int]:
    try:
        import httpx  # already in requirements
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ollama provider: httpx required") from exc
    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        body["tools"] = tools
    resp = httpx.post(f"{base_url.rstrip('/')}/api/chat", json=body, timeout=120.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"ollama_http_{resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    msg = data.get("message", {}) or {}
    content = msg.get("content") or ""
    raw_calls = msg.get("tool_calls") or []
    tool_calls = []
    for i, tc in enumerate(raw_calls):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args or {})
        tool_calls.append(
            {
                "id": tc.get("id") or f"ollama_{i}",
                "function": {"name": fn.get("name", ""), "arguments": args},
            }
        )
    finish = "tool_calls" if tool_calls else "stop"
    in_tok = int(data.get("prompt_eval_count") or estimate_tokens(json.dumps(messages)))
    out_tok = int(data.get("eval_count") or estimate_tokens(content or json.dumps(tool_calls)))
    return (
        ChatMessage(finish_reason=finish, content=content or None, tool_calls=tool_calls),
        in_tok,
        out_tok,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


ProviderFn = Callable[
    [list[dict], Optional[list], float, str],
    tuple[ChatMessage, int, int],
]

PROVIDERS: dict[str, ProviderFn] = {
    "groq": _chat_groq,
    "anthropic": _chat_anthropic,
    "openai_compat": _chat_openai_compat,
    "ollama": _chat_ollama,
}

DEFAULT_MODEL_BY_PROVIDER = {
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-opus-4-7",
    "openai_compat": "llama-3.3-70b-versatile",
    "ollama": "llama3.2",
}


def resolve_provider_model(provider: Optional[str] = None) -> tuple[str, str]:
    """Pick provider + model from env, with sensible defaults.

    Priority:
      1. explicit `provider` arg or `provider:model` spec
      2. $AGENT_LLM_PROVIDER / $AGENT_LLM_MODEL
      3. historical $GROQ_MODEL for backward compat with Phase-1 code
      4. defaults in DEFAULT_MODEL_BY_PROVIDER
    """
    spec = provider or os.environ.get("AGENT_LLM_PROVIDER") or "groq"
    if ":" in spec:
        name, model = spec.split(":", 1)
    else:
        name = spec
        model = (
            os.environ.get("AGENT_LLM_MODEL")
            or (os.environ.get("GROQ_MODEL") if spec == "groq" else "")
            or DEFAULT_MODEL_BY_PROVIDER.get(name, "")
        )
    if name not in PROVIDERS:
        raise RuntimeError(
            f"unknown provider: {name!r} (known: {sorted(PROVIDERS)})"
        )
    if not model:
        raise RuntimeError(f"no model for provider {name!r}")
    return name, model


def call_provider(
    spec: str,
    messages: list[dict],
    tools: Optional[list],
    temperature: float,
) -> tuple[ChatMessage, int, int, str, str]:
    """Resolve + call a single provider; return (msg, in, out, name, model)."""
    name, model = resolve_provider_model(spec)
    msg, in_tok, out_tok = PROVIDERS[name](messages, tools, temperature, model)
    return msg, in_tok, out_tok, name, model
