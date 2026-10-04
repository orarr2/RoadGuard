"""The decision loop that drives the agent across tool hops.

The loop is provider-agnostic. A GroqProvider hits the hosted
function-calling endpoint; a LocalLlamaProvider runs an on-device gguf
and skips the tool loop (simplified fallback mode). A MockProvider lets
tests drive the loop with pre-scripted messages.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from . import policy, prompts, tools
from .tools import ToolContext, execute_tool_call, tool_schemas_for_groq


LOGGER = logging.getLogger("roadguard.agent")

MAX_TOOL_HOPS = 5
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


@dataclass
class ProviderMessage:
    """A normalized LLM response. One of `content` or `tool_calls` is set."""

    finish_reason: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] | None = None


class LLMProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]] | None,
    ) -> ProviderMessage:
        ...


class MockProvider(LLMProvider):
    """Replay a fixed sequence of ProviderMessages. For tests only."""

    name = "mock"

    def __init__(self, scripted: Iterable[ProviderMessage]) -> None:
        self._queue = list(scripted)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]] | None,
    ) -> ProviderMessage:
        if not self._queue:
            raise RuntimeError("MockProvider: scripted queue is empty.")
        return self._queue.pop(0)


class GroqProvider(LLMProvider):
    """OpenAI-compatible chat/completions client against api.groq.com."""

    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        http_client: Any | None = None,
        timeout_s: float = GROQ_TIMEOUT_S,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.timeout_s = timeout_s
        if http_client is None:
            import httpx

            http_client = httpx.Client(timeout=timeout_s)
        self._http = http_client

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]] | None,
    ) -> ProviderMessage:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        if tools_schema:
            body["tools"] = tools_schema
            body["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = self._http.post(GROQ_URL, json=body, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"groq_http_{resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        tool_calls_raw = msg.get("tool_calls") or []
        parsed_calls: list[dict[str, Any]] = []
        for call in tool_calls_raw:
            fn = call.get("function") or {}
            args_text = fn.get("arguments", "{}")
            try:
                arguments = json.loads(args_text) if isinstance(args_text, str) else args_text
            except json.JSONDecodeError:
                arguments = {"__raw__": args_text}
            parsed_calls.append(
                {
                    "id": call.get("id"),
                    "name": fn.get("name", ""),
                    "arguments": arguments,
                }
            )
        return ProviderMessage(
            finish_reason=choice.get("finish_reason", "stop"),
            content=msg.get("content"),
            tool_calls=parsed_calls,
            raw=data,
        )


class LocalLlamaProvider(LLMProvider):
    """llama-cpp-python backed provider for simplified fallback mode only."""

    name = "local_llama"

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 1024,
        n_threads: int = 2,
        max_tokens: int = 180,
    ) -> None:
        self.model_path = model_path or os.environ.get("ROADGUARD_LOCAL_LLM_MODEL", "")
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self._llm: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        if not self.model_path or not os.path.exists(self.model_path):
            raise RuntimeError(
                f"local_llm_model_missing: {self.model_path or '<unset>'}"
            )
        from llama_cpp import Llama  # lazy; only required when the fallback fires

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]] | None,
    ) -> ProviderMessage:
        self._ensure_loaded()
        assert self._llm is not None
        resp = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        choice = resp["choices"][0]
        msg = choice["message"]
        return ProviderMessage(
            finish_reason=choice.get("finish_reason", "stop"),
            content=msg.get("content"),
            tool_calls=[],
            raw=resp,
        )


# ---------------------------------------------------------------------------
# Decision loop
# ---------------------------------------------------------------------------


@dataclass
class AgentDecision:
    action: str
    rationale: str
    priority: int
    cited_sources: list[str]
    provider: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    fallback_reason: str | None = None


def _attach(trace: list[dict[str, Any]], hop: int, **kwargs: Any) -> None:
    entry = {"hop": hop, "ts": round(time.time(), 3), **kwargs}
    trace.append(entry)


def _collect_cited_urls(trace: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for entry in trace:
        for result in entry.get("tool_results", []) or []:
            passages = (result.get("result") or {}).get("passages") or []
            for p in passages:
                url = p.get("source_url")
                if url and url not in urls:
                    urls.append(url)
    return urls


def _parse_final_verdict(content: str | None) -> dict[str, Any] | None:
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _severity_from_hazard(hazard: dict[str, Any]) -> dict[str, Any]:
    return tools.compute_severity(
        hazard_class=hazard["class"],
        confidence=float(hazard["confidence"]),
        road_type=hazard["road_type"],
        lane_blocked=hazard["lane_blocked"],
    )


def _human_review_from_fallback(
    hazard: dict[str, Any],
    fallback: policy.FallbackVerdict,
    provider: str,
    trace: list[dict[str, Any]],
) -> AgentDecision:
    verdict = policy.final_human_review(fallback.reason, fallback.rationale)
    return AgentDecision(
        action="human_review",
        rationale=verdict["rationale"],
        priority=int(verdict["priority"]),
        cited_sources=[],
        provider=provider,
        trace=trace,
        fallback_reason=fallback.reason,
    )


def agent_decide(
    hazard: dict[str, Any],
    provider: LLMProvider,
    ctx: ToolContext,
    max_hops: int = MAX_TOOL_HOPS,
    fallback_provider: LLMProvider | None = None,
) -> AgentDecision:
    """Run the full agent loop and return a validated AgentDecision.

    - `provider`: the primary tool-using provider (Groq in production).
    - `fallback_provider`: optional simplified-mode provider (local Llama).
      Fired when the primary raises or returns an unparseable final.
    """
    trace: list[dict[str, Any]] = []
    _attach(trace, hop=-1, phase="input", hazard=hazard)

    early = policy.check_input_fallbacks(hazard)
    if early is not None:
        return _human_review_from_fallback(hazard, early, provider.name, trace)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.build_user_message(hazard)},
    ]
    tools_schema = tool_schemas_for_groq()

    try:
        verdict = _run_tool_loop(
            messages=messages,
            tools_schema=tools_schema,
            provider=provider,
            ctx=ctx,
            trace=trace,
            max_hops=max_hops,
        )
    except Exception as exc:
        LOGGER.warning("primary_provider_failed: %s", exc)
        _attach(trace, hop=99, phase="primary_error", error=str(exc))
        if fallback_provider is None:
            return _human_review_from_fallback(
                hazard,
                policy.FallbackVerdict(
                    reason="primary_llm_unavailable",
                    rationale=f"Primary LLM failed and no fallback is configured: {exc}.",
                ),
                provider.name,
                trace,
            )
        return _run_simplified_fallback(hazard, fallback_provider, ctx, trace)

    if verdict is None:
        if fallback_provider is None:
            return _human_review_from_fallback(
                hazard,
                policy.FallbackVerdict(
                    reason="unparseable_primary",
                    rationale="Primary provider produced no parseable verdict and no fallback is configured.",
                ),
                provider.name,
                trace,
            )
        return _run_simplified_fallback(hazard, fallback_provider, ctx, trace)

    return verdict


def _run_tool_loop(
    messages: list[dict[str, Any]],
    tools_schema: list[dict[str, Any]],
    provider: LLMProvider,
    ctx: ToolContext,
    trace: list[dict[str, Any]],
    max_hops: int,
) -> AgentDecision | None:
    """Returns a validated AgentDecision, or None if parsing failed."""
    cited_sources: list[str] = []
    severity: dict[str, Any] | None = None
    has_rag = False

    for hop in range(max_hops):
        resp = provider.complete(messages, tools_schema=tools_schema)
        _attach(
            trace,
            hop=hop,
            phase="llm",
            finish_reason=resp.finish_reason,
            tool_calls=resp.tool_calls,
            content_preview=(resp.content or "")[:400],
        )

        if resp.finish_reason == "tool_calls" and resp.tool_calls:
            tool_results = []
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["arguments"]),
                            },
                        }
                        for c in resp.tool_calls
                    ],
                }
            )
            for call in resp.tool_calls:
                result = execute_tool_call(call["name"], call["arguments"], ctx)
                tool_results.append({"name": call["name"], "result": result})
                if call["name"] == "check_hazard_severity":
                    severity = result
                if call["name"] == "query_repair_standards":
                    passages = (result or {}).get("passages") or []
                    has_rag = has_rag or bool(passages)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            _attach(trace, hop=hop, phase="tools", tool_results=tool_results)
            continue

        parsed = _parse_final_verdict(resp.content)
        if parsed is None:
            _attach(trace, hop=hop, phase="parse_failed")
            return None
        action = parsed.get("action")
        if not policy.validate_action(action or ""):
            _attach(trace, hop=hop, phase="invalid_action", action=action)
            return None

        cited_sources = parsed.get("cited_sources") or []
        if not isinstance(cited_sources, list):
            cited_sources = []
        all_urls = _collect_cited_urls(trace)
        if cited_sources:
            cited_sources = [u for u in cited_sources if u in all_urls] or cited_sources

        if action != "human_review" and not cited_sources:
            _attach(trace, hop=hop, phase="ungrounded_non_review")
            return AgentDecision(
                action="human_review",
                rationale="Verdict was not grounded in any retrieved source.",
                priority=0,
                cited_sources=[],
                provider=provider.name,
                trace=trace,
                fallback_reason="ungrounded",
            )

        if severity is None:
            try:
                severity = _severity_from_hazard(trace[0]["hazard"])
            except Exception:
                severity = {"priority": int(parsed.get("priority", 0)), "band": "low"}

        _attach(trace, hop=hop, phase="verdict", parsed=parsed)
        return AgentDecision(
            action=action,
            rationale=parsed.get("rationale", "").strip() or "No rationale provided.",
            priority=int(parsed.get("priority", severity.get("priority", 0))),
            cited_sources=cited_sources,
            provider=provider.name,
            trace=trace,
        )

    _attach(trace, hop=max_hops, phase="hop_budget_exceeded")
    return AgentDecision(
        action="human_review",
        rationale="Tool-hop budget exceeded without a final verdict.",
        priority=0,
        cited_sources=[],
        provider=provider.name,
        trace=trace,
        fallback_reason="hop_budget_exceeded",
    )


def _run_simplified_fallback(
    hazard: dict[str, Any],
    provider: LLMProvider,
    ctx: ToolContext,
    trace: list[dict[str, Any]],
) -> AgentDecision:
    severity = _severity_from_hazard(hazard)
    procedures: list[dict[str, Any]] = []
    if ctx.rag is not None:
        try:
            procedures = ctx.rag.query(
                hazard_class=hazard["class"],
                road_type=hazard["road_type"],
                k=3,
            )
        except Exception as exc:
            _attach(trace, hop=100, phase="fallback_rag_error", error=str(exc))
    _attach(
        trace,
        hop=100,
        phase="fallback_input",
        severity=severity,
        procedures=procedures,
    )

    rag_fallback = policy.check_rag_fallback(procedures)
    if rag_fallback is not None:
        return _human_review_from_fallback(hazard, rag_fallback, provider.name, trace)

    messages = [
        {"role": "system", "content": prompts.SIMPLIFIED_FALLBACK_PROMPT},
        {
            "role": "user",
            "content": prompts.build_simplified_user_message(
                hazard, severity, procedures
            ),
        },
    ]
    try:
        resp = provider.complete(messages, tools_schema=None)
    except Exception as exc:
        _attach(trace, hop=101, phase="fallback_error", error=str(exc))
        return _human_review_from_fallback(
            hazard,
            policy.FallbackVerdict(
                reason="fallback_llm_unavailable",
                rationale=f"Local Llama failed: {exc}.",
            ),
            provider.name,
            trace,
        )

    parsed = _parse_final_verdict(resp.content)
    if parsed is None or not policy.validate_action(parsed.get("action") or ""):
        _attach(trace, hop=101, phase="fallback_parse_failed", raw=resp.content)
        return _human_review_from_fallback(
            hazard,
            policy.FallbackVerdict(
                reason="fallback_unparseable",
                rationale="Local Llama returned no parseable verdict.",
            ),
            provider.name,
            trace,
        )

    cited_sources = parsed.get("cited_sources") or []
    if not isinstance(cited_sources, list):
        cited_sources = []
    _attach(trace, hop=101, phase="fallback_verdict", parsed=parsed)
    return AgentDecision(
        action=parsed["action"],
        rationale=parsed.get("rationale", "").strip() or "Simplified fallback verdict.",
        priority=int(parsed.get("priority", severity["priority"])),
        cited_sources=cited_sources,
        provider=provider.name,
        trace=trace,
        fallback_reason="simplified_mode",
    )
