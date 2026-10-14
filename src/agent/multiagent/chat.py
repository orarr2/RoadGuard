"""Single chat wrapper that hides the provider and runs an offline stub.

Per DS23 S42 slide 8: all agent nodes call chat() rather than the raw
provider SDK. In class USE_LIVE=True hits a live provider; in CI and
eval the stub plays back a scripted policy so results are
reproducible and free.

Provider selection is env-driven and multi-vendor (Groq by default,
Anthropic, any OpenAI-compatible endpoint, or Ollama) via
`AGENT_LLM_PROVIDER` and `AGENT_LLM_MODEL`. Two extra modes layer on
top of single-provider mode:

- Committee (`AGENT_LLM_COMMITTEE=1`): two models judge each turn; on
  agreement the higher-confidence verdict wins, on disagreement the
  more severe verdict is used and the message is flagged for human
  review. Doubles the LLM calls, so it is off by default.
- Panel (`AGENT_LLM_PANEL="model_a,model_b,model_c"`): N>=2 models,
  majority vote with severity tiebreak. Each panelist can carry its
  own provider prefix (e.g. "groq:openai/gpt-oss-120b,
  openai_compat:meta-llama/Llama-3.3-70B-Instruct-Turbo,
  openai_compat:deepseek/deepseek-chat"). Panel precedence overrides
  committee when both are set, and panel/committee only fire when the
  caller passes ``use_panel=True`` (Decision node only).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from .guards import PRICE_IN, PRICE_OUT
from .providers import ChatMessage, call_provider, estimate_tokens  # noqa: F401

USE_LIVE = os.environ.get("ROADGUARD_USE_LIVE", "0") == "1"

# Severity ordering used by committee/panel modes to pick a tiebreak when
# verdicts disagree. Higher index = more severe; `urgent` wins over
# `human_review` wins over `needs_more_evidence` wins over `standard`.
ACTION_SEVERITY = {
    "standard": 0,
    "needs_more_evidence": 1,
    "human_review": 2,
    "urgent": 3,
}


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


def _extract_action(msg: ChatMessage) -> str | None:
    """Pull the `action` field out of a JSON verdict if the message carries one."""
    text = (msg.content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    action = obj.get("action")
    return action if isinstance(action, str) else None


def _severity(action: str | None) -> int:
    return ACTION_SEVERITY.get(action or "", -1)


def _attach_committee_note(msg: ChatMessage, note: str) -> ChatMessage:
    """Add a `committee_note` to the JSON verdict (if parseable), otherwise
    leave the message intact. Used so downstream nodes can see that the
    verdict came from a panel/committee that flagged a disagreement.
    """
    text = (msg.content or "").strip()
    if not text or not text.startswith("{"):
        return msg
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return msg
    if not isinstance(obj, dict):
        return msg
    obj["committee_note"] = note
    return ChatMessage(
        finish_reason=msg.finish_reason,
        content=json.dumps(obj),
        tool_calls=list(msg.tool_calls),
    )


def _panel_specs() -> list[str]:
    raw = os.environ.get("AGENT_LLM_PANEL", "").strip()
    if not raw:
        return []
    specs = [s.strip() for s in raw.split(",") if s.strip()]
    # Deduplicate specs; a repeated spec would fake agreement.
    seen: set[str] = set()
    out: list[str] = []
    for s in specs:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _committee_specs() -> list[str] | None:
    if os.environ.get("AGENT_LLM_COMMITTEE", "0") != "1":
        return None
    a = os.environ.get("AGENT_LLM_PROVIDER") or "groq"
    b = os.environ.get("AGENT_LLM_COMMITTEE_MODEL_B") or "llama-3.1-8b-instant"
    # Spec B inherits provider A unless the env already includes a prefix.
    if ":" not in b:
        prov = a.split(":", 1)[0]
        b = f"{prov}:{b}"
    return [a, b]


def _call_specs(
    specs: Iterable[str],
    messages: list[dict],
    tools: list | None,
    temperature: float,
) -> list[tuple[ChatMessage, int, int, str, str]]:
    """Call every spec once. Returns a list of (msg, in_tok, out_tok, provider, model)."""
    out = []
    for spec in specs:
        out.append(call_provider(spec, messages, tools, temperature))
    return out


def _combine(
    results: list[tuple[ChatMessage, int, int, str, str]],
    mode: str,
) -> tuple[ChatMessage, int, int]:
    """Majority-vote or severity-tiebreak over the panel's verdicts.

    Returns (chosen_message, total_input_tokens, total_output_tokens). Any
    disagreement is recorded on the chosen message as `committee_note`.
    """
    assert results, "combine: at least one result required"
    actions = [_extract_action(m) for (m, *_rest) in results]
    tally: dict[str, int] = {}
    for a in actions:
        if a is None:
            continue
        tally[a] = tally.get(a, 0) + 1
    total_in = sum(r[1] for r in results)
    total_out = sum(r[2] for r in results)

    if not tally:
        chosen = results[0][0]
        return (
            _attach_committee_note(chosen, f"{mode}: no parseable action, used panel[0]"),
            total_in,
            total_out,
        )

    max_votes = max(tally.values())
    top_actions = [a for a, n in tally.items() if n == max_votes]
    if len(top_actions) == 1:
        winner_action = top_actions[0]
    else:
        # tiebreak by severity
        winner_action = max(top_actions, key=_severity)
    # pick the message that voted for the winner_action
    for (msg, *_r), act in zip(results, actions):
        if act == winner_action:
            chosen = msg
            break
    else:
        chosen = results[0][0]

    all_agree = len(tally) == 1 and max_votes == len(results)
    labels = [f"{prov}/{model}:{act}" for (_, _, _, prov, model), act in zip(results, actions)]
    note_bits = [mode, "agree" if all_agree else "split"] + labels
    chosen = _attach_committee_note(chosen, " | ".join(note_bits))
    return chosen, total_in, total_out


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    use_panel: bool = False,
) -> tuple[ChatMessage, int, int]:
    """Return (message, tokens_in, tokens_out). Transparent to live/offline/panel.

    ``use_panel`` is Decision-only: when False, panel/committee env vars are
    ignored and a single provider answers. Perception and Research keep the
    cheap single-model path even when AGENT_LLM_PANEL is set, so panel cost
    stays bounded to the one node whose verdict actually ships.
    """
    if not USE_LIVE:
        return _chat_offline(messages, tools)

    if use_panel:
        panel = _panel_specs()
        if panel and len(panel) >= 2:
            results = _call_specs(panel, messages, tools, temperature)
            return _combine(results, mode="panel")

        committee = _committee_specs()
        if committee:
            results = _call_specs(committee, messages, tools, temperature)
            return _combine(results, mode="committee")

    spec = os.environ.get("AGENT_LLM_PROVIDER") or "groq"
    msg, in_tok, out_tok, _p, _m = call_provider(spec, messages, tools, temperature)
    return msg, in_tok, out_tok


def cost_of(tokens_in: int, tokens_out: int) -> float:
    return tokens_in * PRICE_IN + tokens_out * PRICE_OUT
