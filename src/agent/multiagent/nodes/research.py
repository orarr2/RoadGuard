"""Research Agent: READ-ONLY RAG lookup.

Reads:  hazard_profile, metadata.road_type
Writes: procedures, tool_result_ids, research_coverage
Tools:  query_repair_standards (read-only)

Includes the dual-LLM pattern (AGENTS/architecture.md section 7.4)
for vector 3 of the prompt-injection surface. The Research agent passes
the raw RAG passage through a quarantine extractor that emits only a
structured JSON summary. The Decision agent never sees the raw passage.

The LLM path (USE_LIVE=1) runs after retrieval to label coverage and
confidence. Procedures themselves are NEVER rewritten by the LLM, only
the coverage summary. On any failure the tool-reported coverage stands.
"""

from __future__ import annotations

import json
import logging

from ..asserts import assert_research_handoff
from ..chat import chat, cost_of
from ..node_prompts import RESEARCH_SYSTEM_PROMPT
from ..schemas import build_research_handoff
from ..state import RoadGuardState
from ..tools import RESEARCH_TOOLS

LOGGER = logging.getLogger("roadguard.multiagent.research")

_VALID_COVERAGE = {"direct_match", "partial_match", "no_match"}
_VALID_CONFIDENCE = {"high", "medium", "low"}


def _quarantine_extract(passage_text: str) -> dict:
    """Dual-LLM step: extract a structured summary from a raw passage.

    In production this is a sandboxed small-model call (llama 3.2 1B local)
    whose only output format is this fixed JSON. Any instruction-looking
    text inside the passage never reaches the Decision agent.

    The offline stub keeps the extraction deterministic for tests.
    """
    text = passage_text or ""
    lower = text.lower()
    return {
        "repair_window_hours": (
            24 if "24 hours" in lower else (72 if "72 hours" in lower else None)
        ),
        "severity_level": (
            "severe" if "severe" in lower or "immediate" in lower else "routine"
        ),
        "requires_signage": "signage" in lower,
        "trimmed_summary": text[:160],
    }


def _parse_research_json(text: str) -> dict | None:
    """Pull the research JSON out of an LLM reply. None on any error."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    cov = obj.get("coverage")
    conf = obj.get("confidence")
    if cov not in _VALID_COVERAGE or conf not in _VALID_CONFIDENCE:
        return None
    return obj


def _llm_label(
    hazard_class: str,
    road_type: str,
    procedures_safe: list[dict],
    tool_result_ids: list[str],
    base_coverage: str,
    base_confidence: str,
) -> tuple[str, str, list[str], int, int]:
    """Ask the LLM to label coverage + confidence over the retrieved summaries.

    Returns (coverage, confidence, key_findings, tokens_in, tokens_out).
    On any failure returns the rule-based values with empty findings and
    zero tokens.
    """
    try:
        user_payload = {
            "hazard_class": hazard_class,
            "road_type": road_type,
            "tool_result_ids": tool_result_ids,
            "procedures": [
                {
                    "issuing_body": p.get("issuing_body"),
                    "article_ids": p.get("article_ids", []),
                    "quarantined_summary": p.get("quarantined_summary", {}),
                }
                for p in procedures_safe
            ],
            "tool_reported_coverage": base_coverage,
        }
        messages = [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        msg, in_tok, out_tok = chat(messages, tools=None, temperature=0.0)
    except Exception as exc:
        LOGGER.warning("research_llm_failed: %s", exc)
        return base_coverage, base_confidence, [], 0, 0

    parsed = _parse_research_json(msg.content or "")
    if parsed is None:
        return base_coverage, base_confidence, [], in_tok, out_tok

    findings = parsed.get("key_findings") or []
    if not isinstance(findings, list):
        findings = []
    return (
        parsed["coverage"],
        parsed["confidence"],
        [str(f)[:200] for f in findings][:5],
        in_tok,
        out_tok,
    )


def research_node(state: RoadGuardState) -> dict:
    """Call the RAG tool, optionally LLM-label coverage, build the handoff."""
    profile = state.get("hazard_profile") or {}
    md = state.get("metadata") or {}

    hazard_class = profile.get("primary_class", "D40")
    road_type = md.get("road_type", "main")

    result = RESEARCH_TOOLS["query_repair_standards"][0](
        hazard_class=hazard_class, road_type=road_type, k=3
    )

    procedures_safe = []
    for p in result["procedures"]:
        quarantined = _quarantine_extract(p.get("passage", ""))
        procedures_safe.append(
            {
                "source_url": p.get("source_url"),
                "issuing_body": p.get("issuing_body"),
                "date": p.get("date"),
                "article_ids": p.get("article_ids", []),
                "quarantined_summary": quarantined,
            }
        )

    base_coverage = result["coverage"]
    base_confidence = "high" if base_coverage == "direct_match" else "medium"

    coverage, confidence, findings, in_tok, out_tok = _llm_label(
        hazard_class=hazard_class,
        road_type=road_type,
        procedures_safe=procedures_safe,
        tool_result_ids=result["tool_result_ids"],
        base_coverage=base_coverage,
        base_confidence=base_confidence,
    )

    # Grounding guard: if the LLM says direct_match but the tool returned
    # no procedures, trust the tool - it saw the retrieval.
    if coverage != "no_match" and not procedures_safe:
        coverage = "no_match"
        confidence = "low"

    handoff = build_research_handoff(
        procedures=procedures_safe,
        tool_result_ids=result["tool_result_ids"],
        coverage=coverage,
        confidence=confidence,
    )
    assert_research_handoff(handoff)

    trace_entry = {
        "iter": state.get("iterations", 0) + 1,
        "speaker": "research",
        "kind": "handoff_out",
        "content": {
            "to": "decision",
            "coverage": coverage,
            "n_procedures": len(procedures_safe),
            "tool_result_ids": result["tool_result_ids"],
            "key_findings": findings,
            "llm_used": in_tok > 0,
        },
    }

    update: dict = {
        "procedures": procedures_safe,
        "tool_result_ids": result["tool_result_ids"],
        "research_coverage": coverage,
        "trace": [trace_entry],
        "iterations": 1,
    }
    if in_tok or out_tok:
        update["tokens_in"] = in_tok
        update["tokens_out"] = out_tok
        update["cost_usd"] = cost_of(in_tok, out_tok)
    return update
