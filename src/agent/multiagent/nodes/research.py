"""Research Agent: READ-ONLY RAG lookup.

Reads:  hazard_profile, metadata.road_type
Writes: procedures, tool_result_ids, research_coverage
Tools:  query_repair_standards (read-only)

Includes the dual-LLM pattern (docs/multiagent-architecture.md section 7.4)
for vector 3 of the prompt-injection surface. The Research agent passes
the raw RAG passage through a quarantine extractor that emits only a
structured JSON summary. The Decision agent never sees the raw passage.
"""

from __future__ import annotations

from ..asserts import assert_research_handoff
from ..schemas import build_research_handoff
from ..state import RoadGuardState
from ..tools import RESEARCH_TOOLS


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


def research_node(state: RoadGuardState) -> dict:
    """Call the RAG tool and build the handoff for the Decision agent."""
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

    handoff = build_research_handoff(
        procedures=procedures_safe,
        tool_result_ids=result["tool_result_ids"],
        coverage=result["coverage"],
        confidence="high" if result["coverage"] == "direct_match" else "medium",
    )
    assert_research_handoff(handoff)

    trace_entry = {
        "iter": state.get("iterations", 0) + 1,
        "speaker": "research",
        "kind": "handoff_out",
        "content": {
            "to": "decision",
            "coverage": result["coverage"],
            "n_procedures": len(procedures_safe),
            "tool_result_ids": result["tool_result_ids"],
        },
    }

    return {
        "procedures": procedures_safe,
        "tool_result_ids": result["tool_result_ids"],
        "research_coverage": result["coverage"],
        "trace": [trace_entry],
        "iterations": 1,
    }
