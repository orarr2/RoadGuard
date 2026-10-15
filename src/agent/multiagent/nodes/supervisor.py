"""Supervisor: lightweight Python router (not an LLM).

Per the topology-cost analysis in AGENTS/architecture.md section 9:
keeping the supervisor as a pure Python function saves two LLM calls per
run and is the reason this 3-agent setup is cheaper than the monolith.

Responsibilities:
  1. Sanitize input before the graph runs (boundary defence).
  2. Enforce the budget guard between every node transition.
  3. Decide the next node after Decision (urgent/standard/more_evidence/review).

Budget tracker is held in a ContextVar so each concurrent FastAPI request
gets its own tracker; new_budget() creates a fresh one, current_budget()
reads the current one (lazily initializing if none exists). This avoids
the thread-safety trap of a module-level global.
"""

from __future__ import annotations

import contextvars

from ..guards import BudgetTracker, graceful_stop_decision
from ..sanitize import sanitize_location, sanitize_ocr
from ..state import RoadGuardState

_budget_var: contextvars.ContextVar[BudgetTracker] = contextvars.ContextVar(
    "roadguard_multiagent_budget"
)


def new_budget() -> BudgetTracker:
    """Install a fresh tracker in the current context. Call once per run."""
    bt = BudgetTracker()
    _budget_var.set(bt)
    return bt


def current_budget() -> BudgetTracker:
    """Return the current context's tracker, creating one if missing."""
    try:
        return _budget_var.get()
    except LookupError:
        bt = BudgetTracker()
        _budget_var.set(bt)
        return bt


def sanitize_node(state: RoadGuardState) -> dict:
    """First node in the graph. Not an LLM. Runs boundary defence."""
    original_md = state.get("metadata") or {}
    md = dict(original_md)
    original_location = md.get("location", "")
    md["location"] = sanitize_location(original_location)
    md["location_was_scrubbed"] = md["location"] != original_location

    iq = dict(state.get("image_quality") or {})
    ocr = sanitize_ocr(iq.get("ocr_text_detected"))
    iq["ocr_sanitized"] = ocr

    trace_entry = {
        "iter": 0,
        "speaker": "sanitize",
        "kind": "boundary_defence",
        "content": {
            "location_scrubbed": md["location_was_scrubbed"],
            "ocr_suspicious": ocr["is_suspicious_pattern"],
        },
    }

    return {
        "metadata": md,
        "image_quality": iq,
        "sanitized": True,
        "trace": [trace_entry],
    }


def supervisor_node(state: RoadGuardState) -> dict:
    """Called before every worker. Enforces the guard."""
    budget = current_budget()
    budget.tick_iteration()
    fired, reason = budget.check()
    if fired:
        decision = graceful_stop_decision(reason)
        return {
            "action": decision["action"],
            "rationale": decision["rationale"],
            "priority": decision["priority"],
            "cited_sources": decision["cited_sources"],
            "guard_fired": True,
            "final": True,
            "trace": [
                {
                    "iter": budget.iterations,
                    "speaker": "budget_guard",
                    "kind": "guard_fired",
                    "content": {"reason": reason},
                }
            ],
        }
    return {}


def route_after_decision(state: RoadGuardState) -> str:
    """Conditional edge after the Decision agent (per DS23 S44 slide 26)."""
    if state.get("guard_fired"):
        return "end"
    if state.get("final"):
        action = state.get("action")
        if action in ("urgent", "human_review"):
            return "approval_gate"
        return "end"
    action = state.get("action")
    if action == "needs_more_evidence":
        return "end"
    if action in ("urgent", "human_review"):
        return "approval_gate"
    if action == "standard":
        return "end"
    return "end"
