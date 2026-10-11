"""LangGraph StateGraph wiring - the explicit multi-agent graph.

Per DS23 S44: the manual loop was a graph all along. This file makes it
explicit so it is observable, checkpointable, and extensible without
rewriting the control flow.

LangGraph is an optional dependency. When it is not installed,
run_graph_locally() mirrors the topology by hand so CI and the offline
eval harness still exercise the same control flow.
"""

from __future__ import annotations

from typing import Any

from .nodes import (
    decision_node,
    perception_node,
    research_node,
    route_after_decision,
    sanitize_node,
    supervisor_node,
)
from .state import RoadGuardState, initial_state

try:
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    END = "__end__"
    StateGraph = None  # type: ignore


def build_graph():
    """Return a compiled LangGraph app.

    Topology:
      sanitize -> supervisor -> perception -> supervisor -> research
      -> supervisor -> decision -> conditional edge:
          { needs_more_evidence | standard -> END,
            urgent | human_review -> approval_gate -> END,
            guard_fired -> END }
    """
    if not _HAS_LANGGRAPH:
        raise RuntimeError(
            "langgraph is not installed. `pip install langgraph>=0.2` or use "
            "`run_graph_locally` for the dependency-free fallback."
        )

    graph = StateGraph(RoadGuardState)
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("supervisor_pre_perception", supervisor_node)
    graph.add_node("perception", perception_node)
    graph.add_node("supervisor_pre_research", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("supervisor_pre_decision", supervisor_node)
    graph.add_node("decision", decision_node)
    graph.add_node("approval_gate", _approval_gate_node)

    graph.set_entry_point("sanitize")
    graph.add_edge("sanitize", "supervisor_pre_perception")
    graph.add_edge("supervisor_pre_perception", "perception")
    graph.add_edge("perception", "supervisor_pre_research")
    graph.add_edge("supervisor_pre_research", "research")
    graph.add_edge("research", "supervisor_pre_decision")
    graph.add_edge("supervisor_pre_decision", "decision")
    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {"end": END, "approval_gate": "approval_gate"},
    )
    graph.add_edge("approval_gate", END)

    return graph.compile()


def _approval_gate_node(state: RoadGuardState) -> dict:
    """Writes a pending-approval row. Does NOT execute the action.

    Side effect: in production this triggers a push/email to a human
    supervisor. The URL/row id gets stored for the frontend to display.
    The scaffold stub only emits a trace entry; wiring to the SQLite
    approvals table and the email job happens in a later phase.
    """
    approval_entry = {
        "iter": state.get("iterations", 0) + 1,
        "speaker": "approval_gate",
        "kind": "pending_human_approval",
        "content": {
            "action": state.get("action"),
            "priority": state.get("priority"),
            "hazard_id": state.get("hazard_id"),
        },
    }
    return {"trace": [approval_entry], "iterations": 1}


def run_graph_locally(
    hazard_id: str,
    yolo_output: dict,
    image_quality: dict,
    metadata: dict,
) -> RoadGuardState:
    """Dependency-free execution path for CI and the offline eval harness.

    Mirrors the LangGraph topology exactly but calls each node by hand.
    Use build_graph() in production once langgraph is installed.
    """
    from .nodes.supervisor import new_budget

    new_budget()
    state: RoadGuardState = initial_state(hazard_id, yolo_output, image_quality, metadata)

    def _merge(s: RoadGuardState, update: dict[str, Any]) -> RoadGuardState:
        new: dict[str, Any] = dict(s)
        for k, v in update.items():
            if k == "trace":
                new[k] = list(new.get(k, [])) + list(v)
            elif k in ("iterations", "tokens_in", "tokens_out", "cost_usd"):
                new[k] = (new.get(k, 0) or 0) + v
            else:
                new[k] = v
        return new  # type: ignore[return-value]

    def _step(fn, s: RoadGuardState) -> RoadGuardState:
        return _merge(s, fn(s))

    steps = (
        sanitize_node,
        supervisor_node,
        perception_node,
        supervisor_node,
        research_node,
        supervisor_node,
        decision_node,
    )
    for step_fn in steps:
        state = _step(step_fn, state)
        if state.get("guard_fired") or state.get("final"):
            if state.get("action") in ("urgent", "human_review"):
                state = _step(_approval_gate_node, state)
            break

    return state
