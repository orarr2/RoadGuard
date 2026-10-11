"""Budget guard: four caps + graceful stop + context-safe tracker."""

import time

from agent.multiagent.guards import (
    COST_CAP_USD,
    MAX_ITERATIONS,
    MAX_TOOL_HOPS_PER_AGENT,
    PRICE_IN,
    PRICE_OUT,
    BudgetTracker,
    graceful_stop_decision,
)


def test_fresh_tracker_does_not_fire():
    bt = BudgetTracker()
    fired, _ = bt.check()
    assert fired is False


def test_iteration_cap_fires():
    bt = BudgetTracker()
    bt.iterations = MAX_ITERATIONS
    fired, reason = bt.check()
    assert fired
    assert "MAX_ITERATIONS" in reason


def test_cost_cap_fires_when_tokens_exceed_budget():
    bt = BudgetTracker()
    out_cost = COST_CAP_USD / PRICE_OUT + 1
    bt.tick_tokens(0, int(out_cost))
    fired, reason = bt.check()
    assert fired
    assert "COST_CAP_USD" in reason


def test_wall_clock_cap_fires():
    bt = BudgetTracker()
    bt.started_at = time.time() - 60.0
    fired, reason = bt.check()
    assert fired
    assert "WALL_CLOCK_CAP_SECONDS" in reason


def test_tool_hop_cap_per_agent():
    bt = BudgetTracker()
    for _ in range(MAX_TOOL_HOPS_PER_AGENT):
        bt.tick_tool_hop("perception")
    fired, reason = bt.check_tool_hop("perception")
    assert fired
    assert "perception" in reason

    other_fired, _ = bt.check_tool_hop("research")
    assert other_fired is False


def test_cost_usd_computes_from_tokens():
    bt = BudgetTracker()
    bt.tick_tokens(1000, 500)
    expected = 1000 * PRICE_IN + 500 * PRICE_OUT
    assert bt.cost_usd == expected


def test_graceful_stop_matches_contract():
    out = graceful_stop_decision("MAX_ITERATIONS (8) reached")
    assert out["action"] == "human_review"
    assert out["guard_fired"] is True
    assert out["cited_sources"] == []
    assert "budget guard fired" in out["rationale"]


def test_tracker_tick_increments():
    bt = BudgetTracker()
    bt.tick_iteration()
    bt.tick_iteration()
    assert bt.iterations == 2


def test_context_isolation_via_supervisor():
    """new_budget() in supervisor must give each run its own tracker."""
    from agent.multiagent.nodes.supervisor import current_budget, new_budget

    a = new_budget()
    a.tick_iteration()
    assert current_budget() is a
    assert a.iterations == 1

    b = new_budget()
    assert b is not a
    assert b.iterations == 0
