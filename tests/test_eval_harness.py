"""Smoke test: the 10 seed scenarios all resolve correctly under the rule provider."""

from pathlib import Path

from agent.evaluation.run_eval import run, summarize


def test_rule_provider_passes_all_ten_seed_scenarios():
    root = Path(__file__).resolve().parents[1] / "src" / "agent" / "evaluation"
    results = run(
        scenarios_path=root / "scenarios.json",
        rag_path=root / "fixture_rag.json",
        provider_name="rule",
    )
    summary = summarize(results)
    assert summary["total"] == 10
    assert summary["correct_rate"] >= 0.85
    assert summary["grounded_rate"] >= 0.9


def test_each_scenario_matches_allowed_actions():
    root = Path(__file__).resolve().parents[1] / "src" / "agent" / "evaluation"
    results = run(
        scenarios_path=root / "scenarios.json",
        rag_path=root / "fixture_rag.json",
        provider_name="rule",
    )
    for r in results:
        assert r["correct"], (r["id"], r["expected"], r["got"], r["allowed"])
