"""End-to-end eval: run the 3-agent graph on every scenario in the eval set.

Loads tests/multiagent_eval_scenarios.json, feeds each scenario through
the compiled LangGraph via run_graph, and asserts the final action
matches the labelled expected_action.

The eval set is seeded by the deterministic rule-based tree (USE_LIVE=0),
so a mismatch here means a real regression in the graph or in one of the
nodes. Live-LLM runs are out of scope: that is a separate CI gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roadguard_agent import run_graph

SCENARIOS_PATH = Path(__file__).parent / "multiagent_eval_scenarios.json"
LABELED_SCENARIOS_PATH = Path(__file__).parent / "labeled_scenarios.json"

# Keys every scenario object must carry (the parametrize ids and the
# test body read them without guards).
_REQUIRED_KEYS = frozenset(
    {"id", "kind", "expected_action", "yolo_output", "image_quality", "metadata"}
)


def _load_labeled(path: Path) -> list[dict]:
    """The labelled real scenarios, validated. An absent file is the
    normal case and yields []. A file that exists but does not parse, is
    not a list, or holds an entry without the required keys fails the
    module loudly: silently skipping it would make the eval look green
    while the delivered data is ignored."""
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        pytest.fail(f"{path.name} exists but is not valid JSON: {exc}")
    if not isinstance(data, list):
        pytest.fail(f"{path.name} must be a JSON list of scenario objects, got {type(data).__name__}")
    for i, scenario in enumerate(data):
        if not isinstance(scenario, dict):
            pytest.fail(f"{path.name}[{i}] is not a scenario object: {type(scenario).__name__}")
        missing = sorted(_REQUIRED_KEYS - scenario.keys())
        if missing:
            pytest.fail(f"{path.name}[{i}] is missing required keys: {missing}")
    return data


def _load_scenarios() -> list[dict]:
    """Base eval + optional labelled real scenarios.

    When the data track (Shai) delivers `tests/labeled_scenarios.json`
    (same schema as multiagent_eval_scenarios.json), we concat it so the
    eval harness exercises both synthetic + real data without a code
    change.
    """
    base: list[dict] = []
    with SCENARIOS_PATH.open("r", encoding="utf-8") as fh:
        base = json.load(fh)
    return base + _load_labeled(LABELED_SCENARIOS_PATH)


SCENARIOS = _load_scenarios()


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch):
    """Keep the eval deterministic: no live LLM calls, no network."""
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "0")
    monkeypatch.setenv("GROQ_API_KEY", "")
    yield


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_eval_scenario_matches_expected_action(scenario: dict) -> None:
    final = run_graph(
        hazard_id=scenario["id"],
        yolo_output=scenario["yolo_output"],
        image_quality=scenario["image_quality"],
        metadata=scenario["metadata"],
    )
    assert final["action"] == scenario["expected_action"], (
        f"scenario {scenario['id']} ({scenario['kind']}): "
        f"expected {scenario['expected_action']} but got {final['action']} "
        f"(rationale: {final.get('rationale', '')[:120]})"
    )


def test_eval_set_coverage_summary() -> None:
    """Make the eval set shape explicit so a reviewer can see distribution."""
    assert len(SCENARIOS) >= 50, f"eval set shrank: {len(SCENARIOS)}"
    actions = {s["expected_action"] for s in SCENARIOS}
    assert actions >= {"urgent", "standard", "needs_more_evidence", "human_review"}
    kinds = {s["kind"] for s in SCENARIOS}
    assert "simple" in kinds and "mid" in kinds
