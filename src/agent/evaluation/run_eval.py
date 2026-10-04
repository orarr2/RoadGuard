"""Evaluate the decision agent against scenarios.json.

Default provider is `rule`: a deterministic in-process stand-in that
mirrors the decision table and lets the harness run with no LLM. Pass
`--provider groq` to drive the real Groq function-calling loop (requires
GROQ_API_KEY and network).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_SRC_ROOT = _HERE.parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from agent import loop, policy, tools  # noqa: E402


class FixtureRAG:
    """Reads a tiny procedures fixture used by the rule-driven provider."""

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.procedures: list[dict[str, Any]] = data["procedures"]

    def query(
        self, hazard_class: str, road_type: str, k: int
    ) -> list[dict[str, Any]]:
        hits = [p for p in self.procedures if p.get("hazard_class") == hazard_class]
        general = [p for p in self.procedures if p.get("hazard_class") == "general"]
        both = hits + general
        return both[: max(1, k)]


class RuleProvider(loop.LLMProvider):
    """A deterministic LLM stand-in that follows the decision-table flow.

    Hop 0: emit a tool call for check_hazard_severity.
    Hop 1: emit a tool call for query_repair_standards.
    Hop 2: emit a JSON verdict chosen by policy.decision_table_action.
    """

    name = "rule"

    def __init__(self, hazard: dict[str, Any]) -> None:
        self.hazard = hazard
        self._hop = 0
        self._severity: dict[str, Any] | None = None
        self._passages: list[dict[str, Any]] = []

    def _ingest(self, messages: list[dict[str, Any]]) -> None:
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                if "priority" in payload and "band" in payload:
                    self._severity = payload
                if "passages" in payload:
                    passages = payload.get("passages") or []
                    if isinstance(passages, list):
                        self._passages = passages

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]] | None,
    ) -> loop.ProviderMessage:
        self._ingest(messages)
        if self._hop == 0:
            self._hop += 1
            return loop.ProviderMessage(
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "c-severity",
                        "name": "check_hazard_severity",
                        "arguments": {
                            "class": self.hazard["class"],
                            "confidence": float(self.hazard["confidence"]),
                            "road_type": self.hazard["road_type"],
                            "lane_blocked": self.hazard["lane_blocked"],
                        },
                    }
                ],
            )
        if self._hop == 1:
            self._hop += 1
            return loop.ProviderMessage(
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "c-rag",
                        "name": "query_repair_standards",
                        "arguments": {
                            "hazard_class": self.hazard["class"],
                            "road_type": self.hazard["road_type"],
                            "k": 2,
                        },
                    }
                ],
            )
        severity = self._severity or tools.compute_severity(
            self.hazard["class"],
            float(self.hazard["confidence"]),
            self.hazard["road_type"],
            self.hazard["lane_blocked"],
        )
        action = policy.decision_table_action(
            hazard=self.hazard,
            severity=severity,
            has_rag=bool(self._passages),
        )
        urls = [
            p["source_url"]
            for p in self._passages
            if isinstance(p, dict) and p.get("source_url")
        ]
        verdict = {
            "action": action,
            "rationale": (
                f"Decision table selected {action} for class "
                f"{self.hazard['class']} at confidence "
                f"{float(self.hazard['confidence']):.2f} on a "
                f"{self.hazard['road_type']} road."
            ),
            "priority": severity["priority"],
            "cited_sources": urls if action != "human_review" else [],
        }
        return loop.ProviderMessage(
            finish_reason="stop",
            content=json.dumps(verdict),
        )


def _build_provider(name: str, hazard: dict[str, Any]) -> loop.LLMProvider:
    if name == "rule":
        return RuleProvider(hazard)
    if name == "groq":
        return loop.GroqProvider()
    raise ValueError(f"unknown provider: {name!r}")


def run(
    scenarios_path: Path,
    rag_path: Path,
    provider_name: str,
) -> list[dict[str, Any]]:
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))["scenarios"]
    rag = FixtureRAG(rag_path)
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        hazard = scenario["hazard_input"]
        expected = scenario["expected_action"]
        allowed = set(scenario.get("allowed_actions") or [expected])
        provider = _build_provider(provider_name, hazard)
        ctx = tools.ToolContext(rag=rag)
        start = time.monotonic()
        decision = loop.agent_decide(hazard=hazard, provider=provider, ctx=ctx)
        elapsed = time.monotonic() - start
        is_review = decision.action == "human_review"
        results.append(
            {
                "id": scenario["id"],
                "band": scenario["band"],
                "expected": expected,
                "allowed": sorted(allowed),
                "got": decision.action,
                "correct": decision.action in allowed,
                "grounded": (not is_review) and bool(decision.cited_sources)
                or is_review,
                "priority": decision.priority,
                "latency_s": round(elapsed, 3),
                "fallback_reason": decision.fallback_reason,
                "cited_sources": decision.cited_sources,
            }
        )
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    non_review = [r for r in results if r["got"] != "human_review"]
    grounded = sum(1 for r in non_review if r["cited_sources"])
    action_dist: dict[str, int] = {}
    for r in results:
        action_dist[r["got"]] = action_dist.get(r["got"], 0) + 1
    return {
        "total": total,
        "correct": correct,
        "correct_rate": round(correct / total, 3) if total else 0.0,
        "grounded": grounded,
        "grounded_rate": round(grounded / max(len(non_review), 1), 3),
        "action_distribution": action_dist,
        "avg_latency_s": round(
            sum(r["latency_s"] for r in results) / max(total, 1), 3
        ),
    }


def _print_human(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    print(
        f"scenarios={summary['total']}  "
        f"correct={summary['correct']}  "
        f"correct_rate={summary['correct_rate'] * 100:.1f}%"
    )
    print(f"grounded_rate={summary['grounded_rate'] * 100:.1f}%")
    print(f"avg_latency_s={summary['avg_latency_s']}")
    print(f"action_distribution={summary['action_distribution']}")
    print()
    for r in results:
        status = "OK  " if r["correct"] else "MISS"
        print(
            f"  {status} {r['id']} [{r['band']}] "
            f"expected={r['expected']} got={r['got']} "
            f"priority={r['priority']} latency={r['latency_s']}s"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the RoadGuard agent evaluation harness."
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=_HERE.parent / "scenarios.json",
    )
    parser.add_argument(
        "--rag",
        type=Path,
        default=_HERE.parent / "fixture_rag.json",
    )
    parser.add_argument(
        "--provider",
        choices=["rule", "groq"],
        default="rule",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout only.",
    )
    parser.add_argument(
        "--min-correct",
        type=float,
        default=0.85,
        help="Correct-action rate required for exit code 0 (default: 0.85).",
    )
    args = parser.parse_args(argv)

    results = run(args.scenarios, args.rag, args.provider)
    summary = summarize(results)

    if args.json:
        print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False))
    else:
        _print_human(summary, results)

    return 0 if summary["correct_rate"] >= args.min_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
