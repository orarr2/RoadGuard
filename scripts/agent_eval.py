"""Per-agent evaluation over the labelled scenarios.

Runs the compiled graph on tests/labeled_scenarios.json (real, labelled by
the data track) and tests/multiagent_eval_scenarios.json (synthetic), and
reports what the brief asks for per agent and for the whole run: action
accuracy, LLM calls / failures / fallbacks per node, tool calls made by
Decision, floors applied, and wall-clock P50 / P95.

The run uses whatever ROADGUARD_USE_LIVE / AGENT_* env is in force, so on
a developer machine with USE_LIVE=0 it measures the rule-tree path, and on
the VM with the local Ollama models it measures the live path. It never
pretends otherwise: the report names the mode.

Isolation: the agent tools write approval, work-order and evidence rows
through ROADGUARD_DB_PATH and do not honour ROADGUARD_DRY_RUN, so by
default this script points ROADGUARD_DB_PATH and ROADGUARD_CHECKPOINT_PATH
at a fresh temporary directory and blanks ROADGUARD_APPROVAL_BASE_URL (no
approval emails). Pass --db to evaluate against a copy of a real database
(for catalogue lookups); never the production file.

usage:
    python scripts/agent_eval.py [--only labeled|synthetic] [--limit N]
                                 [--db PATH] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AGENTS = ("sanitize", "signal_extractor", "supervisor", "profile_builder", "rag_retriever", "decision")


def _load(name: str) -> list[dict]:
    path = ROOT / "tests" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[idx], 2)


def run(scenarios: list[dict]) -> dict:
    from roadguard_agent import run_graph

    per_agent: dict[str, Counter] = defaultdict(Counter)
    tool_calls: Counter = Counter()
    floors: Counter = Counter()
    fallback_reasons: Counter = Counter()
    confusion: Counter = Counter()
    walls: list[float] = []
    rows: list[dict] = []
    correct = 0

    for sc in scenarios:
        t0 = time.monotonic()
        try:
            final = run_graph(
                hazard_id=f"eval_{sc['id']}",
                yolo_output=dict(sc["yolo_output"]),
                image_quality=dict(sc["image_quality"]),
                metadata=dict(sc["metadata"]),
            )
            err = None
        except Exception as exc:
            final, err = {}, f"{type(exc).__name__}: {exc}"
        wall = time.monotonic() - t0
        walls.append(wall)
        action = str(final.get("action") or "ERROR")
        expected = sc["expected_action"]
        ok = action == expected
        correct += ok
        confusion[f"{expected}->{action}"] += 1
        if final.get("fallback_reason"):
            fallback_reasons[str(final["fallback_reason"])] += 1

        for entry in final.get("trace") or []:
            sp = str(entry.get("speaker") or "?")
            c = entry.get("content") or {}
            if not isinstance(c, dict):
                continue
            per_agent[sp]["entries"] += 1
            llm = c.get("llm") or {}
            if isinstance(llm, dict) and llm.get("calls"):
                per_agent[sp]["llm_calls"] += int(llm.get("calls") or 0)
                per_agent[sp]["llm_failed_calls"] += int(llm.get("failed_calls") or 0)
                per_agent[sp]["llm_latency_s"] += float(llm.get("latency_s") or 0.0)
            if c.get("llm_fallback_reason"):
                per_agent[sp]["llm_fallback"] += 1
            if str(c.get("llm_skipped_reason") or "").startswith("clear_case_fast_path"):
                per_agent[sp]["fast_path_skips"] += 1
            if entry.get("kind") == "react_step":
                tool_calls[f"{sp}:{c.get('tool')}"] += 1
            if c.get("floor_applied"):
                floors[str(c.get("floor_reason") or "floor")] += 1
            if sp == "decision" and entry.get("kind") == "final":
                per_agent[sp][f"source:{c.get('source')}"] += 1

        rows.append({
            "id": sc["id"], "kind": sc.get("kind"), "expected": expected, "actual": action,
            "ok": ok, "wall_s": round(wall, 2), "error": err,
            "fallback_reason": final.get("fallback_reason"),
            "image_match": (final.get("image_match_status") or {}).get("reason") if isinstance(final.get("image_match_status"), dict) else None,
        })

    n = len(scenarios)
    return {
        "mode": "live" if os.environ.get("ROADGUARD_USE_LIVE") == "1" else "rule_tree_only (ROADGUARD_USE_LIVE!=1)",
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "wall_p50_s": _pct(walls, 0.50),
        "wall_p95_s": _pct(walls, 0.95),
        "wall_max_s": round(max(walls), 2) if walls else None,
        "per_agent": {a: dict(per_agent[a]) for a in AGENTS if per_agent.get(a)},
        "decision_tool_calls": dict(tool_calls),
        "floors_applied": dict(floors),
        "fallback_reasons": dict(fallback_reasons),
        "confusion": dict(confusion),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["labeled", "synthetic"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--db", default=None, help="SQLite file to run against (default: fresh temp db)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not args.db:
        tmp = Path(tempfile.mkdtemp(prefix="rg-agent-eval-"))
        args.db = str(tmp / "eval.db")
        os.environ["ROADGUARD_CHECKPOINT_PATH"] = str(tmp / "checkpoints.sqlite")
    os.environ["ROADGUARD_DB_PATH"] = args.db
    os.environ["ROADGUARD_APPROVAL_BASE_URL"] = ""
    import roadguard_agent as ra
    ra.reset_settings_cache()
    print(f"db={args.db}")

    scenarios: list[dict] = []
    if args.only in (None, "labeled"):
        scenarios += _load("labeled_scenarios.json")
    if args.only in (None, "synthetic"):
        scenarios += _load("multiagent_eval_scenarios.json")
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        print("no scenarios found", file=sys.stderr)
        return 1

    report = run(scenarios)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, ensure_ascii=False))
    wrong = [r for r in report["rows"] if not r["ok"]]
    if wrong:
        print(f"\nmismatches ({len(wrong)}):")
        for r in wrong:
            print(f"  {r['id']}: expected {r['expected']} got {r['actual']}  {r['error'] or ''}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
