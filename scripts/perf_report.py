"""Per-worker latency + success-rate report over recent hazards.

Reads `hazards.agent_trace_json` for every row in the time window and
rolls up:
  - P50 / P95 LLM latency per worker (signal_extractor, profile_builder,
    rag_retriever, decision, evidence_request, intake)
  - LLM success rate (ok_calls / (ok_calls + failed_calls)) per worker
  - Token totals per worker
  - End-to-end pipeline latency (from the oldest trace ts to the final
    event ts) P50 / P95
  - Verdict distribution + grounded-citation rate
  - Budget-guard incidence (how many runs hit the wall-clock cap)

Usage:
    PYTHONPATH=/opt/roadguard/repo /opt/roadguard/venv/bin/python3 \\
        /opt/roadguard/repo/scripts/perf_report.py [--hours N]

Default window: last 24 hours. For a wider shot: `--hours 168` (one
week) or `--hours 720` (30 days). The script reads only; it writes
nothing to the DB.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DB_PATH_ENV = "ROADGUARD_DB_PATH"
DB_PATH_DEFAULT = "/var/lib/roadguard/roadguard.db"


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile. Returns 0.0 on an empty list.

    statistics.quantiles exists in py3.8+ but doesn't do arbitrary
    percentiles cleanly; this hand-rolled version is small and
    testable."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=int, default=24,
                        help="window in hours (default 24)")
    parser.add_argument("--db", default=os.environ.get(DB_PATH_ENV, DB_PATH_DEFAULT),
                        help="path to roadguard.db (default from "
                             "ROADGUARD_DB_PATH env or /var/lib/roadguard/roadguard.db)")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print(f"db not found: {args.db}", file=sys.stderr)
        return 2

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, agent_decision, agent_rationale, reported_at, "
        "cited_sources_json, agent_trace_json "
        "FROM hazards "
        "WHERE reported_at >= strftime('%s','now', ?) "
        "ORDER BY reported_at ASC",
        (f"-{int(args.hours)} hours",),
    ).fetchall()
    con.close()

    if not rows:
        print(f"no hazards in the last {args.hours} hour(s). Nothing to report.")
        return 0

    # Per-worker rollups.
    worker_latency: dict[str, list[float]] = defaultdict(list)
    worker_ok: dict[str, int] = defaultdict(int)
    worker_fail: dict[str, int] = defaultdict(int)
    worker_tokens_in: dict[str, int] = defaultdict(int)
    worker_tokens_out: dict[str, int] = defaultdict(int)
    worker_errors: dict[str, list[str]] = defaultdict(list)

    # End-to-end latency per hazard.
    e2e_latency: list[float] = []
    # Budget guards + verdict distribution.
    guard_fired = 0
    verdicts: dict[str, int] = defaultdict(int)
    grounded_denom = 0
    grounded_num = 0

    for r in rows:
        verdicts[r["agent_decision"] or "n/a"] += 1
        if (r["agent_decision"] or "").lower() in ("urgent", "standard"):
            grounded_denom += 1
            try:
                cited = json.loads(r["cited_sources_json"] or "[]")
                if cited:
                    grounded_num += 1
            except Exception:
                pass
        try:
            trace = json.loads(r["agent_trace_json"] or "[]")
        except Exception:
            continue
        if not trace:
            continue
        first_ts = trace[0].get("ts")
        last_ts = trace[-1].get("ts")
        if isinstance(first_ts, (int, float)) and isinstance(last_ts, (int, float)):
            e2e_latency.append(float(last_ts) - float(first_ts))

        for ev in trace:
            if ev.get("speaker") == "budget_guard" and ev.get("kind") == "guard_fired":
                guard_fired += 1
            content = ev.get("content") or {}
            llm = content.get("llm")
            if not isinstance(llm, dict):
                continue
            speaker = ev.get("speaker", "?")
            latency = llm.get("latency_s")
            if isinstance(latency, (int, float)) and latency > 0:
                worker_latency[speaker].append(float(latency))
            worker_ok[speaker] += int(llm.get("ok_calls", 0) or 0)
            worker_fail[speaker] += int(llm.get("failed_calls", 0) or 0)
            worker_tokens_in[speaker] += int(llm.get("tokens_in", 0) or 0)
            worker_tokens_out[speaker] += int(llm.get("tokens_out", 0) or 0)
            for err in (llm.get("errors") or []):
                worker_errors[speaker].append(str(err))

    # Report.
    print(f"RoadGuard performance report")
    print(f"window:  last {args.hours}h  ({len(rows)} hazards)")
    print(f"db:      {args.db}")
    print("")
    print("Verdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:22} {n}")
    if grounded_denom:
        rate = 100 * grounded_num / grounded_denom
        print(f"Grounded citation rate: {rate:.1f}% ({grounded_num} of {grounded_denom})")
    print(f"Budget-guard fires:     {guard_fired}")
    print("")

    if e2e_latency:
        p50 = _percentile(e2e_latency, 50)
        p95 = _percentile(e2e_latency, 95)
        print(f"End-to-end pipeline latency (n={len(e2e_latency)}):")
        print(f"  P50 {p50:7.2f}s   P95 {p95:7.2f}s   max {max(e2e_latency):7.2f}s")
        print("")

    print("Per-worker LLM latency + success rate:")
    print(f"  {'worker':17} {'n':>5} {'P50':>7} {'P95':>7} {'max':>7} "
          f"{'ok':>6} {'fail':>6} {'tok_in':>8} {'tok_out':>8}")
    for speaker in sorted(worker_latency.keys() | worker_ok.keys()):
        vals = worker_latency.get(speaker, [])
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        max_lat = max(vals) if vals else 0.0
        ok = worker_ok.get(speaker, 0)
        fail = worker_fail.get(speaker, 0)
        tin = worker_tokens_in.get(speaker, 0)
        tout = worker_tokens_out.get(speaker, 0)
        print(f"  {speaker:17} {len(vals):>5} {p50:>6.2f}s {p95:>6.2f}s "
              f"{max_lat:>6.2f}s {ok:>6} {fail:>6} {tin:>8} {tout:>8}")

    if any(worker_errors.values()):
        print("")
        print("Error types seen:")
        for speaker, errs in worker_errors.items():
            from collections import Counter
            tally = Counter(errs).most_common(5)
            if tally:
                print(f"  {speaker:17} " + ", ".join(f"{t}x{n}" for t, n in tally))
    return 0


if __name__ == "__main__":
    sys.exit(main())
