"""Export the formatted agent trace of one round: every hazard of the last
N hours, oldest first, each printed with format_trace_lines, with a summary
block under the header (14.11.2026, the owner's ask after the r31 review:
how often the LLM changed the rule tree's verdict, how many floors fired,
how often the worker LLMs changed their output, and the LLM-call
distribution). The summary is computed from the very lines printed below
it, so the two never disagree.

On the VM, as the roadguard user from the repo root:

    /opt/roadguard/venv/bin/python scripts/export_round_trace.py --round 32 > /tmp/round-r32-trace.txt
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sqlite3
import statistics
import sys
from datetime import datetime, timezone

# The repo root: the script's parent folder, or the working directory when
# the script runs from a copy outside the repo (cd /opt/roadguard/repo first).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.getcwd(), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from roadguard_agent import format_trace_lines  # noqa: E402

_AGENT_VALUE = re.compile(r"agent value: rule_tree=(\S+)\s+with_workers=(\S+)\s+llm=(\S+)\s+final=(\S+)")
_FLOOR = re.compile(r"floor applied: (\S+)")
_PROFILE_CHANGED = re.compile(r"profile_llm_changed=(True|False)")
_RESEARCH_CHANGED = re.compile(r"research_llm_changed=(True|False)")
_PROFILE_LLM_USED = re.compile(r"\[PROFILE \] class=\S+\s+severity=\S+\s+confidence=\S+\s+llm_used=(True|False)")
_RAG_LLM_USED = re.compile(r"\[RAG     \][^\n]*llm_used=(True|False)")
_CALLS = re.compile(r"\[DONE    \][^\n]*llm calls=(\d+)")
# severity order of the final actions, for "raised" / "lowered"
_ORDER = {"known_image": 0, "needs_more_evidence": 1, "standard": 1, "human_review": 2, "urgent": 3}


def summarize(cases: list[list[str]]) -> list[str]:
    """Summary lines from the formatted case blocks."""
    n = len(cases)
    llm_changed = 0
    llm_raised = 0
    llm_lowered = 0
    floors: collections.Counter[str] = collections.Counter()
    profile_runs = profile_changed = research_runs = research_changed = 0
    calls: collections.Counter[int] = collections.Counter()
    for lines in cases:
        text = "\n".join(lines)
        floor = _FLOOR.search(text)
        if floor:
            floors[floor.group(1)] += 1
        av = _AGENT_VALUE.search(text)
        if av:
            rule_tree, _with_workers, llm, final = av.groups()
            if llm not in ("-", rule_tree) and final == llm and not floor:
                llm_changed += 1
                if _ORDER.get(llm, 1) > _ORDER.get(rule_tree, 1):
                    llm_raised += 1
                else:
                    llm_lowered += 1
        pu = _PROFILE_LLM_USED.search(text)
        if pu and pu.group(1) == "True":
            profile_runs += 1
        pc = _PROFILE_CHANGED.search(text)
        if pc and pc.group(1) == "True":
            profile_changed += 1
        ru = _RAG_LLM_USED.search(text)
        if ru and ru.group(1) == "True":
            research_runs += 1
        rc = _RESEARCH_CHANGED.search(text)
        if rc and rc.group(1) == "True":
            research_changed += 1
        c = _CALLS.search(text)
        if c:
            calls[int(c.group(1))] += 1
    floor_bits = ", ".join(f"{k} {v}" for k, v in floors.most_common()) or "none"
    call_bits = "  ".join(f"{k} call{'s' if k != 1 else ''}: {v}" for k, v in sorted(calls.items()))
    all_calls = [k for k, v in calls.items() for _ in range(v)]
    median_calls = statistics.median(all_calls) if all_calls else 0
    return [
        f"summary ({n} cases, from the lines below)",
        f"  rule tree vs final: the LLM changed the tree's verdict in {llm_changed} case(s)"
        f" (raised {llm_raised}, lowered {llm_lowered}); floors overrode the LLM in {sum(floors.values())}: {floor_bits}",
        f"  worker LLMs: profile_llm_changed in {profile_changed} of {profile_runs} profile runs with an LLM;"
        f" research_llm_changed in {research_changed} of {research_runs} research runs with an LLM",
        f"  llm calls per case: {call_bits or 'none'}  (total {sum(all_calls)}, median {median_calls:g})",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--round", required=True, help="round number, e.g. 32")
    ap.add_argument("--hours", type=int, default=24, help="look back this many hours (default 24)")
    ap.add_argument("--db", default="/var/lib/roadguard/roadguard.db")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT id, reported_at, class, confidence, road_type, lane_blocked,
                  location, bboxes_json, priority, agent_decision, agent_rationale,
                  agent_trace_json, status, cited_sources_json
             FROM hazards
             WHERE id LIKE 'hz_%'
               AND datetime(reported_at,'unixepoch') >= datetime('now', ?)
             ORDER BY reported_at ASC""",
        (f"-{int(args.hours)} hours",),
    ).fetchall()

    cases: list[list[str]] = []
    for r in rows:
        trace = json.loads(r["agent_trace_json"]) if r["agent_trace_json"] else []
        metadata = {
            "class": r["class"], "confidence": r["confidence"],
            "road_type": r["road_type"], "lane_blocked": r["lane_blocked"],
            "location": r["location"],
        }
        verdict = {"action": r["agent_decision"], "priority": r["priority"]}
        cases.append(format_trace_lines(r["id"], trace, metadata=metadata, verdict=verdict, status=r["status"]))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"RoadGuard agent trace, round r{args.round}, {len(cases)} cases, exported {stamp}")
    print()
    for line in summarize(cases):
        print(line)
    print()
    for lines in cases:
        print("\n".join(lines))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
