"""RAG relevance check over the (hazard_class x road_type) grid.

For every pair the retriever must return at least one passage whose text
names the hazard class (relevance.class_keyword_hit) and must not report
no_match. Ground truth is by construction: a repair-standards corpus that
cannot produce a pothole passage for a D40 query is not relevant, whatever
the vector distance says.

usage:
    python scripts/rag_relevance_eval.py [--k 3] [--json out.json]

Exit code 1 when the class-hit rate is below 95%. Runs against whatever
corpus ROADGUARD_RAG_PATH / ROADGUARD_RAG_COLLECTION point at; on a host
without a seeded corpus it falls back to the stub corpus and says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ("D00", "D10", "D20", "D40")
ROADS = ("main", "secondary", "bike_path", "parking")
TARGET = 0.95


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--json", default=None, help="write per-query rows here")
    args = ap.parse_args()

    from roadguard_agent import RagClient, query_repair_standards

    shared = RagClient.shared()
    corpus = "chroma" if shared.available() and shared.count() > 0 else "stub"
    rows: list[dict] = []
    for cls in CLASSES:
        for road in ROADS:
            t0 = time.monotonic()
            res = query_repair_standards(cls, road, k=args.k)
            dt = round(time.monotonic() - t0, 3)
            procs = res["procedures"]
            class_hits = sum(1 for p in procs if (p.get("relevance") or {}).get("class_keyword_hit"))
            dists = [p.get("distance") for p in procs if p.get("distance") is not None]
            rows.append({
                "hazard_class": cls,
                "road_type": road,
                "coverage": res["coverage"],
                "n_passages": len(procs),
                "class_hits": class_hits,
                "ids": res["tool_result_ids"],
                "best_distance": min(dists) if dists else None,
                "seconds": dt,
                "pass": res["coverage"] != "no_match" and class_hits >= 1,
            })

    passed = sum(1 for r in rows if r["pass"])
    rate = passed / len(rows)
    print(f"corpus={corpus}  queries={len(rows)}  pass={passed}  rate={rate:.1%}  target={TARGET:.0%}")
    for r in rows:
        flag = "ok " if r["pass"] else "BAD"
        print(
            f"  {flag} {r['hazard_class']}/{r['road_type']:<10} coverage={r['coverage']:<13}"
            f" passages={r['n_passages']} class_hits={r['class_hits']}"
            f" best_dist={r['best_distance']}  {r['seconds']}s  ids={r['ids'][:3]}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps({"corpus": corpus, "rate": rate, "rows": rows}, indent=2), encoding="utf-8")
    return 0 if rate >= TARGET else 1


if __name__ == "__main__":
    sys.exit(main())
