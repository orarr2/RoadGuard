"""Stand-alone RAG truth test. Run on the live VM to prove the
retrieval path is actually wired to ChromaDB and not a static map.

Usage:
    PYTHONPATH=/opt/roadguard/repo /opt/roadguard/venv/bin/python3 \\
        /opt/roadguard/repo/scripts/rag_truth_test.py

Protocol (the reviewer's own):
  1. Insert a canary passage with a unique id into the PRODUCTION
     collection (same path and name as the live retriever uses).
  2. Query the public retriever and verify the canary id comes back.
  3. Delete the canary.
  4. Query again and verify the canary id does NOT come back.

If step 4 still returns the canary, the retriever is NOT grounded in
the live collection. Script exits non-zero on any step that misbehaves.

The canary is uniquely named per run and is removed on exit (success
or failure); the live corpus comes out of the test with exactly the
rows it had going in.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# When invoked outside a venv-installed package, make sure the project
# root is importable.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roadguard_agent import RagClient, query_repair_standards  # noqa: E402


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"{GREEN}OK  {RESET}{msg}")


def _fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")


def _note(msg: str) -> None:
    print(f"{YELLOW}note{RESET} {msg}")


def main() -> int:
    canary_id = f"rag-truth-canary-{uuid.uuid4().hex[:10]}"
    canary = {
        "id": canary_id,
        "text": (
            f"RAG truth-test canary {canary_id}. D40 hazard on a main road, "
            "repair window 24 hours. This passage was written only to prove "
            "that the live retriever is wired to the ChromaDB collection."
        ),
        "hazard_class": "D40",
        "road_type": "main",
        "issuing_body": "RoadGuard truth test",
        "publication_date": "2026-10-25",
        "source_url": "test://rag-truth",
    }

    print(f"RAG truth test on collection="
          f"{os.environ.get('ROADGUARD_RAG_COLLECTION', 'roadguard_repair_standards')} "
          f"path={os.environ.get('ROADGUARD_RAG_PATH', './chroma')}")
    client = RagClient.shared()
    if not client.available():
        _fail("chromadb not available on this host; cannot run the truth test")
        return 2

    exit_code = 0
    try:
        baseline = client.count()
        _note(f"baseline collection size: {baseline} passages")

        after_insert = client.upsert([canary])
        if after_insert <= baseline:
            _fail(f"upsert reported count={after_insert} (<= baseline {baseline}); "
                  "the canary row did not land in the collection")
            exit_code = 1
        else:
            _ok(f"inserted canary {canary_id}; collection now has {after_insert} passages")

        res_present = query_repair_standards(hazard_class="D40", road_type="main", k=5)
        present_ids = set(res_present.get("tool_result_ids") or [])
        if canary_id in present_ids:
            _ok(f"query_repair_standards returned the canary id (coverage="
                f"{res_present.get('coverage')})")
        else:
            _fail(f"query_repair_standards did NOT return the canary id. "
                  f"Returned ids: {present_ids}. Coverage: {res_present.get('coverage')}")
            exit_code = 1

        after_delete = client.delete([canary_id])
        if after_delete == baseline:
            _ok(f"deleted canary; collection back to {after_delete} passages")
        else:
            _fail(f"after delete, collection size is {after_delete}; expected {baseline}")
            exit_code = 1

        res_absent = query_repair_standards(hazard_class="D40", road_type="main", k=5)
        absent_ids = set(res_absent.get("tool_result_ids") or [])
        if canary_id not in absent_ids:
            _ok(f"query after delete does NOT return the canary - the retriever "
                f"tracks the collection (coverage={res_absent.get('coverage')})")
        else:
            _fail(f"query after delete STILL returned the canary id {canary_id}. "
                  "This proves the output does not track the collection - a cache "
                  "or static map is masquerading as a live retrieval.")
            exit_code = 1
    finally:
        # Make absolutely sure the canary is gone even on an unexpected
        # exception, so the live corpus comes out unchanged.
        try:
            client.delete([canary_id])
        except Exception:
            pass

    if exit_code == 0:
        print(f"\n{GREEN}RAG truth test PASSED{RESET} - the retriever is wired to "
              "the live ChromaDB collection.")
    else:
        print(f"\n{RED}RAG truth test FAILED{RESET} - see the FAIL lines above.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
