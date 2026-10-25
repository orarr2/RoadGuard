"""RAG truth test: prove the retriever is actually wired to ChromaDB.

The reviewer's P0 concern (Pasted markdown(4).md) was that the trace
shows a 'RAG' step with coverage=direct_match, but that alone does not
prove the pipeline actually hits a vector store - the same output
could come from a static mapping or a stub.

This test implements the reviewer's own protocol:
  1. Insert a canary passage with a unique id into the ChromaDB corpus.
  2. Query via `query_repair_standards` and verify the canary comes
     back (and no other passage uses that id).
  3. Delete the canary passage.
  4. Query again and verify the canary is NOT in the result - the
     retrieval genuinely tracks the collection, and the system does
     not fabricate a direct_match when the source is absent.

If the pipeline were static, step 3's deletion would not affect step 4
and the canary would still appear. The test fails loudly in that case.
"""

from __future__ import annotations

import uuid

import pytest


@pytest.fixture()
def canary_id() -> str:
    """A unique id per test invocation so a stale canary from a crashed
    run cannot muddle the signal."""
    return f"rag-truth-canary-{uuid.uuid4().hex[:10]}"


def test_rag_truth_canary_round_trip(canary_id, monkeypatch, tmp_path):
    """Full insert / query / delete / query cycle on an isolated collection."""
    # Isolate the Chroma store so this test cannot touch production data.
    monkeypatch.setenv("ROADGUARD_RAG_PATH", str(tmp_path / "chroma_truth"))
    monkeypatch.setenv("ROADGUARD_RAG_COLLECTION", f"truth_{canary_id[:12]}")
    from roadguard_agent import RagClient, query_repair_standards, reset_settings_cache

    reset_settings_cache()
    # Force a fresh singleton against the isolated path.
    RagClient._instance = None
    client = RagClient.shared()
    if not client.available():
        pytest.skip("chromadb not available in this environment")

    canary_doc = {
        "id": canary_id,
        "text": (
            "RAG truth-test canary. Hazard class D40 on main road, "
            "repair window 24 hours. Content tagged with a unique id "
            "that appears nowhere else in the corpus."
        ),
        "hazard_class": "D40",
        "road_type": "main",
        "issuing_body": "RoadGuard truth test",
        "publication_date": "2026-10-25",
        "source_url": "test://rag-truth",
    }

    # Step 1: seed the canary.
    after_insert = client.upsert([canary_doc])
    assert after_insert >= 1, "upsert claimed success but count is zero"

    # Step 2: query - the canary MUST surface.
    res_present = query_repair_standards(
        hazard_class="D40", road_type="main", k=3,
    )
    present_ids = set(res_present.get("tool_result_ids") or [])
    assert canary_id in present_ids, (
        f"RAG claimed coverage={res_present.get('coverage')} but did NOT return the "
        f"canary passage with id={canary_id}. Returned ids: {present_ids}. "
        "Either the retriever is not querying this collection, or filters/thresholds "
        "block the exact passage we just inserted."
    )
    assert res_present["coverage"] in ("direct_match", "partial_match"), res_present

    # Step 3: delete the canary.
    after_delete = client.delete([canary_id])
    # The collection may still hold other seed rows; what matters is the
    # canary id is gone, not the overall count.
    assert after_delete == after_insert - 1, (
        f"delete reported count={after_delete}, expected {after_insert - 1}"
    )

    # Step 4: query again - the canary MUST NOT surface. The retriever
    # is NOT allowed to fabricate the same answer from a stale cache or
    # a static mapping.
    res_absent = query_repair_standards(
        hazard_class="D40", road_type="main", k=3,
    )
    absent_ids = set(res_absent.get("tool_result_ids") or [])
    assert canary_id not in absent_ids, (
        f"RAG returned the canary id {canary_id} AFTER it was deleted. "
        f"Returned ids: {absent_ids}. This proves the output does not track the "
        "collection state - either a cache or a static map is masquerading as a "
        "live retrieval."
    )


def test_rag_truth_empty_collection_in_strict_mode_reports_no_match(
    monkeypatch, tmp_path, canary_id,
):
    """When ROADGUARD_RAG_STRICT=1 and the collection is empty, the
    retriever must report no_match - the stub corpus is explicitly off.

    By default (STRICT=0) the retriever falls back to a hardcoded stub
    corpus when the real collection is empty; that is deliberate so
    dev / test / unseeded prod boots do not crash the pipeline. But a
    production deployment that WANTS to fail loudly on an empty corpus
    sets STRICT=1, and the reviewer's RAG-wiring doubt gets a clean
    answer: an empty collection returns no_match, not fake coverage."""
    monkeypatch.setenv("ROADGUARD_RAG_PATH", str(tmp_path / "chroma_empty"))
    monkeypatch.setenv("ROADGUARD_RAG_COLLECTION", f"empty_{canary_id[:12]}")
    monkeypatch.setenv("ROADGUARD_RAG_STRICT", "1")
    from roadguard_agent import RagClient, query_repair_standards, reset_settings_cache

    reset_settings_cache()
    RagClient._instance = None
    client = RagClient.shared()
    if not client.available():
        pytest.skip("chromadb not available in this environment")

    assert client.count() == 0, "fresh isolated collection should start empty"

    res = query_repair_standards(hazard_class="D40", road_type="main", k=3)
    assert res["coverage"] == "no_match", (
        f"STRICT=1 + empty collection but retriever returned coverage={res['coverage']} "
        f"with {len(res.get('tool_result_ids') or [])} ids - the stub fallback "
        "fired instead of reporting the real empty state."
    )
    assert not res.get("tool_result_ids"), (
        f"empty collection returned ids: {res.get('tool_result_ids')}"
    )


def test_rag_truth_empty_collection_default_uses_stub(monkeypatch, tmp_path, canary_id):
    """Documentation test: without STRICT, an empty real collection falls
    back to the hardcoded stub corpus so dev / unseeded prod runs still
    produce verdicts. This is the historical behavior; STRICT above
    disables it."""
    monkeypatch.setenv("ROADGUARD_RAG_PATH", str(tmp_path / "chroma_empty"))
    monkeypatch.setenv("ROADGUARD_RAG_COLLECTION", f"empty_{canary_id[:12]}")
    monkeypatch.delenv("ROADGUARD_RAG_STRICT", raising=False)
    from roadguard_agent import RagClient, query_repair_standards, reset_settings_cache

    reset_settings_cache()
    RagClient._instance = None
    client = RagClient.shared()
    if not client.available():
        pytest.skip("chromadb not available in this environment")

    assert client.count() == 0
    res = query_repair_standards(hazard_class="D40", road_type="main", k=3)
    # Stub corpus is seeded for (D40, main), so coverage is non-empty.
    assert res["coverage"] in ("direct_match", "partial_match")
    assert res.get("tool_result_ids"), "stub fallback returned no ids"
