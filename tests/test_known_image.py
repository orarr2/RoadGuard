"""Dataset catalogue, known_image action, /sql and /status image counts."""

from __future__ import annotations

import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    from roadguard_agent import Database, reset_settings_cache

    db_path = str(tmp_path / "known.db")
    monkeypatch.setenv("ROADGUARD_DB_PATH", db_path)
    monkeypatch.setenv("ROADGUARD_RAG_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ROADGUARD_CHECKPOINT_PATH", str(tmp_path / "ckpt.sqlite"))
    reset_settings_cache()
    with Database(db_path) as db:
        for pic_id, (name, split, sha, phash, cls) in enumerate(
            [
                ("firc_10001.jpg", "train", "a" * 64, "ffff0000ffff0000", "D20"),
                ("firc_10002.jpg", "test", "b" * 64, "0000ffff0000ffff", "D40"),
            ],
            start=1,
        ):
            db.insert_dataset_image({
                "pic_id": pic_id,
                "file_name": name,
                "split": split,
                "rel_path": f"{split}/images/{name}",
                "width": 640,
                "height": 640,
                "sha256": sha,
                "phash": phash,
                "n_labels": 1,
                "classes": cls,
                "labels_json": "[]",
                "model_top1_class": cls,
                "model_top1_confidence": 0.42,
                "model_bbox": None,
                "catalogued_at": 0.0,
            })
    return db_path


def test_dataset_lookup_matches_exact_sha_and_near_phash(seeded_db):
    from roadguard_agent import lookup_dataset_image

    hit = lookup_dataset_image(sha256="a" * 64)
    assert hit["found"] and hit["match"] == "sha256" and hit["pic_id"] == 1
    near = lookup_dataset_image(phash="ffff0000ffff0001")  # one bit off
    assert near["found"] and near["match"] == "phash" and near["distance"] == 1
    assert lookup_dataset_image(sha256="c" * 64)["found"] is False
    assert lookup_dataset_image()["found"] is False


def test_dataset_catalogue_stats(seeded_db):
    from roadguard_agent import Database

    with Database(seeded_db) as db:
        stats = db.dataset_catalogue_stats()
    assert stats["count"] == 2
    assert stats["by_split"] == {"train": 1, "test": 1}
    assert stats["with_predictions"] == 2


def test_df_view_exposes_pic_id(seeded_db):
    from roadguard_agent import Database

    with Database(seeded_db) as db:
        row = db._conn.execute("SELECT * FROM df WHERE pic_id = 1").fetchone()
    assert row is not None
    assert row["file_name"] == "firc_10001.jpg"
    assert row["classes"] == "D20"


def test_decision_short_circuits_to_known_image_when_catalogue_hits(seeded_db):
    """profile_builder places the catalogue hit on state.dataset_match;
    the deterministic Decision tree then returns known_image without
    calling the LLM or routing to the approval gate."""
    from roadguard_agent import _decide_rule_based

    state = {
        "hazard_profile": {"primary_class": "D20", "primary_confidence": 0.9},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "tool_result_ids": [],
        "research_coverage": "no_match",
        "perception_confidence": "high",
        "dataset_match": {
            "found": True,
            "match": "sha256",
            "distance": 0,
            "pic_id": 1,
            "file_name": "firc_10001.jpg",
            "split": "train",
            "classes": "D20",
            "n_labels": 1,
            "model_prediction": {"top1_class": "D20", "top1_confidence": 0.42},
        },
    }
    out = _decide_rule_based(state)
    assert out["action"] == "known_image"
    assert out["cited_sources"] == []
    assert "pic_id=1" in out["rationale"]
    assert "firc_10001" in out["rationale"]


def test_known_image_contradiction_routes_to_human_review(seeded_db):
    """Catalog says D20, detector says D40 at 0.65 - this is a
    catalog-vs-detector contradiction. The decision tree must NOT return
    known_image; the mismatch has to reach a supervisor."""
    from roadguard_agent import _decide_rule_based

    state = {
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.65},
        "metadata": {"road_type": "main", "lane_blocked": "partial"},
        "tool_result_ids": [],
        "research_coverage": "no_match",
        "perception_confidence": "medium",
        "dataset_match": {
            "found": True,
            "match": "sha256",
            "distance": 0,
            "pic_id": 1,
            "file_name": "firc_10001.jpg",
            "split": "train",
            "classes": "D20",
            "n_labels": 1,
            "model_prediction": {"top1_class": "D20", "top1_confidence": 0.42},
        },
    }
    out = _decide_rule_based(state)
    assert out["action"] == "human_review", out
    assert "contradiction" in out["rationale"].lower()
    assert "D40" in out["rationale"]
    assert "D20" in out["rationale"]


def test_known_image_contradiction_ignored_when_detector_confidence_low(seeded_db):
    """A detector that reports a different class but at near-zero
    confidence (<0.30) is noise, not a real contradiction. Keep the
    catalog verdict."""
    from roadguard_agent import _decide_rule_based

    state = {
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.12},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "tool_result_ids": [],
        "research_coverage": "no_match",
        "perception_confidence": "medium",
        "dataset_match": {
            "found": True,
            "match": "sha256",
            "distance": 0,
            "pic_id": 1,
            "file_name": "firc_10001.jpg",
            "split": "train",
            "classes": "D20",
            "n_labels": 1,
            "model_prediction": {"top1_class": "D20", "top1_confidence": 0.42},
        },
    }
    out = _decide_rule_based(state)
    assert out["action"] == "known_image", out


def test_known_image_contradiction_overlapping_class_is_not_a_contradiction(seeded_db):
    """Catalog has D00,D20 and detector reports D20. The detector's call
    is covered by the catalog labels; no contradiction, keep known_image."""
    from roadguard_agent import _decide_rule_based

    state = {
        "hazard_profile": {"primary_class": "D20", "primary_confidence": 0.72},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "tool_result_ids": [],
        "research_coverage": "no_match",
        "perception_confidence": "high",
        "dataset_match": {
            "found": True,
            "match": "phash",
            "distance": 3,
            "pic_id": 1,
            "file_name": "firc_10001.jpg",
            "split": "train",
            "classes": "D00,D20",
            "n_labels": 2,
            "model_prediction": {"top1_class": "D20", "top1_confidence": 0.42},
        },
    }
    out = _decide_rule_based(state)
    assert out["action"] == "known_image", out


def test_profile_builder_annotates_dataset_match_in_trace(seeded_db):
    from roadguard_agent import profile_builder_node

    state = {
        "yolo_output": {"top1_class": "D20", "top1_confidence": 0.8,
                        "n_detections_by_class": {"D20": 1},
                        "detections": [{"class": "D20", "confidence": 0.8, "area_ratio": 0.1}]},
        "image_quality": {},
        "metadata": {"road_type": "main", "lane_blocked": "none",
                     "image_sha256": "a" * 64, "image_phash": "ffff0000ffff0000",
                     "location": "x"},
        "iterations": 0,
    }
    out = profile_builder_node(state)
    assert out["dataset_match"] is not None
    assert out["dataset_match"]["pic_id"] == 1
    trace = out["trace"][0]
    assert "dataset_lookup" in trace["content"]
    assert trace["content"]["dataset_lookup"]["found"] is True


def test_trace_lines_show_known_image_banner_in_rag_and_decision(seeded_db):
    from roadguard_agent import format_trace_lines

    trace = [
        {
            "iter": 1, "speaker": "profile_builder", "kind": "handoff_out",
            "content": {
                "to": "rag_retriever", "primary_class": "D20", "severity": "mild",
                "confidence": "high", "llm_used": False,
                "dataset_lookup": {
                    "found": True, "match": "sha256", "distance": 0, "pic_id": 1,
                    "split": "train", "file_name": "firc_10001.jpg", "classes": "D20",
                    "n_labels": 1, "model_prediction": "D20 0.42",
                },
            },
        },
        {
            "iter": 2, "speaker": "decision", "kind": "final",
            "content": {
                "action": "known_image", "priority": 0, "source": "rule_based",
                "rationale": "known dataset image", "cited_sources": [],
                "dataset_match": {
                    "found": True, "match": "sha256", "pic_id": 1,
                    "split": "train", "file_name": "firc_10001.jpg", "classes": "D20",
                    "n_labels": 1, "model_prediction": "D20 0.42",
                },
            },
        },
    ]
    lines = format_trace_lines("hz_k", trace, status="open",
                                verdict={"action": "known_image", "priority": 0})
    text = "\n".join(lines)
    assert "dataset: KNOWN IMAGE pic_id=1" in text
    assert "catalogued: pic_id=1" in text
    assert "[DONE    ] action=KNOWN_IMAGE" in text


# ---------------------------------------------------------------- API

@pytest.fixture()
def client(seeded_db, passphrase_hash, monkeypatch):
    from roadguard_agent import reset_settings_cache
    from roadguard_api import create_app

    monkeypatch.setenv("ROADGUARD_PASSPHRASE_HASH", passphrase_hash)
    reset_settings_cache()
    c = TestClient(create_app())
    r = c.post("/auth/login", json={"passphrase": "test-passphrase"})
    assert r.status_code == 200, r.text
    c.headers["Authorization"] = f"Bearer {r.json()['token']}"
    return c


def test_sql_endpoint_returns_rows_for_select_and_blocks_non_select(client):
    assert client.post("/sql", json={"query": "SELECT 1"}).status_code == 404  # not private
    priv = {"host": "127.0.0.1"}
    ok = client.post("/sql", json={"query": "select * from df where pic_id = 1"},
                     headers=priv)
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["row_count"] == 1
    assert "pic_id" in body["columns"]
    bad = client.post("/sql", json={"query": "delete from hazards"}, headers=priv)
    assert bad.status_code == 400
    semi = client.post("/sql", json={"query": "select 1; select 2"}, headers=priv)
    assert semi.status_code == 400


def test_status_banner_shows_dataset_counts(client):
    r = client.get("/status", headers={"host": "127.0.0.1"})
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["images"]["count"] == 2
    assert d["images"]["by_split"] == {"train": 1, "test": 1}
    assert d["images"]["with_predictions"] == 2
    text = "\n".join(r.json()["lines"])
    assert "dataset   2 / 10,000" in text and "with model predictions" in text


def test_submit_async_carries_image_hashes_and_catalogue_hit_yields_known_image(client):
    body = {
        "class": "D20", "confidence": 0.8,
        "bboxes": [[0, 0, 100, 100, "D20", 0.8]],
        "road_type": "main", "lane_blocked": "none", "location": "x",
        "image_sha256": "a" * 64, "image_phash": "ffff0000ffff0000",
    }
    r = client.post("/hazards/submit-async", json=body)
    assert r.status_code == 202, r.text
    hazard_id = r.json()["hazard_id"]
    rep = client.get(f"/report/{hazard_id}").json()
    assert rep["verdict"]["action"] == "known_image"
    assert any(
        "KNOWN IMAGE pic_id=1" in line for line in rep["trace_lines"]
    ), "status banner missing from trace lines"


def test_submit_async_rejects_wrongly_shaped_hash(client):
    r = client.post("/hazards/submit-async", json={
        "class": "D20", "confidence": 0.8, "bboxes": [],
        "road_type": "main", "lane_blocked": "none",
        "image_sha256": "not-a-hash",
    })
    assert r.status_code == 422
