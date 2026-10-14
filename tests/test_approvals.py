"""Approval gate: DB CRUD, HTML page, decide endpoint, graph wiring."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import reset_settings_cache
from db.models import Database


# ---------------------------------------------------------------------------
# Database CRUD
# ---------------------------------------------------------------------------


def test_create_and_get_approval_round_trip():
    db = Database(":memory:")
    try:
        apr_id = db.create_approval(
            hazard_id="hz_x",
            action="urgent",
            priority=90,
            rationale="D40 blocking lane on main",
            cited_sources_json='["sec-4.3"]',
        )
        assert apr_id.startswith("apr_")
        row = db.get_approval(apr_id)
        assert row is not None
        assert row.hazard_id == "hz_x"
        assert row.action == "urgent"
        assert row.priority == 90
        assert row.status == "pending"
        assert row.decided_at is None
    finally:
        db.close()


def test_decide_approval_marks_approved():
    db = Database(":memory:")
    try:
        apr_id = db.create_approval(
            hazard_id="hz_x", action="urgent", priority=90, rationale="x"
        )
        ok = db.decide_approval(apr_id, "approved", decided_by="alice")
        assert ok is True
        row = db.get_approval(apr_id)
        assert row.status == "approved"
        assert row.decided_by == "alice"
        assert row.decided_at is not None
    finally:
        db.close()


def test_decide_approval_denies():
    db = Database(":memory:")
    try:
        apr_id = db.create_approval(
            hazard_id="hz_x", action="human_review", priority=50, rationale="x"
        )
        ok = db.decide_approval(apr_id, "denied", decided_by="bob")
        assert ok is True
        assert db.get_approval(apr_id).status == "denied"
    finally:
        db.close()


def test_decide_approval_rejects_second_decision():
    db = Database(":memory:")
    try:
        apr_id = db.create_approval(
            hazard_id="hz_x", action="urgent", priority=90, rationale="x"
        )
        assert db.decide_approval(apr_id, "approved", decided_by="alice") is True
        # second attempt should fail because row is no longer pending
        assert db.decide_approval(apr_id, "denied", decided_by="bob") is False
        assert db.get_approval(apr_id).status == "approved"
    finally:
        db.close()


def test_decide_approval_rejects_invalid_decision():
    db = Database(":memory:")
    try:
        apr_id = db.create_approval(
            hazard_id="hz_x", action="urgent", priority=90, rationale="x"
        )
        with pytest.raises(ValueError):
            db.decide_approval(apr_id, "maybe", decided_by="x")
    finally:
        db.close()


def test_decide_approval_returns_false_for_missing_id():
    db = Database(":memory:")
    try:
        assert db.decide_approval("apr_nope", "approved", decided_by="x") is False
    finally:
        db.close()


def test_list_pending_approvals_returns_only_pending():
    db = Database(":memory:")
    try:
        a = db.create_approval(
            hazard_id="hz_a", action="urgent", priority=90, rationale="x"
        )
        b = db.create_approval(
            hazard_id="hz_b", action="urgent", priority=50, rationale="x"
        )
        db.decide_approval(a, "approved", decided_by="x")
        pending = db.list_pending_approvals()
        pending_ids = [p.id for p in pending]
        assert b in pending_ids
        assert a not in pending_ids
    finally:
        db.close()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(passphrase_hash, tmp_path, monkeypatch):
    monkeypatch.setenv("ROADGUARD_PASSPHRASE_HASH", passphrase_hash)
    monkeypatch.setenv("ROADGUARD_DB_PATH", str(tmp_path / "test.db"))
    reset_settings_cache()
    return TestClient(create_app())


def _seed_approval(db_path: str, status: str = "pending", **overrides) -> str:
    with Database(db_path) as db:
        base = dict(
            hazard_id="hz_demo",
            action="urgent",
            priority=89,
            rationale="D40 pothole on main road",
            cited_sources_json='["sec-4.3"]',
        )
        base.update(overrides)
        apr_id = db.create_approval(**base)
        if status != "pending":
            db.decide_approval(apr_id, status, decided_by="tester")
        return apr_id


def test_approval_page_404_for_missing_id(client):
    r = client.get("/approvals/apr_doesnotexist")
    assert r.status_code == 404


def test_approval_page_shows_pending_buttons(client, tmp_path):
    apr_id = _seed_approval(str(tmp_path / "test.db"))
    r = client.get(f"/approvals/{apr_id}")
    assert r.status_code == 200
    body = r.text
    assert "Approve" in body
    assert "Deny" in body
    assert apr_id in body


def test_approval_decide_approve_flow(client, tmp_path):
    apr_id = _seed_approval(str(tmp_path / "test.db"))
    r = client.post(
        f"/approvals/{apr_id}/decide",
        data={"decision": "approved"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].endswith(f"/approvals/{apr_id}")

    page = client.get(f"/approvals/{apr_id}")
    assert "approved" in page.text.lower()
    assert "Approve</button>" not in page.text  # buttons gone


def test_approval_decide_deny_flow(client, tmp_path):
    apr_id = _seed_approval(str(tmp_path / "test.db"))
    r = client.post(
        f"/approvals/{apr_id}/decide",
        data={"decision": "denied"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    page = client.get(f"/approvals/{apr_id}")
    assert "denied" in page.text.lower()


def test_approval_decide_rejects_invalid_decision(client, tmp_path):
    apr_id = _seed_approval(str(tmp_path / "test.db"))
    r = client.post(
        f"/approvals/{apr_id}/decide",
        data={"decision": "maybe"},
    )
    assert r.status_code == 400


def test_approval_decide_rejects_second_decision(client, tmp_path):
    apr_id = _seed_approval(str(tmp_path / "test.db"))
    r1 = client.post(
        f"/approvals/{apr_id}/decide",
        data={"decision": "approved"},
        follow_redirects=False,
    )
    assert r1.status_code == 303
    r2 = client.post(
        f"/approvals/{apr_id}/decide",
        data={"decision": "denied"},
    )
    assert r2.status_code == 409


# ---------------------------------------------------------------------------
# Multiagent approval_gate node writes to DB when env points at one
# ---------------------------------------------------------------------------


def test_approval_gate_writes_row(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADGUARD_DB_PATH", str(tmp_path / "ga.db"))
    reset_settings_cache()
    from agent.multiagent.adapter import multiagent_decide

    dec = multiagent_decide(
        {
            "hazard_id": "hz_live_approve",
            "class": "D40",
            "confidence": 0.91,
            "bboxes": [[120, 240, 340, 460, "D40", 0.91]],
            "road_type": "main",
            "lane_blocked": "partial",
            "location": "Allenby 42",
        }
    )
    assert dec.action == "urgent"

    with Database(str(tmp_path / "ga.db")) as db:
        pending = db.list_pending_approvals()
    assert len(pending) == 1
    assert pending[0].hazard_id == "hz_live_approve"
    assert pending[0].action == "urgent"
