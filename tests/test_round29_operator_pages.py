"""Round 29 (6.11.2026 audit, F3 and F15): a run the service restart killed
is marked on /report and in the digest; the operator overview counts the
fleet's last 24 hours on a private host; the KPI is named for what it
measures (agreement with the rule tree, not accuracy)."""
import json
import time

import pytest
from fastapi.testclient import TestClient

INTERRUPTED_TRACE = [
    {"iter": 0, "speaker": "sanitize", "kind": "boundary_defence", "ts": 1.0, "content": {"known_image_shortcut": False}},
    {"iter": 1, "speaker": "pipeline", "kind": "graph_interrupted", "ts": 2.0,
     "content": {"reason": "service restarted while the run was in progress; routed to a human"}},
]
NORMAL_TRACE = [
    {"iter": 0, "speaker": "sanitize", "kind": "boundary_defence", "ts": 1.0, "content": {"known_image_shortcut": False}},
    {"iter": 1, "speaker": "decision", "kind": "final", "ts": 2.0,
     "content": {"action": "human_review", "priority": 50, "source": "rule_based", "rationale": "r", "cited_sources": []}},
]
PRIVATE = {"host": "127.0.0.1"}


@pytest.fixture()
def client(tmp_path, monkeypatch, passphrase_hash):
    from roadguard_agent import reset_settings_cache
    from roadguard_api import create_app

    monkeypatch.setenv("ROADGUARD_DB_PATH", str(tmp_path / "ops.db"))
    monkeypatch.setenv("ROADGUARD_RAG_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ROADGUARD_PASSPHRASE_HASH", passphrase_hash)
    reset_settings_cache()
    c = TestClient(create_app())
    r = c.post("/auth/login", json={"passphrase": "test-passphrase"})
    assert r.status_code == 200, r.text
    c.headers["Authorization"] = f"Bearer {r.json()['token']}"
    return c


def _seed(hz_id, reported_at, trace):
    from roadguard_agent import Database, Hazard, get_settings

    with Database(get_settings().roadguard_db_path) as db:
        db.upsert_user("u_ops", "ops@example.com", "Ops")
        db.create_hazard(Hazard(
            id=hz_id, user_id="u_ops", reported_at=reported_at, hazard_class="D40", confidence=0.7,
            road_type="main", lane_blocked="none", location="x", bboxes_json="[]", priority=50,
            agent_decision="human_review",
            agent_rationale="service restarted while the run was in progress; routed to a human",
            agent_trace_json=json.dumps(trace), status="open", work_order_id=None,
            cited_sources_json="[]", reporter_text="", agent_assessment="",
        ))


def test_report_marks_an_interrupted_run(client):
    _seed("hz_int0001", time.time(), INTERRUPTED_TRACE)
    _seed("hz_norm0001", time.time(), NORMAL_TRACE)
    assert client.get("/report/hz_int0001", headers=PRIVATE).json()["verdict"]["interrupted"] is True
    assert client.get("/report/hz_norm0001", headers=PRIVATE).json()["verdict"]["interrupted"] is False


def test_digest_tags_an_interrupted_run(client):
    _seed("hz_int0002", time.time(), INTERRUPTED_TRACE)
    _seed("hz_norm0002", time.time(), NORMAL_TRACE)
    page = client.get("/digest/latest", headers=PRIVATE)
    assert page.status_code == 200
    assert page.text.count(">interrupted</span>") == 1
    assert "no agent analysis" in page.text


def test_operator_overview_counts_the_fleet_window_on_a_private_host(client):
    now = time.time()
    _seed("hz_today0001", now - 600, NORMAL_TRACE)
    _seed("hz_old0001", now - 48 * 3600, NORMAL_TRACE)
    fleet = client.get("/report", headers=PRIVATE).json()
    assert fleet["window_hours"] == 24
    assert fleet["total_hazards"] == 1
    assert fleet["by_action"] == {"human_review": 1}
    own = client.get("/report").json()
    assert own["window_hours"] is None
    assert own["total_hazards"] == 0, "the citizen view stays per user"


def test_kpi_is_named_for_what_it_measures(client):
    _seed("hz_kpi0001", time.time(), NORMAL_TRACE)
    d = client.get("/stats/kpi?window_hours=24", headers=PRIVATE).json()
    keys = [k["key"] for k in d["kpis"]]
    assert "rule_tree_agreement" in keys and "correct_action" not in keys
    first = next(k for k in d["kpis"] if k["key"] == "rule_tree_agreement")
    assert "not accuracy" in first["label"]
