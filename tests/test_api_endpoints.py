"""End-to-end endpoint tests through FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import reset_settings_cache


@pytest.fixture()
def client(passphrase_hash, monkeypatch):
    monkeypatch.setenv("ROADGUARD_PASSPHRASE_HASH", passphrase_hash)
    reset_settings_cache()
    app = create_app()
    return TestClient(app)


def _authed(client, passphrase="test-passphrase"):
    resp = client.post("/auth/login", json={"passphrase": passphrase})
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


def test_health_is_degraded_without_llm_or_fallback(client):
    data = client.get("/health").json()
    assert data["status"] == "degraded"
    assert data["version"] == "0.1.0"


def test_login_without_body_rejected(client):
    r = client.post("/auth/login", json={})
    assert r.status_code == 400
    assert r.json()["detail"] == "passphrase_or_google_id_token_required"


def test_login_with_bad_passphrase_401(client):
    r = client.post("/auth/login", json={"passphrase": "wrong"})
    assert r.status_code == 401
    assert r.json()["detail"] == "passphrase_invalid"


def test_login_success_and_me(client):
    token = _authed(client)
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    data = me.json()
    assert data["email"] == "orarbeli1@gmail.com"
    assert data["sub"].startswith("passphrase:")


def test_me_requires_bearer(client):
    r = client.get("/auth/me")
    assert r.status_code == 401
    assert r.json()["detail"] == "missing_bearer_token"


def test_submit_low_confidence_resolves_to_human_review(client):
    token = _authed(client)
    r = client.post(
        "/hazards/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "class": "D40",
            "confidence": 0.3,
            "bboxes": [],
            "road_type": "main",
            "lane_blocked": "full",
            "image_object_key": "obj.jpg",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"]["action"] == "human_review"
    assert body["fallback_reason"] == "low_confidence"


def test_submit_high_conf_without_llm_resolves_to_human_review(client):
    token = _authed(client)
    r = client.post(
        "/hazards/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "class": "D40",
            "confidence": 0.9,
            "bboxes": [],
            "road_type": "main",
            "lane_blocked": "full",
            "image_object_key": "obj.jpg",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"]["action"] == "human_review"
    assert body["fallback_reason"] == "primary_llm_unavailable"


def test_history_endpoint_stub_until_db_wired(client):
    token = _authed(client)
    r = client.get("/hazards/history", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 501
    assert r.json()["detail"] == "db_not_wired_yet"


def test_rag_endpoint_stub_until_chromadb_wired(client):
    token = _authed(client)
    r = client.post(
        "/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"hazard_class": "D40", "road_type": "main"},
    )
    assert r.status_code == 501
    assert r.json()["detail"] == "rag_not_wired_yet"


def test_submit_rejects_bad_class(client):
    token = _authed(client)
    r = client.post(
        "/hazards/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "class": "Z99",
            "confidence": 0.9,
            "bboxes": [],
            "road_type": "main",
            "lane_blocked": "full",
            "image_object_key": "obj.jpg",
        },
    )
    assert r.status_code == 422
