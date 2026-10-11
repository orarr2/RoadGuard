"""Streaming /hazards/live endpoint: dedup via decision cache, body shape, limits."""

import pytest
from fastapi.testclient import TestClient

from api.decision_cache import reset_decision_cache
from api.main import create_app
from api.settings import reset_settings_cache


@pytest.fixture()
def client(passphrase_hash, monkeypatch):
    monkeypatch.setenv("ROADGUARD_PASSPHRASE_HASH", passphrase_hash)
    # Force the multiagent path so the cache test exercises the real
    # deterministic graph instead of hitting the (missing) Groq provider.
    monkeypatch.setenv("AGENT_ARCHITECTURE", "multiagent")
    reset_settings_cache()
    reset_decision_cache()
    app = create_app()
    return TestClient(app)


def _authed(client, passphrase="test-passphrase"):
    r = client.post("/auth/login", json={"passphrase": passphrase})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _live_body(**overrides):
    body = {
        "session_id": "sess-abc-123",
        "track_id": "t-1",
        "class": "D40",
        "confidence": 0.92,
        "top1_margin": 0.5,
        "area_ratio": 0.04,
        "n_detections_by_class": {"D40": 1, "D20": 0, "D00": 0, "D10": 0},
        "bbox": [100.0, 200.0, 220.0, 320.0],
        "center_y_rel": 0.65,
        "road_type": "main",
        "lane_blocked": "full",
        "location": "Rothschild 20",
        "image_quality": {
            "blur_score": 180.0,
            "brightness_mean": 0.55,
            "is_road_scene": True,
            "road_scene_confidence": 0.95,
            "occlusion_estimate": "low",
            "ocr_text_detected": "",
        },
    }
    body.update(overrides)
    return body


def test_live_submit_requires_bearer(client):
    r = client.post("/hazards/live", json=_live_body())
    assert r.status_code == 401


def test_live_submit_returns_verdict_and_cache_key(client):
    token = _authed(client)
    r = client.post(
        "/hazards/live",
        headers={"Authorization": f"Bearer {token}"},
        json=_live_body(),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["verdict"]["action"] in {
        "urgent",
        "standard",
        "needs_more_evidence",
        "human_review",
    }
    assert data["cached"] is False
    assert data["cache_key"].startswith("sess-abc-123|t-1|D40|main|full")


def test_live_submit_second_call_is_cached(client):
    token = _authed(client)
    first = client.post(
        "/hazards/live",
        headers={"Authorization": f"Bearer {token}"},
        json=_live_body(),
    )
    second = client.post(
        "/hazards/live",
        headers={"Authorization": f"Bearer {token}"},
        json=_live_body(),
    )
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    # Same verdict shape
    assert (
        first.json()["verdict"]["action"] == second.json()["verdict"]["action"]
    )
    # Different hazard_ids (one per submission even on cache hit)
    assert first.json()["hazard_id"] != second.json()["hazard_id"]


def test_live_submit_different_track_id_misses_cache(client):
    token = _authed(client)
    client.post(
        "/hazards/live",
        headers={"Authorization": f"Bearer {token}"},
        json=_live_body(track_id="t-1"),
    )
    other = client.post(
        "/hazards/live",
        headers={"Authorization": f"Bearer {token}"},
        json=_live_body(track_id="t-2"),
    )
    assert other.json()["cached"] is False


def test_live_submit_rejects_short_session_id(client):
    token = _authed(client)
    r = client.post(
        "/hazards/live",
        headers={"Authorization": f"Bearer {token}"},
        json=_live_body(session_id="abc"),  # under 4 chars
    )
    assert r.status_code == 422


def test_live_submit_rejects_bad_class(client):
    token = _authed(client)
    r = client.post(
        "/hazards/live",
        headers={"Authorization": f"Bearer {token}"},
        json=_live_body(**{"class": "Z99"}),
    )
    assert r.status_code == 422


def test_live_submit_rate_limit_fires(client):
    token = _authed(client)
    hits = []
    for i in range(95):
        r = client.post(
            "/hazards/live",
            headers={"Authorization": f"Bearer {token}"},
            json=_live_body(track_id=f"t-{i}"),
        )
        hits.append(r.status_code)
    assert 429 in hits
