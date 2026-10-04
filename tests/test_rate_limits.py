"""Rate limiting via slowapi (@limiter.limit decorators)."""

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


def _login(client, body=None):
    return client.post("/auth/login", json=body or {"passphrase": "test-passphrase"})


def test_login_endpoint_limits_at_10_per_minute(client):
    results = []
    for _ in range(12):
        results.append(_login(client).status_code)
    assert 429 in results, results
    first_429 = results.index(429)
    assert first_429 >= 10


def test_submit_rate_limit_hits_after_bursts(client):
    token = _login(client).json()["token"]
    hits = []
    for _ in range(22):
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
        hits.append(r.status_code)
    assert 429 in hits
