"""Step-1 auth upgrade: password rules, user cap, per-user submit cap,
new endpoints, bootstrap script. Keep every test deterministic and
isolated via the in-memory DB from conftest."""

from __future__ import annotations
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def strong_password() -> str:
    return "Str0ng!Pass"


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    """A FastAPI client with signup enabled + password login enabled.

    Uses an on-disk SQLite file so every Database() connection in the API
    sees the same rows (`:memory:` would make each request a fresh DB).
    """
    monkeypatch.setenv("ROADGUARD_DB_PATH", str(tmp_path / "auth_step1.db"))
    monkeypatch.setenv("ROADGUARD_SIGNUP_ENABLED", "1")
    monkeypatch.setenv("ROADGUARD_PASSWORD_LOGIN_ENABLED", "1")
    monkeypatch.setenv("ROADGUARD_MAX_USERS", "10")
    monkeypatch.setenv("ROADGUARD_USER_DAILY_SUBMIT_CAP", "100")
    from roadguard_agent import reset_settings_cache
    from roadguard_api import create_app
    reset_settings_cache()
    return TestClient(create_app())


@pytest.fixture()
def _file_db(monkeypatch, tmp_path):
    """Helper for tests that build their own TestClient (not api_client)."""
    monkeypatch.setenv("ROADGUARD_DB_PATH", str(tmp_path / "step1.db"))


def test_password_policy_accepts_strong_passwords():
    from roadguard_agent import validate_password
    for pw in ("Str0ng!Pass", "Abcdefg1!", "LongEnough9#"):
        ok, reason = validate_password(pw)
        assert ok, f"{pw} rejected with {reason}"


def test_password_policy_rejects_weak_passwords():
    from roadguard_agent import validate_password
    cases = [
        ("short1!", "password_too_short"),
        ("alllowercase1!", "password_missing_uppercase"),
        ("ALLUPPERCASE1!", "password_missing_lowercase"),
        ("NoSpecial123", "password_missing_special"),
        ("", "password_too_short"),
    ]
    for pw, expected in cases:
        ok, reason = validate_password(pw)
        assert not ok and reason == expected, f"{pw}: got {reason}"


def test_username_policy():
    from roadguard_agent import validate_username
    for ok_name in ("orpsp8", "or.arbeli", "benny_1", "shai-model"):
        assert validate_username(ok_name)[0]
    for bad_name in ("", "ab", "has space", "weird!chars", "1" * 100, "-starts_dash"):
        assert not validate_username(bad_name)[0], f"{bad_name} should be invalid"


def test_auth_config_reports_state(api_client):
    r = api_client.get("/auth/config")
    assert r.status_code == 200
    data = r.json()
    assert data["signup_enabled"] is True
    assert data["password_login_enabled"] is True
    assert data["max_users"] == 10
    assert data["user_count"] == 0


def test_register_creates_user_and_returns_jwt(api_client, strong_password):
    r = api_client.post("/auth/register", json={
        "username": "testuser1", "password": strong_password,
        "email": "a@b.co",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "token" in body and body["expires_in"] > 0
    cfg = api_client.get("/auth/config").json()
    assert cfg["user_count"] == 1


def test_register_rejects_weak_password(api_client):
    r = api_client.post("/auth/register", json={
        "username": "weakpw", "password": "short",
    })
    assert r.status_code == 400
    assert r.json()["detail"] == "password_too_short"


def test_register_rejects_duplicate_username(api_client, strong_password):
    r1 = api_client.post("/auth/register", json={
        "username": "dup", "password": strong_password,
    })
    assert r1.status_code == 200
    r2 = api_client.post("/auth/register", json={
        "username": "dup", "password": strong_password,
    })
    assert r2.status_code == 409
    assert r2.json()["detail"] == "username_taken"


def test_register_enforces_user_cap(_file_db, monkeypatch, strong_password):
    monkeypatch.setenv("ROADGUARD_SIGNUP_ENABLED", "1")
    monkeypatch.setenv("ROADGUARD_MAX_USERS", "3")
    from roadguard_agent import reset_settings_cache
    from roadguard_api import create_app
    reset_settings_cache()
    client = TestClient(create_app())
    for i in range(3):
        r = client.post("/auth/register", json={
            "username": f"user{i}", "password": strong_password,
        })
        assert r.status_code == 200
    r_overflow = client.post("/auth/register", json={
        "username": "user_overflow", "password": strong_password,
    })
    assert r_overflow.status_code == 403
    assert r_overflow.json()["detail"] == "user_cap_reached"


def test_register_disabled_in_prod(_file_db, monkeypatch, strong_password):
    monkeypatch.setenv("ROADGUARD_SIGNUP_ENABLED", "0")
    monkeypatch.setenv("ROADGUARD_PASSWORD_LOGIN_ENABLED", "1")
    from roadguard_agent import reset_settings_cache
    from roadguard_api import create_app
    reset_settings_cache()
    client = TestClient(create_app())
    r = client.post("/auth/register", json={
        "username": "unauth", "password": strong_password,
    })
    assert r.status_code == 403
    assert r.json()["detail"] == "signup_disabled"


def test_login_password_round_trip(api_client, strong_password):
    api_client.post("/auth/register", json={
        "username": "loginme", "password": strong_password,
    })
    r = api_client.post("/auth/login/password", json={
        "username": "loginme", "password": strong_password,
    })
    assert r.status_code == 200, r.text
    assert "token" in r.json()


def test_login_password_rejects_wrong_password(api_client, strong_password):
    api_client.post("/auth/register", json={
        "username": "loginme2", "password": strong_password,
    })
    r = api_client.post("/auth/login/password", json={
        "username": "loginme2", "password": strong_password + "X",
    })
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_credentials"


def test_login_password_unknown_user_returns_401_not_404(api_client, strong_password):
    r = api_client.post("/auth/login/password", json={
        "username": "nosuchuser", "password": strong_password,
    })
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_credentials"


def test_login_password_disabled(_file_db, monkeypatch, strong_password):
    monkeypatch.setenv("ROADGUARD_PASSWORD_LOGIN_ENABLED", "0")
    monkeypatch.setenv("ROADGUARD_SIGNUP_ENABLED", "0")
    from roadguard_agent import reset_settings_cache
    from roadguard_api import create_app
    reset_settings_cache()
    client = TestClient(create_app())
    r = client.post("/auth/login/password", json={
        "username": "someuser", "password": strong_password,
    })
    assert r.status_code == 403
    assert r.json()["detail"] == "password_login_disabled"


def test_me_shows_uploads_and_admin_flag(api_client, strong_password):
    reg = api_client.post("/auth/register", json={
        "username": "metest", "password": strong_password,
    })
    token = reg.json()["token"]
    r = api_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "metest"
    assert body["is_admin"] is False
    assert body["is_internal"] is False
    assert body["uploads_today"] == 0
    assert body["uploads_total"] == 0
    assert body["daily_cap"] == 100
    assert body["operator_urls"] == []


def test_me_admin_gets_operator_urls(_file_db, monkeypatch, strong_password):
    monkeypatch.setenv("ROADGUARD_SIGNUP_ENABLED", "1")
    monkeypatch.setenv("ROADGUARD_PASSWORD_LOGIN_ENABLED", "1")
    from roadguard_agent import Database, reset_settings_cache, get_settings
    from roadguard_api import create_app
    reset_settings_cache()
    client = TestClient(create_app())
    reg = client.post("/auth/register", json={
        "username": "admin1", "password": strong_password,
    })
    token = reg.json()["token"]
    s = get_settings()
    with Database(s.roadguard_db_path) as db:
        u = db.get_user_by_username("admin1")
        db.create_user_full(
            user_id=u.id, username=u.username, email=u.email,
            display_name=u.display_name,
            password_hash=u.password_hash,
            google_sub=None, is_admin=True, is_internal=False,
        )
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_admin"] is True
    assert any("/digest/latest" in url for url in body["operator_urls"])
    assert any("/trace" in url for url in body["operator_urls"])


def test_logout_returns_ok(api_client, strong_password):
    reg = api_client.post("/auth/register", json={
        "username": "logoutme", "password": strong_password,
    })
    token = reg.json()["token"]
    r = api_client.post("/auth/logout",
                        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_user_daily_submit_cap_limits_regular_user(_file_db, monkeypatch, strong_password):
    monkeypatch.setenv("ROADGUARD_SIGNUP_ENABLED", "1")
    monkeypatch.setenv("ROADGUARD_USER_DAILY_SUBMIT_CAP", "2")
    from roadguard_agent import (
        Database, Hazard, reset_settings_cache, get_settings,
    )
    from roadguard_api import create_app
    import time as _t
    reset_settings_cache()
    client = TestClient(create_app())
    reg = client.post("/auth/register", json={
        "username": "capme", "password": strong_password,
    })
    token = reg.json()["token"]
    s = get_settings()
    with Database(s.roadguard_db_path) as db:
        u = db.get_user_by_username("capme")
        # The API submits as sub="user:<id>" and upserts that row on first
        # submission; mimic the state after the user already submitted twice.
        sub_user_id = f"user:{u.id}"
        db.upsert_user(sub_user_id, u.email, u.display_name)
        for i in range(2):
            db.create_hazard(Hazard(
                id=f"hz_cap{i:04d}", user_id=sub_user_id,
                reported_at=_t.time(), hazard_class="D40", confidence=0.9,
                road_type="main", lane_blocked="none", location="x",
                bboxes_json="[]", priority=0, agent_decision="standard",
                agent_rationale="", agent_trace_json="[]", status="open",
                work_order_id=None, cited_sources_json="[]",
                reporter_text="", agent_assessment="",
            ))
    body = {
        "hazard_class": "D40", "confidence": 0.9,
        "bboxes": [[0, 0, 100, 100, "D40", 0.9]],
        "road_type": "main", "lane_blocked": "none",
    }
    r = client.post("/hazards/submit-async", json=body,
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 429
    assert "daily_submit_cap_reached" in r.json()["detail"]


def test_internal_user_bypasses_cap(_file_db, monkeypatch, strong_password):
    monkeypatch.setenv("ROADGUARD_SIGNUP_ENABLED", "1")
    monkeypatch.setenv("ROADGUARD_USER_DAILY_SUBMIT_CAP", "1")
    from roadguard_agent import (
        Database, Hazard, reset_settings_cache, get_settings,
    )
    from roadguard_api import create_app
    import time as _t
    reset_settings_cache()
    client = TestClient(create_app())
    reg = client.post("/auth/register", json={
        "username": "batchy", "password": strong_password,
    })
    token = reg.json()["token"]
    s = get_settings()
    with Database(s.roadguard_db_path) as db:
        u = db.get_user_by_username("batchy")
        # Flip the registration row to internal.
        db.create_user_full(
            user_id=u.id, username=u.username, email=u.email,
            display_name=u.display_name,
            password_hash=u.password_hash,
            google_sub=None, is_admin=False, is_internal=True,
        )
        # Also upsert the sub-shaped row the API will see at submit time.
        sub_user_id = f"user:{u.id}"
        db.create_user_full(
            user_id=sub_user_id, username=None, email=u.email,
            display_name=u.display_name,
            password_hash=None, google_sub=None,
            is_admin=False, is_internal=True,
        )
        db.create_hazard(Hazard(
            id="hz_internal0", user_id=sub_user_id,
            reported_at=_t.time(), hazard_class="D40", confidence=0.9,
            road_type="main", lane_blocked="none", location="x",
            bboxes_json="[]", priority=0, agent_decision="standard",
            agent_rationale="", agent_trace_json="[]", status="open",
            work_order_id=None, cited_sources_json="[]",
            reporter_text="", agent_assessment="",
        ))
    body = {
        "hazard_class": "D40", "confidence": 0.9,
        "bboxes": [[0, 0, 100, 100, "D40", 0.9]],
        "road_type": "main", "lane_blocked": "none",
    }
    r = client.post("/hazards/submit-async", json=body,
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 202, r.text
