"""A registered user's token subject is ``user:<id>``; the hazards row and
every lookup use the bare ``<id>``. 31.10.2026: the owner's first upload of
the evening upserted ``user:u_owner_orpsp8`` as a second, non-admin users
row, /auth/me resolved to it, and the operator links vanished while the
upload count moved to the shadow row.
"""

from __future__ import annotations

import time

import roadguard_agent as ra
from roadguard_api import _user_id_of, merge_prefixed_user_rows


def test_user_id_of_strips_the_registered_prefix():
    assert _user_id_of({"sub": "user:u_owner_orpsp8"}) == "u_owner_orpsp8"
    assert _user_id_of({"sub": "passphrase:owner@example.com"}) == "passphrase:owner@example.com"
    assert _user_id_of({"sub": "1234567890"}) == "1234567890"  # a Google subject
    assert _user_id_of({"email": "x@example.com"}) == "x@example.com"
    assert _user_id_of({}) == "anonymous"


def _hazard(hid: str, user_id: str) -> ra.Hazard:
    return ra.Hazard(
        id=hid, user_id=user_id, reported_at=time.time(), hazard_class="D40",
        confidence=0.9, road_type="main", lane_blocked="none", location="x",
        bboxes_json="[]", priority=0, agent_decision="standard", agent_rationale="",
        agent_trace_json="[]", status="open", work_order_id=None,
        cited_sources_json="[]", reporter_text="", agent_assessment="",
    )


def test_shadow_rows_fold_into_the_real_user(tmp_path):
    db_path = str(tmp_path / "rg.db")
    with ra.Database(db_path) as db:
        db.create_user_full(
            user_id="u_owner", username="owner", email="o@example.com",
            display_name="Owner", password_hash="x", google_sub=None,
            is_admin=True, is_internal=False,
        )
        db.upsert_user("user:u_owner", "o@example.com", "Owner")  # the old submit path
        db.create_hazard(_hazard("hz_1", "user:u_owner"))
        db.upsert_user("passphrase:o@example.com", "o@example.com", "")  # legacy row, kept
    assert merge_prefixed_user_rows(db_path) == 1
    with ra.Database(db_path) as db:
        assert db.get_user("user:u_owner") is None
        owner = db.get_user("u_owner")
        assert owner is not None and owner.is_admin is True
        assert db.count_user_hazards_since("u_owner", 0.0) == 1
        assert db.get_user("passphrase:o@example.com") is not None
    assert merge_prefixed_user_rows(db_path) == 0  # idempotent


def test_google_sign_in_lands_on_the_registered_admin(tmp_path, monkeypatch):
    """The owner signs in with Google using the e-mail of his registered
    admin account: /auth/me must show that account, operator links included,
    and the Google subject is linked to the row for the next time."""
    import roadguard_api
    from fastapi.testclient import TestClient

    db_path = str(tmp_path / "rg.db")
    monkeypatch.setenv("ROADGUARD_DB_PATH", db_path)
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com")
    monkeypatch.setenv("ROADGUARD_ALLOWED_EMAILS", "o@example.com")
    ra.reset_settings_cache()
    with ra.Database(db_path) as db:
        db.upsert_user("passphrase:o@example.com", "o@example.com", "")  # legacy row, same e-mail
        db.create_user_full(
            user_id="u_owner", username="owner", email="o@example.com",
            display_name="Owner", password_hash="x", google_sub=None,
            is_admin=True, is_internal=False,
        )
    monkeypatch.setattr(
        roadguard_api, "verify_google_id_token",
        lambda **kw: {"sub": "g-123", "email": "o@example.com", "name": "Owner"},
    )
    client = TestClient(roadguard_api.create_app())
    r = client.post("/auth/login", json={"google_id_token": "signed-by-google"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    body = me.json()
    assert body["username"] == "owner" and body["is_admin"] is True
    assert body["sub"] == "user:u_owner"
    assert any(u.endswith("/trace") for u in body["operator_urls"])
    with ra.Database(db_path) as db:
        assert db.get_user("u_owner").google_sub == "g-123"
        assert db.get_user_by_google_sub("g-123").id == "u_owner"
