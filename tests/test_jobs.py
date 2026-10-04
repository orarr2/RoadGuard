"""Nightly image cleanup and daily email summary jobs."""

import json
import time
from email.message import EmailMessage

import httpx
import pytest

from api.settings import Settings
from db.models import Database, Hazard
from jobs.email_report import build_summary_html, run as email_run
from jobs.image_cleanup import run as cleanup_run
from storage.supabase_client import SupabaseClient


class StubSmtp:
    def __init__(self) -> None:
        self.sent: list[EmailMessage] = []

    def ehlo(self) -> None:
        pass

    def starttls(self) -> None:
        pass

    def login(self, user, password) -> None:
        pass

    def send_message(self, msg: EmailMessage) -> None:
        self.sent.append(msg)

    def quit(self) -> None:
        pass


def _hazard(**overrides):
    now = time.time()
    base = dict(
        id="hz1",
        user_id="u1",
        reported_at=now,
        hazard_class="D40",
        confidence=0.9,
        road_type="main",
        lane_blocked="full",
        location="x",
        bboxes_json="[]",
        image_object_key="k.jpg",
        image_expires_at=now + 60,
        priority=95,
        agent_decision="urgent",
        agent_rationale="clear pothole",
        agent_trace_json="[]",
        status="open",
        work_order_id=None,
        cited_sources_json=json.dumps(["https://x"]),
    )
    base.update(overrides)
    return Hazard(**base)


def _supabase_mock():
    def handler(req):
        if req.method == "DELETE":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SupabaseClient("https://x.supabase.co", "k", "b", http_client=client)


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        supabase_url="https://x.supabase.co",
        supabase_service_role_key="k",
        supabase_bucket="b",
        roadguard_db_path=":memory:",
        email_address="from@example.com",
        email_app_password="pw",
        report_recipient="to@example.com",
    )


def test_cleanup_deletes_expired_and_clears_key(settings):
    db = Database(":memory:")
    db.upsert_user("u1", "a@b", "A")
    now = time.time()
    db.create_hazard(_hazard(id="exp", image_expires_at=now - 60))
    db.create_hazard(_hazard(id="fresh", image_expires_at=now + 60))
    sb = _supabase_mock()

    counts = cleanup_run(settings, db=db, supabase=sb, now_ts=now)
    assert counts == {"scanned": 1, "deleted": 1, "failed": 0}
    assert db.get_hazard("exp").image_object_key is None
    assert db.get_hazard("fresh").image_object_key is not None


def test_cleanup_handles_no_candidates(settings):
    db = Database(":memory:")
    db.upsert_user("u1", "a@b", "A")
    sb = _supabase_mock()
    counts = cleanup_run(settings, db=db, supabase=sb, now_ts=time.time())
    assert counts["scanned"] == 0


def test_email_html_lists_urgent_but_omits_standard(settings):
    db = Database(":memory:")
    db.upsert_user("u1", "a@b", "A")
    now = time.time()
    db.create_hazard(_hazard(id="h-urgent", agent_decision="urgent"))
    db.create_hazard(_hazard(id="h-std", agent_decision="standard"))
    html = build_summary_html(db, window_hours=24, now_ts=now)
    assert "Daily Summary" in html
    assert "h-urgent" in html
    assert "h-std" not in html


def test_email_dry_run_counts(settings):
    db = Database(":memory:")
    db.upsert_user("u1", "orarbeli1@gmail.com", "Owner")
    db.set_report_opt_in("u1", True)
    now = time.time()
    db.create_hazard(_hazard(id="h-urgent", agent_decision="urgent"))
    out = email_run(settings, db=db, dry_run=True, now_ts=now)
    assert out["sent"] == 0
    assert out["recipients"] >= 1


def test_email_sends_via_injected_smtp(settings):
    db = Database(":memory:")
    db.upsert_user("u1", "orarbeli1@gmail.com", "Owner")
    db.set_report_opt_in("u1", True)
    db.create_hazard(_hazard(id="h-urgent", agent_decision="urgent"))
    smtp = StubSmtp()
    out = email_run(settings, db=db, smtp_client=smtp, now_ts=time.time())
    assert out["sent"] >= 1
    assert smtp.sent and smtp.sent[0]["Subject"].startswith("RoadGuard Daily Summary")
    assert smtp.sent[0]["From"] == "from@example.com"
