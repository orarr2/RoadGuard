"""SQLite layer: schema, CRUD, and agent-tool adapters."""

import json
import time

from db.models import Database, Hazard, SqliteReviewQueue, SqliteWorkOrderStore


def _mk_hazard(**overrides) -> Hazard:
    now = time.time()
    base = dict(
        id="hz_test",
        user_id="u1",
        reported_at=now,
        hazard_class="D40",
        confidence=0.9,
        road_type="main",
        lane_blocked="full",
        location="x",
        bboxes_json="[]",
        image_object_key="k.jpg",
        image_expires_at=now + 3600,
        priority=95,
        agent_decision="urgent",
        agent_rationale="r",
        agent_trace_json="[]",
        status="open",
        work_order_id=None,
        cited_sources_json=json.dumps(["https://x"]),
    )
    base.update(overrides)
    return Hazard(**base)


def test_schema_applies_idempotently():
    db1 = Database(":memory:")
    db1.close()
    db2 = Database(":memory:")
    db2.close()


def test_user_upsert_and_get():
    db = Database(":memory:")
    try:
        db.upsert_user("u1", "a@b", "Alice")
        got = db.get_user("u1")
        assert got.email == "a@b"
        assert got.display_name == "Alice"
        assert got.opted_in_to_report is False
        db.set_report_opt_in("u1", True)
        got = db.get_user("u1")
        assert got.opted_in_to_report is True
    finally:
        db.close()


def test_hazard_create_and_scoped_fetch():
    db = Database(":memory:")
    try:
        db.upsert_user("u1", "a@b", "Alice")
        db.create_hazard(_mk_hazard(id="hz1"))
        assert db.get_hazard("hz1", user_id="u1").id == "hz1"
        assert db.get_hazard("hz1", user_id="other") is None
    finally:
        db.close()


def test_hazards_ordered_newest_first():
    db = Database(":memory:")
    try:
        db.upsert_user("u1", "a@b", "A")
        db.create_hazard(_mk_hazard(id="old", reported_at=1000.0))
        db.create_hazard(_mk_hazard(id="new", reported_at=2000.0))
        rows = db.list_user_hazards("u1")
        assert [r.id for r in rows] == ["new", "old"]
    finally:
        db.close()


def test_list_expired_images_and_clear():
    db = Database(":memory:")
    try:
        db.upsert_user("u1", "a@b", "A")
        now = time.time()
        db.create_hazard(_mk_hazard(id="exp", image_expires_at=now - 60))
        db.create_hazard(_mk_hazard(id="fresh", image_expires_at=now + 60))
        rows = db.list_expired_images(now_ts=now)
        assert len(rows) == 1 and rows[0]["hazard_id"] == "exp"
        db.clear_image_object_key("exp")
        assert db.list_expired_images(now_ts=now) == []
    finally:
        db.close()


def test_work_order_store_adapter_wires_hazard():
    db = Database(":memory:")
    try:
        db.upsert_user("u1", "a@b", "A")
        db.create_hazard(_mk_hazard(id="hz1"))
        store = SqliteWorkOrderStore(db)
        wo_id = store.create_draft(hazard_id="hz1", priority=95, suggested_action="patch")
        wo = db.get_work_order(wo_id)
        assert wo.hazard_id == "hz1"
        assert wo.priority == 95
        assert db.get_hazard("hz1").work_order_id == wo_id
    finally:
        db.close()


def test_review_queue_adapter_sets_status():
    db = Database(":memory:")
    try:
        db.upsert_user("u1", "a@b", "A")
        db.create_hazard(_mk_hazard(id="hz1", status="open"))
        queue = SqliteReviewQueue(db)
        queue.flag(hazard_id="hz1", reason="test")
        assert db.get_hazard("hz1").status == "human_review"
    finally:
        db.close()


def test_summary_counts_and_critical_list():
    db = Database(":memory:")
    try:
        db.upsert_user("u1", "a@b", "A")
        now = time.time()
        db.create_hazard(_mk_hazard(id="h1", agent_decision="urgent", reported_at=now - 10))
        db.create_hazard(_mk_hazard(id="h2", agent_decision="standard", reported_at=now - 20))
        db.create_hazard(_mk_hazard(id="h3", agent_decision="human_review", reported_at=now - 30))
        counts = db.count_hazards_since(now - 60)
        assert counts == {"D40:urgent": 1, "D40:standard": 1, "D40:human_review": 1}
        critical = db.recent_critical_hazards(now - 60)
        assert {h.id for h in critical} == {"h1", "h3"}
    finally:
        db.close()
