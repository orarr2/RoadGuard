"""Queue management: inbox order, startup recovery of interrupted runs,
expired evidence routed to a human, approval SLA counting."""

from __future__ import annotations

import json
import time

import roadguard_agent as ra
import roadguard_api as api


def _hazard(hid: str, status: str, *, decision: str = "standard", prio: int = 10) -> ra.Hazard:
    return ra.Hazard(
        id=hid, user_id="u1", reported_at=time.time(), hazard_class="D40", confidence=0.7,
        road_type="main", lane_blocked="partial", location="Allenby 42", bboxes_json="[]",
        priority=prio, agent_decision=decision, agent_rationale="fixture",
        agent_trace_json=json.dumps([{"iter": 0, "speaker": "sanitize", "kind": "final", "content": {}}]),
        status=status, work_order_id=None,
    )


def test_pending_approvals_ordered_by_priority_then_age(tmp_path):
    with ra.Database(str(tmp_path / "q.db")) as db:
        db.upsert_user("u1", "a@example.com", "A")
        low_old = db.create_approval("hz1", "human_review", 40, "r")
        time.sleep(0.01)
        high = db.create_approval("hz2", "urgent", 90, "r")
        time.sleep(0.01)
        low_new = db.create_approval("hz3", "human_review", 40, "r")
        assert [a.id for a in db.list_pending_approvals()] == [high, low_old, low_new]


def test_recover_interrupted_runs_routes_stuck_rows_to_human(tmp_path):
    path = str(tmp_path / "q.db")
    with ra.Database(path) as db:
        db.upsert_user("u1", "a@example.com", "A")
        db.create_hazard(_hazard("hz_proc", "processing", prio=0))
        db.create_hazard(_hazard("hz_queued", "queued", prio=70))
        db.create_hazard(_hazard("hz_done", "open", decision="standard"))

    assert api.recover_interrupted_runs(path) == 2

    with ra.Database(path) as db:
        rows = {h.id: h for h in db.list_recent_hazards(limit=10)}
        assert rows["hz_proc"].status == "open"
        assert rows["hz_proc"].agent_decision == "human_review"
        assert rows["hz_proc"].priority == 50  # floor for a run nobody finished
        assert rows["hz_queued"].priority == 70  # an existing higher priority is kept
        assert json.loads(rows["hz_proc"].agent_trace_json)[-1]["kind"] == "graph_interrupted"
        assert rows["hz_done"].agent_decision == "standard"  # untouched
        pending = db.list_pending_approvals()
        assert sorted(a.hazard_id for a in pending) == ["hz_proc", "hz_queued"]
        assert {a.action for a in pending} == {"human_review"}

    # Idempotent: the recovered rows are 'open' now, nothing left to recover.
    assert api.recover_interrupted_runs(path) == 0


def test_expired_evidence_request_routes_hazard_to_human(tmp_path):
    path = str(tmp_path / "q.db")
    with ra.Database(path) as db:
        db.upsert_user("u1", "a@example.com", "A")
        db.create_hazard(_hazard("hz_wait", "waiting_for_evidence", decision="needs_more_evidence", prio=30))
        req_id = db.create_evidence_request("hz_wait", "second_angle", "please add a second photo")
        # Age the request past the TTL.
        db._conn.execute(
            "UPDATE evidence_requests SET created_at = ? WHERE id = ?",
            (time.time() - 100 * 3600, req_id),
        )
        assert db.expire_evidence_requests(time.time() - 72 * 3600) == 1
        assert api.route_expired_evidence(db, 72) == 1
        assert api.route_expired_evidence(db, 72) == 0  # idempotent

        h = {x.id: x for x in db.list_recent_hazards(limit=5)}["hz_wait"]
        assert (h.status, h.agent_decision, h.priority) == ("open", "human_review", 50)
        assert "did not answer" in h.agent_rationale
        assert json.loads(h.agent_trace_json)[-1]["kind"] == "evidence_expired"
        assert [a.hazard_id for a in db.list_pending_approvals()] == ["hz_wait"]


def test_approvals_overdue_counts_against_sla():
    class A:
        def __init__(self, age_h: float) -> None:
            self.created_at = time.time() - age_h * 3600

    pending = [A(1), A(23.9), A(24.5), A(80)]
    assert api.approvals_overdue(pending, 24) == 2
    assert api.approvals_overdue(pending, 100) == 0


def test_policy_queues_defaults_and_file_values(monkeypatch):
    q = ra._policy_queues()
    assert q == {"approval_sla_hours": 24.0, "evidence_ttl_hours": 72.0}
    monkeypatch.setattr(ra, "load_policy", lambda: {"queues": {"approval_sla_hours": "x"}})
    assert ra._policy_queues() == {"approval_sla_hours": 24.0, "evidence_ttl_hours": 72.0}
    monkeypatch.setattr(ra, "load_policy", lambda: {"queues": {"approval_sla_hours": 6, "evidence_ttl_hours": 12}})
    assert ra._policy_queues() == {"approval_sla_hours": 6.0, "evidence_ttl_hours": 12.0}
