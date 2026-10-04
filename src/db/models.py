"""SQLite schema runner, dataclasses, and CRUD used by the API and jobs.

A tiny `Database` wrapper owns a single sqlite3 connection, applies the
idempotent schema at startup, and exposes typed accessors. Two adapter
classes satisfy the WorkOrderStore and ReviewQueue protocols that the
agent tools consume, so the API can wire a real DB without the agent
module importing anything from here.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class User:
    id: str
    email: str
    display_name: str
    created_at: float
    opted_in_to_report: bool = False


@dataclass
class Hazard:
    id: str
    user_id: str
    reported_at: float
    hazard_class: str
    confidence: float
    road_type: str
    lane_blocked: str
    location: Optional[str]
    bboxes_json: str
    image_object_key: Optional[str]
    image_expires_at: Optional[float]
    priority: int
    agent_decision: str
    agent_rationale: str
    agent_trace_json: str
    status: str
    work_order_id: Optional[str]
    cited_sources_json: str = "[]"


@dataclass
class WorkOrder:
    id: str
    hazard_id: str
    created_at: float
    status: str
    priority: int
    suggested_action: str


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class Database:
    """A single-connection SQLite wrapper. Thread-safe for FastAPI workers
    because sqlite3 is compiled with the GIL and WAL journaling is on.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            path,
            isolation_level=None,  # autocommit
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._apply_schema()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _apply_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        self._conn.executescript(sql)

    # ----- Users -----------------------------------------------------------

    def upsert_user(self, user_id: str, email: str, display_name: str) -> None:
        self._conn.execute(
            """
            INSERT INTO users (id, email, display_name, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                email = excluded.email,
                display_name = excluded.display_name
            """,
            (user_id, email, display_name, time.time()),
        )

    def set_report_opt_in(self, user_id: str, opt_in: bool) -> None:
        self._conn.execute(
            "UPDATE users SET opted_in_to_report = ? WHERE id = ?",
            (1 if opt_in else 0, user_id),
        )

    def get_user(self, user_id: str) -> Optional[User]:
        row = self._conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def report_recipients(self) -> list[User]:
        rows = self._conn.execute(
            "SELECT * FROM users WHERE opted_in_to_report = 1"
        ).fetchall()
        return [self._row_to_user(r) for r in rows]

    # ----- Hazards ---------------------------------------------------------

    def create_hazard(self, hazard: Hazard) -> None:
        self._conn.execute(
            """
            INSERT INTO hazards (
                id, user_id, reported_at, class, confidence, road_type,
                lane_blocked, location, bboxes_json, image_object_key,
                image_expires_at, priority, agent_decision, agent_rationale,
                agent_trace_json, status, work_order_id, cited_sources_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hazard.id,
                hazard.user_id,
                hazard.reported_at,
                hazard.hazard_class,
                hazard.confidence,
                hazard.road_type,
                hazard.lane_blocked,
                hazard.location,
                hazard.bboxes_json,
                hazard.image_object_key,
                hazard.image_expires_at,
                hazard.priority,
                hazard.agent_decision,
                hazard.agent_rationale,
                hazard.agent_trace_json,
                hazard.status,
                hazard.work_order_id,
                hazard.cited_sources_json,
            ),
        )

    def get_hazard(
        self,
        hazard_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Hazard]:
        if user_id is not None:
            row = self._conn.execute(
                "SELECT * FROM hazards WHERE id = ? AND user_id = ?",
                (hazard_id, user_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM hazards WHERE id = ?", (hazard_id,)
            ).fetchone()
        return self._row_to_hazard(row) if row else None

    def list_user_hazards(
        self,
        user_id: str,
        limit: int = 50,
        before_ts: Optional[float] = None,
    ) -> list[Hazard]:
        if before_ts is not None:
            rows = self._conn.execute(
                """
                SELECT * FROM hazards
                WHERE user_id = ? AND reported_at < ?
                ORDER BY reported_at DESC LIMIT ?
                """,
                (user_id, before_ts, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM hazards WHERE user_id = ?
                ORDER BY reported_at DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._row_to_hazard(r) for r in rows]

    def list_expired_images(
        self,
        now_ts: float,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, image_object_key, image_expires_at FROM hazards
            WHERE image_object_key IS NOT NULL
              AND image_expires_at IS NOT NULL
              AND image_expires_at < ?
            ORDER BY image_expires_at ASC LIMIT ?
            """,
            (now_ts, limit),
        ).fetchall()
        return [
            {
                "hazard_id": r["id"],
                "image_object_key": r["image_object_key"],
                "image_expires_at": r["image_expires_at"],
            }
            for r in rows
        ]

    def clear_image_object_key(self, hazard_id: str) -> None:
        self._conn.execute(
            "UPDATE hazards SET image_object_key = NULL WHERE id = ?",
            (hazard_id,),
        )

    def set_hazard_status(self, hazard_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE hazards SET status = ? WHERE id = ?",
            (status, hazard_id),
        )

    def set_hazard_work_order(self, hazard_id: str, work_order_id: str) -> None:
        self._conn.execute(
            "UPDATE hazards SET work_order_id = ? WHERE id = ?",
            (work_order_id, hazard_id),
        )

    def count_hazards_since(self, since_ts: float) -> dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT class, agent_decision, COUNT(*) AS n
            FROM hazards
            WHERE reported_at >= ?
            GROUP BY class, agent_decision
            """,
            (since_ts,),
        ).fetchall()
        return {f"{r['class']}:{r['agent_decision']}": r["n"] for r in rows}

    def recent_critical_hazards(
        self,
        since_ts: float,
        limit: int = 100,
    ) -> list[Hazard]:
        rows = self._conn.execute(
            """
            SELECT * FROM hazards
            WHERE reported_at >= ?
              AND agent_decision IN ('urgent', 'human_review')
            ORDER BY reported_at DESC LIMIT ?
            """,
            (since_ts, limit),
        ).fetchall()
        return [self._row_to_hazard(r) for r in rows]

    # ----- Work orders -----------------------------------------------------

    def create_draft_work_order(
        self,
        hazard_id: str,
        priority: int,
        suggested_action: str,
    ) -> str:
        work_order_id = f"wo_{uuid.uuid4().hex[:12]}"
        self._conn.execute(
            """
            INSERT INTO work_orders (
                id, hazard_id, created_at, status, priority, suggested_action
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                work_order_id,
                hazard_id,
                time.time(),
                "draft",
                int(priority),
                suggested_action,
            ),
        )
        self.set_hazard_work_order(hazard_id, work_order_id)
        return work_order_id

    def get_work_order(self, work_order_id: str) -> Optional[WorkOrder]:
        row = self._conn.execute(
            "SELECT * FROM work_orders WHERE id = ?", (work_order_id,)
        ).fetchone()
        if row is None:
            return None
        return WorkOrder(
            id=row["id"],
            hazard_id=row["hazard_id"],
            created_at=row["created_at"],
            status=row["status"],
            priority=row["priority"],
            suggested_action=row["suggested_action"],
        )

    # ----- Row mappers -----------------------------------------------------

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            email=row["email"],
            display_name=row["display_name"],
            created_at=row["created_at"],
            opted_in_to_report=bool(row["opted_in_to_report"]),
        )

    @staticmethod
    def _row_to_hazard(row: sqlite3.Row) -> Hazard:
        return Hazard(
            id=row["id"],
            user_id=row["user_id"],
            reported_at=row["reported_at"],
            hazard_class=row["class"],
            confidence=row["confidence"],
            road_type=row["road_type"],
            lane_blocked=row["lane_blocked"],
            location=row["location"],
            bboxes_json=row["bboxes_json"],
            image_object_key=row["image_object_key"],
            image_expires_at=row["image_expires_at"],
            priority=row["priority"],
            agent_decision=row["agent_decision"],
            agent_rationale=row["agent_rationale"],
            agent_trace_json=row["agent_trace_json"],
            status=row["status"],
            work_order_id=row["work_order_id"],
            cited_sources_json=row["cited_sources_json"] or "[]",
        )


# ---------------------------------------------------------------------------
# Agent-tool adapters
# ---------------------------------------------------------------------------


class SqliteWorkOrderStore:
    """Satisfies agent.tools.WorkOrderStore against the SQLite backend."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def create_draft(
        self,
        hazard_id: str,
        priority: int,
        suggested_action: str,
    ) -> str:
        return self.db.create_draft_work_order(
            hazard_id=hazard_id,
            priority=priority,
            suggested_action=suggested_action,
        )


class SqliteReviewQueue:
    """Satisfies agent.tools.ReviewQueue against the SQLite backend."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def flag(self, hazard_id: str, reason: str) -> None:
        self.db.set_hazard_status(hazard_id, "human_review")
