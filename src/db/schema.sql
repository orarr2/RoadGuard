-- RoadGuard SQLite schema. Applied idempotently by Database.__init__.

CREATE TABLE IF NOT EXISTS users (
    id                  TEXT PRIMARY KEY,
    email               TEXT NOT NULL DEFAULT '',
    display_name        TEXT NOT NULL DEFAULT '',
    created_at          REAL NOT NULL,
    opted_in_to_report  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hazards (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    reported_at         REAL NOT NULL,
    class               TEXT NOT NULL,
    confidence          REAL NOT NULL,
    road_type           TEXT NOT NULL,
    lane_blocked        TEXT NOT NULL,
    location            TEXT,
    bboxes_json         TEXT NOT NULL DEFAULT '[]',
    image_object_key    TEXT,
    image_expires_at    REAL,
    priority            INTEGER NOT NULL DEFAULT 0,
    agent_decision      TEXT NOT NULL DEFAULT 'human_review',
    agent_rationale     TEXT NOT NULL DEFAULT '',
    agent_trace_json    TEXT NOT NULL DEFAULT '[]',
    status              TEXT NOT NULL DEFAULT 'open',
    work_order_id       TEXT,
    cited_sources_json  TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_hazards_user_reported
    ON hazards(user_id, reported_at DESC);

CREATE INDEX IF NOT EXISTS idx_hazards_image_expires
    ON hazards(image_expires_at)
    WHERE image_object_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_hazards_status
    ON hazards(status);

CREATE TABLE IF NOT EXISTS work_orders (
    id                  TEXT PRIMARY KEY,
    hazard_id           TEXT NOT NULL,
    created_at          REAL NOT NULL,
    status              TEXT NOT NULL DEFAULT 'draft',
    priority            INTEGER NOT NULL DEFAULT 0,
    suggested_action    TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(hazard_id) REFERENCES hazards(id)
);

CREATE INDEX IF NOT EXISTS idx_work_orders_hazard
    ON work_orders(hazard_id);

-- Charter §5-§6 require human approval for every urgent / human_review
-- verdict. A row lands here with status='pending'; the supervisor
-- receives an email with a link to /approvals/{id} and must click
-- approve or deny before anything downstream can act.
-- Note: hazard_id is a soft reference. Approvals are often created before
-- the hazard row is persisted (the agent writes approval during the live
-- flow, the hazard row is written later or not at all for cached runs),
-- so no FOREIGN KEY here on purpose.
CREATE TABLE IF NOT EXISTS approvals (
    id                  TEXT PRIMARY KEY,
    hazard_id           TEXT NOT NULL,
    action              TEXT NOT NULL,         -- urgent | human_review
    priority            INTEGER NOT NULL DEFAULT 0,
    rationale           TEXT NOT NULL DEFAULT '',
    cited_sources_json  TEXT NOT NULL DEFAULT '[]',
    created_at          REAL NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | denied
    decided_at          REAL,
    decided_by          TEXT
);

CREATE INDEX IF NOT EXISTS idx_approvals_status
    ON approvals(status);

CREATE INDEX IF NOT EXISTS idx_approvals_hazard
    ON approvals(hazard_id);
