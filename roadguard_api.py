"""RoadGuard HTTP service.

FastAPI application: login, hazard submission (sync, async, live), reports,
approvals, evidence requests, the operator pages (/trace, /supervisor) and
the retention sweep. The agent core comes from roadguard_agent.

Run:  uvicorn roadguard_api:app --host 0.0.0.0 --port 8000
"""

import contextlib
import json
import logging
import os
import platform
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, Body, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from pydantic.warnings import UnsupportedFieldAttributeWarning
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from roadguard_agent import (
    AgentDecision,
    Database,
    Hazard,
    Settings,
    create_jwt,
    decode_jwt,
    format_trace_lines,
    get_current_user,
    get_settings,
    multiagent_decide,
    multiagent_decide_live,
    multiagent_resume,
    sanitize_location,
    sanitize_reporter_text,
    verify_approval,
    verify_google_id_token,
    verify_passphrase,
)

LOGGER = logging.getLogger("roadguard.api")

# Operator pages served on private hosts only (see _private_page_or_404).
_TRACE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow">
<title>RoadGuard - Live Agent Trace (private)</title>
<style>
  html, body { margin: 0; background: #0c0c0c; color: #cccccc; }
  body {
    font-family: Consolas, "Cascadia Mono", Menlo, "DejaVu Sans Mono", monospace;
    font-size: 13px; line-height: 1.4;
    padding: 10px max(10px, env(safe-area-inset-right))
             calc(12px + env(safe-area-inset-bottom)) max(10px, env(safe-area-inset-left));
  }
  pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: inherit; }
  .lbl { color: #16c60c; }
  .dec { color: #f9f1a5; }
  .hdr { color: #8a8a8a; }
  .err { color: #e74856; }
  .sys { color: #a3b1c2; }
  #prompt { display: none; }
  #prompt.on { display: block; }
  #pw {
    background: transparent; border: 0; outline: 0; color: #cccccc;
    font: inherit; font-size: 16px; width: 60%; padding: 0;
  }
  .cur {
    display: none; width: 8px; height: 15px; background: #cccccc;
    vertical-align: -3px; animation: blink 1s steps(2) infinite;
  }
  .cur.on { display: inline-block; }
  @keyframes blink { to { visibility: hidden; } }
</style>
</head>
<body>
<pre id="term"># RoadGuard live agent trace - PRIVATE - follows the newest hazard of this account
</pre>
<form id="prompt" autocomplete="off"><span>roadguard passphrase (the one the webapp login accepts): </span><input id="pw" type="password" autocomplete="current-password" enterkeyhint="go"></form>
<span id="cursor" class="cur"></span>
<script>
const term = document.getElementById("term");
const prompt = document.getElementById("prompt");
const pw = document.getElementById("pw");
const cursor = document.getElementById("cursor");
let token = "";
try { token = localStorage.getItem("rg_token") || ""; } catch (e) {}

function apiPath(path) {
  // Reverse-proxied under /roadguard/ on the NetSec portal: keep the prefix.
  const here = window.location.pathname;
  const idx = here.indexOf("/trace");
  return (idx >= 0 ? here.substring(0, idx) : "") + path;
}

function classOf(line) {
  if (line.startsWith("[DECISION") || line.startsWith("[DONE")) return "dec";
  if (line.startsWith("[")) return "lbl";
  if (/^[=-]{10,}$/.test(line) || line.startsWith("AGENT RUN")) return "hdr";
  return "";
}

function print(line, cls) {
  const s = document.createElement("span");
  if (cls) s.className = cls;
  s.textContent = line + "\n";
  term.appendChild(s);
  window.scrollTo(0, document.body.scrollHeight);
}

async function login(passphrase) {
  const r = await fetch(apiPath("/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ passphrase: passphrase }),
  });
  if (!r.ok) { print("login failed (" + r.status + ")" + (r.status === 401 ? " - not the RoadGuard passphrase (the portal password is a different one)" : ""), "err"); return false; }
  token = (await r.json()).token;
  try { localStorage.setItem("rg_token", token); } catch (e) {}
  return true;
}

function showPrompt() { stop(); prompt.classList.add("on"); pw.focus(); }

prompt.addEventListener("submit", async (e) => {
  // A form submit fires on Enter and on the phone keyboard Go key alike.
  e.preventDefault();
  const value = pw.value; pw.value = "";
  prompt.classList.remove("on");
  if (await login(value)) follow(); else showPrompt();
});

async function get(path) {
  const r = await fetch(apiPath(path), { headers: { "X-RoadGuard-Token": token } });
  if (r.status === 401) {
    token = "";
    try { localStorage.removeItem("rg_token"); } catch (e) {}
    throw new Error("401");
  }
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

let current = null, printed = 0, active = false, timer = null, lastErr = "", idleShown = false;

async function tick() {
  try {
    if (!active) {
      // Between runs: watch the account's history for a newer hazard.
      // During a run only its report is polled, which keeps the request
      // rate at one call per tick.
      const h = await get("/hazards/history");
      const newest = (h.items && h.items[0]) ? h.items[0].hazard_id : null;
      if (!newest) {
        if (!idleShown) { print("(no hazards yet - upload one from the webapp)"); idleShown = true; }
        return;
      }
      if (newest !== current) { current = newest; printed = 0; active = true; print(""); }
      else return;
    }
    const rep = await get("/report/" + encodeURIComponent(current));
    const lines = rep.trace_lines || [];
    for (let i = printed; i < lines.length; i++) print(lines[i], classOf(lines[i]));
    printed = lines.length;
    const st = (rep.verdict || {}).status;
    active = (st === "queued" || st === "processing");
    cursor.classList.toggle("on", active);
    lastErr = "";
  } catch (err) {
    if (err.message === "401") { showPrompt(); return; }
    if (err.message !== lastErr) { print("error: " + err.message, "err"); lastErr = err.message; }
  }
}

let bannerShown = false;
async function banner() {
  // The opening screen: host, service, models, RAG, database, images,
  // tunnel, shortcuts and what is still missing (GET /status, private hosts only).
  if (bannerShown) return;
  try {
    const st = await get("/status");
    print(st.greeting || "", "sys");
    (st.lines || []).forEach(l => print(l, /^[=-]{10,}$/.test(l) ? "hdr" : "sys"));
    print("");
    bannerShown = true;
  } catch (err) {
    if (err.message === "401") throw err;
    print("status unavailable (" + err.message + ")", "err");
  }
}

async function follow() {
  stop();
  try { await banner(); } catch (err) { showPrompt(); return; }
  tick();
  timer = setInterval(tick, 2000);
}
function stop() { if (timer) { clearInterval(timer); timer = null; } cursor.classList.remove("on"); }

if (token) follow(); else showPrompt();
</script>
</body>
</html>
"""
_SUPERVISOR_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>RoadGuard - Supervisor Inbox</title>
<style>
:root {
  --bg: #0b1220; --card: #121a2e; --border: #1f2a44;
  --text: #e6ecff; --dim: #8a95b3;
  --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444; --info: #38bdf8;
}
* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
html, body { -webkit-text-size-adjust: 100%; }
body { margin: 0; padding: 14px max(14px, env(safe-area-inset-right))
  calc(14px + env(safe-area-inset-bottom)) max(14px, env(safe-area-inset-left));
  background: var(--bg); color: var(--text);
  font: 16px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif; }
@media (max-width: 600px) {
  body { font-size: 15px; padding-top: calc(14px + env(safe-area-inset-top)); }
  .hdr h1 { font-size: 19px !important; }
  .row { grid-template-columns: 1fr !important; }
  .row .buttons { grid-column: 1 !important; grid-row: auto !important;
    margin-top: 8px; }
  .row .buttons button { width: 100%; }
}
.hdr { display:flex; align-items:center; gap:12px; margin-bottom: 20px; }
.hdr h1 { margin:0; font-size: 22px; color: var(--info); }
.hdr .badge { background: rgba(239,68,68,.18); color: var(--bad);
  padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700;
  margin-left: auto; letter-spacing: .5px; }
input, button, select { background: var(--card); color: var(--text);
  border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
  font-size: 16px; font-family: inherit; min-height: 48px; }
button { cursor: pointer; font-weight: 600; }
button:active { transform: scale(0.97); }
button.primary { background: var(--info); color: #061020; font-weight: 700;
  min-height: 44px; }
.panel { background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin-bottom: 14px; }
.row { background: var(--card); border: 1px solid var(--border);
  border-left-width: 4px; border-radius: 10px; padding: 14px 16px;
  margin: 10px 0; display: grid; grid-template-columns: 1fr auto;
  grid-template-rows: auto auto; column-gap: 16px; }
.row.urgent  { border-left-color: var(--bad); }
.row.standard { border-left-color: var(--info); }
.row.needs_more_evidence { border-left-color: var(--warn); }
.row.human_review { border-left-color: var(--dim); }
.row .top { font-weight: 700; font-size: 15px; }
.row .top .act { display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 800; margin-left: 8px; letter-spacing: .4px;
  vertical-align: middle; }
.row .top .act.urgent  { background: rgba(239,68,68,.18); color: var(--bad); }
.row .top .act.standard { background: rgba(56,189,248,.18); color: var(--info); }
.row .top .act.human_review { background: rgba(138,149,179,.18); color: var(--dim); }
.row .meta { color: var(--dim); font-size: 12px; margin-top: 4px; }
.row .buttons { grid-column: 2; grid-row: 1 / span 2; display: flex;
  gap: 8px; align-items: center; }
.row .rationale { grid-column: 1 / span 2; color: var(--text);
  margin-top: 10px; font-size: 13px; }
.counts { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }
.counts span { background: var(--card); border: 1px solid var(--border);
  padding: 6px 12px; border-radius: 999px; font-size: 12px; }
.counts b { color: var(--info); }
.form { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.form input { flex: 1; min-width: 240px; }
.hint { color: var(--dim); font-size: 13px; margin-top: 10px; }
.empty { color: var(--dim); text-align: center; padding: 40px 20px;
  font-size: 14px; }
</style>
</head>
<body>
<div class="hdr">
  <h1>Supervisor Inbox</h1>
  <span class="badge">PRIVATE</span>
</div>

<div class="panel">
  <div class="form">
    <input id="token" type="password" placeholder="bearer token (JWT)" autocomplete="off">
    <button id="refresh">Refresh</button>
    <select id="speed">
      <option value="10000">Auto 10s</option>
      <option value="30000">Auto 30s</option>
      <option value="0">Manual</option>
    </select>
  </div>
  <div class="hint" id="hint">Enter a bearer token (POST /auth/login) and press Refresh.</div>
</div>

<div class="panel">
  <h3 style="margin:0;color:var(--info)">Overview</h3>
  <div class="counts" id="counts"></div>
</div>

<div class="panel">
  <h3 style="margin:0 0 10px 0;color:var(--info)">Pending approvals</h3>
  <div id="pending"></div>
</div>

<div class="panel">
  <h3 style="margin:0 0 10px 0;color:var(--info)">Waiting for citizen evidence</h3>
  <div id="waiting"></div>
</div>

<script>
const el = (id) => document.getElementById(id);
let timer = null;

function apiPath(path) {
  const here = window.location.pathname;
  const idx = here.indexOf("/supervisor");
  const base = idx >= 0 ? here.substring(0, idx) : "";
  return base + path;
}

async function fetchJson(path, token) {
  const r = await fetch(apiPath(path), { headers: { "X-RoadGuard-Token": token } });
  if (!r.ok) throw new Error(r.status + " " + r.statusText + " on " + path);
  return r.json();
}

function row(a) {
  const kind = (a.action || "").toLowerCase();
  const div = document.createElement("div");
  div.className = "row " + (kind || "standard");
  const prio = a.priority ?? "?";
  div.innerHTML =
    '<div class="top">' + a.hazard_id +
      ' <span class="act ' + kind + '">' + kind.replace(/_/g, " ") + '</span>' +
      '</div>' +
    '<div class="meta">approval ' + a.approval_id + ' - priority ' + prio +
      ' - created ' + a.created_at + '</div>' +
    '<div class="buttons">' +
      '<button class="primary" onclick="viewTrace(\'' + a.hazard_id + '\')">View trace</button>' +
    '</div>';
  return div;
}

function waitingRow(w) {
  const div = document.createElement("div");
  div.className = "row needs_more_evidence";
  div.innerHTML =
    '<div class="top">' + w.hazard_id +
      ' <span class="act needs_more_evidence">needs ' + (w.missing_field || "evidence") + '</span>' +
      '</div>' +
    '<div class="meta">request ' + w.request_id + ' - created ' + w.created_at + '</div>' +
    '<div class="buttons">' +
      '<button class="primary" onclick="viewTrace(\'' + w.hazard_id + '\')">View trace</button>' +
    '</div>';
  return div;
}

window.viewTrace = function(hazardId) {
  const base = window.location.pathname.replace(/\/supervisor\/?$/, "");
  window.location.href = base + "/trace?hz=" + encodeURIComponent(hazardId);
};

async function tick() {
  const token = el("token").value.trim();
  if (!token) return;
  try {
    const d = await fetchJson("/report", token);
    el("counts").innerHTML =
      '<span><b>Total:</b> ' + (d.total_hazards ?? 0) + '</span>' +
      '<span><b>Urgent:</b> ' + (d.by_action?.urgent ?? 0) + '</span>' +
      '<span><b>Standard:</b> ' + (d.by_action?.standard ?? 0) + '</span>' +
      '<span><b>Needs evidence:</b> ' + (d.by_action?.needs_more_evidence ?? 0) + '</span>' +
      '<span><b>Human review:</b> ' + (d.by_action?.human_review ?? 0) + '</span>';
    const pending = d.pending_approvals || [];
    const pEl = el("pending");
    pEl.innerHTML = "";
    if (!pending.length) pEl.innerHTML = '<div class="empty">no pending approvals</div>';
    else pending.forEach(a => pEl.appendChild(row(a)));
    const waiting = d.waiting_for_evidence || [];
    const wEl = el("waiting");
    wEl.innerHTML = "";
    if (!waiting.length) wEl.innerHTML = '<div class="empty">nothing waiting</div>';
    else waiting.forEach(w => wEl.appendChild(waitingRow(w)));
    el("hint").textContent = "Last refresh " + new Date().toLocaleTimeString();
  } catch (err) {
    el("hint").innerHTML = '<span style="color:var(--bad)">' + err.message + "</span>";
  }
}

function start() {
  if (timer) clearInterval(timer);
  const speed = parseInt(el("speed").value, 10) || 0;
  tick();
  if (speed > 0) timer = setInterval(tick, speed);
}

el("refresh").addEventListener("click", start);
el("speed").addEventListener("change", start);
el("token").addEventListener("keydown", e => { if (e.key === "Enter") start(); });
el("token").addEventListener("change", () => localStorage.setItem("rg_token", el("token").value));
const saved = localStorage.getItem("rg_token");
if (saved) { el("token").value = saved; start(); }
</script>
</body>
</html>
"""
_STATIC_PAGES = {"trace.html": _TRACE_HTML, "supervisor.html": _SUPERVISOR_HTML}

# ---------------------------------------------------------------------------
# section: api/schemas.py
# ---------------------------------------------------------------------------
# Pydantic request and response models for the FastAPI endpoints.



# Pydantic 2.13 warns when Field(alias=...) is attached to a Literal field,
# claiming the alias has no effect. In practice both the alias ("class")
# and the field name ("hazard_class") deserialize correctly here, so the
# warning is a false positive we silence rather than route around.
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


HazardClass = Literal["D00", "D10", "D20", "D40"]
RoadType = Literal["main", "secondary", "bike_path", "parking"]
LaneBlocked = Literal["none", "partial", "full"]
AgentAction = Literal["urgent", "standard", "needs_more_evidence", "human_review", "known_image"]


class LoginRequest(BaseModel):
    passphrase: str | None = None
    google_id_token: str | None = None


class LoginResponse(BaseModel):
    token: str
    expires_in: int


class Me(BaseModel):
    sub: str
    email: str
    name: str


class ImageQualitySignal(BaseModel):
    """Optional client-side quality probe surfaced to the Perception node."""

    blur_score: float | None = None
    brightness_mean: float | None = None
    is_road_scene: bool | None = None
    road_scene_confidence: float | None = None
    occlusion_estimate: Literal["low", "medium", "high"] | None = None
    ocr_text_detected: str | None = Field(default=None, max_length=2000)


class HazardSubmit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hazard_class: Literal["D00", "D10", "D20", "D40"] = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    # Bounded on purpose: every field here is stored on the hazard row
    # and read by the sanitizer, the digest and the location matcher.
    # A row is [x1, y1, x2, y2, class, conf]; anything longer is not a bbox.
    bboxes: list[Annotated[list[Any], Field(max_length=6)]] = Field(
        default_factory=list, max_length=200
    )
    road_type: RoadType
    lane_blocked: LaneBlocked
    location: str | None = Field(default=None, max_length=500)
    # Optional rich fields. Clients that have already computed them (live
    # drive, future webapp versions) can send them; older clients may omit
    # them and the agent falls back to defaults.
    top1_margin: float | None = Field(default=None, ge=0, le=1)
    area_ratio: float | None = Field(default=None, ge=0, le=1)
    n_detections_by_class: dict[HazardClass, int] | None = None
    center_y_rel: float | None = Field(default=None, ge=0, le=1)
    image_quality: ImageQualitySignal | None = None
    # Free text typed by the reporter (roadmap section 1). Sanitised in
    # the graph and read by the signal extractor; never overrides the fields
    # above.
    reporter_text: str | None = Field(default=None, max_length=1000)
    # SHA-256 of the uploaded photo and a 64-bit difference hash of the pixels;
    # the agents use them to recognise a photo already catalogued in the
    # reference dataset (the known_image action).
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    image_phash: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{16}$")


class EvidenceRequestOut(BaseModel):
    """What the agent asks the reporter for (roadmap section 2)."""

    id: str
    hazard_id: str
    missing_field: str
    message: str


class AgentVerdict(BaseModel):
    action: AgentAction
    rationale: str
    priority: int = Field(ge=0, le=100)
    cited_sources: list[str] = Field(default_factory=list)


class HazardSubmitResponse(BaseModel):
    hazard_id: str
    verdict: AgentVerdict
    provider: str
    fallback_reason: str | None = None
    # "waiting_for_evidence": the run is paused on its checkpoint until the
    # reporter answers `evidence_request` through
    # POST /evidence_requests/{id}/fulfill, which returns this same shape
    # with status "decided".
    status: Literal["decided", "waiting_for_evidence"] = "decided"
    evidence_request: EvidenceRequestOut | None = None


class LiveHazardSubmit(BaseModel):
    """Streaming submission from the live-drive screen.

    One per confirmed hazard, not per frame. The client's IoU tracker
    collapses frames into one submission; the server's decision cache
    (keyed by session_id + track_id + canonical attrs) covers the gap
    when the tracker drops and reacquires the same hazard.
    """

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(min_length=4, max_length=64)
    track_id: str | None = Field(default=None, max_length=64)
    hazard_class: Literal["D00", "D10", "D20", "D40"] = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    top1_margin: float | None = Field(default=None, ge=0, le=1)
    area_ratio: float | None = Field(default=None, ge=0, le=1)
    n_detections_by_class: dict[HazardClass, int] | None = None
    bbox: list[float] | None = Field(default=None, max_length=4)
    center_y_rel: float | None = Field(default=None, ge=0, le=1)
    road_type: RoadType
    lane_blocked: LaneBlocked
    location: str | None = Field(default=None, max_length=500)
    image_quality: ImageQualitySignal | None = None


class LiveHazardResponse(BaseModel):
    hazard_id: str
    verdict: AgentVerdict
    provider: str
    fallback_reason: str | None = None
    cached: bool = False
    cache_key: str | None = None


class HazardHistoryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hazard_id: str
    reported_at: str
    hazard_class: Literal["D00", "D10", "D20", "D40"] = Field(alias="class")
    confidence: float
    road_type: RoadType
    lane_blocked: LaneBlocked
    location: str | None = None
    priority: int
    agent_decision: AgentAction | Literal[""]  # "" while a row is queued/processing
    agent_rationale: str
    status: str


class HazardHistoryResponse(BaseModel):
    items: list[HazardHistoryItem] = Field(default_factory=list)
    next_cursor: str | None = None


class HazardDetail(HazardHistoryItem):
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)


class RagQueryRequest(BaseModel):
    hazard_class: HazardClass
    road_type: RoadType
    k: int = Field(default=3, ge=1, le=10)


class RagPassage(BaseModel):
    id: str
    text: str
    issuing_body: str
    publication_date: str
    hazard_type: str
    triggering_conditions: str
    source_url: str
    is_binding_in_il: bool


class RagQueryResponse(BaseModel):
    passages: list[RagPassage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    db_ready: bool
    agent_ready: bool
    rag_ready: bool


# ---------------------------------------------------------------------------
# section: api/decision_cache.py
# ---------------------------------------------------------------------------
# TTL cache of agent decisions keyed by streaming session.
#
# The live-drive flow fires many submissions per minute while the car
# passes a hazard. The IoU tracker on the client already collapses them
# to a single track_id, but the client-side tracker drops when the user
# looks away, so the same canonical hazard can resurface a few seconds
# later with a new track_id. The server-side cache is the second line
# of defence: within a short TTL, two (session, class, road_type)
# submissions resolve to the same verdict without a second agent call.
#
# Backed by SQLite (table `decision_cache`) so the cache survives
# systemctl restart. A fresh instance warm-starts from DB on first use;
# every put() writes through to disk. In a load-balanced deployment this
# should move to Redis - the write-through path is the same, only the
# backend changes.






DEFAULT_TTL_SECONDS = 120
DEFAULT_MAX_ENTRIES = 2000


@dataclass
class _Entry:
    decision: AgentDecision
    expires_at: float


def _decision_to_json(decision: AgentDecision) -> str:
    return json.dumps(asdict(decision), ensure_ascii=False)


def _decision_from_json(text: str) -> AgentDecision:
    data = json.loads(text)
    return AgentDecision(**data)


class DecisionCache:
    """TTL cache with FIFO eviction when the map grows past ``max_entries``.

    Writes through to SQLite via the Database helper so a restart does not
    forget decisions still inside the TTL window. Reads prefer RAM; a miss
    falls through to the DB (populating RAM on the way).
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        db_path: str | None = None,
    ) -> None:
        self.ttl_seconds = int(ttl_seconds)
        self.max_entries = int(max_entries)
        self._db_path = db_path
        self._store: dict[str, _Entry] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._warm_started = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Clear RAM. Does not touch the DB; use purge_expired for DB."""
        with self._lock:
            self._store.clear()
            self._order.clear()
            self._warm_started = False

    def _db(self):
        """Open a short-lived Database connection for the configured path.

        Returns None if no path is configured (test / in-RAM mode).
        """
        if not self._db_path:
            return None
        try:
            from roadguard_agent import Database

            return Database(self._db_path)
        except Exception as exc:
            LOGGER.warning("decision_cache_db_unavailable: %s", exc)
            return None

    def _warm_start(self) -> None:
        if self._warm_started:
            return
        self._warm_started = True
        db_ctx = self._db()
        if db_ctx is None:
            return
        try:
            with db_ctx as db:
                for key, decision_json, expires_at in db.decision_cache_load_unexpired():
                    try:
                        decision = _decision_from_json(decision_json)
                    except Exception:  # pragma: no cover
                        continue
                    self._store[key] = _Entry(
                        decision=decision, expires_at=expires_at
                    )
                    self._order.append(key)
        except Exception as exc:
            LOGGER.warning("decision_cache_warm_start_failed: %s", exc)

    def get(self, key: str, now: float | None = None) -> AgentDecision | None:
        now = time.time() if now is None else now
        with self._lock:
            self._warm_start()
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < now:
                self._store.pop(key, None)
                with contextlib.suppress(ValueError):
                    self._order.remove(key)
                db_ctx = self._db()
                if db_ctx is not None:
                    try:
                        with db_ctx as db:
                            db.decision_cache_delete(key)
                    except Exception:  # pragma: no cover
                        pass
                return None
            return entry.decision

    def put(
        self,
        key: str,
        decision: AgentDecision,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._warm_start()
            expires_at = now + self.ttl_seconds
            self._store[key] = _Entry(decision=decision, expires_at=expires_at)
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            evicted: list[str] = []
            while len(self._store) > self.max_entries:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)
                evicted.append(oldest)
            db_ctx = self._db()
            if db_ctx is not None:
                try:
                    with db_ctx as db:
                        db.decision_cache_put(
                            key=key,
                            decision_json=_decision_to_json(decision),
                            expires_at=expires_at,
                        )
                        for e in evicted:
                            db.decision_cache_delete(e)
                except Exception as exc:
                    LOGGER.warning("decision_cache_put_db_failed: %s", exc)

    @staticmethod
    def canonical_key(
        session_id: str,
        hazard_class: str,
        road_type: str,
        lane_blocked: str,
        track_id: str | None = None,
    ) -> str:
        """Build a stable cache key for the live-drive flow.

        The track_id (when supplied) narrows the key so two distinct
        trackers that happen to see the same class on the same road
        get independent decisions, which matches reporter expectations.
        """
        track = (track_id or "nt").replace("|", "_")
        return (
            f"{session_id}|{track}|{hazard_class}|{road_type}|{lane_blocked}"
        )


# The API hands this single instance into endpoints via a dependency.
# The DB path is resolved lazily on first use so this module can be
# imported before settings exist (tests).
_shared_cache: DecisionCache | None = None
_shared_lock = threading.Lock()


def get_decision_cache() -> DecisionCache:
    global _shared_cache
    if _shared_cache is None:
        with _shared_lock:
            if _shared_cache is None:
                try:
                    from roadguard_agent import get_settings

                    db_path = get_settings().roadguard_db_path
                except Exception:
                    db_path = None
                _shared_cache = DecisionCache(db_path=db_path)
    return _shared_cache


def reset_decision_cache() -> None:
    """Flush the shared cache. Tests use this between runs.

    Clears RAM only; DB rows older than TTL get swept lazily on next read.
    """
    global _shared_cache
    if _shared_cache is not None:
        _shared_cache.clear()
    _shared_cache = None


# ---------------------------------------------------------------------------
# section: api/rate_limits.py
# ---------------------------------------------------------------------------
# slowapi limiter keyed by JWT subject (plus IP for the shared passphrase).
#
# Every passphrase login carries the same subject, so keying on the
# subject alone would give all citizens one shared bucket; those users
# are keyed by subject + address. Google users have their own subject.
# The limit values come from Settings (ROADGUARD_*_LIMIT).





def _rate_limit_key(request: Request) -> str:
    """Prefer the authenticated user; fall back to the remote address."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        try:
            payload = decode_jwt(token, get_settings())
        except Exception:
            payload = None
        if payload and payload.get("sub"):
            sub = str(payload["sub"])
            if sub.startswith("passphrase:"):
                return f"user:{sub}|ip:{get_remote_address(request)}"
            return f"user:{sub}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_rate_limit_key)


# ---------------------------------------------------------------------------
# section: api/retention.py
# ---------------------------------------------------------------------------
# Retention sweeps the API runs on its own.
#
# The twice-daily digest used to carry these; after it was removed the
# 7-day checkpoint policy and the evidence-request expiry stopped running.
# They now run inside the API process: once at startup and then every
# SWEEP_INTERVAL_SECONDS on a daemon thread. Both sweeps are best-effort
# and never raise into the request path.





CHECKPOINT_RETENTION_DAYS = 7
SWEEP_INTERVAL_SECONDS = 6 * 3600


def prune_checkpoints(checkpoint_path: str, keep_thread_ids: list[str]) -> int:
    """Delete LangGraph checkpoint threads not in `keep_thread_ids`.

    thread_id == hazard_id, so the caller passes the ids of hazards
    reported inside the retention window; every other thread is older or
    an orphan. Returns the number of checkpoint rows removed; a missing
    file or missing tables counts as 0.
    """
    path = Path(checkpoint_path)
    if not checkpoint_path or not path.exists():
        return 0
    conn = sqlite3.connect(str(path), timeout=5)
    try:
        tables = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "checkpoints" not in tables:
            return 0
        conn.execute("CREATE TEMP TABLE keep (thread_id TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT OR IGNORE INTO keep (thread_id) VALUES (?)",
            [(t,) for t in keep_thread_ids],
        )
        removed = conn.execute(
            "DELETE FROM checkpoints WHERE thread_id NOT IN (SELECT thread_id FROM keep)"
        ).rowcount
        if "writes" in tables:
            conn.execute(
                "DELETE FROM writes WHERE thread_id NOT IN (SELECT thread_id FROM keep)"
            )
        conn.commit()
        if removed:
            conn.execute("VACUUM")
        return int(removed)
    finally:
        conn.close()


def retention_sweep(db_path: str, checkpoint_path: str) -> dict[str, int]:
    """Expire stale evidence requests and prune checkpoint threads older
    than CHECKPOINT_RETENTION_DAYS. Returns the counts for the log."""
    cutoff = time.time() - CHECKPOINT_RETENTION_DAYS * 86400
    with Database(db_path) as db:
        keep = [
            r["id"]
            for r in db._conn.execute(
                "SELECT id FROM hazards WHERE reported_at >= ?", (cutoff,)
            )
        ]
        expired = db.expire_evidence_requests(cutoff)
    removed = prune_checkpoints(checkpoint_path, keep)
    LOGGER.info(
        "retention_sweep: expired_requests=%d checkpoints_removed=%d kept_threads=%d",
        expired, removed, len(keep),
    )
    return {"expired": int(expired), "removed": removed, "kept": len(keep)}


def start_retention_thread(
    db_path: str,
    checkpoint_path: str,
    interval_seconds: int = SWEEP_INTERVAL_SECONDS,
) -> threading.Thread:
    """Run retention_sweep now and then every `interval_seconds`."""

    def _loop() -> None:
        while True:
            try:
                retention_sweep(db_path, checkpoint_path)
            except Exception as exc:
                LOGGER.warning("retention_sweep_failed: %s", exc)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="roadguard-retention", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# section: api/main.py
# ---------------------------------------------------------------------------
# FastAPI shell wiring the agent, auth, and rate limits.





VERSION = "0.1.0"
_REPO_ROOT = Path(__file__).resolve().parent  # this module sits at the repo root
APP_STARTED_AT = time.time()


def _render_approval_html(row: Any, token: str = "", assessment: str = "") -> str:
    """Minimal HTML page for the approval gate.

    Shows the pending verdict + Approve/Deny form, or a frozen status
    line once the decision has been recorded. No JavaScript, so it
    renders in any mail client's preview and any browser. `assessment`
    is the Decision model's text from the hazard row; it is shown under
    the rationale, labelled as model text, only when non-empty.
    """
    import html as _html
    import json as _json

    sources = []
    try:
        sources = _json.loads(row.cited_sources_json or "[]")
    except Exception:
        sources = []
    cited_html = (
        "<ul>" + "".join(f"<li>{_html.escape(s)}</li>" for s in sources) + "</ul>"
        if sources
        else "<em>no cited sources</em>"
    )
    assessment_html = (
        "<p><strong>Agent assessment (model text, not a finding)</strong>: "
        f"{_html.escape(assessment)}</p>"
        if assessment
        else ""
    )
    if row.status == "pending":
        # The note travels with whichever button is pressed. It is the
        # supervisor's reason for the record ("no budget this week" vs
        # "wrong class"); it is kept on the approvals row.
        decision_block = f"""\
<form method="POST" action="/approvals/{row.id}/decide" style="margin-top:16px">
  <input type="hidden" name="t" value="{_html.escape(token)}">
  <label for="note" style="display:block;font-size:14px;margin-bottom:4px">
    Note for the record (optional). Why you approve or deny, e.g.
    "no budget this week", "already repaired", "wrong class".
  </label>
  <textarea id="note" name="note" rows="3" maxlength="500"
            style="width:100%;box-sizing:border-box;font:inherit;padding:8px;
                   border:1px solid #9CA3AF;border-radius:4px"></textarea>
  <div style="display:flex;gap:12px;margin-top:12px">
    <button name="decision" value="approved"
            style="background:#1B5E20;color:white;border:none;padding:12px 24px;
                   font-size:16px;border-radius:4px;cursor:pointer">
      Approve
    </button>
    <button name="decision" value="denied"
            style="background:#B91C1C;color:white;border:none;padding:12px 24px;
                   font-size:16px;border-radius:4px;cursor:pointer">
      Deny
    </button>
  </div>
</form>
"""
    else:
        color = "#1B5E20" if row.status == "approved" else "#B91C1C"
        note = getattr(row, "note", "") or ""
        note_html = (
            f"<p><strong>Supervisor note</strong>: {_html.escape(note)}</p>"
            if note
            else ""
        )
        decision_block = (
            f'<p style="margin-top:16px"><strong style="color:{color}">'
            f"Status: {_html.escape(row.status)}</strong> · decided by "
            f"{_html.escape(row.decided_by or 'unknown')}</p>{note_html}"
        )

    return f"""\
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RoadGuard approval · {_html.escape(row.id)}</title></head>
<body style="font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;
             padding:0 16px;color:#111">
<h2 style="color:#B91C1C">RoadGuard · {_html.escape(row.action).upper()} awaiting approval</h2>
<p><strong>Hazard</strong>: <code>{_html.escape(row.hazard_id)}</code></p>
<p><strong>Priority</strong>: {row.priority}</p>
<p><strong>Rationale</strong>: {_html.escape(row.rationale)}</p>
{assessment_html}
<p><strong>Cited sources</strong>:</p>
{cited_html}
{decision_block}
<hr style="margin-top:32px">
<p style="font-size:12px;color:#6B7280">
Nothing has been dispatched. No contractor has been contacted. No road
has been closed. Approve creates the local draft work order; Deny
closes the case without action.
</p>
</body></html>
"""
_APPROVAL_LINK_INVALID_HTML = """\
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>RoadGuard approval</title></head>
<body style="font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px;color:#111">
<h2>This approval link is not valid</h2>
<p>Open the link exactly as it appears in the RoadGuard email. The link carries a
signature; an approval id on its own does not open this page.</p>
</body></html>
"""

# Browser origins allowed to call the API. The public webapp origin is
# deployment-specific and comes from ROADGUARD_CORS_ORIGINS (comma-separated);
# the defaults cover local development only.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173,http://127.0.0.1:8080"
)


def _cors_origins() -> list[str]:
    raw = os.environ.get("ROADGUARD_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]

# Host names on which the operator-only HTML pages (/trace, /supervisor)
# are served. Loopback by default; a deployment that serves them through a
# private reverse proxy adds the Host values the proxy forwards via
# ROADGUARD_PRIVATE_HOSTS (comma-separated).
_DEFAULT_PRIVATE_HOSTS = "127.0.0.1,localhost"


# ---------------------------------------------------------------------------
# Readiness helpers
# ---------------------------------------------------------------------------


def _db_ready(settings: Settings) -> bool:
    return os.path.isfile(settings.roadguard_db_path)


def _agent_ready(settings: Settings) -> bool:
    """Any configured LLM path counts as ready: a provider API key, an
    Ollama panel entry, or an explicit single-provider override. The
    Oracle A1.Flex deployment runs a local Ollama panel with no cloud
    API key - that is still a working agent path."""
    if settings.groq_api_key or os.environ.get("GEMINI_API_KEY") or \
            os.environ.get("MISTRAL_API_KEY"):
        return True
    panel = os.environ.get("AGENT_LLM_PANEL", "")
    if "ollama:" in panel.lower():
        return True
    return os.environ.get("AGENT_LLM_PROVIDER", "").lower().startswith("ollama")


def _rag_ready(settings: Settings) -> bool:
    return bool(settings.roadguard_rag_path) and os.path.isdir(settings.roadguard_rag_path)


def _stamp_trace(
    entries: list, stamps: dict[int, float] | None = None, now: float | None = None
) -> list:
    """Give every trace entry a `ts` the first time it is persisted. Index
    based, so the stamp survives LangGraph copying the state between steps."""
    now = now or time.time()
    out: list = []
    for i, e in enumerate(entries or []):
        if isinstance(e, dict) and "ts" not in e:
            e = {**e, "ts": (stamps.setdefault(i, now) if stamps is not None else now)}
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Operator status banner (/status): host, service, models, RAG, DB, images
# ---------------------------------------------------------------------------

_STATUS_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_STATUS_TTL_S = 20.0
_SHORTCUTS = ("agent-log", "rg-health", "rg-hazards", "rg-services", "rg-tunnel",
              "rg-restart", "rg-prompts", "rg-sql", "rg-catalog")


def _display_now() -> str:
    tz = os.environ.get("ROADGUARD_DISPLAY_TZ")
    try:
        from zoneinfo import ZoneInfo

        dt = datetime.now(ZoneInfo(tz)) if tz else datetime.now()
    except Exception:
        dt = datetime.now()
    return dt.strftime("%a %d.%m.%Y %H:%M")


def _read_meminfo() -> dict[str, int]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            return {ln.split(":")[0]: int(ln.split()[1]) for ln in fh if ":" in ln}
    except Exception:
        return {}


def _cpu_busy_percent(sample_s: float = 0.15) -> float | None:
    def _sample() -> tuple[int, int] | None:
        try:
            with open("/proc/stat", encoding="utf-8") as fh:
                parts = fh.readline().split()[1:]
            vals = [int(x) for x in parts]
            return sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0)
        except Exception:
            return None

    a = _sample()
    if a is None:
        return None
    time.sleep(sample_s)
    b = _sample()
    if b is None or b[0] == a[0]:
        return None
    return round(100.0 * (1 - (b[1] - a[1]) / (b[0] - a[0])), 1)


def _os_pretty_name() -> str:
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("PRETTY_NAME="):
                    return ln.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return platform.platform()


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return os.environ.get("ROADGUARD_COMMIT", "unknown")


def _ollama_models() -> dict[str, Any]:
    import urllib.request

    base = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    out: dict[str, Any] = {"installed": [], "loaded": [], "error": None}
    for key, path in (("installed", "/api/tags"), ("loaded", "/api/ps")):
        try:
            with urllib.request.urlopen(base + path, timeout=2) as r:
                data = json.loads(r.read().decode("utf-8"))
            out[key] = [m.get("name", "?").removesuffix(":latest") for m in data.get("models", [])]
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}"
    return out


def _count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def _collect_status(settings: Settings) -> dict[str, Any]:
    mem = _read_meminfo()
    total_kb = mem.get("MemTotal", 0)
    avail_kb = mem.get("MemAvailable", 0)
    try:
        load: tuple[float, ...] | None = os.getloadavg()
    except (AttributeError, OSError):
        load = None
    disk = shutil.disk_usage(str(_REPO_ROOT))
    host = {
        "hostname": platform.node(),
        "os": _os_pretty_name(),
        "arch": platform.machine(),
        "cpus": os.cpu_count(),
        "mem_total_gb": round(total_kb / 1048576, 1) if total_kb else None,
        "mem_used_pct": round(100.0 * (1 - avail_kb / total_kb)) if total_kb else None,
        "load": [round(x, 2) for x in load] if load else None,
        "cpu_busy_pct": _cpu_busy_percent(),
        "disk_used_gb": round(disk.used / 1e9),
        "disk_total_gb": round(disk.total / 1e9),
        "disk_pct": round(100.0 * disk.used / disk.total) if disk.total else None,
    }
    service = {
        "version": VERSION,
        "uptime_s": int(time.time() - APP_STARTED_AT),
        "commit": _git_commit(),
        "python": platform.python_version(),
        "provider": os.environ.get("AGENT_LLM_PROVIDER") or "groq",
        "react": os.environ.get("AGENT_DECISION_REACT", "0") == "1",
        "wall_clock_cap_s": os.environ.get("WALL_CLOCK_CAP_SECONDS", "30"),
        "async_concurrency": os.environ.get("ROADGUARD_ASYNC_CONCURRENCY", "2"),
    }
    models = _ollama_models()
    rag: dict[str, Any] = {"passages": None}
    try:
        from roadguard_agent import RagClient

        rag["passages"] = RagClient.shared().count()
    except Exception as exc:
        rag["error"] = type(exc).__name__
    database: dict[str, Any] = {"total": 0, "by_action": {}, "pending_approvals": 0, "open_evidence": 0}
    try:
        from roadguard_agent import Database

        with Database(settings.roadguard_db_path) as db:
            by_action: dict[str, int] = {}
            for key, n in db.count_hazards_since(0.0).items():
                action = key.split(":", 1)[1] if ":" in key else key
                by_action[action or "queued"] = by_action.get(action or "queued", 0) + n
            database = {
                "total": sum(by_action.values()),
                "by_action": by_action,
                "pending_approvals": len(db.list_pending_approvals(limit=500)),
                "open_evidence": len(db.list_pending_evidence_requests(limit=500)),
            }
    except Exception as exc:
        database["error"] = type(exc).__name__
    images: dict[str, Any] = {
        "source": "dataset_images table",
        "count": 0,
        "target": int(os.environ.get("ROADGUARD_DATASET_TARGET", "10000")),
        "by_split": {},
        "with_predictions": 0,
    }
    try:
        from roadguard_agent import Database as _ImgDb
        with _ImgDb(settings.roadguard_db_path) as _db_img:
            _dstats = _db_img.dataset_catalogue_stats()
            images["count"] = _dstats.get("count", 0)
            images["by_split"] = _dstats.get("by_split", {})
            images["with_predictions"] = _dstats.get("with_predictions", 0)
    except Exception as _exc:
        images["error"] = type(_exc).__name__
    tunnel: dict[str, Any] = {"origin": None, "kind": None}
    try:
        origin = json.loads((_REPO_ROOT / "webapp" / "tunnel-url.json").read_text(encoding="utf-8")).get("origin")
        tunnel = {"origin": origin, "kind": "quick" if origin and "trycloudflare.com" in origin else "named"}
    except Exception:
        pass
    # systemd units run with a short PATH; the operator shortcuts live in /usr/local/bin
    lookup_path = os.environ.get("PATH", "") + os.pathsep + "/usr/local/bin" + os.pathsep + "/usr/bin"
    commands = [c for c in _SHORTCUTS if shutil.which(c, path=lookup_path)]
    missing: list[str] = []
    if images["count"] < images["target"]:
        missing.append(
            f"dataset: {images['target'] - images['count']:,} images to catalogue "
            f"(run `rg-catalog` to rebuild from the archive)"
        )
    if tunnel["kind"] != "named":
        missing.append("named cloudflare tunnel")
    if not (_REPO_ROOT / "tests" / "labeled_scenarios.json").exists():
        missing.append("labelled scenarios (tests/labeled_scenarios.json)")
    if models["error"]:
        missing.append(f"ollama unreachable ({models['error']})")
    return {
        "host": host, "service": service, "models": models, "rag": rag, "database": database,
        "images": images, "tunnel": tunnel, "commands": commands, "missing": missing,
    }


def _status_lines(d: dict[str, Any]) -> list[str]:
    h, s, m, rag, db, im, tn = (d["host"], d["service"], d["models"], d["rag"], d["database"],
                                 d["images"], d["tunnel"])
    state = "operational" if not m["error"] and rag.get("passages") is not None and "error" not in db else "degraded"
    bar = "=" * 56
    up = s["uptime_s"]
    uptime = f"{up // 3600}h {(up % 3600) // 60:02d}m" if up >= 3600 else f"{up // 60}m {up % 60:02d}s"
    load = " ".join(f"{x:.2f}" for x in h["load"]) if h["load"] else "n/a"
    active = s["provider"].split(":", 1)[1] if ":" in s["provider"] else s["provider"]
    installed = " ".join(n + ("*" if n == active else "") for n in m["installed"]) or "none"
    loaded = ", ".join(m["loaded"]) or "none"
    actions = ", ".join(f"{k} {v}" for k, v in sorted(db["by_action"].items())) or "none"
    pct = round(100.0 * im["count"] / im["target"]) if im["target"] else 0
    cpu = f"{h['cpu_busy_pct']}%" if h["cpu_busy_pct"] is not None else "n/a"
    lines = [
        bar,
        f"  RoadGuard - {state}   {h['hostname']}   {_display_now()}",
        bar,
        f"  host      {h['os']}  {h['arch']}  {h['cpus']} cpu  {h['mem_total_gb']} GB ram",
        f"  load      {load}   cpu {cpu}   mem {h['mem_used_pct']}%   "
        f"disk {h['disk_pct']}% ({h['disk_used_gb']}/{h['disk_total_gb']} GB)",
        f"  service   roadguard v{s['version']}  up {uptime}  commit {s['commit']}  python {s['python']}",
        f"  agents    provider {s['provider']}  react {'on' if s['react'] else 'off'}  "
        f"wall clock {s['wall_clock_cap_s']}s  parallel runs {s['async_concurrency']}",
        f"  models    {len(m['installed'])} installed: {installed}   (* active)   loaded now: {loaded}",
        f"  rag       chromadb {rag.get('passages') if rag.get('passages') is not None else 'unavailable'} passages",
        f"  database  {db['total']} hazards: {actions}   approvals pending {db['pending_approvals']}   "
        f"evidence open {db['open_evidence']}",
        f"  dataset   {im['count']:,} / {im['target']:,} ({pct}%) catalogued; "
        f"{im.get('with_predictions', 0):,} with model predictions; split {im.get('by_split', {})}",
        f"  tunnel    {tn['kind'] or 'none'}: {tn['origin'] or 'not configured'}",
        f"  commands  {' '.join(d['commands']) or 'none installed'}",
        f"  missing   {'; '.join(d['missing']) or 'nothing'}",
        bar,
    ]
    return lines


def _status_payload(settings: Settings) -> dict[str, Any]:
    now = time.time()
    cached = _STATUS_CACHE["payload"]
    if cached is not None and now - _STATUS_CACHE["at"] < _STATUS_TTL_S:
        return cached
    data = _collect_status(settings)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "data": data,
        "lines": _status_lines(data),
    }
    _STATUS_CACHE["at"] = now
    _STATUS_CACHE["payload"] = payload
    return payload


# ---------------------------------------------------------------------------
# Request-body helpers (dedupe between /hazards/submit and /hazards/live)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Reported-at timestamp in the shape every agent path expects."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dump_image_quality(body: Any, *, missing: Any) -> Any:
    """Serialize a HazardSubmit/LiveHazardSubmit image_quality block or
    return `missing` when the client did not send one.

    Both endpoints had the same `body.image_quality.model_dump(exclude_none=True)
    if body.image_quality is not None else <default>` ternary - this is the one
    place that owns it.
    """
    if body.image_quality is None:
        return missing
    return body.image_quality.model_dump(exclude_none=True)


def _hazard_input_from_batch(body: Any, hazard_id: str, reported_at: str) -> dict[str, Any]:
    """Build the flat hazard dict for /hazards/submit (batch flow)."""
    return {
        "hazard_id": hazard_id,
        "class": body.hazard_class,
        "confidence": float(body.confidence),
        "bboxes": body.bboxes,
        "road_type": body.road_type,
        "lane_blocked": body.lane_blocked,
        "location": body.location or "",
        "reported_at": reported_at,
        # Optional rich fields - absent on older clients
        "top1_margin": body.top1_margin,
        "area_ratio": body.area_ratio,
        "n_detections_by_class": body.n_detections_by_class,
        "center_y_rel": body.center_y_rel,
        "image_quality": _dump_image_quality(body, missing=None),
        # Roadmap section 1: free text from the reporter, read by Intake.
        "reporter_text": body.reporter_text or "",
        "image_sha256": (body.image_sha256 or "").lower(),
        "image_phash": (body.image_phash or "").lower(),
    }


def _evidence_request_out(decision: Any, hazard_id: str) -> EvidenceRequestOut | None:
    """The one request the agent sent the reporter, or None."""
    req = getattr(decision, "evidence_request", None) or {}
    if decision.action != "needs_more_evidence" or not req.get("request_id"):
        return None
    return EvidenceRequestOut(
        id=str(req["request_id"]),
        hazard_id=hazard_id,
        missing_field=str(req.get("missing_field") or "second_angle"),
        message=str(req.get("message") or ""),
    )


def _submit_dict_from_live(body: Any, reported_at: str) -> dict[str, Any]:
    """Build the dict passed to multiagent_decide_live from /hazards/live."""
    return {
        "class": body.hazard_class,
        "confidence": float(body.confidence),
        "top1_margin": body.top1_margin,
        "area_ratio": body.area_ratio,
        "n_detections_by_class": body.n_detections_by_class,
        "bbox": body.bbox,
        "center_y_rel": body.center_y_rel,
        "road_type": body.road_type,
        "lane_blocked": body.lane_blocked,
        "location": body.location or "",
        "session_id": body.session_id,
        "track_id": body.track_id or "",
        "image_quality": _dump_image_quality(body, missing={}),
        "reported_at": reported_at,
    }


def _user_id_of(user: dict[str, Any]) -> str:
    """The id stored on hazards.user_id and compared on every ownership
    check. Passphrase users share one sub by design."""
    return user.get("sub") or user.get("email") or "anonymous"


def _approval_id_from_trace(decision: Any) -> str | None:
    """The approval row the gate wrote during this run, read from the
    trace rather than the DB: when the hazard INSERT just failed twice,
    a lookup on the same connection is likely to fail as well."""
    for entry in reversed(list(getattr(decision, "trace", None) or [])):
        if isinstance(entry, dict) and entry.get("speaker") == "approval_gate":
            return (entry.get("content") or {}).get("approval_id")
    return None


def _evidence_not_found() -> HTTPException:
    # One detail for missing, not pending, not the caller's and already
    # spent: the response must not tell an outsider which one it was.
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="request_not_pending_or_missing",
    )


def _persist_hazard(
    db: Any,
    user: dict[str, Any],
    *,
    hazard_id: str,
    hazard_class: str,
    confidence: float,
    road_type: str,
    lane_blocked: str,
    location: str,
    bboxes: list[Any],
    decision: Any,
    reporter_text: str = "",
    status_value: str = "open",
) -> None:
    """Record the citizen upload and the agent's verdict as a hazards row.

    Everything downstream - /report, rg-hazards, and the
    approvals JOIN that surfaces urgent cases to the reviewer - reads
    this table, so a triage that is not persisted is invisible to the
    municipality. Best-effort: a DB error is logged, never turned into
    a 500 on a verdict the agent already produced. The INSERT is retried
    once; a second failure is an ERROR naming the hazard and the approval
    row the gate already wrote, so the orphan can be reconciled.
    """
    try:
        user_id = _user_id_of(user)
        db.upsert_user(user_id, user.get("email", ""), user.get("name", ""))
        # The agents only ever saw the sanitised copies; the row stores
        # those too, so no downstream reader ever sees the raw text back in.
        row = Hazard(
            id=hazard_id,
            user_id=user_id,
            reported_at=time.time(),
            hazard_class=hazard_class,
            confidence=float(confidence),
            road_type=road_type,
            lane_blocked=lane_blocked,
            location=sanitize_location(location or ""),
            bboxes_json=json.dumps(bboxes or []),
            priority=int(decision.priority or 0),
            agent_decision=decision.action,
            agent_rationale=(decision.rationale or "")[:600],
            agent_trace_json=json.dumps(_stamp_trace(decision.trace or [])),
            status=status_value,
            work_order_id=db.work_order_id_for_hazard(hazard_id),
            cited_sources_json=json.dumps(list(decision.cited_sources or [])),
            reporter_text=sanitize_reporter_text(reporter_text or "")["text"],
            agent_assessment=(getattr(decision, "assessment", "") or "")[:400],
        )
        try:
            db.create_hazard(row)
        except Exception as first_exc:
            LOGGER.warning("hazard_persist_retry id=%s: %s", hazard_id, first_exc)
            db.create_hazard(row)
    except Exception as exc:
        LOGGER.error(
            "hazard_persist_failed id=%s approval_id=%s: %s",
            hazard_id,
            _approval_id_from_trace(decision),
            exc,
        )


def _update_hazard_after_resume(db: Any, hazard_id: str, decision: Any) -> None:
    """Overwrite the verdict columns once the evidence round trip ends."""
    try:
        db.update_hazard_verdict(
            hazard_id,
            priority=int(decision.priority or 0),
            agent_decision=decision.action,
            agent_rationale=(decision.rationale or "")[:600],
            agent_trace_json=json.dumps(_stamp_trace(decision.trace or [])),
            cited_sources_json=json.dumps(list(decision.cited_sources or [])),
            status="open",
            work_order_id=db.work_order_id_for_hazard(hazard_id),
            agent_assessment=(getattr(decision, "assessment", "") or "")[:400],
        )
    except Exception as exc:
        LOGGER.warning("hazard_update_failed id=%s: %s", hazard_id, exc)


# ---------------------------------------------------------------------------
# Shared DB handle (FastAPI dependency)
# ---------------------------------------------------------------------------
#
# Database() opens a sqlite3 connection AND runs the schema migration from
# schema.sql on every __init__. Each `with Database(path)` under an API
# handler therefore paid that cost on every request. One Database per path
# is safe across FastAPI's threadpool because the connection is opened with
# check_same_thread=False and WAL journaling.
#
# A plain dict (not functools.lru_cache) keeps the cache clearable - tests
# routinely rebuild the app against a fresh tmp DB path and want the old
# handle released so the tmp dir can be deleted on Windows.
_DB_CACHE: dict[str, Any] = {}


def _get_shared_db(path: str):
    """Return a long-lived Database for `path`, creating it on first call."""
    cached = _DB_CACHE.get(path)
    if cached is None:
        from roadguard_agent import Database
        cached = Database(path)
        _DB_CACHE[path] = cached
    return cached


def get_db(settings: Settings = Depends(get_settings)):
    """FastAPI dependency injecting the shared Database for this path."""
    return _get_shared_db(settings.roadguard_db_path)


def _reset_db_cache() -> None:
    """Close any cached Databases and drop their handles.

    create_app() calls this so a rebuilt app (what tests do between cases)
    does not keep a connection pointing at a tmp DB that pytest is about
    to delete.
    """
    for _db in list(_DB_CACHE.values()):
        with contextlib.suppress(Exception):
            _db.close()
    _DB_CACHE.clear()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _warm_agent_stack() -> None:
    """Pay the agent stack's cold start at startup, not inside the first
    citizen request's wall-clock budget (chromadb import and client, then
    the graph compile: 20-30s on the VM). Each step is best-effort; a
    failure only means the first request pays as before."""
    try:
        from roadguard_agent import RagClient

        RagClient.shared()
    except Exception as exc:
        LOGGER.warning("agent_warmup_failed step=rag_client: %s", exc)
    try:
        from roadguard_agent import build_graph

        build_graph()
    except Exception as exc:
        LOGGER.warning("agent_warmup_failed step=graph: %s", exc)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    # Startup only. uvicorn runs this before it binds the port, so a
    # restarted service answers its first upload warm; a TestClient used
    # without a context manager never runs it, so tests stay fast.
    try:
        _warm_agent_stack()
    except Exception as exc:
        LOGGER.warning("agent_warmup_failed: %s", exc)
    # Retention (7-day checkpoints, stale evidence requests) runs in this
    # process now that the digest job is gone. Skipped for the in-memory
    # DB the test suite uses and when ROADGUARD_RETENTION_SWEEP=0.
    settings = get_settings()
    if (
        os.environ.get("ROADGUARD_RETENTION_SWEEP", "1") != "0"
        and settings.roadguard_db_path != ":memory:"
    ):

        start_retention_thread(
            settings.roadguard_db_path,
            os.environ.get("ROADGUARD_CHECKPOINT_PATH", "./roadguard-checkpoints.sqlite"),
        )
    yield


def create_app() -> FastAPI:
    # The @limiter.limit decorators below each append a Limit object to the
    # shared Limiter instance's _route_limits dict. If create_app() is
    # called more than once (which tests routinely do), each build would
    # add another duplicate limit per route, which then counts each
    # request twice, three times, and so on. Clearing the dict at the
    # start of the build keeps every app instance to one limit per route.
    for attr in ("_route_limits", "_dynamic_route_limits"):
        bucket = getattr(limiter, attr, None)
        if bucket is not None and hasattr(bucket, "clear"):
            bucket.clear()

    _reset_db_cache()

    app = FastAPI(title="RoadGuard", version=VERSION, lifespan=_lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def _read_static(name: str) -> str:
        return _STATIC_PAGES[name]

    def _is_private_host(request: Request) -> bool:
        """IDX 28: operator pages and the status banner are served only when
        the request arrived on a private origin (loopback or the Tailscale
        proxy, which preserves the tailnet hostname in Host). The list is
        env-overridable."""
        allowed = {
            h.strip().lower()
            for h in os.environ.get("ROADGUARD_PRIVATE_HOSTS", _DEFAULT_PRIVATE_HOSTS).split(",")
            if h.strip()
        }
        host = (request.headers.get("host") or "").split(":")[0].lower()
        return host in allowed

    def _private_page_or_404(request: Request, name: str) -> HTMLResponse:
        """Anything that is not a private host, including the public
        Cloudflare tunnel, gets a plain 404 so the page is not advertised."""
        if not _is_private_host(request):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
        return HTMLResponse(_read_static(name))

    @app.get("/status")
    def system_status(
        request: Request,
        user: dict[str, Any] = Depends(get_current_user),
        settings: Settings = Depends(get_settings),
    ) -> dict:
        """Operator banner for the trace terminal: host, service, models,
        RAG, database, images, tunnel, shortcuts and what is still missing.
        Private hosts only, like the pages; cached for 20 seconds."""
        if not _is_private_host(request):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
        payload = dict(_status_payload(settings))
        who = user.get("name") or user.get("email") or "operator"
        payload["greeting"] = f"hello {who} - {_display_now()}"
        return payload


    @app.post("/sql")
    def operator_sql(
        payload: dict,
        request: Request,
        user: dict[str, Any] = Depends(get_current_user),
        settings: Settings = Depends(get_settings),
    ) -> dict:
        """Private read-only SQL console (same host gate as /status).

        Body: {"query": "<SELECT ...>"}. Only SELECT and PRAGMA statements are
        accepted; one statement per call; a 1000-row limit is applied when the
        query does not state its own LIMIT. The `df` view exposes the key
        columns of dataset_images; `select * from df where pic_id = 1` works
        exactly as requested.
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
        from roadguard_agent import Database as _SqlDb

        query = str((payload or {}).get("query") or "").strip().rstrip(";")
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        if ";" in query:
            raise HTTPException(status_code=400, detail="one statement per call")
        lead = query.split(None, 1)[0].lower() if query else ""
        if lead not in ("select", "pragma", "with", "explain"):
            raise HTTPException(
                status_code=400,
                detail="only SELECT, PRAGMA, EXPLAIN or WITH are accepted",
            )
        if " limit " not in query.lower() and lead in ("select", "with"):
            query = f"{query} LIMIT 1000"
        t0 = time.time()
        try:
            with _SqlDb(settings.roadguard_db_path) as db_sql:
                cur = db_sql._conn.execute(query)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = [list(r) for r in cur.fetchmany(1000)]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
        return {
            "query": query,
            "columns": cols,
            "rows": rows,
            "row_count": len(rows),
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    @app.get("/trace", response_class=HTMLResponse)
    @app.get("/trace/")
    def trace_viewer(request: Request) -> HTMLResponse:
        """Private live-trace page (see _private_page_or_404)."""
        return _private_page_or_404(request, "trace.html")

    @app.get("/supervisor", response_class=HTMLResponse)
    @app.get("/supervisor/")
    def supervisor_inbox(request: Request) -> HTMLResponse:
        """Private supervisor inbox: live counts, pending approvals and
        hazards waiting for citizen evidence (see _private_page_or_404)."""
        return _private_page_or_404(request, "supervisor.html")

    @app.get("/health", response_model=HealthResponse)
    def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
        agent_ok = _agent_ready(settings)
        return HealthResponse(
            status="ok" if agent_ok else "degraded",
            version=VERSION,
            db_ready=_db_ready(settings),
            agent_ready=agent_ok,
            rag_ready=_rag_ready(settings),
        )

    @app.post("/telemetry/event", status_code=status.HTTP_204_NO_CONTENT)
    def telemetry_event(body: dict, request: Request) -> Response:
        """Light-weight client telemetry sink: the webapp posts {event, meta}
        for every button click, field change and navigation transition. We
        print the event to stdout tagged [CLIENT   ] so rg-trace shows user
        actions interleaved with agent events. No DB write, no auth gate
        (anonymous page-view counts are useful): the endpoint silently
        truncates and sanitises its input so an attacker cannot flood the
        journal with huge payloads.
        """
        try:
            ev = str(body.get("event") or "")[:60]
            meta = body.get("meta") or {}
            if not ev:
                return Response(status_code=status.HTTP_204_NO_CONTENT)
            if not isinstance(meta, dict):
                meta = {"value": str(meta)[:120]}
            # cap total serialised payload at 400 chars so a hostile client
            # cannot write 1 MB lines into the journal
            parts = []
            for k, v in list(meta.items())[:8]:
                kstr = str(k)[:40]
                vstr = str(v)[:120].replace("\n", " ").replace("\r", " ")
                parts.append(f"{kstr}={vstr}")
            line = f"[CLIENT   ] event={ev}  " + "  ".join(parts)
            print(line[:600], flush=True)
        except Exception:  # pragma: no cover - telemetry must never 500
            pass
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/auth/login", response_model=LoginResponse)
    @limiter.limit(lambda: get_settings().roadguard_login_limit)
    def login(
        request: Request,
        body: LoginRequest = Body(...),
        settings: Settings = Depends(get_settings),
    ) -> LoginResponse:
        if body.google_id_token:
            payload = verify_google_id_token(
                id_token=body.google_id_token,
                client_id=settings.google_oauth_client_id,
                allowed_emails=settings.allowed_emails_set(),
            )
            token = create_jwt(
                sub=payload["sub"],
                email=payload["email"],
                display_name=payload["name"],
                settings=settings,
            )
            return LoginResponse(token=token, expires_in=settings.roadguard_jwt_ttl)
        if not body.passphrase:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="passphrase_or_google_id_token_required",
            )
        if not verify_passphrase(
            body.passphrase, settings.roadguard_passphrase_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="passphrase_invalid",
            )
        emails = sorted(settings.allowed_emails_set())
        email = emails[0] if emails else ""
        token = create_jwt(
            sub=f"passphrase:{email or 'owner'}",
            email=email,
            display_name="RoadGuard",
            settings=settings,
        )
        return LoginResponse(token=token, expires_in=settings.roadguard_jwt_ttl)

    @app.get("/auth/me", response_model=Me)
    def me(user: dict[str, Any] = Depends(get_current_user)) -> Me:
        return Me(
            sub=user.get("sub", ""),
            email=user.get("email", ""),
            name=user.get("name", ""),
        )

    # IDX 26: at most this many async graphs run at once; the rest wait
    # in the task thread with the row in status 'queued'. Ollama serves
    # one completion at a time anyway, so a higher number only adds RAM
    # pressure and timeouts.
    async_concurrency = max(
        1, int(os.environ.get("ROADGUARD_ASYNC_CONCURRENCY", "2") or 2)
    )
    app.state.async_concurrency = async_concurrency
    app.state.async_slots = threading.BoundedSemaphore(async_concurrency)

    def _record_pipeline_failure(
        db_path: str, hazard_id: str, exc: BaseException, trace_so_far: list[dict[str, Any]]
    ) -> None:
        """A crashed async run must still reach a human: the row becomes a
        human_review verdict with the error in its rationale and an
        approvals row, instead of an empty 'human_review' status nobody
        is notified about."""
        from roadguard_agent import Database

        rationale = f"pipeline error: {type(exc).__name__}; routed to a human"
        trace = [
            *trace_so_far,
            {
                "iter": len(trace_so_far),
                "speaker": "pipeline",
                "kind": "graph_failed",
                "content": {"error": type(exc).__name__},
            },
        ]
        with Database(db_path) as db2:
            db2.update_hazard_verdict(
                hazard_id=hazard_id,
                priority=50,
                agent_decision="human_review",
                agent_rationale=rationale,
                agent_trace_json=json.dumps(trace),
                cited_sources_json="[]",
                status="open",
                work_order_id=None,
                agent_assessment="",
            )
            db2.create_approval(
                hazard_id=hazard_id,
                action="human_review",
                priority=50,
                rationale=rationale,
                cited_sources_json="[]",
            )

    def _run_graph_and_persist(
        hazard_id: str,
        hazard_input: dict,
        user_dict: dict,
        body: HazardSubmit,
        db_path: str,
    ) -> None:
        """Background task: stream the graph and write the growing trace
        to the hazards row after every node, so /report/{id} shows the
        agents filling in live. Waits for a concurrency slot first."""
        from roadguard_agent import (
            Database,
            final_state_to_decision,
            hazard_to_graph_inputs,
            set_live_trace_sink,
            stream_graph_values,
        )

        def _live_sink(entry: dict[str, Any]) -> None:
            """ReAct steps and other mid-node events: append to the row the
            moment they happen, so the /trace page shows the Decision
            agent reasoning hop by hop, not only after the node ends."""
            try:
                with Database(db_path) as _dbl:
                    _dbl.append_hazard_trace_entry(hazard_id, entry)
            except Exception:
                LOGGER.warning("live_trace_append_failed hazard_id=%s", hazard_id)

        trace_so_far: list[dict[str, Any]] = []
        final_state: dict[str, Any] = {}
        started_at = time.time()
        with app.state.async_slots:
            stamps: dict[int, float] = {}
            set_live_trace_sink(_live_sink)
            try:
                with Database(db_path) as dbq:
                    dbq.set_hazard_status(hazard_id, "processing")
                _hid, yolo_output, image_quality, metadata = hazard_to_graph_inputs(
                    hazard_input
                )
                for snapshot in stream_graph_values(
                    hazard_id=hazard_id,
                    yolo_output=yolo_output,
                    image_quality=image_quality,
                    metadata=metadata,
                ):
                    final_state = dict(snapshot)
                    trace_so_far = _stamp_trace(list(snapshot.get("trace") or []), stamps)
                    try:
                        with Database(db_path) as dbp:
                            dbp.set_hazard_trace(hazard_id, json.dumps(trace_so_far))
                    except Exception:
                        LOGGER.warning(
                            "trace_partial_write_failed hazard_id=%s", hazard_id
                        )
            except Exception as exc:
                LOGGER.exception("graph_failed hazard_id=%s path=async", hazard_id)
                _record_pipeline_failure(db_path, hazard_id, exc, trace_so_far)
                return
            finally:
                set_live_trace_sink(None)
        decision = final_state_to_decision(final_state)
        waiting = _evidence_request_out(decision, hazard_id) is not None
        with Database(db_path) as db2:
            db2.update_hazard_verdict(
                hazard_id=hazard_id,
                priority=decision.priority,
                agent_decision=decision.action,
                agent_rationale=decision.rationale,
                agent_trace_json=json.dumps(_stamp_trace(decision.trace or trace_so_far, stamps)),
                cited_sources_json=json.dumps(decision.cited_sources or []),
                status="waiting_for_evidence" if waiting else "open",
                work_order_id=None,
                agent_assessment=getattr(decision, "assessment", "") or "",
            )
        # Same journal block the synchronous path prints, so an operator
        # tailing the service log sees async runs too (observability only).
        with contextlib.suppress(Exception):
            from roadguard_agent import emit_trace

            emit_trace(hazard_id, final_state, source="api_async",
                       elapsed_s=time.time() - started_at)

    @app.post("/hazards/submit-async", status_code=status.HTTP_202_ACCEPTED)
    @limiter.limit(lambda: get_settings().roadguard_user_limit)
    def hazards_submit_async(
        request: Request,
        background: BackgroundTasks,
        body: HazardSubmit = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
        settings: Settings = Depends(get_settings),
        db: Any = Depends(get_db),
    ) -> dict:
        """Fire-and-poll submit for citizens behind the 100 s Cloudflare
        edge timeout. Returns 202 + hazard_id immediately; the graph runs
        in a BackgroundTask that writes the full trace to the hazards
        row when it finishes. The caller polls GET /report/{hazard_id}
        until status leaves 'processing'."""
        from roadguard_agent import Hazard
        hazard_id = f"hz_{uuid.uuid4().hex[:12]}"
        hazard_input = _hazard_input_from_batch(body, hazard_id, _now_iso())
        user_id = _user_id_of(user)
        db.upsert_user(user_id, user.get("email", ""), user.get("name", ""))
        db.create_hazard(Hazard(
            id=hazard_id,
            user_id=user_id,
            reported_at=time.time(),
            hazard_class=body.hazard_class,
            confidence=float(body.confidence),
            road_type=body.road_type,
            lane_blocked=body.lane_blocked,
            location=body.location or "",
            bboxes_json=json.dumps(body.bboxes or []),
            priority=0,
            agent_decision="",
            agent_rationale="",
            agent_trace_json="[]",
            status="queued",
            work_order_id=None,
        ))
        background.add_task(
            _run_graph_and_persist,
            hazard_id=hazard_id,
            hazard_input=hazard_input,
            user_dict=dict(user),
            body=body,
            db_path=settings.roadguard_db_path,
        )
        return {
            "hazard_id": hazard_id,
            "status": "queued",
            "poll": f"/report/{hazard_id}",
        }

    @app.post("/hazards/submit", response_model=HazardSubmitResponse)
    @limiter.limit(lambda: get_settings().roadguard_user_limit)
    def hazards_submit(
        request: Request,
        body: HazardSubmit = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
        settings: Settings = Depends(get_settings),
        db: Any = Depends(get_db),
    ) -> HazardSubmitResponse:
        hazard_id = f"hz_{uuid.uuid4().hex[:12]}"
        hazard_input = _hazard_input_from_batch(body, hazard_id, _now_iso())
        try:
            decision = multiagent_decide(hazard_input)
        except Exception:
            # Provider failures are absorbed inside the graph; a crash here
            # is a bug and stays a 500 so it is not mistaken for a verdict.
            LOGGER.exception("graph_failed hazard_id=%s path=submit", hazard_id)
            raise
        # Roadmap section 2: a needs_more_evidence verdict leaves the graph
        # paused on its checkpoint; the reporter answers through
        # POST /evidence_requests/{id}/fulfill and the same thread resumes.
        evidence_request = _evidence_request_out(decision, hazard_id)
        waiting = evidence_request is not None
        _persist_hazard(
            db,
            user,
            hazard_id=hazard_id,
            hazard_class=body.hazard_class,
            confidence=body.confidence,
            road_type=body.road_type,
            lane_blocked=body.lane_blocked,
            location=body.location or "",
            bboxes=body.bboxes or [],
            decision=decision,
            reporter_text=body.reporter_text or "",
            status_value="waiting_for_evidence" if waiting else "open",
        )
        return HazardSubmitResponse(
            hazard_id=hazard_id,
            verdict=AgentVerdict(
                action=decision.action,
                rationale=decision.rationale,
                priority=decision.priority,
                cited_sources=decision.cited_sources,
            ),
            provider=decision.provider,
            fallback_reason=decision.fallback_reason,
            status="waiting_for_evidence" if waiting else "decided",
            evidence_request=evidence_request,
        )

    @app.post("/hazards/live", response_model=LiveHazardResponse)
    @limiter.limit(lambda: get_settings().roadguard_live_limit)
    def hazards_live(
        request: Request,
        body: LiveHazardSubmit = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
        settings: Settings = Depends(get_settings),
        cache: DecisionCache = Depends(get_decision_cache),
        db: Any = Depends(get_db),
    ) -> LiveHazardResponse:
        """Streaming submit from the live-drive screen.

        Returns the cached verdict when the same session resubmits the
        same canonical hazard within the TTL. On a miss, runs the
        3-agent graph and caches the result.
        """
        cache_key = DecisionCache.canonical_key(
            session_id=body.session_id,
            hazard_class=body.hazard_class,
            road_type=body.road_type,
            lane_blocked=body.lane_blocked,
            track_id=body.track_id,
        )

        cached = cache.get(cache_key)
        if cached is not None:
            return LiveHazardResponse(
                hazard_id=f"hz_live_{uuid.uuid4().hex[:12]}",
                verdict=AgentVerdict(
                    action=cached.action,
                    rationale=cached.rationale,
                    priority=cached.priority,
                    cited_sources=cached.cited_sources,
                ),
                provider=cached.provider,
                fallback_reason=cached.fallback_reason,
                cached=True,
                cache_key=cache_key,
            )

        hazard_id = f"hz_live_{uuid.uuid4().hex[:12]}"
        reported_at = _now_iso()

        try:
            decision = multiagent_decide_live(
                hazard_id=hazard_id,
                submit=_submit_dict_from_live(body, reported_at),
            )
        except Exception:
            LOGGER.exception("graph_failed hazard_id=%s path=live", hazard_id)
            raise

        cache.put(cache_key, decision)
        # Only cache MISSES persist: a hit is the same physical hazard
        # re-detected across frames of one session, not a new report.
        _persist_hazard(
            db,
            user,
            hazard_id=hazard_id,
            hazard_class=body.hazard_class,
            confidence=body.confidence,
            road_type=body.road_type,
            lane_blocked=body.lane_blocked,
            location=body.location or "",
            bboxes=[list(body.bbox)] if body.bbox else [],
            decision=decision,
        )

        return LiveHazardResponse(
            hazard_id=hazard_id,
            verdict=AgentVerdict(
                action=decision.action,
                rationale=decision.rationale,
                priority=decision.priority,
                cited_sources=decision.cited_sources,
            ),
            provider=decision.provider,
            fallback_reason=decision.fallback_reason,
            cached=False,
            cache_key=cache_key,
        )

    @app.get("/hazards/history", response_model=HazardHistoryResponse)
    def hazards_history(
        user: dict[str, Any] = Depends(get_current_user),
        db=Depends(get_db),
    ) -> HazardHistoryResponse:
        rows = db.list_user_hazards(_user_id_of(user), limit=50)
        return HazardHistoryResponse(
            items=[
                HazardHistoryItem(
                    hazard_id=r.id,
                    reported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime(r.reported_at)),
                    **{"class": r.hazard_class},
                    confidence=r.confidence,
                    road_type=r.road_type,
                    lane_blocked=r.lane_blocked,
                    location=r.location,
                    priority=r.priority,
                    agent_decision=r.agent_decision,
                    agent_rationale=r.agent_rationale,
                    status=r.status,
                )
                for r in rows
            ],
        )

    @app.get("/hazards/{hazard_id}", response_model=HazardDetail)
    def hazard_detail(
        hazard_id: str,
        user: dict[str, Any] = Depends(get_current_user),
        db=Depends(get_db),
    ) -> HazardDetail:
        r = db.get_hazard(hazard_id, user_id=_user_id_of(user))
        if r is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="hazard_not_found",
            )
        trace = json.loads(r.agent_trace_json or "[]")
        cited: list[str] = []
        for entry in trace:
            content = entry.get("content") or {}
            if isinstance(content, dict) and content.get("kind") in (None, "final"):
                for src in content.get("cited_sources") or []:
                    if src not in cited:
                        cited.append(src)
        return HazardDetail(
            hazard_id=r.id,
            reported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(r.reported_at)),
            **{"class": r.hazard_class},
            confidence=r.confidence,
            road_type=r.road_type,
            lane_blocked=r.lane_blocked,
            location=r.location,
            priority=r.priority,
            agent_decision=r.agent_decision,
            agent_rationale=r.agent_rationale,
            status=r.status,
            agent_trace=trace,
            cited_sources=cited,
        )

    @app.post("/rag/query", response_model=RagQueryResponse)
    @limiter.limit(lambda: get_settings().roadguard_user_limit)
    def rag_query(
        request: Request,
        body: RagQueryRequest = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
    ) -> RagQueryResponse:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="rag_not_wired_yet",
        )

    @app.get("/report/{hazard_id}")
    def report_hazard(
        hazard_id: str,
        user: dict[str, Any] = Depends(get_current_user),
        db=Depends(get_db),
    ) -> dict:
        """Full case report: metadata, verdict, per-agent trace, grounded
        sources, decision timeline, token usage."""
        r = db.get_hazard(hazard_id, user_id=_user_id_of(user))
        if r is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="hazard_not_found",
            )
        trace = json.loads(r.agent_trace_json or "[]")
        timeline = []
        tokens_in = tokens_out = 0
        cited: list[str] = []
        llm_calls: list[dict[str, Any]] = []
        for entry in trace:
            content = entry.get("content") or {}
            timeline.append({
                "iter": entry.get("iter"),
                "speaker": entry.get("speaker"),
                "kind": entry.get("kind"),
                "llm_used": content.get("llm_used") if isinstance(content, dict) else None,
                "source": content.get("source") if isinstance(content, dict) else None,
            })
            if isinstance(content, dict):
                tokens_in += int(content.get("tokens_in", 0) or 0)
                tokens_out += int(content.get("tokens_out", 0) or 0)
                for src in content.get("cited_sources") or []:
                    if src not in cited:
                        cited.append(src)
                if content.get("llm_used"):
                    llm_calls.append({
                        "speaker": entry.get("speaker"),
                        "model": content.get("model"),
                        "tokens_in": content.get("tokens_in"),
                        "tokens_out": content.get("tokens_out"),
                    })
        evidence_request = None
        if r.status == "waiting_for_evidence":
            pending = db.list_pending_evidence_requests(hazard_id=hazard_id, limit=1)
            if pending:
                p = pending[0]
                evidence_request = {
                    "id": p["id"],
                    "hazard_id": hazard_id,
                    "missing_field": p["missing_field"],
                    "message": p["reason_shown_to_reporter"],
                    "status": p["status"],
                }
        return {
            "hazard_id": r.id,
            "reported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(r.reported_at)),
            "metadata": {
                "class": r.hazard_class,
                "confidence": r.confidence,
                "road_type": r.road_type,
                "lane_blocked": r.lane_blocked,
                "location": r.location,
            },
            "verdict": {
                "action": r.agent_decision,
                "rationale": r.agent_rationale,
                "priority": r.priority,
                "status": r.status,
                "assessment": r.agent_assessment or "",
            },
            "evidence_request": evidence_request,
            "grounded_sources": cited,
            "decision_timeline": timeline,
            "llm_calls": llm_calls,
            "token_usage": {"in": tokens_in, "out": tokens_out},
            "trace_lines": format_trace_lines(
                r.id,
                trace,
                metadata={
                    "class": r.hazard_class,
                    "confidence": r.confidence,
                    "road_type": r.road_type,
                    "lane_blocked": r.lane_blocked,
                    "location": r.location,
                },
                verdict={"action": r.agent_decision, "priority": r.priority},
                status=r.status,
            ),
            "trace": trace,
        }

    @app.get("/report")
    def report_recent(
        user: dict[str, Any] = Depends(get_current_user),
        db=Depends(get_db),
    ) -> dict:
        """Operator overview: last 50 hazards for this user, counts by
        action and class, pending approvals and evidence requests."""
        rows = db.list_user_hazards(_user_id_of(user), limit=50)
        by_action: dict[str, int] = {}
        by_class: dict[str, int] = {}
        for r in rows:
            by_action[r.agent_decision] = by_action.get(r.agent_decision, 0) + 1
            by_class[r.hazard_class] = by_class.get(r.hazard_class, 0) + 1
        pending = db.list_pending_approvals(limit=20)
        waiting_evidence = db.list_pending_evidence_requests(limit=20)
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_hazards": len(rows),
            "by_action": by_action,
            "by_class": by_class,
            "pending_approvals": [
                {
                    "approval_id": a.id,
                    "hazard_id": a.hazard_id,
                    "action": a.action,
                    "priority": a.priority,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime(a.created_at)),
                }
                for a in pending
            ],
            "waiting_for_evidence": [
                {
                    "request_id": e.get("id"),
                    "hazard_id": e.get("hazard_id"),
                    "missing_field": e.get("missing_field"),
                    "created_at": e.get("created_at"),
                }
                for e in waiting_evidence
            ],
        }

    # ---- Approval gate (Charter 5-6) ------------------------------------

    @app.get("/approvals/{approval_id}", response_class=HTMLResponse)
    def approval_page(
        approval_id: str,
        t: str | None = None,
        db=Depends(get_db),
        settings: Settings = Depends(get_settings),
    ):
        """Lightweight HTML page a supervisor opens from the email.

        No login, so the link must carry the signed token (`t`) the
        email was sent with; an id alone, or a forwarded link with a
        broken token, does not open the page.
        """
        if not verify_approval(approval_id, t, settings.roadguard_jwt_secret):
            return HTMLResponse(
                _APPROVAL_LINK_INVALID_HTML, status_code=status.HTTP_403_FORBIDDEN
            )
        try:
            row = db.get_approval(approval_id)
        except Exception:
            row = None
        if row is None:
            return HTMLResponse(
                "<h2>Approval not found</h2>", status_code=status.HTTP_404_NOT_FOUND
            )
        # The approvals row has no assessment column; the hazard row the
        # API wrote after the run carries it (no row: no block).
        assessment = ""
        with contextlib.suppress(Exception):
            hazard = db.get_hazard(row.hazard_id)
            assessment = getattr(hazard, "agent_assessment", "") or ""
        return HTMLResponse(
            _render_approval_html(row, token=t or "", assessment=assessment)
        )

    @app.post("/approvals/{approval_id}/decide")
    @limiter.limit(lambda: get_settings().roadguard_decide_limit)
    def approval_decide(
        request: Request,
        approval_id: str,
        decision: str = Form(...),
        note: str = Form(""),
        t: str = Form(""),
        db=Depends(get_db),
        settings: Settings = Depends(get_settings),
    ):
        """Approve or deny a pending approval. Called by the HTML page.

        `t` is the signed token from the link (the page passes it on as a
        hidden field); without it the decision is refused. `note` is the
        optional free text from the page's textarea; it is stored on the
        approvals row."""
        if not verify_approval(approval_id, t, settings.roadguard_jwt_secret):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="approval_link_invalid",
            )
        if decision not in ("approved", "denied"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid_decision",
            )
        # In production we would attach the authenticated supervisor
        # identity. For now the gate is anonymous but rate-limited.
        ok = db.decide_approval(
            approval_id, decision, decided_by="supervisor", note=note
        )
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="approval_not_pending_or_missing",
            )
        return RedirectResponse(
            url=f"/approvals/{approval_id}?t={t.strip()}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # ---- Evidence requests (roadmap section 2) --------------------------

    @app.get("/evidence_requests")
    def list_evidence_requests(
        hazard_id: str | None = None,
        user: dict[str, Any] = Depends(get_current_user),
        db=Depends(get_db),
    ):
        """Pending evidence requests on the caller's own hazards.

        request_additional_evidence writes the rows; /hazards/submit
        already returns the request for the case just submitted, so this
        endpoint is for listing what is still open (query param hazard_id
        to narrow). Other reporters' requests are never listed. The
        reporter answers through POST /evidence_requests/{id}/fulfill,
        which resumes the paused run.
        """
        return {
            "items": db.list_pending_evidence_requests(
                hazard_id=hazard_id, user_id=_user_id_of(user)
            )
        }

    @app.post(
        "/evidence_requests/{request_id}/fulfill",
        response_model=HazardSubmitResponse,
    )
    @limiter.limit(lambda: get_settings().roadguard_user_limit)
    def fulfill_evidence_request(
        request: Request,
        request_id: str,
        body: HazardSubmit = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
        db=Depends(get_db),
    ) -> HazardSubmitResponse:
        """The reporter answers the agent's one evidence request.

        The body is the new upload in the same shape as /hazards/submit.
        Only the hazard's owner may answer; anyone else gets the same 404
        as a missing request. The row is claimed (in_progress) so exactly
        one caller resumes the paused LangGraph thread (thread_id ==
        hazard_id); it becomes fulfilled only once the resume returned,
        and goes back to pending if the resume raised. The final verdict
        overwrites the hazard row. Returns the same shape as
        /hazards/submit with status "decided".
        """
        req = db.get_evidence_request(request_id)
        if req is None or req.get("status") != "pending":
            raise _evidence_not_found()
        hazard_id = str(req["hazard_id"])
        if db.get_hazard(hazard_id, user_id=_user_id_of(user)) is None:
            raise _evidence_not_found()
        if not db.claim_evidence_request(request_id):
            raise _evidence_not_found()

        hazard_input = _hazard_input_from_batch(body, hazard_id, _now_iso())
        try:
            decision = multiagent_resume(hazard_id, hazard_input)
        except Exception:
            LOGGER.exception("graph_failed hazard_id=%s path=resume", hazard_id)
            try:
                db.release_evidence_request(request_id)
            except Exception as release_exc:
                LOGGER.error(
                    "evidence_request_release_failed id=%s: %s", request_id, release_exc
                )
            raise
        try:
            db.fulfill_evidence_request(request_id)
        except Exception as fulfill_exc:
            # The verdict and its approval row already exist; a failure
            # here must not turn a decided case into a 500. The row stays
            # in_progress and is expired by the retention sweep.
            LOGGER.error(
                "evidence_request_fulfill_failed id=%s hazard_id=%s: %s",
                request_id, hazard_id, fulfill_exc,
            )

        _update_hazard_after_resume(db, hazard_id, decision)
        return HazardSubmitResponse(
            hazard_id=hazard_id,
            verdict=AgentVerdict(
                action=decision.action,
                rationale=decision.rationale,
                priority=decision.priority,
                cited_sources=decision.cited_sources,
            ),
            provider=decision.provider,
            fallback_reason=decision.fallback_reason,
            status="decided",
            evidence_request=None,
        )

    # Mount the webapp at the repo root last, so API routes above take
    # precedence over the static fallback. Phones can then hit the same
    # cloudflared tunnel URL both for the API and for the webapp assets.
    _webapp_dir = _REPO_ROOT / "webapp"
    if _webapp_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=str(_webapp_dir), html=True), name="webapp"
        )

    return app


app = create_app()

