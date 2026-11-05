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
import secrets
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
    IdenticalEvidenceError,
    Settings,
    create_jwt,
    decode_jwt,
    format_trace_lines,
    get_current_user,
    get_settings,
    hash_password,
    multiagent_decide,
    multiagent_decide_live,
    multiagent_resume,
    sanitize_location,
    validate_password,
    validate_username,
    verify_password,
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
  // Strip optional leading timestamp so lines carrying a clock classify
  // the same as bare lines. Handles the new "YYYY-MM-DD HH:MM:SS " prefix
  // and keeps the old "HH:MM:SS " shape working for already-cached traces.
  const body = line.replace(/^(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2}:\d{2} /, "");
  if (body.startsWith("[DECISION") || body.startsWith("[DONE")) return "dec";
  if (body.startsWith("[")) return "lbl";
  if (/^[=-]{10,}$/.test(body) || body.startsWith("AGENT RUN")) return "hdr";
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

let current = null, printed = 0, active = false, timer = null, lastErr = "", idleShown = false, justCompleted = false;

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
      if (newest !== current) { current = newest; printed = 0; active = true; justCompleted = false; print(""); }
      else return;
    }
    const rep = await get("/report/" + encodeURIComponent(current));
    const lines = rep.trace_lines || [];
    for (let i = printed; i < lines.length; i++) print(lines[i], classOf(lines[i]));
    printed = lines.length;
    const st = (rep.verdict || {}).status;
    const wasActive = active;
    active = (st === "queued" || st === "processing");
    if (wasActive && !active) justCompleted = true;
    cursor.classList.toggle("on", active);
    lastErr = "";
  } catch (err) {
    if (err.message === "401") { showPrompt(); return; }
    // After a run finishes, the tunnel's 100 s idle timeout can throw
    // 502/504 on the next poll. Those are transient and come after a
    // successful DONE line, so swallow them silently; only surface a
    // real error during an active run.
    if (justCompleted && (err.message === "502" || err.message === "504" || err.message === "Failed to fetch")) return;
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

async function replayHistory() {
  // On first open, print the trace of every recent hazard oldest-first,
  // then let the live tick() keep following new ones. Without this the
  // page shows a greeting + empty terminal when nothing new is being
  // submitted RIGHT NOW, which hides the whole last batch run.
  try {
    const h = await get("/hazards/history?limit=200");
    const items = (h.items || []).slice().reverse();  // oldest first
    if (!items.length) return;
    print("--- replaying " + items.length + " recent hazard(s) ---", "hdr");
    // Count the ones we actually render vs the ones with empty / failing
    // traces so the header never claims more than the operator sees.
    let rendered = 0, emptyTrace = 0, failed = 0;
    for (const item of items) {
      try {
        const rep = await get("/report/" + encodeURIComponent(item.hazard_id));
        const lines = rep.trace_lines || [];
        if (!lines.length) { emptyTrace++; continue; }
        for (const line of lines) print(line, classOf(line));
        print("");
        rendered++;
      } catch (e) { failed++; }
    }
    const skipped = emptyTrace + failed;
    if (skipped > 0) {
      print(
        "--- end of replay; rendered " + rendered + "/" + items.length +
        " (skipped " + skipped + ": " + emptyTrace + " empty, " + failed + " errored); following live ---",
        "hdr"
      );
    } else {
      print("--- end of replay; following live ---", "hdr");
    }
    // Mark the newest as "already seen" so the live tick does not
    // reprint it. tick() uses `current` + `printed` to resume.
    const newest = items[items.length - 1];
    current = newest.hazard_id;
    try {
      const rep = await get("/report/" + encodeURIComponent(current));
      printed = (rep.trace_lines || []).length;
      active = (((rep.verdict || {}).status) === "queued" || ((rep.verdict || {}).status) === "processing");
    } catch (e) { printed = 0; }
  } catch (err) {
    print("history replay failed: " + err.message, "err");
  }
}

async function follow() {
  stop();
  try { await banner(); } catch (err) { showPrompt(); return; }
  await replayHistory();
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
.row .top .act.overdue { background: rgba(245,158,11,.22); color: #b45309; }
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
      (a.overdue ? ' <span class="act overdue">overdue ' + a.age_hours + 'h</span>' : '') +
      '</div>' +
    '<div class="meta">approval ' + a.approval_id + ' - priority ' + prio +
      ' - created ' + a.created_at + (a.age_hours != null ? ' - waiting ' + a.age_hours + 'h' : '') + '</div>' +
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
_FLOW_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow">
<title>RoadGuard - Agent Flow (private)</title>
<style>
:root {
  /* Match the RoadGuard LIVE webapp (orarr2.github.io/RoadGuard) palette:
     bright surfaces, sky-to-emerald accents, dark readable text. */
  --bg: #ecfeff;         /* page background */
  --card: #ffffff;        /* card surface */
  --card2: #f0f9ff;       /* nested surface */
  --border: rgba(14, 165, 233, 0.18);
  --border-2: rgba(14, 165, 233, 0.32);
  --text: #0b1a2b;
  --text-mid: #334155;
  --dim: #64748b;
  --fade: #94a3b8;
  --idle: #cbd5e1;
  --active: #0ea5e9;      /* sky - the running color */
  --ok: #22c55e;          /* emerald - done / standard */
  --info: #0284c7;        /* darker sky - needs evidence */
  --warn: #f59e0b;        /* amber - awaiting approval */
  --bad: #ef4444;         /* red - urgent / human review */
  --brand-1: #0ea5e9;
  --brand-2: #06b6d4;
  --brand-3: #22c55e;
  --brand-grad: linear-gradient(120deg, #0ea5e9, #06b6d4 50%, #22c55e);
  --shadow-sm: 0 2px 8px rgba(14, 165, 233, 0.08);
  --shadow-md: 0 10px 30px rgba(14, 165, 233, 0.14);
}
* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg); color: var(--text);
  font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  min-height: 100vh;
  overflow-x: hidden;
}
header {
  display: flex; align-items: center; gap: 14px;
  padding: 14px 24px;
  background: var(--card);
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 10;
  flex-wrap: wrap;
  box-shadow: var(--shadow-sm);
}
header h1 {
  margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.01em;
  background: var(--brand-grad); -webkit-background-clip: text;
  background-clip: text; color: transparent;
}
.badge {
  padding: 4px 10px; border-radius: 999px; font-size: 11px;
  font-weight: 700; letter-spacing: 0.5px;
  background: rgba(34,197,94,.14); color: var(--ok);
}
.badge.paused { background: rgba(239,68,68,.12); color: var(--bad); }
#hz-display { color: var(--dim); font-family: "JetBrains Mono", ui-monospace, Consolas, monospace; font-size: 12px; }
.header-spacer { flex: 1; }
button {
  background: var(--card2); color: var(--text);
  border: 1px solid var(--border-2);
  border-radius: 8px; padding: 8px 16px;
  font-size: 13px; font-weight: 500; cursor: pointer;
  font-family: inherit;
  min-height: 38px;
  transition: all 0.15s;
}
button:hover { background: var(--brand-1); color: #ffffff; border-color: var(--brand-1); }
button:active { transform: scale(0.97); }
/* Round 23 (4.11.2026) layout fix: the flow diagram spans the row, and
   the four side panels (Current hazard + KPI + Run cost + Agent
   advantage) sit side-by-side in a single row BELOW it, so the operator
   does not scroll a three-section sidebar on every visit. Collapses to
   two columns on tablets and one on phones. */
main {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
  padding: 14px;
  max-width: 1600px;
  margin: 0 auto;
}
aside {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  align-items: start;
}
@media (max-width: 1100px) {
  aside { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  aside { grid-template-columns: 1fr; }
}
aside .panel { margin-top: 0 !important; }
.flow-container {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 18px 26px 18px;
  overflow-x: auto;
  position: relative;
  min-height: 280px;
  box-shadow: var(--shadow-md);
}
.flow {
  position: relative;
  display: grid;
  grid-template-columns: repeat(6, minmax(140px, 1fr));
  grid-template-rows: auto auto;
  gap: 24px 28px;
  padding: 10px 0 20px 0;
  min-width: 980px;
}
.arrows {
  position: absolute; top: 0; left: 0;
  pointer-events: none;
  z-index: 1;
}
.node {
  grid-row: 1;
  background: var(--card);
  border: 2px solid var(--idle);
  border-radius: 12px;
  padding: 14px;
  position: relative;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.15s ease, background 0.3s ease;
  cursor: pointer;
  z-index: 2;
  box-shadow: var(--shadow-sm);
}
.node:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(14,165,233,0.12); }
.node.state-active {
  border-color: var(--active);
  background: rgba(14, 165, 233, 0.06);
  box-shadow: 0 0 24px rgba(14,165,233,0.35);
  animation: pulse 1.6s ease-in-out infinite;
}
.node.state-done { border-color: var(--ok); background: rgba(34, 197, 94, 0.04); }
.node.state-info { border-color: var(--info); background: rgba(2, 132, 199, 0.04); }
.node.state-warn { border-color: var(--warn); background: rgba(245, 158, 11, 0.04); }
.node.state-bad {
  border-color: var(--bad);
  background: rgba(239, 68, 68, 0.04);
  box-shadow: 0 0 24px rgba(239,68,68,0.3);
}
.node .node-head {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 6px;
  font-weight: 700; font-size: 13px;
}
.node .node-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--idle);
  box-shadow: 0 0 0 3px rgba(74,90,122,0.15);
}
.node.state-active .node-dot { background: var(--active); box-shadow: 0 0 0 3px rgba(251,191,36,0.3); animation: dot-blink 0.9s steps(2) infinite; }
.node.state-done .node-dot { background: var(--ok); box-shadow: 0 0 0 3px rgba(34,197,94,0.25); }
.node.state-info .node-dot { background: var(--info); box-shadow: 0 0 0 3px rgba(56,189,248,0.25); }
.node.state-warn .node-dot { background: var(--warn); box-shadow: 0 0 0 3px rgba(245,158,11,0.25); }
.node.state-bad .node-dot { background: var(--bad); box-shadow: 0 0 0 3px rgba(239,68,68,0.25); }
.node .node-sub {
  font-size: 11px; color: var(--dim); line-height: 1.3;
  min-height: 28px;
}
.node .node-caption {
  font-family: "JetBrains Mono", ui-monospace, Consolas, monospace; font-size: 11px;
  color: var(--text-mid); margin-top: 6px;
  padding-top: 6px;
  border-top: 1px dashed var(--border);
  min-height: 15px;
  word-break: break-word;
}
.node[data-key="await_evidence"] {
  grid-row: 2; grid-column: 5;
  opacity: 0.4;
  border-style: dashed;
}
.node[data-key="await_evidence"].visible { opacity: 1; border-style: solid; }
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 20px rgba(14,165,233,0.35); }
  50% { box-shadow: 0 0 34px rgba(14,165,233,0.6); }
}
@keyframes dot-blink { to { visibility: hidden; } }
.panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px;
  box-shadow: var(--shadow-md);
}
.panel h2 {
  margin: 0 0 12px 0; font-size: 11px; color: var(--dim);
  letter-spacing: 1px; text-transform: uppercase; font-weight: 700;
}
.panel .kv {
  display: grid; grid-template-columns: 90px 1fr;
  gap: 7px 10px; font-size: 12px;
}
.panel .kv .k { color: var(--dim); font-weight: 500; }
.panel .kv .v {
  color: var(--text); font-family: "JetBrains Mono", ui-monospace, Consolas, monospace;
  word-break: break-all;
}
.verdict-box {
  margin-top: 16px; padding: 14px; border-radius: 10px;
  background: var(--card2);
  border-left: 4px solid var(--idle);
}
.verdict-box.urgent, .verdict-box.human_review { border-left-color: var(--bad); background: rgba(239,68,68,0.06); }
.verdict-box.standard { border-left-color: var(--ok); background: rgba(34,197,94,0.06); }
.verdict-box.needs_more_evidence { border-left-color: var(--info); background: rgba(2,132,199,0.06); }
.verdict-box.known_image { border-left-color: var(--info); background: rgba(2,132,199,0.06); }
.verdict-box .action {
  font-weight: 700; font-size: 14px; letter-spacing: 0.5px;
  text-transform: uppercase;
}
.verdict-box.urgent .action, .verdict-box.human_review .action { color: var(--bad); }
.verdict-box.standard .action { color: var(--ok); }
.verdict-box.needs_more_evidence .action, .verdict-box.known_image .action { color: var(--info); }
.verdict-box .rationale {
  margin-top: 8px; color: var(--text-mid); font-size: 11px; line-height: 1.5;
  max-height: 100px; overflow-y: auto;
}
.live-line {
  margin-top: 16px;
  padding: 10px 12px;
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 10px;
  font-family: "JetBrains Mono", ui-monospace, Consolas, monospace; font-size: 11px;
  color: var(--text);
  min-height: 44px;
  max-height: 120px; overflow-y: auto;
  word-break: break-word;
  white-space: pre-wrap;
}
.live-line .lbl { color: var(--ok); }
.live-line .dec { color: var(--warn); }
footer {
  padding: 0 14px 14px 14px;
  max-width: 1600px;
  margin: 0 auto;
}
/* SNA-style flow enhancements: parallel-path edges + animated flow pulse.
   Default edges are the solid pipeline (sanitize - signal - supervisor -
   profile - rag - decision - approval). Parallel marks the profile/rag
   lane that can run concurrently with signal in some verdicts. Loopback
   is the await-evidence return to sanitize when the decision asks for
   more photos. The .active class drives the animated dash so the viewer
   can SEE which edge the process is traversing right now. */
.arrows path, .arrows line { fill: none; stroke-linecap: round; }
.edge-main { stroke: var(--border-2); stroke-width: 2; }
.edge-parallel { stroke: var(--brand-2); stroke-width: 2; stroke-dasharray: 6 4; opacity: 0.65; }
.edge-loopback { stroke: var(--warn); stroke-width: 2; stroke-dasharray: 4 6; opacity: 0.8; }
.edge-main.active, .edge-parallel.active, .edge-loopback.active {
  stroke: var(--active); stroke-width: 3; opacity: 1;
  stroke-dasharray: 12 6;
  animation: flow-dash 1.1s linear infinite;
  filter: drop-shadow(0 0 6px rgba(14,165,233,0.55));
}
@keyframes flow-dash {
  to { stroke-dashoffset: -36; }
}
.edge-label {
  font-family: "JetBrains Mono", ui-monospace, Consolas, monospace;
  font-size: 10px; fill: var(--dim);
  pointer-events: none;
}
.edge-label.active { fill: var(--active); font-weight: 700; }

#login {
  position: fixed; inset: 0;
  background: rgba(14, 165, 233, 0.14);
  backdrop-filter: blur(8px);
  display: none; align-items: center; justify-content: center;
  z-index: 100; padding: 20px;
}
#login.on { display: flex; }
#login form {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 28px;
  min-width: 280px;
  max-width: 360px;
  width: 100%;
  box-shadow: var(--shadow-md);
}
#login h3 {
  margin: 0 0 14px 0; font-size: 16px; font-weight: 700;
  background: var(--brand-grad); -webkit-background-clip: text;
  background-clip: text; color: transparent;
}
#login input[type="password"] {
  width: 100%; padding: 12px;
  background: var(--card2); color: var(--text);
  border: 1px solid var(--border-2);
  border-radius: 8px; font-size: 15px;
  font-family: inherit;
}
#login button {
  width: 100%; margin-top: 10px;
  background: var(--brand-grad); color: #ffffff;
  border: 0; padding: 12px; font-weight: 700;
  border-radius: 8px;
  min-height: 44px;
}
#login .err {
  color: var(--bad); font-size: 12px; margin-top: 8px;
  min-height: 16px;
}
.empty-state {
  text-align: center; color: var(--dim);
  padding: 30px 20px;
  font-size: 13px;
}
.legend {
  display: flex; gap: 14px; flex-wrap: wrap;
  margin-top: 16px; padding-top: 12px;
  border-top: 1px solid var(--border);
  font-size: 11px; color: var(--dim);
}
.legend .lg { display: inline-flex; align-items: center; gap: 6px; }
.legend .lg-dot {
  width: 10px; height: 10px; border-radius: 50%;
  display: inline-block;
  box-shadow: 0 0 0 1px rgba(14,165,233,0.15);
}
.legend .lg-line {
  width: 22px; height: 2px; display: inline-block;
}
.legend .lg-line.parallel { border-top: 2px dashed var(--brand-2); background: transparent; height: 0; }
.legend .lg-line.loopback { border-top: 2px dashed var(--warn); background: transparent; height: 0; }
</style>
</head>
<body>

<header>
  <h1>RoadGuard Agent Flow</h1>
  <span id="status-badge" class="badge">LIVE</span>
  <span id="hz-display"></span>
  <div class="header-spacer"></div>
  <button id="pause-btn">Pause</button>
</header>

<main>
  <div class="flow-container">
    <svg class="arrows" id="arrows"></svg>
    <div class="flow" id="flow"></div>
    <div class="empty-state" id="empty-state" style="display:none">
      No hazard being processed. Submit one from the webapp to see the agent at work.
    </div>
    <div class="legend">
      <span class="lg"><span class="lg-dot" style="background:var(--idle)"></span>idle</span>
      <span class="lg"><span class="lg-dot" style="background:var(--active)"></span>running</span>
      <span class="lg"><span class="lg-dot" style="background:var(--ok)"></span>done / standard</span>
      <span class="lg"><span class="lg-dot" style="background:var(--info)"></span>needs evidence</span>
      <span class="lg"><span class="lg-dot" style="background:var(--warn)"></span>awaiting approval</span>
      <span class="lg"><span class="lg-dot" style="background:var(--bad)"></span>urgent / human review</span>
      <span class="lg"><span class="lg-line" style="background:var(--border-2)"></span>sequential edge</span>
      <span class="lg"><span class="lg-line parallel"></span>parallel (profile + rag)</span>
      <span class="lg"><span class="lg-line loopback"></span>loopback (await evidence)</span>
    </div>
  </div>

  <aside>
    <div class="panel">
      <h2>Current hazard</h2>
      <div class="kv">
        <span class="k">hazard id</span><span class="v" id="kv-id">&mdash;</span>
        <span class="k">class</span><span class="v" id="kv-class">&mdash;</span>
        <span class="k">confidence</span><span class="v" id="kv-conf">&mdash;</span>
        <span class="k">road type</span><span class="v" id="kv-road">&mdash;</span>
        <span class="k">lane</span><span class="v" id="kv-lane">&mdash;</span>
        <span class="k">status</span><span class="v" id="kv-status">&mdash;</span>
      </div>
      <div class="verdict-box" id="verdict-box">
        <div class="action" id="verdict-action">WAITING</div>
        <div class="rationale" id="verdict-rationale">No hazard selected.</div>
      </div>
      <div class="live-line" id="live-line">waiting for events...</div>
    </div>
    <!-- Round 23 (4.11.2026): live KPI vs targets, per-agent justified
         ratios, run cost totals, advantage-over-tree case ids. Polled
         from /stats/kpi, /stats/agents and /stats/advantage every 30s.
         Each row carries a short inline note the owner can read without
         opening the trace. -->
    <div class="panel" id="kpi-panel" style="margin-top:14px">
      <h2>KPI vs targets (24h)</h2>
      <div id="kpi-rows" class="kpi-rows">loading...</div>
    </div>
    <div class="panel" id="cost-panel" style="margin-top:14px">
      <h2>Run cost (24h)</h2>
      <div id="cost-rows" class="kv">loading...</div>
    </div>
    <div class="panel" id="advantage-panel" style="margin-top:14px">
      <h2>Agent advantage (24h)</h2>
      <div id="advantage-rows" class="kv">loading...</div>
    </div>
  </aside>
</main>
<style>
  .kpi-rows .kpi-row { display:flex; justify-content:space-between; gap:8px; align-items:center;
                       padding:6px 0; border-bottom:1px solid var(--border); font-size:13px; }
  .kpi-rows .kpi-row:last-child { border-bottom:none; }
  .kpi-rows .kpi-label { color: var(--text); font-weight: 600; }
  .kpi-rows .kpi-value { font-family: "JetBrains Mono", ui-monospace, Consolas, monospace; font-size: 13px; }
  .kpi-rows .kpi-value.ok { color: var(--ok); }
  .kpi-rows .kpi-value.fail { color: var(--bad); }
  .kpi-rows .kpi-note { font-size: 11px; color: var(--dim); padding-top:2px; }
  #advantage-rows .adv-case { font-family: "JetBrains Mono", ui-monospace, Consolas, monospace;
                              font-size: 11px; color: var(--text-mid); word-break: break-all; }
</style>


<!-- The passphrase modal was removed. /flow is reached through the Caddy
     basicauth operator proxy on the tailnet, so by the time this page
     renders the viewer has already passed one auth gate; a second one
     only added friction (and was serving an API path Caddy returns 501
     on). Public hosts never see this page because the FastAPI route
     returns 404 for anything that is not a private host. -->
<div id="login" hidden><form id="login-form" autocomplete="off" hidden>
  <input id="pw" type="password" autocomplete="new-password" hidden>
  <div class="err" id="login-err"></div>
</form></div>

<script>
// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const NODES = [
  { key: "sanitize",         label: "Sanitize",         col: 1, hint: "redacts PII, neutralises injection" },
  { key: "signal_extractor", label: "Signal Extractor", col: 2, hint: "reads reporter free text" },
  { key: "profile_builder",  label: "Profile Builder",  col: 3, hint: "YOLO + metadata to profile" },
  { key: "rag_retriever",    label: "RAG Retriever",    col: 4, hint: "queries repair standards" },
  { key: "decision",         label: "Decision",         col: 5, hint: "ReAct loop, cites sources" },
  { key: "approval_gate",    label: "Approval Gate",    col: 6, hint: "pends human sign-off" },
  { key: "await_evidence",   label: "Await Evidence",   col: 5, hint: "paused on citizen reply", row: 2 },
];
const MAIN_ORDER = ["sanitize","signal_extractor","profile_builder","rag_retriever","decision","approval_gate"];
const POLL_MS = 2000;

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
function el(id) { return document.getElementById(id); }
function create(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function svgNode(tag, attrs) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------
let token = "";
try { token = localStorage.getItem("rg_token") || ""; } catch (e) {}

function apiPath(path) {
  const here = window.location.pathname;
  const idx = here.indexOf("/flow");
  return (idx >= 0 ? here.substring(0, idx) : "") + path;
}

async function apiGet(path) {
  // Private operator pages trust Caddy's basicauth gate: no bearer token
  // is attached. The server-side API endpoints that back the page
  // (/hazards/history, /report/{id}) short-circuit on a private host.
  const r = await fetch(apiPath(path));
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

// ---------------------------------------------------------------------------
// Flow diagram - build DOM nodes once
// ---------------------------------------------------------------------------
function buildFlow() {
  const flow = el("flow");
  flow.innerHTML = "";
  for (const n of NODES) {
    const node = create("div", "node state-idle");
    node.dataset.key = n.key;
    node.style.gridColumn = String(n.col);
    if (n.row) node.style.gridRow = String(n.row);

    const head = create("div", "node-head");
    head.appendChild(create("span", "node-dot"));
    head.appendChild(create("span", null, n.label));
    node.appendChild(head);

    node.appendChild(create("div", "node-sub", n.hint));
    node.appendChild(create("div", "node-caption", "idle"));

    flow.appendChild(node);
  }
}

// SNA-style edge map. Each edge declares its topology (which nodes it
// connects) and its kind (main / parallel / loopback). The "activeWhen"
// predicate decides, given the current nodeState, whether the edge is
// the one the process is traversing RIGHT NOW: so the viewer sees a
// pulsing cyan arrow walk the pipeline, dashed turquoise lines mark the
// profile+rag parallel band, and a dashed amber arrow curls back from
// decision to sanitize when the Decision agent asks for more evidence.
const EDGES = [
  { from: "sanitize",         to: "signal_extractor", kind: "main" },
  { from: "signal_extractor", to: "profile_builder",  kind: "main" },
  { from: "profile_builder",  to: "rag_retriever",    kind: "parallel", label: "parallel" },
  { from: "signal_extractor", to: "rag_retriever",    kind: "parallel" },
  { from: "profile_builder",  to: "decision",         kind: "main" },
  { from: "rag_retriever",    to: "decision",         kind: "main" },
  { from: "decision",         to: "approval_gate",    kind: "main" },
  { from: "decision",         to: "await_evidence",   kind: "main", label: "needs_more_evidence" },
  { from: "await_evidence",   to: "sanitize",         kind: "loopback", label: "citizen reply" },
];

function drawArrows() {
  const svg = el("arrows");
  const flow = el("flow");
  svg.innerHTML = "";
  const w = flow.scrollWidth;
  const h = flow.scrollHeight;
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  const flowRect = flow.getBoundingClientRect();

  function nodeBox(key) {
    const n = flow.querySelector('.node[data-key="' + key + '"]');
    if (!n) return null;
    const r = n.getBoundingClientRect();
    return {
      left: r.left - flowRect.left + flow.scrollLeft,
      right: r.right - flowRect.left + flow.scrollLeft,
      top: r.top - flowRect.top,
      bottom: r.bottom - flowRect.top,
      cx: r.left - flowRect.left + flow.scrollLeft + r.width / 2,
      cy: r.top - flowRect.top + r.height / 2,
      w: r.width, h: r.height,
    };
  }

  // Build a quick index: {from->to -> state} from the current node states.
  // The edge is "active" if its target is the currently-running node or
  // its source just became done but the target has not yet started.
  const ns = latestNodeState || {};
  function isActive(edge) {
    const toSt = (ns[edge.to] || {}).state;
    const fromSt = (ns[edge.from] || {}).state;
    if (toSt === "active") return true;
    if (edge.kind === "loopback") {
      // loopback fires when decision verdict is needs_more_evidence
      return (ns.decision && ns.decision.caption || "").indexOf("needs_more_evidence") >= 0
          && (ns.await_evidence && ns.await_evidence.state) === "info";
    }
    return false;
  }

  function arrowHead(x1, y1, x2, y2, cls) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const headLen = 10, headW = 6;
    const bx = x2 - ux * headLen, by = y2 - uy * headLen;
    const px = -uy, py = ux;
    const p1x = bx + px * headW, p1y = by + py * headW;
    const p2x = bx - px * headW, p2y = by - py * headW;
    return svgNode("polygon", {
      points: x2 + "," + y2 + " " + p1x + "," + p1y + " " + p2x + "," + p2y,
      class: "arrowhead " + cls,
      fill: "currentColor",
    });
  }

  for (const edge of EDGES) {
    const a = nodeBox(edge.from), b = nodeBox(edge.to);
    if (!a || !b) continue;
    const active = isActive(edge);
    const cls = ["edge-" + (edge.kind || "main"), active ? "active" : ""].join(" ").trim();

    let x1, y1, x2, y2;
    let pathD = null;

    if (edge.kind === "loopback") {
      // Draw a wide arc from await_evidence UP and LEFT back to sanitize.
      x1 = a.cx; y1 = a.top;
      x2 = b.cx; y2 = b.bottom;
      const archY = Math.max(10, Math.min(a.top, b.bottom) - Math.max(80, a.h * 1.4));
      pathD = "M " + x1 + " " + y1
            + " C " + x1 + " " + archY + ", " + x2 + " " + archY + ", " + x2 + " " + y2;
    } else if (edge.kind === "parallel" && edge.from === "profile_builder" && edge.to === "rag_retriever") {
      // Vertical parallel band on the same stage - keep it short + straight.
      x1 = a.right; y1 = a.cy;
      x2 = b.left;  y2 = b.cy;
    } else if (edge.from === "signal_extractor" && edge.to === "rag_retriever") {
      // Bypass arrow that skips profile_builder - gentle curve under it.
      x1 = a.right; y1 = a.bottom - 8;
      x2 = b.left;  y2 = b.bottom - 8;
      const midX = (x1 + x2) / 2;
      const midY = Math.max(y1, y2) + 36;
      pathD = "M " + x1 + " " + y1
            + " Q " + midX + " " + midY + " " + x2 + " " + y2;
    } else if (edge.from === "decision" && edge.to === "await_evidence") {
      x1 = a.cx; y1 = a.bottom;
      x2 = b.cx; y2 = b.top;
    } else {
      x1 = a.right; y1 = a.cy;
      x2 = b.left;  y2 = b.cy;
    }

    let shape;
    if (pathD) {
      shape = svgNode("path", { d: pathD, class: cls });
    } else {
      shape = svgNode("line", { x1: x1, y1: y1, x2: x2, y2: y2, class: cls });
    }
    shape.setAttribute("data-from", edge.from);
    shape.setAttribute("data-to", edge.to);
    svg.appendChild(shape);

    const head = arrowHead(x1, y1, x2, y2, cls);
    // Color the arrowhead to match the edge class via currentColor.
    const color = active ? "var(--active)"
                : edge.kind === "parallel" ? "var(--brand-2)"
                : edge.kind === "loopback" ? "var(--warn)"
                : "var(--border-2)";
    head.setAttribute("style", "color:" + color);
    svg.appendChild(head);

    if (edge.label) {
      const lx = (x1 + x2) / 2;
      const ly = (y1 + y2) / 2 - 6;
      const txt = svgNode("text", {
        x: lx, y: ly, class: "edge-label " + (active ? "active" : ""),
        "text-anchor": "middle",
      });
      txt.textContent = edge.label;
      svg.appendChild(txt);
    }
  }
}

// ---------------------------------------------------------------------------
// Interpret trace into node states
// ---------------------------------------------------------------------------
function interpretTrace(trace, hazard) {
  const nodeState = {};
  for (const n of NODES) nodeState[n.key] = { state: "idle", caption: "idle" };
  if (!Array.isArray(trace) || trace.length === 0) return nodeState;

  const seen = new Set();
  for (const entry of trace) {
    const sp = entry.speaker;
    if (!sp || !nodeState[sp]) continue;
    seen.add(sp);
    const c = (entry.content && typeof entry.content === "object") ? entry.content : {};

    if (sp === "sanitize") {
      nodeState[sp].state = "done";
      const flags = [];
      if (c.ocr_suspicious) flags.push("ocr suspicious");
      if (c.reporter_text_suspicious) flags.push("text suspicious");
      if (c.location_scrubbed) flags.push("location scrubbed");
      nodeState[sp].caption = flags.length ? flags.join(", ") : "clean";
    } else if (sp === "signal_extractor") {
      nodeState[sp].state = "done";
      if (c.has_text) {
        const m = c.mentioned || {};
        nodeState[sp].caption = "class=" + (m.hazard_class || "?") + " lane=" + (m.lane_blocked || "?");
      } else {
        nodeState[sp].caption = "no reporter text";
      }
    } else if (sp === "profile_builder") {
      nodeState[sp].state = "done";
      const conf = c.confidence != null ? Number(c.confidence).toFixed(2) : "?";
      nodeState[sp].caption = (c.primary_class || "?") + " conf=" + conf;
    } else if (sp === "rag_retriever") {
      nodeState[sp].state = "done";
      const n = c.n_procedures != null ? c.n_procedures : ((c.tool_result_ids || []).length);
      nodeState[sp].caption = n + " passages, " + (c.coverage || "?");
    } else if (sp === "decision") {
      if (entry.kind === "react_step") continue;
      const action = c.action || "?";
      nodeState[sp].caption = action + " prio=" + (c.priority != null ? c.priority : "?");
      if (action === "urgent" || action === "human_review") nodeState[sp].state = "bad";
      else if (action === "needs_more_evidence") nodeState[sp].state = "info";
      else if (action === "known_image") nodeState[sp].state = "info";
      else nodeState[sp].state = "done";
    } else if (sp === "approval_gate") {
      nodeState[sp].state = "warn";
      nodeState[sp].caption = "approval " + (c.approval_id || "pending");
    } else if (sp === "await_evidence") {
      nodeState[sp].state = "info";
      nodeState[sp].caption = "round " + (c.evidence_round || 1);
    }
  }

  const status = hazard && hazard.verdict && hazard.verdict.status;
  if (status === "queued" || status === "processing") {
    let activeIdx = 0;
    for (let i = MAIN_ORDER.length - 1; i >= 0; i--) {
      if (seen.has(MAIN_ORDER[i])) { activeIdx = i + 1; break; }
    }
    if (activeIdx < MAIN_ORDER.length) {
      nodeState[MAIN_ORDER[activeIdx]].state = "active";
      nodeState[MAIN_ORDER[activeIdx]].caption = "running...";
    }
  }
  return nodeState;
}

function applyNodeStates(nodeState) {
  for (const n of NODES) {
    const node = document.querySelector('.node[data-key="' + n.key + '"]');
    if (!node) continue;
    node.classList.remove("state-idle","state-active","state-done","state-info","state-warn","state-bad");
    node.classList.add("state-" + nodeState[n.key].state);
    const cap = node.querySelector(".node-caption");
    if (cap) cap.textContent = nodeState[n.key].caption;
    if (n.key === "await_evidence") {
      node.classList.toggle("visible", nodeState[n.key].state !== "idle");
    }
  }
  // Light arrows for traversed edges
  const svg = el("arrows");
  function set(from, to, lit) {
    const nodes = svg.querySelectorAll('[data-from="' + from + '"][data-to="' + to + '"]');
    nodes.forEach(n => {
      if (n.tagName === "line") n.setAttribute("stroke", lit ? "#38bdf8" : "#1f2a44");
      else if (n.tagName === "polygon") n.setAttribute("fill", lit ? "#38bdf8" : "#1f2a44");
    });
  }
  for (let i = 0; i < MAIN_ORDER.length - 1; i++) {
    const fromState = nodeState[MAIN_ORDER[i]].state;
    const toState = nodeState[MAIN_ORDER[i+1]].state;
    const lit = fromState !== "idle" && fromState !== "active" && toState !== "idle";
    set(MAIN_ORDER[i], MAIN_ORDER[i+1], lit);
  }
  const litEvidence = nodeState["await_evidence"].state !== "idle";
  set("decision", "await_evidence", litEvidence);
}

// ---------------------------------------------------------------------------
// Right panel + live trace line
// ---------------------------------------------------------------------------
function updatePanel(hazard) {
  if (!hazard) {
    el("kv-id").textContent = "—";
    el("kv-class").textContent = "—";
    el("kv-conf").textContent = "—";
    el("kv-road").textContent = "—";
    el("kv-lane").textContent = "—";
    el("kv-status").textContent = "—";
    el("verdict-action").textContent = "WAITING";
    el("verdict-rationale").textContent = "No hazard being processed.";
    el("verdict-box").className = "verdict-box";
    el("live-line").textContent = "waiting for events...";
    el("hz-display").textContent = "";
    return;
  }
  const md = hazard.metadata || {};
  const v = hazard.verdict || {};
  el("kv-id").textContent = hazard.hazard_id || "—";
  el("kv-class").textContent = md.class || "—";
  el("kv-conf").textContent = md.confidence != null ? Number(md.confidence).toFixed(2) : "—";
  el("kv-road").textContent = md.road_type || "—";
  el("kv-lane").textContent = md.lane_blocked || "—";
  el("kv-status").textContent = v.status || "—";
  const action = v.action || "WAITING";
  el("verdict-action").textContent = String(action).toUpperCase();
  el("verdict-rationale").textContent = v.rationale || v.assessment || "Still processing.";
  el("verdict-box").className = "verdict-box " + String(v.action || "").replace(/[^a-z_]/g, "");
  const lines = hazard.trace_lines || [];
  el("live-line").textContent = lines.length ? lines[lines.length - 1] : "(no trace lines yet)";
  el("hz-display").textContent = hazard.hazard_id ? "@ " + hazard.hazard_id : "";
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
let timer = null, paused = false, latestNodeState = null;

// Only a run that is in progress right now is drawn: status "processing"
// and a trace entry (or the submission itself) within LIVE_WINDOW_S
// seconds. A finished run, a queued one still waiting for a slot and a
// row left in "processing" by a crash all fall back to the idle view.
const LIVE_WINDOW_S = 300;

function lastTraceTs(trace) {
  let last = 0;
  for (const e of (Array.isArray(trace) ? trace : [])) {
    const t = Number(e && e.ts);
    if (t > last) last = t;
  }
  return last;
}

async function liveHazard() {
  const history = await apiGet("/hazards/history?limit=200");
  const running = (history.items || []).filter(h => h.status === "processing");
  const now = Date.now() / 1000;
  for (const item of running) {
    const hazard = await apiGet("/report/" + encodeURIComponent(item.hazard_id));
    const submitted = Date.parse(item.reported_at) / 1000;
    const recent = Math.max(lastTraceTs(hazard.trace), isNaN(submitted) ? 0 : submitted);
    if (now - recent <= LIVE_WINDOW_S) return hazard;
  }
  return null;
}

function showIdle() {
  updatePanel(null);
  el("empty-state").style.display = "block";
  el("flow").style.display = "none";
  latestNodeState = interpretTrace([], null);
  applyNodeStates(latestNodeState);
  requestAnimationFrame(drawArrows);
}

async function tick() {
  if (paused) return;
  try {
    const hazard = await liveHazard();
    if (!hazard) { showIdle(); return; }
    el("empty-state").style.display = "none";
    el("flow").style.display = "grid";
    latestNodeState = interpretTrace(hazard.trace, hazard);
    applyNodeStates(latestNodeState);
    updatePanel(hazard);
    requestAnimationFrame(drawArrows);
  } catch (err) {
    // Transient tunnel / 401 / 403 are expected during restarts.
    // There is no passphrase modal to open, so just swallow and retry
    // on the next tick.
  }
}

function start() {
  if (timer) clearInterval(timer);
  tick();
  timer = setInterval(tick, POLL_MS);
}
function stop() {
  if (timer) { clearInterval(timer); timer = null; }
}

// Round 23 (4.11.2026): KPI / cost / advantage panels. These three poll
// their own endpoints every 30 seconds; the diagram's 2-second tick
// focuses on the running hazard and does not need to be slowed down.
let kpiTimer = null;
async function refreshKpi() {
  try {
    const d = await apiGet("/stats/kpi?window_hours=24");
    const rows = el("kpi-rows");
    if (!rows) return;
    rows.innerHTML = "";
    for (const k of (d.kpis || [])) {
      const row = create("div", "kpi-row");
      row.appendChild(create("div", "kpi-label", k.label));
      const vcls = "kpi-value " + (k.status === "ok" ? "ok" : "fail");
      const vtext = (k.key === "median_latency_s")
        ? (k.value_pct + " s / cap 360 s")
        : (k.value_pct + "% / target " + k.target_pct + "%");
      row.appendChild(create("div", vcls, vtext));
      rows.appendChild(row);
      if (k.note) {
        const note = create("div", "kpi-note", k.note);
        note.style.gridColumn = "1 / -1";
        rows.appendChild(note);
      }
    }
    if (!(d.kpis || []).length) rows.textContent = "no cases in the last 24h";
  } catch (e) { /* silent, retry next tick */ }
}
async function refreshCost() {
  try {
    const d = await apiGet("/stats/agents?window_hours=24");
    const rows = el("cost-rows");
    if (!rows) return;
    const c = d.cost_total || {};
    const aj = d.agents_justified || {};
    rows.innerHTML = "";
    function row(k, v) {
      rows.appendChild(create("span", "k", k));
      rows.appendChild(create("span", "v", v));
    }
    row("total hazards", String(d.total_hazards || 0));
    row("LLM calls", String(c.llm_calls || 0));
    row("tokens in", String(c.tokens_in || 0));
    row("tokens out", String(c.tokens_out || 0));
    row("wall seconds", String(c.wall_seconds || 0));
    row("Ollama self-hosted", "$0.00");
    row("commercial equivalent", "$" + Number(c.cost_commercial_equivalent_usd || 0).toFixed(4));
    rows.appendChild(create("div", "kpi-note", "monthly city cost: Oracle A1.Flex free + Ollama + Tailscale = $0"));
    const header = create("div", "kpi-note", "agent justified / runs:");
    header.style.marginTop = "6px";
    rows.appendChild(header);
    for (const sp of ["sanitize","signal_extractor","supervisor","profile_builder","rag_retriever","decision"]) {
      const b = aj[sp] || {};
      row(sp, (b.justified || 0) + " / " + (b.runs || 0));
    }
  } catch (e) { /* silent */ }
}
async function refreshAdvantage() {
  try {
    const d = await apiGet("/stats/advantage?window_hours=24");
    const rows = el("advantage-rows");
    if (!rows) return;
    rows.innerHTML = "";
    const over = d.advantage_over_tree || [];
    const approved = d.agent_approved_by_human || [];
    const denied = d.agent_corrected_by_human || [];
    function section(title, list) {
      const h = create("div", "kpi-note", title + " (" + list.length + ")");
      h.style.marginTop = "6px"; h.style.fontWeight = "600";
      rows.appendChild(h);
      if (!list.length) { rows.appendChild(create("div", "adv-case", "none")); return; }
      for (const c of list.slice(0, 5)) {
        const line = create("div", "adv-case",
          (c.id || "").substring(0, 12) + "  " + (c.class || "") + "  " + (c.decision || c.final_action || ""));
        rows.appendChild(line);
      }
      if (list.length > 5) rows.appendChild(create("div", "adv-case", "... and " + (list.length - 5) + " more"));
    }
    section("Agent > rule tree", over);
    section("Agent approved by human", approved);
    section("Agent corrected by human", denied);
  } catch (e) { /* silent */ }
}
function startKpiPolls() {
  if (kpiTimer) clearInterval(kpiTimer);
  refreshKpi(); refreshCost(); refreshAdvantage();
  kpiTimer = setInterval(() => { refreshKpi(); refreshCost(); refreshAdvantage(); }, 30000);
}
startKpiPolls();

el("pause-btn").addEventListener("click", () => {
  paused = !paused;
  el("pause-btn").textContent = paused ? "Resume" : "Pause";
  el("status-badge").textContent = paused ? "PAUSED" : "LIVE";
  el("status-badge").classList.toggle("paused", paused);
  if (!paused) tick();
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
buildFlow();
window.addEventListener("resize", drawArrows);
setTimeout(drawArrows, 50);

if (token) start(); else el("login").classList.add("on");
</script>
</body>
</html>
"""
_STATIC_PAGES = {"trace.html": _TRACE_HTML, "supervisor.html": _SUPERVISOR_HTML, "flow.html": _FLOW_HTML}

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


class PasswordLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=1, max_length=200)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=1, max_length=200)
    email: str | None = Field(default=None, max_length=200)


class AuthConfigResponse(BaseModel):
    """Which auth methods the server will accept right now. Unauthenticated
    endpoint so the frontend can decide which buttons to render before
    anyone is logged in."""
    signup_enabled: bool
    password_login_enabled: bool
    google_login_enabled: bool
    max_users: int
    user_count: int


class Me(BaseModel):
    sub: str
    email: str
    name: str
    username: str | None = None
    is_admin: bool = False
    is_internal: bool = False
    uploads_today: int = 0
    uploads_total: int = 0
    daily_cap: int = 100
    operator_urls: list[str] = Field(default_factory=list)


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
    # Base64-encoded JPEG bytes (webapp resizes client-side to max 1600x1600,
    # q=80, before encoding). Enforced max ~6.9 MB base64 = 5 MB raw image.
    # The server saves to /var/lib/roadguard/images/<hazard_id>.jpg and
    # exposes it via GET /images/<hazard_id>. 7-day retention per cron.
    image_jpeg_base64: str | None = Field(default=None, max_length=7_000_000)


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


def _trace_with_event(trace_json: str, kind: str, reason: str) -> str:
    """Append a pipeline event to a stored trace so the /trace page shows
    why a verdict changed outside a graph run."""
    try:
        trace = json.loads(trace_json or "[]")
    except Exception:
        trace = []
    if not isinstance(trace, list):
        trace = []
    trace.append({
        "iter": len(trace),
        "speaker": "pipeline",
        "kind": kind,
        "content": {"reason": reason},
        "ts": time.time(),
    })
    return json.dumps(trace)


def _route_to_human(db: Database, hazard: Any, kind: str, reason: str) -> None:
    """Charter route for a run that cannot finish on its own: the row
    becomes a human_review verdict (priority floor 50, like a crashed run)
    with an approvals row, so a person is notified."""
    priority = max(int(hazard.priority or 0), 50)
    db.update_hazard_verdict(
        hazard_id=hazard.id,
        priority=priority,
        agent_decision="human_review",
        agent_rationale=reason,
        agent_trace_json=_trace_with_event(hazard.agent_trace_json, kind, reason),
        cited_sources_json="[]",
        status="open",
        work_order_id=None,
        agent_assessment="",
    )
    db.create_approval(
        hazard_id=hazard.id,
        action="human_review",
        priority=priority,
        rationale=reason,
        cited_sources_json="[]",
    )


def recover_interrupted_runs(db_path: str) -> int:
    """Startup only. A row still 'queued' or 'processing' when the service
    starts belonged to a background task that died with the old process;
    nothing will ever finish it. Route it to a human instead of leaving it
    to look live forever. Returns the number of rows recovered."""
    reason = "service restarted while the run was in progress; routed to a human"
    n = 0
    with Database(db_path) as db:
        for h in db.list_hazards_by_status(("queued", "processing")):
            _route_to_human(db, h, "graph_interrupted", reason)
            n += 1
    if n:
        LOGGER.warning("recovered_interrupted_runs: %d row(s) routed to human_review", n)
    return n


def route_expired_evidence(db: Database, ttl_hours: float) -> int:
    """Hazards whose evidence request expired (expire_evidence_requests set
    their status to 'evidence_expired') become human_review with an
    approvals row, so a case the reporter abandoned reaches a person.
    Idempotent: the routed row leaves 'evidence_expired'."""
    reason = (
        f"reporter did not answer the evidence request within {ttl_hours:g}h; routed to a human"
    )
    n = 0
    for h in db.list_hazards_by_status(("evidence_expired",)):
        _route_to_human(db, h, "evidence_expired", reason)
        n += 1
    return n


def record_decision_lesson(db: Database, approval_id: str, decision: str, note: str) -> str | None:
    """Write the agent_notes row for a decided approval: which agent verdict
    a human confirmed or overturned, for which class / road / lane, plus
    the supervisor's note. Returns the note id, or None when the approval
    is unknown."""
    approval = db.get_approval(approval_id)
    if approval is None:
        return None
    hazard = db.get_hazard_any(approval.hazard_id)
    return db.add_agent_note(
        hazard_id=approval.hazard_id,
        approval_id=approval_id,
        decision=decision,
        agent_action=approval.action,
        hazard_class=getattr(hazard, "hazard_class", "") or "",
        road_type=getattr(hazard, "road_type", "") or "",
        lane_blocked=getattr(hazard, "lane_blocked", "") or "",
        note=note or "",
    )


def approvals_overdue(pending: list[Any], sla_hours: float, now: float | None = None) -> int:
    """How many pending approvals have waited longer than the SLA."""
    at = time.time() if now is None else now
    return sum(1 for a in pending if at - float(a.created_at) > sla_hours * 3600)


def retention_sweep(db_path: str, checkpoint_path: str) -> dict[str, int]:
    """Expire evidence requests older than queues.evidence_ttl_hours and
    route their hazards to a human, then prune checkpoint threads older
    than CHECKPOINT_RETENTION_DAYS. Returns the counts for the log."""
    from roadguard_agent import _policy_queues

    ttl_hours = _policy_queues()["evidence_ttl_hours"]
    cutoff = time.time() - CHECKPOINT_RETENTION_DAYS * 86400
    with Database(db_path) as db:
        keep = [
            r["id"]
            for r in db._conn.execute(
                "SELECT id FROM hazards WHERE reported_at >= ?", (cutoff,)
            )
        ]
        expired = db.expire_evidence_requests(time.time() - ttl_hours * 3600)
        routed = route_expired_evidence(db, ttl_hours)
    removed = prune_checkpoints(checkpoint_path, keep)
    LOGGER.info(
        "retention_sweep: expired_requests=%d routed_to_human=%d checkpoints_removed=%d kept_threads=%d",
        expired, routed, removed, len(keep),
    )
    return {"expired": int(expired), "routed": routed, "removed": removed, "kept": len(keep)}


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


def _aggregate_agent_metrics(rows) -> dict[str, Any]:
    """Per-agent quality counters from a batch of hazard rows.

    Each row must carry an ``agent_trace_json`` column; non-JSON content
    is skipped silently so one broken row cannot kill the aggregate. The
    counters are raw run counts (not percentages); the operator divides
    by ``total_hazards`` to get a rate - except for the RAG counters,
    which divide by ``rag_retriever.runs`` because not every hazard
    reaches the RAG stage. ``llm_fallback_runs`` is per hazard (any
    speaker whose LLM calls all failed in that run)."""
    counters: dict[str, dict[str, int]] = {
        "sanitize": {"location_scrubbed": 0, "ocr_suspicious": 0,
                     "reporter_text_suspicious": 0},
        "signal_extractor": {"has_text": 0, "llm_used": 0},
        "supervisor": {"concern_none": 0, "concern_input_ambiguous": 0,
                       "concern_budget_tight": 0, "concern_other": 0},
        "profile_builder": {"llm_used": 0, "catalog_hit": 0,
                            "class_matches_detector": 0},
        "rag_retriever": {"runs": 0, "coverage_direct_match": 0,
                          "llm_used": 0, "total_passages_returned": 0},
        "decision": {"rule_based": 0, "llm_based": 0,
                     "grounding_failed": 0, "contradiction_floor": 0,
                     "action_standard": 0, "action_known_image": 0,
                     "action_human_review": 0, "action_needs_more_evidence": 0,
                     "action_urgent": 0},
    }
    llm_fallback_runs = 0
    fallback_by_cause: dict[str, int] = {}
    fallback_by_agent: dict[str, dict[str, int]] = {}
    fast_path_skips: dict[str, int] = {}
    autonomy: dict[str, dict[str, int]] = {}
    # Round 22 (4.11.2026): per-run agent value counters. Each hazard
    # contributes at most one increment to each bucket here, built from
    # the same Decision final entry that _build_agent_value_summary reads
    # (worker_impact + rationale_check + floor). Rates are computed
    # client-side against total_hazards, exactly like the other blocks.
    agent_value = {
        "runs": 0,                           # hazards with a Decision final
        "has_worker_impact": 0,              # at least one worker baseline was captured
        "changed_rule_hint": 0,              # worker LLMs moved the tree's verdict
        "profile_llm_changed": 0,            # Profile Builder's LLM moved the tree
        "research_llm_changed": 0,           # RAG's LLM moved the tree
        "decision_llm_react": 0,             # Decision used ReAct (count by source tag)
        "decision_rule_based": 0,            # Decision ran rule_based only
        "decision_catalog_match": 0,         # sha256 catalogue match short-circuit
        "floor_applied": 0,                  # a deterministic floor fired
        "enforcement_rule_tree_verdict": 0,  # Decision LLM overridden (too severe)
        "enforcement_kept": 0,               # LLM verdict kept over the fact-check
        "rule_tree_eq_final": 0,             # the rule tree already matched the final
        "llm_more_severe_than_tree": 0,      # LLM proposed > tree (regardless of enforcement)
        "llm_less_severe_than_tree": 0,      # LLM proposed < tree (floors tend to win)
    }
    _SEV_RANK = {"standard": 1, "known_image": 1, "needs_more_evidence": 2,
                 "human_review": 3, "urgent": 4}
    for row in rows or []:
        try:
            trace = json.loads((row["agent_trace_json"] if hasattr(row, "__getitem__") else getattr(row, "agent_trace_json", None)) or "[]")
        except (TypeError, ValueError):
            continue
        detector_class = str(
            (row["class"] if hasattr(row, "__getitem__") else getattr(row, "class", None)) or ""
        ).upper()
        had_fallback = False
        for entry in trace:
            if not isinstance(entry, dict):
                continue
            speaker = str(entry.get("speaker") or "")
            c = entry.get("content")
            if not isinstance(c, dict):
                continue
            if speaker == "sanitize":
                if c.get("location_scrubbed"):
                    counters["sanitize"]["location_scrubbed"] += 1
                if c.get("ocr_suspicious"):
                    counters["sanitize"]["ocr_suspicious"] += 1
                if c.get("reporter_text_suspicious"):
                    counters["sanitize"]["reporter_text_suspicious"] += 1
            elif speaker == "signal_extractor":
                if c.get("has_text"):
                    counters["signal_extractor"]["has_text"] += 1
                if c.get("llm_used"):
                    counters["signal_extractor"]["llm_used"] += 1
            elif speaker == "supervisor" and c.get("pipeline_concern"):
                concern = c["pipeline_concern"]
                key = (f"concern_{concern}"
                       if concern in ("none", "input_ambiguous", "budget_tight")
                       else "concern_other")
                counters["supervisor"][key] = counters["supervisor"].get(key, 0) + 1
            elif speaker == "profile_builder":
                if c.get("llm_used"):
                    counters["profile_builder"]["llm_used"] += 1
                dl = c.get("dataset_lookup") or {}
                if isinstance(dl, dict) and dl.get("found"):
                    counters["profile_builder"]["catalog_hit"] += 1
                pc = str(c.get("primary_class") or "").upper()
                if detector_class and pc and pc == detector_class:
                    counters["profile_builder"]["class_matches_detector"] += 1
            elif speaker == "rag_retriever":
                counters["rag_retriever"]["runs"] += 1
                if c.get("coverage") == "direct_match":
                    counters["rag_retriever"]["coverage_direct_match"] += 1
                if c.get("llm_used"):
                    counters["rag_retriever"]["llm_used"] += 1
                try:
                    counters["rag_retriever"]["total_passages_returned"] += int(c.get("n_procedures") or 0)
                except (TypeError, ValueError):
                    pass
            elif speaker == "decision" and entry.get("kind") == "final":
                if c.get("source") == "rule_based":
                    counters["decision"]["rule_based"] += 1
                else:
                    counters["decision"]["llm_based"] += 1
                action = str(c.get("action") or "").lower()
                key = f"action_{action}"
                counters["decision"][key] = counters["decision"].get(key, 0) + 1
                rationale = str(c.get("rationale") or "").lower()
                if "grounding failed" in rationale:
                    counters["decision"]["grounding_failed"] += 1
                if c.get("floor_applied"):
                    counters["decision"]["contradiction_floor"] += 1
                # Round 22 (4.11.2026): agent value aggregate - per-case
                # flags pulled from the same Decision final entry.
                agent_value["runs"] += 1
                source = str(c.get("source") or "")
                if source == "llm_react":
                    agent_value["decision_llm_react"] += 1
                elif source == "rule_based":
                    agent_value["decision_rule_based"] += 1
                elif source == "catalog_match":
                    agent_value["decision_catalog_match"] += 1
                wi = c.get("worker_impact")
                final_action = action
                rt_action = final_action
                if isinstance(wi, dict):
                    agent_value["has_worker_impact"] += 1
                    if wi.get("changed_rule_hint"):
                        agent_value["changed_rule_hint"] += 1
                    if wi.get("profile_llm_changed"):
                        agent_value["profile_llm_changed"] += 1
                    if wi.get("research_llm_changed"):
                        agent_value["research_llm_changed"] += 1
                    rt_action = str(wi.get("rule_hint_without_worker_llm") or final_action)
                if rt_action == final_action:
                    agent_value["rule_tree_eq_final"] += 1
                ov = c.get("overridden_llm_verdict")
                llm_action = None
                if isinstance(ov, dict) and ov.get("action"):
                    llm_action = str(ov.get("action") or "")
                elif source == "llm_react":
                    llm_action = final_action
                if llm_action:
                    rt_rank = _SEV_RANK.get(rt_action, 1)
                    llm_rank = _SEV_RANK.get(llm_action, 1)
                    if llm_rank > rt_rank:
                        agent_value["llm_more_severe_than_tree"] += 1
                    elif llm_rank < rt_rank:
                        agent_value["llm_less_severe_than_tree"] += 1
                if c.get("floor_applied"):
                    agent_value["floor_applied"] += 1
                rc = c.get("rationale_check")
                if isinstance(rc, dict):
                    enf = str(rc.get("enforcement") or "")
                    if enf == "rule_tree_verdict":
                        agent_value["enforcement_rule_tree_verdict"] += 1
                    elif enf == "kept":
                        agent_value["enforcement_kept"] += 1
            llm_s = c.get("llm") or {}
            if isinstance(llm_s, dict) and llm_s.get("failed_calls", 0) > 0:
                had_fallback = True
                cause = _classify_llm_failure(llm_s)
                fallback_by_cause[cause] = fallback_by_cause.get(cause, 0) + 1
                by_speaker = fallback_by_agent.setdefault(speaker or "?", {})
                by_speaker[cause] = by_speaker.get(cause, 0) + 1
            reason = str(c.get("llm_fallback_reason") or "")
            if reason.endswith("off_contract"):
                key = "react_reply_off_contract" if reason.startswith("react") else "reply_off_contract"
                fallback_by_cause[key] = fallback_by_cause.get(key, 0) + 1
                by_speaker = fallback_by_agent.setdefault(speaker or "?", {})
                by_speaker[key] = by_speaker.get(key, 0) + 1
            if c.get("llm_skipped_reason", "").startswith("clear_case_fast_path"):
                fast_path_skips[speaker or "?"] = fast_path_skips.get(speaker or "?", 0) + 1
            auto = c.get("autonomy")
            if isinstance(auto, dict):
                # Autonomous workers: runs, tool calls made, contract
                # violations found by verification, and clean runs.
                # Round 11 (1.11.2026): agent_called = tools the agent chose
                # and that ran, auto_ran_tools = tools the contract ran for
                # it, tool_failures = calls it made that failed; the three
                # are counted apart so autonomy is not credited for what
                # the code did.
                a = autonomy.setdefault(speaker or "?", {
                    "runs": 0, "tool_calls": 0, "agent_called": 0, "auto_ran": 0, "auto_ran_tools": 0,
                    "tool_failures": 0, "violations": 0, "clean_runs": 0,
                })
                a["runs"] += 1
                n_called = len(auto.get("tools_used") or [])
                a["tool_calls"] += n_called
                a["agent_called"] += n_called
                n_viol = len(auto.get("violations") or [])
                a["violations"] += n_viol
                if n_viol == 0:
                    a["clean_runs"] += 1
                if auto.get("auto_ran"):
                    a["auto_ran"] += 1
                    a["auto_ran_tools"] += len(auto.get("auto_ran") or [])
                a["tool_failures"] += len(auto.get("failed_tools") or [])
        if had_fallback:
            llm_fallback_runs += 1
    # Round 23 (4.11.2026): per-agent justified aggregate + total cost +
    # advantage over rule tree. Advantage over the human lives on /stats/
    # advantage and reads the approvals table (human decisions on hazards
    # the agent flagged for review); it is not in this trace walk.
    justified_counts = {sp: 0 for sp in (
        "sanitize", "signal_extractor", "supervisor",
        "profile_builder", "rag_retriever", "decision")}
    justified_runs = {sp: 0 for sp in justified_counts}
    cost_total = {"llm_calls": 0, "tokens_in": 0, "tokens_out": 0,
                  "wall_seconds": 0.0, "cost_self_hosted_usd": 0.0,
                  "cost_commercial_equivalent_usd": 0.0}
    advantage_over_tree_ids: list[str] = []
    for row in rows or []:
        # Round 23 (4.11.2026): a test-fixture row (or a hazards row from a
        # DB that has not run the agent_value_json migration yet) may lack
        # this column entirely; fall back to building the summary from the
        # trace, so the aggregate stays correct and never raises KeyError.
        raw = ""
        try:
            keys = row.keys() if hasattr(row, "keys") else []
            if "agent_value_json" in keys:
                raw = row["agent_value_json"] or ""
        except Exception:
            raw = ""
        if not raw:
            try:
                tj = row["agent_trace_json"] if "agent_trace_json" in (row.keys() if hasattr(row, "keys") else []) else "[]"
            except Exception:
                tj = "[]"
            try:
                from roadguard_agent import _build_agent_value_summary_json
                raw = _build_agent_value_summary_json(tj or "[]") or ""
            except Exception:
                raw = ""
        try:
            av = json.loads(raw or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(av, dict):
            continue
        pa = av.get("per_agent_justified") or {}
        for sp, data in pa.items():
            if not isinstance(data, dict):
                continue
            if data.get("ran"):
                justified_runs[sp] = justified_runs.get(sp, 0) + 1
            if data.get("justified"):
                justified_counts[sp] = justified_counts.get(sp, 0) + 1
        c = av.get("cost") or {}
        if isinstance(c, dict):
            cost_total["llm_calls"] += int(c.get("llm_calls") or 0)
            cost_total["tokens_in"] += int(c.get("tokens_in") or 0)
            cost_total["tokens_out"] += int(c.get("tokens_out") or 0)
            try:
                cost_total["wall_seconds"] += float(c.get("wall_seconds") or 0)
            except (TypeError, ValueError):
                pass
            try:
                cost_total["cost_commercial_equivalent_usd"] += float(
                    c.get("cost_commercial_equivalent_usd") or 0)
            except (TypeError, ValueError):
                pass
        if av.get("changed_rule_hint"):
            try:
                advantage_over_tree_ids.append(str(row["id"]))
            except (KeyError, IndexError):
                pass
    cost_total["wall_seconds"] = round(cost_total["wall_seconds"], 2)
    cost_total["cost_commercial_equivalent_usd"] = round(
        cost_total["cost_commercial_equivalent_usd"], 6)

    return {
        "per_agent": counters,
        "llm_fallback_runs": llm_fallback_runs,
        "llm_fallback_by_cause": fallback_by_cause,
        "llm_fallback_by_agent": fallback_by_agent,
        "fast_path_skips": fast_path_skips,
        "autonomy": autonomy,
        "agent_value": agent_value,
        "agents_justified": {
            sp: {"justified": justified_counts[sp], "runs": justified_runs[sp]}
            for sp in justified_counts
        },
        "cost_total": cost_total,
        "advantage_over_tree_ids": advantage_over_tree_ids,
    }


# ---------------------------------------------------------------- round 26
# Round 26 (5.11.2026): 20 data-science metrics in the Agent value tab. The
# owner wanted real metrics beyond the three existing cards. Ground truth is
# the municipal worker's in-system feedback (agent_feedback), not an external
# labeled catalog. The function returns None-valued panels with a 'pending'
# note when there is not enough feedback to compute a metric - a 'metric not
# yet populated' message, not a zero that reads as a failing score.


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(p * len(sorted_vals))))
    return float(sorted_vals[idx])


def _roc_auc_binary(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U formulation of binary ROC-AUC. Returns None if either
    class has no samples (AUC undefined)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


_FEEDBACK_AGREE = {"agree"}
_FEEDBACK_NEGATIVE = {"disagree", "not_relevant"}
_PRED_CLASSES = ("standard", "human_review", "known_image", "needs_more_evidence")
_FEEDBACK_VERDICTS = ("agree", "disagree", "not_relevant")


def _compute_ds_metrics_block(
    hazard_rows: list, feedback_rows: list
) -> dict[str, Any]:
    """Round 26 (5.11.2026): the 20-metric data-science block the owner
    asked for. ``hazard_rows`` carries agent_decision + priority +
    agent_value_json + agent_trace_json for each hazard in the window;
    ``feedback_rows`` is the LATEST agent_feedback row per (hazard_id,
    kind).

    Every panel that needs feedback reports ``pending`` when it has too
    few samples, so the UI shows 'metric not yet populated' instead of
    a zero. Metrics derivable from the trace alone (uplift, cost,
    justified histogram, grounding) populate with any n>=1 hazard.
    """
    import math as _math
    n = len(hazard_rows or [])

    # Group the feedback rows by (hazard_id, kind).
    fb_hazard: dict[str, str] = {}
    fb_priority: dict[str, str] = {}
    for fr in feedback_rows or []:
        try:
            keys = fr.keys() if hasattr(fr, "keys") else []
            kind = (fr["kind"] if "kind" in keys else "hazard") or "hazard"
            verdict = (fr["verdict"] if "verdict" in keys else "") or ""
            if not verdict and "agree" in keys:
                verdict = "agree" if int(fr["agree"] or 0) else "disagree"
            hid = fr["hazard_id"] if "hazard_id" in keys else None
        except Exception:
            continue
        if not hid or not verdict:
            continue
        if kind == "priority":
            fb_priority[hid] = verdict
        else:
            fb_hazard[hid] = verdict

    # Per-hazard structured facts.
    tokens_in: list[int] = []
    tokens_out: list[int] = []
    wall_s: list[float] = []
    llm_calls_list: list[int] = []
    cost_sum = 0.0
    per_run_justified_count: list[int] = []
    rule_tree_eq_final = 0
    floor_fire = 0
    llm_flip = 0
    llm_clean_wins = 0
    pred_action: dict[str, str] = {}
    pred_priority: dict[str, int] = {}
    grounding_ok = 0
    grounding_total = 0

    for row in hazard_rows or []:
        try:
            keys = row.keys() if hasattr(row, "keys") else []
            hid = row["id"] if "id" in keys else None
            dec = str(row["agent_decision"] if "agent_decision" in keys else "" or "").lower()
            pri_val = 0
            try:
                pri_val = int(row["priority"] if "priority" in keys else 0 or 0)
            except (TypeError, ValueError):
                pri_val = 0
            av_raw = (row["agent_value_json"] if "agent_value_json" in keys else "") or ""
            tj_raw = (row["agent_trace_json"] if "agent_trace_json" in keys else "") or "[]"
        except Exception:
            continue
        if hid:
            pred_action[hid] = dec
            pred_priority[hid] = pri_val

        try:
            av = json.loads(av_raw) if av_raw else {}
        except (TypeError, ValueError):
            av = {}
        if not isinstance(av, dict):
            av = {}

        # Uplift: rule_tree == final?
        rt_final = av.get("rule_tree") == av.get("final") and av.get("final")
        if rt_final:
            rule_tree_eq_final += 1
        flags = str(av.get("flags") or "")
        had_floor = "floor:" in flags or av.get("floor_applied")
        if had_floor:
            floor_fire += 1
        llm_val = av.get("llm")
        llm_moved = (
            isinstance(llm_val, str) and llm_val not in ("", "-")
            and av.get("rule_tree") and llm_val != av.get("rule_tree")
        )
        if llm_moved:
            llm_flip += 1
            if not had_floor and llm_val == av.get("final"):
                llm_clean_wins += 1

        # Cost.
        c = av.get("cost") or {}
        if isinstance(c, dict):
            try:
                tokens_in.append(int(c.get("tokens_in") or 0))
            except (TypeError, ValueError):
                tokens_in.append(0)
            try:
                tokens_out.append(int(c.get("tokens_out") or 0))
            except (TypeError, ValueError):
                tokens_out.append(0)
            try:
                wall_s.append(float(c.get("wall_seconds") or 0))
            except (TypeError, ValueError):
                wall_s.append(0.0)
            try:
                llm_calls_list.append(int(c.get("llm_calls") or 0))
            except (TypeError, ValueError):
                llm_calls_list.append(0)
            try:
                cost_sum += float(c.get("cost_commercial_equivalent_usd") or 0)
            except (TypeError, ValueError):
                pass

        # Justified count per run.
        pa = av.get("per_agent_justified") or {}
        j_count = 0
        if isinstance(pa, dict):
            for sp_data in pa.values():
                if isinstance(sp_data, dict) and sp_data.get("justified"):
                    j_count += 1
        per_run_justified_count.append(j_count)

        # Grounding score: for each hazard whose decision had cited
        # sources, check that at least one citation appears in the RAG
        # retriever's n_procedures passages. sec-X.Y tokens are the
        # canonical citation form.
        try:
            trace = json.loads(tj_raw) if tj_raw else []
        except (TypeError, ValueError):
            trace = []
        cited: list[str] = []
        rag_secs: list[str] = []
        for entry in trace or []:
            if not isinstance(entry, dict):
                continue
            speaker = str(entry.get("speaker") or "")
            c_entry = entry.get("content")
            if not isinstance(c_entry, dict):
                continue
            if speaker == "rag_retriever":
                procs = c_entry.get("procedures") or c_entry.get("passages") or []
                for p in procs if isinstance(procs, list) else []:
                    if isinstance(p, dict):
                        sid = str(p.get("sec") or p.get("section") or p.get("id") or "")
                        if sid:
                            rag_secs.append(sid.lower())
                    elif isinstance(p, str):
                        rag_secs.append(p.lower())
            elif speaker == "decision" and entry.get("kind") == "final":
                cs = c_entry.get("cited_sources") or []
                for s in cs if isinstance(cs, list) else []:
                    sv = str(s).lower()
                    if sv.startswith("sec-") or sv.startswith("sec_"):
                        cited.append(sv)
        if cited:
            grounding_total += 1
            if rag_secs and any(cs in rag_secs or any(cs in rs for rs in rag_secs) for cs in cited):
                grounding_ok += 1

    # 1-5 feedback agreement.
    def _rate(num: int, den: int) -> float | None:
        return round(100.0 * num / den, 1) if den else None

    hz_votes = list(fb_hazard.values())
    agree_n = sum(1 for v in hz_votes if v in _FEEDBACK_AGREE)
    dis_n = sum(1 for v in hz_votes if v == "disagree")
    nr_n = sum(1 for v in hz_votes if v == "not_relevant")
    hazard_valid_rate = _rate(agree_n, len(hz_votes))

    pri_votes = list(fb_priority.values())
    pri_ok_n = sum(1 for v in pri_votes if v == "ok")
    priority_correct_rate = _rate(pri_ok_n, len(pri_votes))

    # Safety-floor precision: among hazards where a floor fired AND the
    # worker voted, what % were marked agree. (Did the worker approve
    # the floor's downgrade?)
    floor_hids = {hid for hid in pred_action if hid in fb_hazard}
    floor_fired_hids = set()
    llm_flipped_hids = set()
    for row in hazard_rows or []:
        try:
            hid = row["id"]
            av_raw = row["agent_value_json"] or ""
        except Exception:
            continue
        try:
            av = json.loads(av_raw) if av_raw else {}
        except (TypeError, ValueError):
            av = {}
        if not isinstance(av, dict):
            continue
        flags = str(av.get("flags") or "")
        if "floor:" in flags or av.get("floor_applied"):
            floor_fired_hids.add(hid)
        llm_val = av.get("llm")
        if (isinstance(llm_val, str) and llm_val not in ("", "-")
                and av.get("rule_tree") and llm_val != av.get("rule_tree")):
            llm_flipped_hids.add(hid)
    fp_intersect = [hid for hid in floor_fired_hids if hid in fb_hazard]
    sf_precision = _rate(
        sum(1 for hid in fp_intersect if fb_hazard[hid] in _FEEDBACK_AGREE),
        len(fp_intersect),
    )
    lu_intersect = [hid for hid in llm_flipped_hids if hid in fb_hazard]
    lu_precision = _rate(
        sum(1 for hid in lu_intersect if fb_hazard[hid] in _FEEDBACK_AGREE),
        len(lu_intersect),
    )

    # By-decision-class breakdown: the confusion-matrix rows.
    by_class: dict[str, dict[str, int]] = {}
    for c in _PRED_CLASSES:
        by_class[c] = {v: 0 for v in _FEEDBACK_VERDICTS}
        by_class[c]["total"] = 0
    for hid, verdict in fb_hazard.items():
        c = pred_action.get(hid) or ""
        if c not in by_class:
            continue
        if verdict in _FEEDBACK_VERDICTS:
            by_class[c][verdict] += 1
            by_class[c]["total"] += 1

    # Per-class precision and F1 (binary: agree=1, disagree/not_relevant=0
    # within each predicted class). Recall is NOT derivable from this
    # feedback schema (we have no explicit 'should have been X' label
    # when the worker disagreed), so F1 here is 2*P*A / (P+A) where A is
    # the window-wide agreement rate used as a stand-in. The UI calls
    # this out explicitly as a precision score, not a classical F1.
    per_class_f1: dict[str, dict[str, Any]] = {}
    agree_overall = _rate(agree_n, len(hz_votes))
    for c in _PRED_CLASSES:
        tot = by_class[c]["total"]
        if tot:
            p = round(100.0 * by_class[c]["agree"] / tot, 1)
        else:
            p = None
        per_class_f1[c] = {
            "precision_pct": p,
            "n": tot,
            "agree": by_class[c]["agree"],
            "disagree": by_class[c]["disagree"],
            "not_relevant": by_class[c]["not_relevant"],
        }
    # Macro/weighted precision across the four decision classes.
    vals = [(per_class_f1[c]["precision_pct"], per_class_f1[c]["n"])
            for c in _PRED_CLASSES if per_class_f1[c]["precision_pct"] is not None]
    if vals:
        macro = round(sum(v for v, _ in vals) / len(vals), 1)
        wsum = sum(v * nn for v, nn in vals)
        nsum = sum(nn for _, nn in vals)
        weighted = round(wsum / nsum, 1) if nsum else None
    else:
        macro = None
        weighted = None

    # ROC-AUC: can we discriminate worker-agreed vs not using agent priority?
    scores: list[float] = []
    labels: list[int] = []
    for hid, verdict in fb_hazard.items():
        if hid not in pred_priority:
            continue
        scores.append(float(pred_priority[hid]))
        labels.append(1 if verdict in _FEEDBACK_AGREE else 0)
    auc = _roc_auc_binary(scores, labels)
    if auc is not None:
        auc = round(auc, 3)

    # Confusion matrix 4x3.
    confusion = [
        [by_class[c][v] for v in _FEEDBACK_VERDICTS]
        for c in _PRED_CLASSES
    ]

    # Uplift rates.
    def _pct(num: int) -> float:
        return round(100.0 * num / n, 1) if n else 0.0

    uplift = {
        "rule_tree_eq_final_rate": _pct(rule_tree_eq_final),
        "floor_fire_rate": _pct(floor_fire),
        "llm_flip_rate": _pct(llm_flip),
        "llm_clean_wins": llm_clean_wins,
    }

    # Cost extended.
    tokens_in_sorted = sorted(tokens_in)
    tokens_out_sorted = sorted(tokens_out)
    wall_sorted = sorted(wall_s)
    cost_ext = {
        "tokens_in_mean": round(sum(tokens_in) / len(tokens_in), 1) if tokens_in else 0.0,
        "tokens_in_p95": int(_percentile(tokens_in_sorted, 0.95)),
        "tokens_out_mean": round(sum(tokens_out) / len(tokens_out), 1) if tokens_out else 0.0,
        "tokens_out_p95": int(_percentile(tokens_out_sorted, 0.95)),
        "cost_per_case_usd": round(cost_sum / n, 4) if n else 0.0,
        "wall_p50_s": round(_percentile(wall_sorted, 0.50), 1),
        "wall_p95_s": round(_percentile(wall_sorted, 0.95), 1),
        "llm_calls_per_case_mean": round(sum(llm_calls_list) / len(llm_calls_list), 2)
        if llm_calls_list else 0.0,
    }

    # Justified histogram.
    histogram = {k: 0 for k in range(0, 7)}
    for v in per_run_justified_count:
        histogram[min(6, max(0, v))] += 1

    # Grounding.
    grounding_rate = _rate(grounding_ok, grounding_total)

    return {
        "n_hazards": n,
        "n_feedback_hazard": len(fb_hazard),
        "n_feedback_priority": len(fb_priority),
        "feedback_agreement": {
            "hazard_valid_rate": hazard_valid_rate,
            "priority_correct_rate": priority_correct_rate,
            "safety_floor_precision": sf_precision,
            "safety_floor_n": len(fp_intersect),
            "llm_uplift_precision": lu_precision,
            "llm_uplift_n": len(lu_intersect),
            "hazard_counts": {"agree": agree_n, "disagree": dis_n, "not_relevant": nr_n},
            "priority_counts": {
                "ok": pri_ok_n,
                "too_low": sum(1 for v in pri_votes if v == "too_low"),
                "too_high": sum(1 for v in pri_votes if v == "too_high"),
            },
            "by_decision_class": by_class,
        },
        "uplift": uplift,
        "cost_ext": cost_ext,
        "justified_histogram": histogram,
        "ds_metrics": {
            "confusion_matrix": confusion,
            "confusion_rows": list(_PRED_CLASSES),
            "confusion_cols": list(_FEEDBACK_VERDICTS),
            "per_class_precision": per_class_f1,
            "precision_macro_pct": macro,
            "precision_weighted_pct": weighted,
            "roc_auc": auc,
            "roc_auc_n": len(scores),
            "grounding_score_pct": grounding_rate,
            "grounding_n": grounding_total,
        },
    }


def _classify_llm_failure(llm_s: dict) -> str:
    """Bucket a node's failed LLM calls by the first concrete error text,
    falling back to the exception type. Buckets: timeout, http_5xx,
    http_429, http_4xx, connection, parse, other."""
    msg = str(llm_s.get("error_sample") or "").lower()
    types = " ".join(str(t) for t in (llm_s.get("errors") or [])).lower()
    text = msg + " " + types
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "429" in text or "ratelimit" in text or "rate limit" in text or "quota" in text:
        return "http_429"
    if any(code in text for code in ("http_5", " 500", " 502", " 503", " 504", "_50")):
        return "http_5xx"
    if any(code in text for code in ("401", "403", "404", "400", "http_4")):
        return "http_4xx"
    if "connect" in text or "refused" in text or "unreachable" in text or "dns" in text:
        return "connection"
    if "json" in text or "parse" in text or "contract" in text or "decode" in text:
        return "parse"
    return "other"


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
    database: dict[str, Any] = {
        "total": 0, "by_action": {}, "pending_approvals": 0, "overdue_approvals": 0, "open_evidence": 0,
    }
    try:
        from roadguard_agent import Database, _policy_queues

        with Database(settings.roadguard_db_path) as db:
            by_action: dict[str, int] = {}
            for key, n in db.count_hazards_since(0.0).items():
                action = key.split(":", 1)[1] if ":" in key else key
                by_action[action or "queued"] = by_action.get(action or "queued", 0) + n
            pending = db.list_pending_approvals(limit=500)
            sla_hours = _policy_queues()["approval_sla_hours"]
            database = {
                "total": sum(by_action.values()),
                "by_action": by_action,
                "pending_approvals": len(pending),
                "overdue_approvals": approvals_overdue(pending, sla_hours),
                "approval_sla_hours": sla_hours,
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
        f"  database  {db['total']} hazards: {actions}   approvals pending {db['pending_approvals']} "
        f"(overdue {db.get('overdue_approvals', 0)}, sla {db.get('approval_sla_hours', 24):g}h)   "
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


def _report_trace_summary(
    trace: list,
) -> tuple[list[dict[str, Any]], int, int, list[str], list[dict[str, Any]]]:
    """Timeline, token totals, cited sources and the LLM call list of one
    case, for /report.

    Tokens and models come from the ``llm_call`` entries chat() emits (one
    per provider call), attributed to the node whose entry follows them;
    a worker's ``llm`` summary dict and the legacy top-level fields are
    the fallback for traces written before that. Until 31.10.2026 the
    report read ``tokens_in`` / ``model`` off the node entries, where they
    no longer live, and showed 0 tokens and model null for every case.
    """
    entries = [e for e in trace if isinstance(e, dict)]
    speaker_after: list[str | None] = [None] * len(entries)
    nxt: str | None = None
    for i in range(len(entries) - 1, -1, -1):
        e = entries[i]
        if e.get("kind") != "llm_call" and e.get("speaker"):
            nxt = str(e["speaker"])
        speaker_after[i] = nxt
    has_call_entries = any(e.get("kind") == "llm_call" for e in entries)
    timeline: list[dict[str, Any]] = []
    cited: list[str] = []
    llm_calls: list[dict[str, Any]] = []
    tokens_in = tokens_out = 0
    for i, entry in enumerate(entries):
        raw = entry.get("content")
        content: dict[str, Any] = raw if isinstance(raw, dict) else {}
        kind = entry.get("kind")
        llm = content.get("llm") if isinstance(content.get("llm"), dict) else None
        llm_used = content.get("llm_used")
        if llm_used is None and llm is not None:
            llm_used = bool(llm.get("calls"))
        timeline.append({
            "iter": entry.get("iter"),
            "speaker": entry.get("speaker"),
            "kind": kind,
            "llm_used": llm_used,
            "source": content.get("source"),
        })
        for src in content.get("cited_sources") or []:
            if src not in cited:
                cited.append(src)
        if kind == "llm_call":
            tin = int(content.get("tokens_in") or 0)
            tout = int(content.get("tokens_out") or 0)
            tokens_in += tin
            tokens_out += tout
            llm_calls.append({
                "speaker": entry.get("speaker") or speaker_after[i],
                "model": content.get("model"),
                "tokens_in": tin,
                "tokens_out": tout,
                "latency_s": content.get("latency_s"),
                "ok": content.get("ok"),
            })
        elif not has_call_entries and llm_used:
            tin = int(((llm or {}).get("tokens_in") if llm else content.get("tokens_in")) or 0)
            tout = int(((llm or {}).get("tokens_out") if llm else content.get("tokens_out")) or 0)
            tokens_in += tin
            tokens_out += tout
            llm_calls.append({
                "speaker": entry.get("speaker"),
                "model": (llm or {}).get("model") or content.get("model"),
                "tokens_in": tin or None,
                "tokens_out": tout or None,
            })
    return timeline, tokens_in, tokens_out, cited, llm_calls


def _user_id_of(user: dict[str, Any]) -> str:
    """The id stored on hazards.user_id and compared on every ownership
    check.

    A registered user's token carries ``sub="user:<id>"``; the users row is
    the bare ``<id>``. Until 31.10.2026 the prefixed form was upserted as a
    second row on the first upload, with no admin flag, and /auth/me then
    resolved to that shadow row: the operator links vanished and the
    upload count moved. Passphrase users share one sub by design.
    """
    sub = str(user.get("sub") or "")
    if sub.startswith("user:") and len(sub) > 5:
        return sub[5:]
    return sub or user.get("email") or "anonymous"


def merge_prefixed_user_rows(db_path: str) -> int:
    """Startup migration (31.10.2026): fold ``user:<id>`` shadow rows into
    the real users rows; see Database.merge_prefixed_users."""
    from roadguard_agent import Database

    with Database(db_path) as db:
        n = db.merge_prefixed_users()
    if n:
        LOGGER.info("merged_prefixed_user_rows count=%d", n)
    return n


def _check_user_daily_submit_cap(
    user: dict[str, Any],
    db: Any,
    settings: Settings,
) -> None:
    """Enforce the per-user daily submission cap (step-1 auth upgrade).

    - A user with ``is_internal=True`` (e.g. batch@internal) is exempt - the
      offline batch runner must be able to push hundreds of images without
      starving.
    - Everyone else is capped at ``settings.roadguard_user_daily_submit_cap``
      per rolling 24-hour window; the 101st upload returns 429.
    - Legacy passphrase users whose sub does not resolve to a users-table row
      still get a conservative cap applied (counted by sub).
    """
    user_id = _user_id_of(user)
    cap = int(settings.roadguard_user_daily_submit_cap)
    if cap <= 0:
        return  # cap disabled
    u = None
    try:
        u = db.get_user(user_id)
        if u is None and (user.get("sub") or "").startswith("user:"):
            u = db.get_user(user["sub"].split(":", 1)[1])
    except Exception:
        u = None
    if u is not None and u.is_internal:
        return  # batch runner bypass
    import time as _t
    day_start = _t.time() - 24 * 3600
    count = db.count_user_hazards_since(user_id, day_start)
    if count >= cap:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"daily_submit_cap_reached ({cap}/day)",
        )


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
        # Shadow users rows left by the pre-31.10 submit path fold into the
        # real rows, then rows a dead background task left in
        # queued/processing go to a human before anything else runs.
        try:
            merge_prefixed_user_rows(settings.roadguard_db_path)
        except Exception as exc:
            LOGGER.warning("merge_prefixed_user_rows_failed: %s", exc)
        try:
            recover_interrupted_runs(settings.roadguard_db_path)
        except Exception as exc:
            LOGGER.warning("recover_interrupted_runs_failed: %s", exc)
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

    @app.get("/flow", response_class=HTMLResponse)
    @app.get("/flow/")
    def flow_viewer(request: Request) -> HTMLResponse:
        """Private visual agent-flow page: live node diagram of the pipeline
        (sanitize -> signal_extractor -> profile_builder -> rag_retriever ->
        decision -> approval_gate, with await_evidence branching from
        decision). Polls /report/{hazard_id} every 2s and colours each node
        by state. Same host-gate as /trace and /supervisor
        (see _private_page_or_404)."""
        return _private_page_or_404(request, "flow.html")

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

    @app.get("/health/system")
    def health_system(request: Request) -> dict:
        """Deeper health snapshot: loaded ollama models and their memory
        footprint, host RAM usage, stub-mode flag. Not for the citizen page -
        shown on the operator trace / digest and callable by the Decision
        agent as a tool (check_system_health) when a verdict feels off."""
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="Not Found")
        from roadguard_agent import check_system_health as _chk
        return _chk()

    @app.get("/images/{hazard_id}")
    def hazard_image(hazard_id: str) -> Any:
        """Serve the stored citizen photo for a hazard. Returns 404 if the
        image was never uploaded or was deleted by the 7-day retention cron.
        No auth required: these photos are municipal evidence a supervisor
        or the public can see in the daily digest / case page."""
        from fastapi.responses import FileResponse as _FileResponse
        from pathlib import Path as _Path
        import re as _re
        if not _re.fullmatch(r"hz_[0-9a-f]{6,32}", hazard_id or ""):
            raise HTTPException(status_code=400, detail="invalid hazard_id")
        path = _Path("/var/lib/roadguard/images") / f"{hazard_id}.jpg"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="image not found or expired")
        return _FileResponse(path, media_type="image/jpeg")

    def _build_live_digest_html(window_hours: int = 24) -> str:
        """Build the digest HTML on demand from the current DB.

        Layout: a two-pane console - the hazards table on the left, a sticky
        detail panel on the right that shows the selected row's photo,
        rationale, pipeline and cited sources. Three tabs (Overview /
        Cases / Timeline) keep everything on one scroll-free screen."""
        import sqlite3, json, html, os
        from datetime import datetime, timedelta, timezone
        NOW = datetime.now(timezone.utc)
        CUT = NOW - timedelta(hours=window_hours)
        cutoff_epoch = CUT.timestamp()
        con = sqlite3.connect("/var/lib/roadguard/roadguard.db")
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, class, confidence, agent_decision, priority, agent_rationale, "
            "       reported_at, location, status, cited_sources_json, agent_trace_json, "
            "       agent_value_json, bboxes_json "
            "FROM hazards WHERE reported_at >= ? ORDER BY reported_at DESC",
            (cutoff_epoch,),
        ).fetchall()
        counts = {"urgent": 0, "standard": 0, "needs_more_evidence": 0,
                  "human_review": 0, "known_image": 0, "other": 0}
        for r in rows:
            k = (r["agent_decision"] or "other").lower()
            counts[k if k in counts else "other"] += 1
        approvals_pending = con.execute("SELECT COUNT(*) FROM approvals WHERE status='pending'").fetchone()[0]
        evidence_open = con.execute("SELECT COUNT(*) FROM evidence_requests WHERE status='pending'").fetchone()[0]
        # Round 22 (4.11.2026): agent value snapshot for the KPI strip -
        # how often the workers' LLM moved the rule tree, per the same
        # per-case summary written to hazards.agent_value_json.
        av_rows = con.execute(
            "SELECT agent_value_json FROM hazards WHERE reported_at >= ? AND agent_value_json != ''",
            (cutoff_epoch,),
        ).fetchall()
        # Round 23 (4.11.2026): also walk the full trace aggregator so the
        # per-agent panel in the Agent value tab can read per_agent /
        # agents_justified / cost_total from one source. The call is
        # cheap (reads the same rows the digest already fetched).
        aggregates = _aggregate_agent_metrics(rows)
        av_counts = {"runs": 0, "changed_rule_hint": 0,
                     "profile_llm_changed": 0, "research_llm_changed": 0,
                     "enforcement_rule_tree_verdict": 0, "floor_applied": 0}
        for r in av_rows:
            try:
                avj = json.loads(r["agent_value_json"] or "{}")
            except (TypeError, ValueError):
                continue
            if not isinstance(avj, dict):
                continue
            av_counts["runs"] += 1
            if avj.get("changed_rule_hint"):
                av_counts["changed_rule_hint"] += 1
            if avj.get("profile_llm_changed"):
                av_counts["profile_llm_changed"] += 1
            if avj.get("research_llm_changed"):
                av_counts["research_llm_changed"] += 1
            if avj.get("enforcement") == "rule_tree_verdict":
                av_counts["enforcement_rule_tree_verdict"] += 1
            if avj.get("floor_applied"):
                av_counts["floor_applied"] += 1

        # #11: hazard clusters. Group hazards of the same class at the same
        # free-text location within the window; anything with 3+ reports is
        # worth surfacing as a possible structural issue (one pothole becomes
        # a recurring pattern; a crack grows into a cluster of cases).
        # NULLIF/COALESCE collapses empty-string locations so we do not
        # cluster all 'no location' submissions together.
        cluster_rows = con.execute(
            """
            SELECT
              COALESCE(NULLIF(TRIM(location), ''), '(no location)') AS loc,
              class,
              COUNT(*) AS n,
              MIN(reported_at) AS first_ts,
              MAX(reported_at) AS last_ts,
              GROUP_CONCAT(id) AS ids,
              GROUP_CONCAT(agent_decision) AS decisions
            FROM hazards
            WHERE reported_at >= ?
            GROUP BY loc, class
            HAVING n >= 3
            ORDER BY n DESC, last_ts DESC
            LIMIT 20
            """,
            (cutoff_epoch,),
        ).fetchall()

        # Pre-build a JSON payload the right-pane JS can read without
        # another round-trip. Each case carries everything the detail view
        # needs; the table is just a tbody rendered from this list.
        cases: list[dict] = []
        for r in rows[:100]:
            try:
                trace = json.loads(r["agent_trace_json"] or "[]")
                speakers = " -> ".join(dict.fromkeys(e.get("speaker", "?") for e in trace))
            except Exception:
                speakers = ""
                trace = []
            try:
                cited = json.loads(r["cited_sources_json"] or "[]")
            except Exception:
                cited = []
            # Round 23 iter 4 (4.11.2026): always set the image URL so
            # the detail pane can retry; the client uses onerror to hide
            # the <img> when the file is still being written. The old
            # os.path.isfile gate hid freshly ingested images for up to
            # one digest refresh window (~30s).
            img_rel = f"/images/{r['id']}"
            try:
                av = json.loads(r["agent_value_json"] or "{}")
            except Exception:
                av = {}
            per_agent = av.get("per_agent_justified") or {}
            # Round 23 iter 4: YOLO live confidence for the hazard, read
            # from the hazards row directly (the `confidence` column is
            # the YOLO top1_confidence stamped on submission).
            try:
                score = float(r["confidence"]) if r["confidence"] is not None else None
            except (TypeError, ValueError, IndexError):
                score = None
            cases.append({
                "id": r["id"],
                "decision": (r["agent_decision"] or "n/a").lower(),
                "priority": r["priority"],
                "status": r["status"] or "open",
                "location": r["location"] or "(not provided)",
                "reported_at": r["reported_at"] or 0,
                "rationale": r["agent_rationale"] or "",
                "speakers": speakers,
                "cited": cited,
                "image": img_rel,
                "n_trace": len(trace),
                "justified_count": av.get("agents_justified_count"),
                "agents_total": av.get("agents_total"),
                "rule_tree_action": av.get("rule_tree_action"),
                "final_action": av.get("final_action"),
                "changed_rule_hint": bool(av.get("changed_rule_hint")),
                "per_agent_justified": per_agent,
                "hazard_class": (r["class"] or "").upper(),
                "score": score,
                "bboxes": (lambda: json.loads(r["bboxes_json"] or "[]") if r["bboxes_json"] else [])(),
            })
        con.close()
        cases_json = json.dumps(cases)

        # Row class name per decision colour. The JS keeps the same keys.
        def _css_for(d: str) -> str:
            d = (d or "").lower()
            if d == "urgent": return "urgent"
            if d == "human_review": return "human"
            if d == "standard": return "standard"
            if d == "needs_more_evidence": return "evi"
            return "k"

        # Overview tiles are inlined above the Cases table so the operator
        # sees the headline numbers and the cases themselves on one scroll.
        # Round 23 (4.11.2026): agreement with the rule tree moved out of
        # the main strip into the new "Agent value" tab (see agent_value_html
        # below). The main strip keeps the queue counters and the second
        # round-22 row on their own, as the operator asked.
        correct_pct = 100.0 * sum(
            1 for r in av_rows if (r["agent_value_json"] or "") and
            (json.loads(r["agent_value_json"]).get("rule_tree_action")
             == json.loads(r["agent_value_json"]).get("final_action"))
        ) / max(1, av_counts["runs"]) if av_counts["runs"] else 0.0
        # Round 23 iter 4 (4.11.2026): owner asked for ONE row with only
        # the essentials - the queue shape plus the two agent-value
        # counts that matter to the owner's eye (LLM changed tree, floor
        # fired). The other twelve tiles moved out; deeper detail lives
        # in the Agent value tab.
        kpi_strip = (
            "<div class='kpi-strip'>"
            f"<div class='kpi'><div class='kpi-v'>{len(rows)}</div><div class='kpi-l'>New reports</div></div>"
            f"<div class='kpi urgent'><div class='kpi-v'>{counts['urgent']}</div><div class='kpi-l'>Urgent</div></div>"
            f"<div class='kpi human'><div class='kpi-v'>{counts['human_review']}</div><div class='kpi-l'>Human review</div></div>"
            f"<div class='kpi standard'><div class='kpi-v'>{counts['standard']}</div><div class='kpi-l'>Standard</div></div>"
            f"<div class='kpi'><div class='kpi-v'>{av_counts['changed_rule_hint']}</div><div class='kpi-l'>LLM changed the rule tree</div></div>"
            f"<div class='kpi'><div class='kpi-v'>{av_counts['floor_applied']}</div><div class='kpi-l'>Floor applied</div></div>"
            "</div>"
        )

        # Round 23 (4.11.2026): readable hazard label - the YOLO class
        # the agents identified in the image, with its plain name so a
        # city worker reads it without the dataset code.
        _HAZARD_LABEL = {
            "D00": "longitudinal crack",
            "D10": "transverse crack",
            "D20": "alligator crack",
            "D40": "pothole",
        }

        def _hazard_label(cls_code: str) -> str:
            code = (cls_code or "").upper()
            name = _HAZARD_LABEL.get(code, "")
            return f"{code} {name}".strip() if code else "n/a"

        if rows:
            # Fetch the per-case class the AGENTS identified (profile
            # builder's primary_class, with the YOLO live detection as
            # the fallback when the agent left it blank) alongside the
            # existing hazards.class column.
            id_to_class = {}
            for r in rows:
                try:
                    trace = json.loads(r["agent_trace_json"] or "[]")
                except Exception:
                    trace = []
                primary = ""
                for e in trace:
                    if not isinstance(e, dict):
                        continue
                    if e.get("speaker") == "profile_builder":
                        pc = (e.get("content") or {}).get("primary_class")
                        if pc:
                            primary = str(pc).upper()
                            break
                id_to_class[r["id"]] = primary or str(r["class"] or "")
            rows_html_parts = []
            # Round 23 iter 4 (4.11.2026): class filter counters for the
            # sub-tabs above the Cases table.
            class_counts = {"ALL": 0, "D00": 0, "D10": 0, "D20": 0, "D40": 0, "OTHER": 0}
            for c in cases:
                code = (id_to_class.get(c["id"]) or "").upper()
                class_counts["ALL"] += 1
                if code in class_counts:
                    class_counts[code] += 1
                else:
                    class_counts["OTHER"] += 1
            for i, c in enumerate(cases, 1):
                ts = datetime.fromtimestamp(c["reported_at"], timezone.utc).strftime("%H:%M")
                d = html.escape(c["decision"])
                cls = _css_for(c["decision"])
                hid = html.escape(c['id'])
                cls_code = (id_to_class.get(c["id"]) or "").upper() or "OTHER"
                hazard_label = html.escape(_hazard_label(id_to_class.get(c["id"], "")))
                # Round 23 iter 5 (4.11.2026): per-row feedback has TWO
                # axes - identification (hazard: agree/disagree/not
                # relevant) and urgency (priority: ok / too low / too
                # high). Each axis's clicked button picks up the active-*
                # class so the active vote per axis is visible without
                # a separate column.
                hazard_cell = (
                    f"<button class='fb-btn fb-agree' data-id='{hid}' data-kind='hazard' data-verdict='agree' title='Hazard was correctly identified'>Agree</button>"
                    f"<button class='fb-btn fb-disagree' data-id='{hid}' data-kind='hazard' data-verdict='disagree' title='Hazard identification is wrong'>Disagree</button>"
                    f"<button class='fb-btn fb-nr' data-id='{hid}' data-kind='hazard' data-verdict='not_relevant' title='Image is not a hazard / noise / other'>Not relevant</button>"
                )
                priority_cell = (
                    f"<button class='fb-btn fb-pri-ok' data-id='{hid}' data-kind='priority' data-verdict='ok' title='Priority is right'>OK</button>"
                    f"<button class='fb-btn fb-pri-lo' data-id='{hid}' data-kind='priority' data-verdict='too_low' title='Priority should be higher'>Too low</button>"
                    f"<button class='fb-btn fb-pri-hi' data-id='{hid}' data-kind='priority' data-verdict='too_high' title='Priority should be lower'>Too high</button>"
                )
                score_text = f"{c['score']:.2f}" if c.get("score") is not None else "-"
                rows_html_parts.append(
                    f"<tr data-idx='{i-1}' data-hzc='{cls_code}' class='row'>"
                    f"<td>{i}</td>"
                    f"<td><code>{html.escape(c['id'][:12])}</code></td>"
                    f"<td class='td-hazard'>{hazard_label}</td>"
                    f"<td class='td-score'>{score_text}</td>"
                    f"<td class='{cls}'>{d}</td>"
                    f"<td>{c['priority'] or ''}</td>"
                    f"<td class='td-act' data-hid='{hid}' data-kind='hazard'>{hazard_cell}</td>"
                    f"<td class='td-act' data-hid='{hid}' data-kind='priority'>{priority_cell}</td>"
                    f"<td>{html.escape(c['location'])[:60]}</td>"
                    f"<td>{ts}</td>"
                    f"</tr>"
                )
            rows_html = "".join(rows_html_parts)
            class_chip_bar_parts = []
            for code in ("ALL", "D40", "D20", "D00", "D10", "OTHER"):
                if code != "ALL" and class_counts.get(code, 0) == 0:
                    continue
                active_cls = " active" if code == "ALL" else ""
                chip_label = "ALL" if code == "ALL" else _hazard_label(code)
                class_chip_bar_parts.append(
                    f"<button class='cls-chip{active_cls}' data-cls='{code}'>"
                    f"{html.escape(chip_label)} "
                    f"<span class='cls-n'>({class_counts.get(code, 0)})</span>"
                    f"</button>"
                )
            class_chip_bar = (
                "<div class='cls-chips' id='cls-chips'>"
                + "".join(class_chip_bar_parts) + "</div>"
            )
            table_html = (
                class_chip_bar +
                "<table class='cases'>"
                "<thead><tr><th>#</th><th>id</th>"
                "<th title='Hazard type the agents identified in the image'>hazard the agents saw</th>"
                "<th title='YOLO live confidence (0-1)'>score</th>"
                "<th>decision</th><th>priority</th>"
                "<th title='Was the agent right about the hazard?'>hazard feedback</th>"
                "<th title='Was the agent right about the urgency / priority?'>priority feedback</th>"
                "<th>location</th><th>time</th></tr></thead>"
                f"<tbody id='rows'>{rows_html}</tbody>"
                "</table>"
            )
        else:
            table_html = "<p><i>No hazards reported in the last {h} hours.</i></p>".replace("{h}", str(window_hours))

        # Timeline tab: an accordion - each case shows only its headline
        # line, click to expand and see the rationale + pipeline chain.
        tl_parts = []
        for i, c in enumerate(cases, 1):
            ts = datetime.fromtimestamp(c["reported_at"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            d = html.escape(c["decision"])
            cls = _css_for(c["decision"])
            tl_parts.append(
                f"<details class='tl-item'>"
                f"<summary class='tl-head'><span class='tl-i'>{i}</span>"
                f"<span class='{cls}'>{d}</span>"
                f"<span class='k'>priority {c['priority'] or 'n/a'} &middot; {ts} &middot; {html.escape(c['status'])} &middot; {html.escape(c['location'])}</span></summary>"
                f"<div class='tl-body'>"
                f"<div class='tl-pipe'>{html.escape(c['speakers'])}</div>"
                f"<div class='tl-r'>{html.escape((c['rationale'] or '(empty rationale)'))[:600]}</div>"
                f"</div>"
                f"</details>"
            )
        timeline_html = "<div class='tl'>" + ("".join(tl_parts) or "<p><i>No cases to show.</i></p>") + "</div>"

        # #11: clusters panel. Three or more hazards of the same class at
        # the same free-text location in the window - a possible structural
        # or recurring issue that one case card would hide.
        if cluster_rows:
            cp_rows = []
            for r in cluster_rows:
                ids = (r["ids"] or "").split(",")
                decisions = (r["decisions"] or "").split(",")
                dec_counts: dict[str, int] = {}
                for d in decisions:
                    if d:
                        dec_counts[d] = dec_counts.get(d, 0) + 1
                dec_summary = ", ".join(f"{v}x {k}" for k, v in sorted(dec_counts.items(), key=lambda x: -x[1]))
                first_ts = datetime.fromtimestamp(r["first_ts"] or 0, timezone.utc).strftime("%H:%M")
                last_ts = datetime.fromtimestamp(r["last_ts"] or 0, timezone.utc).strftime("%H:%M")
                cp_rows.append(
                    f"<tr>"
                    f"<td class='cl-n'>{r['n']}&times;</td>"
                    f"<td class='cl-cls'>{html.escape(r['class'] or '?')}</td>"
                    f"<td class='cl-loc'>{html.escape(r['loc'])[:80]}</td>"
                    f"<td class='cl-dec k'>{html.escape(dec_summary)[:100]}</td>"
                    f"<td class='k'>{first_ts} &rarr; {last_ts}</td>"
                    f"</tr>"
                )
            clusters_html = (
                "<div class='clusters-wrap'>"
                "<p class='cl-head'>&#9888; Possible structural / recurring issues "
                f"(same location + same class, &ge; 3 reports in the last {window_hours}h). "
                "Operator should check whether these need a work order rather than one-off repairs.</p>"
                "<table class='clusters'>"
                "<thead><tr><th>n</th><th>class</th><th>location</th><th>decisions</th><th>time range</th></tr></thead>"
                f"<tbody>{''.join(cp_rows)}</tbody>"
                "</table>"
                "</div>"
            )
        else:
            # Round 23 iter 4 (4.11.2026): owner asked to drop the empty-
            # state placeholder ("No clusters..."), the whole block is
            # hidden when no clusters exist in the window.
            clusters_html = ""

        header = (
            f"<header class='topbar'>"
            f"<h1>RoadGuard <span class='live'>LIVE</span></h1>"
            f"<div class='nav-right'>"
            f"<a href='/roadguard/trace' class='np'>Live trace &rarr;</a>"
            f"<a href='/roadguard/flow' class='np'>Agent flow &rarr;</a>"
            f"<a href='/roadguard/supervisor' class='np'>Supervisor inbox &rarr;</a>"
            f"<span class='k'>{NOW.strftime('%Y-%m-%d %H:%M UTC')} - last {window_hours}h <span id='refresh-dot'>&middot; auto-refresh 30s</span></span>"
            f"</div>"
            f"</header>"
            f"<nav class='tabs' id='tabs'>"
            f"<button class='tab-btn active' data-tab='cases'>Cases ({len(rows)})</button>"
            f"<button class='tab-btn' data-tab='agentvalue'>Agent value</button>"
            f"<button class='tab-btn' data-tab='dclasses'>D Classes</button>"
            f"<button class='tab-btn' data-tab='ruletree'>Rule tree</button>"
            f"</nav>"
        )

        style = """
        :root{--bg:#0f172a;--panel:#111827;--line:#1f2937;--text:#e5e7eb;--muted:#9ca3af;--brand:#38bdf8;--brand2:#22c55e;}
        *{box-sizing:border-box}
        html,body{margin:0;background:#f8fafc;color:#111;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
        .topbar{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:#fff;border-bottom:1px solid #e5e7eb;position:sticky;top:0;z-index:5;min-height:49px}
        .topbar h1{margin:0;font-size:20px;font-weight:800;letter-spacing:-0.01em;white-space:nowrap}
        .live{background:#dcfce7;color:#166534;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;margin:0 6px}
        .nav-right{display:flex;align-items:center;gap:14px;font-size:13px;color:#6b7280;flex-wrap:wrap;justify-content:flex-end}
        .np{color:#0ea5e9;text-decoration:none;font-weight:600;white-space:nowrap}
        .np:hover{text-decoration:underline}
        .tabs{display:flex;gap:4px;padding:0 20px;background:#fff;border-bottom:1px solid #e5e7eb;position:sticky;top:49px;z-index:4;min-height:44px}
        .tab-btn{appearance:none;background:transparent;border:0;border-bottom:3px solid transparent;padding:10px 16px;font:inherit;font-size:14px;cursor:pointer;color:#6b7280;font-weight:600}
        .tab-btn.active{color:#0ea5e9;border-bottom-color:#0ea5e9}
        .tab-btn:hover{color:#111}
        .tab-pane{display:none;padding:20px}
        .tab-pane.active{display:block}
        /* KPI strip above the Cases table */
        .kpi-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px}
        .kpi{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;text-align:center}
        .kpi-v{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:22px;font-weight:800;color:#0ea5e9;line-height:1.1}
        .kpi.urgent .kpi-v{color:#b91c1c}
        .kpi.human .kpi-v{color:#1d4ed8}
        .kpi.standard .kpi-v{color:#047857}
        .kpi.evi .kpi-v{color:#b45309}
        .kpi-l{font-size:11px;color:#6b7280;margin-top:2px}
        /* Clusters panel */
        .clusters-wrap{background:#fff;border:1px solid #fecaca;border-left:4px solid #ef4444;border-radius:8px;padding:12px 14px;margin-bottom:16px}
        .clusters-wrap .cl-head{margin:0 0 10px;font-size:13px;color:#991b1b;font-weight:600}
        .clusters{width:100%;border-collapse:collapse;font-size:13px}
        .clusters th{text-align:left;color:#6b7280;font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:0.03em;padding:6px 8px;border-bottom:1px solid #e5e7eb}
        .clusters td{padding:6px 8px;border-bottom:1px solid #f1f5f9}
        .clusters tr:last-child td{border-bottom:0}
        .cl-n{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:700;color:#b91c1c;width:48px}
        .cl-cls{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;width:60px;color:#1f2937}
        /* Cases split */
        .split{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px;align-items:start}
        @media (max-width:980px){.split{grid-template-columns:1fr}}
        .cases{width:100%;background:#fff;border:1px solid #e5e7eb;border-radius:10px;border-collapse:separate;border-spacing:0;overflow:hidden}
        .cases th,.cases td{padding:8px 10px;border-bottom:1px solid #f1f5f9;text-align:left;font-size:13px}
        .cases th{background:#f8fafc;color:#6b7280;font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:0.03em}
        .cases tr.row{cursor:pointer}
        .cases tr.row:hover{background:#f1f5f9}
        .cases tr.row.sel{background:#e0f2fe;box-shadow:inset 3px 0 0 #0ea5e9}
        .urgent{color:#b91c1c;font-weight:600}
        .human{color:#1d4ed8;font-weight:600}
        .standard{color:#047857;font-weight:600}
        .evi{color:#b45309;font-weight:600}
        .k{color:#6b7280}
        /* Detail panel */
        .detail{position:sticky;top:110px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;max-height:calc(100vh - 130px);overflow-y:auto}
        .detail h3{margin:0 0 8px;font-size:15px;letter-spacing:-0.01em}
        .detail .d-img{width:100%;max-height:280px;object-fit:cover;border-radius:8px;background:#f1f5f9;display:block}
        .detail .d-img-wrap{position:relative;margin-bottom:12px;border-radius:8px;overflow:hidden}
        .detail .d-bbox{position:absolute;border:2px solid #22d3ee;box-shadow:0 0 0 1px rgba(0,0,0,0.4);box-sizing:border-box;pointer-events:none}
        .detail .d-bbox.d-bbox-D00{border-color:#f59e0b}
        .detail .d-bbox.d-bbox-D10{border-color:#a855f7}
        .detail .d-bbox.d-bbox-D20{border-color:#ef4444}
        .detail .d-bbox.d-bbox-D40{border-color:#22d3ee}
        .detail .d-bbox-lbl{position:absolute;top:-18px;left:0;background:rgba(17,24,39,0.85);color:#fff;font-family:ui-monospace,monospace;font-size:10px;font-weight:700;padding:2px 5px;border-radius:3px;white-space:nowrap}
        .detail .d-row{display:flex;justify-content:space-between;gap:12px;margin:6px 0;font-size:13px}
        .detail .d-row b{color:#6b7280;font-weight:500}
        .detail .d-rat{background:#f8fafc;border-radius:6px;padding:10px;font-size:13px;line-height:1.5;margin:8px 0}
        .detail .d-sources{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#6b7280}
        .detail .d-empty{color:#9ca3af;text-align:center;padding:40px 0;font-size:13px}
        /* Timeline (accordion) */
        .tl{display:flex;flex-direction:column;gap:8px}
        .tl-item{background:#fff;border:1px solid #e5e7eb;border-left:4px solid #e5e7eb;border-radius:8px;overflow:hidden}
        .tl-item summary{display:flex;gap:10px;align-items:center;font-size:13px;padding:12px 14px;cursor:pointer;list-style:none}
        .tl-item summary::-webkit-details-marker{display:none}
        .tl-item summary::before{content:'\\25B8';display:inline-block;margin-right:4px;color:#9ca3af;transition:transform 0.15s}
        .tl-item[open] summary::before{transform:rotate(90deg)}
        .tl-item .tl-i{color:#6b7280;font-family:ui-monospace,monospace;font-weight:700;min-width:24px}
        .tl-item .tl-body{padding:0 14px 12px 38px;border-top:1px solid #f1f5f9}
        .tl-item .tl-pipe{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:#6b7280;margin:8px 0 4px}
        .tl-item .tl-r{font-size:13px;color:#374151;line-height:1.5;margin-top:6px}
        /* Round 23 iter 4: hazard-label + score cells, 3 feedback buttons
           (always clickable, active verdict gets the active-* class),
           class-filter chips, and the Priority tab. */
        .td-hazard{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#1f2937;white-space:nowrap}
        .td-score{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#1f2937;white-space:nowrap;text-align:right}
        .fb-btn{appearance:none;background:#fff;border:1px solid #d1d5db;border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;margin:0 2px;font-weight:600;transition:background 0.1s,color 0.1s}
        .fb-btn.fb-agree{color:#166534;border-color:#86efac}
        .fb-btn.fb-disagree{color:#991b1b;border-color:#fecaca}
        .fb-btn.fb-nr{color:#6b7280;border-color:#d1d5db}
        .fb-btn.fb-pri-ok{color:#166534;border-color:#86efac}
        .fb-btn.fb-pri-lo{color:#92400e;border-color:#fde68a}
        .fb-btn.fb-pri-hi{color:#1e3a8a;border-color:#93c5fd}
        .fb-btn:hover{background:#f9fafb}
        .fb-btn.active-agree,.fb-btn.active-ok{background:#dcfce7;color:#166534;border-color:#86efac}
        .fb-btn.active-disagree{background:#fee2e2;color:#991b1b;border-color:#fecaca}
        .fb-btn.active-nr,.fb-btn.active-not_relevant{background:#f3f4f6;color:#1f2937;border-color:#d1d5db}
        .fb-btn.active-too_low{background:#fef3c7;color:#92400e;border-color:#fde68a}
        .fb-btn.active-too_high{background:#dbeafe;color:#1e3a8a;border-color:#93c5fd}
        .fb-btn.flash-ok{animation:flash-ok-anim 0.6s ease-out}
        @keyframes flash-ok-anim{0%{box-shadow:0 0 0 0 rgba(34,197,94,0.6)}100%{box-shadow:0 0 0 6px rgba(34,197,94,0)}}
        .fb-btn.flash-err{animation:flash-err-anim 0.6s ease-out}
        @keyframes flash-err-anim{0%{box-shadow:0 0 0 0 rgba(239,68,68,0.6)}100%{box-shadow:0 0 0 6px rgba(239,68,68,0)}}
        .td-act{white-space:nowrap}
        /* Class filter chips above the Cases table */
        .cls-chips{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 10px}
        .cls-chip{appearance:none;background:#fff;border:1px solid #d1d5db;border-radius:999px;padding:5px 12px;font-size:12px;cursor:pointer;font-weight:600;color:#374151;font-family:ui-monospace,monospace}
        .cls-chip:hover{background:#f9fafb}
        .cls-chip.active{background:#0ea5e9;color:#fff;border-color:#0ea5e9}
        .cls-chip .cls-n{opacity:0.75;font-weight:500}
        /* D Classes tab */
        .dc-wrap{max-width:100%}
        .dc-h{margin:0 0 6px;font-size:16px;font-weight:700;color:#111827}
        .dc-sub{color:#6b7280;font-size:13px;margin:0 0 18px}
        .dc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
        .dc-card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:6px}
        .dc-code{font-family:ui-monospace,monospace;font-size:12px;color:#6b7280;font-weight:700}
        .dc-name{font-size:14px;font-weight:700;color:#111827}
        .dc-img{width:100%;height:170px;object-fit:cover;border-radius:6px;background:#f3f4f6}
        .dc-img-missing{display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:12px;font-style:italic}
        .dc-desc{font-size:12px;color:#374151;line-height:1.4}
        /* Agent value tab */
        .av-wrap{max-width:1100px}
        .av-h{margin:0 0 10px;font-size:16px;font-weight:700;color:#111827}
        .av-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
        .av-card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;text-align:center}
        .av-card.ok{border-left:4px solid #16a34a}
        .av-card.fail{border-left:4px solid #dc2626}
        .av-v{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:28px;font-weight:800;color:#0ea5e9;line-height:1.1}
        .av-card.ok .av-v{color:#16a34a}
        .av-card.fail .av-v{color:#dc2626}
        .av-l{font-size:13px;color:#374151;margin-top:4px;font-weight:600}
        .av-t{font-size:11px;color:#6b7280;margin-top:2px}
        .av-overrides{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;font-size:13px}
        .av-overrides .av-ov-row{padding:8px 0;border-bottom:1px solid #f1f5f9}
        .av-overrides .av-ov-row:last-child{border-bottom:0}
        .av-ov-id{font-family:ui-monospace,monospace;color:#1d4ed8;font-weight:600}
        /* Round 23: per-agent breakdown grid in the Agent value tab */
        .ag-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
        .ag-card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px}
        .ag-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;border-bottom:1px solid #f1f5f9;padding-bottom:6px}
        .ag-name{font-family:ui-monospace,monospace;font-weight:700;color:#111827;font-size:14px}
        .ag-just{font-family:ui-monospace,monospace;font-size:12px;color:#166534;font-weight:600}
        .ag-metrics{display:grid;grid-template-columns:1fr;gap:4px}
        .ag-metric{display:flex;justify-content:space-between;font-size:12px}
        .ag-m-k{color:#374151}
        .ag-m-v{font-family:ui-monospace,monospace;font-weight:700;color:#111827}
        /* Round 26 (5.11.2026): the data-science metric panels in the
           Agent value tab. Histogram is a 7-bar block (0..6 of 6 agents
           justified) and the confusion matrix is a plain table. */
        .ds-hist{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;min-height:110px;display:flex;align-items:flex-end;gap:8px;justify-content:space-around}
        .ds-hist .ds-bar{display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;min-width:30px}
        .ds-hist .ds-bar-fill{background:#2563eb;border-radius:4px 4px 0 0;width:100%;min-height:2px;transition:height 0.3s}
        .ds-hist .ds-bar-lbl{font-size:11px;color:#374151;font-family:ui-monospace,monospace}
        .ds-hist .ds-bar-val{font-size:12px;color:#111827;font-weight:700;font-family:ui-monospace,monospace}
        .ds-hist-note{color:#6b7280;font-size:13px;text-align:center;width:100%}
        .ds-conf-wrap{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;overflow-x:auto}
        .ds-conf-note{color:#6b7280;font-size:13px;text-align:center}
        .ds-conf-table{border-collapse:collapse;font-size:13px;margin:0 auto}
        .ds-conf-table th,.ds-conf-table td{border:1px solid #e5e7eb;padding:8px 14px;text-align:center;font-family:ui-monospace,monospace}
        .ds-conf-table th{background:#f3f4f6;font-weight:700;color:#111827}
        .ds-conf-table .ds-row-hdr{background:#f9fafb;font-weight:600;color:#374151;text-align:left}
        .ds-conf-table .ds-cell-agree{background:rgba(22,163,74,0.08);color:#166534;font-weight:700}
        .ds-conf-table .ds-cell-dis{background:rgba(220,38,38,0.08);color:#991b1b;font-weight:700}
        .ds-conf-table .ds-cell-nr{background:rgba(234,88,12,0.08);color:#9a3412;font-weight:700}
        .ds-conf-table .ds-cell-precision{background:#eff6ff;color:#1e40af;font-weight:700}
        /* Round 23 iter 5 (4.11.2026): rule tree canvas scrolls
           horizontally with explicit left / right arrows. The canvas
           breaks out of the tab-pane padding (negative margins of 20px
           both sides) so the scroll region covers the full viewport
           width and the right-hand leaves do not clip against the
           pane's own padding. */
        .rt-wrap{max-width:100%}
        .rt-h{margin:0 0 6px;font-size:16px;font-weight:700;color:#111827}
        .rt-sub{color:#6b7280;font-size:13px;margin:0 0 10px}
        .rt-scrollbar-wrap{display:flex;gap:8px;align-items:center;margin-bottom:8px;position:sticky;left:0}
        .rt-scroll-btn{appearance:none;background:#fff;border:1px solid #d1d5db;border-radius:8px;padding:6px 14px;font-size:16px;font-weight:700;cursor:pointer;color:#1f2937;min-width:48px}
        .rt-scroll-btn:hover{background:#f3f4f6}
        .rt-scroll-btn:active{background:#e5e7eb}
        .rt-scroll-hint{font-size:11px;color:#6b7280;margin-left:4px}
        .rt-canvas{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;overflow-x:scroll;overflow-y:hidden;scroll-behavior:smooth;margin-right:-20px;margin-left:0}
        .rt-canvas::-webkit-scrollbar{height:14px}
        .rt-canvas::-webkit-scrollbar-track{background:#f3f4f6;border-radius:7px}
        .rt-canvas::-webkit-scrollbar-thumb{background:#9ca3af;border-radius:7px;border:2px solid #f3f4f6}
        .rt-canvas::-webkit-scrollbar-thumb:hover{background:#6b7280}
        .rt-svg{min-width:2000px;width:2000px;height:auto;display:block}
        """

        script = """
        const CASES = __CASES__;
        const tabs = document.querySelectorAll('.tab-btn');
        const panes = document.querySelectorAll('.tab-pane');
        tabs.forEach(b => b.addEventListener('click', () => {
          tabs.forEach(x => x.classList.toggle('active', x === b));
          panes.forEach(p => p.classList.toggle('active', p.id === 'pane-' + b.dataset.tab));
        }));
        const detail = document.getElementById('detail');
        const rows = document.querySelectorAll('.cases tr.row');
        function pick(idx) {
          rows.forEach(r => r.classList.toggle('sel', +r.dataset.idx === idx));
          const c = CASES[idx];
          if (!c) { detail.innerHTML = '<div class="d-empty">Pick a case on the left.</div>'; return; }
          // Round 23 iter 6 (4.11.2026): always try the image URL and
          // overlay YOLO bboxes on top of the photo so the worker sees
          // exactly where the agents said the hazard was. The overlay
          // is positioned by percentages against the image's natural
          // size, so it scales with the <img> on resize. The onload /
          // onerror handlers use global helpers (__rgDrawBoxes /
          // __rgImgErr) to keep the quoting simple inside this inline
          // template.
          const imgPath = c.image || (urlBase() + '/images/' + c.id);
          const bboxes = c.bboxes || [];
          const bboxDataAttr = bboxes.length
            ? ' data-bboxes="' + encodeURIComponent(JSON.stringify(bboxes)) + '"'
            : '';
          const img = (
            '<div class="d-img-wrap"' + bboxDataAttr + '>' +
            '<img class="d-img" src="' + imgPath + '" alt="citizen photo" ' +
            'onload="window.__rgDrawBoxes && window.__rgDrawBoxes(this)" ' +
            'onerror="window.__rgImgErr && window.__rgImgErr(this)">' +
            '</div>'
          );
          const cited = c.cited && c.cited.length ? c.cited.map(s => `<code>${s}</code>`).join(', ') : '<i>none cited</i>';
          const decCls = c.decision === 'urgent' ? 'urgent' : (c.decision === 'human_review' ? 'human' : (c.decision === 'standard' ? 'standard' : 'evi'));
          detail.innerHTML = `
            <h3>Case <code>${c.id}</code></h3>
            ${img}
            <div class="d-row"><b>Decision</b><span class="${decCls}">${c.decision}</span></div>
            <div class="d-row"><b>Priority</b><span>${c.priority ?? 'n/a'}</span></div>
            <div class="d-row"><b>Status</b><span>${c.status}</span></div>
            <div class="d-row"><b>Location</b><span>${c.location}</span></div>
            <div class="d-row"><b>Pipeline</b><span class="d-sources">${c.speakers || '(empty)'}</span></div>
            <div class="d-rat">${c.rationale || '<i>empty rationale</i>'}</div>
            <div class="d-row"><b>Cited sources</b><span class="d-sources">${cited}</span></div>
          `;
        }
        rows.forEach(r => r.addEventListener('click', () => pick(+r.dataset.idx)));
        if (rows.length) pick(0);
        // Round 23 iter 3 (4.11.2026): three-button worker feedback.
        // Buttons stay clickable (undo / change your mind), each click
        // POSTs a new verdict row that becomes the case's latest. Any
        // HTTP non-2xx flashes the button red and does NOT update the
        // 'your vote' pill, so the worker sees his click did NOT save.
        function urlBase() { return location.pathname.replace(/\\/digest.*/, ''); }
        // Round 23 iter 6 (4.11.2026): draw YOLO bboxes on the detail-
        // pane image. bboxes format: [[x1, y1, x2, y2, class, conf], ...]
        // in the image's own pixel space; the overlay uses percentages of
        // naturalWidth / naturalHeight so it scales with the <img>.
        window.__rgImgErr = function(imgEl) {
          try {
            imgEl.style.display = 'none';
            const p = document.createElement('div');
            p.className = 'd-img';
            p.style.cssText = 'display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:12px;height:160px';
            p.textContent = '(photo not yet written)';
            if (imgEl.parentNode) imgEl.parentNode.insertBefore(p, imgEl);
          } catch (e) { /* silent */ }
        };
        window.__rgDrawBoxes = function(imgEl) {
          try {
            const wrap = imgEl.closest('.d-img-wrap');
            if (!wrap) return;
            wrap.querySelectorAll('.d-bbox').forEach(b => b.remove());
            const raw = wrap.dataset.bboxes;
            if (!raw) return;
            const bboxes = JSON.parse(decodeURIComponent(raw));
            const w = imgEl.naturalWidth, h = imgEl.naturalHeight;
            if (!w || !h) return;
            bboxes.forEach((b, i) => {
              const [x1, y1, x2, y2, cls, conf] = b;
              const left = (100 * x1 / w) + '%';
              const top = (100 * y1 / h) + '%';
              const width = (100 * (x2 - x1) / w) + '%';
              const height = (100 * (y2 - y1) / h) + '%';
              const div = document.createElement('div');
              div.className = 'd-bbox d-bbox-' + (cls || '').replace(/[^A-Za-z0-9]/g, '');
              div.style.cssText = 'left:' + left + ';top:' + top +
                ';width:' + width + ';height:' + height;
              const lbl = document.createElement('div');
              lbl.className = 'd-bbox-lbl';
              lbl.textContent = (cls || '?') + ' ' + (conf ? Number(conf).toFixed(2) : '');
              div.appendChild(lbl);
              wrap.appendChild(div);
            });
          } catch (e) { /* silent */ }
        };
        const ACTIVE_CLASSES = ['active-agree','active-disagree','active-nr','active-not_relevant','active-ok','active-too_low','active-too_high'];
        // Round 23 iter 5 (4.11.2026): mark the verdict per (hazard, kind)
        // axis so hazard feedback and priority feedback are visible
        // together.
        function markVote(hid, kind, verdict) {
          const sel = ".td-act[data-hid='" + hid + "'][data-kind='" + kind + "'] .fb-btn";
          document.querySelectorAll(sel).forEach(b => {
            ACTIVE_CLASSES.forEach(c => b.classList.remove(c));
            if (b.dataset.verdict === verdict) {
              b.classList.add('active-' + verdict);
            }
          });
        }
        function clearVote(hid, kind) {
          const sel = ".td-act[data-hid='" + hid + "'][data-kind='" + kind + "'] .fb-btn";
          document.querySelectorAll(sel).forEach(b => {
            ACTIVE_CLASSES.forEach(c => b.classList.remove(c));
          });
        }
        // Round 23 iter 5: Rule tree pan arrows. Clicking the chevron
        // buttons scrolls the canvas by ~85% of its visible width so
        // two clicks typically cover the whole tree.
        function bindRuleTreeScroll() {
          const canvas = document.getElementById('rt-canvas');
          const l = document.getElementById('rt-scroll-left');
          const r = document.getElementById('rt-scroll-right');
          if (!canvas || !l || !r) return;
          l.addEventListener('click', () => {
            canvas.scrollBy({left: -Math.round(canvas.clientWidth * 0.85), behavior: 'smooth'});
          });
          r.addEventListener('click', () => {
            canvas.scrollBy({left: Math.round(canvas.clientWidth * 0.85), behavior: 'smooth'});
          });
        }
        bindRuleTreeScroll();
        // Round 23 iter 4: class-filter chips above the Cases table.
        function bindClassChips() {
          const chips = document.querySelectorAll('.cls-chip');
          chips.forEach(chip => {
            chip.addEventListener('click', () => {
              chips.forEach(c => c.classList.toggle('active', c === chip));
              const target = chip.dataset.cls || 'ALL';
              document.querySelectorAll('.cases tr.row').forEach(r => {
                r.style.display = (target === 'ALL' || r.dataset.hzc === target)
                  ? '' : 'none';
              });
            });
          });
        }
        bindClassChips();
        function create(tag, cls, text) {
          const n = document.createElement(tag);
          if (cls) n.className = cls;
          if (text != null) n.textContent = text;
          return n;
        }
        function bindFbButtons(root) {
          (root || document).querySelectorAll('.fb-btn').forEach(btn => {
            if (btn.dataset.bound === '1') return;
            btn.dataset.bound = '1';
            btn.addEventListener('click', async (ev) => {
              ev.stopPropagation();
              const hid = btn.dataset.id;
              const kind = btn.dataset.kind || 'hazard';
              const verdict = btn.dataset.verdict || 'agree';
              try {
                const r = await fetch(urlBase() + '/agent_feedback/' + encodeURIComponent(hid), {
                  method: 'POST',
                  headers: {'content-type': 'application/json'},
                  body: JSON.stringify({kind: kind, verdict: verdict, note: ''}),
                });
                if (!r.ok) { throw new Error('http ' + r.status); }
                btn.classList.remove('flash-err');
                btn.offsetWidth; // reflow to retrigger animation
                btn.classList.add('flash-ok');
                setTimeout(() => btn.classList.remove('flash-ok'), 700);
                markVote(hid, kind, verdict);
                refreshAgentValue();
              } catch (e) {
                btn.classList.remove('flash-ok');
                btn.offsetWidth;
                btn.classList.add('flash-err');
                setTimeout(() => btn.classList.remove('flash-err'), 700);
              }
            });
          });
        }
        bindFbButtons(document);
        // Round 23 iter 3: load the latest vote per hazard and paint
        // the 'your vote' cells, so a worker reloading the page sees
        // what he already voted. Polled every 30s by the digest's own
        // refresh timer.
        async function refreshVotes() {
          try {
            const d = await fetch(urlBase() + '/agent_feedback/by_hazard?window_hours=24', {cache:'no-store'})
                      .then(r => r.json());
            const byH = d.by_hazard || {};
            document.querySelectorAll('.td-act[data-hid]').forEach(td => {
              const hid = td.dataset.hid;
              const kind = td.dataset.kind || 'hazard';
              const entry = byH[hid];
              const v = entry && entry[kind];
              if (v && v.verdict) markVote(hid, kind, v.verdict);
              else clearVote(hid, kind);
            });
          } catch (e) { /* silent */ }
        }
        refreshVotes();
        // Round 23: the Agent value tab loads the three-verdict counters
        // from /stats/feedback, refreshes after every button click.
        async function refreshAgentValue() {
          try {
            const base = location.pathname.replace(/\\/digest.*/, '');
            const d = await fetch(base + '/stats/feedback?window_hours=24', {cache:'no-store'})
                      .then(r => r.json());
            const hz = d.hazard || {agree:0, disagree:0, not_relevant:0, total_cases_voted:0};
            const pri = d.priority || {ok:0, too_low:0, too_high:0, total_cases_voted:0};
            const hzTotal = hz.total_cases_voted || 0;
            function setCard(idv, idt, n, total, kind) {
              const v = document.getElementById(idv);
              const t = document.getElementById(idt);
              if (v) v.textContent = String(n);
              if (t) t.textContent = total ? (n + ' of ' + total + ' cases voted') : 'no ' + kind + ' votes yet';
            }
            setCard('av-agree-v', 'av-agree-t', hz.agree || 0, hzTotal, 'hazard');
            setCard('av-dis-v',   'av-dis-t',   hz.disagree || 0, hzTotal, 'hazard');
            setCard('av-nr-v',    'av-nr-t',    hz.not_relevant || 0, hzTotal, 'hazard');
            function renderList(id, items, empty) {
              const list = document.getElementById(id);
              if (!list) return;
              if (!items || !items.length) {
                list.innerHTML = '<div class="k" style="color:#6b7280">' + empty + '</div>';
                return;
              }
              list.innerHTML = items.map(o =>
                '<div class="av-ov-row"><span class="av-ov-id">' + (o.hazard_id || '').substring(0,12) +
                '</span> &middot; <b>' + (o.class || '') + '</b> &middot; agent said <b>' + (o.decision || '') +
                '</b>' + (o.note ? ' &middot; <i>' + o.note + '</i>' : '') + '</div>'
              ).join('');
            }
            renderList('av-disagree-list', d.disagrees, 'No disagreements yet.');
            renderList('av-nr-list', d.not_relevants, "No 'not relevant' flags yet.");
          } catch (e) { /* silent */ }
        }
        refreshAgentValue();
        // Round 26 (5.11.2026): fetch /stats/ds_metrics and render the
        // 20 data-science panels in the Agent value tab. A metric that
        // comes back null displays 'pending feedback' rather than 0.
        async function refreshDsMetrics() {
          try {
            const base = location.pathname.replace(/\/digest.*/, '');
            const d = await fetch(base + '/stats/ds_metrics?window_hours=24', {cache:'no-store'})
                      .then(r => r.json());
            function setM(idv, idt, value, hint, suffix) {
              const v = document.getElementById(idv);
              const t = document.getElementById(idt);
              if (!v) return;
              if (value === null || value === undefined) {
                v.textContent = '-';
                if (t) t.textContent = hint || 'pending feedback';
              } else {
                v.textContent = String(value) + (suffix || '');
                if (t && hint) t.textContent = hint;
              }
            }
            const fa = d.feedback_agreement || {};
            const upl = d.uplift || {};
            const cx = d.cost_ext || {};
            const ds = d.ds_metrics || {};
            const nH = d.n_feedback_hazard || 0;
            const nP = d.n_feedback_priority || 0;
            // 1-5: feedback agreement
            setM('ds-m1-v','ds-m1-t', fa.hazard_valid_rate,
                 nH ? ('based on ' + nH + ' hazard votes') : 'pending feedback', '%');
            // M2: compute disagree_rate from counts
            const hc = fa.hazard_counts || {};
            const dis_rate = (nH > 0) ? Math.round(100.0 * (hc.disagree || 0) / nH * 10) / 10 : null;
            setM('ds-m2-v','ds-m2-t', dis_rate,
                 nH ? ('based on ' + nH + ' hazard votes') : 'pending feedback', '%');
            setM('ds-m3-v','ds-m3-t', fa.priority_correct_rate,
                 nP ? ('based on ' + nP + ' priority votes') : 'pending feedback', '%');
            setM('ds-m4-v','ds-m4-t', fa.safety_floor_precision,
                 (fa.safety_floor_n || 0) ? ('n=' + fa.safety_floor_n + ' floor cases with feedback') : 'no floor cases voted yet', '%');
            setM('ds-m5-v','ds-m5-t', fa.llm_uplift_precision,
                 (fa.llm_uplift_n || 0) ? ('n=' + fa.llm_uplift_n + ' LLM-flip cases with feedback') : 'no LLM-flip cases voted yet', '%');
            // 6-9: uplift
            setM('ds-m6-v','ds-m6-t', upl.rule_tree_eq_final_rate, 'higher = agents added nothing', '%');
            setM('ds-m7-v','ds-m7-t', upl.floor_fire_rate, 'how often the floor caught the LLM', '%');
            setM('ds-m8-v','ds-m8-t', upl.llm_flip_rate, 'LLM changed rule_tree verdict', '%');
            setM('ds-m9-v','ds-m9-t', upl.llm_clean_wins, 'flip + no floor (LLM standalone value)', '');
            // 10-13: cost
            const mIn = cx.tokens_in_mean, pIn = cx.tokens_in_p95;
            const mOut = cx.tokens_out_mean, pOut = cx.tokens_out_p95;
            document.getElementById('ds-m10-v').textContent = (mIn !== undefined)
              ? (Math.round(mIn) + ' / ' + Math.round(mOut)) : '-';
            document.getElementById('ds-m10-t').textContent = (pIn !== undefined)
              ? ('p95: ' + pIn + ' in / ' + pOut + ' out') : 'pending data';
            setM('ds-m11-v','ds-m11-t', (cx.cost_per_case_usd === undefined) ? null : cx.cost_per_case_usd,
                 'self-hosted $0', '');
            document.getElementById('ds-m11-v').textContent = (cx.cost_per_case_usd !== undefined)
              ? ('$' + (cx.cost_per_case_usd).toFixed(4)) : '-';
            document.getElementById('ds-m12-v').textContent = (cx.wall_p50_s !== undefined)
              ? (cx.wall_p50_s + ' / ' + cx.wall_p95_s) : '-';
            document.getElementById('ds-m12-t').textContent = 'per-case latency (s)';
            setM('ds-m13-v','ds-m13-t', (cx.llm_calls_per_case_mean === undefined) ? null : cx.llm_calls_per_case_mean,
                 'mean over the window', '');
            // 14-15: justified histogram
            const hist = d.justified_histogram || {};
            const histWrap = document.getElementById('ds-hist');
            if (histWrap) {
              const total = Object.values(hist).reduce((a,b) => a+b, 0);
              if (!total) {
                histWrap.innerHTML = '<div class="ds-hist-note">pending data</div>';
              } else {
                const maxV = Math.max(...Object.values(hist));
                let bars = '';
                for (let k = 0; k <= 6; k++) {
                  const v = hist[k] || 0;
                  const h = maxV ? Math.round(80 * v / maxV) : 0;
                  bars += '<div class="ds-bar"><div class="ds-bar-val">' + v + '</div>' +
                          '<div class="ds-bar-fill" style="height:' + h + 'px"></div>' +
                          '<div class="ds-bar-lbl">' + k + ' of 6</div></div>';
                }
                histWrap.innerHTML = bars;
              }
            }
            // 16: confusion matrix + 17: per-class precision
            const confWrap = document.getElementById('ds-conf-wrap');
            if (confWrap) {
              const rows = ds.confusion_rows || [];
              const cols = ds.confusion_cols || [];
              const mat = ds.confusion_matrix || [];
              const perCls = ds.per_class_precision || {};
              const totalFb = rows.reduce((a, r) => a + ((perCls[r] || {}).n || 0), 0);
              if (!totalFb) {
                confWrap.innerHTML = '<div class="ds-conf-note">pending feedback</div>';
              } else {
                let h = '<table class="ds-conf-table"><thead><tr><th>agent said \\ worker</th>';
                for (const c of cols) h += '<th>' + c.replace(/_/g,' ') + '</th>';
                h += '<th>total</th><th>M17 precision</th></tr></thead><tbody>';
                for (let i = 0; i < rows.length; i++) {
                  const r = rows[i];
                  h += '<tr><td class="ds-row-hdr">' + r.replace(/_/g,' ') + '</td>';
                  for (let j = 0; j < cols.length; j++) {
                    const cls = j === 0 ? 'ds-cell-agree' : (j === 1 ? 'ds-cell-dis' : 'ds-cell-nr');
                    h += '<td class="' + cls + '">' + ((mat[i] || [])[j] || 0) + '</td>';
                  }
                  const pc = perCls[r] || {};
                  h += '<td>' + (pc.n || 0) + '</td>';
                  h += '<td class="ds-cell-precision">' + (pc.precision_pct !== null && pc.precision_pct !== undefined
                        ? pc.precision_pct + '%' : 'n/a') + '</td>';
                  h += '</tr>';
                }
                h += '</tbody></table>';
                confWrap.innerHTML = h;
              }
            }
            // M18: macro / weighted precision
            const pm = ds.precision_macro_pct, pw = ds.precision_weighted_pct;
            document.getElementById('ds-m18-v').textContent = (pm !== null && pm !== undefined)
              ? (pm + '% / ' + (pw === null ? 'n/a' : pw + '%')) : '-';
            document.getElementById('ds-m18-t').textContent = (pm !== null && pm !== undefined)
              ? 'macro (equal class weight) / weighted (by n)' : 'pending feedback';
            // M19: ROC-AUC
            const auc = ds.roc_auc, aucN = ds.roc_auc_n || 0;
            setM('ds-m19-v','ds-m19-t', auc,
                 aucN ? ('based on ' + aucN + ' feedback samples') : 'pending feedback', '');
            // M20: grounding score
            const gs = ds.grounding_score_pct, gn = ds.grounding_n || 0;
            setM('ds-m20-v','ds-m20-t', gs,
                 gn ? ('based on ' + gn + ' cited cases') : 'no citations yet', '%');
          } catch (e) { /* silent */ }
        }
        refreshDsMetrics();
        setInterval(refreshDsMetrics, 30000);
        setInterval(refreshVotes, 30000);
        // Silent background refresh: re-fetch the page HTML, swap in the
        // KPIs + table + timeline sections only, preserve the user's active
        // tab, selected row, and scroll position so the page never jumps.
        async function refresh() {
          try {
            const html = await fetch(location.href, {cache: 'no-store'}).then(r => r.text());
            const doc = new DOMParser().parseFromString(html, 'text/html');
            const parts = ['.kpi-strip', 'tbody#rows'];
            for (const sel of parts) {
              const next = doc.querySelector(sel);
              const cur = document.querySelector(sel);
              if (next && cur) cur.replaceWith(next);
            }
            const dot = document.getElementById('refresh-dot');
            if (dot) dot.textContent = '· refreshed ' + new Date().toLocaleTimeString();
            // Re-wire row clicks on the new tbody.
            const newRows = document.querySelectorAll('.cases tr.row');
            const selId = (CASES[0] || {}).id;
            newRows.forEach(r => r.addEventListener('click', () => pick(+r.dataset.idx)));
            bindFbButtons(document);
          } catch (e) { /* silent */ }
        }
        setInterval(refresh, 30000);
        setInterval(refreshAgentValue, 30000);
        """.replace("__CASES__", cases_json)

        # Round 23 iter 5 (4.11.2026): the D Classes tab. The owner
        # asked for a reference grid - one clean exemplar per class
        # (D00 / D10 / D20 / D40) with name and image, grid layout, no
        # scroll. The exemplar pic_ids are picked by looking up the
        # catalog for a single-class labelled sample of each class.
        with sqlite3.connect("/var/lib/roadguard/roadguard.db") as _dc_con:
            def _exemplar(cls_code: str) -> int | None:
                r = _dc_con.execute(
                    "SELECT pic_id FROM dataset_images WHERE classes = ? AND n_labels = 1 "
                    "ORDER BY pic_id LIMIT 1",
                    (cls_code,),
                ).fetchone()
                return int(r[0]) if r else None
            _dc_pics = {c: _exemplar(c) for c in ("D00", "D10", "D20", "D40")}
        _DC_DESCRIPTIONS = {
            "D00": ("D00 Longitudinal crack",
                    "A single crack running along the direction of travel. "
                    "Often appears near the wheel path or lane line."),
            "D10": ("D10 Transverse crack",
                    "A crack running across the direction of travel, roughly "
                    "perpendicular to the road axis."),
            "D20": ("D20 Alligator crack",
                    "A network of interconnected cracks forming irregular polygons, "
                    "resembling alligator skin. Indicates structural fatigue."),
            "D40": ("D40 Pothole",
                    "A bowl-shaped depression in the surface. Varies in diameter "
                    "and depth; may have sharp or broken edges."),
        }
        dc_cards: list[str] = []
        for code in ("D00", "D10", "D20", "D40"):
            name, desc = _DC_DESCRIPTIONS[code]
            pic_id = _dc_pics.get(code)
            img_html = (
                f"<img class='dc-img' src='/roadguard/catalog_image/{pic_id}' "
                f"alt='{name}' onerror=\"this.style.display='none';\">"
                if pic_id else
                "<div class='dc-img dc-img-missing'>no exemplar in catalog</div>"
            )
            dc_cards.append(
                f"<div class='dc-card'>"
                f"<div class='dc-code'>{html.escape(code)}</div>"
                f"<div class='dc-name'>{html.escape(name)}</div>"
                f"{img_html}"
                f"<div class='dc-desc'>{html.escape(desc)}</div>"
                f"</div>"
            )
        d_classes_html = (
            "<div class='dc-wrap'>"
            "<h3 class='dc-h'>D-class reference</h3>"
            "<p class='dc-sub'>Four examples from the labelled catalog, one per class. "
            "Use as a side-by-side reference when voting on the hazard the agents saw.</p>"
            f"<div class='dc-grid'>{''.join(dc_cards)}</div>"
            "</div>"
        )

        # Round 23 (4.11.2026): the "Agent value" tab. Four top cards for
        # the rule-tree agreement + the three worker-verdict counters,
        # then a per-agent panel that lists each of the six agents with
        # its runs, justified count and the behaviour counters pulled
        # from _aggregate_agent_metrics. Everything but the per-agent
        # panel updates live from /stats/feedback on every click.
        per_agent_rows = ""
        _PER_AGENT_METRICS = [
            ("sanitize", [
                ("location_scrubbed", "PII scrubbed"),
                ("ocr_suspicious", "OCR injection blocked"),
                ("reporter_text_suspicious", "Suspicious text filtered"),
            ]),
            ("signal_extractor", [
                ("has_text", "Cases with reporter text"),
                ("llm_used", "LLM calls run"),
            ]),
            ("supervisor", [
                ("concern_input_ambiguous", "Flagged input ambiguous"),
                ("concern_budget_tight", "Flagged budget tight"),
                ("concern_none", "No concern raised"),
            ]),
            ("profile_builder", [
                ("catalog_hit", "Catalogue hits"),
                ("class_matches_detector", "Agent class matches detector"),
                ("llm_used", "LLM calls run"),
            ]),
            ("rag_retriever", [
                ("coverage_direct_match", "Direct match"),
                ("llm_used", "LLM calls run"),
                ("total_passages_returned", "Passages returned"),
            ]),
            ("decision", [
                ("rule_based", "Rule-based verdicts"),
                ("llm_based", "LLM-based verdicts"),
                ("contradiction_floor", "Floors fired"),
            ]),
        ]
        for sp, metrics in _PER_AGENT_METRICS:
            block = aggregates["per_agent"].get(sp, {}) if isinstance(aggregates.get("per_agent"), dict) else {}
            aj = (aggregates.get("agents_justified") or {}).get(sp, {}) if isinstance(aggregates.get("agents_justified"), dict) else {}
            justified = aj.get("justified", 0) if isinstance(aj, dict) else 0
            runs = aj.get("runs", 0) if isinstance(aj, dict) else 0
            metric_cells = "".join(
                f"<div class='ag-metric'><span class='ag-m-k'>{html.escape(label)}</span>"
                f"<span class='ag-m-v'>{block.get(key, 0)}</span></div>"
                for key, label in metrics
            )
            per_agent_rows += (
                f"<div class='ag-card'>"
                f"<div class='ag-head'><span class='ag-name'>{sp}</span>"
                f"<span class='ag-just'>{justified} / {runs} justified</span></div>"
                f"<div class='ag-metrics'>{metric_cells}</div>"
                f"</div>"
            )
        aggregates_json = {
            "per_agent": aggregates.get("per_agent"),
            "agents_justified": aggregates.get("agents_justified"),
            "cost_total": aggregates.get("cost_total"),
        }
        # The window has to be set up for the aggregates call just above
        # to have content for this snapshot: use the same 24h window the
        # digest already reads.
        agent_value_html = (
            "<div class='av-wrap'>"
            "<h3 class='av-h'>How much is the agent worth right now</h3>"
            "<div class='av-grid'>"
            f"<div class='av-card {'ok' if correct_pct >= 85 else 'fail'}'>"
            f"<div class='av-v'>{correct_pct:.0f}%</div>"
            f"<div class='av-l'>Agreement with the rule tree</div>"
            f"<div class='av-t'>target 85%</div></div>"
            "<div class='av-card' id='av-agree-card'>"
            "<div class='av-v' id='av-agree-v'>0</div>"
            "<div class='av-l'>Cases worker agreed with</div>"
            "<div class='av-t' id='av-agree-t'>no votes yet</div></div>"
            "<div class='av-card' id='av-dis-card'>"
            "<div class='av-v' id='av-dis-v'>0</div>"
            "<div class='av-l'>Cases worker disagreed with</div>"
            "<div class='av-t' id='av-dis-t'>no votes yet</div></div>"
            "<div class='av-card' id='av-nr-card'>"
            "<div class='av-v' id='av-nr-v'>0</div>"
            "<div class='av-l'>Cases worker marked not relevant</div>"
            "<div class='av-t' id='av-nr-t'>no votes yet</div></div>"
            "</div>"
            # Round 26 (5.11.2026): the 20 data-science metrics the owner
            # asked for. Five panels, all inside Agent value (no new tab).
            # Each number renders 'pending' when there is not enough
            # feedback yet, so a sparse day shows honestly and not a zero.
            "<h3 class='av-h' style='margin-top:22px'>Feedback agreement (1-5)</h3>"
            "<div class='av-grid'>"
            "<div class='av-card' id='ds-m1-card'>"
            "<div class='av-v' id='ds-m1-v'>-</div>"
            "<div class='av-l'>M1 Hazard-valid rate</div>"
            "<div class='av-t' id='ds-m1-t'>pending feedback</div></div>"
            "<div class='av-card' id='ds-m2-card'>"
            "<div class='av-v' id='ds-m2-v'>-</div>"
            "<div class='av-l'>M2 Hazard-disagree rate</div>"
            "<div class='av-t' id='ds-m2-t'>pending feedback</div></div>"
            "<div class='av-card' id='ds-m3-card'>"
            "<div class='av-v' id='ds-m3-v'>-</div>"
            "<div class='av-l'>M3 Priority-correct rate</div>"
            "<div class='av-t' id='ds-m3-t'>pending feedback</div></div>"
            "<div class='av-card' id='ds-m4-card'>"
            "<div class='av-v' id='ds-m4-v'>-</div>"
            "<div class='av-l'>M4 Safety-floor precision</div>"
            "<div class='av-t' id='ds-m4-t'>pending feedback</div></div>"
            "<div class='av-card' id='ds-m5-card'>"
            "<div class='av-v' id='ds-m5-v'>-</div>"
            "<div class='av-l'>M5 LLM-uplift precision</div>"
            "<div class='av-t' id='ds-m5-t'>pending feedback</div></div>"
            "</div>"
            "<h3 class='av-h' style='margin-top:22px'>Agent uplift over the rule tree (6-9)</h3>"
            "<div class='av-grid'>"
            "<div class='av-card' id='ds-m6-card'>"
            "<div class='av-v' id='ds-m6-v'>-</div>"
            "<div class='av-l'>M6 rule_tree == final</div>"
            "<div class='av-t' id='ds-m6-t'>higher = agents added nothing</div></div>"
            "<div class='av-card' id='ds-m7-card'>"
            "<div class='av-v' id='ds-m7-v'>-</div>"
            "<div class='av-l'>M7 Safety-floor fire rate</div>"
            "<div class='av-t' id='ds-m7-t'>how often the floor caught the LLM</div></div>"
            "<div class='av-card' id='ds-m8-card'>"
            "<div class='av-v' id='ds-m8-v'>-</div>"
            "<div class='av-l'>M8 LLM flip rate</div>"
            "<div class='av-t' id='ds-m8-t'>LLM changed rule_tree verdict</div></div>"
            "<div class='av-card' id='ds-m9-card'>"
            "<div class='av-v' id='ds-m9-v'>-</div>"
            "<div class='av-l'>M9 LLM clean wins</div>"
            "<div class='av-t' id='ds-m9-t'>flip + no floor (LLM standalone value)</div></div>"
            "</div>"
            "<h3 class='av-h' style='margin-top:22px'>Cost and latency (10-13)</h3>"
            "<div class='av-grid'>"
            "<div class='av-card' id='ds-m10-card'>"
            "<div class='av-v' id='ds-m10-v'>-</div>"
            "<div class='av-l'>M10 Tokens in / out (mean)</div>"
            "<div class='av-t' id='ds-m10-t'>p95 shown as well</div></div>"
            "<div class='av-card' id='ds-m11-card'>"
            "<div class='av-v' id='ds-m11-v'>-</div>"
            "<div class='av-l'>M11 $ per case (commercial equiv)</div>"
            "<div class='av-t' id='ds-m11-t'>self-hosted $0</div></div>"
            "<div class='av-card' id='ds-m12-card'>"
            "<div class='av-v' id='ds-m12-v'>-</div>"
            "<div class='av-l'>M12 Wall p50 / p95 (s)</div>"
            "<div class='av-t' id='ds-m12-t'>per-case latency</div></div>"
            "<div class='av-card' id='ds-m13-card'>"
            "<div class='av-v' id='ds-m13-v'>-</div>"
            "<div class='av-l'>M13 LLM calls per case</div>"
            "<div class='av-t' id='ds-m13-t'>mean over the window</div></div>"
            "</div>"
            "<h3 class='av-h' style='margin-top:22px'>Per-agent justified histogram (14-15)</h3>"
            "<div id='ds-hist' class='ds-hist'>"
            "<div class='ds-hist-note'>pending data</div></div>"
            "<h3 class='av-h' style='margin-top:22px'>Confusion matrix and class-wise precision (16-20)</h3>"
            "<div id='ds-conf-wrap' class='ds-conf-wrap'>"
            "<div class='ds-conf-note'>pending feedback</div></div>"
            "<div class='av-grid' style='margin-top:12px'>"
            "<div class='av-card' id='ds-m18-card'>"
            "<div class='av-v' id='ds-m18-v'>-</div>"
            "<div class='av-l'>M18 Precision macro / weighted</div>"
            "<div class='av-t' id='ds-m18-t'>pending feedback</div></div>"
            "<div class='av-card' id='ds-m19-card'>"
            "<div class='av-v' id='ds-m19-v'>-</div>"
            "<div class='av-l'>M19 ROC-AUC (priority vs agree)</div>"
            "<div class='av-t' id='ds-m19-t'>pending feedback</div></div>"
            "<div class='av-card' id='ds-m20-card'>"
            "<div class='av-v' id='ds-m20-v'>-</div>"
            "<div class='av-l'>M20 Grounding score</div>"
            "<div class='av-t' id='ds-m20-t'>citations in retrieved passages</div></div>"
            "</div>"
            "<h3 class='av-h' style='margin-top:22px'>Each agent and its metrics (last 24h)</h3>"
            f"<div class='ag-grid'>{per_agent_rows}</div>"
            "<h3 class='av-h' style='margin-top:18px'>Cases the worker disagreed with (last 24h)</h3>"
            "<div id='av-disagree-list' class='av-overrides'>No disagreements yet.</div>"
            "<h3 class='av-h' style='margin-top:18px'>Cases the worker marked 'not relevant' (last 24h)</h3>"
            "<div id='av-nr-list' class='av-overrides'>No 'not relevant' flags yet.</div>"
            "</div>"
        )
        # Round 23 iter 4 (4.11.2026): rule tree expanded. The owner saw
        # the earlier version as cut off at the edges and missing the
        # depth of the real _decide_rule_based tree. This one draws the
        # catalog-detector-contradiction floor, the evidence-declined
        # branch, the YOLO-confidence band gate on the severity side,
        # and the LLM-failed safety net, in one SVG whose width scales
        # to the viewport (max-width, overflow-x: auto on the canvas).
        rule_tree_html = """
        <div class='rt-wrap'>
          <h3 class='rt-h'>Rule tree the agent is compared against</h3>
          <p class='rt-sub'>Deterministic baseline. Every agent verdict is scored against the leaf this tree would have picked. Use the arrows below to pan the whole tree left / right.</p>
          <div class='rt-scrollbar-wrap'>
            <button class='rt-scroll-btn' id='rt-scroll-left' aria-label='scroll left'>&larr;</button>
            <button class='rt-scroll-btn' id='rt-scroll-right' aria-label='scroll right'>&rarr;</button>
            <span class='rt-scroll-hint'>tree is wider than the screen; use the arrows or drag the scrollbar</span>
          </div>
          <div class='rt-canvas' id='rt-canvas'>
            <svg viewBox='0 0 1800 1150' xmlns='http://www.w3.org/2000/svg' class='rt-svg' preserveAspectRatio='xMinYMin meet'>
              <style>
                .rt-svg .n rect{fill:#fff;stroke:#334155;stroke-width:1.5;rx:8;ry:8}
                .rt-svg .n text{font:600 13px -apple-system,Segoe UI,Roboto,sans-serif;fill:#111827}
                .rt-svg .leaf text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
                .rt-svg .edge{stroke:#334155;stroke-width:1.5;fill:none}
                .rt-svg .edge-lbl{font:600 11px -apple-system,Segoe UI,Roboto,sans-serif;fill:#334155;text-anchor:middle}
                .rt-svg .divider{stroke:#cbd5e1;stroke-dasharray:4,4;stroke-width:1}
                .rt-svg .section-lbl{font:700 11px -apple-system,Segoe UI,Roboto,sans-serif;fill:#6b7280;letter-spacing:0.04em;text-anchor:middle}
              </style>

              <!-- Root -> sha256 -->
              <path class='edge' d='M900,62 V100'/>

              <!-- sha256 match? -> known branch (yes) and -> new-photo branch (no), elbow at y=180 -->
              <path class='edge' d='M900,152 V180 H300 V210'/>
              <path class='edge' d='M900,152 V180 H1200 V210'/>
              <text class='edge-lbl' x='600' y='172'>yes (same file)</text>
              <text class='edge-lbl' x='1050' y='172'>no (new photo)</text>

              <!-- known branch: detector class matches catalog labels? -->
              <path class='edge' d='M300,262 V310'/>
              <text class='edge-lbl' x='300' y='290'>catalogue agrees with detector?</text>
              <path class='edge' d='M300,362 V390 H150 V420'/>
              <path class='edge' d='M300,362 V390 H450 V420'/>
              <text class='edge-lbl' x='210' y='383'>yes</text>
              <text class='edge-lbl' x='390' y='383'>no (floor)</text>

              <!-- new-photo branch: phash? -->
              <path class='edge' d='M1200,262 V310'/>

              <!-- phash yes -> human_review (phash floor); phash no -> evidence_declined? -->
              <path class='edge' d='M1200,362 V390 H900 V420'/>
              <path class='edge' d='M1200,362 V390 H1500 V420'/>
              <text class='edge-lbl' x='1050' y='383'>yes (class mismatch)</text>
              <text class='edge-lbl' x='1350' y='383'>no</text>

              <!-- evidence_declined? -->
              <path class='edge' d='M1500,472 V510 H1200 V540'/>
              <path class='edge' d='M1500,472 V510 H1700 V540'/>
              <text class='edge-lbl' x='1340' y='503'>yes (declined)</text>
              <text class='edge-lbl' x='1600' y='503'>no (carry on)</text>

              <!-- severity_signal? -->
              <path class='edge' d='M1700,592 V630'/>

              <!-- severity? -> three subtrees -->
              <path class='edge' d='M1700,682 V710 H1430 V740'/>
              <path class='edge' d='M1700,682 V740'/>
              <path class='edge' d='M1700,682 V710 H1980 V740'/>
              <text class='edge-lbl' x='1565' y='703'>severe / recurring</text>
              <text class='edge-lbl' x='1750' y='703'>moderate</text>
              <text class='edge-lbl' x='1900' y='703'>minor</text>

              <!-- severity=moderate: YOLO confidence band? -->
              <path class='edge' d='M1700,792 V820 H1620 V850'/>
              <path class='edge' d='M1700,792 V820 H1780 V850'/>
              <text class='edge-lbl' x='1660' y='813'>low conf</text>
              <text class='edge-lbl' x='1860' y='813'>high conf</text>

              <!-- severity=minor: confidence band? -->
              <path class='edge' d='M1980,792 V820 H1900 V850'/>
              <path class='edge' d='M1980,792 V820 H2060 V850'/>
              <text class='edge-lbl' x='1940' y='813'>low conf</text>
              <text class='edge-lbl' x='2140' y='813'>high conf</text>

              <!-- Divider + safety net -->
              <line class='divider' x1='40' y1='960' x2='1760' y2='960'/>
              <text class='section-lbl' x='900' y='983'>SAFETY NET (applies to any leaf above)</text>

              <!-- LLM failed? -->
              <path class='edge' d='M900,1022 V1050 H650 V1080'/>
              <path class='edge' d='M900,1022 V1050 H1150 V1080'/>
              <text class='edge-lbl' x='760' y='1043'>yes (LLM failed)</text>
              <text class='edge-lbl' x='1050' y='1043'>no (LLM returned)</text>

              <!-- Nodes -->
              <g class='n' transform='translate(800,20)'>
                <rect width='200' height='42'/>
                <text x='100' y='27' text-anchor='middle'>incoming hazard</text>
              </g>
              <g class='n' transform='translate(760,100)'>
                <rect width='280' height='52'/>
                <text x='140' y='22' text-anchor='middle'>sha256 match in catalogue?</text>
                <text x='140' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>same file as a labelled photo?</text>
              </g>

              <!-- Known branch: catalog contradiction -->
              <g class='n' transform='translate(180,210)'>
                <rect width='240' height='52'/>
                <text x='120' y='22' text-anchor='middle'>catalogue agrees?</text>
                <text x='120' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>detector class in catalog labels?</text>
              </g>
              <g class='n leaf' transform='translate(50,420)'>
                <rect width='200' height='42'/>
                <text x='100' y='27' text-anchor='middle'>known_image</text>
              </g>
              <g class='n leaf' transform='translate(350,420)'>
                <rect width='200' height='52'/>
                <text x='100' y='22' text-anchor='middle'>human_review</text>
                <text x='100' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>catalog_detector_contradiction</text>
              </g>

              <!-- New-photo branch: phash -->
              <g class='n' transform='translate(1070,210)'>
                <rect width='260' height='52'/>
                <text x='130' y='22' text-anchor='middle'>phash class mismatch?</text>
                <text x='130' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>neighbour label disagrees with detector</text>
              </g>
              <g class='n leaf' transform='translate(800,420)'>
                <rect width='200' height='52'/>
                <text x='100' y='22' text-anchor='middle'>human_review</text>
                <text x='100' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>phash_class_mismatch floor</text>
              </g>

              <g class='n' transform='translate(1370,420)'>
                <rect width='260' height='52'/>
                <text x='130' y='22' text-anchor='middle'>evidence_declined?</text>
                <text x='130' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>reporter: "no other photo"</text>
              </g>
              <g class='n leaf' transform='translate(1100,540)'>
                <rect width='200' height='52'/>
                <text x='100' y='22' text-anchor='middle'>human_review</text>
                <text x='100' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>evidence_declined</text>
              </g>

              <g class='n' transform='translate(1570,540)'>
                <rect width='260' height='52'/>
                <text x='130' y='22' text-anchor='middle'>severity_signal?</text>
                <text x='130' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>from area, lane, class, recurrence</text>
              </g>

              <!-- Severity leaves -->
              <g class='n leaf' transform='translate(1330,740)'>
                <rect width='200' height='52'/>
                <text x='100' y='22' text-anchor='middle'>human_review</text>
                <text x='100' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>severe / recurring</text>
              </g>

              <!-- Moderate: conf band split -->
              <g class='n' transform='translate(1600,740)'>
                <rect width='200' height='52'/>
                <text x='100' y='22' text-anchor='middle'>YOLO confidence?</text>
                <text x='100' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>low or high band</text>
              </g>
              <g class='n leaf' transform='translate(1520,850)'>
                <rect width='200' height='52'/>
                <text x='100' y='22' text-anchor='middle'>needs_more_evidence</text>
                <text x='100' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>moderate + low conf</text>
              </g>
              <g class='n leaf' transform='translate(1700,850)'>
                <rect width='160' height='52'/>
                <text x='80' y='22' text-anchor='middle'>standard</text>
                <text x='80' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>moderate + high conf</text>
              </g>

              <!-- Minor: conf band split -->
              <g class='n' transform='translate(1880,740)'>
                <rect width='200' height='52'/>
                <text x='100' y='22' text-anchor='middle'>YOLO confidence?</text>
                <text x='100' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>low or high band</text>
              </g>
              <g class='n leaf' transform='translate(1800,850)'>
                <rect width='200' height='52'/>
                <text x='100' y='22' text-anchor='middle'>needs_more_evidence</text>
                <text x='100' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>minor + low conf</text>
              </g>
              <g class='n leaf' transform='translate(1980,850)'>
                <rect width='160' height='52'/>
                <text x='80' y='22' text-anchor='middle'>standard</text>
                <text x='80' y='40' text-anchor='middle' font-size='10' fill='#6b7280'>minor + high conf</text>
              </g>

              <!-- Safety net -->
              <g class='n' transform='translate(780,1020)'>
                <rect width='240' height='42'/>
                <text x='120' y='27' text-anchor='middle'>LLM failed?</text>
              </g>
              <g class='n leaf' transform='translate(550,1080)'>
                <rect width='200' height='42'/>
                <text x='100' y='27' text-anchor='middle'>fall back to rule tree</text>
              </g>
              <g class='n leaf' transform='translate(1050,1080)'>
                <rect width='200' height='42'/>
                <text x='100' y='27' text-anchor='middle'>keep agent verdict</text>
              </g>
            </svg>
          </div>
        </div>
        """
        html_doc = (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>RoadGuard LIVE - {NOW.strftime('%Y-%m-%d %H:%M UTC')}</title>"
            f"<style>{style}</style>"
            "</head><body>"
            f"{header}"
            "<section id='pane-cases' class='tab-pane active'>"
            f"{kpi_strip}"
            f"{clusters_html}"
            "<div class='split'>"
            f"<div>{table_html}</div>"
            "<aside class='detail' id='detail'>"
            "<div class='d-empty'>Pick a case on the left.</div>"
            "</aside>"
            "</div></section>"
            f"<section id='pane-agentvalue' class='tab-pane'>{agent_value_html}</section>"
            f"<section id='pane-dclasses' class='tab-pane'>{d_classes_html}</section>"
            f"<section id='pane-ruletree' class='tab-pane'>{rule_tree_html}</section>"
            f"<script>{script}</script>"
            "</body></html>"
        )
        return html_doc

    @app.get("/stats/public")
    def stats_public() -> dict:
        """Aggregate public counters for the landing stats board.
        No PII, no case ids - the kind of numbers a city is happy to show
        the public: total reports, open cases, closed-this-week, median
        resolution time, and the breakdown by decision + defect class."""
        import sqlite3 as _sqlite3
        db_path = "/var/lib/roadguard/roadguard.db"
        try:
            con = _sqlite3.connect(db_path)
            now = time.time()
            week_ago = now - 7 * 86400
            total = con.execute("SELECT COUNT(*) FROM hazards").fetchone()[0]
            by_dec: dict[str, int] = {}
            for row in con.execute("SELECT agent_decision, COUNT(*) FROM hazards GROUP BY agent_decision"):
                by_dec[(row[0] or "unknown")] = row[1]
            by_class: dict[str, int] = {}
            for row in con.execute("SELECT class, COUNT(*) FROM hazards GROUP BY class"):
                by_class[(row[0] or "unknown")] = row[1]
            open_now = con.execute("SELECT COUNT(*) FROM hazards WHERE status='open'").fetchone()[0]
            closed_week = con.execute(
                "SELECT COUNT(*) FROM hazards WHERE status != 'open' AND reported_at >= ?",
                (week_ago,),
            ).fetchone()[0]
            resolve_rows = con.execute(
                "SELECT (a.decided_at - h.reported_at) / 3600.0 "
                "FROM hazards h JOIN approvals a ON a.hazard_id = h.id "
                "WHERE a.decided_at IS NOT NULL AND a.decided_at > h.reported_at"
            ).fetchall()
            median_h: float | None = None
            if resolve_rows:
                vals = sorted([r[0] for r in resolve_rows if r[0] is not None])
                if vals:
                    median_h = round(vals[len(vals) // 2], 1)
            con.close()
            return {
                "total_reports": total,
                "open_now": open_now,
                "closed_this_week": closed_week,
                "by_decision": by_dec,
                "by_class": by_class,
                "median_resolve_hours": median_h,
                "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"stats_failed: {exc}")

    @app.get("/stats/agents")
    def stats_agents(request: Request, window_hours: int = 24) -> dict:
        """Per-agent quality aggregates over the last ``window_hours`` of
        hazards. Private: gated to Tailscale / loopback like /digest.

        Each counter is a raw run count. The review called out that
        latency + tokens alone miss whether an agent actually produced a
        useful signal, so this returns counts of the quality signals
        the trace already records: dataset catalogue hits for
        profile_builder, coverage=direct_match for rag_retriever,
        rule_based vs llm_based for decision, grounding failures and
        catalog-vs-detector contradiction floor fires, and the global
        llm_fallback rate (any node whose LLM calls all failed).

        Clients compute percentages client-side by dividing by
        total_hazards (or rag_retriever.runs for RAG counters).
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        try:
            hrs = max(1, min(720, int(window_hours)))
        except (TypeError, ValueError):
            hrs = 24
        settings = get_settings()
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(settings.roadguard_db_path)
        con.row_factory = _sqlite3.Row
        now = time.time()
        try:
            rows = con.execute(
                "SELECT id, class, agent_decision, agent_trace_json, agent_value_json "
                "FROM hazards WHERE reported_at >= ?",
                (now - hrs * 3600,),
            ).fetchall()
        finally:
            con.close()
        aggregates = _aggregate_agent_metrics(rows)
        return {
            "window_hours": hrs,
            "total_hazards": len(rows),
            "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            **aggregates,
        }

    _VALID_VERDICTS = {"agree", "disagree", "not_relevant"}
    # Round 23 iter 5 (4.11.2026): a feedback row grades ONE axis. The
    # hazard axis uses agree/disagree/not_relevant (did the agent
    # identify the hazard?); the priority axis uses ok/too_low/too_high
    # (was the agent's urgency level right?). The POST writes both the
    # legacy `agree` flag and the new `kind`+`verdict`.
    _VALID_PRIORITY_VERDICTS = {"ok", "too_low", "too_high"}

    @app.get("/agent_feedback/by_hazard")
    def get_agent_feedback_by_hazard(request: Request, window_hours: int = 24) -> dict:
        """Round 23 iter 5 (4.11.2026): latest verdict per hazard FOR
        EACH kind (hazard, priority). The digest JS reads this on page
        load so a case the worker voted on - hazard side, priority side
        or both - shows its latest marks and lets the worker change them.
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        try:
            hrs = max(1, min(720, int(window_hours)))
        except (TypeError, ValueError):
            hrs = 24
        settings = get_settings()
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(settings.roadguard_db_path)
        con.row_factory = _sqlite3.Row
        cutoff = time.time() - hrs * 3600
        try:
            rows = con.execute(
                "WITH ranked AS ("
                "  SELECT hazard_id, kind, verdict, agree, note, created_at, "
                "         ROW_NUMBER() OVER (PARTITION BY hazard_id, kind ORDER BY created_at DESC) rn "
                "  FROM agent_feedback WHERE created_at >= ?"
                ") SELECT hazard_id, kind, verdict, agree, note, created_at "
                "FROM ranked WHERE rn = 1",
                (cutoff,),
            ).fetchall()
        finally:
            con.close()
        out: dict = {}
        for r in rows:
            v = (r["verdict"] or "").strip().lower() if "verdict" in r.keys() else ""
            if not v:
                v = "agree" if r["agree"] else "disagree"
            k = (r["kind"] or "hazard").strip().lower() if "kind" in r.keys() else "hazard"
            entry = out.setdefault(r["hazard_id"], {})
            entry[k] = {"verdict": v, "note": r["note"] or "", "at": r["created_at"]}
        return {"window_hours": hrs, "by_hazard": out}

    @app.post("/agent_feedback/{hazard_id}")
    def post_agent_feedback(hazard_id: str, body: dict, request: Request) -> dict:
        """Round 23 (4.11.2026): the worker's three-state verdict on the
        agent's call for one hazard. Writes one row to agent_feedback
        and returns the running tally for this hazard. Private: gated to
        Tailscale / loopback like the digest page that calls it.
        Body: {"verdict": "agree" | "disagree" | "not_relevant",
               "note": str?}.
        For back-compat an older client that still sends {"agree": bool}
        is accepted (true -> agree, false -> disagree).
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        verdict = str(body.get("verdict") or "").strip().lower()
        kind = str(body.get("kind") or "hazard").strip().lower()
        if kind not in {"hazard", "priority"}:
            raise HTTPException(status_code=400,
                                detail="kind_must_be_hazard_or_priority")
        if not verdict:
            # back-compat: {"agree": true/false} is always hazard
            if "agree" in body:
                verdict = "agree" if bool(body.get("agree")) else "disagree"
                kind = "hazard"
        valid = _VALID_VERDICTS if kind == "hazard" else _VALID_PRIORITY_VERDICTS
        if verdict not in valid:
            raise HTTPException(
                status_code=400,
                detail=(f"verdict_must_be_one_of_{'_'.join(sorted(valid))}")
            )
        note = str(body.get("note") or "")[:400]
        import sqlite3 as _sqlite3
        import uuid as _uuid
        settings = get_settings()
        con = _sqlite3.connect(settings.roadguard_db_path)
        con.row_factory = _sqlite3.Row
        try:
            fid = "fb_" + _uuid.uuid4().hex[:12]
            con.execute(
                "INSERT INTO agent_feedback (id, hazard_id, created_at, worker_id, agree, verdict, note, kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (fid, hazard_id, time.time(), "", 1 if verdict == "agree" else 0, verdict, note, kind),
            )
            con.commit()
            row = con.execute(
                "SELECT "
                "  SUM(CASE WHEN kind='hazard' AND verdict='agree' THEN 1 ELSE 0 END) AS haz_agree, "
                "  SUM(CASE WHEN kind='hazard' AND verdict='disagree' THEN 1 ELSE 0 END) AS haz_disagree, "
                "  SUM(CASE WHEN kind='hazard' AND verdict='not_relevant' THEN 1 ELSE 0 END) AS haz_nr, "
                "  SUM(CASE WHEN kind='priority' AND verdict='ok' THEN 1 ELSE 0 END) AS pri_ok, "
                "  SUM(CASE WHEN kind='priority' AND verdict='too_low' THEN 1 ELSE 0 END) AS pri_lo, "
                "  SUM(CASE WHEN kind='priority' AND verdict='too_high' THEN 1 ELSE 0 END) AS pri_hi, "
                "  COUNT(*) AS total "
                "FROM agent_feedback WHERE hazard_id = ?",
                (hazard_id,),
            ).fetchone()
        finally:
            con.close()
        return {
            "id": fid,
            "hazard_id": hazard_id,
            "kind": kind,
            "verdict": verdict,
            "hazard": {
                "agree": int(row["haz_agree"] or 0),
                "disagree": int(row["haz_disagree"] or 0),
                "not_relevant": int(row["haz_nr"] or 0),
            },
            "priority": {
                "ok": int(row["pri_ok"] or 0),
                "too_low": int(row["pri_lo"] or 0),
                "too_high": int(row["pri_hi"] or 0),
            },
            "feedback_total": int(row["total"] or 0),
        }

    @app.get("/stats/feedback")
    def stats_feedback(request: Request, window_hours: int = 24) -> dict:
        """Round 23 (4.11.2026) iter 3: aggregate the worker's verdicts
        by DISTINCT HAZARD, not by click count. A hazard the worker
        clicked three times (Disagree -> Agree -> Not relevant) is
        counted once, with its LATEST verdict. The 'Agent value' tab's
        counters therefore read as 'how many cases the worker labeled
        X', which is what the owner asked for.
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        try:
            hrs = max(1, min(720, int(window_hours)))
        except (TypeError, ValueError):
            hrs = 24
        settings = get_settings()
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(settings.roadguard_db_path)
        con.row_factory = _sqlite3.Row
        cutoff = time.time() - hrs * 3600
        try:
            # Round 23 iter 5: latest row per (hazard, kind). Hazard and
            # priority are counted on their own axes.
            rows = con.execute(
                "WITH ranked AS ("
                "  SELECT f.hazard_id, f.kind, f.agree, f.verdict, f.note, f.created_at, "
                "         h.class, h.agent_decision, "
                "         ROW_NUMBER() OVER (PARTITION BY f.hazard_id, f.kind ORDER BY f.created_at DESC) rn "
                "  FROM agent_feedback f "
                "  LEFT JOIN hazards h ON h.id = f.hazard_id "
                "  WHERE f.created_at >= ?"
                ") SELECT hazard_id, kind, agree, verdict, note, class, agent_decision "
                "FROM ranked WHERE rn = 1 "
                "ORDER BY hazard_id",
                (cutoff,),
            ).fetchall()
        finally:
            con.close()
        def _verdict(r) -> str:
            v = str(r["verdict"] or "").strip().lower() if "verdict" in r.keys() else ""
            if v:
                return v
            return "agree" if r["agree"] else "disagree"
        hazard_counters = {"agree": 0, "disagree": 0, "not_relevant": 0}
        priority_counters = {"ok": 0, "too_low": 0, "too_high": 0}
        hazard_cases = set()
        priority_cases = set()
        disagrees: list[dict] = []
        not_relevant: list[dict] = []
        pri_wrong: list[dict] = []
        for r in rows:
            v = _verdict(r)
            kind = (r["kind"] or "hazard").strip().lower() if "kind" in r.keys() else "hazard"
            if kind == "hazard":
                hazard_cases.add(r["hazard_id"])
                hazard_counters[v] = hazard_counters.get(v, 0) + 1
                if v == "disagree":
                    disagrees.append({"hazard_id": r["hazard_id"], "class": r["class"],
                                      "decision": r["agent_decision"], "note": r["note"]})
                elif v == "not_relevant":
                    not_relevant.append({"hazard_id": r["hazard_id"], "class": r["class"],
                                         "decision": r["agent_decision"], "note": r["note"]})
            else:
                priority_cases.add(r["hazard_id"])
                priority_counters[v] = priority_counters.get(v, 0) + 1
                if v != "ok":
                    pri_wrong.append({"hazard_id": r["hazard_id"], "class": r["class"],
                                      "decision": r["agent_decision"], "note": r["note"], "verdict": v})
        return {
            "window_hours": hrs,
            "hazard": {
                "total_cases_voted": len(hazard_cases),
                "agree": hazard_counters["agree"],
                "disagree": hazard_counters["disagree"],
                "not_relevant": hazard_counters["not_relevant"],
                "agreement_rate": round(100.0 * hazard_counters["agree"] / len(hazard_cases), 1) if hazard_cases else 0.0,
            },
            "priority": {
                "total_cases_voted": len(priority_cases),
                "ok": priority_counters["ok"],
                "too_low": priority_counters["too_low"],
                "too_high": priority_counters["too_high"],
                "agreement_rate": round(100.0 * priority_counters["ok"] / len(priority_cases), 1) if priority_cases else 0.0,
            },
            "disagrees": disagrees,
            "not_relevants": not_relevant,
            "priority_wrong": pri_wrong,
        }

    _CATALOG_IMAGE_ROOT = "/var/lib/roadguard/dataset/extracted/road_damage_detect"

    @app.get("/catalog_image/{pic_id}")
    def catalog_image(pic_id: int, request: Request):
        """Round 23 iter 5 (4.11.2026): serve one image from the labelled
        catalog so the D Classes tab can show reference exemplars next
        to the worker's live Cases. Private host only, same as the rest
        of the operator surface.
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        from pathlib import Path as _Path
        from fastapi.responses import FileResponse as _FileResponse
        settings = get_settings()
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(settings.roadguard_db_path)
        try:
            row = con.execute(
                "SELECT rel_path FROM dataset_images WHERE pic_id = ?",
                (int(pic_id),),
            ).fetchone()
        finally:
            con.close()
        if not row:
            raise HTTPException(status_code=404, detail="catalog_image_not_found")
        p = _Path(_CATALOG_IMAGE_ROOT) / row[0]
        if not p.is_file():
            raise HTTPException(status_code=404, detail="catalog_image_file_missing")
        return _FileResponse(str(p), media_type="image/jpeg")

    @app.get("/stats/ds_metrics")
    def stats_ds_metrics(request: Request, window_hours: int = 24) -> dict:
        """Round 26 (5.11.2026): the 20 data-science metrics the owner
        asked for, all computed from in-system state (hazards +
        agent_feedback). Returns feedback-agreement, uplift, extended
        cost, justified histogram, confusion matrix, per-class precision,
        ROC-AUC and grounding score. A metric that lacks enough feedback
        is returned as null with a side n= count, so the UI shows
        'pending feedback' rather than a misleading zero.
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        try:
            hrs = max(1, min(720, int(window_hours)))
        except (TypeError, ValueError):
            hrs = 24
        settings = get_settings()
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(settings.roadguard_db_path)
        con.row_factory = _sqlite3.Row
        cutoff = time.time() - hrs * 3600
        try:
            hazard_rows = con.execute(
                "SELECT id, class, confidence, priority, agent_decision, "
                "       agent_value_json, agent_trace_json "
                "  FROM hazards WHERE id LIKE 'hz_%' AND reported_at >= ?",
                (cutoff,),
            ).fetchall()
            fb_rows = con.execute(
                "WITH ranked AS ("
                "  SELECT hazard_id, kind, verdict, agree, created_at, "
                "         ROW_NUMBER() OVER (PARTITION BY hazard_id, kind ORDER BY created_at DESC) rn "
                "  FROM agent_feedback WHERE created_at >= ?"
                ") SELECT hazard_id, kind, verdict, agree "
                "FROM ranked WHERE rn = 1",
                (cutoff,),
            ).fetchall()
        finally:
            con.close()
        block = _compute_ds_metrics_block(hazard_rows, fb_rows)
        block["window_hours"] = hrs
        return block

    @app.get("/stats/kpi")
    def stats_kpi(request: Request, window_hours: int = 24) -> dict:
        """Round 23 (4.11.2026): live KPI board matching the presentation's
        targets table. Returns the four KPIs - correct_action,
        grounded_citations, rag_relevance, median_latency - each with its
        target, current value, pass/fail flag and a plain-language reason.
        Private: gated to Tailscale / loopback like /digest and /stats/
        agents. The /flow page, the /digest KPI strip and the deck all
        read this one source.
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        try:
            hrs = max(1, min(720, int(window_hours)))
        except (TypeError, ValueError):
            hrs = 24
        settings = get_settings()
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(settings.roadguard_db_path)
        con.row_factory = _sqlite3.Row
        now = time.time()
        try:
            rows = con.execute(
                "SELECT id, class, agent_decision, agent_trace_json, agent_value_json "
                "FROM hazards WHERE reported_at >= ?",
                (now - hrs * 3600,),
            ).fetchall()
        finally:
            con.close()
        n = len(rows)
        if n == 0:
            return {"window_hours": hrs, "total_hazards": 0, "kpis": []}
        # Correct action: without ground truth, read as "agreement with
        # the deterministic rule tree". When expected_verdicts.csv is
        # filled the computation moves to that ground truth.
        rule_tree_eq_final = 0
        grounded = 0
        decision_llm_runs = 0
        direct_match_runs = 0
        rag_total_runs = 0
        latencies: list[float] = []
        for r in rows:
            try:
                av = json.loads((r["agent_value_json"] if r["agent_value_json"] else "") or "{}")
            except (TypeError, ValueError):
                av = {}
            if av.get("rule_tree_action") == av.get("final_action"):
                rule_tree_eq_final += 1
            try:
                trace = json.loads(r["agent_trace_json"] or "[]")
            except (TypeError, ValueError):
                trace = []
            case_latency = 0.0
            stamps = []
            for e in trace:
                if not isinstance(e, dict):
                    continue
                if e.get("ts"):
                    try:
                        stamps.append(float(e["ts"]))
                    except (TypeError, ValueError):
                        pass
                c = e.get("content") if isinstance(e.get("content"), dict) else None
                if not c:
                    continue
                if e.get("speaker") == "decision" and e.get("kind") == "final":
                    if c.get("source") == "llm_react":
                        decision_llm_runs += 1
                        rc = c.get("rationale_check") if isinstance(c.get("rationale_check"), dict) else {}
                        if rc.get("ok") and (c.get("cited_sources") or []):
                            grounded += 1
                if e.get("speaker") == "rag_retriever":
                    rag_total_runs += 1
                    if c.get("coverage") == "direct_match":
                        direct_match_runs += 1
            if len(stamps) >= 2:
                case_latency = max(stamps) - min(stamps)
            if case_latency > 0:
                latencies.append(case_latency)
        median_lat = (sorted(latencies)[len(latencies) // 2] if latencies else 0.0)
        kpis = [
            {
                "key": "correct_action",
                "label": "Agreement with rule tree",
                "target_pct": 85,
                "value_pct": round(100.0 * rule_tree_eq_final / max(1, n), 1),
                "numerator": rule_tree_eq_final,
                "denominator": n,
                "status": "ok" if (rule_tree_eq_final / max(1, n)) >= 0.85 else "fail",
                "note": "no ground truth yet - fill expected_verdicts.csv to switch to real accuracy",
            },
            {
                "key": "grounded_citations",
                "label": "Grounded LLM citations",
                "target_pct": 90,
                "value_pct": round(100.0 * grounded / max(1, decision_llm_runs), 1) if decision_llm_runs else 0.0,
                "numerator": grounded,
                "denominator": decision_llm_runs,
                "status": "ok" if (decision_llm_runs and grounded / decision_llm_runs >= 0.90) else "fail",
                "note": "share of Decision LLM runs that passed rationale_check with cited sources",
            },
            {
                "key": "rag_relevance",
                "label": "RAG direct match",
                "target_pct": 95,
                "value_pct": round(100.0 * direct_match_runs / max(1, rag_total_runs), 1) if rag_total_runs else 0.0,
                "numerator": direct_match_runs,
                "denominator": rag_total_runs,
                "status": "ok" if (rag_total_runs and direct_match_runs / rag_total_runs >= 0.95) else "fail",
                "note": "share of RAG runs that returned a direct topic match for the hazard class",
            },
            {
                "key": "median_latency_s",
                "label": "Median latency (s)",
                "target_pct": 360,  # a seconds cap, not a percent
                "value_pct": round(median_lat, 1),
                "numerator": 0,
                "denominator": 0,
                "status": "ok" if median_lat <= 360 else "fail",
                "note": "wall seconds per case; cap 360s",
            },
        ]
        return {"window_hours": hrs, "total_hazards": n, "kpis": kpis}

    @app.get("/stats/advantage")
    def stats_advantage(request: Request, window_hours: int = 24) -> dict:
        """Round 23 (4.11.2026): cases where the agent outperformed the
        deterministic rule tree OR the human (deny on approval). Returns
        two lists the /flow page and the deck case-study slides read.
        Private, like the other /stats endpoints.
        """
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        try:
            hrs = max(1, min(720, int(window_hours)))
        except (TypeError, ValueError):
            hrs = 24
        settings = get_settings()
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(settings.roadguard_db_path)
        con.row_factory = _sqlite3.Row
        now = time.time()
        try:
            rows = con.execute(
                "SELECT h.id, h.class, h.agent_decision, h.agent_value_json, "
                "       h.reported_at, h.priority, "
                "       a.status AS approval_status, a.note AS approval_note, a.decided_at "
                "FROM hazards h "
                "LEFT JOIN approvals a ON a.hazard_id = h.id "
                "WHERE h.reported_at >= ? "
                "ORDER BY h.reported_at DESC",
                (now - hrs * 3600,),
            ).fetchall()
        finally:
            con.close()
        vs_tree: list[dict] = []
        vs_human_approved: list[dict] = []
        vs_human_denied: list[dict] = []
        for r in rows:
            try:
                av = json.loads((r["agent_value_json"] if r["agent_value_json"] else "") or "{}")
            except (TypeError, ValueError):
                av = {}
            if not isinstance(av, dict):
                continue
            if av.get("changed_rule_hint"):
                vs_tree.append({
                    "id": r["id"], "class": r["class"], "decision": r["agent_decision"],
                    "rule_tree_action": av.get("rule_tree_action"),
                    "final_action": av.get("final_action"),
                    "profile_llm_changed": av.get("profile_llm_changed"),
                    "research_llm_changed": av.get("research_llm_changed"),
                    "reported_at": r["reported_at"],
                })
            status = (r["approval_status"] if hasattr(r, "__getitem__") else getattr(r, "approval_status", None)) or ""
            if status == "approved":
                vs_human_approved.append({
                    "id": r["id"], "class": r["class"], "decision": r["agent_decision"],
                    "note": r["approval_note"], "decided_at": r["decided_at"],
                })
            elif status == "denied":
                vs_human_denied.append({
                    "id": r["id"], "class": r["class"], "decision": r["agent_decision"],
                    "note": r["approval_note"], "decided_at": r["decided_at"],
                })
        return {
            "window_hours": hrs,
            "total_hazards": len(rows),
            "advantage_over_tree": vs_tree,
            "agent_approved_by_human": vs_human_approved,
            "agent_corrected_by_human": vs_human_denied,
        }

    @app.get("/digest/latest", response_class=HTMLResponse)
    def digest_latest(request: Request) -> HTMLResponse:
        """Live digest - rebuilt from the DB on every request, so each
        new hazard shows up as soon as the pipeline writes its verdict.
        Private: gated to Tailscale / loopback (ROADGUARD_PRIVATE_HOSTS)
        exactly like /trace and /supervisor, so only internal operators
        see the full case detail."""
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        return HTMLResponse(content=_build_live_digest_html(window_hours=24))

    @app.get("/digest/{date}", response_class=HTMLResponse)
    def digest_html(date: str, request: Request) -> HTMLResponse:
        """Serve the archived digest HTML for a specific date (YYYY-MM-DD),
        built by rg-digest cron at 05:00 and 17:00 UTC. Also private."""
        if not _is_private_host(request):
            raise HTTPException(status_code=404, detail="not found")
        from pathlib import Path as _Path
        import re as _re
        if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
            raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
        path = _Path("/var/lib/roadguard/digests") / f"{date}.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"digest for {date} not available")
        return HTMLResponse(content=path.read_text(encoding="utf-8"))

    @app.post("/telemetry/event")
    def telemetry_event(body: dict, request: Request) -> dict:
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
            if ev:
                if not isinstance(meta, dict):
                    meta = {"value": str(meta)[:120]}
                parts = []
                for k, v in list(meta.items())[:8]:
                    kstr = str(k)[:40]
                    vstr = str(v)[:120].replace("\n", " ").replace("\r", " ")
                    parts.append(f"{kstr}={vstr}")
                line = f"[CLIENT   ] event={ev}  " + "  ".join(parts)
                print(line[:600], flush=True)
        except Exception:  # pragma: no cover - telemetry must never 500
            pass
        return {}

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
            # A Google identity that belongs to a registered user signs in as
            # that user (matched by the linked Google subject, else by the
            # verified e-mail), so an admin keeps the operator links whichever
            # button was used (31.10.2026). An unknown identity keeps the
            # Google subject and gets its row on the first upload.
            from roadguard_agent import Database

            sub = payload["sub"]
            with Database(settings.roadguard_db_path) as db:
                u = db.get_user_by_google_sub(payload["sub"]) or db.get_user_by_email(payload["email"])
                if u is not None:
                    if not u.google_sub:
                        db.set_google_sub(u.id, payload["sub"])
                    sub = f"user:{u.id}"
            token = create_jwt(
                sub=sub,
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

    @app.get("/auth/config", response_model=AuthConfigResponse)
    def auth_config(settings: Settings = Depends(get_settings)) -> AuthConfigResponse:
        """Which login methods will the server accept. Unauthenticated on
        purpose - the frontend needs this to decide which buttons to show
        before anyone logs in. Reads only env toggles + a count of existing
        users; never returns usernames or emails."""
        with Database(settings.roadguard_db_path) as db:
            n_users = db.count_users()
        return AuthConfigResponse(
            signup_enabled=bool(settings.roadguard_signup_enabled),
            password_login_enabled=bool(settings.roadguard_password_login_enabled),
            google_login_enabled=bool(settings.google_oauth_client_id),
            max_users=int(settings.roadguard_max_users),
            user_count=n_users,
        )

    @app.post("/auth/register", response_model=LoginResponse)
    @limiter.limit(lambda: get_settings().roadguard_login_limit)
    def register(
        request: Request,
        body: RegisterRequest = Body(...),
        settings: Settings = Depends(get_settings),
    ) -> LoginResponse:
        """New user signup. Gated by ROADGUARD_SIGNUP_ENABLED (default off:
        the production VM is closed to new users)."""
        if not settings.roadguard_signup_enabled:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="signup_disabled")
        ok, reason = validate_username(body.username)
        if not ok:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)
        ok, reason = validate_password(body.password)
        if not ok:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)
        with Database(settings.roadguard_db_path) as db:
            if db.count_users() >= settings.roadguard_max_users:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                    detail="user_cap_reached")
            if db.get_user_by_username(body.username) is not None:
                # Stable error key; same shape as a wrong-password reply so a
                # scanner cannot enumerate existing usernames via timing or
                # distinct messages (register-vs-login reveal the same thing).
                raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                    detail="username_taken")
            user_id = f"u_{secrets.token_hex(8)}"
            db.create_user_full(
                user_id=user_id,
                username=body.username,
                email=(body.email or "").strip().lower(),
                display_name=body.username,
                password_hash=hash_password(body.password),
                google_sub=None,
                is_admin=False,
                is_internal=False,
            )
        token = create_jwt(
            sub=f"user:{user_id}", email=(body.email or "").lower(),
            display_name=body.username, settings=settings,
        )
        return LoginResponse(token=token, expires_in=settings.roadguard_jwt_ttl)

    @app.post("/auth/login/password", response_model=LoginResponse)
    @limiter.limit(lambda: get_settings().roadguard_login_limit)
    def login_password(
        request: Request,
        body: PasswordLoginRequest = Body(...),
        settings: Settings = Depends(get_settings),
    ) -> LoginResponse:
        """username + password login. Response time is intentionally padded
        to the bcrypt cost whether or not the user exists, so a scanner
        cannot distinguish 'bad user' from 'bad password' by timing."""
        if not settings.roadguard_password_login_enabled:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="password_login_disabled")
        with Database(settings.roadguard_db_path) as db:
            user = db.get_user_by_username(body.username)
        # Always run bcrypt once - even on nonexistent users - to equalize timing.
        # Use the stored hash if present, else a cheap sham bcrypt hash so the
        # check still executes.
        _sham = "$2b$12$CFSdxWEptIE4LPmPSCCJQenfBXAcMFzaS4iN4Bq6Pv/HPWx7uHcSi"
        stored = user.password_hash if (user and user.password_hash) else _sham
        is_valid = verify_password(body.password, stored) if user else False
        if not user or not is_valid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="invalid_credentials")
        token = create_jwt(
            sub=f"user:{user.id}", email=user.email,
            display_name=user.display_name or user.username or user.id,
            settings=settings,
        )
        return LoginResponse(token=token, expires_in=settings.roadguard_jwt_ttl)

    @app.post("/auth/logout")
    def logout(user: dict[str, Any] = Depends(get_current_user)) -> dict:
        """Stateless logout: the client deletes its JWT. The server returns
        {ok: true} so the UI knows it succeeded. We do NOT keep a per-token
        blacklist here because the JWT TTL is already bounded (7 days by
        default) and a blacklist would reset on every restart anyway;
        rotating ROADGUARD_JWT_SECRET invalidates every outstanding token at
        once if a real emergency comes."""
        return {"ok": True, "sub": user.get("sub", "")}

    @app.get("/auth/me", response_model=Me)
    def me(
        user: dict[str, Any] = Depends(get_current_user),
        settings: Settings = Depends(get_settings),
    ) -> Me:
        sub = user.get("sub", "")
        username = None
        is_admin = False
        is_internal = False
        uploads_today = 0
        uploads_total = 0
        user_id = _user_id_of(user)
        with Database(settings.roadguard_db_path) as db:
            u = db.get_user(user_id) if user_id else None
            if u is None and sub.startswith("user:"):
                u = db.get_user(sub.split(":", 1)[1])
            if u is not None:
                username = u.username
                is_admin = u.is_admin
                is_internal = u.is_internal
                import time as _t
                day_start = _t.time() - 24 * 3600
                uploads_today = db.count_user_hazards_since(u.id, day_start)
                uploads_total = db.count_user_hazards_since(u.id, 0.0)
        operator_urls = []
        if is_admin:
            operator_urls = [
                "https://netsec-agent.tail37ac21.ts.net/digest/latest",
                "https://netsec-agent.tail37ac21.ts.net/trace",
                "https://netsec-agent.tail37ac21.ts.net/flow",
                "https://netsec-agent.tail37ac21.ts.net/supervisor",
            ]
        return Me(
            sub=sub, email=user.get("email", ""), name=user.get("name", ""),
            username=username, is_admin=is_admin, is_internal=is_internal,
            uploads_today=uploads_today, uploads_total=uploads_total,
            daily_cap=int(settings.roadguard_user_daily_submit_cap),
            operator_urls=operator_urls,
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

    def _save_hazard_image(hazard_id: str, b64: str | None) -> None:
        """Decode and save a webapp-uploaded JPEG. Enforces a 5 MB raw cap
        plus the 5 GB total-store cap (oldest files purged on overflow).
        Silent on any error - the pipeline continues without the image."""
        if not b64:
            return
        try:
            import base64, os, re as _re
            if not _re.fullmatch(r"hz_[0-9a-f]{6,32}", hazard_id or ""):
                return
            payload = b64.split(",", 1)[-1]  # strip "data:image/jpeg;base64,"
            raw = base64.b64decode(payload, validate=False)
            if len(raw) == 0 or len(raw) > 5 * 1024 * 1024:
                LOGGER.warning("hazard_image_rejected hazard_id=%s bytes=%d", hazard_id, len(raw))
                return
            os.makedirs("/var/lib/roadguard/images", exist_ok=True)
            path = f"/var/lib/roadguard/images/{hazard_id}.jpg"
            with open(path, "wb") as fh:
                fh.write(raw)
            try:
                os.chmod(path, 0o644)
            except Exception:
                pass
            # Enforce 5 GB cap: if total size exceeds, delete oldest first.
            cap = 5 * 1024 * 1024 * 1024
            files = []
            total = 0
            for name in os.listdir("/var/lib/roadguard/images"):
                p = f"/var/lib/roadguard/images/{name}"
                try:
                    st = os.stat(p)
                    files.append((st.st_mtime, st.st_size, p))
                    total += st.st_size
                except Exception:
                    pass
            if total > cap:
                files.sort()  # oldest first
                for _mt, sz, p in files:
                    if total <= cap:
                        break
                    try:
                        os.remove(p)
                        total -= sz
                        LOGGER.info("hazard_image_evicted path=%s total_after=%d", p, total)
                    except Exception:
                        pass
            print(f"[IMAGE    ] hazard_id={hazard_id}  bytes={len(raw)}  path={path}", flush=True)
        except Exception as exc:
            LOGGER.warning("hazard_image_save_failed hazard_id=%s err=%s", hazard_id, exc)

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
        _check_user_daily_submit_cap(user, db, settings)
        hazard_id = f"hz_{uuid.uuid4().hex[:12]}"
        _save_hazard_image(hazard_id, getattr(body, "image_jpeg_base64", None))
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
        _check_user_daily_submit_cap(user, db, settings)
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
        _check_user_daily_submit_cap(user, db, settings)
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
        request: Request,
        limit: int = 100,
        db=Depends(get_db),
    ) -> HazardHistoryResponse:
        """Owned hazards for the signed-in user; on a private operator host
        (tailnet / loopback) returns the full recent fleet instead so the
        /flow and /trace pages work without a JWT, since Caddy basicauth
        has already gatekept access."""
        limit = max(1, min(limit, 500))
        if _is_private_host(request):
            rows = db.list_recent_hazards(limit=limit)
        else:
            user = get_current_user(request, get_settings())
            rows = db.list_user_hazards(_user_id_of(user), limit=limit)
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
        request: Request,
        db=Depends(get_db),
    ) -> dict:
        """Full case report: metadata, verdict, per-agent trace, grounded
        sources, decision timeline, token usage. Private operator hosts
        read any hazard; the citizen path still enforces ownership."""
        if _is_private_host(request):
            r = db.get_hazard(hazard_id)
        else:
            user = get_current_user(request, get_settings())
            r = db.get_hazard(hazard_id, user_id=_user_id_of(user))
        if r is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="hazard_not_found",
            )
        trace = json.loads(r.agent_trace_json or "[]")
        timeline, tokens_in, tokens_out, cited, llm_calls = _report_trace_summary(trace)
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
                    # Round 17: the other detections on the MODEL line.
                    "bboxes": r.bboxes_json,
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
        from roadguard_agent import _policy_queues

        sla_hours = _policy_queues()["approval_sla_hours"]
        now = time.time()
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "total_hazards": len(rows),
            "by_action": by_action,
            "by_class": by_class,
            "approval_sla_hours": sla_hours,
            "overdue_approvals": approvals_overdue(pending, sla_hours, now),
            "pending_approvals": [
                {
                    "approval_id": a.id,
                    "hazard_id": a.hazard_id,
                    "action": a.action,
                    "priority": a.priority,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime(a.created_at)),
                    "age_hours": round((now - a.created_at) / 3600, 1),
                    "overdue": (now - a.created_at) > sla_hours * 3600,
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
        # The human's verdict becomes a lesson the agents can recall
        # (agent_notes). Best-effort: a failed note never fails the decision.
        try:
            record_decision_lesson(db, approval_id, decision, note)
        except Exception as exc:
            LOGGER.warning("agent_note_write_failed approval_id=%s: %s", approval_id, exc)
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
        except IdenticalEvidenceError as exc:
            # Round 17 (1.11.2026): the original upload again is not
            # evidence; the request stays open and is not counted.
            try:
                db.release_evidence_request(request_id)
            except Exception as release_exc:
                LOGGER.error(
                    "evidence_request_release_failed id=%s: %s", request_id, release_exc
                )
            raise HTTPException(status_code=422, detail=str(exc))
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

