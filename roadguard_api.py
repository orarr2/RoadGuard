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
    for (const item of items) {
      try {
        const rep = await get("/report/" + encodeURIComponent(item.hazard_id));
        const lines = rep.trace_lines || [];
        for (const line of lines) print(line, classOf(line));
        print("");
      } catch (e) { /* skip the broken ones */ }
    }
    print("--- end of replay; following live ---", "hdr");
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
main {
  display: grid;
  grid-template-columns: 1fr 340px;
  gap: 14px;
  padding: 14px;
  max-width: 1600px;
  margin: 0 auto;
}
@media (max-width: 900px) {
  main { grid-template-columns: 1fr; }
}
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
  </aside>
</main>


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

async function tick() {
  if (paused) return;
  try {
    const history = await apiGet("/hazards/history?limit=1");
    const items = history.items || [];
    const targetId = items[0] ? items[0].hazard_id : null;
    if (!targetId) {
      updatePanel(null);
      el("empty-state").style.display = "block";
      el("flow").style.display = "none";
      latestNodeState = interpretTrace([], null);
      applyNodeStates(latestNodeState);
      requestAnimationFrame(drawArrows);
      return;
    }
    el("empty-state").style.display = "none";
    el("flow").style.display = "grid";
    const hazard = await apiGet("/report/" + encodeURIComponent(targetId));
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
            "SELECT id, agent_decision, priority, agent_rationale, reported_at, location, "
            "status, cited_sources_json, agent_trace_json "
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
            img_rel = f"/images/{r['id']}" if os.path.isfile(f"/var/lib/roadguard/images/{r['id']}.jpg") else ""
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
        kpi_strip = (
            "<div class='kpi-strip'>"
            f"<div class='kpi'><div class='kpi-v'>{len(rows)}</div><div class='kpi-l'>New reports</div></div>"
            f"<div class='kpi urgent'><div class='kpi-v'>{counts['urgent']}</div><div class='kpi-l'>Urgent</div></div>"
            f"<div class='kpi standard'><div class='kpi-v'>{counts['standard']}</div><div class='kpi-l'>Standard queue</div></div>"
            f"<div class='kpi human'><div class='kpi-v'>{counts['human_review']}</div><div class='kpi-l'>Human review</div></div>"
            f"<div class='kpi evi'><div class='kpi-v'>{counts['needs_more_evidence']}</div><div class='kpi-l'>Needs more evidence</div></div>"
            f"<div class='kpi'><div class='kpi-v'>{counts['known_image']}</div><div class='kpi-l'>Known image</div></div>"
            f"<div class='kpi'><div class='kpi-v'>{approvals_pending}</div><div class='kpi-l'>Approvals pending</div></div>"
            f"<div class='kpi'><div class='kpi-v'>{evidence_open}</div><div class='kpi-l'>Awaiting photo</div></div>"
            "</div>"
        )

        if rows:
            rows_html_parts = []
            for i, c in enumerate(cases, 1):
                ts = datetime.fromtimestamp(c["reported_at"], timezone.utc).strftime("%H:%M")
                d = html.escape(c["decision"])
                cls = _css_for(c["decision"])
                rows_html_parts.append(
                    f"<tr data-idx='{i-1}' class='row'>"
                    f"<td>{i}</td>"
                    f"<td><code>{html.escape(c['id'][:12])}</code></td>"
                    f"<td class='{cls}'>{d}</td>"
                    f"<td>{c['priority'] or ''}</td>"
                    f"<td>{html.escape(c['location'])[:60]}</td>"
                    f"<td>{ts}</td>"
                    f"</tr>"
                )
            rows_html = "".join(rows_html_parts)
            table_html = (
                "<table class='cases'>"
                "<thead><tr><th>#</th><th>id</th><th>decision</th><th>priority</th><th>location</th><th>time</th></tr></thead>"
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
            clusters_html = (
                "<div class='clusters-wrap'>"
                f"<p class='k'>No clusters (no location+class group with &ge; 3 reports in the last {window_hours}h). Add a few more submissions from the same place and this table will fill in.</p>"
                "</div>"
            )

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
            f"<button class='tab-btn' data-tab='timeline'>Timeline</button>"
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
        .detail .d-img{width:100%;max-height:280px;object-fit:cover;border-radius:8px;background:#f1f5f9;margin-bottom:12px;display:block}
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
          const img = c.image
            ? `<img class="d-img" src="${c.image}" alt="citizen photo">`
            : '<div class="d-img" style="display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:12px;height:160px">(photo not available)</div>';
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
        // Silent background refresh: re-fetch the page HTML, swap in the
        // KPIs + table + timeline sections only, preserve the user's active
        // tab, selected row, and scroll position so the page never jumps.
        async function refresh() {
          try {
            const html = await fetch(location.href, {cache: 'no-store'}).then(r => r.text());
            const doc = new DOMParser().parseFromString(html, 'text/html');
            const parts = ['.kpi-strip', 'tbody#rows', '#pane-timeline .tl'];
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
          } catch (e) { /* silent */ }
        }
        setInterval(refresh, 30000);
        """.replace("__CASES__", cases_json)

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
            f"<section id='pane-timeline' class='tab-pane'>{timeline_html}</section>"
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
            user = get_current_user(request)
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
            user = get_current_user(request)
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

