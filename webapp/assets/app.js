/**
 * RoadGuard web app - entry point.
 *
 * Two view states (data-view on <body>):
 *   - landing:  hero + how it works, gated start button
 *   - live:     back camera + bbox overlay + HUD + verdict log
 *
 * Access is gated by a sign-in modal (passphrase, with Google if the
 * client id is configured). Only signed-in users reach the VM.
 */

import { resolveApiOrigin } from "./auth.js";
import {
  clearToken,
  fetchJson,
  fetchMe,
  isSignedIn,
  loginWithGoogleIdToken,
  loginWithPassphrase,
  mountGoogleSignIn,
  parseJwt,
} from "./auth.js";
import { LiveDriver } from "./drive.js";

const state = {
  driver: null,
  view: "landing",
  verdicts: [],
  counts: { urgent: 0, standard: 0, needs_more_evidence: 0, human_review: 0 },
  topClass: null,
  startedAt: 0,
  timerHandle: null,
  pendingAfterSignin: null,
  tracks: 0,
};

const LEVEL_BY_ACTION = {
  urgent: "bad",
  standard: "good",
  needs_more_evidence: "warn",
  human_review: "info",
};

// ============================================================ BOOT
window.addEventListener("DOMContentLoaded", async () => {
  wireLanding();
  wireLive();
  wireSignIn();
  wireNavSignIn();
  refreshSignInLabel();
  resolveApiOrigin().catch(() => {});
});

// ============================================================ VIEW HELPERS
function setView(view) {
  state.view = view;
  document.body.dataset.view = view;
  document.getElementById("view-landing").hidden = view !== "landing";
  document.getElementById("view-live").hidden = view !== "live";
  const sessionBtn = document.getElementById("nav-session");
  sessionBtn.hidden = view !== "live";
  window.scrollTo({ top: 0 });
}

function toast(msg, ms = 2500) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._h);
  toast._h = setTimeout(() => {
    t.hidden = true;
  }, ms);
}

function refreshSignInLabel() {
  const label = document.getElementById("signin-label");
  if (!label) return;
  const payload = parseJwt();
  if (payload && payload.email) {
    label.textContent = payload.email.split("@")[0];
  } else if (payload && payload.name) {
    label.textContent = payload.name;
  } else {
    label.textContent = "Sign in";
  }
}

// ============================================================ LANDING
function wireLanding() {
  const startBtn = document.getElementById("start-btn");
  startBtn.addEventListener("click", () => {
    if (!isSignedIn()) {
      state.pendingAfterSignin = () => startLive();
      openSignIn();
      return;
    }
    startLive();
  });
}

// ============================================================ LIVE
function wireLive() {
  const stopBtn = document.getElementById("stop-btn");
  stopBtn.addEventListener("click", () => stopLive());
}

async function startLive() {
  setView("live");
  resetSession();
  showLoading(true);

  const video = document.getElementById("cam");
  const overlay = document.getElementById("overlay");
  const roadSelect = document.getElementById("drive-road-type");
  const laneSelect = document.getElementById("drive-lane-blocked");

  const driver = new LiveDriver({
    video,
    overlayCanvas: overlay,
    getMetadata: () => ({
      road_type: roadSelect.value || "main",
      lane_blocked: laneSelect.value || "none",
    }),
  });
  state.driver = driver;

  driver.on("state", ({ running, mode }) => {
    if (mode) {
      document.getElementById("live-provider").textContent =
        mode === "mock" ? "Mock" : mode.toUpperCase();
    }
    if (running) {
      showLoading(false);
    }
  });
  driver.on("tick", ({ frameCount, tracks, model }) => {
    state.tracks = tracks.length;
    document.getElementById("hud-tracks").textContent = String(tracks.length);
    document.getElementById("live-fps").textContent =
      driver.fps + " fps";
    if (tracks.length) {
      const best = tracks.reduce(
        (a, b) => (a.confidence > b.confidence ? a : b),
      );
      state.topClass = best.class;
      document.getElementById("hud-topclass").textContent = best.class;
    }
  });
  driver.on("verdict", ({ response }) => recordVerdict(response));
  driver.on("error", ({ code, message }) => {
    if (code === "camera_denied") {
      showLoading(false);
      setBanner({
        action: "human_review",
        rationale:
          "Camera permission denied. Allow back-camera access and press start again.",
        priority: null,
        cited_sources: [],
        cached: false,
      });
    } else {
      toast(`Error: ${code}${message ? " - " + message : ""}`, 4000);
    }
  });

  // Session nav chip
  document.getElementById("nav-session-label").textContent =
    "session " + driver.sessionId.slice(0, 6);

  state.startedAt = Date.now();
  state.timerHandle = setInterval(updateElapsed, 1000);
  updateElapsed();

  try {
    await driver.start();
  } catch (err) {
    showLoading(false);
    toast(`Could not start: ${err && err.message ? err.message : err}`, 4000);
  }
}

function stopLive() {
  if (state.driver) {
    state.driver.stop();
    state.driver = null;
  }
  if (state.timerHandle) {
    clearInterval(state.timerHandle);
    state.timerHandle = null;
  }
  setView("landing");
}

function resetSession() {
  state.verdicts = [];
  state.counts = { urgent: 0, standard: 0, needs_more_evidence: 0, human_review: 0 };
  state.topClass = null;
  state.tracks = 0;
  document.getElementById("verdict-log").innerHTML =
    `<div class="verdict-empty">Verdicts arrive here as RoadGuard clears each confirmed hazard.</div>`;
  document.getElementById("verdict-count").textContent = "0";
  document.getElementById("hud-tracks").textContent = "0";
  document.getElementById("hud-topclass").textContent = "--";
  document.getElementById("hud-verdicts").textContent = "0";
  document.getElementById("mini-urgent").textContent = "0";
  document.getElementById("mini-standard").textContent = "0";
  document.getElementById("mini-elapsed").textContent = "0:00";
  setBanner({
    action: null,
    rationale:
      "Point the back camera at the road. RoadGuard streams detections and will post a verdict as soon as a hazard holds for a few frames.",
    priority: null,
    cited_sources: [],
    cached: false,
  });
  const ringFill = document.getElementById("ring-fill");
  if (ringFill) {
    ringFill.style.strokeDashoffset = "263.9";
  }
  document.getElementById("hud-ring").removeAttribute("data-level");
}

function showLoading(on) {
  document.getElementById("live-loading").hidden = !on;
}

function updateElapsed() {
  if (!state.startedAt) return;
  const ms = Date.now() - state.startedAt;
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  document.getElementById("mini-elapsed").textContent =
    `${m}:${String(s).padStart(2, "0")}`;
}

function recordVerdict(response) {
  const action = response.verdict.action;
  state.verdicts.unshift({ ...response, at: Date.now() });
  state.counts[action] = (state.counts[action] || 0) + 1;
  document.getElementById("verdict-count").textContent =
    String(state.verdicts.length);
  document.getElementById("hud-verdicts").textContent =
    String(state.verdicts.length);
  document.getElementById("mini-urgent").textContent =
    String(state.counts.urgent);
  document.getElementById("mini-standard").textContent =
    String(state.counts.standard);
  setBanner({
    action,
    rationale: response.verdict.rationale,
    priority: response.verdict.priority,
    cited_sources: response.verdict.cited_sources,
    cached: response.cached,
    fallback_reason: response.fallback_reason,
  });
  const level = LEVEL_BY_ACTION[action] || "info";
  const ring = document.getElementById("hud-ring");
  ring.setAttribute("data-level", level);
  const ringFill = document.getElementById("ring-fill");
  if (ringFill) {
    const pct = Math.max(0, Math.min(1, (response.verdict.priority || 0) / 100));
    const dash = 263.9 * (1 - pct);
    ringFill.style.strokeDashoffset = String(dash);
  }
  appendVerdictLog(action, response);
}

function appendVerdictLog(action, response) {
  const log = document.getElementById("verdict-log");
  const empty = log.querySelector(".verdict-empty");
  if (empty) empty.remove();
  const level = LEVEL_BY_ACTION[action] || "info";
  const item = document.createElement("div");
  item.className = "verdict-item";
  item.dataset.level = level;
  const t = new Date();
  const stamp = `${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}:${String(t.getSeconds()).padStart(2, "0")}`;
  const safe = (response.verdict.rationale || "").replace(/[<>]/g, "");
  const cite = (response.verdict.cited_sources || [])[0];
  item.innerHTML = `
    <div class="verdict-time">
      <span>${stamp}</span>
      <span class="verdict-time-tag">${response.cached ? "cached" : "fresh"}</span>
    </div>
    <div class="verdict-msg">
      <strong>${action.replace(/_/g, " ").toUpperCase()}</strong>
      ${safe}
    </div>
    ${cite ? `<a class="verdict-src" href="${cite}" target="_blank" rel="noopener">${cite}</a>` : ""}
  `;
  log.insertBefore(item, log.firstChild);
  while (log.children.length > 50) {
    log.removeChild(log.lastChild);
  }
}

function setBanner({ action, rationale, priority, cited_sources, cached, fallback_reason }) {
  const banner = document.getElementById("verdict-banner");
  const level = action ? LEVEL_BY_ACTION[action] || "info" : "idle";
  banner.dataset.level = level;
  const head = document.getElementById("verdict-head-action");
  const tag = document.getElementById("verdict-head-tag");
  head.textContent = action ? action.replace(/_/g, " ") : "idle";
  if (action) {
    const bits = [];
    if (cached) bits.push("cached");
    if (priority !== null && priority !== undefined) bits.push(`priority ${priority}`);
    if (fallback_reason) bits.push(fallback_reason);
    tag.textContent = bits.join(" - ") || "fresh";
  } else {
    tag.textContent = "waiting";
  }
  document.getElementById("verdict-text").textContent = rationale || "";
  const src = document.getElementById("verdict-source");
  const firstCite = (cited_sources || [])[0];
  if (firstCite) {
    src.hidden = false;
    src.href = firstCite;
    src.textContent = `view procedure (${firstCite.replace(/^https?:\/\//, "").split("/")[0]})`;
  } else {
    src.hidden = true;
    src.removeAttribute("href");
  }
}

// ============================================================ SIGN IN
function wireSignIn() {
  const form = document.getElementById("signin-form");
  const err = document.getElementById("signin-error");
  const backdrop = document.getElementById("signin-backdrop");

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    err.hidden = true;
    err.textContent = "";
    const pass = document.getElementById("signin-pass").value;
    try {
      await loginWithPassphrase(pass);
      refreshSignInLabel();
      closeSignIn();
      runPending();
      toast("Signed in");
    } catch (e) {
      err.hidden = false;
      err.textContent = e && e.message ? e.message : "Sign-in failed";
    }
  });

  backdrop.addEventListener("click", () => closeSignIn());

  mountGoogleSignIn("google-signin-button", async (credential) => {
    err.hidden = true;
    try {
      await loginWithGoogleIdToken(credential);
      refreshSignInLabel();
      closeSignIn();
      runPending();
      toast("Signed in with Google");
    } catch (e) {
      err.hidden = false;
      err.textContent = e && e.message ? e.message : "Google sign-in failed";
    }
  });
}

function openSignIn() {
  document.getElementById("signin-modal").hidden = false;
  const googleWrap = document.getElementById("signin-google");
  googleWrap.hidden = !document.querySelector(
    "#google-signin-button iframe, #google-signin-button div",
  );
  setTimeout(() => document.getElementById("signin-pass").focus(), 60);
}
function closeSignIn() {
  document.getElementById("signin-modal").hidden = true;
}

function runPending() {
  if (state.pendingAfterSignin) {
    const fn = state.pendingAfterSignin;
    state.pendingAfterSignin = null;
    fn();
  }
}

function wireNavSignIn() {
  const btn = document.getElementById("nav-signin");
  btn.addEventListener("click", () => {
    if (isSignedIn()) {
      clearToken();
      refreshSignInLabel();
      toast("Signed out");
      if (state.view === "live") stopLive();
    } else {
      openSignIn();
    }
  });
}
