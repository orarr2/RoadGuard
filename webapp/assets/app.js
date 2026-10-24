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
  setUnauthorizedHandler,
} from "./auth.js";
import { LiveDriver } from "./drive.js";
import { YoloDetector, computeImageFingerprints } from "./detect.js";

const state = {
  driver: null,
  view: "landing",
  verdicts: [],
  counts: { urgent: 0, standard: 0, needs_more_evidence: 0, human_review: 0, known_image: 0 },
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
  known_image: "info",
};

// Human-readable labels for the four RDD2022 pavement defect classes.
// The server speaks the D-codes; the UI shows the descriptions so a
// citizen without civil-engineering vocabulary can tell what the detector
// just flagged.
const HAZARD_CLASS_LABELS = {
  D00: "Longitudinal crack",
  D10: "Transverse crack",
  D20: "Alligator cracking",
  D40: "Pothole",
};

// ============================================================ BOOT
window.addEventListener("DOMContentLoaded", async () => {
  wireLanding();
  wireLive();
  wireUpload();
  wireSignIn();
  wireNavSignIn();
  wireHeroSlideshow();
  wireAutoTelemetry();
  refreshSignInLabel();
  setUnauthorizedHandler(() => {
    refreshSignInLabel();
    openSignIn();
  });
  resolveApiOrigin().catch(() => {});
  track("page_load", { url: location.pathname, ua: navigator.userAgent.slice(0, 80) });
});

// ============================================================ HERO SLIDESHOW
function wireHeroSlideshow() {
  const slides = document.querySelectorAll(".hero-slide");
  if (!slides.length) return;
  // Rotate through the 25 real pavement images with crossfade + per-slide
  // metadata on the HUD so it reads as a live detector feed, not a static shot.
  const META = [
    { count: 2, pct: 72, pill: "D40 - standard",           level: "warn" },
    { count: 1, pct: 86, pill: "D40 - urgent",             level: "bad"  },
    { count: 1, pct: 54, pill: "D20 - standard",           level: "good" },
    { count: 3, pct: 88, pill: "D40 - urgent",             level: "bad"  },
    { count: 1, pct: 48, pill: "D10 - standard",           level: "good" },
    { count: 1, pct: 65, pill: "D40 - needs more evidence",level: "warn" },
    { count: 2, pct: 92, pill: "D20 - standard",           level: "good" },
    { count: 1, pct: 76, pill: "D40 - urgent",             level: "bad"  },
    { count: 1, pct: 59, pill: "D00 - standard",           level: "good" },
    { count: 2, pct: 83, pill: "D20 - standard",           level: "good" },
    { count: 1, pct: 94, pill: "D40 - urgent",             level: "bad"  },
    { count: 1, pct: 42, pill: "D10 - needs more evidence",level: "warn" },
    { count: 1, pct: 71, pill: "D40 - urgent",             level: "bad"  },
    { count: 2, pct: 68, pill: "D20 - standard",           level: "good" },
    { count: 1, pct: 55, pill: "D00 - standard",           level: "good" },
    { count: 1, pct: 81, pill: "D40 - urgent",             level: "bad"  },
    { count: 1, pct: 63, pill: "D10 - standard",           level: "good" },
    { count: 3, pct: 89, pill: "D40 - urgent",             level: "bad"  },
    { count: 1, pct: 50, pill: "D20 - needs more evidence",level: "warn" },
    { count: 1, pct: 77, pill: "D40 - urgent",             level: "bad"  },
    { count: 2, pct: 85, pill: "D20 - standard",           level: "good" },
    { count: 1, pct: 46, pill: "D00 - standard",           level: "good" },
    { count: 1, pct: 95, pill: "D40 - urgent",             level: "bad"  },
    { count: 1, pct: 58, pill: "D10 - standard",           level: "good" },
  ];
  const PILL_LVL = { bad: "bad", warn: "warn", good: "good" };
  const countEl  = document.getElementById("hero-hud-count");
  const fillEl   = document.getElementById("hero-hud-fill");
  const pillEl   = document.getElementById("hero-hud-pill");
  let i = 0;
  const update = () => {
    const next = (i + 1) % slides.length;
    slides[i].classList.remove("hero-slide--active");
    slides[next].classList.add("hero-slide--active");
    i = next;
    const m = META[i] || META[0];
    if (countEl) countEl.textContent = String(m.count);
    if (fillEl)  fillEl.style.width = m.pct + "%";
    if (pillEl) {
      pillEl.textContent = m.pill;
      pillEl.className = "hero-hud-pill hero-hud-pill--" + (PILL_LVL[m.level] || "bad");
    }
  };
  setInterval(update, 3500);
}

// ============================================================ VIEW HELPERS
function setView(view) {
  const prev = state.view;
  state.view = view;
  document.body.dataset.view = view;
  document.getElementById("view-landing").hidden = view !== "landing";
  document.getElementById("view-live").hidden = view !== "live";
  document.getElementById("view-upload").hidden = view !== "upload";
  const sessionBtn = document.getElementById("nav-session");
  sessionBtn.hidden = view !== "live";
  window.scrollTo({ top: 0 });
  track("navigate", { from: prev || "none", to: view });
}

// Lightweight client telemetry: fire-and-forget POST /telemetry/event.
// The server prints a [CLIENT   ] line to the journal so rg-trace shows
// user actions interleaved with the pipeline events. Silent on failure
// (never blocks the UI). The POST is wrapped in try/catch at the call
// site style; here we just rely on fetch's error promise and ignore it.
async function track(event, meta) {
  try {
    const resolved = await resolveApiOrigin().catch(() => "");
    const base = resolved || "";
    await fetch(`${base}/telemetry/event`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ event, meta: meta || {} }),
      keepalive: true,
    }).catch(() => {});
  } catch {
    // telemetry must never break the UI
  }
}

// Wire a convenience auto-tracker on buttons/links tagged data-track="label"
// so we do not have to sprinkle track() in every handler. Fires once per
// click; meta carries the element id, label and current view.
function wireAutoTelemetry() {
  document.addEventListener("click", (e) => {
    const el = e.target.closest("[data-track], button, a.btn, .nav-btn");
    if (!el) return;
    const label = el.dataset.track || el.id || (el.textContent || "").trim().slice(0, 40) || el.tagName.toLowerCase();
    track("click", { label, view: state.view });
  }, { capture: true });
  document.addEventListener("change", (e) => {
    const el = e.target;
    if (!el || !el.id) return;
    const kind = el.tagName === "SELECT" ? "select" : el.type || "input";
    const value = el.tagName === "SELECT" ? (el.value || "") : (kind === "checkbox" ? (el.checked ? "on" : "off") : (el.value ? "<set>" : "<empty>"));
    track("field_change", { field: el.id, kind, value, view: state.view });
  }, { capture: true });
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

  const uploadBtn = document.getElementById("upload-btn");
  const uploadInput = document.getElementById("upload-input");
  uploadBtn.addEventListener("click", () => {
    if (!isSignedIn()) {
      state.pendingAfterSignin = () => uploadInput.click();
      openSignIn();
      return;
    }
    uploadInput.click();
  });
  uploadInput.addEventListener("change", (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    openUploadWithFile(file);
    ev.target.value = "";  // allow re-selecting the same file
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
        mode.toUpperCase();
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
    ${cite ? `<div class="verdict-src">${citeHtml(cite)}</div>` : ""}
  `;
  log.insertBefore(item, log.firstChild);
  while (log.children.length > 50) {
    log.removeChild(log.lastChild);
  }
}

let _bannerFadeTimer = null;
function setBanner({ action, rationale, priority, cited_sources, cached, fallback_reason }) {
  const banner = document.getElementById("verdict-banner");
  const level = action ? LEVEL_BY_ACTION[action] || "info" : "idle";
  banner.dataset.level = level;
  banner.classList.remove("verdict-fade");
  if (_bannerFadeTimer) clearTimeout(_bannerFadeTimer);
  // On mobile the banner covers a quarter of the stage, so fade it out
  // after a few seconds once a verdict is in. New verdicts cancel the
  // pending fade and show the banner again.
  if (action) {
    _bannerFadeTimer = setTimeout(() => {
      banner.classList.add("verdict-fade");
    }, 6000);
  }
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
    src.innerHTML = citeHtml(firstCite);
  } else {
    src.hidden = true;
    src.innerHTML = "";
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

// ============================================================ UPLOAD
const uploadState = {
  detector: null,
  imageEl: null,
  lastDetections: null,
  // Set when the agent asked for one more photo (roadmap section 2):
  // {id, hazard_id, missing_field, message}. The next submit answers it
  // through /evidence_requests/{id}/fulfill instead of opening a new case.
  pendingEvidence: null,
};

const MISSING_FIELD_LABELS = {
  second_angle: "a second angle",
  closer_shot: "a closer, sharper shot",
  lighting: "better lighting",
  wider_shot: "a wider shot",
  location: "the location",
};

function setUploadSubmitLabel(waiting) {
  const btn = document.getElementById("upload-submit");
  const label = btn.querySelector("span.upload-submit-label");
  if (label) label.textContent = waiting ? "Send the new photo" : "Triage with agent";
}

// The pending request survives reloads and Back: the case stays paused on
// the server until the reporter answers it or it expires.
const PENDING_EVIDENCE_KEY = "roadguard.pendingEvidence";

// Stale pending-evidence requests (older than 10 minutes) are discarded on
// restore so the webapp does not try to fulfil an expired case and surface
// the "already answered or expired" error to the next upload. Each pending
// request carries the client-side timestamp of when it was stored.
const PENDING_EVIDENCE_TTL_MS = 10 * 60 * 1000;

function setPendingEvidence(req) {
  const stamped = { ...req, _stored_at: Date.now() };
  uploadState.pendingEvidence = stamped;
  try {
    localStorage.setItem(PENDING_EVIDENCE_KEY, JSON.stringify(stamped));
  } catch {
    // storage unavailable
  }
  setUploadSubmitLabel(true);
}

function clearPendingEvidence() {
  uploadState.pendingEvidence = null;
  try {
    localStorage.removeItem(PENDING_EVIDENCE_KEY);
  } catch {
    // ignore
  }
  setUploadSubmitLabel(false);
}

function restorePendingEvidence() {
  if (uploadState.pendingEvidence) return;
  try {
    const saved = JSON.parse(localStorage.getItem(PENDING_EVIDENCE_KEY) || "null");
    if (saved && typeof saved.id === "string" && saved.id) {
      const age = Date.now() - (saved._stored_at || 0);
      if (age > PENDING_EVIDENCE_TTL_MS) {
        // Older than the TTL: the server almost certainly expired this
        // evidence request already. Drop it so the next upload opens a
        // new case instead of hitting a 404 on fulfil.
        try { localStorage.removeItem(PENDING_EVIDENCE_KEY); } catch {}
        return;
      }
      uploadState.pendingEvidence = saved;
      setUploadSubmitLabel(true);
    }
  } catch {
    // corrupt or unavailable storage: start without a pending request
  }
}

function pendingEvidenceNoticeHtml() {
  const req = uploadState.pendingEvidence;
  if (!req) return "";
  const want = MISSING_FIELD_LABELS[req.missing_field] || req.missing_field || "one more photo";
  const caseRef = String(req.hazard_id || "").slice(-8);
  return `
    <div class="upload-evidence-ask">
      <strong>The agent asks for one more photo: ${String(want).replace(/[<>]/g, "")}</strong>
      ${String(req.message || "").replace(/[<>]/g, "")}
      <div class="upload-evidence-field">Case #${caseRef.replace(/[<>]/g, "")} resumes when you press <em>Send the new photo</em>.</div>
    </div>`;
}

function renderEvidenceRequest(resp) {
  const req = resp.evidence_request || {};
  const want = MISSING_FIELD_LABELS[req.missing_field] || req.missing_field || "one more photo";
  const caseRef = (resp.hazard_id || "").slice(-8);
  const html = `
    <div class="upload-verdict-card" data-level="warn">
      <div class="upload-verdict-head">
        <span class="upload-verdict-action">The agent asks for one more photo</span>
        <span class="upload-verdict-pri" title="Case is paused until you answer">paused</span>
      </div>
      <div class="upload-verdict-rationale">${String(req.message || "").replace(/[<>]/g, "")}</div>
      <div class="upload-evidence-ask">
        <strong>What to send: ${String(want).replace(/[<>]/g, "")}</strong>
        Pick the new photo with <em>Pick another</em>, then press <em>Send the new photo</em>.
        The case continues from where it paused; you get one such request per case.
        <div class="upload-evidence-field">Missing field: <code>${String(req.missing_field || "").replace(/[<>]/g, "")}</code></div>
      </div>
      <div class="upload-verdict-footer">
        <span>Case <strong>#${caseRef}</strong></span>
        <span>Waiting for your photo</span>
      </div>
    </div>`;
  document.getElementById("upload-verdict").innerHTML = html;
}

function wireUpload() {
  document.getElementById("upload-close").addEventListener("click", () => {
    setView("landing");
  });
  document.getElementById("upload-pick-again").addEventListener("click", () => {
    document.getElementById("upload-input").click();
  });
  document.getElementById("upload-pick-another").addEventListener("click", () => {
    document.getElementById("upload-input").click();
  });
  document.getElementById("upload-submit").addEventListener("click", submitUploadedImage);
}

async function openUploadWithFile(file) {
  setView("upload");
  restorePendingEvidence();
  const empty = document.getElementById("upload-empty");
  const canvas = document.getElementById("upload-canvas");
  const submitBtn = document.getElementById("upload-submit");
  const verdictBox = document.getElementById("upload-verdict");
  const trace = document.getElementById("upload-trace");
  // The agent's open request stays above whatever the detector reports.
  const show = (html) => {
    verdictBox.innerHTML = pendingEvidenceNoticeHtml() + html;
  };
  trace.hidden = true;
  show(`<div class="upload-verdict-empty">Loading image + running YOLO on-device...</div>`);
  submitBtn.disabled = true;

  const img = new Image();
  img.onload = async () => {
    empty.hidden = true;
    const ctx = canvas.getContext("2d");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    uploadState.imageEl = img;

    if (!uploadState.detector) {
      uploadState.detector = new YoloDetector();
      await uploadState.detector.load();
    }
    try {
      const result = await uploadState.detector.detect(canvas);
      uploadState.lastDetections = result;
      // Redraw image + overlay boxes WITHOUT clearing the canvas (drawBoxes
      // from detect.js clears, which is correct for the live overlay but
      // wrong here - the canvas IS the image, not a transparent overlay).
      ctx.drawImage(img, 0, 0);
      _overlayBoxes(ctx, result.detections, canvas.width);
      if (result.detections && result.detections.length) {
        const top = result.detections[0];
        const topLabel = HAZARD_CLASS_LABELS[top.class] || top.class;
        show(`
          <div class="upload-verdict-ready">
            <div class="upload-verdict-stage">YOLO on-device found <strong>${result.detections.length}</strong> detection${result.detections.length === 1 ? "" : "s"}.</div>
            <div class="upload-verdict-top">Top class: <strong>${topLabel}</strong> (<code>${top.class}</code>) at confidence <strong>${top.confidence.toFixed(2)}</strong></div>
            <div class="upload-verdict-cta">Press <em>Triage with agent</em> to send to the signal_extractor &rarr; profile_builder &rarr; rag_retriever &rarr; decision pipeline.</div>
          </div>`);
        submitBtn.disabled = false;
      } else {
        show(`
          <div class="upload-verdict-ready" style="border-left:4px solid #0ea5e9;padding-left:10px">
            <div class="upload-verdict-stage">YOLO on-device did not match any of the four known classes (D00/D10/D20/D40) on this image.</div>
            <div class="upload-verdict-top">This usually means a new or borderline defect the detector was not trained on.</div>
            <div class="upload-verdict-cta"><strong>Press <em>Triage with agent</em> below</strong> &mdash; the agent pipeline will still analyze the image and return a <code>human_review</code> verdict so a supervisor sees it.</div>
          </div>`);
        submitBtn.disabled = false;
      }
    } catch (err) {
      show(`<div class="upload-verdict-empty">Detector error: ${(err && err.message) || err}</div>`);
    }
  };
  img.onerror = () => {
    show(`<div class="upload-verdict-empty">Could not load image.</div>`);
  };
  uploadState.lastFile = file;
  img.src = URL.createObjectURL(file);
}

// The server accepts at most this many bbox rows per submit.
const MAX_BBOX_ROWS = 200;

async function submitUploadedImage() {
  const submitBtn = document.getElementById("upload-submit");
  const verdictBox = document.getElementById("upload-verdict");
  const trace = document.getElementById("upload-trace");
  const tracePre = document.getElementById("upload-trace-pre");
  const canvas = document.getElementById("upload-canvas");
  const roadType = document.getElementById("upload-road-type").value;
  const laneBlocked = document.getElementById("upload-lane-blocked").value;
  const location = (document.getElementById("upload-location").value || "").trim();
  const reporterText = (document.getElementById("upload-reporter-text").value || "").trim();
  const det = uploadState.lastDetections;
  if (!det) return;
  // All four context fields are REQUIRED before a case is created on the server.
  // Partial submits were allowed earlier and produced thin cases the agent could
  // not reason about; the form now blocks the submit until every field is filled.
  const missing = [];
  if (!roadType) missing.push("Road type");
  if (!laneBlocked) missing.push("Lane obstruction");
  if (!location) missing.push("Location");
  if (!reporterText) missing.push("Describe what you see");
  if (missing.length) {
    verdictBox.innerHTML = `<div class="upload-verdict-empty" style="color:#b91c1c;font-weight:600">Please fill in all fields before sending: ${missing.join(", ")}.</div>`;
    return;
  }

  const top = (det.detections && det.detections[0]) || null;
  const bboxes = (det.detections || [])
    .slice()
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, MAX_BBOX_ROWS)
    .map((d) => [d.x1, d.y1, d.x2, d.y2, d.class, d.confidence]);
  // Fingerprints let the agents recognise a photo already catalogued
  // in the reference dataset (the known_image action). SHA-256 of the
  // original file bytes + a 64-bit perceptual hash of the pixels.
  let fingerprints = { sha256: null, phash: null };
  if (uploadState.lastFile) {
    try {
      fingerprints = await computeImageFingerprints(uploadState.lastFile, canvas);
    } catch (err) {
      console.warn("fingerprint failed", err);
    }
  }
  // No image_quality field: the client does not measure it and the
  // server falls back to the same defaults when it is absent.
  const body = {
    class: top ? top.class : "D00",
    confidence: top ? Math.max(0, Math.min(0.999, top.confidence)) : 0.0,
    top1_margin: det.top1_margin ?? 0,
    area_ratio: top ? (top.area_ratio ?? 0) : 0,
    n_detections_by_class: det.n_detections_by_class || {},
    bboxes,
    road_type: roadType,
    lane_blocked: laneBlocked,
    location,
    reporter_text: reporterText,
    image_sha256: fingerprints.sha256,
    image_phash: fingerprints.phash,
  };

  // Deduplicate: refuse to re-send the SAME image bytes in the same session.
  // Prevents accidental double-click floods and the user re-uploading an image
  // the agent is already working on. User can pick a different image to proceed.
  if (fingerprints.sha256 && uploadState.inflightHashes && uploadState.inflightHashes.has(fingerprints.sha256)) {
    verdictBox.innerHTML = `<div class="upload-verdict-empty" style="color:#b45309;font-weight:600">This exact image is already in flight or was just submitted. Pick a different photo (or wait for the current case to finish) to send another one.</div>`;
    return;
  }
  if (!uploadState.inflightHashes) uploadState.inflightHashes = new Set();
  if (fingerprints.sha256) uploadState.inflightHashes.add(fingerprints.sha256);

  // Answering the agent's evidence request resumes the paused case
  // instead of opening a new one.
  const pending = uploadState.pendingEvidence;
  // Use the async submit for fresh cases so the request returns in
  // under a second (the Cloudflare free edge times out at 100 s and the
  // Ollama panel takes longer on ARM CPU). Evidence resume stays
  // synchronous because its payload is small and the resumed graph
  // short-circuits most LLM calls.
  const path = pending ? `/evidence_requests/${encodeURIComponent(pending.id)}/fulfill` : "/hazards/submit-async";

  submitBtn.disabled = true;
  // Loading state: a visible spinner + text so the user knows the request is in flight.
  const spinnerHtml = `<div class="rg-spinner" aria-hidden="true" style="display:inline-block;width:16px;height:16px;border:2px solid #cbd5e1;border-top-color:#0ea5e9;border-radius:50%;animation:rg-spin 0.9s linear infinite;vertical-align:middle;margin-right:8px"></div>`;
  if (!document.getElementById("rg-spin-style")) {
    const style = document.createElement("style");
    style.id = "rg-spin-style";
    style.textContent = "@keyframes rg-spin { to { transform: rotate(360deg); } }";
    document.head.appendChild(style);
  }
  verdictBox.innerHTML = pending
    ? `<div class="upload-verdict-empty">${spinnerHtml}Sending the new photo, the agent continues from where it paused...</div>`
    : `<div class="upload-verdict-empty">${spinnerHtml}Submitted. Agents are working on your case; the verdict will appear below in 1-3 minutes.</div>`;
  try {
    const resp = await fetchJson(path, {
      method: "POST",
      body: JSON.stringify(body),
    });
    if (!pending && resp.hazard_id && (resp.status === "queued" || resp.status === "processing")) {
      // Async flow: poll /report/{hazard_id} until the run finishes. The
      // poller returns true when the agent paused for evidence, in which
      // case the pending request must survive for the next upload.
      const pausedForEvidence = await pollAsyncHazard(resp.hazard_id, verdictBox);
      if (!pausedForEvidence) clearPendingEvidence();
    } else if (resp.status === "waiting_for_evidence" && resp.evidence_request) {
      setPendingEvidence(resp.evidence_request);
      renderEvidenceRequest(resp);
    } else {
      clearPendingEvidence();
      renderUploadVerdict(resp);
    }
  } catch (err) {
    const status = err && err.status;
    if (pending && status === 404) {
      // Already answered or expired on the server; the next submit opens a new case.
      clearPendingEvidence();
      verdictBox.innerHTML = `<div class="upload-verdict-empty">The agent's request for this case is closed - it was already answered or it expired. You can send this photo as a new case with <em>Triage with agent</em>.</div>`;
    } else if (status === 401) {
      // auth.js has already dropped the token and opened the sign-in modal.
      state.pendingAfterSignin = () => submitUploadedImage();
      verdictBox.innerHTML = `<div class="upload-verdict-empty">Your session expired. Sign in to send this photo.</div>`;
    } else {
      verdictBox.innerHTML = `<div class="upload-verdict-empty">Agent call failed: ${(err && err.message) || err}</div>`;
    }
  } finally {
    submitBtn.disabled = false;
  }
}

async function pollAsyncHazard(hazardId, verdictBox) {
  const maxWait = 300000;
  const started = Date.now();
  // Transient fetch failures (tunnel hiccup, mobile network wobble) are
  // retried in-place; only after N consecutive misses do we surface the
  // "Failed to fetch" line to the user. Each successful fetch resets.
  const MAX_CONSECUTIVE_ERRORS = 5;
  let consecutiveErrors = 0;
  while (Date.now() - started < maxWait) {
    await new Promise(r => setTimeout(r, 3000));
    try {
      const report = await fetchJson(`/report/${encodeURIComponent(hazardId)}`);
      consecutiveErrors = 0;
      const v = report.verdict || {};
      const inProgress = !v.status || v.status === "queued" || v.status === "processing";
      if (!inProgress) {
        if (v.status === "waiting_for_evidence" && report.evidence_request) {
          // The agent paused for one more photo: keep the request so the
          // next upload resumes this case instead of opening a new one.
          setPendingEvidence(report.evidence_request);
          renderEvidenceRequest({ hazard_id: hazardId, evidence_request: report.evidence_request });
          return true;
        }
        const resp = {
          hazard_id: hazardId,
          verdict: {
            action: v.action,
            rationale: v.rationale,
            priority: v.priority,
            cited_sources: report.grounded_sources || [],
          },
          provider: "multiagent",
          fallback_reason: null,
          status: v.status,
        };
        renderUploadVerdict(resp);
        return false;
      }
      const elapsed = Math.round((Date.now() - started) / 1000);
      const steps = (report.trace || []).length;
      const queuePos = Number(report.queue_position || 0);
      const spin = `<div class="rg-spinner" aria-hidden="true" style="display:inline-block;width:16px;height:16px;border:2px solid #cbd5e1;border-top-color:#0ea5e9;border-radius:50%;animation:rg-spin 0.9s linear infinite;vertical-align:middle;margin-right:8px"></div>`;
      if (v.status === "queued") {
        const posTxt = queuePos > 0 ? ` (position ${queuePos} in queue)` : "";
        verdictBox.innerHTML = `<div class="upload-verdict-empty">${spin}Queued behind another case${posTxt}... ${elapsed}s elapsed. Each case takes 1-3 minutes on this VM.</div>`;
      } else {
        verdictBox.innerHTML = `<div class="upload-verdict-empty">${spin}Working... ${elapsed}s elapsed, ${steps} pipeline steps done (local agents on CPU take 1-5 minutes).</div>`;
      }
    } catch (err) {
      consecutiveErrors += 1;
      if (consecutiveErrors < MAX_CONSECUTIVE_ERRORS) {
        // Transient: keep polling, let the user know we're retrying so the
        // UI does not appear frozen while the background retry happens.
        const spin = `<div class="rg-spinner" aria-hidden="true" style="display:inline-block;width:16px;height:16px;border:2px solid #cbd5e1;border-top-color:#f59e0b;border-radius:50%;animation:rg-spin 0.9s linear infinite;vertical-align:middle;margin-right:8px"></div>`;
        verdictBox.innerHTML = `<div class="upload-verdict-empty">${spin}Reconnecting... (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}) - agent is still working on the server.</div>`;
        continue;
      }
      verdictBox.innerHTML = `<div class="upload-verdict-empty">Could not reach the server after ${MAX_CONSECUTIVE_ERRORS} tries: ${(err && err.message) || err}. The agent may still finish - come back in a minute and reload.</div>`;
      return false;
    }
  }
  verdictBox.innerHTML = `<div class="upload-verdict-empty">Timed out waiting for the verdict. The agent may still be running on the server; check back in a minute.</div>`;
  return false;
}

function _overlayBoxes(ctx, detections, width) {
  if (!detections || !detections.length) return;
  ctx.lineWidth = Math.max(3, Math.round(width / 300));
  ctx.font = `${Math.max(16, Math.round(width / 40))}px system-ui, sans-serif`;
  ctx.textBaseline = "top";
  for (const d of detections) {
    ctx.strokeStyle = "#ff8800";
    ctx.fillStyle = "rgba(255, 136, 0, 0.9)";
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const label = `${HAZARD_CLASS_LABELS[d.class] || d.class} ${(d.confidence * 100).toFixed(0)}%`;
    const metrics = ctx.measureText(label);
    const padding = 6;
    const textH = parseInt(ctx.font, 10) + padding;
    ctx.fillRect(d.x1, Math.max(0, d.y1 - textH), metrics.width + padding * 2, textH);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, d.x1 + padding, Math.max(0, d.y1 - textH + padding / 2));
  }
}

// Lookup table built from data/mini_corpus.jsonl - each known cited source
// id -> {label, url, body} so the municipal reviewer sees a clickable
// standard, not an internal code. Add more entries here when Benny ships
// the full RAG corpus.
const STANDARDS = {
  "sec-4.3":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 4.3 - D40 on arterial, 24h patch window", url: "https://iroads.example/standards/pavement-2024-sec-4.3" },
  "sec-2.1":           { body: "Israel Traffic Safety Authority", label: "Traffic Safety § 2.1 - lane obstruction, immediate signage", url: "https://iroads.example/standards/traffic-safety-sec-2.1" },
  "sec-4.4":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 4.4 - D40 on secondary, 1-week window", url: "https://iroads.example/standards/pavement-2024-sec-4.4" },
  "parking-sec-3.1":   { body: "Municipality Public Works",       label: "Municipal Parking § 3.1 - D40 in parking, monthly cycle", url: "https://iroads.example/standards/municipal-parking-sec-3.1" },
  "sec-5.1":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 5.1 - D20 on arterial, 72h window", url: "https://iroads.example/standards/pavement-2024-sec-5.1" },
  "sec-5.2":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 5.2 - D20 on secondary, 2-week window", url: "https://iroads.example/standards/pavement-2024-sec-5.2" },
  "sec-5.4":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 5.4 - D20 maintenance guidance", url: "https://iroads.example/standards/pavement-2024-sec-5.4" },
  "sec-6.1":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 6.1 - D00 on arterial, quarterly resurface", url: "https://iroads.example/standards/pavement-2024-sec-6.1" },
  "sec-6.2":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 6.2 - D00 on secondary, annual resurface", url: "https://iroads.example/standards/pavement-2024-sec-6.2" },
  "sec-7.1":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 7.1 - D10 on arterial, quarterly resurface", url: "https://iroads.example/standards/pavement-2024-sec-7.1" },
  "sec-7.2":           { body: "Israel National Roads Company",   label: "Pavement 2024 § 7.2 - D10 on secondary, annual resurface", url: "https://iroads.example/standards/pavement-2024-sec-7.2" },
  "bike-sec-2.5":      { body: "Municipality Public Works",       label: "Bike paths § 2.5 - surface cracking, 2-week window", url: "https://iroads.example/standards/bike-sec-2.5" },
  "bike-sec-4.1":      { body: "Municipality Public Works",       label: "Bike paths § 4.1 - D40 on bike paths, 72h window", url: "https://iroads.example/standards/bike-sec-4.1" },
  "safety-ocr-01":     { body: "RoadGuard internal policy",       label: "Safety OCR § 1 - suspicious text routes to human review", url: "https://roadguard.example/policy/safety-ocr-01" },
  "safety-non-road-01":{ body: "RoadGuard internal policy",       label: "Safety non-road § 1 - non-road scenes to human review", url: "https://roadguard.example/policy/safety-non-road-01" },
  "safety-location-01":{ body: "RoadGuard internal policy",       label: "Safety location § 1 - sanitized location to human review", url: "https://roadguard.example/policy/safety-location-01" },
};

function citeHtml(id) {
  const meta = STANDARDS[id];
  if (meta) {
    // The stub corpus carries placeholder .example URLs until the real
    // standards ship - render those as a badge, never as a dead link.
    const linkable = /^https?:\/\/[^/]+/.test(meta.url) && !/\.example(\/|$)/.test(meta.url);
    return linkable
      ? `<a class="verdict-cite" href="${meta.url}" target="_blank" rel="noopener" title="${meta.body}">${meta.label}</a>`
      : `<span class="verdict-cite" title="${meta.body} - source link pending corpus delivery">${meta.label}</span>`;
  }
  return `<span class="verdict-cite">${String(id).replace(/[<>]/g, "")}</span>`;
}

const ACTION_COPY = {
  urgent: {
    headline:    "Urgent dispatch recommended",
    guidance:    "Approve to notify the on-call maintenance crew. The approval is logged against this case.",
  },
  standard: {
    headline:    "Standard repair queue",
    guidance:    "Routed to the regular maintenance queue. No supervisor action required unless you want to override.",
  },
  needs_more_evidence: {
    headline:    "More evidence needed",
    guidance:    "Image was ambiguous. The reporter will be asked for a clearer / closer shot before triage finishes.",
  },
  human_review: {
    headline:    "Awaiting supervisor review",
    guidance:    "Case flagged to you for a manual decision. Common reasons: non-road scene, borderline confidence, or suspicious text.",
  },
  known_image: {
    headline:    "Already in the reference dataset",
    guidance:    "This photo is one of the catalogued dataset images. The agent reports the catalogued labels and the recorded model prediction; no new triage is produced.",
  },
};

// Shown in place of verdict.rationale, which is internal text (guard and
// fallback reasons). needs_more_evidence shows the server's request message.
const ACTION_SUMMARY = {
  urgent: "Flagged for urgent inspection. A municipal supervisor reviews it before any work is scheduled.",
  standard: "Queued for standard maintenance review.",
  human_review: "Sent to a municipal supervisor for a manual decision.",
  known_image: "This photo is already catalogued in the reference dataset.",
};

function renderUploadVerdict(resp) {
  const action = resp.verdict.action;
  const level = LEVEL_BY_ACTION[action] || "info";
  const copy  = ACTION_COPY[action] || { headline: action.toUpperCase(), guidance: "" };
  const priority = resp.verdict.priority || 0;
  const caseRef = (resp.hazard_id || "").slice(-8);
  const timeNow = new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

  const cited = (resp.verdict.cited_sources || []).map(citeHtml).join("");
  const summary = action === "needs_more_evidence"
    ? String((resp.evidence_request && resp.evidence_request.message) || "")
    : ACTION_SUMMARY[action] || "";

  const html = `
    <div class="upload-verdict-card" data-level="${level}">
      <div class="upload-verdict-head">
        <span class="upload-verdict-action">${copy.headline}</span>
        <span class="upload-verdict-pri" title="Priority score 0-100 - higher means more urgent">priority ${priority}</span>
      </div>
      <div class="upload-verdict-prio-bar">
        <div class="upload-verdict-prio-fill" style="width:${Math.max(2, Math.min(100, priority))}%"></div>
      </div>
      ${summary ? `<div class="upload-verdict-rationale">${summary.replace(/[<>]/g, "")}</div>` : ""}
      <div class="upload-verdict-guidance">${copy.guidance}</div>
      ${cited ? `
        <div class="upload-verdict-cites-wrap">
          <div class="upload-verdict-cites-head">Grounded in maintenance standards:</div>
          <div class="upload-verdict-cites">${cited}</div>
        </div>` : ""}
      <div class="upload-verdict-footer">
        <span>Case <strong>#${caseRef}</strong></span>
        <span>Logged at ${timeNow}</span>
      </div>
    </div>`;
  document.getElementById("upload-verdict").innerHTML = html;
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
