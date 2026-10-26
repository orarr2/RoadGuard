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
  fetchAuthConfig,
  fetchJson,
  fetchMe,
  isSignedIn,
  loginWithGoogleIdToken,
  loginWithPassphrase,
  loginWithUsernamePassword,
  logoutCall,
  mountGoogleSignIn,
  parseJwt,
  registerAccount,
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
  wireNavProfile();
  wireNavLogout();
  wireHeroSlideshow();
  wireAutoTelemetry();
  refreshSignInLabel();  // also calls refreshUserChrome()
  setUnauthorizedHandler(() => {
    state.me = null;
    refreshSignInLabel();
    openSignIn();
  });
  resolveApiOrigin().catch(() => {});
  track("page_load", { url: location.pathname, ua: navigator.userAgent.slice(0, 80) });
  loadMyReports();
});

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

// ============================================================ MY REPORTS
async function loadMyReports() {
  const section = document.getElementById("my-reports");
  const body = document.getElementById("my-reports-body");
  if (!section || !body) return;
  try {
    const origin = await resolveApiOrigin();
    const r = await fetch(`${origin}/hazards/history`, { credentials: "include" });
    if (r.status === 401 || r.status === 403) {
      section.hidden = true;
      return;
    }
    if (!r.ok) throw new Error(`http ${r.status}`);
    const data = await r.json();
    const items = data.items || [];
    section.hidden = false;
    if (!items.length) {
      body.innerHTML = '<tr><td colspan="6" class="my-reports-empty">No reports yet. Submit a hazard and it will show up here.</td></tr>';
      return;
    }
    body.innerHTML = items.map(it => {
      const reported = (it.reported_at || "").replace("T", " ").replace("Z", " UTC");
      const dec = it.agent_decision || "pending";
      const stat = (it.status || "open").toLowerCase();
      const label = HAZARD_CLASS_LABELS[it.class] || it.class || "unknown";
      return `<tr>
        <td>${escapeHtml(reported)}</td>
        <td>${escapeHtml(label)}</td>
        <td>${escapeHtml(it.road_type || "-")}</td>
        <td>${it.priority ?? "-"}</td>
        <td><span class="my-reports-decision ${escapeHtml(dec)}">${escapeHtml(dec)}</span></td>
        <td><span class="my-reports-status ${escapeHtml(stat)}">${escapeHtml(stat)}</span></td>
      </tr>`;
    }).join("");
  } catch (err) {
    section.hidden = true;
  }
}

// ============================================================ HERO SLIDESHOW
function wireHeroSlideshow() {
  const slides = document.querySelectorAll(".hero-slide");
  if (!slides.length) return;
  let i = 0;
  const update = () => {
    const next = (i + 1) % slides.length;
    slides[i].classList.remove("hero-slide--active");
    slides[next].classList.add("hero-slide--active");
    i = next;
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
  // Legacy function kept so other call sites still work. The new UX uses
  // refreshUserChrome() which swaps the whole header based on sign-in state.
  const label = document.getElementById("signin-label");
  if (label) {
    const payload = parseJwt();
    if (payload && payload.email) {
      label.textContent = payload.email.split("@")[0];
    } else if (payload && payload.name) {
      label.textContent = payload.name;
    } else {
      label.textContent = "Sign in";
    }
  }
  // Also run the full chrome refresh (sign-in button vs profile+logout).
  refreshUserChrome().catch(() => {});
}

// Step 2 (25.10.2026): swap the header between signed-out ("Sign in" button)
// and signed-in (Profile dropdown button + separate Logout button). On
// sign-in it also pre-fetches /auth/me so the profile dropdown opens with
// fresh counts.
async function refreshUserChrome() {
  const navSignin = document.getElementById("nav-signin");
  const navProfile = document.getElementById("nav-profile");
  const navLogout = document.getElementById("nav-logout");
  if (!navSignin || !navProfile || !navLogout) return;
  const signed = isSignedIn();
  navSignin.hidden = signed;
  navProfile.hidden = !signed;
  navLogout.hidden = !signed;
  if (!signed) {
    state.me = null;
    return;
  }
  // Fetch /auth/me and cache; the dropdown reads from state.me on open.
  // If the call fails on network error (backend unreachable, DNS failure,
  // tunnel down, CORS block) the token in localStorage is still formally
  // valid but useless, and the old UX showed the stale username as if the
  // user were signed in. Treat any failure as "backend says no" and sign
  // the user out locally so the Sign-in button comes back.
  try {
    const me = await fetchMe();
    state.me = me;
    const profileLabel = document.getElementById("nav-profile-label");
    if (profileLabel) {
      profileLabel.textContent = me.username || (me.email || "").split("@")[0] || "Profile";
    }
  } catch (e) {
    clearToken();
    state.me = null;
    navSignin.hidden = false;
    navProfile.hidden = true;
    navLogout.hidden = true;
  }
}

function _populateProfileDropdown() {
  const me = state.me || {};
  const setText = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text == null ? "-" : String(text);
  };
  setText("profile-username", me.username || me.name || me.sub || "(unknown)");
  setText("profile-email", me.email || "(none)");
  const cap = me.daily_cap || 100;
  setText("profile-uploads-today", `${me.uploads_today || 0} / ${cap}`);
  setText("profile-uploads-total", `${me.uploads_total || 0}`);
  const adminBlock = document.getElementById("profile-pop-admin");
  const urlsList = document.getElementById("profile-operator-urls");
  if (me.is_admin && me.operator_urls && me.operator_urls.length) {
    adminBlock.hidden = false;
    urlsList.innerHTML = me.operator_urls.map((u) =>
      `<li><a href="${u}" target="_blank" rel="noopener noreferrer">${u}</a></li>`
    ).join("");
  } else {
    adminBlock.hidden = true;
    urlsList.innerHTML = "";
  }
}

function wireNavProfile() {
  const btn = document.getElementById("nav-profile");
  const pop = document.getElementById("profile-pop");
  const backdrop = document.getElementById("profile-pop-backdrop");
  const closeBtn = document.getElementById("profile-pop-close");
  if (!btn || !pop) return;
  btn.addEventListener("click", async () => {
    await refreshUserChrome();
    _populateProfileDropdown();
    pop.hidden = false;
  });
  if (backdrop) backdrop.addEventListener("click", () => { pop.hidden = true; });
  if (closeBtn) closeBtn.addEventListener("click", () => { pop.hidden = true; });
}

function wireNavLogout() {
  const btn = document.getElementById("nav-logout");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    await logoutCall();
    state.me = null;
    await refreshUserChrome();
    const pop = document.getElementById("profile-pop");
    if (pop) pop.hidden = true;
    toast("Logged out");
    if (state.view === "live") stopLive();
  });
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

// ============================================================ SIGN IN (step 2)
// GitHub / Hugging Face style single-card modal: Google button on top,
// email+password below the "or" divider, "Sign up" flip link at bottom
// (shown only when the server enables signup).
let _authConfigCache = null;
async function getAuthConfig() {
  if (_authConfigCache) return _authConfigCache;
  try {
    _authConfigCache = await fetchAuthConfig();
  } catch {
    _authConfigCache = {
      signup_enabled: false, password_login_enabled: true,
      google_login_enabled: false, max_users: 10, user_count: 0,
    };
  }
  return _authConfigCache;
}

let _signinMode = "signin";  // "signin" or "signup"

function _setSigninMode(mode) {
  _signinMode = mode;
  const signinForm = document.getElementById("signin-form-signin");
  const signupForm = document.getElementById("signin-form-signup");
  const title = document.getElementById("signin-title");
  const subtitle = document.getElementById("signin-subtitle");
  const flipPrompt = document.getElementById("signin-flip-prompt");
  const flipBtn = document.getElementById("signin-flip-btn");
  const err = document.getElementById("signin-error");
  const brandTitle = document.querySelector(".signin-brand h2");
  if (mode === "signup") {
    if (signinForm) signinForm.hidden = true;
    if (signupForm) signupForm.hidden = false;
    if (brandTitle) brandTitle.textContent = "Create your account";
    if (subtitle) subtitle.textContent = "Join RoadGuard with a username and password.";
    if (flipPrompt) flipPrompt.textContent = "Already have an account?";
    if (flipBtn) flipBtn.textContent = "Sign in";
    setTimeout(() => {
      const el = document.getElementById("signup-username");
      if (el) el.focus();
    }, 50);
  } else {
    if (signinForm) signinForm.hidden = false;
    if (signupForm) signupForm.hidden = true;
    if (brandTitle) brandTitle.textContent = "Sign in to RoadGuard";
    if (subtitle) subtitle.textContent = "Welcome back. Sign in to continue reporting hazards.";
    if (flipPrompt) flipPrompt.textContent = "Don't have an account?";
    if (flipBtn) flipBtn.textContent = "Sign up";
    setTimeout(() => {
      const el = document.getElementById("signin-username");
      if (el) el.focus();
    }, 50);
  }
  if (err) { err.hidden = true; err.textContent = ""; }
}

function _wirePasswordToggle(btnId, inputId) {
  const btn = document.getElementById(btnId);
  const input = document.getElementById(inputId);
  if (!btn || !input) return;
  btn.addEventListener("click", () => {
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    btn.textContent = show ? "Hide" : "Show";
    btn.setAttribute("aria-pressed", show ? "true" : "false");
    btn.setAttribute("aria-label", show ? "Hide password" : "Show password");
  });
}

// GSI's rendered button lives off-screen; our styled "Continue with Google"
// button programmatically triggers Google's One-Tap prompt or clicks the
// hidden GSI button, so the UX is a single visible button that matches
// GitHub / Hugging Face.
let _gsiReady = false;
function _triggerGoogleSignIn() {
  const err = document.getElementById("signin-error");
  if (!_gsiReady || !window.google || !window.google.accounts || !window.google.accounts.id) {
    if (err) {
      err.hidden = false;
      err.textContent = "Google sign-in is not configured on this server yet.";
    }
    return;
  }
  // Prefer the hidden GSI button so the user sees a consistent native
  // consent sheet. If that fails, fall back to One-Tap prompt().
  const nativeBtn = document.querySelector(
    "#google-signin-button div[role='button'], #google-signin-button iframe",
  );
  if (nativeBtn && typeof nativeBtn.click === "function") {
    nativeBtn.click();
    return;
  }
  try {
    window.google.accounts.id.prompt();
  } catch (e) {
    if (err) { err.hidden = false; err.textContent = _friendlyAuthError(e); }
  }
}

function wireSignIn() {
  const err = document.getElementById("signin-error");
  const backdrop = document.getElementById("signin-backdrop");

  // Sign-in form
  const signinForm = document.getElementById("signin-form-signin");
  signinForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    err.hidden = true; err.textContent = "";
    const username = document.getElementById("signin-username").value.trim();
    const password = document.getElementById("signin-password").value;
    try {
      if (!username) {
        await loginWithPassphrase(password);  // legacy
      } else {
        await loginWithUsernamePassword(username, password);
      }
      await refreshUserChrome();
      closeSignIn();
      runPending();
      loadMyReports();
      toast("Signed in");
    } catch (e) {
      err.hidden = false;
      err.textContent = _friendlyAuthError(e);
    }
  });

  // Sign-up form (hidden by default; shown when _setSigninMode('signup'))
  const signupForm = document.getElementById("signin-form-signup");
  signupForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    err.hidden = true; err.textContent = "";
    const username = document.getElementById("signup-username").value.trim();
    const password = document.getElementById("signup-password").value;
    const email = document.getElementById("signup-email").value.trim();
    try {
      await registerAccount(username, password, email);
      await refreshUserChrome();
      closeSignIn();
      runPending();
      loadMyReports();
      toast("Account created, signed in");
    } catch (e) {
      err.hidden = false;
      err.textContent = _friendlyAuthError(e);
    }
  });

  // Flip link signin <-> signup
  const flipBtn = document.getElementById("signin-flip-btn");
  if (flipBtn) {
    flipBtn.addEventListener("click", () => {
      _setSigninMode(_signinMode === "signin" ? "signup" : "signin");
    });
  }

  // Password visibility toggles
  _wirePasswordToggle("signin-pw-toggle", "signin-password");
  _wirePasswordToggle("signup-pw-toggle", "signup-password");

  // Our styled Google button triggers the hidden GSI widget
  const googleBtn = document.getElementById("btn-google");
  if (googleBtn) googleBtn.addEventListener("click", _triggerGoogleSignIn);

  backdrop.addEventListener("click", () => closeSignIn());

  // Mount the real GSI widget off-screen so our styled button can delegate to it
  mountGoogleSignIn("google-signin-button", async (credential) => {
    err.hidden = true;
    try {
      await loginWithGoogleIdToken(credential);
      await refreshUserChrome();
      closeSignIn();
      runPending();
      loadMyReports();
      toast("Signed in with Google");
    } catch (e) {
      err.hidden = false;
      err.textContent = _friendlyAuthError(e);
    }
  });
  _gsiReady = true;
}

function _friendlyAuthError(e) {
  const msg = (e && e.message) ? e.message : "Sign-in failed";
  const map = {
    "invalid_credentials": "Wrong username or password.",
    "signup_disabled": "Signup is closed on this server.",
    "password_login_disabled": "Password login is disabled on this server.",
    "username_taken": "That username is already taken.",
    "user_cap_reached": "User limit reached on this server.",
    "password_too_short": "Password must be at least 8 characters.",
    "password_missing_uppercase": "Password must include an uppercase letter.",
    "password_missing_lowercase": "Password must include a lowercase letter.",
    "password_missing_special": "Password must include a special character.",
    "username_invalid_format": "Username: 3-32 chars, letters/digits/_.-, must start with letter/digit/_.",
    "daily_submit_cap_reached (100/day)": "Daily upload cap reached - try tomorrow.",
  };
  for (const [key, friendly] of Object.entries(map)) {
    if (msg.includes(key)) return friendly;
  }
  return msg;
}

async function openSignIn() {
  const modal = document.getElementById("signin-modal");
  modal.hidden = false;
  const cfg = await getAuthConfig();

  // Social block: show whole block only if Google is enabled on the server.
  const socialBlock = document.getElementById("signin-social");
  if (socialBlock) socialBlock.hidden = !cfg.google_login_enabled;
  const googleNote = document.getElementById("signin-google-note");
  if (googleNote) googleNote.hidden = !!cfg.google_login_enabled;

  // Flip link (signin <-> signup) only when signup is enabled.
  const flipWrap = document.getElementById("signin-flip-wrap");
  if (flipWrap) flipWrap.hidden = !cfg.signup_enabled;

  _setSigninMode("signin");
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
      // bbox overlay disabled per operator preference 24.10: the citizen
      // should not see the detector's class label on the image (fear of
      // the model picking the wrong object and misleading the reporter).
      // Detections are still computed and sent to the server as metadata.
      // _overlayBoxes(ctx, result.detections, canvas.width);
      // Internals-hidden citizen UX (per operator preference 24.10):
      // the on-device YOLO detections are STILL run (they feed the server
      // payload + bbox overlay suppressed on the canvas) but we do not
      // surface class / confidence / count to the citizen. All the citizen
      // sees before Send is a neutral message. The verdict after Send is
      // likewise minimal (thanks + known-hazard flag only, see
      // renderUploadVerdict / CITIZEN_COPY).
      show(`<div class="upload-verdict-empty">Image received. Please fill in the details below and press <em>Send to Agent</em>.</div>`);
      submitBtn.disabled = false;
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
  const risk = document.getElementById("upload-risk").value;
  const location = (document.getElementById("upload-location").value || "").trim();
  const reporterText = (document.getElementById("upload-reporter-text").value || "").trim();
  const det = uploadState.lastDetections;
  if (!det) return;
  // All 5 context fields are REQUIRED before a case is created on the server.
  // Partial submits produced thin cases the agent could not reason about.
  const missing = [];
  if (!roadType) missing.push("Road type");
  if (!laneBlocked) missing.push("Blocks a traffic lane?");
  if (!risk) missing.push("Risk level");
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
  // Resize + JPEG-encode the uploaded image client-side so the request
  // payload stays small and the server never has to receive the original
  // 20 MB phone photo. Max dimension 1600 px, quality 0.8 -> ~300-500 KB
  // typical. The server rejects anything over 5 MB raw.
  let imageJpegBase64 = null;
  try {
    const srcCanvas = canvas;
    const maxDim = 1600;
    let w = srcCanvas.width, h = srcCanvas.height;
    if (w > maxDim || h > maxDim) {
      const scale = Math.min(maxDim / w, maxDim / h);
      w = Math.round(w * scale);
      h = Math.round(h * scale);
    }
    const outCanvas = document.createElement("canvas");
    outCanvas.width = w;
    outCanvas.height = h;
    outCanvas.getContext("2d").drawImage(srcCanvas, 0, 0, w, h);
    imageJpegBase64 = outCanvas.toDataURL("image/jpeg", 0.8);
    // 5 MB raw -> ~6.9 MB base64 (incl. "data:image/jpeg;base64," prefix)
    if (imageJpegBase64.length > 7_000_000) {
      verdictBox.innerHTML = `<div class="upload-verdict-empty" style="color:#b91c1c;font-weight:600">The compressed image is still larger than 5 MB. Pick a smaller photo (max 5 MB after resize).</div>`;
      submitBtn.disabled = false;
      return;
    }
  } catch (err) {
    // Submission must not proceed without an image: the operator relies
    // on it for the daily digest and for same-street comparisons.
    verdictBox.innerHTML = `<div class="upload-verdict-empty" style="color:#b91c1c;font-weight:600">Could not read the image file (${(err && err.message) || err}). Pick another photo.</div>`;
    submitBtn.disabled = false;
    return;
  }
  if (!imageJpegBase64) {
    verdictBox.innerHTML = `<div class="upload-verdict-empty" style="color:#b91c1c;font-weight:600">An image is required for submission. Please pick a photo first.</div>`;
    submitBtn.disabled = false;
    return;
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
    // The citizen-perceived risk level ("small"|"moderate"|"dangerous") is
    // appended to reporter_text as a tagged suffix the signal_extractor can
    // pick up. We piggy-back on reporter_text instead of adding a brand-new
    // field so the HazardSubmit server schema stays compatible with older
    // clients that do not know about the risk dropdown.
    reporter_text: `[risk=${risk}] ${reporterText}`,
    image_sha256: fingerprints.sha256,
    image_phash: fingerprints.phash,
    image_jpeg_base64: imageJpegBase64,
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
  const resumingHazardId = pending ? pending.hazard_id : null;
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
    } else if (pending && resumingHazardId) {
      // Evidence resume is a synchronous call that can run 2-3 minutes
      // while the resumed graph re-executes; the Cloudflare quick tunnel
      // caps a single HTTP request at ~100 s and returns 502/504. The
      // agent is still running on the server, so fall over to polling
      // /report/{hazard_id} the same way the async submit does. If the
      // agent paused for another round of evidence (unlikely - cap is
      // one per case), the pending request stays. Otherwise clear it.
      verdictBox.innerHTML = `<div class="upload-verdict-empty">Server took the photo but kept the connection too long (the tunnel caps at 100s). Waiting for the resumed agent to finish...</div>`;
      try {
        const pausedForEvidence = await pollAsyncHazard(resumingHazardId, verdictBox);
        if (!pausedForEvidence) clearPendingEvidence();
      } catch (pollErr) {
        verdictBox.innerHTML = `<div class="upload-verdict-empty">Could not reach the server after the second photo was sent. The agent may still be finishing on its own; check back in a minute.</div>`;
      }
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
      const trace = report.trace || [];
      const steps = trace.length;
      const queuePos = Number(report.queue_position || 0);
      const spin = `<div class="rg-spinner" aria-hidden="true" style="display:inline-block;width:16px;height:16px;border:2px solid #cbd5e1;border-top-color:#0ea5e9;border-radius:50%;animation:rg-spin 0.9s linear infinite;vertical-align:middle;margin-right:8px"></div>`;
      // Translate the last agent step into a plain-language status line so
      // the user sees progress ("Reading your photo", "Looking up municipal
      // standards", ...) instead of a dry step counter.
      const AGENT_STATUS = {
        sanitize: "Checking your submission",
        signal_extractor: "Reading your description",
        supervisor: "Routing the case",
        profile_builder: "Analysing the photo",
        rag_retriever: "Looking up municipal standards",
        decision: "Deciding the verdict",
        approval_gate: "Opening a supervisor approval row",
        await_evidence: "Waiting for more evidence",
      };
      const last = steps > 0 ? trace[steps - 1] : null;
      const stage = (last && AGENT_STATUS[last.speaker]) || "Starting the agents";
      const stageHtml = `<div style="margin-top:6px;font-size:13px;color:var(--text-mid)">Stage ${steps}/7: <b>${stage}</b></div>`;
      if (v.status === "queued") {
        const posTxt = queuePos > 0 ? ` (position ${queuePos} in queue)` : "";
        verdictBox.innerHTML = `<div class="upload-verdict-empty">${spin}Queued behind another case${posTxt}... ${elapsed}s elapsed.<div style="margin-top:6px;font-size:13px;color:var(--text-mid)">Local open-source agents on an ARM CPU, so each case takes 1-3 minutes.</div></div>`;
      } else {
        verdictBox.innerHTML = `<div class="upload-verdict-empty">${spin}Working... ${elapsed}s elapsed.${stageHtml}<div style="margin-top:4px;font-size:12px;color:var(--text-mid)">Five local open-source agents, no cloud API, no data leaves the city server.</div></div>`;
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

// CITIZEN VIEW copy only: short, objective, friendly. No priority, no
// rationale text, no agent notes, no cited policy codes. All of those
// stay internal and feed the supervisor digest + approval page; the
// citizen just needs to know their report landed and whether it is
// already a known hazard.
const CITIZEN_COPY = {
  urgent: {
    headline: "Thank you - your report has been received",
    body:     "A municipal supervisor has been notified and will take it from here.",
  },
  standard: {
    headline: "Thank you - your report has been received",
    body:     "The municipality will handle it as part of regular road maintenance.",
  },
  needs_more_evidence: {
    headline: "Thank you - one more photo would help",
    body:     "The image was a bit unclear. Please send one more photo of the same spot from a slightly different angle so the team can look at it properly.",
  },
  human_review: {
    headline: "Thank you - your report has been received",
    body:     "A municipal supervisor will take a look at this one personally.",
  },
  known_image: {
    headline: "Thank you - we already know about this one",
    body:     "This hazard has already been reported and is being tracked. No action needed from you.",
  },
};

function renderUploadVerdict(resp) {
  const action = resp.verdict.action;
  const level = LEVEL_BY_ACTION[action] || "info";
  const copy  = CITIZEN_COPY[action] || {
    headline: "Thank you - your report has been received",
    body:     "The municipality will review it.",
  };
  const caseRef = (resp.hazard_id || "").slice(-8);
  const timeNow = new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

  // The internal fields (priority, rationale, cited standards, agent note)
  // ARE kept on the server and reach the supervisor through the daily digest
  // and the approval page. They are intentionally not rendered here: the
  // citizen does not need them and showing them creates noise.
  const bodyText = action === "needs_more_evidence"
    ? String((resp.evidence_request && resp.evidence_request.message) || copy.body)
    : copy.body;

  const html = `
    <div class="upload-verdict-card" data-level="${level}">
      <div class="upload-verdict-head">
        <span class="upload-verdict-action">${copy.headline}</span>
      </div>
      <div class="upload-verdict-rationale" style="margin-top:8px">${bodyText.replace(/[<>]/g, "")}</div>
      <div class="upload-verdict-footer">
        <span>Reference <strong>#${caseRef}</strong></span>
        <span>Received at ${timeNow}</span>
      </div>
    </div>`;
  document.getElementById("upload-verdict").innerHTML = html;
}

function wireNavSignIn() {
  // Signed-out: show the signin modal. Signed-in: open the profile
  // dropdown (user asked for separate Login and Logout buttons; this one
  // handles only the signed-out path).
  const btn = document.getElementById("nav-signin");
  if (!btn) return;
  btn.addEventListener("click", () => {
    if (!isSignedIn()) {
      openSignIn();
    }
  });
}
