// Screen router + flow controller for the nine RoadGuard screens.

import { CONFIG } from "./config.js";
import {
  clearToken,
  fetchJson,
  fetchMe,
  isSignedIn,
  loginWithGoogleIdToken,
  loginWithPassphrase,
  mountGoogleSignIn,
  resolveApiOrigin,
} from "./auth.js";
import { CameraController, requestCameraPermission } from "./camera.js";
import { YoloDetector, drawBoxes, topDetection } from "./detect.js";

const SCREENS = [
  "signin",
  "capture",
  "detect",
  "verdict",
  "history",
  "settings",
];

const state = {
  camera: null,
  detector: new YoloDetector(),
  lastSnapshot: null,
  lastDetections: [],
  lastVerdict: null,
};

const SETTINGS_KEY = "roadguard.settings.v1";

function loadLocalSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveLocalSettings(obj) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(obj));
  } catch {
    // ignore
  }
}

function show(screen) {
  for (const name of SCREENS) {
    const el = document.getElementById(`screen-${name}`);
    if (!el) continue;
    el.classList.toggle("active", name === screen);
  }
  for (const btn of document.querySelectorAll(".nav-btn")) {
    btn.classList.toggle("active", btn.dataset.screen === screen);
  }
  if (screen === "capture") {
    startCamera();
  } else if (state.camera) {
    state.camera.stop();
  }
  if (screen === "history") {
    loadHistory();
  }
}

async function startCamera() {
  const video = document.getElementById("camera-video");
  const select = document.getElementById("camera-select");
  const hint = document.getElementById("capture-hint");
  const captureBtn = document.getElementById("capture-btn");
  if (!state.camera) {
    state.camera = new CameraController(video, select);
  }
  state.camera.setUseFront(Boolean(loadLocalSettings().use_front_camera));
  try {
    const ok = await requestCameraPermission();
    if (!ok) {
      hint.textContent = "Camera permission denied.";
      return;
    }
    await state.camera.start(select.value || "");
    hint.textContent = "Camera ready. Point at the hazard and capture.";
    captureBtn.disabled = false;
  } catch (err) {
    hint.textContent = `Could not open camera: ${err.message || err}.`;
    captureBtn.disabled = true;
  }
}

async function runDetection() {
  const hint = document.getElementById("capture-hint");
  const list = document.getElementById("detect-list");
  const canvas = document.getElementById("detect-canvas");
  const snapshot = state.camera.snapshot();
  state.lastSnapshot = snapshot;
  hint.textContent = "Running on-device detection.";
  const result = await state.detector.detect(snapshot.canvas);
  state.lastDetections = result.detections;
  drawBoxes(canvas, snapshot.canvas, result.detections);
  list.innerHTML = "";
  if (!result.detections.length) {
    list.innerHTML = `<p class="muted">No damage detected. Try a different angle.</p>`;
  }
  for (const d of result.detections) {
    const row = document.createElement("div");
    row.className = "detect-row";
    row.innerHTML = `
      <span class="cls">${d.class}</span>
      <span class="conf">${(d.confidence * 100).toFixed(0)}%</span>
    `;
    list.appendChild(row);
  }
  const modeHint = result.model === "mock" ? " (mock detector - drop a YOLO ONNX in assets/models to replace)" : "";
  hint.textContent = `Detected ${result.detections.length} hazard(s).${modeHint}`;
  show("detect");
}

async function submitHazard(ev) {
  ev.preventDefault();
  const form = ev.currentTarget;
  const errBox = document.getElementById("submit-error");
  const hint = document.getElementById("submit-hint");
  errBox.hidden = true;
  errBox.textContent = "";
  const top = topDetection(state.lastDetections);
  if (!top) {
    errBox.hidden = false;
    errBox.textContent = "No detection to submit. Retake the photo.";
    return;
  }
  const fd = new FormData(form);
  const body = {
    class: top.class,
    confidence: top.confidence,
    bboxes: state.lastDetections.map((d) => [
      d.x1,
      d.y1,
      d.x2,
      d.y2,
      d.class,
      d.confidence,
    ]),
    road_type: fd.get("road_type"),
    lane_blocked: fd.get("lane_blocked"),
    location: fd.get("location") || "",
    image_object_key: `pending/${crypto.randomUUID()}.jpg`,
  };
  hint.textContent = "Submitting.";
  try {
    const verdict = await fetchJson("/hazards/submit", {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.lastVerdict = verdict;
    renderVerdict(verdict);
    show("verdict");
  } catch (err) {
    errBox.hidden = false;
    errBox.textContent = `Submit failed: ${err.message || err}`;
  } finally {
    hint.textContent = "";
  }
}

function renderVerdict(verdict) {
  const card = document.getElementById("verdict-card");
  const v = verdict.verdict || {};
  card.className = "";
  card.classList.add(v.action || "human_review");
  const cited = (v.cited_sources || [])
    .map((u) => `<li><a href="${u}" target="_blank" rel="noopener">${u}</a></li>`)
    .join("");
  const action = (v.action || "human_review").replace(/_/g, " ");
  card.innerHTML = `
    <h3>${action.toUpperCase()}</h3>
    <p>${v.rationale || ""}</p>
    <p class="priority">Priority: ${v.priority ?? "n/a"} / 100</p>
    ${cited ? `<p class="muted">Cited procedures:</p><ul>${cited}</ul>` : ""}
    ${
      verdict.fallback_reason
        ? `<p class="muted">Fallback: ${verdict.fallback_reason}</p>`
        : ""
    }
  `;
}

async function loadHistory() {
  const list = document.getElementById("history-list");
  list.innerHTML = `<p class="muted">Loading.</p>`;
  try {
    const data = await fetchJson("/hazards/history");
    const items = data.items || [];
    if (!items.length) {
      list.innerHTML = `<p class="muted">No reports yet.</p>`;
      return;
    }
    list.innerHTML = "";
    for (const row of items) {
      const el = document.createElement("div");
      el.className = "history-row";
      el.innerHTML = `
        <div>
          <strong>${row.class || row.hazard_class}</strong>
          <span class="badge badge-${row.agent_decision}">${row.agent_decision.replace(/_/g, " ")}</span>
          <div class="meta">${row.reported_at} - ${row.road_type} / ${row.lane_blocked}</div>
        </div>
        <div class="meta">priority ${row.priority}</div>
      `;
      list.appendChild(el);
    }
  } catch (err) {
    if (err.status === 501) {
      list.innerHTML = `<p class="muted">History will be available once the database layer is wired to the VM.</p>`;
    } else {
      list.innerHTML = `<p class="error">${err.message || err}</p>`;
    }
  }
}

async function signInViaPassphrase(ev) {
  ev.preventDefault();
  const errBox = document.getElementById("signin-error");
  errBox.hidden = true;
  const value = ev.currentTarget.passphrase.value;
  try {
    await loginWithPassphrase(value);
    afterSignIn();
  } catch (err) {
    errBox.hidden = false;
    errBox.textContent = err.message || "Sign-in failed.";
  }
}

async function afterSignIn() {
  document.getElementById("nav").hidden = false;
  show("capture");
  try {
    const me = await fetchMe();
    document.getElementById("brand-sub")?.replaceChildren(
      document.createTextNode(me.email || "signed in")
    );
  } catch {
    // non-fatal
  }
}

function signOut() {
  clearToken();
  if (state.camera) state.camera.stop();
  document.getElementById("nav").hidden = true;
  show("signin");
}

function wireNav() {
  for (const btn of document.querySelectorAll(".nav-btn")) {
    btn.addEventListener("click", () => show(btn.dataset.screen));
  }
  document.getElementById("passphrase-form").addEventListener("submit", signInViaPassphrase);
  document.getElementById("capture-btn").addEventListener("click", runDetection);
  document.getElementById("detect-back").addEventListener("click", () => show("capture"));
  document.getElementById("metadata-form").addEventListener("submit", submitHazard);
  document.getElementById("verdict-new").addEventListener("click", () => show("capture"));
  document.getElementById("verdict-history").addEventListener("click", () => show("history"));
  document.getElementById("history-refresh").addEventListener("click", loadHistory);
  document.getElementById("signout-btn").addEventListener("click", signOut);

  const settings = loadLocalSettings();
  const form = document.getElementById("settings-form");
  form.daily_email.checked = Boolean(settings.daily_email);
  form.use_front_camera.checked = Boolean(settings.use_front_camera);
  document.getElementById("settings-save").addEventListener("click", () => {
    const next = {
      daily_email: form.daily_email.checked,
      use_front_camera: form.use_front_camera.checked,
    };
    saveLocalSettings(next);
    document.getElementById("settings-hint").textContent = "Saved locally.";
    setTimeout(
      () => (document.getElementById("settings-hint").textContent = ""),
      1500
    );
  });

  mountGoogleSignIn("google-signin-container", async (credential) => {
    const errBox = document.getElementById("signin-error");
    errBox.hidden = true;
    try {
      await loginWithGoogleIdToken(credential);
      afterSignIn();
    } catch (err) {
      errBox.hidden = false;
      errBox.textContent = err.message || "Google sign-in failed.";
    }
  });
}

async function boot() {
  await resolveApiOrigin();
  wireNav();
  if (isSignedIn()) {
    afterSignIn();
  } else {
    show("signin");
  }
}

boot();
