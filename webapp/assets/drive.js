// Live driving controller. Owns the per-session UUID, the back-camera
// loop, an IoU tracker that collapses many frames into one submit per
// confirmed hazard, and a toast queue that surfaces verdicts without
// stealing the user's attention.

import { fetchJson } from "./auth.js";
import { CameraController, requestCameraPermission } from "./camera.js";
import { YoloDetector, drawBoxes } from "./detect.js";

const DEFAULT_FPS = 2;
const DEFAULT_STABLE_HITS = 3;
const DEFAULT_IOU_THRESHOLD = 0.4;
const DEFAULT_MAX_AGE_MS = 2200;
const DEFAULT_CONFIDENCE_FLOOR = 0.4;

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const aa = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bb = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / Math.max(1e-6, aa + bb - inter);
}

export class IoUTracker {
  constructor({
    iouThreshold = DEFAULT_IOU_THRESHOLD,
    stableHits = DEFAULT_STABLE_HITS,
    maxAgeMs = DEFAULT_MAX_AGE_MS,
    confidenceFloor = DEFAULT_CONFIDENCE_FLOOR,
  } = {}) {
    this.tracks = [];
    this.nextId = 1;
    this.iouThreshold = iouThreshold;
    this.stableHits = stableHits;
    this.maxAgeMs = maxAgeMs;
    this.confidenceFloor = confidenceFloor;
  }

  update(detections, nowTs) {
    this.tracks = this.tracks.filter(
      (t) => nowTs - t.lastSeen <= this.maxAgeMs,
    );
    const matched = new Set();
    for (const d of detections) {
      if (d.confidence < this.confidenceFloor) continue;
      let best = null;
      let bestIou = 0;
      for (const t of this.tracks) {
        if (matched.has(t.id)) continue;
        if (t.class !== d.class) continue;
        const score = iou(d, t);
        if (score > bestIou) {
          bestIou = score;
          best = t;
        }
      }
      if (best && bestIou >= this.iouThreshold) {
        best.x1 = d.x1;
        best.y1 = d.y1;
        best.x2 = d.x2;
        best.y2 = d.y2;
        best.confidence = Math.max(best.confidence, d.confidence);
        if (d.area_ratio !== undefined) best.area_ratio = d.area_ratio;
        if (d.position?.center_y_rel !== undefined) {
          best.center_y_rel = d.position.center_y_rel;
        }
        best.hits += 1;
        best.lastSeen = nowTs;
        matched.add(best.id);
      } else {
        this.tracks.push({
          id: `t-${this.nextId++}`,
          class: d.class,
          confidence: d.confidence,
          area_ratio: d.area_ratio ?? 0,
          center_y_rel: d.position?.center_y_rel ?? null,
          x1: d.x1,
          y1: d.y1,
          x2: d.x2,
          y2: d.y2,
          hits: 1,
          firstSeen: nowTs,
          lastSeen: nowTs,
          submitted: false,
        });
      }
    }
  }

  readyToSubmit() {
    return this.tracks.filter((t) => t.hits >= this.stableHits && !t.submitted);
  }

  activeTracks() {
    return this.tracks.slice();
  }
}

export class LiveDriver {
  constructor({
    video,
    overlayCanvas,
    detector = null,
    fps = DEFAULT_FPS,
    getMetadata = () => ({ road_type: "main", lane_blocked: "none" }),
  } = {}) {
    this.sessionId = crypto.randomUUID();
    this.video = video;
    this.overlayCanvas = overlayCanvas;
    this.camera = new CameraController(video, null);
    this.detector = detector || new YoloDetector();
    this.tracker = new IoUTracker();
    this.fps = fps;
    this.loopHandle = null;
    this.getMetadata = getMetadata;
    this.listeners = { verdict: [], tick: [], error: [], state: [] };
    this.running = false;
    this.detectorReady = false;
    this.frameCount = 0;
    this.lastFrameAt = 0;
  }

  on(event, fn) {
    (this.listeners[event] = this.listeners[event] || []).push(fn);
  }

  _emit(event, payload) {
    for (const fn of this.listeners[event] || []) {
      try {
        fn(payload);
      } catch (err) {
        console.warn(`listener for ${event} threw`, err);
      }
    }
  }

  async start() {
    if (this.running) return;
    const perm = await requestCameraPermission();
    if (!perm) {
      this._emit("error", { code: "camera_denied" });
      return;
    }
    await this.camera.start("");
    if (!this.detectorReady) {
      await this.detector.load();
      this.detectorReady = true;
    }
    this.running = true;
    this._emit("state", { running: true, mode: this.detector.mode });
    const period = Math.max(100, Math.round(1000 / this.fps));
    const tick = () => {
      if (!this.running) return;
      this._tick().finally(() => {
        if (this.running) this.loopHandle = setTimeout(tick, period);
      });
    };
    this.loopHandle = setTimeout(tick, period);
  }

  stop() {
    this.running = false;
    if (this.loopHandle) clearTimeout(this.loopHandle);
    this.loopHandle = null;
    this.camera.stop();
    this._emit("state", { running: false });
  }

  async _tick() {
    if (!this.camera.stream) return;
    const snap = this.camera.snapshot();
    const result = await this.detector.detect(snap.canvas);
    const now = Date.now();
    this.frameCount += 1;
    this.lastFrameAt = now;
    this.tracker.update(result.detections, now);
    if (this.overlayCanvas) {
      drawBoxes(this.overlayCanvas, snap.canvas, result.detections);
    }
    this._emit("tick", {
      frameCount: this.frameCount,
      detections: result.detections,
      tracks: this.tracker.activeTracks(),
      model: result.model,
    });
    const ready = this.tracker.readyToSubmit();
    for (const track of ready) {
      track.submitted = true;
      this._submit(track, snap, result).catch((err) => {
        console.warn("live submit failed", err);
        track.submitted = false;
        this._emit("error", { code: "submit_failed", message: String(err) });
      });
    }
  }

  async _submit(track, snap, detResult) {
    const meta = this.getMetadata() || {};
    const body = {
      session_id: this.sessionId,
      track_id: track.id,
      class: track.class,
      confidence: Math.min(0.999, Math.max(0.0, track.confidence)),
      top1_margin: detResult.top1_margin ?? 0,
      area_ratio: track.area_ratio ?? 0,
      n_detections_by_class: detResult.n_detections_by_class || {},
      bbox: [track.x1, track.y1, track.x2, track.y2],
      center_y_rel: track.center_y_rel ?? 0.5,
      road_type: meta.road_type || "main",
      lane_blocked: meta.lane_blocked || "none",
      location: meta.location || "",
    };
    const resp = await fetchJson("/hazards/live", {
      method: "POST",
      body: JSON.stringify(body),
    });
    this._emit("verdict", { track, response: resp });
  }
}
