// getUserMedia camera controller with environment-facing default.

import { CONFIG } from "./config.js";

export class CameraController {
  constructor(video, select) {
    this.video = video;
    this.select = select;
    this.stream = null;
    this.devices = [];
    this.useFront = false;
  }

  setUseFront(flag) {
    this.useFront = Boolean(flag);
  }

  async enumerateDevices() {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      this.devices = all.filter((d) => d.kind === "videoinput");
    } catch {
      this.devices = [];
    }
    if (!this.select) return;
    this.select.innerHTML = "";
    for (const d of this.devices) {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Camera ${d.deviceId.slice(0, 6)}`;
      this.select.appendChild(opt);
    }
  }

  async start(preferredDeviceId = "") {
    this.stop();
    const facingMode = this.useFront ? "user" : CONFIG.CAMERA_DEFAULT_FACING_MODE;
    const constraints = {
      audio: false,
      video: preferredDeviceId
        ? { deviceId: { exact: preferredDeviceId } }
        : { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
    };
    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;
    await this.video.play().catch(() => {});
    await this.enumerateDevices();
  }

  stop() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
  }

  snapshot() {
    const canvas = document.createElement("canvas");
    const w = this.video.videoWidth || 1280;
    const h = this.video.videoHeight || 720;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(this.video, 0, 0, w, h);
    return { canvas, width: w, height: h };
  }
}

export async function requestCameraPermission() {
  try {
    const s = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    for (const t of s.getTracks()) t.stop();
    return true;
  } catch {
    return false;
  }
}
