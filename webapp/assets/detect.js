// On-device detection. Loads onnxruntime-web and the YOLO ONNX model
// lazily. When the model file is not present the detector falls back
// to a deterministic mock, so the UI shell is testable without a
// trained network.

import { CONFIG } from "./config.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/+esm";

export class YoloDetector {
  constructor(opts = {}) {
    this.modelUrl = opts.modelUrl || CONFIG.YOLO_MODEL_URL;
    this.inputSize = opts.inputSize || CONFIG.YOLO_INPUT_SIZE || 640;
    this.classNames = opts.classNames || CONFIG.YOLO_CLASS_NAMES;
    this.confThreshold =
      opts.confidenceThreshold ?? CONFIG.YOLO_CONFIDENCE_THRESHOLD ?? 0.35;
    this.session = null;
    this.ort = null;
    this.mode = "unloaded";
  }

  async load() {
    try {
      const head = await fetch(this.modelUrl, { method: "HEAD" });
      if (!head.ok) {
        this.mode = "mock";
        return;
      }
    } catch {
      this.mode = "mock";
      return;
    }
    try {
      const ort = await import(ORT_URL);
      this.ort = ort;
      ort.env.wasm.numThreads = Math.max(
        1,
        Math.min(4, (navigator.hardwareConcurrency || 2) - 1)
      );
      ort.env.wasm.simd = true;
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      this.mode = "ort";
    } catch (err) {
      console.warn("YoloDetector: ORT load failed, falling back to mock", err);
      this.mode = "mock";
    }
  }

  async detect(canvas) {
    if (this.mode === "unloaded") await this.load();
    if (this.mode === "mock") {
      return this.#detectMock(canvas);
    }
    return this.#detectOrt(canvas);
  }

  #detectMock(canvas) {
    // One deterministic bbox centered on the frame, class D40, moderate
    // confidence. Enough to drive the metadata form and the submit
    // flow without a real model.
    const w = canvas.width;
    const h = canvas.height;
    const bx = Math.round(w * 0.35);
    const by = Math.round(h * 0.45);
    const bw = Math.round(w * 0.3);
    const bh = Math.round(h * 0.25);
    return {
      model: "mock",
      input_size: this.inputSize,
      detections: [
        {
          x1: bx,
          y1: by,
          x2: bx + bw,
          y2: by + bh,
          class: "D40",
          confidence: 0.62,
        },
      ],
    };
  }

  async #detectOrt(canvas) {
    const { tensor, scale, pad } = this.#preprocess(canvas);
    const feeds = { images: tensor };
    const output = await this.session.run(feeds);
    const first = Object.values(output)[0];
    const data = first.data;
    const dims = first.dims; // typically [1, 84, 8400] or [1, N, 6]
    const raw = this.#postprocess(data, dims, scale, pad, canvas.width, canvas.height);
    return {
      model: "yolov8",
      input_size: this.inputSize,
      detections: this.#nms(raw, 0.5),
    };
  }

  #preprocess(canvas) {
    const size = this.inputSize;
    const scale = Math.min(size / canvas.width, size / canvas.height);
    const sw = Math.round(canvas.width * scale);
    const sh = Math.round(canvas.height * scale);
    const padX = Math.floor((size - sw) / 2);
    const padY = Math.floor((size - sh) / 2);

    const letterbox = document.createElement("canvas");
    letterbox.width = size;
    letterbox.height = size;
    const ctx = letterbox.getContext("2d");
    ctx.fillStyle = "#727272";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(canvas, padX, padY, sw, sh);

    const imageData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(size * size * 3);
    for (let i = 0, j = 0; i < imageData.length; i += 4, j += 1) {
      float[j] = imageData[i] / 255;
      float[j + size * size] = imageData[i + 1] / 255;
      float[j + 2 * size * size] = imageData[i + 2] / 255;
    }
    const tensor = new this.ort.Tensor("float32", float, [1, 3, size, size]);
    return { tensor, scale, pad: { x: padX, y: padY } };
  }

  #postprocess(data, dims, scale, pad, origW, origH) {
    // yolov8 output in [1, 4 + numClasses, N] layout.
    const numClasses = this.classNames.length;
    const stride = dims[2];
    const bboxes = [];
    for (let i = 0; i < stride; i += 1) {
      let bestClass = -1;
      let bestConf = 0;
      for (let c = 0; c < numClasses; c += 1) {
        const v = data[(4 + c) * stride + i];
        if (v > bestConf) {
          bestConf = v;
          bestClass = c;
        }
      }
      if (bestConf < this.confThreshold || bestClass < 0) continue;
      const cx = data[0 * stride + i];
      const cy = data[1 * stride + i];
      const w = data[2 * stride + i];
      const h = data[3 * stride + i];
      const x1 = (cx - w / 2 - pad.x) / scale;
      const y1 = (cy - h / 2 - pad.y) / scale;
      const x2 = (cx + w / 2 - pad.x) / scale;
      const y2 = (cy + h / 2 - pad.y) / scale;
      bboxes.push({
        x1: Math.max(0, x1),
        y1: Math.max(0, y1),
        x2: Math.min(origW, x2),
        y2: Math.min(origH, y2),
        class: this.classNames[bestClass],
        confidence: Number(bestConf.toFixed(3)),
      });
    }
    return bboxes;
  }

  #nms(boxes, iouThreshold = 0.5) {
    const sorted = boxes.slice().sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    for (const b of sorted) {
      let suppress = false;
      for (const k of kept) {
        if (k.class !== b.class) continue;
        if (iou(b, k) > iouThreshold) {
          suppress = true;
          break;
        }
      }
      if (!suppress) kept.push(b);
    }
    return kept;
  }
}

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

export function drawBoxes(canvas, source, detections) {
  const ctx = canvas.getContext("2d");
  canvas.width = source.width;
  canvas.height = source.height;
  ctx.drawImage(source, 0, 0, source.width, source.height);
  ctx.lineWidth = Math.max(2, Math.round(source.width / 400));
  ctx.font = `${Math.max(14, Math.round(source.width / 50))}px system-ui, sans-serif`;
  ctx.textBaseline = "top";
  for (const d of detections) {
    ctx.strokeStyle = "#ff8800";
    ctx.fillStyle = "rgba(255, 136, 0, 0.85)";
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const label = `${d.class} ${(d.confidence * 100).toFixed(0)}%`;
    const metrics = ctx.measureText(label);
    const padding = 4;
    const textH = parseInt(ctx.font, 10) + padding;
    ctx.fillRect(d.x1, Math.max(0, d.y1 - textH), metrics.width + padding * 2, textH);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, d.x1 + padding, Math.max(0, d.y1 - textH + padding / 2));
  }
}

export function topDetection(detections) {
  if (!detections || !detections.length) return null;
  return detections.reduce((best, d) =>
    d.confidence > (best ? best.confidence : 0) ? d : best,
  null);
}
