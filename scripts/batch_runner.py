"""
Batch tester for the RoadGuard agent pipeline.

Walks every .jpg/.jpeg/.png in /tmp/batch_test, runs the YOLOv8n ONNX
detector locally (numpy + onnxruntime - no ultralytics required), builds
the same hazard dict the webapp would send to /hazards/submit, calls
multiagent_decide() straight on the pipeline, and appends one JSON line
per case to /tmp/batch_results.jsonl.

Usage:  sudo -u roadguard /opt/roadguard/venv/bin/python3 batch_runner.py
Progress is also printed to stdout so the operator can follow with
`journalctl -f` or `tail -F /tmp/batch_runner.log`.
"""
import os, sys, json, time, uuid, hashlib, pathlib, base64, io
# Run this script via `systemd-run --unit=rg-batch -p EnvironmentFile=/etc/roadguard/env ...`
# so the service's env (provider, JWT, RAG, etc.) is already in os.environ
# before roadguard_agent imports; otherwise LLM calls fall back to the
# offline stub and the whole test is a no-op.
sys.path.insert(0, "/opt/roadguard/repo")
import numpy as np
import onnxruntime as ort
from PIL import Image

BATCH_DIR = pathlib.Path(os.environ.get("BATCH_DIR", "/tmp/batch_test"))
RESULTS = pathlib.Path("/tmp/batch_results.jsonl")
MODEL = "/opt/roadguard/repo/webapp/assets/models/yolov8n-roadguard.onnx"
CLASSES = ["D00", "D10", "D20", "D40"]
SIZE = 640
# Low threshold so the test actually exercises the pipeline on realistic
# detections instead of the zero-conf fast-path for every image.
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5

from roadguard_agent import dhash_image, create_jwt  # type: ignore
from roadguard_api import get_settings  # type: ignore
import urllib.request, urllib.error

# Submit through the live HTTP API over localhost so every hazard also
# lands in the hazards table and shows up on the /trace page the way a
# real user submission would.
API_BASE = "http://127.0.0.1:8000"
USER_SUB = "user:u_batch_internal"  # the token subject of a registered user; hazards land under u_batch_internal
USER_EMAIL = "batch@internal"
USER_NAME = "batch"


def _http_json(method: str, path: str, token: str, body: dict | None = None, timeout: int = 300) -> dict:
    data = None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(API_BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def letterbox(img: Image.Image, size: int = SIZE):
    w, h = img.size
    scale = min(size / w, size / h)
    sw, sh = round(w * scale), round(h * scale)
    pad_x = (size - sw) // 2
    pad_y = (size - sh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    resized = img.resize((sw, sh), Image.BILINEAR)
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, :, :, :]
    return arr, scale, (pad_x, pad_y), (w, h)


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-6, aa + bb - inter)


def nms(boxes, iou_thr=IOU_THRESHOLD):
    boxes = sorted(boxes, key=lambda b: -b["confidence"])
    kept = []
    for b in boxes:
        if any(k["class"] == b["class"] and iou((b["x1"], b["y1"], b["x2"], b["y2"]),
                                                 (k["x1"], k["y1"], k["x2"], k["y2"])) > iou_thr
               for k in kept):
            continue
        kept.append(b)
    return kept


def postprocess(out, scale, pad, orig_w, orig_h):
    # out: [1, 4+numC, N]
    out = out[0]  # [4+numC, N]
    n = out.shape[1]
    boxes = []
    cls_scores = out[4:4 + len(CLASSES), :]
    best_cls = cls_scores.argmax(axis=0)
    best_conf = cls_scores.max(axis=0)
    for i in range(n):
        if best_conf[i] < CONF_THRESHOLD:
            continue
        cx, cy, w, h = out[0, i], out[1, i], out[2, i], out[3, i]
        x1 = (cx - w/2 - pad[0]) / scale
        y1 = (cy - h/2 - pad[1]) / scale
        x2 = (cx + w/2 - pad[0]) / scale
        y2 = (cy + h/2 - pad[1]) / scale
        boxes.append({
            "x1": float(max(0, x1)), "y1": float(max(0, y1)),
            "x2": float(min(orig_w, x2)), "y2": float(min(orig_h, y2)),
            "class": CLASSES[int(best_cls[i])],
            "confidence": round(float(best_conf[i]), 3),
        })
    return nms(boxes)



def _image_as_jpeg_b64(path):
    """Open any supported format with PIL (jpg/png/webp/avif/jfif),
    convert to JPEG, return a base64 string. Server caps raw at 5MB."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82, optimize=True)
            data = buf.getvalue()
        if len(data) > 4_900_000:
            return None
        return base64.b64encode(data).decode("ascii")
    except Exception as exc:
        print(f"[BATCH   ] image encode failed for {path.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None


def build_hazard(image_path: pathlib.Path, detections: list, w: int, h: int) -> dict:
    """Build the body for /hazards/submit-async (same shape the webapp sends)."""
    sha = hashlib.sha256(image_path.read_bytes()).hexdigest().lower()
    phash, iw, ih = dhash_image(image_path)
    det_list = detections or []
    top1 = max(det_list, key=lambda d: d["confidence"]) if det_list else {}
    # HazardSubmit.bboxes is list-of-list-rows [x1, y1, x2, y2, class, conf].
    bboxes = [[d["x1"], d["y1"], d["x2"], d["y2"], d["class"], d["confidence"]] for d in det_list]
    area_ratio = 0.0
    if top1:
        area = (top1["x2"] - top1["x1"]) * (top1["y2"] - top1["y1"])
        area_ratio = area / max(1.0, w * h)
    counts = {}
    for d in det_list:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    return {
        "hazard_class": top1.get("class") or "D40",
        "confidence": float(top1.get("confidence") or 0.0),
        "bboxes": bboxes,
        "road_type": "main",
        "lane_blocked": "none",
        "location": f"batch test ({image_path.name})",
        "top1_margin": 0.0,
        "area_ratio": float(area_ratio),
        "n_detections_by_class": counts,
        "center_y_rel": 0.5,
        "blur_score": 150.0,
        "is_road_scene": True,
        "exposure": "ok",
        "ocr_text_detected": "",
        "reporter_text": "",
        "image_sha256": sha,
        "image_phash": phash,
        "image_jpeg_base64": _image_as_jpeg_b64(image_path),
    }


def main() -> int:
    print("[BATCH   ] =====================================================", flush=True)
    print(f"[BATCH   ] loading YOLO model {MODEL}", flush=True)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    settings = get_settings()
    token = create_jwt(sub=USER_SUB, email=USER_EMAIL, display_name=USER_NAME, settings=settings)
    print(f"[BATCH   ] JWT issued for {USER_EMAIL}; submitting via {API_BASE}/hazards/submit-async", flush=True)
    # Image formats PIL/Pillow decodes safely into a bitmap canvas.
    # No SVG/PDF/HTML/XML - those are text/vector containers that can
    # carry executable payloads; keep them out of the pipeline.
    SAFE_SUFFIXES = (
        ".jpg", ".jpeg", ".jfif",      # JPEG family (JFIF is JPEG under a different extension)
        ".png",                         # lossless raster
        ".webp", ".avif",               # modern lossy
        ".bmp", ".gif", ".tif", ".tiff",# older raster
        ".heic", ".heif",               # iPhone (needs pillow-heif but we try regardless)
    )
    files = sorted([p for p in BATCH_DIR.iterdir()
                    if p.suffix.lower() in SAFE_SUFFIXES])

    def _category(name: str) -> str:
        n = name.lower()
        if n.startswith("firc_") or "roadcrack" in n:
            return "baseline (RDD2022 catalog)"
        if n.startswith("web_"):
            return "web - real citizen report (Bing search)"
        return "unknown source (user web pick)"

    print(f"[BATCH   ] {len(files)} images queued. starting pass. tail with: sudo journalctl -u rg-batch -f -o cat", flush=True)
    print("[BATCH   ] =====================================================", flush=True)
    RESULTS.unlink(missing_ok=True)
    with RESULTS.open("w") as out_fh:
        for i, path in enumerate(files, 1):
            cat = _category(path.name)
            print("", flush=True)
            print("[BATCH   ] =====================================================", flush=True)
            print(f"[BATCH   ] IMAGE {i}/{len(files)}  file={path.name}  type={cat}", flush=True)
            print("[BATCH   ] -----------------------------------------------------", flush=True)
            t0 = time.time()
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:
                print(f"[BATCH   ] IMAGE {i} SKIP: cannot open ({exc})", flush=True)
                continue
            print(f"[BATCH   ] IMAGE {i} step 1/3  running YOLO on {img.size[0]}x{img.size[1]}", flush=True)
            arr, scale, pad, (w, h) = letterbox(img)
            yolo_out = sess.run(None, {inp_name: arr})[0]
            detections = postprocess(yolo_out, scale, pad, w, h)
            det_summary = ", ".join(f"{d['class']}@{d['confidence']}" for d in detections) or "no detections above threshold"
            print(f"[BATCH   ] IMAGE {i} step 1/3  YOLO -> {det_summary}", flush=True)
            hazard = build_hazard(path, detections, w, h)
            print(f"[BATCH   ] IMAGE {i} step 2/3  POST /hazards/submit-async (shows up on /trace page as a real user hazard)", flush=True)
            hazard_id = ""
            verdict = {}
            try:
                resp = _http_json("POST", "/hazards/submit-async", token, hazard)
                hazard_id = resp.get("hazard_id") or ""
                print(f"[BATCH   ] IMAGE {i} step 2/3  accepted hazard_id={hazard_id}, polling until DONE...", flush=True)
                # Poll /report every 3s until status is final. A
                # needs_more_evidence verdict is answered the way a
                # reporter would, with the same upload, so the paused
                # graph resumes and the case ends in a real verdict instead
                # of waiting 72 h for evidence no batch can send.
                evidence_answered = False
                while True:
                    time.sleep(3)
                    rep = _http_json("GET", f"/report/{hazard_id}", token)
                    v = rep.get("verdict") or {}
                    st = v.get("status")
                    ev = rep.get("evidence_request") or {}
                    if st == "waiting_for_evidence" and ev.get("id") and not evidence_answered:
                        evidence_answered = True
                        # Round 17 (1.11.2026): the same photo alone is
                        # rejected (422, not evidence); the batch has no
                        # second photo, so it answers the way a reporter
                        # without one would: the same upload plus a note.
                        print(f"[BATCH   ] IMAGE {i} step 2b   agent asked for evidence ({ev.get('missing_field')}): answering with the same upload and a note that no other photo exists", flush=True)
                        try:
                            answer = {**hazard, "reporter_text": "No other photo is available, this is the only picture I took."}
                            _http_json("POST", f"/evidence_requests/{ev['id']}/fulfill", token, answer, timeout=420)
                        except Exception as exc:
                            print(f"[BATCH   ] IMAGE {i} step 2b   evidence answer failed: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    if st not in ("queued", "processing", "waiting_for_evidence") or (
                        st == "waiting_for_evidence" and not ev.get("id")
                    ):
                        verdict = {
                            "action": v.get("action"),
                            "priority": v.get("priority"),
                            "rationale": (v.get("rationale") or "")[:300],
                            "cited_sources": list(rep.get("grounded_sources") or []),
                            "status": st,
                            "evidence_answered": evidence_answered,
                        }
                        break
                    if time.time() - t0 > 420:
                        verdict = {"action": "TIMEOUT", "rationale": "poll timeout after 7 min"}
                        break
                dt = time.time() - t0
                print(f"[BATCH   ] IMAGE {i} step 3/3  VERDICT: action={verdict.get('action')}  priority={verdict.get('priority')}  elapsed={dt:.1f}s", flush=True)
                if verdict.get("rationale"):
                    print(f"[BATCH   ] IMAGE {i} rationale: {verdict['rationale'][:160]}", flush=True)
            except Exception as exc:
                dt = time.time() - t0
                verdict = {"action": "ERROR", "rationale": f"{type(exc).__name__}: {exc}"[:300]}
                print(f"[BATCH   ] IMAGE {i} step 3/3  ERROR: {type(exc).__name__}: {exc}", flush=True)
            record = {
                "idx": i,
                "image": path.name,
                "category": cat,
                "hazard_id": hazard_id,
                "elapsed_s": round(dt, 1),
                "yolo_count": len(detections),
                "yolo_top1_conf": hazard["confidence"],
                "yolo_top1_class": hazard["hazard_class"],
                "verdict": verdict,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            print(f"[BATCH   ] IMAGE {i}/{len(files)} DONE ({dt:.1f}s)  total so far: {i} images", flush=True)
    print("[BATCH   ] =====================================================", flush=True)
    print("[BATCH   ] ALL DONE - run `jq -c .verdict.action /tmp/batch_results.jsonl | sort | uniq -c` for the breakdown", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
