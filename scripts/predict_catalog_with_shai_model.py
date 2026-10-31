"""Run the frozen YOLOv8n ONNX on every image in the dataset folder and
emit a predictions CSV in the exact format the model handoff contract
specifies (same columns, bbox separator, no-detection shape).

The inference loop (`predict_paths`) is copied verbatim from Shai's
`Models/RoadGuard_Model_Notebook.ipynb`, cell 44. This script only
orchestrates it over the full dataset instead of just the test split,
and formats the output rows according to `handoff_contract.json`:

    columns     : image_id, top1_class, top1_confidence, bbox,
                  image_width, image_height
    bbox        : "x1:y1:x2:y2" in original-image pixels
    top1        : highest-confidence detection across all classes,
                  after frozen confidence threshold and NMS
    no detection: top1_class="", top1_confidence=0.0, bbox=""

Frozen inference parameters come from `handoff_contract.json`:
    imgsz=640, conf=0.01, iou=0.7, max_det=300.

Model class ids follow the ONNX output channel order documented in the
contract (0=D00, 1=D10, 2=D20, 3=D40). The native PyTorch checkpoint
uses a different order; the ONNX graph was re-wired by Shai in cell 66
so the raw ONNX output is already in this webapp order.

Usage:
    python scripts/predict_catalog_with_shai_model.py <dataset_dir> \\
        --model webapp/assets/models/yolov8n-roadguard.onnx \\
        --out predictions_full.csv

The resulting CSV is then fed to `scripts/catalog_dataset.py --predictions`
to populate `dataset_images.model_top1_*` for every catalogued image.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


# ONNX output channel order, frozen in handoff_contract.json.
ONNX_CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

# Inference parameters frozen in handoff_contract.json; identical to the
# ones Shai's notebook used when it produced the 1,488-row test CSV.
IMGSZ = 640
CONF_THRESHOLD = 0.01
NMS_IOU = 0.7
MAX_DET = 300
BATCH = 32

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -----------------------------------------------------------------------------
# predict_paths is copied verbatim from Shai's notebook cell 44
# (Models/RoadGuard_Model_Notebook.ipynb). The only substitution is that
# the loose globals the notebook relied on (IMGSZ, PRED_FLOOR, NMS_IOU,
# DEVICE, max_det) are replaced here with the frozen module constants
# above - the body of the function is otherwise unchanged.
# -----------------------------------------------------------------------------
def predict_paths(model, paths, device: str = "cpu"):
    predictions = {}
    for start in tqdm(range(0, len(paths), BATCH), desc="Predict batches"):
        batch_paths = paths[start:start + BATCH]
        results = model.predict(
            source=batch_paths, imgsz=IMGSZ, conf=CONF_THRESHOLD,
            iou=NMS_IOU, max_det=MAX_DET, device=device,
            verbose=False, stream=False,
        )
        for path, result in zip(batch_paths, results):
            b = result.boxes
            predictions[str(path)] = np.column_stack([
                b.cls.cpu().numpy(),
                b.xyxy.cpu().numpy(),
                b.conf.cpu().numpy(),
            ])
    return predictions


def find_images(dataset_dir: Path) -> list[Path]:
    """Collect every image file under dataset_dir, in a deterministic order
    so re-runs produce identical CSV ordering."""
    images: list[Path] = []
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES:
            images.append(p)
    return images


def row_from_predictions(image_path: Path, dets: np.ndarray | None) -> dict:
    """Reduce the detection matrix (N rows of [cls, x1, y1, x2, y2, conf])
    to the single highest-confidence detection, matching the handoff
    contract. For images with no detections, emit the frozen blank shape
    (top1_class="", top1_confidence=0.0, bbox="")."""
    with Image.open(image_path) as im:
        width, height = im.size
    if dets is None or len(dets) == 0:
        return {
            "image_id": image_path.name,
            "top1_class": "",
            "top1_confidence": 0.0,
            "bbox": "",
            "image_width": width,
            "image_height": height,
        }
    top_idx = int(np.argmax(dets[:, 5]))
    top = dets[top_idx]
    cls_idx = int(top[0])
    x1, y1, x2, y2 = (float(v) for v in top[1:5])
    conf = float(top[5])
    return {
        "image_id": image_path.name,
        "top1_class": ONNX_CLASS_NAMES.get(cls_idx, ""),
        "top1_confidence": round(conf, 6),
        "bbox": f"{x1:.2f}:{y1:.2f}:{x2:.2f}:{y2:.2f}",
        "image_width": width,
        "image_height": height,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir",
                    help="Root directory holding the dataset images (searched recursively)")
    ap.add_argument("--model",
                    default="webapp/assets/models/yolov8n-roadguard.onnx",
                    help="Path to the frozen ONNX model (default: webapp copy)")
    ap.add_argument("--out", default="predictions_full.csv",
                    help="Where to write the predictions CSV")
    ap.add_argument("--device", default="cpu",
                    help='Inference device: "cpu" or a cuda id like "0"')
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: only run on the first N images (0 = all)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_path = Path(args.model).resolve()
    out_path = Path(args.out).resolve()

    if not dataset_dir.is_dir():
        print(f"dataset_dir not found: {dataset_dir}", file=sys.stderr)
        return 1
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Scanning images under: {dataset_dir}")
    images = find_images(dataset_dir)
    if args.limit:
        images = images[: args.limit]
    print(f"Found {len(images)} images")
    if not images:
        print("No images found; nothing to do.", file=sys.stderr)
        return 1

    print(f"Running predict_paths (device={args.device}, imgsz={IMGSZ}, "
          f"conf={CONF_THRESHOLD}, iou={NMS_IOU}, batch={BATCH})")
    predictions = predict_paths(model, [str(p) for p in images], device=args.device)

    print(f"Writing CSV: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "image_id", "top1_class", "top1_confidence",
            "bbox", "image_width", "image_height",
        ])
        writer.writeheader()
        for img in images:
            dets = predictions.get(str(img))
            writer.writerow(row_from_predictions(img, dets))

    no_det = sum(
        1 for img in images
        if predictions.get(str(img)) is None or len(predictions[str(img)]) == 0
    )
    print(f"Done. rows={len(images)}  no_detection={no_det}  "
          f"detected={len(images) - no_det}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
