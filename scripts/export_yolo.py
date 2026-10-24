"""Export an Ultralytics YOLO .pt checkpoint to ONNX for the webapp.

The webapp loads the ONNX model via onnxruntime-web in the browser. The
file lives at `webapp/assets/models/yolov8n-roadguard.onnx` and is NOT
committed to the repo (binary ~15-30MB). This script converts a .pt file
into that path.

Expected source models (team hand-off):
  * `Ayus44/road_damage_detection_yolo` - YOLOv8 fine-tune (64MB .pt)
    https://huggingface.co/Ayus44/road_damage_detection_yolo
  * `cvtechniques/road-damage-detection-yolov11` - YOLOv11s, RDD2022
    classes [Alligator, Longitudinal, Potholes, Transverse] matching
    the webapp's YOLO_CLASS_NAMES=["D20","D00","D40","D10"] order.
    (Weights not yet published in the repo as of 2026-10; follow the
    repo for the eventual .pt.)
  * Ovad's final `.pt` - replaces either of the above once delivered.

Usage:
  python scripts/export_yolo.py --src /path/to/best.pt \\
    --dst webapp/assets/models/yolov8n-roadguard.onnx \\
    --imgsz 640

If `ultralytics` is not installed, the script prints the pip command
and exits non-zero. Converting is a one-shot setup task; the webapp
does the inference at runtime without touching ultralytics.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_DST = Path("webapp/assets/models/yolov8n-roadguard.onnx")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="path to .pt or hf://owner/repo/best.pt")
    parser.add_argument("--dst", default=str(DEFAULT_DST), help="output .onnx path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--half", action="store_true", help="FP16 export (smaller, needs ONNX runtime support)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics not installed. Run:\n"
            "  pip install ultralytics onnx onnxruntime\n",
            file=sys.stderr,
        )
        return 2

    src = Path(args.src)
    if not src.is_file():
        print(f"src not found: {src}", file=sys.stderr)
        return 1

    print(f"Loading {src} ...")
    model = YOLO(str(src))

    print(f"Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset}, half={args.half}) ...")
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, half=args.half)
    # ultralytics returns the path written next to the .pt
    exported = Path(exported)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, dst)
    print(f"Written {dst} ({dst.stat().st_size / (1024*1024):.1f} MB)")
    print("\nNext steps:")
    print("  * Confirm webapp/assets/config.js YOLO_CLASS_NAMES matches the model's class order.")
    print("  * Reload the webapp; detect.js will switch from mock to live inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
