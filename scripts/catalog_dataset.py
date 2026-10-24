"""Rebuild the dataset_images catalogue from a dataset folder.

usage:
    python scripts/catalog_dataset.py <dataset_dir> [--db PATH] [--predictions CSV]

The dataset folder is a Roboflow/YOLO layout with <split>/images/*.jpg next
to <split>/labels/*.txt. The CSV is the per-image model predictions
(image_id, top1_class, top1_confidence, bbox). When the CSV is omitted the
rows are catalogued without a recorded prediction.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir")
    ap.add_argument("--db", default=os.environ.get("ROADGUARD_DB_PATH", "roadguard.db"))
    ap.add_argument("--predictions", default=None)
    args = ap.parse_args()

    # The agent module reads POLICY_PATH at import time; the dataset_images
    # table lives in the same DB.
    from roadguard_agent import catalog_dataset

    stats = catalog_dataset(args.dataset_dir, args.db, predictions_csv=args.predictions)
    print("dataset catalogued:", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
