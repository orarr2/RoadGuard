"""Fetch a roboflow dataset to a local directory for backup and offline work.

Shai's training pipeline downloads from roboflow directly; this script is
a safety net so the RoadGuard VM keeps its own copy. By default it pulls
`xjsj/road-damage-10k` v1 in YOLOv8 format and writes a manifest with
the class list so the webapp's YOLO_CLASS_NAMES can be synced.

Usage::

    export ROBOFLOW_API_KEY=rf_xxx
    python -m tools.download_dataset \\
        --workspace xjsj --project road-damage-10k --version 1 \\
        --dest /opt/roadguard/datasets/road-damage-10k

The roboflow SDK is a soft dependency. If it is not installed the script
prints the pip command and exits non-zero without trying anything.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


DEFAULT_WORKSPACE = "xjsj"
DEFAULT_PROJECT = "road-damage-10k"
DEFAULT_VERSION = 1
DEFAULT_FORMAT = "yolov8"
DEFAULT_DEST = "/opt/roadguard/datasets/road-damage-10k"


def _die(msg: str, code: int = 1) -> None:
    print(f"download_dataset: {msg}", file=sys.stderr)
    sys.exit(code)


def _ensure_sdk() -> object:
    try:
        import roboflow  # type: ignore
    except ImportError:
        _die(
            "roboflow SDK not installed. Run:\n"
            "  /opt/roadguard/venv/bin/pip install roboflow\n"
            "or add roboflow to a requirements-dataset.txt for Shai's pipeline."
        )
    return roboflow  # type: ignore


def _write_manifest(dest: Path, workspace: str, project: str, version: int) -> None:
    classes_path = dest / "data.yaml"
    classes: list[str] = []
    if classes_path.exists():
        for line in classes_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("names"):
                raw = line.split(":", 1)[1].strip()
                if raw.startswith("[") and raw.endswith("]"):
                    classes = [c.strip().strip("'\"") for c in raw[1:-1].split(",")]
                break
    manifest = {
        "workspace": workspace,
        "project": project,
        "version": version,
        "format": DEFAULT_FORMAT,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": classes,
        "note": (
            "If this class list differs from the agent enum (D00, D10, D20, D40), "
            "update webapp/assets/config.js YOLO_CLASS_NAMES to match the order "
            "of this list; the API enum stays the same."
        ),
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _free_bytes(path: Path) -> int:
    while not path.exists():
        path = path.parent
        if path == path.parent:
            break
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return 0


def download(
    workspace: str,
    project: str,
    version: int,
    dest: Path,
    api_key: str,
) -> None:
    free = _free_bytes(dest)
    if free and free < 8 * 1024 * 1024 * 1024:
        print(
            f"warning: only {free // (1024*1024)} MB free at {dest.parent}; "
            "a 10K-image YOLO dataset is typically 4-6 GB.",
            file=sys.stderr,
        )
    dest.mkdir(parents=True, exist_ok=True)
    rf = _ensure_sdk().Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    v = proj.version(version)
    print(f"fetching {workspace}/{project}:{version} ({DEFAULT_FORMAT}) -> {dest}")
    v.download(DEFAULT_FORMAT, location=str(dest), overwrite=True)
    _write_manifest(dest, workspace, project, version)
    print("done")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backup a roboflow dataset.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--dest", type=Path, default=Path(DEFAULT_DEST))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="roboflow API key (default: $ROBOFLOW_API_KEY)",
    )
    args = parser.parse_args(argv)
    if not args.api_key:
        _die(
            "no API key: set ROBOFLOW_API_KEY in /etc/roadguard/env or pass "
            "--api-key. Each roboflow account has a key under Settings > API."
        )
    download(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        dest=args.dest,
        api_key=args.api_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
