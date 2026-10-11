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


def _parse_data_yaml_classes(yaml_text: str) -> tuple[list[str], int]:
    """Return (ordered class names, nc). Handles the three shapes roboflow
    emits: single-line inline list, multi-line YAML list, and dict form.
    """
    classes: list[str] = []
    nc = 0
    lines = yaml_text.splitlines()
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("nc:"):
            try:
                nc = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                nc = 0
            continue
        if stripped.startswith("names:"):
            rest = stripped.split(":", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):
                classes = [c.strip().strip("'\"") for c in rest[1:-1].split(",")]
                continue
            # multi-line forms: either a YAML sequence (`- name`) or a mapping
            # (`0: name`) starting with the next line.
            j = i + 1
            while j < len(lines):
                lraw = lines[j]
                l = lraw.rstrip()
                if not l.strip():
                    j += 1
                    continue
                if not (lraw.startswith(" ") or lraw.startswith("\t") or lraw.startswith("-")):
                    break
                ls = l.strip()
                if ls.startswith("- "):
                    classes.append(ls[2:].strip().strip("'\""))
                elif ":" in ls:
                    _, name = ls.split(":", 1)
                    classes.append(name.strip().strip("'\""))
                j += 1
    return classes, nc


def _write_manifest(dest: Path, workspace: str, project: str, version: int) -> None:
    classes_path = dest / "data.yaml"
    classes: list[str] = []
    nc = 0
    if classes_path.exists():
        classes, nc = _parse_data_yaml_classes(classes_path.read_text(encoding="utf-8"))
    manifest = {
        "workspace": workspace,
        "project": project,
        "version": version,
        "format": DEFAULT_FORMAT,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": classes,
        "nc": nc,
        "agent_class_map": _suggest_class_map(classes),
        "note": (
            "The agent API enum stays D00/D10/D20/D40. agent_class_map "
            "maps each dataset class (indexed by its position in classes) "
            "to the agent enum; apply it to the YOLO output in "
            "webapp/assets/detect.js before posting to /hazards/live."
        ),
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


_RDD_SYNONYMS = {
    # dataset label (lowered, no non-alphanum) -> RDD2022 enum code
    "longitudinalcracks": "D00",
    "longitudinalcrack": "D00",
    "longitudinal": "D00",
    "transversecracks": "D10",
    "transversecrack": "D10",
    "transverse": "D10",
    "alligatorcracks": "D20",
    "alligatorcrack": "D20",
    "alligator": "D20",
    "crocodilecracks": "D20",
    "potholes": "D40",
    "pothole": "D40",
    "d00": "D00",
    "d10": "D10",
    "d20": "D20",
    "d40": "D40",
}


def _suggest_class_map(classes: list[str]) -> dict[str, str]:
    """Map each dataset class name to the agent enum (D00/D10/D20/D40)
    using a small synonym table. Unknown labels map to "" so the caller
    can decide (usually: route to D20 by default, or flag the label).
    """
    out: dict[str, str] = {}
    for name in classes:
        key = "".join(c.lower() for c in name if c.isalnum())
        out[name] = _RDD_SYNONYMS.get(key, "")
    return out


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
