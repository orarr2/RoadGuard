"""Bridge between the FastAPI shell and the 3-agent graph.

Phase-1 of the migration: the API's /hazards/submit still ships a flat
hazard dict (per the current HazardSubmit pydantic model). This adapter
reshapes it into the graph's three input blocks (yolo_output,
image_quality, metadata), runs run_graph_locally, and reshapes the
final state back into the AgentDecision dataclass the API already
consumes. The API contract therefore does not change.

The current HazardSubmit shape has no image-quality fields (blur, OCR,
is_road_scene) and no top1_margin on YOLO. The adapter supplies safe
defaults so the Perception node still produces a reasonable confidence
label; later phases should push these fields into HazardSubmit itself
when the browser-side pipeline starts emitting them.
"""

from __future__ import annotations

from typing import Any

from agent.loop import AgentDecision

from .graph import run_graph_locally


def hazard_to_graph_inputs(hazard: dict[str, Any]) -> tuple[str, dict, dict, dict]:
    """Map a flat HazardSubmit-style dict to the graph's three input blocks.

    Returns (hazard_id, yolo_output, image_quality, metadata).
    """
    hazard_id = hazard["hazard_id"]
    hazard_class = hazard["class"]
    confidence = float(hazard["confidence"])
    bboxes = hazard.get("bboxes") or []

    detections = _bboxes_to_detections(bboxes, hazard_class, confidence)
    yolo_output = {
        "top1_class": hazard_class,
        "top1_confidence": confidence,
        "top1_margin": 0.0,  # not emitted by current client; perception has a default
        "detections": detections,
        "n_detections_by_class": _count_by_class(detections),
    }

    image_quality = {
        "blur_score": 150.0,
        "brightness_mean": 0.5,
        "is_road_scene": True,
        "road_scene_confidence": 0.9,
        "occlusion_estimate": "low",
        "ocr_text_detected": "",
    }

    metadata = {
        "road_type": hazard["road_type"],
        "lane_blocked": hazard["lane_blocked"],
        "location": hazard.get("location") or "",
        "reported_at": hazard.get("reported_at", ""),
    }
    return hazard_id, yolo_output, image_quality, metadata


def _bboxes_to_detections(
    bboxes: list[list[Any]],
    fallback_class: str,
    fallback_conf: float,
) -> list[dict]:
    """Translate the API's [x1,y1,x2,y2,class,conf] rows to graph detections.

    When bboxes is empty, synthesize a single detection from the top-level
    class and confidence so downstream nodes still have something to work on.
    """
    if not bboxes:
        return [{"class": fallback_class, "confidence": fallback_conf, "area_ratio": 0.0}]

    detections: list[dict] = []
    for row in bboxes:
        if len(row) < 4:
            continue
        try:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        except (TypeError, ValueError):
            continue
        cls = row[4] if len(row) >= 5 else fallback_class
        conf = float(row[5]) if len(row) >= 6 else fallback_conf
        # area_ratio needs image dims; API doesn't carry them yet so leave 0.
        detections.append({
            "class": cls,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "area_ratio": 0.0,
        })

    if not detections:
        detections.append(
            {"class": fallback_class, "confidence": fallback_conf, "area_ratio": 0.0}
        )
    return detections


def _count_by_class(detections: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
    for d in detections:
        cls = d.get("class")
        if cls in counts:
            counts[cls] += 1
    return counts


def final_state_to_decision(final_state: dict[str, Any]) -> AgentDecision:
    """Translate the graph's final state back to the API's AgentDecision."""
    guard_fired = bool(final_state.get("guard_fired"))
    return AgentDecision(
        action=final_state.get("action", "human_review"),
        rationale=final_state.get("rationale", ""),
        priority=int(final_state.get("priority", 0)),
        cited_sources=list(final_state.get("cited_sources") or []),
        provider="multiagent",
        trace=list(final_state.get("trace") or []),
        fallback_reason="guard_fired" if guard_fired else None,
    )


def multiagent_decide(hazard: dict[str, Any]) -> AgentDecision:
    """One-call adapter: hazard dict → final AgentDecision via the 3-agent graph."""
    hazard_id, yolo_output, image_quality, metadata = hazard_to_graph_inputs(hazard)
    final_state = run_graph_locally(
        hazard_id=hazard_id,
        yolo_output=yolo_output,
        image_quality=image_quality,
        metadata=metadata,
    )
    return final_state_to_decision(final_state)
