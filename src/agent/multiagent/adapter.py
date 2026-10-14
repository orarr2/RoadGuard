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

    HazardSubmit now carries the optional rich fields (top1_margin,
    area_ratio, n_detections_by_class, center_y_rel, image_quality). When
    present, they override the defaults; when absent, the perception node
    still has sensible fallbacks.
    """
    hazard_id = hazard["hazard_id"]
    hazard_class = hazard["class"]
    confidence = float(hazard["confidence"])
    bboxes = hazard.get("bboxes") or []

    detections = _bboxes_to_detections(bboxes, hazard_class, confidence)
    # If the client sent area_ratio, surface it onto the top-1 detection so
    # Perception's area-based severity gate sees it.
    if hazard.get("area_ratio") is not None and detections:
        detections[0]["area_ratio"] = float(hazard["area_ratio"])
    if hazard.get("center_y_rel") is not None and detections:
        detections[0].setdefault("position", {})["center_y_rel"] = float(
            hazard["center_y_rel"]
        )

    yolo_output = {
        "top1_class": hazard_class,
        "top1_confidence": confidence,
        "top1_margin": float(hazard.get("top1_margin") or 0.0),
        "detections": detections,
        "n_detections_by_class": (
            hazard.get("n_detections_by_class") or _count_by_class(detections)
        ),
    }

    quality_in = hazard.get("image_quality") or {}
    image_quality = {
        "blur_score": float(quality_in.get("blur_score") or 150.0),
        "brightness_mean": float(quality_in.get("brightness_mean") or 0.5),
        "is_road_scene": bool(
            quality_in["is_road_scene"]
            if quality_in.get("is_road_scene") is not None
            else True
        ),
        "road_scene_confidence": float(quality_in.get("road_scene_confidence") or 0.9),
        "occlusion_estimate": quality_in.get("occlusion_estimate") or "low",
        "ocr_text_detected": quality_in.get("ocr_text_detected") or "",
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


def live_hazard_to_graph_inputs(
    hazard_id: str,
    submit: dict[str, Any],
) -> tuple[str, dict, dict, dict]:
    """Shape a LiveHazardSubmit (dict form) for run_graph_locally.

    LiveHazardSubmit already carries the richer fields the Perception
    node wants (top1_margin, area_ratio, n_detections_by_class,
    image_quality), so no defaults are injected except where the client
    skipped a field.
    """
    hazard_class = submit["class"] if "class" in submit else submit["hazard_class"]
    confidence = float(submit["confidence"])
    top1_margin = float(submit.get("top1_margin") or 0.0)
    area_ratio = float(submit.get("area_ratio") or 0.0)
    counts = submit.get("n_detections_by_class") or {
        "D00": 0,
        "D10": 0,
        "D20": 0,
        "D40": 0,
    }
    counts.setdefault(hazard_class, 0)
    if counts.get(hazard_class, 0) < 1:
        counts[hazard_class] = 1

    detection: dict[str, Any] = {
        "class": hazard_class,
        "confidence": confidence,
        "area_ratio": area_ratio,
    }
    if submit.get("bbox"):
        detection["bbox"] = list(submit["bbox"])
    if submit.get("center_y_rel") is not None:
        detection["position"] = {"center_y_rel": float(submit["center_y_rel"])}

    yolo_output = {
        "top1_class": hazard_class,
        "top1_confidence": confidence,
        "top1_margin": top1_margin,
        "detections": [detection],
        "n_detections_by_class": counts,
    }

    quality_in = submit.get("image_quality") or {}
    image_quality = {
        "blur_score": float(quality_in.get("blur_score") or 150.0),
        "brightness_mean": float(quality_in.get("brightness_mean") or 0.5),
        "is_road_scene": bool(
            quality_in.get("is_road_scene")
            if quality_in.get("is_road_scene") is not None
            else True
        ),
        "road_scene_confidence": float(
            quality_in.get("road_scene_confidence") or 0.9
        ),
        "occlusion_estimate": quality_in.get("occlusion_estimate") or "low",
        "ocr_text_detected": quality_in.get("ocr_text_detected") or "",
    }

    metadata = {
        "road_type": submit["road_type"],
        "lane_blocked": submit["lane_blocked"],
        "location": submit.get("location") or "",
        "reported_at": submit.get("reported_at", ""),
        "session_id": submit.get("session_id", ""),
        "track_id": submit.get("track_id") or "",
    }
    return hazard_id, yolo_output, image_quality, metadata


def multiagent_decide_live(
    hazard_id: str,
    submit: dict[str, Any],
) -> AgentDecision:
    """Live-drive variant: expects a LiveHazardSubmit-style dict."""
    hazard_id, yolo_output, image_quality, metadata = live_hazard_to_graph_inputs(
        hazard_id=hazard_id, submit=submit
    )
    final_state = run_graph_locally(
        hazard_id=hazard_id,
        yolo_output=yolo_output,
        image_quality=image_quality,
        metadata=metadata,
    )
    return final_state_to_decision(final_state)
