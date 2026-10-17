"""Bridge between the FastAPI shell and the 3-agent graph.

Phase-1 of the migration: the API's /hazards/submit still ships a flat
hazard dict (per the current HazardSubmit pydantic model). This adapter
reshapes it into the graph's three input blocks (yolo_output,
image_quality, metadata), runs run_graph_locally, and reshapes the
final state back into the AgentDecision dataclass the API already
consumes. The API contract therefore does not change.

Both /hazards/submit (batch) and /hazards/live (streaming) land here.
The two public entry points share one set of private normalizers so
there is a single source of truth for every default value.
"""

from __future__ import annotations

from typing import Any

from agent.loop import AgentDecision

from .graph import run_graph_locally


# ---------------------------------------------------------------------------
# Private normalizers - single source of truth for every default
# ---------------------------------------------------------------------------


_DEFAULT_COUNTS: dict[str, int] = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}


def _normalize_image_quality(quality_in: dict[str, Any] | None) -> dict[str, Any]:
    """Fill the Perception node's image_quality block with safe defaults.

    Clients that do not yet carry an in-browser quality probe hit this
    with an empty dict; the Perception node still has values to reason
    about. One place to change a default - not two.
    """
    q = quality_in or {}
    is_road = q.get("is_road_scene")
    return {
        "blur_score": float(q.get("blur_score") or 150.0),
        "brightness_mean": float(q.get("brightness_mean") or 0.5),
        "is_road_scene": bool(is_road if is_road is not None else True),
        "road_scene_confidence": float(q.get("road_scene_confidence") or 0.9),
        "occlusion_estimate": q.get("occlusion_estimate") or "low",
        "ocr_text_detected": q.get("ocr_text_detected") or "",
    }


def _normalize_metadata(src: dict[str, Any], *, extra_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Metadata block fed to the graph.

    `extra_keys` lets the live-drive flow pass through `session_id` and
    `track_id` without duplicating the base shape in two places.
    """
    out: dict[str, Any] = {
        "road_type": src["road_type"],
        "lane_blocked": src["lane_blocked"],
        "location": src.get("location") or "",
        "reported_at": src.get("reported_at", ""),
    }
    for key in extra_keys:
        out[key] = src.get(key) or ""
    return out


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
    counts = dict(_DEFAULT_COUNTS)
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


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def hazard_to_graph_inputs(hazard: dict[str, Any]) -> tuple[str, dict, dict, dict]:
    """Map a flat HazardSubmit-style dict to the graph's three input blocks.

    HazardSubmit carries optional rich fields (top1_margin, area_ratio,
    n_detections_by_class, center_y_rel, image_quality). When present
    they override the defaults; when absent, Perception still has
    sensible fallbacks via _normalize_image_quality.
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

    return (
        hazard_id,
        yolo_output,
        _normalize_image_quality(hazard.get("image_quality")),
        _normalize_metadata(hazard),
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
    node wants; this builder keeps them on the top-1 detection and
    reuses the same image_quality / metadata normalizers as the batch
    path.
    """
    hazard_class = submit.get("class") or submit["hazard_class"]
    confidence = float(submit["confidence"])
    top1_margin = float(submit.get("top1_margin") or 0.0)
    area_ratio = float(submit.get("area_ratio") or 0.0)
    counts = submit.get("n_detections_by_class") or dict(_DEFAULT_COUNTS)
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

    return (
        hazard_id,
        yolo_output,
        _normalize_image_quality(submit.get("image_quality")),
        _normalize_metadata(submit, extra_keys=("session_id", "track_id")),
    )


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
