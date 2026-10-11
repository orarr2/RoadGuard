"""Perception Agent: interprets YOLO + image quality + metadata into a HazardProfile.

Reads: yolo_output, image_quality, metadata
Writes: hazard_profile, perception_confidence, similar_cases
Tools:  lookup_similar_cases (vector memory)

The output HazardProfile is the single structured object that the
Research and Decision agents consume. The raw YOLO payload stays here.
"""

from __future__ import annotations

from ..schemas import build_perception_handoff
from ..state import HazardProfile, RoadGuardState
from ..tools import PERCEPTION_TOOLS


def _severity_from_signals(
    hazard_class: str,
    confidence: float,
    area_ratio: float,
    lane_blocked: str,
) -> tuple[str, str]:
    """Rule-based severity label with a one-sentence rationale."""
    if hazard_class == "D40" and lane_blocked in ("partial", "full"):
        return "severe", "pothole with lane obstruction"
    if hazard_class in ("D40", "D20") and area_ratio >= 0.05 and confidence >= 0.6:
        return "severe", f"{hazard_class} with large bbox area"
    if hazard_class in ("D40", "D20") and confidence >= 0.5:
        return "moderate", f"{hazard_class} detected with usable confidence"
    if hazard_class in ("D00", "D10"):
        return "minor", f"{hazard_class} is a linear crack, routine"
    return "moderate", "default severity for ambiguous signal"


def _perception_confidence(
    yolo_top1_margin: float,
    blur_score: float,
    is_road_scene: bool,
    ocr_suspicious: bool,
) -> str:
    """Overall confidence in the perception output, not the YOLO confidence."""
    if not is_road_scene or blur_score < 50 or ocr_suspicious:
        return "low"
    if yolo_top1_margin >= 0.3 and blur_score >= 100:
        return "high"
    return "medium"


def perception_node(state: RoadGuardState) -> dict:
    """Build the HazardProfile from raw signals."""
    yolo = state.get("yolo_output") or {}
    iq = state.get("image_quality") or {}
    md = state.get("metadata") or {}

    top1_class = yolo.get("top1_class", "D40")
    top1_conf = float(yolo.get("top1_confidence", 0.0))
    top1_margin = float(yolo.get("top1_margin", 0.0))
    detections = yolo.get("detections", [])
    area_ratio = max(
        (d.get("area_ratio", 0.0) for d in detections), default=0.0
    )

    blur = float(iq.get("blur_score", 100.0))
    is_road = bool(iq.get("is_road_scene", True))
    ocr_suspicious = bool(
        (iq.get("ocr_sanitized") or {}).get("is_suspicious_pattern", False)
    )
    location_scrubbed = bool(md.get("location_was_scrubbed", False))

    lane_blocked = md.get("lane_blocked", "none")
    location = md.get("location", "")

    sim_result = PERCEPTION_TOOLS["lookup_similar_cases"][0](
        location=location, hazard_class=top1_class
    )
    recurrence_count = sim_result.get("count", 0)
    is_recurring = sim_result.get("is_recurring_location", False)

    severity, rationale = _severity_from_signals(
        hazard_class=top1_class,
        confidence=top1_conf,
        area_ratio=area_ratio,
        lane_blocked=lane_blocked,
    )

    quality_warnings = []
    if blur < 100:
        quality_warnings.append(f"blur_score={blur:.0f} is low")
    if not is_road:
        quality_warnings.append("image may not be a road scene")
    if ocr_suspicious:
        quality_warnings.append(
            "suspicious text detected in image (approval gate forced)"
        )

    profile: HazardProfile = {
        "primary_class": top1_class,
        "primary_confidence": top1_conf,
        "secondary_classes": [
            {"class": d.get("class"), "confidence": d.get("confidence")}
            for d in detections
            if d.get("class") != top1_class
        ][:3],
        "severity_signal": severity,
        "severity_rationale": rationale,
        "affects_vehicle_path": lane_blocked in ("partial", "full"),
        "is_recurring_location": is_recurring,
        "recurrence_count_30d": recurrence_count,
        "quality_warnings": quality_warnings,
    }

    pc = _perception_confidence(
        top1_margin, blur, is_road, ocr_suspicious or location_scrubbed
    )

    needs_more = (
        0.4 <= top1_conf < 0.6
        and not is_recurring
        and blur < 150
    )

    missing: list[str] = []
    if blur < 80:
        missing.append("closer_shot")
    if iq.get("brightness_mean", 0.5) < 0.2:
        missing.append("lighting")

    build_perception_handoff(
        profile=profile,
        confidence=pc,
        needs_more=needs_more,
        missing=missing,
    )

    trace_entry = {
        "iter": state.get("iterations", 0) + 1,
        "speaker": "perception",
        "kind": "handoff_out",
        "content": {
            "to": "research",
            "primary_class": profile["primary_class"],
            "severity": profile["severity_signal"],
            "confidence": pc,
        },
    }

    return {
        "hazard_profile": profile,
        "perception_confidence": pc,
        "similar_cases": sim_result.get("cases", []),
        "trace": [trace_entry],
        "iterations": 1,
    }
