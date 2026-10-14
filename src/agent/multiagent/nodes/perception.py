"""Perception Agent: interprets YOLO + image quality + metadata into a HazardProfile.

Reads: yolo_output, image_quality, metadata
Writes: hazard_profile, perception_confidence, similar_cases
Tools:  lookup_similar_cases (vector memory)

The output HazardProfile is the single structured object that the
Research and Decision agents consume. The raw YOLO payload stays here.

The LLM path (USE_LIVE=1) refines severity_signal, severity_rationale,
confidence, and needs_more_evidence by calling chat() with the
PERCEPTION_SYSTEM_PROMPT. On any failure - parse error, schema violation,
or budget-guard - the rule-based profile stands.
"""

from __future__ import annotations

import json
import logging

from ..chat import chat, cost_of
from ..node_prompts import PERCEPTION_SYSTEM_PROMPT
from ..schemas import build_perception_handoff
from ..state import HazardProfile, RoadGuardState
from ..tools import PERCEPTION_TOOLS

LOGGER = logging.getLogger("roadguard.multiagent.perception")

_VALID_SEVERITY = {"minor", "moderate", "severe"}
_VALID_CONFIDENCE = {"high", "medium", "low"}


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


def _parse_perception_json(text: str) -> dict | None:
    """Pull the perception JSON out of an LLM reply. None on any error."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    sev = obj.get("severity_signal")
    conf = obj.get("confidence")
    if sev not in _VALID_SEVERITY or conf not in _VALID_CONFIDENCE:
        return None
    return obj


def _llm_refine(
    base_profile: HazardProfile,
    base_confidence: str,
    base_needs_more: bool,
    base_missing: list[str],
    yolo: dict,
    iq: dict,
    md: dict,
) -> tuple[dict, int, int]:
    """Ask the LLM to refine the perception answer.

    Returns (refined, tokens_in, tokens_out). ``refined`` is either a
    dict with the LLM's parsed fields, or the original rule-based values
    on any failure. Tokens are 0 when the call didn't happen.
    """
    try:
        user_payload = {
            "yolo_output": {
                "top1_class": yolo.get("top1_class"),
                "top1_confidence": yolo.get("top1_confidence"),
                "top1_margin": yolo.get("top1_margin"),
                "detections": yolo.get("detections", [])[:5],
            },
            "image_quality": {
                "blur_score": iq.get("blur_score"),
                "is_road_scene": iq.get("is_road_scene"),
                "brightness_mean": iq.get("brightness_mean"),
                "ocr_suspicious": bool(
                    (iq.get("ocr_sanitized") or {}).get("is_suspicious_pattern", False)
                ),
            },
            "metadata": {
                "road_type": md.get("road_type"),
                "lane_blocked": md.get("lane_blocked"),
                "location_was_scrubbed": md.get("location_was_scrubbed", False),
            },
            "rule_based_hint": {
                "severity_signal": base_profile.get("severity_signal"),
                "severity_rationale": base_profile.get("severity_rationale"),
                "is_recurring_location": base_profile.get("is_recurring_location"),
            },
        }
        messages = [
            {"role": "system", "content": PERCEPTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        msg, in_tok, out_tok = chat(messages, tools=None, temperature=0.0)
    except Exception as exc:
        LOGGER.warning("perception_llm_failed: %s", exc)
        return (
            {
                "severity_signal": base_profile.get("severity_signal"),
                "severity_rationale": base_profile.get("severity_rationale"),
                "confidence": base_confidence,
                "needs_more_evidence": base_needs_more,
                "missing_fields": base_missing,
            },
            0,
            0,
        )

    parsed = _parse_perception_json(msg.content or "")
    if parsed is None:
        return (
            {
                "severity_signal": base_profile.get("severity_signal"),
                "severity_rationale": base_profile.get("severity_rationale"),
                "confidence": base_confidence,
                "needs_more_evidence": base_needs_more,
                "missing_fields": base_missing,
            },
            in_tok,
            out_tok,
        )

    missing = parsed.get("missing_fields") or []
    if not isinstance(missing, list):
        missing = base_missing

    return (
        {
            "severity_signal": parsed["severity_signal"],
            "severity_rationale": str(parsed.get("severity_rationale") or "")[:200],
            "confidence": parsed["confidence"],
            "needs_more_evidence": bool(parsed.get("needs_more_evidence", base_needs_more)),
            "missing_fields": [str(m) for m in missing][:5],
        },
        in_tok,
        out_tok,
    )


def perception_node(state: RoadGuardState) -> dict:
    """Build the HazardProfile from raw signals, optionally LLM-refined."""
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

    base_profile: HazardProfile = {
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

    base_confidence = _perception_confidence(
        top1_margin, blur, is_road, ocr_suspicious or location_scrubbed
    )

    base_needs_more = (
        0.4 <= top1_conf < 0.6
        and not is_recurring
        and blur < 150
    )

    base_missing: list[str] = []
    if blur < 80:
        base_missing.append("closer_shot")
    if iq.get("brightness_mean", 0.5) < 0.2:
        base_missing.append("lighting")

    refined, in_tok, out_tok = _llm_refine(
        base_profile=base_profile,
        base_confidence=base_confidence,
        base_needs_more=base_needs_more,
        base_missing=base_missing,
        yolo=yolo,
        iq=iq,
        md=md,
    )

    profile: HazardProfile = dict(base_profile)  # type: ignore[assignment]
    profile["severity_signal"] = refined["severity_signal"]
    profile["severity_rationale"] = refined["severity_rationale"]
    pc = refined["confidence"]
    needs_more = refined["needs_more_evidence"]
    missing = refined["missing_fields"]

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
            "llm_used": in_tok > 0,
        },
    }

    update: dict = {
        "hazard_profile": profile,
        "perception_confidence": pc,
        "similar_cases": sim_result.get("cases", []),
        "trace": [trace_entry],
        "iterations": 1,
    }
    if in_tok or out_tok:
        update["tokens_in"] = in_tok
        update["tokens_out"] = out_tok
        update["cost_usd"] = cost_of(in_tok, out_tok)
    return update
