"""Monolith build_user_message surfaces the optional rich fields."""

import json

from agent.prompts import build_user_message


def _base_hazard(**overrides):
    base = {
        "hazard_id": "hz_test",
        "class": "D40",
        "confidence": 0.91,
        "bboxes": [[120, 240, 340, 460, "D40", 0.91]],
        "road_type": "main",
        "lane_blocked": "partial",
        "location": "Allenby 42",
        "reported_at": "2026-10-14T10:00:00Z",
    }
    base.update(overrides)
    return base


def test_basic_payload_without_rich_fields():
    msg = build_user_message(_base_hazard())
    payload = json.loads(msg)
    assert payload["detections"]["top1_class"] == "D40"
    assert "top1_margin" not in payload["detections"]
    assert "area_ratio" not in payload["detections"]
    assert "image_quality" not in payload


def test_payload_includes_top1_margin_when_present():
    msg = build_user_message(_base_hazard(top1_margin=0.42))
    payload = json.loads(msg)
    assert payload["detections"]["top1_margin"] == 0.42


def test_payload_includes_area_ratio_when_present():
    msg = build_user_message(_base_hazard(area_ratio=0.08))
    payload = json.loads(msg)
    assert payload["detections"]["area_ratio"] == 0.08


def test_payload_includes_n_detections_by_class():
    msg = build_user_message(
        _base_hazard(n_detections_by_class={"D00": 0, "D10": 0, "D20": 2, "D40": 1})
    )
    payload = json.loads(msg)
    assert payload["detections"]["n_detections_by_class"]["D20"] == 2


def test_payload_includes_image_quality_when_present():
    msg = build_user_message(
        _base_hazard(
            image_quality={
                "blur_score": 145,
                "is_road_scene": True,
                "ocr_text_detected": "",
            }
        )
    )
    payload = json.loads(msg)
    assert payload["image_quality"]["blur_score"] == 145
    assert payload["image_quality"]["is_road_scene"] is True
    # None values are stripped
    assert "ocr_text_detected" in payload["image_quality"]


def test_payload_omits_image_quality_when_only_none_values():
    msg = build_user_message(
        _base_hazard(image_quality={"blur_score": None, "is_road_scene": None})
    )
    payload = json.loads(msg)
    assert "image_quality" not in payload


def test_payload_includes_center_y_rel():
    msg = build_user_message(_base_hazard(center_y_rel=0.71))
    payload = json.loads(msg)
    assert payload["detections"]["center_y_rel"] == 0.71


def test_system_prompt_mentions_optional_signals():
    """The system prompt should describe the rich fields so the LLM knows about them."""
    from agent.prompts import AGENT_SYSTEM_PROMPT

    assert "top1_margin" in AGENT_SYSTEM_PROMPT
    assert "image_quality" in AGENT_SYSTEM_PROMPT
    assert "area_ratio" in AGENT_SYSTEM_PROMPT
