"""HazardSubmit now carries optional rich fields; verify they thread through."""

from agent.multiagent.adapter import hazard_to_graph_inputs
from api.schemas import HazardSubmit


def test_hazard_submit_accepts_rich_fields():
    body = HazardSubmit.model_validate(
        {
            "class": "D40",
            "confidence": 0.91,
            "bboxes": [[120, 240, 340, 460, "D40", 0.91]],
            "road_type": "main",
            "lane_blocked": "partial",
            "image_object_key": "obj.jpg",
            "top1_margin": 0.42,
            "area_ratio": 0.08,
            "n_detections_by_class": {"D00": 0, "D10": 0, "D20": 2, "D40": 1},
            "center_y_rel": 0.71,
            "image_quality": {
                "blur_score": 145,
                "is_road_scene": True,
                "ocr_text_detected": "",
            },
        }
    )
    assert body.top1_margin == 0.42
    assert body.area_ratio == 0.08
    assert body.n_detections_by_class["D20"] == 2
    assert body.center_y_rel == 0.71
    assert body.image_quality is not None
    assert body.image_quality.blur_score == 145


def test_hazard_submit_rich_fields_are_optional():
    """Older clients that don't send the new fields must still validate."""
    body = HazardSubmit.model_validate(
        {
            "class": "D40",
            "confidence": 0.91,
            "bboxes": [],
            "road_type": "main",
            "lane_blocked": "partial",
            "image_object_key": "obj.jpg",
        }
    )
    assert body.top1_margin is None
    assert body.area_ratio is None
    assert body.image_quality is None


def test_adapter_uses_rich_fields_when_present():
    hazard = {
        "hazard_id": "hz_test",
        "class": "D40",
        "confidence": 0.91,
        "bboxes": [[120, 240, 340, 460, "D40", 0.91]],
        "road_type": "main",
        "lane_blocked": "partial",
        "location": "Allenby 42",
        "top1_margin": 0.42,
        "area_ratio": 0.08,
        "n_detections_by_class": {"D00": 0, "D10": 0, "D20": 2, "D40": 1},
        "center_y_rel": 0.71,
        "image_quality": {
            "blur_score": 145,
            "is_road_scene": True,
            "ocr_text_detected": "",
        },
    }
    _hid, yolo, iq, _md = hazard_to_graph_inputs(hazard)
    assert yolo["top1_margin"] == 0.42
    assert yolo["detections"][0]["area_ratio"] == 0.08
    assert yolo["detections"][0]["position"]["center_y_rel"] == 0.71
    assert yolo["n_detections_by_class"]["D20"] == 2
    assert iq["blur_score"] == 145
    assert iq["is_road_scene"] is True


def test_adapter_falls_back_when_rich_fields_absent():
    hazard = {
        "hazard_id": "hz_test",
        "class": "D40",
        "confidence": 0.91,
        "bboxes": [],
        "road_type": "main",
        "lane_blocked": "partial",
    }
    _hid, yolo, iq, _md = hazard_to_graph_inputs(hazard)
    assert yolo["top1_margin"] == 0.0
    assert iq["blur_score"] == 150.0
    assert iq["is_road_scene"] is True
