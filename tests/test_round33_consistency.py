"""Round 33 (14.11.2026, after the owner's review of the r31 trace): a
submission whose detector found nothing is not triaged as a hazard, and
the RAG floor's catalogue exemption counts only a true match."""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import roadguard_agent as ra  # noqa: E402


def test_detector_found_nothing_only_for_placeholder_submissions():
    # the webapp sends D00 at 0.0, the batch runner D40 at 0.0, no boxes
    assert ra._detector_found_nothing({"top1_class": "D40", "top1_confidence": 0.0, "detections": []})
    assert ra._detector_found_nothing({"top1_class": "D00", "top1_confidence": 0, "detections": [], "n_detections_by_class": {}})
    assert ra._detector_found_nothing({"top1_class": "D00"})
    # a real detection, however weak, is not "nothing"
    assert not ra._detector_found_nothing({"top1_class": "D40", "top1_confidence": 0.28,
                                           "detections": [{"class": "D40", "confidence": 0.28}]})
    assert not ra._detector_found_nothing({"top1_class": "D20", "top1_confidence": 0.15, "detections": []})
    assert not ra._detector_found_nothing({"top1_class": "D40", "top1_confidence": 0.0,
                                           "detections": [{"class": "D40", "confidence": 0.2}]})
    # the placeholder the server synthesizes for a body without boxes (r33:
    # it hid every no-detection case from the branch)
    assert ra._detector_found_nothing({"top1_class": "D40", "top1_confidence": 0.0,
                                       "detections": [{"class": "D40", "confidence": 0.0, "area_ratio": 0.0}],
                                       "n_detections_by_class": {"D40": 1}})


def test_api_path_marks_a_body_without_boxes_as_no_detection():
    """The webapp and the batch runner reach the graph through
    hazard_to_graph_inputs, which synthesizes one placeholder detection
    when the body has no boxes; the flag must survive that path."""
    body = {"hazard_id": "hz_test_nothing", "class": "D40", "confidence": 0.0, "bboxes": [],
            "road_type": "main", "lane_blocked": "none", "location": "x"}
    _hid, yolo, _iq, _md = ra.hazard_to_graph_inputs(body)
    assert yolo["detections"] and yolo["detections"][0]["confidence"] == 0.0
    assert yolo["n_detections_by_class"].get("D40") == 1
    assert ra._detector_found_nothing(yolo) is True

    body2 = dict(body, hazard_id="hz_test_weak", confidence=0.28,
                 bboxes=[[0.1, 0.1, 0.4, 0.4, "D40", 0.28]])
    _hid, yolo2, _iq, _md = ra.hazard_to_graph_inputs(body2)
    assert ra._detector_found_nothing(yolo2) is False


def _state(found_nothing: bool) -> dict:
    return {
        "hazard_profile": {
            "primary_class": "D40",
            "primary_confidence": 0.0,
            "severity_signal": "moderate",
            "secondary_classes": [],
            "detector_found_nothing": found_nothing,
        },
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "yolo_output": {"top1_class": "D40", "top1_confidence": 0.0, "detections": []},
        "reporter_intake": {},
        "research_coverage": "direct_match",
        "perception_confidence": "medium",
        "tool_result_ids": ["segment_doc_21_7941_3_c2"],
    }


def test_rule_tree_routes_a_no_detection_photo_to_a_human():
    out = ra._decide_rule_based(_state(True))
    assert out["action"] == "human_review", out
    assert "found no road damage" in out["rationale"]
    assert out["cited_sources"] == []


def test_without_the_flag_the_same_inputs_still_fall_through_to_standard():
    # the r31 behaviour the flag guards against: confidence 0.00 below the
    # ambiguous band was a standard work order
    out = ra._decide_rule_based(_state(False))
    assert out["action"] == "standard", out


def test_deterministic_rationale_says_the_detector_found_nothing():
    state = _state(True)
    text = ra._deterministic_rationale({"action": "human_review", "priority": 55}, state)
    assert "found no road damage" in text
    assert "placeholder" in text
    assert "found no road damage" not in ra._deterministic_rationale({"action": "standard", "priority": 55}, _state(False))


def _match(kind: str, distance: int, classes: str = "D40", stored: str = "D40") -> dict:
    return {
        "found": True,
        "match": kind,
        "distance": distance,
        "pic_id": 7,
        "classes": classes,
        "model_prediction": {"top1_class": stored, "top1_confidence": 0.7},
    }


def test_rag_floor_exemption_counts_only_a_true_catalogue_match():
    base = {"hazard_profile": {"primary_class": "D40"}}
    # the same file
    assert ra._catalog_match_exempts_rag_floor({**base, "dataset_match": _match("sha256", 0)})
    # a close neighbour whose labels name the live class
    assert ra._catalog_match_exempts_rag_floor({**base, "dataset_match": _match("phash", 3)})
    # a close neighbour whose labels disagree but whose recorded detector class agrees
    assert ra._catalog_match_exempts_rag_floor({**base, "dataset_match": _match("phash", 3, classes="D20", stored="D40")})
    # a far neighbour (the r31 exemption: distances 12 to 16, "not the same file")
    assert not ra._catalog_match_exempts_rag_floor({**base, "dataset_match": _match("phash", 12)})
    assert not ra._catalog_match_exempts_rag_floor({**base, "dataset_match": _match("phash", 11)})
    # a close neighbour of another class
    assert not ra._catalog_match_exempts_rag_floor({**base, "dataset_match": _match("phash", 3, classes="D20", stored="D20")})
    # no match at all
    assert not ra._catalog_match_exempts_rag_floor({**base, "dataset_match": None})
    assert not ra._catalog_match_exempts_rag_floor({**base, "dataset_match": {"found": False, "reason": "none"}})
    assert not ra._catalog_match_exempts_rag_floor({"dataset_match": _match("phash", 3)})


def test_exemption_radius_comes_from_the_policy():
    assert ra._policy_known_image()["phash_floor_max_distance"] == 10
    assert ra._policy_known_image()["min_stored_confidence"] == 0.4


def test_policy_version_and_trigger_name_the_change():
    pol = ra.load_policy()
    assert pol["version"] == "2026.11.14"
    text = pathlib.Path(ra.__file__).with_name("policy.yaml").read_text(encoding="utf-8")
    assert "detector_found_nothing == true" in text
    assert "only a true catalogue match exempts this floor" in text
