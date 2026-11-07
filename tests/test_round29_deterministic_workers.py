"""Round 29 (6.11.2026 audit, F12): when sanitize already knows the verdict
is fixed by the catalogue (a sealed sha256 match, or a phash neighbour inside
the floor radius whose labels disagree with the live detector), the three
workers run their rule paths and skip their LLM passes. In r23 those passes
took 45% of the round's LLM time on such cases and moved no verdict."""
import pytest

import roadguard_agent as ra


PASSAGE = {
    "passage": "Pothole repair: patch within 48 hours on main roads",
    "article_ids": ["p1"], "hazard_types": ["D40"], "road_types": ["main"],
    "issuing_body": "x", "source_url": "", "date": "", "distance": 0.4,
    "relevance": {"class_keyword_hit": True, "repair_term_hit": True},
}


def _hit(match, distance, classes, stored_cls, stored_conf=0.7):
    return {"found": True, "match": match, "distance": distance, "pic_id": 77,
            "file_name": "x.jpg", "split": "train", "classes": classes, "n_labels": 1,
            "model_prediction": {"top1_class": stored_cls, "top1_confidence": stored_conf}}


def _sanitize(monkeypatch, hit, live_cls="D20", live_conf=0.7):
    monkeypatch.setitem(ra.PERCEPTION_TOOLS, "lookup_dataset_image",
                        (lambda **kw: hit, ra.PERCEPTION_TOOLS["lookup_dataset_image"][1]))
    state = {
        "metadata": {"road_type": "main", "lane_blocked": "none", "location": "x",
                     "reporter_text": "", "image_sha256": "a" * 64, "image_phash": "f" * 16},
        "image_quality": {},
        "yolo_output": {"top1_class": live_cls, "top1_confidence": live_conf, "detections": []},
    }
    return ra.sanitize_node(state)


# ---------------------------------------------------------------- sanitize sets the flag


def test_sealed_sha256_match_marks_the_workers_deterministic(monkeypatch):
    upd = _sanitize(monkeypatch, _hit("sha256", 0, "D20", "D20"))
    assert upd["trace"][0]["content"]["known_image_shortcut"] is True
    assert upd["deterministic_workers_reason"] == "known_image_shortcut"
    assert upd["trace"][0]["content"]["deterministic_workers"] == "known_image_shortcut"


def test_phash_neighbour_inside_the_radius_with_other_labels_marks_deterministic(monkeypatch):
    upd = _sanitize(monkeypatch, _hit("phash", 6, "D40", "D40"), live_cls="D20")
    assert upd["deterministic_workers_reason"] == "phash_class_mismatch_floor"
    # the rule tree agrees: this state ends as human_review on the floor
    decision = ra._decide_rule_based({**upd, "yolo_output": {"top1_class": "D20", "top1_confidence": 0.7}})
    assert decision["floor_reason"] == "phash_class_mismatch"


@pytest.mark.parametrize("hit, live_cls, live_conf", [
    (_hit("phash", 14, "D40", "D40"), "D20", 0.7),   # beyond the floor radius: context only
    (_hit("phash", 6, "D20", "D20"), "D20", 0.7),    # labels agree: the tree decides
    (_hit("phash", 6, "D40", "D40"), "D20", 0.0),    # 0.00 is a placeholder, not a detection
    (_hit("sha256", 0, "D20", "D40"), "D20", 0.7),   # stored score disagrees: not sealed
])
def test_other_catalogue_hits_leave_the_workers_llm_on(monkeypatch, hit, live_cls, live_conf):
    upd = _sanitize(monkeypatch, hit, live_cls=live_cls, live_conf=live_conf)
    assert upd["deterministic_workers_reason"] is None


def test_no_catalogue_hit_leaves_the_workers_llm_on(monkeypatch):
    upd = _sanitize(monkeypatch, {"found": False})
    assert upd["deterministic_workers_reason"] is None


# ---------------------------------------------------------------- the workers honour it


def test_signal_extractor_keeps_the_keywords_and_skips_its_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(ra, "_extract_by_llm", lambda *a, **k: calls.append(1) or (None, 0, 0, None))
    state = {
        "metadata": {"reporter_text": "huge pothole, cars swerving", "road_type": "main",
                     "lane_blocked": "none", "location": "x"},
        "yolo_output": {"top1_class": "D20", "top1_confidence": 0.7, "detections": []},
        "deterministic_workers_reason": "known_image_shortcut",
    }
    upd = ra.signal_extractor_node(state)
    assert calls == []
    content = upd["trace"][0]["content"]
    assert content["skip_reason"] == "verdict_fixed:known_image_shortcut"
    assert content["source"] == "skipped:verdict_fixed:known_image_shortcut"
    assert upd["reporter_intake"]["text"] == "huge pothole, cars swerving"


def test_signal_extractor_llm_still_runs_without_the_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(ra, "_extract_by_llm", lambda *a, **k: calls.append(1) or (None, 0, 0, None))
    state = {
        "metadata": {"reporter_text": "huge pothole, cars swerving", "road_type": "main",
                     "lane_blocked": "none", "location": "x"},
        "yolo_output": {"top1_class": "D20", "top1_confidence": 0.7, "detections": []},
        "deterministic_workers_reason": None,
    }
    ra.signal_extractor_node(state)
    assert calls == [1]


def _profile_state(reason):
    # confidence 0.5 sits in the ambiguous band, so the F7 gate lets the
    # LLM run and only the verdict-fixed flag decides here
    return {
        "yolo_output": {"top1_class": "D20", "top1_confidence": 0.5,
                        "n_detections_by_class": {"D20": 1},
                        "detections": [{"class": "D20", "confidence": 0.5, "area_ratio": 0.1}]},
        "image_quality": {"blur_score": 120, "is_road_scene": True},
        "metadata": {"road_type": "main", "lane_blocked": "none", "location": "x",
                     "reporter_text": ""},
        "iterations": 0,
        "deterministic_workers_reason": reason,
    }


def _stub_perception_tools(monkeypatch):
    monkeypatch.setitem(ra.PERCEPTION_TOOLS, "lookup_dataset_image",
                        (lambda **kw: {"found": False}, ra.PERCEPTION_TOOLS["lookup_dataset_image"][1]))
    monkeypatch.setitem(ra.PERCEPTION_TOOLS, "lookup_similar_cases",
                        (lambda **kw: {"count": 0, "cases": []}, ra.PERCEPTION_TOOLS["lookup_similar_cases"][1]))


def test_profile_builder_runs_its_rule_path_and_skips_its_llm(monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_BUILDER_LLM", "1")
    _stub_perception_tools(monkeypatch)

    def boom(**kw):
        raise AssertionError("the profile builder LLM must not run on a fixed verdict")

    monkeypatch.setattr(ra, "_llm_refine", boom)
    upd = ra.profile_builder_node(_profile_state("phash_class_mismatch_floor"))
    content = upd["trace"][0]["content"]
    assert content["llm_used"] is False
    assert content["llm_skipped_reason"] == "verdict_fixed:phash_class_mismatch_floor"
    assert upd["hazard_profile"]["primary_class"] == "D20"
    assert upd["hazard_profile"]["severity_signal"]


def test_profile_builder_llm_still_runs_without_the_flag(monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_BUILDER_LLM", "1")
    _stub_perception_tools(monkeypatch)
    calls = []

    def refine(**kw):
        calls.append(1)
        return ({"severity_signal": "moderate", "severity_rationale": "r", "confidence": "medium",
                 "needs_more_evidence": False, "missing_fields": []}, 5, 2)

    monkeypatch.setattr(ra, "_llm_refine", refine)
    ra.profile_builder_node(_profile_state(None))
    assert calls == [1]


def _rag_state(reason):
    return {
        "hazard_profile": {"primary_class": "D40", "severity_signal": "moderate"},
        "metadata": {"road_type": "main", "lane_blocked": "partial", "location": "x",
                     "reporter_text": ""},
        "yolo_output": {"top1_confidence": 0.5},
        "image_quality": {"blur_score": 120, "is_road_scene": True},
        "perception_confidence": "medium",
        "iterations": 3,
        "deterministic_workers_reason": reason,
    }


def test_rag_retriever_retrieves_deterministically_and_skips_its_llm(monkeypatch):
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "1")
    calls = []

    def det(hazard_class, road_type, k=3):
        calls.append((hazard_class, road_type))
        return {"procedures": [PASSAGE], "tool_result_ids": ["p1"], "coverage": "direct_match"}

    monkeypatch.setitem(ra.RESEARCH_TOOLS, "query_repair_standards", (det, ra.RESEARCH_TOOLS["query_repair_standards"][1]))

    def boom(*a, **kw):
        raise AssertionError("the research LLM must not run on a fixed verdict")

    monkeypatch.setattr(ra, "_llm_label", boom)
    monkeypatch.setattr(ra, "_autonomous_retrieval", boom)
    upd = ra.rag_retriever_node(_rag_state("known_image_shortcut"))
    assert calls == [("D40", "main")]
    content = upd["trace"][0]["content"]
    assert content["llm_skipped_reason"] == "verdict_fixed:known_image_shortcut"
    assert upd["tool_result_ids"] == ["p1"]
    assert upd["research_coverage"] == "direct_match"
