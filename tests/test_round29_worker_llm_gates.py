"""Round 29 (6.11.2026 audit, F7 and F8): the research LLM runs only when
the tool-graded coverage is not direct_match and the retrieval is cached per
class and road type; the perception LLM runs only on a signal the rules do
not read (reporter text, the ambiguous confidence band, more than one class),
and the lessons tool is offered only once a supervisor left a lesson."""
import pytest

import roadguard_agent as ra


PASSAGE = {
    "passage": "Pothole repair: patch within 48 hours on main roads",
    "article_ids": ["p1"], "hazard_types": ["D40"], "road_types": ["main"],
    "issuing_body": "x", "source_url": "", "date": "", "distance": 0.4,
    "relevance": {"class_keyword_hit": True, "repair_term_hit": True},
}


def _rag_state(road_type="main"):
    return {
        "hazard_profile": {"primary_class": "D40", "severity_signal": "moderate"},
        "metadata": {"road_type": road_type, "lane_blocked": "partial", "location": "x",
                     "reporter_text": ""},
        "yolo_output": {"top1_confidence": 0.5},
        "image_quality": {"blur_score": 120, "is_road_scene": True},
        "perception_confidence": "medium",
        "iterations": 3,
    }


PARTIAL = dict(PASSAGE, road_types=["secondary"])   # class hit, road mismatch: partial_match


def _stub_query(monkeypatch, coverage, calls):
    def det(hazard_class, road_type, k=3):
        calls.append((hazard_class, road_type))
        # the node regrades coverage from the passages, so the passage
        # decides: PASSAGE is direct_match on a main road, PARTIAL is not
        passage = PASSAGE if coverage == "direct_match" else PARTIAL
        return {"procedures": [passage], "tool_result_ids": ["p1"], "coverage": coverage}

    monkeypatch.setitem(ra.RESEARCH_TOOLS, "query_repair_standards", (det, ra.RESEARCH_TOOLS["query_repair_standards"][1]))
    ra._clear_retrieval_cache()


def _boom(*a, **kw):
    raise AssertionError("the research LLM must not run here")


# ---------------------------------------------------------------- F8: LLM only when coverage is not direct


def test_research_llm_is_skipped_when_the_tool_grades_direct_match(monkeypatch):
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "1")
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "0")
    calls = []
    _stub_query(monkeypatch, "direct_match", calls)
    monkeypatch.setattr(ra, "_llm_label", _boom)
    monkeypatch.setattr(ra, "_autonomous_retrieval", _boom)
    upd = ra.rag_retriever_node(_rag_state())
    content = upd["trace"][0]["content"]
    assert content["llm_used"] is False
    assert content["llm_skipped_reason"] == "coverage_direct_match"
    assert upd["research_coverage"] == "direct_match" and upd["tool_result_ids"] == ["p1"]


def test_autonomous_research_is_skipped_on_direct_match_too(monkeypatch):
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "1")
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    calls = []
    _stub_query(monkeypatch, "direct_match", calls)
    monkeypatch.setattr(ra, "_autonomous_retrieval", _boom)
    upd = ra.rag_retriever_node(_rag_state())
    assert upd["trace"][0]["content"]["llm_skipped_reason"] == "coverage_direct_match"
    assert calls == [("D40", "main")]


def test_research_llm_runs_when_the_coverage_is_partial(monkeypatch):
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "1")
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "0")
    calls = []
    _stub_query(monkeypatch, "partial_match", calls)
    labelled = []

    def label(**kw):
        labelled.append(kw["base_coverage"])
        return "partial_match", "medium", ["needs a closer standard"], 7, 3, None

    monkeypatch.setattr(ra, "_llm_label", label)
    upd = ra.rag_retriever_node(_rag_state())
    assert labelled == ["partial_match"]
    assert upd["trace"][0]["content"]["llm_used"] is True
    assert upd["trace"][0]["content"]["key_findings"] == ["needs a closer standard"]


# ---------------------------------------------------------------- F8: retrieval cache


def test_retrieval_is_cached_per_class_and_road_type(monkeypatch):
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "0")
    calls = []
    _stub_query(monkeypatch, "direct_match", calls)
    first = ra.rag_retriever_node(_rag_state())
    second = ra.rag_retriever_node(_rag_state())
    assert calls == [("D40", "main")], "the same question is embedded once"
    assert first["trace"][0]["content"]["rag_cache"] == "miss"
    assert second["trace"][0]["content"]["rag_cache"] == "hit"
    # the cached copy is intact after the first run filtered its own copy
    assert second["tool_result_ids"] == ["p1"] and second["research_coverage"] == "direct_match"
    ra.rag_retriever_node(_rag_state(road_type="secondary"))
    assert calls == [("D40", "main"), ("D40", "secondary")]


def test_retrieval_cache_entries_expire_and_clear(monkeypatch):
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "0")
    calls = []
    _stub_query(monkeypatch, "direct_match", calls)
    ra.rag_retriever_node(_rag_state())
    monkeypatch.setattr(ra, "_RETRIEVAL_CACHE_TTL_S", -1.0)
    ra._clear_retrieval_cache()
    ra.rag_retriever_node(_rag_state())
    ra.rag_retriever_node(_rag_state())
    assert len(calls) == 3, "an expired entry is fetched again"


# ---------------------------------------------------------------- F7: perception LLM only on a signal


def _profile_state(conf=0.7, text="", detections=None):
    dets = detections if detections is not None else [{"class": "D20", "confidence": conf, "area_ratio": 0.1}]
    return {
        "yolo_output": {"top1_class": "D20", "top1_confidence": conf,
                        "n_detections_by_class": {d["class"]: 1 for d in dets},
                        "detections": dets},
        "image_quality": {"blur_score": 120, "is_road_scene": True},
        "metadata": {"road_type": "main", "lane_blocked": "none", "location": "x",
                     "reporter_text": text},
        "iterations": 0,
    }


def _stub_perception(monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_BUILDER_LLM", "1")
    monkeypatch.setitem(ra.PERCEPTION_TOOLS, "lookup_dataset_image",
                        (lambda **kw: {"found": False}, ra.PERCEPTION_TOOLS["lookup_dataset_image"][1]))
    monkeypatch.setitem(ra.PERCEPTION_TOOLS, "lookup_similar_cases",
                        (lambda **kw: {"count": 0, "cases": []}, ra.PERCEPTION_TOOLS["lookup_similar_cases"][1]))
    calls = []

    def refine(**kw):
        calls.append(1)
        return ({"severity_signal": "moderate", "severity_rationale": "r", "confidence": "medium",
                 "needs_more_evidence": False, "missing_fields": []}, 5, 2)

    monkeypatch.setattr(ra, "_llm_refine", refine)
    return calls


def test_perception_llm_skipped_on_one_clear_class_without_text(monkeypatch):
    calls = _stub_perception(monkeypatch)
    upd = ra.profile_builder_node(_profile_state(conf=0.7))
    assert calls == []
    content = upd["trace"][0]["content"]
    assert content["llm_used"] is False
    assert content["llm_skipped_reason"].startswith("no_llm_signal")
    assert upd["hazard_profile"]["primary_class"] == "D20"


@pytest.mark.parametrize("state", [
    _profile_state(conf=0.5),                                      # ambiguous band
    _profile_state(conf=0.7, text="deep hole, cars swerving"),     # reporter text
    _profile_state(conf=0.7, detections=[                          # two classes in the frame
        {"class": "D20", "confidence": 0.7, "area_ratio": 0.1},
        {"class": "D40", "confidence": 0.4, "area_ratio": 0.05},
    ]),
])
def test_perception_llm_runs_on_a_signal_the_rules_do_not_read(monkeypatch, state):
    calls = _stub_perception(monkeypatch)
    ra.profile_builder_node(state)
    assert calls == [1]


def test_no_signal_helper_reads_the_policy_band():
    yolo = {"top1_class": "D20", "top1_confidence": 0.45, "detections": [{"class": "D20"}]}
    assert ra._perception_llm_no_signal(yolo, {"reporter_text": ""}) is None
    yolo["top1_confidence"] = 0.61
    assert ra._perception_llm_no_signal(yolo, {"reporter_text": ""}).startswith("no_llm_signal")
    assert ra._perception_llm_no_signal(yolo, {"reporter_text": "pothole"}) is None


# ---------------------------------------------------------------- F7: lessons tool offered once a lesson exists


def test_recall_agent_notes_is_offered_only_when_the_table_has_a_lesson(monkeypatch):
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    monkeypatch.setattr(ra, "_agent_notes_count", lambda: 0)
    assert "recall_agent_notes" not in ra._worker_tools("profile_builder", ra.PERCEPTION_WORKER_TOOLS)
    assert "recall_agent_notes" not in ra._worker_tools("rag_retriever", ra.RESEARCH_WORKER_TOOLS)
    assert "recall_agent_notes" not in ra._decision_tools()
    monkeypatch.setattr(ra, "_agent_notes_count", lambda: 2)
    assert "recall_agent_notes" in ra._worker_tools("profile_builder", ra.PERCEPTION_WORKER_TOOLS)
    assert "recall_agent_notes" in ra._worker_tools("rag_retriever", ra.RESEARCH_WORKER_TOOLS)
    assert "recall_agent_notes" in ra._decision_tools()
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "0")
    assert "recall_agent_notes" not in ra._worker_tools("profile_builder", ra.PERCEPTION_WORKER_TOOLS)
