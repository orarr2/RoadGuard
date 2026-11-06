"""Stage A correctness: RAG relevance grading, cite-supports-claim,
KNOWN_IMAGE gate, and policy-driven thresholds in the rule tree."""

from __future__ import annotations

import roadguard_agent as ra


# ------------------------------------------------------------------ policy

def test_policy_thresholds_match_policy_yaml():
    pol = ra._policy_decision_thresholds()
    assert pol["nme_lo"] == 0.4 and pol["nme_hi"] == 0.6
    assert pol["urgent_min_conf"] == 0.6
    assert pol["urgent_classes"] == {"D40"} and pol["urgent_roads"] == {"main"}
    sev = ra._policy_severity()
    assert sev["large_area_min_ratio"] == 0.05 and sev["cracks_minor"] == ("D00", "D10")


def _state(conf: float, cls: str = "D20") -> dict:
    return {
        "hazard_profile": {"primary_class": cls, "primary_confidence": conf,
                           "is_recurring_location": False, "recurrence_count_30d": 0},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "tool_result_ids": ["sec-1"],
        "procedures": [],
        "research_coverage": "direct_match",
        "perception_confidence": "high",
    }


def test_rule_tree_follows_policy_override(monkeypatch):
    monkeypatch.setattr(ra, "load_policy", lambda: {
        "decision": {"needs_more_evidence": {"confidence_range": ["0.3", "0.5"]},
                     "urgent": {"min_confidence": 0.6}},
    })
    # 6.11.2026: the ambiguous band routes to a human, not to a photo request.
    assert ra._decide_rule_based(_state(0.35))["action"] == "human_review"
    assert ra._decide_rule_based(_state(0.55))["action"] == "standard"


def test_malformed_policy_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr(ra, "load_policy", lambda: {})
    pol = ra._policy_decision_thresholds()
    assert (pol["nme_lo"], pol["nme_hi"], pol["urgent_min_conf"]) == (0.4, 0.6, 0.6)


# ------------------------------------------------------------ known image

def _match(kind: str, pred: dict | None, classes: str = "D20") -> dict:
    return {"found": True, "match": kind, "distance": 0 if kind == "sha256" else 9,
            "pic_id": 7, "file_name": "f.jpg", "split": "train", "classes": classes,
            "n_labels": 1, "model_prediction": pred}


def test_known_image_sealed_only_on_sha256_with_agreeing_stored_prediction():
    state = _state(0.9) | {"hazard_profile": {"primary_class": "D20", "primary_confidence": 0.9,
                                              "severity_signal": "minor"}}
    out = ra._known_image_decision(_match("sha256", {"top1_class": "D20", "top1_confidence": 0.4}), state)
    assert out and out["action"] == "known_image" and out["priority"] == 0
    assert out["image_match_status"]["short_circuit_ok"] is True


def test_known_image_not_sealed_on_phash_or_missing_or_mismatching_prediction():
    state = _state(0.9) | {"hazard_profile": {"primary_class": "D20", "primary_confidence": 0.9,
                                              "severity_signal": "minor"}}
    assert ra._known_image_decision(_match("phash", {"top1_class": "D20", "top1_confidence": 0.4}), state) is None
    assert ra._known_image_decision(_match("sha256", None), state) is None
    # stored D40 vs live D20 while both are catalogued classes: no contradiction, no seal
    assert ra._known_image_decision(_match("sha256", {"top1_class": "D40", "top1_confidence": 0.5}, "D20,D40"), state) is None


def test_rule_tree_explains_declined_seal_and_cites_catalog():
    state = _state(0.9) | {"dataset_match": _match("phash", None)}
    out = ra._decide_rule_based(state)
    assert out["action"] != "known_image"
    assert "catalog:pic_id=7" in out["cited_sources"]
    assert out["image_match_status"]["reason"] == "phash_similar_not_same_file"


def test_sanitize_fast_path_only_for_qualifying_sha256(monkeypatch):
    def fake_lookup(sha256=None, phash=None, **_):
        return _match("sha256" if sha256 == "ok" else "phash", {"top1_class": "D20", "top1_confidence": 0.4})
    monkeypatch.setitem(ra.PERCEPTION_TOOLS, "lookup_dataset_image", (fake_lookup, {}))
    base = {"image_quality": {}, "yolo_output": {"top1_class": "D20"}}
    sealed = ra.sanitize_node({**base, "metadata": {"image_sha256": "ok", "location": ""}})
    assert sealed["next_hop"] == "decision"
    similar = ra.sanitize_node({**base, "metadata": {"image_phash": "ab", "location": ""}})
    assert similar["next_hop"] == "profile_builder" and similar["dataset_match"]["found"]
    assert similar["trace"][0]["content"]["image_match_reason"] == "phash_similar_not_same_file"


# ------------------------------------------------------------------- RAG

class _FakeRag:
    def __init__(self, semantic):
        self._semantic = semantic

    def query(self, **_):
        return []

    def query_semantic(self, query_text, k=5):
        return self._semantic


def _chunk(_id, text):
    return {"passage": text, "article_ids": [_id], "hazard_types": [""], "road_types": ["any"],
            "distance": 0.5, "source_url": "", "issuing_body": "", "date": ""}


def test_semantic_hits_are_graded_not_stamped():
    client = _FakeRag([
        _chunk("boiler", "the public through the National Technical Information Service Springfield"),
        _chunk("pothole", "Pothole repair operations use hot mix patching on arterial roads."),
        _chunk("generic", "Pavement maintenance records are kept per district."),
    ])
    res = ra.query_repair_standards("D40", "main", k=3, rag_client=client)
    ids = res["tool_result_ids"]
    assert "boiler" not in ids and "pothole" in ids and "generic" in ids
    assert res["coverage"] == "direct_match"
    by_id = {p["article_ids"][0]: p["relevance"] for p in res["procedures"]}
    assert by_id["pothole"]["class_keyword_hit"] is True
    assert by_id["generic"]["class_keyword_hit"] is False and by_id["generic"]["repair_term_hit"] is True


def test_only_repair_terms_gives_partial_match_and_boilerplate_only_gives_no_match():
    partial = ra.query_repair_standards("D40", "main", k=3, rag_client=_FakeRag([
        _chunk("generic", "Pavement maintenance records are kept per district.")]))
    assert partial["coverage"] == "partial_match"
    none = ra.query_repair_standards("D40", "main", k=3, rag_client=_FakeRag([
        _chunk("boiler", "the public through the National Technical Information Service")]))
    assert none["coverage"] == "no_match" and none["procedures"] == []


# ------------------------------------------------------------- grounding

def _procs():
    return [
        {"article_ids": ["good"], "relevance": {"class_keyword_hit": True, "repair_term_hit": True}},
        {"article_ids": ["weak"], "relevance": {"class_keyword_hit": False, "repair_term_hit": True}},
        {"article_ids": ["legacy"]},
    ]


def test_llm_cite_must_name_the_class():
    state = {"procedures": _procs()}
    ids = ["good", "weak", "legacy"]
    bad = {"action": "standard", "priority": 50, "cited_sources": ["weak"], "missing_evidence": []}
    good = {**bad, "cited_sources": ["good"]}
    legacy = {**bad, "cited_sources": ["legacy"]}
    assert ra._grounded_or_none(bad, ids, state) is None
    assert ra._grounded_or_none(good, ids, state) is good
    assert ra._grounded_or_none(legacy, ids, state) is legacy
    assert ra._grounded_or_none(bad, ids) is bad  # no state: legacy behaviour


def test_rule_tree_prefers_supported_cites():
    state = _state(0.9) | {"tool_result_ids": ["weak", "good"], "procedures": _procs()}
    out = ra._decide_rule_based(state)
    assert out["action"] == "standard" and out["cited_sources"] == ["good"]
