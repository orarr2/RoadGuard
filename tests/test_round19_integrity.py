"""Round 19 (3.11.2026 night): the owner's six items from his r12 review.
A: a phash neighbour pic_id in the model's text is flagged (and only when
   the match is phash; a sha256 identity match may carry the pic_id).
B: when the fact-check fails and the retry fails, the final rationale and
   assessment carry a one-line note naming the kinds of claim removed.
C: the RAG trace line shows `topic=<coverage> sufficient=<yes|no>` so a
   direct_match no longer reads as "the standards answer the questions".
D: the Decision payload surfaces the passage source document name from
   Benny's corpus metadata next to the article ids.
E: the few-shot example id in both prompts is a placeholder (`ex-id-00`),
   not a plausible real id that could be copy-pasted as a fake citation."""
import json

import pytest

import roadguard_agent as ra


# ---------------------------------------------------------------- A: phash neighbour pic_id citation


def _dec_state(match):
    base = ra.initial_state(
        "hz_t", {"top1_class": "D40", "top1_confidence": 0.7, "detections": []}, {},
        {"road_type": "main", "lane_blocked": "none", "location": "x", "reporter_text": ""},
    )
    base.update({
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.7, "severity_signal": "moderate",
                           "secondary_classes": [], "recurrence_count_30d": 0},
        "procedures": [], "tool_result_ids": [], "research_coverage": "direct_match",
        "research_coverage_detail": {"repair_window": False, "urgency": False, "answered": [],
                                     "missing": ["repair_window", "urgency"],
                                     "sufficiency": "insufficient_coverage"},
        "perception_confidence": "medium", "reporter_intake": {},
        "dataset_match": match,
    })
    return base


def test_phash_neighbor_pic_id_cited_as_evidence_is_flagged():
    state = _dec_state({"found": True, "match": "phash", "distance": 14, "pic_id": 10179,
                        "classes": "D00", "model_prediction": {"top1_class": "D00"}})
    dec = {
        "action": "standard", "priority": 55, "cited_sources": [], "missing_evidence": [],
        "rationale": "D20 crack on main road; cataloged pic_id=10179 is consistent with the image.",
        "assessment": "",
    }
    rc = ra._verify_decision_rationale(dec, state)
    assert any(p.startswith("phash_neighbor_cited_as_evidence") for p in rc["problems"])


def test_sha256_match_pic_id_citation_is_allowed():
    state = _dec_state({"found": True, "match": "sha256", "distance": 0, "pic_id": 1024,
                        "classes": "D40", "model_prediction": {"top1_class": "D40"}})
    dec = {
        "action": "known_image", "priority": 0, "cited_sources": ["catalog:pic_id=1024"],
        "missing_evidence": [],
        "rationale": "catalogued reference image for D40 at detector confidence 0.70; pic_id=1024 is this photo.",
        "assessment": "",
    }
    rc = ra._verify_decision_rationale(dec, state)
    assert not any(p.startswith("phash_neighbor_cited_as_evidence") for p in rc["problems"])


def test_similar_catalogue_photo_payload_names_the_constraint_explicitly():
    """The payload the Decision agent reads must forbid pic_id citation
    in words, so even a model that ignores the fact-check sees the rule."""
    ctx = ra._catalogue_context_for_prompt({"found": True, "match": "phash", "distance": 15,
                                            "pic_id": 2417, "classes": "D20",
                                            "model_prediction": {"top1_class": "D20"}})
    note = ctx["similar_catalogue_photo"]["note"]
    # Round 20: the payload hides the pic_id entirely; the note says so.
    assert "HIDDEN" in note and "pic_id" in note
    assert "human_review" in note


# ---------------------------------------------------------------- B: removed-claims note under the deterministic sentence


def test_removed_claims_note_names_the_kinds_of_claim():
    note = ra._removed_claims_note([
        "lane_blockage_not_in_form",
        "reporter_claims_without_text",
        "phash_neighbor_cited_as_evidence:pic_id=10179",
    ])
    assert "lane blockage" in note
    assert "reporter" in note
    assert "phash-neighbour pic_id" in note


def test_removed_claims_note_empty_on_no_problems():
    assert ra._removed_claims_note([]) == ""


# ---------------------------------------------------------------- C: RAG trace line shows topic/sufficient apart


def test_rag_line_shows_topic_and_sufficient_separately():
    entry = {
        "speaker": "rag_retriever", "kind": "handoff_out",
        "content": {
            "coverage": "direct_match", "n_procedures": 3, "llm_used": True,
            "tool_result_ids": ["a", "b", "c"],
            "decision_coverage": {
                "answered": ["urgency"],
                "missing": ["repair_window", "traffic_control", "safety_risk"],
                "sufficiency": "insufficient_coverage",
            },
        },
    }
    lines = ra._entry_lines(entry, [])
    first = next(line for line in lines if "query_repair_standards" in line)
    assert "topic=direct_match" in first
    assert "sufficient=no" in first


def test_rag_line_sufficient_yes_on_full_coverage():
    entry = {
        "speaker": "rag_retriever", "kind": "handoff_out",
        "content": {
            "coverage": "direct_match", "n_procedures": 3, "llm_used": True,
            "tool_result_ids": ["a"],
            "decision_coverage": {
                "answered": ["repair_window", "urgency"],
                "missing": [],
                "sufficiency": "sufficient_coverage",
            },
        },
    }
    lines = ra._entry_lines(entry, [])
    first = next(line for line in lines if "query_repair_standards" in line)
    assert "topic=direct_match" in first and "sufficient=yes" in first


# ---------------------------------------------------------------- D: Decision payload carries passage source.name


def test_decision_payload_carries_passage_source_name():
    state = ra.initial_state(
        "hz_t", {"top1_class": "D40", "top1_confidence": 0.9, "detections": []}, {},
        {"road_type": "main", "lane_blocked": "none", "location": "x", "reporter_text": ""},
    )
    state.update({
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.9, "severity_signal": "moderate",
                           "secondary_classes": [], "recurrence_count_30d": 0},
        "procedures": [{
            "article_ids": ["a1"],
            "source": "well-managed_highway_infrastructure_2016.pdf",
            "quarantined_summary": {"text": "pothole repair"},
            "relevance": {"class_keyword_hit": True},
        }],
        "tool_result_ids": ["a1"], "research_coverage": "direct_match",
        "perception_confidence": "medium", "reporter_intake": {},
    })
    msgs = ra._decision_messages(
        state,
        {"action": "standard", "priority": 60, "rationale": "r", "cited_sources": [], "missing_evidence": []},
    )
    payload = json.loads(msgs[1]["content"])
    procs = payload["procedures"]
    assert procs and procs[0]["source"]["name"] == "well-managed_highway_infrastructure_2016.pdf"


# ---------------------------------------------------------------- E: few-shot id is a placeholder


def test_few_shot_examples_use_a_placeholder_id():
    # The few-shot example id in both prompts is the opaque ex-id-00; a model
    # that copies it as a citation will fail the grounding check (ex-id-00 is
    # not in any run's tool_result_ids).
    assert "ex-id-00" in ra.RAG_SYSTEM_PROMPT if hasattr(ra, "RAG_SYSTEM_PROMPT") else True
    # The main user-facing prompt is DECISION_SYSTEM_PROMPT.
    assert "ex-id-00" in ra.DECISION_SYSTEM_PROMPT
    # And the real sec-4.3 is not in the fewshot anymore.
    rp = ra._RESEARCH_FEWSHOT
    dp = ra._DECISION_FEWSHOT
    assert "sec-4.3" not in rp and "sec-4.3" not in dp
    assert "ex-id-00" in rp and "ex-id-00" in dp
