"""Round 11 (1.11.2026), the reliability round the owner asked for after his
review of run r7: worker tool contracts, a fact-check of the Decision
rationale against the inputs, tool failures as violations and in the agent
metrics, RAG support grading, and seconds per stage in the trace."""
import json

import roadguard_agent as ra
import roadguard_api as api


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(finish_reason="tool_calls" if tool_calls else "stop", content=content, tool_calls=tool_calls or [])


def _call(name, args, call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


SIMILAR = ra.PERCEPTION_WORKER_TOOLS["lookup_similar_cases"][1]


def _registry(record):
    def similar(location: str, hazard_class: str, state: dict | None = None) -> dict:
        record.append(("similar", location, hazard_class))
        return {"ok": True, "count": 0, "cases": [], "is_recurring_location": False}

    def quality(blur_score: float = 0, is_road_scene: bool = True, brightness_mean: float = 0.5, ocr_suspicious: bool = False) -> dict:
        record.append(("quality", blur_score, is_road_scene))
        return {"ok": True, "quality_verdict": "high", "concerns": []}

    return {
        "lookup_similar_cases": (similar, SIMILAR),
        "inspect_image_quality": (quality, ra.PERCEPTION_WORKER_TOOLS["inspect_image_quality"][1]),
    }


def _run_worker(monkeypatch, replies, registry):
    seen = []

    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False, provider_override=None, **kw):
        seen.append(tools)
        return replies[len(seen) - 1], 10, 5

    monkeypatch.setattr(ra, "chat", fake_chat)
    out = ra._react_worker_rounds(
        agent_name="profile_builder", system_prompt="s", user_payload={"x": 1}, tool_registry=registry,
        state={"hazard_id": "hz_t", "metadata": {}}, provider_override=None, max_hops=2,
    )
    return out, seen


def test_worker_reply_sent_as_a_tool_call_through_a_lookup_tool_is_the_reply(monkeypatch):
    """Round 18 (2.11.2026) rejected a final JSON riding a lookup tool call
    and asked again; round 29 (6.11.2026 audit F6) uses the worker's own
    reply as the reply at once (the retry threw it away in r23) and still
    runs nothing with the reply as arguments."""
    record = []
    reply = {"severity_signal": "severe", "severity_rationale": "D40 pothole", "confidence": "high",
             "needs_more_evidence": False, "missing_fields": [], "tools_used": ["inspect_image_quality"]}
    replies = [
        _msg(tool_calls=[_call("lookup_similar_cases", reply)]),
    ]
    (parsed, _, _, info), seen = _run_worker(monkeypatch, replies, _registry(record))
    assert parsed == reply and len(seen) == 1
    assert record == [], "nothing may run with the reply as arguments"
    first = info["calls"][0]
    assert first["ok"] is False and first.get("reply_in_tool_call") is True


def test_worker_tool_arguments_are_validated_and_unknown_keys_dropped(monkeypatch):
    record = []
    final = json.dumps({"severity_signal": "minor", "confidence": "medium", "needs_more_evidence": False, "missing_fields": []})
    replies = [
        _msg(tool_calls=[_call("lookup_similar_cases", {"location": "x", "hazard_class": "D99"})]),
        _msg(tool_calls=[_call("inspect_image_quality", {"blur_score": 150, "is_road_scene": True, "brightness_mean": 0.5,
                                                          "ocr_suspicious": False, "note": "extra key"}, "c2")]),
        _msg(content=final),
    ]
    (parsed, _, _, info), _ = _run_worker(monkeypatch, replies, _registry(record))
    assert parsed["severity_signal"] == "minor"
    first, second = info["calls"][0], info["calls"][1]
    assert first["ok"] is False and first["executed"] is True
    assert "hazard_class must be one of" in info["steps"][0]["content"]["result"]
    assert second["ok"] is True
    assert record == [("quality", 150, True)], "the valid call ran once, without the extra key, the invalid one never ran"
    assert info["hops"] == 1
    rec = ra._autonomy_record(parsed, info, required=("inspect_image_quality",))
    assert "tool_call_failed:lookup_similar_cases" in rec["violations"]
    assert rec["failed_tools"] == ["lookup_similar_cases"]
    assert "required_not_called:inspect_image_quality" not in rec["violations"]


def test_worker_required_key_with_a_function_default_is_not_a_problem(monkeypatch):
    record = []
    final = json.dumps({"severity_signal": "minor", "confidence": "medium", "needs_more_evidence": False, "missing_fields": []})
    replies = [_msg(tool_calls=[_call("inspect_image_quality", {"is_road_scene": True})]), _msg(content=final)]
    (parsed, _, _, info), _ = _run_worker(monkeypatch, replies, _registry(record))
    assert info["calls"][0]["ok"] is True and record == [("quality", 0, True)]


def test_autonomy_line_names_failed_tools_and_a_reply_in_a_tool_call():
    entry = {"speaker": "profile_builder", "kind": "handoff_out", "content": {
        "primary_class": "D40", "severity": "severe", "confidence": "medium", "llm_used": True,
        "autonomy": {"mode": "autonomous", "tools_used": ["inspect_image_quality"], "declared_tools": [], "hops": 1,
                     "failed_tools": ["lookup_similar_cases"], "final_in_tool_call": True,
                     "violations": ["tool_call_failed:lookup_similar_cases"]}}}
    text = "\n".join(ra._entry_lines(entry, []))
    assert "failed=lookup_similar_cases" in text and "reply_via=tool_call" in text
    assert "violations=tool_call_failed:lookup_similar_cases" in text


_STATE_NO_TEXT = {
    "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.47, "secondary_classes": []},
    "metadata": {"road_type": "main", "lane_blocked": "none"},
    "reporter_intake": {}, "dataset_match": None, "research_coverage": "direct_match",
}


def test_rationale_check_flags_claims_the_inputs_do_not_carry():
    dec = {"action": "standard", "priority": 45, "rationale": "The reporter's claims are consistent with the detector's class.", "assessment": ""}
    check = ra._verify_decision_rationale(dec, _STATE_NO_TEXT)
    assert check["ok"] is False and check["problems"] == ["reporter_claims_without_text"]
    assert check["rationale_problems"] == ["reporter_claims_without_text"] and check["assessment_problems"] == []
    only_assessment = {"action": "standard", "priority": 45, "rationale": "D40 pothole on main road; routine repair.",
                       "assessment": "The reporter's claims are consistent with the pothole."}
    split = ra._verify_decision_rationale(only_assessment, _STATE_NO_TEXT)
    assert split["rationale_problems"] == [] and split["assessment_problems"] == ["reporter_claims_without_text"]
    with_text = {**_STATE_NO_TEXT, "reporter_intake": {"text": "big hole, cars swerve"}}
    assert ra._verify_decision_rationale(dec, with_text)["ok"] is True

    lane = {"action": "standard", "priority": 45, "rationale": "D40 pothole affects the vehicle path on a main road.", "assessment": ""}
    assert ra._verify_decision_rationale(lane, _STATE_NO_TEXT)["problems"] == ["lane_blockage_not_in_form"]
    no_lane = {"action": "standard", "priority": 45, "rationale": "D40 pothole, no lane blockage.", "assessment": ""}
    assert ra._verify_decision_rationale(no_lane, _STATE_NO_TEXT)["ok"] is True
    partial = {**_STATE_NO_TEXT, "metadata": {"road_type": "main", "lane_blocked": "partial"}}
    assert ra._verify_decision_rationale(no_lane, partial)["problems"] == ["lane_free_claim_but_form_says_partial"]

    cls = {"action": "standard", "priority": 45, "rationale": "D20 alligator crack; routine queue.", "assessment": ""}
    assert ra._verify_decision_rationale(cls, _STATE_NO_TEXT)["problems"] == ["class_not_in_inputs:D20"]
    with_match = {**_STATE_NO_TEXT, "dataset_match": {"found": True, "match": "phash", "distance": 9, "pic_id": 1, "classes": "D00,D20",
                                                       "model_prediction": {"top1_class": "D20", "top1_confidence": 0.8}}}
    assert ra._verify_decision_rationale(cls, with_match)["ok"] is True

    known = {"action": "standard", "priority": 45, "rationale": "This is a known image in the catalogue.", "assessment": ""}
    assert ra._verify_decision_rationale(known, with_match)["problems"] == ["known_image_claim_without_sha256_match"]
    similar_ok = {"action": "standard", "priority": 45, "rationale": "The catalogued photo is similar but not the same file.", "assessment": ""}
    assert ra._verify_decision_rationale(similar_ok, with_match)["ok"] is True
    sha = {**with_match, "dataset_match": {**with_match["dataset_match"], "match": "sha256", "distance": 0}}
    assert ra._verify_decision_rationale(known, sha)["ok"] is True
    catalog = {"action": "standard", "priority": 45, "rationale": "catalog matches this pothole", "assessment": ""}
    assert ra._verify_decision_rationale(catalog, _STATE_NO_TEXT)["problems"] == ["catalogue_claim_without_match"]

    prio = {"action": "standard", "priority": 45, "rationale": "Routine repair queue. Priority 66.", "assessment": ""}
    assert ra._verify_decision_rationale(prio, _STATE_NO_TEXT)["problems"] == ["priority_text_66_vs_45"]
    ok = {"action": "standard", "priority": 45, "rationale": "Routine repair for the D40 pothole on the main road, priority 45.", "assessment": ""}
    assert ra._verify_decision_rationale(ok, _STATE_NO_TEXT)["ok"] is True


def test_deterministic_rationale_is_built_from_the_inputs():
    text = ra._deterministic_rationale({"action": "standard", "priority": 45}, _STATE_NO_TEXT)
    assert text.startswith("routine repair queue for D40 at detector confidence 0.47 on a main road, lane none")
    assert "no reporter text" in text and "research coverage direct_match" in text
    similar = {**_STATE_NO_TEXT, "dataset_match": {"found": True, "match": "phash", "distance": 15, "pic_id": 1024, "classes": "D10"}}
    assert "similar catalogued photo exists (pic_id=1024, phash distance 15), not the same file" in ra._deterministic_rationale({"action": "human_review"}, similar)


def test_trace_lines_show_the_rationale_check():
    """Round 21 (3.11.2026): the operator trace shows the FAIL line and the
    deterministic sentence, but no longer the rejected model text (kept on
    rationale_check.model_rationale / model_assessment for the audit only)."""
    failed = {"speaker": "decision", "kind": "final", "content": {
        "action": "standard", "priority": 45, "source": "llm_react", "rationale": "routine repair queue for D40 ...", "cited_sources": [],
        "rationale_check": {"ok": False, "problems": ["reporter_claims_without_text"], "model_rationale": "The reporter's claims are consistent"}}}
    text = "\n".join(ra._entry_lines(failed, []))
    assert "rationale check: FAIL (reporter_claims_without_text); deterministic sentence shown (model text kept on rationale_check for the audit)" in text
    assert "model rationale:" not in text
    both = {"speaker": "decision", "kind": "final", "content": {
        "action": "standard", "priority": 45, "source": "llm_react", "rationale": "deterministic", "cited_sources": [],
        "rationale_check": {"ok": False, "problems": ["reporter_claims_without_text"], "model_assessment": "The reporter's claims are unverified"}}}
    assert "model assessment:" not in "\n".join(ra._entry_lines(both, []))
    ok = {"speaker": "decision", "kind": "final", "content": {"action": "standard", "priority": 45, "source": "llm_react",
                                                              "rationale": "fine", "cited_sources": [], "rationale_check": {"ok": True, "problems": []}}}
    assert "rationale check: ok" in "\n".join(ra._entry_lines(ok, []))


def test_passage_support_grades_other_classes_and_coverage_uses_it():
    weak = ra._passage_relevance("Alligator cracking is repaired by patching; alligator cracks and a longitudinal crack appear.", "D00")
    assert weak["class_keyword_hit"] is True and weak["other_class_hits"] == ["D20"] and weak["support"] == "weak"
    assert weak["sentence_complete"] is True
    direct = ra._passage_relevance("Longitudinal cracks are sealed with bitumen emulsion.", "D00")
    assert direct["support"] == "direct" and direct["other_class_hits"] == []
    none = ra._passage_relevance("pothole patching with cold mix", "D00")
    assert none["support"] == "none" and none["class_keyword_hit"] is False and none["sentence_complete"] is False

    def passage(support, tag_hit=False):
        return {"hazard_types": ["D00"], "road_types": ["any"], "relevance": {"support": support, "tag_hit": tag_hit}}

    assert ra._coverage([passage("weak")], "D00", "main") == "partial_match"
    assert ra._coverage([passage("weak"), passage("direct")], "D00", "main") == "direct_match"
    assert ra._coverage([passage("none", tag_hit=True)], "D00", "main") == "direct_match"
    assert ra._coverage([{"hazard_types": ["D00"], "road_types": ["any"]}], "D00", "main") == "direct_match"
    assert ra._coverage([], "D00", "main") == "no_match"


def test_quarantine_summary_is_a_whole_sentence_up_to_320_characters():
    long = "Potholes occur on asphalt pavements. " * 20
    summary = ra._quarantine_extract(long)["trimmed_summary"]
    assert len(summary) <= 320 and summary.endswith(".") and not summary.endswith("..")
    assert ra._sentence_cut("short text", 50) == "short text"
    cut = ra._sentence_cut("word " * 100, 60)
    assert len(cut) <= 63 and cut.endswith("...")


def test_stage_seconds_follow_the_timestamps():
    trace = [
        {"ts": 100.0, "speaker": "sanitize", "kind": "boundary_defence", "content": {}},
        {"ts": 105.0, "speaker": "supervisor", "kind": "supervisor_plan", "content": {}},
        {"ts": 150.0, "speaker": "profile_builder", "kind": "handoff_out", "content": {}},
        {"ts": 180.0, "speaker": "rag_retriever", "kind": "handoff_out", "content": {}},
        {"ts": 240.0, "speaker": "decision", "kind": "final", "content": {"action": "standard", "priority": 45}},
    ]
    assert ra._stage_seconds(trace) == {"supervisor": 5.0, "profile_builder": 45.0, "rag_retriever": 30.0, "decision": 60.0}
    lines = ra.format_trace_lines("hz_s", trace, metadata={"class": "D40", "confidence": 0.5}, verdict={"action": "standard", "priority": 45}, status="open")
    assert any(l.strip().startswith("stage seconds: supervisor=5  profile_builder=45  rag_retriever=30  decision=60") for l in lines)


def test_agent_metrics_count_failures_and_auto_ran_tools_apart():
    entry = {"speaker": "profile_builder", "kind": "handoff_out", "content": {
        "autonomy": {"tools_used": ["inspect_image_quality"], "auto_ran": ["lookup_similar_cases"],
                     "failed_tools": ["recall_agent_notes"], "violations": ["tool_call_failed:recall_agent_notes"]}}}
    agg = api._aggregate_agent_metrics([{"agent_trace_json": json.dumps([entry]), "class": "D40", "confidence": 0.5}])
    pb = agg["autonomy"]["profile_builder"]
    assert pb["agent_called"] == 1 and pb["auto_ran"] == 1 and pb["auto_ran_tools"] == 1 and pb["tool_failures"] == 1
    assert pb["violations"] == 1 and pb["clean_runs"] == 0
