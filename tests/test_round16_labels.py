"""Round 16 (1.11.2026): the owner's six items from his r9 review. A reply in
a tool envelope is a reply, not a failed tool; result lines carry
tool_failures / reply_via; the autonomy line separates the channels and
shows the time against the cap (now 90 s); RAG lines say which decision
questions the passages answer; a perceptual-hash match is a SIMILAR IMAGE."""
import json

import roadguard_agent as ra


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(finish_reason="tool_calls" if tool_calls else "stop", content=content, tool_calls=tool_calls or [])


def _call(name, args, call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def test_worker_reply_through_a_lookup_tool_is_the_reply(monkeypatch):
    """Round 18 (2.11.2026) retried a final answer that rode a lookup tool;
    round 29 (6.11.2026 audit F6) uses the worker's own reply as the reply
    (reply_in_tool_call, the round 16 label), nothing runs, and the autonomy
    record does not count it as a failed tool."""
    ran = []

    def similar(location: str, hazard_class: str, state: dict | None = None) -> dict:
        ran.append(1)
        return {"ok": True, "count": 0, "cases": [], "is_recurring_location": False}

    registry = {"lookup_similar_cases": (similar, ra.PERCEPTION_WORKER_TOOLS["lookup_similar_cases"][1])}
    reply = {"severity_signal": "minor", "severity_rationale": "D40", "confidence": "low",
             "needs_more_evidence": False, "missing_fields": []}
    replies = iter([
        _msg(tool_calls=[_call("lookup_similar_cases", reply)]),
    ])
    monkeypatch.setattr(ra, "chat", lambda messages, tools=None, temperature=0.0, use_panel=False, provider_override=None, **kw: (next(replies), 10, 5))
    parsed, _, _, info = ra._llm_react_worker(
        agent_name="profile_builder", system_prompt="s", user_payload={}, tool_registry=registry,
        state={"hazard_id": "hz_t", "metadata": {}}, provider_override=None, max_hops=2, time_cap_s=90.0,
    )
    assert parsed == reply and ran == []
    first = info["calls"][0]
    assert first["ok"] is False and first.get("reply_in_tool_call") is True
    rec = ra._autonomy_record(parsed, info, required=("inspect_image_quality",))
    assert "tool_call_failed:lookup_similar_cases" not in rec["violations"]

def test_result_lines_carry_tool_failures_and_reply_via():
    prof = {"speaker": "profile_builder", "kind": "handoff_out", "content": {
        "primary_class": "D40", "severity": "minor", "confidence": "low", "llm_used": True,
        "autonomy": {"mode": "autonomous", "tools_used": ["inspect_image_quality"], "declared_tools": [], "hops": 2,
                     "failed_tools": ["recall_agent_notes"], "final_in_tool_call": True,
                     "time_used_s": 44.3, "time_cap_s": 90.0, "cut_by_cap": False,
                     "violations": ["tool_call_failed:recall_agent_notes"]}}}
    lines = ra._entry_lines(prof, [])
    assert lines[0].endswith("llm_used=True  tool_failures=1  reply_via=tool_call")
    auto = [l for l in lines if "autonomy:" in l][0]
    assert "agent_called=inspect_image_quality  hops=2  failed=recall_agent_notes  reply_via=tool_call  time=44s/90s  source=agent_choice  violations=tool_call_failed:recall_agent_notes" in auto
    assert "tools=" not in auto

    cut = {"speaker": "rag_retriever", "kind": "handoff_out", "content": {
        "coverage": "direct_match", "n_procedures": 3, "tool_result_ids": ["a1"], "llm_used": True,
        "decision_coverage": {"answered": ["repair_window", "traffic_control"], "missing": ["urgency", "safety_risk"]},
        "autonomy": {"tools_used": [], "hops": 0, "auto_ran": ["query_repair_standards"], "fallback": "deterministic_query (3 passages)",
                     "time_used_s": 90.4, "time_cap_s": 90.0, "cut_by_cap": True, "violations": ["required_not_called:query_repair_standards", "time_cap"]}}}
    lines2 = ra._entry_lines(cut, ["a1"])
    # Round 19: the RAG line now shows topic/sufficient apart (direct_match
    # is the topic axis; sufficient reflects whether the questions are answered).
    assert "topic=direct_match  sufficient=no  answers=repair_window,traffic_control  missing=urgency,safety_risk  llm_used=True" in lines2[0]
    auto2 = [l for l in lines2 if "autonomy:" in l][0]
    # Round 20: auto_ran without an agent-called hop prints source=deterministic_fallback.
    assert "agent_called=none  hops=0  auto_ran=query_repair_standards  fallback=deterministic_query (3 passages)  time=90s/90s (cut by cap)  source=deterministic_fallback  violations=" in auto2

    dec = {"speaker": "decision", "kind": "final", "content": {
        "action": "standard", "priority": 45, "source": "llm_react", "rationale": "r", "cited_sources": [],
        "react": {"hops": 0, "final_in_tool_call": True,
                  "calls": [{"tool": "check_hazard_severity", "ok": False, "executed": False, "reply_in_tool_call": True, "args": {}, "error": "x"},
                            {"tool": "lookup_dataset_image", "ok": False, "executed": True, "args": {}, "error": "boom"}]}}}
    lines3 = ra._entry_lines(dec, [])
    assert lines3[0].endswith("source=llm_react  tool_failures=1  reply_via=tool_call")
    text3 = "\n".join(lines3)
    assert "reply arrived as a tool call (check_hazard_severity); parsed as the verdict, no tool ran" in text3
    assert "tool called: lookup_dataset_image (failed)" in text3


def test_decision_coverage_names_the_questions_the_passages_answer():
    procs = [
        {"passage": "Potholes on category 1 roads shall be made safe within 24 hours; place warning signs and cones around the defect."},
        {"passage": "Patching with cold mix is the routine repair method."},
    ]
    cov = ra._decision_coverage(procs)
    assert cov["repair_window"] is True and cov["traffic_control"] is True and cov["urgency"] is True
    assert cov["safety_risk"] is False
    assert cov["answered"] == ["repair_window", "urgency", "traffic_control"] and cov["missing"] == ["safety_risk"]
    thin = ra._decision_coverage([{"passage": "Compaction provides a tighter patch for traffic than loose material."}])
    assert thin["answered"] == [] and len(thin["missing"]) == 4
    assert ra._decision_coverage([])["missing"] == ["repair_window", "urgency", "traffic_control", "safety_risk"]


def test_decision_payload_carries_the_coverage_detail():
    state = {"hazard_profile": {"primary_class": "D40", "primary_confidence": 0.5}, "metadata": {"road_type": "main", "lane_blocked": "none"},
             "procedures": [], "tool_result_ids": [], "research_coverage": "direct_match",
             "research_coverage_detail": {"repair_window": True, "urgency": False, "traffic_control": False, "safety_risk": True,
                                          "answered": ["repair_window", "safety_risk"], "missing": ["urgency", "traffic_control"]}}
    payload = json.loads(ra._decision_messages(state, {"action": "standard", "priority": 55})[1]["content"])
    assert payload["research_coverage_detail"] == {"answered": ["repair_window", "safety_risk"], "missing": ["urgency", "traffic_control"]}
    without = json.loads(ra._decision_messages({**state, "research_coverage_detail": None}, {"action": "standard", "priority": 55})[1]["content"])
    assert without["research_coverage_detail"] is None


def test_similar_image_wording_and_live_note():
    base = {"speaker": "profile_builder", "kind": "handoff_out", "content": {
        "primary_class": "D40", "severity": "minor", "confidence": "medium", "llm_used": False,
        "dataset_lookup": {"found": True, "match": "phash", "distance": 9, "pic_id": 1381, "split": "train", "file_name": "f.jpg", "classes": "D20", "n_labels": 2, "model_prediction": "D20 0.88"}}}
    text = "\n".join(ra._entry_lines(base, []))
    assert "dataset: SIMILAR IMAGE pic_id=1381" in text and "KNOWN IMAGE" not in text
    same = {**base, "content": {**base["content"], "dataset_lookup": {**base["content"]["dataset_lookup"], "match": "sha256", "distance": 0}}}
    assert "dataset: KNOWN IMAGE pic_id=1381" in "\n".join(ra._entry_lines(same, []))
    note = {"speaker": "decision", "kind": "tool_call", "hop": 0, "tool": "lookup_dataset_image", "ok": True,
            "note": "similar catalogued photo pic_id=1598 phash distance 15 (not the same file)"}
    assert "tool: similar catalogued photo pic_id=1598 phash distance 15 (not the same file)" in "\n".join(ra._entry_lines(note, []))


def test_worker_time_cap_is_ninety_seconds():
    assert ra._policy_latency()["worker_time_cap_s"] == 90
