"""Round 10b (1.11.2026): two faults seen live in run r7.

A. The model sent its verdict JSON as the arguments of a check_hazard_severity
   call; the tool computed priority 10 for the empty class and the verdict
   carried it. Now tool arguments are validated against the schema, and a
   final answer that arrives as a tool call is parsed as the verdict.
B. A perceptual-hash neighbour (a different, similar photo) reached the model
   as `dataset_match`, and it called the neighbour's labels a catalogue
   contradiction. Now only the same file (sha256) is `dataset_match`; the
   neighbour travels as `similar_catalogue_photo` with its distance.
"""
import json

import roadguard_agent as ra


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(finish_reason="tool_calls" if tool_calls else "stop", content=content, tool_calls=tool_calls or [])


def test_invalid_severity_arguments_are_rejected_instead_of_scored():
    state = {"hazard_id": "hz_t", "metadata": {}}
    bad = {"cited_sources": ["a"], "assessment": "looks bad", "uncertainty": "..."}
    result = ra._run_decision_tool("check_hazard_severity", bad, state, {})
    assert result["ok"] is False and "invalid arguments" in result["error"]
    assert "hazard_class is required" in result["error"]
    wrong_class = {"hazard_class": "D99", "confidence": 0.5, "road_type": "main", "lane_blocked": "none"}
    result2 = ra._run_decision_tool("check_hazard_severity", wrong_class, state, {})
    assert result2["ok"] is False and "hazard_class must be one of" in result2["error"]
    good = {"hazard_class": "D40", "confidence": 0.5, "road_type": "main", "lane_blocked": "none"}
    result3 = ra._run_decision_tool("check_hazard_severity", good, state, {})
    assert result3["ok"] is True and result3["priority"] == ra.check_hazard_severity("D40", 0.5, "main", "none")["priority"]


def test_validate_tool_args_covers_enum_range_type_and_state_filled_keys():
    schema = ra.DECISION_TOOLS["request_additional_evidence"][1]
    # hazard_id is filled from the state, so its absence is not a problem.
    assert ra._validate_tool_args(schema, {"missing_field": "second_angle", "reason_shown_to_reporter": "x"}) == []
    problems = ra._validate_tool_args(schema, {"missing_field": "the_moon", "reason_shown_to_reporter": 5})
    assert any("missing_field must be one of" in p for p in problems)
    assert any("reason_shown_to_reporter must be string" in p for p in problems)
    sev = ra.DECISION_TOOLS["check_hazard_severity"][1]
    assert any("confidence must be <= 1" in p for p in ra._validate_tool_args(sev, {"hazard_class": "D40", "confidence": 1.5, "road_type": "main", "lane_blocked": "none"}))
    assert ra._validate_tool_args(ra.DECISION_TOOLS["lookup_dataset_image"][1], {}) == []


def test_final_answer_sent_as_a_tool_call_is_parsed_as_the_verdict(monkeypatch):
    monkeypatch.setattr(ra, "_decision_messages", lambda state, hint: [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    monkeypatch.setattr(ra, "_run_deadline", lambda: None)
    executed = []
    monkeypatch.setattr(ra, "_run_decision_tool", lambda name, args, state, tools_run: executed.append(name) or {"ok": True, "priority": 10})
    verdict = {"action": "standard", "rationale": "routine", "priority": 55, "cited_sources": ["a1"], "missing_evidence": [],
               "assessment": "the pothole is routine", "uncertainty": ""}
    seen = []

    def fake_chat(messages, tools=None, temperature=0.0, **kw):
        seen.append(tools)
        return _msg(tool_calls=[{"id": "c1", "function": {"name": "check_hazard_severity", "arguments": json.dumps(verdict)}}]), 10, 5

    monkeypatch.setattr(ra, "chat", fake_chat)
    monkeypatch.setattr(ra, "_grounded_or_none", lambda parsed, ids, state=None: parsed)
    state = {"hazard_id": "hz_t", "tool_result_ids": ["a1"], "metadata": {}, "budget": None, "iterations": 0}
    decision, _, _, info = ra._llm_decide_react(state, {"action": "standard"})
    assert decision is not None and decision["action"] == "standard" and decision["priority"] == 55
    assert executed == [], "nothing may run with the verdict as arguments"
    assert len(seen) == 1 and info["hops"] == 0
    assert info["final_in_tool_call"] == "check_hazard_severity"
    assert info["calls"][0]["executed"] is False and "parsed as the verdict" in info["calls"][0]["error"]


def test_looks_like_final_answer():
    assert ra._looks_like_final_answer({"action": "human_review", "rationale": "x"})
    assert ra._looks_like_final_answer({"action": "standard", "assessment": "x", "priority": 3})
    assert not ra._looks_like_final_answer({"hazard_class": "D40", "confidence": 0.5, "road_type": "main", "lane_blocked": "none"})
    assert not ra._looks_like_final_answer({"action": "fly", "rationale": "x"})
    assert not ra._looks_like_final_answer({"action": "standard"})


_BASE = {
    "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.5, "recurrence_count_30d": 0},
    "metadata": {"road_type": "main", "lane_blocked": "none"},
    "procedures": [], "tool_result_ids": [],
}
_HINT = {"action": "standard", "priority": 55, "rationale": "r", "cited_sources": [], "missing_evidence": []}


def _match(kind, distance):
    return {"found": True, "match": kind, "distance": distance, "pic_id": 1024, "split": "train", "file_name": "f.jpg",
            "classes": "D10", "n_labels": 1, "model_prediction": {"top1_class": "D10", "top1_confidence": 0.76}}


def test_a_similar_photo_is_not_presented_as_the_catalogued_photo():
    """Round 20 (3.11.2026): phash neighbours are hidden from the Decision
    prompt entirely. The payload tells the model only that an approximate
    match exists, without the pic_id, labels or recorded prediction."""
    payload = json.loads(ra._decision_messages({**_BASE, "dataset_match": _match("phash", 15)}, _HINT)[1]["content"])
    assert payload["dataset_match"] is None
    sim = payload["similar_catalogue_photo"]
    assert "pic_id" not in sim and "its_labels" not in sim
    assert "its_recorded_prediction" not in sim and "phash_distance" not in sim
    assert "HIDDEN" in sim["note"] and "approximate" in sim["note"].lower()

    same = json.loads(ra._decision_messages({**_BASE, "dataset_match": _match("sha256", 0)}, _HINT)[1]["content"])
    assert same["similar_catalogue_photo"] is None
    assert same["dataset_match"]["already_run"] is True and same["dataset_match"]["pic_id"] == 1024

    none = json.loads(ra._decision_messages({**_BASE, "dataset_match": None}, _HINT)[1]["content"])
    assert none["dataset_match"] is None and none["similar_catalogue_photo"] is None

    prompt = ra._build_decision_prompt()
    assert "similar_catalogue_photo" in prompt and "not a catalogue contradiction" in prompt
