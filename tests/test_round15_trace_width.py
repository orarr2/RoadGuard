"""Round 15 (1.11.2026): the trace keeps tool results, payloads and RAG
passages at the length the agent saw (up to 2,000 characters) instead of
220 / 320 / 160-character stubs."""
import json

import roadguard_agent as ra


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(finish_reason="tool_calls" if tool_calls else "stop", content=content, tool_calls=tool_calls or [])


def test_worker_step_keeps_the_full_tool_result(monkeypatch):
    long_passage = "Potholes occur on asphalt-surfaced pavements. " * 30  # about 1,400 characters

    def query(hazard_class: str, road_type: str = "main", k: int = 3) -> dict:
        return {"ok": True, "procedures": [{"passage": long_passage, "article_ids": ["a1"]}], "tool_result_ids": ["a1"], "coverage": "direct_match"}

    registry = {"query_repair_standards": (query, ra.RESEARCH_WORKER_TOOLS["query_repair_standards"][1])}
    final = json.dumps({"coverage": "direct_match", "confidence": "high", "key_findings": []})
    replies = [
        _msg(tool_calls=[{"id": "c1", "function": {"name": "query_repair_standards", "arguments": json.dumps({"hazard_class": "D40", "road_type": "main"})}}]),
        _msg(content=final),
    ]
    n = []
    monkeypatch.setattr(ra, "chat", lambda messages, tools=None, temperature=0.0, use_panel=False, provider_override=None, **kw: (n.append(1), replies[len(n) - 1], 5, 5)[1:])
    _parsed, _, _, info = ra._react_worker_rounds(
        agent_name="rag_retriever", system_prompt="s", user_payload={}, tool_registry=registry,
        state={"hazard_id": "hz_t", "metadata": {}}, provider_override=None, max_hops=2,
    )
    stored = info["steps"][0]["content"]["result"]
    assert "Potholes occur on asphalt-surfaced pavements. " * 20 in stored
    assert not stored.endswith("...") or len(stored) >= 1990
    text = "\n".join(ra._entry_lines(info["steps"][0], []))
    assert "Potholes occur on asphalt-surfaced pavements. " * 20 in text


def test_react_step_lines_print_up_to_two_thousand_characters():
    big = {"ok": True, "procedures": [{"passage": "x" * 1500}]}
    step = {"speaker": "decision", "kind": "react_step", "content": {"hop": 1, "tool": "lookup_dataset_image", "args": {"why": "w"}, "ok": True, "result": big, "duration_s": 0.1}}
    text = "\n".join(ra._entry_lines(step, []))
    assert "x" * 1500 in text


def test_passage_previews_keep_the_retrieved_text(monkeypatch):
    long_passage = "Alligator cracking is repaired by patching the base. " * 20  # about 1,060 characters
    monkeypatch.setattr(ra, "_rag_llm_enabled", lambda: False)
    monkeypatch.setattr(ra, "worker_autonomy_enabled", lambda: False)
    monkeypatch.setitem(ra.RESEARCH_TOOLS, "query_repair_standards", (
        lambda hazard_class, road_type, k=3: {"procedures": [{"passage": long_passage, "article_ids": ["a1"], "hazard_types": ["D20"], "road_types": ["any"], "distance": 0.4, "relevance": {"class_keyword_hit": True, "repair_term_hit": True}}],
                                             "tool_result_ids": ["a1"], "coverage": "direct_match"},
        ra.RESEARCH_TOOLS["query_repair_standards"][1],
    ))
    state = {"hazard_id": "hz_t", "hazard_profile": {"primary_class": "D20"}, "metadata": {"road_type": "main"},
             "yolo_output": {"top1_class": "D20", "top1_confidence": 0.9}, "image_quality": {"blur_score": 150, "is_road_scene": True},
             "perception_confidence": "high", "iterations": 2}
    out = ra.rag_retriever_node(state)
    preview = [e for e in out["trace"] if e.get("kind") == "handoff_out"][0]["content"]["passage_previews"][0]["preview"]
    assert preview == long_passage.strip()
