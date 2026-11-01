"""Round 14 (1.11.2026): resume_graph stores the first pass's inputs before
the new evidence replaces them, so the evidence node's delta is real and
the reuse of perception and research can fire. In r6 to r8 every delta read
"None -> x" because previous_yolo / previous_metadata were never written."""
import types

import roadguard_agent as ra


def test_resume_graph_stores_the_previous_inputs_before_updating(monkeypatch):
    calls = {}

    class FakeApp:
        def get_state(self, config):
            return types.SimpleNamespace(values={
                "yolo_output": {"top1_class": "D40", "top1_confidence": 0.5},
                "metadata": {"image_sha256": "a" * 64, "lane_blocked": "none", "road_type": "main"},
                "hazard_profile": {"primary_class": "D40"},
            })

        def update_state(self, config, values):
            calls["config"] = config
            calls["values"] = values

        def invoke(self, state, config=None):
            calls["invoked"] = (state, config)
            return {"action": "standard"}

    monkeypatch.setattr(ra, "build_graph", lambda *a, **k: FakeApp())
    monkeypatch.setattr(ra, "thread_waits_for_evidence", lambda hid: True)
    # Round 17: the same file with no new text is refused as evidence, so
    # the resumed pass here carries a new photo.
    new_md = {"image_sha256": "b" * 64, "lane_blocked": "none", "road_type": "main", "location": "Herzl 10"}
    out = ra.resume_graph("hz_t", {"top1_class": "D40", "top1_confidence": 0.5}, {"blur_score": 150}, new_md)
    assert out == {"action": "standard"}
    assert calls["values"]["previous_yolo"] == {"top1_class": "D40", "top1_confidence": 0.5}
    assert calls["values"]["previous_metadata"]["image_sha256"] == "a" * 64
    assert calls["values"]["metadata"] == new_md
    assert calls["invoked"] == (None, {"configurable": {"thread_id": "hz_t"}})

    # The evidence node then sees a re-upload (the same file, sent with a
    # note since round 17) and keeps the workers.
    state = {
        **calls["values"],
        "metadata": dict(calls["values"]["previous_metadata"]),
        "hazard_profile": {"primary_class": "D40"}, "procedures": [{"article_ids": ["a1"]}],
        "evidence_request": {"missing_field": "location"}, "evidence_round": 0, "iterations": 3, "tokens_in": 0, "tokens_out": 0,
    }
    node = ra._await_evidence_node(state)
    assert node["evidence_reuse"] is True
    assert node["trace"][0]["content"]["delta"] == {}


def test_resume_graph_without_previous_values_still_resumes(monkeypatch):
    class BareApp:
        def get_state(self, config):
            raise RuntimeError("no snapshot")

        def update_state(self, config, values):
            self.values = values

        def invoke(self, state, config=None):
            return {"action": "standard", "previous": self.values["previous_metadata"]}

    monkeypatch.setattr(ra, "build_graph", lambda *a, **k: BareApp())
    monkeypatch.setattr(ra, "thread_waits_for_evidence", lambda hid: True)
    out = ra.resume_graph("hz_t", {}, {}, {"image_sha256": "b" * 64})
    assert out["previous"] == {}
