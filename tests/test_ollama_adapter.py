"""Ollama adapter: tool-call arguments go out as JSON objects (Ollama 0.40
returns HTTP 400 on the string form) and come back as JSON strings (the
internal OpenAI-style contract)."""

from __future__ import annotations

import json
import types

import httpx

import roadguard_agent as ra


class _Resp:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_assistant_tool_call_arguments_are_sent_as_objects(monkeypatch):
    seen: dict = {}

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - httpx signature
        seen["url"] = url
        seen["body"] = json
        return _Resp({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": {"x": "2"}}}],
            },
            "prompt_eval_count": 10,
            "eval_count": 3,
        })

    monkeypatch.setattr(httpx, "post", fake_post)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": \"1\"}"}},
            {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "not json"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "{\"ok\": true}"},
    ]
    snapshot = json.dumps(messages)
    msg, in_tok, out_tok = ra._chat_ollama(messages, tools=None, temperature=0.0, model="qwen2.5:3b")

    sent = seen["body"]["messages"]
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == {"x": "1"}
    assert sent[2]["tool_calls"][1]["function"]["arguments"] == {}
    assert sent[3] == messages[3]
    assert json.dumps(messages) == snapshot  # caller's list untouched
    assert seen["url"].endswith("/api/chat") and seen["body"]["model"] == "qwen2.5:3b"
    # Inbound normalisation unchanged: arguments come back as a JSON string.
    assert msg.finish_reason == "tool_calls"
    assert msg.tool_calls[0]["function"]["arguments"] == json.dumps({"x": "2"})
    assert (in_tok, out_tok) == (10, 3)


def test_ollama_messages_helper_leaves_plain_messages_alone():
    plain = [{"role": "user", "content": "hi"}]
    assert ra._ollama_messages(plain) == plain
    fixed = ra._ollama_messages([{"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}])
    assert fixed[0]["tool_calls"][0]["function"]["arguments"] == {}
