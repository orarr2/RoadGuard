"""Multi-provider chat routing: single, committee, panel modes."""

import json

import pytest

from agent.multiagent import chat as chat_mod
from agent.multiagent import providers
from agent.multiagent.providers import ChatMessage


def _msg(action: str = "standard", content_extra: dict | None = None) -> ChatMessage:
    body = {"action": action, "rationale": "x", "priority": 50, "cited_sources": []}
    if content_extra:
        body.update(content_extra)
    return ChatMessage(finish_reason="stop", content=json.dumps(body))


# ---------------------------------------------------------------------------
# resolve_provider_model
# ---------------------------------------------------------------------------


def test_resolve_provider_defaults_to_groq(monkeypatch):
    for k in ("AGENT_LLM_PROVIDER", "AGENT_LLM_MODEL", "GROQ_MODEL"):
        monkeypatch.delenv(k, raising=False)
    name, model = providers.resolve_provider_model()
    assert name == "groq"
    assert model == providers.DEFAULT_MODEL_BY_PROVIDER["groq"]


def test_resolve_provider_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openai_compat")
    monkeypatch.setenv("AGENT_LLM_MODEL", "llama-x")
    name, model = providers.resolve_provider_model()
    assert name == "openai_compat"
    assert model == "llama-x"


def test_resolve_provider_colon_shorthand(monkeypatch):
    for k in ("AGENT_LLM_MODEL", "GROQ_MODEL"):
        monkeypatch.delenv(k, raising=False)
    name, model = providers.resolve_provider_model("ollama:llama3.2")
    assert name == "ollama"
    assert model == "llama3.2"


def test_resolve_provider_unknown_raises(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "nope")
    monkeypatch.setenv("AGENT_LLM_MODEL", "whatever")
    with pytest.raises(RuntimeError, match="unknown provider"):
        providers.resolve_provider_model()


# ---------------------------------------------------------------------------
# chat() routing with provider patching
# ---------------------------------------------------------------------------


def _install_fake_providers(monkeypatch, scripted):
    """Replace PROVIDERS registry with a dict that pops scripted replies
    off a queue keyed by (provider_name, model)."""
    calls: list[tuple[str, str]] = []

    def _factory(name):
        def _fn(messages, tools, temperature, model):
            calls.append((name, model))
            key = f"{name}:{model}"
            if key in scripted and scripted[key]:
                reply = scripted[key].pop(0)
            else:
                reply = _msg("standard")
            return reply, 10, 5

        return _fn

    monkeypatch.setattr(
        providers,
        "PROVIDERS",
        {name: _factory(name) for name in ("groq", "openai_compat", "ollama")},
    )
    return calls


def test_chat_live_single_provider_default(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "groq")
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("AGENT_LLM_PANEL", raising=False)
    monkeypatch.delenv("AGENT_LLM_COMMITTEE", raising=False)
    # Reload module-level USE_LIVE
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    calls = _install_fake_providers(
        monkeypatch,
        {"groq:" + providers.DEFAULT_MODEL_BY_PROVIDER["groq"]: [_msg("urgent")]},
    )
    msg, _in, _out = chat_mod.chat(messages=[{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    assert calls[0][0] == "groq"
    body = json.loads(msg.content)
    assert body["action"] == "urgent"
    assert "committee_note" not in body  # single provider: no note


def test_chat_committee_agreement(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv("AGENT_LLM_COMMITTEE", "1")
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "groq:llama-a")
    monkeypatch.setenv("AGENT_LLM_COMMITTEE_MODEL_B", "llama-b")
    monkeypatch.delenv("AGENT_LLM_PANEL", raising=False)
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    calls = _install_fake_providers(
        monkeypatch,
        {
            "groq:llama-a": [_msg("standard")],
            "groq:llama-b": [_msg("standard")],
        },
    )
    msg, in_tok, out_tok = chat_mod.chat(
        messages=[{"role": "user", "content": "x"}], use_panel=True
    )
    assert len(calls) == 2
    body = json.loads(msg.content)
    assert body["action"] == "standard"
    assert "agree" in body["committee_note"]
    assert in_tok == 20 and out_tok == 10


def test_chat_committee_disagreement_picks_severe(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv("AGENT_LLM_COMMITTEE", "1")
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "groq:llama-a")
    monkeypatch.setenv("AGENT_LLM_COMMITTEE_MODEL_B", "llama-b")
    monkeypatch.delenv("AGENT_LLM_PANEL", raising=False)
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    _install_fake_providers(
        monkeypatch,
        {
            "groq:llama-a": [_msg("standard")],
            "groq:llama-b": [_msg("urgent")],
        },
    )
    msg, *_ = chat_mod.chat(
        messages=[{"role": "user", "content": "x"}], use_panel=True
    )
    body = json.loads(msg.content)
    assert body["action"] == "urgent"  # severity tiebreak
    assert "split" in body["committee_note"]


def test_chat_panel_majority_vote(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv(
        "AGENT_LLM_PANEL",
        "groq:llama-a,openai_compat:llama-b,ollama:llama-c",
    )
    monkeypatch.delenv("AGENT_LLM_COMMITTEE", raising=False)
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    _install_fake_providers(
        monkeypatch,
        {
            "groq:llama-a": [_msg("standard")],
            "openai_compat:llama-b": [_msg("standard")],
            "ollama:llama-c": [_msg("urgent")],
        },
    )
    msg, *_ = chat_mod.chat(
        messages=[{"role": "user", "content": "x"}], use_panel=True
    )
    body = json.loads(msg.content)
    assert body["action"] == "standard"  # 2 of 3
    assert "panel" in body["committee_note"]


def test_chat_panel_dedupes_identical_specs(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv(
        "AGENT_LLM_PANEL",
        "groq:llama-a,groq:llama-a,openai_compat:llama-b",
    )
    monkeypatch.delenv("AGENT_LLM_COMMITTEE", raising=False)
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    calls = _install_fake_providers(
        monkeypatch,
        {
            "groq:llama-a": [_msg("standard")],
            "openai_compat:llama-b": [_msg("urgent")],
        },
    )
    chat_mod.chat(messages=[{"role": "user", "content": "x"}], use_panel=True)
    # Only 2 distinct specs should have been called
    assert len(calls) == 2


def test_chat_offline_when_use_live_false(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "0")
    monkeypatch.setattr(chat_mod, "USE_LIVE", False)
    # Panel env should be ignored when offline
    monkeypatch.setenv("AGENT_LLM_PANEL", "groq:a,groq:b")
    msg, in_tok, out_tok = chat_mod.chat(
        messages=[{"role": "user", "content": "hi"}],
    )
    # Offline stub returns one of the fixed strings; it must not call real providers.
    assert isinstance(msg, ChatMessage)
    assert in_tok > 0 and out_tok > 0


def test_chat_panel_ignored_without_use_panel_flag(monkeypatch):
    """Panel/committee env fires ONLY when caller asks (Decision node)."""
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv(
        "AGENT_LLM_PANEL",
        "groq:llama-a,openai_compat:llama-b,ollama:llama-c",
    )
    monkeypatch.delenv("AGENT_LLM_COMMITTEE", raising=False)
    monkeypatch.delenv("AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    calls = _install_fake_providers(
        monkeypatch,
        {"groq:" + providers.DEFAULT_MODEL_BY_PROVIDER["groq"]: [_msg("standard")]},
    )
    # Caller did NOT pass use_panel; must stay single-provider.
    chat_mod.chat(messages=[{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_chat_committee_ignored_without_use_panel_flag(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv("AGENT_LLM_COMMITTEE", "1")
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "groq:llama-a")
    monkeypatch.setenv("AGENT_LLM_COMMITTEE_MODEL_B", "llama-b")
    monkeypatch.delenv("AGENT_LLM_PANEL", raising=False)
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    calls = _install_fake_providers(
        monkeypatch,
        {
            "groq:llama-a": [_msg("standard")],
            "groq:llama-b": [_msg("urgent")],
        },
    )
    chat_mod.chat(messages=[{"role": "user", "content": "x"}])
    # Single call, not two
    assert len(calls) == 1
