"""The in-process DecisionCache: TTL, eviction, key canonicalization."""

from agent.loop import AgentDecision
from api.decision_cache import DecisionCache


def _decision(action="standard", priority=50):
    return AgentDecision(
        action=action,
        rationale="r",
        priority=priority,
        cited_sources=["https://x"],
        provider="multiagent",
    )


def test_put_then_get_within_ttl():
    cache = DecisionCache(ttl_seconds=60)
    cache.put("k1", _decision(), now=1000.0)
    assert cache.get("k1", now=1030.0).priority == 50


def test_expired_entry_is_evicted_on_read():
    cache = DecisionCache(ttl_seconds=60)
    cache.put("k1", _decision(), now=1000.0)
    assert cache.get("k1", now=1061.0) is None
    assert len(cache) == 0


def test_canonical_key_components():
    k = DecisionCache.canonical_key(
        session_id="s1",
        hazard_class="D40",
        road_type="main",
        lane_blocked="full",
        track_id="t-7",
    )
    assert k == "s1|t-7|D40|main|full"


def test_canonical_key_without_track_id_uses_placeholder():
    k = DecisionCache.canonical_key(
        session_id="s1",
        hazard_class="D40",
        road_type="main",
        lane_blocked="full",
    )
    assert k.split("|")[1] == "nt"


def test_cache_evicts_oldest_when_over_cap():
    cache = DecisionCache(ttl_seconds=600, max_entries=3)
    for i in range(5):
        cache.put(f"k{i}", _decision(priority=i), now=1000.0 + i)
    assert len(cache) == 3
    assert cache.get("k0", now=1010.0) is None
    assert cache.get("k4", now=1010.0).priority == 4


def test_put_same_key_refreshes_entry():
    cache = DecisionCache(ttl_seconds=60)
    cache.put("k1", _decision(priority=1), now=1000.0)
    cache.put("k1", _decision(priority=2), now=1050.0)
    # Original would have expired at 1060; refresh pushes it to 1110
    assert cache.get("k1", now=1070.0).priority == 2


def test_clear_removes_all_entries():
    cache = DecisionCache(ttl_seconds=60)
    for i in range(3):
        cache.put(f"k{i}", _decision(), now=1000.0)
    cache.clear()
    assert len(cache) == 0
