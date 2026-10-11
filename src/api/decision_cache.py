"""In-process TTL cache of agent decisions keyed by streaming session.

The live-drive flow fires many submissions per minute while the car
passes a hazard. The IoU tracker on the client already collapses them
to a single track_id, but the client-side tracker drops when the user
looks away, so the same canonical hazard can resurface a few seconds
later with a new track_id. The server-side cache is the second line
of defence: within a short TTL, two (session, class, road_type)
submissions resolve to the same verdict without a second agent call.

The cache is per process. On the single-worker uvicorn running on the
e2-micro that is enough. If the service ever moves behind a load
balancer this needs to swap to a shared backend (Redis).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from agent.loop import AgentDecision


DEFAULT_TTL_SECONDS = 120
DEFAULT_MAX_ENTRIES = 2000


@dataclass
class _Entry:
    decision: AgentDecision
    expires_at: float


class DecisionCache:
    """TTL cache with FIFO eviction when the map grows past ``max_entries``.

    Not meant to be a general-purpose cache: it stores AgentDecision only,
    the key is an opaque string the API builds, and the TTL is a single
    value per cache instance.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.ttl_seconds = int(ttl_seconds)
        self.max_entries = int(max_entries)
        self._store: dict[str, _Entry] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()

    def get(self, key: str, now: Optional[float] = None) -> Optional[AgentDecision]:
        now = time.time() if now is None else now
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < now:
                self._store.pop(key, None)
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
                return None
            return entry.decision

    def put(
        self,
        key: str,
        decision: AgentDecision,
        now: Optional[float] = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._store[key] = _Entry(
                decision=decision, expires_at=now + self.ttl_seconds
            )
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            while len(self._store) > self.max_entries:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)

    @staticmethod
    def canonical_key(
        session_id: str,
        hazard_class: str,
        road_type: str,
        lane_blocked: str,
        track_id: Optional[str] = None,
    ) -> str:
        """Build a stable cache key for the live-drive flow.

        The track_id (when supplied) narrows the key so two distinct
        trackers that happen to see the same class on the same road
        get independent decisions, which matches reporter expectations.
        """
        track = (track_id or "nt").replace("|", "_")
        return (
            f"{session_id}|{track}|{hazard_class}|{road_type}|{lane_blocked}"
        )


# The API hands this single instance into endpoints via a dependency.
_shared_cache = DecisionCache()


def get_decision_cache() -> DecisionCache:
    return _shared_cache


def reset_decision_cache() -> None:
    """Flush the shared cache. Tests use this between runs."""
    _shared_cache.clear()
