"""Human-readable agent trace with speaker labels.

Per DS23 S43 slide 15: a trace turns the agent from a black box into
something you can read like a protocol. Each entry has a speaker, an
intent, and the raw tool call or final answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceEntry:
    iter: int
    speaker: str
    kind: str
    content: Any
    cost_so_far: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trace:
    entries: list[TraceEntry] = field(default_factory=list)

    def tool(
        self,
        iter_n: int,
        speaker: str,
        tool_name: str,
        args: dict,
        result: Any,
        cost: float = 0.0,
    ) -> None:
        self.entries.append(
            TraceEntry(
                iter=iter_n,
                speaker=speaker,
                kind="tool",
                content={"tool": tool_name, "args": args, "result": result},
                cost_so_far=cost,
            )
        )

    def handoff(
        self,
        iter_n: int,
        from_agent: str,
        to_agent: str,
        payload_summary: str,
    ) -> None:
        self.entries.append(
            TraceEntry(
                iter=iter_n,
                speaker="mediator",
                kind="handoff",
                content={"from": from_agent, "to": to_agent, "summary": payload_summary},
            )
        )

    def final(self, iter_n: int, speaker: str, decision: dict) -> None:
        self.entries.append(
            TraceEntry(iter=iter_n, speaker=speaker, kind="final", content=decision)
        )

    def guard(self, iter_n: int, reason: str) -> None:
        self.entries.append(
            TraceEntry(
                iter=iter_n,
                speaker="budget_guard",
                kind="guard_fired",
                content={"reason": reason},
            )
        )

    def print(self) -> None:
        """Reads like a protocol. Great for S44-style demos."""
        for e in self.entries:
            tag = f"[iter {e.iter:>2}] [{e.speaker:<12}]"
            if e.kind == "tool":
                c = e.content
                print(f"{tag} {c['tool']}({_short(c['args'])}) -> {_short(c['result'])}")
            elif e.kind == "handoff":
                c = e.content
                print(f"{tag} HANDOFF {c['from']} -> {c['to']}: {c['summary']}")
            elif e.kind == "final":
                print(
                    f"{tag} FINAL: action={e.content.get('action')} "
                    f"priority={e.content.get('priority')}"
                )
            elif e.kind == "guard_fired":
                print(f"{tag} GUARD FIRED: {e.content['reason']}")

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]


def _short(obj: Any, limit: int = 60) -> str:
    s = repr(obj)
    return s if len(s) <= limit else s[: limit - 3] + "..."
