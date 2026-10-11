"""Budget guard - locked at the top of every agent run.

Per DS23 S42 slide 9: autonomy is a budget, not a switch. Four caps:
MAX_ITERATIONS, MAX_TOOL_HOPS_PER_AGENT, COST_CAP_USD, WALL_CLOCK_CAP_SECONDS.
When any cap fires, the agent returns a legitimate human_review answer -
never a crash, never an unbounded run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

MAX_ITERATIONS = 8
MAX_TOOL_HOPS_PER_AGENT = 3
COST_CAP_USD = 0.10
WALL_CLOCK_CAP_SECONDS = 30.0

PRICE_IN = 0.59 / 1_000_000
PRICE_OUT = 0.79 / 1_000_000


class BudgetExceeded(Exception):
    """Raised when any of the four caps is hit."""


@dataclass
class BudgetTracker:
    """Shared across all nodes in a single run. Enforced by Supervisor."""

    iterations: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    started_at: float = field(default_factory=time.time)
    tool_hops_per_agent: dict[str, int] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT

    @property
    def wall_clock_seconds(self) -> float:
        return time.time() - self.started_at

    def check(self) -> tuple[bool, str]:
        """Return (fired, reason). Supervisor calls this before every hop."""
        if self.iterations >= MAX_ITERATIONS:
            return True, f"MAX_ITERATIONS ({MAX_ITERATIONS}) reached"
        if self.cost_usd >= COST_CAP_USD:
            return (
                True,
                f"COST_CAP_USD (${COST_CAP_USD}) reached (spent ${self.cost_usd:.4f})",
            )
        if self.wall_clock_seconds >= WALL_CLOCK_CAP_SECONDS:
            return True, f"WALL_CLOCK_CAP_SECONDS ({WALL_CLOCK_CAP_SECONDS}) reached"
        return False, ""

    def check_tool_hop(self, agent_name: str) -> tuple[bool, str]:
        """Per-agent tool-hop cap. Prevents one agent looping on same tool."""
        hops = self.tool_hops_per_agent.get(agent_name, 0)
        if hops >= MAX_TOOL_HOPS_PER_AGENT:
            return True, f"MAX_TOOL_HOPS_PER_AGENT ({MAX_TOOL_HOPS_PER_AGENT}) for {agent_name}"
        return False, ""

    def tick_iteration(self) -> None:
        self.iterations += 1

    def tick_tool_hop(self, agent_name: str) -> None:
        self.tool_hops_per_agent[agent_name] = (
            self.tool_hops_per_agent.get(agent_name, 0) + 1
        )

    def tick_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out


def graceful_stop_decision(reason: str) -> dict:
    """The legitimate human_review answer when guard fires.

    Per DS23 S43 slide 16: "I could not finish" is a legitimate answer,
    not a crash. The API returns 200 with this body.
    """
    return {
        "action": "human_review",
        "rationale": f"budget guard fired: {reason}",
        "priority": 50,
        "cited_sources": [],
        "guard_fired": True,
    }
