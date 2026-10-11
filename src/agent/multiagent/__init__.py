"""Orchestrator-Worker multi-agent subpackage.

Perception + Research + Decision agents coordinated by a lightweight
Python supervisor. See docs/multiagent-architecture.md for the full spec.

This subpackage is installed side-by-side with the monolith agent in
src/agent/{loop,policy,tools,prompts}.py. The API still points at the
monolith until a later phase swaps it over, so breaking changes here do
not affect production endpoints.
"""

__version__ = "0.2.0"
