# AGENTS

The agent layer in one place. If you are new to this repo, start here.

RoadGuard's agent layer is a **3-worker orchestrator pipeline** plus a
lightweight Python supervisor. Each hazard flows through the same graph:

```
    sanitize -> Perception -> Research -> Decision ----+---> END
                                                       |
                                     (urgent / human_review)
                                                       |
                                                       v
                                                 Approval gate
                                                       |
                                                       v
                                                      END
```

- **Perception** turns the raw YOLO detection + image-quality signals into
  a structured `HazardProfile`. It is the only worker that reads image-side
  signals; downstream workers see the profile, not the pixels.
- **Research** calls `query_repair_standards` (read-only RAG) and returns
  a handoff of grounded procedures. It never writes anything. The retrieved
  passage first passes through a quarantine extractor (dual-LLM pattern) so
  any instruction-looking text in a repair standard cannot reach Decision.
- **Decision** is the one worker that writes. It picks exactly one of
  four Charter-mandated actions - `urgent`, `standard`, `needs_more_evidence`,
  `human_review` - and must ground every citation in the tool_result_ids
  that Research returned. Decision runs the LLM **panel** (3 models from
  3 different vendors with majority vote + severity tiebreak); Perception
  and Research run a single cheap model. Any parse error, schema violation,
  or ungrounded verdict falls back to a deterministic rule tree.
- **Supervisor** is a Python control loop (no LLM): it enforces 4 budget
  caps (iterations, hops per agent, cost, wall-clock) via `ContextVar`
  so each FastAPI request has its own tracker.
- **Approval gate** writes a `pending` row into the `approvals` table and
  emails the human supervisor whenever Decision returns `urgent` or
  `human_review`. The supervisor opens `/approvals/{id}` and clicks approve
  or deny before anything downstream can act (Charter §5-§6).

## Files in this folder

| File | What it is |
|---|---|
| `README.md`       | This page - the one-paragraph overview per agent. |
| `architecture.md` | Full RTL Hebrew spec: state, handoffs, grounding, guards, dual-LLM pattern, cost model, retry policy. **Authoritative.** |
| `contract.md`     | The 4-action / 4-class / 5-tool contract. Any change to these must land here first. |
| `diagrams.html`   | Interactive diagrams (open in a browser). Also exported inline into `architecture.md`. |

## Code map

| Component | Path |
|---|---|
| Nodes       | [`src/agent/multiagent/nodes/{perception,research,decision,supervisor}.py`](../src/agent/multiagent/nodes) |
| Graph       | [`src/agent/multiagent/graph.py`](../src/agent/multiagent/graph.py) (LangGraph + dependency-free fallback) |
| Prompts     | [`src/agent/multiagent/node_prompts.py`](../src/agent/multiagent/node_prompts.py) |
| Tools       | [`src/agent/multiagent/tools/`](../src/agent/multiagent/tools) (5 canonical tools) |
| Chat        | [`src/agent/multiagent/chat.py`](../src/agent/multiagent/chat.py) + providers.py (Groq, OpenAI-compat, Ollama) |
| Guards      | [`src/agent/multiagent/guards.py`](../src/agent/multiagent/guards.py) (budget guard via ContextVar) |
| Sanitize    | [`src/agent/multiagent/sanitize.py`](../src/agent/multiagent/sanitize.py) (location + OCR scrubs) |
| Asserts     | [`src/agent/multiagent/asserts.py`](../src/agent/multiagent/asserts.py) (schema + grounding checks) |
| Adapter     | [`src/agent/multiagent/adapter.py`](../src/agent/multiagent/adapter.py) (graph <-> AgentDecision for the API) |
| Approval    | [`src/db/schema.sql`](../src/db/schema.sql) (approvals table) + [`src/jobs/approval_email.py`](../src/jobs/approval_email.py) + `/approvals/{id}` endpoints in `src/api/main.py` |

## Tool contract

5 canonical tools, enforced by `Literal` types in `src/agent/tools.py` and
mirrored for the multiagent nodes under `src/agent/multiagent/tools/`:

| Tool | Reads from | Side effects |
|---|---|---|
| `check_hazard_severity`       | agent state                 | none (returns a priority) |
| `query_repair_standards`      | RAG (Chroma / stub)         | none |
| `request_additional_evidence` | hazard id + missing field   | writes to `requested_evidence` |
| `create_draft_work_order`     | hazard id + priority + cite | writes a draft row |
| `flag_for_urgent_review`      | hazard id + rationale       | emits approval row |

`lookup_similar_cases` is Perception-only (vector memory, no writes).

## LLM wiring

Env-driven, no code edits to switch provider:

```bash
# off (deterministic rule tree only, used in CI / eval)
ROADGUARD_USE_LIVE=0

# on (Groq single provider on Perception + Research)
ROADGUARD_USE_LIVE=1
AGENT_LLM_PROVIDER=groq
AGENT_LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=...

# Decision panel (3 vendors, majority vote with severity tiebreak)
AGENT_LLM_PANEL=groq:openai/gpt-oss-120b,openai_compat:meta-llama/Llama-3.3-70B-Instruct-Turbo,openai_compat:deepseek/deepseek-chat
OPENAI_COMPAT_BASE_URL=https://api.together.xyz/v1
OPENAI_COMPAT_API_KEY=...
```

The panel fires **only** on Decision (`use_panel=True` in the chat call);
Perception and Research keep the cheap single-provider path even when
`AGENT_LLM_PANEL` is set, so panel cost stays bounded.

## Charter compliance (the 4 metrics)

1. **80% recall on D40** - measured by `src/agent/evaluation/run_eval.py`.
2. **mAP@0.5 >= 0.55** - Shai owns this; the agent only consumes the YOLO
   payload.
3. **>= 85% correct decisions on 50 scenarios** -
   `tests/multiagent_eval_scenarios.json` (currently 50/50 with the stub).
4. **>= 90% grounded citations** - enforced per-run by
   `assert_decision_grounded`; ungrounded citations route to `human_review`.

## Running the agent locally

```bash
pytest -q                              # full suite, 260 tests, green
python -m agent.evaluation.run_eval    # offline eval harness
jupyter notebook notebooks/agent_walkthrough.ipynb
```

## Who to ping

| Area | Owner |
|---|---|
| Perception, Research, Decision, approval gate, prompts, API | Or |
| YOLO ONNX + training on 10K dataset | Shai |
| Ground-truth labels + eval set | Ovad |
| ChromaDB RAG corpus | Benny |
