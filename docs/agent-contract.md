# Agent Contract

This document is the authoritative specification for the RoadGuard
decision agent. It fixes the tools, the decision loop, the policy
fallbacks, and the evaluation protocol. Any change here is a breaking
change for the API, the webapp, and the evaluation harness.

## 1. Scope and non-goals

The agent recommends one of four actions per hazard. It never:

- closes a road,
- dispatches a contractor,
- notifies a public map or municipal queue,
- determines engineering severity (depth, structural impact).

External actions are strictly mocked. `create_draft_work_order` writes
to a local SQLite table only. `flag_for_urgent_review` writes a row and
sends an internal email summary; it does not contact anyone else.

## 2. Inputs

Each invocation receives a JSON payload with:

| Field | Type | Required | Notes |
|---|---|---|---|
| `hazard_id` | string | yes | server-assigned UUID |
| `class` | enum[D00, D10, D20, D40] | yes | top-1 class from the YOLO run |
| `confidence` | number [0, 1] | yes | top-1 confidence from the YOLO run |
| `bboxes` | list of [x1, y1, x2, y2, class, conf] | yes | all detections above the client-side threshold |
| `road_type` | enum[main, secondary, bike_path, parking] | yes | user-provided |
| `lane_blocked` | enum[none, partial, full] | yes | user-provided |
| `location` | string | no | free-form, optional |
| `image_object_key` | string | yes | Supabase object key; the agent never dereferences the image |
| `reported_at` | ISO 8601 | yes | server timestamp |

Invariants:

- Inputs are validated with pydantic before the loop starts. A validation
  failure returns `human_review` with the raw rule-based priority.
- The agent never fetches the raw image. All visual signal is in the
  bbox list and the class/confidence pair.

## 3. Tools

Five tools are exposed to the LLM via function calling. Each tool
declares a JSON schema that the Groq function-calling API consumes
verbatim; the fallback path (section 7) does not use the tool loop.

### 3.1 `check_hazard_severity`

Compute an operational priority score from transparent rules.

- Input: `{ class, confidence, road_type, lane_blocked }`.
- Output: `{ priority: int in [0, 100], band: enum[low, medium, high, critical], rationale: string }`.
- Side effect: none (pure function).

Rules (summary):

- Base score by class: D40 = 70, D20 = 55, D00 = 40, D10 = 40.
- Multiply by confidence (soft cap at 0.95).
- `+15` if `road_type == main`, `+5` if `secondary`, `0` otherwise.
- `+25` if `lane_blocked == full`, `+10` if `partial`.
- Clamp to `[0, 100]`.
- Band: `>= 80` critical, `>= 60` high, `>= 40` medium, else low.

### 3.2 `query_repair_standards`

Retrieve procedure passages from the local RAG corpus (ChromaDB).

- Input: `{ hazard_class, road_type, k?: int = 3 }`.
- Output: list of `{ id, text, issuing_body, publication_date, hazard_type, triggering_conditions, source_url, is_binding_in_il }`.
- Side effect: none; a local vector search only.

Empty results trigger the policy fallback (section 7).

### 3.3 `request_additional_evidence`

Request a specific missing piece of information from the user.

- Input: `{ missing_field: enum[second_angle, location, closer_shot, lighting], prompt_hint: string }`.
- Output: `{ prompt_text: string }`.
- Side effect: stored on the hazard record for the webapp to show.

The webapp surfaces `prompt_text` with a resubmit button. No new photo
is routed through the agent until the user responds.

### 3.4 `create_draft_work_order`

Write a draft work order to the local SQLite `work_orders` table.

- Input: `{ hazard_id, priority: int, suggested_action: string }`.
- Output: `{ work_order_id, status: "draft" }`.
- Side effect: inserts a `work_orders` row and updates
  `hazards.work_order_id`.

The row is a draft and nothing dispatches it anywhere.

### 3.5 `flag_for_urgent_review`

Mark a hazard for a human supervisor.

- Input: `{ hazard_id, reason: string }`.
- Output: `{ ok: true }`.
- Side effect: sets `hazards.status = "human_review"`, writes a row to
  the daily email queue. Does not contact a contractor, does not alter
  a public map.

## 4. The four final actions

The agent must terminate with exactly one of:

| Action | Trigger example |
|---|---|
| `urgent` | A clear pothole blocking at least one lane on a main road, confidence >= 0.6 |
| `standard` | Routine damage, no blockage |
| `needs_more_evidence` | Confidence between 0.4 and 0.6, or missing metadata |
| `human_review` | Any policy fallback (section 7) |

Every verdict must include:

- `action` (one of the four above),
- `rationale` (a single sentence),
- `priority` (int from `check_hazard_severity`),
- `cited_sources` (list of `source_url` strings from
  `query_repair_standards`; empty only for `human_review`).

## 5. Decision loop

```python
def agent_decide(hazard_input: HazardInput) -> AgentDecision:
    context = build_context(hazard_input)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context)},
    ]
    for hop in range(MAX_TOOL_HOPS):
        resp = call_groq_or_fallback(messages, tools=TOOL_SCHEMAS)
        if resp.finish_reason == "tool_calls":
            for call in resp.tool_calls:
                result = execute_tool(call)
                messages.append({"role": "tool", "content": result})
            continue
        return parse_decision(resp.message.content)
    return human_review("tool-hop budget exceeded")
```

- `MAX_TOOL_HOPS = 5`.
- Every tool call and every LLM response are appended to
  `hazards.agent_trace_json` for audit.
- `parse_decision` validates the final JSON against the schema in
  section 4. On validation failure the agent falls to `human_review`.

## 6. Four actions, decision table

The LLM is instructed to prefer the first row that matches. The agent
code enforces the ordering as a sanity post-check.

| Condition | Action |
|---|---|
| Any policy fallback trigger (section 7) | `human_review` |
| `class == D40` and `lane_blocked in {partial, full}` and `confidence >= 0.6` | `urgent` |
| `band == critical` and `confidence >= 0.6` | `urgent` |
| `confidence < 0.6` and `class in {D00, D10, D20, D40}` | `needs_more_evidence` |
| `band in {low, medium}` and procedure retrieved | `standard` |
| otherwise | `human_review` |

The LLM may deviate; the sanity post-check will not. A deviation that
produces an action inconsistent with the table is logged for later
analysis.

## 7. Policy fallbacks (charter-mandated)

Any one of the following forces `human_review`:

- `confidence < 0.4` for the top-1 detection.
- `query_repair_standards` returns zero passages.
- Any tool raises or exceeds its timeout (3 seconds per tool).
- The agent proposes an action not in the enum of section 4.
- The structured verdict fails schema validation.
- Groq is unavailable and the local Llama fallback also fails or
  returns an unparseable response.

When Groq is unavailable but the local Llama fallback succeeds, the agent
enters a **simplified mode**:

- The tool loop is skipped.
- `check_hazard_severity` is called inline from Python (pure rules).
- `query_repair_standards` is called inline for the top-k passages.
- Llama is asked for a single structured completion:
  `{ "action": "...", "rationale": "...", "cited_sources": [...] }`.

Simplified-mode output must still validate against the schema in
section 4. Simplified-mode decisions are tagged in
`agent_trace_json` with `provider: "local_llama"`.

## 8. Evaluation protocol

Scenarios live in `src/agent/evaluation/scenarios.json`. Each scenario
is a dict:

```json
{
  "id": "s001",
  "band": "easy",
  "hazard_input": { ... same shape as section 2 ... },
  "expected_action": "urgent",
  "allowed_actions": ["urgent"],
  "notes": "clear pothole, full-lane block, main road, high confidence"
}
```

`allowed_actions` lets a scenario accept any of a short list where
multiple answers are defensible. `expected_action` is the preferred
choice used in the primary metric.

The harness in `src/agent/evaluation/run_eval.py` reports:

- Correct-action rate: `expected_action in chosen` divided by total.
- Groundedness rate: fraction of non-`human_review` verdicts whose
  `cited_sources` is non-empty and whose URLs appear in the local RAG
  corpus.
- Action distribution across the set.
- Per-scenario latency (seconds).

Thresholds for a passing run (per charter):

- Correct-action rate >= 85% across the full 50-scenario set.
- Groundedness rate >= 90% across the full 50-scenario set.

Any scenario whose chosen action is `human_review` is excluded from the
groundedness denominator but counted in correctness.

## 9. Observability

Every decision writes:

- `agent_trace_json`: full ordered list of `{hop, llm_response, tool_calls, tool_results}` plus the parsed verdict.
- `agent_rationale`: the one-sentence rationale.
- `agent_decision`: one of `urgent`, `standard`, `needs_more_evidence`, `human_review`.
- `priority`: int.
- `cited_sources`: list of URLs.

`/hazards/{id}` returns the full trace for debugging and for the webapp
to render the "how did it decide" panel.
