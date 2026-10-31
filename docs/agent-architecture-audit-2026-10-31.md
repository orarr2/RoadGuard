# RoadGuard agent architecture audit - 2026-10-31

Mapped from code (roadguard_agent.py @ fc80780), not from docs. Baseline
numbers come from the 197-run Oracle replay log of 27.10.2026; the local
DB holds 0 runs so no fresh measurement was possible from here.

## 1. Graph

```
sanitize -> signal_extractor -> supervisor -> profile_builder -> supervisor
         -> rag_retriever -> supervisor -> decision -> approval_gate | await_evidence | END
```

Supervisor is visited 3x per run (hub). Every worker is a LangGraph node
that reads/writes one shared TypedDict state. No inter-agent messaging.

## 2. Per-node map (what actually runs on Oracle, inferred from the trace)

| Node | Type | LLM (Oracle) | Loop | Tools offered | Tools actually used in log | Fallback |
|---|---|---|---|---|---|---|
| sanitize | deterministic | none | - | lookup_dataset_image (hard-coded call) | catalog lookup every run | n/a |
| signal_extractor | LLM single call (ReAct only if AGENT_WORKER_REACT=1 and AGENT_SIGNAL_LLM set) | llama3.2:3b (deploy README) | 0-1 hop | extract_measurements, lookup_prior_reports | never (batch had no reporter text) | keyword regex |
| supervisor | deterministic router + budget; plus ONE advisory LLM call per hazard | qwen2.5:3b | 0-1 hop | check_budget_status, assess_input_complexity | not visible in log | silent {} |
| profile_builder | rule-based profile; LLM refine when AGENT_PROFILE_BUILDER_LLM=1 (ON on Oracle) | qwen2.5:3b | ReAct then single-call fallback | inspect_image_quality, lookup_similar_cases | 0 tool calls seen; 3 LLM calls on several runs | rule profile stands |
| rag_retriever | tool call (ChromaDB) + LLM relabel when AGENT_RAG_RETRIEVER_LLM=1 (ON) | granite3.3:2b | ReAct then single-call fallback | query_repair_standards, broaden_rag_query | 0 tool calls seen | tool coverage stands |
| decision | rule tree + LLM ReAct when AGENT_DECISION_REACT=1 (ON); panel path unused | qwen2.5:3b | <=2 hops | 8 tools (5 read-only) | check_hazard_severity, check_system_health, lookup_dataset_image | rule tree |
| approval_gate / await_evidence | deterministic | none | - | - | - | - |

Truly autonomous today: only Decision (chooses tools, inspects results).
signal_extractor / profile_builder / rag_retriever are a single LLM call
that may lower confidence or relabel; they cannot change retrieval or
ask for more input unless ReAct is on AND the model chooses a tool, which
the log shows it never did.

## 3. Budget and harness

- One BudgetTracker per run: MAX_ITERATIONS=8, MAX_TOOL_HOPS_PER_AGENT=3,
  COST_CAP_USD=0.10, WALL_CLOCK_CAP_SECONDS=240 on Oracle (30 default).
- `_llm_react_worker` is a generic harness for workers; Decision has its
  own copy (`_llm_decide_react`). Two implementations of the same loop.
- Per-call provider timeout AGENT_LLM_TIMEOUT_SECONDS (30 default);
  backoff 1/2/4 s; AGENT_LLM_FALLBACK chain; deadline from the budget.
- No cache, no parallelism between workers, no per-agent step budget
  other than the shared tool-hop cap.

## 4. Grounding

- Single validator `DecisionOutputModel._check_grounding`: cited ids must
  be in `tool_result_ids` OR start with `catalog:`. Both call paths
  (`_grounded_or_none`, rule-tree retry) go through it.
- The 82 "grounding failed" events in the log are from the 11:xx batch,
  BEFORE the `catalog:*` fold-in (code comment dates it 27.10). The
  18:xx-19:xx batch shows `grounding check: PASS` on the same images.
  Root cause is fixed in code; verify on Oracle after restart.
- Weakness that remains: grounding = "id exists in the retrieved set".
  Nothing checks that the passage supports the rationale.

## 5. RAG coverage is not evidence

`query_repair_standards` -> structured where-filter -> on miss, pure
semantic top-k -> `_augment_passage_for_coverage` STAMPS the queried
class/road onto every returned chunk -> `_coverage` therefore returns
`direct_match` for any semantic hit. No distance threshold anywhere
(`RagClient._query_where` returns whatever Chroma gives). The log's
`coverage=direct_match` on 100% of runs is an artefact, and the same 3
passage ids repeat for every D40/main query (`segment_doc_21_7941_*`).
One of them is the NTIS distribution boilerplate.

## 6. KNOWN_IMAGE

`_known_image_decision`: contradiction (catalog class != detector class
at conf>=0.30) -> human_review; severity severe -> fall through to rule
tree; otherwise `known_image priority=0` regardless of stored prediction.
`prediction=none` is never consulted. The backfill script (commit
2d47739) will populate predictions, but the rule still ignores them.
sanitize's fast path sends catalog hits straight to decision, skipping
signal/profile/rag - so for known images the "detector class" in the
contradiction check is the raw YOLO class, never the profile.

## 7. Policy enforcement

`load_policy()` is called only by `_policy_summary_for_prompt()`. The
rule tree hardcodes 0.4/0.6, D40/main/lane, conf>=0.6 as literals
(lines ~7906-7930). policy.yaml governs the prompt, not the code.

## 8. Latency baseline (from the 27.10 log, per visible run)

Typical D40/main run, e.g. hz_7f749ccbfa0c (134 s total):
supervisor 10 s + profile 52 s (3 LLM calls) + supervisor 5 s + rag 10 s
+ supervisor 5 s + decision 52 s. Across the log: Decision LLM 15-55 s,
Profile LLM 15-52 s, RAG LLM 9-42 s, supervisor advisory 4-10 s x3.
Known-image fast-path runs: 23-50 s (one supervisor visit + decision).
Review aggregates: median 55 s, P90 ~97 s, max 184 s, 197 DONE / 200.

Waste visible in the trace:
- profile_builder "3 call(s)": ReAct returned None -> single-call retry
  -> 2x cost for no tool use.
- supervisor advisory LLM runs 3x per hazard on some runs (dedup fix
  dated 27.10 exists; the 11:xx batch predates it).
- Decision ReAct hop spent on `check_system_health` (no decision value).
- llm_fallback 22x: `RuntimeError` only; error_message capture added in
  fc80780, not yet deployed.

## 9. Evaluation assets

tests/multiagent_eval_scenarios.json = 70 synthetic (rule-tree seeded)
+ tests/labeled_scenarios.json = 27 labelled = 97 scenarios (< 100).
All run with ROADGUARD_USE_LIVE=0, i.e. they test the rule tree, not
the LLM path. No per-agent metric, no latency assertion, no RAG
relevance set. Mock path: `_STUB_CORPUS` is still reachable when the
Chroma collection is empty and ROADGUARD_RAG_STRICT!=1.

## 10. Status of items from the 27.10 review

| Item | Status |
|---|---|
| Dates in trace | FIXED, RUNTIME NOT VERIFIED (5604c34, awaits restart) |
| Catalog predictions for 15,612 images | FIXED (script 2d47739), RUNTIME NOT VERIFIED (needs run on VM) |
| Explicit live-model line, sha256 vs phash, stored_prediction wording | FIXED, RUNTIME NOT VERIFIED (fc80780) |
| llm_fallback error detail | FIXED, RUNTIME NOT VERIFIED (fc80780) |
| 197/200 replay header | FIXED, RUNTIME NOT VERIFIED (fc80780) |
| /stats/agents per-agent counters | FIXED, RUNTIME NOT VERIFIED (fc80780) |
| 82 grounding failures (catalog cites) | FIXED in code before this audit; RUNTIME NOT VERIFIED on a fresh batch |
| Grounding = passage supports claim | NOT FIXED |
| RAG coverage stamped as direct_match, no threshold | NOT FIXED |
| KNOWN_IMAGE ignores stored prediction | NOT FIXED (blocked on backfill run) |
| policy.yaml not enforced in code | NOT FIXED |
| Duplicate ReAct harness, 2x LLM cost on worker ReAct miss | NOT FIXED |
| 97 scenarios, rule-tree only, no per-agent metrics | NOT FIXED |

## 11. Staged plan (proposed, awaiting approval)

Stage A - correctness, no new LLM calls:
1. RAG: real relevance - keep Chroma distance, apply a threshold, label
   coverage from distance + class keyword hit instead of stamping;
   drop boilerplate chunks (no D-code keyword, no repair verb).
2. Decision grounding: a cited passage must contain a keyword of the
   hazard class or a repair term; otherwise the cite is rejected and
   the model is told why (one retry, already budgeted).
3. KNOWN_IMAGE: short-circuit to priority=0 only when sha256 match AND
   stored prediction exists AND stored class agrees with live class;
   phash-only or prediction=none -> regular rule tree with the catalog
   as a cited supporting source.
4. Rule tree reads thresholds from load_policy() (same values as today).

Stage B - harness and latency, fewer LLM calls:
5. One ReAct harness: Decision uses `_llm_react_worker` too; on a
   ReAct parse miss do NOT re-run single-call (saves 1 call / ~25 s).
6. Supervisor advisory LLM: run once, before the first worker, never
   again (hard guard on state, not on trace scan).
7. Fast path for clear cases: detector conf >= 0.8, no reporter text,
   direct RAG hit -> skip profile_builder LLM refine (rule profile is
   identical on structured input per the 22.10 audit note in code).
8. Remove check_system_health from Decision's offered tools.

Stage C - evaluation:
9. Classify llm_fallback by error_message prefix (timeout / http 5xx /
   parse) in /stats/agents.
10. Add a RAG relevance set (25 labelled queries) and a per-agent
    eval that runs the live graph on the 27 labelled scenarios and
    reports action accuracy, tool usage, fallbacks, P50/P95.

Out of scope (per constraints): new agents, new tools beyond a
threshold check, prompt rewrites, any commit/push without approval.

## 12. Stage A results (2026-10-31, working tree, not committed)

| Item | Status | Evidence |
|---|---|---|
| RAG relevance: distance kept, keyword grading replaces stamping, boilerplate dropped | FIXED, RUNTIME NOT VERIFIED | tests: NTIS boilerplate dropped; chunk naming the class -> direct_match; repair-only chunk -> partial_match; live coverage/dist values need an Oracle batch |
| Cited passage must name the hazard class (urgent/standard) | FIXED, RUNTIME NOT VERIFIED | `_grounded_or_none` rejects unsupported LLM cites; rule tree prefers supported ids; trace prints `cite_support` and `=rag_unsupported` |
| KNOWN_IMAGE seal only on sha256 + stored score agreeing with live detector | FIXED, RUNTIME NOT VERIFIED | phash / no stored score / mismatch / live conf 0.00 -> regular tree with `catalog:` cite and `image_match_status`; sanitize no longer skips signal/profile/rag for those. Train/val seals can only occur after the backfill run |
| Rule tree and severity read policy.yaml | FIXED AND VERIFIED | identical behaviour with current values; override test flips the tree; malformed file -> defaults |
| /report and /hazards/history 500 on public hosts (pre-existing) | FIXED AND VERIFIED | `get_current_user(request)` received a raw `Depends` object; now gets `get_settings()` |

Suite: 148 passed, 0 failed, 0 deselected (before Stage A: 135 passed + 2
pre-existing failures). Policy change made explicit: an identical frame
whose stored score disagrees with the live detector is no longer sealed,
even below the 0.30 contradiction floor; the two tests that encoded the
old contract were rewritten to the new one.

## 13. Stage B and C results (2026-10-31, working tree, not committed)

| Item | Status | Evidence |
|---|---|---|
| B5 no second single-call LLM round after a ReAct miss (signal, profile, rag) | FIXED AND VERIFIED (unit) | tests assert chat() is not called after a ReAct miss; trace now says `llm_fallback_reason=react_reply_off_contract` instead of llm_used=True over rule values. Call-count drop on the VM not yet measured |
| B5 unify the two ReAct harnesses | NOT FIXED (deliberate) | the duplication itself showed no measured waste; refactoring Decision's loop risks the only autonomous agent for no latency gain |
| B6 supervisor advisory LLM once per hazard | FIXED before this audit (dedup dated 27.10) | 18:xx batch shows one [SUPERVIS] line per run vs 3 in the 11:xx batch |
| B7 clear-case fast path skips profile/rag LLM passes (policy.yaml latency.fast_path) | FIXED, RUNTIME NOT VERIFIED | policy-driven: conf>=0.8, usable image, no reporter text; agent_eval shows 5/12 scenarios take it; trace prints `llm_skipped_reason=clear_case_fast_path (...)`. Saves the 15-52 s profile pass and 9-42 s rag pass on those runs |
| B8 check_system_health no longer offered to Decision | FIXED AND VERIFIED | schema list test; function stays for /health |
| C9 /stats/agents classifies fallbacks (timeout, http_5xx, http_429, http_4xx, connection, parse, other), per agent, plus fast_path_skips | FIXED AND VERIFIED (unit) | |
| C10a RAG relevance eval (scripts/rag_relevance_eval.py, 16-pair class x road grid, target 95% class-hit) | FIXED; local run BLOCKED | local chromadb 0.5.23 cannot open the corpus store (written by chromadb 1.x; `KeyError '_type'`), so the local run fell to the stub corpus (11/16, a property of the stub). Must run on the VM |
| C10b per-agent eval (scripts/agent_eval.py) | FIXED AND VERIFIED (rule-tree mode) | 12 scenarios: accuracy 1.0, P50 0.19 s, P95 0.23 s; reports per-agent LLM calls/failures/fallbacks, Decision tool calls, floors, confusion. Live-mode numbers need the VM. Runs against a temp DB because ROADGUARD_DRY_RUN is not honoured by the agent tools |
| ROADGUARD_DRY_RUN claim in .env.example | NOT FIXED | the agent tools never read it; eval isolation is done by pointing ROADGUARD_DB_PATH at a temp file instead |
| Evaluation set >= 100 scenarios | NOT FIXED | 97 (70 synthetic + 27 labelled); needs labelled data from the data track |

Suite after Stage B/C: 154 passed, 0 failed.

VM runbook to turn RUNTIME NOT VERIFIED into verified:
1. git pull, rg-restart.
2. `python scripts/rag_relevance_eval.py --json /var/lib/roadguard/reports/rag_eval.json` (real corpus).
3. `python scripts/agent_eval.py --only labeled --json /var/lib/roadguard/reports/agent_eval.json` with the service env (USE_LIVE=1); compare accuracy, P50/P95, fallback causes with the 27.10 baseline (median 55 s, P90 97 s).
4. Re-run the 94-image batch; expect fewer KNOWN_IMAGE seals on phash hits, `image_match_status` on every catalogue hit, `cite_support` on every verdict, and no `prediction=none` seals.

## 14. Round 4 (2026-10-31 evening): the items left open, each decided by the owner

Operational state first. The Oracle VM was moved from its diverged checkout (4497a23 + 709 uncommitted lines, all of it already in GitHub except two functions Stage A replaced; kept in `stash@{0}` and branch `vm-pre-reset-20261031T155428Z`) to origin/main, and at the owner's explicit choice, with no backup, every case was wiped so the whole image set can be re-run from zero: 301 hazards, 143 approvals, 20 evidence requests, 59 work orders, decision_cache, the LangGraph checkpoint file, 376 uploaded photos and the archived digests. Catalogue (17,099) and users (4) stayed. /flow now draws only a run that is in progress (status processing with a trace entry in the last 5 minutes); verified in a browser against a seeded DB (finished run + stale processing row -> idle; live run -> active; finished -> idle). The RAG relevance grid ran on the real corpus: 16/16 direct_match, 2-3 class hits per query, 0.16 s per query, so C10a is closed.

| Item | Status | Evidence |
|---|---|---|
| ROADGUARD_DRY_RUN claim | FIXED AND VERIFIED | the claim was removed from .env.example, docs/agents-architecture.md and docs/agents-contract.md (owner's choice: no dry-run mode comes back; evaluations isolate through a temporary ROADGUARD_DB_PATH / ROADGUARD_CHECKPOINT_PATH) |
| Evaluation set >= 100 | FIXED AND VERIFIED | 73 synthetic + 27 labelled = 100. The three new synthetic cases pin policy edges and the new code: D40/main/partial at exactly 0.60 -> urgent, at 0.59 -> needs_more_evidence, and a confident clean D10 -> standard through the fast path. Labels come from the rule tree, like the other 70 |
| Queues: interrupted runs | FIXED AND VERIFIED (unit) | `recover_interrupted_runs` at startup: rows still queued/processing (no background task survives a restart) become human_review with an approval row and a `graph_interrupted` trace event. Test: 2 stuck rows recovered, open row untouched, second call recovers 0 |
| Queues: approval SLA and inbox order | FIXED AND VERIFIED (unit) | `list_pending_approvals` orders by priority DESC then age; policy.yaml `queues.approval_sla_hours` (24); /status prints `approvals pending N (overdue M, sla 24h)`; /report items carry `age_hours` and `overdue`; the supervisor inbox shows an overdue badge |
| Queues: expired evidence | FIXED AND VERIFIED (unit) | policy.yaml `queues.evidence_ttl_hours` (72) replaces the 7-day checkpoint cutoff for evidence expiry; an expired request's hazard is routed to human_review with an approval row (`route_expired_evidence`, idempotent) instead of sitting in `evidence_expired` |
| Semantic grounding (owner first chose local embeddings) | NOT FIXED, by evidence | calibration on the live corpus (2,433 passages, 227 naming exactly one class): cosine between rule-tree claims and passages does not separate classes (threshold 0.35: 74% of correct passages pass, 44% of wrong-class passages pass too; plain claims without class terms: median 0.12); nearest-class-description assignment is right for 39% of labelled passages (D00 7/61, D40 39/97). MiniLM sees "road repair" everywhere. The owner then chose to keep the keyword rule; the calibration is recorded here and nothing shipped |
| Two ReAct loops | NOT FIXED (owner's decision: keep two, document) | Decision's `_llm_decide_react` (2 hops, decision-specific grounding retry) and the workers' `_llm_react_worker` (generic, now with a time cap) stay separate; no measured waste from the duplication, and the Decision loop is the one agent whose behaviour is benchmarked |
| Worker autonomy (all three workers, behind AGENT_WORKER_AUTONOMY) | FIXED AND VERIFIED (unit); RUNTIME NOT VERIFIED | information gap: signal gets text + location only (no detector / form), profile gets no rule hint, research gets no passages; each worker must call its tools (`extract_measurements`, `inspect_image_quality`, `query_repair_standards`) and declares `tools_used`; the code verifies declared vs. actually-run tools, recomputes or drops numbers the regex cannot confirm, rejects a profile judged without the quality tool, assembles the research handoff from the passages the agent itself retrieved and recomputes coverage, flags cited ids the agent never saw; hops default to 3, `latency.worker_time_cap_s` (45 s) bounds each worker; every worker's trace entry carries `autonomy: tools=... hops=... violations=...` and /stats/agents aggregates runs / tool calls / violations / clean runs per worker. Also fixed on the way: the ReAct reply of the intake agent was merged raw into the intake instead of being normalised like the single-call reply (contradiction detection never saw `mentioned.*` on the react path) |
| Shared memory: agent_notes | FIXED AND VERIFIED (unit); RUNTIME NOT VERIFIED | new table written by POST /approvals/{id}/decide (which agent verdict a human confirmed or overturned, for which class / road / lane, with the supervisor's note) and read through `recall_agent_notes` by profile_builder, rag_retriever and Decision (read-only tool, offered only under the flag). Hints, never rules |

Suite after round 4: 174 passed, 0 failed.

Still open after this round: the autonomy and queue behaviour at runtime (needs the re-run batch with AGENT_WORKER_AUTONOMY=1 on the VM); `agent_eval.py` in live mode; the catalogue backfill ingest once `predictions_full.csv` is complete.

## 15. Round 5 (2026-10-31 night): backfill, re-run, what the runtime showed, webapp sign-in, documents

Backfill and ingest: `scripts/predict_catalog_with_shai_model.py` ran on the VM in its own venv (torch CPU, ultralytics 8.4.184, onnxruntime, headless OpenCV) over all 17,099 catalogue images: 17,096 detections, 3 without; `rg-catalog` ingested the CSV, `dataset_images` now carries a stored prediction for 17,096 rows (D20 6,965, D00 4,490, D40 3,888, D10 1,753). The 94-image batch (`/tmp/batch_runner.py`, through the public API as `u_batch_internal`) was then started as a systemd unit.

What the first 20 runs showed, with AGENT_WORKER_AUTONOMY=1 live:

| Observation | Evidence | Consequence |
|---|---|---|
| Every second ReAct round against qwen2.5:3b failed with HTTP 400 (`Value looks like object, but can't find closing '}' symbol`), for Decision and for profile_builder | `/stats/agents`: decision `llm_based 0`, `http_4xx 6`; profile_builder `http_4xx 12`; a direct probe of Ollama 0.40.9 on the VM: assistant tool-call `arguments` as a JSON string -> 400, as an object -> 200 | The adapter normalises outbound arguments to objects (`_ollama_messages`, unit test `tests/test_ollama_adapter.py`). Until the fix is deployed, every tool-using run falls back to the rule tree: correct verdicts, no model verdicts |
| rag_retriever (granite3.3:2b) never called a tool in 17 autonomous runs | `autonomy.rag_retriever.tool_calls 0`, violations `required_not_called:query_repair_standards` on every run | The deterministic query ran instead (by design), so retrieval quality is unchanged; research autonomy needs a model that emits tool calls (qwen2.5:3b or llama3.2:3b) |
| profile_builder called `lookup_similar_cases` / `recall_agent_notes` but not the required `inspect_image_quality` | violations `required_not_called:inspect_image_quality` on every run that reached a tool | The rule profile stood (by design); the prompt asks for the quality tool first, the model prefers the lookups |
| Clear-case fast path and KNOWN_IMAGE seal | `fast_path_skips` 2 per worker; one `known_image` verdict on a catalogue image | Both paths fire at runtime |
| Queue recovery | `recovered_interrupted_runs` logged on each restart | The two 24.10/26.10 rows would have been routed instead of sitting in `processing` |

Webapp sign-in (reproduced in the built-in browser against a local copy of the webapp and a local API, then fixed):

| Symptom reported | Root cause | Fix |
|---|---|---|
| "My reports" never loaded on the live site | `loadMyReports` fetched `/hazards/history` with `credentials: "include"`; the API's CORS policy has `allow_credentials=False`, so the browser blocked the response | The table now goes through `fetchJson` with the bearer token and is skipped when signed out (no 401 that pops the modal on page load) |
| Google button did nothing | `CONFIG.GOOGLE_CLIENT_ID` was empty in `webapp/assets/config.js` while the server reported Google enabled; the widget never mounted and the click called an uninitialised Google API | The public client id (the same one the server verifies) is in config.js; `mountGoogleSignIn` reports whether it mounted and the button explains when it did not |
| No register button | `/auth/config` says `signup_enabled: false` because `ROADGUARD_SIGNUP_ENABLED` is unset on the VM; the UI hides the flip link by design | Server toggle, enabled at this deploy (cap 10 users, password rules enforced) |
| Logout "did nothing" | a stale token made `/auth/logout` return 401, which `fetchJson` treated as a rejected session and reopened the sign-in modal | `/auth/logout`, `/auth/login/password` and `/auth/register` are excluded from the 401 handler; the token is always cleared locally |

Verified locally in the browser: fresh visitor -> no modal on load; Sign in -> modal with Google widget rendered and the Sign up link; Sign up -> signed in, header shows the username and Log out, "My reports" table loads; Log out -> header back to Sign in, token cleared, no modal; Upload while signed out -> modal (gate); wrong password -> "Wrong username or password."; correct password -> signed in. The live API answered the same flows over curl (CORS preflight for the Pages origin OK, wrong password 401, login -> /auth/me 200, logout 200).

Documents: README, CHARTER, standalone/README, docs/agents-contract.md and docs/agents-architecture.md were cross-checked against the code (paths, env names, scripts, counts) and corrected: the corpus embedder (all-MiniLM-L6-v2, 384 dimensions, verified on the collection; not nomic-embed-text), the scenario count (100), the test count (174), the removed `scripts/benchmark.py` and `PYTHONPATH=src` commands, the implemented `/hazards/history`, the retired `check_system_health`, the Decision tool list, the autonomy flag, missing files (roadmap, diagrams, walkthrough notebook) pointed at what exists. The standalone notebook's markdown (9 passages), the team deck (8 passages: 7b vs 3b, duplicated tool line, retired tool, embedder, autonomy update) and the hand-off sheet (3 cells: catalogue path, embedder, tool list) were corrected in place.
