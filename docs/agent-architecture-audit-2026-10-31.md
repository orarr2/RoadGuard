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

Deploy and runtime check of the fixes: the Ollama adapter fix went live at 17:54 UTC; the first full run after it (hz_54887ff4ef4a) completed two Decision tool hops (check_hazard_severity, lookup_dataset_image) and a final reply without a single HTTP 400, but the three Decision calls took 55 s, 105 s and 29 s on the ARM CPU (the second-round prompt carried 3,780 tokens), the run finished at 232 s of the 240 s wall clock, and the final reply was not accepted (rule tree stood). ReAct latency, not failures, is now the limiting factor for Decision on this hardware. The restart also cost the first batch 65 images (the runner has no retry; it recorded `URLError` for every image it submitted while the service was down), so those 65 were re-run as a second unit afterwards. GitHub Pages turned out to build from the main branch root, where a mirror of `webapp/` lives (`index.html`, `assets/`, `tunnel-url.json`); the sign-in fixes reached the live site only after the mirror was synced, and the README now says so. On the live site the Google button opens Google's account picker (no console errors); only `ROADGUARD_ALLOWED_EMAILS` may complete it; signup is enabled on the server (`/auth/config` reports it, cap 10 users).

Documents: README, CHARTER, standalone/README, docs/agents-contract.md and docs/agents-architecture.md were cross-checked against the code (paths, env names, scripts, counts) and corrected: the corpus embedder (all-MiniLM-L6-v2, 384 dimensions, verified on the collection; not nomic-embed-text), the scenario count (100), the test count (174), the removed `scripts/benchmark.py` and `PYTHONPATH=src` commands, the implemented `/hazards/history`, the retired `check_system_health`, the Decision tool list, the autonomy flag, missing files (roadmap, diagrams, walkthrough notebook) pointed at what exists. The standalone notebook's markdown (9 passages), the team deck (8 passages: 7b vs 3b, duplicated tool line, retired tool, embedder, autonomy update) and the hand-off sheet (3 cells: catalogue path, embedder, tool list) were corrected in place.

## 16. Round 6 (2026-10-31 night): the 48-run log review, each finding traced to its cause

The owner reviewed the first 48 autonomous runs from the live trace. Every finding below was checked against the database and the service journal before anything was changed.

| Finding | Root cause | Fix | Test |
|---|---|---|---|
| pic_id=13786 sealed as `known_image` although the catalogue labels say D20 and both detector readings (stored D00 0.20, live D00) say otherwise | `_image_match_status` compared the stored score with the live detector only; two model readings agreeing with each other passed as agreement with the ground truth | New reason `catalog_labels_disagree`: `_known_image_decision` routes it to `human_review` (floor `known_image_catalog_label_vs_prediction`, priority 50, catalogue cited). New policy key `known_image.min_stored_confidence: 0.4` (reason `stored_confidence_below_floor` leaves the frame to the rule tree with the catalogue as context) | `tests/test_known_image.py`, 2 new |
| LLM calls of 650 s (signal_extractor, hz_0e793b78c107) and 401 s (profile_builder, hz_cc4af5d26e4d) in runs capped at 240 s | Not the timeout logic. The service journal is silent from 18:26:48 to 18:32:53 UTC, including the `/report` polls that arrived every two seconds before, and the kernel log shows the OOM killer at 18:32:47 (`llama-server`, 11.3 GB anonymous RSS, killed). The VM has 24 GB and no swap; the shared Ollama keeps up to five models resident for 24 h (`OLLAMA_MAX_LOADED_MODELS=5`, `OLLAMA_KEEP_ALIVE=24h`, `OLLAMA_NUM_PARALLEL=4`) and held four (about 20 GB). The process made no progress for six minutes, so no timer could fire. Two real gaps surfaced on the way: the wall clock started at the Supervisor's first check, one node after the signal extractor, so intake calls had no deadline; and a worker's 45 s cap was checked only between rounds, so a call could start at second 44 and run the full provider timeout | `sanitize_node` starts the clock and publishes the deadline; `_llm_react_worker` publishes min(run deadline, worker cap) for the length of its loop, so `chat()` cuts every call's timeout to it and restores the previous value afterwards; no ReAct round (worker or Decision) starts with under 5 s of headroom (`_MIN_HOP_HEADROOM_SECONDS`); a Decision stopped by the deadline reports `fallback_reason=run_deadline`. Ops at this deploy: an 8 GB swap file, and rag_retriever on qwen2.5:3b so one model fewer is resident | `tests/test_run_deadline.py`, 4 new; the first runs the whole graph live against a fake provider and checks the timeout every node's call received (workers: at most the 45 s cap; Decision: what is left of the 240 s) |
| profile_builder never called `inspect_image_quality`; the rule profile stood and the agent's reading was dropped | The required-tool rule voided the whole reply | The code runs the skipped tool itself and feeds it into the contract: a `low` quality verdict caps the agent's confidence (`confidence_capped_by_quality_tool`); the violation stays on record; `auto_ran` is printed on the trace line and counted in `/stats/agents` | `tests/test_worker_autonomy.py`, 1 updated + 1 new |
| rag_retriever autonomy line read `violations=required_not_called:query_repair_standards,no_passages_retrieved` while Decision received 3 passages | The deterministic query ran, but the record said nothing was retrieved | `agent_retrieved: false`, `fallback=deterministic_query (N passages)`, `auto_ran=query_repair_standards`; `source=agent_retrieval` when the agent did retrieve | `tests/test_worker_autonomy.py` |
| granite3.3:2b never emits a tool call | Model capability | `AGENT_RESEARCH_LLM=ollama:qwen2.5:3b` at this deploy | runtime (next runs) |
| "profile None" and doubled handoff lines in the pasted log | Not in the database: 0 traces with a None profile and 0 doubled handoffs in 91 runs; an artifact of the pasted text | none | - |

Signal extractor stays on llama3.2:3b (owner's choice); the swap file is a safety net against the thrash, not a fix for the model mix.

Batch of 31.10 (94 images, after the backfill wrote predictions for all 17,099 catalogue rows): `known_image` 41, `standard` 26, `human_review` 17, `needs_more_evidence` 10. Submit-to-verdict time of the successful runs as measured by the batch runner: p50 81 s, p90 178 s, max 249 s. The 65 images that hit the service restart window were re-run and are in the counts. The wall-clock guard fired in 4 of 83 runs in the last 24 h (`budget_state` lines in the journal: median 76 s, p90 166 s).

Suite after this round: 183 tests pass (176 before, 7 new).

Two webapp findings reported by the owner the same night, both verified live in his Chrome after the deploy:

| Finding | Root cause | Fix | Test |
|---|---|---|---|
| The operator links vanished from the admin's profile; the upload count showed 1 | A registered user's token subject is `user:<id>`, and the submit path upserted that subject as a second users row (no admin flag) on the first upload; `/auth/me` then resolved to the shadow row. The batch user had the same shadow row since 27.10 | `_user_id_of` returns the bare id for `user:<id>` subjects; `Database.merge_prefixed_users` folds existing shadow rows into the real rows at startup (`merged_prefixed_user_rows count=2` on the VM: 97 hazards re-pointed, two rows removed) | `tests/test_user_identity.py` |
| "Continue with Google" did nothing | The styled stand-in forwarded a synthetic click to Google's hidden button, which the widget ignores | Google's own button renders in the modal (text `continue_with`, width 400, English locale) and replaces the stand-in once rendered; a Google identity with the e-mail of a registered account signs in as that account and the Google subject is linked to the row, so the admin keeps the operator links whichever button he used | `tests/test_user_identity.py` (Google sign-in lands on the registered admin) |

Deploy of 20:29 UTC (commit 722fd5d): database backed up to `backups/roadguard.db.pre-round6-*`, `AGENT_RESEARCH_LLM=ollama:qwen2.5:3b`, granite unloaded, 8 GB swap file active and in fstab, service up in 4 s; memory after the restart 13 GB used of 24 GB. Suite: 186 tests pass.

## 17. Round 7 (31.10.2026, late night): live audit of the re-run from zero

The owner had all 97 cases deleted (backup `backups/roadguard.db.pre-wipe2-*`) and the 94 images run again from zero while he watched `/trace`. After six cases he asked for a live audit of what the trace showed. Findings, each checked against the database, and what was done:

| Finding | Evidence | Cause | Fix |
|---|---|---|---|
| 3 of the first 5 verdicts were `human_review` by `known_image_catalog_detector_contradiction`, all on **phash** matches (a similar photo, distance up to 15 of 64 bits, not the same file) | rationale "match by phash"; in the previous batch 9 of 19 human_review verdicts had the same floor | `_known_image_decision` compared the catalogue labels with the detector whatever the match kind | The contradiction floor requires `match == sha256`; a phash hit stays context for the rule tree; the floor's trace entry now carries `image_match_status` (match kind, distance) |
| The supervisor's note said "main lane is blocked" on open roads: 3 of the first 4 runs, 54 of 97 in the previous batch | `supervisor_plan.note` vs `lane_blocked=none` | The note is free text from the supervisor model (granite3.3:2b, `AGENT_SUPERVISOR_LLM`); routing never depends on it | `_verify_supervisor_note` checks the note against the fields it was given (lane, reporter text, class) and replaces a contradicting note with a deterministic summary; `note_check` on the trace entry and in the trace line. Model moved to qwen2.5:3b (owner's choice; granite leaves memory) |
| `/report` showed `token_usage 0/0` and `llm_calls` with `model: null` while the trace showed 4 calls and 7,118 / 426 tokens | report of hz_4e2a969b0052 | The report read `tokens_in` / `model` off the node entries, where they no longer live | `_report_trace_summary`: tokens and models from the `llm_call` entries chat() emits, attributed to the node whose entry follows; legacy fallback kept |
| "[PROFILE] class=None severity=None ... handoff" once per hop, "[RAG] 0 passages coverage=None", "[DECISION] action=? priority=None" | the live page | The workers' `react_step` entries fell through to the speaker's handoff template; Decision's relaxed catalogue lookup is a bare `tool_call` entry without content | `_entry_lines` renders any `react_step` as a step (shared `_react_step_lines`) and a bare tool call as one note line. This corrects section 16: the lines were in the stored trace, not an artifact of the pasted log |
| Verdict, rationale, grounded sources, approval rows and the digest list agree with the trace and the database for every finished case; `needs_more_evidence` lands in `waiting_for_evidence` (72 h expiry); the first run after the restart hit the 240 s cap on cold model loads and ended cleanly | checked case by case | - | none |

The run was stopped after 9 cases at the owner's request, the cases deleted again, and the 94 images started from zero on this build.

Tests: `tests/test_round7_audit.py` (5) and one more in `tests/test_known_image.py`.

## 18. Round 8 (1.11.2026, after midnight): the three improvements the live audit named

| Improvement | Before | After | Test |
|---|---|---|---|
| Worker ReAct steps in the stored trace | The steps reached the live page through the sink, but the node's trace update held only the handoff entry, so every snapshot dropped them; a finished case showed Decision's steps and none of the workers' | `record_worker_step` buffers each step the worker loop emits (per thread, next to the LLM log) and `drain_worker_steps` puts them into the node's `trace` update before the handoff entry (signal_extractor, profile_builder, rag_retriever, supervisor) | `tests/test_round7_audit.py` (buffer; rag node trace order) |
| RAG passages for D40 cut mid-word and far (0.72-0.77) | One templated query ("road maintenance repair standards for pothole on arterial main road: safety requirements, response procedures, repair window"); chunks split on characters | A second, plainer query per class (`_HAZARD_REPAIR_QUERY`, e.g. "pothole repair: patching procedure, materials and response time"; measured on the VM corpus: D40 0.42-0.45 instead of 0.72-0.77, D20 0.37-0.56 instead of 0.43-0.66); both queries run, hits merge, `_rank_semantic_hits` orders by class term, repair term, distance; `_clean_passage_text` trims a cut start or end to the nearest sentence and collapses whitespace; the client returns up to 12 hits per query instead of 5 | `tests/test_round7_audit.py` (clean, rank, two-query merge) |
| `needs_more_evidence` in a batch waits 72 h | The batch runner polled until the status left queued/processing and recorded `waiting_for_evidence` | `scripts/batch_runner.py` (now versioned) answers the agent's one evidence request the way a reporter would, with the same upload, through `/evidence_requests/{id}/fulfill`; the paused graph resumes and the case ends in a verdict (`evidence_answered` in the results row) | runtime (next run) |

Measured query distances come from `scratchpad/rag_query_experiment.py` run against the live corpus (2,433 passages); D10 and D00 stay weaker (0.63-0.75) because the manuals name those cracks less often.

Overnight run on this build (1.11.2026, 94 images, 3.74 h, unit result success, no OOM, swap unused): known_image 29, standard 33, human_review 32, needs_more_evidence 0 (all 10 evidence requests were answered by the runner and resumed to a verdict). Catalogue seals fell from 41 to 29 because the label check and the 0.4 stored-confidence floor now apply (3 label disputes went to a human, the rest to the rule tree). Contradiction floors: 1 (sha256) instead of 9. Supervisor notes: 93 ok, 1 replaced. RAG passage distance median 0.48. Worker ReAct steps stored: 188. Per image: p50 108 s, p90 241 s, max 484 s (an evidence round). 17 of 104 decision passes reached the 240 s cap and were decided by the rule tree (Decision's model decided 14); with every agent now using its hops the pipeline needs about 280 s on this CPU, so the owner raised `WALL_CLOCK_CAP_SECONDS` to 360 on the VM (environment only).


## 19. Round 9 (1.11.2026, afternoon): run r6 from zero and the Decision timeout

Run r6 (same build 09caff3, `WALL_CLOCK_CAP_SECONDS=360`, every case, image, checkpoint and agent note wiped first so no agent memory carried over): 94/94 verdicts in 4 h 06 min (09:28-13:35 UTC), no errors, no OOM, swap unused.

| | batch of 31.10 (old build) | r5 (night, cap 240) | r6 (cap 360) |
|---|---|---|---|
| known_image / standard / human_review / needs_more_evidence | 41 / 26 / 17 / 10 | 29 / 33 / 32 / 0 (10 answered) | 28 / 37 / 29 / 0 (10 answered) |
| submit-to-verdict p50 / p90 / max (s) | 81 / 178 / 249 | 108 / 241 / 484 | 114 / 358 / 568 |
| p50 without an evidence round (s) | 81 | 84 | 86 |
| passes past 240 s | - | 17 (cut at the cap) | 17 (4 LLM-decided, 9 rule-based before 360, 4 at the cap) |
| LLM-decided passes | 17 / 121 | 14 / 104 | 13 / 104 |
| supervisor notes ok | - | 93 / 94 | 94 / 94 |

Of the 50 catalogue images 28 sealed; the 22 that did not: 12 `stored_confidence_below_floor` (policy 0.4), 6 `severity_blocks_shortcut`, 4 label disputes (3 `label_vs_prediction`, 1 contradiction) to human_review. `/report` shows tokens and models for all 94 cases (checked with the builder on the stored traces). RAG distances identical to r5 (median 0.484). 187 worker ReAct steps stored.

**Why the run got longer than the batch of 31.10, and what did not change.** Per call the model is as fast as on 31.10 (service journal, qwen2.5:3b: latency p50 20.3 s then, 17.8 s now; p90 56.4 then, 54.7 now; prompts of 2,000+ tokens that succeeded took p50 48.7 s then, 54.1 s now; the largest prompts 4,034 then, 4,061 now). Four deliberate changes lengthened the run: research and supervisor moved from granite3.3:2b (8.7 s per call) to qwen2.5:3b (15-20 s) to stop the hallucinated lane-blocked notes (+15-20 s per case); the stricter KNOWN_IMAGE seal (sha256, label agreement, 0.4 floor) sends 14 more cases through the full pipeline (about 100 s instead of 28 s); the batch runner answers evidence requests, so 10 cases run the graph twice (the 568 s maximum); the cap moved from 240 to 360 s, so the slow tail runs two minutes longer before the rule tree decides.

**The pre-existing cause of the slow tail (31.10: 30 calls at the 60 s timeout; r6: 32).** The Ollama server logged the final Decision call of case hz_6088b2b03457 at 4,044 prompt tokens on a 4,096 context (`truncated = 1`); the CPU evaluates 30-45 tokens per second, so that call needs 100-130 s, while `AGENT_LLM_TIMEOUT_SECONDS` was 60. The call failed with ReadTimeout and the backoff repeated the identical call: four requests of exactly 60 s (HTTP 500 after 1m0s in the Ollama log at 10:02:24, 10:03:25, 10:04:27) and a last one cut at the deadline (46.86 s at 10:05:18), then the rule tree decided. Nine Decision passes, ten profile_builder calls and two rag_retriever calls ended this way in r6. The prompt was 12,071 characters of system text plus 5,146 characters of tool schemas (seven tools, two of them answered by a stub, and a section about `check_system_health`, which Decision has not been offered since 31.10) plus the tool results fed back verbatim (the catalogue row with its 20-label list).

| Change (round 9) | Before | After | Test |
|---|---|---|---|
| Decision system prompt | 12,071 characters; sections about three tools the model cannot use | 7,472 characters; every rule kept, the dead sections and the duplicated prose gone | `tests/test_round9_decision_prompt.py` (size ceiling 7,600; the contract phrases present; the three tool names absent) |
| Tools offered to Decision | 7 schemas (5,146 characters), two answered by a stub | 5 schemas (3,485 characters); `_UNWIRED_DECISION_TOOLS` keeps `rank_similarity_to_catalog` and `check_recent_cluster` out of the offer, the dispatcher still answers them with the stub | same file; `tests/test_stage_b_latency.py` |
| Tool results fed back to the model | `json.dumps(result)[:2000]` | `_tool_result_for_model`: the catalogue row as `_dataset_lookup_summary` (no label list), other results with lists cut to 5 items and strings to 240 characters, the message to 1,200; the react_step in the trace keeps the full result | same file |
| Retry after a timeout | backoff repeated the identical call up to 4 times; the fallback chain could name the same model twice | `_is_timeout`: a call that ran out of its SDK timeout is not retried; the chain calls each distinct spec once | same file (timeout: 1 call, no sleep; 503: still 4 calls; chain dedupe) |
| Per-call timeout on the VM | `AGENT_LLM_TIMEOUT_SECONDS=60` | 150 (env only), so a 4,000-token call completes in one attempt; the run deadline still cuts it | runtime |

## 20. Round 10 (1.11.2026, afternoon): the owner's run loop and its first improvements

The owner now drives the work as a loop: every round deletes all cases without a backup (the agents start with no memory), runs the 94 images, and he reads the full trace of the round as a text file (only the trace, no summary) and names the problems; the next round starts only on his word. Round r7 started at 14:49 UTC on the round 9 build with `AGENT_LLM_TIMEOUT_SECONDS=150`. Its first cases already show the timeout fix: 13 of 13 Decision passes were decided by the model (r6: 13 of 104), no Decision call failed, and Ollama reused a warm prefix on all 13 Decision prompts (about 3,200 tokens each; prompt evaluation about 31 s, generation about 26 s for about 270 output tokens per call). The four evidence requests the model made were all inside the policy's ambiguous band (detector confidence 0.47 to 0.58).

From the comprehensive review he asked for, the owner approved items 1, 2, 3, 4, 5 and 7 below (6, a shorter `reasoning` field, declined; 8, splitting the module, deferred).

| # | Change (round 10) | Before | After | Test |
|---|---|---|---|---|
| 1 | Deterministic results in the Decision payload | the agent called `check_hazard_severity` in 47 of 47 tool-using passes and re-ran `lookup_dataset_image` on every catalogue match; each call is a ReAct round of 50-130 s on the CPU host | `severity_check` (the executed formula, `already_run=true`) and `dataset_match` marked as the confirmed lookup sit in the payload; the prompt says so; both tools stay in the offer | `tests/test_round10_loop.py` |
| 2 | Evidence pass reuse | the resumed run re-ran sanitize, intake, perception and research on the same photo (10 cases per round, up to 568 s each) | `_await_evidence_node` sets `evidence_reuse` when the photo hash, class, confidence, lane and road fields are unchanged and a profile and passages exist; the supervisor routes the pass straight to Decision; the trace says which workers were kept | same file |
| 3 | Warm prompt cache, shorter schemas and payload | the final Decision round dropped the tool schemas, which changed the prompt right after the system text and made Ollama re-evaluate about 1,000 tokens; schemas 3,485 characters; payload with source URLs and the full rule-hint dict | the schemas stay in every round and a one-line nudge closes the budget (one extra round for a model that still calls a tool); schemas 2,933 characters; payload without URLs, rule hint cut to five fields | same file |
| 4 | Expected-verdict table | no ground truth for any of the 94 images | `data/expected_verdicts.csv` (image, category, detector class and confidence, r6 verdict, `expected_verdict` for the owner to fill) | manual |
| 5 | Fallback reason in the trace | the rule tree's takeover showed only in the log as `llm_unavailable_or_unparseable` | `fallback_reason` on the stored decision entry with one slug per cause (`run_deadline`, `llm_call_failed:<type>`, `llm_reply_unparseable`, `llm_reply_ungrounded`, `llm_kept_calling_tools`) and the model's reply excerpt under it | same file |
| 7 | Worker impact | the workers' LLM labels were merged silently | `llm_delta` on the profile_builder and rag_retriever entries; `worker_baseline` on the state; Decision records `worker_impact` (the rule tree's action with and without the workers' labels) | same file |

Suite after the round: 216 tests. The build deploys for the next loop round, after r7 ends.

### 20.1 Live findings in r7 (round 10b, same day)

Watching the live trace during r7 (the owner asked for continuous live monitoring) showed two faults, both fixed before the round 10 deploy:

| Finding | Case | Cause | Fix | Test |
|---|---|---|---|---|
| A verdict with `priority=10` instead of 55 | hz_403dd7849e13 (34037.webp) | the model sent its final JSON (rationale, assessment, cited_sources) as the arguments of a second `check_hazard_severity` call; the tool scored the empty class at 10 and the verdict carried it | `_validate_tool_args` checks every Decision tool call against its JSON schema (required keys except the ones the executor fills, enums, ranges, types) and returns `invalid arguments`; `_looks_like_final_answer` reads a tool call whose arguments are the verdict as the final answer and executes nothing (the unwired stub tools keep answering regardless) | `tests/test_round10b_live_fixes.py` |
| Invented catalogue contradictions on web photos | hz_34b3e7d1f9a0, hz_e5b2736154e9, hz_ffb73fa18705 | the relaxed lookup (perceptual hash up to distance 16) and the profile builder's distance-10 hits reached the model as `dataset_match`; it read the neighbour's labels as this photo's and answered human_review or asked for evidence | `_catalogue_context_for_prompt`: only the same file (sha256) is `dataset_match`; a neighbour travels as `similar_catalogue_photo` with its distance and similarity, and the prompt says it is neither a known image nor a contradiction | same file |

Suite after 10b: 221 tests. The Decision prompt is 8,241 characters (12,071 before round 9).

## 21. Round 11 (1.11.2026, evening): the owner's reliability round

The owner read the r7 trace himself and named six problems; round 10/10b had already covered two of them (the second `check_hazard_severity` call with the verdict as arguments, and the Decision tool contracts). He asked for a reliability round before any new capability, and for r8 to run only once it is deployed together with rounds 10 and 10b.

| # | His finding | Change (round 11) | Test |
|---|---|---|---|
| 1 | `TypeError: _tool_lookup_similar_cases_worker() got an unexpected keyword argument 'severity_signal'` with `violations=none` | the worker loop applies the Decision contract: `_validate_tool_args` against the tool schema (a required key the function fills itself is fine), keys the function does not take are dropped, a reply JSON sent as a tool call (`_worker_final_in_tool_call`, two or more reply keys and none of the tool's) is used as the reply, and `_autonomy_record` adds `tool_call_failed:<tool>` plus `failed_tools`; the worker `lookup_similar_cases` schema gets the class enum | `tests/test_round11_reliability.py` |
| 2 | "The reporter's claims are consistent with the detector" on a photo with `has_text=False`; grounding (citations) passed | `_verify_decision_rationale` checks rationale and assessment against the inputs (reporter attribution without text, lane blockage not in the form, known-image wording without sha256, class codes outside the inputs, a priority number other than the verdict's); on a failure `_deterministic_rationale` replaces the text, the verdict and priority stay, the trace prints `rationale check: FAIL (...)` and the model's text | same file |
| 3 | `check_hazard_severity` twice, priority 10 | round 10b (schema validation, verdict-in-tool-call) | `tests/test_round10b_live_fixes.py` |
| 4 | autonomy metrics mix what the agent chose with what the code ran | `/stats/agents` counts `agent_called`, `auto_ran_tools` and `tool_failures` apart; the trace line shows `failed=` and `reply_in_tool_call=yes` | same file; `tests/test_worker_autonomy.py` |
| 5 | `direct_match` almost automatic, cut texts, D20 passages for D00 queries | `_passage_relevance` grades `support` (direct / weak / none) from the strongest class term against the other classes' terms and records `other_class_hits` and `sentence_complete`; `_coverage` gives `direct_match` only on direct support (tagged passages keep their tag); the summary the model reads is a whole sentence up to 320 characters (`_sentence_cut`) instead of the first 160 characters; the trace prints `support=` and `other_class=` per passage | same file |
| 6 | median above 90 s, no per-stage timing | `stage seconds:` under the DONE line (seconds between timestamped entries, attributed to the running node); the speed changes themselves are rounds 10 and 10b | same file |

Suite after round 11: 232 tests. Rounds 10, 10b and 11 deploy together for r8, after r7 ends.

## 22. Round 12 (1.11.2026, evening): the `why` argument

The owner asked why the trace reads `[RAG] step 1: (no reasoning text)`. The step's reasoning line is the text the model writes next to a tool call; qwen2.5:3b through Ollama emits the `<tool_call>` block alone, so every worker and Decision step in r7 and the start of r8 had no text, and only the Decision prompt had asked for a sentence. Since a small model fills arguments far more reliably than it writes prose, every offered tool schema now carries an optional `why` argument (`_with_why`, applied when the schemas are offered; the registry schema and the validation stay unchanged), the prompts ask for one sentence in it, both executors drop it before the function runs, and the step's `thought` falls back to it (`_step_thought`). Tests: `tests/test_round12_why.py`. Deploys for r9 with whatever the owner finds in the r8 trace.

## 23. Round 15 (1.11.2026, evening): the trace shows what the agent saw

The owner asked why a RAG step's `result:` line ends in "A..." mid-word. The worker loop stored each step's result cut at 220 characters and its payload at 160, the step formatter printed at most 320, and the passage previews in the RAG handoff were cut at 160; the model itself had received the tool message cut at 2,000. The stored result and the printed result and passages now keep up to 2,000 characters (payloads 600), so the trace carries the passages the agent actually read. Storage grows by a few kilobytes per case. Tests: `tests/test_round15_trace_width.py`.

## 24. Round 16 (1.11.2026, evening): the owner's r9 review, and Benny's new corpus

The owner stopped r9 after two cases, read them, and named six items; all six were built, and the worker time cap went from 45 to 90 s at his choice.

| # | His finding | Change (round 16) | Test |
|---|---|---|---|
| 1 | a reply JSON sent as a tool call was labelled a failed tool (`tool_call_failed`, `status: failed`) | the worker and Decision loops mark such a call `reply_in_tool_call` with `executed=false`; the step prints `status: reply (JSON arrived as a tool call; used as the agent's reply, no tool ran)`; it is no longer a `tool_call_failed` violation | `tests/test_round16_labels.py` |
| 2 | a result that followed a tool failure read as a clean result | the result line itself carries `tool_failures=<n>` and `reply_via=tool_call` ([PROFILE], [RAG], [DECISION]) | same file |
| 3 | autonomy metrics mixed the channels | the autonomy line reads `agent_called= hops= auto_ran= fallback= failed= reply_via= time=<used>s/<cap>s violations=`; /stats/agents already counts them apart (round 11) | same file; `tests/test_worker_autonomy.py` |
| 4 | `time_cap` without the numbers; does the cap bite? | the cap is enforced through each call's timeout (the call in flight is cut); the record now carries `time_used_s`, `time_cap_s`, `cut_by_cap`; `policy.yaml latency.worker_time_cap_s` 45 -> 90, because two worker calls on this CPU take 15 to 45 s each | same file |
| 5 | `direct_match` is not proof the standards answer the operational questions | `_decision_coverage` says which of four questions the retrieved passages answer (repair window, urgency, traffic control, safety risk); the RAG line prints `answers= missing=`, Decision receives `research_coverage_detail` | same file |
| 6 | an approximate catalogue match read as a hit | `SIMILAR IMAGE` in the trace for a perceptual-hash match (`KNOWN IMAGE` only for the same file); the live note reads `similar catalogued photo pic_id=... phash distance ... (not the same file)` | same file |

**Benny's corpus.** The team repository (shaisadeh/Final-Project-DS, commit 2fbdcfc of 31.10.2026 21:54) carries `Data/processed/english_vector_corpus_backup.json` (SHA256 be8b38274fcff413298cb916ed99877e14b16ca278add1a02956b3e468ed4398): 3,409 English chunks from 27 source documents (median 1,000 characters; the largest sources are Well-managed highway infrastructure, the 2024 general specification, temporary traffic arrangements, the Distress Identification Manual, IRC:82-2015), with the same id scheme as the previous corpus and no vectors. It was embedded on the VM with the application's own embedding function (ONNX all-MiniLM-L6-v2, 384 dimensions, 3 minutes) into `charter_grounded_corpus`, replacing the 2,433-chunk corpus of 26.10, whose directory the owner chose to delete. The relevance grid result is recorded with the deploy of r10.

Suite after round 16: 248 tests.

Relevance grid on the new corpus (1.11.2026 17:38 UTC, `scripts/rag_relevance_eval.py --k 3`): 16 of 16 class x road pairs `direct_match` with 3 of 3 class hits each (rate 100%, target 95%). Best distances: D20 0.21 to 0.29, D40 0.22, D10 0.30 to 0.36, D00 0.34 to 0.39 (the previous corpus gave 0.43 to 0.48 for D40/D20 and 0.63 to 0.75 for D10/D00); D00 and D10 now come from IRC:82-2015, the Distress Identification Manual and document 13092. Round r10 started at 17:39:54 UTC on e01c0de with this corpus.

## 25. Round 17 (1.11.2026, late evening): the owner's r10 review

Run r10 (build e01c0de, Benny's corpus) was stopped at 25 of 94 on the owner's word after he read the live trace twice and named his priorities. Two of his readings were checked against the stored data before anything was built:

- **hz_81282e45e8bf is not a state mix-up.** The stored row carries seven YOLO boxes: D20 at 0.885 (the whole frame), D40 at 0.457, D00 at 0.248 and four D40 boxes below 0.2. The Decision's "secondary D40 class" is the second box, which the trace never showed. "D20 pothole" and "no evidence of severe conditions" (the deterministic profile said `severe`) are the model's wording against the inputs, and "low detector confidence (0.885)" is the supervisor's; none of the three was a check the fact-check ran.
- **The 131 s on approval_gate were a timestamp bug, not waiting.** Only ReAct steps were stamped when created; every other entry got `ts` when the row was persisted (`roadguard_api.py` on save, `emit_live_trace` on print), so `stage seconds` attributed the span to the last entries.

| # | His finding | Change (round 17) | Test |
|---|---|---|---|
| 1 | state integrity (D20 / D40, severity, "low confidence") | MODEL and PROFILE lines list every detection; the fact-check gains class-name, severity-word and confidence-band checks; the supervisor note check gains the band check; every agent receives `detector_confidence_band` from the policy thresholds | `tests/test_round17_integrity.py` |
| 2, 3 | the fact-check replaced the text and the verdict stayed | one correction round, then the rule tree over a verdict more severe than its own (`rationale_unverified`); a verdict not more severe is kept | same file |
| 4 | the same photo counted as evidence | the API refuses the original file without new text (422, request not consumed); the same file with a note is `evidence_declined` and goes to a human without a model pass; the batch runner answers with the note | same file; `tests/test_round14_resume_inputs.py` |
| 5 | `check_hazard_severity` called with `{}` | missing arguments come from the state (`args_from_state` in the result and the step); wrong values still fail | same file; `tests/test_round10b_live_fixes.py` |
| self-match | `lookup_similar_cases` returned the case being triaged | the current id is excluded on every path (the API stores the row before the graph runs); `note` names the outcome | same file |
| 6 | `direct_match` read as a full basis | `sufficiency` next to `coverage` (sufficient only with a repair window and an urgency or a safety risk); the Decision payload says in words when the standards state none; `repair_window_not_in_sources` | same file |
| 7 | performance accounting | entries stamped at creation; `waiting_evidence` apart from compute on the DONE and stage lines | same file |
| mine | `NEEDS_MORE_EVIDENCE priority=0` next to a severity check of 32 | priority 0 on a non-sealed verdict takes the severity check (`priority_from_severity_check`); the worker's reason follows a severity change on the `llm changed` line | same file |

Suite after round 17: 267 tests. Deployed for run r11 after the wipe.

