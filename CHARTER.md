# RoadGuard Project Charter

AI-powered road hazard detection and prioritization system.

This charter adapts the course project proposal to English markdown. It
describes the full RoadGuard system; this repository focuses on the agent
track and ships the end-to-end scaffolding required to run and evaluate it.

## 1. Problem and user

A municipal infrastructure control-center manager, or a public works
manager, receives road-hazard reports and photographs of inconsistent
quality. They must quickly determine the hazard type and whether it
requires urgent inspection, scheduling in the maintenance plan, or
additional evidence. Uniform manual triage delays important cases and
may dispatch crews to false alarms.

RoadGuard analyzes an image, detects cracks and potholes, and recommends a
justified next action. It is a decision-support system only: it will not
determine engineering fitness or close a road without approval from an
authorized professional.

The long-term operational objective is to shorten triage and repair times
and reduce unnecessary site visits. Reducing response time from 72 hours
to 4 hours and contractor costs by 35% are treated as future pilot
targets, because the public dataset does not contain repair-time, cost, or
claim data with which to measure them during the course project.

## 2. Success metrics

| Component | Target and evaluation |
|---|---|
| Primary metric | At least 80% recall for pothole class D40 on the test set |
| Secondary metric | mAP@0.5 of at least 0.55 across all classes, reported by class and country |
| Baseline | HOG with SVM or Random Forest on annotated damage regions, compared with YOLO or Faster R-CNN |
| Agent evaluation | 85% correct decisions across 50 scenarios; 90% of explanations grounded in a retrieved source |

Where metadata permits, the split is performed by country or image
sequence to prevent leakage between neighboring frames. A field test is
conducted on 30 to 50 local images that are not used for training.

## 3. Data and plan B

**Primary source.** RDD2022 contains 47,420 road images from six countries
and more than 55,000 annotated damage instances. Its four shared classes
are longitudinal crack D00, transverse crack D10, alligator crack D20,
and pothole D40. The annotations support object detection, and the
dataset is distributed under the CC BY-SA 4.0 license. Source:
https://github.com/sekilab/RoadDamageDetector

The dataset does not include pothole depth, engineering severity, repair
time, or cost. Bounding-box area is not presented as physical damage area
or true severity. Operational priority is calculated from the hazard
class, model confidence, and user-provided metadata such as road type and
lane obstruction.

**RAG corpus.** A controlled corpus of official road-maintenance guidance
and response procedures. Each passage retains the issuing body, date,
hazard type, triggering conditions, and source link. If sufficiently
detailed Israeli guidance is unavailable, official international manuals
are used and clearly labeled as non-binding in Israel.

**Plan B.** If dataset size or training time becomes a constraint, a
subset containing potholes and alligator cracks is used. If object
detection does not converge, the task switches to crop classification
without precise localization. The thematic fallback is TACO, a
public-space litter detection dataset licensed under CC BY 4.0.

## 4. System architecture

```
Image + metadata  ->  Quality check  ->  CV model  ->  Prioritization agent  ->  Recommended action
```

The agent also receives the procedures corpus and the project safety
rules, then chooses among urgent inspection, the standard work queue, a
request for additional evidence, and human review. The first version
processes an uploaded image through validation, detection, bounding-box
display, procedure retrieval, and recommendation generation. Connections
to a municipal call center, map, or contractor-management system are
demonstrated with mock tools only.

## 5. The agent's seat

**Decision.** The agent combines the computer-vision output, image
quality, road type, lane obstruction, model confidence, and retrieved
procedure. It selects urgent inspection, the standard queue, a request
for additional evidence, or human review. A pothole obstructing a lane on
a major road is escalated for urgent inspection; an uncertain finding is
routed for a new photograph or for human review.

**Agentic architecture is required.** The hazard triage and dispatch
workflow cannot be reduced to a static script or a fixed `if-else` rule
chain. The operational sequence and tool usage must adapt dynamically at
runtime based on unstructured and variable inputs.

**Tools.** The Decision agent can call any of the following during its
ReAct loop; the other workers each hold a smaller subset.

Action tools (write):
- `check_hazard_severity` calculates operational priority using
  transparent rules.
- `request_additional_evidence` requests another photograph or missing
  information (capped at one request per hazard).
- `create_draft_work_order` creates a local draft task only.
- `flag_for_urgent_review` alerts a human supervisor and does not close
  a road.

Read-only tools (observation):
- `query_repair_standards` retrieves relevant guidance and source
  passages from the ChromaDB corpus.
- `lookup_dataset_image` exact-matches the submitted photo against the
  RDD2022 catalogue; a hit short-circuits to `known_image` with no LLM
  triage.
- `rank_similarity_to_catalog` returns the top-K nearest catalogued
  images with a similarity percentage, so the agent can hedge
  ("looks ~82% like catalogued D40 pic_id=1024") instead of a binary
  known / new.
- `check_recent_cluster` counts same-class reports at the same
  free-text location; returns `cluster_flag` = none / emerging /
  strong, so recurring issues get flagged.
- `check_system_health` reports ollama RAM, loaded models and the
  stub-mode flag so the agent knows when the pipeline is degraded.
- `lookup_similar_cases` (profile builder) returns prior hazards of
  the same class at the same location.

**Fallback.** If the image is insufficient, confidence is low, no
supporting procedure is found, or a tool fails, the agent does not create
a work order. The case is transferred to human review with a plain-language
rationale explaining what could not be determined. Without the agent, the
system still displays bounding boxes and a findings list, but does not
prioritize cases automatically.

**Agent character.** The Decision prompt frames the LLM as a seasoned
municipal road-safety inspector with explicit "take ownership and decide"
guidance - low detector confidence is not by itself a reason to escalate,
contradictions are signals to use judgement, exaggeration should be
named, non-road images and non-Israeli submissions should be called out.
Determinstic floor rules that pre-empted the LLM (zero confidence, below
0.5 confidence, contradiction) were removed after they were shown to
route ~60% of real submissions to human review.

**Input hardening.** All free text (reporter description, location, OCR
of in-image signage) runs through sanitize before any LLM sees it:
injection patterns are neutralised, the reporter text is wrapped in
`[USER_SUPPLIED_UNVERIFIED]` markers, and PII (phone, email, Israeli
national id, IBAN, credit-card-shaped numbers) is redacted before the
hazard row is stored or shown on an operator page.

## 6. Risks and cut-down version

| Risk | Mitigation |
|---|---|
| Domain gap between dataset countries and Israel | Evaluate by country and on a local field set; report performance degradation explicitly |
| No severity or depth labels | Do not train a severity target; use rule-based operational priority and metadata |
| An incorrect action creates safety risk or expense | Keep external actions mocked; require human approval for every urgent or uncertain case |

**Cut-down version.** If the advanced components are not working by the
mid-project checkpoint, we deliver a system that detects potholes and
alligator cracks, retrieves relevant guidance, and classifies each case
as urgent or routine using fixed rules. Severity estimation, mapping, and
work-order generation are omitted.

## 7. Milestones and ownership

The course project is delivered by a four-person team. This repository
hosts the agent track and the scaffolding needed to run and evaluate it
end to end.

| Phase | Deliverable |
|---|---|
| Phase 1 | Download, licensing, EDA, and leakage-safe split |
| Phase 2 | Baseline and image preprocessing |
| Phase 3 | Detector training and error analysis |
| Phase 4 | Corpus, retrieval, and groundedness |
| Phase 5 | Agent tools and evaluation scenarios |
| Phase 6 | Integration, field test, demo, and presentation |

At the mid-project checkpoint, the team makes a Go or No-Go decision on
supporting all four classes. Expansion occurs only after the two-class
version works end to end. Every substantial change is reviewed by
another team member.

## Addendum - 2026-10-27

The 26.10 batch of 94 citizen hazard images was replayed on the
production VM after wiring in Benny's `charter_grounded_corpus`
(2,433 passages, ChromaDB 1.5.9, nomic-embed-text embedder). The
batch surfaced four classes of defect that have since been fixed:

1. **KNOWN_IMAGE floor** on catalogued photos was closing severe
   hazards at `priority=0`, hiding the live detector's reading of
   high severity. The floor is now gated on
   `severity in {minor, moderate}`; severe or urgent catalog matches
   flow through the regular rule tree with `catalog:pic_id=X`
   appended to `cited_sources` and the rationale tagged with
   `catalog recognised this frame...`.
2. **Three Decision ReAct tools** (`check_system_health`,
   `check_recent_cluster`, `rank_similarity_to_catalog`) were listed
   in the schema with no dispatcher branch. The first is now wired to
   the real probe; the other two return a structured stub so the
   model stops retrying and the trace records the attempt.
3. **RAG retrieval** returned `safety-non-road-*` chunks for every
   D40/D20/D10/D00 road query. Those chunks are now filtered out
   unless the current frame reads as a non-road scene
   (`perception_confidence=low`); coverage is recomputed on the
   filtered set.
4. **Supervisor LLM** ran 3-5 times per hazard because the dedup
   check missed `None` deliveries from LangGraph. A belt-and-braces
   trace scan was added; the LLM now runs exactly once per hazard.

Trace output was enriched for post-hoc audit:
`rag_query`, `rag_filter`, `passage_previews`,
`floor_reason` + `floor_trigger`, per-cite `grounded_per_source`,
`uncertainty`, `llm_fallback`, `budget_state`, and an
`await_evidence` `delta` showing before/after of the reporter's
upload. Line-length budgets grew too, so trace lines are no longer
clipped mid-sentence. See `PLAN.md` section 11 and
`docs/agents-architecture.md` for the detail.
