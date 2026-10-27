# RoadGuard - Standalone Agent Pipeline

Self-contained Jupyter notebook that runs the RoadGuard road-damage
triage agent pipeline end-to-end **without the production VM, SQLite,
ChromaDB, the FastAPI service, the webapp, or any paid API**. The agent
layer reasons purely from the Python state passed between cells.

## Contents of this folder

| File | Purpose |
|------|---------|
| `roadguard_standalone.ipynb` | The notebook. 27 cells: 15 markdown + 12 code. Runs top-to-bottom. |
| `README.md` | This file. |

Nothing else is needed. The notebook ships five demo hazard scenarios
built into the last section; no external data file, no model weight, no
network call is required for the deterministic rule-tree path.

## Requirements

- Python 3.10 or newer (tested on 3.11 and 3.12).
- Jupyter (`pip install jupyter notebook` or any IDE that opens
  `.ipynb` such as VS Code or PyCharm).
- Optional: `numpy` for the seed-pin diagnostic cell. Everything else
  the core pipeline uses is in the Python standard library.
- Optional for the live-LLM path: `requests` and a Groq API key
  (`GROQ_API_KEY` env var). The notebook falls back to the deterministic
  rule tree when the key is missing - exactly as the production system
  does when its own LLM provider is unavailable.

Install the two optional deps in one line if you want them:

```bash
pip install numpy requests
```

## How to run

```bash
cd standalone
jupyter notebook roadguard_standalone.ipynb
```

Then run every cell top to bottom. The last section runs five demo
scenarios through the pipeline and prints a verdict table. Each row is
a hazard with its final `action`, `priority`, `cited_sources` and the
rationale the Decision agent produced.

## What the pipeline does

Given a `(hazard_class, confidence)` detection output plus a reporter's
free text, image-quality signals and road metadata, the pipeline
produces one of four verdicts:

- `urgent` - a human supervisor must approve dispatch of a maintenance
  crew within hours.
- `standard` - the hazard enters the routine repair queue.
- `needs_more_evidence` - the reporter is asked for one more photo
  (second angle, closer shot, lighting, etc.).
- `human_review` - the pipeline cannot ground a safe decision; the case
  is routed to a supervisor's inbox with a full trace of what every
  agent saw.

Plus one short-circuit (`known_image`) that the full production system
raises against the dataset catalogue; it is intentionally not
demonstrated here because the catalogue sits on the production VM.

## Agent list

Five workers + one supervisor + one approval gate:

1. **sanitize** - boundary defence: strips PII from reporter text and
   neutralises prompt injections.
2. **signal_extractor** - parses the reporter's description into
   structured signals (claimed class, claimed lane blocked, urgency).
3. **profile_builder** - fuses YOLO output with image-quality signals
   into a hazard profile (primary class, severity signal, perception
   confidence).
4. **rag_retriever** - fetches repair-standard passages that back the
   verdict's citations. In this standalone demo the retrieval is a
   small in-memory corpus; the production system queries a seeded
   ChromaDB corpus.
5. **decision** - takes everything the first four produced and emits
   the verdict + priority + cited sources + optional evidence request.
6. **supervisor** - a lightweight Python router that enforces the
   budget guard and the per-agent tool-hop cap.
7. **approval_gate** - every `urgent` and `human_review` verdict waits
   for a human sign-off.

The pipeline is implemented as pure functions
(`state -> state`), so the Jupyter diff between cells is clean and the
rule tree is reproducible across runs.

## Attribution

The agent architecture, prompts and rule-tree thresholds are mine
(Or Arbeli, orarbeli1@gmail.com). The YOLO detector weights (not
used here - we take its output as input) are the team's deliverable.
See the full production system at <https://github.com/orarr2/RoadGuard>.
