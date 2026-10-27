# RoadGuard - Agent Pipeline (standalone)

End-to-end run of the 5-worker + supervisor + approval-gate agent pipeline
that recommends a hazard action from a road-damage image. No FastAPI
service, no SQLite, no VM, no webapp, no paid API; a plain Python kernel
is enough.

## Contents of this folder

| File | Purpose |
|------|---------|
| `roadguard_standalone.ipynb` | The notebook. Runs top-to-bottom. |
| `README.md` | This file. |

## What the pipeline reads from the team

The notebook does NOT ship the detector, images or RAG corpus. It expects
the team's deliverables at these conventional paths next to it:

| Dependency | Path | Owner |
|------------|------|-------|
| YOLOv8n ONNX detector | `../models/yolov8n-roadguard.onnx` | Shai |
| Sample road images | `../data/sample/` | Ovad |
| ChromaDB corpus `charter_grounded_corpus` | `../rag/chroma_db/` | Benny |

The folder layout the notebook assumes relative to this directory:

```
<team-repo-root>/
  models/
    yolov8n-roadguard.onnx
  data/
    sample/
      *.jpg | *.png | *.webp
  rag/
    chroma_db/
      ...chromadb persistent directory...
  standalone/
    roadguard_standalone.ipynb
    README.md
```

If any dependency is missing the cell that uses it raises a clear
`FileNotFoundError` naming the expected path.

## Requirements

- Python 3.10 or newer.
- `pip install onnxruntime numpy pillow chromadb jupyter`
- Open the notebook from this directory (`cd standalone`) so the relative
  paths above resolve correctly.

## How to run

```bash
cd standalone
jupyter notebook roadguard_standalone.ipynb
```

Then run every cell top to bottom. The demo section loads up to five
images from `../data/sample/`, runs YOLO on each, and prints the
pipeline's verdict + priority + cited passages per image.

## Verdicts

- `urgent` - a human supervisor must approve dispatch of a crew within
  hours.
- `standard` - the hazard enters the routine repair queue.
- `needs_more_evidence` - the reporter is asked for one more photo.
- `human_review` - the pipeline cannot ground a safe decision; the case
  is routed to a supervisor with a full trace.

## Agents

1. **sanitize** - boundary defence: strips PII from the reporter's free
   text and neutralises prompt injection patterns.
2. **signal_extractor** - parses the reporter's description into
   structured signals (claimed class, lane blocked, urgency).
3. **profile_builder** - fuses YOLO output + image quality + metadata
   into a hazard profile (primary class, severity signal, perception
   confidence).
4. **rag_retriever** - queries Benny's `charter_grounded_corpus` under
   `../rag/chroma_db/` for repair-standard passages. The Decision agent
   later cites a subset by passage id.
5. **decision** - takes everything the first four produced and emits
   the verdict, priority, cited sources and an optional evidence
   request.
6. **supervisor** - a lightweight Python router. Enforces the per-run
   budget (iterations, cost, wall clock) and the per-agent tool-hop
   cap.
7. **approval_gate** - every `urgent` or `human_review` verdict is
   wrapped in an approval record waiting for a human sign-off.
