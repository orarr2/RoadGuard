# RoadGuard

**Live site: https://orarr2.github.io/RoadGuard/**

Real-time road hazard decision-support agent. The back camera streams
while the car drives, a YOLO model runs on-device, an IoU tracker
collapses frames into one submission per confirmed hazard, and a
3-agent orchestrator-worker system (Perception + Research + Decision,
with a Python supervisor) recommends one of four actions: `urgent`,
`standard`, `needs_more_evidence`, or `human_review`. Every verdict
cites a maintenance procedure. Nothing closes a road or dispatches a
contractor; a human reviews each recommendation.

## Access

- Webapp (public): https://orarr2.github.io/RoadGuard/
- Alternate URL (served from the VM through a Cloudflare quick tunnel):
  the URL rotates when `cloudflared.service` restarts; the latest value
  is pushed into [`webapp/tunnel-url.json`](webapp/tunnel-url.json).
- API routes (same origin as the webapp through the tunnel):
  `GET /health`, `POST /auth/login`, `GET /auth/me`,
  `POST /hazards/live`, `POST /hazards/submit`.

## Architecture at a glance

```
Phone / dash
  back camera -> YOLO ONNX (on-device, onnxruntime-web)
                -> IoU tracker -> one POST per confirmed hazard
                                        |
                                        v
Cloudflare quick tunnel  ------->  FastAPI VM (e2-micro)
                                        |
            server-side DecisionCache (session_id + track_id + class
            + road + lane, TTL 120 s) deduplicates resurfaces
                                        |
                                        v
          +--------------------- SUPERVISOR (python router) --------+
          |                              |                          |
          v                              v                          v
   Perception node             Research node                 Decision node
   (hazard profile +           (RAG query_repair_            (chooses one of four
    IoU + quality signals)      standards, read-only)        actions, cites sources)
                                        |
                                        v
                              AgentDecision -> HazardSubmitResponse
```

- Every handoff is validated by `assert_v3` (schema + grounding check).
- Budget guard (4 caps: iterations, hops per agent, cost, wall-clock)
  returns `human_review` with `fallback_reason=guard_fired` instead of
  crashing when any cap trips.
- Boundary defence sanitizes free-text location, OCR text pulled from
  the image, and RAG passages (dual-LLM pattern).

See [AGENTS/architecture.md](AGENTS/architecture.md) for the
authoritative spec and [AGENTS/README.md](AGENTS/README.md) for the
one-page overview. The walk-through is
[notebooks/agent_walkthrough.ipynb](notebooks/agent_walkthrough.ipynb)
(deterministic, no network needed, runs end to end in a few seconds).

## What is live today

| Component | Status |
|---|---|
| `/health`, `/auth/*`, `/hazards/live`, `/hazards/submit` on the VM | up |
| GitHub Pages mirror of the webapp | up |
| Cloudflare quick tunnel from the VM | up |
| 3-agent graph (multiagent) with grounded citations | up, 50/50 scenarios, 100% grounded on stub |
| Groq provider (gpt-oss-120b) for the agent LLM | wired, `ROADGUARD_USE_LIVE=1` |
| Decision cache (per-session, 120 s TTL) | up |
| YOLO ONNX model | **pending Shai** (mock detector active until the ONNX lands under `webapp/assets/models/`) |
| ChromaDB production corpus | **pending Benny** (stub corpus answers `query_repair_standards` today) |
| Local Llama fallback (gguf) | optional, not installed on the e2-micro to leave disk headroom |
| Supabase image storage | deferred (live flow does not upload full frames) |
| SQLite history + daily email | wired in code, surface-only in API |

## Repository layout

```
src/
  agent/
    multiagent/     3-node orchestrator-worker: nodes, tools, guards,
                    chat wrapper, sanitize, measure, graph, adapter
    loop.py         single-agent monolith (older path, kept for A/B)
    evaluation/     scenarios.json + run_eval.py (10 seed scenarios)
    policy.py       charter-mandated fallbacks + decision table
  api/              FastAPI shell, auth, rate limits, decision_cache,
                    schemas (incl. LiveHazardSubmit + LiveHazardResponse)
  storage/          Supabase client (ready, not yet wired to a bucket)
  db/               SQLite schema + models + approvals table
  jobs/             Daily email + nightly image cleanup + approval email
  tools/            Operational helpers (download_dataset.py, ...)
tests/              pytest suite (260 tests; multiagent + api + db + jobs)
notebooks/          agent_walkthrough.ipynb (standalone demo)
AGENTS/             README, architecture.md, contract.md, diagrams.html
                    (one place for everything about the agent layer)
docs/               CHARTER.md, dataset-handoff.md
deploy/             setup.sh, verify-vm.sh, VM-COMMANDS.md, systemd units,
                    Cloudflare tunnel notes
webapp/             GitHub Pages client (vanilla ES modules, dark design
                    system, live-drive flow, verdict toasts)
tests/multiagent_eval_scenarios.json   50 scenarios across 4 approaches
```

## Local development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

Full suite: 260 tests, green.

## VM provisioning

```bash
curl -L https://raw.githubusercontent.com/orarr2/RoadGuard/main/deploy/setup.sh -o /tmp/roadguard-setup.sh
sudo bash /tmp/roadguard-setup.sh
```

Idempotent. Uses `requirements-vm.txt` (the lean subset that fits the
e2-micro Free Tier) and leaves the heavy packages (`llama-cpp-python`,
`chromadb`) off until there is disk and RAM headroom. See
[deploy/README.md](deploy/README.md).

## Dataset

Backup script: `src/tools/download_dataset.py` pulls
`xjsj/road-damage-10k` v1 (YOLOv8 format, 9,926 images, CC BY 4.0)
from Roboflow Universe to `/opt/roadguard/datasets/road-damage-10k`
on the VM. Shai trains the detector; the agent only needs the class
list from the resulting `data.yaml` to match
`webapp/assets/config.js` `YOLO_CLASS_NAMES`.

## License

MIT. See [LICENSE](LICENSE).

External resources (not redistributed through this repository):

- Roboflow dataset `xjsj/road-damage-10k` is CC BY 4.0; backed up on
  the VM only.
- RDD2022 (Sekilab) is CC BY-SA 4.0; referenced in
  [docs/CHARTER.md](docs/CHARTER.md).
