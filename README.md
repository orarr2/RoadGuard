# RoadGuard

A decision-support system for municipal infrastructure managers. A photo of
a road is captured in the browser, a YOLOv8 model running on-device detects
cracks and potholes, and a tool-using LLM agent recommends one of four
actions: urgent inspection, standard work queue, request for additional
evidence, or human review.

The system is decision support only. It never closes a road, dispatches a
contractor, or files a real work order. Every recommendation is surfaced
with a cited maintenance procedure so a human can act on it.

## Architecture

```
Browser  ->  Supabase Storage  ->  FastAPI VM
  |                                   |
  | YOLO ONNX inference (on-device)   | Agent loop (5 tools, Groq + Llama fallback)
  | Signed upload (image stays here)  | ChromaDB RAG over road-maintenance corpus
  | Metadata POST only                | SQLite log, nightly email, 48h image cleanup
```

- **Client**: GitHub Pages webapp, environment-facing camera, in-browser
  YOLO via `onnxruntime-web`, signed PUT to Supabase, metadata POST to the
  VM.
- **VM**: Google Cloud e2-micro Free Tier (1 GB RAM), FastAPI behind a
  Cloudflare quick tunnel. Agent orchestrates five tools via Groq function
  calling; a local Llama 3.2 1B Q4_K_M gguf is the fallback.
- **Storage**: User-owned Supabase bucket, 48 hour TTL enforced by a
  nightly cron on the VM. The VM database stores only derived metadata.

## Repository layout

```
src/
  agent/            Tool-using decision agent (five tools, decision loop)
  api/              FastAPI endpoints, auth, rate limits
  storage/          Supabase client (presign, delete, signed read)
  db/               SQLite schema and models
  jobs/             Daily email + nightly image cleanup
  tools/            Operational helpers (RAG index build, etc.)
tests/              pytest suite
docs/               Charter, agent contract, deployment notes
deploy/             VM setup script, systemd units, Cloudflare notes
webapp/             GitHub Pages client (vanilla ES modules, YOLO ONNX)
data/               Split lists for RDD2022 (filenames only, no raw bytes)
corpus/             Road-maintenance RAG corpus (procedures, standards)
```

## Local development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## License

MIT. See [LICENSE](LICENSE).

External datasets used only locally during training:

- RDD2022 (CC BY-SA 4.0, Sekilab) is downloaded for YOLO fine-tuning and
  is not redistributed through this repository.
