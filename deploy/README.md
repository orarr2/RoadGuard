# RoadGuard VM Deploy

The VM is a Google Cloud `e2-micro` Free Tier instance in `us-east1-c`,
hostname `turkey-collector`, running Debian 13. Scripts in this
directory bring it from a clean slate to a running RoadGuard service.

> **Day-to-day commands** live in [VM-COMMANDS.md](VM-COMMANDS.md) -
> dataset download, verify-vm, SQLite inspection, enabling the LLM
> panel, approving hazards from the shell, etc. This file covers the
> one-shot setup and the systemd layout.

## One-shot setup

On a freshly provisioned VM:

```bash
curl -L https://raw.githubusercontent.com/orarr2/RoadGuard/main/deploy/setup.sh \
  -o /tmp/roadguard-setup.sh
sudo bash /tmp/roadguard-setup.sh
```

What the script does:

1. Installs system prerequisites (`git`, `python3-venv`, `build-essential`,
   `cmake`, `pkg-config`, `sqlite3`).
2. Creates a `roadguard` system user and the directories under
   `/opt/roadguard`, `/var/lib/roadguard`, `/etc/roadguard`.
3. Clones (or updates) this repo at `/opt/roadguard/repo`.
4. Creates a Python 3.11 venv and installs `requirements.txt`.
5. Downloads the Llama 3.2 1B Q4_K_M gguf (~800 MB) for the fallback
   LLM.
6. Copies `.env.example` to `/etc/roadguard/env` with mode 0600 (only
   on first run). Generates a `ROADGUARD_JWT_SECRET` if the placeholder
   is still present.
7. Installs and enables the systemd units.

## Services and timers

| Unit | Role |
|---|---|
| `roadguard.service` | FastAPI + uvicorn behind Cloudflare quick tunnel |
| `roadguard-image-cleanup.service` + `.timer` | Nightly 03:30 UTC: purge Supabase images past TTL |
| `roadguard-daily-email.service` + `.timer` | Daily 06:00 UTC: Gmail SMTP summary to opted-in users |
| `cloudflared.service` | Quick tunnel from `127.0.0.1:8000` to a public `trycloudflare.com` URL (see `cloudflared.md`) |

Service hardening is on by default (`ProtectSystem=strict`, read-write
paths limited to `/var/lib/roadguard`, `MemoryMax=700M`). The e2-micro
has 1 GB of RAM; leaving headroom lets llama.cpp load the fallback
model only when it is needed.

## After setup

1. Fill in `/etc/roadguard/env` with real values:
   `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`,
   `ROADGUARD_PASSPHRASE_HASH`, `GOOGLE_OAUTH_CLIENT_ID`,
   `EMAIL_APP_PASSWORD`, and so on.
2. (Optional) Build the RAG index:
   `sudo -u roadguard /opt/roadguard/venv/bin/python -m tools.build_rag_index`
3. Start the units:
   ```bash
   sudo systemctl start roadguard.service
   sudo systemctl start roadguard-image-cleanup.timer
   sudo systemctl start roadguard-daily-email.timer
   ```
4. Follow `cloudflared.md` to open the quick tunnel.

## Health

```bash
curl -s http://127.0.0.1:8000/health | jq .
```

Should return `{"status": "ok", ..., "agent_ready": true, "llm_fallback_ready": true}`
once the env file is populated and the gguf is in place.

## End-to-end verification

`verify-vm.sh` is a read-only check that covers the services, the API,
the approval gate, the dataset, and disk space in one pass. It never
mutates state, so it is safe to rerun whenever something changes.

```bash
# Minimal (services + /health + dataset only)
sudo bash /opt/roadguard/repo/deploy/verify-vm.sh

# Full (also runs /hazards/submit and /hazards/live and checks that an
# approval row was written for the urgent verdict)
sudo bash /opt/roadguard/repo/deploy/verify-vm.sh \
    --passphrase "$(systemd-ask-password 'RoadGuard passphrase:')"

# Pass through the dataset path if it is not the default
sudo bash /opt/roadguard/repo/deploy/verify-vm.sh \
    --dataset-dir /opt/roadguard/datasets/road-damage-10k \
    --passphrase "$MY_PASSPHRASE"
```

The script prints `PASS`/`WARN`/`FAIL` lines, ends with a count summary,
and exits with the number of FAILs (0 means every required check passed).

## Logs

```bash
sudo journalctl -u roadguard.service --since "-10 min"
sudo journalctl -u roadguard-image-cleanup.service --since "-1 day"
sudo journalctl -u roadguard-daily-email.service --since "-1 day"
```

## Updating

```bash
sudo bash /opt/roadguard/repo/deploy/setup.sh
sudo systemctl restart roadguard.service
```

The script is idempotent: it pulls `origin/main` with `git reset --hard`,
reinstalls requirements if they changed, leaves the env file alone, and
reloads systemd units.

## Uninstall

```bash
sudo systemctl disable --now roadguard.service \
  roadguard-image-cleanup.timer roadguard-daily-email.timer \
  cloudflared.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/roadguard*.service \
  /etc/systemd/system/roadguard*.timer \
  /etc/systemd/system/cloudflared.service
sudo systemctl daemon-reload
sudo rm -rf /opt/roadguard /var/lib/roadguard
sudo rm -rf /etc/roadguard
sudo userdel roadguard
```
