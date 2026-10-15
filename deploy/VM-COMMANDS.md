# VM Commands

Cheat-sheet for the Google Cloud `e2-micro` running RoadGuard. Copy-paste
blocks top to bottom. Nothing here mutates data you cannot recreate.

> The host is `turkey-collector` (us-east1-c, Debian 13). The service
> user is `roadguard`; everything lives under `/opt/roadguard`,
> `/var/lib/roadguard`, and `/etc/roadguard`.

---

## 1 - First boot (one-shot setup)

```bash
curl -L https://raw.githubusercontent.com/orarr2/RoadGuard/main/deploy/setup.sh \
  -o /tmp/roadguard-setup.sh
sudo bash /tmp/roadguard-setup.sh
```

Then fill `/etc/roadguard/env` with the real secrets (see
[../.env.example](../.env.example)) and start the services:

```bash
sudo systemctl start roadguard.service
sudo systemctl start roadguard-image-cleanup.timer
sudo systemctl start roadguard-daily-email.timer
```

Follow [cloudflared.md](cloudflared.md) to open the public tunnel.

---

## 2 - Download the 10K labelled dataset

```bash
sudo -u roadguard /opt/roadguard/venv/bin/pip install roboflow
sudo -u roadguard /opt/roadguard/venv/bin/python -m tools.download_dataset \
    --workspace xjsj --project road-damage-10k --version 1 \
    --dest /opt/roadguard/datasets/road-damage-10k
```

The script needs `ROBOFLOW_API_KEY` either in `/etc/roadguard/env` or
exported in the shell before the call. It is idempotent: a second run
re-fetches only missing files.

---

## 3 - End-to-end verification

```bash
# Fast version: services + /health + dataset + env sanity
sudo bash /opt/roadguard/repo/deploy/verify-vm.sh

# Full version: also POSTs /hazards/submit and /hazards/live,
# then checks that an approval row was written for the urgent verdict
sudo bash /opt/roadguard/repo/deploy/verify-vm.sh \
    --passphrase "$(systemd-ask-password 'RoadGuard passphrase:')"

# With a non-default dataset path
sudo bash /opt/roadguard/repo/deploy/verify-vm.sh \
    --dataset-dir /opt/roadguard/datasets/road-damage-10k \
    --passphrase "$MY_PASSPHRASE"
```

Prints `PASS` / `WARN` / `FAIL` per check and exits with the FAIL count.
`0` means every required check passed.

---

## 4 - Health and status

```bash
# API health JSON
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

# Public tunnel URL (also saved in webapp/tunnel-url.json)
cat /etc/cloudflared/tunnel.url 2>/dev/null \
  || systemctl status cloudflared.service

# Services at a glance
systemctl status \
  roadguard.service \
  roadguard-image-cleanup.timer \
  roadguard-daily-email.timer \
  cloudflared.service
```

---

## 5 - Logs

```bash
# Last 10 minutes of the API
sudo journalctl -u roadguard.service --since "-10 min" --no-pager

# Nightly image-cleanup run
sudo journalctl -u roadguard-image-cleanup.service --since "-1 day" --no-pager

# Daily email run
sudo journalctl -u roadguard-daily-email.service --since "-1 day" --no-pager

# Follow the API live
sudo journalctl -u roadguard.service -f
```

---

## 6 - Database inspection (SQLite)

```bash
# Pending approvals right now
sudo sqlite3 /var/lib/roadguard/roadguard.db \
    "SELECT id, hazard_id, action, priority, created_at FROM approvals
     WHERE status='pending' ORDER BY created_at DESC;"

# Last 10 hazards triaged
sudo sqlite3 /var/lib/roadguard/roadguard.db \
    "SELECT id, class, road_type, lane_blocked, agent_decision, priority
     FROM hazards ORDER BY reported_at DESC LIMIT 10;"

# Backup
sudo cp /var/lib/roadguard/roadguard.db \
        /var/lib/roadguard/backups/roadguard-$(date +%F).db
```

---

## 7 - Updating the service

The setup script is idempotent and the fastest way to pull `origin/main`:

```bash
sudo bash /opt/roadguard/repo/deploy/setup.sh
sudo systemctl restart roadguard.service
```

The env file and the dataset are left alone across upgrades.

---

## 8 - Running tests on the VM

```bash
cd /opt/roadguard/repo
sudo -u roadguard ./venv/bin/pytest -q
```

---

## 9 - Enabling the LLM panel (optional)

By default the Decision node runs the deterministic rule tree. To turn
on the 3-vendor panel (Groq + OpenAI-compatible provider x2), edit
`/etc/roadguard/env`:

```bash
ROADGUARD_USE_LIVE=1
AGENT_LLM_PANEL=groq:openai/gpt-oss-120b,openai_compat:meta-llama/Llama-3.3-70B-Instruct-Turbo,openai_compat:deepseek/deepseek-chat
OPENAI_COMPAT_BASE_URL=https://api.together.xyz/v1
OPENAI_COMPAT_API_KEY=...
```

Then:

```bash
sudo systemctl restart roadguard.service
sudo bash /opt/roadguard/repo/deploy/verify-vm.sh --passphrase "$MY_PASSPHRASE"
```

The `verify-vm.sh` output will show `panel is configured` in the env
sanity section.

---

## 10 - Approve a hazard from the terminal (no browser)

Pending approvals carry a public link `/approvals/{id}`. If you need
to decide one from the shell:

```bash
# List pending ids
sudo sqlite3 /var/lib/roadguard/roadguard.db \
    "SELECT id FROM approvals WHERE status='pending';"

# Approve one (replace APR_ID)
curl -fsS -X POST \
    -d "decision=approved" \
    http://127.0.0.1:8000/approvals/APR_ID/decide
```

---

## 11 - Shutdown and cleanup (do not run casually)

```bash
sudo systemctl stop roadguard.service \
    roadguard-image-cleanup.timer \
    roadguard-daily-email.timer \
    cloudflared.service
```

Full uninstall is documented in [README.md](README.md) under
*Uninstall*.
