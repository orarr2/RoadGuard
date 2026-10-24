#!/usr/bin/env bash
# RoadGuard VM provisioning script.
#
# Target: Google Cloud e2-micro Free Tier running Debian 13 (bookworm or
# trixie). Idempotent - rerunning will update the clone, refresh deps,
# and keep the env file intact.
#
# Usage: sudo bash setup.sh

set -euo pipefail

INSTALL_DIR="/opt/roadguard"
VENV_DIR="$INSTALL_DIR/venv"
REPO_DIR="$INSTALL_DIR/repo"
STATE_DIR="/var/lib/roadguard"
ETC_DIR="/etc/roadguard"
ENV_FILE="$ETC_DIR/env"
SERVICE_USER="roadguard"
REPO_URL="${REPO_URL:?set REPO_URL to the git URL of this repository}"
REPO_BRANCH="${REPO_BRANCH:-main}"

log() { printf '==> %s\n' "$*"; }

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    echo "setup.sh must run as root: sudo bash setup.sh" >&2
    exit 1
fi

log "apt-get update + install prerequisites"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    git curl ca-certificates sqlite3 \
    python3 python3-venv python3-pip python3-dev \
    build-essential cmake pkg-config ninja-build

log "ensure $SERVICE_USER system user"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi

log "create install, state, and etc directories"
install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_USER" \
    "$INSTALL_DIR" "$STATE_DIR" "$STATE_DIR/reports" "$STATE_DIR/backups"
install -d -m 0750 -o root -g root "$ETC_DIR"

log "clone or update the repo at $REPO_DIR"
if [ -d "$REPO_DIR/.git" ]; then
    sudo -u "$SERVICE_USER" git -C "$REPO_DIR" fetch --prune origin
    sudo -u "$SERVICE_USER" git -C "$REPO_DIR" reset --hard "origin/$REPO_BRANCH"
else
    sudo -u "$SERVICE_USER" git clone --depth 1 --branch "$REPO_BRANCH" \
        "$REPO_URL" "$REPO_DIR"
fi

log "create venv at $VENV_DIR"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    sudo -u "$SERVICE_USER" python3 -m venv "$VENV_DIR"
fi

log "install Python requirements"
VM_REQS="$REPO_DIR/requirements-vm.txt"
if [ ! -f "$VM_REQS" ]; then
    VM_REQS="$REPO_DIR/requirements.txt"
fi
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --upgrade pip wheel
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install -r "$VM_REQS"

log "ensure $ENV_FILE exists"
if [ ! -f "$ENV_FILE" ]; then
    cp "$REPO_DIR/.env.example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    chown root:root "$ENV_FILE"
    echo "  Created $ENV_FILE from .env.example. Edit it before starting the service."
fi

log "generate JWT secret if not set"
if grep -q '^ROADGUARD_JWT_SECRET=replace_me$' "$ENV_FILE"; then
    NEW_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
    sed -i "s|^ROADGUARD_JWT_SECRET=replace_me$|ROADGUARD_JWT_SECRET=${NEW_SECRET}|" "$ENV_FILE"
    echo "  Generated a fresh ROADGUARD_JWT_SECRET."
fi

log "install systemd units"
install -m 0644 "$REPO_DIR/deploy/roadguard.service" /etc/systemd/system/roadguard.service
install -m 0644 "$REPO_DIR/deploy/roadguard-daily-report.service" /etc/systemd/system/roadguard-daily-report.service
install -m 0644 "$REPO_DIR/deploy/roadguard-daily-report.timer" /etc/systemd/system/roadguard-daily-report.timer
# Stop the retired live tracker first: systemd refuses to disable a unit
# whose file is already gone, and the process would run until reboot.
systemctl disable --now roadguard-agent-live.service >/dev/null 2>&1 || true
# Remove retired units if they exist from an older install.
rm -f /etc/systemd/system/roadguard-image-cleanup.service \
      /etc/systemd/system/roadguard-image-cleanup.timer \
      /etc/systemd/system/roadguard-daily-email.service \
      /etc/systemd/system/roadguard-daily-email.timer \
      /etc/systemd/system/roadguard-agent-live.service
systemctl daemon-reload
systemctl disable --now roadguard-image-cleanup.timer >/dev/null 2>&1 || true
systemctl disable --now roadguard-daily-email.timer >/dev/null 2>&1 || true
systemctl enable roadguard.service >/dev/null
systemctl enable roadguard-daily-report.timer >/dev/null

cat <<'NEXT'

Setup complete.

Next steps:
  1. Edit /etc/roadguard/env with real values:
       - GROQ_API_KEY, GROQ_MODEL
       - ROADGUARD_PASSPHRASE_HASH (bcrypt)
       - ROADGUARD_ALLOWED_EMAILS
       - EMAIL_ADDRESS, EMAIL_APP_PASSWORD, REPORT_RECIPIENT
     Generate the passphrase hash with:
       sudo -u roadguard /opt/roadguard/venv/bin/python -c \
         "import bcrypt, getpass; print(bcrypt.hashpw(getpass.getpass().encode(), bcrypt.gensalt()).decode())"

  2. Seed the RAG index (optional at this stage; docs/corpus-handoff.md section 5).
     HOME matches the systemd units so the embedding-model cache is shared:
       cd /opt/roadguard/repo && sudo -u roadguard env PYTHONPATH=. \
         ROADGUARD_RAG_PATH=/var/lib/roadguard/chroma HOME=/var/lib/roadguard \
         /opt/roadguard/venv/bin/python -c "from roadguard_agent import seed_from_jsonl; print('rows:', seed_from_jsonl('data/mini_corpus.jsonl'))"

  3. Start the service:
       systemctl start roadguard.service

  4. Install cloudflared and the named tunnel (fixed hostname):
       See deploy/cloudflared.md - it also sets ROADGUARD_APPROVAL_BASE_URL.

  5. Commit the hostname to webapp/tunnel-url.json (step 3 in cloudflared.md).
     GitHub Pages serves it; the webapp reads it before config.js.
NEXT
