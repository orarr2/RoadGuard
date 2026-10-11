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
MODELS_DIR="$INSTALL_DIR/models"
REPO_DIR="$INSTALL_DIR/repo"
STATE_DIR="/var/lib/roadguard"
ETC_DIR="/etc/roadguard"
ENV_FILE="$ETC_DIR/env"
SERVICE_USER="roadguard"
REPO_URL="${REPO_URL:-https://github.com/orarr2/RoadGuard.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
LLAMA_GGUF_URL="${LLAMA_GGUF_URL:-https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
LLAMA_GGUF_PATH="$MODELS_DIR/llama-3.2-1b-instruct-q4_k_m.gguf"

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
    "$INSTALL_DIR" "$MODELS_DIR" "$STATE_DIR"
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

log "install Python requirements (lean VM set)"
# requirements-vm.txt is the subset the VM actually needs. It omits
# chromadb (3.13 wheels unavailable, source build needs >1 GB tmp) and
# llama-cpp-python (source build is ~15 min and fills /tmp on e2-micro).
# Both are lazy-imported from the agent code and gracefully stubbed, so
# the service runs without them. Install them manually later when disk
# headroom and Benny's RAG index are ready.
VM_REQS="$REPO_DIR/requirements-vm.txt"
if [ ! -f "$VM_REQS" ]; then
    VM_REQS="$REPO_DIR/requirements.txt"
fi
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --upgrade pip wheel
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install -r "$VM_REQS"

log "download Llama 3.2 1B gguf (fallback LLM, ~800 MB) [optional]"
if [ "${ROADGUARD_SKIP_LLAMA_DOWNLOAD:-0}" = "1" ]; then
    echo "  Skipping gguf download (ROADGUARD_SKIP_LLAMA_DOWNLOAD=1)."
elif [ ! -s "$LLAMA_GGUF_PATH" ]; then
    # The fallback LLM is optional; if llama-cpp-python is not installed
    # the service still runs and simply returns human_review when Groq
    # is down. Try the download but do not fail the whole script if the
    # VM is tight on disk.
    sudo -u "$SERVICE_USER" curl -fL -o "$LLAMA_GGUF_PATH" "$LLAMA_GGUF_URL" || {
        echo "  Llama gguf download failed; continuing without it."
        rm -f "$LLAMA_GGUF_PATH"
    }
fi

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
install -m 0644 "$REPO_DIR/deploy/roadguard-image-cleanup.service" /etc/systemd/system/roadguard-image-cleanup.service
install -m 0644 "$REPO_DIR/deploy/roadguard-image-cleanup.timer" /etc/systemd/system/roadguard-image-cleanup.timer
install -m 0644 "$REPO_DIR/deploy/roadguard-daily-email.service" /etc/systemd/system/roadguard-daily-email.service
install -m 0644 "$REPO_DIR/deploy/roadguard-daily-email.timer" /etc/systemd/system/roadguard-daily-email.timer
systemctl daemon-reload
systemctl enable roadguard.service >/dev/null
systemctl enable roadguard-image-cleanup.timer >/dev/null
systemctl enable roadguard-daily-email.timer >/dev/null

cat <<'NEXT'

Setup complete.

Next steps:
  1. Edit /etc/roadguard/env with real values:
       - GROQ_API_KEY, GROQ_MODEL
       - SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_BUCKET
       - ROADGUARD_PASSPHRASE_HASH (bcrypt), GOOGLE_OAUTH_CLIENT_ID
       - ROADGUARD_ALLOWED_EMAILS
       - EMAIL_ADDRESS, EMAIL_APP_PASSWORD, REPORT_RECIPIENT
     Generate the passphrase hash with:
       sudo -u roadguard /opt/roadguard/venv/bin/python -c \
         "import bcrypt, getpass; print(bcrypt.hashpw(getpass.getpass().encode(), bcrypt.gensalt()).decode())"

  2. Build the RAG index (optional at this stage):
       sudo -u roadguard /opt/roadguard/venv/bin/python \
         -m tools.build_rag_index

  3. Start the services:
       systemctl start roadguard.service
       systemctl start roadguard-image-cleanup.timer
       systemctl start roadguard-daily-email.timer

  4. Install cloudflared and start a quick tunnel:
       See deploy/cloudflared.md

  5. Update webapp/assets/config.js on GitHub Pages with the tunnel URL.
NEXT
