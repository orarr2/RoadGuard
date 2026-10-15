#!/usr/bin/env bash
# RoadGuard VM verification script.
#
# Read-only health check. Nothing in here mutates state; it is safe to
# rerun any time.
#
# Usage (on the VM, as root or the roadguard user):
#   sudo bash /opt/roadguard/repo/deploy/verify-vm.sh
#   sudo bash /opt/roadguard/repo/deploy/verify-vm.sh --passphrase 'my-pass'
#
# Flags / env:
#   --passphrase <s>              or $ROADGUARD_VERIFY_PASSPHRASE
#     Enables the authenticated endpoint tests. Without it the /hazards
#     tests are skipped with a WARN.
#   --api-base http://host:port   or $ROADGUARD_API_BASE
#     Where to reach FastAPI (default http://127.0.0.1:8000).
#   --dataset-dir <path>          or $ROADGUARD_DATASET_DIR
#     Where the 10K dataset lives (default /opt/roadguard/datasets/road-damage-10k).
#
# Exit code:
#   0 - every required check passed (WARNs still allowed)
#   N - number of FAILs (one per failed check)

set -u

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INSTALL_DIR="${INSTALL_DIR:-/opt/roadguard}"
VENV_DIR="${VENV_DIR:-$INSTALL_DIR/venv}"
REPO_DIR="${REPO_DIR:-$INSTALL_DIR/repo}"
STATE_DIR="${STATE_DIR:-/var/lib/roadguard}"
ENV_FILE="${ENV_FILE:-/etc/roadguard/env}"
DB_PATH="${ROADGUARD_DB_PATH:-$STATE_DIR/roadguard.db}"
API_BASE="${ROADGUARD_API_BASE:-http://127.0.0.1:8000}"
DATASET_DIR="${ROADGUARD_DATASET_DIR:-$INSTALL_DIR/datasets/road-damage-10k}"
EXPECTED_IMAGES="${ROADGUARD_DATASET_MIN_IMAGES:-9000}"
PASSPHRASE="${ROADGUARD_VERIFY_PASSPHRASE:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --passphrase)    PASSPHRASE="$2"; shift 2 ;;
        --api-base)      API_BASE="$2"; shift 2 ;;
        --dataset-dir)   DATASET_DIR="$2"; shift 2 ;;
        --min-images)    EXPECTED_IMAGES="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0 ;;
        *)
            echo "verify-vm: unknown arg: $1" >&2
            exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

FAILS=0
WARNS=0
PASSES=0

if [ -t 1 ]; then
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    BOLD=$'\033[1m';   DIM=$'\033[2m';     RESET=$'\033[0m'
else
    GREEN=""; YELLOW=""; RED=""; BOLD=""; DIM=""; RESET=""
fi

header() { printf '\n%s==> %s%s\n' "$BOLD" "$1" "$RESET"; }
pass()   { printf '  %sPASS%s  %s\n' "$GREEN"  "$RESET" "$1"; PASSES=$((PASSES+1)); }
warn()   { printf '  %sWARN%s  %s\n' "$YELLOW" "$RESET" "$1"; WARNS=$((WARNS+1)); }
fail()   { printf '  %sFAIL%s  %s\n' "$RED"    "$RESET" "$1"; FAILS=$((FAILS+1)); }
info()   { printf '  %s%s%s\n' "$DIM" "$1" "$RESET"; }

# ---------------------------------------------------------------------------
# Dependencies for this script itself
# ---------------------------------------------------------------------------

need() {
    command -v "$1" >/dev/null 2>&1 || {
        fail "missing command: $1 (install it before running verify-vm)"
        exit 3
    }
}
need curl
need python3
need systemctl
need df

# ---------------------------------------------------------------------------
# 1. systemd services
# ---------------------------------------------------------------------------

header "systemd services"

check_unit() {
    local unit="$1" required="$2"
    local state
    state=$(systemctl is-active "$unit" 2>/dev/null || true)
    if [ "$state" = "active" ]; then
        pass "$unit is active"
    elif [ "$required" = "optional" ]; then
        warn "$unit is $state (optional)"
    else
        fail "$unit is $state"
    fi
}

check_unit roadguard.service required
check_unit roadguard-image-cleanup.timer required
check_unit roadguard-daily-email.timer required
check_unit cloudflared.service optional

# ---------------------------------------------------------------------------
# 2. GET /health
# ---------------------------------------------------------------------------

header "GET $API_BASE/health"

HEALTH_JSON="$(curl -fsS --max-time 5 "$API_BASE/health" 2>/dev/null || true)"

if [ -z "$HEALTH_JSON" ]; then
    fail "cannot reach $API_BASE/health - is roadguard.service listening?"
else
    get() { echo "$HEALTH_JSON" | python3 -I -c 'import json,sys;print(json.load(sys.stdin).get(sys.argv[1],""))' "$1"; }
    status=$(get status)
    [ "$status" = "ok" ] && pass "status=ok" || warn "status=$status"
    for key in agent_ready db_ready; do
        val=$(get "$key")
        if [ "$val" = "True" ] || [ "$val" = "true" ]; then
            pass "$key=true"
        else
            fail "$key=$val"
        fi
    done
    for key in rag_ready supabase_ready llm_fallback_ready; do
        val=$(get "$key")
        if [ "$val" = "True" ] || [ "$val" = "true" ]; then
            pass "$key=true"
        else
            warn "$key=$val (optional subsystem)"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 3. Authenticated POST /hazards/submit and /hazards/live
# ---------------------------------------------------------------------------

header "authenticated endpoints"

if [ -z "$PASSPHRASE" ]; then
    warn "no --passphrase and no \$ROADGUARD_VERIFY_PASSPHRASE; skipping /hazards tests"
    JWT=""
else
    LOGIN_JSON="$(curl -fsS --max-time 5 \
        -H 'Content-Type: application/json' \
        -d "{\"passphrase\":\"$PASSPHRASE\"}" \
        "$API_BASE/auth/login" 2>/dev/null || true)"
    JWT="$(echo "$LOGIN_JSON" | python3 -I -c 'import json,sys
try: print(json.load(sys.stdin).get("token",""))
except Exception: print("")' )"
    if [ -z "$JWT" ]; then
        fail "login failed (passphrase rejected or /auth/login unreachable)"
    else
        pass "/auth/login returned a JWT"
    fi
fi

post_json() {
    local path="$1" body="$2"
    curl -fsS --max-time 15 \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $JWT" \
        -d "$body" \
        "$API_BASE$path" 2>/dev/null || true
}

HZ_SUBMIT_PAYLOAD=$(cat <<'JSON'
{
    "class": "D40",
    "confidence": 0.92,
    "bboxes": [[120, 240, 340, 460, "D40", 0.92]],
    "road_type": "main",
    "lane_blocked": "partial",
    "location": "Allenby 42 verify-vm",
    "image_object_key": "verify-vm/placeholder.jpg",
    "top1_margin": 0.4,
    "area_ratio": 0.08,
    "center_y_rel": 0.72,
    "image_quality": {"blur_score": 180, "is_road_scene": true}
}
JSON
)

HZ_LIVE_PAYLOAD=$(cat <<'JSON'
{
    "session_id": "verify-vm-session",
    "track_id": "verify-vm-track-1",
    "class": "D40",
    "confidence": 0.92,
    "top1_margin": 0.4,
    "area_ratio": 0.08,
    "bbox": [120, 240, 340, 460],
    "center_y_rel": 0.72,
    "road_type": "main",
    "lane_blocked": "partial",
    "location": "Allenby 42 verify-vm live",
    "image_quality": {"blur_score": 180, "is_road_scene": true}
}
JSON
)

LAST_ACTION=""
check_action_json() {
    # Reports pass/fail for one hazard endpoint and sets $LAST_ACTION
    # so the caller can test the verdict without capturing stdout
    # (which would swallow the pretty-printed output).
    local label="$1" body="$2"
    LAST_ACTION="$(printf '%s' "$body" | python3 -I -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
v = d.get("verdict") or {}
print(v.get("action", d.get("action", "")))
')"
    case "$LAST_ACTION" in
        urgent|standard|needs_more_evidence|human_review)
            pass "$label -> action=$LAST_ACTION" ;;
        "")
            fail "$label returned no verdict" ;;
        *)
            fail "$label returned unknown action: $LAST_ACTION"
            LAST_ACTION="" ;;
    esac
}

SUBMIT_ACTION=""
LIVE_ACTION=""
if [ -n "$JWT" ]; then
    SUBMIT_RESP="$(post_json /hazards/submit "$HZ_SUBMIT_PAYLOAD")"
    if [ -z "$SUBMIT_RESP" ]; then
        fail "POST /hazards/submit returned empty body"
    else
        check_action_json 'POST /hazards/submit' "$SUBMIT_RESP"
        SUBMIT_ACTION="$LAST_ACTION"
    fi
    LIVE_RESP="$(post_json /hazards/live "$HZ_LIVE_PAYLOAD")"
    if [ -z "$LIVE_RESP" ]; then
        fail "POST /hazards/live returned empty body"
    else
        check_action_json 'POST /hazards/live' "$LIVE_RESP"
        LIVE_ACTION="$LAST_ACTION"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Approval row for urgent/human_review verdicts
# ---------------------------------------------------------------------------

header "approvals table"

if [ ! -r "$DB_PATH" ]; then
    warn "cannot read $DB_PATH (is the DB on a different path? set ROADGUARD_DB_PATH)"
else
    if ! command -v sqlite3 >/dev/null 2>&1; then
        warn "sqlite3 CLI not installed; skipping approvals-table checks"
    else
        TOTAL_APPROVALS=$(sqlite3 "$DB_PATH" 'SELECT COUNT(*) FROM approvals;' 2>/dev/null || echo "")
        if [ -z "$TOTAL_APPROVALS" ]; then
            fail "approvals table missing - run the API once so the schema is applied"
        else
            info "approvals table: $TOTAL_APPROVALS total row(s)"
            if [ -n "$JWT" ] && { [ "$SUBMIT_ACTION" = "urgent" ] || [ "$SUBMIT_ACTION" = "human_review" ] \
                              || [ "$LIVE_ACTION"   = "urgent" ] || [ "$LIVE_ACTION"   = "human_review" ]; }; then
                NEW_PENDING=$(sqlite3 "$DB_PATH" \
                    "SELECT COUNT(*) FROM approvals WHERE created_at > strftime('%s','now','-5 minutes');" 2>/dev/null || echo 0)
                if [ "${NEW_PENDING:-0}" -ge 1 ]; then
                    pass "at least one fresh approval row written in the last 5 min"
                else
                    fail "verdict was $SUBMIT_ACTION/$LIVE_ACTION but no approval row was written"
                fi
            else
                info "no urgent/human_review verdict this run; approvals row check skipped"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 5. 10K image dataset
# ---------------------------------------------------------------------------

header "road-damage-10k dataset"

if [ ! -d "$DATASET_DIR" ]; then
    fail "dataset dir not found: $DATASET_DIR (run tools/download_dataset.py)"
else
    pass "dataset dir exists: $DATASET_DIR"
    MANIFEST="$DATASET_DIR/manifest.json"
    DATA_YAML="$DATASET_DIR/data.yaml"
    if [ -f "$MANIFEST" ]; then
        PROJECT=$(python3 -I -c "import json;print(json.load(open('$MANIFEST')).get('project',''))" 2>/dev/null || echo "")
        NC=$(python3 -I -c "import json;print(json.load(open('$MANIFEST')).get('nc',0))" 2>/dev/null || echo 0)
        info "manifest project=$PROJECT nc=$NC"
    else
        warn "no manifest.json at $MANIFEST (download may be partial)"
    fi
    [ -f "$DATA_YAML" ] && pass "data.yaml present" || warn "no data.yaml at $DATA_YAML"

    # Count images recursively (jpg/jpeg/png). Fast and avoids `find` quoting.
    IMAGE_COUNT=$(find "$DATASET_DIR" -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
    if [ "$IMAGE_COUNT" -ge "$EXPECTED_IMAGES" ]; then
        pass "found $IMAGE_COUNT images (>= $EXPECTED_IMAGES expected)"
    elif [ "$IMAGE_COUNT" -gt 0 ]; then
        warn "only $IMAGE_COUNT images found (expected around 10K); download may be partial"
    else
        fail "no images found under $DATASET_DIR"
    fi
    DATASET_SIZE=$(du -sh "$DATASET_DIR" 2>/dev/null | awk '{print $1}')
    info "dataset size: ${DATASET_SIZE:-?}"
fi

# ---------------------------------------------------------------------------
# 6. Disk usage
# ---------------------------------------------------------------------------

header "disk usage"

for p in "$INSTALL_DIR" "$STATE_DIR" "$DATASET_DIR"; do
    if [ -d "$p" ]; then
        line=$(df -h "$p" 2>/dev/null | tail -1)
        use=$(echo "$line" | awk '{print $5}' | tr -d '%')
        avail=$(echo "$line" | awk '{print $4}')
        info "$(printf '%-45s  used=%s%%  avail=%s' "$p" "${use:-?}" "${avail:-?}")"
        if [ -n "${use:-}" ] && [ "$use" -ge 95 ] 2>/dev/null; then
            fail "disk at $p is ${use}% full"
        elif [ -n "${use:-}" ] && [ "$use" -ge 85 ] 2>/dev/null; then
            warn "disk at $p is ${use}% full"
        fi
    fi
done

# ---------------------------------------------------------------------------
# 7. Environment sanity
# ---------------------------------------------------------------------------

header "/etc/roadguard/env sanity"

if [ ! -r "$ENV_FILE" ]; then
    warn "cannot read $ENV_FILE (verify as root to inspect)"
else
    while IFS= read -r key; do
        if grep -qE "^${key}=(replace_me|)$" "$ENV_FILE"; then
            fail "$key is unset in $ENV_FILE"
        else
            pass "$key is set"
        fi
    done <<'KEYS'
GROQ_API_KEY
ROADGUARD_PASSPHRASE_HASH
ROADGUARD_JWT_SECRET
KEYS
    if grep -qE '^AGENT_LLM_PANEL=' "$ENV_FILE"; then
        info "panel is configured ($(grep '^AGENT_LLM_PANEL=' "$ENV_FILE" | head -1))"
    else
        info "AGENT_LLM_PANEL not set (Decision runs single provider)"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

printf '\n%s==> summary%s  %sPASS=%d%s  %sWARN=%d%s  %sFAIL=%d%s\n' \
    "$BOLD" "$RESET" \
    "$GREEN" "$PASSES" "$RESET" \
    "$YELLOW" "$WARNS" "$RESET" \
    "$RED" "$FAILS" "$RESET"

exit "$FAILS"
