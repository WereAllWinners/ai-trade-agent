#!/usr/bin/env bash
# verify_daemon_liveness.sh — confirm /health stays healthy through real work.
#
# Why this exists rather than "the tests pass":
#
# The daemons' heartbeat broke twice, and a green suite missed it both times.
# First `datetime.now(datetime.UTC)` raised inside a bare except, freezing every
# heartbeat for six months while the bots traded normally. Then the fix for the
# staleness threshold chunked the daemons' idle sleeps — which passed its unit
# tests, because those tests mocked time.sleep and therefore asserted the
# mechanism rather than the outcome. In production it still failed: /health read
# "down" at 443s because the daemon was not sleeping at all, it was WORKING.
# A trading session holds the main loop for many minutes (one symbol's LLM
# inference takes ~2 minutes and a session walks dozens of symbols).
#
# So the only verification that means anything for daemon liveness is one that
# spans the failure condition: sample /health repeatedly WHILE a trading session
# is running. This script does that, and — importantly — fails if no session was
# actually active in the window, because an idle daemon passes trivially and
# would give false confidence.
#
# Run it after any change to the heartbeat, the daemon main loop, or the
# health-server thresholds. Best run during market hours, when sessions fire.
#
# Usage:
#   scripts/tools/verify_daemon_liveness.sh            # 13-minute watch
#   scripts/tools/verify_daemon_liveness.sh 1800       # custom duration (seconds)
#
# Exit 0 = healthy throughout a genuinely active window. Non-zero = investigate.
set -uo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

DURATION="${1:-780}"
INTERVAL="${HEALTH_SAMPLE_INTERVAL:-45}"
PORT="${HEALTH_PORT:-8765}"
URL="http://localhost:${PORT}/health"

command -v curl >/dev/null || { echo "curl not found" >&2; exit 2; }

START=$(date +%s)
MAX_AGE=0
BAD=0
SAMPLES=0

echo "Watching ${URL} for ${DURATION}s (sampling every ${INTERVAL}s)…"

while [ $(( $(date +%s) - START )) -lt "${DURATION}" ]; do
    PAYLOAD="$(curl -s --max-time 10 "${URL}" 2>/dev/null)"
    PARSED="$(printf '%s' "${PAYLOAD}" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('unreachable -1'); raise SystemExit
# Judge only daemons the health server itself judged: a live bot that was never
# started reports 'missing' by design and must not fail this check.
ages = [v['age_seconds'] for v in d['daemons'].values()
        if v.get('age_seconds') is not None]
print(d.get('status', '?'), max(ages) if ages else -1)
" 2>/dev/null)"

    STATUS="${PARSED%% *}"
    AGE="${PARSED##* }"
    SAMPLES=$((SAMPLES + 1))

    if awk -v a="${AGE}" -v m="${MAX_AGE}" 'BEGIN{exit !(a>m)}'; then MAX_AGE="${AGE}"; fi
    if [ "${STATUS}" != "ok" ]; then
        BAD=$((BAD + 1))
        echo "  !! $(date +%H:%M:%S)  status=${STATUS}  max_age=${AGE}s"
    fi
    sleep "${INTERVAL}"
done

# Was the daemon actually busy? Without this the whole check is vacuous.
WINDOW_MIN=$(( DURATION / 60 + 1 ))
SESSION_LINES=$(journalctl -u ai-trading-bot.service --since "${WINDOW_MIN} min ago" \
    --no-pager 2>/dev/null | grep -cE "Analyzing|STARTING TRADING SESSION" || true)

echo
echo "=== ${SAMPLES} samples over ${DURATION}s ==="
echo "  max heartbeat age  : ${MAX_AGE}s"
echo "  non-ok samples     : ${BAD}"
echo "  session activity   : ${SESSION_LINES} log lines"

if [ "${BAD}" -ne 0 ]; then
    echo "FAIL: /health reported unhealthy during the window." >&2
    exit 1
fi
if [ "${SESSION_LINES}" -eq 0 ]; then
    echo "INCONCLUSIVE: no trading session ran in the window — an idle daemon" >&2
    echo "  passes trivially. Re-run during market hours." >&2
    exit 3
fi
echo "PASS: healthy throughout an active window."
