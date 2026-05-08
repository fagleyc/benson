#!/usr/bin/env bash
# ha_watchdog.sh — check Home Assistant's API and restart it if wedged.
#
# HA's listening socket can stay open while the application loop has
# stalled (we hit this on 2026-05-08 after HA had been up 11+ days —
# recv-q backlog of 129 connections, systemctl is-active=true, but the
# /api/ endpoint timed out).
#
# Run by the benson-ha-watchdog.timer (every 30 min). Idempotent +
# self-rate-limited: won't restart HA twice within an hour, even if
# the second check still fails.

set -uo pipefail

HA_URL="http://localhost:8123/api/"
ENV_FILE="/etc/benson/env"
LAST_RESTART_FILE="/run/benson-ha-watchdog.last_restart"
RATE_LIMIT_SEC=3600   # don't restart again within 1 hour
PROBE_TIMEOUT=10
SETTLE_WAIT=20

# Pull HA token from /etc/benson/env (root-readable). The service runs
# as root so we can read it directly.
TOKEN=$(grep -E '^HA_LONG_LIVED_TOKEN=' "$ENV_FILE" 2>/dev/null | cut -d= -f2-)
if [[ -z "$TOKEN" ]]; then
    echo "ha_watchdog: no HA_LONG_LIVED_TOKEN in $ENV_FILE — skipping"
    exit 0
fi

# Probe the API. Non-2xx (incl. timeout) is a wedge.
http_code=$(curl --max-time "$PROBE_TIMEOUT" -s -o /dev/null \
    -H "Authorization: Bearer $TOKEN" -w '%{http_code}' "$HA_URL" || echo "000")

if [[ "$http_code" =~ ^2 ]]; then
    echo "ha_watchdog: OK ($http_code)"
    exit 0
fi

echo "ha_watchdog: HA unhealthy — http_code=$http_code"

# Rate limit: don't restart again within RATE_LIMIT_SEC.
if [[ -f "$LAST_RESTART_FILE" ]]; then
    last=$(cat "$LAST_RESTART_FILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    if (( now - last < RATE_LIMIT_SEC )); then
        echo "ha_watchdog: skipping restart — last was $((now - last))s ago (< $RATE_LIMIT_SEC)"
        exit 0
    fi
fi

# Capture diagnostics before the restart so we have a record.
recvq=$(ss -ltn '( sport = :8123 )' 2>/dev/null | awk 'NR==2 {print $2}')
uptime=$(systemctl show homeassistant.service -p ActiveEnterTimestamp --value 2>/dev/null)
echo "ha_watchdog: pre-restart  recv-q=$recvq  ha_up_since=$uptime"

systemctl restart homeassistant.service
date +%s > "$LAST_RESTART_FILE"
echo "ha_watchdog: restart issued, sleeping ${SETTLE_WAIT}s for HA to settle"

sleep "$SETTLE_WAIT"

# Verify recovery. Don't loop — one shot, log result.
http_code=$(curl --max-time "$PROBE_TIMEOUT" -s -o /dev/null \
    -H "Authorization: Bearer $TOKEN" -w '%{http_code}' "$HA_URL" || echo "000")
if [[ "$http_code" =~ ^2 ]]; then
    echo "ha_watchdog: HA recovered ($http_code)"
    exit 0
fi
echo "ha_watchdog: HA still unhealthy after restart (http_code=$http_code)"
exit 1
