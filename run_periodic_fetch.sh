#!/usr/bin/env bash
# ==============================================================================
# OpenSky Flight Data Fetcher Periodic Wrapper
# ==============================================================================
# Runs fetch_flight_data.py periodically (every 15 minutes by default).
#
# Usage:
#   ./run_periodic_fetch.sh          # Runs continuously in background loop
#   ./run_periodic_fetch.sh --once   # Runs once and exits (ideal for cron)
#
# Crontab setup example (to run every 15 minutes via cron instead of loop):
#   */15 * * * * /path/to/flight_heatmap/run_periodic_fetch.sh --once >> /path/to/flight_heatmap/cron_fetch.log 2>&1
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FETCHER_SCRIPT="${SCRIPT_DIR}/fetch_flight_data.py"
INTERVAL_SEC=900  # 15 minutes

# Dynamically locate Python: respect PYTHON_EXEC / VIRTUAL_ENV / local venv / system python3
if [ -n "${PYTHON_EXEC:-}" ] && [ -x "${PYTHON_EXEC}" ]; then
    : # Use specified PYTHON_EXEC
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_EXEC="${VIRTUAL_ENV}/bin/python"
elif [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON_EXEC="${SCRIPT_DIR}/.venv/bin/python"
elif [ -x "${SCRIPT_DIR}/venv/bin/python" ]; then
    PYTHON_EXEC="${SCRIPT_DIR}/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXEC="$(command -v python)"
else
    echo "❌ Error: Python interpreter not found." >&2
    exit 1
fi

if [ "$1" == "--once" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] 🚀 Running single OpenSky fetch cycle..."
    shift
    "${PYTHON_EXEC}" "${FETCHER_SCRIPT}" "$@"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ✅ Completed single fetch cycle."
    exit 0
fi

echo "=============================================================================="
echo "🛫 OpenSky Periodic Fetcher Daemon Started"
echo "📍 Directory : ${SCRIPT_DIR}"
echo "⏱️ Interval  : Every $((INTERVAL_SEC / 60)) minutes (${INTERVAL_SEC}s)"
echo "🐍 Python    : ${PYTHON_EXEC}"
echo "=============================================================================="

while true; do
    echo ""
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] 🔄 Starting 15-minute fetch cycle..."
    
    # Run fetcher, capturing status
    if "${PYTHON_EXEC}" "${FETCHER_SCRIPT}" "$@"; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ✅ Cycle finished successfully."
    else
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ⚠️ Cycle finished with errors. Will retry in $((INTERVAL_SEC / 60)) min."
    fi
    
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] 💤 Sleeping for $((INTERVAL_SEC / 60)) minutes until next fetch..."
    sleep "${INTERVAL_SEC}"
done
