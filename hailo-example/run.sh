#!/usr/bin/env bash
set -e

SOCK_DIR="/share/hailo"
SOCK_FILE="${SOCK_DIR}/hailort_service.sock"
export HAILORT_SERVICE_ADDRESS="unix:${SOCK_FILE}"

# Read options from HA
OPTIONS_FILE="/data/options.json"
if [ -f "${OPTIONS_FILE}" ]; then
    export CAMERA_SOURCE=$(python3 -c "import json; print(json.load(open('${OPTIONS_FILE}')).get('camera_source', '/dev/video0'))")
    export CONFIDENCE_THRESHOLD=$(python3 -c "import json; print(json.load(open('${OPTIONS_FILE}')).get('confidence_threshold', 0.5))")
fi

echo "============================================"
echo " Hailo Example App — Real-time Detection"
echo "============================================"
echo "Socket: ${SOCK_FILE}"
echo "HAILORT_SERVICE_ADDRESS=${HAILORT_SERVICE_ADDRESS}"
echo "CAMERA_SOURCE=${CAMERA_SOURCE:-/dev/video0}"
echo "CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.5}"
echo "Waiting for HailoRT service..."

# Wait up to 60s for the service socket to appear
WAITED=0
while [ $WAITED -lt 60 ]; do
    if [ -S "${SOCK_FILE}" ]; then
        echo "Service socket found: ${SOCK_FILE}"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "  ...waiting (${WAITED}s)"
done

if [ $WAITED -ge 60 ]; then
    echo "WARNING: Socket not found after 60s. Trying anyway..."
fi

echo "Starting web server..."
cd /app
exec python3 server.py
