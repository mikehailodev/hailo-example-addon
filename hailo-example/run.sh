#!/usr/bin/env bash
set -e

SOCK_DIR="/share/hailo"
SOCK_FILE="${SOCK_DIR}/hailort_service.sock"
export HAILORT_SERVICE_ADDRESS="unix:${SOCK_FILE}"

echo "============================================"
echo " Hailo Example App — Blueprint Client"
echo "============================================"
echo "Socket: ${SOCK_FILE}"
echo "HAILORT_SERVICE_ADDRESS=${HAILORT_SERVICE_ADDRESS}"
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
    echo "  ...waiting (${WAITED}s) [ls: $(ls ${SOCK_DIR}/ 2>&1)]"
done

if [ $WAITED -ge 60 ]; then
    echo "WARNING: Socket not found after 60s. Trying anyway..."
    echo "[diag] Socket dir: $(ls -la ${SOCK_DIR}/ 2>&1)"
fi

echo "Running inference demo..."
# Clear cached model if arch changed
rm -f /data/models/yolov6n.hef 2>/dev/null || true
exec python3 /detect.py
