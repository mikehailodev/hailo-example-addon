#!/usr/bin/env bash
set -e

SOCK_DIR="/share/hailo"
export HAILO_SOCK_PATH="${SOCK_DIR}"

echo "============================================"
echo " Hailo Example App — Blueprint Client"
echo "============================================"
echo "Socket dir: ${SOCK_DIR}"
echo "Waiting for HailoRT service..."

# Wait up to 60s for the service socket to appear
WAITED=0
while [ $WAITED -lt 60 ]; do
    if ls "${SOCK_DIR}"/*.sock >/dev/null 2>&1 || \
       ls "${SOCK_DIR}"/hailort* >/dev/null 2>&1; then
        echo "Service socket found."
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "  ...waiting (${WAITED}s)"
done

if [ $WAITED -ge 60 ]; then
    echo "WARNING: No socket found after 60s. Trying anyway..."
fi

echo "Running inference demo..."
exec python3 /detect.py
