#!/bin/bash
# Bring up a local SandboxFusion Docker container for Code RL grading.
# Idempotent: if a container named "$CONTAINER_NAME" is already running, exit
# success; if it exists but is stopped, start it; otherwise pull the image and
# `docker run` a fresh detached container.
#
# Usage:
#   bash tools/sandbox_up.sh                 # start (or re-use) the sandbox
#   bash tools/sandbox_up.sh --status        # print status of the container
#   bash tools/sandbox_up.sh --logs          # tail last 50 log lines
#   bash tools/sandbox_up.sh --stop          # stop the container
#   bash tools/sandbox_up.sh --remove        # stop + delete the container
#
# Environment overrides:
#   CONTAINER_NAME  (default: sandbox-fusion)
#   SANDBOX_PORT    (default: 8080)         host port to expose
#   SANDBOX_IMAGE   (default: volcengine/sandbox-fusion:server-20250609)
#   WAIT_TIMEOUT    (default: 60)           seconds to wait for /run_code
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-sandbox-fusion}"
SANDBOX_PORT="${SANDBOX_PORT:-8080}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-volcengine/sandbox-fusion:server-20250609}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
ENDPOINT="http://localhost:${SANDBOX_PORT}/run_code"

status() {
    docker ps -a --filter "name=^${CONTAINER_NAME}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
}

case "${1:-}" in
    --status)
        status
        exit 0
        ;;
    --logs)
        docker logs --tail 50 "$CONTAINER_NAME"
        exit 0
        ;;
    --stop)
        docker stop "$CONTAINER_NAME" >/dev/null && echo "Stopped $CONTAINER_NAME"
        exit 0
        ;;
    --remove)
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" >/dev/null && echo "Removed $CONTAINER_NAME"
        exit 0
        ;;
esac

# Precondition: docker CLI reachable.
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: 'docker' not on PATH. Install Docker or add it to PATH." >&2
    exit 1
fi

# If the container is already running, we're done.
running_id=$(docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.ID}}')
if [ -n "$running_id" ]; then
    echo "Sandbox already running: $CONTAINER_NAME ($running_id)"
    echo "Endpoint: $ENDPOINT"
    exit 0
fi

# If the container exists but is stopped, start it.
existing_id=$(docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.ID}}')
if [ -n "$existing_id" ]; then
    echo "Starting existing container: $CONTAINER_NAME ($existing_id)"
    docker start "$CONTAINER_NAME" >/dev/null
else
    # Warn if the host port is already bound by something else.
    if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${SANDBOX_PORT}" | tail -n +2 | grep -q .; then
        echo "WARNING: host port ${SANDBOX_PORT} appears to be in use." >&2
        echo "         Set SANDBOX_PORT=<other> or free the port before continuing." >&2
    fi

    echo "Pulling ${SANDBOX_IMAGE}..."
    docker pull "$SANDBOX_IMAGE" >/dev/null

    echo "Starting sandbox container: $CONTAINER_NAME on port $SANDBOX_PORT"
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "${SANDBOX_PORT}:8080" \
        "$SANDBOX_IMAGE" >/dev/null
fi

# Wait for the endpoint to become reachable. The server accepts POSTs; any
# HTTP response (even 4xx on a GET) means the process is up.
echo "Waiting up to ${WAIT_TIMEOUT}s for $ENDPOINT to respond..."
deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
while :; do
    if curl -s -o /dev/null -w '%{http_code}' -m 2 "$ENDPOINT" 2>/dev/null | grep -qE '^(2|3|4)'; then
        echo "Sandbox is up: $ENDPOINT"
        echo "export SANDBOX_URL=$ENDPOINT"
        exit 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "ERROR: sandbox did not become reachable within ${WAIT_TIMEOUT}s." >&2
        echo "Try:  docker logs $CONTAINER_NAME" >&2
        exit 1
    fi
    sleep 2
done
