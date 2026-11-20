#!/usr/bin/env bash
# Stop the demo listener and release the agent-guard lock.
set -euo pipefail
PIDFILE=${DEMO_PIDFILE:-/tmp/opencode/nq-demo.pid}
pkill -f "nq-serve --model .*qwen2.5-0.5b" 2>/dev/null || true
if [ -r "$PIDFILE" ]; then
  GUARD_PID=$(cat "$PIDFILE")
  kill "$GUARD_PID" 2>/dev/null || true
  rm -f "$PIDFILE"
fi
sleep 1
if curl -sf "http://127.0.0.1:${DEMO_PORT:-8080}/health" >/dev/null 2>&1; then
  echo "still answering on ${DEMO_PORT:-8080}; check pgrep -af nq-serve" >&2
  exit 1
fi
echo "stopped"
