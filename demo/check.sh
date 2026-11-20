#!/usr/bin/env bash
# Pre-flight for the demo: is the box quiet, is there room, is the model there,
# and is the listener bound to loopback only?
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${DEMO_MODEL:-/home/lucas/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf}
PORT=${DEMO_PORT:-8080}
fail=0

echo "== time and load"
date -Is
uptime
cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')
echo "cpus          $cpus logical; demo pool ${DEMO_THREADS:-4} (demo/start.sh)"
echo "power         $(powerprofilesctl get 2>/dev/null || echo '?') (quote one arm only; see README)"

echo
echo "== memory (want MemAvailable above 2,500 MiB; swap near zero)"
free -m

echo
echo "== largest processes"
ps -eo pid,rss,etime,comm --sort=-rss | head -6

echo
echo "== model"
if [ -r "$MODEL" ]; then
  printf '%s bytes  %s\n' "$(stat -c %s "$MODEL")" "$(sha256sum "$MODEL" | cut -c1-16)"
else
  echo "MISSING $MODEL"; fail=1
fi

echo
echo "== listener"
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "health ok on 127.0.0.1:$PORT"
  ss -ltnp 2>/dev/null | grep ":$PORT " || true
  PID=$(pgrep -f "^/.*/nq-serve --model" | head -1 || true)
  if [ -n "${PID:-}" ]; then
    awk '/VmHWM|VmRSS/ {printf "  %s %s kB\n", $1, $2}' "/proc/$PID/status"
    echo "  established connections:"
    ss -tnp 2>/dev/null | grep "pid=$PID" || echo "    none outside the listener"
  fi
else
  echo "not running (start it with demo/start.sh)"
fi

echo
avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
if (( avail < 2500 )); then
  echo "WARN  MemAvailable ${avail} MiB; close browsers or other model jobs before the demo"
  fail=1
fi
if pgrep -x rustc >/dev/null || pgrep -x cargo >/dev/null; then
  echo "WARN  a cargo/rustc build is running; it will make the demo slower"
  fail=1
fi
big=$(ps -eo rss=,comm= | awk '$2 == "python3" && $1 > 1000000 {printf "%.0f MiB", $1/1024}')
if [ -n "$big" ]; then
  echo "WARN  a python3 is using $big; it can OOM this 6.7 GiB box. Stop it before the demo."
  fail=1
fi
if [ "$fail" -ne 0 ]; then
  echo
  echo "pre-flight has warnings; the demo can still run, but expect a slower first token"
fi
