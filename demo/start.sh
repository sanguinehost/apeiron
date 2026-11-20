#!/usr/bin/env bash
# Start the local demo model and wait until it answers /health.
#
# The model is Qwen2.5-0.5B-Instruct at Q4_K_M: 491 MB, CPU only, no network.
# It is the live demo model because it answers in seconds on this laptop. The
# 7B-class target is described in demo/explain.md; it does not run at demo speed
# yet.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
MODEL=${DEMO_MODEL:-/home/lucas/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf}
TOKENIZER=${DEMO_TOKENIZER:-/home/lucas/models/qwen2.5-0.5b/tokenizer.json}
# The pool defaults to 8 threads. This i5-10210U is 4 cores / 8 threads at 15 W.
# The demo pins 4, one per physical core: under the `performance` profile that is
# 0.3-0.4 s to first token and 35-41 tok/s (demo/README.md). The older thread
# sweep (2/4/6/8 threads = 5.7/8.6/3.2/3.6 tok/s) ran under `power-saver` at
# 0.6-0.9 GHz before the decode-once row kernels; it is not a comparison of the
# current path under `performance`. The guard inherits this.
THREADS=${DEMO_THREADS:-4}
export APEIRON_THREADS="$THREADS"
PORT=${DEMO_PORT:-8080}
LOG=${DEMO_LOG:-/tmp/opencode/nq-demo-serve.log}
PIDFILE=${DEMO_PIDFILE:-/tmp/opencode/nq-demo.pid}
BIN="$ROOT/serve/target/release/nq-serve"
GUARD=/home/lucas/Workspace/tools/agent-guard

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "already serving on 127.0.0.1:$PORT"
  exit 0
fi
if [ ! -x "$BIN" ]; then
  echo "missing $BIN; build it with: (cd serve && cargo build --release)" >&2
  exit 1
fi
if [ ! -r "$MODEL" ] || [ ! -r "$TOKENIZER" ]; then
  echo "missing model or tokenizer: $MODEL $TOKENIZER" >&2
  exit 1
fi

: >"$LOG"
setsid nohup "$GUARD" run -- \
  "$BIN" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --bind "127.0.0.1:$PORT" --model-name qwen2.5-0.5b-instruct \
    --max-tokens 96 --kernel q8act --think \
  >"$LOG" 2>&1 </dev/null &
echo $! >"$PIDFILE"

printf 'starting the model'
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo
    break
  fi
  printf '.'
  sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo
  echo "the listener did not come up; here is its log:" >&2
  cat "$LOG" >&2
  exit 1
fi

echo
grep -E "loaded|warmup|listening" "$LOG" || true
printf 'model file   %s bytes\n' "$(stat -c %s "$MODEL")"
printf 'threads      %s\n' "$THREADS"
printf 'answer with  %s/ask.sh "your question"\n' "$HERE"
printf 'stop with    %s/stop.sh\n' "$HERE"
