#!/usr/bin/env bash
# Ask the running demo model one question and print the answer with timings.
#
#   ./demo/ask.sh "Write a two-sentence welcome message for a community clinic."
#
# Streams the answer to the terminal (the model is greedy, so the answer is the
# same on every run). Prints time to first token, total time, tokens and the
# resident set of the listener.
set -euo pipefail
PORT=${DEMO_PORT:-8080}
LOG=${DEMO_LOG:-/tmp/opencode/nq-demo-serve.log}
PROMPT=${1:-Write a warm, two-sentence welcome message for a community mental health clinic. Use plain language.}

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "the listener is not up; run demo/start.sh first" >&2
  exit 1
fi

python3 - "$PORT" "$PROMPT" <<'PY'
import json
import sys
import textwrap
import time
import urllib.request

port = int(sys.argv[1])
prompt = sys.argv[2]
body = json.dumps({
    "model": "qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 96,
    "stream": True,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body,
    headers={"content-type": "application/json"},
)
t0 = time.monotonic()
first = None
out = []
usage = {}
finish = None
with urllib.request.urlopen(req, timeout=900) as r:
    print(f"prompt  {prompt}")
    print("answer  ", end="", flush=True)
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            piece = (choice.get("delta") or {}).get("content") or ""
            if piece:
                if first is None:
                    first = time.monotonic()
                out.append(piece)
                print(piece, end="", flush=True)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    print()
total = time.monotonic() - t0
text = "".join(out)
print()
ttft = 0.0 if first is None else first - t0
decode_s = max(total - ttft, 1e-9)
n = (usage.get("completion_tokens", len(out)) if usage else len(out))
print(f"time to first token  {ttft:.2f} s")
print(f"decode               {n} tok in {decode_s:.2f} s ({n / decode_s:.2f} tok/s)")
print(f"end-to-end           {total:.2f} s")
print(f"finish       {finish}")
print(f"characters   {len(text)}")
PY

PID=$(pgrep -f "^/home/lucas/Workspace/apeiron-nq/serve/target/release/nq-serve --model" | head -1 || true)
if [ -n "${PID:-}" ]; then
  awk '/VmHWM|VmRSS/ {printf "memory       %s %s kB\n", $1, $2}' "/proc/$PID/status"
fi
tail -n 1 "$LOG" 2>/dev/null | sed 's/^/server       /'
