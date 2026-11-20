#!/usr/bin/env bash
# Thirteenth queue: the reproducibility control the five-window Q2 numbers need.
#
# The per-window ratios spread 1.467-1.646, and the window-1 margin that was called a pass is 0.0517 %
# = 0.0005 nats. Two questions follow:
#   1. Does a rerun of the same window reproduce to the last digit? The projections are integer and
#      the softmax f64, but the pool splits each row's 32-lane groups across participants, and the
#      sum order could vary between runs. If it does, a 0.05 % margin is inside the run noise and
#      the two-window pass was never a pass.
#   2. Is window 3's Q4 line (10.462647 against 11.834074 on window 2) real, or a fluke of that run?
#
# Both arms' window 1 and the Q4 window 3 are rerun with the identical command, under new names.
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
TS=1794912865
LOGS="bench/cpu/$TS-nq8b-overnight-logs"
DONE="bench/overnight/done"
STATUS="bench/overnight/status.txt"
MODEL=/home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf
PACK=/home/lucas/models/arelath/nq8b-late-v4/model.nqp
Q4=/home/lucas/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf
TOK=/home/lucas/models/arelath/nq8b/tokenizer.json
DATA=/home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw
PPL="$ROOT/target/release/examples/ppl"
SCOPE=(systemd-run --user --scope -q -p MemoryMax=3600M -p MemorySwapMax=0 --)

mkdir -p "$DONE" "$LOGS"

exec 9>bench/overnight/.lock
flock 9

export APEIRON_THREADS=4

log_status() {
    printf '%s\n' "$*" >>"$STATUS"
}

run_job() {
    local id=$1 timeout_s=$2
    shift 2
    local log="$LOGS/$id.log" marker="$DONE/$id" t0 t1 rc
    if [ -e "$marker" ]; then
        log_status "skip $id (done)"
        return 0
    fi
    log_status "start $id epoch=$(date +%s) :: $*"
    t0=$(date +%s)
    timeout "$timeout_s" "$@" >>"$log" 2>&1
    rc=$?
    t1=$(date +%s)
    log_status "end $id rc=$rc wall=$((t1 - t0))s"
    if [ "$rc" -eq 0 ]; then
        touch "$marker"
    fi
    return 0
}

wait_mem() {
    local need_mb=${1:-3400} waited=0 avail
    while [ "$waited" -lt 3600 ]; do
        avail=$(awk '/MemAvailable/{print int($2 / 1024)}' /proc/meminfo)
        if [ "$avail" -ge "$need_mb" ]; then
            log_status "mem ok avail=${avail}MiB need=${need_mb}MiB"
            return 0
        fi
        log_status "mem wait avail=${avail}MiB need=${need_mb}MiB"
        sleep 60
        waited=$((waited + 60))
    done
    log_status "mem refused after ${waited}s at ${avail}MiB < ${need_mb}MiB"
    return 1
}

log_status "run13 start epoch=$(date +%s)"

if wait_mem 3400; then
    run_job NQw1-repeat 3600 "${SCOPE[@]}" "$PPL" \
        --model "$MODEL" --pack "$PACK" --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip 0 --ctx 2048 --head-batch 16 --kernel q8act --second-half
fi

if wait_mem 3400; then
    run_job Q4w1-repeat 10800 "${SCOPE[@]}" "$PPL" \
        --model "$Q4" --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip 0 --ctx 2048 --head-batch 16 --second-half
fi

if wait_mem 3400; then
    run_job Q4w3-repeat 10800 "${SCOPE[@]}" "$PPL" \
        --model "$Q4" --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip 4096 --ctx 2048 --head-batch 16 --second-half
fi

python3 - "$LOGS" "$STATUS" <<'PY'
import re, sys
from pathlib import Path
logs, status = Path(sys.argv[1]), Path(sys.argv[2])
# Window 1's NQ arm is the depth receipt's log: the fifth queue never reran it.
BASES = {
    "nq w1": Path("bench/cpu/1794896874-nq8b-depth-logs/ppl-late-v4-q6k-q4kembed-ctx2048.log"),
    "q4 w1": logs / "Q4W1-ppl.log",
    "q4 w3": logs / "Q4W4096-ppl.log",
}
REPEATS = {
    "nq w1": logs / "NQw1-repeat.log",
    "q4 w1": logs / "Q4w1-repeat.log",
    "q4 w3": logs / "Q4w3-repeat.log",
}

def last(f):
    if not f.exists():
        return None
    m = re.findall(r"^ppl ([\d.]+)", f.read_text(), re.M)
    return float(m[-1]) if m else None

with status.open("a") as out:
    for key in BASES:
        first, again = last(BASES[key]), last(REPEATS[key])
        if first is None or again is None:
            out.write(f"determinism {key}: missing first={first} repeat={again}\n")
            continue
        out.write(f"determinism {key}: first={first:.6f} repeat={again:.6f} "
                  f"delta={again - first:+.6f}\n")
PY

log_status "run13 done epoch=$(date +%s)"
