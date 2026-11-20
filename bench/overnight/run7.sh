#!/usr/bin/env bash
# Seventh queue: the Q4 arm on windows 3–5, so every NQ window has a same-window bar.
#
# Gated on Q4W2: if the 5.03 GB file cannot run two windows under the 3.6 GiB scope, more windows do
# not help. Each ratio is NQ_w / Q4_w for the same skip; window 1's 7.587145 is not comparable.
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
TS=1794912865
LOGS="bench/cpu/$TS-nq8b-overnight-logs"
DONE="bench/overnight/done"
STATUS="bench/overnight/status.txt"
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
    log_status "mem refused avail=${avail}MiB need=${need_mb}MiB after ${waited}s"
    return 1
}

log_status "run7 start epoch=$(date +%s)"

if [ ! -e "$DONE/Q4W2-ppl" ]; then
    log_status "run7 skipped: no Q4W2 marker"
    exit 0
fi

for skip in 4096 6144 8192; do
    if wait_mem 3400; then
        run_job "Q4W${skip}-ppl" 10800 "${SCOPE[@]}" "$PPL" \
            --model "$Q4" --tokenizer "$TOK" --data "$DATA" \
            --tokens 2048 --skip "$skip" --ctx 2048 --head-batch 16 --second-half
    fi
done

python3 - "$LOGS" "$STATUS" <<'PY'
import re, sys
from pathlib import Path
logs, status = Path(sys.argv[1]), Path(sys.argv[2])

def ppl(name):
    f = logs / f"{name}.log"
    if not f.exists():
        return None
    m = re.findall(r"^ppl ([\d.]+)", f.read_text(), re.M)
    return float(m[-1]) if m else None

# run4 names window 1 and 2 `Q4W1`/`Q4W2`; this queue names the rest by skip.
names = {2048: "Q4W2-ppl", 4096: "Q4W4096-ppl", 6144: "Q4W6144-ppl", 8192: "Q4W8192-ppl"}
with status.open("a") as out:
    for skip, q4name in names.items():
        q4 = ppl(q4name)
        nq = ppl(f"W{skip}-ppl")
        if q4 and nq:
            out.write(f"Q2 window skip={skip} nq={nq:.6f} q4={q4:.6f} ratio={nq/q4:.6f} "
                      f"bar={1.5*q4:.6f} margin={1.5*q4-nq:+.6f}\n")
PY

log_status "run7 done epoch=$(date +%s)"
