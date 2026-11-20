#!/usr/bin/env bash
# Third overnight queue: the NQ arm on windows 2 onward, as a Q2 risk screen.
#
# The Q4 bar exists only for window 1 (7.587145). For each further 2,048-token window this
# measures the NQ arm's PPL and reports the break-even Q4 PPL, NQ/1.5: if the Q4 arm scores
# below that on the same window, the window fails the 1.5x bar. It is a screen, not a ratio.
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
TS=1794912865
LOGS="bench/cpu/$TS-nq8b-overnight-logs"
DONE="bench/overnight/done"
STATUS="bench/overnight/status.txt"
MODEL=/home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf
PACK=/home/lucas/models/arelath/nq8b-late-v4/model.nqp
TOK=/home/lucas/models/arelath/nq8b/tokenizer.json
DATA=/home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw
PPL="$ROOT/target/release/examples/ppl"
SCOPE=(systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 --)

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
    local need_mb=${1:-2600} waited=0 avail
    while [ "$waited" -lt 1800 ]; do
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

log_status "run3 start epoch=$(date +%s)"

for skip in 2048 4096 6144 8192; do
    if wait_mem 2600; then
        run_job "W${skip}-ppl" 1800 "${SCOPE[@]}" "$PPL" \
            --model "$MODEL" --pack "$PACK" --tokenizer "$TOK" --data "$DATA" \
            --tokens 2048 --skip "$skip" --ctx 2048 --head-batch 16 --kernel q8act --second-half
    fi
done

# The break-even line: the Q4 PPL that would put the window exactly at 1.5x.
for skip in 2048 4096 6144 8192; do
    f="$LOGS/W${skip}-ppl.log"
    [ -s "$f" ] || continue
    ppl=$(awk '/^ppl /{print $2}' "$f" | tail -1)
    [ -n "$ppl" ] || continue
    python3 - "$skip" "$ppl" >>"$STATUS" <<'PY'
import sys
skip, ppl = int(sys.argv[1]), float(sys.argv[2])
print(f"window skip={skip} nq_ppl={ppl:.6f} breakeven_q4={ppl/1.5:.6f} "
      f"vs_window1_q4=7.587145 ratio_if_same={ppl/7.587145:.6f}")
PY
done

log_status "run3 done epoch=$(date +%s)"
