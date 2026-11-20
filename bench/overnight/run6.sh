#!/usr/bin/env bash
# Sixth queue: is the shipped q8act *decode* actually faster than the exact reader?
#
# Prefill is (q8act 1.9 tok/s vs exact 0.5 tok/s on the PPL pair), so the multi-column matmat path
# uses the code reader. Decode calls `project_q8` per column; if `CodeTerms::new` refuses the
# activation bounds, the stage silently keeps the exact lane reader and the decode number would be
# the exact rate. One `--kernel exact` decode next to the q8act runs decides it.
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
GEN="$ROOT/target/release/examples/generate"
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

log_status "run6 start epoch=$(date +%s)"

if wait_mem 2600; then
    run_job K6-exact-decode 1800 bash -c '
        p=$(cat /tmp/opencode/nq8b-2k-prompt.txt)
        systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
            '"$GEN"' --model '"$MODEL"' --pack '"$PACK"' --tokenizer '"$TOK"' \
            --prompt "$p" --n 32 --kernel exact
    '
fi

log_status "run6 done epoch=$(date +%s)"
