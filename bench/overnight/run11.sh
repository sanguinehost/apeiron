#!/usr/bin/env bash
# Eleventh queue: the NQ arm of the six-task 0-shot screen, cheapest task first.
#
# The screen is 201,440 prefill + 43,827 head positions total; at the measured 0.239 s/position of a
# 2k prefill that is ~14.2 h for this arm (the harness runs one prefill per candidate, so the model's
# 4.66 tok/s "prefill" rate does not apply — short prompts amortise the 1.02 GB pack over few
# columns and the wall is the dot stream, not the read). Ordering by token count puts a completed
# task marker on disk every 1-4 h, and every job is skipped when its marker exists, so the queue is
# resumable after a reboot or a compaction.
#
# This arm alone cannot establish Q2: `quality-compare.py` needs the Q4 arm on the same items and
# returns exit 3 (`screen`) while any task is below its full split. It can refute.
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
ITEMS="$ROOT/bench/quality/items"
QUALITY="$ROOT/target/release/examples/quality"
OUT="$LOGS/quality-nq"

mkdir -p "$DONE" "$LOGS" "$OUT"

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

# task:timeout — timeout is ~4x the 0.239 s/position estimate for the task's measured token count.
TASKS='
winogrande:8400
piqa:14000
arc_easy:24000
arc_challenge:28000
boolq:58000
hellaswag:60000
'

log_status "run11 start epoch=$(date +%s)"
wait_mem 3400 || exit 0

for spec in $TASKS; do
    task=${spec%%:*}
    limit=${spec##*:}
    run_job "QN-$task" "$limit" \
        systemd-run --user --scope -q -p MemoryMax=3400M -p MemorySwapMax=0 -- \
        "$QUALITY" --model "$MODEL" --pack "$PACK" --tokenizer "$TOK" \
        --items "$ITEMS/$task-200.jsonl" --out "$OUT/$task-200.nq.jsonl" \
        --kernel q8act --head-batch 16 --ctx 2048
done

log_status "run11 done epoch=$(date +%s)"
