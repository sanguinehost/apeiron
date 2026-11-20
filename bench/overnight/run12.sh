#!/usr/bin/env bash
# Twelfth queue: the Q4_K_M arm of the six-task 0-shot screen, same items, same tokenizer.
#
# Runs after `run11`. The two arms are paired by `(task, item)`, so only identical item files and an
# identical tokenizer make the ratio mean anything. The Q4 file embeds no tokenizer this harness can
# read, so both arms use the product `tokenizer.json`: both are Qwen3-8B derivatives with the same
# 151,936-entry vocabulary, and `quality.rs` refuses any id at or above the model's `vocab_size`.
#
# Budget: the Q4 arm reads 5.03 GB of weights per prefill against the NQ arm's 1.89 GB, so the
# per-position dot is cheaper (VNNI q4_K) while the read is 2.7x larger. Without a measurement the
# per-task timeouts are 3x the NQ arm's; a timeout leaves no marker, so the task reruns on resume.
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
TS=1794912865
LOGS="bench/cpu/$TS-nq8b-overnight-logs"
DONE="bench/overnight/done"
STATUS="bench/overnight/status.txt"
Q4=/home/lucas/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf
TOK=/home/lucas/models/arelath/nq8b/tokenizer.json
ITEMS="$ROOT/bench/quality/items"
QUALITY="$ROOT/target/release/examples/quality"
OUT="$LOGS/quality-q4"

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
    local need_mb=${1:-3600} waited=0 avail
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

# Only tasks whose NQ arm produced a file; a Q4 arm alone has nothing to pair with.
TASKS='
winogrande:25200
piqa:42000
arc_easy:72000
arc_challenge:84000
boolq:174000
hellaswag:180000
'

log_status "run12 start epoch=$(date +%s)"
wait_mem 3600 || exit 0

for spec in $TASKS; do
    task=${spec%%:*}
    limit=${spec##*:}
    if [ ! -s "$LOGS/quality-nq/$task-200.nq.jsonl" ]; then
        log_status "skip QQ-$task (no NQ arm file)"
        continue
    fi
    run_job "QQ-$task" "$limit" \
        systemd-run --user --scope -q -p MemoryMax=3600M -p MemorySwapMax=0 -- \
        "$QUALITY" --model "$Q4" --tokenizer "$TOK" \
        --items "$ITEMS/$task-200.jsonl" --out "$OUT/$task-200.q4.jsonl" \
        --kernel q8act --head-batch 16 --ctx 2048
done

log_status "run12 done epoch=$(date +%s)"
