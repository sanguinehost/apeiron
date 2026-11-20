#!/usr/bin/env bash
# Eighth queue: the decode thread-scaling curve on the v4 pair, via `phase_split` so no 2k prefill
# is paid. Replaces the 1- and 2-thread 2k runs that were cut from K4 (see `stop-k4.sh`).
#
# The question it answers: at 4 participants the NQ reader runs 8.36 G terms/s per core and the
# token is 285 ms; does the pool scale, or is the 3.5 tok/s a single-core rate wearing four threads?
# A width that does not improve ffn means the lever is the kernel, not the pool.
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
TS=1794912865
LOGS="bench/cpu/$TS-nq8b-overnight-logs"
DONE="bench/overnight/done"
STATUS="bench/overnight/status.txt"
MODEL=/home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf
PACK=/home/lucas/models/arelath/nq8b-late-v4/model.nqp
SPLIT="$ROOT/target/release/examples/phase_split"
SCOPE=(systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 --)

mkdir -p "$DONE" "$LOGS"

exec 9>bench/overnight/.lock
flock 9

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

log_status "run8 start epoch=$(date +%s)"

for t in 1 2 8; do
    run_job "S${t}-split" 1800 "${SCOPE[@]}" env APEIRON_THREADS="$t" \
        "$SPLIT" --model "$MODEL" --pack "$PACK" --steps 24
done

log_status "run8 done epoch=$(date +%s)"
