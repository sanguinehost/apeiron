#!/usr/bin/env bash
# Ninth queue: prove the 0-shot harness end to end on the Q2 pair, at five items per arm.
#
# Five piqa items are ~100 tokens each and two candidates each, so each arm is ~1k scored tokens:
# minutes, not the 45.9 h screen. The point is that the tokenizer boundary, the prefill slice, the
# argmax metrics and the two-arm pairing all run on the product pair, not that five items decide
# anything. `quality-compare.py` must print `undecided` and exit 3 here.
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
ITEMS="$ROOT/bench/quality/items/piqa-200.jsonl"
QUALITY="$ROOT/target/release/examples/quality"
CMP="$ROOT/tools/quality-compare.py"
OUT="$LOGS/quality-smoke"

mkdir -p "$DONE" "$LOGS" "$OUT"

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

log_status "run9 start epoch=$(date +%s)"

run_job T1-quality-nq 3600 systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
    "$QUALITY" --model "$MODEL" --pack "$PACK" --tokenizer "$TOK" \
    --items "$ITEMS" --out "$OUT/piqa-5.nq.jsonl" --limit 5 --kernel q8act --head-batch 16

run_job T2-quality-q4 3600 systemd-run --user --scope -q -p MemoryMax=3600M -p MemorySwapMax=0 -- \
    "$QUALITY" --model "$Q4" --tokenizer "$TOK" \
    --items "$ITEMS" --out "$OUT/piqa-5.q4.jsonl" --limit 5 --kernel q8act --head-batch 16

# The comparison is a screen at five items; exit 3 is the documented answer, so it is not a failure.
if [ -e "$DONE/T1-quality-nq" ] && [ -e "$DONE/T2-quality-q4" ]; then
    run_job T3-quality-compare 600 bash -c '
        python3 '"$CMP"' --nq '"$OUT"'/piqa-5.nq.jsonl --q4 '"$OUT"'/piqa-5.q4.jsonl
        rc=$?
        [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ]
    '
fi

log_status "run9 done epoch=$(date +%s)"
