#!/usr/bin/env bash
# Second driver: finish the six-task screen and the 25-window PPL line with two 30-thread lanes.
#
# The single-lane 61-thread run measured 8.9 s per piqa item on the NQ arm (2.5x the laptop at 15x
# the threads), so the process does not saturate the host. Two lanes at 30 threads cannot be slower
# than one at 60 and use the idle core time; the pair list starts each lane with its heaviest task.
set -u
exec >>/workspace/run2.log 2>&1
set -x
cd /workspace/apeiron || exit 1
Q=target/release/examples/quality
P=target/release/examples/ppl
ITEMS=bench/quality/items
RESULTS=/workspace/results
DATA=/workspace/models/wiki.test.raw
TOK=/workspace/models/tokenizer.json

# The in-flight Q4 piqa run was orphaned by stopping the first driver; let it finish its 200 items.
for _ in $(seq 1 120); do
    [ "$(wc -l <"$RESULTS/q4-piqa.jsonl" 2>/dev/null || echo 0)" -ge 200 ] && break
    sleep 15
done
wc -l "$RESULTS/nq-piqa.jsonl" "$RESULTS/q4-piqa.jsonl"

mkdir -p "$RESULTS/ppl"
export APEIRON_THREADS=30
export Q P ITEMS RESULTS DATA TOK

run_one() {
    local arm=$1 task=$2
    if [ "$arm" = nq ]; then
        "$Q" --model /workspace/models/skeleton.gguf --pack /workspace/models/model.nqp \
            --tokenizer "$TOK" --items "$ITEMS/$task-200.jsonl" \
            --out "$RESULTS/nq-$task.jsonl" --kernel q8act --head-batch 16
    else
        "$Q" --model /workspace/models/Qwen3-8B-Q4_K_M.gguf --tokenizer "$TOK" \
            --items "$ITEMS/$task-200.jsonl" \
            --out "$RESULTS/q4-$task.jsonl" --kernel q8act --head-batch 16
    fi
}
export -f run_one

printf '%s\n' \
    "nq boolq" "q4 hellaswag" \
    "nq hellaswag" "q4 boolq" \
    "nq arc_challenge" "q4 arc_easy" \
    "nq arc_easy" "q4 arc_challenge" \
    "nq winogrande" "q4 winogrande" |
    xargs -P 2 -n 2 bash -c 'run_one "$0" "$1"'

python3 tools/quality-compare.py \
    --nq "$RESULTS/nq-boolq.jsonl" "$RESULTS/nq-piqa.jsonl" "$RESULTS/nq-hellaswag.jsonl" \
    "$RESULTS/nq-winogrande.jsonl" "$RESULTS/nq-arc_easy.jsonl" "$RESULTS/nq-arc_challenge.jsonl" \
    --q4 "$RESULTS/q4-boolq.jsonl" "$RESULTS/q4-piqa.jsonl" "$RESULTS/q4-hellaswag.jsonl" \
    "$RESULTS/q4-winogrande.jsonl" "$RESULTS/q4-arc_easy.jsonl" "$RESULTS/q4-arc_challenge.jsonl" |
    tee "$RESULTS/compare.txt"

SKIPS="0 2048 4096 6144 8192 10240 22016 33792 45568 57344 69120 80896 92672 104448 116224
128000 139776 151552 163328 175104 186880 198656 210432 222208 233984"

ppl_one() {
    local arm=$1 s=$2
    if [ "$arm" = nq ]; then
        "$P" --model /workspace/models/skeleton.gguf --pack /workspace/models/model.nqp \
            --tokenizer "$TOK" --data "$DATA" --tokens 2048 --skip "$s" --ctx 2048 \
            --head-batch 16 --kernel q8act --second-half >"$RESULTS/ppl/NQ$s.log" 2>&1
    else
        "$P" --model /workspace/models/Qwen3-8B-Q4_K_M.gguf --tokenizer "$TOK" --data "$DATA" \
            --tokens 2048 --skip "$s" --ctx 2048 --head-batch 16 --second-half \
            >"$RESULTS/ppl/Q4W$s.log" 2>&1
    fi
}
export -f ppl_one
export SKIPS

( for s in $SKIPS; do ppl_one nq "$s"; done ) &
( for s in $SKIPS; do ppl_one q4 "$s"; done ) &
wait

NQ_LOGS=""
Q4_LOGS=""
for s in $SKIPS; do
    NQ_LOGS="$NQ_LOGS $RESULTS/ppl/NQ$s.log"
    Q4_LOGS="$Q4_LOGS $RESULTS/ppl/Q4W$s.log"
done
# shellcheck disable=SC2086
python3 tools/ppl-pooled.py --nq $NQ_LOGS --q4 $Q4_LOGS --label '25 windows, one host, two lanes' |
    tee "$RESULTS/ppl-all.txt"
SPANS="2048 2048 2048 2048 2048"
for _ in $(seq 20); do SPANS="$SPANS 11776"; done
# shellcheck disable=SC2086
python3 tools/ppl-pooled.py --nq $NQ_LOGS --q4 $Q4_LOGS --label '25 windows, weighted' \
    --spans $SPANS | tee "$RESULTS/ppl-weighted.txt"

{
    echo "=== screen"
    cat "$RESULTS/compare.txt"
    echo "=== ppl pooled"
    tail -12 "$RESULTS/ppl-all.txt"
    echo "=== ppl weighted"
    tail -10 "$RESULTS/ppl-weighted.txt"
} >"$RESULTS/SUMMARY.txt"
tar -czf /workspace/results.tar.gz -C /workspace results
date +%s
echo DONE
