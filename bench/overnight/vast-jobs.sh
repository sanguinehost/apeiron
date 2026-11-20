#!/usr/bin/env bash
# Run a list of `arm task` pairs, two jobs at a time at 22 threads each.
#
# $1 is a file with one "arm task" pair per line, ordered so that consecutive lines are the two
# concurrent lanes. Items are the 100-item systematic halves (`task-100.jsonl`), written by
# `awk 'NR%2==1'` from the 200-item files.
set -u
exec >>/workspace/jobs.log 2>&1
set -x
cd /workspace/apeiron || exit 1
Q=target/release/examples/quality
ITEMS=bench/quality/items
RESULTS=/workspace/results
TOK=/workspace/models/tokenizer.json
mkdir -p "$RESULTS"
export APEIRON_THREADS="${THREADS:-22}"

run_one() {
    local arm=$1 task=$2
    if [ "$arm" = nq ]; then
        "$Q" --model /workspace/models/skeleton.gguf --pack /workspace/models/model.nqp \
            --tokenizer "$TOK" --items "$ITEMS/$task-100.jsonl" \
            --out "$RESULTS/nq-$task.jsonl" --kernel q8act --head-batch 16
    else
        "$Q" --model /workspace/models/Qwen3-8B-Q4_K_M.gguf --tokenizer "$TOK" \
            --items "$ITEMS/$task-100.jsonl" --out "$RESULTS/q4-$task.jsonl" \
            --kernel q8act --head-batch 16
    fi
}
export -f run_one
export Q ITEMS RESULTS TOK

xargs -P 2 -n 2 bash -c 'run_one "$0" "$1"' <"$1"
date +%s
echo JOBS_DONE
