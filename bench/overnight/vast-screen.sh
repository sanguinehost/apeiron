#!/usr/bin/env bash
# Screen-only driver on the Cascade Lake host: validate the shared scorer, run the six-task 0-shot
# screen in two 22-thread lanes, and write the compare verdict. The PPL line runs on host A.
set -u
exec >>/workspace/screen.log 2>&1
set -x
cd /workspace/apeiron || exit 1
# shellcheck disable=SC1091
. /root/.cargo/env
Q=target/release/examples/quality
ITEMS=bench/quality/items
RESULTS=/workspace/results
TOK=/workspace/models/tokenizer.json
mkdir -p "$RESULTS"

cargo build --release --locked --example quality -j 40 || exit 1

NQARGS=(--model /workspace/models/skeleton.gguf --pack /workspace/models/model.nqp
    --tokenizer "$TOK")

export APEIRON_THREADS=22

# validation: piqa five items, then one item per task
APEIRON_THREADS=22 "$Q" "${NQARGS[@]}" --items "$ITEMS/piqa-200.jsonl" \
    --out "$RESULTS/smoke-shared.jsonl" --limit 5 --kernel q8act --head-batch 16
APEIRON_THREADS=22 "$Q" "${NQARGS[@]}" --items "$ITEMS/piqa-200.jsonl" \
    --out "$RESULTS/smoke-joint.jsonl" --limit 5 --kernel q8act --head-batch 16 --no-share
python3 - <<'PY' >"$RESULTS/smoke-diff.txt"
import json


def lls(path):
    return [json.loads(line)["ll"] for line in open(path) if line.strip()]


shared = lls("/workspace/results/smoke-shared.jsonl")
joint = lls("/workspace/results/smoke-joint.jsonl")
print("shared_vs_joint", shared == joint)
print("shared", shared)
print("joint ", joint)
PY
cat "$RESULTS/smoke-diff.txt"
if ! grep -q 'shared_vs_joint True' "$RESULTS/smoke-diff.txt"; then
    echo "VALIDATION_FAILED"
    exit 1
fi

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
export Q ITEMS RESULTS TOK

printf '%s\n' \
    "nq boolq" "q4 hellaswag" \
    "nq hellaswag" "q4 boolq" \
    "nq piqa" "q4 piqa" \
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
date +%s
echo SCREEN_DONE
