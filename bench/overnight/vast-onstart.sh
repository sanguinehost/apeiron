#!/usr/bin/env bash
# Onstart for the Q2 campaign on a rented host: validate the shared-prefix scorer, run the six-task
# 0-shot screen on both arms, then the 25-window wikitext-2 PPL line on one host.
#
# Inputs: /workspace/inputs.tar (uploaded with `vastai copy` after the instance starts). The Q4 bar
# file is downloaded here from the stock Hugging Face repo and its sha is checked.
#
# Results: /workspace/results (per-item JSONL, per-window PPL logs, compare/pooled text) and a
# results.tar.gz. The script prints DONE at the end; the container log is the live transcript.
set -u
exec >>/workspace/run.log 2>&1
set -x
date +%s
export DEBIAN_FRONTEND=noninteractive
export APEIRON_THREADS="${APEIRON_THREADS:-44}"
THREADS="$APEIRON_THREADS"

apt-get update -qq
apt-get install -y -qq curl ca-certificates build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal --default-toolchain 1.89.0
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
mkdir -p /workspace/models /workspace/results /workspace/apeiron

# The bar file: stock repo, 5,027,783,488 B, sha d98cdcbd...5785.
cd /workspace/models || exit 1
if [ ! -s Qwen3-8B-Q4_K_M.gguf ]; then
    curl -L --retry 5 --retry-delay 5 -o Qwen3-8B-Q4_K_M.gguf \
        https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
fi
stat -c '%n %s' Qwen3-8B-Q4_K_M.gguf | tee /workspace/results/q4.size
sha256sum Qwen3-8B-Q4_K_M.gguf | tee /workspace/results/q4.sha

# Inputs: source tree, pack, skeleton, tokenizer, test text, item files, laptop smoke reference.
for _ in $(seq 1 240); do
    [ -s /workspace/inputs.tar ] && break
    sleep 15
done
if [ ! -s /workspace/inputs.tar ]; then
    echo "NO_INPUTS after 60 min"
    exit 1
fi
tar -C /workspace -xf /workspace/inputs.tar
sha256sum /workspace/models/model.nqp /workspace/models/skeleton.gguf \
    /workspace/models/tokenizer.json | tee /workspace/results/inputs.sha

cd /workspace/apeiron || exit 1
cargo build --release --locked --example ppl --example quality -j 40 || exit 1
Q=target/release/examples/quality
P=target/release/examples/ppl
ITEMS=bench/quality/items

# ---- 1. the scorer's reference check ----------------------------------------
# The shared-prompt path must reproduce the one-prefill-per-choice path bit for bit, and the joint
# path must reproduce the laptop's 5-item smoke. A mismatch on the first falls back to --no-share
# for the whole screen; a mismatch on the second is recorded and does not stop the run.
NQARGS=(--model /workspace/models/skeleton.gguf --pack /workspace/models/model.nqp
    --tokenizer /workspace/models/tokenizer.json)
APEIRON_THREADS="$THREADS" "$Q" "${NQARGS[@]}" --items "$ITEMS/piqa-200.jsonl" \
    --out /workspace/results/smoke-shared.jsonl --limit 5 --kernel q8act --head-batch 16
APEIRON_THREADS="$THREADS" "$Q" "${NQARGS[@]}" --items "$ITEMS/piqa-200.jsonl" \
    --out /workspace/results/smoke-joint.jsonl --limit 5 --kernel q8act --head-batch 16 --no-share
python3 - <<'PY' | tee /workspace/results/smoke-diff.txt
import json


def lls(path):
    return [json.loads(line)["ll"] for line in open(path) if line.strip()]


shared = lls("/workspace/results/smoke-shared.jsonl")
joint = lls("/workspace/results/smoke-joint.jsonl")
laptop = lls("/workspace/reference/piqa-5.nq.jsonl")
print("shared_vs_joint", shared == joint)
print("joint_vs_laptop", joint == laptop)
print("shared", shared)
print("joint ", joint)
print("laptop", laptop)
PY
if grep -q 'shared_vs_joint True' /workspace/results/smoke-diff.txt; then
    SHARE=""
    echo "SHARE=on"
else
    SHARE="--no-share"
    echo "SHARE=off (fallback)"
fi

# ---- 2. the six-task 0-shot screen -----------------------------------------
TASKS="piqa boolq hellaswag winogrande arc_easy arc_challenge"
for t in $TASKS; do
    APEIRON_THREADS="$THREADS" "$Q" "${NQARGS[@]}" --items "$ITEMS/$t-200.jsonl" \
        --out "/workspace/results/nq-$t.jsonl" --kernel q8act --head-batch 16 $SHARE
    APEIRON_THREADS="$THREADS" "$Q" --model /workspace/models/Qwen3-8B-Q4_K_M.gguf \
        --tokenizer /workspace/models/tokenizer.json --items "$ITEMS/$t-200.jsonl" \
        --out "/workspace/results/q4-$t.jsonl" --kernel q8act --head-batch 16 $SHARE
done
NQ_FILES=""
Q4_FILES=""
for t in $TASKS; do
    NQ_FILES="$NQ_FILES /workspace/results/nq-$t.jsonl"
    Q4_FILES="$Q4_FILES /workspace/results/q4-$t.jsonl"
done
# shellcheck disable=SC2086
python3 tools/quality-compare.py --nq $NQ_FILES --q4 $Q4_FILES |
    tee /workspace/results/compare.txt

# ---- 3. the 25-window PPL line on one host ----------------------------------
SKIPS="0 2048 4096 6144 8192 10240 22016 33792 45568 57344 69120 80896 92672 104448
116224 128000 139776 151552 163328 175104 186880 198656 210432 222208 233984"
DATA=/workspace/models/wiki.test.raw
TOK=/workspace/models/tokenizer.json
mkdir -p /workspace/results/ppl
NQ_LOGS=""
Q4_LOGS=""
SPANS=""
i=0
for s in $SKIPS; do
    APEIRON_THREADS="$THREADS" "$P" --model /workspace/models/skeleton.gguf \
        --pack /workspace/models/model.nqp --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip "$s" --ctx 2048 --head-batch 16 --kernel q8act --second-half \
        >"/workspace/results/ppl/NQ$s.log" 2>&1
    APEIRON_THREADS="$THREADS" "$P" --model /workspace/models/Qwen3-8B-Q4_K_M.gguf \
        --tokenizer "$TOK" --data "$DATA" --tokens 2048 --skip "$s" --ctx 2048 \
        --head-batch 16 --second-half >"/workspace/results/ppl/Q4W$s.log" 2>&1
    NQ_LOGS="$NQ_LOGS /workspace/results/ppl/NQ$s.log"
    Q4_LOGS="$Q4_LOGS /workspace/results/ppl/Q4W$s.log"
    if [ "$i" -lt 5 ]; then SPANS="$SPANS 2048"; else SPANS="$SPANS 11776"; fi
    i=$((i + 1))
done
# shellcheck disable=SC2086
python3 tools/ppl-pooled.py --nq $NQ_LOGS --q4 $Q4_LOGS --label '25 windows, one host' |
    tee /workspace/results/ppl-all.txt
# shellcheck disable=SC2086
python3 tools/ppl-pooled.py --nq $NQ_LOGS --q4 $Q4_LOGS --label '25 windows, weighted' \
    --spans $SPANS | tee /workspace/results/ppl-weighted.txt

{
    echo "=== q4 sha"
    cat /workspace/results/q4.sha
    echo "=== inputs sha"
    cat /workspace/results/inputs.sha
    echo "=== scorer check"
    cat /workspace/results/smoke-diff.txt
    echo "=== screen"
    cat /workspace/results/compare.txt
    echo "=== ppl pooled"
    tail -12 /workspace/results/ppl-all.txt
    echo "=== ppl weighted"
    tail -10 /workspace/results/ppl-weighted.txt
} >/workspace/results/SUMMARY.txt

tar -czf /workspace/results.tar.gz -C /workspace results
date +%s
echo DONE
