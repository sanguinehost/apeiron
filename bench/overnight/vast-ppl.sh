#!/usr/bin/env bash
# PPL-only driver on the first rented host: 25 windows per arm in two 30-thread lanes.
#
# The six-task 0-shot screen moved to the Cascade Lake host; this one keeps the PPL half, which is
# independent of it. Two lanes because one 60-thread process does not saturate the host.
set -u
exec >>/workspace/ppl.log 2>&1
set -x
cd /workspace/apeiron || exit 1
P=target/release/examples/ppl
RESULTS=/workspace/results
DATA=/workspace/models/wiki.test.raw
TOK=/workspace/models/tokenizer.json
mkdir -p "$RESULTS/ppl"
export APEIRON_THREADS=30
export P RESULTS DATA TOK

SKIPS="0 2048 4096 6144 8192 10240 22016 33792 45568 57344 69120 80896 92672 104448 116224
128000 139776 151552 163328 175104 186880 198656 210432 222208 233984"
export SKIPS

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
date +%s
echo PPL_DONE
