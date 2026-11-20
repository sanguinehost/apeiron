#!/usr/bin/env bash
# Finish the 25-window PPL line on host A: the eight windows the laptop never measured.
#
# Skips 0..139776 come from the laptop's 17 pairs; this host reproduced three of them to the last
# digit (NQ 0 = 11.374839, NQ 2048 = 11.834074, Q4 0 = 7.587145), so the line is one estimate with
# a cross-host overlap check. Two lanes, each four windows per arm, ~51 min at the measured rates.
set -u
exec >>/workspace/ppl-rem.log 2>&1
set -x
cd /workspace/apeiron || exit 1
P=target/release/examples/ppl
RESULTS=/workspace/results
DATA=/workspace/models/wiki.test.raw
TOK=/workspace/models/tokenizer.json
export APEIRON_THREADS=30
export P RESULTS DATA TOK

ppl_one() {
    local arm=$1 s=$2 out
    if [ "$arm" = nq ]; then out="$RESULTS/ppl/NQ$s.log"; else out="$RESULTS/ppl/Q4W$s.log"; fi
    if grep -q '^ppl ' "$out" 2>/dev/null; then echo "have $out"; return 0; fi
    if [ "$arm" = nq ]; then
        "$P" --model /workspace/models/skeleton.gguf --pack /workspace/models/model.nqp \
            --tokenizer "$TOK" --data "$DATA" --tokens 2048 --skip "$s" --ctx 2048 \
            --head-batch 16 --kernel q8act --second-half >"$out" 2>&1
    else
        "$P" --model /workspace/models/Qwen3-8B-Q4_K_M.gguf --tokenizer "$TOK" --data "$DATA" \
            --tokens 2048 --skip "$s" --ctx 2048 --head-batch 16 --second-half >"$out" 2>&1
    fi
    date +%s >>"$RESULTS/ppl/done-rem.txt"
}
export -f ppl_one

(
    for s in 151552 163328 175104 186880; do ppl_one nq "$s"; ppl_one q4 "$s"; done
) &
(
    for s in 198656 210432 222208 233984; do ppl_one nq "$s"; ppl_one q4 "$s"; done
) &
wait

date +%s
echo PPL_REM_DONE
