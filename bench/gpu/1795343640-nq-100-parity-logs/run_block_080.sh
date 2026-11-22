#!/bin/bash
# 0.80 b/w block loop: same tape, ppm-800000 ranks, fresh loop root.
CARGO=/root/.cargo/bin/cargo
cd /workspace/apeiron-nq || exit 1
export APEIRON_NQ_MODEL=/workspace/Qwen3-0.6B-Base.f16.gguf
export APEIRON_NQ_DUMP=/workspace/dump-080
export APEIRON_NQ_LOOP=/workspace/loop-080
export APEIRON_NQ_TAG=nq-080
export APEIRON_NQ_BLOCKS=0..28
export APEIRON_NQ_SAMPLES=128
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_BATCH=4
export APEIRON_NQ_PACK=/workspace/loop-080/nq-080/pack.nqp
mkdir -p /workspace/loop-080/nq-080
echo "== block_loop 080 $(date -u +%H:%M:%S)"
"$CARGO" test --release --locked --features gpu --test learn_block_tune -- --ignored --exact block_loop --nocapture
echo "== block_loop 080 rc=$? $(date -u +%H:%M:%S)"
