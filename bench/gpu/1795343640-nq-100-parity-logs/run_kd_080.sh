#!/bin/bash
# 0.80 b/w model KD: eight epochs over the same tape, then the pack.
CARGO=/root/.cargo/bin/cargo
cd /workspace/apeiron-nq || exit 1
export APEIRON_NQ_MODEL=/workspace/Qwen3-0.6B-Base.f16.gguf
export APEIRON_KD_STUDENT=/workspace/loop-080/nq-080
export APEIRON_NQ_DUMP=/workspace/dump-080
export APEIRON_NQ_SAMPLES=128
export APEIRON_KD_EPOCHS=8
export APEIRON_NQ_PACK=/workspace/loop-080/nq-080/pack.nqp
echo "== kd_fit 080 $(date -u +%H:%M:%S)"
"$CARGO" test --release --locked --features gpu --test learn_kd -- --ignored --exact kd_fit --nocapture
echo "== kd_fit 080 rc=$? $(date -u +%H:%M:%S)"
