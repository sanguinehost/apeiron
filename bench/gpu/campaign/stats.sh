#!/bin/bash
# The campaign's input step for Qwen3-8B-Base: the 128x2048 wikitext-2 calibration base, the
# reference's online 99.9th-percentile diagonals, then the dump (ids, ranks at 1.00 b/w, perms,
# stats). Idempotent: every stage skips what exists. Run as root from /workspace.
exec >> /workspace/stats.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
export APEIRON_CALIB_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_NQ_MODEL=$APEIRON_CALIB_MODEL
export APEIRON_NQ_TOKENIZER=/workspace/models/Qwen3-8B-Base/tokenizer.json
export APEIRON_NQ_TEXT=/workspace/wikitext2/wiki.train.raw
export APEIRON_NQ_BASE=/workspace/calib8b/calib.s128.kld
export APEIRON_CALIB_BASE=$APEIRON_NQ_BASE
export APEIRON_CALIB_CACHE=/workspace/calib8b/cache
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_SAMPLES=128
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_SEED=0
export APEIRON_CALIB_STRATEGY=online
mkdir -p "$APEIRON_CALIB_CACHE" "$APEIRON_NQ_DUMP/stats" /workspace/calib8b

echo "== base $(date -u)"
if [ ! -s "$APEIRON_NQ_BASE" ]; then
  cargo test --release --locked --features gpu --test learn_nanoquant -- \
    --ignored --exact nq_calib_base --nocapture || exit 1
fi

echo "== diagonals $(date -u)"
cargo test --release --locked --features gpu --test learn_calib -- \
  --ignored --exact calib_diag_write --nocapture || exit 1

echo "== dump $(date -u)"
APEIRON_NQ_PPM=1000000 cargo test --release --locked --features gpu --test learn_calib -- \
  --ignored --exact nq_dump_write --nocapture || exit 1

ls -l "$APEIRON_NQ_DUMP" "$APEIRON_CALIB_CACHE" | head -30
echo "== stats done $(date -u) rc=$?"
gcloud storage cp /workspace/stats.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -2
