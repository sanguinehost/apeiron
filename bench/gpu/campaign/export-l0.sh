#!/bin/bash
# Export the finished layer-0 checkpoints as `pack-l0.nqp` without fitting again. `PHASES=0` skips
# every unit, so an unfinished layer ends with `done = false` and writes no pack; a finished layer
# skips all units, advances, and writes the layer-0 pack over the full 36-layer header.
# Run as root from /workspace.
exec > /workspace/export-l0.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
export APEIRON_NQ_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_LOOP=/workspace/loop-8b-trace
export APEIRON_NQ_TAG=nq-100
export APEIRON_NQ_BLOCKS=0..1
export APEIRON_NQ_SAMPLES=128
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_BATCH=4
export APEIRON_NQ_HOST_STREAMS=1
export APEIRON_NQ_PACK_ONLY=0
export APEIRON_NQ_PHASES=0
export APEIRON_NQ_PACK=/workspace/loop-8b-trace/nq-100/pack-l0.nqp
cargo test --release --locked --features gpu --test learn_block_tune -- \
  --ignored --exact block_loop --nocapture
rc=$?
ls -l "$APEIRON_NQ_PACK"
gcloud storage cp "$APEIRON_NQ_PACK" \
  gs://sanguine-checkpoints-scribe-461316/nq/packs/q3-8b-l0-final.nqp
cp_rc=$?
echo "export rc=$rc cp_rc=$cp_rc"
[ "$rc" -eq 0 ] && [ "$cp_rc" -eq 0 ]
