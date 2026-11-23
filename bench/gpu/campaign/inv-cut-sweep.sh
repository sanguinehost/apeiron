#!/bin/bash
# Sweep `APEIRON_NQ_INV_CUT` on the layer-0 q_proj ADMM at the campaign geometry: one non-fact phase
# to make the fit's input, snapshot it, then one ADMM-only run per cut. Each run restores the
# snapshot, so every cut factors the same dense weight. Prints the resident split per cut.
# Run as root from /workspace; needs the trace round's final tree.
exec > /workspace/inv-cut-sweep.log 2>&1
set -xe
. /root/.cargo/env
cd /workspace
ROOT=/workspace/loop-8b-cut
rm -rf "$ROOT"
mkdir -p "$ROOT/nq-100"
export APEIRON_NQ_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_LOOP="$ROOT"
export APEIRON_NQ_TAG=nq-100
export APEIRON_NQ_BLOCKS=0..1
export APEIRON_NQ_SLOTS=q
export APEIRON_NQ_SAMPLES=128
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_BATCH=4
export APEIRON_NQ_HOST_STREAMS=1
export APEIRON_MEM_TRACE=1
export APEIRON_NQ_PACK="$ROOT/nq-100/pack.nqp"

run() { # $1 = phases
  cargo test --release --locked --features gpu --test learn_block_tune -- \
    --ignored --exact block_loop --nocapture
}

echo "== sweep setup: q_proj non-fact $(date -u)"
APEIRON_NQ_PHASES=1 run
mkdir -p "$ROOT/pristine"
cp -a "$ROOT/nq-100/l0" "$ROOT/pristine/l0"

for cut in 32 48 64 96; do
  rm -rf "$ROOT/nq-100/l0"
  cp -a "$ROOT/pristine/l0" "$ROOT/nq-100/l0"
  echo "== cut $cut $(date -u)"
  APEIRON_NQ_INV_CUT=$cut APEIRON_NQ_PHASES=1 APEIRON_NQ_ADMM_TRACE=1 run
done
echo "== sweep done $(date -u)"
gcloud storage cp /workspace/inv-cut-sweep.log \
  gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -1
