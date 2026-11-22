#!/bin/bash
# One round of the 8B block loop: layers $1..$2 at T=8192 (four 2048-token chunks), 128 samples,
# batch 4. Resumable per slot; the full checkpoint root is mirrored to the bucket after the round
# so a spot preemption loses at most one round. Usage: campaign.sh 0 4
exec >> /workspace/block-8b.log 2>&1
set -x
[ $# -eq 2 ] || { echo "usage: $0 START END"; exit 2; }
. /root/.cargo/env
cd /workspace
export APEIRON_NQ_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_LOOP=/workspace/loop-8b
export APEIRON_NQ_TAG=nq-100
export APEIRON_NQ_BLOCKS="$1..$2"
export APEIRON_NQ_SAMPLES=96
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_BATCH=4
export APEIRON_NQ_PACK=/workspace/loop-8b/nq-100/pack.nqp
mkdir -p "$APEIRON_NQ_LOOP/nq-100"

echo "== block_loop $1..$2 $(date -u)"
cargo test --release --locked --features gpu --test learn_block_tune -- \
  --ignored --exact block_loop --nocapture
rc=$?
echo "== block_loop $1..$2 rc=$rc $(date -u)"
gcloud storage rsync -r "$APEIRON_NQ_LOOP/nq-100" \
  gs://sanguine-checkpoints-scribe-461316/nq/loop-8b/nq-100 2>&1 | tail -3
gcloud storage cp /workspace/block-8b.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -2
exit $rc
