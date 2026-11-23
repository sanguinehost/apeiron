#!/bin/bash
# One full 128-sample layer-0 round at the campaign geometry with the resident ADMM sub-phase trace
# on: the seven slots in the reference's order, in a fresh checkpoint root so the split is the
# final tree's (the preflight's earlier slots ran before the tape and frame release). Run as root
# from /workspace.
exec >> /workspace/trace-round.log 2>&1
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
export APEIRON_NQ_PACK=/workspace/loop-8b-trace/nq-100/pack-l0.nqp
export APEIRON_NQ_ADMM_TRACE=1
mkdir -p "$APEIRON_NQ_LOOP/nq-100"

echo "== trace round $(date -u)"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) \
  > /workspace/trace-gpu.log 2>&1 &
smi=$!
trap 'kill "$smi" 2>/dev/null' EXIT
cargo test --release --locked --features gpu --test learn_block_tune -- \
  --ignored --exact block_loop --nocapture
rc=$?
kill "$smi" 2>/dev/null
echo "== trace round rc=$rc $(date -u)"
gcloud storage rsync -r "$APEIRON_NQ_LOOP/nq-100" \
  gs://sanguine-checkpoints-scribe-461316/nq/loop-8b-trace/nq-100 2>&1 | tail -1
gcloud storage cp /workspace/trace-round.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -1
gcloud storage cp /workspace/trace-gpu.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -1
exit $rc
