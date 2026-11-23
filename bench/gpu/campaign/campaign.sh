#!/bin/bash
# One round of the 8B block loop: layers $1..$2 at T=8192 (four 2048-token chunks), 128 samples,
# batch 4. Resumable per slot; the checkpoint root, the pack and this log are mirrored to the bucket
# every five minutes, so a spot preemption loses at most the slots since the last mirror. Usage:
#   campaign.sh 1 36
#   APEIRON_NQ_LOOP=/workspace/loop-8b-trace campaign.sh 1 36
exec >> /workspace/block-8b.log 2>&1
set -x
[ $# -eq 2 ] || { echo "usage: $0 START END"; exit 2; }
. /root/.cargo/env
cd /workspace
export APEIRON_NQ_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_LOOP=${APEIRON_NQ_LOOP:-/workspace/loop-8b}
export APEIRON_NQ_TAG=nq-100
export APEIRON_NQ_BLOCKS="$1..$2"
export APEIRON_NQ_SAMPLES=128
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_BATCH=4
export APEIRON_NQ_HOST_STREAMS=1
export APEIRON_NQ_ADMM_TRACE=1
# The inverse cutoff: cut 48 is 110.96 s on the q ADMM against 123.58 s at 96 with the same recon
# (bench/notes/2026-11-23-admm-split.md); the gate's r=3040 shape is in the campaign, not swept.
export APEIRON_NQ_INV_CUT=48
export APEIRON_NQ_PACK="$APEIRON_NQ_LOOP/nq-100/pack.nqp"
mkdir -p "$APEIRON_NQ_LOOP/nq-100"

# The per-slot checkpoints are the resumable state; mirror them while the round runs, not only at its
# end, so a preemption costs the slots since the last pass rather than the whole round.
loop=$APEIRON_NQ_LOOP
root=$(basename "$loop")
mirror() {
  gcloud storage rsync -r "$loop/nq-100" \
    "gs://sanguine-checkpoints-scribe-461316/nq/$root/nq-100" >/dev/null 2>&1
  gcloud storage cp /workspace/block-8b.log \
    gs://sanguine-checkpoints-scribe-461316/nq/l4/ >/dev/null 2>&1
}
( while :; do sleep 300; mirror; done ) &
mirror_pid=$!
trap 'kill "$mirror_pid" 2>/dev/null' EXIT

echo "== block_loop $1..$2 $(date -u)"
cargo test --release --locked --features gpu --test learn_block_tune -- \
  --ignored --exact block_loop --nocapture
rc=$?
echo "== block_loop $1..$2 rc=$rc $(date -u)"
kill "$mirror_pid" 2>/dev/null
gcloud storage rsync -r "$APEIRON_NQ_LOOP/nq-100" \
  "gs://sanguine-checkpoints-scribe-461316/nq/$root/nq-100" 2>&1 | tail -3
gcloud storage cp /workspace/block-8b.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -2
exit $rc
