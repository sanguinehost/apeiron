#!/bin/bash
# Probe: the gate_proj ADMM stage that OOM'd at 24.7 GiB before the tape/frame release, plus its
# fact phase on the reallocated narrow tape. Resumes the layer-0 checkpoint at slot 4 phase 1.
# Run as root from /workspace.
exec >> /workspace/probe-gate.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
export APEIRON_NQ_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_LOOP=/workspace/loop-8b
export APEIRON_NQ_TAG=nq-100
export APEIRON_NQ_BLOCKS=0..1
export APEIRON_NQ_SAMPLES=128
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_BATCH=4
export APEIRON_NQ_HOST_STREAMS=1
export APEIRON_NQ_PHASES=2
export APEIRON_MEM_TRACE=1
echo "== probe gate $(date -u)"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) \
  > /workspace/probe-gpu.log 2>&1 &
smi=$!
trap 'kill "$smi" 2>/dev/null' EXIT
cargo test --release --locked --features gpu --test learn_block_tune -- \
  --ignored --exact block_loop --nocapture
rc=$?
kill "$smi" 2>/dev/null
echo "== probe gate rc=$rc $(date -u)"
gcloud storage cp /workspace/probe-gate.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -1
exit $rc
