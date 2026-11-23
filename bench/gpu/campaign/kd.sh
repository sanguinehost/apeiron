#!/bin/bash
# The model KD over the finished block loop: eight epochs, then the pack. Resumable per step
# (`APEIRON_KD_EPOCHS` from the done state is not a thing here; a rerun resumes at the last
# checkpoint written by the test). Run as root from /workspace.
exec >> /workspace/kd-8b.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
export APEIRON_NQ_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_KD_STUDENT=/workspace/loop-8b/nq-100
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_SAMPLES=128
export APEIRON_KD_EPOCHS=8
export APEIRON_NQ_PACK=/workspace/loop-8b/nq-100/pack.nqp

echo "== kd_fit $(date -u)"
cargo test --release --locked --features gpu --test learn_kd -- \
  --ignored --exact kd_fit --nocapture
rc=$?
echo "== kd_fit rc=$rc $(date -u)"
ls -l "$APEIRON_NQ_PACK" 2>/dev/null
gcloud storage cp "$APEIRON_NQ_PACK" gs://sanguine-checkpoints-scribe-461316/nq/packs/q3-8b-100.nqp 2>&1 | tail -2
gcloud storage cp /workspace/kd-8b.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -2
exit $rc
