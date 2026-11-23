#!/bin/bash
# After the traced layer-0 round: compile the final tree, run the release equality tests, capture the
# host-import probe, sweep the inverse cutoff, then clippy and fmt. One remote invocation; the
# sub-scripts keep their own logs. Run as root from /workspace.
exec > /workspace/verify-sweep.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
mirror() {
  for f in verify-sweep.log release-tape-tests.log hostmem-probe.log \
           hostmem-probe-gpu.log inv-cut-sweep.log; do
    [ -f "/workspace/$f" ] && gcloud storage cp "/workspace/$f" \
      gs://sanguine-checkpoints-scribe-461316/nq/l4/ >/dev/null 2>&1
  done
}
trap mirror EXIT

echo "== check $(date -u)"
cargo check --tests --features gpu --locked -j 4 || exit 1
echo "== release tests $(date -u)"
bash bench/gpu/campaign/release-tape-tests.sh || exit 1
echo "== layer-0 pack export $(date -u)"
(
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
    --ignored --exact block_loop --nocapture || exit 1
) || exit 1
gcloud storage cp /workspace/loop-8b-trace/nq-100/pack-l0.nqp \
  gs://sanguine-checkpoints-scribe-461316/nq/packs/q3-8b-l0-final.nqp || exit 1
echo "== host-memory probe $(date -u)"
bash bench/gpu/campaign/hostmem-probe.sh || exit 1
echo "== inverse-cut sweep $(date -u)"
bash bench/gpu/campaign/inv-cut-sweep.sh || exit 1
echo "== clippy $(date -u)"
cargo clippy --workspace --all-features --all-targets --locked -j 4 -- -D warnings || exit 1
echo "== fmt $(date -u)"
cargo fmt --all -- --check || exit 1
echo "== all rc=0 $(date -u)"
