#!/bin/bash
# The contributor gate on the final tree at the CI pin (nightly-2026-04-14): compile, the four
# release equality tests, clippy, fmt, then the layer-0 pack export on that binary.
# Run as root from /workspace.
exec > /workspace/final-check.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
export RUSTUP_TOOLCHAIN=nightly-2026-04-14
rustc --version
cargo clippy --version
mirror() {
  for f in final-check.log release-tape-tests.log export-l0.log; do
    [ -f "/workspace/$f" ] && gcloud storage cp "/workspace/$f" \
      gs://sanguine-checkpoints-scribe-461316/nq/l4/ >/dev/null 2>&1
  done
}
trap mirror EXIT

echo "== check $(date -u)"
cargo check --tests --features gpu --locked -j 4 || exit 1
echo "== release tests $(date -u)"
bash bench/gpu/campaign/release-tape-tests.sh || exit 1
echo "== clippy $(date -u)"
cargo clippy --workspace --all-features --all-targets --locked -j 4 -- -D warnings || exit 1
echo "== fmt $(date -u)"
cargo fmt --all -- --check || exit 1
echo "== export $(date -u)"
bash bench/gpu/campaign/export-l0.sh || exit 1
echo "== final-check rc=0 $(date -u)"
