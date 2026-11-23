#!/bin/bash
# The full contributor test gate on the final tree, run on nq-l4 with the GPU idle between campaign
# legs: `cargo test --workspace --all-features --locked` under the CI test job's environment
# (`VK_ICD_FILENAMES=/nonexistent`, so every device test skips as it does in CI). The four equality
# tests already ran on the real device (release-tape-tests.log). Run as root from /workspace.
exec > /workspace/full-tests.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
export VK_ICD_FILENAMES=/nonexistent
rustc --version
cargo --version
cargo test --workspace --all-features --locked -j 4
rc=$?
echo "== full tests rc=$rc $(date -u)"
gcloud storage cp /workspace/full-tests.log \
  gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -1
exit $rc
