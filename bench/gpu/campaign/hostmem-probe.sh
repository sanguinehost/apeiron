#!/bin/bash
# The imported-stream probe: four 4 GiB host mappings touched and copied both ways through a 4 GiB
# device buffer, with `nvidia-smi` polled every 0.5 s. The artifact behind the handoff's "the import
# is not device memory" sentence. Run as root from /workspace.
exec > /workspace/hostmem-probe.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) \
  > /workspace/hostmem-probe-gpu.log 2>&1 &
smi=$!
trap 'kill "$smi" 2>/dev/null' EXIT
cargo run --release --locked --features gpu --example nq_hostmem_probe -- copy
rc=$?
kill "$smi" 2>/dev/null
echo "== hostmem probe rc=$rc $(date -u)"
awk 'BEGIN{m=0} {if ($1+0 > m) m=$1+0} END {print "peak " m " MiB over " NR " polls"}' \
  /workspace/hostmem-probe-gpu.log
gcloud storage cp /workspace/hostmem-probe.log \
  gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -1
gcloud storage cp /workspace/hostmem-probe-gpu.log \
  gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -1
exit $rc
