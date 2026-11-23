#!/bin/bash
# One fresh process per layer. The block loop's Vulkan allocator grows across layers; the second
# layer's 12288x4096 r3040 down_proj fails at `vkAllocateMemory` with `VK_ERROR_OUT_OF_DEVICE_MEMORY`
# after the first layer's 20.1 GiB peak, on the 24 GiB L4. A process per layer bounds the allocator;
# the per-slot checkpoints make a resumed layer cheap. Usage:
#   campaign-per-layer.sh 2 36
set -u
[ $# -eq 2 ] || { echo "usage: $0 START END"; exit 2; }
campaign=${APEIRON_CAMPAIGN:-/workspace/bench/gpu/campaign/campaign.sh}
for l in $(seq "$1" "$2"); do
  echo "== per-layer $l $(date -u)"
  "$campaign" "$l" "$l"
  rc=$?
  [ "$rc" -eq 0 ] || { echo "== per-layer stop at $l rc=$rc $(date -u)"; exit "$rc"; }
done
