#!/bin/bash
# usage: st.sh <only> <bin>...   single-thread alternating, 3 rounds, prints GMAC/s best
ONLY=$1; shift
F=/home/socol/oc-tmp/w7-fixtures
export APEIRON_DENSE_ONLY=$ONLY APEIRON_THREADS=1 APEIRON_ECORES=0 APEIRON_F32_CARRIER=1 APEIRON_DENSE_MODEL=$F/qwen05b-Q8_0.gguf APEIRON_DENSE_IDS=$F/qwen25.ids APEIRON_ATTRIB_ATT=1 APEIRON_ATTRIB_REPS=${REPS:-30}
cd /home/socol/oc-tmp/wt/worktree-agent-APEIRON-W7-T03
for r in 1 2 3; do
 for b in "$@"; do
  o=$(timeout 300 flock -x -w 900 /home/socol/oc-tmp/w6-host.lock taskset -c 2 $b gemm_dense_shapes --ignored --nocapture 2>&1 | grep SHAPE | sed -E 's/.*best ([0-9.]+) ms = ([0-9.]+) GMAC.*foreign ([0-9.]+), digest (.*)/\2 f\3 \4/')
  echo "$(basename $b) $ONLY: $o"
 done
done
