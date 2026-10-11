#!/bin/bash
# var.sh <bin> <only> <variants> [pooled]
F=/home/socol/oc-tmp/w7-fixtures
export APEIRON_DENSE_ONLY=$2 APEIRON_DENSE_VARIANTS=$3 APEIRON_F32_CARRIER=1 APEIRON_DENSE_MODEL=${MODEL:-$F/qwen05b-Q8_0.gguf} APEIRON_DENSE_IDS=$F/qwen25.ids APEIRON_ATTRIB_ATT=${ATT:-6} APEIRON_ATTRIB_REPS=${REPS:-20}
if [ "$4" != pooled ]; then export APEIRON_THREADS=1 APEIRON_ECORES=0; PIN="taskset -c 2"; fi
cd /home/socol/oc-tmp/wt/worktree-agent-APEIRON-W7-T03
timeout 500 flock -x -w 900 /home/socol/oc-tmp/w6-host.lock $PIN $1 gemm_dense_shapes --ignored --nocapture 2>&1 | grep -E "variant|VARIANT|panicked|error"
