#!/bin/bash
# usage: dense.sh <bin> <tag> <0.5b|1.5b|q35> [single]
BIN=$1; TAG=$2; M=$3; S=$4
F=/home/socol/oc-tmp/w7-fixtures
case $M in
 0.5b) export APEIRON_DENSE_MODEL=$F/qwen05b-Q8_0.gguf APEIRON_DENSE_IDS=$F/qwen25.ids;;
 1.5b) export APEIRON_DENSE_MODEL=$F/qwen15b-Q8_0.gguf APEIRON_DENSE_IDS=$F/qwen25.ids;;
 q35) export APEIRON_DENSE_MODEL=/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf APEIRON_DENSE_IDS=/home/socol/oc-tmp/q35-t5/prompt2.ids;;
esac
if [ "$S" = single ]; then export APEIRON_THREADS=1 APEIRON_ECORES=0; PIN="taskset -c 2"; fi
export APEIRON_F32_CARRIER=1
cd /home/socol/oc-tmp/wt/worktree-agent-APEIRON-W7-T03
flock -x -w 900 /home/socol/oc-tmp/w6-host.lock timeout 540 ${PIN} $BIN gemm_dense_shapes --ignored --nocapture > /home/socol/oc-tmp/w7t03/dense-$TAG-$M${S:+-single}.log 2>&1
echo "exit $?" >> /home/socol/oc-tmp/w7t03/dense-$TAG-$M${S:+-single}.log
