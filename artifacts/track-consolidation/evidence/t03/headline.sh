#!/bin/bash
cd /home/socol/oc-tmp/wt/worktree-agent-APEIRON-W7-T03
F=/home/socol/oc-tmp/w7-fixtures
APEIRON_F32_CARRIER=1 flock -x -w 900 /home/socol/oc-tmp/w6-host.lock timeout 540 /home/socol/oc-tmp/wt/bin/engine_profile-w7t03-after --model $F/qwen05b-Q8_0.gguf --ids $F/qwen25.ids --pp 200 --tg 64 --reps 5 --llama /home/socol/oc-tmp/astra-projection-profile/llama_cpu --llama-threads 8,16,24,32 --label w7t03-headline-q05b --out /home/socol/oc-tmp/w7t03/ab > /home/socol/oc-tmp/w7t03/headline.log 2>&1
echo "exit $?" >> /home/socol/oc-tmp/w7t03/headline.log
