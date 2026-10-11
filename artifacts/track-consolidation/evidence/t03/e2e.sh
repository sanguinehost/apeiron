#!/bin/bash
# e2e.sh <model> <ids> <label>
B=/home/socol/oc-tmp/wt/bin
E=APEIRON_F32_CARRIER=1
cd /home/socol/oc-tmp/wt/worktree-agent-APEIRON-W7-T03
flock -x -w 900 /home/socol/oc-tmp/w6-host.lock timeout 540 $B/engine_profile-w7t03-after --model $1 --ids $2 --pp 200 --tg 8 --reps 7 \
  --arm "before:EXE=$B/engine_profile-04d2060,$E" --arm "same:EXE=$B/engine_profile-w7t03-base,$E" --arm "after:EXE=$B/engine_profile-w7t03-after,$E" \
  --label $3 --out /home/socol/oc-tmp/w7t03/ab > /home/socol/oc-tmp/w7t03/e2e-$3.log 2>&1
echo "exit $?" >> /home/socol/oc-tmp/w7t03/e2e-$3.log
