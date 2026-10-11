#!/bin/bash
cd /home/socol/oc-tmp/wt/worktree-agent-APEIRON-W7-T03
APEIRON_F32_CARRIER=1 flock -x -w 900 /home/socol/oc-tmp/w6-host.lock timeout 540 /home/socol/oc-tmp/wt/bin/w7t03-test-new gemm_probe --ignored --nocapture > /home/socol/oc-tmp/w7t03/probe-new.log 2>&1
echo "exit $?" >> /home/socol/oc-tmp/w7t03/probe-new.log
