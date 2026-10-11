#!/bin/bash
export APEIRON_ATTRIB_ATT=4 APEIRON_ATTRIB_REPS=30
D=/home/socol/oc-tmp/w7t03
B=/home/socol/oc-tmp/wt/bin
while pgrep -f "dense.sh $B/w7t03-test-base base 1.5b" >/dev/null; do sleep 2; done
$D/dense.sh $B/w7t03-test-new new 1.5b
$D/dense.sh $B/w7t03-test-base base q35
$D/dense.sh $B/w7t03-test-new new q35
$D/dense.sh $B/w7t03-test-base base 0.5b single
$D/dense.sh $B/w7t03-test-new new 0.5b single
echo ALLDONE > $D/gateb2.done
