#!/bin/bash
export APEIRON_ATTRIB_ATT=4 APEIRON_ATTRIB_REPS=30
D=/home/socol/oc-tmp/w7t03
B=/home/socol/oc-tmp/wt/bin
for mode in pooled single; do
 for m in 0.5b 1.5b q35; do
  for v in base new; do
   s=""; [ $mode = single ] && s=single
   $D/dense.sh $B/w7t03-test-$v $v $m $s
  done
 done
done
echo ALLDONE > $D/gateb.done
