#!/bin/bash
# Wait for the campaign-2 fit to exit cleanly, then run both PPL arms.
set -x
while pgrep -f "examples/nq_fit" > /dev/null; do sleep 20; done
date -u
tail -3 /workspace/fit.log
if ! grep -q "^FIT 0$" /workspace/fit-console.log; then
  echo "FIT_DID_NOT_EXIT_ZERO"
  exit 1
fi
bash /workspace/src/ppl-campaign2.sh
date -u
