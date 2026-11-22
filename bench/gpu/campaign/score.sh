#!/bin/bash
# Score a pack against the 8B base on the GPU oracle: the 146-window wikitext-2 stream, the
# same windows as the 0.6B parity scan. Usage: score.sh <pack.nqp> [first:last]
exec >> /workspace/score-8b.log 2>&1
set -x
[ $# -ge 1 ] || { echo "usage: $0 PACK [FIRST:LAST]"; exit 2; }
PACK=$1
RANGE=${2:-0:146}
. /workspace/venv/bin/activate
cd /workspace
pip -q install transformers==4.55.4 accelerate 2>/dev/null
python /workspace/bench/gpu/1795343640-nq-100-parity-logs/scan_pack.py \
  "$PACK" --base /workspace/models/Qwen3-8B-Base --dtype bfloat16 --windows "$RANGE"
rc=$?
echo "== score rc=$rc $(date -u)"
gcloud storage cp /workspace/score-8b.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -2
exit $rc
