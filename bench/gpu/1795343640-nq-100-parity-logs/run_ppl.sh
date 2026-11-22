#!/bin/bash
# Serve the 1.00 pack against the F16 teacher on the reference token stream.
cd /workspace/apeiron-nq || exit 1
SNAP=$(ls -d /workspace/hf/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*/ | head -1)
echo "== pack ppl $(date -u +%H:%M:%S)"
target/release/examples/ppl \
  --model /workspace/Qwen3-0.6B-Base.f16.gguf \
  --pack /workspace/loop-ref/nq-100/pack.nqp \
  --tokenizer "$SNAP/tokenizer.json" \
  --data /workspace/wikitext2-test.txt \
  --tokens 0 --ctx 2048 --head-batch 16 --kernel q8act
echo "== pack ppl rc=$? $(date -u +%H:%M:%S)"
echo "== fp16 control $(date -u +%H:%M:%S)"
target/release/examples/ppl \
  --model /workspace/Qwen3-0.6B-Base.f16.gguf \
  --tokenizer "$SNAP/tokenizer.json" \
  --data /workspace/wikitext2-test.txt \
  --tokens 0 --ctx 2048 --head-batch 16
echo "== fp16 rc=$? $(date -u +%H:%M:%S)"
