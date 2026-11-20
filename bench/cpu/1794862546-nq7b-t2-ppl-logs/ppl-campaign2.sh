#!/bin/bash
# campaign 2: Q4_K_M then NQ pack, same token stream, ctx 512.
set -x
cd /workspace/apeiron
export APEIRON_THREADS=23
M=/workspace/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
TOK=/workspace/models/tokenizer.json
DATA=/workspace/models/wiki.test.raw
CTX=${CTX:-512}
TAG=${TAG:-ctx512}

python3 /workspace/src/runner.py ./target/release/examples/ppl \
  --model "$M" --tokenizer "$TOK" --data "$DATA" \
  --tokens 2048 --skip 0 --ctx "$CTX" --head-batch 16 --kernel q8act \
  > /workspace/ppl-q4-$TAG.log 2>&1
echo "Q4 $?"
python3 /workspace/src/runner.py ./target/release/examples/ppl \
  --model "$M" --pack /workspace/nq7b-t2.nqp --tokenizer "$TOK" --data "$DATA" \
  --tokens 2048 --skip 0 --ctx "$CTX" --head-batch 16 --kernel q8act \
  > /workspace/ppl-nq-$TAG.log 2>&1
echo "NQ $?"
cat /workspace/ppl-q4-$TAG.log
cat /workspace/ppl-nq-$TAG.log
