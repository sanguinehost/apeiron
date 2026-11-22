#!/bin/bash
# Layer-0 preflight: fit block 0 at the campaign's tape and rate on the device path, write its
# one-layer pack over the dense skeleton, and score 8 tokens of wikitext-2. A refusal here stops
# the campaign before the full wall. Run as root from /workspace.
exec >> /workspace/preflight-8b.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
export APEIRON_NQ_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_LOOP=/workspace/loop-8b
export APEIRON_NQ_TAG=nq-100
export APEIRON_NQ_BLOCKS=0..1
export APEIRON_NQ_SAMPLES=96
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_BATCH=4
export APEIRON_NQ_PACK_ONLY=0
export APEIRON_NQ_PACK=/workspace/loop-8b/nq-100/pack-l0.nqp
mkdir -p "$APEIRON_NQ_LOOP/nq-100"

echo "== preflight layer 0 $(date -u)"
cargo test --release --locked --features gpu --test learn_block_tune -- \
  --ignored --exact block_loop --nocapture
fit=$?
echo "== preflight fit rc=$fit $(date -u)"
[ $fit -eq 0 ] && [ -s "$APEIRON_NQ_PACK" ] || exit 1

cargo build --release --locked --example ppl || exit 1
./target/release/examples/ppl \
  --model "$APEIRON_NQ_MODEL" \
  --tokenizer /workspace/models/Qwen3-8B-Base/tokenizer.json \
  --data /workspace/wikitext2/wiki.test.raw \
  --pack "$APEIRON_NQ_PACK" --tokens 8 --ctx 8 --partial-window
ppl=$?
echo "== preflight ppl rc=$ppl $(date -u)"
gcloud storage cp "$APEIRON_NQ_PACK" gs://sanguine-checkpoints-scribe-461316/nq/packs/q3-8b-l0.nqp 2>&1 | tail -1
gcloud storage cp /workspace/preflight-8b.log gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -1
exit $ppl
