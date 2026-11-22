#!/bin/bash
# Fetch the 8B campaign's inputs: the HF checkpoint and tokenizer, the reference's calibration
# text, and a bf16 GGUF for the teacher. Idempotent; status and log go to the bucket.
exec >> /workspace/fetch.log 2>&1
set -ex
ROOT=/workspace
GCS=gs://sanguine-checkpoints-scribe-461316/nq/l4
BUCKET=gs://sanguine-checkpoints-scribe-461316/nq
MODEL=$ROOT/models/Qwen3-8B-Base
GGUF=$ROOT/q3-8b-base-bf16.gguf
status() { echo "$*" > $ROOT/fetch.status; gcloud storage cp $ROOT/fetch.status $GCS/ 2>/dev/null; }
mkdir -p $MODEL $ROOT/wikitext2
status "start $(date -u)"

if [ ! -x $ROOT/venv/bin/pip ]; then
  rm -rf $ROOT/venv
  python3 -m venv $ROOT/venv
fi
. $ROOT/venv/bin/activate
pip -q install --upgrade pip
# Conversion reads safetensors; the CPU wheel is enough and is a tenth of the CUDA wheel.
if ! python -c 'import torch' 2>/dev/null; then
  pip -q install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip -q install "huggingface_hub[cli]" safetensors numpy sentencepiece transformers
status "pip done $(date -u)"

if [ ! -f $MODEL/model.safetensors.index.json ]; then
  # The console script is `hf` on current huggingface_hub; the module call covers both.
  python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-8B-Base', local_dir='$MODEL')"
fi
if [ ! -f $ROOT/wikitext2/wiki.train.raw ]; then
  curl -sSfL https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip \
    -o $ROOT/wikitext2.zip
  unzip -oq $ROOT/wikitext2.zip -d $ROOT/wikitext2
  # The zip expands one directory deep; the campaign paths are flat.
  mv $ROOT/wikitext2/wikitext-2-raw/*.raw $ROOT/wikitext2/
  rmdir $ROOT/wikitext2/wikitext-2-raw
fi
status "model and text done $(date -u)"

if [ ! -f $GGUF ]; then
  [ -d $ROOT/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp $ROOT/llama.cpp
  python3 $ROOT/llama.cpp/convert_hf_to_gguf.py $MODEL --outtype bf16 --outfile $GGUF
fi
sha256sum $GGUF > $GGUF.sha256
ls -l $GGUF
status "done $(date -u)"
gcloud storage cp $ROOT/fetch.log $GCS/fetch.log 2>&1 | tail -2
touch $ROOT/fetch.done