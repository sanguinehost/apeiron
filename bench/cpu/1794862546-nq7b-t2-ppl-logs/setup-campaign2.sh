#!/bin/bash
# campaign 2 box setup: toolchain + model downloads
set -x
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/models /workspace/cal /workspace/src
apt-get update -qq && apt-get install -y -qq build-essential curl ca-certificates pkg-config >/tmp/apt.log 2>&1
echo "APT_DONE $?"
curl -sSf https://sh.rustup.rs -o /tmp/rustup.sh && sh /tmp/rustup.sh -y --profile minimal --default-toolchain stable >/tmp/rustup.log 2>&1
echo "RUSTUP_DONE $?"
export PATH=/root/.cargo/bin:$PATH
rustc -Vv
BASE=https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main
curl -sSL -o /workspace/models/Qwen2.5-7B-Instruct-f16.gguf "$BASE/Qwen2.5-7B-Instruct-f16.gguf" &
P1=$!
curl -sSL -o /workspace/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf "$BASE/Qwen2.5-7B-Instruct-Q4_K_M.gguf" &
P2=$!
wait $P1; echo "F16_DL $?"
wait $P2; echo "Q4_DL $?"
sha256sum /workspace/models/*.gguf
