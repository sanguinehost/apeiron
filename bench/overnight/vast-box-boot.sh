#!/usr/bin/env bash
# Bootstrap for a screen box: toolchain, the Q4 bar file, and the directories the driver uses.
# The source tree and the NQ inputs arrive from box B over ssh (1.9 GB), not from the laptop.
set -u
exec >>/workspace/boot.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates build-essential rsync
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal --default-toolchain 1.89.0
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
mkdir -p /workspace/models /workspace/results /workspace/apeiron
cd /workspace/models || exit 1
if [ ! -s Qwen3-8B-Q4_K_M.gguf ]; then
    curl -L --retry 5 --retry-delay 5 -o Qwen3-8B-Q4_K_M.gguf \
        https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
fi
stat -c '%n %s' Qwen3-8B-Q4_K_M.gguf >/workspace/results/q4.size
sha256sum Qwen3-8B-Q4_K_M.gguf >/workspace/results/q4.sha
echo BOOT_DONE
