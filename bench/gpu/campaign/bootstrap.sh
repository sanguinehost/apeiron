#!/bin/bash
# The L4 campaign's boot step: kernel, Vulkan, Rust, the tree at the tested commit, a build,
# and the 8B inputs. Idempotent; every log is on the boot disk and the status goes to the bucket.
# Started by the instance's startup metadata; the run cap is the instance's max-run-duration.
exec >> /var/log/nq-bootstrap.log 2>&1
set -x
# The startup runner does not guarantee root's HOME; pin the rustup roots so the build finds the
# toolchain that `sudo -EH` uses (a boot with HOME=/home/lucas exits 127).
export HOME=/root
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y build-essential pkg-config libvulkan1 vulkan-tools unzip curl git tmux \
  python3-pip python3-venv

# The DL image ships nvidia-utils but not the full driver, so the ICD loads llvmpipe alone.
if ! dpkg -s nvidia-driver-580-server >/dev/null 2>&1; then
  apt-get install -y nvidia-driver-580-server
fi

if [ ! -x /root/.cargo/bin/cargo ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
. /root/.cargo/env

mkdir -p /workspace
gcloud storage cp gs://sanguine-checkpoints-scribe-461316/nq/boot/apeiron-latest.tgz /workspace/apeiron.tgz
tar -xzf /workspace/apeiron.tgz -C /workspace
cd /workspace
vulkaninfo --summary 2>&1 | grep -E "deviceName|deviceType|driverName" | head -6
cargo build --release --locked --features gpu --tests
echo "bootstrap rc=$? $(date -u)" > /workspace/bootstrap.status
gcloud storage cp /workspace/bootstrap.status \
  gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -2
gcloud storage cp /var/log/nq-bootstrap.log \
  gs://sanguine-checkpoints-scribe-461316/nq/l4/ 2>&1 | tail -2

# The inputs are the long pole; run them detached so a preemption does not lose the tree.
if [ ! -f /workspace/fetch.done ]; then
  setsid nohup bash /workspace/bench/gpu/campaign/fetch.sh > /dev/null 2>&1 < /dev/null &
fi
