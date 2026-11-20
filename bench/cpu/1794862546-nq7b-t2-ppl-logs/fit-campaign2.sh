#!/bin/bash
# campaign 2: full 28-layer T=2 fit, activation-diagonal preconditioner.
set -x
cd /workspace/apeiron
export PATH=/root/.cargo/bin:$PATH
cargo build --release --locked --example nq_fit > /workspace/build2.log 2>&1
echo "REBUILD $?"
sha256sum examples/nq_fit.rs
export APEIRON_THREADS=23
date -u +%s > /workspace/fit.start
LAYERS=$(seq -s, 0 27)
./target/release/examples/nq_fit \
  --model /workspace/models/Qwen2.5-7B-Instruct-f16.gguf \
  --out /workspace/nq7b-t2.nqp \
  --layers "$LAYERS" \
  --kinds q,k,v,o,gate,up,down \
  --iters 2 \
  --bits-ppm 1100000 \
  --diag-dir /workspace/cal/diag-c6x29 \
  --diag-tag c6x29 > /workspace/fit.log 2>&1
echo "FIT $?"
date -u +%s > /workspace/fit.end
tail -3 /workspace/fit.log
