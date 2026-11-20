#!/bin/bash
# campaign 2: verify diagonals in place, build the drivers.
set -x
( cd /workspace/cal/diag-c6x29 && sha256sum -c /workspace/src/diag.sha256 > /workspace/diag-check.log 2>&1; echo "DIAG_CHECK $?" )
grep -c ': OK$' /workspace/diag-check.log
cd /workspace/apeiron
export PATH=/root/.cargo/bin:$PATH
export CARGO_TERM_COLOR=never
cargo build --release --locked --example nq_fit --example ppl > /workspace/build.log 2>&1
echo "BUILD $?"
tail -3 /workspace/build.log
ls -l /workspace/apeiron/target/release/examples/nq_fit /workspace/apeiron/target/release/examples/ppl
