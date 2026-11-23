#!/bin/bash
# The tape release and `set_t` equality tests, as one log artifact. Run as root from /workspace.
exec > /workspace/release-tape-tests.log 2>&1
set -x
. /root/.cargo/env
cd /workspace
date -u
rc=0
for t in \
  gpu::nn::train::block::tests::release_tape_then_set_t_matches_a_fresh_tuner \
  gpu::nn::train::front::tests::release_tape_then_set_t_matches_a_fresh_front \
  gpu::nn::train::block::tests::set_t_matches_a_fresh_tuner \
  gpu::nn::train::front::tests::set_t_matches_a_fresh_front
do
  out=$(cargo test --release --locked --features gpu --lib -- --exact "$t" --nocapture 2>&1)
  r=$?
  printf '%s\n' "$out" | grep -E "^test |^set_t|^release_tape|test result"
  [ "$r" -eq 0 ] || rc=1
done
date -u
echo "== release tests rc=$rc"
exit $rc
