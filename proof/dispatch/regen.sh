#!/bin/sh
# One whole deadline and one compatible Lean lane, including controls and descendants.
set -eu
DISPATCH_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dispatch_proof_deadline=$(($(date +%s) + 535))
if [ -n "${APEIRON_DISPATCH_DEADLINE_EPOCH:-}" ]; then
  case "$APEIRON_DISPATCH_DEADLINE_EPOCH" in
    *[!0-9]*) echo DISPATCH_BUDGET_PARSE_REJECT >&2; exit 1 ;;
  esac
  if [ "$APEIRON_DISPATCH_DEADLINE_EPOCH" -lt "$dispatch_proof_deadline" ]; then
    dispatch_proof_deadline=$APEIRON_DISPATCH_DEADLINE_EPOCH
  fi
fi
export APEIRON_DISPATCH_DEADLINE_EPOCH=$dispatch_proof_deadline
exec timeout --kill-after=5 540 flock -w 530 /tmp/opencode/apeiron-lane.lock \
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
  timeout --kill-after=5 530 env LEAN_NUM_THREADS=1 LEAN_PATH= \
  python3 "$DISPATCH_DIR/gate.py" "$@"
