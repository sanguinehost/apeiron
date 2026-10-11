#!/usr/bin/env bash
set -euo pipefail
cd ~/apeiron/proof/corpus
LEAN_NUM_THREADS=1 lake env lean --root ~/apeiron/.held-control ~/apeiron/.held-control/ApeironWord.lean -o ~/apeiron/.held-control/ApeironWord.olean
LEAN_NUM_THREADS=1 lake env bash -c 'LEAN_PATH=~/apeiron/.held-control:$LEAN_PATH lean --root ~/apeiron/.held-control ~/apeiron/.held-control/LocatorSpec.lean'
