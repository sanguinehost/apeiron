#!/usr/bin/env bash
set -euo pipefail
cd /home/socol/Workspace/apeiron/proof/corpus
LEAN_NUM_THREADS=1 lake env lean --root /home/socol/Workspace/apeiron/.held-control /home/socol/Workspace/apeiron/.held-control/ApeironWord.lean -o /home/socol/Workspace/apeiron/.held-control/ApeironWord.olean
LEAN_NUM_THREADS=1 lake env bash -c 'LEAN_PATH=/home/socol/Workspace/apeiron/.held-control:$LEAN_PATH lean --root /home/socol/Workspace/apeiron/.held-control /home/socol/Workspace/apeiron/.held-control/LocatorSpec.lean'
