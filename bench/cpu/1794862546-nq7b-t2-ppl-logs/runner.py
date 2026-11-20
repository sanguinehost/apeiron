#!/usr/bin/env python3
"""Run a child, report its wall clock and peak RSS (ru_maxrss of children, KiB)."""
import resource
import subprocess
import sys
import time

t0 = time.monotonic()
r = subprocess.run(sys.argv[1:], check=False)
dt = time.monotonic() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"WALL {dt:.2f} s  PEAK_RSS_KB {ru.ru_maxrss}  EXIT {r.returncode}")
sys.exit(r.returncode)
