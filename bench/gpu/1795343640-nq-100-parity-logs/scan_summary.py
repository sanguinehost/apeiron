#!/usr/bin/env python3
"""Recompute the receipt's PPL rows from the committed campaign logs.

Each scan prints one line of 146 per-window mean NLLs. Tail-free is the mean
over windows whose NLL is at most 8 nats, exponentiated.

usage: python3 scan_summary.py
"""
import math
import pathlib
import sys

ROWS = [
    ("base f32", "window_scan.log", "base windows"),
    ("reference 1.00", "window_scan.log", "ref windows"),
    ("ours 1.00", "window_scan.log", "ours windows"),
    ("reference 0.80", "window_scan_080.log", "ref windows"),
    ("ours 0.80 pre-KD", "scan_pre080.log", "windows"),
    ("ours 0.80", "scan_post080.log", "windows"),
    ("ours 1.00 pre-KD", "scan_prekd.log", "windows"),
]


def windows(path: pathlib.Path, prefix: str):
    for line in path.read_text().splitlines():
        if line.startswith(prefix + " "):
            vals = [float(x) for x in line[len(prefix):].split()]
            if len(vals) == 146:
                return vals
    raise SystemExit(f"{path}: no {prefix!r} line with 146 windows")


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    print(f"{'row':20s} {'all PPL':>9s} {'tail-free':>10s} {'>8':>3s}  worst")
    for name, fn, prefix in ROWS:
        w = windows(here / fn, prefix)
        all_ppl = math.exp(sum(w) / len(w))
        kept = [x for x in w if x <= 8.0]
        tail_ppl = math.exp(sum(kept) / len(kept))
        worst = sorted(range(len(w)), key=lambda i: -w[i])[:3]
        print(f"{name:20s} {all_ppl:9.4f} {tail_ppl:10.4f} {len(w) - len(kept):3d}  "
              + ", ".join(f"w{i}={w[i]:.2f}" for i in worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
