#!/usr/bin/env python3
"""Recompute a v1 result record from its logs and check its hashes.

The record (`bench/results/<pack-sha12>.json`, `bench/PROTOCOL.md` §6) is a claim; the logs are the
evidence. This tool recomputes the pooled PPL of every arm from the per-window `mean nll` lines,
checks the pack and skeleton sha256, and exits 1 if any field disagrees.

Usage:
    python3 tools/bench-check.py --record bench/results/<sha>.json --logs bench/results/<sha>/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

LINE = re.compile(r"^ppl ([\d.]+) on (\d+) tokens.*?mean nll ([\d.]+)", re.M)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pooled(logs: list[Path]) -> tuple[float, int]:
    """exp(Σ nll_i / N) over the printed per-window mean nll and scored counts."""
    total, count = 0.0, 0
    for path in logs:
        matches = LINE.findall(path.read_text(encoding="utf-8"))
        if len(matches) != 1:
            raise SystemExit(f"{path}: expected one ppl line, found {len(matches)}")
        _ppl, n, nll = matches[0]
        total += float(nll) * int(n)
        count += int(n)
    if count == 0:
        raise SystemExit("no scored positions")
    return math.exp(total / count), count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, type=Path)
    ap.add_argument("--logs", required=True, type=Path)
    args = ap.parse_args()

    rec = json.loads(args.record.read_text(encoding="utf-8"))
    if rec.get("protocol") != 1:
        print(f"unsupported protocol {rec.get('protocol')!r}", file=sys.stderr)
        return 1
    bad = 0
    for key in ("pack_sha256", "skeleton_sha256"):
        path = rec.get(key + "_path")
        if not path or not Path(path).is_file():
            print(f"{key}: no path in record or file missing", file=sys.stderr)
            bad = 1
            continue
        got = sha256(Path(path))
        if got != rec.get(key):
            print(f"{key}: {got[:16]} != {rec[key][:16]}", file=sys.stderr)
            bad = 1

    for arm, claimed in rec.get("ppl", {}).items():
        logs = sorted((args.logs / arm).glob("*.log"))
        if not logs:
            print(f"arm {arm}: no logs", file=sys.stderr)
            bad = 1
            continue
        got, n = pooled(logs)
        if abs(got - float(claimed)) > 1e-6:
            print(f"arm {arm}: pooled {got:.6f} != record {float(claimed):.6f}", file=sys.stderr)
            bad = 1
        else:
            print(f"OK arm {arm}: {got:.6f} over {n} positions, {len(logs)} windows")

    base = rec.get("ppl", {})
    for arm, ratio in rec.get("ratios", {}).items():
        num, den = arm.split("/")
        if num not in base or den not in base or base[den] == 0:
            print(f"ratio {arm}: missing arm", file=sys.stderr)
            bad = 1
            continue
        got = base[num] / base[den]
        if abs(got - float(ratio)) > 1e-6:
            print(f"ratio {arm}: {got:.6f} != record {float(ratio):.6f}", file=sys.stderr)
            bad = 1
        else:
            print(f"OK ratio {arm}: {got:.6f}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
