#!/usr/bin/env python3
"""Measure the 8B block loop from its log.

The unit is one layer: seven slots x three phases (`nonfact`, `ADMM`, `fact`), then the
teacher pass and the advance. A layer is *full* when the log has all 21 units and a
`round total`; a resumed layer is partial. The script prints the measured full layers,
their mean and spread, the per-slot phase means, and a projection for the remaining
layers that names the number of full layers it is based on. No projection is printed
before three full layers exist: a one- or two-layer mean is a sample, not a rate.

Usage:
    timing.py LOG [--first 1] [--last 35] [--json]
"""

import argparse
import json
import re
import statistics
import sys

SLOTS = [
    "self_attn.q_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.k_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
PHASES = ["nonfact", "ADMM", "fact"]

RE_UNIT = (
    ("nonfact", re.compile(r"^BT l(\d+) (\S+) nonfact \d+ steps ([\d.]+) s")),
    ("ADMM", re.compile(r"^BT l(\d+) (\S+) ADMM weight recon error: .*ADMM time=([\d.]+)s")),
    ("fact", re.compile(r"^BT l(\d+) (\S+) fact \d+ steps ([\d.]+) s")),
)
RE_TEACHER = re.compile(r"^BT l(\d+) teacher ([\d.]+) s")
RE_ADVANCE = re.compile(r"^BT l(\d+) advanced ([\d.]+) s")
RE_ROUND = re.compile(r"^BT l(\d+) round total ([\d.]+) s")


def parse(path):
    layers = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            for phase, pat in RE_UNIT:
                m = pat.search(line)
                if m:
                    l, slot, s = int(m.group(1)), m.group(2), float(m.group(3))
                    layers.setdefault(l, {"units": {}, "teacher": None, "advance": None, "round": None})
                    layers[l]["units"][(slot, phase)] = s
                    continue
            m = RE_TEACHER.search(line)
            if m:
                l, s = int(m.group(1)), float(m.group(2))
                layers.setdefault(l, {"units": {}, "teacher": None, "advance": None, "round": None})
                layers[l]["teacher"] = s
                continue
            m = RE_ADVANCE.search(line)
            if m:
                l, s = int(m.group(1)), float(m.group(2))
                layers.setdefault(l, {"units": {}, "teacher": None, "advance": None, "round": None})
                layers[l]["advance"] = s
                continue
            m = RE_ROUND.search(line)
            if m:
                l, s = int(m.group(1)), float(m.group(2))
                layers.setdefault(l, {"units": {}, "teacher": None, "advance": None, "round": None})
                layers[l]["round"] = s
    return layers


def is_full(rec):
    return rec["round"] is not None and all((slot, phase) in rec["units"] for slot in SLOTS for phase in PHASES)


def unit_sum(rec):
    return sum(rec["units"].values())


def fmt(x):
    return "—" if x is None else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=35)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    layers = parse(args.log)
    if not layers:
        sys.exit(f"{args.log}: no BT layer lines")

    full = [l for l in sorted(layers) if is_full(layers[l])]
    partial = [l for l in sorted(layers) if l not in full and layers[l]["units"]]

    print(f"log: {args.log}")
    print(f"layers in log: {min(layers)}..{max(layers)}; full: {full or 'none'}; partial: {partial or 'none'}")
    print()
    print("| layer | units | sum units s | teacher s | advance s | round total s | residual s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for l in sorted(layers):
        r = layers[l]
        n = len(r["units"])
        t = unit_sum(r)
        teacher, advance, total = r["teacher"], r["advance"], r["round"]
        residual = None if total is None else total - t - (teacher or 0.0) - (advance or 0.0)
        print(f"| {l} | {n}/21 | {t:.1f} | {fmt(teacher)} | {fmt(advance)} | {fmt(total)} | {fmt(residual)} |")

    if not full:
        print("\nno full campaign layer measured yet; no layer total or projection is printed.")
        return

    totals = [layers[l]["round"] for l in full]
    mean = statistics.fmean(totals)
    print()
    print(f"full layers: n={len(full)}; round total mean {mean:.1f} s ({mean / 60:.2f} min), "
          f"min {min(totals):.1f}, max {max(totals):.1f}"
          + (f", stdev {statistics.stdev(totals):.1f}" if len(totals) >= 2 else ""))

    # Per-slot, per-phase means over the full layers, for the current-layer remainder.
    means = {}
    for slot in SLOTS:
        for phase in PHASES:
            vals = [layers[l]["units"][(slot, phase)] for l in full]
            means[(slot, phase)] = statistics.fmean(vals)
    print()
    print("| slot | nonfact s | ADMM s | fact s |")
    print("|---|---:|---:|---:|")
    for slot in SLOTS:
        print(f"| {slot} | {means[(slot, 'nonfact')]:.1f} | {means[(slot, 'ADMM')]:.1f} | {means[(slot, 'fact')]:.1f} |")

    # The current layer is the highest layer with units but no full round. Its missing
    # units are costed at the full-layer means; this is a projection, not a measurement.
    current = partial[-1] if partial else None
    projection = None
    remaining = None
    if len(full) < 3:
        print()
        print(f"n={len(full)} full campaign layers; no layer-total projection is printed "
              "(three full layers are required before a campaign ETA).")
    else:
        remaining = 0.0
        if current is not None:
            have = layers[current]["units"]
            missing = [(slot, phase) for slot in SLOTS for phase in PHASES if (slot, phase) not in have]
            remaining = sum(means[u] for u in missing)
            print()
            print(f"current layer l{current:02d}: {len(have)}/21 units; "
                  f"remaining units costed at the n={len(full)} means: {remaining:.1f} s")
        after = [l for l in range(args.first, args.last + 1) if l not in full and l != current]
        projection = remaining + len(after) * mean
        lo = remaining + len(after) * min(totals)
        hi = remaining + len(after) * max(totals)
        print(f"projection over layers {args.first}..{args.last}: {len(after)} full layers after "
              f"l{current if current is not None else 0:02d} + {remaining:.1f} s current remainder "
              f"= {projection:.0f} s ({projection / 3600:.2f} h); "
              f"range {lo / 3600:.2f}..{hi / 3600:.2f} h from the n={len(full)} spread")
        print("projection, not a measurement: it assumes later layers cost the full-layer mean.")

    if args.json:
        out = {
            "log": args.log,
            "full": full,
            "partial": partial,
            "totals": totals,
            "mean_s": mean,
            "min_s": min(totals),
            "max_s": max(totals),
            "current": current,
            "current_remaining_s": remaining,
            "projection_s": projection,
        }
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
