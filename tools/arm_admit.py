#!/usr/bin/env python3
"""Decide whether a residual arm may authorise a run or a claim.

The measured arms in `bench/cpu/1794896874-nq8b-depth-arms.csv` are the only numeric
inputs. Each arm is classified by how far it leaves the measured domain:

  decision   an exact repeat of a measured arm (same layers, kinds and plane count). The only
             class that may carry a product claim.
  control    exactly one axis outside the measured domain (a new block, or a new plane
             count on a measured block), with a registered reading. It may run; it may
             not be quoted as a product result.
  refuse     two axes outside the domain, or a plane count whose removal fraction is an
             anchor. AGENTS.md: an anchor cannot authorise a run or a claim.

The one-plane case is checked against the source pack, not argued: `--arm --pack` counts how often
plane 1's sign differs from plane 0's. A sequential fit makes plane 1 cancel plane 0 wherever plane
0 overshot, so a high disagreement means the first plane is not a nested prefix and its 62.2%
energy share does not describe it alone.

Exits 1 when a proposed arm is refused. `--audit` replays the recorded arms and exits 0.
"""

import argparse
import csv
import math
import os
import struct
import sys

import numpy as np

BASE_PPL = 12.189530
# Energy removed by a plane pair, from the stored row scales. This is an `anchor` for one
# plane: the scales are an energy decomposition of the pair, not a nested code.
REMOVAL = {0: 0.0, 1: 0.622, 2: 0.857}
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARMS = os.path.join(HERE, os.pardir, "bench", "cpu", "1794896874-nq8b-depth-arms.csv")
KINDS = ["q", "k", "v", "o", "gate", "up", "down"]


def read_arms(path):
    with open(path, newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            row["order"] = int(row["order"])
            row["planes"] = int(row["planes"])
            row["energy_share"] = float(row["energy_share"])
            row["nats"] = float(row["nats"])
            row["ppl"] = float(row["ppl"])
            row["bw"] = float(row["bw"])
            row["layer_set"] = parse_layers(row["layers"])
            row["kind_set"] = parse_kinds(row["kinds"])
            rows.append(row)
    return sorted(rows, key=lambda r: r["order"])


def parse_layers(spec):
    if spec in ("", "none"):
        return set()
    layers = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return layers


def parse_kinds(spec):
    if spec == "all":
        return set(KINDS)
    if spec in ("", "none"):
        return set()
    return {k.strip() for k in spec.split("+") if k.strip()}


def sensitivity(row):
    removed = row["energy_share"] * REMOVAL[row["planes"]]
    return row["nats"] / removed if removed else float("nan")


def classify(prior, layers, kinds, planes, energy_share, disagreement=None):
    """Return (verdict, reasons, support). `prior` is the arms measured before the proposal."""
    reasons = []
    exact = [a for a in prior
             if a["planes"] == planes and a["layer_set"] == set(layers)
             and a["kind_set"] == set(kinds)]
    if exact:
        return "decision", reasons, exact

    planes_new = planes not in {a["planes"] for a in prior}
    # A block arm (energy < 1) with the same kinds and a superset of the layers is the
    # measured support for a subsection. The whole-model arm is not a block: its
    # sensitivity is an average over all depths, which is the quantity the depth arms
    # showed is not transferable.
    section_support = [a for a in prior
                       if a["energy_share"] < 1 and a["kind_set"] == set(kinds)
                       and layers <= a["layer_set"]]
    section_new = not section_support

    axes = []
    if planes_new:
        axes.append(f"plane count {planes} is not measured")
        if planes == 1:
            reasons.append("the 62.2% removal of plane 0 is an anchor read from the pair's scales")
    if section_new:
        axes.append(f"layers {sorted(layers)} / kinds {sorted(kinds)} leave every measured block")
    if disagreement is not None and planes == 1 and disagreement > 0.52:
        reasons.append(
            f"plane 1 differs from plane 0 on {disagreement:.2%} of the source weights; "
            "the first plane is not a nested prefix")
    if len(axes) > 1:
        return "refuse", axes + reasons, []
    if len(axes) == 1:
        return "control", axes + reasons, []
    return "control", reasons + ["no exact arm at this section set and plane count; a new allocation is a control"], section_support


def bound(prior, planes, energy_share):
    """Nats and PPL over the measured sensitivity range at this plane count."""
    same = [sensitivity(a) for a in prior if a["planes"] == planes and a["nats"] > 0]
    if not same:
        return None
    lo, hi = min(same), max(same)
    removed = energy_share * REMOVAL[planes]
    nats = (lo * removed, hi * removed)
    return nats, (BASE_PPL * math.exp(-nats[1]), BASE_PPL * math.exp(-nats[0])), (lo, hi)


def ppl_after(nats):
    return BASE_PPL * math.exp(-nats)


def open_pack(path):
    with open(path, "rb") as f:
        magic, version, header, layers, kinds, linears, extra = struct.unpack("<8sIIIIII", f.read(32))
        if magic != b"APNQPK01" or version != 3:
            raise SystemExit(f"{path}: want a version 3 pack")
        return dict(version=version, header=header, layers=layers, linears=linears)


def sign_disagreement(path, wanted_layers):
    """Plane-1 vs plane-0 sign disagreement over the requested sections."""
    meta = open_pack(path)
    differ = total = sections = 0
    with open(path, "rb") as f:
        at = meta["header"]
        for _ in range(meta["linears"]):
            f.seek(at)
            raw = f.read(36)
            layer, kind, out, inn, rank, u_words, v_words, k = struct.unpack_from("<IIIIIIII", raw)
            res_layers = struct.unpack_from("<I", raw, 32)[0]
            sizes = (out * u_words * 4, rank * v_words * 4, inn * 2, rank * 2, out * 2,
                     k * 4, out * k * 2, res_layers * out * v_words * 4, res_layers * out * 2)
            if res_layers == 2 and layer in wanted_layers:
                f.seek(at + 36 + sum(sizes[:7]))
                data = f.read(sizes[7])
                plane = out * v_words * 4
                a = np.frombuffer(data[:plane], "<u4")
                b = np.frombuffer(data[plane:], "<u4")
                x = (a ^ b).astype(np.uint32)
                x = x - ((x >> 1) & 0x55555555)
                x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
                x = (x + (x >> 4)) & 0x0F0F0F0F
                differ += int(((x * 0x01010101) >> 24).sum())
                total += v_words * out * 32
                sections += 1
            at += 36 + sum(sizes)
    if not sections:
        raise SystemExit(f"{path}: no two-plane sections in layers {sorted(wanted_layers)}")
    return differ / total, sections


def report_arms(arms):
    print(f"{'arm':22s} {'planes':>6s} {'energy':>8s} {'nats':>9s} {'PPL':>10s} {'nats/E0':>8s} {'x avg':>6s}")
    for a in arms:
        s = sensitivity(a)
        print(f"{a['arm']:22s} {a['planes']:6d} {a['energy_share']:8.5f} {a['nats']:9.6f} "
              f"{a['ppl']:10.6f} {s:8.4f} {s / 0.4777:6.2f}")


def audit(arms, pack):
    print("## Measured arms\n")
    report_arms(arms)
    print("\n## Replay\n")
    seen = []
    for a in arms:
        if a["order"] == 1:
            print(f"{a['arm']:18s} baseline")
            seen.append(a)
            continue
        if a["order"] == 2:
            verdict, reasons = "calibration", ["first measurement of the plane pair"]
        else:
            verdict, reasons, _ = classify(seen, a["layer_set"], a["kind_set"], a["planes"],
                                           a["energy_share"])
        print(f"{a['arm']:18s} {verdict:11s} {a['nats']:+.6f} nats  " + "; ".join(reasons))
        seen.append(a)
    print("\n## Recorded prediction that failed\n")
    r1 = next(a for a in arms if a["arm"].startswith("r1late"))
    print(f"{r1['arm']}: predictions table 11.441108 (0.063365 nats; a working note quoted "
          f"11.025591 with the plane factor dropped), measured {r1['ppl']:.6f}, "
          f"{r1['nats']:+.6f} nats. The replay refuses it on two axes.")
    if pack:
        d, n = sign_disagreement(pack, parse_layers("30-35"))
        print(f"plane-1 sign disagreement on layers 30-35: {d:.4%} over {n} sections "
              f"(the pair is a cancellation code; plane 0 alone is not half of it).")


def self_test(arms):
    late = next(a for a in arms if a["arm"].startswith("late 33"))
    early = next(a for a in arms if a["arm"].startswith("early"))
    r1 = next(a for a in arms if a["arm"].startswith("r1late"))
    prior = [a for a in arms if a["order"] < late["order"]]
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got}, want {want}")

    check("late2 as the first block arm", classify(prior, late["layer_set"], late["kind_set"],
                                                   late["planes"], late["energy_share"])[0], "control")
    trust = prior + [late, early]
    check("late1 (same sections, one plane)",
          classify(trust, late["layer_set"], late["kind_set"], 1, late["energy_share"])[0], "control")
    check("r1late (new sections and one plane)",
          classify(trust, r1["layer_set"], r1["kind_set"], r1["planes"], r1["energy_share"])[0],
          "refuse")
    # A control on the same 15 sections at the measured plane count is a decision.
    check("same sections at two planes",
          classify(trust, late["layer_set"], late["kind_set"], 2, late["energy_share"])[0], "decision")
    # Mutation: removing the measured late block leaves late1 with two new axes, so the
    # control becomes a refusal. A check that cannot fail carries no weight.
    without_late = [a for a in trust if a["arm"] != late["arm"]]
    check("late1 without its measured block",
          classify(without_late, late["layer_set"], late["kind_set"], 1, late["energy_share"])[0],
          "refuse")
    # The gate that matters: a one-plane arm that also leaves the block is refused.
    check("one plane on a new block",
          classify(trust, r1["layer_set"], r1["kind_set"], 1, r1["energy_share"])[0], "refuse")
    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print("  " + f)
        return 1
    print("self-test passed: late1 control; late1 without late2 refuse; r1late refuse; "
          "exact repeat decision")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--pack", help="the two-plane v3 pack for the sign check")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arm", help="a proposed arm's name")
    ap.add_argument("--layers", default="")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--planes", type=int, choices=(1, 2))
    ap.add_argument("--energy", type=float, help="the selected sections' share of total plane-0 energy")
    ap.add_argument("--disagreement", type=float, help="override the pack sign check")
    args = ap.parse_args()
    arms = read_arms(args.arms)

    if args.self_test:
        return self_test(arms)
    if args.audit:
        audit(arms, args.pack)
        return 0
    if not args.arm:
        ap.error("one of --audit, --self-test or --arm is required")

    layers, kinds = parse_layers(args.layers), parse_kinds(args.kinds)
    if args.planes is None or args.energy is None:
        ap.error("--arm needs --planes and --energy")
    disagreement = args.disagreement
    if disagreement is None and args.pack and args.planes == 1:
        disagreement, n = sign_disagreement(args.pack, layers)
    match = next((a for a in arms if a["arm"] == args.arm), None)
    prior = [a for a in arms if match is None or a["order"] < match["order"]]
    verdict, reasons, support = classify(prior, layers, kinds, args.planes, args.energy,
                                         disagreement)
    print(f"{args.arm}: {verdict}")
    for r in reasons:
        print(f"  {r}")
    if verdict == "decision":
        s = sensitivity(support[0])
        nats = s * args.energy * REMOVAL[args.planes]
        print(f"  interpolation on {support[0]['arm']}: {nats:.6f} nats, PPL {ppl_after(nats):.6f}")
    elif verdict == "control":
        b = bound(prior, args.planes, args.energy)
        if b:
            (lo, hi), (plo, phi), (slo, shi) = b
            print(f"  bound over measured sensitivity {slo:.4f}..{shi:.4f}: "
                  f"{lo:.6f}..{hi:.6f} nats, PPL {plo:.6f}..{phi:.6f}; a control, not a claim")
        else:
            print(f"  no positive measured arm at {args.planes} plane(s); "
                  "the run is a control with no numeric prediction")
    return 1 if verdict == "refuse" else 0


if __name__ == "__main__":
    sys.exit(main())
