#!/usr/bin/env python3
"""Stored rate of the 0.6B NanoQuant pack: the per-kind rank table and its alternatives.

The pack's section stores `U` (`out × ceil(rank/32)` u32), `V` (`rank × ceil(in/32)` u32), and the
three bf16 scale vectors `s_pre` (`in`), `s_mid` (`rank`), `s_post` (`out`):

    stored_bits(out, in, r) = 32*(out*ceil(r/32) + r*ceil(in/32)) + 16*(in + r + out)

A dense sign-residual layer adds `32*out*ceil(in/32) + 16*out`; a salient column adds
`32 + 16*out`. The 0.6B geometry is 28 layers of q 2048x1024, k 1024x1024, v 1024x1024,
o 1024x2048, gate 3072x1024, up 3072x1024, down 1024x3072.

`rank_for_bits` (`src/learn/nanoquant.rs`) solves `r = a*b*ppm/(a+b) - 16`, floored to 32; the
`- 16` accounts for `16*(a+b)` of scales and omits the `16*r` of `s_mid`, so the ranks it returns
land about 0.02 b/w below the requested budget at this geometry. The `--fit` mode searches +32
steps per kind for the table that makes the whole pack `--ppm` within `--tol` of the target.

Usage:
  nq_rate.py                 # the tables below
  nq_rate.py --fit 1174000   # the 1.174 table within 0.002
"""

import argparse
import itertools
import math

LAYERS = 28
KINDS = [
    ("q", 2048, 1024),
    ("k", 1024, 1024),
    ("v", 1024, 1024),
    ("o", 1024, 2048),
    ("gate", 3072, 1024),
    ("up", 3072, 1024),
    ("down", 1024, 3072),
]
# v3 (`/tmp/opencode/nq06-v3.nqp`, 53,815,472 B, 196 sections).
V3 = {"q": 640, "k": 480, "v": 480, "o": 640, "gate": 736, "up": 736, "down": 736}


def bits(kind, rank):
    _, out, inn = kind
    return 32 * (out * math.ceil(rank / 32) + rank * math.ceil(inn / 32)) + 16 * (inn + rank + out)


def residual_bits(kind):
    """One dense sign-residual layer: `out*in` sign bits plus one bf16 row scale."""
    _, out, inn = kind
    return 32 * out * math.ceil(inn / 32) + 16 * out


def rank_for_bits(kind, ppm):
    """`calculate_ranks` at `ppm` millionths of a bit per weight (`min_rank = 32`)."""
    _, out, inn = kind
    per = (out * inn * ppm) // (1_000_000 * (out + inn))
    return max((max(per - 16, 0) // 32) * 32, 32)


def total(ranks, header=32):
    """Payload bits, dense weights, and the pack file: `write` uses 32-B sections for version 2
    (no residual) and 36-B sections for version 3; `v3` on disk is version 3."""
    payload = sum(bits(k, ranks[k[0]]) for k in KINDS) * LAYERS
    weights = sum(o * i for _, o, i in KINDS) * LAYERS
    size = payload // 8 + 32 + 196 * header
    return payload, weights, size


def report(name, ranks, header=32):
    payload, weights, size = total(ranks, header)
    print(f"{name}: " + " ".join(f"{k[0]} {ranks[k[0]]}" for k in KINDS))
    print(
        f"  payload {payload} bits = {payload // 8} B; pack {size} B "
        f"({header}-B sections); "
        f"{payload / weights:.6f} b/w payload, {size * 8 / weights:.6f} b/w file"
    )
    for k in KINDS:
        b = bits(k, ranks[k[0]])
        print(
            f"  {k[0]:4} {k[1]}x{k[2]} r {ranks[k[0]]:4}: {b:9} bits, "
            f"{b / (k[1] * k[2]):.6f} b/w"
        )


def fit(ppm, tol):
    """Per-kind +32 steps around `rank_for_bits` whose total is within `tol` of `ppm`."""
    weights = sum(o * i for _, o, i in KINDS) * LAYERS
    target = ppm / 1e6 * weights
    base = {k[0]: rank_for_bits(k, ppm) for k in KINDS}
    best = []
    for steps in itertools.product(range(3), repeat=len(KINDS)):
        ranks = {k[0]: base[k[0]] + 32 * s for k, s in zip(KINDS, steps)}
        payload = sum(bits(k, ranks[k[0]]) for k in KINDS) * LAYERS
        err = abs(payload / weights - ppm / 1e6)
        if err <= tol:
            best.append((err, steps, ranks, payload))
    best.sort(key=lambda x: (x[0], sum(x[1])))
    print(f"fit {ppm / 1e6:.3f} b/w within {tol}: {len(best)} table(s)")
    for err, steps, ranks, payload in best[:8]:
        print(
            f"  err {err:.6f}: "
            + " ".join(f"{k[0]}:{ranks[k[0]]}" for k in KINDS)
            + f"  payload {payload // 8} B, {payload / weights:.6f} b/w"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=0, help="ppm to solve for (e.g. 1174000)")
    ap.add_argument("--tol", type=float, default=0.002)
    args = ap.parse_args()
    if args.fit:
        fit(args.fit, args.tol)
        return
    report("v3 (served)", V3, header=36)
    ranks = {k[0]: rank_for_bits(k, 1_174_000) for k in KINDS}
    report("rank_for_bits at 1.174", ranks)
    print(f"  rank_for_bits reports {total(ranks)[0] / total(ranks)[1]:.6f} b/w, not the budget")
    # The note's table: one more 32-block on each attention kind, FFN at its per-kind budget.
    chosen = {"q": 800, "k": 608, "v": 608, "o": 800, "gate": 864, "up": 864, "down": 864}
    report("chosen 1.174 (attention +32)", chosen)
    payload, weights, _ = total(V3)
    residual = sum(residual_bits(k) for k in KINDS) * LAYERS
    print(
        f"one residual layer on all 196 linears: {residual} bits = "
        f"{residual / weights:.6f} b/w; "
        f"affordable above v3 at 1.174 b/w: {(1.174 * weights - payload) / residual:.4f} layers"
    )


if __name__ == "__main__":
    main()
