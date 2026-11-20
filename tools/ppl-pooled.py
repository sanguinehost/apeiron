#!/usr/bin/env python3
"""Pool two PPL arms' windows into one estimate, and print the per-window ratios.

Wikitext-2 perplexity is a mean over scored positions, so the estimate over several ctx-sized windows
is `exp(Σ_i nll_i · n_i / Σ_i n_i)`, not a mean of the windows' PPLs. `examples/ppl` prints one
`ppl <value> … mean nll <value> … in <s> s` line per window with equal counts (1,023 at the
`--second-half` protocol), so pooling is the mean of the printed per-window `mean nll` values,
weighted by the counts when they differ.

Per-window ratios are printed too. They are not the criterion: the criterion is the pooled ratio over
a named span. A per-window ratio is a statement about that window and its spread is the
window-to-window variance, which on five windows is 1.47-1.65 for this pair.

Usage:
    python3 tools/ppl-pooled.py --nq W*.log --q4 Q4W*.log [--label 'windows 1-5']
    python3 tools/ppl-pooled.py --nq W1.log W2.log ... --q4 Q4W1.log ... \
        --spans 2048 2048 2048 2048 2048 11776 ...      # stratified, full-file estimate
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

LINE = re.compile(r"^ppl ([\d.]+) on (\d+) tokens.*?mean nll ([\d.]+)", re.M)
# Two-sided 97.5 % points of Student's t, df 1..30; the normal point above 30.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
        30: 2.042}


def windows(paths: list[Path]) -> list[tuple[str, float, int]]:
    """[(name, mean nll, scored positions)] in the order the files are given."""
    out = []
    for path in paths:
        matches = LINE.findall(path.read_text(encoding="utf-8"))
        if not matches:
            raise SystemExit(f"{path}: no `ppl … mean nll` line")
        ppl, tokens, nll = matches[-1]
        out.append((path.stem, float(nll), int(tokens)))
    return out


def pooled(rows: list[tuple[str, float, int]]) -> float:
    total = sum(n for _, _, n in rows)
    return sum(nll * n for _, nll, n in rows) / total


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nq", nargs="+", type=Path, required=True)
    ap.add_argument("--q4", nargs="+", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--spans",
        nargs="+",
        type=float,
        default=None,
        help="tokens of text each window represents, for a stratified estimate; "
        "equal spans reproduce the unweighted row",
    )
    args = ap.parse_args(argv)

    nq, q4 = windows(args.nq), windows(args.q4)
    if len(nq) != len(q4):
        print(f"warning: {len(nq)} NQ windows against {len(q4)} Q4 windows")
    if args.label:
        print(f"span: {args.label}")
    print(f"{'#':>2s} {'nq window':28s} {'nq nll':>8s} {'q4 nll':>8s} {'nq ppl':>9s} {'q4 ppl':>9s} {'ratio':>8s}")
    pairs = list(zip(nq, q4))
    for i, (a, b) in enumerate(pairs, 1):
        gap = a[1] - b[1]
        print(f"{i:2d} {a[0][:28]:28s} {a[1]:8.6f} {b[1]:8.6f} "
              f"{math.exp(a[1]):9.6f} {math.exp(b[1]):9.6f} {math.exp(gap):8.6f}")
    common = min(n for _, _, n in nq), min(n for _, _, n in q4)
    if common[0] != common[1]:
        print(f"warning: scored positions differ per window: nq {common[0]}, q4 {common[1]}")
    pnq, pq4 = pooled(nq), pooled(q4)
    gap = pnq - pq4
    ratio = math.exp(gap)
    positions = sum(n for _, _, n in nq)
    print()
    print(f"pooled nq nll    {pnq:.6f}")
    print(f"pooled q4 nll    {pq4:.6f}")
    print(f"pooled nq ppl    {math.exp(pnq):.6f}")
    print(f"pooled q4 ppl    {math.exp(pq4):.6f}")
    print(f"pooled gap       {gap:.6f} nats")
    print(f"pooled ratio     {ratio:.6f}   (bar 1.500000, margin {1.5 - ratio:+.6f} = "
          f"{(1.5 - ratio) / 1.5 * 100:+.4f}% of the bar)")
    print(f"positions        {positions} over {len(nq)} windows")
    per_window = [math.exp(a[1] - b[1]) for a, b in pairs]
    lo, hi = min(per_window), max(per_window)
    print(f"per-window ratio {lo:.6f} .. {hi:.6f} (spread {hi - lo:.6f})")

    # Window-level interval on the gap: the window is the unit because positions inside one window
    # are one coherent passage. This is the conservative bound; the paired-token bound is tighter.
    gaps = [a[1] - b[1] for a, b in pairs]
    n = len(gaps)
    if n > 1:
        g_mean = mean(gaps)
        g_se = math.sqrt(sum((g - g_mean) ** 2 for g in gaps) / (n - 1) / n)
        crit = T975.get(n - 1, 1.96)
        half = crit * g_se
        print()
        print(f"window sd        {math.sqrt(sum((g - g_mean) ** 2 for g in gaps) / (n - 1)):.6f} nats")
        print(f"95 % interval    gap {g_mean:.6f} +- {half:.6f} (t {crit}, df {n - 1})")
        print(f"                 ratio [{math.exp(g_mean - half):.6f}, {math.exp(g_mean + half):.6f}]")
        if math.exp(g_mean - half) > 1.5:
            verdict = "over the bar"
        elif math.exp(g_mean + half) < 1.5:
            verdict = "under the bar"
        else:
            verdict = "undecided: the interval contains 1.5"
        print(f"verdict          {verdict}")

    if args.spans is not None:
        if len(args.spans) != len(pairs):
            raise SystemExit(f"{len(args.spans)} spans for {len(pairs)} windows")
        sw = sum(args.spans)
        g_w = sum(w * (a[1] - b[1]) for w, (a, b) in zip(args.spans, pairs)) / sw
        # Reliability-weight variance: equal weights reproduce sd^2/n exactly; with unequal
        # weights the effective n is below the count, as it should be.
        var_w = sum(w * w * (a[1] - b[1] - g_w) ** 2 for w, (a, b) in zip(args.spans, pairs))
        var_w /= (n - 1) * sum(w * w for w in args.spans)
        se_w = math.sqrt(var_w)
        crit_w = T975.get(n - 1, 1.96)
        half_w = crit_w * se_w
        n_eff = sw * sw / sum(w * w for w in args.spans)
        print()
        print(f"span weights     {args.spans[0]:.0f} x {sum(1 for w in args.spans if w == args.spans[0])}"
              f" + {args.spans[-1]:.0f} x {sum(1 for w in args.spans if w == args.spans[-1])}"
              f"  (text tokens represented)")
        print(f"weighted gap     {g_w:.6f} nats")
        print(f"weighted ratio   {math.exp(g_w):.6f}   (bar 1.500000, margin "
              f"{1.5 - math.exp(g_w):+.6f})")
        print(f"weighted 95 %    gap {g_w:.6f} +- {half_w:.6f} (t {crit_w}, df {n - 1}, "
              f"effective n {n_eff:.1f})")
        print(f"                 ratio [{math.exp(g_w - half_w):.6f}, "
              f"{math.exp(g_w + half_w):.6f}]")
        if math.exp(g_w - half_w) > 1.5:
            verdict_w = "over the bar"
        elif math.exp(g_w + half_w) < 1.5:
            verdict_w = "under the bar"
        else:
            verdict_w = "undecided: the interval contains 1.5"
        print(f"weighted verdict {verdict_w}")
    return 0


def mean(values):
    return sum(values) / len(values)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
