#!/usr/bin/env python3
"""Ternary-versus-q8 activation error on a captured stage dump.

Reads the files `examples/act_dump` writes (`layer{L}-{stage}.i64`: one ASCII header line, then
`columns x width` little-endian i64 words, row-major) and compares, per 64-term block:

  (a) the optimal ternary quantizer: `t in {-1, 0, +1}` per term with one power-of-two scale per
      64-term block, both chosen to minimize `sum (x - s*t)^2`. For a fixed `s` each term is
      independent (`t = 0` when `2|x| < s`, else `sign(x)`), and the block cost over every
      power-of-two `s` is evaluated exactly in integer arithmetic from the sorted magnitudes:
      `cost(s) = sum_kept a^2 - 2s*sum_selected a + n_selected*s^2`.
  (b) the engine's q8 reader (`quant_act`, `src/engine/chart/mod.rs`): one int8 code per term and
      one power-of-two scale per 32-term block, `S = 2^ceil(log2(ceil(absmax/127)))`,
      `code = sign(x)*floor((2|x| + S)/(2S))`, exact integer arithmetic.
  (c) the ratio of the two squared errors and the relative dot RMS error
      `sqrt(sum e^2 / sum x^2)` for each stage and layer.

Two residual variants: the top-4-by-|x| terms of each block (per column) kept exact, and a static
top-4 mask per block chosen by the variance of each position across the capture's columns, applied
to every column. Both re-optimize the ternary scale over the 60 quantized terms.

The error model is a projection: the captured stage input is `x`, while a fitted NanoQuant linear
reads the pre-scaled `t = round(s_pre*x / 2^32)`; `s_pre` is per linear, so one stage shared by
several linears has no single pre-scale. The screen is `model`, not a PPL.

Usage:
  python3 tools/ternary_act_proxy.py --dir /tmp/opencode/act-dump [--json summary.json]
"""

import argparse
import bisect
import json
import math
import os
import struct
import sys

LAYERS = (0, 13, 27)
STAGES = ("attn_norm", "ffn_norm")
BLOCK = 64
QK = 32


def read_dump(path):
    """Return (header, columns, width, [column][word]) for one act_dump file."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip("\n")
        fields = {}
        for token in header.split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        if fields.get("dtype") != "i64le":
            raise SystemExit(f"{path}: dtype {fields.get('dtype')!r}, not i64le")
        width = int(fields["width"])
        columns = int(fields["columns"])
        payload = f.read()
    expect = columns * width * 8
    if len(payload) != expect:
        raise SystemExit(f"{path}: {len(payload)} bytes, expected {expect}")
    words = struct.unpack(f"<{columns * width}q", payload)
    rows = [words[c * width:(c + 1) * width] for c in range(columns)]
    return header, columns, width, rows


def ternary_cost(vals, keep):
    """(min SSE, scale) of the ternary quantizer on the terms in `keep`; removed terms cost 0."""
    absv = sorted(abs(vals[j]) for j in keep)
    if not absv or absv[-1] == 0:
        return 0, 0
    a2 = sum(a * a for a in absv)
    prefix = [0] * (len(absv) + 1)
    for i, a in enumerate(absv):
        prefix[i + 1] = prefix[i] + a
    n = len(absv)
    best = None
    top = absv[-1].bit_length()  # smallest 2^top >= absmax
    for e in range(top + 1):
        s = 1 << e
        idx = bisect.bisect_left(absv, (s + 1) // 2)
        cnt = n - idx
        kept = prefix[n] - prefix[idx]
        cost = a2 - 2 * s * kept + cnt * s * s
        if best is None or cost < best[0]:
            best = (cost, s)
    return best


def cell_cost(vals, k):
    """The designed cell (`bench/notes/trichotomy-cell.md` §3): per-64 scale by the q8 rule
    (`127*S >= absmax`), code `c = round(x/S)`, trit `M=64` with `t = sign(c)` when `2|c| > 64`,
    and the `k` largest `|r| = |c - 64*t|` terms corrected exactly. Returns
    `(sse, zeroed, clipped)`; `sse` is exact in integer arithmetic (`S*r` per term)."""
    absmax = max(abs(v) for v in vals)
    if absmax == 0:
        return 0, len(vals), 0
    q = (absmax + 126) // 127
    e = 0 if q <= 1 else (q - 1).bit_length()
    s = 1 << e
    level = 64
    twice = 2 * s
    residuals = []
    zeroed = 0
    clipped = 0
    for v in vals:
        a = abs(v)
        code = (2 * a + s) // twice
        if code > 127:
            code = 127
        if code == 127:
            clipped += 1
        signed = code if v > 0 else -code
        tau = 0 if 2 * code <= level else (1 if v > 0 else -1)
        if tau == 0:
            zeroed += 1
        r = signed - level * tau
        residuals.append(r * r)
    residuals.sort(reverse=True)
    sse = s * s * sum(residuals[k:])
    return sse, zeroed, clipped


def q8_cost(vals):
    """The engine's q8 per-32 squared error over `vals` (a 64-term block: two 32-term blocks)."""
    sse = 0
    for start in range(0, len(vals), QK):
        block = vals[start:start + QK]
        absmax = max((abs(v) for v in block), default=0)
        if absmax == 0:
            continue
        q = (absmax + 126) // 127
        e = 0 if q <= 1 else (q - 1).bit_length()  # ceil(log2(q))
        s = 1 << e
        twice = 2 * s
        for v in block:
            a = abs(v)
            code = (2 * a + s) // twice
            d = a - code * s
            sse += d * d
    return sse


def sum_sq(vals):
    return sum(v * v for v in vals)


def percentile(sorted_vals, frac):
    if not sorted_vals:
        return None
    at = frac * (len(sorted_vals) - 1)
    lo = int(math.floor(at))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (at - lo)


def static_masks(rows, width):
    """Per 64-block, the top-4 positions by variance across columns (ties by index)."""
    columns = len(rows)
    masks = []
    for base in range(0, width, BLOCK):
        take = min(BLOCK, width - base)
        if take <= 4:
            masks.append(None)
            continue
        var = []
        for j in range(take):
            total = 0
            total2 = 0
            for c in range(columns):
                v = rows[c][base + j]
                total += v
                total2 += v * v
            # population variance, exact integer numerator; the order is what matters
            var.append((total2 * columns - total * total, j))
        var.sort(key=lambda t: (-t[0], t[1]))
        masks.append(sorted(j for _, j in var[:4]))
    return masks


def analyse(rows, width, label):
    """Accumulate the five error models over every 64-term block of every column."""
    columns = len(rows)
    nblocks = (width + BLOCK - 1) // BLOCK
    masks = static_masks(rows, width)
    totals = {
        "x2": 0,
        "ternary": 0,
        "q8": 0,
        "abs4": 0,
        "var4": 0,
        "cell0": 0,
        "cell1": 0,
        "cell2": 0,
        "cell4": 0,
        "zeroed": 0,
        "clipped": 0,
        "terms": 0,
    }
    ratios = []
    cell_ratios = []
    for base in range(0, width, BLOCK):
        take = min(BLOCK, width - base)
        block_index = base // BLOCK
        static_keep = None
        if masks[block_index] is not None:
            drop = set(masks[block_index])
            static_keep = [j for j in range(take) if j not in drop]
        if static_keep is None:
            static_keep = list(range(take))
        for c in range(columns):
            block = rows[c][base:base + take]
            x2 = sum_sq(block)
            totals["x2"] += x2
            tern, _ = ternary_cost(block, range(take))
            q8 = q8_cost(block)
            totals["ternary"] += tern
            totals["q8"] += q8
            if q8:
                ratios.append(tern / q8)
            if take > 4:
                order = sorted(range(take), key=lambda j: (-abs(block[j]), j))
                abs_keep = [j for j in range(take) if j not in set(order[:4])]
                abs4, _ = ternary_cost(block, abs_keep)
                var4, _ = ternary_cost(block, static_keep)
            else:
                abs4 = tern
                var4 = tern
            totals["abs4"] += abs4
            totals["var4"] += var4
            c0, zeroed, clipped = cell_cost(block, 0)
            c1, _, _ = cell_cost(block, 1)
            c2, _, _ = cell_cost(block, 2)
            c4, _, _ = cell_cost(block, 4)
            totals["cell0"] += c0
            totals["cell1"] += c1
            totals["cell2"] += c2
            totals["cell4"] += c4
            totals["zeroed"] += zeroed
            totals["clipped"] += clipped
            totals["terms"] += take
            if q8:
                cell_ratios.append(c0 / q8)

    ratios.sort()

    def rel(sse):
        return math.sqrt(sse / totals["x2"]) if totals["x2"] else 0.0

    out = {
        "label": label,
        "columns": columns,
        "width": width,
        "blocks_per_column": nblocks,
        "sum_x2": totals["x2"],
        "ternary_sse": totals["ternary"],
        "q8_sse": totals["q8"],
        "abs4_sse": totals["abs4"],
        "var4_sse": totals["var4"],
        "ternary_rel_rms": rel(totals["ternary"]),
        "q8_rel_rms": rel(totals["q8"]),
        "abs4_rel_rms": rel(totals["abs4"]),
        "var4_rel_rms": rel(totals["var4"]),
        "ternary_over_q8_sse": totals["ternary"] / totals["q8"] if totals["q8"] else None,
        "ternary_over_q8_rms": (math.sqrt(totals["ternary"] / totals["q8"])
                                if totals["q8"] else None),
        "abs4_removed": (1.0 - totals["abs4"] / totals["ternary"]
                         if totals["ternary"] else None),
        "var4_removed": (1.0 - totals["var4"] / totals["ternary"]
                         if totals["ternary"] else None),
        "cell0_rel_rms": rel(totals["cell0"]),
        "cell1_rel_rms": rel(totals["cell1"]),
        "cell2_rel_rms": rel(totals["cell2"]),
        "cell4_rel_rms": rel(totals["cell4"]),
        "cell0_sse": totals["cell0"],
        "cell1_sse": totals["cell1"],
        "cell2_sse": totals["cell2"],
        "cell4_sse": totals["cell4"],
        "cell_zeroed": totals["zeroed"],
        "cell_clipped": totals["clipped"],
        "terms": totals["terms"],
        "cell0_over_q8_sse": totals["cell0"] / totals["q8"] if totals["q8"] else None,
        "cell4_over_q8_sse": totals["cell4"] / totals["q8"] if totals["q8"] else None,
        "cell4_removed": (1.0 - totals["cell4"] / totals["cell0"]
                          if totals["cell0"] else None),
        "cell_zero_frac": totals["zeroed"] / totals["terms"] if totals["terms"] else None,
        "cell_clip_frac": totals["clipped"] / totals["terms"] if totals["terms"] else None,
        "cell_block_ratio_median": percentile(cell_ratios, 0.5),
        "block_ratio_median": percentile(ratios, 0.5),
        "block_ratio_p95": percentile(ratios, 0.95),
        "blocks": len(ratios),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="/tmp/opencode/act-dump")
    parser.add_argument("--json", default="/tmp/opencode/act-dump/summary.json")
    args = parser.parse_args()

    results = []
    for layer in LAYERS:
        for stage in STAGES:
            path = os.path.join(args.dir, f"layer{layer}-{stage}.i64")
            if not os.path.exists(path):
                print(f"missing {path}", file=sys.stderr)
                continue
            header, columns, width, rows = read_dump(path)
            print(f"{path}: {header}", file=sys.stderr)
            results.append(analyse(rows, width, f"L{layer} {stage}"))

    if not results:
        raise SystemExit("no dumps found")

    # A pooled row weights each stage by its captured energy, not by its width.
    pooled = {}
    for key in ("sum_x2", "ternary_sse", "q8_sse", "abs4_sse", "var4_sse",
                "cell0_sse", "cell1_sse", "cell2_sse", "cell4_sse",
                "cell_zeroed", "cell_clipped", "terms"):
        pooled[key] = sum(r[key] for r in results)
    x2 = pooled["sum_x2"]
    rel = lambda sse: math.sqrt(sse / x2) if x2 else 0.0
    pooled_row = {
        "label": "all",
        "ternary_rel_rms": rel(pooled["ternary_sse"]),
        "q8_rel_rms": rel(pooled["q8_sse"]),
        "abs4_rel_rms": rel(pooled["abs4_sse"]),
        "var4_rel_rms": rel(pooled["var4_sse"]),
        "ternary_over_q8_sse": pooled["ternary_sse"] / pooled["q8_sse"],
        "ternary_over_q8_rms": math.sqrt(pooled["ternary_sse"] / pooled["q8_sse"]),
        "abs4_removed": 1.0 - pooled["abs4_sse"] / pooled["ternary_sse"],
        "var4_removed": 1.0 - pooled["var4_sse"] / pooled["ternary_sse"],
        "cell0_rel_rms": rel(pooled["cell0_sse"]),
        "cell1_rel_rms": rel(pooled["cell1_sse"]),
        "cell2_rel_rms": rel(pooled["cell2_sse"]),
        "cell4_rel_rms": rel(pooled["cell4_sse"]),
        "cell0_over_q8_sse": pooled["cell0_sse"] / pooled["q8_sse"],
        "cell4_over_q8_sse": pooled["cell4_sse"] / pooled["q8_sse"],
        "cell4_removed": 1.0 - pooled["cell4_sse"] / pooled["cell0_sse"],
        "cell_zero_frac": pooled["cell_zeroed"] / pooled["terms"],
        "cell_clip_frac": pooled["cell_clipped"] / pooled["terms"],
    }

    cols = [
        ("label", "stage"),
        ("ternary_rel_rms", "tern rms"),
        ("q8_rel_rms", "q8 rms"),
        ("ternary_over_q8_sse", "tern/q8 sse"),
        ("ternary_over_q8_rms", "tern/q8 rms"),
        ("abs4_rel_rms", "top4-|x| rms"),
        ("abs4_removed", "top4-|x| removed"),
        ("var4_rel_rms", "top4-var rms"),
        ("var4_removed", "top4-var removed"),
        ("block_ratio_median", "block med"),
        ("block_ratio_p95", "block p95"),
    ]
    cell_cols = [
        ("label", "stage"),
        ("cell0_rel_rms", "cell k=0 rms"),
        ("cell1_rel_rms", "k=1 rms"),
        ("cell2_rel_rms", "k=2 rms"),
        ("cell4_rel_rms", "k=4 rms"),
        ("cell4_removed", "k=4 removed"),
        ("cell0_over_q8_sse", "cell0/q8 sse"),
        ("cell_zero_frac", "zeroed frac"),
        ("cell_clip_frac", "clipped frac"),
        ("cell_block_ratio_median", "block med"),
    ]
    for title, table_cols in (("optimal ternary", cols), ("designed cell", cell_cols)):
        print(f"\n{title}")
        print("| " + " | ".join(title for _, title in table_cols) + " |")
        print("|" + "|".join("---" for _ in table_cols) + "|")
        for row in results + [pooled_row]:
            cells = []
            for key, _ in table_cols:
                value = row.get(key)
                if value is None:
                    cells.append("-")
                elif isinstance(value, float):
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(str(value))
            print("| " + " | ".join(cells) + " |")

    payload = {"results": results, "pooled": pooled_row}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
