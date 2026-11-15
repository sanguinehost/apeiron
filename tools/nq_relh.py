#!/usr/bin/env python3
"""Hessian-weighted relative reconstruction error of one or more dense reconstructions.

For each `(layer, kind)` this reads the cached Gram `g{l}.{tag}.{gram}.i64`, the teacher linear
from the F16 GGUF, and every named reconstruction `l{l}.{kind}-{name}.i64`, then prints

  relH_E = tr(E H Eᵀ) / tr(W H Wᵀ)        (an energy fraction)
  relH   = sqrt(relH_E)                    (‖E‖_H / ‖W‖_H, the essay's definition)

and the ratio `relH(reference) / relH(candidate)` for each candidate. The distinction matters: a
ratio of energy fractions is the square of a ratio of norms, so mixing the two conventions
inflates a distortion advantage by its own square root.

Usage:
  nq_relh.py --model Qwen3-0.6B-Base-F16.gguf --cache DIR --gram-tag c6x29 \
             --layers 0,26,27 --kinds down,gate,up,o --ref nqref [--fit ours]
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The Gram each pack kind reads; `q`, `k` and `v` share `attn_norm`'s output.
GRAM = {
    "q": "qkv",
    "k": "qkv",
    "v": "qkv",
    "o": "o",
    "gate": "gate_up",
    "up": "gate_up",
    "down": "down",
}
# The F16 tensor of each pack kind.
TENSOR = {
    "q": "attn_q",
    "k": "attn_k",
    "v": "attn_v",
    "o": "attn_output",
    "gate": "ffn_gate",
    "up": "ffn_up",
    "down": "ffn_down",
}
ONE = 1 << 32


def read_i64(path, header):
    """`(u64, u64)` header words followed by `rows × cols` Q32.32 words, as float64."""
    import numpy as np
    with open(path, "rb") as f:
        h = np.frombuffer(f.read(header * 8), "<u8")
        a, b = int(h[0]), int(h[1])
        w = np.fromfile(f, "<i8", count=a * b)
    if w.size != a * b:
        raise SystemExit(f"{path}: {w.size} words, want {a}×{b}")
    return a, b, w.astype(np.float64) / ONE


def teacher(model, layer, kind):
    """The F16 linear's dense value, `out × inn`."""
    import numpy as np
    import nq_gguf
    src = teacher.src
    name = f"blk.{layer}.{TENSOR[kind]}.weight"
    ten = teacher.tensors.get(name)
    if ten is None:
        raise SystemExit(f"{name}: missing")
    if ten["type"] != 1:
        raise SystemExit(f"{name}: type {ten['type']}, want F16")
    inn, out = ten["dims"]
    with open(src["path"], "rb") as f:
        f.seek(ten["abs"])
        raw = f.read(ten["nbytes"])
    return np.frombuffer(raw, "<f2").astype(np.float64).reshape(out, inn)


teacher.src = None
teacher.tensors = None


def relh(w, h, r):
    """`(energy fraction, norm ratio)` of `W - R` under `H`."""
    import numpy as np
    e = w - r
    num = float(((e @ h) * e).sum())
    den = float(((w @ h) * w).sum())
    return num / den, math.sqrt(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gram-tag", default="c6x29")
    ap.add_argument("--layers", default="0,26,27")
    ap.add_argument("--kinds", default="down,gate,up,o")
    ap.add_argument("--ref", default="nqref", help="the baseline reconstruction's name suffix")
    ap.add_argument("--fit", default="", help="a second reconstruction, e.g. the in-tree fit tag")
    ap.add_argument("--csv", default="", help="write the rows here")
    args = ap.parse_args()

    import numpy as np
    import nq_gguf
    teacher.src = nq_gguf.parse_gguf(args.model)
    teacher.tensors = {t["name"]: t for t in teacher.src["tensors"]}
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    names = [n for n in (args.ref, args.fit) if n]

    rows = []
    for l in layers:
        for kind in kinds:
            g = os.path.join(args.cache, f"g{l}.{args.gram_tag}.{GRAM[kind]}.i64")
            if not os.path.exists(g):
                print(f"l{l} {kind}: no Gram {g}", flush=True)
                continue
            _, hn, h = read_i64(g, 2)
            h = h.reshape(hn, hn)
            w = teacher(args.model, l, kind)
            out, inn = w.shape
            if hn != inn:
                raise SystemExit(f"l{l} {kind}: Gram {hn}², linear {out}×{inn}")
            base = float(((w @ h) * w).sum())
            row = {"layer": l, "kind": kind, "out": out, "inn": inn}
            for name in names:
                p = os.path.join(args.cache, f"l{l}.{kind}-{name}.i64")
                if not os.path.exists(p):
                    row[name] = None
                    continue
                ro, rn, r = read_i64(p, 2)
                if (ro, rn) != (out, inn):
                    raise SystemExit(f"{p}: {ro}×{rn}, want {out}×{inn}")
                r = r.reshape(out, inn)
                e = w - r
                frac = float(((e @ h) * e).sum()) / base
                row[name] = frac
            rows.append(row)
            cells = " ".join(
                f"{n} E {row[n]:.6f} norm {math.sqrt(row[n]) * 100:.2f}%"
                if row[n] is not None
                else f"{n} n/a"
                for n in names
            )
            extra = ""
            if all(row[n] is not None for n in names) and len(names) == 2:
                a, b = row[names[0]], row[names[1]]
                extra = f" | norm ratio {math.sqrt(a / b):.2f}x energy ratio {a / b:.2f}x"
            print(f"l{l:02} {kind:4s} {out}x{inn} {cells}{extra}", flush=True)
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("layer,kind,out,inn," + ",".join(f"{n}_energy,{n}_norm" for n in names) + "\n")
            for row in rows:
                cells = []
                for n in names:
                    v = row[n]
                    cells.append("" if v is None else f"{v:.9f}")
                    cells.append("" if v is None else f"{math.sqrt(v):.9f}")
                f.write(f"{row['layer']},{row['kind']},{row['out']},{row['inn']}," + ",".join(cells) + "\n")
        print(f"csv: {args.csv}", flush=True)


if __name__ == "__main__":
    main()
