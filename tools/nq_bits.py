#!/usr/bin/env python3
"""Stored bits per weight of a GGUF, per tensor and over the sets the card compares.

The card's DRAM-traffic ratio is a ratio of stored bytes on the same tensors, so both sides have
to be read from their images rather than quoted: this prints, for each file, the payload
`bits/weight` of every scored `(layer, kind)` row, of the whole pack candidate (the FFN, the
attention output and the q/k/v projections), and of the whole image.

The denominator of a row is the dense matrix's element count, read from the `.weight` tensor of
any argument that carries one (an F16 or `Q4_K_M` image) and from `attn_q.weight` +
`attn_k.weight` + `attn_v.weight` for the concatenated `attn_qkv` entry a NanoQuant image stores
in their place. A file with no dense image among the arguments therefore reports bytes only.

The block table in [`nq_gguf`] is checked against the file: the last payload must end at the file
size within one alignment, or the layout is wrong and nothing is printed.

Usage:
  nq_bits.py Qwen3-0.6B-Base-F16.gguf Qwen3-0.6B-Base-Q4_K_M.gguf qwen3-0-6b-nanoquant.gguf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nq_gguf  # noqa: E402

# The twelve rows the campaign scores.
ROWS = [
    (0, "ffn_down"),
    (0, "ffn_gate"),
    (0, "ffn_up"),
    (0, "attn_output"),
    (26, "ffn_down"),
    (26, "ffn_gate"),
    (26, "ffn_up"),
    (26, "attn_output"),
    (27, "ffn_down"),
    (27, "ffn_gate"),
    (27, "ffn_up"),
    (27, "attn_output"),
]
# The kinds a NanoQuant pack replaces. `attn_qkv` is the three attention projections in one.
PACK = ("ffn_down", "ffn_gate", "ffn_up", "attn_output", "attn_qkv")
ATTN = ("attn_q", "attn_k", "attn_v")


def elements(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def prefix_bytes(tensors, layer, kind):
    """Payload bytes of every tensor `blk.{layer}.{kind}.*` (the weight or the `nq_*` parts).

    `attn_qkv` is one fused tensor in a NanoQuant image and three separate ones in a dense image,
    so the fused prefix falls back to q, k and v when it is absent.
    """
    pref = "blk.%d.%s." % (layer, kind)
    if kind == "attn_qkv" and not any(t["name"].startswith(pref) for t in tensors):
        return sum(prefix_bytes(tensors, layer, k) for k in ATTN)
    return sum(t["nbytes"] for t in tensors if t["name"].startswith(pref))


def dense_shapes(tensors):
    """`(layer, kind) -> elements` from the dense `.weight` tensors that are present."""
    by_name = {t["name"]: t for t in tensors}
    shapes = {}
    layers = sorted(
        {
            int(t["name"].split(".")[1])
            for t in tensors
            if t["name"].startswith("blk.") and t["name"].split(".")[1].isdigit()
        }
    )
    for l in layers:
        for kind in ("ffn_down", "ffn_gate", "ffn_up", "attn_output"):
            t = by_name.get("blk.%d.%s.weight" % (l, kind))
            if t is not None:
                shapes[(l, kind)] = elements(t["dims"])
        attn = [by_name.get("blk.%d.%s.weight" % (l, k)) for k in ATTN]
        if all(t is not None for t in attn):
            shapes[(l, "attn_qkv")] = sum(elements(t["dims"]) for t in attn)
    return layers, shapes


def check_extent(src):
    end = max(t["abs"] + t["nbytes"] for t in src["tensors"])
    padded = (end + src["align"] - 1) // src["align"] * src["align"]
    size = os.path.getsize(src["path"])
    if not 0 <= size - padded < src["align"]:
        raise SystemExit(
            "%s: payload ends at %d (padded %d) but the file is %d B; the block table is wrong"
            % (src["path"], end, padded, size)
        )


def report(path, layers, shapes):
    src = nq_gguf.parse_gguf(path)
    check_extent(src)
    tensors = src["tensors"]
    total_n = sum(elements(t["dims"]) for t in tensors)
    total_b = sum(t["nbytes"] for t in tensors)
    types = sorted({nq_gguf.GGML_NAMES.get(t["type"], str(t["type"])) for t in tensors})
    print("== %s" % path)
    print(
        "   types %s  %d elements, %d B, %.3f b/w"
        % (",".join(types), total_n, total_b, 8.0 * total_b / total_n)
    )
    pack_kinds = [k for k in PACK if any((l, k) in shapes for l in layers)]
    pack_b = sum(prefix_bytes(tensors, l, k) for l in layers for k in pack_kinds)
    pack_n = sum(shapes.get((l, k), 0) for l in layers for k in pack_kinds)
    if pack_n:
        print(
            "   pack candidate (%s): %d elements, %d B, %.3f b/w"
            % ("+".join(pack_kinds), pack_n, pack_b, 8.0 * pack_b / pack_n)
        )
    for l, k in ROWS:
        pref = "blk.%d.%s." % (l, k)
        parts = [t for t in tensors if t["name"].startswith(pref)]
        if not parts:
            print("   l%-2d %-12s absent" % (l, k))
            continue
        b = sum(t["nbytes"] for t in parts)
        n = shapes.get((l, k))
        rate = "%.3f b/w" % (8.0 * b / n) if n else "b/w n/a"
        print(
            "   l%-2d %-12s %8d B  %s  %s"
            % (
                l,
                k,
                b,
                rate,
                ",".join(
                    sorted(
                        {nq_gguf.GGML_NAMES.get(t["type"], str(t["type"])) for t in parts}
                    )
                ),
            )
        )
    print()


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    images = [(path, nq_gguf.parse_gguf(path)) for path in sys.argv[1:]]
    layers, shapes = set(), {}
    for _, src in images:
        ls, sh = dense_shapes(src["tensors"])
        layers.update(ls)
        shapes.update(sh)
    layers = sorted(layers)
    for path, _ in images:
        report(path, layers, shapes)


if __name__ == "__main__":
    main()
