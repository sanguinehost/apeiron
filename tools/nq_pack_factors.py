#!/usr/bin/env python3
"""Pack in-tree `NqFactor` files into an `APNQPK01` pack plus a frozen skeleton.

`tests/learn_nanoquant.rs::nq_fit` writes `l{l}.{kind}-nq{tag}.nqf` (the ADMM factor's signs and
two scale vectors) and the dense `-nq{tag}.i64`. This tool turns that factor set into the pair the
engine serves: `model.nqp` (version 2, no salient term and no residual) and `skeleton.gguf` (the
source's metadata and every non-linear tensor). A kind with no `.nqf` is left out of the pack; if
the skeleton also lacks it, the engine refuses the image, so a partial fit is a partial pack.

The scale vectors are stored as binary16, exactly as the kernel decodes them, and `s_mid` is `1`;
`--dense` writes each packed section's reconstruction as `l{l}.{kind}-nqpack.i64`, which is the
served form `tools/nq_relh.py --fit nqpack` measures.

Usage:
  nq_pack_factors.py --model Qwen3-0.6B-Base-F16.gguf --factors DIR --tag -qwen3-0-6B-Base-F16 \
                     --out-dir DIR [--layers 0,27] [--kinds down,gate,up,o] [--dense]
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import nq_convert
import nq_gguf

ONE = 1 << 32
BF16_ONE = 0x3F80


def read_factor(path):
    """`(out, inn, rank, s_post, s_pre, u, v)` of a `.nqf`; `u` is `out × rank`, `v` `rank × inn`."""
    with open(path, "rb") as f:
        head = np.frombuffer(f.read(32), "<u8")
        out, inn, rank = int(head[0]), int(head[1]), int(head[2])
        body = f.read()
    want = 8 * (out + inn + out * rank + rank * inn) + out * rank + rank * inn
    if len(body) != want:
        raise SystemExit(f"{path}: {len(body)} B, want {want}")
    at = 0

    def take(n):
        nonlocal at
        v = np.frombuffer(body, "<f8", count=n, offset=at).astype(np.float64)
        at += 8 * n
        return v

    s_post, s_pre = take(out), take(inn)
    take(out * rank)
    take(rank * inn)
    u = np.frombuffer(body, np.int8, count=out * rank, offset=at) > 0
    at += out * rank
    v = np.frombuffer(body, np.int8, count=rank * inn, offset=at) > 0
    return out, inn, rank, s_post, s_pre, u.reshape(out, rank), v.reshape(rank, inn)


def pack_bits(bits, words):
    """`bits` (`rows × n`, bool) as `rows × words` u32 words, bit 1 = `+1`, little-endian."""
    rows, n = bits.shape
    pad = np.zeros((rows, words * 32 - n), dtype=bool)
    flat = np.concatenate([bits, pad], axis=1)
    packed = np.packbits(flat, axis=1, bitorder="little")
    return np.frombuffer(packed.tobytes(), "<u4").reshape(rows, words)


def bf16(values):
    """Float64 → binary16 bits, round-to-nearest-even through float32."""
    return nq_convert.bf16_words(np.asarray(values, dtype=np.float32))


def decode_bf16(bits):
    """The binary16 words `bf16` wrote, as float64."""
    return (np.frombuffer(bits.tobytes(), "<u2").astype(np.uint32) << 16).view(np.float32)


def write_section(out, layer, kind, factor):
    """Write one version-2 section; return `(signs bits, scale bits, bytes)`."""
    out_rows, inn, rank, s_post, s_pre, u, v = factor
    u_words = (rank + 31) // 32
    v_words = (inn + 31) // 32
    uw = pack_bits(u, u_words)
    vw = pack_bits(v, v_words)
    pre, post = bf16(s_pre), bf16(s_post)
    mid = np.full(rank, BF16_ONE, dtype="<u2")
    out.write(struct.pack("<IIIIIIII", layer, kind, out_rows, inn, rank, u_words, v_words, 0))
    out.write(uw.tobytes())
    out.write(vw.tobytes())
    out.write(pre.tobytes())
    out.write(mid.tobytes())
    out.write(post.tobytes())
    bits = 32 * (out_rows * u_words + rank * v_words) + 16 * (inn + rank + out_rows)
    total = 32 + 4 * (out_rows * u_words + rank * v_words) + 2 * (inn + rank + out_rows)
    return bits, total


def served_dense(factor):
    """The section's reconstruction from the stored bits and the binary16-decoded scales."""
    out_rows, inn, rank, s_post, s_pre, u, v = factor
    uu = u.astype(np.float64) * 2 - 1
    vv = v.astype(np.float64) * 2 - 1
    pre = decode_bf16(bf16(s_pre)).astype(np.float64)
    post = decode_bf16(bf16(s_post)).astype(np.float64)
    return post[:, None] * (uu @ vv) * pre[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the F16 GGUF the factors were fitted against")
    ap.add_argument("--factors", required=True)
    ap.add_argument("--tag", default="", help="the `-nq{tag}.nqf` suffix the fit wrote")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="", help="default: every layer of the model")
    ap.add_argument("--kinds", default="q,k,v,o,gate,up,down")
    ap.add_argument("--dense", action="store_true", help="write each packed section's dense form")
    ap.add_argument("--dense-dir", default="", help="default: --factors")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src = nq_gguf.parse_gguf(args.model)
    tensors = {t["name"]: t for t in src["tensors"]}
    layers_all = 1 + max(int(t["name"].split(".")[1]) for t in src["tensors"]
                         if t["name"].startswith("blk."))
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers else list(range(layers_all)))
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in nq_convert.KINDS:
            raise SystemExit(f"--kinds {k}: not one of {nq_convert.KINDS}")
    dense_dir = args.dense_dir or args.factors
    if args.dense:
        os.makedirs(dense_dir, exist_ok=True)

    pack_path = os.path.join(args.out_dir, "model.nqp")
    total = nq_convert.HEADER_BYTES
    weights = bits = sections = 0
    with open(pack_path, "wb") as out:
        out.write(nq_convert.MAGIC)
        out.write(struct.pack("<IIIIII", nq_convert.VERSION_BASE, nq_convert.HEADER_BYTES,
                              layers_all, len(nq_convert.KINDS), 0, 0))
        for layer in layers:
            for kname in kinds:
                kind = nq_convert.KINDS.index(kname)
                path = os.path.join(args.factors, f"l{layer}.{kname}-nq{args.tag}.nqf")
                if not os.path.exists(path):
                    print(f"skip l{layer} {kname}: no {path}", flush=True)
                    continue
                factor = read_factor(path)
                (o, n) = factor[0], factor[1]
                key = f"blk.{layer}.{nq_convert.DENSE_NAMES[kname]}"
                if key not in tensors:
                    raise SystemExit(f"l{layer} {kname}: {args.model} has no {key}")
                b, t = write_section(out, layer, kind, factor)
                bits += b
                total += t
                weights += o * n
                sections += 1
                if args.dense:
                    header = np.asarray([o, n], dtype="<u8").tobytes()
                    words = np.rint(served_dense(factor) * ONE).astype("<i8")
                    with open(os.path.join(dense_dir, f"l{layer}.{kname}-nqpack.i64"), "wb") as df:
                        df.write(header)
                        df.write(words.tobytes())
        # `linears` sits after magic, version, header bytes, layers and kinds.
        out.seek(24)
        out.write(struct.pack("<I", sections))
    if os.path.getsize(pack_path) != total:
        raise SystemExit(f"{pack_path}: {os.path.getsize(pack_path)} B, want {total}")
    print(f"pack: {pack_path} {total} B, {sections} sections, "
          f"{bits / weights:.6f} b/w on {weights} fitted weights", flush=True)

    skeleton_path = os.path.join(args.out_dir, "skeleton.gguf")
    n, entries, _ = nq_convert.write_skeleton(skeleton_path, src)
    print(f"skeleton: {skeleton_path} {n} B, {len(entries)} tensors", flush=True)
    print(f"artifact: {total + n} B on disk", flush=True)


if __name__ == "__main__":
    main()
