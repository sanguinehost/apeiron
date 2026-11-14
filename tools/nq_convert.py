#!/usr/bin/env python3
"""Convert an arelath `nq_*` GGUF into an `APNQPK01` pack and a frozen-tensor skeleton.

The source carries one binary sign-rank linear per `attn_qkv`, `attn_output`, `ffn_gate`,
`ffn_up` and `ffn_down`, plus the embedding, head and norms. This tool writes:

  model.nqp       the fitted layers' seven linears (`q`, `k`, `v`, `o`, `gate`, `up`, `down`) in
                  the format `src/format/nq.rs` reads. The fused `attn_qkv` factor is split by
                  output row into its q/k/v sections; `V`, `s_pre`, `s_mid` and the salient indices
                  are shared, so the split is exact. `--keep-dense` omits sections.
  skeleton.gguf   the source metadata and every non-NQ tensor, with NQ payloads omitted. The pack
                  is the resident copy of each fitted projection. A kept-dense section is copied
                  from `--dense-ref` instead, so the engine serves that linear from the skeleton.

Both writes stream: the source and the outputs are never all resident. Usage:

  nq_convert.py --source qwen3-8b-nanoquant.gguf --out-dir /home/lucas/models/arelath/nq8b
  nq_convert.py --source ... --out-dir ... --dense-ref Qwen3-8B.Q8_0.gguf --residual 1
  nq_convert.py --source ... --out-dir ... --dense-ref Qwen3-8B.Q8_0.gguf \
                --keep-dense 0:q --keep-dense 0:up
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_gguf

MAGIC = b"APNQPK01"
VERSION = 3
VERSION_BASE = 2
HEADER_BYTES = 32
KINDS = ["q", "k", "v", "o", "gate", "up", "down"]
GGML_I32 = 26

# The source's five fused/split names, and the pack kinds each one covers.
PREFIXES = [
    ("attn_qkv", (0, 1, 2)),
    ("attn_output", (3,)),
    ("ffn_gate", (4,)),
    ("ffn_up", (5,)),
    ("ffn_down", (6,)),
]

LINEAR_WEIGHTS = {
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_qkv.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
}

# The dense tensor of each pack kind in a reference GGUF.
DENSE_NAMES = {
    "q": "attn_q.weight",
    "k": "attn_k.weight",
    "v": "attn_v.weight",
    "o": "attn_output.weight",
    "gate": "ffn_gate.weight",
    "up": "ffn_up.weight",
    "down": "ffn_down.weight",
}


def read_payload(src, ten):
    with open(src["path"], "rb") as f:
        f.seek(ten["abs"])
        data = f.read(ten["nbytes"])
    if len(data) != ten["nbytes"]:
        raise ValueError(f"{ten['name']}: {len(data)} B, want {ten['nbytes']}")
    return data


def dims_of(tensors, name):
    try:
        return tensors[name]["dims"]
    except KeyError:
        raise SystemExit(f"missing tensor {name}")


def meta_ints(meta):
    """The integer metadata keys of a raw GGUF metadata block."""
    out = {}
    o = 0
    fmts = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
    while o < len(meta):
        key, o = nq_gguf._rstr(meta, o)
        (t,) = struct.unpack_from("<I", meta, o)
        o += 4
        if t in fmts:
            v, = struct.unpack_from(fmts[t], meta, o)
            out[key] = v
        o = nq_gguf._skip_value(meta, o, t)
    if o != len(meta):
        raise ValueError("metadata does not parse to its declared extent")
    return out


def qkv_split(meta, out_total):
    """`(q_out, kv_out)` of a fused Qwen3 qkv, from its head geometry."""
    m = meta_ints(meta)
    heads = m["qwen3.attention.head_count"]
    kv_heads = m["qwen3.attention.head_count_kv"]
    key_len = m.get("qwen3.attention.key_length", m["qwen3.embedding_length"] // heads)
    value_len = m.get("qwen3.attention.value_length", key_len)
    q_out = heads * key_len
    kv_out = kv_heads * value_len
    if q_out + 2 * kv_out != out_total:
        raise ValueError(f"qkv {out_total} != q {q_out} + 2*kv {kv_out}")
    return q_out, kv_out


def linear_payload(src, tensors, prefix):
    """The five raw payloads of one source linear, with their derived geometry."""
    get = lambda s: read_payload(src, tensors[prefix + s])
    u = get(".nq_u")
    v = get(".nq_v")
    pre = get(".nq_scale_pre")
    mid = get(".nq_scale_mid")
    post = get(".nq_scale_post")
    idx = get(".nq_salient_idx")
    sal = get(".nq_salient_weight")
    u_dims = dims_of(tensors, prefix + ".nq_u")
    v_dims = dims_of(tensors, prefix + ".nq_v")
    out_rows, u_words = u_dims[1], u_dims[0]
    rank, v_words = v_dims[1], v_dims[0]
    inn = v_words * 32
    k = dims_of(tensors, prefix + ".nq_salient_idx")[0]
    if len(u) != out_rows * u_words * 4:
        raise ValueError(f"{prefix}.nq_u: {len(u)} B for {out_rows}x{u_words} words")
    if len(v) != rank * v_words * 4:
        raise ValueError(f"{prefix}.nq_v: {len(v)} B for {rank}x{v_words} words")
    if len(pre) != inn * 2 or len(mid) != rank * 2 or len(post) != out_rows * 2:
        raise ValueError(f"{prefix}: scale lengths do not match {inn}/{rank}/{out_rows}")
    if len(idx) != k * 4 or len(sal) != out_rows * k * 2:
        raise ValueError(f"{prefix}: salient lengths do not match {k}/{out_rows}")
    return dict(out=out_rows, inn=inn, rank=rank, u_words=u_words, v_words=v_words, k=k,
                u=u, v=v, pre=pre, mid=mid, post=post, idx=idx, sal=sal)


def row_slice(raw, rows_total, row_bytes, r0, r1):
    if r0 < 0 or r1 > rows_total or r0 >= r1:
        raise ValueError(f"rows [{r0},{r1}) outside {rows_total}")
    return raw[r0 * row_bytes:r1 * row_bytes]


def dense_rows(ref, name, r0, r1):
    """The Q8_0 rows [r0, r1) of a dense reference tensor, decoded to float32."""
    import numpy as np
    ten = {t["name"]: t for t in ref["tensors"]}.get(name)
    if ten is None:
        raise ValueError(f"{name}: missing from the dense reference")
    if ten["type"] != 8:
        raise ValueError(f"{name}: dense reference type {ten['type']}, want Q8_0")
    inn = ten["dims"][0]
    row_bytes = inn // 32 * 34
    with open(ref["path"], "rb") as f:
        f.seek(ten["abs"] + r0 * row_bytes)
        raw = f.read((r1 - r0) * row_bytes)
    if len(raw) != (r1 - r0) * row_bytes:
        raise ValueError(f"{name}: short read for rows [{r0},{r1})")
    blk = np.frombuffer(raw, np.uint8).reshape(r1 - r0, inn // 32, 34)
    d = blk[:, :, :2].copy().view("<f2").astype(np.float32)
    q = blk[:, :, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(r1 - r0, inn)


def recon_rows(f, r0, r1):
    """The factor's reconstruction of output rows [r0, r1), float32."""
    import numpy as np
    n = r1 - r0
    out, rank, inn = f["out"], f["rank"], f["inn"]
    u = np.unpackbits(np.frombuffer(f["u"], np.uint8).reshape(out, f["u_words"] * 4),
                      axis=1, bitorder="little")[r0:r1, :rank].astype(np.float32) * 2 - 1
    v = np.unpackbits(np.frombuffer(f["v"], np.uint8).reshape(rank, f["v_words"] * 4),
                      axis=1, bitorder="little")[:, :inn].astype(np.float32) * 2 - 1
    bf = lambda b: (np.frombuffer(b, "<u2").astype(np.uint32) << 16).view(np.float32)
    pre, mid, post = bf(f["pre"]), bf(f["mid"]), bf(f["post"])
    rec = (u @ (v * mid[:, None])) * post[r0:r1, None] * pre[None, :]
    idx = np.frombuffer(f["idx"], "<u4").astype(np.int64)
    sal = np.frombuffer(f["sal"], "<f2").astype(np.float32).reshape(out, len(idx))[r0:r1]
    for j, col in enumerate(idx):
        rec[np.arange(n), col] += sal[:, j]
    return rec


def bf16_words(values):
    """float32 values rounded to binary16, as little-endian words (round-to-nearest-even)."""
    import numpy as np
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    round = np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return ((bits + round) >> 16).astype("<u2")


def residual_planes(f, r0, r1, dense, layers):
    """The dense sign residual of rows [r0, r1).

    `R_0 = W - W_hat`, `alpha_j = mean|R_j|` per row, `W_hat += alpha_j*sign(R_j)`. Bit 1 is `+1`.
    Returns `(res words, s_res words, rel, that, res_sal)`: the pre-residual reconstruction,
    `W_hat`, and the residual evaluated at the salient columns, which the section's sparse values
    absorb so the precise columns carry no residual error.
    """
    import numpy as np
    n = r1 - r0
    res_words = (f["inn"] + 31) // 32
    idx = np.frombuffer(f["idx"], "<u4").astype(np.int64)
    that = recon_rows(f, r0, r1)
    r = dense - that
    bits = np.zeros((layers, n, res_words * 32), dtype=bool)
    scales = np.zeros((layers, n), dtype=np.float32)
    res_sal = np.zeros((n, len(idx)), dtype=np.float32)
    for l in range(layers):
        alpha = np.abs(r).mean(axis=1)
        scales[l] = alpha
        signed = r >= 0
        bits[l, :, : f["inn"]] = signed
        res_sal += (signed[:, idx].astype(np.float32) * 2 - 1) * alpha[:, None]
        r -= np.where(signed, alpha[:, None], -alpha[:, None])
    packed = np.packbits(bits, axis=2, bitorder="little")
    if packed.shape[2] % 4:
        packed = np.pad(packed, ((0, 0), (0, 0), (0, 4 - packed.shape[2] % 4)))
    words = np.frombuffer(packed.tobytes(), "<u4").reshape(layers * n, res_words)
    rel = float(np.linalg.norm(r) / np.linalg.norm(dense))
    return words.tobytes(), bf16_words(scales).tobytes(), rel, that, res_sal


def write_section(out, layer, kind, f, r0, r1, version, ref, residual, dense):
    """Write one pack section: rows [r0, r1) of the factor, sharing V/mid/pre/idx."""
    out_rows = r1 - r0
    u = row_slice(f["u"], f["out"], f["u_words"] * 4, r0, r1) if (r0, r1) != (0, f["out"]) else f["u"]
    post = row_slice(f["post"], f["out"], 2, r0, r1) if (r0, r1) != (0, f["out"]) else f["post"]
    sal = row_slice(f["sal"], f["out"], f["k"] * 2, r0, r1) if (r0, r1) != (0, f["out"]) else f["sal"]
    out.write(struct.pack(
        "<IIIIIIII",
        layer, kind, out_rows, f["inn"], f["rank"], f["u_words"], f["v_words"], f["k"],
    ))
    if version == VERSION:
        out.write(struct.pack("<I", residual))
    out.write(u)
    out.write(f["v"])
    out.write(f["pre"])
    out.write(f["mid"])
    out.write(post)
    out.write(f["idx"])
    rel = None
    if version == VERSION:
        import numpy as np
        w = dense_rows(ref, dense, 0, out_rows)
        res, s_res, rel, that, res_sal = residual_planes(f, r0, r1, w, residual)
        # The sparse values are refit after the residual: `y = U·V + S + R` is linear, so
        # `S = W_sal - (U·V)_sal - R_sal` restores the dense value at the columns stored in f16.
        idx = np.frombuffer(f["idx"], "<u4").astype(np.int64)
        sal_old = np.frombuffer(sal, "<f2").astype(np.float32).reshape(out_rows, len(idx))
        sal_new = (w[:, idx] - that[:, idx] + sal_old - res_sal).astype("<f2")
        out.write(sal_new.tobytes())
        out.write(res)
        out.write(s_res)
    else:
        out.write(sal)
    return out_rows, rel


def write_pack(path, src, layers, keep_dense=frozenset(), ref=None, residual=0):
    """Write the pack, omitting every `(layer, kind)` in `keep_dense`.

    With `residual` layers, writes version 3: every section carries `out × ceil(inn/32)` sign
    words and `out` binary16 row scales per layer, fitted against `ref`.
    """
    tensors = {t["name"]: t for t in src["tensors"]}
    q_out, kv_out = qkv_split(src["meta"], dims_of(tensors, "blk.0.attn_qkv.nq_u")[1])
    version = VERSION if residual else VERSION_BASE
    section_header = 36 if residual else 32
    total = 0
    weights = 0
    bits = 0
    linears = layers * len(KINDS) - len(keep_dense)
    rels = []
    with open(path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IIIIII", version, HEADER_BYTES, layers, len(KINDS),
                              linears, 0))
        total += HEADER_BYTES
        for layer in range(layers):
            for prefix, kinds in PREFIXES:
                f = linear_payload(src, tensors, f"blk.{layer}.{prefix}")
                if len(kinds) == 3:
                    # The fused qkv: q rows, then k, then v, each 32-word aligned.
                    if (f["out"] - q_out) % 2:
                        raise ValueError(f"blk.{layer}.attn_qkv: {f['out']} rows, q {q_out}")
                    spans = [(kinds[0], 0, q_out), (kinds[1], q_out, q_out + kv_out),
                             (kinds[2], q_out + kv_out, f["out"])]
                else:
                    spans = [(kinds[0], 0, f["out"])]
                for kind, r0, r1 in spans:
                    if (layer, kind) in keep_dense:
                        continue
                    dense = f"blk.{layer}.{DENSE_NAMES[KINDS[kind]]}"
                    rows, rel = write_section(out, layer, kind, f, r0, r1, version, ref, residual,
                                              dense)
                    if rel is not None:
                        rels.append(rel)
                    weights += rows * f["inn"]
                    bits += 32 * (rows * f["u_words"] + f["rank"] * f["v_words"] + f["k"])
                    bits += 16 * (f["inn"] + f["rank"] + rows + rows * f["k"])
                    total += section_header + rows * f["u_words"] * 4 + f["rank"] * f["v_words"] * 4
                    total += (f["inn"] + f["rank"] + rows) * 2 + rows * f["k"] * 2
                    total += f["k"] * 4
                    if residual:
                        bits += residual * (rows * f["v_words"] * 32 + rows * 16)
                        total += residual * (rows * f["v_words"] * 4 + rows * 2)
    if os.path.getsize(path) != total:
        raise ValueError(f"{path}: wrote {os.path.getsize(path)} B, want {total}")
    if rels:
        print(f"residual: {residual} layer(s), mean rel {sum(rels) / len(rels):.6f}, "
              f"max {max(rels):.6f}", flush=True)
    return total, weights, bits


def write_skeleton(path, src, ref=None, keep_dense=frozenset()):
    """Write the frozen tensors; a kept-dense linear comes from `ref` instead of the pack."""
    entries = []
    for ten in src["tensors"]:
        name = ten["name"]
        if ".nq_" in name:
            continue
        suffix = name.split(".", 2)[2] if name.startswith("blk.") else name
        if suffix in LINEAR_WEIGHTS:
            continue
        entries.append((src, ten))
    if keep_dense and ref is None:
        raise ValueError("--keep-dense needs a dense reference")
    dense_bytes = 0
    if keep_dense:
        rt = {t["name"]: t for t in ref["tensors"]}
        for layer, kind in sorted(keep_dense):
            name = f"blk.{layer}.{DENSE_NAMES[KINDS[kind]]}"
            ten = rt.get(name)
            if ten is None:
                raise ValueError(f"{name}: missing from the dense reference")
            entries.append((ref, ten))
            dense_bytes += ten["nbytes"]
    if not entries:
        raise ValueError("skeleton would be empty")
    n = nq_gguf.build_gguf_stream(
        path, src["meta"], [nq_gguf.frozen_entry(source, t) for source, t in entries]
    )
    return n, entries, dense_bytes


def parse_keep(items, layers):
    keep = set()
    for item in items:
        layer_s, sep, kind = item.partition(":")
        if not sep or kind not in KINDS:
            raise SystemExit(f"--keep-dense {item}: want L:kind with kind in {KINDS}")
        layer = int(layer_s)
        if not 0 <= layer < layers:
            raise SystemExit(f"--keep-dense {item}: layer outside 0..{layers - 1}")
        keep.add((layer, KINDS.index(kind)))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dense-ref", default="")
    ap.add_argument("--keep-dense", action="append", default=[],
                    help="L:kind served from --dense-ref instead of the pack")
    ap.add_argument("--residual", type=int, default=0,
                    help="dense sign-residual layers written into the pack (needs --dense-ref)")
    args = ap.parse_args()
    if args.residual < 0:
        raise SystemExit("--residual must be non-negative")
    os.makedirs(args.out_dir, exist_ok=True)
    src = nq_gguf.parse_gguf(args.source)
    layers = 1 + max(int(t["name"].split(".")[1]) for t in src["tensors"]
                     if t["name"].startswith("blk."))
    keep = parse_keep(args.keep_dense, layers)
    if keep and not args.dense_ref:
        raise SystemExit("--keep-dense needs --dense-ref")
    if args.residual and not args.dense_ref:
        raise SystemExit("--residual needs --dense-ref")
    ref = nq_gguf.parse_gguf(args.dense_ref) if (keep or args.residual) else None
    print(f"source: {args.source} {os.path.getsize(args.source)} B, {layers} layers, "
          f"{len(src['tensors'])} tensors, {len(keep)} dense linears, "
          f"{args.residual} residual layers", flush=True)

    pack_path = os.path.join(args.out_dir, "model.nqp")
    total, weights, bits = write_pack(pack_path, src, layers, keep, ref, args.residual)
    print(f"pack: {pack_path} {total} B, {layers * 7 - len(keep)} linears, "
          f"{bits / weights:.6f} b/w on {weights} fitted weights", flush=True)

    skeleton_path = os.path.join(args.out_dir, "skeleton.gguf")
    n, entries, dense_bytes = write_skeleton(skeleton_path, src, ref, keep)
    print(f"skeleton: {skeleton_path} {n} B, {len(entries)} tensors, "
          f"{dense_bytes} B of dense linears", flush=True)
    print(f"artifact: {total + n} B on disk", flush=True)


if __name__ == "__main__":
    main()
