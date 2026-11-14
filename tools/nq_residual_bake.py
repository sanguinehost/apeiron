#!/usr/bin/env python3
"""Bake an NQ reconstruction, plus optional dense sign residual layers, into a Q8_0 GGUF.

The engine reads a served NQ linear as `s_post·U·diag(s_mid)·V·s_pre + salient`. This tool writes
that reconstruction back into a Q8_0 reference image, one weight at a time, so the existing Q8_0
path measures the representation's perplexity with no engine or pack-format change. With
`--residual N` it adds N dense sign-correction layers

    R_0 = W - W_hat,   alpha_j = mean|R_j| per row,   W_hat += alpha_j·sign(R_j),

then restores the sparse salient columns to `W`, which is the closed-form fit a version-3 pack
stores. The Q8_0 image still re-quantises every 32-block, so the pack through the engine, not this
image, is the measurement for the dense residual.

Both images are streamed: the source and the output are never both resident. Requires numpy.

    nq_residual_bake.py --ref Qwen3-8B-Base-Q8_0.gguf --nq qwen3-8b-nanoquant.gguf \
        --out /tmp/baked-1.gguf --residual 1 --layers 0-35 --kinds gate,up,down
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_gguf
import nq_convert

BLOCK = 34
QK = 32

KIND_PREFIX = {}
for _p, _ks in nq_convert.PREFIXES:
    for _k in _ks:
        KIND_PREFIX[nq_convert.KINDS[_k]] = _p


def bf16_words(b):
    return (np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def f16_words(b):
    return np.frombuffer(b, dtype="<f2").astype(np.float32)


def tensor_map(image):
    return {t["name"]: t for t in image["tensors"]}


def payload(image, name):
    t = tensor_map(image)[name]
    with open(image["path"], "rb") as f:
        f.seek(t["abs"])
        b = f.read(t["nbytes"])
    if len(b) != t["nbytes"]:
        raise ValueError(f"{name}: {len(b)} B, want {t['nbytes']}")
    return b


class Factor:
    """One fitted linear: the sign factors, the three scales and the salient columns."""

    def __init__(self, nq, prefix):
        ten = tensor_map(nq)
        u = payload(nq, prefix + ".nq_u")
        v = payload(nq, prefix + ".nq_v")
        u_dims = ten[prefix + ".nq_u"]["dims"]
        v_dims = ten[prefix + ".nq_v"]["dims"]
        self.u_words, self.out = u_dims
        self.v_words, self.rank = v_dims
        self.inn = self.v_words * 32
        self.u = u
        self.v = v
        self.pre = bf16_words(payload(nq, prefix + ".nq_scale_pre"))
        self.mid = bf16_words(payload(nq, prefix + ".nq_scale_mid"))
        self.post = bf16_words(payload(nq, prefix + ".nq_scale_post"))
        self.idx = np.frombuffer(payload(nq, prefix + ".nq_salient_idx"), dtype="<u4").astype(np.int64)
        self.sal = f16_words(payload(nq, prefix + ".nq_salient_weight")).reshape(self.out, len(self.idx))
        if len(self.pre) != self.inn or len(self.mid) != self.rank or len(self.post) != self.out:
            raise ValueError(f"{prefix}: scale lengths disagree with {self.inn}/{self.rank}/{self.out}")
        sign = np.unpackbits(np.frombuffer(v, dtype=np.uint8).reshape(self.rank, self.v_words * 4),
                             axis=1, bitorder="little")[:, : self.inn].astype(np.float32) * 2 - 1
        self.mv = sign * self.mid[:, None]
        del sign

    def rows(self, r0, r1):
        """`W_hat` rows [r0, r1), float32."""
        n = r1 - r0
        words = np.frombuffer(self.u[r0 * self.u_words * 4 : r1 * self.u_words * 4], dtype=np.uint8)
        u = np.unpackbits(words.reshape(n, self.u_words * 4), axis=1, bitorder="little")
        u = u[:, : self.rank].astype(np.float32) * 2 - 1
        out = (u @ self.mv) * self.post[r0:r1, None] * self.pre[None, :]
        for t, col in enumerate(self.idx):
            out[np.arange(n), col] += self.sal[r0:r1, t]
        return out


def decode_q8(raw, rows, inn):
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(rows, inn // QK, BLOCK)
    d = blk[:, :, :2].copy().view("<f2").astype(np.float32)
    q = blk[:, :, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(rows, inn)


def encode_q8(w):
    rows, inn = w.shape
    blk = w.reshape(rows, inn // QK, QK)
    amax = np.abs(blk).max(axis=2)
    d = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float16)
    q = np.rint(blk / d[..., None].astype(np.float32)).clip(-128, 127).astype(np.int8)
    out = np.empty((rows, inn // QK, BLOCK), dtype=np.uint8)
    out[:, :, :2] = d.view(np.uint8).reshape(rows, inn // QK, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.tobytes()


def bake_tensor(ref_path, ten, factor, r0, r1, layers, stats):
    """The write callback for one ref tensor fed from `factor` rows [r0, r1)."""
    inn = int(ten["dims"][0])
    rows = r1 - r0
    row_bytes = inn // QK * BLOCK
    step = max(1, (8 << 20) // row_bytes)

    def write(out):
        sq0 = sqn = norm = 0.0
        with open(ref_path, "rb") as rf:
            rf.seek(ten["abs"])
            for c0 in range(0, rows, step):
                c1 = min(rows, c0 + step)
                w = decode_q8(rf.read((c1 - c0) * row_bytes), c1 - c0, inn)
                that = factor.rows(r0 + c0, r0 + c1)
                r = w - that
                sq0 += float((r * r).sum())
                for _ in range(layers):
                    a = np.abs(r).mean(axis=1)
                    that += np.sign(r) * a[:, None]
                    r = w - that
                if layers:
                    # Refit the sparse salient values after the residual, as the v3 pack does:
                    # those columns are stored at f16 precision because they dominate the loss,
                    # and a sign step would otherwise move each by alpha.
                    cols = np.arange(c1 - c0)[:, None]
                    that[cols, factor.idx] = w[cols, factor.idx]
                    r = w - that
                sqn += float((r * r).sum())
                norm += float((w * w).sum())
                out.write(encode_q8(that))
        rel0 = (sq0 / norm) ** 0.5
        reln = (sqn / norm) ** 0.5
        stats.append((ten["name"], sq0, sqn, norm))
        print(f"  {ten['name']}: rel {rel0:.6f} -> {reln:.6f} "
              f"(sq x{reln * reln / (rel0 * rel0):.4f})", flush=True)

    return {"name": ten["name"], "dims": ten["dims"], "type": nq_gguf.GGML_Q8_0,
            "nbytes": nq_gguf.ggml_nbytes(nq_gguf.GGML_Q8_0, ten["dims"]), "write": write}


def parse_layers(spec, count):
    if spec == "all":
        return list(range(count))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < count]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="Q8_0 GGUF carrying the served dense weights")
    ap.add_argument("--nq", required=True, help="the arelath NQ source GGUF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--kinds", default=",".join(nq_convert.KINDS))
    ap.add_argument("--residual", type=int, default=0, help="dense sign layers (0, 1, 2, ...)")
    args = ap.parse_args()

    ref = nq_gguf.parse_gguf(args.ref)
    nq = nq_gguf.parse_gguf(args.nq)
    ref_ten = tensor_map(ref)
    nq_names = set(tensor_map(nq))
    layers_n = 1 + max(int(n.split(".")[1]) for n in nq_names if n.startswith("blk."))
    layers = parse_layers(args.layers, layers_n)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in nq_convert.KINDS:
            raise SystemExit(f"unknown kind {k}")

    q_out = kv_out = None
    if "blk.0.attn_qkv.nq_u" in nq_names:
        q_out, kv_out = nq_convert.qkv_split(nq["meta"], tensor_map(nq)["blk.0.attn_qkv.nq_u"]["dims"][1])

    # (ref tensor name, factor prefix, factor row span) per selected (layer, kind)
    jobs = {}
    for layer in layers:
        for kind in kinds:
            dense = f"blk.{layer}.{nq_convert.DENSE_NAMES[kind]}"
            if dense not in ref_ten:
                raise SystemExit(f"{dense} missing from {args.ref}")
            if kind in ("q", "k", "v"):
                prefix = f"blk.{layer}.attn_qkv"
                if prefix + ".nq_u" not in nq_names:
                    prefix = f"blk.{layer}.attn_{kind}"
                    if prefix + ".nq_u" not in nq_names:
                        raise SystemExit(f"no factor for blk.{layer}.attn_{kind}")
                    span = (0, None)
                else:
                    if q_out is None:
                        raise SystemExit("fused qkv without head geometry")
                    base = {"q": 0, "k": q_out, "v": q_out + kv_out}[kind]
                    span = (base, base + (q_out if kind == "q" else kv_out))
            else:
                prefix = f"blk.{layer}.{KIND_PREFIX[kind]}"
                span = (0, None)
            jobs[dense] = (prefix, span)

    stats = []
    entries = []
    factors = {}
    factor_layer = None
    for ten in ref["tensors"]:
        job = jobs.get(ten["name"])
        if job is None:
            entries.append(nq_gguf.frozen_entry(ref, ten))
            continue
        if ten["type"] != nq_gguf.GGML_Q8_0:
            raise SystemExit(f"{ten['name']}: type {ten['type']}, want Q8_0")
        prefix, (r0, r1) = job
        layer = int(ten["name"].split(".")[1])
        if layer != factor_layer:
            factors.clear()
            factor_layer = layer
        f = factors.get(prefix)
        if f is None:
            f = Factor(nq, prefix)
            factors[prefix] = f
        if r1 is None:
            r1 = f.out
        if r1 > f.out:
            raise SystemExit(f"{ten['name']}: rows [{r0},{r1}) outside {f.out}")
        print(f"bake {ten['name']} <- {prefix}[{r0}:{r1}] rank {f.rank} "
              f"layers {args.residual}", flush=True)
        entries.append(bake_tensor(ref["path"], ten, f, r0, r1, args.residual, stats))

    total = nq_gguf.build_gguf_stream(args.out, ref["meta"], entries)
    print(f"wrote {args.out}: {total} B, {len(stats)} tensors baked")
    if stats:
        s0 = sum(x[1] for x in stats)
        sn = sum(x[2] for x in stats)
        norm = sum(x[3] for x in stats)
        print(f"aggregate rel: {(s0 / norm) ** 0.5:.6f} -> {(sn / norm) ** 0.5:.6f}")


if __name__ == "__main__":
    main()
