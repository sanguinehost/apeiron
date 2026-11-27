#!/usr/bin/env python3
"""Add a `Q4_K` or `Q6_K` `output.weight` to a GGUF skeleton from its tied F16 embedding.

The engine's LM head prefers `output.weight` and falls back to the tied embedding. A `Q8_0`
head is 1.0625 bytes per weight; the K-quant heads are 0.5625 (`Q4_K`) and 0.8203 (`Q6_K`),
and both take the fused block reader (`q8act_q4_k_row`, `q8act_q6_k_row`).

The quantizers are ports of llama.cpp's reference ones at `049326a`
(`quantize_row_q4_K_ref`, `quantize_row_q6_K_ref` in `ggml/src/ggml-quants.c`), including the
weighted iterative `make_qkx2_quants` search and the signed `make_qx_quants` search, in float32.
The stored image is what the reader decodes, so compare logits against the source before serving.

Usage:
  nq_head_k.py --in skeleton.gguf --out skeleton-q4khead.gguf --type q4_k
  nq_head_k.py --in skeleton.gguf --out skeleton-q6khead.gguf --type q6_k
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_gguf

Q4_K = 12
Q6_K = 14
QK_K = 256
ROWS_PER_CHUNK = 512


def _make_qkx2(x, w, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20):
    """`make_qkx2_quants` over the last axis: returns (scale, min_out, L).

    `x`, `w`: `(..., 32)` float32. `scale` and `min_out` are `(...,)`; `L` is `(..., 32)`
    in `0..=nmax`. `min_out` is `-min` as in the reference, so it is non-negative.
    """
    mn = np.minimum(x.min(axis=-1, keepdims=True), np.float32(0.0))
    mx = x.max(axis=-1, keepdims=True)
    span = mx - mn
    zero = span == 0
    span_safe = np.where(zero, np.float32(1.0), span)
    iscale = np.float32(nmax) / span_safe
    scale = np.float32(1.0) / iscale
    L = np.clip(np.rint(iscale * (x - mn)), 0, nmax)
    best = (w * (scale * L + mn - x) ** 2).sum(axis=-1)
    gscale = np.squeeze(scale, axis=-1).copy()
    gmin = np.squeeze(mn, axis=-1).copy()
    sum_x = (w * x).sum(axis=-1)
    sum_w = w.sum(axis=-1)
    for step in range(nstep + 1):
        isc = (np.float32(rmin) + np.float32(rdelta) * np.float32(step) + np.float32(nmax)) / span_safe
        laux = np.clip(np.rint(isc * (x - mn)), 0, nmax)
        sum_l = (w * laux).sum(axis=-1)
        sum_l2 = (w * laux * laux).sum(axis=-1)
        sum_xl = (w * laux * x).sum(axis=-1)
        d = sum_w * sum_l2 - sum_l * sum_l
        ok = d > 0
        ds = np.where(ok, d, np.float32(1.0))
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / ds
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / ds
        pos = this_min > 0
        this_min = np.where(pos, np.float32(0.0), this_min)
        l2s = np.where(sum_l2 > 0, sum_l2, np.float32(1.0))
        this_scale = np.where(pos, sum_xl / l2s, this_scale)
        cur = (w * (this_scale[..., None] * laux + this_min[..., None] - x) ** 2).sum(axis=-1)
        better = ok & (cur < best)
        if better.any():
            L = np.where(better[..., None], laux, L)
            best = np.where(better, cur, best)
            gscale = np.where(better, this_scale, gscale)
            gmin = np.where(better, this_min, gmin)
    return gscale, -gmin, L


def quantize_q4_k(x):
    """`(R, nb, 8, 32)` float32 -> `(R, nb, 144)` bytes, llama.cpp's reference Q4_K."""
    rows, blocks = x.shape[0], x.shape[1]
    av = np.sqrt((x * x).mean(axis=-1, keepdims=True))
    w = av + np.abs(x)
    scales, mins, L = _make_qkx2(x, w)
    max_scale = scales.max(axis=-1)
    max_min = mins.max(axis=-1)
    inv_scale = np.where(max_scale > 0, np.float32(63.0) / max_scale, np.float32(0.0))
    inv_min = np.where(max_min > 0, np.float32(63.0) / max_min, np.float32(0.0))
    ls = np.minimum(63, np.rint(inv_scale[..., None] * scales)).astype(np.uint8)
    lm = np.minimum(63, np.rint(inv_min[..., None] * mins)).astype(np.uint8)
    sc = np.zeros((rows, blocks, 12), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            sc[..., j] = ls[..., j]
            sc[..., j + 4] = lm[..., j]
        else:
            sc[..., j + 4] = (ls[..., j] & 0xF) | ((lm[..., j] & 0xF) << 4)
            sc[..., j - 4] |= (ls[..., j] >> 4) << 6
            sc[..., j] |= (lm[..., j] >> 4) << 6
    d = (max_scale / np.float32(63.0)).astype("<f2")
    dmin = (max_min / np.float32(63.0)).astype("<f2")
    # The stored scale/min decode decides the final codes, as in the reference.
    d32 = d.astype(np.float32)
    dmin32 = dmin.astype(np.float32)
    sub_sc = np.zeros((rows, blocks, 8), dtype=np.uint8)
    sub_m = np.zeros((rows, blocks, 8), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            sub_sc[..., j] = sc[..., j] & 63
            sub_m[..., j] = sc[..., j + 4] & 63
        else:
            sub_sc[..., j] = (sc[..., j + 4] & 0xF) | ((sc[..., j - 4] >> 6) << 4)
            sub_m[..., j] = (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4)
    dj = d32[..., None] * sub_sc
    dm = dmin32[..., None] * sub_m
    live = dj > 0
    lf = np.clip(np.rint((x + dm[..., None]) / np.where(live, dj, np.float32(1.0))[..., None]), 0, 15)
    L = np.where(live[..., None], lf, L)
    l2 = L.reshape(rows, blocks, QK_K).astype(np.uint8)
    q = np.zeros((rows, blocks, 128), dtype=np.uint8)
    for t in range(4):
        lo = l2[..., t * 64 : t * 64 + 32]
        hi = l2[..., t * 64 + 32 : t * 64 + 64]
        q[..., t * 32 : (t + 1) * 32] = lo | (hi << 4)
    blk = np.zeros((rows, blocks, 144), dtype=np.uint8)
    blk[..., 0:2] = d.view(np.uint8).reshape(rows, blocks, 2)
    blk[..., 2:4] = dmin.view(np.uint8).reshape(rows, blocks, 2)
    blk[..., 4:16] = sc
    blk[..., 16:144] = q
    return blk


def _make_qx(x, nmax=32, rmse_type=1):
    """`make_qx_quants` over the last axis: returns (scale, L) with `L` in `0..=2*nmax-1`."""
    ax = np.abs(x)
    amax = ax.max(axis=-1)
    idx = ax.argmax(axis=-1)
    mx = np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]
    zero = amax < np.float32(1e-15)
    mxs = np.where(zero, np.float32(1.0), mx)
    iscale = -np.float32(nmax) / mxs
    w = x * x if rmse_type == 1 else np.ones_like(x)
    l = np.clip(np.rint(iscale[..., None] * x), -nmax, nmax - 1)
    L = (l + nmax).astype(np.int8)
    sumlx = (w * x * l).sum(axis=-1)
    suml2 = (w * l * l).sum(axis=-1)
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, np.float32(1.0)), np.float32(0.0))
    best = scale * sumlx
    for step in range(-9, 10):
        if step == 0:
            continue
        isc = -(np.float32(nmax) + np.float32(0.1) * np.float32(step)) / mxs
        l = np.clip(np.rint(isc[..., None] * x), -nmax, nmax - 1)
        sumlx = (w * x * l).sum(axis=-1)
        suml2 = (w * l * l).sum(axis=-1)
        better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        if better.any():
            L = np.where(better[..., None], (l + nmax).astype(np.int8), L)
            scale = np.where(better, sumlx / np.where(suml2 > 0, suml2, np.float32(1.0)), scale)
            best = np.where(better, scale * sumlx, best)
    return np.where(zero, np.float32(0.0), scale), L


def quantize_q6_k(x):
    """`(R, nb, 16, 16)` float32 -> `(R, nb, 210)` bytes, llama.cpp's reference Q6_K."""
    rows, blocks = x.shape[0], x.shape[1]
    scales, L = _make_qx(x)
    abs_scale = np.abs(scales)
    pick = abs_scale.argmax(axis=-1)
    max_abs = np.take_along_axis(abs_scale, pick[..., None], axis=-1)[..., 0]
    max_scale = np.take_along_axis(scales, pick[..., None], axis=-1)[..., 0]
    zero = max_abs < np.float32(1e-15)
    mxs = np.where(zero, np.float32(1.0), max_scale)
    iscale = -np.float32(128.0) / mxs
    d = (np.float32(1.0) / iscale).astype("<f2")
    q = np.minimum(127, np.rint(iscale[..., None] * scales)).astype(np.int8)
    d32 = d.astype(np.float32)
    dj = d32[..., None] * q.astype(np.float32)
    live = dj != 0
    lf = np.clip(np.rint(x / np.where(live, dj, np.float32(1.0))[..., None]), -32, 31)
    L = np.where(live[..., None], (lf + 32).astype(np.int8), L)
    L = np.where(zero[..., None, None], np.int8(0), L)
    q = np.where(zero[..., None], np.int8(0), q)
    l2 = L.reshape(rows, blocks, QK_K).astype(np.uint8)
    ql = np.zeros((rows, blocks, 128), dtype=np.uint8)
    qh = np.zeros((rows, blocks, 64), dtype=np.uint8)
    for j in range(2):
        base = j * 128
        for l in range(32):
            q1 = l2[..., base + l] & 0xF
            q2 = l2[..., base + l + 32] & 0xF
            q3 = l2[..., base + l + 64] & 0xF
            q4 = l2[..., base + l + 96] & 0xF
            ql[..., j * 64 + l] = q1 | (q3 << 4)
            ql[..., j * 64 + l + 32] = q2 | (q4 << 4)
            qh[..., j * 32 + l] = (
                (l2[..., base + l] >> 4)
                | ((l2[..., base + l + 32] >> 4) << 2)
                | ((l2[..., base + l + 64] >> 4) << 4)
                | ((l2[..., base + l + 96] >> 4) << 6)
            )
    blk = np.zeros((rows, blocks, 210), dtype=np.uint8)
    blk[..., 0:128] = ql
    blk[..., 128:192] = qh
    blk[..., 192:208] = q.view(np.uint8)
    blk[..., 208:210] = d.view(np.uint8).reshape(rows, blocks, 2)
    return blk


def encode(src, name, dims, out, kind):
    n_in, rows = dims
    if n_in % QK_K:
        raise SystemExit(f"{name}: {n_in} columns is not a multiple of {QK_K}")
    blocks = n_in // QK_K
    block_bytes = 144 if kind == "q4_k" else 210
    row_bytes = n_in * 2
    got = 0
    with open(src, "rb") as f:
        ten = next(t for t in nq_gguf.parse_gguf(src)["tensors"] if t["name"] == name)
        f.seek(ten["abs"])
        while got < rows:
            take = min(ROWS_PER_CHUNK, rows - got)
            raw = f.read(take * row_bytes)
            if len(raw) != take * row_bytes:
                raise SystemExit(f"{name}: short at row {got}")
            if kind == "q4_k":
                x = np.frombuffer(raw, "<f2").astype(np.float32).reshape(take, blocks, 8, 32)
                blk = quantize_q4_k(x)
            else:
                x = np.frombuffer(raw, "<f2").astype(np.float32).reshape(take, blocks, 16, 16)
                blk = quantize_q6_k(x)
            out.write(blk.tobytes())
            got += take
    return rows * blocks * block_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--type", dest="kind", choices=("q4_k", "q6_k"), default="q4_k")
    a = ap.parse_args()
    h = nq_gguf.parse_gguf(a.src)
    emb = next((t for t in h["tensors"] if t["name"] == "token_embd.weight"), None)
    if emb is None or emb["type"] != 1:
        raise SystemExit(f"{a.src}: token_embd.weight is not F16")
    n_in, vocab = emb["dims"]
    ggml_type = Q4_K if a.kind == "q4_k" else Q6_K
    block_bytes = 144 if a.kind == "q4_k" else 210
    nbytes = vocab * (n_in // QK_K) * block_bytes

    entries = [
        nq_gguf.frozen_entry(h, t)
        for t in h["tensors"]
        if t["name"] != "output.weight"
    ]

    def write_head(out, src=a.src, name=emb["name"], dims=emb["dims"], nbytes=nbytes):
        wrote = encode(src, name, dims, out, a.kind)
        if wrote != nbytes:
            raise SystemExit(f"output.weight: wrote {wrote} B, want {nbytes}")

    entries.append(
        {
            "name": "output.weight",
            "dims": list(emb["dims"]),
            "type": ggml_type,
            "nbytes": nbytes,
            "write": write_head,
        }
    )
    total = nq_gguf.build_gguf_stream(a.dst, h["meta"], entries)
    print(f"{a.dst}: {total} B, {len(entries)} tensors, output.weight {a.kind} {nbytes} B")


if __name__ == "__main__":
    sys.exit(main())
