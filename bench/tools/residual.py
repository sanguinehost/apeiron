#!/usr/bin/env python3
"""Measure the residual of one served NQ linear and what a dense sign layer buys.

Reconstructs W_hat = s_post . (U diag(s_mid) V) . s_pre + salient on a row sample,
compares with the Q8_0 reference, then reports the sign-quantiser energy capture.
Usage: residual.py <nq.gguf> <ref.gguf> <blk.N.ffn_gate> [rows]
"""
import struct, sys
import numpy as np

BLOCK = 34  # Q8_0

def header(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, = struct.unpack("<I", f.read(4))
    nt, nkv = struct.unpack("<QQ", f.read(16))
    align = 32
    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")
    def skip(t):
        if t in (0, 1, 7): f.read(1)
        elif t in (2, 3): f.read(2)
        elif t in (4, 5, 6): f.read(4)
        elif t in (10, 11, 12): f.read(8)
        elif t == 8:
            n, = struct.unpack("<Q", f.read(8)); f.seek(n, 1)
        elif t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n): skip(et)
        else: raise ValueError(t)
    def val(t):
        if t == 0: return f.read(1)[0]
        if t == 4: return struct.unpack("<I", f.read(4))[0]
        if t == 5: return struct.unpack("<i", f.read(4))[0]
        if t == 6: return struct.unpack("<f", f.read(4))[0]
        if t == 8: return rstr()
        if t == 10: return struct.unpack("<Q", f.read(8))[0]
        if t == 11: return struct.unpack("<q", f.read(8))[0]
        if t == 12: return struct.unpack("<d", f.read(8))[0]
        if t == 9:
            skip(t); return None
        skip(t); return None
    for _ in range(nkv):
        k = rstr(); t, = struct.unpack("<I", f.read(4)); v = val(t)
        if k == "general.alignment": align = int(v) if v else 32
    ten = {}
    for _ in range(nt):
        name = rstr(); nd, = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tt, off = struct.unpack("<IQ", f.read(12))
        ten[name] = (dims, tt, off)
    start = (f.tell() + align - 1) // align * align
    return f, start, ten

def slice_of(f, start, ten, name):
    dims, tt, off = ten[name]
    n = int(np.prod(dims))
    size = {0: 4, 1: 2, 24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 30: 2}.get(tt)
    if size is None: raise ValueError(f"{name}: type {tt} size unknown")
    f.seek(start + off)
    return f.read(n * size), dims

def bf16(b):
    u = np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32)

def f16(b):
    return np.frombuffer(b, dtype="<f2").astype(np.float32)

def bits_of(b, width):
    """LSB-first sign bits of packed u32 words -> +-1 float, one row of `width`."""
    u = np.frombuffer(b, dtype="<u4")
    raw = np.unpackbits(u.view(np.uint8), bitorder="little")[:width]
    return raw.astype(np.float32) * 2.0 - 1.0

def main():
    nqp, refp, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    f, start, ten = header(nqp)
    u_b, u_dims = slice_of(f, start, ten, prefix + ".nq_u")
    v_b, v_dims = slice_of(f, start, ten, prefix + ".nq_v")
    pre_b, pre_d = slice_of(f, start, ten, prefix + ".nq_scale_pre")
    mid_b, mid_d = slice_of(f, start, ten, prefix + ".nq_scale_mid")
    post_b, post_d = slice_of(f, start, ten, prefix + ".nq_scale_post")
    idx_b, idx_d = slice_of(f, start, ten, prefix + ".nq_salient_idx")
    sal_b, sal_d = slice_of(f, start, ten, prefix + ".nq_salient_weight")
    f.close()
    u_words, out_rows = u_dims
    v_words, rank = v_dims
    inn = v_words * 32
    k = idx_d[0]
    print(f"{prefix}: out={out_rows} inn={inn} rank={rank} k={k}")
    pre = bf16(pre_b); mid = bf16(mid_b); post = bf16(post_b)
    idx = np.frombuffer(idx_b, dtype="<u4").astype(np.int64)
    sal = f16(sal_b).reshape(out_rows, k)

    # V: rank x inn, +-1
    vb = np.frombuffer(v_b, dtype=np.uint8).reshape(rank, v_words * 4)
    V = np.unpackbits(vb, axis=1, bitorder="little")[:, :inn].astype(np.float32) * 2 - 1
    MV = V * mid[:, None]

    # U sample rows
    ub = np.frombuffer(u_b, dtype=np.uint8).reshape(out_rows, u_words * 4)
    U = np.unpackbits(ub[:rows], axis=1, bitorder="little")[:, :rank].astype(np.float32) * 2 - 1
    recon = (U @ MV) * post[:rows, None] * pre[None, :]
    for t in range(k):
        np.add.at(recon, (slice(None), idx[t]), sal[:rows, t])

    # reference Q8_0 rows
    g, gstart, gten = header(refp)
    name = prefix + ".weight"
    gdims, gtt, goff = gten[name]
    assert gtt == 8, f"ref type {gtt}"
    g.seek(gstart + goff)
    row_bytes = inn // 32 * BLOCK
    raw = np.frombuffer(g.read(rows * row_bytes), dtype=np.uint8)
    g.close()
    blk = raw.reshape(rows, inn // 32, BLOCK)
    d = blk[:, :, :2].copy().view("<f2").astype(np.float32)
    q = blk[:, :, 2:].view(np.int8).astype(np.float32)
    W = (q * d).reshape(rows, inn)

    err = W - recon
    rel = float(np.linalg.norm(err) / np.linalg.norm(W))
    print(f"  reconstruction rel={rel:.6f} cos={1 - rel*rel/2:.6f}")

    R = err
    alpha = np.abs(R).mean(axis=1)
    S = np.sign(R) * alpha[:, None]
    rho = float((S * S).sum() / (R * R).sum())
    rel1 = rel * (1 - rho) ** 0.5
    rel2 = rel * (1 - rho)
    print(f"  residual sign capture rho={rho:.6f} (Gaussian 2/pi={2/np.pi:.6f})")
    print(f"  one dense sign layer -> rel={rel1:.6f}; two -> rel={rel2:.6f}")
    # per-row spread of the capture, and the row-norm share of the worst rows
    per = ((S * S).sum(axis=1) / (R * R).sum(axis=1))
    print(f"  per-row capture min={per.min():.4f} p10={np.percentile(per,10):.4f} "
          f"median={np.median(per):.4f} max={per.max():.4f}")

if __name__ == "__main__":
    main()
