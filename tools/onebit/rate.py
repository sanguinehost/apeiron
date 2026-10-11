#!/usr/bin/env python3
"""Gaussian rate of x -> Wx on the calibration support, per linear tensor of a GGUF model.

Distortion of an approximation W' is the output error on the calibration inputs:
    D(W') = sum over rows r of e_r H e_r^T,  e = W' - W,  H = X^T X / T (the Gram files).
Each column of a coded basis is modelled as an i.i.d. Gaussian source across the m rows with
variance s^2, weighted in the distortion by a per-column gain g. Reverse water-filling
(Shannon 1959; Cover & Thomas, Elements of Information Theory, 2nd ed., Thm 10.3.3) gives

    D(theta) = m * sum_k min(theta, v_k),  R(theta) = m * sum_k max(0, 1/2 log2(v_k / theta)),
    v_k = g_k s_k^2.

The Gaussian is the hardest source of a given variance, so R is an upper bound on the rate this
source model needs. Three bases:

  eig   g = eigenvalues of H, s^2 = row mean of (W U)^2 (U the eigenvectors of H): transform
        coding with the basis known to the decoder (U itself not counted).
  ldl   g = Var(x_j | x_later) (pivots of H's triangular factorization in GPTQ's column order),
        s^2 = row mean of W_j^2: sequential coding with error feedback in the stored basis;
        the decoder needs no transform (GPTQ, Frantar et al. 2022; LDLQ, Chee et al. 2023).
  diag  g = H_jj, s^2 as ldl: per-coordinate coding with no feedback.

Cost: per block, one eigh of each distinct Gram (n^3 flops, n <= 4864) and one Cholesky; per tensor
O(m n^2) for the distortion, O(n log(1/eps)) for the bisection. Memory O(n^2 + m n) f64.
"""
import argparse
import json
import math
import sys

import numpy as np

KINDS = [
    ("attn_q", "attn_in"),
    ("attn_k", "attn_in"),
    ("attn_v", "attn_in"),
    ("attn_output", "attn_out_in"),
    ("ffn_gate", "ffn_in"),
    ("ffn_up", "ffn_in"),
    ("ffn_down", "ffn_down_in"),
]


def tensors(path):
    from gguf import GGUFReader
    from gguf.quants import dequantize

    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        out[t.name] = (t, dequantize)
    return out


def q1_0(raw):
    """llama.cpp `dequantize_row_q1_0`: 18-byte blocks, f16 d then 128 sign bits LSB-first, bit 1 -> +d."""
    blk = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=np.uint8).reshape(-1, 18)
    d = blk[:, :2].copy().view("<f2").astype(np.float64)
    bits = np.unpackbits(blk[:, 2:], axis=1, bitorder="little").astype(np.float64)
    return (2.0 * bits - 1.0) * d


def load(tab, name):
    t, deq = tab[name]
    w = q1_0(t.data) if t.tensor_type.name == "Q1_0" else deq(t.data, t.tensor_type)
    m, n = int(t.shape[1]), int(t.shape[0])
    return np.asarray(w, dtype=np.float64).reshape(m, n), t.tensor_type.name


def rate(v, theta):
    pos = v > theta
    return 0.5 * np.log2(v[pos] / theta).sum()


def dist(v, theta):
    return np.minimum(v, theta).sum()


def theta_for_dist(v, d_row):
    """Water level with sum_k min(theta, v_k) = d_row; bisection in log space, 200 halvings."""
    if d_row >= v.sum():
        return float(v.max())
    lo, hi = math.log(max(v.min(), 1e-300) * 1e-6 + 1e-300), math.log(v.max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dist(v, math.exp(mid)) > d_row:
            hi = mid
        else:
            lo = mid
    return math.exp(0.5 * (lo + hi))


def theta_for_rate(v, r_row):
    lo, hi = math.log(max(v.min(), 1e-300) * 1e-12 + 1e-300), math.log(v.max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rate(v, math.exp(mid)) > r_row:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f16", required=True)
    ap.add_argument("--gram", required=True)
    ap.add_argument("--ref", action="append", default=[], help="label=path: distortion targets")
    ap.add_argument("--rates", default="1.125,1.185")
    ap.add_argument("--blocks", default="0-23")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.blocks.split("-"))
    tokens = int(open(f"{a.gram}/token_count.txt").read().split()[0])
    base = tensors(a.f16)
    refs = [(lab, tensors(p)) for lab, p in (s.split("=", 1) for s in a.ref)]
    rates = [float(x) for x in a.rates.split(",")]
    rows = []
    for b in range(lo, hi + 1):
        cache = {}
        for kind, gk in KINDS:
            name = f"blk.{b}.{kind}.weight"
            w, _ = load(base, name)
            m, n = w.shape
            if gk not in cache:
                h = np.fromfile(f"{a.gram}/blk.{b}.{gk}.f64", dtype="<f8").reshape(n, n) / tokens
                h = 0.5 * (h + h.T)
                lam, u = np.linalg.eigh(h)
                lam = np.clip(lam, 0.0, None)
                rev = h[::-1, ::-1]
                damp = 0.0
                while True:
                    try:
                        c = np.linalg.cholesky(rev + damp * np.eye(n))
                        break
                    except np.linalg.LinAlgError:
                        damp = max(damp * 10, 1e-9 * float(np.mean(np.diag(h))))
                piv = (np.diag(c) ** 2)[::-1].copy()
                cache[gk] = (h, lam, u, piv, damp)
            h, lam, u, piv, damp = cache[gk]
            col = (w * w).mean(axis=0)
            v = {
                "eig": lam * ((w @ u) ** 2).mean(axis=0),
                "ldl": piv * col,
                "diag": np.diag(h) * col,
            }
            row = {"tensor": name, "m": m, "n": n, "damp": damp, "out_power": float(((w @ h) * w).sum() / m)}
            for lab, tab in refs:
                wr, ty = load(tab, name)
                e = wr - w
                d_row = float(((e @ h) * e).sum() / m)
                row[f"{lab}.type"] = ty
                row[f"{lab}.d_row"] = d_row
                for basis, vb in v.items():
                    row[f"{lab}.{basis}.bpw"] = rate(vb, theta_for_dist(vb, d_row)) / n
            for r in rates:
                for basis, vb in v.items():
                    row[f"at{r}.{basis}.d_row"] = dist(vb, theta_for_rate(vb, r * n))
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump({"tokens": tokens, "rows": rows}, open(a.out, "w"))


if __name__ == "__main__":
    sys.path.insert(0, "~/llama.cpp/gguf-py")
    main()
