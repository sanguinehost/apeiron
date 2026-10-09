#!/usr/bin/env python3
"""HATTN-T1: does one-bit attention on the hyperbolic face survive an existing model?

Qwen2.5-0.5B (Q4_K_M) prompt 0, T = 512, 24 layers, 14 query heads, 2 KV heads (GQA 7:1), head dim 64.
q, k dumps are unrotated; RoPE uses the engine's own table cs.bin (z = x[:32] + i x[32:], rotated z * E_t).
Float reference: causal softmax over s <= t of (q~_t . k~_s) / 8.

Arms (signs are taken after RoPE and after R or P_b):
  A  score = s * sum_i sign(P q~)_i sign(P k~)_i,                 b in {8,12,16,32,64}, and b = 64 through R
  B  score = c * (|k~_s| / mean|k~|) * sum_i sign(.)_i sign(.)_i, same projections (key norm carried per key)
  C  float score with q~ rescaled to the head's mean query norm
  E  float score with k~ rescaled to the head's mean key norm, scale fitted (extension: key-norm loss)
  G  sign sum plus additive per-key bias g*sum + beta*log(|k~|/mean|k~|), b=64 via R (extension: carry-compatible key norm)
  F  float cosine q~.k~/(|q~||k~|) with a fitted scale             (extension: sign loss = A - F, exact angles kept)
  D  kernel sum_{j<=r} tanh(s)^j e_j(x), x = q (.) k in {+-1}^64 through R, r in {1,2,3}

  python3 tools/hattn/qk_onebit.py --check     identity control, exact 2^8 carry, D(r=64) = A kernel
  python3 tools/hattn/qk_onebit.py --full      tables to stdout (timing on stderr; stdout is deterministic)
"""
import itertools
import os
import sys
import time

import numpy as np

D = os.environ.get("HATTN_QK_DIR", "/home/socol/oc-tmp/faces-qk")
T, HD, H2, NH, NKV, NL = 512, 64, 32, 14, 2, 24
TFIT = 256
GRID = 10.0 ** np.linspace(-3, 1, 64)
BETA = np.array([-8.0, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8])
GRID_F = 10.0 ** np.linspace(-1, 4, 64)  # arm F scores are cosines in [-1, 1], so the scale is ~|q||k|/8
BITS = [8, 12, 16, 32, 64]


# ---------------------------------------------------------------- causal row sets
class Rows:
    """Rows r0..r1-1 of the causal matrix, flattened row-major (row t has t+1 entries)."""

    def __init__(self, r0, r1):
        ti, si = np.tril_indices(T)
        sel = (ti >= r0) & (ti < r1)
        self.ti, self.si = ti[sel], si[sel]
        self.lens = np.arange(r0, r1) + 1
        self.off = np.concatenate([[0], np.cumsum(self.lens)[:-1]])

    def flat(self, M):
        return M[self.ti, self.si]

    def rsum(self, v):
        return np.add.reduceat(v, self.off)

    def rmax(self, v):
        return np.maximum.reduceat(v, self.off)

    def rep(self, v):
        return np.repeat(v, self.lens)


class Ref:
    """Float attention on a row set: P and sum_s P log P per row."""

    def __init__(self, rows, Xf):
        mx = rows.rmax(Xf)
        e = np.exp(Xf - rows.rep(mx))
        den = rows.rsum(e)
        self.P = e / rows.rep(den)
        self.negH = rows.rsum(self.P * (Xf - rows.rep(mx + np.log(den))))


def kl_tv(rows, ref, Z):
    """Per-row KL(float || softmax(Z)) in nats and TV distance."""
    mx = rows.rmax(Z)
    e = np.exp(Z - rows.rep(mx))
    den = rows.rsum(e)
    kl = ref.negH - rows.rsum(ref.P * Z) + mx + np.log(den)
    tv = 0.5 * rows.rsum(np.abs(ref.P - e / rows.rep(den)))
    return kl, tv


def fit_scale(rows, ref, X, grid=None):
    """Grid fit of g > 0 minimising mean KL of softmax(g X) over rows 1.. of `rows`."""
    grid = GRID if grid is None else grid
    mx = rows.rmax(X)
    Xc = X - rows.rep(mx)
    PX = rows.rsum(ref.P * X)
    out = np.empty(len(grid))
    for i, g in enumerate(grid):
        kl = ref.negH - g * PX + g * mx + np.log(rows.rsum(np.exp(g * Xc)))
        out[i] = kl[1:].mean()
    return int(np.argmin(out)), out


# ---------------------------------------------------------------- data
def load_E():
    cs = np.fromfile(f"{D}/cs.bin", dtype="<f8").reshape(T, H2, 2)
    return cs[..., 0] + 1j * cs[..., 1]


def rope(x, E):
    z = (x[:, :H2] + 1j * x[:, H2:]) * E
    return np.concatenate([z.real, z.imag], axis=1)


def load_layer(l, E):
    q = np.fromfile(f"{D}/q{l}.bin", dtype="<f4").reshape(T, NH, HD).astype(np.float64)
    k = np.fromfile(f"{D}/k{l}.bin", dtype="<f4").reshape(T, NKV, HD).astype(np.float64)
    qt = np.stack([rope(q[:, h], E) for h in range(NH)])
    kt = np.stack([rope(k[:, h], E) for h in range(NKV)])
    return qt, kt


def walsh(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def make_R():
    d = np.random.default_rng(0).choice([-1.0, 1.0], HD)
    return walsh(HD) / np.sqrt(HD) @ np.diag(d)


def make_P():
    return np.random.default_rng(1).standard_normal((HD, HD))


def sgn(y):
    return np.where(y >= 0, 1.0, -1.0)


# ---------------------------------------------------------------- arm D weights
def d_table():
    """Ef[m, j]: coefficient of z^j in (1+z)^m (1-z)^(64-m), exact integers then float."""
    rows = []
    for m in range(HD + 1):
        p = [1]
        for factor in [(1, 1)] * m + [(1, -1)] * (HD - m):
            q = [0] * (len(p) + 1)
            for j, c in enumerate(p):
                q[j] += c * factor[0]
                q[j + 1] += c * factor[1]
            p = q
        rows.append(p)
    return np.array(rows, dtype=np.float64)


EF = d_table()


def d_kernel(r, t):
    """tab[g, m] = sum_{j<=r} t_g^j e_j(m)."""
    t = np.atleast_1d(t)
    return sum((t[:, None] ** j) * EF[None, :, j] for j in range(r + 1))


# ---------------------------------------------------------------- checks
def carry_lambda(d, s):
    lam = np.empty(1 << d)
    for S in range(1 << d):
        w = bin(S).count("1")
        lam[S] = np.cosh(s) ** (d - w) * np.sinh(s) ** w
    return lam


def carry_features(x, d):
    """phi_S(x) = prod_{i in S} x_i for all S, as a (n, 2^d) array (bitmask order)."""
    out = np.ones((x.shape[0], 1 << d))
    for S in range(1 << d):
        for i in range(d):
            if S >> i & 1:
                out[:, S] *= x[:, i]
    return out


def check_carry():
    rng = np.random.default_rng(7)
    n, d, dv, s = 1000, 8, 4, 0.37
    K = rng.choice([-1.0, 1.0], (n, d))
    V = rng.standard_normal((n, dv))
    q = rng.choice([-1.0, 1.0], (1, d))
    w = np.exp(s * (K @ q[0]))
    direct = (w[:, None] * V).sum(0) / w.sum()
    lam = carry_lambda(d, s)
    PK = carry_features(K, d)
    num, den = PK.T @ V, PK.sum(0)
    pq = carry_features(q, d)[0]
    read = (lam * pq) @ num / ((lam * pq) @ den)
    return float(np.abs(read - direct).max())


def check_identity():
    E = load_E()
    qt, kt = load_layer(0, E)
    R = make_R()
    rows = Rows(0, T)
    worst = 0.0
    for h in range(NH):
        k = kt[h // (NH // NKV)]
        ref = Ref(rows, rows.flat(qt[h] @ k.T / 8.0))
        Z = rows.flat((qt[h] @ R.T) @ (k @ R.T).T / 8.0)
        kl, _ = kl_tv(rows, ref, Z)
        worst = max(worst, float(np.abs(kl).max()))
    return worst, float(np.abs(R @ R.T - np.eye(HD)).max())


def check_d64():
    rng = np.random.default_rng(11)
    n, s = 400, 0.05
    K = rng.choice([-1.0, 1.0], (n, HD))
    q = rng.choice([-1.0, 1.0], HD)
    Sg = K @ q
    m = ((HD + Sg) / 2).astype(int)
    a = np.exp(s * (Sg - Sg.max()))
    a /= a.sum()
    w = d_kernel(HD, np.tanh(s))[0][m]
    w = w / w.sum()
    return float(np.abs(a - w).max())


def run_check():
    t0 = time.time()
    ok = True
    e2 = check_carry()
    print(f"(ii)  exact 2^8 carry vs softmax attention, n=1000 d=8 s=0.37 dv=4: max diff {e2:.3e} (need < 1e-12)")
    if not e2 < 1e-12:
        print("FAIL check (ii)")
        ok = False
    e3 = check_d64()
    print(f"(iii) arm D at r=64 vs arm A kernel, b=64 s=0.05: max diff {e3:.3e} (need < 1e-9)")
    if not e3 < 1e-9:
        print("FAIL check (iii)")
        ok = False
    e1, orth = check_identity()
    print(f"(i)   identity control layer 0, s=1/8, R on both: max KL {e1:.3e} (need < 1e-10); |RR^T-I| {orth:.1e}")
    if not e1 < 1e-10:
        print("FAIL check (i)")
        ok = False
    print(f"check {'OK' if ok else 'FAILED'} in {time.time() - t0:.1f} s", file=sys.stderr)
    return 0 if ok else 1


# ---------------------------------------------------------------- full run
def st(a):
    a = np.asarray(a)
    return np.median(a), np.percentile(a, 90), a.max()


def run_full(nl):
    t0 = time.time()
    E = load_E()
    R, P = make_R(), make_P()
    r_tr, r_ho = Rows(0, TFIT), Rows(TFIT, T)
    variants = [("P", b) for b in BITS] + [("R", 64)]
    proj = {("P", b): P[:b] for b in BITS}
    proj[("R", 64)] = R
    res = {}
    ent, coll = [], {}

    def put(key, l, h, **kw):
        res.setdefault(key, {})[(l, h)] = kw

    for l in range(nl):
        qt, kt = load_layer(l, E)
        # collisions: distinct key codes among the first t keys, per kv head
        for kv in range(NKV):
            for v in variants:
                codes = sgn(kt[kv] @ proj[v].T) > 0
                packed = np.packbits(codes, axis=1)
                seen, cnt = set(), {}
                for i in range(T):
                    seen.add(packed[i].tobytes())
                    if i + 1 in (64, 128, 256, 512):
                        cnt[i + 1] = len(seen)
                coll.setdefault(v, []).append(cnt)
        sq, sk = {}, {}
        for v in variants:
            sq[v] = sgn(qt @ proj[v].T)
            sk[v] = sgn(kt @ proj[v].T)
        for h in range(NH):
            kv = h // (NH // NKV)
            q, k = qt[h], kt[kv]
            Xf = q @ k.T / 8.0
            ref_tr = Ref(r_tr, r_tr.flat(Xf))
            ref_ho = Ref(r_ho, r_ho.flat(Xf))
            ent.append((-(ref_tr.negH[1:]).sum() - ref_ho.negH.sum()) / (T - 1))
            wk = np.linalg.norm(k, axis=1)
            wk = wk / wk.mean()
            for v in variants:
                Sg = sq[v][h] @ sk[v][kv].T
                for arm, M in (("A", Sg), ("B", Sg * wk[None, :])):
                    Xtr, Xho = r_tr.flat(M), r_ho.flat(M)
                    i, curve = fit_scale(r_tr, ref_tr, Xtr)
                    kl, tv = kl_tv(r_ho, ref_ho, GRID[i] * Xho)
                    put((arm, v), l, h, kl_ho=kl.mean(), tv_ho=tv.mean(), kl_in=curve[i], edge=i in (0, len(GRID) - 1), lo=i == 0)
            # C: query norm removed
            qn = np.linalg.norm(q, axis=1)
            qc = q * (qn.mean() / qn)[:, None]
            Zc = r_ho.flat(qc @ k.T / 8.0)
            kl, tv = kl_tv(r_ho, ref_ho, Zc)
            put(("C", None), l, h, kl_ho=kl.mean(), tv_ho=tv.mean())
            # E: float with the key norm removed; F: float cosine only (both norms removed), scale fitted
            kn = np.linalg.norm(k, axis=1)
            ke = k * (kn.mean() / kn)[:, None]
            Xe = q @ ke.T / 8.0
            i, curve = fit_scale(r_tr, ref_tr, r_tr.flat(Xe))
            kl, tv = kl_tv(r_ho, ref_ho, GRID[i] * r_ho.flat(Xe))
            put(("E", None), l, h, kl_ho=kl.mean(), tv_ho=tv.mean(), kl_in=curve[i], edge=i in (0, len(GRID) - 1), lo=i == 0)
            Cos = (q / qn[:, None]) @ (k / kn[:, None]).T
            i, curve = fit_scale(r_tr, ref_tr, r_tr.flat(Cos), GRID_F)
            kl, tv = kl_tv(r_ho, ref_ho, GRID_F[i] * r_ho.flat(Cos))
            put(("F", None), l, h, kl_ho=kl.mean(), tv_ho=tv.mean(), kl_in=curve[i], edge=i in (0, len(GRID_F) - 1), lo=i == 0)
            # G: sign sum plus a per-key additive bias, score = g sum + beta log(|k~_s|/mean|k~|), b = 64 through R
            Sg = sq[("R", 64)][h] @ sk[("R", 64)][kv].T
            u = np.log(kn / kn.mean())
            Xtr, Xho = r_tr.flat(Sg), r_ho.flat(Sg)
            Utr, Uho = u[r_tr.si], u[r_ho.si]
            best = (np.inf, 0, 0)
            for bi, beta in enumerate(BETA):
                for gi, g in enumerate(GRID):
                    kl, _ = kl_tv(r_tr, ref_tr, g * Xtr + beta * Utr)
                    m_ = kl[1:].mean()
                    if m_ < best[0]:
                        best = (m_, gi, bi)
            kl, tv = kl_tv(r_ho, ref_ho, GRID[best[1]] * Xho + BETA[best[2]] * Uho)
            put(("G", None), l, h, kl_ho=kl.mean(), tv_ho=tv.mean(), kl_in=best[0], edge=best[1] in (0, len(GRID) - 1) or best[2] in (0, len(BETA) - 1), lo=best[1] == 0, beta=BETA[best[2]])
            # D: degree truncation, b = 64 through R
            Sg = sq[("R", 64)][h] @ sk[("R", 64)][kv].T
            m_tr = ((HD + r_tr.flat(Sg)) / 2).astype(int)
            m_ho = ((HD + r_ho.flat(Sg)) / 2).astype(int)
            for r in (1, 2, 3):
                tab = d_kernel(r, np.tanh(GRID))
                curve = np.empty(len(GRID))
                for i in range(len(GRID)):
                    W = tab[i][m_tr]
                    den = r_tr.rsum(W)
                    ok = den > 0
                    q_ = W / r_tr.rep(np.where(ok, den, 1.0))
                    tv = np.where(ok, 0.5 * r_tr.rsum(np.abs(ref_tr.P - q_)), 1.0)
                    curve[i] = tv[1:].mean()
                i = int(np.argmin(curve))
                W = tab[i][m_ho]
                den = r_ho.rsum(W)
                ok = den > 0
                q_ = W / r_ho.rep(np.where(ok, den, 1.0))
                tv = np.where(ok, 0.5 * r_ho.rsum(np.abs(ref_ho.P - q_)), 1.0)
                put(("D", r), l, h, tv_ho=tv.mean(), tv_in=curve[i], neg=float((W < 0).mean()), bad=float((~ok).mean()), edge=i in (0, len(GRID) - 1))
        print(f"layer {l} done {time.time() - t0:.0f} s", file=sys.stderr)

    # ---------------- report
    print(f"HATTN-T1 qk_onebit --full: layers {nl}, heads {nl * NH}, T={T}, fit rows t<{TFIT} (grid {len(GRID)} log-spaced 1e-3..1e1), held-out rows t>={TFIT}")
    print(f"float attention mean row entropy (nats): median {np.median(ent):.3f} p90 {np.percentile(ent, 90):.3f} max {max(ent):.3f}")
    print("\nHeld-out per-head mean-row KL(float||arm), nats, over (layer,head)  [in-sample = rows 1..255 at the fitted scale]")
    print(f"{'arm':<4}{'proj':<5}{'b':>3} | {'ho med':>8}{'ho p90':>8}{'ho max':>8} | {'in med':>8}{'in p90':>8}{'in max':>8} | {'TV med':>7} | scale at grid edge (low = near-uniform fit)")
    tabk = {}
    for arm in ("A", "B"):
        for v in variants:
            r = res[(arm, v)]
            ho = [x["kl_ho"] for x in r.values()]
            ino = [x["kl_in"] for x in r.values()]
            tvs = [x["tv_ho"] for x in r.values()]
            edge = np.mean([x["edge"] for x in r.values()])
            lo_ = np.mean([x["lo"] for x in r.values()])
            tabk[(arm, v)] = np.median(ho)
            a, b_, c = st(ho)
            d_, e_, f_ = st(ino)
            print(f"{arm:<4}{v[0]:<5}{v[1]:>3} | {a:8.4f}{b_:8.4f}{c:8.4f} | {d_:8.4f}{e_:8.4f}{f_:8.4f} | {np.median(tvs):7.4f} | {100 * edge:.1f}% (low {100 * lo_:.1f}%)")
    for arm in ("C", "E", "F", "G"):
        r = res[(arm, None)]
        ho = [x["kl_ho"] for x in r.values()]
        a, b_, c = st(ho)
        if arm in ("E", "F", "G"):
            d_, e_, f_ = st([x["kl_in"] for x in r.values()])
            edge = f"{100 * np.mean([x['edge'] for x in r.values()]):.1f}% (low {100 * np.mean([x['lo'] for x in r.values()]):.1f}%)"
            ins = f"{d_:8.4f}{e_:8.4f}{f_:8.4f}"
        else:
            edge = "-"
            ins = f"{'-':>8}{'-':>8}{'-':>8}"
        print(f"{arm:<4}{'-':<5}{'-':>3} | {a:8.4f}{b_:8.4f}{c:8.4f} | {ins} | {np.median([x['tv_ho'] for x in r.values()]):7.4f} | {edge}")
    gb = [x["beta"] for x in res[("G", None)].values()]
    print(f"G: fitted beta (grid {BETA.min():g}..{BETA.max():g}) median {np.median(gb):g}; fraction beta=0 {np.mean(np.array(gb) == 0):.3f}; fraction at beta grid end {np.mean(np.abs(gb) == BETA.max()):.3f}")
    print("\nArm D (b=64 through R): held-out TV(float, arm) over (layer,head); weights are signed")
    print(f"{'r':>2} | {'TV med':>8}{'TV p90':>8}{'TV max':>8} | {'in med':>8} | {'neg weights med':>16}{'p90':>7}{'max':>7} | invalid rows (norm<=0) mean | scale at grid edge")
    for r_ in (1, 2, 3):
        r = res[("D", r_)]
        tv = [x["tv_ho"] for x in r.values()]
        ng = [x["neg"] for x in r.values()]
        a, b_, c = st(tv)
        n1, n2, n3 = st(ng)
        print(f"{r_:>2} | {a:8.4f}{b_:8.4f}{c:8.4f} | {np.median([x['tv_in'] for x in r.values()]):8.4f} | {n1:16.4f}{n2:7.4f}{n3:7.4f} | {np.mean([x['bad'] for x in r.values()]):.4f} | {100 * np.mean([x['edge'] for x in r.values()]):.1f}%")
    print("\nLayer medians of held-out KL (nats); D rows are TV")
    print(f"{'arm':<9}" + "".join(f"{l:6d}" for l in range(nl)))
    for key in [("A", v) for v in variants] + [("B", v) for v in variants] + [("C", None), ("E", None), ("F", None), ("G", None)] + [("D", r_) for r_ in (1, 2, 3)]:
        f = "tv_ho" if key[0] == "D" else "kl_ho"
        row = [np.median([res[key][(l, h)][f] for h in range(NH)]) for l in range(nl)]
        name = f"{key[0]}{'' if key[1] is None else ('/' + (key[1][0] + str(key[1][1]) if isinstance(key[1], tuple) else 'r' + str(key[1])))}"
        print(f"{name:<9}" + "".join(f"{x:6.3f}" for x in row))
    print("\nDistinct key codes among the first t keys, per kv head (RoPE-rotated keys)")
    print(f"{'proj':<5}{'b':>3} | " + "".join(f"{'t=' + str(t):>20}" for t in (64, 128, 256, 512)) + "   (median distinct | fraction of kv heads with distinct = t)")
    for v in variants:
        cells = []
        for t in (64, 128, 256, 512):
            vals = [c[t] for c in coll[v]]
            cells.append(f"{np.median(vals):8.0f} | {np.mean([x == t for x in vals]):.2f}   ")
        print(f"{v[0]:<5}{v[1]:>3} | " + "".join(f"{c:>20}" for c in cells))
    # ---------------- verdicts
    print("\nVerdicts")
    a64 = tabk[("A", ("R", 64))]
    b64 = tabk[("B", ("R", 64))]
    print(f"P1 A b=64 (R) held-out median KL {a64:.4f} > 0.05 -> {'held' if a64 > 0.05 else 'REFUTED'}")
    print(f"P2 B/A at b=64 (R): {b64:.4f}/{a64:.4f} = {b64 / a64:.3f}; need <= 0.5 -> {'held' if b64 <= a64 / 2 else 'REFUTED'}")
    f16 = np.mean([c[512] == 512 for c in coll[("P", 16)]])
    f8 = np.mean([c[512] == 512 for c in coll[("P", 8)]])
    ok3 = all(np.mean([c[512] == 512 for c in coll[("P", b)]]) >= 0.9 for b in (16, 32, 64)) and f8 < 0.9
    print(f"P3 fraction of kv heads with distinct = 512 at t=512: b=8 {f8:.2f}, b=12 {np.mean([c[512] == 512 for c in coll[('P', 12)]]):.2f}, b=16 {f16:.2f}, b=32 {np.mean([c[512] == 512 for c in coll[('P', 32)]]):.2f}, b=64 {np.mean([c[512] == 512 for c in coll[('P', 64)]]):.2f}; need >=0.90 for b>=16 and collisions at b=8 -> {'held' if ok3 else 'REFUTED'}")
    d3 = np.median([x["tv_ho"] for x in res[("D", 3)].values()])
    a16 = np.median([x["tv_ho"] for x in res[("A", ("P", 16))].values()])
    print(f"P4 D r=3 median TV {d3:.4f} vs A b=16 median TV {a16:.4f} -> {'held' if d3 > a16 else 'REFUTED'}")
    meets = [(key, val) for key, val in tabk.items() if val <= 0.05]
    meets.sort(key=lambda kv_: (kv_[0][1][1], kv_[0][0]))
    if meets:
        key, val = meets[0]
        print(f"Decision: cheapest sign arm with held-out median KL <= 0.05 nats: {key[0]} b={key[1][1]} ({key[1][0]}) at {val:.4f}")
    else:
        best = min(tabk.items(), key=lambda kv_: kv_[1])
        print(f"Decision: no sign arm reaches 0.05 nats; best is {best[0][0]} b={best[0][1][1]} ({best[0][1][0]}) at {best[1]:.4f}; float C (query norm removed) {np.median([x['kl_ho'] for x in res[('C', None)].values()]):.4f}, E (key norm removed) {np.median([x['kl_ho'] for x in res[('E', None)].values()]):.4f}, F (cosine only, exact angles) {np.median([x['kl_ho'] for x in res[('F', None)].values()]):.4f}")
    print(f"elapsed {time.time() - t0:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(run_check())
    if "--full" in sys.argv:
        nl = NL
        if "--layers" in sys.argv:
            nl = int(sys.argv[sys.argv.index("--layers") + 1])
        sys.exit(run_full(nl))
    print(__doc__)
    sys.exit(2)
