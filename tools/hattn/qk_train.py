#!/usr/bin/env python3
"""HATTN-T3: trained one-bit key codes on Qwen2.5 q, k over a 32768-token prose prefix.

Data: `attn_cut_probe` dumps (`q{l}.bin` [t, nh, hd], `k{l}.bin` [t, nkv, hd], f32, unrotated; `cs.bin` [t, hd/2]
(cos, sin) f64). RoPE pairs dimension j with j + hd/2. Float reference: causal softmax of q~.k~ / sqrt(hd).
Scale-fit rows: NTR query positions in [TR0, TR1); training pool: NPOOL positions in [TR0, TR1); held-out rows: NHO
positions in [HO0, t), attending over all keys s <= t. One key code per kv head is shared by its query heads (GQA),
as in the decode scan.

Arms (raw score before the scale; `R` = normalised Walsh-Hadamard times a fixed +-1 diagonal):
  S0  q . (sign(k) * mean|k|)                         the probe's KV_TAIL_LOWBIT=1 score
  S1  (R q) . (sign(R k) * mean|R k|)                  rotated one-bit key code
  L4  q . quant4(k)                                    the probe's KV_TAIL_LOWBIT=4 score
  SR  sign(R q) . sign(R k)                            one-bit q and k (T1 arm A at b = hd)
  ST  (U_h q) . (sign(W k + c) * mean|W k|)            trained: W, c per kv head, U_h, scale per query head; init S1
  SS  sign(U_h q + d_h) . sign(W k + c)                trained one-bit q and k; init SR
  STI as ST, init S0 (W = U_h = I)
Fixed arms get a grid-fitted scale on the scale-fit rows. Trained arms: Adam on minibatches from the pool (see
`train`), loss mean KL(float || arm), forward with exact signs, backward through sign as (1/sd)(1 - tanh^2(u/sd)) per code dimension
(straight-through estimator; Hubara et al., arXiv 1602.02830).

Metrics per query head on held-out rows: mean KL(float || arm) in nats; float mass of the top-B keys by the arm's
score, B = t * f for f in FRAC, and the same for the exact top-B (oracle).

  python3 tools/hattn/qk_train.py --check                      gradient check (finite difference) + init identity
  python3 tools/hattn/qk_train.py --model 7b --layers 2,6 [--fixed] [--out f.jsonl]   one JSON line per (layer, head, arm);
                                                               --fixed runs only S0, S1, L4, SR;
                                                               --groups 0,1 limits kv heads; --arms ST,STI limits trained arms
  python3 tools/hattn/qk_train.py --summary f.jsonl [...]      medians over heads
"""
import json
import sys
import time

import numpy as np

PRESET = {
    "0.5b": dict(dir="/home/socol/oc-tmp/hattn-long/g05-32k", nh=14, nkv=2, hd=64, nl=24),
    "7b": dict(dir="/home/socol/oc-tmp/hattn-long/g7-32k", nh=28, nkv=4, hd=128, nl=28),
}
TR0, TR1, HO0 = 4096, 16384, 24576
NTR, NHO = 64, 32
STEPS, LR, LR_S = 300, 2e-3, 2e-2
NPOOL, NB, NKEY, NTOP = 1024, 64, 4096, 16
FRAC = (1 / 256, 1 / 64, 1 / 16)
GRID = 10.0 ** np.linspace(-2, 2, 41)  # multiplier on the arm's natural scale


def walsh(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def make_R(hd):
    d = np.random.default_rng(0).choice([-1.0, 1.0], hd)
    return walsh(hd) / np.sqrt(hd) @ np.diag(d)


def sgn(x):
    return np.where(x >= 0, 1.0, -1.0)


def ste(u):
    sd = u.std(axis=0, keepdims=True) + 1e-12
    return (1.0 - np.tanh(u / sd) ** 2) / sd


def quant4(k):
    mx = np.abs(k).max(axis=1, keepdims=True)
    sc = np.where(mx > 0, mx / 7.0, 1.0)
    return np.round(k / sc) * sc


# ---------------------------------------------------------------- data
def load(cfg, l):
    hd, nh, nkv, half = cfg["hd"], cfg["nh"], cfg["nkv"], cfg["hd"] // 2
    d = cfg["dir"]
    k = np.fromfile(f"{d}/k{l}.bin", dtype="<f4")
    tn = k.size // (nkv * hd)
    k = k.reshape(tn, nkv, hd).astype(np.float64)
    cs = np.fromfile(f"{d}/cs.bin", dtype="<f8").reshape(tn, half, 2)
    q = np.memmap(f"{d}/q{l}.bin", dtype="<f4", mode="r", shape=(tn, nh, hd))
    rng = np.random.default_rng(1000 + l)
    tr = np.sort(rng.choice(np.arange(TR0, TR1), NTR, replace=False))
    ho = np.sort(rng.choice(np.arange(HO0, tn), NHO, replace=False))

    def rope(x, pos):
        c, s = cs[pos, :, 0], cs[pos, :, 1]
        if x.ndim == 3:
            c, s = c[:, None], s[:, None]
        x0, x1 = x[..., :half], x[..., half:]
        return np.concatenate([x0 * c - x1 * s, x1 * c + x0 * s], axis=-1)

    f = np.float32
    kt = rope(k, np.arange(tn)).astype(f)
    qtr = rope(np.asarray(q[tr], dtype=np.float64), tr).astype(f)
    qho = rope(np.asarray(q[ho], dtype=np.float64), ho).astype(f)
    pool = np.sort(rng.choice(np.arange(TR0, TR1), NPOOL, replace=False))
    qpool = rope(np.asarray(q[pool], dtype=np.float64), pool).astype(f)
    return kt, qtr, qho, tr, ho, qpool, pool


class Rows:
    """Float reference over query rows at positions `pos` against keys `keys` (default 0..max(pos)), causal."""

    def __init__(self, Q, K, pos, hd, keys=None):
        keys = np.arange(int(pos.max()) + 1) if keys is None else keys
        self.n = len(keys)
        self.mask = keys[None, :] <= pos[:, None]
        self.pos = pos
        X = Q @ K[keys].T / np.sqrt(hd)
        self.P, self.negH = softmax_masked(X, self.mask)


def softmax_masked(Z, mask):
    Z = np.where(mask, Z, -np.inf)
    mx = Z.max(axis=1, keepdims=True)
    e = np.exp(Z - mx)
    den = e.sum(axis=1, keepdims=True)
    P = e / den
    logP = np.where(mask, Z - mx - np.log(den), 0.0)
    return P, (P * logP).sum(axis=1)


def kl_rows(ref, Z):
    """Per-row KL(ref || softmax(Z)) and the softmax, one exp pass."""
    Zm = np.where(ref.mask, Z, -np.inf)
    mx = Zm.max(axis=1, keepdims=True)
    e = np.exp(Zm - mx)
    den = e.sum(axis=1, keepdims=True)
    lse = (mx + np.log(den))[:, 0]
    kl = ref.negH - (ref.P * np.where(ref.mask, Z, 0.0)).sum(axis=1) + lse
    return kl, e / den


def recall(ref, Z):
    """Float mass of the top-B keys by Z and by the exact score, per row, B = round(f * (t + 1))."""
    out, orc = [], []
    for f in FRAC:
        m, o = [], []
        for r, t in enumerate(ref.pos):
            B = max(1, int(round(f * (t + 1))))
            z, p = Z[r, : t + 1], ref.P[r, : t + 1]
            m.append(p[np.argpartition(-z, B - 1)[:B]].sum())
            o.append(p[np.argpartition(-p, B - 1)[:B]].sum())
        out.append(float(np.mean(m)))
        orc.append(float(np.mean(o)))
    return out, orc


# ---------------------------------------------------------------- arms
def raw_fixed(arm, Q, K, R, hd):
    """Raw score (rows, keys) of a fixed arm and its natural scale."""
    if arm == "S0":
        c = sgn(K) * np.abs(K).mean(axis=1, keepdims=True)
        return Q @ c.T, 1 / np.sqrt(hd)
    if arm == "S1":
        KR = K @ R.T
        c = sgn(KR) * np.abs(KR).mean(axis=1, keepdims=True)
        return (Q @ R.T) @ c.T, 1 / np.sqrt(hd)
    if arm == "L4":
        return Q @ quant4(K).T, 1 / np.sqrt(hd)
    if arm == "SR":
        return sgn(Q @ R.T) @ sgn(K @ R.T).T, 1.0 / hd
    raise ValueError(arm)


def fit_scale(refs, raws, base):
    best = (np.inf, 1.0)
    for g in GRID:
        m = np.mean([kl_rows(ref, base * g * X)[0].mean() for ref, X in zip(refs, raws)])
        if m < best[0]:
            best = (m, base * g)
    return best[1]


def forward(arm, p, Qs, K):
    """Raw scores per head and the cache for `backward`. p: W, c, U[h], d[h], ls[h]."""
    Uk0 = K @ p["W"].T
    Uk = Uk0 + p["c"]
    Bk = sgn(Uk)
    nk = np.abs(Uk0).mean(axis=1) if arm == "ST" else np.ones(len(K))
    M = Bk * nk[:, None]
    A, pre = [], []
    for h, Q in enumerate(Qs):
        x = Q @ p["U"][h].T
        if arm == "SS":
            x = x + p["d"][h]
            pre.append(x)
            x = sgn(x)
        A.append(x)
    raws = [a @ M.T for a in A]
    return raws, dict(Uk0=Uk0, Uk=Uk, Bk=Bk, nk=nk, M=M, A=A, pre=pre)


def loss_grad(arm, p, Qs, K, refs):
    raws, c = forward(arm, p, Qs, K)
    n = refs[0].n
    nrow = sum(len(r.pos) for r in refs)
    g = {k_: np.zeros_like(v) for k_, v in p.items()}
    dM = np.zeros_like(c["M"])
    L = 0.0
    for h, (ref, X) in enumerate(zip(refs, raws)):
        s = np.exp(p["ls"][h])
        kl, Qp = kl_rows(ref, s * X)
        L += kl.sum() / nrow
        G = np.where(ref.mask, Qp - ref.P, 0.0) / nrow
        g["ls"][h] = s * (G * X).sum()
        dX = s * G
        dA = dX @ c["M"][:n]
        dM[:n] += dX.T @ c["A"][h]
        if arm == "SS":
            dpre = dA * ste(c["pre"][h])
            g["U"][h] = dpre.T @ Qs[h]
            g["d"][h] = dpre.sum(axis=0)
        else:
            g["U"][h] = dA.T @ Qs[h]
    dB = dM * c["nk"][:, None]
    dUk = dB * ste(c["Uk"])
    g["W"] = dUk.T @ K
    g["c"] = dUk.sum(axis=0)
    if arm == "ST":
        dnk = (dM * c["Bk"]).sum(axis=1)
        dU0 = dnk[:, None] * sgn(c["Uk0"]) / K.shape[1]
        g["W"] += dU0.T @ K
    return L, g


def pool_top(Qs, K, pos):
    """Per head, the exact top-NTOP key indices s <= t of every pool row. Time O(rows * t * hd)."""
    tops = []
    for Q in Qs:
        T_ = np.empty((len(pos), NTOP), dtype=np.int64)
        for a in range(0, len(pos), 128):
            sl = slice(a, a + 128)
            n = int(pos[sl].max()) + 1
            X = np.where(np.arange(n)[None, :] <= pos[sl, None], Q[sl] @ K[:n].T, -np.inf)
            T_[sl] = np.argpartition(-X, NTOP - 1, axis=1)[:, :NTOP]
        tops.append(T_)
    return tops


def train(arm, p, Qs, K, pool, tops, hd, seed):
    """Adam on minibatches: NB pool positions per step, keys = NKEY uniform prefix keys plus every batch row's exact
    top-NTOP, fresh each step. Each row's reference and arm softmax run over that key subset (causal)."""
    rng = np.random.default_rng(seed)
    m = {k_: np.zeros_like(v) for k_, v in p.items()}
    v = {k_: np.zeros_like(x) for k_, x in p.items()}
    hist = []
    for i in range(1, STEPS + 1):
        b = np.sort(rng.choice(len(pool), NB, replace=False))
        pos = pool[b]
        n = int(pos.max()) + 1
        keys = np.unique(np.concatenate([rng.choice(n, min(NKEY, n), replace=False)] + [t[b].ravel() for t in tops]))
        Ks = K[keys]
        refs = [Rows(Q[b], K, pos, hd, keys) for Q in Qs]
        L, g = loss_grad(arm, p, [Q[b] for Q in Qs], Ks, refs)
        hist.append(L)
        for k_ in p:
            lr = LR_S if k_ == "ls" else LR
            m[k_] = 0.9 * m[k_] + 0.1 * g[k_]
            v[k_] = 0.999 * v[k_] + 0.001 * g[k_] ** 2
            p[k_] = p[k_] - lr * (m[k_] / (1 - 0.9**i)) / (np.sqrt(v[k_] / (1 - 0.999**i)) + 1e-12)
    return p, hist


def init_params(M0, nq, hd, scale, dt=np.float64):
    return dict(
        W=M0.astype(dt),
        c=np.zeros(hd, dt),
        U=np.stack([M0.astype(dt) for _ in range(nq)]),
        d=np.zeros((nq, hd), dt),
        ls=np.full(nq, np.log(scale), dt),
    )


# ---------------------------------------------------------------- run
def run_layer(cfg, l, out, trained=True, groups=None, arms=("ST", "SS", "STI")):
    t0 = time.time()
    hd, nh, nkv = cfg["hd"], cfg["nh"], cfg["nkv"]
    G = nh // nkv
    R = make_R(hd).astype(np.float32)
    kt, qtr, qho, tr, ho, qpool, pool = load(cfg, l)
    for g in range(nkv) if groups is None else groups:
        K = kt[:, g]
        heads = range(g * G, (g + 1) * G)
        Qtr = [qtr[:, h] for h in heads]
        Qho = [qho[:, h] for h in heads]
        rtr = [Rows(Q, K, tr, hd) for Q in Qtr]
        rho = [Rows(Q, K, ho, hd) for Q in Qho]
        scales = {}
        for arm in ("S0", "S1", "L4", "SR"):
            ntr = rtr[0].n
            raws = [raw_fixed(arm, Q, K[:ntr], R, hd)[0] for Q in Qtr]
            base = raw_fixed(arm, Qtr[0][:1], K[:1], R, hd)[1]
            s = fit_scale(rtr, raws, base)
            scales[arm] = s
            for j, (h, Q, ref) in enumerate(zip(heads, Qho, rho)):
                Z = s * raw_fixed(arm, Q, K[: ref.n], R, hd)[0]
                kl, _ = kl_rows(ref, Z)
                rc, orc = recall(ref, Z)
                out.write(json.dumps(dict(layer=l, head=h, kv=g, arm=arm, kl=float(kl.mean()), recall=rc, oracle=orc, scale=s)) + "\n")
        if trained:
            Qp = [qpool[:, h] for h in heads]
            tops = pool_top(Qp, K, pool)
        for name, arm, M0, init in (("ST", "ST", R, "S1"), ("SS", "SS", R, "SR"), ("STI", "ST", np.eye(hd, dtype=np.float32), "S0")) if trained else ():
            if name not in arms:
                continue
            p = init_params(M0, G, hd, scales[init], np.float32)
            p, hist = train(arm, p, Qp, K, pool, tops, hd, 7 * l + g)
            raws, _ = forward(arm, p, Qho, K[: rho[0].n])
            for j, (h, ref) in enumerate(zip(heads, rho)):
                Z = np.exp(p["ls"][j]) * np.where(np.arange(rho[0].n)[None, :] <= ref.pos[:, None], raws[j], 0.0)
                kl, _ = kl_rows(ref, Z)
                rc, orc = recall(ref, Z)
                out.write(json.dumps(dict(layer=l, head=h, kv=g, arm=name, kl=float(kl.mean()), recall=rc, oracle=orc, loss0=float(np.mean(hist[:10])), loss=float(np.mean(hist[-10:])))) + "\n")
        out.flush()
        print(f"layer {l} kv {g} done {time.time() - t0:.0f} s", file=sys.stderr)


def summary(paths):
    rows = [json.loads(x) for p_ in paths for x in open(p_)]
    arms = ["S0", "S1", "L4", "SR", "ST", "SS", "STI"]
    nh = len({(r["layer"], r["head"]) for r in rows})
    print(f"heads {nh}, layers {sorted({r['layer'] for r in rows})}")
    print(f"{'arm':<4} | {'KL med':>8}{'KL p90':>8} | " + "".join(f"{'mass@' + str(round(1 / f)):>14}" for f in FRAC) + " | oracle mass (same B)")
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        if not rs:
            continue
        kl = np.array([r["kl"] for r in rs])
        rc = np.array([r["recall"] for r in rs])
        orc = np.array([r["oracle"] for r in rs])
        print(f"{a:<4} | {np.median(kl):8.4f}{np.percentile(kl, 90):8.4f} | " + "".join(f"{np.mean(rc[:, i]):14.4f}" for i in range(len(FRAC))) + " | " + " ".join(f"{np.mean(orc[:, i]):.4f}" for i in range(len(FRAC))))


# ---------------------------------------------------------------- check
def run_check():
    ok = True
    rng = np.random.default_rng(3)
    hd, n, nq, rows = 8, 40, 2, 6
    R = make_R(hd)
    K = rng.standard_normal((n, hd)) * 2
    pos = np.sort(rng.choice(np.arange(10, n), rows, replace=False))
    Qs = [rng.standard_normal((rows, hd)) for _ in range(nq)]
    refs = [Rows(Q, K, pos, hd) for Q in Qs]
    Kn = K[: refs[0].n]
    # init identity: ST at init equals S1, SS at init equals SR
    for arm, base in (("ST", "S1"), ("SS", "SR")):
        p = init_params(R, nq, hd, 0.3)
        raws, _ = forward(arm, p, Qs, Kn)
        e = max(float(np.abs(x - raw_fixed(base, Q, Kn, R, hd)[0]).max()) for x, Q in zip(raws, Qs))
        print(f"init {arm} = {base}: max diff {e:.2e} (need < 1e-12)")
        ok &= e < 1e-12
    # gradient check on the continuous parameters (U, ls for ST; ls for SS): sign is piecewise constant
    p = init_params(R, nq, hd, 0.3)
    p["U"] += rng.standard_normal(p["U"].shape) * 0.1
    _, g = loss_grad("ST", p, Qs, Kn, refs)
    worst = 0.0
    for key, idx in (("U", (1, 2, 3)), ("U", (0, 5, 1)), ("ls", (0,)), ("ls", (1,))):
        e_ = 1e-6
        pp = {k_: v.copy() for k_, v in p.items()}
        pp[key][idx] += e_
        Lp, _ = loss_grad("ST", pp, Qs, Kn, refs)
        pp[key][idx] -= 2 * e_
        Lm, _ = loss_grad("ST", pp, Qs, Kn, refs)
        fd = (Lp - Lm) / (2 * e_)
        worst = max(worst, abs(fd - g[key][idx]) / max(1e-9, abs(fd)))
    print(f"ST gradient vs finite difference (U, log-scale): max rel err {worst:.2e} (need < 1e-5)")
    ok &= worst < 1e-5
    # STE path through nk only (W at fixed signs): the nk term is exact where codes do not flip
    print("check", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    a = sys.argv
    if "--check" in a:
        sys.exit(run_check())
    if "--summary" in a:
        summary(a[a.index("--summary") + 1 :])
        sys.exit(0)
    if "--model" in a:
        cfg = PRESET[a[a.index("--model") + 1]]
        layers = [int(x) for x in a[a.index("--layers") + 1].split(",")] if "--layers" in a else range(cfg["nl"])
        out = open(a[a.index("--out") + 1], "a") if "--out" in a else sys.stdout
        for l in layers:
            groups = [int(x) for x in a[a.index("--groups") + 1].split(",")] if "--groups" in a else None
            arms = tuple(a[a.index("--arms") + 1].split(",")) if "--arms" in a else ("ST", "SS", "STI")
            run_layer(cfg, l, out, "--fixed" not in a, groups, arms)
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
