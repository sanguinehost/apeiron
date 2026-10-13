"""ONEBIT-side-information probe (spec: 2026-10-14-w16-t6-sideinfo.md, sections 1 and 6).

R = Ws - Wp is the residual of the -nb quantiser. In the Hessian-whitened Hadamard basis E = fwht128(R) S, S = Hr^{1/2},
Hr = had_gram(Hl_m), and ||E||^2 = J(Ws, Wp, Hl_m). The probe prices two routes:
  (a) encode E at a tolerance: reverse water-filling over the eigenvalues of E^T E / O  (R_a(f));
  (b) predict E from decoded junction context (Wyner-Ziv side information): out-of-fold ridge R^2_H and the rate of E - Ehat.

subcommands
  capture --variants V --seeds 0 --dump PATH --cap DIR --out JSON   (GPU; wraps ft.fold_fit, runs the normal -nb path)
  probe   --cap DIR --dump PATH --out JSON [--synthetic]            (synthetic controls run in-process first, abort on miss)

Notation. An 8-block e in R^8 is eight consecutive whitened columns of one row; b is the block index along the columns.
F2 scale vectors are written alpha (column) and beta (row) as in the spec; F0/F3 use b for the block index.
Run with ~/spinor-ttl/.venv/bin/python.
"""
import argparse, bisect, hashlib, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fold_transfer as ft

F64 = torch.float64
KINDS = ["q", "k", "v", "o", "gate", "up", "down"]
STAGE_OF = {nm: st for st, nms, _ in ft.STAGES for nm in nms}
BLK = 8
FS = (0.5, 0.9, 0.98479)
F_STAR = 0.98479
RIDGE_REL = 1e-3
GATE_REL = 1e-4
# model key -> (family, arm, shuffle).  F0 is self context; Fk d = decoded partner; Fk o = oracle partner; Fk s = shuffled decoded partner.
MODELS = {"F0": ("F0", "dec", False),
          "F1d": ("F1", "dec", False), "F1o": ("F1", "orc", False),
          "F2d": ("F2", "dec", False), "F2o": ("F2", "orc", False),
          "F3d": ("F3", "dec", False), "F3o": ("F3", "orc", False),
          "F1s": ("F1", "dec", True), "F2s": ("F2", "dec", True), "F3s": ("F3", "dec", True)}
SPEC_KEYS = ["F0", "F1d", "F1o", "F2d", "F2o", "F3d", "F3o"]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, file=sys.stderr, flush=True)


T0 = time.time()


# ------------------------------------------------------------------------------------------------ geometry
class Geo:
    """Tensor shapes of one transformer layer (Qwen2.5-0.5B: hid 896, hd 64, nh 14, nkv 2, inter 4864)."""

    def __init__(s, hid, hd, nh, nkv, inter, nl):
        assert nh % nkv == 0
        s.hid, s.hd, s.nh, s.nkv, s.inter, s.nl, s.gq = hid, hd, nh, nkv, inter, nl, nh // nkv
        s.shape = dict(q=(nh * hd, hid), k=(nkv * hd, hid), v=(nkv * hd, hid), o=(hid, nh * hd),
                       gate=(inter, hid), up=(inter, hid), down=(hid, inter))
        for sh in s.shape.values():
            assert sh[1] % ft.GROUP == 0 and sh[1] % BLK == 0
        s.stage_n = {"qkv": hid, "o": nh * hd, "ffn": hid, "down": inter}

    def nw(s, kind):
        return s.shape[kind][0] * s.shape[kind][1]


QWEN = Geo(896, 64, 14, 2, 4864, 24)
SYN = Geo(256, 32, 4, 2, 512, 8)


def kv_map(geo):
    """q row / o column (h, d) -> kv row (h // gq, d)."""
    i = torch.arange(geo.nh * geo.hd)
    return (i // geo.hd // geo.gq) * geo.hd + i % geo.hd


# ------------------------------------------------------------------------------------------------ whitening
def sqrt_root(Hl, dev):
    """S = Hr^{1/2}, Hr = had_gram(Hl), fp64 eigh on the CPU (CPU eigh 4864: 2.2 s measured, 32 threads).
    Cost: temporal ~9 n^3 flops (symmetric eigendecomposition, n = Hl.shape[0] <= 4864); spatial 3 n^2 words."""
    Hr = ft.had_gram(Hl.to(dev, F64))
    Hr = 0.5 * (Hr + Hr.T)
    w, V = torch.linalg.eigh(Hr.cpu())
    w = w.clamp_min(0.0)
    return (V * w.sqrt()) @ V.T


class Ctx:
    """One layer on the compute device, fp64: decoded Wp, true Ws (optional), whitened Pd, Pt and residual E."""

    def __init__(s, geo, Wp, Ws, S, dev):
        # Cost: temporal sum over tensors of O n^2 flops (fwht128 O(n log 128) is lower order); spatial 3 * #weights words.
        s.geo, s.dev = geo, dev
        s.S = {st: v.to(dev, F64) for st, v in S.items()}
        s.Wp = {nm: Wp[nm].to(dev, F64) for nm in KINDS}
        s.Pd = {nm: ft.fwht128(s.Wp[nm]) @ s.S[STAGE_OF[nm]] for nm in KINDS}
        s.Ws = s.Pt = s.E = None
        if Ws is not None:
            s.Ws = {nm: Ws[nm].to(dev, F64) for nm in KINDS}
            s.Pt = {nm: ft.fwht128(s.Ws[nm]) @ s.S[STAGE_OF[nm]] for nm in KINDS}
            s.E = {nm: ft.fwht128(s.Ws[nm] - s.Wp[nm]) @ s.S[STAGE_OF[nm]] for nm in KINDS}


def blocks(M):
    """[O, n] -> [O * n/8, 8]; block (o, b) at row o * (n/8) + b."""
    return M.reshape(M.shape[0] * (M.shape[1] // BLK), BLK)


# ------------------------------------------------------------------------------------------------ routing
def lower_bound_vec(sk, x):
    """Vectorised lowerBound on sorted columns: returns (#keys < x, comparison count) per query.
    sk [m, nb] ascending per column, x [m, nb].
    Cost: temporal ceil(log2(m+1)) key comparisons per query, c(m) = 1 + c(floor(m/2)), c(0) = 0, closes at ceil(log2(m+1));
    m*nb queries -> O(m nb log m) work in total.  Spatial: 4 m nb words (lo, hi, count, gathered key)."""
    m, nb = sk.shape
    lo = torch.zeros(m, nb, dtype=torch.long, device=sk.device)
    hi = torch.full((m, nb), m, dtype=torch.long, device=sk.device)
    cnt = torch.zeros(m, nb, dtype=torch.long, device=sk.device)
    for _ in range(math.ceil(math.log2(m + 1))):
        act = lo < hi
        mid = (lo + hi) // 2
        less = sk.gather(0, mid.clamp(max=m - 1)) < x
        lo = torch.where(act & less, mid + 1, lo)
        hi = torch.where(act & ~less, mid, hi)
        cnt += act.long()
    assert bool((lo == hi).all())
    return lo, cnt


def route_f3(Pd):
    """F3 routing on one tensor. Key kappa = <decoded whitened block, e1>, e1 = top eigenvector of the decoded 8x8 block covariance.
    Per column block b the keys of the O rows are sorted once; each row looks up its position by lowerBound and takes the
    predecessor (successor at position 0) as partner. Returns (partner row [O, nb], comparison counts, bound).
    Cost: sort O nb log O comparisons once per tensor; lookup <= ceil(log2(O+1)) comparisons per block (asserted);
    spatial 5 O nb words."""
    O, n = Pd.shape
    nb = n // BLK
    B = Pd.reshape(O, nb, BLK)
    flat = B.reshape(-1, BLK)
    cov = (flat.T @ flat / flat.shape[0]).cpu()
    e1 = torch.linalg.eigh(cov)[1][:, -1].to(Pd.device, F64)
    key = B @ e1
    order = key.argsort(dim=0, stable=True)
    sk = key.gather(0, order)
    lo, cnt = lower_bound_vec(sk, key)
    bound = math.ceil(math.log2(O + 1))
    assert int(cnt.max()) <= bound, (int(cnt.max()), bound)
    ref = torch.searchsorted(sk.T.contiguous(), key.T.contiguous()).T
    assert bool((ref == lo).all())
    j = torch.where(lo > 0, lo - 1, torch.full_like(lo, min(1, O - 1)))
    return order.gather(0, j), cnt, bound


# ------------------------------------------------------------------------------------------------ features
def f0_feats(Pd):
    """F0: own decoded block and its two column neighbours (b-1, b+1), zero padded: [N, 24]."""
    O, n = Pd.shape
    B = Pd.reshape(O, n // BLK, BLK)
    L = torch.zeros_like(B)
    R = torch.zeros_like(B)
    L[:, 1:] = B[:, :-1]
    R[:, :-1] = B[:, 1:]
    return torch.cat([B, L, R], -1).reshape(-1, 3 * BLK)


def f1_partner_mats(geo, kind, P):
    """F1 junction-linear contexts, each [O, n] in the target's whitened coordinates (same stage => same S):
    q row (h,d) <- k row (h//gq, d); k <- v same row and mean of its gq q rows; v <- k same row; gate <-> up same row."""
    if kind == "q":
        return [P["k"].index_select(0, kv_map(geo).to(P["k"].device))]
    if kind == "k":
        n = P["q"].shape[1]
        return [P["v"], P["q"].reshape(geo.nkv, geo.gq, geo.hd, n).mean(1).reshape(geo.nkv * geo.hd, n)]
    if kind == "v":
        return [P["k"]]
    if kind == "gate":
        return [P["up"]]
    if kind == "up":
        return [P["gate"]]
    return []


def _std(v):
    v = v - v.mean()
    return v / v.pow(2).mean().sqrt().clamp_min(1e-30)


def _summ_rows(M):
    """Two scalar summaries of each row: signed sum / sqrt(len) (linear) and L2 norm."""
    return [M.sum(1) / math.sqrt(M.shape[1]), M.norm(dim=1)]


def f2_fields(geo, kind, W):
    """F2 product-junction scale fields from partner raw weights W (decoded or true): (axis, [vectors]) or None.
    o col (h,d) <- v row (h//gq, d); v row <- mean of its gq o columns; down col i <- up row i, gate row i;
    up/gate row i <- down col i.  Each field is centred and scaled to unit RMS."""
    if kind == "o":
        vm = kv_map(geo).to(W["v"].device)
        return "col", [_std(f[vm]) for f in _summ_rows(W["v"])]
    if kind == "v":
        Wo = W["o"]
        Mv = Wo.reshape(Wo.shape[0], geo.nkv, geo.gq, geo.hd).mean(2).permute(1, 2, 0).reshape(geo.nkv * geo.hd, Wo.shape[0])
        return "row", [_std(f) for f in _summ_rows(Mv)]
    if kind == "down":
        return "col", [_std(f) for f in _summ_rows(W["up"]) + _summ_rows(W["gate"])]
    if kind in ("up", "gate"):
        Wd = W["down"]
        return "row", [_std(Wd.sum(0) / math.sqrt(Wd.shape[0])), _std(Wd.norm(dim=0))]
    return None


ROW_AXIS = ("q", "k", "v", "gate", "up")


def shuffle_partners(D, gen):
    """Control: permute every tensor along its junction axis (rows for q,k,v,gate,up; columns for o,down)."""
    out = {}
    for nm, M in D.items():
        ax = 0 if nm in ROW_AXIS else 1
        out[nm] = M.index_select(ax, torch.randperm(M.shape[ax], generator=gen).to(M.device))
    return out


def features(key, geo, kind, ctx, seed=0):
    """Design for model `key` on tensor `kind`: dict(X [N,q], Z [N,8,K] | None, deg, route).
    deg=True when the family has no context for this kind (the model equals F0).
    Cost: temporal O(N q) words moved plus F2 O(K O n^2) flops for the K fields (fwht128 @ S); spatial N (q + 8K) words,
    N = O n / 8 blocks."""
    fam, arm, shuf = MODELS[key]
    X0 = f0_feats(ctx.Pd[kind])
    if fam == "F0":
        return dict(X=X0, Z=None, deg=False, route=None)
    P = ctx.Pd if arm == "dec" else ctx.Pt
    W = ctx.Wp if arm == "dec" else ctx.Ws
    if shuf:
        gen = torch.Generator().manual_seed(seed)
        P, W = shuffle_partners(P, gen), shuffle_partners(W, gen)
    if fam == "F1":
        mats = f1_partner_mats(geo, kind, P)
        if not mats:
            return dict(X=X0, Z=None, deg=True, route=None)
        return dict(X=torch.cat([X0] + [blocks(m) for m in mats], 1), Z=None, deg=False, route=None)
    if fam == "F2":
        fz = f2_fields(geo, kind, W)
        if fz is None:
            return dict(X=X0, Z=None, deg=True, route=None)
        axis, fields = fz
        Wt, S = ctx.Wp[kind], ctx.S[STAGE_OF[kind]]
        Zs = [blocks(ft.fwht128(Wt * (f[None, :] if axis == "col" else f[:, None])) @ S) for f in fields]
        return dict(X=X0, Z=torch.stack(Zs, 2), deg=False, route=None)
    assert fam == "F3"
    partner, cnt, bound = route_f3(ctx.Pd[kind])
    if shuf:
        gen = torch.Generator().manual_seed(seed + 1)
        O, nb = partner.shape
        partner = torch.rand(O, nb, generator=gen).argsort(0).to(partner.device)
    src = ctx.Pd[kind] if arm == "dec" else ctx.E[kind]
    O, n = src.shape
    Bs = src.reshape(O, n // BLK, BLK)
    ar = torch.arange(n // BLK, device=src.device)
    cx = Bs[partner, ar[None, :]].reshape(-1, BLK)
    return dict(X=torch.cat([X0, cx], 1), Z=None, deg=False, route=(int(cnt.max()), float(cnt.double().mean()), bound, cnt.numel()))


# ------------------------------------------------------------------------------------------------ ridge, cross-fit
def stats(X, Z, Y):
    """Sufficient statistics of the design (A: q x 8 block map, d: K shared scalar coefficients).
    Cost: temporal 2 N (q^2 + 8q + 8qK + 8K^2) flops, N blocks; spatial q^2 + 8q + 8qK + K^2 words kept."""
    N, q = X.shape
    st = dict(q=q, K=0, Gxx=(X.T @ X).cpu().numpy(), Xy=(X.T @ Y).cpu().numpy(), yy=float((Y * Y).sum()), B=None, D=None, Zy=None)
    if Z is not None:
        K = Z.shape[2]
        Zf = Z.reshape(N * BLK, K)
        st.update(K=K, B=(X.T @ Z.reshape(N, BLK * K)).reshape(q, BLK, K).cpu().numpy(),
                  D=(Zf.T @ Zf).cpu().numpy(), Zy=(Zf.T @ Y.reshape(N * BLK)).cpu().numpy())
    return st


def assemble(st):
    """Normal equations G theta = c, theta = (A[:, i] for i in 0..7, d); entry i*q + j of theta is A[j, i].
    Cost: temporal and spatial O(P^2) with P = 8q + K <= 400."""
    q, K = st["q"], st["K"]
    P = BLK * q + K
    G, c = np.zeros((P, P)), np.zeros(P)
    for i in range(BLK):
        G[i * q:(i + 1) * q, i * q:(i + 1) * q] = st["Gxx"]
        c[i * q:(i + 1) * q] = st["Xy"][:, i]
    if K:
        Bm = st["B"].transpose(1, 0, 2).reshape(BLK * q, K)
        G[:BLK * q, BLK * q:], G[BLK * q:, :BLK * q], G[BLK * q:, BLK * q:] = Bm, Bm.T, st["D"]
        c[BLK * q:] = st["Zy"]
    return G, c


def _solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def fit_lolo(stl):
    """Leave-one-layer-out ridge, lambda = 1e-3 tr(X^T X) / p on the training Gram.  sse_l = yy - 2 th.c + th.G.th on the held-out layer.
    Cost: temporal L P^3 flops (L layers, one solve each); spatial L P^2 words."""
    GL, cL = zip(*[assemble(st) for st in stl])
    Gt, ct, P = sum(GL), sum(cL), len(cL[0])
    thetas, sse = [], []
    for l, st in enumerate(stl):
        Gtr, ctr = Gt - GL[l], ct - cL[l]
        th = _solve(Gtr + (RIDGE_REL * np.trace(Gtr) / P) * np.eye(P), ctr)
        thetas.append(th)
        sse.append(st["yy"] - 2 * th @ cL[l] + th @ GL[l] @ th)
    full = _solve(Gt + (RIDGE_REL * np.trace(Gt) / P) * np.eye(P), ct)
    sse_in = [st["yy"] - 2 * full @ cL[l] + full @ GL[l] @ full for l, st in enumerate(stl)]
    return dict(P=P, thetas=thetas, sse=sse, sse_in=sse_in, yy=[st["yy"] for st in stl])


def predict(theta, X, Z):
    q = X.shape[1]
    th = torch.as_tensor(theta, dtype=F64, device=X.device)
    out = X @ th[:BLK * q].reshape(BLK, q).T
    if Z is not None:
        out = out + Z @ th[BLK * q:]
    return out


# ------------------------------------------------------------------------------------------------ rate
def waterfill(lam, w, D):
    """Reverse water-filling over independent Gaussian components of variance lam (multiplicity w): returns total bits
    sum w * 0.5 log2^+(lam / theta) with sum w * min(theta, lam) = D (0 bits when D >= sum w lam).
    Cost: temporal 80 m flops (bisection on theta, m components); spatial m words."""
    lam = np.maximum(np.asarray(lam, dtype=np.float64), 0.0)
    w = np.broadcast_to(np.asarray(w, dtype=np.float64), lam.shape)
    if D >= float((w * lam).sum()) or lam.max() <= 0:
        return 0.0
    lo, hi = 0.0, float(lam.max())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float((w * np.minimum(mid, lam)).sum()) < D:
            lo = mid
        else:
            hi = mid
    th = 0.5 * (lo + hi)
    m = lam > th
    return float((w[m] * 0.5 * np.log2(lam[m] / th)).sum())


def spec_eigs(M):
    """Eigenvalues of M^T M / O (the nonzero ones, via the smaller Gram).
    Cost: temporal O n min(O,n) + 9 min(O,n)^3 flops; spatial min(O,n)^2 words."""
    O, n = M.shape
    G = (M.T @ M if n <= O else M @ M.T) / O
    return torch.linalg.eigvalsh(G.cpu()).clamp_min(0).numpy()


def rate_table(geo, specs, Jt, coef_bits=None):
    """specs[(kind, l)] -> eigenvalues; Jt[(kind, l)] = ||E||^2 of the original residual (target D = (1 - f) J).
    Per-tensor water-filling aggregated by weights ('tensor') and one common theta over all components ('global').
    Cost: temporal 80 f-count * total components flops; spatial total components words."""
    out = {}
    for f in FS:
        bits = {k: 0.0 for k in KINDS}
        wts = {k: 0 for k in KINDS}
        lam_all, w_all = [], []
        for (kind, l), lam in specs.items():
            O, n = geo.shape[kind]
            bits[kind] += waterfill(lam, O, (1 - f) * Jt[(kind, l)])
            wts[kind] += O * n
            lam_all.append(lam)
            w_all.append(np.full(len(lam), float(O)))
        coef = coef_bits or {}
        r = {k: bits[k] / wts[k] + coef.get(k, 0.0) / wts[k] for k in KINDS if wts[k]}
        tot_w = sum(wts.values())
        r["total"] = sum(bits.values()) / tot_w + sum(coef.get(k, 0.0) for k in KINDS) / tot_w
        Dg = (1 - f) * sum(Jt.values())
        r["total_global_theta"] = waterfill(np.concatenate(lam_all), np.concatenate(w_all), Dg) / tot_w + sum(coef.get(k, 0.0) for k in KINDS) / tot_w
        out[repr(f)] = r
    return out


def d_rate(r2):
    return 0.5 * math.log2(1.0 / (1.0 - r2)) if r2 < 1 else float("inf")


# ------------------------------------------------------------------------------------------------ analysis
def analyze(source, geo, dev, spectra=True, oracle=True, shuffles=True, tag=""):
    """Two passes over the layers. Pass 1: S, E, J gate, per-model sufficient statistics, E spectra.  Pass 2 (spectra): fold-specific
    Ehat, residual spectra, SSE consistency check.  `source` yields (l, Ws, Wp, Hl) with Ws, Wp dicts of CPU tensors by kind and
    Hl a dict of CPU Hessians by stage.
    Cost: temporal per layer sum over kinds and models of O(N q^2) + the eigendecompositions in sqrt_root / spec_eigs; spatial
    one layer on the device plus all layers' Ws, Wp (fp32) and S (fp32) on the CPU for pass 2."""
    keys = [k for k in MODELS if (oracle or not k.endswith("o")) and (shuffles or not k.endswith("s"))]
    st = {k: {kd: [] for kd in KINDS} for k in keys}
    deg = {kd: {} for kd in KINDS}
    Jt, specE, route = {}, {}, {kd: [0, 0.0, 0, 0] for kd in KINDS}
    cache = []
    gate_max = 0.0
    tm = dict(pass1=0.0, pass2=0.0)
    t1 = time.time()
    for l, Ws, Wp, Hl in source:
        S = {stg: sqrt_root(Hl[stg], dev) for stg in geo.stage_n}
        ctx = Ctx(geo, Wp, Ws, S, dev)
        cache.append((l, {k: v.float() for k, v in Ws.items()}, {k: v.float() for k, v in Wp.items()}, {k: v.float() for k, v in S.items()}))
        for kind in KINDS:
            stg = STAGE_OF[kind]
            Jref = ft.J(ctx.Ws[kind], ctx.Wp[kind], Hl[stg].to(dev, F64))
            JE = float((ctx.E[kind] ** 2).sum())
            gate_max = max(gate_max, abs(JE - Jref) / max(abs(Jref), 1e-300))
            Jt[(kind, l)] = JE
            Y = blocks(ctx.E[kind])
            if spectra:
                specE[(kind, l)] = spec_eigs(ctx.E[kind])
            for key in keys:
                ft_ = features(key, geo, kind, ctx, seed=7919 * (l + 1) + 31 * KINDS.index(kind))
                if ft_["deg"]:
                    deg[kind][key] = True
                    st[key][kind].append(st["F0"][kind][-1])
                else:
                    st[key][kind].append(stats(ft_["X"], ft_["Z"], Y))
                    if key == "F3d" and ft_["route"] is not None:
                        r = route[kind]
                        r[0], r[1], r[2], r[3] = max(r[0], ft_["route"][0]), r[1] + ft_["route"][1] * ft_["route"][3], ft_["route"][2], r[3] + ft_["route"][3]
                del ft_
            del Y
        del ctx
        log(f"{tag}pass1 layer {l} done (gate max rel {gate_max:.2e})")
    tm["pass1"] = time.time() - t1
    if gate_max > GATE_REL:
        raise SystemExit(f"J gate failed: max rel |‖E‖² - J| = {gate_max:.3e} > {GATE_REL}")
    fits = {k: {kd: fit_lolo(st[k][kd]) for kd in KINDS} for k in keys}
    yyk = {kd: sum(fits["F0"][kd]["yy"]) for kd in KINDS}
    yy_tot = sum(yyk.values())
    res = dict(gate_max_rel=gate_max, J_sum=yy_tot, J_per_weight=yy_tot / sum(geo.nw(kd) * len(cache) for kd in KINDS),
               J_share={kd: yyk[kd] / yy_tot for kd in KINDS}, models=keys)
    r2, r2in, incr, dr = {}, {}, {}, {}
    for k in keys:
        r2[k] = {kd: 1 - sum(fits[k][kd]["sse"]) / yyk[kd] for kd in KINDS}
        r2[k]["total"] = 1 - sum(sum(fits[k][kd]["sse"]) for kd in KINDS) / yy_tot
        r2in[k] = {kd: 1 - sum(fits[k][kd]["sse_in"]) / yyk[kd] for kd in KINDS}
        r2in[k]["total"] = 1 - sum(sum(fits[k][kd]["sse_in"]) for kd in KINDS) / yy_tot
        incr[k] = {kd: r2[k][kd] - r2["F0"][kd] for kd in list(KINDS) + ["total"]}
        dr[k] = {kd: d_rate(max(min(incr[k][kd], 0.999999), 0.0)) for kd in incr[k]}
    res.update(r2_oof=r2, r2_in_sample=r2in, incr_over_F0=incr, dR_of_incr_bpw=dr,
               coef_count={k: {kd: fits[k][kd]["P"] for kd in KINDS} for k in keys},
               degenerate={kd: sorted(deg[kd]) for kd in KINDS},
               f3_routing={kd: dict(max_comparisons=route[kd][0], mean_comparisons=(route[kd][1] / route[kd][3] if route[kd][3] else None),
                                    bound=route[kd][2], lookups=route[kd][3]) for kd in KINDS})
    if shuffles:
        sh = {}
        for fam in ("F1", "F2", "F3"):
            sh[fam] = dict(total=r2[fam + "s"]["total"] - r2["F0"]["total"], per_kind={kd: r2[fam + "s"][kd] - r2["F0"][kd] for kd in KINDS})
        res["shuffle_delta_r2"] = sh
        res["shuffle_pass_total"] = all(abs(v["total"]) <= 0.005 for v in sh.values())
    if spectra:
        t2 = time.time()
        coef = {k: {kd: 16.0 * fits[k][kd]["P"] for kd in KINDS} for k in SPEC_KEYS if k in keys}
        specB = {k: {} for k in SPEC_KEYS if k in keys}
        sse_dev = 0.0
        for pos, (l, Ws, Wp, S) in enumerate(cache):
            ctx = Ctx(geo, Wp, Ws, S, dev)
            for kind in KINDS:
                Y = blocks(ctx.E[kind])
                O, n = geo.shape[kind]
                for key in specB:
                    if key != "F0" and key in deg[kind]:
                        specB[key][(kind, l)] = specB["F0"][(kind, l)]
                        continue
                    ft_ = features(key, geo, kind, ctx)
                    resid = Y - predict(fits[key][kind]["thetas"][pos], ft_["X"], ft_["Z"])
                    sse_dev = max(sse_dev, abs(float((resid ** 2).sum()) - fits[key][kind]["sse"][pos]) / fits[key][kind]["yy"][pos])
                    specB[key][(kind, l)] = spec_eigs(resid.reshape(O, n))
                    del ft_, resid
                del Y
            del ctx
            log(f"{tag}pass2 layer {l} done")
        res["sse_consistency_max_rel"] = sse_dev
        res["R_a"] = rate_table(geo, specE, Jt)
        res["R_b"] = {k: rate_table(geo, specB[k], Jt, coef_bits={kd: coef[k][kd] for kd in KINDS}) for k in specB}
        res["R_b_note"] = "R_b = rate of E - Ehat (out-of-fold) at D = (1-f)J plus 16 P / #weights coefficient cost; o-arms (F*o) exclude the oracle's own rate"
        tm["pass2"] = time.time() - t2
    res["timings_s"] = tm
    return res


# ------------------------------------------------------------------------------------------------ real source
def real_source(cap, dump, geo):
    meta = json.load(open(os.path.join(cap, "meta.json")))
    layers = torch.load(dump, map_location="cpu")["layers"]
    assert meta["layers"] == geo.nl == len(layers), (meta["layers"], geo.nl, len(layers))
    for l in range(geo.nl):
        Ws = {nm: torch.load(os.path.join(cap, f"W_l{l}_{nm}.pt"), map_location="cpu") for nm in KINDS}
        Hl = {stg: torch.load(os.path.join(cap, f"H_l{l}_{stg}.pt"), map_location="cpu") for stg in geo.stage_n}
        Wp = {nm: layers[l][nm].float().cpu() for nm in KINDS}
        for nm in KINDS:
            assert Ws[nm].shape == Wp[nm].shape == geo.shape[nm], (l, nm, Ws[nm].shape, Wp[nm].shape)
        yield l, Ws, Wp, Hl


# ------------------------------------------------------------------------------------------------ synthetic source
def _rand_spd(n, g):
    Zm = torch.randn(2 * n, n, generator=g, dtype=F64) * torch.exp(0.4 * torch.randn(n, generator=g, dtype=F64))
    H = Zm.T @ Zm / (2 * n)
    return H + 0.05 * H.diagonal().mean() * torch.eye(n, dtype=F64)


def synth_source(geo, fam, rho, seed):
    """Planted structure: for every kind E = sqrt(rho) sd z(Xb theta*) + sqrt(1 - rho) sd z(noise) with the family-`fam` design
    (F0 where the family has no context for the kind), theta* shared across layers; Ws = Wp + fwht128(E S^-1).
    Population R^2 of the planted model is rho.
    Cost: temporal per layer sum over kinds of the feature build plus O n^2 flops for S^-1; spatial one layer."""
    g = torch.Generator().manual_seed(seed)
    th_star, sd = {}, {kd: float(0.5 + 1.5 * torch.rand(1, generator=g)) for kd in KINDS}
    key = fam + "d"
    for l in range(geo.nl):
        Hl = {stg: _rand_spd(n, g) for stg, n in geo.stage_n.items()}
        S = {stg: sqrt_root(Hl[stg], "cpu") for stg in geo.stage_n}
        Wp = {nm: torch.randn(*geo.shape[nm], generator=g, dtype=F64) * 0.1 * torch.exp(0.5 * torch.randn(geo.shape[nm][0], 1, generator=g, dtype=F64))
              for nm in KINDS}
        ctx = Ctx(geo, Wp, None, S, "cpu")
        Ws = {}
        for kind in KINDS:
            ft_ = features(key, geo, kind, ctx)
            if ft_["deg"]:
                ft_ = features("F0", geo, kind, ctx)
            X, Z = ft_["X"], ft_["Z"]
            q, K = X.shape[1], (0 if Z is None else Z.shape[2])
            if kind not in th_star:
                th_star[kind] = torch.randn(BLK * q + K, generator=g, dtype=F64)
            sig = predict(th_star[kind], X, Z)
            noise = torch.randn(sig.shape, generator=g, dtype=F64)
            z = lambda a: a / a.pow(2).mean().sqrt()
            Y = sd[kind] * (math.sqrt(rho) * z(sig) + math.sqrt(1 - rho) * z(noise))
            O, n = geo.shape[kind]
            Eg = Y.reshape(O, n)
            Rr = torch.linalg.solve(S[STAGE_OF[kind]], Eg.T).T
            Ws[kind] = Wp[kind] + ft.fwht128(Rr)
        yield l, Ws, Wp, Hl


# ------------------------------------------------------------------------------------------------ controls
def run_controls():
    """In-process controls (CPU): water-filling white bound, GQA-map references, bisect vs stdlib bisect, planted R^2 in {0, 0.3}
    for F1/F2/F3 (recovered within 0.02 per kind and in total), shuffled-partner |dR^2| <= 0.005, J gate (inside analyze).
    Cost: temporal 6 analyses on the synthetic geometry (8 layers, 256-dim); spatial < 100 MB."""
    out, ok = {}, True
    lam = np.full(100, 1.2956)
    wb = waterfill(lam, 1.0, (1 - F_STAR) * lam.sum()) / 100
    out["waterfill_white"] = dict(value=wb, expected=0.5 * math.log2(1.2956 / 0.0197), pass_=abs(wb - 3.0196) < 1e-3)
    # GQA maps against explicit loops
    g = torch.Generator().manual_seed(5)
    geo = SYN
    P = {nm: torch.randn(*geo.shape[nm], generator=g, dtype=F64) for nm in KINDS}
    qm = f1_partner_mats(geo, "q", P)[0]
    km = f1_partner_mats(geo, "k", P)[1]
    okm = True
    for o in range(geo.nh * geo.hd):
        h, d = divmod(o, geo.hd)
        okm &= bool(torch.equal(qm[o], P["k"][(h // geo.gq) * geo.hd + d]))
    for gi in range(geo.nkv):
        for d in range(geo.hd):
            ref = sum(P["q"][(gi * geo.gq + j) * geo.hd + d] for j in range(geo.gq)) / geo.gq
            okm &= bool(torch.allclose(km[gi * geo.hd + d], ref))
    fz = f2_fields(geo, "v", P)
    Mv = torch.stack([sum(P["o"][:, (gi * geo.gq + j) * geo.hd + d] for j in range(geo.gq)) / geo.gq for gi in range(geo.nkv) for d in range(geo.hd)])
    okm &= bool(torch.allclose(fz[1][0], _std(Mv.sum(1) / math.sqrt(Mv.shape[1])))) and bool(torch.allclose(fz[1][1], _std(Mv.norm(dim=1))))
    out["gqa_maps"] = dict(pass_=okm)
    # bisect vs stdlib
    sk = torch.sort(torch.randn(37, 5, generator=g, dtype=F64), dim=0)[0]
    x = torch.randn(37, 5, generator=g, dtype=F64)
    lo, cnt = lower_bound_vec(sk, x)
    okb = int(cnt.max()) <= math.ceil(math.log2(38))
    for c in range(5):
        col = sk[:, c].tolist()
        for r in range(37):
            okb &= bisect.bisect_left(col, float(x[r, c])) == int(lo[r, c])
    out["bisect"] = dict(max_comparisons=int(cnt.max()), bound=math.ceil(math.log2(38)), pass_=bool(okb))
    ok &= out["waterfill_white"]["pass_"] and okm and bool(okb)
    out["planted"] = {}
    for fam in ("F1", "F2", "F3"):
        for rho in (0.0, 0.3):
            first = (fam == "F1" and rho == 0.3)
            r = analyze(synth_source(SYN, fam, rho, seed=11 + int(10 * rho)), SYN, "cpu", spectra=first, tag=f"ctl {fam} rho={rho} ")
            r2 = r["r2_oof"][fam + "d"]
            dev = {kd: r2[kd] - rho for kd in KINDS + ["total"]}
            shuf = abs(r["r2_oof"][fam + "s"]["total"] - r["r2_oof"]["F0"]["total"])
            case_ok = max(abs(v) for v in dev.values()) <= 0.02 and shuf <= 0.005 and r["gate_max_rel"] <= GATE_REL
            if first:
                case_ok &= r["sse_consistency_max_rel"] < 1e-4  # pass 2 rebuilds from fp32 caches of Ws, Wp, S
            out["planted"][f"{fam}_rho{rho}"] = dict(r2=r2, dev_from_planted=dev, shuffled_dR2_total=shuf, gate_max_rel=r["gate_max_rel"],
                                                     sse_consistency=r.get("sse_consistency_max_rel"), pass_=bool(case_ok))
            ok &= bool(case_ok)
    out["pass_"] = bool(ok)
    return out


# ------------------------------------------------------------------------------------------------ commands
def cmd_capture(a):
    """Wrap ft.fold_fit; on the first call per (layer, kind) from run_variant store Ws_nm and the stage's Hl_m, then run the normal path.
    Cost: one -nb run (about 130 s per seed) plus 4 GB of writes."""
    vs = [v for v in a.variants.split(",") if v]
    assert len(vs) == 1 and a.seeds.strip() == "0", "capture takes exactly one variant and seed 0"
    assert a.dump and a.cap and a.out
    os.makedirs(a.cap, exist_ok=True)
    seen, held = set(), {}
    orig = ft.fold_fit

    def wrapped(Ws, H, *args, **kw):
        fr = sys._getframe(1)
        if fr.f_code.co_name == "run_variant":
            loc = fr.f_locals
            l, nm = loc.get("l"), loc.get("nm")
            if l is not None and nm in STAGE_OF and loc.get("Ws_nm") is Ws and (l, nm) not in seen:
                seen.add((l, nm))
                Hl, stg = loc["Hl_m"], STAGE_OF[nm]
                torch.save(Ws.detach().cpu().contiguous(), os.path.join(a.cap, f"W_l{l}_{nm}.pt"))
                for kk in [k for k in held if k[0] != l]:
                    del held[kk]
                if (l, stg) not in held:
                    torch.save(Hl.detach().cpu().contiguous(), os.path.join(a.cap, f"H_l{l}_{stg}.pt"))
                    held[(l, stg)] = Hl
                else:
                    assert held[(l, stg)] is Hl or torch.equal(held[(l, stg)], Hl), f"Hl_m differs within stage {stg} layer {l}"
            del fr, loc
        return orig(Ws, H, *args, **kw)

    ft.fold_fit = wrapped
    ft_out = a.out + ".ft.json"
    t0 = time.time()
    sys.argv = [ft.__file__, "--out", ft_out, "--variants", vs[0], "--seeds", "0", "--dump", a.dump]
    ft.main()
    ft.fold_fit = orig
    res = json.load(open(ft_out))
    v = res["variants"][vs[0]]
    nl = len(torch.load(a.dump, map_location="cpu")["layers"])
    json.dump(dict(layers=nl, kinds=KINDS, stage_of=STAGE_OF, variant=vs[0], ft_sha256=hashlib.sha256(open(ft.__file__, "rb").read()).hexdigest()),
              open(os.path.join(a.cap, "meta.json"), "w"), indent=1)
    out = dict(variant=vs[0], kl_heldout=v["kl_heldout"], top1_heldout=v["top1_heldout"], kl_insample=v.get("kl_insample"), bits=v["bits"],
               params=v["params"], bpw_linears=v.get("bpw_linears"), captured=len(seen), expected_captured=nl * len(KINDS),
               expected=dict(kl_heldout=1.2777, kl_tol=0.0005, bits=421541376),
               kl_ok=abs(v["kl_heldout"] - 1.2777) <= 0.0005, bits_ok=int(round(v["bits"])) == 421541376,
               capture_complete=len(seen) == nl * len(KINDS), wall_s=time.time() - t0, ft_json=ft_out, dump=a.dump, cap=a.cap)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    if not (out["capture_complete"]):
        raise SystemExit("capture incomplete")


def summarize(res):
    L = []
    L.append(f"J_sum {res['J_sum']:.6g}  J/weight {res['J_per_weight']:.6g}  gate max rel {res['gate_max_rel']:.2e}")
    L.append("J_share " + " ".join(f"{k}={v:.3f}" for k, v in res["J_share"].items()))
    for k in res["models"]:
        L.append(f"R2 oof {k:4s} " + " ".join(f"{kd}={res['r2_oof'][k][kd]:+.4f}" for kd in KINDS + ["total"]))
    for k in res["models"]:
        if k != "F0":
            L.append(f"incr   {k:4s} " + " ".join(f"{kd}={res['incr_over_F0'][k][kd]:+.4f}" for kd in KINDS + ["total"]))
    if "R_a" in res:
        for f, r in res["R_a"].items():
            L.append(f"R_a(f={f}) total {r['total']:.4f} global-theta {r['total_global_theta']:.4f} " + " ".join(f"{kd}={r[kd]:.3f}" for kd in KINDS))
        for k, tab in res["R_b"].items():
            for f, r in tab.items():
                L.append(f"R_b {k:4s} f={f} total {r['total']:.4f} global-theta {r['total_global_theta']:.4f}")
    return "\n".join(L)


def cmd_probe(a):
    out = dict(script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), mode="synthetic" if a.synthetic else "real",
               fold_transfer_sha256=hashlib.sha256(open(ft.__file__, "rb").read()).hexdigest(), f_star=F_STAR,
               white_bound_bpw=0.5 * math.log2(1.2956 / 0.0197))
    t0 = time.time()
    out["controls"] = run_controls()
    out["controls_wall_s"] = time.time() - t0
    log("controls", "PASS" if out["controls"]["pass_"] else "FAIL")
    print(json.dumps({k: (v if k != "planted" else {c: dict(pass_=x["pass_"], dev=x["dev_from_planted"], shuf=x["shuffled_dR2_total"]) for c, x in v.items()})
                      for k, v in out["controls"].items()}, indent=1, default=float))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
    if not out["controls"]["pass_"]:
        raise SystemExit("synthetic controls failed; aborting")
    if a.synthetic:
        return
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = analyze(real_source(a.cap, a.dump, QWEN), QWEN, dev)
    out["device"] = dev
    out["result"] = res
    dec = ["F1d", "F2d", "F3d"]
    inc_tot = {k: res["incr_over_F0"][k]["total"] for k in dec}
    out["verdict"] = dict(
        incr_decoded_total=inc_tot,
        route_b_open=bool(max(inc_tot.values()) >= 0.13 and res["shuffle_pass_total"]),
        route_b_closed_linear=bool(all(res["incr_over_F0"][k][kd] <= 0.02 for k in dec for kd in KINDS)),
        falsifier_hit=[(k, kd) for k in dec for kd in KINDS if res["J_share"][kd] >= 0.05 and res["incr_over_F0"][k][kd] >= 0.05],
        R_a_fstar_total=res["R_a"][repr(F_STAR)]["total"], route_a_closed=bool(res["R_a"][repr(F_STAR)]["total"] >= 2.0),
        controls_pass=out["controls"]["pass_"] and res["shuffle_pass_total"] and res["gate_max_rel"] <= GATE_REL)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(summarize(res))
    print(json.dumps(out["verdict"], indent=1, default=float))
    log("probe done")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("--variants", required=True); c.add_argument("--seeds", default="0"); c.add_argument("--dump", required=True)
    c.add_argument("--cap", required=True); c.add_argument("--out", required=True)
    p = sp.add_parser("probe")
    p.add_argument("--cap"); p.add_argument("--dump"); p.add_argument("--out"); p.add_argument("--synthetic", action="store_true")
    p.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.cmd == "capture":
        cmd_capture(a)
    else:
        if not a.synthetic:
            assert a.cap and a.dump and a.out, "probe needs --cap --dump --out unless --synthetic"
        cmd_probe(a)


if __name__ == "__main__":
    main()
