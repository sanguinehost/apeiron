"""Variance-model ladder for the K-FAC water-filling floor on Qwen2.5-0.5B (ONEBIT-VARIANCE-LADDER).

Per tensor (168 linear tensors, 24 layers x 7 kinds) the K-FAC eigen-coefficients C = U^T W V are formed from the
f16 GGUF weights W, with G and H from `converse.capture` (100 C4 chunks, 4 sampled targets). Each variance model
gives lam_ij = max(EPS, 1/2 g_i h_j sigma2_ij); global reverse water-filling is KL(theta) = sum min(lam, theta),
coefficient rate = sum max(0, 1/2 log2(lam/theta)), all-in rate adds the side information of the model.

Models: sep (converse.sep_sigma2, control), block32 / block128 (column groups in h-descending order, mean C^2),
nmf2 / nmf4 / nmf8 (multiplicative-update NMF of C^2), oracle (sigma2 = C^2, control). Uniform floor: every tensor
at the same all-in rate with separable sigma2. Precision is fp32 on the tensors throughout.

Usage (repo root): python tools/onebit/variance_ladder.py OUT.json
Factors are cached at FACT and reloaded when present (torch.save of C^2, g, h and the Q4 KL per tensor).
"""
import json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from converse import capture, load_model, sep_sigma2, EPS  # noqa: E402
from kfac_floor import read, module, KINDS  # noqa: E402
from fisher_capture import gguf  # noqa: E402

HF = "Qwen/Qwen2.5-0.5B"
GGUF16 = "/home/socol/models/gguf/qwen05b-f16.gguf"
GGUFQ4 = "/home/socol/oc-tmp/seal/qwen05b-Q4_K_M-imat.gguf"
FACT = "/home/socol/oc-tmp/vladder/factors.pt"
N_CHUNKS, K_SAMPLES, GROUP, N_LAYERS = 100, 4, 8, 24
NMF_ITERS = 300
RATES = (1.18, 1.133)
MODELS = ("sep", "block32", "block128", "nmf2", "nmf4", "nmf8", "oracle")
BISECT = 100


def log(*a):
    print(*a, flush=True)


def capture_factors():
    """Per tensor: g, h (eigenvalues, clamped), C2 = (U^T W V)^2 (fp32), Q4 KL = 1/2 tr(D^T G D H), shape."""
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    model, ids = load_model(HF, torch.float32)
    r16, rq4 = gguf.GGUFReader(GGUF16), gguf.GGUFReader(GGUFQ4)
    out, max_rel = [], 0.0
    for s in range(0, N_LAYERS, GROUP):
        grp = list(range(s, min(N_LAYERS, s + GROUP)))
        G, H = capture(model, ids, N_CHUNKS, K_SAMPLES, grp)
        for l in grp:
            for name, path, key in KINDS:
                gname = f"blk.{l}.{name}.weight"
                w = read(r16, gname).cuda()
                hfw = module(model, l, path).weight.detach().float()
                max_rel = max(max_rel, float((w - hfw).norm() / hfw.norm()))
                q4 = read(rq4, gname).cuda() - w
                Gm, Hm = G.pop((l, name)), H[(l, key)]
                kq4 = float(0.5 * ((Gm @ q4) * (q4 @ Hm)).sum())
                g, U = torch.linalg.eigh(Gm)
                h, V = torch.linalg.eigh(Hm)
                g, h = g.clamp(min=EPS), h.clamp(min=EPS)
                C = U.t() @ w @ V
                out.append({"name": gname, "g": g, "h": h, "C2": C * C, "kq4": kq4, "shape": list(w.shape)})
                del Gm, U, V, C, q4, w
            H.pop((l, key), None)
        del G, H
        torch.cuda.empty_cache()
        log("capture group", grp[0], "t", round(time.time() - t0, 1))
    del model
    torch.cuda.empty_cache()
    torch.save({"tensors": out, "max_rel_gguf_vs_hf": max_rel, "capture_s": time.time() - t0}, FACT)
    return out, max_rel, time.time() - t0


def nmf(X, k, seed):
    """Rank-k nonnegative factorisation X ~ W H by Lee-Seung multiplicative updates (Frobenius), fp32."""
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed)
    m, n = X.shape
    s = (X.mean() / k).sqrt()
    W = torch.rand(m, k, generator=gen, device=X.device) * s + EPS
    H = torch.rand(k, n, generator=gen, device=X.device) * s + EPS
    for _ in range(NMF_ITERS):
        H = H * (W.t() @ X) / ((W.t() @ W) @ H + 1e-20)
        W = W * (X @ H.t()) / (W @ (H @ H.t()) + 1e-20)
    return (W @ H).clamp(min=EPS)


def sigma2(kind, t, idx):
    C2, h = t["C2"], t["h"]
    if kind == "sep":
        return sep_sigma2(C2)
    if kind == "oracle":
        return C2.clamp(min=EPS)
    if kind.startswith("block"):
        gs = int(kind[5:])
        perm = torch.argsort(h, descending=True)
        Cs = C2[:, perm]
        m, n = Cs.shape
        assert n % gs == 0, (n, gs)
        gm = Cs.reshape(m, n // gs, gs).mean(-1, keepdim=True).expand(m, n // gs, gs).reshape(m, n)
        s = torch.empty_like(C2)
        s[:, perm] = gm.clamp(min=EPS)
        return s
    if kind.startswith("nmf"):
        return nmf(C2, int(kind[3:]), seed=idx)
    raise KeyError(kind)


def side_bits(kind, t):
    m, n = t["shape"]
    if kind == "sep":
        return 16 * (m + n)
    if kind == "oracle":
        return 0
    if kind.startswith("block"):
        return 16 * m * (n // int(kind[5:]))
    if kind.startswith("nmf"):
        return 16 * int(kind[3:]) * (m + n)
    raise KeyError(kind)


def rc_at(F, th):
    """Coefficient rate in bits: sum max(0, 1/2 log2(lam/theta))."""
    return float(torch.clamp(0.5 * (torch.log2(F) - math.log2(th)), min=0).sum())


def kl_at(F, th):
    return float(torch.clamp(F, max=th).sum())


def realized_at(F, Ftrue, th):
    """Assumed realized KL: distortion of a coefficient is its true lam scaled by min(1, theta/lam_model)."""
    return float((Ftrue * torch.clamp(th / F, max=1.0)).sum())


def solve(f, target, lo, hi, increasing):
    """theta with f(theta) = target by bisection on log theta; f increasing (KL) or decreasing (rate) in theta."""
    lo, hi = math.log(lo) - 5.0, math.log(hi) + 1.0
    for _ in range(BISECT):
        mid = 0.5 * (lo + hi)
        v = f(math.exp(mid))
        need_larger = (v < target) if increasing else (v > target)
        lo, hi = (mid, hi) if need_larger else (lo, mid)
    return math.exp(0.5 * (lo + hi))


def evaluate(tensors, kind, q4_total, Ftrue, N):
    """Global floors for one variance model (flat lam over all tensors). Ftrue = oracle lam, for the realized column."""
    res = {}
    sizes = [t["shape"][0] * t["shape"][1] for t in tensors]
    flat = torch.empty(sum(sizes), device="cuda")
    S, off = 0, 0
    for i, t in enumerate(tensors):
        s = sigma2(kind, t, i)
        lam = (0.5 * t["g"][:, None] * t["h"][None, :] * s).clamp(min=EPS)
        flat[off:off + sizes[i]] = lam.reshape(-1)
        off += sizes[i]
        S += side_bits(kind, t)
        del s, lam
    views = list(flat.split(sizes))
    lo, hi = EPS, float(flat.max())
    res["side_bpw"] = S / N
    res["KL_full"] = float(flat.sum())
    # global floor at Q4 KL
    th_q4 = solve(lambda th: kl_at(flat, th), q4_total, lo, hi, increasing=True)
    res["theta_q4"] = th_q4
    res["R_star_allin"] = (rc_at(flat, th_q4) + S) / N
    res["R_star_coef"] = rc_at(flat, th_q4) / N
    # realized column (assumption stated in the header): solve the same KL target on the realized function
    th_real = solve(lambda th: realized_at(flat, Ftrue, th), q4_total, lo, hi, increasing=True)
    res["R_star_allin_realized"] = (rc_at(flat, th_real) + S) / N
    for r in RATES:
        th = solve(lambda th: rc_at(flat, th), r * N - S, lo, hi, increasing=False)
        res[f"KL_at_{r}"] = kl_at(flat, th)
        res[f"realized_KL_at_{r}"] = realized_at(flat, Ftrue, th)
    return res, flat, views, S


def uniform_floor(tensors, views, sizes_side, rate, N):
    """Each tensor at the same all-in rate (rate b/w of its own params), separable sigma2, reverse water-filling inside."""
    kl_tot = 0.0
    for v, sb, t in zip(views, sizes_side, tensors):
        n_t = v.numel()
        budget = rate * n_t - sb
        hi = float(v.max())
        th = solve(lambda th: rc_at(v, th), budget, EPS, hi, increasing=False)
        kl_tot += kl_at(v, th)
    return kl_tot


def main(out_path):
    t_all = time.time()
    if os.path.exists(FACT):
        d = torch.load(FACT, map_location="cuda")
        log("loaded factors", FACT)
    else:
        tens, mr, cs = capture_factors()
        d = {"tensors": tens, "max_rel_gguf_vs_hf": mr, "capture_s": cs}
    tensors = d["tensors"]
    N = sum(t["shape"][0] * t["shape"][1] for t in tensors)
    q4_total = sum(t["kq4"] for t in tensors)
    log("tensors", len(tensors), "N", N, "Q4 KL", q4_total)

    res = {"model": "Qwen/Qwen2.5-0.5B", "n_tensors": len(tensors), "N": N, "q4_kl_total": q4_total,
           "max_rel_gguf_vs_hf": d["max_rel_gguf_vs_hf"], "capture_s": d["capture_s"], "models": {}}
    # oracle lam (true C^2): reference for the realized column of every model
    Ftrue = torch.cat([(0.5 * t["g"][:, None] * t["h"][None, :] * t["C2"].clamp(min=EPS)).clamp(min=EPS).reshape(-1)
                       for t in tensors])
    for kind in MODELS:
        t0 = time.time()
        r, flat, views, S = evaluate(tensors, kind, q4_total, Ftrue, N)
        r["side_bits_total"] = S
        r["wall_s"] = time.time() - t0
        res["models"][kind] = r
        log(kind, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))
        if kind == "sep":
            sep_side = [side_bits("sep", t) for t in tensors]
            res["uniform_sep_1133_KL"] = uniform_floor(tensors, views, sep_side, 1.133, N)
            res["global_sep_1133_KL"] = r["KL_at_1.133"]
        del flat, views
        torch.cuda.empty_cache()
    res["wall_s"] = time.time() - t_all
    json.dump(res, open(out_path, "w"), indent=1)
    log("uniform_sep_1133_KL", res["uniform_sep_1133_KL"], "global_sep_1133_KL", res["global_sep_1133_KL"])
    log("wall", res["wall_s"])


if __name__ == "__main__":
    main(sys.argv[1])
