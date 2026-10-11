"""ONEBIT W15c: the converse side of the water-filling floor (negentropy, structure, coupling, scale).

The W15b floor (`kfac_floor.py`) is an achievability bound for a Gaussian source: at fixed second moments the Gaussian is
the hardest source, so the true rate-distortion function is lower by at most the negentropy J (Shannon lower bound,
Shannon 1959, `anchor`, not retrieved here): R(D) >= h(X) - sum 1/2 log2(2 pi e D_i), and for a component of variance s^2
the gap to the Gaussian rate 1/2 log2(s^2/D) is J = 1/2 log2(2 pi e s^2) - h(X) >= 0. J is a marginal part (per-component
density) plus a total-correlation part (dependence between components). This script measures both on Qwen2.5-0.5B.

Modes (usage: python tools/onebit/converse.py MODE OUT.json [args]):
  controls          estimator controls on synthetic data (no model): Gaussian -> J = 0, Laplace -> 1/2 log2(pi/e) =
                    0.1038, AR(1) Gaussian -> known total correlation, scale-mixture -> positive context gain, shuffled
                    null -> zero excess.
  main              K-FAC capture of all 168 linear tensors (as `kfac_floor.py`), then
                    - K0 = N exp(mean log lambda) from the log-dets of G, H and the separable variance (`model`),
                      against the bisection K0(r) = KL(r) 4^((r N - side)/N), and the active-set identity of
                      `Perf.CostCorrespondence.waterfill_highres` at the 1.18 and Q4-matching levels;
                    - marginal negentropy J of z = C/sigma_sep (C = U^T W V in the K-FAC eigenbases) over all
                      components, over the components active at the 1.18 and Q4-matching levels, and of the raw
                      weights W/sigma_raw;
                    - causal-context gain: bits saved per component when sigma^2 is the mean of 12 causal neighbours
                      (rows i-2..i, columns j-2..j+2, shrunk to sigma_sep^2 with weight 4) against sigma_sep^2;
                    - joint structure: Gaussian total correlation -1/2 sum log2 eig(corr(Z)) and the eigenvalue mass
                      above the edge, against a shuffled-entry null of the same marginal.
  ekfac             eigenvalue-corrected K-FAC on the 21 `ffn_down` tensors: s_ij = E[(U^T g)_i^2 (V^T x)_j^2] per token
                    (George et al. 2018, `anchor`, not retrieved), validated on the 5 arms of W15a/W15b.
  scale MODEL OUT.json HF GGUF16 GGUFQ4 DTYPE N_CHUNKS K_SAMPLES [LAYERS]
                    K0/N and Q4_K_M K-FAC KL of another model; R* = log4(K0 / KL_Q4) net of and with side information.

Run with ~/spinor-ttl/.venv/bin/python (torch + CUDA), from the repository root.
Cost (main): capture as `kfac_floor.py` (~80 s) + per tensor 2 eigh, a causal conv, one eigvalsh of at most 896^2 for Z
and for its shuffle (~60 s); CPU holds 3 float32 arrays of 357.8M entries (4.3 GB).
"""
import json, math, os, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch: the 8960^2 buffers fragment
import numpy as np
import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kfac_floor import read, module, KINDS, EXACT_QUAD, RATES, level_for_rate  # noqa: E402
from fisher_capture import ARMS, BLOCKS, CALIB, CHUNK, FILES, gguf  # noqa: E402

DELTA, LIM = 0.01, 40.0
NB = int(2 * LIM / DELTA)
EPS = 1e-30
LN2 = math.log(2.0)
KAPPA = 4.0
C_NATS = 2.0 * LN2  # rate = log(lam/theta)/c nats -> bits: 1/2 log2


class Hist:
    """Histogram of a standardised variable: plug-in differential entropy (bits) and negentropy."""

    def __init__(self):
        self.h = torch.zeros(NB, dtype=torch.int64, device="cuda")
        self.n, self.s2 = 0, 0.0

    def add(self, z):
        z = z.reshape(-1).float()
        idx = ((z.clamp(-LIM + DELTA, LIM - DELTA) + LIM) / DELTA).long()
        self.h += torch.bincount(idx, minlength=NB)
        self.n += z.numel()
        self.s2 += float((z.double() ** 2).sum())

    def h_bits(self):
        p = self.h.double() / max(self.n, 1)
        nz = p > 0
        return float(-(p[nz] * torch.log2(p[nz] / DELTA)).sum())

    def J(self):
        return 0.5 * math.log2(2 * math.pi * math.e * self.s2 / self.n) - self.h_bits()


def sep_sigma2(C2):
    """Separable variance r_i c_j / m from the row and column means of C^2 (grand mean m)."""
    m = C2.mean()
    return (C2.mean(1, keepdim=True) * C2.mean(0, keepdim=True) / m).clamp(min=EPS)


_K = torch.zeros(1, 1, 3, 5)
_K[0, 0, 0, :] = 1
_K[0, 0, 1, :] = 1
_K[0, 0, 2, :2] = 1


def ctx_sigma2(C2, s2sep):
    """Mean of C^2 over the 12 causal neighbours (rows i-2..i, columns j-2..j+2, row i only j-2, j-1), shrunk to s2sep."""
    k = _K.to(C2.device)
    x = C2[None, None]
    # conv2d is correlation: out[i, j] = sum k[r, c] x[i + r - 2, j + c - 2]; zero pad 2 above, 2 left and right
    pad = (2, 2, 2, 0)
    num = F.conv2d(F.pad(x, pad), k)[0, 0]
    cnt = F.conv2d(F.pad(torch.ones_like(x), pad), k)[0, 0]
    return ((num + KAPPA * s2sep) / (cnt + KAPPA)).clamp(min=EPS)


def tc_gauss(Z):
    """Gaussian total correlation per sample vector, bits: -1/2 sum log2 eig(corr). Variables = the smaller dimension.
    Returns (tc, eigenvalues, samples)."""
    if Z.shape[0] < Z.shape[1]:
        Z = Z.t()
    a, b = Z.shape
    Zd = Z.double()
    S = Zd.t() @ Zd
    d = S.diag().clamp(min=1e-300).sqrt()
    ev = torch.linalg.eigvalsh(S / d[:, None] / d[None, :]).clamp(min=1e-12)
    return float(-0.5 * torch.log2(ev).sum()), ev, a


def structure(Z):
    """Real Z against a shuffle of its own entries (same marginal, no structure): excess TC (bits, total over samples),
    eigenvalue mass above the shuffle's top eigenvalue as a fraction of the total."""
    tc, ev, a = tc_gauss(Z)
    perm = torch.randperm(Z.numel(), device=Z.device)
    tc0, ev0, _ = tc_gauss(Z.reshape(-1)[perm].reshape(Z.shape))
    edge = float(ev0.max())
    mass = float(ev[ev > edge].sum() / ev.sum())
    mass0 = float(ev0[ev0 > edge].sum() / ev0.sum())
    b = ev.numel()
    y = b / a
    return {"tc": tc, "tc_null": tc0, "excess_bits_total": a * (tc - tc0), "a": a, "b": b,
            "n_above_null_edge": int((ev > edge).sum()), "mass_above_null_edge": mass, "mass_null": mass0,
            "mp_edge": (1 + math.sqrt(y)) ** 2, "top_eig": float(ev.max()), "null_top_eig": edge}


def levels_theta_for_rate(lam, budget_bits):
    """theta (bisection on log theta) with sum max(0, 1/2 log2(lam/theta)) = budget_bits; lam on the GPU."""
    lo, hi = math.log(float(lam.min())) - 50, math.log(float(lam.max()))
    for _ in range(80):
        mid = (lo + hi) / 2
        rate = float(torch.clamp(0.5 * (torch.log2(lam) - mid / LN2), min=0).sum(dtype=torch.float64))
        lo, hi = (mid, hi) if rate > budget_bits else (lo, mid)
    return math.exp(hi)


def theta_for_kl(lam, kl_target):
    lo, hi = math.log(float(lam.min())) - 50, math.log(float(lam.max()))
    for _ in range(80):
        mid = (lo + hi) / 2
        kl = float(torch.clamp(lam, max=math.exp(mid)).sum(dtype=torch.float64))
        lo, hi = (mid, hi) if kl < kl_target else (lo, mid)
    return math.exp(lo)


def capture(model, ids, n_chunks, k_samples, layers, only=None):
    """As `kfac_floor.capture`, with float32 casts (bf16 models) and an optional set of kind names `only`."""
    G, H, outs, ins, handles = {}, {}, {}, {}, []
    for l in layers:
        for name, path, key in KINDS:
            if only is not None and name not in only:
                continue
            m = module(model, l, path)
            G[(l, name)] = torch.zeros(m.out_features, m.out_features, device="cuda")
            if (l, key) not in H:
                H[(l, key)] = torch.zeros(m.in_features, m.in_features, device="cuda")

            def pre(mod, args, l=l, key=key):
                if (l, key) not in ins:
                    ins[(l, key)] = args[0].detach().reshape(-1, args[0].shape[-1]).float()

            def post(mod, args, out, l=l, name=name):
                outs[(l, name)] = out

            handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    keys = list(G)
    for c in range(n_chunks):
        x = torch.tensor([ids[c * CHUNK:(c + 1) * CHUNK]], device="cuda")
        logp = torch.log_softmax(model(x).logits[0].float(), -1)
        for k, a in ins.items():
            H[k].addmm_(a.t(), a)
        for s_i in range(k_samples):
            s = torch.multinomial(logp.exp().detach(), 1)
            grads = torch.autograd.grad(-logp.gather(1, s).sum(), [outs[k] for k in keys],
                                        retain_graph=s_i < k_samples - 1)
            for k, g in zip(keys, grads):
                g = g.reshape(-1, g.shape[-1]).float()
                G[k].addmm_(g.t(), g)
            del grads
        outs.clear()
        ins.clear()
    for hd in handles:
        hd.remove()
    n_tok = n_chunks * CHUNK
    for v in G.values():
        v.div_(n_tok * k_samples)
    for v in H.values():
        v.div_(n_tok)
    return G, H


def load_model(hf, dtype):
    tok = transformers.AutoTokenizer.from_pretrained(hf)
    ids = tok.encode(open(CALIB, encoding="utf-8").read())
    model = transformers.AutoModelForCausalLM.from_pretrained(hf, dtype=dtype).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.register_forward_hook(lambda mod, a, out: out.detach().requires_grad_(True))
    return model, ids


# ----------------------------------------------------------------------------------------------------------------
def run_controls(out_path):
    torch.manual_seed(1)
    res = {}
    n = 20_000_000
    for name, z, expect in (
        ("gaussian", torch.randn(n, device="cuda"), 0.0),
        ("laplace", torch.distributions.Laplace(0.0, 1.0).sample((n,)).cuda(), 0.5 * math.log2(math.pi / math.e)),
    ):
        h = Hist()
        h.add(z)
        res[name] = {"J_bits": h.J(), "expected": expect, "n": n}
        print(name, res[name], flush=True)
    # AR(1) Gaussian: TC per sample = -1/2 (b-1) log2(1 - rho^2)
    b, a, rho = 512, 8192, 0.3
    x = torch.randn(a, b, device="cuda", dtype=torch.float64)
    z = torch.empty_like(x)
    z[:, 0] = x[:, 0]
    for j in range(1, b):
        z[:, j] = rho * z[:, j - 1] + math.sqrt(1 - rho ** 2) * x[:, j]
    st = structure(z.float())
    st["expected_tc"] = -0.5 * (b - 1) * math.log2(1 - rho ** 2)
    res["ar1"] = st
    print("ar1", st, flush=True)
    iid = structure(torch.randn(a, b, device="cuda"))
    res["iid"] = iid
    print("iid", iid, flush=True)
    # context gain: Laplace iid (zero gain) and a smooth variance field (positive gain)
    m_, n_ = 1024, 2048
    for name, field in (("ctx_iid_laplace", torch.zeros(m_, n_, device="cuda")),
                        ("ctx_smooth_field", 0.9 * torch.sin(torch.arange(m_, device="cuda")[:, None] / 40.0)
                         * torch.cos(torch.arange(n_, device="cuda")[None, :] / 25.0))):
        c = torch.distributions.Laplace(0.0, 1.0).sample((m_, n_)).cuda() * torch.exp(field)
        c2 = c ** 2
        s2 = sep_sigma2(c2)
        s2c = ctx_sigma2(c2, s2)
        zs, zc = c / s2.sqrt(), c / s2c.sqrt()
        hs, hc = Hist(), Hist()
        hs.add(zs)
        hc.add(zc)
        gain = float(0.5 * torch.log2(s2 / s2c).mean()) + hs.h_bits() - hc.h_bits()
        res[name] = {"gain_bits": gain, "J_sep": hs.J()}
        print(name, res[name], flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------------
def run_main(out_path, n_chunks=100, k_samples=4):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    model, ids = load_model("Qwen/Qwen2.5-0.5B", torch.float32)
    assert len(ids) >= n_chunks * CHUNK
    readers = {k: gguf.GGUFReader(FILES[k]) for k in ("f16", "q4_k_m")}
    N, side_bits, sumlog, nfloor = 0, 0, 0.0, 0
    kfac_q4_all, max_rel = 0.0, 0.0
    store = []  # (kind, lam, z, lsd) on the CPU
    hist_all, hist_raw, hist_ctx_all = Hist(), Hist(), Hist()
    sum_lsd_all = 0.0
    struct = {}
    t_cap = 0.0
    for half in (range(0, 12), range(12, 24)):
        t1 = time.time()
        G, H = capture(model, ids, n_chunks, k_samples, half)
        t_cap += time.time() - t1
        print("capture", list(half)[0], time.time() - t1, flush=True)
        for l in half:
            for name, path, key in KINDS:
                gname = f"blk.{l}.{name}.weight"
                w = read(readers["f16"], gname).cuda()
                hfw = module(model, l, path).weight.detach()
                max_rel = max(max_rel, float((w - hfw).norm() / hfw.norm()))
                Gm, Hm = G.pop((l, name)), H[(l, key)]
                q4 = read(readers["q4_k_m"], gname).cuda() - w
                kfac_q4_all += 0.5 * float(((Gm @ q4) * (q4 @ Hm)).sum(dtype=torch.float64))
                g, U = torch.linalg.eigh(Gm)
                h, V = torch.linalg.eigh(Hm)
                nfloor += int((g < EPS).sum() + (h < EPS).sum())
                g, h = g.clamp(min=EPS), h.clamp(min=EPS)
                C = U.t() @ w @ V
                C2 = C ** 2
                s2 = sep_sigma2(C2)
                rows, cols = w.shape
                # log-det form: mean log lambda = mean log(g_i/2) + mean log h_j + mean log r_i + mean log c_j - log m
                m = C2.mean()
                r = C2.mean(1).clamp(min=EPS)
                c = C2.mean(0).clamp(min=EPS)
                sumlog += float(cols * (torch.log(g / 2) + torch.log(r)).sum(dtype=torch.float64)
                                + rows * (torch.log(h) + torch.log(c)).sum(dtype=torch.float64)
                                - rows * cols * torch.log(m).double())
                lam = (0.5 * g[:, None] * h[None, :] * s2)
                z = C / s2.sqrt()
                s2c = ctx_sigma2(C2, s2)
                lsd = 0.5 * torch.log2(s2 / s2c)
                zc = C / s2c.sqrt()
                hist_all.add(z)
                hist_ctx_all.add(zc)
                sum_lsd_all += float(lsd.double().sum())
                # raw basis
                W2 = w ** 2
                hist_raw.add(w / sep_sigma2(W2).sqrt())
                st = structure(z)
                acc = struct.setdefault(name, {"excess_bits_total": 0.0, "n": 0, "n_above": 0, "mass_w": 0.0,
                                               "mass_null_w": 0.0, "tc_total": 0.0})
                acc["excess_bits_total"] += st["excess_bits_total"]
                acc["n"] += rows * cols
                acc["n_above"] += st["n_above_null_edge"]
                acc["mass_w"] += st["mass_above_null_edge"] * rows * cols
                acc["mass_null_w"] += st["mass_null"] * rows * cols
                acc["tc_total"] += st["a"] * st["tc"]
                store.append((lam.flatten().cpu(), z.flatten().cpu(), lsd.flatten().cpu()))
                N += rows * cols
                side_bits += 16 * (rows + cols)
                del Gm, U, V, C, C2, s2, s2c, lsd, zc, q4, lam, z, W2
            for name, _, key in KINDS:
                H.pop((l, key), None)
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    print("analysis A done", time.time() - t0, "max_rel", max_rel, flush=True)

    res = {"n_chunks": n_chunks, "k_samples": k_samples, "N": N, "side_bits": side_bits,
           "side_bpw": side_bits / N, "floored_eigenvalues": nfloor, "max_rel_gguf_vs_hf": max_rel,
           "q4_k_m_kfac_all": kfac_q4_all}
    # K0 from log-dets
    res["K0_logdet"] = N * math.exp(sumlog / N)
    res["mean_log_lam_logdet"] = sumlog / N
    lam = torch.cat([s[0] for s in store]).cuda()
    res["mean_log_lam_array"] = float(torch.log(lam.clamp(min=EPS)).mean(dtype=torch.float64))
    res["K0_array"] = N * math.exp(res["mean_log_lam_array"])
    res["lam_min"], res["lam_max"] = float(lam.min()), float(lam.max())
    kb = {}
    for rr in RATES:
        th = levels_theta_for_rate(lam, rr * N - side_bits)
        kl = float(torch.clamp(lam, max=th).sum(dtype=torch.float64))
        act = float((lam > th).sum()) / N
        kb[str(rr)] = {"theta": th, "kl": kl, "active_frac": act,
                       "K0_bis": kl * 4 ** ((rr * N - side_bits) / N)}
        print("rate", rr, kb[str(rr)], flush=True)
    res["bisection"] = kb
    # Q4-matching level and the active-set identity of waterfill_highres
    th_q4 = theta_for_kl(lam, kfac_q4_all)
    levels = {"r1.18": levels_theta_for_rate(lam, 1.18 * N - side_bits), "q4": th_q4}
    res["levels"] = {}
    for tag, th in levels.items():
        mask = lam > th
        nA = int(mask.sum())
        rate_net = float((0.5 * (torch.log2(lam[mask]) - math.log2(th))).sum(dtype=torch.float64))
        sumlogA = float(torch.log(lam[mask]).sum(dtype=torch.float64))
        th_pred = math.exp((sumlogA - C_NATS * rate_net) / nA)
        kl = float(torch.clamp(lam, max=th).sum(dtype=torch.float64))
        res["levels"][tag] = {"theta": th, "n_active": nA, "active_frac": nA / N, "rate_net_bpw": rate_net / N,
                              "rate_allin_bpw": (rate_net + side_bits) / N, "kl": kl,
                              "theta_pred_logdet_active": th_pred, "theta_rel_err": th_pred / th - 1,
                              "K0_active": nA * math.exp(sumlogA / nA) * 4 ** (-0.0) }
        print("level", tag, res["levels"][tag], flush=True)
    del lam
    torch.cuda.empty_cache()
    # negentropy and context gain over the active sets
    res["J"] = {"C_all": {"J_bits": hist_all.J(), "frac": 1.0, "var": hist_all.s2 / hist_all.n},
                "raw_all": {"J_bits": hist_raw.J(), "frac": 1.0, "var": hist_raw.s2 / hist_raw.n}}
    res["ctx"] = {"all": {"gain_bits": sum_lsd_all / N + hist_all.h_bits() - hist_ctx_all.h_bits()}}
    for tag, th in levels.items():
        hs, hc, nA, sl = Hist(), Hist(), 0, 0.0
        for lam_t, z_t, lsd_t in store:
            mk = lam_t > th
            if not bool(mk.any()):
                continue
            z_a = z_t[mk].cuda()
            lsd_a = lsd_t[mk].cuda()
            hs.add(z_a)
            hc.add(z_a * torch.exp2(lsd_a))  # z_ctx = z_sep sigma_sep / sigma_ctx = z_sep 2^lsd
            nA += z_a.numel()
            sl += float(lsd_a.double().sum())
        res["J"]["C_active_" + tag] = {"J_bits": hs.J(), "frac": nA / N, "J_bpw": hs.J() * nA / N,
                                       "var": hs.s2 / hs.n}
        res["ctx"]["active_" + tag] = {"gain_bits": sl / nA + hs.h_bits() - hc.h_bits(), "frac": nA / N,
                                       "gain_bpw": (sl / nA + hs.h_bits() - hc.h_bits()) * nA / N}
        print("J", tag, res["J"]["C_active_" + tag], res["ctx"]["active_" + tag], flush=True)
    res["J"]["C_all"]["J_bpw"] = res["J"]["C_all"]["J_bits"]
    res["J"]["raw_all"]["J_bpw"] = res["J"]["raw_all"]["J_bits"]
    print("J all", res["J"]["C_all"], res["J"]["raw_all"], res["ctx"]["all"], flush=True)
    tot_excess = sum(v["excess_bits_total"] for v in struct.values())
    res["structure"] = {k: {"excess_bpw": v["excess_bits_total"] / v["n"], "n_above_null_edge": v["n_above"],
                            "mass_above_edge": v["mass_w"] / v["n"], "mass_null": v["mass_null_w"] / v["n"],
                            "tc_bpw": v["tc_total"] / v["n"]} for k, v in struct.items()}
    res["structure_total_excess_bpw"] = tot_excess / N
    print("structure", json.dumps(res["structure"]), "total", tot_excess / N, flush=True)
    res["wall_s"], res["capture_s"] = time.time() - t0, t_cap
    json.dump(res, open(out_path, "w"), indent=1)
    print("wall", res["wall_s"])


# ----------------------------------------------------------------------------------------------------------------
def run_ekfac(out_path, n_chunks=100, k_samples=4):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    model, ids = load_model("Qwen/Qwen2.5-0.5B", torch.float32)
    readers = {k: gguf.GGUFReader(FILES[k]) for k in ("f16", *ARMS)}
    layers = list(BLOCKS)
    # pass 1: G, H of the 21 ffn_down tensors
    G, H = capture(model, ids, n_chunks, k_samples, layers, only={"ffn_down"})
    print("pass1", time.time() - t0, flush=True)
    U, V, gd, hd = {}, {}, {}, {}
    for l in layers:
        gg, U[l] = torch.linalg.eigh(G.pop((l, "ffn_down")))
        hh, V[l] = torch.linalg.eigh(H.pop((l, "down_in")))
        gd[l], hd[l] = gg.clamp(min=0), hh.clamp(min=0)
    V = {l: v.cpu() for l, v in V.items()}
    # pass 2: s_ij = E[(U^T g)_i^2 (V^T x)_j^2] per token
    torch.manual_seed(0)
    S = {l: torch.zeros(896, 4864, device="cuda") for l in layers}
    outs, xs, handles = {}, {}, []
    for l in layers:
        m = module(model, l, "mlp.down_proj")

        def pre(mod, args, l=l):
            xs[l] = args[0].detach().reshape(-1, args[0].shape[-1]).float()

        def post(mod, args, out, l=l):
            outs[l] = out

        handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    Vg = {}
    for c in range(n_chunks):
        x = torch.tensor([ids[c * CHUNK:(c + 1) * CHUNK]], device="cuda")
        logp = torch.log_softmax(model(x).logits[0].float(), -1)
        B = {}
        for l in layers:
            Vl = Vg.get(l)
            if Vl is None:
                Vl = V[l].cuda()
            B[l] = (xs[l] @ Vl) ** 2
            del Vl
        for s_i in range(k_samples):
            s = torch.multinomial(logp.exp().detach(), 1)
            grads = torch.autograd.grad(-logp.gather(1, s).sum(), [outs[l] for l in layers],
                                        retain_graph=s_i < k_samples - 1)
            for l, gr in zip(layers, grads):
                A = (gr.reshape(-1, 896).float() @ U[l]) ** 2
                S[l].addmm_(A.t(), B[l])
            del grads
        outs.clear()
        xs.clear()
    for hdl in handles:
        hdl.remove()
    n_tok = n_chunks * CHUNK
    for l in layers:
        S[l].div_(n_tok * k_samples)
    print("pass2", time.time() - t0, flush=True)
    del model
    torch.cuda.empty_cache()
    kf = {a: 0.0 for a in ARMS}
    ek = {a: 0.0 for a in ARMS}
    ratio_w = {a: 0.0 for a in ARMS}
    for l in layers:
        gname = f"blk.{l}.ffn_down.weight"
        w = read(readers["f16"], gname).cuda()
        Vl = V[l].cuda()
        K = gd[l][:, None] * hd[l][None, :]
        for a in ARMS:
            d = read(readers[a], gname).cuda() - w
            E2 = (U[l].t() @ d @ Vl) ** 2
            kf[a] += 0.5 * float((K * E2).sum(dtype=torch.float64))
            ek[a] += 0.5 * float((S[l] * E2).sum(dtype=torch.float64))
            ratio_w[a] += float((S[l] * E2).sum(dtype=torch.float64)) / max(float((K * E2).sum(dtype=torch.float64)), 1e-300) / len(layers)
    res = {"n_chunks": n_chunks, "k_samples": k_samples, "validation": {}}
    for a in ARMS:
        res["validation"][a] = {"kfac": kf[a], "ekfac": ek[a], "exact_quadratic": EXACT_QUAD[a],
                                "kfac_ratio": kf[a] / EXACT_QUAD[a], "ekfac_ratio": ek[a] / EXACT_QUAD[a],
                                "s_over_gh_weighted_by_E2": ratio_w[a]}
        print(a, res["validation"][a], flush=True)
    order_ex = sorted(ARMS, key=lambda a: EXACT_QUAD[a])
    res["order_exact"] = order_ex
    res["order_kfac"] = sorted(ARMS, key=lambda a: kf[a])
    res["order_ekfac"] = sorted(ARMS, key=lambda a: ek[a])
    print(res["order_exact"], res["order_kfac"], res["order_ekfac"])
    res["wall_s"] = time.time() - t0
    json.dump(res, open(out_path, "w"), indent=1)
    print("wall", res["wall_s"])


# ----------------------------------------------------------------------------------------------------------------
LB0, LBW, LBN = 130.0, 0.02, 7500  # log2(lambda) histogram: lo edge -LB0, bin width, count


def run_scale(out_path, hf, gguf16, ggufq4, dtype, n_chunks, k_samples, layers_arg, group):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    model, ids = load_model(hf, dt)
    assert len(ids) >= n_chunks * CHUNK, (len(ids), n_chunks * CHUNK)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    layers = list(range(n_layers)) if layers_arg == "all" else [int(x) for x in layers_arg.split(",")]
    r16, rq4 = gguf.GGUFReader(gguf16), gguf.GGUFReader(ggufq4)
    cnt = torch.zeros(LBN, dtype=torch.int64, device="cuda")
    lsum = torch.zeros(LBN, dtype=torch.float64, device="cuda")
    n_sel, sumlog, side_bits, kq4, max_rel = 0, 0.0, 0, 0.0, 0.0
    per_kind = {}
    t_cap = 0.0
    # G for gate/up and H for down are intermediate_size^2 fp32 (321 MB at 8960): cap the layers held per capture so
    # the buffers (about 1 GB per layer at 8960) fit beside the model on a 16 GB card
    group = max(1, min(group, int(4 * 4864 * 4864 // (cfg.intermediate_size ** 2))))
    groups = [layers[i:i + group] for i in range(0, len(layers), group)]
    for grp in groups:
        t1 = time.time()
        G, H = capture(model, ids, n_chunks, k_samples, grp)
        t_cap += time.time() - t1
        print("capture", grp[0], time.time() - t1, flush=True)
        t2 = time.time()
        for l in grp:
            for name, path, key in KINDS:
                gname = f"blk.{l}.{name}.weight"
                w = read(r16, gname).cuda()
                hfw = module(model, l, path).weight.detach().float()
                max_rel = max(max_rel, float((w - hfw).norm() / hfw.norm()))
                Gm, Hm = G.pop((l, name)), H[(l, key)]
                q4 = read(rq4, gname).cuda() - w
                k4 = 0.5 * float(((Gm @ q4) * (q4 @ Hm)).sum(dtype=torch.float64))
                kq4 += k4
                per_kind[name] = per_kind.get(name, 0.0) + k4
                g, U = torch.linalg.eigh(Gm)
                h, V = torch.linalg.eigh(Hm)
                g, h = g.clamp(min=EPS), h.clamp(min=EPS)
                C2 = (U.t() @ w @ V) ** 2
                rows, cols = w.shape
                m = C2.mean()
                r = C2.mean(1).clamp(min=EPS)
                c = C2.mean(0).clamp(min=EPS)
                sumlog += float(cols * (torch.log(g / 2) + torch.log(r)).sum(dtype=torch.float64)
                                + rows * (torch.log(h) + torch.log(c)).sum(dtype=torch.float64)
                                - rows * cols * torch.log(m).double())
                lam = (0.5 * g[:, None] * h[None, :] * sep_sigma2(C2)).flatten()
                idx = ((torch.log2(lam.clamp(min=EPS)) + LB0) / LBW).long().clamp(0, LBN - 1)
                cnt += torch.bincount(idx, minlength=LBN)
                lsum.index_add_(0, idx, lam.double())
                n_sel += rows * cols
                side_bits += 16 * (rows + cols)
                del Gm, U, V, C2, lam, idx, q4
            for name, _, key in KINDS:
                H.pop((l, key), None)
            torch.cuda.empty_cache()
        w = hfw = Gm = Hm = q4 = g = h = U = V = C2 = None
        torch.cuda.empty_cache()
        print("analysis", grp[0], time.time() - t2, torch.cuda.memory_allocated() / 2 ** 20, flush=True)
    frac = len(layers) / n_layers
    N_full = None
    # parameters in the 7 kinds per layer
    nl = sum(module(model, 0, p).weight.numel() for _, p, _ in KINDS)
    N_full = nl * n_layers
    K0 = N_full * math.exp(sumlog / n_sel)
    kq4_full = kq4 / frac
    side_bpw = side_bits / n_sel
    res = {"model": hf, "dtype": dtype, "layers": layers, "n_layers": n_layers, "n_chunks": n_chunks,
           "k_samples": k_samples, "N_sample": n_sel, "N_full": N_full, "side_bpw": side_bpw,
           "max_rel_gguf_vs_hf": max_rel, "mean_log_lam": sumlog / n_sel, "K0": K0, "K0_over_N": K0 / N_full,
           "kl_q4_sample": kq4, "kl_q4_full_est": kq4_full, "kl_q4_by_kind_sample": per_kind,
           "R_star_net": math.log(K0 / kq4_full, 4), "R_star_allin": math.log(K0 / kq4_full, 4) + side_bpw}
    # histogram bisection (bin-resolution water-filling) on the sampled layers
    cen = cnt.double()
    lbar = (lsum / cnt.clamp(min=1)).cpu()
    cen = cen.cpu()
    ok = cen > 0
    lb, cn = lbar[ok], cen[ok]
    log2l = torch.log2(lb)

    def at(th):
        kl = float((cn * torch.clamp(lb, max=th)).sum())
        rate = float((cn * torch.clamp(0.5 * (log2l - math.log2(th)), min=0)).sum())
        return kl, rate

    def theta_rate(budget):
        lo, hi = math.log(float(lb.min())) - 50, math.log(float(lb.max()))
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if at(math.exp(mid))[1] > budget else (lo, mid)
        return math.exp(hi)

    scale = N_full / n_sel
    res["hist_bisection"] = {}
    for rr in (1.18, 3.0):
        th = theta_rate(rr * n_sel - side_bits)
        kl, rate = at(th)
        res["hist_bisection"][str(rr)] = {"theta": th, "kl_full_est": kl * scale}
    lo, hi = math.log(float(lb.min())) - 50, math.log(float(lb.max()))
    target = kq4  # sample KL of Q4
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if at(math.exp(mid))[0] < target else (lo, mid)
    th_q4 = math.exp(lo)
    kl, rate = at(th_q4)
    res["hist_bisection"]["q4_match"] = {"theta": th_q4, "kl_sample": kl, "rate_net_bpw": rate / n_sel,
                                         "rate_allin_bpw": rate / n_sel + side_bpw}
    res["wall_s"], res["capture_s"] = time.time() - t0, t_cap
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "controls":
        run_controls(out)
    elif mode == "main":
        run_main(out)
    elif mode == "ekfac":
        run_ekfac(out)
    elif mode == "scale":
        hf, g16, gq4, dtype, nc, ks = sys.argv[3:9]
        run_scale(out, hf, g16, gq4, dtype, int(nc), int(ks), sys.argv[9] if len(sys.argv) > 9 else "all",
                  int(sys.argv[10]) if len(sys.argv) > 10 else 4)
    else:
        raise SystemExit(__doc__)
