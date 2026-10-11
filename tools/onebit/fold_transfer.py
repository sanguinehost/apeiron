"""ONEBIT fold-transfer: sequential output-fitted one-bit transfer of Qwen2.5-0.5B, with the apeiron fold solve.

Mirrors, in torch f64 on the GPU:
  src/learn/onebit.rs  gptq_row / feedback_u  (arm Gptq1: damp 0.01*mean(diag), sign grid {-a,+a}, a = mean|error-fed
                       128-block|, sign(0)=+1, err=(w-q)/U[j,j], wk[k] -= err*U[j,k]);  UtU = H^-1, U upper
  src/learn/ridge.rs   FoldTrainer::solve     (W = (XtX + lambda I)^-1 XtY, one SPD Cholesky solve)
The Rust is not driven from here (the deliverable needs forward passes and KL in torch); the mirror is checked by
--controls against a line-by-line numpy transcription of gptq_row.

Student weight of one linear tensor:  W'[o,i] = r[o,g(i)] * S[o,i] * c[i],  S in {+-1} (1 b/w), r f16 per (row, 128-group),
c f16 per input column (one vector per Gram-sharing group).  Alternation per tensor group, fixed Gram H = X_s^T X_s:
  continuous step (exact fold solves):  W* = (H + ridge I)^-1 X_s^T Y_t ;  c = argmin  (A c = b, A = H_l o (S^T S));
                                        r = argmin per row (G x G normal equations)
  discrete step:                        S, r0 = gptq1(W*/c ; diag(c) H diag(c))
Layers are fitted in order; each stage (qkv, o, gate/up, down) takes its X_s and H from the already-quantised upstream.

usage: fold_transfer.py --out OUT.json [--variants a,b,..] [--controls] [--layers N] [--iters K]
variants: rtn gptq1-t gptq1-s fold-ns fold-local fold-block fp-ridge fp-ridge-mut q4km gguf-f16
Run with /home/socol/Workspace/spinor-ttl/.venv/bin/python (torch + CUDA).
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, "/home/socol/Workspace/llama.cpp/gguf-py")
import gguf
from gguf.quants import dequantize

DEV = "cuda"
F64 = torch.float64
CALIB = "/home/socol/oc-tmp/seal/calib-c4.txt"
GGUF_F16 = "/home/socol/models/gguf/qwen05b-f16.gguf"
GGUF_Q4KM = "/home/socol/oc-tmp/seal/qwen05b-Q4_K_M-imat.gguf"
GRAM_DIR = "/home/socol/oc-tmp/gptq-w2/gram"
CHUNK, GROUP = 512, 128
KINDS = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "gate": "ffn_gate", "up": "ffn_up",
         "down": "ffn_down"}
STAGES = [("qkv", ["q", "k", "v"], "xa"), ("o", ["o"], "a"), ("ffn", ["gate", "up"], "xf"), ("down", ["down"], "m")]


def tick():
    torch.cuda.synchronize()
    return time.time()


# ---------------------------------------------------------------- model (Qwen2, fp32)
class Cfg:
    pass


def rope_tables(cfg):
    inv = 1.0 / (cfg.theta ** (torch.arange(0, cfg.hd, 2, device=DEV, dtype=torch.float32) / cfg.hd))
    fr = torch.outer(torch.arange(CHUNK, device=DEV, dtype=torch.float32), inv)
    emb = torch.cat([fr, fr], -1)
    return emb.cos(), emb.sin()


def rot(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def rms(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def parts(cfg, L, h):
    """One decoder layer; every intermediate a stage may need."""
    B, T, _ = h.shape
    xa = rms(h, L["ln1"], cfg.eps)
    q = xa @ L["q"].T + L["qb"]
    k = xa @ L["k"].T + L["kb"]
    v = xa @ L["v"].T + L["vb"]
    qh = q.view(B, T, cfg.nh, cfg.hd).transpose(1, 2)
    kh = k.view(B, T, cfg.nkv, cfg.hd).transpose(1, 2)
    vh = v.view(B, T, cfg.nkv, cfg.hd).transpose(1, 2)
    qh = qh * cfg.cos + rot(qh) * cfg.sin
    kh = kh * cfg.cos + rot(kh) * cfg.sin
    rep = cfg.nh // cfg.nkv
    kh, vh = kh.repeat_interleave(rep, 1), vh.repeat_interleave(rep, 1)
    a = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True).transpose(1, 2).reshape(B, T, cfg.nh * cfg.hd)
    mid = h + a @ L["o"].T
    xf = rms(mid, L["ln2"], cfg.eps)
    g = xf @ L["gate"].T
    u = xf @ L["up"].T
    m = F.silu(g) * u
    out = mid + m @ L["down"].T
    return dict(xa=xa, q=q, k=k, v=v, a=a, mid=mid, xf=xf, g=g, u=u, m=m, out=out)


def logits_of(cfg, Ls, ids):
    h = cfg.E[ids]
    for L in Ls:
        h = parts(cfg, L, h)["out"]
    return rms(h, cfg.fnorm, cfg.eps) @ cfg.E.T


def load_model():
    cfg = Cfg()
    m = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
    c = m.config
    cfg.nl, cfg.nh, cfg.nkv, cfg.hid, cfg.inter = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.hidden_size, c.intermediate_size
    cfg.hd, cfg.eps, cfg.theta = c.hidden_size // c.num_attention_heads, c.rms_norm_eps, float(c.rope_parameters["rope_theta"])
    cfg.cos, cfg.sin = rope_tables(cfg)
    sd = {k: v.detach().to(DEV) for k, v in m.state_dict().items()}
    cfg.E = sd["model.embed_tokens.weight"]
    cfg.fnorm = sd["model.norm.weight"]
    TL = []
    for l in range(cfg.nl):
        p = f"model.layers.{l}."
        TL.append(dict(ln1=sd[p + "input_layernorm.weight"], ln2=sd[p + "post_attention_layernorm.weight"],
                       q=sd[p + "self_attn.q_proj.weight"], qb=sd[p + "self_attn.q_proj.bias"],
                       k=sd[p + "self_attn.k_proj.weight"], kb=sd[p + "self_attn.k_proj.bias"],
                       v=sd[p + "self_attn.v_proj.weight"], vb=sd[p + "self_attn.v_proj.bias"],
                       o=sd[p + "self_attn.o_proj.weight"], gate=sd[p + "mlp.gate_proj.weight"],
                       up=sd[p + "mlp.up_proj.weight"], down=sd[p + "mlp.down_proj.weight"]))
    return cfg, TL, m


def gguf_layers(cfg, TL, path):
    """Teacher layers with the 168 linear weights replaced by the GGUF tensors (dequantised); returns bits."""
    rd = gguf.GGUFReader(path)
    byname = {t.name: t for t in rd.tensors}
    Ls, nbytes, nparam = [], 0, 0
    for l in range(cfg.nl):
        L = dict(TL[l])
        for k, kind in KINDS.items():
            t = byname[f"blk.{l}.{kind}.weight"]
            O, n = TL[l][k].shape
            if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                a = np.asarray(t.data, dtype=np.float32)
            else:
                a = dequantize(t.data, t.tensor_type)
            L[k] = torch.from_numpy(np.ascontiguousarray(a.reshape(O, n))).float().to(DEV)
            nbytes += int(t.n_bytes)
            nparam += O * n
        Ls.append(L)
    return Ls, 8.0 * nbytes / nparam


# ---------------------------------------------------------------- mirrors of src/learn
def feedback_u(H, damp=0.01):
    """onebit.rs feedback_u: H+damp*mean(diag) I (zero diag -> 1), U upper with U^T U = H^-1."""
    h = H.clone()
    md = H.diagonal().mean()
    h.diagonal().add_(damp * md)
    d = h.diagonal()
    d[d == 0] = 1.0
    Lc = torch.linalg.cholesky(h)
    hinv = torch.cholesky_inverse(Lc)
    return torch.linalg.cholesky((hinv + hinv.T) / 2, upper=True)


def gptq_signs(W, H, group=GROUP, damp=0.01):
    """onebit.rs gptq_row (identity order), rows batched, columns in lazy blocks of `group` (same arithmetic as the
    per-column Rust loop up to reassociation). Returns (a*sign, sign, a per (row, group))."""
    O, n = W.shape
    U = feedback_u(H, damp)
    wk = W.clone()
    Wq, S = torch.empty_like(W), torch.empty_like(W)
    A = torch.empty(O, n // group, dtype=W.dtype, device=W.device)
    for gi, g0 in enumerate(range(0, n, group)):
        g1 = g0 + group
        a = wk[:, g0:g1].abs().mean(1).clamp_min(1e-12)
        A[:, gi] = a
        Err = torch.empty(O, group, dtype=W.dtype, device=W.device)
        for j in range(g0, g1):
            wv = wk[:, j]
            s = 1.0 - 2.0 * (wv < 0).to(W.dtype)
            q = a * s
            err = (wv - q) / U[j, j]
            if j + 1 < g1:
                wk[:, j + 1:g1] -= err[:, None] * U[j, j + 1:g1]
            Err[:, j - g0] = err
            Wq[:, j] = q
            S[:, j] = s
        if g1 < n:
            wk[:, g1:] -= Err @ U[g0:g1, g1:]
    return Wq, S, A


def gptq_naive_np(W, H, group=GROUP, damp=0.01):
    """Line-by-line transcription of onebit.rs gptq_row/mean_abs/sign in numpy f64 (the control)."""
    O, n = W.shape
    h = H.copy()
    h[np.diag_indices(n)] += damp * np.mean(np.diag(H))
    h[np.diag_indices(n)] = np.where(np.diag(h) == 0, 1.0, np.diag(h))
    V = np.linalg.cholesky(h[::-1, ::-1])[::-1, ::-1]  # UL: h = V V^T, V upper
    U = np.linalg.inv(V)
    wk = W.copy()
    wq = np.zeros_like(W)
    for o in range(O):
        g0 = 0
        while g0 < n:
            g1 = min(g0 + group, n)
            a = max(np.mean(np.abs(wk[o, g0:g1])), 1e-12)
            for j in range(g0, g1):
                wv = wk[o, j]
                q = a * (-1.0 if wv < 0 else 1.0)
                wq[o, j] = q
                err = (wv - q) / U[j, j]
                for k in range(j + 1, n):
                    wk[o, k] -= err * U[j, k]
            g0 = g1
    return wq


def ridge_solve(H, C, ridge):
    """ridge.rs FoldTrainer::solve: (H + ridge*mean(diag) I)^-1 C, one Cholesky. C is [n, out]; returns W [out, n]."""
    Hr = H.clone()
    Hr.diagonal().add_(ridge * H.diagonal().mean())
    return torch.cholesky_solve(C, torch.linalg.cholesky(Hr)).T.contiguous()


def metric(H, ridge):
    Hl = H.clone()
    Hl.diagonal().add_(ridge * H.diagonal().mean())
    return Hl


def J(Ws, Wp, Hl):
    D = Ws - Wp
    return float(((D @ Hl) * D).sum())


def compose(S, r, c):
    return (S * r.repeat_interleave(GROUP, 1)) * c[None, :]


def c_step(Ws, Hl, S, r):
    Sh = S * r.repeat_interleave(GROUP, 1)
    A = Hl * (Sh.T @ Sh)
    b = ((Ws @ Hl) * Sh).sum(0)
    A.diagonal().add_(1e-12 * A.diagonal().mean())
    return torch.cholesky_solve(b[:, None], torch.linalg.cholesky(A))[:, 0]


def r_step(Ws, Hl, S, c):
    O, n = Ws.shape
    G = n // GROUP
    P = (S * c[None, :]).view(O, G, GROUP)
    M = torch.empty(O, G, G, dtype=Ws.dtype, device=Ws.device)
    for g in range(G):
        K = P[:, g, :] @ Hl[g * GROUP:(g + 1) * GROUP, :]
        M[:, g, :] = (K.view(O, G, GROUP) * P).sum(-1)
    t = (P * (Ws @ Hl).view(O, G, GROUP)).sum(-1)
    eye = torch.eye(G, dtype=Ws.dtype, device=Ws.device)
    M = M + 1e-12 * M.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * eye
    return torch.linalg.solve(M, t[:, :, None])[:, :, 0]


def f16r(x):
    return x.half().to(F64)


def fold_fit(Ws, H, iters, ridge, refit=True):
    """Alternation of the discrete step (gptq1) and the exact continuous steps. Returns (W', info)."""
    n = H.shape[0]
    Hl = metric(H, ridge)
    c = torch.ones(n, dtype=F64, device=Ws.device)
    Wq, S, r = gptq_signs(Ws, H)
    cands = [(J(Ws, Wq, Hl), "gptq", S, r, c)]
    trace = [cands[0][0]]
    if refit:
        for it in range(iters):
            c = c_step(Ws, Hl, S, r)
            s = c.abs().mean()
            c, r = c / s, r * s
            r = r_step(Ws, Hl, S, c)
            cands.append((J(Ws, compose(S, r, c), Hl), f"refit{it}", S, r, c))
            trace.append(cands[-1][0])
            if it < iters - 1:
                cs = torch.where(c.abs() < 0.05, 0.05 * (1.0 - 2.0 * (c < 0).to(F64)), c)
                _, S, r = gptq_signs(Ws / cs[None, :], H * cs[:, None] * cs[None, :])
    best = min(cands, key=lambda x: x[0])
    _, tag, S, r, c = best
    if not refit:
        return compose(S, f16r(r), torch.ones_like(c)), dict(tag=tag, J=trace, J_best=best[0])
    rh, ch = f16r(r), f16r(c)
    Wp = compose(S, rh, ch)
    return Wp, dict(tag=tag, J=trace, J_best=best[0], J_f16=J(Ws, Wp, Hl))


# ---------------------------------------------------------------- sequential transfer
def run_variant(cfg, TL, ids_cal, name, a):
    """Returns (student layers, bits info, timings)."""
    mode = dict(rtn="rtn", **{"gptq1-t": "gptq", "gptq1-s": "gptq", "fold-ns": "fold", "fold-local": "fold",
                              "fold-block": "fold", "fp-ridge": "ridge", "fp-ridge-mut": "ridge"})[name]
    teacher_in = name == "gptq1-t"
    refit = name in ("fold-local", "fold-block")
    block = name in ("fold-block",)
    tm = dict(gram=0.0, solve=0.0, gptq=0.0, refit=0.0, fwd=0.0)
    nsign = nr = nc = ntot = 0
    Ls = []
    if mode == "rtn":
        for l in range(cfg.nl):
            L = dict(TL[l])
            if l < a.layers:
                for k in KINDS:
                    W = TL[l][k].to(F64)
                    O, n = W.shape
                    A = W.view(O, n // GROUP, GROUP).abs().mean(-1).clamp_min(1e-12)
                    S = 1.0 - 2.0 * (W < 0).to(F64)
                    L[k] = (S * f16r(A).repeat_interleave(GROUP, 1)).float()
                    nsign += O * n; nr += O * (n // GROUP) * 16; ntot += O * n
            Ls.append(L)
        return Ls, dict(bits=nsign + nr, params=ntot), tm
    N = ids_cal.shape[0]
    ht = cfg.E[ids_cal]
    hs = ht.clone()
    for l in range(cfg.nl):
        Lt = TL[l]
        Lq = dict(Lt)
        if l >= a.layers:
            Ls.append(Lq)
            continue
        for sname, names, inkey in STAGES:
            n = Lt[names[0]].shape[1]
            Hm = torch.zeros(n, n, dtype=F64, device=DEV)
            Cs = {nm: torch.zeros(n, Lt[nm].shape[0], dtype=F64, device=DEV) for nm in names}
            t0 = tick()
            for s0 in range(0, N, a.slab):
                hts = ht[s0:s0 + a.slab]
                Pt = parts(cfg, Lt, hts)
                if teacher_in:
                    Pin, hss = Pt, hts
                else:
                    hss = hs[s0:s0 + a.slab]
                    Pin = parts(cfg, Lq, hss)
                X = Pin[inkey].reshape(-1, n)
                Hm += (X.T @ X).to(F64)
                if mode in ("fold", "ridge"):
                    for nm in names:
                        if nm in ("q", "k", "v"):
                            Y = Pt[nm] - Lt[nm + "b"]
                        elif nm == "o":
                            Y = Pt["mid"] - (hss if block else hts)
                        elif nm == "gate":
                            Y = Pt["g"]
                        elif nm == "up":
                            Y = Pt["u"]
                        else:
                            Y = Pt["out"] - (Pin["mid"] if block else Pt["mid"])
                        Y = Y.reshape(-1, Y.shape[-1])
                        if name == "fp-ridge-mut":
                            Y = Y[torch.randperm(Y.shape[0], device=DEV)]
                        Cs[nm] += (X.T @ Y).to(F64)
            tm["gram"] += tick() - t0
            t0 = tick()
            if mode in ("fold", "ridge"):
                Ws = ridge_solve(Hm, torch.cat([Cs[nm] for nm in names], 1), a.ridge)
            else:
                Ws = torch.cat([Lt[nm].to(F64) for nm in names], 0)
            tm["solve"] += tick() - t0
            t0 = tick()
            O = Ws.shape[0]
            if mode == "ridge":
                Wp, info = Ws, None
            elif mode == "gptq":
                _, S_, A_ = gptq_signs(Ws, Hm)
                Wp = compose(S_, f16r(A_), torch.ones(n, dtype=F64, device=DEV))
                nsign += O * n; nr += 16 * O * (n // GROUP); ntot += O * n
            else:
                Wp, info = fold_fit(Ws, Hm, a.iters, a.ridge, refit=refit)
                nsign += O * n; nr += 16 * O * (n // GROUP); ntot += O * n
                if refit:
                    nc += 16 * n
                if a.verbose:
                    print(f"  L{l} {sname} J={['%.4g' % x for x in info['J']]} best={info['tag']}", flush=True)
            if mode == "ridge":
                ntot += O * n
            tm["gptq" if mode == "gptq" else "refit"] += tick() - t0
            o0 = 0
            for nm in names:
                Oi = Lt[nm].shape[0]
                Lq[nm] = Wp[o0:o0 + Oi].float().contiguous()
                o0 += Oi
        t0 = tick()
        for s0 in range(0, N, a.slab):
            ht[s0:s0 + a.slab] = parts(cfg, Lt, ht[s0:s0 + a.slab])["out"]
            hs[s0:s0 + a.slab] = parts(cfg, Lq, hs[s0:s0 + a.slab])["out"]
        tm["fwd"] += tick() - t0
        Ls.append(Lq)
        if a.verbose:
            print(f"  layer {l} done", flush=True)
    return Ls, dict(bits=nsign + nr + nc, params=ntot, bits_sign=nsign, bits_r=nr, bits_c=nc), tm


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def kl_eval(cfg, TL, Ls, ids, bs=2):
    kls, agree, n = 0.0, 0, 0
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs]
        lt = F.log_softmax(logits_of(cfg, TL, x).float(), -1)
        ls = F.log_softmax(logits_of(cfg, Ls, x).float(), -1)
        kls += float((lt.exp() * (lt - ls)).sum(-1).sum())
        agree += int((lt.argmax(-1) == ls.argmax(-1)).sum())
        n += x.numel()
        del lt, ls
    return kls / n, agree / n


# ---------------------------------------------------------------- controls
def controls(cfg, TL, hf, ids_all):
    out = {}
    g = torch.Generator(device="cpu").manual_seed(1)
    O, n = 5, 256
    Wn = torch.randn(O, n, generator=g, dtype=F64)
    X = torch.randn(600, n, generator=g, dtype=F64) * (1 + torch.rand(n, generator=g, dtype=F64) * 3)
    Hn = X.T @ X
    ref = gptq_naive_np(Wn.numpy(), Hn.numpy())
    got, _, _ = gptq_signs(Wn.to(DEV), Hn.to(DEV))
    out["c1_gptq_vec_vs_naive_numpy_maxabs"] = float(np.abs(got.cpu().numpy() - ref).max())
    got_i, _, _ = gptq_signs(Wn.to(DEV), torch.eye(n, dtype=F64, device=DEV))
    A = Wn.view(O, n // GROUP, GROUP).abs().mean(-1)
    rtn = (1.0 - 2.0 * (Wn < 0).to(F64)) * A.repeat_interleave(GROUP, 1)
    out["c2_identity_gram_equals_rtn_maxabs"] = float((got_i.cpu() - rtn).abs().max())
    U = feedback_u(Hn.to(DEV))
    h = Hn.to(DEV).clone(); h.diagonal().add_(0.01 * Hn.diagonal().mean())
    out["c3_UtU_h_minus_I_maxabs"] = float((U.T @ U @ h - torch.eye(n, dtype=F64, device=DEV)).abs().max())
    # c4: exact steps are stationary minimisers and the discrete step is the only non-monotone one
    Ws, Hd = Wn.to(DEV).repeat(40, 1), Hn.to(DEV)
    Hl = metric(Hd, 1e-3)
    _, S, r = gptq_signs(Ws, Hd)
    c0 = torch.ones(n, dtype=F64, device=DEV)
    j0 = J(Ws, compose(S, r, c0), Hl)
    c1 = c_step(Ws, Hl, S, r)
    j1 = J(Ws, compose(S, r, c1), Hl)
    bad = 0
    for _ in range(20):
        d = 1e-3 * torch.randn(n, dtype=F64, device=DEV)
        bad += J(Ws, compose(S, r, c1 + d), Hl) < j1 - 1e-9 * abs(j1)
    r1 = r_step(Ws, Hl, S, c1)
    j2 = J(Ws, compose(S, r1, c1), Hl)
    bad_r = 0
    for _ in range(20):
        d = 1e-3 * torch.randn_like(r1)
        bad_r += J(Ws, compose(S, r1 + d, c1), Hl) < j2 - 1e-9 * abs(j2)
    out["c4_J_gptq_cstep_rstep"] = [j0, j1, j2]
    out["c4_perturbations_below_minimum_c_r"] = [int(bad), int(bad_r)]
    # c5: fold solve recovers a linear map exactly, and is unique (ridge -> 0)
    Wt = torch.randn(7, n, generator=g, dtype=F64).to(DEV)
    Xd = X.to(DEV)
    Wr = ridge_solve(Xd.T @ Xd, Xd.T @ (Xd @ Wt.T), 1e-12)
    out["c5_fold_solve_recovery_maxabs"] = float((Wr - Wt).abs().max())
    # c6: teacher forward vs HF
    x = ids_all[100:102]
    with torch.no_grad():
        lm = F.log_softmax(logits_of(cfg, TL, x).float(), -1)
        lh = F.log_softmax(hf(x).logits.float(), -1)
    out["c6_forward_vs_hf_maxabs_logp"] = float((lm - lh).abs().max())
    out["c6_forward_vs_hf_KL"] = float((lh.exp() * (lh - lm)).sum(-1).mean())
    # c7: Gram vs the file used by the earlier arms (blk.5 ffn_down input)
    Hf = torch.from_numpy(np.fromfile(f"{GRAM_DIR}/blk.5.ffn_down_in.f64", dtype="<f8").reshape(cfg.inter, cfg.inter)).to(DEV)
    ids_cal = ids_all[:100]
    L5 = TL[5]
    Hm = torch.zeros(cfg.inter, cfg.inter, dtype=F64, device=DEV)
    ht = cfg.E[ids_cal]
    with torch.no_grad():
        for l in range(6):
            for s0 in range(0, 100, 10):
                P = parts(cfg, TL[l], ht[s0:s0 + 10])
                if l == 5:
                    Xm = P["m"].reshape(-1, cfg.inter)
                    Hm += (Xm.T @ Xm).to(F64)
                ht[s0:s0 + 10] = P["out"]
    out["c7_gram_blk5_down_rel_frob_vs_file"] = float((Hm - Hf).norm() / Hf.norm())
    out["c7_gram_trace_ratio"] = float(Hm.trace() / Hf.trace())
    # c8: rtn variant vs independent numpy for one tensor
    W = TL[3]["o"].cpu().numpy().astype(np.float64)
    ref = np.where(W < 0, -1.0, 1.0) * np.abs(W).reshape(W.shape[0], -1, GROUP).mean(-1).astype(np.float16).astype(np.float64).repeat(GROUP, 1)
    A2 = torch.from_numpy(W).view(W.shape[0], -1, GROUP).abs().mean(-1).clamp_min(1e-12)
    mine = ((1.0 - 2.0 * (torch.from_numpy(W) < 0).to(F64)) * f16r(A2).repeat_interleave(GROUP, 1)).numpy()
    out["c8_rtn_vs_numpy_maxabs"] = float(np.abs(mine - ref).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-cal", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-in", type=int, default=20)
    ap.add_argument("--slab", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    t00 = tick()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ids = tok.encode(open(CALIB, encoding="utf-8").read())
    nch = len(ids) // CHUNK
    assert nch >= a.n_cal + a.n_eval, (nch, a.n_cal, a.n_eval)
    ids_all = torch.tensor(ids[:nch * CHUNK], device=DEV).view(nch, CHUNK)
    cfg, TL, hf = load_model()
    hf = hf.to(DEV).eval()
    res = dict(script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), args=vars(a),
               n_tokens=len(ids), n_chunks=nch, cal_chunks=[0, a.n_cal], eval_chunks=[a.n_cal, a.n_cal + a.n_eval],
               insample_chunks=[0, a.n_in], gpu=torch.cuda.get_device_name(0))
    ids_cal, ids_ev, ids_in = ids_all[:a.n_cal].contiguous(), ids_all[a.n_cal:a.n_cal + a.n_eval], ids_all[:a.n_in]
    if a.controls:
        res["controls"] = controls(cfg, TL, hf, ids_all)
        print(json.dumps(res["controls"], indent=1), flush=True)
    del hf
    torch.cuda.empty_cache()
    res["variants"] = {}
    n_lin = sum(TL[l][k].numel() for l in range(cfg.nl) for k in KINDS)
    res["linear_params"] = n_lin
    for name in [v for v in a.variants.split(",") if v]:
        t0 = tick()
        torch.cuda.reset_peak_memory_stats()
        if name in ("q4km", "gguf-f16"):
            Ls, bpw = gguf_layers(cfg, TL, GGUF_Q4KM if name == "q4km" else GGUF_F16)
            info, tm = dict(bpw_linears=bpw), {}
        else:
            Ls, binfo, tm = run_variant(cfg, TL, ids_cal, name, a)
            info = dict(binfo)
            if binfo["params"]:
                info["bpw_linears"] = binfo["bits"] / binfo["params"]
        tq = tick() - t0
        kl, ag = kl_eval(cfg, TL, Ls, ids_ev)
        kli, agi = kl_eval(cfg, TL, Ls, ids_in)
        info.update(kl_heldout=kl, top1_heldout=ag, kl_insample=kli, top1_insample=agi, wall_transfer_s=tq,
                    timings=tm, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30)
        res["variants"][name] = info
        print(name, json.dumps(info), flush=True)
        del Ls
        torch.cuda.empty_cache()
    res["wall_total_s"] = tick() - t00
    json.dump(res, open(a.out, "w"), indent=1)
    print("wall_total_s", res["wall_total_s"])


if __name__ == "__main__":
    main()
