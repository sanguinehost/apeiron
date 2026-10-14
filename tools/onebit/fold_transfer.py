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
variants: teacher rtn gptq1-t gptq1-s fold-ns fold-local fold-block fp-ridge fp-ridge-mut q4km gguf-f16
Run with ~/spinor-ttl/.venv/bin/python (torch + CUDA).
"""
import argparse, contextlib, hashlib, json, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers

sys.path.insert(0, "~/llama.cpp/gguf-py")
import gguf
from gguf.quants import dequantize

DEV = "cuda"
F64 = torch.float64
CALIB = "~/seal/calib-c4.txt"
# C4 shard 00001 (disjoint from CALIB), Qwen-tokenised by kl_distill.train_ids; --cal-extra appends its first chunks
CAL_EXTRA_NPY = "~/lanes/w18/kd/c4data/train-c4.qwen.npy"
GGUF_F16 = "~/models/gguf/qwen05b-f16.gguf"
GGUF_Q4KM = "~/seal/qwen05b-Q4_K_M-imat.gguf"
MODEL = "Qwen/Qwen2.5-0.5B"  # --model; GGUF_* above are 0.5B files, replaced by --q4km for another model
GRAM_DIR = "~/gptq-w2/gram"
CHUNK, GROUP = 512, 128
# Pack sink (pv_finish.py): None leaves run_variant unchanged.  A dict receives, per linear (layer, kind), the planes whose
# sum is the stored weight: [dict(S_rot, r, c, d, rgrid=(lo, span), rows)] (+ dict(dense=Corr, U, V) for an accepted corrector, Corr = U @ V the stored factors),
# all on CPU; rows is None or the row indices a plane is added to.  PACK_STATS counts the per-linear reconstruction checks.
PACK_SINK = None
PACK_STATS = dict(equal=0, tol=0, max_rel=0.0)
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

def attn_a(cfg, L, xa, qkv=None):
    B, T, _ = xa.shape
    q, k, v = qkv if qkv is not None else (xa @ L["q"].T + L["qb"], xa @ L["k"].T + L["kb"], xa @ L["v"].T + L["vb"])
    qh, kh, vh = q.view(B, T, cfg.nh, cfg.hd).transpose(1, 2), k.view(B, T, cfg.nkv, cfg.hd).transpose(1, 2), v.view(B, T, cfg.nkv, cfg.hd).transpose(1, 2)
    cs, sn = cfg.cos[:T], cfg.sin[:T]
    qh, kh = qh * cs + rot(qh) * sn, kh * cs + rot(kh) * sn
    rep = cfg.nh // cfg.nkv
    kh, vh = kh.repeat_interleave(rep, 1), vh.repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True).transpose(1, 2).reshape(B, T, cfg.nh * cfg.hd)

def parts(cfg, L, h):
    xa = rms(h, L["ln1"], cfg.eps)
    q, k, v = xa @ L["q"].T + L["qb"], xa @ L["k"].T + L["kb"], xa @ L["v"].T + L["vb"]
    a = attn_a(cfg, L, xa, (q, k, v))
    mid = h + a @ L["o"].T
    xf = rms(mid, L["ln2"], cfg.eps)
    g, u = xf @ L["gate"].T, xf @ L["up"].T
    m = F.silu(g) * u
    return dict(xa=xa, q=q, k=k, v=v, a=a, mid=mid, xf=xf, g=g, u=u, m=m, out=mid + m @ L["down"].T)

def logits_of(cfg, Ls, ids):
    h = cfg.E[ids.cpu()].to(DEV)
    for L in Ls:
        Ld = {k: v.to(DEV) for k, v in L.items() if k != "fnorm"}
        h = parts(cfg, Ld, h)["out"]
        del Ld
    fn = Ls[-1].get("fnorm", cfg.fnorm).to(DEV) if Ls else cfg.fnorm.to(DEV)
    return rms(h, fn, cfg.eps).cpu() @ cfg.E.T



def load_model():
    cfg = Cfg()
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    c = m.config
    cfg.nl, cfg.nh, cfg.nkv, cfg.hid, cfg.inter = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.hidden_size, c.intermediate_size
    cfg.hd, cfg.eps, cfg.theta = c.hidden_size // c.num_attention_heads, c.rms_norm_eps, float(c.rope_parameters["rope_theta"])
    cfg.cos, cfg.sin = rope_tables(cfg)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
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
            L[k] = torch.from_numpy(np.ascontiguousarray(a.reshape(O, n))).float().cpu()
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


def gptq_signs(W, H, group=GROUP, damp=0.01, act_order=False, wscale=False):
    """onebit.rs gptq_row, rows batched, columns in lazy blocks of `group` (same arithmetic as the
    per-column Rust loop up to reassociation). When act_order=True, orders 128-groups by descending trace and
    within-group columns by descending H_jj (CostCorrespondence.act_order_curvature_dominance), restoring native
    order at 0 bit overhead. When wscale=True, applies exact whitened Cholesky scale projection
    (CostCorrespondence.whitened_scale_shift_identity + corrector_attained) and 2-pass intra-block refinement
    before inter-block diffusion. Returns (a*sign, sign, a per (row, group))."""
    O, n = W.shape
    if act_order:
        G = n // group
        d = H.diagonal().view(G, group)
        blk_perm = torch.argsort(d.sum(1), descending=True)
        inv_blk = torch.argsort(blk_perm)
        in_perm = torch.argsort(d[blk_perm], dim=1, descending=True)
        perm = (blk_perm[:, None] * group + in_perm).reshape(-1)
        inv_perm = torch.argsort(perm)
        Wq_p, S_p, A_p = gptq_signs(W[:, perm], H[perm][:, perm], group=group, damp=damp, act_order=False, wscale=wscale)
        return Wq_p[:, inv_perm], S_p[:, inv_perm], A_p[:, inv_blk]
    U = feedback_u(H, damp)
    wk = W.clone()
    Wq, S = torch.empty_like(W), torch.empty_like(W)
    A = torch.empty(O, n // group, dtype=W.dtype, device=W.device)
    for gi, g0 in enumerate(range(0, n, group)):
        g1 = g0 + group
        U_blk = U[g0:g1, g0:g1]
        if wscale:
            w0_blk = wk[:, g0:g1].clone()
            w_prec = 1.0 / (U_blk.diagonal() ** 2).clamp_min(1e-12)
            a = ((w0_blk.abs() * w_prec[None, :]).sum(1) / w_prec.sum()).clamp_min(1e-12)
        else:
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
        if wscale:
            S1 = S[:, g0:g1].clone()
            u_S1 = torch.linalg.solve_triangular(U_blk, S1, upper=True, left=False)
            da1 = (Err * u_S1).sum(1) / (u_S1 * u_S1).sum(1).clamp_min(1e-12)
            a1 = (a + da1).clamp_min(1e-12)
            Err1 = Err - (a1 - a)[:, None] * u_S1
            # Pass 2: refine intra-block signs using a1, then re-project whitened scale
            wk2 = w0_blk.clone()
            S2 = torch.empty(O, group, dtype=W.dtype, device=W.device)
            Err2_raw = torch.empty(O, group, dtype=W.dtype, device=W.device)
            for j in range(group):
                wv = wk2[:, j]
                s = 1.0 - 2.0 * (wv < 0).to(W.dtype)
                q = a1 * s
                err = (wv - q) / U_blk[j, j]
                if j + 1 < group:
                    wk2[:, j + 1:] -= err[:, None] * U_blk[j, j + 1:]
                Err2_raw[:, j] = err
                S2[:, j] = s
            u_S2 = torch.linalg.solve_triangular(U_blk, S2, upper=True, left=False)
            da2 = (Err2_raw * u_S2).sum(1) / (u_S2 * u_S2).sum(1).clamp_min(1e-12)
            a2 = (a1 + da2).clamp_min(1e-12)
            Err2 = Err2_raw - (a2 - a1)[:, None] * u_S2
            better2 = (Err2 * Err2).sum(1) < (Err1 * Err1).sum(1)
            a_best = torch.where(better2, a2, a1)
            S_best = torch.where(better2[:, None], S2, S1)
            Err = torch.where(better2[:, None], Err2, Err1)
            A[:, gi] = a_best
            S[:, g0:g1] = S_best
            Wq[:, g0:g1] = a_best[:, None] * S_best
        if g1 < n:
            wk[:, g1:] -= Err @ U[g0:g1, g1:]
    return Wq, S, A


_E8_CACHE = {}
def e8_o8_codebook(dtype, device):
    """256-entry 8-D codebook C in R^{256 x 8} (1 byte per 8 weights = 1.0000 b/w, 0 bits overhead):
    128 even-parity half-cube spinors S_8^+ (norm sqrt(8)), 112 two-sparse D_8 roots ({+-2, +-2, 0^6}, norm sqrt(8)),
    and 16 one-sparse O_8 cross-polytope vertices ({+-sqrt(8), 0^7}, norm sqrt(8))."""
    key = (dtype, str(device))
    if key not in _E8_CACHE:
        idx = torch.arange(256, device=device)[:, None]
        bits = (idx >> torch.arange(7, -1, -1, device=device)[None, :]) & 1
        signs8 = 1.0 - 2.0 * bits.to(dtype)
        s128 = signs8[bits.sum(-1) % 2 == 0]
        d8 = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in (-2.0, 2.0):
                    for sj in (-2.0, 2.0):
                        v = [0.0] * 8; v[i], v[j] = si, sj; d8.append(v)
        d112 = torch.tensor(d8, dtype=dtype, device=device)
        s8 = 8.0 ** 0.5
        o8 = []
        for i in range(8):
            for s in (-s8, s8):
                v = [0.0] * 8; v[i] = s; o8.append(v)
        o16 = torch.tensor(o8, dtype=dtype, device=device)
        _E8_CACHE[key] = torch.cat([s128, d112, o16], dim=0).contiguous()
    return _E8_CACHE[key]


_GPTQ_H_CACHE = {}
def gptq_e8(W, H, group=GROUP, damp=0.01, act_order=False, wscale=False, beam=1):
    """8-D Block-Cholesky exact CVP (and Beam-K delayed-decision search for beam > 1) over E8 U O8 (§15-16)."""
    O, n = W.shape
    C = e8_o8_codebook(W.dtype, W.device)
    hkey = (n, float(H[0, 0].item()), float(H[-1, -1].item()), float(H.diagonal().sum().item()), damp, act_order)
    if hkey not in _GPTQ_H_CACHE:
        if len(_GPTQ_H_CACHE) > 8:
            _GPTQ_H_CACHE.clear()
        if act_order:
            G, sub = n // group, group // 8
            d = H.diagonal().view(G, sub, 8)
            blk_perm = torch.argsort(d.sum(dim=(1, 2)), descending=True)
            inv_blk = torch.argsort(blk_perm)
            sub_perm = torch.argsort(d[blk_perm].sum(dim=2), dim=1, descending=True)
            perm = (blk_perm[:, None, None] * group + sub_perm[:, :, None] * 8 + torch.arange(8, device=W.device)[None, None, :]).reshape(-1)
            inv_perm, Hp = torch.argsort(perm), H[perm][:, perm]
        else:
            perm = inv_perm = inv_blk = None
            Hp = H
        U = feedback_u(Hp, damp)
        zc_list = [torch.linalg.solve_triangular(U[b0:b0+8, b0:b0+8], C, upper=True, left=False) for b0 in range(0, n, 8)]
        _GPTQ_H_CACHE[hkey] = (perm, inv_perm, inv_blk, U, zc_list, [(zc * zc).sum(-1) for zc in zc_list])
    perm, inv_perm, inv_blk, U, zc_list, zcn_list = _GPTQ_H_CACHE[hkey]
    if act_order:
        W = W[:, perm]
    wk = W.clone()
    Wq, S = torch.empty_like(W), torch.empty_like(W)
    A = torch.empty(O, n // group, dtype=W.dtype, device=W.device)
    row_1d = torch.arange(O, device=W.device)
    row_idx = row_1d[:, None]
    for gi, g0 in enumerate(range(0, n, group)):
        g1 = g0 + group
        U_blk = U[g0:g1, g0:g1]
        w0_blk = wk[:, g0:g1].clone()
        w_prec = 1.0 / (U_blk.diagonal() ** 2).clamp_min(1e-12)
        w_rms = (((w0_blk ** 2) * w_prec[None, :]).sum(1) / w_prec.sum()).clamp_min(1e-24).sqrt()
        a = (0.828 * w_rms).clamp_min(1e-12)
        for pass_id in range(2 if wscale else 1):
            if beam <= 1:
                wk_cur = w0_blk.clone()
                S_cur = torch.empty(O, group, dtype=W.dtype, device=W.device)
                Err_cur = torch.empty(O, group, dtype=W.dtype, device=W.device)
                for b0 in range(0, group, 8):
                    b1 = b0 + 8
                    m = (g0 + b0) // 8
                    U_sub = U_blk[b0:b1, b0:b1]
                    w_sub = wk_cur[:, b0:b1]
                    Z_w = torch.linalg.solve_triangular(U_sub, w_sub, upper=True, left=False)
                    Z_c = zc_list[m]
                    zc_norm2 = zcn_list[m][None, :]
                    dots = Z_w @ Z_c.T
                    cost = (a[:, None] ** 2) * zc_norm2 - 2.0 * a[:, None] * dots
                    idx = cost.argmin(dim=-1)
                    s_sel = C[idx]
                    err_sub = Z_w - a[:, None] * Z_c[idx]
                    if b1 < group:
                        wk_cur[:, b1:] -= err_sub @ U_blk[b0:b1, b1:]
                    S_cur[:, b0:b1] = s_sel
                    Err_cur[:, b0:b1] = err_sub
                if wscale:
                    u_S = torch.linalg.solve_triangular(U_blk, S_cur, upper=True, left=False)
                    da = (Err_cur * u_S).sum(1) / (u_S * u_S).sum(1).clamp_min(1e-12)
                    a_new = (a + da).clamp_min(1e-12)
                    Err_new = Err_cur - (a_new - a)[:, None] * u_S
                else:
                    a_new, Err_new = a, Err_cur
            else:
                B = 1
                wk_b = w0_blk[:, None, :].clone()
                S_b = torch.empty(O, 1, group, dtype=W.dtype, device=W.device)
                Err_b = torch.empty(O, 1, group, dtype=W.dtype, device=W.device)
                cum_err2 = torch.zeros(O, 1, dtype=W.dtype, device=W.device)
                for b0 in range(0, group, 8):
                    b1 = b0 + 8
                    m = (g0 + b0) // 8
                    U_sub = U_blk[b0:b1, b0:b1]
                    w_sub = wk_b[:, :, b0:b1]
                    Z_w = torch.linalg.solve_triangular(U_sub, w_sub.reshape(-1, 8), upper=True, left=False).view(O, B, 8)
                    Z_c = zc_list[m]
                    zc_norm2 = zcn_list[m][None, None, :]
                    zw_norm2 = (Z_w * Z_w).sum(-1, keepdim=True)
                    dots = torch.einsum("obk,ck->obc", Z_w, Z_c)
                    step_err2 = zw_norm2 + (a[:, None, None] ** 2) * zc_norm2 - 2.0 * a[:, None, None] * dots
                    tot_err2 = cum_err2[:, :, None] + step_err2
                    K = min(beam, B * 256)
                    best_vals, flat_idx = torch.topk(tot_err2.view(O, -1), K, dim=-1, largest=False)
                    b_parent = flat_idx // 256
                    c_idx = flat_idx % 256
                    wk_b = wk_b[row_idx, b_parent].clone()
                    S_b = S_b[row_idx, b_parent].clone()
                    Err_b = Err_b[row_idx, b_parent].clone()
                    z_w_sel = Z_w[row_idx, b_parent]
                    s_sel = C[c_idx]
                    err_sub = z_w_sel - a[:, None, None] * Z_c[c_idx]
                    if b1 < group:
                        wk_b[:, :, b1:] -= torch.einsum("okj,jm->okm", err_sub, U_blk[b0:b1, b1:])
                    S_b[:, :, b0:b1] = s_sel
                    Err_b[:, :, b0:b1] = err_sub
                    cum_err2 = best_vals
                    B = K
                if wscale:
                    u_S = torch.linalg.solve_triangular(U_blk, S_b.reshape(-1, group), upper=True, left=False).view(O, B, group)
                    da = (Err_b * u_S).sum(-1) / (u_S * u_S).sum(-1).clamp_min(1e-12)
                    a_cand = (a[:, None] + da).clamp_min(1e-12)
                    Err_cand = Err_b - (a_cand - a[:, None])[:, :, None] * u_S
                    cand_err2 = (Err_cand * Err_cand).sum(-1)
                    win_b = cand_err2.argmin(dim=-1)
                    a_new = a_cand[row_1d, win_b]
                    S_cur = S_b[row_1d, win_b]
                    Err_new = Err_cand[row_1d, win_b]
                else:
                    a_new, S_cur, Err_new = a, S_b[:, 0], Err_b[:, 0]
            if wscale:
                if pass_id == 0:
                    a1, S1, Err1 = a_new, S_cur, Err_new
                    a = a1
                else:
                    better2 = (Err_new * Err_new).sum(1) < (Err1 * Err1).sum(1)
                    a_best = torch.where(better2, a_new, a1)
                    S_best = torch.where(better2[:, None], S_cur, S1)
                    Err_best = torch.where(better2[:, None], Err_new, Err1)
            else:
                a_best, S_best, Err_best = a_new, S_cur, Err_new
        A[:, gi] = a_best
        S[:, g0:g1] = S_best
        Wq[:, g0:g1] = a_best[:, None] * S_best
        if g1 < n:
            wk[:, g1:] -= Err_best @ U[g0:g1, g1:]
    if act_order:
        return Wq[:, inv_perm], S[:, inv_perm], A[:, inv_blk]
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
    Hr = H.clone(); Hr.diagonal().add_(ridge * H.diagonal().mean())
    return torch.cholesky_solve(C, torch.linalg.cholesky(Hr)).T.contiguous()

def metric(H, ridge):
    Hl = H.clone(); Hl.diagonal().add_(ridge * H.diagonal().mean())
    return Hl

def J(Ws, Wp, Hl):
    D = Ws - Wp
    return float(((D @ Hl) * D).sum())

def compose(S, r, c):
    return (S * r.repeat_interleave(GROUP, 1)) * c[None, :]

def fwht128(x):
    s, y, h = x.shape, x.reshape(-1, GROUP), 1
    while h < GROUP:
        y = y.view(-1, GROUP // (2 * h), 2, h)
        y = torch.stack((y[:, :, 0, :] + y[:, :, 1, :], y[:, :, 0, :] - y[:, :, 1, :]), dim=2)
        h *= 2
    return (y.reshape(s) * (GROUP ** -0.5)).contiguous()

def had_gram(H):
    return fwht128(fwht128(H).T).T


def c_step(Ws, Hl, S, r, g_row=None):
    Sh = S * r.repeat_interleave(GROUP, 1)
    Sh_g = Sh * g_row[:, None] if g_row is not None else Sh
    A = Hl * (Sh.T @ Sh_g)
    b = ((Ws @ Hl) * Sh_g).sum(0)
    eps = 1e-6 if Ws.dtype == torch.float32 else 1e-12
    A.diagonal().add_(eps * A.diagonal().mean())
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
    eps = 1e-6 if Ws.dtype == torch.float32 else 1e-12
    M = M + eps * M.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * eye
    return torch.linalg.solve(M, t[:, :, None])[:, :, 0]


def d_step(Ws, Hl, S_rot, r, c, g_row=None):
    """Exact Hadamard-domain butterfly column scale minimizer (CostCorrespondence.butterfly_d_step_dominance)."""
    B = S_rot * r.repeat_interleave(GROUP, 1)
    B_g = B * g_row[:, None] if g_row is not None else B
    G_rot = fwht128((Ws @ Hl) * c[None, :])
    H_rot = had_gram(Hl * c[:, None] * c[None, :])
    A = H_rot * (B.T @ B_g)
    b = (B_g * G_rot).sum(0)
    eps = 1e-6 if Ws.dtype == torch.float32 else 1e-12
    A.diagonal().add_(eps * A.diagonal().mean().clamp_min(1e-12))
    return torch.cholesky_solve(b[:, None], torch.linalg.cholesky(A))[:, 0]


def drift_filter(W0, Ws_tik, Hl, rank=64, gamma_tail=0.65):
    """Reduced-rank + shrinkage filter on cross-layer drift DeltaW = Ws_tik - W0 in metric Hl
    (CostCorrespondence.reduced_rank_drift_generalization_dominance)."""
    dW = Ws_tik - W0
    O, n = dW.shape
    rk = min(rank, O, n)
    if O <= n:
        M = (dW @ Hl) @ dW.T
        U = torch.linalg.eigh(0.5 * (M + M.T))[1][:, -rk:]
        dW_top = U @ (U.T @ dW)
    else:
        Lc = torch.linalg.cholesky(Hl)
        Z = dW @ Lc
        evals, V = torch.linalg.eigh(0.5 * (Z.T @ Z + (Z.T @ Z).T))
        V_r = V[:, -rk:]
        U = (Z @ V_r) / evals[-rk:].clamp_min(1e-12).sqrt()[None, :]
        dW_top = U @ (U.T @ dW)
    return W0 + gamma_tail * dW + (1.0 - gamma_tail) * dW_top


def f16r(x):
    return x.half().to(x.dtype)


def quant_c_u8(c, bits=8):
    """Affine column scale quantization: 2 fp16 scalars (32 bits) + bits/col."""
    c_min, c_max = f16r(c.min()), f16r(c.max())
    span = (c_max - c_min).clamp_min(1e-12)
    levels = float((1 << bits) - 1)
    q = torch.round((c - c_min) / span * levels).clamp(0.0, levels)
    return c_min + (q / levels) * span, bits * c.numel() + 32


def quant_r_hier(r):
    """Hierarchical group scale quantization (CostCorrespondence.full_attn_2plane_hier_scale_budget)."""
    O, G = r.shape
    rm = r.mean(1)
    mu_min, mu_max = f16r(rm.min()), f16r(rm.max())
    mu_span = (mu_max - mu_min).clamp_min(1e-12)
    u6_base = torch.round((rm - mu_min) / mu_span * 63.0).clamp(0.0, 63.0)
    dev = (r - rm[:, None]).abs().amax(1) / rm.abs().clamp_min(1e-12)
    sp_min = f16r(dev.quantile(0.05).clamp_min(1e-4))
    sp_max = f16r(dev.max().clamp_min(sp_min + 1e-4))
    sp_levels = sp_min + torch.linspace(0.0, 1.0, 4, dtype=r.dtype, device=r.device) * (sp_max - sp_min)
    best_err = torch.full((O,), float("inf"), dtype=r.dtype, device=r.device)
    best_r = r.clone()
    track = PACK_SINK is not None
    if track:
        best_lo, best_sp = torch.zeros_like(rm), torch.ones_like(rm)
    for du in (-1.0, 0.0, 1.0):
        u6 = (u6_base + du).clamp(0.0, 63.0)
        mu = mu_min + (u6 / 63.0) * mu_span
        for s_idx in range(4):
            sp = sp_levels[s_idx]
            r_lo, r_hi = mu * (1.0 - sp), mu * (1.0 + sp)
            r_sp = (r_hi - r_lo).clamp_min(1e-12)
            k = torch.round((r - r_lo[:, None]) / r_sp[:, None] * 15.0).clamp(0.0, 15.0)
            rq = r_lo[:, None] + (k / 15.0) * r_sp[:, None]
            err = ((r - rq) ** 2).sum(1)
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_r = torch.where(mask[:, None], rq, best_r)
            if track:
                best_lo, best_sp = torch.where(mask, r_lo, best_lo), torch.where(mask, r_sp, best_sp)
    if track:
        best_r.grid = (best_lo, best_sp)  # per-row chosen (r_lo, r_sp): rq = r_lo + k / 15 * r_sp, k in 0..15
    return best_r, 4 * O * G + 8 * O + 64


def lowrank_corrector(DeltaW, Hl, rank, round_f16=True, u8_corr=False, factors=False):
    """Exact rank-k corrector in metric Hl (CostCorrespondence.orthogonal_corrector_attained).  factors=True appends the
    stored factors (U, V) with Corr = U @ V, so a consumer can train the store itself rather than a re-factoring."""
    O, n = DeltaW.shape
    M = (DeltaW @ Hl) @ DeltaW.T
    evals, evecs = torch.linalg.eigh(0.5 * (M + M.T))
    U = evecs[:, -rank:]
    top_sum = float(evals[-rank:].clamp_min(0.0).sum())
    if u8_corr:
        Uh = torch.stack([quant_c_u8(U[:, k], bits=8)[0] for k in range(rank)], dim=1)
        eye = torch.eye(rank, dtype=DeltaW.dtype, device=DeltaW.device) * 1e-6
        V = torch.linalg.solve(Uh.T @ Uh + eye, Uh.T @ DeltaW)
        Vh = torch.stack([quant_c_u8(V[k, :], bits=8)[0] for k in range(rank)], dim=0)
        out = (Uh @ Vh, rank * (8 * (O + n) + 64), top_sum)
        return out + ((Uh, Vh),) if factors else out
    if round_f16:
        Uh = f16r(U)
        eye = torch.eye(rank, dtype=DeltaW.dtype, device=DeltaW.device) * 1e-6
        Vh = f16r(torch.linalg.solve(Uh.T @ Uh + eye, Uh.T @ DeltaW))
    else:
        Uh, Vh = U, U.T @ DeltaW
    out = (Uh @ Vh, 16 * rank * (O + n), top_sum)
    return out + ((Uh, Vh),) if factors else out


def e8_icm_step(Ws, Hl, S_rot, r, c, d, sweeps=2, icm_ridge=0.0):
    """Global Elliptic 8-tuple ICM coordinate descent over E8 U O8 in the (r,c,d)-scaled Hadamard-Gram metric (§17)."""
    O, n = Ws.shape
    if icm_ridge > 0.0:
        Hl = Hl.clone()
        Hl.diagonal().add_((icm_ridge * (n / 896.0)) * Hl.diagonal().mean())
    C = e8_o8_codebook(Ws.dtype, Ws.device)
    G_rot = fwht128((Ws @ Hl) * c[None, :]) * d[None, :]
    H_rot = had_gram(Hl * c[:, None] * c[None, :]) * d[:, None] * d[None, :]
    S_cur = S_rot.clone()
    B = S_cur * r.repeat_interleave(GROUP, 1)
    F_res = G_rot - B @ H_rot
    nb = n // 8
    H_blk = H_rot.view(nb, 8, nb, 8).diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()
    Q_all = ((C[None, :, :] @ H_blk) * C[None, :, :]).sum(-1)
    for sw in range(sweeps):
        order = range(nb - 1, -1, -1) if sw % 2 == 0 else range(nb)
        for m in order:
            b0, b1 = m * 8, (m + 1) * 8
            a = r[:, b0 // GROUP]
            s_old = S_cur[:, b0:b1]
            f_m = F_res[:, b0:b1] + (a[:, None] * s_old) @ H_blk[m]
            cost = (a[:, None] ** 2) * Q_all[m][None, :] - 2.0 * a[:, None] * (f_m @ C.T)
            s_new = C[cost.argmin(dim=-1)]
            diff = s_new - s_old
            S_cur[:, b0:b1] = s_new
            F_res.sub_((a[:, None] * diff) @ H_rot[b0:b1, :])
    return S_cur


def _plane_w(P):
    S_rot, r, c, d = P
    return compose(fwht128(S_rot * d[None, :]), r, c)


def _j_rows(Ws, W, Hl, g_row=None):
    D = Ws - W
    j = ((D @ Hl) * D).sum(1)
    return float((j * g_row).sum()) if g_row is not None else float(j.sum())


def e8_joint_icm_2plane(Ws, Hl, P1, P2, sweeps=2, icm_ridge=0.0, g_row=None, row_chunk=256, blocks=None):
    """Joint block-ICM over the 256x256 pair set E8 U O8 x E8 U O8 per 8-block of two stacked planes (T3, W16 E8 lift).
    P_k = (S_rot, r, c, d): plane k weight = fwht128(S_rot*d) * r_rep * c.  The pair (i1, i2) of codebook indices of block m
    minimises, per row, cost = a1^2 Q1[i1] + a2^2 Q2[i2] + 2 a1 a2 X_m[i1, i2] - 2 a1 u[i1] - 2 a2 v[i2] in the coupled metric
    H_kl = had_gram(Hl*c_k c_l^T)*d_k d_l^T (X_m = C H12[m,m] C^T).  Returns (P1', P2', dict(J_before, J_after, moves)); J in the
    icm-ridged metric, row-weighted by g_row when given.  Scales are not touched.
    Complexity: temporal  sweeps * (n/8) * O * 65536 additions to build the cost tensor and as many comparisons in the argmin
    (+ 2*O*256*8 MACs for u, v and O*8*n MACs for the residual update per block); the per-sweep count is the recurrence
    sum over the n/8 blocks, each block closing in one pass over the 65536 pairs of every row (rows are independent given the
    residual field).  Setup: 3 had_gram of n x n (3 n^2 log 128 flops) plus X: (n/8) * 256*256*8 MACs.
    Spatial  row_chunk * 65536 * dtype bytes for the cost tensor (64 MB at 256 rows, fp32) + (n/8)*65536*bytes for X
    (29 MB at n=896) + 4 n^2 for H11, H22, H12, H21."""
    O, n = Ws.shape
    if icm_ridge > 0.0:
        Hl = Hl.clone()
        Hl.diagonal().add_((icm_ridge * (n / 896.0)) * Hl.diagonal().mean())
    S1, r1, c1, d1 = P1[0].clone(), P1[1], P1[2], P1[3]
    S2, r2, c2, d2 = P2[0].clone(), P2[1], P2[2], P2[3]
    j_before = _j_rows(Ws, _plane_w((S1, r1, c1, d1)) + _plane_w((S2, r2, c2, d2)), Hl, g_row)
    C = e8_o8_codebook(Ws.dtype, Ws.device)
    nb = n // 8
    WH = Ws @ Hl
    G1, G2 = fwht128(WH * c1[None, :]) * d1[None, :], fwht128(WH * c2[None, :]) * d2[None, :]
    H11 = had_gram(Hl * c1[:, None] * c1[None, :]) * d1[:, None] * d1[None, :]
    H22 = had_gram(Hl * c2[:, None] * c2[None, :]) * d2[:, None] * d2[None, :]
    H12 = had_gram(Hl * c1[:, None] * c2[None, :]) * d1[:, None] * d2[None, :]
    H21 = H12.T.contiguous()
    B1, B2 = S1 * r1.repeat_interleave(GROUP, 1), S2 * r2.repeat_interleave(GROUP, 1)
    F1 = G1 - B1 @ H11 - B2 @ H21
    F2 = G2 - B2 @ H22 - B1 @ H12

    def dblk(Hm_):
        return Hm_.view(nb, 8, nb, 8).diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()
    Hb11, Hb22, Hb12 = dblk(H11), dblk(H22), dblk(H12)
    Hb21 = Hb12.transpose(1, 2).contiguous()
    Q1 = ((C[None, :, :] @ Hb11) * C[None, :, :]).sum(-1)
    Q2 = ((C[None, :, :] @ Hb22) * C[None, :, :]).sum(-1)
    X = (C[None, :, :] @ Hb12) @ C.T[None, :, :]
    moves = 0
    for sw in range(sweeps):
        order = list(range(nb - 1, -1, -1) if sw % 2 == 0 else range(nb)) if blocks is None else list(blocks)
        for m in order:
            b0, b1 = m * 8, (m + 1) * 8
            a1, a2 = r1[:, b0 // GROUP], r2[:, b0 // GROUP]
            s1o, s2o = S1[:, b0:b1], S2[:, b0:b1]
            f1 = F1[:, b0:b1] + (a1[:, None] * s1o) @ Hb11[m] + (a2[:, None] * s2o) @ Hb21[m]
            f2 = F2[:, b0:b1] + (a2[:, None] * s2o) @ Hb22[m] + (a1[:, None] * s1o) @ Hb12[m]
            A = (a1[:, None] ** 2) * Q1[m][None, :] - 2.0 * a1[:, None] * (f1 @ C.T)
            Bv = (a2[:, None] ** 2) * Q2[m][None, :] - 2.0 * a2[:, None] * (f2 @ C.T)
            idx = torch.empty(O, dtype=torch.long, device=Ws.device)
            for o0 in range(0, O, row_chunk):
                o1 = min(O, o0 + row_chunk)
                t = (2.0 * a1[o0:o1] * a2[o0:o1])[:, None, None] * X[m][None, :, :]
                t.add_(A[o0:o1, :, None])
                t.add_(Bv[o0:o1, None, :])
                idx[o0:o1] = t.view(o1 - o0, -1).argmin(dim=-1)
                del t
            s1n, s2n = C[idx // 256], C[idx % 256]
            e1, e2 = a1[:, None] * (s1n - s1o), a2[:, None] * (s2n - s2o)
            moves += int(((e1 != 0).any(1) | (e2 != 0).any(1)).sum())
            S1[:, b0:b1], S2[:, b0:b1] = s1n, s2n
            F1.sub_(e1 @ H11[b0:b1, :] + e2 @ H21[b0:b1, :])
            F2.sub_(e2 @ H22[b0:b1, :] + e1 @ H12[b0:b1, :])
    j_after = _j_rows(Ws, _plane_w((S1, r1, c1, d1)) + _plane_w((S2, r2, c2, d2)), Hl, g_row)
    return (S1, r1, c1, d1), (S2, r2, c2, d2), dict(J_before=j_before, J_after=j_after, moves=moves)


def r_step2(Ws, Hl, S1, c1, S2, c2):
    """Joint least squares for both planes' group scales: min over (r1, r2) in R^{O x G} x R^{O x G} of J(Ws, compose(S1,r1,c1)+compose(S2,r2,c2), Hl).
    Per row a (2G x 2G) normal system M x = t, M[(k,g),(l,g')] = P_k[g]^T Hl[g,g'] P_l[g'], P_k = S_k*c_k.
    Complexity: temporal 2G Hl-row-panel products of O x 128 by 128 x n (2G*O*128*n MACs) + O batched 2G x 2G solves (O*(2G)^3/3);
    spatial O*(2G)^2 for M, O*n for P (G = n/128)."""
    O, n = Ws.shape
    G = n // GROUP
    P = torch.cat([(S1 * c1[None, :]).view(O, G, GROUP), (S2 * c2[None, :]).view(O, G, GROUP)], dim=1)
    M = torch.empty(O, 2 * G, 2 * G, dtype=Ws.dtype, device=Ws.device)
    for a_ in range(2 * G):
        ga = a_ % G
        K = (P[:, a_, :] @ Hl[ga * GROUP:(ga + 1) * GROUP, :]).view(O, G, GROUP)
        M[:, a_, :] = (torch.cat([K, K], dim=1) * P).sum(-1)
    WH = (Ws @ Hl).view(O, G, GROUP)
    t = (P * torch.cat([WH, WH], dim=1)).sum(-1)
    eye = torch.eye(2 * G, dtype=Ws.dtype, device=Ws.device)
    eps = 1e-6 if Ws.dtype == torch.float32 else 1e-12
    M = M + eps * M.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * eye
    x = torch.linalg.solve(M, t[:, :, None])[:, :, 0]
    return x[:, :G].contiguous(), x[:, G:].contiguous()


NB_SWEEPS = int(os.environ.get("ONEBIT_NB_SWEEPS", "2"))


def nb_refine(Ws, Hl, P1, P2, g_row=None, icm_ridge=0.20, sweeps=None, rounds=2):
    """Reduced joint block-ICM post-pass on a finished 2-plane pair (T3).  <= `rounds` rounds of: joint sweeps on stored
    (r, c, d); r_step2 + quant_r_hier; c_step per plane against Ws minus the other plane + quant_c_u8(., 6); keep a round iff
    J(Ws, W1+W2, Hl) falls, else revert.  Bits unchanged (same quantizers).  Returns (W1, W2, log, P1', P2') with P*' the
    final (S_rot, r, c, d) packs.
    Complexity: temporal rounds * (e8_joint_icm_2plane + r_step2 + 2 c_step + 2 quant) + 1 + rounds evaluations of J (O*n^2 MACs each);
    spatial as e8_joint_icm_2plane plus O*n per plane."""
    sweeps = NB_SWEEPS if sweeps is None else sweeps
    W1, W2 = _plane_w(P1), _plane_w(P2)
    j_seq = J(Ws, W1 + W2, Hl)
    j_cur = j_seq
    log = dict(J_seq=j_seq, rounds_kept=0, moves=0, icm=[])
    cur = (P1, P2, W1, W2)
    for _ in range(rounds):
        Pa, Pb = cur[0], cur[1]
        Pa_n, Pb_n, st = e8_joint_icm_2plane(Ws, Hl, Pa, Pb, sweeps=sweeps, icm_ridge=icm_ridge, g_row=g_row)
        log["moves"] += st["moves"]
        log["icm"].append(st)
        if st["moves"] == 0:
            break
        S1e, S2e = fwht128(Pa_n[0] * Pa_n[3][None, :]), fwht128(Pb_n[0] * Pb_n[3][None, :])
        r1, r2 = r_step2(Ws, Hl, S1e, Pa_n[2], S2e, Pb_n[2])
        rh1, _ = quant_r_hier(r1)
        rh2, _ = quant_r_hier(r2)
        c1n, _ = quant_c_u8(c_step(Ws - compose(S2e, rh2, Pb_n[2]), Hl, S1e, rh1, g_row=g_row), bits=6)
        W1n = compose(S1e, rh1, c1n)
        c2n, _ = quant_c_u8(c_step(Ws - W1n, Hl, S2e, rh2, g_row=g_row), bits=6)
        W2n = compose(S2e, rh2, c2n)
        j_new = J(Ws, W1n + W2n, Hl)
        if not (j_new < j_cur):
            break
        cur = ((Pa_n[0], rh1, c1n, Pa_n[3]), (Pb_n[0], rh2, c2n, Pb_n[3]), W1n, W2n)
        j_cur = j_new
        log["rounds_kept"] += 1
    log["J_nb"] = j_cur
    return cur[2], cur[3], log, cur[0], cur[1]


def _yaqa_JO(HO, Hl, D):
    """J_O(D) = tr(D^T HO D Hl) = sum((HO @ D @ Hl) * D).  Temporal O*O*n + O*n*n MACs, spatial O*n."""
    return float(((HO @ D @ Hl) * D).sum())


def _yaqa_dJ(HO, Hl, D, rows, delta):
    """Exact J_O(D with rows replaced by D[rows] + delta) - J_O(D), no linearisation:
    2 <HO[rows,:] D Hl, delta> + <HO[rows,rows] delta Hl, delta>.  Temporal |rows|*O*n + |rows|*n*n + |rows|^2*n MACs."""
    first = 2.0 * (((HO[rows, :] @ D) @ Hl) * delta).sum()
    sec = (((HO[rows][:, rows] @ delta) @ Hl) * delta).sum()
    return float(first + sec)


# rows per Gauss-Seidel block of yaqa_refine; recorded in binfo["yaqa_chunk"].  Each ICM call loops over the n/8 E8
# groups regardless of its row count, so the refine wall scales with O/chunk calls: chunk 32 is 8x the calls of 256.
YAQA_CHUNK = 256


def yaqa_refine(Ws, Hl, HO, P1, P2=None, W_fixed=None, icm_ridge=0.0, chunk=YAQA_CHUNK, sweeps=2, rounds=1, shift_sign=1.0):
    """Output-side (YAQA) code refinement of finished planes: minimise J_O(D) = tr(D^T HO D Hl) = sum((HO @ D @ Hl) * D),
    D = Ws - (_plane_w(P1) + [_plane_w(P2)] + [W_fixed]), by Gauss-Seidel over row chunks of the E8/O8 codes only (scales
    r, c, d and the bit count are untouched).  Plane P = (S_rot, r, c, d) as in _plane_w; W_fixed is a dense additive
    plane (a top-rows partial second plane).  Tseng et al., arXiv 2505.22988 Eq. 5-6: the output-side feedback term.
    Block derivation: with the rows outside C fixed, J_O = tr((D_C + E_C)^T HO_CC (D_C + E_C) Hl) + const for
    E_C = HO_CC^-1 HO[C,notC] D_notC, so the continuous optimum of the block is D_C = -E_C and the ICM pass targets
    Ws_C + E_C (minus W_fixed_C).  For |C| = 1 this is the per-row identity E_r = HO[r,notr] D_notr / HO[r,r]; the
    per-row (block-Jacobi) shift at |C| > 1 counts the coupling HO[C,C] of rows that all move at once and overshoots
    (selftest: two planes, chunk 16, block-Jacobi accepts nothing).  The ICM rounds each row of C independently, so the
    rounding errors' coupling inside C is left to the exact second-order dJ of _yaqa_dJ, which decides acceptance;
    the update is applied to S_rot only (rejected chunks keep their rows).
    Complexity: temporal per round O*O*n + O*n*n MACs for the shifts and the dJ (summed over the O/chunk chunks), plus
    O*chunk^2 for the block solves (O/chunk solves of chunk^3) and one ICM pass over all rows; spatial O*O for HO,
    O*n for D (and chunk*n transient per chunk).
    Returns (P1', P2', log) with log = dict(JO_before, JO_after, chunks, accepted, rows_moved); P2' is None for one plane."""
    O, n = Ws.shape
    HO = HO.to(Ws.dtype).to(Ws.device)
    Hl = Hl.to(Ws.dtype).to(Ws.device)
    two = P2 is not None
    S1, r1, c1, d1 = P1[0].clone(), P1[1], P1[2], P1[3]
    if two:
        S2, r2, c2, d2 = P2[0].clone(), P2[1], P2[2], P2[3]
    W = _plane_w((S1, r1, c1, d1))
    if two:
        W = W + _plane_w((S2, r2, c2, d2))
    if W_fixed is not None:
        W = W + W_fixed
    D = Ws - W
    j0 = _yaqa_JO(HO, Hl, D)
    log = dict(JO_before=j0, JO_after=j0, chunks=0, accepted=0, rows_moved=0)
    for rnd in range(int(rounds)):
        cuts = list(range(0, O, int(chunk)))
        if rnd % 2 == 1:
            cuts = cuts[::-1]        # Gauss-Seidel order reversed on odd rounds
        for c0 in cuts:
            Cs = slice(c0, min(O, c0 + int(chunk)))
            log["chunks"] += 1
            HCC = HO[Cs, Cs]
            E_C = shift_sign * torch.linalg.solve(HCC, HO[Cs, :] @ D - HCC @ D[Cs])
            T_C = Ws[Cs] + E_C
            if W_fixed is not None:
                T_C = T_C - W_fixed[Cs]
            if two:
                P1n, P2n, _st = e8_joint_icm_2plane(T_C, Hl, (S1[Cs], r1[Cs], c1, d1), (S2[Cs], r2[Cs], c2, d2),
                                                    sweeps=sweeps, icm_ridge=icm_ridge)
                W_new = _plane_w(P1n) + _plane_w(P2n)
            else:
                P1n = (e8_icm_step(T_C, Hl, S1[Cs], r1[Cs], c1, d1, sweeps=sweeps, icm_ridge=icm_ridge), r1[Cs], c1, d1)
                W_new = _plane_w(P1n)
            if W_fixed is not None:
                W_new = W_new + W_fixed[Cs]
            delta = (Ws[Cs] - W_new) - D[Cs]
            if _yaqa_dJ(HO, Hl, D, Cs, delta) < 0.0:
                S1[Cs] = P1n[0]
                if two:
                    S2[Cs] = P2n[0]
                D[Cs] += delta
                log["accepted"] += 1
                log["rows_moved"] += int((delta.abs().amax(-1) > 0).sum())
            del E_C, T_C, delta, HCC
    log["JO_after"] = _yaqa_JO(HO, Hl, D)
    return (S1, r1, c1, d1), ((S2, r2, c2, d2) if two else None), log


def fold_fit(Ws, H, iters, ridge, refit=True, had=False, damp=0.01, inner_iters=1, act_order=False, wscale=False, hier_scale=False, bscale=False, cgauge=False, bgptq=False, hpre=False, e8_code=False, e8_beam=1, e8_icm=False, icm_ridge=0.0, g_row=None):
    """Alternation of the discrete step (gptq1 / had-gptq1 / had-e8) and the exact continuous steps. Returns (W', info)."""
    n = H.shape[0]
    Hl = metric(H, ridge)
    c = torch.ones(n, dtype=Ws.dtype, device=Ws.device)
    d = torch.ones(n, dtype=Ws.dtype, device=Ws.device)
    q_fn = (lambda W_, H_, **kw: gptq_e8(W_, H_, beam=e8_beam, **kw)) if e8_code else gptq_signs
    if had:
        if hpre:
            w_rms = ((Ws ** 2) * g_row[:, None]).mean(0).clamp_min(1e-12).sqrt() if g_row is not None else (Ws ** 2).mean(0).clamp_min(1e-12).sqrt()
            h_rms = Hl.diagonal().clamp_min(1e-12).sqrt()
            c = (w_rms / h_rms).sqrt()
            c = (c / c.mean().clamp_min(1e-12)).clamp(0.2, 5.0)
            _, S_rot, r = q_fn(fwht128(Ws / c[None, :]), had_gram(H * c[:, None] * c[None, :]), damp=damp, act_order=act_order, wscale=wscale)
        else:
            _, S_rot, r = q_fn(fwht128(Ws), had_gram(H), damp=damp, act_order=act_order, wscale=wscale)
        S = fwht128(S_rot)
        Wq = compose(S, r, c)
    else:
        Wq, S, r = q_fn(Ws, H, damp=damp, act_order=act_order, wscale=wscale)
        S_rot = S
    cands = [(J(Ws, Wq, Hl), "gptq", S_rot, S, r, c, d)]
    trace = [cands[0][0]]
    if refit:
        for it in range(iters):
            for _ in range(inner_iters):
                c = c_step(Ws, Hl, S, r, g_row=g_row)
                s = c.abs().mean()
                c, r = c / s, r * s
                r = r_step(Ws, Hl, S, c)
                if bscale and had:
                    d = d_step(Ws, Hl, S_rot, r, c, g_row=g_row)
                    sd_sign = torch.where(d < 0, -1.0, 1.0).to(Ws.dtype)
                    if e8_code:
                        sd_8 = sd_sign.view(-1, 8).clone()
                        odd_m = (sd_8 < 0).sum(-1) % 2 == 1
                        if odd_m.any():
                            min_i = d.abs().view(-1, 8).argmin(-1)
                            sd_8[odd_m, min_i[odd_m]] *= -1.0
                        sd_sign = sd_8.reshape(-1)
                    S_rot = S_rot * sd_sign[None, :]
                    d_abs = (d * sd_sign).view(-1, GROUP)
                    m_g = d_abs.abs().amax(1).clamp_min(1e-12)
                    d = (d_abs / m_g[:, None]).reshape(-1)
                    r = r * m_g[None, :]
                    if e8_icm and e8_code:
                        S_rot = e8_icm_step(Ws, Hl, S_rot, r, c, d, sweeps=2, icm_ridge=icm_ridge)
                    S = fwht128(S_rot * d[None, :])
                    r = r_step(Ws, Hl, S, c)
            cands.append((J(Ws, compose(S, r, c), Hl), f"refit{it}", S_rot, S, r, c, d))
            trace.append(cands[-1][0])
            if it < iters - 1:
                cs = torch.where(c.abs() < 0.05, 0.05 * (1.0 - 2.0 * (c < 0).to(Ws.dtype)), c)
                Wc = Ws / cs[None, :]
                Hc = H * cs[:, None] * cs[None, :]
                if had:
                    _, S_rot_1, r_1 = q_fn(fwht128(Wc), had_gram(Hc), damp=damp, act_order=act_order, wscale=wscale)
                    S_1 = fwht128(S_rot_1)
                    if (cgauge or bgptq) and bscale:
                        ds = d.clamp_min(0.1)
                        _, S_rot_d, r_d = q_fn(fwht128(Wc) / ds[None, :], had_gram(Hc) * ds[:, None] * ds[None, :], damp=damp, act_order=act_order, wscale=wscale)
                        S_d = fwht128(S_rot_d * ds[None, :])
                        if J(Ws, compose(S_d, r_d, c), Hl) < J(Ws, compose(S_1, r_1, c), Hl):
                            S_rot, r, S = S_rot_d, r_d, S_d
                        else:
                            S_rot, r, d, S = S_rot_1, r_1, torch.ones_like(c), S_1
                    else:
                        S_rot, r, d, S = S_rot_1, r_1, torch.ones_like(c), S_1
                else:
                    _, S, r = q_fn(Wc, Hc, damp=damp, act_order=act_order, wscale=wscale)
                    S_rot = S
    best = min(cands, key=lambda x: x[0])
    _, tag, S_rot, S, r, c, d = best
    if not refit:
        return compose(S, f16r(r), torch.ones_like(c)), dict(tag=tag, J=trace, J_best=best[0])
    b_d = 0
    pack = None
    if cgauge and bscale and had:
        dh, b_d, b_c = torch.round(d.clamp(0.0, 1.0) * 15.0) / 15.0, 4 * n, 4 * n
        Wp, best_j = None, float("inf")
        for S_c in (fwht128(S_rot * dh[None, :]), fwht128(S_rot)):
            r0 = r_step(Ws, Hl, S_c, c)
            c0 = c_step(Ws, Hl, S_c, r0, g_row=g_row).clamp_min(0.0)
            mcg = c0.view(-1, GROUP).amax(1).clamp_min(1e-12)
            ch0 = torch.round((c0.view(-1, GROUP) / mcg[:, None]).reshape(-1).clamp(0.0, 1.0) * 15.0) / 15.0
            rh1, b_r = quant_r_hier(r_step(Ws, Hl, S_c, ch0.clamp_min(1e-4)))
            rh2, _ = quant_r_hier(r0 * mcg[None, :])
            c2 = c_step(Ws, Hl, S_c, rh2, g_row=g_row).clamp(0.0, 1.0)
            ch2 = torch.round(c2 * 15.0) / 15.0
            for W_cand in (compose(S_c, rh1, ch0), compose(S_c, rh2, ch0), compose(S_c, rh2, ch2)):
                jc = J(Ws, W_cand, Hl)
                if jc < best_j:
                    Wp, best_j = W_cand, jc
    elif bscale and had:
        dh, b_d = quant_c_u8(d, bits=5)
        S_dh = fwht128(S_rot * dh[None, :])
        S_rot_dh = S_rot
        r_dh = r_step(Ws, Hl, S_dh, c)
        rh, b_r = quant_r_hier(r_dh)
        c_ref = c_step(Ws, Hl, S_dh, rh, g_row=g_row)
        ch, b_c = quant_c_u8(c_ref, bits=6)
        Wp = compose(S_dh, rh, ch)
        pack = (S_rot_dh, rh, ch, dh)
        if e8_icm and e8_code:
            S_rot_q = e8_icm_step(Ws, Hl, S_rot, rh, ch, dh, sweeps=2, icm_ridge=icm_ridge)
            S_dh_q = fwht128(S_rot_q * dh[None, :])
            rh_q, _ = quant_r_hier(r_step(Ws, Hl, S_dh_q, ch))
            ch_q, _ = quant_c_u8(c_step(Ws, Hl, S_dh_q, rh_q, g_row=g_row), bits=6)
            Wp_q = compose(S_dh_q, rh_q, ch_q)
            if J(Ws, Wp_q, Hl) < J(Ws, Wp, Hl):
                Wp, S_rot = Wp_q, S_rot_q
                pack = (S_rot_q, rh_q, ch_q, dh)
        if hpre:
            mcg = c_ref.abs().view(-1, GROUP).amax(1).clamp_min(1e-12)
            rh_g, _ = quant_r_hier(r_dh * mcg[None, :])
            c_g = c_step(Ws, Hl, S_dh, rh_g, g_row=g_row)
            ch_g, _ = quant_c_u8(c_g, bits=6)
            Wp_g = compose(S_dh, rh_g, ch_g)
            if J(Ws, Wp_g, Hl) < J(Ws, Wp, Hl):
                Wp = Wp_g
                pack = (S_rot_dh, rh_g, ch_g, dh)
        S_1 = fwht128(S_rot)
        r_1 = r_step(Ws, Hl, S_1, c)
        rh_1, _ = quant_r_hier(r_1)
        ch_1, _ = quant_c_u8(c_step(Ws, Hl, S_1, rh_1, g_row=g_row), bits=6)
        Wp_1 = compose(S_1, rh_1, ch_1)
        if J(Ws, Wp_1, Hl) < J(Ws, Wp, Hl):
            Wp = Wp_1
            pack = (S_rot, rh_1, ch_1, torch.ones_like(dh))
        assert torch.equal(compose(fwht128(pack[0] * pack[3][None, :]), pack[1], pack[2]), Wp), "winning pack != Wp"
    elif hier_scale:
        rh, b_r = quant_r_hier(r)
        ch, b_c = quant_c_u8(c_step(Ws, Hl, S, rh, g_row=g_row), bits=8)
        Wp = compose(S, rh, ch)
    else:
        rh, ch = f16r(r), f16r(c)
        b_r, b_c = 16 * r.numel(), 16 * c.numel()
        Wp = compose(S, rh, ch)
    return Wp, dict(tag=tag, J=trace, J_best=best[0], J_f16=J(Ws, Wp, Hl), bits_r=b_r, bits_c=b_c, bits_d=b_d, S_rot=S_rot, pack=pack)


def fnorm_fit(cfg, ht, hs, steps=4, ridge=0.05, lr=1.0, coh_prior=False):
    """Exact damped Newton-Fisher descent on final RMSNorm g in R^896 minimizing readout KL (CostCorrespondence §14, §20)."""
    g0 = cfg.fnorm.to(DEV).float()
    g = g0.clone()
    E_g = cfg.E.to(DEV).float()
    E2_g = E_g ** 2
    N = ht.shape[0]
    sl = slice(1, None) if coh_prior else slice(None)
    n_tok = float(N * (ht.shape[1] - (1 if coh_prior else 0)))
    if coh_prior:
        num = torch.zeros(cfg.hid, device=DEV)
        den_t = torch.zeros(cfg.hid, device=DEV)
        den_s = torch.zeros(cfg.hid, device=DEV)
        for s0 in range(0, N, 5):
            ht_b = ht[s0:s0 + 5, sl].to(DEV).float()
            hs_b = hs[s0:s0 + 5, sl].to(DEV).float()
            ut = (ht_b * torch.rsqrt(ht_b.pow(2).mean(-1, keepdim=True) + cfg.eps)).reshape(-1, cfg.hid)
            us = (hs_b * torch.rsqrt(hs_b.pow(2).mean(-1, keepdim=True) + cfg.eps)).reshape(-1, cfg.hid)
            num += (ut * us).sum(0)
            den_t += (ut ** 2).sum(0)
            den_s += (us ** 2).sum(0)
        rho = (num / (den_t * den_s).clamp_min(1e-12).sqrt()).clamp(0.0, 1.0)
        alpha = (den_s / den_t.clamp_min(1e-12)).sqrt()
        g_target = g0 * (rho / alpha.clamp_min(0.5))
    else:
        g_target = g0
    for _ in range(steps):
        grad = torch.zeros_like(g)
        fisher = torch.zeros_like(g)
        for s0 in range(0, N, 2):
            ht_b = ht[s0:s0 + 2, sl].to(DEV).float()
            hs_b = hs[s0:s0 + 2, sl].to(DEV).float()
            ut = (ht_b * torch.rsqrt(ht_b.pow(2).mean(-1, keepdim=True) + cfg.eps)).reshape(-1, cfg.hid)
            us = (hs_b * torch.rsqrt(hs_b.pow(2).mean(-1, keepdim=True) + cfg.eps)).reshape(-1, cfg.hid)
            pt = F.softmax((ut * g0[None, :]) @ E_g.T, dim=-1)
            ps = F.softmax((us * g[None, :]) @ E_g.T, dim=-1)
            grad += (us * ((ps - pt) @ E_g)).sum(0)
            mu_e = ps @ E_g
            var_e = (ps @ E2_g - mu_e ** 2).clamp_min(1e-6)
            fisher += ((us ** 2) * var_e).sum(0)
            del ht_b, hs_b, ut, us, pt, ps, mu_e, var_e
        grad /= n_tok
        fisher /= n_tok
        reg = ridge * fisher.mean().clamp_min(1e-6)
        g = f16r(g - lr * (grad + reg * (g - g_target)) / (fisher + reg))
    del E_g, E2_g
    torch.cuda.empty_cache()
    return g.cpu()


# ---------------------------------------------------------------- pack sink
def _sink_plane(pack, rows=None):
    """Plane record of a finished (S_rot, r, c, d) pack; r must come from quant_r_hier under the sink (carries .grid)."""
    if pack is None:
        raise NotImplementedError("pack sink: path produced no (S_rot, r, c, d) pack")
    S_rot, r, c, d = pack
    grid = getattr(r, "grid", None)
    if grid is None:
        raise NotImplementedError("pack sink: r was not produced by quant_r_hier")
    return dict(S_rot=S_rot, r=r, c=c, d=d, rgrid=grid, rows=rows)


def _sink_recon(planes):
    """Stored weight from the planes, summed in run_variant's order.  Temporal O(P * O * n * log 128) flops (P planes, 7 FWHT
    stages each), spatial O(O * n) words."""
    W = None
    for p in planes:
        if "dense" in p:
            W = W + p["dense"]
            continue
        Wi = compose(fwht128(p["S_rot"] * p["d"][None, :]), p["r"], p["c"])
        if W is None:
            W = Wi
        elif p["rows"] is None:
            W = W + Wi
        else:
            W = W.clone()
            W[p["rows"]] += Wi
    return W


def _sink_store(l, nm, planes, Wl):
    """Check sum(planes) == Wl (the weight run_variant stored: torch.equal, else max|diff| <= 1e-6 max|W| and counted in
    PACK_STATS['tol']), then keep the planes on CPU in PACK_SINK[(l, nm)].  Temporal as _sink_recon."""
    if planes is None:
        raise NotImplementedError(f"pack sink: no plane record for L{l} {nm}")
    Wr = _sink_recon(planes).to(Wl.device).float()
    if torch.equal(Wr, Wl):
        PACK_STATS["equal"] += 1
    else:
        rel = float((Wr - Wl).abs().max() / Wl.abs().max())
        assert rel <= 1e-6, f"pack sink: planes != stored weight at L{l} {nm}: max|diff|/max|W| = {rel}"
        PACK_STATS["tol"] += 1
        PACK_STATS["max_rel"] = max(PACK_STATS["max_rel"], rel)
    out = []
    for p in planes:
        if "dense" in p:
            out.append(dict(dense=p["dense"].cpu(), U=p["U"].cpu(), V=p["V"].cpu()))
        else:
            out.append(dict(S_rot=p["S_rot"].cpu(), r=p["r"].cpu(), c=p["c"].cpu(), d=p["d"].cpu(),
                            rgrid=(p["rgrid"][0].cpu(), p["rgrid"][1].cpu()),
                            rows=None if p["rows"] is None else p["rows"].cpu()))
    PACK_SINK[(l, nm)] = out


# ---------------------------------------------------------------- sequential transfer

# ---------------------------------------------------------------- end-loss guidance (variant token "-guided")
# GuidedQuant (Kim et al., arXiv 2505.07004, Sec 3.1 Eq 4, Sec 3.2 Eq 7): the layer-wise least-squares objective is weighted by
# the gradient of the end loss w.r.t. the linear's output z.  A token weight a_stage[l][t] multiplies each token's term in the
# stage Grams; a row weight b[l][nm][j] replaces g_row.  Captured on the calibration chunks only.
GQ_CACHE = {}
GQ_FORCE_ONES = False   # null control: a_t = 1 (Grams equal the base arm's); b is left to GQ_NULL_B
GQ_NULL_B = False       # null control: keep the base arm's g_row (b not applied)
GQ_DEBUG = None         # dict receiving layer-0 Hm / Hm_w per stage (control)
GQ_STAGE_OF = dict(q="qkv", k="qkv", v="qkv", o="o", gate="ffn", up="ffn", down="down")
# The input of each linear whose per-token ||x_t||^2 weights the output-side Hessian HO (YAQA Sketch A): q/k/v: x_a; o: a;
# gate/up: x_f; down: silu(g)*u.
GQ_IN_OF = dict(q="xa", k="xa", v="xa", o="at", gate="xf", up="xf", down="m")
GQ_SUFFIX = "-guided"

def lw_intensity(S4, Q, H, n):
    """Analytic shrinkage intensity of the output-side Hessian sketch toward its diagonal, from the capture's own
    samples.  Target D: diagonal, unequal variances (Schafer and Strimmer, Stat. Appl. Genet. Mol. Biol. 4 (2005)
    art. 32; the Ledoit-Wolf (2004) construction for that target).  H = sum_t s_t s_t^T is the RAW scatter of the
    n samples s (length O), S4 = sum_t ||s_t||^4 and Q = sum_t s_t^4.  With M = H / n:
      S4 - sum(Q) - n * off(M) = sum_{i != j} sum_t (s_ti s_tj)^2 - n * sum_{i != j} M_ij^2
      num = that / (n (n - 1))      = sum_{i != j} Var-hat(mean_t s_ti s_tj)
      lambda = clamp(num / off(M), 0, 1),  off(M) = ||M||_F^2 - sum_i M_ii^2
    Returns (num, lambda); lambda is the per-linear alpha of the convex combination HO <- (1-lambda) HO + lambda
    diag(HO).  Cost: O(O^2) elementwise on H plus O(O) on Q, once per (layer, kind)."""
    S4f = float(S4)
    Qd = Q.to(torch.float64).flatten()
    Hd = H.to(torch.float64)
    n = int(n)
    M = Hd / n
    d = M.diagonal()
    off = float((M * M).sum() - (d * d).sum())
    if off <= 0.0:
        return 0.0, 0.0
    num = (S4f - float(Qd.sum()) - n * off) / (n * (n - 1))
    return num, float(min(1.0, max(0.0, num / off)))

def _block_off(Sk):
    """off(H_b) of ONE block without ever forming the O x O scatter H_b = sum_k S_k^T S_k.  With the block's samples
    stacked per label draw k as S_k (T x O, s = g * sqrt(w)), ||H_b||_F^2 = sum_{k,k'} ||S_k S_k'^T||_F^2 is the
    squared Frobenius norm of the (K T) x (K T) Gram of the stacked samples (each (k, k') pair of T x T Grams is one
    block of it), so one Gram carries all K^2 pairs, and diag(H_b) = sum_k (S_k ** 2).sum(0).  Returns the fp64 scalar
    ||H_b||_F^2 - ||diag(H_b)||^2.  Cost: K^2 T^2 O MACs per linear (K^2 pairs, each T x T Gram T^2 O MACs) plus
    O(K T) for the diagonal; the fp64 sum of the Gram's squares is one O(K^2 T^2) elementwise pass."""
    Sall = torch.cat(list(Sk), 0)                      # (K T, O): every (k, k') T x T Gram is one block of it
    G = Sall @ Sall.T                                  # (K T, K T)
    dg = (Sall.double() ** 2).sum(0)                   # diag(H_b)
    return (G.double() ** 2).sum() - (dg * dg).sum()

def lw_block_intensity(P, B, H, n):
    """Analytic shrinkage intensity of the output-side Hessian sketch toward its diagonal, with the SEQUENCE as the
    sample unit.  Target D: diagonal, unequal variances (Schafer and Strimmer, Stat. Appl. Genet. Mol. Biol. 4 (2005)
    art. 32; the Ledoit-Wolf (2004) construction for that target).  lw_intensity treats the n = K*N*T samples s_t =
    sqrt(w_t) g_t as independent; the tokens of one sequence, and the K label draws of one token, are correlated, so
    that mean's variance is underestimated.  gq_capture runs bs = 1, so one block is one batch b0 = one independent
    draw of ids_cal: B blocks, block b holding n_b samples with scatter H_b = sum_t s_t s_t^T and mean M_b = H_b/n_b,
    M = (sum_b H_b) / (B n_b) = H / n.  With off(X) = ||X||_F^2 - sum_i X_ii^2:
      P = sum_b off(M_b)
      num = (P - B * off(M)) / (B (B - 1))   = sum_{i != j} Var-hat_b(M_b,ij)
      lambda = clamp(num / off(M), 0, 1)
    Returns (num, lambda); lambda is the per-linear alpha of the convex combination HO <- (1-lambda) HO + lambda
    diag(HO).  P is accumulated off(M_b) = off(H_b) / n_b^2 with _block_off, i.e. without the O(O^2) scatter.
    Cost: B * K^2 T^2 O MACs per linear for the Grams (the +K*T*sum O floats of storage live in gq_capture), plus one
    O(O^2) elementwise pass over H.  B < 2 has no between-block variance and returns (0.0, 0.0)."""
    B = int(B)
    n = int(n)
    if B < 2:
        return 0.0, 0.0
    Hd = H.to(torch.float64)
    M = Hd / n
    d = M.diagonal()
    off = float((M * M).sum() - (d * d).sum())
    if off <= 0.0:
        return 0.0, 0.0
    num = (float(P) - B * off) / (B * (B - 1))
    return num, float(min(1.0, max(0.0, num / off)))

def gq_capture(cfg, TL, ids_cal, K=4, seed=0, bs=1, outer=False, blocks=False):
    """True-Fisher output-gradient statistics of the teacher on ids_cal (N x T), K sampled targets per position.
    Returns dict(a=[l][stage] float64 CPU (N,T) mean 1, b=[l][nm] float64 CPU (O,) mean 1, wall_s).
    With outer=True it also accumulates, per layer l and kind nm, HO[l][nm] = sum over tokens t and the K samples of
    w_t * g_t g_t^T, g_t (length O) the sampled-label gradient of that linear's OUTPUT at token t and w_t = ||x_t||^2 the
    squared norm of its input at t.  This is YAQA Sketch A, Eq. 11 (Tseng et al., arXiv 2505.22988, Eq. 5-6: the
    output-side feedback term L_O^T Delta), one round from (H_I)_0 = I.  fp32 accumulation on the GPU, then per (l, nm)
    divided by its mean diagonal (mean diag 1), 0.01 added to the diagonal, moved to CPU fp32, returned under out["ho"].
    Cost: one teacher forward plus K backward passes through the retained graph (data gradients only, weights are
    constants): (2 + 2K) * P * N * T FLOPs with P ~ 0.49e9 (24 layers + tied head), i.e. 10 P N T at K = 4 (~ 5e13 at
    N = 50, T = 512); counted in FLOPs, derived as forward 2P per token + K backward passes of 2P.
     outer=True adds K * N * T * sum_nm O_nm^2 MACs (the accumulation recurrence is one rank-1 update per token and sample)
    and sum_nm O_nm^2 fp32 floats of accumulator (about 4.8 GB over the 24 Qwen2.5-0.5B layers, freed before return);
    per (l, nm) it also accumulates S4 = sum_t ||s_t||^4 in one fp64 scalar and Q = sum_t s_t^4 in one fp64 O-vector
    (O(O+1) extra MACs and 8*(O+1) bytes per linear, i.e. cost O per sample), from which out["ho_lw"][l][nm] = the
    Schafer and Strimmer (2005) intensity of the raw scatter HO_raw toward its diagonal is computed before the
    mean-diagonal normalisation;
     blocks=True (with outer, and bs = 1) additionally keeps the K samples S_k (T x O fp32, s = g * sqrt(w)) of the
    current block per linear (K*T*sum_nm O_nm fp32 floats, ~623 MB per k on Qwen2.5-0.5B, ~2.5 GB at K = 4, staged as
    an exact host copy and freed at the end of each block: a 15.5 GiB card does not hold it beside HO), accumulates
    one fp64 GPU scalar P[l][nm] += off(H_b) / n_b^2 after each K loop
    (B * K^2 T^2 O_nm MACs per linear over the capture, K^2 T^2 O_nm per block), and gives out["ho_lb"][l][nm] = the
    batch-means (sequence-unit) intensity of the same raw scatter, computed before the mean-diagonal normalisation;
    the per-token input norms are L*B*T*4 floats per batch (each layer's own inputs).  With outer=True the capture runs
    under the math SDPA backend (retains B*nh*T^2 attention floats per layer, ~14.7 MB at bs=1, T=512, nh=14) and asserts
    on the first batch that two backward passes give bit-equal gradients, so H_O is a function of (ids_cal, K, seed).
    Space: B*T*(~8e4 floats/layer * 24 retained activations + V logits) = bs=1: ~3 GB peak on GPU; accumulators
    N*T*4*L doubles (a) + sum_l O_l doubles (b) = ~10 MB at N = 50; weights 2.5 GB resident fp32."""
    key = (hashlib.sha256(ids_cal.numpy().tobytes()).hexdigest(), K, seed, bool(outer), bool(blocks))
    if key in GQ_CACHE:
        return GQ_CACHE[key]
    t0 = tick()
    N, T = ids_cal.shape
    assert T == CHUNK
    if blocks:
        assert outer, "the block intensity needs the output-side (outer) capture"
        assert bs == 1, f"the batch-means intensity takes one sequence per block, got bs={bs}"
    nl = cfg.nl
    Ld = [{k: v.to(DEV) for k, v in L.items()} for L in TL]
    E = cfg.E.to(DEV)
    fn = cfg.fnorm.to(DEV)
    kinds = list(KINDS)
    chan = {nm: TL[0][nm].shape[0] for nm in kinds}
    stage_ch = {"qkv": chan["q"] + chan["k"] + chan["v"], "o": chan["o"], "ffn": chan["gate"] + chan["up"], "down": chan["down"]}
    A = [{st: torch.zeros(N, T, dtype=F64) for st in stage_ch} for _ in range(nl)]
    B = [{nm: torch.zeros(TL[l][nm].shape[0], dtype=F64) for nm in kinds} for l in range(nl)]
    HO = [{nm: torch.zeros(chan[nm], chan[nm], dtype=torch.float32, device=DEV) for nm in kinds} for _ in range(nl)] if outer else None
    # Schafer-Strimmer (2005) shrinkage intensities of HO_raw: S4 = sum_t ||s_t||^4 (fp64 scalar) and Q = sum_t s_t^4
    # (fp64 O-vector) per (l, nm), accumulated next to HO; cost O per sample
    S4 = [{nm: torch.zeros((), dtype=F64, device=DEV) for nm in kinds} for _ in range(nl)] if outer else None
    Q4 = [{nm: torch.zeros(chan[nm], dtype=F64, device=DEV) for nm in kinds} for _ in range(nl)] if outer else None
    # batch-means (sequence-unit) intensity: P[l][nm] = sum over blocks of off(H_b), one fp64 scalar per (l, nm)
    PB = [{nm: torch.zeros((), dtype=F64, device=DEV) for nm in kinds} for _ in range(nl)] if (outer and blocks) else None
    gen = torch.Generator(device=DEV).manual_seed(seed)
    max_dlogit = 0.0
    # the fp32 memory-efficient SDPA backward accumulates with atomics (not bit-reproducible); the math backend is
    # explicit matmul + softmax, so the H_O capture is a function of (ids_cal, K, seed)
    with (sdpa_kernel(SDPBackend.MATH) if outer else contextlib.nullcontext()):
        for b0 in range(0, N, bs):
            ids = ids_cal[b0:b0 + bs]
            # the block's per-(l, nm) fp32 samples S_k = g * sqrt(w) for k = 0..K-1, kept for the K^2 Grams and freed
            # at the end of the block; K*T*sum_nm O_nm floats per block
            sbuf = [{nm: [None] * K for nm in kinds} for _ in range(nl)] if (outer and blocks) else None
            h = E[ids.to(DEV)].clone().requires_grad_(True)
            zs = []
            nins = []
            for L in Ld:
                xa = rms(h, L["ln1"], cfg.eps)
                q, k, v = xa @ L["q"].T + L["qb"], xa @ L["k"].T + L["kb"], xa @ L["v"].T + L["vb"]
                at = attn_a(cfg, L, xa, (q, k, v))
                o = at @ L["o"].T
                mid = h + o
                xf = rms(mid, L["ln2"], cfg.eps)
                g, u = xf @ L["gate"].T, xf @ L["up"].T
                mi = F.silu(g) * u
                d = mi @ L["down"].T
                h = mid + d
                zs.append(dict(q=q, k=k, v=v, o=o, gate=g, up=u, down=d))
                if outer:  # Sketch-A token weights w_t = ||x_t||^2 of THIS layer's linear inputs
                    with torch.no_grad():
                        nins.append(dict(xa=(xa * xa).sum(-1), at=(at * at).sum(-1), xf=(xf * xf).sum(-1), m=(mi * mi).sum(-1)))
            logits = rms(h, fn, cfg.eps) @ E.T
            flat = [z[nm] for z in zs for nm in kinds]
            if b0 == 0:  # the capture forward is the teacher forward
                ref_lg = logits_of(cfg, TL, ids[:1])
                max_dlogit = float((logits.detach()[:1].cpu() - ref_lg).abs().max())
                assert max_dlogit < 1e-2 * float(ref_lg.abs().max()), max_dlogit
                del ref_lg
            probs = F.softmax(logits.detach(), -1).view(-1, logits.shape[-1])
            if outer and b0 == 0:  # fail loud on a nondeterministic backward: two passes on the argmax labels are bit-equal
                sa = probs.argmax(-1)
                g1 = torch.autograd.grad(F.cross_entropy(logits.view(-1, logits.shape[-1]), sa, reduction="sum"), flat, retain_graph=True)
                g2 = torch.autograd.grad(F.cross_entropy(logits.view(-1, logits.shape[-1]), sa, reduction="sum"), flat, retain_graph=True)
                det_ok = all(torch.equal(x, y) for x, y in zip(g1, g2))
                assert det_ok, "gq_capture backward is not bit-reproducible"
                del g1, g2, sa
            for kk in range(K):
                s = torch.multinomial(probs, 1, generator=gen)[:, 0]
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), s, reduction="sum")
                grads = torch.autograd.grad(loss, flat, retain_graph=(kk < K - 1))
                for l in range(nl):
                    tok = {st: torch.zeros(ids.shape[0], T, dtype=F64, device=DEV) for st in stage_ch}
                    for ji, nm in enumerate(kinds):
                        g2 = grads[l * len(kinds) + ji].double() ** 2
                        tok[GQ_STAGE_OF[nm]] += g2.sum(-1)
                        B[l][nm] += g2.sum((0, 1)).cpu()
                        if outer:
                            # Sketch A, Eq. 11: HO += sum_t w_t g_t g_t^T with w_t = ||x_t||^2 of this linear's input.
                            g_ = grads[l * len(kinds) + ji].float().reshape(-1, chan[nm])
                            w_ = nins[l][GQ_IN_OF[nm]].reshape(-1).float()
                            HO[l][nm] += (g_ * w_[:, None]).T @ g_
                            if blocks:
                                # the block's samples s = g * sqrt(w) of this draw, staged on the HOST (an exact fp32
                                # copy): the K x (T x O) block buffer is ~2.5 GB at K = 4 on Qwen2.5-0.5B and does not
                                # fit beside HO on a 15.5 GiB card; only the (K T) x O stack of one linear is on the
                                # GPU at a time when its Grams are formed
                                sbuf[l][nm][kk] = (g_ * w_.sqrt()[:, None]).cpu()
                            # s_ti^2 = w_t g_ti^2 in fp64: S4 += ||s_t||^4, Q_i += s_ti^4
                            s2 = g_.double() ** 2 * w_.double()[:, None]
                            S4[l][nm] += (s2.sum(-1) ** 2).sum()
                            Q4[l][nm] += (s2 ** 2).sum(0)
                    for st in stage_ch:
                        A[l][st][b0:b0 + ids.shape[0]] += tok[st].cpu()
                del grads, loss
            if outer and blocks:
                # P += off(M_b) = off(H_b) / n_b^2 per (l, nm), from the block's K^2 T x T Grams; H_b is never formed
                nb = K * T
                for l in range(nl):
                    for nm in kinds:
                        Sk = [s.to(DEV) for s in sbuf[l][nm]]      # one linear's (K T) x O stack, ~80 MB at most
                        PB[l][nm] += _block_off(Sk) / float(nb * nb)
                        del Sk
                sbuf = None
            del zs, flat, logits, probs, h
    for l in range(nl):
        for st, c in stage_ch.items():
            x = A[l][st] / (K * c)
            A[l][st] = x / x.mean()
        for nm in kinds:
            x = B[l][nm] / (K * N * T)
            B[l][nm] = x / x.mean()
    del Ld, E, fn
    ho = None
    ho_lw = None
    ho_lb = None
    if outer:
        ho = []
        ho_lw = []
        ho_lb = [] if blocks else None
        n_samp = K * N * T
        for l in range(nl):
            dl = {}
            dlw = {}
            dlb = {} if blocks else None
            for nm in kinds:
                x = HO[l][nm]
                dlw[nm] = lw_intensity(S4[l][nm], Q4[l][nm], x, n_samp)[1]  # before normalisation
                if blocks:
                    dlb[nm] = lw_block_intensity(PB[l][nm], N, x, n_samp)[1]  # before normalisation
                x = x / x.diagonal().mean().clamp_min(1e-30)   # mean diagonal 1
                x.diagonal().add_(0.01)
                dl[nm] = x.cpu()                                # fp32 on the CPU
            ho.append(dl)
            ho_lw.append(dlw)
            if blocks:
                ho_lb.append(dlb)
        HO = None
        S4 = None
        Q4 = None
        PB = None
    torch.cuda.empty_cache()
    out = dict(a=A, b=B, wall_s=tick() - t0, K=K, seed=seed, max_dlogit=max_dlogit)
    if outer:
        out["ho"] = ho
        out["ho_lw"] = ho_lw
        if blocks:
            out["ho_lb"] = ho_lb
    GQ_CACHE[key] = out
    return out

def gq_stats(cap, layers):
    st = {}
    for l in layers:
        for sname, x in cap["a"][l].items():
            f = x.flatten()
            st[f"L{l}.{sname}"] = dict(max=float(f.max()), p99=float(torch.quantile(f, 0.99)), pos0_mean=float(x[:, 0].mean()),
                                       pos0_argmax_frac=float((x.argmax(1) == 0).double().mean()),
                                       pos0_mass_share=float(x[:, 0].sum() / x.sum()))
    return st

FLAG_PROBE = None  # control: run_variant appends its resolved scalar flags and returns

def run_variant(cfg, TL, ids_cal, name, a):
    """Returns (student layers, bits info, timings)."""
    dt = torch.float32 if a.dtype == "fp32" else torch.float64
    # "-yaqa": output-side (YAQA) code refinement of every finished plane; stripped before the rest of the name, so it
    # may follow a "-guided..." suffix.  Only `yaqa` differs between the base arm and base + "-yaqa".  "-yaqaxv" is the
    # same refinement plus the held-out-half cross-validation diagnostic: `yaqa` is True for both, `yaqa_xv` marks it.
    # "-yaqasNN": the same refinement with the output-side Hessian shrunk toward its diagonal by NN/100 (`yaqa_shrink`).
    # "-yaqalw": the same refinement with the per-linear analytic (Schafer and Strimmer 2005) shrinkage `yaqa_lw`.
    # "-yaqalb": the same refinement with the per-linear block (batch-means, sequence-unit) intensity `yaqa_lb`.
    yaqa_m = re.search(r"-yaqa(xv|s\d\d|lw|lb)?$", name)
    yaqa = yaqa_m is not None
    yaqa_xv = bool(yaqa and yaqa_m.group(1) == "xv")
    yaqa_lw = bool(yaqa and yaqa_m.group(1) == "lw")
    yaqa_lb = bool(yaqa and yaqa_m.group(1) == "lb")
    yaqa_shrink = 0.0
    if yaqa and yaqa_m.group(1) and yaqa_m.group(1)[0] == "s":
        yaqa_shrink = int(yaqa_m.group(1)[1:]) / 100.0
        assert 0.0 < yaqa_shrink <= 1.0, yaqa_shrink
    if yaqa:
        name = name[:yaqa_m.start()]
    # "-guided": every flag below resolves from the base arm's name; only `guided` is new
    # "-guided[a|b]NN": NN tempers the token weights to a_t^(NN/100) (renormalised to mean 1; bare is the paper's a_t);
    # "a" applies the token weights only (row weights stay the base proxy), "b" the row weights only (a_t = 1)
    gq_m = re.search(r"-guided([ab])?(\d+)?$", name)
    guided = gq_m is not None
    gq_gamma, gq_part = 1.0, "ab"
    if guided:
        assert name.count(GQ_SUFFIX) == 1, name
        gq_gamma = int(gq_m.group(2)) / 100 if gq_m.group(2) else 1.0
        gq_part = gq_m.group(1) or "ab"
        name = name[:gq_m.start()]
    if yaqa_xv:
        # a "-guided...-yaqaxv" name would weight half A's H_O by the end-loss gradient, so the halves would not be
        # exchangeable; refused by construction rather than silently captured
        assert not guided, name
    mode = dict(rtn="rtn", **{"gptq1-t": "gptq", "gptq1-s": "gptq", "had-gptq1-t": "gptq", "had-gptq1-s": "gptq",
                              "fold-ns": "fold", "fold-local": "fold", "fold-block": "fold", "had-fold-local": "fold",
                              "had-fold-block": "fold", "had-fold-block-w0": "fold", "fp-ridge": "ridge", "fp-ridge-mut": "ridge"}).get(name, "fold_gqa")
    teacher_in = name in ("gptq1-t", "had-gptq1-t")
    had = name.startswith("had-")
    w0_anchor = name.endswith("-w0") or mode == "fold_gqa"
    icm_names = ("had-fold-2plane-attn-bscale-e8-icm", "had-fold-2plane-attn-bscale-e8-b1-icm",
                 "had-fold-2plane-attn-bscale-e8-icm15", "had-fold-2plane-attn-bscale-e8-icm20",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25", "had-fold-2plane-attn-bscale-e8-icm25-gin25",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-glu", "had-fold-2plane-attn-bscale-e8-icm20-gin3515-glu",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink", "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-mlp2p",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-res",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-align-qmlp",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-asym-qmlp",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-align-asym-qmlp",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-skew-qmlp",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp-nb",
                 "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-resq-qmlp-nb")
    split_intra_inter = name in ("had-fold-2plane-gqa-split", "had-fold-2plane-gqa-split-corr", "had-fold-2plane-attn-rk2-split",
                                 "had-fold-2plane-attn-bscale-split", "had-fold-2plane-attn-bscale-split-bgptq",
                                 "had-fold-2plane-attn-bscale-hpre", "had-fold-2plane-attn-rk2-hpre",
                                 "had-fold-2plane-attn-bscale-fnorm", "had-fold-2plane-attn-rk2-fnorm",
                                 "had-fold-2plane-attn-bscale-e8", "had-fold-2plane-attn-rk2-e8",
                                 "had-fold-2plane-attn-bscale-e8-beam", "had-fold-2plane-attn-bscale-e8-beam8") + icm_names
    attn_names = ("had-fold-2plane-attn-hier", "had-fold-2plane-attn-backfit", "had-fold-2plane-attn-bscale", "had-fold-2plane-attn-rrr",
                  "had-fold-2plane-attn-rk2", "had-fold-2plane-attn-rk2-split", "had-fold-2plane-attn-bscale-split",
                  "had-fold-2plane-attn-bscale-split-bgptq", "had-fold-2plane-attn-bscale-hpre", "had-fold-2plane-attn-rk2-hpre",
                  "had-fold-2plane-attn-bscale-fnorm", "had-fold-2plane-attn-rk2-fnorm",
                  "had-fold-2plane-attn-bscale-e8", "had-fold-2plane-attn-rk2-e8",
                  "had-fold-2plane-attn-bscale-e8-beam", "had-fold-2plane-attn-bscale-e8-beam8") + icm_names
    act_order = name in ("had-fold-2plane-gqa-act", "had-fold-2plane-gqa-qronos", "had-fold-2plane-gqa-wscale", "had-fold-2plane-gqa-zellner") + attn_names
    wscale = name in ("had-fold-2plane-gqa-wscale", "had-fold-2plane-gqa-zellner") + attn_names
    qronos_metric = name in ("had-fold-2plane-gqa-qronos", "had-fold-2plane-gqa-wscale", "had-fold-2plane-gqa-zellner") + attn_names
    zellner = name == "had-fold-2plane-gqa-zellner"
    hier_scale = name in attn_names
    full_attn_2plane = hier_scale and not ("-mlp2p" in name or "-qmlp" in name)
    mlp_2plane_k896 = "-mlp2p" in name
    q_mlp_trans = "-qmlp" in name
    backfit_2plane = name in attn_names and name != "had-fold-2plane-attn-hier"
    bscale = backfit_2plane and name != "had-fold-2plane-attn-backfit"
    cgauge = name in ("had-fold-2plane-attn-rk2", "had-fold-2plane-attn-rk2-split", "had-fold-2plane-attn-rk2-hpre", "had-fold-2plane-attn-rk2-fnorm", "had-fold-2plane-attn-rk2-e8")
    bgptq = name == "had-fold-2plane-attn-bscale-split-bgptq"
    hpre = name in ("had-fold-2plane-attn-bscale-hpre", "had-fold-2plane-attn-rk2-hpre")
    fnorm_align = name in ("had-fold-2plane-attn-bscale-fnorm", "had-fold-2plane-attn-rk2-fnorm")
    sink_gate = "-sink" in name
    sink_fnorm = "-sink-fnorm" in name
    e8_code = name in ("had-fold-2plane-attn-bscale-e8", "had-fold-2plane-attn-rk2-e8", "had-fold-2plane-attn-bscale-e8-beam", "had-fold-2plane-attn-bscale-e8-beam8") + icm_names
    e8_beam = 8 if name == "had-fold-2plane-attn-bscale-e8-beam8" else (4 if (name == "had-fold-2plane-attn-bscale-e8-beam" or (name in icm_names and name != "had-fold-2plane-attn-bscale-e8-b1-icm")) else 1)
    e8_icm = name in icm_names
    nb_arm = name.endswith("-nb")
    nb_stats = {k: dict(J_seq=0.0, J_nb=0.0, moves=0, rounds_kept=0, tensors=0) for k in ("k", "v", "o")}
    yaqa_stats = {nm: dict(JO_before=0.0, JO_after=0.0, accepted=0, chunks=0, rows_moved=0) for nm in KINDS}
    if yaqa_xv:
        for _d in yaqa_stats.values():
            _d.update(JB_before=0.0, JB_after=0.0, JB_after_a50=0.0, JB_after_a90=0.0, accepted_a50=0, accepted_a90=0)
    icm_ridge = 0.25 if "icm25" in name else (0.20 if "icm20" in name else (0.15 if "icm15" in name else 0.0))
    gin_val = 0.25 if ("-gin25" in name) else a.inter_gamma_in
    glu_comp = name.endswith("-glu")
    rrr_mode = name == "had-fold-2plane-attn-rrr"
    use_grow = "-grow" in name
    use_res = "-res" in name
    use_resq = "-resq" in name
    use_align = "-align" in name
    use_asym = "-asym" in name
    use_skew = "-skew" in name
    SKEW_K_ROWS = [1024, 128, 512, 512, 256, 256, 256, 256, 512, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 384, 384, 384, 384, 768]
    use_corr = name in ("had-fold-2plane-gqa-corr", "had-fold-2plane-gqa-wfill", "had-fold-2plane-gqa-split",
                        "had-fold-2plane-gqa-split-corr") or act_order
    corr_wfill = name in ("had-fold-2plane-gqa-wfill", "had-fold-2plane-gqa-split-corr", "had-fold-2plane-gqa-qronos",
                          "had-fold-2plane-gqa-wscale", "had-fold-2plane-gqa-zellner")
    refit = name in ("fold-local", "fold-block", "had-fold-local", "had-fold-block", "had-fold-block-w0") or mode == "fold_gqa"
    block = name in ("fold-block", "had-fold-block", "had-fold-block-w0") or mode == "fold_gqa"
    tm = dict(gram=0.0, solve=0.0, gptq=0.0, refit=0.0, fwd=0.0)
    if yaqa:
        tm["yaqa"] = 0.0
    if FLAG_PROBE is not None:
        loc = dict(locals())
        FLAG_PROBE.append({k: v for k, v in loc.items() if isinstance(v, (bool, int, float, str)) and k not in ("name", "loc")})
        return None, None, None
    nsign = nr = nc = nd = ncorr = ntot = 0
    Ls = []
    if mode == "rtn":
        for l in range(cfg.nl):
            L = dict(TL[l])
            if l < a.layers:
                for k in KINDS:
                    W = TL[l][k].to(DEV).to(dt)
                    O, n = W.shape
                    A = W.view(O, n // GROUP, GROUP).abs().mean(-1).clamp_min(1e-12)
                    S = 1.0 - 2.0 * (W < 0).to(dt)
                    L[k] = (S * f16r(A).repeat_interleave(GROUP, 1)).float().cpu()
                    nsign += O * n; nr += O * (n // GROUP) * 16; ntot += O * n
            Ls.append(L)
        return Ls, dict(bits=nsign + nr, params=ntot), tm
    N = ids_cal.shape[0]
    gq = None
    gq_b = None
    if guided or yaqa:
        assert mode == "fold_gqa" and use_grow, "-guided / -yaqa apply to the fold_gqa arms with row weights"
        if yaqa_xv:
            # Two captures on disjoint halves of the calibration set: the refine is fitted on half A, half B is the
            # held-out estimate.  Same K, seed and CHUNK per call, so the only difference is the token population.
            assert ids_cal.shape[0] >= 2, ids_cal.shape
            gq = gq_capture(cfg, TL, ids_cal.cpu()[0::2], K=a.gq_k, seed=a.gq_seed, outer=True)
            gq_b = gq_capture(cfg, TL, ids_cal.cpu()[1::2], K=a.gq_k, seed=a.gq_seed, outer=True)
            tm["capture"] = gq["wall_s"] + gq_b["wall_s"]
        else:
            gq = gq_capture(cfg, TL, ids_cal.cpu(), K=a.gq_k, seed=a.gq_seed, outer=yaqa, blocks=yaqa_lb)
            tm["capture"] = gq["wall_s"]
        if yaqa:
            assert "ho" in gq, "outer capture produced no output-side Hessian"
        if guided:
            if gq_gamma != 1.0:
                gq = dict(gq, a=[{st: x ** gq_gamma / (x ** gq_gamma).mean() for st, x in d.items()} for d in gq["a"]])
            if GQ_FORCE_ONES or gq_part == "b":
                gq = dict(gq, a=[{st: torch.ones_like(x) for st, x in d.items()} for d in gq["a"]])
    ho = gq["ho"] if yaqa else None
    ho_lw = gq["ho_lw"] if yaqa_lw else None
    ho_lb = gq["ho_lb"] if yaqa_lb else None
    ho_b = gq_b["ho"] if yaqa_xv else None
    ht = cfg.E[ids_cal.cpu()]
    hs = ht.clone()
    for l in range(cfg.nl):
        if l >= a.layers:
            Ls.append(dict(TL[l]))
            continue
        Lt = {k: v.to(DEV) for k, v in TL[l].items()}
        Lq = dict(Lt)
        low_vram = torch.cuda.mem_get_info()[0] < 2 * 2**30
        slab_sz = 1 if low_vram else a.slab
        t0_fwd = tick()
        Pt_cache = []
        keep_keys = ("q", "k", "v", "mid", "g", "u", "out") + (("xa", "a", "xf", "m") if teacher_in else ())
        # cached teacher parts are N * CHUNK * 4 bytes per kept channel; on CPU when they would not leave 5 GiB free
        hid = Lt["o"].shape[0]
        per_tok = sum(Lt[k].shape[0] for k in ("q", "k", "v", "gate", "up")) + 2 * hid
        per_tok += 3 * hid + Lt["up"].shape[0] if teacher_in else 0
        pt_dev = "cpu" if torch.cuda.mem_get_info()[0] < 5 * 2**30 + N * ids_cal.shape[1] * 4 * per_tok else DEV
        for s0 in range(0, N, slab_sz):
            Pt_f = parts(cfg, Lt, ht[s0:s0 + slab_sz].to(DEV))
            Pt_cache.append({k: (Pt_f[k].cpu() if pt_dev == "cpu" else Pt_f[k]) for k in keep_keys})
            del Pt_f
        hs_mid = torch.empty_like(hs) if not teacher_in else None
        tm["fwd"] += tick() - t0_fwd
        for sname, names, inkey in STAGES:
            n = Lt[names[0]].shape[1]
            dev_s = "cpu" if low_vram else DEV
            Hm = torch.zeros(n, n, dtype=dt, device=dev_s)
            Hm_w = torch.zeros(n, n, dtype=dt, device=dev_s) if sink_gate else None
            Cs = {nm: torch.zeros(n, Lt[nm].shape[0], dtype=dt, device=dev_s) for nm in names}
            Cts = {nm: torch.zeros(n, Lt[nm].shape[0], dtype=dt, device=dev_s) if (split_intra_inter and sname in ("o", "down")) else None for nm in names} if (split_intra_inter and sname in ("o", "down")) else None
            x_sum = torch.zeros(n, dtype=dt, device=dev_s)
            y_sum = {nm: torch.zeros(Lt[nm].shape[0], dtype=dt, device=dev_s) for nm in names}
            X_slabs = [] if ((glu_comp or use_res or use_align) and sname == "ffn") else None
            stage_in_slabs = [] if (not teacher_in and sname in ("o", "down")) else None
            n_toks = 0
            t0 = tick()
            stage_keys = tuple(set((("mid", "out") if sink_gate else ()) + ((inkey, "mid", "xa", "xf") if teacher_in else ()) + (("q", "k", "v") if sname == "qkv" else (("mid",) if sname == "o" else (("g", "u") if sname == "ffn" else ("mid", "out"))))))
            for idx_s, s0 in enumerate(range(0, N, slab_sz)):
                hts = ht[s0:s0 + slab_sz].to(DEV)
                Pt_sl = Pt_cache[idx_s]
                Pt = {k: Pt_sl[k].to(DEV) for k in stage_keys if k in Pt_sl}
                if teacher_in:
                    Pin_in, Pin_mid, hss = Pt[inkey], Pt["mid"], hts
                    Pin_xa = Pt["xa"] if (split_intra_inter and sname == "o") else None
                    Pin_xf = Pt["xf"] if (split_intra_inter and sname == "down") else None
                else:
                    hss = hs[s0:s0 + slab_sz].to(DEV)
                    if sname == "qkv":
                        Pin_in = rms(hss, Lq["ln1"], cfg.eps)
                        Pin_mid = Pin_xa = Pin_xf = None
                    elif sname == "o":
                        Pin_xa = rms(hss, Lq["ln1"], cfg.eps)
                        Pin_in = attn_a(cfg, Lq, Pin_xa)
                        Pin_mid = Pin_xf = None
                        stage_in_slabs.append(Pin_in.cpu() if pt_dev == "cpu" else Pin_in)
                    elif sname == "ffn":
                        Pin_mid = hs_mid[s0:s0 + slab_sz].to(DEV)
                        Pin_in = rms(Pin_mid, Lq["ln2"], cfg.eps)
                        Pin_xa = Pin_xf = None
                    else:
                        Pin_mid = hs_mid[s0:s0 + slab_sz].to(DEV)
                        Pin_xf = rms(Pin_mid, Lq["ln2"], cfg.eps)
                        Pin_in = F.silu(Pin_xf @ Lq["gate"].T) * (Pin_xf @ Lq["up"].T)
                        Pin_xa = None
                        stage_in_slabs.append(Pin_in.cpu() if pt_dev == "cpu" else Pin_in)
                X = Pin_in.reshape(-1, n)
                Xd = X.to(dt).to(dev_s)
                Xraw = Xd
                a_sl = None
                if guided:
                    # token t contributes a_t * (x_t x_t^T) and a_t * (x_t y_t^T): weighted least squares sum_t a_t |y_t - W x_t|^2.
                    # Slab order and chunk order equal the capture's (ids_cal rows s0:s0+slab_sz, T = CHUNK).
                    a_sl = gq["a"][l][sname][s0:s0 + slab_sz].reshape(-1, 1).to(dt).to(dev_s)
                    assert a_sl.shape[0] == X.shape[0], (a_sl.shape, X.shape)
                    Xd = Xraw * a_sl.sqrt()
                Hm += Xd.T @ Xd
                if sink_gate:
                    r_t = torch.maximum(hts.float().norm(dim=-1), torch.maximum(Pt["mid"].float().norm(dim=-1), Pt["out"].float().norm(dim=-1)))
                    r_s_base = hss.float().norm(dim=-1)
                    r_s = torch.maximum(r_s_base, Pin_mid.float().norm(dim=-1)) if Pin_mid is not None else r_s_base
                    r_tok = torch.maximum(r_t, r_s)
                    r_med = r_t[:, 1:].median()
                    w_tok = torch.where(r_tok > a.sink_thresh * r_med, (r_med / r_tok.clamp_min(1e-6)), torch.ones_like(r_tok)).reshape(-1, 1).to(dt).to(dev_s)
                    Xw = Xd * w_tok
                    Hm_w += Xw.T @ Xw
                else:
                    Xw = Xd
                    w_tok = None
                if X_slabs is not None:
                    X_slabs.append(Xraw.cpu())  # unweighted: the -resq GLU solve reads raw activations
                if mode == "fold_gqa":
                    if guided:
                        x_sum += (Xraw * a_sl).sum(0)
                        n_toks += float(a_sl.sum())
                    else:
                        x_sum += Xd.sum(0)
                        n_toks += X.shape[0]
                if split_intra_inter and sname == "o":
                    a_ts = attn_a(cfg, Lt, Pin_xa)
                    Y_ts = (a_ts @ Lt["o"].T).reshape(-1, Lt["o"].shape[0]).to(dt).to(dev_s)
                    if guided:
                        Y_ts = Y_ts * a_sl.sqrt()
                    Cts["o"] += Xw.T @ (Y_ts * w_tok if w_tok is not None else Y_ts)
                    del a_ts, Y_ts, Pin_xa
                elif split_intra_inter and sname == "down":
                    m_ts = F.silu(Pin_xf @ Lt["gate"].T) * (Pin_xf @ Lt["up"].T)
                    Y_ts = (m_ts @ Lt["down"].T).reshape(-1, Lt["down"].shape[0]).to(dt).to(dev_s)
                    if guided:
                        Y_ts = Y_ts * a_sl.sqrt()
                    Cts["down"] += Xw.T @ (Y_ts * w_tok if w_tok is not None else Y_ts)
                    del m_ts, Y_ts, Pin_xf
                if mode in ("fold", "fold_gqa", "ridge"):
                    alpha = a.block_alpha if mode == "fold_gqa" else 1.0
                    for nm in names:
                        if nm in ("q", "k", "v"):
                            Y = Pt[nm] - Lt[nm + "b"]
                        elif nm == "o":
                            Y = (Pt["mid"] - hts) + (alpha * (hts - hss) if block else 0.0)
                        elif nm == "gate":
                            Y = Pt["g"]
                        elif nm == "up":
                            Y = Pt["u"]
                        else:
                            Y = (Pt["out"] - Pt["mid"]) + (alpha * (Pt["mid"] - Pin_mid) if block else 0.0)
                        Y = Y.reshape(-1, Y.shape[-1])
                        if name == "fp-ridge-mut":
                            Y = Y[torch.randperm(Y.shape[0], device=DEV)]
                        Yd = Y.to(dt).to(dev_s)
                        Yg = Yd * a_sl.sqrt() if guided else Yd
                        Cs[nm] += Xw.T @ (Yg * w_tok if w_tok is not None else Yg)
                        if mode == "fold_gqa":
                            y_sum[nm] += (Yd * a_sl).sum(0) if guided else Yd.sum(0)
                        del Y, Yd
                del Pt, Pin_in, Pin_mid, X, Xd, Xw, w_tok, hts, hss, Xraw, a_sl
            tm["gram"] += tick() - t0
            if GQ_DEBUG is not None and l == 0:
                GQ_DEBUG[sname] = (Hm.clone().cpu(), Hm_w.clone().cpu() if Hm_w is not None else None, x_sum.clone().cpu(), n_toks)
            if mode == "fold_gqa":
                t0 = tick()
                H_solve = Hm_w if sink_gate else Hm
                if split_intra_inter:
                    Ws_list = []
                    for nm in names:
                        W0 = Lt[nm].to(dt).to(dev_s)
                        C0 = H_solve @ W0.T
                        if nm in ("o", "down"):
                            C_intra, C_inter = Cts[nm] - C0, Cs[nm] - Cts[nm]
                            if use_asym and nm == "down":
                                dP = H_solve.diagonal()
                                lam_d = a.ridge_intra * (n / 896.0)
                                Hr = H_solve.clone()
                                Hr.diagonal().add_(lam_d * dP)
                                C_reg = Cts[nm] + lam_d * (dP[:, None] * W0.T)
                                Ws_list.append(torch.cholesky_solve(C_reg, torch.linalg.cholesky(Hr)).T.contiguous())
                            else:
                                dW_intra = ridge_solve(H_solve, C_intra, a.ridge_intra * (n / 896.0))
                                dW_inter = ridge_solve(H_solve, C_inter, a.ridge_inter * (n / 896.0))
                                dW = dW_intra + a.inter_gamma * dW_inter
                                if sink_gate and a.sink_tau > 0.0:
                                    w0_rn = W0.norm(dim=1, keepdim=True).clamp_min(1e-6)
                                    dw_rn = dW.norm(dim=1, keepdim=True).clamp_min(1e-6)
                                    dW = torch.where(dw_rn > a.sink_tau * w0_rn, dW * (a.sink_tau * w0_rn / dw_rn), dW)
                                Ws_list.append(W0 + dW)
                        else:
                            gin_nm = (0.35 if nm in ("q", "k", "v") else 0.15) if ("gin3515" in name) else gin_val
                            dW_inter = ridge_solve(H_solve, Cs[nm] - C0, a.ridge_w0_gqa * (n / 896.0))
                            Ws_list.append(W0 + gin_nm * dW_inter)
                    Ws_cat = torch.cat(Ws_list, 0)
                else:
                    rg_prior = a.ridge_w0_gqa * (n / 896.0)
                    dbar = Hm.diagonal().mean()
                    C_cat = torch.cat([Cs[nm] + (rg_prior * dbar) * Lt[nm].to(dt).to(dev_s).T for nm in names], 1)
                    Ws_cat = ridge_solve(Hm, C_cat, rg_prior)
                    del C_cat
                del Cs, Cts, Hm_w
                tm["solve"] += tick() - t0
                t0 = tick()
                rg_q = a.quant_ridge * (n / 896.0) if qronos_metric else a.ridge
                damp_q = a.quant_damp * (n / 896.0) if qronos_metric else 0.01
                Hl_m = metric(Hm, rg_q)
                o0 = 0
                for nm in names:
                    Oi = Lt[nm].shape[0]
                    W0_nm = Lt[nm].to(dt).to(dev_s)
                    Ws_tik_nm = Ws_cat[o0:o0 + Oi]
                    o0 += Oi
                    g_row = None
                    if rrr_mode or use_grow:
                        Ws_nm = drift_filter(W0_nm, Ws_tik_nm, Hl_m, rank=64, gamma_tail=0.65) if rrr_mode else Ws_tik_nm
                        if nm in ("gate", "up"):
                            g_row = (Lt["down"].to(dt).to(dev_s) ** 2).sum(0)
                            g_row = g_row / g_row.mean().clamp_min(1e-12)
                        elif nm in ("o", "down"):
                            ln_t = Lt["ln2"] if nm == "o" else (TL[l + 1]["ln1"] if (l + 1 < len(TL)) else cfg.fnorm)
                            g_row = (ln_t.to(dt).to(dev_s) ** 2)
                            g_row = g_row / g_row.mean().clamp_min(1e-12)
                    elif zellner and nm in ("o", "down"):
                        Ws_nm = (1.0 - a.zellner_gamma) * W0_nm + a.zellner_gamma * Ws_tik_nm
                    else:
                        Ws_nm = Ws_tik_nm
                    if guided and not GQ_NULL_B and gq_part != "a":
                        # end-loss row weight: mean_t g_tj^2 of the output gradient, for every kind (q, k, v included)
                        g_row = gq["b"][l][nm].to(dt).to(dev_s)
                        assert g_row.shape[0] == Oi
                    if glu_comp and nm == "up" and X_slabs is not None:
                        W_g0 = Ws_cat[:Lt["gate"].shape[0]]
                        Wq_g = Lq["gate"].to(dt).to(dev_s)
                        C_glu = torch.zeros(n, Oi, dtype=dt, device=dev_s)
                        for X_b in X_slabs:
                            X_bd = X_b.to(dev_s)
                            s_0 = F.silu(X_bd @ W_g0.T)
                            s_q = F.silu(X_bd @ Wq_g.T)
                            u_0 = X_bd @ Ws_nm.T
                            du_opt = (s_q * (s_0 - s_q) / (s_q ** 2 + a.glu_mu)) * u_0
                            C_glu += X_bd.T @ (a.glu_gamma * du_opt)
                            del X_bd, s_0, s_q, u_0, du_opt
                        Ws_nm = Ws_nm + ridge_solve(Hm, C_glu, a.glu_ridge * (n / 896.0))
                        del W_g0, Wq_g, C_glu, X_slabs
                        X_slabs = None
                    fk = dict(refit=True, had=True, damp=damp_q, inner_iters=(3 if e8_icm else 2), act_order=act_order, wscale=wscale, hier_scale=hier_scale, bscale=bscale, cgauge=cgauge, bgptq=bgptq, hpre=hpre, e8_code=e8_code, e8_beam=e8_beam, e8_icm=e8_icm, icm_ridge=icm_ridge, g_row=g_row)
                    fit_iters = 1 if e8_icm else a.iters
                    Wp1, info1 = fold_fit(Ws_nm, Hm, fit_iters, rg_q, **fk)
                    nsign += Oi * n; nr += info1["bits_r"]; nc += info1["bits_c"]; nd += info1.get("bits_d", 0); ntot += Oi * n
                    if q_mlp_trans:
                        two_plane_nm = ("k", "v", "o")
                    elif full_attn_2plane:
                        two_plane_nm = ("q", "k", "v", "o")
                    else:
                        two_plane_nm = ("k", "v")
                    pk1 = info1["pack"]; pk2 = None
                    pk2_top = None; top_sel = None
                    sink_planes = [_sink_plane(pk1)] if PACK_SINK is not None else None
                    if nm in two_plane_nm:
                        Wp2, info2 = fold_fit(Ws_nm - Wp1, Hm, fit_iters, rg_q, **fk)
                        pk2 = info2["pack"]
                        if backfit_2plane:
                            Wp1_b, info1_b = fold_fit(Ws_nm - Wp2, Hm, fit_iters, rg_q, **fk)
                            if J(Ws_nm, Wp1_b + Wp2, Hl_m) < J(Ws_nm, Wp1 + Wp2, Hl_m):
                                Wp1, info1 = Wp1_b, info1_b
                                pk1 = info1["pack"]
                                Wp2_b, info2_b = fold_fit(Ws_nm - Wp1, Hm, fit_iters, rg_q, **fk)
                                if J(Ws_nm, Wp1 + Wp2_b, Hl_m) < J(Ws_nm, Wp1 + Wp2, Hl_m):
                                    Wp2, info2 = Wp2_b, info2_b
                                    pk2 = info2["pack"]
                        if PACK_SINK is not None:
                            sink_planes = [_sink_plane(pk1), _sink_plane(pk2)]
                        if nb_arm and nm in ("k", "v", "o"):
                            W1n, W2n, nb_log, Pn1, Pn2 = nb_refine(Ws_nm, Hl_m, pk1, pk2, g_row=g_row, icm_ridge=icm_ridge)
                            Wp1, Wp2 = W1n, W2n
                            pk1, pk2 = Pn1, Pn2
                            if PACK_SINK is not None:
                                sink_planes = [_sink_plane(pk1), _sink_plane(pk2)]
                            st_ = nb_stats[nm]
                            st_["J_seq"] += nb_log["J_seq"]; st_["J_nb"] += nb_log["J_nb"]; st_["moves"] += nb_log["moves"]
                            st_["rounds_kept"] += nb_log["rounds_kept"]; st_["tensors"] += 1
                        Wp = Wp1 + Wp2
                        nsign += Oi * n; nr += info2["bits_r"]; nc += info2["bits_c"]; nd += info2.get("bits_d", 0)
                    elif (mlp_2plane_k896 or q_mlp_trans) and nm in ("gate", "up"):
                        k_rows = (SKEW_K_ROWS[l] if use_skew else a.qmlp_rows) if q_mlp_trans else 832
                        if k_rows > 0:
                            row_mass = (Ws_nm ** 2).sum(1)
                            top_idx = torch.topk(row_mass, k_rows).indices
                            Ws_top = Ws_nm[top_idx] - Wp1[top_idx]
                            fk_top = dict(fk, g_row=g_row[top_idx] if g_row is not None else None)
                            Wp2_top, info2 = fold_fit(Ws_top, Hm, fit_iters, rg_q, **fk_top)
                            Wp = Wp1.clone()
                            Wp[top_idx] += Wp2_top
                            pk2_top = info2["pack"]; top_sel = top_idx
                            if PACK_SINK is not None:
                                sink_planes = [_sink_plane(pk1), _sink_plane(pk2_top, rows=top_idx)]
                            nsign += k_rows * n; nr += info2["bits_r"]; nc += info2["bits_c"]; nd += info2.get("bits_d", 0)
                        else:
                            Wp = Wp1
                    else:
                        Wp = Wp1
                    if yaqa:
                        # Output-side (YAQA) refinement of the finished codes: every full-row plane (1 or 2) is refined
                        # against HO[l][nm], a top-rows partial plane enters as the frozen W_fixed.  Runs before any
                        # row-scale fold, sink store or Lq assignment, so every later consumer sees the refined packs.
                        # -yaqaxv: the applied refine fits half A, so its J_O gain need not transfer; half B (HOB)
                        # scores D0 (before), D1 (after) and the diagonal-shrunk probes, held-out.  Cost: two extra
                        # yaqa_refine passes per linear plus three J_O evaluations, each O*O*n + O*n*n MACs.
                        t0y = tick()
                        HO_nm = ho[l][nm].to(dev_s).to(dt)
                        if yaqa_lw or yaqa_lb:
                            # per-linear analytic intensity (Schafer and Strimmer 2005); -yaqalb's is the batch-means
                            # (sequence-unit) form, whose sample unit is the sequence, not the token
                            al = float(ho_lw[l][nm]) if yaqa_lw else float(ho_lb[l][nm])
                            HO_nm = (1.0 - al) * HO_nm + al * torch.diag(HO_nm.diagonal())
                        elif yaqa_shrink > 0.0:
                            # the held-out split showed the unshrunk refine fits HO estimation noise; shrinkage toward
                            # the diagonal is the bias-variance trade (Ledoit and Wolf 2004); O(O^2) once per linear
                            HO_nm = (1.0 - yaqa_shrink) * HO_nm + yaqa_shrink * torch.diag(HO_nm.diagonal())
                        if yaqa_xv:
                            HOB = ho_b[l][nm].to(dev_s).to(dt)
                            JB_before = _yaqa_JO(HOB, Hl_m, Ws_nm - Wp)
                            pk1_pre, pk2_pre = pk1, pk2
                            W_fixed_pre = None
                            if nm not in two_plane_nm and pk2_top is not None:
                                W_fixed_pre = torch.zeros_like(Ws_nm)
                                W_fixed_pre[top_sel] = _plane_w(pk2_top)
                        if nm in two_plane_nm:
                            P1r, P2r, ylog = yaqa_refine(Ws_nm, Hl_m, HO_nm, pk1, pk2, icm_ridge=icm_ridge)
                            pk1, pk2 = P1r, P2r
                            info1["pack"], info2["pack"] = P1r, P2r
                            Wp1, Wp2 = _plane_w(P1r), _plane_w(P2r)
                            Wp = Wp1 + Wp2
                            if PACK_SINK is not None:
                                sink_planes = [_sink_plane(P1r), _sink_plane(P2r)]
                        else:
                            W_fixed = None
                            if pk2_top is not None:
                                W_fixed = torch.zeros_like(Ws_nm)
                                W_fixed[top_sel] = _plane_w(pk2_top)
                            P1r, _, ylog = yaqa_refine(Ws_nm, Hl_m, HO_nm, pk1, None, W_fixed=W_fixed, icm_ridge=icm_ridge)
                            pk1 = P1r
                            info1["pack"] = P1r
                            Wp1 = _plane_w(P1r)
                            Wp = Wp1.clone()
                            if pk2_top is not None:
                                Wp[top_sel] += _plane_w(pk2_top)
                            if PACK_SINK is not None:
                                sink_planes = [_sink_plane(P1r)] + ([_sink_plane(pk2_top, rows=top_sel)] if pk2_top is not None else [])
                            del W_fixed
                        if yaqa_xv:
                            assert ylog["JO_before"] != JB_before, ("half B equals half A: HO estimate not held out", JB_before)
                            JB_after = _yaqa_JO(HOB, Hl_m, Ws_nm - Wp)
                            probe = {}
                            for tag, alpha in (("a50", 0.5), ("a90", 0.9)):
                                HOa = (1.0 - alpha) * HO_nm + alpha * torch.diag(HO_nm.diagonal())
                                if pk2_pre is not None:
                                    Q1, Q2, lg_a = yaqa_refine(Ws_nm, Hl_m, HOa, pk1_pre, pk2_pre, icm_ridge=icm_ridge)
                                    Wa = _plane_w(Q1) + _plane_w(Q2)
                                else:
                                    Q1, _, lg_a = yaqa_refine(Ws_nm, Hl_m, HOa, pk1_pre, None, W_fixed=W_fixed_pre, icm_ridge=icm_ridge)
                                    Wa = _plane_w(Q1)
                                    if W_fixed_pre is not None:
                                        Wa = Wa + W_fixed_pre
                                probe[tag] = (_yaqa_JO(HOB, Hl_m, Ws_nm - Wa), lg_a["accepted"])
                                del Q1, Wa, HOa
                        st_y = yaqa_stats[nm]
                        for k_ in ("JO_before", "JO_after", "accepted", "chunks", "rows_moved"):
                            st_y[k_] += ylog[k_]
                        if yaqa_xv:
                            st_y["JB_before"] += JB_before
                            st_y["JB_after"] += JB_after
                            st_y["JB_after_a50"] += probe["a50"][0]
                            st_y["JB_after_a90"] += probe["a90"][0]
                            st_y["accepted_a50"] += probe["a50"][1]
                            st_y["accepted_a90"] += probe["a90"][1]
                            del HOB, pk1_pre, pk2_pre, W_fixed_pre, probe
                        tm["yaqa"] += tick() - t0y
                        del HO_nm
                    corr_targets = ("q", "down") if q_mlp_trans else (("q", "o", "down") if mlp_2plane_k896 else (("down",) if full_attn_2plane else ("q", "o", "down")))
                    if use_corr and nm in corr_targets:
                        rk = 2 if ((mlp_2plane_k896 or q_mlp_trans) and nm == "q") else (2 if (mlp_2plane_k896 and nm == "o") else 1)
                        Ws_corr = ((1.0 - a.corr_gamma) * W0_nm + a.corr_gamma * Ws_tik_nm) if zellner else Ws_nm
                        Corr, b_corr, _, *fac = lowrank_corrector(Ws_corr - Wp, Hl_m, rank=rk, round_f16=not bscale, u8_corr=bscale,
                                                                  factors=PACK_SINK is not None)
                        if J(Ws_corr, Wp + Corr, Hl_m) < J(Ws_corr, Wp, Hl_m):
                            Wp = Wp + Corr
                            if PACK_SINK is not None:
                                sink_planes.append(dict(dense=Corr, U=fac[0][0], V=fac[0][1]))
                        ncorr += b_corr
                    if nm in ("q", "k", "v"):
                        db = (y_sum[nm] - (x_sum @ Wp.T)) / n_toks
                        Lq[nm + "b"] = (Lt[nm + "b"].to(dt) + db.to(DEV)).float().contiguous()
                    if PACK_SINK is not None and nm == "up" and X_slabs is not None and (use_align or (use_res and not use_resq)):
                        raise NotImplementedError("pack sink: -align / non-resq -res rescale Wp outside the stored planes")
                    if use_align and nm == "up" and X_slabs is not None:
                        W_u0 = Lt["up"].to(dt).to(dev_s)
                        Wq_g = Lq["gate"].to(dt).to(dev_s)
                        Wq_u = Wp.to(dt).to(dev_s)
                        num_align = torch.zeros(Oi, dtype=dt, device=dev_s)
                        den_align = torch.zeros(Oi, dtype=dt, device=dev_s)
                        for X_b in X_slabs:
                            X_bd = X_b.to(dev_s)
                            if sink_gate:
                                r_tok = X_bd.norm(dim=-1, keepdim=True)
                                r_med = r_tok.median()
                                w_t = torch.where(r_tok > a.sink_thresh * r_med, (r_med / r_tok.clamp_min(1e-6)), torch.ones_like(r_tok)).to(dt)
                            else:
                                w_t = 1.0
                            s_q = F.silu(X_bd @ Wq_g.T)
                            u_q = X_bd @ Wq_u.T
                            u_0 = X_bd @ W_u0.T
                            sq2 = (s_q ** 2) * w_t
                            num_align += (sq2 * u_q * u_0).sum(0)
                            den_align += (sq2 * (u_q ** 2)).sum(0)
                            del X_bd, s_q, u_q, u_0, sq2
                        r_align = (num_align / den_align.clamp_min(1e-12)).clamp(0.5, 2.0)
                        Wp = Wp * r_align[:, None]
                        del W_u0, Wq_g, Wq_u, num_align, den_align, r_align
                        if not use_res:
                            del X_slabs
                            X_slabs = None
                    if use_res and nm == "up" and X_slabs is not None:
                        W_g0 = Lt["gate"].to(dt).to(dev_s)
                        W_u0 = Lt["up"].to(dt).to(dev_s)
                        Wq_g = Lq["gate"].to(dt).to(dev_s)
                        Wq_u = Wp.to(dt).to(dev_s)
                        W_down_t = Lt["down"].to(dt).to(dev_s)
                        G = W_down_t.T @ W_down_t
                        P_mat = torch.zeros(Oi, Oi, dtype=dt, device=dev_s)
                        C_mat = torch.zeros(Oi, Oi, dtype=dt, device=dev_s)
                        for X_b in X_slabs:
                            X_bd = X_b.to(dev_s)
                            P_0 = F.silu(X_bd @ W_g0.T) * (X_bd @ W_u0.T)
                            P_q = F.silu(X_bd @ Wq_g.T) * (X_bd @ Wq_u.T)
                            P_mat += P_q.T @ P_q
                            C_mat += P_q.T @ P_0
                            del X_bd, P_0, P_q
                        GoP = G * P_mat
                        GoC = G * C_mat
                        ones = torch.ones(Oi, 1, dtype=dt, device=dev_s)
                        rhs = GoC @ ones
                        lam = 0.05 * GoP.diagonal().mean().clamp_min(1e-12)
                        GoP.diagonal().add_(lam)
                        rhs += lam * ones
                        s_sol = torch.linalg.solve(GoP, rhs)[:, 0]
                        s_sol = s_sol.clamp(0.5, 2.0)
                        if use_resq:
                            # -resq: s_sol is folded into each plane's stored hierarchical row scale,
                            # so the multiplier costs no bits beyond quant_r_hier's accounted r-bits.
                            S_r1, rh_1s, ch_1s, dh_1s = info1["pack"]
                            rh_1n, _ = quant_r_hier(rh_1s * s_sol[:, None])
                            Wp = compose(fwht128(S_r1 * dh_1s[None, :]), rh_1n, ch_1s)
                            if pk2_top is not None:
                                S_r2, rh_2s, ch_2s, dh_2s = pk2_top
                                rh_2n, _ = quant_r_hier(rh_2s * s_sol[top_sel][:, None])
                                Wp[top_sel] += compose(fwht128(S_r2 * dh_2s[None, :]), rh_2n, ch_2s)
                            if PACK_SINK is not None:
                                sink_planes = [_sink_plane((S_r1, rh_1n, ch_1s, dh_1s))]
                                if pk2_top is not None:
                                    sink_planes.append(_sink_plane((S_r2, rh_2n, ch_2s, dh_2s), rows=top_sel))
                        else:
                            Wp = Wp * s_sol[:, None]
                        del W_g0, W_u0, Wq_g, Wq_u, W_down_t, G, P_mat, C_mat, GoP, GoC, ones, rhs, s_sol, X_slabs
                        X_slabs = None
                    Lq[nm] = Wp.to(DEV).float().contiguous()
                    if PACK_SINK is not None:
                        _sink_store(l, nm, sink_planes, Lq[nm])
                    del Wp, Wp1, Ws_nm, W0_nm
                del Hm, Hl_m, Ws_cat
                tm["refit"] += tick() - t0
                if not teacher_in and sname in ("o", "down"):
                    t0_s = tick()
                    for idx_s, s0 in enumerate(range(0, N, slab_sz)):
                        inp_s = stage_in_slabs[idx_s].to(DEV)
                        if sname == "o":
                            hs_mid[s0:s0 + slab_sz] = (hs[s0:s0 + slab_sz].to(DEV) + inp_s @ Lq["o"].T).cpu()
                        else:
                            hs[s0:s0 + slab_sz] = (hs_mid[s0:s0 + slab_sz].to(DEV) + inp_s @ Lq["down"].T).cpu()
                        del inp_s
                    del stage_in_slabs
                    tm["fwd"] += tick() - t0_s
                torch.cuda.empty_cache()
                continue
            t0 = tick()
            rg = a.ridge_w0 if w0_anchor else a.ridge
            if mode in ("fold", "ridge"):
                C_cat = torch.cat([Cs[nm] for nm in names], 1)
                if w0_anchor:
                    W0 = torch.cat([Lt[nm].to(dt) for nm in names], 0)
                    C_cat = C_cat + (rg * Hm.diagonal().mean()) * W0.T
                Ws = ridge_solve(Hm, C_cat, rg)
            else:
                Ws = torch.cat([Lt[nm].to(dt) for nm in names], 0)
            tm["solve"] += tick() - t0
            t0 = tick()
            O = Ws.shape[0]
            if mode == "ridge":
                Wp, info = Ws, None
            elif mode == "gptq":
                if had:
                    _, S_rot, A_ = gptq_signs(fwht128(Ws), had_gram(Hm))
                    S_ = fwht128(S_rot)
                else:
                    _, S_, A_ = gptq_signs(Ws, Hm)
                Wp = compose(S_, f16r(A_), torch.ones(n, dtype=dt, device=DEV))
                nsign += O * n; nr += 16 * O * (n // GROUP); ntot += O * n
            else:
                Wp, info = fold_fit(Ws, Hm, a.iters, rg, refit=refit, had=had)
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
            if not teacher_in and sname in ("o", "down"):
                t0_s = tick()
                for idx_s, s0 in enumerate(range(0, N, slab_sz)):
                    inp_s = stage_in_slabs[idx_s].to(DEV)
                    if sname == "o":
                        hs_mid[s0:s0 + slab_sz] = (hs[s0:s0 + slab_sz].to(DEV) + inp_s @ Lq["o"].T).cpu()
                    else:
                        hs[s0:s0 + slab_sz] = (hs_mid[s0:s0 + slab_sz].to(DEV) + inp_s @ Lq["down"].T).cpu()
                    del inp_s
                del stage_in_slabs
                tm["fwd"] += tick() - t0_s
        t0 = tick()
        for idx_s, s0 in enumerate(range(0, N, slab_sz)):
            ht[s0:s0 + slab_sz] = Pt_cache[idx_s]["out"].cpu()
            if teacher_in:
                hs[s0:s0 + slab_sz] = parts(cfg, Lq, hs[s0:s0 + slab_sz].to(DEV))["out"].cpu()
        del Pt_cache, hs_mid
        tm["fwd"] += tick() - t0
        Ls.append({k: v.cpu() for k, v in Lq.items()})
        del Lt, Lq
        torch.cuda.empty_cache()
        if a.verbose:
            print(f"  layer {l} done", flush=True)
    if (fnorm_align or sink_fnorm) and Ls:
        t0 = tick()
        if sink_fnorm:
            Ls[-1]["fnorm"] = fnorm_fit(cfg, ht, hs, steps=2, ridge=0.10, lr=a.fnorm_lr, coh_prior=True)
        else:
            Ls[-1]["fnorm"] = fnorm_fit(cfg, ht, hs, steps=4, ridge=0.05)
        tm["refit"] += tick() - t0
    binfo = dict(bits=nsign + nr + nc + nd + ncorr, params=ntot, bits_sign=nsign, bits_r=nr, bits_c=nc, bits_d=nd, bits_corr=ncorr)
    if guided:
        binfo["gq"] = dict(K=a.gq_k, seed=a.gq_seed, gamma=gq_gamma, part=gq_part, ess_frac={f"L{l}.{st}": float(x.sum() ** 2 / (x ** 2).sum() / x.numel()) for l, d in enumerate(gq["a"]) for st, x in d.items()}, capture_s=gq["wall_s"], max_dlogit=gq["max_dlogit"], stats=gq_stats(gq, sorted({0, cfg.nl // 2, cfg.nl - 1})))
    if nb_arm:
        agg_seq = sum(v["J_seq"] for v in nb_stats.values())
        agg_nb = sum(v["J_nb"] for v in nb_stats.values())
        binfo["nb"] = dict(sweeps=NB_SWEEPS, per_kind={k: dict(v, ratio=(v["J_nb"] / v["J_seq"] if v["J_seq"] else None)) for k, v in nb_stats.items()},
                           J_seq=agg_seq, J_nb=agg_nb, ratio=(agg_nb / agg_seq if agg_seq else None))
    if yaqa:
        binfo["yaqa"] = {nm: dict(v) for nm, v in yaqa_stats.items()}
        binfo["yaqa_shrink"] = yaqa_shrink
        if yaqa_lw:
            binfo["yaqa_lw"] = {f"L{l}.{nm}": float(v) for l, d in enumerate(ho_lw) for nm, v in d.items()}
        if yaqa_lb:
            binfo["yaqa_lb"] = {f"L{l}.{nm}": float(v) for l, d in enumerate(ho_lb) for nm, v in d.items()}
        binfo["yaqa_chunk"] = YAQA_CHUNK
    return Ls, binfo, tm



GQ_ARM = "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-resq-qmlp-nb"

def gq_controls(cfg, TL, ids_all, a):
    """(1) base and base+"-guided" resolve the same flags except `guided`; (2) with a_t = 1 the layer-0 Grams, x_sum, n_toks
    equal the base arm's exactly (torch.equal) and, with the row weights left to the base proxy, so does every stored layer
    and the bit count; (3) with the real a_t the Grams differ (the comparison can fail).  (4) base and base+"-yaqa" resolve
    the same flags except `yaqa`; (5) base and base+"-yaqaxv" resolve the same flags except `yaqa` and `yaqa_xv` (the xv
    capture halves the calibration rows, which changes no scalar flag); (6) base and base+"-yaqas50" resolve the same flags
    except `yaqa` and `yaqa_shrink`; (7) base and base+"-yaqalw" resolve the same flags except `yaqa` and `yaqa_lw`;
    (8) base and base+"-yaqalb" resolve the same flags except `yaqa` and `yaqa_lb`."""
    global FLAG_PROBE, GQ_FORCE_ONES, GQ_NULL_B, GQ_DEBUG
    out = {}
    ids_c = ids_all[:a.n_cal].contiguous()
    FLAG_PROBE = []
    run_variant(cfg, TL, ids_c, GQ_ARM, a)
    run_variant(cfg, TL, ids_c, GQ_ARM + GQ_SUFFIX, a)
    fb, fg = FLAG_PROBE
    FLAG_PROBE = None
    diff = sorted(k for k in set(fb) | set(fg) if fb.get(k) != fg.get(k))
    out["flag_diff"] = diff
    out["n_flags"] = len(fb)
    assert diff == ["guided"], diff
    FLAG_PROBE = []
    run_variant(cfg, TL, ids_c, GQ_ARM + "-yaqa", a)
    fy = FLAG_PROBE[0]
    FLAG_PROBE = None
    diff_y = sorted(k for k in set(fb) | set(fy) if fb.get(k) != fy.get(k))
    out["flag_diff_yaqa"] = diff_y
    assert diff_y == ["yaqa"], diff_y
    FLAG_PROBE = []
    run_variant(cfg, TL, ids_c, GQ_ARM + "-yaqas50", a)
    fys = FLAG_PROBE[0]
    FLAG_PROBE = None
    diff_ys = sorted(k for k in set(fb) | set(fys) if fb.get(k) != fys.get(k))
    out["flag_diff_yaqas"] = diff_ys
    assert diff_ys == ["yaqa", "yaqa_shrink"], diff_ys
    FLAG_PROBE = []
    run_variant(cfg, TL, ids_c, GQ_ARM + "-yaqalw", a)
    flw = FLAG_PROBE[0]
    FLAG_PROBE = None
    diff_lw = sorted(k for k in set(fb) | set(flw) if fb.get(k) != flw.get(k))
    out["flag_diff_yaqalw"] = diff_lw
    assert diff_lw == ["yaqa", "yaqa_lw"], diff_lw
    FLAG_PROBE = []
    run_variant(cfg, TL, ids_c, GQ_ARM + "-yaqalb", a)
    flb = FLAG_PROBE[0]
    FLAG_PROBE = None
    diff_lb = sorted(k for k in set(fb) | set(flb) if fb.get(k) != flb.get(k))
    out["flag_diff_yaqalb"] = diff_lb
    assert diff_lb == ["yaqa", "yaqa_lb"], diff_lb
    FLAG_PROBE = []
    run_variant(cfg, TL, ids_c, GQ_ARM + "-yaqaxv", a)
    fxv = FLAG_PROBE[0]
    FLAG_PROBE = None
    diff_xv = sorted(k for k in set(fb) | set(fxv) if fb.get(k) != fxv.get(k))
    out["flag_diff_yaqaxv"] = diff_xv
    assert diff_xv == ["yaqa", "yaqa_xv"], diff_xv
    GQ_DEBUG = {}
    Lb, ib, _ = run_variant(cfg, TL, ids_c, GQ_ARM, a)
    dbg_b, GQ_DEBUG = GQ_DEBUG, {}
    GQ_FORCE_ONES, GQ_NULL_B = True, True
    try:
        Lg, ig, _ = run_variant(cfg, TL, ids_c, GQ_ARM + GQ_SUFFIX, a)
    finally:
        GQ_FORCE_ONES, GQ_NULL_B = False, False
    dbg_g, GQ_DEBUG = GQ_DEBUG, {}
    eq = {}
    for st in dbg_b:
        hb, hwb, xb, nb_ = dbg_b[st]; hg, hwg, xg, ng_ = dbg_g[st]
        eq[st] = bool(torch.equal(hb, hg) and (hwb is None or torch.equal(hwb, hwg)) and torch.equal(xb, xg) and float(nb_) == float(ng_))
    out["null_grams_equal_by_stage"] = eq
    out["null_layers_equal"] = all(all(torch.equal(Lb[l][k], Lg[l][k]) for k in Lb[l]) for l in range(len(Lb)))
    out["null_bits_equal"] = all(ib[k] == ig[k] for k in ib if k != "nb") and ib["nb"] == ig["nb"] if "nb" in ib else all(ib[k] == ig[k] for k in ib)
    GQ_DEBUG = {}
    Lr, ir, _ = run_variant(cfg, TL, ids_c, GQ_ARM + GQ_SUFFIX, a)
    dbg_r, GQ_DEBUG = GQ_DEBUG, None
    out["real_a_grams_differ_by_stage"] = {st: not torch.equal(dbg_b[st][0], dbg_r[st][0]) for st in dbg_b}
    out["real_bits_equal_base"] = all(ib[k] == ir[k] for k in ("bits", "params", "bits_sign", "bits_r", "bits_c", "bits_d", "bits_corr"))
    # capture self-check: the capture forward reproduces the teacher logits of logits_of
    out["capture_max_dlogit_vs_logits_of"] = None
    out["capture_wall_s"] = ir["gq"]["capture_s"]
    assert all(out["null_grams_equal_by_stage"].values()) and out["null_layers_equal"] and out["null_bits_equal"], out
    assert all(out["real_a_grams_differ_by_stage"].values()) and out["real_bits_equal_base"], out
    return out

# ---------------------------------------------------------------- evaluation
_TEACHER_H_CACHE = {}
SPEC_KMAX = 8


def spec_stats(lt, ls, kmax=SPEC_KMAX):
    """Speculative-sampling acceptance of draft `ls` against target `lt` (log-probs, (B, T, V)), teacher-forced.

    Per position the acceptance is `a_t = sum_x min(p_t(x), q_t(x)) = 1 - TV(p_t, q_t)` (Leviathan, Kalman, Matias 2022,
    arXiv 2211.17192; `Nous.Model.SpeculativeAccept.accept_rate`). For a block of `k` drafted tokens starting at `t`
    the expected emitted count, with the acceptances read along the held-out text, is
    `E_k(t) = 1 + sum_{j=1..k} prod_{i<j} a_{t+i}` (the first rejection or the bonus token ends the block); returned
    as sums over the `T - kmax` start positions of each row so every k averages one population.
    Cost: O(B T V) adds for `a`, O(B T kmax) multiplies for the products; extra memory one (B, T, V) min.
    """
    a = torch.minimum(lt.exp(), ls.exp()).sum(-1).clamp(max=1.0)  # (B, T)
    T = a.shape[1]
    S = T - kmax
    prod = torch.ones_like(a[:, :S])
    ek = []
    tot = torch.ones_like(prod)
    for j in range(1, kmax + 1):
        prod = prod * a[:, j - 1:j - 1 + S]
        tot = tot + prod
        ek.append(float(tot.sum()))
    return float(a.sum()), int(a.numel()), ek, int(prod.numel())


@torch.no_grad()
def kl_eval(cfg, TL, Ls, ids, bs=10, stats=None):
    """Mean token KL(teacher || student) and top-1 agreement over `ids`; with `stats` (a dict) also the speculative
    acceptance `1 - TV` and `E_k` for k = 1..SPEC_KMAX from `spec_stats`."""
    N = ids.shape[0]
    key = (len(TL), ids.data_ptr(), N, ids.shape[1])
    if key not in _TEACHER_H_CACHE:
        ht = cfg.E[ids.cpu()]
        for l in range(len(TL)):
            Lt = {k: v.to(DEV) for k, v in TL[l].items() if k != "fnorm"}
            for s0 in range(0, N, bs):
                ht[s0:s0 + bs] = parts(cfg, Lt, ht[s0:s0 + bs].to(DEV))["out"].cpu()
            del Lt
        _TEACHER_H_CACHE[key] = ht
    ht = _TEACHER_H_CACHE[key]
    hs = cfg.E[ids.cpu()]
    for l in range(len(Ls)):
        Lq = {k: v.to(DEV) for k, v in Ls[l].items() if k != "fnorm"}
        for s0 in range(0, N, bs):
            hs[s0:s0 + bs] = parts(cfg, Lq, hs[s0:s0 + bs].to(DEV))["out"].cpu()
        del Lq
    fn_t = TL[-1].get("fnorm", cfg.fnorm).to(DEV) if TL else cfg.fnorm.to(DEV)
    fn_s = Ls[-1].get("fnorm", cfg.fnorm).to(DEV) if Ls else cfg.fnorm.to(DEV)
    E_g = cfg.E.to(DEV)
    kls, agree, n = 0.0, 0, 0
    acc, na, eks, ne = 0.0, 0, [0.0] * SPEC_KMAX, 0
    for i in range(0, N, 2):
        ut = rms(ht[i:i + 2].to(DEV), fn_t, cfg.eps)
        us = rms(hs[i:i + 2].to(DEV), fn_s, cfg.eps)
        lt = F.log_softmax((ut @ E_g.T).float(), -1)
        ls = F.log_softmax((us @ E_g.T).float(), -1)
        kls += float((lt.exp() * (lt - ls)).sum(-1).sum().item())
        agree += int((lt.argmax(-1) == ls.argmax(-1)).sum().item())
        n += int(ids[i:i + 2].numel())
        if stats is not None:
            a_s, a_n, ek, e_n = spec_stats(lt, ls)
            acc, na, ne = acc + a_s, na + a_n, ne + e_n
            eks = [x + y for x, y in zip(eks, ek)]
        del ut, us, lt, ls
    del E_g, hs
    torch.cuda.empty_cache()
    if stats is not None:
        stats.update(accept=acc / na, tv=1 - acc / na, e_k=[x / ne for x in eks], kmax=SPEC_KMAX, positions=na,
                     block_starts=ne)
    return kls / n, agree / n


# ---------------------------------------------------------------- controls
def controls(cfg, TL, hf, ids_all):
    out = {}
    dev_c = "cpu" if torch.cuda.mem_get_info()[0] < 2 * 2**30 else DEV
    g = torch.Generator(device="cpu").manual_seed(1)
    O, n = 5, 256
    Wn = torch.randn(O, n, generator=g, dtype=F64)
    X = torch.randn(600, n, generator=g, dtype=F64) * (1 + torch.rand(n, generator=g, dtype=F64) * 3)
    Hn = X.T @ X
    ref = gptq_naive_np(Wn.numpy(), Hn.numpy())
    got, _, _ = gptq_signs(Wn.to(dev_c), Hn.to(dev_c))
    out["c1_gptq_vec_vs_naive_numpy_maxabs"] = float(np.abs(got.cpu().numpy() - ref).max())
    got_i, _, _ = gptq_signs(Wn.to(dev_c), torch.eye(n, dtype=F64, device=dev_c))
    A = Wn.view(O, n // GROUP, GROUP).abs().mean(-1)
    rtn = (1.0 - 2.0 * (Wn < 0).to(F64)) * A.repeat_interleave(GROUP, 1)
    out["c2_identity_gram_equals_rtn_maxabs"] = float((got_i.cpu() - rtn).abs().max())
    U = feedback_u(Hn.to(dev_c))
    h = Hn.to(dev_c).clone(); h.diagonal().add_(0.01 * Hn.diagonal().mean())
    out["c3_UtU_h_minus_I_maxabs"] = float((U.T @ U @ h - torch.eye(n, dtype=F64, device=dev_c)).abs().max())
    # c4: exact steps are stationary minimisers and the discrete step is the only non-monotone one
    Ws, Hd = Wn.to(dev_c).repeat(40, 1), Hn.to(dev_c)
    Hl = metric(Hd, 1e-3)
    _, S, r = gptq_signs(Ws, Hd)
    c0 = torch.ones(n, dtype=F64, device=dev_c)
    j0 = J(Ws, compose(S, r, c0), Hl)
    c1 = c_step(Ws, Hl, S, r)
    j1 = J(Ws, compose(S, r, c1), Hl)
    bad = 0
    for _ in range(20):
        d = 1e-3 * torch.randn(n, dtype=F64, device=dev_c)
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
    Wt = torch.randn(7, n, generator=g, dtype=F64).to(dev_c)
    Xd = X.to(dev_c)
    Wr = ridge_solve(Xd.T @ Xd, Xd.T @ (Xd @ Wt.T), 1e-12)
    out["c5_fold_solve_recovery_maxabs"] = float((Wr - Wt).abs().max())
    # c6: teacher forward vs HF
    x = ids_all[100:102]
    with torch.no_grad():
        lm = F.log_softmax(logits_of(cfg, TL, x).float(), -1)
        lh = F.log_softmax(hf(x.cpu()).logits.float(), -1)
    out["c6_forward_vs_hf_maxabs_logp"] = float((lm - lh).abs().max())
    out["c6_forward_vs_hf_KL"] = float((lh.exp() * (lh - lm)).sum(-1).mean())
    # c7: Gram vs the file used by the earlier arms (blk.5 ffn_down input)
    Hf = torch.from_numpy(np.fromfile(f"{GRAM_DIR}/blk.5.ffn_down_in.f64", dtype="<f8").reshape(cfg.inter, cfg.inter))
    ids_cal = ids_all[:100]
    Hm = torch.zeros(cfg.inter, cfg.inter, dtype=F64)
    ht = cfg.E[ids_cal.cpu()]
    with torch.no_grad():
        for l in range(6):
            Ll = {k: v.to(DEV) for k, v in TL[l].items()}
            for s0 in range(0, 100, 2):
                P = parts(cfg, Ll, ht[s0:s0 + 2].to(DEV))
                if l == 5:
                    Xm = P["m"].reshape(-1, cfg.inter).cpu().to(F64)
                    Hm += Xm.T @ Xm
                ht[s0:s0 + 2] = P["out"].cpu()
                del P
            del Ll
            torch.cuda.empty_cache()
    out["c7_gram_blk5_down_rel_frob_vs_file"] = float((Hm - Hf).norm() / Hf.norm())
    out["c7_gram_trace_ratio"] = float(Hm.trace() / Hf.trace())
    # c8: rtn variant vs independent numpy for one tensor
    W = TL[3]["o"].cpu().numpy().astype(np.float64)
    ref = np.where(W < 0, -1.0, 1.0) * np.abs(W).reshape(W.shape[0], -1, GROUP).mean(-1).astype(np.float16).astype(np.float64).repeat(GROUP, 1)
    A2 = torch.from_numpy(W).view(W.shape[0], -1, GROUP).abs().mean(-1).clamp_min(1e-12)
    mine = ((1.0 - 2.0 * (torch.from_numpy(W) < 0).to(F64)) * f16r(A2).repeat_interleave(GROUP, 1)).numpy()
    out["c8_rtn_vs_numpy_maxabs"] = float(np.abs(mine - ref).max())
    # c9: fwht128 involution and orthogonality in fp32 (CostCorrespondence.rotation_elliptic / spectrum_conj)
    Wf = Wn.float().to(dev_c)
    Hf32 = Hn.float().to(dev_c)
    out["c9_fwht128_involution_fp32_maxabs"] = float((fwht128(fwht128(Wf)) - Wf).abs().max())
    out["c9_had_gram_quadratic_rel_diff_fp32"] = float(abs(J(fwht128(Wf), torch.zeros_like(Wf), had_gram(Hf32)) - J(Wf, torch.zeros_like(Wf), Hf32)) / J(Wf, torch.zeros_like(Wf), Hf32))
    # c10: orthogonal two-plane corrector exact eigenvalue drop (CostCorrespondence.orthogonal_corrector_attained)
    Hl32 = metric(Hf32, 1e-3)
    Corr_ex, _, top2_sum = lowrank_corrector(Wf, Hl32, rank=2, round_f16=False)
    j_before = J(Wf, torch.zeros_like(Wf), Hl32)
    j_after = J(Wf, Corr_ex, Hl32)
    out["c10_orthogonal_corrector_rel_err_fp32"] = float(abs((j_before - j_after) - top2_sum) / j_before)
    # c11: two-component telescoping split matches single-ridge when ridge_intra == ridge_inter (CostCorrespondence.telescoping_target_split)
    W0_c = torch.randn(7, n, generator=g, dtype=F64).float().to(dev_c)
    Y_ts_c = Xd.float() @ W0_c.T + 0.1 * torch.randn(600, 7, generator=g, dtype=F64).float().to(dev_c)
    Y_t_c = Y_ts_c + 0.2 * torch.randn(600, 7, generator=g, dtype=F64).float().to(dev_c)
    C_s_c = Xd.float().T @ Y_t_c
    C_ts_c = Xd.float().T @ Y_ts_c
    C_0_c = Hf32 @ W0_c.T
    lam_c = 0.05
    W_single = ridge_solve(Hf32, C_s_c + (lam_c * Hf32.diagonal().mean()) * W0_c.T, lam_c)
    W_split = W0_c + ridge_solve(Hf32, C_ts_c - C_0_c, lam_c) + ridge_solve(Hf32, C_s_c - C_ts_c, lam_c)
    out["c11_two_component_telescoping_rel_err_fp32"] = float((W_single - W_split).norm() / W_single.norm())
    # c12: block-preserving descending curvature act_order strictly reduces GPTQ quadratic error (CostCorrespondence.act_order_curvature_dominance)
    Wq_id, S_id, A_id = gptq_signs(Wf, Hf32, act_order=False)
    Wq_act, S_act, A_act = gptq_signs(Wf, Hf32, act_order=True)
    Wq_act_recomp = S_act * A_act.repeat_interleave(GROUP, 1)
    out["c12_act_order_recompose_maxabs_fp32"] = float((Wq_act - Wq_act_recomp).abs().max())
    out["c12_act_order_j_ratio_fp32"] = float(J(Wf, Wq_act, Hl32) / J(Wf, Wq_id, Hl32))
    # c13: whitened Cholesky scale projection strictly reduces GPTQ quadratic error on top of act_order (CostCorrespondence.whitened_scale_shift_identity)
    Wq_ws, S_ws, A_ws = gptq_signs(Wf, Hf32, act_order=True, wscale=True)
    Wq_ws_recomp = S_ws * A_ws.repeat_interleave(GROUP, 1)
    out["c13_wscale_recompose_maxabs_fp32"] = float((Wq_ws - Wq_ws_recomp).abs().max())
    out["c13_wscale_vs_act_j_ratio_fp32"] = float(J(Wf, Wq_ws, Hl32) / J(Wf, Wq_act, Hl32))
    # c14: hierarchical scale quantization + 2nd plane strictly beats 1-plane fp16 scale (CostCorrespondence.scale_vs_plane_distortion_dominance)
    A_hq, _ = quant_r_hier(A_ws)
    out["c14_hier_scale_rel_frob_fp32"] = float((A_ws - A_hq).norm() / A_ws.norm())
    Wp_1p, _ = fold_fit(Wf, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=False)
    Wp_h1, _ = fold_fit(Wf, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True)
    Wp_h2, _ = fold_fit(Wf - Wp_h1, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True)
    out["c14_2plane_hier_vs_1plane_f16_j_ratio_fp32"] = float(J(Wf, Wp_h1 + Wp_h2, Hl32) / J(Wf, Wp_1p, Hl32))
    # c15: Hadamard butterfly d_step stationarity & 2-plane back-fitting descent (CostCorrespondence.butterfly_d_step_dominance / two_plane_backfit_descent)
    _, S_rot_d, r_d = gptq_signs(fwht128(Ws), had_gram(Hd))
    S_d0 = fwht128(S_rot_d)
    c_d0 = c_step(Ws, Hl, S_d0, r_d)
    j_d0 = J(Ws, compose(S_d0, r_d, c_d0), Hl)
    d_opt = d_step(Ws, Hl, S_rot_d, r_d, c_d0)
    j_d1 = J(Ws, compose(fwht128(S_rot_d * d_opt[None, :]), r_d, c_d0), Hl)
    bad_d = 0
    for _ in range(20):
        dd = 1e-3 * torch.randn(n, dtype=F64, device=dev_c)
        bad_d += J(Ws, compose(fwht128(S_rot_d * (d_opt + dd)[None, :]), r_d, c_d0), Hl) < j_d1 - 1e-9 * abs(j_d1)
    out["c15_dstep_j_ratio_f64"] = float(j_d1 / j_d0)
    out["c15_dstep_perturbations_below_min"] = int(bad_d)
    Wp_h1_b, _ = fold_fit(Wf - Wp_h2, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True)
    Wp_h2_b, _ = fold_fit(Wf - Wp_h1_b, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True)
    out["c15_2plane_backfit_j_ratio_fp32"] = float(J(Wf, Wp_h1_b + Wp_h2_b, Hl32) / J(Wf, Wp_h1 + Wp_h2, Hl32))
    # c16: butterfly sign-group gauge, G-weighted c_step & reduced-rank drift filter (CostCorrespondence §12)
    sd_s = torch.where(d_opt < 0, -1.0, 1.0).to(F64)
    d_a = d_opt.abs().view(-1, GROUP)
    m_g = d_a.amax(1)
    W_g1 = compose(fwht128((S_rot_d * sd_s[None, :]) * (d_a / m_g[:, None]).reshape(1, -1)), r_d * m_g[None, :], c_d0)
    out["c16_butterfly_gauge_maxabs_f64"] = float((W_g1 - compose(fwht128(S_rot_d * d_opt[None, :]), r_d, c_d0)).abs().max())
    g_w = torch.linspace(0.1, 10.0, Ws.shape[0], dtype=F64, device=dev_c)
    c_g = c_step(Ws, Hl, S_d0, r_d, g_row=g_w)
    out["c16_g_weighted_cstep_j_ratio_f64"] = float(J(Ws * g_w.sqrt()[:, None], compose(S_d0, r_d, c_g) * g_w.sqrt()[:, None], Hl) / J(Ws * g_w.sqrt()[:, None], compose(S_d0, r_d, c_d0) * g_w.sqrt()[:, None], Hl))
    u_true = torch.randn(Ws.shape[0], 2, generator=g, dtype=F64).to(dev_c)
    v_true = torch.randn(2, n, generator=g, dtype=F64).to(dev_c)
    W_true = Ws + u_true @ v_true
    W_noisy = W_true + 0.5 * torch.randn(Ws.shape[0], n, generator=g, dtype=F64).to(dev_c)
    W_rrr = drift_filter(Ws, W_noisy, Hl, rank=2, gamma_tail=0.2)
    out["c16_drift_filter_heldout_j_ratio_f64"] = float(J(W_true, W_rrr, Hl) / J(W_true, W_noisy, Hl))
    # c17: column-scale group-max gauge, butterfly GPTQ metric invariance & rank-2 down corrector dominance (§13)
    mcg_c = c_d0.abs().view(-1, GROUP).amax(1)
    W_cg = compose(S_d0, r_d * mcg_c[None, :], (c_d0.view(-1, GROUP) / mcg_c[:, None]).reshape(-1))
    out["c17_column_scale_gauge_maxabs_f64"] = float((W_cg - compose(S_d0, r_d, c_d0)).abs().max())
    ds_c = d_opt.abs().clamp_min(0.1)
    j_m1 = J(fwht128(Ws) / ds_c[None, :], S_rot_d * r_d.repeat_interleave(GROUP, 1), had_gram(Hl) * ds_c[:, None] * ds_c[None, :])
    j_m2 = J(fwht128(Ws), (S_rot_d * ds_c[None, :]) * r_d.repeat_interleave(GROUP, 1), had_gram(Hl))
    out["c17_butterfly_gptq_metric_rel_err_f64"] = float(abs(j_m1 - j_m2) / j_m2)
    Wp_b1, _ = fold_fit(Wf, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, cgauge=False)
    C_b1, _, _ = lowrank_corrector(Wf - Wp_b1, Hl32, rank=1, round_f16=False, u8_corr=True)
    Wp_b2, _ = fold_fit(Wf, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, cgauge=True)
    C_b2, _, _ = lowrank_corrector(Wf - Wp_b2, Hl32, rank=2, round_f16=False, u8_corr=True)
    out["c17_rk2_cgauge_vs_rk1_bscale_j_ratio_fp32"] = float(J(Wf, Wp_b2 + C_b2, Hl32) / J(Wf, Wp_b1 + C_b1, Hl32))
    # c18: Face Swap #1 — Hyperbolic Pre-Hadamard Cauchy-Schwarz squeeze (§14)
    w_aniso = torch.linspace(0.2, 5.0, n, dtype=F64, device=dev_c)
    W_an = Ws * w_aniso[None, :]
    w_r = (W_an ** 2).mean(0).sqrt()
    h_r = Hl.diagonal().sqrt()
    c_cs = (w_r / h_r).sqrt()
    prod_1 = float((w_r ** 2).sum() * (h_r ** 2).sum())
    prod_cs = float(((w_r / c_cs) ** 2).sum() * ((c_cs * h_r) ** 2).sum())
    prod_opt = float((w_r * h_r).sum() ** 2)
    out["c18_cauchy_schwarz_opt_err_f64"] = float(abs(prod_cs - prod_opt) / prod_opt)
    out["c18_cauchy_schwarz_vs_elliptic_ratio_f64"] = float(prod_cs / prod_1)
    Wp_no_hpre, _ = fold_fit(W_an.float(), Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, hpre=False)
    Wp_hpre, _ = fold_fit(W_an.float(), Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, hpre=True)
    out["c18_hpre_vs_elliptic_j_ratio_fp32"] = float(J(W_an.float(), Wp_hpre, Hl32) / J(W_an.float(), Wp_no_hpre, Hl32))
    # c19: Face Swap #1B — Convex readout RMSNorm Newton-Fisher alignment (§14)
    E_syn = torch.randn(64, 32, generator=g, dtype=F64).to(dev_c)
    ut_syn = F.normalize(torch.randn(200, 32, generator=g, dtype=F64).to(dev_c), dim=-1)
    us_syn = F.normalize(ut_syn * torch.linspace(0.6, 1.4, 32, dtype=F64, device=dev_c) + 0.25 * torch.randn(200, 32, generator=g, dtype=F64).to(dev_c), dim=-1)
    g0_syn = torch.ones(32, dtype=F64, device=dev_c)
    g_syn = g0_syn.clone()
    lt_syn = F.log_softmax((ut_syn * g0_syn) @ E_syn.T, dim=-1)
    pt_syn = lt_syn.exp()
    ls0_syn = F.log_softmax((us_syn * g0_syn) @ E_syn.T, dim=-1)
    kl0_syn = float((pt_syn * (lt_syn - ls0_syn)).sum(-1).mean())
    for _ in range(4):
        ps_syn = F.softmax((us_syn * g_syn) @ E_syn.T, dim=-1)
        gr_syn = (us_syn * ((ps_syn - pt_syn) @ E_syn)).mean(0)
        fe_syn = ((us_syn ** 2) * (ps_syn @ (E_syn ** 2) - (ps_syn @ E_syn) ** 2).clamp_min(1e-6)).mean(0)
        rg_syn = 0.05 * fe_syn.mean()
        g_syn = g_syn - (gr_syn + rg_syn * (g_syn - g0_syn)) / (fe_syn + rg_syn)
    ls1_syn = F.log_softmax((us_syn * g_syn) @ E_syn.T, dim=-1)
    kl1_syn = float((pt_syn * (lt_syn - ls1_syn)).sum(-1).mean())
    out["c19_fnorm_newton_kl_ratio_f64"] = float(kl1_syn / kl0_syn)
    # c20: Face Swap #2 — 8-D E8 U O8 spinor + parabolic-sparse codebook at 1.0000 b/w (§15)
    C_e8_c = e8_o8_codebook(torch.float32, dev_c)
    out["c20_e8_codebook_size"] = int(C_e8_c.shape[0])
    Wp_e8_1, info_e8_1 = fold_fit(Wf, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, e8_code=True)
    Wp_e8_2, _ = fold_fit(Wf - Wp_e8_1, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, e8_code=True)
    Wp_b2_s, _ = fold_fit(Wf - Wp_b1, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, e8_code=False)
    S_rot_tup = info_e8_1["S_rot"].view(-1, 8)
    out["c20_e8_stored_tuple_max_dist_fp32"] = float(((S_rot_tup[:, None, :] - C_e8_c[None, :, :]) ** 2).sum(-1).min(dim=-1).values.max())
    out["c20_e8_1plane_j_ratio_fp32"] = float(J(Wf, Wp_e8_1, Hl32) / J(Wf, Wp_b1, Hl32))
    out["c20_e8_2plane_j_ratio_fp32"] = float(J(Wf, Wp_e8_1 + Wp_e8_2, Hl32) / J(Wf, Wp_b1 + Wp_b2_s, Hl32))
    # c21: Face Swap #3 — Delayed-Decision Beam-4 Block-Cholesky Trajectory Search over E8 U O8 (§16)
    Wp_e8b_1, info_e8b_1 = fold_fit(Wf, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, e8_code=True, e8_beam=4)
    Wp_e8b_2, _ = fold_fit(Wf - Wp_e8b_1, Hf32, 2, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, e8_code=True, e8_beam=4)
    out["c21_e8_beam4_vs_e8_1plane_j_ratio_fp32"] = float(J(Wf, Wp_e8b_1, Hl32) / J(Wf, Wp_e8_1, Hl32))
    out["c21_e8_beam4_vs_e8_2plane_j_ratio_fp32"] = float(J(Wf, Wp_e8b_1 + Wp_e8b_2, Hl32) / J(Wf, Wp_e8_1 + Wp_e8_2, Hl32))
    # c22: Face Swap #4 — Global Elliptic 8-tuple ICM coordinate descent over E8 U O8 (§17)
    Wp_icm_1, info_icm_1 = fold_fit(Wf, Hf32, 1, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, e8_code=True, e8_beam=4, e8_icm=True, icm_ridge=0.15, inner_iters=3)
    Wp_icm_2, _ = fold_fit(Wf - Wp_icm_1, Hf32, 1, 1e-3, refit=True, had=True, act_order=True, wscale=True, hier_scale=True, bscale=True, e8_code=True, e8_beam=4, e8_icm=True, icm_ridge=0.15, inner_iters=3)
    out["c22_e8_icm_vs_beam4_1plane_j_ratio_fp32"] = float(J(Wf, Wp_icm_1, Hl32) / J(Wf, Wp_e8b_1, Hl32))
    out["c22_e8_icm_vs_beam4_2plane_j_ratio_fp32"] = float(J(Wf, Wp_icm_1 + Wp_icm_2, Hl32) / J(Wf, Wp_e8b_1 + Wp_e8b_2, Hl32))
    # c23: Face Swap #5 — Stage-1 expansion drift attenuation dominance over gamma=1 and gamma=0 (§18)
    dW_sig = 0.15 * torch.randn(Ws.shape[0], n, generator=g, dtype=F64).to(dev_c)
    dW_est = dW_sig + 0.26 * torch.randn(Ws.shape[0], n, generator=g, dtype=F64).to(dev_c)
    W_true_s1 = Ws + dW_sig
    j_g100 = J(W_true_s1, Ws + 1.00 * dW_est, Hl)
    j_g000 = J(W_true_s1, Ws + 0.00 * dW_est, Hl)
    j_g025 = J(W_true_s1, Ws + 0.25 * dW_est, Hl)
    out["c23_stage1_gin25_vs_gin100_j_ratio_f64"] = float(j_g025 / j_g100)
    out["c23_stage1_gin25_vs_gin000_j_ratio_f64"] = float(j_g025 / j_g000)
    # c24: Face Swap #6 — Hyperbolic bilinear SwiGLU compensation dominance over uncompensated up_proj (§19)
    W_g_syn = torch.randn(16, n, generator=g, dtype=F64).to(dev_c) / (n ** 0.5)
    W_u_syn = torch.randn(16, n, generator=g, dtype=F64).to(dev_c) / (n ** 0.5)
    Wq_g_syn, _, _ = gptq_signs(W_g_syn, Hd)
    s0_syn = F.silu(Xd @ W_g_syn.T)
    sq_syn = F.silu(Xd @ Wq_g_syn.T)
    u0_syn = Xd @ W_u_syn.T
    du_syn = (sq_syn * (s0_syn - sq_syn) / (sq_syn ** 2 + 0.05)) * u0_syn
    W_u_comp = W_u_syn + ridge_solve(Hd, Xd.T @ (0.75 * du_syn), 0.045)
    m0_syn = s0_syn * u0_syn
    m_uncomp = sq_syn * u0_syn
    m_comp = sq_syn * (Xd @ W_u_comp.T)
    out["c24_swiglu_bilinear_comp_j_ratio_f64"] = float(((m_comp - m0_syn) ** 2).mean() / ((m_uncomp - m0_syn) ** 2).mean())
    # c25: Face Swap #7 — Riemannian RMSNorm pullback sink gating + row trust dominance (§20)
    X_s = Xd.clone(); X_s[0] *= 14.333333333333334
    Y_s = X_s @ Ws.T; Y_s[0] += 8.0 * X_s[0, :Ws.shape[0]]
    w_s = torch.ones(X_s.shape[0], 1, dtype=F64, device=dev_c); w_s[0] = 120.0 / 1720.0
    Xw_s = X_s * w_s
    W_unw = Ws + ridge_solve(X_s.T @ X_s, X_s.T @ (Y_s - X_s @ Ws.T), 0.025)
    W_wgt = Ws + ridge_solve(Xw_s.T @ Xw_s, Xw_s.T @ ((Y_s - X_s @ Ws.T) * w_s), 0.025)
    out["c25_rmsnorm_pullback_sink_weight_f64"] = float(w_s[0].item())
    out["c25_rmsnorm_pullback_sink_j_ratio_f64"] = float(J(Ws, W_wgt, Hl) / J(Ws, W_unw, Hl))
    return out


# ---------------------------------------------------------------- outcome metrics (true-label, task, generated text)
EVAL_DIR = "~/evaldata"
GEN_PROMPTS = ["The capital of France is", "Once upon a time, in a small village by the sea,", "def fibonacci(n):",
               "The three laws of thermodynamics state that", "Q: What is the boiling point of water in degrees Celsius?\nA:",
               "In 1969, the Apollo 11 mission"]
_GEN_REF = {}


def _jsonl(path, n=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return rows if n is None else rows[:n]


def _hs_clean(t):
    t = t.strip().replace(" [title]", ". ")
    return re.sub(r"  ", " ", re.sub(r"\[.*?\]", "", t))


@torch.no_grad()
def _fwd(cfg, Lg, fn, E_g, ids):
    h = E_g[ids]
    for L in Lg:
        h = parts(cfg, L, h)["out"]
    return rms(h, fn, cfg.eps)


@torch.no_grad()
def score_seqs(cfg, Lg, fn, E_g, seqs, tok_budget=8192):
    """Sum log p of the last n_cont tokens of each (ids, n_cont) in `seqs` (teacher-forced, fp32 forward on the
    resident layers). Cost: one forward per padded batch, O(sum_i len_i) tokens x (linear + O(len) attention) FLOPs;
    the vocab matmul runs on the n_cont rows only. Padding is on the right, so causal rows are unaffected."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out = [0.0] * len(seqs)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * len(seqs[order[j]][0]) <= tok_budget:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        mx = max(len(seqs[k][0]) for k in idx)
        ids = torch.zeros(len(idx), mx, dtype=torch.long)
        for r, k in enumerate(idx):
            ids[r, :len(seqs[k][0])] = torch.tensor(seqs[k][0])
        ids = ids.to(DEV)
        h = _fwd(cfg, Lg, fn, E_g, ids)
        rows, cols, tgt, owner = [], [], [], []
        for r, k in enumerate(idx):
            s, nc = seqs[k]
            for t in range(len(s) - nc, len(s)):
                rows.append(r); cols.append(t - 1); tgt.append(s[t]); owner.append(k)
        rows_t, cols_t, tgt_t = (torch.tensor(x, device=DEV) for x in (rows, cols, tgt))
        hs = h[rows_t, cols_t]
        for c0 in range(0, hs.shape[0], 1024):
            lp = F.log_softmax((hs[c0:c0 + 1024] @ E_g.T).float(), -1)
            v = lp.gather(1, tgt_t[c0:c0 + 1024, None])[:, 0].cpu().tolist()
            for q, val in enumerate(v):
                out[owner[c0 + q]] += val
        i = j
    return out


@torch.no_grad()
def greedy_text(cfg, Lg, fn, E_g, ids, n_new):
    """Greedy continuation, no KV cache: n_new forwards of length <= len(ids)+n_new (O(n_new * L^2) attention)."""
    s = list(ids)
    for _ in range(n_new):
        h = _fwd(cfg, Lg, fn, E_g, torch.tensor([s], device=DEV))
        s.append(int((h[0, -1] @ E_g.T).argmax().item()))
    return s[len(ids):]


def outcome_eval(cfg, Ls, tok, ids_ev, n_hs, n_new, ref_key=None):
    t0 = tick()
    Lg = [{k: v.to(DEV) for k, v in L.items() if k != "fnorm"} for L in Ls]
    fn = (Ls[-1].get("fnorm", cfg.fnorm) if Ls else cfg.fnorm).to(DEV)
    E_g = cfg.E.to(DEV)
    o = {}
    # C4 held-out true-label NLL (the same chunks kl_eval scores against the teacher)
    sc = score_seqs(cfg, Lg, fn, E_g, [(r.tolist(), CHUNK - 1) for r in ids_ev])
    n = ids_ev.shape[0] * (CHUNK - 1)
    o["c4_nll"] = -sum(sc) / n; o["c4_ppl"] = float(np.exp(o["c4_nll"])); o["c4_tokens"] = n
    # WikiText-2 test, HF-style: "\n\n".join(rows), non-overlapping windows of CHUNK tokens, first token of a window unscored
    wt = tok.encode("\n\n".join(r["text"] for r in _jsonl(f"{EVAL_DIR}/wt2/test.jsonl")))
    wins = [(wt[i:i + CHUNK], CHUNK - 1) for i in range(0, len(wt) - CHUNK + 1, CHUNK)]
    sc = score_seqs(cfg, Lg, fn, E_g, wins)
    n = len(wins) * (CHUNK - 1)
    o["wt2_nll"] = -sum(sc) / n; o["wt2_ppl"] = float(np.exp(o["wt2_nll"])); o["wt2_tokens"] = n; o["wt2_window"] = CHUNK
    # multiple choice by continuation log-likelihood (lm-evaluation-harness conventions: acc = argmax sum log p,
    # acc_norm = argmax sum log p / len(choice in characters))
    def mc(items):
        seqs, meta = [], []
        for ctx, conts, gold in items:
            ce = tok.encode(ctx)
            for c in conts:
                we = tok.encode(ctx + c)
                nc = len(we) - len(ce)
                assert nc >= 1 and we[:len(ce)] == ce, (ctx, c)
                seqs.append((we, nc)); meta.append(len(c))
        s = score_seqs(cfg, Lg, fn, E_g, seqs)
        a = an = p = 0
        for q, (ctx, conts, gold) in enumerate(items):
            ll = s[p:p + len(conts)]; ln = [ll[m] / meta[p + m] for m in range(len(conts))]
            a += int(max(range(len(conts)), key=lambda m: ll[m]) == gold)
            an += int(max(range(len(conts)), key=lambda m: ln[m]) == gold)
            p += len(conts)
        return a / len(items), an / len(items), len(items)
    arc = _jsonl(f"{EVAL_DIR}/arc/test.jsonl")
    o["arc_acc"], o["arc_acc_norm"], o["arc_n"] = mc([(f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                                                       r["choices"]["label"].index(r["answerKey"])) for r in arc])
    hs = _jsonl(f"{EVAL_DIR}/hs/val.jsonl", n_hs)
    o["hs_acc"], o["hs_acc_norm"], o["hs_n"] = mc([(_hs_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()),
                                                    [" " + _hs_clean(e) for e in r["endings"]], int(r["label"])) for r in hs])
    gen = []
    for p in GEN_PROMPTS:
        pe = tok.encode(p)
        g = greedy_text(cfg, Lg, fn, E_g, pe, n_new)
        ref = _GEN_REF.get(p)
        gen.append(dict(prompt=p, text=tok.decode(g), tokens=g,
                        first_divergence=(None if ref is None else next((i for i, (x, y) in enumerate(zip(g, ref)) if x != y), len(g)))))
        if ref_key == "teacher":
            _GEN_REF[p] = g
    o["generation"] = gen; o["gen_new_tokens"] = n_new
    o["wall_outcomes_s"] = tick() - t0
    del Lg, E_g
    torch.cuda.empty_cache()
    return o


def yaqa_controls():
    """Synthetic fp64 checks of yaqa_refine (cuda when available), no model: O = 64, n = 256 = 2 * GROUP, Hl = X^T X / N
    + 1e-3 I, HO = U U^T + 0.05 I with U of rank 4, and an initial single-plane P (and a two-plane pair) from the
    module's own fold_fit path.  (a) the incremental dJ of _yaqa_dJ equals the full J_O(after) - J_O(before) recomputed
    from the objective, rel err < 1e-9; (b) JO_after <= JO_before always and, at chunk 16, JO_after < 0.5 * JO_before with
    accepted >= 1 for the single- and the two-plane case; (c) drop(shift_sign=+1) > drop(shift_sign=-1)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device="cpu").manual_seed(11)
    gd = gen if dev == "cpu" else torch.Generator(device=dev).manual_seed(11)
    O, n, N = 64, 256, 4096
    Ws = (torch.randn(O, n, generator=gen, dtype=F64) * 0.5).to(dev)
    X = torch.randn(N, n, generator=gen, dtype=F64).to(dev)
    Hl = X.T @ X / N + 1e-3 * torch.eye(n, dtype=F64, device=dev)
    U = torch.randn(O, 4, generator=gen, dtype=F64).to(dev) / 2.0
    HO = U @ U.T + 0.05 * torch.eye(O, dtype=F64, device=dev)
    fit = lambda W_: fold_fit(W_, Hl, 2, 0.0, refit=True, had=True, damp=0.01, bscale=True, inner_iters=2)
    _, i1 = fit(Ws)
    P1 = i1["pack"]
    _, i2 = fit(Ws - _plane_w(P1))
    P2 = i2["pack"]
    assert P1 is not None and P2 is not None and P1[0].shape == (O, n), (P1, P2)
    out = dict(O=O, n=n, device=dev, group=GROUP)
    # (a) exact incremental dJ, on one chunk of rows and a random delta the size of the residual
    D = Ws - _plane_w(P1)
    Cs = slice(0, 16)
    delta = (torch.rand(16, n, generator=gd, dtype=F64, device=dev) - 0.5) * D.abs().amax()
    d_inc = _yaqa_dJ(HO, Hl, D, Cs, delta)
    D2 = D.clone(); D2[Cs] += delta
    j_ref = _yaqa_JO(HO, Hl, D)
    d_full = _yaqa_JO(HO, Hl, D2) - j_ref
    out["a"] = dict(d_inc=d_inc, d_full=d_full,
                    rel_err=abs(d_inc - d_full) / max(abs(d_full), 1e-6 * abs(j_ref)))
    assert out["a"]["rel_err"] < 1e-9, out["a"]
    # (b) both plane counts: no regression ever, and at chunk 16 a drop above half of J_O.  The block target
    # E_C = HO_CC^-1 HO[C,notC] D_notC is what makes chunk 16 move at all: the per-row (block-Jacobi) shift measured
    # 42.7% single-plane and 0% (nothing accepted) two-plane here, against 72.7% and 82.3% for the block target.
    CH = 16
    out["chunk"] = CH
    for tag, P2i in (("single", None), ("two", P2)):
        _, _, lg = yaqa_refine(Ws, Hl, HO, P1, P2i, icm_ridge=0.0, chunk=CH, sweeps=2, rounds=1)
        drop = lg["JO_before"] - lg["JO_after"]
        out[tag] = dict(JO_before=lg["JO_before"], JO_after=lg["JO_after"], drop=drop, accepted=lg["accepted"],
                        chunks=lg["chunks"], rows_moved=lg["rows_moved"], ratio=lg["JO_after"] / lg["JO_before"])
        print(f"yaqa {tag}: JO {lg['JO_before']:.6g} -> {lg['JO_after']:.6g}  drop {drop:.6g} "
              f"({100.0 * drop / lg['JO_before']:.3f}%)  accepted {lg['accepted']}/{lg['chunks']} rows_moved {lg['rows_moved']}", flush=True)
        assert lg["JO_after"] <= lg["JO_before"], out[tag]
        assert lg["JO_after"] < 0.5 * lg["JO_before"], out[tag]
        assert lg["accepted"] >= 1, out[tag]
    # (c) sign control: the wrong-sign shift must lower J_O less
    _, _, lgp = yaqa_refine(Ws, Hl, HO, P1, P2, icm_ridge=0.0, chunk=CH, sweeps=2, rounds=1, shift_sign=1.0)
    _, _, lgm = yaqa_refine(Ws, Hl, HO, P1, P2, icm_ridge=0.0, chunk=CH, sweeps=2, rounds=1, shift_sign=-1.0)
    drop_p = lgp["JO_before"] - lgp["JO_after"]
    drop_m = lgm["JO_before"] - lgm["JO_after"]
    out["c"] = dict(drop_plus=drop_p, drop_minus=drop_m, JO_minus=lgm["JO_after"], JO_before=lgm["JO_before"])
    print(f"yaqa sign: drop(+1) {drop_p:.6g} vs drop(-1) {drop_m:.6g}", flush=True)
    assert drop_p > drop_m, out["c"]
    # (d) Schafer-Strimmer (2005) intensity estimator via the module's own lw_intensity: (a) the closed-form num equals
    # the brute-force sum over i != j of the per-entry sample variance of the mean of s_ti s_tj; (b) an unequal-variance
    # diagonal population gives lambda > 0.5; (c) a strongly correlated population gives lambda < 0.05.  float64, CPU.
    gd = torch.Generator(device="cpu").manual_seed(23)
    Olw, n_a = 12, 400
    Sa = torch.randn(n_a, Olw, generator=gd, dtype=F64)
    Ha = Sa.T @ Sa
    num_a, _ = lw_intensity(float((Sa ** 2).sum(-1).pow(2).sum()), (Sa ** 4).sum(0), Ha, n_a)
    Ma = Ha / n_a
    W2 = Sa[:, :, None] * Sa[:, None, :]
    varmean = ((W2 ** 2).sum(0) - n_a * Ma ** 2) / (n_a * (n_a - 1))
    bf = float(varmean[~torch.eye(Olw, dtype=torch.bool)].sum())
    rel_a = abs(num_a - bf) / max(abs(bf), 1e-300)
    assert rel_a < 1e-10, (num_a, bf, rel_a)
    n_lw = 20000
    Sd = torch.randn(n_lw, Olw, generator=gd, dtype=F64) * torch.arange(1, Olw + 1, dtype=F64)
    _, lam_b = lw_intensity(float((Sd ** 2).sum(-1).pow(2).sum()), (Sd ** 4).sum(0), Sd.T @ Sd, n_lw)
    assert lam_b > 0.5, lam_b
    Lc = torch.linalg.cholesky(0.9 * torch.ones(Olw, Olw, dtype=F64) + 0.1 * torch.eye(Olw, dtype=F64))
    Sc = torch.randn(n_lw, Olw, generator=gd, dtype=F64) @ Lc.T
    _, lam_c = lw_intensity(float((Sc ** 2).sum(-1).pow(2).sum()), (Sc ** 4).sum(0), Sc.T @ Sc, n_lw)
    assert lam_c < 0.05, lam_c
    out["lw"] = dict(a=rel_a, b=lam_b, c=lam_c)
    print(f"yaqa_lw a={rel_a:.3g} b={lam_b:.6g} c={lam_c:.6g}", flush=True)
    # (e) batch-means (sequence-unit) intensity via the module's own _block_off / lw_block_intensity: (a) the K^2 Gram
    # route's P equals the P formed from every H_b directly, and its num equals the brute-force sum over i != j of
    # ((sum_b M_b,ij^2) - B M_ij^2) / (B (B - 1)); (b) an iid population gives lambda_block within 0.25 of
    # lambda_iid; (c) a within-block correlated population gives lambda_block > 5 * lambda_iid.  float64, CPU.
    Ob, Bb, Kb, Tb = 10, 30, 2, 16
    nb, nSb = Kb * Tb, Bb * Kb * Tb
    Sb = torch.randn(Bb, Kb, Tb, Ob, generator=gd, dtype=F64)          # (block, draw, sample, O)
    Hb = torch.stack([b.reshape(nb, Ob).T @ b.reshape(nb, Ob) for b in Sb])         # (B, O, O), the O x O route
    Pb_dir = float(sum((h * h).sum() - (h.diagonal() ** 2).sum() for h in Hb)) / nb ** 2
    Pb_gr = float(sum(_block_off([b[k] for k in range(Kb)]) for b in Sb)) / nb ** 2
    Mb = Hb / nb
    bf = float((((Mb ** 2).sum(0) - Bb * Mb.mean(0) ** 2) / (Bb * (Bb - 1)))[~torch.eye(Ob, dtype=torch.bool)].sum())
    Ht = Sb.reshape(-1, Ob).T @ Sb.reshape(-1, Ob)
    num_gr, _ = lw_block_intensity(Pb_gr, Bb, Ht, nSb)
    rel_P = abs(Pb_gr - Pb_dir) / max(abs(Pb_dir), 1e-300)
    rel_num = abs(num_gr - bf) / max(abs(bf), 1e-300)
    assert rel_P < 1e-10, (Pb_gr, Pb_dir, rel_P)
    assert rel_num < 1e-10, (num_gr, bf, rel_num)
    Ob2, Bb2, Kb2, Tb2 = 10, 200, 2, 32
    nb2, n2 = Kb2 * Tb2, Bb2 * Kb2 * Tb2
    Sd2 = torch.randn(Bb2, Kb2, Tb2, Ob2, generator=gd, dtype=F64) * torch.arange(1, Ob2 + 1, dtype=F64).sqrt()
    H2 = Sd2.reshape(-1, Ob2).T @ Sd2.reshape(-1, Ob2)
    P2 = float(sum(_block_off([b[k] for k in range(Kb2)]) for b in Sd2)) / nb2 ** 2
    li = lw_intensity(float((Sd2.reshape(-1, Ob2) ** 2).sum(-1).pow(2).sum()), (Sd2.reshape(-1, Ob2) ** 4).sum(0), H2, n2)[1]
    lb = lw_block_intensity(P2, Bb2, H2, n2)[1]
    out["lb"] = dict(a=dict(rel_P=rel_P, rel_num=rel_num, P=P2), b=dict(lb=lb, li=li))
    assert abs(lb - li) < 0.25, (lb, li)
    Lc2 = torch.linalg.cholesky(0.9 * torch.ones(Ob2, Ob2, dtype=F64) + 0.1 * torch.eye(Ob2, dtype=F64))
    S3 = (torch.randn(Bb2, Ob2, generator=gd, dtype=F64) @ Lc2.T)[:, None, None, :] \
        + 0.1 * torch.randn(Bb2, Kb2, Tb2, Ob2, generator=gd, dtype=F64)
    H3 = S3.reshape(-1, Ob2).T @ S3.reshape(-1, Ob2)
    P3 = float(sum(_block_off([b[k] for k in range(Kb2)]) for b in S3)) / nb2 ** 2
    li3 = lw_intensity(float((S3.reshape(-1, Ob2) ** 2).sum(-1).pow(2).sum()), (S3.reshape(-1, Ob2) ** 4).sum(0), H3, n2)[1]
    lb3 = lw_block_intensity(P3, Bb2, H3, n2)[1]
    out["lb"].update(c=dict(lb=lb3, li=li3, ratio=(lb3 / li3 if li3 else float("inf"))))
    assert lb3 > 5 * li3, (lb3, li3)
    print(f"yaqa_lb a={rel_P:.3g},{rel_num:.3g} b={lb:.6g}/{li:.6g} c={lb3:.6g}/{li3:.6g}", flush=True)
    # spec_stats: (a) identical draft accepts every token and emits k + 1 per block; (b) p = (1/2, 1/2), q = (3/4, 1/4)
    # at every position accepts 3/4 and emits the geometric sum (1 - (3/4)^(k+1)) / (1/4).
    lt = torch.log_softmax(torch.randn(2, 20, 7, dtype=torch.float64, generator=gen), -1)
    a_s, a_n, ek, e_n = spec_stats(lt, lt)
    rel_a = max(abs(a_s / a_n - 1), max(abs(x / e_n - (k + 1)) / (k + 1) for k, x in enumerate(ek, 1)))
    p2 = torch.tensor([0.5, 0.5], dtype=torch.float64).log().expand(2, 20, 2)
    q2 = torch.tensor([0.75, 0.25], dtype=torch.float64).log().expand(2, 20, 2)
    a_s, a_n, ek, e_n = spec_stats(p2, q2)
    rel_b = max(abs(a_s / a_n - 0.75) / 0.75,
                max(abs(x / e_n - (1 - 0.75 ** (k + 1)) / 0.25) / ((1 - 0.75 ** (k + 1)) / 0.25) for k, x in enumerate(ek, 1)))
    out["spec"] = dict(rel_identical=rel_a, rel_geometric=rel_b)
    assert rel_a < 1e-12 and rel_b < 1e-12, (rel_a, rel_b)
    print(f"spec_stats a={rel_a:.3g} b={rel_b:.3g}", flush=True)
    return out


def main():
    global MODEL, GGUF_Q4KM
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--variants", default="")
    ap.add_argument("--dump", default=None, help="Path to dump quantized layer weights for text_probe.py")
    ap.add_argument("--controls", action="store_true"); ap.add_argument("--gq-controls", action="store_true"); ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--yaqa-controls", action="store_true", help="synthetic fp64 checks of yaqa_refine, then exit (no model)")
    ap.add_argument("--outcomes", action="store_true", help="true-label NLL, WikiText-2 ppl, ARC-Easy/HellaSwag acc, greedy text (first seed)")
    ap.add_argument("--hs-n", type=int, default=1000); ap.add_argument("--gen-new", type=int, default=48)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp64"])
    ap.add_argument("--model", default=MODEL); ap.add_argument("--q4km", default=None, help="Q4_K_M GGUF of --model")
    ap.add_argument("--seeds", default="0", help="comma list; seed s permutes the calibration chunks (identity for 0)")
    for k, v in [("layers", 24), ("iters", 2), ("n-cal", 100), ("n-eval", 40), ("n-in", 20), ("slab", 10), ("cal-extra", 0), ("qmlp-rows", 416), ("gq-k", 4), ("gq-seed", 0)]:
        ap.add_argument(f"--{k}", type=int, default=v)
    for k, v in [("ridge", 1e-3), ("ridge-w0", 0.01), ("ridge-w0-gqa", 0.03), ("quant-ridge", 0.005), ("quant-damp", 0.02),
                 ("zellner-gamma", 0.85), ("corr-gamma", 0.75), ("ridge-intra", 0.045), ("ridge-inter", 0.025),
                 ("inter-gamma", 1.0), ("inter-gamma-in", 1.0), ("block-alpha", 1.0),
                 ("glu-gamma", 0.75), ("glu-mu", 0.05), ("glu-ridge", 0.045),
                 ("sink-thresh", 5.0), ("sink-tau", 1.5), ("fnorm-lr", 0.15)]:
        ap.add_argument(f"--{k}", type=float, default=v)
    a = ap.parse_args()
    if a.yaqa_controls:
        ctl = yaqa_controls()
        res = dict(script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), args=vars(a),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", yaqa_controls=ctl)
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(ctl, indent=1), flush=True)
        sys.exit(0)
    torch.manual_seed(0)
    t00 = tick()
    MODEL, GGUF_Q4KM = a.model, a.q4km or GGUF_Q4KM
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    ids = tok.encode(open(CALIB, encoding="utf-8").read())
    nch = len(ids) // CHUNK
    assert nch >= a.n_cal + a.n_eval, (nch, a.n_cal, a.n_eval)
    ids_all = torch.tensor(ids[:nch * CHUNK], device="cpu").view(nch, CHUNK)
    cfg, TL, hf = load_model()
    hf = hf.eval()
    res = dict(script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), args=vars(a),
               n_tokens=len(ids), n_chunks=nch, cal_chunks=[0, a.n_cal], eval_chunks=[a.n_cal, a.n_cal + a.n_eval],
               insample_chunks=[0, a.n_in], gpu=torch.cuda.get_device_name(0))
    ids_cal, ids_ev, ids_in = ids_all[:a.n_cal].contiguous(), ids_all[a.n_cal:a.n_cal + a.n_eval], ids_all[:a.n_in]
    if a.cal_extra:
        ex = np.load(CAL_EXTRA_NPY)[:a.cal_extra * CHUNK]
        assert len(ex) == a.cal_extra * CHUNK, (len(ex), a.cal_extra)
        ids_cal = torch.cat([ids_cal, torch.from_numpy(ex.astype(np.int64)).view(a.cal_extra, CHUNK)]).contiguous()
        res.update(cal_extra_sha256=hashlib.sha256(open(CAL_EXTRA_NPY, "rb").read()).hexdigest(),
                   cal_extra_chunks=[0, a.cal_extra], cal_tokens=ids_cal.numel())
    if a.controls:
        res["controls"] = controls(cfg, TL, hf, ids_all)
        print(json.dumps(res["controls"], indent=1), flush=True)
    del hf
    torch.cuda.empty_cache()
    if a.gq_controls:
        res["gq_controls"] = gq_controls(cfg, TL, ids_all, a)
        print("gq_controls", json.dumps(res["gq_controls"]), flush=True)
    res["variants"] = {}
    n_lin = sum(TL[l][k].numel() for l in range(cfg.nl) for k in KINDS)
    res["linear_params"] = n_lin
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    for name in [v for v in a.variants.split(",") if v]:
        runs = []
        for sd in seeds:
            t0 = tick()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(sd)
            ids_c = ids_cal if sd == 0 else ids_cal[torch.randperm(ids_cal.shape[0], generator=torch.Generator().manual_seed(sd))]
            if name == "teacher":
                Ls, info, tm = TL, dict(bpw_linears=32.0), {}
            elif name in ("q4km", "gguf-f16"):
                Ls, bpw = gguf_layers(cfg, TL, GGUF_Q4KM if name == "q4km" else GGUF_F16)
                info, tm = dict(bpw_linears=bpw), {}
            else:
                Ls, binfo, tm = run_variant(cfg, TL, ids_c, name, a)
                info = dict(binfo)
                if binfo["params"]:
                    info["bpw_linears"] = binfo["bits"] / binfo["params"]
            tq = tick() - t0
            kl, ag = kl_eval(cfg, TL, Ls, ids_ev)
            kli, agi = kl_eval(cfg, TL, Ls, ids_in)
            info.update(kl_heldout=kl, top1_heldout=ag, kl_insample=kli, top1_insample=agi, wall_transfer_s=tq,
                        timings=tm, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30, seed=sd)
            if a.outcomes and sd == seeds[0]:
                info["outcomes"] = outcome_eval(cfg, Ls, tok, ids_ev, a.hs_n, a.gen_new, "teacher" if name == "teacher" else None)
            runs.append(info)
            print(name, json.dumps(info), flush=True)
            if a.dump and sd == seeds[0]:
                torch.save({"layers": [{k: v.cpu() for k, v in L.items()} for L in Ls], "fnorm": cfg.fnorm.cpu()}, a.dump)
            if name != "teacher":
                del Ls
            torch.cuda.empty_cache()
        info = dict(runs[0])
        if len(runs) > 1:
            info["seeds"] = {str(r["seed"]): {k: r[k] for k in ("kl_heldout", "top1_heldout", "kl_insample", "top1_insample", "wall_transfer_s", "peak_gpu_gb")} for r in runs}
            for k in ("kl_heldout", "top1_heldout", "kl_insample"):
                xs = [r[k] for r in runs]
                info[k + "_stats"] = dict(mean=sum(xs) / len(xs), min=min(xs), max=max(xs))
        res["variants"][name] = info
    res["wall_total_s"] = tick() - t00
    json.dump(res, open(a.out, "w"), indent=1)
    print("wall_total_s", res["wall_total_s"])


if __name__ == "__main__":
    main()
