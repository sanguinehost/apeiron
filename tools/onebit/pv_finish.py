"""ONEBIT PV-Tuning-style end-to-end KL finishing of the 2-plane E8 -resq arm of fold_transfer (Qwen2.5-0.5B).

Arm: had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-resq-qmlp-nb (fold_transfer.run_variant, PACK_SINK set).
Stored weight of one linear = sum over planes of compose(fwht128(S_rot * d), r, c) (+ one fixed dense corrector plane on q/down),
S_rot a field of 8-blocks, each an index into the 256-row E8/O8 codebook; r a 16-level per-row grid (6-bit mu + 2-bit spread per row,
4-bit k per group), c a 6-bit and d a 5-bit affine column grid.  Finishing (Malinovskii et al., arXiv 2405.14852):
  P-step  Adam on continuous latents (log-multiplicative r, c, d; STE through the arm's quantizers) + zero-bit tunables
          (ln1, ln2, q/k/v biases, final norm), loss forward KL(teacher || student) on C4 train text;
  V-step  every --v-every steps, dL/dS per plane, per 8-block the best of the 16 nearest codebook rows by the linearised gain
          <g_b, s_b - C_i>, the top --v-frac of blocks per plane with gain > 0 move (reverted, v_frac halved, if the same-batch loss rises).
Eval after on the same chunks as eval before, with weights rebuilt from the integer stored codes.
--v-gain obs replaces the one-batch linear gain by the Optimal Brain Surgeon diagonal gain from window statistics of the P-steps'
batch gradients (gbar, diagonal empirical Fisher F), candidates lam in --obs-lams (default {1,4,16,64}) x fraction in {0.001,0.01} of positive-gain blocks.
--codes ste replaces the V-step: each coded plane keeps an fp32 latent L (initialised to the stored codewords), the forward uses the
nearest of the 16 nearest codewords of the current index (straight-through gradient to L), and the index follows that argmin; the
stored code stays one 8-bit index per 8-block, so the bits are unchanged.
--code-init preimage starts L at the continuous pre-image P of the plane (the plane's share of the teacher weight, mapped back through
the quantized r, c, d and the involution fwht128), shrunk along C[idx] -> P until the nearest-of-16 projection still returns idx
(step 0 forward is the stored code); --code-opt nsgd moves L by lr * g / rms(g), rms over the whole plane's latent gradient.
--rotation kan (needs --compose step) replaces fwht128 in the plane weight with the learned register of kan_probe (seven stages of
64 reflections, angle offset d = theta - pi/4 per (layer, site) of shape (7, 64), init 0 = fwht128 bit for bit (unscaled stages, one GROUP**-0.5 at the end), shared by every coded plane of the site's linears) trained end to end
on KL by RegisterCompose; the final weights use d quantised to 16 bits (the theta grid), 7*64*16 bits per learning site per layer added to bits_after.
--train corr trains each dense corrector plane (p.dense, added on q and down) through its stored factors: p.U, p.V start
as the u8 codes lowrank_corrector stores (pack sink U, V; p.dense = U @ V), train in their own Adam group at --lr-corr,
and every add of the plane uses corr_weight(p) = u8(U) @ u8(V), each factor vector coded as lowrank_corrector's u8 branch
codes it (fp16 min/max, 8-bit levels), so step 0 is the stored plane exactly; the export stores that product, so the
stored bits are unchanged (bits_corr).  With corr off the dense plane stays frozen and every code path is today's.
--selftest-kan runs the float64 register/adjoint checks with no model.
--attrib N (default 0 = off; every other code path is unchanged) uses the first N held-out eval chunks and, once before and once after finishing, restores the teacher's linear weights one group at a time (each layer's seven linears, each kind across all layers, all linears) to record where the held-out KL lives: base, per-layer, per-kind and all-linear KL, their deltas, and a teacher-as-student zero control.  It also restores the student's linears back over the all-linear configuration in each single layer (iso), each prefix of layers (prefix) and each suffix of layers (suffix), recording the KL each costs in isolation; set against base - all_linear, the iso-cost sum and the prefix/suffix costs show whether the KL of a layer set is more or less than the sum of its layers' isolated costs.
--block-s S (default 0.0 = off; needs --codes vstep, --v-every 0, --v-gain linear, --compose step, --rotation had) makes the train loop start with
S seconds of AQLM phase-3 block-wise fine-tuning (Egiazarian et al., arXiv 2401.06118, Sec. 3.4): each code block fits its relative squared residual on
the teacher's own hidden-state input, then the Adam state is cleared and the warmup restarts for the KL phase; the stored bits are unchanged.
usage: pv_finish.py --out OUT.json [--smoke] [--v-gain {linear,obs}] [--codes {vstep,ste}] ; run with ~/spinor-ttl/.venv/bin/python (torch + CUDA).
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_transfer as ft
import kan_probe as kp
import transformers

DEV, GROUP, CHUNK = ft.DEV, ft.GROUP, ft.CHUNK
KINDS = list(ft.KINDS)
ARM = "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-resq-qmlp-nb"
TRAIN_NPY = "~/lanes/w18/kd/c4data/train-c4.qwen.npy"
REF_RECEIPT = "~/apeiron/the receipts/onebit/w18-size05-n50 note.json"
REF_KL_HELDOUT = 1.1636724948883057
BUDGET_S = 580.0
OBS_FRACS = (0.001, 0.01)


def proc_start():
    """Unix time the process started (kernel start ticks + boot time), so the 580 s `timeout` is measured from the same origin."""
    clk = os.sysconf("SC_CLK_TCK")
    st = int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19]) / clk
    btime = [int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime")][0]
    return btime + st


T0 = proc_start()


def elapsed():
    return time.time() - T0


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def arm_args(n_cal, n_eval, n_in, out):
    """The argparse namespace ft.main builds for this arm (defaults of ft.main)."""
    d = dict(out=out, variants=ARM, dump=None, controls=False, verbose=False, outcomes=False, hs_n=1000, gen_new=48, dtype="fp32",
             model="Qwen/Qwen2.5-0.5B", q4km=None, seeds="0", layers=24, iters=2, n_cal=n_cal, n_eval=n_eval, n_in=n_in, slab=10,
             cal_extra=0, qmlp_rows=416, ridge=1e-3, ridge_w0=0.01, ridge_w0_gqa=0.03, quant_ridge=0.005, quant_damp=0.02,
             zellner_gamma=0.85, corr_gamma=0.75, ridge_intra=0.045, ridge_inter=0.025, inter_gamma=1.0, inter_gamma_in=1.0,
             block_alpha=1.0, glu_gamma=0.75, glu_mu=0.05, glu_ridge=0.045, sink_thresh=5.0, sink_tau=1.5, fnorm_lr=0.15)
    return argparse.Namespace(**d)


# ------------------------------------------------------------------------------------------------ codes
@torch.no_grad()
def encode(S, C, chunk=1 << 19):
    """Nearest codebook row per 8-block of S (uint8 (O, n/8)) and whether every block equals its row exactly.
    Temporal 2*256*8*M MACs, M = O*n/8 blocks; spatial chunk*256 words (matmul workspace)."""
    B = S.reshape(-1, 8)
    cn = (C * C).sum(1)
    out = torch.empty(B.shape[0], dtype=torch.uint8, device=S.device)
    for s0 in range(0, B.shape[0], chunk):
        out[s0:s0 + chunk] = (2.0 * B[s0:s0 + chunk] @ C.T - cn[None, :]).argmax(1).to(torch.uint8)
    return out.view(S.shape[0], -1), bool(torch.equal(C[out.long()], B))


@torch.no_grad()
def nn_table(C, k=16):
    """(256, k) indices of the k nearest codebook rows of each row (itself first).  Temporal 256^2*8 MACs + sort, spatial 256^2."""
    dist = torch.cdist(C, C)
    return torch.sort(dist, dim=1, stable=True)[1][:, :k].contiguous()


@torch.no_grad()
def block_gain(g, S, idx, C, nn, chunk=1 << 18):
    """Per 8-block: best gain max_i <g_b, s_b - C[nn[idx_b, i]]> and the row achieving it.
    Temporal 16*8*M MACs (M blocks), spatial chunk*16*8 words."""
    M = idx.numel()
    gb, sb, ib = g.reshape(-1, 8), S.reshape(-1, 8), idx.reshape(-1).long()
    gain = torch.empty(M, device=g.device)
    new = torch.empty(M, dtype=torch.uint8, device=g.device)
    for s0 in range(0, M, chunk):
        cand = nn[ib[s0:s0 + chunk]]
        diff = sb[s0:s0 + chunk, None, :] - C[cand]
        gi = (diff * gb[s0:s0 + chunk, None, :]).sum(-1)
        best, arg = gi.max(1)
        gain[s0:s0 + chunk] = best
        new[s0:s0 + chunk] = cand.gather(1, arg[:, None])[:, 0].to(torch.uint8)
    return gain, new


@torch.no_grad()
def block_gain_obs(gsum, fsum, count, S, idx, C, nn, lams, chunk=1 << 18):
    """Per 8-block and per lam: best curvature-corrected gain over the 16 nearest rows and the row achieving it.
    With D = C_i - s_b, gbar = gsum/count, Fd = fsum/count (diagonal empirical Fisher of the window's batch gradients),
    the predicted loss change is <gbar, D> + (lam/2) sum_k Fd_k D_k^2 (Hassibi and Stork, Optimal Brain Surgeon, diagonal
    order, 1993), and gain = minus that.  The row itself (D = 0) is candidate 0, so gain >= 0.
    Returns gain (L, M) float32 and new (L, M) uint8.  Temporal L*16*8*M MACs for the quadratic term plus 16*8*M for the
    linear term (M blocks, L = len(lams)); spatial chunk*16*8 words of workspace plus L*M*5 bytes of output.  The lams share
    diff, lin and qd, so the L-fold cost is only the 16*M reduction per lam."""
    M, L = idx.numel(), len(lams)
    gb, fb = gsum.reshape(-1, 8), fsum.reshape(-1, 8)
    sb, ib = S.reshape(-1, 8), idx.reshape(-1).long()
    lam = torch.tensor(lams, dtype=torch.float32, device=gb.device)[:, None, None]
    inv = 1.0 / count
    gain = torch.empty(L, M, device=gb.device)
    new = torch.empty(L, M, dtype=torch.uint8, device=gb.device)
    for s0 in range(0, M, chunk):
        s1 = min(M, s0 + chunk)
        cand = nn[ib[s0:s1]]
        diff = sb[s0:s1, None, :] - C[cand]
        lin = (diff * gb[s0:s1, None, :].float()).sum(-1) * inv
        qd = (diff * diff * fb[s0:s1, None, :].float()).sum(-1) * inv
        best, arg = (lin[None] - 0.5 * lam * qd[None]).max(2)
        gain[:, s0:s1] = best
        new[:, s0:s1] = cand[None].expand(L, -1, -1).gather(2, arg[..., None])[..., 0].to(torch.uint8)
    return gain, new


@torch.no_grad()
def project_local(L, idx, C, nn, chunk=1 << 18):
    """Per 8-block of the latent L (O, n): the nearest of the 16 codebook rows nn[idx_b] (Euclidean) and its index.
    Returns (S_hat (O, n) float32 of exact codebook rows, new idx (O, n/8) uint8).
    Temporal 16*8*M MACs (M = O*n/8 blocks; per chunk a gather of chunk*16 rows, a subtraction, a square-sum and an argmin);
    spatial chunk*16*8 words of workspace plus the M*8-word output (chunk = 2^18: 134 MB)."""
    M = idx.numel()
    Lb, ib = L.reshape(-1, 8), idx.reshape(-1).long()
    new = torch.empty(M, dtype=torch.uint8, device=L.device)
    S_hat = torch.empty_like(Lb)
    for s0 in range(0, M, chunk):
        cand = nn[ib[s0:s0 + chunk]]
        rows = C[cand]
        d2 = ((Lb[s0:s0 + chunk, None, :] - rows) ** 2).sum(-1)
        arg = d2.argmin(1)
        best = cand.gather(1, arg[:, None])[:, 0]
        new[s0:s0 + chunk] = best.to(torch.uint8)
        S_hat[s0:s0 + chunk] = C[best]
    return S_hat.view(L.shape), new.view(idx.shape)


@torch.no_grad()
def preimage_init(P, idx, C, nn, chunk=1 << 18):
    """Latent start inside the stored cell.  Per 8-block with v = P_b - C[idx_b] and, for each of the 16 rows j in nn[idx_b],
    u = C[j] - C[idx_b]: t*_j = |u|^2 / (2 v.u) when v.u > 0 (else inf), rho_b = min(1, 0.9 min_j t*_j), L0_b = C[idx_b] + rho_b v.
    |L - C[j]|^2 - |L - C[idx_b]|^2 = |u|^2 - 2 t v.u > 0 for t < t*_j, so nearest-of-16(L0) == idx_b.
    Returns (L0 (O, n) float32, pnear (O, n/8) uint8 = Euclidean-nearest of all 256 rows to P_b, number of blocks with rho_b < 1,
    number of blocks with pnear == idx_b).
    Temporal (16*8*3 + 256*8)*M MACs (M = O*n/8 blocks; per chunk the 16 difference rows, their dot and square-sum with v, and one
    matmul against the 256 rows); spatial chunk*16*8 words of workspace plus chunk*256 words for the matmul plus the M*8-word output."""
    M = idx.numel()
    Pb, ib = P.reshape(-1, 8), idx.reshape(-1).long()
    cn = (C * C).sum(1)
    L0 = torch.empty_like(Pb)
    pnear = torch.empty(M, dtype=torch.uint8, device=P.device)
    nclip = nnear = 0
    for s0 in range(0, M, chunk):
        pb, i0 = Pb[s0:s0 + chunk], ib[s0:s0 + chunk]
        c0 = C[i0]
        v = pb - c0
        u = C[nn[i0]] - c0[:, None, :]
        vu = (v[:, None, :] * u).sum(-1)
        uu = (u * u).sum(-1)
        t = torch.where(vu > 0, uu / (2.0 * vu.clamp_min(1e-30)), torch.full_like(vu, float("inf")))
        rho = torch.clamp(0.9 * t.min(1)[0], max=1.0)
        L0[s0:s0 + chunk] = c0 + rho[:, None] * v
        pn = (2.0 * pb @ C.T - cn[None, :]).argmax(1)
        pnear[s0:s0 + chunk] = pn.to(torch.uint8)
        nclip += int((rho < 1.0).sum())
        nnear += int((pn == i0).sum())
    return L0.view(P.shape), pnear.view(idx.shape), nclip, nnear


# ------------------------------------------------------------------------------------------------ student
class Plane:
    """One stored plane: integer codes idx (or a fixed S_fix when a block is not a codebook row), frozen per-row r grid
    (lo, sp), latents r0/c0/d0 with log-space offsets (pa, pb, pd)."""

    def __init__(self, d, C):
        self.L = None  # --codes ste: fp32 latent of the code field
        self.U = self.V = self.k = None  # --train corr: fp32 factors of the dense corrector, set by factor_dense
        self.dense = d["dense"].to(DEV) if "dense" in d else None
        if self.dense is not None:
            self.Uc, self.Vc = d["U"].to(DEV), d["V"].to(DEV)  # the stored factors, p.dense = Uc @ Vc
            return
        S = d["S_rot"].to(DEV)
        self.O, self.n = S.shape
        self.rows = None if d["rows"] is None else d["rows"].to(DEV)
        self.idx, self.S_fix = None, None
        idx, ok = encode(S, C)
        self.exact = ok
        if ok:
            self.idx = idx
        else:
            self.S_fix = S
        self.lo, self.sp = d["rgrid"][0].to(DEV), d["rgrid"][1].to(DEV)
        self.r0, self.c0, self.d0 = d["r"].to(DEV), d["c"].to(DEV), d["d"].to(DEV)
        G = self.n // GROUP
        self.pa = torch.zeros(self.O, G, device=DEV, requires_grad=True)
        self.pb = torch.zeros(self.n, device=DEV, requires_grad=True)
        self.pd = torch.zeros(self.n, device=DEV, requires_grad=True)

    def S(self, C):
        return self.S_fix if self.idx is None else C[self.idx.long()].view(self.O, self.n)

    def latents(self):
        return self.r0 * torch.exp(self.pa), self.c0 * torch.exp(self.pb), self.d0 * torch.exp(self.pd)

    def r_grid(self, r_l):
        k = torch.round(((r_l - self.lo[:, None]) / self.sp[:, None] * 15.0).clamp(0.0, 15.0))
        return self.lo[:, None] + (k / 15.0) * self.sp[:, None], k

    def qvals(self):
        """STE-quantized (r, c, d): arm quantizers (frozen row grid for r, ft.quant_c_u8 6 / 5 bits for c / d)."""
        r_l, c_l, d_l = self.latents()
        with torch.no_grad():
            rq, _ = self.r_grid(r_l)
            cq = ft.quant_c_u8(c_l, bits=6)[0]
            dq = ft.quant_c_u8(d_l, bits=5)[0]
        return r_l + (rq - r_l).detach(), c_l + (cq - c_l).detach(), d_l + (dq - d_l).detach()

    def regrid(self):
        """Re-fit the per-row r grid (lo, sp) to the current latents with ft.quant_r_hier; the stored cost is unchanged.
        Temporal 12 candidate grids x O*G words: the du x s_idx loop of ft.quant_r_hier is 3 x 4 = 12 candidates, each one
        rounding over the O*G latent words. Spatial O*G words per candidate, not kept across candidates.
        Derivation: the loop is 3 x 4 by construction; PACK_SINK is a non-None sentinel so the chosen grid is attached as .grid."""
        prev, ft.PACK_SINK = ft.PACK_SINK, {}
        try:
            with torch.no_grad():
                best, _ = ft.quant_r_hier(self.latents()[0])
                lo, sp = best.grid
                self.lo, self.sp = lo.detach().to(self.lo), sp.detach().to(self.sp)
        finally:
            ft.PACK_SINK = prev

    def sat_frac(self):
        """Fraction of r latents strictly outside [lo, lo + sp] of their row. Temporal O*G compares plus one reduction;
        spatial O*G bool words. Derivation: one elementwise comparison over the O*G latent matrix, then a mean."""
        with torch.no_grad():
            r_l = self.latents()[0]
            out = (r_l < self.lo[:, None]) | (r_l > (self.lo + self.sp)[:, None])
            return float(out.float().mean())

    def params(self):
        return [self.pa, self.pb, self.pd]


def plane_weight(p, S):
    """Dense weight of one plane from the code field S: compose(fwht128(S * d), r, c).
    Temporal O*n*(1 + 7 + 2) flops (O*n scale by d; 7 butterfly stages of one add or sub per word; two elementwise scales in
    compose) plus O(O*G + 2n) for qvals (G = n/128 groups); spatial ~9*O*n words (fwht128 stacks one new O*n tensor per stage,
    and autograd keeps the stage outputs when S carries grad).  Derivation: the stage loop in ft.fwht128 runs log2(128) = 7 times."""
    r, c, d = p.qvals()
    return ft.compose(ft.fwht128(S * d[None, :]), r, c)


def u8_ste(x, dim):
    """x coded as ft.quant_c_u8 codes it (per vector along dim: fp16 min and max, 8-bit affine levels), with a
    straight-through gradient (forward is the coded value, backward is d/dx = 1).  This is the store
    fold_transfer.lowrank_corrector prices on the u8 branch, rank * (8 * (O + n) + 64) bits.
    Temporal O(N) over N = x.numel() (one min/max reduction per vector, one round per entry); spatial one N-word
    temporary.  Derivation: the body is the per-vector affine round; the .detach() on the correction keeps identity Jacobian."""
    lo, hi = ft.f16r(x.amin(dim, keepdim=True)), ft.f16r(x.amax(dim, keepdim=True))
    span = (hi - lo).clamp_min(1e-12)
    q = lo + torch.round((x - lo) / span * 255.0).clamp(0.0, 255.0) / 255.0 * span
    return x + (q - x).detach()


def u8_factors(p):
    """The stored corrector factors: U coded per column (dim 0), V per row (dim 1), as lowrank_corrector stores them."""
    return u8_ste(p.U, 0), u8_ste(p.V, 1)


def factor_dense(p):
    """Make the stored factors of one dense corrector plane trainable: p.U = Uc (O x k), p.V = Vc (k x n) in fp32, k the
    stored rank.  Refuses (ValueError) factors that are not fixed points of u8_ste, so u8(U) @ u8(V) at step 0 is the
    stored product exactly; an f16-branch or unrounded corrector is refused, not re-coded.  Sets p.O, p.n, p.k.
    Temporal O(k (O + n)) for the two u8_ste passes and comparisons; spatial O(k (O + n)) words for the factor copies."""
    U, V = p.Uc.float(), p.Vc.float()
    if not (torch.equal(u8_ste(U, 0), U) and torch.equal(u8_ste(V, 1), V)):
        raise ValueError("factor_dense: stored corrector factors are not u8 codes (u8_ste is not the identity on them)")
    p.O, p.k, p.n = U.shape[0], U.shape[1], V.shape[1]
    p.U = U.clone().contiguous().requires_grad_()
    p.V = V.clone().contiguous().requires_grad_()
    return p.k


def corr_weight(p):
    """The dense corrector plane's weight from its trainable factors, each 8-bit coded as stored: u8(U) @ u8(V).  With
    p.U None (the dense plane frozen, no --train corr) returns p.dense, so the frozen code path is byte-for-byte today's.
    Temporal O(O*k + k*n) roundings plus O(O*k*n) multiply-adds; spatial one O*k and one k*n coded temporary plus the
    O*n product. Derivation: u8_ste over O*k + k*n entries, then one matmul.  The product runs in fp32 with autocast off:
    layer_dict calls this inside the bf16 autocast and layer_dicts_step outside it, and the two must compose one weight."""
    if p.U is None:
        return p.dense
    Uq, Vq = u8_factors(p)
    with torch.autocast("cuda", enabled=False):
        return Uq @ Vq


def add_plane(W, p, Wi):
    """Sum a plane weight Wi into W (all rows when p.rows is None, else the rows p.rows).
    Temporal O*n adds (rows given: a clone of W, O_W*n words copied, plus |rows|*n adds); spatial O_W*n words for the clone,
    none otherwise.  Derivation: one elementwise add or one index_put over |rows|*n words, no loop."""
    if W is None:
        return Wi
    if p.rows is None:
        return W + Wi
    W = W.clone()
    W[p.rows] += Wi
    return W


class PlaneCompose(torch.autograd.Function):
    """W = compose(fwht128(S * d), r, c) with the code field S fixed (no gradient to S).  The forward is plane_weight's
    arithmetic (bit-identical W); the backward is the analytic adjoint and recomputes Y = fwht128(S * d) instead of keeping
    autograd's stage tensors.  With G = dL/dW and W_ij = Y_ij r_{i,g(j)} c_j: dL/dc = sum_i G*Y*r_rep,
    dL/dr_{i,g} = sum_{j in g} G*Y*c, dL/dd = sum_i S * fwht128(G * r_rep * c) (fwht128 is symmetric orthogonal, its own adjoint).
    Temporal: forward O*n*(1 + 7 + 2) flops; backward two fwht128 (7 O*n add/sub each) plus ~7 O*n elementwise and three
    reductions.  Spatial: saves references to S, d, r, c only; ~4 O*n transient words in backward."""

    @staticmethod
    def forward(ctx, S, d, r, c):
        ctx.save_for_backward(S, d, r, c)
        return ft.compose(ft.fwht128(S * d[None, :]), r, c)

    @staticmethod
    def backward(ctx, G):
        S, d, r, c = ctx.saved_tensors
        O, n = S.shape
        r_rep = r.repeat_interleave(GROUP, 1)
        GY = G * ft.fwht128(S * d[None, :])
        gc = (GY * r_rep).sum(0)
        gr = (GY * c[None, :]).view(O, n // GROUP, GROUP).sum(-1)
        gd = (S * ft.fwht128(G * r_rep * c[None, :])).sum(0)
        return None, gd, gr, gc


KIND_SITE = {nm: site for site, names, _ in kp.SITES for nm in names}
ANGLE_BITS = 16


def kan_ab(dl, h):
    """Stage coefficients a = cos d - sin d, b = cos d + sin d, each (A, h) with A = GROUP / (2 h), from the 64 angle offsets dl
    (d = theta - pi/4).  At d = 0 both are exactly 1.  Temporal ~4 flops per butterfly; spatial two 64-word tensors."""
    A = GROUP // (2 * h)
    c, sn = dl.cos().view(A, h), dl.sin().view(A, h)
    return c - sn, c + sn


def kan_stage(y, dl, h):
    """One unscaled reflection stage on y (M, 128), stride h = 2^st, dl (64,) its offsets d = theta - pi/4:
    (u, v) -> (a u + b v, b u - a v) = sqrt(2) (cos theta u + sin theta v, sin theta u - cos theta v).  Symmetric with square 2I
    (a^2 + b^2 = 2), so its adjoint is itself and its inverse is itself / 2 (exact in binary).  At d = 0 it is fwht128's
    (u + v, u - v).  Temporal ~3 M*128 flops; spatial one M*128 output."""
    a, b = kan_ab(dl, h)
    y4 = y.reshape(-1, GROUP // (2 * h), 2, h)
    u, v = y4[:, :, 0, :], y4[:, :, 1, :]
    return torch.stack((a * u + b * v, b * u - a * v), dim=2).reshape(-1, GROUP)


def kan_register_raw(x, delta):
    """Seven unscaled stages (no 1/sqrt(128)); x (..., n), n a multiple of 128, delta (7, 64).  Temporal 7 x 3 n flops."""
    y = x.reshape(-1, GROUP)
    for st in range(kp.NST):
        y = kan_stage(y, delta[st], 1 << st)
    return y.reshape(x.shape)


def kan_register(x, delta):
    """The learned register: kan_register_raw scaled once by GROUP**-0.5 in fwht128's place and form.  Equals
    kp.register(x, pi/4 + delta) and, at delta = 0, is bit-identical to ft.fwht128.  Temporal 7 x 3 n + n flops."""
    return (kan_register_raw(x, delta) * (GROUP ** -0.5)).contiguous()


def kan_matrix(delta):
    """The register as a 128 x 128 row-vector matrix T(delta): the map kan_register(., delta) applied to the identity rows, so
    x @ T == kan_register(x, delta) for x (..., 128).  Built from plain torch ops (the register on GROUP rows), so autograd
    carries dT -> ddelta and no hand-written adjoint is needed here.  Temporal 7 stages x 3 flops x GROUP^2 = 7 x 3 x 128^2
    flops; spatial GROUP^2 = 128^2 words.  At delta = 0 every entry is exactly +-fp(GROUP**-0.5), so T(0) is torch.equal to
    had_mm's H128 / sqrt(128) (HN) in fp32 and fp64 (checked by selftest_kan control 6)."""
    return kan_register(torch.eye(GROUP, dtype=delta.dtype, device=delta.device), delta)


def kan_matrix_batch(D):
    """kan_matrix for a batch of offsets D (K, 7, 64) -> (K, 128, 128) with out[k] == kan_matrix(D[k]) bit for bit: y starts as
    the identity expanded to (K, GROUP, GROUP) and stage st with h = 1 << st, A = GROUP // (2 h) applies
    (u, v) -> (a u + b v, b u - a v) with a = cos d - sin d, b = cos d + sin d, one multiply by GROUP**-0.5 at the end.  The
    stage loop, the a, b and the final scale are the same ops in the same order as kan_matrix, so both fp32 and fp64 are
    torch.equal per key (selftest_kan control 9).  Temporal 7 x 3 x K x GROUP^2 = 7 x 3 x K x 128^2 flops in 7 launches of each
    elementwise op; spatial K x GROUP^2 = K x 128^2 words per stage."""
    K = D.shape[0]
    y = torch.eye(GROUP, dtype=D.dtype, device=D.device)[None].expand(K, GROUP, GROUP)
    for st in range(kp.NST):
        h = 1 << st
        A = GROUP // (2 * h)
        c, sn = D[:, st].cos().view(K, 1, A, h), D[:, st].sin().view(K, 1, A, h)
        a, b = c - sn, c + sn
        y4 = y.reshape(K, GROUP, A, 2, h)
        u, v = y4[..., 0, :], y4[..., 1, :]
        y = torch.stack((a * u + b * v, b * u - a * v), dim=3).reshape(K, GROUP, GROUP)
    return (y * (GROUP ** -0.5)).contiguous()


class RegisterCompose(torch.autograd.Function):
    """W = compose(kan_register(S * d, delta), r, c) with the code field S fixed; delta (7, 64) = theta - pi/4 (init 0).
    Backward without stored stage tensors: Z = kan_register_raw(S * d) is recomputed, Y = Z s with s = GROUP**-0.5.  With
    G = dL/dW: dL/dc = sum_i G*Y*r_rep, dL/dr_{i,g} = sum_{j in g} G*Y*c, and gY = G * r_rep * c.  Y = s M_6 ... M_0 x, so the
    adjoint starts at s gY and walks the stages 6..0 with the unscaled M.  Each M is symmetric with M^2 = 2I: the stage input is
    x_in = M x_out / 2.  With M = [[a, b], [b, -a]], da/dd = -b, db/dd = a, the stage's offset gradient is
    gdelta[st] = sum over rows and groups of g1 (-b u + a v) + g2 (a u + b v), (u, v) the x_in pair, (g1, g2) the gradient at the
    stage output; then g <- M g.  After stage 0, dL/dd = sum_i S * g.  dL/dtheta = dL/ddelta.
    Temporal: forward 7 stages x 3 O*n + 3 O*n ~ 24 O*n; backward the recompute (21 O*n) plus per stage one inverse stage (3), the
    offset product-sum (~6) and one adjoint stage (3) = 12 O*n x 7 = 84 O*n, plus ~8 O*n elementwise and reductions.  Spatial:
    saves S, d, r, c, delta only; ~8 O*n transient words in backward.  The loop runs log2(128) = 7 times over O*n words."""

    @staticmethod
    def forward(ctx, S, d, r, c, delta):
        ctx.save_for_backward(S, d, r, c, delta)
        return ft.compose(kan_register(S * d[None, :], delta), r, c)

    @staticmethod
    def backward(ctx, G):
        S, d, r, c, delta = ctx.saved_tensors
        O, n = S.shape
        s = GROUP ** -0.5
        r_rep = r.repeat_interleave(GROUP, 1)
        x = kan_register_raw(S * d[None, :], delta).reshape(-1, GROUP)  # Z
        GY = G * (x.view(O, n) * s)
        gc = (GY * r_rep).sum(0)
        gr = (GY * c[None, :]).view(O, n // GROUP, GROUP).sum(-1)
        del GY
        g = (G * r_rep * c[None, :] * s).reshape(-1, GROUP)
        gdelta = torch.empty_like(delta)
        for st in range(kp.NST - 1, -1, -1):
            h = 1 << st
            A = GROUP // (2 * h)
            x = kan_stage(x, delta[st], h) * 0.5  # stage input
            a, b = kan_ab(delta[st], h)
            xi, gg = x.view(-1, A, 2, h), g.view(-1, A, 2, h)
            u, v, g1, g2 = xi[:, :, 0, :], xi[:, :, 1, :], gg[:, :, 0, :], gg[:, :, 1, :]
            gdelta[st] = (g1 * (a * v - b * u) + g2 * (a * u + b * v)).sum(0).reshape(kp.NBF)
            g = kan_stage(g, delta[st], h)
        gd = (S * g.view(O, n)).sum(0)
        return None, gd, gr, gc, gdelta


def kan_recon(planes, delta_q):
    """ft._sink_recon with fwht128 replaced by kan_register(., delta_q): the stored weight from the planes, summed in the same
    order (dense planes added, coded planes composed, row-restricted planes added into W[rows]).  Temporal O(P * O * n * 7)
    multiply-adds (P planes, 7 reflection stages each), spatial O(O * n) words."""
    W = None
    for p in planes:
        if "dense" in p:
            W = W + p["dense"]
            continue
        Wi = ft.compose(kan_register(p["S_rot"] * p["d"][None, :], delta_q), p["r"], p["c"])
        if W is None:
            W = Wi
        elif p["rows"] is None:
            W = W + Wi
        else:
            W = W.clone()
            W[p["rows"]] += Wi
    return W


def selftest_kan(out):
    """--selftest-kan: no model, on DEV.  Twelve controls, all must hold: (1) kan_register at delta = 0 is torch.equal to
    ft.fwht128 on random fp32 and fp64 input, and RegisterCompose's forward is torch.equal to ft.compose(fwht128(.)) in fp32;
    (2) float64 forward at random |delta| ~ 0.1 against autograd's reference ft.compose(kp.register(S d, pi/4 + delta), r, c),
    rel < 1e-12; (3) the adjoints gd, gr, gc, gdelta against autograd on that reference, rel < 1e-10 each; (4) gdelta against
    central finite differences of the loss <W, G> in float64 on 16 random entries, rel < 1e-7; (5) two mutated local copies of the
    backward (gdelta sign flipped; final scale dropped from the adjoint seed) must each fail control (3) with rel >= 0.5.
    (6) kan_matrix(zeros) is torch.equal to had_mm's H128 / sqrt(128) in fp32 and fp64; (7) in float64 on the same S, d, r, c,
    Gup, RegisterComposeMM's forward and its d, r, c and delta adjoints (delta via autograd through kan_matrix) match
    RegisterCompose's, rel < 1e-12 each; (8) local copies of the mm backward with gc formed from Y instead of G * Y
    give gc rel >= 0.5, and with gX = gY @ T (T not transposed) give gd rel >= 1e-6, against the RegisterCompose reference,
    while the unmutated copies stay < 1e-12; (9) the batched register pass kan_matrix_batch on D (5, 7, 64) with |D| ~ 0.1 is
    torch.equal to kan_matrix per key in fp32 and fp64, and its dL/dD matches the stacked per-key gradients, rel < 1e-12 in
    fp64; (10) a local copy of the batch with the stage's a and b swapped moves away from kan_matrix at D != 0 by rel >= 1e-3
    (red), so control (9) sees the stage; (11) with --train corr, factor_dense on a synthetic q-like dense plane (rank 2,
    stored as quant_c_u8 factors U, V with p.dense = U @ V, as lowrank_corrector's u8 branch stores it) gives k = 2, and
    corr_weight = u8(U) @ u8(V) is torch.equal to p.dense while its factors receive a nonzero gradient in one backward;
    (12) a local copy of corr_weight that returns the frozen p.dense has rel exactly 0 but no factor gradient, so control
    (11) is red on it; (13) factor_dense refuses f16-rounded factors (not u8 codes) with ValueError.
    Writes the result to `out`, prints one JSON line, returns the exit status.
    Temporal: a few passes over O*n = 2048 words plus 32 forward evaluations, u8 codes and products of three 896x2 and 2x896 factor pairs;
    spatial O(O*n)."""
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)
    O, n = 8, 256
    f64 = dict(device=DEV, dtype=torch.float64)
    S = torch.randint(0, 2, (O, n), generator=gen, device=DEV).double() * 2 - 1
    d = torch.rand(n, generator=gen, **f64) + 0.5
    r = torch.rand(O, n // GROUP, generator=gen, **f64) + 0.5
    c = torch.rand(n, generator=gen, **f64) + 0.5
    delta = torch.randn(kp.NST, kp.NBF, generator=gen, **f64) * 0.1
    Gup = torch.randn(O, n, generator=gen, **f64)
    # (1) identity at delta = 0
    ident = {}
    for dt in (torch.float32, torch.float64):
        X = torch.randn(64, 4 * GROUP, generator=gen, device=DEV, dtype=torch.float64).to(dt)
        z = torch.zeros(kp.NST, kp.NBF, device=DEV, dtype=dt)
        Y, Yf = kan_register(X, z), ft.fwht128(X)
        ident[str(dt)] = dict(equal=bool(torch.equal(Y, Yf)), max_abs=float((Y - Yf).abs().max()))
    S32, d32, r32, c32 = S.float(), d.float(), r.float(), c.float()
    z32 = torch.zeros(kp.NST, kp.NBF, device=DEV, dtype=torch.float32)
    Wk, Wf = RegisterCompose.apply(S32, d32, r32, c32, z32), ft.compose(ft.fwht128(S32 * d32[None, :]), r32, c32)
    ident["RegisterCompose_fp32"] = dict(equal=bool(torch.equal(Wk, Wf)), max_abs=float((Wk - Wf).abs().max()))
    ok1 = all(v["equal"] and v["max_abs"] == 0.0 for v in ident.values())

    def ref():
        leaves = [t.clone().requires_grad_() for t in (d, r, c, delta)]
        W = ft.compose(kp.register(S * leaves[0][None, :], math.pi / 4 + leaves[3]), leaves[1], leaves[2])
        W.backward(Gup)
        return W.detach(), [t.grad for t in leaves]

    def new():
        leaves = [t.clone().requires_grad_() for t in (d, r, c, delta)]
        W = RegisterCompose.apply(S, leaves[0], leaves[1], leaves[2], leaves[3])
        W.backward(Gup)
        return W.detach(), [t.grad for t in leaves]

    def rel(a_, b_):
        return float((a_ - b_).abs().max() / b_.abs().max())

    def mutated(flip, noscale):
        """Local copy of RegisterCompose.backward's gdelta walk with one injected defect."""
        s = 1.0 if noscale else GROUP ** -0.5
        r_rep = r.repeat_interleave(GROUP, 1)
        x = kan_register_raw(S * d[None, :], delta).reshape(-1, GROUP)
        g = (Gup * r_rep * c[None, :] * s).reshape(-1, GROUP)
        gdl = torch.empty_like(delta)
        for st in range(kp.NST - 1, -1, -1):
            h = 1 << st
            A = GROUP // (2 * h)
            x = kan_stage(x, delta[st], h) * 0.5
            a, b = kan_ab(delta[st], h)
            xi, gg = x.view(-1, A, 2, h), g.view(-1, A, 2, h)
            u, v, g1, g2 = xi[:, :, 0, :], xi[:, :, 1, :], gg[:, :, 0, :], gg[:, :, 1, :]
            gdl[st] = (g1 * (a * v - b * u) + g2 * (a * u + b * v)).sum(0).reshape(kp.NBF)
            g = kan_stage(g, delta[st], h)
        return -gdl if flip else gdl

    Wr, gr_ = ref()
    names = ("gd", "gr", "gc", "gdelta")
    Wn, gn_ = new()
    fwd = rel(Wn, Wr)
    errs = {k: rel(x, y) for k, x, y in zip(names, gn_, gr_)}
    ok2, ok3 = fwd < 1e-12, all(v < 1e-10 for v in errs.values())
    # (4) central finite differences of <W, Gup> in delta
    hh, fd_err = 1e-6, 0.0
    flat = gn_[3].reshape(-1)
    idx = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(1))[:16].tolist()
    for k in idx:
        dp, dm = delta.clone().reshape(-1), delta.clone().reshape(-1)
        dp[k] += hh
        dm[k] -= hh
        with torch.no_grad():
            Lp = float((RegisterCompose.apply(S, d, r, c, dp.view_as(delta)) * Gup).sum())
            Lm = float((RegisterCompose.apply(S, d, r, c, dm.view_as(delta)) * Gup).sum())
        fd_err = max(fd_err, abs((Lp - Lm) / (2 * hh) - float(flat[k])) / float(flat.abs().max()))
    ok4 = fd_err < 1e-7
    # (5) mutations
    mut = {"flip_gdelta": rel(mutated(True, False), gr_[3]), "drop_final_scale": rel(mutated(False, True), gr_[3]),
           "unmutated_copy": rel(mutated(False, False), gr_[3])}
    ok5 = mut["flip_gdelta"] >= 0.5 and mut["drop_final_scale"] >= 0.5 and mut["unmutated_copy"] < 1e-10
    # (6) kan_matrix at delta = 0 is had_mm's H128 / sqrt(128) bit for bit, in fp32 and fp64
    mm_id = {}
    for dt in (torch.float32, torch.float64):
        hx = torch.ones(1, GROUP, device=DEV, dtype=dt)
        had_mm(hx)  # populate HN[(hx.device, dt)]
        Tm = kan_matrix(torch.zeros(kp.NST, kp.NBF, device=DEV, dtype=dt))
        mm_id[str(dt)] = dict(equal=bool(torch.equal(Tm, HN[(hx.device, dt)])), max_abs=float((Tm - HN[(hx.device, dt)]).abs().max()))
    ok6 = all(v["equal"] and v["max_abs"] == 0.0 for v in mm_id.values())
    # (7) RegisterComposeMM in float64 on the same S, d, r, c, Gup as the RegisterCompose reference (Wr, gr_): forward and the
    # d, r, c adjoints from its backward, delta via autograd from gT through kan_matrix
    d_l, r_l, c_l = (t.clone().requires_grad_() for t in (d, r, c))
    dl_l = delta.clone().requires_grad_()
    Wmm = RegisterComposeMM.apply(S, d_l, r_l, c_l, kan_matrix(dl_l))
    gmm = torch.autograd.grad((Wmm * Gup).sum(), (d_l, r_l, c_l, dl_l))
    fwd_mm = rel(Wmm.detach(), Wr)
    mm_errs = {k: rel(x, y) for k, x, y in zip(names, gmm, gr_)}
    ok7 = fwd_mm < 1e-12 and all(v < 1e-12 for v in mm_errs.values())
    # (8) mutations of the mm backward against the RegisterCompose reference, the unmutated copies < 1e-12 on gd and gc:
    # gc formed from Y instead of G * Y (the defect of the first mm backward) must give gc rel >= 0.5; gX = gY @ T (T not
    # transposed) must fail control (7) by >= 10^6 x its tolerance, gd rel >= 1e-6 (H128 is symmetric, so T - T^T = O(delta)
    # and this mutation moves gd by O(|delta|), not O(1)).
    Td = kan_matrix(delta)
    Yd = (S * d[None, :]).reshape(-1, GROUP) @ Td
    gYd = (Gup.view(O, n // GROUP, GROUP) * r[:, :, None]).view(O, n) * c[None, :]

    def mut_mm(no_T):
        gX = gYd.reshape(-1, GROUP) @ (Td if no_T else Td.T)
        return (S * gX.view(O, n)).sum(0)

    def mut_gc(drop_G):
        GY = (Yd.view(O, n) if drop_G else Gup * Yd.view(O, n)).view(O, n // GROUP, GROUP)
        return (GY * r[:, :, None]).sum(0).reshape(n)

    mm_mut = dict(no_transpose=rel(mut_mm(True), gr_[0]), unmutated=rel(mut_mm(False), gr_[0]),
                  gc_drop_G=rel(mut_gc(True), gr_[2]), gc_unmutated=rel(mut_gc(False), gr_[2]))
    ok8 = (mm_mut["gc_drop_G"] >= 0.5 and mm_mut["no_transpose"] >= 1e-6
           and mm_mut["unmutated"] < 1e-12 and mm_mut["gc_unmutated"] < 1e-12)
    # (9) the batched register pass is bit for bit kan_matrix per element: D (5, 7, 64) with |D| ~ 0.1, fp32 and fp64
    # torch.equal for every k, and the dL/dD through the batch matches the stacked per-key gradients, rel < 1e-12 in fp64
    Kb = 5
    bat = {}
    for dt in (torch.float32, torch.float64):
        Db = (torch.randn(Kb, kp.NST, kp.NBF, generator=gen, device=DEV, dtype=torch.float64) * 0.1).to(dt)
        Bb = kan_matrix_batch(Db)
        eq = all(bool(torch.equal(Bb[k], kan_matrix(Db[k]))) for k in range(Kb))
        mad = max(float((Bb[k] - kan_matrix(Db[k])).abs().max()) for k in range(Kb))
        bat[str(dt)] = dict(equal=eq, max_abs=mad)
    Db = (torch.randn(Kb, kp.NST, kp.NBF, generator=gen, device=DEV, dtype=torch.float64) * 0.1).requires_grad_()
    g = torch.randn(Kb, GROUP, GROUP, generator=gen, device=DEV, dtype=torch.float64)
    (kan_matrix_batch(Db) * g).sum().backward()
    gs = []
    for k in range(Kb):
        Dk = Db.detach()[k].clone().requires_grad_()
        (kan_matrix(Dk) * g[k]).sum().backward()
        gs.append(Dk.grad)
    gbatch = rel(Db.grad.detach(), torch.stack(gs))
    ok9 = all(v["equal"] and v["max_abs"] == 0.0 for v in bat.values()) and gbatch < 1e-12

    def kan_matrix_batch_mut(D):
        """Local copy of kan_matrix_batch with the stage coefficients a and b swapped: the injected defect of control (10).
        Temporal and spatial the same 7 x 3 x K x 128^2 flops and K x 128^2 words per stage as kan_matrix_batch."""
        K = D.shape[0]
        y = torch.eye(GROUP, dtype=D.dtype, device=D.device)[None].expand(K, GROUP, GROUP)
        for st in range(kp.NST):
            h = 1 << st
            A = GROUP // (2 * h)
            c, sn = D[:, st].cos().view(K, 1, A, h), D[:, st].sin().view(K, 1, A, h)
            a, b = c + sn, c - sn
            y4 = y.reshape(K, GROUP, A, 2, h)
            u, v = y4[..., 0, :], y4[..., 1, :]
            y = torch.stack((a * u + b * v, b * u - a * v), dim=3).reshape(K, GROUP, GROUP)
        return (y * (GROUP ** -0.5)).contiguous()

    # (10) the a <-> b swap mutant must move the batch away from kan_matrix at D != 0 by rel >= 1e-3 (red), so control (9)
    # sees the stage
    Dm = torch.randn(Kb, kp.NST, kp.NBF, generator=gen, device=DEV, dtype=torch.float64) * 0.1
    Bm = kan_matrix_batch_mut(Dm)
    mut_batch = max(float((Bm[k] - kan_matrix(Dm[k])).abs().max() / kan_matrix(Dm[k]).abs().max()) for k in range(Kb))
    ok10 = mut_batch >= 1e-3
    # (11) --train corr: a dense corrector plane stored as u8 factors (q-like O x n, rank 2, as fold_transfer.
    # lowrank_corrector's u8 branch stores it) trains from those factors: corr_weight is torch.equal to the frozen
    # corr-off weight and its factors receive a nonzero gradient in one backward; (12) a local copy of corr_weight that
    # returns the frozen p.dense drops the factors from the graph: rel 0 but no gradient, so control (11)'s gradient
    # check is red on it; (13) f16-rounded factors are refused by factor_dense.
    gq = torch.Generator(device=DEV)
    gq.manual_seed(2)
    Oq, nq = 896, 896  # the arm's q linear
    Uq0 = torch.randn(Oq, 2, generator=gq, device=DEV) * 0.1
    Vq0 = torch.randn(2, nq, generator=gq, device=DEV) * 0.1
    Uh0 = torch.stack([ft.quant_c_u8(Uq0[:, j])[0] for j in range(2)], 1)
    Vh0 = torch.stack([ft.quant_c_u8(Vq0[j])[0] for j in range(2)], 0)
    pq = Plane(dict(dense=Uh0 @ Vh0, U=Uh0, V=Vh0), None)
    kq = factor_dense(pq)
    with torch.no_grad():
        rel11 = float((corr_weight(pq) - pq.dense).abs().max() / pq.dense.abs().max())
    loss11 = corr_weight(pq).pow(2).sum()
    if loss11.requires_grad:
        loss11.backward()
    gU11 = float(pq.U.grad.abs().sum()) if pq.U.grad is not None else 0.0
    gV11 = float(pq.V.grad.abs().sum()) if pq.V.grad is not None else 0.0
    ok11 = kq == 2 and rel11 == 0.0 and gU11 > 0.0 and gV11 > 0.0

    def corr_weight_mut(p):
        """Local copy of corr_weight with the injected defect of control (12): returns the frozen p.dense, so p.U and
        p.V drop out of the graph.  Temporal and spatial as corr_weight with the two cast pairs and the matmul removed."""
        return p.dense

    pm = Plane(dict(dense=Uh0 @ Vh0, U=Uh0, V=Vh0), None)
    factor_dense(pm)
    with torch.no_grad():
        rel12 = float((corr_weight_mut(pm) - pm.dense).abs().max() / pm.dense.abs().max())
    loss12 = corr_weight_mut(pm).pow(2).sum()
    if loss12.requires_grad:
        loss12.backward()
    grad12 = pm.U.grad is not None
    ok12 = rel12 == 0.0 and not grad12 and not loss12.requires_grad
    Uf, Vf = ft.f16r(Uq0), ft.f16r(Vq0)
    try:
        factor_dense(Plane(dict(dense=Uf @ Vf, U=Uf, V=Vf), None))
        ok13 = False
    except ValueError:
        ok13 = True

    res = dict(identity=ident, identity_ok=ok1, forward_rel=fwd, forward_ok=ok2, adjoint_rel=errs, adjoint_tol=1e-10, adjoint_ok=ok3,
               fd_gdelta_rel=fd_err, fd_ok=ok4, mutation_rel=mut, mutation_red=ok5, kan_matrix_hn=mm_id, kan_matrix_hn_ok=ok6,
               mm_forward_rel=fwd_mm, mm_adjoint_rel=mm_errs, mm_ok=ok7, mm_mutation_rel=mm_mut, mm_mutation_red=ok8,
               kan_matrix_batch=bat, kan_matrix_batch_grad_rel=gbatch, kan_matrix_batch_ok=ok9,
               kan_matrix_batch_mutation_rel=mut_batch, kan_matrix_batch_mutation_red=ok10,
               corr_refactor_k=kq, corr_refactor_rel=rel11, corr_refactor_grad=dict(U=gU11, V=gV11), corr_refactor_ok=ok11,
               corr_mutation_rel=rel12, corr_mutation_requires_grad=bool(loss12.requires_grad),
               corr_mutation_grad_present=grad12, corr_mutation_red=ok12, corr_f16_refused=ok13,
               ok=ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7 and ok8 and ok9 and ok10 and ok11 and ok12 and ok13)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res), flush=True)
    return 0 if res["ok"] else 1


HN = {}


def mm_highest(x, T):
    """The high-precision form of x @ T: one matmul with TF32 switched off for the call (the prior mode restored after), so
    the tensor-core path is the full fp32 one.  Shared by had_mm and RegisterComposeMM.  Temporal 2*GROUP flops per word on
    the tensor units; spatial one output tensor."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    y = x @ T
    torch.set_float32_matmul_precision(prev)
    return y


def had_mm(x):
    """fwht128 as one matmul: x viewed (-1, 128) times Sylvester's H128 / sqrt(128) (symmetric, its own inverse), in full
    fp32 (TF32 is switched off for the call).  Temporal 2 * 128 flops per word on the tensor units against fwht128's 7
    stacked butterfly stages (7 reads and writes of the whole tensor); spatial one output tensor.  Equal to fwht128 up to
    fp32 summation order."""
    key = (x.device, x.dtype)
    if key not in HN:
        H = torch.ones(1, 1, dtype=torch.float64)
        while H.shape[0] < GROUP:
            H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
        HN[key] = (H * GROUP ** -0.5).to(x.device, x.dtype)
    return mm_highest(x.reshape(-1, GROUP), HN[key]).view(x.shape)


def jitter(W, seed):
    """W * (1 + 2^-22 s), s a seeded +-1 field: a two-ulp fp32 perturbation of every word, the size of had_mm's summation-order
    difference from fwht128, used by the grad_equiv control as the reference for what any fp32 reordering does to the
    bf16 student.  Temporal 2 O*n (one randint, one multiply-add); spatial two O*n words.  Control path only."""
    g = torch.Generator(device=W.device)
    g.manual_seed(seed)
    s = torch.randint(0, 2, W.shape, generator=g, device=W.device, dtype=torch.int8).float() * 2.0 - 1.0
    return W * (1.0 + 2.0 ** -22 * s)


class PlaneComposeMM(torch.autograd.Function):
    """PlaneCompose with fwht128 replaced by had_mm and r broadcast over its 128-column groups instead of
    repeat_interleave'd (the product order (Y * r) * c is kept).  Same adjoint as PlaneCompose.  Temporal: forward one
    had_mm plus 3 O*n elementwise; backward two had_mm plus ~6 O*n elementwise and three reductions.  Spatial as
    PlaneCompose without the O*n r_rep tensor."""

    @staticmethod
    def forward(ctx, S, d, r, c):
        ctx.save_for_backward(S, d, r, c)
        O, n = S.shape
        return (had_mm(S * d[None, :]).view(O, n // GROUP, GROUP) * r[:, :, None]).view(O, n) * c[None, :]

    @staticmethod
    def backward(ctx, G):
        S, d, r, c = ctx.saved_tensors
        O, n = S.shape
        GY = (G * had_mm(S * d[None, :])).view(O, n // GROUP, GROUP)
        gc = (GY * r[:, :, None]).sum(0).reshape(n)
        gr = (GY * c.view(1, n // GROUP, GROUP)).sum(-1)
        gd = (S * had_mm((G.view(O, n // GROUP, GROUP) * r[:, :, None]).view(O, n) * c[None, :])).sum(0)
        return None, gd, gr, gc


class RegisterComposeMM(torch.autograd.Function):
    """PlaneCompose with fwht128 replaced by one 128-wide matmul against the register matrix T = kan_matrix(delta) and r
    broadcast over its 128-column groups (the product order (Y * r) * c is kept).  Identical op sequence to PlaneComposeMM
    with T in HN's place, and the same adjoint as PlaneCompose.  With W = (Y r) c, Y = X T, X = S * d: dL/dc = sum_i G*Y*r,
    dL/dr_{i,g} = sum_{j in g} G*Y*c, and gY = G*r*c, so gX = gY T^T, gd = sum_i S*gX, gT = X^T gY, dL/ddelta = backprop of
    gT through kan_matrix.  Temporal: forward one 128-wide matmul plus 3 O*n elementwise; backward three matmuls (Y
    recompute, gX, gT) plus ~6 O*n elementwise and three reductions.  Spatial: saves S, d, r, c, T (T is GROUP^2 words)."""

    @staticmethod
    def forward(ctx, S, d, r, c, T):
        ctx.save_for_backward(S, d, r, c, T)
        O, n = S.shape
        X = (S * d[None, :]).reshape(-1, GROUP)
        Y = mm_highest(X, T)
        return (Y.view(O, n // GROUP, GROUP) * r[:, :, None]).view(O, n) * c[None, :]

    @staticmethod
    def backward(ctx, G):
        S, d, r, c, T = ctx.saved_tensors
        O, n = S.shape
        X = (S * d[None, :]).reshape(-1, GROUP)
        GY = (G * mm_highest(X, T).view(O, n)).view(O, n // GROUP, GROUP)
        gc = (GY * r[:, :, None]).sum(0).reshape(n)
        gr = (GY * c.view(1, n // GROUP, GROUP)).sum(-1)
        gY = (G.view(O, n // GROUP, GROUP) * r[:, :, None]).view(O, n) * c[None, :]
        gX = mm_highest(gY.reshape(-1, GROUP), T.T)
        gd = (S * gX.view(O, n)).sum(0)
        gT = mm_highest(X.T, gY.reshape(-1, GROUP))
        return None, gd, gr, gc, gT


class KLHead(torch.autograd.Function):
    """sum_t KL(p_t || q_t) over a chunk of rows and the top-1 agreement count, p = softmax(zt E^T) (teacher, no gradient),
    q = softmax(zs E^T).  The gradient to zs is formed in the forward as (q - p) E (the log-softmax adjoint, sum p = 1) and
    saved, so neither vocab-head matmul is recomputed in backward.  Temporal 3 * 2 * rows * hid * V flops (two head matmuls
    and the adjoint matmul; five under checkpoint(kl_chunk)); spatial ~3 * rows * V transient words, rows * hid saved."""

    @staticmethod
    def forward(ctx, zs, zt, E):
        lt = F.log_softmax(zt @ E.T, -1)
        ls = F.log_softmax(zs @ E.T, -1)
        pt = lt.exp()
        k = (pt * (lt - ls)).sum()
        agree = (lt.argmax(-1) == ls.argmax(-1)).sum()
        ctx.save_for_backward((ls.exp() - pt) @ E)
        ctx.mark_non_differentiable(agree)
        return k, agree

    @staticmethod
    def backward(ctx, gk, _ga):
        (gz,) = ctx.saved_tensors
        return gk * gz, None, None


# ------------------------------------------------------------------------------------------------ KL attribution
@torch.no_grad()
def attrib(cfg, TL, Lstud, ids_at):
    """Where the held-out KL of `Lstud` on `ids_at` lives, measured by restoring the teacher's linear weights one group at a time.
    Groups: base = Lstud unchanged; layer[l] = layer l's KINDS linears from TL[l], every other entry of Lstud (norms, biases,
    fnorm, the other layers); kind[k] = kind k from TL in every layer; all_linear = every KINDS linear in every layer.
    Deltas are base - group, so a positive delta is the KL the student's own weights of that group cost.
    Complementing them, three families start from all_linear and put the student's KINDS linears back into a layer set S:
    iso[l] (S = {l}) and prefix[k]/suffix[k] (S = {0..k-1} / {k..nl-1}, k in 6, 12, 18); their costs are group - all_linear,
    the KL the student's linears of S cost in isolation.  iso_cost_sum is their sum over layers.
    Each modified list is a shallow copy (Lm = list(Lstud); Lm[l] = dict(Lstud[l]); Lm[l][k] = TL[l][k]), so neither Lstud nor
    TL is mutated.  Temporal (nl + len(KINDS) + 2 + nl + 6) kl_eval passes of the student over N = ids_at.shape[0] chunks of CHUNK
    tokens, i.e. (24 + 7 + 2 + 24 + 6) student forwards over N chunks in the 24-layer arm; spatial the layer dict copies (nl + 1 dicts
    of the 11 weight tensors' references) plus kl_eval's own hidden state.  One teacher-hidden cache fill for ids_at is paid
    on the first call and reused by every group (ft.kl_eval keys _TEACHER_H_CACHE on len(TL), ids.data_ptr(), N, seqlen)."""
    t0 = ft.tick()
    base, base_top1 = ft.kl_eval(cfg, TL, Lstud, ids_at)
    nl = len(Lstud)
    layer, layer_top1 = [], []
    for l in range(nl):
        Lm = list(Lstud)
        Lm[l] = dict(Lstud[l])
        for k in KINDS:
            Lm[l][k] = TL[l][k]
        kl, t1 = ft.kl_eval(cfg, TL, Lm, ids_at)
        layer.append(kl)
        layer_top1.append(t1)
        del Lm
    kind, kind_top1 = {}, {}
    for k in KINDS:
        Lm = list(Lstud)
        for l in range(nl):
            Lm[l] = dict(Lstud[l])
            Lm[l][k] = TL[l][k]
        kl, t1 = ft.kl_eval(cfg, TL, Lm, ids_at)
        kind[k] = kl
        kind_top1[k] = t1
        del Lm
    Lm = list(Lstud)
    for l in range(nl):
        Lm[l] = dict(Lstud[l])
        for k in KINDS:
            Lm[l][k] = TL[l][k]
    all_linear, all_linear_top1 = ft.kl_eval(cfg, TL, Lm, ids_at)
    del Lm
    # iso[l]: the all-linear configuration with layer l's KINDS linears put back to the student's, so iso_cost[l] is the KL
    # that layer l's own linears cost in isolation (all_linear - all of them at once is the base-relative all_linear_delta).
    iso, iso_top1, iso_cost = [], [], []
    for l in range(nl):
        Lm = list(Lstud)
        for ll in range(nl):
            Lm[ll] = dict(Lstud[ll])
            for k in KINDS:
                Lm[ll][k] = TL[ll][k]
        for k in KINDS:
            Lm[l][k] = Lstud[l][k]
        kl, t1 = ft.kl_eval(cfg, TL, Lm, ids_at)
        iso.append(kl)
        iso_top1.append(t1)
        iso_cost.append(kl - all_linear)
        del Lm
    # prefix[k] / suffix[k]: the student's KINDS linears back over layers {0..k-1} / {k..nl-1}; k >= nl makes prefix[k] the
    # base configuration and suffix[k] the all-linear one.  Costs are referenced to all_linear, so they are comparable with
    # the sum of the iso costs and show whether the cross-layer part of the KL compounds through depth.
    prefix, prefix_cost, suffix, suffix_cost = {}, {}, {}, {}
    for K in (6, 12, 18):
        Lm = list(Lstud)
        for ll in range(nl):
            Lm[ll] = dict(Lstud[ll])
            for k in KINDS:
                Lm[ll][k] = TL[ll][k]
        for ll in range(K):
            for k in KINDS:
                Lm[ll][k] = Lstud[ll][k]
        kl, _ = ft.kl_eval(cfg, TL, Lm, ids_at)
        prefix[str(K)] = kl
        prefix_cost[str(K)] = kl - all_linear
        del Lm
        Lm = list(Lstud)
        for ll in range(nl):
            Lm[ll] = dict(Lstud[ll])
            for k in KINDS:
                Lm[ll][k] = TL[ll][k]
        for ll in range(K, nl):
            for k in KINDS:
                Lm[ll][k] = Lstud[ll][k]
        kl, _ = ft.kl_eval(cfg, TL, Lm, ids_at)
        suffix[str(K)] = kl
        suffix_cost[str(K)] = kl - all_linear
        del Lm
    return dict(base=base, base_top1=base_top1, layers=nl, layer=layer, layer_top1=layer_top1, kind=kind, kind_top1=kind_top1,
                all_linear=all_linear, all_linear_top1=all_linear_top1, layer_delta=[base - x for x in layer],
                kind_delta={k: base - kind[k] for k in kind}, all_linear_delta=base - all_linear, wall_s=ft.tick() - t0,
                iso=iso, iso_top1=iso_top1, iso_cost=iso_cost, iso_cost_sum=sum(iso_cost),
                prefix=prefix, prefix_cost=prefix_cost, suffix=suffix, suffix_cost=suffix_cost)


def attrib_line(tag, r):
    """One line: group KLs, the 5 largest layer_delta, every kind_delta, iso_cost_sum with the 3 largest iso_cost, and
    every prefix_cost and suffix_cost."""
    top = sorted(range(len(r["layer_delta"])), key=lambda l: -r["layer_delta"][l])[:5]
    itop = sorted(range(len(r["iso_cost"])), key=lambda l: -r["iso_cost"][l])[:3]
    print(f"attrib {tag}: base {r['base']:.6f} all_linear {r['all_linear']:.6f} (delta {r['all_linear_delta']:.6f})  "
          f"top layer_delta " + " ".join(f"L{l}:{r['layer_delta'][l]:.6f}" for l in top) + "  kind_delta "
          + " ".join(f"{k}:{r['kind_delta'][k]:.6f}" for k in sorted(r["kind_delta"], key=lambda k: -r["kind_delta"][k]))
          + f"  iso_cost_sum {r['iso_cost_sum']:.6f} top iso_cost "
          + " ".join(f"L{l}:{r['iso_cost'][l]:.6f}" for l in itop)
          + "  prefix_cost " + " ".join(f"{k}:{r['prefix_cost'][k]:.6f}" for k in sorted(r["prefix_cost"], key=int))
          + "  suffix_cost " + " ".join(f"{k}:{r['suffix_cost'][k]:.6f}" for k in sorted(r["suffix_cost"], key=int))
          + f"  wall {r['wall_s']:.1f}s", flush=True)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-cal", type=int, default=50); ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-in", type=int, default=20)
    ap.add_argument("--steps", type=int, default=10 ** 9); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-lat", type=float, default=1e-3); ap.add_argument("--lr-norm", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--v-every", type=int, default=25); ap.add_argument("--v-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v-gain", choices=("linear", "obs"), default="linear")
    # OBS curvature multipliers. F is the mean square of batch-mean gradients, about 1/(tokens per batch) of the
    # per-token Fisher, so lam near batch*seq puts the quadratic term on the per-token scale.
    ap.add_argument("--obs-lams", default="1,4,16,64")
    ap.add_argument("--codes", choices=("vstep", "ste"), default="vstep")
    ap.add_argument("--lr-code", type=float, default=1e-1)
    ap.add_argument("--code-init", choices=("codeword", "preimage"), default="codeword")
    ap.add_argument("--code-opt", choices=("adam", "nsgd"), default="adam")
    ap.add_argument("--code-kinds", default="all", help="comma list of linear kinds whose codes train under --codes ste")
    ap.add_argument("--rec-every", type=int, default=25)
    ap.add_argument("--regrid", type=int, default=0,
                    help="0 off; N > 0 re-fits every trainable plane's r grid (lo, sp) to its latents every N steps (training follows every re-fit)")
    ap.add_argument("--profile", type=int, default=0,
                    help="0 off; N > 0 records steps 6..5+N under torch.profiler, writes the op table to the receipt and exits")
    ap.add_argument("--attrib", type=int, default=0,
                    help="0 off; N > 0 uses the first N held-out eval chunks and, before and after finishing, restores the teacher's "
                         "linears one group at a time (each layer, each kind, all) to record where the held-out KL lives, and "
                         "restores the student's linears back into each single layer, each prefix of layers and each suffix of "
                         "layers over the all-linear configuration to record the KL they cost in isolation")
    ap.add_argument("--compose", choices=("layer", "step"), default="layer",
                    help="layer: plane weights composed inside each layer checkpoint (forward and recompute); step: composed once "
                         "per step through PlaneCompose (frozen codes only: --v-every 0, --codes vstep)")
    ap.add_argument("--kl", choices=("ckpt", "fused"), default="ckpt",
                    help="ckpt: checkpoint(kl_chunk) recomputes both head matmuls in backward; fused: KLHead saves dL/dz")
    ap.add_argument("--block-s", type=float, default=0.0,
                    help="0.0 off; > 0 starts the train loop with that many seconds of AQLM phase-3 block-wise fine-tuning "
                         "(Egiazarian et al., arXiv 2401.06118, Sec. 3.4): each code block fits its relative squared residual on the "
                         "teacher's hidden-state input (no LM head, no final norm), then the Adam state is cleared and the warmup "
                         "restarts for the KL phase; needs --codes vstep, --v-every 0, --v-gain linear, --compose step, --rotation had")
    ap.add_argument("--hadamard", choices=("butterfly", "mm"), default="butterfly",
                    help="training-path fwht128 under --compose step: butterfly (ft.fwht128) or mm (one fp32 H128 matmul); "
                         "eval, controls and the final weights always use ft.fwht128")
    ap.add_argument("--train", default="r,c,d,norm",
                    help="comma subset of r (pa), c (pb), d (pd), norm (zero-bit d_* tunables and d_fn), corr (dense "
                         "corrector factors U, V through corr_weight at --lr-corr); unselected tensors are frozen")
    ap.add_argument("--lr-corr", type=float, default=1e-4)
    ap.add_argument("--rotation", choices=("had", "kan"), default="had",
                    help="had: fwht128 in every plane weight; kan: the learned 7 x 64 reflection register (kan_probe.register) per layer and "
                         "site, angles trained end to end on KL (needs --compose step)")
    ap.add_argument("--kan-sites", default=",".join(site for site, _, _ in kp.SITES),
                    help="comma list of sites whose angles learn under --rotation kan; the others keep fwht128")
    ap.add_argument("--lr-angle", type=float, default=1e-3)
    ap.add_argument("--selftest-kan", action="store_true",
                    help="float64 register / RegisterCompose adjoint checks with a built-in negative control; no model is loaded")
    a = ap.parse_args()
    if a.selftest_kan:
        sys.exit(selftest_kan(a.out))
    kan_sites = set(a.kan_sites.split(",")) if a.rotation == "kan" else set()
    assert kan_sites <= {site for site, _, _ in kp.SITES}, a.kan_sites
    assert a.rotation == "had" or (a.compose == "step" and a.codes == "vstep" and a.v_every == 0
                                   and a.hadamard in ("butterfly", "mm") and kan_sites), \
        "--rotation kan needs --compose step, frozen codes, --hadamard butterfly or mm, a site"
    assert a.compose == "layer" or (a.v_every == 0 and a.codes == "vstep"), "--compose step needs frozen codes"
    assert a.hadamard == "butterfly" or a.compose == "step", "--hadamard mm needs --compose step"
    train_set = set(a.train.split(","))
    assert train_set and train_set <= {"r", "c", "d", "norm", "corr"}, a.train
    LAMS = tuple(float(x) for x in a.obs_lams.split(","))
    if a.block_s > 0:
        assert a.codes == "vstep", "--block-s needs --codes vstep"
        assert a.v_every == 0, "--block-s needs --v-every 0"
        assert a.v_gain == "linear", "--block-s needs --v-gain linear"
        assert a.compose == "step", "--block-s needs --compose step"
        assert a.rotation == "had", "--block-s needs --rotation had"
    if a.smoke:
        a.n_cal, a.n_eval, a.n_in, a.steps, a.batch = 8, 8, 8, 20, 2
        a.v_every = 10 if a.v_every else 0  # an explicit --v-every 0 stays off in smoke
        a.rec_every = 5
    res = dict(script_sha256=dict(pv_finish=sha_file(__file__), fold_transfer=sha_file(ft.__file__)), arm=ARM, args=vars(a),
               protocol="pack sink -> codes (E8/O8 index per 8-block) -> P-step Adam (STE) + V-step linearised code moves, KL(teacher||student)"
               + (" [v-gain obs: window gbar/Fisher, OBS diagonal gain]" if a.v_gain == "obs" else "")
               + (" [codes ste: latent L, nearest-of-16 projection, straight-through; V-steps off]" if a.codes == "ste" else "")
               + (f" [rotation kan: learned 7x64 reflection register, sites {sorted(kan_sites)}]" if a.rotation == "kan" else ""),
               v_gain=a.v_gain, codes=a.codes,
               gpu=torch.cuda.get_device_name(0), controls={}, walls={}, partial=True)

    def dump():
        json.dump(res, open(a.out, "w"), indent=1)

    torch.manual_seed(0)
    t_load = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(ft.MODEL)
    ids = tok.encode(open(ft.CALIB, encoding="utf-8").read())
    nch = len(ids) // CHUNK
    assert nch >= a.n_cal + a.n_eval, (nch, a.n_cal, a.n_eval)
    ids_all = torch.tensor(ids[:nch * CHUNK], device="cpu").view(nch, CHUNK)
    cfg, TL, hf = ft.load_model()
    del hf
    torch.cuda.empty_cache()
    ids_cal, ids_ev, ids_in = ids_all[:a.n_cal].contiguous(), ids_all[a.n_cal:a.n_cal + a.n_eval], ids_all[:a.n_in]
    res.update(n_cal=a.n_cal, eval_chunks=[a.n_cal, a.n_cal + a.n_eval], insample_chunks=[0, a.n_in])

    # ---- arm transfer with the pack sink
    ns = arm_args(a.n_cal, a.n_eval, a.n_in, a.out)
    if not a.smoke:
        ref_args = json.load(open(REF_RECEIPT))["args"]
        diff = {k: (v, ref_args[k]) for k, v in vars(ns).items() if k in ref_args and k not in ("out", "variants") and ref_args[k] != v}
        res["controls"]["args_match_reference_receipt"] = not diff
        assert not diff, diff
    ft.PACK_SINK = {}
    t0 = ft.tick()
    torch.manual_seed(0)
    Ls, binfo, tm = ft.run_variant(cfg, TL, ids_cal, ARM, ns)
    wall_transfer = ft.tick() - t0
    sink, ft.PACK_SINK = ft.PACK_SINK, None
    res["walls"]["load_s"] = t0 - t_load
    res["walls"]["transfer_s"] = wall_transfer
    res["controls"]["pack_equality"] = dict(linears=len(sink), torch_equal=ft.PACK_STATS["equal"], within_1e6=ft.PACK_STATS["tol"],
                                            max_rel=ft.PACK_STATS["max_rel"])
    assert ft.PACK_STATS["equal"] + ft.PACK_STATS["tol"] == len(sink) == cfg.nl * len(KINDS), (ft.PACK_STATS, len(sink))
    bits_before = binfo["bits"]
    res.update(bits_before=bits_before, params=binfo["params"], bpw_before=bits_before / binfo["params"],
               bits_components={k: binfo[k] for k in ("bits_sign", "bits_r", "bits_c", "bits_d", "bits_corr")})
    print(f"transfer {wall_transfer:.1f}s  elapsed {elapsed():.1f}s  bits {bits_before}  pack {ft.PACK_STATS}", flush=True)

    # ---- eval before
    torch.set_float32_matmul_precision("highest")
    t0 = ft.tick()
    kl_ev0, t1_ev0 = ft.kl_eval(cfg, TL, Ls, ids_ev)
    kl_in0, t1_in0 = ft.kl_eval(cfg, TL, Ls, ids_in)
    wall_eval = ft.tick() - t0
    res.update(kl_heldout_before=kl_ev0, top1_heldout_before=t1_ev0, kl_insample_before=kl_in0, top1_insample_before=t1_in0)
    res["walls"]["eval_before_s"] = wall_eval
    print(f"before heldout KL {kl_ev0:.6f} top1 {t1_ev0:.6f}  insample KL {kl_in0:.6f} top1 {t1_in0:.6f}  eval {wall_eval:.1f}s", flush=True)
    if not a.smoke:
        ok = abs(kl_ev0 - REF_KL_HELDOUT) <= 0.002
        res["controls"]["before_kl_regression"] = dict(ref=REF_KL_HELDOUT, got=kl_ev0, ok=ok)
        if not ok:
            res["failure"] = "held-out KL before differs from reference by more than 0.002: sink regression"
            dump(); print(res["failure"]); sys.exit(2)
    else:
        res["controls"]["before_kl_regression"] = dict(ref=None, got=kl_ev0, note="smoke: n_cal differs from reference")

    if a.attrib:
        # Where the held-out KL lives, before finishing: one teacher-hidden cache fill on ids_at, then base + one group per
        # layer + one group per kind + all linears, each group a shallow-copied layer list over the unmutated Ls.
        ids_at = ids_ev[:a.attrib].clone()
        tz, tz_t1 = ft.kl_eval(cfg, TL, TL, ids_at)
        res["controls"]["attrib_teacher_zero"] = dict(kl=tz, top1=tz_t1, chunks=int(ids_at.shape[0]), seqlen=int(ids_at.shape[1]), ok=tz < 1e-9)
        assert tz < 1e-9, ("teacher as student is not the teacher: KL(TL, TL) is nonzero", tz)
        res["attrib"] = dict(N=a.attrib, chunks=int(ids_at.shape[0]), kinds=list(KINDS))
        res["attrib"]["before"] = attrib(cfg, TL, Ls, ids_at)
        assert max(res["attrib"]["before"]["layer_delta"]) > 0, ("no layer moves the KL: attribution cannot see the student's weights", res["attrib"]["before"]["layer_delta"])
        assert max(res["attrib"]["before"]["iso_cost"]) > 0, ("no layer's student linears cost KL in isolation", res["attrib"]["before"]["iso_cost"])
        attrib_line("before", res["attrib"]["before"])
        res["walls"]["attrib_before_s"] = res["attrib"]["before"]["wall_s"]

    # ---- student planes
    C = ft.e8_o8_codebook(torch.float32, DEV)
    nn = nn_table(C)
    PL = []
    n_inexact = 0
    for l in range(cfg.nl):
        pl = {}
        for nm in KINDS:
            pl[nm] = [Plane(d, C) for d in sink[(l, nm)]]
            for p in pl[nm]:
                if p.dense is None and not p.exact:
                    n_inexact += 1
            sink[(l, nm)] = None
        PL.append(pl)
    del sink
    n_planes = sum(1 for pl in PL for nm in KINDS for p in pl[nm] if p.dense is None)
    res["controls"]["codebook_match"] = dict(planes=n_planes, inexact_planes=n_inexact, ok=n_inexact == 0)
    print(f"codes: {n_planes} planes, inexact {n_inexact}  elapsed {elapsed():.1f}s", flush=True)

    # ---- --train corr: factor every dense corrector plane into fp32 factors (p.U, p.V) at the arm's stored rank
    if "corr" in train_set:
        corr_planes = [p for pl in PL for nm in KINDS for p in pl[nm] if p.dense is not None]
        for p in corr_planes:
            factor_dense(p)
        store_bits = sum(p.k * (8 * (p.O + p.n) + 64) for p in corr_planes)  # the u8 price of the factors trained
        res["corr"] = dict(planes=len(corr_planes), k=sum(p.k for p in corr_planes), ranks=sorted({p.k for p in corr_planes}),
                           bits_corr=binfo["bits_corr"], store_bits=store_bits)
        print(f"corr: {len(corr_planes)} dense planes, ranks {sorted(p.k for p in corr_planes)}, factor store bits {store_bits} "
              f"vs bits_corr {binfo['bits_corr']}", flush=True)
        assert store_bits == binfo["bits_corr"], ("stored factor ranks do not reproduce bits_corr", store_bits, binfo["bits_corr"])

    # tunables (zero-bit)
    def vec(x):
        return x.to(DEV).float()
    tun = []
    for l in range(cfg.nl):
        t = dict(ln1=vec(Ls[l]["ln1"]), ln2=vec(Ls[l]["ln2"]))
        for b in ("qb", "kb", "vb"):
            t[b] = vec(Ls[l][b])
            t[b + "_s"] = t[b].abs().mean().clamp_min(1e-3)
        for k in ("ln1", "ln2", "qb", "kb", "vb"):
            t["d_" + k] = torch.zeros_like(t[k], requires_grad=True)
        tun.append(t)
    fn0 = vec(Ls[-1].get("fnorm", cfg.fnorm))
    d_fn = torch.zeros_like(fn0, requires_grad=True)

    def tuned(li):
        t = tun[li]
        return dict(ln1=t["ln1"] * torch.exp(t["d_ln1"]), ln2=t["ln2"] * torch.exp(t["d_ln2"]),
                    qb=t["qb"] + t["d_qb"] * t["qb_s"], kb=t["kb"] + t["d_kb"] * t["kb_s"], vb=t["vb"] + t["d_vb"] * t["vb_s"])

    ste_on, pending = [False], {}
    MODE = dict(compose=a.compose, kl=a.kl, had=a.hadamard, amp=True, rot=a.rotation)
    # angle offsets d = theta - pi/4 of the learned register: (layer, site) -> (7, 64) fp32, init 0 (= fwht128 bit for bit); empty under --rotation had
    TH = {(li, site): torch.zeros(kp.NST, kp.NBF, dtype=torch.float32, device=DEV, requires_grad=True)
          for li in range(cfg.nl) for site in sorted(kan_sites)}

    def ste_S(p):
        """Straight-through code field: the value is the exact codebook row nearest to L among nn[p.idx]; the gradient goes to L.
        The new index is parked in pending by the first call of a step (the forward); the checkpoint recompute gathers C[pending]
        (one gather, no search), the same value; the index is committed after the optimiser step.  Temporal: project_local once
        per plane per step plus one gather per recompute."""
        if id(p) in pending:
            S_hat = C[pending[id(p)].long()].view(p.O, p.n)
        else:
            S_hat, pending[id(p)] = project_local(p.L, p.idx, C, nn)
        return S_hat + (p.L - p.L.detach())

    def layer_dict(li, Ss):
        Ld, k = tuned(li), 0
        for nm in KINDS:
            W = None
            for p in PL[li][nm]:
                if p.dense is not None:
                    W = W + corr_weight(p)
                    continue
                S = Ss[k] if Ss else (ste_S(p) if ste_on[0] and p.L is not None else p.S(C))
                k += 1
                W = add_plane(W, p, plane_weight(p, S))
            Ld[nm] = W
        return Ld

    def layer_fn(h, li, *Ss):
        return ft.parts(cfg, layer_dict(li, Ss), h)["out"]

    TLg = [{k: v.to(DEV) for k, v in L.items()} for L in TL]
    Eg, fn_t = cfg.E.to(DEV), cfg.fnorm.to(DEV)

    def layer_dicts_step():
        """Every layer's weights composed once for the step through PlaneCompose, outside the layer checkpoints, so the
        recompute re-runs only ft.parts.  Temporal one PlaneCompose forward per coded plane per step (two under --compose
        layer: forward and recompute), plus under --hadamard mm one batched kan_matrix_batch register pass per call (7 stage
        launches for all (layer, site) keys) instead of one kan_matrix per (layer, site); spatial the composed fp32 weights of
        the model (4 bytes per parameter) held until backward plus one GROUP^2 register matrix per (layer, site) under mm."""
        out = []
        # Under --rotation kan --hadamard mm the register matrices T(delta) for every (layer, site) are built in one batched
        # register pass (7 stage launches total) here, once per call; Butterfly and non-kan paths keep using their own build.
        if MODE["rot"] == "kan" and MODE["had"] == "mm":
            keys = list(TH)  # fixed (layer, site) order; one batched build covers all of TH
            Tb = kan_matrix_batch(torch.stack([TH[k] for k in keys]))
            Tk = {k: Tb[i] for i, k in enumerate(keys)}  # (layer, site) -> T(delta) of this call
        else:
            Tk = {}
        for li in range(cfg.nl):
            Ld = tuned(li)
            for ki, nm in enumerate(KINDS):
                W = None
                for pi, p in enumerate(PL[li][nm]):
                    if p.dense is not None:
                        W = W + corr_weight(p)
                        continue
                    r, c, d = p.qvals()
                    th = TH.get((li, KIND_SITE[nm])) if MODE["rot"] == "kan" else None
                    if th is not None and MODE["had"] == "mm":
                        Wi = RegisterComposeMM.apply(p.Sc, d, r, c, Tk[(li, KIND_SITE[nm])])
                    elif th is not None:
                        Wi = RegisterCompose.apply(p.Sc, d, r, c, th)
                    else:
                        Wi = (PlaneComposeMM if MODE["had"] == "mm" else PlaneCompose).apply(p.Sc, d, r, c)
                    W = add_plane(W, p, jitter(Wi, (li * len(KINDS) + ki) * 8 + pi) if MODE["had"] == "jitter" else Wi)
                Ld[nm] = W
            out.append(Ld)
        return out

    def student_h(idsb, grad, Ss_all):
        h = Eg[idsb]
        Lds = layer_dicts_step() if grad and MODE["compose"] == "step" and not Ss_all and not ste_on[0] else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=MODE["amp"]):
            for li in range(cfg.nl):
                Ss = Ss_all[li] if Ss_all else ()
                if Lds is not None:
                    h = checkpoint(lambda h_, Ld: ft.parts(cfg, Ld, h_)["out"], h, Lds[li], use_reentrant=False)
                else:
                    h = checkpoint(layer_fn, h, li, *Ss, use_reentrant=False) if grad else layer_fn(h, li, *Ss)
        return h.float()

    @torch.no_grad()
    def teacher_h(idsb):
        h = Eg[idsb]
        for li in range(cfg.nl):
            h = ft.parts(cfg, TLg[li], h)["out"]
        return h

    def kl_chunk(hs, ht):
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad():
                lt = F.log_softmax(ft.rms(ht, fn_t, cfg.eps) @ Eg.T, -1)
            ls = F.log_softmax(ft.rms(hs, fn0 * torch.exp(d_fn), cfg.eps) @ Eg.T, -1)
            return (lt.exp() * (lt - ls)).sum(), (lt.argmax(-1) == ls.argmax(-1)).sum()

    def kl_batch(idsb, grad, Ss_all=None):
        hs, ht = student_h(idsb, grad, Ss_all).reshape(-1, cfg.hid), teacher_h(idsb).reshape(-1, cfg.hid)
        tot, agree = 0.0, 0
        for s0 in range(0, hs.shape[0], 256):
            if grad and MODE["kl"] == "fused":
                with torch.autocast("cuda", enabled=False):
                    zs = ft.rms(hs[s0:s0 + 256], fn0 * torch.exp(d_fn), cfg.eps)
                    with torch.no_grad():
                        zt = ft.rms(ht[s0:s0 + 256], fn_t, cfg.eps)
                    k, g = KLHead.apply(zs, zt, Eg)
            elif grad:
                k, g = checkpoint(kl_chunk, hs[s0:s0 + 256], ht[s0:s0 + 256], use_reentrant=False)
            else:
                k, g = kl_chunk(hs[s0:s0 + 256], ht[s0:s0 + 256])
            tot = tot + k
            agree += int(g)
        return tot / hs.shape[0], agree / hs.shape[0]

    def block_batch(idsb):
        """AQLM phase 3 (Egiazarian et al., arXiv 2401.06118, Sec. 3.4): per-layer fine-tuning of each code block on the teacher's
        own input.  Lds = layer_dicts_step() composes the student weights for the step (as the KL path under --compose step); the
        teacher's nl+1 hidden states are recomputed under no_grad; layer l's block is run on the detached teacher state ht[l]
        under checkpoint, inside the same bf16 autocast as student_h, and its relative squared residual error
        e_l = mean over the batch*CHUNK rows of ||o_l - ht[l+1]||^2 / ||ht[l+1] - ht[l]||^2 is minimised; the loss is the mean of
        the e_l.  Every trainable of layer l therefore receives gradient only from e_l, and no final-norm tunable receives any.
        Returns (loss, e), e the nl detached floats for logging.
        Temporal: one teacher forward keeping nl+1 hidden states and one checkpointed student forward/backward over the nl blocks
        per step, no LM head.  Spatial: (nl+1) * batch * CHUNK * hid fp32 teacher states plus today's composed weights."""
        Lds = layer_dicts_step()
        with torch.no_grad():
            ht = [Eg[idsb]]
            for li in range(cfg.nl):
                ht.append(ft.parts(cfg, TLg[li], ht[li])["out"])
        es = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=MODE["amp"]):
            for li in range(cfg.nl):
                o = checkpoint(lambda h_, Ld: ft.parts(cfg, Ld, h_)["out"], ht[li], Lds[li], use_reentrant=False)
                e_l = ((o.float() - ht[li + 1].float()).pow(2).sum(-1)
                       / (ht[li + 1].float() - ht[li].float()).pow(2).sum(-1).clamp_min(1e-12)).mean()
                es.append(e_l)
        loss = torch.stack(es).mean()
        return loss, [float(x.detach()) for x in es]

    def block_probe(idsb):
        """block_batch on a fixed batch with no gradient: the block phase draws a fresh train chunk every step, so comparing the
        loss of the first block step with the loss of the last block step on their own batches measures the text variation between
        two chunks, not the phase.  The phase's effect is measured on the inputs it started on.  Returns (loss float, e list).
        Temporal one teacher forward plus one student forward over the nl blocks, no backward; spatial as block_batch."""
        with torch.no_grad():
            loss, e = block_batch(idsb)
        return float(loss.detach()), e

    @torch.no_grad()
    def quick_eval():
        ks = []
        for s0 in range(0, min(8, ids_ev.shape[0]), 4):
            ks.append(float(kl_batch(ids_ev[s0:s0 + 4].to(DEV), False)[0]))
        return sum(ks) / len(ks)

    # ---- control: step-0 student weights equal the arm's stored weights
    with torch.no_grad():
        n_eq, max_rel = 0, 0.0
        for li in range(cfg.nl):
            Ld = layer_dict(li, ())
            for nm in KINDS:
                Wt = Ls[li][nm].to(DEV)
                if torch.equal(Ld[nm], Wt):
                    n_eq += 1
                else:
                    max_rel = max(max_rel, float((Ld[nm] - Wt).abs().max() / Wt.abs().max()))
            del Ld
        res["controls"]["step0_weights_equal_arm"] = dict(linears=cfg.nl * len(KINDS), torch_equal=n_eq, max_rel_other=max_rel)
        print(f"step0 weights: {n_eq}/{cfg.nl * len(KINDS)} torch.equal, max rel of others {max_rel:.2e}", flush=True)
        torch.set_float32_matmul_precision("high")
        kq0 = quick_eval()
        res["quick_kl_step0"] = kq0
        print(f"step0 quick heldout KL (bf16 student) {kq0:.5f}", flush=True)

    # ---- data
    tr = np.load(TRAIN_NPY)
    ntr = len(tr) // CHUNK
    tr = torch.from_numpy(tr[:ntr * CHUNK].astype(np.int64)).view(ntr, CHUNK)
    perm = torch.randperm(ntr, generator=torch.Generator().manual_seed(a.seed))
    cal_hashes = {hashlib.sha256(ids_all[i].numpy().tobytes()).hexdigest() for i in range(min(90, nch))}
    used_hashes = set()

    # ---- optimiser
    lat_keyed = [(k, t) for pl in PL for nm in KINDS for p in pl[nm] if p.dense is None
                 for k, t in zip(("r", "c", "d"), p.params())]
    norm_all = [t["d_" + k] for t in tun for k in ("ln1", "ln2", "qb", "kb", "vb")] + [d_fn]
    for k, t in lat_keyed:
        if k not in train_set:
            t.requires_grad_(False)
    if "norm" not in train_set:
        for t in norm_all:
            t.requires_grad_(False)
    lat = [t for k, t in lat_keyed if k in train_set]
    norm = norm_all if "norm" in train_set else []
    coded = [(li, nm, p) for li in range(cfg.nl) for nm in KINDS for p in PL[li][nm] if p.dense is None and p.idx is not None]
    groups, base_lr = [], []
    if lat:
        groups.append(dict(params=lat, lr=a.lr_lat)); base_lr.append(a.lr_lat)
    if norm:
        groups.append(dict(params=norm, lr=a.lr_norm)); base_lr.append(a.lr_norm)
    corr_g = [t for pl in PL for nm in KINDS for p in pl[nm] if p.U is not None for t in (p.U, p.V)]
    if corr_g:
        groups.append(dict(params=corr_g, lr=a.lr_corr)); base_lr.append(a.lr_corr)
    assert groups, "--train selected no tensor"
    ste = a.codes == "ste"
    code_kinds = list(KINDS) if a.code_kinds == "all" else a.code_kinds.split(",")
    ste_planes = []
    if ste:
        assert all(k in KINDS for k in code_kinds), (code_kinds, KINDS)
        for li, nm, p in coded:
            if nm in code_kinds:
                p.L = p.S(C).clone().requires_grad_()
                p.idx0, p.idx_prev = p.idx.clone(), p.idx.clone()
                p.pnear = None
                ste_planes.append(p)
        res.update(lr_code=a.lr_code, codes_trained_kinds=code_kinds, code_planes=len(ste_planes),
                   code_blocks=sum(p.idx.numel() for p in ste_planes), code_init=a.code_init, code_opt=a.code_opt)
        if a.code_init == "preimage":
            # Per linear at step 0: R_k = teacher weight - (every other plane and dense corrector) restricted to p.rows, then
            # P = fwht128(R_k / (r_rep * c)) / d with the quantized r, c, d of plane_weight (inverse of compose and fwht128).
            # Temporal per coded plane one plane_weight plus one fwht128 plus preimage_init; spatial ~10 * O*n words per linear.
            mine, nmis, nclip, nnear, nfall, nblk = {id(p) for p in ste_planes}, 0, 0, 0, 0, 0
            with torch.no_grad():
                for li in range(cfg.nl):
                    for nm in KINDS:
                        if not any(id(p) in mine for p in PL[li][nm]):
                            continue
                        W, Wi = None, []
                        for p in PL[li][nm]:
                            if p.dense is not None:
                                W = W + corr_weight(p)
                                Wi.append(None)
                                continue
                            w = plane_weight(p, p.S(C))
                            Wi.append(w)
                            W = add_plane(W, p, w)
                        Rfull = TLg[li][nm].float() - W
                        for p, w in zip(PL[li][nm], Wi):
                            if id(p) not in mine:
                                continue
                            R = (Rfull if p.rows is None else Rfull[p.rows]) + w
                            r, c, d = p.qvals()
                            P = ft.fwht128(R / (r.repeat_interleave(GROUP, 1) * c[None, :])) / d[None, :]
                            bad = (~torch.isfinite(P)).view(-1, 8).any(1) | (d == 0).view(1, -1, 8).any(-1).expand(p.O, -1).reshape(-1)
                            P.view(-1, 8)[bad] = C[p.idx.reshape(-1).long()[bad]]
                            L0, p.pnear, nc, nn_ = preimage_init(P, p.idx, C, nn)
                            _, back = project_local(L0, p.idx, C, nn)
                            nmis += int((back != p.idx).sum())
                            nclip += nc
                            nnear += nn_
                            nfall += int(bad.sum())
                            nblk += p.idx.numel()
                            p.L = L0.clone().requires_grad_()
                            del R, P, L0, w
                        del W, Wi, Rfull
            torch.cuda.empty_cache()
            res.update(step0_projection_mismatch=nmis, preimage_nearest_is_idx=nnear / nblk, preimage_clipped=nclip / nblk,
                       preimage_fallback_blocks=nfall, preimage_blocks=nblk)
            res["controls"]["step0_projection_mismatch_zero"] = dict(mismatch=nmis, blocks=nblk, ok=nmis == 0)
            print(f"preimage: blocks {nblk} fallback {nfall} nearest_is_idx {nnear / nblk:.4f} clipped {nclip / nblk:.4f} "
                  f"step0_projection_mismatch {nmis}", flush=True)
            dump()
            assert nmis == 0, f"step0_projection_mismatch {nmis}"
        if a.code_opt == "adam":
            groups.append(dict(params=[p.L for p in ste_planes], lr=a.lr_code))
            base_lr.append(a.lr_code)
        print(f"ste: {len(ste_planes)} coded planes, {res['code_blocks']} blocks, kinds {code_kinds}, lr_code {a.lr_code}", flush=True)
    ang = list(TH.values())
    if ang:
        groups.append(dict(params=ang, lr=a.lr_angle)); base_lr.append(a.lr_angle)
    opt = torch.optim.Adam(groups, fused=True)
    res.update(train_groups=sorted(train_set), trainable_params=sum(t.numel() for g in groups for t in g["params"]))
    print(f"train {sorted(train_set)}: {res['trainable_params']} scalar params in {len(groups)} groups", flush=True)
    code_changes = []
    if a.compose == "step":
        for li in range(cfg.nl):
            for nm in KINDS:
                for p in PL[li][nm]:
                    if p.dense is None:
                        p.Sc = p.S(C)
    if a.rotation == "kan":
        # control (a): step-0 composed weights of every linear under the register (all offsets 0) against the PlaneCompose
        # (fwht128) weights.  Under --hadamard butterfly the register is bit-identical to ft.fwht128, and under mm
        # kan_matrix at delta = 0 is bit-identical to had_mm's H128 / sqrt(128) with RegisterComposeMM running
        # PlaneComposeMM's op sequence, so the difference is exactly 0 in both hadamard modes: torch.equal and max_rel == 0.
        # Temporal two layer_dicts_step forwards; spatial two fp32 model copies held at once.
        with torch.no_grad():
            MODE.update(rot="had")
            Wb = layer_dicts_step()
            MODE.update(rot="kan")
            Wk = layer_dicts_step()
            rel_s0, rel_site, all_equal = 0.0, {}, True
            for li in range(cfg.nl):
                for nm in KINDS:
                    all_equal = all_equal and bool(torch.equal(Wk[li][nm], Wb[li][nm]))
                    rl = float((Wk[li][nm] - Wb[li][nm]).abs().max() / Wb[li][nm].abs().max())
                    rel_s0 = max(rel_s0, rl)
                    rel_site[KIND_SITE[nm]] = max(rel_site.get(KIND_SITE[nm], 0.0), rl)
            del Wb, Wk
        res["controls"]["kan_step0_vs_planecompose"] = dict(max_rel=rel_s0, max_rel_per_site=rel_site, torch_equal=all_equal,
                                                            bound=0.0, ok=all_equal and rel_s0 == 0.0)
        print(f"kan step0 composed weights vs PlaneCompose: max rel {rel_s0:.3e}, torch.equal {all_equal} "
              f"(bound exactly 0; T(0) == HN bitwise under mm)", flush=True)
        dump()
        assert res["controls"]["kan_step0_vs_planecompose"]["ok"], res["controls"]["kan_step0_vs_planecompose"]
    if a.compose != "layer" or a.kl != "ckpt":
        # control: on two calibration chunks, loss and every trainable gradient under the requested (compose, kl) against
        # (layer, ckpt).  Temporal two forward+backward passes at batch 2; spatial one fp32 copy of the trainable gradients.
        angle_ids = {id(t) for t in ang}
        tparams = [t for g in groups for t in g["params"] if id(t) not in angle_ids]

        def probe(mode):
            MODE.update({"amp": True, "rot": "had", **mode})
            opt.zero_grad(set_to_none=True)
            loss_, agree_ = kl_batch(ids_cal[:2].to(DEV), True)
            loss_.backward()
            probe.ang_g2 = sum(float(t.grad.pow(2).sum()) for t in ang if t.grad is not None) if ang else 0.0
            gs = torch.cat([(t.grad if t.grad is not None else torch.zeros_like(t)).reshape(-1) for t in tparams])
            opt.zero_grad(set_to_none=True)
            return float(loss_.detach()), agree_, gs

        # Under --hadamard mm the composed weight differs from fwht128's by fp32 summation order, which the bf16 cast inside
        # the layers amplifies (the jitter arm, a seeded two-ulp fp32 perturbation of the butterfly weight, shows the size of
        # that for an isotropic perturbation).  Equality of the mm arithmetic is gated in float64 on real planes:
        # PlaneComposeMM against PlaneCompose, forward and the d, r, c adjoints, max relative < 1e-12.  At model level the
        # fp32 student arms hold loss relative < 1e-5 and gradient relative L2 < 1e-2 (the bound of the butterfly fast
        # path), the bf16 mm arms gradient relative L2 < 1e-1.  Every butterfly arm stays bit-equal.
        l0, g0, gs0 = probe(dict(compose="layer", kl="ckpt", had="butterfly"))
        arms, wdiff = {}, {}
        arm_modes = [("ref_again", dict(compose="layer", kl="ckpt", had="butterfly")),
                     ("compose_only", dict(compose=a.compose, kl="ckpt", had="butterfly")),
                     ("kl_only", dict(compose="layer", kl=a.kl, had="butterfly"))]
        if a.hadamard == "mm":
            arm_modes += [("jitter", dict(compose="step", kl="ckpt", had="jitter")),
                          ("had_only", dict(compose="step", kl="ckpt", had="mm"))]
            # float64 PlaneComposeMM vs PlaneCompose on the first 8 coded planes, seeded upstream gradient.  Temporal
            # 8 x (two forwards, two backwards) at O*n words; spatial ~10 O*n float64 words per plane.
            unit, gq = [], torch.Generator(device=DEV)
            gq.manual_seed(0)
            for p in [p for pl in PL for nm in KINDS for p in pl[nm] if p.dense is None][:8]:
                r_, c_, d_ = (t.detach().double() for t in p.qvals())
                Sd = p.Sc.double()
                Gq = torch.randn(Sd.shape, generator=gq, device=DEV, dtype=torch.float64)
                o_ = []
                for F_ in (PlaneCompose, PlaneComposeMM):
                    dd, rr, cc = (t.clone().requires_grad_() for t in (d_, r_, c_))
                    with torch.autocast("cuda", enabled=False):
                        Wq = F_.apply(Sd, dd, rr, cc)
                        Wq.backward(Gq)
                    o_.append((Wq.detach(), dd.grad, rr.grad, cc.grad))
                unit.append(max(float((x - y).abs().max() / y.abs().max()) for x, y in zip(o_[1], o_[0])))
            print(f"grad_equiv float64 PlaneComposeMM vs PlaneCompose, 8 planes: max rel {max(unit):.2e}", flush=True)
            fp32_modes = [("fp32_ref", dict(compose="layer", kl="ckpt", had="butterfly", amp=False)),
                          ("fp32_ref_again", dict(compose="layer", kl="ckpt", had="butterfly", amp=False)),
                          ("fp32_had", dict(compose="step", kl="ckpt", had="mm", amp=False))]
            # composed weights of every linear: fp32 max relative difference and bf16 rounding flips against butterfly.
            # Temporal three layer_dicts_step forwards; spatial two fp32 model copies held at once.
            with torch.no_grad():
                MODE.update(had="butterfly")
                Wb = layer_dicts_step()
                for md_ in ("mm", "jitter"):
                    MODE.update(had=md_)
                    Wx, rel, flips, zflips, tot, dn2, n2 = layer_dicts_step(), 0.0, 0, 0, 0, 0.0, 0.0
                    for li in range(cfg.nl):
                        for nm in KINDS:
                            A, B = Wb[li][nm], Wx[li][nm]
                            Ab, Bb = A.bfloat16(), B.bfloat16()
                            rel = max(rel, float((B - A).abs().max() / A.abs().max()))
                            flips += int((Ab != Bb).sum())
                            zflips += int(((A == 0) & (B != 0)).sum())
                            dn2 += float((Bb.float() - Ab.float()).pow(2).sum())
                            n2 += float(Ab.float().pow(2).sum())
                            tot += A.numel()
                    wdiff[md_] = dict(w_rel_max=rel, bf16_flips=flips, words=tot, bf16_flip_frac=flips / tot,
                                      zero_to_nonzero=zflips, bf16_rel_l2=(dn2 / n2) ** 0.5)
                    print(f"grad_equiv weights {md_}: fp32 max rel {rel:.3e}, bf16 flips {flips}/{tot}, zero->nonzero {zflips}, "
                          f"bf16 rel L2 {wdiff[md_]['bf16_rel_l2']:.3e}", flush=True)
                    del Wx
                del Wb
        arm_modes.append(("requested", dict(compose=a.compose, kl=a.kl, had=a.hadamard, rot=a.rotation)))
        fp32, fp32_ok = {}, True
        if a.rotation == "kan":
            # as the mm arm: the register weight differs from fwht128's by fp32 rounding, which the bf16 layers amplify, so
            # equality is gated in fp32 (amp off): loss relative < 1e-5 and gradient relative L2 < 1e-2 against the butterfly.
            # The kan arm runs in the requested hadamard mode (had=a.hadamard), so under --hadamard mm it takes the mm path;
            # its fp32 keys are prefixed kan_ so the mm block's fp32_ref cannot clobber them.
            lf, gf, gsf = probe(dict(compose="layer", kl="ckpt", had="butterfly", amp=False))
            fp32["kan_ref"] = dict(loss=lf, agree=gf)
            for nm_, md in (("kan_ref_again", dict(compose="layer", kl="ckpt", had="butterfly", amp=False)),
                            ("kan_arm", dict(compose="step", kl="ckpt", had=a.hadamard, rot="kan", amp=False))):
                l1, g1, gs1 = probe(md)
                fp32[nm_] = dict(loss=l1, agree=g1, loss_rel=abs(l1 - lf) / abs(lf),
                                 grad_rel_l2=float((gs1 - gsf).norm() / gsf.norm()), grad_cos=float(F.cosine_similarity(gs1, gsf, 0)))
                print(f"grad_equiv {nm_}: loss {l1!r} vs {lf!r}, grad rel L2 {fp32[nm_]['grad_rel_l2']:.3e}", flush=True)
            fp32["theta_grad_l2"] = probe.ang_g2 ** 0.5
            fp32_ok = (fp32["kan_arm"]["loss_rel"] < 1e-5 and fp32["kan_arm"]["grad_rel_l2"] < 1e-2
                       and math.isfinite(fp32["theta_grad_l2"]) and fp32["theta_grad_l2"] > 0.0)
            del gsf
        if a.hadamard == "mm":
            # Temporal three fp32 forward+backward passes at batch 2; spatial one more fp32 gradient copy.
            lf, gf, gsf = probe(fp32_modes[0][1])
            fp32["fp32_ref"] = dict(loss=lf, agree=gf)
            for nm_, md in fp32_modes[1:]:
                l1, g1, gs1 = probe(md)
                fp32[nm_] = dict(loss=l1, agree=g1, loss_rel=abs(l1 - lf) / abs(lf),
                                 grad_rel_l2=float((gs1 - gsf).norm() / gsf.norm()), grad_cos=float(F.cosine_similarity(gs1, gsf, 0)))
                print(f"grad_equiv {nm_}: loss {l1!r} vs {lf!r}, grad rel L2 {fp32[nm_]['grad_rel_l2']:.3e}", flush=True)
            fp32["bf16_vs_fp32_loss_rel"] = abs(l0 - lf) / abs(lf)
            fp32["float64_unit_max_rel"] = max(unit)
            fp32_ok = fp32_ok and max(unit) < 1e-12 and fp32["fp32_had"]["loss_rel"] < 1e-5 and fp32["fp32_had"]["grad_rel_l2"] < 1e-2
            del gsf
        if fp32:
            fp32["ok"] = fp32_ok
        for nm_, md in arm_modes:
            l1, g1, gs1 = probe(md)
            loss_rel = abs(l1 - l0) / abs(l0)
            arms[nm_] = dict(loss=l1, agree=g1, loss_equal=l1 == l0, loss_rel=loss_rel,
                             grad_rel_l2=float((gs1 - gs0).norm() / gs0.norm()), grad_cos=float(F.cosine_similarity(gs1, gs0, 0)))
            if a.rotation == "kan":
                arms[nm_]["theta_grad_l2"] = probe.ang_g2 ** 0.5
            print(f"grad_equiv {nm_}: loss {l1!r} vs {l0!r}, agree {g1} vs {g0}, grad rel L2 {arms[nm_]['grad_rel_l2']:.3e} "
                  f"cos {arms[nm_]['grad_cos']:.8f}", flush=True)
        for nm_, md in arm_modes:
            v = arms[nm_]
            if md["had"] == "mm":
                v["loss_ok"] = fp32["ok"] and v["grad_rel_l2"] < 1e-1
            elif md.get("rot") == "kan":
                v["loss_ok"] = fp32["ok"] and v["grad_rel_l2"] < 1e-1 and v["theta_grad_l2"] > 0.0
            else:
                v["loss_ok"] = md["had"] == "jitter" or v["loss_equal"]
        MODE.update(compose=a.compose, kl=a.kl, had=a.hadamard, amp=True, rot=a.rotation)
        floor = arms["ref_again"]["grad_rel_l2"]
        grad_ok = arms["requested"]["grad_rel_l2"] < (1e-2 if a.hadamard == "butterfly" and a.rotation == "had" else 1e-1)
        res["controls"]["grad_equiv"] = dict(mode=dict(compose=a.compose, kl=a.kl, had=a.hadamard, **({"rot": a.rotation} if a.rotation == "kan" else {})), loss_ref=l0, agree_ref=g0,
                                             arms=arms, weights=wdiff, fp32=fp32, noise_floor_rel_l2=floor,
                                             ok=all(v["loss_ok"] for v in arms.values()) and grad_ok)
        dump()
        assert res["controls"]["grad_equiv"]["ok"], res["controls"]["grad_equiv"]
        del gs0, gs1
        torch.cuda.empty_cache()

    def rec_codes(step_, loss_):
        """Blocks whose index differs from step 0 (cumulative) and from the last record (window).
        Temporal one compare per block (M blocks over all trained planes); spatial one bool temporary per plane."""
        cum = sum(int((p.idx != p.idx0).sum()) for p in ste_planes)
        win_ = sum(int((p.idx != p.idx_prev).sum()) for p in ste_planes)
        rec = {}
        if a.code_init == "preimage":
            tc = sum(int(((p.idx != p.idx0) & (p.idx == p.pnear)).sum()) for p in ste_planes)
            tw = sum(int(((p.idx != p.idx_prev) & (p.idx == p.pnear)).sum()) for p in ste_planes)
            rec = dict(flips_toward_preimage=tc / cum if cum else None, flips_toward_preimage_window=tw / win_ if win_ else None,
                       flips_toward_preimage_cum_count=tc, flips_toward_preimage_window_count=tw)
        for p in ste_planes:
            p.idx_prev = p.idx.clone()
        nb = res["code_blocks"]
        code_changes.append(dict(step=step_, changed_cum=cum, changed_window=win_, frac_cum=cum / nb, frac_window=win_ / nb, train_loss=loss_, **rec))
        print(f"codes step {step_}: changed cum {cum} ({cum / nb:.3e}) window {win_} ({win_ / nb:.3e}) loss {loss_:.4f}"
              + (f" toward_preimage cum {rec['flips_toward_preimage']} window {rec['flips_toward_preimage_window']}" if rec else ""), flush=True)

    est_after = 1.15 * wall_eval + 12.0
    if a.attrib:
        # the AFTER attribution repeats every group eval, so the finishing loop must leave room for it too
        est_after += 1.15 * res["attrib"]["before"]["wall_s"]
    t_train0 = time.time()
    t_end = T0 + BUDGET_S - (est_after + 25.0)
    res["walls"]["projected_finish_s"] = est_after
    print(f"projected: elapsed {elapsed():.1f}s, train budget {t_end - t_train0:.1f}s, after-eval {est_after:.1f}s, "
          f"projected wall {(t_end - T0) + est_after:.1f}s of {BUDGET_S:.0f}s", flush=True)
    dump()

    def make_Ss(grad_leaves):
        """Per layer, the S field of every non-dense plane in layer_dict order; the coded ones are fresh leaves when grad_leaves.
        Temporal one gather C[idx] per coded plane (M blocks of 8 words); spatial the S fields of the whole model (fp32, ~1.4 GB at 0.5B)."""
        Ss_all, leaves = [], []
        for li in range(cfg.nl):
            row = []
            for nm in KINDS:
                for p in PL[li][nm]:
                    if p.dense is not None:
                        continue
                    S = p.S(C)
                    if grad_leaves and p.idx is not None:
                        S = S.requires_grad_()
                        leaves.append(S)
                    row.append(S)
            Ss_all.append(row)
        return Ss_all, leaves

    # window statistics for --v-gain obs: gsum += g, fsum += g*g per coded plane over the P-steps since the last V-step
    gsum = [torch.zeros(p.O, p.n, device=DEV) for _, _, p in coded] if a.v_gain == "obs" else []
    fsum = [torch.zeros(p.O, p.n, device=DEV) for _, _, p in coded] if a.v_gain == "obs" else []
    win = dict(n=0, loss=0.0)

    def make_hook(k):
        def hook(S):
            g = S.grad
            gsum[k] += g
            fsum[k] += g * g
            S.grad = None
        return hook

    def v_step(step, idsb, idss, v_frac):
        """linear: gradient on idsb, candidate moves scored on the disjoint train batch idss (no selection on the gradient batch).
        obs: gain from the window statistics accumulated by the P-steps (no gradient pass here); scored on idss the same way."""
        obs = a.v_gain == "obs"
        if obs:
            assert win["n"] > 0, "no P-step in the window"
            lossg = win["loss"] / win["n"]
            with torch.no_grad():
                loss0 = kl_batch(idss, False)[0].item()
            gains_by_lam = {lam: [] for lam in LAMS}
            for k, (li, nm, p) in enumerate(coded):
                gl, nl_ = block_gain_obs(gsum[k], fsum[k], win["n"], p.S(C), p.idx, C, nn, LAMS)
                for i, lam in enumerate(LAMS):
                    gains_by_lam[lam].append((gl[i], nl_[i]))
                del gl, nl_
            gains = gains_by_lam[LAMS[0]]
        else:
            Ss_all, leaves = make_Ss(True)
            loss, _ = kl_batch(idsb, True, Ss_all)
            grads = torch.autograd.grad(loss, leaves)
            lossg = loss.item()
            with torch.no_grad():
                loss0 = kl_batch(idss, False)[0].item()
            gains = []
            for (li, nm, p), S, g in zip(coded, leaves, grads):
                gains.append(block_gain(g, S.detach(), p.idx, C, nn))
            del Ss_all, leaves, grads, loss
        # gain distribution over positive-gain blocks (all planes): the linearisation is trusted only where the gain is moderate
        allpos = torch.cat([g[g > 0] for g, _ in gains])
        qs = torch.quantile(allpos[::max(1, allpos.numel() >> 22)], torch.tensor([0.5, 0.9, 0.99, 0.999], device=DEV))
        gstat = dict(n_pos=int(allpos.numel()), p50=float(qs[0]), p90=float(qs[1]), p99=float(qs[2]), p999=float(qs[3]), max=float(allpos.max()),
                     sum_top_1e3=float(torch.topk(allpos, max(1, allpos.numel() // 1000))[0].sum()), sum=float(allpos.sum()))
        del allpos
        tries = []

        def apply(f, skip, gains=gains, of_pos=False):
            """Move the blocks ranked [skip*P, skip*P + k) by gain among each plane's P positive-gain blocks, k = f*M (linear)
            or f*P (of_pos, obs); returns (moves, pred, undo).
            Temporal one topk of the plane's M gains per plane (O(M log k)); spatial one undo list of <= k entries per plane."""
            moves, pred, undo = 0, 0.0, []
            for (li, nm, p), (gain, new) in zip(coded, gains):
                P = int((gain > 0).sum())
                if of_pos and P == 0:
                    continue
                k = max(1, int(f * (P if of_pos else gain.numel())))
                lo = int(skip * P)
                top, sel = torch.topk(gain, min(P, lo + k))
                sel = sel[lo:][top[lo:] > 0]
                flat = p.idx.view(-1)
                undo.append((flat, sel, flat[sel].clone()))
                flat[sel] = new[sel]
                moves += int(sel.numel()); pred += float(gain[sel].sum())
            return moves, pred, undo

        def revert(undo):
            for flat, sel, old in undo:
                flat[sel] = old

        # Trial grid (skip x f); each is applied, scored on the same batch with no grad, and reverted; the best one that lowers the loss is re-applied.
        # Cost: |grid| forwards of the batch (temporal), one undo list of <= f*M entries per plane (spatial).
        if obs:
            # Trial grid (lam x fraction of positive-gain blocks per plane): 8 forwards of idss.
            for lam in LAMS:
                for f in OBS_FRACS:
                    moves, pred, undo = apply(f, 0.0, gains_by_lam[lam], True)
                    with torch.no_grad():
                        loss1 = kl_batch(idss, False)[0].item()
                    revert(undo)
                    tries.append(dict(lam=lam, fraction=f, moves=moves, loss_after=loss1, predicted_dloss=-pred))
        else:
            for skip in (0.1, 0.5, 0.9, 0.97):
                for f in (v_frac, v_frac / 4):
                    moves, pred, undo = apply(f, skip)
                    with torch.no_grad():
                        loss1 = kl_batch(idss, False)[0].item()
                    revert(undo)
                    tries.append(dict(skip=skip, v_frac=f, moves=moves, loss_after=loss1, predicted_dloss=-pred))
        best = min(range(len(tries)), key=lambda i: tries[i]["loss_after"])
        accepted = tries[best]["loss_after"] < loss0
        if accepted:
            if obs:
                apply(tries[best]["fraction"], 0.0, gains_by_lam[tries[best]["lam"]], True)
            else:
                apply(tries[best]["v_frac"], tries[best]["skip"])
        window_steps = win["n"]
        if obs:
            for t in gsum + fsum:
                t.zero_()
            win["n"], win["loss"] = 0, 0.0
            del gains_by_lam
        del gains
        torch.cuda.empty_cache()
        return dict(step=step, accepted=accepted, moves=tries[best]["moves"] if accepted else 0, best=best if accepted else None,
                    loss_grad_batch=lossg, loss_before=loss0, window_steps=window_steps, gain_stats=gstat, tries=tries), v_frac

    def r_sat_mean():
        """Mean of Plane.sat_frac over the non-dense planes, weighted by numel of the r latents (O*G per plane)."""
        tot, den = 0.0, 0
        for li in range(cfg.nl):
            for nm in KINDS:
                for p in PL[li][nm]:
                    if p.dense is None:
                        n = p.pa.numel()
                        tot += p.sat_frac() * n
                        den += n
        return tot / den if den else None

    def regrid_all():
        for li in range(cfg.nl):
            for nm in KINDS:
                for p in PL[li][nm]:
                    if p.dense is None and p.pa.requires_grad:
                        p.regrid()

    res.update(regrid=a.regrid, r_sat_frac_before=r_sat_mean())

    # ---- train
    traj, vlog, step, v_frac = [], [], 0, a.v_frac
    last_loss = float("nan")
    prof = None
    # ---- --block-s: AQLM phase 3 runs from step 1 while the train clock is under a.block_s, then hands the run to the KL phase
    block_mode = a.block_s > 0
    block_on, switch_step = block_mode, None
    b_steps, b_first, b_last, b_e_first, b_e_last, b_switch_kl, b_wall = 0, None, None, None, None, None, 0.0
    idsb_first = None
    while step < a.steps and time.time() < t_end:
        step += 1
        if block_on and time.time() - t_train0 >= a.block_s:
            # first step whose train clock has reached a.block_s: the block phase ends, the KL phase starts here.  The Adam
            # moments of the block phase are discarded and the warmup restarts from this step; the cosine part is unchanged.
            block_on, switch_step = False, step
            opt.state.clear()
            b_wall = time.time() - t_train0
            with torch.no_grad():
                b_switch_kl = quick_eval()
            if b_steps:
                b_last, b_e_last = block_probe(idsb_first)  # the block-phase-final weights, before any KL update
            print(f"block phase: {b_steps} steps in {b_wall:.1f}s, Adam state cleared, warmup restarted at step {step}, "
                  f"quick heldout KL {b_switch_kl:.5f}", flush=True)
        if a.profile and step == 6:
            torch.cuda.synchronize()
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
            t_prof0 = time.time()
        if prof is not None and step == 6 + a.profile:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            wall_p = time.time() - t_prof0
            ka = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
            # kernel events only: an aten op's self device time repeats the time of the kernels it launched
            dev_total = sum(e.self_device_time_total for e in ka if e.device_type == torch.autograd.DeviceType.CUDA)
            res["profile"] = dict(steps=a.profile, wall_s=wall_p, s_per_step=wall_p / a.profile, self_device_s=dev_total / 1e6,
                                  ops=[dict(name=e.key, calls=e.count, device_s=e.device_time_total / 1e6,
                                            self_device_s=e.self_device_time_total / 1e6) for e in ka[:40]])
            print(f"profile: {a.profile} steps in {wall_p:.2f}s ({wall_p / a.profile:.3f} s/step), self device {dev_total / 1e6:.2f}s", flush=True)
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), flush=True)
            dump()
            sys.exit(0)
        frac = max(min(1.0, (time.time() - t_train0) / max(t_end - t_train0, 1e-6)), step / a.steps)
        wu = (step - switch_step) / a.warmup if switch_step is not None else step / a.warmup
        f = min(1.0, wu) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = base_lr[gi] * f
        idsb = tr[perm[((step - 1) * a.batch) % ntr:((step - 1) * a.batch) % ntr + a.batch]]
        assert idsb.shape[0] == a.batch, "train chunk wrap"
        for i in range(idsb.shape[0]):
            used_hashes.add(hashlib.sha256(idsb[i].numpy().tobytes()).hexdigest())
        idsb = idsb.to(DEV)
        if a.regrid and step % a.regrid == 0:
            regrid_all()
        if a.v_every and not ste and step % a.v_every == 0 and step < a.steps:
            j = (ntr // 2 + (step // a.v_every) * a.batch) % ntr
            idss = tr[perm[j:j + a.batch]]
            assert idss.shape[0] == a.batch, "score chunk wrap"
            for i in range(idss.shape[0]):
                used_hashes.add(hashlib.sha256(idss[i].numpy().tobytes()).hexdigest())
            rec, v_frac = v_step(step, idsb, idss.to(DEV), v_frac)
            vlog.append(rec)
            print(f"V-step {step}: accepted {rec['accepted']} moved {rec['moves']} blocks; loss {rec['loss_before']:.4f}; gains {rec['gain_stats']}", flush=True)
            for t in rec["tries"]:
                tag = f"lam {t['lam']} frac {t['fraction']}" if "lam" in t else f"skip {t['skip']} f={t['v_frac']:.2e}"
                print(f"   {tag}: {t['moves']} mv, pred {t['predicted_dloss']:.3g}, loss {t['loss_after']:.4f}"
                      f" (realised {t['loss_after'] - rec['loss_before']:+.4g})", flush=True)
        if block_on:
            loss, e = block_batch(idsb)
            agree = float("nan")
            b_steps += 1
            if b_steps == 1:
                b_first, b_e_first, idsb_first = float(loss.detach()), e, idsb
            if b_steps % 25 == 0:
                print(f"block step {b_steps} loss {float(loss.detach()):.4f} t {time.time() - t_train0:.1f}s", flush=True)
        elif a.v_gain == "obs":
            Ss_all, leaves = make_Ss(True)
            for k, S in enumerate(leaves):
                S.register_post_accumulate_grad_hook(make_hook(k))
            loss, agree = kl_batch(idsb, True, Ss_all)
        else:
            ste_on[0] = ste
            loss, agree = kl_batch(idsb, True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ste and a.code_opt == "nsgd":
            with torch.no_grad():
                for p in ste_planes:
                    g = p.L.grad
                    if g is not None:
                        rms = g.pow(2).mean().sqrt()
                        if float(rms) > 0.0 and bool(torch.isfinite(rms)):
                            p.L.add_(g / rms, alpha=-a.lr_code * f)
                    p.L.grad = None
        last_loss = loss.item()
        if ste:
            for p in ste_planes:
                p.idx = pending[id(p)]
            pending.clear()
            if step % a.rec_every == 0:
                rec_codes(step, last_loss)
        if a.v_gain == "obs":
            win["n"] += 1
            win["loss"] += last_loss
            del Ss_all, leaves
        if step == 1 or step % 25 == 0:
            traj.append(dict(step=step, train_loss=last_loss, top1=agree, lr_factor=f, t=time.time() - t_train0))
            print(f"step {step} loss {last_loss:.4f} top1 {agree:.4f} f {f:.3f} t {time.time() - t_train0:.1f}s elapsed {elapsed():.1f}s", flush=True)
    wall_train = time.time() - t_train0
    ste_on[0] = False
    if block_mode:
        # the block phase ended at the switch or, if the loop ran out first, at loop exit; either way it is over here.
        if switch_step is None:
            b_wall = wall_train
            if b_steps:
                b_last, b_e_last = block_probe(idsb_first)
        res["block"] = dict(block_s=a.block_s, steps=b_steps, wall_s=b_wall,
                            s_per_step=b_wall / b_steps if b_steps else None, loss_first=b_first, loss_last=b_last,
                            layer_err_first=b_e_first, layer_err_last=b_e_last, quick_kl_at_switch=b_switch_kl)
        res["walls"]["block_s"] = b_wall
        print(f"block: {b_steps} steps, {b_wall:.1f}s" + (f" ({b_wall / b_steps:.3f} s/step)" if b_steps else "")
              + f", loss {b_first} -> {b_last} (both on the first block step's batch)", flush=True)
        assert b_steps and b_last < b_first, "block phase did not lower the block loss"
    if ste:
        if not code_changes or code_changes[-1]["step"] != step:
            rec_codes(step, last_loss)
        res.update(code_changes=code_changes, code_changed_total=sum(int((p.idx != p.idx0).sum()) for p in ste_planes))
        for p in ste_planes:
            p.L = None
        torch.cuda.empty_cache()
    with torch.no_grad():
        kq1 = quick_eval()
    print(f"trained {step} steps in {wall_train:.1f}s; quick heldout KL {kq1:.5f} (step0 {kq0:.5f})", flush=True)
    res.update(steps=step, tokens=step * a.batch * CHUNK, batch=a.batch, train_trajectory=traj, v_steps=vlog, quick_kl_step0=kq0,
               quick_kl_final=kq1, train_loss_last=last_loss)
    res["walls"]["train_s"] = wall_train
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    inter = used_hashes & cal_hashes
    res["controls"]["data_disjoint"] = dict(train_chunks_used=len(used_hashes), calib_chunks_checked=len(cal_hashes), overlap=len(inter),
                                            ok=not inter, train_npy_sha256=sha_file(TRAIN_NPY))
    assert not inter, "train chunk equals a calib-c4 chunk"
    del opt
    torch.cuda.empty_cache()

    res.update(r_sat_frac_after=r_sat_mean())  # no regrid here: a grid change with no training step after it is untrained

    # ---- final stored values, integer codes, eval weights
    t0 = ft.tick()
    torch.set_float32_matmul_precision("highest")
    Ls2, n_eq, n_ne, max_rel = [], 0, 0, 0.0
    bits = dict(bits_sign=0, bits_r=0, bits_c=0, bits_d=0)
    corr_rel = []  # per dense plane: ||Corr_after - Corr_before|| / ||Corr_before||, Corr_after = f16r(U) @ f16r(V)
    code_ok = True
    THq = {k: kp.quant_angles(v.detach(), ANGLE_BITS) for k, v in TH.items()}  # the stored 16-bit offsets d = theta - pi/4; pi/4 is grid point 8192 of 2 pi / 65536, so round(d / q) q is the theta grid
    with torch.no_grad():
        for li in range(cfg.nl):
            L = dict(Ls[li])
            t = tuned(li)
            for k, v in t.items():
                L[k] = v.detach().cpu()
            if li == cfg.nl - 1:
                L["fnorm"] = (fn0 * torch.exp(d_fn)).detach().cpu()
            for nm in KINDS:
                st_planes, qt_planes = [], []
                for p in PL[li][nm]:
                    if p.dense is not None:
                        # --train corr: store the refactored corrector as its 8-bit coded factors' product (bits
                        # unchanged); otherwise the frozen dense plane verbatim.  Corr_after vs Corr_before recorded.
                        d_exp = p.dense if p.U is None else torch.matmul(*u8_factors(p)).detach()
                        if p.U is not None:
                            with torch.no_grad():
                                corr_rel.append(float((d_exp - p.dense).norm() / p.dense.norm().clamp_min(1e-12)))
                        st_planes.append(dict(dense=d_exp)); qt_planes.append(dict(dense=d_exp))
                        continue
                    r_l, c_l, d_l = p.latents()
                    rq, kint = p.r_grid(r_l)
                    cq, dq = ft.quant_c_u8(c_l, bits=6)[0], ft.quant_c_u8(d_l, bits=5)[0]
                    # integer codes -> rebuilt values
                    rbuilt = p.lo[:, None] + (kint / 15.0) * p.sp[:, None]
                    def affine(x_l, xq, bits_):
                        lv = float((1 << bits_) - 1)
                        mn = ft.f16r(x_l.min()); sp_ = (ft.f16r(x_l.max()) - mn).clamp_min(1e-12)
                        q = torch.round((xq - mn) / sp_ * lv)
                        return q, mn + (q / lv) * sp_, lv
                    cint, cbuilt, lc = affine(c_l, cq, 6)
                    dint, dbuilt, ld = affine(d_l, dq, 5)
                    code_ok &= bool(torch.equal(rbuilt, rq) and torch.equal(cbuilt, cq) and torch.equal(dbuilt, dq))
                    code_ok &= bool(kint.min() >= 0 and kint.max() <= 15 and cint.min() >= 0 and cint.max() <= lc and dint.min() >= 0 and dint.max() <= ld)
                    S = p.S(C)
                    if p.idx is not None:
                        code_ok &= bool(encode(S, C)[1] and p.idx.dtype == torch.uint8)
                    st_planes.append(dict(S_rot=S, r=rbuilt, c=cbuilt, d=dbuilt, rows=p.rows))
                    qt_planes.append(dict(S_rot=S, r=rq, c=cq, d=dq, rows=p.rows))
                    bits["bits_sign"] += (p.idx.numel() * 8 if p.idx is not None else S.numel())
                    bits["bits_r"] += 4 * kint.numel() + 8 * p.O + 64
                    bits["bits_c"] += 6 * cint.numel() + 32
                    bits["bits_d"] += 5 * dint.numel() + 32
                th_q = THq.get((li, KIND_SITE[nm]))
                if th_q is None:
                    W_st, W_qt = ft._sink_recon(st_planes), ft._sink_recon(qt_planes)
                else:
                    W_st, W_qt = kan_recon(st_planes, th_q), kan_recon(qt_planes, th_q)
                if torch.equal(W_st, W_qt):
                    n_eq += 1
                else:
                    n_ne += 1; max_rel = max(max_rel, float((W_st - W_qt).abs().max() / W_qt.abs().max()))
                L[nm] = W_st.float().cpu()
                del st_planes, qt_planes, W_st, W_qt
            Ls2.append(L)
    angle_bits = kp.NST * kp.NBF * ANGLE_BITS * len(THq)  # 7 * 64 * 16 per learning site per layer
    if corr_rel:
        res["corr"].update(k=sum(p.k for pl in PL for nm in KINDS for p in pl[nm] if p.U is not None),
                           planes=len(corr_rel), rel_mean=float(np.mean(corr_rel)), rel_max=float(max(corr_rel)),
                           bits_corr=binfo["bits_corr"])
        print(f"corr export: {len(corr_rel)} planes, ||Corr_after-Corr_before||/||Corr_before|| mean "
              f"{res['corr']['rel_mean']:.3e} max {res['corr']['rel_max']:.3e}, bits_corr {binfo['bits_corr']}", flush=True)
    bits_after = sum(bits.values()) + binfo["bits_corr"] + angle_bits
    comp_ok = all(bits[k] == binfo[k] for k in bits)
    res.update(bits_after=bits_after, bpw_after=bits_after / binfo["params"], bits_components_after=dict(bits, bits_corr=binfo["bits_corr"]))
    res["controls"]["quantized_eval_equality"] = dict(linears_equal=n_eq, linears_not_equal=n_ne, max_rel=max_rel,
                                                      integer_codes_roundtrip_and_in_range=code_ok)
    res["controls"]["bits_equal"] = dict(before=bits_before, after=bits_after, components_equal=comp_ok,
                                         ok=bits_after == bits_before + angle_bits and comp_ok)
    if a.rotation == "kan":
        # bits_equal reads after == before + angle_bits; the register angles are the only added stored bits.
        drift = {}
        for site in sorted(kan_sites):
            dd = torch.stack([THq[(li, site)].abs() for li in range(cfg.nl)])
            drift[site] = dict(max_abs=float(dd.max()), rms=float(dd.pow(2).mean().sqrt()),
                               per_layer_rms=[float(x) for x in dd.pow(2).mean((1, 2)).sqrt()])
        res["controls"]["bits_equal"].update(angle_bits=angle_bits, rule="after == before + angle_bits")
        res["rotation"] = dict(mode="kan", sites=sorted(kan_sites), lr_angle=a.lr_angle, angle_bits_each=ANGLE_BITS, angle_bits=angle_bits,
                               bpw_after=res["bpw_after"], bpw_angle=angle_bits / binfo["params"], theta_drift_vs_pi4=drift,
                               kan_probe_sha256=sha_file(kp.__file__))
    print(f"final weights built {ft.tick() - t0:.1f}s; eval-vs-quantizer equal {n_eq}/{n_eq + n_ne} code_ok {code_ok} bits {bits_before}->{bits_after}", flush=True)
    for t in (PL, tun, TLg, lat, norm):
        t.clear()
    torch.cuda.empty_cache()

    kl_ev1, t1_ev1 = ft.kl_eval(cfg, TL, Ls2, ids_ev)
    kl_in1, t1_in1 = ft.kl_eval(cfg, TL, Ls2, ids_in)
    res.update(kl_heldout_after=kl_ev1, top1_heldout_after=t1_ev1, kl_insample_after=kl_in1, top1_insample_after=t1_in1)
    if a.attrib:
        res["attrib"]["after"] = attrib(cfg, TL, Ls2, ids_at)
        attrib_line("after", res["attrib"]["after"])
        res["walls"]["attrib_after_s"] = res["attrib"]["after"]["wall_s"]
    res["walls"]["finish_s"] = wall_train + (ft.tick() - t0)
    res["walls"]["total_s"] = elapsed()
    res["partial"] = False
    dump()
    print(f"after heldout KL {kl_ev1:.6f} (before {kl_ev0:.6f}) top1 {t1_ev1:.6f}  insample KL {kl_in1:.6f} (before {kl_in0:.6f})  total {elapsed():.1f}s", flush=True)
    assert res["controls"]["bits_equal"]["ok"] and code_ok and n_ne == 0, res["controls"]


if __name__ == "__main__":
    main()
