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
usage: pv_finish.py --out OUT.json [--smoke] [--v-gain {linear,obs}] [--codes {vstep,ste}] ; run with /home/socol/Workspace/spinor-ttl/.venv/bin/python (torch + CUDA).
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_transfer as ft
import transformers

DEV, GROUP, CHUNK = ft.DEV, ft.GROUP, ft.CHUNK
KINDS = list(ft.KINDS)
ARM = "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-resq-qmlp-nb"
TRAIN_NPY = "/home/socol/oc-tmp/lanes/w18/kd/c4data/train-c4.qwen.npy"
REF_RECEIPT = "/home/socol/Workspace/apeiron/docs/evidence/receipts/onebit/w18-size05-n50.json"
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
        self.dense = d["dense"].to(DEV) if "dense" in d else None
        if self.dense is not None:
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
    ap.add_argument("--compose", choices=("layer", "step"), default="layer",
                    help="layer: plane weights composed inside each layer checkpoint (forward and recompute); step: composed once "
                         "per step through PlaneCompose (frozen codes only: --v-every 0, --codes vstep)")
    ap.add_argument("--kl", choices=("ckpt", "fused"), default="ckpt",
                    help="ckpt: checkpoint(kl_chunk) recomputes both head matmuls in backward; fused: KLHead saves dL/dz")
    ap.add_argument("--train", default="r,c,d,norm",
                    help="comma subset of r (pa), c (pb), d (pd), norm (zero-bit d_* tunables and d_fn); unselected tensors are frozen")
    a = ap.parse_args()
    assert a.compose == "layer" or (a.v_every == 0 and a.codes == "vstep"), "--compose step needs frozen codes"
    train_set = set(a.train.split(","))
    assert train_set and train_set <= {"r", "c", "d", "norm"}, a.train
    LAMS = tuple(float(x) for x in a.obs_lams.split(","))
    if a.smoke:
        a.n_cal, a.n_eval, a.n_in, a.steps, a.batch = 8, 8, 8, 20, 2
        a.v_every = 10 if a.v_every else 0  # an explicit --v-every 0 stays off in smoke
        a.rec_every = 5
    res = dict(script_sha256=dict(pv_finish=sha_file(__file__), fold_transfer=sha_file(ft.__file__)), arm=ARM, args=vars(a),
               protocol="pack sink -> codes (E8/O8 index per 8-block) -> P-step Adam (STE) + V-step linearised code moves, KL(teacher||student)"
               + (" [v-gain obs: window gbar/Fisher, OBS diagonal gain]" if a.v_gain == "obs" else "")
               + (" [codes ste: latent L, nearest-of-16 projection, straight-through; V-steps off]" if a.codes == "ste" else ""),
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
    MODE = dict(compose=a.compose, kl=a.kl)

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
                    W = W + p.dense
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
        layer: forward and recompute); spatial the composed fp32 weights of the model (4 bytes per parameter) held until backward."""
        out = []
        for li in range(cfg.nl):
            Ld = tuned(li)
            for nm in KINDS:
                W = None
                for p in PL[li][nm]:
                    if p.dense is not None:
                        W = W + p.dense
                        continue
                    r, c, d = p.qvals()
                    W = add_plane(W, p, PlaneCompose.apply(p.Sc, d, r, c))
                Ld[nm] = W
            out.append(Ld)
        return out

    def student_h(idsb, grad, Ss_all):
        h = Eg[idsb]
        Lds = layer_dicts_step() if grad and MODE["compose"] == "step" and not Ss_all and not ste_on[0] else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
                                W = W + p.dense
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
    if a.compose != "layer" or a.kl != "ckpt":
        # control: on two calibration chunks, loss and every trainable gradient under the requested (compose, kl) against
        # (layer, ckpt).  Temporal two forward+backward passes at batch 2; spatial one fp32 copy of the trainable gradients.
        tparams = [t for g in groups for t in g["params"]]

        def probe(mode):
            MODE.update(mode)
            opt.zero_grad(set_to_none=True)
            loss_, agree_ = kl_batch(ids_cal[:2].to(DEV), True)
            loss_.backward()
            gs = torch.cat([(t.grad if t.grad is not None else torch.zeros_like(t)).reshape(-1) for t in tparams])
            opt.zero_grad(set_to_none=True)
            return float(loss_.detach()), agree_, gs

        l0, g0, gs0 = probe(dict(compose="layer", kl="ckpt"))
        arms = {}
        for nm_, md in (("ref_again", dict(compose="layer", kl="ckpt")), ("compose_only", dict(compose=a.compose, kl="ckpt")),
                        ("kl_only", dict(compose="layer", kl=a.kl)), ("requested", dict(compose=a.compose, kl=a.kl))):
            l1, g1, gs1 = probe(md)
            arms[nm_] = dict(loss=l1, agree=g1, loss_equal=l1 == l0, grad_rel_l2=float((gs1 - gs0).norm() / gs0.norm()),
                             grad_cos=float(F.cosine_similarity(gs1, gs0, 0)))
            print(f"grad_equiv {nm_}: loss {l1!r} vs {l0!r}, agree {g1} vs {g0}, grad rel L2 {arms[nm_]['grad_rel_l2']:.3e} "
                  f"cos {arms[nm_]['grad_cos']:.8f}", flush=True)
        MODE.update(compose=a.compose, kl=a.kl)
        floor = arms["ref_again"]["grad_rel_l2"]
        res["controls"]["grad_equiv"] = dict(mode=dict(compose=a.compose, kl=a.kl), loss_ref=l0, agree_ref=g0, arms=arms,
                                             noise_floor_rel_l2=floor,
                                             ok=all(v["loss_equal"] for v in arms.values()) and arms["requested"]["grad_rel_l2"] < 1e-2)
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
    while step < a.steps and time.time() < t_end:
        step += 1
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
        f = min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
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
        if a.v_gain == "obs":
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
    code_ok = True
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
                        st_planes.append(dict(dense=p.dense)); qt_planes.append(dict(dense=p.dense))
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
                W_st, W_qt = ft._sink_recon(st_planes), ft._sink_recon(qt_planes)
                if torch.equal(W_st, W_qt):
                    n_eq += 1
                else:
                    n_ne += 1; max_rel = max(max_rel, float((W_st - W_qt).abs().max() / W_qt.abs().max()))
                L[nm] = W_st.float().cpu()
                del st_planes, qt_planes, W_st, W_qt
            Ls2.append(L)
    bits_after = sum(bits.values()) + binfo["bits_corr"]
    comp_ok = all(bits[k] == binfo[k] for k in bits)
    res.update(bits_after=bits_after, bpw_after=bits_after / binfo["params"], bits_components_after=dict(bits, bits_corr=binfo["bits_corr"]))
    res["controls"]["quantized_eval_equality"] = dict(linears_equal=n_eq, linears_not_equal=n_ne, max_rel=max_rel,
                                                      integer_codes_roundtrip_and_in_range=code_ok)
    res["controls"]["bits_equal"] = dict(before=bits_before, after=bits_after, components_equal=comp_ok, ok=bits_after == bits_before and comp_ok)
    print(f"final weights built {ft.tick() - t0:.1f}s; eval-vs-quantizer equal {n_eq}/{n_eq + n_ne} code_ok {code_ok} bits {bits_before}->{bits_after}", flush=True)
    for t in (PL, tun, TLg, lat, norm):
        t.clear()
    torch.cuda.empty_cache()

    kl_ev1, t1_ev1 = ft.kl_eval(cfg, TL, Ls2, ids_ev)
    kl_in1, t1_in1 = ft.kl_eval(cfg, TL, Ls2, ids_in)
    res.update(kl_heldout_after=kl_ev1, top1_heldout_after=t1_ev1, kl_insample_after=kl_in1, top1_insample_after=t1_in1)
    res["walls"]["finish_s"] = wall_train + (ft.tick() - t0)
    res["walls"]["total_s"] = elapsed()
    res["partial"] = False
    dump()
    print(f"after heldout KL {kl_ev1:.6f} (before {kl_ev0:.6f}) top1 {t1_ev1:.6f}  insample KL {kl_in1:.6f} (before {kl_in0:.6f})  total {elapsed():.1f}s", flush=True)
    assert res["controls"]["bits_equal"]["ok"] and code_ok and n_ne == 0, res["controls"]


if __name__ == "__main__":
    main()
