"""W18-TRELLIS microbench: seeded bitshift-trellis generator (QTIP 1MAD hash, k = 1 bit/weight) vs E8 (gptq_e8) at 1.0 b/w,
on the one-bit arm's own rotated problem  J(Wr, Wq, Hr) = tr((Wr-Wq) Hr (Wr-Wq)^T),  Wr = fwht128(W), Hr = had_gram(Hl),
Hl = metric(teacher input Gram / tokens, 0.005 n / 896), Qwen2.5-0.5B layers {0, 11, 23}, calibration chunks [0, 20).

Trellis: state s in [0, 2^L), step s' = ((s << 1) | b) & (2^L - 1), emitted value v(s') = 1MAD(s') (Tseng et al. 2024,
arXiv 2406.11235) normalised to zero mean / unit RMS over the table.  Per row: 1 bit per weight + L bits start state.
Encoding: GPTQ order, feedback U = feedback_u(Hr, 0.02), blocks of 16 columns, per-survivor in-block feedback Viterbi
(survivors merge on state; the feedback history is the survivor's own).  Scale per (row, 128-group) a = c * w_rms, then the
same least-squares scale refit as the E8 baselines (beam 1 and beam 4).

usage: trellis_gen.py --out OUT.json [--quick] [--layers 0,23,11] [--cs 0.8,0.95,1.1]
Run with ~/spinor-ttl/.venv/bin/python (torch + CUDA).
"""
import argparse, hashlib, json, math, os, statistics, sys, time
import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fold_transfer as ft

DEV = ft.DEV
F64 = torch.float64
GROUP = ft.GROUP
CHUNK = ft.CHUNK
BLK = 16                       # columns per Viterbi block
DAMP = 0.02
ROWCAP = 8_000_000             # rows x states per Viterbi chunk (bounds the 2 x BLK x rows x states accumulators)
KINDS = ["q", "k", "v", "o", "gate", "up", "down"]
STAGE_OF = {"q": "xa", "k": "xa", "v": "xa", "o": "a", "gate": "xf", "up": "xf", "down": "m"}
INF = float("inf")
T0 = time.time()


# ------------------------------------------------------------------ generator tables
def mad1_table(L, dtype=torch.float32, device=DEV):
    """QTIP 1MAD over s' in [0, 2^L): x = (34038481 s' + 76625530) mod 2^32; v = (sum of 4 bytes of x - 510)/147.8;
    table normalised to zero mean, unit RMS.  O(2^L) time and space."""
    s = np.arange(1 << L, dtype=np.uint64)
    x = (np.uint64(34038481) * s + np.uint64(76625530)) & np.uint64(0xFFFFFFFF)
    by = (x & np.uint64(255)) + ((x >> np.uint64(8)) & np.uint64(255)) + ((x >> np.uint64(16)) & np.uint64(255)) + ((x >> np.uint64(24)) & np.uint64(255))
    v = (by.astype(np.float64) - 510.0) / 147.8
    v -= v.mean()
    v /= math.sqrt(float((v * v).mean()))
    return torch.tensor(v, dtype=dtype, device=device)


def sign_table(L, dtype=torch.float32, device=DEV):
    """Degenerate control generator: v(s') = +1 if s' & 1 else -1 (plain sign quantiser on the emitted bit).  O(2^L)."""
    s = torch.arange(1 << L, device=device)
    return torch.where((s & 1) == 1, 1.0, -1.0).to(dtype)


# ------------------------------------------------------------------ Viterbi with per-survivor feedback
def seq_errs(wt, Ub, a, qv):
    """Exact sequential-feedback errors e_t = (w_t - sum_{j<t} e_j U[j,t] - q_t) / U[t,t] of a fixed path.  O(Oc * B^2) time."""
    Oc, B = wt.shape
    errs = torch.zeros(Oc, B, dtype=wt.dtype, device=wt.device)
    for t in range(B):
        fb = errs[:, :t] @ Ub[:t, t] if t else 0.0
        errs[:, t] = (wt[:, t] - fb - a * qv[:, t]) / Ub[t, t]
    return errs


def viterbi_block(wt, Ub, a, vtab, s0, L, bufs):
    """One block of B columns for Oc rows.  wt (Oc,B) targets (earlier-block feedback already applied), Ub = U[b0:b1,b0:b1],
    a (Oc,) scale, vtab (2^L,), s0 (Oc,) known start state.  Returns (dp_cost, errs, qv, end_state, traceback_start).
    Time O(Oc * 2^L * B^2 / 2) element-ops (step t: where + addcmul over (B-t-1) x Oc x 2^L feedback accumulators),
    space 2 B Oc 2^L accumulators + B Oc 2^L bool backpointers.  Recurrence: state s'=2m+b has predecessors m and
    m + 2^(L-1) (top bit differs), cost(s') = min_x cost(m + x 2^(L-1)) + err^2, err from the survivor's own feedback."""
    Oc, B = wt.shape
    S = 1 << L
    half = S >> 1
    dt = wt.dtype
    Fo, Fn = bufs                     # (B, Oc, S): Fo[t'][o, s] = accumulated feedback of survivor s onto column t'
    Fo.zero_()
    ar = torch.arange(Oc, device=wt.device)
    cost = torch.full((Oc, S), INF, dtype=dt, device=wt.device)
    cost[ar, s0] = 0.0
    av = a[:, None, None] * vtab.view(half, 2)[None]          # (Oc, half, 2): a * v(2m+b)
    bps = []
    for t in range(B):
        tg = wt[:, t, None] - Fo[t]                            # (Oc, S)
        inv = 1.0 / Ub[t, t]
        e0 = (tg[:, :half, None] - av) * inv                   # predecessor x=0 -> state 2m+b
        e1 = (tg[:, half:, None] - av) * inv                   # predecessor x=1
        c0 = cost[:, :half, None] + e0 * e0
        c1 = cost[:, half:, None] + e1 * e1
        ch = c1 < c0
        cost = torch.where(ch, c1, c0).reshape(Oc, S)
        if t < B - 1:
            en = torch.where(ch, e1, e0)
            Fnv = Fn.view(B, Oc, half, 2)[t + 1:]
            torch.where(ch[None], Fo[t + 1:, :, half:].unsqueeze(-1), Fo[t + 1:, :, :half].unsqueeze(-1), out=Fnv)
            Fnv.addcmul_(en[None], Ub[t, t + 1:].view(-1, 1, 1, 1))
            Fo, Fn = Fn, Fo
        bps.append(ch)
    cb, best = cost.min(1)
    s = best
    qv = torch.empty(Oc, B, dtype=dt, device=wt.device)
    for t in range(B - 1, -1, -1):                            # traceback, O(Oc * B)
        qv[:, t] = vtab[s]
        x = bps[t][ar, s >> 1, s & 1].long()
        s = (s >> 1) | (x << (L - 1))
    return cb, seq_errs(wt, Ub, a, qv), qv, best, s


def encode(Wr, U, vtab, L, c, check=False):
    """GPTQ-order trellis encode of Wr (O,n).  Returns (Wq, Sv, A, path_cost (O,), max rel dev DP-cost vs sequential recompute).
    Time O(O n 2^L B / 2) element-ops (n/B blocks x Viterbi block above); space 2 B ROWCAP accumulators."""
    O, n = Wr.shape
    S = 1 << L
    dt = Wr.dtype
    Sv = torch.empty_like(Wr)
    A = torch.empty(O, n // GROUP, dtype=dt, device=Wr.device)
    pc = torch.zeros(O, dtype=dt, device=Wr.device)
    ud = U.diagonal()
    dev = torch.zeros((), dtype=dt, device=Wr.device)
    nchunk = math.ceil(O * S / ROWCAP)
    oc = math.ceil(O / nchunk)
    for r0 in range(0, O, oc):
        r1 = min(O, r0 + oc)
        rc = r1 - r0
        bufs = (torch.empty(BLK, rc, S, dtype=dt, device=Wr.device), torch.empty(BLK, rc, S, dtype=dt, device=Wr.device))
        wk = Wr[r0:r1].clone()
        s = torch.zeros(rc, dtype=torch.long, device=Wr.device)
        for gi, g0 in enumerate(range(0, n, GROUP)):
            g1 = g0 + GROUP
            w_prec = 1.0 / (ud[g0:g1] ** 2).clamp_min(1e-12)
            w_rms = (((wk[:, g0:g1] ** 2) * w_prec[None, :]).sum(1) / w_prec.sum()).clamp_min(1e-24).sqrt()
            a = (c * w_rms).clamp_min(1e-12)
            A[r0:r1, gi] = a
            for b0 in range(g0, g1, BLK):
                b1 = b0 + BLK
                cb, errs, qv, s_end, s_tb = viterbi_block(wk[:, b0:b1], U[b0:b1, b0:b1], a, vtab, s, L, bufs)
                if check:
                    assert bool((s_tb == s).all()), "traceback did not return to the block start state"
                e2 = (errs * errs).sum(1)
                dev = torch.maximum(dev, ((e2 - cb).abs() / cb.clamp_min(1e-30)).max())
                pc[r0:r1] += e2
                Sv[r0:r1, b0:b1] = qv
                s = s_end
                if b1 < n:
                    wk[:, b1:] -= errs @ U[b0:b1, b1:]
    return A.repeat_interleave(GROUP, 1) * Sv, Sv, A, pc, float(dev)


# ------------------------------------------------------------------ shared metric / refit
def Jf(Wr, Wq, Hr64):
    """J in fp64: sum_o (w-q) Hr (w-q)^T.  O(O n^2)."""
    return ft.J(Wr.double(), Wq.double(), Hr64)


def refit(Wr, Hr, Sv):
    """Least-squares refit of the per-(row,128-group) scales given the emitted values Sv: per row the GxG normal equations
    M a = r, M[g,h] = s_g^T Hr[g,h] s_h, r[g] = s_g^T (Hr w).  Time O(O n^2 G) (fp32 matmul, fp64 solve), space O(rc n G)."""
    O, n = Wr.shape
    Gn = n // GROUP
    Hw = Wr @ Hr
    out = torch.empty(O, Gn, dtype=torch.float32, device=Wr.device)
    eye = torch.eye(Gn, device=Wr.device, dtype=Wr.dtype)
    rc = max(1, 20_000_000 // (n * Gn))
    for r0 in range(0, O, rc):
        r1 = min(O, r0 + rc)
        k = r1 - r0
        Sg = Sv[r0:r1].reshape(k, Gn, GROUP)
        Sb = (Sg[..., None] * eye[None, :, None, :]).reshape(k, n, Gn)
        Z = torch.matmul(Hr, Sb).reshape(k, Gn, GROUP, Gn)
        M = (Z * Sg[..., None]).sum(2).double()
        rhs = (Hw[r0:r1].reshape(k, Gn, GROUP) * Sg).sum(2).double()
        M = M + torch.diag_embed(1e-9 * M.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True).expand(k, Gn))
        out[r0:r1] = torch.linalg.solve(M, rhs[..., None])[..., 0].to(torch.float32)
    return out


def refit_J(Wr, Hr, Hr64, Sv):
    a = refit(Wr, Hr, Sv)
    Wq = a.repeat_interleave(GROUP, 1) * Sv
    return Jf(Wr, Wq, Hr64)


def e8_J(Wr, Hr, Hr64, beam):
    """E8 baseline: gptq_e8 (damp 0.02, given beam) then the scale refit.  Returns (J_refit, J_raw)."""
    Wq, S, A = ft.gptq_e8(Wr, Hr, damp=DAMP, beam=beam)
    return refit_J(Wr, Hr, Hr64, S), Jf(Wr, Wq, Hr64)


def trellis_best(Wr, Hr, Hr64, U, L, cs, vtab=None):
    """Sweep c, keep the best refit J.  Returns (J, c, J_raw_at_best, max DP-vs-recompute dev, path_cost_sum_at_best)."""
    vtab = mad1_table(L) if vtab is None else vtab
    best = None
    for c in cs:
        Wq, Sv, A, pc, dev = encode(Wr, U, vtab, L, c)
        j = refit_J(Wr, Hr, Hr64, Sv)
        if best is None or j < best[0]:
            best = (j, c, Jf(Wr, Wq, Hr64), dev, float(pc.sum()))
    return best


# ------------------------------------------------------------------ calibration Grams
def gram_pass(cfg, TL, ids_all, layers, nchunks=20, bs=10):
    """Teacher forward on chunks [0, nchunks); per requested layer the fp64 Gram of stage inputs, / tokens.  Returns
    {layer: {stage: Hr fp32 (rotated)}}.  O(nchunks * CHUNK * layers * params) + the Gram matmuls."""
    top = max(layers)
    G = {l: {} for l in layers}
    N = 0
    for c0 in range(0, nchunks, bs):
        h = cfg.E[ids_all[c0:c0 + bs]].to(DEV)
        N += h.shape[0] * h.shape[1]
        for l in range(top + 1):
            Ld = {k: v.to(DEV) for k, v in TL[l].items()}
            P = ft.parts(cfg, Ld, h)
            if l in layers:
                for st in ("xa", "a", "xf", "m"):
                    X = P[st].reshape(-1, P[st].shape[-1]).to(F64)
                    G[l][st] = G[l].get(st, 0) + X.T @ X
            h = P["out"]
            del Ld, P
    out = {}
    for l in layers:
        out[l] = {}
        for st, g in G[l].items():
            n = g.shape[0]
            Hl = ft.metric(g / N, 0.005 * n / 896).float()
            out[l][st] = ft.had_gram(Hl)
    return out


# ------------------------------------------------------------------ controls
def brute_min(w, U, a, vtab, L, s0=0):
    """Min over all 2^n bit paths from state s0 of the exact sequential-feedback cost.  O(2^n n^2) time, O(2^n n) space."""
    n = w.shape[0]
    P = 1 << n
    idx = torch.arange(P, device=w.device)
    s = torch.full((P,), s0, dtype=torch.long, device=w.device)
    E = torch.zeros(P, n, dtype=w.dtype, device=w.device)
    for t in range(n):
        s = ((s << 1) | ((idx >> t) & 1)) & ((1 << L) - 1)
        fb = E[:, :t] @ U[:t, t] if t else 0.0
        E[:, t] = (w[t] - fb - a * vtab[s]) / U[t, t]
    return float((E * E).sum(1).min())


def toy_instance(seed, diag=False):
    g = torch.Generator().manual_seed(seed)
    rows, n, L = 3, 16, 4
    W = torch.randn(rows, n, generator=g, dtype=F64)
    if diag:
        H = torch.diag(torch.rand(n, generator=g, dtype=F64) + 0.2)
    else:
        A = torch.randn(n, n, generator=g, dtype=F64)
        H = A.T @ A / n + 0.1 * torch.eye(n, dtype=F64)
    U = ft.feedback_u(H, DAMP).to(DEV)
    W = W.to(DEV)
    a = 0.9 * W.pow(2).mean(1).sqrt()
    vtab = mad1_table(L, F64)
    bufs = (torch.zeros(n, rows, 1 << L, dtype=F64, device=DEV), torch.zeros(n, rows, 1 << L, dtype=F64, device=DEV))
    cb, errs, qv, _, s_tb = viterbi_block(W, U, a, vtab, torch.zeros(rows, dtype=torch.long, device=DEV), L, bufs)
    bf = [brute_min(W[r], U, a[r], vtab, L) for r in range(rows)]
    return cb.cpu().tolist(), bf, (errs * errs).sum(1).cpu().tolist(), bool((s_tb == 0).all())


def control_toy():
    """Exhaustive toy control: rows=3, n=16, L=4.  Reports the brief's equality (seed 1234) plus the DP-bookkeeping and
    exactness-regime checks, and the exact-DP gap statistics over 20 seeds."""
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-30)
    cb, bf, pc, tb = toy_instance(1234)
    gen_eq = max(rel(x, y) for x, y in zip(cb, bf)) <= 1e-5
    path_ok = max(rel(x, y) for x, y in zip(cb, pc)) <= 1e-5 and tb
    ge_ok = all(x >= y * (1 - 1e-9) for x, y in zip(cb, bf))
    cbd, bfd, pcd, tbd = toy_instance(77, diag=True)
    diag_eq = max(rel(x, y) for x, y in zip(cbd, bfd)) <= 1e-5
    nrow = neq = 0
    gapmax = 0.0
    for sd in range(100, 120):
        c2, b2, _, _ = toy_instance(sd)
        for x, y in zip(c2, b2):
            nrow += 1
            neq += rel(x, y) <= 1e-5
            gapmax = max(gapmax, x / y - 1.0)
    return dict(general_equal_seed1234=gen_eq, viterbi_cost=cb, brute_min=bf, path_cost_matches_dp=path_ok, viterbi_ge_brute=ge_ok,
                diag_h_equal=diag_eq, general_rows_equal=f"{neq}/{nrow}", general_max_gap=gapmax)


def control_degenerate(Wr, Hr, Hr64, U, J_e8):
    vt = sign_table(8)
    j = trellis_best(Wr, Hr, Hr64, U, 8, (0.8, 0.95, 1.1), vtab=vt)[0]
    j1 = trellis_best(Wr, Hr, Hr64, U, 1, (0.8, 0.95, 1.1), vtab=sign_table(1))[0]       # 2 states: ~greedy GPTQ sign
    sg, sn, _ = ft.gptq_signs(Wr, Hr, damp=DAMP)
    jg = refit_J(Wr, Hr, Hr64, sn)
    return dict(J_sign_L8=j, J_sign_L1=j1, J_gptq_sign=jg, J_e8=J_e8, ratio=j / J_e8, ratio_L1=j1 / J_e8, ratio_gptq_sign=jg / J_e8,
                ok=j / J_e8 > 1.0)


def control_identity(Wr, Hr):
    """fp64 L=8: sum of chosen-path err^2 == (w-q)(Hr + damp mean(diag Hr) I)(w-q)^T (U^T U = (Hr + damp..)^-1)."""
    W6, H6 = Wr.double(), Hr.double()
    U6 = ft.feedback_u(H6, DAMP)
    Wq, Sv, A, pc, dev = encode(W6, U6, mad1_table(8, F64), 8, 0.95, check=True)
    D = W6 - Wq
    md = H6.diagonal().mean()
    Jd = float(((D @ H6) * D).sum() + DAMP * md * (D * D).sum())
    pcs = float(pc.sum())
    return dict(path_cost=pcs, J_damped=Jd, J_undamped=float(((D @ H6) * D).sum()), rel=abs(pcs - Jd) / Jd, dp_dev=dev, ok=abs(pcs - Jd) / Jd <= 1e-3)


# ------------------------------------------------------------------ main
def stats(xs):
    return dict(median=statistics.median(xs), geomean=math.exp(sum(math.log(x) for x in xs) / len(xs)), n=len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--layers", default="0,23,11")
    ap.add_argument("--cs", default="0.8,0.95,1.1")
    ap.add_argument("--budget", type=float, default=520.0)
    a = ap.parse_args()
    torch.manual_seed(0)
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    cs = tuple(float(x) for x in a.cs.split(","))
    layers = [0] if a.quick else [int(x) for x in a.layers.split(",")]
    kinds = ["q"] if a.quick else KINDS
    if a.quick:
        cs = (0.95,)
    rec = dict(script_sha256=sha, model=ft.MODEL, chunks=[0, 20], cs=list(cs), quick=a.quick, linears=[], controls={}, notes=[])

    def flush(final=False):
        rec["total_wall"] = time.time() - T0
        for key in ("L8", "L12"):
            for base in ("beam1", "beam4"):
                xs = [x[f"ratio_{key}_vs_{base}"] for x in rec["linears"] if f"ratio_{key}_vs_{base}" in x]
                rec.setdefault("summary", {})[f"{key}_vs_{base}"] = stats(xs)
        rec["bits_per_weight"] = dict(e8="1 + 16/128 (f16 scale per 128-group) = 1.125", trellis="1 + L/n + 16/128: n=896 -> L8 1.1339, L12 1.1384; n=4864 -> L8 1.1266, L12 1.1275")
        if a.out:
            json.dump(rec, open(a.out + ".tmp", "w"), indent=1)
            os.replace(a.out + ".tmp", a.out)

    t = time.time()
    rec["controls"]["toy"] = ct = control_toy()
    print("toy control:", json.dumps({k: v for k, v in ct.items() if k not in ("viterbi_cost", "brute_min")}), f"({time.time()-t:.1f}s)", flush=True)
    hard = ct["path_cost_matches_dp"] and ct["viterbi_ge_brute"] and ct["diag_h_equal"]
    if not hard:
        rec["notes"].append("toy hard control failed; no numbers")
        flush()
        print("CONTROL FAILED", ct, flush=True)
        sys.exit(2)
    if not ct["general_equal_seed1234"]:
        rec["notes"].append("brief's general-SPD equality (Viterbi == brute min) does not hold: per-survivor-feedback Viterbi merges paths "
                            "with different feedback histories, so it is a heuristic upper bound on the brute min unless H is diagonal")

    tok = transformers.AutoTokenizer.from_pretrained(ft.MODEL)
    ids = tok.encode(open(ft.CALIB, encoding="utf-8").read())
    nch = len(ids) // CHUNK
    ids_all = torch.tensor(ids[:nch * CHUNK], device="cpu").view(nch, CHUNK)
    cfg, TL, hf = ft.load_model()
    del hf
    t = time.time()
    with torch.no_grad():
        Hs = gram_pass(cfg, TL, ids_all, set(layers))
    print(f"grams {time.time()-t:.1f}s, elapsed {time.time()-T0:.1f}s", flush=True)

    with torch.no_grad():
        # controls on layer 0 q
        l0 = layers[0]
        Hr0 = Hs[l0]["xa"]
        Wr0 = ft.fwht128(TL[l0]["q"].to(DEV))
        Hr064 = Hr0.double()
        U0 = ft.feedback_u(Hr0, DAMP)
        je8_0 = e8_J(Wr0, Hr0, Hr064, 1)[0]
        rec["controls"]["degenerate"] = cd = control_degenerate(Wr0, Hr0, Hr064, U0, je8_0)
        rec["controls"]["identity"] = ci = control_identity(Wr0, Hr0)
        print("degenerate:", cd, "\nidentity:", ci, flush=True)
        if not cd["ok"]:
            rec["notes"].append("degenerate control as specified (sign-quantiser J ratio > 1 vs E8) FAILED: L=8 sign trellis beats E8 beam 1; numbers kept, flagged")
        if not ci["ok"]:
            rec["notes"].append("identity control failed; no numbers")
            flush()
            print("CONTROL FAILED", flush=True)
            sys.exit(2)

        Ucache = {}
        layer_times = []
        done_numel = 0
        for l in layers:
            elapsed = time.time() - T0
            if layer_times and elapsed + statistics.mean(layer_times) > a.budget:
                rec["notes"].append(f"layer {l} dropped: elapsed {elapsed:.0f}s + mean layer {statistics.mean(layer_times):.0f}s > {a.budget:.0f}s")
                print(rec["notes"][-1], flush=True)
                continue
            tl = time.time()
            for kind in kinds:
                tm = time.time()
                st = STAGE_OF[kind]
                Hr = Hs[l][st]
                Hr64 = Hr.double()
                Wr = ft.fwht128(TL[l][kind].to(DEV))
                if (l, st) not in Ucache:
                    Ucache.clear()
                    Ucache[(l, st)] = ft.feedback_u(Hr, DAMP)
                U = Ucache[(l, st)]
                O, n = Wr.shape
                r = dict(layer=l, kind=kind, O=O, n=n)
                r["J_e8"], r["J_e8_raw"] = e8_J(Wr, Hr, Hr64, 1)
                r["J_e8_beam4"], r["J_e8_beam4_raw"] = e8_J(Wr, Hr, Hr64, 4)
                for L in (8, 12):
                    j, c, jraw, dev, pcs = trellis_best(Wr, Hr, Hr64, U, L, cs)
                    r[f"J_trellis_L{L}"], r[f"best_c_L{L}"], r[f"J_trellis_L{L}_raw"], r[f"dp_dev_L{L}"] = j, c, jraw, dev
                    r[f"ratio_L{L}_vs_beam1"] = j / r["J_e8"]
                    r[f"ratio_L{L}_vs_beam4"] = j / r["J_e8_beam4"]
                    r[f"ratio_L{L}"] = r[f"ratio_L{L}_vs_beam1"]
                    r[f"path_cost_L{L}"] = pcs   # sum err^2 = J_raw + damp-term (identity control checks the equality)
                r["wall"] = time.time() - tm
                rec["linears"].append(r)
                done_numel += O * n
                print(f"L{l} {kind:5s} {O}x{n} J_e8 {r['J_e8']:.4g} b4 {r['J_e8_beam4']:.4g} | L8 {r['ratio_L8_vs_beam1']:.3f}/{r['ratio_L8_vs_beam4']:.3f} "
                      f"(c {r['best_c_L8']}) L12 {r['ratio_L12_vs_beam1']:.3f}/{r['ratio_L12_vs_beam4']:.3f} (c {r['best_c_L12']}) "
                      f"dev {r['dp_dev_L12']:.1e} {r['wall']:.1f}s | elapsed {time.time()-T0:.0f}s", flush=True)
                if len(rec["linears"]) == 1 and not a.quick:
                    proj = (time.time() - T0) + r["wall"] * (len(layers) * 14.9e6 / (O * n) - 1) * 0.2
                    print(f"projection after first matrix (informational, numel-scaled x0.2 for launch overhead): {proj:.0f}s", flush=True)
            layer_times.append(time.time() - tl)
            print(f"layer {l} done in {layer_times[-1]:.1f}s, elapsed {time.time()-T0:.0f}s", flush=True)
            flush()
    flush(final=True)
    print(json.dumps(rec["summary"]), f"total {rec['total_wall']:.1f}s sha {sha}", flush=True)


if __name__ == "__main__":
    main()
