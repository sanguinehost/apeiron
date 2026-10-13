"""ONEBIT-W17-C generalisation probe for the -nb arm (spec: 2026-10-14-w17-foundation.md section 2).

Weight space: W[o, i], Hessians over the input dimension i.  R = Ws - Wp (target minus decoded weight, all planes and corrector),
J_A(R) = tr(R A R^T) = sum_o R_o A R_o^T (the frame of ft.J; the Hadamard rotation is orthogonal and does not change it).
Hessians are per-token means.  Every metric is ft.metric(mean_XtX, quant_ridge * n / 896), the form the quantiser used for Hl_m.
  H_cal   captured Hl_m (student inputs on calibration chunks [0, n_cal)), divided by n_cal * CHUNK
  H_tc    teacher inputs on [0, n_cal)           (finite-sample pair of H_out; O_fs = J_out / J_tc)
  H_in    teacher inputs on [0, n_in)            (Q denominator)
  H_out   teacher inputs on held-out [100, 140)  (fixed for every n_cal)
O = J_out / J_cal = O_fs * O_shift, O_shift = J_tc / J_cal (student-vs-teacher input shift).
Sandwich: mu_min J_out <= J_cal <= mu_max J_out with mu the extreme generalised eigenvalues of H_cal v = mu H_out v (a theorem; a violation
is an instrument bug and stops the run).

subcommands
  controls                                  CPU; synthetic Marchenko-Pastur control of the eigensolver, O formula and sandwich
  run      --n-cal N --cap DIR --dump PT --out JSON   GPU; the -nb arm at n_cal=N, seed 0, held-out fixed to [100, 140); captures Ws, Hl_m
  measure  --cap DIR --dump PT --n-cal N --out JSON [--run-receipt JSON]   GPU; teacher Hessians, O/mu/Q per tensor, controls, sandwich
  fit      --runs J1,J2,J3 --out JSON       E_gen ~ N^-alpha from three run receipts
  tilt     --dump PT --run-receipt JSON --measure-receipt JSON --out JSON   GPU; Q = Q_tilt * Q_lin, per-position KL / q2 (q2 = Var_p(delta)/2)
Run with ~/spinor-ttl/.venv/bin/python.
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fold_transfer as ft
import side_probe as sp

F64 = torch.float64
KINDS = sp.KINDS
STAGE_OF = sp.STAGE_OF
STAGE_T = {"qkv": "xa", "o": "a", "ffn": "xf", "down": "m"}   # parts() key of the stage input
CHUNK = ft.CHUNK
VARIANT = "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp-nb"
RECEIPT_ARGS = "~/w16/t3-nb-3seed.json"      # the script's own argument set; n_cal is overridden
EVAL = (100, 140)
N_IN = 20
SANDWICH_TOL = 1e-7
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, file=sys.stderr, flush=True)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ linear algebra
def metric64(S, ridge):
    """ft.metric in fp64: S + ridge * mean(diag S) I."""
    H = S.clone()
    H.diagonal().add_(ridge * S.diagonal().mean())
    return H


def row_j(R, H):
    """Per-row J_o = R_o H R_o^T.  Cost: temporal 2 O n^2 flops (one [O, n] x [n, n] product); spatial O n words."""
    return ((R @ H) * R).sum(1)


def gen_eigs(Hc, Ho):
    """Extreme generalised eigenvalues of Hc v = mu Ho v (Ho SPD): eigvalsh(L^-1 Hc L^-T), Ho = L L^T.
    Cost: temporal ~ (1 + 2 + 4/3) n^3 flops (Cholesky, two triangular solves, eigenvalues of a symmetric matrix); spatial 3 n^2 words.
    Returns (mu_min, mu_max, min eig of Ho / mean diag of Ho)."""
    L = torch.linalg.cholesky(Ho)
    C = torch.linalg.solve_triangular(L, Hc, upper=False)
    C = torch.linalg.solve_triangular(L, C.T, upper=False)
    C = 0.5 * (C + C.T)
    w = torch.linalg.eigvalsh(C)
    return float(w[0]), float(w[-1]), float(torch.diagonal(L).min() ** 2 / Ho.diagonal().mean())


def check_sandwich(tag, Jc, Jo, mu_min, mu_max):
    lo, hi = mu_min * Jo, mu_max * Jo
    ok = (Jc >= lo * (1 - SANDWICH_TOL)) and (Jc <= hi * (1 + SANDWICH_TOL))
    return bool(ok), dict(tag=tag, J_cal=Jc, J_out=Jo, mu_min=mu_min, mu_max=mu_max, lower=lo, upper=hi)


# ------------------------------------------------------------------------------------------------ synthetic controls
def synthetic_controls():
    """(a) Marchenko-Pastur: X_cal iid N(0, I) with d = 512, n = 5400 (c = d/n); H_out from 2e5 samples (about I).  R = bottom eigenvector of H_cal
    -> J_cal = mu_min, J_out ~ 1, so O = 1/mu_min and mu_min ~ (1 - sqrt c)^2.  (b) H_out = H_cal gives O = 1, mu = 1 exactly.
    (c) A column-permuted R sits near O = tr(H_out)/tr(H_cal) on a real-shaped SPD pair."""
    g = torch.Generator().manual_seed(7)
    d, n, nout = 512, 5400, 200000
    c = d / n
    Xc = torch.randn(n, d, generator=g, dtype=F64)
    Xo = torch.randn(nout, d, generator=g, dtype=F64)
    Hc, Ho = Xc.T @ Xc / n, Xo.T @ Xo / nout
    w, V = torch.linalg.eigh(Hc)
    R = V[:, :1].T.contiguous()
    Jc, Jo = float(row_j(R, Hc).sum()), float(row_j(R, Ho).sum())
    mu_min, mu_max, _ = gen_eigs(Hc, Ho)
    mp_lo, mp_hi = (1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2
    a = dict(c=c, mu_min=mu_min, mu_max=mu_max, mp_lo=mp_lo, mp_hi=mp_hi, O=Jo / Jc, inv_mu_min=1 / mu_min,
             pass_=abs(mu_min / mp_lo - 1) < 0.06 and abs(mu_max / mp_hi - 1) < 0.06 and abs(Jo / Jc * mu_min - 1) < 0.06
             and check_sandwich("mp", Jc, Jo, mu_min, mu_max)[0])
    R5 = torch.randn(40, d, generator=g, dtype=F64)
    j1 = float(row_j(R5, Hc).sum())
    mu1, mu2, _ = gen_eigs(Hc, Hc)
    b = dict(O=j1 / j1, mu_min=mu1, mu_max=mu2, pass_=abs(mu1 - 1) < 1e-9 and abs(mu2 - 1) < 1e-9)
    perm = torch.randperm(d, generator=g)
    Rs = R5[:, perm]
    tr = float(Ho.diagonal().sum() / Hc.diagonal().sum())
    cc = dict(O_shuffled=float(row_j(Rs, Ho).sum() / row_j(Rs, Hc).sum()), trace_ratio=tr)
    cc["pass_"] = abs(cc["O_shuffled"] / tr - 1) < 0.1
    # negative control: a mutated sandwich (swap the roles) must be rejected
    bad, _ = check_sandwich("mut", Jc, Jo, mu_min * 2.0, mu_max)
    return dict(mp=a, identity=b, shuffle=cc, mutated_sandwich_rejected=not bad, pass_=bool(a["pass_"] and b["pass_"] and cc["pass_"] and not bad))


# ------------------------------------------------------------------------------------------------ data / arguments
def load_ids():
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ids = tok.encode(open(ft.CALIB, encoding="utf-8").read())
    nch = len(ids) // CHUNK
    assert nch >= EVAL[1], nch
    return torch.tensor(ids[:nch * CHUNK], device="cpu").view(nch, CHUNK), nch, len(ids)


def run_args(n_cal):
    d = json.load(open(RECEIPT_ARGS))["args"]
    d = dict(d, n_cal=n_cal, out=None, dump=None)
    return argparse.Namespace(**d)


# ------------------------------------------------------------------------------------------------ run
def cmd_run(a):
    """The -nb arm at n_cal = a.n_cal (seed 0), argument set copied from the 3-seed receipt, held-out chunks fixed to [100, 140), in-sample [0, 20).
    Wraps ft.fold_fit as side_probe.cmd_capture does to store Ws_nm and Hl_m per (layer, tensor).
    Cost: one -nb run (about 130 s at n_cal = 100, proportionally less for fewer chunks) plus 4 GB of writes."""
    N = a.n_cal
    assert N >= N_IN and N <= EVAL[0]
    ids_all, nch, ntok = load_ids()
    cfg, TL, hf = ft.load_model()
    del hf
    torch.cuda.empty_cache()
    args = run_args(N)
    ids_c, ids_ev, ids_in = ids_all[:N].contiguous(), ids_all[EVAL[0]:EVAL[1]], ids_all[:N_IN]
    os.makedirs(a.cap, exist_ok=True)
    seen, held = set(), {}
    orig = ft.fold_fit

    def wrapped(Ws, H, *x, **kw):
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
            del fr, loc
        return orig(Ws, H, *x, **kw)

    ft.fold_fit = wrapped
    t0 = ft.tick()
    torch.manual_seed(0)
    Ls, binfo, tm = ft.run_variant(cfg, TL, ids_c, VARIANT, args)
    ft.fold_fit = orig
    tq = ft.tick() - t0
    kl, ag = ft.kl_eval(cfg, TL, Ls, ids_ev)
    kli, agi = ft.kl_eval(cfg, TL, Ls, ids_in)
    torch.save({"layers": [{k: v.cpu() for k, v in L.items()} for L in Ls], "fnorm": cfg.fnorm.cpu()}, a.dump)
    nl = len(Ls)
    json.dump(dict(layers=nl, kinds=KINDS, stage_of=STAGE_OF, variant=VARIANT, n_cal=N, ft_sha256=sha(ft.__file__)),
              open(os.path.join(a.cap, "meta.json"), "w"), indent=1)
    rec = json.load(open(RECEIPT_ARGS))["variants"][VARIANT]["seeds"]["0"]
    out = dict(script_sha256=sha(__file__), fold_transfer_sha256=sha(ft.__file__), variant=VARIANT, seed=0, n_cal=N,
               cal_chunks=[0, N], heldout_chunks=list(EVAL), insample_chunks=[0, N_IN], n_chunks=nch,
               kl_heldout=kl, kl_insample=kli, top1_heldout=ag, top1_insample=agi, E_gen=kl - kli, G=kl / kli,
               bits=binfo["bits"], params=binfo["params"], captured=len(seen), expected_captured=nl * len(KINDS),
               receipt_seed0=dict(path=RECEIPT_ARGS, kl_heldout=rec["kl_heldout"], kl_insample=rec["kl_insample"]),
               wall_transfer_s=tq, cap=a.cap, dump=a.dump)
    if N == 100:
        out["repro_dkl_heldout"] = kl - rec["kl_heldout"]
        out["repro_dkl_insample"] = kli - rec["kl_insample"]
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    if len(seen) != nl * len(KINDS):
        raise SystemExit("capture incomplete")


# ------------------------------------------------------------------------------------------------ measure
def teacher_layers(cfg, TL, ids_all, N):
    """Groups of chunk indices: A = [0, n_in), B = [n_in, N), O = [100, 140).  Returns the chunk index list and the group slices."""
    idx = torch.cat([torch.arange(0, N), torch.arange(EVAL[0], EVAL[1])])
    grp = dict(A=(0, N_IN), B=(N_IN, N), O=(N, N + EVAL[1] - EVAL[0]))
    return idx, grp


def g_row_of(cfg, TL, l, nm):
    """The row weight fold_fit used (run_variant, use_grow): gate/up -> column mass of Down, o -> ln2^2, down -> next ln1^2 (fnorm at the last layer)."""
    if nm in ("gate", "up"):
        g = (TL[l]["down"].double() ** 2).sum(0)
    elif nm in ("o", "down"):
        ln = TL[l]["ln2"] if nm == "o" else (TL[l + 1]["ln1"] if l + 1 < len(TL) else cfg.fnorm)
        g = ln.double() ** 2
    else:
        return None
    return (g / g.mean().clamp_min(1e-12)).cuda()


def cmd_measure(a):
    N = a.n_cal
    t_start = time.time()
    ctl = synthetic_controls()
    log("synthetic controls", "PASS" if ctl["pass_"] else "FAIL", json.dumps({k: v for k, v in ctl.items() if k != "pass_"}, default=float))
    out = dict(script_sha256=sha(__file__), fold_transfer_sha256=sha(ft.__file__), side_probe_sha256=sha(sp.__file__), n_cal=N,
               cap=a.cap, dump=a.dump, heldout_chunks=list(EVAL), insample_chunks=[0, N_IN], synthetic_controls=ctl)
    if not ctl["pass_"]:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        raise SystemExit("synthetic controls failed")
    args = run_args(N)
    ids_all, nch, ntok = load_ids()
    cfg, TL, hf = ft.load_model()
    del hf
    idx, grp = teacher_layers(cfg, TL, ids_all, N)
    h = cfg.E[ids_all[idx]].to(ft.DEV)           # [N + 40, CHUNK, hid], teacher residual stream
    cnt = {k: (e - s) * CHUNK for k, (s, e) in grp.items()}
    src = sp.real_source(a.cap, a.dump, sp.QWEN)
    rows, stage_rows, violations, ctrl_rows = [], [], [], []
    nl = a.layers
    for l in range(nl):
        Lt = {k: v.to(ft.DEV) for k, v in TL[l].items()}
        S = {g: {st: torch.zeros(n, n, dtype=F64, device=ft.DEV) for st, n in sp.QWEN.stage_n.items()} for g in grp}
        for g, (s0, e0) in grp.items():
            for b0 in range(s0, e0, 10):
                P = ft.parts(cfg, Lt, h[b0:min(b0 + 10, e0)])
                for st, key in STAGE_T.items():
                    X = P[key].reshape(-1, P[key].shape[-1])
                    S[g][st] += (X.T @ X).to(F64)
                del P, X
        for g in grp:
            for st in S[g]:
                S[g][st] /= cnt[g]
        for b0 in range(0, h.shape[0], 10):
            h[b0:b0 + 10] = ft.parts(cfg, Lt, h[b0:b0 + 10])["out"]
        l_, Ws, Wp, Hl = next(src)
        assert l_ == l
        stg_eig = {}
        Hs = {}
        for st, n in sp.QWEN.stage_n.items():
            rg = args.quant_ridge * (n / 896.0)
            Hc = Hl[st].to(ft.DEV, F64) / (N * CHUNK)
            S_tc = (S["A"][st] * cnt["A"] + S["B"][st] * cnt["B"]) / (cnt["A"] + cnt["B"]) if cnt["B"] else S["A"][st]
            Ho = metric64(S["O"][st], rg)
            Htc = metric64(S_tc, rg)
            Hin = metric64(S["A"][st], rg)
            Hs[st] = (Hc, Htc, Hin, Ho)
            m_p = gen_eigs(Hc, Ho)
            m_fs = gen_eigs(Htc, Ho)
            stg_eig[st] = dict(primary=m_p, fs=m_fs)
            stage_rows.append(dict(layer=l, stage=st, n=n, mu_min=m_p[0], mu_max=m_p[1], ho_floor=m_p[2], fs_mu_min=m_fs[0], fs_mu_max=m_fs[1],
                                   c_cal=n / (N * CHUNK), c_out=n / cnt["O"], mp_lo=(1 - math.sqrt(n / (N * CHUNK))) ** 2))
        for nm in KINDS:
            st = STAGE_OF[nm]
            Hc, Htc, Hin, Ho = Hs[st]
            R = (Ws[nm].to(ft.DEV, F64) - Wp[nm].to(ft.DEV, F64))
            g = g_row_of(cfg, TL, l, nm)
            rj = {k: row_j(R, H) for k, H in dict(cal=Hc, tc=Htc, inn=Hin, out=Ho).items()}
            J = {k: float(v.sum()) for k, v in rj.items()}
            Jg = {k: (float(v @ g) if g is not None else float(v.sum())) for k, v in rj.items()}
            for pair, (Jn, Jd, ev) in dict(primary=(J["out"], J["cal"], stg_eig[st]["primary"]),
                                            fs=(J["out"], J["tc"], stg_eig[st]["fs"])).items():
                ok, info = check_sandwich(f"l{l}.{nm}.{pair}", Jd, Jn, ev[0], ev[1])
                if not ok:
                    violations.append(dict(info, frame="original weight frame, per-token mean Hessians, metric(..., quant_ridge*n/896)"))
            row = dict(layer=l, kind=nm, shape=list(R.shape), J_cal=J["cal"], J_tc=J["tc"], J_in=J["inn"], J_out=J["out"],
                       O=J["out"] / J["cal"], O_fs=J["out"] / J["tc"], O_shift=J["tc"] / J["cal"],
                       Jg_cal=Jg["cal"], Jg_out=Jg["out"], Og=Jg["out"] / Jg["cal"],
                       mu_min=stg_eig[st]["primary"][0], mu_max=stg_eig[st]["primary"][1],
                       fs_mu_min=stg_eig[st]["fs"][0], fs_mu_max=stg_eig[st]["fs"][1])
            rows.append(row)
            if l in (0, 12) and nm in ("q", "down", "gate"):
                gen = torch.Generator(device="cpu").manual_seed(1000 + l)
                Rs = R[:, torch.randperm(R.shape[1], generator=gen).to(ft.DEV)]
                O_shuf = float(row_j(Rs, Ho).sum() / row_j(Rs, Hc).sum())
                tr_ratio = float(Ho.diagonal().sum() / Hc.diagonal().sum())
                O_plant = float(row_j(R, Hc).sum() / row_j(R, Hc).sum())
                mu_p = gen_eigs(Hc, Hc)
                ctrl_rows.append(dict(layer=l, kind=nm, O_real=row["O"], O_shuffled=O_shuf, trace_ratio=tr_ratio,
                                      O_shuf_over_trace=O_shuf / tr_ratio, O_planted_same_H=O_plant, mu_planted=[mu_p[0], mu_p[1]]))
        if violations:
            out.update(violations=violations, stopped_at_layer=l, rows=rows, stage_rows=stage_rows)
            json.dump(out, open(a.out, "w"), indent=1, default=float)
            log("SANDWICH VIOLATION", json.dumps(violations[0], default=float))
            raise SystemExit("sandwich violation (instrument bug); stopped")
        log(f"layer {l}: " + " ".join(f"{r['kind']}:O={r['O']:.3f}" for r in rows if r["layer"] == l))
        del S, Hs, Hl, Ws, Wp
        torch.cuda.empty_cache()
    # ---- aggregates
    def agg(sel, num="J_out", den="J_cal"):
        return sum(r[num] for r in sel) / sum(r[den] for r in sel)

    per_kind = {}
    Jtot = sum(r["J_cal"] for r in rows)
    for kd in KINDS:
        sel = [r for r in rows if r["kind"] == kd]
        per_kind[kd] = dict(O=agg(sel), O_fs=agg(sel, "J_out", "J_tc"), O_shift=agg(sel, "J_tc", "J_cal"), Og=agg(sel, "Jg_out", "Jg_cal"),
                            mu_min_med=float(np.median([r["mu_min"] for r in sel])), mu_min_min=min(r["mu_min"] for r in sel),
                            mu_max_med=float(np.median([r["mu_max"] for r in sel])), mu_max_max=max(r["mu_max"] for r in sel),
                            J_share=sum(r["J_cal"] for r in sel) / Jtot,
                            O_layer_min=min(r["O"] for r in sel), O_layer_max=max(r["O"] for r in sel),
                            tightness_Jw=sum(r["J_out"] * r["mu_min"] for r in sel) / sum(r["J_cal"] for r in sel),
                            tightness_min=min(r["O"] * r["mu_min"] for r in sel))
    wide = [r for r in rows if r["shape"][1] == 896]
    out["per_kind"] = per_kind
    out["aggregate"] = dict(O=agg(rows), O_fs=agg(rows, "J_out", "J_tc"), O_shift=agg(rows, "J_tc", "J_cal"), Og=agg(rows, "Jg_out", "Jg_cal"),
                            O_896_wide=agg(wide), O_down=per_kind["down"]["O"],
                            J_cal_sum=Jtot, J_out_sum=sum(r["J_out"] for r in rows), J_in_sum=sum(r["J_in"] for r in rows),
                            J_tc_sum=sum(r["J_tc"] for r in rows))
    out["rows"], out["stage_rows"], out["controls_real"] = rows, stage_rows, ctrl_rows
    out["sandwich"] = dict(checked_pairs=2 * len(rows), violations=0, tol_rel=SANDWICH_TOL)
    if a.run_receipt:
        rr = json.load(open(a.run_receipt))
        G = rr["kl_heldout"] / rr["kl_insample"]
        Q = (rr["kl_heldout"] / out["aggregate"]["J_out_sum"]) / (rr["kl_insample"] / out["aggregate"]["J_in_sum"])
        O = out["aggregate"]["O"]
        out["QT"] = dict(run_receipt=a.run_receipt, kl_heldout=rr["kl_heldout"], kl_insample=rr["kl_insample"], G=G, Q=Q, O=O, T_derived=G / (O * Q),
                         identity_T_eq_Jcal_over_Jin=out["aggregate"]["J_cal_sum"] / out["aggregate"]["J_in_sum"],
                         G_3seed_receipt=1.29564 / 0.35064, grade="Q measured; T derived = G/(O Q), bookkeeping")
    out["wall_s"] = time.time() - t_start
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(aggregate=out["aggregate"], per_kind={k: {x: round(y, 4) for x, y in v.items()} for k, v in per_kind.items()},
                          QT=out.get("QT"), controls_real=ctrl_rows), indent=1, default=float))


# ------------------------------------------------------------------------------------------------ fit
def cmd_fit(a):
    rs = [json.load(open(p)) for p in a.runs.split(",")]
    rs.sort(key=lambda r: r["n_cal"])
    N = np.array([r["n_cal"] for r in rs], float)
    E = np.array([r["E_gen"] for r in rs], float)
    X = np.stack([np.ones_like(N), np.log(N)], 1)
    beta, *_ = np.linalg.lstsq(X, np.log(E), rcond=None)
    alpha = -float(beta[1])
    pred = np.exp(X @ beta)
    # two-point local exponents
    loc = [-math.log(E[i + 1] / E[i]) / math.log(N[i + 1] / N[i]) for i in range(len(N) - 1)]
    out = dict(script_sha256=sha(__file__), n_cal=N.tolist(), kl_heldout=[r["kl_heldout"] for r in rs], kl_insample=[r["kl_insample"] for r in rs],
               E_gen=E.tolist(), G=[r["G"] for r in rs], alpha_loglog_fit=alpha, prefactor=float(math.exp(beta[0])), local_exponents=loc,
               fit_pred=pred.tolist(), fit_resid_rel=((pred - E) / E).tolist(), runs=a.runs.split(","),
               E_gen_25_vs_predicted_3p8=float(E[0]) if N[0] == 25 else None, grade="measured, 1 seed, 3 points")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ tilt
GRID = torch.linspace(0.0, 1.0, 41, dtype=F64)


def pos_stats(lt, ls, grid=False):
    """Per-position statistics of delta = ls - lt (student minus teacher log-prob), p = exp(lt).  lt, ls: [P, V] fp32 log-probs on GPU.
    Returns dict of fp64 [P] tensors: kl32 = sum p (lt - ls) (fp32 path, as ft.kl_eval), kl (fp64), q2 = Var_p(delta)/2, rng = max - min of delta
    and, if grid, the 41-point Var_{p_t}(delta), p_t ~ p exp(t delta).
    Cost: temporal O(P V) flops per statistic (grid: 41 P V exp evaluations); spatial O(P V) words in fp64."""
    kl32 = (lt.exp() * (lt - ls)).sum(-1).double()
    lt, ls = lt.double(), ls.double()
    d = ls - lt
    p = lt.exp()
    kl = (p * (lt - ls)).sum(-1)
    m = (p * d).sum(-1, keepdim=True)
    q2 = 0.5 * (p * (d - m) ** 2).sum(-1)
    out = dict(kl32=kl32, kl=kl, q2=q2, rng=d.max(-1).values - d.min(-1).values)
    if grid:
        dc = d - m
        v = []
        for t in GRID.tolist():
            w = torch.softmax(lt + t * dc, -1)
            mt = (w * d).sum(-1, keepdim=True)
            v.append((w * (d - mt) ** 2).sum(-1))
        out["var_t"] = torch.stack(v, 1)
    return out


def tilt_set(cfg, TL, Ls, ids, sub_stride, sub_max, ctl=None):
    """Per-position statistics on ids [n, CHUNK]: same hidden-state / logit path as ft.kl_eval (bs = 2 chunks).  Grid statistics on every
    sub_stride-th position (at most sub_max of them).  ctl: None | 'zero' | eps -> replace the student log-prob by log_softmax(lt + eps delta)
    (planted controls).  Cost: 2 forward passes over n chunks plus 41 softmax passes over the grid positions."""
    N = ids.shape[0]
    ht, hs = cfg.E[ids.cpu()], cfg.E[ids.cpu()]
    for l in range(len(TL)):
        Lt = {k: v.to(ft.DEV) for k, v in TL[l].items() if k != "fnorm"}
        for s0 in range(0, N, 10):
            ht[s0:s0 + 10] = ft.parts(cfg, Lt, ht[s0:s0 + 10].to(ft.DEV))["out"].cpu()
        del Lt
    for l in range(len(Ls)):
        Lq = {k: v.to(ft.DEV) for k, v in Ls[l].items() if k != "fnorm"}
        for s0 in range(0, N, 10):
            hs[s0:s0 + 10] = ft.parts(cfg, Lq, hs[s0:s0 + 10].to(ft.DEV))["out"].cpu()
        del Lq
    fn_t = TL[-1].get("fnorm", cfg.fnorm).to(ft.DEV)
    fn_s = Ls[-1].get("fnorm", cfg.fnorm).to(ft.DEV)
    E_g = cfg.E.to(ft.DEV)
    res, pos = [], 0
    for i in range(0, N, 2):
        ut = ft.rms(ht[i:i + 2].to(ft.DEV), fn_t, cfg.eps)
        us = ft.rms(hs[i:i + 2].to(ft.DEV), fn_s, cfg.eps)
        lt = torch.log_softmax((ut @ E_g.T).float(), -1).reshape(-1, E_g.shape[0])
        ls = torch.log_softmax((us @ E_g.T).float(), -1).reshape(-1, E_g.shape[0])
        if ctl is not None:
            ls = lt.clone() if ctl == "zero" else torch.log_softmax(lt + float(ctl) * (ls - lt), -1)
        P = lt.shape[0]
        for b in range(0, P, 256):
            sl = slice(b, min(b + 256, P))
            gpos = pos + torch.arange(sl.start, sl.stop)
            sel = ((gpos % sub_stride == 0) & (gpos // sub_stride < sub_max)).to(ft.DEV)
            part = pos_stats(lt[sl], ls[sl], grid=False)
            part["var_t"] = torch.full((sl.stop - sl.start, len(GRID)), float("nan"), dtype=F64, device=ft.DEV)
            if bool(sel.any()):
                part["var_t"][sel] = pos_stats(lt[sl][sel], ls[sl][sel], grid=True)["var_t"]
            res.append({k: v.cpu() for k, v in part.items()})
        pos += P
        del ut, us, lt, ls
    del E_g, hs, ht
    torch.cuda.empty_cache()
    return {k: torch.cat([x[k] for x in res]) for k in res[0]}


def summarise(st):
    """Set-level numbers from per-position tensors (fp64 on CPU)."""
    kl, q2, rng, vt = st["kl"], st["q2"], st["rng"], st["var_t"]
    ok = q2 > 0
    tilt_pos = (kl[ok] / q2[ok]).numpy()
    qs = [float(np.quantile(tilt_pos, q)) for q in (0.1, 0.5, 0.9, 0.99)] if tilt_pos.size else [None] * 4
    hoeff_bad = int(((kl > rng ** 2 / 8 * (1 + 1e-9) + 1e-12)).sum())
    g = ~torch.isnan(vt[:, 0])
    vg, klg = vt[g], kl[g]
    lo, hi = vg.min(1).values, vg.max(1).values
    tol = 1e-3
    # abs floor 1e-6: the log-probs are fp32 (resolution ~ 1e-7 |log p|), so Var and KL below ~1e-6 are rounding noise
    floor = 1e-6
    below, above = lo * (1 - tol) - 2 * klg, 2 * klg - hi * (1 + tol)
    mv_bad = int(((below > floor) | (above > floor)).sum())
    mv_bad_nofloor = int(((below > 1e-12) | (above > 1e-12)).sum())
    mv_excess = float(torch.maximum(below, above).max()) if klg.numel() else 0.0
    # exact quadrature: 2 KL = int_0^1 2(1-t) Var_t dt, Simpson on the 41-point grid
    wq = torch.ones(41, dtype=F64); wq[1:-1:2] = 4; wq[2:-1:2] = 2; wq *= (1 / 40) / 3
    quad = (vg * (2 * (1 - GRID))[None, :] * wq[None, :]).sum(1)
    qerr = (quad - 2 * klg).abs() / (2 * klg).clamp_min(1e-30)
    q2m = float(q2.mean())
    return dict(n_pos=kl.numel(), n_q2_zero=int((~ok).sum()), kl_mean=float(kl.mean()), kl32_mean=float(st["kl32"].mean()), q2_mean=q2m,
                tilt_set=float(kl.mean()) / q2m if q2m > 0 else None,
                tilt_pos_quantiles={"p10": qs[0], "p50": qs[1], "p90": qs[2], "p99": qs[3]}, tilt_pos_mean=float(tilt_pos.mean()) if tilt_pos.size else None,
                kl_share_range_gt_8=float(kl[rng > 8].sum() / kl.sum()) if float(kl.sum()) > 0 else None, frac_pos_range_gt_8=float((rng > 8).double().mean()),
                range_max=float(rng.max()), range_median=float(rng.median()),
                hoeffding_violations=hoeff_bad, grid_positions=int(g.sum()), meanvalue_violations=mv_bad, meanvalue_tol_rel=tol,
                meanvalue_abs_floor=floor, meanvalue_violations_nofloor=mv_bad_nofloor, meanvalue_max_excess=mv_excess,
                simpson_rel_err_max=float(qerr.max()) if qerr.numel() else None, simpson_rel_err_median=float(qerr.median()) if qerr.numel() else None)


def cmd_tilt(a):
    """Q = Q_tilt * Q_lin, Q_tilt = (KL/q2)_out / (KL/q2)_in, Q_lin = (q2/J)_out / (q2/J)_in, set-level means over positions.
    Cost: forward passes (teacher, student) over 60 chunks plus controls on 16 chunks, 41 softmax passes over about 30k grid positions."""
    t0 = time.time()
    rr, mm = json.load(open(a.run_receipt)), json.load(open(a.measure_receipt))
    Jo, Ji = mm["aggregate"]["J_out_sum"], mm["aggregate"]["J_in_sum"]
    ids_all, nch, ntok = load_ids()
    cfg, TL, hf = ft.load_model()
    del hf
    torch.cuda.empty_cache()
    Ls = torch.load(a.dump, map_location="cpu")["layers"]
    ids_ev, ids_in = ids_all[EVAL[0]:EVAL[1]], ids_all[:N_IN]
    sets = {}
    for tag, ids in (("in", ids_in), ("out", ids_ev)):
        sets[tag] = summarise(tilt_set(cfg, TL, Ls, ids.contiguous(), 1, 20000))
        log(tag, json.dumps(sets[tag], default=float))
    ctl = {}
    for name, c in (("delta_zero", "zero"), ("eps_0.05", 0.05)):
        ctl[name] = {tag: summarise(tilt_set(cfg, TL, Ls, ids[:4].contiguous(), 1, 20000, ctl=c)) for tag, ids in (("in", ids_in), ("out", ids_ev))}
        log(name, json.dumps({t: {k: v for k, v in s.items() if k != "tilt_pos_quantiles"} for t, s in ctl[name].items()}, default=float))
    zero_ok = all(abs(v["kl_mean"]) < 1e-12 and abs(v["q2_mean"]) < 1e-12 for v in ctl["delta_zero"].values())
    eps_ok = all(abs(v["tilt_set"] - 1) < 0.02 and v["hoeffding_violations"] == 0 and v["meanvalue_violations"] == 0 for v in ctl["eps_0.05"].values())
    ctl["delta_zero"]["pass_"], ctl["eps_0.05"]["pass_"] = bool(zero_ok), bool(eps_ok)
    kl_ok = abs(sets["in"]["kl32_mean"] - rr["kl_insample"]) < 1e-6 and abs(sets["out"]["kl32_mean"] - rr["kl_heldout"]) < 1e-6
    ctl["run100_reproduced"] = dict(kl_in=sets["in"]["kl32_mean"], kl_in_receipt=rr["kl_insample"], kl_out=sets["out"]["kl32_mean"],
                                    kl_out_receipt=rr["kl_heldout"], pass_=bool(kl_ok))
    tin, tout = sets["in"]["tilt_set"], sets["out"]["tilt_set"]
    lin_in, lin_out = sets["in"]["q2_mean"] / Ji, sets["out"]["q2_mean"] / Jo
    Q_tilt, Q_lin = tout / tin, lin_out / lin_in
    Qr = mm["QT"]["Q"]
    split = dict(J_out=Jo, J_in=Ji, tilt_in=tin, tilt_out=tout, q2_over_J_in=lin_in, q2_over_J_out=lin_out, Q_tilt=Q_tilt, Q_lin=Q_lin,
                 Q_product=Q_tilt * Q_lin, Q_receipt=Qr, product_minus_receipt=Q_tilt * Q_lin - Qr, product_ok=bool(abs(Q_tilt * Q_lin - Qr) < 1e-3),
                 tilt_fraction_of_logQ=math.log(Q_tilt) / math.log(Q_tilt * Q_lin))
    ok = bool(zero_ok and eps_ok and kl_ok and split["product_ok"] and all(s["hoeffding_violations"] == 0 and s["meanvalue_violations"] == 0 for s in sets.values()))
    out = dict(script_sha256=sha(__file__), fold_transfer_sha256=sha(ft.__file__), dump=a.dump, dump_sha256=sha(a.dump),
               run_receipt=a.run_receipt, measure_receipt=a.measure_receipt, heldout_chunks=list(EVAL), insample_chunks=[0, N_IN],
               sets=sets, split=split, controls=ctl, pass_=ok, wall_s=time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(sets=sets, split=split, controls={k: v.get("pass_") for k, v in ctl.items()}, pass_=ok, wall_s=out["wall_s"]), indent=1, default=float))
    if not ok:
        raise SystemExit("tilt probe gate failed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("controls")
    r = sub.add_parser("run")
    r.add_argument("--n-cal", type=int, required=True); r.add_argument("--cap", required=True); r.add_argument("--dump", required=True); r.add_argument("--out", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--cap", required=True); m.add_argument("--dump", required=True); m.add_argument("--n-cal", type=int, required=True)
    m.add_argument("--out", required=True); m.add_argument("--run-receipt", default=None); m.add_argument("--layers", type=int, default=24)
    tl = sub.add_parser("tilt")
    tl.add_argument("--dump", required=True); tl.add_argument("--run-receipt", required=True); tl.add_argument("--measure-receipt", required=True); tl.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--runs", required=True); f.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "controls":
        c = synthetic_controls()
        print(json.dumps(c, indent=1, default=float))
        if not c["pass_"]:
            raise SystemExit("controls failed")
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "measure":
        cmd_measure(a)
    elif a.cmd == "tilt":
        cmd_tilt(a)
    else:
        cmd_fit(a)


if __name__ == "__main__":
    main()
