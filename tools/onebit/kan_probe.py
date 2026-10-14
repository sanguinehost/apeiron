"""Learned elliptic register rotation vs the fixed Sylvester Hadamard, scored by the E8 code's Hessian-weighted error.

docs/epics/quantum-register.md W1-T1. A 128-group is a register of 2^7 amplitudes; fwht128 is seven stages of 64
butterflies (u, v) -> (u + v, u - v)/sqrt2. Each butterfly here is the reflection [[c, s], [s, -c]] with
(c, s) = (cos t, sin t): the elliptic cell multiply by (c, s) followed by the sign diagonal Z. At t = pi/4 the stage is
the normalised butterfly, so the register at all-pi/4 angles is fwht128 (control `identity`, float64).

Angles are tied per input site (attn_in for q,k,v; attn_out_in for o; ffn_in for gate,up; ffn_down_in for down) and per
(stage, butterfly), shared across the site's 128-groups: 448 angles per site, 16 bits each (counted in `bits`).

Learning: minimise sum over the site's linears of tr(E Gr E^T), E = W T - a * Q(W T / a), Gr = T^T G T + ridge, Q the
nearest point of E8 u O8 per 8-block (detached; a per row per group, invariant under T because T is orthogonal within a
group). Scoring: the real gptq_e8 (act_order, wscale, beam 4, as the -nb arm) on (W T, T^T G_cal T), J measured on the
calibration Gram and on a held-out Gram of disjoint chunks, against the same solve with fwht128.

Cost per site and step: one register sweep over W (7 * 64 * 2 multiplies per 128 words per row) and over G twice,
one (blocks x 8) @ (8 x 256) codeword product, and one O x n x n product for tr(E Gr E^T); n = 4864 dominates
(896 * 4864^2 = 2.1e10 multiply-adds per step for ffn_down_in).
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_transfer as ft  # noqa: E402

DEV, GROUP, CHUNK = ft.DEV, ft.GROUP, ft.CHUNK
SITES = [("attn_in", ["q", "k", "v"], "xa"), ("attn_out_in", ["o"], "a"), ("ffn_in", ["gate", "up"], "xf"),
         ("ffn_down_in", ["down"], "m")]
NST, NBF = 7, GROUP // 2


def register(x, theta):
    """x (..., n) with n a multiple of 128; theta (7, 64). Row-vector map x -> x T. O(7 * 64 * 2) multiplies per group."""
    s = x.shape
    y = x.reshape(-1, GROUP)
    h = 1
    for st in range(NST):
        c, sn = theta[st].cos().view(GROUP // (2 * h), h), theta[st].sin().view(GROUP // (2 * h), h)
        y = y.view(-1, GROUP // (2 * h), 2, h)
        u, v = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((c * u + sn * v, sn * u - c * v), dim=2)
        h *= 2
    return y.reshape(s)


def rot_gram(G, theta):
    return register(register(G, theta).T.contiguous(), theta).T.contiguous()


def quant_angles(theta, bits=16):
    q = 2 * math.pi / (1 << bits)
    return torch.round(theta / q) * q


def surrogate(Ws, G, theta, C, ridge):
    """sum_W tr(E Gr E^T) with Q detached. Per row per group a = 0.828 * rms (gptq_e8's initial scale, unweighted)."""
    Gr = rot_gram(G, theta)
    Gr = Gr + ridge * Gr.diagonal().mean() * torch.eye(Gr.shape[0], dtype=Gr.dtype, device=Gr.device)
    tot = 0.0
    for W in Ws:
        Wr = register(W, theta)
        O, n = Wr.shape
        a = (0.828 * Wr.view(O, n // GROUP, GROUP).pow(2).mean(-1).clamp_min(1e-24).sqrt()).detach()
        z = (Wr.view(O, n // GROUP, GROUP) / a[..., None]).reshape(-1, 8)
        with torch.no_grad():
            idx = ((C * C).sum(-1)[None, :] - 2.0 * z @ C.T).argmin(-1)
        E = (z - C[idx]).view(O, n // GROUP, GROUP) * a[..., None]
        E = E.reshape(O, n)
        tot = tot + ((E @ Gr) * E).sum()
    return tot


def gptq_J(Ws, G_cal, G_ev, rotate, ridge, damp):
    """rotate: X -> X T on rows. Returns (J_cal, J_ev) summed over Ws, each J = tr(D Gl D^T) in the rotated frame."""
    Gc = rotate(rotate(G_cal).T.contiguous()).T.contiguous()
    Ge = rotate(rotate(G_ev).T.contiguous()).T.contiguous()
    Hc, He = ft.metric(Gc, ridge), ft.metric(Ge, ridge)
    jc = je = 0.0
    for W in Ws:
        Wr = rotate(W)
        Wq, _, _ = ft.gptq_e8(Wr, Gc, damp=damp, act_order=True, wscale=True, beam=4)
        jc += ft.J(Wr, Wq, Hc)
        je += ft.J(Wr, Wq, He)
    return jc, je


def grams(cfg, L, hs, lo, hi, bs):
    """Teacher input Grams of the four sites over chunks [lo, hi); X^T X accumulated in float64."""
    acc = {}
    for b0 in range(lo, hi, bs):
        h = hs[b0:min(b0 + bs, hi)].to(DEV)
        p = ft.parts(cfg, L, h)
        for site, _, key in SITES:
            x = p[key].reshape(-1, p[key].shape[-1]).double()
            acc[site] = acc.get(site, 0) + x.T @ x
    return {k: v.float() for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--n-cal", type=int, default=100); ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--steps", type=int, default=120); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--ridge", type=float, default=1e-3); ap.add_argument("--damp", type=float, default=0.02)
    ap.add_argument("--angle-bits", type=int, default=16); ap.add_argument("--bs", type=int, default=10)
    a = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(ft.MODEL)
    ids = tok.encode(open(ft.CALIB, encoding="utf-8").read())
    nch = len(ids) // CHUNK
    ids_all = torch.tensor(ids[:nch * CHUNK]).view(nch, CHUNK)[:a.n_cal + a.n_eval]
    cfg, TL, hf = ft.load_model()
    del hf
    C = ft.e8_o8_codebook(torch.float32, DEV)

    # control: all-pi/4 register == fwht128 in float64
    X = torch.randn(64, 4 * GROUP, dtype=torch.float64, device=DEV)
    th4 = torch.full((NST, NBF), math.pi / 4, dtype=torch.float64, device=DEV)
    ident = float((register(X, th4) - ft.fwht128(X)).abs().max())
    assert ident < 1e-12, ident
    print(f"identity max|register(pi/4) - fwht128| = {ident:.3e}", flush=True)

    hs = cfg.E[ids_all].float()
    layers = list(range(cfg.nl)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    rows, site_bits = [], 0
    for l in range(cfg.nl):
        if l > max(layers):
            break
        L = {k: v.to(DEV) for k, v in TL[l].items()}
        if l in layers:
            Gc = grams(cfg, L, hs, 0, a.n_cal, a.bs)
            Ge = grams(cfg, L, hs, a.n_cal, a.n_cal + a.n_eval, a.bs)
            for site, names, _ in SITES:
                Ws = [L[nm].float() for nm in names]
                G = Gc[site]
                theta = torch.full((NST, NBF), math.pi / 4, dtype=torch.float32, device=DEV, requires_grad=True)
                opt = torch.optim.Adam([theta], lr=a.lr)
                s0 = float(surrogate(Ws, G, theta.detach(), C, a.ridge))
                for _ in range(a.steps):
                    opt.zero_grad(set_to_none=True)
                    loss = surrogate(Ws, G, theta, C, a.ridge)
                    loss.backward()
                    opt.step()
                th_q = quant_angles(theta.detach(), a.angle_bits)
                s1 = float(surrogate(Ws, G, th_q, C, a.ridge))
                jhc, jhe = gptq_J(Ws, G, Ge[site], ft.fwht128, a.ridge, a.damp)
                jtc, jte = gptq_J(Ws, G, Ge[site], lambda x: register(x, th_q), a.ridge, a.damp)
                site_bits += NST * NBF * a.angle_bits
                r = dict(layer=l, site=site, surr_h=s0, surr_t=s1, J_had_cal=jhc, J_kan_cal=jtc, J_had_ev=jhe,
                         J_kan_ev=jte, dtheta_max=float((th_q - math.pi / 4).abs().max()),
                         dtheta_rms=float((th_q - math.pi / 4).pow(2).mean().sqrt()))
                rows.append(r)
                print(f"L{l:2d} {site:12s} surr {s1/s0:.4f}  J_cal {jtc/jhc:.4f}  J_ev {jte/jhe:.4f}  "
                      f"dth_rms {r['dtheta_rms']:.4f}  t {time.time()-t0:.0f}s", flush=True)
        with torch.no_grad():
            for b0 in range(0, hs.shape[0], a.bs):
                hs[b0:b0 + a.bs] = ft.parts(cfg, L, hs[b0:b0 + a.bs].to(DEV))["out"].cpu()
        del L
    agg = {}
    for key in ("surr", "cal", "ev"):
        h = sum(r["surr_h" if key == "surr" else f"J_had_{key}"] for r in rows)
        t = sum(r["surr_t" if key == "surr" else f"J_kan_{key}"] for r in rows)
        agg[f"ratio_{key}"] = t / h
        agg[f"geo_{key}"] = float(np.exp(np.mean([np.log((r["surr_t"] / r["surr_h"]) if key == "surr" else
                                                         (r[f"J_kan_{key}"] / r[f"J_had_{key}"])) for r in rows])))
    n_params = sum(TL[0][k].numel() for k in ("q", "k", "v", "o", "gate", "up", "down")) * cfg.nl
    res = dict(script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               fold_transfer_sha256=hashlib.sha256(open(ft.__file__, "rb").read()).hexdigest(),
               args=vars(a), gpu=torch.cuda.get_device_name(0), identity_max_abs=ident, layers=layers,
               angle_bits_total=site_bits, angle_bits_full_model=NST * NBF * a.angle_bits * len(SITES) * cfg.nl,
               linear_params=n_params, angle_bpw_full_model=NST * NBF * a.angle_bits * len(SITES) * cfg.nl / n_params,
               rows=rows, wall_s=time.time() - t0, **agg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
