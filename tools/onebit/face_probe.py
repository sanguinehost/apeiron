"""Junction probe: sequential versus joint two-plane sign codes at equal bits (W16 T2).

Question: does fitting w ~ r1 s1 + r2 s2 jointly, one coordinate move setting (s1_p, s2_p) to one
of the four levels {+a, +b, -b, -a} with a = r1 + r2, b = r1 - r2, reach lower curvature-weighted
error J than the sequential fit (plane 1, plane 2 on the residual, backfit rounds, joint 2x2 LS)?

Fits, per tensor, in the Hadamard-rotated basis (W_r = W R, H_r = R^T H R, R block-diagonal
orthonormal Hadamard, fold_transfer.fwht128 / had_gram), row-batched, J_i = (w - wh)^T H (w - wh):
  SEQ        plane 1, plane 2 on the residual, T = 2 backfit rounds, joint 2x2 LS on (r1, r2).
  JOINT-COLD SEQ's first two stages (no backfit) as init, then T + 2 rounds of K joint sweeps and a
             joint 2x2 LS refit; a round is kept per row only if J falls.
  JOINT-WARM the same rounds started from SEQ's final code.
A joint coordinate move changes (s1_p, s2_p) together, which a plane-wise move cannot do in one step.
Reports J/J0 (J0 = sum_i w^T H w) per tensor kind, layer and aggregate; sweeps; wall.
Also: the diagonal/off-diagonal share of SEQ's cross term s1^T H s2, and the junction imbalance of
every layer on the unrotated weights.

Tensors: k, v, o per layer (Grams attn_in, attn_in, attn_out_in); top-416 rows of gate and of up by
row mass (W**2).sum(1) (Gram ffn_in). Inputs: fold_transfer.load_model, f64 Grams in
~/gptq-w2/gram.

Scope: sign-level. The best arm's planes are E8 u O8 lattice codes (fold_transfer e8_code), not
+-1 signs; this probe bounds the mechanism, not the arm.

Complexity: one ICM or joint sweep is n coordinate steps; each step evaluates dJ in O(B) and updates
g += d H[p] in O(B n), so a sweep is O(B n^2) multiply-adds for B rows and width n. An LS refit is
O(B n^2). Per-tensor cost is (number of sweeps) x O(B n^2); the sweep counts are in the JSON.
"""
import argparse, itertools, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_transfer as ft

GRAM = "~/gptq-w2/gram"
F64 = torch.float64
GROUP = 128
TOP_MLP_ROWS = 416
# (s1, s2) of the four joint levels a, b, -b, -a: value r1 s1 + r2 s2.
C1 = torch.tensor([1., 1., -1., -1.], dtype=F64)
C2 = torch.tensor([1., -1., 1., -1.], dtype=F64)


def delta_j(g_p, delta, h_pp):
    """dJ for moving yhat_p by delta, with g = H(yhat - w): 2 delta g_p + delta^2 H_pp."""
    return 2 * delta * g_p + delta * delta * h_pp


def J(w, what, H):
    D = w - what
    return ((D @ H) * D).sum(1)


def yhat(S, R):
    return R[0][:, None] * S[0] + R[1][:, None] * S[1]


def grad(w, what, H):
    return (what - w) @ H


def sign1(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def refit_plane(w, H, S, R, p):
    """LS r_p per row with plane 1-p fixed: minimise ||T - r s_p||_H, T = w - r_(1-p) s_(1-p)."""
    o = 1 - p
    T = w - R[o][:, None] * S[o]
    s = S[p]
    return ((T @ H) * s).sum(1) / ((s @ H) * s).sum(1)


def icm_plane(w, H, S, R, p, K):
    """K single-flip ICM sweeps on plane p (other plane fixed), r_p refit after each sweep."""
    n = w.shape[1]
    sweeps = 0
    for _ in range(K):
        g = grad(w, yhat(S, R), H)
        flips = 0
        for j in range(n):
            delta = -2 * R[p] * S[p][:, j]
            acc = delta_j(g[:, j], delta, H[j, j]) < 0
            if bool(acc.any()):
                d = torch.where(acc, delta, torch.zeros_like(delta))
                S[p][:, j] = torch.where(acc, -S[p][:, j], S[p][:, j])
                g += d[:, None] * H[j][None, :]
                flips += int(acc.sum())
        R[p] = refit_plane(w, H, S, R, p)
        sweeps += 1
        if flips == 0:
            break
    return sweeps


def joint_levels(R):
    return R[0][:, None] * C1 + R[1][:, None] * C2  # (B, 4): values +a, +b, -b, -a


def joint_sweep(w, H, S, R):
    """One sweep: each coordinate (s1_p, s2_p) takes the best of the four levels. Returns flips."""
    n = w.shape[1]
    g = grad(w, yhat(S, R), H)
    vals = joint_levels(R)
    flips = 0
    for j in range(n):
        cur = R[0] * S[0][:, j] + R[1] * S[1][:, j]
        delta = vals - cur[:, None]
        dj = delta_j(g[:, j:j + 1], delta, H[j, j])
        best = dj.argmin(1)
        dmin = dj.gather(1, best[:, None])[:, 0]
        acc = dmin < 0
        if bool(acc.any()):
            dl = torch.where(acc, delta.gather(1, best[:, None])[:, 0], torch.zeros_like(cur))
            S[0][:, j] = torch.where(acc, C1[best], S[0][:, j])
            S[1][:, j] = torch.where(acc, C2[best], S[1][:, j])
            g += dl[:, None] * H[j][None, :]
            flips += int(acc.sum())
    return flips


def refit_joint(w, H, S):
    """Joint LS (r1, r2) per row for the codes (s1, s2): pinv of the 2x2 normal matrix."""
    HU0, HU1 = S[0] @ H, S[1] @ H
    A = torch.stack([torch.stack([(HU0 * S[0]).sum(1), (HU0 * S[1]).sum(1)], 1),
                     torch.stack([(HU1 * S[0]).sum(1), (HU1 * S[1]).sum(1)], 1)], 1)
    c = torch.stack([(HU0 * w).sum(1), (HU1 * w).sum(1)], 1)
    r = (torch.linalg.pinv(A) @ c[:, :, None])[:, :, 0]
    return [r[:, 0], r[:, 1]]


def fit_seq(w, H, K, T):
    """Returns (init state after stages 1-2, final state after backfit and joint LS, sweeps)."""
    S = [sign1(w), torch.zeros_like(w)]
    R = [torch.zeros(w.shape[0], dtype=F64), torch.zeros(w.shape[0], dtype=F64)]
    R[0] = refit_plane(w, H, S, R, 0)
    sw = icm_plane(w, H, S, R, 0, K)
    S[1] = sign1(w - R[0][:, None] * S[0])
    R[1] = refit_plane(w, H, S, R, 1)
    sw += icm_plane(w, H, S, R, 1, K)
    init = ([S[0].clone(), S[1].clone()], [R[0].clone(), R[1].clone()], sw)
    for _ in range(T):
        trial = [S[0].clone(), S[1].clone()]
        Rt = [R[0].clone(), R[1].clone()]
        sw += icm_plane(w, H, trial, Rt, 0, K)
        sw += icm_plane(w, H, trial, Rt, 1, K)
        keep = J(w, yhat(trial, Rt), H) < J(w, yhat(S, R), H)
        S = [torch.where(keep[:, None], trial[p], S[p]) for p in range(2)]
        R = [torch.where(keep, Rt[p], R[p]) for p in range(2)]
    r = refit_joint(w, H, S)
    return init, (S, r, sw)


def fit_joint(w, H, S, R, K, rounds):
    """Joint alternation: K joint sweeps then joint LS, a round kept per row only if J falls."""
    sw = 0
    Jc = J(w, yhat(S, R), H)
    for _ in range(rounds):
        trial = [S[0].clone(), S[1].clone()]
        for _k in range(K):
            sw += 1
            if joint_sweep(w, H, trial, R) == 0:
                break
        Rt = refit_joint(w, H, trial)
        Jt = J(w, yhat(trial, Rt), H)
        keep = Jt < Jc
        if not bool(keep.any()):
            break
        S = [torch.where(keep[:, None], trial[p], S[p]) for p in range(2)]
        R = [torch.where(keep, Rt[p], R[p]) for p in range(2)]
        Jc = torch.where(keep, Jt, Jc)
    return S, R, sw


def fit_all(w, H, K, T):
    (Si, Ri, swi), (Ss, Rs, sws) = fit_seq(w, H, K, T)
    Sc, Rc, swc = fit_joint(w, H, Si, Ri, K, T + 2)
    Sw, Rw, sww = fit_joint(w, H, Ss, Rs, K, T + 2)
    return {"seq": (Ss, Rs, sws), "cold": (Sc, Rc, swi + swc), "warm": (Sw, Rw, sww)}


def cross_split(H, S):
    diag = (S[0] * S[1] * torch.diagonal(H)[None, :]).sum(1)
    off = ((S[0] @ H) * S[1]).sum(1) - diag
    return float(diag.abs().sum()), float(off.abs().sum())


def exhaustive_min(w, H):
    """Per row: min over all (s1, s2) in {+-1}^n x {+-1}^n of J with LS (r1, r2)."""
    n = w.shape[1]
    codes = torch.tensor(list(itertools.product([-1., 1.], repeat=n)), dtype=F64)
    M = codes.shape[0]
    a = ((codes @ H) * codes).sum(1)
    G = codes @ H @ codes.T
    Cw = codes @ H @ w.T  # (M, B)
    out = []
    for i in range(w.shape[0]):
        A = torch.stack([torch.stack([a[:, None].expand(M, M), G], -1),
                         torch.stack([G.T, a[None, :].expand(M, M)], -1)], -2).reshape(M * M, 2, 2)
        c = torch.stack([Cw[:, i][:, None].expand(M, M), Cw[:, i][None, :].expand(M, M)], -1).reshape(M * M, 2, 1)
        quad = (c.transpose(1, 2) @ torch.linalg.pinv(A) @ c).reshape(-1)
        out.append(float((w[i] @ H @ w[i]) - quad.max()))
    return out


def self_check(K, T):
    g = torch.Generator().manual_seed(0)
    # (a) J(w, w) = 0
    w = torch.randn(5, 16, generator=g, dtype=F64)
    assert float(J(w, w, torch.eye(16, dtype=F64)).abs().max()) < 1e-12, "(a) J(w,w) != 0"
    # (b) batched J equals an explicit double loop on a 4x8 case
    A = torch.randn(8, 8, generator=g, dtype=F64)
    Hs = A @ A.T + torch.eye(8, dtype=F64)
    w, wh = torch.randn(4, 8, generator=g, dtype=F64), torch.randn(4, 8, generator=g, dtype=F64)
    Jb = J(w, wh, Hs)
    for i in range(4):
        d = (w[i] - wh[i]).tolist()
        e = sum(d[p] * Hs[p, q].item() * d[q] for p in range(8) for q in range(8))
        assert abs(e - float(Jb[i])) < 1e-12, f"(b) row {i}: loop {e} batched {float(Jb[i])}"
    # (c) exact synthetic, n = 128, 16 rows, r1 > r2 > 0, H = I: JOINT-COLD must reach J/J0 < 1e-9
    n, B = 128, 16
    r1 = 1.0 + torch.rand(B, generator=g, dtype=F64)
    r2 = 0.05 + 0.85 * torch.rand(B, generator=g, dtype=F64)
    s1, s2 = sign1(torch.randn(B, n, generator=g, dtype=F64)), sign1(torch.randn(B, n, generator=g, dtype=F64))
    wc = r1[:, None] * s1 + r2[:, None] * s2
    Ic = torch.eye(n, dtype=F64)
    fits = fit_all(wc, Ic, K, T)
    J0c = float((wc * wc).sum())
    rs = {k: float(J(wc, yhat(v[0], v[1]), Ic).sum()) / J0c for k, v in fits.items()}
    print(f"self-check (c) J/J0: seq {rs['seq']:.3e} cold {rs['cold']:.3e} warm {rs['warm']:.3e}", flush=True)
    assert rs["cold"] < 1e-9, f"(c) JOINT-COLD J/J0 {rs['cold']:.3e} >= 1e-9"
    # (d) n = 8, 4 rows, random SPD H: exhaustive optimum lower-bounds SEQ and JOINT
    A = torch.randn(8, 8, generator=g, dtype=F64)
    Hd = A @ A.T + 0.1 * torch.eye(8, dtype=F64)
    wd = torch.randn(4, 8, generator=g, dtype=F64)
    opt = exhaustive_min(wd, Hd)
    fd = fit_all(wd, Hd, K, T)
    for k, v in fd.items():
        Jk = J(wd, yhat(v[0], v[1]), Hd)
        for i in range(4):
            assert float(Jk[i]) >= opt[i] - 1e-9, f"(d) {k} row {i}: {float(Jk[i])} < optimum {opt[i]}"
    print(f"self-check (d) exhaustive optimum per row {['%.4g' % x for x in opt]}", flush=True)
    # (e) descent: every accepted ICM or joint move, and each LS refit, does not raise J (random start)
    A = torch.randn(16, 16, generator=g, dtype=F64)
    He = A @ A.T + 0.5 * torch.eye(16, dtype=F64)
    we = torch.randn(8, 16, generator=g, dtype=F64)
    S0 = [sign1(torch.randn(8, 16, generator=g, dtype=F64)), sign1(torch.randn(8, 16, generator=g, dtype=F64))]
    R0 = [0.5 + torch.rand(8, generator=g, dtype=F64), torch.rand(8, generator=g, dtype=F64) - 0.5]
    for name in ("icm", "joint"):
        Sx, Rx = [S0[0].clone(), S0[1].clone()], [R0[0].clone(), R0[1].clone()]
        J_before = J(we, yhat(Sx, Rx), He)
        if name == "icm":
            icm_plane(we, He, Sx, Rx, 0, 1)
        else:
            joint_sweep(we, He, Sx, Rx)
        J_after = J(we, yhat(Sx, Rx), He)
        assert bool((J_after <= J_before + 1e-9).all()), f"(e) {name} raised J: {J_before.tolist()} -> {J_after.tolist()}"
    print("self-check (e) descent ok", flush=True)
    print("self-check ok", flush=True)


def stats(x):
    x = x.double().cpu().numpy()
    return dict(median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                max=float(x.max()), min=float(x.min()), max_over_min=float(x.max() / x.min()))


def imbalance(TL):
    out = {}
    for l, L in enumerate(TL):
        up, down = L["up"].double(), L["down"].double()
        rho = down.norm(dim=0) / up.norm(dim=1)
        v, o = L["v"].double(), L["o"].double()
        assert v.shape == (128, 896) and o.shape == (896, 896)
        o3 = o.reshape(896, 14, 64)
        vn = v.norm(dim=1).reshape(2, 64)
        ocol = torch.stack([o3[:, 7 * h:7 * h + 7, :].pow(2).sum(dim=(0, 1)).sqrt() for h in range(2)])
        out[l] = {"up_down": stats(rho), "v_o": stats((ocol / vn).reshape(-1))}
    return out


def gram(l, name):
    path = f"{GRAM}/blk.{l}.{name}.f64"
    a = np.fromfile(path, dtype="<f8")
    n = int(round(a.size ** 0.5))
    assert n * n == a.size, path
    H = torch.from_numpy(a.reshape(n, n).copy()).to(F64)
    return (H + H.T) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="0,5,11,17,23")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    self_check(a.K, a.T)
    if a.check_only:
        return
    layers = [int(x) for x in a.layers.split(",")]
    cfg, TL, _ = ft.load_model()
    assert ft.GROUP == GROUP if hasattr(ft, "GROUP") else True
    kinds = {"k": "attn_in", "v": "attn_in", "o": "attn_out_in"}
    agg = {k: dict(J0=0.0, seq=0.0, cold=0.0, warm=0.0) for k in ("k", "v", "o", "gate", "up")}
    res = {}
    for l in layers:
        res[l] = {}
        for kind in ("k", "v", "o", "gate", "up"):
            W = TL[l][kind].double()
            if kind in ("gate", "up"):
                idx = torch.topk((W ** 2).sum(1), TOP_MLP_ROWS).indices
                W = W[idx]
                gname = "ffn_in"
            else:
                gname = kinds[kind]
            Hr = ft.had_gram(gram(l, gname))
            Wr = ft.fwht128(W).contiguous()
            t0 = time.time()
            f = fit_all(Wr, Hr, a.K, a.T)
            J0 = float(J(Wr, torch.zeros_like(Wr), Hr).sum())
            vals = {k: float(J(Wr, yhat(v[0], v[1]), Hr).sum()) for k, v in f.items()}
            share = cross_split(Hr, f["seq"][0])
            rec = dict(J0=J0, seq=vals["seq"] / J0, cold=vals["cold"] / J0, warm=vals["warm"] / J0,
                       cold_over_seq=vals["cold"] / vals["seq"], warm_over_seq=vals["warm"] / vals["seq"],
                       sweeps=dict(seq=f["seq"][2], cold=f["cold"][2], warm=f["warm"][2]),
                       cross_diag_abs=share[0], cross_off_abs=share[1],
                       cross_diag_share=share[0] / (share[0] + share[1]), wall_s=time.time() - t0)
            res[l][kind] = rec
            ag = agg[kind]
            ag["J0"] += J0
            ag["seq"] += vals["seq"]
            ag["cold"] += vals["cold"]
            ag["warm"] += vals["warm"]
            print(f"L{l} {kind:4s} seq {rec['seq']:.5f} cold {rec['cold']:.5f} warm {rec['warm']:.5f} "
                  f"cold/seq {rec['cold_over_seq']:.4f} warm/seq {rec['warm_over_seq']:.4f} "
                  f"sweeps {rec['sweeps']} wall {rec['wall_s']:.1f}s", flush=True)
    kinds_out = {}
    for k, ag in agg.items():
        kinds_out[k] = dict(seq=ag["seq"] / ag["J0"], cold=ag["cold"] / ag["J0"], warm=ag["warm"] / ag["J0"],
                            cold_over_seq=ag["cold"] / ag["seq"], warm_over_seq=ag["warm"] / ag["seq"])
    kv = {x: agg[x] for x in ("k", "v", "o")}
    J0kv = sum(x["J0"] for x in kv.values())
    gain_warm = (sum(x["seq"] for x in kv.values()) - sum(x["warm"] for x in kv.values())) / J0kv
    seq_kvo = sum(x["seq"] for x in kv.values())
    warm_kvo = sum(x["warm"] for x in kv.values())
    decision = dict(kvo_seq_minus_warm_over_J0=gain_warm, kvo_seq_minus_warm_over_seq_J=(seq_kvo - warm_kvo) / seq_kvo,
                    W16b_worth_opening_J0_units=bool(gain_warm > 0.01),
                    W16b_worth_opening_relative_to_seq_J=bool((seq_kvo - warm_kvo) / seq_kvo > 0.01))
    out = dict(
        note="Best arm's planes are E8 u O8 lattice codes (fold_transfer e8_code), not +-1 signs; "
             "sign-level probe, bounds the mechanism not the arm.",
        args=dict(layers=layers, K=a.K, T=a.T), per_layer=res, per_kind=kinds_out,
        imbalance={str(l): v for l, v in imbalance(TL).items()}, decision=decision,
        wall_s=time.time() - t_all)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in kinds_out.items():
        print(f"KIND {k}: seq {v['seq']:.5f} cold {v['cold']:.5f} warm {v['warm']:.5f} "
              f"cold/seq {v['cold_over_seq']:.4f} warm/seq {v['warm_over_seq']:.4f}", flush=True)
    print(f"decision: {decision}  wall {out['wall_s']:.1f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
