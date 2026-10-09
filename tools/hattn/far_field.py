#!/usr/bin/env python3
"""HATTN-T3: attention as a Gaussian-screened N-body sum on Qwen2.5-0.5B at 32k context.

Identity: s q.k = s|q|^2/2 + s|k|^2/2 - s|q-k|^2/2, so softmax over keys is a sum over particles of mass
e^{s|k|^2/2} screened by e^{-s|q-k|^2/2} (the query term cancels). Far field: the keys outside a fixed near
set (4 sinks + last W) summarised by their first two moments (N, kbar, Sigma_kk, vbar, Sigma_vk); for jointly
Gaussian far keys
    log sum_far e^{s q.k} = log N + s q.kbar + s^2 q' Sigma_kk q / 2,   E_tilted[v] = vbar + s Sigma_vk q,
at O(d^2) per query, independent of t.

Dump (examples/attn_cut_probe.rs KV_DUMP mode): unrotated q{l}.bin (T x 14 x 64), k{l}.bin, v{l}.bin
(T x 2 x 64) f32, cs.bin (T x 32 (cos, sin) f64); rotation z = x[:32] + i x[32:], z * (cos + i sin).

  python3 tools/hattn/far_field.py --check   synthetic controls (Gaussian cloud exact, bimodal cloud and
                                             sign-flipped second cumulant must fail)
  python3 tools/hattn/far_field.py --full    P-PH1..P-PH4 tables (stdout deterministic, timing on stderr)
"""
import os
import sys
import time

import numpy as np

D = os.environ.get("HATTN_LONG_DIR", "/home/socol/oc-tmp/hattn-long/t32k")
TN, HD, H2, NH, NKV, NL = 32768, 64, 32, 14, 2, 24
GQA = NH // NKV
S = 1.0 / 8.0
QT = [512, 2048, 8192, 32768]
NQ = 8
NSINK, W = 4, 256
FAR = 1024
BLK = 2048
EPS = 1.0 - np.exp(-0.05)


def load(name, heads, n):
    a = np.fromfile(os.path.join(D, name), dtype="<f4", count=n * heads * HD)
    return a.reshape(n, heads, HD).astype(np.float64)


def rotate(x, cs):
    """x: (n, heads, 64) unrotated; cs: (n, 32, 2). Returns rotated, same layout."""
    c, s = cs[:, None, :, 0], cs[:, None, :, 1]
    a, b = x[..., :H2], x[..., H2:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


def far_field(q, kf, vf, order):
    """Log partition and tilted value mean of the far set from its moments (order 1 or 2)."""
    n = kf.shape[0]
    kb, vb = kf.mean(0), vf.mean(0)
    lz = np.log(n) + S * q @ kb
    vo = vb.copy()
    if order == 2:
        kc, vc = kf - kb, vf - vb
        skk = kc.T @ kc / n
        svk = vc.T @ kc / n
        lz = lz + S * S * q @ skk @ q / 2
        vo = vo + S * svk @ q
    return lz, vo


def combine(parts):
    """parts: list of (log Z, value mean). Returns the mixture value."""
    lz = np.array([p[0] for p in parts])
    m = lz.max()
    w = np.exp(lz - m)
    return sum(wi * p[1] for wi, p in zip(w, parts)) / w.sum()


def exact_part(q, k, v):
    sc = S * k @ q
    m = sc.max()
    e = np.exp(sc - m)
    return m + np.log(e.sum()), e @ v / e.sum()


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def check():
    rng = np.random.default_rng(7)
    ok = True
    n, d = 200_000, HD
    A = rng.normal(size=(d, d)) * 0.08
    kf = rng.normal(size=(n, d)) @ A + 0.5
    vf = kf @ (rng.normal(size=(d, d)) * 0.1) + rng.normal(size=(n, d)) * 0.2
    q = rng.normal(size=d)
    lz_t, v_t = exact_part(q, kf, vf)
    lz2, v2 = far_field(q, kf, vf, 2)
    lz1, v1 = far_field(q, kf, vf, 1)
    e2, e1 = rel(v2, v_t), rel(v1, v_t)
    print(f"gaussian cloud: |dlogZ| order2 {abs(lz2 - lz_t):.2e} order1 {abs(lz1 - lz_t):.2e}; "
          f"value rel-L2 order2 {e2:.2e} order1 {e1:.2e}")
    ok &= abs(lz2 - lz_t) < 1e-2 and e2 < 1e-2 and e1 > 5 * e2
    # bimodal cloud: two far-separated modes; the Gaussian field must miss
    kb = np.concatenate([kf[: n // 2] + 6.0, kf[n // 2:] - 6.0])
    lz_t, v_t = exact_part(q, kb, vf)
    lzb, vb = far_field(q, kb, vf, 2)
    print(f"bimodal cloud: |dlogZ| order2 {abs(lzb - lz_t):.2e}, value rel-L2 {rel(vb, v_t):.2e}")
    ok &= abs(lzb - lz_t) > 0.5
    # mutation: flip the sign of the second cumulant on the Gaussian cloud
    lz_t, v_t = exact_part(q, kf, vf)
    kc = kf - kf.mean(0)
    lzm = np.log(n) + S * q @ kf.mean(0) - S * S * q @ (kc.T @ kc / n) @ q / 2
    print(f"mutation (second cumulant sign flipped): |dlogZ| {abs(lzm - lz_t):.2e}")
    ok &= abs(lzm - lz_t) > 10 * abs(lz2 - lz_t) and abs(lzm - lz_t) > 1e-2
    print("check", "PASS" if ok else "FAIL")
    return ok


def slope(ts, ys):
    x, y = np.log(ts), np.log(ys)
    return float(np.polyfit(x, y, 1)[0])


def full():
    t0 = time.time()
    cs = np.fromfile(os.path.join(D, "cs.bin"), dtype="<f8").reshape(TN, H2, 2)
    keps = np.zeros((NL, NH, len(QT), NQ))
    neff = np.zeros_like(keps)
    err = {a: np.full((NL, NH, len(QT), NQ), np.nan) for a in ("trunc", "g1", "g2", "blk2")}
    dlz = np.full((NL, NH, len(QT), NQ), np.nan)
    share = np.zeros_like(keps)
    coh = np.zeros((NL, NKV, H2))
    for l in range(NL):
        q = rotate(load(f"q{l}.bin", NH, TN), cs)
        k = rotate(load(f"k{l}.bin", NKV, TN), cs)
        v = load(f"v{l}.bin", NKV, TN)
        for g in range(NKV):
            kf = k[: TN - 1 - FAR, g]
            z = kf[:, :H2] + 1j * kf[:, H2:]
            coh[l, g] = np.abs(z.mean(0)) / np.sqrt((np.abs(z) ** 2).mean(0))
        for ti, t in enumerate(QT):
            for qi in range(NQ):
                p = t - NQ + qi
                for h in range(NH):
                    g = h // GQA
                    qq, kk, vv = q[p, h], k[: p + 1, g], v[: p + 1, g]
                    sc = S * kk @ qq
                    m = sc.max()
                    e = np.exp(sc - m)
                    pr = e / e.sum()
                    ex = pr @ vv
                    srt = np.sort(pr)[::-1]
                    keps[l, h, ti, qi] = np.searchsorted(np.cumsum(srt), 1 - EPS) + 1
                    neff[l, h, ti, qi] = np.exp(-(pr * np.log(np.maximum(pr, 1e-300))).sum())
                    mass = S * (kk * kk).sum(1) / 2
                    scr = S * ((kk - qq) ** 2).sum(1) / 2
                    vm, vs = mass.var(), scr.var()
                    share[l, h, ti, qi] = vm / (vm + vs)
                    if p + 1 <= NSINK + W + 16:
                        continue
                    near = np.r_[0:NSINK, p + 1 - W: p + 1]
                    fs, fe = NSINK, p + 1 - W
                    pn = exact_part(qq, kk[near], vv[near])
                    pf = exact_part(qq, kk[fs:fe], vv[fs:fe])
                    err["trunc"][l, h, ti, qi] = rel(pn[1], ex)
                    g1 = far_field(qq, kk[fs:fe], vv[fs:fe], 1)
                    g2 = far_field(qq, kk[fs:fe], vv[fs:fe], 2)
                    err["g1"][l, h, ti, qi] = rel(combine([pn, g1]), ex)
                    err["g2"][l, h, ti, qi] = rel(combine([pn, g2]), ex)
                    dlz[l, h, ti, qi] = g2[0] - pf[0]
                    bl = [far_field(qq, kk[a: min(a + BLK, fe)], vv[a: min(a + BLK, fe)], 2)
                          for a in range(fs, fe, BLK) if min(a + BLK, fe) - a >= 2]
                    err["blk2"][l, h, ti, qi] = rel(combine([pn] + bl), ex)
        print(f"layer {l} {time.time() - t0:.0f}s", file=sys.stderr)

    med = lambda a: float(np.nanmedian(a))
    print(f"dump {D}; scale {S}; near = {NSINK} sinks + last {W}; eps = {EPS:.5f} (KL 0.05)")
    print("\n== P-PH1 concentration: k_eps (median over queries, then over heads), N_eff ==")
    km = np.median(keps, axis=3)
    for ti, t in enumerate(QT):
        print(f"t={t:6d}: k_eps median {med(km[:, :, ti]):8.1f}  p90 {np.percentile(km[:, :, ti], 90):8.1f}  "
              f"k_eps/t median {med(km[:, :, ti] / t):.4f}  N_eff median {med(np.median(neff[:, :, ti], 2)):8.1f}")
    al = np.array([[slope(QT[1:], np.maximum(km[l, h, 1:], 1)) for h in range(NH)] for l in range(NL)])
    fr = float((al > 0.8).mean())
    print(f"alpha (t=2048..32768) median {med(al):.3f}  p10 {np.percentile(al, 10):.3f}  p90 {np.percentile(al, 90):.3f}"
          f"  share alpha>0.8 {fr:.3f}  share alpha<=0.5 {float((al <= 0.5).mean()):.3f}")
    print("alpha by layer (median over heads):", " ".join(f"{np.median(al[l]):.2f}" for l in range(NL)))
    p1 = med(al) <= 0.5 and fr >= 0.2
    print(f"P-PH1 {'HELD' if p1 else 'REFUTED'}: median alpha {med(al):.3f} (<=0.5), share alpha>0.8 {fr:.3f} (>=0.2)")

    print("\n== P-PH2 Gaussian far field: attention-output rel-L2 vs exact (median over layer, head, query) ==")
    for ti, t in enumerate(QT):
        print(f"t={t:6d}: trunc {med(err['trunc'][:, :, ti]):.4f}  g1 {med(err['g1'][:, :, ti]):.4f}  "
              f"g2 {med(err['g2'][:, :, ti]):.4f}  blk{BLK} g2 {med(err['blk2'][:, :, ti]):.4f}  "
              f"g2 p90 {np.nanpercentile(err['g2'][:, :, ti], 90):.4f}  |dlogZ_far| median {med(np.abs(dlz[:, :, ti])):.4f}")
    e2, e1 = med(err["g2"][:, :, 3]), med(err["g1"][:, :, 3])
    print("g2 at 32768 by layer:", " ".join(f"{np.nanmedian(err['g2'][l, :, 3]):.3f}" for l in range(NL)))
    p2 = e2 <= 0.10 and e1 >= 2 * e2
    print(f"P-PH2 {'HELD' if p2 else 'REFUTED'}: median g2 {e2:.4f} (<=0.10), g1/g2 {e1 / e2:.2f} (>=2)")

    print("\n== P-PH3 mass vs screening: Var(mass)/(Var(mass)+Var(screen)) ==")
    for ti, t in enumerate(QT):
        print(f"t={t:6d}: median {med(share[:, :, ti]):.4f}  p90 {np.percentile(share[:, :, ti], 90):.4f}")
    p3 = med(share) <= 0.3
    print(f"P-PH3 {'HELD' if p3 else 'REFUTED'}: median share {med(share):.4f} (<=0.3)")

    print(f"\n== P-PH4 RoPE coherence |E z_j|/sqrt(E|z_j|^2), keys 0..{TN - 2 - FAR}, median over (layer, kv head) ==")
    cm = np.median(coh.reshape(-1, H2), axis=0)
    print("j:  " + " ".join(f"{j:5d}" for j in range(H2)))
    print("c:  " + " ".join(f"{c:5.2f}" for c in cm))
    p4 = bool((cm[:16] <= 0.1).all() and (cm[24:] >= 0.5).all())
    print(f"P-PH4 {'HELD' if p4 else 'REFUTED'}: max c_j (j<16) {cm[:16].max():.3f} (<=0.1), min c_j (j>=24) {cm[24:].min():.3f} (>=0.5)")
    print(f"elapsed {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    full()
