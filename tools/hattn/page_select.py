#!/usr/bin/env python3
"""HATTN-T4: keys read by page-bound selection on Qwen2.5-0.5B at 32k context.

Quest (Tang et al., arXiv 2406.10774v2, sec. 3): keys in pages of P; each page keeps its element-wise min m and
max M key; for query q the page bound is sum_i max(q_i m_i, q_i M_i) = M.q+ + m.q- >= max over the page of q.k.
Pages are read in bound order. Arms: A (P=16), A64 (P=64), B (P=16, bound over the RoPE pairs j >= 16 only),
H (best-first over a fanout-16 tree of bounds: 16-key pages, 256-key and 4096-key nodes), OP (pages in true
mass order). Near set (always read): 4 sinks, the last W keys, and the partial far page.

Cost: key-vector reads per query; a bound costs 2 (min, max), a B bound 1, a selected page P. The adaptive arms
stop on the oracle kept mass >= 1 - eps (KL 0.05): they measure ranking quality, not a deployable stop rule.

  python3 tools/hattn/page_select.py --check   bound soundness, planted-needle ranking; HATTN_MUT=1 (min-only
                                               bound) must fail soundness
  python3 tools/hattn/page_select.py --full    PQ1..PQ6 tables (stdout deterministic, timing on stderr)
"""
import heapq
import os
import sys
import time

import numpy as np

D = os.environ.get("HATTN_LONG_DIR", "/home/socol/oc-tmp/hattn-long/t32k")
TN, HD, H2, NH, NKV, NL = 32768, 64, 32, 14, 2, 24
GQA = NH // NKV
S = 1.0 / 8.0
QT = [2048, 8192, 32768]
NQ = 8
NSINK, W = 4, 256
FAN = 16
BUDGETS = [512, 1024, 2048]
EPS = 1.0 - np.exp(-0.05)
SLOW = np.r_[16:32, 48:64]
MUT = os.environ.get("HATTN_MUT") == "1"


def load(name, heads, n):
    a = np.fromfile(os.path.join(D, name), dtype="<f4", count=n * heads * HD)
    return a.reshape(n, heads, HD).astype(np.float64)


def rotate(x, cs):
    c, s = cs[:, None, :, 0], cs[:, None, :, 1]
    a, b = x[..., :H2], x[..., H2:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


def minmax(k, p):
    """k: (n, d), n a multiple of p. Element-wise (min, max) per page of p rows."""
    r = k.reshape(-1, p, k.shape[1])
    return r.min(1), r.max(1)


def bound(mn, mx, q):
    """Quest page bound: sum_i max(q_i m_i, q_i M_i). O(pages * d) multiplies."""
    if MUT:
        return mn @ q
    return mx @ np.maximum(q, 0) + mn @ np.minimum(q, 0)


def need(pm, order, near):
    """Pages read in `order` until near + kept mass >= 1 - eps."""
    rest = 1 - EPS - near
    if rest <= 0:
        return 0
    c = np.cumsum(pm[order])
    return min(int(np.searchsorted(c, rest - 1e-12)) + 1, len(order))


def tree(mn, mx):
    """Levels of (min, max): level 0 pages, level l nodes of FAN children (last node partial). O(pages * d)."""
    lv = [(mn, mx)]
    while lv[-1][0].shape[0] > 1:
        a, b = lv[-1]
        ix = np.arange(0, a.shape[0], FAN)
        lv.append((np.minimum.reduceat(a, ix), np.maximum.reduceat(b, ix)))
    return lv


def best_first(lv, q, pm, near):
    """Reads for best-first search to the oracle stop. Each pop expands one node: FAN child bounds (2 reads each);
    a popped page is selected (16 reads). Worst case O(pages * d) like the flat scan; the measured count is the claim."""
    top = len(lv) - 1
    reads, kept = 2 * lv[top][0].shape[0], near
    bt = bound(lv[top][0], lv[top][1], q)
    hp = [(-bt[i], top, i) for i in range(len(bt))]
    heapq.heapify(hp)
    while hp and kept < 1 - EPS - 1e-12:
        _, l, i = heapq.heappop(hp)
        if l == 0:
            kept += pm[i]
            reads += 16
            continue
        lo, hi = i * FAN, min((i + 1) * FAN, lv[l - 1][0].shape[0])
        b = bound(lv[l - 1][0][lo:hi], lv[l - 1][1][lo:hi], q)
        reads += 2 * (hi - lo)
        for j in range(hi - lo):
            heapq.heappush(hp, (-b[j], l - 1, lo + j))
    return reads


def check():
    rng = np.random.default_rng(7)
    k = rng.standard_normal((4096, HD))
    q = rng.standard_normal(HD)
    mn, mx = minmax(k, 16)
    sc = (k @ q).reshape(-1, 16).max(1)
    sound = bool(np.all(bound(mn, mx, q) >= sc - 1e-9))
    lv = tree(mn, mx)
    mono = all(np.all(np.repeat(bound(*lv[l], q), FAN)[: lv[l - 1][0].shape[0]] >= bound(*lv[l - 1], q) - 1e-9)
               for l in range(1, len(lv)))
    k2 = k.copy()
    k2[37 * 16 + 5] = 3.0 * q
    mn2, mx2 = minmax(k2, 16)
    needle = int(np.argmax(bound(mn2, mx2, q))) == 37
    e = np.exp(S * (k2 @ q) - (S * (k2 @ q)).max())
    pm = (e / e.sum()).reshape(-1, 16).sum(1)
    hreads = best_first(tree(mn2, mx2), q, pm, 0.0)
    flat = 2 * pm.shape[0] + 16 * need(pm, np.argsort(-bound(mn2, mx2, q)), 0.0)
    print(f"soundness (bound >= page max, 256 pages) {sound}; tree bounds monotone {mono}; planted needle ranked first {needle}")
    print(f"planted needle: best-first reads {hreads}, flat reads {flat} of 4096 keys")
    ok = sound and mono and needle and hreads < flat
    print("CHECK PASS" if ok else "CHECK FAIL")
    return 0 if ok else 1


def slope(ts, ys):
    return float(np.polyfit(np.log(ts), np.log(ys), 1)[0])


def full():
    t0 = time.time()
    cs = np.fromfile(os.path.join(D, "cs.bin"), dtype="<f8").reshape(TN, H2, 2)
    shp = (NL, NH, len(QT), NQ)
    arms = ("A", "A64", "B", "OP")
    pages = {a: np.zeros(shp) for a in arms}
    reads = {a: np.zeros(shp) for a in arms + ("H",)}
    keps = np.zeros(shp)
    nfar = np.zeros(shp)
    fixed = {b: np.zeros(shp, dtype=bool) for b in BUDGETS}
    for l in range(NL):
        q = rotate(load(f"q{l}.bin", NH, TN), cs)
        k = rotate(load(f"k{l}.bin", NKV, TN), cs)
        for ti, t in enumerate(QT):
            for qi in range(NQ):
                p = t - NQ + qi
                far = p + 1 - W - NSINK
                n16, n64 = far // 16, far // 64
                for g in range(NKV):
                    kf = k[NSINK: NSINK + n16 * 16, g]
                    mn, mx = minmax(kf, 16)
                    mn64, mx64 = minmax(k[NSINK: NSINK + n64 * 64, g], 64)
                    lv = tree(mn, mx)
                    for h in range(g * GQA, (g + 1) * GQA):
                        qq = q[p, h]
                        sc = S * k[: p + 1, g] @ qq
                        e = np.exp(sc - sc.max())
                        pr = e / e.sum()
                        keps[l, h, ti, qi] = np.searchsorted(np.cumsum(np.sort(pr)[::-1]), 1 - EPS) + 1
                        pm = pr[NSINK: NSINK + n16 * 16].reshape(n16, 16).sum(1)
                        pm64 = pr[NSINK: NSINK + n64 * 64].reshape(n64, 64).sum(1)
                        near, near64 = 1 - pm.sum(), 1 - pm64.sum()
                        o = np.argsort(-bound(mn, mx, qq), kind="stable")
                        ob = np.argsort(-bound(mn[:, SLOW], mx[:, SLOW], qq[SLOW]), kind="stable")
                        o64 = np.argsort(-bound(mn64, mx64, qq), kind="stable")
                        op = np.argsort(-pm, kind="stable")
                        nn, nn64 = p + 1 - n16 * 16, p + 1 - n64 * 64
                        pages["A"][l, h, ti, qi] = need(pm, o, near)
                        pages["B"][l, h, ti, qi] = need(pm, ob, near)
                        pages["A64"][l, h, ti, qi] = need(pm64, o64, near64)
                        pages["OP"][l, h, ti, qi] = need(pm, op, near)
                        reads["A"][l, h, ti, qi] = nn + 2 * n16 + 16 * pages["A"][l, h, ti, qi]
                        reads["B"][l, h, ti, qi] = nn + n16 + 16 * pages["B"][l, h, ti, qi]
                        reads["A64"][l, h, ti, qi] = nn64 + 2 * n64 + 64 * pages["A64"][l, h, ti, qi]
                        reads["OP"][l, h, ti, qi] = nn + 16 * pages["OP"][l, h, ti, qi]
                        reads["H"][l, h, ti, qi] = nn + best_first(lv, qq, pm, near)
                        nfar[l, h, ti, qi] = n16 * 16
                        for b in BUDGETS:
                            fixed[b][l, h, ti, qi] = near + pm[o[: b // 16]].sum() >= 1 - EPS - 1e-12
        print(f"layer {l} {time.time() - t0:.0f}s", file=sys.stderr)

    med = lambda a: float(np.median(a))
    print(f"dump {D}; near = {NSINK} sinks + last {W} + partial page; eps = {EPS:.5f} (KL 0.05); oracle stop")
    print("\n== selected far tokens (median over layer, head, query) and oracle k_eps ==")
    for ti, t in enumerate(QT):
        row = "  ".join(f"{a} {med(pages[a][:, :, ti] * (64 if a == 'A64' else 16)):7.1f}" for a in arms)
        print(f"t={t:6d}: k_eps {med(keps[:, :, ti]):7.1f}  {row}")
    print("\n== total reads / t (median; p90) ==")
    for ti, t in enumerate(QT):
        row = "  ".join(f"{a} {med(reads[a][:, :, ti]) / t:.4f} ({np.quantile(reads[a][:, :, ti], .9) / t:.4f})"
                        for a in arms + ("H",))
        print(f"t={t:6d}: {row}")
    print("\n== fixed far budget, arm A: share of (layer, head, query) with KL <= 0.05 ==")
    for ti, t in enumerate(QT):
        print(f"t={t:6d}: " + "  ".join(f"{b}: {fixed[b][:, :, ti].mean():.3f}" for b in BUDGETS))
    sel = np.median(pages["A"], axis=3) * 16
    al = np.array([[slope(QT, np.maximum(sel[l, h], 1)) for h in range(NH)] for l in range(NL)])
    print(f"\nA selected-token growth exponent t=2048..32768: median {np.median(al):.3f}  p90 {np.quantile(al, .9):.3f}")
    print("A reads/t at 32768 by layer: " + " ".join(f"{med(reads['A'][l, :, 2]) / 32768:.3f}" for l in range(NL)))
    print("H reads/t at 32768 by layer: " + " ".join(f"{med(reads['H'][l, :, 2]) / 32768:.3f}" for l in range(NL)))

    a32 = lambda a: med(pages[a][:, :, 2])
    v = [("PQ1", med(pages["A"][:, :, 2]) * 16 <= 1024, f"A median far tokens {a32('A') * 16:.0f} (<=1024)"),
         ("PQ2", med(reads["A"][:, :, 2]) / 32768 <= 0.20, f"A median reads/t {med(reads['A'][:, :, 2]) / 32768:.4f} (<=0.20)"),
         ("PQ3", a32("B") <= a32("A"), f"B median pages {a32('B'):.1f} vs A {a32('A'):.1f}"),
         ("PQ4", fixed[2048][:, :, 2].mean() >= 0.9, f"fixed 2048 share {fixed[2048][:, :, 2].mean():.3f} (>=0.9)"),
         ("PQ5", np.median(al) <= 0.5, f"A growth exponent median {np.median(al):.3f} (<=0.5)"),
         ("PQ6", med(reads["H"][:, :, 2]) / 32768 <= 0.10, f"H median reads/t {med(reads['H'][:, :, 2]) / 32768:.4f} (<=0.10)")]
    print()
    for name, ok, msg in v:
        print(f"{name} {'HELD' if ok else 'REFUTED'}: {msg}")
    print(f"total {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    full()
