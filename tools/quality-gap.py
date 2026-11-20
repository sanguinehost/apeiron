#!/usr/bin/env python3
"""Decompose a zero-shot gap between two arms from per-item JSONL files.

Usage:
    python3 tools/quality-gap.py --dir <dir> [--items <item-dir>]

`<dir>` holds `nq-<task>.jsonl` and `q4-<task>.jsonl` as `examples/quality` writes them: one line
per item with `ll` (sum of ln p over the continuation), `n_tok`, `norm_len`, `gold` and `item`.
The primary metric per task is lm-eval's: `acc` on boolq and winogrande, `acc_norm` on the other
four. A candidate's score is `ll` (acc) or `ll / norm_len` (acc_norm); the gold margin is
`s[gold] - max(s[other])`, and the item is correct exactly when the margin is positive.

The output separates two causes that need different fixes:

- **attenuation**: `m_nq = a + b*m_q4 + e` with `b < 1` and residual `e` comparable to the signal.
  A positive scale on all logits would leave argmax unchanged, so `b < 1` with a large `e` is a
  direction change, not a temperature change.
- **bias**: the intercept `a`, and a label or position skew such as boolq's yes rate.

`--items` adds the boolq yes/no analysis, which needs the candidate strings; pass the directory of
the 200-item files. The scored rows are every second line, matching the `awk 'NR%2==1'` selection.
"""
import argparse
import json
import math
import os
import statistics as st

TASKS = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge"]
PRIMARY = {
    "boolq": "acc",
    "winogrande": "acc",
    "piqa": "acc_norm",
    "hellaswag": "acc_norm",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
}


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load(arm, task, directory):
    path = os.path.join(directory, f"{arm}-{task}.jsonl")
    out = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if PRIMARY[task] == "acc":
                s = [float(x) for x in d["ll"]]
            else:
                s = [float(x) / float(n) for x, n in zip(d["ll"], d["norm_len"])]
            g = d["gold"]
            other = max(s[j] for j in range(len(s)) if j != g)
            out[d["item"]] = {
                "m": s[g] - other,
                "gold": g,
                "k": len(s),
                "n": max(d["n_tok"]),
                "ll_tok": float(d["ll"][g]) / max(d["n_tok"][g], 1),
                "s": s,
            }
    return out


def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    vx = st.mean([(x - mx) ** 2 for x in xs])
    if not vx:
        return float("nan"), float("nan"), float("nan")
    cov = st.mean([(x - mx) * (y - my) for x, y in zip(xs, ys)])
    b = cov / vx
    a = my - b * mx
    sd = st.pstdev([y - (a + b * x) for x, y in zip(xs, ys)])
    return a, b, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--items")
    a = ap.parse_args()

    rows = []
    pooled = []
    for task in TASKS:
        q4 = load("q4", task, a.dir)
        nq = load("nq", task, a.dir)
        keys = sorted(set(q4) & set(nq))
        if len(keys) != len(q4) or len(keys) != len(nq):
            raise SystemExit(f"{task}: item sets differ ({len(q4)} q4, {len(nq)} nq, {len(keys)} paired)")
        mq = [q4[k]["m"] for k in keys]
        mn = [nq[k]["m"] for k in keys]
        acc_q = sum(1 for x in mq if x > 0) / len(keys)
        acc_n = sum(1 for x in mn if x > 0) / len(keys)
        both = sum(1 for x, y in zip(mq, mn) if x > 0 and y > 0)
        q4only = sum(1 for x, y in zip(mq, mn) if x > 0 and y <= 0)
        nqonly = sum(1 for x, y in zip(mq, mn) if x <= 0 and y > 0)
        neither = len(keys) - both - q4only - nqonly
        a0, b, sd = fit(mq, mn)
        pred = st.mean([phi((a0 + b * x) / sd) for x in mq]) if sd and sd == sd else float("nan")
        rows.append((task, len(keys), acc_n, acc_q, acc_n / acc_q if acc_q else float("nan"),
                     q4only, nqonly, both, neither, st.mean(mq), st.pstdev(mq), st.mean(mn),
                     st.pstdev(mn), b, sd, pred, a0, mq, mn, q4, nq, keys))
        pooled.extend((task, x, y) for x, y in zip(mq, mn))

    hdr = (f"{'task':14s} {'n':>4s} {'acc_nq':>7s} {'acc_q4':>7s} {'ratio':>6s} {'q4->nq':>6s} "
           f"{'nq->q4':>6s} {'both':>5s} {'neith':>5s} {'mean_q4':>8s} {'sd_q4':>7s} "
           f"{'mean_nq':>8s} {'sd_nq':>7s} {'slope':>6s} {'sd_res':>7s} {'pred':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:14s} {r[1]:4d} {r[2]:7.4f} {r[3]:7.4f} {r[4]:6.3f} {r[5]:6d} {r[6]:6d} "
              f"{r[7]:5d} {r[8]:5d} {r[9]:+8.3f} {r[10]:7.3f} {r[11]:+8.3f} {r[12]:7.3f} "
              f"{r[13]:6.2f} {r[14]:7.3f} {r[15]:6.3f}")

    mean_n = st.mean([r[2] for r in rows])
    mean_q = st.mean([r[3] for r in rows])
    print()
    print(f"{'task-mean':14s}    {mean_n:7.4f} {mean_q:7.4f} {mean_n / mean_q:6.3f}")
    flips_in = sum(r[5] for r in rows)
    flips_out = sum(r[6] for r in rows)
    print(f"net flips: {flips_in} Q4->NQ losses, {flips_out} NQ->Q4 gains, "
          f"{flips_in - flips_out} net of {sum(r[1] for r in rows)} items")

    print()
    print("Q4 margin in task sd units        n   NQ acc   Q4->NQ flips      NQ->Q4 flips")
    sds = {r[0]: r[10] for r in rows}
    bins = [(-99.0, -1.0), (-1.0, -0.3), (-0.3, 0.0), (0.0, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 99.0)]
    for lo, hi in bins:
        sel = [(t, x, y) for t, x, y in pooled if lo <= x / sds[t] < hi]
        if not sel:
            continue
        n = len(sel)
        acc = sum(1 for _, _, y in sel if y > 0) / n
        f_in = sum(1 for _, x, y in sel if x > 0 and y <= 0)
        f_out = sum(1 for _, x, y in sel if x <= 0 and y > 0)
        print(f"[{lo:+.1f}, {hi:+.1f})".ljust(34) + f"{n:4d}   {acc:6.4f}   "
              f"{f_in:5d} ({f_in / n:.2f})     {f_out:5d} ({f_out / n:.2f})")

    if a.items:
        rows_b = [json.loads(l) for i, l in enumerate(open(os.path.join(a.items, "boolq-200.jsonl")))
                  if i % 2 == 0]
        order = {tuple(c["continuation"].strip().lower() for c in d["candidates"]) for d in rows_b}
        if order == {("no", "yes")}:
            gold_yes = sum(1 for d in rows_b if d["gold"] == 1) / len(rows_b)
            print()
            print(f"boolq label: gold=yes {gold_yes:.3f}; always-yes {max(gold_yes, 1 - gold_yes):.3f}")
            for r in rows:
                if r[0] != "boolq":
                    continue
                q = [v["s"][1] - v["s"][0] for v in r[19].values()]
                n = [v["s"][1] - v["s"][0] for v in r[20].values()]
                print(f"  yes/no contrast nats   q4 mean {st.mean(q):+.3f} sd {st.pstdev(q):.3f} "
                      f"P(yes) {sum(1 for x in q if x > 0) / len(q):.3f}")
                print(f"                         nq mean {st.mean(n):+.3f} sd {st.pstdev(n):.3f} "
                      f"P(yes) {sum(1 for x in n if x > 0) / len(n):.3f}")

    print()
    print("gold ln p per token vs continuation length: slope (nats/token per token)")
    for r in rows:
        ins = []
        for arm, key in (("q4", 19), ("nq", 20)):
            xs = [v["n"] for v in r[key].values()]
            ys = [v["ll_tok"] for v in r[key].values()]
            _, b, _ = fit(xs, ys)
            ins.append(f"{arm} {b:+.4f}" if b == b else f"{arm} n/a")
        print(f"  {r[0]:14s} " + "   ".join(ins))

    print()
    print("accuracy by continuation-length tercile (q4 / nq)")
    for r in rows:
        keys = sorted(r[21], key=lambda k: r[19][k]["n"])
        cut = len(keys) // 3
        parts = []
        for name, sl in (("short", keys[:cut]), ("mid", keys[cut:2 * cut]), ("long", keys[2 * cut:])):
            aq = sum(1 for k in sl if r[19][k]["m"] > 0) / len(sl)
            an = sum(1 for k in sl if r[20][k]["m"] > 0) / len(sl)
            parts.append(f"{name} n={len(sl)} len~{st.mean([r[19][k]['n'] for k in sl]):.0f} {aq:.2f}/{an:.2f}")
        print(f"  {r[0]:14s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
