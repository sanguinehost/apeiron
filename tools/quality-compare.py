#!/usr/bin/env python3
"""Compare the two 0-shot arms of the Q2 screen on identical items.

Reads the per-item JSONL that `examples/quality.rs` writes (one file per arm per task), pairs items
by `(task, item)`, and reports each task's primary metric and the six-task mean. The Q2 bar is
`mean_t(acc_nq(t)) >= 0.90 * mean_t(acc_q4(t))` over boolq, piqa, hellaswag, winogrande, arc_easy and
arc_challenge, each at its lm-eval primary metric.

**Unit of analysis.** The criterion is a mean over six task accuracies, so the six tasks are the
sampling units: `d(t) = acc_nq(t) - acc_q4(t)` is a paired difference per task (the same items, the
same tokenizer, the same continuation strings on both arms), and the interval is a t interval over
the six `d(t)`. Pooling items across tasks instead would treat a hellaswag item and a boolq item as
two draws from one population, which they are not: the tasks differ in difficulty by tens of
accuracy points, and the criterion means over tasks, not over items.

Verdict at the 95 % level:
- **establish** — the lower end of the interval clears `-0.10 * mean_t(acc_q4)`; exit 0.
- **refute** — the upper end is below the bar; exit 1.
- **undecided** — otherwise; exit 3. Add items (the full split) or rerun.

A 200-item-per-task screen is a probability sample (a centred systematic sample of each split), so
the interval is a statement about the split, not only about the sampled items. The printed design
line states the sample size, and the grade a receipt earns from a screen says so.

Usage:
    python3 tools/quality-compare.py --nq nq-boolq.jsonl nq-piqa.jsonl ... \
        --q4 q4-boolq.jsonl q4-piqa.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TOTALS = {
    "boolq": 3270,
    "piqa": 1838,
    "hellaswag": 10042,
    "winogrande": 1267,
    "arc_easy": 2376,
    "arc_challenge": 1172,
}
PRIMARY = {
    "boolq": "acc",
    "piqa": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
}

# Two-sided 97.5 % points of Student's t, indexed by degrees of freedom 1..30. Above 30 the normal
# point is within 4 % and the pooled error is smaller than the between-task spread.
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042,
}


def load(paths: list[Path]) -> dict[tuple[str, int], dict]:
    """(task, item) -> the row, refusing duplicates and rows of unknown tasks."""
    out: dict[tuple[str, int], dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("summary") or "correct" not in row:
                continue
            key = (row["task"], int(row["item"]))
            if key[0] not in PRIMARY:
                raise SystemExit(f"unknown task {key[0]!r} in {path}")
            if key in out:
                raise SystemExit(f"duplicate item {key} in {path}")
            out[key] = row
    return out


def is_correct(row: dict, metric: str) -> float:
    return float(bool(row["correct"][metric]))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nq", nargs="+", type=Path, required=True)
    parser.add_argument("--q4", nargs="+", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=None, help="items per task expected in both arms")
    args = parser.parse_args(argv)

    nq, q4 = load(args.nq), load(args.q4)
    if set(nq) != set(q4):
        only_nq = sorted(set(nq) - set(q4))[:5]
        only_q4 = sorted(set(q4) - set(nq))[:5]
        raise SystemExit(f"item sets differ: nq-only {only_nq}, q4-only {only_q4}")

    keys = sorted(nq)
    tasks = sorted({task for task, _ in keys})
    print(f"{'task':14s} {'n':>5s} {'ref':>4s} {'metric':>9s} {'nq':>7s} {'q4':>7s} {'ratio':>7s}")
    per_task: dict[str, tuple[int, int, float, float]] = {}
    for task in tasks:
        items = [key for key in keys if key[0] == task]
        metric = PRIMARY[task]
        a = [is_correct(nq[key], metric) for key in items]
        b = [is_correct(q4[key], metric) for key in items]
        refused = sum(
            1
            for key in items
            if nq[key].get("refused") is not None or q4[key].get("refused") is not None
        )
        per_task[task] = (len(items), refused, mean(a), mean(b))
        ratio = mean(a) / mean(b) if mean(b) else float("nan")
        print(
            f"{task:14s} {len(items):5d} {refused:4d} {metric:>9s} "
            f"{mean(a):7.4f} {mean(b):7.4f} {ratio:7.4f}"
        )

    acc_nq = [per_task[t][2] for t in tasks]
    acc_q4 = [per_task[t][3] for t in tasks]
    mean_nq, mean_q4 = mean(acc_nq), mean(acc_q4)
    ratio = mean_nq / mean_q4 if mean_q4 else float("nan")

    # Paired difference per task, t interval over the six tasks.
    diffs = [a - b for a, b in zip(acc_nq, acc_q4)]
    d_mean = mean(diffs)
    df = len(diffs) - 1
    tcrit = T975.get(df, 1.96)
    half = tcrit * sd(diffs) / math.sqrt(len(diffs))
    need = -0.10 * mean_q4
    if d_mean - half >= need:
        verdict, code = "establish", 0
    elif d_mean + half < need:
        verdict, code = "refute", 1
    else:
        verdict, code = "undecided", 3

    print()
    print(f"tasks            {len(tasks)}  ({', '.join(tasks)})")
    print(f"items            {len(keys)}")
    print(f"mean acc nq      {mean_nq:.6f}")
    print(f"mean acc q4      {mean_q4:.6f}")
    print(f"ratio            {ratio:.6f}   (bar 0.900000)")
    print(f"paired diff      {d_mean:+.6f} +- {half:.6f}  (95 % t interval, df {df}, t {tcrit})")
    print(f"task sd          {sd(diffs):.6f}")
    print(f"needed diff      {need:+.6f}   (= -0.10 x mean acc q4)")
    print(f"verdict          {verdict}")

    short = {t: per_task[t][0] for t in tasks if per_task[t][0] < TOTALS[t]}
    if short:
        sampled = sum(per_task[t][0] for t in tasks)
        full = sum(TOTALS[t] for t in tasks)
        print(
            f"design           sample: {sampled} of {full} split items "
            f"({', '.join(f'{t} {per_task[t][0]}/{TOTALS[t]}' for t in sorted(short))})"
        )
    else:
        print(f"design           full split: {sum(TOTALS[t] for t in tasks)} items, no sampling error")
    if args.expect is not None and len(keys) != args.expect * len(tasks):
        print(f"warning: {len(keys)} items, expected {args.expect} x {len(tasks)}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
