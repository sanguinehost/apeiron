#!/usr/bin/env python3
"""Join two `nq_relh.py --csv` files and print the per-row and aggregate distortion ratios.

Each row is a `(layer, kind)` linear; the ratio is `norm(candidate) / norm(reference)`, so `< 1`
is the candidate's lower reconstruction error. The aggregate is the geometric mean of the norm
ratios, which is the ratio of the two errors on a log scale; the per-row spread is printed beside
it so a mean cannot hide a row that reverses.

Usage:
  nq_compare.py REF.csv CAND.csv [--label REF=CAND ...]
"""

import csv
import math
import sys

FIELDS = ("layer", "kind", "out", "inn")


def read(path, name):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in FIELDS)
            e = row.get(name + "_energy", "")
            if e:
                rows[key] = float(e)
    if not rows:
        raise SystemExit(f"{path}: no `{name}_energy` column with values")
    return rows


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    ref_path, cand_path = sys.argv[1], sys.argv[2]
    pairs = [a.split("=", 1) for a in sys.argv[3:] if "=" in a]
    if not pairs:
        # Default: the file stems are the two names `nq_relh.py` wrote.
        def stem(p):
            return p.rsplit("/", 1)[-1].removeprefix("relh-").removesuffix(".csv")

        pairs = [(stem(ref_path), stem(cand_path))]
    ref_name, cand_name = pairs[0]
    ref = read(ref_path, ref_name)
    cand = read(cand_path, cand_name)
    keys = sorted(set(ref) & set(cand), key=lambda k: (int(k[0]), k[1]))
    if not keys:
        raise SystemExit("no rows in common")
    print(
        "layer kind        ref_norm  cand_norm   norm_ratio  energy_ratio"
    )
    ratios = []
    for key in keys:
        e_ref, e_cand = ref[key], cand[key]
        n_ref, n_cand = math.sqrt(e_ref), math.sqrt(e_cand)
        ratios.append(n_cand / n_ref)
        print(
            "%5s %-10s %9.4f%% %9.4f%% %11.3f %12.3f"
            % (key[0], key[1], n_ref * 100, n_cand * 100, n_cand / n_ref, e_cand / e_ref)
        )
    gmean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    better = sum(1 for r in ratios if r < 1.0)
    print()
    print(
        "rows %d  geometric mean norm ratio %.3f  min %.3f  max %.3f  candidate lower on %d"
        % (len(ratios), gmean, min(ratios), max(ratios), better)
    )


if __name__ == "__main__":
    main()
