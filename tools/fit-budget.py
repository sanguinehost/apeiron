#!/usr/bin/env python3
"""Refuse a fit campaign whose projected wall exceeds the cap.

The unit is one outer iteration over **every** linear the campaign fits: sum the
per-linear iteration times, then multiply by T.  The 7B case that motivated this
gate: `down` at 1.1M ppm costs 94.06 s/iteration on an AVX-VNNI box; one layer's
seven kinds are 3.45x that (324 s); 196 linears are 28 x 324 = 9,072 s/iteration;
T=200 is 504 h, not the 18 h a per-layer figure suggests when it is mislabeled as
the model.

Exit 0 when the projection fits, 1 when it does not, 2 on bad arguments.
"""

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--iteration-seconds",
        type=float,
        required=True,
        help="sum of one outer iteration over every fitted linear, seconds",
    )
    p.add_argument("--iters", type=int, required=True, help="outer iterations T")
    p.add_argument(
        "--cap-hours", type=float, default=6.0, help="wall cap in hours (default 6)"
    )
    a = p.parse_args()
    if a.iteration_seconds <= 0 or a.iters <= 0 or a.cap_hours <= 0:
        print("fit-budget: every argument must be positive", file=sys.stderr)
        return 2
    wall_h = a.iteration_seconds * a.iters / 3600.0
    print(
        f"{a.iteration_seconds:.1f} s/iteration x {a.iters} = "
        f"{wall_h:.2f} h; cap {a.cap_hours:.2f} h"
    )
    if wall_h > a.cap_hours:
        print(f"refuse: {wall_h:.2f} h > {a.cap_hours:.2f} h", file=sys.stderr)
        return 1
    print(f"admit: {wall_h:.2f} h <= {a.cap_hours:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
