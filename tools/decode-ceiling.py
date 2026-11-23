#!/usr/bin/env python3
"""Refuse a decode tok/s target that is above the machine's two bounds.

A decode claim is bounded twice: by bytes read per token over the measured bandwidth, and by
issued operations per token over the measured issue rate. This tool computes both from the pack
geometry and exits 1 when the target is above either, printing the budget the target needs. It is
the executable form of the `## Ceilings` rule in AGENTS.md: a target is quoted with its margin, and
a target above the ceiling is refused before a kernel is written.

Usage:

    python3 tools/decode-ceiling.py --bytes 1637275152 --terms 6945767424 --bw 28.42 \
        --ops-per-term 0.31 --ips 7.1 --target 20
    # bandwidth 17.36 tok/s, issue 3.29 tok/s, target 20.00 refused (exit 1)
    #   bytes budget at 20 tok/s: 1,421,000,000 B (have 1,637,275,152, over by 216,275,152)

`--bw` is GB/s and `--ips` is 1e9 instructions/s, both measured, never the spec peak. `--terms` is
the number of sign terms the projection evaluates per token (one per fitted weight when the rank
is near-full); `--ops-per-term` is counted from the kernel or its assembly. Omit `--ips` to check
the byte bound alone.
"""

import argparse
import sys


def gi(n: float) -> str:
    """A byte or count in units of 1e9, with the exact integer beside it when it is one."""
    return f"{n / 1e9:.3f}e9"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=float, required=True, help="bytes read per decode token")
    ap.add_argument("--terms", type=float, default=0, help="sign terms per decode token")
    ap.add_argument("--bw", type=float, required=True, help="measured bandwidth, GB/s")
    ap.add_argument("--ops-per-term", type=float, default=0.0)
    ap.add_argument("--ips", type=float, default=0.0, help="measured issue rate, 1e9 ops/s")
    ap.add_argument("--cores", type=int, default=0, help="physical cores the kernel runs on")
    ap.add_argument("--ghz", type=float, default=0.0, help="sustained clock")
    ap.add_argument("--ipc", type=float, default=0.0, help="instructions per cycle per core")
    ap.add_argument("--target", type=float, required=True, help="tok/s to check")
    args = ap.parse_args()

    if args.ips > 0.0 and args.ipc > 0.0:
        ap.error("give --ips or --cores/--ghz/--ipc, not both")
    if args.ipc > 0.0:
        if args.cores <= 0 or args.ghz <= 0.0:
            ap.error("--ipc needs --cores and --ghz")
        args.ips = args.cores * args.ghz * args.ipc

    bw = args.bw * 1e9
    bw_ceiling = bw / args.bytes
    rows = [("bandwidth", bw_ceiling, f"{gi(args.bytes)} B/token at {args.bw:.2f} GB/s")]

    if args.ips > 0.0:
        if args.terms <= 0.0 or args.ops_per_term <= 0.0:
            ap.error("--ips needs --terms and --ops-per-term")
        issue_ceiling = args.ips * 1e9 / (args.terms * args.ops_per_term)
        rows.append(
            (
                "issue",
                issue_ceiling,
                f"{gi(args.terms)} terms x {args.ops_per_term:g} ops at {args.ips:.2f}e9 ops/s",
            )
        )

    print(f"bytes/token {args.bytes:,.0f} B, target {args.target:.2f} tok/s")
    for name, ceiling, why in rows:
        margin = ceiling / args.target
        print(f"  {name:<9} ceiling {ceiling:6.2f} tok/s  ({why}); margin {margin:.2f}x")

    binding = min(rows, key=lambda r: r[1])
    budget = bw / args.target
    if args.target > binding[1]:
        over = args.bytes - budget
        print(
            f"refuse: {args.target:.2f} tok/s is above the {binding[0]} ceiling "
            f"{binding[1]:.2f} tok/s"
        )
        print(
            f"  bytes budget at {args.target:.2f} tok/s: {budget:,.0f} B "
            f"(have {args.bytes:,.0f}, over by {over:,.0f})"
        )
        return 1
    print(f"admit: {args.target:.2f} tok/s is below the {binding[0]} ceiling {binding[1]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
