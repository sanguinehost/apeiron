#!/usr/bin/env python3
"""Three-way matched-shape ratio table: incumbent `q8_tile.comp` | ported DP4A batch | CUDA MMQ.

Runs the incumbent gate `tools/gpu/q8_shapes.py` (device-timestamp GMAC/s of the current SPIR-V
kernel) and the ported-kernel gate `tools/gpu/mmq_shape_gate.py` (which builds+runs the ported
`gemv_dp4a_batch_module` and llama.cpp CUDA MMQ at the same `(M, K, tokens)`), joins them on
`(shape, tokens)`, and prints:

  incumbent GMAC/s | ported GMAC/s | ratio ported/incumbent | CUDA GMAC/s | ratio CUDA/incumbent | CUDA/ported

Control: the ported/incumbent ratio must VARY with shape (spread >= MIN_RATIO_SPREAD); a flat ratio
means the join carries no per-shape information. `--control-self-test` seeds a flat series and
requires rejection.

Every number names the command that produced it. Budget <= 540 s.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIN_RATIO_SPREAD = 1.20
BUDGET_SECONDS = 540.0
DEADLINE = time.monotonic() + BUDGET_SECONDS - 5.0


def remaining() -> float:
    left = DEADLINE - time.monotonic()
    if left <= 0:
        raise RuntimeError("incumbent_ratio_table exceeded its 540-second budget")
    return left


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(argv), flush=True)
    return subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=remaining(), check=False)


def last_json_line(text: str, prefix: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    if not lines:
        raise RuntimeError(f"no line starting {prefix!r} in captured output")
    return json.loads(lines[-1])


def ratio_variation_control(ratios: list[float]) -> list[str]:
    if len(ratios) < 2:
        return ["ratio_variation_control: fewer than 2 shape rows"]
    lo, hi = min(ratios), max(ratios)
    if lo <= 0:
        return [f"ratio_variation_control: non-positive ratio {lo}"]
    spread = hi / lo
    if spread < MIN_RATIO_SPREAD:
        return [
            f"ratio_variation_control: ported/incumbent ratio spread {spread:.4f}x < "
            f"{MIN_RATIO_SPREAD}x; the shape dimension is not a measurable effect"
        ]
    return []


def control_self_test() -> int:
    if not ratio_variation_control([8.0] * 10):
        raise RuntimeError("control self-test: flat ratio series survived")
    if ratio_variation_control([13.6, 14.2, 14.0, 17.0, 19.1, 20.9]):
        raise RuntimeError("control self-test: varying ratio series was rejected")
    print("control self-test PASS: flat series rejected, varying series accepted", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9, help="incumbent gate repetitions (default: 9)")
    ap.add_argument("--control-self-test", action="store_true")
    ap.add_argument("--seed-flat", action="store_true",
                    help="control-of-control: force the inc/ported ratio flat; the control must redden")
    args = ap.parse_args()
    if args.control_self_test:
        return control_self_test()

    t0 = time.monotonic()
    print("=== three-way matched-shape table: incumbent | ported | CUDA ===", flush=True)
    control_self_test()

    inc_cp = run([sys.executable, str(ROOT / "tools/gpu/q8_shapes.py"),
                  "--reps", str(args.reps), "--json"])
    if inc_cp.returncode != 0:
        print(inc_cp.stdout, flush=True)
        raise RuntimeError(f"q8_shapes exited {inc_cp.returncode}")
    incumbent = {(r["shape"], int(r["tokens"])): r for r in last_json_line(inc_cp.stdout, '{"incumbent"')["incumbent"]}

    mm_cp = run([sys.executable, str(ROOT / "tools/gpu/mmq_shape_gate.py")])
    if mm_cp.returncode != 0:
        print(mm_cp.stdout, flush=True)
        raise RuntimeError(f"mmq_shape_gate exited {mm_cp.returncode}")
    cuda = {(r["shape"], int(r["tokens"])): r for r in last_json_line(mm_cp.stdout, '{"results"')["results"]}
    ported = {(r["shape"], int(r["tokens"])): r for r in last_json_line(mm_cp.stdout, '{"apeiron"')["apeiron"]}

    keys = sorted(incumbent)
    if set(keys) != set(ported) or set(keys) != set(cuda):
        raise RuntimeError(
            f"shape sets differ: incumbent={sorted(incumbent)} ported={sorted(ported)} cuda={sorted(cuda)}"
        )

    print("\n| shape | M | K | tokens | incumbent GMAC/s | ported GMAC/s | ported/incumbent | CUDA GMAC/s | CUDA/incumbent | CUDA/ported |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ratios = []
    for key in keys:
        i, p, c = incumbent[key], ported[key], cuda[key]
        r = p["gmac_s"] / i["gmac_s"]
        ratios.append(r)
        print(f"| {i['shape']:<11} | {i['m']:>5} | {i['k']:>5} | {i['tokens']:>6} | "
              f"{i['gmac_s']:>16.2f} | {p['gmac_s']:>13.2f} | {r:>16.2f}x | "
              f"{c['gmac_s']:>11.2f} | {c['gmac_s']/i['gmac_s']:>14.2f}x | {c['gmac_s']/p['gmac_s']:>11.2f}x |")

    print(f"\nported/incumbent range {min(ratios):.2f}x..{max(ratios):.2f}x "
          f"(spread {max(ratios)/min(ratios):.2f}x across shapes)")
    problems = ratio_variation_control(ratios)
    if args.seed_flat:
        print("[--seed-flat] forcing a flat ratio series (control-of-control)")
        problems = ratio_variation_control([ratios[0]] * len(ratios))
    print("\n--- Control ---")
    if problems:
        for p in problems:
            print(f"CONTROL FAIL: {p}")
        print("\nGATE RED")
        return 1
    print(f"CONTROL PASS: ported/incumbent ratio varies with shape, spread "
          f"{max(ratios)/min(ratios):.2f}x >= {MIN_RATIO_SPREAD}x")
    print(f"\nGATE GREEN: three-way table produced; elapsed {time.monotonic()-t0:.2f} s (budget <= 540 s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"incumbent_ratio_table: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
