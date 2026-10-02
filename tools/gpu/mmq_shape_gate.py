#!/usr/bin/env python3
"""APEIRON-GPU-SHAPES gate: the matched-shape MMQ ratio, per shape, with a control.

The ported batch-tiled DP4A GEMM (`examples/simt_batch.rs`, `gemv_dp4a_batch_module`) is run at the
exact `(M, K, tokens)` rows that `tools/gpu/cuda_q8_gate.py` measures llama.cpp's CUDA MMQ Q8
prefill at, so the comparison is like-for-like (not batch-1 decode vs batched prefill).

Stages:
  cuda      build+run `tools/gpu/cuda_q8_bench.cpp` (`--all-shapes --reps 5 --warmup 2 --json`);
            capture per-shape device-median GMAC/s.
  apeiron   run `cargo run --release --features gpu --example simt_batch -- --check --bench --json`;
            capture per-shape device-timestamp GMAC/s (and its exact CPU-fold parity).
  compare   per-shape ratio = cuda_gmac_s / apeiron_gmac_s.
  control   the ratio must VARY with shape (the instrument sees a per-shape effect), otherwise the
            gate is red; a seeded constant-ratio series must be rejected (self-test).

Every number printed names the command that produced it. Exit 0 only when parity holds, the
variation control is green, and its seeded self-test reddens. Budget <= 540 s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA = Path("/home/socol/Workspace/llama.cpp")
BUILD = LLAMA / "build-cuda/bin"
SOURCE = ROOT / "tools/gpu/cuda_q8_bench.cpp"
APEIRON_BIN = ROOT / "target" / "release" / "examples" / "simt_batch"
DEADLINE = time.monotonic() + 535.0

# The magnitude the ratio must move across shapes before the shape dimension is judged to be a real
# degree of freedom the instrument can see (a stuck/flat comparator would give spread ~1.0).
MIN_RATIO_SPREAD = 1.20

# The gate's per-shape control is that the CUDA/Apeiron ratio is not a single number. Constant
# within this tolerance = the instrument sees no difference.
FLAT_TOL = 1e-6


def remaining() -> float:
    left = DEADLINE - time.monotonic()
    if left <= 0:
        raise RuntimeError("mmq_shape_gate exceeded its 540-second budget")
    return left


def run(argv: list[str], *, cwd: Path = ROOT, env=None) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(argv), flush=True)
    cp = subprocess.run(
        argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=remaining(), check=False, env=env,
    )
    return cp


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio_variation_control(ratios: list[float]) -> list[str]:
    """Control: the CUDA/Apeiron ratio must vary with shape.

    A flat ratio (max/min within FLAT_TOL) means the comparison carries no shape information — a
    stuck instrument, an uncoupled shape list, or a selector that always picked one row. Require a
    spread of at least MIN_RATIO_SPREAD as well, so the shape dimension is a measurable effect and
    not numerical noise.
    """
    problems: list[str] = []
    if len(ratios) < 2:
        return ["ratio_variation_control: fewer than 2 shape rows"]
    lo, hi = min(ratios), max(ratios)
    if lo <= 0:
        problems.append(f"ratio_variation_control: non-positive ratio {lo}")
        return problems
    spread = hi / lo
    if spread < 1.0 + FLAT_TOL:
        problems.append(
            f"ratio_variation_control: ratio is flat ({lo:.6f}..{hi:.6f}); the instrument sees no "
            "per-shape difference"
        )
    elif spread < MIN_RATIO_SPREAD:
        problems.append(
            f"ratio_variation_control: ratio spread {spread:.4f}x < {MIN_RATIO_SPREAD}x; the shape "
            "dimension is not a measurable effect at this resolution"
        )
    return problems


def control_self_test() -> int:
    """A constant-ratio series must be rejected; a genuinely varying one accepted."""
    flat = [8.0] * 10
    if not ratio_variation_control(flat):
        raise RuntimeError("control self-test: flat ratio series survived")
    alive = [2.25, 4.43, 3.65, 5.87, 9.59, 14.76, 3.66, 5.14, 9.81, 15.87]
    if ratio_variation_control(alive):
        raise RuntimeError("control self-test: varying ratio series was rejected")
    print("control self-test PASS: flat series rejected, varying series accepted", flush=True)
    return 0


def build_and_run_cuda() -> dict[tuple[str, int], dict]:
    cxx = shutil.which("g++") or shutil.which("c++")
    if not cxx:
        raise RuntimeError("C++ compiler not found")
    if not (BUILD / "libggml-cuda.so").exists():
        raise RuntimeError(f"libggml-cuda.so not found at {BUILD}")
    with tempfile.TemporaryDirectory(prefix="apeiron-gpushapes-cuda-") as td:
        binary = Path(td) / "cuda_q8_bench"
        build = [
            cxx, "-std=c++17", "-O3", "-Wall", "-Wextra",
            f"-I{LLAMA / 'ggml/include'}", f"-I{LLAMA / 'ggml/src'}", "-I/opt/cuda/include",
            str(SOURCE), f"-L{BUILD}", f"-Wl,-rpath,{BUILD}", "-L/opt/cuda/lib64",
            "-Wl,-rpath,/opt/cuda/lib64", "-lggml", "-lggml-base", "-lggml-cuda", "-lggml-cpu",
            "-lcudart", "-o", str(binary),
        ]
        cp = run(build)
        if cp.returncode != 0:
            print(cp.stdout, flush=True)
            raise RuntimeError(f"cuda bench build failed ({cp.returncode})")
        print(f"source_sha256 tools/gpu/cuda_q8_bench.cpp = {sha256(SOURCE)}")
        bench = run([str(binary), "--all-shapes", "--reps", "5", "--warmup", "2", "--json"])
        print(bench.stdout, flush=True)
        if bench.returncode != 0:
            raise RuntimeError(f"cuda bench exited {bench.returncode}")
        line = [ln for ln in bench.stdout.splitlines() if ln.startswith('{"results":')][-1]
        data = json.loads(line)
        if len(data["results"]) != 10:
            raise RuntimeError(f"expected 10 CUDA rows, got {len(data['results'])}")
        return {(r["shape"], int(r["tokens"])): r for r in data["results"]}


def run_apeiron() -> dict[tuple[str, int], dict]:
    build = run(["cargo", "build", "--release", "--features", "gpu", "--example", "simt_batch"])
    if build.returncode != 0:
        print(build.stdout, flush=True)
        raise RuntimeError(f"simt_batch build failed ({build.returncode})")
    if not APEIRON_BIN.exists():
        raise RuntimeError(f"apeiron batch example not built at {APEIRON_BIN}")
    cp = run([str(APEIRON_BIN), "--check", "--bench", "--json"])
    print(cp.stdout, flush=True)
    if cp.returncode != 0:
        raise RuntimeError(f"simt_batch exited {cp.returncode} (parity or device failure)")
    lines = [ln for ln in cp.stdout.splitlines() if ln.startswith('{"apeiron":')]
    if not lines:
        raise RuntimeError("simt_batch emitted no JSON; device skipped?")
    data = json.loads(lines[-1])
    return {(r["shape"], int(r["tokens"])): r for r in data["apeiron"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-self-test", action="store_true")
    ap.add_argument(
        "--seed-flat",
        action="store_true",
        help="control-of-control: force every ratio to one value; the variation control must redden",
    )
    args = ap.parse_args()
    if args.control_self_test:
        return control_self_test()

    t0 = time.monotonic()
    print("=== APEIRON-GPU-SHAPES: matched-shape MMQ ratio gate ===", flush=True)
    # Prove the control can redden before trusting it green (seeded flat series must be rejected).
    control_self_test()

    cuda = build_and_run_cuda()
    apeiron = run_apeiron()

    keys = sorted(cuda.keys())
    if set(keys) != set(apeiron.keys()):
        raise RuntimeError(
            f"shape sets differ: cuda={sorted(cuda)} apeiron={sorted(apeiron)}"
        )

    print("\n=== Per-shape comparison: llama.cpp CUDA MMQ Q8_0 vs ported apeiron DP4A batch GEMM ===")
    print("Both at the same (M, K, tokens); GMAC/s = M*tokens*K / device_median_ns.")
    print("| shape | M | K | tokens | MACs | apeiron dev_ns | apeiron GMAC/s | CUDA dev_ns | CUDA GMAC/s | ratio CUDA/apeiron |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    ratios: list[float] = []
    worst = None
    best = None
    for key in keys:
        a, c = apeiron[key], cuda[key]
        ratio = c["gmac_s"] / a["gmac_s"]
        ratios.append(ratio)
        macs = a["m"] * a["tokens"] * a["k"]
        print(
            f"| {a['shape']:<11} | {a['m']:>5} | {a['k']:>5} | {a['tokens']:>6} | {macs:>12} | "
            f"{a['dev_ns']:>14.0f} | {a['gmac_s']:>14.2f} | {c['dev_med_ns']:>11.0f} | "
            f"{c['gmac_s']:>11.2f} | {ratio:>17.2f}x |"
        )
        if worst is None or ratio > worst[0]:
            worst = (ratio, a["shape"], a["tokens"])
        if best is None or ratio < best[0]:
            best = (ratio, a["shape"], a["tokens"])

    a_all = [apeiron[k]["gmac_s"] for k in keys]
    c_all = [cuda[k]["gmac_s"] for k in keys]
    print(
        f"\napeiron GMAC/s range {min(a_all):.2f}..{max(a_all):.2f} "
        f"(spread {max(a_all)/min(a_all):.2f}x across shapes)"
    )
    print(
        f"CUDA GMAC/s range {min(c_all):.2f}..{max(c_all):.2f} "
        f"(spread {max(c_all)/min(c_all):.2f}x across shapes)"
    )
    print(
        f"ratio range {min(ratios):.2f}x..{max(ratios):.2f}x "
        f"(best {best[1]} tokens={best[2]}, worst {worst[1]} tokens={worst[2]})"
    )

    problems = ratio_variation_control(ratios)
    if args.seed_flat:
        print("[--seed-flat] forcing a flat ratio series (control-of-control)")
        ratios = [ratios[0]] * len(ratios)
        problems = ratio_variation_control(ratios)
    print("\n--- Controls ---")
    if problems:
        for p in problems:
            print(f"CONTROL FAIL: {p}")
    else:
        print(
            f"CONTROL PASS: ratio varies with shape, spread {max(ratios)/min(ratios):.2f}x "
            f">= {MIN_RATIO_SPREAD}x"
        )
    for k in keys:
        if apeiron[k]["max_rel_diff"] > 1e-2:
            print(f"PARITY FAIL: {apeiron[k]}")
            problems.append(f"parity {k}")

    if problems:
        print("\nGATE RED")
        return 1
    print("\nGATE GREEN: matched-shape table produced; parity exact; ratio-shape variation control green")
    print(f"elapsed {time.monotonic() - t0:.2f} s (budget <= 540 s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"mmq_shape_gate: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
