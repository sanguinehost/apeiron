#!/usr/bin/env python3
"""Negative controls and restoration verification for Apeiron Q8 Vulkan benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SWARM_TMP = Path("/home/socol/oc-tmp/gpu-swarm")


def main() -> int:
    parser = argparse.ArgumentParser(description="Q8 Vulkan benchmark negative controls")
    parser.add_argument("--profile", choices=["synthetic", "quantizer"], default="synthetic")
    parser.add_argument("--fixture", type=Path, default=os.environ.get("APEIRON_NATIVE_Q8_FIXTURE"))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--deadline", type=float, default=None)
    args = parser.parse_args()

    deadline = args.deadline or (time.monotonic() + 535.0)
    SWARM_TMP.mkdir(parents=True, exist_ok=True)
    throwaway = SWARM_TMP / f"throwaway-w5t1-control-{os.getpid()}-{int(time.time() * 1000)}"

    try:
        # Create throwaway copy
        # Bounded copy of exactly the files the bench needs; never copy
        # target/, .git or .lake (a whole-repo cp -a is 51 GiB; failure register).
        needed = [
            "tools/__init__.py",
            "tools/gpu/__init__.py",
            "tools/gpu/q8_bench.py",
            "tools/gpu/q8_bench_controls.py",
            "tools/gpu/q8_gate.py",
            "tools/gpu/q8_driver.c",
            "src/spirv/q8_tile.comp",
        ]
        for rel in needed:
            src = ROOT / rel
            if not src.is_file():
                continue
            dst = throwaway / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        driver_c = throwaway / "tools/gpu/q8_driver.c"
        driver_src = driver_c.read_text()

        bench_cmd = [
            sys.executable,
            str(throwaway / "tools/gpu/q8_bench.py"),
            "--profile", args.profile,
            "--reps", str(args.reps),
            "--deadline", str(deadline),
        ]
        if args.fixture:
            bench_cmd.extend(["--fixture", str(args.fixture)])
        if args.timing_only:
            bench_cmd.append("--timing-only")

        # 1. Negative control A: Corrupt timestamp delta (force zero delta)
        needle_delta = "uint64_t delta_ticks = (t1 - t0) & mask;"
        if needle_delta not in driver_src:
            raise RuntimeError("anchor for zero timestamp delta corruption missing in driver source")
        mutated_src_a = driver_src.replace(needle_delta, "uint64_t delta_ticks = 0; /* SEEDED CORRUPTION */")
        driver_c.write_text(mutated_src_a)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("budget exceeded before running negative control A")

        cp_a = subprocess.run(bench_cmd, cwd=throwaway, capture_output=True, text=True, timeout=remaining)
        if cp_a.returncode == 0:
            raise RuntimeError("negative control A survived: zero timestamp delta was not rejected")
        combined_a = cp_a.stdout + "\n" + cp_a.stderr
        if "rejected zero timestamp delta" not in combined_a:
            raise RuntimeError(f"negative control A failed with unexpected diagnostic: {combined_a}")
        print(f"W5-T1 negative control red: rejected zero timestamp delta (exit={cp_a.returncode})", flush=True)

        # 2. Negative control B: Misread timestampValidBits (force 0)
        needle_bits = "chosen_timestamp_valid_bits = qp[q].timestampValidBits;"
        if needle_bits not in driver_src:
            raise RuntimeError("anchor for timestampValidBits corruption missing in driver source")
        mutated_src_b = driver_src.replace(needle_bits, "chosen_timestamp_valid_bits = 0; /* SEEDED CORRUPTION */")
        driver_c.write_text(mutated_src_b)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("budget exceeded before running negative control B")

        cp_b = subprocess.run(bench_cmd, cwd=throwaway, capture_output=True, text=True, timeout=remaining)
        if cp_b.returncode == 0:
            raise RuntimeError("negative control B survived: zero timestampValidBits was not rejected")
        combined_b = cp_b.stdout + "\n" + cp_b.stderr
        if "invalid timestampValidBits" not in combined_b:
            raise RuntimeError(f"negative control B failed with unexpected diagnostic: {combined_b}")
        print(f"W5-T1 negative control red: invalid timestampValidBits (0) (exit={cp_b.returncode})", flush=True)

    finally:
        if throwaway.exists():
            shutil.rmtree(throwaway, ignore_errors=True)

    # 3. Restoration check: Run bench on the UNCHANGED worktree root
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("budget exceeded before running restoration check")

    res_cmd = [
        sys.executable,
        str(ROOT / "tools/gpu/q8_bench.py"),
        "--profile", args.profile,
        "--reps", str(args.reps),
        "--deadline", str(deadline),
    ]
    if args.fixture:
        res_cmd.extend(["--fixture", str(args.fixture)])
    if args.timing_only:
        res_cmd.append("--timing-only")

    cp_res = subprocess.run(res_cmd, cwd=ROOT, text=True, timeout=remaining)
    if cp_res.returncode != 0:
        raise RuntimeError(f"restoration check failed with exit code {cp_res.returncode}")

    print("W5-T1 restoration green", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"q8_bench_controls: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
