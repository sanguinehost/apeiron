#!/usr/bin/env python3
"""Build, test, and gate matched llama.cpp CUDA MMQ Q8 prefill projection latency.

SCOPE: tools/gpu/cuda_q8_bench.cpp, tools/gpu/cuda_q8_gate.py.
REGISTER: measurement; matched CUDA measurement of llama.cpp Q8 prefill (MMQ int8 MMA)
          at Apeiron native-kernel shapes.
GATE: python3 tools/gpu/cuda_q8_gate.py; whole gate <= 540 s.
"""

from __future__ import annotations

import hashlib
import json
import os
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
DEADLINE = time.monotonic() + 535.0

# Apeiron native-kernel SPIR-V baseline from docs/evidence/gpu-prefill/primitive-latency.md
APEIRON_BASELINES = {
    ("attn_q", 200): {"dev_med_ns": 4887744, "gmac_s": 171.63},
    ("attn_q", 512): {"dev_med_ns": 12168704, "gmac_s": 176.48},
    ("attn_k_v", 200): {"dev_med_ns": 503072, "gmac_s": 208.43},
    ("attn_k_v", 512): {"dev_med_ns": 1223840, "gmac_s": 219.34},
    ("attn_output", 200): {"dev_med_ns": 1112128, "gmac_s": 377.14},
    ("attn_output", 512): {"dev_med_ns": 3045984, "gmac_s": 352.51},
    ("ffn_gate_up", 200): {"dev_med_ns": 4119136, "gmac_s": 178.19},
    ("ffn_gate_up", 512): {"dev_med_ns": 10348640, "gmac_s": 181.57},
    ("ffn_down", 200): {"dev_med_ns": 1449568, "gmac_s": 506.36},
    ("ffn_down", 512): {"dev_med_ns": 4180608, "gmac_s": 449.47},
}


def remaining() -> float:
    left = DEADLINE - time.monotonic()
    if left <= 0:
        raise RuntimeError("cuda_q8_gate exceeded its 540-second budget")
    return left


def run_cmd(
    argv: list[str],
    *,
    expected: int | list[int] = 0,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(argv), flush=True)
    expected_list = [expected] if isinstance(expected, int) else expected
    cp = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=remaining(),
        check=False,
        env=env,
    )
    if cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n", flush=True)
    print(f"exit={cp.returncode}", flush=True)
    if cp.returncode not in expected_list:
        raise RuntimeError(
            f"command exit {cp.returncode}, expected {expected_list}: {argv[0]}"
        )
    return cp


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> int:
    t_start = time.monotonic()
    print("=== Apeiron CUDA MMQ Q8 Matched Measurement Gate ===", flush=True)

    cxx = shutil.which("g++") or shutil.which("c++")
    if not cxx:
        raise RuntimeError("C++ compiler not found")

    if not (BUILD / "libggml-cuda.so").exists():
        raise RuntimeError(f"libggml-cuda.so not found at {BUILD}")

    with tempfile.TemporaryDirectory(prefix="apeiron-cuda-q8-") as td:
        binary = Path(td) / "cuda_q8_bench"

        # Step 1: Build the benchmark binary
        build_argv = [
            cxx,
            "-std=c++17",
            "-O3",
            "-Wall",
            "-Wextra",
            f"-I{LLAMA / 'ggml/include'}",
            f"-I{LLAMA / 'ggml/src'}",
            "-I/opt/cuda/include",
            str(SOURCE),
            f"-L{BUILD}",
            f"-Wl,-rpath,{BUILD}",
            "-L/opt/cuda/lib64",
            "-Wl,-rpath,/opt/cuda/lib64",
            "-lggml",
            "-lggml-base",
            "-lggml-cuda",
            "-lggml-cpu",
            "-lcudart",
            "-o",
            str(binary),
        ]
        run_cmd(build_argv)
        print(f"source_sha256 tools/gpu/cuda_q8_bench.cpp = {file_sha256(SOURCE)}")

        # Step 2: Run positive benchmark across all 10 canonical shapes
        print("\n--- Running 10 Canonical Shapes on CUDA Backend (MMQ int8 path) ---", flush=True)
        bench_cp = run_cmd([str(binary), "--all-shapes", "--reps", "5", "--warmup", "2", "--json"])
        
        # Parse JSON results
        json_line = [l for l in bench_cp.stdout.splitlines() if l.startswith('{"results":')][-1]
        data = json.loads(json_line)
        results = data["results"]
        if len(results) != 10:
            raise RuntimeError(f"Expected 10 shape results, got {len(results)}")

        res_map: dict[tuple[str, int], dict] = {}
        for r in results:
            key = (r["shape"], r["tokens"])
            res_map[key] = r
            if r["dev_med_ns"] <= 0 or r["gmac_s"] <= 0:
                raise RuntimeError(f"Invalid non-positive measurement: {r}")

        # Step 3: Verify Scaling Controls
        print("\n--- Verifying Scaling Controls ---", flush=True)
        # 3a. Token scaling: dev_med_ns(512) > dev_med_ns(200) for all shapes
        unique_shapes = sorted(list(set(r["shape"] for r in results)))
        for s in unique_shapes:
            t200 = res_map[(s, 200)]["dev_med_ns"]
            t512 = res_map[(s, 512)]["dev_med_ns"]
            if t512 <= t200:
                raise RuntimeError(f"Scaling anomaly: {s} 512 tokens ({t512} ns) <= 200 tokens ({t200} ns)")
            print(f"control token_scaling {s}: t200={t200:.0f}ns < t512={t512:.0f}ns PASS")

        # 3b. Block scaling: M=1024, tokens=200: attn_output (blocks=64) vs ffn_down (blocks=112)
        ns_b64 = res_map[("attn_output", 200)]["dev_med_ns"]
        ns_b112 = res_map[("ffn_down", 200)]["dev_med_ns"]
        if ns_b112 <= ns_b64:
            raise RuntimeError(f"Block scaling anomaly: blocks=112 ({ns_b112} ns) <= blocks=64 ({ns_b64} ns)")
        print(f"control block_scaling: blocks=64 ({ns_b64:.0f}ns) < blocks=112 ({ns_b112:.0f}ns) PASS")

        # Step 4: Correctness Check (CPU reference vs GPU MMQ output)
        print("\n--- Correctness Check: CPU Reference vs GPU MMQ Output ---", flush=True)
        run_cmd([str(binary), "--check-cpu"])
        print("correctness_baseline: PASS (CPU vs GPU output matches within tolerance)")

        # Negative Control 1: Seeded defect in output must be rejected
        print("\n--- Negative Control 1: Seeded Defect Detection ---", flush=True)
        run_cmd([str(binary), "--check-cpu", "--seed-defect"])
        print("negative_control_seeded_defect: RED/PASS (corrupted output caught and rejected)")

        # Negative Control 2: Shape mismatch rejection
        print("\n--- Negative Control 2: Shape Mismatch Detection ---", flush=True)
        run_cmd([str(binary), "--shape-mismatch-control"])
        print("negative_control_shape_mismatch: RED/PASS (mismatched contraction dimension rejected)")

        # Negative Control 3: Direct Q8_1 activation tensor on CUDA
        # In llama.cpp ggml-cuda, MMQ requires F32 activations and asserts src1->type == GGML_TYPE_F32.
        # Passing a direct Q8_1 tensor to ggml_mul_mat on CUDA triggers GGML_ASSERT and aborts.
        print("\n--- Negative Control 3: Direct Q8_1 on CUDA Rejection ---", flush=True)
        # Expected exit is non-zero (either 134/SIGABRT or exit 2)
        cp_q8_1 = subprocess.run(
            [str(binary), "--direct-q8-1-control"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=remaining(),
            check=False,
        )
        print(cp_q8_1.stdout, end="" if cp_q8_1.stdout.endswith("\n") else "\n", flush=True)
        print(f"exit={cp_q8_1.returncode}", flush=True)
        # It must either abort due to GGML_ASSERT or exit non-zero
        if cp_q8_1.returncode == 0:
            raise RuntimeError("direct Q8_1 on CUDA unexpectedly succeeded without assertion")
        print("negative_control_direct_q8_1: RED/PASS (direct Q8_1 activation rejected by CUDA backend as expected)")

        # Step 5: Print Comparative Table vs Apeiron Native SPIR-V Kernel
        print("\n=== Matched Comparison: llama.cpp CUDA MMQ vs Apeiron SPIR-V Primitive ===")
        print("| shape | M | tokens | blocks | MACs | Apeiron dev_ns | Apeiron GMAC/s | CUDA dev_ns | CUDA GMAC/s | Ratio (CUDA/Apeiron) |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            key = (r["shape"], r["tokens"])
            ap = APEIRON_BASELINES[key]
            ratio = r["gmac_s"] / ap["gmac_s"]
            print(
                f"| {r['shape']:<11} | {r['m']:4d} | {r['tokens']:6d} | {r['blocks']:6d} | {r['macs']:12d} | "
                f"{ap['dev_med_ns']:14d} | {ap['gmac_s']:14.2f} | {r['dev_med_ns']:11.0f} | {r['gmac_s']:11.2f} | "
                f"{ratio:20.2f}x |"
            )

    elapsed = time.monotonic() - t_start
    print(f"\ncuda_q8_gate completed successfully in {elapsed:.2f} s (budget <= 540 s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"cuda_q8_gate: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
