#!/usr/bin/env python3
"""Device-primitive latency of the Apeiron Q8 SPIR-V kernel at real projection shapes.

Drives the W5-T1 timestamp instrumentation (``q8_driver.c`` via the compile path of
``q8_bench.py``) over the Qwen3.5-0.8B projection shapes registered in
``docs/evidence/gpu-prefill/comparator-shapes.md`` and prints, per shape,
device/submit/wait nanoseconds plus derived device rates.

This measures the compute-kernel primitive only: the GPU timestamp delta across one
dispatch. It excludes host transport, whole-model throughput and any llama.cpp
comparison.

Formulas (stated explicitly, per dispatch):
  weight_bytes = M * blocks * 32            # Q8_0 packed codes, 32 int8 per block
  MACs         = M * tokens * blocks * 32   # one MAC per (row, token, block, lane)
  GB/s         = weight_bytes / device_ns_med     # bytes/ns == GB/s
  GMAC/s       = MACs         / device_ns_med     # MACs/ns  == GMAC/s
  (multiply GMAC/s by 2 for a flop/s count.)

The script never modifies q8_bench.py, q8_bench_controls.py or q8_driver.c: it
imports q8_bench's compile/synthesis helpers and invokes the compiled driver.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_BUDGET_SECONDS = 540.0
BUDGET_GRACE_SECONDS = 5.0

# Real projection shapes (M = weight rows, blocks = K/32), from Qwen3.5-0.8B.
# attn_k and attn_v are the identical (K=1024, M=512) shape; one row covers both.
SHAPES: list[tuple[str, int, int]] = [
    ("attn_q", 4096, 32),
    ("attn_k_v", 512, 32),
    ("attn_output", 1024, 64),
    ("ffn_gate_up", 3584, 32),
    ("ffn_down", 1024, 112),
]
PROMPT_TOKENS = (200, 512)
# Fixed-(M, tokens) block-ordering control: same M=1024 rows, blocks 64 vs 112.
BLOCKS_CONTROL = ("attn_output", "ffn_down")


def load_bench():
    """Load tools/gpu/q8_bench.py by path so we reuse its compile path unchanged."""
    spec = importlib.util.spec_from_file_location("q8_bench", HERE / "q8_bench.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load q8_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def run_case(bench, driver_bin: Path, shader_spv: Path, scratch: Path,
             label: str, m: int, tokens: int, blocks: int, reps: int,
             deadline: float) -> dict[str, object]:
    left = remaining(deadline)
    if left <= 0:
        raise RuntimeError(f"q8_shapes exceeded its wall-clock budget before {label} tokens={tokens}")

    inp = scratch / f"{label}-{tokens}.input"
    out = scratch / f"{label}-{tokens}.output"
    ts = scratch / f"{label}-{tokens}.ts.json"
    inp.write_bytes(bench.synthesize_inputs(m, tokens, blocks))

    cmd = [
        str(driver_bin), str(shader_spv), str(m), str(tokens), str(blocks),
        str(inp), str(out), "--timestamps", str(ts), "--reps", str(reps),
    ]
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=left)
    if cp.returncode != 0:
        print(cp.stdout, flush=True)
        print(cp.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(
            f"q8_driver failed with code {cp.returncode} on {label} tokens={tokens}: {cp.stderr.strip()}"
        )
    if not ts.exists():
        raise RuntimeError(f"q8_driver did not emit timestamps file for {label} tokens={tokens}")

    data = json.loads(ts.read_text())
    reps_data = data["reps"]
    dev = [float(r["device_ns"]) for r in reps_data]
    sub = [int(r["submit_ns"]) for r in reps_data]
    wait = [int(r["wait_ns"]) for r in reps_data]

    weight_bytes = m * blocks * 32
    macs = m * tokens * blocks * 32
    dev_med = statistics.median(dev)
    return {
        "label": label,
        "m": m,
        "tokens": tokens,
        "blocks": blocks,
        "reps": len(reps_data),
        "device_ns": dev,
        "submit_ns": sub,
        "wait_ns": wait,
        "device_ns_min": min(dev),
        "device_ns_med": dev_med,
        "device_ns_max": max(dev),
        "submit_ns_med": statistics.median(sub),
        "wait_ns_med": statistics.median(wait),
        "weight_bytes": weight_bytes,
        "macs": macs,
        "gb_per_s": weight_bytes / dev_med,
        "gmac_per_s": macs / dev_med,
        "device_name": data["device_name"],
        "driver_version": data["driver_version"],
        "api_version": data["api_version"],
        "timestamp_period": float(data["timestamp_period"]),
        "timestamp_valid_bits": int(data["timestamp_valid_bits"]),
    }


def check_ordering(results: list[dict[str, object]]) -> list[str]:
    """Control: device_ns must grow with blocks (fixed M, tokens) and with tokens.

    A stuck or zero timer flattens device_ns and violates one of these orderings.
    """
    problems: list[str] = []
    by = {(str(r["label"]), int(r["tokens"])): float(r["device_ns_med"]) for r in results}

    for label, _m, _b in SHAPES:
        if (label, PROMPT_TOKENS[0]) not in by or (label, PROMPT_TOKENS[1]) not in by:
            continue
        small, large = by[(label, PROMPT_TOKENS[0])], by[(label, PROMPT_TOKENS[1])]
        if not large > small:
            problems.append(
                f"tokens ordering violated for {label}: tokens={PROMPT_TOKENS[0]} -> {small:.1f} ns "
                f"not < tokens={PROMPT_TOKENS[1]} -> {large:.1f} ns"
            )

    small_label, large_label = BLOCKS_CONTROL
    for tokens in PROMPT_TOKENS:
        if (small_label, tokens) not in by or (large_label, tokens) not in by:
            continue
        small, large = by[(small_label, tokens)], by[(large_label, tokens)]
        if not large > small:
            problems.append(
                f"blocks ordering violated at M=1024, tokens={tokens}: "
                f"{small_label} (blocks=64) -> {small:.1f} ns not < "
                f"{large_label} (blocks=112) -> {large:.1f} ns"
            )
    return problems


def control_self_test() -> int:
    """Seed an inverted ordering and require the checker to reject it."""
    good = []
    for label, m, blocks in SHAPES:
        for tokens in PROMPT_TOKENS:
            good.append({"label": label, "m": m, "tokens": tokens, "blocks": blocks,
                         "device_ns_med": float(blocks * tokens)})
    if check_ordering(good):
        raise RuntimeError("control self-test: monotone series was rejected")
    seeded = [dict(r) for r in good]
    for r in seeded:
        if r["label"] == "ffn_down" and r["tokens"] == PROMPT_TOKENS[1]:
            r["device_ns_med"] = 1.0  # zeroed/stuck timer
    problems = check_ordering(seeded)
    if not problems:
        raise RuntimeError("control self-test: inverted ordering survived")
    print(f"control self-test PASS: inverted series rejected ({problems[0]})", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apeiron Q8 primitive latency at real projection shapes")
    parser.add_argument("--reps", type=int, default=5, help="repetitions per shape (default: 5)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS,
                        help="wall-clock budget in seconds (default: 540)")
    parser.add_argument("--control-self-test", action="store_true",
                        help="only run the seeded control-checker self-test")
    args = parser.parse_args()

    start = time.monotonic()
    deadline = start + args.budget - BUDGET_GRACE_SECONDS

    if args.control_self_test:
        return control_self_test()

    bench = load_bench()
    glslc = bench.tool("glslc", "GLSLC", "glslc")
    spirv_val = bench.tool("spirv-val", "SPIRV_VAL", "spirv-val")
    cc = bench.tool("cc", "CC", "cc")

    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="apeiron-q8-shapes-") as td:
        scratch = Path(td)
        shader_spv = scratch / "q8_tile.spv"
        driver_bin = scratch / "q8_driver"
        bench.compile_shader(glslc, spirv_val, bench.SHADER, shader_spv, deadline=deadline)
        bench.compile_driver(cc, bench.DRIVER, driver_bin, deadline=deadline)

        for label, m, blocks in SHAPES:
            for tokens in PROMPT_TOKENS:
                results.append(run_case(bench, driver_bin, shader_spv, scratch,
                                        label, m, tokens, blocks, args.reps, deadline))

    elapsed = time.monotonic() - start
    ident = results[0]
    print("=" * 118)
    print("APEIRON Q8 SPIR-V PRIMITIVE LATENCY — REAL PROJECTION SHAPES")
    print("=" * 118)
    print(f"device        = {ident['device_name']} (driver={ident['driver_version']}, api={ident['api_version']})")
    print(f"timestampPeriod = {ident['timestamp_period']:.2f} ns   timestampValidBits = {ident['timestamp_valid_bits']}")
    print(f"reps          = {args.reps}   budget = {args.budget:.0f} s   elapsed = {elapsed:.2f} s")
    print("device_ns is the GPU timestamp delta across one dispatch; submit/wait are host-side.")
    print(f"GB/s = (M*blocks*32) / device_ns_med;  GMAC/s = (M*tokens*blocks*32) / device_ns_med")
    print("-" * 118)
    hdr = f"{'shape':<12}{'M':>6}{'tokens':>7}{'blocks':>7} | {'dev_min':>10}{'dev_med':>11}{'dev_max':>11} | {'sub_med':>9}{'wait_med':>10} | {'GB/s':>8}{'GMAC/s':>9}"
    print(hdr)
    print("-" * 118)
    for r in results:
        print(f"{r['label']:<12}{r['m']:>6}{r['tokens']:>7}{r['blocks']:>7} | "
              f"{r['device_ns_min']:>10.1f}{r['device_ns_med']:>11.1f}{r['device_ns_max']:>11.1f} | "
              f"{r['submit_ns_med']:>9.0f}{r['wait_ns_med']:>10.0f} | "
              f"{r['gb_per_s']:>8.2f}{r['gmac_per_s']:>9.2f}")

    failures = [r for r in results if not float(r["device_ns_med"]) > 0.0]
    if failures:
        for r in failures:
            print(f"FAIL: {r['label']} tokens={r['tokens']} device_ns_med={r['device_ns_med']} not positive")
        print("=" * 118)
        return 1

    problems = check_ordering(results)
    print("-" * 118)
    if problems:
        for p in problems:
            print(f"CONTROL FAIL: {p}")
        print("=" * 118)
        return 1
    print("CONTROL PASS: device_ns strictly grows with tokens (each shape) "
          "and with blocks (M=1024, fixed tokens)")
    print("=" * 118)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"q8_shapes: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
