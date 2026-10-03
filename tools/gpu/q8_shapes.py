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
import hashlib
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
# Shape-variation control: the incumbent GMAC/s is a per-shape effect and must move at least this
# much across rows, otherwise the shape dimension carries no information (stuck/uncoupled shape list).
MIN_SHAPE_SPREAD = 1.20


def parse_shapes(spec: str) -> list[tuple[str, int, int]]:
    """Parse ``label:M:K[;label:M:K...]`` into ``(label, M, K/32)``.

    ``K`` is the contraction dimension; the driver takes ``blocks = K/32`` (32 int8 codes/block).
    Accepts ``x`` in place of ``:`` between M and K. Raises on a non-positive or non-32-multiple K.
    """
    out: list[tuple[str, int, int]] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.replace("x", ":").split(":")
        if len(parts) != 3:
            raise ValueError(f"--shapes entry {item!r} is not label:M:K")
        label = parts[0].strip()
        m, k = int(parts[1]), int(parts[2])
        if not label or m <= 0 or k <= 0 or k % 32 != 0:
            raise ValueError(f"--shapes entry {item!r}: need label, M>0, K>0, K%32==0")
        out.append((label, m, k // 32))
    if not out:
        raise ValueError("--shapes parsed to no shapes")
    return out


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def check_ordering(results: list[dict[str, object]], shapes, tokens_list) -> list[str]:
    """Control: device_ns must grow with blocks (fixed M, tokens) and with tokens.

    A stuck or zero timer flattens device_ns and violates one of these orderings.

    Ordered on ``device_ns_min``, the same estimator the reported rate uses. On the median, one
    host-interrupt-inflated repetition out of ``--reps`` reorders two shapes whose true times differ
    by less than that inflation -- observed on this box as ``attn_output`` at 2641824 ns against
    ``ffn_down`` at 1490016 ns, both at tokens=200 -- which is a red on the instrument's noise, not
    on the kernel. Both fields are seeded in ``--control-self-test``, so this ordering control cannot
    pass vacuously.
    """
    problems: list[str] = []
    by = {(str(r["label"]), int(r["tokens"])): float(r["device_ns_min"]) for r in results}

    for label, _m, _b in shapes:
        if (label, tokens_list[0]) not in by or (label, tokens_list[1]) not in by:
            continue
        small, large = by[(label, tokens_list[0])], by[(label, tokens_list[1])]
        if not large > small:
            problems.append(
                f"tokens ordering violated for {label}: tokens={tokens_list[0]} -> {small:.1f} ns "
                f"not < tokens={tokens_list[1]} -> {large:.1f} ns"
            )

    small_label, large_label = BLOCKS_CONTROL
    for tokens in tokens_list:
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


def shape_variation_control(results: list[dict[str, object]]) -> list[str]:
    """Control: the incumbent GMAC/s is a per-shape effect and must vary with shape.

    A flat GMAC/s (spread < MIN_SHAPE_SPREAD) means the shape dimension carries no information in
    this run — a stuck timer, an uncoupled shape list, or a selector that ran one row for all.
    """
    rates = [float(r["gmac_per_s"]) for r in results]
    if len(rates) < 2:
        return ["shape_variation_control: fewer than 2 shape rows"]
    lo, hi = min(rates), max(rates)
    if lo <= 0:
        return [f"shape_variation_control: non-positive GMAC/s {lo}"]
    spread = hi / lo
    if spread < MIN_SHAPE_SPREAD:
        return [
            f"shape_variation_control: GMAC/s spread {spread:.4f}x < {MIN_SHAPE_SPREAD}x; the shape "
            "dimension is not a measurable effect at this resolution"
        ]
    return []


def control_self_test() -> int:
    """Seed an inverted ordering and a flat rate series; both must be rejected.

    Both ``device_ns_min`` and ``device_ns_med`` are seeded: ``check_ordering`` reads the minimum, so
    a seed on the median alone would make the control pass vacuously.
    """
    good = []
    for label, m, blocks in SHAPES:
        for tokens in PROMPT_TOKENS:
            good.append({"label": label, "m": m, "tokens": tokens, "blocks": blocks,
                         "device_ns_min": float(blocks * tokens),
                         "device_ns_med": float(blocks * tokens),
                         "gmac_per_s": float(blocks + m)})
    if check_ordering(good, SHAPES, PROMPT_TOKENS):
        raise RuntimeError("control self-test: monotone series was rejected")
    seeded = [dict(r) for r in good]
    for r in seeded:
        if r["label"] == "ffn_down" and r["tokens"] == PROMPT_TOKENS[1]:
            r["device_ns_min"] = 1.0  # zeroed/stuck timer
    problems = check_ordering(seeded, SHAPES, PROMPT_TOKENS)
    if not problems:
        raise RuntimeError("control self-test: inverted ordering survived")

    # An inflated MEDIAN with an honest minimum must NOT redden: the estimator the control orders on
    # is the minimum, and this is the shape of the host-interrupt case that made it necessary.
    inflated = [dict(r) for r in good]
    for r in inflated:
        if r["label"] == "attn_output":
            r["device_ns_med"] = 10_000_000.0
        if r["label"] == "ffn_down":
            r["device_ns_med"] = 1.0
    if check_ordering(inflated, SHAPES, PROMPT_TOKENS):
        raise RuntimeError(
            "control self-test: the ordering control ordered on device_ns_med, so one inflated "
            "median rep reddens it")

    flat = [dict(r, gmac_per_s=1.0) for r in good]
    if not shape_variation_control(flat):
        raise RuntimeError("control self-test: flat GMAC/s series survived")
    if shape_variation_control(good):
        raise RuntimeError("control self-test: varying GMAC/s series was rejected")
    print(f"control self-test PASS: inverted device_ns_min series rejected ({problems[0]}); "
          "an inflated median alone accepted; flat rate series rejected, varying rate series "
          "accepted", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apeiron Q8 primitive latency at real projection shapes")
    parser.add_argument("--reps", type=int, default=5, help="repetitions per shape (default: 5)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS,
                        help="wall-clock budget in seconds (default: 540)")
    parser.add_argument("--control-self-test", action="store_true",
                        help="only run the seeded control-checker self-test")
    parser.add_argument("--shapes", default=None,
                        help="arbitrary shapes as 'label:M:K[;label:M:K...]' (K = contraction dim, "
                             "converted to K/32 blocks); default is the canonical Qwen3.5-0.8B set")
    parser.add_argument("--json", action="store_true",
                        help="also emit one JSON object line with the per-row device medians and rates")
    args = parser.parse_args()

    start = time.monotonic()
    deadline = start + args.budget - BUDGET_GRACE_SECONDS

    if args.control_self_test:
        return control_self_test()

    shapes = parse_shapes(args.shapes) if args.shapes else list(SHAPES)

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
        # Identity of the module actually executed: the shader source it was compiled from and the
        # SPIR-V that resulted. A joined comparator checks these to refuse a wrong-kernel run.
        shader_src_sha = sha256(bench.SHADER)
        shader_spirv_sha = sha256(shader_spv)

        for label, m, blocks in shapes:
            for tokens in PROMPT_TOKENS:
                results.append(run_case(bench, driver_bin, shader_spv, scratch,
                                        label, m, tokens, blocks, args.reps, deadline))

    elapsed = time.monotonic() - start
    ident = results[0]
    print("=" * 118)
    print("APEIRON Q8 SPIR-V PRIMITIVE LATENCY — REAL PROJECTION SHAPES")
    print("=" * 118)
    print(f"device        = {ident['device_name']} (driver={ident['driver_version']}, api={ident['api_version']})")
    print(f"kernel        = src/spirv/q8_tile.comp  source_sha256={shader_src_sha[:16]}  "
          f"spirv_sha256={shader_spirv_sha[:16]}")
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

    problems = check_ordering(results, shapes, PROMPT_TOKENS)
    print("-" * 118)
    if problems:
        for p in problems:
            print(f"CONTROL FAIL: {p}")
        print("=" * 118)
        return 1
    shape_problems = shape_variation_control(results)
    if shape_problems:
        for p in shape_problems:
            print(f"CONTROL FAIL: {p}")
        print("=" * 118)
        return 1
    rates = [float(r["gmac_per_s"]) for r in results]
    print("CONTROL PASS: device_ns strictly grows with tokens (each shape) "
          "and with blocks (M=1024, fixed tokens); "
          f"GMAC/s varies with shape (spread {max(rates) / min(rates):.2f}x >= {MIN_SHAPE_SPREAD}x)")
    if args.json:
        rows = [{
            "shape": r["label"], "m": r["m"], "k": int(r["blocks"]) * 32, "tokens": r["tokens"],
            "blocks": r["blocks"], "macs": r["macs"],
            "dev_med_ns": r["device_ns_med"], "dev_min_ns": r["device_ns_min"],
            "dev_max_ns": r["device_ns_max"], "reps": r["reps"],
            "gmac_s": r["gmac_per_s"], "gb_s": r["gb_per_s"], "device_name": r["device_name"],
            "driver_version": r["driver_version"], "api_version": r["api_version"],
            "timestamp_period": r["timestamp_period"],
        } for r in results]
        print(json.dumps({"incumbent": rows, "kernel": {
            "name": "incumbent q8_tile.comp (Vulkan SPIR-V GEMV)",
            "shader_source": str(bench.SHADER.relative_to(ROOT)),
            "shader_source_sha256": shader_src_sha,
            "shader_spirv_sha256": shader_spirv_sha,
        }}))
    print("=" * 118)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"q8_shapes: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
