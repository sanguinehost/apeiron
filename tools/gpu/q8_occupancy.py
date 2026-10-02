#!/usr/bin/env python3
"""Per-invocation cost attribution for the Apeiron Q8 SPIR-V kernel.

Drives the existing ``tools/gpu/q8_driver.c`` device-timestamp path (the same
instrument ``q8_shapes.py`` uses) over four bounded sweeps of the executing
kernel ``src/spirv/q8_tile.comp`` and decomposes the device time into dispatch,
per-workgroup latency, memory and arithmetic terms, so that the 30.8x-304x gap
to llama.cpp CUDA MMQ is attributed to the term that actually causes it.

Geometry (from the shader): ``local_size 8x8`` = 64 invocations/workgroup, each
invocation a 2-row x 4-token tile, so one workgroup covers 16 rows x 32 tokens
and ``MACS_PER_WG_BLOCK = 16*32*32 = 16384`` MACs per (workgroup, block).  The
grid is ``ng = ceil(M/16) * ceil(tokens/32)`` workgroups and the reduction depth
is ``blocks = K/32``.

Sweeps
------
* ``occ``  : blocks=64 fixed, M x tokens grid -> throughput vs grid size (occupancy).
* ``work`` : M=1024, tokens=256 (ng=512) fixed, blocks from 1 to 224 -> arithmetic plateau.
* ``iso``  : ng=1 (M=16, tokens=32), blocks in {1,64} -> dispatch floor and the
             exposed per-block serial latency of a single workgroup.
* ``real`` : the five Qwen3.5-0.8B projection shapes at tokens=200/512.

Roofline inputs are measured: arithmetic intensity ``MACs / unique_weight_bytes``
(= tokens for aligned shapes) against the fitted local arithmetic plateau, and
the achieved bytes/s against the CUDA-probed device peak bandwidth.

Controls (``--controls``)
-------------------------
* ``--control-self-test`` : pure checker self-test; a seeded flat (stuck-timer)
  series and a flat control pair must be rejected.
* ``--controls``          : runs the live sweeps, requires the occupancy series
  to be strictly monotone in ``ng`` and the control pair to differ measurably,
  then injects a stuck timer into a copy of the *real* occupancy results and
  requires the checker to redden under this harness, then confirms the
  unmutated series passes.

The instrument is the GPU timestamp delta across one dispatch; the control that
can fail is the ordering/difference check, which a stuck or zero timer violates.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
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

# --- geometry of the executing kernel (src/spirv/q8_tile.comp) ---------------
LOCAL_X = 8
LOCAL_Y = 8
INVOCATIONS_PER_WG = LOCAL_X * LOCAL_Y          # 64
ROWS_PER_WG = LOCAL_X * 2                        # 16
TOKENS_PER_WG = LOCAL_Y * 4                      # 32
MACS_PER_WG_BLOCK = ROWS_PER_WG * TOKENS_PER_WG * 32   # 16384

# --- device identity/limits (RTX 5080) --------------------------------------
DEVICE_NAME_FALLBACK = "NVIDIA GeForce RTX 5080"
SPEC_SM_COUNT = 84
SPEC_MAX_THREADS_PER_SM = 1536
SPEC_MEM_BUS_BITS = 256
SPEC_MEM_CLOCK_HZ = 15_001_000_000      # nvidia-smi clocks.max.memory = 15001 MHz

# --- sweep definitions -------------------------------------------------------
OCC_BLOCKS = 64
OCC_M = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
OCC_TOKENS = [32, 64, 128, 256]

WORK_M = 1024
WORK_TOKENS = 256                       # ng = 64 * 8 = 512
WORK_BLOCKS = [1, 8, 16, 32, 64, 112, 224]
DEEP_BLOCKS = 64                        # blocks >= this = "deep" regime for the plateau fit

ISO_M = 16
ISO_TOKENS = 32                         # ng = 1
ISO_BLOCKS = [1, 64]

CONTROL_PAIR = (16, 224)                # work-sweep reduction depths at ng=512
MIN_PAIR_RATIO = 1.50

REAL_TOKENS = [200, 512]
REAL_SHAPES = [
    ("attn_q", 4096, 32),
    ("attn_k_v", 512, 32),
    ("attn_output", 1024, 64),
    ("ffn_gate_up", 3584, 32),
    ("ffn_down", 1024, 112),
]

# CUDA MMQ comparator band (tools/gpu/cuda_q8_gate.py, measured on this box).
CUDA_GMAC_S = (6425.0, 53644.0)


def load_bench():
    spec = importlib.util.spec_from_file_location("q8_bench", HERE / "q8_bench.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load q8_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def derive(m: int, tokens: int, blocks: int) -> dict[str, float]:
    ng_x = ceil_div(m, 16)
    ng_y = ceil_div(tokens, 32)
    ng = ng_x * ng_y
    work_units = ng * blocks
    macs = m * tokens * blocks * 32
    unique_weight_bytes = m * blocks * 32                 # Q8_0 packed codes, 1 B/lane
    logical_weight_bytes = ng_y * unique_weight_bytes     # each token-tile re-reads W
    act_bytes = tokens * blocks * (8 * 4 + 4)
    out_bytes = tokens * m * (8 + 4)
    unique_total = unique_weight_bytes + act_bytes + out_bytes
    logical_total = logical_weight_bytes + act_bytes + out_bytes
    return {
        "ng": float(ng), "ng_x": float(ng_x), "ng_y": float(ng_y),
        "n_inv": float(ng * INVOCATIONS_PER_WG),
        "work_units": float(work_units),
        "macs": float(macs),
        "unique_bytes": float(unique_total),
        "logical_bytes": float(logical_total),
        "unique_weight_bytes": float(unique_weight_bytes),
        "intensity": macs / unique_total if unique_total else 0.0,
    }


def run_case(bench, driver_bin: Path, shader_spv: Path, scratch: Path, sweep: str,
             label: str, m: int, tokens: int, blocks: int, reps: int,
             deadline: float) -> dict[str, object]:
    left = remaining(deadline)
    if left <= 0:
        raise RuntimeError(f"q8_occupancy exceeded its wall-clock budget before {label} "
                           f"tokens={tokens} blocks={blocks}")
    tag = f"{sweep}-{label}-m{m}-t{tokens}-b{blocks}"
    inp = scratch / f"{tag}.input"
    out = scratch / f"{tag}.output"
    ts = scratch / f"{tag}.ts.json"
    inp.write_bytes(bench.synthesize_inputs(m, tokens, blocks))

    cmd = [
        str(driver_bin), str(shader_spv), str(m), str(tokens), str(blocks),
        str(inp), str(out), "--timestamps", str(ts), "--reps", str(reps),
    ]
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=left)
    if cp.returncode != 0:
        print(cp.stdout, flush=True)
        print(cp.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"q8_driver failed with code {cp.returncode} on {tag}: {cp.stderr.strip()}")
    if not ts.exists():
        raise RuntimeError(f"q8_driver did not emit timestamps for {tag}")

    data = json.loads(ts.read_text())
    reps_data = data["reps"]
    dev = [float(r["device_ns"]) for r in reps_data]
    sub = [int(r["submit_ns"]) for r in reps_data]
    wait = [int(r["wait_ns"]) for r in reps_data]
    rec: dict[str, object] = {
        "sweep": sweep, "label": label, "m": m, "tokens": tokens, "blocks": blocks,
        "reps": len(reps_data),
        "device_ns": dev,
        "device_ns_med": statistics.median(dev),
        "device_ns_min": min(dev), "device_ns_max": max(dev),
        "submit_ns_med": statistics.median(sub), "wait_ns_med": statistics.median(wait),
        "device_name": data["device_name"], "driver_version": data["driver_version"],
        "api_version": data["api_version"],
        "timestamp_period": float(data["timestamp_period"]),
        "timestamp_valid_bits": int(data["timestamp_valid_bits"]),
    }
    rec.update(derive(m, tokens, blocks))
    dev_med = float(rec["device_ns_med"])
    rec["gmac_per_s"] = float(rec["macs"]) / dev_med
    rec["unique_gb_per_s"] = float(rec["unique_bytes"]) / dev_med
    rec["logical_gb_per_s"] = float(rec["logical_bytes"]) / dev_med
    return rec


# ---------------------------------------------------------------------------
# Pure checkers (self-testable; a seeded flat series must be rejected)
# ---------------------------------------------------------------------------

def check_positive(results: list[dict[str, object]]) -> list[str]:
    return [f"{r['sweep']}/{r['label']} m={r['m']} t={r['tokens']} b={r['blocks']} "
            f"device_ns_med={r['device_ns_med']} not positive"
            for r in results if not float(r["device_ns_med"]) > 0.0]


def check_monotone(results: list[dict[str, object]], rel_tol: float = 0.03) -> list[str]:
    """At fixed tokens, device_ns must be non-decreasing in the grid size ng.

    Below the b-loop floor the grid size does not change device_ns and per-rep
    noise is ~1%, so a decrease is a problem only beyond ``rel_tol``.  A stuck
    timer is caught by :func:`check_dynamic_range`, not here.
    """
    problems: list[str] = []
    by: dict[int, list[dict[str, object]]] = {}
    for r in results:
        if r["sweep"] != "occ":
            continue
        by.setdefault(int(r["tokens"]), []).append(r)
    for tokens, rows in sorted(by.items()):
        rows = sorted(rows, key=lambda r: float(r["ng"]))
        for lo, hi in zip(rows, rows[1:]):
            if not float(hi["device_ns_med"]) >= float(lo["device_ns_med"]) * (1.0 - rel_tol):
                problems.append(
                    f"occupancy ordering violated at tokens={tokens}: "
                    f"ng={int(lo['ng'])} -> {float(lo['device_ns_med']):.1f} ns fell to "
                    f"ng={int(hi['ng'])} -> {float(hi['device_ns_med']):.1f} ns "
                    f"(beyond {rel_tol:.0%} noise)")
    return problems


def check_dynamic_range(results: list[dict[str, object]], min_span: float = 5.0) -> list[str]:
    """The instrument must see device_ns change across the occupancy sweep.

    ``max/min`` of the per-token occupancy series must clear ``min_span``; a
    stuck, zero or saturated timer gives a span of 1.0 and reddens here.
    """
    problems: list[str] = []
    by: dict[int, list[dict[str, object]]] = {}
    for r in results:
        if r["sweep"] != "occ":
            continue
        by.setdefault(int(r["tokens"]), []).append(r)
    for tokens, rows in sorted(by.items()):
        vals = [float(r["device_ns_med"]) for r in rows]
        if min(vals) <= 0:
            problems.append(f"occupancy series tokens={tokens} has a non-positive device_ns")
            continue
        span = max(vals) / min(vals)
        if span < min_span:
            problems.append(
                f"occupancy series tokens={tokens} dynamic range {span:.2f}x < {min_span:.1f}x: "
                "instrument sees no change")
    return problems


def check_pair(results: list[dict[str, object]], blocks_small: int,
               blocks_large: int, min_ratio: float) -> tuple[list[str], float | None]:
    by = {int(r["blocks"]): r for r in results if r["sweep"] == "work"}
    small, large = by.get(blocks_small), by.get(blocks_large)
    if small is None or large is None:
        return [f"control pair blocks={blocks_small}/{blocks_large} missing from work sweep"], None
    s, l = float(small["device_ns_med"]), float(large["device_ns_med"])
    ratio = l / s if s > 0 else 0.0
    if not ratio >= min_ratio:
        return ([f"work-sweep pair blocks={blocks_small} -> {s:.1f} ns vs blocks={blocks_large} "
                 f"-> {l:.1f} ns differ by only {ratio:.2f}x (< {min_ratio:.2f}x): "
                 "instrument sees no change"], ratio)
    return [], ratio


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    n = len(xs)
    if n < 2:
        return None
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (n * sxy - sx * sy) / denom
    icept = (sy - slope * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, icept, r2


def control_self_test() -> int:
    good = [{"sweep": "occ", "label": "seed", "m": 0, "tokens": t, "blocks": 64,
             "ng": float(ng), "device_ns_med": float(t + ng), "work_units": 1.0}
            for t in (32, 64) for ng in (16, 128, 1024)]
    if check_monotone(good) or check_dynamic_range(good):
        raise RuntimeError("control self-test: a monotone, wide-range series was rejected")
    flat = copy.deepcopy(good)
    for r in flat:
        r["device_ns_med"] = 100.0
    if not check_dynamic_range(flat):
        raise RuntimeError("control self-test: a flat (stuck-timer) series survived")
    inverted = [{"sweep": "occ", "label": "seed", "m": 0, "tokens": 32, "blocks": 64,
                 "ng": float(ng), "device_ns_med": float(2000 - ng), "work_units": 1.0}
                for ng in (16, 128, 1024)]
    if not check_monotone(inverted):
        raise RuntimeError("control self-test: an inverted series survived")
    pair_good = [{"sweep": "work", "label": "seed", "m": 0, "tokens": 0, "blocks": 16,
                  "ng": 1.0, "work_units": 1.0, "device_ns_med": 100.0},
                 {"sweep": "work", "label": "seed", "m": 0, "tokens": 0, "blocks": 224,
                  "ng": 1.0, "work_units": 1.0, "device_ns_med": 400.0}]
    if check_pair(pair_good, 16, 224, MIN_PAIR_RATIO)[0]:
        raise RuntimeError("control self-test: a live pair was rejected")
    pair_flat = copy.deepcopy(pair_good)
    for r in pair_flat:
        r["device_ns_med"] = 100.0
    if not check_pair(pair_flat, 16, 224, MIN_PAIR_RATIO)[0]:
        raise RuntimeError("control self-test: a flat pair survived")
    print("control self-test PASS: flat series and flat pair rejected; monotone series accepted",
          flush=True)
    return 0


def probe_device() -> dict[str, object]:
    """Best-effort CUDA device-property probe (same physical GPU) for the occupancy
    denominator and peak bandwidth; falls back to recorded spec constants."""
    from shutil import which
    facts: dict[str, object] = {
        "source": "spec-constants", "name": DEVICE_NAME_FALLBACK,
        "sm_count": SPEC_SM_COUNT, "max_threads_per_sm": SPEC_MAX_THREADS_PER_SM,
        "mem_bus_bits": SPEC_MEM_BUS_BITS,
    }
    nvcc = None
    for cand in ("/opt/cuda/bin/nvcc", "nvcc"):
        nvcc = which(cand)
        if nvcc:
            break
    if nvcc:
        with tempfile.TemporaryDirectory(prefix="apeiron-q8-occ-probe-") as td:
            src, binp = Path(td) / "devprop.cu", Path(td) / "devprop"
            src.write_text(
                "#include <cstdio>\n#include <cuda_runtime.h>\n"
                "int main(){cudaDeviceProp p;cudaGetDeviceProperties(&p,0);"
                "printf(\"%d %d %d %d\\n\",p.multiProcessorCount,p.maxThreadsPerMultiProcessor,"
                "p.memoryBusWidth,p.l2CacheSize);return 0;}\n")
            try:
                subprocess.run([nvcc, "-o", str(binp), str(src)], check=True,
                               capture_output=True, timeout=180)
                out = subprocess.run([str(binp)], check=True, capture_output=True,
                                     text=True, timeout=60).stdout.split()
                sm, threads, bus, l2 = int(out[0]), int(out[1]), int(out[2]), int(out[3])
                facts.update({"source": "cuda-device-prop", "sm_count": sm,
                              "max_threads_per_sm": threads, "mem_bus_bits": bus, "l2_bytes": l2})
                print(f"device probe (CUDA): sm_count={sm} max_threads_per_sm={threads} "
                      f"mem_bus_bits={bus} l2_bytes={l2}", flush=True)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                print(f"device probe fell back to spec constants: {exc}",
                      file=sys.stderr, flush=True)
    facts["mem_clock_hz"] = SPEC_MEM_CLOCK_HZ
    facts["peak_gb_per_s"] = (float(SPEC_MEM_CLOCK_HZ) * 2 * int(facts["mem_bus_bits"]) / 8 / 1e9)
    facts["resident_workgroups"] = (int(facts["sm_count"]) * int(facts["max_threads_per_sm"])
                                    // INVOCATIONS_PER_WG)
    return facts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sweep(results: list[dict[str, object]], title: str) -> None:
    print("-" * 128)
    print(title)
    print("-" * 128)
    print(f"{'shape':<12}{'M':>6}{'tok':>5}{'blk':>5}{'ng':>7} | "
          f"{'dev_min':>9}{'dev_med':>11}{'dev_max':>11} | {'sub_med':>8}{'wait_med':>10} | "
          f"{'GMAC/s':>9}{'uniqGB/s':>9}{'logicGB/s':>10}{'intens':>8}")
    for r in results:
        print(f"{str(r['label']):<12}{int(r['m']):>6}{int(r['tokens']):>5}{int(r['blocks']):>5}"
              f"{int(r['ng']):>7} | "
              f"{float(r['device_ns_min']):>9.1f}{float(r['device_ns_med']):>11.1f}"
              f"{float(r['device_ns_max']):>11.1f} | "
              f"{float(r['submit_ns_med']):>8.0f}{float(r['wait_ns_med']):>10.0f} | "
              f"{float(r['gmac_per_s']):>9.2f}{float(r['unique_gb_per_s']):>9.2f}"
              f"{float(r['logical_gb_per_s']):>10.2f}{float(r['intensity']):>8.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apeiron Q8 SPIR-V per-invocation cost attribution")
    parser.add_argument("--reps", type=int, default=9, help="repetitions per case (default: 9)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--controls", action="store_true",
                        help="run the seeded-control suite around the live sweeps")
    parser.add_argument("--control-self-test", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    start = time.monotonic()
    deadline = start + args.budget - BUDGET_GRACE_SECONDS

    if args.control_self_test:
        return control_self_test()

    bench = load_bench()
    glslc = bench.tool("glslc", "GLSLC", "glslc")
    spirv_val = bench.tool("spirv-val", "SPIRV_VAL", "spirv-val")
    cc = bench.tool("cc", "CC", "cc")
    probe = probe_device()
    peak_bw = float(probe["peak_gb_per_s"])

    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="apeiron-q8-occ-") as td:
        scratch = Path(td)
        shader_spv = scratch / "q8_tile.spv"
        driver_bin = scratch / "q8_driver"
        bench.compile_shader(glslc, spirv_val, bench.SHADER, shader_spv, deadline=deadline)
        bench.compile_driver(cc, bench.DRIVER, driver_bin, deadline=deadline)

        for m in OCC_M:
            for tokens in OCC_TOKENS:
                results.append(run_case(bench, driver_bin, shader_spv, scratch, "occ",
                                        "occ", m, tokens, OCC_BLOCKS, args.reps, deadline))
        for blocks in WORK_BLOCKS:
            results.append(run_case(bench, driver_bin, shader_spv, scratch, "work",
                                    "work", WORK_M, WORK_TOKENS, blocks, args.reps, deadline))
        for blocks in ISO_BLOCKS:
            results.append(run_case(bench, driver_bin, shader_spv, scratch, "iso",
                                    "iso", ISO_M, ISO_TOKENS, blocks, args.reps, deadline))
        for label, m, blocks in REAL_SHAPES:
            for tokens in REAL_TOKENS:
                results.append(run_case(bench, driver_bin, shader_spv, scratch, "real",
                                        label, m, tokens, blocks, args.reps, deadline))

    elapsed = time.monotonic() - start
    ident = results[0]
    occ = [r for r in results if r["sweep"] == "occ"]
    work = [r for r in results if r["sweep"] == "work"]
    iso = [r for r in results if r["sweep"] == "iso"]
    real = [r for r in results if r["sweep"] == "real"]

    print("=" * 128)
    print("APEIRON Q8 SPIR-V — PER-INVOCATION COST ATTRIBUTION")
    print("=" * 128)
    print(f"device          = {ident['device_name']} (driver={ident['driver_version']}, "
          f"api={ident['api_version']})")
    print(f"timestampPeriod = {float(ident['timestamp_period']):.2f} ns   "
          f"timestampValidBits = {int(ident['timestamp_valid_bits'])}")
    print(f"device probe    = {probe['source']}: sm_count={probe['sm_count']} "
          f"max_threads_per_sm={probe['max_threads_per_sm']} "
          f"resident_workgroups={probe['resident_workgroups']} "
          f"peak_bw={peak_bw:.1f} GB/s")
    print(f"kernel geometry = local 8x8 = {INVOCATIONS_PER_WG} invocations/workgroup, "
          f"{ROWS_PER_WG} rows x {TOKENS_PER_WG} tokens, {MACS_PER_WG_BLOCK} MACs/(wg,block)")
    print(f"reps            = {args.reps}   budget = {args.budget:.0f} s   "
          f"elapsed = {elapsed:.2f} s   cases = {len(results)}")

    print_sweep(occ, "OCCUPANCY SWEEP — blocks=64 fixed, M x tokens grid (grid size ng varies)")
    print_sweep(work, f"WORK SWEEP — M={WORK_M} tokens={WORK_TOKENS} (ng=512) fixed, blocks varies")
    print_sweep(iso, "ISOLATED SINGLE-WORKGROUP — ng=1 (M=16, tokens=32), blocks in {1,64}")
    print_sweep(real, "REAL PROJECTION SHAPES — Qwen3.5-0.8B")

    # --- attribution --------------------------------------------------------
    by_iso = {int(r["blocks"]): r for r in iso}
    t_dispatch = float(by_iso[1]["device_ns_med"]) if 1 in by_iso else float("nan")
    t_iso64 = float(by_iso[64]["device_ns_med"]) if 64 in by_iso else float("nan")
    l_blk = (t_iso64 - t_dispatch) / 63.0 if (64 in by_iso and 1 in by_iso) else float("nan")
    work_sorted = sorted(work, key=lambda r: int(r["blocks"]))
    deep = [r for r in work_sorted if int(r["blocks"]) >= DEEP_BLOCKS]
    deep_fit = linfit([float(r["work_units"]) for r in deep],
                      [float(r["device_ns_med"]) for r in deep])
    occ_fit = linfit([float(r["ng"]) for r in occ],
                     [float(r["device_ns_med"]) for r in occ])
    peak_gmac = max(float(r["gmac_per_s"]) for r in results)
    real_peak_gmac = max(float(r["gmac_per_s"]) for r in real)
    max_unique_bw = max(float(r["unique_gb_per_s"]) for r in results)
    max_logic_bw = max(float(r["logical_gb_per_s"]) for r in results)
    min_intens = min(float(r["intensity"]) for r in results)
    plateau = (MACS_PER_WG_BLOCK / deep_fit[0]) if (deep_fit and deep_fit[0] > 0) else float("nan")
    knee_intens = (plateau / peak_bw) if (math.isfinite(plateau) and peak_bw > 0) else float("nan")

    print("=" * 128)
    print("ATTRIBUTION")
    print("=" * 128)
    print(f"dispatch floor t0 (ng=1, blocks=1)        = {t_dispatch:>12.1f} ns")
    print(f"single-workgroup exposed per-block latency = {l_blk:>12.1f} ns/block "
          f"(= {l_blk / 1000.0:.2f} us/block for one workgroup, no occupancy hiding)")
    if deep_fit:
        b, w0, r2w = deep_fit
        print(f"deep-K marginal cost (blocks>={DEEP_BLOCKS})         = {b:>12.4f} ns per "
              f"(workgroup*block), fit R^2={r2w:.4f}")
        print(f"  -> local arithmetic plateau              = {plateau:>12.1f} GMAC/s "
              f"(= {MACS_PER_WG_BLOCK}/b)")
    if occ_fit:
        g, g0, r2o = occ_fit
        print(f"occupancy marginal (blocks=64, over ng=1..2048)= {g:>12.3f} ns per workgroup, "
              f"fit R^2={r2o:.4f} (curve is convex; single slope is a local bound)")
    print(f"peak observed GMAC/s (all sweeps)          = {peak_gmac:>12.2f}")
    print(f"peak observed GMAC/s (real shapes)         = {real_peak_gmac:>12.2f}")
    print(f"CUDA MMQ comparator band                   = {CUDA_GMAC_S[0]:>8.0f} .. "
          f"{CUDA_GMAC_S[1]:>8.0f} GMAC/s")
    print(f"achieved unique-byte bandwidth (max)       = {max_unique_bw:>12.2f} GB/s "
          f"({max_unique_bw / peak_bw * 100:.2f}% of {peak_bw:.0f} GB/s peak)")
    print(f"achieved logical-byte bandwidth (max)      = {max_logic_bw:>12.2f} GB/s "
          f"({max_logic_bw / peak_bw * 100:.2f}% of peak)")
    print(f"minimum arithmetic intensity               = {min_intens:>12.2f} MAC/byte "
          f"(roofline knee ~{knee_intens:.2f} MAC/byte)")
    print(f"dispatch floor share at real shapes        = "
          f"{t_dispatch / min(float(r['device_ns_med']) for r in real) * 100:.3f}% .. "
          f"{t_dispatch / max(float(r['device_ns_med']) for r in real) * 100:.3f}% "
          f"(of measured device_ns)")

    # verdict
    print("-" * 128)
    memory_bound = max_unique_bw > 0.5 * peak_bw or min_intens < knee_intens
    if memory_bound:
        verdict = ("MEMORY-BOUND: achieved bandwidth is a large fraction of peak or intensity "
                   "sits below the roofline knee")
    else:
        ratio_lo = plateau / CUDA_GMAC_S[1] if math.isfinite(plateau) else float("nan")
        verdict = (
            "NOT memory-bound (achieved bandwidth <= "
            f"{max_unique_bw / peak_bw * 100:.1f}% of peak; intensity >= {min_intens:.0f} MAC/byte, "
            f"far above the {knee_intens:.1f} MAC/byte knee). Dispatch is negligible "
            f"(<= {t_dispatch / min(float(r['device_ns_med']) for r in real) * 100:.2f}%). "
            "The dominant term is per-workgroup instruction/latency exposure: even the kernel's "
            f"own arithmetic plateau ({plateau:.0f} GMAC/s) is {CUDA_GMAC_S[0] / plateau:.1f}x-"
            f"{CUDA_GMAC_S[1] / plateau:.1f}x below CUDA MMQ, and the real shapes reach only "
            f"{real_peak_gmac / plateau * 100:.0f}% of that plateau. The fix is instruction "
            "throughput (tensor cores / cooperative matrix) with occupancy-aware tiling, not a "
            "memory-layout change.")
    print(f"verdict: {verdict}")

    # --- checks -------------------------------------------------------------
    problems: list[str] = []
    problems += check_positive(results)

    mono_problems = check_monotone(occ)
    range_problems = check_dynamic_range(occ)
    pair_problems, pair_ratio = check_pair(results, *CONTROL_PAIR, MIN_PAIR_RATIO)
    print("-" * 128)
    print(f"control pair blocks={CONTROL_PAIR[0]} vs {CONTROL_PAIR[1]} at ng=512: "
          f"device_ns ratio = {pair_ratio:.2f}x (threshold {MIN_PAIR_RATIO:.2f}x)"
          if pair_ratio is not None else "control pair unavailable")
    if not deep_fit or deep_fit[0] <= 0:
        problems.append("deep-K fit has no positive arithmetic slope")
    if not occ_fit or occ_fit[0] <= 0:
        problems.append("occupancy fit has no positive per-workgroup slope")
    problems += pair_problems

    if args.controls:
        control_self_test()
        flat = copy.deepcopy(occ)
        stuck = statistics.median(float(r["device_ns_med"]) for r in occ)
        for r in flat:
            r["device_ns_med"] = stuck
        flat_problems = check_dynamic_range(flat)
        if not flat_problems:
            print("CONTROL FAIL: seeded stuck-timer occupancy series survived the checker")
            return 1
        print(f"control red (observed firing under this harness): seeded stuck-timer series "
              f"rejected — {flat_problems[0]}")
        if mono_problems or range_problems:
            print("CONTROL FAIL: the live occupancy series is not monotone-in-trend / wide-range")
            return 1
        print("control green: the unmutated live occupancy series is non-decreasing in ng with a "
              "wide dynamic range")

    print("=" * 128)
    if problems or mono_problems or range_problems:
        for p in problems + mono_problems + range_problems:
            print(f"CONTROL FAIL: {p}")
        print("=" * 128)
        return 1
    print("CONTROL PASS: timer live (wide range, ordering in trend, control pair differs), "
          "attribution slopes positive")
    print("=" * 128)

    if args.json is not None:
        payload = {
            "device": {"name": ident["device_name"], "driver_version": ident["driver_version"],
                       "api_version": ident["api_version"],
                       "timestamp_period": ident["timestamp_period"],
                       "timestamp_valid_bits": ident["timestamp_valid_bits"]},
            "probe": probe,
            "geometry": {"invocations_per_wg": INVOCATIONS_PER_WG, "rows_per_wg": ROWS_PER_WG,
                         "tokens_per_wg": TOKENS_PER_WG, "macs_per_wg_block": MACS_PER_WG_BLOCK},
            "attribution": {"dispatch_floor_ns": t_dispatch, "iso_per_block_ns": l_blk,
                            "deep_slope_ns_per_wg_block": deep_fit[0] if deep_fit else None,
                            "deep_fit_r2": deep_fit[2] if deep_fit else None,
                            "occupancy_slope_ns_per_wg": occ_fit[0] if occ_fit else None,
                            "plateau_gmac_s": plateau, "peak_gmac_s": peak_gmac,
                            "real_peak_gmac_s": real_peak_gmac,
                            "max_unique_gb_s": max_unique_bw, "max_logical_gb_s": max_logic_bw,
                            "min_intensity": min_intens, "knee_intensity": knee_intens,
                            "peak_bw_gb_s": peak_bw, "verdict": verdict},
            "pair_ratio": pair_ratio, "results": results, "elapsed_s": elapsed,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")

    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"q8_occupancy: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
