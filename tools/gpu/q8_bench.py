#!/usr/bin/env python3
"""Benchmark and timing measurement for Apeiron's native Q8 Vulkan baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SHADER = ROOT / "src/spirv/q8_tile.comp"
DRIVER = ROOT / "tools/gpu/q8_driver.c"
CONTROLS = ROOT / "tools/gpu/q8_bench_controls.py"

START_TIME = time.monotonic()
DEFAULT_BUDGET_SECONDS = 540.0


def tool(name: str, env_name: str, fallback: str) -> str:
    value = os.environ.get(env_name, fallback)
    found = shutil.which(value)
    if found is None:
        raise RuntimeError(f"required tool {value!r} ({env_name}) is not available")
    return found


def compile_shader(glslc: str, spirv_val: str, source: Path, output: Path, *, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("APEIRON-GPU-PREFILL-W5-T1 exceeded its 540-second wall-clock budget")
    subprocess.run([glslc, "-O", "--target-env=vulkan1.3", str(source), "-o", str(output)],
                   check=True, timeout=remaining)
    subprocess.run([spirv_val, "--target-env", "vulkan1.3", str(output)],
                   check=True, timeout=remaining)


def compile_driver(cc: str, source: Path, output: Path, *, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("APEIRON-GPU-PREFILL-W5-T1 exceeded its 540-second wall-clock budget")
    cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(source),
           "-lm", "-lvulkan", "-o", str(output)]
    subprocess.run(cmd, check=True, timeout=remaining)


def synthesize_inputs(m: int, tokens: int, blocks: int) -> bytes:
    words_w = m * blocks * 8
    words_dm = m * blocks * 2
    words_x = tokens * blocks * 8
    words_e = tokens * blocks

    wb = struct.pack("<" + "I" * words_w, *[((i * 1103515245 + 12345) & 0xFFFFFFFF) for i in range(words_w)])
    dm_vals = []
    for _ in range(m * blocks):
        dm_vals.append(100)  # mantissa <= 2047
        dm_vals.append(16)   # exp 8..37
    dmb = struct.pack("<" + "I" * words_dm, *dm_vals)
    xb = struct.pack("<" + "I" * words_x, *[((i * 1664525 + 1013904223) & 0xFFFFFFFF) for i in range(words_x)])
    eb = struct.pack("<" + "I" * words_e, *([12] * words_e))
    return wb + dmb + xb + eb


def run_benchmark(args: argparse.Namespace, deadline: float) -> dict[str, object]:
    from tools.gpu.q8_gate import read_fixture

    glslc = tool("glslc", "GLSLC", "glslc")
    spirv_val = tool("spirv-val", "SPIRV_VAL", "spirv-val")
    cc = tool("cc", "CC", "cc")

    driver_c_sha = hashlib.sha256(DRIVER.read_bytes()).hexdigest()
    shader_comp_sha = hashlib.sha256(SHADER.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory(prefix="apeiron-q8-bench-") as td:
        scratch = Path(td)
        shader_spv = scratch / "q8_tile.spv"
        driver_bin = scratch / "q8_driver"

        compile_shader(glslc, spirv_val, SHADER, shader_spv, deadline=deadline)
        compile_driver(cc, DRIVER, driver_bin, deadline=deadline)
        spv_sha = hashlib.sha256(shader_spv.read_bytes()).hexdigest()

        cases: list[dict[str, object]]
        if args.timing_only:
            m = args.m or 256
            tokens = args.tokens or 32
            blocks = args.blocks or 64
            inp_bytes = synthesize_inputs(m, tokens, blocks)
            cases = [{
                "id": 0,
                "m": m,
                "tokens": tokens,
                "blocks": blocks,
                "input": inp_bytes,
                "noutputs": tokens * m,
                "expected": None,
                "refused": None,
            }]
        else:
            if args.fixture is None:
                raise ValueError("pass --fixture or set APEIRON_NATIVE_Q8_FIXTURE")
            fixture = Path(args.fixture).expanduser().resolve()
            all_cases = read_fixture(fixture, args.profile)
            if args.m is not None or args.tokens is not None or args.blocks is not None:
                filtered = [
                    c for c in all_cases
                    if (args.m is None or c["m"] == args.m)
                    and (args.tokens is None or c["tokens"] == args.tokens)
                    and (args.blocks is None or c["blocks"] == args.blocks)
                ]
                if not filtered:
                    raise ValueError(
                        f"no case matching shape m={args.m}, tokens={args.tokens}, blocks={args.blocks} in {fixture}"
                    )
                cases = filtered
            else:
                cases = all_cases

        case_results = []
        device_info: dict[str, object] = {}

        for i, case in enumerate(cases):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("APEIRON-GPU-PREFILL-W5-T1 exceeded its 540-second wall-clock budget")

            inp_file = scratch / f"bench-{i}.input"
            out_file = scratch / f"bench-{i}.output"
            ts_file = scratch / f"bench-{i}.ts.json"

            inp_file.write_bytes(case["input"])

            cmd = [
                str(driver_bin),
                str(shader_spv),
                str(case["m"]),
                str(case["tokens"]),
                str(case["blocks"]),
                str(inp_file),
                str(out_file),
                "--timestamps",
                str(ts_file),
                "--reps",
                str(args.reps),
            ]

            cp = subprocess.run(cmd, text=True, capture_output=True, timeout=remaining)
            if cp.returncode != 0:
                print(cp.stdout, flush=True)
                print(cp.stderr, file=sys.stderr, flush=True)
                raise RuntimeError(f"q8_driver failed with code {cp.returncode} on case {case['id']}: {cp.stderr.strip()}")

            if not ts_file.exists():
                raise RuntimeError(f"q8_driver did not emit timestamps file {ts_file}")

            ts_data = json.loads(ts_file.read_text())
            if not device_info:
                device_info = {
                    "device_name": ts_data["device_name"],
                    "driver_version": ts_data["driver_version"],
                    "api_version": ts_data["api_version"],
                    "timestamp_period": ts_data["timestamp_period"],
                    "timestamp_compute_and_graphics": ts_data["timestamp_compute_and_graphics"],
                    "timestamp_valid_bits": ts_data["timestamp_valid_bits"],
                }

            # Check parity if fixture expected data is present
            if case.get("expected") is not None and case.get("refused") is not None:
                raw_out = out_file.read_bytes()
                n = int(case["noutputs"])
                if len(raw_out) != n * 12:
                    raise RuntimeError(f"case {case['id']}: driver returned {len(raw_out)} bytes, expected {n * 12}")
                got = list(struct.unpack_from("<" + "q" * n, raw_out, 0))
                flags = list(struct.unpack_from("<" + "I" * n, raw_out, n * 8))
                exp_flags = case["refused"]
                exp_values = case["expected"]
                for j, (value, flag, want, want_flag) in enumerate(zip(got, flags, exp_values, exp_flags)):
                    if flag != want_flag or (flag == 0 and value != want):
                        raise RuntimeError(f"PARITY MISMATCH case={case['id']} output={j} expected=({want},{want_flag}) actual=({value},{flag})")

            reps = ts_data["reps"]
            dev_ns = [r["device_ns"] for r in reps]
            sub_ns = [r["submit_ns"] for r in reps]
            wait_ns = [r["wait_ns"] for r in reps]

            case_results.append({
                "id": case["id"],
                "m": case["m"],
                "tokens": case["tokens"],
                "blocks": case["blocks"],
                "reps": len(reps),
                "device_ns_min": min(dev_ns),
                "device_ns_med": statistics.median(dev_ns),
                "device_ns_max": max(dev_ns),
                "submit_ns_min": min(sub_ns),
                "submit_ns_med": statistics.median(sub_ns),
                "submit_ns_max": max(sub_ns),
                "wait_ns_min": min(wait_ns),
                "wait_ns_med": statistics.median(wait_ns),
                "wait_ns_max": max(wait_ns),
            })

    # Device SHA256 identity
    dev_str = f"{device_info.get('device_name')}|{device_info.get('driver_version')}|{device_info.get('api_version')}"
    device_sha = hashlib.sha256(dev_str.encode()).hexdigest()

    agg_device_ns_med = sum(c["device_ns_med"] for c in case_results)
    agg_submit_ns_med = sum(c["submit_ns_med"] for c in case_results)
    agg_wait_ns_med = sum(c["wait_ns_med"] for c in case_results)

    print("=" * 80)
    print("APEIRON Q8 VULKAN BASELINE BENCHMARK REPORT")
    print("=" * 80)
    print(f"source_sha256 tools/gpu/q8_driver.c = {driver_c_sha}")
    print(f"source_sha256 src/spirv/q8_tile.comp = {shader_comp_sha}")
    print(f"spirv_sha256  q8_tile.spv           = {spv_sha}")
    print(f"device_sha256                       = {device_sha} (name={device_info.get('device_name')}, driver={device_info.get('driver_version')}, api={device_info.get('api_version')})")
    print()
    print(f"limits timestampPeriod              = {device_info.get('timestamp_period'):.2f} ns")
    print(f"limits timestampComputeAndGraphics  = {device_info.get('timestamp_compute_and_graphics')}")
    print(f"limits timestampValidBits           = {device_info.get('timestamp_valid_bits')}")
    print()
    print(f"PER-CASE TIMINGS (cases={len(case_results)}, reps={args.reps}):")
    for cr in case_results:
        print(f"case {cr['id']:2d} (M={cr['m']:4d}, tokens={cr['tokens']:3d}, blocks={cr['blocks']:2d}): "
              f"device_ns=[min={cr['device_ns_min']:8.1f}, med={cr['device_ns_med']:8.1f}, max={cr['device_ns_max']:8.1f}], "
              f"submit_ns={cr['submit_ns_med']:6.0f}, wait_ns={cr['wait_ns_med']:6.0f}")
    print()
    print("AGGREGATE TIMINGS:")
    print(f"aggregate device_ns_median = {agg_device_ns_med:10.1f} ns")
    print(f"aggregate submit_ns_median = {agg_submit_ns_med:10.0f} ns")
    print(f"aggregate wait_ns_median   = {agg_wait_ns_med:10.0f} ns")
    print("=" * 80)

    return {
        "cases": case_results,
        "aggregate_device_ns_med": agg_device_ns_med,
        "aggregate_submit_ns_med": agg_submit_ns_med,
        "aggregate_wait_ns_med": agg_wait_ns_med,
        "device_info": device_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apeiron Q8 Vulkan baseline benchmark")
    parser.add_argument("--profile", choices=["synthetic", "quantizer"], default="synthetic",
                        help="registered fixture profile; synthetic is default")
    parser.add_argument("--fixture", type=Path, default=os.environ.get("APEIRON_NATIVE_Q8_FIXTURE"),
                        help="path to CPU-produced fixture")
    parser.add_argument("--m", type=int, default=None, help="explicit M shape override")
    parser.add_argument("--tokens", type=int, default=None, help="explicit tokens shape override")
    parser.add_argument("--blocks", type=int, default=None, help="explicit blocks shape override")
    parser.add_argument("--timing-only", action="store_true",
                        help="synthesize bounded deterministic inputs in-process without fixture")
    parser.add_argument("--reps", type=int, default=5, help="repetitions per case (default: 5)")
    parser.add_argument("--controls", action="store_true", help="run negative control and restoration suite")
    parser.add_argument("--deadline", type=float, default=None, help="wall-clock monotonic deadline")
    args = parser.parse_args()

    deadline = args.deadline or (START_TIME + DEFAULT_BUDGET_SECONDS - 5.0)

    if args.controls:
        # Run controls suite (which runs throwaway negative controls + worktree benchmark restoration)
        cmd = [
            sys.executable,
            str(CONTROLS),
            "--profile", args.profile,
            "--reps", str(args.reps),
            "--deadline", str(deadline),
        ]
        if args.fixture:
            cmd.extend(["--fixture", str(args.fixture)])
        if args.timing_only:
            cmd.append("--timing-only")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("APEIRON-GPU-PREFILL-W5-T1 exceeded its 540-second wall-clock budget")

        cp = subprocess.run(cmd, cwd=ROOT, timeout=remaining)
        if cp.returncode != 0:
            return cp.returncode
        print("PASS", flush=True)
        return 0

    run_benchmark(args, deadline)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"q8_bench: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
