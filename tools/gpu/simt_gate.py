#!/usr/bin/env python3
"""APEIRON-GPU-PORT gate: the ported fold SIMT kernels (GEMV + DP4A), end to end.

Stages, each with a declared wall-clock budget (§10):

  val     spirv-val --target-env vulkan1.1 accepts every emitted module, and rejects the
          corrupted control (`cargo test --test simt_spirv_val`).
  ops     the 12-op alphabet runs on the GPU and compares against `simt::op_ref`
          (`cargo run --example simt_check`).
  gemv    the carrier-dial GEMV (F32) and the integer-dot chart GEMV (Q8) run on the GPU, are
          graded against the CPU fold, and are timed (`cargo run --example simt_gemv`).

  control the negative control: mutate the real emitter source in a throwaway copy
          (`Op::AAdd => OP_IADD` -> `OP_IMUL`) so the module still validates but computes the
          wrong result; `simt_check` must fail **by name** (`FAIL aadd`) and exit nonzero;
          restore and confirm green again.

The base tree this runs in is the disposable port copy (`/home/socol/oc-tmp/apeiron-gpuport`),
never the live workspace. The mutation swaps the one source file and restores it in a `finally`.
Exit 0 only when every stage is green and the control is red-then-restored.

Usage: python3 tools/gpu/simt_gate.py [--no-bench] [--shape MxK]
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
EMITTER = ROOT / "src" / "simt" / "mod.rs"
CARGO = "cargo"
FEATURES = ["--features", "gpu"]

BUDGET = {
    "val": 540,
    "ops": 300,
    "gemv": 540,
    "control": 300,
}


def run(stage: str, argv: list[str], env=None) -> tuple[int, str, float]:
    budget = BUDGET[stage]
    start = time.monotonic()
    try:
        out = subprocess.run(
            argv,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=budget,
        )
        rc = out.returncode
        text = out.stdout
    except subprocess.TimeoutExpired as exc:
        rc = 124
        text = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    elapsed = time.monotonic() - start
    return rc, text, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bench", action="store_true", help="skip the timing measurement")
    ap.add_argument("--shape", default="32768x4096", help="SIMT_M x SIMT_K for the bench")
    args = ap.parse_args()

    m, k = args.shape.split("x")
    import os

    env = dict(os.environ, SIMT_M=m, SIMT_K=k, SIMT_REPS="5")

    failures: list[str] = []

    # ── Stage val: spirv-val accepts every module; the corrupt control is rejected ─────────────
    rc, text, dt = run("val", [CARGO, "test", *FEATURES, "--test", "simt_spirv_val", "--", "--nocapture"])
    print(f"[val] exit={rc} {dt:.1f}s")
    if rc != 0 or "Invalid opcode" not in text:
        failures.append("val")

    # ── Stage ops: 12/12 ops match op_ref on the GPU ───────────────────────────────────────────
    rc, text, dt = run("ops", [CARGO, "run", "--release", *FEATURES, "--example", "simt_check"])
    print(f"[ops] exit={rc} {dt:.1f}s")
    if rc != 0 or "12/12 ops PASS" not in text:
        failures.append("ops")
    else:
        print("      " + next((ln for ln in text.splitlines() if "12/12" in ln), "").strip())

    # ── Stage gemv: CPU-reference parity (F32 exact, Q8 chart) + measurement ────────────────────
    gemv_args = [CARGO, "run", "--release", *FEATURES, "--example", "simt_gemv", "--", "--check"]
    if not args.no_bench:
        gemv_args.append("--bench")
    rc, text, dt = run("gemv", gemv_args, env=env)
    print(f"[gemv] exit={rc} {dt:.1f}s")
    if rc != 0 or "CHECK f32 gemv" not in text or "CHECK q8 dp4a gemv" not in text:
        failures.append("gemv")
    for ln in text.splitlines():
        if ln.startswith(("CHECK", "BENCH")):
            print("      " + ln)

    # ── Stage control: mutate the real emitter -> simt_check red by name -> restore ─────────────
    if not EMITTER.exists():
        print(f"[control] emitter source missing: {EMITTER}")
        failures.append("control")
    else:
        original = EMITTER.read_text()
        needle = "Op::AAdd => bin(OP_IADD),"
        assert original.count(needle) == 1, "mutation anchor not unique"
        mutated = original.replace(needle, "Op::AAdd => bin(OP_IMUL),")
        try:
            EMITTER.write_text(mutated)
            rc, text, dt = run("control", [CARGO, "run", "--release", *FEATURES, "--example", "simt_check"], env=env)
            red_by_name = rc != 0 and "FAIL aadd" in text
            print(f"[control] mutated exit={rc} {dt:.1f}s  red_by_name={red_by_name}")
            if not red_by_name:
                failures.append("control:red")
        finally:
            EMITTER.write_text(original)
        # restore check
        rc, text, dt = run("control", [CARGO, "run", "--release", *FEATURES, "--example", "simt_check"], env=env)
        restored = rc == 0 and "12/12 ops PASS" in text
        print(f"[control] restored exit={rc} {dt:.1f}s  green={restored}")
        if not restored:
            failures.append("control:restore")

    if failures:
        print(f"\nGATE RED: {failures}")
        return 1
    print("\nGATE GREEN: val ok, ops 12/12, gemv parity ok, control red-by-name then restored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
