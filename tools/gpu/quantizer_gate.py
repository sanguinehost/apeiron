#!/usr/bin/env python3
"""Gate actual `quant_act` fixtures, native Vulkan parity, and tie-rounding sensitivity."""

from __future__ import annotations

import hashlib
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/tmp/opencode/native-real-quantizer")
LOCK = Path("/tmp/opencode/apeiron-rust-lane.lock")
TARGET: Path | None = None
DEADLINE = time.monotonic() + 525
START = time.monotonic()
ENV = os.environ.copy()
ENV["CARGO_TARGET_DIR"] = str(TARGET)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run(argv: list[str], *, cwd: Path = ROOT, expected: int = 0,
        env: dict[str, str] = ENV, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    left = DEADLINE - time.monotonic()
    if left <= 0:
        raise RuntimeError("APEIRON-GPU-PREFILL-W3-T1 exceeded 540 seconds")
    limit = min(left, timeout) if timeout is not None else left
    print("$ " + " ".join(argv), flush=True)
    child = subprocess.Popen(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = child.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            output, _ = child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            output, _ = child.communicate()
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        raise RuntimeError(f"process group timed out after {limit:.1f}s: {argv[0]}")
    cp = subprocess.CompletedProcess(argv, child.returncode, output)
    if cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n", flush=True)
    print(f"exit={cp.returncode}", flush=True)
    if cp.returncode != expected:
        raise RuntimeError(f"command exit {cp.returncode}, expected {expected}")
    return cp


def producer(worktree: Path, dest: Path, *, expected: int = 0,
             target: Path | None = None) -> subprocess.CompletedProcess[str]:
    target = target or TARGET
    if target is None:
        raise RuntimeError("private live Cargo target has not been initialized")
    env = ENV.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["APEIRON_NATIVE_Q8_FIXTURE"] = str(dest)
    return run([
        "flock", str(LOCK), "timeout", f"{max(1, int(DEADLINE-time.monotonic()))}s",
        "env", f"CARGO_TARGET_DIR={target}", f"APEIRON_NATIVE_Q8_FIXTURE={dest}",
        "cargo", "test", "--locked", "--lib",
        "engine::kern::gemm::native_gpu_probe::native_real_quantizer_q8_fixture",
        "--", "--ignored", "--exact", "--nocapture",
    ], cwd=worktree, env=env, expected=expected)


def gpu_gate(fixture: Path) -> None:
    run([sys.executable, "tools/gpu/q8_gate.py", "--profile", "quantizer",
         "--fixture", str(fixture)])


def tracked_copy(destination: Path) -> None:
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                             stdout=subprocess.PIPE).stdout
    for name in listing.decode().split("\0"):
        if not name or not (name.startswith(("src/", "derive/", "fold-abi/"))
                            or name in {"Cargo.toml", "Cargo.lock"}):
            continue
        src = ROOT / name
        dst = destination / name
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def run_gate(task_tmp: Path) -> int:
    fixture = OUT / "actual-quantizer.bin"
    initial = producer(ROOT, fixture)
    if "REAL_Q8_FIXTURE_OK cases=12 some=14581 refusal=5" not in initial.stdout:
        raise RuntimeError("focused producer did not report the pinned case and result counts")
    original_digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if original_digest != "49f14cc4d2a219f278353245b454ed31ef7da10af429c4ed122beadc57b3b958":
        raise RuntimeError(f"actual fixture digest changed: {original_digest}")
    print(f"actual_fixture={fixture} sha256={original_digest} bytes={fixture.stat().st_size}", flush=True)
    gpu_gate(fixture)

    with tempfile.TemporaryDirectory(prefix="apeiron-real-quantizer-mutation-") as td:
        copy = Path(td) / "repo"
        tracked_copy(copy)
        source = copy / "src/engine/chart/mod.rs"
        text = source.read_text()
        anchor = "2 * remainder >= s"
        if text.count(anchor) != 1:
            raise RuntimeError("actual quant_act tie-rounding mutation anchor missing or ambiguous")
        source.write_text(text.replace(anchor, "2 * remainder > s"))
        mutated_fixture = Path(td) / "mutated.bin"
        mutation_target = task_tmp / "mutation-target"
        cp = producer(copy, mutated_fixture, expected=101, target=mutation_target)
        if "actual quant_act signed tie must round away from zero" not in cp.stdout:
            raise RuntimeError("quantizer mutation did not fail at the signed-tie assertion")
        if "error[E" in cp.stdout or "could not compile" in cp.stdout:
            raise RuntimeError("quantizer mutation failed to compile instead of failing its assertion")
        print("quantizer-negative-control=red (strict tie comparison rejected by actual signed-tie witness)",
              flush=True)

    restored = OUT / "restored-quantizer.bin"
    producer(ROOT, restored)
    restored_digest = hashlib.sha256(restored.read_bytes()).hexdigest()
    if restored_digest != original_digest or restored.read_bytes() != fixture.read_bytes():
        raise RuntimeError("restored producer did not reproduce the original fixture byte-for-byte")
    print(f"restoration=green sha256={restored_digest}", flush=True)
    gpu_gate(restored)
    for rel in ("src/engine/kern/native_gpu_probe.rs", "tools/gpu/q8_gate.py",
                "tools/gpu/quantizer_gate.py"):
        path = ROOT / rel
        print(f"source_sha256 {rel}={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"gate_elapsed_seconds={time.monotonic()-START:.3f}", flush=True)
    print("gate_exit=0", flush=True)
    return 0


def main() -> int:
    global TARGET
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / "gate.log").open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = sys.stdout
    with tempfile.TemporaryDirectory(prefix="cargo-targets-", dir=OUT) as td:
        task_tmp = Path(td)
        TARGET = task_tmp / "live-target"
        ENV["CARGO_TARGET_DIR"] = str(TARGET)
        return run_gate(task_tmp)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"quantizer_gate: {exc}", file=sys.stderr, flush=True)
        print("gate_exit=1", file=sys.stderr, flush=True)
        sys.exit(1)
