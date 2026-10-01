#!/usr/bin/env python3
"""Build and smoke-check the exact-token llama.cpp prefill comparator.

SCOPE: tools/gpu/llama_cmp.c, tools/gpu/llama_cmp_gate.py.
REGISTER: comparator tooling only; no Apeiron inference change and no speed claim.
GATE: python3 tools/gpu/llama_cmp_gate.py [--ngl N]; a wrong id file must reject.

The smoke runs with --ngl 0 (CPU) by default so it does not contend with a concurrent GPU
measurement lane. Pass --ngl -1 for the GPU campaign once the GPU lane is free.
"""

from __future__ import annotations

import argparse
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
SOURCE = ROOT / "tools/gpu/llama_cmp.c"
DEFAULT_MODEL = Path("/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf")
DEFAULT_IDS = ROOT / "tests/fixtures/qwen35-prompt2.ids"
DEADLINE = time.monotonic() + 535.0


def remaining() -> float:
    left = DEADLINE - time.monotonic()
    if left <= 0:
        raise RuntimeError("llama_cmp_gate exceeded its 540-second budget")
    return left


def run(argv: list[str], *, expected: int = 0, cwd: Path = ROOT,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(argv), flush=True)
    cp = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=remaining(), check=False, env=env)
    if cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n", flush=True)
    print(f"exit={cp.returncode}", flush=True)
    if cp.returncode != expected:
        raise RuntimeError(f"command exit {cp.returncode}, expected {expected}: {argv[0]}")
    return cp


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="cpu hides every GPU backend; gpu offloads all layers")
    ap.add_argument("--ngl", type=int, default=None, help="override gpu layers")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=16, help="llama.cpp thread count (0 = default)")
    args = ap.parse_args()

    # `n_gpu_layers = 0` does NOT force CPU on this build: llama.cpp still schedules ops onto CUDA.
    # The only reliable CPU comparator hides the GPUs from the child.
    env = os.environ.copy()
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["GGML_VK_VISIBLE_DEVICES"] = "-1"
        ngl = 0 if args.ngl is None else args.ngl
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        ngl = -1 if args.ngl is None else args.ngl
    print(f"device_mode={args.device} n_gpu_layers={ngl}", flush=True)

    cc = shutil.which("cc")
    if cc is None:
        raise RuntimeError("cc not found")
    if not (BUILD / "libllama.so").exists():
        raise RuntimeError(f"llama build not found at {BUILD}")

    with tempfile.TemporaryDirectory(prefix="apeiron-llama-cmp-") as td:
        binary = Path(td) / "llama_cmp"
        run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
             f"-I{LLAMA / 'include'}", f"-I{LLAMA / 'ggml/include'}", str(SOURCE),
             f"-L{BUILD}", f"-Wl,-rpath,{BUILD}", "-lllama", "-o", str(binary)])

        print(f"source_sha256 tools/gpu/llama_cmp.c = {sha256(SOURCE)}")
        print(f"model_sha256 {args.model} = {sha256(args.model)}")
        print(f"ids_sha256 {args.ids} = {sha256(args.ids)}")
        build_commit = subprocess.run(
            ["git", "-C", str(LLAMA), "rev-parse", "HEAD"], text=True,
            stdout=subprocess.PIPE, check=True).stdout.strip()
        print(f"llama_build_commit = {build_commit}")

        cp = run([str(binary), str(args.model), str(args.ids), str(ngl),
                  str(args.n_ctx), str(args.n_batch), str(args.n_batch), str(args.threads)], env=env)
        line = [l for l in cp.stdout.splitlines() if l.startswith("{")][-1]
        rec = json.loads(line)
        if not (rec["tok_per_s"] > 0.0 and rec["n_tokens"] > 0):
            raise RuntimeError(f"non-positive comparator timing: {rec}")
        if not (0 <= rec["next_token"] < 200000):
            raise RuntimeError(f"next_token out of range: {rec['next_token']}")
        if args.device == "cpu" and "CUDA0 compute buffer" in cp.stdout:
            raise RuntimeError("CPU mode still allocated a CUDA compute buffer")
        print(f"smoke ok: device={args.device} n_tokens={rec['n_tokens']} "
              f"tok/s={rec['tok_per_s']:.2f} next={rec['next_token']} ngl={rec['n_gpu_layers']}")

        # Negative control: a malformed id file must be rejected, not silently accepted.
        bad = Path(td) / "bad.ids"
        bad.write_text("1 2 not_a_number 4\n")
        cp_bad = run([str(binary), str(args.model), str(bad), str(ngl),
                      str(args.n_ctx), str(args.n_batch), str(args.n_batch), str(args.threads)], expected=2, env=env)
        if "malformed id" not in cp_bad.stdout:
            raise RuntimeError("malformed-id control produced an unexpected diagnostic")
        print("negative-control=red (malformed id file rejected)")

        # Second control: ids exceeding n_batch must be refused by the harness, not truncated.
        big = Path(td) / "big.ids"
        big.write_text(" ".join(["1"] * (args.n_batch + 1)) + "\n")
        cp_big = run([str(binary), str(args.model), str(big), str(ngl),
                      str(args.n_ctx), str(args.n_batch), str(args.n_batch), str(args.threads)], expected=2, env=env)
        if "exceeds n_batch" not in cp_big.stdout:
            raise RuntimeError("over-batch control produced an unexpected diagnostic")
        print("negative-control=red (id count over n_batch refused)")

    print(f"gate_elapsed_seconds={time.monotonic() - (DEADLINE - 535.0):.3f}")
    print("llama_cmp_gate OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, KeyError) as exc:
        print(f"llama_cmp_gate: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
