#!/usr/bin/env python3
"""Compile, validate, execute and mutation-check Apeiron's native Q8 Vulkan baseline."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
SHADER = ROOT / "src/spirv/q8_tile.comp"
DRIVER = ROOT / "tools/gpu/q8_driver.c"
MAGIC = b"SPIQ8V01"
FIXTURE_PROFILES = {
    "synthetic": {
        "sha256": "573a0e1c020dba300ac0dd58693fa5e31b8212e96881e9793c388e42892d5fa2",
        "cases": 14, "success": 14678, "refusal": 32,
    },
    "quantizer": {
        "sha256": "49f14cc4d2a219f278353245b454ed31ef7da10af429c4ed122beadc57b3b958",
        "cases": 12, "success": 14581, "refusal": 5,
    },
}
DEADLINE = time.monotonic() + 535
START_TIME = time.monotonic()


def run(argv: list[str], *, expected: int = 0) -> subprocess.CompletedProcess[str]:
    remaining = DEADLINE - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("APEIRON-GPU-PREFILL-W2-T2 exceeded its 540-second wall-clock budget")
    print("$ " + " ".join(argv), flush=True)
    cp = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=remaining, check=False)
    if cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n", flush=True)
    print(f"exit={cp.returncode}", flush=True)
    if cp.returncode != expected:
        raise RuntimeError(f"command exit {cp.returncode}, expected {expected}: {argv[0]}")
    return cp


def tool(name: str, env_name: str, fallback: str) -> str:
    value = os.environ.get(env_name, fallback)
    found = shutil.which(value)
    if found is None:
        raise RuntimeError(f"required tool {value!r} ({env_name}) is not available")
    return found


def read_fixture(path: Path, profile: str) -> list[dict[str, object]]:
    data = path.read_bytes()
    if len(data) < 12 or data[:8] != MAGIC:
        raise ValueError("fixture must start with SPIQ8V01 and a u32 case count")
    count = struct.unpack_from("<I", data, 8)[0]
    if count == 0 or count > 100_000:
        raise ValueError(f"invalid fixture case count {count}")
    off = 12
    cases = []
    ids = set()

    def take_words(n: int, label: str) -> tuple[list[int], bytes]:
        nonlocal off
        if n < 0 or n > (1 << 30):
            raise ValueError(f"{label}: unreasonable word count {n}")
        size = n * 4
        if off + size > len(data):
            raise ValueError(f"{label}: truncated fixture at byte {off}")
        raw = data[off:off + size]
        off += size
        return list(struct.unpack("<" + "I" * n, raw)), raw

    for index in range(count):
        if off + 16 > len(data):
            raise ValueError(f"case {index}: truncated dimensions")
        m, tokens, blocks, case_id = struct.unpack_from("<IIII", data, off)
        off += 16
        if not m or not tokens or not blocks:
            raise ValueError(f"case {case_id}: dimensions must be positive")
        if case_id in ids:
            raise ValueError(f"duplicate case_id {case_id}")
        ids.add(case_id)
        # Bound products before allocating or handing them to Vulkan.
        if m > 1_000_000 or tokens > 1_000_000 or blocks > 1_000_000 or \
                max(m * blocks * 8, m * blocks * 2, tokens * blocks * 8,
                    tokens * blocks, tokens * m) > (1 << 26):
            raise ValueError(f"case {case_id}: dimensions exceed the fixture limit")
        w, wb = take_words(m * blocks * 8, f"case {case_id} W")
        dm, dmb = take_words(m * blocks * 2, f"case {case_id} DM")
        x, xb = take_words(tokens * blocks * 8, f"case {case_id} X")
        e, eb = take_words(tokens * blocks, f"case {case_id} E")
        nout = tokens * m
        need = nout * 12
        if off + need > len(data):
            raise ValueError(f"case {case_id}: truncated expected values/refusals")
        expected = list(struct.unpack_from("<" + "q" * nout, data, off))
        off += nout * 8
        refused = list(struct.unpack_from("<" + "I" * nout, data, off))
        off += nout * 4
        for j, word in enumerate(dm):
            if j & 1:
                continue
            mantissa = struct.unpack("<i", struct.pack("<I", word))[0]
            exp = dm[j + 1]
            if abs(mantissa) > 2047 or (exp != 0xFFFFFFFF and not 8 <= exp <= 37):
                raise ValueError(f"case {case_id}: invalid DM pair at word {j // 2}")
        if any(v != 0xFFFFFFFF and v > 58 for v in e):
            raise ValueError(f"case {case_id}: activation exponent outside 0..58 or zero sentinel")
        if any(flag not in (0, 1) for flag in refused):
            raise ValueError(f"case {case_id}: unknown expected refusal flag")
        packed = wb + dmb + xb + eb
        cases.append({"id": case_id, "m": m, "tokens": tokens, "blocks": blocks,
                      "input": packed, "expected": expected, "refused": refused,
                      "noutputs": nout})
    if off != len(data):
        raise ValueError(f"fixture has {len(data) - off} trailing bytes")
    ok_count = sum(flag == 0 for case in cases for flag in case["refused"])
    refusal_count = sum(flag != 0 for case in cases for flag in case["refused"])
    contract = FIXTURE_PROFILES[profile]
    if len(cases) != contract["cases"] or ok_count != contract["success"] or refusal_count != contract["refusal"]:
        raise ValueError(f"{profile} fixture counts differ: cases={len(cases)}, success={ok_count}, refusal={refusal_count}")
    if not any(value != 0 and flag == 0
               for case in cases
               for value, flag in zip(case["expected"], case["refused"])):
        raise ValueError("fixture needs a successful nonzero CPU observation so the dot mutation is observable")
    digest = hashlib.sha256(data).hexdigest()
    if digest != contract["sha256"]:
        raise ValueError(f"unexpected {profile} CPU fixture identity {digest}; expected {contract['sha256']}")
    print(f"fixture={path} sha256={digest} cases={len(cases)}")
    return cases


def compile_shader(glslc: str, spirv_val: str, source: Path, output: Path) -> None:
    run([glslc, "-O", "--target-env=vulkan1.3", str(source), "-o", str(output)])
    run([spirv_val, "--target-env", "vulkan1.3", str(output)])


def run_cases(driver: Path, shader: Path, cases: list[dict[str, object]], scratch: Path,
              label: str) -> bool:
    all_match = True
    mismatch_count = 0
    mismatch_samples = 0
    for i, case in enumerate(cases):
        inp = scratch / f"{label}-{i}.input"
        out = scratch / f"{label}-{i}.output"
        inp.write_bytes(case["input"])
        run([str(driver), str(shader), str(case["m"]), str(case["tokens"]),
             str(case["blocks"]), str(inp), str(out)])
        raw = out.read_bytes()
        n = int(case["noutputs"])
        if len(raw) != n * 12:
            raise RuntimeError(f"case {case['id']}: driver returned {len(raw)} bytes, expected {n * 12}")
        got = list(struct.unpack_from("<" + "q" * n, raw, 0))
        flags = list(struct.unpack_from("<" + "I" * n, raw, n * 8))
        exp_flags = case["refused"]
        exp_values = case["expected"]
        for j, (value, flag, want, want_flag) in enumerate(zip(got, flags, exp_values, exp_flags)):
            if flag != want_flag or (flag == 0 and value != want):
                mismatch_count += 1
                if mismatch_samples < 8:
                    print(f"PARITY MISMATCH ({label}) case={case['id']} output={j} "
                          f"expected=({want},{want_flag}) actual=({value},{flag})", flush=True)
                    mismatch_samples += 1
                all_match = False
    if mismatch_count:
        print(f"parity-mismatches ({label})={mismatch_count}; samples={mismatch_samples}", flush=True)
    return all_match


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(FIXTURE_PROFILES), default="synthetic",
                        help="registered fixture profile; synthetic stays the default")
    parser.add_argument("--fixture", type=Path, default=os.environ.get("APEIRON_NATIVE_Q8_FIXTURE"))
    args = parser.parse_args()
    if args.fixture is None:
        parser.error("pass --fixture or set APEIRON_NATIVE_Q8_FIXTURE to the actual CPU-produced fixture")
    fixture = Path(args.fixture).expanduser().resolve()
    cases = read_fixture(fixture, args.profile)
    glslc = tool("glslc", "GLSLC", "glslc")
    spirv_val = tool("spirv-val", "SPIRV_VAL", "spirv-val")
    cc = tool("cc", "CC", "cc")
    with tempfile.TemporaryDirectory(prefix="apeiron-q8-native-") as td:
        scratch = Path(td)
        shader = scratch / "q8_tile.spv"
        driver = scratch / "q8_driver"
        print(f"git={run(['git', 'rev-parse', 'HEAD']).stdout.strip()}")
        for name, binary, argsv in (("glslc", glslc, ["--version"]),
                                    ("spirv-val", spirv_val, ["--version"]),
                                    ("cc", cc, ["--version"])):
            print(f"tool={name} path={binary}")
            run([binary, *argsv])
        compile_shader(glslc, spirv_val, SHADER, shader)
        for path in (SHADER, DRIVER, Path(__file__).resolve()):
            print(f"source_sha256 {path.relative_to(ROOT)}="
                  f"{hashlib.sha256(path.read_bytes()).hexdigest()}")
        run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(DRIVER),
             "-lvulkan", "-o", str(driver)])
        baseline = run_cases(driver, shader, cases, scratch, "baseline")
        if not baseline:
            raise RuntimeError("unmodified shader failed CPU fixture parity")
        original = SHADER.read_text()
        needle = "dots[c] += dotPacked4x8EXT(wr[r], xt[t]);"
        if original.count(needle) != 1:
            raise RuntimeError("negative control anchor missing or ambiguous")
        mutated = scratch / "q8_tile_mutated.comp"
        mutated.write_text(original.replace(needle, "dots[c] -= dotPacked4x8EXT(wr[r], xt[t]);"))
        bad_spv = scratch / "q8_tile_mutated.spv"
        compile_shader(glslc, spirv_val, mutated, bad_spv)
        if run_cases(driver, bad_spv, cases, scratch, "mutated"):
            raise RuntimeError("negative control survived: negated signed-dot update matched all CPU observations")
        print("negative-control=red (negated real dot update rejected by CPU/device parity)")
        # Recompile the source as it stands and repeat the positive gate after the mutation.
        restored_spv = scratch / "q8_tile_restored.spv"
        compile_shader(glslc, spirv_val, SHADER, restored_spv)
        if not run_cases(driver, restored_spv, cases, scratch, "restored"):
            raise RuntimeError("restored source failed CPU fixture parity")
        print("restoration=green")
        print(f"gate_elapsed_seconds={time.monotonic() - START_TIME:.3f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"q8_gate: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
