#!/usr/bin/env python3
"""Source-identity gate for a non-Rust (shader/SPIR-V) executing source.

This is the non-Rust analogue of ``proof/regen.sh --check``: it re-derives the executing
identity of the registered shader execution unit from the real source and refuses when any
byte of the shader, the dispatch driver, the toolchain or the emitted SPIR-V drifts from the
committed identity. It establishes source identity only; it makes no preservation or
performance claim, and device parity (``tools/gpu/q8_gate.py``) is a separate, tested
observation protocol.

Commands:
  --check      recompile the shader, validate the SPIR-V, compare every identity field,
               print ``ADMISSION_SOURCE_IDENTITY_OK``; non-zero on any drift.
  --emit       recompile and rewrite ``tools/gpu/q8_spirv_identity.json`` from the current
               source and toolchain (authoring step).
  --controls   seed a shader-byte drift, a SPIR-V identity drift and a toolchain drift in a
               throwaway copy; require each to be refused, then require the unmodified
               identity to be accepted.

Exit 0 only on success; every refusal prints ``ADMISSION_SOURCE_STALE <field>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_FILE = ROOT / "tools/gpu/q8_spirv_identity.json"
DEADLINE_SECONDS = 535.0
FIELDS = (
    "shader",
    "shader_sha256",
    "entry_point",
    "shader_abi",
    "dispatch_driver",
    "dispatch_driver_sha256",
    "toolchain",
    "toolchain_version",
    "target_env",
    "spirv_sha256",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toolchain_version(toolchain: str) -> str:
    """First line of ``<toolchain> --version``; the full report is not part of identity."""
    cp = subprocess.run([toolchain, "--version"], text=True, capture_output=True, check=True)
    return cp.stdout.splitlines()[0].strip()


def compile_spirv(shader: Path, glslc: str, target_env: str, out: Path) -> None:
    cp = subprocess.run(
        [glslc, "-O", f"--target-env={target_env}", str(shader), "-o", str(out)],
        text=True, capture_output=True, timeout=DEADLINE_SECONDS,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"ADMISSION_SOURCE_COMPILE_REJECT {cp.stderr.strip()}")
    cp = subprocess.run(
        ["spirv-val", "--target-env", target_env, str(out)],
        text=True, capture_output=True, timeout=DEADLINE_SECONDS,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"ADMISSION_SOURCE_VALIDATION_REJECT {cp.stderr.strip()}")


def derive(root: Path, identity: dict[str, str], scratch: Path) -> dict[str, str]:
    """O(B) hashing plus one bounded glslc/spirv-val compile; no GPU access."""
    shader = root / identity["shader"]
    driver = root / identity["dispatch_driver"]
    glslc = identity["toolchain"]
    target_env = identity["target_env"]
    spv = scratch / (shader.stem + ".spv")
    compile_spirv(shader, glslc, target_env, spv)
    return {
        "shader": identity["shader"],
        "shader_sha256": sha256(shader),
        "entry_point": identity["entry_point"],
        "shader_abi": identity["shader_abi"],
        "dispatch_driver": identity["dispatch_driver"],
        "dispatch_driver_sha256": sha256(driver),
        "toolchain": glslc,
        "toolchain_version": toolchain_version(glslc),
        "target_env": target_env,
        "spirv_sha256": sha256(spv),
    }


def compare(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    """Field-wise equality; every mismatch is named for attributable refusal."""
    return [f for f in FIELDS if expected.get(f) != actual.get(f)]


def check(root: Path, identity: dict[str, str]) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="apeiron-q8-identity-") as td:
        actual = derive(root, identity, Path(td))
    return compare(identity, actual)


def load_identity(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    missing = [f for f in FIELDS if not isinstance(data.get(f), str) or not data[f]]
    if missing:
        raise RuntimeError(f"ADMISSION_SOURCE_IDENTITY_REJECT missing {missing}")
    return data


def resolve_identity(root: Path, override: Path | None) -> Path:
    if override is None:
        return root / IDENTITY_FILE.relative_to(ROOT)
    return override if override.is_absolute() else root / override


def do_check(root: Path, identity_path: Path) -> int:
    identity = load_identity(identity_path)
    bad = check(root, identity)
    if bad:
        print(f"ADMISSION_SOURCE_STALE {bad}", flush=True)
        return 1
    print(f"ADMISSION_SOURCE_IDENTITY_OK shader={identity['shader']} "
          f"spirv={identity['spirv_sha256']}", flush=True)
    return 0


def do_emit(root: Path, identity_path: Path) -> int:
    target = identity_path
    identity = load_identity(target)
    with tempfile.TemporaryDirectory(prefix="apeiron-q8-identity-") as td:
        actual = derive(root, identity, Path(td))
    target.write_text(json.dumps(actual, indent=2) + "\n")
    print(f"ADMISSION_SOURCE_EMITTED {target.relative_to(root)}", flush=True)
    return 0


def do_controls(root: Path) -> int:
    """Independently seed three drifts and require attributable refusal, then restore."""
    base = load_identity(root / IDENTITY_FILE.relative_to(ROOT))
    with tempfile.TemporaryDirectory(prefix="apeiron-q8-controls-") as td:
        scratch = Path(td)
        copy = scratch / "repo"
        # A throwaway copy of just the files the gate reads; the working tree is never touched.
        for rel in (base["shader"], base["dispatch_driver"]):
            dest = copy / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / rel, dest)

        cases: list[tuple[str, str, dict[str, str]]] = []

        shader_mutant = dict(base)
        mutated = copy / base["shader"]
        original = mutated.read_bytes()
        # Append a comment: byte-identical compilation semantics, different source bytes.
        mutated.write_bytes(original + b"\n// admission-control-drift\n")
        cases.append(("shader-drift", "shader_sha256", shader_mutant))

        spv_mutant = dict(base, spirv_sha256="0" * 64)
        cases.append(("spirv-drift", "spirv_sha256", spv_mutant))

        tc_mutant = dict(base, toolchain_version="0.0.0")
        cases.append(("toolchain-drift", "toolchain_version", tc_mutant))

        for name, field, ident in cases:
            bad = check(copy, ident)
            if field not in bad:
                raise RuntimeError(
                    f"ADMISSION_CONTROL_SURVIVED {name}: expected {field}, got {bad}")
            print(f"ADMISSION_SOURCE_REJECT {name} {bad}", flush=True)
            # Isolate each drift: the shader copy is restored before the next case.
            mutated.write_bytes(original)

        mutated.write_bytes(original)
        bad = check(copy, base)
        if bad:
            raise RuntimeError(f"ADMISSION_SOURCE_RESTORE_REJECT {bad}")
    print("ADMISSION_SOURCE_CONTROL_OK drift=3 restored=1", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--identity", type=Path, default=None,
                        help="identity JSON, relative to --root (default tools/gpu/q8_spirv_identity.json)")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    identity_path = resolve_identity(root, args.identity)
    # glslc/spirv-val are named by the committed identity; fail closed if absent.
    if shutil.which("glslc") is None or shutil.which("spirv-val") is None:
        print("ADMISSION_SOURCE_TOOLCHAIN_MISSING", file=sys.stderr)
        return 1
    if args.emit:
        return do_emit(root, identity_path)
    if args.controls:
        return do_controls(root)
    if args.check:
        return do_check(root, identity_path)
    parser.error("one of --check, --emit, --controls is required")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"q8_admission_gate: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
