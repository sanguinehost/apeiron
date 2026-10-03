#!/usr/bin/env python3
"""Author the registered int8-coopmat-spirv contract and its proposal (authoring step).

Adds the int8 cooperative-matrix Q8 ``shader-spirv`` contract beside the landed ``q8-tile-spirv``
contract and writes ``tools/dispatch/contracts.json`` bound to both, then writes the cooperative
proposal and refreshes the q8 proposal's registry digest (the registry now carries two contracts).
The cooperative contract's identity map covers the shader execution unit, the reused dispatch
driver and the admission machinery closure the launcher requires; the source-identity gate
``tools/gpu/coopmat_admission_gate.py`` re-derives the SPIR-V and toolchain identity at check time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import author_q8_contract as q8

ROOT = Path(__file__).resolve().parents[2]
CLOSURE = q8.CLOSURE
# The executing source, the reused dispatch driver, and this contract's own gate + identity.
COOPMAT_SHADER_FILES = [
    "src/spirv/int8_coopmat_q8.comp",
    "tools/gpu/q8_driver.c",
    "tools/gpu/coopmat_admission_gate.py",
    "tools/gpu/coopmat_spirv_identity.json",
]
PROOF_FILE = "proof/dispatch/contracts/Int8CoopmatSpirv.lean"
IDENTITY_FILE = "tools/gpu/coopmat_spirv_identity.json"
CONTRACT_KEY = "int8-coopmat-spirv"


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def build_contract() -> dict:
    shader_abi = json.loads((ROOT / IDENTITY_FILE).read_text())["shader_abi"]
    identity = {rel: sha(rel) for rel in [*COOPMAT_SHADER_FILES, *CLOSURE, PROOF_FILE]}
    return {
        "grade": "source-connected-cost-model",
        "source_kind": "shader-spirv",
        "namespace": "Int8CoopmatSpirv",
        "before": "Int8CoopmatSpirv.before",
        "after": "Int8CoopmatSpirv.after",
        "reachable": "Int8CoopmatSpirv.reachable",
        "admissible": "Int8CoopmatSpirv.admissible",
        "before_cost": "Int8CoopmatSpirv.beforeCost",
        "after_cost": "Int8CoopmatSpirv.afterCost",
        "certificate": "Int8CoopmatSpirv.admission",
        "proof_file": PROOF_FILE,
        "build_target": "Int8CoopmatSpirv",
        "shader_source": {
            "shader": "src/spirv/int8_coopmat_q8.comp",
            "entry_point": "main",
            "dispatch_driver": "tools/gpu/q8_driver.c",
            "shader_abi": shader_abi,
            "toolchain": "glslc",
            "target_env": "vulkan1.3",
            "source_gate": "tools/gpu/coopmat_admission_gate.py",
            "identity_file": IDENTITY_FILE,
        },
        "observation": {
            "protocol": "python3 tools/gpu/coopmat_gate.py --profile synthetic --fixture <APEIRON_NATIVE_Q8_FIXTURE>",
            "metric": "device output value and refusal flag parity against the CPU exact-dot-plus-epilogue oracle",
            "interval": "one whole gate invocation",
        },
        "cost": {
            "protocol": "python3 tools/gpu/coopmat_shapes.py --reps 5",
            "metric": (
                "device_ns median (GMAC/s derived) across one dispatch; the registered before "
                "anchor is the measured 909-932 GMAC/s incumbent arithmetic plateau and the "
                "500.7 GMAC/s real-shape peak "
                "(timeout 540 python3 tools/gpu/q8_occupancy.py --controls, exit 0, 11.39 s, "
                "RTX 5080 / Vulkan driver 2580660800)"
            ),
            "interval": "five reps per shape; median",
        },
        "identity": identity,
        "owned_files": ["src/spirv/int8_coopmat_q8.comp"],
        "worker": ["python3", "tools/gpu/coopmat_shapes.py", "--reps", "5"],
    }


def write_proposal(path: Path, key: str, task: str, contract: dict, registry: Path) -> None:
    proposal = {
        "task": task,
        "contract": key,
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "identity": contract["identity"],
        "owned_files": contract["owned_files"],
        "worker": contract["worker"],
    }
    path.write_text(json.dumps(proposal, indent=2) + "\n")


def main() -> int:
    q8_contract = q8.build_contract()
    coopmat_contract = build_contract()
    registry = {
        "schema": 2,
        "contracts": {
            "q8-tile-spirv": q8_contract,
            CONTRACT_KEY: coopmat_contract,
        },
        "reason": (
            "Two source-connected non-Rust contracts over SPIR-V execution units: q8_tile (the "
            "scalar-packed incumbent baseline, dot-only certificate) and int8_coopmat_q8 (the signed "
            "int8 M16N16K32 cooperative-matrix replacement carrying the checked i128 epilogue in its "
            "preservation certificate)."
        ),
    }
    reg_path = ROOT / "tools/dispatch/contracts.json"
    reg_path.write_text(json.dumps(registry, indent=2) + "\n")
    write_proposal(ROOT / "tools/dispatch/q8-tile-spirv.proposal.json", "q8-tile-spirv",
                   "APEIRON-GPU-ADMISSION", q8_contract, reg_path)
    write_proposal(ROOT / "tools/dispatch/int8-coopmat-spirv.proposal.json", CONTRACT_KEY,
                   "APEIRON-COOPMAT-CONTRACT", coopmat_contract, reg_path)
    print(f"REGISTERED q8-tile-spirv {CONTRACT_KEY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
