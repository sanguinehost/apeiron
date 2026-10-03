#!/usr/bin/env python3
"""Author the registered q8-tile-spirv contract and its proposal (authoring step).

Writes ``tools/dispatch/contracts.json`` (schema 2, one shader-spirv contract) and
``tools/dispatch/q8-tile-spirv.proposal.json`` bound to the registry digest. The contract's
identity map covers the shader execution unit and the admission machinery closure the launcher
requires; the source-identity gate re-derives the SPIR-V and toolchain identity at check time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLOSURE = [
    "proof/DispatchAdmission.lean",
    "tools/dispatch/admit.py",
    "tools/dispatch/core.rs",
    "tools/dispatch/main.rs",
    "tools/dispatch/self_test.rs",
    "tools/dispatch/Cargo.toml",
    "tools/dispatch/Cargo.lock",
    "tools/dispatch/run",
    "proof/corpus/lean-toolchain",
    "proof/dispatch/lakefile.lean",
    "proof/dispatch/lean-toolchain",
    "proof/dispatch/regen.sh",
    "proof/dispatch/gate.py",
]
SHADER_FILES = [
    "src/spirv/q8_tile.comp",
    "tools/gpu/q8_driver.c",
    "tools/gpu/q8_admission_gate.py",
    "tools/gpu/q8_spirv_identity.json",
]
PROOF_FILE = "proof/dispatch/contracts/Q8TileSpirv.lean"


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def build_contract() -> dict:
    identity = {rel: sha(rel) for rel in [*SHADER_FILES, *CLOSURE, PROOF_FILE]}
    return {
        "grade": "source-connected-cost-model",
        "source_kind": "shader-spirv",
        "namespace": "Q8TileSpirv",
        "before": "Q8TileSpirv.before",
        "after": "Q8TileSpirv.after",
        "reachable": "Q8TileSpirv.reachable",
        "admissible": "Q8TileSpirv.admissible",
        "before_cost": "Q8TileSpirv.beforeCost",
        "after_cost": "Q8TileSpirv.afterCost",
        "certificate": "Q8TileSpirv.admission",
        "proof_file": PROOF_FILE,
        "build_target": "Q8TileSpirv",
        "shader_source": {
            "shader": "src/spirv/q8_tile.comp",
            "entry_point": "main",
            "dispatch_driver": "tools/gpu/q8_driver.c",
            "shader_abi": "descriptor set 0: binding 0 Weights, 1 Dm, 2 Activations, 3 Exponents; binding 4 Output, 5 Flags; push_constant Params{M,tokens,blocks}; local_size 8x8x1",
            "toolchain": "glslc",
            "target_env": "vulkan1.3",
            "source_gate": "tools/gpu/q8_admission_gate.py",
            "identity_file": "tools/gpu/q8_spirv_identity.json",
        },
        "observation": {
            "protocol": "python3 tools/gpu/q8_gate.py --profile synthetic --fixture <APEIRON_NATIVE_Q8_FIXTURE>",
            "metric": "device output value and refusal flag parity against the CPU oracle",
            "interval": "one whole gate invocation",
        },
        "cost": {
            "protocol": "python3 tools/gpu/q8_shapes.py --reps 5",
            "metric": "device_ns median (GMAC/s derived) across one dispatch",
            "interval": "five reps per shape; median",
        },
        "identity": identity,
        "owned_files": ["src/spirv/q8_tile.comp"],
        "worker": ["python3", "tools/gpu/q8_shapes.py", "--reps", "5"],
    }


def main() -> int:
    contract = build_contract()
    identity = contract["identity"]
    registry = {
        "schema": 2,
        "contracts": {"q8-tile-spirv": contract},
        "reason": "One source-connected non-Rust contract: the q8_tile SPIR-V execution unit, its declared dispatch domain, its observation/cost interface and its shader-to-canonical preservation certificate.",
    }
    reg_path = ROOT / "tools/dispatch/contracts.json"
    reg_path.write_text(json.dumps(registry, indent=2) + "\n")
    proposal = {
        "task": "APEIRON-GPU-ADMISSION",
        "contract": "q8-tile-spirv",
        "registry_sha256": hashlib.sha256(reg_path.read_bytes()).hexdigest(),
        "identity": identity,
        "owned_files": ["src/spirv/q8_tile.comp"],
        "worker": contract["worker"],
    }
    (ROOT / "tools/dispatch/q8-tile-spirv.proposal.json").write_text(
        json.dumps(proposal, indent=2) + "\n")
    print("REGISTERED q8-tile-spirv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
