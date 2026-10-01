# Apeiron to Fold adapter model review

SCOPE: independently review the repaired adapter checker, identities, and contract; run the single requested model gate with controls.
REGISTER: source identity and adapter numeric/layout/state model; excludes implementation, Rust refinement, SPIR-V execution, GPU equality, and performance.
CORPUS: exact pinned Apeiron/Fold source files listed under “Source identity”; no canonical Lean theorem is claimed.
REFERENCE: `artifacts/gpu-prefill/adapter-contract.md`, model-only contract and residual boundaries.
GATE: `timeout 120s python3 artifacts/gpu-prefill/adapter-gate.py --check --controls`, cwd `/tmp/worktree-agent-APEIRON-GPU-PREFILL-W1-T2`; exit 0, 0.059 s reported by gate, 0.024 s wall time.
GRADE: source-bound `anchor` and Python `model`; gate execution is `tested`; Rust/GPU adapter refinement remains open.

## Decision

Accept the repaired gate for the source-bound layout, numeric-conversion, and state model it actually exercises. The requested command passes and all 14 mutations produce their named rejection followed by a restored positive check. Empty-input and allocation-overflow policies remain untested declarations as noted below. The gate is not evidence that an Apeiron-to-Fold production adapter exists or that its GPU result matches Apeiron.

## Source and contract review

The checker derives Apeiron’s root from its own file path and uses that root for `src/...`; it resolves the Fold repository separately. It requires the exact ten-path source set, checks every listed digest against a hardcoded expected digest and the live file, and checks the declared symbols against source text. The adapter identity file pins Apeiron revision `ad707f448e187697af0577dc2c5e5b610409f371` and Fold revision `d497161f32e0d366220f2065ec36e680bd9aed86`; the revisions are descriptive fields, while the checker’s live byte digests are the enforced source identity.

I checked the listed source entry points and the contract’s schema descriptions: Apeiron `q8_0_codes`, `QuantAct`, `quant_act`, and `exact_dot`; Fold’s six-binding DP4A emitter, `GemvSpec`, Q8 row packer, GPU stack, batched RMSNorm, and DeltaNet rule. The contract explicitly leaves the Q8 addition-overflow edge as a panic boundary, Q8 CPU serving fallback, fixed-point versus f32 epilogue equality, and resident state integration open. The model does not claim those source bodies have been linked to a new converter.

The corrected numeric model uses `q + 128` for signed weight lanes, and checks a finite exact f16 pair `(d, -128d)` for its positive witness. It rejects an overflowing offset and an inexact subnormal-scale witness. Activation encoding selects signed QuantAct codes directly for XQ, takes SX from the QuantAct scale, and computes XS only from reconstructed code-times-scale lanes. The original-input XS model and constant-one SX model both fail. This arithmetic is exercised on finite witnesses, not proved for every source input.

The structured checks reject the transposed W and DM addresses, an unpadded token extent, observation of padded outputs, and nonzero signed-lane corruption. The state model requires model, generation, prefix, and capacity together; changes to each owner field fail continuation. Reset is represented by a fresh generation and cleared modeled KV/recurrent arrays. These are executable model checks, not tests of live serving state.

The contract also lists empty-input and checked-size refusal witnesses, but this gate does not execute those refusals. It checks the model fields `empty_policy == "refuse"` and `overflow_policy == "refuse"`; its `SIZE_OVERFLOW_REFUSAL` assertion is only a fixed arithmetic fact, and its shape assertion shows a non-multiple-of-32 `K` alongside a positive batch. Treat empty and allocation-overflow refusal as stated contract policy, not as demonstrated witnesses from this run.

## Gate evidence

Independent command:

```text
timeout 120s python3 artifacts/gpu-prefill/adapter-gate.py --check --controls
exit 0
SOURCE_BOUND_MODEL_CHECK_PASS
CONTROL_RED empty-sources SOURCE_SET_INCOMPLETE
CONTROL_RED wrong-source SOURCE_DIGEST
CONTROL_RED signedness SIGNED_WEIGHT_RECODE
CONTROL_RED weight-stride WEIGHT_STRIDE
CONTROL_RED dm-stride DM_STRIDE
CONTROL_RED padded-extent PADDED_EXTENT
CONTROL_RED padded-observation PADDED_OUTPUT_OBSERVATION
CONTROL_RED original-input-xs XS_SX_RECONSTRUCTION
CONTROL_RED wrong-sx XS_SX_RECONSTRUCTION
CONTROL_RED foreign-generation SESSION_GUARD_DOMAIN
CONTROL_RED stale-prefix SESSION_GUARD_DOMAIN
CONTROL_RED old-dm-contract DM_CONTRACT
CONTROL_RED old-xs-contract XS_SX_CONTRACT
CONTROL_RED old-state-contract CONTINUATION_CONTRACT
GRADE source-bound model; checker tested; Rust/GPU refinement OPEN
```

Each red was followed by `CONTROL_RESTORED` for the same label. This includes the original four bypasses: incomplete sources, wrong DM address, XS from original input, and stale/foreign continuation. The gate digest in `adapter-identities.json` matches the reviewed checker. Its embedded verification record gives Python 3.14.7, the command, prior exit 0, and a 0.1-second rounded time; this review independently observed 0.059 seconds printed by the gate.

## Frozen identity

| Artifact | SHA-256 |
|---|---|
| `artifacts/gpu-prefill/adapter-gate.py` | `2115a62e46a2ad529d7c7bcb15475aaf179981aa71c8cd3c33ffc7a590f8c6ce` |
| `artifacts/gpu-prefill/adapter-identities.json` | `8f7040c36033e77ab3352f42d75756459c41c2cd5c146c29ce8977d210bdb335` |
| `artifacts/gpu-prefill/adapter-contract.md` | `3d0a7441fba1a029a3530dba3779994382f538e22c21d83bd20dd067b1a1a52e` |

| Source path | SHA-256 |
|---|---|
| Apeiron `src/format/quant.rs` | `e574ea15520ed3c17dbfb90d514c2134a3883f681bbfba63b1eb200992f65335` |
| Apeiron `src/engine/chart/mod.rs` | `ae4a4a247f01b8435cb93860554d4b7d795b4c41d804b1d27abe614f23ad68c0` |
| Apeiron `src/engine/kern/gemm.rs` | `44127bd20c1b54ce058b97df16529bc6346a1e22661a45916abbf16dad29d444` |
| Fold `modelfold/src/simt/dp4a.rs` | `ffe4f6c313492b84ecfc2c5d99c9fa8b2945f5bf081282e9b13b1374da50901e` |
| Fold `modelfold/src/simt/superblock.rs` | `e15e18bacfa02b5508dfefc959fc9bd7f2ef1593ff208d8de6e5e3c716745387` |
| Fold `modelfold/src/simt_nn/spec.rs` | `f54e9fe3590aa4c6c87d382a8a4c9b76a56376eda58d212e4f2dd725a2c445b0` |
| Fold `bedrock/src/serve/model/pack.rs` | `b927d380a78eb283abe7ef47631a3d33de48a700febc6fd67f7cc0a8a65e3211` |
| Fold `bedrock/src/serve/gpu.rs` | `1e01838dca24f7751274249af671bb485109b80f8f0e7f9fa940e73744e9b17c` |
| Fold `modelfold/src/simt_nn/batch.rs` | `d183f9cb8decb9ffccb37c91a7973c39003b33361babc8dc6f98615cfc618fd7` |
| Fold `modelfold/src/simt_nn/deltanet.rs` | `c72f1859e82a974339b208ccf6ae67455fa7be73b85f6845887c5d6b69906879` |

## Residual scope

There is no production packer for this mapping, no source-linked conversion or `exact_dot` refinement, no checked whole-call state/reset integration, no emitted-SPIR-V result, and no GPU parity or timing receipt. The next bounded source-linked target is the real `exact_dot` body and its actual `QuantAct`/rounding callees, with a negative control on its multiply or rounding behavior. This review does not register an optimization or alter either repository.
