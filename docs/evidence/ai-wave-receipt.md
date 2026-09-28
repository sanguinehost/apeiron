# Receipt — AI formalization wave (ternary/Hadamard, block grounding, MoE, engine, evaluation)

Barrier run on the dirty tree at `74c037b`. Evidence binds to the digests below, not to HEAD alone.
The prior wave's receipt is `model-algebra-wave-receipt.md`.

## Identities

| item | value |
|---|---|
| extraction | `proof/ApeironWord.lean` `206e96bafc6aaf2b732978d0b227f350e268d653a185eb8138dd6728f4bc32f9`, 24009 lines (was `7b4a0831…` → `72a8b7d4…` at ternary W2 → this at MoE W2) |
| axiom audit | `proof/corpus/AxiomAudit.lean` `b0f4516cc9fc10e638917238b8073f311c10468452fc6b5b203467b14a8a6bd4`, 1501 `#audit` (was 1390) |
| Lean / Mathlib | `leanprover/lean4:v4.31.0`; Mathlib `v4.31.0` |
| Aeneas / Charon | `nightly-2026.09.09-505b6ca` / `0.1.254 (b104e24f)` |

## Barrier gates (lane-locked; each ≤ 540 s)

| command | exit | wall | result |
|---|---|---|---|
| `cd proof/corpus && lake build` | 0 | 401 s | 4115 jobs (cold, after regeneration) |
| `cd proof/corpus && lake build AxiomAudit` | 0 | 4 s | 3985 jobs; every `#audit` inside `{propext, Classical.choice, Quot.sound}` |
| `cd proof && lake build` | 0 | 1 s | 1708 jobs |
| `proof/regen.sh --check` | 0 | 65 s | no drift |
| `cargo test --workspace --all-features --locked` | 0 | 41 s | 434 passed, 0 failed |
| `cargo clippy … -D warnings` | 0 | — | clean (first run red on `moe.rs` `if_same_then_else`; fixed and re-extracted) |
| `cargo fmt --all -- --check` | 0 | — | clean |
| `tools/check-claims.py --check`, `tools/check-kepler.py` | 0 | — | OK / clean |

The lead re-ran every lane's gate, axiom print and seeded control before registration. Controls are
kept under `/home/socol/oc-tmp/`; the re-seeded ones and each red line are in the session record.

## What this wave establishes

| area | headline (grade) |
|---|---|
| low-bit GGUF | Q1_0, Q2_0, PQ2_0, TQ2_0, TQ1_0, PTQ1_0 row laws + refusals (proved); fork-layout vectors (tested) |
| Hadamard | Sylvester algebra and `H_2 = channelPair 1` (proved); `fwht_int_spec`, `fwht_fixed_refines(_sylvester)` — the exact integer FWHT with one proved rounding (proved); loader + folded projection path (tested) |
| block | `matvec_stage_grid_proved`, `rmsVectorStage`, `rotationStage`, `attnHeadStage` (proved); `block_error_bound_grounded` (proved) |
| attention | `attn_domain_v2_structural` — reachability from `KvState::new` + `append` (proved) |
| MoE | `topkSel_spec`, `softmax_topk_renorm`, `moe_not_continuous`, `moe_lipschitz_on_stable` (proved); router (tested); extraction of `route_topk`/`gate_weights` (identity) |
| recurrence / R3 | `recur_eq_scan`, `scan_split`, per-face `*_contraction`, `contraction_orbit`, `bounded_orbit` (proved) |
| speculative decoding | `specStep_eq_target`, `accept_rate` (proved) |
| packing / RoPE / SPIR-V | `decode_block_spec`, `rotate_loop_spec`, `apply_shape_spec` (proved) |
| engine / evaluation | resident weights + row-parallel projections, bit-identical (tested); `lm-eval` parity with the HF reference, 40/40 chosen answers (tested; E1) |

## Findings

- **Vacuity (fixed).** `BlockResidual.mul_ref` pinned the gate-product stage to the global product
  and made `BlockResidual p h` uninhabited for `h ≥ 1`, so the block bounds were vacuous for real
  FFNs.
  - Fix: `mul_ref_on` (`Nous/Model/BlockComposition.lean:384`).
  - Refutation kept: `pinnedMul_stage_impossible`.
  - Real inhabitant and joint witness: `blockResidual_nonempty` and
    `fullBlockGrounded_dom_nonempty_111` (`Nous/Model/BlockMulBox.lean`).
- **MoE discontinuity.** Hard top-k routing has no global Lipschitz constant (`moe_not_continuous`),
  so MoE enters the block only on selection-stable regions.
- **Extraction fragment.** Aeneas rejects index-carried maxima across loop iterations; see
  `docs/epics/moe-router.md` "Extraction note".

## Trust boundary and open obligations

- **Not covered:** rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V execution and the `unsafe`
  container.
- **Kernel refinement still open:** the gate product has none (`mulBoxStage_exact` executes the
  exact product), and the FFN elementwise product is not extracted.
- **Tested, not proved:** `step.rs`, `loader.rs`, `par.rs` and `ffn.rs` are not extracted. The engine
  throughput and folded-path claims are therefore at grade `tested`, and the chunking lemma
  (throughput T2) is open.
- **Open measurements:** W5 of `ternary-hadamard` (Bonsai parity against the fork) and the throughput
  T3 quiet-host measurement.
- **Open proofs:** `checkFragment_sound` (`check_fragment_loop`, `τ`), version-2 tied-output execution,
  and MoE W3-T1 (extracted `route_topk` = `topkSel`).

## Digests (sha256 prefix, lines)

| file | sha256 | lines |
|---|---|---|
| `Numeric/DequantQ1Q2.lean` | `d644691f3c260bef` | 994 |
| `Numeric/DequantQ1Q2Block.lean` | `a69129b28445162d` | 933 |
| `Numeric/DequantTQ2_0.lean` | `e29db00c851ca1c8` | 1056 |
| `Numeric/DequantTQ1_0.lean` | `c2c0f91afed15868` | 913 |
| `Numeric/DequantTQ1_0Row.lean` | `d1d8d3c8bdab03ae` | 1244 |
| `Numeric/DequantPTQ1_0.lean` | `f93a4f904c27a02f` | 1461 |
| `Numeric/DequantLowBitRefusals.lean` | `e79208a5efcd9b2b` | 612 |
| `Numeric/Sylvester.lean` | `9be3a31b72180fdd` | 229 |
| `Numeric/HadamardStage.lean` | `f347efc72754cad3` | 722 |
| `Numeric/HadamardRefinement.lean` | `c70826ccdc28c147` | 847 |
| `Numeric/HadamardFixed.lean` | `0a3a9a192e49b53f` | 731 |
| `Numeric/HadamardFixedSpec.lean` | `044442068f894da1` | 668 |
| `Numeric/MatvecBridge.lean` | `a531a1876765bdaf` | 187 |
| `Numeric/MatvecBridgeGrid.lean` | `cac665c2fbcb5167` | 229 |
| `Numeric/KvReachable.lean` | `463c08297b122b91` | 717 |
| `Numeric/RopeRotate.lean` | `de70d4dac8569993` | 533 |
| `Numeric/PackingDecodeFields.lean` | `9139359647fde2a4` | 1071 |
| `Numeric/PackingDecodeBlock.lean` | `649e41f4ca17265e` | 467 |
| `Nous/Model/FaceRecurrence.lean` | `d23759de401d92b0` | 257 |
| `Nous/Model/FaceContraction.lean` | `faadcfcd4076d48d` | 466 |
| `Nous/Model/SpeculativeAccept.lean` | `4c26710142cfedc1` | 174 |
| `Nous/Model/RmsVectorStage.lean` | `3641b8909db15ec0` | 487 |
| `Nous/Model/RotationStage.lean` | `b625667403fd6419` | 310 |
| `Nous/Model/BlockGrounded.lean` | `0ff3401f1225243e` | 386 |
| `Nous/Model/BlockGroundedWitness.lean` | `bb60a6c5b266a166` | 525 |
| `Nous/Model/BlockGroundedWitness1.lean` | `14c0a50a68ea4bcd` | 611 |
| `Nous/Model/BlockMulBox.lean` | `eb43e1c6ae71e34f` | 468 |
| `Nous/Model/AttnStage.lean` | `02e1a44dfe8004c0` | 707 |
| `Nous/Model/AttnStageWitness.lean` | `540c9bcca82c2b9d` | 277 |
| `Nous/Model/TopKRouter.lean` | `fbca3c7efc65baf9` | 371 |
| `Nous/Model/MoeStage.lean` | `c0732eba6c4c8a95` | 720 |
| `Nous/Model/BlockComposition.lean` | `05585258ebaa9e3a` | 551 |
| `Spirv/CheckApplyShapeAssembly.lean` | `d91c8248fa49a942` | 398 |
| `src/engine/dequant_lowbit.rs` | `49cd6385a9d07352` | 914 |
| `src/engine/hadamard.rs` | `3ea1ed17a356c6a3` | 452 |
| `src/engine/moe.rs` | `899500f18eae9763` | 956 |
| `src/engine/par.rs` | `cb2e34c827bf7bd6` | 204 |
| `src/engine/step.rs` | `14f8bc04fb345012` | 1394 |
| `src/engine/loader.rs` | `a6a839a2bd125279` | 1149 |
| `src/engine/dequant.rs` | `8ecee4cbbe5363bd` | 576 |
| `src/format/quant.rs` | `e574ea15520ed3c1` | 897 |
| `examples/lm_eval_bridge.rs` | `7d453854514903a3` | 161 |
| `proof/extract/lib.rs` | `bd559395981da293` | 944 |
| `proof/regen.sh` | `7f6f0d29af9f45bd` | 230 |
