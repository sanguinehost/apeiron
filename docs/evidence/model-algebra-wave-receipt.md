# Receipt — model-algebra closing wave (attention end to end, dequant faces, block stages, SPIR-V scans)

Barrier run on the dirty tree at `74c037b`. Evidence binds to the digests below, not to HEAD alone.

## Identities

| item | value |
|---|---|
| source revision | `74c037b` (dirty; untracked wave files listed below) |
| extraction | `proof/ApeironWord.lean` `7b4a0831ed7cb092c1f065cb5b8db1f959613fc03773cddf631150d1ac67bf98`, 21265 lines (unchanged by this wave) |
| axiom audit | `proof/corpus/AxiomAudit.lean` `85c358a843d87bb2549b37edd278206e38404547e694dec02ac8069b67849854`, 1390 `#audit` (was 1255) |
| Lean / Mathlib | `leanprover/lean4:v4.31.0`; Mathlib `v4.31.0` (`proof/corpus/lake-manifest.json`) |
| Aeneas / Charon | Aeneas `nightly-2026.09.09-505b6ca` (`proof/README.md:567`); Charon `0.1.254 (b104e24f)` (`target/aeneas/charon version`) |

## Barrier gates

Each command below ran lane-locked with a 540 s bound. The Lean commands also ran with
`LEAN_NUM_THREADS=1`.

| command | exit | wall | result |
|---|---|---|---|
| `cargo test --workspace --all-features --locked` | 0 | 46 s | 387 passed, 0 failed (22 result blocks) |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 | 1 s | clean; the log carries a `duplicate key` manifest diagnostic from the `fold` git dependency (`opsfold/Cargo.toml:61`), not this crate |
| `cargo fmt --all -- --check` | 0 | 0 s | clean |
| `AENEAS_HOME=…/target/aeneas proof/regen.sh --check` | 0 | 52 s | no drift; 1708 jobs |
| `cd proof && lake build` | 0 | 1 s | 1708 jobs |
| `cd proof/corpus && lake build` | 0 | 3 s | 4084 jobs (warm) |
| `cd proof/corpus && lake build AxiomAudit` | 0 | 3 s | 3954 jobs; every `#audit` inside `{propext, Classical.choice, Quot.sound}` |

`AxiomAudit` control: a scratch copy of the `#audit` elaborator on a theorem built from a fresh
`axiom` fails with `depends on axioms outside the closed cone … [fakeAx]`.

## Headlines (proved) and controls

The lead re-ran each gate and each axiom print. Every control ran in a throwaway copy under
`/home/socol/oc-tmp/`, not in the tree. A control is marked "re-seeded" where the lead rebuilt a
mutant the worker had restored.

| module | headline(s) | control (red at) |
|---|---|---|
| `Numeric/AttnHeadRefines.lean` | `attn_head_refines_v2_proved : attn_head_refines_v2`, with no hypothesis beyond `attn_domain_v2` | drop the `n·V/(2q)` term (`:309`); drop the `exp(−16)` saturation term from `softmaxPerturbBound` (`:314`) |
| `Numeric/AttnHeadWindowsScore.lean` | `attn_v2_prefix`, `attn_v2_loop0_domain`, `attn_v2_loop1_domain`, `attn_loop0_ok`, `attn_loop1_ok`, `attn_loop_stage_facts` | score window 2^62 → 2^63 (`:291`, `:310`) |
| `Numeric/AttnHeadWindowsValue.lean` | `attn_weight_bounds` (`0 ≤ W ≤ 2^32`, `ΣW ≤ 2^32+n`), `attn_weight_window_of_numerators`, `plane_at_spec`, `attn_value_fit`, `attn_value_window_of_v2`, `attn_value_components` | `ΣW ≤ 2^32` (`:109`) |
| `Numeric/AttnScoreStage.lean` | `attn_scale_bound`, `attnScoreInt_err`, `attn_score_stage` | drop the epilogue `1/(2q)` (`:258`) |
| `Numeric/EngineLmHeadRefinement.lean` | `lm_head_refines_proved` | inherits `matvec_refines_proved`'s control (one-line instantiation) |
| `Numeric/SiluSaturation.lean` | `silu_high_real`, `silu_low_real`, `silu_refines_total` (every `I64` input) | `16e^-16 → 15e^-16`, re-seeded (`:99`) |
| `Numeric/RmsNormIdeal.lean` | `rmsMean_real_bound`, `inv_sqrt_lipschitz`, `rms_norm_refines_ideal` | drop the mean-rounding term (`:286`) |
| `Nous/Model/BlockGrid.lean` | `Stage.ofGrid`, `siluStage`, `rmsStage` (explicit grid domains), `block_error_bound'`, `fullBlock'_delta_eq`, domain witnesses | delta at `siluErrTotal 0`, re-seeded (`:166`) |
| `Numeric/DequantQ8_0.lean` | `dequant_q8_0_row_spec`, `dequant_row_fixed_q8_0`, witness | drop `d`, re-seeded (`:263`) |
| `Numeric/DequantQ5_0.lean` | `dequant_q5_0_row_spec`, `q5_0_codes_ok`, witness | `qh` bit `j → j+1`, re-seeded (`:118`) |
| `Numeric/DequantQ2K.lean` | `dequant_q2_k_row_spec`, `q2_k_block_spec`, witness | drop `dmin·min` (`:993`) |
| `Numeric/DequantQ3K.lean` + `DequantQ3KBlock.lean` | `dequant_q3_k_row_spec`, `q3_k_block_ok`, witness | sub-block `/16 → /8` (`:1259`, pre-split) |
| `Numeric/DequantRefusals.lean` | 18 refusal equalities (short payload, non-finite `d`/`dmin`) for F16/Q4_K/Q6_K; overflow arm shown unreachable at block 0 | wrong scale offset, re-seeded (`:359`) |
| `Numeric/PackingEpilogueFold.lean` | `epilogue_loop2_fixed_spec`, `epilogue_loop1_fixed_spec`, `epilogue_entry_fixed_min` | drop `bias`, re-seeded (`:286`) |
| `Numeric/RopeHeadWitness.lean` | `rope_head_domain_witness`, `rope_head_witness_ok` (non-vacuity at head `[2^32, 0]`) | head of length 1 (`:143`, `:208`) |
| `Spirv/CheckScan.lean` | `ty_get_spec`, `defs_contains_spec`, `ty_get_eq_lookupTy_of_nodup` | reverse lookup (`:245`, `:338`) |
| `Spirv/CheckScanState.lean` | `decoration_admitted_spec`, `storage_class_admitted_spec`, `resolve_tag_spec`, `ty_insert_spec`, `apply_shape_loop0_body_{cont,undefined}` | drop decoration 34, re-seeded (`:57`) |
| `Spirv/CheckApplyShape.lean` | `apply_shape_scan_spec`, `apply_shape_loop{0..7}_spec` | last instead of first undefined id (`:252`) |
| `Numeric/Sylvester.lean` | `sylvester_mul_self`, `sylvester_transpose`, `sylvester_one_channelPair`, `sylvester_mulVec_butterfly`, `two_not_isUnit_int` | bottom-right block `+H_k`, re-seeded (`:60`, `:143`, `:182`) |

Registered earlier in this wave and re-verified at the barrier:
- `AttnSoftmaxPerturb`, `AttnLoopScore`, `AttnLoopOutput`;
- `DequantRefinement`, `DequantQ4K(Block)`, `DequantQ6K`, `DequantKWitness`;
- `PackingUnpack`, `RopeHeadContent`, `BlockComposition`.

`DequantQ3K`/`DequantQ4K` were split under the 1500-line cap. The name sets are identical
(70 = 70 and 76 = 76), and all dependents rebuild.

## Contract and name changes

- **`attn_head_refines_v2` adopted and proved.** The planned `|W_s| ≤ 2^33` amendment was not
  needed: `attn_weight_bounds` derives `0 ≤ W ≤ 2^32` from `E_j ≤ T`, so `attn_domain_v2` stands as
  stated.
- **`Spirv.gpu_dp4a_refines_cpu_spec` renamed to `Spirv.gpu_dp4a_word_refines_cpu_word`.** It proves
  the word-model refinement. The contract name is reserved for the unproved
  `denote (decode ws) = dp4a_gemv_spec …`.
- **Descriptive drift corrected** in:
  - `AttnSoftmaxPerturb.lean` header and docstrings;
  - `ErrorComposition.lean` (the residual `Layer` set is "not built from a refinement in this
    file");
  - `EngineSiluRefinement.lean` header;
  - `PackingRefinement.lean` residual list;
  - `docs/epics/gpu-spv.md` evidence cells;
  - `docs/epics/gpu-spv.md` name note.

## Trust boundary

The proofs reach the Aeneas extraction of the real source and the canonical definitions they name.
They do not cover:
- rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V execution or the `unsafe` container;
- the GGUF reader beyond its tested boundary;
- the block-level `matvecEps` of `BlockComposition`, which remains `model` grade: no Lean term
  links `Slice I64` matvec to `Fin m → Fin n → ℝ`.

`attn_domain_v2`'s reachability from `KvState::new` + `append` is `anchor`; its witness is a
hand-built state.

## Open obligations (discharge queue)

| obligation | grade now | next gate |
|---|---|---|
| owned-file cap: `AxiomAudit.lean` 1790 (was 1596 before the wave), `EngineAttnRefinement` 1510, `FormatRefinement` 2054, `Growable/Scan` 2055, `Native/LocatorSpec` 1869, `src/format/packing.rs` 1751 | open | split per file; name-set equality as control |
| slice ↔ `Fin`-matrix matvec bridge for `BlockComposition` | model | new bridge module |
| vector RMSNorm stage whose radicand is the rounded mean of the input | open (scalar stage + ideal bound proved) | `Nous/Model/BlockGrid` extension |
| `checkFragment_sound` (restated `∃ m, decode ws = some m`): `apply_shape` assembly, `check_fragment_loop` invariant, `τ` tag translation | open | `Spirv/` next slice |
| `packing` `decode_block_fields` composition over `unpack_code_ref`; top-level `epilogue` | open | `Numeric/` next slice |
| RoPE outer `rotate` loop and refusal arm | open | `Numeric/` next slice |
| G1 performance: 325 passing / 1125 failing pairs | open | exclusive quiet host only |
