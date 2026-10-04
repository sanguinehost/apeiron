# `Icarus` — SHA-256 / SHA-256d rooting library

44 tracked Lean modules ported from the `icarus` staging campaign
(`~/icarus/work/proofs/`) into the apeiron corpus. The goal is a *rooted* reading
of SHA-256: the round step realised over the trichotomy carrier `R[x]/(x² − σ)`, at declared widths
and faces, with the algebraic obstructions proved rather than assumed.

## Two jumps

| want | go to |
|---|---|
| **the keystone theorem** | `Keystone/RootedSha256.lean` → `AlgebraRootedSha256.sha256_round_rooted` |
| **the cost model** | `Cost/CostAlgebra.lean` → `CostAlgebra.Cost`; its image under the round plan is `Cost/AlgebraCostBridge.lean` → `shaToCost_*` |

`sha256_round_rooted` is one conjunction of nine named clauses (`RoundValue`, `ProductPrimitives`,
`CarryPrimitives`, `WirePrimitives`, `ReadoutPrimitives`, `FaceVerdict`, `RefutationWall`,
`CostClause`, `Witnesses`). It re-proves nothing: each clause is a bound component applied.

## Build

`Icarus` is **not** a default target (`lakefile.lean:142` declares `lean_lib Icarus` without
`@[default_target]`), so a bare `lake build` does not build it.

```sh
cd proof/corpus
../../../../bin/lane lake build Icarus                      # whole library
../../../../bin/lane lake build +Icarus.Keystone.RootedSha256   # one module
../../../../bin/lane lake env lean /abs/path/Icarus/Foo.lean  # one file
```

`bin/lane` is the only sanctioned Lean lane: serialized on `/tmp/apeiron/apeiron-lane.lock`,
`MemoryMax=3800M`, 590 s budget. Single-module gates are written `+Icarus.<Module>`.

## Evidence convention

**This README carries no claims.** Every claim about an `Icarus` module is graded in the *workspace*
receipts, not here — `receipts/*.md` and `work/proofs/*.md` under `~/icarus/`,
each bound to an ` ```evidence ` block (8 keys) naming the claim, grade, scope, source digest and a
**durable** log under `receipts/logs/review/`. The convention is `receipts/EVIDENCE-CONVENTION.md`;
the checker is `work/check_receipts.py` (`--check`, `--strict`, `--tracked-only`).

The corpus module is the *source* named in such a block; the staging file under `work/proofs/` is
what the receipt's prose is written against. Port receipts carry the module digest, the gate rc, the
job count, the in-cone `#print axioms` row count and `sorryAx 0`.

## Integration status

* the port is **pushed**: `origin/main` is `2d7e569` (22 commits rebased onto `688f041`, post-rebase
  barrier green at 3348 jobs, synced to the workspace checkout). The local campaign checkout stands
  ahead by the docs-preserve + epic-status commits, deliberately unpushed.
* `Extraction/{Schedule,Compression,Store}.lean` import `ApeironWord`, the **generated** extraction
  file. `Extraction/DenotesPrep.lean` deliberately does not: it imports
  `Icarus.Composition.AlgebraSha256dCompose` instead, and carries `WordSha256DenotesAll` as a
  hypothesis that is **named and never asserted**.
* **Regen window**: `repos/apeiron/the icarus-regen-manifest note is the exclusive handoff
  for `proof/regen.sh` to emit `word_sha256d` into `proof/ApeironWord.lean`. It has not been run.
  The insert sits *before* the trailing chacha items, so the regen re-binds the AR-14/31/40/73 pins
  in the same commit; manifest §3 is already discharged.
* The keystone module's header still marks AR-36, AR-33 and AR-58 `[NOT YET IN THE CORPUS]`. That is
  **stale**: `AlgebraRotBridge32`, `RingModelWeb` and `Cost/AlgebraCostBridge` all landed in
  `6d53961`, `5b643d0` and `dd2bd71`. The import list at the top of the file is the authority.
* `Cost/AlgebraCost.lean` and `Cost/AlgebraCostKeccak.lean` landed 2026-10-04 (the W4 gap
  DOC-1 found; both tracked and indexed in the Cost group).

## Modules

### 1 · Carry / degree / shift core, faces, and the rotation cluster (20)

`AlgebraCarry.lean` — carries in `R[x]/(x²−σ)`: `carry_identity_two`/`_four`/`_eight_instance`,
`ringUpper_parabolic_blind_two`/`_four`, `deviation_witness_k2`/`_k4`/`_k8`,
`no_channel_elliptic`, `channel_read_shape`.

`AlgebraCarryGeneral.lean` — the same carry identity at parametric width, plus total degree:
`denoteBV_carry`, `carryCell_eq_ringMul_sum`, `totalDegree_carryPoly_le`,
`residual_recomposes_sum_two`/`_four`/`_eight`.

`AlgebraCarryComputable.lean` — a computable ANF over `F₂` for modular addition: `carryPoly_one`,
`bitPoly_eval_two`/`_four`, `anfSumBV_two`/`_four`, `carry_identity_two`/`_four`, `anfDegree_bitPoly`.

`AlgebraCarryTermCount.lean` — AR-62, the exact ANF term count at every width: `carryPoly_card`,
`bitPoly_card_two`/`_four`, `anfTermCount_eq_card`, `anfTermCount_bitPoly`.

`AlgebraShrExact.lean` — when `x >>> n` has a cell/algebraic reading: `shr_exact_amounts`,
`shr_not_family`, `shr32_3_not_family`, `shr32_10_not_family`, `shr32_3_not_ringpoly`,
`shr32_10_not_ringpoly`.

`AlgebraShrExactOpen.lean` — AR-43, AR-23's open obligations in a separate file so AR-23's source
binding stays valid: `shr_agree_inj`, `shr32_3_agreement_inj`, `shr32_10_agreement_inj`,
`shr_agree_multiples_not_family`.

`AlgebraSigmaHalves.lean` — the forced bit-halves pairing, and which composites span it:
`span_rotr_iff`, `span_shr_iff`, `halves_multiplier_norm_form`, `bigSigma0_not_span`,
`composites_not_span_at_32`, `c5RoundOps_eq_ar15`.

`AlgebraRotorSha.lean` — is SHA-256 rotation the elliptic face? `verdict_elliptic`,
`wraparound_is_the_elliptic_face`, `elliptic_read_two_slot`, `no_sqrt_neg_one_width4`,
`sha256_amounts_split`.

`AlgebraRotorUnit.lean` — is the `n > k/2` failure a real boundary or an artefact of AR-17's two
pairings? `verdict_unitary_beyond_half`, `halfPair_reproduces_iff`, `halfPair_family_add`,
`singular_pairing_reproduces_width2`, `rope_no_rotr_mult`.

`AlgebraRotorShift.lean` — the cyclic shift diagonalised by Fourier modes, norm-one rotors:
`rotorPow_order`, `period_eq_rotor_order`, `inst4_verdict`, `inst8_verdict`, `dft_collision_ZMod4`,
`no_decoder_ZMod4`.

`AlgebraRotate32.lean` — rotation at the SHA-256 word width, and exactly where the two-slot reading
fails: `rotateRight_eq_rotateLeft32`, `no_affine_reads_rotation32`,
`no_read_computes_rotation32`, `read_family_additive32`, `sigma0_32_rotation_terms_not_affine`.

`AlgebraRotExact.lean` — the closed-form exactness set for the algebraic rotation reading:
`exact32_iff_offset_iff`, `sha2_rotation_amounts`, `Sigma0_enters_exact10`, `Sigma1_enters_exact7`,
`exact10_neither_confinement`, `joint_not_injective_not_exact`.

`AlgebraRotBridge32.lean` — AR-36: no equivariant multiplier repairs the halves pairing at width 32.
`no_equivariant_multiplier32`, `bridge_verdict`, `read_refinement32`, `rotr32_eq_rotateRight`,
`action_fails`, `composition_breaks`.

`AlgebraBoolCarrier.lean` — the Boolean carrier `B_k = Fin k → ZMod 2`, packing and the four wiring
links: `xor_wiring`, `and_wiring`, `not_wiring`, `unpack_rotr_wiring`, `wiringCarry_and_coord`,
`carry_identity_eight_instance`, `residual_recomposes_eight_instance`.

`AlgebraBoolCarrier32.lean` — the same at width 32, with the FIPS ROTR/SHR amounts:
`pack_unpack_32`, `xor_wiring_32`, `wiring_rotr_7_32`/`_18_32`/`_17_32`/`_19_32`,
`wiring_shr_3_32`/`_10_32`, `residual_ne_overflow_k32`, `sched3_s3_inv_32`.

`AlgebraBoolCarrier32Round.lean` — AR-60, the round step on the width-32 carrier (the keystone's
value and carry clauses): `carrierRound_coord`, `round_products_face_free`, `compress_eq_rounds`,
`roundStep_eq_fun`, `carrierT1Stage_inv`, `carrierRound_zero`.

`RingModelWeb.lean` — AR-33, the implication web over the ring models:
`no_read_computes_rotation`, `rotateLeft_denotation_not_additive`, `face_elliptic_no_root`,
`c5_refuted_at_both_widths`.

`AlgebraFaceComposition.lean` — AR-19 item 1, the per-face capability table:
`ringMul_dial_only_in_first_slot`, `no_root_neg_one`, `readBlind_iff`, `norm_hyperbolic_separates`,
`rotCell_not_additive`.

`AlgebraFaceCompositionRound.lean` — AR-19 item 3, the single-face verdict for one SHA-256 round:
`one_face_serves_the_round`, `chi_term_face_free`, `rotation_not_ringMul`, `rotWrap_not_additive`.

`AlgebraFaceCompositionSwitch.lean` — AR-19 item 2, what mediates a face switch:
`elliptic_face_no_hom_out`, `elliptic_no_hom_to_parabolic`, `elliptic_no_hom_to_hyperbolic`,
`FaceHom_root_necessary`, `mod4_no_norm_matching`.

### 2 · SHA-256 width-32 reads (2)

`Sha256/ReadTrichotomy32.lean` — AR-12's read-only trichotomy instances at width 32, residuals in
place: `read_trichotomy_32`, `parabolic_root_32`, `hyperbolic_root_32`,
`no_channel_root_elliptic_32`, `faces_read_agree_32`, `rotResidual32_ne_zero`.

`Sha256/CarrierSigma32.lean` — AR-68 residual R1, the `zxor3` composition of the six SHA-2 amount
wires on `BoolRing 32`: `zxor3_is_addition`, `sigma0Cell32_eq_sha2`, `sigma1Cell32_eq_sha2`,
`Sigma0Cell32_eq`, `Sigma1Cell32_read_eq_unpack`.

### 3 · Composition (6)

`Composition/AlgebraSha256d.lean` — the composition-level algebra statement for SHA-256d and the
iteration verdict on the round obstruction: `sha256d_is_composition`, `cong_preserves`,
`rot_not_cong_pres`, `shr_not_cong_pres`, `Sigma0_not_cong_pres`, `Sigma1_not_cong_pres`.

`Composition/AlgebraSha256dCompose.lean` — the canonical block reader in public form (AR-40; the
single source `Extraction/DenotesPrep` now imports): `hash_outer_eq_compress_H0`,
`sha256d_eq_outer_compress_H0`, `bytesToWords_eq_blockOf`, `padBlock32_eq_paddedBlock`.

`Composition/AlgebraSha256dBlockArray.lean` — the 64-byte canonical block array:
`sha256d_eq_outer_compress_bytesToWords_blockArray`, `blockArray_length`,
`hash_outer_eq_compress_padBlock32`.

`Composition/Sha256BE.lean` — the four-byte big-endian decoding bridge: `from_be_bytes_make4`,
`beWord16_eq_beWord4`, `fromBEBytes_toNat_four`, `from_be_bytes_asym`.

`Composition/SpecImplicationWeb.lean` — padding/rate/domain equalities across the SHA-3/SHAKE
specification web: `padB_length`, `zeroFill_boundary_55`/`_56`, `rate_sha3_256_eq`,
`sha3_256B_eq_sha3_256`, `domain_sha3_eq`.

`Composition/SpongeBlocks.lean` — the sponge block structure over `Crypto.Keccak`
(`sponge_loop0_sha3_256`, `absorbBlocks_succ`, `rate_136`).

> The last two are carried specification context, **not** inputs to `sha256_round_rooted`.

### 4 · Cost (11)

`Cost/CostAlgebra.lean` — **the cost model.** The vendored `CostAlgebra.Cost` record (seven
resources: `ops`, `pack`, `alloc`, `xfer`, `dispatch`, `peak`, `path`) with componentwise dominance
`costLe`, sequence/layer composition, and the strict-`Nat` improvement order: `Cost`, `costLe`,
`dominates_improves`, `scalar_not_sufficient`, `forked_critical_path_reduced`.

`Cost/AlgebraCostBridge.lean` — AR-58, the map from the algebra round plan into that record:
`shaToCost_monotone`, `shaToCost_roundCost`, `shaToCost_attemptCost`, `shaToCost_rangeCost`,
`shaToCost_threeFaces`, `shaToCost_attemptCost_faceBlind`.

`Cost/AlgebraCostSource.lean` — the source-bound one-round field figure (the keystone's cost
clause): `roundCost_fields`, `loadCost_fields`, `compressionCost_eq`, `blockAdds_eq`,
`blockDialMuls_eq`, `storeAddrOps_eq`, `executing_dial_mul_zero`.

`Cost/AlgebraCostRound.lean` — the per-round chain cost: `roundCost_fields`, `roundChainCost_eq`,
`blockCost_split`, `roundCost_faceBlind`, `rotationReading_needs_no_root`.

`Cost/AlgebraCost.lean` — the round cost core: `ringUpperCost`, `ringNormFormCost`,
`rotationCost_fields`, `chainMulCost_fields` (W4-gap port, 2026-10-04).

`Cost/AlgebraCostKeccak.lean` — the Keccak cross-primitive counts: `columnCost_fields`,
`thetaCost_fields`, `keccak_round_face_blind` (W4-gap port, 2026-10-04).

`Cost/AlgebraCostCompletePart1.lean` — part 1 of the complete-cost split: `roundCost_fields`,
`roundChain64`, `roundChainCost_rep`, `blockCost_eq`, `roundCostTwoSlot_fields`.

`Cost/AlgebraCostCompletePart2.lean` — part 2, the padding and message arithmetic:
`pad_spec`, `blockCount_eq`, `ceilDiv64_spec`, `loadCost_fields`, `storeCost_fields`,
`attemptCost_threeFaces`, `rangeCost_nonceSpace_eq_pow2`.

`Cost/AlgebraCostPow.lean` — the proof-of-work cost in the same measure: `attemptCost_two`,
`attemptCost_is_two_blocks`, `blocksCost_eq`, `messageCost_eq_blocks`, `blockCount_million`,
`faceCertifyCost`.

`Cost/AlgebraCostPowComparePart1.lean` — the source side of the pow-vs-SHA comparison:
`sha256dAttemptCost_eq`, `blockCost_ranking`, `sha256dAttemptCost_inner_outer`,
`sha256dRangeCost_nonceSpace_eq_pow2`.

`Cost/AlgebraCostPowComparePart2.lean` — the certificate side and the divergence:
`sha256dAttemptSha_eq`, `sha256dAttemptShaSource_divergence`, `carryStageInventory`,
`sha256dAttemptSha_threeFaces`.

### 5 · Extraction (4)

`Extraction/Schedule.lean` — the message-schedule loop mirror and its `rfl` binding:
`loopRows_eq_canonical`, `loopRows_49_eq_canonical`, `derived_cover`, `derived_unique`,
`loaded_rows_never_rewritten`, `blockLoad_id`.

`Extraction/Compression.lean` — the 64-round compression loop: `compress_eq_rounds`,
`roundLoop_spec`, `roundLoop_eq`, `compress_feedForward`, `zeroBlock_initialState_certificate`.

`Extraction/Store.lean` — the digest/store half, big-endian regrouping included: `store_spec_exists`,
`store_spec_of_ok`, `stored_word_eq`, `reversed_bytes_regrouped`, `beWordStored_eq`.

`Extraction/DenotesPrep.lean` — the conditional headline handed to the regen window, rewired onto
`Icarus.Composition.AlgebraSha256dCompose`: `word_sha256d_strength_denotes` and
`word_sha256d_strength_eq_outer_compress_H0`, both carried under `WordSha256DenotesAll`, a
hypothesis that is **never asserted** until the regen window supplies `word_sha256d`.

> `Schedule`, `Compression` and `Store` import the generated `ApeironWord`; `DenotesPrep` does not.

### 6 · Keystone (1)

`Keystone/RootedSha256.lean` — AR-74, the composed round theorem: `sha256_round_rooted`
(= `RoundValue ∧ ProductPrimitives ∧ CarryPrimitives ∧ WirePrimitives ∧ ReadoutPrimitives ∧
FaceVerdict ∧ RefutationWall ∧ CostClause ∧ Witnesses`), with the clause table in
`work/proofs/AlgebraRootedSha256.md` and a fail-closed `#audit` on all nine clauses plus the theorem
itself.

**Not in scope, named exactly**: the message-schedule injection, the 64-round fold, the compression
feed-forward, the digest/store, and the executing `Cell<i64>` refinement. No security, collision,
preimage, constant-time, performance or mining claim.