# Plan — Security-algebra rebase

- **Owner:** lead (direction); Sol (grades, attribution, contract verdicts); corpus lead (transport)
- **Intent:** `docs/intent/security-algebra-rebase.md`
- **Base:** `c450c222905c050f793a23332708c230c658f82c`; dirty supplied checkout
- **Status:** captured, **not actioned**. No wave below is dispatch-ready until the storage/table epic
  closes and the lead opens it. This plan exists so the recorded results are not left behind.
- **Source corpus:** `../sanguine` (old Sanguine proofs and prose), read-only; canonical target
  `../sanguine-apeiron/proof/Apeiron` (`3d66ea3aa2fe1de5997b6b757247454a2f771358`).

## What this plan is not

It is not a new cryptography and it is not an adjudication. It transports an existing, largely
proved security module onto the standardized trichotomy, preserves each result's declared grade, and
states the corpus's minimum-encryption standard as a claim to be confirmed or refuted. It decides
nothing in advance: mainstream computational security and the corpus's positions are weighed by
evidence, and an unresolved outcome is a valid result. Every load-bearing row names what would
refute it, and a row with no possible refutation is a defect of the plan.

## End state and the three questions

The arc ends in an **end-to-end formalized implementation** per target standard, not a survey:

1. a **Rust implementation** based on that standard's canonical/trusted reference implementation,
   **formally proved** through the extraction lane (real code refines the spec; the spec's algebraic
   claims are checked);
2. the **primitives it needs extracted into Apeiron's core** as reusable, refined objects;
3. most likely a **secondary crypto repository** holding the implementations and proofs under a
   pinned correspondence with Apeiron's core.

Each standard's deliverable is answers to the **three primary questions**, each with a
confirm/refute or a documented undecidable:

1. **Stated goals, formally and empirically** — code refines spec and spec claims checked, plus the
   measured properties (standard test vectors, byte-exactness, constant-time, key-material entropy,
   refusal). Both senses required; otherwise **partial**.
2. **Structural/algebraic/mathematical weaknesses** — unintended quotients, idempotents, zero
   divisors, nilpotents, hidden homomorphisms, or a collapse where a bijection was intended; whether
   any hardness claim rests on a present structure. Unresolved → **undecided**.
3. **Implementation weaknesses from the empirical hacks** — the reference's heuristics, the
   code-to-spec hole in vertical proofs, the syscall/ABI and side-channel surface, and the
   protocol/mode conventions that have historically broken.

A "safe" verdict requires (1) in both senses and no open handle in (2) or (3); a partial or
undecided answer is a valid deliverable.

## Phase 2 — per-standard implementation (after Phase 1)

After the rebase waves below and after the growable-table epic, each target standard takes one
serial wave `I-<standard>`: (a) Rust implementation from the canonical reference, (b) extract or
bind the needed primitives into Apeiron's core, (c) the formal proof, (d) the three-question review
with empirical vectors. Gate: canonical `lake build` + `AxiomAudit` + the standard's own test
vectors. Control: a mutated round/primitive rejects a named invariant. Targets are chosen by the
lead; no target is committed here.

## Rebase map (the contract every transport row obeys)

| Old vocabulary | Standardized face | Exact canonical targets |
|---|---|---|
| Being cell `ℂ`, `i²=−1`, `/` total | elliptic, `σ < 0` | `Apeiron/Trichotomy/Basic.lean`: `normForm_eq_zero_iff_of_neg`, `eq_zero_or_eq_zero_of_mul_eq_zero_of_neg`, `noZeroDivisors_iff_disc_neg`, `isUnit_iff_normForm_ne_zero` |
| Null cell `j²=+1`, null pair, complementary channels | hyperbolic, `σ > 0` (the split) | `Apeiron/Trichotomy/Basic.lean`: `zeroDivisorPair_of_disc_one`; `Apeiron/Trichotomy/ChannelsAnyRing.lean`: `ringUpper_channel`, `ringLower_channel`; `Apeiron/Trichotomy/IdempotentsAnyRing.lean`: `ringComplementary_idem` |
| collapse, collision fiber, dropped coordinate, nilpotent | parabolic, `σ = 0` (the collapse) | `Apeiron/Trichotomy/Basic.lean`: `nilpotent_of_disc_zero`; no canonical cipher/hash theorem exists |
| `x = B/U`, `B` seed, `U` dropped window | statement-dependent; never one fused face | `Apeiron/Trichotomy/AnyRing.lean`: `RingCell`, `ringMul`, `ringNormForm`; these definitions alone establish no secrecy claim |

**Reject rule:** any row that maps the old fused “Null” to one face is invalid. A valid transport
maps split-complex null pairs, idempotents and two-channel readout to `σ > 0`, and maps nilpotent,
many-to-one or dropped-coordinate collapse to `σ = 0`; a statement containing both is split into two
obligations or marked unresolved. The rule does not infer secrecy from either face.

### Ratified theorem and claim ledger

Paths in the first column are relative to `/home/socol/Workspace/sanguine/`. A grade citation is the
source line that declares the grade, not the theorem line. `all faces` means the statement is generic
and receives no sign-specific interpretation. `none` means the pinned canonical corpus has no theorem
with the stated contract; a nearby algebraic theorem is named only in the note.

| Old file : old name | Declared grade and source | Transport face | Exact canonical target | Note |
|---|---|---|---|---|
| `proof/Substrate/Algebra/Security/SecurityCore.lean:xorStream_involutive` | proved (native), `SecurityCore.lean:16` | hyperbolic `σ > 0` split | none | Round trip only; the source excludes full-stream secrecy. |
| `proof/Substrate/Algebra/Security/SecurityCore.lean:otp_changes_the_wire` | proved (native), `SecurityCore.lean:21` | hyperbolic `σ > 0` split | none | One non-identity witness; source line 22 denies a masking theorem. |
| `proof/Substrate/Algebra/Security/SecurityCore.lean:passive-reader secrecy` | anchor, `SecurityCore.lean:24` | hyperbolic `σ > 0` split | none | No observer, distribution or advantage in this file. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:cipher_secrecy_is_only_cost` | proved (riding), `SanguineSecurity.lean:89` | elliptic `σ < 0` | none | Fiber cardinality is proved; “only cost” remains a reading at line 95. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:confidentiality_is_entropy` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Rides the old OTP statement; no canonical secrecy theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:seed_survives_the_channel` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Ratio identity, not a canonical transport. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:channel_blind_to_one_flaw` | proved (native), `SanguineSecurity.lean:85` | hyperbolic `σ > 0` split | none | Existential non-injectivity only. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:channel_recovered_with_key` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Candidate basis: `ChannelsAnyRing.ringUpper_ringOfHalves` and `ringLower_ringOfHalves`; contract differs. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:key_decodes_interior` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Candidate basis as above; no exact target. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:generator_cipher_roundtrips` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | Losslessness only; the PRG reading is retired at line 111. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:keystream_secrecy_is_the_factor` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Group equation, no canonical secrecy theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:keystream_window_is_blind` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Production scrambler remains open at lines 115-117. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:meaning_invariant_under_protocol` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | Naturality is not obfuscation; source marks that reading anchor at line 102. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:meaning_transports_across_protocols` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical protocol theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:rotation_round_trip_is_identity` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical protocol theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:integrity_decided_at_the_cell` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | Threat-model interpretation is anchor at line 106. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:dependability_any_survivor_reseeds` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical federation theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:dependability_seed_over_wire` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical wire theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:dependability_containment` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical federation theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:security_is_the_substrate` | proved (native), `SanguineSecurity.lean:96` | unresolved composite: elliptic + hyperbolic + all faces | none | Must be decomposed; a one-face transport is rejected. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:obfuscation reading` | anchor, `SanguineSecurity.lean:102` | all faces | none | No hiding theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:live SangNet wrapping` | open, `SanguineSecurity.lean:115` | hyperbolic `σ > 0` split | none | Executing correspondence absent. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:decryptRecovery` | proved (native), `EncryptionProjection.lean:69` | elliptic `σ < 0` | none | Generic injective-map construction; no trichotomy premise. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:fiberResidual_nonneg` | proved (native), `EncryptionProjection.lean:69` | all faces | none | Generic residual fact. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:decryption_is_unique_floor` | proved (riding), `EncryptionProjection.lean:73` | elliptic `σ < 0` | none | Generic injection; no canonical cipher theorem. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:injective_iff_decryptor_exists` | proved (riding), `EncryptionProjection.lean:73` | elliptic `σ < 0` | none | Mathlib re-export. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:collision_breaks_unique_floor` | proved (native), `EncryptionProjection.lean:76` | parabolic `σ = 0` collapse | none | Generic many-to-one map; `Basic.nilpotent_of_disc_zero` is only an algebraic basis. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:collision_no_decryptor` | proved (native), `EncryptionProjection.lean:76` | parabolic `σ = 0` collapse | none | No canonical hash theorem. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:cipher_fiber_is_a_singleton` | proved (native), `EncryptionProjection.lean:81` | elliptic `σ < 0` | none | Set-cardinality theorem, not entropy. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:collision_fiber_is_not_a_singleton` | proved (native), `EncryptionProjection.lean:81` | parabolic `σ = 0` collapse | none | Set theorem, not entropy. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:cipher_pins_the_plaintext` | proved (riding), `EncryptionProjection.lean:87` | elliptic `σ < 0` | none | Injectivity instantiated. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:encryption_is_projection` | proved (native), `EncryptionProjection.lean:101` | split obligation: elliptic + parabolic | none | Its Being and collapse conjuncts must not be assigned one face. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:AES/RSA placement and cost` | anchor, `EncryptionProjection.lean:94` | elliptic `σ < 0` | none | Named, not re-proven at lines 62-65. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:SHA placement` | anchor, `EncryptionProjection.lean:98` | parabolic `σ = 0` collapse | none | Empirical placement, no reduction. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:cellAdjoin_injective` | proved (native), `DescentIsTheEasyInverse.lean:35` | elliptic `σ < 0` | none | Generic injection. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:descent_strips_back` | proved (native), `DescentIsTheEasyInverse.lean:35` | elliptic `σ < 0` | none | Left inverse only. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:descent_is_being_cell_cipher` | proved (riding), `DescentIsTheEasyInverse.lean:38` | elliptic `σ < 0` | none | Rides the old unique-floor theorem. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:adjoin_decryptor_is_descent` | proved (riding/native split), `DescentIsTheEasyInverse.lean:38` | elliptic `σ < 0` | none | Source explicitly splits its two conjuncts. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:cellCollapse_off_frame_one_way` | proved (riding), `DescentIsTheEasyInverse.lean:43` | parabolic `σ = 0` collapse | none | Two residual zeros, not no-decryptor. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:autotropism and no-search readings` | open, `DescentIsTheEasyInverse.lean:47` | unresolved | none | Statements mention neither frame/sign nor cost. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:masked_key_lt_size` | proved (native), `MandalaMemory.lean:33` | parabolic `σ = 0` collapse | none | Arithmetic masking; no sigma occurs. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:access_in_region` | proved (native), `MandalaMemory.lean:33` | parabolic `σ = 0` collapse | none | Closest canonical result is `Apeiron/Capability/Naming.lean:nameMap_lt`; statements differ. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:no_key_escapes` | proved (native), `MandalaMemory.lean:33` | parabolic `σ = 0` collapse | none | Canonical `Capability.MaskedTable.guardedRead_ungranted` has a different contract. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:nested_access_in_parent` | proved (native), `MandalaMemory.lean:33` | parabolic `σ = 0` collapse | none | Closest canonical result is `Apeiron/Region/Graded.lean:graded_access_in_region`; statements differ. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:no_overflow_into_neighbor` | proved (native), `MandalaMemory.lean:33` | parabolic `σ = 0` collapse | none | Closest canonical result is `Apeiron/Region/Graded.lean:graded_no_overflow_into_neighbor`; statements differ. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:load_address_bounded` | proved (native), `MandalaMemory.lean:38` | parabolic `σ = 0` collapse | none | Restates the masked access theorem. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:masked_load_never_oob` | anchor, `MandalaMemory.lean:42` | parabolic `σ = 0` collapse | none | Emitter coverage is not in the statement. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:mandala_is_closed` | proved (native), `MandalaMemory.lean:56` | parabolic `σ = 0` collapse | none | Region layout remains an implementation bridge. |
| `proof/Substrate/Algebra/Security/RaceFreedom.lean:write_comm` | proved (Mathlib), `RaceFreedom.lean:31` | all faces | none | Functional updates only. |
| `proof/Substrate/Algebra/Security/RaceFreedom.lean:disjoint_addr` | proved (riding), `RaceFreedom.lean:33` | all faces | none | Rides old region arithmetic. |
| `proof/Substrate/Algebra/Security/RaceFreedom.lean:write_pushes_through` | proved (native), `RaceFreedom.lean:36` | all faces | none | Sequential functional-store theorem. |
| `proof/Substrate/Algebra/Security/RaceFreedom.lean:no_race_on_disjoint_regions` | proved (native), `RaceFreedom.lean:36` | all faces | none | One-write commutation, not a memory model. |
| `proof/Substrate/Algebra/Security/RaceFreedom.lean:writes_comm` | proved (native), `RaceFreedom.lean:36` | all faces | none | Batch-order swap, not arbitrary interleavings. |
| `proof/Substrate/Algebra/Security/RaceFreedom.lean:data-race freedom reading` | next, `RaceFreedom.lean:40` | all faces | none | No concurrent semantics. |
| `proof/Substrate/Algebra/Security/CapProvenance.lean:eval_in_scope` | proved (native), `CapProvenance.lean:29` | all faces | `Apeiron/Capability/Basic.lean:eval_in_scope` | Exact port. |
| `proof/Substrate/Algebra/Security/CapProvenance.lean:forge_reaches_only_scope_discharged` | proved (native), `CapProvenance.lean:35` | all faces | `Apeiron/Capability/Basic.lean:forge_reaches_only_scope` | Canonical theorem strengthens naming and keeps the same contract. |
| `proof/Substrate/Algebra/Security/CapProvenance.lean:cexpr_exhaustive` | proved (native), `CapProvenance.lean:29` | all faces | `Apeiron/Capability/Basic.lean:cexpr_exhaustive` | Exact port; source grade group covers the grammar result. |
| `proof/Substrate/Algebra/Security/CapProvenance.lean:Slang grammar correspondence` | anchor, `CapProvenance.lean:41` | all faces | none | Compiler parser/type-rule binding absent. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:cone_hash_stores_no_key` | proved (native), `GoldenStorage.lean:34` | all faces | none | Assumes injective placement. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:cone_hash_lookup_exact` | proved (native), `GoldenStorage.lean:34` | all faces | none | No canonical perfect-hash theorem. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:trit_hash_stores_no_key` | proved (native), `GoldenStorage.lean:34` | all faces | none | No `ZMod 3`-specific content. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:injective_placement_recovers` | proved (native), `GoldenStorage.lean:34` | elliptic `σ < 0` | none | Generic injective-map left inverse. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:peeling_exists_iff_hall` | proved (Mathlib), `GoldenStorage.lean:41` | all faces | none | Re-export of Hall. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:hall_gives_perfect_hash` | proved (Mathlib/native composition), `GoldenStorage.lean:41` | all faces | none | Golden-cone expansion remains open. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:genEncode_append` | proved (native), `GoldenStorage.lean:44` | all faces | none | Stream append law. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:dynamic_seed_incremental_and_lossless` | proved (native/riding split), `GoldenStorage.lean:44` | all faces | none | Cost and trit-storage readings are anchors at lines 49-55. |
| `proof/Substrate/Algebra/Security/GoldenStorage.lean:golden peeling threshold` | open, `GoldenStorage.lean:56` | all faces | none | Injectivity and `1.23·n` threshold unproved. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:get_insertWith_self` | grade unconfirmed; open, declaration at `HashTrinityCore.lean:88` | all faces | `Apeiron/Table/Basic.lean:get_insertWith_self` | Exact canonical port exists; no source grade line names the old declaration. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:get_insertWith_ne` | grade unconfirmed; open, declaration at `HashTrinityCore.lean:91` | all faces | `Apeiron/Table/Basic.lean:get_insertWith_ne` | Exact canonical port exists; no source grade line names the old declaration. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:merge_is_the_only_freedom` | proved (native), `HashTrinityCore.lean:29` | all faces | `Apeiron/Table/Basic.lean:merge_is_the_only_freedom` | Exact canonical port. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:set_member` | proved (native), `HashTrinityCore.lean:33` | all faces | none | Worked instance intentionally not ported. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:map_lookup` | proved (native), `HashTrinityCore.lean:33` | all faces | none | Worked instance intentionally not ported. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:count_two` | proved (native), `HashTrinityCore.lean:33` | all faces | none | Worked instance intentionally not ported. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:graph_edge` | proved (native), `HashTrinityCore.lean:33` | all faces | none | Worked instance intentionally not ported. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:hashmap_trinity` | proved (native), `HashTrinityCore.lean:33` | all faces | none | Conjunction intentionally not ported. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:get_build_acc` | proved (native), `HashTrinityCore.lean:38` | all faces | `Apeiron/Table/Basic.lean:get_build_acc` | Exact canonical port. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:build_gauge_invariance` | proved (native), `HashTrinityCore.lean:38` | all faces | `Apeiron/Table/Basic.lean:build_gauge_invariance` | Exact canonical port. |
| `proof/Substrate/Algebra/Security/HashTrinityCore.lean:estimation and BLUP theorem set` | proved (Mathlib), `HashTrinityCore.lean:42` and `:46` | all faces | none | Statistical interpretation remains anchor at lines 53-61. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:schedule_free_is_anagram_blind` | proved (native), `FoldDepth.lean:77` | all faces | none | Sequence theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:key_fold_cannot_be_schedule_free` | proved (native), `FoldDepth.lean:77` | all faces | none | Sequence theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:interleave_evens_odds` | proved (native), `FoldDepth.lean:81` | all faces | none | No canonical fold target. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:deinterleave_lossless` | proved (native), `FoldDepth.lean:81` | all faces | none | No canonical fold target. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:sheetSplit_lossless` | proved (native), `FoldDepth.lean:81` | all faces | none | No canonical fold target. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:fold_is_lossy` | proved (native), `FoldDepth.lean:85` | parabolic `σ = 0` collapse | none | Pigeonhole only. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:mulUnit_bijective` | proved (riding), `FoldDepth.lean:85` | elliptic `σ < 0` | none | Canonical `Table.WordIdentity.odd_mul_zmod_injective` is related but not the same theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:windows_from_one_product` | proved (riding), `FoldDepth.lean:85` | all faces | none | Old RadixDescent result. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:snd_balanced` | proved (native), `FoldDepth.lean:89` | parabolic `σ = 0` collapse | none | Counted-fiber theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:exactlyToOne_comp_equiv` | proved (native), `FoldDepth.lean:89` | all faces | none | Generic transport through a bijection. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:xorHalves_balanced` | proved (native), `FoldDepth.lean:89` | parabolic `σ = 0` collapse | none | Counted-fiber theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:fold128_balanced` | next, `FoldDepth.lean:113` | parabolic `σ = 0` collapse | none | Not connected to the golden multiply. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:pack_injective` | next, `FoldDepth.lean:120` | elliptic `σ < 0` | none | Not composed with multiply/coordinate. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:depth theorem set` | anchor, `FoldDepth.lean:93` | all faces | none | Arithmetic over named naturals, not a fold cost theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:closed-form-collision-absence claims` | next, `FoldDepth.lean:106` | all faces | none | Concrete witnesses do not establish a universal absence. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:accum_append` | proved (native), `KeyResolution.lean:43` | all faces | none | No exact canonical theorem. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:resolution_breaks_key_independence` | proved (native), `KeyResolution.lean:46` | parabolic `σ = 0` collapse | none | Concrete counterexample. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:resolution_preserves_gauge` | proved (riding), `KeyResolution.lean:49` | all faces | `Apeiron/Table/Basic.lean:build_gauge_invariance` | Target is the premise it instantiates, not the relabel theorem itself. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:resolution_max_converges` | proved (riding), `KeyResolution.lean:49` | all faces | none | No exact target. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:the_resolution_law` | grade unconfirmed; open, declaration at `KeyResolution.lean:161` | split obligation: all faces + parabolic collapse | none | Headline is not listed in its warrant bullets. |
| `proof/Substrate/Algebra/Security/IncrementalResolution.lean:incremental_resolution_breaks_the_split` | proved (native), `IncrementalResolution.lean:43` | parabolic `σ = 0` collapse | none | Concrete counterexample. |
| `proof/Substrate/Algebra/Security/IncrementalResolution.lean:relabelFrom_const` | proved (native), `IncrementalResolution.lean:48` | all faces | none | Fixed-crosswalk bridge. |
| `proof/Substrate/Algebra/Security/IncrementalResolution.lean:relabelInc_const` | proved (native), `IncrementalResolution.lean:48` | all faces | none | Fixed-crosswalk bridge. |
| `proof/Substrate/Algebra/Security/IncrementalResolution.lean:merge_BU_is_commutative_monoid` | proved (riding), `IncrementalResolution.lean:50` | all faces | none | Nat laws restated. |
| `proof/Substrate/Algebra/Security/IncrementalResolution.lean:the_incremental_resolution_law` | grade unconfirmed; open, declaration at `IncrementalResolution.lean:143` | split obligation: all faces + parabolic collapse | none | Headline is not assigned a grade in the warrant block. |
| `proof/Substrate/Algebra/Security/ResumableProbe.lean:probe_split` | proved (native), `ResumableProbe.lean:23` | all faces | none | List split; golden grounding is open at line 31. |
| `proof/Substrate/Algebra/Security/ResumableProbe.lean:probe_resumes` | proved (native), `ResumableProbe.lean:23` | all faces | none | List split specialization. |
| `proof/Substrate/Algebra/Security/ResumableProbe.lean:probe_resumes_twice` | proved (native), `ResumableProbe.lean:23` | all faces | none | List split composition. |
| `proof/Substrate/Algebra/Security/ResumableProbe.lean:probeFrom_length` | proved (native), `ResumableProbe.lean:23` | all faces | none | Range length. |
| `proof/Substrate/Algebra/Security/ResumableProbe.lean:batch_splits` | proved (native), `ResumableProbe.lean:28` | all faces | none | Map fact. |
| `proof/Substrate/Algebra/Security/ResumableProbe.lean:batch_reorder` | proved (native), `ResumableProbe.lean:28` | all faces | none | Map permutation fact. |
| `proof/Substrate/Algebra/Security/ResumableProbe.lean:batch_independent` | proved (native), `ResumableProbe.lean:28` | all faces | none | Map indexing fact. |
| `proof/Substrate/Algebra/Security/ResumableProbe.lean:shared mutable interleaving` | open, `ResumableProbe.lean:45` | all faces | none | Bisimulation absent. |
| `proof/Substrate/Algebra/Security/TimerWheel.lean:slot_lt` | proved (riding), `TimerWheel.lean:39` | all faces | none | Radix digit bound. |
| `proof/Substrate/Algebra/Security/TimerWheel.lean:level_lt_depth` | proved (native), `TimerWheel.lean:34` | all faces | none | Relational Nat band theorem. |
| `proof/Substrate/Algebra/Security/TimerWheel.lean:level_unique` | proved (native), `TimerWheel.lean:34` | all faces | none | Relational Nat band theorem. |
| `proof/Substrate/Algebra/Security/TimerWheel.lean:cascade_antitone` | proved (native), `TimerWheel.lean:34` | all faces | none | No runtime migration theorem. |
| `proof/Substrate/Algebra/Security/TimerWheel.lean:level_exists` | proved (native), `TimerWheel.lean:34` | all faces | none | Relational Nat band theorem. |
| `proof/Substrate/Algebra/Security/TimerWheel.lean:level_zero_iff` | proved (native), `TimerWheel.lean:34` | all faces | none | Scan reading remains anchor at line 47. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:perm_map_sum` | proved (Mathlib), `WorkStealing.lean:66` | all faces | none | Permutation-invariant monoid sum. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:parallel_reduce_correct` | proved (Mathlib), `WorkStealing.lean:66` | all faces | none | Bucket permutation is a hypothesis. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:counts_partition` | proved (native), `WorkStealing.lean:69` | all faces | none | Nat interval model. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:each_index_once` | proved (native), `WorkStealing.lean:69` | all faces | none | Nat interval model. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:steal_moves_one` | proved (native), `WorkStealing.lean:69` | all faces | none | Sequential model step. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:pop_moves_one` | proved (native), `WorkStealing.lean:69` | all faces | none | Sequential model step. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:last_element_race exclusivity` | next, `WorkStealing.lean:73` | all faces | none | Existing theorem does not state exclusivity. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:sum/list/load/makespan theorem set` | proved (native), `WorkStealing.lean:62` | all faces | none | Divided makespan form remains open at line 93. |
| `proof/Substrate/Algebra/Security/WorkStealing.lean:greedy busy invariant` | anchor, `WorkStealing.lean:98` | all faces | none | Hypothesis, not scheduler theorem. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:sum_le_length_mul` | proved (native), `AggregationBound.lean:74` | all faces | none | List arithmetic. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:build_linear_of_bounded_probes` | proved (native), `AggregationBound.lean:74` | all faces | none | Probe bound remains a hypothesis. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:flood_is_quadratic` | proved (Mathlib), `AggregationBound.lean:77` | parabolic `σ = 0` collapse | none | Cost function is defined, not linked to a table. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:flood_dominates_bounded` | proved (Mathlib), `AggregationBound.lean:77` | parabolic `σ = 0` collapse | none | Arithmetic comparison. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:exists_collision_at_step` | proved (native), `AggregationBound.lean:81` | parabolic `σ = 0` collapse | none | Refutes the old separation inference. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:separation_does_not_bound_probe` | proved (native), `AggregationBound.lean:81` | parabolic `σ = 0` collapse | none | Refutation result. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:descent_can_be_wholly_blocked` | proved (native), `AggregationBound.lean:91` | parabolic `σ = 0` collapse | none | Worst-case blocking witness. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:shipped_probe_can_be_wholly_blocked` | proved (native), `AggregationBound.lean:91` | parabolic `σ = 0` collapse | none | Model result, not current Apeiron source. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:probe_prefix_forces_distinct_occupied_slots` | proved (native), `AggregationBound.lean:100` | all faces | none | Purchase bound. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:probe_length_forces_occupied_slots` | proved (native), `AggregationBound.lean:100` | all faces | none | Conditional on no revisit. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:build_bounded_by_n_squared` | proved (native), `AggregationBound.lean:107` | all faces | none | Quadratic ceiling. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:build_linear_of_terminating_descents` | proved (native), `AggregationBound.lean:119` | all faces | none | Termination is a hypothesis. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:build_bounded_by_sweep` | proved (riding), `AggregationBound.lean:129` | all faces | none | Old GroupProbe theorem; no current source correspondence. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:better-than-quadratic adversarial bound` | open, `AggregationBound.lean:139` | all faces | none | No such bound. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:expansion theorem set` | proved (native/Mathlib/riding), `BeliefContraction.lean:63`, `:75`, `:77` | all faces | `Apeiron/Table/Basic.lean:get_insertWith_self` | Only `expansion_is_one_insert` roots in canonical `insertWith`; remaining contracts have no exact target. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:no-inverse/no-negation theorem set` | proved (native), `BeliefContraction.lean:64` | hyperbolic `σ > 0` split | none | Lattice facts; no canonical contraction theory. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:kernel/incision theorem set` | proved (native), `BeliefContraction.lean:66` | all faces | none | No canonical AGM module. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:recovery_holds_in_the_group` | proved (native/Mathlib), `BeliefContraction.lean:72`, `:76` | elliptic `σ < 0` | none | Group identity only. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:recovery_fails` | proved (native), `BeliefContraction.lean:72` | hyperbolic `σ > 0` split | none | Idempotent-fold witness. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:maximal_removal_overshoots` | proved (native), `BeliefContraction.lean:73` | all faces | none | No canonical target. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:levi_collapses_to_expansion` | proved (native), `BeliefContraction.lean:73` | all faces | none | No canonical target. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:harper_is_blocked` | proved (native), `BeliefContraction.lean:73` | all faces | none | No canonical target. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:the_contraction_trinity` | proved (native), `BeliefContraction.lean:74` | split obligation: elliptic + hyperbolic + all faces | none | Composite must remain decomposed. |
| `proof/Substrate/Algebra/Security/BeliefContraction.lean:AGM correspondence` | anchor, `BeliefContraction.lean:80` | all faces | none | Reading onto AGM, not a theorem. |

Supporting source rows required by the transport waves:

| Old file : old name | Declared grade and source | Transport face | Exact canonical target | Note |
|---|---|---|---|---|
| `proof/Substrate/Algebra/NullCell/NullCell.lean:nj_sq` | proved (native), `NullCell.lean:71` | hyperbolic `σ > 0` split | none | Follows from canonical `ringMul`; no same-name theorem. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:nNorm_mul` | proved (native), `NullCell.lean:71` | hyperbolic `σ > 0` split | `Apeiron/Trichotomy/Basic.lean:normForm_mul` | Exact specialization at `σ = 1`. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:ncell_indefinite` | proved (native), `NullCell.lean:71` | hyperbolic `σ > 0` split | none | No exact canonical target. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:nNull_norm` | proved (native), `NullCell.lean:71` | hyperbolic `σ > 0` split | none | Covered by `Basic.zeroDivisorPair_of_disc_one`, not the same statement. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:null_times_mirror` | proved (native), `NullCell.lean:71` | hyperbolic `σ > 0` split | `Apeiron/Trichotomy/Basic.lean:zeroDivisorPair_of_disc_one` | Canonical bundle contains the same product and nonzero witnesses. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:null_not_invertible` | proved (native), `NullCell.lean:71` | hyperbolic `σ > 0` split | `Apeiron/Trichotomy/Basic.lean:normForm_zero_not_invertible` | General canonical theorem instantiated at `σ = 1`. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:phase_no_zero_divisors` | proved (native), `NullCell.lean:71` | elliptic `σ < 0` | `Apeiron/Trichotomy/Basic.lean:eq_zero_or_eq_zero_of_mul_eq_zero_of_neg` | Canonical generalization. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:phase_invertible` | proved (native), `NullCell.lean:71` | elliptic `σ < 0` | `Apeiron/Trichotomy/Basic.lean:isUnit_iff_normForm_ne_zero` | Canonical criterion; representation transport still required. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:e_orthogonal` | grade unconfirmed; open, declaration at `NullCell.lean:302` | hyperbolic `σ > 0` split | `Apeiron/Trichotomy/IdempotentsAnyRing.lean:ringComplementary_idem` | No source grade line names it; exact generalized target after parameter witnesses. |
| `proof/Substrate/Algebra/NullCell/NullCell.lean:physical-time identification` | anchor, `NullCell.lean:78` | hyperbolic `σ > 0` split | none | Physics mapping only. |
| `proof/Substrate/Algebra/Navigation/Federation.lean:indexed algebra theorem set` | proved (native), `Federation.lean:82` | all faces | none | No canonical federation module. |
| `proof/Substrate/Algebra/Navigation/Federation.lean:adversarial theorem set` | proved (native), `Federation.lean:87` | all faces | none | No canonical federation module. |
| `proof/Substrate/Algebra/Navigation/Federation.lean:excision_exact` | proved (riding), `Federation.lean:91` | all faces | none | Old Collapse theorem. |
| `proof/Substrate/Algebra/Navigation/Federation.lean:federation_reseeds_the_genome` | proved (riding), `Federation.lean:91` | all faces | none | No canonical reseed theorem. |
| `proof/Substrate/Algebra/Navigation/Federation.lean:conflict_floor theorem set` | proved (Mathlib), `Federation.lean:97` | all faces | none | Finset minimum; cost claims are anchor at line 101. |
| `proof/Substrate/Algebra/Navigation/Federation.lean:scheduler_floor theorem set` | anchor, `Federation.lean:105` | all faces | none | Scheduler/energy interpretation absent. |
| `proof/Substrate/Algebra/Navigation/Federation.lean:operational wire and SHA seams` | open, `Federation.lean:116` | unresolved | none | Protocol, observation and key-ungrindability remain open. |
| `proof/Closures/Applied/WebCipher.lean:otp_covers_fully` | proved (native), `WebCipher.lean:75` | hyperbolic `σ > 0` split | none | Full-pad coverage, not secrecy by itself. |
| `proof/Closures/Applied/WebCipher.lean:bulk_collapses_np` | proved (riding), `WebCipher.lean:79` | all faces | none | Source lines 89-95 deny `P = NP`. |
| `proof/Closures/Applied/WebCipher.lean:computational_cipher_pins_the_plaintext` | proved (riding), `WebCipher.lean:79` | elliptic `σ < 0` | none | Carries only injectivity equation. |
| `proof/Closures/Applied/WebCipher.lean:web_boundary_is_a_blur` | proved (riding), `WebCipher.lean:79` | hyperbolic `σ > 0` split | none | Physical-pad interpretation is anchor at line 96. |
| `proof/Closures/Applied/WebCipher.lean:web_edge_is_the_pad` | proved (riding), `WebCipher.lean:79` | hyperbolic `σ > 0` split | none | Shared-random-bit interpretation is anchor at line 96. |
| `proof/Closures/Applied/WebCipher.lean:web_edges_are_channels` | proved (riding), `WebCipher.lean:79` | hyperbolic `σ > 0` split | none | Dimension count only. |
| `proof/Closures/Applied/WebCipher.lean:web_otp_perfect_secrecy` | proved (riding), `WebCipher.lean:79` | hyperbolic `σ > 0` split | none | No canonical secrecy theorem. |
| `proof/Closures/Applied/WebCipher.lean:web_cipher` | proved (native composition), `WebCipher.lean:75` | hyperbolic `σ > 0` split | none | Bundle adds no new content. |
| `proof/Closures/Applied/WebCipher.lean:cover_is_indistinguishable` | proved (native composition), `WebCipher.lean:75` | hyperbolic `σ > 0` split | none | Cover-traffic reading is anchor at line 101. |
| `proof/Closures/Applied/WebCipher.lean:bulk-capable adversary and classical bridge` | open, `WebCipher.lean:89` and `:104` | unresolved | none | Confirm or refute; do not presuppose either direction. |
| `proof/ThreatModel.lean:ThreatModel theorem set` | proved (native/riding), `ThreatModel.lean:80` and `:86` | split obligation: elliptic + hyperbolic | none | The old threat headline fuses reversible and split-cone claims. |
| `proof/ThreatModel.lean:O(1) admission` | anchor, `ThreatModel.lean:98` | all faces | none | Universal quantification is not a cost bound. |
| `proof/ThreatModel.lean:threat-is-friction identification` | anchor, `ThreatModel.lean:123` | all faces | none | Security interpretation, not algebra. |
| `proof/ThreatModel.lean:threat completeness` | open, `ThreatModel.lean:114` | unresolved | none | No theorem quantifies over every threat. |
| `proof/Runtime/SangNet.lean:executable wire` | open, `SangNet.lean:47` | all faces | none | File states no theorem. |
| `proof/Runtime/SangNet.lean:no-new-primitive` | proved (riding), `SangNet.lean:50` | all faces | none | Upstream evaluator theorem; no canonical runtime target. |
| `proof/Runtime/SangNet.lean:ABI and socket behavior` | anchor, `SangNet.lean:54` | all faces | none | Runtime validation, not Lean-certified. |
| `proof/Runtime/SangNet.lean:frictionFree direction` | open, `SangNet.lean:59` | all faces | none | Upstream theorem proves the opposite encode/decode direction. |
| `proof/Runtime/SangNet.lean:pad entropy and threat model` | open, `SangNet.lean:72` | hyperbolic `σ > 0` split | none | File input is not proved fresh/uniform. |
| `theory/MessageScheduleTopology.lean:rot_zero` | proved (native), `MessageScheduleTopology.lean:46` | elliptic `σ < 0` | none | Finite-word rotation theorem. |
| `theory/MessageScheduleTopology.lean:rot_add` | proved (native), `MessageScheduleTopology.lean:46` | elliptic `σ < 0` | none | Finite-word rotation theorem. |
| `theory/MessageScheduleTopology.lean:rot_injective` | proved (native), `MessageScheduleTopology.lean:46` | elliptic `σ < 0` | none | Finite-word injection. |
| `theory/MessageScheduleTopology.lean:rot_linear` | proved (native), `MessageScheduleTopology.lean:46` | elliptic `σ < 0` | none | GF(2) linearity. |
| `theory/MessageScheduleTopology.lean:shr_linear` | proved (native), `MessageScheduleTopology.lean:46` | parabolic `σ = 0` collapse | none | Linearity does not prove loss. |
| `theory/MessageScheduleTopology.lean:sigma0_linear` | proved (native), `MessageScheduleTopology.lean:46` | elliptic `σ < 0` | none | SHA placement remains unproved. |
| `theory/MessageScheduleTopology.lean:sigma1_linear` | proved (native), `MessageScheduleTopology.lean:46` | elliptic `σ < 0` | none | SHA placement remains unproved. |
| `theory/MessageScheduleTopology.lean:schedule_is_rotor_string_over_null_thread` | grade unconfirmed; open, declaration at `MessageScheduleTopology.lean:161` | split obligation: elliptic + parabolic | none | No source grade line names the headline; it omits the carry theorem its prose claims. |
| `theory/MessageScheduleTopology.lean:carry-thread/hardness localization` | anchor, `MessageScheduleTopology.lean:49` | parabolic `σ = 0` collapse | none | Not re-proved in the file. |

Prose claims retain their own labels and have no canonical theorem target:

| Old file : old name | Declared grade and source | Transport face | Exact canonical target | Note |
|---|---|---|---|---|
| `theory/SECURITY_TOOLS.md:per-atom detector` | proved (riding), `SECURITY_TOOLS.md:302` | all faces | none | Upstream old-corpus theorems only. |
| `theory/SECURITY_TOOLS.md:per-cell sequence detector` | proved (riding), `SECURITY_TOOLS.md:308` | all faces | none | Upstream old-corpus theorems only. |
| `theory/SECURITY_TOOLS.md:zoning` | proved (riding), `SECURITY_TOOLS.md:313` | all faces | none | Runtime correspondence is not canonical. |
| `theory/SECURITY_TOOLS.md:operational and benchmark tables` | anchor, `SECURITY_TOOLS.md:319` and `:323` | all faces | none | Historical measurements, not rerun. |
| `theory/SECURITY_TOOLS.md:novel-encoding bypass resistance` | open, `SECURITY_TOOLS.md:331` | all faces | none | Lifter/decode coverage absent. |
| `theory/SECURITY_TOOLS.md:secure-wire adversary and entropy` | open, `SECURITY_TOOLS.md:340` | hyperbolic `σ > 0` split | none | Assumed adversary and deployment entropy. |
| `theory/THE_HARDENING.md:stationary-phase invariance` | proved (riding), `THE_HARDENING.md:241` | all faces | none | Conditional upstream theorem. |
| `theory/THE_HARDENING.md:structural universality` | proved (riding), `THE_HARDENING.md:246` | all faces | none | Pointwise applicative agreement only. |
| `theory/THE_HARDENING.md:eight-row attack exhaustion` | open, `THE_HARDENING.md:255` | unresolved | none | Proposed map, not theorem. |
| `theory/THE_HARDENING.md:status columns` | anchor, `THE_HARDENING.md:262` | all faces | none | Dated ledger. |
| `theory/THE_HARDENING.md:K-elevation` | open, `THE_HARDENING.md:272` | unresolved | none | Runtime machinery absent. |
| `theory/THE_THREAT_MODEL.md:STRIDE theorem holders` | proved (riding), `THE_THREAT_MODEL.md:447` | all faces | none | Algebraic statements only. |
| `theory/THE_THREAT_MODEL.md:friction primitive and instances` | proved (riding), `THE_THREAT_MODEL.md:460` | all faces | none | No canonical security target. |
| `theory/THE_THREAT_MODEL.md:operational findings and DREAD` | anchor, `THE_THREAT_MODEL.md:473` and `:478` | all faces | none | Runs and engineering judgement. |
| `theory/THE_THREAT_MODEL.md:threat completeness` | open, `THE_THREAT_MODEL.md:487` | unresolved | none | No universal closure theorem. |
| `theory/THE_THREAT_MODEL.md:aggregation coverage` | open, `THE_THREAT_MODEL.md:490` | all faces | none | Old refutation is retained. |
| `theory/THE_FLAW_TRINITY.md:abstract trichotomy spine` | proved (riding), `THE_FLAW_TRINITY.md:224` | split obligation: elliptic + parabolic + hyperbolic | `Apeiron/Trichotomy/Basic.lean:normForm_eq_zero_iff_of_neg`, `nilpotent_of_disc_zero`, `zeroDivisorPair_of_disc_one` | Canonical targets cover the three algebraic faces, not old theorem equivalence. |
| `theory/THE_FLAW_TRINITY.md:physics register` | anchor, `THE_FLAW_TRINITY.md:238` | split obligation: all three | none | L3 identification. |
| `theory/THE_FLAW_TRINITY.md:biology register` | anchor, `THE_FLAW_TRINITY.md:244` | split obligation: all three | none | No measured sigma. |
| `theory/THE_FLAW_TRINITY.md:wick_swaps_breaks` | next, `THE_FLAW_TRINITY.md:253` | split obligation: elliptic + parabolic + hyperbolic | none | Named declaration does not exist. |
| `theory/P2P_SURVIVAL_LAYER.md:survival claim` | open, `P2P_SURVIVAL_LAYER.md:74` | all faces | none | Failure-model theorem absent. |
| `theory/P2P_SURVIVAL_LAYER.md:deleted lift inventory` | retired, `P2P_SURVIVAL_LAYER.md:80` | all faces | none | Historical only. |
| `theory/P2P_SURVIVAL_LAYER.md:step-8 form licences` | proved (riding), `P2P_SURVIVAL_LAYER.md:86` | all faces | none | Says nothing about deleted lifts. |
| `theory/P2P_SURVIVAL_LAYER.md:appear-as-Tor capability` | anchor, `P2P_SURVIVAL_LAYER.md:92` | all faces | none | No fixture or gate. |

The ledger freezes source grades. Transport can confirm or refute claims, but this planning pass does
not promote a theorem because it compiles, demote a theorem because its interpretation is narrow, or
presuppose that either mainstream cryptography or the old corpus is correct.

## Waves

Each row owns at most three files, has one gate, and one rejecting control. Transport rows
preserve statements; a row that must change a statement, a grade or an attribution is a **Sol**
row (`Sol` trigger: contract/theorem/attribution/grade). Prose-only rows carry a documentary gate.

| Wave / task / owner | Owned files and completion | One gate | One rejecting control |
|---|---|---|---|
| 0 / S0 / Sol | intent + plan only: ratify the map, the per-file grade inventory and the statement-transport obligation; name the exact canonical targets per source theorem. | documentary gate (whitespace/complete-file) | A verdict fixture that maps the fused "Null" onto a single face must be rejected. |
| 1 / R1 / Sol | transport `NullCell/NullCell.lean` + `Security/EncryptionProjection.lean` into the canonical corpus at unchanged statements; split Being/Null across elliptic/hyperbolic/parabolic with the σ=0 collapse separated. | `lake build` (canonical) + `AxiomAudit` closed cone | Swap the OTP channel to the σ=0 face; the idempotent/orthogonality check rejects. |
| 2 / R2 / Sol | transport `Security/SanguineSecurity.lean` + `Security/SecurityCore.lean` + `WebCipher.otp_perfect_secrecy`: the minimum-encryption standard (cost rejected; split-channel and collapse ends admitted; key re-attaches; seed survives). | `lake build` + `AxiomAudit` | Admit a bijective/cost-based cipher as information-theoretically secure; `cipher_secrecy_is_only_cost` rejects. |
| 3 / R3 / Sol | transport `EncryptionProjection`'s cipher/hash fork + `DescentIsTheEasyInverse` (unique floor, collision wall, descent as decryptor). | `lake build` + `AxiomAudit` | Make the cipher map non-injective but keep `decryption_is_unique_floor`; the singleton-fiber obligation rejects. |
| 4 / R4 / Sol | transport `Navigation/Federation.lean` + `RecoveryTrinity`/`WebCipher` wire results: composition/decomposition, reseed, adversarial cell verdict, `conflict_floor`/`scheduler_floor`. | `lake build` + `AxiomAudit` | Break the survivor reseed (drop the diagonal genome); `federation_reseeds_the_genome` rejects. |
| 5 / R5 / Sol | transport `MandalaMemory.lean` + `RaceFreedom.lean` + `CapProvenance.lean` (spatial safety, race freedom, capability provenance). | `lake build` + `AxiomAudit` | Add an unmasked access term; `no_key_escapes`/`masked_load_never_oob` rejects. |
| 6 / D1 / Flash | prose rebase: `theory/{THE_THREAT_MODEL,THE_HARDENING,SECURITY_TOOLS,P2P_SURVIVAL_LAYER,THE_FLAW_TRINITY}.md` and `experiments/{otp_battery.py,waf_rules,vuln_form}` onto the faces and the standard. | documentary gate | A passage that presents cost-based crypto as the floor is rejected in review. |
| 7 / P1 / Sol | the minimum-standard headline in the canonical corpus: a named theorem bundle relating the split-channel and collapse ends to the primitive, with the cost-rejection as a corollary. | `lake build` + `AxiomAudit` | Weaken the bundle to a single face; the two-ended witness rejects. |

Conditional rows: the canonical target corpus and the storage epic's completion are prerequisites;
if a source theorem's statement cannot be preserved, stop and return a Sol attribution task rather
than weakening it.

## Deeper review — the strong claims (spikes, each ends in evidence or a blocker)

| Spike | Question | Source |
|---|---|---|
| `SPIKE-sha256d` | Is "SHA-256d is technically EASIER to break than SHA-256" proved, anchor or conjecture, and does the three-depth reading survive rebasing? | `theory/THE_PASS_THE_PARCEL_CHAIN.md`; `NullCell/TensionPinch.lean` (labels it conjecture) |
| `SPIKE-message-schedule` | What exactly is proved about localising SHA-256 preimage resistance to the message schedule, and what is its falsifier? | `theory/MessageScheduleTopology.lean`; `THE_HASHMAP_PATH.md` (45→52 steps frontier) |
| `SPIKE-aes-rsa-bridge` | The exact statement of the AES/RSA bulk-collapse bridge, to be proved or explicitly left open. | `Security/EncryptionProjection.lean:62` ("named, not re-proven") |
| `SPIKE-flaw-trinity` | Does the attack-as-unintended-quotient reading (linear≈split, differential≈nilpotent, Bellcore≈zero-divisor) survive scrutiny as a method, and what does it actually locate? | `THE_FLAW_TRINITY.md`; attack table in `docs/intent/security-algebra-rebase.md` |
| `SPIKE-federation-security` | Rebase and grade the distributed-security core: `honesty_decided_at_the_cell`, the leaderless floors, partition/duplicate semantics. | `Navigation/Federation.lean`; `docs/direction.md` "Federation / distributed" |

Each spike is time-boxed, ends in evidence (a named theorem, a grade, or a blocker), and does not by
itself close the epic or ship a security claim.

## Gates

Canonical proof build (cwd `proof/corpus`, lane-locked, bounded):

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  env LEAN_NUM_THREADS=1 timeout 900s lake build
```

Closed-cone audit: every registered headline in `proof/corpus/AxiomAudit.lean` stays inside
`{propext, Classical.choice, Quot.sound}`; the Slang empty cone is untouched. Documentary rows use
the complete-file whitespace gate.

## Ownership, order and stop conditions

- Waves are serial barriers; each row's gate and control pass before the next opens.
- `R1` before `R2`/`R3` (the face split is the prerequisite); `R4` and `R5` are independent of
  `R2`/`R3` but share the canonical corpus lane and serialize on it.
- `P1` follows `R2`/`R3`; `D1` follows the transport rows so the prose cites settled names.
- Stop on any statement that cannot be preserved, on an unsupported canonical target, or on a
  hardness claim with no named theorem — return the exact obligation, do not weaken the result.

## Governing-document edits (lead-applied, not applied here)

| File + section | Insertion |
|---|---|
| `docs/direction.md`, Programme outcomes | New row "Security / cryptography rebase" pointing at this intent/plan; extend "Federation / distributed" to cite `Navigation/Federation.lean` and `WebCipher`. |
| `docs/kernel-boundary.md`, trust boundary | Note that cryptographic hardness and the AES/RSA bridge stay outside the proved claim until a named reduction exists; the security layer is read off the primitive. |
| `docs/plan/programme-roadmap.md` | Section "Security-algebra rebase (staged)" summarizing the waves and spikes. |

## Acceptance

This plan is accepted when S0 ratifies the map and the deeper-review spikes return evidence. No
security claim, hardness result or standard break is part of this plan's acceptance.
