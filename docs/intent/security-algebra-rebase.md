# Intent — Security-algebra rebase

- **Owner:** lead (direction); corpus lead (Apeiron) for the canonical transport; Sol for grades
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c` (working tree dirty)
- **Status:** captured, **not actioned** — staged for a later epic after the growable-table work
- **Spec / Plan:** `docs/plan/security-algebra-rebase.md` (waves, gates, review obligations)

## Problem

The old Sanguine corpus contains a large, largely proved security and cryptography module, and a
decentralised networking / federation substrate. It predates the standardized trichotomy
`R[x]/(x²−σ)` and is written in the older "Being cell / Null cell / `x = B/U`" vocabulary. Left
unrebased it would force the programme to re-derive results it already has. This record is a
**foundation for deciding the corpus's cryptographic claims one way or the other**, not an
adjudication of them. The central open question is the old **"Null cell"**, which names what appear
to be two algebraically distinct objects at once — a **split** (nontrivial idempotents, zero
divisors) and a **collapse** (nilpotent/truncating, no complement) — and whether the corpus's
security results actually separate along the trichotomy faces. The corpus also advances a
**minimum encryption standard** (one-time-pad-grade secrecy and a hash/SHA-256-class face) as the
floor for *any* task, bound to federation and the networking layer, as the content of named
theorems. This record states the claims at their own grades, the tests that would settle them, and
the deeper review, so none of it is lost and none of it is presupposed.

## Method — no imported priors

The record exists to make the claims **decidable**, not to endorse or reject them. It therefore:

1. states each result at the grade the source declares, and neither upgrades nor downgrades it;
2. states, for every load-bearing claim, what would **confirm** it and what would **refute** it;
3. treats mainstream computational cryptography and the corpus's information-theoretic and
   structural positions **symmetrically** — as bodies of claims under test, never as the default
   against which the other is judged;
4. leaves every verdict open until a named theorem, a reproducible refutation, or a documented
   undecidable outcome exists.

An imported prior, in either direction, is a defect of this record, not evidence.

## End state

The arc ends in an **end-to-end formalized implementation** of each target standard, not a survey:

1. An **end-to-end Rust implementation** per target, based on that standard's canonical/trusted
   reference implementation, that the programme **formally proves** — the real code refines the spec
   through the Apeiron extraction lane, and the spec's algebraic claims are checked.
2. The **primitives it needs are extracted into Apeiron's core** as reusable, refined objects (the
   `Cell`/`RingCell` word kernels, the field/extension rings, the permutation/compression cores) if
   they are not already there.
3. Most likely a **secondary crypto repository** holds the implementations and their proofs,
   consuming Apeiron's core primitives under a pinned correspondence — the same shape as the
   existing sibling repos, so the provenance core is not enlarged by each standard.

Each implementation is then asked the three primary questions below, and its answer is the
deliverable for that standard.

## Three questions asked of each implementation

1. **Does it achieve its stated goals, formally and empirically?** Formally: does the real code
   refine the spec, and are the spec's algebraic/security claims checked? Empirically: do the
   measured properties hold (the standard's own test vectors, byte-exactness, timing/constant-time,
   entropy of key material, refusal behavior)? Both senses are required; a formal-only or
   empirical-only answer is recorded as **partial**.
2. **Are there structural, algebraic or fundamental mathematical weaknesses?** Locate the algebraic
   handle: unintended quotients, idempotents, zero divisors, nilpotents, hidden homomorphisms, or a
   collapse where a bijection was intended — and whether any hardness claim rests on a structure
   present in the object. An unresolved outcome is recorded as **undecided**, not as safe.
3. **Are there implementation weaknesses from the empirical hacks?** The gap between spec and
   deployed artifact: the reference implementation's own heuristics, the code-to-spec hole in
   vertical proofs (Jasmin/Vale/HACL*-style: code refines a document whose security is not proved),
   the syscall/ABI and side-channel surface, and the protocol/mode conventions that have historically
   broken (nonce reuse, padding oracles, IV reuse). Empirical weaknesses are first-class findings,
   not footnotes.

The three questions are the verification question (formal + empirical), the cryptanalysis question
(structural/algebraic), and the implementation question (engineering/protocol), asked of the same
artifact.

## What the corpus declares (inventory, `../sanguine/proof`)

`Substrate/Algebra/Security/` (17 files):

| File | Named results (as the file states them) | Source-declared grades |
|---|---|---|
| `SecurityCore.lean` | `xorStream_involutive`, `otp_changes_the_wire` | proved (native), `SecurityCore.lean:16,21`; passive-reader secrecy anchor, `:24` |
| `SanguineSecurity.lean` | confidentiality, channel, protocol, federation and headline results | proved (native/riding), `SanguineSecurity.lean:72-101`; obfuscation/threat reading anchor, `:102-110`; live wire open, `:115` |
| `EncryptionProjection.lean` | injective/collision fork and fiber results | proved (native/riding), `EncryptionProjection.lean:69-104`; AES/RSA/SHA placement anchor, `:94-100` |
| `DescentIsTheEasyInverse.lean` | adjoin/strip and collapse results | proved (native/riding), `DescentIsTheEasyInverse.lean:35-46`; autotropism, half-cell and cost readings open, `:47-55` |
| `MandalaMemory.lean` | masked-region arithmetic | proved (native), `MandalaMemory.lean:33-41,56`; emitter/ISA claim anchor, `:42-55` |
| `RaceFreedom.lean` | functional-store commutation | proved (Mathlib/native/riding), `RaceFreedom.lean:31-39`; concurrent data-race interpretation next/anchor, `:40-52` |
| `CapProvenance.lean` | grammar provenance | proved (native/riding), `CapProvenance.lean:29-40,49`; compiler grammar correspondence anchor, `:41-48` |
| `GoldenStorage.lean` | injective placement and seed append | proved (native/Mathlib/riding), `GoldenStorage.lean:34-48`; cost/storage readings anchor and peeling open, `:49-62` |
| `FoldDepth.lean` | sequence, split, fiber and depth claims | mixed proved/riding/anchor/next/retired at `FoldDepth.lean:77-134`; no file-wide grade |
| `HashTrinityCore.lean` | table fold and estimation/BLUP results | proved (native/Mathlib), `HashTrinityCore.lean:29-52`; interpretations anchor/open, `:53-70`; `get_insertWith_self`/`get_insertWith_ne` have no source-declared grade |
| `KeyResolution.lean` | fixed-crosswalk resolution | proved (native/riding), `KeyResolution.lean:43-51`; n-key extension next, `:55`; headline grade unconfirmed |
| `IncrementalResolution.lean` | prefix-reading counterexample and fixed bridge | proved (native/riding), `IncrementalResolution.lean:43-52`; quantified negative open, `:65`; headline grade unconfirmed |
| `ResumableProbe.lean` | list/map resumability | proved (native), `ResumableProbe.lean:23-30`; golden grounding and mutable interleaving open, `:31,45` |
| `TimerWheel.lean` | relational Nat bands and slot bound | proved (native/riding), `TimerWheel.lean:34-41`; runtime/cost readings anchor/open, `:42-56` |
| `WorkStealing.lean` | Nat/list deque and reduction model | proved (native/Mathlib), `WorkStealing.lean:62-72`; exclusivity and composition next, `:73-85`; divided bound open, `:93` |
| `AggregationBound.lean` | linear hypotheses, adversarial refutations and sweep bound | proved (native/Mathlib/riding), `AggregationBound.lean:74-138`; better adversarial asymptote open, `:139` |
| `BeliefContraction.lean` | expansion, incision and recovery results | proved (native/Mathlib/riding), `BeliefContraction.lean:63-79`; AGM/tracker readings anchor and entrenchment correspondence open, `:80-94` |

The plan's [ratified theorem and claim ledger](../plan/security-algebra-rebase.md#ratified-theorem-and-claim-ledger)
is normative for every individual result, source line, face and canonical target. Supporting modules:
`Substrate/Algebra/NullCell/NullCell.lean` (`phase_no_zero_divisors`,
`phase_invertible`, `null_not_invertible`, `null_times_mirror`, `e_orthogonal`);
`Substrate/Algebra/Navigation/Federation.lean` (`federation_algebra`, `federation_defense`,
`member_sovereign`, `survivors_state_exact`, `federation_reseeds_the_genome`,
`honesty_decided_at_the_cell`, `conflict_floor`, `scheduler_floor`);
`Closures/Applied/WebCipher.lean` (`otp_perfect_secrecy`, `computational_cipher_pins_the_plaintext`,
`web_edge_is_the_pad`, `cover_is_indistinguishable`, `meaning_invariant_under_disguise`,
`routing_is_lossless`); `Trinity/RecoveryTrinity.otp_perfect_secrecy`;
`Runtime/SangNet.lean` (`gateway`/`via`, `frictionFree`, pad material); `ThreatModel.lean`,
`UniversalThreat.lean`, `ProtocolZoning.lean`, `StructuralFriction.lean`.

Prose and tooling to rebase: `theory/THE_THREAT_MODEL.md` (STRIDE/PASTA/TRIKE rotation, threat =
non-intertwining contact), `THE_HARDENING.md` (attack-shape map), `SECURITY_TOOLS.md`
(`sang guard` WAF, `Wire.friction`), `P2P_SURVIVAL_LAYER.md` (the deleted P2P/Tor/Bitcoin/
Matrix/QUIC lifts), `THE_FLAW_TRINITY.md`, `MessageScheduleTopology.lean`,
`THE_PASS_THE_PARCEL_CHAIN.md`, `THE_HASHMAP_PATH.md`, `experiments/{otp_battery.py,waf_rules,vuln_form}`.

## The minimum-encryption standard (corpus position, to be rebased as a theorem)

The corpus rejects deterministic cost-based secrecy and takes a stronger floor:

1. **A bijective cipher cannot be information-theoretically secure.** Its ciphertext's consistent
   plaintext set is a singleton, so its secrecy is only inversion cost — `cipher_secrecy_is_only_cost`
   (proved) and `computational_cipher_pins_the_plaintext` (proved). A cost-based cipher is a lock to
   be picked, not a secret.
2. **The floor is a deliberately non-injective public map.** The dropped window is a
   σ>0 **split** channel (`e₊e₋=0`): the distinguishing information is not in the ciphertext at all
   — `channel_blind_to_one_flaw`, `sanguine_encryption_information_theoretic_security`,
   `otp_perfect_secrecy`, `WebCipher.web_cipher`.
3. **The other floor is the collapse/hash face** (σ=0 parabolic): many-to-one, no unique floor, no
   decryptor-to-a-point — `collision_breaks_unique_floor`, `collision_fiber_is_not_a_singleton`.
4. **The key re-attaches the dropped factor** (`channel_recovered_with_key`, `round_trip_id`), and
   the conserved seed survives the lossy ratio (`information_survives_the_horizon`,
   `B/(B/U) = U`).
5. **The standard is bound to federation**: sovereign members compose/decompose losslessly, the
   whole reseeds from any survivor, and the wire is the pad
   (`Federation.federation_reseeds_the_genome`, `survivors_state_exact`,
   `WebCipher.web_edge_is_the_pad`).

The user's framing — SHA-256-class and OTP as the **minimum** level for *any* task — is the corpus's
position read forward: (2)+(3) are the only two ends that do not rest on an assumed hardness of a
structure present in the object.

## Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Leave the old security corpus behind and re-derive | Re-runs proved work and loses the falsifiable record; the user's explicit concern. |
| Rebase by copying the old "Being/Null" vocabulary unchanged | Pre-empts the central question by keeping one label for two objects; the face assignment must be decided by the transport, not assumed. |
| Presuppose either the mainstream computational standard or the corpus's information-theoretic standard | Choosing either in advance biases the investigation. Both are recorded as claims and weighed by evidence, not by prior. |
| Assert the corpus has settled modern cryptography | Its own files label the AES/RSA bridge **named, not re-proven** and the SHA readings conjecture/anchor; settlement would outrun the stated grades. |

## Rebase map (old vocabulary → trichotomy)

| Old | Algebra | Standardized face | Exact canonical targets |
|---|---|---|---|
| Being cell `ℂ`, `i²=−1`, `/` total, `phase_invertible`, `phase_no_zero_divisors` | compact/field, no zero divisors | **elliptic, `σ < 0`** | `Apeiron/Trichotomy/Basic.lean`: `normForm_eq_zero_iff_of_neg`, `eq_zero_or_eq_zero_of_mul_eq_zero_of_neg`, `noZeroDivisors_iff_disc_neg`, `isUnit_iff_normForm_ne_zero` |
| Null cell `j²=+1`, `(1+j)(1−j)=0`, `null_times_mirror`, `e_orthogonal` | split, nontrivial idempotents | **hyperbolic, `σ > 0` (the split)** | `Apeiron/Trichotomy/Basic.lean:zeroDivisorPair_of_disc_one`; `Apeiron/Trichotomy/ChannelsAnyRing.lean:ringUpper_channel,ringLower_channel`; `Apeiron/Trichotomy/IdempotentsAnyRing.lean:ringComplementary_idem` |
| “collapse”, `collision_breaks_unique_floor`, `cellCollapse_off_frame_one_way` | many-to-one, nilpotent, no complement | **parabolic, `σ = 0` (the collapse)** | `Apeiron/Trichotomy/Basic.lean:nilpotent_of_disc_zero`; no canonical cipher/hash theorem |
| `x = B/U`; `B` seed, `U` dropped window, `/` key | statement-dependent | split into its applicable faces | `Apeiron/Trichotomy/AnyRing.lean:RingCell,ringMul,ringNormForm`; definitions confer no secrecy grade |

The fused “Null” must split: the old OTP claim is transported for confirmation or refutation on the
`σ > 0` orthogonal-channel face, while the one-way/hash claim is transported for confirmation or
refutation on the `σ = 0` nilpotent collapse face. Mapping the fused label to either face alone is
invalid; neither assignment proves the security interpretation.

## Invariants

1. **Statements are transported, not weakened.** Every rebased theorem keeps its quantifiers,
   domain and true width; a "proved (riding)" result stays riding; an "anchor" stays an anchor.
2. **Grades are frozen, not adjudicated.** A claim's grade moves only on a named theorem (upgrade)
   or a reproducible refutation (downgrade); the record neither assumes nor denies hardness.
3. **Symmetry.** Mainstream computational security and the corpus's positions are recorded with the
   same standard of evidence; neither is the default.
4. **`(Sigma, K)` identity and the algebra are unchanged**: the primitive remains
   `R[x]/(x²−σ)`, `Cell<i64>` the word kernel; the security layer is read *off* it, not added under it.
5. **The minimum standard is stated as a claim pair**: cost-based secrecy rejected; split-channel
   and collapse ends admitted, with the key re-attachment and the reseed — each with its falsifier.
6. **The federation substrate is a first-class consumer**, not an appendix: composition,
   decomposition, seed/reseed, the adversarial cell verdict and the conflict/scheduler floors.
7. **Every load-bearing claim is testable.** A claim with no possible test is recorded as such and
   is not treated as true. Nothing in the record is deleted from the source history.


## Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| A bijective cipher's consistent-plaintext set is a singleton, so its secrecy is inversion cost. | proved (old corpus) | `cipher_secrecy_is_only_cost`; `computational_cipher_pins_the_plaintext` |
| The dropped-window/public-map construction is OTP-grade. | proved (old corpus) | `sanguine_encryption_information_theoretic_security`, `otp_perfect_secrecy` |
| OTP-grade secrecy is the σ>0 split face; the one-way wall is the σ=0 parabolic face. | anchor → proved target | rebase map; `ringUpper/Lower_channel` vs `nilpotent_of_disc_zero` |
| The federation composes/decomposes losslessly and reseeds from any survivor. | proved (old corpus) | `federation_algebra`, `federation_reseeds_the_genome` |
| SHA-256-class primitives sit on the collapse face with no exploitable algebra. | anchor | `EncryptionProjection`; SHA readings are labelled conjecture/open in their files |
| AES/RSA hardness reduces to the bulk model's unique-floor collapse. | anchor/open | `EncryptionProjection.lean:62` — named, **not re-proven** |
| "Most existing security is flawed in fundamental ways solvable in O(1)" is a claim about locating an exploitable algebraic handle, not a general break. | anchor | the corpus's attack-as-unintended-quotient reading; requires the deeper review below |

## Non-goals

- No claim to have broken or to have vindicated SHA-256, AES, RSA or any standard.
- No presupposition that one cryptographic standard is correct; the investigation decides.
- No re-litigation of the storage/table epic; this runs after it.
- No new prose ontology (`//@` etc.); results go to Lean, tests or governing documents.

## Trust boundary

The rebase **transports** the old corpus's grades; it neither upgrades nor downgrades them and does
not adjudicate hardness in either direction. Mainstream computational security is recorded as an
assumption, and the corpus's information-theoretic and structural claims as claims, until a named
theorem or a reproducible refutation moves them. The old corpus's own labels
(`DEFENDED`/`BELOW-CUT`/`NAMED-OPEN`/`FINDING`, and per-file `anchor`/`open`/conjecture notes) are
authoritative over any summary here. The federation and P2P material includes deleted artifacts kept
only as a historical record.

## Open questions — the deeper review

Each is stated with what would confirm and what would refute it; an unresolved outcome is itself a
result.

- `THE_PASS_THE_PARCEL_CHAIN.md`: "SHA-256d is technically EASIER to break than SHA-256" —
  **confirm**: a reduction of SHA-256d preimage to SHA-256 preimage, or a stated attack gap;
  **refute**: a proof the doubling adds work, or a counterexample; **else** undecided. (`TensionPinch.lean`
  labels the SHA reading conjecture.)
- `MessageScheduleTopology.lean`: preimage resistance localises to the message schedule —
  **confirm**: a named theorem at the stated scope; **refute**: an attack that does not factor
  through the schedule; **else** anchor.
- `EncryptionProjection.lean:62`: the AES/RSA bulk-collapse bridge — **confirm**: a checked
  reduction; **refute**: a break of the reduction's premise; **else** open (labelled "named, not
  re-proven").
- `Federation.lean`: `honesty_decided_at_the_cell` / `law_violation_is_detectable` and the
  leaderless `conflict_floor` / `scheduler_floor` — **confirm**: transport at unchanged statements;
  **refute**: a model in which an honest member is convicted or a dishonest member passes;
  **else** the transport's own counterexamples.
- The deleted P2P/Tor/Bitcoin/Matrix/QUIC lifts (`P2P_SURVIVAL_LAYER.md`): historical record only;
  decide whether `docs/direction.md`'s "Federation / distributed" row absorbs them or they stay
  historical. Not a security claim either way.
