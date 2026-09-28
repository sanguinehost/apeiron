# Security-algebra rebase

Merged epic record: `intent/security-algebra-rebase.md`, `spec/security-algebra-rebase.md`, `plan/security-algebra-rebase.md` (one file per epic; see `docs/README.md`).

## Intent — Security-algebra rebase

- **Owner:** lead (direction); corpus lead (Apeiron) for the canonical transport; Sol for grades
- **Source revisions:** old Sanguine `e830d3fbdeb40319d1be0d1fc803e706748b6a3c`;
  canonical Apeiron `3d66ea3aa2fe1de5997b6b757247454a2f771358`
- **Status:** attribution review active; canonical transport and security conclusions open
- **Spec / Plan:** [specification](security-algebra-rebase.md) and
  [plan](security-algebra-rebase.md)

### Problem

The old Sanguine corpus contains a large, largely proved security and cryptography module, and a
decentralised networking / federation substrate. It predates the standardized trichotomy
`R[x]/(x²−σ)` and is written in the older "Being cell / Null cell / `x = B/U`" vocabulary. Left
unrebased it would force the programme to re-derive results it already has. This record is a
**foundation for deciding the corpus's cryptographic claims one way or the other**, not an
adjudication of them. The pinned old **"Null cell"** is a split real algebra (`j²=1`,
zero divisors). The canonical parabolic example has a nonzero square-zero point; generic
information loss need not come from that example. The open question is which source statements
admit exact face transport and which are generic observer, key or map statements. The corpus also advances a
**minimum encryption standard** (one-time-pad-grade secrecy and a hash/SHA-256-class face) as the
floor for *any* task, bound to federation and the networking layer, as the content of named
theorems. This record states the claims at their own grades, the tests that would settle them, and
the deeper review, so none of it is lost and none of it is presupposed.

### Method — no imported priors

The record exists to make the claims **decidable**, not to endorse or reject them. It therefore:

1. states each result at the grade the source declares, and neither upgrades nor downgrades it;
2. states, for every load-bearing claim, what would **confirm** it and what would **refute** it;
3. treats mainstream computational cryptography and the corpus's information-theoretic and
   structural positions **symmetrically** — as bodies of claims under test, never as the default
   against which the other is judged;
4. leaves every verdict open until a named theorem, a reproducible refutation, or a documented
   undecidable outcome exists.

An imported prior, in either direction, is a defect of this record, not evidence.

### End state

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

### Three questions asked of each implementation

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

### What the corpus declares (inventory, `../sanguine/proof`)

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

The plan's [source-grade theorem and claim ledger](security-algebra-rebase.md#source-grade-theorem-and-claim-ledger)
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

### The minimum-encryption standard (corpus position, to be tested)

The corpus rejects deterministic cost-based secrecy and takes a stronger floor:

1. `cipher_fiber_is_a_singleton` proves a singleton plaintext fiber for one fixed injective
   public function. `cipher_secrecy_is_only_cost` rides it, but its source labels inversion cost
   a reading without a cost model. A hidden-key family has a different observer experiment.
2. `channel_blind_to_one_flaw` proves one collision under the real split-cell readout;
   `otp_perfect_secrecy` proves one compatible additive key for every message/ciphertext pair.
   Neither statement specifies a key distribution or observer indistinguishability. The
   [specification](security-algebra-rebase.md) supplies a finite confirm/refute pair.
3. `collision_fiber_is_not_a_singleton` concerns a generic many-to-one map. Canonical
   `nilpotent_of_disc_zero` witnesses the parabolic algebra only. A particular hash or
   compression map needs a separate representation theorem before it receives that face.
4. **The key re-attaches the dropped factor** (`channel_recovered_with_key`, `round_trip_id`), and
   the conserved seed survives the lossy ratio (`information_survives_the_horizon`,
   `B/(B/U) = U`).
5. **The standard is bound to federation**: sovereign members compose/decompose losslessly, the
   whole reseeds from any survivor, and the wire is the pad
   (`Federation.federation_reseeds_the_genome`, `survivors_state_exact`,
   `WebCipher.web_edge_is_the_pad`).

The SHA-256-class/OTP minimum is the corpus's proposed standard. The current algebraic
statements do not prove it for every task or deployed system.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Leave the old security corpus behind and re-derive | Re-runs proved work and loses the falsifiable record; the user's explicit concern. |
| Rebase by copying the old "Being/Null" vocabulary unchanged | Pre-empts the central question by keeping one label for two objects; the face assignment must be decided by the transport, not assumed. |
| Presuppose either the mainstream computational standard or the corpus's information-theoretic standard | Choosing either in advance biases the investigation. Both are recorded as claims and weighed by evidence, not by prior. |
| Assert the corpus has settled modern cryptography | Its own files label the AES/RSA bridge **named, not re-proven** and the SHA readings conjecture/anchor; settlement would outrun the stated grades. |

### Rebase map (old vocabulary → trichotomy)

| Old | Algebra | Standardized face | Exact canonical targets |
|---|---|---|---|
| Being cell `ℂ`, `i²=−1`, `/` total, `phase_invertible`, `phase_no_zero_divisors` | compact/field, no zero divisors | **elliptic, `σ < 0`** | `Apeiron/Trichotomy/Basic.lean`: `normForm_eq_zero_iff_of_neg`, `eq_zero_or_eq_zero_of_mul_eq_zero_of_neg`, `noZeroDivisors_iff_disc_neg`, `isUnit_iff_normForm_ne_zero` |
| Null cell `j²=+1`, `(1+j)(1−j)=0`, `null_times_mirror`, `e_orthogonal` | split, nontrivial idempotents | **hyperbolic, `σ > 0` (the split)** | `Apeiron/Trichotomy/Basic.lean:zeroDivisorPair_of_disc_one`; `Apeiron/Trichotomy/ChannelsAnyRing.lean:ringUpper_channel,ringLower_channel`; `Apeiron/Trichotomy/IdempotentsAnyRing.lean:ringComplementary_idem` |
| Canonical square-zero witness | nonzero nilpotent | **parabolic, `σ = 0`** | `Apeiron/Trichotomy/Basic.lean:nilpotent_of_disc_zero`; generic collisions have no forced face |
| `x = B/U`; `B` seed, `U` dropped window, `/` key | statement-dependent | split into its applicable faces | `Apeiron/Trichotomy/AnyRing.lean:RingCell,ringMul,ringNormForm`; definitions confer no secrecy grade |

Old `NCell` maps to the `σ=1` split algebra only after exact statement transport. A separately
defined nilpotent statement can map to `σ=0`. Generic OTP/group equations, collisions, fixed-map
fibers, XOR streams and federation identities have no sign-specific face merely from their names.

### Invariants

1. **Statements are transported, not weakened.** Every rebased theorem keeps its quantifiers,
   domain and true width; a "proved (riding)" result stays riding; an "anchor" stays an anchor.
2. **Grades are frozen, not adjudicated.** A claim's grade moves only on a named theorem (upgrade)
   or a reproducible refutation (downgrade); the record neither assumes nor denies hardness.
3. **Symmetry.** Mainstream computational security and the corpus's positions are recorded with the
   same standard of evidence; neither is the default.
4. **`(Sigma, K)` identity and the algebra are unchanged**: the primitive remains
   `R[x]/(x²−σ)`, `Cell<i64>` the word kernel; the security layer is read *off* it, not added under it.
5. **The minimum standard remains a claim under test**: distinguish fixed-function fibers,
   hidden-key observation, split channels and nilpotent algebra; each has a falsifier.
6. **The federation substrate is a first-class consumer**, not an appendix: composition,
   decomposition, seed/reseed, the adversarial cell verdict and the conflict/scheduler floors.
7. **Every load-bearing claim is testable.** A claim with no possible test is recorded as such and
   is not treated as true. Nothing in the record is deleted from the source history.


### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| A fixed injective public function has a singleton plaintext fiber. Inversion cost is a reading. | proved (old corpus) for fiber; anchor for cost | `cipher_secrecy_is_only_cost`; `cipher_fiber_is_a_singleton`; source states no cost model |
| Every message/ciphertext pair admits a unique additive key; OTP-grade observer secrecy is separate. | proved (old corpus) for key existence; anchor for secrecy | `RecoveryTrinity.otp_perfect_secrecy`; finite biased-pad refuter in spec |
| Old `NCell` is split at `σ=1`; canonical square-zero witness is parabolic at `σ=0`. Generic information loss has no forced face. | proved algebraic statements; transport open | `zeroDivisorPair_of_disc_one`; `nilpotent_of_disc_zero`; representation bridge missing |
| The federation composes/decomposes losslessly and reseeds from any survivor. | proved (old corpus) | `federation_algebra`, `federation_reseeds_the_genome` |
| SHA-256-class primitives sit on the collapse face with no exploitable algebra. | anchor | `EncryptionProjection`; SHA readings are labelled conjecture/open in their files |
| AES/RSA hardness reduces to the bulk model's unique-floor collapse. | anchor/open | `EncryptionProjection.lean:62` — named, **not re-proven** |
| "Most existing security is flawed in fundamental ways solvable in O(1)" is a claim about locating an exploitable algebraic handle, not a general break. | anchor | the corpus's attack-as-unintended-quotient reading; requires the deeper review below |

### Non-goals

- No claim to have broken or to have vindicated SHA-256, AES, RSA or any standard.
- No presupposition that one cryptographic standard is correct; the investigation decides.
- No re-litigation of the storage/table epic. Attribution and finite interpretation checks
  proceed now; canonical transport and implementation have their own later barriers.
- No new prose ontology (`//@` etc.); results go to Lean, tests or governing documents.

### Trust boundary

The rebase **transports** the old corpus's grades; it neither upgrades nor downgrades them and does
not adjudicate hardness in either direction. Mainstream computational security is recorded as an
assumption, and the corpus's information-theoretic and structural claims as claims, until a named
theorem or a reproducible refutation moves them. The old corpus's own labels
(`DEFENDED`/`BELOW-CUT`/`NAMED-OPEN`/`FINDING`, and per-file `anchor`/`open`/conjecture notes) are
authoritative over any summary here. The federation and P2P material includes deleted artifacts kept
only as a historical record.

### Open questions — the deeper review

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

## Specification — security algebra attribution

- Owner: Sol for statement and grade attribution; canonical corpus owner for later transport.
- Identity: old Sanguine `e830d3fbdeb40319d1be0d1fc803e706748b6a3c`;
  canonical Apeiron `3d66ea3aa2fe1de5997b6b757247454a2f771358`.
- Status: contract for finite interpretation checks. No deployed cipher, standard or protocol
  satisfies it yet. The ledger in [the plan](security-algebra-rebase.md) records old
  source-native grades; those grades do not transfer to this specification.

### Algebra and selected domains

| Question | Domain and exact source | Boundary |
|---|---|---|
| Split face | Old `NullCell.NCell = ℝ × ℝ`, `j² = 1`; canonical real `Cell` at `σ = 1` in `Trichotomy/Basic.lean:zeroDivisorPair_of_disc_one`. | A checked representation map is needed to transport the old multiplication. `IdempotentsAnyRing.ringComplementary_idem` separately requires `r*r=σ`, `r*s=1`, `2*h=1`; `ChannelsAnyRing.ringUpper_channel` and `ringLower_channel` require `r²=σ`. Supply joint witnesses at the chosen scalar. |
| Dual-number face | Canonical real `Cell` at `σ = 0`; `Basic.nilpotent_of_disc_zero` proves a nonzero square-zero point. | This does not classify an arbitrary many-to-one map, coordinate projection or SHA component. Hyperbolic channel projections can also discard a coordinate. |
| Finite interpretation fixture | `G = ZMod 2` as an additive group, message and pad in `G`, ciphertext `m+k`; key law `μ : G → ℚ` with nonnegative masses summing to one. | This is a model of an observer experiment, not a transport of the old real split cell and not a Rust cipher. `RecoveryTrinity.otp_perfect_secrecy` holds for every additive commutative group; it asserts `∀ c m, ∃! k, m+k=c`, with no `μ`. |
| Word execution | `Cell<i64>` and table paths retain their own extracted refinement obligations. | A real-half witness cannot be reused in the wrapping word ring, where `2*h=1` is unavailable. No source-connected security claim follows from the real theorem. |

No scalar is selected for a target standard. A later standard selects its actual field/ring, key
space, representation and reference revision before a transport or implementation task begins.

### Key, observer and claim contracts

Let `Enc m k := m+k` on the finite fixture. The observer sees only `c`; the key is sampled
independently of `m` from `μ` and is not supplied to that observer. Define
`Prμ[c | m] := μ(c-m)`. Perfect secrecy in this fixture is the equation
`∀ m₀ m₁ c, Prμ[c | m₀] = Prμ[c | m₁]`. A uniform `μ` confirms this equation;
`μ(0)=3/4`, `μ(1)=1/4`, `m₀=0`, `m₁=1`, `c=0` refutes it while unique compatible keys remain.
An observer holding `k` has a different experiment: `(m+k)-k=m`; a fixed public
`f : α → β` with `Injective f` has another, single-function fiber contract. Neither experiment
substitutes for the hidden-key experiment.

| Source-native statement | What it establishes | Confirm/refute interpretation |
|---|---|---|
| `RecoveryTrinity.otp_perfect_secrecy`; `WebCipher.web_otp_perfect_secrecy` | Unique compatible key for each `c,m`, proved in the old source. | Confirm the hidden-key distribution equation under uniform independent pads; refute the implication from key existence alone with the biased fixture. |
| `EncryptionProjection.cipher_fiber_is_a_singleton`; `SanguineSecurity.cipher_secrecy_is_only_cost` | A singleton fiber of one fixed injective `f`, proved native/riding in the old source. The latter source explicitly labels inversion cost a reading without a cost model. | Confirm a cost claim only with an observer, oracle and cost bound; refute transfer to a secret-key family by distinguishing `keyed k` from public `(m,k) ↦ m+k`. Do not assert either standard secure or insecure from fiber cardinality alone. |
| `SanguineSecurity.channel_blind_to_one_flaw` | One pair of distinct real split-cell states with the same `readPlus`, proved native. | Confirm channel-specific indistinguishability only after a distribution and observation relation; refute the universal interpretation with a differing observable channel or a biased input. |
| `SecurityCore.xorStream_involutive`; `otp_changes_the_wire` | Round trip for every list and one non-identity witness, proved native. | The fixture `key=[1], data=[0,1]` leaves the second data element unchanged. Any full-coverage claim must reject it or state a key-length admission condition. |
| `Federation.honesty_decided_at_the_cell`; `law_violation_is_detectable` | Universal one-cell equation and detection from a supplied violating pair, proved native. | A finite sampled-probe completeness claim needs an exhaustive finite domain or a sampling guarantee; refute unrestricted completeness with an unqueried bad pair. |
| `Federation.federation_reseeds_the_genome` | Decode a diagonal survivor copy through a supplied codec, proved riding. | A deployment claim needs survivor availability, authenticity and a failure model; a projection identity alone does not establish them. |

`EncryptionProjection.decryptRecovery` takes the target plaintext `a₀`; its unique-zero
residual is not an algorithm to find an unknown plaintext. AES/RSA placement is explicitly
“named, not re-proven” in that file. SHA-256d, schedule localization, hardness and side-channel
claims need separate named targets, reductions and measurements. Their present grade is anchor/open.

### Admission, controls and trust

The first proof task checks only finite interpretation witnesses. It must preserve the old
unique-key theorem as true in both uniform and biased cases, prove the conditional uniform
observation equation, and check a biased counterexample to the unconditional inference. Its
negative control changes `3/4,1/4` to `1/2,1/2` in a disposable copy and requires the named
bias witness to fail. A short-key passthrough guard is independent of the probability result.
The Lean gate and audit check this model, not current Rust execution or a cryptographic standard.

Canonical transport needs a source statement, an exact carrier map and a checked target under
the canonical repository's own bounded gate and audit. A new canonical security theorem is not
named in advance. A later consumer must pin its revision and separately check extraction,
source correspondence, witnesses and controls. Codegen, entropy source, key delivery, side
channels, network adversaries and deployed protocols remain outside this specification.

## Plan — Security-algebra rebase

- **Owner:** lead (direction); Sol (grades, attribution, contract verdicts); corpus lead (transport)
- **Intent:** `docs/epics/security-algebra-rebase.md`
- **Spec:** [security-algebra-rebase](security-algebra-rebase.md)
- **Base:** `262bb0a94eff2cc7d5655122a954f44d74de96b6`, plus accepted dirty
  `target/open-areas/baseline.json`; task `APEIRON-SECURITY-W1-T1` in its isolated worktree.
- **Status:** attribution review active. The finite interpretation task below can follow the
  documentary gate; canonical transport awaits a specified contract, owner and local gates.
- **Source corpus:** `../sanguine` (old Sanguine proofs and prose), read-only; canonical target
  `../sanguine-apeiron/proof/Apeiron` (`3d66ea3aa2fe1de5997b6b757247454a2f771358`).
  Old source revision: `e830d3fbdeb40319d1be0d1fc803e706748b6a3c`.

```text
SCOPE: docs/epics/security-algebra-rebase.md, docs/epics/security-algebra-rebase.md,
       docs/epics/security-algebra-rebase.md; freeze source attribution and bounded tasks.
REGISTER: planning/algebra attribution; no canonical transport, deployed secrecy or standard verdict.
CORPUS: pinned Basic.nilpotent_of_disc_zero, zeroDivisorPair_of_disc_one;
        IdempotentsAnyRing.ringComplementary_idem needs r²=σ, rs=1, 2h=1;
        old RecoveryTrinity.otp_perfect_secrecy is unique compatible key.
REFERENCE: /home/socol/Workspace/sanguine/proof/Substrate/Algebra/Trinity/RecoveryTrinity.lean;
           /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Trichotomy/Basic.lean;
           both read at the pinned revisions above.
GATE: complete-file/source attribution and git diff --check in this worktree;
      prior semantic security gate MISSING; documentary controls specified below.
GRADE: old source-native/riding retained; new interpretation is anchor, no Lean build claimed.
```

### What this plan is not

It is not a new cryptography and it is not an adjudication. It transports an existing, largely
proved security module onto the standardized trichotomy, preserves each result's declared grade, and
states the corpus's minimum-encryption standard as a claim to be confirmed or refuted. It decides
nothing in advance: mainstream computational security and the corpus's positions are weighed by
evidence, and an unresolved outcome is a valid result. Every load-bearing row names what would
refute it, and a row with no possible refutation is a defect of the plan.

### End state and the three questions

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

### Later per-standard implementation

Each target needs a selected standard, reference revision, threat model and scalar/key domain
before implementation is decomposed. No target is selected here. Rust implementation, extraction,
algebraic claims, empirical vectors and review require separate ≤3-file tasks with one gate and
one control each; this section is an outcome, not a dispatchable wave.

### Rebase map (the contract every transport row obeys)

| Old vocabulary | Standardized face | Exact canonical targets |
|---|---|---|
| Being cell `ℂ`, `i²=−1`, `/` total | elliptic, `σ < 0` | `Apeiron/Trichotomy/Basic.lean`: `normForm_eq_zero_iff_of_neg`, `eq_zero_or_eq_zero_of_mul_eq_zero_of_neg`, `noZeroDivisors_iff_disc_neg`, `isUnit_iff_normForm_ne_zero` |
| Null cell `j²=+1`, null pair, complementary channels | hyperbolic, `σ > 0` (the split) | `Apeiron/Trichotomy/Basic.lean`: `zeroDivisorPair_of_disc_one`; `Apeiron/Trichotomy/ChannelsAnyRing.lean`: `ringUpper_channel`, `ringLower_channel`; `Apeiron/Trichotomy/IdempotentsAnyRing.lean`: `ringComplementary_idem` |
| nonzero square-zero canonical cell | parabolic, `σ = 0` | `Apeiron/Trichotomy/Basic.lean`: `nilpotent_of_disc_zero`; no canonical cipher/hash theorem exists |
| generic collision or dropped coordinate | face unassigned | Needs an exact carrier/operation representation; a hyperbolic channel projection may also lose a coordinate. |
| `x = B/U`, `B` seed, `U` dropped window | statement-dependent; never one fused face | `Apeiron/Trichotomy/AnyRing.lean`: `RingCell`, `ringMul`, `ringNormForm`; these definitions alone establish no secrecy claim |

**Reject rule:** transport pinned old `NCell` multiplication as the `σ=1` split algebra only after
an exact representation theorem, with joint root/inverse/half witnesses for any generic
idempotent theorem. A `σ=0` statement needs a separately identified square-zero source.
Generic collisions, dropped coordinates, OTP/group equations, fixed-function fibers and
federation identities have no forced face. Neither face proves secrecy by itself.

#### Source-grade theorem and claim ledger

Paths in the first column are relative to `/home/socol/Workspace/sanguine/`. A grade citation is the
source line that declares the grade, not the theorem line. The face column records the old
interpretive proposal, not a proved face assignment; a generic statement remains **face unassigned**
until a representation theorem names its algebra. `all faces` means the statement is generic.
`none` means the pinned canonical corpus has no theorem with the stated contract; a nearby
algebraic theorem is named only in the note. The previous face labels for generic collision,
fiber, XOR, arithmetic-mask and schedule statements are candidates only and cannot support a
security headline. The specification controls interpretation and observer assumptions.

| Old file : old name | Declared grade and source | Face candidate / status | Exact canonical target | Note |
|---|---|---|---|---|
| `proof/Substrate/Algebra/Security/SecurityCore.lean:xorStream_involutive` | proved (native), `SecurityCore.lean:16` | face unassigned (generic) | none | Round trip only; the source excludes full-stream secrecy. |
| `proof/Substrate/Algebra/Security/SecurityCore.lean:otp_changes_the_wire` | proved (native), `SecurityCore.lean:21` | face unassigned (generic) | none | One non-identity witness; source line 22 denies a masking theorem. |
| `proof/Substrate/Algebra/Security/SecurityCore.lean:passive-reader secrecy` | anchor, `SecurityCore.lean:24` | face unassigned (generic) | none | No observer, distribution or advantage in this file. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:cipher_secrecy_is_only_cost` | proved (riding), `SanguineSecurity.lean:89` | face unassigned (generic) | none | Fiber cardinality is proved; “only cost” remains a reading at line 95. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:confidentiality_is_entropy` | proved (riding), `SanguineSecurity.lean:72` | face unassigned (generic) | none | Rides the old OTP statement; no canonical secrecy theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:seed_survives_the_channel` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Ratio identity, not a canonical transport. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:channel_blind_to_one_flaw` | proved (native), `SanguineSecurity.lean:85` | hyperbolic `σ > 0` split | none | Existential non-injectivity only. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:channel_recovered_with_key` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Candidate basis: `ChannelsAnyRing.ringUpper_ringOfHalves` and `ringLower_ringOfHalves`; contract differs. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:key_decodes_interior` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Candidate basis as above; no exact target. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:generator_cipher_roundtrips` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | Losslessness only; the PRG reading is retired at line 111. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:keystream_secrecy_is_the_factor` | proved (riding), `SanguineSecurity.lean:72` | face unassigned (generic) | none | Group equation, no canonical secrecy theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:keystream_window_is_blind` | proved (riding), `SanguineSecurity.lean:72` | hyperbolic `σ > 0` split | none | Production scrambler remains open at lines 115-117. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:meaning_invariant_under_protocol` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | Naturality is not obfuscation; source marks that reading anchor at line 102. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:meaning_transports_across_protocols` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical protocol theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:rotation_round_trip_is_identity` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical protocol theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:integrity_decided_at_the_cell` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | Threat-model interpretation is anchor at line 106. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:dependability_any_survivor_reseeds` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical federation theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:dependability_seed_over_wire` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical wire theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:dependability_containment` | proved (riding), `SanguineSecurity.lean:72` | all faces | none | No canonical federation theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:security_is_the_substrate` | proved (native), `SanguineSecurity.lean:96` | unresolved composite: real split-cell and generic clauses | none | Must be decomposed; a one-face transport is rejected. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:obfuscation reading` | anchor, `SanguineSecurity.lean:102` | all faces | none | No hiding theorem. |
| `proof/Substrate/Algebra/Security/SanguineSecurity.lean:live SangNet wrapping` | open, `SanguineSecurity.lean:115` | hyperbolic `σ > 0` split | none | Executing correspondence absent. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:decryptRecovery` | proved (native), `EncryptionProjection.lean:69` | face unassigned (generic) | none | Generic injective-map construction; no trichotomy premise. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:fiberResidual_nonneg` | proved (native), `EncryptionProjection.lean:69` | all faces | none | Generic residual fact. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:decryption_is_unique_floor` | proved (riding), `EncryptionProjection.lean:73` | face unassigned (generic) | none | Generic injection; no canonical cipher theorem. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:injective_iff_decryptor_exists` | proved (riding), `EncryptionProjection.lean:73` | face unassigned (generic) | none | Mathlib re-export. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:collision_breaks_unique_floor` | proved (native), `EncryptionProjection.lean:76` | face unassigned (generic) | none | Generic many-to-one map; `Basic.nilpotent_of_disc_zero` is only an algebraic basis. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:collision_no_decryptor` | proved (native), `EncryptionProjection.lean:76` | face unassigned (generic) | none | No canonical hash theorem. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:cipher_fiber_is_a_singleton` | proved (native), `EncryptionProjection.lean:81` | face unassigned (generic) | none | Set-cardinality theorem, not entropy. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:collision_fiber_is_not_a_singleton` | proved (native), `EncryptionProjection.lean:81` | face unassigned (generic) | none | Set theorem, not entropy. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:cipher_pins_the_plaintext` | proved (riding), `EncryptionProjection.lean:87` | face unassigned (generic) | none | Injectivity instantiated. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:encryption_is_projection` | proved (native), `EncryptionProjection.lean:101` | face unassigned (generic) | none | Injective and colliding-map conjuncts carry no sigma; assigning either face needs an exact representation. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:AES/RSA placement and cost` | anchor, `EncryptionProjection.lean:94` | face unassigned (generic) | none | Named, not re-proven at lines 62-65. |
| `proof/Substrate/Algebra/Security/EncryptionProjection.lean:SHA placement` | anchor, `EncryptionProjection.lean:98` | face unassigned (generic) | none | Empirical placement, no reduction. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:cellAdjoin_injective` | proved (native), `DescentIsTheEasyInverse.lean:35` | elliptic `σ < 0` | none | Generic injection. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:descent_strips_back` | proved (native), `DescentIsTheEasyInverse.lean:35` | elliptic `σ < 0` | none | Left inverse only. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:descent_is_being_cell_cipher` | proved (riding), `DescentIsTheEasyInverse.lean:38` | elliptic `σ < 0` | none | Rides the old unique-floor theorem. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:adjoin_decryptor_is_descent` | proved (riding/native split), `DescentIsTheEasyInverse.lean:38` | elliptic `σ < 0` | none | Source explicitly splits its two conjuncts. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:cellCollapse_off_frame_one_way` | proved (riding), `DescentIsTheEasyInverse.lean:43` | parabolic `σ = 0` collapse | none | Two residual zeros, not no-decryptor. |
| `proof/Substrate/Algebra/Security/DescentIsTheEasyInverse.lean:autotropism and no-search readings` | open, `DescentIsTheEasyInverse.lean:47` | unresolved | none | Statements mention neither frame/sign nor cost. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:masked_key_lt_size` | proved (native), `MandalaMemory.lean:33` | face unassigned (generic) | none | Arithmetic masking; no sigma occurs. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:access_in_region` | proved (native), `MandalaMemory.lean:33` | face unassigned (generic) | none | Closest canonical result is `Apeiron/Capability/Naming.lean:nameMap_lt`; statements differ. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:no_key_escapes` | proved (native), `MandalaMemory.lean:33` | face unassigned (generic) | none | Canonical `Capability.MaskedTable.guardedRead_ungranted` has a different contract. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:nested_access_in_parent` | proved (native), `MandalaMemory.lean:33` | face unassigned (generic) | none | Closest canonical result is `Apeiron/Region/Graded.lean:graded_access_in_region`; statements differ. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:no_overflow_into_neighbor` | proved (native), `MandalaMemory.lean:33` | face unassigned (generic) | none | Closest canonical result is `Apeiron/Region/Graded.lean:graded_no_overflow_into_neighbor`; statements differ. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:load_address_bounded` | proved (native), `MandalaMemory.lean:38` | face unassigned (generic) | none | Restates the masked access theorem. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:masked_load_never_oob` | anchor, `MandalaMemory.lean:42` | face unassigned (generic) | none | Emitter coverage is not in the statement. |
| `proof/Substrate/Algebra/Security/MandalaMemory.lean:mandala_is_closed` | proved (native), `MandalaMemory.lean:56` | face unassigned (generic) | none | Region layout remains an implementation bridge. |
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
| `proof/Substrate/Algebra/Security/FoldDepth.lean:fold_is_lossy` | proved (native), `FoldDepth.lean:85` | face unassigned (generic) | none | Pigeonhole only. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:mulUnit_bijective` | proved (riding), `FoldDepth.lean:85` | face unassigned (generic) | none | Canonical `Table.WordIdentity.odd_mul_zmod_injective` is related but not the same theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:windows_from_one_product` | proved (riding), `FoldDepth.lean:85` | all faces | none | Old RadixDescent result. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:snd_balanced` | proved (native), `FoldDepth.lean:89` | face unassigned (generic) | none | Counted-fiber theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:exactlyToOne_comp_equiv` | proved (native), `FoldDepth.lean:89` | all faces | none | Generic transport through a bijection. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:xorHalves_balanced` | proved (native), `FoldDepth.lean:89` | face unassigned (generic) | none | Counted-fiber theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:fold128_balanced` | next, `FoldDepth.lean:113` | face unassigned (generic) | none | Not connected to the golden multiply. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:pack_injective` | next, `FoldDepth.lean:120` | face unassigned (generic) | none | Not composed with multiply/coordinate. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:depth theorem set` | anchor, `FoldDepth.lean:93` | all faces | none | Arithmetic over named naturals, not a fold cost theorem. |
| `proof/Substrate/Algebra/Security/FoldDepth.lean:closed-form-collision-absence claims` | next, `FoldDepth.lean:106` | all faces | none | Concrete witnesses do not establish a universal absence. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:accum_append` | proved (native), `KeyResolution.lean:43` | all faces | none | No exact canonical theorem. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:resolution_breaks_key_independence` | proved (native), `KeyResolution.lean:46` | face unassigned (generic) | none | Concrete counterexample. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:resolution_preserves_gauge` | proved (riding), `KeyResolution.lean:49` | all faces | `Apeiron/Table/Basic.lean:build_gauge_invariance` | Target is the premise it instantiates, not the relabel theorem itself. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:resolution_max_converges` | proved (riding), `KeyResolution.lean:49` | all faces | none | No exact target. |
| `proof/Substrate/Algebra/Security/KeyResolution.lean:the_resolution_law` | grade unconfirmed; open, declaration at `KeyResolution.lean:161` | split obligation: all faces + parabolic collapse | none | Headline is not listed in its warrant bullets. |
| `proof/Substrate/Algebra/Security/IncrementalResolution.lean:incremental_resolution_breaks_the_split` | proved (native), `IncrementalResolution.lean:43` | face unassigned (generic) | none | Concrete counterexample. |
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
| `proof/Substrate/Algebra/Security/AggregationBound.lean:flood_is_quadratic` | proved (Mathlib), `AggregationBound.lean:77` | face unassigned (generic) | none | Cost function is defined, not linked to a table. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:flood_dominates_bounded` | proved (Mathlib), `AggregationBound.lean:77` | face unassigned (generic) | none | Arithmetic comparison. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:exists_collision_at_step` | proved (native), `AggregationBound.lean:81` | face unassigned (generic) | none | Refutes the old separation inference. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:separation_does_not_bound_probe` | proved (native), `AggregationBound.lean:81` | face unassigned (generic) | none | Refutation result. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:descent_can_be_wholly_blocked` | proved (native), `AggregationBound.lean:91` | face unassigned (generic) | none | Worst-case blocking witness. |
| `proof/Substrate/Algebra/Security/AggregationBound.lean:shipped_probe_can_be_wholly_blocked` | proved (native), `AggregationBound.lean:91` | face unassigned (generic) | none | Model result, not current Apeiron source. |
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
| `proof/Closures/Applied/WebCipher.lean:computational_cipher_pins_the_plaintext` | proved (riding), `WebCipher.lean:79` | face unassigned (generic) | none | Carries only injectivity equation. |
| `proof/Closures/Applied/WebCipher.lean:web_boundary_is_a_blur` | proved (riding), `WebCipher.lean:79` | hyperbolic `σ > 0` split | none | Physical-pad interpretation is anchor at line 96. |
| `proof/Closures/Applied/WebCipher.lean:web_edge_is_the_pad` | proved (riding), `WebCipher.lean:79` | hyperbolic `σ > 0` split | none | Shared-random-bit interpretation is anchor at line 96. |
| `proof/Closures/Applied/WebCipher.lean:web_edges_are_channels` | proved (riding), `WebCipher.lean:79` | hyperbolic `σ > 0` split | none | Dimension count only. |
| `proof/Closures/Applied/WebCipher.lean:web_otp_perfect_secrecy` | proved (riding), `WebCipher.lean:79` | face unassigned (generic) | none | No canonical secrecy theorem. |
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
| `theory/MessageScheduleTopology.lean:shr_linear` | proved (native), `MessageScheduleTopology.lean:46` | face unassigned (generic) | none | Linearity does not prove loss. |
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

### Bounded tasks and barriers

The current documentary task uses exactly the three files at the top of this plan. It freezes
source statements and selects a finite observer fixture; it does not ratify canonical security
transport. Every future implementation row has its own isolated worktree, owner, base identity,
one focused gate and disposable rejecting control before dispatch. Lean caches are serialized.

| Order / task / owner | Owned files and completion | One focused gate | Rejecting control / grade |
|---|---|---|---|
| 1 / `APEIRON-SECURITY-W1-T1` / Sol | This plan, its intent and the new spec. Resolve the incompatible interpretations without changing old theorem grades. | Complete-file/source-attribution documentary gate below, including `git diff --check`; no compiler lane. | Feed the checker two false conclusions: compatible-key existence implies secrecy, and all generic projections have the `σ=0` face. Both must reject. Grade `anchor` for this contract. |
| 2 / `APEIRON-SECURITY-W2-T1` / Sol after this gate | New `proof/corpus/Security/InterpretationWitnesses.lean`, `proof/corpus/lakefile.lean`, `proof/corpus/AxiomAudit.lean`. Kernel-check `ZMod 2` uniform observer equality, biased `3/4,1/4` counterexample with unique compatible keys retained, fixed-key/public-map distinction, and short-key passthrough. The source-level theorem names used are stated in the spec. | Proposed bounded `lake build Security.InterpretationWitnesses AxiomAudit` from `proof/corpus`; register the new declarations in the audit. This gate has not run. | In a disposable copy set both key masses to `1/2`; the named biased distinguishing witness must fail. Grade `model` even if proved by Lean: no deployed cipher connection. |
| 3 / `APEIRON-SECURITY-W3-T1` / canonical corpus owner with Sol, conditional | At most one new canonical module plus that repository's audit and root registration (≤3 files, selected only after reading its local `AGENTS.md`). State an exact old `NCell` to canonical `σ=1` multiplication relation, then prove the split pair; retain the generic theorem's root/inverse/half premises if used. | Canonical package's focused bounded build/audit runner: **MISSING** until its owner selects the actual module and local gate. | Change `σ=1` to `σ=0` in a disposable copy; a split-pair theorem must reject. No canonical edit or claim from this plan. |
| 4 / later semantic-claim tasks / assigned after 2–3 | One statement/observer and ≤3 named files per task. Security, SHA, AES/RSA, federation, Rust extraction and empirical standard gates are distinct. | **MISSING** until each selected contract and source are fixed. | A mutation that targets that statement; no inherited control from the finite fixture. |

Execution status 2 / `APEIRON-SECURITY-W2-T1`: `proof/corpus/Security/InterpretationWitnesses.lean`
kernel-checks the eight source-named witnesses; the module gate `lake build
Security.InterpretationWitnesses` exits 0 (2966 jobs). The disposable `1/2`-mass mutation rejects at
`biased_observer_distinguishes` (`unsolved goals`), restored green. The eight declarations are
registered in `AxiomAudit.lean`; the integrated `lake build AxiomAudit` run and the lead's audit
review are pending. Grade `model`: no deployed-cipher, entropy or protocol claim.

The first executable proof task is order 2. It proves a conditional uniform-pad observation
equation and a biased countermodel, not an unconditional OTP or standard-security theorem.
Canonical transport and source-connected Rust work remain blocked on their exact contract,
owner, substrate and local gate. This review is authorized now; storage closure is not a reason
to leave the attribution error in place.

#### Documentary gate and negative fixtures

Run from this worktree. This is one Python command; it checks complete owned files, whitespace,
source-pinned theorem strings and two deliberately false inferences. It is a documentary control,
not a Lean or crypto gate. The fixture classifier is narrow by design: it rejects exactly the
unsupported conclusions named here, and a later semantic proof must have its own mutation control.

```sh
python3 - <<'PY'
from pathlib import Path
import subprocess
paths = [Path('docs/epics/security-algebra-rebase.md'),
         Path('docs/epics/security-algebra-rebase.md'),
         Path('docs/epics/security-algebra-rebase.md')]
for p in paths:
    b = p.read_bytes()
    assert b.endswith(b'\n') and not b.endswith(b'\n\n\n'), p
    assert all(line == line.rstrip(b' \t') for line in b.splitlines()), p
text = '\n'.join(p.read_text() for p in paths)
assert 'e830d3fbdeb40319d1be0d1fc803e706748b6a3c' in text
assert '3d66ea3aa2fe1de5997b6b757247454a2f771358' in text
assert 'unique compatible key' in text and 'biased' in text
assert 'face unassigned' in text and 'nilpotent_of_disc_zero' in text
old = subprocess.check_output(['git', '-C', '../sanguine', 'show',
    'e830d3fbdeb40319d1be0d1fc803e706748b6a3c:proof/Substrate/Algebra/Trinity/RecoveryTrinity.lean'], text=True)
canonical = subprocess.check_output(['git', '-C', '../sanguine-apeiron', 'show',
    '3d66ea3aa2fe1de5997b6b757247454a2f771358:proof/Apeiron/Trichotomy/Basic.lean'], text=True)
assert 'theorem otp_perfect_secrecy (c m : G) : ∃! k : G, m + k = c' in old
assert 'theorem nilpotent_of_disc_zero' in canonical
assert 'theorem zeroDivisorPair_of_disc_one' in canonical
invalid = ('unique compatible key therefore secrecy', 'every projection is parabolic')
def documentary_accepts(s):
    return not any(x in s for x in invalid)
prose = '\n'.join(part for i, part in enumerate(text.split('```')) if i % 2 == 0)
assert documentary_accepts(prose)
assert all(not documentary_accepts(prose + '\n' + x) for x in invalid)
assert documentary_accepts('unique compatible key does not specify a distribution')
subprocess.run(['git', 'diff', '--check', '--', *(str(p) for p in paths)], check=True)
PY
```

### Deeper review — the strong claims (spikes, each ends in evidence or a blocker)

| Spike | Question | Source |
|---|---|---|
| `SPIKE-sha256d` | Is "SHA-256d is technically EASIER to break than SHA-256" proved, anchor or conjecture, and does the three-depth reading survive rebasing? | `theory/THE_PASS_THE_PARCEL_CHAIN.md`; `NullCell/TensionPinch.lean` (labels it conjecture) |
| `SPIKE-message-schedule` | What exactly is proved about localising SHA-256 preimage resistance to the message schedule, and what is its falsifier? | `theory/MessageScheduleTopology.lean`; `THE_HASHMAP_PATH.md` (45→52 steps frontier) |
| `SPIKE-aes-rsa-bridge` | The exact statement of the AES/RSA bulk-collapse bridge, to be proved or explicitly left open. | `Security/EncryptionProjection.lean:62` ("named, not re-proven") |
| `SPIKE-flaw-trinity` | Does the attack-as-unintended-quotient reading (linear≈split, differential≈nilpotent, Bellcore≈zero-divisor) survive scrutiny as a method, and what does it actually locate? | `THE_FLAW_TRINITY.md`; attack table in `docs/epics/security-algebra-rebase.md` |
| `SPIKE-federation-security` | Rebase and grade the distributed-security core: `honesty_decided_at_the_cell`, the leaderless floors, partition/duplicate semantics. | `Navigation/Federation.lean`; `docs/direction.md` "Federation / distributed" |

Each spike is time-boxed, ends in evidence (a named theorem, a grade, or a blocker), and does not by
itself close the epic or ship a security claim.

### Ownership, order and stop conditions

- Each named task's gate and control pass before dependent work begins. The finite model can be
  checked in Apeiron without editing the canonical corpus. A canonical transport task belongs
  to its owner and uses its own local lifecycle and audit.
- Old source grades stay as source grades. No new theorem inherits `proved` by name or by a
  nearby algebraic witness. The closed cone and real-source extraction gates apply when such
  claims are eventually attempted.
- Stop on any statement that cannot be preserved, on an unsupported canonical target, or on a
  hardness claim with no named theorem; return the exact obligation.

### Acceptance

The documentary task closes on its gate and rejecting fixtures. The finite interpretation task
requires its own focused Lean gate and mutation. The epic remains open until exact canonical
transport, observer/security theorems, real-source implementation evidence, standards and
protocol assumptions, and applicable system gates close on the same identities. No security
claim, hardness result or standard break follows from this plan's acceptance.
