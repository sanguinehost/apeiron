# Apeiron Programme Direction

> **Correction (repository state).** The REG-1 comparison harness (`examples/reg1`, `examples/smhasher`, `examples/fold`, `examples/support`) and its comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash) were removed from the tree in `c9a2d0c`. Every `cargo run ... --example reg1` (and any other removed-harness example) below is a historical command that cannot run on this tree; the live Rust gates are `cargo test`/`cargo clippy`/`cargo fmt` from `AGENTS.md` §6, and the live performance entry point is whatever the current harness names. Registered-incumbent and acceptance-registry claims below describe the removed harness.

Kani is retired from development and acceptance by user direction;
[the retirement contract](epics/kani-retirement.md) supersedes earlier Kani requirements.
Rust tests, source mutation controls, extraction and Lean refinement retain their distinct scopes.
No universal read, representation or allocator obligation closes through this policy change.

This file is the sole programme authority. It states the shared thesis, ownership, outcomes,
primitive correspondence, migration and execution order. Local documents own local representations,
trust and execution detail and link here: [local boundary](kernel-boundary.md) and
[operator handoff](plan.md). The central programme coordinates named cross-repo tasks; it
does not dispatch an entire repository or replace its local gates. Routine work proceeds within
the repository's scope; changing a programme boundary requires an explicit owner decision.

## Programme destination

The roadmap is organised toward a system that **supersedes monolithic Linux**, built from the
unifying `R[x]/(x²−σ)` foundation. These are forward-looking plans (design/`anchor` status), not
present-tense implementation claims. Evidence grades below describe existing artifacts, not the
aspiration or the importance of a milestone.

| First-class programme thread | Destination and ownership | Engineering route |
|---|---|---|
| Physics and prediction; the hyperbolic face (`σ>0`) | Develop the descent/ascent programme in [Paper I](../../sanguine/release/paper1/INDEX.md) and test the [registered physical predictions](../../sanguine/theory/PREDICTIONS_REGISTRATION.md) against their own falsifiers. The hyperbolic face is the real split specialization, not a replacement for the full σ-family. | Canonical algebra, explicit physical interpretation, numerical refinement and scientific conformance; empirical prediction acceptance stays independent of software/game acceptance. |
| **Whispering Tides** | A playable, scientifically constrained generated world, owned by its [application roadmap](../../whispering-tides/ROADMAP.md). | Song A5 services and A6's pinned planet-generation/interactive slice; full game/science scope remains first-class programme work beyond that release slice. |
| Video games | Playable systems and reusable engine machinery; Penglai owns game policy and host integration, Lyrium later extracts engine machinery. | Checked readouts, rendering/input, assets and bounded simulation through application-owned gates; no game policy in the canonical core. |
| World simulation / universe generation | Generate and evolve worlds from the common basis, from planetary terrain/fluids to wider physical models. | Song research C4–C6/G7 and application conformance, with seeded state, units, tolerances and independent references; not merely a graphics benchmark. |
| Verified execution and OS capabilities | Secure OS-language-proof composition with independent Song execution, universal translation/lifting, structural resonance and direct hardware projection. | Retain A0–A6/B6 and Apeiron G1/REG-1; post-A4 [Song A7](../../song/docs/ROADMAP.md#a7--universal-algorithm-bridge-and-projection) owns imported-program lift/adapt/resonate/cancel/project. |

These threads inform the engineering interfaces; their priority does not move application or
protocol policy into the protected core. Scientific work and bounded interface probes continue
alongside foundational verification. The next large AI implementation retains its readiness gate.

## Purpose

Apeiron is one R0 mathematical/computational foundation: canonical Lean and reusable Rust in one
target repository, with Song/Slang as independent execution sharing the mathematics.
Hashtrinity/fold is its historical implementation; the two-slot spinor and affine recursion collapse
are its semantic law; the Table ordered fold is its state/fold form; SKI/derived-Iota supplies its
normal form/translation; Rust and Song/Slang are concrete realizations of that primitive.
These are layers/views of one primitive, not separate foundations or programme tracks. This is the
architecture requirement, not a claim that the complete connecting theorem already exists.

Apeiron evolves from hashtrinity; `fold/hashtrinity` is the predecessor. The basis is
`R[x]/(x^2 - sigma)`, `Cell<R>` with explicit `Sigma`, and face-correct `Coord` addressing.
The selected map and count arms execute the ported hashtrinity table (`src/hashtrinity/`,
rev `4cc14471`, reached as `apeiron::freq`); that body is the refinement target, rebased onto this
basis (`docs/plan.md`). The generic growable [src/table.rs](../src/table.rs) — a flat
certified-coordinate resident area plus grouped spill regions, keyed by `(Sigma,K)`, with
`Stored`/`Regen` carriers — is removed once the executed body is verified (`APEIRON-RETIRE-GENERIC`).
`SizeOverflow` and `AllocationFailed` are resource outcomes; `ShardFull` is compatibility-only.
Implementation presence does not close the extraction/refinement integration wave. The earlier
fixed-shard layout and its bounded-admission proofs are historical, not proofs of this layout.
Every executing face must refine the
canonical object; anything not rooted in it is not accepted. The canonical object is
`Apeiron.Table.{HMap,get,insertWith,build}` and
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` with the extracted `Cell<i64>` word
bodies (`word_mul_denote`, `word_norm_denote`).

Name the algebra and its faces by their standard mathematical names and notation: `R[x]/(x²−σ)`;
elliptic (`σ<0`), parabolic (`σ=0`), hyperbolic (`σ>0`); the norm form, idempotents and zero divisors
keep their usual names. Cite the originator(s) of a borrowed object before its reference, and
retrieve the reference before calling it evidence. Legacy coinages are not reused for new objects.

Fold/hashtrinity -> Apeiron consolidation must be **performance-preserving**: it carries forward
the high-performance implementation's useful tested surfaces, not merely its names or formulas.
The selected production primitive must both refine canonical `Apeiron.Table` / `AnyRing` and meet
the performance restoration gate below. Extraction convenience cannot justify replacing a
materially faster implementation with a slower structure. The central proof obligation is the
canonical denotation of the performance-bearing executing implementation; empirical confidence of
the SwissTable/hashbrown kind alone is insufficient. Proof and performance are joint G1 acceptance:
neither proof without performance nor speed without canonical refinement passes.

The earlier O(1)-at-fixed-N qualifier belongs to fixed-capacity primitives with bounded callbacks,
not to the growable table: a fixed probe-group width does not bound the number of groups searched.
Reading a whole source remains at least linear in its bytes; no constant-time whole-source checking
or constant-size proof term follows. Resource accounting remains implementation-specific; this qualification
does not replace the required collapse/fold correspondence or the G1/REG-1 performance gate below,
including restoration of historical performance.

## Owner definition of done for the apeiron core

The owner sets one bar for the apeiron core. A face is done only when all four hold jointly:

1. **Formally verified end to end.** Real-source extraction; refinement to the canonical
   `Apeiron.Table.{HMap,get,insertWith,build}` and
   `Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` objects; the closed axiom cone
   `{propext, Classical.choice, Quot.sound}` for the Rust/Aeneas bridge, with an **empty** cone for
   the Slang source-refinement chain; and mutation controls on the real source or generated kernel
   that turn the named proof or test red and pass restored.
2. **Faster than `hashbrown`+FxHasher and `hashbrown`+XXH3** on every workload column of the map
   face: `apeiron / incumbent <= 1.0`, mean of ≥5 paired runs with spread; `<= 0.5` is aspirational
   (owner, 2026-10-06; [plan](plan.md#outcome)). The all-incumbent G1/REG-1 contract below is the
   historical record, not the bar.
3. **Not HashDOS-vulnerable.** Adversarial key populations must not collapse the structure to
   quadratic behaviour; collision resistance is a required property, not an optional face.
4. **Cryptographic strength not mandatory but achievable as a face.** A cryptographic-strength hash
   is not required for admission and is a realizable face with its own cost and gate.

Proof without performance, performance without canonical refinement, or verified speed that an
adversary can collapse does not pass.

## Thesis

The basis is `R[x]/(x^2 - sigma)`: one two-slot algebra, with elliptic, parabolic and hyperbolic
regimes over the reals. The chosen scalar and each theorem's hypotheses remain explicit. The goal
is to construct computation, state, transformations and their verification from this basis and use
them in literal machines. The scientific thesis, "physics == computation", motivates the physical
models; it is not a claim that every physical interpretation is already proved.

Machine words, cells, combinator terms and proof terms have distinct representations and roles.
Each executing implementation requires its own refinement to the relevant canonical definition.
Song's independent implementation is intentional; an unproved substitute presented as a refinement
is not. Canonical mathematical content stays in Lean alongside Apeiron's Rust implementation.

The primary system deliverable is a secure kernel with a coupled language and proof foundation:
memory and authority safety follow from the admitted representations and checked transitions, not
merely from an optional checker applied to otherwise unrelated code. This is the system to build,
not a claim that the present components already establish its security.

Each refinement is directional and explicit. A representation theorem links one concrete layout to
the canonical object; a second refinement links another layout to the same object. The canonical
composition theorems are then reused by both, rather than re-derived per layout. Whether that reuse
reduces total effort is a measurable programme goal, not an assumption.

A table state denotes a function; an update denotes a function update; a carrier step denotes a ring
operation. Those denotations are stated per implementation against the canonical definitions, and
the representation and admission relation of each implementation is proved separately.

The programme goal is a single R0 chain through the same canonical collapse/fold object:

```text
lift (source / LLVM)
  -> canonical collapse / ordered fold (two-slot algebra and spinor law)
  -> SKI / derived-Iota normal form / translation and checked reduction / subsumption
  -> actual realization (Rust and independent Song / Slang execution)
  -> safe and useful programmes
```

Local algebra, matrix, Cayley, online-chart, table and evaluator proofs close only their named
layer until a checked theorem connects the complete hashtrinity/fold/spinor/SKI/Slang chain.
Historical recurrence-to-constant-loop and evaluator-speed anchors remain anchors, not proofs or
measurements. An admitted normal-form/translation outcome does not assert that every SKI term
normalizes or that every implementation runs in constant time.

Terms are used precisely:

- **Entry encoding vs semantic lift.** Translating a source or compiler artifact into a proof
  assistant is an entry encoding. A semantic lift additionally relates the artifact's actual
  behaviour to a canonical definition on a matched domain. Extraction alone is an entry encoding.
- **Syntax, convertibility, route and observable equalities.** Syntactic equality, convertibility,
  equality of a reduction route and equality of observable results are different relations. A
  verdict that decides one does not silently decide another.
- **Lossless reconstruction vs observable relative subsumption.** Reconstructing the original
  program exactly is stronger than showing a transformed program is observationally equivalent on a
  stated set of observations. The weaker relation is admissible only when its observation set,
  refusal and resource behaviour are stated.
- **Code, data and runtime cost vs actual computer work.** Bytes of source, bytes moved and
  wall-clock time are different quantities. A cost claim names the metric, the machine and the
  interval.

## Lineage: the fold crossover

The fold crosses from hashtrinity into apeiron, and that crossover is the lineage.
`HashTrinity/Core.lean` (`insertWith`, `build`, `get_build_acc`) and `Song/Table.lean` (`put`,
`run`, `merge_is_state_fold`) are the same total map, new-first update and ordered build fold as
canonical `Apeiron.Table`; `hashtrinity/src/algebra.rs` supplies the predecessor
`Coord`/`CoordInv`/`Stored`/`Regen`, folded multiplication and the four merges. Apeiron changes that
lineage by separating exact packing from lossy coordinates, repairing the unguarded byte fold, and
using bounded home-shard storage with explicit refusal; canonical
`Table.Basic.build_gauge_invariance_mul`, `AnyRing.ringMul_comm` and `AnyRing.ringMul_assoc` close
the algebraic side. An audited cross-revision equivalence artifact is still missing, so a
byte-hash-equals-`ringMul` claim or an arbitrary-merge classification is not made.

## Ownership

- **Apeiron** owns the canonical definitions, the reusable Rust primitives and their refinements.
  Lean is retained: the canonical mathematics, its history and its attribution stay in the tree.
  Reusable Rust and refinements converge in Apeiron. The approved destination is Apeiron, not a
  sibling tree. Source preservation targets a canonical package that is independent of any single
  Aeneas extraction link.
- **Song** remains an independent, adjacent assembly-up implementation: its own assembly, Slang
  implementation, contracts and local build/runtime acceptance. Where Song is slower, it derives its
  approaches from Apeiron; Apeiron's proven core is the reference. There is no Rust or Aeneas
  runtime linkage between Apeiron and Song, and Lean is not deleted. Song's local gates stay owned
  by Song. The required cross-repo result is a proof of primitive correspondence, not runtime
  linkage.
- **The substrate is shared, and Song's data operations run on it.** Song MUST carry the canonical
  Apeiron table and MUST route its data operations through it. A private probe, lookup or merge in
  Song that the canonical table already supplies is a **rebasing obligation, not an alternative
  implementation**: the duplicate is the defect, the canonical object is the replacement. This is
  not a trade between hard-and-good and easy-and-bad — the shared substrate *is* the architecture,
  and a second implementation of it is not the cheap route. Song predates Apeiron, so the rebasing is
  real work with an order. Where Song's architecture cannot host the canonical table, **Song's
  architecture changes**; the requirement is not discharged by keeping the duplicate. Local gates
  stay owned by Song; convergence tasks are cross-repo tasks closed by Song's own gates.
- **Applications** retain local policy. Penglai owns fiction, materials' physical interpretation,
  gameplay policy and host integration; Lyrium extracts engine machinery later; Fold is not a
  consumer or requirements source; Whispering Tides owns game and scientific conformance, not
  network. Application Rust and game policy remain local.
- The central programme coordinates named cross-repo tasks. Local owners retain implementation
  detail and gates; they may not silently remove a programme outcome or change the shared boundary.
- The physical canonical corpus is still `../sanguine-apeiron/proof/Apeiron/`. Apeiron is the
  approved destination, but the sources have not moved and no pin changes in this pass. See
  [Migration](#migration).
- All AI-written code is written under user direction. Commit authors and optional trailers are not
  contribution measures.

Current owner roadmaps: [canonical corpus](../../sanguine-apeiron/theory/ROADMAP.md),
[Song](../../song/docs/ROADMAP.md), [Penglai](../../penglai/README.md),
[Whispering Tides](../../whispering-tides/ROADMAP.md).

## Programme outcomes

A programme outcome is required. It is not thereby privileged, immediate, or a prerequisite for
every local release. Each row names the owner assigned before dispatch, the consumer, the
dependencies, the evidence that exists today versus the open gate or missing profile, and the
closure condition. A missing gate or profile requires a bounded design task before software
implementation; a missing gate is named, never replaced by a fictional command.

The readiness profiles are the gate between the foundation and the next large implementation.
`docs/contracts/joint-primitives.md`, `joint-table.md` and `joint-evaluator.md` name the selected
primitives, domains and obligations, and their checkers establish specification consistency only.
Their table R1/R2, unmatched B1 algebra and fuel-evaluator profiles are layer-local evidence/spec
artifacts, not separate foundational authorities. The algebra/combinator, correspondence, lift,
reduction and realization outcomes below all depend on the same canonical collapse/fold object and
its checked connecting theorem. A fuel evaluator, table refinement or source binding alone cannot
close that dependency; the rows' local prerequisites do not replace it.
They do not establish joint readiness: the Rust–Song word/map transport is checked, while Song
source binding and the checked primitive composite remain open. Song A0 alone does not satisfy
readiness, and the existing Song milestones are retained, not replaced by it.

| Outcome | Owner | Consumer | Dependencies | Current evidence vs open gate | Closure condition |
|---|---|---|---|---|---|
| Retained Lean + reusable Rust source convergence | Apeiron lead | all programme work | none | The canonical source location remains `../sanguine-apeiron/proof`; the bridge manifest resolves Git `880aa57e47e782ed7a9d9ee1974cfc87360b446b`, not the older `88866412` pin. Migration into Apeiron remains open. | One revision holds canonical Lean and reusable Rust, both built and audited on one source/tool identity. |
| Literal algebra, combinator and evaluator foundation | corpus lead (Apeiron) | Song, Rust, physics | retained Lean | `Apeiron.Combinator.routes_are_a_combinatory_algebra` and `Apeiron.Combinator.denote_reduces` are checked theorems. `Apeiron.Combinator.Comb.normalize` is a fuel-bounded evaluator definition; `Apeiron.Combinator.Comb.verdict_decided_exact` proves exactness when the verdict is not `.fuel`. `Omega.lean` proves non-normalization for a named term; that is a boundary, not an unfinished theorem. Open: selected finite-machine representations and executing evaluator refinement. | The selected executing evaluator refines the canonical semantics on an explicit domain and reports resource exhaustion without a false verdict. |
| Independent Rust–Slang primitive correspondence | cross-repo lead (Apeiron + Song) | both executing implementations | retained Lean; combinator evaluator | Profiles exist, separate Rust and Song replacement refinements are checked, and `Joint.WordBridge` proves the `I64 ↔ Slang.Word` pointwise-map transport. Open: unconditional Song source binding, decoded-context instantiation and the combined theorem on a matched domain. | A checked joint primitive theorem relates both executing implementations to the same canonical definition on a matched domain, including representation, state/effects, admission/refusal and resources. |
| Lift / universal translation | Apeiron lead, with Song owning its source/target adapters | subsumption, realization | source and semantic profiles; Song A0–A4 before bulk ingestion | Plan: Universal Algorithm Bridge, `song lift` and `song adapt` (A7), with modality-blind `universalLiftCons` ingestion and SSA cons-list state vectors. Historical Lean authorities and their real scopes are recorded in the [transfer analysis](roadmap.md#translation-authorities-and-missing-links). Missing links: LLVM domain/effects/undefined behaviour, list→Church state, opcode interpretation, historical→canonical term transport and Song source binding. | Supported source interpretations and target realizations compose through canonical semantics, refusing unsupported inputs; imported-program claims require A7, not every A0–A6 release. |
| Checked reduction / subsumption | Apeiron lead; Song transformation owner | realization, transformation | lift; canonical term transport | Plan: structural resonance (`song resonate`) and harmonic cancellation (`song cancel`), with collision-confirmed structural sharing, checked hoist/rewrite certificates and explicit observations. `Apeiron.Combinator.Comb.hoist_preserves_semantics`, `Apeiron.Combinator.denote_reduces` and `Comb.verdict_decided_exact` cover their named term relations, not a source-level subsumption pipeline. | Lossless reconstruction or observable relative subsumption is checked, including effects, refusal and resources; distinguish source compression from runtime benefit. |
| CPU / Slang / GPU realization | Apeiron + Song leads | applications, measurement | subsumption | Named `Cell<i64>` source kernels are refined. The programme-wide joint refinement is not established; existing Slang contracts and model bridges, and GPU experiments, have their own scope. | Each realization refines the same supported semantics, with target features, reduction order and numerical tolerance specified. |
| Secure OS-language-proof-kernel composition | cross-repo lead (Apeiron semantics/refinement + Song execution) | literal-machine, games and world-simulation tracks | retained Lean; Song A0–A6/B6; primitive correspondence; lift/subsumption; realization | Plan: supersede monolithic Linux with the capability OS, coupled language/proof core, Universal Algorithm Bridge, resonance/cancellation and direct hardware/native ELF projection (`song project`, A7). Hosted ELF output is not native isolation. Open: end-to-end composition through source semantics, authority, boot/ABI and hardware. | A revision-pinned vertical system passes Song's local gates; composing refinements cover the admitted language, proof kernel, OS/ABI and executing primitives with explicit threat model, refusal and resources. A7 additionally gates imported-program claims, not every local release. |
| Measured transformation benefit and productivity | lead | programme | realization | Historical Fold numbers exist; no benchmark was rerun in this pass. | A named machine, metric and interval measure before/after transformation cost and developer productivity. |
| Active test-time learning and lower-cost useful AI | AI-track lead | applications | joint foundation readiness; realization | Historical formal and executable records exist: [paper1](../../sanguine/release/paper1/INDEX.md) and [spinor-ttl](../../spinor-ttl/README.md). Existing work is not new acceptance. Open: a joint learning/update profile and its executing correspondence. | Streaming updates and retained knowledge satisfy the selected contracts; adaptation quality, learning/inference cost and resource bounds meet a pinned comparison profile. The research target is order-of-magnitude cost improvement at matched useful capability, not an established result. Inference or quantization alone does not close learning. |
| Physics: classical/quantum correspondence and neutrino prediction | Apeiron physics lead | scientific and application tracks | algebra foundation | Conditional model theorems exist in `Apeiron/Physics/Madelung.lean`, `Apeiron/Physics/Flavour.lean` and `Apeiron/Cosmology/NeutrinoSum.lean`. Real-valued carrier conservation and scoped kinetic-energy results are proved under their stated hypotheses; ring-generic theorems establish conservation and norm-form-ledger identities, not kinetic energy or nonnegativity. The word ring carries no energy claim. Open: joint scientific conformance profiles separating model theorems from empirical confirmation. | Named model predictions are checked against their test profiles under stated domains and tolerances, independently of game integration. |
| Application and game integration | Penglai + Whispering Tides leads | players, scientific users | physics foundation; realization | Historical toy, game and scientific examples exist; this pass establishes neither current integration testing nor refinement. Open: a playable or scientific loop consuming checked model readouts. | A playable or scientific loop consumes checked model readouts with its own integration gate. |
| Federation / distributed | Song network lead or an assigned distributed-service lead | applications, including Whispering Tides if the profile calls for it | primitive correspondence | The historical Sanguine federation transfer lineage is recorded in the [song network roadmap](../../song/docs/network/ROADMAP.md) (C3/G5) as design input, and the old corpus's decentralised core (`Navigation/Federation.lean`: `federation_algebra`, `federation_defense`, `federation_reseeds_the_genome`, `conflict_floor`, `scheduler_floor`) is captured for rebase in [security-algebra-rebase](epics/security-algebra-rebase.md). Open: a joint current protocol profile; existing network experiments are historical. | Distributed peers exchange checked, revision-pinned state with explicit delivery, identity, partition and duplicate semantics. |
| Security / cryptography rebase | lead; Sol for grades; corpus lead (Apeiron) for transport | every executing face and the federation layer | retained Lean; canonical trichotomy | The old Sanguine security module (`../sanguine/proof/Substrate/Algebra/Security/`, 17 files) is largely proved pre-trichotomy: cost-based secrecy rejected (`cipher_secrecy_is_only_cost`), OTP-grade dropped-window secrecy (`sanguine_encryption_information_theoretic_security`, `otp_perfect_secrecy`), the cipher/hash fork (`EncryptionProjection`), and the systems-security results (`MandalaMemory`, `RaceFreedom`, `CapProvenance`). Captured, **not actioned**, in [intent](epics/security-algebra-rebase.md) and [plan](epics/security-algebra-rebase.md). Open: the rebase onto the trichotomy faces, the per-theorem grade transport, and the deeper-review spikes (SHA-256d reading, message-schedule localisation, the AES/RSA bridge named but not re-proven). | Every security theorem is rebound to a face at its declared grade in the canonical corpus, and the corpus's minimum-encryption standard is a claim with a confirm/refute pair; grades are frozen and neither direction is presupposed. The end state is an end-to-end proved Rust implementation per target standard (primitives in Apeiron's core, implementations in a secondary crypto repository), each answering the three questions: stated goals (formal + empirical), structural/algebraic weaknesses, and implementation weaknesses from the empirical hacks. |

The following G1/REG-1 requirements are unchanged. Its recorded ratios, exits and fixed-shard API
references are historical receipts, not measurements or API descriptions of the growable source
read in this review. Current acceptance requires rerunning its exact gate on the integrated identity.

## Required R0 acceptance: G1 / REG-1

The bar is the owner definition of done above and [the plan's outcome](plan.md#outcome). This
section is the REG-1 record; its harness (`examples/reg1`) was removed in `c9a2d0c`, and the section
is removed by `APEIRON-DOCS-CONSOLIDATE`.

<!-- APEIRON-STRATEGY-DECISION-T3-STATUS BEGIN -->
**Current performance status.** Fresh full-profile REG-1 on the current source (`a2866eb`,
dirty=yes; `cargo run --release --locked --example reg1 -- --check performance --profile full
--accept --controls`), 2026-10-03 on this box (`i9-14900KF`, `rustc 1.97.0-nightly`, single strict
accept run): exit 1, `registered=2592 load_bearing=2268 applicable=1450 pass=545 fail=905 missing=0
concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`. Three commits landed this
session, each a named mechanism: `a3104c9` (append a fresh locator residue run in one `Vec::resize`
instead of 53 checked appends), `a6e03a8` (`slots_for` floor `n -> 2n`, so a group-width region
keeps a vacancy and an absent read stops early), `a2866eb` (reserve the batch's locator extent before
the first merge; the per-insert 64-level directory walk is removed). Headline trajectory: `654c6f2`
453/997 -> `a6e03a8` 466/984 -> `a2866eb` 545/905. The dominant failing class remains the
**mutating insert paths**: `admitted-fresh-insert` worst
`occ1-direct-index-resident-admitted-fresh-insert-wrapping-add` **8.0898** (was 79.3958 at
`a6e03a8`; cells pass 0/32 -> 8/24), `growth-insert` worst **3.4173** (was 69.7023; 0/8 -> 2/6).
Other failing classes: `direct-index-resident` 0/40, `offset-beyond-direct` 1/39, `existing-update`
10/30; read classes pass at 0.25x–0.60x. The parallel spatial face (10 pairs, all failing
10.16x–67.29x at `a6e03a8`) is a separate axis, registered at only 2 cells. The earlier
[post-correctness receipt](evidence/perf-ledger.md) (`pass=421 fail=1029`; "worst serial pair spill
absent-miss 20.4682x"; "growth-insert remains about 1.5x") is SUPERSEDED — both headline claims fail
to hold on this source — and its absolute ns/op is not cross-receipt comparable. Ratios and the
pass/fail census are the evidence; intervals are within-run only. The extraction (`ApeironWord.lean`)
is NOT regenerated against `a2866eb`: `src/table.rs` and `src/table/locator.rs` drift until
`proof/regen.sh`, blocked on the patched translator (`AENEAS-TOOLCHAIN-BUILD`); the committed
extraction therefore still refines `654c6f2`, not this tip. G1 remains **OPEN**.
<!-- APEIRON-STRATEGY-DECISION-T3-STATUS END -->

RFC 0001's [G1/REG-1](rfcs/0001-core-architecture/RFC.md), lines 196-209,
is restored here as a REQUIRED current gate: `quality ∧ speed ∧ baseline`, not an optional
performance claim. The RFC package remains historical; this correction asserts no new RFC
publication or joint architecture acceptance. G1 and storage S8 remain OPEN until REG-1 runs and
every required row meets its stated acceptance criterion on one recorded source/tool identity.
The owner's performance-preserving requirement strengthens the current migration gate without
rewriting the historical RFC: recording a failing ratio as a boundary is evidence, not acceptance.

- Identity: zero mismatches over `10^8` adversarially coordinate-colliding keys against the
  reference map's key set. An identity failure is a different correctness contract, not a slowdown.
- Round-trip: `{0, 1, 2^63-1, 2^63, u64::MAX}` and packed-coordinate boundaries survive
  insert -> encode -> decode -> get as identical keys.
- Determinism: identical key/value results across separate processes.
- Growth/merge: get, insert, remove, growth and ordered versus commutative merges agree with the
  reference, including refusal. Missing operations in the current growable API must be reported
  as explicit contract boundaries, not silently omitted or counted as passes.
- Performance: match or beat **every** named applicable incumbent on every applicable cell,
  `selected/incumbent <= 1.0`, never behind and ideally ahead. The named incumbents are
  `hashtrinity::{Table,CoordMap,FreqMap}` in serial and packed-parallel forms
  (`count_par`/`get_each`), `hashbrown+foldhash`, `hashbrown+fxhash`, `hashbrown+rayon`, `std`
  SipHash, `foldhash`/`fxhash` par16, and — per the [hash comparison](hash-comparison.md) registry —
  `xxHash` XXH3/XXH64, `wyhash`, `ahash` and `fnv`. **Correction (c9a2d0c):** the REG-1 acceptance
  harness and these comparison dev-dependencies were removed from the tree; this incumbent list
  records the removed harness, not a registry a reader can run now. This supersedes the RFC elapsed-median floor
  `selected/hashtrinity <= 1.25` and replaces the historical-envelope question: there is no
  separate envelope tolerance and no family-wide exemption. An inconclusive or failed required cell
  keeps G1 open.
- Coverage: report the fraction of the admitted operation surface with Aeneas extraction, Lean refinement and Rust tests separately; the container
  stays outside reported coverage until evidence includes it.

The required comparison matrix carries the historical tested surfaces, not just one full16 case:

| Surface | Required matched comparison |
|---|---|
| Original spatial parallel workload | `71942db` CountTable and replacement versus the original packed hashtrinity baseline; preserve input, packing, partitioning, threads and timed region. |
| Serial operations and lookup cost | `3a95bb5` Map and replacement versus hashbrown/foldhash and std where historically applicable; report elapsed cost and instructions/lookup separately. Those historical Map results are not hashtrinity comparisons. |
| Executing fixed-shard surface | Occupancy 1/2/4/8/16 at fixed `N = 16`; direct-index-resident, offset, skewed and seeded-random coordinates; hashtrinity, hashbrown/foldhash, std and the other named incumbents where applicable. |
| Operation strata | Hit, miss, update, fresh insert and refusal timed separately; replacement and wrapping-add merges separate; retain growth/remove and ordered/commutative merge correctness requirements. |

An independent full-key correctness oracle must pass before timing; an injective timing population
does not replace the adversarial identity row. Register every matrix cell, baseline applicability,
admission and missing operation explicitly. No silent workload reduction or aggregate that hides a
failed required cell; full16/direct-index-resident is an extreme boundary, not a representative
aggregate. Historical numbers are hypotheses until rerun with durable harness/source identity.

### Global acceptance contract

For every applicable cell, `selected/incumbent <= 1.0` against every named incumbent; a cell is
never behind and ideally ahead. Applicability is registered per cell/incumbent pair before timing,
with a semantic or API reason; there is no family-wide exemption. Missing, duplicate, invalid or
inconclusive pairs reject. `cargo run --release --locked --example reg1 -- --check performance
--profile full --accept --controls` is the strict acceptance command; `--accept --profile smoke` is
diagnostic only and does not close G1. **Owner target scope (`APEIRON-REORIENT`):** the same
command under `REG1_TARGET_RATIO=half` applies `selected/incumbent <= 0.5` to the three hashtrinity
arms (`hashtrinity::Table`, `hashtrinity::CoordMap`, `hashtrinity::FreqMap`, serial and
packed-parallel) and `selected/incumbent <= 1.0` to every other named incumbent; neither "<= 0.5
everywhere" nor a default 1.0 acceptance describes this target. Full acceptance requires every
registered pair measured on one source/tool identity. `docs/hash-comparison.md` records the
comparison axes —
cryptographic versus non-cryptographic, throughput at small and large sizes, latency, collisions
and avalanche, HashDOS resistance, determinism, `no_std`, licensing and verification — and the
per-domain "good enough" reading for operating systems, languages, provers, games, servers and
aerospace.

Each claim requires matched before/after on the real build path: source/harness/input SHA-256,
dependency and tool identities, named command, machine, target/features, compiler, thread/cache
conditions, metric, timed interval, raw samples and uncertainty. Pre-register repetitions and the
uncertainty decision rule; inconclusive intervals do not pass. Use same-run relative comparisons,
not copied absolute milliseconds. Require rejecting oracle/digest, workload-identity and
slowed-path or selected-implementation-mismatch controls, then an unchanged restored baseline.
The [lifecycle annex](epics/storage-refinement.md#performance-preserving-consolidation-lifecycle-annex)
registers the history and bounded implementation sequence. The existing `examples/reg1/` harness
carries the strict all-row acceptance command
`cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`.
Strict mode exits nonzero when any applicable strata speed cell is `fail`/`rejected`/`missing`, or
when a `pass` cell's ratio interval max is above `1.0` (each applicable cell must satisfy
`selected/incumbent <= 1.0`); `not-applicable` speed cells and the separate resource rows are
excluded. The same run without `--accept` is diagnostic, exits 0 and reports the axis `boundary`. The
command currently exits 1. The fresh strict full-profile receipt
([`perf-remeasure-2026-09-19-post-spill.md`](evidence/perf-ledger.md), sha256
`946b5d40…`) gives HEADLINE `registered=2592 load_bearing=2268 applicable=1450 pass=387 fail=1063
missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`. The spill
absent-miss cells fell ~15–21x (for example `occ16-seeded-random-absent-miss-replacement`
373.85x/314.62 ns/op to 23.59x/19.68 ns/op) and growth-insert fell ~145–152x (231.88x/1653.21 ns/op
to 1.53x/13.31 ns/op); the worst cell moved to the spill present-hit path (~30.9x, ~24.7 ns/op).
`load_bearing` rose 1458 to 2268 because the five named hashers are now load-bearing, so raw
pass/fail counts are not directly comparable to the prior receipt. Every compared cell still fails
the `<= 1.0` interval contract. No benchmark is rerun in this documentation task, and no speed
recovery is claimed. Local proofs and perimeter checks do not pass REG-1.

## Evidence status

Claims are stated at the grade the artifact supports.

- **Historical fixed-shard Table.** The following statements concern the earlier extracted layout,
  not the executing growable flat/spill body. The generated table refinement covers arbitrary callback instances satisfying
  `CallbackLawsG`. For the concrete `I64` word instance the constructor establishes `PhysicalInv`
  at every `N`; `get` reads the represented total map; `insert_with` preserves `PhysicalInv` and
  canonical `insertWith` on success and returns the identical representative on `ShardFull`, with
  exact refusal at `N = 0`. For positive `N`, `BuildAdmissibleN` exactly characterizes build success
  versus `ShardFull`, and success establishes canonical ordered `ReadRep` and `PhysicalInv`
  (`table_build_physicalInv_of_buildAdmissibleN`); the zero-capacity build is characterized exactly by
  `wordBuildN_zero_of_empty` / `wordBuildN_zero_refuses_of_nonempty`. Generic admitted insertion and
  build preserve `PhysicalInvG`; zero capacity is exact, and distinct routed keys give an all-`N`
  sufficient admission theorem. The joint replacement-map profile remains the concrete `N = 16`
  instance.
- **Decider F2.** `capped_certificate_complete` proves an accepted capped certificate exists iff no
  assignment inside the box satisfies the rows. The uncapped fragment is now closed:
  `certificate_complete` proves `(∃ log : List CertOp, checkCert raw [] log = true) ↔ ¬ ∃ ρ, SatLines ρ raw`
  and `synthesizeCert_accepts` exhibits the log; `completeSat_eq_true_iff` / `completeSat_eq_false_iff`
  decide `SatRows` exactly; `residue_certificate_complete` gives the same iff for the
  `ResidueCertificate` language; and `checkCertStream_sound` / `checkCertRecords_sound` replay framed
  streams framewise. These are the canonical `Row`/`Line`/`CertOp`/`ResidueCertificate` fragments over
  `ℤ`; the Song executing reader refinement and the Slang source binding are not thereby established.
- **Combinator evaluator.** `Apeiron/Combinator/Verdict.lean` defines `reduceDeep`, `normalize` and
  `verdict`. `Apeiron.Combinator.Comb.reduces_normalize` proves that `normalize fuel c` is reachable
  from `c`; `Apeiron.Combinator.Comb.verdict_decided_exact` is exact when the verdict is not `.fuel`.
  Not every SKI term normalizes; `Combinator/Omega.lean` carries one with none. The
  denotation theorems `routes_are_a_combinatory_algebra` and `denote_reduces` exist and are audited.
- **Carrier and models.** Real-valued carrier conservation and scoped kinetic-energy results are
  proved under their stated hypotheses. Ring-generic theorems establish conservation and
  norm-form-ledger identities, not kinetic energy or nonnegativity; the word ring carries no energy
  claim. Madelung, Flavour and NeutrinoSum contain conditional model theorems, not an empirical
  confirmation of their physical interpretations. Historical game and scientific examples are not
  fresh integration acceptance.
- **Controls.** A mutation of the real source or generated body must turn the named proof or test
  red; restoring it must pass. A check that cannot fail has no weight.
- **Freshness.** A source-proof run log and a committed generated file are not fresh acceptance on
  their own. Fresh acceptance re-runs the named gate on the recorded source/tool identity.

## Primitive correspondence

Primitive correspondence is the required cross-repo result. It relates two executing
implementations — the Rust primitive in Apeiron and the Slang/assembly primitive in Song — to the
same canonical definition on a matched domain. The obligations are joint, not separate:

1. **Representation.** Each concrete layout denotes the canonical object, with the denotation
   stated and its domain explicit.
2. **State and effects.** Each implementation's state transition denotes the canonical transition;
   side effects and ordering are stated where they exist.
3. **Admission, success and refusal.** The profile states the matched semantic domain and each
   implementation's admission and resource bounds. On success the implementation preserves the
    canonical meaning. The profile specifies unsupported-input and resource-exhaustion behavior,
    including whether state or effects can change on refusal. Where nonmutation is part of the
    primitive contract, state preservation must be proved. A defined canonical value may still be
    refused when an implementation lacks capacity; the domain is not silently reduced and no
    canonical partiality is fabricated.
4. **Resources.** Cost and resource bounds are stated per implementation and are not silently
   identified across layouts.

Equal names, worked examples, citations or identical layouts are not that proof. Rust and Slang have
different physical layouts: Rust executes the growable flat/spill `Table`; Song's Slang/assembly
representation is its own. The historical fixed-shard proof does not discharge the new Rust layout.
Different layouts can correspond, but only under a matched admission relation and the four
obligations above. A differential test or a shared example
is supporting evidence, not a substitute for the correspondence theorem.

This correspondence is a required programme outcome and is not established today. See the
[local boundary](kernel-boundary.md) for the Rust-side representation and trust inventory.

Correspondence is directional and explicit. A refinement may prove that implementation A refines the
canonical definition; a second refinement proves that implementation B refines the same definition.
The correspondence between A and B is then the composite, and it is checked, not assumed from shared
naming or a shared test suite. This is why the canonical definition must be the shared object, and
why an independently written implementation needs a checked refinement rather than a claim of
equivalence. The artifacts remain distinct; their canonical semantic authority is shared.

## Migration

**Corpus lineage.** `../sanguine` is the original corpus and `../sanguine-apeiron` its successor;
**this repository is the canonical root**, with both as predecessors. A theorem is cited from the
canonical root; a predecessor is provenance at its recorded revision, not a second authority and not
a fallback.

**Primitive provenance — search the lineage first.** A primitive this tree lacks very often exists
already in a predecessor corpus in some form — most likely `../sanguine` or `../sanguine-apeiron`,
and sometimes solved in another sibling repo. That form is a port with a named target, not a
derivation from nothing, and the search is part of the spec: name the declaration, its revision and
digest, and what the port must add. When the primitive is absent everywhere, one gets made — and the
record says the search happened, because "we made it" is a claim about prior art as much as about
code. Milestone 7 is the worked example: the exact three-axis, six-side, `dt`-bearing law is proved at
`CarrierVolumeAnyRing.lean:472` and referenced nowhere in Penglai, so the missing work was a port.

The physical canonical source remains at `../sanguine-apeiron/proof/Apeiron/`. The bridge's
`proof/corpus/lake-manifest.json` resolves corpus Git
`880aa57e47e782ed7a9d9ee1974cfc87360b446b`, Mathlib
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. Apeiron is the approved destination; the sources have
not moved, and no extraction, package or pin change is authorized by this document. A migration is
a separately gated task with its own lifecycle artifacts.

Keep source paths and pins accurate: a path written in a document must resolve. Distinguish the
approved destination from the not-yet-moved source. A source-proof run log records what happened at
a revision; it is not fresh acceptance. The release identity is recorded in
[proof/README.md](../proof/README.md); the run narrative is in
[archive/autonomous-run-log.md](archive/autonomous-run-log.md).

Historical evidence stays at its source and is pointed to, not copied. The predecessor documentation
at Git `1403767e671c81737525702946a3eb1eae4f0b0b` is retained: the storage lifecycle
([intent](epics/storage-refinement.md), [spec](epics/storage-refinement.md),
[plan](epics/storage-refinement.md)), the [carrier refinement plan](epics/carrier-refinement.md) and
the [corpus release pin plan](epics/corpus-release-pin.md). The combined
[programme plan](roadmap.md) records this documentation pass.

## Revised representation boundary

The Astra verdict revises the selected representation from a single growable key-partitioned
structure to two coupled regions:

- A **table-level flat resident area**: shard-major `256 x L`, admission
  `direct_word = Some(w) ∧ lane < L`, checked sizing, a stated byte budget and a density test.
  Failures spill to the grouped regions rather than refuse.
- **Independently load-sized grouped spill regions** beside the flat area.

One representative per key is held under a composite `DirectCover`; extent growth must reclassify
newly eligible spills. Key-dropping uses the concrete compile-time carrier `Stored<K>` or `Regen`
under `KeyRep` and an exact codec law, not a sibling count table or a lossy `FreqMap`. Restoration also
includes the key-partitioned load-sized shard build, deterministic schedule, grouped probe and
key-dropping count/dedup; the proof, extraction, registration and measurement work follows on
the same source identity. The flat/spill structure is present in executing Rust; completion of
these extraction, refinement and measurement obligations is separate. No speed recovery is claimed.

## Execution order

The execution order is [the plan's ladder](plan.md#ladder). The subsections below are the record of
earlier orders, not dispatch; they are removed by `APEIRON-DOCS-CONSOLIDATE`.

<!-- APEIRON-REORIENT BEGIN -->
### Five-barrier execution sequence (`APEIRON-REORIENT`)

`APEIRON-REORIENT` supersedes the active tuning dispatch below for the performance leg, and the
Intent/Song development scheduling in the [execution plan](plan.md#subsumption-execution-sequence-apeiron-subsum).
It is architecture and scheduling at grade `anchor`: it establishes no performance, test or proof
result, and the plateau and velocity figures behind it are supplied to this document, not
re-produced here. All five lifecycle milestones keep **no due date**; G1 is "in flight — five planned
barriers, completion horizon unestablished". Five is the decomposition, not a forecast that five
attempts suffice, and a barrier counts only when its gate and its negative control pass. G1/REG-1
stays **OPEN**.

| barrier | work, hardest contribution first | gate that opens the next barrier |
|---|---|---|
| **1. Contract and trustworthy correctness** | ratify the counting representation and its proof/resource interfaces; independently reproduce and resolve the six failing digest cells | accepted Intent/Spec/Plan; an executable production-path gate with an attributable red/restored control; a corrected full-key oracle on every affected cell. No implementation before contract acceptance; no timing claim before oracle acceptance |
| **2. Serial counting representation** | implement the single-region production container and the real counting consumers; preserve exact identity, ordering, growth and transactional refusal | the contract-named Rust gate and its semantic controls green; source-bound repeated runs show improvement outside the unchanged-code spread. A miss triggers representation adjudication, not another speculative tuning campaign |
| **3. Matched parallel counting/readback** | implement the required `count_par`/`get_each` paths with matched workers, operations, ordering and ownership | correctness controls green; workload, concurrency and address-face matching established; repeated affected-pair runs reported without exemptions |
| **4. Source-linked refinement and residual discharge** | extract the selected production bodies; establish constructor, preservation, read, build and refusal links; finish the remaining generic-`Table` obligations, REVIEW4 R1 only where its own contract requires it | fresh `proof/regen.sh --check`; bounded builds in `proof/` and `proof/corpus/`; required headlines, audits, joint witnesses and semantic controls green. Named assumed premises do not constitute end-to-end discharge |
| **5. Joint acceptance** | run the unchanged full registered population under the owner target scope, with the identity and HashDOS requirements, the coverage census and the shared integration gates | strict full-profile acceptance exits 0 with `full_acceptance=true`; every required Rust and proof gate and control green on one source/tool identity. Only then does downstream development reopens |

**Counting-face supersession.** Two statements below are superseded **for the counting face only**:
the T3 clause "not an exemption or a second container", and the T1 non-goal "a new container … is a
non-goal of this performance leg". For that face a separate single-region counting representation is
**selected**; it refines the same canonical object `Apeiron.Table.HMap`, `Apeiron.Table.get`,
`Apeiron.Table.insertWith` and `Apeiron.Table.build`, and the generic growable `Table` remains. This
is a second representation of one canonical object, not a second semantic object and not a licence to
relax any obligation on either face. Every unrelated non-goal and obligation in those lines is
preserved: cryptographic-face implementation, GPU/ISA work, lossy identity and universal
O(1)/WCET claims remain non-goals of this performance leg; HashDOS resistance, the remove/growth
obligations, every named comparison, both axiom cones and the realization exclusions remain required;
changed flat geometry, cached summaries, first-empty termination or payload layout still require a
new Astra/Sol contract/refinement decision; and the payload-initialization removal rejected in T2
remains rejected. The contract is [the counting-container
contract](epics/storage-refinement.md#coordinate-keyed-counting-container-contract); the
representation, canonical denotation, extraction perimeter, resource/refusal semantics and trust
exclusions are [the counting-face
boundary](kernel-boundary.md#counting-face-representation-boundary-apeiron-reorient).

**Parking rule.** New Intent and Song development is parked behind joint G1 acceptance; it reopens
when `full_acceptance=true` and the required real-source end-to-end refinement both hold on the
accepted identity. Parked requirements are retained, not waived, and already-dispatched returns
integrate only under their existing gates and ownership.

**Owner target scope.** The harness applies `REG1_TARGET_RATIO=half` to the three hashtrinity arms —
`hashtrinity::Table`, `hashtrinity::CoordMap` and `hashtrinity::FreqMap`, serial and packed-parallel —
so `selected/incumbent <= 0.5` there, and `selected/incumbent <= 1.0` on every other named incumbent:

```sh
REG1_TARGET_RATIO=half cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

Neither "<= 0.5 everywhere" nor a default 1.0 acceptance describes this owner target.
<!-- APEIRON-REORIENT END -->

<!-- APEIRON-STRATEGY-DECISION-T3 BEGIN -->
### Serial read specialization and stable parallel scatter

`APEIRON-STRATEGY-DECISION-T3` supersedes T1/T2's current status and next dispatch, not their gates,
requirements or trust exclusions. **Keep the all-pair interval ceiling `selected/incumbent <= 1.0`.
Authorize size-only factoring, then measured read-path specialization; do not accept the current
performance or promise parity from local tuning.**

**Attainability verdict: unestablished, not structurally impossible.** No read scans 256 shard
descriptors: an admitted resident match/empty returns before spill access; a spill read selects one
region. Neither 256 shards nor asymptotic O(1) establishes a lower bound in ns/op or a relative-speed
bound against another implementation. The 4096 independent-query schedule measures amortized elapsed
cost, not a dependent-load latency floor. A categorical promise of 20x recovery or proof that parity
is impossible would exceed the evidence. This is a decision to test a specific mechanism, not to
weaken “never behind” or turn an inconclusive cell into a pass.

The selected mechanism removes insertion-only work from reads: a sound spill-summary negative needs
no first-vacancy search; a read-only positive path needs matching/coverage, not vacancy bookkeeping.
Inline the existing resident classification/apply path and compute the address/index once, preserving
`live`, full `(Sigma,K)` identity and spill fallback. The existing resident lane already bypasses
region summaries; adding that bypass again cannot explain or close its gap. Summary indexing also
recomputes a shift/count width on every lookup; optimize it only with extracted equivalence or a
separately ratified cached-state invariant. These remove named work but have **no established 20x or
3–5x speedup**. No special case for the benchmark's `RawStrataHome` type or key population is allowed.

After the bounded mechanism campaign, any still-red pair blocks acceptance. If metadata indirection
or summary-positive coverage dominates, the next architecture task must design a compact hot lookup
directory/control lane and a source-proved absence certificate within the same `Table`; cold 256-way
build ownership need not dictate serial lookup layout. Changed flat geometry, cached summaries,
first-empty termination or payload layout require a new Astra/Sol contract/refinement decision,
not an exemption or a second container. That clause is superseded **for the counting face only** by
the counting-face supersession in
[`APEIRON-REORIENT`](#five-barrier-execution-sequence-apeiron-reorient); for the generic key/value
face it stands unchanged, and the geometry/summaries/termination/payload requirements keep it there.
Smaller/lazier descriptors alone do not remove the selected
region's dependent loads. The payload-initialization removal rejected in T2 remains rejected.

**Spatial selection:** stable parallel scatter, retaining `Parts` validation and query-order
readback. The supplied build split (kernel 46–55%, partition+validate about 28%) is diagnostic,
not a fresh phase receipt. Even eliminating that whole 28% would yield only `1 / 0.72 ≈ 1.39x`
under an unchanged additive build model; scatter alone saves less and cannot establish spatial parity.
Removing `is_intact` requires replacing its production assurance with a source-linked constructor
argument, not merely calling a tested, unextracted validator unnecessary. Grouped readback already
exists; eliminating its copies/reducing locally is deferred behind an order/effect contract.

The [serialized owner plan](roadmap.md#serialized-table-owner-and-spatial-plan)
and [read/scatter boundary](kernel-boundary.md#read-specialization-and-scatter-boundary) govern these
tasks. The active Sol owner finishes on its frozen source before factoring changes generated paths.
W7 composition remains frozen under its unchanged reopening condition; regression re-proof continues.
All named incumbents, coverage gaps, HashDOS, source-linked proofs, controls, complete Rust/proof
barriers, both axiom cones and realization exclusions remain required. No runtime or proof gate is
run by this documentation decision.
Scheduling of that dispatch is superseded by [`APEIRON-REORIENT`](#five-barrier-execution-sequence-apeiron-reorient):
the mechanism selection above is historical; its invariants, gates, controls and trust statements
stand.
<!-- APEIRON-STRATEGY-DECISION-T3 END -->

<!-- APEIRON-STRATEGY-DECISION-T2 BEGIN -->
### Current cut: partition validation before payload-layout replacement

`APEIRON-STRATEGY-DECISION-T2` supersedes the performance priorities/status in the older checkpoints
below, not their requirements. **Authorize the partition-validation cost cut; reject removal of the
region payload initialization in this dispatch.** `Region::reserve` materializes every payload before
indexed writes; deleting that pass is not a redundant-write elimination on the present representation.
The [bounded dispatch](roadmap.md#partition-cut-and-frozen-bridge-maintenance) requires
intent/spec ratification before changing the fail-closed validator. A safe replacement payload layout
requires a separate Astra/Sol decision with initialization, drop, refusal and extraction obligations.

The last read strict receipt remains [post-spill](evidence/perf-ledger.md):
exit 1, `full_acceptance=false`, 387 passing and 1063 failing applicable pairs; all spatial pairs
fail, including concurrency/address-face rejection independent of speed. Its measured examples are
absent-miss 373.85x → 23.59x and growth-insert 231.88x → 1.53x. The supplied later fingerprint
diagnostic reports present-hit about 24.7 → 9.9 ns/op; it is not a new strict campaign or evidence of
parity. The supplied phase split selects partition/validation and the per-shard kernel for investigation;
its percentages have different build/probe denominators and are not additive. No runtime command runs
in this decision. Resident serial and spill present-hit remain priorities without dropping other red
cells. “No speed recovery” in older checkpoints means no accepted global restoration, not denial of
the scoped improvements recorded in the receipt. That receipt's identity is not the later source.

`Region.summary`, the shifted tag fingerprint, contiguous partition and phased parallel API require
fresh extraction and regression checking. Add the explicit open obligations
`SpillSummaryShape`, `SpillSummaryRecorded`, `SpillSummaryMissSound` and `SpillFingerprintExact`
to [W7's ledger](kernel-boundary.md#spill-filter-obligations-and-partition-cut). Sol owns their exact
bridge contracts and eventual source-linked proofs for both carriers; Flash owns bounded source tests
and extraction. They are temporary named premises at grade `anchor`, not axioms, established `RepR`
fields or replacements for preservation. Local arithmetic/generated-body repairs continue while
`W7-SearchCommitBoundary` and whole-state composition remain frozen. No existing theorem is silently
weakened to accommodate the new state. All premises must be discharged before end-to-end acceptance.

The all-pair `<= 1.0` interval ceiling, named incumbents, controls, HashDOS requirement, complete
Rust/proof barrier, both axiom cones and realization exclusions remain unchanged. G1/REG-1 stays
**OPEN**, `report_complete=false`; no performance improvement closes a proof or coverage obligation.
<!-- APEIRON-STRATEGY-DECISION-T2 END -->

### Current strategy decision: freeze composition, restore performance

`APEIRON-STRATEGY-DECISION-T1` changes work priority, **not joint acceptance**. Suspend repeated W7
`prepared_rebuild_refines_insertWith` attempts and new extraction undertaken solely to unblock them.
Name the pending search/commit perimeter `W7-SearchCommitBoundary`, with the explicit premises and
rejecting-test obligations in the [local boundary](kernel-boundary.md#w7-searchcommitboundary).
It is trusted/unproved, with only individually evidenced Rust tests labelled
`tested`; the boundary as a whole is not yet test-accepted. Existing audited core implications keep
their exact scopes. Whole-state `apply_insert` preservation and the pending composition remain open.
Tests, more assumptions and a green build of existing modules do not establish either theorem.

This scheduling decision supersedes the immediate W7 retry instructions below and in earlier
handoff/roadmap checkpoints; their theorem requirements, gates and controls remain required.
G1/REG-1 and the owner definition of done remain **OPEN**, `report_complete=false`. A tested/trusted
search/commit boundary is not an accepted replacement for real-Rust canonical refinement. The
allocator/executor exclusions do not permit moving sequential search or commit out of eventual
end-to-end acceptance. The next large AI implementation remains gated; bounded physics/prediction,
The hyperbolic-face (`σ>0`) physics workstream, Whispering Tides, game and world-simulation interface work retains its independent gates.

Reopen W7 only when the lead receives a Sol-ratified, source-pinned task for full `apply_insert`
`RepR` preservation (both carriers, joint found/fresh flat/spill witnesses, rejecting controls), an
acyclic composition target, and a finite discharge map for every boundary premise. Each map entry
must name its real body, exact statement, gate, owner and checked primitive/tool dependency; a
missing primitive needs a separately accepted feasibility result first. A new premise, assumed
postcondition or another unchanged retry is not progress through this condition. Performance parity
does not itself reopen or discharge W7. Once the measured production mechanism is selected, restoring
this proof route is mandatory before G1 closure; freezing it indefinitely leaves the epic incomplete.

**Performance workstream.** The fresh strict full-profile receipt
([`perf-remeasure-2026-09-19-post-spill.md`](evidence/perf-ledger.md), sha256
`946b5d40…`) supersedes the prior full-profile receipt. Its dispatch is superseded by
[`APEIRON-REORIENT`](#five-barrier-execution-sequence-apeiron-reorient): the immediate tuning order is
a closed experiment, not an active recovery task, and its requirements stay retained. The supplied 84-pass/566-fail run, serial
direct ratios 1.75–3.94 versus hashtrinity/hashbrown (about 0.50 versus std), and spatial ratios 2–25
are stale diagnostics, not measurements of the selected identity here. Earlier receipts below remain
historical. Comparison coverage is repaired: the five named hashers are load-bearing. Next reduce
serial direct-path overhead while preparing the spatial partition/build/probe recovery on disjoint
paths. The [decision matrix and tasks](roadmap.md#performance-re-anchor-and-bounded-dispatch)
choose mechanisms from fresh per-operation evidence, not the old ratios alone.

`examples/reg1/acceptance.rs` and the [hash comparison](hash-comparison.md) registry now classify
XXH3/XXH64, wyhash, ahash and FNV as load-bearing on the opaque-key exact map face; each is the same
`hashbrown::HashMap` surface with a different `BuildHasher`, so each is subject to
`selected/incumbent <= 1.0`. **Correction (c9a2d0c):** `examples/reg1/acceptance.rs` and the
comparison dev-dependencies were removed from the tree; this classification belongs to the removed
harness. `blake3` and `sha256` remain informational cryptographic comparators
with no named authentication/integrity contract. Missing adapters or semantic/API applicability
decisions still block complete acceptance. Independently check coverage of hashtrinity `Table`,
`CoordMap`, `FreqMap` and the historical serial/spatial surfaces. Do not turn mismatched concurrency
or a different timed operation into a speed pass.

Keep the exact full-profile strict command and every-pair interval ceiling `<= 1.0` above. Additional
stretch target (`anchor`, not another acceptance floor): interval maximum `<= 0.8` versus the fastest
named applicable incumbent on both dense-coordinate serial lookup and packed spatial build+probe,
with unchanged correctness, workload and worker budgets. This supports a scoped lead over those
comparators only, not a universal SOTA claim. No aggregate, std-only win, historical envelope, reduced
matrix or trusted-boundary grade closes G1. Cryptographic-face implementation, GPU/ISA work, a new
container, lossy identity and universal O(1)/WCET claims are non-goals of this performance leg;
HashDOS resistance, remove/growth obligations and every named comparison remain required. "A new
container" is superseded **for the counting face only** by the counting-face supersession in
[`APEIRON-REORIENT`](#five-barrier-execution-sequence-apeiron-reorient), which selects one
single-region counting representation refining the same canonical `Apeiron.Table` object. Every other
item in that non-goal list, the strict command, the `<= 1.0` every-pair ceiling, the `<= 0.8` stretch
target and every HashDOS/remove/growth/comparison obligation are unchanged.

Foundational verification precedes the next large AI implementation. A joint readiness profile is an
explicit specification obligation and is currently MISSING; it is not satisfied by Song A0 alone,
and the existing Song A0–A6/B6 milestones are not dropped. Verification that is not complete keeps
the next large implementation gated.

**Terminal goal:** one admitted R0 collapse/fold primitive, connected by checked refinements from
its historical implementation and semantic law through normal form/translation to Rust and
Song/Slang execution, satisfying G1/REG-1 and supporting the secure OS-language-proof system.

The current extraction/refinement leg continues on the real growable implementation. The
[Kani-retirement receipt](evidence/kani-retirement.md) records fresh regeneration/checking at
generated SHA-256 `92c771510c259bc56c9e437e7d3f3108fa9e2796e6ba41ca2cd826f4127ff74e`
(1708 extraction-package jobs; 3482 corpus jobs). It does not close the universal read/search links,
`SearchDecisionCorrect`, or the allocator adapter. Older
fixed-shard execution statements and the repeated W4–W8 dispatch chain in the handoff are stale;
the [consolidation report](roadmap.md#conflicts-and-follow-on-documentation) records
the exact reconciliation tasks. The resource/refusal contract remains transactional checked
reservation (option (b)); allocator realization remains outside the extracted core.

1. **Continue the read/search proof leg.** Follow [the current bounded plan](epics/read-proof.md)
   for the still-open universal read/search links and allocator boundary. Retain the
   identity-injectivity, constructor, relocation, schedule and refusal obligations. Source/pin
   presence and fresh extraction are not semantic closure. G1/REG-1 stays **OPEN** under the
   unchanged all-row contract above; historical benchmark receipts are not reruns of this growable
   source. No restored historical source is automatically admitted.
2. **Connect the same primitive.** Close the audited cross-revision collapse/fold correspondence
   and the admitted SKI/derived-Iota translation links without promoting historical anchors.
3. **Complete the execution refinements.** Song source binding is a later refinement leg of that
   primitive, not a separate semantic project. Compose the checked `Joint.WordBridge` transport
   with source-connected refinements and a decoded-context witness on a matched domain, retaining
   the local source-binding, evaluator, axiom-cone and integration gates.
4. **Compose the Universal Algorithm Bridge and realization demonstration.** Song A7 follows the
   A0–A4 core for bulk ingestion; lift/adapt, resonate/cancel and project compose on one admitted
   source/target profile. Independent A5/A6 application slices inform core interfaces in parallel.
   A0–A6/B6 release conditions stay unchanged. This assigns bounded cross-repo tasks, not an entire
   repository; no giant AI port now.

**Verified-AI order.** The staged order to a demonstrable verified-AI artifact — apeiron core
barrier, Neural rebase, end-to-end extraction, Song base and correspondence, pinned-domain universal
translation, AI architecture iteration, release only after rebase and end-to-end proof — is recorded
with its prior-art index (paper1, paper2, spinor-ttl, the Neural corpus) in the
[programme roadmap](roadmap.md#verified-ai-sequence-and-prior-art-index). Publication
of pre-rebase artifacts is not a programme step. Survival income is a separate track.

**AI-track re-anchor (lead direction).** The stage-2 port is reference material, not the AI track:
no ported module imports the canonical object, and the historical count-LM/entropy/ridge names
carry no reduction to `R[x]/(x²−σ)`. The AI track restarts from the trichotomy with a minimum
standard model/training surface rebuilt natively in apeiron — TorchLean at `de3192df…` is read as the
reference for that surface's scope, not adopted as a dependency; reusable pieces (rings, tensors,
folds, autograd) are generalised where they serve beyond the model. On that foundation sits the O(1)
training/finetuning/inference and forward-only-learning model. Intent:
[`docs/epics/ai-track-reanchor.md`](epics/ai-track-reanchor.md). Spec and plan follow before any
task is allocated.

**Inference-parity priority (user direction).** Before the architecture track resumes, the AI work
runs an existing open-weight model from the apeiron tree and reproduces the `fold` engine's recorded
llama.cpp comparison. No source-level component sits outside the model, and the executing path is
integer/fixed-point: the existing Aeneas integer model with transcendentals as proven integer
implementations carrying error bounds. The Aeneas float fork is deferred with a trigger — a float
artifact that must itself be verified, not a test oracle — so Song's frozen generation-one SoftFloat
core and one-ulp bound (`song-ref` `ae688f66…`) remain seeds, not an active lane; Song's current prover
is integer-linear and cannot carry IEEE-754. `rustc`/LLVM codegen, the CPU ISA and — for the GPU lane —
the SPIR-V toolchain, driver and hardware are the named platform boundary. The engine source is the old
`fold` tree; the comparison receipts are `fold/results/vs-llamacpp-27b.md`,
`fold/results/h2h-8b-at-the-served-carrier.md`, `fold/results/vs-llamacpp.md`. The seam is rejected and
the integer decision recorded in [`docs/epics/inference-parity.md`](epics/inference-parity.md); the
toolchain feasibility spikes (integer transcendentals, Aeneas float model, Song prover) are complete,
and the parity [spec](epics/inference-parity.md) and [plan](epics/inference-parity.md) precede task
allocation. The `Nous` waves (W4-T2 face instances, W4-T3 channel-form attention) are paused behind
this milestone.

Open acceptance obligations retained by the revised boundary: the `10^8` identity row and the coverage
census; the unextracted `std::thread::scope` orchestration boundary and its trust statement; integration
of the canonical/bridge identity-injectivity chain on the selected source; allocator success/failure correspondence of the
ratified resource/refusal contract as an unextracted trust boundary; and the Slang empty-axiom-cone
separation from the Rust/Aeneas bridge cone. None of these is closed by this documentation pass.
Extraction freshness must bind the selected source and generated roots through `proof/regen.sh --check`.
Earlier green receipts remain historical; this documentation task does not rerun that gate.

Scientific concurrency is preserved: independent tracks continue in parallel, and small bounded
interface probes are permitted and expected. A probe answers one interface question and ends in
evidence or a blocker; it does not become a port.

Dispatch discipline: each task owns at most three files and one gate with one negative control; a
wave is a barrier; the next wave opens only when every child and shared gate is green. A missing
gate is named as an artifact to create, never replaced by a command that does not exist. Local commands are in
[AGENTS.md](../AGENTS.md) §6 and [proof/README.md](../proof/README.md). A successful build, a
sampled test and a universal proof are different evidence and are labelled separately.

## What is not claimed

No claim is made that the migration is complete, that the cross-repo correspondence exists, that
the Song certificate reader refines the canonical decider, that callbacks outside `CallbackLawsG`
refine the canonical table contract, that the word ring
carries the real-valued energy or nonnegativity results, that the model theorems establish a global
physics equality, or that rustc/LLVM, the kernel ABI or a GPU backend implement the source
semantics. Those are open programme obligations, not silent assumptions. A citation to a theorem, a
paper or a sibling repository is a pointer, not evidence, until the named artifact exists and the
in-repo link is checked.
