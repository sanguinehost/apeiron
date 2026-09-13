# Apeiron Programme Direction

This file is the sole programme authority. It states the shared thesis, ownership, outcomes,
primitive correspondence, migration and execution order. Local documents own local representations,
trust and execution detail and link here: [local boundary](kernel-boundary.md) and
[operator handoff](flash-handoff.md). The central programme coordinates named cross-repo tasks; it
does not dispatch an entire repository or replace its local gates. Routine work proceeds within
the repository's scope; changing a programme boundary requires an explicit owner decision.

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

The programme goal is a chain:

```text
lift (source / LLVM)
  -> canonical semantics
  -> checked reduction / subsumption
  -> actual realization
  -> safe and useful programmes
```

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

## Ownership

- **Apeiron** owns the canonical definitions, the reusable Rust primitives and their refinements.
  Lean is retained: the canonical mathematics, its history and its attribution stay in the tree.
  Reusable Rust and refinements converge in Apeiron. The approved destination is Apeiron, not a
  sibling tree. Source preservation targets a canonical package that is independent of any single
  Aeneas extraction link.
- **Song** remains an independent, adjacent assembly-up implementation: its own assembly, Slang
  implementation, contracts and local build/runtime acceptance. There is no Rust or Aeneas runtime
  linkage between Apeiron and Song, and Lean is not deleted. Song's local gates stay owned by Song.
  The required cross-repo result is a proof of primitive correspondence, not runtime linkage.
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

The readiness profile is the gate between the foundation and the next large implementation. It is an
explicit specification obligation: it names the primitives, their matched domains, the
correspondence obligations and the evaluator that the first vertical demonstration consumes. It is
currently MISSING. Song A0 alone does not satisfy it, and the existing Song milestones are retained,
not replaced by it. Joint readiness is not an Apeiron-only gate: it also requires the Song
foundation, the joint primitive correspondence and the named profiles. An Apeiron barrier is a
necessary component, not sufficient by itself.

| Outcome | Owner | Consumer | Dependencies | Current evidence vs open gate | Closure condition |
|---|---|---|---|---|---|
| Retained Lean + reusable Rust source convergence | Apeiron lead | all programme work | none | Both source trees and recorded successful builds exist. Open: the canonical corpus is still at `../sanguine-apeiron/proof` (Git `50c94192`), not in the Apeiron tree. | One revision holds canonical Lean and reusable Rust, both built and audited on one source/tool identity. |
| Literal algebra, combinator and evaluator foundation | corpus lead (Apeiron) | Song, Rust, physics | retained Lean | `Apeiron.Combinator.routes_are_a_combinatory_algebra` and `Apeiron.Combinator.denote_reduces` are checked theorems. `Apeiron.Combinator.Comb.normalize` is a fuel-bounded evaluator definition; `Apeiron.Combinator.Comb.verdict_decided_exact` proves exactness when the verdict is not `.fuel`. `Omega.lean` proves non-normalization for a named term; that is a boundary, not an unfinished theorem. Open: selected finite-machine representations and executing evaluator refinement. | The selected executing evaluator refines the canonical semantics on an explicit domain and reports resource exhaustion without a false verdict. |
| Independent Rust–Slang primitive correspondence | cross-repo lead (Apeiron + Song) | both executing implementations | retained Lean; combinator evaluator | The required joint primitive correspondence is not established by the recorded evidence. `Apeiron/Combinator/Verdict.lean` exists; the missing artifact is the concrete-runtime refinement and the matched primitive profile. | A checked joint primitive profile relates both executing implementations to the same canonical definition on a matched domain, including representation, state/effects, admission/refusal and resources. |
| Lift / universal translation | Apeiron lead, with Song owning its source/target adapters | subsumption, realization | source and semantic profiles | Real Rust extraction and selected canonical refinements exist; historical LLVM/Lean lifting supplies transfer input. Open: an admitted source population, effects/undefined-behaviour contract and complete lift-to-realization gate. | Supported source interpretations and target realizations compose through canonical semantics, with unsupported inputs refused explicitly. |
| Checked reduction / subsumption | Apeiron lead | realization, transformation | lift | `Apeiron.Combinator.Comb.hoist_preserves_semantics`, `Apeiron.Combinator.denote_reduces` and `Apeiron.Combinator.Comb.verdict_decided_exact` establish a scoped hoist reduction, route-denotation preservation under reduction and exactness on the decided fragment, not a general subsumption or translation theorem. Open: the source-level checked reduction/subsumption relation. | Lossless reconstruction or observable relative subsumption is checked, with the observation set, refusal and resources stated. |
| CPU / Slang / GPU realization | Apeiron + Song leads | applications, measurement | subsumption | Named `Cell<i64>` source kernels are refined. The programme-wide joint refinement is not established; existing Slang contracts and model bridges, and GPU experiments, have their own scope. | Each realization refines the same supported semantics, with target features, reduction order and numerical tolerance specified. |
| Secure OS-language-proof-kernel composition | cross-repo lead (Apeiron semantics/refinement + Song execution) | literal-machine and application tracks | retained Lean; Song A0-A6/B6; primitive correspondence; lift/subsumption; realization | Song retains its local security/runtime gates, and historical kernel, language and prover artifacts exist. Open: checked end-to-end composition relating OS isolation, language semantics, proof checking and executing primitives to the canonical algebra and SKI/derived-Iota semantics. | A revision-pinned vertical system passes Song's local gates; checked composing refinements cover the admitted language, proof kernel, OS/ABI boundary and executing primitives, with threat model, trust boundary, refusal and resources explicit. |
| Measured transformation benefit and productivity | lead | programme | realization | Historical Fold numbers exist; no benchmark was rerun in this pass. | A named machine, metric and interval measure before/after transformation cost and developer productivity. |
| Active test-time learning and lower-cost useful AI | AI-track lead | applications | joint foundation readiness; realization | Historical formal and executable records exist: [paper1](../../sanguine/release/paper1/INDEX.md) and [spinor-ttl](../../spinor-ttl/README.md). Existing work is not new acceptance. Open: a joint learning/update profile and its executing correspondence. | Streaming updates and retained knowledge satisfy the selected contracts; adaptation quality, learning/inference cost and resource bounds meet a pinned comparison profile. The research target is order-of-magnitude cost improvement at matched useful capability, not an established result. Inference or quantization alone does not close learning. |
| Physics: classical/quantum correspondence and neutrino prediction | Apeiron physics lead | scientific and application tracks | algebra foundation | Conditional model theorems exist in `Apeiron/Physics/Madelung.lean`, `Apeiron/Physics/Flavour.lean` and `Apeiron/Cosmology/NeutrinoSum.lean`. Real-valued carrier conservation and scoped kinetic-energy results are proved under their stated hypotheses; ring-generic theorems establish conservation and norm-form-ledger identities, not kinetic energy or nonnegativity. The word ring carries no energy claim. Open: joint scientific conformance profiles separating model theorems from empirical confirmation. | Named model predictions are checked against their test profiles under stated domains and tolerances, independently of game integration. |
| Application and game integration | Penglai + Whispering Tides leads | players, scientific users | physics foundation; realization | Historical toy, game and scientific examples exist; this pass establishes neither current integration testing nor refinement. Open: a playable or scientific loop consuming checked model readouts. | A playable or scientific loop consumes checked model readouts with its own integration gate. |
| Federation / distributed | Song network lead or an assigned distributed-service lead | applications, including Whispering Tides if the profile calls for it | primitive correspondence | The historical Sanguine federation transfer lineage is recorded in the [song network roadmap](../../song/docs/network/ROADMAP.md) (C3/G5) as design input. Open: a joint current protocol profile; existing network experiments are historical. | Distributed peers exchange checked, revision-pinned state with explicit delivery, identity, partition and duplicate semantics. |

## Evidence status

Claims are stated at the grade the artifact supports.

- **Table.** Current proofs cover the constructor, the default read, first/present writes at key 0,
  the ordered two-write build, the bounded 16-slot search and the concrete full home-shard refusal.
  They do not establish arbitrary `Rep` preservation, admission completeness or physical
  subsumption. Universal `Rep` preservation is the open priority.
- **Decider F2.** `capped_certificate_complete` proves an accepted capped certificate exists iff no
  assignment inside the box satisfies the rows. It is not a `CertOp` derivation, and it does not
  prove that every uncapped unsatisfiable system has a `checkCert`-accepted log.
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
different physical layouts: the Rust `Table<K, V, H, N>` is fixed-shard inline storage; Song's
Slang/assembly representation is its own. Different layouts can correspond, but only under a
matched admission relation and the four obligations above. A differential test or a shared example
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

The physical canonical corpus remains at `../sanguine-apeiron/proof/Apeiron/`, Git
`50c94192399add3b9f8e64bd14e36d8cba270705`, Mathlib
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. Apeiron is the approved destination; the sources have
not moved, and no extraction, package or pin change is authorized by this document. A migration is
a separately gated task with its own lifecycle artifacts.

Keep source paths and pins accurate: a path written in a document must resolve. Distinguish the
approved destination from the not-yet-moved source. A source-proof run log records what happened at
a revision; it is not fresh acceptance. The release identity is recorded in
[proof/README.md](../proof/README.md); the run narrative is in
[autonomous-run-log.md](autonomous-run-log.md).

Historical evidence stays at its source and is pointed to, not copied. The predecessor documentation
at Git `1403767e671c81737525702946a3eb1eae4f0b0b` is retained: the storage lifecycle
([intent](intent/storage-refinement.md), [spec](spec/storage-refinement.md),
[plan](plan/storage-refinement.md)), the [carrier refinement plan](plan/carrier-refinement.md) and
the [corpus release pin plan](plan/corpus-release-pin.md). The combined
[programme plan](plan/programme-roadmap.md) records this documentation pass.

## Execution order

Foundational verification precedes the next large AI implementation. A joint readiness profile is an
explicit specification obligation and is currently MISSING; it is not satisfied by Song A0 alone,
and the existing Song A0–A6/B6 milestones are not dropped. Verification that is not complete keeps
the next large implementation gated.

Priorities, in order, as prerequisites admit:

1. **Current Apeiron work.** Universal `Rep` preservation and admission completeness for the
   fixed-shard `Table`, plus the remaining Rust/Kani, extraction and bridge barriers. The current
   table proofs cover the constructor, the default read, first/present writes at key 0, the ordered
   two-write build, the bounded 16-slot search and the concrete full home-shard refusal; they do not
   establish arbitrary `Rep` preservation, admission completeness or physical subsumption.
2. **Independent Song A0–A6/B6**, owned by Song, with its local gates unchanged, progressing in
   parallel with Apeiron work. Cross-repo coordination assigns bounded tasks, not whole repositories.
3. **A cross-repo matched primitive profile and executing evaluator refinement**, the prerequisite
   for the correspondence outcome above. Profile design can proceed alongside the local foundations.
4. **A bounded vertical lift/reduction demonstration**, only as the prerequisites admit. No giant AI
   port now.

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
the uncapped decider is complete, that arbitrary table states preserve `Rep`, that the word ring
carries the real-valued energy or nonnegativity results, that the model theorems establish a global
physics equality, or that rustc/LLVM, the kernel ABI or a GPU backend implement the source
semantics. Those are open programme obligations, not silent assumptions. A citation to a theorem, a
paper or a sibling repository is a pointer, not evidence, until the named artifact exists and the
in-repo link is checked.
