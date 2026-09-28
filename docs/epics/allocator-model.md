# allocator backend repair

Merged epic record: `intent/allocator-model.md`, `spec/allocator-model.md`, `spec/allocator-may-fail.md`, `plan/allocator-model.md` (one file per epic; see `docs/README.md`).

## Intent — allocator backend repair

- Owner: Astra `APEIRON-RESOURCE-W2-DESIGN`; Sol adjudicates semantic changes; lead dispatches.
- Base: `262bb0a94eff2cc7d5655122a954f44d74de96b6` plus `target/open-areas/baseline.json`.
- Status: proposed architecture, `anchor`; no backend edit or compiler run.
- [Specification](allocator-model.md); [plan](allocator-model.md).

### Problem and decision

The active Aeneas vector has a capacity field, but `push`, `extend_from_slice`, `resize` and `deref_mut` can replace it with the resulting length. Reservation succeeds deterministically below the unsigned size ceiling. Its effect has only nonreturning failure, and the existing WP identifies a successful computation with `ok`. These facts block capacity receipts and ordinary allocator-refusal conformance.

Repair capacity preservation first in the existing repository-owned backend patch. Preserve production Rust and the existing abstract reservation contract in [allocator-may-fail](allocator-model.md); do not add another abstract vector/reservation model to obtain a passing gate. Ordinary refusal then requires source-bound layout information, a returning allocation effect, and an explicitly named effect-aware WP over the actual Aeneas computation. Existing pure WP statements retain their meanings.

The backend path remains the selected candidate. Its generic type-layout lowering and returning-effect proof feasibility are unestablished. No implementation may substitute an assumed type layout, deterministic overflow refusal, always-success allocator, or assumed `ResourceAdapterSimulates`.

### Completion conditions

| Result | Evidence required |
|---|---|
| Capacity repair | Real patched `Vec` reserve→push/extend/resize/back-borrow probes; reverting the real changed body makes the probe fail |
| Ordinary may-failure | Same valid growing request has success and refusal continuations; every admitted reply satisfies its postcondition; no-growth/ZST requests issue no allocator effect |
| Production connection | Real-source extraction, actual observed-capacity simulation, source-linked ordering/trace, both adapter branches and rejecting controls |
| Integration | Both Lake packages, extraction freshness, closed-cone audit, Rust gates, source/tool digests on one identity |

The first row alone is a backend-model repair, not whole-adapter refinement. No performance claim follows.

### Invariants and rejected alternatives

- Keep canonical pins and definitions, generated-source selection, Rust APIs, existing admitted domains and failure mappings unchanged.
- `ResourceAdapterSimulates.refused` still requires whole-state equality until Sol separately accepts a replacement contract and every consumer migrates.
- No installed `target/aeneas*` edit is source. Patch and test a disposable backend materialized from the recorded installed base.
- Preserve the current pure `spec_imp_exists`; changing it into a nondeterministic theorem would invalidate its stated equality and downstream uses.
- Existing failure injection tests establish branch/order behavior, not global allocator realization. Oversized requests establish neither valid layout nor ordinary allocation refusal.

### Trust boundary

The allocator, compiler/codegen, process abort, arbitrary callback effects and runtime layout correspondence remain separate obligations. The proposed WP covers admitted terminating replies; it does not prove the operating system returns one. Rust/canonical closed cone remains `{propext, Classical.choice, Quot.sound}`; no resource model enters the Slang empty-cone chain.

## Specification — allocator backend repair

- Owner: Astra interface proposal; Sol ratification required.
- Source identity and status: [intent](allocator-model.md).
- Refines the existing [may-fail contract](allocator-model.md), not a second reservation model.

### Capacity interface

Use the active `alloc.vec.Vec α = { slice, cap : Usize }`. Define a predicate over it, not another executing vector:

```text
CapacityInv(v) := v.val.length ≤ v.cap.val
CapacitySim(layout, rustVec, v) :=
  ElementRep(rustVec.elements, v.val) ∧
  rustVec.len = v.val.length ∧ rustVec.capacity = v.cap.val ∧
  CapacityInv(v) ∧ CapacityLayout(layout, v.cap.val)
```

Exact capacity equality is selected because generated code executes `Vec.capacity` and places its value in receipts. A mere lower bound needs a separately proved observation simulation and is not silently substituted. `VecState`/`ReserveRel` in the accepted may-fail specification remain the abstract view: when their implementation exists, project the active vector under `CapacityInv`, rather than introduce another oracle model. No `Resource/AllocatorModel.lean` implementation exists in this checkout.

| Operation | Required successful result | Remaining semantic boundary |
|---|---|---|
| `new` | Empty contents; cap 0 for non-ZST, `Usize.max` for ZST | Current generic `new` always uses cap 0; layout translation is required for exact ZST capacity |
| `from`, `ofSlice` | Internal exact-length constructor establishes `CapacityInv` | A list/slice alone does not identify a concrete allocation's excess capacity |
| `capacity` | Return exactly the modeled observed capacity | No fabricated minimum-capacity receipt |
| `try_reserve`, `try_reserve_exact` | Contents unchanged; cap unchanged if no growth; otherwise successful returned capacity ≥ old cap and need | Ordinary failure is a returning effect on valid growth; invalid layout/overflow is separate |
| `push`, `extend_from_slice` | Contents append; if result length ≤ old cap, cap remains old cap and no allocation event occurs | Growth uses an admitted allocation response; temporary `max oldCap newLen` is a model repair, not exact growth allocation |
| `resize` | Preserve cap on shrink and within-cap growth; preserve required clone behavior | Existing body clones once on growth; generic clone-order equivalence is not established by a capacity-only patch |
| `truncate`, `clear`, `set`, `set_opt`, `update`, `setSlice!`, `index_mut` | Preserve cap; length-preserving callbacks or bounded shorter slices retain `CapacityInv` | Backward slice functions require equal-length/bounded-output obligations; arbitrary slice replacement is not admitted without proof |
| `deref_mut` | Backward update preserves original cap for an equal-length returned slice | Current backward function uses `ofSlice`, resetting cap |
| `with_capacity`, `from_elem`, slice conversion, array conversion, `CloneVec.clone` | Contents and observed cap satisfy source-specific constructor/allocator relation | Current exact-request or copied-source cap is not a general exact-allocation theorem |
| `insert` | Inventory only; no new conformance claim | Current backend replaces `List.set` at `i < length`, whereas production insertion needs its own semantics review; no extracted call on this identity |

The smallest patch changes `push` and `extend_from_slice` capacity construction to `max oldCap resultingLength` without changing bodies' element results, signatures or guards. Prove preservation on the joint `CapacityInv` and within-capacity domain. Subsequent tasks cover every other active construction/backward-update path before any whole-adapter claim.

### Layout and returning-effect interface

For target word width `w`, let `U = 2^w - 1`, `I = 2^(w-1) - 1`. A source-linked element layout records `size`, `align`, target identity, and evidence that `align` is a nonzero power of two and `size` is a multiple of `align`. Define:

```text
Need(v, additional) = v.length + additional
LayoutValid(layout, n) :=
  n ≤ U ∧ roundedUp(n * layout.size, layout.align) ≤ I
```

All arithmetic above is mathematical natural arithmetic; the implementation must reject multiplication/addition/rounding overflow before converting to machine words. For a genuine Rust type, the size/alignment multiple property simplifies the byte bound; unsigned element count alone does not prove it. For ZST, observed cap is `U`; `Need ≤ U` needs no allocation and cannot witness ordinary refusal. `Need > U` is deterministic capacity overflow. No-growth likewise returns success without consulting an allocator.

Proposed backend request, after source-layout lowering feasibility:

```text
ReserveRequest = { target, elementSize, elementAlign, oldLength, oldCapacity,
                   additional, kind : Exact | Amortized }
ReserveReply = Refused | Reserved(observedCapacity)
```

`RustEffect.Input.reserve request` has a returning response, unlike `.fail`. Admission fixes `CapacityInv`, non-ZST, `oldCapacity < Need`, `Need ≤ U` and valid requested layout. A `Reserved c` response is admitted only when `max oldCapacity Need ≤ c ≤ U` and `LayoutValid(layout,c)`; `Refused` preserves the input vector. Both replies exist for the same byte-layout request `(length=0, capacity=0, additional=1)`. Returned excess is allowed; do not identify real capacity with the minimum request. Invalid input must reject before the effect, never create an empty admitted-response population that makes a universal postcondition vacuous.

The effect response type is `{ reply : ReserveReply // AdmittedReply request reply }`, so successful continuations receive the bound needed to construct `Usize`; no clamping or unchecked conversion is permitted. The request is emitted only after admissibility is established, and nonemptiness is proved by its refusal reply. Keep request/layout arithmetic in `Nat` in `Primitives.lean`: importing `Scalar` there would create a dependency cycle. Conversion to/from `Usize` stays in the vector model.

For `try_reserve` the real growth policy may select a larger allocation layout than the minimum need. Its selected-layout arithmetic and possible deterministic rejection require source-specific correspondence; both `Exact` and `Amortized` are retained in the request. The valid-byte witness uses `Exact`. Treating every amortized refusal as allocator exhaustion is unauthorized. Rust's wrapper currently maps all `TryReserveError` causes to `StorageError::AllocationFailed`; the backend proof must distinguish causes internally without claiming Rust exposes that distinction.

**Layout lowering is a missing implementation interface.** Current `Vec.try_reserve {T} (_A : Type)` receives no size/alignment values; the inspected backend has no `size_of`/`align_of` model. `Global` is filtered metadata, not a layout dictionary. The proposed request requires actual type-layout metadata from the selected translation, including generic `T`, ZST and target width. No axiom, universal `size=1` instance, Lean `Type` size guess or extra unbound signature argument can fill this gap. A bounded generic/monomorphic extraction feasibility task must establish the calling convention before an effect patch is dispatch-ready.

### Effect-aware WP

Keep `Aeneas.Std.WP.spec` and `dspec` unchanged: the former means an actual `ok`, and `spec_imp_exists` retains its equality conclusion. Add a distinct, explicit `AllocationSpec` over the actual `Result` interaction tree, with these required rules:

```text
AllocationSpec (ok value) Q        iff Q value
AllocationSpec (vis (reserve q) k) Q iff
  RequestAdmissible q ∧ (∃ reply, AdmittedReply q reply) ∧
  ∀ reply, AdmittedReply q reply → AllocationSpec (k reply) Q
AllocationSpec (fail error) Q     iff False
AllocationSpec div Q              iff False
```

This is a proposed total effect-aware verification predicate, not a second reservation-state relation. Prove monotonicity, bind, pure-spec embedding and pure/effect-free equivalence. Never use an existential choice of reply to prove a universal adapter postcondition. No conversion back to `m = ok value` is valid for an unhandled reserve effect. A proof of termination must reach every admitted continuation; successful existence on one policy does not suffice.

The existing `mvcgen` instance may continue to reject unsupported effects through its dummy exception, with `Result.of_wp` repaired to reject the new case. It must not report allocator-aware verification until a separately checked bridge exists. Direct `AllocationSpec` rules are the initial target. `Vec.extend_from_slice` must preserve returning clone continuations: its current `nomatch` response branch cannot remain. Use a checked continuation-preserving bind and a returned-length contract for every admitted callback result; fail/panic/divergence stay excluded unless separately specified.

No generic handler is presumed available: `Aeneas/Data/Coinductive/ITree.lean:290` states that `ITree.interp` was removed. A handler-based alternative requires a checked interpreter/continuity feasibility result first. The direct inductive WP also remains compiler/kernel feasibility work, with no result claimed here.

### Refusal contract requiring Sol adjudication

`Growable/Resources.lean::ResourceAdapterSimulates.refused` presently requires `committedAfter = state`, zero clone/merge/relocation calls, and the exact published refusal. Keep this contract unchanged in the capacity/effect tasks.

Proposed later contract: `CommittedObservationEq(before,after)` plus `ScratchNeutral(after)` and zero payload-effect counts. It must preserve default/home, canonical reads for every admitted `(Sigma,K)`, live count, resident/spill payload and occupancy observations, all committed locator links/lookup results, and future accepted addressing behavior. It may allow unobservable increased scratch capacities or first-use empty directory materialization only after a reachability/representation argument proves those differences do not change subsequent behavior. A changed profile is not erased merely because no key was inserted; first initialization needs the actual `H3_40::new` and routing correspondence. Staging must be absent on refusal.

Sol must choose either (a) require physical restoration in production and prove current equality, or (b) separately approve this observation relation, preserve the current strict theorem at its original scope, and migrate all consumers/audits under new names and a source-linked proof. No projection theorem supplies either execution fact. `reserve_path` can materialize `Some(Directory::new())`, grow links and initialize a profile before a later refusal; `Directory::reserve_for` can grow several capacities before another reserve fails. Generated `prepare_from_plan` returns `self1` on locator refusal. These are reasons to adjudicate the contract, not permission to weaken it.

### Source inventory

Backend paths below are relative to `/home/socol/Workspace/apeiron/target/aeneas-model/lean/`; their editable source is the repository patch.

| Surface | Exact consumers / obligation |
|---|---|
| Effect constructors/eliminators | `Aeneas/Std/Primitives.lean`: `RustEffect.Input/Output`, `Result.vis/fail/cases/match`, `bind_vis`, `bind_fail`; generic ITree operations must preserve replies |
| Explicit impossible-response eliminator | `Aeneas/Std/Vec.lean::extend_from_slice`, `.vis eff k => ... nomatch x` |
| WP/automation | `Aeneas/Std/WP.lean`: `spec_vis`, `dspec_vis`, `spec_equiv_exists`, `spec_imp_exists`, `dspec_admissible`, bind rules, `Result.instWP`, `instWPMonad`, `of_wp`, `spec_to_mvcgen`, `dspec_to_mvcgen`, loop rules |
| Direct legacy-WP consumers | `Aeneas/Std/{Array/Array,Scalar/CheckedOps/Rem,Slice,Vec,WP}.lean`; `Aeneas/Tactic/Step/Init.lean` |
| Direct vector construction consumers | `Aeneas/Std/Vec.lean` operations in the capacity table; backend reverse-import roots `Aeneas.Std` and `Aeneas.Std.VecIter`; local `Growable/Scan.lean` concrete `Vec.from` fixture |
| Local legacy-WP consumers | `Growable/{Identity,Scan,Search}.lean`, `Joint/TableInstance.lean`, `TableRefinement/{Callback,PhysicalBuildGeneric}.lean`, and `TableRefinement{,Admission,AdmissionGeneric,Build,BuildGeneric,Generic,Physical,PhysicalBuild,ReachablePrefixExactN}.lean` under `proof/corpus/` |
| Adapter relation consumers | `Growable/Resources.lean`: `resourceAdapter_refusal_stutter`, stored/regen reachable-publication witnesses; four matching audit entries in `AxiomAudit.lean` plus publication headlines |
| Actual extracted vector calls | `proof/ApeironWord.lean`: 23 `push`, 5 `resize`, 15 `deref_mut`, 2 `extend_from_slice`, 5 `try_reserve`, 3 `try_reserve_exact`, 7 `capacity`, 2 `clear`, 3 `truncate`, 1 clone; source-text occurrences, not execution counts |
| Actual Rust reservation/initialization | `src/table/{resources,state,locator}.rs`, `src/table.rs`, `src/coord/locator_core.rs`; generated reserve/observe/materialize/prepare bodies already exist |

A source import-graph scan found reverse closures of 149 modules from `Primitives`, 143 from `WP`, and 41 from `Vec`. Direct effect/eliminator search found exactly the three backend files above. These inventories are source findings, not compiler acceptance or an assertion that every transitive module needs edits. Freeze them per patch and build the backend/corpus gates; do not silently broaden one task across consumers.

Primary Rust references read locally: `library/alloc/src/raw_vec/mod.rs` and `library/core/src/alloc/layout.rs` under `/home/socol/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/`. They show ZST capacity, layout checks, amortized growth and allocator-result handling; their correspondence to the extraction toolchain remains to be proved. Local `rustc` reports `1.97.0-nightly (17584a181 2026-04-13)`, not the pinned Charon Rust toolchain.

## Allocator may-fail boundary

- **Owner:** Apeiron
- **Source revision:** `262bb0a94eff2cc7d5655122a954f44d74de96b6`; accepted dirty baseline in
  `target/readlink/baseline.json`
- **Status:** design verdict
- **Grade:** `model`; no allocator implementation or conformance theorem is established

### Decision

Represent reservation as a relation with two possible outcomes for the same admissible input. Keep
that abstract relation separate from `ResourceAdapterSimulates`, which remains the named obligation
connecting production planning, allocation, capacity observations and transactional refusal to the
generated values.

Do not describe the current extracted `Vec.try_reserve` as may-fail. Its patched body selects success
whenever `length + additional ≤ Usize.max`; the outer Aeneas `Result` has no effect that can choose an
ordinary allocation refusal. A size overflow is a checked arithmetic refusal, not evidence that an
allocator may refuse a representable request.

The recommended order is:

1. kernel-check the relation and its same-input joint witnesses as a local `model`;
2. inventory the full backend effect change and the remaining active capacity-model invariants;
3. decide between that backend change and a source-level injected reservation boundary;
4. implement and prove production conformance only after the selected interface has a real refusing
   control.

Steps 2–4 are not dispatch-ready from this specification.

### Scope and admissible domain

For an element type `α`, let an abstract vector state be its logical elements and an observed
capacity:

```text
VecState α = { elements : List α, capacity : Nat,
               length elements ≤ capacity, capacity ≤ Usize.max }
```

For fixed `before : VecState α` and `additional : Nat`, define
`need = length before.elements + additional`. Arithmetic admissibility requires
`need ≤ Usize.max`. A refusing joint witness additionally requires a non-zero-sized element type,
`need > before.capacity`, and a valid allocation layout for `need` elements. `LayoutValid α need` is
left abstract until the implementation choice fixes the exact Rust layout bound; `Usize`
arithmetic alone does not establish it. The relation returns one of:

```text
ReserveOutcome α = Success (after : VecState α) | Refused AllocationFailed
```

`ReserveRel before additional outcome` has these branches:

- `Success after`: `after.elements = before.elements`,
  `length after.elements = length before.elements`, `before.capacity ≤ after.capacity`, and
  `need ≤ after.capacity`.
- `Refused AllocationFailed`: the logical state is `before`; no prepared vector or capacity receipt
  is exposed. The ordinary may-fail branch requires `need > before.capacity` and
  `LayoutValid α need`; sufficient existing capacity is not a refusal witness because no growth is
  required.

The refusal branch does not permit overflow: its joint witness also satisfies `need ≤ Usize.max`.
Overflow belongs to the existing checked-sizing boundary and may return the appropriate sizing or
capacity error deterministically before allocation.

#### Same-input non-vacuity

One fixed admissible input must support both branches. For example, the future executable model may
choose a concrete non-zero-sized element type, `before.elements = []`, `before.capacity = 0`, and
`additional = 1`, prove that the one-element layout is valid, then construct:

- a success witness with unchanged elements and capacity at least one; and
- an `AllocationFailed` witness with the same `before` and `additional` and unchanged logical state.

Separate success and refusal examples with different inputs do not establish this property. These
witnesses show only that the relation is non-vacuous; they say nothing about which result a real
allocator returns.

An oracle-indexed model may choose the branch:

```text
reserveModel : Bool → VecState α → Nat → ReserveOutcome α
```

The Boolean is specification input, not hidden Aeneas nondeterminism and not a Rust allocator. The
operation and its joint witnesses retain grade `model`.

### Production boundary

Production `src/table/resources.rs::try_reserve_exact` calls
`Vec::try_reserve_exact` and maps its error to `StorageError::AllocationFailed`. On success,
production later observes real capacities and publishes a prepared envelope. On error,
`publish_prepared` exposes `PreparedResult::Refused(error)` without a plan, storage, receipt or held
carrier.

`src/table/tests.rs::selected_resident_reservation_failure_leaves_table_unchanged` supplies
`usize::MAX / 256` to the flat reservation path and observes an error while a separately constructed
table retains its inserted value. This is useful refusal-path test evidence, but it does not
distinguish allocator exhaustion from `Vec` capacity or layout rejection: the production wrapper
maps every `TryReserveError` variant to `AllocationFailed`. The test therefore is not the required
same-valid-layout may-failure witness. The summary-sizing refusal test in the same file is explicitly
a pre-allocation `SizeOverflow` control.

`proof/corpus/Growable/Resources.lean::ResourceAdapterSimulates` is deliberately stronger than the
abstract reservation relation:

- its `prepared` constructor requires the concrete plan/storage certificates, sufficient observed
  capacities and the exact published generated value;
- its `refused` constructor requires committed-state stutter and zero clone, merge and relocation
  calls.

Neither `ReserveRel` nor `reserveModel` constructs either constructor. In particular, an oracle
refusal does not prove that Rust allocation refused, and a real Rust error does not by itself prove
the no-effect facts for the whole adapter. No downstream theorem may assume
`ResourceAdapterSimulates` merely because the abstract relation has both branches.

### Stock and active model limitations

The stock backend and the active patched model establish different facts:

1. Stock `target/aeneas/backends/lean/Aeneas/Std/Primitives.lean` defines `RustEffect.Input` with only
   `fail`; its output is `PEmpty`. Active
   `target/aeneas-model/lean/Aeneas/Std/Primitives.lean` has the same SHA-256 and therefore the same
   limitation. `Aeneas.Std.Result` offers no environment response that could select an allocator
   outcome.
2. Stock `target/aeneas/backends/lean/Aeneas/Std/Vec.lean` represents `Vec α` only by `slice`. This
   explains why the patch had to change the representation; it is not the active representation.
3. Active `target/aeneas-model/lean/Aeneas/Std/Vec.lean` carries `cap : Usize`, preserves capacity
   through truncation/clear, and sets successful reservation capacity to `max oldCapacity need`.
   Its remaining gap is not missing capacity state: it lacks a recorded invariant
   `length ≤ cap`, an explicit valid-layout predicate, and a may-fail choice for a valid request that
   actually needs growth.

Stock `target/aeneas/backends/lean/Aeneas/Std/WP.lean` admits a returned `ok` in `spec` and
additionally divergence in `dspec`; active `target/aeneas-model/lean/Aeneas/Std/WP.lean` is
byte-identical. Neither adds allocator choice. The active model has the capacity field but keeps
`try_reserve` deterministic below the size ceiling. An effect response or explicit input is still
necessary for same-input choice.

### Architecture alternatives

| Alternative | Semantic result | Production impact | Decision |
|---|---|---|---|
| Complete the active Aeneas capacity invariants and add allocator choice to the effect/WP | Extraction can retain a may-fail result and prove both model branches | No Rust behavior change; broad backend proof and handler migration | Candidate, pending complete backend dependency inventory |
| Inject a source-level reservation capability/outcome into the adapter | Rust tests can force the refusing branch; extraction sees explicit input | Changes Rust interfaces and threads a capability through table and locator reservation | Candidate only if the bounded source surface is demonstrated |
| Keep a Lean-only oracle relation | Gives executable joint witnesses and downstream conditional lemmas | No production link | Recommended first step, grade `model` only |
| Treat `need > Usize.max` as allocator refusal | Deterministic and currently expressible | Conflates sizing overflow with ordinary allocation failure | Rejected |
| Assume `ResourceAdapterSimulates` | Makes downstream implications usable | Hides the missing production proof | Rejected |

The active capacity-carrying backend is preferable if its invariant and effect migration can be
bounded: it leaves the runtime allocation path unchanged and keeps the actual write path in the
extraction cone. That preference is not an implementation authorization. The installed effect is
matched in other backend modules, so changing only `Primitives.lean`, `WP.lean` and `Vec.lean` is
not known sound.

The source-injection alternative has a stronger executable Rust control but an unknown propagation
surface. Reservation occurs in table storage and locator paths, including direct `Vec::try_reserve`
calls outside `src/table/resources.rs`. A single wrapper edit would not establish whole-adapter
refusal or no-effect behavior.

### Bounded downstream decomposition

Every task below owns at most three paths. Only the first is ready after review of this design.

| Task | Owned paths | Result and gate | Control |
|---|---|---|---|
| Executable relation | `proof/corpus/Growable/AllocatorModel.lean`, `proof/corpus/lakefile.lean`, `proof/corpus/AxiomAudit.lean` | Define `VecState`, `ReserveRel`, both same-input witnesses and audit them; bounded `lake build Growable.AllocatorModel AxiomAudit` | In a disposable copy, remove the refusal constructor or require overflow; the named same-input joint-witness theorem must fail, then the restored gate passes |
| Backend dependency inventory | `docs/evidence/allocator-effect-inventory.md` | Enumerate every constructor, eliminator and effect handler affected by capacity and allocator choice; path/diff review only | Not applicable to a design inventory |
| Backend architecture plan | `docs/epics/allocator-model.md`, `docs/epics/allocator-model.md`, `docs/epics/allocator-model.md` | Select effect response versus explicit input and decompose upstream edits into disjoint tasks | Not applicable until executable tasks are allocated |
| Source-injection feasibility, if still competitive | `docs/evidence/allocator-source-injection.md` | Enumerate all reservation call paths and the minimum interface propagation; path/diff review only | Not applicable to a design inventory |

The executable relation task must not import generated `Vec`, claim production conformance, or
alter `ResourceAdapterSimulates`. It establishes the mathematical contract and semantic sensitivity
only. Backend or source implementation tasks require a later Astra decomposition after the two
inventories decide the actual file surfaces.

### Required later evidence

A production claim requires all of the following on one source/tool identity:

- actual source extraction of the selected reservation interface;
- a capacity representation supporting the success postcondition;
- an executable same-input success/refusal model theorem;
- a real-source control that forces ordinary allocation refusal on a representable request;
- construction of both relevant `ResourceAdapterSimulates` branches, including refusal stutter and
  zero clone/merge/relocation calls;
- the focused extraction, Lean build and axiom-audit gates.

No deterministic huge-size test substitutes for the fourth item. It may test overflow handling,
which remains a separate contract.

### Unknowns and trust boundary

- The complete Aeneas effect-handler migration surface has not been counted.
- The minimum source-level injection surface across table and locator reservation has not been
  counted.
- Stable Rust does not provide a portable, deterministic way to force the global allocator to
  refuse one otherwise representable `Vec` reservation.
- The production allocator, rustc/LLVM, system allocator and out-of-memory process behavior remain
  outside current proof evidence.
- `ResourceAdapterSimulates` remains unproved.
- This document makes no performance claim and changes no allocator behavior.

### Acceptance

Review every cited path and the distinction among overflow, abstract refusal and production
conformance, then run:

```sh
git diff --check -- docs/epics/allocator-model.md
```

Semantic mutation controls are inapplicable to this prose-only design artifact. The executable
relation task above owns the first semantic red/restore control.

## Plan — allocator backend repair

- Owner: lead; architecture: Astra; contracts/proof sufficiency: Sol.
- [Intent](allocator-model.md); [specification](allocator-model.md).
- Base: `262bb0a94eff2cc7d5655122a954f44d74de96b6` plus accepted dirty `target/open-areas/baseline.json`.
- Slice: `APEIRON-RESOURCE-W2-DESIGN`, twenty-minute architecture bound; supplied isolated worktree exception. Own only these three lifecycle documents; no build lane or commits.
- Status: proposed, `anchor`. Compiler/kernel feasibility is unrun and not inferred from source review.

```text
SCOPE: docs/epics/allocator-model.md; docs/epics/allocator-model.md; docs/epics/allocator-model.md.
REGISTER: planning/refinement; excludes runtime/backend edits and proof/allocator closure.
CORPUS: Growable.Resources.ResourceAdapterSimulates.refused requires committedAfter = state;
        its resourceAdapter_refusal_stutter theorem projects this premise, not an execution proof.
REFERENCE: /home/socol/Workspace/apeiron/target/aeneas-model/lean/Aeneas/Std/Vec.lean
GATE: complete-file identity/path review and git diff --check in the assigned checkout.
      Compiler/kernel feasibility MISSING/unrun; no invented red output.
GRADE: anchor before/after. Existing capacity-reset terms are source evidence only.
```

Process exception: the Arc block was emitted after the initial documentation write; no production edit occurred. Subsequent implementation tasks must emit their Arc before editing.

### Next production-path patch

`APEIRON-RESOURCE-W3-T1`, Sol worker, owns exactly:

1. `proof/patches/aeneas-lean-model.patch` — alter only the patched `Vec.push` and `extend_from_slice` capacity construction; keep element terms, signatures, guards and current callback handling unchanged.
2. `proof/controls/AllocatorCapacity.lean` — probes importing the actually patched `Aeneas.Std.Vec`, not a copied vector definition.
3. `proof/controls/allocator-capacity.sh` — one rejecting runner, currently MISSING.

The runner materializes the installed base plus the repository patch in a disposable isolated tree, verifies the base/patch hashes and actual selected Lean toolchain, and builds `Aeneas.Std.Vec` plus the probe. Do not run current `proof/model-patches.sh` against the shared checkout: it deletes/rebuilds its destination. The runner uses an isolated destination and leaves installed `target/aeneas*` unchanged. Its fixture must not bypass the real patch or omit backend proof checking.

Gate: `bash proof/controls/allocator-capacity.sh --controls`. Required probe statements:

- Under `CapacityInv` and `length + 1 ≤ cap`, actual `Vec.push` appends and returns the same cap; no new effect appears in this within-cap branch.
- Reserve exactly eight byte elements from empty, then push one: final length 1 and cap 8.
- Reserve eight, then extend from a one-byte slice with `CloneU8`: final length 1 and cap 8.
- General successful append results satisfy `CapacityInv`; existing `push_spec` and list-result behavior still build.
- Each probe's axiom cone is checked fail-closed against the Rust bridge's allowed set. Printing axiom names without rejecting extras is insufficient.

Baseline desired probes are predicted red from current source but have not run. The first action is to materialize and run them. A baseline compilation/setup failure is not the named capacity counterexample. Patch-green follows only after the actual failure is observed. For the negative control, revert `push`'s capacity expression to the current exact-new-length expression in the disposable patched backend; require the named reserve→push cap theorem to fail. Restore and pass. Repeat with `extend_from_slice` to bind both changed bodies. This gate fixes a concrete backend defect; it does not prove exact allocator growth or ordinary refusal.

### Finite dependency sequence

Each row uses an isolated task checkout and at most three repository paths. Model-patch edits serialize; their upstream hunk scopes are explicit below. Lead assigns an actual worker/base and one build lane before dispatch. No row is permitted to expand into unowned proof consumers on a failed build.

| Task ID | Owner / exact files | Required result; one gate; control |
|---|---|---|
| `APEIRON-RESOURCE-W3-T1` | Sol; patch + capacity probe + runner listed above | Capacity-preserving actual push/extend; `bash proof/controls/allocator-capacity.sh --controls`; cap-reset mutants red/restore green |
| `APEIRON-RESOURCE-W4-T1` | Sol; `proof/patches/aeneas-lean-model.patch`, `proof/controls/AllocatorCapacity.lean`, `proof/controls/allocator-capacity.sh` | Cover actual resize and equal-length deref-back capacity, all active constructor/backward-update invariants; same runner; reset either reserved capacity red. Stop if callback semantics require more than capacity repair. |
| `APEIRON-RESOURCE-W5-T1` | tooling worker/Sol; `proof/extract/allocator-layout.rs`, `proof/controls/allocator-layout.sh`, `docs/evidence/allocator-layout.md` | MISSING feasibility runner `bash proof/controls/allocator-layout.sh`: extract actual Rust layout queries for generic `T`, byte/word/ZST and admitted table-carrier wrappers with pinned Charon/Aeneas; record concrete layouts and argument plumbing. Wrong size/alignment fixture must reject. Fixtures are feasibility evidence, not a production replacement. |
| `APEIRON-RESOURCE-W6-T1` | Sol; `proof/patches/aeneas-lean-model.patch`, `proof/controls/AllocatorEffect.lean`, `proof/controls/allocator-effect.sh` | Only after layout calling convention succeeds: returning request/reply skeleton, `AllocationSpec` ret/effect/bind/embedding laws, legacy `spec` unchanged. Gate MISSING: `bash proof/controls/allocator-effect.sh --controls`; an existential-reply WP mutant must fail a two-reply counterexample. Atomic upstream scope: `Primitives.lean`, `WP.lean`, `Vec.lean`; remove impossible-response elimination, preserve continuation behavior. |
| `APEIRON-RESOURCE-W7-T1` | Sol; same three paths as previous row | Connect actual `Vec.try_reserve{,_exact}` to the admitted effect, valid-layout/growth/no-growth/ZST branches; same runner; forced-success or overflow-only-refusal mutant breaks same-input valid-byte theorem. No Vec API signature change without successful layout-lowering evidence. |
| `APEIRON-RESOURCE-W8-T1` | Sol; same three paths | Cover capacity-observing constructors/clone/growth and every actual generated vector call family; same runner; ZST cap and excess-capacity mutants reject. Unsupported constructor domain remains an explicit blocker, never assumed away. |
| `APEIRON-RESOURCE-W9-T1` | extraction owner; `proof/ApeironWord.lean`, `proof/controls/allocator-effect.sh` | Fresh actual-source extraction on repaired backend; `bash proof/controls/allocator-effect.sh --extraction-controls`; real reserve-effect erasure must fail a named generated-wrapper branch theorem. Existing wrapper source remains unchanged. |
| `APEIRON-RESOURCE-W10-T1` | Sol; `docs/epics/allocator-model.md`, `docs/kernel-boundary.md`, `docs/epics/storage-refinement.md` | Adjudicate exact-state restoration versus committed observations from real-path refusal fixtures. Whole-file contract review is the gate. No theorem/production change yet; reject a proposed relation permitting changed public reads, payloads, staging or routing. |
| `APEIRON-RESOURCE-W11-T1` | Sol; `proof/corpus/Resource/AdapterExecution.lean`, `proof/corpus/lakefile.lean`, `proof/corpus/AxiomAudit.lean` | Actual generated reserve/observe/materialize/prepare execution proofs and source-linked callback ordering. Bounded `lake build Resource.AdapterExecution AxiomAudit`; an early real hold or erased refusal must break its named theorem. Gate statement/negative harness must be made concrete from the accepted observation decision before dispatch. |
| `APEIRON-RESOURCE-W12-T1` | Sol; `proof/corpus/Growable/Resources.lean`, `proof/corpus/Resource/AdapterExecution.lean`, `proof/corpus/AxiomAudit.lean` | Construct the existing strict simulation if physical equality holds; otherwise separately named approved observation contract and its exact downstream migration. Bounded `lake build Resource.AdapterExecution AxiomAudit`; allow one changed public observation in a throwaway and require refusal preservation to fail. |

The last two rows are conditional interfaces, not dispatch-ready proof tasks: actual state/trace theorem shapes and all consumers depend on Sol's refusal decision. If observation migration needs additional consumer edits, split them into one-owner rows after enumerating exact statements; do not change `ResourceAdapterSimulates` in place and count weaker premises as closure. The finite prerequisite chain is capacity → layout translation → effect/WP → actual reserve outcomes → extraction → refusal adjudication → execution/trace → simulation → public composition. W7 search/commit composition retains its separate reopening condition.

The preceding [resource continuation](resource-security-continuation.md) already specifies the real late-edge injection regression and abstract joint witness. Reuse those tasks; do not duplicate their relation, synthetic witness or test work. The physical-versus-observable decision requires first-use and initialized-locator evidence with hold/clone/merge counters and capacities recorded separately. A source-linked symbolic no-call theorem may supply counters only after its execution connection is checked.

### Build and compatibility boundary

All Lean commands use the global lane lock `/tmp/opencode/apeiron-lane.lock`, `LEAN_NUM_THREADS=1`, `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`, and `timeout 900s`. Do not raise reductions, heartbeats or recursion depth. A cap/timeout failure is a feasibility failure, not the expected mutation red. Model patch/cache materialization is serialized with extraction and both Lake builds.

Every effect step builds the three directly changed backend modules and their backend package, then the affected local package at the barrier. Retain the source inventory in the spec; 149/143/41 reverse-import counts are not selective build exclusions. Existing pure WP consumers retain their contracts; if a proof fails because it actually relies on the unmodeled reserve result, report the exact equation and give it a separate ≤3-file task. No automatic premise addition or downgrade.

At the implementation barrier: Rust tests/clippy/fmt, `proof/regen.sh --check`, bounded `lake build` in `proof/` and `proof/corpus/`, closed-cone audit and named controls. Materialized backend capacity repair is not fresh extraction acceptance. Missing type-layout transport, interpreter/WP kernel checks, allocator realization, trace construction or refusal simulation each keeps its own row open.

### Identity and design acceptance

| Artifact read | SHA-256 |
|---|---|
| `proof/patches/aeneas-lean-model.patch` | `210e56e1ed54bfd4b91e12a102e7dceb17c429b7f14903a176c27b1c0b23b191` |
| `proof/model-patches.sh` | `22e5c507d6d9670d473c0daab5a79346b5a2a9a88ca6540d1f7baf71871899d1` |
| Active backend `Aeneas/Std/Vec.lean` | `5ff3818d75d3f831959c0e02baaae46ab281934d922c4691ad0a38421eaa4ba0` |
| Active `Aeneas/Std/Primitives.lean` | `702946866c83fa0bb8b9d2a248c14164d38aa15e1eebd67caf4f39b0076d0ff8` |
| Active `Aeneas/Std/WP.lean` | `28d78a34e3033cb1856a6a1029f3c187abe14ef597ffcc6865910306c250928f` |
| Local Rust `alloc/src/raw_vec/mod.rs` | `39ae2ccc61fd3bad236978201cda2ad97599403eb103de7eafdf6d7f6dd8b6f3` |
| Local Rust `core/src/alloc/layout.rs` | `19119ede8a60d1a989f907118f9e7f36e24434b8e931875f66a379b856784124` |

`proof/patches/PINS` pins the translator patches, not the complete Lean-model source population. The future runner must record and verify base backend hashes too; patch applicability alone does not identify its source. No pin changes occur here.

Design gate: complete-file trailing-whitespace/final-newline checks, existing referenced-path review, baseline digest preservation and `git diff --check`. Missing future files/runners are explicitly proposed. This slice runs no Rust/Lean build, benchmark, allocator failure or mutation control. Only the lead integrates/removes the task worktree; no commit or push is performed.
