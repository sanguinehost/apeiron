# Apeiron Local Boundary: Representation And Trust

> **Correction (repository state).** The REG-1 comparison harness (`examples/reg1`, `examples/smhasher`, `examples/fold`, `examples/support`) and its comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash) were removed from the tree in `c9a2d0c`. Every `cargo run ... --example reg1` (and any other removed-harness example) below is a historical command that cannot run on this tree; the live Rust gates are `cargo test`/`cargo clippy`/`cargo fmt` from `AGENTS.md` §6, and the live performance entry point is whatever the current harness names. Registered-incumbent and acceptance-registry claims below describe the removed harness.

Kani is retired from all current gates; the
[retirement contract](epics/kani-retirement.md) supersedes earlier Kani requirements.
Historical bounded-check receipts are not current coverage. Rust regression tests and semantic
controls remain required at their stated scope; universal source refinement remains open.

This document owns Apeiron's local representation and trust boundary. The programme thesis,
ownership and outcomes are in the [programme direction](direction.md); this file does not restate
them. Local authority: the executing growable flat/spill `Table` representation, the historical
fixed-shard `Rep` relation, the counting-face representation boundary, the translation/trust
inventory and the kernel scope below are owned here. The programme authority is
`docs/direction.md`; the execution sequence is in the [operator handoff](plan.md).

## Canonical object and executing root

<!-- APEIRON-STRATEGY-DECISION-T3 BEGIN -->
### Read specialization and scatter boundary

Owner Astra, `APEIRON-STRATEGY-DECISION-T3`; architectural grade `anchor`. Current performance and
the unresolved attainability verdict are in [direction](direction.md#serial-read-specialization-and-stable-parallel-scatter).
This section supersedes older current-status/dispatch text, not any invariant, gate or trust statement.
The retirement receipt's generated `92c77151…` extraction/check (1708 jobs) and corpus build
(3482 jobs) preserve the recorded executable content after annotation-only changes; they do not
discharge the universal read/search links, `SearchDecisionCorrect`, the allocator adapter,
`W7-SearchCommitBoundary` or the four spill-filter obligations below. The read source
now uses checked summary reservation/receipts and `publish_live`'s payload → summary → tag order;
T2's earlier source-discrepancy list describes its old snapshot, not acceptance of all resource paths.

**Source-level cost facts, not cycle bounds.** In `src/table.rs`, `get_at` calls
`search_decision_at` → `search_decision` → `apply_get`. Flat match/empty returns without reading
`shards`; occupied non-match must fall back. `spill_locate` selects one region. With valid geometry,
`g = slots.len()/N`, a summary-positive absent query executes exactly `7 + g` group iterations,
including repeated prefix groups; a negative summary still searches for the first vacancy. The
nonempty summary width is recomputed by `summary_trailing_zeros`. These operations explain targets
for removal, not attribution of all elapsed time. No measured cache-miss count, instruction count,
or machine lower bound establishing impossibility is supplied. 256 shards are a storage/build
choice, not 256 serial metadata reads per lookup.

**Read-only interface decision, pending Sol contract.** Preserve `SearchDecision` and its five valid
insertion cases, exact first fit, growth and all existing roots. A proposed read classifier may return
only found-flat/found-spill/absent; absence is not an invented `Vacant` or `Exhausted` insertion
decision. Its absent result needs one of: admitted empty flat under composite `DirectCover`, empty
routed spill after flat exclusion, sound summary-negative after flat exclusion, or complete finite
match coverage. Summary positives/fingerprint collisions establish no presence; full `Sigma` and
carrier identity still decide it. Keep the mandatory late-hit sweep on summary positives. A
first-empty shortcut on those positives remains unauthorized.

The optimized read must access the same payloads through shared production helpers, not leave the
old extracted apply as a dead proof-only path. Sol must pin generated read/apply correspondence,
both-carrier joint reachable witnesses and the public `get_at` link before implementation; added
roots and local read refinement require separate extraction/proof tasks. Inlining alone keeps the
existing interface and needs regression checking, not a new theorem premise. No assumed summary
postcondition, reduced domain or changed evidence grade is allowed. Cached summary width, a new
control directory or altered flat geometry adds representation/constructor/preservation obligations
and is not part of the size-only or inline slice.

**Scatter choice, pending lifecycle/Sol verdict.** Keep `Parts { buf, ends }`, deterministic routing,
one contiguous final shard run, and the production `is_intact`/schedule checks. Add a parallel
constructor for the existing parallel entry points; keep `Parts::of` and the sequential consumers.
Workers partition contiguous input chunks into private, pre-reserved index buffers; offsets remain
global input indexes. Concatenate each shard's chunk runs in input-chunk order, never completion
order, into disjoint initialized slices of the final buffer. A safe ownership partition of those
slices is required; no raw concurrent writes or `MaybeUninit`. Join every task before propagating
panic. Cap workers by input size/crossover and the registered worker budget, including scatter;
do not silently introduce nested pools or increase the timed arm's concurrency.

This preserves the constructor's output, not merely a commutative sum. Canonical
`Apeiron.Table.build_eq_of_key_streams_eq` requires equal per-key value sequences for arbitrary
merge; it proves neither the scatter implementation nor thread realization. Reserve/check new
scratch and destination storage before holds/clones/merges; checked failure must preserve the
existing resource/refusal contract. Freeze the `Home` stability/thread-safety assumptions and exact
constructor-output relation in Sol's verdict. New scratch storage/copy cost must be measured.
The output-construction work is O(n + 256p), auxiliary O(n + 256p) words for n indexes and p chunks;
chunk scatter and final concatenation each visit each index once. No p-fold speedup follows.

The partition validator in `src/table/partition.rs` is tested by
`cargo test --locked --lib table::partition::tests::partition_integrity_`, not an extracted constructor proof. Its removal, debug-only
replacement or producer-set flag is not authorized. Borrowed grouped readback or worker-local
`sum_parallel` reduction is a separate effect/order/overflow task; the ordered REG-1 digest and
query callback order cannot be replaced with a sum. Payload initialization stays materialized after
reserve-all. Existing allocator, executor/spawn/join/unwind, arbitrary-callback, codegen, ABI,
GPU/SPIR-V and unsafe-container exclusions remain; Rust/canonical closed cone and Slang empty cone
remain separate. None of these exclusions waives eventual source-linked sequential refinement.
<!-- APEIRON-STRATEGY-DECISION-T3 END -->

Current scheduling and evidence qualification: [W7-SearchCommitBoundary](#w7-searchcommitboundary)
freezes pending composition, not its required result. Older extraction/status paragraphs are
revision-bound checkpoints; in particular, the fixed-shard constructor/insertion claims in the
trust inventory do not establish growable end-to-end refinement. No gate or exclusion is retired.

The canonical object is `Apeiron.Table.HMap`, `Apeiron.Table.get`, `Apeiron.Table.insertWith` and
`Apeiron.Table.build`, together with `Apeiron.Trichotomy.AnyRing.RingCell`,
`Apeiron.Trichotomy.AnyRing.ringMul` and `Apeiron.Trichotomy.AnyRing.ringNormForm`. The extracted
`Cell<i64>` word bodies are `word_mul_denote` and `word_norm_denote`.

The basis is `R[x]/(x^2 - sigma)`: `Cell<R>` with `Sigma` as a value in every executing signature,
`Coord` addressing correct per face, and the executing growable `Table` — a flat certified-coordinate
resident area plus grouped spill regions, keyed by `(Sigma, K)` with `Stored`/`Regen` carriers. The
historical fixed-shard `Table<K, V, H, N>` (one home shard, `ShardFull` refusal, `N = 0` refusal, no
probing) and its `Rep` proofs are revision-bound historical evidence, not proofs of this layout.
Every executing face refines the canonical `Apeiron.Table.HMap` object; a face that names it only by
path, namespace or import is not rooted.

## Layer separation

The following layers are distinct; none silently stands for another:

- Canonical semantics: `HMap`, `get`, `insertWith`, `build`.
- Face-specific key interpretation: elliptic exact packing, parabolic carried key, hyperbolic
  scramble/discard, each addressed by `Coord` per face.
- Historical fixed-shard admission: `ShardFull` and `N = 0` refusal, one home shard, no probing.
- The source-checker's memo store.
- Song's represented storage.

A result at one layer does not transfer to another without its own refinement.

## O(1) at fixed N

The O(1) qualifier is the parameter boundary: fixed N, bounded callbacks and nonconstant key length
bound the table. Reading a whole source remains at least linear in its bytes at fixed N, so
constant-time table operations at fixed N do not imply constant-time whole-source checking or
constant-size proof terms.

## Fold crossover

`HashTrinity/Core.lean` and `Song/Table.lean` carry the predecessor `insertWith`, `build` and
ordered build fold; `hashtrinity/src/algebra.rs` supplies the predecessor `Coord`, `CoordInv`,
`Stored` and `Regen`, folded multiplication and the four merges. That lineage crosses into the
canonical `Apeiron.Table` and `Apeiron.Trichotomy.AnyRing.ringMul`. An audited cross-revision
equivalence artifact is missing, so no byte-hash-equals-`ringMul` claim and no arbitrary-merge
classification is made.

## The historical fixed-shard representation

The executing **map-face** representation is the growable flat/spill `Table` in `src/table.rs`: one
`Table` with two concrete storage faces (`Stored<K>`, `Regen<C>`), a flat certified-coordinate
resident area and grouped spill regions, keyed by `(Sigma, K)`. "One `Table`" is the whole statement
for the generic key/value face and is superseded **for the counting face only** by the
[counting-face boundary](#counting-face-representation-boundary-apeiron-reorient), which retains this
`Table` unchanged and adds one single-region container refining the same canonical object. Its
boundary, theorem set and extraction status are in
the [extraction-shaped representation boundary](#extraction-shaped-representation-boundary-astra-apeiron-extract-rep)
below. The fixed-shard paragraphs here describe the earlier extracted layout; its proofs remain
revision-bound historical evidence, not proofs of the executing layout.

The historical type `apeiron::Table<K, V, H, const N: usize = 16>` was defined in `src/table.rs`; no
aliases or wrapper containers. `src/map.rs`, `src/safe_table.rs` and `src/set.rs` are deleted,
including `Map`,
`Words`, `Exposed`, `ExposedWords`, `SafeTable`, `TableFull`, `Set`, `WordSet`, `ExposedSet` and the
map entry/iterator/storage-policy types. The former `CountTable`, regions, scanner and worker code
are removed from `src/table.rs`, not retained under a second name. Algebra modules remain. The
counting face is a distinct production type, not an alias, a wrapper or this `Table` renamed, and the
removed `CountTable` is not its predecessor: it is specified by the
[counting-container contract](epics/storage-refinement.md#coordinate-keyed-counting-container-contract)
and bounded by the [counting-face boundary](#counting-face-representation-boundary-apeiron-reorient).

Storage is an inline `[Option<[Option<(K, V)>; N]>; 256]`: 256 fixed shards, each an optional
length-`N` array. An outer `None` denotes an untouched empty shard; it does not defer a heap
allocation. There is no `Vec` in the table or its build body. Scalar loops and full-key comparisons
are the only search path.

## Public API and contract

- `new(d, home: H) -> Self`
- `get(&self, &K) -> &V`
- `insert_with(&mut self, f: &F, k, v) -> Result<(), StorageError>`
- `build(f: &F, d, kvs: &[(K, V)], home: H) -> Result<Self, StorageError>`

`H: Home<K>`, `F: Merge<V>`, writes use `K: Eq`, `V: Clone`, and build additionally uses
`K: Clone`. Blanket function/closure adapters use the same bodies; concrete `WordHome` and
`WordMerge` avoid arrow-valued Rust fields in the extracted word instance. The latter is wrapping
addition, not an assumption that all merges commute.

Semantics and the required representation invariant:

- `HMap K V := K → V` is the shared canonical semantic object, total on all keys.
- `D(t)(k)` is the unique matching override in shard `home(k) mod 256`, or the stored default.
- `Rep(t, h)` is the physical invariant: the fixed outer array, optional length-`N` shard arrays,
  globally unique keys, every entry in its home shard, and `∀ k, D(t)(k) = h(k)` through key/value
  denotations when Rust and Lean types differ. `TableRefinement.PhysicalInv` instantiates it for the
  generated word table; `ReadRep` is its extensional component.
- The obligations are: constructors establish `Rep(t, fun _ => d)`; reads denote
  `Apeiron.Table.get`; successful writes establish `Rep(t', Apeiron.Table.insertWith f k v h)`. The
  concrete `I64` word instance proves them at every fixed shard capacity `N`:
  `Apeiron.TableRefinement.emptyWordTableN_physicalInv` for the constructor and
  `Apeiron.TableRefinement.physicalInv_insert_with_all` for insertion, including `N = 0`. The
  arbitrary-callback `Table<K,V,H,N>` obligations are proved under the explicit `CallbackLawsG`
  contracts. `insert_with_callback_admitted` preserves `PhysicalInvG`, and
  `table_build_physicalInv_of_buildAdmissibleG` supplies the successful-build postcondition.
- Even the first write computes `f v d`, not `v`. `build` starts from `d` and applies exactly those
  writes to the finite stream.

This realizes the constant-default fragment with at most `N` stored keys per shard. It does not
enumerate all `K → V`. All keys remain readable, including when storage is full or `N = 0`.

The home callback must be total, stable and respect key equality. Generic `Eq`, `Clone` and merge
callbacks need explicit denotation and termination contracts; Rust trait names alone prove no laws.

## Admission, refusal and partiality

- `ShardFull`: insertion refuses exactly when the key is absent and its home shard has no vacant
  slot. The refusal changes no table field and invokes neither clone nor merge; existing keys still
  update in a full shard.
- `N = 0` admits construction and reads and refuses every write.
- A build is admitted exactly when each shard receives at most `N` distinct keys, including writes
  whose merged value equals the default. It returns no partial table on refusal.
- Prove success on this explicit domain, exact refusal and preservation with joint witnesses; a
  conditional-on-success theorem alone is insufficient. This changes the physical admission
  contract, not the total canonical `insertWith`/`build` definitions.
- Inline object/stack size grows with `N`; stack availability is not proved. Borrowed input avoids
  table-owned stream allocation but adds key/value cloning. Generic `Home`, equality, clone, merge
  and drop may allocate, panic or diverge; their pure/total denotations require separate instance
  proofs.

## Shared canonical map vs physical representation

The canonical corpus defines `HMap K V := K → V` and the total `get`/`insertWith`/`build`. The
physical representation described here is the fixed-capacity sharded array with the private invariant
`Rep`; for the counting face a second physical representation of the same canonical object carries
its own boundary and `Rep`-free admission contract under the
[counting-face boundary](#counting-face-representation-boundary-apeiron-reorient). These
are distinct. `HMap` is the shared semantic object; `Rep` is the physical invariant under which a
particular table state denotes it. `Rep` is not the public contract and is not exposed; the public
contract is the four methods and their refusal semantics.

`TableRefinement.PhysicalInv` is the general word-instance invariant, not scoped to `N = 16`.
`Apeiron.TableRefinement.emptyWordTableN_physicalInv` proves the real constructor establishes it at
every `N`, `Apeiron.TableRefinement.physicalInv_insert_with_all` proves the real insertion body
preserves it at every `N`, including `N = 0`, and
`Apeiron.TableRefinement.table_build_physicalInv_of_buildAdmissibleN` proves the real positive-`N`
build returns a `PhysicalInv` table refining canonical `build`; the zero-capacity build is
characterized at `Apeiron.TableRefinement.wordBuildN_zero_ok_iff_empty`. The generic callback layer
proves constructor, insertion and admitted-build preservation under `CallbackLawsG`, including the
exact zero-capacity build split.

## Current evidence at actual scope

- The generated word-table refinement covers arbitrary callback instances satisfying
  `CallbackLawsG`. For the concrete `I64` word instance the constructor establishes `PhysicalInv` at every
  `N`; `get` reads the represented total map; `insert_with` preserves `PhysicalInv` and canonical
  `insertWith` on success and returns the identical representative on `ShardFull`, with exact refusal
  at `N = 0`. For positive `N`, `BuildAdmissibleN` exactly characterizes build success versus
  `ShardFull`, and success establishes canonical ordered `ReadRep` and `PhysicalInv`.
- Generic headlines include `CallbackLawsG`, `insert_with_callback_admitted`,
  `table_build_physicalInv_of_buildAdmissibleG`, `table_build_zero_ok_iff_empty`, and
  `table_build_physicalInv_of_distinctRoutedKeysPerShardG_all`.
- Headlines: `emptyWordTableN_physicalInv`, `physicalInv_insert_with_all`, `insert_with_zero_exact`,
  `table_build_physicalInv_of_buildAdmissibleN`, `wordBuildN_zero_ok_iff_empty`,
  `table_build_readRepN_of_buildAdmissibleN` and `table_build_refusesN_iff_not_buildAdmissibleN`,
  each `#audit`ed in `proof/corpus/AxiomAudit.lean`. The older named witnesses in
  `proof/corpus/TableRefinement.lean` remain at their concrete scope.
- The real `new`, `get`, `insert_with` and `build` bodies occur in `proof/ApeironWord.lean`, and
  `proof/regen.sh --check` accepts the real generic bodies and concrete word roots. Extraction is not
  a proof of termination, `Rep`, or equality to `HMap`.
- The manifest resolves corpus Git `880aa57e47e782ed7a9d9ee1974cfc87360b446b` and Mathlib
  `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. A source-proof run log is historical evidence, not
  fresh acceptance.

## Rust and Slang correspondence

Rust and Slang have different physical layouts. Different layouts can correspond, but only under a
matched admission relation: the same canonical definition, matched domains, and explicit
representation, state/effect, refusal and resource obligations. A worked example, a differential
test, identical names or identical layouts are not the correspondence proof. The cross-repo
primitive correspondence is a required programme outcome and is not established today; see the
[programme direction](direction.md#primitive-correspondence).

The Slang source-refinement chain carries a stricter evidence rule than the Rust bridge: every
registered declaration, instance, witness and headline audits with an **empty** axiom set
(`proof/corpus/Slang/Audit/Empty.lean`, `tools/slang/zero_axioms.py`). `propext`, `Classical.choice`,
`Quot.sound`, `noncomputable` and quotient-based representations such as `Finset` and `ZMod` are
rejected there. The Rust `Cell`/`Table` bridge and the canonical corpus retain the
`{propext, Classical.choice, Quot.sound}` cone. Historical closed-cone Stream 0 evidence is
historical, not current acceptance.

## Choice and trust cost (anchor)

> **Prior selection.** This section records the historical fixed-shard resource/trust choice. It is
> revision-bound historical evidence; the growable flat/spill `Table` executes, and the growable
> refusal contract is ratified in the revised boundary below (option (b), checked reservation,
> transactional).

Fixed shards remove the need to model `Vec::capacity`, `Vec::try_reserve_exact` and
`TryReserveError`. A growable model would need capacity/length bounds, reservation success and
refusal, content preservation on both outcomes, and a correspondence to the actual allocator
primitive. A Lean model definition alone would not prove that correspondence; assuming it enlarges
external trust even if it adds no Lean axiom. Merely deleting reservation and using `Vec::push`
hides allocation failure and is rejected. Caller-provided growable arenas introduce another
ownership/resource interface without eliminating its proof obligation. Fixed inline storage is the
smallest selected interface; reduced admission and larger stack objects are its explicit cost.

The SSE2/SWAR dispatch, raw `MaybeUninit` lifecycle, and scoped worker scheduling and joins are
removed. No intrinsic or heap-simulation axiom replaces their missing proofs. A later optimization
must refine this same type and executing path before acceptance. Pure shard independence can be
proved, but that does not prove thread memory semantics, panic propagation or liveness.

## Revised production boundary (Astra decision)

Owner Astra; recorded by `APEIRON-PERF-RESTORE-W4-T1`; documentation only, no source, proof or
measurement claim. This section replaces the fixed-shard production decomposition as the selected
migration boundary. The fixed-shard `Table<K,V,H,N>` above is revision-bound historical evidence;
the growable flat/spill `Table` executes in `src/table.rs`. The growable extraction, refinement and
measurement integration wave remains open on an integrated source/tool identity.

**Selected representation.** One growable, key-partitioned `Table`, generic over `V` and merge `f`,
with load-sized home regions, tag-line scan and window-descent probe, carrying `Sigma` in every
executing signature. `N` is the lanes examined by one tag-group primitive, not keys per home, so no
positive `N` imposes a `256·N` ceiling. The representation refines canonical
`Apeiron.Table.get`/`insertWith`/`build`; the canonical schedule licence is
`Apeiron.Table.build_eq_of_key_streams_eq`. The universal O(1)-at-fixed-`N` claim does not transfer:
cache residency is workload-dependent and a fixed probe-group width does not establish constant
time. The fixed-`N` qualifier is not renamed onto the growable representation.

**Staging.** The representation change is a multi-task serialized integration wave, not a three-file
edit. Each task owns at most three files and keeps `cargo test`, `cargo clippy` and `cargo fmt`
green plus its focused gate; the wave closes only when extraction, refinement, the REG-1
harness and the shared gates pass on the integrated identity. An intermediate proof failure is not
an accepted wave result. Reason: changing the representation simultaneously changes admission
(`src/table.rs` positive-`N` `ShardFull`), the extracted array representation
(`proof/ApeironWord.lean`), the bounded-admission refinements, Rust regression tests and
the REG-1 workloads/oracles.

**Unextracted execution boundary.** Threading (`std::thread::scope`) sits behind an unextracted
orchestration boundary; the extracted roots are the sequential partition, worker and assembly
bodies. The schedule obligation composes the implementation-specific per-key subsequence equation
with `Apeiron.Table.build_eq_of_key_streams_eq`
(`proof/corpus/.lake/packages/apeiron/proof/Apeiron/Table/Basic.lean:167-174`, package revision
`880aa57e47e782ed7a9d9ee1974cfc87360b446b`). This covers sequential worker semantics and schedule
preservation only. It does not cover Rust thread memory semantics, exclusive-borrow realization,
spawn/join failure, panic propagation, liveness, allocator behaviour or rustc/LLVM codegen. Those
realization obligations remain open and are not waived by this statement.

**Identity theorem placement (lead-ratified).** Reusable odd-multiply injectivity on `ZMod (2^64)` (generalized to
`2^w` if practical) belongs in the canonical corpus at
`../sanguine-apeiron/proof/Apeiron/Table/WordIdentity.lean`, registered in `Audit.lean`. The Rust
refinement belongs in `proof/corpus/TableRefinement/CoordIdentity.lean`, unfolding the generated
`coord.spread` low-word truncation and instantiating at `PHI`. Pin-transport rule: the bridge pins
canonical `880aa57e47e782ed7a9d9ee1974cfc87360b446b` (the revision carrying
`Table/WordIdentity.lean`); no dirty-sibling import is admitted, and a separately gated
`lakefile.lean` / `lake-manifest.json` / `proof/README.md` update is required before the import. This chain uses the Rust/canonical `{propext, Classical.choice, Quot.sound}` cone
and must not enter `Slang.*`, whose audit requires the empty cone.

### Resource/refusal contract for the growable table

**Lead ratification (`APEIRON-PERF-RESTORE-W4-T2`): option (b), checked reservation/allocation
failure, transactional. RATIFIED.** `insert`/`build` return `Result`; on `Err` the table is unchanged
and no partial table is returned; `ShardFull` and positive-`N` `BuildAdmissibleN` are retired from the
growable production contract, and the fixed-shard Lean modules remain revision-bound historical
evidence. The prior fixed-shard boundary above rejects hiding allocation failure through `Vec::push`;
the growable selected representation grows and can fail allocation.

- (a) retain a bounded `ShardFull` mode as a separate admitted domain: refinement-sound only under an
  explicit logical capacity budget, but a bounded growable layout can still fail its initial
  allocation, so `ShardFull` cannot encode allocator failure, and an inline fixed-shard mode beside
  the growable one is a forbidden second production representation. Not selected.
- (b) checked reservation/allocation-failure result: **selected; lead-ratified**. `insert`/`build` return `Result`; on
  `Err` the existing table is unchanged and no partial table is returned. Required ordering: checked
  layout arithmetic, then reserve all storage for the atomic operation, then invoke merge/clone and
  mutate committed state. Variants `SizeOverflow`/`AllocationFailed` (one opaque `ReservationFailed`
  is acceptable if stable Rust cannot separate the `TryReserveError` kinds, with overflow separated by
  checked arithmetic). Canonical `Apeiron.Table` stays total; resource failure is an implementation
  stutter, not a canonical object.
- (c) caller arena or pre-reservation token: refinement-sound but adds a public ownership/resource
  interface without improving canonical refinement. Not selected.

Refinement shape: success gives `Rep(t', insertWith f k v h)`; failure leaves `Rep(t, h)` with the
canonical successor defined but unrealized. Build success gives `Rep(t, build f d kvs)`; build failure
returns no table. `ShardFull`, positive-`N` `BuildAdmissibleN` and the `ShardFull ↔ ¬BuildAdmissibleN`
headlines are retired from the growable production contract; the fixed-shard Lean modules remain
revision-bound historical evidence. `src/proofs.rs:1147-1154` (`table_refusal_leaves_the_table_unchanged`)
is replaced by allocation/overflow refusal-preservation harnesses. The OOM rule is revised from
"growable rejected" to: unchecked `Vec::push` remains rejected; `push` is permitted only after
proved/checked reservation.

Axiom placement: the abstract relation and transactional preservation close in
`{propext, Classical.choice, Quot.sound}`. No checked Aeneas model for `Vec::try_reserve`/
`TryReserveError` exists in the tree, so actual allocator success/failure correspondence is an
unextracted trust boundary with a simulation statement; it must not enter the `Slang.*` empty-cone
chain. `BuildPre` carries `checkedSizing = Ok plan ∧ reserveAll plan = Ok resources ∧ every worker
terminates normally`; semantic premises (callback laws, identity injectivity, first-fit/search
correctness, per-key subsequence preservation) remain separate. Provable: checked arithmetic, bounds,
sequential loop termination, prefix denotation, worker-body no-panic under total callbacks.
Trusted/open: allocator realization, Rust spawn/join/unwind, arbitrary callback panic freedom.

Residual: exact error variants and transaction point; reservation feasibility/simulation boundary;
replace fixed-shard extraction/refinement artifacts; deterministic overflow and injected
reservation-failure controls; update REG-1 refusal/resource rows to test
`SizeOverflow`/`AllocationFailed`.

## Reworked growable representation (Astra decision, first cut rejected)

Owner Astra; recorded by the rework contract; documentation only, no source, proof or measurement
claim. This section supersedes the **Selected representation** and probe wording above where they
differ; the fixed-shard representation is revision-bound historical evidence and the growable
flat/spill `Table` executes in `src/table.rs`; its integration wave remains open on an integrated
source/tool identity. The contract itself is
in the [spec](epics/storage-refinement.md#revised-production-contract-reworked-growable-key-partitioned-table);
the dispatch order is in the [plan](epics/storage-refinement.md#rework-dispatch-order).

**First cut rejected.** The first growable `Table` at `src/table.rs` sha256
`c34ddf17ef2ed7a81fd4b3e16447479b55d737af8f5253b99dd397414d9b1dc1` is rejected as the production
result and reworked in place. The bounded fixed-shard `Table<K,V,H,N>` is not restored as production
and the benchmark-only candidate `examples/reg1/candidate.rs` (removed in `c9a2d0c`) is not promoted. One growable
representation is kept for the generic key/value face; the counting face selected by
`APEIRON-REORIENT` is the qualified exception recorded in [the counting-face
boundary](#counting-face-representation-boundary-apeiron-reorient).

**Measured red (supplied).** The strict current run has 126 failures among 160 strata cells. These
are serial-incumbent failures: the `occ16` strata incumbent is serial `hashtrinity::Table` at
`4cc14471ae24f3f204fb0b0e91736ef75c909358`. Serial recovery continues. The spatial packed family
is separate production work: restore partitioned/parallel build and grouped probe before measuring
it against its spatial incumbent. A serial improvement does not establish spatial restoration.
These results are not rerun in this documentation task; G1/REG-1 remains OPEN.

**Confirmed defects.** (a) tag and shard duplicate the placement low bits: `shard_index(home) =
home % 256` and `tag_of(home) = 0x80 | (home & 0x7F)` read the same bits, so the tag carries no
independent occupancy information. (b) a miss scans the whole region. (c) the group walk starts at a
hashed group and steps linearly instead of descending first-fit. (d) growth rounds **slots** to a
power of two, which is wrong for general `N` because `N` must divide the slot count. (e)
`min_len = N.max(1) * 2` and `next_power_of_two` are unchecked. (f) the growth-complexity note
overstates: re-placement is linear in the region, not `O(1)` per key. (g) `examples/reg1/strata.rs`
(removed in `c9a2d0c`) still models `ShardFull`. (h) the performance command reports slow cells correctly but exits 0, so
exit 0 is not the per-cell gate.

**Representation, probe and routing.** One generic `Table` has the optional coordinate-bearing
specialization `TableAddress { place, direct_word }` through `Home::address(sigma,key)`; placement-only
adapters return `None`, while `WordHome` certifies an exact word below `2^32`. Direct lane is the raw
word shifted by eight, not `place >> 8`; full `(Sigma,K)` equality always decides identity.
`coord::Addr` keeps its `id`/`place` meaning without a blanket injectivity claim. The exact laws,
finite search theorem, reachable-state direct-empty certificate and corrected seven-bit window mask
are stated in the
[spec](epics/storage-refinement.md#revised-production-contract-reworked-growable-key-partitioned-table).
The former coverage-before-every-absence wording is superseded: direct-empty may certify absence
under `DirectCover`; otherwise complete finite coverage is mandatory. A recorded prefix vacancy may
skip later prefix windows but not the all-group match sweep.

**Flat-resident boundary (Sol contract).** The final representation contract is the table-level
shard-major `256 × L` flat area beside independently load-sized grouped spill regions, as specified
in [storage-refinement.md](epics/storage-refinement.md#flat-resident-and-grouped-spill-contract-sol-decision).
Flat slots use `{live,sigma,key,value}` with `live` as sole occupancy authority; spill tags are the
sole spill occupancy authority. Checked sizing, budget and density determine flat residency;
density/budget exclusion routes to spill rather than refusing the operation. `Site` distinguishes
`FlatFound`, `FlatVacant`, `SpillFound`, `SpillVacant` and `NeedSpillGrowth`. Extent growth is a
transactional rebuild that reclassifies newly admitted spilled keys, not an append. The composite
invariant has one representative per logical key and strengthens `DirectCover`: every live key with
an admitted certified flat coordinate has an occupied flat slot even if its representative is in
spill.

`KeyRep<K,H>` separates `Stored<K>::Held = K` from sealed exact-codec
`Regen<C>::Held = u64`; both preserve `Sigma` separately. Key dropping requires
`decode(sigma, encode(sigma,k)) = k` on the admitted domain and regeneration through the verified
inverse of `Addr.id`, never `Addr.place`. `direct_word = Some(w)` alone proves no codec law.
Direct-empty absence requires composite `DirectCover`; otherwise finite composite coverage is
mandatory. Historical variable-width descent and current seven-bit descent are distinct algorithms,
so a geometry change requires separate Sol adjudication.

**`Sigma` (mandatory).** The logical and payload key is exactly `(Sigma,K)`; reads, writes and
address selection receive `sigma`; build consumes `&[((Sigma,K),V)]`; the relation is
`Rep t (HMap (Sigma × K) V)`. Distinct `Sigma` values with identical `K` remain distinct through
build, relocation, lookup and update, including when their addresses coincide.

**Required theorem set and premises.** The complete bridge set is
`Growable{Geometry,Directory,Identity,Scan,Search,Relocation,Refinement,Witnesses,Schedule,Batch}`.
Its headlines, premises and canonical roots are normative in the
[Sol contract](epics/storage-refinement.md#flat-resident-and-grouped-spill-contract-sol-decision).
The earlier first-cut rows below remain a dependency summary, not the complete set:

| Module | Premise and required headline | Canonical root |
|---|---|---|
| `GrowableGeometry` | checked sizing; admissible group width; region divisibility, power-of-two group count and every computed index in bounds | physical premise for `HMap` representation |
| `GrowableDirectory` | address totality/stability/equality-respect; directory permutation and inverse; `WordHome` routing equation; certified direct lane and placement-only `None` | key interpretation for `Apeiron.Table.get`/`insertWith` |
| `GrowableScan` | exact `match_tag`/`match_empty`; tag zero iff vacancy; no payload observation through an empty tag | observation premise for `Apeiron.Table.get` |
| `GrowableSearch` | constructor reachability base; `DirectCover`; monotone occupancy; relocation preservation; termination, bounds, `search_complete` and `locate_first_fit` | `get_insertWith_self` and `get_insertWith_ne` at `Basic.lean:87-96` |
| `GrowableRefinement` | `Rep` from the real constructor; relocation preserves `Rep`, uniqueness, tags and `DirectCover`; get success/absence; insert success denotes new-first `insertWith`; `SizeOverflow`/`AllocationFailed` stutter; ordered sequential build denotes `build` | `Apeiron.Table.{HMap,get,insertWith,build}`, `get_build_acc` at `Basic.lean:132-150` |
| `GrowableSchedule` | partition is by full logical key and concatenated worker runs preserve every key's value subsequence; assembly is disjoint and complete | `Apeiron.Table.build_eq_of_key_streams_eq` at `Basic.lean:167-174` |

Constructor, successful insertion, refusal and ordered build require joint reachable witnesses at the
actual Rust types. Relocation requires both the pre-state and post-state invariants and no merge call.
The schedule theorem is unnecessary for the serial ordered loop but mandatory for partitioned or
parallel build. If same-key values are reordered or partial accumulators combined,
`build_eq_of_key_streams_eq` is insufficient; the applicable merge laws and
`build_gauge_invariance` are then required.

These modules build on canonical `Apeiron.Table.phi64_lowWord_injective` and bridge
`Apeiron.TableRefinement.coord_spread_id_injective`. Every headline belongs in
`proof/corpus/AxiomAudit.lean` under `{propext, Classical.choice, Quot.sound}`, never in `Slang.*`.
After real-source extraction and the audited bounded bridge build, each named semantic headline may
carry `proved`. Focused Rust cases may carry `tested`. Until then the new address, `Sigma`, direct
absence, relocation and schedule links remain anchors; the canonical corpus theorems alone remain
proved.

**Verification and trust boundary.** “Formally verified end to end” requires real `src/` bodies
included by `#[path]`, refinement of their generated definitions to the canonical object, bounded
`lake build`, fail-closed `AxiomAudit`, `proof/regen.sh --check` freshness and rejecting mutation
controls. It covers address arithmetic, geometry, safe indexing, explicit-state locate, SWAR,
insertion, relocation and sequential build at the audited theorem domain. It excludes
allocator/global-allocator realization, thread/executor realization and `std::thread::scope`,
rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V and the `unsafe` container. The Slang chain retains
its separate empty cone. Astra's verdict and the Sol contract are `anchor`, not `proved`; REG-1/G1
remains **OPEN**. Bridge pin `880aa57e47e782ed7a9d9ee1974cfc87360b446b`.

## Extraction-shaped representation boundary (Astra, APEIRON-EXTRACT-REP)

Owner Astra; recorded by `APEIRON-EXTRACT-REP-W1-T1`; documentation only, no source, proof, extraction
or measurement claim. This section supersedes the `KeyRep`/`Held` carrier and allocator wording of the
flat-resident contract above where they differ. The fixed-shard `Table` is revision-bound historical
evidence; the growable flat/spill `Table` executes. REG-1/G1 stays **OPEN**.

**One `Table`, two concrete storage faces, one executing kernel (decision (b))** — for the generic
key/value face; the counting face selected by `APEIRON-REORIENT` is the qualified exception in [the
counting-face boundary](#counting-face-representation-boundary-apeiron-reorient). The fifth type
parameter is the stored carrier itself, not a zero-sized policy with an associated `Held`:

```text
Stored<K> { key: K }
Regen<C>  { identity: u64, codec: PhantomData<C> }   // manual Clone, no C: Clone bound
FlatSlot<R, V> { live, sigma, held: R, value }
spill slot = (Sigma, R, V)
```

`FlatSlot<R,V>` contains `R`; spill slots contain `(Sigma,R,V)`; there is no `R::Held` projection in
storage, relocation temporaries or worker results. Public spellings `Table<…,Stored<K>>` and
`Table<…,Regen<I64WordCodec>>` are preserved. Heterogeneity is between compile-time instantiations,
never a runtime enum or heterogeneous vector. `Sigma` is retained separately on both faces;
representation is selected by `R`, not switched by `Sigma`. A `KeyRep` trait may keep
`hold`/`matches`/`key_owned` and add `address_held`, so relocation does not clone arbitrary carried
keys.

**Both carrier denotations are required.** The root candidates specialize the shared core at
`Table<i64,i64,WordHome,16,Stored<i64>>` and
`Table<i64,i64,WordHome,16,Regen<I64WordCodec>>`; they are not yet extracted or refined. Canonical
`HMap (Sigma × K) V` and the per-key-stream mathematics stay generic. Word-instance proofs alone do
not establish arbitrary-`K` refinement. Denotation: `decodeStored(s,Stored(k)) = k`,
`decodeRegen(s,Regen(id)) = C.decode(s,id)`, `logicalKey = (slot.sigma, decodeR(...))`; `RepR`
composites the physical invariant with decoded observations. `GrowableIdentity.keyRep_identity`
establishes faithful carried-key equality/clone, codec round-trip on the joint admitted domain,
`matches` iff decoded logical equality, separate `Sigma`, and address agreement for reconstructed
carriers. `proof/corpus/TableRefinement/CoordIdentity.lean` gives the `spread`/identity injectivity
link (`coord_spread_id_injective`), not the codec inverse theorem.

**Allocator boundary.** `Region::reserve` and `Flat::reserve` call `Vec::try_reserve_exact`, a
reachable unmodeled primitive. It blocks an axiom-free accepted extraction of the complete allocating
public methods.

### Resource-interface boundary after the context-join spike

Owner Sol; recorded by `APEIRON-EXTRACT-REP-W5-T1`; documentation only. The supplied spike is a
historical extraction diagnostic, not current acceptance. Its executable spike gate is MISSING. Aeneas `505b6ca-dirty` / Charon `0.1.254` reports
`Could not match the contexts` at `Table::plan_rebuild` (`src/table.rs:678:16-680:17`). The minimal
reproducer requires a generic borrowed carrier in a loop, an inner live-field guard and a branch on a
computed `Option`/enum classification from the carrier address method in `src/table/keyrep.rs`;
the supplied log reports scalar-guard, straight-line, accumulated-option and cast variants passing. Loop translation and strict-join flags do not remove it.
The historical `PinnedJoinBoundary` classified this as an Aeneas context-join limitation. Current
`proof/ApeironWord.lean` contains `table.Table.plan_rebuild`, so this diagnostic is not a current
extraction blocker for that body. Allocator may-failure and capacity/layout correspondence remain
separate model/refinement obligations; generated syntax does not discharge them.

**Extraction-root decision (`anchor`).** The proof roots are monomorphic wrappers for both
`Stored<i64>` and `Regen<I64WordCodec>` that call the real shared prepared-plan sequential core/apply
operation. Their resource control flow accepts prepared plan/storage/outcome and then enters the
core; it has no loop-carried borrowed generic-carrier live guard followed by a computed-classification
branch. `plan_rebuild`, `reserve_rebuild`, every `try_reserve*` allocation, and the allocating public
`build`/`insert` methods are unextracted. They feed the same production core and are not a reference,
mock or second implementation. The current `proof/ApeironWord.lean`, sha256
`faa257a8720c71ee7a08497621ddecd40495cf3729ee10c56f18119f5778113a`, contains prepared-spill roots
only; it omits composite `Table::locate`, `Site` and `DirectCover`.

**Resource adapter law (`PreparedResources`, `anchor`).** The adapter returns either
`Prepared { plan, storage, outcome }` or `Refused(StorageError)`. `Prepared` certifies checked
arithmetic, exact source/pending classification, bounded destinations, sufficient initialized and
reserved storage, and no further fallible core allocation. `Refused` supplies no executable plan or
storage. `resource_adapter_simulates` requires: prepared core success preserves `RepR` and denotes
the canonical `Apeiron.Table.{get,insertWith,build}` operation; refusal is the
`insert_err_stutters` branch—unchanged committed state and no clone, merge or relocation—or, for
build, no returned table. The refusal observations are supplied grade `tested`; allocation
success/failure realization is not proved.

### Search-interface boundary after the context-join spike

Owner Sol; recorded by `APEIRON-EXTRACT-REP-W11-T2`; documentation only. Law
`PinnedJoinBoundary` also applies to current composite `Table::locate`: its loop carries a borrowed
generic-carrier payload through an inner occupancy guard and computed match/classification branch.
No different whole-body shape is selected on the supplied evidence.

**Decision/apply split (`anchor`).** The unextracted production decision walks the direct candidate,
window prefix and mandatory cyclic sweep and returns
`(Found|Vacant|Exhausted, Flat(i)|Spill(shard,i)|NoIndex)`. Its five well-formed cases map exactly to
`Site::{FlatFound,FlatVacant,SpillFound,SpillVacant,NeedSpillGrowth}`. The rootable straight-line
production apply consumes that classification and the real selected region: a read returns the real
indexed value or default; insertion updates a found value, initializes the first-fit vacant slot, or
uses the prepared growth outcome for exhaustion. The apply remains the executed storage access and
commit path, not a model or parallel implementation.

**Decision law (`SearchDecisionCorrect`, `anchor`).** `Found` holds iff the full `(Sigma,K)` key has
its unique live representative at the returned in-bounds index. `Vacant` holds only when the key is
absent and the returned index is first fit. `Exhausted` holds iff the key is absent and the searched
space is full. Direct-empty absence requires composite `DirectCover`; an occupied direct non-match
continues. A window vacancy may stop the prefix but not the all-group match sweep. Every other absence
requires complete finite coverage. `search_decision_simulates` composes this law with the extracted
apply refinement to canonical total-by-default `Apeiron.Table.get` and to
`Apeiron.Table.get_insertWith_self` / `Apeiron.Table.get_insertWith_ne` for insertion.

Rust tests cover concrete classification and late-hit/first-fit regressions with rejecting
sweep/first-fit mutations. That evidence is `tested`, not a proof of the
generic decision law. `GrowableSearch` refines the apply under `SearchDecisionCorrect`;
`GrowableRelocation`, `GrowableRefinement` and `GrowableWitnesses` carry that premise.
`GrowableDirectory`, `GrowableSchedule` and `GrowableBatch` are unchanged.

`RepR` composes the flat/spill physical invariant with decoded `Stored`/`Regen` carrier observations.
`GrowableSchedule` retains the exact per-key-subsequence premise and composes it with
`Apeiron.Table.build_eq_of_key_streams_eq`
(`proof/corpus/.lake/packages/apeiron/proof/Apeiron/Table/Basic.lean:167-174`). Song's external twin
is `Song/kernel/Web.lean` `preduce_schedule_free`/`preduce_append`, read at
`/home/socol/Workspace/song-ref/lean/Song/kernel/Web.lean` revision `ae688f66`; it supplies no
declaration to the Apeiron cone.

The claim excludes the search decision loop and its correctness law, planner/reserver correctness,
allocator/global-allocator realization, allocation success/failure realization,
executor/`std::thread::scope`, rustc/LLVM, ABI, GPU/SPIR-V and the `unsafe` container. This is not a
model-versus-real gap after source factoring: the apply is the real production region access, and the
decision is a controlled classification with its own Rust test and source-refinement evidence obligations. The Slang
source-refinement empty-cone separation is unchanged. Allocating public methods remain unrooted until
a checked Aeneas primitive/tool model exists. REG-1/G1 remains **OPEN**.

**Last extraction verdict (supplied, not rerun here).** The generated artifact carries the prepared
spill roots and the straight-line apply roots; the probe-order decision (`Table::locate`) stays
unextracted under `PinnedJoinBoundary`, and no proof of that decision or speed is claimed.
`proof/corpus/lakefile.lean` still lists fixed-shard/joint consumers, so a separate archival manifest
is required before a full corpus build.

### Relocation root and initialized-resource interface

Owner Astra; task `APEIRON-EXTRACT-REP-RELOCATION-BOUNDARY-T1`. This is an additive boundary
extension at grade `anchor`, conditional on Sol's relocation contract in
`docs/epics/storage-refinement.md`; no extraction, proof or performance acceptance is recorded.

**Verdict: same architecture, additional root.** Factor one shared per-entry `relocate_entry`
operation out of the executing `Table::apply_prepared_rebuild` and call it from production. Add
monomorphic wrappers for both `Stored<i64>` and `Regen<I64WordCodec>`: one generic operation means
two carrier roots, not a second implementation. Carrier-as-type, separate `Sigma`, flat/spill layout,
canonical `HMap`, search geometry, transactional refusal and executor separation do not change.
No wrapper may restate relocation or become the only caller of an otherwise unused production helper.

The cumulative selected root set retains prepared-shard get/apply, straight-line get/insert apply,
their rooted scan dependencies, identity/word roots and the selected partition/schedule checks in
`proof/regen.sh`; it adds the per-entry relocation wrappers. `plan_rebuild`, `reserve_rebuild`, all
`try_reserve*`/allocator calls, the composite decision loop, allocating public methods and the executor
remain unextracted. Rooting one entry does not root the whole `apply_prepared_rebuild` traversal,
pending-insertion decision or final table commit. These still require a source-linked composition
argument; the complete `GrowableRelocation.rebuild_preserves` obligation is retained, not renamed to
a single-entry update lemma.

**Initializer placement and ordering.** Move the rebuild's `Region::initialize` loop into the
unextracted resource adapter after **all** fallible reservations succeed. Prepared destination
storage arrives initialized, not merely reserved: spill payload lengths match zeroed tag lengths,
flat slots have checked `256 × lanes` length and `live = false`, and every planned indexed write is
in bounds. The rebuild's `Flat::initialize` and initializer `R::hold`/payload/default clones belong
to that same preparation phase; do not claim them proved by the per-entry root. `Region::reserve_empty`
already zeroes tags outside extraction; allocation itself is not newly moved. Initialization uses
the real available held key/default, never a fabricated inhabitant or new unsafe storage model.

Do not initialize each region immediately after reserving it while later reservations can fail:
that violates the unchanged no-clone-on-`Err` rule. Preserve the order checked plan → reserve all →
initialize → relocate without merge → pending merge exactly once → commit. Under the existing
callback laws the core needs no further fallible allocation; arbitrary allocating/panicking clones
are not made safe by reserving vector capacity. Refusal still changes no committed state, invokes no
clone/merge/relocation and returns no partial build. Prepared-resource initialization is an explicit
adapter obligation, not a discharged constructor witness.

**Evidence does not cover:** establishing initialized destination lengths, vacant occupancy,
initializer clone/hold correctness or initialization termination/panic freedom in the adapter;
planner/reserver correctness or realization of `PreparedResources`; search-decision correctness or
the generic `SearchDecisionCorrect` law; allocation/global-allocator success/failure realization;
the whole rebuild traversal and commit merely from the entry root; executor/`std::thread::scope`,
rustc/LLVM, ABI, GPU/SPIR-V or the `unsafe` container. Conditional entry refinement must prove actual
source reads, destination writes, occupancy, separate `Sigma`, decoded identity/value preservation
and no merge under explicit input premises, without assuming that very postcondition. Whole-state
uniqueness, composite `DirectCover`, extent reclassification and pending-operation composition remain
required. The Rust/canonical closed cone and separate Slang empty cone are unchanged.

**Blocker and precedent.** The supplied Flash blocker is corroborated by
`/tmp/opencode/reloc-repro/aeneas.log`: whole-body extraction reports `Could not find: type_var_id`
at the region-initialization loop (`src/table.rs:1055-1058` in the read source).
`/tmp/opencode/reloc-repro/aeneas4.log` reports generation for the scratch entry root, with an
external-definition warning; it is not the strict `proof/regen.sh` gate. The scratch helper still
calls `spill_locate_held`, and its wrapper specializes only `Stored`. Inventory transitive callees
and check both carriers before calling the new root accepted or its entire call graph straight-line.
`proof/corpus/Growable/Search.lean` explicitly separates the decision premise from extracted apply
refinement; it supplies this interface precedent, not a relocation theorem.

Sol owns the exact initialized-resource and relocation premises and their non-vacuous witnesses.
If its contract requires whole-body refinement, entry-only evidence leaves that requirement open;
if it permits initialization before the last fallible reservation or assumes the desired relocation
postcondition, stop for reconciliation. The implementation order is in
[the roadmap](roadmap.md#relocation-root-integration-order). Every earlier gate,
command, theorem requirement and realization obligation is retained; REG-1/G1 remains **OPEN**.

The concurrently supplied Sol section `Relocation entry interface` agrees on both roots, materialized
storage and the split between preserving `HMap` and the pending `insertWith` composition. One proof
interface ambiguity remains: `RebuildPlanCorrect` clause 5 says the rebuilt state restores uniqueness,
occupancy and `DirectCover`, also conclusions of `rebuild_preserves`. It must describe an input
plan/destination certificate, not assume the actual execution post-state. Sol must resolve that
reading and source-bind `relocateAll`/`applyPendingAndCommit` before proof dispatch; Astra does not
adjudicate or edit that premise. This does not change the architectural approval.

### Joint prepared-resource boundary

Owner Astra; task `APEIRON-EXTRACT-REP-RELOCATION-BOUNDARY-T2`; grade `anchor`, docs only.
This specifies the missing resource handoff, not acceptance of `rebuild_preserves` or
`prepared_rebuild_refines_insertWith`. The preceding obligations and gates remain required.

**Verdict: approve the adapter move; extract the handoff, not the allocator.** Keep carrier-as-type,
the shared production relocation/apply bodies, both carrier roots and the decision/apply split.
`src/table.rs:1180-1187` still holds/materializes payloads inside `apply_prepared_rebuild`;
`reserve_rebuild` returns only capacity-reserved staging storage. This does not yet implement the
adapter-owned-initialization clause in `docs/epics/storage-refinement.md:393-412`. Move that phase
after successful reserve-all into the unextracted adapter, including the pending `R::hold`, and pass
its held carrier forward rather than holding it again in apply. Keep `RebuildReserved` and the
existing relocation/commit signatures unchanged; add a separate internal handoff envelope.

**Selected interface (proposed source schema, not existing declarations):**

```text
CapacityBound { required_len: usize, observed_capacity: usize }
ReservationReceipt {
  flat: CapacityBound,
  directory: CapacityBound,
  regions: [ { tags: CapacityBound, payloads: CapacityBound }; 256 ]
}
PreparedOutcome<R> { reservation: ReservationReceipt, pending_held: R }
PreparedResult<R,V> =
    Prepared { plan: RebuildPlan, storage: RebuildReserved<R,V>, outcome: PreparedOutcome<R> }
  | Refused(StorageError)
```

The receipt is internal transaction data, not a public caller-supplied arena/token (resource option
(c)) and not a second storage representation. Its fixed directory avoids adding a fallible receipt
allocation after reserve-all. Its stack/copy cost is unmeasured and must respect existing size and
performance gates. The adapter returns this envelope for the same pre-state and pending
`(sigma,key,value,address)` used by production; no independently assembled plan/storage/outcome is
accepted as a production certificate.

- `required_len` is the checked target length, not `try_reserve_exact`'s additional-count argument.
  Flat target is checked `256 * plan.lanes`; directory target is 256; each spill tag/payload target
  is the checked `slots_for(plan.spill_loads[shard], N)`, or zero for zero load. Byte sizing and all
  planning/scratch reservations remain mandatory, even though the receipt describes destination
  storage only. Complete all fallible planning, sizing and reservations before `hold` or clones.
- Observe actual `Vec::capacity()` in the unextracted adapter after successful reservations and
  before initialization. A shared production scalar helper checks `required_len <= observed_capacity`
  for each bound. Root this helper separately; it proves only the scalar comparison, not the origin
  of either number. A violated bound after successful reservation is an internal invariant failure,
  not a new allocation-refusal case or an allowed late `Err` after cloning.
- Initialize the real flat and region payloads with the available pending held carrier and default;
  do not fabricate an inhabitant. On `Prepared`, `storage.resident.lanes = plan.lanes`, flat length
  equals its target, directory length is 256, and every spill payload/tag length equals its target.
  All flat `live` bits are false and all tags zero. The receipt stays paired with that storage;
  initialization must not replace its vectors or allocate beyond checked capacity. No rooted core
  operation initializes, pushes, reserves or resizes these vectors.
- Add a real production-called, nonallocating publication helper that explicitly matches the
  adapter's completed `Result<(plan,storage,outcome),StorageError>` and returns the envelope above.
  Its Stored/Regen wrappers only specialize and call it. The error arm carries no plan, storage or
  held carrier. The success arm moves the exact joint tuple; it neither runs the planner nor
  certifies `RebuildPlanCorrect`. Production must consume this envelope before relocation, not
  bypass it while the extraction wrapper alone calls the helper.
- `outcome` is resource success plus the pending carrier, **not** the pending `SearchDecision`.
  After relocation, unextracted `locate_rebuilt` still computes that decision and the extracted
  `apply_pending_and_commit` consumes it. The symbolic pending certificate and its connection to
  that decision stay separate Sol proof obligations. A refusal is propagated before initialization,
  clone, merge or relocation, with unchanged committed state and no partial build.

**Capacity is not length.** Generated `table.RebuildReserved` has only `resident`/`regions`
(`proof/ApeironWord.lean:2455-2457`); Aeneas `alloc.vec.Vec` wraps a `Slice`
(`target/aeneas/backends/lean/Aeneas/Std/Vec.lean:28-29`). Do not add a Lean capacity axiom, infer an
actual reservation from a list length, or pretend that `Prepared` construction proves its premises.
The new root exposes explicit scalar observations and initialized storage jointly. Sol can check
non-vacuous input-side predicates/witnesses over those generated values, and prove their transport
through the handoff. Correspondence of those observations to actual allocations and reachability of
the witness through the real adapter remain separate simulation/test obligations. An unconditional
reservation/constructor refinement is still blocked without a checked allocator/tool model.

**Exact extraction perimeter.** `plan_rebuild`, `reserve_rebuild`, `try_reserve*`, capacity reads,
`Flat::initialize`, `Region::initialize`, composite `search_decision`/`locate_rebuilt`, their
`Flat::index` path, `apply_prepared_rebuild` orchestration, allocating public methods and the thread
executor remain unextracted. The new roots are the scalar capacity comparison and joint publication.
Preserve all existing roots, including `relocate_entry`, `relocate_all` and pending commit. Contrary
to the older blanket spill-decision wording, `spill_locate` and `spill_locate_held` already occur in
`proof/ApeironWord.lean` (the latter is a relocation dependency); do not remove them. Their presence
does not prove generic search correctness: the spill entry proof assumes the actual extracted
vacancy equation. “Straight-line apply” does not assert a loop-free transitive call graph.

**Trust consequence and reconciliation.** Resource option (b) remains checked transactional
reservation with `SizeOverflow`/`AllocationFailed`, not unchecked `push` or a public arena. Pending
commit option (b) independently remains the precomputed-decision split. Moving initialization repairs
the recorded ownership mismatch; it does not move initialization into the proof cone. Physical
allocator success/failure, capacity-read fidelity, reserve/content preservation, reservation receipt
attachment, initialization execution/lengths/vacancy/termination and adapter simulation are still
outside the proved claim. Core refinement remains conditional on `PreparedResources`, input-side
`RebuildPlanCorrect`, search/codec and callback laws. No new axiom is authorized. Arbitrary callback
allocation/panic, executor/spawn/join/unwind, rustc/LLVM, ABI, GPU/SPIR-V and the unsafe container
remain excluded; the Rust closed cone and Slang empty-cone separation are unchanged.

**Separate contract and headline impact.** Whole-table `RepR`/`ReservedRepR` (complete live set,
decoded uniqueness, default observations, lengths and composite `DirectCover`) are a separate Sol
contract task, not defined here. The existing single-source relations in `Growable/Relocation.lean`
must not silently acquire that meaning. The current Sol clause 5 is already explicitly input-side;
the older ambiguity note above is historical, not a reason to assume the execution postcondition.
The four `word_{stored,regen}_relocate_entry_{flat,spill}_preserves` headlines, two
`word_{stored,regen}_relocate_all_preserves` headlines and two
`word_{stored,regen}_apply_pending_and_commit_eq_apply_insert` headlines keep their statements,
premises and recorded `proved` scopes. None establishes the adapter witness or whole-table result.
This design leaves their bodies/interfaces intact; regeneration, bounded audit/build and controls
must still revalidate all eight on the new identity. Any required statement change returns to Sol.

Residual blockers: actual handoff and extraction are absent; Sol must reconcile the receipt with
`PreparedResources`, construct joint success/refusal and reachable witnesses, define the whole-table
contract, prove disjoint-write frames and the semantic composition. No fabricated success tuple
discharges adapter reachability. Initialization tests must detect a late-reservation clone, not just
correct final values. Implementation order is in
[the roadmap](roadmap.md#prepared-resource-handoff-integration-order).
REG-1/G1 and all full-barrier requirements remain **OPEN** and unchanged.

### Resource adapter simulation premise

Owner Astra; task `APEIRON-EXTRACT-REP-RESOURCE-BOUNDARY-T3`; grade `anchor`, docs only.
This records the boundary for Sol's concurrent contract/proof task, not its acceptance. Earlier
absence/initialization-placement notes describe their checkpoints: the read `src/table.rs` now calls
`prepare_rebuild` → `prepare_from_plan` → reserve-all → `materialize_rebuild` (including
`observe_receipt`, hold and initialization) → `publish_prepared`; production consumes the envelope.
`proof/corpus/Growable/Resources.lean` defines `PreparedResources_R` and `RebuildPlanCorrect_R`
and proves conditional publication transport. The supplied W3 receipt reports 3472 build jobs and
audits; this documentation task does not rerun or extend that evidence.

**Verdict: `ResourceAdapterSimulates` is a named, unproved trust-boundary premise.** Sol defines it
as an explicit proposition/contract argument, never an `axiom`, `sorry`, admitted theorem or global
instance supplying its truth. A reachable publication witness may be proved **conditional on this
premise**; that proves the implication, not that the real adapter satisfies it. This follows the
`SearchDecisionCorrect` interface pattern in `proof/corpus/Growable/Search.lean`, not a theorem
discharging either law. The existing lowercase `resource_adapter_simulates` whole-operation
obligation remains: it is not discharged or replaced by naming the adapter premise.

| Premise scope / adapter obligation | Evidence still required to discharge it |
|---|---|
| Same pre-state, pending `(sigma,key,value,address)`, checked plan and completed adapter result; publication and production consumption of that exact joint tuple | Source-linked execution relation for `prepare_rebuild`/`prepare_from_plan`, not an existential tuple assembled solely in Lean. Rust borrows the key/address during preparation; the pending value remains unchanged for apply. |
| Checked targets and reservation realization, including planning/scratch and all destination allocations | Checked sizing/planning/reservation correspondence under an explicit allocator/tool model; scalar inequalities alone establish none of these execution facts. |
| Capacity reads and correspondence to the actual reserved allocations | `observe_receipt` reads the same vectors after reserve-all and before initialization; receipt/storage attachment persists. Generated vector lengths are not capacities. |
| Initialization execution and pending carrier correctness | `materialize_rebuild` holds the real key and initializes the same reserved vectors with the default; target lengths, vacant live/tag bits, no vector replacement or allocation beyond checked capacity, and the admitted callback/termination laws. |
| Refusal semantics | `SizeOverflow`/`AllocationFailed` exposes no partial tuple/table, leaves committed observations unchanged, and precedes hold, clones, merge and relocation. A failed post-reservation capacity invariant is not an allowed late allocation refusal. |

The premise must not assume `rebuild_preserves`, resulting `ReservedRepR`, uniqueness,
`DirectCover` or pending `insertWith` correctness. `RebuildPlanCorrect_R` remains input-side;
its concrete clause interpretations and their realization are obligations, not consequences of
defining parameterized predicates. Joint success/refusal witnesses and actual-path tests remain
required; tests of the adapter are `tested`, not a proof of this premise or allocator simulation.

**Graph position (arrows are dependencies, not claims of completed proofs):**

```text
real unextracted adapter execution
  -- ResourceAdapterSimulates (explicit, unproved) --> resource facts for its exact result
  -- extracted publication + scalar-capacity theorems --> conditional reachable publication witness
  -- whole-table RepR/ReservedRepR + input-side plan/carrier laws --> disjoint-write frames
  -- rooted traversal + prefix invariant --> rebuild_preserves
  -- SearchDecisionCorrect + rooted pending commit + callback laws --> pending insertWith composition
```

The extracted `word_{stored,regen}_publish_prepared` and `word_capacity_bound_holds` roots remain
the proved core at their stated scopes: faithful packaging and scalar comparison. The graph does
not make unextracted `apply_prepared_rebuild` orchestration or allocating public methods extracted.
It supplies no unconditional reachable constructor, reservation, initialization or refusal theorem;
no generic search-law proof; and no executor/spawn/join/unwind, arbitrary callback allocation/panic,
rustc/LLVM, ABI, GPU/SPIR-V or unsafe-container coverage. All prior exclusions and both axiom cones
remain unchanged. The missing source-to-adapter link stays visible in every composed claim using it.

**Discharge decision:** retain a standing explicit premise for this derivation, not a permanent
waiver of the source-linked realization obligation. A small extra wrapper/root cannot cheaply
discharge it: publication/comparison are already rooted, `Vec` extraction erases capacity, and the
adapter reaches reservation, initialization and the planner's recorded extraction barrier. Discharge
requires either extracting and refining the actual adapter with checked primitive models, or a
source-linked adapter theorem connecting its actual execution/effects to the generated result and
all clauses above. Merely extracting more syntax, a model tuple, a scalar accessor returning an
unrelated number, or passing tests is insufficient. Any tool/model/root extension is a separately
approved, bounded task with rejecting controls; none is authorized here.

**Headline impact:** no statement, premise or grade change to the eight relocation/traversal/commit
headlines listed above, or the four resource theorems `word_capacity_bound_holds_iff`,
`word_stored_publish_prepared_carries`, `word_regen_publish_prepared_carries` and
`preparedResources_of_publication` in `Apeiron.GrowableResources`. The resource predicate definitions
and their audits also remain unchanged by this decision. Register the new conditional witness
separately; do not relabel the old transport as unconditional adapter reachability. After any
regeneration, revalidate all twelve headlines and predicate audits by bounded builds/audits and
freshness plus the existing and resource-specific controls on the same source/tool identity.
The [derivation order](roadmap.md#resource-premise-to-whole-table-derivation)
retains the whole-table, frame and composition obligations. REG-1/G1 and the full barrier stay open.

### Pending-composition premise boundary

Owner Astra; task `APEIRON-EXTRACT-REP-RESOURCE-BOUNDARY-T4`; grade `anchor`, docs only.
`Growable/Rebuild.lean` now contains `Apeiron.GrowableRebuild.word_stored_rebuild_preserves`
and `word_regen_rebuild_preserves`: rooted traversal preserves the old map in `ReservedRepR`
under the source representation and input resource/plan certificates. Their supplied build/control
evidence is not rerun here. They do not perform the pending `insertWith`. The earlier graph's last
arrow is incomplete without the four obligations below; W7 remains open.

**Acceptance decision.** All four are legitimate **named, unproved premises of a conditional core
refinement**, following `ResourceAdapterSimulates`/`SearchDecisionCorrect`. None must be discharged
before accepting that explicitly conditional implication, provided its exact premises, joint
witnesses, source bindings, audits and rejecting controls pass Sol's semantic adjudication. None
authorizes unconditional production, allocating-public-method or end-to-end acceptance. Such claims
require source-linked discharge of every applicable premise, including adapter/search realization.
Use explicit contract arguments, never axioms, admitted results or instances asserting their truth.
Projection proofs establish consequences of a supplied premise, not that execution satisfies it.
No premise may assume the pending committed `RepR`, canonical successor, or successful execution
whose correctness the composition must establish.

Sol owns the pinned names and exact definitions/projections in `Growable/{Search,Resources}.lean`;
this section records their required scope, not acceptance of the concurrent implementation. All facts
must refer to the same pre-state, prepared tuple, pending `(sigma,key,value,address)`, actual
post-relocation storage and decision consumed by the rooted commit, for each carrier at width 16.

| Named premise | Decision, scope and missing obligation | Discharge path / immediate feasibility |
|---|---|---|
| `DecisionSymbolicCorrespondence` | Admit explicitly for conditional composition. Connect `RebuildPlanClauses.pendingDecision` (carried through `RebuildPlanCorrect_R`) to the supplied runtime `SearchDecision`: found/vacant class and exact flat/spill destination agree. The found entry refers to the same source enumeration/assignment; exhaustion and malformed pairs are excluded, not assumed harmless. Plan validity alone says nothing about what `locate_rebuilt` returns. | Refine actual `locate_rebuilt`/composite decision execution on the storage returned by rooted traversal; transport the exact symbolic prefix and prove agreement with its input-side pending certificate. A supplied equality can be projected cheaply; production correspondence cannot. Retain the extraction/tool barrier and do not substitute a model locator. |
| `QueryAddressCorrect` | Admit explicitly for conditional composition. Bind the pending address to generated `WordHome.address` for `(sigma,key)`; a vacant flat location equals its certified `preparedFlatIndex` at the rebuilt lane extent, and a vacant spill location uses generated `table.shard address`. Retain in-bounds/first-fit/search coverage and composite `DirectCover` obligations separately. `FirstFit state order location` does not constrain that caller-supplied order to query routing. | Refine `Flat::index` checked conversions, multiplication/addition and admission against `preparedFlatIndex`, then the actual decision routing through `WordHome.address`, `shard` and `SearchDecision::from_spill`. Existing `RepR.address` can supply found-entry addressing, not addressing for a new vacant representative. Scalar sublemmas may be small; full vacant-path realization is not a cheap wrapper proof. |
| `FreshLengthBound` | Admit as an explicit arithmetic/domain premise, not a new allocator law. On a fresh/vacant insertion require exactly `state.len.val + 1 < Usize.max`, transported to the rebuilt view whose old length is unchanged. Found updates retain length. `LengthCorrect` counts live keys but supplies no strict fresh-length headroom; separate vector lengths/capacity receipts do not bound their sum. | Derive headroom from a source-linked reachable aggregate live-count/resource invariant, initialization and preservation, or separately authorize checked length admission before hold/clone/merge/mutation and refine success plus `SizeOverflow` stutter. Check the strict inequality against generated `Usize` arithmetic; do not silently weaken it or exclude admitted boundary inputs. No cheap derivation from the current premises is established. |
| `GrowableCallbackLaws` | Admit explicitly for conditional composition, but bind the statement's `merge` to the **actual generated monomorphic** `WordMerge` result `core.num.I64.wrapping_add new old` (and required clone/carrier equations). This is a callback realization restriction, not permission to prove arbitrary `merge` semantics from word roots. Keep purity/termination and new-first requirements explicit; historical fixed-shard `CallbackLawsG` supplies no growable instance automatically. | First discharge the concrete wrapping-add instance by unfolding the generated `WordMerge`/`CloneI64` dictionaries and reusing checked carrier equations. This is the cheapest bounded Sol follow-up and should precede the retry if its final definition permits it; no new extraction is needed for those already-generated bodies. Generic callbacks require refinement of the real generic body/dictionaries under generic laws and a rejecting noncommutative order control, not specialization alone. |

**Length and callback limits are not runtime measurements.** The source pending commit merges
`value` with the old/default value and performs unchecked `self.len += 1` on its two vacant branches
(`src/table.rs`, `apply_pending_and_commit`); ordinary `apply_insert` does likewise. Conditional
no-overflow of the extracted addition proves neither public reachability of the strict bound nor
the behavior outside it. Rust checked sizing of individual allocations does not by itself prove that
bound, and no new `SizeOverflow` branch is asserted here. Similarly, the generated word callback is
wrapping addition, not mathematical unbounded addition or an arbitrary user callback. Discharging its
concrete instance would remove that instance's premise only; arbitrary callback allocation, panic,
divergence and drop behavior remain excluded.

**Exact residual perimeter.** Even after a conditional W7 proof, evidence does not establish execution
of composite `search_decision`/`locate_rebuilt`, its direct-empty/finite-sweep/first-fit decisions, the
production symbolic/runtime decision connection, or the full `Flat::index` checked-conversion and
query-routing path. Generated address/word/spill subroutines keep their existing scopes; their
presence is not composite correctness. The evidence does not establish that all fresh production
states satisfy `FreshLengthBound`, that overflow outside it returns a transactional error, or that an
arbitrary callback realizes the theorem's `merge`. The same source-to-adapter, reservation, capacity,
initialization and orchestration exclusions remain. No roots are added or removed. Allocator,
executor/spawn/join/unwind, rustc/LLVM, ABI, GPU/SPIR-V and unsafe-container exclusions, the Rust
closed cone and separate Slang empty cone all remain unchanged.

**No-clone-on-`Err` remains a separate required branch.** `ResourceAdapterSimulates.refused`
explicitly carries unchanged committed state and zero clone/merge/relocation counts; these counts
are assumed observations, not an extracted effect trace. Retain `insert_err_stutters` and the
whole-operation `resource_adapter_simulates` obligation: source-link checked planning/reserve-all
refusals and `PreparedResult::Refused` propagation before hold/initialization/clone/merge/relocation,
with no returned partial build. A conditional refusal projection may transport those facts but
cannot discharge their realization. Actual-path late-reservation/overflow tests and an early-clone
mutation remain required at grade `tested`; unconditional proof requires a source-linked effect
refinement. Neither the prepared-success composition nor an apply function with no `Err` branch
proves this rule, and an internal capacity/placement failure after cloning is not an allowed late
allocation refusal.

**Headline and contract impact.** No existing proved statement or recorded grade changes: entry,
traversal, commit equalities, publication/comparison, conditional reachable witnesses and both
`word_{stored,regen}_rebuild_preserves` retain their exact premises and scopes. The unproved W7
headline gains explicit assumptions; a successful bounded proof may be called `proved` only as that
implication, not as production realization. Earlier wording in the read-only spec saying no extra
premise is needed is insufficient for this stronger whole-table composition. Sol must reconcile its
exact contract before W7 acceptance; Astra neither edits nor waives it. Recheck joint satisfiability
of the plan's symbolic-decision uniqueness and all new premises for found and fresh flat/spill cases;
projections or incompatible separate witnesses do not meet that requirement. The
[retry and adjudication conditions](roadmap.md#pending-composition-retry-and-semantic-adjudication)
retain every previous control, realization obligation and full barrier. REG-1/G1 remains **OPEN**.

### W7-SearchCommitBoundary

Owner Astra; `APEIRON-STRATEGY-DECISION-T1`; architecture decision at grade `anchor`. **Freeze W7
proof retries; retain the search/commit composition as named trusted/unproved work.** This is a
priority change, not approval of a tested-only production replacement or a change to Sol's spec.
The seven supplied residual names are:

`ResourceAdapterSimulates`, `SearchDecisionCorrect`, `DecisionSymbolicCorrespondence`,
`QueryAddressCorrect`, `FreshLengthBound`, `GrowableCallbackLaws`, `SpillVacantDirectGuard`.

The first six have the interfaces/discharge obligations recorded above. `SpillVacantDirectGuard`
is a supplied retry obligation, not a declaration found in the read `Growable` sources: Sol must
pin its exact type and joint witnesses. Its required content is that a fresh spill write cannot
leave the pending key's admitted direct slot vacant and thereby violate composite `DirectCover`.
Do not assert a definition, proof or source binding for it here. The list supplements, not replaces,
`RepR`, `PreparedResources_R`, `RebuildPlanCorrect_R`, carrier/codec and other existing hypotheses.
No axiom, admitted theorem, global instance or assumed committed post-state supplies these facts.

| Ledger entry | Exact status to retain |
|---|---|
| Word/identity/geometry/scan, scalar capacity comparison, publication, decision-conditioned apply, disjoint-write frames and commit equalities | Existing `proved` results only at their audited statements, input premises and recorded source/tool identities; not freshly rebuilt here. Capacity comparison is not allocation correspondence. |
| `word_stored_rebuild_preserves`, `word_regen_rebuild_preserves` | Existing conditional old-map `ReservedRepR` preservation under source `RepR`, publication and resource/plan certificates; no pending canonical insertion. |
| Whole-table `RepR`/`ReservedRepR` | Definitions and checked lemmas, not an unconditional reachable-state preservation theorem for every public operation. |
| `apply_insert` whole-state preservation; `prepared_rebuild_refines_insertWith` | Unproved/open and suspended. No conditional composition theorem is asserted merely because its dependencies build. |
| Search/commit adapter evidence | Record each Rust test with actual source hash, domain, positive/red/restored exits and exclusions. Until all required controls run, boundary test acceptance is OPEN. The recorded two-lane Kani timeout (900 s, exit 124 in the spec) is inconclusive, not a successful bounded check. |
| G1/REG-1 | OPEN; `report_complete=false`. Neither these proofs nor a future performance pass discharges the other joint obligations. |

**Required rejecting evidence, not new proof.** Exercise the real path for both carriers: direct and
spill found/fresh writes, unequal `Sigma`, colliding direct coordinates, window vacancy followed by
a late sweep hit, exhaustion/growth, extent reclassification, unchanged unrelated keys and exact
length. Compare ordered traces against an independent full-key map with a noncommutative new-first
merge. Disposable mutations must remove the sweep, choose a wrong shard/direct index, corrupt
value/occupancy/length, omit final commit, or reverse merge order and be rejected for those reasons.
Inject overflow and a late reservation failure through the real adapter: unchanged committed
observations, zero hold/clone/merge/relocation, no partial build. A hand-built prepared tuple tests
apply only; it is not actual adapter reachability. A generated-body mutation and a Rust-path mutation
test different links; keep both existing obligations. Timeouts, OOM and compile errors are not
semantic red controls. Ordinary tests execute production geometry; concrete helper fixtures do not replace public
constructor, representation or universal source-refinement evidence.

**Trust statement.** Actual composite search/`locate_rebuilt`, query routing, symbolic/runtime
decision agreement, aggregate fresh-length headroom, callback effect counts, adapter planning,
reservation/capacity/initialization and their realization remain unproved where named above.
Whole-state pending apply/commit is also unproved even though its real bodies are extracted.
Allocator/global allocator, executor/spawn/join/unwind, arbitrary callback panic/allocation,
rustc/LLVM codegen, ABI, GPU/SPIR-V and unsafe-container exclusions remain unchanged. The
Rust/canonical cone stays `{propext, Classical.choice, Quot.sound}`; Slang retains its empty cone.
This records trust, not assurance, and does not narrow the public admitted domain.

The [reopening condition](direction.md#current-strategy-decision-freeze-composition-restore-performance)
requires a source-linked finite discharge plan and a bounded full-`RepR` task before another W7
attempt. Sol must retain all six semantic-adjudication conditions in the roadmap, including joint
witnesses and noncommutative controls. A counterexample or surviving mutation triggers immediate
correctness repair/adjudication rather than waiting for performance; it does not authorize weakening
a premise or claiming W7 closed. Existing core freshness/build/audit/control regressions continue
after every production change. The eventual required theorem set and end-to-end criterion are intact.

## Counting-face representation boundary (`APEIRON-REORIENT`)

Owner reorientation decision; grade `anchor` — architecture and scheduling, with no source, proof,
extraction or measurement claim. It is recorded in
[direction](direction.md#five-barrier-execution-sequence-apeiron-reorient) and specified by [the
counting-container contract](epics/storage-refinement.md#coordinate-keyed-counting-container-contract).
This section adds the selected counting face beside the generic growable `Table` and qualifies,
**for that face only**, the local-authority sentence at the head of this file, the "executing
**map-face** representation is … one `Table`" and "no aliases or wrapper containers" sentences under
*The historical fixed-shard representation*, and the "physical representation described here" wording
under *Shared canonical map vs physical representation*. Each holds in full for the generic key/value
face, which remains the selected representation there, and each remains revision-bound historical
evidence there. The counting face is not a coherence layer, a cache, a benchmark-only structure, or a
wrapper that restates a formula, and no existing proof admits it.

**Representation.** One single region of open-addressed slots holding the coordinate word, `Sigma`
and the count, with tags as the sole occupancy authority, a growth policy, and no locator and no
256-region directory. The generic growable `Table<K, V, H, N, R>` in `src/table.rs` is retained
unchanged as the map face. It is production code with real counting consumers, not a reference or
test-only structure.

**Canonical denotation.** The admitted domain is the packed `u32` coordinate word already certified by
the resident area; a coordinate outside it is refused, not truncated. The logical key is exactly
`(Sigma, coordinate)` and the value is the `Int64` count, so a counting read is the canonical
`Apeiron.Table.get` of a canonical `Apeiron.Table.build` under the additive merge with default `0`:
`get a = |{i · addresses[i] = a}|` is `build (fun new old => new + old) 0 (addresses.map (·, 1))`. The
refinement targets are exactly `Apeiron.Table.HMap`, total-by-default `Apeiron.Table.get`, new-first
`Apeiron.Table.insertWith` and ordered `Apeiron.Table.build` — the same canonical object the generic
`Table` refines, with the per-key value-subsequence equation supplying
`build_eq_of_key_streams_eq`. A second representation of that one object is added; no second
semantic object, canonical theorem or attribution is introduced or changed.

**Existing proofs do not extend to it.** The `TableRefinement` headlines and every `Growable/*`
module refine the fixed-shard and growable generic bodies only. They do **not** establish this
container: its constructor, insertion, growth, batch build, read/readback and refusal links are
unproved until the barrier-4 extraction and refinement land. Historical `CountTable`/`Map` results
are archival measurement inputs and carry no checked Lean refinement.

**Extraction perimeter.** Candidate roots are the sequential bodies — checked sizing, reservation and
initialization, the bounded probe, insertion, the ordered batch build and the serial read —
included from the real source and called, never restated by a wrapper. The parallel
`count_par`/`get_each` bodies and their scheduling sit behind the same unextracted `std::thread::scope`
orchestration boundary recorded above, and the pinned `JoinBoundary` and `try_reserve_exact`
translation limits recorded there apply to the counting roots unchanged. Every added root is its own
extraction task, registered in `proof/corpus/AxiomAudit.lean` inside
`{propext, Classical.choice, Quot.sound}` and never inside `Slang.*`.

**Resource and refusal semantics.** The ratified option (b) contract above applies unchanged: checked
sizing, reserve-all ordering, initialization, one transaction point, and a `Result` whose `Err` leaves
the container unchanged with no partial result. The denials are the resource ones — overflow and
allocation failure — not `ShardFull` and not positive-`N` `BuildAdmissibleN`, which stay retired from
production. An absent key is a successful read of the default, never a refusal, and the canonical
`Apeiron.Table` object stays total.

**Trust exclusions.** Allocator success/failure correspondence and `try_reserve_exact` itself, thread
memory semantics, exclusive-borrow realization, spawn/join failure, panic propagation, liveness,
`rustc`/LLVM codegen, the kernel ABI, GPU/SPIR-V and the `unsafe` container are excluded exactly as
above, and the Slang chain keeps its empty cone. In addition, coordinate/codec injectivity on this
face is **unproved**: the codec must be total, stable, equality-respecting and satisfy
`decode(s, encode(s, c)) = c` on the admitted domain, and that law has no canonical link yet.
HashDOS resistance is required, not established. A verified speed an adversary can collapse does not
pass.

The five barriers in [the plan](plan.md#subsumption-execution-sequence-apeiron-subsum) sequence
this face; implementation stays blocked until the contract names an executable gate and that gate's
rejecting control is established.

## Trust inventory

<!-- APEIRON-STRATEGY-DECISION-T2 BEGIN -->
### Spill-filter obligations and partition cut

Owner Astra; `APEIRON-STRATEGY-DECISION-T2`; grade `anchor`. This updates the current boundary ledger,
not historical proof grades. Read `Growable/Representation.lean` defines `RepR` without summary
coverage or fingerprint agreement; `Resources.lean` checks tag/payload lengths but no summary receipt.
`Search.lean`'s mandatory-sweep clauses predate the summary-miss shortcut. Source extraction is in
flight, not freshly accepted here. Older complete-sweep descriptions therefore specify the unfiltered
route; any shorter route requires the additional sound absence certificate below, never mere vacancy.

**Invariant decision:** retain these explicit named premises during the W7 freeze; require their
formalization and source-linked establishment/preservation in the bridge before end-to-end acceptance.
Names below are ledger identifiers, not declarations claimed to exist. Sol owns the exact definitions,
joint witnesses, dependency order and audits; Flash cannot add assumptions to a theorem to turn it green.

| Ledger entry | Exact obligation / current evidence | Owner and discharge |
|---|---|---|
| `SpillSummaryShape` | Empty summary disables filtering (`summary_maybe = true`). Otherwise its word count is a power of two, `bits = trailing_zeros(len) + 6 < 64`, and generated mixing/word/bit indexes are in bounds. Checked byte sizing/reservation includes summary storage. OPEN / `anchor`. | Sol: generated arithmetic/bounds lemma and joint resource predicate; Flash: actual adapter sizing/refusal tests. Capacity correspondence remains under `ResourceAdapterSimulates`, not inferred from length. |
| `SpillSummaryRecorded` | At each search-visible completed state, every nonzero-tag payload's reconstructed placement has its summary bit set when filtering is enabled. Empty summary is a disabled filter, not a recorded bit. Initialize, fresh insert, prepared-shard apply, growth, relocation and pending commit establish/preserve this for both carriers; found updates retain it. OPEN / `anchor`. | Sol: generated write/frame proofs and constructor/adapter linkage, eventually compose with whole-state `RepR`. Input invariant is legitimate; assuming the output invariant to prove preservation is not. |
| `SpillSummaryMissSound` | Under shape, recordedness, routing and stable equality-respecting address/carrier laws, `summary_maybe(query.place) = false` implies no matching full `(Sigma,K)` in that routed spill region. It says nothing alone about flat absence. Summary hit proves no presence. First-fit vacancy order remains unchanged. OPEN / `anchor`. | Sol: derive from recordedness, then source-bind both spill locators and the composite route; reconcile `SearchDecisionCorrect`'s coverage certificate without assuming the desired absence. |
| `SpillFingerprintExact` | A live tag equals `0x80 OR ((place >> 8) AND 0x7f)` for its payload's address, hence is nonzero. Equal logical keys have equal tags; a mismatching tag excludes equality. Exact scan masks identify byte equality, not key equality; collisions still require full `Sigma`/carrier comparison. OPEN / `anchor`. | Sol: generated `tag` equation/nonzero, write preservation, address agreement and scan composition; no injectivity or HashDOS claim. |

Required rejecting controls: omit each live-write summary update, clear a required bit after growth or
relocation, corrupt a fingerprint while retaining the live bit, and bypass full-key comparison on
collisions. Both carriers, unequal `Sigma`, disabled/empty summaries, summary collisions and late hits
must have joint witnesses and actual-path tests. Run generated-body semantic controls separately from
freshness controls; an elaboration error or missing field is not semantic rejection.

**Source discrepancies, not newly passed contracts.** `apply_prepared_shard`, `grow_spill`,
`relocate_entry`, `apply_insert` and pending commit write the tag before `summary_set`; the comment
claiming “recorded before its tag becomes nonzero” is not implemented. The ledger specifies completed
search-visible states, not that temporal claim; Sol must check no intervening observation/failure,
or require source ordering repair. `reserve_empty` allocates the summary with infallible `vec!`, while
`summary_words` uses saturating/unchecked sizing; summary allocation is absent from the reservation
receipt. `build_prepared_shard` also holds before reservation, and parallel workers can merge before
another worker's reservation fails. These are open conflicts with reserve-all/no-clone-on-`Err`, not
permission to weaken it. Source-linked resource correction needs its own bounded task before admitting
the kernel or a complete resource claim. The allocator/executor exclusions do not excuse source-level
ordering defects. No counterexample run is asserted by this source review.

**Partition authorization.** Retain tile endpoints, exact input length, in-range indexes and recomputed
routing checks. Replace the `seen` allocation/coverage scan with strict ascending indexes within each
shard: local order excludes local duplicates; deterministic routing excludes duplicates across shards;
exact cardinality then gives complete coverage. This is a mathematical design argument, not a checked
Lean theorem. The new checker additionally rejects same-shard reordering, which the current checker
does not detect; `Parts::of` already promises stable order, so no public input domain is narrowed.
Sol must ratify soundness and constructor completeness before Flash implements it. Do not substitute
a checksum, cached producer assertion, debug-only check or unconditional `true` for validation.

**Payload cut rejected for this dispatch.** `Region.slots` is `Vec<(Sigma,R,V)>`, and core operations
index it at arbitrary vacant lanes. Reserved capacity is not initialized length. Deleting
`initialize`, using unchecked length/`MaybeUninit`, or assuming initialized destinations is not an
approved optimization. Reconsider only after a source-bound phase receipt isolates bytes/clones saved,
a concrete safe layout accounts for vacant/live payloads and drop, and Sol ratifies constructor,
success/refusal, relocation and extraction feasibility with inhabited witnesses. The existing
initialized-resource interface and all old obligations remain until that separately gated decision.

After either production change, regenerate both carriers and all retained roots, repair generated
state constructors/write models, and revalidate Search, Relocation, Representation, Resources and
their downstream audits/controls. In particular, adding an empty summary to every Lean update would
misrepresent the executed writes. The open `generated_tag_nonzero` target alone would prove neither tag/payload
agreement nor summary soundness. New summary bounds can make old universally quantified apply
statements false on malformed regions: report the exact goal to Sol, not a silent domain restriction.
W7 stays frozen; regression maintenance is not a composition retry. Rust/canonical closed cone,
Slang empty cone, allocator/executor, callback, codegen, ABI, GPU and unsafe-container exclusions stay
unchanged. This ledger enlarges visible open work, not assurance.
<!-- APEIRON-STRATEGY-DECISION-T2 END -->

| Layer | Owner | Status |
|---|---|---|
| Canonical algebra and combinator definitions | Apeiron (Lean corpus) | proved at stated scope (`Apeiron.Combinator.Comb.reduceDeep_sound`, `Apeiron.Trichotomy.AnyRing.ringNormForm_mul`) |
| Slang source refinement (Stream 0) | Apeiron | empty-cone gate `tools/slang/zero_axioms.py`; unconditional `Slang.SourceAudit.source_binding` remains absent from `proof/corpus/Slang/SourceAudit.lean`; no whole-source acceptance |
| Bounded source-witness generator | Apeiron, existing `tools/slang/export.py` | untrusted producer; chunked generation remains required; retirement gate `tools/rooting/retire_check.py` R6 rejects reintroduction of the binned route and forbidden option strings |
| Structural source witnesses | Lean source relations in `proof/corpus/Slang/Source/Tokenizer/` and `proof/corpus/Slang/Source/Parsing/` | kernel-checked witnesses and empty axiom audits required; composition lemmas at `37b95c3`, concrete chunked table/std UTF-8 witnesses still open; default recursion depth and bounded heartbeats/time/memory; no codegen/ABI claim |
| Rust `Cell<i64>` kernels | Apeiron | refined to canonical definitions; axiom cone `{propext, Classical.choice, Quot.sound}` |
| Rust `Table` bodies | Apeiron | extracted and elaborated; word-instance and arbitrary-callback constructor/insertion/admitted-build refinement proved under their explicit laws and domains |
| Charon/Aeneas translation fidelity | pinned toolchain | external trust; patched for const `Repeat` to `Array.repeat` |
| Lean kernel | pinned toolchain | trusted |
| rustc/LLVM codegen and generated machine code | compiler | not covered |
| Foreign `unsafe`/ABI code outside the word claim, kernel ABI, stack/OS | platform | not covered by the table refinement |
| GPU / SPIR-V | target | not covered |
| Generic `Home`/`Eq`/`Clone`/`Merge`/drop | caller | separate instance obligations; may allocate, panic or diverge |

The executing table's generic callback proof assumes `CallbackLawsG`; it does not establish
termination or purity for callbacks outside those laws. Neither a reference container nor a
tested-only optimized container is an accepted substitute.
Former storage proofs remain historical evidence about deleted source, not this table.

The [bounded source-witness route](contracts/slang-source-refinement-plan.md#bounded-source-witness-route)
shares data/index equations and composes proofs of existing structural relations; the retired
named proof DAG / segment-lemma evaluator and retired generated traces remain binned. R6 checks
producer text, not arbitrary Python behavior or default-depth compliance: `tools/slang/export.py`
currently emits `maxRecDepth 10000`, which must be removed for this route. A compiled `#guard`, a
generator result or the reported barrier-probe measurements are not kernel-checked source binding.

## Kernel scope and isolation

The kernel ABI, rustc/LLVM code generation and the GPU backend remain outside the current
word-storage claim, but they are not forever out of the programme: they are future lift and
realization targets with their own gates. A userspace runtime cannot be dependency-free in the
absolute; the kernel ABI is the floor. The isolation goal, including ring 0, is local to Song;
Apeiron does not claim it. Any foreign `unsafe` or ABI code outside the word claim is named and
small where it exists; it is not part of the table and is not covered by the extracted table
refinement.

The floor is the kernel. Above it, pick the smallest layer that must be right, prove it, and let
everything else be an adapter.
