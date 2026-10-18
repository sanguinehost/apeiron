# Joint Primitive Profile

This file is the frozen specification of the joint primitive profile for
`APEIRON-PROGRAMME-W17-T1`. It names the primitives whose Rust and Song executions are to be related
to one canonical definition, the seven obligations each profile row carries, the selected boundary,
the unmatched counterparts recorded as blockers, the known hazards, the integration order and what
is explicitly not covered.

Authority: [`docs/direction.md`](../direction.md#primitive-correspondence) is the programme
authority; [`docs/kernel-boundary.md`](../kernel-boundary.md) owns the Rust representation and trust
boundary; [`proof/README.md`](../../proof/README.md) owns extraction and bridge scope. The canonical
corpus remains at `../sanguine/proof/` at Git
`8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f`, Mathlib `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.

## Canonical object and executing root

The canonical object is `Apeiron.Table.HMap`, `Apeiron.Table.get`, `Apeiron.Table.insertWith` and
`Apeiron.Table.build`, together with `Apeiron.Trichotomy.AnyRing.RingCell`,
`Apeiron.Trichotomy.AnyRing.ringMul` and `Apeiron.Trichotomy.AnyRing.ringNormForm`. The extracted
`Cell<i64>` word bodies are `word_mul_denote` and `word_norm_denote`. Every executing face in this
profile roots in `Apeiron.Table.HMap` and the other canonical roots named here; a face that names
the object only by path, namespace or import is not rooted.

The O(1) qualifier is the parameter boundary: fixed N, bounded callbacks and nonconstant key length
bound the table. Reading a whole source remains at least linear in its bytes at fixed N, so
constant-time table operations at fixed N do not imply constant-time whole-source checking or
constant-size proof terms.

This document specifies obligations. It does not claim the correspondence exists. No row is closed:
every row below is `selected-open` or `blocker`. The machine-readable form is
[`tools/joint/profile_rows.json`](../../tools/joint/profile_rows.json); the checker
[`tools/joint/profile_check.py`](../../tools/joint/profile_check.py) establishes specification
consistency only.

## Seven required fields

Every joint profile row carries these seven fields. A row that drops one is incomplete and the
checker rejects it.

| Field | Obligation |
|---|---|
| `canonical` | The canonical object and the observation relation, with an exact scalar type. |
| `entry_points` | The concrete entry points and source identity on each side. |
| `representation_invariant` | The representation invariant, established by the actual constructor. |
| `admission` | Separate Rust and Song admission predicates; joint success requires both, and unequal capacities need not give identical refusals. |
| `behavior` | Success, unsupported-input and resource-exhaustion behaviour, including the raw return encoding and the post-state. |
| `resources` | Separate Rust and Song resource accounting; arithmetic ops, probes, nodes, stack entries and input consumption are not interchangeable fuel. |
| `refinement` | A source-connected refinement and a rejecting control. A copied model or a differential suite alone does not close a row. |

## Selected boundary

The selected boundary is word arithmetic and table init/read/write. `Table::build` has no Song
transactional equivalent and is a composition obligation (B6), not a matched primitive.

| Row | Canonical object | Rust entry | Song entry | Status |
|---|---|---|---|---|
| R1 | merge `f` of `Apeiron.Table.HMap`; addition observation | `src/map/table.rs::WordMerge::merge`, `src/scalar.rs::<i64 as Ring>::add` | `table/table.slang::mergeAcc`, `tableInsertWith` | selected-open |
| R2 | `Apeiron.Table.HMap` with `get`, `insertWith`, `build` | `src/map/table.rs::Table::{new,get,insert_with}` | `table/table.slang::{tableInit,tableFind,tableGet,tableInsertWith}` | selected-open |

Canonical authority for R2 is `Apeiron/Table/Basic.lean` (`HMap`, `get`, `insertWith`, `build`,
`get_insertWith_self`, `get_insertWith_ne`, `get_build_acc`, `build_gauge_invariance`).

### R1 · Word arithmetic merge: the addition slice

**1 · Canonical object and observation relation.** The merge parameter `f : V → V → V` of
`Apeiron.Table.HMap`; the arithmetic merge is commutative-ring addition on the scalar. Observation:
`Apeiron.Table.get (Apeiron.Table.build f d kvs) x` is the fold of `f` over that key's values in
stream order (`get_build_acc`); schedule freedom needs `hcomm`/`hassoc` (`build_gauge_invariance`).
Exact scalar type: Rust `i64` as the wrapping `Z/2^64` (`i64 as Ring` add = `i64::wrapping_add`); Song
`mergeAcc(left, right) = left + right` on bounded machine integers with
`0 ≤ left, right ≤ COORD_CEILING`; canonical scalar `ZMod (2^64)` under `denote`. The corpus
theorems are `Apeiron.Table.get_build_acc`, `Apeiron.Table.build_gauge_invariance`,
`Apeiron.Table.get_insertWith_self`, `Apeiron.Table.get_insertWith_ne`.

**2 · Entry points and source identity.** Rust: `src/map/table.rs::WordMerge::merge` (the concrete
`Merge<i64>` wrapping-add method) and `src/scalar.rs::<i64 as Ring>::add` (the wrapping `i64`
addition instance method). Song: `song/src/table/table.slang::mergeAcc` (pure accumulator merge with
the in-tree laws `mergeacc_comm`/`mergeacc_assoc`) and `tableInsertWith` (whose `MERGE_ACC` arm
computes `oldVal + value` inline). The inline arm, not `mergeAcc`, is the executing merge.

**3 · Representation invariant.** Scalar arguments have no table constructor: their representation
obligation is the input encoding and denotation. `WordMerge` is a stateless Rust instance;
`mergeAcc` and the inline addition are operations, not constructors. Table-state construction
belongs to R2. Rust and Song agree only when the two additions agree on the matched domain; that
agreement is not yet proved for the Song side.

**4 · Admission predicates.** Rust: the merge admits every `(i64, i64)`; `i64 as Ring` add is total
wrapping; there is no merge-level refusal and admission is inherited from the table write. Song:
`mergeAcc` admits every pair; the inline addition in `tableInsertWith` executes only for an existing
key with `merge = MERGE_ACC`, after capacity, key, merge-code and slot checks succeed. `mergeAcc`'s
law givens require `0 ≤ left, right ≤ COORD_CEILING`; the inline arm carries no such guard.

**5 · Behaviour.** `mergeAcc` is pure addition. In `tableInsertWith`, the inline addition executes
only for an existing key with `merge = MERGE_ACC`, after capacity, key, merge-code and slot checks
succeed. A fresh `MERGE_ACC` or `MERGE_MAP` insertion stores and returns `value` without addition; a
fresh `MERGE_SET` insertion stores and returns `1`. An existing `MERGE_ACC` insertion stores and
returns `oldVal + value`; existing MAP and SET writes return `value` and `1`, respectively. Unsupported
input: none at the merge on either side; a Song value outside `0..COORD_CEILING` is outside the
matched domain but the inline arm still computes. Resource exhaustion: not applicable, one addition
either fits the word or wraps. Raw
return encoding: a bare `i64` word, no separate success tag. Post-state: Rust merge is pure; Song
writes the merged value into the slot for an existing key and stores without addition for a fresh
insertion, while `mergeAcc` alone is pure.

**6 · Resource accounting.** Rust: one wrapping addition per successful merge call; no probes, nodes
or stack entries; two scalar arguments. Song: zero merge additions on a fresh ACC insertion and one
on an existing ACC update; address, hash and probe arithmetic are additional costs; no probes, nodes
or stack entries at the merge; two scalar arguments. These are counted separately from R2's probe,
node and stack costs.

**7 · Source-connected refinement and rejecting control.** Rust:
`proof/Denotation.lean::denote_wrapping_add` proves the wrapping word addition is `ZMod (2^64)`
addition, but the extracted `WordMerge` has no dedicated bridge theorem, so the Rust merge is
source-connected only through that denotation. Song: `mergeacc_comm` and `mergeacc_assoc` are
in-tree Song laws, not canonical Lean; no checked Song-to-Lean refinement exists. The joint
refinement is OPEN. Rejecting control: change `new.wrapping_add(old)` to `new.wrapping_sub(old)` in a
throwaway copy of `src/map/table.rs`, re-extract, and `proof/regen.sh --check` fails on the
generated-body diff; restoring passes. This is an extraction-drift control; the semantic Song
refinement remains the open obligation.

### R2 · Table init, read and write

**1 · Canonical object and observation relation.** `Apeiron.Table.HMap K V := K → V` with `get`,
`insertWith f k v m := fun x => if x = k then f v (m x) else m x`, and `build f d kvs`. Observation:
`get (insertWith f k v m) k = f v (m k)`; `get (insertWith f k v m) x = get m x` for `x ≠ k`; a
built table reads as a per-key fold. Exact scalar type: word instance `Table I64 I64 WordHome 16`,
keys and values `i64`, denotation `I64 → ZMod (2^64)`, physical home `home(key) mod 256`. Corpus
theorems: `Apeiron.Table.get_insertWith_self`, `Apeiron.Table.get_insertWith_ne`,
`Apeiron.Table.get_build_acc`, `Apeiron.Table.build_gauge_invariance`,
`Apeiron.Table.merge_is_the_only_freedom`, `Apeiron.Table.build_eq_of_key_streams_eq`.

**2 · Entry points and source identity.** Rust `src/map/table.rs::Table::{new,get,insert_with}`.
`Table::build` and canonical build theorems are composition context only, tracked under B6. Song
`song/src/table/table.slang`: `tableInit`, `tableInitGo`, `tableCapacityOk`, `tableFind`,
`tableFindSlotGo`, `tableLookupSlotGo`, `tableGet`, `tableInsertWith`, `tableSpread`, `tableSlotKey`,
`tableSlotVal`, `tableSlotSet`, `tableSlotValSet`.

**3 · Representation invariant.** Rust constructor `Table::new(default, home)` sets
`shards = [None; 256]` and stores `default`. The invariant `Rep(t, h)` is the fixed outer
`[Option<[Option<(K, V)>; N]>; 256]`, an untouched shard `None`, globally unique keys, every entry in
its home shard `home(k) mod 256`, and `∀ k, D(t)(k) = h(k)` with `D` the matching override or the
stored default. For the concrete `I64` word instance the physical invariant is established at every
fixed shard capacity `N`: `emptyWordTableN_physicalInv` establishes the real constructor and
`physicalInv_insert_with_all` preserves it, including `N = 0`; `get` reads the represented total
map; `insert_with` preserves `PhysicalInv` and canonical `insertWith` on success and returns the
identical representative on `ShardFull`, with exact refusal at `N = 0`; for positive `N`,
`BuildAdmissibleN` exactly characterizes build success versus `ShardFull`,
`table_build_physicalInv_of_buildAdmissibleN` gives the successful-build `PhysicalInv`, and
`wordBuildN_zero_ok_iff_empty` characterizes the zero-capacity build. The generic callback layer
proves constructor, admitted insertion and admitted-build preservation under `CallbackLawsG`
(`insert_with_callback_admitted`, `table_build_physicalInv_of_buildAdmissibleG`,
`table_build_zero_ok_iff_empty`). `word_table_new_rep` remains the concrete read-only extensional
relation `∀ key, word_table_get t key ⦃ value => value = h key ⦄` at `Table I64 I64 WordHome 16`,
with canonical map type `HMap I64 I64`; it is the extensional component, not the physical
invariant. A table-wide `ZMod (2^64)` representation theorem is not named here. Required Song
invariant: an owned, valid extent of `2*cap` memory cells; representable address calculations;
unique occupied keys distinct from `TABLE_EMPTY`; probe reachability before the first empty slot;
and an explicit denotation of occupied and absent keys. That invariant is over Song's represented
storage, the caller `memory[]` extent, a different layer from the source-checker's memo store; no
representation result transfers between the two without its own refinement. `tableInit` initializes
the empty layout on valid memory. Constructor establishment, preservation and canonical denotation
remain proof obligations for the Song side. Capacity validation alone does not validate the memory
extent.

**4 · Admission predicates.** On valid memory, Song `tableInit` returns `0` after initialization and
returns `-1` without writes for invalid capacity. `tableFind` returns a slot index or `-1`;
`tableGet` returns the stored value or `-1`. Reads do not write. Every explicit `tableInsertWith`
refusal occurs before slot writes and leaves the table unchanged; earlier successful calls in a
sequence remain visible. Invalid memory/address extent and stack exhaustion have no checked refusal
encoding in this profile. Rust `N = 0` is a supported zero-capacity configuration, not an
unsupported input; writes return `ShardFull`. Full-capacity refusal requires an absent key. Existing
keys can still update in full Rust shards and full Song tables.

**5 · Behaviour.** Rust: `Ok(())` with an existing key updating `stored = merge(value,
stored.clone())` and a new key inserting `merge(value, default.clone())` in the home shard; `N = 0`
is a supported zero-capacity configuration, not an unsupported input, and writes return `ShardFull`.
Song returns the stored value: a fresh `MERGE_ACC` or `MERGE_MAP` insertion stores and returns
`value` without addition, a fresh `MERGE_SET` insertion stores and returns `1`, an existing
`MERGE_ACC` insertion stores and returns `oldVal + value`, and existing MAP and SET writes return
`value` and `1`. Unsupported input: Rust has none at this primitive; Song non-positive or
non-power-of-two `cap`, key -1, or merge code outside `MERGE_SET..MERGE_ACC`. Resource exhaustion:
Rust `ShardFull` exactly when the key is absent and the home shard has no vacant slot; Song -1 when
all `cap` probes are occupied. Raw return encoding: Rust `Result<(), StorageError>`; Song a raw word
where -1 (`TABLE_EMPTY`) is overloaded across missing read, empty slot, invalid capacity, reserved
key, invalid merge code and full table, and a successful `MERGE_SET` write returns 1; successful MAP
or ACC writes and present-key reads can also return -1 (H1). Post-state: Rust refusal changes no
field (`word_table_shardFull_refusal` at `N = 16`); Song writes in place, so a mid-stream full table
leaves earlier writes in place (H3); `tableInit` returns -1 on invalid capacity without writing.

**6 · Resource accounting.** Rust: word-instance writes perform one wrapping addition on success; an
absent-key insertion into an initialized shard can perform an `N`-slot key search followed by an
`N`-slot vacancy scan; 0 heap nodes because storage is inline; constant call depth with object size
growing with `N`; `insert_with` consumes one pair and `build` consumes `kvs` in order and clones each
key and value. Song: zero merge additions on fresh ACC insertion and one on existing ACC update;
address, hash and probe arithmetic are additional costs; initialization writes `2*cap` memory cells;
up to `cap` probe steps with measure `cap - step`; 0 allocated nodes (caller or static memory, the
`cells(32)` test arena); probe/initialization recursion is `O(cap)` source call depth, not a checked
machine-stack bound; no input consumption because the primitive reads and writes caller `memory[]`.
Allocation, probes, arithmetic, stack and consumed arguments/streams remain separate accounts and
are not converted to a common fuel.

**7 · Source-connected refinement and rejecting control.** Rust:
`proof/corpus/TableRefinement.lean` proves `word_table_new_rep`, `word_table_get_new_first`,
`word_table_get_replaced`, `word_table_build_get_acc` and `word_table_shardFull_refusal` against
`Apeiron.Table` at the concrete `N = 16` word instance. The concrete `I64` word instance proves the
physical invariant at every fixed shard capacity `N` (`emptyWordTableN_physicalInv`,
`physicalInv_insert_with_all`, `table_build_physicalInv_of_buildAdmissibleN`,
`wordBuildN_zero_ok_iff_empty`). The generic callback layer proves physical constructor, admitted
insertion and admitted-build preservation under `CallbackLawsG`, including the exact zero-capacity
split. Song's empty-cone table modules prove `RepSong`, constructor/read/write witnesses, exact
replacement admission/refusal, resource projections, and `Iota.Joint.songTable_replacement_link`.
That headline is not yet instantiated from the decoded source context;
`proof/corpus/Joint/WordBridge.lean` checks the `I64 ↔ Iota.Word` pointwise-map transport, but no
combined Rust–Song table theorem exists. The joint row is OPEN and T3 replacement write is the first
joint storage target. Rejecting control:
the same-filesystem throwaway generated-body mutation
`merge value default` → `merge default value` fails `table_insert_empty_zero` (recorded in
`proof/README.md`); restoring builds. This controls the Rust fresh-write denotation only and does
not control the Song side.

## Unmatched counterparts

These counterparts have no matched primitive on one side. They are recorded as blockers, not
assumed, and the checker requires every one of them to be present.

| ID | Counterpart | Reason | Status |
|---|---|---|---|
| B1 | Rust `src/cell.rs::{mul,norm_form}` (with `add`, channels, idempotents); Song side NONE. | Song has no executing two-slot algebra with a `Sigma` dial; `prover0.slang` operates on SKI/iota terms, so there is no matched domain. | blocker |
| B2 | Rust `src/carrier.rs::{point_density,point_flux,conservative_density_step,diffusive_flux_half,diffusive_line3_step}`; Song side NONE. | No Song executing primitive denotes the carrier readout or the exact integer carrier update. | blocker |
| B3 | Rust `src/exchange.rs::{exchange_impulse,elastic_exchange,accept_derived_impulse}`; Song side NONE. | No Song executing primitive performs the equal-mass exchange or the checked-claim acceptance. | blocker |
| B4 | Song `song/src/prover/ring/prover0.slang::{verdict,normalize,pass,structEq}`; Rust side NONE. | There is no executing Rust evaluator for the canonical `Apeiron.Combinator` semantics. | blocker |
| B5 | Song `song/src/prover/cone/certcheck.slang::{main,ccBatch,ccRun}`; Rust side NONE. | No Rust decider. `certcheck` must refine `Apeiron.Decider.checkCert`/`checkCert_sound`. | blocker |
| B6 | Rust `src/map/table.rs::Table::build` is transactional; Song has only per-insert `tableInsertWith` plus `tableInit`. | Song has no transactional `build`. Rust returns `Err` without exposing its locally constructed partial table; this is not an empty-table return. A separate composition task must specify Song's externally observable success and refusal post-state. A success-only restricted-domain result does not discharge B6's refusal/resource obligation. | blocker |

## Hazards

| ID | Hazard | Consequence |
|---|---|---|
| H1 | Song reserves key -1 (`TABLE_EMPTY`) and returns -1 for a missing read; Rust does not. | Song's raw return is not a reliable success/refusal tag; -1 conflates missing read, empty slot, invalid capacity, reserved key, invalid merge code and full table. Successful Song MAP or ACC writes and present-key reads can also return -1; it is not merely a shared code for several failures. |
| H2 | Rust `default` is a stored value for any unwritten key; Song `tableGet` returns -1 for a missing key and has no stored default. | The matched observation relation remains OPEN. For additive fresh writes, Song stores `value`, whereas Rust stores `value + default`; equality under direct word observation requires default zero. But an absent Song read returns `-1`, whereas Rust then returns zero. Choosing default `-1` fixes the direct absent-read comparison but breaks additive fresh-write equality. Excluding key `-1` does not resolve this value mismatch. A state-aware observation/adaptation relation must be specified; no such relation is established here. |
| H3 | Song `tableInsertWith` mutates slots in place; Rust `build` returns `Err` with no partial table. | Song's multi-insert is not transactional; a mid-stream full table leaves partial writes. This is B6. |
| H4 | Canonical `I` is `S K K`, while Song treats primitive `i` and the normal form `**skk` as distinct. | A regression fixture must pin the distinction; it is not patched away by conflating the primitive and the normal form. |
| H5 | Song `certcheck` must refine `Apeiron.Decider.checkCert`/`checkCert_sound`. | Until that refinement exists, Song's UNSAT replay is not established as a refinement of the canonical checker. |

## Integration order

1. The matched observation relation remains OPEN. For additive fresh writes, Song stores `value`,
   whereas Rust stores `value + default`; equality under direct word observation requires default
   zero. But an absent Song read returns `-1`, whereas Rust then returns zero. Choosing default `-1`
   fixes the direct absent-read comparison but breaks additive fresh-write equality. Excluding key
   `-1` does not resolve this value mismatch. A state-aware observation/adaptation relation must be
   specified; no such relation is established here.
2. R2 table init/read/write: the first joint storage target; define the matched admission relation
   and a Song `Rep` invariant.
3. Song has no transactional `build`. Rust returns `Err` without exposing its locally constructed
   partial table; this is not an empty-table return. A separate composition task must specify Song's
   externally observable success and refusal post-state. A success-only restricted-domain result
   does not discharge B6's refusal/resource obligation.
4. B1–B5 unmatched counterparts, in dependency order: cell algebra, carrier, exchange, combinator
   evaluator, certificate replayer.
5. A vertical lift/reduction demonstration only after the profile and the executing evaluator
   refinement.

## Not covered

- rustc/LLVM code generation and the generated machine code.
- The kernel ABI, stack and OS.
- GPU / SPIR-V.
- Runtime linkage between Apeiron and Song; the required result is a checked correspondence, not
  linkage.
- Song local registers, ontology and commands.
- Semantic correspondence itself. `tools/joint/profile_check.py` establishes specification
  consistency only; it does not report correspondence.
- Callback behavior outside `CallbackLawsG`.
- The unconditional Song source binding, decoded-context instantiation and combined R2/T3
  replacement theorem. `Joint.WordBridge` already checks the `I64 ↔ Iota.Word` pointwise-map
  transport. R1 and the remaining R2 rows stay open.
- The retired `Iota/Derivation` tower, the retired `Iota/Generated/Trace` outputs and the retired
  `DecodeTrace.Run`/`Derives` acceptance. They are historical and are not the active architecture;
  no row in this profile names them as a route.

## Checker

`python3 tools/joint/profile_check.py --check` validates the seven populated fields in each Markdown
row and JSON row; requires actual blocker/hazard table records rather than incidental ID mentions;
requires all five resource kinds separately for every implementation of every row; requires the
frozen selected rows to remain `selected-open`; resolves the profile-level canonical object and its
roots; requires the parameter-boundary qualifier; verifies recorded source hashes and corpus/Mathlib
identities; and resolves fully qualified declarations and concrete executing entries. It checks every
named reference, including blocker entries, rather than only selected structured references. It exits
nonzero on a missing field, row, blocker, hazard, theorem or source entry. `--controls` and
`--self-test` mutate a copy of the profile and the spec text in memory and require each mutation to be
rejected, including deletion of a Markdown row record, deletion of a blocker/hazard table record,
incomplete per-implementation resources, a false pin, a `closed` status, a reversed namespace, a
non-executing Song entry, a falsified canonical root and a blocker with no resolvable reference; the
working tree is never touched. The checker establishes specification consistency only.
