# Joint Primitive Profile

This file is the frozen specification of the joint primitive profile for
`APEIRON-PROGRAMME-W17-T1`. It names the primitives whose Rust and Song executions are to be related
to one canonical definition, the seven obligations each profile row carries, the selected boundary,
the unmatched counterparts recorded as blockers, the known hazards, the integration order and what
is explicitly not covered.

Authority: [`docs/direction.md`](../direction.md#primitive-correspondence) is the programme
authority; [`docs/kernel-boundary.md`](../kernel-boundary.md) owns the Rust representation and trust
boundary; [`proof/README.md`](../../proof/README.md) owns extraction and bridge scope. The canonical
corpus remains at `../sanguine-apeiron/proof/` at Git
`50c94192399add3b9f8e64bd14e36d8cba270705`, Mathlib `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.

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
| R1 | merge `f` of `Apeiron.Table.HMap`; addition observation | `src/table.rs::WordMerge`, `src/scalar.rs::Ring` | `table/table.slang::mergeAcc`, `tableInsertWith` | selected-open |
| R2 | `Apeiron.Table.HMap` with `get`, `insertWith`, `build` | `src/table.rs::Table::{new,get,insert_with,build}` | `table/table.slang::{tableInit,tableFind,tableGet,tableInsertWith}` | selected-open |

Canonical authority for R2 is `Apeiron/Table/Basic.lean` (`HMap`, `get`, `insertWith`, `build`,
`get_insertWith_self`, `get_insertWith_ne`, `get_build_acc`, `build_gauge_invariance`).

### R1 · Word arithmetic merge (wrapping addition)

**1 · Canonical object and observation relation.** The merge parameter `f : V → V → V` of
`Apeiron.Table.HMap`; the arithmetic merge is commutative-ring addition on the scalar. Observation:
`Apeiron.Table.get (Apeiron.Table.build f d kvs) x` is the fold of `f` over that key's values in
stream order (`get_build_acc`); schedule freedom needs `hcomm`/`hassoc` (`build_gauge_invariance`).
Exact scalar type: Rust `i64` as the wrapping `Z/2^64` (`Ring::add` = `wrapping_add`); Song
`mergeAcc(left, right) = left + right` on bounded machine integers with
`0 ≤ left, right ≤ COORD_CEILING`; canonical scalar `ZMod (2^64)` under `denote`. The corpus
theorems are `Apeiron.Table.get_build_acc`, `Apeiron.Table.build_gauge_invariance`,
`Apeiron.Table.get_insertWith_self`, `Apeiron.Table.get_insertWith_ne`.

**2 · Entry points and source identity.** Rust: `src/table.rs::WordMerge` (the concrete
`Merge<i64>` wrapping-add instance) and `src/scalar.rs::Ring` (whose `i64` instance is wrapping).
Song: `song/src/table/table.slang::mergeAcc` (pure accumulator merge with the in-tree laws
`mergeacc_comm`/`mergeacc_assoc`) and `tableInsertWith` (whose `MERGE_ACC` arm computes
`oldVal + value` inline). The inline arm, not `mergeAcc`, is the executing merge.

**3 · Representation invariant.** A scalar is its `i64` bit pattern. Rust wrapping add denotes
addition in `ZMod (2^64)`. Song `mergeAcc(left, right) = left + right` is that same addition on the
matched bounded domain. The constructor is `src/table.rs::WordMerge` / `impl Ring for i64` on the
Rust side and `mergeAcc` / the `MERGE_ACC` arm on the Song side. Rust and Song agree only when the
two additions agree on the matched domain; that agreement is not yet proved for the Song side.

**4 · Admission predicates.** Rust: the merge admits every `(i64, i64)`; `Ring::add` is total
wrapping; there is no merge-level refusal and admission is inherited from the table write. Song:
`mergeAcc` admits every pair; the `MERGE_ACC` arm is reached only when `tableCapacityOk(cap)` is
nonzero, `key` is not `TABLE_EMPTY` (-1), and `merge` is in `MERGE_SET..MERGE_ACC`. `mergeAcc`'s law
givens require `0 ≤ left, right ≤ COORD_CEILING`; the inline arm carries no such guard.

**5 · Behaviour.** Success: Rust `new.wrapping_add(old)`; Song writes `oldVal + value` and returns
the new value. Unsupported input: none at the merge on either side; a Song value outside
`0..COORD_CEILING` is outside the matched domain but the inline arm still computes. Resource
exhaustion: not applicable, one addition either fits the word or wraps. Raw return encoding: a bare
`i64` word, no separate success tag. Post-state: Rust merge is pure; Song `MERGE_ACC` writes the
merged value into the slot, while `mergeAcc` alone is pure.

**6 · Resource accounting.** Rust: one wrapping addition per merge call; no probes, nodes or stack
entries; two scalar arguments. Song: one addition in `mergeAcc`, the inline `MERGE_ACC` arm
computing `oldVal + value`; no probes, nodes or stack entries; two scalar arguments. These are
counted separately from R2's probe, node and stack costs.

**7 · Source-connected refinement and rejecting control.** Rust:
`proof/Denotation.lean::denote_wrapping_add` proves the wrapping word addition is `ZMod (2^64)`
addition, but the extracted `WordMerge` has no dedicated bridge theorem, so the Rust merge is
source-connected only through that denotation. Song: `mergeacc_comm` and `mergeacc_assoc` are
in-tree Song laws, not canonical Lean; no checked Song-to-Lean refinement exists. The joint
refinement is OPEN. Rejecting control: change `new.wrapping_add(old)` to `new.wrapping_sub(old)` in a
throwaway copy of `src/table.rs`, re-extract, and `proof/regen.sh --check` fails on the
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

**2 · Entry points and source identity.** Rust `src/table.rs`: `Table` (fixed-shard inline table),
`new`, `get`, `insert_with`, `build`, `StorageError` (`ShardFull`), `WordHome`, `Home`, `Merge`. Song
`song/src/table/table.slang`: `tableInit`, `tableInitGo`, `tableCapacityOk`, `tableFind`,
`tableFindSlotGo`, `tableLookupSlotGo`, `tableGet`, `tableInsertWith`, `tableSpread`, `tableSlotKey`,
`tableSlotVal`, `tableSlotSet`, `tableSlotValSet`.

**3 · Representation invariant.** Rust constructor `Table::new(default, home)` sets
`shards = [None; 256]` and stores `default`. The invariant `Rep(t, h)` is the fixed outer
`[Option<[Option<(K, V)>; N]>; 256]`, an untouched shard `None`, globally unique keys, every entry in
its home shard `home(k) mod 256`, and `∀ k, D(t)(k) = h(k)` with `D` the matching override or the
stored default. The constructor establishes `Rep(t, fun _ => d)` at `N = 16` (`word_table_new_rep`);
the general-`N` clause is open. Song constructor `tableInit`/`tableInitGo` writes key
`TABLE_EMPTY` and value `0` into every slot; the Song invariant is `memory[base + 2*slot]` = key
(`TABLE_EMPTY = -1` for empty), `memory[base + 2*slot + 1]` = value. No Song `Rep` theorem exists.

**4 · Admission predicates.** Rust: `insert_with` refuses with `StorageError::ShardFull` exactly
when the key is absent and its home shard has no vacant slot; `N = 0` refuses every write; reads are
total; `build` refuses at the first refused insertion. Refusal invokes neither clone nor merge and
changes no field. Song: `tableInsertWith` returns `TABLE_EMPTY = -1` when `tableCapacityOk(cap)` is
0, when `key` is `TABLE_EMPTY` (-1), when the merge code is outside `MERGE_SET..MERGE_ACC`, or when
`tableFindSlotGo` exhausts the `cap` probes; `tableGet`/`tableFind` return -1 for a missing key.
Joint success requires both; the Rust shard capacity and the Song power-of-two capacity are
different, so the refusals need not coincide.

**5 · Behaviour.** Success: Rust `Ok(())`, existing key updates `stored = merge(value,
stored.clone())`, new key inserts `merge(value, default.clone())` in the home shard. Song returns
the merged value, updating the slot for an existing key or filling the first empty probe slot for a
new key (`TABLE_PRESENT = 1` under `MERGE_SET`). Unsupported input: Rust `N = 0` write; Song
non-positive or non-power-of-two `cap`, key -1, or merge code outside `1..3`. Resource exhaustion:
Rust `ShardFull` when the home shard holds `N` absent-key entries; Song -1 when all `cap` probes are
occupied. Raw return encoding: Rust `Result<(), StorageError>`; Song a raw word where -1
(`TABLE_EMPTY`) is overloaded across missing read, empty slot, invalid capacity, reserved key,
invalid merge code and full table, and a successful `MERGE_SET` write returns 1. Post-state: Rust
refusal changes no field (`word_table_shardFull_refusal` at `N = 16`); Song writes in place, so a
mid-stream full table leaves earlier writes in place (H3).

**6 · Resource accounting.** Rust: one wrapping add per successful insert plus home callback
arithmetic; up to `N` full-key comparisons per `find_slots` per shard; 0 heap nodes because storage
is inline; constant call depth with object size growing with `N`; `build` consumes `kvs` in order
and clones each key and value. Song: `tableSpread` one multiply and one `and` plus one add for
`MERGE_ACC`; up to `cap` probe steps with measure `cap - step`; 0 allocated nodes (caller or static
memory, the `cells(32)` test arena); recursion depth bounded by `cap`; no input consumption because
the primitive reads and writes caller `memory[]`. The Rust and Song accounts are separate and are
not converted to a common fuel.

**7 · Source-connected refinement and rejecting control.** Rust:
`proof/corpus/TableRefinement.lean` proves `word_table_new_rep`, `word_table_get_new_first`,
`word_table_get_replaced`, `word_table_build_get_acc` and `word_table_shardFull_refusal` against
`Apeiron.Table` at the concrete `N = 16` word instance; `Rep` there is read-only extensional and
universal `Rep`/admission completeness is open. Song: no checked Song-to-Lean refinement of
`tableInit`, `tableFind`, `tableGet` or `tableInsertWith` exists. The joint row is OPEN and is the
first joint storage target. Rejecting control: the same-filesystem throwaway generated-body mutation
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
| B6 | Rust `src/table.rs::Table::build` is transactional; Song has only per-insert `tableInsertWith` plus `tableInit`. | Song's insertion loop is not Rust's transactional build; a refused insertion leaves earlier writes in place. | blocker |

## Hazards

| ID | Hazard | Consequence |
|---|---|---|
| H1 | Song reserves key -1 (`TABLE_EMPTY`) and returns -1 for a missing read; Rust does not. | Song's raw return is not a reliable success/refusal tag; -1 conflates missing read, empty slot, invalid capacity, reserved key, invalid merge code and full table. |
| H2 | Rust `default` is a stored value for any unwritten key; Song `tableGet` returns -1 for a missing key and has no stored default. | The canonical `HMap` total default is not represented by Song's missing-read; the matched domain must fix the default and exclude the reserved key. |
| H3 | Song `tableInsertWith` mutates slots in place; Rust `build` returns `Err` with no partial table. | Song's multi-insert is not transactional; a mid-stream full table leaves partial writes. This is B6. |
| H4 | Canonical `I` is `S K K`, while Song treats primitive `i` and the normal form `**skk` as distinct. | A regression fixture must pin the distinction; it is not patched away by conflating the primitive and the normal form. |
| H5 | Song `certcheck` must refine `Apeiron.Decider.checkCert`/`checkCert_sound`. | Until that refinement exists, Song's UNSAT replay is not the canonical checker. |

## Integration order

1. R1 word arithmetic: fix the matched scalar domain (`0 ≤ v ≤ COORD_CEILING`, key -1 excluded) and
   the shared addition observation.
2. R2 table init/read/write: the first joint storage target; define the matched admission relation
   and a Song `Rep` invariant.
3. B6 table build composition: define a transactional Song build or restrict the admitted domain.
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
- Arbitrary-`N` Rust `Rep` preservation and admission completeness.
- Song-to-Lean refinement for R1 and R2.

## Checker

`python3 tools/joint/profile_check.py --check` validates `tools/joint/profile_rows.json` against this
document and the pinned sources: every row carries all seven fields, every named canonical theorem
resolves as a declaration in the pinned corpus, every named Rust and Song entry exists as a
declaration, every required unmatched blocker and hazard is present, and the resource accounting
names the five distinct kinds. It exits nonzero on a missing field, row, blocker, hazard, theorem or
source entry. `--controls` and `--self-test` mutate a copy in memory (remove a required field, remove
an unmatched blocker, remove a row, rename a theorem to a nonexistent name, point a source entry at
a missing file, rename a resource kind to `fuel`) and require each mutation to be rejected; the
working tree is never touched. The checker establishes specification consistency only.
