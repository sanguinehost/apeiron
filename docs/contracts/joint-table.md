# Joint Table Subprofile

This file is the detailed table subprofile of the joint primitive profile for
`APEIRON-PROGRAMME-W18-T3`. It takes the table row of
[`joint-primitives.md`](joint-primitives.md) and fixes the selected merge, the layout, the
admission predicates, the resource accounts and the rejecting controls. The machine-readable form is
[`tools/joint/table_profile_rows.json`](../../tools/joint/table_profile_rows.json); the checker
[`tools/joint/table_profile_check.py`](../../tools/joint/table_profile_check.py) establishes
specification consistency only.

Authority: [`docs/direction.md`](../direction.md#primitive-correspondence) is the programme
authority; [`docs/kernel-boundary.md`](../kernel-boundary.md) owns the Rust representation and trust
boundary; [`proof/README.md`](../../proof/README.md) owns extraction and bridge scope. The canonical
corpus remains at `../sanguine-apeiron/proof/` at Git
`8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f`, Mathlib
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. The Apeiron sources were read at revision
`9a805439640ff63b5a2028aeb887f50e31cf6351` and the Song sources at revision
`47869e76f982cfa3085a8618a82b80463e22ef46`; the per-file digests are recorded under
[Pins and source digests](#pins-and-source-digests). The canonical authority for every row is
`Apeiron/Table/Basic.lean`.

This document specifies obligations. It does not claim the correspondence exists. No row is closed:
every row below is `selected-open`. The checker establishes specification consistency only; it never
reports correspondence, refinement or proof.

## Required fields

Every row carries these six fields. A row that drops one is incomplete and the checker rejects it.

| Field | Obligation |
|---|---|
| `canonical` | The canonical object, the module `Apeiron/Table/Basic.lean`, and the exact declaration targets per obligation. |
| `rust` | The Rust layout, entry points, full-key identity, the selected merge, refusal and bounds. |
| `song` | The Song layout, entry points, stride, keys, probes, merge and transactionality. |
| `admission` | Separate Rust and Song admission predicates. Joint success requires both; unequal capacities need not give identical refusals. |
| `resources` | Separate Rust and Song resource accounting. Arithmetic ops, probes, nodes, stack entries and input consumption are counted separately and are not converted to a common fuel. |
| `refinement` | A source-connected refinement and one named rejecting control. A copied model or a differential suite alone does not close a row. |

## Obligation rows

| Row | Obligation | Canonical targets | Status |
|---|---|---|---|
| T1 | Fixed-shard layout and representation | `HMap`, `get` | selected-open |
| T2 | Read | `get`, `get_insertWith_self` | selected-open |
| T3 | Replacement write | `insertWith`, `get_insertWith_self`, `get_insertWith_ne`, `merge_is_the_only_freedom` | selected-open |
| T4 | Refusal, post-state and mutation | `insertWith`, `build` | selected-open |
| T5 | Separate resource accounts | `get`, `build` | selected-open |
| T6 | Build composition | `build`, `get_build_acc`, `build_gauge_invariance`, `build_eq_of_key_streams_eq` | selected-open |

Canonical authority: `Apeiron/Table/Basic.lean` declares `HMap`, `get`, `insertWith`, `build`,
`get_insertWith_self`, `get_insertWith_ne`, `get_build_acc` and `build_gauge_invariance`.
`build_gauge_invariance` is a theorem only with its actual hypotheses `hcomm : ∀ a b, f a b = f b a`
and `hassoc : ∀ a b c, f (f a b) c = f a (f b c)`; without both, insertion order is part of the
meaning. The checker rejects the theorem target if either hypothesis is absent from the recorded
target.

### T1 · Fixed-shard layout and representation

**Canonical.** `Apeiron.Table.HMap K V := K → V` is a total function; absence is the stored default,
not a partial lookup, and reading is `Apeiron.Table.get`.

**Rust.** `Table<K, V, H, N>` holds `default: V`, `home: H` and
`shards: [Option<[Option<(K, V)>; N]>; 256]`: 256 inline shards, each an optional array of at most
`N` key/value slots. Storage never allocates. Placement is `home(key) mod 256` (`shard_index`);
identity is full-key equality in `find_slots` (`stored == key`), so shard truncation selects a shard
and never replaces full-key identity. Every entry sits in its home shard and keys are globally
unique. The default `N` is 16.

**Song.** `tableInit` / `tableInitGo` write key `TABLE_EMPTY` and value `0` into `cap` slots. Each
slot is two consecutive cells, key at `base + 2*slot` and value at `base + 2*slot + 1`
(`TABLE_STRIDE = 2`), so initialization writes `2*cap` cells. A live slot has a key other than
`TABLE_EMPTY`; `TABLE_EMPTY = -1` is the reserved empty key, so `-1` is not an admissible live key.
Live keys are unique. `tableCapacityOk(cap)` checks only that `cap` is a positive power of two;
the backing store being non-aliasing and the address extent `base + 2*cap ≤ COORD_CEILING` are
separate caller obligations, not established by `tableCapacityOk`.

**Admission.** Rust refuses with `StorageError::ShardFull` exactly when the key is absent and its
home shard has no vacant slot; `N = 0` refuses every write; reads are total. Song refuses with
`TABLE_EMPTY = -1` on invalid capacity, reserved key, out-of-range merge code, or probe exhaustion.

**Resources.** Rust: 0 heap nodes, constant call depth with object size growing with `N`. Song: 0
allocated nodes, initialization recursion bounded by `cap`.

**Refinement.** Rust `word_table_new_rep` and `word_table_new_eq_empty` prove the constructor at the
concrete `N = 16` word instance. The proved relation `Rep` is the read-only extensional relation at
the public read boundary, `∀ key, word_table_get t key ⦃ value => value = h key ⦄`; no physical
layout invariant is claimed. The general-`N` read-only extensional `Rep` clause is open. Song has no
checked Song-to-Lean refinement of `tableInit` or the slot invariant. Control C1.

### T2 · Read

**Canonical.** `get` reads the value at a key; the canonical table is total, so an unwritten key
reads the default and a written key reads its stored value. `get_insertWith_self` is the read-back
equation at a written key.

**Rust.** `get` computes the home shard, scans it with full-key equality (`find_slots`), and returns
the stored override or `&self.default` on a miss. Reads are total; there is no read refusal and no
sentinel return. At most `N` full-key inspections per read.

**Song.** `tableGet` calls `tableFind` and reads `tableSlotVal` at the located slot; a missing key
returns `TABLE_EMPTY = -1`. There is no stored default, so an absent key and a key holding value
`-1` both return `-1`; the occupied key must be read separately. `tableFind` returns `-1` on an empty
slot or after `cap` probes.

**Admission.** Rust reads are total for every key. Song `tableGet` returns `-1` for a missing key and
for the reserved key; a `-1` raw return is overloaded and is not a status.

**Resources.** Rust: up to `N` full-key comparisons per read, plus one home callback and residue.
Song: up to `cap` steps of `tableLookupSlotGo`, measure `cap - step`, plus one `tableSpread`
multiply and mask per read.

**Refinement.** Rust `word_table_get_new_first` and `word_table_get_replaced` connect the generated
read to `Apeiron.Table.get` on the checked word states. Song has no checked refinement of
`tableGet`. Control C5.

### T3 · Replacement write

**Canonical.** `insertWith f k v m := fun x => if x = k then f v (m x) else m x`. This subprofile
fixes `f` to replacement, `f v old = v`, so a write overwrites the stored value and leaves other keys
untouched. `get_insertWith_self` and `get_insertWith_ne` are the two read-back equations;
`merge_is_the_only_freedom` shows `insertWith` depends on `f` only through its values.

**Rust.** `insert_with` locates the key in its home shard, replaces the stored value for a present
key, or installs the merged value at the first vacant slot for an absent key. The selected merge is
replacement, realized by a `Merge<V>` instance; `WordMerge` (wrapping add) is the R1 merge and is
excluded here. A present key always updates, even in a full shard. A same-residue collision is a
different key.

**Song.** `tableInsertWith` probes with `tableFindSlotGo`, then either updates the located present
slot or fills the first empty probe slot. `MERGE_MAP` computes `newVal = value`, the replacement
merge. `MERGE_SET` (presence `1`) and `MERGE_ACC` are outside the matched merge. `MERGE_ACC` adds
`oldVal + value` only on the existing-key path; a fresh (absent) key stores `value` unchanged, so the
first accumulator insert does not add the default. The reserved key `-1` is refused before probing.
At most `cap` probes.

**Admission.** Rust: the replacement `Merge` admits every `(i64, i64)`; admission is inherited from
`insert_with` (a present key always admits; an absent key admits iff its home shard has a vacant
slot). Song with `MERGE_MAP` admits only when `tableCapacityOk(cap)` is nonzero, `key` is not
`TABLE_EMPTY`, the merge code is in range, and `tableFindSlotGo` returns a slot.

**Resources.** Rust: one full-key scan of the home shard, at most `N` comparisons, plus one home
callback and residue; replacement merge is constant work. Song: up to `cap` steps of
`tableFindSlotGo`, measure `cap - step`, plus one `tableSpread` multiply and mask; `MERGE_MAP`
assigns `value`.

**Refinement.** Rust `table_insert_empty_zero` and `table_insert_present_zero` are `WordMerge`
wrapping-add witnesses: they exercise the generated insertion with the wrapping-add callback, and are
scoped to that callback; they do not refine the replacement merge this subprofile selects. The
replacement-merge source-connected refinement is established at `N = 16` by
`Apeiron.Joint.TableInstance.replacement_insert_with`, with
`Apeiron.Joint.TableInstance.replacement_new_readRep` and
`Apeiron.Joint.TableInstance.replacement_get` for the read relation, in
`proof/corpus/Joint/TableInstance.lean` against canonical `Apeiron.Table.insertWith`. The universal
generic physical invariant and admitted insertion/build preservation are proved under
`CallbackLawsG`. Song's empty-cone `Iota.Joint.songTable_replacement_link` proves admitted
`MERGE_MAP` execution preserves `RepSong` for canonical replacement. It is still parameterized by
context lookup/shape hypotheses rather than instantiated from the decoded source, and no checked
`I64 ↔ Iota.Word` map transport or combined Rust–Song theorem exists. The joint refinement is
OPEN. Controls C1 and C3.

### T4 · Refusal, post-state and mutation

**Canonical.** The canonical `insertWith` is total and has no refusal. On success the implementation
post-state denotes `insertWith`; on resource exhaustion the implementations refuse and the profile
states whether state changes. `build` is the only canonical composition form.

**Rust.** A refusal returns `StorageError::ShardFull` and the table value is unchanged
(`word_table_shardFull_refusal` at `N = 16`). A refusal is for an absent key only; a present key in a
full shard updates rather than refuses. Refusal invokes neither clone nor merge. For an absent key
in an initialized full shard, the refusal path performs `N` full-key search inspections followed by
`N` vacancy inspections, then returns without writing.

**Song.** A refusal returns `TABLE_EMPTY = -1`. Because `tableInsertWith` writes in place, the
insertion loop mutates its prefix: a refusal after earlier writes leaves those writes in place. The
refused call itself writes no cell, but the caller's earlier writes remain. This is not
transactional, and it is the B6 composition blocker. No deletion operation is selected.

**Admission.** Separate predicates; joint success requires both; unequal capacities need not give
identical refusals.

**Resources.** Rust: `N` full-key search inspections followed by `N` vacancy inspections on the
refusal path, no merge arithmetic. Song: at most `cap` probe steps before refusal, no merge
arithmetic.

**Refinement.** Rust `word_table_shardFull_refusal` proves the absent-key full-shard refusal with an
unchanged state at `N = 16`. Song has no checked refinement of the refusal post-state. Controls C4
and C6.

### T5 · Separate resource accounts

**Canonical.** The canonical definitions are total and resource-free; resource bounds belong to each
implementation and are not identified across layouts. No canonical theorem carries a cost.

**Rust.** Cost is counted on the inline fixed-shard table: at most `N` full-key search inspections
(`find_slots`), then up to `N` vacancy inspections. The replacement merge is constant work; the
excluded `WordMerge` would add one wrapping addition per successful insert. A full shard costs the
full `N` vacancy inspections before refusing. 0 heap nodes; constant call depth with object size
growing with `N`.

**Song.** Cost is counted on the caller memory table: at most `cap` probe steps
(`tableFindSlotGo` / `tableLookupSlotGo`), measure `cap - step`; recursion depth bounded by `cap`; 0
allocated nodes; no input consumption because the primitive reads and writes caller `memory[]`.

**Admission.** The bound is part of admission only through the Rust `N`-slot shard capacity and the
Song `cap`-probe exhaustion; it does not admit or refuse independently.

**Resources.** The five kinds `arithmetic_ops`, `probes`, `nodes`, `stack_entries` and
`input_consumption` are counted separately for Rust and for Song. They are not converted to a common
fuel.

**Refinement.** Rust `find_slots_spec` and `insert_full_loop_spec` prove the `N`-inspection search
and the `N`-vacancy refusal loop at `N = 16`. Song `tablefindslotgo_bound` and
`tablelookupslotgo_bound` are in-tree Song laws, not canonical Lean. No joint resource refinement
exists. Control C2.

### T6 · Build composition

**Canonical.** `build f d kvs` folds `insertWith` over the stream from the constant default.
`get_build_acc` gives the per-key fold in stream order; `build_gauge_invariance` licenses reordering
only under `hcomm` and `hassoc`; `build_eq_of_key_streams_eq` licenses cross-key reordering under
every merge.

**Rust.** `build` constructs a fresh table and folds `insert_with` over `kvs` in order. It returns
`Err(StorageError::ShardFull)` at the first refused insertion; no partial table is returned. The
built table is the per-key replacement fold, so the last value at a key wins. `build` consumes `kvs`
in order and clones each key and value.

**Song.** Song has only per-insert `tableInsertWith` plus `tableInit`; there is no transactional
build. The insertion loop mutates its prefix and is not transactional. This is the B6 composition
blocker, not a matched primitive.

**Admission.** Rust `build` admits a stream iff every prefix insertion is admitted; the first
refusal makes the whole build `Err`. Song has no build admission predicate; each `tableInsertWith`
call is admitted independently.

**Resources.** Rust: consumes `kvs` in order and clones each key and value; 0 heap nodes. Song: no
input consumption; the caller drives the loop and owns `memory[]`; 0 allocated nodes.

**Refinement.** Rust `word_table_build_get_acc` and `word_table_build_ordered` are `WordMerge`
wrapping-add witnesses, scoped to the wrapping-add callback; they are not replacement-merge
refinements. The replacement-merge build composition is established at `N = 16` by
`Apeiron.Joint.TableInstance.replacement_build` and
`Apeiron.Joint.TableInstance.replacement_build_get_acc`, with
`Apeiron.Joint.TableInstance.replacement_shardFull_refusal` covering the refusal branch, in
`proof/corpus/Joint/TableInstance.lean` against canonical `Apeiron.Table.build`. The universal
generic physical build postcondition is proved under `CallbackLawsG` and `BuildAdmissibleG`. Song
has no checked refinement of an insertion loop. `build` stays
outside the matched primitive boundary: it is the B6 composition obligation, not a matched primitive.
Control C3.

## Separate admission

The Rust and Song admission predicates are separate, and joint success requires both.

- **Rust.** `insert_with` refuses with `StorageError::ShardFull` exactly when the key is absent and
  its home shard has no vacant slot; `N = 0` refuses every write; reads are total.
- **Song.** `tableInsertWith` refuses with `TABLE_EMPTY = -1` on invalid capacity, reserved key,
  out-of-range merge code, or probe exhaustion; `tableGet` and `tableFind` refuse a missing key with
  `-1`.
- **Joint.** A write is matched only when the Rust `insert_with` returns `Ok` and the Song
  `tableInsertWith` returns the merged value for the same key and value.
- **Unequal capacities.** The Rust shard capacity `N` and the Song power-of-two capacity `cap` are
  different resources, so unequal capacities need not give identical refusals. A key admitted by one
  side may be refused by the other; the matched domain is the intersection, not either predicate
  alone.

## Hazards

| ID | Hazard | Consequence |
|---|---|---|
| H1 | Song returns `-1` for a missing read while Rust returns the stored default. Song's raw return is not a reliable success/refusal tag: `-1` conflates a missing read, an empty slot, invalid capacity, the reserved key, an out-of-range merge code and a full table. | A consumer that treats the raw return as a status misclassifies a missing read as a refusal, or a refusal as a read. The matched status relation is open. |
| H2 | Song's checked `observeRead` relation separates the raw word from occupied-key state, but no joint adapter transports that state to Rust's `I64` map. | Until the carrier/map transport and composite theorem exist, the joint success/refusal relation is not established, and no row closes on the raw return alone. |
| H3 | Values equal to `-1` are legitimate: a live key may hold value `-1`, and `MERGE_MAP` can store it. | A `-1` return cannot be read as missing without checking the occupied key separately; a sentinel-value success would be misclassified as a missing read. |
| H4 | The occupied-key set must be retained separately from the extensional map. Rust distinguishes an unwritten key (default) from a written key whose value equals the default; Song's extensional read alone cannot. | The joint representation must carry occupied keys, not only the map, or the correspondence is false for a written key whose value equals the default. |

## Rejecting controls

Each control is a spec-level mutation to run in a throwaway copy. The checker verifies only that the
control is named, well formed and attached to a row; it does not execute the control and does not
claim the control goes red. A control that cannot fail has no weight.

| ID | Control | Target | Mutation | Expected red |
|---|---|---|---|---|
| C1 | collision-key substitution | T1, T3 | Substitute a key with the same home residue (for example `0 -> 256`) for a present key in the Rust and Song sources. | The read and the update must locate the original full key and must not report or overwrite the collision key; Rust `oneSlots_find` and `table_insert_present_zero`, Song probe reachability. |
| C2 | lost probe continuation | T4, T5 | Remove the `step + 1` continuation in `tableFindSlotGo` and `tableLookupSlotGo` so each probe returns after one step. | A key placed after a collision becomes unreachable; `tablefindslotgo_bound`, `tablelookupslotgo_bound` and the probe-reachability check fail. |
| C3 | wrong fresh merge/default | T3, T6 | Change the Rust fresh write `merge value default` to `merge default value`; make Song `MERGE_MAP` merge with the old value instead of replacing. | `table_insert_empty_zero` fails; the Song replacement face no longer denotes `insertWith`. |
| C4 | full-table existing-key refusal | T3, T4 | Make a full home shard refuse an existing key instead of updating it. | The existing-key update in a full shard must still succeed; the Rust test `full_shard_refuses_unchanged_but_existing_key_still_updates` and `insert_with_present_exact` fail. |
| C5 | sentinel-value success misclassification | T2, T4 | Classify a `-1` raw return as missing regardless of the occupied key. | A legitimate stored `-1` under `MERGE_MAP` is misread as missing; the adapter status check and the sentinel-value fixture fail. |
| C6 | mutation on refusal | T4 | Write a field on a Rust `ShardFull` refusal, or a slot on a refused Song insert. | The refusal post-state must equal the pre-state; Rust `word_table_shardFull_refusal` fails and the Song refused-call post-state check fails. |

## Not covered

- rustc/LLVM code generation and the generated machine code.
- The kernel ABI, stack and OS.
- GPU / SPIR-V.
- Runtime linkage between Apeiron and Song; the required result is a checked correspondence, not
  linkage.
- Song local registers, ontology and commands.
- Semantic correspondence itself: this profile and its checker establish specification consistency
  only.
- Callback behavior outside the explicit Rust `CallbackLawsG` contract.
- An unconditional decoded-source instantiation of the Song table theorem or a theorem composing
  the Rust and Song replacement refinements. `Joint.WordBridge` checks the `I64 ↔ Iota.Word`
  pointwise-map relation and its preservation by new-first replacement.
- Deletion: no deletion operation is selected by this subprofile.
- Song backing/nonaliasing and address extent beyond `tableCapacityOk`; those remain separate caller
  obligations.

## Pins and source digests

The checker resolves and verifies these pins; a drift in any revision or digest fails `--check`.

| Item | Revision |
|---|---|
| Canonical corpus (`../sanguine-apeiron/proof`) | `8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f` |
| Mathlib | `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f` |
| Apeiron sources | `9a805439640ff63b5a2028aeb887f50e31cf6351` |
| Song sources | `47869e76f982cfa3085a8618a82b80463e22ef46` |

| Source | sha256 |
|---|---|
| `src/table.rs` | `f30e31e46c18680283d0333ff04a679d519e904cf0958e86763372a639b074e3` |
| `song/src/table/table.slang` | `bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d` |
| `proof/corpus/TableRefinement.lean` | `fcf322ccb1c743097105514bfef422c03ee9d58ee690512a9cf0d3760b1a9100` |
| `../sanguine-apeiron/proof/Apeiron/Table/Basic.lean` | `e7558bac1d255643f58100f5a1adf32926a423fe6b49e7c991f8d3a3b0eb26b9` |

## Checker

`python3 tools/joint/table_profile_check.py --check` validates
`tools/joint/table_profile_rows.json` against this document and the pinned sources: every row carries
all six fields and stays `selected-open`; every named canonical target resolves as the exact
fully-qualified declaration (name and kind) in the pinned corpus, so a merely opened prefix or a
missing namespace is rejected; every named Rust and Song entry exists as a concrete executing
declaration (a `law`/`carrier`/`spec` header alone is not an entry); the corpus Git revision, the
Mathlib revision resolved from a manifest, and per-source sha256 digests match the resolved checkout,
and each recorded digest is bound to the blob at the recorded revision; every implementation of every
row carries its required resource kinds separately, and no generic `fuel` is admitted; every named
reference in the structured prose fields and in this document, including control targets, resolves,
and a reference that does not resolve is rejected unless its exact path is in the checker's explicit
missing-artifact set; each row's Markdown fields are present with nonempty bodies and its Markdown
status agrees with the JSON status; every required row, hazard and control is present; and this
document carries the actual records and fields, not incidental mentions. It exits nonzero on a
missing field, row, hazard, control, target or source entry, on a declaration-name or namespace
mismatch, on a corpus or digest drift, on an unresolved named reference, on a Markdown field or status
that disagrees with the JSON, or on a row that is not `selected-open`.

`--controls` and `--self-test` mutate a deep copy of the loaded profile and the spec text in memory
and require each mutation to be rejected. The controls target the false-green classes: a missing
resource account on one implementation; a wrong corpus revision and a changed source digest; a
reversed canonical namespace and a canonical target missing its `Table` namespace; a Song entry that
is only a `law` header; a deleted required-field record, a deleted or emptied row field and a deleted
hazard record in the spec; a `closed` row and a Markdown row status that disagrees with the JSON; an
unknown control target and an unresolved named reference in a row field. The retained structural
controls remove a required field, drop a field from the fields list, remove a required row, hazard or
control, rename a canonical target to a nonexistent name, mislabel a def as a theorem, point a Rust
entry at a missing file, point a Song entry at a missing symbol, rename a resource kind to `fuel`,
remove the joint admission clause and drop the `get_build_acc` target. The working tree is never
touched. The checker establishes specification consistency only; it never reports correspondence.
