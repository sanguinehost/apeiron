# Apeiron Local Boundary: Representation And Trust

This document owns Apeiron's local representation and trust boundary. The programme thesis,
ownership and outcomes are in the [programme direction](direction.md); this file does not restate
them. Local authority: the fixed-shard `Table` representation, its canonical `Rep` relation, the
translation/trust inventory and the kernel scope below are owned here. The programme authority is
`docs/direction.md`; the execution sequence is in the [operator handoff](flash-handoff.md).

## The executing representation

One type: `apeiron::Table<K, V, H, const N: usize = 16>`, defined in `src/table.rs`; no aliases or
wrapper containers. `src/map.rs`, `src/safe_table.rs` and `src/set.rs` are deleted, including `Map`,
`Words`, `Exposed`, `ExposedWords`, `SafeTable`, `TableFull`, `Set`, `WordSet`, `ExposedSet` and the
map entry/iterator/storage-policy types. The former `CountTable`, regions, scanner and worker code
are removed from `src/table.rs`, not retained under a second name. Algebra modules remain.

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
- `Rep(t, h)` is the required physical invariant (a target, not yet the general proved relation):
  the fixed outer array, optional length-`N` shard arrays, globally unique keys, every entry in its
  home shard, and `∀ k, D(t)(k) = h(k)` through key/value denotations when Rust and Lean types
  differ.
- The obligations to prove are: constructors establish `Rep(t, fun _ => d)`; reads denote
  `Apeiron.Table.get`; successful writes establish `Rep(t', Apeiron.Table.insertWith f k v h)`.
  These are desired general obligations, not general proved facts. The actual `TableRefinement.Rep`
  today is an extensional relation instantiated at the concrete word instance `N = 16`.
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
physical representation is the fixed-capacity sharded array with the private invariant `Rep`. These
are distinct. `HMap` is the shared semantic object; `Rep` is the physical invariant under which a
particular table state denotes it. `Rep` is not the public contract and is not exposed; the public
contract is the four methods and their refusal semantics.

`TableRefinement.Rep` is already a valid scoped relation at the concrete word instance `N = 16`.
Quantifying the invariant and its preservation over arbitrary `N`, keys and streams is the remaining
target, not a precondition for the scoped relation to be meaningful. The current evidence is a set
of checked word-instance witnesses, not a universal preservation theorem.

## Current evidence at actual scope

- The current `Table` proofs cover the constructor, the default read, first/present writes at key 0,
  the ordered two-write build, the bounded 16-slot search and the concrete full home-shard refusal.
  They do **not** establish arbitrary `Rep` preservation, admission completeness or physical
  subsumption.
- The named witnesses in `proof/corpus/TableRefinement.lean` are `word_table_new_rep`,
  `emptyWordTable_read`, `word_table_insert_new_first`, `word_table_get_new_first`,
  `word_table_insert_replaces_present`, `word_table_get_replaced`, `word_table_build_ordered`,
  `word_table_build_get_acc`, `word_table_shardFull_refusal` and
  `word_table_joint_branch_witness`.
- The real `new`, `get`, `insert_with` and `build` bodies occur in `proof/ApeironWord.lean`, and
  `proof/regen.sh --check` accepts the real generic bodies and concrete word roots. Extraction is not
  a proof of termination, `Rep`, or equality to `HMap`.
- The manifest resolves corpus Git `50c94192399add3b9f8e64bd14e36d8cba270705` and Mathlib
  `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. A source-proof run log is historical evidence, not
  fresh acceptance.

## Rust and Slang correspondence

Rust and Slang have different physical layouts. Different layouts can correspond, but only under a
matched admission relation: the same canonical definition, matched domains, and explicit
representation, state/effect, refusal and resource obligations. A worked example, a differential
test, identical names or identical layouts are not the correspondence proof. The cross-repo
primitive correspondence is a required programme outcome and is not established today; see the
[programme direction](direction.md#primitive-correspondence).

## Choice and trust cost (anchor)

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

## Trust inventory

| Layer | Owner | Status |
|---|---|---|
| Canonical algebra and combinator definitions | Apeiron (Lean corpus) | proved at stated scope (`Apeiron.Combinator.Comb.reduceDeep_sound`, `Apeiron.Trichotomy.AnyRing.ringNormForm_mul`) |
| Rust `Cell<i64>` kernels | Apeiron | refined to canonical definitions; axiom cone `{propext, Classical.choice, Quot.sound}` |
| Rust `Table` bodies | Apeiron | extracted and elaborated; word-instance witnesses proved; universal `Rep` open |
| Charon/Aeneas translation fidelity | pinned toolchain | external trust; patched for const `Repeat` to `Array.repeat` |
| Lean kernel | pinned toolchain | trusted |
| rustc/LLVM codegen and generated machine code | compiler | not covered |
| Foreign `unsafe`/ABI code outside the word claim, kernel ABI, stack/OS | platform | not covered by the table refinement |
| GPU / SPIR-V | target | not covered |
| Generic `Home`/`Eq`/`Clone`/`Merge`/drop | caller | separate instance obligations; may allocate, panic or diverge |

The executing table's universal preservation refinement remains open; the checked word-instance
witnesses above remain established at their stated scope. Neither a reference container nor a
tested-only optimized container is an accepted substitute. Former storage proofs remain historical
evidence about deleted source, not this table.

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
