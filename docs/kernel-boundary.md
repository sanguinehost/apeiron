# Apeiron Local Boundary: Representation And Trust

This document owns Apeiron's local representation and trust boundary. The programme thesis,
ownership and outcomes are in the [programme direction](direction.md); this file does not restate
them. Local authority: the fixed-shard `Table` representation, its canonical `Rep` relation, the
translation/trust inventory and the kernel scope below are owned here. The programme authority is
`docs/direction.md`; the execution sequence is in the [operator handoff](flash-handoff.md).

## Canonical object and executing root

The canonical object is `Apeiron.Table.HMap`, `Apeiron.Table.get`, `Apeiron.Table.insertWith` and
`Apeiron.Table.build`, together with `Apeiron.Trichotomy.AnyRing.RingCell`,
`Apeiron.Trichotomy.AnyRing.ringMul` and `Apeiron.Trichotomy.AnyRing.ringNormForm`. The extracted
`Cell<i64>` word bodies are `word_mul_denote` and `word_norm_denote`.

The basis is `R[x]/(x^2 - sigma)`: `Cell<R>` with `Sigma` as a value in every executing signature,
`Coord` addressing correct per face, and the fixed-shard `Table<K, V, H, N>` (one home shard,
`ShardFull` refusal, `N = 0` refusal, no probing). Every executing face refines the canonical
`Apeiron.Table.HMap` object; a face that names it only by path, namespace or import is not rooted.

## Layer separation

The following layers are distinct; none silently stands for another:

- Canonical semantics: `HMap`, `get`, `insertWith`, `build`.
- Face-specific key interpretation: elliptic exact packing, parabolic carried key, hyperbolic
  scramble/discard, each addressed by `Coord` per face.
- Fixed-shard admission: `ShardFull` and `N = 0` refusal, one home shard, no probing.
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
physical representation is the fixed-capacity sharded array with the private invariant `Rep`. These
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
- The manifest resolves corpus Git `8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f` and Mathlib
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
| Slang source refinement (Stream 0) | Apeiron | in zero-axiom migration; empty-cone gate `tools/slang/zero_axioms.py`; current W23/W25 evidence reopened |
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
