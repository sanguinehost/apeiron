# Specification - Bounded Storage Refinement

- **Owner:** OpenCode
- **Intent:** `docs/intent/storage-refinement.md`
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean before this epic
- **Status:** approved

> **Record status.** The two-container content below (`SafeTable<K, V, N>`, `Map<K, V>`, the
> fixed-capacity-four reference) records retired source. The current executing representation is the
> fixed-shard `Table<K, V, H, N>`. Its face/key semantics, layer separation and cost boundary are
> stated here; the historical rows below are kept, not rewritten.

## Face and key semantics

The canonical object is `Apeiron.Table.HMap` (`K → V`) with `Apeiron.Table.get`,
`Apeiron.Table.insertWith` and `Apeiron.Table.build`. A face is one sign of `Sigma`, and it fixes how
a key becomes an address and what the stored entry carries. Face semantics interpret the canonical
object; they do not replace it.

| Face | `Sigma` | Address | Identity |
|---|---|---|---|
| Elliptic | `sigma < 0` | exact packing on an explicit admitted domain | the address determines the key on that domain |
| Parabolic | `sigma = 0` | a lossy spreading fold | the complete key is carried beside the address |
| Hyperbolic | `sigma > 0` | a scramble and a discard | the key is discarded on an explicit observation domain |

- **Elliptic: exact packing on an explicit admitted domain.** The address is a bijection
  (`Packed`/`unpack`) exactly for keys that fit the packed width. Outside that admitted domain the
  packing does not exist and the key is refused rather than packed lossily.
- **Parabolic: carried complete key.** The address spreads the key; it never decides identity. A
  coordinate or hash equality is a placement probe; the stored complete key is compared, and the
  coordinate never replaces that comparison.
- **Hyperbolic: scramble/discard on an explicit observation domain.** The address is a scrambler
  judged on distribution, and the discard is defined only on the observations the face retains. It
  makes no identity claim outside that domain.

Every executing address signature carries `Sigma` as a value. `Cell<R>` operations take the dial, and
`Coord` addressing is correct per face; a signature that omits `Sigma` selects no face and is not an
executing address signature.

`H: Home<K>` selects exactly one home shard, `home(key) mod 256`. There is no fallback probing: a
miss in the home shard is a miss. Insertion refuses with `ShardFull` exactly when the key is absent
and its home shard has no vacant slot; `N = 0` refuses every write and keeps reads total. Refusal
changes no field and invokes neither clone nor merge.

## Three layers

- **Canonical semantics.** `Apeiron.Table.HMap`, `get`, `insertWith`, `build`: the shared total map,
  its new-first write and its ordered build fold.
- **Physical fixed-shard invariant.** The inline `[Option<[Option<(K, V)>; N]>; 256]` state with
  `Rep`: one home shard per key, no probing, `ShardFull`/`N = 0` refusal. A physical state denotes the
  canonical map under `Rep`; `Rep` is the private invariant, not the public contract.
- **Source-checker memo store.** The source checker's memo store is a separate store with its own
  refinement obligation.

A result at one layer does not transfer to another without its own refinement.

## Fixed-N cost boundary

`N` is a compile-time constant, callbacks are bounded, and key length is not constant. Lookup and
update are bounded per operation at fixed `N` with bounded callbacks and nonconstant key length.
Reading a whole source remains at least linear in its bytes at fixed `N`, so constant-time table
operations at fixed `N` do not imply constant-time whole-source checking or constant-size proof
terms. No performance measurement is claimed here.

## Historical record (superseded): two-container bounded reference

The sections below describe the retired `SafeTable`/`Map` reference. They are historical evidence,
not the current table.

## Scope and admissible domain

| Item | Value |
|---|---|
| Operation | Safe fixed-capacity lookup, replace and removal |
| Domain | `K: Eq`, arbitrary `V`, compile-time slot count `N` |
| Totality | Lookup/removal total; insertion returns `Result` on full absent-key insertion |
| Preconditions | None; allocation failure is absent because storage is an inline array |
| Postcondition | Successful observations equal a total function `K -> Option V` |

The first executable instance uses word keys and values at a small fixed capacity for extraction.
This bounds proof cost without narrowing the generic Rust reference's domain.

## Representation relation

| Layer | Symbol | Path |
|---|---|---|
| Safe Rust reference | `SafeTable<K, V, N>` | `src/safe_table.rs` |
| Production comparison | `Map<K, V>` | `src/map.rs` |
| Extraction wrapper | Monomorphic word-table wrappers | `proof/extract/lib.rs` |
| Extracted body | Generated safe-table operations | `proof/ApeironWord.lean` |
| Canonical definition | `Apeiron.Table.get`, `insertWith`, `replace` | Pinned corpus |
| Refinement theorem | Named safe-table denotation theorems | `proof/corpus/WordRefinement.lean` |

The wrappers fix word types and capacity. They call the reference body and do not restate its
transitions.

## Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| Lookup | `get k` returns the unique stored value or `None`. | proved | Safe-table lookup refinement. |
| Replacement | Existing `k` changes to `v` and returns the old value. | proved | Canonical `insertWith replace`. |
| Fresh insertion | Absent `k` stores `v` when a slot is free and returns no old value. | proved | Canonical `insertWith replace`. |
| Full refusal | Absent `k` with `len = N` returns the original key/value and no successor. | proved | Extracted branch characterization and witness. |
| Removal | Present `k` becomes absent and returns its value; other keys are unchanged. | proved | Canonical replacement by `None`. |
| Production agreement | `Map` matches the safe reference after every selected operation. | tested | Differential trace test. |

## Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Fresh key with free slot | `Ok(None)` and committed successor | `len < N` |
| Existing key at any occupancy | `Ok(Some(old))` and committed successor | key present |
| Fresh key at capacity | `Err(Full { key, value })` | key absent and `len = N` |
| Missing removal | `None`, state unchanged | key absent |

The optimized `Map::insert` remains allocation-backed and infallible except process allocation
failure. Its comparison uses a pre-reserved capacity and does not equate allocation failure with the
reference's logical full result.

## Evidence plan

1. Add focused nominal, full, replacement-at-full, removal and trace tests for the safe reference.
2. Compare production `Map` with the reference after every admitted trace operation.
3. Extract only monomorphic safe operations after the Rust API and tests are fixed.
4. Prove denotation equalities against the pinned corpus and audit every headline.
5. Mutate full handling, replacement order and removal in throwaway copies; each unchanged gate must
   fail and the original must pass.

## Trust boundary

Proof of the safe transition system does not transfer to optimized storage. Until its representation
relation is universal, the production comparison grade is tested.

## Open questions

- The optimized relation may require Kani over bounded layouts, a proof-friendly probe extraction or
  a redesign that shares a safe transition core. Select only after measuring the reference boundary.
