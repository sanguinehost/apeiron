# Specification - Bounded Storage Refinement

- **Owner:** OpenCode
- **Intent:** `docs/intent/storage-refinement.md`
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean before this epic
- **Status:** approved

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
