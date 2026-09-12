# Intent - Bounded Storage Refinement

- **Owner:** OpenCode
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean
- **Status:** approved
- **Spec:** `docs/spec/storage-refinement.md`
- **Plan:** `docs/plan/storage-refinement.md`

## Problem

`Map` implements lookup, replacement and removal with open addressing, `MaybeUninit`, SIMD tag
scans and growth. Random differential tests and Miri exercise it, but no safe executable state
machine defines its finite failure semantics and no proof links storage transitions to
`Apeiron.Table.insertWith`.

## Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Extract the complete optimized map first | Aeneas does not cover its unsafe allocation, SIMD and generic closure surface. |
| Treat `std::collections::HashMap` as the specification | It supplies an oracle but not the canonical Table theorem link or fixed-capacity refusal. |
| Restate table equations beside Rust | A second formula is not a refinement of the executing body. |
| Replace the optimized map immediately | Existing measured paths remain useful until an equivalent implementation is proved and measured. |

## Completion condition

A bounded safe reference is extracted from real Rust, proved to implement canonical Table reads and
writes on successful transitions, and the production map passes a discriminating transition trace
against it. The optimized map remains labelled tested until the relation itself is universal.

```sh
sh -c 'cargo test --workspace --all-features --locked && proof/regen.sh --check && cd proof/corpus && lake build'
```

## Invariants

1. Full capacity refuses only insertion of an absent key; replacement at capacity remains valid.
2. Refusal returns no successor state and leaves the input state available to the caller.
3. Lookup and removal preserve full key identity; physical slot order is unobservable.
4. Canonical merge argument order is new value then old value.
5. Algebraic merge claims instantiate `Apeiron.Trichotomy.AnyRing`; generic replacement semantics
   are identified as table semantics rather than attributed to the trichotomy.
6. No proof about the safe reference is reported as proof of the optimized map.

## Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| The safe finite reference implements its stated transitions. | tested | `cargo test --locked --test safe_table` |
| Extracted successful transitions refine canonical `Apeiron.Table` operations. | proved | Named bridge theorems and fail-closed audits. |
| The optimized map agrees on selected transition traces. | tested | Differential production/reference test and mutation control. |
| The optimized unsafe storage universally refines the reference. | open | Requires a separate universal representation proof. |

## Non-goals

- Proving allocation, SIMD, threads, rustc/LLVM or machine execution.
- Covering every `Map` convenience API in the first reference.
- Replacing the optimized implementation before equivalent behavior and cost are established.
- Claiming that generic table replacement is derived from the trichotomy.

## Trust boundary

The safe Rust extraction and Lean theorem do not cover the optimized map's unsafe slots, allocation,
SIMD scan, concurrency, compiler backend or hardware. Differential traces are tests, not a universal
refinement.

## Open questions

- Which bounded relation can cover the optimized slot representation without importing unsafe
  allocation into the Aeneas proof? This is answered by a later wave after the reference is fixed.
