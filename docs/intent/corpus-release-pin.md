# Intent - Immutable Corpus Dependency

- **Owner:** OpenCode
- **Source revision:** `7cec238d7544a9914e8f71703a2c0c08bf9e0883`; working tree clean
- **Status:** approved
- **Spec:** `docs/spec/corpus-release-pin.md`
- **Plan:** `docs/plan/corpus-release-pin.md`

## Problem

The refinement bridge loads the canonical mathematical corpus through
`../../../sanguine-apeiron/proof`. A sibling path can resolve to any working-tree state, so a clean
Apeiron checkout does not identify the corpus source checked by `lake build`.

## Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Retain the sibling path and record a commit in prose | Lake still builds the unchecked sibling state. |
| Copy selected theorem statements into the bridge | This creates a second mathematical authority and breaks literal trichotomy rooting. |
| Vendor only the current import closure | The selected closure can drift from the canonical package and its fail-closed audits. |
| Pin the repository branch name | A branch is mutable and does not identify release input. |

## Completion condition

Lake resolves package `apeiron` from the published repository at the full corpus commit, loads its
`proof` package, and builds the bridge.

```sh
sh -c 'cd proof/corpus && lake update apeiron && lake build'
```

## Invariants

1. The bridge imports canonical declarations from `Apeiron.Trichotomy.AnyRing`; it does not restate
   their formulas.
2. The corpus, bridge, Aeneas backend and Mathlib use Lean 4.31.0-compatible revisions.
3. The local extraction package remains a path dependency because it is part of this repository.
4. No corpus source or theorem statement changes in this epic.

## Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| Lake selects corpus commit `d9aefc93522e2109bc96158526de6dbd52477276`. | tested | Generated `proof/corpus/lake-manifest.json` assertion. |
| The pinned corpus and bridge elaborate together. | proved | `lake build` in `proof/corpus`. |
| Existing extracted Rust remains fresh. | tested | `proof/regen.sh --check`. |

## Non-goals

- Moving the corpus history into this repository.
- Changing a theorem, extracted Rust kernel or application integration.
- Vendoring Mathlib, Aeneas or Charon.
- Establishing availability of the private Git service without credentials.

## Trust boundary

The commit pin identifies source content. It does not verify Git transport, hosting availability,
rustc/LLVM, Aeneas translation, Lean binaries or the optimized storage layer.

## Open questions

- None for this arc. Corpus history convergence remains a separate repository-level migration.
