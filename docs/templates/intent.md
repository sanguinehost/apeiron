# Intent — <epic title>

- **Owner:** <name>
- **Source revision:** <git rev-parse HEAD>; working tree <clean|dirty, list files>
- **Status:** draft | approved
- **Spec:** `docs/spec/<slug>.md` (to be written)
- **Plan:** `docs/plan/<slug>.md` (to be written)

## Problem

<The concrete defect, gap or capability. One paragraph, present tense.>

## Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| <approach> | <the measured or proved reason it fails> |

## Completion condition

<One executable sentence. What command, or which theorem, is green when this is done.>

```sh
<the command whose exit 0 closes the epic, or the Lean build target that must pass>
```

## Invariants

<What must never break while the epic is open. Algebraic laws, layout, partiality, determinism.>

1. <invariant>
2. <invariant>

## Verification hypotheses

<Measurable claims the epic must establish, each with its evidence grade.>

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| <claim> | proved | `<lake build>` / `<theorem name>` |
| <claim> | tested | `cargo test --workspace --all-features --locked` |
| <claim> | measured | `<bench command>`, <machine>, <metric> |
| <claim> | model/anchor | <the mapping, stated as a mapping> |

## Non-goals

<What this epic explicitly does not attempt. Each entry prevents a scope expansion.>

- <non-goal>
- <non-goal>

## Trust boundary

<What the evidence will not reach: rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, the `unsafe`
container, or a layer outside the extracted fragment. State it here, before the proof is written.>

## Open questions

- <question, with the decision needed and who owns it>
