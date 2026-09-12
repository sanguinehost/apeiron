# Plan - Bounded Storage Refinement

- **Owner:** OpenCode
- **Intent:** `docs/intent/storage-refinement.md`
- **Spec:** `docs/spec/storage-refinement.md`
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean before this epic
- **Status:** approved

## Goal and completion condition

Establish one safe finite storage semantics, prove its successful transitions against the canonical
Table definitions, then relate the production map without promoting sampled agreement to proof.

```sh
sh -c 'cargo test --workspace --all-features --locked && proof/regen.sh --check && cd proof/corpus && lake build'
```

## Non-goals

- Whole-container extraction in one task.
- Proof by differential testing.
- Performance changes before semantic equivalence.

## Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | Safe finite reference and production trace | `src/safe_table.rs`, `src/lib.rs`, `tests/safe_table.rs` | OpenCode | `cargo test --locked --test safe_table` | Full-check or removal mutation makes the trace fail. |
| W2 | Monomorphic extraction | `proof/extract/lib.rs`, `proof/ApeironWord.lean`, `proof/README.md` | OpenCode | `proof/regen.sh --check` | Generated transition mutation breaks freshness or proof. |
| W3 | Canonical Table refinement | `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean` | OpenCode | `lake build` in `proof/corpus` | Transition mutation breaks unchanged theorem. |
| W4 | Optimized relation decision | `docs/kernel-boundary.md`, `docs/direction.md` | OpenCode | `git diff --check` | Not applicable: records the proved/tested boundary and next executable gate. |

## Task blocks

### W1-T1 - Safe finite reference

- **Scope:** Implement fixed-capacity safe lookup/replace/remove and compare the production map on
  the same bounded traces.
- **Files:** `src/safe_table.rs`, `src/lib.rs`, `tests/safe_table.rs`
- **Depends on:** None
- **Grade:** tested
- **Gate:** `cargo test --locked --test safe_table`
- **Negative control:** In a throwaway copy, treat a full-table replacement as refusal or leave a
  removed key readable; the unchanged test must fail and the original must pass.
- **Evidence to record:** Test population, full/refusal witnesses and source digest.

### W2-T1 - Extract the safe transition body

- **Scope:** Add concrete word/capacity wrappers that call the real safe implementation and regenerate
  the accepted Aeneas output.
- **Files:** `proof/extract/lib.rs`, `proof/ApeironWord.lean`, `proof/README.md`
- **Depends on:** W1-T1
- **Grade:** tested
- **Gate:** `proof/regen.sh --check`
- **Negative control:** A generated body mutation must make freshness or W3 fail.
- **Evidence to record:** Source/tool digests, selected wrappers and extraction result.

### W3-T1 - Canonical refinement

- **Scope:** Prove successful word-table transitions denote `Apeiron.Table` lookup and replacement,
  with explicit full and missing-key witnesses.
- **Files:** `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean`
- **Depends on:** W2-T1
- **Grade:** proved
- **Gate:** `lake build` in `proof/corpus`
- **Negative control:** Mutate replacement or removal in a throwaway generated file; unchanged
  refinement must fail and the original must pass.
- **Evidence to record:** Theorem names and complete axiom cones.

### W4-T1 - Production relation boundary

- **Scope:** Select the smallest universal relation target for optimized slots from W1 evidence and
  retain tested status until that target passes.
- **Files:** `docs/kernel-boundary.md`, `docs/direction.md`
- **Depends on:** W3-T1
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. No universal optimized-map claim closes in this task.
- **Evidence to record:** Selected representation, excluded layers and next red command.

## Barrier checklist

- [ ] Safe reference tests and controls pass.
- [ ] Extraction is fresh and free of `axiom`, `sorry` and `admit`.
- [ ] Canonical refinement and fail-closed audits pass.
- [ ] Production map remains tested unless a universal relation is checked.
- [ ] Rust tests, clippy, formatting and Kani pass.
- [ ] Source and tool digests and trust boundaries are recorded.

## Stop conditions

- Stop extraction after three unsupported Aeneas representations and record the smallest missing
  safe construct; do not rewrite formulas in wrappers.
- Stop optimized-map promotion if the relation remains sampled; retain tested status and name the
  exact universal obligation.
