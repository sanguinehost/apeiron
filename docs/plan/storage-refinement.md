# Plan - Bounded Storage Refinement

- **Owner:** OpenCode
- **Intent:** `docs/intent/storage-refinement.md`
- **Spec:** `docs/spec/storage-refinement.md`
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean before this epic
- **Status:** approved

## Goal and completion condition

Establish one safe finite storage semantics, prove its successful transitions against the canonical
Table definitions, then relate `Map` without promoting sampled agreement to proof.

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
| W1 | Fixed-capacity container and differential trace | `src/safe_table.rs`, `src/lib.rs`, `tests/safe_table.rs` | OpenCode | `cargo test --locked --test safe_table` | Full-check or removal mutation makes the trace fail. |
| W2 | Monomorphic extraction | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | OpenCode | `proof/regen.sh --check` | Generated transition mutation breaks freshness or proof. |
| W2 | Extraction evidence | `proof/README.md` | OpenCode | `git diff --check` | Not applicable: extraction freshness owns the executable control. |
| W3 | Canonical Table refinement | `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean` | OpenCode | `lake build` in `proof/corpus` | Transition mutation breaks unchanged theorem. |
| W4 | Optimized relation decision | `docs/kernel-boundary.md`, `docs/direction.md` | OpenCode | `git diff --check` | Not applicable: records the proved/tested boundary and next executable gate. |

## Task blocks

### W1-T1 - Safe finite reference

- **Scope:** Implement fixed-capacity lookup/replace/remove and compare against `Map` on
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
- **Files:** `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`
- **Depends on:** W1-T1
- **Grade:** tested
- **Gate:** `proof/regen.sh --check`
- **Negative control:** A generated body mutation must make freshness or W3 fail.
- **Evidence to record:** Source/tool digests, selected wrappers and extraction result.

### W2-T2 - Extraction evidence

- **Scope:** Record the selected safe-table bodies, source digest, result and exclusions.
- **Files:** `proof/README.md`
- **Depends on:** W2-T1
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. W2-T1 owns extraction freshness.
- **Evidence to record:** Generated names, source digest and unsupported operations.

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

### W4-T1 - Universal relation to `Map`

- **Scope:** Select the smallest universal relation target from W1 evidence; closing it authorizes removing `Map`.
- **Files:** `docs/kernel-boundary.md`, `docs/direction.md`
- **Depends on:** W3-T1
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. No universal optimized-map claim closes in this task.
- **Evidence to record:** Selected representation, excluded layers and next red command.

## Barrier checklist

- [ ] Container tests and controls pass.
- [ ] Extraction is fresh and free of `axiom`, `sorry` and `admit`.
- [ ] Canonical refinement and fail-closed audits pass.
- [ ] The universal relation to `Map` is checked, authorizing its removal.
- [ ] Rust tests, clippy, formatting and Kani pass.
- [ ] Source and tool digests and trust boundaries are recorded.

## Stop conditions

- Stop extraction after three unsupported Aeneas representations and record the smallest missing
  safe construct; do not rewrite formulas in wrappers.
- Stop optimized-map promotion if the relation remains sampled; retain tested status and name the
  exact universal obligation.

## W3 blocker record

`W1` and `W2` are closed: the safe fixed-capacity reference passes its focused tests and the
10,000-step differential comparison, and `word_safe_table4_get/insert/remove` are extracted from the
real `src/safe_table.rs` with a fresh, axiom-free generated file. `W3` is blocked before the first
refinement theorem.

Three materially distinct attempts were made at the refinement:

| Attempt | Change | Result |
|---|---|---|
| 1 | Iterator/`for`-loop reference | Charon refused outer-loop breaks and iterator closures. |
| 2 | Bounded recursive `find_from`/`empty_from` reference | Extracts, but the generated body is a `partial_fixpoint`; direct `simp`/`rw` cannot unfold the recursion and the scalar `Result` binds are not `[simp]`. |
| 3 | Aeneas `fun_induction` / `step` on the generated body | `fun_induction` reports no functional induction theorem; `step` accepts only `WP` specs, not pure `Eq` goals. |

The precise unresolved goal: a total Lean function for the safe search, or a canonical refinement
stated as a `WP.dspec` Hoare triple. The extracted search is a `partial_fixpoint`; the only supplied
reasoning principle is the auto-generated `.fixpoint_induct`, which targets `WP.dspec` goals and
requires admissibility over a `Result` CCPO. A pure structural characterization additionally needs a
denotation from the Aeneas array state to `Apeiron.Table.HMap`.

Next bounded task, unblocked now: give the Rust search a form Aeneas translates to a structurally or
well-founded recursive function (for example recursion on a slice/list tail rather than a numeric
index, with the index carried as an accumulator), so the extracted body has equation lemmas and
`fun_induction` support. If that fails, state the refinement through `Apeiron.WP`/`loop.spec_decr_nat`
with the array denotation above.

### W3 unblock attempt (2026-09-12)

Two further materially distinct shapes were tried.

| Attempt | Change | Result |
|---|---|---|
| 4 | Recursion on a slice tail via `split_first` | The generated body is still a `partial_fixpoint`, and `core.slice.Slice.split_first` enters as an unmodeled external `axiom`. Rejected. |
| 5 | Fixed-array `while index < N` search with an early `return` | Charon translates each search to a total `loop` combinator marked `@[rust_loop]`; no `partial_fixpoint`, no external axiom. The supplied `Aeneas.Std.loop.spec_decr_nat` measure principle now applies. |

`src/safe_table.rs::find_from` and `empty_from` are now the `while` loops of attempt 5. The
extraction is regenerated: `word_safe_table4_get/insert/remove` call the loop-based generated
bodies, `proof/regen.sh --check` exits 0, and the throwaway source mutation (`stored == key` to
`stored != key`) makes `--check` exit 1. `cargo test --locked --test safe_table` passes 4/4 and
`cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` exits 0.

The remaining obligation is the canonical refinement itself: a `loop.spec_decr_nat` invariant
relating the generated `Array (Option (K × V)) N` state to a denotation over that array, then
`Apeiron.Table.get`/`insertWith`/`replace` lemmas for `get`, `try_insert` and `remove`. That is a
loop-invariant proof, not an extraction blocker; the first four attempts failed because no total
generated search existed, and one does now.

### W3 first slice (2026-09-12)

`proof/corpus/SafeTableRefinement.lean` proves the generated search semantics with
`Aeneas.Std.loop.spec_decr_nat` at measure `4 - index` and invariant `NoMatchIn key l start index`:

- `find_from_loop_spec`: a returned `some j` is the first key match in `[start, 4)`, and `none` means
  no match there. `find_from_spec` lifts it to the `@[reducible]` `find_from`.
- The body is discharged through `Array.index_usize_spec` and a local `@[step]` `I64` equality lemma.
- Cone `{propext, Classical.choice, Quot.sound}`; `#audit`ed in `proof/corpus/AxiomAudit.lean`.
- Control: a throwaway copy with the generated `done (some index)` changed to `done none` fails the
  same proof. The real `proof/regen.sh --check` and the 3277-job bridge build stay green.

### W3 write links

`try_insert_spec` and `remove_spec` in `proof/corpus/SafeTableRefinement.lean` link the extracted
`try_insert`/`remove` to the canonical table writes, under
`WellFormed t := t.len.val = List.countP (fun e => e.isSome) t.slots.val ∧ KeysUnique t.slots.val`:

- `try_insert_spec`: `Ok old` gives `old = tableHMap self key` and
  `tableHMap t' = Apeiron.Table.insertWith (fun new _ => new) key (some value) (tableHMap self)`;
  `Err _` gives `t' = self`.
- `remove_spec`: `old = tableHMap self key` and
  `tableHMap t' = Apeiron.Table.insertWith (fun _ _ => none) key none (tableHMap self)`.

The invariant supplies the `self.len + 1`/`self.len - 1` scalar bounds (`Aeneas.Std.Usize.add_spec`
needs `len + 1 ≤ max`; `Usize.sub_spec` needs `1 ≤ len`) and, for `remove`, the key uniqueness
that makes clearing the first match equal to the canonical `none` write. Cones inside
`{propext, Classical.choice, Quot.sound}`; `#audit`ed. Control: a throwaway copy with `try_insert`'s
generated `Ok none` changed to `Ok (some value)` fails `try_insert_spec` at
`SafeTableRefinement.lean:786`; the unmutated bridge builds 3351 jobs.

Both specs now conclude `WellFormed t'`. Preservation is proved by the `countP` step
(`countP_set_some`/`countP_set_none`, `one_le_countP_of_get`) and the uniqueness step
(`keysUnique_set_some`/`keysUnique_set_none`). Still open: `WellFormed` at the initial state
(`SafeTable::new`); that is the gate before `Map` is removed.

## S6 baseline/red/restore controls

Throwaway root `.mut-s6/` (btrfs reflink of `proof/`, `target/aeneas` symlink; same filesystem as the
repository). Baseline `cd .mut-s6/proof/corpus && lake build` exit 0, 3351 jobs. Toolchain: the single
patched `target/aeneas` symlink, no parallel toolchain directory.

| Obligation | Mutated file (throwaway) | Mutation | Red command (cwd) | Exit | Red line | Restore sha256 | Restore gate |
|---|---|---|---|---|---|---|---|
| Search `find_from_loop_spec` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.find_from_loop.body`: `if index < N` -> `if index > N` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:103:10` (`find_from_loop_spec`) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Write `try_insert_fresh_iff`, `try_insert_err_iff` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.try_insert`: fresh-branch `Ok none` -> `Ok (some value)` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:877:16` (`try_insert_spec`; write link the exactness headlines consume) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Initialization `safe_table_new_eq_emptyTable` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.new`: `len := 0#usize` -> `len := 1#usize` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:1067:2` (`safe_table_new_eq_emptyTable`) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Full handling `try_insert_fullTable_refusal` | `.mut-s6/proof/corpus/SafeTableRefinement.lean` | `fullTable`: slot 0 `some (k0, v0)` -> `none` (falsifies `NoEmptyIn`) | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:1222:67` (`fullTable_noEmpty`; `NoEmptyIn` witness the refusal theorem consumes) | `6d089d616581d958b8379d1c68b69fdca01d1fa481d63c9424f23528ed278685` | exit 0, 3351 jobs |

Restored repository gate: `cd proof/corpus && lake build` exit 0, 3351 jobs; `proof/regen.sh --check`
exit 0, fresh extraction, `ApeironWord` 1708 jobs.
