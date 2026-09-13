# Plan - Single Table Refinement

## Binding boundary replacement

Owner Astra; base `da01485d24be8900824750431a70f19100bee200`, initially clean. The lead authorizes
source deletion and a red boundary commit in the supplied checkout. No branch/worktree is created;
the existing `one-allocation` branch is unrelated and is not modified. The two-container decision
below is superseded, not a prerequisite to this sequence. No new markdown or other governing-doc
edits are authorized. `docs/kernel-boundary.md` contains the replacement contract and trust cut.

**Intent:** the executing table is the thesis's proof subject. A proved reference beside an unproved
container does not satisfy it. Delete both old containers and their wrappers rather than preserving
that boundary under aliases. Compatibility breakage is accepted; no performance claim transfers.
**Spec:** one `Table<K,V>`, default-valued total lookup, new-first `insertWith`, finite-stream `build`,
and home-sharded finite overrides, as specified in the boundary document. No fixed-capacity-four
restriction, automatic `Option` lifting or commutativity assumption for ordinary builds survives.
The old intent/spec documents below describe retired source, not the replacement's contract.

The boundary change is three serial owned slices: `APEIRON-TABLE-W0-T1` owns these two documents;
`APEIRON-TABLE-W0-T2` deletes `src/map.rs`, `src/safe_table.rs`, `src/set.rs`;
`APEIRON-TABLE-W0-T3` owns `src/table.rs`, `src/lib.rs`, `src/merge.rs`. Astra owns all three for this
explicit lead-assigned deletion. Only the last slice takes lane R. They share one authorized
boundary commit, not a claim of implementation completion.

```text
SCOPE:      the three slices above; remove competing containers, expose only Table, freeze its API
REGISTER:   planning/API removal; excludes completed implementation, refinement and performance
CORPUS:     Apeiron.Table "get_build_acc", "build_eq_of_key_streams_eq",
            "build_gauge_invariance"; Adjacency and Home, native grade proved
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean
GATE:       root: cargo check --lib --locked baseline 0; cargo test --doc --locked after deletion
            101: E0599, Table::new missing at src/merge.rs:47; git diff --check 0
GRADE:      anchor; declaration only, no methods or executing container; no new proof/measurement
```

The merge doctest is rewired to the required API and deliberately fails without its implementation.
The library alone compiles, with unread-field warnings. Do not suppress those warnings or add a
compatibility implementation to manufacture green. The old integration tests, benchmarks, examples,
fuzz targets and storage Kani harnesses are retained as red migration inputs, not silently disabled.
`proof/extract/lib.rs` still names deleted source; generated Lean and its old bridge are stale.
A successful build of those cached definitions alone cannot establish freshness or new-table proof.

Observed boundary checks (root):

| Command | Outcome |
|---|---|
| `cargo test --doc --locked` | 101; one passes, the Table example fails with E0599 (`Table::new` missing). |
| `cargo test --workspace --all-features --locked` | 101; E0432/E0433 on removed `map`, `safe_table`, `set`, `CountTable` and scanner imports. |
| `proof/regen.sh --check` | 1; source digest stage rejects deleted `src/safe_table.rs`, before extraction. Charon `0.1.254` (`b104e24f`), Aeneas `505b6ca-dirty`, Rust extraction toolchain `nightly-2026-08-18`; script reports Lean pin `4.31.0`. |
| `git diff --check` | 0. Formatting evidence only; no storage proof or mutation-control closure. |
| `cargo fmt --all -- --check` | 0 after correcting the changed re-export formatting. |

### Identity

Corpus read: clean `apeiron-rebuild` at `50c94192399add3b9f8e64bd14e36d8cba270705`.
The bridge currently resolves `d9aefc93522e2109bc96158526de6dbd52477276`.

| Source | SHA-256 at boundary edit |
|---|---|
| `src/lib.rs` | `56c0ce5d9fcf14a7075e286cc7710461dc2b986aaf3cecea91844b6b5c15c7a7` |
| `src/table.rs` | `18c9ce400f8fb8f872e6c9cfe370d4f0415081589833d58f0668240c3feea72f` |
| `src/merge.rs` | `f1a056ad3d19c460ff49ec565ad1e22d06ac77ac08da3d24efdff48dc63d1874` |
| corpus `Table/Basic.lean` | `e7558bac1d255643f58100f5a1adf32926a423fe6b49e7c991f8d3a3b0eb26b9` |
| corpus `Table/Adjacency.lean` | `5593d39db83213694ef51b9cb4d73686b9e2b5e8b548a2840e61b5a94d382732` |
| corpus `Table/Home.lean` | `bab40ae04d1bd19cde8c97e09eb2453652cbddf253969ebad03ac582cf5d599b` |

No Lean/toolchain acceptance is inferred from those digests. Each execution task records its actual
Rust, Charon/Aeneas binaries (including patches), Lean, Mathlib and corpus identities. Tool pins
and the historical constructor patch are in `proof/README.md`; do not assume the patch solves
growable vectors, fallible reservation, function pointers or generic callback translation.

### Extraction feasibility stop

`APEIRON-TABLE-W4-T1`, Sol, base `dde9d58`, lanes E+B. Charon emits the real bodies for all four
monomorphic probes. Aeneas rejects each LLBC before translating a method because
`Table.home : fn(&K) -> u64` is an arrow-valued struct field. Verbatim:
`Arrow types are not supported yet`; `Could not translate type decl
'apeiron_word::table::Table because of previous error`. The backend therefore does not reach a
separate verdict on `Vec`, allocation, 256-shard initialization, `usize`, default values, iterator
loops or callback invocation. `StorageError` and both constructors extract. Wave 4 stops here;
waves 5-7 remain blocked on a generated `Table` representation.

```text
SCOPE:      proof extraction roots/generated file, TableRefinement/audits/pin, storage status docs
REGISTER:   extraction/refinement; excludes compiler, allocator and ungenerated Table transitions
CORPUS:     Apeiron.Table Basic/Adjacency/Home at 50c94192399add3b9f8e64bd14e36d8cba270705
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean
GATE:       proof/regen.sh --check; lake build in proof/corpus; initial stale source digest exit 1
GRADE:      proved canonical contracts/error constructors; Rust Table implementation link open
```

### Serial waves and lanes

IDs are `APEIRON-TABLE-W<n>-T1`; one owner per file. Before dispatch the lead records the integrated
base and creates `worktree-agent-<id>`. Each row is a serial wave; a rename is one owned artifact.
**R** = `target/`; **E** = R + `proof/ApeironWord.lean` + `proof/.lake/`;
**B** = `proof/corpus/.lake/` + `proof/.lake/` + Aeneas backend cache. Controls use throwaway copies
and separate writable build outputs. No simultaneous lane occupants. Future filenames denote
required artifacts, not already existing tests or proofs.

| Wave | Owned artifacts / obligation | Owner; lane | One focused gate | One rejecting control |
|---|---|---|---|---|
| 1 | `src/table.rs`, `src/merge.rs`: implement the declared constructor/get/insert_with/build, initialized growable shards and fallible resources; expand executable doctests for nonzero default, collisions and directional merges | Terra; R | `cargo test --doc --locked` (root) | First insertion bypasses `f v d` in a throwaway real body; nonidentity-default fixture must fail. |
| 2 | `tests/safe_table.rs` renamed to `tests/table.rs`: replace the two-container trace with single-table default/write/build tests, growth, exact error witnesses and replace/append order fixtures | Flash; R | `cargo test --locked --test table` (root) | Compare only home addresses instead of full keys in a throwaway real body; collision fixture must fail. |
| 3 | `proof/corpus/lakefile.lean`, `proof/corpus/lake-manifest.json`: pin the read gen2 revision, check Basic/Adjacency/Home on the bridge toolchain | Sol; B | `lake build Apeiron.Table.Basic Apeiron.Table.Adjacency Apeiron.Table.Home` (cwd `proof/corpus`; preparation: `lake update apeiron`) | Nonexistent corpus revision in a throwaway package must fail resolution, never fall back to sibling source. |
| 4 | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: replace `word_safe_table4_*` roots with actual Table constructor/get/insert_with/build roots, no capacity bound; include executing routing/growth and concrete merge instances | Sol; E | `proof/regen.sh --check` (root; first regenerate with `proof/regen.sh`) | Mutate the real default-read branch; unchanged generated output must fail freshness. |
| 5 | `proof/corpus/SafeTableRefinement.lean` renamed/reshaped to `TableRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: remove retired roots/audits; prove constructor, joint representation witnesses, terminating home-shard lookup and total default read | Sol; B | `lake build` (cwd `proof/corpus`) | Generated lookup returns the default for a present key; unchanged lookup theorem must fail. |
| 6 | `proof/corpus/TableRefinement.lean`, `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/AxiomAudit.lean`: writes/growth refine canonical insertWith, preserve Rep, characterize success/resource refusal with witnesses; audit imports the new module | Sol; B | `lake build` (cwd `proof/corpus`) | Generated first-write branch stores `v` instead of `f v d`; unchanged insertWith theorem must fail. |
| 7 | `proof/corpus/TableBuildRefinement.lean`, `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/AxiomAudit.lean`: actual build = canonical build; compose get_build_acc, per-key stream equality and conditional gauge invariance; exact Home attribution and order counterexamples | Sol; B | `lake build` (cwd `proof/corpus`) | Reverse same-key contributions in the generated build; unchanged replace/append build theorem or witness must fail. |

All gates are required with baseline/red/restored controls; the proof tasks target `proved`, tests
only `tested`. A type or empty module that elaborates is not a completed row. No `sorry`, admission,
opaque reservation axiom, assumed desired simulation, or reduced-capacity proof can close a row.
If proof size would exceed 1,500 lines, split the obligation into another serialized ≤3-file task
before editing beyond the declared paths. Sol must approve the resource and callback premises;
this architecture decision is not that proof-sufficiency verdict.

**First gate:** `cargo test --doc --locked`, cwd repository root. It currently exits 101 on the
missing `Table::new`. Implement the same path; do not ignore the doctest. The extraction row is an
explicit feasibility stop: after three materially different unsupported shapes, return the exact
unsupported primitive and next tool-model task. Never restore a separate reference to evade it.

### Caller migration and integration barrier

The seven rows do not repair the whole compatibility surface. After the API lands, Flash migrates
retained callers in serialized ≤3-file batches, without introducing old-name aliases:

- `tests/a_map_is_a_map.rs`, `tests/a_set_is_a_set.rs`, `tests/the_counts_are_exact.rs`: one initial
  batch, gate `cargo test --locked --test a_map_is_a_map --test a_set_is_a_set --test the_counts_are_exact`;
  control: change a new-first merge to old-first and require the unchanged directional fixture red.
  Preserve corresponding semantic/boundary cases; entry/iterator-only contracts are explicitly
  retired, not claimed for the new minimal API.
- `src/proofs.rs`: separate Kani batch, gate `cargo kani`; replace retired storage/scan claims with
  bounds/routing/collision harnesses for Table; a routing-mask mutation must fail. Preserve unrelated
  algebra harnesses. Bounded Kani is supplementary, not the storage refinement theorem.
- Remaining test/example/bench/fuzz callers need an exact path manifest and ≤3-file, single-command
  batches selected by their owners; no unbounded "repair all callers" dispatch. That manifest is
  **MISSING** and blocks whole-workspace closure, not the seven focused gates. Benchmarks must be
  remeasured on the new executing path before old numbers are reused.

Integration order: declaration/deletion → implementation and tests → corpus pin → actual extraction
→ constructor/read → write/resource → build/layout → caller migrations → full assurance barrier.
The lead, not an individual task, runs all AGENTS §6 Rust commands (including Kani), regeneration
freshness, both Lake builds and the controls on one source/tool identity. Existing S7/S8 requirements
remain: claim/witness/audit coverage, trusted checker ≤5,000 lines, owned source ≤1,500 lines and
≤12 immediate source files. Missing assurance gates remain missing; deletion alone does not close
them. Corrections to out-of-scope README, direction, handoff and old lifecycle documents need a
separate lead-owned documentation batch; their retired-container claims are not current acceptance.

**Residual:** no executing Table, no new extraction/refinement, no runtime measurement. SIMD, raw
slot lifecycle and workers are removed rather than left in a tested-only path. Source-level proof
still trusts translation/model fidelity, compiler/runtime/allocator realization and machine execution
as explicitly bounded in `docs/kernel-boundary.md`; it may not assume the table semantics themselves.

## Historical bounded-reference plan and evidence (superseded)

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
| W4 | Optimized relation decision | `docs/kernel-boundary.md`, `docs/plan/storage-refinement.md` | Astra | `git diff --check` | Requirement review; no executable equivalence claim. |

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

### W4-T1 - Independent relation decision

- **Scope:** Freeze the independent relation and trust cuts; neither decision nor reference proof authorizes removing `Map`.
- **Files:** `docs/kernel-boundary.md`, `docs/plan/storage-refinement.md`
- **Depends on:** W3-T1
- **Grade:** anchor (architecture), not implementation equivalence
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. No universal optimized-map claim closes in this task.
- **Evidence to record:** Selected representation, excluded layers and next red command.

## Barrier checklist

- [ ] Container tests and controls pass.
- [ ] Extraction is fresh and free of `axiom`, `sorry` and `admit`.
- [ ] Canonical refinement and fail-closed audits pass.
- [ ] Independent optimized relation decision and complete controls satisfy S6; no `Map` retirement authorized.
- [ ] Universal optimized representation/transition relation is checked before any optimized proof claim; remains open independently of reference closure.
- [ ] S7 integration and S8 ledger/perimeter enforcement pass; see `docs/flash-handoff.md`.
- [ ] Rust tests, clippy, formatting and Kani pass.
- [ ] Source and tool digests and trust boundaries are recorded.

## Stop conditions

- Stop extraction after three unsupported Aeneas representations and record the smallest missing
  safe construct; do not rewrite formulas in wrappers.
- Stop optimized-map promotion if the relation remains sampled; retain tested status and name the
  exact universal obligation.

## Historical W3 blocker record

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
(`SafeTable::new`) at this historical checkpoint. The constructor and exact outcomes are now closed;
neither result authorizes removing `Map`.

## Optimized representation decision and remaining sequence

`APEIRON-STORAGE-W4-T1`, owner Astra; base `7d6100c450e5da1fa469ffe9adc16a65c7e720ba`.
User-authorized supplied-checkout documentation exception; initially clean. No branch/worktree created.
The frozen relation is in `docs/kernel-boundary.md` §4. `SafeTable` is the checked reference;
whole unsafe/SIMD/threaded extraction is rejected. Share safe executing scalar bodies where extraction
permits; an abstract layout proof is not a Rust heap proof. S4/S5 stand; S6-S8 and optimized refinement
remain separate open obligations. The earlier W1-W4 table is historical, not renewed acceptance.

```text
SCOPE:      docs/kernel-boundary.md, docs/plan/storage-refinement.md; boundary and bounded sequence
REGISTER:   planning; excludes Rust/Lean edits, optimized promotion and retirement
CORPUS:     Apeiron.Table.Basic: "get_insertWith_self", "get_insertWith_ne",
            "build_eq_of_key_streams_eq"; native grade proved
REFERENCE:  /home/socol/Workspace/apeiron/proof/corpus/.lake/packages/apeiron/proof/Apeiron/Table/Basic.lean
GATE:       git diff --check (root), baseline exit 0; optimized refinement gate MISSING
GRADE:      anchor; bridge baseline lake build exit 0, 3351 jobs; no new refinement or measurement
```

Baseline source SHA-256 (bridge build only; extraction freshness not rerun):

| Path | SHA-256 |
|---|---|
| `src/map.rs` | `4226acc5282ce7626eefb955dff55ce3023d6e5e80167fd536bec982d86c1961` |
| `src/table.rs` | `5f6a62cb47a791162c0173f41d62342f7a0981c650ceed004e8cf0c83f0b9b9d` |
| `src/safe_table.rs` | `c09c5419456d355b2856064479114cf1c575db5016d9787078c0645a663a0479` |
| `proof/extract/lib.rs` | `e5a86d5add7d83a4c9b72dba045dd25cf6e9aa8924442cdc88faa23ae7a79fc8` |
| `proof/ApeironWord.lean` | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` |
| `proof/corpus/SafeTableRefinement.lean` | `6d089d616581d958b8379d1c68b69fdca01d1fa481d63c9424f23528ed278685` |

Resolved corpus checkout: `d9aefc93522e2109bc96158526de6dbd52477276`; manifest Mathlib:
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. Full source/tool identity acceptance remains S7.

### Dispatch and lanes

Each row below is its own serial wave, task T1, with ID `APEIRON-STORAGE-W<number>-T1`.
Flash owns ordinary execution; Sol owns the explicitly marked proof-sufficiency tasks. Lead records
the integrated base and assigns `worktree-agent-<id>` before dispatch. This is a decomposition, not
permission to edit all listed paths now. Proposed files/commands marked **MISSING** require gate-first
implementation; exit 0 from a missing-test filter is never acceptance.

Lanes: **R** = `target/`; **E** = `proof/ApeironWord.lean` plus R and `proof/.lake/`;
**B** = `proof/corpus/.lake/` plus `proof/.lake/` and the Aeneas backend cache under R.
Any task running both extraction and bridge takes E+B exclusively. Controls use throwaway source and
generated copies and separate build outputs; immutable dependency caches may be reused read-only.
No sibling task shares an occupied lane. Every control records baseline/red/restored exit statuses,
source/generated digests, actual patched tool binaries and resolved Lean/Mathlib/corpus identities.

| Wave | Owned files (maximum three) / result | Owner; lane | Single gate (root unless stated) | Control |
|---|---|---|---|---|
| 5 | `proof/storage-controls.py`, this plan: durable S6 proof-control runner and record; **MISSING runner** | Flash; E+B | `python3 proof/storage-controls.py` | Independently mutate generated search hit, fresh write/old return, removal, constructor len, full payload/state, and force refusal on free/present keys. Each unchanged bridge build must fail semantically, not merely fail freshness; restored copies pass. Runner must reject a surviving mutation and missing tools. |
| 6 | `tests/safe_table.rs`, this plan: I64/capacity-four common-domain traces, compare all mappings/old values; separate fifth-key refusal; retain generic tests | Flash; R | `cargo test --locked --test safe_table` | In a throwaway `src/map.rs`, replace the displaced-value return with `None`; unchanged trace must fail. Safe full-refusal test must not call the unbounded Map insert as if it could refuse. |
| 7 | `proof/corpus/OptimizedTableRelation.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: formal logical projection/common-domain relation, joint witnesses and capacity mismatch counterexample; **MISSING module** | Sol; B | `lake build` (cwd `proof/corpus`) | Mutate the logical projection to drop a live entry; unchanged witnessed lookup/size correspondence fails. Grade is proved mathematics about layouts, only model/anchor for actual Map. No assumed Map transition theorem. |
| 8 | `src/table.rs`, `src/table/scan.rs`, `tests/both_tag_paths_agree.rs`: move the existing scanner unchanged to a shared module; preserve exports and target selection; **MISSING module** | Flash; R | `cargo test --locked --test both_tag_paths_agree` | Corrupt one returned lane bit in the executing scanner; unchanged scalar oracle fails. Fix fixtures to include actual EMPTY/TOMB, one-live and one-free lines. No algorithm or performance claim. |
| 9 | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: extract actual shared SWAR match/free bodies only | Flash; E | `proof/regen.sh --check` | Mutate a real SWAR mask constant; unchanged generated artifact must fail freshness. Unsupported calls/axioms stop the task, not trigger a proof-only scalar rewrite. |
| 10 | `proof/corpus/ScanRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: exact masks for all 16 lanes/all byte values; **MISSING module** | Sol; B | `lake build` (cwd `proof/corpus`) | Mutate generated gather or free polarity; unchanged exact-mask theorem fails; restored copy passes. This proves SWAR only, not SSE2 dispatch or Map. |
| 11 | `docs/spec/storage-refinement.md`, this plan: gate-first design for the actual scalar probe/transition core and heap adapter, constructor/preservation/refusal obligations | Astra owner, Sol read-only verdict; no build lane | `git diff --check` | Requirement review must reject a proposal that assumes the desired heap simulation, omits unwind, equates full with OOM, or selects a proof-only path. Semantic acceptance command **MISSING** until the memory semantics/tool route is selected; this row cannot close optimized refinement. |

The scanner sequence is a follow-up after S6 and the shared assurance barrier, not a prerequisite
invented for S4/S5 or a substitute for the heap bridge. Scalar probe geometry (`Geom`, `Descent`,
`group_at`) is the next candidate after scan, but contains `assert_unchecked` and finite counters:
do not allocate its implementation before the last design row specifies actual-source extraction,
overflow bounds and rejecting controls. Serial `CountTable` partition/count refinement follows that
as a distinct contract; live workers/SSE2/unsafe heap semantics need separately approved tool routes.

### Integration barriers and exact blockers

1. Run `lake build` in `proof/corpus` as the single next baseline gate; then implement the S6 runner,
   execute its controls, integrate the focused trace tests, and obtain Sol's relation/premise verdict.
2. Before optimized-source work, complete S7 on that identity: all four AGENTS §6 Rust commands,
   `proof/regen.sh --check`, `lake build` in both packages, and named controls, serialized. Missing
   tools block. The constructor needs the patched tools documented in `proof/README.md`; record
   `AENEAS_HOME`, do not substitute the unpatched release silently.
3. S8 is a separate gate-design dependency, not waived: contrary to the handoff's installed wording,
   `tools/check-claims.py` and `tools/check-assurance.py` are absent at this base. The latter command
   `python3 tools/check-assurance.py` exits 2 (file not found). Lead must authorize a separate
   assurance task and correction to the out-of-scope handoff/AGENTS wiring. Proposed separate tasks:
   - Claim gate: `tools/check-claims.py` plus the existing ledger selected by its owner; gate
     `python3 tools/check-claims.py`; delete a required joint witness/audit binding in a throwaway
     fixture and require rejection. Gate/ledger design is **MISSING**, not dispatch-ready proof work.
   - Perimeter gate: `tools/check-assurance.py` plus this plan; gate
     `python3 tools/check-assurance.py`; inject unlisted production source, a 1,501-line owned file,
     a thirteenth immediate source file and a >5,000-line trusted checker population, requiring
     rejection. Bind all production `src/`/`tests/`, inventory external trust separately; do not
     exclude optimized files to turn the size check green. Existing over-limit populations require
     a separately frozen structural plan, not an exemption here.
   - Gate wiring: `AGENTS.md`, `docs/flash-handoff.md`; gate `git diff --check`; requirement review
     checks that missing checker execution cannot be reported as S8 closed. No semantic grade.
   These are separate serial waves owned by Flash except boundary/claim decisions, which require
   Astra/Sol. Checker tasks take no Rust/Lean lane unless their approved design invokes those tools.
4. Integrate shared scanner, extraction, then mask proof in order. Re-run S7/S8 and controls on one
   identity. Stop before claiming the optimized relation: no real heap-to-layout simulation exists.

**Sol trigger:** the relation changes theorem premises/attribution. Required decision: are the
joint `MemoryValid`/`PlacementValid` witnesses and operation domains sufficient and non-vacuous,
and which actual Rust semantics could discharge the heap simulation without an extra axiom?
No Sol verdict is supplied by this architecture decision. Source observations needing scrutiny:
`Map::clear` drops before blanking tags; `rehome` temporarily removes arrays while counters remain;
generic callbacks may unwind. These are proof obligations, not a completed safety audit.

**Residual claim:** only the existing safe/kernel refinements are checked. No performance measurement,
whole-Map equivalence, generic trait correctness, CountTable overflow/scheduling theorem, compiled
execution guarantee or ring-0 guarantee is added. A stored-state kernel consumer also needs its real
read/compute/write body and exact failure contract selected before composition work is dispatchable.

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
