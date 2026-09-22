# ANATIVE W4-E-I7 extraction-lane feasibility

Result: the lane is **red**. Charon translates the new typed-face/locator unit and emits the LLBC;
Aeneas aborts in `translate_type_decls` on an arrow type. No `proof/ApeironWord.lean` was written.

```text
SCOPE:      proof/regen.sh SOURCE_FILES and two --start-from roots; proof/ApeironWord.lean left
            byte-identical; this receipt. Completion: the lane reaches or fails on the new
            typed-face/locator modules with a recorded first error.
REGISTER:   local obligation (refinement/extraction lane). Exclusions: src/, tests/, examples/,
            docs/, Cargo.toml; the Growable summary-shape debt is not touched.
CORPUS:     proof/corpus/Native/LocatorSpec.lean, extraction_targets_exact (lines 874-906), read;
            no theorem, grade or corpus declaration changed.
REFERENCE:  LocatorSpec.requiredExtractionTargets; 13 of its wrapper roots do not exist in
            proof/extract/lib.rs and cannot be expressed without new wrappers.
GATE:       flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=6G
            -p MemorySwapMax=0 -- env LEAN_NUM_THREADS=1 timeout 2400 bash -lc './proof/regen.sh'
            -> exit 2; first Aeneas error table.rs 560:4-615:5; fatal table/locator.rs 268:0-271:1.
GRADE:      measured (one lane-locked run); extraction red, not proved, not tested.
```

## Delta

`proof/regen.sh` `SOURCE_FILES` adds the five missing modules and the `include!`/`#[path]` comment is
corrected:

- `src/table/probe.rs`, `src/table/locator.rs` (textual `include!` from `src/table.rs`)
- `src/coord/address.rs`, `src/coord/locator.rs`, `src/coord/locator_core.rs` (`#[path]` from
  `src/coord.rs`, then `src/coord/locator.rs`)

`--start-from` adds the only two unrooted wrappers in `extraction_targets_exact` that already exist
in `proof/extract/lib.rs`:

- `crate::word_stored_table_build_at`, `crate::word_regen_table_build_at`

Thirteen targets in `extraction_targets_exact` cannot be expressed: `word_face_address`; the eight
`word_{stored,regen}_locator_{decide,publish,relink,entry}`; and the four
`word_{stored,regen}_{build_at,get_each_at}_parallel`. No wrapper for them exists in the extraction
unit. The typed table paths are nevertheless exercised: the existing
`word_stored_prepared_shard_apply`/`word_regen_prepared_shard_apply` roots reach
`apply_prepared_shard_at` -> `reserve_locator_path` -> `Locator::reserve_for`, and
`word_*_table_build_at` reaches the build and parallel paths.

## Command and outcome

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=6G \
  -p MemorySwapMax=0 -- env LEAN_NUM_THREADS=1 timeout 2400 bash -lc './proof/regen.sh'
# exit=2
```

Tools: charon `0.1.254 (b104e24fea7d721b71e6c39fd70f26ff20bc0980)`, aeneas `505b6ca-dirty`,
rust `nightly-2026-08-18`, lean `v4.31.0`. Charon succeeded and wrote `proof/apeiron_word.llbc`
(12,066,259 bytes); the script aborted before its normal `rm`, so that artifact was removed after
the run. All `[Error]` lines below are Aeneas reading that LLBC.

Aeneas errors, in order:

1. `Early returns inside of loops are not supported yet` — `src/table.rs` 560:4-615:5,
   `Table::selected_lanes`.
2. `Early returns inside of loops are not supported yet` — `src/table.rs` 814:4-857:5,
   `Table::observe_receipt`.
3. `Returns inside of nested loops are not supported yet` — `src/coord/locator_core.rs`
   659:4-706:5, `Directory::stage_reserved`.
4. `Arrow types are not supported yet` — `src/table/locator.rs` 268:0-271:1, type decl
   `apeiron_word::table::QueryIdentity`, then `Could not translate type decl ... because of previous
   error`, then uncaught `Aeneas.Errors.CFailure` in `SymbolicToPure.translate_type_decls`.

The arrow type is the field `_carrier: PhantomData<fn() -> (H, R)>` on `QueryIdentity`. Type-decl
translation is crate-wide and independent of the `--start-from` roots, so this blocks every locator
extraction regardless of root selection.

## thread_local / injection-seam risk

No `thread_local`, `LocalKey`, or `LocalKey::new` error appears. The two seams
(`src/coord/locator_core.rs:311`, `src/table/parallel.rs:101`) are statically reachable from the
existing and new roots, but Aeneas aborts in `translate_type_decls` before function translation, so
its `LocalKey::new` rejection is **not reached and not confirmed**. This is a residual obligation,
not a confirmed blocker.

## Generated file

`proof/ApeironWord.lean` is unchanged by this run: sha256
`9af54e7d2467a5a19ebccfd7fd70691072a17fda76e78e2366c3832675749d3e`, identical to the pre-run
working-tree file. No new digest exists because regeneration failed. `proof/regen.sh --check` was not
run; it applies only after successful regeneration. The Growable summary-shape debt in `proof/corpus`
was not exercised — the run stops before any `lake build`.

## Residual and next bounded task

- First blocker (source, out of this task's scope): replace the `PhantomData<fn() -> (H, R)>` marker
  on `src/table/locator.rs::QueryIdentity` with an Aeneas-translatable variance marker.
- Then the loop-return bodies: `Table::selected_lanes`, `Table::observe_receipt`,
  `Directory::stage_reserved`.
- Then the `reservation_failure` seams, expected to need a source-side feature gate like
  `read-stats`.

Next bounded task: a new source assignment to remove the `fn()` phantom from `QueryIdentity`, then
re-run this same lane-locked command; the injection-seam gate is a later slice.
