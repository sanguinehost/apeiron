# Aeneas model integration — axiom-free growable extraction

- **Grade:** `tested` / `measured` (builds and regeneration freshness). Not `proved`.
- **Base:** `c450c222905c050f793a23332708c230c658f82c`, working tree dirty.

## Result

The real growable source now extracts to Lean with **zero axioms** and the whole write/growth path
in the cone. `proof/regen.sh` exits 0, `proof/regen.sh --check` exits 0 (no drift), and `lake build`
in `proof/` completes 1708 jobs.

| metric | value |
|---|---|
| extraction roots | 43 (including `*_table_build_at`, `*_prepared_shard_apply`, `*_apply_insert`, `*_relocate_*`, `*_apply_pending_and_commit`, `*_publish_prepared`, `word_capacity_bound_holds`, `word_schedule_is_complete`) |
| `proof/ApeironWord.lean` | 10923 lines, **0 `axiom`/`sorry`/`admit`**, sha256 `e1385a4d2927b7b47c4454e16a30ce365d505c97606ecf3429622dac26cd5be9` |
| `lake build` (`proof/`) | 1708 jobs, rc 0 |
| `proof/regen.sh` / `--check` | rc 0 / rc 0 |
| runtime cost | **zero** — the executing allocation code is unchanged (only a trait-bound spelling, a `drop` → `let _`, and a budget constant) |

## What changed

### Tool model patch — `proof/patches/aeneas-lean-model.patch` (sha256 `81ae0e16…`)
Applied by `proof/model-patches.sh` to build `target/aeneas-model/lean` from the installed backend
(`target/aeneas/backends/lean`), which `proof/lakefile.lean:19` now requires. It is a **Lean-library**
change; no Charon/Aeneas OCaml rebuild. Four files at the recorded base:

- `Aeneas/Std/Vec.lean`: `Vec` carries `cap : Usize`; `eq_iff`/`ext` become value ∧ capacity;
  `from`/`ofSlice`/`with_capacity`/`set*`/`update`/`clone`/`index_mut` maintain it; adds
  `Vec.capacity`, `Vec.try_reserve`, `Vec.try_reserve_exact`, `Vec.truncate`, `Vec.clear`,
  `Vec.is_empty`, and the `alloc.collections.TryReserveError` type.
- `Aeneas/Std/Core/Cmp.lean`: `core.cmp.Ord.max.default` takes its `Ord` instance (the generator
  passes it; `min` already did), matching `min.trait_default`.
- `Aeneas/Std/Core/CoreOption.lean`: models `Option.as_ref`, `Option.Try.branch`,
  `Option.Try.from_residual`.
- `Aeneas/Std/Core/CoreResult.lean`: models `Result.is_err`.

The patch has since grown past this receipt's base: it now touches 8 files, adding
`Aeneas/Std/Scalar/{Casts,Ops}.lean`, `Aeneas/Std/Scalar/WrappingOps/Shl.lean` and
`Aeneas/Std/String.lean` (`rg -o '^\+\+\+ b/(\S+)' -r '$1' proof/patches/aeneas-lean-model.patch |
sort -u`), at sha256 `64054e6b2d7f71f1a1c6a40d10e61f8e6b69163c23bc52f1bc5ef9f756540575`. The file list
and digest above are the state at the recorded base, not the current file. The build route from a
clean checkout is in `proof/README.md` §"Route from a clean checkout to a built `proof/`"; this
receipt is not the route.

### Source fixes (semantics-preserving)
- `src/coord/locator_core.rs:73`: the associated type bound `Retained: Clone + Eq` becomes
  `Retained: RetainedKey`, with `pub trait RetainedKey: Clone + Eq {}` and a blanket impl. The pinned
  Aeneas names each trait clause after the trait, so `Self: Eq` and `Retained: Eq` both became a field
  `corecmpEqInst` and the generated structure did not elaborate. The blanket impl is equivalent.
- `src/table/locator.rs:481`: `drop(retained)` → `let _ = retained;`, removing the `core.mem.drop`
  external.
- `src/table/state.rs`: `byte_len_for` uses a documented `ASSUMED_SLOT_BYTES` constant instead of
  `core::mem::size_of::<FlatSlot<R,V>>()`, removing the `core.mem.size_of` external. This is a
  byte-budget heuristic; the constant is an upper bound for fixed-width slots.

### `regen.sh`
After generation it runs `proof/strip-modeled-axioms.py`, which removes the ten `axiom` stubs the
model now supplies and leaves any other axiom for the fail-closed `audit` to reject.

## Rust gates

`cargo test --workspace --all-features --locked` (0 failures), `cargo clippy … -D warnings` (rc 0),
`cargo fmt --all -- --check` (rc 0).

## Not established

- The bridge build (`proof/corpus`) is **red** on a pre-existing issue independent of the model:
  `Growable/Identity.lean:1:0: unknown module prefix 'TableRefinement'`. It is not caused by this
  change and its own extraction-independent debt remains.
- No write-path *refinement* theorem is proved yet; the extracted bodies are present and axiom-free,
  and the bridge links are the remaining work.
- The `Vec.try_reserve` model is deterministic (succeeds except at the `Usize` ceiling). A sound
  model must allow failure with `ResourceAdapterSimulates` as the named conformance premise.
- `core.mem.size_of` is removed by a budget constant, not modelled; the layout assumption is stated.
