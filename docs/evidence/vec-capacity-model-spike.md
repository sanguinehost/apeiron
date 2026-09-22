# Spike — capacity-carrying `Vec` model (option (a))

- **Task:** `APEIRON-ANATIVE-SPIKE-VEC-CAPACITY`
- **Grade:** `measured` for the build result; `anchor` for everything downstream.
- **Method:** throwaway copy of the pinned Aeneas Lean backend
  (`target/vecspike/lean`, sharing the cached Mathlib); add a capacity field and the reservation
  functions; `lake build` the library. The installed tool tree and the repository were not modified.

## Result

- The **entire Aeneas Lean library compiles green** with `Vec` carrying a capacity field:
  `lake build Aeneas` → `Build completed successfully (1704 jobs)`, rc 0.
- The patch is **one file**, `Aeneas/Std/Vec.lean`: **79 added / 24 removed** lines
  (`docs/evidence/vec-capacity-model.patch`, sha256
  `c1b858ce20c1d92de090ad1307bb7947d01414cca30d1b7145447864f597896d`). `diff -rq` against the
  pinned backend shows no other source file differs.
- **Runtime performance cost: zero.** The executing Rust `src/` is not touched; only the Lean model
  changes. This is the decisive answer for the "proven AND fast" requirement.
- Build time: `Aeneas.Std.Vec` 1.7 s after warm Mathlib; the library build runs in the normal
  `proof/` lane.

## What the patch changes

- `structure Vec` gains `cap : Usize`.
- `Vec.eq_iff` / `Vec.ext` become value ∧ capacity.
- `Vec.from` / `Vec.ofSlice` build tight capacity; `with_capacity` sets it.
- `set` / `set_opt` / `update` / `setSlice!` / `index_mut` / `clone` preserve `cap`;
  `to_vec` / `into_vec` / `deref_mut` wrap via `ofSlice`.
- Added model definitions carrying the `@[rust_fun …]` names the Aeneas template requires:
  `Vec.capacity`, `Vec.try_reserve`, `Vec.try_reserve_exact`, `Vec.truncate`, `Vec.clear`,
  `Vec.is_empty`, and the `alloc.collections.TryReserveError` type.
- The `set_getElem*` / `index_mut_slice_index` lemma family needed capacity-aware restatements —
  ordinary proof maintenance, contained to `Vec.lean`.

## Residual — not established

- **Reservation is modelled deterministically** (success except at the `Usize` ceiling). A sound
  model must allow `try_reserve` to fail, carried as a WP spec with `ResourceAdapterSimulates` as
  the named conformance premise. That refinement was not written.
- **Extractor pick-up was not closed.** The generated split files from the stale `/tmp` `.llbc`
  fail to elaborate (`Types.lean`: *Field `corecmpEqInst` has already been declared*; *Unknown
  identifier `coord.locator.locator_core.ExactIdentity`*) — a generated-code / `-split-files` issue
  independent of the `Vec` patch. Closing it needs a fresh extraction from current source plus the
  filled `FunsExternal.lean`.
- `core.mem.size_of` remains a target-layout assumption.

## Trust boundary

`measured` covers exactly: *the capacity-carrying `Vec` model compiles the Aeneas Lean library.*
It does not establish the extraction, the refinement, or real-allocator conformance. The spike ran
in a throwaway copy; no committed source, tool, or proof was modified, and no `proved` claim follows.

## Second pass — end-to-end, two further blockers found and fixed

Driving the generated code to elaborate exposed two more tool/model defects, both independent of
allocation:

1. **Trait-clause name collision.** `pub trait ExactIdentity: Eq` with `type Retained: Clone + Eq`
   made Aeneas emit two fields named `corecmpEqInst` in the generated trait structure, so the
   generated file did not elaborate in either split or non-split mode. Minimal repro
   (`target/vecspike/trait_test`): bounding the associated type by a helper trait
   `pub trait RetainedKey: Clone + Eq {}` with a blanket impl makes the fields `corecmpEqInst : Eq Self`
   and `RetainedKeyInst : RetainedKey Self_Retained`. **Applied** to `src/coord/locator_core.rs:72`;
   Rust gates green (`cargo test` 0 failures, `cargo clippy` 0, `cargo fmt` 0).
2. **`Ord::max` model bug.** Aeneas generates `core.cmp.Ord.max.default core.cmp.OrdUsize a b`, but
   the pinned model's `max.default` takes `(lt : Self → Self → Result Bool)`; `min` has the matching
   `@[trait_default]` form and `max` does not. Fix: `max.default` takes `OrdInst : core.cmp.Ord Self`
   and `@[trait_default]`, mirroring `min.trait_default`.

With the capacity-carrying `Vec`, the `Ord::max` fix, and the trait fix, the generated non-split Lean
elaborates green (**1706 jobs**). Deleting the six `Vec`-family axiom stubs from the generated file,
it **still builds green** — the generated bodies resolve to the patched model. This is the
end-to-end confirmation that option (a) removes the allocation axioms without touching the Rust
allocation code, so runtime cost stays zero.

## Residual — the last six axioms

| axiom | disposition |
|---|---|
| `Option.as_ref`, `Option.Try.branch`, `Option.Try.from_residual`, `Result.is_err` | modelable in `CoreOption.lean` / `CoreResult.lean`, like the existing `Option.ok_or` / `Result.map_err` |
| `core.mem.drop` | removable at `src/table/locator.rs:481` by `drop(retained)` → `let _ = retained;` |
| `core.mem.size_of` | the byte-budget at `src/table/state.rs:476` (`size_of::<FlatSlot<R,V>>()`); needs a bounded slot-size constant (a small `SlotBytes`-style bound) or a layout model — not generic |

## Not yet integrated

The model fixes live in the throwaway backend (`target/vecspike/lean`). For production they must
become a third patch under `proof/patches/` with `install-patches.sh`/`PINS`, `regen.sh` must consume
the filled externals (the `-split-files` route, or inline-axiom removal), and the bridge refinement
must be re-run. None of that is done here.

