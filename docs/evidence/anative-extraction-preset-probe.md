# Extraction probe — `--preset aeneas` on the current source

- **Task:** `APEIRON-ANATIVE-SPIKE-PRESET` (extraction feasibility probe)
- **Revision:** `c450c222905c050f793a23332708c230c658f82c`; working tree dirty
- **Grade:** `anchor` — a probe receipt. The generated file carries 23 axioms and is **not** an
  accepted extraction. No source, governing document or committed artifact was modified.

## Question

Does the current source (per-face address rings, scalar-arena locator, growable table) translate
under the reproducible historical Charon configuration `--preset aeneas`, or does the failure that
motivated the explicit monomorphizing option set (`APEIRON-ANATIVE-AEN-PIPE`) persist?

## Result

`--preset aeneas` **succeeds structurally** on the current source. The explicit monomorphizing
option set is the cause of the supplied red, not the locator or a generic-binder gap.

| Stage | Command (throwaway output) | Exit | Result |
|---|---|---|---|
| Charon | `charon rustc --preset aeneas --dest-file /tmp/.../preset.llbc <43 --start-from roots> -- --edition 2021 --crate-type lib --crate-name apeiron_word proof/extract/lib.rs` | 0 | `preset.llbc` (11 983 294 bytes) |
| Aeneas | `aeneas -backend lean -dest /tmp/.../out -warnings-as-errors -no-progress-bar /tmp/.../preset.llbc` | 0 | `out/Preset.lean`, 525 467 bytes, 24.4 s |

Generated content: **450 top-level definitions, 23 axioms.** The real production bodies are present,
including `table.Table.get_at`, `table.Table.reserve_rebuild`, `table.Table.materialize_rebuild`,
`table.Table.insert_with_at`, and the scalar-arena locator (`locator_core` referenced 878 times,
`Directory` 237, `reserve_for` 51, `walk_residue` 5). There is no `type_var_id` failure and no
`TArray`/`TSlice` failure under the preset.

## Why the explicit option set failed

The option set of `APEIRON-ANATIVE-AEN-PIPE` omits `--ops-to-function-calls`,
`--index-to-function-calls` and `--no-gen-tuple-structs`. Those three are load-bearing for **every**
array/slice index and array repeat in the tree, not only for the locator. The recorded E1 first
errors are all of this one class:

| Error | Site (real source) |
|---|---|
| `Could not find core::array::repeat` | `src/coord/locator_core.rs:1024` (`[0u32; MAX_LEVELS + 1]`), `src/table/scan.rs:65`, `src/table/partition.rs:51` |
| `expand_symbolic_value_no_branching: unexpected type: TSlice` | `src/coord/locator_core.rs:657` (`words[index]`), `:662` (store), `:159`, `:395`, `:414`; `src/table.rs:212`, `:246`, `:260` |
| `expand_symbolic_value_no_branching: unexpected type: TArray` | `src/table/partition.rs:72`, `:78`, `:104`; `src/table.rs:1495` |

These are pre-existing production sites, including the historically accepted word kernels; the
candidate configuration breaks array lowering globally. The spec's own prediction for
`index_to_function_calls` ("unsupported indexing blocks the candidate") is confirmed.

## The remaining 23 axioms

Aeneas models `Vec` as a functional `Slice`-backed value (`Vec.new`/`push`/`index_usize`/`set`/`len`;
`alloc.vec.Vec.with_capacity` ignores capacity). It does not model reservation, capacity or the std
combinators the growable table and resource path reach, so it emits an `axiom` per external.

| Class | Axioms | Source family |
|---|---|---|
| Vec reservation/growth | `alloc.vec.Vec.capacity`, `try_reserve`, `try_reserve_exact`, `truncate`, `clear`, `is_empty`, `alloc.collections.TryReserveError` | `src/table/resources.rs`, `src/coord/locator_core.rs` rollback/materialize |
| allocator glue | `core.mem.size_of`, `core.mem.drop` | `Vec` growth |
| integer conversion | `Usize…TryFromU64…try_from`, `Usize…TryFromU32…try_from` | `u32::try_from`/`usize::try_from` sites |
| `Option` combinators | `is_none_or`, `as_ref`, `and_then`, `filter`, `OptionShared0T.copied`, `Try.branch` | table/probe |
| `Result` combinators | `is_err`, `ok`, `and_then` | table |
| `usize` arithmetic | `saturating_mul`, `div_ceil`, `checked_next_power_of_two` | `src/table/state.rs` (`slots_for`), `src/table/partition.rs` |

No `sorry`/`admit`; the 23 are the fail-closed audit target. The historical accepted
`proof/ApeironWord.lean` (HEAD, 1029 lines; worktree, 3511 lines) has **zero** axioms, so the
current reach set is what introduces these.

## Implication for the direction

1. The reproducible `--preset aeneas` configuration must be restored; the explicit monomorphizing
   candidate is rejected by its own gate. This removes the rationale for a Charon
   monomorphization/preset bypass.
2. The dominant axiom class is the **growable-table resource machinery** (reserve/capacity/truncate),
   which the canonical fixed-shard object does not have: refusal on a full fiber needs no growth.
   This is independent, tool-level support for the quotient principle's fixed-shard bounded-fiber
   realization — it retires the largest, least-modelable axiom class rather than modelling it.
3. The remaining classes (`TryFrom`, `Option`/`Result` combinators, `usize` ops) are removable by
   source reshaping to explicit matches and arithmetic, or by a checked model.
4. The scalar-arena locator itself translates under the preset, so "retire the trie" is an algebraic
   and cost decision, **not** an extraction prerequisite. The corrected claim is narrower than the
   handoff's.

## Digests

| Input | SHA-256 |
|---|---|
| `src/coord/locator_core.rs` | `b4699816521d89623988e4e92ee687b33d7243b3a70634bfbff1c8cd7e00eee8` |
| `src/coord/address.rs` | `a8c5c08164c1c007a72e3089f964cb2cc2167f864f76414e641c2b9e563c70e6` |
| `src/table.rs` | `b253c60f442939a6553800b1b68cf05fc73db1947db491272d280f4a18e9baeb` |
| `proof/regen.sh` | `d39c7ef7f120b1ef118922c0a735670bde6c43cb1a942f8a4053a9c70e7d4600` |
| `proof/extract/lib.rs` | `94cc97cabe97d790b669dd44958025735694bd6a2c60b9fa7849cedba2b7e5b1` |
| `target/aeneas/charon` | `747e165bc62bb963f8d741e8c193c721a4b94555bcb1f9f86c79090fd7a59d0e` |
| `target/aeneas/charon-driver` | `207b223af35cf9a05227b329819b73532e94f655ccdefd64e77b755a15fdf297` |
| `target/aeneas/aeneas` | `33d0ce1a1cc2f1d3935c0cd2ab34f283797da61c531ed6d4e09d5c97b0d0228a` |
| generated `out/Preset.lean` | `27202a968e268a5172fd4fc270d857d3ec3d4633af07a661e852da239c673892` |
| `aeneas.log` | `dfd55e4f5c3948d472dca8e24d4def7a193711b68b37934a2d9e27b49b6845b6` |
| `charon.log` | `e6e5144892651085ecd93452f1a2db1c369254d0d8aa9cfbaeb0ed297c8c2756` |

Tools: charon `0.1.254 (b104e24fea7d721b71e6c39fd70f26ff20bc0980)`, aeneas `505b6ca-dirty`.

## Exact axiom-to-site map (reached source)

`-split-files` puts all 23 in `FunsExternal_Template.lean` / `TypesExternal_Template.lean`, so a
model file is the intended hook. Sites below are the reached uses.

| Axiom | Reached sites |
|---|---|
| `Vec.try_reserve` | `coord/locator_core.rs:951,954,956,1133`; `coord/locator.rs:677`; `table/locator.rs:422` |
| `Vec.try_reserve_exact` | `table/resources.rs:198-203` (central helper); `coord/locator_core.rs:224,963` |
| `Vec.capacity` | `table.rs:876,877,903,907,911`; `table/locator.rs:424` |
| `Vec.truncate` | `coord/locator_core.rs:1159-1161` |
| `Vec.clear` | `coord/locator_core.rs:1060,1067`; `table/state.rs:71` |
| `Vec.is_empty` | `coord/locator_core.rs:1002`; `coord/locator.rs:415`; `table/locator.rs:41`; `table/partition.rs:38,67` |
| `TryReserveError` | error type of the above |
| `core.mem.size_of` | `table/keyrep.rs:297,332` (`size_of::<u64>`); `table/state.rs:459` (`size_of::<FlatSlot<R,V>>`) |
| `Option.is_none_or` | `table.rs:840` |
| `Option.as_ref` | `table/locator.rs:564,605` |
| `Option.and_then` | `coord/locator_core.rs:923,924,943`; `table.rs:1501`; `table/parallel.rs:201` |
| `Option.filter` | `table.rs:804,825` |
| `Option.copied` | `table.rs:366`; `table/parallel.rs:151`; `table/locator.rs:155` |
| `Try.branch` | residual `?` conversion on `Option` (one site) |
| `Result.is_err` | `coord/locator_core.rs:935-937,951,955,956,1133` |
| `Result.ok` | `table.rs:1034`; `table/locator.rs:118` |
| `usize::try_from` | `table.rs:606,1029,1033`; `table/locator.rs:99,112` |
| `u32::try_from` | `coord/locator_core.rs:515,519,935-937,1040,1141`; `coord/locator.rs:486,495,537,674`; `table/locator.rs:93,118,420` |
| `Usize.saturating_mul` | `table.rs:680` |
| `Usize.div_ceil` | `table.rs:607,1483`; `table/state.rs:43` |
| `checked_next_power_of_two` | `table/state.rs:44` |
| `core.mem.drop` | drop glue on owned `R`/`V` in `Region`/`Flat` |

Classes: **allocation/growth** (the first eight rows) is the fixed-shard object's removed machinery;
**combinators/conversions** (the rest) is exact-match reshaping.

## Reshape round 1 — pure combinator/conversion class removed (23 → 11)

Source reshaping replaced the unmodeled **pure** operations with exact equivalents. All Rust gates
pass on this state: `cargo test --workspace --all-features --locked` (258 passed, 0 failed),
`cargo clippy … -D warnings` exit 0, `cargo fmt --all -- --check` exit 0. Re-extraction under
`--preset aeneas` translates (charon exit 0, aeneas exit 0) with **444 defs and 11 axioms**,
generated file sha `b3644dbf789ce6a2e9d72a0d44c3177642fa3efe00b973eaa623c7b30519d5e3`.

Removed (12 axiom names): `Usize…TryFromU64…try_from`, `Usize…TryFromU32…try_from`,
`core.num.Usize.saturating_mul`, `core.num.Usize.div_ceil`,
`core.num.Usize.checked_next_power_of_two`, `core.option.Option.is_none_or`,
`core.option.Option.and_then`, `core.option.Option.filter`, `OptionShared0T.copied`,
`core.result.Result.ok`, `core.result.Result.and_then`, `alloc.vec.Vec.is_empty`.

Mechanisms: `x / n + usize::from(x % n != 0)` for `div_ceil`; an explicit doubling loop for
`checked_next_power_of_two`; `checked_mul(k)` + fallback for `saturating_mul`; `(x as usize)` after
a `> u32::MAX as usize` guard for `try_from`; explicit `match`/`if` for the `Option`/`Result`
combinators and iterator `first().copied()`; `Option` `?` replaced by `match`; a named
`Table::keep_unoccupied` for `Option::filter` so the body stays out of the joined loop context.
Extractor-shaped forms are kept and the corresponding clippy lints are allowed, with a reason, at
the top of `src/table.rs`, `src/table/parallel.rs`, `src/table/partition.rs`,
`src/coord/locator.rs` and `src/coord/locator_core.rs`.

### Aeneas shape is load-bearing

The original source was deliberately shaped for the extractor, so a semantically exact replacement
is not automatically translatable. Three naive replacements regressed translation and were repaired
by restoring the single-exit form:

| Change | Aeneas failure | Repair |
|---|---|---|
| `let slots = slots_for(..)?;` inside `build_at`'s `while` loop | `Early returns inside of loops` | restored the in-expression `match` (no `?` in the loop) |
| guarded `match flat { Some(i) if !occupied[i] => .. }` inside the nested planning loop | `Could not match the contexts` | named `keep_unoccupied` call (opaque body, like the original closure) |
| `match &self.directory` for `Option::as_ref` | macro-origin `Not implemented yet` in `linked_handle` | restored `self.directory.as_ref()?` |

This confirms the class the earlier bridge hit: loop-context joins, in-loop early returns and inline
guards are the extractor's shape boundaries, independent of the algebra.

## Residual — 11 axioms (Path 2b)

| Axiom | Class |
|---|---|
| `Vec.capacity`, `try_reserve`, `try_reserve_exact`, `truncate`, `clear`, `TryReserveError`, `core.mem.size_of`, `core.mem.drop`, `Result.is_err` | **allocator/growth** — removed by the fixed-shard, bounded-fiber realization (no reserve, no capacity, no growth, no rollback truncation) |
| `Option.as_ref`, `Option.…Try.branch` | the `self.directory.as_ref()?` site in `TableLocator::linked_handle`/`linked_handle_query`; a `&self.directory` match regressed translation, so it stays until the re-founding removes the field |

The allocator class is not reducible by exact substitution: Aeneas models `Vec` as a functional
sequence with no reservation, capacity or length-shrink. Removing it is the storage re-founding.

## Reshape digests (round 1)

| Input | SHA-256 |
|---|---|
| `src/table.rs` | `9ec504cec3d244683936779e2e07f5d83aafcb9b8ae8e9ab2043ec89e55ac548` |
| `src/table/locator.rs` | `72c9d11ce5699c8ad1cd9ab35d855416617bd2bf8a646248bccb2e2bb7ce7c83` |
| `src/table/parallel.rs` | `08f42f76867e80e8ca4696e045b84461a08c8e3c3185670cc10f2bceed5dbdac` |
| `src/table/partition.rs` | `54a80b996ea9df07fff9077c54a20f8e436cc8843c2daae543f8d77116708edd` |
| `src/coord/locator.rs` | `93833607fe434b5c2f4788e200800388ed13c9b47bd39e832e0f4d0dd859fd2d` |
| `src/coord/locator_core.rs` | `1d85a66893cd20fbdff03c5d58e649b1844fb70f3edc42fc591638caf144c95d` |
| `src/table/state.rs` | `4a72f3834f067b62642daeeb48e4cbffde5b617e5b9132dd2eebabf176c423a0` |
| `src/table/keyrep.rs` | `2b0db4d24967322b0cc95dd16fd68ec7dae734f7502c24c81db2a7ea242ee03c` |
| `proof/regen.sh` | `88ebc85ccb46f1f5a686cbbe1e51a20401aecf8cdfe7184cae1b814d653089a8` |
| generated `out/Preset.lean` | `b3644dbf789ce6a2e9d72a0d44c3177642fa3efe00b973eaa623c7b30519d5e3` |

## Adapter-out-of-cone probe — zero-axiom read/kernel extraction

Removing the write/growth adapter roots (`*_table_build_at`, `*_prepared_shard_apply`,
`*_apply_insert`, `*_relocate_*`, `*_apply_pending_and_commit`, `*_publish_prepared`,
`word_capacity_bound_holds`, `word_schedule_is_complete`; 27 of 43 roots) and re-extracting under
`--preset aeneas` gives:

| Metric | Full 43 roots | Reduced 27 roots |
|---|---|---|
| definitions | 444 | **154** |
| axioms | 11 | **0** |
| `table.Table.get_at` | present | present |
| `reserve_rebuild` / `try_reserve` / `Directory::link` | present | absent |

Generated file sha `9ead8f3319ee0b236aad4a56d4bf7fd2b2755c61dcab3d0499e778b23abbaee4`. This is the
design's adapter-out-of-cone boundary: the **read/query face and the word/locator kernels extract
axiom-free**, while the fallible write/growth adapter is the named `ResourceAdapterSimulates`
residual. It is evidence for that boundary, not an accepted extraction (the production root set and
the Lean bridge are unchanged; the write face is not established).

## Verdicts on the residual (not this probe's work)

Two escalations were consulted on how to remove the 11 axioms.

- **Astra (`anchor`)**: select hybrid C. The resource family (`Vec.capacity/try_reserve/
  try_reserve_exact/TryReserveError`) is blocked because the pinned `Vec` model identifies a vector
  by its element list only, so it cannot carry capacity or allocation outcome; a definition alone
  cannot preserve two observations of equal-contents/different-capacity vectors. `size_of` needs
  target-layout evidence. The ordinary names (`is_err`, `as_ref`, `Try.branch`, `truncate`, `clear`,
  `drop`) are simplifiable or modelable.
- **Sol (contract/attribution)**: the fixed-shard selection in `docs/kernel-boundary.md:258-272` is
  historical ("Prior selection"); the current boundary (`:279-285`) and `docs/direction.md:39-45`
  select the growable table. Adopting fixed shards therefore requires amending `docs/direction.md`,
  `docs/kernel-boundary.md`, `docs/flash-handoff.md` and the ANATIVE lifecycle, and retiring
  `ReserveBeforeRetainContract`, the eight `LocatorReservationReceipt` categories,
  `NativeLocator.TableStorageCoverage`, `Growable/{Relocation,Rebuild,Resources,Representation}`, the
  `injection` failure controls and W7 — with a recorded performance blocker
  (`docs/flash-handoff.md:45-52`). **No third zero-axiom path exists** for the growable table's
  allocating methods under the pinned backend; excluding the adapter gives a zero-axiom core but
  leaves `ResourceAdapterSimulates` unproved.

Consequence: the 11-axiom residual is not a source-shape defect and is not removed by the quotient
storage re-founding (which the current ANATIVE spec does not select). The decision is the lead's:
accept the adapter exclusion at the existing `ResourceAdapterSimulates` boundary (zero-axiom read/
kernel cone, write face open), or authorize a governing-document amendment plus the resource/
backend representation work.

## Trust boundary

A clean structural translation is entry-encoding feasibility, not semantic preservation, generic
refinement or end-to-end correctness. The generated file's 23 axioms keep the fail-closed cone red;
no claim of `proved` follows. The probe command used a throwaway destination; no production artifact
was written.
