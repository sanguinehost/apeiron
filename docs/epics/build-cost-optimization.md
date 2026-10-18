# Build-cost optimization (G1 spatial parallel build)

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

Decision: repair the parallel build's reservation boundary first, then replace the per-key serial
reserve/rebase work with pre-reserved storage and worker-recorded payload locations. The boundary
verdict is Astra session `ses_f32b8e1f9ffeJcp3CB1SFnBxh7` (architecture `anchor`; source inspection,
no proof or measurement of its own). No commit, worktree or protocol change is part of this plan.

## Measured baseline

`micro_spatial`, n1M distinct u64 keys, 16 workers, build median **1311.1 ms**:

| phase | ms | contents |
|---|---|---|
| reserve | 658.1 | `reserve_all_shards` + `preassemble_directory` (three trie walks and one per-key `identity_path` per distinct key) |
| assemble | 630.6 | `rebase_directory` (one trie walk plus one `search_decision` probe per key) + region install |
| kernel | 9.7 | the scoped fill workers |
| qualify/partition/validate/schedule | ~14 | plan, scatter, validate, bins |

Reference: `docs/evidence/perf-ledger.md` measured the same build end to end at
27.7 ms. The full REG-1 campaign completes in 477.79 s
(`docs/evidence/locator-reserve-quadratic-and-reg1-attribution.md`), so this epic is about the G1
ratio, not the budget.

After W1-T1..W1-T3 (`micro_spatial`, same command and host): n1M build median **724.6 ms** —
reserve 688.7 (95.4%), kernel 10.5, assemble 9.6 (1.3%, was 630.6), qualify+partition+validate+
schedule ~14; n100k build median **59.8 ms** (reserve 55.7, assemble 0.54). The per-key rebase walk
and probe are gone; reserve dominates and is the W2 target. Raw output `/tmp/opencode/w1t3-micro.log`;
host i9-14900KF, 32 threads; release profile from the digests above.

## Base and ownership

Base `262bb0a94eff2cc7d5655122a954f44d74de96b6` plus the accepted dirty tree. One in-progress owner
per file; tasks in a wave touch disjoint files. Worktree exception: the accepted dirty tree cannot be
relocated, so tasks run in the supplied checkout with the owner recorded here, and every task is
closed by its gate command, never by a report. Shared lanes: `proof/ApeironWord.lean`, `target/`,
`proof/.lake/`, `proof/corpus/.lake/`.

Extraction note: the scoped executor is not extracted (`proof/ApeironWord.lean` contains no
`build_at_parallel`, `join_or_panic` or `reservation_failure`), but `src/map/table/prepared.rs`,
`src/map/table/locator.rs` and `src/map/coord/locator_core.rs` are, so those tasks regenerate and audit.

## Tasks

| ID | owned files | one gate | state |
|---|---|---|---|
| APEIRON-G1-BUILD-W1-T1 reservation-boundary repair | `src/map/table/parallel.rs` (`14b49724…`), `tests/parallel/main.rs` (`8bca94bf…`) | `cargo test --locked --features injection --test parallel` | closed: gate green 14/14, `cargo clippy --locked --all-features --lib -- -D warnings` exit 0, `cargo fmt --all -- --check` exit 0; seeded lazy-allocation mutation red on the attributable refusal assertion (`tests/parallel/main.rs:976`), restored sha256-verified and green |
| APEIRON-G1-BUILD-W1-T2 placement seam | `src/map/table/prepared.rs` (`7dd26a31…`) + focused internal tests | `cargo test --locked --lib table::prepared_placement_tests` | closed: `apply_prepared_shard_placed` returns `(fresh, slot)`, `apply_prepared_shard` is the bool projection; 2/2 tests green; `clippy --lib --tests` and `fmt` exit 0. Extracted body: regen at the W3 checkpoint |
| APEIRON-G1-BUILD-W1-T3 recording and finalization | `src/map/table/parallel.rs` (`ee35e07b…`), `src/map/table/locator.rs` (`e1e02976…`), `tests/parallel/main.rs` (`cc76ed71…`) | closed: worker-recorded `(input_index, slot)` with pre-scatter record buffers and the `arm_records` seam; `rebase_directory` retired (sole caller); gate 16/16 green, clippy `--lib --tests` and fmt exit 0; seeded `Spill(0,0)` finalization mutation red at the coverage assertion (`tests/parallel/main.rs:805`), sha-verified restore green. Extraction checkpoint (W3-T1) ran; the finalize refusal witnesses landed as W1-T4, which superseded this row's `parallel.rs` and test digests (`8018d8f4…`, `d66073f0…`) |
| APEIRON-G1-BUILD-W1-T4 finalize refusal witnesses | `src/map/table/parallel.rs` (`8018d8f4…`), `tests/parallel/main.rs` (`d66073f0…`) | `cargo test --locked --features injection --test parallel` | closed: `FinalizeArm` enum plus one-shot `arm_finalize`/`consume_finalize` merged into each real condition (`:637` SizeOverflow; `:641`/`:644`/`:655` AllocationFailed); gate exit 0, 20 passed (16 prior + 4 new), re-verified on the live tree at these digests; clippy `--features injection --lib --tests -D warnings` and fmt exit 0; control: `MissingHandle` arm seeded to an `unwrap_or(0)` silent placeholder rejected with the attributable assertion, restored sha-verified; success path byte-identical without the feature |
| APEIRON-G1-BUILD-W2-T1 lookup/staging optimization | `src/map/coord/locator_core.rs` (`f2708a6f…`), `src/map/table/locator.rs` (`a017cae6…`), `tests/native_locator/main.rs` (`20bf1689…`) | `cargo test --locked --features injection --lib --test native_locator --test parallel` | closed: `reserve_or_linked` found-or-staged with terminal-word detection; `linked_handle_query` removed (sole caller migrated); write/commit callers unchanged by inspection; gate exit 0 (lib 98 passed/1 ignored, native_locator 25, parallel 20); clippy/fmt 0; control: found-arm-never-fires defect rejected by the repeat-key fixture (`Some(2)` vs `Some(0)`), restored sha-verified. Residuals: `reserve_all_capacity` duplicates `reserve_for`'s sizing phase by design; extraction checkpoint running (expected byte-identical, the new chain is outside the extraction roots); pre-existing rustfmt drift in the include!d `src/map/table/locator.rs` untouched; the `Cargo.toml` injection-feature doc still names only `reservation_failure` |
| APEIRON-G1-BUILD-W2-T2 scratch recycling | `src/map/coord/locator_core.rs` (`05846ad8…`), `src/map/coord/locator.rs` (`ab1a2271…`), `src/map/table/locator.rs` (`fbd8b78d…`) | closed: Directory-owned idle scratch recycled through the outstanding `StagedPath` on publication and refusal, both consumers included; gate 95/24/16 passed, clippy/fmt 0; seeded clear-before-reuse defect red (exit 101, coverage assertions), sha-verified restore green; extraction checkpoint `2a1138e4…`; bridge + `AxiomAudit` green (3369 jobs). Residual (closed by W2-T3): `src/map/table.rs` found-update drop |
| APEIRON-G1-BUILD-W2-T3 table consumer recycle | `src/map/table.rs` (`b5ddd10f…`), `src/map/table/tests.rs` (`195a5975…`) | closed: both found-update discards route through `Table::recycle_unused_staged_path` to `Directory::recycle_staged`; gate `cargo test --locked --features injection --lib --test native_locator --test parallel` exit 0 (97 passed/1 ignored, 24, 20); clippy/fmt 0; control: seeded `staged = None` drop rejected both pointer/capacity instrument tests with attributable diagnostics, restored sha-verified. Residuals: extraction checkpoint owed for the new helper; the pre-existing "discarded staged arena words are not rolled back" leak is preserved unchanged |
| APEIRON-G1-BUILD-W3-T1 extraction regeneration | `proof/ApeironWord.lean` via `proof/regen.sh` | `proof/regen.sh` then `proof/regen.sh --check` | checkpoint after W1-T3/W2-T2: regen exit 0, `--check` fresh, `2a1138e4…`. After W1-T4/W2-T3/W2-M1 the check was stale and the difference was not cosmetic: the regenerated extraction adds `table.Table.recycle_unused_staged_path` and rewrites the found-update arms of `apply_pending_and_commit`/`build_at` (W2-T3's `src/map/table.rs` is inside the extraction closure); most other hunks are `Source:` line-number comments. Expected repair after regen: `Growable/Refinement.lean` `word_{stored,regen}_apply_pending_and_commit_eq_apply_insert` (they unfold `table.Table.apply_pending_and_commit`), and the `Native/LocatorSpec.lean` source-refinement table entries for those functions must re-check. `proof/regen.sh` exit 0 and `--check` exit 0 at the **new generated `8e76b917…`**; `Growable.ReadLoop`, `Growable.Scan`, `Growable.Representation` and `Native.LocatorSpec` rebuild green against it, while `Growable.Refinement` is red at its two `apply_pending_and_commit` theorems (`:65`, `:88`, `:125`) and is under repair as W3-T3; later control receipts re-anchor to `8e76b917…`. Barrier side: `cargo test --workspace --all-features --locked` exit 0, `cargo fmt --check` exit 0, `clippy --all-targets` exit 0 after fixing `examples/reg1/performance.rs` `too_many_lines` (first run exit 101). The package-wide corpus re-run is externally blocked on the other lane's `Kepler.Certificate` (24 errors), while module-scoped gates are green. See [toolchain incident](../evidence/lean-toolchain-mixing-incident.md) for a local `lake -d` invocation that mixed the nightly toolchain into the corpus cache and its recovery |
| APEIRON-G1-BUILD-W3-T2 bridge audit | none initially | `lake build` in `proof/corpus` | closed at the regenerated identity: `lake build` exit 0 (3817 jobs) and `lake build AxiomAudit` exit 0 (3692 jobs, no errors) at generated `8e76b917…`, including the other lane's Kepler audits in the closed cone; re-run at each later corpus change |
| APEIRON-G1-BUILD-W3-T3 refinement repair after regen | `proof/corpus/Growable/Refinement.lean` (`404d8bf5…`) | `lake build Growable.Refinement` | closed: unfold `table.Table.recycle_unused_staged_path` and `core.option.Option.take` in the two `word_{stored,regen}_apply_pending_and_commit_eq_apply_insert` proofs; statements and premises unchanged; gate exit 0 (3347 jobs, live re-verified) and the deps set (`Growable.ReadLoop Growable.Scan Native.LocatorSpec Growable.Representation`) exit 0; control: a value-changing mutation (`len := s.len + 1`) rejected with the expected-versus-mutated goal, while a no-op mutation proved not to be a semantic control under `hstaged` (recorded), restored at generated `8e76b917…` |
| APEIRON-G1-BUILD-W4-T1 performance acceptance | none | unchanged full REG-1 command | after W3-T2. `micro_spatial` at the W2-T1 source identity (release, i9-14900KF, 1M; identity `/tmp/opencode/w4-micro-identity.txt`): build 625.3 ms, reserve 587.0, kernel 10.25, assemble 15.0, end-to-end 648.2, `digest_ok=true` (after W1: 724.6/688.7/10.5/9.6; pre-fix baseline 1311.1; raw `/tmp/opencode/w4-micro.log`). Full REG-1 re-run executed: exit 1 (strict), wall 316.0 s < 600, 55 controls red, `pass=308 fail=1142 missing=0 unauthorized=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`; the pass count moved 325 -> 308 with unchanged unmatched counts (17 ratio/interval-borderline flips under the faster build, attribution open). G1 stays open; receipt updated |

`src/map/table/state.rs` and `examples/reg1/*` stay outside the source tasks; `src/map/table.rs` is W2-T3's owned file.

## W1-T1 Arc Block

```text
SCOPE:      src/map/table/parallel.rs, tests/parallel/main.rs. Every project-owned container the
            scoped build writes into - each worker's outcome buffer, the joined-results sink,
            the shard schedule and bins - is prepared on the calling thread before the first
            scatter worker; the grouped probe pre-reserves its readback buffers, sink, ordered
            slot vector and output vector likewise. Completion: the injection test refuses with
            zero holds/clones/merges and the seeded lazy-allocation mutation reddens it.
REGISTER:   state/test - the parallel build's reserve-before-workers boundary. Exclusions:
            executor/runtime memory, allocation inside a caller's `each` callback, and the
            unreachable-by-reservation `SpillSite::Grow` arm.
CORPUS:     no canonical-corpus theorem. Files read: src/map/table/parallel.rs,
            src/map/table/prepared.rs, src/map/table/partition.rs, tests/parallel/main.rs.
REFERENCE:  NONE - inventing. Astra boundary verdict ses_f32b8e1f9ffeJcp3CB1SFnBxh7 is an
            architecture anchor, not evidence.
GATE:       flock /tmp/opencode/apeiron-lane.lock timeout 540 \
              cargo test --locked --features injection --test parallel
            Baseline green. Control: in a saved copy, move the outcome-buffer reservation
            (seam included) into the worker closure; the new test must reject on the refusal
            variant and the hold/clone/merge counts; restore and re-run green.
GRADE:      tested - gate green 14/14; clippy and fmt clean; seeded lazy-allocation mutation
            red on the refusal assertion with the pre-mutation copy sha256-verified and green
            when restored.
```

W1-T1 landed `schedule_bins_with_regions` and `reserve_worker_outcomes` in
`src/map/table/parallel.rs` (the build body exceeded clippy's 100-line threshold once the pre-scatter
work was added) and the new test `injected_worker_buffer_reservation_has_zero_cross_worker_effects`.
The same change set also cleaned `src/map/table/locator.rs` (`f12d0413…`): with the borrowed
`QueryIdentity` view the reservation's `Reserved` is `()`, so the release binding became the unit
pattern `((), staged)`, removing the `let_unit_value` and `ignored_unit_patterns` findings. That
edit changes an extracted body, so `proof/regen.sh`, `proof/regen.sh --check` and the corpus bridge
ran at this checkpoint: regen exit 0 (1708 jobs), `--check` exit 0 (fresh), corpus bridge exit 0
(3493 jobs). The generated `proof/ApeironWord.lean` is `d0a34ee5…`, superseding `491b4970…`; the
`((), staged)` unit pattern translates.

Findings folded into the implementation from the verdict:
`ScatterScratch::scatter` and `ScatterScratch::assemble` spawn their own scoped workers
(`src/map/table/partition.rs:426`, `:483`), so the boundary is *before the first scatter worker*, not
before the fill workers. The schedule, bins and outcome buffers are therefore derived from
`ScatterPlan::loads` ahead of scatter, and `debug_assert_eq!(parts.loads(), loads)` guards that the
assembled partition carries the planned loads. The outcome-buffer reservation has its own one-shot
injection seam (`arm_output`/`consume_output`) so a refusal starts no worker at all.

## W2-T1 seam: found-or-linked reservation

Current cost per distinct non-elliptic key in `TableLocator::preassemble_directory`
(`src/map/table/locator.rs:763`): one `linked_handle_query` directory walk (repeat detection), and on a
miss one `Directory::reserve_for` staging walk plus `publish_for_query`; `link`
(`src/map/coord/locator_core.rs:1263`) writes from the staged path arrays, so publication is walk-free.
The removable walk is the separate probe.

Traversal fact the seam uses: `stage_reserved` (`src/map/coord/locator_core.rs:1109`) walks the residue
levels and the identity bytes reusing existing children. A repeat key reuses every child and ends at
the existing terminal node, whose published handle is its `IDENTITY_TERMINAL` word (`link` writes it
at `:1297`); a genuinely new key creates at least one residue or identity node. "Terminal reused" is
therefore decidable inside the walk, and the terminal word read afterward is the existing handle or
`NO_CHILD`.

Seam contract:

1. `Directory::reserve_or_linked<I, K>(index, key)` returns a found-or-staged outcome. Found arm: no
   node, edge or retained identity is created; the idle scratch returns to its slot; the arena
   capacity reservations are capacity-only; the returned handle is the existing terminal word.
   Staged arm: exactly the current `reserve_for` behavior, refusal semantics included (arena lengths
   and counts rolled back, scratch recycled, nothing published).
2. The found arm calls no `reserve_retain`, `link` or `record_link`; the caller records the existing
   handle only.
3. `TableLocator::reserve_or_linked` wraps it with the checks `reserve_path` performs for the staged
   arm (elliptic face, link-registry capacity, profile initialization) and returns the existing
   handle or the staged reservation; `preassemble_directory` calls it instead of
   `linked_handle_query` + `reserve_path`, keeping the elliptic branch before the call as today.
4. `Directory::reserve_for`, `TableLocator::reserve_path` and their write/commit callers
   (`src/map/table.rs:838`, `:1280`, `:1311`) keep the staged-only contract: an already linked key still
   stages a path whose terminal is the existing node and republishes as today. W2-T1 changes no
   write-path behavior.
5. Extraction: `src/map/coord/locator_core.rs` is extracted. If the outcome enum does not translate, a
   tuple or boolean encoding is acceptable, but `proof/regen.sh` and `proof/regen.sh --check` must
   stay green and call the real function; keep single-exit loops and no second key copy.

Implementation gate and control: `cargo test --locked --features injection --lib --test
native_locator --test parallel`; a repeat-key fixture asserts the second occurrence returns the
first handle and adds no link, node or edge (counts before and after); a disposable-copy defect in
which the found arm never fires (always stage) must reject on that assertion, then restore
sha-verified green; then `proof/regen.sh`, `proof/regen.sh --check` and `cd proof/corpus && lake
build`.

Implemented by W2-T1: `Directory::reserve_or_linked` and `TableLocator::reserve_or_linked` landed,
`TableLocator::linked_handle_query` removed, `Directory::reserve_for` and the write/commit callers
unchanged; the extraction checkpoint runs at the integrator's next quiet point.

## Budget and residuals

Astra set 180 s total per Rust task campaign (baseline, mutation rejection, restoration). On this
host the first gate attempt hit the 180 s bound inside compilation with the machine at load
average 338 (another workstream's bevy link); the source gate is re-run under a declared 540 s
budget, inside the §10 600 s ceiling. The deviation and the observed elapsed time are residuals for
the lead. `micro_spatial` timings are machine-bound and this epic claims no ratio until the
unchanged full REG-1 campaign is re-run and read.
