# Apeiron execution plan

This file owns Apeiron's execution order: the ladder, its tasks, their gates and controls, and the
board rows that track them. The programme thesis is in [direction](direction.md); layer ownership and
the trust boundary are in [kernel-boundary](kernel-boundary.md). Where this file disagrees with the
source or a gate, the source and the gate win.

The previous plan (status log, REG-1 tallies, W1–W7 storage record, E1–E6 phasing, resume queue) is
at `01aff6a:docs/plan.md` (`git show 01aff6a:docs/plan.md`). Its receipts stay where they are under
`docs/evidence/`; nothing here re-grades them.

## Outcome

Two releases by **2026-10-31** come from this repository: the hash table below, and the
[inference track](#inference-track-parallel-release-by-2026-10-31).

`apeiron` replaces `std::HashMap` in practice and in claims, published Apache-2.0 at
`github.com/sanguinehost/apeiron` by **2026-10-31**:

1. **Verified on the executed body.** The body the map and count paths run is extracted from the
   real source by Aeneas and refines `Apeiron.Table.{HMap,get,insertWith,build}` under the closed cone
   `{propext, Classical.choice, Quot.sound}`, with a firing mutation control.
2. **HashDOS-bounded.** A stated adversary model and a proved probe-length bound on that body.
3. **Faster than `hashbrown`+FxHasher and `hashbrown`+XXH3** on every workload column: ratio
   `apeiron / incumbent <= 1.0`, mean of ≥5 paired runs with spread; `<= 0.5` is aspirational.
4. **Shippable.** The README separates what is proved, tested and open at the release identity; CI
   runs the Rust gates; a fresh clone reproduces the extraction check.

## Decisions in force

| decision | source |
|---|---|
| Speed comparators are `hashbrown`+FxHasher and `hashbrown`+XXH3, map face, every column. The REG-1 all-incumbent matrix (2592 registered pairs, `examples/reg1`) is not the bar. | owner, 2026-10-06 |
| `fold/hashtrinity` is deprecated. Its table is rebased onto the `R[x]/(x²−σ)` basis and verified through Aeneas. Vendored `src/hashtrinity` modules no consumer uses are deleted. | owner, 2026-10-06 |
| The generic growable `Table` (`src/table.rs`, `src/table/`) and the `proof/corpus/Growable/*` modules only it needs retire once L1 lands. No new work targets them. | owner, 2026-10-06 |
| Order: verify the executed body (L1), then HashDOS (L2), then speed (L3). This reverses the "HashDOS deferred until speed" order of `d621ee5`. | owner, handover |
| Subsume the measured winner, then formalize it (`AGENTS.md` §3a). The executed body is the vendored hashtrinity table, so it is the refinement target. | owner, 2026-10-04 |
| Webfold: Lucas and the lead do not touch it until `REBASE-ONTO-APEIRON` (webfold onto this crate in place of hashtrinity). It is not frozen on its other track. | owner, 2026-10-06 |
| The inference stack (SPIR-V/Vulkan kernels, the llama.cpp replacement) ships by 2026-10-31 too, in parallel with this ladder, not behind it; the old parking rule (Intent and Song behind G1) does not apply to it or to intent. | owner, handover and 2026-10-06 |
| The inference stack ships inside the `apeiron` repository for now; it is expected to move to its own Nous repository later. | owner, 2026-10-06 |
| Contract verdicts, drafted by Sol and adopted by the lead (grade `anchor`, `/home/socol/oc-tmp/escalation/hashdos-mechanism-verdict.md`): D1 restrict the refinement to merges with `f v d = v`, canonical `insertWith` unchanged; D2 refine the sequential `FreqLine` helpers, fan-out a named premise; D3 capped primary probing plus a deterministic AVL overflow, no secret seed. The lead ratifies the key-ordering boundary: the HashDOS claim covers keys with a total order agreeing with equality. | lead, 2026-10-06 |
| Kani is retired ([contract](epics/kani-retirement.md)). | owner |

## Measured state at `01aff6a`

| fact | command | grade |
|---|---|---|
| Corpus green at the apex: exit 0, 9089 jobs, zero `sorryAx`; every audited declaration, including `word_mul_denote` and `word_norm_denote`, in the closed cone | `lake build AxiomAudit` in `proof/corpus/` (bounded, below) | proved |
| Rust gates green: 714 tests passed, 0 failed, 26 binaries; clippy and fmt exit 0 | `cargo test --workspace --all-features --locked` | tested |
| Extraction fresh; it prints `lake build: SKIPPED` and does not build Lean | proof/regen.sh with `--check` | tested |
| The executed map body is not extracted: the four `word_hashtrinity_get`-family wrappers exist in `proof/extract/lib.rs` but `proof/ApeironWord.lean` has zero `hashtrinity` matches | rg over `proof/ApeironWord.lean` | observed |
| `proof/corpus/Map/Selected.lean` states the executed face's contract over an abstract `SelectedFace`; the read refinement `Apeiron.MapFace.selectedReadRep_of_mapRep` is conditional on the named premise `hbridge` | `lake build AxiomAudit` | proved, conditional |
| Executed vacant insert stores `new` without calling `f`; canonical `insertWith` reads `f new default` | `Apeiron.MapFace.selected_insert_vacant_differs_from_canonical` | proved |
| Count arm runs `FreqMap::count_par` over a thread pool, raw send pointers and length writes, none of which the Aeneas fragment models | `src/hashtrinity/par.rs` | model |
| Parity with hashtrinity was reached (`e83bed4` count 0.94/0.99; `b519047` map 1.7736 → 0.9757); the harness that measured it was removed in `c9a2d0c` | `docs/evidence/map-parity-landed.md` | measured, historical; not re-runnable |
| No benchmark exists on the tree: `benches/` is empty and `examples/` holds only engine probes | ls | observed |
| Aeneas reports `505b6ca3-dirty`; the patched backend is not vendored, so a fresh clone cannot reproduce extraction | `proof/lakefile.lean` | model |
| `src/hashtrinity` vendors the whole crate: 29 entries including rustls, webauthn, sync (5596 lines), proc-masq, tregex and openapi; four files exceed 1500 lines | ls and wc | observed |

## Consumers

| consumer | uses | consequence |
|---|---|---|
| `../penglai` | `apeiron::table::{Table, WordHome, ReplacementMerge, Merge, WordMerge, StorageError}`, `apeiron::merge::Face`, `carrier`, `kepler`, `cell`, `coord`, `scalar`, `exchange` | the rebased body takes over the `apeiron::table` API before the generic `Table` is removed |
| `../webfold` | its own `Ptyktos/hashtrinity@2c36b995` (`features = ["codec"]`): `Table` (108 uses), `CoordMap`, `Coord`, `tregex`, `codec`, `crypto` | `REBASE-ONTO-APEIRON` needs those names from this crate |
| `../intent`, `../song` | no import of apeiron or hashtrinity | none |
| in-crate | `src/freq.rs` re-exports the vendored table; `proof/extract/lib.rs:1213` includes `src/hashtrinity/mod.rs` by `#[path]` | the purge keeps what these reach |

## Ladder

A rung opens when the previous rung's exit gate is green on the live tree. Tasks within a rung that
share a file serialize in the order listed.

| rung | exit gate |
|---|---|
| L0 · reconcile | `python3 tools/board/board.py check --live --plan docs/plan.md --project apeiron`: no drift findings |
| L1 · executed body verified | `proof/regen.sh --check` with the `word_hashtrinity_*` roots present; `lake build AxiomAudit` with unconditional read/insert/build refinements of the executed body and the count face's disposition; each task's mutation control red then green |
| L2 · HashDOS bound | a proved probe-length bound on the extracted body under the stated adversary, in `AxiomAudit`; the ported red-team tests pass |
| L3 · speed | `cargo bench --bench map --locked -- --check` exits 0: every column `<= 1.0` against both incumbents, ≥5 paired runs, spread printed |
| R · release | README, CI, licence and fresh-clone reproduction green; the push is confirmed by the owner |

### Tasks

The board mirror: every open row below names its board row; `board.py check` reconciles the two.

| task | board | scope | files | gate | control | state |
|---|---|---|---|---|---|---|
| `APEIRON-PLAN-RESET` | `apeiron-plan-reset-GcnP2JnZnUecOAzUIbBqrJ` | L0. This plan; the board reconciled to it; stale routing and target lines in `AGENTS.md` and `docs/direction.md` corrected. | `docs/plan.md`, `AGENTS.md`, `docs/direction.md` | `python3 tools/board/board.py check --live --plan docs/plan.md --project apeiron` | a plan row with no board row reddens `drift-plan-row-not-on-board` | open |
| `APEIRON-DOCS-CONSOLIDATE` | `apeiron-docs-consolidate-C4VXIpMyNUFzOr7paAjM1k` | L0. Remove the earlier-order and REG-1 sections of `docs/direction.md` (`## Required R0 acceptance`, `## Revised representation boundary`, the `## Execution order` subsections) and the dispatch tables of `docs/roadmap.md`; repoint every inbound link (`rg -n 'direction\.md#\|plan\.md#' docs`) to this plan or to `01aff6a`. `docs/contracts/` and `docs/evidence/` are not edited. | `docs/direction.md`, `docs/roadmap.md`, `docs/kernel-boundary.md` | `python3 tools/check-claims.py && python3 tools/rooting/doc_reach.py --check` | leave one link to a removed anchor in a copy: `check-claims` fails by name | open |
| `APEIRON-BOARD-GATEHEAD` | `apeiron-board-gatehead-GRf8pkvLC4wbhMkFuiTGaf` | L0. `test` is a gate head, so `test -s LICENSE` reads as a gate. | `tools/board/board_core.py`, `tools/board/test_board.py` | `python3 -m unittest discover -s tools/board -p 'test_*.py'` | drop `"test"` from `GATE_HEADS` in a copy: `test_shell_test_is_a_gate` fails | landed |
| `APEIRON-VENDOR-PURGE` | `apeiron-vendor-purge-B7P4Q0hmPKDkYftLsbxHIm` | L1. Delete every `src/hashtrinity` module that the table core, `src/freq.rs`, the extraction and the webfold rebase names (`Table`, `CoordMap`, `Coord`, `tregex`, `codec`, `crypto`) do not reach. Each removal names its reachability check. | `src/hashtrinity/mod.rs`, `src/hashtrinity/` (removals), `Cargo.toml` | `cargo test --workspace --all-features --locked && AENEAS_HOME=/home/socol/oc-tmp/aeneas-build2/install proof/regen.sh --check` | keep a `pub mod` line for a removed file in a copy: the build fails by name | landed: `3d389de` |
| `APEIRON-HT-IDENTITY` | `apeiron-ht-identity-BvOqNvKHHfxZS7sswjc6j6` | L1. Exact logical-key identity in the executed body (contract verdict D3): occupancy out of band, so no sentinel remap (`coord_of` maps `EMPTY`, `TOMB` and `TOMB - 1` onto one value); full-key confirmation on stored-key hits and updates, because `insert_with` merges any two keys with equal coordinates (`src/hashtrinity/mod.rs:1113-1128`). `Sigma` is part of identity. | `src/hashtrinity/mod.rs`, `tests/table.rs` | `cargo test --locked --test table` | restore the remap, or drop the full-key check, in a copy: the sentinel-aliasing and coordinate-collision fixtures fail | landed: `a090aaa` |
| `APEIRON-AENEAS-MODELS` | `apeiron-aeneas-models-BI50FpQ87zv78YqXuReDz1` | L1, critical path for both faces. The executed bodies reach externals the pinned backend does not model; count_aggregate alone rejects 19 (SSE2 `_mm_cmpeq_epi8`/`_mm_set1_epi8`/`_mm_loadu_si128`/`_mm_movemask_epi8`/`__m128i`, `MaybeUninit` ops, `needs_drop`, `U16.trailing_zeros`, `Usize.saturating_mul`/`next_power_of_two`, raw-pointer `cast`/`add`, `Slice.as_ptr`/`get_unchecked_mut`, `Vec.into_boxed_slice`). Add exact-semantics Lean definitions for them to the patched backend (no `axiom`); anything that cannot be defined becomes a named, audited boundary item stated in `docs/kernel-boundary.md`. | `proof/patches/aeneas-lean-model.patch`, `proof/model-patches.sh`, `docs/kernel-boundary.md` | `proof/regen.sh --check` | remove one model from the patch in a copy: the audit names that external again | landed: `fb28a28` |
| `APEIRON-HT-EXTRACT` | `apeiron-ht-extract-DjjuLApKg3g7zaM9kRUkUE` | L1, hardest. Add the `word_hashtrinity_*` roots to the extraction selection and restructure the executed `Table::{new,with_capacity,get,insert_with}` into the Aeneas fragment (loops carry values, no `return` inside a loop; `MaybeUninit`, `Box<[u64]>`). | `proof/regen.sh`, `proof/extract/lib.rs`, `src/hashtrinity/mod.rs` | `AENEAS_HOME=/home/socol/oc-tmp/aeneas-build2/install proof/regen.sh --check && rg -q word_hashtrinity_get proof/ApeironWord.lean` | mutate the home multiplier in a copy: `--check` reports stale | landed `3bf2907`; `prefix_masks` AVX2≡scalar `c6a2be2` |
| `APEIRON-HT-SOUND` | `apeiron-ht-sound-HuFJWskE6qumuLwAlSJ0xi` | L1, before any further speed lever on the map face. Independent review (reproduced): a caught panic in a user merge double-drops a side-store value (F1), leaves an `EMPTY` hole in a probe chain (F2) or a poisoned overflow entry (F3); remove/insert churn at size T−1 rebuilds the table on every insert (F4); `IntoIter` leaks the side store (F5). A panicking merge removes the key; same-size rehash only when `len+1 ≤ cap/2`. | `src/hashtrinity/{mod,face,side,overflow,algebra}.rs`, `tests/ht_panic.rs` (lead exception: 5 sources, one defect class) | `cargo test --locked --test ht_panic --test hashdos --test ht_identity` | revert each fix in a copy: its `ht_panic` test fails | open; F1–F5 landed 40248fe; round 2: three unwind-order defects from the independent review (SideStore::clear, remove/remove_at, insert_fetch_at absorb) |
| `APEIRON-HT-READ` | `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` | L1. Instantiate `SelectedFace.read` with the extracted `get`; prove `SelectedReadBridge` from a carrier invariant the constructor establishes and `insert_with` preserves, so `selected_get_refines_canonical_get` holds on every reachable state with no `hbridge` premise. | `proof/corpus/Map/Selected.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build Map.Selected AxiomAudit` | flip the `EMPTY` branch of the generated `get` in a copy: the headline fails | open |
| `APEIRON-HT-INSERT` | `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` | L1. `insert_with` and `build` refine `Apeiron.Table.insertWith`/`build` for merges with `∀ v, f v d = v` (verdict D1; canonical unchanged, the vacant no-call branch kept); whole-table preservation, not only the written key; constructor, success and refusal statements. Serialized after the read refinement row above (same file). | `proof/corpus/Map/Selected.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build Map.Selected AxiomAudit` | reverse new and old on a live update with the replacement merge in a generated copy: the headline fails | open |
| P0 · COUNT-BODY: get the executed count body into a verifiable form | `p0-count-body-get-the-a-EF9n3OLyqnX2ouQQWL8w9C` | L1. The count face runs `FreqLine` aggregation (`src/hashtrinity/freq.rs:21-137`), not the `Table` body. Factor production-called sequential helpers (histogram, checked offsets, scatter, per-bucket aggregation) and extract them; the thread fan-out stays outside Aeneas as the named premise `FanoutSimulates` (verdict D2). Checked admission for `i64` totals, `u32` histogram entries and `usize` offsets. | `src/hashtrinity/freq.rs`, `src/hashtrinity/par.rs`, `proof/extract/lib.rs` | `cargo test --workspace --all-features --locked` | omit one scatter write, corrupt one offset, or drop one shard publication in a copy: the count tests fail | open |
| P0 · TOOL-IDENTITY: pin Aeneas/Charon to a clean reproducible revision and vendor the backend | `p0-tool-identity-pin-to-a-Gtnw7HF7pOv26R2cBoI3wu` | L1 and R. A clean Aeneas/Charon revision with archive sha256, a scripted build of the patched backend, and `proof/regen.sh --check` green from a fresh clone. | `proof/regen.sh`, `proof/lakefile.lean`, `proof/README.md` | `proof/regen.sh --check` in a fresh clone | point the pin at a revision with a different archive sha256: the build refuses | open |
| `APEIRON-RETIRE-GENERIC` | `apeiron-retire-generic-ChcL1AcRB1c9JwAxUntmKA` | L1 exit. After the read and insert refinement rows above: the rebased body takes the `apeiron::table` API penglai imports; the generic `Table`, its extraction roots and the `Growable/*` modules only it reaches are removed, each with its reachability check. The W7 ledger premises in `kernel-boundary.md` that name only the removed body go with it. | `src/table.rs`, `proof/corpus/lakefile.lean`, `docs/kernel-boundary.md` | `python3 tools/rooting/retire_check.py --check && cargo test --workspace --all-features --locked` | leave one removed module on the `lakefile` root list in a copy: the build fails by name | open; prefix engine ported off the generic Table (PREFIX-PORT, apeiron-retire-generic-ChcL1AcRB1c9JwAxUntmKA) |
| `APEIRON-HASHDOS-SPIKE` | `apeiron-hashdos-spike-FVZlqG9DtCd2pOfookIOTj` | L2. Which mechanism bounds probe length under a key-choosing adversary on this body (keyed home, bounded fallback, or a radix descent with a proved bound), given hashtrinity's own `descent_can_be_wholly_blocked`. Ends in a verdict, not code. | `/home/socol/oc-tmp/escalation/` | `test -s /home/socol/oc-tmp/escalation/hashdos-mechanism-verdict.md` (names the mechanism, the adversary model and the theorem statement) | not applicable: a spike, no soundness claim | landed: verdict `/home/socol/oc-tmp/escalation/hashdos-mechanism-verdict.md` |
| `APEIRON-HASHDOS-CAP` | `apeiron-hashdos-cap-B8tcM9Je2duKmuNmBjvHzM` | L2. Implement the verdict-D3 mechanism in the executed body: stop primary probing after `L` slot examinations, route unresolved operations to one lazy table-wide AVL overflow keyed by the full logical key (with `Sigma`), consult it before reporting absence or inserting, keep uniqueness across both, cap rebuild placement, checked overflow allocation. Port hashtrinity's red-team set (`../fold/hashtrinity/tests/hashdos.rs`). The guarantee covers keys with a total order agreeing with equality (`u64` exactly); other keys get no HashDOS claim. | `src/hashtrinity/mod.rs`, `src/hashtrinity/overflow.rs`, `tests/hashdos.rs` | `cargo test --locked --test hashdos` | remove the cap in a copy: the adaptive-flood test exceeds its probe bound | landed: `cc696cb` |
| `APEIRON-HASHDOS-CAP-COUNT` | `apeiron-hashdos-cap-count-MUBqkA7OYTvluKSIRdbjC` | L2. The count face (`FreqLine` in `src/hashtrinity/freq.rs`, behind `FreqMap::add` and `count_par`) has no probe cap: the map face's cap and AVL overflow (cc696cb) do not cover it. Apply the same capped probe + deterministic overflow to `FreqLine`, with red-team tests for the count face. Serialized after the count-face refinement work on `freq.rs`. | `src/hashtrinity/freq.rs`, `src/hashtrinity/overflow.rs`, `tests/hashdos.rs` | `cargo test --locked --test hashdos` | remove the cap in a copy: the count-face flood test exceeds its bound | open |
| P1 · HASHDOS-FORMALISE: formalise the invulnerability (carried over) | `p1-hashdos-formalise-the-CuQBvGaIws0dH6rzQYsl7j` | L2. Prove over the extracted body (verdict D3): every lookup visits at most `L` primary slots; the overflow AVL height is at most `2·log2(size+1)+1`; reads are exact; growth reinserts through the capped placement; total work for `q` adaptive operations is `O(q·(L + log N))` plus initial capacity. Adversary: fully adaptive, knows every constant; no randomness assumed. Starting point: `proof/corpus/Security/HashDos.lean`. | `proof/corpus/Security/HashDos.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build AxiomAudit` | remove the cap, disable AVL balancing, or omit the overflow lookup in a generated copy: the matching bound fails | open |
| `APEIRON-BENCH-RESTORE` | `apeiron-bench-restore-EaEUYoDjUS4nFW9F4W7iUP` | L3. `benches/map.rs`: the apeiron map against `hashbrown`+FxHasher and `hashbrown`+XXH3 on present-hit, absent-miss, insert, build and remove, occupancy bands and key distributions (distinct, low-byte, random). Interleaved paired runs (≥5), mean and spread per column, a header binding source sha256, `rustc -V` and the CPU. Recover pieces from `c9a2d0c^:examples/reg1`. | `benches/map.rs`, `Cargo.toml`, `Cargo.lock` | `cargo bench --bench map --locked -- --check` | add a fixed spin to the apeiron arm in a copy: its columns fail, restore green | landed: `8c7a0ca`, baseline `docs/evidence/bench-map-baseline.md` |
| `APEIRON-BEAT` | `apeiron-beat-Bu3jZusxHvuMEJ8FOQ3C9O` | L3. One lane per losing column, in the lead's ranking, each closing on the bench and its own control. Rows created from the first bench output. | per lane | `cargo bench --bench map --locked -- --check` | per lane | open |
| `REL-APEIRON-README` | `rel-apeiron-readme-EyLt2dDFoyBV9XLo0UBz8Z` | R. README states what is proved, tested and open at the release identity, with commands. | `README.md` | `python3 tools/check-claims.py` | a claim citing a missing path fails `check-claims` | open |
| `REL-APEIRON-CI` | `rel-apeiron-ci-GBMtEGprgTYdtW1ms9fzoP` | R. CI on GitHub runs the three `cargo` gates and `tools/check-claims.py`. | `.github/workflows/ci.yml` | `gh run list --repo sanguinehost/apeiron --branch main --status success --limit 1` names the release commit | a clippy warning in a branch reddens the run | open |
| `REL-APEIRON-PUBLISH` | `rel-apeiron-publish-sJWFSORzQbMoqOA9rKqCj` | R. Push to `github.com/sanguinehost/apeiron`. The owner confirms the push. | none | `git ls-remote git@github.com:sanguinehost/apeiron.git` names the release commit | not applicable | open |

### Off the ladder (open, not dispatched while a rung is red)

| task | board | scope | gate | state |
|---|---|---|---|---|
| P2 · COST-MODEL: cost bounds over the extracted bodies (waits on P0/P1) | `p2-cost-model-cost-bounds-HKikFfeUVGz3cF6yRnLCNx` | Cost bounds behind the O(1) claim, re-targeted to the extracted executed body after L1. | `cd proof/corpus && lake build` | open, after L1 |
| P2 · COST-HARDWARE: lift the machine constants and add a hardware model | `p2-cost-hardware-lift-the-GfBM4ICC3HDbzgxnIPw4tG` | Machine constants as explicit inputs to the cost model. | `cd proof/corpus && lake build` | open, after `P2 · COST-MODEL` |
| APEIRON-BLOG: collect inputs for the hash-function formal-verification post | `apeiron-blog-collect-for-CaUoYZ7QfIpTPxN2CaMdqh` | Inputs for the hash-function verification post. | every prior-art claim retrieved | open, after L2 |
| [M2][apeiron] Rooting gate green | `m2-apeiron-rooting-gate-FYyCFN1P8Kgu2Dg381dxCZ` | `tools/rooting/check.py --check` reports 37 violations. | `python3 tools/rooting/check.py --check` | open |
| APEIRON-SPLIT-GATE: tools/gate under the 1500-line bound | `apeiron-split-gate-tools-BX6YWj0AN71fddeI5FQlMN` | `tools/gate/main.rs` under 1500 lines. | `cargo build --manifest-path tools/gate/Cargo.toml` | open |
| APEIRON-PORT-ROOTING: port tools/rooting/*.py into the Rust gate driver | `apeiron-port-rooting-port-CdYBgA9GbrCraNjcqLhOFz` | `tools/rooting/*.py` into the Rust gate driver. | `cargo test --manifest-path tools/gate/Cargo.toml` | open |
| APEIRON-HARNESS-LANE-FETCH-INSTALL: land the bounded quorum fetch on main | `apeiron-harness-lane-land-CnFUhFHaC4TnQIA1dG3yRF` | The bounded quorum fetch in `tools/hooks/`. | `python3 tools/hooks/lane.py self-test` | open |

### Inference track (parallel; release by 2026-10-31)

Outcome: a practical llama.cpp replacement on Vulkan/SPIR-V — load a GGUF, prefill, decode, sample,
serve. Acceptance bar (owner, 2026-09-30): ≥2.0× llama.cpp on the same model, prefill and decode
each, on SPIR-V, against llama.cpp's Vulkan backend; CUDA is out of scope.

State, from the receipts:

| fact | source | grade |
|---|---|---|
| The "within ~10%" figures are fold's Vulkan decode, not this tree: fold vs llama.cpp Vulkan tg is 0.94× (27B IQ2_M), 0.98× (8B), 1.16× (0.5B), 2.0× (Bonsai 1.7B ternary) | `docs/epics/inference-parity.md`, from fold's `gpu_decode_bench` | fold receipt; not reproducible from this tree |
| This tree has no end-to-end GPU pipeline: "Apeiron GPU resident prefill — none exists"; llama.cpp CUDA pp200 on Qwen3.5-0.8B Q8_0 is 22947 tok/s, this tree's CPU path 1291 tok/s | `python3 tools/gpu/llama_cmp_gate.py`, receipt `docs/evidence/gpu-prefill/matched-cuda-baseline.md` | measured |
| The isolated Q8 prefill shader `src/spirv/q8_tile.comp` reaches 171.63–506.36 GMAC/s against llama.cpp CUDA MMQ 6387.52–52224.80 GMAC/s on ten shapes | `python3 tools/gpu/q8_shapes.py`, receipt `docs/evidence/gpu-prefill/primitive-latency.md` | measured |
| `src/gpu/vulkan.rs` is byte-identical to fold's `modelfold/src/gpu/vulkan.rs`; the decode GEMV emitters are ported into `src/simt/` | cmp | observed |
| fold's serving path that produced its decode receipts, `bedrock/src/serve/gpu.rs` (5990 lines, fold `d497161f`, sha256 `1e01838dca24f775…`) driven by `examples/gpu_decode_bench.rs`, is not ported | `../fold/bedrock/Cargo.toml` | observed |

| task | board | scope | files | gate | control | state |
|---|---|---|---|---|---|---|
| `INF-DECODE-SUBSUME` | `inf-decode-subsume-C9XByuwivyxA4RMxX5KVlG` | I1, hardest. Port fold's GPU decode serving path (`bedrock/src/serve/gpu.rs`, `bedrock/src/serve/model/pack.rs`, `examples/gpu_decode_bench.rs` at fold `d497161f`) onto the ported `src/gpu/` and `src/simt/`, split under 1500 lines per file, one wave per ≤3-file slice. Reproduce fold's Qwen2.5-0.5B decode with zero device-vs-host argmax mismatches. | `src/engine/gpu/`, `examples/gpu_decode.rs`, `Cargo.toml` | `cargo run --release --locked --features gpu --example gpu_decode` | perturb one dequant scale in a copy: the argmax-mismatch check fails | open |
| `INF-DECODE-MEASURE` | `inf-decode-measure-E3CPdJamkoVAQSrj96zC6o` | I1. This tree's whole-model decode against llama.cpp Vulkan and CUDA on fold's four models (Qwen2.5-0.5B, Qwen3-8B, Qwen3.8-27B IQ2_M, Bonsai-1.7B), ≥5 paired runs, mean and spread. | `tools/gpu/llama_cmp_gate.py`, `docs/evidence/gpu-prefill/` | `python3 tools/gpu/llama_cmp_gate.py --device gpu` | a malformed id file is rejected (the gate's own control) | open, after INF-DECODE-SUBSUME |
| `INF-RESIDENT-PREFILL` | `inf-resident-prefill-GEB633UfORCK7XSmzJifHd` | I2. Resident GPU prefill: KV cache, attention and RoPE on device; the int8 cooperative-matrix shader `src/spirv/int8_coopmat_q8.comp` admitted through `tools/dispatch/admit.py`; dense Q4_K weight reuse. | `src/engine/gpu/`, `src/spirv/`, `examples/gpu_decode.rs` | `python3 tools/gpu/llama_cmp_gate.py --device gpu` | drop the KV write in a copy: next-token agreement with llama.cpp fails | open; coopmat int8 GEMM exact vs DP4A, 2.6–3.5× per GEMM (d23ddac); runtime capability with DP4A floor (COOPMAT-CAP); not yet wired into prefill |
| `INF-API-SURFACE` | `inf-api-surface-G703Ar71jVcLS2spLzZRn` | I3. The serving surface: GGUF load, prefill, decode, sampling, a CLI and an HTTP endpoint, starting from fold's `bedrock/src/serve/{generate,gguf}.rs` and `bedrock/src/bin/serve.rs`. | `src/engine/serve.rs`, `examples/serve.rs`, `tests/serve.rs` | `cargo test --locked --features gpu --test serve` | return the prompt's last token as the sample in a copy: the end-to-end completion test fails | open |
| `REL-INFERENCE` | `rel-inference-HCfpkY6uwxCrf8R4BqO6OM` | R. README with the measured ratios and what is proved, tested and open; CI; publication inside this repository. | `README.md` | `python3 tools/check-claims.py` | a ratio without a receipt fails `check-claims` | open |
| [M4][apeiron] Whole-model matched llama.cpp comparison | `m4-apeiron-whole-model-BkqukF7CetU7IetP37c0Gc` | The matched prefill comparison (model, quantization, quality, device, threads, workload) once INF-RESIDENT-PREFILL runs ([epic](epics/inference-parity.md)). | `examples/engine_profile.rs` | `cargo run --release --locked --example engine_profile` | the device-pin check rejects a GPU comparator in a CPU row | open |
| [M4][apeiron] GPU q8 incumbent cost attribution and matched DP4A ratio | `m4-apeiron-gpu-q8-cost-CWLLlERorlJoZIsX2oYSyP` | Attribution of the 30.6–295.9× primitive gap, feeding INF-RESIDENT-PREFILL. | `tools/gpu/incumbent_ratio_table.py` | `python3 tools/gpu/incumbent_ratio_table.py` | per the tool's controls | open |
| [M4][apeiron] GPU dot and epilogue source-refinement barrier | `m4-apeiron-gpu-dot-and-FFXDtD6vIROrsj8ixnyXBU` | Verification of the shader epilogue against the CPU dot ([epic](epics/gpu-prefill.md)); runs beside I1–I3, not before them. | `docs/epics/gpu-prefill.md` | `cargo test --workspace --all-features --locked` | per the epic | open |

The Slang source binding, policy-engine rows P6–P9 and the Rust–Song correspondence keep their
requirements in their epics and have no open board row.

## Open decisions

1. **Non-table vendored modules.** `tregex`, `codec` and `crypto` are kept because webfold imports
   them from hashtrinity; whether they belong in this crate at all. Owner.

## Gates

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo test --workspace --all-features --locked
AENEAS_HOME=/home/socol/oc-tmp/aeneas-build2/install proof/regen.sh --check
cd proof/corpus && flock /tmp/opencode/apeiron-lane.lock timeout 570 systemd-run --user --scope -q \
  -p MemoryMax=16G -p MemorySwapMax=0 -- env LEAN_NUM_THREADS=1 lake build AxiomAudit
python3 tools/check-claims.py
python3 tools/rooting/read_gate.py --check
python3 tools/board/board.py check --live --plan docs/plan.md --project apeiron
```

Every Rust and Lean command runs under the memory cap and the §10 budget in `AGENTS.md` §6.

## Stop points

- A task owns at most three files, one gate and one negative control; a lane returns its artifact,
  gate output and decision, and the lead re-runs the gate on the live tree before committing.
- No speed claim without `cargo bench --bench map` output from the identity-bound bench.
- No refinement claim about the generic `Table` counts toward L1.
- No push to a public remote without the owner's confirmation.
