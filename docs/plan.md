# Apeiron execution plan

> **Correction (repository state).** The REG-1 comparison harness (`examples/reg1`, `examples/smhasher`, `examples/fold`, `examples/support`) and its comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash) were removed from the tree in `c9a2d0c`. Every `cargo run ... --example reg1` (and any other removed-harness example) below is a historical command that cannot run on this tree; the live Rust gates are `cargo test`/`cargo clippy`/`cargo fmt` from `AGENTS.md` §6, and the live performance entry point is whatever the current harness names. Registered-incumbent and acceptance-registry claims below describe the removed harness.

Kani is retired by user direction. The [retirement contract](epics/kani-retirement.md) supersedes
all older Kani gates and dispatch requirements in this handoff and linked plans. Historical
receipts remain historical. Current acceptance uses Rust tests/clippy/fmt, real-source mutation
controls, extraction freshness and the bounded Lean builds/audits; open proof premises remain open.
Current dispatch is [the bounded read/search proof plan](epics/read-proof.md); it supersedes the
older execution tables below without erasing their revision-bound receipts or obligations. The
[five-barrier sequence](#subsumption-execution-sequence-apeiron-subsum) supersedes the immediate
tuning dispatch and the Intent/Song development scheduling below; it retains their requirements,
gates, controls and obligations.

This file owns Apeiron execution detail: order, gates and stop points for Apeiron work. The
programme thesis, ownership and outcomes are in the [programme direction](direction.md); local
representation and trust are in the [local boundary](kernel-boundary.md). Local authority: this
document owns the Apeiron execution sequence and its gates. It coordinates named cross-repo tasks
but does not dispatch an entire repository, and those local gates stay owned there.

## Canonical root and scope

The [single R0 chain](direction.md#thesis) governs: historical hashtrinity/fold, spinor/affine
collapse law, Table ordered fold, SKI/derived-Iota translation and Rust/Song/Slang execution are
layers of one admitted primitive. Local proofs do not establish the missing connecting theorem.

The canonical object is `Apeiron.Table.{HMap,get,insertWith,build}` and
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` with the extracted `Cell<i64>` word
bodies `word_mul_denote` and `word_norm_denote`; the dotted roots are `Apeiron.Table.HMap`,
`Apeiron.Table.get`, `Apeiron.Table.insertWith` and `Apeiron.Table.build`. Every executing face must
refine the canonical `Apeiron.Table.insertWith` / `Apeiron.Table.build` object; anything not rooted
in it is not accepted.

The basis is O(1) at fixed N with bounded callbacks and nonconstant key length; reading a whole
source stays at least linear in its bytes at fixed N. The rooting gate is
`python3 tools/rooting/check.py --self-test`; `python3 tools/rooting/check.py --check` is red on the
binned routes (A=1, B=3, C=1) and on D=38 sites — 16 genuine, 20 false positives with a reduction
in flight — of which the §8 numeral decision (`470e3ab`) surfaced twelve.

## Current status

Extraction and the named word-kernel refinements are done at their stated scopes. `word_mul_denote`
and `word_norm_denote` quantify over every `Cell<I64>` input and dial under the `ZMod (2^64)`
denotation; partial carrier kernels retain their named success and refusal premises. The fixed-line
carrier refinement is closed at its stated scope. The historical fixed-shard `Table` bodies are
extracted, and its generated table refinement covers arbitrary callbacks satisfying `CallbackLawsG`:
the constructor
establishes `PhysicalInvG`; admitted `insert_with` and build preserve it; the concrete word instance preserves
`PhysicalInv` and canonical `insertWith` on success and returns the identical representative on
`ShardFull`, with exact refusal at `N = 0`; for positive `N`, `BuildAdmissibleN` exactly
characterizes build success versus `ShardFull` and `table_build_physicalInv_of_buildAdmissibleN`
gives the physical `build` postcondition; and `wordBuildN_zero_ok_iff_empty` characterizes the
zero-capacity build. The release identity is in
[proof/README.md](../proof/README.md); the run narrative is in
[archive/autonomous-run-log.md](archive/autonomous-run-log.md). A source-proof run log is historical evidence, not
fresh acceptance.

The executing representation is the growable flat/spill `Table` in `src/table.rs`: one kernel with
two concrete storage faces (`Stored<K>`, `Regen<C>`), a flat certified-coordinate resident area and
grouped spill regions, keyed by `(Sigma, K)`. The fixed-shard `Table<K, V, H, N>` and its
bounded-admission proofs are revision-bound historical evidence; the growable extraction/refinement
integration wave remains open. The fixed-shard table fails performance-preserving
consolidation on the measured full16/direct-index-resident boundary; it is not accepted as the final
migrated primitive solely because proofs exist. This is not a universal slowdown claim: offset
coordinates give different results. G1/REG-1 remains OPEN: the strict command exits 1 and the applicable-pair tally is unsettled. The
receipt of record is [perf-ledger](evidence/perf-ledger.md) (sha256 `5b94d6cd…` measured on this
tree): its `HEADLINE` line reads `registered=2592 load_bearing=2268 applicable=1450 pass=421
fail=1029 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`, its
earlier strict section reads `applicable=650 pass=133 fail=517 missing=0 unauthorized=0
not_applicable=808 informational=1134`, and the authority records 387 passing / 1063 failing
applicable pairs ([direction](direction.md#current-cut-partition-validation-before-payload-layout-replacement)).
Three tallies disagree; the refresh belongs to the measurement owner (`APEIRON-REANCHOR-W5-T1`), and
the exit status and `full_acceptance=false` stand on every one of them. The
pathological cells of the earlier strict run were the spill absent-miss distributions (worst `apeiron_table`/`hashbrown_fxhash`
`occ16-seeded-random-absent-miss-replacement`, median ratio 373.8538, interval max 375.1818) and the
resident growth-insert cells (worst `apeiron_table`/`hashtrinity_table`
`occ16-direct-index-resident-growth-insert-replacement`, median ratio 231.8779); the resident
direct-index healthy cells are not pathological (present-hit 3.3860x and absent-miss 4.6780x versus
serial `hashtrinity_table`, about 2.7–4.7x across the load-bearing band when the receipt was cut; the current cut reproduces 1.68–2.63x across the same cells), and the current cut measures
those two cells at 23.59x and 1.53x. No speed recovery is
claimed. That receipt binds `examples/reg1/acceptance.rs` sha256 `f0f9aff0…` and `src/table.rs`
sha256 `a89e9ec7…`; the current tree carries the T1 registry plus a lint-only clippy fix
(`examples/reg1/acceptance.rs` sha256 `e74f8dd4…`) and a post-receipt spill/growth fix
(`src/table.rs` sha256 `aee4526a…`; both digests recomputed here with `sha256sum`), so the receipt
is the last strict full-profile result but not a measurement of the current identity. The
re-anchored performance workstream in the barrier section continues: comparison coverage is repaired
(T1) and the combined `get_each` arm is now `concurrency_unmatched`; serial direct-path overhead and
spatial recovery on disjoint paths remain — with `W7-SearchCommitBoundary` frozen, unchanged under
`APEIRON-STRATEGY-DECISION-T2` and `-T3`.

The per-entry relocation step is extracted and proved for both carriers and both destinations:
`word_stored_relocate_entry` and `word_regen_relocate_entry` are rooted in the fresh
`proof/ApeironWord.lean`, and `proof/corpus/Growable/Relocation.lean` proves
`word_stored_relocate_entry_flat_preserves`, `word_regen_relocate_entry_flat_preserves`,
`word_stored_relocate_entry_spill_preserves` and `word_regen_relocate_entry_spill_preserves` under the
closed cone, with `reservedObservation_writeSpill_same`/`reservedObservation_writeFlat_same` carrying
the nested destination `Vec::set` readback, all registered in `proof/corpus/AxiomAudit.lean`; the spill-tag
mutation `tag(address.place) -> 0` is red and the restored source green. The traversal skeleton and
the commit equation are now proved in the same closed cone:
`proof/corpus/Growable/Relocation.lean` proves `word_stored_relocate_all_preserves` and
`word_regen_relocate_all_preserves` (the rooted traversal preserves any initially-established,
per-entry-preserved invariant; no callback argument), and `proof/corpus/Growable/Refinement.lean`
(now a `lakefile.lean` root) proves `word_stored_apply_pending_and_commit_eq_apply_insert` and
`word_regen_apply_pending_and_commit_eq_apply_insert` (the commit root equals the ordinary extracted
`table.Table.apply_insert` in `proof/ApeironWord.lean` on `rebuiltView` under the input-side carrier equation `hhold`), all registered in
`proof/corpus/AxiomAudit.lean` and building under the closed cone (`lake build` 3471 jobs;
`proof/corpus/Growable/Relocation.lean` sha256 `0604166a…`, `proof/corpus/Growable/Refinement.lean` sha256 `3e21b9de…`). The
rejecting control runner
`proof/controls/relocation.sh` (receipt `docs/evidence/relocation-controls.md`) exercises that
mutation: step 2 reddens the four spill headlines and the restored source rebuilds green; the default
baseline and restored freshness steps are green on the landed `proof/ApeironWord.lean`. Run the
control standalone (`bash proof/controls/relocation.sh`): it acquires `/tmp/opencode/apeiron-lane.lock`
itself, so do not wrap it in `flock`. The runner is hardened
(`APEIRON-EXTRACT-REP-RELOCATION-CONTROL-T2`): the whole-repo `rsync` is replaced by a bounded
`cp -a` (`T_COPY=120`s), every step carries a timeout, the lane lock is an inherited file descriptor
rather than a child `flock` process, the default run is the freshness step only, and
`RELOCATION_CONTROL_HEAVY=1` adds the semantic corpus build; the recorded root cause is whole-repo
`rsync` under `btrfs-cleaner` I/O pressure leaving an orphan lock holder. `Table::relocate_all` and
`Table::apply_pending_and_commit` are production-factored and production-called by
`Table::apply_prepared_rebuild` (`relocate_entry` unchanged). Option (b) is implemented in
`src/table.rs` sha256 `2a60a19d…`: `Table::apply_pending_and_commit<F: Merge<V>>` consumes the
precomputed `SearchDecision`, `Table::locate_rebuilt` returns the value-only `SearchDecision`, the
private `enum Site` is removed, and `locate_rebuilt`/`Flat::index` remain in the unextracted
`apply_prepared_rebuild` orchestration. The two `apply_pending_and_commit` extraction roots are
present and rooted in the landed `proof/ApeironWord.lean` sha256
`c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2` (fresh for `src/table.rs`
sha256 `2a60a19d…`, audit clean, `proof/` `lake build` 1708 jobs; all six
`relocate_entry`/`relocate_all`/`apply_pending_and_commit` wrappers). W6-T2 landed:
`proof/corpus/Growable/Rebuild.lean` (sha256 `8db24388…`) proves `word_stored_rebuild_preserves` and
`word_regen_rebuild_preserves` over the traversal, closed cone, `lake build` 3474 jobs,
relocation-write mutation red then green. Still open and frozen: the W7 retry `prepared_rebuild_refines_insertWith`
(needs `SearchDecisionCorrect` + the callback laws, carried by the seven named boundary-ledger
premises `ResourceAdapterSimulates`, `SearchDecisionCorrect`, `DecisionSymbolicCorrespondence`,
`QueryAddressCorrect`, `FreshLengthBound`, `GrowableCallbackLaws` and `SpillVacantDirectGuard`,
[`kernel-boundary.md`](kernel-boundary.md#w7-searchcommitboundary)), suspended under
`W7-SearchCommitBoundary` because the
reachable publication witness is conditional on the explicit unproved `ResourceAdapterSimulates`
premise; the W6-T1 disjoint-write frames are landed, and the freeze and its reopening condition are
unchanged under `APEIRON-STRATEGY-DECISION-T2` and `-T3`. W6-T4 interface repair landed: `proof/corpus/Growable/Representation.lean` (sha256
`4dc7df700d8fc9eb50a88755934f5e632c702f77b7de0992114611a70c3139c1`) adds
`reservedRepR_of_components` and the
`reservedRepR_{liveSet,unique,coverage,directCover,directory,length}` projections, and
`proof/corpus/Growable/Resources.lean` (sha256
`687c0e6d381e18481017f9dac92535bb77cf4010473f9d89743a148bf00a089e`) strengthens
`ReceiptMatchesStorage` with the target lanes, flat extent `256 × plan.lanes`, the 256-entry
storage/receipt directories and per-region tag/payload lengths, and adds `TargetGeometryCertificate`
with the `receiptMatches_*`/`rebuildPlan_target*` projections; the bounded gates are green in the
closed cone (`lake build` 3473 jobs) and the flat-extent `256 × lanes + 1` mutation is red then green.
W6-T5 landed: `RebuildPlanClauses` binds `cloneHeld`/`addressHeld`/`locateHeld` to the generated
carrier clone, `address_held state.home` and `table.spill_locate_held … 16#usize`
(`cloneHeld_eq`/`addressHeld_eq`/`locateHeld_eq`); `RebuildPlanCorrect_R` requires the production
`keyRep` dictionary (`planCertificate.keyRep = ProductionRelocationKeyRep.keyRep`);
`ResourceAdapterSimulates.prepared` carries the same equality; and the
`rebuildPlan_stored_operations_binding`/`rebuildPlan_regen_operations_binding` projections expose the
three bindings for both carriers. `proof/corpus/Growable/Resources.lean` sha256
`31d17cca7bc25707be6d96628ab4730f65e67da8bbdbb741adf368611dacc576`, closed cone
`{propext, Classical.choice, Quot.sound}`, `lake build` 3473 jobs; the `locateHeld := Vacant 1`
mutation is red on the mutated source and green on the restored source. W6-T2 landed:
`proof/corpus/Growable/Rebuild.lean` (sha256
`8db24388133248ef29ab483c740e1b0462e10d826bc31cdbdc1d2c1238c19e9a`) proves
`word_stored_rebuild_preserves`/`word_regen_rebuild_preserves` over
`word_{stored,regen}_relocate_all` from source `RepR`, the publication equality,
`PreparedResources_R` and `RebuildPlanCorrect_R`, with no merge or callback parameter; registered in
`proof/corpus/AxiomAudit.lean` (sha256
`56d73dbe2bebda144f2d7a13a46fbb7b5fb471a6e28a03c46b6084472e69fa85`) with `lakefile.lean` sha256
`bdd14f9fa9fdbd79ce72b02d2c52da320c9950f103847a7ceb1361056899600a`, closed cone, `lake build` 3474
jobs, relocation-write mutation red then green. The W7-T2 premise interface is landed:
`proof/corpus/Growable/Search.lean` (sha256
`d349cd481bdc2a33d68c7b1487b64e72ff8bfc98be89f17d3029e585382c7600`) defines `QueryAddressCorrect`
(`queryAddressCorrect_vacant_firstFit`), `FreshLengthBound` (`freshLengthBound_vacant`),
`GrowableCallbackLaws` (`growableCallbackLaws_identity`, `growableCallbackLaws_generated_once`) and
`wordMerge_newFirst`; `proof/corpus/Growable/Resources.lean` (sha256
`3708d69cbdc85616ac69a35b58e81915844ba91961cbfa99f95416f98a812461`) defines
`DecisionSymbolicCorrespondence` (`decisionSymbolic_{found,vacant,vacant_valid}`) and proves
`resourceAdapter_refusal_stutter`. All are registered in `proof/corpus/AxiomAudit.lean`, build in the
closed cone `{propext, Classical.choice, Quot.sound}` (`lake build` 3474 jobs), and the spill-shard
mutation is red on the mutated source and green on the restored source. Per Sol, `FreshLengthBound`
and the arbitrary-merge semantic identity are input-side; `DecisionSymbolicCorrespondence`,
`QueryAddressCorrect` and `GrowableCallbackLaws.exactlyOnce` require source binding/extraction; and
`ResourceAdapterSimulates.refused` carries no-clone-on-`Err`. Astra approved all four as explicit
conditional trust-boundary premises with discharge paths in
[`kernel-boundary.md`](kernel-boundary.md#pending-composition-premise-boundary); no existing headline
or grade changed. The W7 retry `prepared_rebuild_refines_insertWith` is suspended under
`W7-SearchCommitBoundary` and not asserted; the freeze and its reopening condition are unchanged
under `APEIRON-STRATEGY-DECISION-T2` and `-T3`.
Whole-table
`RepR`/`ReservedRepR`/`PendingCommittedR`
landed in `proof/corpus/Growable/Representation.lean` (W5); payload-initialization placement remains
open. Kani is retired and supplies no current coverage requirement.

### Joint prepared-resource boundary (Astra verdict; Sol contract W0; W1 landed; W2 landed; W3 landed; W4 landed; W5 landed; W6-T1 landed; W6-T4 interface repair landed; W6-T5 landed; W6-T2 landed; W7-T2 premise interface landed; W7-SearchCommitBoundary frozen, unchanged under APEIRON-STRATEGY-DECISION-T2/-T3)

Astra approved the adjusted extraction boundary (`kernel-boundary.md` Joint prepared-resource
boundary): extract the joint resource handoff, not reservation or initialization. `R::hold`,
`Flat::initialize` and `Region::initialize` move out of `Table::apply_prepared_rebuild` into the
unextracted adapter after reserve-all, so prepared storage is materialized and vacant; add a
nonallocating publication helper and a separately rooted scalar capacity comparison; the
decision-consuming commit interface is unchanged. `plan_rebuild`, `reserve_rebuild`, every
`try_reserve*`, capacity reads, initialization, composite search and the executor stay unextracted.
The eight existing proved headlines keep their statements and premises and must be revalidated by
regeneration, bounded audit/build and the existing controls on the new source identity.

Sol reconciled the contract in `docs/epics/storage-refinement.md` (W0): `CapacityBound`,
`ReservationReceipt`, `PreparedOutcome` and `PreparedResult`, with the only admitted order
`checked plan -> reserve all -> observe/check capacities -> hold and initialize -> publish ->
relocate -> pending decision -> merge once/commit`. Whole-table `RepR`/`ReservedRepR`/`PendingCommittedR`
are landed in `proof/corpus/Growable/Representation.lean` (W5, extended by W6-T4; current sha256
`4dc7df70…`; focused gate bounded
`lake build Growable.Representation AxiomAudit` in `proof/corpus/`). The resource
predicates and conditional publication transport are landed in
`proof/corpus/Growable/Resources.lean` (task `APEIRON-EXTRACT-REP-RESOURCE-W3-T1`/`W4-T1`); its
focused gate is bounded `lake build Growable.Resources AxiomAudit`.

Implementation W1 landed: `CapacityBound`, `RegionReservation`, `ReservationReceipt`,
`PreparedOutcome<R>` and `PreparedResult<R, V>` are in `src/table/resources.rs` (sha256
`5b04be96131a360d8245ee3061c58dba85059f39a3efeeb0261442a7051368bf`) with `capacity_bound_holds`
and `publish_prepared`; `observe_receipt`, `materialize_rebuild` and `prepare_rebuild` are in
`src/table.rs` (sha256 `a89e9ec743405ac7d80183e431f6e932ba90353bc777f142865f44005261d09a`);
`apply_prepared_rebuild` consumes `PreparedResult` and returns `Result<(), StorageError>`, returning
the refusal without touching `self`; hold and initialize run in the adapter after reserve-all; the
`apply_pending_and_commit` signature is unchanged. Focused gate `cargo test --locked --lib --test
table` is green (36 lib + 42 table); both mutation controls red then green; `tests/table.rs` is
unchanged (the handoff types are `pub(crate)` and the tests live in the lib unit module). This phase
changes no claim grade. W2 landed: `word_stored_publish_prepared`, `word_regen_publish_prepared` and
`word_capacity_bound_holds` are added to `proof/extract/lib.rs` (sha256 `1e4fa586…`) and
`proof/regen.sh` (sha256 `7df851dc…`); `proof/ApeironWord.lean` is regenerated to sha256
`95766c650fd94da6546d21ef6dff115db22d55cac90b1949f96352217055d1e5` (audit clean, `proof/` `lake
build` 1708 jobs), `proof/regen.sh --check` is fresh on `src/table.rs` sha256 `a89e9ec7…` /
`src/table/resources.rs` sha256 `5b04be96…`, the relocation control default path passes, and the
bounded `proof/corpus` build (3471 jobs) revalidates the eight existing headlines. W3/W4 landed
(`APEIRON-EXTRACT-REP-RESOURCE-W3-T1`/`W4-T1`): `proof/corpus/Growable/Resources.lean` (sha256
`db14c8ae…`) defines `CapacityBound_R`, `EveryScalarCapacityCheck_R`, `PreparedResources_R`,
`RebuildPlanCorrect_R` and `ResourceAdapterSimulates`, and proves `word_capacity_bound_holds_iff`,
`word_stored_publish_prepared_carries`, `word_regen_publish_prepared_carries`,
`preparedResources_of_publication` and the two `word_{stored,regen}_reachable_publication_witness`
headlines, all registered in `proof/corpus/AxiomAudit.lean`. The supplied W3/W4 receipt is closed
cone `{propext, Classical.choice, Quot.sound}`, `lake build` 3472 jobs and a red-then-green mutation
control; this documentation record does not rerun it. The two reachable-publication headlines are
`proved` only as implications conditional on `ResourceAdapterSimulates`, an explicit unproved named
trust-boundary premise (local boundary `Resource adapter simulation premise`); they establish no
actual adapter reachability. Current derivation order (Astra, roadmap): W5 whole-table
`RepR`/`ReservedRepR` landed → W6-T1 disjoint-write frames landed → W6-T3 certificate redesign landed
→ W6-T4 interface repair landed (reserved construction/projections; strengthened receipt and target
geometry) → W6-T5 landed (production `keyRep` dictionary plus clone/address/locate bindings and the
two operations-binding projections; `proof/corpus/Growable/Resources.lean` sha256 `31d17cca…`) → W6-T2
`rebuild_preserves` landed in `proof/corpus/Growable/Rebuild.lean` (`word_{stored,regen}_rebuild_preserves`;
sha256 `8db24388…`) → W7-T2 premise interface landed (Search/Resources interfaces, see above) → W7
retry composition `prepared_rebuild_refines_insertWith` suspended under `W7-SearchCommitBoundary`.
W7 composes `rebuild_preserves`
with the commit equality, `SearchDecisionCorrect` and the callback laws (`GrowableCallbackLaws`) to
canonical `Apeiron.Table.insertWith`; it is not asserted here, and the freeze is unchanged under
`APEIRON-STRATEGY-DECISION-T2` and `-T3`.

### Supplied measurement receipts (REG-1, Kani and historical figures not rerun; the relocation control was rerun, receipt `docs/evidence/relocation-controls.md`)

| Evidence / grade | Result and limit |
|---|---|
| REG-1 runtime, `measured` (receipt of record `docs/evidence/perf-ledger.md`, sha256 `5b94d6cd…` measured on this tree; the applicable-pair tally is unsettled and is not refreshed here) | The receipt's `HEADLINE` line is `registered=2592 load_bearing=2268 applicable=1450 pass=421 fail=1029 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`, exit 1; its earlier strict section is `applicable=650 pass=133 fail=517 missing=0 unauthorized=0 not_applicable=808 informational=1134`, and the authority records 387 passing / 1063 failing applicable pairs. All three exit 1 with `full_acceptance=false`, and the refresh belongs to the measurement owner (`APEIRON-REANCHOR-W5-T1`). Earlier strict-run pathological cells: spill absent-miss 373.85x (interval max 375.18) and resident growth-insert 231.9x, measured at 23.59x and 1.53x by the current cut; the resident direct-index healthy band reproduces at 1.68–2.63x behind the pinned predecessor on the current cut (E6 resident spike, measured; the receipt's `ht_median_ns` strata line names exactly one comparator, `hashtrinity_table` — hashbrown/foldhash are paired in the acceptance run, not in the recorded ratios, and the 2.7–4.7x band is stale on both ends). Sole full16 workload is an extreme boundary, not a representative aggregate. |
| Spill/growth micro, `measured` (diagnostic; not a pair-gate result) | `cargo run --release --locked --example micro_spill` on the post-receipt spill/growth fix, same 4096-op schedules: `occ16-seeded-random-absent-miss-replacement` median `1314468`/`1328764` → `78758` ns (`320.92`/`324.41` → `19.23` ns/op) and `occ16-direct-index-resident-growth-insert-replacement` median `7294013`/`6805921` → `93218` ns (`1780.76`/`1661.60` → `22.76` ns/op); before logs `/tmp/opencode/diag.out`/`diag2.out`, after `/tmp/opencode/micro-final.out`. The two cells remain the strict receipt's worst; no post-fix full-profile receipt exists and the pair-gate grade is unchanged. |
| Cached-placement experiment, reported `measured` | Candidate/hashtrinity 2.65–2.69 on the measured boundary; rejected, benchmark-only, not production or proof-covered. |
| Historical spatial reproduction, archival pending identity registration | `71942db` parallel CountTable: historical 3.16 ms versus packed hashtrinity 10.27 ms; supplied temporary-worktree rerun 2.494 ms versus 8.976 ms (~3.6x win), i9-14900KF / rustc 1.97 nightly. No CountTable refinement is established. |
| Kani verification latency, reported bounded-check result and `measured` elapsed time | `cargo kani`: 448.58 s, 62/62 green, one unreachable; `the_norm_factors_into_the_channels` about 94.23 s, largest. This is verification latency, not table runtime or a Lean proof. |
| Extraction freshness, landed at the recorded revision | `proof/regen.sh --check` fresh on `src/table.rs` sha256 `2a60a19dae103fbd6981824c11ca7b436ef344071cd979905746c9ec531d5856`; `proof/ApeironWord.lean` sha256 `c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2`, `lake build` 1708 jobs, generated audit clean, all six `relocate_entry`/`relocate_all`/`apply_pending_and_commit` roots present for both carriers. The fresh standalone control receipt `docs/evidence/relocation-controls.md` records the same baseline green and restored freshness green (exit 0). The earlier red was source-line comment drift, not semantic drift. |

No production `src/` changes belong to the supplied measurement session. Preserve uncommitted
`examples/reg1/` evidence and unrelated spinor work. The performance entry point recorded here is
`cargo run --release --locked --example reg1 -- --check performance --profile full --controls`
(the `examples/reg1` harness was removed in `c9a2d0c`, so this command no longer runs on the tree);
its exit 0 validates a boundary report, not G1. Offset and historical rerun command/log/digest
receipts and full machine/sample attribution remain **MISSING** from the durable register; bind
them before promoting these supplied figures to acceptance. See the
[lifecycle annex](epics/storage-refinement.md#performance-preserving-consolidation-lifecycle-annex).

## Apeiron storage exit: all rows required

| ID | Required output | Gate (cwd) / remaining work |
|---|---|---|
| S1 | Executing `new`/`get`/`insert_with`/`build` on the real source path | `cargo test --locked --test table` (root); `proof/regen.sh --check` accepts the real bodies. |
| S2 | Constructor and witness from the actual constructor | `word_table_new_rep`; `word_table_new_joint_witness`; `word_table_joint_branch_witness`. |
| S3 | Canonical default read, new-first write and ordered merge | `emptyWordTable_read`, `word_table_get_new_first`, `word_table_get_replaced`, `word_table_build_get_acc`; bridge `lake build` exits 0. |
| S4 | Joint admission precondition and success/refusal invariant preservation for arbitrary `N` | Closed under explicit `CallbackLawsG`: generic admitted insertion and build preserve `PhysicalInvG`; `table_build_zero_ok_iff_empty` gives the exact zero-capacity split; `table_build_physicalInv_of_distinctRoutedKeysPerShardG_all` gives the all-`N` sufficient admission result. |
| S5 | Actual extraction audit | `proof/regen.sh --check`; `proof/corpus/AxiomAudit.lean` `#audit`s every headline inside `{propext, Classical.choice, Quot.sound}`. |
| S6 | Baseline / red / restore control on the real source | Same-filesystem generated-body mutation `merge value default` -> `merge default value` fails `table_insert_empty_zero`; the restored copy builds. |
| S7 | Full Rust and proof integration on one source/tool identity | `cargo test --workspace --all-features --locked`; `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`; `cargo fmt --all -- --check`; `proof/regen.sh --check`; `lake build` in `proof/` and `proof/corpus/`. All are required for epic closure. |
| S8 | Cone, identity, perimeter, size and G1/REG-1 | OPEN; details below. |

Grades at actual scope: S2–S4 are proved at their named theorems; S1 is test evidence; S5 is an
extraction audit; S6 is a mutation control, not a proof. S7 passed the full Rust/Kani/extraction and
Lean barrier on the recorded source/tool identity. S8's partial evidence passed
`tools/check-claims.py` and `tools/check-assurance.py`; the latter's 6/6 rejecting controls enforce
TCB <= 5000 physical lines, owned files <= 1500 lines and <= 12 source files directly per directory.
New headlines are audited inside the standard cone. S8/G1 remains **OPEN** until required REG-1
acceptance is recorded; perimeter checks do not close it. These storage results do not establish
the Rust-Song correspondence for the same primitive.

**Owner definition of done (apeiron core).** A face is done only when all four hold jointly:
formally verified end to end (real-source extraction, refinement to canonical
`Apeiron.Table`/`AnyRing`, closed cone `{propext, Classical.choice, Quot.sound}` with an empty cone
for the Slang chain, mutation controls); competitive-or-SOTA speed; not HashDOS-vulnerable; and
cryptographic strength not mandatory but achievable as a face. Proof without performance,
performance without canonical refinement, or verified speed an adversary can collapse does not
pass. The full statement is in
[direction](direction.md#owner-definition-of-done-for-the-apeiron-core).

**Exact all-row matrix acceptance command (root).** The strict acceptance command is
`cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`.
It exits nonzero when any applicable strata speed cell is `fail`/`rejected`/`missing` or when a
`pass` cell's ratio interval max is above `1.0`; `not-applicable` speed cells and the separate
resource rows are excluded. Without `--accept` the same run is diagnostic, exits 0 and reports the
axis `boundary`. The command currently exits 1: the receipt of record
([receipt](evidence/perf-ledger.md), sha256 `5b94d6cd…` measured on this tree) reports
`registered=2592 load_bearing=2268 applicable=1450 pass=421 fail=1029 missing=0
concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false` on its headline line,
`applicable=650 pass=133 fail=517 missing=0 unauthorized=0` on its earlier strict section, and the
authority records 387 passing / 1063 failing applicable pairs, so the applicable-pair tally is
unsettled and its refresh belongs to the measurement owner. Address-face and concurrency-unmatched
pairs fail regardless of ratio, as does every applicable pair whose interval maximum exceeds `1.0`;
the failures are dominated by the
spill absent-miss distributions (373.85x; 23.59x on the current cut) and the resident growth-insert cells (231.9x; 1.53x on the current cut), while the
resident direct-index healthy cells reproduce at 1.68–2.63x behind the pinned predecessor on the current cut (the 2.7–4.7x band is stale; E6 resident spike, measured);
`reg1_pass=false` and `report_complete=false`. No speed recovery is claimed. The remaining
[RFC rows](direction.md#required-r0-acceptance-g1--reg-1) — identity, round-trip, cross-process
determinism, growth/merge with refusal and coverage — are not all closed by it: `quality ∧ speed ∧
baseline` requires identity, round-trip, cross-process determinism, growth/merge with refusal, speed
and coverage.
The acceptance threshold is the owner decision: for every applicable cell,
`selected/incumbent <= 1.0` against **every** named incumbent, never behind and ideally ahead.
Named incumbents: `hashtrinity::{Table,CoordMap,FreqMap}` serial and packed-parallel
(`count_par`/`get_each`), `hashbrown+foldhash`, `hashbrown+fxhash`, `hashbrown+rayon`, `std`
SipHash, `foldhash`/`fxhash` par16, and — per the landed `docs/hash-comparison.md` and
`examples/reg1/acceptance.rs` — `xxHash` XXH3/XXH64, `wyhash`, `ahash` and `fnv`, each now
load-bearing on the opaque-key exact map face; `blake3` and `sha256` are registered informational
cryptographic comparators and carry no ratio bound, because this harness names no
authentication/integrity contract. Coverage of `hashtrinity` `CoordMap` and the serial `FreqMap` is
**open**: no adapter for either is registered, which is a coverage gap and not a `not-applicable`
finding. Applicability is registered per cell/incumbent pair before
timing with a semantic or API reason; no family-wide exemption; missing, duplicate, invalid or
inconclusive pairs reject. `--accept --profile smoke` is diagnostic only; full acceptance requires
all registered pairs. This supersedes the RFC elapsed-median floor `selected/hashtrinity <= 1.25`;
there is no historical-envelope tolerance. Require the
[matrix](direction.md#required-r0-acceptance-g1--reg-1): spatial parallel CountTable, serial cost,
occupancy 1/2/4/8/16, direct-index/offset/skewed/random coordinates and separated operations, with
hashtrinity, hashbrown/foldhash, hashbrown/fxhash, hashbrown/rayon and std baselines where
applicable. The landed `docs/hash-comparison.md` carries the comparison axes (crypto versus
non-crypto, throughput small/large, latency, collisions/avalanche, HashDOS resistance, determinism,
`no_std`, licensing, verification) and the per-domain "good enough" reading for OS, languages,
provers, games, servers and aerospace. A named failing boundary keeps
G1 open. Absent remove/growth/codec
operations require explicit contract-boundary reporting, not silent exclusion or fabricated success.
No performance number is rerun here; partial measurements do not pass REG-1. Canonical refinement
through actual Aeneas extraction and the performance matrix must bind the same selected source.

## Slang source-binding exit: all rows required

Source binding is a later refinement leg of the same R0 primitive, not a separate semantic project.
The [bounded source-witness route](contracts/slang-source-refinement-plan.md#bounded-source-witness-route)
extends the existing `tools/slang/export.py`; memo-store evidence retains its own refinement scope.

| Required output | Gate / remaining work |
|---|---|
| Retirement before generation | `python3 tools/rooting/retire_check.py --check` and `--self-test`; R6 rejects producer references to the binned routes and emission of `maxHeartbeats 0` / `maxRecDepth 1000000`. |
| Chunked table/std UTF-8 witness | Unrun; named bounded `Utf8Scan` witnesses over the exact `tableSource` and `stdSource`, composed by `Utf8Scan.continue` at default recursion depth, with shared data/index equations and empty axiom audits. Focused output module and gate command are MISSING pending bounded implementation dispatch. |
| Unconditional `Slang.SourceAudit.source_binding` | Unrun: `python3 tools/slang/refine.py --stage source --controls` (root), exact type in `tools/slang/controls.json`; `proof/corpus/Slang/SourceAudit.lean` currently supplies only the conditional helper. Full-byte freshness and an empty cone remain required. |

Lean-backed gates use the serialized, memory-capped, single-threaded, timed wrapper in `AGENTS.md`
§6; no raised recursion depth or unbounded heartbeat setting. The supplied barrier-probe figures
in the plan are reported measurements, not proof acceptance or a fresh feasibility run.

## Policy engine exit: all rows required

Epic [`epics/policy-engine.md`](epics/policy-engine.md). A row is green only when its gate and its control are both re-run by
the lead on the same source/tool identity. The extraction identity is `proof/ApeironWord.lean` `a4599646…`.

| ID | Required output | Gate / state |
|---|---|---|
| P1 | Rust core `src/policy/{mod,decide}.rs` on the crate path | green, `tested`: 12 `policy::` tests, clippy, fmt |
| P2 | Canonical `Apeiron.Policy.decide` with empty-cone definitions and laws `decide_default_deny`, `decide_forbid_wins`, `decide_granted_le_want`, `decide_granted_mono_completion`, `decide_perm`, `decide_add_forbid_le`, `decide_deny_named`, `oneBit_no_attenuate`, `noAsk_no_ask`, five verdict witnesses | green, `proved`: `proof/corpus/Policy/{Core,Laws}.lean`; registered in `AxiomAudit.lean` |
| P3 | Real-source extraction of `policy::decide` | green, `tested`: `proof/regen.sh --check` fresh; stale-diff control red |
| P4 | Analysis layer on `Apeiron.Decider` (`canAllow` soundness and completeness, 1-bit, `eq`/`le`/`ge`) | green, `proved`: `proof/corpus/Policy/Analysis.lean`; registered in `AxiomAudit.lean` |
| P5 | Cedar differential oracle over the real crate | green, `tested`: `tools/policy-oracle` (20,000 cases, 0 mismatches) |
| P6 | Refinement `word_decide_denote`: extracted body ↔ `Apeiron.Policy.decide` on the whole input domain | **open**: `proof/corpus/Policy/Refinement.lean` (W3). No gate yet; control: swap arms in a copy of the generated body |
| P7 | Song slice attenuation under the empty cone (`songSliceFixed_le`) | **open**: `python3 tools/slang/zero_axioms.py --stage policy-attenuation --controls` (W4); spike S3b2 is model-only |
| P8 | Relationships (D2), compiler (D3), `revocation_is_forbid` (D1) | **open**: spikes S6/S7 are spike-scope only; modules are not yet planned as tasks |
| P9 | Intent seam adoption (cross-repo) | **open**: W5, owned by the intent lead |

## Other owners

- **Song A0–A6/B6** is owned by Song. Joint profiles exist, but readiness remains open until their
  selected rows have checked source-connected correspondence; A0 alone does not satisfy it.
- **Corpus F2/F3 and A5** are owned by the corpus repository. The canonical decider is now complete
  on its named fragments: `certificate_complete` (uncapped `CertOp`), `completeSat_eq_true_iff`
  (residue decision), `residue_certificate_complete` and `checkCertStream_sound` (framed replay). The
  Song executing reader refinement and the Slang source binding remain open.
- **Penglai material integration** is owned by Penglai.
- **Fold sunset** is owned by Fold.

Apeiron points to these owners; it does not dispatch whole other repositories. Current owner
roadmaps: [canonical corpus](../../sanguine-apeiron/theory/ROADMAP.md),
[Song](../../song/docs/ROADMAP.md), [Penglai](../../penglai/README.md),
[Whispering Tides](../../whispering-tides/ROADMAP.md).

## Local commands

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && lake build
python3 tools/rooting/check.py --self-test
```

`proof/regen.sh` requires the pinned Aeneas/Charon under `target/aeneas/` or `AENEAS_HOME`. Tool pins
and the archive checksum are in [proof/README.md](../proof/README.md); record the versions a run
actually used rather than citing a pin from memory. `python3 tools/rooting/check.py --self-test`
exercises the canonical-rooting rejection paths (14 reject / 8 accept); `--check` is red on the
binned routes and on the 38 rule-D sites, 16 of them genuine and un-repaired.

## Barrier order

**Terminal goal:** the admitted single R0 collapse/fold chain in
[direction](direction.md#execution-order), with checked execution refinements and G1/REG-1
acceptance, supporting the secure OS-language-proof system.

The current extraction/refinement leg continues on the real growable implementation; its dispatch
order is recorded in the
[storage plan](epics/storage-refinement.md#extraction-shaped-representation-dispatch-order-apeiron-extract-rep).
The earlier `APEIRON-PERF-RESTORE-W4-T1`–`W8-T1` chain is historical. The resource/refusal contract
is ratified in the
[local boundary](kernel-boundary.md#resourcerefusal-contract-for-the-growable-table)
(option (b), checked reservation/allocation failure, transactional); the representation wave is
unblocked, subject to separate authorization of the canonical release/commit.

**Current strategy decision (`APEIRON-STRATEGY-DECISION-T1`; Astra;
[direction](direction.md#current-strategy-decision-freeze-composition-restore-performance),
[boundary](kernel-boundary.md#w7-searchcommitboundary)).** Freeze the pending search/commit
composition as `W7-SearchCommitBoundary`: suspend repeated W7 `prepared_rebuild_refines_insertWith`
attempts and new extraction undertaken solely to unblock them. The freeze changes priority, not
joint acceptance; every theorem requirement, gate, control and trust statement is retained. The
boundary is trusted/unproved, with only individually evidenced Rust tests
labelled `tested`; it is not test-accepted as a whole. Reopen W7 only when the lead receives a
Sol-ratified, source-pinned task for full `apply_insert` `RepR` preservation (both carriers, joint
found/fresh flat/spill witnesses, rejecting controls), an acyclic composition target and a finite
discharge map for every boundary premise; each map entry must name its real body, exact statement,
gate, owner and checked primitive/tool dependency, and a missing primitive needs a separately
accepted feasibility result first. A new premise, assumed postcondition or another unchanged retry
is not progress through this condition. Performance parity does not itself reopen or discharge W7;
once the measured production mechanism is selected, restoring this proof route is mandatory before
G1 closure, and freezing it indefinitely leaves the epic incomplete.

**`APEIRON-STRATEGY-DECISION-T2` and `-T3` reconcile this schedule; neither reopens W7.** They
supersede immediate W7 retry scheduling in earlier records, not its requirements, gates, controls or
trust statements: the freeze stands with the reopening condition above unchanged, regression re-proof
continues, whole-state `apply_insert` composition and `prepared_rebuild_refines_insertWith` remain
unproved/open and suspended, and local arithmetic/generated-body repairs continue at their own gates.
T2 authorizes the partition-validation cost cut, **rejects** removal of the region payload
initialization in the current dispatch, and adds `SpillSummaryShape`, `SpillSummaryRecorded`,
`SpillSummaryMissSound` and `SpillFingerprintExact` to W7's
[ledger](kernel-boundary.md#spill-filter-obligations-and-partition-cut) as temporary named premises at
grade `anchor` — not axioms, not established `RepR` fields, not replacements for preservation — each
with a Sol-owned bridge contract and a source-linked proof owed for both carriers; every premise must
be discharged before end-to-end acceptance and no existing theorem is weakened to accommodate the new
state. T3 keeps the all-pair interval ceiling `selected/incumbent <= 1.0`, authorizes size-only
factoring and then measured read-path specialization, and retains every named incumbent, the coverage
gaps, the HashDOS requirement, the controls, both axiom cones and the realization exclusions.

### Subsumption execution sequence (`APEIRON-SUBSUME`)

Owner reorientation decision; grade `anchor` — architecture and scheduling. It establishes no new
performance, test or proof result. All lifecycle milestones keep **no due date**; G1 is "in flight —
completion horizon unestablished". A stage counts only when its gate and its negative control pass.
G1/REG-1 stays **OPEN**, `full_acceptance=false` on every recorded tally.

**First principle.** Empiricism and formalism are two sides of one coin: **subsume the empirically
best known solution first, and formalize the subsumption**; never iterate from an obviously inferior
primitive. The empirical side selects the base — the measured winner becomes the working artifact, on
the measured path from step one; the formal side certifies that same artifact. **The base is
`../fold/hashtrinity`** (pinned `Cargo.toml:76-80`, rev `4cc14471`). The generic growable `Table` is
the inferior primitive and is not a valid base for a count-face optimization. The prior five-barrier
derivation sequence is **superseded for the count face**: its obligations are retained, its base is
corrected, and its from-scratch container path is parked (below).

**Targets — two faces, both validated.**
- **Count face** (packed `(i21,i21,i21)`; the registered `hashtrinity_*` arms). INITIAL TARGET = **parity**
  (`selected/incumbent <= 1.0`), reached by the literal subsumption — **landed: 3.985 -> 0.921**. FINAL
  TARGET = **0.50** on the three hashtrinity arms, reached on the axis the winner did not optimize.
- **Map face** (stored key; the serial/`current-strata` arms). `hashbrown_fxhash` beats `hashtrinity_table`
  by **27.5%** on this face (`B-LEARN`, 13/18 runs; wins 100% of absent-miss cells). The map face must
  **beat `hashbrown_fxhash`** — `selected/hashbrown_fxhash <= 1.0` — and the technique is extracted, not
  invented: the **16-lane fingerprint glance over a 1 B/slot ctrl array** (apeiron's spill tail: 176
  insns / 48 branches / 16 loads per group against hashbrown's **6 SIMD insns / 0 branches / 1 load**) plus
  **fxhash's hash choice on a fixed table**. Note the falsifier: hashbrown's serial `get` is **4.7× longer
  in instructions** (47 vs 10) and still 27.5% faster, so the win is latency/cache, not instruction count.
- **Both faces formally validated.** The refinement of the selected production body to the canonical
  object, the closed axiom cone, real-source extraction and mutation controls apply to the map face as
  they do to the count face. The HashDOS vulnerability is formalized with the chain and patched after
  (deferred obligation, below), not before.
- **Horizon.** Under 24 hours to the acceptance point. Space Bunny (free) is the dispatch default and is
  used at width — many small bounded lanes, disjoint paths, serialized only on shared files.



| stage | work, hardest contribution first | gate that opens the next stage |
|---|---|---|
| **1. Subsume** | bring the empirical winner's counting code onto Apeiron's production path **unmodified** — the exact crate at the pin, or in-tree vendored with no logic change; wire it to the timed path (`examples/reg1/adapters/spatial.rs:674-681`); name the removable differential | `cargo build --locked` on the winner's code path; the selected counting arms wired; the counting gate green |
| **2. Measure the subsumption** | the **INITIAL TARGET**: the ratio the subsumption reaches, per arm, ≥5 paired runs, mean and spread | `selected/incumbent <= 1.0` on the three hashtrinity arms, outside the unchanged-code spread |
| **3. Beat it** | the axis the winner did not optimize — exactness/identity without a duplicate key column (`pack` is injective; address == key); the fused liveness+identity+hit decision; mask-for-divide (`& (2^k−1)`); the 16-slot group glance; the enumerated waste. Each a bounded change with its gate and control | measured pairs move outside spread toward `0.50` |
| **4. Formalize the subsumption** | extract the **selected production body**; constructor, preservation, read, build and refusal links; joint reachable witnesses; every headline in `proof/corpus/AxiomAudit.lean` under the closed cone; mutation control on the real source | fresh `proof/regen.sh --check`; bounded builds in `proof/` and `proof/corpus/`; required headlines, audits, joint witnesses and semantic controls green |
| **5. Joint acceptance** | run the unchanged full registered population under the owner target scope, with the identity and HashDOS requirements, the coverage census and the shared integration gates | strict full-profile acceptance exits 0 with `full_acceptance=true`; every required Rust and proof gate and control green on one source/tool identity. Only then does downstream development reopen |

**Owner target scope.** The harness applies `REG1_TARGET_RATIO=half` to the three hashtrinity arms —
`hashtrinity::Table`, `hashtrinity::CoordMap` and `hashtrinity::FreqMap`, serial and packed-parallel —
so `selected/incumbent <= 0.5` there, and `selected/incumbent <= 1.0` on every other named incumbent:

```sh
REG1_TARGET_RATIO=half cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

Neither "<= 0.5 everywhere" nor a default 1.0 acceptance describes this owner target. The unchanged
all-pair interval ceiling and every named incumbent, coverage gap, HashDOS requirement and control
below stay in force.

**Historical tuning routes: closed experiments, not active recovery tasks.** Items (2) and (3) of
the superseded performance workstream order — size-only factoring of `src/table.rs`, then serial
direct-path overhead reduction — are **closed experiments** on their recorded source identities: each
was tried, and neither closed the applicable-pair gap, so neither is an active recovery task and
neither is re-dispatched under the barriers. Item (1), comparison coverage under
`APEIRON-REANCHOR-W1-T1`, landed and is retained; items (4) and (5) are absorbed by barriers 2–3 and
barrier 5. The [roadmap dispatch table](roadmap.md#performance-re-anchor-and-bounded-dispatch) keeps
their historical task IDs, files and gates. Their receipts stand as history and establish no rate. No
aggregate, std-only win, historical envelope, reduced matrix or trusted-boundary grade closes G1.

**Task IDs and dependencies, recorded here before any board row is created.** The authoritative task
list, its owned files, gates, negative controls and budgets is the contract's plan in [the storage
epic](epics/storage-refinement.md); a board row is created from an entry there and never precedes it.
The stage order and dependencies:

| stage | tasks | dependency that opens it |
|---|---|---|
| 1 | `APEIRON-SUBSUME-HT`, `APEIRON-SUBSUME-PROBE`, `APEIRON-SUBSUME-COST`, `APEIRON-P-P1`…`P7`, `APEIRON-B-B10` | the subsumption decision; `P1`–`P3` land the winner's code (`cargo build --locked` exit 0), `P4`–`P6` wire the selected arms, `P7` measures parity |
| 2 | `APEIRON-SUBSUME-PARITY` | stage 1 wired and the counting gate green |
| 3 | `APEIRON-B-B1`…`APEIRON-B-DIV` (identity without a duplicate column, group glance, fused decision, the enumerated waste, the arms, packing, mask-for-divide) | stage 2 measured at parity |
| 4 | `APEIRON-V-V1`…`V5` | stage 3's selected body frozen |
| 5 | `APEIRON-A-A1`, plus HashDOS, the coverage census and the oracle | all of the above |

**Deferred obligation (owner direction).** The **HashDOS requirement is deferred until the speed target is
met across the board** — it is not waived. Order: speed first; then **formalize the entire chain,
including the vulnerability**; then **patch the vulnerability** as hashtrinity already does, this time
with the formal understanding on record. Rationale: the vulnerable quadratic is understood intimately,
which is the evidence a written account needs, and the patch is then a known action applied to a known,
formalized object rather than a bolt-on.

**Before any transfer, document the formalization challenges of the succeeding path.** The count face
succeeds now (parity drawn from the incumbent) and its formalization is where the open obstacles are:
Aeneas rejects `return` inside loops (fix: a `while` carrying the result out); `RangeInclusive::contains`
and `IntoIterator for &Vec` put bodies outside the closed cone; `build` has no subject in the counting
body; the extraction/audit round costs ~1200 s so it does not fit one 570 s invocation; and the base
corpus (`Growable/Scan`, `Growable/Search`) is stale against the generated region. Those are recorded
against the path that works, before the vulnerability work is layered on it.

**The HashDOS defense exists in the old corpus and is a port candidate** — `fold/hashtrinity`:
`tests/hashdos.rs` (the red-team set, mechanisms 1–5), `proof/HashTrinity/RadixDescent.lean`
(`fold_injective`, `chain_separates_within_radix`), and `synth_breaks_the_exploit`
(`sanguinehost.com/public/wiki/md/FoldDepth.md:444`), with the theorems of record in
`sanguine proof/Closures/Engineering/RadixDescent.lean`. **Caveat, from hashtrinity's own header:** those
theorems bound **aliasing, not probe length**; this tree's failure is probe length, so the port is a
starting point, not a discharge. Transfer after the succeeding path is formalized and documented.
**The transfer is already scoped on the Icarus board** (`icarus-rooted-sha-256-GXc6ulF46UDoJGtahKV4lS`):
`H1 — Port FoldDepth (fold_is_lossy, xorHalves_balanced, …)` and `H2 — Port RadixDescent core
(fold_injective, chain_separates_within_radix)`, with `H3` (keyed fold), `S1–S4` (finalizer budget,
SHA-256-compress-as-finalizer, peppers), `Y1` (the cryptographic-hashing position doc) and `T1–T5`
alongside — 16 rows, all Not started. Do not duplicate them here.

**Parked (retained, not waived).** The prior derivation path — the from-scratch counting container
`APEIRON-COUNT-W2-T1`/`W2-T2`/`W2-T3`, `APEIRON-R1-W1-T1`, the `W2-T1b-DOMAIN`/`FULLWIDTH` splits,
`RESIDENT-GROWTH`, `SPILL-PROVISION`, `COUNT-READBACK` — is parked: obligations and evidence retained,
the base corrected to the subsumed structure. It may be reopened only where the subsumed path leaves an
obligation it alone discharges, and then with its own contract.

No task opens before its dependency closes, every task owns at most three files with one gate and one
negative control under an enforced budget below 600 s, and no file has two owners.

**Parking rule (`APEIRON-REORIENT`).** New Intent and Song development is parked behind joint G1
acceptance. Reopen when `full_acceptance=true` and the required real-source end-to-end refinement
both hold on the accepted identity. Requirements are **retained, not waived**: the [E5
Apeiron–Intent engine](#e5--apeironintent-engine) waves, the Song A0–A6/B6 rows under
[Other owners](#other-owners), the sustained-width campaign below and the cross-repo correspondence
keep every obligation, gate and control. They change status, not requirement. Already-dispatched
returns integrate only under their existing gates and ownership, and parking authorizes no new
implementation. Projects stay retained, not cancelled or archived, and Done rows keep their evidence.
This is a scheduling decision: `AGENTS.md` §3a freezes Penglai and webfold behind the same condition
and does not, on its own, freeze every Intent and Song row.

**Superseded performance workstream order (historical).** The order it replaced was: consume the
fresh full-profile receipt before changing timed source or harness; then (1) repair comparison
coverage, which landed under `APEIRON-REANCHOR-W1-T1` (the five named hashers are load-bearing in
`examples/reg1/acceptance.rs` with rejecting omission/demotion controls, and the combined
`get_each` arm is recorded as `concurrency_unmatched`); (2) size-only factoring; (3) serial
direct-path overhead; (4) spatial partition/build/probe recovery on disjoint paths; (5) a matched
before/after full-profile receipt. Items (2) and (3) are the closed experiments above; items (4) and
(5) are barrier work.

**Revised representation boundary (Astra verdict).** The selected structure becomes a table-level
flat resident area (shard-major `256 x L`, admission `direct_word = Some(w) ∧ lane < L`, checked
sizing, a stated byte budget, a density test; failures spill rather than refuse) beside
independently load-sized grouped spill regions. One representative per key is held under a composite
`DirectCover`; extent growth must reclassify newly eligible spills. Key-dropping uses a compile-time
`KeyRep` policy (`Stored<K>` or `Regen` with an exact codec law), not a sibling count table and not a
lossy `FreqMap`. Restoration also includes the key-partitioned load-sized shard build, deterministic
schedule, grouped probe and key-dropping count/dedup; proof, extraction, registration and
measurement work follows on the same source identity. This is a recorded plan, not a built,
extracted or measured result, and no speed recovery is claimed.

1. **The immediate task:** the five barriers above. The superseded tuning dispatch — serial
   direct-path overhead reduction, then spatial partition/build/probe recovery on disjoint paths — is
   a closed experiment recorded above, not an active recovery task. The per-entry relocation roots
   for both carriers are bound and their flat
   and spill headlines are proved; the prepared-resource and search-decision/apply roots remain bound
   at their audited scopes, but the W7 composition that would close them is frozen under
   `W7-SearchCommitBoundary` until its reopening condition is met. The fixed-shard table is not
   optimized from one workload. S8/G1 stays OPEN until every required row passes jointly with
   canonical refinement.
2. **Connect the canonical collapse/fold chain.** Close the audited cross-revision equivalence and
   admitted normal-form/translation links. Local proofs and old speed anchors do not close it.
3. **Gate repair before source refinement.** Restore rejecting gates and run the rooting gate
   `python3 tools/rooting/check.py --self-test`. Gate repair and the canonical-root acceptance
   below precede any producer, generator, source-translation, proof-producing generation or
   derivation work.
4. **Canonical-root acceptance.** Every executing face is accepted only when it is rooted in
   `Apeiron.Table.HMap`, `Apeiron.Table.get`, `Apeiron.Table.insertWith` and `Apeiron.Table.build`,
   and in `Apeiron.Trichotomy.AnyRing.RingCell`, `ringMul` and `ringNormForm`, with the extracted
   `Cell<i64>` word bodies `word_mul_denote` and `word_norm_denote`. Anything not rooted in it is
   not accepted.
5. **Retire the rejected route.** The binned `proof/corpus/Slang/Derivation/` tower and the binned
   `Slang/Generated/Trace/` outputs are unrooted; the `Run`/`Derives` acceptance in the binned
   `tools/slang/trace_check.py` is unrooted. They are binned and re-rooted, not extended. The
   retired `DecodeTrace.Run`/`Derives` trace representation, the retired named proof DAG and the
   retired segment lemma are historical and are not the active architecture.
   Require `python3 tools/rooting/retire_check.py --check` and `--self-test`, including producer
   rule R6, before extending `tools/slang/export.py` with bounded source witnesses. Proof-term
   chunking over the existing structural source relations does not reinstate the retired evaluator.
6. **Storage and integration.** Retain S1-S7 at their explicit scopes; rerun S7 on one source/tool
   identity for closure. S8/G1 remains OPEN pending REG-1, not closed by a narrower claim.
7. **Execution correspondence.** Close Song source binding, then compose source-connected
   refinements through `Joint.WordBridge` with the decoded-context witness and R2/T3 composite.
   This is the later execution leg of the same primitive, before any larger vertical implementation.
8. **Apeiron barrier** is a necessary component of joint readiness, not sufficient: the Song
   foundation, the joint primitive correspondence and the named profiles are also required before
   the next large implementation opens.

## Phasing: epics, waves, tasks — draft for review

**Fleet assumption.** About 50 agents available, 10 lanes, and **at most 3 simultaneous heavy Lean
checks** (about 44 GiB; ten 12 GiB scopes would reserve 120 GiB). Isolated caches; one owner per
file; one writer per shared build product. The **default worker is `openrouter/stealth/space-bunny-alpha`**
(free, OpenRouter, **free window ends 2026-10-05**; measured 147.9 tok/s / TTFT 1.19 s versus the
slower OpenCode Zen route `opencode-go/space-bunny-free` at 57.5 tok/s / TTFT 6.95 s) for every
dispatch-ready task. `deepseek-flash` is the paid escalation taken only on a mandatory trigger;
anything beyond that escalates manually to the owner. `gemini-flash` is a non-default lane, not
spent on ordinary work.

**Sustained width is a requirement on Intent, not only on this host.** It is **parked**, not
cancelled, under the [parking rule](#parking-rule-apeiron-reorient): the requirement is retained and
no sustained-width campaign lane opens before joint G1 acceptance. Observed at ~35–50
concurrent Space Bunny workers through this harness: tool calls slow, sessions restart mid-task and
resume without their conversation. That is a statement about the current router and is **not** a
reason to reduce the width — the target is that Intent carries ~50 concurrent agents seamlessly.
Every dispatch is then a message on a channel with its own identity, budget and receipt, so a worker
that restarts loses a lease rather than a task, and no coordination state lives in the harness.
The programme's own decentralised-comms machinery is what this rests on: the relay/leaf split, the
per-run nonce and the end-of-case drain token, and frame admission with its refusals. The
Intent-side obligation is that those cover fleet coordination as they cover evaluation traffic.
Until then a restarted session re-derives its state from the tree and the fleet-state file, never
from conversation.

**Sustained width is testable, and its first measurement was due within 48 hours** (owner direction,
2026-10-02); that scheduling is superseded by the parking rule above and the obligation is retained,
not waived. The claim is not accepted on architecture alone: Intent must run a harness that
dispatches **50 concurrent workers** and reports, from its own instrumentation, that every worker
carried its own identity, that no dispatch was lost when one worker was killed and re-leased, and
that a receipt exists per worker. The harness is a gate: it has a reachable failure (dispatch fewer
workers, or drop the lease, and it reddens), it names the machine, and it records wall-clock and
completion counts. Until that run exists, the width requirement stays open and is reported as open.

**Vocabulary.** `epic → wave → task → subtask`. A **wave** is a parallel batch: its tasks hold
disjoint files and run concurrently, and the wave closes only when every task's gate and its control
are green on the same source/tool identity. A **task** owns ≤3 files, one gate, one control and one
arc. A **subtask** is one agent's unit inside a task — one file edit and one gate run. IDs are
`APEIRON-<EPIC>-W<wave>-T<task>`; every wave below names its files and gate, so nothing is a bare
code.

**Lane budget.** Each wave declares agents (≤50), lanes used (≤10) and heavy Lean checks (≤3). A wave
wanting more than 3 heavy checks splits into sub-waves; a wave whose tasks are gate-heavy but not
Lean-heavy (cargo, docs, schema) can use all 10 lanes.

**Anti-duplication.** Before dispatch the wave's file sets are intersected mechanically; two tasks
naming one file is a planning error, not a merge. Two tasks over one source body is the same error:
`Table::relocate_all` was modelled twice (`StoredRelocateAll`, `RegenRelocateAll`) and one was
wasted. The wave table is the check, and `gate docs check` automates it once wired.

### E1 — Gate and receipt machinery (Rust, `tools/gate`)

| wave | tasks | files | gate | agents | heavy |
|---|---|---|---|---|---|
| W1 | freeze the conformance record for the 36 `proof/resource/*_gate.py` (30 tracked, 6 untracked): green markers, named mutation red | `docs/contracts/artifacts.toml`, `tools/gate/conformance.json` | record matches a re-run of a sample | 4 | 0 |
| W2 | extract `admit` through Charon/Aeneas and refine it | `proof/gate/Admission*.lean`, `tools/gate/core.rs` | `proof/regen.sh --check` on the gate root | 1 | 1 |
| W3 | manifest schema + loader; adapter (fs, hashing, subprocess, locks, budget) | `docs/contracts/artifacts.toml`, `tools/gate/adapter.rs`, `tools/gate/main.rs` | `cargo test --manifest-path tools/gate/Cargo.toml` | 2 | 0 |
| W4 | aggregate `gate docs check --controls`; wire into dispatch and the wave barrier | `tools/gate/main.rs`, `docs/plan.md` | the aggregate on the repo, controls red then green | 1 | 0 |
| W5 | pilot: RowDot under the standard cone via `tools/gate`, parity with its deleted script | `proof/resource/RowDot.gate.toml` | parity + retirement receipt | 1 | 1 |
| W6 | migrate the remaining 33, batched 5 per wave | `proof/resource/<module>.gate.toml` | each equals its frozen record | 5/wave | ≤2 |
| W7 | delete every `*_gate.py`; reject new ones | `proof/resource/` | none remain; rejection control | 2 | 0 |

### E2 — Receipt consolidation (option C)

| wave | tasks | files | gate | agents | heavy |
|---|---|---|---|---|---|
| W1 | index the evidence receipts — **landed** `003bfad` (67 rows: 62 receipt, 2 log, 3 figure) at `docs/evidence/ledger.jsonl`; the index sits beside the files it indexes and records the kind per row, rather than being split across `artifacts/<kind>/`. Remaining: fold the per-run wrappers and the stray lane docs (bench done; reviews, intent-lane docs remain) | `docs/evidence/ledger.jsonl`, `docs/evidence/` | `python3 tools/rooting/index_check.py --check` (exactly one fresh row per receipt; `--self-test`, `--controls`) | 4 | 0 |
| W2 | reachability gate for removals | `tools/rooting/retire_check.py` | `python3 tools/rooting/retire_check.py --check` plus `--self-test` and `--controls` (the bare string cannot run; E2.W2 receipt) | 1 | 0 |
| W3 | delete the absorbed documents; classify the root junk | `docs/`, repo root | no dangling citation | 2 | 0 |

### E3 — Source-cost models (the outstanding returns)

| wave | tasks | files | gate | agents | heavy |
|---|---|---|---|---|---|
| W1 | LmHead row-and-dot resource proof: the premature witness selection, the equality-vs-existential mismatch in the continuation, the missing `vis`/`div` totality argument, and the `row_visits_closed_form` prose that claims the loop "performs `vocab.val` row visits in total" while the theorem proves only the initial rank | `proof/resource/LmHeadResource.lean` | `python3 proof/resource/LmHead_gate.py --check --controls` | 1 | 1 |
| W2 | `regen_relocate_entry` gate + commit (returned) | `proof/resource/RegenRelocateEntry*` | its gate | 1 | 0 |
| W3 | **retired as a split** — no `proof/resource/ExactDotResource.lean` exists; the dot-loop work measure (`dotVisits`, `dot_body_visit`, `dot_body_exit_end`) lives in `LmHeadResource.lean` and is W1. Two tasks on one file would be the §9.2 anti-duplication error. | — | — | 0 | 0 |
| W4 | `matvec` completion — `RowDotAdmission` landed at `405df80`; matvec in flight | `proof/resource/MatvecResource.lean`, `Matvec_gate.py` | `python3 proof/resource/Matvec_gate.py --check --controls` | 1 | 1 |
| W5 | `RegenRelocateAll` vs `StoredRelocateAll` disposition — compare carrier assumptions, domains, refusals and claimed resource properties; do **not** merge for a shared body | `proof/resource/{Stored,Regen}RelocateAll*.lean` | disposition receipt, §9.2 | 1 | 0 |

### E4 — Extraction freshness

| wave | tasks | files | gate | agents | heavy |
|---|---|---|---|---|---|
| W1 | move `dequant_f32_row` into `src/engine/f32_convert.rs`; point `loader.rs` at it | `src/engine/{mod,f32_convert,loader}.rs` | `cargo build --manifest-path proof/extract/Cargo.toml` exit 0 | 1 | 0 |
| W2 | close the `SOURCE_FILES` digest gap for the mirrored sources | `proof/regen.sh` | digest list covers every mirrored path | 1 | 0 |
| W3 | clear the E0514 sysroot mismatch; regenerate `ApeironWord.lean` | `proof/ApeironWord.lean` | `proof/regen.sh --check` | 1 | 1 |
| W4 | attn-family re-proof against the regenerated root | `proof/corpus/` | `cd proof/corpus && lake build` | 1 | 1 |

### E5 — Apeiron–Intent engine

| wave | tasks | files | gate | agents | heavy |
|---|---|---|---|---|---|
| W1 | the three corrections: D7 rebase epoch, D2 cost bound, loop-brief target | `intent/docs/EVOLVER.md`, `intent/crates/intent-core/src/selection.rs`, `docs/epics/apeiron-intent-loop.md` | each names its new statement and its control | 3 | 0 |
| W2 | Ring 6 selection gate — **landed** `df33ee9`/`14c6932` (four D1–D4 controls; reddens under a Ring-3 admission mutation, sha-clean restore). Plan-line correction per Sol: the memorizer / reference-grading / held-out controls belong to the oracle slice W2b, not here | `intent/crates/intent-cli/tests/ring6_selection.rs` | `cargo test -p intent-cli --locked --test ring6_selection -- --test-threads=1` | 1 | 0 |
| W2b | oracle Ring 6 (Sol route: randomised, reference-graded inputs first; secret held-out cases later): seeded generator + trusted-reference labels, then candidate grading with the public-table memorizer control; 540 s is the candidate-grading budget, not the gate command's | `intent/crates/intent-cli/tests/oracle_cases.rs`, the generator, one focused test | T1: reproducible seed + inputs absent from the public table; T2: reference passes, memorizer fails | 2 | 0 |
| W3 | evaluator and isolation repairs (Ring 2 `--unshare-pid`) | `intent/crates/` | the named probes | 2 | 0 |
| W4 | runtime linkage and canonical rooting of the admission cores | `intent/Cargo.toml`, `intent/crates/intent-core/` | executing-source correspondence to `Apeiron.Table` | 2 | 1 |
| W5 | calibration: ≥30 paired incumbent reruns | `intent/runs/`, receipts | registered population, no protocol change | 1 | 0 |
| W6 | five candidate screens (matched settings, one cap per cell) | `intent/runs/` | each candidate's own control | 5 | 0 |
| W7 | one complete improvement cycle | receipts under `artifacts/` | the full acceptance barrier | 2 | 0 |
| W8 | second cycle driven by the first cycle's evidence | as W7 | as W7 | 2 | 0 |

### Dependencies and the critical path

`E4.W1 → E4.W3 → E4.W4` gates any new extraction. `E1.W2 → E1.W4` gates dispatch wiring.
`E1.W5 → E1.W6` gates the script deletion. `E5.W2 → E5.W7` gates an accepted cycle. `E3` and `E2`
run in parallel with all of these: they are cargo- and document-bound, not Lean-bound, so they use
lanes the Lean waves leave free. For the G1 goal the governing critical path is now the
[five barriers](#subsumption-execution-sequence-apeiron-subsum); the E-waves below it are
retained dependencies of barrier 4 and of the shared integration gates, not an alternative order.

### Questions for review

1. Is 3 heavy Lean checks the right cap, or should the measurement scale to the source of the 44 GiB?
2. Does `E1.W6` batch at 5 modules per wave, or is 10 lanes with ≤2 heavy the better packing?
3. Should `E3.W5` delete `RegenRelocateAllSource.lean` now, or retain it until `StoredRelocateAll`'s
   gate is re-run under the new driver?
4. Does `E5` start before `E4` closes, given Ring 6 needs no extraction?

### E6 — Close the hashtrinity gap

Owner direction: the hashtrinity-to-apeiron conversion was never done, and it is now the
highest-value item, because Penglai's fluid simulation depends on this table and the gap is the
difference between a workstation and a modest machine.

**Measured** (`cargo run --release --features read-stats --example table_cost_probe`, this machine,
192 keys, `N = 16`):

| population | metric | apeiron | hashtrinity `4cc14471` | ratio |
|---|---|---|---|---|
| distinct | present ns/key | 9.73 | 1.04 | **9.35x** |
| distinct | insert ns/key | 480.98 | 4.04 | 119x |
| lowbyte | present ns/key | 31.52 | 2.12 | 14.9x |
| adversarial | present ns/key | 292.35 | 0.78 | 374x |

**The requirement already exists.** `docs/direction.md:62` makes the consolidation
performance-preserving — it carries forward the predecessor's high-performance surfaces, not merely
its names or formulas — and the G1 acceptance cells (`docs/direction.md:283-287`) name the
comparators each required cell is measured against. A 9.35x gap on the plain distinct baseline,
where nothing is adversarial, is that requirement unmet.

**Attributed, not argued** (E6.W1 and E6.W1b, scratch tree at `30180f4`, same harness): the
resident read is at parity with the predecessor — 1.0625 against 1.0521 ns/key, ratio 1.01 — and
the gap is **entirely the spill read path**, which the harness reached because its `IdentityHome`
returns `direct_word: None`. With an injecting certificate on every key the `read_stats` counters
are all zero, proving `spill_read` is never entered, against 1.00 sweep step/key on the spill
control. The read prologue, region indirection and the bit scan are refuted as contributors by the
same run (256 regions to 1: −0.74 ns/key; `least_set_lane` already lowers to `tzcnt`).
E6.W2's port is therefore about the spill read only, and it **cannot assume residency**: the
resident extent is 256 words and the density rule admits only lane 1 at 192 live keys, so the
injective certificate is 8 bits; a `place < 2^32` population is not a resident population. The
insert column is ~100x in every arm and is locator reservation, not the probe.

**Decomposed further (E6.W2a; the box was contended, ±2 ns/key noise band, stated).** The additive
model fails: removing the shape guards, the summary test, the lane-search ladder, the full compare,
the loop and the double region index — individually or together — moved nothing outside the band,
and their joint removal measured slower. What holds: the tag SWAR is the largest single implemented
mechanism (~4.3 ns/key) and is already the fast form (a scalar lane loop is worse), and the
remaining ~8.4 ns/key is per-read fixed overhead — indirect generic dispatch, bounds-checked slice
construction, region-header deref, address derivation — independent of cache state, since a
single-region population with identical geometry still reads ~10 ns. The profiler pass (E6.W2b;
shares only — the release binary carries no DWARF and fp unwinding truncates) locates the largest
block in `matches()` at 27.81% of `spill_read` self time, whose hottest instruction is the dependent
address formation for the 24-byte-stride slot payload. The candidate slot-order change was refuted
by an interleaved measurement (E6.W2c: delta −0.19 ± 0.75 ns/key, and both instantiations
instruction-identical in disassembly), so that route is closed. The structural route is confirmed:
a flat open-addressed spill region prototype measured 1.34 ns/key against the canonical 8.14 in the
same process (6.06×, below the resident read; E6.W2d, reproduced by E6.W2e), so the port's target is
the descent — one load per probe step where the canonical carries dependent header loads, stack
round-trips and a serial summary bit-loop. Two instrument facts from W2e: the `read-stats` feature
costs 1.90 ns/read on the canonical path, so timing and attribution are separate builds, and the
resident arm reconciles with `micro_spill` (1.355 against 1.327 ns/key) once the keys land in one
contiguous lane rather than scattered across lanes. The port's absence/Grow contract is answered by
Sol — shape A, transcribed in [the read-path epic](epics/read-path-specialization.md).

**The adversarial column was a filter defect, closed by `55421fa`**: the tag read the same low 16
bits the shard selector reads, so a population folding those bits matched every lane of every
visited group. The spread top-bits fix takes full compares per key from 107.25 to 1.3646 under the
starving identity home, with the raw formula retained in the harness as a live control column
(`distinct_fingerprints_raw`).

**Reference-first rule for the port.** The layout to study is the predecessor's cold read path at
hashtrinity `4cc14471`, `src/lib.rs`: `get_coord` (line 671) is a one-load home probe whose tail is
`get_coord_cold` (696) — one branchless SIMD glance over a contiguous `LINEAR_G = 8` linear prefix
(`prefix_masks`, 320; `LINEAR_G`, 304), then `get_coord_walk` (721) over `descend` (589)'s MSB-first
radix windows and its terminal sweep. That is the descent the flat prototype measured at 1.34 ns/key,
and it is the layout E6.W2's three slices converge on: one load per probe step, no dependent header
loads, no serial summary bit loop. Two properties are **not** carried. `coord_of` (553) remaps
`EMPTY`, `TOMB` and `TOMB - 1` onto one value, so its `get` trusts coordinate identity across an
aliasing the sibling's own comment states; Apeiron keeps exact `(Sigma,K)` identity and its
transactional refusal contracts. And the terminal sweep is termination, not HashDOS resistance: the
predecessor's red-team file records `descent_can_be_wholly_blocked` building keys whose homes cover
every address a chosen key's descent tries, and says so in its own header — non-aliasing is not a
probe bound. Apeiron's bound is therefore stated as a *bound* (a completed cycle visits every slot,
so absence follows) and never as a flood guarantee.

**Gate of record.** `examples/spill_read_probe.rs` (landed `715592c`) is the graded gate for every
port slice; the scratch prototype harness that produced the 6.06x figure is gone. The graded
acceptance is the matched ratio `spill median / resident median` from **one interleaved run**,
`<= 2.2`, on the **un-instrumented** release build `cargo run --release --example spill_read_probe`.
The per-arm band `(0.90, 3.00)` ns/key is a reference print, not the criterion; a
`cargo run --release --features read-stats --example spill_read_probe` run is counters-only, prints
`not-graded` and supplies no timing figure in either direction. Every timed run binds to a source
identity: it must not share a target directory with another worktree — force the rebuild
(`touch src/lib.rs`) or set a private `CARGO_TARGET_DIR` — because a build that linked a sibling
worktree's objects reported 2.02 ns/key and "passed" against a live tree that measures 9.26–9.48
([receipt](evidence/locator-reserve-quadratic-and-reg1-attribution.md)). That matched ratio is the
port's graded acceptance and is `measured` only from
`cargo run --release --example spill_read_probe`.

The port's file set is the eight `src` files the scratch port at `/home/socol/oc-tmp/e6w1` changes,
sliced in compile order. It is one representation change to one type, so no intermediate state
compiles: the three W2 tasks are serialized in the order below, share one gate, and the wave barrier
carries the compile. `src/table/scan.rs` is **not** in the port's set (it was retired under section 10 in `c3b0d63`) — the port retires the tag bitmap and
the group descent, and all five production callers it had — three `scan::match_tag` and two
`scan::match_empty`, in `src/table/probe.rs` — are inside the set, so those two helpers are left
reached only by their own unit tests.

| wave | task | files | gate | control | agents | heavy |
|---|---|---|---|---|---|---|
| W1 | attribution — **done** (E6.W1 and E6.W1b): resident parity 1.01; the gap is the spill read; residency is not free (an 8-bit certificate at 192 live keys) | `examples/table_cost_probe.rs` and its variants | `cargo run --release --features read-stats --example table_cost_probe` — counters plus ns per key | the `distinct_fingerprints_raw` column stays live: a tag that reads the shard selector's low bits must be rejected | 1 | 0 |
| W2 | `APEIRON-E6-W2-T1` flat region representation and the bounded probe: `SpillSlot` carries marker, `Sigma`, held carrier and payload in one array; `bound()`/`home()`/`markers()`; `spill_locate`, `spill_locate_held` and `spill_read` walk one bounded linear cycle over the capacity; summary, tag array and group descent retired | `src/table/state.rs`, `src/table/probe.rs` | `cargo test --locked --lib --test table` | in a throwaway copy stop the read cycle one step early: the finite late-hit fixture must fail on a wrong default, restore green | 1 | 0 |
| W2 | `APEIRON-E6-W2-T2` production callers: `get_at`/`apply_get`/relocation/rebuild read the named slot fields; `grow_spill` sizes the successor from the region `live` count on a `Grow` signal; `observe_receipt` records one `slots` bound; `RegionReservation` drops `tags`/`payloads`/`summary` | `src/table.rs`, `src/table/prepared.rs`, `src/table/resources.rs` | `cargo test --locked --lib --test table` | in a throwaway copy size the successor from `slots.len()` instead of `live`: the growth fixture must fail, restore green | 1 | 0 |
| W2 | `APEIRON-E6-W2-T3` remaining consumers and fixtures: `locations`/`payload_at` and the region load count read `slot.live()`; publication-order and geometry fixtures move to the marker protocol; the finite probe fixtures cover fingerprint collisions, `Grow` and late hits | `src/table/locator.rs`, `src/table/parallel.rs`, `src/table/tests.rs` | `cargo test --locked --lib --test table`; and the W2 grade `cargo run --release --example spill_read_probe` | in a throwaway copy write the nonzero tag before the payload: the publication-order test must fail, restore green | 1 | 0 |
| W3 | `APEIRON-E6-W3-T1` probe-bound replacement: the constructor establishes the placement invariant (capacity zero or a power of two, `live` exact), insert and relocate preserve it, and `SpillProbeBoundAbsentSound` derives absence from a completed cycle; the summary fields leave `RepR`/`ReservedRepR` | `proof/corpus/Growable/Scan.lean`, `proof/corpus/Growable/Representation.lean`, `proof/corpus/AxiomAudit/Core.lean` | bounded `cd proof/corpus && lake build Growable.Scan Growable.Representation AxiomAudit` | in a throwaway copy drop the constructor's power-of-two obligation from the placement headline: it must fail, restore green | 1 | 1 |
| W3 | `APEIRON-E6-W3-T2a` retire `SpillSummaryMissSound`'s definition, its `summaryMissSound_of_repR` classifier bridge and its use in the `SearchDecisionCorrect` non-found sum — only once T1's replacement is green | `proof/corpus/Growable/Representation.lean`, `proof/corpus/Growable/Classifier.lean`, `proof/corpus/Growable/Search.lean` | bounded `cd proof/corpus && lake build Growable.Representation Growable.Classifier Growable.Search` | in a throwaway copy keep the retirement and weaken the read absence certificate to a per-slot statement: the search headline must fail, restore green | 1 | 1 |
| W3 | `APEIRON-E6-W3-T2b` retire the read-loop's dependence on the summary certificate and its audit entry, at the checked-caller inventory `SpillSummaryMissSound` → `Representation.lean:141,611`, `Classifier.lean:13`, `Search.lean:119`, `ReadLoop.lean:772`, `AxiomAudit/Core.lean:791` | `proof/corpus/Growable/ReadLoop.lean`, `proof/corpus/Growable/Read.lean`, `proof/corpus/AxiomAudit/Core.lean` | bounded `cd proof/corpus && lake build Growable.ReadLoop Growable.Read AxiomAudit` | in a throwaway copy delete the completed-cycle premise from the read-loop absence conjunct: the headline must fail semantically, restore green | 1 | 1 |
| W3 | `APEIRON-E6-W3-T3` extraction re-link: `spill_read`, `spill_locate`, `spill_locate_held` and `Region::publish_live` change shape, so regenerate on the pinned tools and re-link the refinements that name them; the regeneration rides behind the pin-assert blocker | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | `proof/regen.sh --check` (behind the pin assert) | in a throwaway copy mutate the generated bounded-probe step: freshness must fail, restore green — not the semantic control | 1 | 1 |
| W4 | `APEIRON-E6-W4-T1` gate of record — **landed** at `715592c`: the numeric target lives in the harness rather than a one-off script, and its acceptance is the matched ratio above | `examples/spill_read_probe.rs` | `cargo run --release --example spill_read_probe`; its acceptance logic under `cargo test --locked --example spill_read_probe` | the unit check `the_acceptance_rejects_a_slow_spill_read` must reject the measured pre-port ratios 5.75 and 6.56 and accept 2.2 | 1 | 0 |
| W4 | `APEIRON-E6-W4-T2` the comparator cells G1 already names, so the port is measured against the predecessor and not only against its own resident arm | `examples/table_cost_probe.rs`, the G1 cells | `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls` | the command's wrong-workload/oracle, omitted-incumbent, selected-implementation, interval-crossing and single-failing-cell controls must all report red | 1 | 0 |
| W5 | 48-hour acceptance: the production spill reads meet the existing `<= 1.0` interval criterion against the pinned predecessor on the registered affected workloads; correctness and the semantic mutation/restore controls of T1–T3 are green; extraction and refinement bind the same integrated identity; every remaining G1 failure is reported | no edit; read the registered sources and the W2–W3 artifacts | `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`, then `proof/regen.sh --check` and bounded `cd proof/corpus && lake build` | the strict command's own controls must all report red; a surviving control rejects the campaign, and any interval maximum above `1.0` stays red | 1 | 1 |

## Outstanding obligations

Each item is open, not a completed result:

- **G1/REG-1.** The strict all-row acceptance command
  `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`
  exists and currently exits 1: the receipt of record
  ([receipt](evidence/perf-ledger.md), sha256 `5b94d6cd…` measured on this tree) gives
  `registered=2592 load_bearing=2268 applicable=1450 pass=421 fail=1029 missing=0
  concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false` at
  `selected/incumbent <= 1.0`, its earlier strict section gives
  `applicable=650 pass=133 fail=517 missing=0 unauthorized=0`, and the authority records 387 passing /
  1063 failing applicable pairs; the applicable-pair tally is unsettled and belongs to the
  measurement owner, while the exit status does not. Address-face and concurrency-unmatched pairs fail
  regardless of ratio and the rest are ratio failures, dominated by the spill absent-miss
  distributions (373.85x; 23.59x on the current cut) and the
  resident growth-insert cells (231.9x; 1.53x on the current cut), while the resident direct-index healthy cells are about
  2.7–4.7x when the receipt was cut; the current cut reproduces 1.68–2.63x (E6 resident spike, measured). The
  non-strict diagnostic mode exits 0 and reports the axis `boundary`. `reg1_pass=false`,
  `report_complete=false`; S8/G1 remains OPEN pending joint performance/refinement evidence. The two
  missing rows are the `10^8` identity row and the coverage census. No speed recovery is claimed.
- **Resource/refusal contract (ratified).** The growable table uses a checked
  reservation/allocation-failure result (option (b), transactional): `insert`/`build` return
  `Result`, `Err` leaves the table unchanged and returns no partial table, `ShardFull` and positive-`N`
  `BuildAdmissibleN` are retired from the growable production contract, and the fixed-shard Lean
  modules remain revision-bound historical evidence. Allocator success/failure correspondence is an
  unextracted trust boundary with a simulation statement and must not enter the `Slang.*` empty-cone
  chain. The canonical release/commit remains separately authorized.
- **Resource-handoff extraction (W1 landed; W2 landed; W3/W4 landed; W5 landed; W6-T1 landed; W6-T4 interface repair landed; W6-T5 landed; W6-T2 landed; W7-T2 premise interface landed; W7-SearchCommitBoundary frozen, unchanged under APEIRON-STRATEGY-DECISION-T2/-T3).** The joint
  handoff is landed at
  `src/table.rs` sha256 `a89e9ec743405ac7d80183e431f6e932ba90353bc777f142865f44005261d09a` and
  `src/table/resources.rs` sha256
  `5b04be96131a360d8245ee3061c58dba85059f39a3efeeb0261442a7051368bf`: `CapacityBound`,
  `RegionReservation`, `ReservationReceipt`, `PreparedOutcome<R>` and `PreparedResult<R, V>`,
  `capacity_bound_holds` and `publish_prepared` in `src/table/resources.rs`;
  `observe_receipt`/`materialize_rebuild`/`prepare_rebuild` in `src/table.rs`; `apply_prepared_rebuild`
  consumes `PreparedResult` and returns `Result<(), StorageError>` without touching `self` on
  `Refused`; `R::hold`/`Flat::initialize`/`Region::initialize` run in the adapter after reserve-all;
  `apply_pending_and_commit` keeps its signature. The focused gate `cargo test --locked --lib --test
  table` is green (36 lib + 42 table), both mutation controls are red then green, and `tests/table.rs`
  is unchanged. W2 landed: `word_stored_publish_prepared`, `word_regen_publish_prepared` and
  `word_capacity_bound_holds` are rooted in `proof/extract/lib.rs` (sha256 `1e4fa586…`) and
  `proof/regen.sh` (sha256 `7df851dc…`); `proof/ApeironWord.lean` is regenerated to sha256
  `95766c650fd94da6546d21ef6dff115db22d55cac90b1949f96352217055d1e5` (audit clean, 1708 jobs),
  `proof/regen.sh --check` is fresh, the relocation control default path passes, and the bounded
  `proof/corpus` build (3471 jobs) revalidates the eight existing headlines. W3/W4 landed:
  `proof/corpus/Growable/Resources.lean` (sha256 `db14c8ae…`) defines `CapacityBound_R`,
  `EveryScalarCapacityCheck_R`, `PreparedResources_R`, `RebuildPlanCorrect_R` and
  `ResourceAdapterSimulates`, proves the scalar-check, both publication-carry, publication-transport
  and both conditional reachable-publication headlines, all registered in
  `proof/corpus/AxiomAudit.lean`; the supplied receipt is closed cone, `lake build` 3472 jobs and a
  red-then-green mutation control, not rerun here. The two reachable-publication headlines are
  `proved` only as implications conditional on `ResourceAdapterSimulates`, an explicit unproved named
  trust-boundary premise. `PreparedResources_R`/`RebuildPlanCorrect_R` are therefore Lean predicates
  and premises, not discharged obligations. Whole-table `RepR`/`ReservedRepR`/`PendingCommittedR`
  landed in `proof/corpus/Growable/Representation.lean` (W5; W6-T4 extended it, current sha256
  `4dc7df70…`; bounded gate `lake build Growable.Representation AxiomAudit`), registered in
  `AxiomAudit.lean` and closed cone with `lake build` 3473 jobs and a red-then-green uniqueness
  mutation control. W6-T1 landed: the
  disjoint-write frame lemmas `writeFlat_frame_flat`, `writeFlat_frame_spill`, `writeSpill_frame_flat`,
  `writeSpill_frame_other_shard` and `writeSpill_frame_same_shard` are proved in
  `proof/corpus/Growable/Relocation.lean` and audited in `proof/corpus/AxiomAudit.lean`; the supplied
  W6-T1 receipt records the closed cone, `lake build` 3473 jobs and an index-zero mutation control red
  then green, not rerun here. W6-T4 interface repair landed: `reservedRepR_of_components` and the
  `reservedRepR_{liveSet,unique,coverage,directCover,directory,length}` projections are in
  `proof/corpus/Growable/Representation.lean`; `ReceiptMatchesStorage` now carries the target lanes, flat extent
  `256 × plan.lanes`, the 256-entry storage/receipt directories and per-region tag/payload lengths;
  `TargetGeometryCertificate` carries the target `committedLive`, uniqueness, coverage, addressing,
  `directAfter` with `DirectIndexCorrect`/`DirectCover`, directory consistency and length, exposed by
  the `receiptMatches_*`/`rebuildPlan_target*` projections in `proof/corpus/Growable/Resources.lean` (current sha256
  `31d17cca7bc25707be6d96628ab4730f65e67da8bbdbb741adf368611dacc576`); the bounded gates are green in the
  closed cone (`lake build` 3473 jobs) and the flat-extent `256 × lanes + 1` mutation is red then green.
  W6-T5 landed: `RebuildPlanClauses` binds `cloneHeld`/`addressHeld`/`locateHeld` to the generated
  carrier clone, `address_held state.home` and `table.spill_locate_held … 16#usize`;
  `RebuildPlanCorrect_R` requires the production `keyRep` dictionary;
  `ResourceAdapterSimulates.prepared` carries the same equality; the
  `rebuildPlan_stored_operations_binding`/`rebuildPlan_regen_operations_binding` projections expose the
  bindings for both carriers; closed cone, `lake build` 3473 jobs, and the `locateHeld := Vacant 1`
  mutation is red then green. W6-T2 `rebuild_preserves` landed in `proof/corpus/Growable/Rebuild.lean` (sha256
  `8db24388133248ef29ab483c740e1b0462e10d826bc31cdbdc1d2c1238c19e9a`) as
  `word_stored_rebuild_preserves`/`word_regen_rebuild_preserves`, registered in
`proof/corpus/AxiomAudit.lean`, closed cone, `lake build` 3474 jobs, relocation-write mutation red
then green. The W7-T2 premise interface is landed (`proof/corpus/Growable/Search.lean` sha256 `d349cd48…`,
`proof/corpus/Growable/Resources.lean` sha256 `3708d69c…`; `QueryAddressCorrect`, `FreshLengthBound`,
`GrowableCallbackLaws`, `DecisionSymbolicCorrespondence`, `resourceAdapter_refusal_stutter`), and the
  W7 retry `prepared_rebuild_refines_insertWith` is suspended under `W7-SearchCommitBoundary` and not
  asserted; the freeze and its reopening condition are unchanged under `APEIRON-STRATEGY-DECISION-T2`
  and `-T3`, which add `SpillSummaryShape`, `SpillSummaryRecorded`, `SpillSummaryMissSound` and
  `SpillFingerprintExact` to the boundary ledger as temporary named premises at grade `anchor`.
- **Identity-injectivity integration.** Canonical odd-multiply injectivity at
  `Table/WordIdentity.lean`, the extracted `coord` roots and `TableRefinement/CoordIdentity.lean`
  (`coord_spread_id_injective`) are present and registered for audit; binding that chain to the
  selected source and the growable search/relocation refinements remains open.
- **Unextracted threading boundary.** `std::thread::scope` sits behind an unextracted orchestration
  boundary; Rust thread memory semantics, exclusive-borrow realization, spawn/join failure, panic
  propagation, liveness, allocator behaviour and rustc/LLVM codegen remain open and are not covered
  by the sequential schedule obligation.
- **Extraction freshness.** The [Kani-retirement receipt](evidence/kani-retirement.md) records
  `proof/regen.sh` and `proof/regen.sh --check` at exit 0 (1708 jobs), the corpus build at exit 0
  (3482 jobs), and generated SHA-256
  `92c771510c259bc56c9e437e7d3f3108fa9e2796e6ba41ca2cd826f4127ff74e`. The generated executable
  content is unchanged after stripping the 94 source-annotation changes. This is extraction/build
  freshness, not universal read/search refinement or allocator realization; those remain open under
  [the current bounded plan](epics/read-proof.md). Earlier hash pairs below are historical
  receipts bound to their revisions. A further source change still requires re-running extraction
  on one recorded source/tool identity.
- **Slang empty-cone separation.** The constructive Slang source-refinement chain requires an empty
  axiom set and rejects `propext`, `Classical.choice`, `Quot.sound`, `noncomputable` and
  quotient-based representations; the Rust/Aeneas bridge retains the closed cone. The two cones
  share no declarations.
- **Stable `Sigma`/address interface.** The interface that carries `Sigma` in every executing
  signature and fixes `Coord` addressing per face is open; it must root in `Apeiron.Table.get` and
  `Apeiron.Table.insertWith`.
- **Hyperbolic observation domain.** The hyperbolic regime's admitted observation domain is open.
- **Constructive canonical roots.** Canonical `Apeiron.Table.HMap`, `Apeiron.Table.get`,
  `Apeiron.Table.insertWith`, `Apeiron.Table.build`, `RingCell`, `ringMul` and `ringNormForm` are
  open as constructive roots for every executing face.
- **Source-fold re-root.** Re-rooting the fold onto canonical `Apeiron.Table.insertWith` and
  `Apeiron.Table.build` is open.
- **Audited cross-revision equivalence artifact.** The audited cross-revision equivalence between
  the fold lineage and canonical `ringMul` is missing.

## Consolidation and purge queue

Operator direction: every agreed action from the working conversation is recorded here, in the
execution plan of record, not in a new file. Each removal names its replacement, its consumers and
its reachability check (failure register, consolidation row). No bare `rm`.

- **Documentation purge.** 142 markdown files under `docs/` (138 at HEAD, 4 untracked). Citation
  counts are method-dependent and the old `169`/`79` pair was a stale upper bound: with `rg --json`
  15 files carry no exact repo-path citation and 73 carry none from outside `docs/`; batch 1's
  `rg -o | awk` reading (52) is superseded because it credited stale-worktree paths. No deletion
  decision rests on the count: each deletion needs its per-file reachability receipt at dispatch.
  Keep the governing documents (`AGENTS.md`, `direction.md`, `kernel-boundary.md`, `plan.md`,
  `roadmap.md`, `README.md`, `proof/README.md`) plus `docs/README.md` as the index; a closed epic is
  deleted once its commits supersede it. **A new document must absorb or replace an existing one;
  no net new files.** Batches 1-2 found nothing deletable as they stand: the prerequisites were
  absorption of `extraction-requirements.md` into `proof/README.md`, of four named items of
  `apeiron-intent-escalation.md` into this plan, a commit/fold/loss-receipt decision for the
  untracked `apeiron-intent-*.md` pair, and carrying a `docs/reviews` file's index row with it if
  one is ever deleted. **Discharged** (APEIRON-DOCS-PURGE-W1, receipts in
  `tools/rooting/removals.json`): `extraction-requirements.md` absorbed into `proof/README.md`
  ("Extraction requirements on the source"); the four escalation items and the four handover
  obligations absorbed into the decision-A row above; the `apeiron-intent-*.md` pair retired (both
  absorbed, nothing committed); `docs/reviews/post-compaction-continuation-13.md` retired, superseded
  by `continuation-14.md`, which carries the supersession line. Still open: no epic is deletable —
  the only epic recorded complete (`epics/carrier-refinement.md`) is **retained by `direction.md` and
  this plan's historical-evidence pointers**, so retiring it needs an owner decision on those two
  governing sentences first; the draft, proposed and in-progress epics are not closable; and the
  `docs/call-census/data/` machine output is untracked-pending because its only regeneration input is
  absent (see the receipt).
- **Python gate purge.** `proof/resource` holds 34 `*_gate.py` on disk (30 committed) in 6,513 lines
  (5,622 committed); each re-implements one harness (`flock` lane, `systemd-run` caps, private lake
  package, charon/aeneas extraction, drift check, `lake build`, cone audit, real-source mutation,
  restore, receipt, sha256). Only 18 lines are byte-identical across all 34; identifier-blinding any
  two leaves 116 changed lines. Replace with `tools/gate` (Rust) plus one
  `proof/resource/<module>.gate.toml` per module (root, sources+sha256, wiring, wrapper, headline,
  mutation, markers, lane, budget_s, cone). Waves: freeze the conformance record; driver plus the
  Fwht spec end to end; migrate the remaining modules with retirement receipts; delete every
  `*_gate.py` and reject new ones. The driver must reproduce each gate's green markers and its named
  mutation red before that gate is removed; a mutation that leaves the proof green is exit 1.
  This proceeds separately from the Apeiron–Intent engine; the file count alone does not justify
  rebuilding the harness.
- **Apeiron–Intent, decision A (repair).** Retain the lane; hold candidate acceptance until the
  selection instrument passes rejecting controls. The selection gate is **landed**
  (`crates/intent-cli/tests/ring6_selection.rs`, `df33ee9`/`14c6932`; four D1–D4 controls green in
  0.4 s; it reddens under a Ring-3 admission mutation with a sha-clean restore; not vacuous). The
  earlier "MISSING" line and its memorizer/static-table controls describe a different, future
  object: the oracle Ring 6, decided by Sol as **seeded randomised inputs whose expected outputs
  come from the trusted reference** (secret held-out cases later); the minimal falsifying control
  is a public-table memorizer that fails on inputs confirmed absent from the table while the
  reference passes. 540 s is the candidate-grading budget inside that slice, not the existing
  0.4 s gate command (W2b; task 1 in flight). Rooting: direct linkage supplies no refinement; the
  `e_±` / null-cone / σ→0 reading stays a proposed mapping until carrier, hypotheses and connecting
  theorems exist. Sequence: oracle Ring 6 (W2b) → evaluator and isolation repairs → Apeiron linkage
  and canonical refinement → calibration and candidate screens → one accepted improvement cycle.
  Four obligations carried here from the retired `docs/apeiron-intent-escalation.md` and the retired
  `docs/apeiron-intent-rsii-handover.md` (both absorbed; nothing below is new work, and each item was
  confirmed absent from this plan by `rg -c` before absorption):
  1. **Forgery closed, selection open.** The relay split plus the pure `admit_frame` Ring-0 core moved
     `fullforge` **15/15 → 0/15** (reference 15/15, `oracle_integrity` 38/38); `MAX-PAYLOAD-FORGE` is
     still 15/15 because a candidate that knows the public case table (`oracle_cases.rs`) answers the
     outstanding invocation. This is the failure-register row *Measurement instrument defects*: `S` has
     no control separating implementing the task from memorising it. The numbers are receipts in the
     `intent` repository and carry that repository's grade; this plan states no grade for them and
     re-runs no gate.
  2. **Intent-local harness defects (4), each a retained sequence, not a fix.** The path jail refuses
     its own work dir (`edit path="/work/solution.py"` → "escapes the session work dir"); `/tmp` is a
     fresh tmpfs per command, so model scratch vanishes; no wall clock on a tool call (one `shell` ran
     271 s); compaction drops tool outputs. This is the harness-failure coverage the panel must expose.
  3. **Preserve list.** The campaign dataset, the forgery controls, the frozen proofs and the current
     harness are preserved for comparison. The dataset
     `intent/artifacts/space-bunny-evolution/sb-2026-10-01/` is **untracked in `intent`**: commit it
     there before any clean-up that would drop it. Nothing in this repository is its owner.
  4. **Cap-50 incumbent incomplete.** The cap-32 baseline is S ≈ 0.9867 over 10 evolve-split tasks (0
     missing, 3/10 `ceiling`); the cap-50 incumbent is **8/10, halted on owner instruction, no aggregate
     S**, so it cannot be the calibration incumbent until it is re-run to an aggregate. Calibration
     (W5) records `delta_1` per class and is what makes `selection_ready` true; without it W5–W7 do not
     open.
- **Outstanding returns (5).** RowDotAdmission, LmHead, RegenRelocateEntry (never returned);
  extraction-root repair and `regen_relocate_all` (returned empty — a no-artifact return does not
  close the task). All five agents are live; their files are owned; no second writer.
- **Template for the coming dispatch.** Every task is ≤3 files, one gate, one control, one Arc
  block; disjoint files, one owner per file; per-gate wall-clock budget; the lead re-runs each gate
  before commit. A task that cannot name its gate is not dispatchable. Every task prompt carries the
  escalation demand: run `gate hold` before any write; stop and hand back after three materially
  different approaches or five patch-and-rebuild cycles on one proof; escalate immediately on budget
  expiry, a contract/premise/grade/trust-boundary blocker, a missing tool, a file owned by another
  task, or an undeclared scope; and state the exact failing goal, the approaches tried, what is on
  disk and the one decision requested. Never commit, never decide scope. The ladder is
  `../AGENTS.md` §Escalation ladder.

- **Extraction-root repair (two blockers cleared; one boundary decision open).**
  `proof/extract/lib.rs` builds; the `dequant_f32_row` move and both `f32_core.rs` loop-shape
  reshapes landed (`8fc50a5`, `3a69a4f`, `4642760`). The pinned-Aeneas internal assert that aborted
  every run was a redundant pass in the in-repo patch; removing that hunk and rebuilding the tool
  from the pinned bases (`install-patches.sh --check` exit 0, build 38 s) clears it, proven by the
  control where the pre-existing binary still asserts. The pipeline now proceeds into translation
  and stops on 20 binary32 functions rejected with `Improperly typed constant value`
  (`InterpExpressions.ml:117` has no float case) — the boundary decision is with Sol and is answered:
  route (c), an explicit opaque/foreign boundary with a translatable interface, a named
  boundary-axiom allowlist in `AxiomAudit`, and caller claims only relative to that boundary; if a
  strictly axiom-free cone is required the caller chain is excluded or abstracted instead. The f32
  chain hangs from the single root `crate::word_attn_head`, and a probe to settle the mechanism runs
  as APEIRON-F32BOUNDARY-T1. Until it lands `proof/ApeironWord.lean` cannot regenerate, which also
  holds the tag and attn-reshape drift. The T1 probe found the mechanism and a tool regression: with
  Charon `--opaque` patterns on `crate::attn::f32_core` and `crate::mixer::f32_carrier` the OLD tool
  clears the assert and translates `core::array::repeat`, while the REBUILT one refuses that
  signature at `Translate.ml:390` — so the rebuild regresses on a construct the previous build
  handled, and the durable tool identity is open. T1 also showed the 20-error count is a front slice
  (32 errors over 29 array sites under `-print-llbc` without `-warnings-as-errors`); enumerating
  before reading a rejection count as the population is now required. The boundary interface is
  float-free (the Q32.32 word edge plus the `Carrier` enum), and the completion probe runs as
  APEIRON-F32BOUNDARY-T2. T2 settled the mechanism and the emitted artifact: with the old tool and
  four Charon `--opaque` patterns — `crate::attn::f32_core`, `crate::mixer::f32_carrier`,
  `crate::attn::_::attn_head_f32`, `crate::attn::_::with_carrier` (the `_` is a `Glob` at impl
  position; inherent impls never match a brace pattern) — Aeneas emits `ApeironWord.lean`
  reproducibly (byte-identical over two runs, 24 864 lines). The run then fails `regen.sh`'s axiom
  `audit()` on 16 axioms: the 3 intended boundary items, plus 13 pre-existing model axioms the
  drifted source now needs (`std::env`, `OnceLock`, `OsStr`, `I128`, `String` deref, `Str` equality),
  and `core.f32.F32.is_finite` still appears inside `word_kv_append`'s loop bodies because
  `KvState.occupied` is `Vec<Vec<F32>>` since the carrier wave — so the boundary is not closed by
  the flags alone. The audit's 12 non-f32 axioms are attributed to three Rust sites: `mixer.rs:307-313`
  (`env::var` behind a `OnceLock`, 11 axioms, reached only from `word_kv_state_new` — production
  constructs through `with_carrier`), `attn.rs:231/266/283` (`i128::from(u64)` in `recip_u64` and
  `weight_at_recip`), and `attn.rs:952/960` (`f32::is_finite` as the `KvStorage::F32` arm's only
  overflow guard). The first two are being applied mechanically (the wrapper calls `with_carrier(..,
  Carrier::Q32)`; the model gains `I128.Insts.CoreConvertFromU64.from` in parallel with its existing
  `From` entries). Sol fixed the last two: `core.f32.F32.is_finite` is admitted as the fourth
  boundary item — the pinned model's `F32` is opaque, so the F32 arm's overflow guard is not proved
  and claims about `append` for F32 storage are relative to that boundary — and the corpus scope is
  the production constructor, covering `KvState::with_carrier` while explicitly not claiming
  coverage of `KvState::new` or `carrier_from_env`, whose environment-dependent behavior is outside
  the pinned model. The scratch pipeline now exits 0 (68.8 s): `proof/boundary-axioms.txt` names the
  four admitted items with reasons, `proof/audit-extraction.py` rejects `sorry`/`admit`, unlisted
  axioms and allowlist entries with no matching declaration (eight controls, one of which caught a
  parser defect in its own trailing-newline handling), `regen.sh`'s `audit()` calls it, the README
  carries the boundary statement, and `AxiomAudit/Core.lean`'s closed cone gains the four names.
  Landing conditions, in order: the exact axiom name form `auditClosedAxioms` must use
  (`Lean.collectAxioms` returns module-prefixed names; one `#print axioms` probe in the warm corpus
  settles it), the four `--opaque` flags must land with the corpus, and the tool identity is open —
  the Aeneas that translates (`33d0ce1a`) has no recoverable source on this box while the rebuilt
  one (`2a0d7827`) regresses on `core::array::repeat`, so landing needs either a rebuild fix or an
  explicitly recorded tool identity. A cold `proof/` `lake build` cannot finish in the local
  budget; the barrier must use the repo's warm `proof/corpus/.lake`. Also open:
  `proof/regen.sh`'s `SOURCE_FILES` digest list has no entry for any newly mirrored extraction
  source (`mixer.rs`, `step.rs`, `chart/`, `kern/`, `loader.rs`, `trace.rs`, `numeric/fixed_ln.rs`,
  and the `mixer/` and `step/` children) — obligation 8 binds the evidence to an under-specified
  source set until that list covers them. One-owner-per-file was violated in this
  checkout: a second writer edited `proof/extract/lib.rs` at 21:18 while the repair owned it.
- **Working-tree hygiene.** The checkout carries root-level junk that is not part of any task:
  `a.js`, `test_edition2`, `test_closure_pointer`, `libtest_closure_capture.rlib`, `proof/timeout`,
  `.held-control/`, `.held-scratch/`, `.licenses/`, `VP_VULKANINFO_NVIDIA_GeForce_RTX_5080_*.json`,
  plus uncommitted modifications to generated `proof/ApeironWord.lean` and `proof/corpus/`
  (`AxiomAudit*.lean`, `Policy/Core.lean`, `lakefile.lean`). Classify each as tracked, ignored or
  removed before the dispatch; a build product is never tracked (failure register, consolidation
  row), and a generated proof file is regenerated, not hand-edited. Scratch reclaimed:
  `oc-tmp` 154.7G → 77G, `/tmp/opencode` 2.45G → 1.83G, `/home` free 501G → 551G; 2 stale worktree
  records pruned and 3 clean, merged worktrees removed (`task/w5-integ`, `nightshift`,
  `ns/affine-head-signed-mul-r30`). The owner authorized abandonment of the remainder: 31 worktrees
  were captured first (per-worktree tracked patch, untracked tar and an index under
  `/tmp/opencode/abandoned-worktrees`, 640K) and force-removed, leaving `main` as the only worktree;
  33 merged branches were deleted and `one-allocation` kept (unmerged). `oc-tmp` 154.7G → 18G and
  `/home` free 501G → 569G.

## Apeiron–Intent: the repeatable improvement engine

Keep the direction. The goal is to turn the mechanism that produced the 2× llama.cpp result into a
repeatable improvement engine, with tighter connections between evidence, proposals and execution.

### Three immediate corrections

1. **Rebase specification.** `intent/docs/EVOLVER.md:172` (D7) states that a drift-triggered rebase
   restarts from a fresh baseline but specifies `M ← max(S_fresh, M_t)`: when the fresh score is below
   the old maximum, the blocking maximum is unchanged and the rebase cannot lower the bar. Require an
   explicit evaluation epoch and its new baseline, and retain the previous epoch's evidence.
2. **Historical cost is not an upper bound.** `intent/docs/EVOLVER.md:64` (D2), implemented at
   `intent/crates/intent-core/src/selection.rs:133`, takes the minimum of historical observed cost
   and a budget-derived bound. A past maximum does not bound a future candidate. Use an enforced cap
   or a justified resource bound for admission; historical estimates guide search only.
3. **Connect the optimisation to its target.** `docs/epics/apeiron-intent-loop.md` states the scalar
   `row_dot_fixed` motivation from a GPU MMQ throughput gap without naming the connecting execution
   path. Row-dot is a reasonable admission pilot; a GPU improvement claim needs the actual
   kernel/caller connection and the command whose output records it. Resolve the competing proposed
   transformations before dispatch.

### Documentation becomes executable — option C, enforced in Rust

Chosen: **C — indexed, schema-checked receipts.** Rejected: A's deletion rule (citation from a
governing document does not establish usefulness, and absence of a citation does not authorize
destroying evidence) and B as the primary architecture (generating inventories cannot derive
architectural policy, admissible domains or trust boundaries; generated tables may support C later).

**Layout.** The governing set stays hand-written. `docs/contracts/artifacts.toml` is the single dispatch
manifest: governing paths, approved plans, protected contracts, ledger locations, schemas and
required checks. `artifacts/<kind>/ledger.jsonl` holds append-only receipt records;
`artifacts/<kind>/<id>/` holds raw payloads identified by digest. Existing epic plans and
machine-parsed contracts stay separately indexed. Workers produce structured returns; creating
another Markdown document requires an explicit documentation task. Each receipt records
task/owner, source/tool identities, the scoped claim and grade, command, exit status, elapsed
budget, control and restoration results, and resolvable payload references; a correction appends a
superseding record.

**Enforcement is Rust.** One gate runner: `tools/gate docs check --controls`. Pure checking cores
(schema validity, reference resolution, identity matching, claim/evidence compatibility, admission)
are extracted through Charon/Aeneas and refined in Lean; a tested adapter owns filesystem, hashing,
subprocess, locks and budgets, and states that boundary. Per-repository specifications preserve each
repository's contracts, grades and axiom policy; shared machinery does not require a shared ontology.
A submission is refused for missing registration, undeclared documents, broken references, identity
mismatches or incomplete required evidence.

**Migration.** Existing Python behaviour supplies comparison cases, including the known defects the
Rust implementation must reject. Each checker is deleted once its Rust replacement passes the
required positive, rejection and restoration controls and every consumer invokes Rust; Python leaves
the final execution path entirely. The same retirement applies to workload interpreters and campaign
helpers, sequenced behind the acceptance machinery. Delete the duplicated gate scripts first (after
replacement acceptance), then the redundant per-run Markdown wrappers (after receipt migration);
orphan status alone is insufficient.

**No gate resolves a conventional tool through `PATH`.** `~/.local/bin/grep` is a symlink to `rg`,
where `-r` means replace rather than recurse, and `~/.local/bin/find` is a findutils 4.4.2 binary
that exits 0 and prints nothing on a recursive scan of a populated directory. Both precede the
system tools on this host, so a bare `grep -r` or a `find`-based absence check is vacuously true;
two claims in this programme were made false by them before the cause was found, and the false
"thirteen absent symbols" reading was published from one. The shims are **not patched** — that is
one machine's environment, not this programme's contract. The durable fix is that no gate, enforcer
or receipt depends on a `PATH`-resolved conventional tool, and the Rust runner is the vehicle, with
Python leaving the execution path entirely on the migration schedule above. Until the runner carries
it, a check refuses a bare `find` or `grep` invocation in a tracked script.

**Manifest lifecycle (settled).** Reserve task → submit provisional output → independently verify →
seal accepted artifact. Before dispatch the manifest freezes ownership, inputs, required
dependencies, gates and budgets. After execution the trusted adapter computes the output identity. A
verifier other than the worker checks the named gate. Only the lead seals. A worker cannot change its
own admission criteria or mark a submission accepted; `admit` decides a provisional submission
against a frozen manifest, and sealing is the lead's act.

**First bounded task — done, with the two probe counterexamples closed.** `tools/gate/core.rs`
validates the manifest before admitting anything: a duplicate id refuses (order-independent) and a
registered requirement that does not resolve refuses (manifest closure), so a required dependency
cannot disappear. Admission then refuses an unregistered id, an identity mismatch and a dangling
reference, reporting every cause. Gate: `cargo test --manifest-path tools/gate/Cargo.toml --offline`
— 7 controls green. Mutation controls on throwaway copies, each attributable to its named control:
the duplicate check, the registered-requirement check, the identity check, the reference check, the
unregistered-id report. Both counterexamples reproduced against the previous five-test version, which
passed while admitting them; those tests were too weak, and the probe cases are now the controls.

**Lane budget.** The host has about 44 GiB available; ten 12 GiB Lean scopes would reserve 120 GiB.
Run at most three simultaneous heavy Lean checks, with isolated caches and exclusive ownership of
shared build products; other agents continue writing and analysing.

**Migration pilot boundary.** The pilot is RowDot under the standard closed cone, not Fwht: the Fwht
audit permits five additional width axioms, so it cannot establish the standard-cone guarantee.
Conformance must preserve required guarantees, including the rejection of known defects.

Next, separately scoped: extract and refine the real core body through Charon/Aeneas, then the
adapter and `docs/contracts/artifacts.toml`, then mandatory-path wiring. This gate-first task does not close
the governing-coverage and barrier-wiring obligations. Direction: drain the four returns, settle the
reservation/sealing contract, assign the small corrections, then the bounded-task campaign; routine
proof and Rust tasks continue without recurring architectural review.

### Research basis (plan changes)

- **Self-Modifying Lean Proof Agents**, §3.2 — a fixed runtime verifies proofs and claimed
  intermediate results. Intent may improve proof retrieval, decomposition and repair; statement
  identity, extraction checks, axiom policy and acceptance stay outside the candidate's authority.
- **GEPA**, §§3, 5.1 — reflection consumes execution traces. Feed proposals the diagnostic, compiler
  output, counterexample or profiling receipt plus the relevant source/spec excerpt; documentation
  retrieval is part of the loop.
- **Darwin Gödel Machine**, §3, Appendix H — an archive preserves useful branches; its
  objective-hacking example scores perfectly by removing detection logging. Separate research-archive
  eligibility from incumbent promotion, preserve unsuccessful branches and failure evidence, and
  validate the measurement path independently.
- **AlphaEvolve**, §§2.4–2.6 — cheap screening precedes expensive evaluation; parallel proposal and
  cheap rejection; serialize shared proof lanes and quiet measurements; every promoted candidate
  still clears the complete acceptance barrier.
- **STOP** Algorithm 1; **SICA** §3 — evaluate an improver by the downstream improvements it
  produces. Measure accepted verified gains per budget, time to acceptance, and transfer to fresh
  tasks; include a second cycle driven by the first cycle's evidence.

### Ecosystem ownership and the transfer test

Preserve the ownership at `docs/direction.md:170`: Apeiron supplies reusable primitives and
refinements; Intent improves how work is proposed and executed; Song independently realises
corresponding primitives; Penglai and Whispering Tides supply workloads. Choose one accepted
primitive for transfer into a second consumer; that tests whether the improvement is reusable.

### Sequencing

Let the outstanding returns finish; integrate their scoped evidence; resolve the three corrections;
close the selection-measurement gap (Ring 6); run one complete Apeiron–Intent cycle, then a second
cycle driven by the first cycle's evidence.

## Resume queue (paused; not resumed until the owner's go)

The mechanism binds only agents that run the `gate hold` preflight. Agents already executing when the
marker appeared continue until they check it. Checked against `~/.local/share/opencode/opencode.db`.

| task | session | state | note |
|---|---|---|---|
| Row-dot admission certificate | `ses_f08c92f1fffeMT3WSkyUc83AtG` | live at 22:47 | fixing `RowDotAdmission.lean` statement index validity (`:666-677`) and `Result.ok_injective`; read the HOLD and ran the preflight |
| Cost: lm_head | `ses_f08f5a7c1ffeTXDugldOACo3U6` | **stopped by the lead on the cycle cap** | 7 materially different approaches. Blocker: `proof/resource/LmHeadResource.lean:557` in `row_body_visit`, the dequantizer-`Ok` width-matches branch — the `let (r, row_buf3) := (Result.Ok u, row_buf3)` monadic pattern bind is not zeta-reduced by `simp only [bind_tc_ok]`, so `if Vec.len row_buf3 = n_in` is unreachable to `rw`. Also open: a redundant `simp only [bind_tc_ok]`, a `dot_body_exit_end` hypothesis flip, and a `dotVisits` signature in `lm_head_weight_positions`. `LmHeadSource.lean` and the gate are fine (`--prepare` succeeds); sections 0–3, `VisitTrace`, `matvec_*` and the work-measure theorems build. |
| Cost: regen_relocate_entry | `ses_f08f53f27ffeB1Sao5aOgJs92M` | idle 21:15:35, succeeded | `RegenRelocateEntry{Source,Resource}.lean` + gate on disk; no handback received |
| Cost: regen_relocate_all | `ses_f08f53f25ffeJsd9gyq11A644K` | idle 22:11:53, succeeded | `RegenRelocateAllSource.lean` + gate on disk, no `Resource.lean`; matches its empty handback |
| Resource cost: exact_dot | `ses_f091bf9deffelBQEv5RnVaEXx3` | stopped by the HOLD | real extraction and guard/exit/witness/refusal proofs in scratch `/home/socol/oc-tmp/ed-probe`, zero `axiom`/`sorry`; blocked on the loop work-measure rewrite (`bind` over a pattern binder); refused to write because its task is new, not a return |
| Resource cost: matvec | `ses_f091bf9dcffeQRlG25` | live at 19:55, no activity since | same batch |

Resuming any of these requires the owner's go; the `HOLD` marker is releasable only by the owner.

## Evidence labels

A claim is `proved` only when a Lean kernel check passes for a named theorem with the extraction from
the real source and a closed axiom cone; `tested` when a `cargo test` executes the build path and
its boundary cases; `measured` when a named command ran and its output was read. A model or anchor
is a mapping, not evidence of correctness.

## Historical evidence pointers

- [proof/README.md](../proof/README.md) — extraction, bridge, pins, axiom cone and controls.
- [archive/autonomous-run-log.md](archive/autonomous-run-log.md) — run narrative and decisions.
- The predecessor documentation at Git `1403767e671c81737525702946a3eb1eae4f0b0b`:
  [storage plan](epics/storage-refinement.md), [carrier plan](epics/carrier-refinement.md) and
  [corpus release pin plan](epics/corpus-release-pin.md).

These are pointed to, not copied. Their contents bind to the revisions they record.

## Stop points and evidence

- No production representation migration until the same-run matrix identifies the
  performance-bearing implementation/mechanism. No deletion of performance harnesses without a
  replacement gate retaining their useful tested surfaces, identities and controls.
- Extraction convenience does not select a slower replacement. Historical CountTable/Map source
  can supply measurement inputs, not accepted production without its own canonical refinement.
- A documentation pass does not require the full software barrier. A source, extraction, package or
  pin change needs a separate task and its own local barrier.
- Each task owns at most three files and one gate with one negative control; the next wave opens
  only when every child and shared gate is green. A missing gate is named as an artifact to
  create, never replaced by a command that does not exist.
- Report per slice: changed files, the actual commands and exit codes, what is proved versus tested
  versus open, and the next bounded task. A successful build, a sampled test and a universal proof
  are different evidence.
- Preserve unrelated dirty work. Do not commit, push, post externally, or modify another active
  workstream unless the user requests it.
