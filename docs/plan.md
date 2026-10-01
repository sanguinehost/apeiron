# Apeiron execution plan

Kani is retired by user direction. The [retirement contract](epics/kani-retirement.md) supersedes
all older Kani gates and dispatch requirements in this handoff and linked plans. Historical
receipts remain historical. Current acceptance uses Rust tests/clippy/fmt, real-source mutation
controls, extraction freshness and the bounded Lean builds/audits; open proof premises remain open.
Current dispatch is [the bounded read/search proof plan](epics/read-proof.md); it supersedes the
older execution tables below without erasing their revision-bound receipts or obligations.

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
`python3 tools/rooting/check.py --self-test`; `python3 tools/rooting/check.py --check` is red only
on the binned routes until their retirement.

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
coordinates give different results. G1/REG-1 remains OPEN on the fresh strict full-profile run
([receipt](evidence/perf-ledger.md), sha256 `e164f808…`, superseding the stale
figures): the HEADLINE is `registered=2592 load_bearing=1458 applicable=650 pass=78 fail=572
missing=0 concurrency_unmatched=0 address_face_unmatched=6 full_acceptance=false`, exit 1. The
pathological cells are the spill absent-miss distributions (worst `apeiron_table`/`hashbrown_fxhash`
`occ16-seeded-random-absent-miss-replacement`, median ratio 373.8538, interval max 375.1818) and the
resident growth-insert cells (worst `apeiron_table`/`hashtrinity_table`
`occ16-direct-index-resident-growth-insert-replacement`, median ratio 231.8779); the resident
direct-index healthy cells are not pathological (present-hit 3.3860x and absent-miss 4.6780x versus
serial `hashtrinity_table`, about 2.7–4.7x across the load-bearing band). No speed recovery is
claimed. That receipt binds `examples/reg1/acceptance.rs` sha256 `f0f9aff0…` and `src/table.rs`
sha256 `a89e9ec7…`; the current tree carries the T1 registry (`examples/reg1/acceptance.rs` sha256
`d2f8c034…`) and a post-receipt spill/growth fix (`src/table.rs` sha256 `79621926…`), so the receipt
is the last strict full-profile result but not a measurement of the current identity. The
re-anchored performance workstream in the barrier section continues: comparison coverage is repaired
(T1) and the combined `get_each` arm is now `concurrency_unmatched`; serial direct-path overhead and
spatial recovery on disjoint paths remain — with `W7-SearchCommitBoundary` frozen.

The per-entry relocation step is extracted and proved for both carriers and both destinations:
`word_stored_relocate_entry` and `word_regen_relocate_entry` are rooted in the fresh
`proof/ApeironWord.lean`, and `proof/corpus/Growable/Relocation.lean` proves
`word_stored_relocate_entry_flat_preserves`, `word_regen_relocate_entry_flat_preserves`,
`word_stored_relocate_entry_spill_preserves` and `word_regen_relocate_entry_spill_preserves` under the
closed cone, with `nested_region_set_readback`/`nested_region_payload_set_readback` carrying the
nested spill `Vec::set` readback, all registered in `proof/corpus/AxiomAudit.lean`; the spill-tag
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
(needs `SearchDecisionCorrect` + `CallbackLaws`), suspended under `W7-SearchCommitBoundary` because the
reachable publication witness is conditional on the explicit unproved `ResourceAdapterSimulates`
premise; the W6-T1 disjoint-write frames are landed. W6-T4 interface repair landed: `proof/corpus/Growable/Representation.lean` (sha256
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
`W7-SearchCommitBoundary` and not asserted.
Whole-table
`RepR`/`ReservedRepR`/`PendingCommittedR`
landed in `proof/corpus/Growable/Representation.lean` (W5); payload-initialization placement remains
open. Kani is retired and supplies no current coverage requirement.

### Joint prepared-resource boundary (Astra verdict; Sol contract W0; W1 landed; W2 landed; W3 landed; W4 landed; W5 landed; W6-T1 landed; W6-T4 interface repair landed; W6-T5 landed; W6-T2 landed; W7-T2 premise interface landed; W7-SearchCommitBoundary frozen)

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
with the commit equality, `SearchDecisionCorrect` and `CallbackLaws` to canonical
`Apeiron.Table.insertWith`; it is not asserted here.

### Supplied measurement receipts (REG-1, Kani and historical figures not rerun; the relocation control was rerun, receipt `docs/evidence/relocation-controls.md`)

| Evidence / grade | Result and limit |
|---|---|
| REG-1 runtime, `measured` (fresh receipt `docs/evidence/perf-ledger.md`, sha256 `e164f808…`, supersedes the stale figures) | Fresh strict full-profile run on `src/table.rs` sha256 `a89e9ec7…`: HEADLINE `registered=2592 load_bearing=1458 applicable=650 pass=78 fail=572 missing=0 concurrency_unmatched=0 address_face_unmatched=6 full_acceptance=false`, exit 1. Pathological: spill absent-miss 373.85x (interval max 375.18) and resident growth-insert 231.9x; resident direct-index healthy about 2.7–4.7x behind the serial load-bearing incumbents. Sole full16 workload is an extreme boundary, not a representative aggregate. |
| Spill/growth micro, `measured` (diagnostic; not a pair-gate result) | `cargo run --release --locked --example micro_spill` on the post-receipt spill/growth fix, same 4096-op schedules: `occ16-seeded-random-absent-miss-replacement` median `1314468`/`1328764` → `78758` ns (`320.92`/`324.41` → `19.23` ns/op) and `occ16-direct-index-resident-growth-insert-replacement` median `7294013`/`6805921` → `93218` ns (`1780.76`/`1661.60` → `22.76` ns/op); before logs `/tmp/opencode/diag.out`/`diag2.out`, after `/tmp/opencode/micro-final.out`. The two cells remain the strict receipt's worst; no post-fix full-profile receipt exists and the pair-gate grade is unchanged. |
| Cached-placement experiment, reported `measured` | Candidate/hashtrinity 2.65–2.69 on the measured boundary; rejected, benchmark-only, not production or proof-covered. |
| Historical spatial reproduction, archival pending identity registration | `71942db` parallel CountTable: historical 3.16 ms versus packed hashtrinity 10.27 ms; supplied temporary-worktree rerun 2.494 ms versus 8.976 ms (~3.6x win), i9-14900KF / rustc 1.97 nightly. No CountTable refinement is established. |
| Kani verification latency, reported bounded-check result and `measured` elapsed time | `cargo kani`: 448.58 s, 62/62 green, one unreachable; `the_norm_factors_into_the_channels` about 94.23 s, largest. This is verification latency, not table runtime or a Lean proof. |
| Extraction freshness, landed at the recorded revision | `proof/regen.sh --check` fresh on `src/table.rs` sha256 `2a60a19dae103fbd6981824c11ca7b436ef344071cd979905746c9ec531d5856`; `proof/ApeironWord.lean` sha256 `c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2`, `lake build` 1708 jobs, generated audit clean, all six `relocate_entry`/`relocate_all`/`apply_pending_and_commit` roots present for both carriers. The fresh standalone control receipt `docs/evidence/relocation-controls.md` records the same baseline green and restored freshness green (exit 0). The earlier red was source-line comment drift, not semantic drift. |

No production `src/` changes belong to the supplied measurement session. Preserve uncommitted
`examples/reg1/` evidence and unrelated spinor work. The existing performance entry point is
`cargo run --release --locked --example reg1 -- --check performance --profile full --controls`;
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
axis `boundary`. The command currently exits 1: the fresh strict full-profile run
([receipt](evidence/perf-ledger.md), sha256 `e164f808…`) gives
`registered=2592 load_bearing=1458 applicable=650 pass=78 fail=572 missing=0
concurrency_unmatched=0 address_face_unmatched=6 full_acceptance=false`; the 6 address-face
unmatched pairs fail regardless of ratio and the other 566 are ratio failures, dominated by the
spill absent-miss distributions (373.85x) and the resident growth-insert cells (231.9x), while the
resident direct-index healthy cells are about 2.7–4.7x behind the serial load-bearing incumbents;
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
load-bearing on the opaque-key exact map face. Applicability is registered per cell/incumbent pair before
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
exercises the canonical-rooting rejection paths; `--check` is red only on the binned routes until
their retirement.

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

**Performance workstream order.** Consume the fresh full-profile receipt before changing timed
source or harness. Then, in order: (1) comparison coverage repaired — `APEIRON-REANCHOR-W1-T1`
made the five named hashers (`xxHash` XXH3/XXH64, `wyhash`, `ahash`, `fnv`) load-bearing in
`examples/reg1/acceptance.rs` with rejecting omission/demotion controls and recorded the combined
`get_each` arm as `concurrency_unmatched`; (2) size-only factoring of `src/table.rs` before hot-path
changes; (3) reduce serial direct-path overhead; (4) restore spatial partition/build/probe recovery
on disjoint paths; (5) publish a matched before/after full-profile receipt. The
[roadmap dispatch table](roadmap.md#performance-re-anchor-and-bounded-dispatch) owns
the task IDs, files and gates. No aggregate, std-only win, historical envelope, reduced matrix or
trusted-boundary grade closes G1.

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

1. **The immediate task:** reduce the measured serial direct-path overhead, then restore the spatial
   partition/build/probe recovery, per the performance workstream order above (comparison coverage is
   repaired under T1). The per-entry relocation roots for both carriers are bound and their flat
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

## Outstanding obligations

Each item is open, not a completed result:

- **G1/REG-1.** The strict all-row acceptance command
  `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`
  exists and currently exits 1: the fresh strict full-profile run
  ([receipt](evidence/perf-ledger.md), sha256 `e164f808…`) gives
  `registered=2592 load_bearing=1458 applicable=650 pass=78 fail=572 missing=0
  concurrency_unmatched=0 address_face_unmatched=6 full_acceptance=false` at
  `selected/incumbent <= 1.0`; the 6 address-face unmatched pairs fail regardless of ratio and the
  other 566 are ratio failures, dominated by the spill absent-miss distributions (373.85x) and the
  resident growth-insert cells (231.9x), while the resident direct-index healthy cells are about
  2.7–4.7x behind the serial load-bearing incumbents. The
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
- **Resource-handoff extraction (W1 landed; W2 landed; W3/W4 landed; W5 landed; W6-T1 landed; W6-T4 interface repair landed; W6-T5 landed; W6-T2 landed; W7-T2 premise interface landed; W7-SearchCommitBoundary frozen).** The joint
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
asserted.
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

- **Documentation purge.** Census: 169 markdown files under `docs/`, 48,481 lines — 40 epics, 52
  evidence, 11 contracts, 4 reviews, 4 archive, 10 top-level. 79 are not cited by filename anywhere
  outside `docs/` (upper bound: the test catches filename citations only). Keep the governing six
  (`AGENTS.md`, `direction.md`, `kernel-boundary.md`, `plan.md`, `roadmap.md`, `README.md`,
  `proof/README.md`) plus `docs/README.md` as the index; delete the remainder after a per-file
  reachability check. A closed epic is deleted once its commits supersede it. **A new document must
  absorb or replace an existing one; no net new files.** Deletions land with their reachability
  receipt.
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
  selection instrument passes rejecting controls. Ring 6 gate MISSING:
  `crates/intent-cli/tests/ring6_selection.rs`, command
  `cargo test -p intent-cli --locked --test ring6_selection -- --test-threads=1`, 540 s budget,
  evolve-split inputs with candidates never receiving expected answers, held-out
  reporting/checkpoint policy preserved. Controls: reference passes; public-table memorizer fails;
  wrong-result and frame-forgery fail; a static-table-grading mutation makes memorizer-rejection fail
  and restoration passes. Rooting: direct linkage supplies no refinement; the `e_±` / null-cone /
  σ→0 reading stays a proposed mapping until carrier, hypotheses and connecting theorems exist.
  Sequence: Ring 6 spec/gate → evaluator and isolation repairs → Apeiron linkage and canonical
  refinement → calibration and candidate screens → one accepted improvement cycle.
- **Outstanding returns (5).** RowDotAdmission, LmHead, RegenRelocateEntry (never returned);
  extraction-root repair and `regen_relocate_all` (returned empty — a no-artifact return does not
  close the task). All five agents are live; their files are owned; no second writer.
- **Template for the coming dispatch.** Every task is ≤3 files, one gate, one control, one Arc
  block; disjoint files, one owner per file; per-gate wall-clock budget; the lead re-runs each gate
  before commit. A task that cannot name its gate is not dispatchable.

- **Extraction-root repair (returned; blocked on one `src/` move).** `proof/extract/lib.rs` is down
  from 31 errors to 1: `src/engine/loader.rs:24 use super::dequant_f32_row`, which is defined only
  inline at `src/engine/mod.rs:236-259` and cannot be mirrored as a fragment. The repair's verified
  fix is three lines: move `dequant_f32_row` verbatim from `src/engine/mod.rs:236-259` into
  `src/engine/f32_convert.rs` (already in `regen.sh`'s `SOURCE_FILES` and already the
  `#[path]`-addressable-children pattern `src/table.rs:30-43` uses), and change the import to
  `use super::f32_convert::dequant_f32_row`. `cargo build --lib --manifest-path
  proof/extract/Cargo.toml` then exits 0 on a throwaway copy. Gate 2 (`proof/regen.sh --check`)
  stays red for three pre-existing reasons outside that move: the pinned rustc resolves `std`/`core`
  from an incompatible `/home/socol/.cache/miri` sysroot (E0514, 626 cascading errors, reproduced on
  committed `HEAD`); `proof/ApeironWord.lean` is stale against the post-F32 `KvState` and dirty in
  the tree; and the pinned Aeneas refuses `KvState::append` with "Returns inside of nested loops are
  not supported yet". Also open: `proof/regen.sh`'s `SOURCE_FILES` digest list has no entry for any
  newly mirrored extraction source (`mixer.rs`, `step.rs`, `chart/`, `kern/`, `loader.rs`, `trace.rs`,
  `numeric/fixed_ln.rs`, and the `mixer/` and `step/` children) — obligation 8 binds the evidence to
  an under-specified source set until that list covers them. One-owner-per-file was violated in this
  checkout: a second writer edited `proof/extract/lib.rs` at 21:18 while the repair owned it.
- **Working-tree hygiene.** The checkout carries root-level junk that is not part of any task:
  `a.js`, `test_edition2`, `test_closure_pointer`, `libtest_closure_capture.rlib`, `proof/timeout`,
  `.held-control/`, `.held-scratch/`, `.licenses/`, `VP_VULKANINFO_NVIDIA_GeForce_RTX_5080_*.json`,
  plus uncommitted modifications to generated `proof/ApeironWord.lean` and `proof/corpus/`
  (`AxiomAudit*.lean`, `Policy/Core.lean`, `lakefile.lean`). Classify each as tracked, ignored or
  removed before the dispatch; a build product is never tracked (failure register, consolidation
  row), and a generated proof file is regenerated, not hand-edited.

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

**Layout.** The governing set stays hand-written. `docs/artifacts.toml` is the single dispatch
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

**First bounded task — done.** `tools/gate/core.rs` is the pure admission core (`admit` over a
`Manifest` and a `Submitted`): registered + matching digest + resolvable references admits; an
unregistered id, an identity mismatch and a dangling reference each refuse, and every cause is
reported. Interface kept pure and extractable, with no IO. Gate:
`cargo test --manifest-path tools/gate/Cargo.toml --offline` — 5 controls green in a throwaway
target. Mutation controls on throwaway copies: the identity check, the reference check and the
unregistered-id report each redden their named control and pass after restoration. This gate-first
task does not close the governing-coverage and barrier-wiring obligations. Next, separately scoped:
extract and refine the real core body, then the adapter, then mandatory-path wiring.

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
