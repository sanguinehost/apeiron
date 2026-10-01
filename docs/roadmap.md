# Programme roadmap

> **Kani retirement.** The [user-directed retirement](epics/kani-retirement.md) supersedes
> every Kani run, preservation task and barrier requirement below. Historical receipts remain
> revision-bound evidence; current work uses ordinary Rust gates and semantic mutation controls.

This is the consolidation and transfer record, not a second programme roadmap. The
[single R0 chain](direction.md#thesis) and required
[G1/REG-1](direction.md#required-r0-acceptance-g1--reg-1) govern; the
[current read/search proof plan](epics/read-proof.md) owns the immediate task and execution order and
supersedes the older dispatch tables below.
The review below records the forward roadmap; earlier lifecycle receipts are folded at the end.

## Verified-AI sequence and prior-art index

```text
SCOPE: docs/roadmap.md, docs/direction.md, docs/epics/agent-throughput-segmentation.md;
       verified-AI ordering and prior-art index.
REGISTER: planning; no theorem, gate or grade change.
CORPUS: none claimed here; historical artifacts are indexed, not promoted.
REFERENCE: paper1/paper2/spinor-ttl and the Neural corpus paths below, read this session.
GATE: git diff --check on the owned files; no compiler gate run by this documentation edit.
GRADE: anchor; historical grades retained, not fresh acceptance.
```

Publication and release are downstream of rebase and end-to-end proof. A pre-rebase artifact is not
publishable and publication is not a programme step; the staged order below is the dispatch order
for the verified-AI outcome.

| # | stage | closes | depends on | gate/evidence |
|---|---|---|---|---|
| 1 | Apeiron core barrier | read/classifier/representation/`apply_insert` refinement to `Apeiron.Table`/`Trichotomy` | current read-map waves | module gates, `AxiomAudit`, Rust barrier, controls |
| 2 | [Neural rebase wave](epics/neural-rebase.md) | `Substrate/Algebra/Neural` and `NullCell` results rebound to the canonical corpus at current grades | stage 1 | focused corpus gate per module; mutation controls |
| 3 | End-to-end extraction wave | executing AI primitives in Rust, extracted with Aeneas/Charon and refined to the rebased lemmas | stage 2 | `proof/regen.sh --check`; module gates; controls |
| 4 | Song base and correspondence | A0 closure and the primitive correspondence on a matched domain | stages 1-3; Song gates | `./song check/test/controls`; joint theorem |
| 5 | Universal translation, pinned domain first | lift and refusal on one admitted source; widening follows A0-A4 | stage 4; translation corpus modules | lift gates; refusing-input controls |
| 6 | AI architecture iteration (`AI-ARCH`); cross-track plan [frontier-consumer](epics/frontier-consumer.md) | topologies derived from the proven base, with mechanism lemmas and capability-per-FLOP measurements on the RTX 5080 | stages 2-3 | mechanism lemma + mutation control + measurement |
| 7 | Release/demonstration | only after a slice is rebased and proved end-to-end | any stage output | that stage's own gates |

Stages 2 and 6 are amended by the [AI-track re-anchor intent](epics/ai-track-reanchor.md): the
historical neural port is frozen reference material (no dependents, no new names), and the AI track is
rebuilt natively on the trichotomy — a minimum standard surface first, then the O(1)
training/finetuning/inference and forward-only-learning model. TorchLean at `de3192df…` is the
reference for the surface's scope, never a dependency. Stages 1, 3, 4 and 5 stand.

Survival income is a separate track (trade roles, platform tasking). It does not consume roadmap
files or release timing.

### Prior-art index (historical; not fresh acceptance)

Sibling-checkout paths. None of these is a current claim until its grade is re-bound under the
current canonical identity (stage 2) and, where it executes, extracted (stage 3).

| artifact | path | establishes | status |
|---|---|---|---|
| Paper I bundle | [`../../sanguine/release/paper1/INDEX.md`](../../sanguine/release/paper1/INDEX.md) | `null_cell_characterization`; `spinor_subtrinity` including forward-only exact gradient/Laplacian; `six_ops_one_cell`; `ridgeHead_unique`, `gram_stack`, `streaming_law`; `rexpand_rcollapse`; the numerics seam (`flash_attention_exact`, quant dual) | Lean v4.30.0-rc2, Mathlib-only, zero `sorry`; historical |
| Paper II bundle | [`../../sanguine/release/paper2/INDEX.md`](../../sanguine/release/paper2/INDEX.md) | one object/one fold, closed-form loops (`constantStep_fixed_point`, `affine_orbit_closed_form`), determinism/confluence, capability-as-data | historical; `Machine`/`Syscall`/`Servitor` rows in progress |
| spinor-ttl | [`../../spinor-ttl/PAPER.md`](../../spinor-ttl/PAPER.md), [`STATUS.md`](../../spinor-ttl/STATUS.md), [`TERNARY_FINDINGS.md`](../../spinor-ttl/TERNARY_FINDINGS.md) | PyTorch ports and measurements: count-LM 2.205 vs 2.392 b/B; ridge fold 16-71x to the same optimum; seal vs NF4 and Q4_K_M; fold finetune vs LoRA with exact unlearning | prototype measurements; ports are reference, not the extraction path |
| Neural corpus | [`../../sanguine/proof/Substrate/Algebra/Neural/`](../../sanguine/proof/Substrate/Algebra/Neural/) | `MirasCountLM` (`hebb_comm`, `forget_hebb`, `surprise_eq_info`), `AttentionSpinor`, `NullTransformer`, `KneserNey`, `CountLMFloor`, `TrainingCollapse`, `MLPrimitivesBridge` §6/§6b | historical; stage 2 re-binds it |
| Fold/softmax algebra | [`../../sanguine/proof/Substrate/Algebra/NullCell/SpinorFold.lean`](../../sanguine/proof/Substrate/Algebra/NullCell/SpinorFold.lean) | `osFuse_is_add`, `osFuse_grouping_free` | historical |
| Frontier ledger | [`../../sanguine/proof/Substrate/Algebra/Neural/FrontierLadder.lean`](../../sanguine/proof/Substrate/Algebra/Neural/FrontierLadder.lean) | the five open seams (`kappa_sufficient`, `kn_attains_kl_floor`, `turing_bridge`, `compositional_metatheorem`, `book_anchor_regular`) and the absent L1 `intelligence` predicate | ledger, prose-bounded; seams are research tasks |

### Private demo milestones (falsifiable, no release)

1. Verified translation of a pinned program set that refuses unsupported inputs.
2. One AI mechanism with its mechanism lemma checked and a matched-compute measurement on the 5080
   against a named baseline, stated as capability per FLOP.
3. One admitted primitive refined end to end: canonical definition through Rust and Song execution
   to an application readout.

## Roadmap review and historical transfer

<!-- APEIRON-STRATEGY-DECISION-T3 BEGIN -->
### Serialized table owner and spatial plan

`APEIRON-STRATEGY-DECISION-T3`, Astra; base `c450c222905c050f793a23332708c230c658f82c`,
dirty supplied-checkout documentation exception. Own only direction, kernel-boundary and this file;
no source/proof/spec edits, build lane, new worktree, commit or push. This is the current dispatch
decision; older task rows remain historical schedules with their requirements retained.

```text
SCOPE: docs/direction.md, docs/kernel-boundary.md, docs/roadmap.md;
       performance verdict, read/scatter boundaries and serialized ownership.
REGISTER: architecture/planning only; no implementation, theorem, pin or measurement change.
CORPUS: Apeiron/Table/Basic.lean, build_eq_of_key_streams_eq:
        (∀ x, per-key value streams equal) → build f d kvs₁ = build f d kvs₂;
        corpus-native proved scope, not a runtime bound or newly checked theorem.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/.lake/packages/apeiron/proof/Apeiron/Table/Basic.lean:167-174;
           /home/socol/Workspace/apeiron/src/table.rs.
GATE: git diff --check -- docs/direction.md docs/kernel-boundary.md
      docs/roadmap.md; repository root; pre-edit exit 0.
      Exact inherited-text preservation with deletion control; no documentary red invented.
GRADE: anchor decision; cited measured receipt not rerun; no new proved/tested runtime result.
```

**Decision and evidence.** [Direction](direction.md#serial-read-specialization-and-stable-parallel-scatter)
retains the strict bar: attainability is unestablished, not ruled out by a 256-shard bound. Select
resident inlining plus a read-only absence path, then stable parallel scatter retaining validation.
The [post-correctness receipt](evidence/perf-ledger.md) records
strict exit 1, `pass=421 fail=1029 full_acceptance=false`; all ten spatial pairs fail, six on
face/concurrency before speed. Its source is `00c3c7f8…`; the read `src/table.rs` is now SHA-256
`d2bf704f31f8d59f2e41b27b06bdca90a24a8a1a044ff35206253c50f81658d1` (2,525 lines).
Do not attribute its timing to this later source. The generated file read hashes to
`b590a393f96c64c7df123d41a89209bf2f0e3c70d5678a4183d6b39ad3b32591`; the supplied 1708-job
green extraction is not a fresh run here, and the active Sol bridge re-proof keeps its own lane.

The historical bodies were read with `git show 3a95bb5:src/table.rs` and
`git show 71942db:src/table.rs` (full revisions retained below). Both use 256 load-sized regions,
chunk-private scatter, deterministic LPT and grouped `sum_parallel`. They differ: `71942db` routes
the top bits of `finish`, while `3a95bb5` routes low placement bits and separates identity from
placement. Neither justifies changing current certified-word routing or substituting sum for ordered
readback. Their historical L2 and speed figures are anchors here, not measured properties of this
source or proof of HashDOS resistance.

**Ownership sequence.** Prefix new rows with `APEIRON-READ-SCATTER-`. Every row is serialized; no
second `table.rs` owner, even for spatial work. Release active source and Sol owners first and pin
their dirty snapshots. Each implementation task has an isolated worktree/private target; after each
source row run the extraction/regression barrier below before the next source owner. Flash is default;
Sol owns contract/theorem/grade decisions. No Terra-style sustained implementation is necessary or
authorized for these bounded slices. A larger hot-directory/payload redesign returns to Astra and
Sol for decomposition, then needs explicit lead assignment if Terra is selected.

| Task / owner | Owned files and completion condition | One focused gate (root unless stated) | Rejecting control in disposable copy |
|---|---|---|---|
| W1-T1 / Flash, size-only | `src/table.rs`, new `src/table/state.rs`, new `src/table/tests.rs`. Move tests and state/scalar helpers as specified below; no semantic tuning. | `cargo test --locked --lib --test table` | Change moved flat index/routing; existing actual-path tests fail, restored pass. Compare moved bodies and size inventory as review prerequisites. |
| W2-T1 / Flash, resident | `src/table.rs`, `src/table/state.rs`, `tests/table.rs`. Inline existing `get_at`/decision/apply and flat-index helpers, remove only demonstrated duplicate work without changing geometry or result classes. | `cargo test --locked --lib --test table` | Suppress spill fallback on occupied unequal-Sigma/direct collision; readback rejects for both carriers. Direct-empty/default and extent-growth cases remain required. |
| W3-T1 / Sol, read contract prerequisite | New `docs/intent/table-read-path.md`, `docs/spec/table-read-path.md`, `docs/plan/table-read-path.md`. Ratify found/absent interface, existing filter-law prerequisites, shared executing apply, root mapping, witnesses and exact proof/control targets. No occupied storage-spec edit. | `git diff --check -- docs/intent/table-read-path.md docs/spec/table-read-path.md docs/plan/table-read-path.md` plus Sol semantic review | Reject summary-positive-as-present, direct non-match-as-absent, or an absent read passed as insertion vacancy. No implementation dispatch without verdict and named semantic gate. |
| W4-T1 / Flash, read-only locator after W3 | `src/table.rs`, `src/table/tests.rs`, `tests/table.rs`. Route public reads through the shared production read classifier/apply; skip vacancy scans on certified absence and vacancy bookkeeping on read sweeps. Insertion locator unchanged. | `cargo test --locked --lib --test table` | Clear a required summary bit or terminate on first empty despite a late hit; named both-carrier collision/late-hit tests reject. Add an actual-path work assertion that negative-summary reads scan no vacancy; zero matching tests never pass. |
| W5-T1 / Sol, spatial lifecycle prerequisite | New `docs/intent/parallel-scatter.md`, `docs/spec/parallel-scatter.md`, `docs/plan/parallel-scatter.md`. Fix stable output equality, worker budget, safe slices, reservations/refusal and constructor proof obligations from the boundary. | `git diff --check -- docs/intent/parallel-scatter.md docs/spec/parallel-scatter.md docs/plan/parallel-scatter.md` plus Sol semantic review | Reject completion-order concatenation, concurrent aliasing, late fallible reservation after callbacks or removal of production validation. |
| W6-T1 / Flash, spatial scatter after W5 | `src/table/partition.rs`, `src/table/parallel.rs`, `tests/parallel/main.rs`. Add chunk helpers and executor-owned scopes, concatenate into unchanged contiguous `Parts`; use it for parallel build/probe, retaining `is_intact`, LPT and query-order callbacks. | `cargo test --locked --lib --test parallel` | Reverse chunk order for one shard containing a repeated key; partition equality/noncommutative public build tests fail. Also retain duplicate/omission, uneven chunks, skew, empty input and 1/many-worker cases. |
| W7-T1 / Flash, exclusive measurement host | New `docs/evidence/perf-read-scatter.md`, `docs/evidence/perf-read-scatter.log`, `docs/evidence/perf-read-scatter.stderr.log`. Full matched before/after receipt after integrated regression, with raw output retained durably. | `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls` | Existing oracle/digest, workload, worker/face, missing-pair and slowed-path/implementation-mismatch controls reject; unchanged restored campaign. Exit 1 records remaining red evidence, never speed acceptance. |

**Exact size cut.** Keep public `Home`/`Merge`, `Table`, search and mutation/build operations in
`table.rs`. Move `Region`/summary helpers/publication trace, decision value types/constructors,
scalar geometry helpers, `FlatSlot`/`Flat` and rebuild state declarations into `state.rs`;
move the whole `#[cfg(test)] mod tests` body to `tests.rs`. Preserve the logical `table` namespace
for state with `include!("table/state.rs")`, and use explicit `#[path = "table/tests.rs"]` for
the test module, so this cut changes physical files rather than generated type identities. It is
one source definition per body, not a copied implementation. Preserve bodies, layout, field privacy,
signatures and visibility; no inlining annotations until the resident task. The state cut includes
`valid_n`, `shard`, `tag`, `matches`, `slots_for`, `capacity_bound`, `group`, `least_set_lane` and
`mark_direct_word`; with the 685-line test block moved, this leaves room below 1,500 lines in the
parent. Verify actual counts after formatting, each file ≤1,500 and directory ≤12; do not merely
hide an oversized file with another name. If pinned Charon cannot preserve the namespace through
the include, stop for an explicit module-path/bridge transport task, not a simultaneous proof edit.

**Mandatory serialized extraction/proof barrier after each source row.** These are separate task
instances, not extra file ownership assigned to the source worker:

| Barrier task / owner | Files | One gate | Rejecting control |
|---|---|---|---|
| E / Flash | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | Bounded `proof/regen.sh`; preserve both carriers/all roots, include new source digests and W3's approved read roots. | Restored `proof/regen.sh --check` passes; scratch changed actual body/generated write rejects freshness. No formula-restating wrapper. |
| P / Sol, regression | At most two affected bridge modules plus `proof/corpus/AxiomAudit.lean`, selected from the actual generated diff by the active owner. | Bounded `lake build` in `proof/corpus`; exact focused modules must be frozen before edits. | Generated wrong value/tag/summary write rejects the named semantic theorem, not just elaboration/freshness. Missing scope requires a new task. |

W3 must additionally name a source-bound both-carrier read theorem and rejecting semantic gate;
W5 must name constructor/schedule linkage. Those new proof tasks are **not dispatch-ready** until
their exact statements/modules are ratified. Existing green modules do not prove the new path.
Preserve current AGENTS §6 tests/clippy/fmt, `proof/regen.sh --check`, `lake build` in both packages,
all named controls, source/tool digests and size/perimeter checks at integration. Lean commands keep
the `flock /tmp/opencode/apeiron-lane.lock`, `systemd-run --user --scope -p MemoryMax=8G
-p MemorySwapMax=0`, `LEAN_NUM_THREADS=1`, `timeout 900s` wrapper and bounded reductions. The
relocation runner owns its lock and runs standalone. No live cache overlaps or source edits while
extraction/measurement consumes that identity. W7 composition reopening is unchanged.

**Measurement and stop.** `micro_spill` is a diagnostic, not a pair gate: it has no incumbent and
does not cover the offset-miss/resident cells. Use the full registered campaign to decide parity;
an exit-0 micro run or test cannot do so. Pre-register same 4096-op traces, campaigns/uncertainty,
worker budgets, quiet host, machine/tool/input hashes and finite 3600 s limit; preserve the receipt's
per-pair interval rule. Attribute negative/positive summary frequency, group/vacancy work and
resident dispatch using a separately assigned bounded diagnostic if needed, not an invented cycle
bound. If these slices leave either target red, stop speculative tuning and return that cell and
causal profile for the hot-directory/absence-certificate design in direction; the bar does not move.

Comparison registration repair of the five hashers is done, not complete historical coverage:
`hash-comparison.md` still records missing `CoordMap`/serial `FreqMap` adapters. Spatial worker/face
repairs remain a separate harness-owner task, not relabeling mismatches or changing the timed region
inside this optimization. Its exact focused gate is still missing; freeze a gate-first task before
dispatch. The unedited handoff's old performance checkpoint needs a later ≤3-file reconciliation;
this section and direction supersede its status only, retaining all of its gates.

**Preservation gate.** Remove only the marked T3 blocks to reproduce the dirty pre-edit documents
exactly: direction SHA-256 `2df811dbe0d57de7a4bc95be68bd0573cc2a82d6d3bca84615c49bd5d5933980`,
boundary `bc1ed96e6243c5475874b0887372e7989f01039868afaba3f82e3fd3ee661655`, roadmap
`172c784fdf4c57a5f60cb5604df0d30f65f7211ac39e54f398a0685c7a9eb708`.
The scratch check deletes an inherited requirement line in memory and must reject each document,
then pass restored. Readback checks additions for contrary waivers; byte preservation alone cannot
establish that. No inherited gate, requirement, acceptance bar or trust statement is retired.
Observed documentation acceptance: scoped `git diff --check` exits 0;
`python3 /tmp/opencode/apeiron-strategy-t3-preserve.py` exits 0, with 3/3 exact preimages,
3/3 inherited-requirement deletion controls rejected and restored inputs passing. Readback retains
the strict threshold, complete gates and trust exclusions without a contrary waiver. Runtime and
Lean gates are unrun in this docs-only task; attainability remains the stated empirical blocker.
<!-- APEIRON-STRATEGY-DECISION-T3 END -->

<!-- APEIRON-STRATEGY-DECISION-T2 BEGIN -->
### Partition cut and frozen bridge maintenance

`APEIRON-STRATEGY-DECISION-T2`, Astra, base `c450c222905c050f793a23332708c230c658f82c`;
dirty supplied-checkout documentation exception. Own only `docs/direction.md`,
`docs/kernel-boundary.md`, `docs/roadmap.md`; no build lane, branch, commit or push.
This section supersedes the older dispatch below, not any gate or acceptance requirement.

```text
SCOPE: three owned documents above; partition decision, filter ledger, disjoint dispatch.
REGISTER: architecture/planning; no src/proof/tests/examples/spec/pin edits.
CORPUS: historical note named Growable.Relocation.generated_tag_nonzero, but no current declaration
        resolves under that name; target/open, not proved evidence.
        Growable.Representation.RepR lacks summary/fingerprint invariants.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Growable/Relocation.lean;
           /home/socol/Workspace/apeiron/proof/corpus/Growable/Representation.lean.
GATE: git diff --check -- docs/direction.md docs/kernel-boundary.md
      docs/roadmap.md; cwd repository root; pre-edit exit 0.
      Exact pre-edit preservation plus deletion control; no documentary red invented.
GRADE: anchor before/after; no fresh mathematical proof, runtime test or measurement.
```

**Decision:** authorize only the cheaper fail-closed partition validator from the
[boundary contract](kernel-boundary.md#spill-filter-obligations-and-partition-cut); reject deleting
the payload initialization pass in this dispatch. Priorities are serial spill present-hit/resident
cost and spatial partition/kernel cost, not another W7 composition attempt. The read strict receipt
is still red; its absent-miss/growth improvements do not measure the later fingerprint/parallel source.
The supplied 9.9 ns/op present-hit and build kernel 46–55%, partition+validate 28%, assembly 16%,
probe readback 16% are diagnostic inputs only: raw source-pinned phase receipts are missing from this
handoff. Probe/build percentages cannot be summed or used as a predicted full-campaign speedup.
Registry repair of the five hashers is present; the older instruction to make them load-bearing is
historical. `CoordMap`/serial `FreqMap` coverage and spatial face/concurrency mismatches remain open.

**Intent/spec prerequisite.** Before production validation changes, Flash prepares a ≤3-file lifecycle
task owning `docs/intent/spatial-validation.md`, `docs/spec/spatial-validation.md` and
`docs/plan/spatial-validation.md` (new, not created here). Its one gate is scoped `git diff --check`
plus contract review; scratch deletion of a rejection class must fail review. Specify old/new acceptance
sets, deterministic routing, stable per-key streams, tile/cardinality proof, refusal behavior, exact
extraction consumers, tests and measurement criterion. Require strictly increasing in-shard indexes;
prove sound coverage from routing and cardinality, and completeness for the real `Parts::of`. Sol's
contract verdict is mandatory because this changes a fail-closed check; no software dispatch until it
arrives. Do not edit the occupied storage spec. The kernel-layout alternative remains a design request,
not hidden authorization to change `PreparedResources_R`.

**Next parallel editing wave.** Prefix task IDs with `APEIRON-STRATEGY-CUT-`. Lead must release
current owners and pin each dirty source snapshot first; each task gets an isolated worktree/private
target. W1 tasks below are disjoint, but heavy gates lease the host serially. An in-flight extraction
may finish only against its immutable input snapshot; regeneration cannot read concurrently changing
source. W1-T3 is preparation of the handoff, not a second extraction owner. No row asserts an unrun
gate passes; prospective tests must exist and execute, not match zero tests.

| Task / owner | Owned files and output | One gate (root unless noted) | Rejecting control |
|---|---|---|---|
| W1-T1 / Flash, after lifecycle/Sol verdict | `src/table/partition.rs`: remove `seen` and final coverage pass using checked tile/length, bounds, strict run order and real routing; same-file tests. No `Parts` layout/API or executor edit. | `cargo test --locked --lib table::partition::tests::` | Scratch duplicate within/across shards, omitted index, reversed same-key run, malformed endpoints, out-of-bounds index and wrong route all reject; valid stable scatter passes restored. Mutate the production validator to bypass one check and require the corresponding test to fail. |
| W1-T2 / Flash | `src/table/scan.rs`, `tests/table.rs`: continue serial present-hit work only within exact scan implementation and public-path regression tests. Retain byte-mask exactness, full-key/Sigma comparison, summary semantics and first-fit order. Resident/public-path attribution becomes the handoff for the next table-owner wave, not a concurrent `table.rs` edit. | `cargo test --locked --lib --test table` | In scratch regress SWAR to the borrowing/superset mask or suppress a live matching lane; mask/readback tests reject. Actual-path tests include both carriers, collisions, late hits and resident/spill fallback. If no justified scan improvement exists, return attribution without speculative code. |
| W1-T3 / Flash | New `docs/evidence/filter-extraction-handoff.md`: collect the active extraction owner's source/generated/tool hashes, command/exit and exact failing goals. Inventory summary fields/writes, changed partition roots and phased API perimeter; freeze root/consumer list. No generated/proof edits. | `git diff --check -- docs/evidence/filter-extraction-handoff.md` plus source/receipt review | Scratch omission of one carrier/root or promotion of stale extraction to fresh must fail review. Missing logs remain blockers; predicted tag-unfold failure is not reported as an observed Lean error. |

**Sequenced continuation, not falsely parallel ownership.** Resident dispatch, spill locator and
kernel initialization all share `src/table.rs` (2,123 lines at read time). No simultaneous owners and
no size waiver. Before any edit there, a separate Flash factoring task owns `src/table.rs`, new
`src/table/state.rs`, new `src/table/tests.rs`; move state/region definitions and tests without semantic
changes, preserve extraction paths and end with each owned source ≤1,500 lines and directory ≤12
immediate source files. Gate `cargo test --locked --lib --test table`; scratch changed routing fails.
If this three-file split cannot preserve the root/interface boundary, return to Astra before editing.
After factoring and its extraction/bridge barrier, a serial continuation task owns only `src/table.rs`
and `tests/table.rs`, gate `cargo test --locked --lib --test table`: reduce measured redundant resident
dispatch/address work and spill-hit overhead without changing absence/identity laws. Wrong direct-empty,
unequal-Sigma and late-hit mutations must reject. Kernel-layout work is not part of that authorization.

| Later task / owner | Owned files and dependency | One gate / control |
|---|---|---|
| W2-T1 / Flash extraction owner, exclusive | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`; after W1 source freeze (and repeat after later source waves). Keep all roots for both carriers; no wrappers restating formulas. | Bounded `proof/regen.sh`; restored `proof/regen.sh --check` and scratch source/generated drift must reject. A green extraction is not a bridge proof. |
| W3-T1 / Sol (theorem trigger), exclusive proof lane | `proof/corpus/Growable/Relocation.lean`, `proof/corpus/Growable/Search.lean`, `proof/corpus/AxiomAudit.lean`; use exact regenerated errors, repair tag unfolding and actual summary-state writes/readback without silent premise changes. Flash prepares reproduction; Sol owns proof edits/verdict. | Bounded `lake build Growable.Relocation Growable.Search AxiomAudit` in `proof/corpus`; corrupt generated tag/payload write, require named semantic theorem failure and restored pass. Dependency failures outside these files block and require a new ≤3-file task, not expanded ownership. |
| W4-T1 / Sol contract, before invariant proofs | `docs/epics/storage-refinement.md` (only after its owner releases it): ratify the four ledger predicates, summary-miss search certificate, resource conflicts and finite source-linked discharge map. | Scoped `git diff --check` plus semantic contract review; reject assumed post-state preservation or summary-hit-as-presence. This is a contract gate, not proof. |
| W5-T1 / Flash measurement owner, exclusive host | New `docs/evidence/perf-strategy-cut.md`, `perf-strategy-cut.log`, `perf-strategy-cut.stderr.log`; after the integrated source/proof regression barrier. | `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`; retain oracle/digest, workload, worker/face, missing-pair and slowed-path/implementation-mismatch rejection, with unchanged restored campaign. Exit 1 records red evidence and does not close speed acceptance. |

W4's verdict must decompose invariant proof work before dispatch: generated summary shape/set/maybe
lemmas and fingerprint equation first, then source-write preservation and actual constructor/resource
links, then Representation/Resources integration and search-certificate composition. Use at most two
semantic modules plus audit per task; exact statements/gates are **MISSING** until Sol ratifies them.
Do not add all these premises to existing theorems or reopen W7 to hide the work. Local proof repair
that requires new summary bounds stops for Sol's contract decision; adding `summary := empty` to
symbolic writes is not a faithful repair. Source allocation/refusal conflicts identified in the
boundary need a separately decomposed correctness task; they block complete integration, not permission
to relax no-clone-on-`Err`.

**Integration/measurement discipline.** Preserve every current AGENTS §6 Rust command, regeneration
freshness, both-package `lake build`, audits, baseline/red/restored controls, source/tool digests and
size/perimeter checks. All Lean commands use `flock /tmp/opencode/apeiron-lane.lock systemd-run --user
--scope -p MemoryMax=8G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s <command>`; retain
bounded heartbeats/recursion. Existing relocation control owns its lock and runs standalone; semantic
mode remains required. Builds/controls serialize on the generated file and both `.lake` lanes.
The exclusive measurement pauses all compilers/Kani/Lean/source and harness edits, uses the receipt's
registered machine/worker/cache/sample policy and finite 3600 s timeout, captures the actual command
exit (not `tee`), raw samples and before/after identities. New workers, timed phases or applicability
need registration before timing, not metadata relabeling. No missing spatial pair becomes a pass.

Sol escalation: theorem/premise/domain/grade change, filter/search/routing/resource law or surviving
mutation. Astra escalation: >3 files, overlapping owners, a different payload representation,
unapproved root/tool boundary or document conflict. Both verdicts are required when both triggers hold.
The current extraction owner retains its lane; W7 reopening and joint G1 acceptance remain exactly
as in direction. No runtime or Lean gate runs in this documentation task.

**Preservation check.** The three T2 blocks are additive. Removing only their marked ranges must
reproduce the dirty pre-edit documents byte-for-byte: direction `f6673440223602abdf3e920fbd2abbb73633b8ec8d423166da2c24165347ce6c`,
boundary `48c94962ea6f31dd963ac702e6ecd8867861a9b838030ced78efa2ba9d80c6a0`, roadmap
`fcc76b203d30a8d77adfe70826be898d03eca5031e62a671d028529f49e5eb56` (SHA-256).
Deletion of an inherited acceptance line in memory must fail that check. Readback additionally checks
that new status/dispatch text does not waive a requirement. The strict threshold, owner DoD, all
incumbents, `10^8` identity, refusal/ordered merge, witnesses, both cones, full barrier, coverage and
trust exclusions remain required; the new ledger is unproved, not accepted end-to-end refinement.
Observed documentation acceptance: scoped `git diff --check` exits 0;
`python3 /tmp/opencode/apeiron-strategy-t2-preserve.py` exits 0, all three dirty preimages match,
all three inherited-requirement deletion controls reject, and restored inputs pass. The scratch
checker writes no repository file. Readback finds no removed or weakened acceptance requirement.
<!-- APEIRON-STRATEGY-DECISION-T2 END -->

### Performance re-anchor and bounded dispatch

`APEIRON-STRATEGY-DECISION-T1`, owner Astra; base
`c450c222905c050f793a23332708c230c658f82c`, dirty supplied checkout. Exception: documentation only
in this checkout, no branch/worktree, build lane, commit or push. Own `docs/direction.md`,
`docs/kernel-boundary.md` and this file; preserve every other owner's work, including the in-flight
benchmark and Sol's spec. This section supersedes immediate W7 retry scheduling in earlier records,
not their acceptance requirements. No new proof, test or runtime result is recorded.

```text
SCOPE: docs/direction.md, docs/kernel-boundary.md, docs/roadmap.md;
       freeze W7 retries without closing G1; performance priorities and bounded tasks.
REGISTER: architecture/planning; source, proof, spec, pins and runtime changes excluded.
CORPUS: Growable/Search.lean excludes whole-table preservation;
        Growable/Rebuild.lean word_{stored,regen}_rebuild_preserves concludes ReservedRepR
        under source RepR and resource/plan certificates; existing proved scopes, not rerun.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Growable/Search.lean;
           /home/socol/Workspace/apeiron/proof/corpus/Growable/Rebuild.lean.
GATE: git diff --check -- docs/direction.md docs/kernel-boundary.md
      docs/roadmap.md; repository root; pre-edit exit 0;
      line-by-line preservation review, no documentary red claimed.
GRADE: anchor before/after; no new proved, tested or measured result.
```

**Decision.** Freeze W7 as [W7-SearchCommitBoundary](kernel-boundary.md#w7-searchcommitboundary).
Preserve core proofs and their regression gates; do not spend another proof attempt adding premises
to an absent whole-state `apply_insert` result. This is not epic closure: the unchanged joint G1
contract still requires real-Rust canonical refinement and every applicable performance pair.
The exact reopening condition is in [direction](direction.md#current-strategy-decision-freeze-composition-restore-performance).
Independent bounded physics/Null-cell, Whispering Tides, game and world-simulation probes continue;
the next large AI implementation remains behind joint readiness.

**Fresh-measurement admission.** The current measurement owner finishes and releases the source,
harness and host lanes first. Read the full receipt, raw samples, oracle/control outcomes, source /
harness / input digests, machine/tool identities, worker policies and uncertainty. Do not reuse the
stale 84/566 tally or the older global/first-cut receipts as an after-result. A source change during
measurement invalidates that identity. A missing or mismatched pair, inconclusive interval, missing
historical surface or informational classification of a required incumbent keeps acceptance open.
In particular, the read registry excludes XXH3/XXH64, wyhash, ahash and FNV; repair that gap before a
complete speed verdict. `hash-comparison.md`'s contrary recommendation cannot weaken direction.

| Fresh finding | Priority and selected mechanism | Constraint / stop |
|---|---|---|
| Oracle, workload, registration or concurrency defect | First: repair evidence/correctness, then remeasure the same complete matrix. | No optimization justified by an invalid pair. `arm_get_each` currently times `run_count_par` but labels its worker policy as single-core probe: reconcile the timed build/probe phases, not just the printed width. |
| Serial direct hit/miss/update still behind hashtrinity or hashbrown | First production slice: remove redundant address/dispatch/materialization overhead on the existing direct path; retain checked flat index, live bit, full `(Sigma,K)` identity and spill fallback. Compare elapsed time and instructions/lookup separately. | Do not remove identity or infer absence from occupied non-match. If serial direct already passes, skip speculative tuning and advance spatial work. A std-only win is insufficient. |
| Packed parallel build/probe still behind | Restore the historical mechanism end to end: 256 low-placement-bit shards, stable chunk scatter, deterministic LPT, independently load-sized cache-local regions, one shard owner, grouped probe/assembly. These functions already exist; measure their extra copies, repeated address computation, serial partition, initialization and scans before replacing them. | The current certified route uses raw-word low bits, not universally placement low bits; changing routing requires Sol. L2 residency is measured per region/worker, not guaranteed by 256 shards. Match workers and timed phases against hashtrinity, hashbrown+rayon and foldhash/fxhash par16. |
| Fresh insertion / clustered spill dominates | Attribute full-group miss scans and density/rebuild work separately; retain exact SWAR tag masks and first-fit order. Optimize work without changing the absence certificate first. | `spill_locate` scans all groups on absence; faster first-empty termination or historical variable-width descent needs Sol's reachable-state theorem/contract decision before code. Repeated density scans are a separate cold-path cost, not lookup cost. |
| Common costs recovered, isolated incumbent pairs remain red | Select the worst remaining interval excess per operation, then reproduce its causal path; all named incumbents remain required. | No mean/geomean closure, benchmark-specific production dispatch, pair deletion or family exemption. |

Historical `3a95bb512d59e56456d5a2edb1324639f63922b0` Map and
`71942db30f8cf2970c7d3ceb7b5d7389d66fd0a4` CountTable are pinned mechanism/measurement anchors
(registered in `examples/reg1/history.rs`), not source automatically authorized for restoration.
Before porting a mechanism, read its exact historical body and bind its digest. Keep one production
`Table`; use existing `Regen<I64WordCodec>` for exact coordinate key dropping, not probabilistic
`FreqMap` identity or a second CountTable. Preserve the query-order callback API. A worker-local
`sum_parallel` reduction is only a later separate task after its exact wrapping/overflow/merge-law
contract is fixed; current `sum_parallel` materializes ordered results and then sums. It cannot
replace the order-sensitive REG-1 digest to manufacture a win.

**Acceptance and non-goals.** Mandatory speed gate remains
`cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`,
with every applicable pair's ratio interval maximum `<= 1.0`; missing, invalid, duplicated,
unauthorized or inconclusive pairs reject. Require matched before/after on real production, all
registered surfaces and independent correctness before timing. The additional scoped stretch target
is `<= 0.8` against the fastest named incumbent on dense-coordinate serial lookup and packed spatial
build+probe; it is a target, not a measurement or permission to relax other pairs. No global SOTA
claim follows. No new cryptographic face, GPU/ISA backend, second container, unsafe layout or lossy
key scheme in this leg. HashDOS resistance is still mandatory; determinism, `10^8` identity,
round-trip, remove/growth/refusal, ordered merges and coverage remain open until their own gates pass.

**Dispatch table.** IDs below have prefix `APEIRON-REANCHOR-`. All are proposed tasks requiring lead
assignment and release of existing file owners, not edits authorized by this docs pass. `W1` is a
parallel Flash editing wave with disjoint files; gates lease the host serially. Gate cwd is root
unless stated otherwise. New test files/runner modes are explicitly prospective, never existing
evidence. Every control mutates a disposable copy and requires green/red/restored-green for the
named defect; resource/tool failures are inconclusive.

| Task / owner | Owned files (at most three); output | One task completion gate | Control |
|---|---|---|---|
| W1-T1 / Flash | `examples/reg1/acceptance.rs`, new `examples/reg1/acceptance/tests.rs`: add unit tests of the real acceptance evaluator, make the five required hasher comparators load-bearing, and reject their demotion/omission. Inventory unmatched named surfaces explicitly. Keep each file at most 1,500 lines (parent currently 1,499); stop for decomposition if necessary. | `cargo test --locked --example reg1 acceptance::` (prospective tests; zero matching tests cannot close the task). | A single newly load-bearing ratio over 1, an omitted pair, or demotion to informational must reject in the same evaluator; restored complete synthetic campaign passes. |
| W1-T2 / Flash | `tests/table.rs`: actual-public-path differential traces for both carriers, direct/spill collisions, unequal `Sigma`, extent changes, whole observation set/length and noncommutative merge. No production edits. | `cargo test --locked --test table` | Scratch wrong value/length/occupancy, omitted commit and reversed merge must fail named tests, including unrelated-key readback. |
| W1-T3 / Flash | `docs/plan.md`, `docs/hash-comparison.md`: reconcile suspended W7 scheduling and all named incumbents; retain every gate and historical receipt. | `git diff --check -- docs/plan.md docs/hash-comparison.md`, plus line-by-line requirement preservation as for this docs task. | Scratch deletion of a required incumbent or replacement of W7-open by proved must be rejected by the review; this is documentary review, not a software control. |
| W2-T1 / Flash, own wave | `src/table.rs`, new `src/table/geometry.rs`, new `src/table/tests.rs`: size-only factoring before hot-path changes; move tests and scalar geometry without changing bodies, visibility semantics or root behavior. Current `table.rs` is 2,017 lines, above the 1,500-line owned-source bound. | `cargo test --locked --lib --test table` | Scratch changed routing/index equation must fail the corresponding tests; unchanged factoring must preserve observations. Existing size/perimeter checks and extraction barrier also remain mandatory. |
| W3-T1 / Flash, after size/freshness barrier | `src/table.rs`: measured serial direct-path overhead reduction only, no new representation/search law. | `cargo test --locked --lib --test table` | Wrong direct-empty or unequal-`Sigma` result rejects; performance outcome evaluated separately at W5, not inferred from tests. |
| W3-T2 / Flash, parallel edits with T1 | `src/table/partition.rs`: optimize stable scatter/address work at the existing interface only if fresh attribution selects it; add same-file order/completeness tests. | `cargo test --locked --lib table::partition::tests::` | Reverse same-key runs or duplicate/omit a scatter index; tests reject. No routing-law or unchecked-allocation change. |
| W4-T1 / Flash, after W3 integration | `src/table/parallel.rs`, new `tests/table_parallel_reanchor.rs`: reduce wrapper/query copies and assembly overhead under unchanged callback order and worker ownership. No new reduction API. | `cargo test --locked --test table_parallel_reanchor` (prospective target). | Reorder query callbacks or same-key values, or omit a shard; the independent oracle rejects. Retain refusal effects and join-before-unwind behavior. |
| W4-T2 / retired; ordinary regression retained | `src/table/probe.rs`: retain finite late-match, collision, summary, first-vacancy, growth and unequal-`Sigma` regressions over the real helpers. | `cargo test --lib table::probe::tests::` | In a disposable copy, truncate the real cyclic sweep; the late-match regression must fail and the unchanged source pass. |
| W5-T1 / Flash measurement owner, exclusive host | New `docs/evidence/perf-reanchor.md`, `perf-reanchor.log`, `perf-reanchor.stderr.log` in that directory: full matched before/after receipt, including identities and all raw samples. | Exact full-profile strict command above. | Existing oracle/digest, workload, worker-width, missing-pair and slowed-path/implementation-mismatch controls reject; unchanged restored campaign is measured. Exit 1 records red evidence, not a completed speed task. |

Before W3, the measurement owner must assign any timed-phase/coverage repair as its own task:
`examples/reg1/adapters/spatial.rs` plus at most two named harness/test files. **Focused gate MISSING**
for asserting per-phase concurrency and complete historical-surface coverage; freeze a rejecting
gate-first task before implementation. Neither W1's evaluator unit tests nor a printed matched width
settles that issue. Do not run or alter the active measurement to repair it in place.

**Lanes and barriers.** Each implementation owner uses `worktree-agent-<task-id>` and records the
integrated dirty-source digests; HEAD alone is not the baseline. Use a private `CARGO_TARGET_DIR`
per task, never another owner's `target/`. Parallelism is editing/review, not simultaneous host load:
one compiler/Lean job at a time; one benchmark campaign exclusively, with all other heavy work
paused. The in-flight measurement retains its lease. Ordinary Rust gates use a timed 8G,
no-swap scope with `CARGO_BUILD_JOBS=1`. Benchmark cap/time/thread policy must
match its registered baseline (no `taskset`/worker change invented after timing); reserve the host
and freeze a finite timeout before the next run. An unavailable lease or cap failure blocks the run.

After each source wave, serialize a separate Flash extraction task owning `proof/extract/lib.rs`,
`proof/regen.sh`, `proof/ApeironWord.lean`, gate bounded `proof/regen.sh`, with freshness mutation and
restoration controls. Do not edit wrappers unless factoring requires it. Then run existing bounded
`lake build` in both packages and core controls; generated-file and both `.lake/` lanes are exclusive,
including scratch controls. Preserve the documented `flock /tmp/opencode/apeiron-lane.lock`,
8G/no-swap/`LEAN_NUM_THREADS=1`/900s wrapper; relocation runner remains standalone because it owns
the lock. No consumer sees a partially regenerated file. A proof regression blocks integration,
not permission to drop a root. Shared integration remains the complete AGENTS §6 Rust,
freshness, both-package builds, controls, digests and size/perimeter set; focused test exits do not
close a wave in their absence. Performance acceptance additionally needs W5 and eventual W7 discharge.

**Escalations and stop points.** Flash is default. Sol, one task at a time including the spec owner,
adjudicates a theorem/premise/grade change, geometry/absence/codec or reduction law, or a surviving
semantic mutation. Astra resolves >3 files, overlapping ownership, source-size
decomposition, changed representation/resource boundaries or governing-document conflict. If both
triggers hold, both verdicts precede implementation; neither waives a gate. No source-size exception,
automatic promotion of historical code, new allocator axiom or giant proof/tool rewrite is authorized.
Counterexamples get correctness repair immediately. The unedited spec's stale W7-in-flight and
no-extra-premise wording needs its Sol owner's reconciliation before proof resumption, not a silent
contract rewrite here.

**Preservation review.** This task inserts current decision sections only. Every inherited line of
the three dirty pre-edit documents remains unchanged and in order: all commands, gates, owner DoD,
strict per-pair threshold, required theorem/witness sets, refusal order, HashDOS requirement, both
axiom cones and realization exclusions are retained. Supersession concerns dispatch/status only;
trusted/open is not proved or accepted. Old receipts are not fresh results. The documentation gate
does not test software, discharge W7 or establish runtime benefit.

Documentation acceptance: scoped `git diff --check` with the three owned paths exits 0. Readback
review covers every inserted line and the retained acceptance/boundary clauses; no inherited gate,
command, requirement or boundary statement is removed or weakened. No software gate or benchmark
runs in this task. Source changes and fresh-measurement publication remain with their existing owners.

Task `APEIRON-SONG-DOCS-ROADMAP-REVIEW`, owner Astra. Supplied-checkout exception: no branch,
worktree, commit, push, source change or build lane. Bases: Apeiron
`c450c222905c050f793a23332708c230c658f82c`, Song
`aeaf3baac434b2e3c01154d275bea65fdadc69bf`; both have pre-existing dirty work.

```text
SCOPE: docs/direction.md, docs/roadmap.md, song/docs/ROADMAP.md;
       consolidate the destination, add A7 and record historical transfer decisions.
REGISTER: planning/documentation; no implementation, theorem, pin or runtime changes.
CORPUS: universalLift_map / universalLiftCons_state (fold naturality/state);
        automated_lifting_sound (SKIShaped premise); universal_generation (Law substitution);
        proj_zero_cons / proj_succ_cons (List equations). Source read, not rebuilt.
REFERENCE: /home/socol/Workspace/sanguine/proof/Substrate/Algebra/Translation/UniverseSyntaxSerialisation.lean;
           sanguine@7d132ffa17fa19b62ff53487d85f254aac108c46:interop/src/.
GATE: python3 /tmp/opencode/roadmap_review.py check; initial exit 1:
      AssertionError: song resonate. git diff --check in each repo initially exits 0.
GRADE: forward-looking design and source-reading analysis; no new proof or speed claim.
```

### The hashtrinity gap is the open performance item

Owner direction: the hashtrinity-to-apeiron conversion was never done, and it is the highest-value
outstanding item — Penglai's fluid simulation depends on this table, and a 9x closing is the
difference between a workstation and a modest machine.

**Measured** at `examples/table_cost_probe.rs` on 192 keys with `N = 16`, this machine:

| population | metric | apeiron | hashtrinity `4cc14471` | ratio |
|---|---|---|---|---|
| distinct | present ns/key | 9.73 | 1.04 | **9.35x** |
| distinct | insert ns/key | 480.98 | 4.04 | 119x |
| lowbyte | present ns/key | 31.52 | 2.12 | 14.9x |
| adversarial | present ns/key | 292.35 | 0.78 | 374x |

The requirement is not new. [direction.md](direction.md) already makes the consolidation
**performance-preserving** — it carries forward the predecessor's high-performance surfaces, not
merely its names or formulas — and the G1 acceptance cells already name the comparators each
required cell is measured against. What was missing is the measurement that says how far short the
current table falls on the **plain** baseline, where no adversary chooses keys.

The attribution is now measured. With a certified word on every key the canonical table reads at
1.0625 ns/key against the predecessor's 1.0521 — parity, 1.01 — and with `direct_word: None` every
key falls through the spill read, which is where the whole gap sits; the counters show `spill_read`
entering not at all on the resident arm against one sweep step per key on the spill control.
Region indirection is worth 0.74 ns/key, the bit-scan is already a `tzcnt`, and the insert column
is locator reservation rather than the probe. What follows for the port: it is a spill-read port,
and it cannot assume residency — the resident extent is 256 words, so 192 live keys admit an 8-bit
injecting certificate, not a 32-bit one.

The adversarial column was a filter defect, now closed: the group fingerprint read the same low 16
bits as the shard selector, so a population folding those bits matched every lane of every visited
group. The spread top-bits fingerprint (`55421fa`) takes full compares per key from 107.25 to
1.3646 under the starving home, and the raw formula stays in the harness as a live control column.

Plan: [plan.md E6](plan.md), waves W1 attribution, W2 the port, W3 refinement under the extraction,
W4 the gate of record.

### Consolidation decision

| Material | Merge / cut / keep |
|---|---|
| Programme priorities | Put physics/prediction, Paper I descent/ascent and the Null cell, **Whispering Tides**, video games, and world simulation / universe generation first in direction and Song's roadmap. The OS/capability and translation engineering serves these threads. |
| End-state | Retain **supersedes monolithic Linux** as the forward destination. Milestones are plans with design/`anchor` status; only statements about existing implementations carry evidence grades. |
| Authority duplication | Direction owns programme outcomes; Song ROADMAP owns local barriers; DESIGN owns contracts; ARCHITECTURE owns execution/trust; ONTOLOGY owns Song metadata; CLI owns command reference; TESTING owns suites. Track roadmaps retain service/protocol/research detail rather than duplicate system acceptance. |
| Repeated dispatch | Cut the obsolete repeated W4–W8 chain and benchmark recap from direction's execution order. Replace with the existing growable extraction/refinement leg and one link to G1/REG-1. No measurement or requirement is removed. |
| Current versus historical storage | Correct direction to the actual growable flat/spill Rust source. Label fixed-shard preservation theorems historical; do not assign their scope to a different layout. Read manifest pin is `3d66ea3a`, not `88866412`; no package pin changes. |
| Translation / resonance | Merge the programme-level lift and reduction descriptions with named Song A7 children; U1–U5 remain supporting research, not a competing pipeline or a deleted requirement. Reserve all five command names without asserting CLI implementation. |
| Long checkpoint narratives | Keep inherited Song acceptance/checkpoint body byte-for-byte in this slice: it contains active dirty-work receipts as well as history. Add a reading map; fold the older lifecycle record here. Extracting Song's long receipts is a separate preservation-gated task, not an unchecked deletion. |
| Release and ownership | Keep every A0–A6/B6 requirement, local gate, budget, axiom cone and release condition. No bulk ingestion before A0–A4; A7 gates imported-program claims only. Apeiron owns canonical Lean/Rust refinements; Song executes independently, with no Rust/Aeneas runtime linkage. G1/REG-1 stays **OPEN**. |

Whispering Tides was already mandatory in A6 and userspace, but buried below hundreds of checkpoint
lines; research G7 also retained its broader models. Both remain and gain top-level placement.
Networking C3/G5 and observability S1–S7 keep their owners; neither becomes a core dependency just
because games or world simulation might consume it.

### Translation authorities and missing links

Historical Lean was read at Sanguine checkout `e830d3fbdeb40319d1be0d1fc803e706748b6a3c`, under
`../sanguine/proof/` (not the canonical Apeiron package). Canonical combinator statements were read
in `proof/corpus/.lake/packages/apeiron/proof/Apeiron/` at bridge pin
`3d66ea3aa2fe1de5997b6b757247454a2f771358`. No Lean build or new audit ran here.
The seven `L` identifiers below label missing links, not evidence grades or Song registers.

| Authority / exact scope | Required connecting task |
|---|---|
| `Substrate/Algebra/Translation/UniverseSyntaxSerialisation.lean`: `universalLiftCons B U rest := rest.foldl Term.S_routeF (Term.K B U)`; `universalLift_map` commutes with alphabet remapping; `universalLiftCons_state` yields `rest.foldl app (app B U)`. `universalLift` refuses lists shorter than two. | **L1:** domain/effects profile and faithful source interpretation. Modality blindness belongs to the structural fold; it does not supply LLVM parsing or opcode semantics. |
| `Substrate/Ski/StateVectorProjection.lean`: `proj_zero_cons : (h :: t).head? = some h`; `proj_succ_cons : (h :: t).drop (n+1) = t.drop n`. | **L2:** SSA index invariant plus list→Church-state representation and projection refinement. Head-most-recent state requires the `N−1−d` correspondence; the two List equations alone are not it. |
| `Substrate/Algebra/Bulk/UniversalLawGeneration.lean`: `abstractLaw`; `universal_generation : app (abstractLaw x L) arg ≃ subst x arg L` in `Law V`'s SK congruence. | **L3:** historical `Law`/`LiftIR`/`Comb`→canonical term transport, substitution, naming and derived-Iota agreement. Preserve remaining free variables; one bracket does not close an arbitrary open term. |
| `Substrate/Algebra/Translation/PracticalUniversalTranslation.lean`: `automated_lifting_sound` gives equal denotations under `h : SKIShaped app K_val S_val`. Its evaluator composition starts with an already denoted K leaf. | **L4:** inhabited applicative/encoding contract and opcode/environment interpretation, including width, overflow, poison, memory, effects and exhaustion. It is not a theorem about the Rust lifter or LLVM execution. |
| Canonical `Apeiron.Combinator.Comb.bracket_correct` (`Basic.lean`): application of bracket reduces to substitution; `Comb.hoist_preserves_semantics` (`Resonance.lean`) requires `subst i body s = T`; `Apeiron.Combinator.denote_reduces` (`Denotation.lean`) preserves route denotation at real `σ`. The last name is not in namespace `Comb`. | **L5:** source-bound rewrite certificates, scope/freshness, exact structural equality, contextual effects and the selected observation relation. Hoist correctness does not prove XOR parity or finite-machine execution. |
| Canonical `Apeiron.Table.{HMap,get,insertWith,build}`; `AnyRing.{RingCell,ringMul,ringNormForm}` and bridge `word_mul_denote` / `word_norm_denote`. | **L6:** actual associative-store/word-operation refinements and joint Rust–Song correspondence; no identification of arbitrary bytes, opcode dispatch or hash equality with ring multiplication. |
| Song executing source, admitted compiler and native platform | **L7:** Song source binding and composed term→ISA→bytes→ABI/hardware simulation with capabilities, resources and refusal. Keep the Slang empty cone distinct from the Rust/canonical closed cone. |

### Historical Rust analysis

Read with `git -C /home/socol/Workspace/sanguine show '4556df7d^:<path>'`. The parent resolves to
`7d132ffa17fa19b62ff53487d85f254aac108c46`. Paths below are relative to that Git tree, not claims
that deleted `interop/` files exist in the current checkout. The supplied prior-review findings
(parity/hash equality, restricted textual LLVM, byte-result compiler, Linux ABI and linker defects)
are retained, not presented as newly reproduced tests. Other limits below are source-reading results.

**Transfer** means preserve mechanisms/fixtures/specification input, not copy unproved Rust into
Song. **Rewrite** means independently implement the bounded contract; **discard-and-redesign** means
do not retain that mechanism as an acceptance path. No historical executable was run.

| Historical module | Purpose and actual mechanism | Defects / limit | Salvage | Target / missing checked link |
|---|---|---|---|---|
| `interop/src/resonance.rs` | Counts closing parentheses; parses expressions; hashes recurrent subtrees, inserts dependency-ordered hoists; XOR/complement parity rewrite; payload canonicalisation for pattern discovery. | Supplied parity-rewrite concerns and hash-equality without collision confirmation remain. `verify_equivalence` compares loaded terms, excludes `_h_*`, prints mismatches without a failing exit; not a semantic certificate. | Rewrite; transfer pattern/hoist fixtures | A7.3/A7.4; L3/L5/L6, hygienic substitution and a separate word-XOR law. |
| `interop/src/commands/resonate.rs` | File/Comb depth sequences become pitch, PCM/WAV and spectrogram PNG; configurable rendering and bounded/downsampled traversal. | Sonification and structural frequency are diagnostics, not semantic equivalence or physical resonance detection; expression fallback can encode prose. | Transfer diagnostic design; rewrite Song service | A7.3, optional userspace audio/image; L1/L6, explicit metric and encoding. |
| `interop/src/commands/sweep.rs` | Load kernel once, cache fixture loads, project/evaluate bindings, compare truncated structural SHA-256 receipts. | Malformed rows are skipped; only first eight digest bytes compared; shared mutable environment/load order and bounded evaluator determine receipts. | Transfer batch-fixture idea; rewrite strict runner | A7.7; L3/L7, complete population and explicit FUEL/error verdicts. |
| `interop/src/commands/analyze.rs` | CLI dispatch for frequency, patterns, cancellation and verification. | Inherits `resonance.rs`; diagnostic payload normalisation discards distinctions intentionally and cannot justify replacement. | Rewrite thin command interface | A7.3/A7.4; L5, fail-closed verdict propagation. |
| `interop/src/translator_executable.rs` | Textual `.ll` parser; SSA operand expressions/state projections; HP markers; bracket abstraction; single-block and restricted four-block diamond paths, selected calls. | Supplied restricted CFG/skipped-function finding; stale header understates later diamond/call support. Missing SSA lookup falls back to identity; type/effect/UB semantics need a profile. | Rewrite; transfer admitted/refused examples | A7.1/A7.2; L1–L4, whole-module accounting. |
| `interop/src/commands/translate.rs` | Runs executable LLVM lift, then cancellation by default (`--no-collapse` bypass). | A command success can describe a partial source population; inherits both translation and cancellation defects. | Rewrite orchestration | A7.2/A7.4/A7.7; L1/L5/L7. |
| `interop/src/commands/lift_lean.rs` | Invokes `MetaDump.lean`, ingests NDJSON, lifts constants and emits/cancels `.s`. | Export/elaboration/lifter are separate trust steps; per-constant failures and partial emission cannot become complete module acceptance. | Transfer export-schema idea; rewrite adapter | A7.2/A7.6; L1/L3/L4, retained closure and rejection manifest. |
| `interop/src/commands/bracket_abstract.rs` | Small lambda parser; names→de Bruijn indices; invokes shared abstraction; renders SKI. | Restricted grammar and remaining free constants; no executing parser/abstraction refinement. | Transfer fixtures; rewrite via shared Song frontend | A7.2; L3 and canonical `Comb.bracket_correct`. |
| `interop/src/lean_lifter.rs` | Iterative elaborated-Expr walker; registry, prop erasure, Scott/tuple encodings, substitution and SK abstraction, dependency ordering. | Numerals above 65,536 become Iota; default erasure/recursor handling, sanitised-name collisions and unresolved references need refusal. Constructive-shadow redirects need correspondence, not naming. | Rewrite; transfer iterative walk and encoding cases | A7.2/A7.6; L1–L4 and source-linked erasure/codec laws. |
| `interop/src/platonizer.rs` | Line-oriented LLVM renaming into bindings, operation markers and operand strings; branches/phi represented textually. | `Identity`/`Op_Unknown` fallbacks; drops semantic flags/types and makes labels comments. A textual embedding is not an executable CFG. | Transfer diagnostic fixtures only; discard-and-redesign semantic route | A7.1/A7.2; L1/L2/L4. |
| `interop/src/iota.rs` | Boxed SK/Iota trees, Church codecs, parser/environment, bounded reducers, HP-marker recognition and native dispatch, raw syscall bridge. | Reduction cap returns current form; unknown bindings may become Iota; native dispatch adds semantics beyond pure SK. A bounded divergence heuristic is not a proof of divergence. | Rewrite evaluator/projection; transfer reduction/codec fixtures | A7.2/A7.5 and canonical evaluator; L2–L7, explicit FUEL and effect boundary. |
| `interop/src/smt_principia.rs` | Recursive Boolean CNF search, unit propagation, assignment map and depth cap 100. | `simplify_clause` conflates empty conflict with satisfied `None`; two non-SAT branches become UNSAT even when TIMEOUT. No independent UNSAT certificate. Not a general SMT implementation. | Discard-and-redesign verdict path; transfer CNF counterexamples | Song prover/research extension, not a replacement for A0; L6 store relation plus its own SAT/certificate theorem and L7 replay. |
| `interop/src/compiler_gen.rs` | Load environment, parse/evaluate a top-level expression, pass reduced value to `SToNativeCompiler`. | Header's O(1) compilation is not the work performed; byte-result packaging is not general source compilation. | Rewrite composition | A7.5; L4/L7. |
| `interop/src/commands/compile.rs` | Reads source, checks top-level phantom names, calls compiler and writes ELF. | Preflight repairs some unknown-name failures, not full parse/closure/translation validity; inherits byte-result scope. | Transfer refusal fixture; rewrite | A7.5/A3; L1/L7. |
| `interop/src/commands/apply.rs` | Binds arguments before fixture load/inlining; HP projection and environment-aware evaluation. | Input naming/load-order convention and missing/default bindings are not a checked call ABI. | Transfer call fixtures; rewrite | A7.2/A7.6; L2/L4/L7. |
| `interop/src/commands/compositor.rs` | Reads geometry/colours from S bindings; minifb nested display, native Rust pixel/event logic, DRM mode hook. | Hosted display is not native compositor isolation; geometry lifted into bindings does not subsume remaining renderer/input effects. | Transfer interface requirements; rewrite confined services | Song A5 G2/G4, A6 games; L6/L7 and grant/resize/event lifetime. |
| `interop/src/commands/eval.rs` | REPL plus explicit `!` effect route; decodes syscall tuples, performs Linux calls and threads results into continuations. | CString pointer leaks, heuristic argument decoding, raw three-argument syscall ABI and depth cutoff are not a capability contract. | Rewrite; transfer explicit data/effect separation | A7.6, A1/A2/A5; L4/L7, ownership and effect simulation. |
| `interop/src/commands/emit_step_8.rs` | MetaDump constructor masks generate tuple/Scott constructors and field projectors; last data field selected as recovery answer. | Recovering a supplied field is not computing that answer; last-field convention and proposition erasure need explicit semantics. | Transfer schema/projector fixtures; rewrite | A7.2/A7.6; L2–L4, constructive witness and erasure links. |
| `interop/src/commands/ingest_syscalls.rs` | Parses Linux `syscall_64.tbl` and emits names/numbers as Church bit lists. | Number encoding does not specify argument layouts, effects, permissions or compatibility. | Transfer format profile; rewrite generator | A7.6 hosted adapter; L2/L4/L7. |
| `interop/src/commands/project.rs` | Parses expression, optionally applies `Extract_Dynamic_Resonance`, uses environment-less eval, packages through `compile_comb`. | Raw-eval fallback and inherited byte-result scope; no general hardware semantic preservation. | Discard-and-redesign command internals | A7.5; L4/L5/L7. |
| `interop/src/drm_subsume.rs` | Opens `/dev/dri/card0` and drops file descriptor. | Master acquisition, connectors/CRTC, buffer allocation/mapping and modeset are TODOs; printed projection is not those operations. | Discard-and-redesign | A5 native display/A2 devices; L7, actual DRM or native device state/lifetime and authority. |
| `interop/src/s_to_native.rs` | Decodes reduced result bytes; embeds them as data; emits Linux `write(stdout)` and return code. | Supplied byte-result limitation: neither general compiler nor execution of payload bytes. Tests contain vacuous assertions and obsolete call arities. | Transfer byte-output fixture only; discard-and-redesign compiler | A7.5; L7 and nonconstant-input/effect-preserving code generation. |
| `interop/src/elf_linker.rs` | Writes ELF64/PT_LOAD, symbols/relocations and `_start` calling entry then `sys_exit(60)`. | Supplied Linux-loader/ABI dependency. Emits 24-byte RELA-shaped records under REL/16-byte headers; symbols emitted undefined; missing symbol defaults to first index; unchecked displacement narrowing. | Rewrite; transfer layout cases as negative controls | A7.5/A3; L7, relocation/symbol/entry/ISA and boot profile. |

### Subsumption and embedding material

| Historical artifact | Retained purpose / mechanism | Disposition and missing link |
|---|---|---|
| `book/src/06_subsumption/README.md` | Linux subsystem inventory: permissions, VM, interrupts, firewall/sockets, DRM and scheduler; compares instruction/parenthesis counts to small named combinator bindings. | Transfer the subsystem requirements to A1–A7. Counts and projections do not preserve Linux effects, scheduling, MMU/DMA or failure semantics. Keep the supersession destination; redesign the claimed implementation correspondence (L1/L4/L7). |
| `book/src/06_subsumption/new_user_experience.md` | First boot, namespace, actor, compositor, shell and filesystem workflow; small expression stress examples. | Transfer user journeys to A5/A6; rewrite namespace/persistence/authority interfaces. Expression outputs are not boot, filesystem, cleanup or process isolation (L6/L7). |
| `book/src/02_toolchain/lift_lean.md` | MetaDump→LiftIR→S→cancel recipe, encoding registry and constructive shadows. | Transfer reproducibility/closure requirements; do not inherit coverage percentages or one-step runtime claims. Missing erasure, shadow, parser and evaluator refinements L1–L5/L7. |
| `book/src/02_toolchain/universal_lifter.md` | Game-asset/ELF/PE/Mach-O header extraction and entropy-based unknown-format embedding. | Transfer format/admission requirements to A7.6 and game asset services. Header/entropy extraction is not lossless whole-file reconstruction or program semantics; a depth threshold is not a runtime proof (L1/L4/L7). |
| `book/src/00_foundation/vulnerability_subsumption.md` | Proposed divergence→Void containment across temporal, spatial and arithmetic failure. | Retain containment goal, discard-and-redesign the universal zero-result mechanism: name source error, exhaustion and denied authority separately under A0–A4. L4/L7 must connect actual effects and hardware, not identify all errors with zero. |
| `experiments/resonance_compare/{findutils_collapse_tree,findutils_pred_spinor_eval,ripgrep_quantum_traversal,z3_sat_clause_use_list}.lifted.s` | Tagged instructions retain LLVM text inside `Payload`, with repeated `_h_*` hoists. | Transfer as diagnostic/parser and refusal fixtures, not runnable replacements. Require executable SSA/CFG/effects interpretation and exact hoist reconstruction (L1–L5). |
| `experiments/resonance_compare/z3_full_lift.summary.md` | Historical LTO/llvm-dis ingestion report: explicitly records 8,561 skipped bindings and tagged diagnostic output; names cancellation scaling bottleneck. | Keep as historical workload/provenance input, not an executable Z3 port or fresh timing. Reproduce a pinned admitted population and reject skipped obligations before any imported-program claim (A7.7). |
| `nursery/UNIVERSAL_COMPUTATION.md` | Stationary-phase, affine evolution and dimensional/K-branch research programme. | Transfer research questions; no measured constant-time machine bound follows from the displayed continuum formula. A7 must supply the discrete algorithm, domains, cost measure and refinement rather than importing the prose as proof. |

### Conflicts and follow-on documentation

Read scope includes all requested direction/handoff/RFC and Song architecture/design/ontology/CLI,
userspace/network/research/security/TESTING documents. The RFC stays frozen; its historical
performance floor is superseded by direction's unchanged current G1 contract, not silently edited.

| Conflict / duplication | Precise next edit boundary |
|---|---|
| Apeiron handoff, kernel-boundary, README and proof README still call fixed shards executing; later boundary appendices disagree with their openings. | In a ≤3-file task, replace current-layout paragraphs with direction's growable flat/spill description; move fixed-shard API/proof paragraphs under an explicit historical heading. Preserve all exit rows and resource/search premises. Follow with the remaining entry-point documents in a second task. |
| Handoff still dispatches identity/pin creation, while `CoordIdentity.lean`, coordinate extraction wrappers, `3d66ea3a` manifest and growable roots exist. | Replace the old dispatch chain, not the acceptance obligations. Read `proof/extract/lib.rs`, `proof/regen.sh`, `proof/corpus/lakefile.lean` and the active storage plan to identify the next *unaccepted* root/bridge. Re-run freshness only in that serialized implementation lane. |
| `88866412` appears as current pin in older paragraphs. | Use `3d66ea3aa2fe1de5997b6b757247454a2f771358` for the read bridge manifest; retain `88866412` only with its historical run identity. A sibling checkout HEAD is a different identity. |
| Song DESIGN §3.6 describes sibling fixed shards. | Replace its paragraph beginning “The current sibling” with: “The sibling Apeiron Rust table executes a growable flat certified-coordinate resident area plus grouped spill regions. It is an implementation reference, not Song execution evidence; each layout requires its own refinement to canonical `Apeiron.Table`. Song remains independent and must meet its own collision, capacity, authority and performance gates.” |
| Song DESIGN §8.1 and ROADMAP require six pillars; DESIGN §8.2 and AGENTS §4.1 call performance non-pillar. ONTOLOGY also uses five-pillar lifecycle wording. | Owner reconciliation must preserve all inherited A0–A6/B6 measurements and local gates; this task does not adjudicate away a release condition. Consolidate the requirement once in DESIGN, link from other documents. |
| Song CLI says only four verbs live, index is already ported, and no shell/Python executes; `song` is a shell dispatcher with additional verbs and Python `onto`. “Zero runtime dependencies” also hides the hosted OS. | Replace CLI's live-command list from `song` dispatch; separate planned flags/services and A7 reserved verbs; state hosted Linux/assembler/linker dependencies. TESTING's Slang-runner comparison becomes target design, not delivered status; retain suite/budget obligations. |
| Song ROADMAP has obsolete A0 counts and “next” paragraphs beside newer dirty-work receipts. | Extract receipts to one evidence document with revision links only after a removal-control baseline; keep A0.1–A0.10, P1–P6, A1–A6 and B6 text intact until reviewed replacements map every requirement. Do not overwrite the active A0 worker's status. |
| Song ONTOLOGY and AGENTS disagree on merge vocabulary; architecture prose can be mistaken for deployed protection. | Local source/gates decide accepted metadata tokens; keep its resolution Song-local. Retain ARCHITECTURE's Linux-hosted execution and native-isolation obligations; do not import this ontology into Apeiron. |

### Next-leg decomposition

These are proposed bounded tasks, not dispatch authorization where a gate is missing. Each owner
freezes ≤3 actual paths and one completion command before implementation; shared proof caches remain
serialized. No A7 work displaces the existing extraction/refinement leg.

| Order / owner | Bounded output and files | Gate / control |
|---|---|---|
| 1 · Apeiron docs owner | `docs/plan.md`, `docs/kernel-boundary.md`, `proof/README.md`: collapse append-only current/history contradictions and refresh the next unaccepted extraction obligation. | Documentation preservation check plus `git diff --check`; remove a storage/local-gate requirement in scratch and require rejection. |
| 2 · Apeiron/Song docs owners, separate tasks | Apeiron `README.md`/`AGENTS.md`; Song `docs/DESIGN.md`/`docs/CLI.md`/`docs/TESTING.md`, then Song `AGENTS.md`/`docs/ONTOLOGY.md` and checkpoint extraction. | Requirement-preserving documentation gate; explicit owner decision on five/six-pillar conflict, no waived B6 or local measurements. |
| 3 · Existing extraction owner | Resume production prepared-resource and search-decision/apply rooting for `Stored` and `Regen`; select only missing work after reading current roots. Candidate owned paths: `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`; source reshaping, if needed, is a separate task. | Bounded/serialized `proof/regen.sh --check`; source mutation in throwaway copy must break freshness/refinement, restored copy pass. No fresh result recorded here. |
| 4 · Existing refinement owner / Sol | One growable bridge headline at a time under the resource/search premises, with constructor and refusing witnesses; ≤2 bridge modules plus audit. | Bounded/serialized focused Lean gate, then full `lake build` at integration; mutation must refute the named property. Preserve G1/REG-1 and the complete Rust/proof barrier. |
| 5 · Song A7 profile owner | A7.1 admitted LLVM/effects/target profile and a rejecting validator/fixture (≤3 paths assigned at dispatch). | **Missing gate:** create it first; remove an instruction/function/effect and reject. Design can precede A4; bulk ingestion cannot. |
| 6 · Canonical/bridge owners, separate waves | L2 list/Church projection; L3 historical/canonical transport; L4 opcode/applicative witnesses; L5 hoist/parity certificates. One theorem/representation link per task, canonical audit then separately gated bridge pin transport. | **Missing focused gates:** define exact statements and witnesses before assigning proofs; no new axiom or weakened premise. |
| 7 · Song command owners, post-A4 | A7.2–A7.6 source-bound command children, one command and selected fragment per task; then A7.7 vertical demonstration. A5/A6 game/world interface probes continue independently. | Each row's missing positive/refusal/mutation gate first; applicable Song check/test/controls/B6 and native/hosted profile gates at closure. |

### Relocation root integration order

Task `APEIRON-EXTRACT-REP-RELOCATION-BOUNDARY-T1`, owner Astra; supplied-checkout docs-only exception,
base `c450c222905c050f793a23332708c230c658f82c` with pre-existing dirty work. Owned paths:
`docs/kernel-boundary.md` and this file. No branch, worktree, source/proof edits, build lane, commit
or push. Sol exclusively owns `docs/epics/storage-refinement.md`; its verdict precedes implementation.
This decomposes the existing extraction/refinement leg, not a replacement programme or gate set.

```text
SCOPE: docs/kernel-boundary.md; docs/roadmap.md; additive relocation boundary/order.
REGISTER: architecture/planning; no implementation, theorem, pin or runtime changes.
CORPUS: Growable/Search.lean: word_stored_search_decision_simulates and
        word_regen_search_decision_simulates; conditional apply precedent, not relocation.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Growable/Search.lean (read, not rebuilt).
GATE: git diff --check -- docs/kernel-boundary.md docs/roadmap.md;
      cwd repository root; pre-edit exit 0, no documentary red.
      Supplied extraction red: Could not find: type_var_id at Region::initialize loop;
      corroborated in /tmp/opencode/reloc-repro/aeneas.log, not rerun.
GRADE: anchor before/after; documentary consistency only, no new proof/test/measurement.
```

Boundary verdict and exact exclusions are in
[the local boundary](kernel-boundary.md#relocation-root-and-initialized-resource-interface).
One shared production entry mover gains two carrier wrappers; initialization belongs after all
reservations in the adapter. A per-entry proof alone cannot close `rebuild_preserves`.

The following are proposed serialized tasks, not fresh acceptance or authority to edit occupied
paths. IDs use prefix `APEIRON-EXTRACT-REP-RELOCATION-`. Each row has one completion command; controls
run against disposable copies with baseline/red/restore, never by mutating the working tree.

| Order / task / owner model | Owned files and output | One gate (cwd) | Required control |
|---|---|---|---|
| 0 · contract prerequisite · Sol (parallel existing owner) | `docs/epics/storage-refinement.md`: settle initialized-resource premises, entry versus whole-rebuild scope, joint witnesses and composition obligations. No transfer of ownership to Astra/Flash. | Existing contract task's gate; its exact command/result must accompany the handoff before dispatch. | Reject initialization before the last reservation and a postcondition smuggled into the relocation premise. Missing contract verdict blocks the source task. |
| 1 · W1-T1 · Flash | `src/table.rs`, `tests/table.rs`: production calls shared `relocate_entry`; prepare both flat and spill destinations only after reserve-all; retain transaction and geometry. Add actual-path tests for both carriers, source/destination faces, unequal `Sigma`, extent reclassification and refusal effects. | `cargo test --workspace --all-features --locked` (root). | In a disposable copy corrupt moved `Sigma` or value; corresponding readback test must fail. Inject failure at a late reservation and require zero clone/merge/relocation and unchanged table. |
| 2 · W2-T1 · Flash | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: add thin Stored/Regen wrappers and both selection roots, then regenerate. Preserve every existing root/audit and record source/tool hashes. | Bounded `proof/regen.sh` (root), using the wrapper below. | Bounded `proof/regen.sh --check` passes baseline/restored; corrupt a generated relocation write in a disposable copy and require freshness rejection. Strict extraction must accept both carriers and all transitive callees without placeholders or relaxed warnings. |
| 3 · W3-T1 · Sol (theorem/attribution trigger) | `proof/corpus/Growable/Relocation.lean` (new), `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: refine the actual generated entry body for both carriers under the ratified premises, add joint witnesses and audit/register headlines. Keep `SearchDecisionCorrect`; prove the entry postcondition rather than assuming it. | Bounded `lake build Growable.Relocation AxiomAudit` (`proof/corpus/`). | Corrupt a generated relocation value/occupancy write in a disposable proof copy; the named entry-preservation theorem must fail semantically, and restoration pass. A stale-file failure alone does not count as this proof control. |
| 4 · W4-T1 · Flash; Sol if control still passes | `docs/evidence/relocation-controls.md` (new receipt only); no production/proof edits. Run and bind positive, mutation and restored results to the same source/tool digests. | **MISSING aggregate control command:** freeze a bounded runner and exact mutation/theorem assertions in a gate-first task before dispatching this receipt task. | Require both freshness rejection and semantic proof rejection, plus post-reservation initialization/refusal tests; crashes, timeouts, memory exhaustion and unrelated missing modules are inconclusive, not successful controls. |

The gate-first control task owns only a disposable runner under
`/tmp/opencode/apeiron-relocation-controls/`; the lead must freeze its path, command and self-test
before dispatch. It may be prepared after Sol's contract, but the live control/receipt task follows
the proof. Do not invent an existing aggregate harness or mark this row dispatch-ready without it.

**Lane and command discipline.** Source → extraction wrappers/selection → regeneration → proof →
live control is mandatory. Root selection and regeneration share one three-file task because they
form one extraction identity. No proof consumer runs between them. Release `src/table.rs` and
`tests/table.rs` owners before rooting; serialize `target/`, `proof/ApeironWord.lean`, `proof/.lake/`
and `proof/corpus/.lake/`, including disposable controls that reference either package's cache.
For each Lean-backed command above, from its stated cwd use:

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s <command>
```

`<command>` is the row's `proof/regen.sh`, `proof/regen.sh --check`, or `lake build ...`, not a new
executable. Resource-limit failures do not pass a positive or negative gate. Retain default bounded
heartbeats/recursion and the fail-closed `{propext, Classical.choice, Quot.sound}` audit; no Slang
imports. Flash remains default except where a theorem, premise, attribution, grade or ineffective
control triggers Sol; any expanded file/boundary scope returns to Astra.

**Residual and integration stop.** The read scratch success is Stored-only, still reaches
`spill_locate_held` and reports an external-definition warning. It is not strict integrated
extraction, nor evidence of a fully straight-line call graph. Both-carrier rooting and production
reuse are still required. Sol's concurrent `Relocation entry interface` is read before handoff; it
agrees on initialized storage, both roots and the `HMap`/pending-`insertWith` theorem split. Resolve
`RebuildPlanCorrect` clause 5 before proof dispatch: its restored-invariant wording must constrain the
input plan/destination certificate, not assume the actual execution postcondition. Sol owns this
clarification and the source binding of `relocateAll`/`applyPendingAndCommit`; no contract edit is
made here. Moving initialization into the adapter agrees with `PreparedResources`, provided no
initialization occurs before reserve-all. The older full
`rebuild_preserves` requirement cannot be discharged by naming an entry lemma the same thing.
After the entry proof, the next separately scoped Sol task must bind traversal, extent
reclassification, uniqueness/composite `DirectCover`, pending insertion and final commit to the real
rebuild; freeze its exact source-linked statement and focused gate first. Missing extraction of the
whole traversal stays an explicit blocker, not an assumed postcondition or silently enlarged claim.

None of these focused tasks waives the full implementation barrier: all current `AGENTS.md` §6 Rust
commands, bounded `proof/regen.sh --check`, bounded `lake build` in both `proof/` and
`proof/corpus/`, all named mutation controls, source/tool identities, size/perimeter checks and the
unchanged REG-1/G1 requirements remain mandatory on the integrated identity. Existing unrelated
corpus failures require their own owned repair, not removal of targets to make the focused gate pass.
No runtime benchmark or proof gate runs in this documentation task.

**Preservation review.** Relative to this task's dirty pre-edit text, both changes are insertions:
every inherited line remains in order and unchanged. Boundary review retains carrier-as-type, both
faces, all resource/search laws, all theorem/witness requirements, historical evidence, every gate
and command, transaction/refusal order, external-trust exclusions and the two axiom cones. The new
initialization exclusion limits evidence, not the required implementation behavior or eventual
whole-rebuild obligation. The roadmap's inherited tasks, controls and acceptance receipts remain
unchanged; none is attributed to this pass.

### Prepared-resource handoff integration order

Task `APEIRON-EXTRACT-REP-RELOCATION-BOUNDARY-T2`, owner Astra. Supplied-checkout docs-only exception;
base `c450c222905c050f793a23332708c230c658f82c` with pre-existing dirty work. Own only
`docs/kernel-boundary.md` and this file; Sol retains `docs/epics/storage-refinement.md`. No branch,
worktree, source/proof edit, build lane, commit or push. This adds the missing resource handoff after
the existing entry/traversal/commit roots, not a replay or closure of the earlier integration rows.

```text
SCOPE: docs/kernel-boundary.md; docs/roadmap.md; resource boundary and sequencing.
REGISTER: architecture/planning; source, proof, spec, pins and measurements excluded.
CORPUS: Growable/Relocation.lean: four per-entry and two conditional traversal headlines;
        Growable/Refinement.lean: two commit/apply equalities; existing proved scopes, not rerun.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Growable/Relocation.lean;
           /home/socol/Workspace/apeiron/proof/corpus/Growable/Refinement.lean.
GATE: git diff --check -- docs/kernel-boundary.md docs/roadmap.md
      (repository root); pre-edit exit 0, no documentary red.
      Supplied semantic blocker corroborated: missing joint resource root/predicates,
      erased Vec capacity, initialization still in apply_prepared_rebuild.
GRADE: anchor before/after; documentary review, no new proof/test/measurement.
```

The [boundary verdict](kernel-boundary.md#joint-prepared-resource-boundary) selects initialized
storage plus explicit capacity observations, packaged with the plan and resource outcome through a
real production-called extracted handoff. It does not extract reservation/initialization or establish
their simulation. Whole-table `RepR` is a separate Sol decision. Existing single-entry relations and
all eight recorded headlines remain at their current scope; they are regression obligations, not
substitutes for the two missing semantic headlines.

Proposed tasks use prefix `APEIRON-EXTRACT-REP-RESOURCE-`; files marked new are proposed artifacts,
not existing gates. Freeze exact predicates/witnesses and control assertions in W0 before dispatch.
Each implementation row owns at most three files and one completion command. All control mutations
run in disposable copies, requiring baseline green, named rejection and restored green; resource
exhaustion, missing imports or timeout is not a valid negative result.

| Order / task / owner | Owned files and bounded output | One completion gate (cwd) | Required control / stop |
|---|---|---|---|
| 0 / W0-T1 / Sol | `docs/epics/storage-refinement.md`: reconcile the explicit receipt and handoff with `PreparedResources`; separately ratify whole-table representation, joint witness scope and input-side plan predicates. Preserve existing local relations/headlines. | `git diff --check -- docs/epics/storage-refinement.md` (root), plus contract review; not a semantic gate. | Reject a capacity-as-length claim, post-state-as-premise, or model tuple advertised as an actual adapter witness. Contract verdict required before source; Astra does not edit Sol's file. |
| 1 / W1-T1 / Flash | `src/table.rs`, `src/table/resources.rs`, `tests/table.rs`: move hold/initialization after reserve-all into adapter; implement receipt, scalar bound check and production-consumed joint publication helper. Preserve relocation/commit bodies and signatures. Add source-local tests where private adapter observations/failure injection are needed. | `cargo test --workspace --all-features --locked` (root). | Both carriers: materialized vacant lengths/bounds, extent/spill growth, one pending merge; inject the last reservation failure and require unchanged state and zero hold/clone/merge/relocation. In scratch move initialization before that failure or skip a payload initialization; the named tests must fail. No live allocator exhaustion test. |
| 2 / W2-T1 / Flash | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: thin Stored/Regen publication roots and scalar comparison root; preserve all existing selections; strict regeneration and source/tool digests. | Bounded `proof/regen.sh` (root). | Baseline/restored bounded `proof/regen.sh --check`; scratch mutation of the publication success/refusal branch must break freshness. Verify all joint fields survive generation and no allocator/initializer/capacity-read primitive enters the cone. Do not relax audits/warnings. |
| 3 / W3-T1 / Sol | `proof/corpus/Growable/Resources.lean` (new), `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: generated-type resource/plan predicates, explicit joint witnesses and handoff/receipt-comparison lemmas. Distinguish conditional input witnesses from actual adapter reachability. | Bounded `lake build Growable.Resources AxiomAudit` (`proof/corpus/`; target missing until created). | Scratch wrong receipt comparison or swapped/discarded plan/storage/outcome must reject its named equation/witness; success and refusal witnesses cannot both rely on incompatible inputs. No assumed capacity accessor/allocator axiom. |
| 4 / W4-T1 / Sol | `proof/corpus/Growable/Representation.lean` (new), `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: implement separately ratified whole-table relations and joint reachable/base witnesses; connect, do not redefine, single-source observations. | Bounded `lake build Growable.Representation AxiomAudit` (`proof/corpus/`; target missing until created). | Missing live entry, duplicate decoded key and incorrect default/coverage must reject their named witnesses; mere canonical-map agreement is insufficient. Any missing actual constructor/adapter link remains an explicit blocker. |
| 5 / W5-T1 / Sol | `proof/corpus/Growable/Relocation.lean`, `proof/corpus/AxiomAudit.lean`: disjoint flat/spill write frames, unchanged source and preservation of earlier destination observations. | Bounded `lake build Growable.Relocation AxiomAudit` (`proof/corpus/`). | Scratch later write to an earlier destination must break a named frame theorem, not just freshness. No geometry or entry-headline weakening. |
| 6 / W6-T1 / Sol | `proof/corpus/Growable/Relocation.lean`, `proof/corpus/AxiomAudit.lean`: establish `rebuild_preserves` from input certificates, prefix invariant, frames and rooted traversal; complete live-set equality, uniqueness, coverage and no merge. | Bounded `lake build Growable.Relocation AxiomAudit` (`proof/corpus/`). | Scratch dropped/duplicated traversal step or lost occupancy must refute the semantic headline. An arbitrary-invariant traversal theorem alone does not pass. |
| 7 / W7-T1 / Sol | `proof/corpus/Growable/Refinement.lean`, `proof/corpus/AxiomAudit.lean`: `prepared_rebuild_refines_insertWith`, composing relocation with the actual pending commit under search/carrier/callback laws and the resource handoff. | Bounded `lake build Growable.Refinement AxiomAudit` (`proof/corpus/`). | Scratch wrong pending value/occupancy/len or omitted commit must refute the semantic headline; commutative word addition is not a rejecting merge-order control. Retain generic new-first obligations separately. |

These build commands are prospective focused gates, **not yet semantic acceptance**: W0 must name
the declarations/witnesses audited in each row, and each row must register its new declarations before
build exit 0 can close it. No empty module or previously green build satisfies a row. A row exceeding
the file/size limit, requiring a new primitive model, or overlapping another owner stops for a new
bounded task. Source already exceeds the owned-source line budget; this plan does not waive that
existing integration blocker or authorize an unrelated source split.

**Serialization.** W0 → source/tests → extraction roots and regeneration → resource predicates →
whole-table predicates → frames → relocation composition → pending composition. Only one Sol task
runs at a time, including the current spec owner; release its ownership before the next Sol dispatch.
Release `src/table.rs`/resource/tests and `target/` before extraction. `proof/ApeironWord.lean`,
`proof/.lake/` and `proof/corpus/.lake/` are exclusive shared lanes; no proof task consumes a partially
regenerated file and no control shares either live cache concurrently. Every Lean-backed command
uses the existing `flock`/`systemd-run`/8G/no-swap/single-thread/900s wrapper above, with bounded
heartbeats/recursion. A lane or tool failure blocks acceptance.

The existing `proof/controls/relocation.sh` remains required: invoke standalone because it owns the
lane lock; `RELOCATION_CONTROL_HEAVY=1 bash proof/controls/relocation.sh` includes its semantic build.
Its old spill-tag control does not test the new resource handoff, frames or composition; new controls
must accompany those tasks. No new aggregate control runner is claimed to exist. If durable runner
changes are needed beyond a row's three files, assign a separate Flash gate-first task before closing
that row; Sol adjudicates any control that still passes its mutation.

**Preservation and residuals.** This task inserts text only in both dirty pre-edit documents; every
inherited line remains unchanged and in order. Line-by-line review retains all prior commands/gates,
option-(b) transaction requirements, initialized-storage clause, all roots and theorem obligations,
both axiom cones, trust exclusions and REG-1/G1. Clarifying that already-generated spill routines
remain generated does not grant them a generic correctness theorem or root the composite decision.
The source initialization mismatch and absent capacity correspondence are exposed, not waived.
Neither initialized input witnesses nor a packaging root prove planner/allocator/initializer
realization. Full Rust/freshness/both-package builds, all old and new controls, identity/digests,
perimeter/size and REG-1 remain required on the integrated identity. No software gate or runtime
measurement runs in this docs-only task.

**Documentation acceptance for this task:** scoped `git diff --check -- docs/kernel-boundary.md
docs/roadmap.md` and repository-wide `git diff --check` exit 0. The insert-only patch
and both inserted sections were reviewed line by line; no inherited requirement was removed or
weakened. The following documentation acceptance section is an earlier task's receipt, not this run.

### Resource premise to whole-table derivation

Task `APEIRON-EXTRACT-REP-RESOURCE-BOUNDARY-T3`, owner Astra. Supplied-checkout docs-only exception;
base `c450c222905c050f793a23332708c230c658f82c` with pre-existing dirty work. Own only
`docs/kernel-boundary.md` and this file; Sol retains the contract/proof lane. No source/proof/spec
edit, branch, worktree, build lane, commit or push.

```text
SCOPE: docs/kernel-boundary.md; docs/roadmap.md; named premise and derivation order.
REGISTER: architecture/trust-boundary/planning; source, proof, spec, pins and measurements excluded.
CORPUS: Growable/Resources.lean: preparedResources_of_publication concludes PreparedResources_R
        under htargets, hcapacity, hmaterialized and hheld; supplied W3 proved status, not rerun.
        Growable/Search.lean: SearchDecisionCorrect is an explicit Prop premise, not a proof of it.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Growable/Resources.lean;
           /home/socol/Workspace/apeiron/proof/corpus/Growable/Search.lean (read).
GATE: git diff --check -- docs/kernel-boundary.md docs/roadmap.md
      (repository root); pre-edit exit 0, no documentary red; line-by-line preservation review.
GRADE: anchor before/after; no new mathematical proof, implementation test or runtime measurement.
```

The [premise record](kernel-boundary.md#resource-adapter-simulation-premise) approves explicit,
unproved `ResourceAdapterSimulates`, following `SearchDecisionCorrect`. The supplied W3 build/audit
receipt is 3472 jobs; the read resource module supplies predicates and conditional publication
transport, not source-linked adapter execution. W4 adds the conditional reachable witness without
changing those headlines. It does not close unconditional adapter realization or REG-1/G1.

**Current derivation order:** W4 resources/premise/witness → W5 whole-table representation → W6
disjoint-write frames then `rebuild_preserves` → W7 pending composition. This is a sequencing update
to the earlier proposed resource rows, not deletion of any artifact, gate or control: the earlier
W4 representation work maps to W5; earlier W5 frames and W6 composition are separate serialized
tasks within W6. W0–W3 retain their obligations and revision-bound evidence. Stale “missing” status
in earlier checkpoint text is not current absence evidence for `Growable/Resources.lean`.

IDs below use prefix `APEIRON-EXTRACT-REP-RESOURCE-`. Sol owns all proof/contract decisions;
**one Sol task at a time**, including the concurrent spec owner. Finish/release W4 before W5;
W6-T1 and W6-T2 own the same files sequentially, never in parallel. Sol must ratify the separate
whole-table contract in its contract lane before W5 implementation; this docs task cannot supply or
weaken that contract. Each row retains at most three owned files and one completion command.

| Order / task / owner | Owned files and output | One completion gate (`proof/corpus/`) | Required control / stop |
|---|---|---|---|
| W4-T1 / Sol (current premise/witness task) | `docs/epics/storage-refinement.md`, `proof/corpus/Growable/Resources.lean`, `proof/corpus/AxiomAudit.lean`: resource clause obligations, explicit `ResourceAdapterSimulates`, joint success/refusal witnesses and reachable publication witness conditional on it. Resource module registration already exists; preserve its four theorems. | Bounded `lake build Growable.Resources AxiomAudit`. | Wrong scalar comparison or swapped/discarded joint field rejects the named equation/witness in scratch; refusal must not yield a success witness. Review that every reachable conclusion retains the simulation premise, no invented execution relation is presented as source-linked, and no post-state conclusion is assumed. Actual adapter/late-reservation tests remain separately required. |
| W5-T1 / Sol | `proof/corpus/Growable/Representation.lean` (new), `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: whole-table `RepR`/`ReservedRepR`, complete live-set denotation, decoded uniqueness, default, lengths and composite `DirectCover`; connect existing observations and joint base/reachable witnesses. Retain W4's explicit premise wherever adapter reachability is used. | Bounded `lake build Growable.Representation AxiomAudit` (prospective target). | Missing live entry, duplicate decoded key or incorrect default/coverage rejects its named witness; no map-only substitute. An unlinked actual constructor remains an explicit blocker, not discharged by the conditional adapter witness. |
| W6-T1 / Sol | `proof/corpus/Growable/Relocation.lean`, `proof/corpus/AxiomAudit.lean`: disjoint flat/spill write frames, unchanged source, preservation of all earlier destination observations. | Bounded `lake build Growable.Relocation AxiomAudit`. | A later write corrupting an earlier destination must reject the named frame theorem semantically; freshness failure alone is insufficient. Preserve both-carrier entry premises and geometry. |
| W6-T2 / Sol, after W6-T1 | `proof/corpus/Growable/Relocation.lean`, `proof/corpus/AxiomAudit.lean`: `rebuild_preserves` from input-side plan/resources, whole-table relations, frames, prefix invariant and actual rooted traversal; complete live set, uniqueness, coverage and no merge. | Bounded `lake build Growable.Relocation AxiomAudit`. | Dropped/duplicated traversal step or lost occupancy rejects the semantic headline; a generic arbitrary-invariant traversal theorem does not close it. No resulting-state invariant may enter the plan premise. |
| W7-T1 / Sol | `proof/corpus/Growable/Refinement.lean`, `proof/corpus/AxiomAudit.lean`: `prepared_rebuild_refines_insertWith`, composing `rebuild_preserves` with the actual pending commit, symbolic pending certificate/decision connection, search/carrier/callback laws and conditional resource handoff. | Bounded `lake build Growable.Refinement AxiomAudit`. | Wrong pending value/occupancy/len or omitted commit rejects the semantic headline; retain generic new-first obligations and a rejecting noncommutative merge-order control, not commutative word addition. Unextracted orchestration/source linkage remains explicit. |

Every Lean gate uses the existing `flock`/`systemd-run`/8G/no-swap/single-thread/900s wrapper above,
bounded heartbeats/recursion and fail-closed `AxiomAudit`. Register exact declarations/witnesses
before acceptance; an empty new module or previously green build is not completion. Controls use
throwaway baseline/red/restore copies, reject the named semantic defect and share no active cache.
Timeout, memory exhaustion, missing imports or tool/lane failures are inconclusive, never valid red
controls. Freeze missing control commands in separate bounded gate-first tasks; the existing
`RELOCATION_CONTROL_HEAVY=1 bash proof/controls/relocation.sh` remains required and runs standalone
because it owns the lock. It does not replace the resource/frame/composition controls.

**Discharge and regression decision.** Keep `ResourceAdapterSimulates` as a standing explicit
premise for W4–W7. A small extra extraction root cannot prove capacity/allocation correspondence or
the real planner/reserver/initializer execution. A separately approved adapter extraction/refinement
with checked primitives, or a source-linked adapter theorem, must discharge those obligations;
no allocator axiom or tests promoted to proof. This is not a waiver of eventual realization.
The eight existing relocation/traversal/commit and four resource theorem statements stay unchanged.
Any regeneration requires freshness, bounded builds/audits and controls revalidating all twelve,
their predicates and dependencies on the same source/tool digests. No proof acceptance is recorded
here. All inherited Rust, both-package builds, size/trust inventories, controls and REG-1/G1
requirements remain mandatory; W4–W7 focused gates do not replace that barrier.

**Preservation review.** This task adds this section and the linked boundary section only; every
dirty pre-edit line is retained unchanged and in order. Review retains all commands/gates, root
selections, initialized-storage/refusal order, source-linked witness requirements, prior boundaries,
callback/search laws, both axiom cones and the full integration barrier. Updated numbering preserves
each prior task's output/control; conditional reachability is labelled conditional, never substituted
for unconditional source-linked realization. Sol's exact premise definition/audit and whole-table
contract verdict remain separate acceptance obligations.

### Pending composition retry and semantic adjudication

Task `APEIRON-EXTRACT-REP-RESOURCE-BOUNDARY-T4`, owner Astra. Supplied-checkout docs-only exception;
base `c450c222905c050f793a23332708c230c658f82c`, preserving pre-existing dirty work. Own only
`docs/kernel-boundary.md` and this file; no source/proof/spec edit, build lane, branch, worktree,
commit or push. Sol retains the concurrent premise-definition lane.

```text
SCOPE: docs/kernel-boundary.md; docs/roadmap.md; additive premise boundary and retry.
REGISTER: architecture/trust-boundary/planning; source, proof, spec, pins and measurements excluded.
CORPUS: SearchDecisionCorrect/FirstFit; RebuildPlanCorrect_R/ResourceAdapterSimulates;
        word_{stored,regen}_rebuild_preserves (recorded proved, not rebuilt here).
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Growable/{Search,Resources,Rebuild}.lean;
           /home/socol/Workspace/apeiron/src/table.rs (read).
GATE: git diff --check -- docs/kernel-boundary.md docs/roadmap.md
      (repository root); pre-edit exit 0; line-by-line preservation review.
      Supplied W7 blocker corroborated in interfaces; no Lean failure rerun.
GRADE: anchor before/after; no new proof, implementation test or runtime measurement.
```

The [premise boundary](kernel-boundary.md#pending-composition-premise-boundary) permits explicit
`DecisionSymbolicCorrespondence`, `QueryAddressCorrect`, `FreshLengthBound` and
`GrowableCallbackLaws` for conditional core refinement only. Their realization remains required for
unconditional production claims. `SearchDecisionCorrect` alone supplies neither query-routed
vacancy, symbolic/runtime agreement, fresh-length headroom nor an arbitrary-merge realization.

**Checkpoint correction, not a waived gate.** The traversal-preservation obligation is now supplied
by `Apeiron.GrowableRebuild.word_stored_rebuild_preserves` and `word_regen_rebuild_preserves` in
`proof/corpus/Growable/Rebuild.lean`, not a missing declaration literally named
`GrowableRelocation.rebuild_preserves`. Their recorded build/control evidence is not rerun here.
The semantic-adjudication gate is **OPEN** because W7 and its controls are missing; the older
“MISSING until both headlines exist” wording must not restart completed relocation work or accept
its old-map result as the pending canonical write. The aggregate adjudication/control command is
still **MISSING**; a green existing focused build is not that command.

**Serialized retry order.** These are bounded follow-ups, not permission to overlap the current Sol
owner. Each proof row registers its declarations in `AxiomAudit.lean`; freeze exact statement,
witnesses and mutations at dispatch. Release the previous file owner and shared lane first.

| Order / owner | Files and output | One focused completion command / stop |
|---|---|---|
| 1 / Sol, current premise task | `proof/corpus/Growable/Search.lean`, `proof/corpus/Growable/Resources.lean`, `proof/corpus/AxiomAudit.lean`: the four pinned premises and projections, no postcondition-as-premise. Freeze exact names/types and joint witness obligations. | Bounded `lake build Growable.Search Growable.Resources AxiomAudit` in `proof/corpus/`; definition/projection acceptance only, not premise discharge or W7. |
| 2 / Sol, contract reconciliation | `docs/epics/storage-refinement.md` only, separately assigned to its owner: reconcile the older no-extra-premise composition statement and semantic-gate wording with the exact interfaces. Preserve all success/refusal/domain requirements. | `git diff --check -- docs/epics/storage-refinement.md` in root plus Sol contract verdict; this docs-only gate proves no semantics. This task does not edit that file. |
| 3 / Sol, concrete discharge/witness slice | `proof/corpus/Growable/Search.lean`, `proof/corpus/Growable/Resources.lean`, `proof/corpus/AxiomAudit.lean`: attempt the already-generated wrapping-add callback instance first; establish joint found/fresh flat/spill witnesses for both carriers or return the exact incompatible condition. No allocator/search realization claim. | Bounded `lake build Growable.Search Growable.Resources AxiomAudit`; if callback law scope exceeds the concrete dictionaries or witnesses cannot be built, report that blocker rather than assume their existence. |
| 4 / Sol, W7 retry | At most three files: actual composition module, `proof/corpus/AxiomAudit.lean`, and `proof/corpus/lakefile.lean` only if registration is needed. Compose landed rebuild preservation, decision correspondence, query addressing, strict fresh bound, carrier/hold and callback laws with rooted pending commit; prove full `RepR` at new-first `insertWith` for both carriers. | Existing bounded `lake build Growable.Refinement AxiomAudit` remains required. Before dispatch choose the acyclic composition module/target: `Growable.Resources` already depends through `Growable.Representation` on `Growable.Refinement`, so importing Resources/Rebuild back into Refinement is forbidden. If a new module is needed, freeze its path and include its target in the single bounded build command; that exact command is **MISSING** until assigned. No broad module move is authorized. |
| 5 / Flash gate-first; Sol reviews semantic rejection | At most three owned runner/receipt paths, frozen before dispatch: implement the missing adjudication/control command with named theorem/witness assertions and baseline/red/restore copies. Then a separate execution/receipt task runs it on the W7 identity. | **MISSING command/path:** lead must assign and self-test the runner before control acceptance. Keep the existing standalone `RELOCATION_CONTROL_HEAVY=1 bash proof/controls/relocation.sh`; it is not the new composite gate. |

**Semantic-adjudication condition set (conjunction, not alternatives):**

1. Both landed rebuild headlines and both carrier W7 instances exist, are registered/audited, and
   build under the bounded closed-cone gate on one fresh source/tool identity. W7 must prove the
   actual extracted pending commit succeeds and establishes complete `RepR` of
   `Apeiron.Table.insertWith`, including unequal-key observations, live set, uniqueness, occupancy,
   address/`DirectCover`, directory and length. Do not replace this by readback at the pending key,
   an arbitrary-invariant traversal, a commit equality alone or an assumed successful post-state.
2. Sol ratifies the exact premise set and joint domain. `ResourceAdapterSimulates`,
   `PreparedResources_R`, `RebuildPlanCorrect_R`, `SearchDecisionCorrect`, the four new names and
   carrier laws remain visible wherever used; no implicit global witness discharges them. Check
   found/fresh flat/spill joint witnesses for both carriers, including symbolic correspondence and
   fresh headroom. A stricter proof premise is not a silently reduced public admission contract.
3. Source-linked call equations use the actual traversal, relocated storage and rooted pending
   commit. Exclude malformed/exhausted prepared decisions via explicit input certificates; prove
   found length unchanged and fresh length incremented once without overflow under the exact bound.
   Relocation uses no merge; pending merge is new-first exactly once. Keep generic callback
   obligations separate from the wrapping-add instance and its commutativity.
4. Baseline/red/restore controls reject dropped/duplicated relocation, lost occupancy, wrong pending
   value, wrong fresh length and omitted final commit, plus wrong-shard/direct-index and mismatched
   symbolic/runtime decisions. Invalid-input controls must reject a joint certificate/witness, not
   count a theorem legitimately accepting only valid inputs as defective. Retain a noncommutative
   merge-order control for the generic requirement; reversed wrapping addition is not rejecting
   evidence. Timeouts, cap failures, missing imports and freshness-only failures do not substitute
   for semantic rejection.
5. Carry refusal separately: retain `insert_err_stutters`/`resource_adapter_simulates`, the explicit
   `ResourceAdapterSimulates.refused` assumptions, and actual-path overflow/late-reservation tests
   with an early-hold/clone mutation. State unchanged table, zero clone/merge/relocation and no
   partial build. Prepared-success W7 cannot discharge these effects. A conditional core verdict
   records their unproved realization and tested scope; it cannot close the whole-operation gate.
6. Preserve old roots/headlines, audits, controls and freshness; record source/tool digests and
   exact theorem premises. Full Rust, both-package bounded builds, all controls, size/trust
   inventories and REG-1/G1 remain required. Conditional semantic adjudication does not close that
   barrier, allocator/search realization or an end-to-end claim.

Every Lean command retains the earlier lock/scope/8G/no-swap/single-thread/900s wrapper, bounded
heartbeats/recursion and cache serialization. The existing relocation runner owns its lock and runs
standalone. No existing command is replaced by a newly named but nonexistent aggregate gate.

**Preservation review and remaining work.** The T4 patch inserts these two sections only; every
dirty pre-edit line remains unchanged and in order. Line-by-line review retains all prior gates,
commands, root selections, contract requirements, bounds, refusal ordering, both axiom cones and
trust exclusions. Explicit W7 premises limit the new conditional claim, not the required production
behavior; the old no-extra-premise spec needs its owner's reconciliation, not silent precedence.
No proved headline or grade changes. Remaining work is Sol's interface/contract verdict, the concrete
callback instance and joint witnesses, acyclic W7 composition, missing control command, and eventual
source-linked premise/refusal discharge. No mathematical proof or runtime measurement runs here.

T4 documentation acceptance: scoped `git diff --check -- docs/kernel-boundary.md
docs/roadmap.md` and repository-wide `git diff --check` exit 0. Both insertions were
read back line by line; no inherited gate, command, requirement or boundary statement is removed or
weakened. The following documentation acceptance section is the earlier consolidation receipt, not
this task's evidence.

### Documentation acceptance

The scratch checker freezes the entire inherited Song acceptance body (§§1–2) and direction's
G1/REG-1 section from the dirty pre-edit snapshot. It rejects changes/removals, exercises individual
nonblank-line removal controls in memory, and hashes tracked/nonignored-untracked files in both
repositories to reject edits outside the three owned documents. It is a limited documentary check,
not semantic verification or a permanent project gate. Runtime/Lean/Kani gates are unrun and
inapplicable to this docs-only change; every implementation obligation retains them.

Acceptance command: `python3 /tmp/opencode/roadmap_review.py check`, followed by `git diff --check`
in each repository. The final handoff records outcomes; no milestone is marked closed by this pass.

Observed: checker exit 0, **617/617** requirement-removal controls reject, restoration passes,
**11** new local Markdown links/anchors resolve, and only the three owned documents differ from the
770-file dirty baseline. Both repository-wide `git diff --check` commands exit 0. Checker SHA-256:
`25d0b6438509e1a972d705f1a4c316e41d497ba4d35ca519ee2366e888cdbe1e`; baseline SHA-256:
`80962d5daefe053a46b169d2b7d5150637c5ea4f1c47f42acaf7f9760f77e852`.
Residual: the conflict table's unedited documents still need sequenced owner reconciliation;
historical transfer requires all seven missing links, not a bulk Rust port. The scratch checks do
not inventory ignored caches or establish mathematical, implementation or runtime correctness.

## Security-algebra rebase (staged)

The old Sanguine corpus holds a proved security/cryptography module and a decentralised federation
substrate written before the standardized trichotomy ("Being cell / Null cell / `x = B/U`"). It is
captured, **not actioned**: `docs/epics/security-algebra-rebase.md` and
`docs/epics/security-algebra-rebase.md`. The transfer is required so the programme does not re-derive
existing results and does not keep the old fused "Null cell" (OTP-grade secrecy is the σ>0 split
face; the one-way/hash wall is the σ=0 parabolic face).

Staged waves: S0 Sol ratifies the map and grades; R1–R5 transport `NullCell`/`EncryptionProjection`,
`SanguineSecurity`/`SecurityCore`/`WebCipher`, the cipher/hash fork, `Navigation/Federation`, and
`MandalaMemory`/`RaceFreedom`/`CapProvenance`; D1 rebases `theory/*.md` and `experiments/*`; P1
states the corpus's minimum-encryption standard (cost-based secrecy rejected, split-channel and
collapse ends admitted) as a canonical headline. Deeper-review spikes cover the SHA-256d reading,
the message-schedule localisation, the AES/RSA bridge the corpus names but does not re-prove, and
the federation security core. The corpus's own per-file grades are authoritative; the rebase
transports them and presupposes neither direction. It exists to make the corpus's cryptographic
claims decidable, with a confirm/refute pair per load-bearing claim.

The historical Sanguine federation/networking lineage remains recorded in
`../song/docs/network/ROADMAP.md`; the "Federation / distributed" row in `docs/direction.md` is its
current owner.

<details>
<summary>Earlier consolidation lifecycle and acceptance receipts (historical)</summary>

## Intent

The user directs the programme; AI writes its code. Commit authors and optional trailers do not
partition that direction. The approved decision retains Lean and consolidates canonical mathematics,
reusable Rust primitives and implementation refinements in Apeiron. Song remains an adjacent,
independent assembly-up implementation. Correspondence between implemented primitives is required;
Rust linkage or replacing Lean is not the method. Song is not runtime-linked to Apeiron; the
required cross-repo result is a checked primitive correspondence. Fold is not a consumer or
requirements source; applications retain local policy.

Reuse `docs/direction.md` as the programme authority. Do not create a second programme roadmap.
Local roadmaps retain implementation detail, gates and release acceptance. A programme outcome may
be required without being privileged, an immediate task, or a prerequisite for every local release.

Rejected alternatives: retiring Lean loses a retained asset and contradicts the user; merging Song
into the Rust implementation loses its independent assembly-up boundary; making every programme
outcome an A0 prerequisite prevents bounded foundational work; maintaining duplicate master ledgers
recreates the current drift.

## Specification

- Retain Lean sources, packages, history and attribution. Current source locations and Git/Lake
  pins remain unchanged until a separately gated migration.
- Apeiron owns canonical definitions and reusable Rust/refinement work. Song owns its assembly,
  Slang implementation, contracts and local build/runtime acceptance. Applications retain policy.
- A primitive correspondence relates both executing implementations to the same canonical
  definition on matched domains, including representation, state/effects, refusal and resources.
  Equal names, example outputs, citations or identical layouts are not that proof.
- Preserve programme outcomes for algebraic computation, combinators, lift, checked subsumption,
  realization, measured transformation benefit, active test-time learning, physical models and
  federation. Scope/gate gaps remain open; no new software completion is claimed.
- Preserve Song A0-A6/B6, local Rust/Lean gates, axiom cones, controls, existing evidence and bounds.
  Foundational verification precedes the next large AI implementation. The joint readiness profile
  is an explicit specification obligation, not satisfied by A0 alone.
- Remove stale current-direction assertions, not mathematical source. Historical RFC publication
  artifacts stay unchanged; their entry page points to the current programme authority.

This is an approved documentation pass on the supplied checkouts. No task branch/worktree is
created, no source/build lane is occupied, and no commit or push is authorized. Temporary checking
code lives under `/tmp/opencode/programme-docs/`, not on a production acceptance path. This combined
intent/spec/plan is the one retained lifecycle record for the pass.

## Superseded re-root order

The former nine-stage source-arena/memo-store sequence is not current programme authority.
Source binding remains a later execution refinement of the same primitive under the
[handoff](plan.md#slang-source-binding-exit-all-rows-required), not another foundation.

## Retired route

The `Slang/Derivation/` `Layer`/`Derives` tower is retired. The `Slang/Generated/Trace/` generated
decoder traces are retired. The `DecodeTrace.Run` representation is retired, the named proof DAG is
retired, and the segment lemma is retired; none is the active architecture. The binned
`tools/slang/trace_check.py` `Run`/`Derives` acceptance is retired. Historical references to a
retired route carry a retirement marker and are not acceptance. The handoff governs their
replacement; they are not extended.

## Input identities

| Checkout | Base revision | Pre-existing work to preserve |
|---|---|---|
| `apeiron` | `1403767e671c81737525702946a3eb1eae4f0b0b` | Clean |
| `sanguine-apeiron` | `8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f` | Clean; shares Git history with `sanguine` |
| `sanguine` | `e830d3fbdeb40319d1be0d1fc803e706748b6a3c` | Clean; predecessor worktree, not canonical root |
| `song` | `47869e76f982cfa3085a8618a82b80463e22ef46` | Clean; `song-ref` remains untouched |
| `penglai` | `ed11d3b34c148eb35050b615dc47e8340a9e224f` | Clean |
| `whispering-tides` | `e62ae9bca358bacb9fc5e042830f6a5059442c7d` | Clean |
| `fold` | `d497161f32e0d366220f2065ec36e680bd9aed86` | Modified `modelfold/src/arch.rs`, `lib.rs`; untracked CQ source/examples |
| `lyrium` | `665ccfe1da85e6b5e8d312cb324cc61211ca1640` | Modified `src/lib.rs`, `main.rs`, `tools/mod.rs`, `ui/render.rs`; untracked `.cargo/`, `examples/vfsreal.rs`, `src/render/` |

Existing unrelated branches/worktrees are inventoried, not removed. The baseline checker records
working-source digests, including nonignored untracked files, rather than attributing dirty inputs
to HEAD alone.

## Ownership and waves

Task IDs use `APEIRON-PROGRAMME-` followed by the table key. Paths are relative to the workspace.
All lanes are documentation-only. One owner per file; the lead alone maintains this plan.

| Task | Owner | Owned files | Completion |
|---|---|---|---|
| W0-T1 | Lead | `apeiron/docs/roadmap.md` | Intent/spec/plan recorded; `git diff --check` |
| W0-T2 | Flash gate worker | `/tmp/opencode/programme-docs/check.py`, `baseline.json` | Checker built, snapshot captured, expected red baseline observed |
| W1-T1 | Flash central worker; lead architecture review | `apeiron/docs/direction.md`, `docs/kernel-boundary.md`, `docs/plan.md` | Central authority, outcomes and local ownership consistent |
| W2-T1 | Flash Apeiron worker; Sol claim review | `apeiron/AGENTS.md`, `README.md`, `proof/README.md` | Entry points and current API/evidence agree with source |
| W2-T2 | Flash corpus worker; Sol claim review | `sanguine-apeiron/AGENTS.md`, `theory/ROADMAP.md`, `theory/POTENTIAL_CLAIMS.md` | Lean retained; current delivery/primitive correspondence distinguished |
| W2-T3 | Flash Song roadmap worker | `song/AGENTS.md`, `docs/ROADMAP.md`, `docs/research/ROADMAP.md` | Programme obligations coexist with unchanged local release barriers |
| W2-T4 | Flash Song architecture worker | `song/docs/DESIGN.md`, `docs/ARCHITECTURE.md`, `plans/ROADMAP.md` | Independent assembly-up boundary; obsolete Lean-retirement goals superseded |
| W2-T5 | Flash predecessor worker | `sanguine/AGENTS.md`, `fold/AGENTS.md`, `lyrium/AGENTS.md` | Current authority pointers, local autonomy and historical scope explicit |
| W2-T6 | Flash Penglai worker | `penglai/AGENTS.md`, `docs/substrate-port-scope.md` | Programme linkage; gameplay/adapter ownership and local gates preserved |
| W2-T7 | Flash application/network worker | `whispering-tides/AGENTS.md`, `ROADMAP.md`, `song/docs/network/ROADMAP.md` | Application science and federation remain programme outcomes, not core privilege |
| W2-T8 | Flash RFC worker | `apeiron/docs/rfcs/0001-core-architecture/README.md`, `ownership.md` | Frozen proposal clearly separate from approved programme; attribution corrected |
| W3-T1 | Lead | `apeiron/docs/roadmap.md` | Combined checks, review and preservation evidence recorded |

The central wording lands before the disjoint propagation tasks. W2 may run in parallel. Checker
snapshots and final evidence publication serialize; no worker writes another task's file or this
plan. A newly required path is sequenced and recorded before editing.

## Arc

```text
SCOPE:      The owned documentation above; one programme authority with independent local implementation gates.
REGISTER:   Planning and documentation; excludes implementation, proof promotion, source migration and publication.
CORPUS:     Governing documents and named combinator/table sources read; no theorem establishes repository policy.
REFERENCE:  /home/socol/Workspace/apeiron/AGENTS.md; /home/socol/Workspace/song/docs/ROADMAP.md
GATE:       python3 /tmp/opencode/programme-docs/check.py --task <ID> --controls; initially MISSING, W0-T2 creates it.
GRADE:      Architecture decision and checked documentation consistency; no new software or performance evidence.
```

Each worker emits this arc specialized to its owned files and reads its actual red result before
editing. W0-T1 uses `git diff --check`; W0-T2 uses the checker's self-test plus expected-red task
baseline. All later tasks use the single command above; final integration uses `--task all`.

## Documentation gate

The temporary checker captures HEAD, worktree files/digests and original owned text before W1.
It checks exact write allowlists, preservation of nonowned files and source/tool identities,
newly broken local Markdown links/anchors, retained Song checklists and A0/B6 rows, and per-task
programme authority/outcome requirements. Existing broken historical links are recorded separately,
not silently promoted or rewritten. Directory links and line anchors are distinguished from headings.

Controls operate on scratch document copies, never working sources: remove the authority pointer or
a required outcome, remove a retained acceptance requirement, and introduce the opposite boundary
(retire Lean or require Song to execute through Rust). Baseline and restoration must pass; each
intended mutation must fail for its named reason. These checks establish limited documentary
consistency, not semantic truth of arbitrary prose. Lead review remains required.

The central outcome ledger must cover ownership, consumer, dependencies, existing evidence, missing
specification/gate, and completion conditions for each outcome. It is not a dispatch list: missing
gates and profiles require bounded design tasks before software implementation.

Sol reviews storage/benchmark scope, F2 delivery and stale combinator/evaluator statements before
acceptance. No grade, premise, theorem or attribution is inferred from a title. Astra's recorded
boundary retains all local gates. Runtime/Lean rebuilds are inapplicable to this documentation-only
pass: no code, extraction, package, pin or theorem changes are allowed. A software change would
require a separate task and its local barrier.

## Bounded authorization decision core (staged)

A Cedar-class authorization core: forbid wins, fail-closed Kleene unknowns, and a 64-bit rights lattice with
`allow`/`attenuate`/`ask`/`deny` verdicts. It has one Rust body, one canonical Lean definition and an analysis
layer on `Apeiron.Decider` (σ = 1 face). Epic: [`epics/policy-engine.md`](epics/policy-engine.md).
Study: [`study/policy-engines.md`](study/policy-engines.md). Exit rows: `docs/plan.md`, "Policy engine exit".

| Stage | Content | State |
|---|---|---|
| W0 | Spikes S1–S7 (fragment fitness, Cedar alignment, laws, Song slice, cedar-lean rebuild, analysis, relationships, compiler) | closed (spike scope) |
| W1 | Canonical core + laws; Rust core; registration | closed: `proved` / `tested` |
| W2 | Extraction; Cedar oracle crate; analysis layer | closed: `tested` / `tested` / `proved` |
| W3 | Refinement `word_decide_denote` (proposed split: atom/guard file, then the decide loops) | next; not dispatched |
| W4 | Song attenuation stage `policy-attenuation` (empty cone) | open |
| W5 | Intent seam adoption (cross-repo) | open |
| later | Relationship module on the Table adjacency face with certified budgets (D2); compiler in the extraction fragment (D3); `revocation_is_forbid` (D1); multi-bit and `all`/`any` analysis | open; spike evidence only |

The `docs/direction.md` paragraph (a new programme boundary) is drafted in the epic and is not applied: it is an owner decision.

## Physical computation bounds (open)

Four physical constraints on execution are named in design discussion and carry no artifact today.
They are not programme outcomes and do not enter the verified-AI sequence; they are the local
mathematical delivery of the retained-Lean physics foundation, dispatched because each has a
provable counting or real-analytic core. Lifecycle:
[intent](epics/physical-computation-bounds.md) / [spec](epics/physical-computation-bounds.md) /
[plan](epics/physical-computation-bounds.md).

| Absent item | What is missing | Corpus target (first pass) | Status |
|---|---|---|---|
| Holographic state capacity | no counting core for a boundary-scaled capacity; Bekenstein `A/4` exists only in the predecessor | `Apeiron/Physics/Capacity.lean` | green + audited (`APEIRON-PHYS-W1-T4`, W2) |
| Causal metric locality | no access-time or light-cone lower bound; `Projective/Length.lean` is elliptic only | `Apeiron/Projective/Latency.lean` | green + audited (`APEIRON-PHYS-W1-T3`, W2) |
| Action-bounded transitions | no first-orthogonality time; `Trichotomy/Flow.lean` has the flow, not the clock | `Apeiron/Physics/ActionClock.lean` | green + audited (`APEIRON-PHYS-W1-T2`, W2) |
| Non-equilibrium dissipation | no contraction accounting; fuel is a monotone counter, not a potential, `kT` appears nowhere | `Apeiron/Physics/Landauer.lean` | green + audited (`APEIRON-PHYS-W1-T1`, W2) |

Song executes the resource half already (fixed `FUEL_PASSES`/`NODE_CAP`/`STACK_CAP`, `FUEL` as a
third honest verdict); the thermodynamic reservoir half has no artifact in this tree, the corpus or
Song, and no SI constant enters. The physical identifications stay `anchor`; the kernel checks the
cores. The plan's portable inventory lists the predecessor items from `../sanguine` and the current
`Apeiron/Physics` modules that a later re-root wave may consume; predecessor modules are historical
until re-derived on the trichotomy.

```text
SCOPE:      four new corpus modules for the physical-computation-bounds epic; first-pass dispatch
            and the programme-level pointer; completion is the epic's barrier in its plan.
REGISTER:   planning + local algebra/analysis obligation; excludes Rust, extraction, Song,
            measurement and any completed physical claim.
CORPUS:     Apeiron/Trichotomy/Flow.lean flow_neg_one / flow_add / flow_returns_iff_disc_neg;
            Apeiron/Projective/Length.lean arcLength_vertical / log_height_le_arcLength, all
            corpus-native proved; the four absent statements have no corpus declaration.
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/{Trichotomy/Flow,Projective/Length,
            Physics/CarrierDissipationAnyRing}.lean;
            /home/socol/Workspace/sanguine/proof/Substrate/Algebra/{Gravity/BlackHoleThermodynamics,
            Matter/ThermalStatistics,NullCell/NullCellCoding}.lean (read this pass).
GATE:       per module, bounded lake build in the corpus; audit lines and linters are W2.
            Baseline: every target module is absent (lake: unknown target).
GRADE:      planning/anchor; before: four absent items with no target; after: the four targets and
            the dispatch, no theorem claimed yet.
```

First-pass workers were `deepseek-flash` by user direction; they ran in the supplied corpus checkout
under the recorded documentation/dispatch exception (warm `.lake` cannot be duplicated per task
within the §10 budget) and edited only their declared new module. The first pass landed: four
modules green with red-then-restored controls, W2 `#audit` registration in
`proof/Apeiron/Audit.lean`, the corpus barrier `lake build` 3530 jobs exit 0, and
`scripts/check-style.py` / `check-imports.py` / `check-lints.py` clean. No commit or push; the
corpus pin does not move. W3 (predecessor re-root) remains open.

## Acceptance record

### R0 / REG-1 correction

`APEIRON-R0-RESTORE-W1-T1`, owner Astra, uses the supplied-checkout documentation exception at
base `c450c222905c050f793a23332708c230c658f82c`; no source/build lane is occupied.

```text
SCOPE: docs/direction.md, docs/plan.md, docs/roadmap.md;
       restore RFC 0001 topology and REG-1
REGISTER: planning; no implementation, proof, runtime or universal-speed claim
CORPUS: docs/rfcs/0001-core-architecture/{topology.json,RFC.md}; REG-1 at RFC.md:196-209
REFERENCE: c7b3a7c (numeric gate), a9360d4 (authority consolidation),
           99955af (narrow O(1) rewrite)
GATE: git diff --check; RFC renderer --check; consistency search across current authorities
GRADE: planning correction only; restores the required measurement gate, no measurement claimed
```

The correction restores one R0 chain and required G1/REG-1 acceptance in direction and handoff;
it removes the competing active-looking source-arena order here. S8/G1 is OPEN, the all-row
REG-1 command is MISSING, and the audited cross-revision connecting theorem remains missing.
The immediate task is the single comparison gate, not source binding or a separate semantic project.
Old constant-loop/evaluator-speed anchors are not promoted. RFC status and generated artifacts,
source, proof and pins are unchanged; no publication or joint RFC approval is asserted.
The two pre-existing untracked spinor plans remain outside this task. No commit or push is allowed.

Documentation checks: scoped `git diff --check` exits 0; RFC `render.py --check` exits 0,
`Checked 6 local artifacts; publication remains blocked.` The required consistency search exits 0;
the stale no-performance-closure search returns no matches (exit 1). No runtime gate is run.

### Earlier consolidation acceptance

Historical record of the documentation-consolidation pass. The canonical-rooting directive in
`docs/direction.md` and `docs/plan.md` now governs.

- Architecture decision: recorded by `openai-astra`, session `ses_f66026973ffejJuS8pXwtdbGkM`.
- Preparation: `git diff --check` exits 0. Checker unit checks exit 0; lead reran the
  pre-edit `--task all` baseline: exit 1, 50 documentary findings. Live controls were not ready
  until the relevant documents were green; synthetic detector tests are not acceptance controls.
- Baseline: `/tmp/opencode/programme-docs/baseline.json`, sha256
  `399382c1d0ced58c4c024495a01cbcc15157064b7a65f7001a41537d92bb8150`; eight repositories,
  4,811 files, 26 owned documents, 100 unique pre-existing broken-link keys. Do not recapture.
- Sol claim wording approved in `ses_f65dc7229ffeTavtqB83fwFrVS`: current fixed-shard table
  witnesses are not universal preservation/subsumption; capped certificates are not uncapped
  `CertOp` completeness; `normalize`/`verdict` exist, with exactness only on decided outcomes.
  No current-table performance or Rust-Slang runtime correspondence is claimed.
- Central task W1-T1: lead read all three outputs, corrected unsupported absence/identity claims
  and separated scientific, application and learning acceptance. The final focused command exits 0:
  zero findings, two live baseline/red/restore controls. This opened W2 propagation on disjoint files.
- Propagation tasks pass their focused commands after review corrections. Each command is
  `python3 /tmp/opencode/programme-docs/check.py --task APEIRON-PROGRAMME-<task> --controls`,
  run from the Apeiron checkout:

| Task | Initial findings / exit | Accepted findings / exit | Live controls passed |
|---|---|---|---|
| W2-T1 | 4 / 1 | 0 / 0 | 1/1 |
| W2-T2 | 5 / 1 | 0 / 0 | 1/1 |
| W2-T3 | 3 / 1 | 0 / 0 | 2/2 |
| W2-T4 | 5 / 1 | 0 / 0 | 2/2 |
| W2-T5 | 6 / 1 | 0 / 0 | 1/1 |
| W2-T6 | 2 / 1 | 0 / 0 | 1/1 |
| W2-T7 | 5 / 1 | 0 / 0 | 1/1 |
| W2-T8 | 2 / 1 | 0 / 0 | 1/1 |

- Final integration: `python3 /tmp/opencode/programme-docs/check.py --task all --controls`
  exits 0, zero findings, 5/5 live baseline/red/restore controls. The mutations remove a programme
  pointer, a required outcome or a retained Song requirement, or introduce Lean deletion or Song
  runtime linkage to Rust. All fail for their named reason; restoration passes.
- Lead read the diffs and checked the affected source. Sol approves the corrected theorem/refinement
  wording in `ses_f65dc7229ffeTavtqB83fwFrVS`; Astra approves the documentation architecture in
  `ses_f66026973ffejJuS8pXwtdbGkM`. Neither verdict claims fresh software acceptance.
- Review corrections restore the capped-certificate existential, distinguish a bounded reduct from
  a normal form, scope the Engeler singleton limitation rather than applying it to weighted routes,
  and retain the checked table witnesses without claiming universal preservation. Architecture
  corrections retain the secure OS-language-proof outcome, independent learning and scientific
  acceptance, local engine ownership, and complete Penglai coverage obligations.
- `git diff --check` exits 0 in all eight checkouts. `git status --short` confirms only the 24
  intended tracked documents plus the new plan differ from the recorded dirty baseline. The other
  edited document, `song/plans/ROADMAP.md`, is already ignored and is checked directly, not included
  in a Git diff. Its 275 original checkboxes remain; the current Song roadmap retains all 49
  checkbox lines, 34 A0 rows and four B6 rows.
- Source/pin/generated-artifact preservation passes against the immutable baseline: all eight HEADs
  and every captured nonowned file remain unchanged, including the dirty Fold/Lyrium source and
  unchanged RFC/publication artifacts. The baseline records tracked and nonignored untracked files;
  it is not an inventory of ignored build caches. No build lane was used.
- The baseline contained 100 unique pre-existing broken-link keys. No newly broken local Markdown
  link or anchor is introduced; this is not a claim that all historical links are repaired. Review
  also replaced the old Song design links to deleted Map source and benchmarks with the current
  `apeiron/src/table.rs` reference.
- Checker sha256: `92e8ff0fbc8aac095b55d49b3fb3db06047f6b262ec7cfd2663954954134b44a`.
  The baseline hash above is unchanged. The ignored historical Song plan's final sha256 is
  `909bae260644bf2b8c9e9db967fbeb8edfb5cdc7a7bb9a222a123030ca222ddf`.
- Decision: documentation consolidation accepted, 26 documents across eight repositories. This is
  checked documentary consistency and a reviewed architecture decision, not a new `proved`,
  software-`tested` or performance-`measured` claim. Rust/Lean/Kani barriers are inapplicable to this
  source-preserving documentation pass; implementation epics retain all their required gates.
- No commit, push or physical migration in this pass.

## Remaining implementation (historical)

Historical. This section records the state before the canonical-rooting directive. Current order
is in direction and handoff, not this record. The joint primitive/readiness profile, canonical
source migration and joint correspondence below remain open, not closed.

The next bounded design task specifies the joint primitive/readiness profile: selected executing
primitives, their domains and representations, success/refusal/effects, resource bounds, exact
theorem links and rejecting gates. Existing Apeiron universal table preservation/admission work and
Song's local foundation continue under their own plans. Canonical source migration needs a separate
identity-preserving package task. None of these implementation obligations is closed by this pass,
and the next large AI implementation remains behind joint foundational readiness.

</details>
