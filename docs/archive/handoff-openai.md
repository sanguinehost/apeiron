# Handoff: continue Apeiron on the OpenAI stack

Current dispatch: [open-area ledger](../epics/open-areas.md),
[build-cost optimization](../epics/build-cost-optimization.md) for G1,
[read/classifier continuation](../epics/read-proof.md) with its lane execution status,
and [assurance factoring](../epics/assurance-factoring.md).
G1 W1-T1–W1-T4, W2-T1–W2-T3 are closed at `tested`: worker-recorded placement locations replaced the
per-key rebase probe/walk, the directory owns one recycled staged-path scratch, and the
found-or-linked reservation removed the per-key probe walk. The extraction
regenerated to `proof/ApeironWord.lean` `2a1138e4…` earlier; the extraction was regenerated again
after W1-T4/W2-T3/W2-T1 and `proof/regen.sh` plus `--check` exit 0 at the new identity `8e76b917…`,
with `Growable/Refinement.lean` repaired for the new found-update body. Workspace
`cargo test --workspace --all-features --locked` exit 0 and `cargo fmt --check` exit 0; `clippy
--all-targets` exit 0 after fixing `examples/reg1/performance.rs` `too_many_lines`. The other lane's
`Kepler.Certificate` builds green, and the package-wide bridge and `AxiomAudit` re-run is green
after the repair (`lake build` 3817 jobs; `AxiomAudit` 3693 jobs). The changed-crate
suites (`--lib`, `native_locator`, `parallel`) are green; the workspace barrier is green at the
current identity.
Read lane: the marker chain, all four scan primitives and the inner/outer/tail sweep coverage are
checked and registered, including the readiness derivations `inner_ready_of_scan_geometry` and
`outer_step_ready_of_geometry` and the logical-absence composition
(`spill_read_none_implies_no_full_match`, `spill_read_some_implies_full_match`), so row 5 closes
conditionally on the traversal certificate; publication preservation is covered by the live
tag/address invariant chain through commit and relocation. Row 6 first slice: `Growable/Read.lean`
proves the constructor-reachable direct-hit partition at both public roots
(`word_stored_table_get_at_direct_hit`, `word_regen_table_get_at_direct_hit`, registered), with
partitions 2-5 and the paired control open; the localization core
(`query_rep`, `absent_of_exclusions`, `default_of_absent`, `Read.lean` `0f8752e4…`) is landed and
the tail case-split finding is recorded in the plan. Partition 2 shapes A/B/C and flat index pinning landed in `Read.lean` `1eb64f08…` (`W5-T13-CONT5`,
Gemini Flash) with the mandatory mutation control passing (0/1/0, semantic rejection on 7/5 mutations).
Partition 3 occupied non-match landed in `Read.lean` `cb3bfa49…` (`W5-T13-CONT6`, Gemini Flash)
with all 22 declarations in `AxiomAudit.lean` (3696 jobs green). Residual: P5.
The two open contract defects are decided by lead direction: repair structurally and prove out — the
`hstep` locator well-formedness predicate is closed (W5-T14: `Growable/RelocationWF.lean` `3d0f9897…`
lands the 64-bit width premise, edge/residue/identity/directory loops, `linked_handle_succeeds`
and `retarget_held_hstep_of_wf` with the mutation control 0/1/0, all 15 declarations registered in
`AxiomAudit.lean` at 3696 jobs green) and a satisfiable `RelocationEffectsPrepared.publish`
premise; no weakening, no relabel.
Row 7 interface landed (`Growable/Search.lean` `a52502ad…`): `absenceCovered` is the three-way
`AbsenceCertificate` with the `SummaryMissSound` mirror and route bridges; the
`Growable/Classifier.lean` producer proves the five bridges (`fd16f901…`) and the generated-decision
provenance (`search_decision_provenance`, `search_decision_wellFormed`,
`search_decision_found_flat_guards`; `ffbdb84a…`, gate 3350 jobs, cone clean, control 0/1/0, live
tree untouched). `SpillLocateFoundFullMatch` is a statement only; `vacantExact`/`exhaustedExact`/
`complete`, first fit, `SpillVacantDirectGuard`, `entryAt` packaging and `SearchDecisionCorrect`
production remain open.
The package-wide bridge and
`AxiomAudit` are green at the regenerated identity `8e76b917…` (`lake build` 3817 jobs;
`AxiomAudit` 3696 jobs with the Classifier, `Growable.RelocationWF`, and
`GrowableRead.spill_read_none_of_summary_miss` registrations, no errors, closed cone). Integrator note: the `RelocationWF` root addition to `lakefile.lean` left three stray
backticks (`:30`, two builds exited 1 at 0.2 s); repaired, and the roots line is integrator-owned
from here. Address lane:
`held_address_agrees` and `held_address_joint_witness` are proved under the closed cone, with a
generated-address mutation control. Security interpretation witnesses are
grade `model`; the allocator handback is blocked. All seven size breaches pass the unchanged
assurance checker ([receipt](../evidence/assurance-factoring.md)). The full REG-1 re-run at the W2-T1 build identity
completes in 316.0 s below the 600-second ceiling with all 55 control rejections expected; strict
acceptance exited 1 (pass=308 / fail=1142, `concurrency_unmatched=2`, `address_face_unmatched=4`,
`full_acceptance=false`), so G1 remains open; the earlier 477.79 s / 325-passing profile and the
17-pair delta have no per-pair attribution yet
([receipt](../evidence/locator-reserve-quadratic-and-reg1-attribution.md)). Universal read,
representation, classifier and allocator correspondence remain open. Kani remains retired.
Previous continuation: [owned tasks](openai-continuation-plan.md) and
[fresh evidence](openai-continuation-evidence.md). The original receipts below are historical.
The initial rerun found `proof/regen.sh --check` **red**, including executable constant/consumer
drift. Regeneration and a proof-body repair now pass freshness and the 3482-job bridge, with
rejecting drift/shard-bit controls; see the continuation receipt for source digests and scope.
Kani is retired by user direction; do not run it or retry the old timeout harnesses.
[Retirement plan](../epics/kani-retirement.md) supersedes all earlier Kani preservation and gate
requirements. The missing universal read, representation and allocator proofs remain open.

You are the lead orchestrator. You do not implement directly; you decompose, sequence, dispatch,
and close. You are picking up a Rust + Lean repository at a **green-but-open** checkpoint and driving
it to the next wave barrier. Read this whole document before acting.

---

## 0 · Standing rules (do not rediscover)

- **One in-progress owner per file.** Every task declares its owned files; tasks in one wave touch
  disjoint files. If a task needs a file another owns, sequence it.
- **A task is ≤3 files, one gate, one dedicated negative control** where soundness is at stake, and
  is closed by a command exit status, never by an agent saying so.
- **Parallel branches** are allowed only when file sets are disjoint and no shared build product is
  contested. Shared lanes (serialize):
  `flock /tmp/opencode/apeiron-lane.lock`, `proof/ApeironWord.lean`, `target/`,
  `proof/.lake/`, `proof/corpus/.lake/`.
- **Lean runs are bounded and serialized**: `systemd-run --user --scope -p MemoryMax=<cap>
  -p MemorySwapMax=0` with `LEAN_NUM_THREADS=1` and a `timeout`, on the lane lock. `set_option
  maxHeartbeats 0`, whole-source `decide`/`rfl`, and raised `maxRecDepth` are **forbidden** in
  committed proofs and in agent runs. A job that exhausts the cap is a failed run, not a red gate.
- **Do not commit, push, branch, worktree, stash, reset, checkout, restore or clean** without
  explicit lead authorization. The checkpoint is committed; accepted continuation and retirement changes are uncommitted. Preserve them.
- **A mutation control edits a throwaway copy, never the working tree.**
- **Never generate URLs** except to retrieve a cited paper/repo for a comparison; retrieve it
  locally, read it, cite the exact file/revision/theorem.
- The sibling `song`/`song-ref` tree is a different repository; its ontology, `R1`–`R5` registers,
  `//@` semantics and `./song` gates **do not apply here**.
- **Worker lanes and escalation** (per `AGENTS.md` §9.1): **Gemini Flash** (`gemini-flash`, Vertex)
  is the default high-throughput worker (~300 tok/s); **Xiaomi MiMo Pro** (`xiaomi-pro`) is the
  higher-ceiling fallback for tasks where Gemini fails or proof search binds; **DeepSeek Flash**
  (`deepseek-flash`) is the conserving speed lane. Escalation is by artifact: an unresolved proof or
  architecture obligation goes to a fresh session on an independent verifier model with user approval;
  the Lean kernel and mutation controls arbitrate; the OpenAI agents (Sol, Astra, Terra) are
  decommissioned with their credit. Contract or claim-grade questions go to the lead.

---

## 1 · Repository and reading order

- Repo: `/home/socol/Workspace/apeiron` (Rust crate + Lean extraction + Lean bridge).
- Implementation checkpoint: `dbc1bc8` — *storage: axiom-free extraction, green bridge, bounded Kani probe fixtures*.
  Handoff HEAD: `262bb0a94eff2cc7d5655122a954f44d74de96b6`; its only change is this document.
  Working tree **dirty** with accepted continuation and Kani-retirement changes; HEAD alone does not identify the evidence.
- Canonical corpus (pinned, do not move): `/home/socol/Workspace/sanguine-apeiron` at
  `3d66ea3aa2fe1de5997b6b757247454a2f771358`. `proof/corpus` depends on it.
- Read, in order, before acting: `AGENTS.md` → `docs/direction.md` → `docs/kernel-boundary.md` →
  `docs/flash-handoff.md` → `docs/roadmap.md` →
  `docs/evidence/kani-table-harness-audit.md` → `docs/extraction-requirements.md` →
  `proof/README.md`. Source and gates beat documents; fix documents that disagree.

Recorded host: `i9-14900KF`, 32 threads, 62 GiB RAM. Check free space and cache availability before dispatch; use the documented bounds even with warm caches.

---

## 2 · Environment and pins

- Charon Rust: `nightly-2026-08-18`. Ordinary Cargo rerun: rustc
  `1.97.0-nightly (17584a181 2026-04-13)`. Lean: `v4.31.0`.
- Charon `0.1.254 (b104e24f)`, charon-driver `207b223a`, Aeneas `505b6ca-dirty`.
- `AENEAS_HOME` unset → `target/aeneas`, a **symlink to another checkout**; never mutate it.
  The patched backend lives at `target/aeneas-model/lean` and is built by `proof/model-patches.sh`
  from `proof/patches/aeneas-lean-model.patch` (sha256
  `210e56e1ed54bfd4b91e12a102e7dceb17c429b7f14903a176c27b1c0b23b191`). `proof/lakefile.lean`
  requires it. A fresh clone must run `proof/model-patches.sh` before `lake build`.
- Historical Kani receipts used `0.67.0` / CBMC `6.8.0`; neither tool is required.
- `/tmp` is tmpfs; put disposable proof/cache copies on workspace disk and check space before copying.

---

## 3 · Accepted evidence (with grade)

- **Rust: `tested`.** Fresh `cargo test --workspace --all-features --locked` after factoring: 263 passed, one ignored, exit 0; source manifest and logs in `target/open-areas/final-rust/`.
  `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` rc 0;
  `cargo fmt --all -- --check` rc 0.
- **Extraction: `proved` cone, fresh.** `proof/regen.sh` rc 0 and `proof/regen.sh --check` rc 0.
  `proof/ApeironWord.lean` is 10974 lines, **0 `axiom`/`sorry`/`admit`**, sha256
  `5599fe7d22685de05fa8349b727f852ebf5cb70308a1bb67b95f21128d3e9d40`, from **43 roots** including
  the write/growth path (`*_table_build_at`, `*_prepared_shard_apply`, `*_apply_insert`,
  `*_relocate_*`, `*_apply_pending_and_commit`, `*_publish_prepared`, `word_capacity_bound_holds`,
  `word_schedule_is_complete`). `lake build` in `proof/`: 1708 jobs rc 0.
- **Bridge: builds green, some statements conditional.** `lake build` in `proof/corpus/`: 3493 jobs
  rc 0; `AxiomAudit` reached under `{propext, Classical.choice, Quot.sound}`. The historical
  fixed-shard set (`TableRefinement*`, `Joint.TableInstance*`) is **retired from the roots** (files
  kept on disk). Relocation/rebuild headlines use `PayloadStateEq` and require
  `TraversalEffectsPrepared`.
- **Historical Kani receipt: 58/67 verified, 9 timeouts, no counterexamples.** This receipt is
  retired. The probe cases now run as five ordinary Rust tests with real-source mutation controls;
  Kani harnesses and alternate geometry are removed. Production geometry remains `8`/`256`/`255`.
- **Security track: specification and attribution reviewed.** Finite observation witnesses and canonical transport remain unproved (see §5·G).

---

## 4 · Level-1 acceptance reference (exact commands)

Rust (serialize with `nice -n 10` + `timeout`):

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
```

Kani is not an acceptance command.

Proof regeneration (needs Aeneas/Charon under `target/aeneas` or `AENEAS_HOME`):

```sh
proof/model-patches.sh          # build target/aeneas-model/lean (patched backend)
proof/regen.sh                  # extract, strip modeled axiom stubs, audit, build
proof/regen.sh --check          # fail on drift from the committed ApeironWord.lean
```

Bridge build:

```sh
cd proof/corpus && lake build
```

Barrier set = all of the above plus the named controls, on one source/tool identity.

---

## 5 · Open obligations, ranked, with the branch that owns each

### A · Kani retirement — source removed, ordinary regressions retained

The user retires Kani from development and acceptance. The nine historical whole-Table timeouts
are not converted to passes and are no longer retry or closure obligations. Retain useful concrete
probe regressions as ordinary Rust tests, with semantic sweep controls. They do not establish
public-constructor reachability, canonical `get` or universal `SearchDecisionCorrect`.
The source and coverage cleanup is accepted; see [retirement evidence](../evidence/kani-retirement.md).

### B · The missing universal read-path links (the real hole) — Branch B
No bridge theorem proves the generated matcher returns the represented key's location or sound
absence. `word_{stored,regen}_search_decision_simulates`
(`proof/corpus/Growable/Search.lean:338,397`) conclude about `apply_get` and take
`SearchDecisionCorrect` as a premise. Canonical `get` is `m k` (`Apeiron/Table/Basic.lean:69`).
Named missing goals (in `docs/epics/read-path-specialization.md`):
- `Growable/Scan.lean`: source-linked summary/fingerprint exclusion facts.
- `Growable/Representation.lean`: representation facts and their construction/preservation.
- Downstream `Growable.ReadLoop`/`Growable.Read` modules: prove generated `spill_read` and both
  `word_{stored,regen}_table_get_at` roots refine canonical `get`; a separate classifier module
  establishes `SearchDecisionCorrect`, including `spill_locate`. Keep the existing Search interface
  acyclic; exact task signatures and remaining producer obligations are in the continuation map.
Do **not** manufacture the read result by assuming the desired classifier postcondition.

### C · Extraction/bridge premises — Branch C
`TraversalEffectsPrepared` is a premise the extracted planner/resource path must establish.
Generated reservation, materialization and preparation roots exist; their semantic links are missing.
`SearchDecisionCorrect` and `ResourceAdapterSimulates` are unproved. `core.mem.size_of` was replaced
by a documented budget constant (`ASSUMED_SLOT_BYTES = 32`), not modelled — a layout assumption.

### D · Allocator model soundness

[Allocator intent](../epics/allocator-model.md), [spec](../epics/allocator-model.md) and
[plan](../epics/allocator-model.md) separate capacity preservation from allocation effects.
The active deterministic model loses reserved capacity through push/extend and other vector updates.
Sol ratified a bounded repository-patch/probe/runner repair for push/extend only. The candidate
backend builds, but three probe goals remain; it is preserved unintegrated under
`target/open-areas/allocator-capacity-w3-t1/`. Mutation/restoration acceptance was not reached. The installed backend is not a mutation target.
Ordinary allocation may-failure, type-layout transport and effect/WP correspondence remain open.
Keep `ResourceAdapterSimulates.refused` whole-state equality: replacing it requires checked
first-use/initialized refusal traces and preservation of future routing, not a weaker declaration.

### E · Performance gate `G1`/`REG-1` — Branch E
Strict acceptance harness `examples/reg1/` plus `examples/micro_spill.rs`, `examples/micro_spatial.rs`.
Contract: every applicable `(cell × incumbent)` pair `<= 1.0`, never behind; stretch `<= 0.8`;
incumbents `hashtrinity::{Table,CoordMap,FreqMap}` (`4cc14471`), `hashbrown+{foldhash,fxhash,rayon}`,
`std`, par16, `xxHash`, `wyhash`, `ahash`, `fnv`; `blake3`/`sha256` informational. A number is a
measurement only with a named command, machine, metric and interval, before and after.

The [fresh full-profile attempt](../evidence/open-area-performance.md) exited 124 at 600.015 seconds,
before any pair table or acceptance headline. No new passing/failing pair counts or speed comparison
are available. Diagnose phase duration before another campaign; keep the full population and ceiling.

### F · Storage epic waves — Branch F
Intent/spec/plan already exist for `algebra-native-addressing`, `parallel-stable-scatter`,
`read-path-specialization`, `spinor-*`. Implement the next wave's tasks (≤3 files, one gate, one
control each). Do not start an epic without a written intent, spec and plan.

### G · Security/algebra rebase

[Intent](../epics/security-algebra-rebase.md), [spec](../epics/security-algebra-rebase.md) and
[plan](../epics/security-algebra-rebase.md) now distinguish the source theorems from cryptographic
interpretations. Unique compatible keys do not establish secrecy; generic collisions do not force
one sigma value. A finite `ZMod 2` witness candidate is preserved unintegrated under
`target/open-areas/APEIRON-SECURITY-W2-T1/`; a wrapper setup error prevented its Lean gate from
starting. The next task checks it and its uniform-mass negative control. No absent canonical security target is treated as proved.
Canonical substrate/migration and independent Song correspondence retain their own prerequisites.

### H · Documentation and assurance

Claim references resolve and all seven original size breaches pass unchanged limits. Eleven
generated table shards preserve the frozen source bytes and public declaration bodies; both audit
moves retain their registered declarations and pass empty-cone controls. See
[the factoring evidence](../evidence/assurance-factoring.md). Concurrent benchmark instrumentation
after the recorded Rust run is outside that receipt's source identity.

## 6 · Current parallel dispatch

The [open-area ledger](../epics/open-areas.md) supersedes the original branch table. Disjoint source
preparation runs concurrently; Rust/Lean builds and measurements lease the shared lane serially.
The [factoring plan](../epics/assurance-factoring.md) records the completed shard barriers and audit moves;
final extraction,3493-job bridge and tokenizer/parsing controls pass.
The allocator and security plans name their exact next artifacts. No Kani task, assumed read result,
weakened refusal premise or unconditional W7 closure is authorized.

**Current batch.** Apeiron core: the read composition carries concrete fixtures —
`Growable/ReadComposition.lean` (`70d90c21…`, 737 lines, root in `lakefile.lean` `d92071db…`, gate
3354 jobs, 19 kernel `#guard` pairs, control 0/1/0); the six route theorems and the root admit no
concrete instantiation because no concrete `ConstructorReachableRepR` value exists (named blocker;
row 4/8 work). `Growable/Relocation.lean` is `24a8eeb6…`: `RelocationEffectsPrepared.publish` repaired
at the prepared spill facts (added `reserved`; premises `sourcePayload`, `address_held`,
`reserved.regions[shard]`), four consumer conclusions unchanged, gate 3344 jobs, both controls red and
restored, `Growable.Read`/`ReadComposition`/`RelocationWF`/`Refinement`/`Rebuild` and `AxiomAudit`
green (3880 jobs). The independent verifier returned **accept with residual**: premises satisfiable,
non-circular and pinned to the prepared facts, conclusions preserved, gates re-run, its own
region-index mutation red at `Relocation.lean:520:33`/`635:33` and restored to `24a8eeb6…`; the
stale `Growable/Read.lean:30` note was corrected in the same lead pass (`Growable/Read.lean` now
`311db56531fd7cac2bde6251454a828471f9b6d6df26223816201a31fa0cc946`, gate 3880 jobs). Neural stage 2: Wave 1 exits green (seven adaptations, vendored
tree 48 files, Wave-1 controls T1–T6 pass, barrier 4005 jobs) and Wave 2 is partially in
(FrontierLadder and SpinorFold vendored, compile green, controls pass, 15 headlines registered); the
archived transformer composite was split; the tree is frozen reference material and Wave 2 claims no
exit.
**AI-track re-anchor (lead direction):** the vendored neural port is reference material only; the
track is specified and planned — intent `docs/epics/ai-track-reanchor.md` (`63d35906…`), spec
`docs/epics/ai-track-reanchor.md` (`57d0b141…`), plan `docs/epics/ai-track-reanchor.md` (`592edc39…`).
The library and root namespace is `Nous` (νοῦς; Anaxagoras and Aristotle; homage, not a claim) under
`proof/corpus/Nous/`, `lean_lib Nous` in `proof/corpus/lakefile.lean` (`336ba4b4…`). Wave 1 closed
after the rename pass: `Nous/Core/Shape.lean` `bd7c07b3208e5fbb51a2310a06a7bd88068145b2a3346ad5ae8ebd5b862baf69`
(147 lines, gate 2 jobs) and `Nous/Model/Weighting.lean`
`f81a72656d7ee6c25274c3f362304efd9af313c50287ba25ab86f9a258121fd7` (155 lines, gate 2974 jobs), lib
gate `lake build Nous` 2976 jobs; controls re-run on the renamed digests in
`/tmp/opencode/ai-w1-t2-control/proof/corpus` (baseline 0 / shape red 1 / weighting red 1 / restore 0),
`ApeironWord.lean` still `8e76b917…`. Wave 2 closed: tensors
`38bb2f5a52084622998af438e031f71730b61be97ee36984a4ef75a9a3a46525` (73 lines, gate 112 jobs),
token state `63247287a76d16f30c5aaae547fc48526621c32f68080c12cbec7de4fc101461` (105 lines, gate 2972
jobs, the spec contract `token_step_fold` proved), table memory
`5754d4d8548f640c58ac599176338c5ef5b12e33f28a7dcb30abb821df16e900` (59 lines, gate 3046 jobs); the
five-module barrier `lake build Nous.Core.Shape Nous.Core.Tensor Nous.Model.Weighting
Nous.Model.State Nous.Model.Memory` is exit 0 (3052 jobs), every per-task control 0/1/0, and the
read-link statement-change verifier returned accept with residual (see the core batch above). Wave 3
closed: `Nous/Core/Fold.lean` `2a34004c8db796f3d2bcfb246e46e80ac89934b2f84fec349cf9d64ce92914c2`
(187 lines, gate 2966 jobs, no `Nous.Model` import) and `Nous/Core/Autograd.lean`
`db9052d211b564926fac326af4af69c84ef09e1ec240a2ca4c5ca958188e490e` (155 lines, gate 113 jobs,
`OpSpec`/VJP laws, no TorchLean import). Wave 4: the central bet is proved —
`Nous/Model/Decomposition.lean`
`6ca711482144cac434f535501312982bd6edb8e5d615595080e3b797854cd65e` (277 lines, gate 3051 jobs,
control red and restored): `channelPair` is bijective and multiplicative, with the fold and
`memoryStep` diagonalized, under three hypothesis tiers (explicit inverse witnesses, `IsUnit`,
characteristic ≠ 2 fields). Wave 4 assembly closed: `Nous/Model/Model.lean`
`4324b389e4337b6a8a2bc1cf072bbad10ace8e1f84b7a0bb68f02ba58fa33572` (201 lines, gate 3051 jobs,
control red and restored); `run_state_eq_tokenFold`, `run_memory_eq_memoryFold`, `run_eq` and the
readout laws hold, with `probe_width_bound` explicitly on the executing face. All 24 Nous headlines
are registered and the cone is closed at `AxiomAudit.lean`
`6273f809bea21cab136e414c63e0341450948cd165ac6c0249f99acb57e6c305` (build 3889 jobs, every
declaration inside `{propext, Classical.choice, Quot.sound}`). The Nous architecture track is paused
here by user direction. The parity pivot's `fold` engine inventory is complete
(`/tmp/opencode/fold-engine-inventory.md`: a full Vulkan/SPIR-V engine, measured 620 vs 520 GB/s at
27B and 524 vs 476 at 8B against llama.cpp on the RTX 5080). The parity decision is resolved to the
**integer/fixed-point executing path**: no source-level component sits outside the model, the
bounded runtime seam is rejected, and the Aeneas float fork is deferred with a trigger (a float
artifact that must itself be verified, not a test oracle). `APEIRON-TOOLCHAIN-SPIKE-INTEGER-TRANSCENDENTALS`
returned (`/tmp/opencode/toolchain-spike-integer-transcendentals.md`): accuracy is not the binding
constraint — a degree-6 fixed-point Taylor `exp` reaches ~1e-8 relative and the recorded served path
passes `gpugate` with f32 per-op error — the proof is; pinned Mathlib v4.31.0 ships `Real.exp_bound`,
`Complex.exp_bound`, `Real.sin_bound`/`Real.cos_bound`; `exp`/`rsqrt` are weeks each of Lean by
anchor, RoPE argument reduction is the long pole (Q32 first, 32768+ later), and the honest artifact is
a proved per-function ε plus a runtime-checked decision margin. The parity epic's
`docs/epics/inference-parity.md` and `docs/epics/inference-parity.md` follow the inventory; the first
slice landed — `APEIRON-INT-TRANSCENDENTAL-W1-T0` (`src/numeric/fixed_exp.rs` `d225d1e8…`, tested,
max relative error 8.4e-9 for `x ≥ 0` / 3.4e-7 for `x ≥ -8`, control 0/101/0) — and the extraction
spike returned: Charon/Aeneas extract the body (`i128` included), but as-is it emits three unmodeled
axioms; the tested range-check form is axiom-free and pointwise equivalent, so
`APEIRON-INT-TRANSCENDENTAL-W1-T0B` applied it — `src/numeric/fixed_exp.rs` `3538878f…` (216 lines),
cargo trio green, live re-extraction audit `NO_AXIOMS`, equivalence `mismatches=0`. The `rsqrt` body
landed too (`APEIRON-INT-TRANSCENDENTAL-W1-T2A`: `4181249b…`, 204 lines, cargo 106 passed, extraction
`NO_AXIOMS`, max relative error 1.16e-7, control 0/101/0). The core lane is closed, so `W1-T1`
(extraction + `fixed_exp_refines`) starts: it regenerates `proof/ApeironWord.lean`, moving the
identity off `8e76b917…`; the lead pre-added the `Numeric` lib root (`d66a71cc…`) and the module
skeleton (`ba1ad45a…`). The move landed, then the batch wiring moved it again: `proof/ApeironWord.lean` was `b85742bf…`
(11251 lines; `regen.sh` exit 0 and `regen.sh --check` exit 0 after an unreproduced transient
package-build exit 120); the exp restructure moved it to `328655cd…` (11273 lines). `Numeric/ExpRefinement.lean` now proves the refusal
instance, the post-guard rewrite (`word_fixed_exp_window_past_guards`), the `round_div_2p32` spec and
the reduction budget; `Numeric/RsqrtRefinement.lean` proves its refusal instance and bound
decomposition; the shared scale definitions moved to `Numeric/Scale.lean` (`1a215651…`) to decouple
evidence binding; `AxiomAudit.lean` `00ab1794…` (3894 jobs, closed cone). The `fixed_trig` body
landed (`0813a402…`, cargo 110 tests, extraction `NO_AXIOMS`, 4.47e-7 max relative error, control
0/101/0) and `Numeric/RopeRefinement.lean` is its root. Exp: `acc6` is **closed** —
`word_fixed_exp_horner_acc` proved with one lead-authorized `set_option maxHeartbeats 8000000 in` on
that theorem (2M/4M insufficient; 167 s compile), audited in `AxiomAudit.lean` `5c59dfd2…`. The scaling shift-count lemma and the
`k ≥ 0` branch value landed (`fixed_exp_scale_shift_count_lt`, `fixed_exp_scale_nonneg`,
`80069af6…` → `f7f02a50…`, both now verified in the green `AxiomAudit.lean` `27653a4e…`); `W1-T1Q`–`W1-T1U` closed the `k < 0` core, both branch theorems, the two scaling helpers and
**`word_fixed_exp_scale`** (the full scaling block with the exact value/refusal post;
`f7f0efd5…` → `4c3d1db5…`; the reserved 8M cap was not used). The whole-body composition then pinned
the structural facts: **Aeneas WP cannot consume a multi-operation chunk on the flat extracted body**
(guards resolve semantically only inside `step*`, reassociation has no bind node, `step*` stops inside
the Horner chunk), the pair chunk's inner `Result.Ok (k, r)` cannot head the flat success path, and the
guarded proof does not transplant across guard removal. `W1-T1AD` landed the fresh **success-path
chunk** (`word_fixed_exp_reduction_pair_succ`, no guard; `f83cc8e8…`). `W1-T1AF` identified the precise
mismatch: the trig body is **natively chunked** (call/`match` boundaries) so it composes by
`spec_bind`, while the exp extraction is a **single flat do-block** with inline guards and no `match`.
**Owner decision: option (a)** — `W1-T1AG` landed the refactor (`src/numeric/fixed_exp.rs`
`e3dc208a…`; Rust gates green) and the lead cleared a Charon sysroot defect (the miri cache; see the
environment note) so the regen exited 0: `proof/ApeironWord.lean` is now `328655cd…` (11273 lines;
`--check` exit 0), with `fixed_exp.fixed_exp` emitted as the trig-shaped
`let r ← exp_reduce x; match r with …`.

Extraction identity moved again with the format wave: the `W3` Rust landed (`src/format/{gguf,quant}.rs`,
`Result` refusals, 19 tests, cargo trio green) and the two translating integer-code roots
(`word_q4_k_codes`, `word_q6_k_codes`) put `proof/ApeironWord.lean` at `a3ad8c27…` (11585 lines;
`regen.sh` and `--check` exit 0). The parser root is now **live**: `W3-T20` eliminated the recursion
(`src/format/gguf.rs` `3ee474ed…`: loop-only `Cur::value` over a flat arena; the generated Lean
compiles — the check the earlier probes missed), and `proof/ApeironWord.lean` is `71ccf8e1…`
(13066 lines; regen/`--check` exit 0, package 1708 jobs); corpus and audit re-verification pending
the Q6_K lane's settle. The two float-dequant roots (outside the integer fragment; fixed-point
rewrite is the consistent route) remain migration items. `W3-T3` landed the refusal half
(`Numeric/FormatRefinement.lean` `709b7b62…`) with the Q4_K positive equality landed
(`word_q4_k_codes_ok`) and Q6_K in flight (`W3-T18`). The canonical packing grammar
landed (`src/format/packing.rs`, D3); `APEIRON-ALGEBRA-A5` added the F16/F32 identity faces (f16 exact
per element, f32 round-to-nearest Q32.32 with refusals; `from_gguf_kind(0)=F32`, `(1)=F16`;
`PACKED_GGUF_KINDS` now 8; focused `format::packing` 15 passed) and extracted `word_packing_decode_block`
clean (Charon/Aeneas exit 0, no residual `axiom`/`sorry`, no float externals, generated Lean compiles).
`word_packing_convert_row` is removed (early returns inside loops in `import_row`); `packing.rs` is
1751 lines and splits next. The decoder root is now in `regen.sh` (lead `W1-T1` done): `proof/ApeironWord.lean` is `b07482b9…`
(14803 lines), regen and `--check` both exit 0 at 1708 jobs; earlier `71ccf8e1…` receipts are superseded
by this identity.

Engine wave: `W4-T1` landed the skeleton (`src/engine/mod.rs` `b7c203a3…`, `src/engine/loader.rs`
`feb8590b…`, `examples/engine_probe.rs` `f37b6c06…`) — the probe loads `qwen05b-Q4_K_M.gguf` (290
tensors, Qwen2.5-0.5B config resolved), computes exact Q32.32 embeddings for fixed token ids with
determinism and refusals, and the decode-loop port map is recorded in `src/engine/mod.rs` (rows 0–8).
Engine rows 0–8 kernels landed (`W4-T2`–`W4-T7`: dequant, RMSNorm, RoPE, projections, attention,
FFN, LM head tied to `token_embd`, argmax/top-p; `src/engine/mod.rs` `6ee03602…`), on top of the
Q5_0/accessor work (`W3-T14`), the raw-bits metadata (`W3-T16`) and the live parser extraction
(`71ccf8e1…`); the BPE tokenizer landed (`W3-T26`: `src/format/bpe.rs` `fe712a65…`, ten vectors
matched to fold's tokenizer, gates green); the decode-loop integration (`W4-T8`) landed
(`Engine::step`/`step_with_kv` over rows 0–8, fold ordering verified against `cpu_step_f32`).
`W4-T9` (real 0.5B decode vs llama.cpp) is **BLOCKED on kernel range**, not on wiring: the loader gains
resolve once at `Engine::open` and the tokenizer ids match llama.cpp exactly (`[785, 6722, 315, 9625, 374]`
for `The capital of France is`), but prefill token 0 refuses with `ffn: ExpOutOfRange`. Instrumented
(Q32.32): layer magnitudes 3.62/3.33/**20.19**/**31.37**/…/41.29 with a negative tail to **−22.11**,
exceeding the `fixed_exp` window `[-16, 21]`, the derived `silu_fixed` domain `[-21, 16]` and the attention
softmax window (score below −16). Dequant is ruled out (llama.cpp `q5_0`/`q6_K`/`q4_K` match `quant.rs`),
and the llama.cpp reference stream is `12095 13 576 6722 315 279 3639 4180 374 6515 11 422 727 13 576 6722`
(`" Paris. The capital of the United States is Washington, D.C. The capital"`); no tok/s exists because the
run refuses in <1 s. `loader.rs` `ab182220…` (855), `src/engine/mod.rs` `84ae603d…` (3477),
`examples/engine_decode.rs` `2645b09e…` (110); all cargo gates green; control (preloaded-gain ×2) red.

The engine split landed (`W4-T9` follow-up): `src/engine/mod.rs` 3477 → 229 plus twelve modules
(`dequant`, `norm`, `rope`, `matvec`, `attn`, `ffn`, `lm_head`, `sampler`, `step`, `value`,
`f32_convert`, `tests`; max `tests.rs` 1196 ≤1,500), 26 public items unchanged at `crate::engine::*`,
snapshot checksum `0x483c6d3ece968247` identical before/after, cargo trio exit 0; the `mod.rs` digest is
now `777cddf5…`. `W4-T9B` adds saturating `silu`/softmax branches outside the exact in-window domains
(`tested`; the bounds are the G3 refinement obligation) to unblock the real decode and the llama.cpp
comparison. That landed: the real Qwen2.5-0.5B Q4_K_M greedy decode now matches the llama.cpp reference
**16/16 ids** (`[12095, 13, 576, 6722, 315, 279, 3639, 4180, 374, 6515, 11, 422, 727, 13, 576, 6722]`,
`" Paris. The capital of the United States is Washington, D.C. The capital"`), prefill 3.369 s/5 tok,
decode 9.226 s/16 tok → **1.73 tok/s** on CPU (functional parity; the performance gap to llama.cpp is
open). `ffn.rs` `28e82ddb…` (133), `attn.rs` `b3366c40…` (381), `tests.rs` `b9f9d523…` (1199); control
(the saturation forced back to the refusal) red. The row-6/softmax descriptions in `mod.rs` were
corrected to the saturating contract (`c41aac34…`, 232). `APEIRON-ALGEBRA-G3-SILU` rooted `silu_fixed` for
extraction (the refinement carries the saturation bounds): the real body moved to the leaf
`src/engine/silu.rs` (92; `ffn.rs` delegates, 115), the extract unit mirrors the module tree, and the
root is live — `proof/ApeironWord.lean` is `5a8f4082…` (14907 lines), regen and `--check` exit 0 (1708
jobs), `SOURCE_FILES` now carries `engine/{mod,value,silu,ffn}.rs` (the `mod.rs` digest closes the
mirrored-`ONE` drift). `Numeric.EngineSiluRefinement` (`976b5399…`, 237; gate 2450 jobs, control red)
proved `word_silu_fixed_high`/`word_silu_fixed_low` (the two saturation equalities),
`div_round_nearest_abs_bound`, and reduced the in-window bound to `silu_window_refines` via the proved
`word_silu_fixed_unfold`; the extracted-body reduction did **not** hit the `UScalar`/pair-defeq wall
(isolated to the parser cursor chain). The in-window bound is still open (`silu_window_refines` remains a `def … : Prop`; a continuation
session returned no change) and is re-dispatched to a fresh session.

Model algebra and rigor (lead): `docs/epics/model-algebra.md` + `docs/epics/model-algebra.md` (D1)
define the layer fold, the three-class reflection contract (extracted kernels source→abstraction;
generated/data parameters abstraction→source; tested driver) and the composed-margin target;
`Nous/Model/ErrorComposition.lean` (D2) proves the fold budget and the margin-preserved decision at
the closed cone. The size limit (§4, 1,500 lines) is now enforced: `Numeric/ExpRefinement.lean` (16)
→ `Numeric/Exp/Part{1..4}` (106 declarations identical) and `Numeric/RopeRefinement.lean` (15) →
`Numeric/Rope/Part{1..4}` (112 identical), gates green; a stray `Numeric/ProbeBits.lean` was pruned;
`src/engine/mod.rs` (3477) and `src/format/packing.rs` (1711) are the two remaining violations, both
split queued. `APEIRON-ALGEBRA-D5` landed the TorchLean baseline at the pin `de3192df…`
(`docs/evidence/rigor-baseline.md`, 327 lines; we exceed on executing-artifact refinement, closed cone
and mutation controls, they lead breadth at 3,007 theorems; G5 corrected, G6 closed, G1–G4/G7 open with
gates). `G4` is written: `docs/epics/model-algebra.md` (342 lines) maps every intent/spec headline and the
gap list to waves W0–W5 with one gate and one control each; the intent (167) and spec (212) carry the
same drift corrections (G5 restated as the log-substitution identity, composed-margin headlines renamed
to the landed `decision_margin_preserved`/`numeric_fold_margin_usable`). `W3-T1` and `W3-T2` both
**closed**: `Nous/Model/Attention.lean` (`ecdd61fe…`, 301) proves `attention_channel_form` (the running
`Σ wᵢ, Σ wᵢvᵢ` scan equals `Σ (wᵢ/Σ wⱼ)vᵢ` in the factorized-score form under the inhabited
`ChannelFormDomain`; gate 3052 jobs; dropped-normalizer control red), and `Nous/Model/Weighting.lean`
(`e2ebfac4…`, 194) proves the G5 log-substitution identity (`ratio_eq_exp_log`, `channelWeight_binary_softmax`).
The `Attention`↔ℝ-scan correspondence is a stated reading, not a proved refinement. Both headlines are
**registered** in `proof/corpus/AxiomAudit.lean` (`4c3b3b16…`, 1077 `#audit` lines; audit exit 0 at 3910
jobs, all inside `{propext, Classical.choice, Quot.sound}`) — G1 and G5 cone obligations close. `G2`
closes too: `Nous/Model/Instances.lean` (`c958bed3…`, 346) fixes the hyperbolic/parabolic/elliptic faces
over `ℤ` and the `ZMod (2^64)` word face via `denoteSigma` (`zmod_two_not_unit` keeps the 2-adic caveat),
ternary excluded as `anchor`; 29 headlines registered, audit now `bfb055b7…` (1106 `#audit`, 3911 jobs).
Only **G3** (engine executing-body refinements) remains a proof obligation. G3 status: `EngineNormRefinement`
(`f7a5a4ec…`, 245) proves the empty/sum/argument/rsqrt refusals and the structural unfold, with
is **complete** (`d8508674…`, 882): `rms_norm_refines_proved` is proved with the exact target constant, no
weakening, and registered (`0b20ff53…`, 1125 `#audit`, 3913 jobs) — the second engine kernel refined
end-to-end; `EngineSiluRefinement` is **complete** (`89fb4a6e…`, 767 lines): `silu_window_refines`
is a proved theorem with the constant unchanged (`epsExp·|v| + (4/q32)·|v| + 1/(2·q32)`), the first engine
kernel refined end-to-end, and its headlines are registered (audit `78134f58…`, 1118 `#audit`, 3912 jobs);
the closing technique was compositional `WP.spec_bind`/`spec_mono` + `dsimp only`/`cases` instead of manual
`rw` on monadic binds; `EngineRopeRefinement` (`9ee6458e…`, 949) has the per-pair bound (`rope_pair_spec_proved`
via `fixed_sin_cos_refines_proved`), the refusal equalities, `rotate_window_arith`, the `rotate_loop_zero` base
case, `rotate_head_loop_spec` (**totality**: the inner loop never `fail`s, for every even `head_dim ≥ 2` on the
domain), the cast/value helpers and both `ropePairInt` write identities, the standalone `rotate_head_write_step`
(the `RotateHeadInv` maintenance), and the standalone `rotate_head_body_spec` (the explicit `WP.spec_bind` body
walk, cheap: 13 s, no heartbeat diagnostics). **Parked residual**: carrying the content invariant through the
loop spec's post (`rotate_head_body_spec_content`) and the outer `rotate_loop`/`RopeHeadRefines`; the body walk
is proved and the content is proved standalone, so this is completeness, not a correctness hole; attempts
preserved at `/home/socol/oc-tmp/rope-ctl/{body-lemma-LANDED,content-body-attempt-slice12}.lean`.
Two false-spec defects were
fixed by lead decision (`head_dim = 0`, and `RopeHeadDomain` on short `freqs` = the approved
`head_dim/2 ≤ freqs.length` conjunct). `matvec_fixed`/`row_dot_fixed`: `EngineError` now lives in the leaf
`src/engine/error.rs` (`G3-MATVEC-LEAF`), so the extraction batch for those roots is in flight; the remaining
G3 work is the `attn`/`lm_head`/`sampler` roots and the parked rope content switch.

Measurement requirement (owner): the performance proposition must be compared against a **standard PyTorch**
implementation, not only llama.cpp. Environment verified: `/home/socol/opt/arena-venv/bin/python` (torch
2.10.0+cu128, CUDA live on the RTX 5080 sm_120, transformers 5.5.0), local HF model
`/home/socol/models/qwen2.5-0.5b`. `APEIRON-PERF-PYTORCH` writes `tools/bench/torch_qwen_decode.py` and
`docs/evidence/pytorch-parity.md` — prefill latency, decode tok/s and kernel elements/s for CUDA bf16/fp32 and
CPU fp32 vs our engine, with the numeric-contract caveat stated. `APEIRON-ALGEBRA-SILU-CLOSURE` writes
`Nous/Model/SiluChannel.lean` tying `siluGate` to `channelWeight`/`softmax2` (the gate is the carrier's binary
channel weight at `(e^x, 1)`, whose cell norm form is `e^x`).

`APEIRON-INT-FORMAT-W3-T28` resolved the cursor wall: the `u8`/`u16` equality chain landed via the
`show`-chain route (`Numeric/FormatRefinement.lean` `f0d01b73…`, 1339 lines; gate 1707 jobs; control red);
`W3-T29` extends it to `u32`/`u64`/`four` and the refusal theorems.

**Cone finding (open).** `word_gguf_parse_header_truncated_magic`/`_bad_magic` are proved but depend on
`apeiron_word.gguf.parse_header._native.decide.ax_k` — Aeneas-generated decidability axioms from the
extracted parser body — so they are **outside** `{propext, Classical.choice, Quot.sound}` and cannot be
registered; the audit stays green with them unregistered (`78134f58…`, 1118 `#audit`, 3912 jobs). This is
the parser-refinement cone obligation: either the proofs avoid the native-decide path, or the GGUF parse
stays the tested import boundary and the canonical packing decoder carries the proved claim (the owner
direction). `W3-T30` landed four more refusals (`Truncated "version"`, `UnsupportedVersion`,
`Truncated "tensor count"`, `Truncated "kv count"`; `FormatRefinement.lean` `093ae843…`, 1946 lines, gate
green) with the same cone dependence; the residual is the counts guard (needs a `bind_assoc_eq` restructuring,
since the guard chain has no subterm boundary) and `LengthOverflow` (loop-level). **The parser cone is now
CLOSED**: `W3-T31` landed the cone-clean `gguf_counts_guard_tail` plus `Truncated "kv/tensor count"`, and
`APEIRON-FORMAT-SPIKE-TOSTR` found the one-line model fix — `Aeneas/Std/String.lean`'s `toStr` default changed
from `by decide +native` to `by rw [U32.max_eq]; decide` (the blocker was `U32.max` being an `irreducible_def`,
not `String.toByteArray`). The hunk is appended to `proof/patches/aeneas-lean-model.patch` (`43cc0d27…`), the
model is re-materialised (`proof/model-patches.sh`), the `proof/` package rebuilt (1709 jobs), and after a
forced `Numeric.FormatRefinement` re-elaboration all seven refusals report exactly
`{propext, Classical.choice, Quot.sound}` (no `_native.decide`). All seven are registered in `AxiomAudit.lean`
(now 1,165 `#audit` lines, `70ce9eb0…`); the audit build is deferred until the `ErrorComposition` lane settles.
GGUF stays a tested import boundary, but its refinement is now registrable rather than blocked.

`W1-T1AH` re-anchored `Numeric/ExpRefinement.lean` to the new extraction (`084ba5cc…`; the retired
flat-body rewrite is noted in-file) and landed the `exp_reduce`/`exp_scale` specs — the attempted
`exp_horner` spec doubled the defeq cost past the gate and was withdrawn, so the 8M cap stays on the
retained inline `word_fixed_exp_horner_acc` only; `W1-T1AI` landed the fresh `exp_horner` spec (`word_fixed_exp_horner_acc` on
`fixed_exp.exp_horner`'s body; the single 8M cap retained there; module 153 s, full gate green;
`53182c60…`); `W1-T1AJ` landed `word_fixed_exp_body` — the whole-body value with the refusal disjuncts, by the trig
call-chain technique (`448abb53…`), with the tight `|r| ≤ 2^32` and the `|horner_acc| ≤ 2^40` spec
conjunct; **exp's discrete value semantics is complete**. `E1a` landed the exp reduction displacement (`deltaRed = 31/10^9`, from the
`log 2` constant defect `≤ 10^-9`) and the exponential transfer (`≤ exp·2·deltaRed`; `754295f1…`,
audited `9eb66152…`); `E1b`–`E1B4` completed the Horner error: `horner_acc_error ≤ epsHorner = 2·10^-7` via the list
reformulation (`realHorner` + `List` induction; the deep-nesting wall is gone at default heartbeats;
`7be36e01…`); `E1C` landed the sharpened `|r| ≤ 2^31`/`|toReal r| ≤ 1/2` bounds (`6bc9a9ed…`) and exposed two spec
gaps; `E1C2` discharged Gap A (`8f8031b3…`, 2758 lines, audited `3a3192f1…`): body-level exact values
for both `exp_scale` branches — `fixed_exp_bmod128_low`, `fixed_exp_scale_nonneg_exact`
(`v.val = acc·2^k`), `fixed_exp_two_pow_bmod`, `fixed_exp_scale_neg_core_exact`,
`fixed_exp_scale_neg_exact` (the rounded shift + `≥64` flush) — each excluding `Err` under explicit
range hypotheses, so Gap B reduces to the on-window range budget; `E1C3` supplied it and `E1C4`
**closed `fixed_exp_refines`** (`ExpRefinement.lean` `33c1efc3…`, 3160 lines):
`fixed_exp_refines_proved` — `|toReal v − exp (toReal x)| ≤ 2^-20·exp (toReal x) + 4/q32`, k≥0 exact,
k<0 round-to-nearest under the `4/q32` floor, total ≈ 4.6e-7; audited `99c83c9e…`. Both numeric
headlines are closed; the barrier machine checks are green (`regen.sh --check` with
`AENEAS_HOME=target/aeneas`, both `lake build`s, cargo trio, full audit); `E3` restructuring removed
rsqrt's two 1.6M (`916ca0ef…`) and exp's three `maxRecDepth 4000` (`6c436055…`), leaving the 8M on
`word_fixed_exp_horner_acc` as the one named accepted opening (owner acceptance pending; the
guard-free red run fails at the theorem statement's `whnf`, and the removal pattern is factoring the
~15 body stages into small-term WP lemmas); the independent-verifier pass (three fresh glm sessions) returned **verified** on all three artifacts
with independent mutation controls (trig: `trig_delta` mutation red; rsqrt: proof-body-only diff vs
`b50c6c59…`, domain mutation red; exp: two mutations red, cache-independent re-elaboration green), and
the final `regen.sh --check` at the freeze is exit 0. Only the owner's acceptance of the single kept
8M heartbeat raise remains open; commits are unauthorized. Rsqrt: **closed** —
`fixed_rsqrt_refines` and `fixed_rsqrt_refuses` are proved (`b50c6c59…`, audited `a11ddcd3…`), the
first complete numeric refinement; the two bounded local `maxHeartbeats 1600000` raises on the
`isqrt` specs remain as recorded residuals, and the closure is a candidate for the
independent-verifier pass at the barrier. `W2-T1` (trig refinement): the refusal surface, the
statement (`2^-20` relative with a `2^-20` absolute floor), the reduction facts and value, and the
fold functions/specs landed (`7e3dcd6d…` → `54ae5666…` → `ac33bcee…`, audited through
`6502f7dd…`; the pre-authorized `maxRecDepth` was tested then removed — the file carries no option);
`W2-T1D` landed the reduction→fold composition (bind/match route) and the Taylor head (`25789c2a…`,
audited `d8f57039…`); `W2-T1E` falsified the Taylor domain as first stated (`2^36` overflows the
final cast) and re-stated it at the true post-fold bound `2^32`, landing the stage-bound tools
(`4f6f84b9…`, audited `327a4a92…`); `W2-T1F`/`W2-T1G`/`W2-T1H` pinned the Taylor wall: the definitional decomposition bridge exceeds even
the authorized ceiling (`maxHeartbeats 1000000` still times out at `isDefEq`), so the cap route is
closed as failed with no option landed; the structural calc route proves the whole nine-stage Taylor
body **total** on `|t.val| ≤ 2^32` (`word_taylor_spec`, `ef533655…`, helpers included, no option);
`W2-T1I` closed the value post and `fixed_trig_taylor_value` (`3ef4b3d6…`, no option; the fold found
and fixed an extra `round_q32` in `cos_poly`), so the whole Taylor evaluation is a proved function of
`t`; `W2-T1J` landed `fold_arg_abs_le` and exposed the loose remainder bound as the Taylor-path
prerequisite; `W2-T1K` proved the tight bound — corrected to `2^34` (the `2^32` premise was false:
`x = HALF_PI_Q32` gives `r = 6746518852 > 2^32`; true magnitude `≈ 2^33.65`) — and relaxed
`fold_arg_abs_le` to `|r| ≤ 2^34 ⇒ |arg| ≤ 2^32` (`74022c67…`, audited through `c4e120e1…`);
**`word_fixed_sin_cos_value` is closed** — the end-to-end value spec (reduction → fold → Taylor →
quadrant swap/negate) with the exact integer output formulas (`c05c15f7…`, 1682 lines; the
strengthened `word_taylor_spec` post landed with it); control non-slack red, no option; the full
`AxiomAudit` is green (`efedc36e…`). **Trig's discrete value semantics is complete**; `E2a` landed the argument displacement
(`trig_delta = 326/q32`, `trig_reduction_displacement`, the periodicity + Lipschitz transfers to
`sin`/`cos`; correction: the reduction is `x − k.val·2π` with `k` full turns, not `k·π`) —
`f8227bf9…`, audited `5514d868…`; `E2b` landed the fold-constant defect lemmas
(`pi_q32_defect`/`half_pi_q32_defect`/`quarter_pi_q32_defect`, half-unit each) and
`trig_fold_delta = 1/q32` (`be5c6a88…`). `E2B3` landed the threshold floors
(`half_pi_q32_floor`/`quarter_pi_q32_floor`), the exact folds `fold1T`/`fold2T` and the displacements
`fold1_displacement`/`fold2_displacement` (`bc9238f4…`), with a correction — the composed
`|toReal arg − θ| ≤ trig_delta + trig_fold_delta` is false (the fold maps into `[-π/4, π/4]`).
`E2B4` landed the phase identities (`sin/cos_fold1T`, `sin/cos_fold2T`) and the phase-select lemmas
identifying the extracted `negate`/`swap` flags with the fold jumps (`ab948c35…`, audited
`672238f2…`; the flag compensates across fold thresholds). `E2B5` landed the flagged-output bounds
(`flagSin`/`flagCos`, fold bounds and reduction bounds at `327/q32 ≈ 7.61e-8` on `trig_window x`;
`a0f04645…`; control red, no option — the combined gate fails only in the exp lane mid-re-anchor).
`E2C` landed the quarter-angle bound (`fold_arg_abs_le_quarter`, `b03bc862…`) and derived the full
residual with exact constants: single-rounding ledger (cast obstruction), chain ledgers (sin/cos `7δ`,
Horner-vs-exact `≤ 1/q32`), degree-7/8 Taylor remainders (Mathlib has only degree-3 bounds); totals sin
`≈ 3.9e-7`, cos `≈ 1.0e-7`, both under the `2^-20` floor uniformly. `E2C2` landed the single-rounding ledger
(`round_q32_half`, `round_q32_ledger`, `δ = 1/(2q32)`; `0fd55970…`); `E2C3` landed the Horner chain ledgers (`sinHorner`/`cosHorner`; `≤ 15/(2q32)` and `≤ 40/(2q32)`,
coarser than `7δ` but ≪ budget; `604366f6…`, audited `141e3dfe…`); `E2C4` landed the coefficient defects and the degree-7/8 remainders via `Complex.exp_bound` at
`exp(s·I)` (constants `(10/9)`/`(11/10)` times `1/n!`, inside the floor; `e07bbd74…`); `E2C5` composed every lemma in a probe (total
`≤ trig_floor = 2^-20`; the verifier-corrected worst case is 4.92e-7, 4.29e-7 at `|s| = π/4`); the residual is one definitional bridge (`word_fixed_sin_cos x` vs the value
spec's transcribed do-block); `E2C6` pinned the blocker: the extracted
inlined body (P) and the value spec's wrapper transcription (Q) are equal only via the reduction's
success value, not definitionally; `E2C7` stopped under the three-attempt rule on the
program equality `P = Q` (only the reduction wrapper survives normalisation; its collapse is semantic);
`E2C8` pinned the last encoding gap (after
`bind_assoc`+`fold_step`, only Q's projection-lets over the fold 5-tuple and the pair results survive
against P's named tuple-lets); `E2C9` sharpened the gap to a monadic bind-push
through `let`-headed do-blocks (`bind_tc_ok` won't fire; the pair lemma misses on concrete
continuations); `E2C10` proved the kit's push/pair lemmas but
the fold collapse dies on a pattern-let bind inside a continuation (`dsimp` blocked); `E2C11` scoped the direct-P replay exactly (the
reduction guard block plus the inlined fold/Taylor, ~150 lines of fragile `step*`/case work); `E2C12` closed the whole reduction in the
direct-P probe; `E2C13` narrowed the residual to one tuple-let redex in the final post assembly and the
lane escalated to **xiaomi-pro** (`E2C14`, §9.1); `E2C14` **closed `fixed_sin_cos_refines`** —
`fixed_sin_cos_refines_proved` (`RopeRefinement.lean` `3bd26da3…`, 3524 lines): per output
`|toReal sc.f − f (toReal x)| ≤ epsExp·|f (toReal x)| + trig_floor`, `epsExp = trig_floor = 2^-20`,
composed worst case 4.92e-7 (4.29e-7 at `|s| = π/4`); the post normalises to the nested-if form (15-slot `refine` + `hsw`/`hfn`),
and the tuple-let redex fell to `step*` once the outer `if swap = 1#i32` was resolved; gate exit 0
(2500 jobs), audit green (`a235b779…`), control non-slack red, restore byte-identical; no `set_option`. Guard raises inventory was six; the `E3` wave discharged five (exp's three
`maxRecDepth 4000`, rsqrt's two 1.6M) and the remaining `maxHeartbeats 8000000` on
`word_fixed_exp_horner_acc` is a named accepted opening, **owner acceptance pending** (the guard-free
red run fails at the theorem statement's `whnf`; the removal pattern is factoring the ~15 body stages
into small-term WP lemmas).
GPU/SPIR-V subset verification (owner decision): for Apeiron/Nous and the integer numerics, formal
verification of a **subset** of SPIR-V is in scope; the driver lowering and the GPU hardware remain the
named platform boundary with a runtime-checked margin (as rustc/LLVM/ISA are for the CPU lane), with
mechanized source and target semantics for the used fragment following the CompCert pattern (the
assembler/linker/hardware equivalents excluded). Song's separate full end-to-end language/OS/prover
requirement is unchanged. `APEIRON-GPU-SPIKE-SPIRV-SUBSET` returned (201 s of its 40-min box; report
`/tmp/opencode/toolchain-spike-spirv-subset.md`): the served decode kernel `gemv_dp4a_module_dyn`
(`modelfold/src/simt/dp4a.rs`, sha256 `ffe4f6c3…`, spot-checked in the lead) uses 47 SPIR-V opcodes
with no barriers, atomics, subgroups, workgroup memory, function calls or `OpPhi`, one `GLCompute`
entry point, `Logical`+`GLSL450`, integer core `OpSUDot` (packed 4×8-bit) — a bounded fragment the
existing `ZMod (2^w)` word-denotation pattern can carry; no mechanized SPIR-V value/memory semantics
exists to reuse, so the epic builds it. The binding risk is the float scale epilogue (no Aeneas
model): the v0 spec takes the integer fixed-point epilogue (owner integer-numerics direction), with
the fold f32/f16 epilogue as a migration item; phase 2 (barriers/atomics/subgroups/coopmat, poison
`AccSat`) is gated behind v0. `APEIRON-GPU-SPV-W0-T1`/`W0-T2` landed `docs/{intent,spec,plan}/gpu-spv.md` (`7e056f4b…` /
`5393c34b…` / `c2c904af…`; path check 31 resolved / 0 miss, 9 planned creation targets); the intent
carries the face-parameterized target, the 2-adic caveat and the layout-vs-representation question,
with `sdot_denote` graded "proved or refuted" and Nous targets cited as planned modules where they do
not yet exist. The parity barrier is closed and independently verified, so the GPU waves are open:
the lead registered `lean_lib Spirv` in `proof/corpus/lakefile.lean` (inert full-corpus build green)
and **`W1-T1` is closed** (`Spirv/Fragment.lean` `bf2cc021…`, `Spirv/Semantics.lean` `47f12803…`;
gate exit 0, 2973 jobs; both controls red and restored): the served opcode set re-derives to 44
(executing 39) mechanically from the emitter, and the layout verdict is **proved at the hyperbolic
face `σ = +1`** (`sdot_split_ringMul`, `sdot_split_realized`, root witness checked), with
cross-channel accumulation a named emitter migration item and the 2-adic caveat retained; `W2-T1`
(decoder) **closed** (`Spirv/Decode.lean` `905f5710…`, 558 lines: D1–D4 + `decode_sound`, gate exit 0
2967 jobs, controls red+restored, audited `65e949c7…`); the Rust checker **landed** (`src/spirv/check.rs`
`2fac3308…`, 676 generated emitter words, 42 served opcodes exercised, controls red+restored) and
surfaced two reconciliation items — the served set is 42 not 44, and `Inst.uses` is over-strict on
SPIR-V forward references — being aligned by `W3-T19`/`W2-T3`. The spike report was lost in the reboot;
the re-derivation came from the emitters and the public SPIR-V 1.3 + `SPV_KHR_integer_dot_product`
specs.
Continuity requirement (user direction): the v0 refinement target must be a face-parameterized
instance of the canonical `Cell`/`ringMul`/norm-form object, not a standalone `ZMod (2^w)`
semantics — the GPU face must meet the CPU face at the same canonical object, with the Nous channel
decomposition (`RingCell ≅ R × R`), fold associativity (split-K) and channel-form attention as the
named consumers; whether the emitted Q4 even/odd packing realizes a face representation is the first
provable-or-refutable layout question, to be settled in the epic's spec.
`APEIRON-TOOLCHAIN-SPIKE-SONG-PROVER`
returned: current Song has no Lean tree and its prover is integer-linear (cannot carry IEEE-754); the
frozen gen1 `song-ref` holds the tested SoftFloat core and the proved one-ulp `Float3` bound, recorded
as seeds only. Core track: **`SpillLocateFoundFullMatch` is proved.** `Growable/LocateLoop.lean`
`a745a9172ed773b7a8d71727d4a7c8ac920a27a5f96f31278fa9b0b70c4efb20` (1666 lines) proves L1–L5, the
inner and outer found extractions and the premise-free run-relative unfold;
`Growable/Classifier.lean` `9a04fbce34232df3a26149e22070e06734731572cf7c52d725fa739df974de53`
(659 lines) adds `spill_locate_found_full_match` with the `SpillLocateFoundFullMatch` statement
unchanged; root in `lakefile.lean`, twenty-five declarations registered in `AxiomAudit.lean`
`8cf72c78…` (3890 jobs, cone closed); one slice overran its 45-min budget (~48–50 min estimated),
five landed in 22, 27, 22, 22 and 32. Residuals: the `LocateOuterStepReady` geometry premises stay
explicit, and the per-lane miss-content claim is asserted nowhere. **Priority change (user direction):**
existing-model inference under the proven stack, compared with llama.cpp, precedes the Nous
architecture; W4-T2/W4-T3 are paused. TorchLean at `de3192df…` stays reference-only; nothing
publishable until rebased and proved end-to-end.
Environment note: `grep` and `find` in PATH resolve to shadowed `~/.local/bin` binaries; use `rg` or
`/usr/bin/grep` and `/usr/bin/find` in gates and receipts.
Vertex lane state (current): `~/.config/opencode/opencode.jsonc` sets `google-vertex` `location:
global`, mapping to `aiplatform.googleapis.com`; the active `google-vertex` credential is an API key
(Express mode, live probe: 429 "Resource has been exhausted (e.g. check quota)"), the `google` AI
Studio key is over its monthly spend cap (429, billing cap), and both `gcloud auth login` and ADC are
expired — gemini-flash dispatches return 429 until `gcloud auth login` + `gcloud auth application-default
login` + `opencode auth logout` (Vertex) restore project auth. `APEIRON-INT-TRANSCENDENTAL-W1-T0` was
rerouted to deepseek-flash on this basis.
tmpfs note: `/tmp` is a 32G tmpfs; keep scratch and control copies under `/home/socol/oc-tmp`, only
named receipts and logs under `/tmp/opencode`. A second exhaustion recurred (100%, an `ENOSPC`
live-file write failure); the lead pruned ~32G of stale finished-slice trees (`w8-t1`–`w8-t5-control`
and the `w*rc/w*-target` build trees) — `/tmp` now 329M/32G. Prune each finished slice at its receipt;
the 93%-full incident is registered in `AGENTS.md` §10.
Reboot note: the machine restarted mid-slice; the in-flight lanes were reconciled — `ExpRefinement`
`b38375df…` builds green with the E1B pieces registered (`AxiomAudit.lean` `348b1c63…`),
`RopeRefinement` untouched (`e07bbd74…`), trig probes under `/home/socol/oc-tmp/`, `/tmp` wiped and
recreated, and a pristine E1a copy at
`/home/socol/oc-tmp/int-exp-e1a-control/ExpRefinementControl.lean` (`754295f1…`).
Aeneas/Charon toolchain defect: `proof/regen.sh` resolved `std` from the **miri cache**
(`~/.cache/miri`; `E0514`, while the pinned `nightly-2026-08-18` sysroot was correct);
`RUSTUP_TOOLCHAIN` did not change it. Fix: park the cache
(`mv ~/.cache/miri ~/.cache/miri.bak-lead`) before regen — regen and `--check` both exit 0 after.

## 7 · Barrier and closure

A wave barrier requires **every** child gate green with its control run, plus the shared gates in
§4 on one source/tool identity. Record the sha256 of the bound sources and the Aeneas/Charon/Lean/
Mathlib revisions; evidence from a dirty tree is attributed to those digests, not HEAD alone. The
epic exits are owned by `docs/flash-handoff.md`. Report format for every slice: **changed files,
acceptance result (command + exit status), proved vs tested vs open, next bounded task.** No
`proved` grade for anything without a Lean kernel check and, if about Rust, a refinement theorem
from the real extracted source under the closed cone.

---

## 8 · First actions

1. Inventory the dirty source and task worktrees; compare source/tool digests with retirement evidence.
2. Use the next plan's bounded tasks. Reuse unchanged receipts with their exact scope; rerun a gate
   when its inputs changed, its evidence is missing, or the named integration barrier requires it.
3. Prepare independent branches concurrently; lease the verification lane to one owner at a time.
4. Stop at the planned local proof/contract barrier. Universal read and whole-epic acceptance remain
   open until their own source-linked theorems, witnesses, controls and shared gates pass.
