# Multi-repo fleet brief: apeiron · penglai · song · webfold

Coordinator: one. Pool: one (`lane`). Workers: many. This brief is the shared skeleton plus
per-repo deltas; the deltas are the only part that differs.

## 1 · Why one prompt does not yet cover all four

The four repos *deliberately* forbid each other's vocabulary today, and each ban is written into a
governing doc:

- apeiron `AGENTS.md`: "Do not import \[song's\] commands, its `R1`–`R5` registers, its `//@`
  ontology, or its `law`/`spec` block semantics."
- penglai `AGENTS.md:181`: the `REGISTER` field "must not import Song R1-R5"; `./song` gates invalid
  (`:206`).
- song `ONTOLOGY.md:4`: must not import sibling ontologies; must not union corpus grades with Song
  grades (`§5.4`); the apeiron zero-axiom Slang rule is apeiron-only.
- webfold `AGENTS.md:13`: "independent, sovereign standalone"; no borrowed register; "purge mystical
  vocabulary" (git logs).

So convergence is a **specification change**, not a parallel workstream: until one shared contract
says which keys/registers/grades/task-IDs are common and which stay repo-local, and each `AGENTS.md`
is amended to point at it and strike the corresponding ban, 50 agents aimed at "make them agree"
edit against four contradictory authorities.

## 2 · Shared skeleton (the part that should not disagree)

- Work unit: epic → wave → task; a task owns ≤3 files, carries one gate and one Arc block
  (`SCOPE`/`REGISTER`/`CORPUS`/`REFERENCE`/`GATE`/`GRADE`).
- Evidence grades, same semantics everywhere: `proved` / `tested` / `measured` / `model`·`anchor`.
- Gate discipline: real-source mutation makes the **named** headline fail; restore is green;
  declared budget; no `sorry`/`axiom`/`admit` outside the stated cone; bounded wall clock.
- Lanes: every compile/gate runs as `lane <command…>` (box-wide pool, auto-acquires and waits).
- Handoff: paths + sha256 + gate command + exit + red/restore evidence + budget + one question.
- Coordinator re-runs each gate on the frozen submission before commit; workers never commit.

## 3 · Per-repo deltas

| repo | authority | gate commands | task IDs | forbidden import |
|---|---|---|---|---|
| apeiron | `AGENTS.md`, `docs/direction.md` | `cargo test/clippy/fmt --locked`; `proof/regen.sh --check`; `cd proof && lake build`; `cd proof/corpus && lake build` | `APEIRON-<EPIC>-W<n>-T<k>` | song `//@`/`R1–R5`/`law`·`spec` |
| penglai | `AGENTS.md`, `docs/substrate-port-scope.md` | `scripts/check.sh --all`; `cargo fmt/clippy/test --locked`; `proof/regen.sh --check` + `lake build`; `tools/benchmark.py`; world-profile/solver-cost/water checks | `PENGLAI-<EPIC>-W<n>-T<k>` | Song R1-R5 |
| song | `AGENTS.md`, `docs/DESIGN.md`/`ONTOLOGY.md`/`ROADMAP.md` | `./song check\|test\|controls\|build\|laws\|crosscheck\|bench`; `lake build Apeiron.Parser.Arithmetic` (sanguine-apeiron) | `<PROJECT>-<EPIC>-W<n>-T<n>` | sibling ontologies; apeiron zero-axiom rule |
| webfold | `AGENTS.md` (sovereign), `docs/EXTRACTION-PIPELINE.md`, `ADVERSARIAL-REVIEW.md`, `CONTENT-RECOVERY-PLAN.md` | `tools/webfold/check.sh` + `controls.sh`; `proof/regen.sh --check`, `regen_acquisition.sh --check`, `audit.py`, `controls.py`; receipt checkers | (webfold IDs) | any other repo's register/vocabulary |

## 4 · Dispatchable census (width for ~50)

- **apeiron**: cost-model roots still uncommitted; `CostBridge` (gate missing); extraction-root
  repair (attn drift); the source-connected production `Admission` certificate; layout/size task.
- **penglai**: `SUBSTRATE1-W1-T1`; `WORLD-W1-T1/T2`; `WATER-W2-T2/W3-T1/W4-T1` + 3D spike;
  `ISLAND-MATERIALS-W1-T1` (red 101); `TREE-W1` S1–S6; `PERF`; `WORLDGEN-W2-T1/T2`.
- **song**: `A0.10` access discharge; `A0.6.P3–P6` parser semantics; `A0.4` stream replay;
  `K2.1.1–K2.1.4` Slang gate driver; `B6.1–B6.4` verifier acceptance; `A0.9`/`A0.6` reader (blocked).
- **webfold**: `B3`, `C0–C3`, `D1–D5`, `E1–E5`, `F1–F4`, `G1`; 32 unresolved case IDs;
  `PROOF-01..07`; open APP/RT/HARNESS/CLI ledger rows.

## 5 · Prerequisites (green the baseline before the wave)

- **penglai**: `scripts/structure.sh` red (tree hygiene); milestones 0,1,5–9 MISSING/PARTIAL.
- **webfold**: both zero-target gates red; merged source needs fresh gates/report; the 637 dirty
  files are receipts/docs, not `src/` — confirm before freezing.
- **apeiron**: extraction-root repair — `proof/regen.sh --check` cannot pass, so no attn-family task
  or admission registration is valid until it does.
- **song**: `./song check` exits 2 on the access census; `A0.10` is the discharge.

## 6 · Sol prompt template (one per repo, same shape)

> You are reviewing **<repo>** at `<branch>@<HEAD>`. Read, in order: `<AGENTS.md>` then
> `<authority doc>`. The gates are: `<gate commands>`. The open obligations are: `<top 5–8 with doc
> paths>`. The baseline blockers are: `<list>`. Return exactly one decision among the named options,
> or a blocker — do not review the repo generally, do not request further context, do not modify any
> file. Keep it under 400 words.
> Decision requested: `<the single highest-leverage choice>`, with the option you reject and why.

Budget note: four one-shot calls fit the remaining OpenRouter balance; only lose it if Sol runs
multi-round, so each prompt ends in one question with a named fallback.

## 7 · Tooling debt: retire the per-task Python harness

Each cost-model task minted its own convenience script. The measured population, before the wave:

| repo | `.py` files | loc | concentration |
|---|---|---|---|
| sanguine | 138 | 55,301 | |
| apeiron | 93 | 32,710 | `proof/resource` 36, `tools/slang` 12, `tools/gpu` 8, `tools/joint` 7 |
| TorchLean | 48 | 14,564 | |
| intent | 242 | 13,239 | average 55 lines/file; 107 in `runs/`, 58 in `domains/` |
| song | 6 | 1,094 | |

`apeiron/proof/resource` holds **33 `*_gate.py` totalling 6,274 lines**. All 33 carry the same
harness — `flock`, `systemd-run -p MemoryMax=12G`, a private lake package, sha256 `digest()`, a
receipt JSON, `main()`. Only **18 lines** are byte-identical across all 33; identifier-blinding any
two leaves **116 changed lines**. That is 33 divergent bespoke implementations of one role, each
requiring its own audit, and it is why several gates serialize on one hardcoded lock file
(`/tmp/opencode/n.lock`, `n-1..3.lock`, `apeiron-lane-3.lock`) while the box-wide `lane` pool is not
on the gate path at all.

The replacement shape already exists in-tree: `tools/dispatch/admit.py` is a thin entry into the
Rust core `tools/dispatch/{main.rs,core.rs}`. Apply it to every convenience script:

- **apeiron**: one `tools/gate` Rust driver plus one declarative `proof/resource/<module>.gate.toml`
  per module (source paths, `#[path]` wiring, claimed headlines, mutation needle, budget, controls).
  33 scripts and 6,274 lines reduce to 1 driver and 33 small specs. `zero_axioms.py` and `admit.py`
  keep their paths as thin entries because contracts bind them; their logic moves to Rust.
- **intent**: 242 files at 55 lines average collapse to the same driver plus per-run specs.
- **sanguine**, **TorchLean**: the same, after the two above.

Two rules constrain the retirement:

- Gates **are** the acceptance protocol. AGENTS §10 requires a changed acceptance protocol to have
  its own approved task, and §9.1 forbids a candidate rewriting its own gate. The driver must
  reproduce every existing gate's green **and** its named mutation red before any bespoke gate is
  removed; the old gate stays until the new driver's receipt matches it.
- Removal is a migration or an authorized retirement: name the replacement, inventory affected
  consumers and gate reachability (failure register, consolidation row). A deleted `*_gate.py`
  ships with its reachability receipt, not a bare `rm`.

First bounded task, if authorized: **freeze the 33 gates' current green and named red as a
conformance suite** the driver must reproduce. It is read-only, reversible, touches no gate, and
can run in parallel with the row-dot/admission chain.

Until the driver exists, hand every new cost-model task a **spec to fill in**, not a script to
write.

## 8 · Review gates before full-pool dispatch

Do not open the ten-lane swarm until each review below returns a named verdict on a named artifact.
Each review is one question with a stated decision; a review that returns "looks fine" is not a
verdict.

| # | review | artifact reviewed | verdict required |
|---|---|---|---|
| R1 | Contract | the shared skeleton (§2) plus each repo's `AGENTS.md` gate and grade sections | one shared contract named, or the blocking disagreement named |
| R2 | Gate sufficiency | the dispatchable census (§4), row by row | every dispatchable row has one named gate and one control, or the row is withdrawn |
| R3 | Baseline | the prerequisite list (§5) | each prerequisite green, or the exact red command and output |
| R4 | Tooling | §7 | driver underway, or the retirement plan accepted; no new bespoke gate admitted |
| R5 | Lane and review bandwidth | the lane pool and verifier availability | the worker count the review bandwidth can actually re-run and merge |
| R6 | Trust boundary | each task's obligation-11 statement | the uncovered layers named per task (rustc/LLVM, kernel ABI, GPU/SPIR-V, `unsafe`, host adapter) |

Dispatch rule between R4 and the driver's arrival: cap the swarm at the number of tasks whose
gates the lead can re-run inside the wall-clock budget (§10) and whose files are disjoint; a full
ten-lane dispatch needs R1–R6 green first. A review that finds a missing gate, theorem, digest or
measurement records the residual obligation rather than waiving it.

## 9 · Programme phasing (draft for review)

The apeiron lane's detail is in `docs/plan.md` §Phasing. This is the whole programme, sized to the
fleet.

**Assumptions.** ~50 agents, 10 lanes, at most 3 simultaneous heavy Lean checks, one owner per file,
every wave a barrier. A wave's gates run inside the 540 s cap; with integration a wave of about ten
tasks costs under an hour, so a four-day window holds roughly 30–40 wave-slots — up to about 350
bounded tasks. The open programme below is ~280–320 bounded tasks: **one full pass of the ecosystem
in the window, not a fraction of it**, which is the argument for the plan being cross-repo.

**Every epic cites the document it comes from.** No task is invented.

### Song — `docs/ROADMAP.md` §2 (~45 tasks)

| wave | content | gate |
|---|---|---|
| S1 | A0 verification integrity children A0.1–A0.5 (K1, K2, K6) — the implemented/partial remainder | each child's reproducible completion command |
| S2 | A0.6 parser semantics P2–P6 refinements + the emitted obligation stream | `./song check` + the P-series controls |
| S3 | A0.7–A0.9, K11, D1, P1–P5, O2 | per child |
| S4 | A0.10 census lane (1,244 items: 548 raw, 696 region) | the census gate |
| S5 | K2.1.1 Slang driver utility library — **gate MISSING**, first task is the red fixture | red fixture walks the closure from Slang alone |
| S6 | K2.1.2→K2.1.4 build/chain driver in Slang, then the Slang gate driver | `song build` verdicts equal the shell chain's |
| S7 | A1 memory-authority inventory → A1.x creation | inventory receipt, then the new children's gates |
| S8 | A3 translation-trust inventory (K8/K13) → A3.x; R5 | inventory receipt |
| S9 | A5 interface probes; A6 application profiles (wl_term, a pinned Whispering Tides slice) | probe / profile gates |
| S10 | B6.1–B6.4 verifier acceptance | the verifier gate |

### Penglai — `docs/substrate-port-scope.md` §7 (~30 tasks)

| wave | content | gate |
|---|---|---|
| P1 | milestone 5 closure: independent orbit/globe/margin fixtures, the f64 error/enclosure bridge, corpus Kepler composition, final-margin `RefinesWithin` | milestone-5 exit table |
| P2 | milestone 6: data-permission/estimator gate first, then the affine/catchment folds | data-license, calibration and latency gates |
| P3 | worldgen lanes `WORLDGEN-W2-T1/T2`, `WORLD-W1-T1/T2` | per-lane |
| P4 | water lanes `WATER-W2-T2/W3-T1/W4-T1` + the 3-D spike | per-lane |
| P5 | `ISLAND-MATERIALS-W1-T1` (red 101), `TREE-W1` S1–S6 | per-lane |
| P6 | milestone 7 (temporal/material/three-factor) — the major unknown, spike-first | derivation and measurement gates |

### Webfold — `docs/REPAIR-LEDGER.md` Repair order (~125 tasks)

| wave | content | gate |
|---|---|---|
| F1 | network and service boundaries: APP-SSRF-01/02, APP-PROXY-01, APP-FLARE-01/03, APP-BOUND-01, APP-MCP-01, RT-02/03/04/05/06/21, CLI-02/05/07 (14 items) | per-hop address checks + negative connection/resource controls |
| F2 | runtime correctness and bounds: RT-07…RT-26, CORE-06/08/10–14/17–23, CLI-03/04/06/09–12 (~30 items) | real callers, bounded counterexamples, mutants |
| F3 | page counterexamples 158/208/279/511 as a post-hoc set | explicit page-purpose policy |
| F4 | evidence and portability: independent annotations, structural scorer, BUILD-02/03, EVAL-05 | holdout evidence, not aggregate pass |
| F5 | formal obligations PROOF-01..07 | source-connected statements + deletion/fault controls |

### Apeiron — `docs/plan.md` §Phasing (~45 tasks)

Epics E1 gate machinery, E2 receipt consolidation, E3 source-cost returns, E4 extraction freshness,
E5 the Apeiron–Intent engine. The `E4.W1→W3→W4` and `E1.W2→W4`, `E1.W5→W6`, `E5.W2→W7` chains are the
critical path; `E2` and `E3` are cargo- and document-bound and fill the lanes the Lean waves leave.

### Intent — `docs/DIRECTION.md`, `docs/EVOLVER.md` (~20 tasks)

The three corrections (D7 rebase epoch, D2 cost bound, loop-brief target), the Ring 6 gate with its
four controls, evaluator/isolation repairs, the runtime linkage and canonical rooting, the ≥30-pair
calibration, five candidate screens, one accepted cycle, then a second cycle on the first cycle's
evidence.

### Packing across repos

Each repo's waves are independent except where a repo's own barrier orders them; the fleet packs one
wave per repo per slot and fills remaining lanes from the cargo/document waves (Apeiron E2/E3, Webfold
F4, Penglai P3–P5, Song S9) that need no heavy Lean. That is how 10 lanes stay busy without exceeding
3 simultaneous heavy checks.

### Questions for review

1. Song's A1–A6 are "not dispatch-ready" pending inventories — dispatch the inventory tasks now, or
   keep Song at A0/K2.1/B6 only for this window?
2. Penglai 7 is the major unknown; is it spike-first in this window or deferred behind 5/6?
3. Webfold's 117 findings include 14 historical and 18 unmeasured — does the window take the 85 with
   current evidence only?
4. Is the per-repo barrier order the right cross-repo rule, or does a shared wave counter run across
   all repos?

### 9.1 Owner decisions on the phasing

1. **Song — dispatch the inventories now.** A0/K2 work continues alongside bounded inventories for
   A1–A7 and B6. An inventory identifies actual source paths, dependencies, missing gates and the next
   independent tasks; implementation waves open as those prerequisites clear.
2. **Penglai 7 — one bounded spike alongside 5/6**, with a precise question and a stop condition. Its
   return establishes a usable contract, a counterexample or the smallest missing derivation.
   Milestone closure stays separate from exploratory results.
3. **Webfold — current evidence first, the rest retained.** Dispatch repairs with reproducible
   counterexamples first; give historical or unmeasured high-impact findings bounded *reproduction*
   tasks (age alone does not discharge them); deduplicate overlapping findings by mechanism and
   affected population before counting tasks.
4. **Per-epic dependency barriers, one global resource scheduler.** The scheduler owns memory, cache
   ownership, file ownership and verification capacity. Independent epics in one repository run
   concurrently; cross-repository dependencies create explicit connecting barriers. A shared wave
   counter adds little.

### 9.2 Corrections before dispatch

- **The task totals are estimates.** A 540 s gate ceiling does not establish an hour per wave:
  implementation, proof search, retries and integration also consume time. Forecast completion from
  the drained wave's observed turnaround and acceptance rate. "One full pass" means every obligation
  gets a **disposition** — accepted, rejected, blocked or explicitly deferred.
- **E3 is substantial Lean work.** Source-cost proofs and `RowDotAdmission` are not cargo/document
  fillers. Classify each task's actual resource demand; keep three heavy Lean checks as the initial
  cap and adjust from measured peaks.
- **Do not delete `RegenRelocateAll` for sharing a source body.** Compare carrier assumptions,
  domains, refusals and claimed resource properties; reuse a common extraction where appropriate and
  retain distinct obligations where the contracts differ.
- **Ecosystem coverage is implicit.** Add explicit disposition rows for `sanguine`/`sanguine-apeiron`,
  `TorchLean` and Whispering Tides; a Song application profile covers one interface slice, not the
  application programme.
- **Replace the per-child gate placeholders before dispatch.** Each task needs its exact command,
  owned paths, rejecting control, budget and prerequisite identities. These tables are the programme
  map; that registry is the executable schedule.

### 9.3 First batch

Prioritise the Rust gate/receipt machinery, extraction repair, Intent's selection controls and Song's
verification integrity; fill independent capacity with Penglai and Webfold work. Integrate
continuously and replenish from ready tasks. Progress is accepted, reusable results — and completed
improvement cycles — across the ecosystem. The main architectural choices are settled: most remaining
work proceeds through bounded tasks without re-escalating to the owner.
