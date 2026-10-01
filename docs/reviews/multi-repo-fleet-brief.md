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
