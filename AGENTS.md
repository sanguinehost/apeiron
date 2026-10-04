# apeiron: working agreement

`R[x]/(x^2 - sigma)`, its `Cell<i64>` word kernel, and the growable flat/spill table that consumes it. One
Rust crate, one Lean extraction package, one Lean refinement bridge. This file governs the work; the
source and the gates are the record.

Apeiron owns the canonical definitions, the reusable Rust primitives and their refinements. The
programme thesis, ownership, outcomes, primitive correspondence, migration and execution order are
in the [programme direction](docs/direction.md); that document is the programme authority. This file
is the local repository authority for the lifecycle, decomposition, gates, evidence grades and trust
boundary below. The physical canonical corpus remains at `../sanguine-apeiron/proof/Apeiron/` with
unchanged pins until a separately gated migration; no source has moved. Song is an independent
assembly-up implementation: no Rust or Aeneas runtime dependency links it to Apeiron, and the
required cross-repo result is a checked primitive correspondence, not runtime linkage.

## Purpose

Apeiron is the O(1)-primitive basis at fixed probe width with bounded callbacks and nonconstant key length,
evolved from hashtrinity; `fold/hashtrinity` is the naive predecessor. The basis is
`R[x]/(x^2 - sigma)`, `Cell<R>` with `Sigma` as a value in every executing signature, `Coord`
addressing correct per face, and the growable flat/spill `Table<K, V, H, N>` with transactional
checked-sizing and reservation refusal. Every executing face must refine the canonical object;
anything not rooted in it is not accepted. A judgment about the object is admissible only at its
stated evidence grade and scope, through an explicit link to the claimed property of its named
denotation: equal denotations preserve that property, and implementation, resource or performance
observations require their own stated model and evidence rather than following from functional
denotation alone. The canonical object is
`Apeiron.Table.{HMap,get,insertWith,build}` and
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` with the extracted `Cell<i64>` word
bodies (`word_mul_denote`, `word_norm_denote`).

The fold crosses from hashtrinity into apeiron, and that crossover is the lineage; the precise
statement is in [programme direction](docs/direction.md#lineage-the-fold-crossover).

Reply with the result and the evidence. State a decision or a blocker; do not narrate tool calls.

The sibling `song` tree is a different repository with its own ontology, registers and `./song`
gates. None of those apply here. Do not import its commands, its `R1`–`R5` registers, its `//@`
ontology, or its `law`/`spec` block semantics. Adopt only the lifecycle, the wave/task discipline and
the writing rules below.

## 1 · Reading order

Read the file, not a summary of it, not an agent's report on it, not memory. A citation names a path
that resolves; run the path before citing it.

| Stage | Document | Purpose | Exit condition |
|---|---|---|---|
| 1 | [`docs/direction.md`](docs/direction.md) | Programme authority: decisions, evidence, open work, correctness items | Current stage and its unresolved obligations are known |
| 2 | `docs/kernel-boundary.md` | Layer ownership and the trust boundary | Which layer is proved, tested, measured or out of scope is known |
| 3 | `docs/plan.md` | Execution order, gates, specialist approval, stop points | The current stage's gate and stop condition are known |
| 4 | `proof/README.md` | Extraction and bridge, exact commands, what is **not** established | Tool pins, extraction selection and axiom cone are known |
| 5 | `README.md` | Crate behavior and its stated non-claims | The actual `Cell`/`Table` behavior and the open obligations are known |
| 6 | exact source | `src/`, `tests/`, `proof/extract/lib.rs`, `proof/corpus/` | The symbols to change have been read, not recalled |

`docs/direction.md` and `docs/kernel-boundary.md` govern. `docs/plan.md` governs execution
order. Where a document disagrees with the source or a gate, the source and the gate win; fix the
document.

**This order is enforced, not assumed.** [`tools/rooting/read_gate.py`](tools/rooting/read_gate.py)
`--check` reddens by name when a stage document is missing or is not referenced in this file;
`--receipt` pins each stage document's sha256 into a dispatch, and `--verify` reddens on drift
between the receipt and the tree. **Anything every agent must know to act is stated in this file**,
because this is the one document every agent is delivered; `docs/` is the referenced detail record,
not required background. A gate establishes document identity and drift, never comprehension; that
boundary is stated, not papered over.

## 1a · Enforcement map

A rule below is non-optional only when a named artefact enforces it and a command shows the verdict.
Prose that names no enforcer is not acceptance (§4, obligation 11).

| rule | enforcing artefact | command |
|---|---|---|
| §1 a citation names a path that resolves; §5 a grade maps to a named artefact | [`tools/check-claims.py`](tools/check-claims.py) | `python3 tools/check-claims.py` (`--self-test`: 11/11 controls) |
| §1 the reading order — every stage document resolves and is referenced in this file; a dispatch pins their sha256 | [`tools/rooting/read_gate.py`](tools/rooting/read_gate.py) | `python3 tools/rooting/read_gate.py --check` (`--self-test`: 3 rejection fixtures; `--controls`) |
| §2 epic intent/spec/plan; §3 task ≤3 files with one gate and one control; §8 the per-function cost comment; a retired route not presented as active | [`tools/rooting/check.py`](tools/rooting/check.py) | `python3 tools/rooting/check.py --check` (`--self-test`: 30 reject / 23 accept) |
| §4 obligations 1, 3, 5 — real-source extraction and the closed axiom cone | [`proof/regen.sh`](proof/regen.sh), [`proof/corpus/AxiomAudit.lean`](proof/corpus/AxiomAudit.lean) | `proof/regen.sh --check`; `cd proof/corpus && lake build` |
| §4 obligation 5 — the Slang empty cone | [`tools/slang/zero_axioms.py`](tools/slang/zero_axioms.py) | `python3 tools/slang/zero_axioms.py --stage <stage> --controls` |
| §4 obligation 7 — the per-task mutation control | the task's own gate | that gate's command; a mutation that leaves the proof green is a failure, not a pass |
| §4 obligation 9 — Rust gates | `cargo` | the §6 commands |
| §5 — a stated artifact resolves against the evidence index: exactly one fresh row per receipt | [`tools/rooting/index_check.py`](tools/rooting/index_check.py) | `python3 tools/rooting/index_check.py --check` (plus `--self-test`, `--controls`) |
| §6 — bounded Lean under the lane | `lane` (box-wide pool, outside this repository) | `lane <cmd>` |
| §7 — `//@` metadata | **MISSING** — no local gate reads an annotation | n/a |
| §9.1 — inference-optimisation admission | [`tools/dispatch/admit.py`](tools/dispatch/admit.py), [`proof/DispatchAdmission.lean`](proof/DispatchAdmission.lean) | `python3 tools/dispatch/admit.py --launch <proposal.json>` |
| §10 — a removal names its replacement, consumers and reachability; no bare `rm` | [`tools/rooting/retire_check.py`](tools/rooting/retire_check.py) | `python3 tools/rooting/retire_check.py --check` (plus `--self-test`, `--controls`) |
| §10 — a registered documentation removal names a resolvable successor, its removed path is absent, and no live document names it | [`tools/rooting/doc_reach.py`](tools/rooting/doc_reach.py) over `docs/moved.tsv` | `python3 tools/rooting/doc_reach.py --check` (`--self-test`: 18 rejection / 4 accept fixtures; `--controls`; `--inventory PATH`) |
| §9.3 — a dispatch names its board row; a named row exists and its project carries a milestone | [`.githooks/commit-msg`](.githooks/commit-msg), [`tools/hooks/lane.py`](tools/hooks/lane.py) | `bash tools/hooks/install.sh`; `python3 tools/hooks/lane.py self-test` (9/9 controls) |
| §3 the dispatch loop — Sol ranks, one lane per item (serialize on shared files), the lead installs on the live tree, measure then board | the operator pulse (`/home/socol/oc-tmp/bin/pool-ping.sh`); [`.githooks/commit-msg`](.githooks/commit-msg) + [`tools/hooks/lane.py`](tools/hooks/lane.py) | the pulse fires the checklist and fails a checkpoint with no measured movement; the hook refuses a commit naming no row |
| §9.3 — the board is reconciled against the plan at the barrier: every row mirrored, internal-by-design or missing | [`tools/board/board.py`](tools/board/board.py) | `python3 tools/board/board.py check --controls` (the pre-push hook reports it; `APEIRON_HOOKS_BOARD_CHECK=block` refuses) |
| §6 — `cargo fmt` at commit, `clippy` and the focused `cargo test` at push | [`.githooks/pre-commit`](.githooks/pre-commit), [`.githooks/pre-push`](.githooks/pre-push) | `bash tools/hooks/install.sh`; `bash /home/socol/oc-tmp/hooks/gate.sh` (10/10 controls) |

Enforcer that is MISSING and named so that prose cannot stand in for it: a Rust gate driver
(`tools/gate`) replacing the per-task `*_gate.py` scripts. It is tracked in
[`docs/plan.md`](docs/plan.md). The documentation-removal reachability gate for the documentation
consolidations is not missing: it is [`tools/rooting/doc_reach.py`](tools/rooting/doc_reach.py), the
row above, with its [discharge receipt](docs/evidence/documentation-removal-reachability.md); the
evidence-receipt half runs as [`tools/rooting/index_check.py`](tools/rooting/index_check.py).

## 2 · Lifecycle

Every nontrivial epic proceeds through five phases. The first three are sections of one committed
file per epic, `docs/epics/<slug>.md`, started from `docs/templates/epic.md`; `docs/README.md` indexes
every epic. A document parsed by a gate lives in `docs/contracts/` and is never merged or reformatted.

1. **Intent** — the problem, why alternatives were rejected, the invariants, the measurable success
   condition, the non-goals.
2. **Spec** — the contracts, the admissible domain and partiality, the Rust-to-Lean representation
   relation, the failure semantics, the evidence plan.
3. **Plan** — the waves, files, owners, one gate command per task, and the negative control where
   soundness is at stake.
4. **Implementation** — the smallest change that turns the named gate green, on the path the crate and proof build and run.
5. **Verification** — run the gate and its control, record the result, then register the claim at its
   evidence grade.

An epic without a written intent, spec and plan is still design. Do not allocate tasks from it.

## 3 · Work decomposition and collision discipline

`epic -> wave -> task`.

- A **wave** is a barrier: every gate green, with its negative control run, before the next opens.
- A **task** touches at most three files, carries one arc (scope, gate, control, grade), and is
  closed by one command's exit status, never by an agent saying so.
- A task that cannot run parallel to its siblings takes its own wave.
- **One in-progress owner per file.** Tasks in one wave touch disjoint files. Declare the owner in
  the plan. If a task needs a file another task owns, stop and sequence it; do not edit across the
  boundary.
- **Shared build products are lanes.** `proof/ApeironWord.lean`, `target/`, `proof/.lake/` and
  `proof/corpus/.lake/` each take one agent at a time. Do not mix incompatible Lean/Mathlib
  artifacts in a shared cache.
- **A branch and its worktree exist only for their task.** Work merges to `main`, the integration
  target, or is deleted; the worktree is removed with `git worktree remove`. Inventory with
  `git worktree list` and `git branch -vv` before starting and after closing, and prune stale
  records (`git worktree prune`). No branch or worktree outlives its task; one that does is a failed
  task, not a note.
- Do not run working-tree-mutating git commands (`stash`, `reset`, `checkout`, `restore`, `clean`).
  Preserve unrelated dirty work; a mutation control edits a throwaway copy, never the working tree.
- A task that needs conversation history to execute is not yet a task.

### The dispatch loop

A red goal is worked by one loop, never by picking the easiest bounded task:

1. **Sol ranks.** The lead asks Sol for the **highest-impact items** that move the goal, ordered by
   contribution *and difficulty*, hardest first. The ranking is written under
   `/home/socol/oc-tmp/escalation/` and is the dispatch order.
2. **One lane per item.** Each item gets **one** lane, and no more lanes than the item's
   independent-file set: items that share a file (or a shared build product) **serialize** in Sol's
   order, never run in parallel. Never open the easy, bounded, safe, verification, corpus,
   instrument or documentation task while the goal's number is red; use spare capacity for
   independent verification, not invented parallel work.
3. **Each lane closes on its gate and control**, on the path the crate builds, in an isolated copy.
   A lane returns the edited artefact, the exact gate output, the measurement and the decision — never
   a live write and never a commit. **A lane runs to completion inside one turn: it must not
   background its campaign, `sleep`, spawn a detached `runs.sh`/loop, or return "waiting" on a
   background waiter.** A lane that comes back without its numbers is not a lane; it is a stall the
   owner cannot distinguish from progress. The lead dispatches agents, not ad-hoc background shell
   campaigns, and follows them to their return.
4. **The lead installs.** The lead applies the lane to the live tree, **re-runs the gate on the live
   tree**, and commits. A lane's green gate is not the live tree's green gate.
5. **Measure, then board.** Run the averaged measurement, then update the board: the project
   check-in with the fresh numbers, and the task rows whose status changed. A checkpoint that reports
   no measured movement while the goal is red has failed and says so. **The board update is a field of
   the checkpoint lane's own return, not a step the lead performs from memory: the lane posts the
   check-in and moves the rows before it returns, and a return that carries numbers without the
   check-in and the moved rows has failed this step.** The per-cycle trigger is the operator pulse
   (`/home/socol/oc-tmp/bin/pool-ping.sh`); with it off, the board is stale until the next lane posts
   it — the cadence is the pulse, the content is the lane. The `commit-msg` hook enforces a board row
   per commit only; it does not force a per-cycle check-in.

Enforced at exactly two points: the operator pulse (`/home/socol/oc-tmp/bin/pool-ping.sh`) fires the
checklist and fails a checkpoint with no measured movement; and the per-commit board declaration
(`.githooks/commit-msg`, `tools/hooks/lane.py`) refuses a commit naming no row. The ordering rule —
hardest first — has **no** mechanical enforcer: it is owner direction, and a dispatch that inverts it
is a failure the lead owns.

Use `APEIRON-<EPIC>-W<wave>-T<task>` IDs and an isolated `worktree-agent-<id>` per task.
Record owner, base revision and build lane before editing; serialize shared caches. A documentation
pass explicitly assigned to the supplied checkout may record that exception. A spike uses
`APEIRON-<EPIC>-SPIKE-<slug>`, a time limit and a question; it ends in evidence or a blocker, not
epic closure. Existing short IDs remain historical aliases.

Emit this Arc Block before the first nontrivial edit; retain it in the plan and, when authorized,
the commit body. Templates do not waive these fields:

```text
SCOPE:      owned files, change and completion condition
REGISTER:   local obligation (algebra/refinement/state/test/measurement/planning); exclusions
CORPUS:     exact module and quoted theorem with native grade, or files read and no match
REFERENCE:  absolute path read, derived from <theorem>, or NONE - inventing
GATE:       command, cwd, observed red output, numeric wall-clock budget in seconds (§10);
            if absent, MISSING and a gate-first task
GRADE:      evidence grade and before/after result; unmeasured or not applicable with reason
```

`REGISTER` classifies the obligation; it does not import Song's deciders or R1–R5 certification.

## 3a · Dispatch priority

Order dispatch by contribution to the goal and by difficulty, never by dispatchability.

- The question is which task moves the programme outcome most and is hardest to do — not which
  task is bounded, well-specified, or likely to return green. A body of easy, safe, locally-green
  work (control registration, citation repair, splits, documentation sweeps) is not progress while
  the governing outcome is an open quantitative gate; it is avoidance, and it is not a reason to
  open a lane.
- Work each goal's written dependencies in order. A goal whose dependency is open gets no lane.
  **Penglai and webfold are downstream of the apeiron core and are frozen until apeiron beats
  hashtrinity *and* is formally verified** — i.e. G1 acceptance green (`full_acceptance=true`) and
  the real-source extraction with its refinement proved under the closed axiom cone. No penglai or
  webfold lane opens before that condition holds. Penglai's water frame rate and webfold's parser
  both ride the apeiron core; working them first is the same avoidance as hygiene.
- When a goal's gap is a measured number — a failing ratio, a frame rate, a latency — the lane that
  attacks that number outranks every hygiene task, and its first task is to reproduce the number on
  the current source. A recorded number is stale until re-measured; a stale number is not a status.
- Rank by expected contribution, then by difficulty. The hardest high-contribution task is
  dispatched before the easiest, however many lanes it needs.

- **Subsume the empirical best before deriving.** Empiricism and formalism are two sides of one
  coin: if an empirically better solution exists for a role, **subsume it first and formalize the
  subsumption**; never iterate from an obviously inferior primitive. The empirical side selects the
  base — the measured winner becomes the working artifact, on the measured path from step one; the
  formal side certifies that same artifact — the refinement targets the subsumed body. A
  formalization of a losing implementation is a certified loss; an unproved winner is an unverified
  gain. **Every dispatch names the empirically best known solution for its role by repository,
  revision and digest, and records `subsumed`, or `not-subsumable` with the exact reason and the
  search command and its output; a new primitive with an unstated or unexplained absence is
  refused.** Corollary: a lane whose base is a structure the measured winner does not pay is not an
  optimization but a tax — name the base and its measured ratio first. **Enforcer: the
  dispatch-admission `subsume` field (MISSING until built; until then this rule is review-enforced
  only and is recorded as such, not as acceptance).** Worked example: the empirically best counting
  structure is `../fold/hashtrinity` (pinned `Cargo.toml:76-80`, rev `4cc14471`); it must be
  subsumed and proved, not re-derived, and the generic `Table` (`src/table.rs`) is the inferior
  primitive — not a valid base for a count-face optimization.

- **The incumbent's evidence is pinned, not regenerated.** Read `../fold/hashtrinity` at the G1
  revision `4cc14471` (`src/lib.rs` sha256 `f8c8fd12fc2b7a69…`; fold HEAD `d497161f`'s `lib.rs` is
  `5bc1c9f48ef0bb17…`, a **different** file — do not mix the two) and its monomorphized disassembly,
  already emitted, at `/home/socol/oc-tmp/ht-llvm/`: `full.asm` sha256 `b68a950e375e5516…`, `ap.asm`
  `2841d4faa631e537…`, `disasm.txt` `61bc3365ac8c71d5…`, with the instruction-level work order in
  `ht-llvm/receipt.md` and the attribution in `ht-probe/receipt.md`. A lane **reads those**; it
  re-builds `../fold` or re-emits IR only to cover a body not already disassembled, and then records
  the new body, digest and path. Re-emitting the same body is not evidence.

Owner direction.

## 4 · Definition of done

A task closes on its focused gate and control, completed within its §10 wall-clock budget; a
timeout is not closure. A wave requires every child plus the shared gates.
An epic closes only when every row of its exit table is green on the same source/tool identity;
missing gates, witnesses or evidence keep it open. `docs/plan.md` owns the epic exits;
Historical completion records are not fresh acceptance.
Record inapplicability with its excluded population and reason, never because a gate is difficult.

The eleven obligations below instantiate five pillars; none is removed:

| Pillar | Obligations |
|---|---|
| Primary artifact | 1, 3: build and run path and real-source extraction |
| Non-vacuous contracts | 2: domain, joint witnesses, initialization and preservation |
| Mathematical attribution | 4, 5: exact authority, refinement and closed axiom cone |
| Falsifiable verification | 6, 7: positives and baseline/red/restore controls |
| System pass and trusted perimeter | 8, 9, 11: identities, integration gates and bounded trust |

Item 10 applies whenever a performance claim or epic budget is required; documentation cannot waive it.

1. **Build and run path.** The change is on the path the crate and proof actually build and run; there is no separate production, reference or mock path. No stub, parallel copy or dead code presented as complete.
2. **Explicit admissible domain and partiality.** Total, `Option`/`Result`, checked, or panicking is
   stated. Every claimed `Pre`/`WellFormed` domain has a checked witness term or `#guard` for the
   joint hypotheses at the claimed types, not separate incompatible witnesses. Require initialization
   from the actual constructor, preservation, and successful/refusing branch witnesses for partial
   operations. A model witness does not refine a Rust constructor. No silent domain reduction or
   infallible signature over an operation that can lose bits.
3. **Actual Rust extraction where a Rust claim is made.** The Aeneas/Charon extraction includes the
   real source by `#[path]` and calls the real body; a wrapper may fix a type, not restate the
   formula.
4. **Lean theorem link.** The representation relation names the exact canonical definition or
   theorem it refines. Generic mathematics lives in the pinned canonical corpus, navigated at
   `../sanguine-apeiron/proof/Apeiron/`; its audits are `Audit.lean` and `AuditCarrier.lean`.
   Rust representation theorems live in `proof/corpus/`, audited by `AxiomAudit.lean`.
   Record the resolved corpus revision; a sibling path or `@grade` alone proves nothing.

   **Search the lineage and the sibling repos before commissioning a primitive.** A primitive this
   tree lacks very often already exists — most likely in the older corpus (`../sanguine`,
   `../sanguine-apeiron`), sometimes solved in a different sibling repo entirely. That form is a
   **port with a named target**, not a new derivation, and the spec phase records the search: the
   declaration found, its revision and digest, and what the port must add. Only absence everywhere
   authorizes making one — and then the spec says so explicitly, because "we made it" is a claim
   about prior art as much as about code. The search is a command over the loaded in-tree lineage plus
   `~/Workspace/*` (fleet brief §9 names the repositories), not a recollection.

   **Use a search tool that works here.** `find` is shadowed by a broken shim on this box
   (`/home/socol/.local/bin/find`, findutils 4.4.2): it exits 0 and returns **nothing** on directories
   that plainly hold files, so a `find`-based absence check is unsound and has already produced one
   false "the corpus has 0 Lean files" claim. Use `rg --files`, `ls -R`, or a shell glob, and state the
   command in a claim of absence. `grep` is likewise shadowed/mangling; `rg` is the working tool.
 5. **Fail-closed axiom cone.** `proof/regen.sh` audits the extraction for `axiom`/`sorry`/`admit`;
    `proof/corpus/AxiomAudit.lean` `#audit`s each headline and throws on anything outside
    `{propext, Classical.choice, Quot.sound}`. A `sorry` is a build failure, not a note. The Slang
    source-refinement chain is stricter: every registered declaration, instance, witness and headline
    must audit with an **empty** axiom set under `proof/corpus/Slang/Audit/Empty.lean` and
    `tools/slang/zero_axioms.py`; `propext`, `Classical.choice`, `Quot.sound`, `noncomputable` and
    quotient-based representations such as `Finset` and `ZMod` are rejected there. The Rust/Aeneas
    bridge and the canonical corpus retain the closed cone above.
6. **Positive tests.** `cargo test` covers the nominal path and the boundary cases that the contract
   names.
7. **Mutation control.** A mutation of the real source (or the generated kernel) in a throwaway copy
   must turn the proof or test red; restoring it must pass. A check that cannot fail has no weight.
8. **Source and tool digests.** Record the sha256 of the sources the evidence binds to, and the
   Aeneas/Charon/Lean/Mathlib revisions. Evidence from a dirty tree is attributed to those digests,
   not to HEAD alone.
9. **Rust gates.** `cargo test`, `cargo clippy` and `cargo fmt` as in §6 pass.
10. **Measurement for performance claims.** A number is a measurement only when a named command ran
    and its output was read. Name the machine, the metric and the interval; before and after.
    **Average across runs.** Report the mean over at least three runs with the observed spread; a
    single run is a sample, not a result. A pass-count or ratio that moved within the spread
    already measured on unchanged code is not a change and is not reported as one. The
    repeated-run protocol belongs to the harness, not to the reader's patience.
11. **Trust-boundary statement.** State what the claim does not cover: rustc/LLVM codegen, the
     kernel ABI, GPU/SPIR-V, the `unsafe` container, or any layer the evidence does not reach.

At an implementation epic barrier, run all §6 Rust commands,
`proof/regen.sh --check`, and `lake build` in both `proof/` and `proof/corpus/`, plus the named
controls. These are the local barrier set, not one existing aggregate harness. Missing tools block
acceptance. Local trusted checking code ≤5,000 physical lines, owned source files ≤1,500 lines, ≤12 immediate source files
per directory. Inventory external trust separately; neither the whole proof library nor a bridge
line count is the full TCB. Until the inventory and rejecting checks exist, size acceptance is open.

## 5 · Evidence grades

Every stated result carries one grade. The grade is part of the claim.

| grade | means |
|---|---|
| `proved` | A Lean kernel check (`lake build`) passes for a named theorem. If it is about Rust, the extraction is from the real source and a refinement theorem links the extracted body to the canonical definition, under the closed axiom cone. The Slang source-refinement chain instead requires an empty axiom cone (`tools/slang/zero_axioms.py`), not `{propext, Classical.choice, Quot.sound}`. |
| `tested` | A `cargo test` executes the build path and its boundary cases. |
| `measured` | A benchmark or `perf` command ran, with a named machine, metric and interval. |
| `model` / `anchor` | A reading of the algebra onto a domain, or a design assumption. True as a mapping; not evidence of correctness. |

A citation to a Lean theorem, a paper, or a sibling repository is a pointer, not evidence. It becomes
`proved` only when the named theorem exists and the in-repo link is checked. **Never call a citation
alone verified.** Label an `anchor` as an `anchor`; an `anchor` reported as `proved` costs a redesign
when someone checks.

## 6 · Local commands

Kani is retired by user direction. Do not run, schedule, repair or require it for development,
acceptance or closure. This supersedes every older Kani gate, retry, harness-retention and
no-retirement instruction in handoffs, plans and receipts. Historical results remain historical;
retirement proves no missing theorem. Use ordinary Rust tests and source mutation controls for
regressions, and extracted Lean refinements for proof claims. See
[`docs/epics/kani-retirement.md`](docs/epics/kani-retirement.md).

These are the gates. There is no `./song` gate in this repository; a command that only exists in the
sibling tree is not an acceptance gate here.

Rust:

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
```

Proof regeneration (requires the pinned Aeneas/Charon under `target/aeneas/` or `AENEAS_HOME`, plus
the patched Lean backend that `proof/lakefile.lean:19` requires):

```sh
proof/model-patches.sh     # build target/aeneas-model/lean from the backend plus the model patch
proof/regen.sh             # extract, audit, build
proof/regen.sh --check     # fail on drift from the committed ApeironWord.lean
AENEAS_HOME=/path/to/aeneas proof/regen.sh
```

`proof/model-patches.sh` takes no arguments, has no `--check`, and rebuilds
`target/aeneas-model/lean` in place; `proof/README.md` carries the ordered route from a clean
checkout. Step 2 cannot be skipped: `target/` is gitignored, so neither tree exists in a fresh
clone.

Bridge build (canonical corpus plus the refinement package):

```sh
cd proof/corpus && lake build
```

Slang source-refinement zero-axiom gate (Stream 0; empty cone, per stage):

```sh
python3 tools/slang/zero_axioms.py --stage <stage> --controls
```

Evidence tooling:

```sh
python tools/call-census/audit_reports.py --raw-root /tmp/opencode/census-v2
```

Tool pins and the archive checksum are in `proof/README.md`. Record the versions a run actually used;
do not cite a pin from memory.

Every Lean command runs bounded: under `systemd-run --user --scope -p MemoryMax=<cap>
-p MemorySwapMax=0` with `LEAN_NUM_THREADS=1` and a `timeout`, serialized on
`flock /tmp/opencode/apeiron-lane.lock`. The `timeout` carries the §10 budget; the older 900 s and
3600 s local-run allowances are superseded for new runs, and historical receipts stand. `set_option maxHeartbeats 0` and any unbounded kernel
reduction (for example `decide` or `rfl` over a whole source, or `maxRecDepth` raised to defeat the
guard) are forbidden in committed proofs and in an agent's runs; a reduction that needs them is
inconclusive, not a proof, and a job that exhausts the cap is a failure of the run, not a red gate.

**Every Rust command runs under the same bound.** `cargo build|test|run|clippy|rustc`, the
`examples/reg1` campaigns, and every lane's build or test step run under
`systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0` with a `timeout` carrying the §10
budget; a lane that spawns a build without the cap is not admitted and its result is not evidence. The
bound exists because an uncapped workspace build at 16 codegen units is 0.5–2.7 GB: twenty concurrent
lanes drove this 62 GB box to a **54 GB memory peak plus 50 GB swap inside one terminal cgroup**, with
no OOM-kill and no panic — the kernel stopped scheduling the cgroup, which presents as a dead terminal.
A build is not exempt because it is short.

**Concurrency is capped, not aspirational.** At most **eight** build-bearing lanes run at once on this
box; a ninth waits on `flock /tmp/opencode/apeiron-lane.lock` or on a finished lane's return, and the
sum of the declared caps stays under half of RAM. Twelve to twenty concurrent workspace builds is the
ceiling being exceeded, not a width being achieved — a fan-out wider than the file-disjoint set pays
contention, not throughput, and it is what crashed the terminals. **A measurement is averaged, never
refused for a busy box.** A contended run is a valid measurement when it is repeated: run at least
five paired runs and report the per-pair mean with the observed spread. Contention is answered with
more runs, not with a quiet window, a refusal, or a guard override that prints a single contended
sample as a result. Do not hold an accepted campaign behind a quiesced box.

## 7 · Metadata policy

Some source files carry `//@ key: value` annotations inherited from the sibling tree. They are
descriptive only.

- **`//@` metadata is not mandatory and not authoritative here until a local gate reads it.** This
  repository has no `song index` or ontology validator, so an annotation changes no gate.
- **Do not proliferate annotations.** Add one only when it carries what the code cannot state. A
  metadata claim that matters becomes a test, a proof, or a line in a governing document instead.
- A comment earns its line by carrying a *why* the code cannot express. Do not restate the
  declaration in prose.

## 8 · Writing

- **No anthropomorphic or story prose.** State mechanism and evidence. Do not write that a table
  "wants", a map "refuses", a dial "decides", or a crate "believes". Name the value, the operation
  and the measured effect.
- **Standard mathematical language, with attribution.** Name every mathematical object by its most
  common standard term and notation (`R[x]/(x²−σ)`; elliptic `σ<0`, parabolic `σ=0`, hyperbolic
  `σ>0`; norm form, idempotent, zero divisor) — never by a project or legacy coinage. For a borrowed
  object cite the originator(s) first, then the reference (norm form: Gauss, *Disquisitiones
  Arithmeticae* (1801); dual numbers: Clifford (1873); split-complex numbers: Cockle (1848); softmax:
  Boltzmann (1877) and Gibbs (1902), Bridle (1990) for the name). The reference is retrieved and read
  before it is called evidence; a citation alone is a pointer (§9.3).
- Present tense for what is; past tense for what happened. No dates in prose; git has them.
- Lowest word count that keeps every path, hash, identifier and number.
- Tables and bullets over paragraphs. A qualifier belongs in the sentence that needs it.
- Do not narrate a plan code (`E1`, `Wave 2`) in a durable document. Describe the change in terms
  that resolve to the tree: paths, gate names, theorem names.
- A number from a document is a hypothesis until the command is re-run.
- **State per-function complexity where it is not O(1).** A function whose temporal cost or spatial
  footprint is not O(1), or is not immediate from its type, carries a comment at its definition: the
  temporal and spatial complexity, the measure they are counted in (comparisons, allocations, bytes,
  words, bytes copied), and the derivation (the recurrence or loop and why it closes). A complexity
  figure is a claim like any other: it names the recurrence or the command, never a bare big-O.
  Trivial O(1) accessors, projections and constructors carry none.

## 9 · Agents and escalation

### 9.1 Default worker and trigger-selected escalation

`space-bunny-alpha` (`openrouter/stealth/space-bunny-alpha`, OpenRouter) executes every
dispatch-ready task by default. It is free and the fastest free route measured: 400 tokens in
2.70 s wall, 147.9 tok/s, TTFT 1.19 s. The OpenCode Zen route to the same family
(`opencode-go/space-bunny-free`, `opencode/space-bunny-free`) measured 6.95 s wall / 57.5 tok/s /
TTFT 6.95 s on the same prompt — it is slower; use the OpenRouter route. **The free window ends
2026-10-05**; until then dispatch Space Bunny on every bounded task and lead-verify each result. It
is mid-tier (cloaked weights, MMLU-Pro ≈ 75%): adequate for bounded, gated, ≤3-file tasks, not a
proof-search ceiling. `deepseek-flash` (`deepseek/deepseek-flash`, the DeepSeek API — never the OpenRouter route) is the paid escalation lane,
taken only when a mandatory trigger holds or the free lane fails the same bounded task twice.
Anything beyond that escalates manually to the owner. `gemini-flash` (Vertex prepaid) remains an
available lane but is not the default, and its balance is not spent on ordinary work. Escalation is
selected by the artifact at risk, not by a subjective difficulty estimate. A mandatory trigger, not familiarity or a first failed hypothesis, selects the model.
Mandatory escalation triggers override every worker lane. An unresolved proof or architecture
obligation goes to a fresh session on a different worker model; the Lean kernel and the mutation
control arbitrate, and contract or claim-grade questions go to the lead. A contract, grade or
architecture question that would otherwise wait on the owner goes to **Sol** on the owner's ChatGPT
Pro subscription signed into OpenCode, model `openai/gpt-6.1-sol`
(`/home/socol/oc-tmp/bin/sol.sh <prompt-file> [model]`): the lead records the
answer under `/home/socol/oc-tmp/escalation/`, acts on it and reports the verdict. The lead does not
idle waiting for the owner.

**Escalation cost discipline.** Sol runs on the owner's ChatGPT Pro subscription (OpenCode model
`openai/gpt-6.1-sol`), not a metered API key. Attach excerpts, never whole trees. Astra and Terra
remain historical records.

| role | assignment | mandatory trigger | required return |
|---|---|---|---|
| Space Bunny (`openrouter/stealth/space-bunny-alpha`) | **default bounded worker for every dispatch-ready task**, free, OpenRouter, until 2026-10-05 | task is ≤3 files, one task-specific gate, no escalation trigger holds, and the free window is open | edited artifact, exact gate output, measurement, decision |
| DeepSeek Flash (`deepseek-flash`, `deepseek/deepseek-flash`, the DeepSeek API — never the OpenRouter route) | paid escalation worker, taken only when necessary | a mandatory escalation trigger holds, or the Space Bunny lane failed the same bounded task twice, within ≤3 files and one gate | edited artifact, exact gate output, measurement, decision |
| Gemini Flash (`gemini-flash`) | non-default lane, not for ordinary work | the free window is closed, or the owner directs the lane | edited artifact, exact gate output, measurement, decision |
| independent verifier (fresh session on a different model from the writer, with user approval) | verification, proof and architecture escalation | passing requires changing a contract premise, a Lean theorem or its attribution, a claim grade, a canonical-corpus theorem, or a control still passes under its stated mutation; or the change exceeds 3 files or one gate, overlaps another task's paths, modifies a repository or governing-document boundary, or governing documents conflict | proof or gate verdict with named premises; architecture boundary, task decomposition, integration order, residual risk |
| lead | closure authority | escalation verdict received or wave barrier reached | merge, rejection, or the next dispatch-ready wave |

- A worker continues only while the task stays within its declared files and one task-specific
  gate and no escalation trigger holds. When both the verification trigger and the architecture
  trigger hold, both verdicts are required.
- A verification escalation states the theorem, contract, claim grade, control or missing
  artifact at issue, and the exact decision requested. It adjudicates proof sufficiency through the
  kernel and the mutation control; it does not waive a gate or a DoD pillar.
- An architecture escalation states the affected repositories, overlapping paths or conflicting
  documents, and the proposed architecture boundary. It defines interfaces and decomposes work; it
  does not waive a register, bound or control.
- Every handoff includes `SCOPE`, declared paths, the current diff, an available red command and
  output, available measurements, the wall-clock budget with observed elapsed time, and one explicit
  question. A missing gate, theorem, attribution, diff or measurement is named as the residual
  obligation, never invented. Conversation history is not a handoff.
- The independent verifier returns a bounded decision or artifact. Execution returns to the default
  worker when that result restores a ≤3-file, one-gate task, unless a speed-lane condition holds;
  escalation does not transfer ordinary work.
- Standing user authorization covers `openrouter/stealth/space-bunny-alpha` invocations and
  fresh-session verification that meet a trigger, and `deepseek-flash` only where a mandatory trigger
  holds. Spend no other paid lane without the owner. Only the lead closes a wave, merges, commits or
  pushes unless the user delegates that authority. A missing required verdict keeps the task open.

### Mandatory inference optimization admission

Before starting an inference optimization implementation worker, the lead must invoke
`python3 tools/dispatch/admit.py --launch <proposal.json>` in its task worktree. This rule applies
to every provider and native-agent route; a route that cannot execute its bound worker through
this launcher is not admitted for optimization implementation. Read-only analysis and separately
scoped extraction, refinement and cost-model tasks may establish the prerequisites without
changing an inference implementation. Admission does not replace their gates or the epic barrier.

`admit.py` is a compatibility entry into `tools/dispatch/run` and the Rust adapter; the executing
core is extracted and refined under `proof/dispatch`, while IO/process/hash provenance remains a
tested/trusted adapter boundary.

The lead-reviewed `tools/dispatch/contracts.json` fixes the real extraction root, source/proof
identities, before/after observations including refusals, reachable joint domain, resource model,
owned paths and exact worker argv. `proof/DispatchAdmission.lean` requires a reachable joint
witness, universal preservation and strict improvement in that registered model. Registration
requires the executing-source/refinement/cost correspondence, its rejecting controls and the
independent verdict under §9.1; a worker cannot register its own optimization. Registry/checker
changes are separate tasks. A cost-model proof is not a hardware speed measurement.

The registry is empty until such a contract is accepted. A cited theorem, Boolean approval,
`--check` output, cached receipt or sampled numerical tolerance cannot authorize a worker.
`--self-test` uses synthetic marker workers only and establishes no inference theorem. Each launch
freshly checks extraction, the typed certificate, its axiom cone and current identities before
starting the bound worker within one budget. New contracts require a successful production-path
positive and attributable refusal/restoration controls; synthetic controls alone cannot accept one.

The launcher enforces rejection only on its own execution path. Direct shell/native tools are not
intercepted by a repository script. Protection of the registry/checker, tool/dependency integrity,
source/cost registration and concurrent mutation between hashing and launch remain trust boundaries.
Host-side interception and immutable source snapshots remain open; this rule supplies operational
discipline without claiming those mechanisms exist.

### 9.2 Agent conduct

- Lead with the result, the exact measurement and the decision.
- Ground every claim in a primary artifact: read the source, run the gate, quote the line.
- Resolve via executables; execute the command and inspect the diff rather than describing it.
- Maintain forward momentum: complete every actionable subgoal, and on a barrier report the exact
  obstacle and advance the remaining units.
- Frame any necessary assumption in one sentence, then execute toward completion.
- After three materially different failed approaches to one obstacle, or five patch-and-rebuild
  cycles on one proof, stop and report the smallest unresolved goal. Escalate to the lead with the
  exact failing goal, the approaches tried, what is on disk and the one decision requested. Never
  patch past the cap, never commit, never decide scope. Ladder:
  [`../AGENTS.md`](../AGENTS.md) §Escalation ladder.
- Do not commit, push, post externally, or modify another active workstream unless the user asks.
- End every slice with: changed files, acceptance results (command and exit status), what is proved
  versus tested versus open, and the next bounded task.
- The owner-layer authority for this workspace is [`../AGENTS.md`](../AGENTS.md) §Authority, hold,
  momentum. A `HOLD` marker (repo root or `~/Workspace/HOLD`) stops new work; run the preflight
  `gate hold` (built from [`tools/gate`](tools/gate)) before any write or dispatch, and treat a
  marker as releasable only by the owner.

### 9.3 Assessing work

Assess an artifact against its stated goal and its named gates. A limitation is an unrun gate with a
path — a row in an exit table, a theorem not yet linked, a machine not yet measured — and it carries
its grade (§5) and its command.

- "The Rust set is green: `cargo test` 98, extraction fresh, `lake build` 1708
  jobs."
- "The next unrun gate is `cd proof/corpus && lake build`."
- "Residual obligation: the bridge build, at grade `tested`."

An external or comparative claim ships with its source: retrieve the paper or repository locally
first (arXiv, GitHub), read it, and cite the exact file, revision, section or theorem, left where the
reader can open it.

- "`lean-dojo/TorchLean@<rev>`, `NN/Proofs/<file>.lean`, `<theorem>`: the autograd statement this
  refines."
- "`Substrate/Ski/CombForcedSelector.lean:18` in the local corpus: the file's own caveat on the word
  'forced'."
- "Source not yet retrieved: the comparison is an `anchor`, not evidence."

A citation is a pointer until the artifact is in hand and read (§5). A finding resolves to a gate, a
digest, a theorem or a retrieved source.

### Job board (Operately)

Operately is the primary job board for the Sanguine stream (`apeiron`, `song`, `penglai`). Check it
before starting tracked work and move the task's status as the work proceeds; receipts and evidence
stay in this repository. The board tracks status; it does not replace the epic/wave/task
decomposition, a gate or a control.

- Board: company `TWN`, space `Sanguine` (`sanguine-COpRXpkvCi19pKoDZnOPg0`), Tasks tool — one
  board across the three repositories; projects `apeiron`, `song` and `penglai` hold their scope.
  Statuses: Not started, In progress, Done, Canceled.
- OpenCode MCP: server `operately`; `tools.operately.list_tasks`, `get_task`, `update_task_status`
  (a status ID, value or label from `list_task_statuses`) and `create_task`.
- REST fallback for reads: `https://projects.tas.twn.network/api/external/v1/...`, header
  `Authorization: Bearer $OPERATELY_API_KEY`; the token is untracked (`.env`; owner-provided).
  `spaces/list_tasks?space_id=sanguine-COpRXpkvCi19pKoDZnOPg0` and `tasks/get?id=<task>` are
  verified.
- Tasks were seeded from each repository's planning documents and each description names its source
  paths. New tasks come from a repository plan or owner direction, not from board invention.
- Instance quirks: REST `create_space` and `tasks/update_status` return 500 — use MCP (or the UI)
  for those; OpenCode can report a response-schema error for a mutation that landed — confirm with
  a read.

## 10 · Failure register

Observed failures in this programme's histories and the rule each one fixes. A row cites an artifact;
re-run the path before quoting a number. Snapshot facts (branch counts, a timed-out run) are evidence
of that snapshot, not standing claims.

**Wall-clock budget.** Every local gate invocation declares a numeric wall-clock budget and enforces
it across the invocation and its descendants. Ten minutes is a failure ceiling, not a target:
completion at or beyond 600 seconds fails the budget. Stop on expiry, record elapsed time, exit
status and incomplete stages, and report the blocker. This supersedes older instructions permitting
900-second or 3600-second local runs; historical receipts remain unchanged. Lean memory, swap, thread
and lane-lock bounds still apply. Budget exhaustion is neither a semantic counterexample nor a
successful mutation control, and establishes no benchmark ratio. Redesign the check without weakening
its contract, coverage, controls or statistical protocol; retiring a required gate requires explicit
owner authorization and leaves its obligation open. Do not exempt a full campaign or a cold build, and
do not reset the budget through cosmetic subprocess splitting; a changed acceptance protocol needs its
own approved task, and a required gate that cannot complete within budget stays blocked.

The register is a discharge queue. Each row states the failure scope, present safeguards, residual
opening, owner and discharge gate; "unstatable" means impossible to accept within that scope through
the enforced acceptance path, not impossible to write or encounter. Record new failures immediately.
At a successful wave barrier the number of unresolved obligations, including accepted openings, must
not exceed its opening count; otherwise the barrier remains blocked pending discharge or explicit
owner revision of this requirement. Renaming, merging or moving entries does not discharge
obligations. An accepted opening remains open and appears in the obligation-11 trust-boundary
statement. Removing a row requires an owner-reviewed discharge receipt or authorized retirement, with
residual obligations retained. A discharge receipt identifies the mechanism and covered population,
shows it is on the mandatory path, records fresh positive, attributable rejection and restoration
evidence within budget on named source/tool identities, and retains exclusions and owner acceptance.

| failure | evidence | rule | present safeguard | open discharge |
|---|---|---|---|---|
| Optimization implementation preceded an executing-source and cost proof | `artifacts/affine-block-experiment/README.md`; `docs/epics/structural-decode-search.md`; `tools/dispatch/admit.py` | Admit inference optimization implementation only through the fresh proof-carrying launcher; exact-ring citations and sampled tolerances are not source/cost evidence. | Typed joint witness, preservation and resource-improvement certificate; closed registry; seeded rejection/restore controls; unregistered affine worker refused. | Lead: source-connected production contract and positive/control/restoration gate; host interception, immutable source snapshot and protected registry/checker remain open. |
| Bounded verification priced itself out: one CBMC harness bit-blasted a degree-four `i8` multiply past 25 minutes; 9 of 67 harnesses timed out | `apeiron` `f89b18b`; `docs/evidence/kani-table-harness-audit.md`; `docs/evidence/kani-retirement.md` | The local loop carries a wall-clock budget. A command that reaches ten minutes has already failed: stop it, record the failure, redesign or retire the check. Kani is retired; no replacement inherits its cost. | Authorized retirement of Kani; ordinary probe regressions retained. | Campaign-budget bounds for any replacement check. |
| Green gates over broken artifacts: a ladder was green over a program that did not compile; a check exited 1 silently for 29 commits; a regression gate carried the defect it exists to catch; a harness asserted the wrong sentinel polarity for four commits; a symbol gate refused before scanning a single document under the harness that ran it, because its index came from `git ls-files` in a copy with no `.git` | `song` `1fc16782`, `4f44dd6a`, `5263e8d4`; `apeiron` `cf94f81`, `aa1afcf` | Trust a gate only after a seeded defect makes it fail with an attributable diagnostic. Require the captured exit status and expected artifacts; missing evidence is not success. Quiet success is legitimate; silence instead of evidence is not. Timeout, missing tools and unrelated build failures are not the intended red. A gate is not accepted until its own refusal has been observed firing **under the harness that runs it**, not only by hand. | Seeded-defect rule; checker self-tests; relocation mutation controls. | Enforcement that every accepted gate carries attributable baseline, red and restore evidence with complete output. |
| Claims outran their theorem or domain: real-ring classification was attributed to wrapping words; cited theorems did not carry their claims; vacuity and false-positive checks missed cases | `apeiron` `f89b18b`; `sanguine-apeiron` `b6296b4a`, `688238aa` | Inspect the exact declaration, scalar domain and premises; require a checked implementation link and joint reachable witnesses. Neither an unrelated theorem nor an unreachable antecedent establishes the claimed result. | Evidence grades (§5); exact refinement naming; `AxiomAudit` axiom-cone checks. | Non-vacuity, premise correctness, constructor reachability and prose-to-theorem correspondence. |
| Measurement instrument defects: BEST-of-3 hid a fusion; a 105–144x gap was a convention mix; a 0.00 instruction count measured nothing | `fold` `77d87dc1`, `c3a83f9e`; `apeiron` `71d2c45` | A measurement names command, machine, metric and interval, and ships the control that can see the effect. Validate the instrument before the result. | Obligation 10 measurement definition. | Named per-instrument controls and protocol/source binding. |
| Representation defect before timing: signed-grid collisions reached 49% | `fold` `4e8534ca` | Full-key identity and oracle validation precede timing; a collision rate is a correctness fact, not an instrument note. | Historical collision fix in `fold`; not revalidated here. | Current full-key oracle coverage and required collision-population evidence. |
| Consolidation lost files: a 25→12 crate merge dropped files a second commit restored; a ~1M-line WIP reconciliation; `target/` tracked then untracked | `fold` `d84bda37`+`24ed7451`, `d697cd0b`; `apeiron` `d371b5a` | A removal is a migration or an authorized retirement: name the replacement or the retained obligation, inventory affected consumers, and gate reachability. Commits stay reviewable; build products are never tracked. | `python3 tools/rooting/doc_reach.py --check` refuses a registered documentation removal (`docs/moved.tsv`) whose successor does not resolve, whose removed path is still on disk, or that a live document still names; [receipt](docs/evidence/documentation-removal-reachability.md). | Unregistered removals are invisible without a previous-tree baseline: a documentation removal with no `docs/moved.tsv` row is seen by no gate (`docs/extraction-requirements.md`, `docs/apeiron-intent-escalation.md`); every removal requires its consumer inventory and its task-specific reachability gate now. |
| Gates ran against the wrong object: `GIT_DIR` beat `-C` and one gate ate the repository index; `lake env lean` selected a stale olean tree; a validation flag was never committed; a chained command at the repository root polluted a receipt; a whole-repo `rsync` left an orphan lock holder | `sanguine-apeiron` `97d28533`, `b6296b4a`; `docs/archive/autonomous-run-log.md` R12, R17; `proof/controls/relocation.sh` | Pin repo, cwd and environment per gate; name the artifact path and freshness per audit; bound copies and inherit the lane lock. | Runner-local hardening in `proof/controls/relocation.sh`: cwd selection, bounded copies, locking, cleanup, preflight. | Complete identity, environment and freshness enforcement for every gate; whole-invocation budgeting. |
| Dispatch assumptions failed on contact: provider config not hot-reloaded, insufficient balance, `bool::then` emitted as an unsupported external | `docs/archive/autonomous-run-log.md` D5, R18 | Verify the run environment before evidence; prefer source forms the extraction fragment supports. | Relocation `preflight()` covers its own runner inputs. | No complete dispatch preflight established by review; provider state and extraction-fragment fitness uncovered. |
| Documents drifted from source: a front page claimed a face the compiler disproves; a header said open after the proof landed; a governing file taught a grammar no file uses | `apeiron` `0857035`; `sanguine-apeiron` `b39aa6a9`; `song` `0d2fd6ad` | Source and gates determine implemented behavior; correct descriptive drift within declared ownership, or sequence a document correction before closure. They grant no permission to weaken an approved contract; escalate normative disagreement. | Structural artifact-resolution checker (`tools/check-claims.py`); limited coverage, excludes `AGENTS.md`. | Prose-to-source correspondence and uncovered documents and claims. |
| Task branches and worktrees outlived their tasks: four `codex/open-*` branches at the base commit with empty diffs, `main` ahead of origin over a dirty tree (snapshot) | `git worktree list`; `git diff --stat main..<branch>` per branch | At every barrier diff each branch against its base and confirm with the owner; remove a finished branch and worktree only after integration or authorized abandonment and a clean worktree inspection. Never force-delete to satisfy hygiene; report the cleanup blocker. | None. | Barrier inventory, per-branch base diff, integration or authorized abandonment, clean inspection and owner confirmation. |
| A required campaign exceeded its declared budget: the REG-1 full profile exited 124 at its 600-second bound with no pair table or acceptance headline | `docs/evidence/open-area-performance.md`; `examples/reg1` | A required campaign that exceeds its declared budget remains blocked. Diagnose or redesign execution without weakening the registered population, controls or statistical protocol; acceptance-protocol changes require separate approval. | Budget wrapper and recorded receipt. | Budget discharge pending lead acceptance: the unchanged full campaign completed in 477.79 s below the 600-second ceiling, including 55 expected control rejections; strict acceptance exited 1. G1 remains open: 325 passing / 1125 failing pairs, `concurrency_unmatched=2`, `address_face_unmatched=4`, `full_acceptance=false`. [Receipt](docs/evidence/locator-reserve-quadratic-and-reg1-attribution.md). Owner: lead. |
| Addressing proposals outran their rate: a hidden-state lattice cache (≤8.7% coverage), whole-key attention addresses (occupancy = t, α = 1.00 at layers ≥6) and suffix tables on human text (36–71% precision) were proposed as O(1)/O(log t) mechanisms; each addressed a contextual activation whose rate at quality-preserving precision exceeds log₂ t | `docs/evidence/quotient-cache-probe.md`; `docs/evidence/address-attention-probe.md`; `docs/evidence/resonance-ladder.md` | Hoisting, tables and codebooks pay only through collisions. Before proposing one, state the addressed object's source (fixed weights, token-id function, exact prefix, or contextual activation), estimate its rate R(D) at the required distortion against log₂ of the population, and measure distinct-count growth on a sample. Contextual activations at quality-preserving precision are unique; do not claim sublinear cost from addressing them. | Occupancy and growth-exponent columns in the probe receipts. | A pre-build rate check in every addressing task prompt. |
| Lean toolchain mixed in the shared corpus cache: `lake -d proof/corpus build …` from the repository root (no root `lean-toolchain`) selected the nightly toolchain and began rebuilding dependency oleans with it | [receipt](docs/evidence/lean-toolchain-mixing-incident.md); nightly trace `/tmp/opencode/scan-locator-check.log`; recovery `/tmp/opencode/scan-locator-recover.log` | Run Lean as `cd proof/corpus && lake build …`; never `lake -d` from a directory without the package's `lean-toolchain` | Recovery rerun under the pinned v4.31.0 rebuilt the affected dependencies; the trace line names `leanprover--lean4---v4.31.0` | Toolchain-identity preflight for every Lean invocation |
| tmpfs exhaustion stalled a Lean lane: the 32G `/tmp` tmpfs fell to ~285M free mid-slice (29G used, led by finished 18G copy trees `spike-locate` and `verify-w5t15` plus stale wave/control copies) | `/tmp/opencode` `du` output; `APEIRON-INT-TRANSCENDENTAL-W1-T2C` report; `df -h /tmp` | Disposable copies live outside the tmpfs: scratch and control copies under `/home/socol/oc-tmp`, only named receipts and logs under `/tmp/opencode`; prune a finished slice's copy at its receipt, preserving cited small files | Lead cleanup reclaimed ~26G while keeping every `*.log`; the stalled worker switched `TMPDIR=/home/socol/oc-tmp` | A tmpfs-headroom check before dispatch and a per-slice prune step in the integration checklist |
| Cross-worktree target sharing poisoned timed builds: live-tree runs of `spill_read_probe` measured the ported scratch code at 2.02–2.04 ns/key and "passed" while the same sources measure 9.26–9.48 ns/key after a forced rebuild; the porting task's `CARGO_TARGET_DIR` pointed at the live tree, so `target/` held a sibling worktree's objects, and its cached warnings named `src/table.rs:84` against the live line 97; an early read-stats direction claim came from the same contaminated set | `/tmp/opencode/w4a-fix-run.log`, `/tmp/opencode/w4a-clean-uninstr2.log`, `/tmp/opencode/w4a-clean-instr2.log`; apeiron `715592c` | A timed or acceptance run binds to a source identity: force a rebuild (`touch src/lib.rs`) or use a private target directory, and check cached warnings' line numbers against the tree before quoting a figure | The harness header records the rule and the clean builds; the baseline was re-run after a forced rebuild | An enforcement that refuses a timing claim whose target was shared with another worktree |
