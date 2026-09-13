# apeiron: working agreement

`R[x]/(x^2 - sigma)`, its `Cell<i64>` word kernel, and the table that consumes it. One Rust crate,
one Lean extraction package, one Lean refinement bridge. This file governs the work; the source and
the gates are the record.

Reply with the result and the evidence. State a decision or a blocker; do not narrate tool calls.

The sibling `song` tree is a different repository with its own ontology, registers and `./song`
gates. None of those apply here. Do not import its commands, its `R1`-`R5` registers, its `//@`
ontology, or its `law`/`spec` block semantics. Adopt only the lifecycle, the wave/task discipline and
the writing rules below.

## 1 · Reading order

Read the file, not a summary of it, not an agent's report on it, not memory. A citation names a path
that resolves; run the path before citing it.

| Stage | Document | Purpose | Exit condition |
|---|---|---|---|
| 1 | `docs/direction.md` | Decisions, evidence, open work, correctness items | Current stage and its unresolved obligations are known |
| 2 | `docs/kernel-boundary.md` | Layer ownership and the trust boundary | Which layer is proved, tested, measured or out of scope is known |
| 3 | `docs/flash-handoff.md` | Execution order, gates, specialist approval, stop points | The current stage's gate and stop condition are known |
| 4 | `proof/README.md` | Extraction and bridge, exact commands, what is **not** established | Tool pins, extraction selection and axiom cone are known |
| 5 | `README.md` | Crate behavior and the measured claims | Claimed numbers have a command behind them |
| 6 | exact source | `src/`, `tests/`, `proof/extract/lib.rs`, `proof/corpus/` | The symbols to change have been read, not recalled |

`docs/direction.md` and `docs/kernel-boundary.md` govern. `docs/flash-handoff.md` governs execution
order. Where a document disagrees with the source or a gate, the source and the gate win; fix the
document.

## 2 · Lifecycle

Every nontrivial epic proceeds through five phases, each producing a committed artifact. Templates
are in `docs/templates/`.

1. **Intent** (`docs/templates/intent.md`) — the problem, why alternatives were rejected, the
   invariants, the measurable success condition, the non-goals.
2. **Spec** (`docs/templates/spec.md`) — the contracts, the admissible domain and partiality, the
   Rust-to-Lean representation relation, the failure semantics, the evidence plan.
3. **Plan** (`docs/templates/plan.md`) — the waves, files, owners, one gate command per task, and
   the negative control where soundness is at stake.
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
- Do not run working-tree-mutating git commands (`stash`, `reset`, `checkout`, `restore`, `clean`).
  Preserve unrelated dirty work; a mutation control edits a throwaway copy, never the working tree.
- A task that needs conversation history to execute is not yet a task.

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
GATE:       command, cwd, observed red output; if absent, MISSING and a gate-first task
GRADE:      evidence grade and before/after result; unmeasured or not applicable with reason
```

`REGISTER` classifies the obligation; it does not import Song's deciders or R1–R5 certification.

## 4 · Definition of done

A task closes on its focused gate and control. A wave requires every child plus the shared gates.
An epic closes only when every row of its exit table is green on the same source/tool identity;
missing gates, witnesses or evidence keep it open. `docs/flash-handoff.md` owns the epic exits;
`CLAIMS.md` owns headline evidence. Historical completion records are not fresh acceptance.
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
5. **Fail-closed axiom cone.** `proof/regen.sh` audits the extraction for `axiom`/`sorry`/`admit`;
   `proof/corpus/AxiomAudit.lean` `#audit`s each headline and throws on anything outside
   `{propext, Classical.choice, Quot.sound}`. A `sorry` is a build failure, not a note.
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
11. **Trust-boundary statement.** State what the claim does not cover: rustc/LLVM codegen, the
     kernel ABI, GPU/SPIR-V, the `unsafe` container, or any layer the evidence does not reach.

At an implementation epic barrier, run all §6 Rust commands, including `cargo kani`,
`proof/regen.sh --check`, and `lake build` in both `proof/` and `proof/corpus/`, plus the named
controls. These are the local barrier set, not one existing aggregate harness. Missing tools block
acceptance. `CLAIMS.md` specifies the missing ledger/witness and perimeter checks: local trusted
checking code ≤5,000 physical lines, owned source files ≤1,500 lines, ≤12 immediate source files
per directory. Inventory external trust separately; neither the whole proof library nor a bridge
line count is the full TCB. Until the inventory and rejecting checks exist, size acceptance is open.

## 5 · Evidence grades

Every stated result carries one grade. The grade is part of the claim.

| grade | means |
|---|---|
| `proved` | A Lean kernel check (`lake build`) passes for a named theorem. If it is about Rust, the extraction is from the real source and a refinement theorem links the extracted body to the canonical definition, under the closed axiom cone. |
| `tested` | A `cargo test` executes the build path and its boundary cases. |
| `measured` | A benchmark or `perf` command ran, with a named machine, metric and interval. |
| `model` / `anchor` | A reading of the algebra onto a domain, or a design assumption. True as a mapping; not evidence of correctness. |

A citation to a Lean theorem, a paper, or a sibling repository is a pointer, not evidence. It becomes
`proved` only when the named theorem exists and the in-repo link is checked. **Never call a citation
alone verified.** Label an `anchor` as an `anchor`; an `anchor` reported as `proved` costs a redesign
when someone checks.

## 6 · Local commands

These are the gates. There is no `./song` gate in this repository; a command that only exists in the
sibling tree is not an acceptance gate here.

Rust:

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
cargo kani                 # bounded harnesses in src/proofs.rs (cfg(kani)); no feature needed
```

Proof regeneration (requires the pinned Aeneas/Charon under `target/aeneas/` or `AENEAS_HOME`):

```sh
proof/regen.sh             # extract, audit, build
proof/regen.sh --check     # fail on drift from the committed ApeironWord.lean
AENEAS_HOME=/path/to/aeneas proof/regen.sh
```

Bridge build (canonical corpus plus the refinement package):

```sh
cd proof/corpus && lake build
```

Evidence tooling:

```sh
python3 tools/check-claims.py       # ledger: resolving #audit, named gate, grade/revision; duplicate ID
python3 tools/check-assurance.py    # perimeter: <=5,000 lines total, <=1,500/file, <=12 files/directory
python tools/call-census/audit_reports.py --raw-root /tmp/opencode/census-v2
```

`check-claims.py` exits nonzero for a claimed headline with no resolving `#audit`, no gate or no
grade/revision, and for a duplicate ID. `check-assurance.py` exits nonzero when a bounded population
exceeds its limit or a declared member is absent; the crate's `src/`/`tests/` are currently inventory
(printed, not enforced) so the committed baseline is green. Control: delete one `#audit` line for a
claimed headline, or add source files until a bounded directory exceeds twelve; each turns its gate
red and restore returns it green.

Tool pins and the archive checksum are in `proof/README.md`. Record the versions a run actually used;
do not cite a pin from memory.

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
- Present tense for what is; past tense for what happened. No dates in prose; git has them.
- Lowest word count that keeps every path, hash, identifier and number.
- Tables and bullets over paragraphs. A qualifier belongs in the sentence that needs it.
- Do not narrate a plan code (`E1`, `Wave 2`) in a durable document. Describe the change in terms
  that resolve to the tree: paths, gate names, theorem names.
- A number from a document is a hypothesis until the command is re-run.

## 9 · Agents and escalation

- Work directly. No swarm, generic exploration agents, or automatic fan-out.
- **One specialist at a time**, and only for a specific unresolved proof, implementation or
  architecture obligation. Obtain user approval before each invocation; approval covers one
  invocation, not retries or session resumes.
- State before the invocation: agent, scoped files, exact question, evidence already obtained, and
  the acceptance command. Do not change permissions or launch agents through a shell/API to bypass
  the gate.
- After three materially different failed approaches to one obstacle, stop and report the smallest
  unresolved goal.
- Do not commit, push, post externally, or modify another active workstream unless the user asks.
- End every slice with: changed files, acceptance results (command and exit status), what is proved
  versus tested versus open, and the next bounded task.
