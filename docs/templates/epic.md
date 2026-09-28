# Epic template (intent, spec, plan in one file)

Rolled up from: `templates/intent.md`, `templates/spec.md`, `templates/plan.md`.

## Intent — <epic title>

- **Owner:** <name>
- **Source revision:** <git rev-parse HEAD>; working tree <clean|dirty, list files>
- **Status:** draft | approved
- **Spec:** `docs/spec/<slug>.md` (to be written)
- **Plan:** `docs/plan/<slug>.md` (to be written)

### Problem

<The concrete defect, gap or capability. One paragraph, present tense.>

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| <approach> | <the measured or proved reason it fails> |

### Completion condition

<One executable sentence. What command, or which theorem, is green when this is done.>

```sh
<the command whose exit 0 closes the epic, or the Lean build target that must pass>
```

### Invariants

<What must never break while the epic is open. Algebraic laws, layout, partiality, determinism.>

1. <invariant>
2. <invariant>

### Verification hypotheses

<Measurable claims the epic must establish, each with its evidence grade.>

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| <claim> | proved | `<lake build>` / `<theorem name>` |
| <claim> | tested | `cargo test --workspace --all-features --locked` |
| <claim> | measured | `<bench command>`, <machine>, <metric> |
| <claim> | model/anchor | <the mapping, stated as a mapping> |

### Non-goals

<What this epic explicitly does not attempt. Each entry prevents a scope expansion.>

- <non-goal>
- <non-goal>

### Trust boundary

<What the evidence will not reach: rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, the `unsafe`
container, or a layer outside the extracted fragment. State it here, before the proof is written.>

### Open questions

- <question, with the decision needed and who owns it>

## Specification — <epic title>

- **Owner:** <name>
- **Intent:** `docs/intent/<slug>.md`
- **Source revision:** <git rev-parse HEAD>; working tree <clean|dirty, list files>
- **Status:** draft | approved

### Scope and admissible domain

<What operation is specified. State the domain explicitly.>

| Item | Value |
|---|---|
| Operation | <Rust symbol or Lean declaration> |
| Domain | <every admitted input, with width/type/range> |
| Totality | total / `Option` / `Result` / checked / panics |
| Preconditions | <named, or "none"> |
| Postcondition | <the exact relation the result satisfies> |

No silent domain reduction. If a proof needs a narrower domain, that domain is named here and the
operation keeps its honest partial signature.

### Representation relation (Rust to Lean)

<The exact link. Name the real source, the extracted body, and the canonical definition it refines.>

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `<fn>` | `src/<file>.rs` |
| Extraction wrapper | `word_<fn>` | `proof/extract/lib.rs` |
| Extracted body | `<module>.<fn>` | `proof/ApeironWord.lean` (generated) |
| Canonical definition | `<Namespace.definition>` | corpus / `proof/corpus/WordRefinement.lean` |
| Refinement theorem | `word_<fn>_denote` | `proof/corpus/WordRefinement.lean` |

The wrapper fixes a type; it does not restate the formula. The refinement theorem is a link only if
it resolves and the axiom audit passes.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| <name> | <universally quantified relation> | proved | `<theorem>` |
| <name> | <behavior on boundary input> | tested | `<test fn>` |

### Failure semantics

<For each failing input, what the operation returns or raises. `None`, `Err`, panic, saturation and
the exact condition. An infallible signature over a lossy operation is a defect, not a contract.>

| Input class | Result | Named condition |
|---|---|---|
| <overflow / unrepresentable / empty> | <`None` / `Err` / panic> | <precondition or check> |

### Evidence plan

1. **Positive tests.** `cargo test --workspace --all-features --locked` covers the nominal path and
   the boundary cases above. Name the test file.
2. **Proof.** `proof/regen.sh` regenerates the extraction; `cd proof/corpus && lake build` checks the
   refinement theorem.
3. **Axiom cone.** `proof/corpus/AxiomAudit.lean` `#audit`s the headline; anything outside
   `{propext, Classical.choice, Quot.sound}` fails the build.
4. **Negative control.** <The mutation in a throwaway copy that must turn the gate red, and the
   command that shows it red. Restore and show green.>
5. **Measurement** (only for performance claims). <Command, machine, metric, before/after.>

### Trust boundary

<What this specification and its evidence do not cover.>

### Open questions

- <question and the decision needed>

## Plan — <epic title>

- **Owner:** <name>
- **Intent:** `docs/intent/<slug>.md`
- **Spec:** `docs/spec/<slug>.md`
- **Source revision:** <git rev-parse HEAD>; working tree <clean|dirty, list files>
- **Status:** draft | approved

### Goal and completion condition

<The wave-level barrier. Every task gate green, with its negative control run.>

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && lake build
```

### Non-goals

- <non-goal>
- <non-goal>

### Wave map

Tasks in one wave touch disjoint files. One in-progress owner per file. Shared build products
(`proof/ApeironWord.lean`, `target/`, `proof/.lake/`, `proof/corpus/.lake/`) are one lane each.

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | <task> | `<path>` | <owner> | `<command>` | <mutation that must turn it red> |
| W1 | <task> | `<path>` | <owner> | `<command>` | <mutation> |
| W2 | <task, blocked on W1> | `<path>` | <owner> | `<command>` | <mutation> |

### Task blocks

Copy one block per task. A task is ≤3 files, one arc, one gate command whose exit 0 closes it, and a
negative control where soundness is at stake.

#### <W1-T1> · <title>

- **Scope:** <the change and its completion condition, one line>
- **Files:** `<path>`, `<path>`
- **Depends on:** <task id or none>
- **Grade:** proved | tested | measured | model/anchor
- **Gate:**

```sh
<one command; exit 0 closes this task>
```

- **Negative control:** <mutation in a throwaway copy; the command that shows it red; restore and
  show green. "Not applicable — <reason>" if soundness is not at stake.>
- **Evidence to record:** <source sha256, tool revisions, command output, before/after number>

### Barrier checklist

Run at the wave barrier, not before:

- [ ] Every task gate exit 0, with its negative control run and restored.
- [ ] No task touched a file owned by another task.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` pass.
- [ ] `proof/regen.sh --check` reports fresh; `proof/corpus` `lake build` passes.
- [ ] Source and tool digests recorded.
- [ ] Claims registered at their evidence grade; citations not called verified.
- [ ] Trust-boundary statement present.

### Stop and rollback conditions

- <condition that stops the wave and what to report>
- <the smallest unresolved goal if three materially different approaches fail>
