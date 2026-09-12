# Plan — <epic title>

- **Owner:** <name>
- **Intent:** `docs/intent/<slug>.md`
- **Spec:** `docs/spec/<slug>.md`
- **Source revision:** <git rev-parse HEAD>; working tree <clean|dirty, list files>
- **Status:** draft | approved

## Goal and completion condition

<The wave-level barrier. Every task gate green, with its negative control run.>

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && lake build
```

## Non-goals

- <non-goal>
- <non-goal>

## Wave map

Tasks in one wave touch disjoint files. One in-progress owner per file. Shared build products
(`proof/ApeironWord.lean`, `target/`, `proof/.lake/`, `proof/corpus/.lake/`) are one lane each.

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | <task> | `<path>` | <owner> | `<command>` | <mutation that must turn it red> |
| W1 | <task> | `<path>` | <owner> | `<command>` | <mutation> |
| W2 | <task, blocked on W1> | `<path>` | <owner> | `<command>` | <mutation> |

## Task blocks

Copy one block per task. A task is ≤3 files, one arc, one gate command whose exit 0 closes it, and a
negative control where soundness is at stake.

### <W1-T1> · <title>

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

## Barrier checklist

Run at the wave barrier, not before:

- [ ] Every task gate exit 0, with its negative control run and restored.
- [ ] No task touched a file owned by another task.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` pass.
- [ ] `proof/regen.sh --check` reports fresh; `proof/corpus` `lake build` passes.
- [ ] Source and tool digests recorded.
- [ ] Claims registered at their evidence grade; citations not called verified.
- [ ] Trust-boundary statement present.

## Stop and rollback conditions

- <condition that stops the wave and what to report>
- <the smallest unresolved goal if three materially different approaches fail>
