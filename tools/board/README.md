# `tools/board` — Operately board driver

Shared protocol for the three-stream Operately board (space `sanguine`, projects
`apeiron`, `song`, `penglai`). One tool, one protocol, each repository keeping its
own source of truth.

The board **tracks status**; it does not replace the epic/wave/task decomposition,
a gate or a control (`AGENTS.md` §9.3). This tool makes the board *drivable*: a
session reads `next` and starts, a wave boundary reads `--check` and stops on
drift, and every reconciliation leaves its audit trail on the board.

| | |
|---|---|
| gate | `python3 tools/board/board.py --check --controls` (exit 0) |
| tests | `python3 -m unittest discover -s tools/board -p 'test_*.py'` |
| live board | company `TWN`, space `sanguine-COpRXpkvCi19pKoDZnOPg0`, apeiron project `apeiron-ExNAtv3hObMmq01Ab20DfA` |
| credential | `OPERATELY_API_KEY` in the environment; the tool never reads it from a file and scrubs it from every message |
| rules | `python3 tools/board/board.py --rules` |

## Protocol

### One writer per row

The **lead** creates, closes, re-dates and re-describes a row. A **worker** moves
only its own task's status, and read-confirms it.

This instance answers `HTTP 500` from `tasks/update_status` for a mutation that
**landed**, so a transport error is not evidence of refusal. `set-status`
therefore always reads the row back, and the read-back is the verdict. The same
holds for `post`: the check-in is read back and must be readable before the mode
reports success.

### A dispatch names its row

Every row's description carries its own labelled field:

```
Board: `APEIRON-BOARD-W1-T2`.
```

or, for a lane that has no board row:

```
Board: none -- internal lane.
```

The named row is the task's own id or the plan row it serves. A row without the
declaration is a finding (`task-without-board-declaration`), as is a declaration
pointing at a row the task does not serve. A mention in prose (`the row is on the
board: ...`) is **not** a declaration: the field is a labelled line of its own.

A fold without its board move is incomplete, and the reconciliation check says so.

### The reconciliation check is the red

`--check` compares the board's non-closed tasks against the repository plan's
non-closed rows and classifies **every** row:

| class | meaning |
|---|---|
| `mirrored` | the board row names the plan row, or its slug does |
| `stale` | the board row is non-closed while its plan row reads landed/closed |
| `internal-by-design` | no plan row, and the row *names* that in its description |
| `missing` | no match either way — drift |

Drift is a named line and a nonzero exit. A row with no project is checked too: it
is outside a project-scoped selection, which is exactly why it drifts.

**Drift stops dispatch.** `--check` is a wave-barrier gate, not a report to read
afterwards.

### The check-in is the audit trail

`post` writes the reconciliation record — the stamp line naming the plan file and
its sha256, the classification tally, every finding — as a project check-in, and
**read-confirms** that it landed. Dry-run without `--apply`. Status defaults to
`red` on drift.

### Due dates only from dated facts

The tool never writes a date and never infers one. A `due:` field in a description
is a claim about the world, so it is reported (`due-date-without-dated-fact`)
unless the row carries a dated fact behind it: a receipt or verdict reference, a
sha256, or a revision. Only the lead re-dates.

## Sources of truth

Named explicitly in every `check`/`post` output with the sha256 of the file read,
so a finding is bound to the plan revision it was computed against.

| project | source |
|---|---|
| apeiron | `docs/plan.md` (epics/waves/tasks tables) |
| song | `plan/ledger.tsv`, `plan/fleet.tsv` |
| penglai | none declared — reported as `plan-source-undeclared`, not guessed |

`--plan PATH` (repeatable) names them; a relative path resolves against
`--repo-root`. A board document may carry its plan inline, which is how the
offline fixtures work: no key, no network, no repository read. `--fixtures` makes
a run hermetic — the inline plan *is* the source of truth.

A row is identified by the task id it carries in backticks (`APEIRON-E6-W4-T1`),
else by its section and wave. Closed-ness is the row's **own wording**: a row that
says landed, closed, retired, done, complete, historical, superseded or returned
is closed, every other row is non-closed. This is a text classification over a
prose document, not a verified parse; `--check` prints every classification so a
reader can audit it.

## Modes

| mode | what it does |
|---|---|
| `list` | status, milestone, the **gate** and the **source path(s)** parsed from the description, id, name |
| `next` | the highest-priority actionable task (`pending` / `in_progress`), ordered by milestone then id, with its full id, gate and sources |
| `check` | reconcile mode; nonzero when the board cannot drive or has drifted |
| `post` | write the reconciliation record as a project check-in, read-confirmed |
| `set-status` | move one task's status, read-confirmed. Dry-run without `--apply` |

A gate is a *runnable* command, not a path that happens to be a script:
`tools/rooting/check.py` is a source path, `python3 tools/rooting/check.py --check`
is a gate.

## Rules

`--rules` prints them. They are a stable interface — the control suite asserts on
the names, so a rename breaks a control instead of quietly weakening it.

**Board contract** (a row that cannot be started from the board):
`milestone-without-tasks`, `task-without-source-paths`, `task-without-gate`,
`done-without-receipt`, `write-did-not-land`.

**Protocol** (a dispatch names its row; a due date names a fact):
`task-without-board-declaration`, `due-date-without-dated-fact`.

**Reconciliation** (drift against the named source of truth):
`plan-source-undeclared`, `drift-plan-row-not-on-board`,
`drift-board-row-not-in-plan`, `drift-board-open-against-closed-plan`.

**Extension** (same failure class: the board cannot drive):
`task-with-unknown-milestone`, `unattributed-task`, `board-project-missing`.

## Controls

`--controls` is the control suite, and the gate is `--check --controls` together:

- **baseline** — the in-file fixture board: 0 findings.
- **ten seeded defects**, each red *by name*, with the subject named in the finding:
  the four original contract defects, three reconciliation defects, and the
  missing/mis-pointed board declaration plus the unsupported due date.
- **restore** — the unmutated baseline again: 0 findings.
- **post read-back** — the record is confirmed by a read-back.
- **post refusal** — a write that did not land is reported as not landed.

A check that cannot fail has no weight: `test_a_seeded_defect_stops_firing_when_the_rule_is_removed`
replaces `check` with a stub and asserts every seeded control reports `CONTROL FAIL`.

## Exit status

`0` success; `1` a finding (rule or drift); `2` usage, transport or control-suite
failure; `3` `next` found no actionable task.

## Known limits

- **The space-level / project-level split has no move-task capability.** Over
  REST, a task can be enumerated (`spaces/list_tasks`) and its status read and
  written (`tasks/update_status`), but there is no route to move a task *into* a
  project or a milestone. A row sitting at space level is therefore reported by
  `--check` and must be moved by the lead in the UI.
- `projects/list_tasks` is not exposed over REST, so a project task carrying no
  milestone is invisible to `--live`.
- Task **comments** are not read over the external v1 API, so a receipt held only
  in a comment is not seen by `--live`.
- `projects/list` is company-wide; `--project` filters by name over that
  inventory.
- Plan rows are classified from row wording, not from a machine-readable field.

These four coverage notes are printed by `--live` on every run.