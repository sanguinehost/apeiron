# `tools/hooks` — the gates git runs for you

Three git hooks, one checker, one installer. They enforce the fast tiers of
`AGENTS.md` §6 and the per-commit half of the board protocol in §9.3. Nothing
here replaces a wave barrier, and no hook substitutes for running the gate.

```sh
tools/hooks/install.sh            # point git at .githooks (commits nothing)
tools/hooks/install.sh --check    # report what is configured
python3 tools/hooks/lane.py self-test   # the declaration controls, 9 cases
```

## Tiers

| tier | command | budget (declared) | measured here | why this tier |
|---|---|---|---|---|
| pre-commit | `cargo fmt --all -- --check` | 90 s step / 180 s hook | 0–1 s | formatting is per-commit by nature; rustfmt compiles nothing |
| pre-commit | `python3 tools/check-claims.py` | 90 s | 1.34 s live, 2 s with the corpus | a citation either resolves now or not, and the corpus does not move under a commit |
| pre-commit | `python3 tools/hooks/lane.py staged` | 90 s | 0.00 s snapshot, 2.65 s live | a declaration in the *staged change* is checked before the commit exists |
| commit-msg | `python3 tools/hooks/lane.py message $1` | 90 s | 0.1 s snapshot, 2.7 s live | the declaration must exist; only this hook sees the message |
| pre-push | `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 420 s step | 8.15 s cold, fresh clone | needs a check build; not a commit-shaped cost |
| pre-push | `cargo test --locked --lib --test table --test native_locator` | 420 s step | 203.30 s cold (195.23 s of it `--lib`) | codegen plus a 195 s suite; too heavy per commit |
| pre-push | `python3 tools/board/board.py check --controls --plan docs/plan.md --live` | 120 s | 3 s, 170 findings over 8 rules | a wave-boundary reconcile, reported and configurable |
| barrier | `cargo test --workspace --all-features` | 600 s ceiling | exceeds budget under load | the full workspace selection; the barrier runs it |
| barrier | `proof/regen.sh --check` | 600 s ceiling | extraction, minutes | the pinned Aeneas/Charon toolchain |
| barrier | `cd proof/corpus && lake build` | 600 s ceiling, lane lock | Lean build | the refinement package |

Measured on this machine (`cargo 1.97.0-nightly`, `rustc 1.97.0-nightly`) in a
fresh `git clone` at `122835c`; receipts under `/home/socol/oc-tmp/hooks/`.

## Why three hooks, and why the declaration is split

A git `pre-commit` hook is not given the message being committed. At that point
`COMMIT_EDITMSG` still holds the **previous** commit's message — verified with a
probe hook and `git commit -F` — and git writes the new message file after the
hook returns. So:

| hook | what it may see | what it enforces |
|---|---|---|
| `pre-commit` | the staged change only | a declaration carried by the staged change is resolved against the board; absence is *not* a finding here |
| `commit-msg` | the message file, as `$1` | the declaration must be present, in the message or the staged change, and the row it names must exist |

Requiring the field in `pre-commit` would refuse every commit that declares its
row in the message; reading `COMMIT_EDITMSG` there would read the wrong message.
The split is the only shape git allows.

## Why the reconcile is not a commit gate

`board.py check` compares every non-closed board row against `docs/plan.md` and
exits 1 on any row it cannot mirror. Between wave boundaries that is red **by
design**: a dispatch is live before its plan row is written. A per-commit hook
that enforced it would block every commit in the window between a plan edit and
its board move, so the reconcile runs at the wave boundary and the pre-push hook
only *reports* it.

The per-commit requirement is smaller and always enforceable: the commit
declares its row. That is what `lane.py` checks.

## The declaration

Every commit carries one labelled field on its own line, in the message or in
the staged change:

```
board: <row-id>
board: none -- internal lane
```

The field is parsed by `tools/board/board.py` itself (`board_declaration`,
`declared_row`, `slug`), so the hook enforces the driver's rule rather than a
second spelling of it. A named row must exist and sit in a milestone of a
project: a row outside a milestone cannot be selected or ordered, which is the
`unattributed-task` failure class the driver already reports.

Name the **full row id** (`m2-apeiron-rooting-gate-FYyCFN1P8Kgu2Dg381dxCZ`).
The live board currently holds an old canceled row and a current row for several
plan rows, so a bare prefix matches two rows and is refused as
`board-row-ambiguous`.

| rule | meaning |
|---|---|
| `board-declaration-missing` | no `board:` field in the message or the staged change (`commit-msg`) |
| `board-declaration-conflict` | the message and the staged change name different rows |
| `board-row-not-found` | no row matches the token by id, name slug, plan id or its own board field |
| `board-row-ambiguous` | the token matches more than one row |
| `board-row-not-in-milestone-bearing-project` | the row exists but carries no milestone, or its milestone carries no project |
| `board-unavailable` | a row was named and no board could be read to check it |
| `board-driver-missing` | `tools/board/board.py` is not importable, so the rule cannot be enforced |
| `staged-state-unreadable` | git would not report the staged state |

## The board a commit is checked against

1. `APEIRON_HOOKS_BOARD_SOURCE=<path>` — a schema-1 board document, read exactly
   as `board.py --source` reads it. Offline and fast.
2. the live board over REST when `OPERATELY_API_KEY` is in the environment
   (`tools/board/README.md`; the key is never read from a file).
3. nothing — and a commit naming a row is then refused by `board-unavailable`,
   with the fix printed. A commit declaring `none -- internal lane` needs no
   board and passes without one.

Write a snapshot with:

```sh
OPERATELY_API_KEY=… python3 tools/hooks/lane.py capture --out /tmp/board.json
```

`capture` re-reads the file it wrote and compares every row's milestone and
project with the live read, so a lossy snapshot is reported rather than used.

## Configuration

| variable | default | effect |
|---|---|---|
| `APEIRON_HOOKS_PRECOMMIT_BUDGET` | `180` | total pre-commit budget, seconds |
| `APEIRON_HOOKS_STEP_BUDGET` | `90` | per pre-commit step |
| `APEIRON_HOOKS_COMMITMSG_BUDGET` | `90` | the commit-msg step |
| `APEIRON_HOOKS_PREPUSH_BUDGET` | `540` | total pre-push budget |
| `APEIRON_HOOKS_PUSH_STEP_BUDGET` | `420` | per pre-push Rust step |
| `APEIRON_HOOKS_BOARD_BUDGET` | `120` | the reconcile step |
| `APEIRON_HOOKS_BOARD_CHECK` | `report` | `block` makes reconcile drift refuse the push |
| `APEIRON_HOOKS_BOARD_PLAN` | `docs/plan.md` | source of truth passed to the reconcile |
| `APEIRON_HOOKS_PUSH_SCOPE` | `rust` | `always` runs the Rust steps on every push; `rust` runs them only when the pushed range touches `*.rs`, `Cargo.toml`, `Cargo.lock` or a rustfmt config, and prints the skip |
| `APEIRON_HOOKS_BOARD_SOURCE` | unset | schema-1 board document for offline checks |

A budget is a refusal, not a warning: a step that reaches its budget fails the
commit or push and says so (`AGENTS.md` §10). No variable disables a gate.

## What the hooks do not cover

- `git commit --no-verify` and `git push --no-verify` skip them, as they skip
  every hook. The wave barrier is the authority on a bypassed commit.
- Neither push-tier hook runs the barrier gates listed in the table above.
- The reconcile is reported, not enforced, in `report` mode. Drift it prints is
  open until the barrier runs it. In `report` mode a reconcile that cannot read
  the board is reported as `ERROR`, not as clean.
- `proof/` and `proof/corpus/` changes are not gated here at all.
- A hook cannot see an untracked file, so a violation that is never staged is
  never caught — the barrier still runs on the tree.
- Over REST there is no move-task route, so a row outside a milestone is
  reported, not repaired.

## Files

| path | role |
|---|---|
| `.githooks/pre-commit` | tier 1: formatting, citation resolution, a declaration in the staged change |
| `.githooks/commit-msg` | tier 1: the declaration, where git gives the message |
| `.githooks/pre-push` | tier 2: clippy, the named test selection, the reconcile |
| `tools/hooks/lane.py` | the declaration check, the board reader, `capture`, `rows`, `self-test` |
| `tools/hooks/install.sh` | `core.hooksPath`, `--check`, `--uninstall` |