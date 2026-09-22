# Relocation control receipt

`proof/controls/relocation.sh` is the rejecting control for the growable per-entry relocation
refinement (`word_stored_relocate_entry`, `word_regen_relocate_entry`, and the four
`Growable/Relocation.lean` `*_preserves` headlines). It proves the relocation evidence is
falsifiable, not merely stated. The `step2c-frame-*` checks extend it to the W6-T1 disjoint-write
frame headlines (`writeFlat_frame_flat`, `writeFlat_frame_spill`, `writeSpill_frame_*`).

## Hardening (APEIRON-EXTRACT-REP-RELOCATION-CONTROL-T2)

**Root cause of the ~57-minute hang.** The runner copied the whole repository into
`/tmp/opencode` with a bare `rsync`, held `/tmp/opencode/apeiron-lane.lock` around that copy and
around two `cp -a` of the `.lake/build` trees, and had no per-step bound. Under `btrfs-cleaner` I/O
pressure the `rsync` stalled indefinitely; killing the outer `timeout`/`bash` chain left the
`flock` and `rsync` processes alive, still holding the lane lock.

**Bounded design.**

- **Preflight refuses, it does not wait.** Free space on the copy filesystem must be at least
  `2048 MiB` (default) or `4096 MiB` with `RELOCATION_CONTROL_HEAVY=1` or
  `RELOCATION_CONTROL_FRAME=1`. A detected IO hog also refuses: 1-minute load ≥ `2 × nproc`,
  D-state tasks ≥ `nproc / 2`, `btrfs-cleaner` accumulating ≥ 20 CPU ticks/s, or an `rsync`/`dd`
  process running. The refusal prints a `SKIPPED` line and exits 2 without creating a copy.
- **Minimal bounded copy.** The whole-repo `rsync` is replaced by `cp -a --reflink=auto --parents`
  of only the paths the control reads, under `timeout 120`. Default copy is ~11 MiB. A copy
  timeout is `NOT-RUN` with rc=124 and exit 3; the runner never waits past the bound.
- **Every step is timed.** Copy, both `regen.sh --check` runs, the corpus builds and the `python3`
  mutations each carry a `timeout`.
- **The lane lock is an inherited file descriptor, not a child process.** `exec 9>lock` then
  `flock -w <T_LOCK_WAIT> 9`; the fd is closed in the child before `exec` so the transient scope
  cannot inherit it, and it is always released. No killed runner can orphan a lock holder.
- **The lock is not held during the copy.** It is acquired only around the Lean/Aeneas command.
- **EXIT/INT/TERM trap.** Kills the active process group (the step runs under `setsid`) and stops
  the transient `systemd-run` scope; then releases the lock and removes the copy.
- **Never writes the working tree.** All mutation happens in the copy. The copy is removed on
  exit; `APEIRON_RELOC_CONTROL_KEEP=1` keeps it for inspection.
- **Default is the freshness step.** The semantic-mutation and frame-mutation corpus builds are behind
  `RELOCATION_CONTROL_HEAVY=1`; the frame mutation alone is behind `RELOCATION_CONTROL_FRAME=1`. The
  heavy path keeps the existing spill mutation (`table.tag entry_address.place` → `0#u8`) and adds
  the new flat destination mutation (`reserved.resident.slots.set index slot` → `.set 0#usize slot`).

### Step and timeout table

| step | command (in the copy) | timeout | on timeout |
|---|---|---|---|
| preflight | free-space / load / D-state / `btrfs-cleaner` / `rsync`-`dd` check | — | `SKIPPED`, exit 2 |
| copy | `cp -a --reflink=auto --parents <minimal set> $RUN/` | `T_COPY=120` | `NOT-RUN` rc=124, exit 3 |
| baseline | `proof/regen.sh --check` on the unmutated copy | `T_EXTRACT=300` | `FAIL` (inconclusive) |
| step 1 | `python3` `live := true` → `live := false`; `proof/regen.sh --check` | `T_EXTRACT=300` | `FAIL` (inconclusive) |
| step 2 (heavy) | `python3` `let i ← table.tag entry_address.place` → `let i := 0#u8`; `(cd proof/corpus && lake build Growable.Relocation)` | `T_LEAN=300` | `FAIL` (inconclusive) |
| step 2 restore (heavy) | `(cd proof/corpus && lake build Growable.Relocation)` | `T_LEAN=300` | `FAIL` |
| step 2c (heavy or frame) | `python3` `reserved.resident.slots.set index slot` → `.set 0#usize slot`; `(cd proof/corpus && lake build)` | `T_LEAN=300` | `FAIL` (inconclusive) |
| step 2c restore (heavy or frame) | `(cd proof/corpus && lake build)` | `T_LEAN=300` | `FAIL` |
| step 3 | `proof/regen.sh --check` on the restored copy | `T_EXTRACT=300` | `FAIL` (inconclusive) |

The Lean/Aeneas commands run through the AGENTS.md §6 wrapper, exactly:

```sh
exec 9>/tmp/opencode/apeiron-lane.lock; flock -w 120 9
systemd-run --user --scope --unit=<unit> --collect \
  -p MemoryMax=6G -p MemorySwapMax=0 -- \
  env LEAN_NUM_THREADS=1 timeout --signal=TERM --kill-after=10 <T> <cmd>
```

The timeouts are overridable with `RELOCATION_CONTROL_T_{COPY,EXTRACT,LEAN,MUTATE,LOCK_WAIT}`.

### Default vs heavy vs frame

| mode | command | steps |
|---|---|---|
| default | `bash proof/controls/relocation.sh` | baseline, step 1, step 3 |
| heavy | `RELOCATION_CONTROL_HEAVY=1 bash proof/controls/relocation.sh` | baseline, step 1, step 2, step 2 restore, step 2c, step 2c restore, step 3 |
| frame | `RELOCATION_CONTROL_FRAME=1 bash proof/controls/relocation.sh` | baseline, step 1, step 2c, step 2c restore, step 3 |

### Invocation: run standalone, never nested in `flock`

The runner acquires `/tmp/opencode/apeiron-lane.lock` itself, as an inherited file descriptor
(`exec 9>lock` then `flock -w <T_LOCK_WAIT> 9`). The AGENTS.md §6 lane wrapper applies to the
individual `proof/regen.sh`/`lake build` commands *inside* the runner, not to the runner itself.
Invoking the control as `flock /tmp/opencode/apeiron-lane.lock bash proof/controls/relocation.sh`
nests the lock: the outer `flock` holds the lane and the runner's own acquisition times out, so the
baseline reads `FAIL lane lock busy` and the control exits 1. Run it standalone:

```sh
bash proof/controls/relocation.sh
# heavy: RELOCATION_CONTROL_HEAVY=1 bash proof/controls/relocation.sh
# frame: RELOCATION_CONTROL_FRAME=1 bash proof/controls/relocation.sh
```

A wall-clock bound may wrap the standalone invocation (`timeout 1500s bash
proof/controls/relocation.sh`); the runner's trap releases the lock and removes the copy on
`INT`/`TERM`, so no orphan lock holder survives. Do not add an outer `flock`.

## Receipt format

One line per check:

```text
relocation-control: <check> <PASS|FAIL|SKIP|NOT-RUN|WARN> <detail>
```

| check | meaning |
|---|---|
| `copy` | bounded copy completed; `NOT-RUN` on timeout. |
| `baseline` | `proof/regen.sh --check` on the unmutated copy. `PASS` = fresh; `SKIP` = stale extraction (a concurrent source lane is ahead of `proof/ApeironWord.lean`); `FAIL` = timeout, lock busy or extraction error. |
| `step1-freshness-mutation` | flat relocation write mutated; `regen.sh --check` must be red and its diff must name the mutated line. |
| `step2-semantic-mutation` (heavy) | spill relocation write mutated; `lake build Growable.Relocation` must fail inside `Growable/Relocation.lean`. |
| `step2-semantic-restore` (heavy) | restored spill write must rebuild `Growable.Relocation` green. |
| `step2c-frame-mutation` (heavy or frame) | flat relocation destination index mutated to zero; `lake build` must fail with the first `error: Growable/Relocation.lean:<line>` inside the `writeFlat_frame_flat` span (`theorem writeFlat_frame_flat` up to `theorem writeFlat_frame_spill`). |
| `step2c-frame-restore` (heavy or frame) | restored destination index must rebuild `lake build` green. |
| `step3-restore-freshness` | all writes restored; `regen.sh --check` must be green. `NOT-RUN` when the baseline is `SKIP`. |
| `working-tree-guard` | `proof/ApeironWord.lean` digest must be unchanged by the control; `WARN` is informational. |
| `RESULT` | `PASS`, `FAIL`, or `SKIP` as below. |

Exit status: `0` only when `RESULT` is `PASS`; `1` when any counted check failed; `2` when the
preflight or a stale baseline skipped attribution; `3` when the bounded copy did not run.
`working-tree-guard` is not counted.

## Current receipt

`timeout 300 bash proof/controls/relocation.sh` standalone (no outer `flock`), exit status **0**
(no hang, no orphan, no lane-lock holder). The unmutated baseline is fresh, the flat-write mutation
is red, and the restored copy is fresh.

```text
relocation-control: copy                       PASS    bounded cp 120s, /tmp/opencode/apeiron-reloc-control.3801344 (11 MiB)
relocation-control: baseline                   PASS    regen.sh --check fresh on the unmutated copy
relocation-control: step1-freshness-mutation   PASS    rc=1, diff names the flat write
relocation-control: step3-restore-freshness    PASS    regen.sh --check green
relocation-control: working-tree-guard         PASS    proof/ApeironWord.lean digest unchanged
relocation-control: RESULT                     PASS    checks=2 passed=2 failed=0
```

After the run, `fuser -v /tmp/opencode/apeiron-lane.lock` is empty and no
`apeiron-reloc-control.*` copy remains. Machine `leviathan` x86_64, `nproc=32`; `/tmp` is a 32 GiB
tmpfs with ~9 GiB free. Recorded source identity: `src/table.rs` sha256
`a89e9ec743405ac7d80183e431f6e932ba90353bc777f142865f44005261d09a`; `proof/ApeironWord.lean` sha256
`95766c650fd94da6546d21ef6dff115db22d55cac90b1949f96352217055d1e5`. Tool identity (lane pins, see
`proof/README.md`): Charon `0.1.254` (`b104e24fea7d721b71e6c39fd70f26ff20bc0980`), Aeneas
`505b6ca-dirty`, Charon Rust `nightly-2026-08-18`, Lean `leanprover/lean4:v4.31.0`. A prior receipt
at `src/table.rs` `2a60a19d…` / `proof/ApeironWord.lean` `c8de66e3…` recorded the same default-path
result in 36 s; the digests moved with the tree, the default-path verdict did not.

The W6-T3 concrete certificate redesign in `proof/corpus/Growable/Resources.lean` (sha256
`563f8196…`) changes no relocation-control step, red target or recorded digest: it edits only
`Growable/Resources.lean` and `AxiomAudit.lean`, not `Growable/Relocation.lean`, so the
`writeFlat_frame_flat` red span and the `src/table.rs` / `proof/ApeironWord.lean` identities above
are unchanged. The frame step remains **NOT-RUN**.

### Step 1 evidence

The `--check` diff names the mutation (`- live := false` against the regenerated `+ live := true`),
so the flat relocation write is on the freshness-checked path.

### Semantic mutation (heavy), not run in this receipt

`RELOCATION_CONTROL_HEAVY=1` mutates `let i ← table.tag entry_address.place` to `let i := 0#u8`
and requires `lake build Growable.Relocation` to fail inside `Growable/Relocation.lean`; the prior
control revision observed the four spill headlines as `unsolved goals` and a restored build of
3343 jobs. That path was not exercised by this receipt.

### Frame mutation (heavy or frame), **NOT-RUN**

`RELOCATION_CONTROL_HEAVY=1` and `RELOCATION_CONTROL_FRAME=1` reproduce the manual index-zero
mutation that validated the W6-T1 disjoint-write frame headlines. In the copy's
`proof/corpus/Growable/Relocation.lean` the flat destination write

```lean
      { reserved.resident with slots := reserved.resident.slots.set index slot } }
```

is replaced by `.set 0#usize slot`. `writeFlat` is consumed only by its definition, the `rfl`
`writeFlat_frame_spill`, and the non-trivial `writeFlat_frame_flat`; the redirected index makes
`writeFlat_frame_flat` false for any `earlier = 0`, so its proof fails. The step then requires:

```sh
# red: error at a line inside [theorem writeFlat_frame_flat, theorem writeFlat_frame_spill)
(cd proof/corpus && lake build)          # T_LEAN=300, run under the §6 lane wrapper
# green: restore the destination index
(cd proof/corpus && lake build)          # T_LEAN=300, must exit 0
```

Expected receipt (this has **not** been run; the heavy/frame path was not exercised):

```text
relocation-control: step2c-frame-mutation      PASS    rc=1, error at Relocation.lean:<L> in writeFlat_frame_flat span
relocation-control: step2c-frame-restore       PASS    restored destination index builds green
```

The red line is attributed by line number, not by theorem name: Lean reports the failing tactic at a
line inside the theorem's declaration span, and the runner computes that span from the copy before
mutating (`FRAME_LO` = `theorem writeFlat_frame_flat` line, `FRAME_HI` = `theorem
writeFlat_frame_spill` line). A green build under the mutation, a timeout, or an error outside the
span is `FAIL`.

## Residual obligation

None for freshness: the current `proof/ApeironWord.lean` (sha256 `95766c65…`) is fresh for
`src/table.rs` (sha256 `a89e9ec7…`), so the baseline and restored-freshness steps are attributable
and green, and the control exits 0. The frame mutation is **NOT-RUN** in this receipt: its heavy/frame
path was not executed, so the frame control is a supplied receipt until `RELOCATION_CONTROL_FRAME=1`
(or heavy) is run and its red→green receipt is recorded here. The still-open semantic items are
outside this control's scope and owned by the proof lane: `GrowableRelocation.rebuild_preserves` over
the traversal and `GrowableRefinement.prepared_rebuild_refines_insertWith` (which additionally needs
`SearchDecisionCorrect` and `CallbackLaws`). Until those exist, build under the bounded Lean gate,
are registered in `AxiomAudit.lean` and fail their relocation/merge controls, the semantic
adjudication gate stays MISSING. This control rejects the flat-write freshness mutation, the
spill-write semantic mutation (heavy), and the flat destination index-zero mutation (heavy/frame);
it does not prove any headline.
