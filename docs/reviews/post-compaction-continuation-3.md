# Post-compaction continuation — programme dispatch

Repo `/home/socol/Workspace/apeiron`, branch `main`, **HEAD `a6faebb`** (plus the fleet-brief edit for
Sol's decisions). Read `AGENTS.md`, `../AGENTS.md` (owner layer: §Authority/hold/momentum,
§Escalation ladder), `docs/direction.md`, `docs/kernel-boundary.md`, `docs/plan.md`,
`docs/reviews/multi-repo-fleet-brief.md` (§9 programme phasing, §9.1–9.3 decisions and corrections).
The source and the gates are the record; a document is corrected to them, never the reverse.

## What this session is

Dispatch a large, cross-repo fleet. The programme frame and sizing are in
`docs/reviews/multi-repo-fleet-brief.md` §9; the apeiron lane detail is in `docs/plan.md` §Phasing.
Expected scale: **dozens of agents**, ~50 available, 10 lanes, at most **3 simultaneous heavy Lean
checks** (~44 GiB). One owner per file; one writer per shared build product; every wave's gate and its
control green before the next opens.

## Decisions already made — do not re-ask

1. **Song: dispatch the A1–A7/B6 inventories now**, alongside A0/K2 (`song/docs/ROADMAP.md` §2). An
   inventory names actual source paths, dependencies, missing gates and the next independent tasks.
2. **Penglai 7: one bounded spike** alongside milestones 5/6, with a precise question and stop
   condition (`penglai/docs/substrate-port-scope.md` §7). Closure stays separate from exploration.
3. **Webfold: current evidence first, the rest retained.** Reproducible counterexamples first;
   bounded reproduction tasks for the historical/unmeasured high-impact findings; deduplicate by
   mechanism and affected population before counting (`webfold/docs/REPAIR-LEDGER.md`).
4. **Per-epic dependency barriers, one global resource scheduler** (memory, cache ownership, file
   ownership, verification capacity). Independent epics in one repo run concurrently; cross-repo
   dependencies get explicit connecting barriers. No shared wave counter.

## Corrections to apply to the phasing

- Task totals are **estimates**: forecast from a drained wave's observed turnaround and acceptance
  rate, not from the 540 s ceiling. "One full pass" means **every obligation gets a disposition** —
  accepted, rejected, blocked or explicitly deferred.
- **E3 is substantial Lean work** (source-cost proofs, `RowDotAdmission`), not cargo/doc filler.
  Classify each task's real resource demand.
- **Do not delete `RegenRelocateAll` for sharing a body with `StoredRelocateAll`.** Compare carrier
  assumptions, domains, refusals and claimed resource properties; reuse a common extraction where
  right, retain distinct obligations where the contracts differ.
- Add **explicit disposition rows** for `sanguine`/`sanguine-apeiron`, `TorchLean` and Whispering
  Tides. A Song application profile is one interface slice, not the application programme.
- **Replace every "per-child gate" placeholder** with the executable registry: exact command, owned
  paths, rejecting control, budget, prerequisite identities. The tables are the map; that registry is
  the schedule.

## First batch priority

Rust gate/receipt machinery (`tools/gate`), extraction repair (`E4`), Intent's selection controls
(Ring 6), Song's verification integrity (A0/K2). Fill independent capacity with Penglai and Webfold
work. Integrate continuously and replenish from ready tasks. Progress is accepted, reusable results
and completed improvement cycles across the ecosystem.

## Mechanisms in force

- **Hold.** `~/Workspace/HOLD` is in force: no new document, dispatch, scope change or source edit;
  the permitted action is *integrating a returned task* (re-run its named gate, commit it). Run
  `gate hold` (installed at `~/.local/bin/gate`, built from `tools/gate`) before any write or
  dispatch. Only the owner removes the marker.
- **Escalation ladder** (`../AGENTS.md`): worker → lead → owner → Sol (owner-relayed; there is no
  OpenRouter credential in `~/.local/share/opencode/auth.json`). A worker stops and hands back on
  three materially different approaches or five patch-and-rebuild cycles on one proof, on budget
  expiry, on a contract/grade/trust-boundary blocker, or on a file owned by another task. An
  escalation states the exact failing goal with the error verbatim, the approaches tried, the
  smallest unresolved goal, what is on disk, and the one decision requested.
- **Gate driver.** `tools/gate/core.rs` is the pure admission core (manifest validation: unique ids,
  registered-requirement closure; then registered id, matching digest, resolvable references) with 7
  controls green and attributable mutations. `tools/gate/main.rs` is the adapter bin. Extending it —
  extract/refine `admit`, the `docs/artifacts.toml` manifest, the adapter, `gate docs check
  --controls`, mandatory-path wiring — is epic E1.
- **Lanes.** Compile/gate through the box pool (`lane`); heavy Lean ≤3 at once; receipts under
  `/tmp/opencode`, scratch under `/home/socol/oc-tmp`; prune a finished slice's copy at its receipt.

## State at compaction

- **RowDotAdmission**: finished green by its worker (348 s of 540; 48 headlines, closed cone;
  mutant red on `widened_product_fits`/`srcOps_of_cont`, restored green) and **being integrated** —
  its three files are in `proof/resource/`, the gate of record is running. Commit it when the gate
  returns exit 0.
- **Resume queue** (`docs/plan.md` §Resume queue): LmHead stopped on the cycle cap with a named
  blocker (`LmHeadResource.lean:557` bind zeta-reduction); `regen_relocate_entry` and
  `regen_relocate_all` returned without handbacks; `exact_dot` stopped by the hold with real scratch
  proofs and a bind-over-pattern-binder blocker; `matvec` and Row-dot are the last cost-model items.
- **Extraction root**: `proof/extract/lib.rs` is one error from compiling; the fix is a three-line
  `src/` move (`dequant_f32_row` into `src/engine/f32_convert.rs`). `proof/regen.sh --check` also
  needs the E0514 sysroot cleared and `ApeironWord.lean` regenerated.
- **Cleanup done**: `oc-tmp` 154.7G → 18G, `/tmp/opencode` 1.8G, `/home` free 569G; 31 stale
  worktrees captured to `/tmp/opencode/abandoned-worktrees` and removed; 33 merged branches deleted
  (`one-allocation` kept, unmerged).
- **Dirty**: `main` still carries ~95 paths (this session's plan/gate edits plus pre-existing
  generated `proof/` modifications that are not ours).

## Rules

Routine proof and Rust tasks continue without re-escalating to the owner. Every task ≤3 files, one
gate, one control, one Arc block; the lead re-runs each gate before commit; no `sorry`/`axiom`/`admit`
outside the stated cone; every local gate declares a numeric wall-clock budget. Do not start a
document that does not replace one. Report results and evidence, not narration.
