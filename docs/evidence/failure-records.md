# Failure records — standard and register

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

A failure is not a mark against the work. An **unrecorded** failure is, because the next lane repeats
it. This file is the standard and the register; Operately is the board that points at it.

## 1 · The standard

A **failure record** is created when an attempt, implemented as scoped, fails to materially improve the
objective — a **no-move**, a **regression**, or an **abandonment for a structural reason**. One record
per failure, in this register. A no-move is **not** a board row per experiment (a thousand rows pollute
the board; one register does not).

Every field is required. `n/a` is a value; `—` is not.

```text
[FAIL-<n>] <one line: what was attempted>
  ATTEMPTED:   what, as scoped — files, gate, the task as dispatched
  HYPOTHESIS:  the speed/correctness model it assumed, stated so it can be falsified
  MEASURED:    before -> after, spread, the command that ran (or "not measured: <why>")
  OUTCOME:     improved | no-move | regression | abandoned-before-measure
  WHY:         the actual mechanism it failed
  BASE:        the primitive it iterated from (the subsumption axis)
  MODEL-GAP:   which belief about where the speed lives was wrong
  DISPOSITION: parked | canceled | reverted | regression-recorded | retained
  EVIDENCE:    file:line, receipt path, digests
```

Rules:

- A **regression** is a distinct outcome and is recorded before the change is reverted or re-scoped.
- The `WHY` is a **proposition about the speed or correctness model**, not a narrative.
- A failure that was never measured is `abandoned-before-measure`, and its `WHY` names the structural
  reason it could not be measured — never a story about what it "would have" done.

## 2 · Register

### [FAIL-1] Item 2 — load-sized serial counting build (`APEIRON-BATCH-BUILD`, Sol item 2)

```text
ATTEMPTED:   a load-sized serial counting build inside the generic Table. examples/reg1 + src/table.rs.
HYPOTHESIS:  the serial counting pairs are slow because the build over-provisions; a load-sized build
             reduces provisioning and therefore the ratio.
MEASURED:    spatial pairs 3.35 -> 4.26 and 3.40 -> 4.42 (REGRESSED); hashtrinity_table flat.
OUTCOME:     regression
WHY:         it changed provisioning *inside* the generic Table; the spatial pair's charged cost is
             per-key work, not provisioning. The build is on a branch the ratio does not charge.
BASE:        the generic Table (`src/table.rs`).
MODEL-GAP:   "apeiron is slow because of provisioning/locator overhead." False: directory assembly is
             already excluded from selected timing (examples/reg1/adapters/spatial.rs:688-694).
DISPOSITION: reverted (f89b18b history); Sol item 2 closed experiment.
EVIDENCE:    docs/plan.md (historical tuning routes); board row APEIRON-BATCH-BUILD (Done).
```

### [FAIL-2] Item 3 — resident-vacancy / flat direct-address count (A-serial)

```text
ATTEMPTED:   a serial direct-address counting arm on a resident vacancy.
HYPOTHESIS:  a flat direct-address count (no probe, no locator) removes the counting overhead.
MEASURED:    4.929 -> 4.6539; delta 0.9647 vs tolerance 0.9000 -> WITHIN SPREAD.
OUTCOME:     no-move
WHY:         the flat direct-address count is a *new container*, not a patch to the existing one; the
             measured move is inside the unchanged-code spread, so it is not a change.
BASE:        the generic Table (patched).
MODEL-GAP:   "a small patch can recover the counting pairs." The pairs require a different container.
DISPOSITION: parked; A-serial stopped.
EVIDENCE:    escalation receipts (item-2 revert decision); docs/plan.md.
```

### [FAIL-3] `APEIRON-COUNT-W2-T1` — the from-scratch counting container

```text
ATTEMPTED:   a single-region open-addressed coordinate-keyed container (`src/table/count.rs`, 881
             lines) + codec `CountKeyCodec`/`U32CoordWord` + module declaration. Gate
             `cargo test --locked --lib table::count::tests::`.
HYPOTHESIS:  a purpose-built single-region container without the locator or the 256-region directory
             reaches the counting pairs' target.
MEASURED:    gate green (10 tests). Both rejecting mutations observed red then restored (coordinate
             identity; sentinel alias). NOT wired to any production path -> no ratio measured.
OUTCOME:     abandoned-before-measure
WHY:         (a) off the measured path: examples/reg1/adapters/spatial.rs:674-681 calls neither
             `Table::sum_parallel` nor `Table::get_each_at`, so editing either cannot move a
             registered ratio; (b) `admit_coord` (`src/table/count.rs:159-163`) rejects every value
             above `u32::MAX`, while the registered spatial population packs three 21-bit fields
             (`spatial.rs:522-529`, max `0x7ffffc001a1fffd9`) -> the container cannot admit the
             population at all.
BASE:        derived from the generic Table's cost model, not ported from the empirical winner.
MODEL-GAP:   "derive a better primitive and prove it." The empirical winner already exists and is
             pinned; deriving re-paid the inferior base's cost and chose a domain (`u32`) narrower
             than the registered population.
DISPOSITION: parked (retained, not waived); base corrected by APEIRON-SUBSUME-HT.
EVIDENCE:    /home/socol/oc-tmp/count-w2t1.diff; /home/socol/oc-tmp/escalation/{VERIFY-CONTRACT,
             APEIRON-PARITY-PATH2}.md.
```

### [FAIL-4] `W2-T1b` and its `DOMAIN` / `FULLWIDTH` splits

```text
ATTEMPTED:   wire the container's counting build into `src/table/parallel.rs` and
             `examples/reg1/adapters/spatial.rs:674-681`; then the domain-compatibility gate and the
             full-width representation.
HYPOTHESIS:  integration of the container into the timed path moves `hashtrinity_count_par` and
             `hashtrinity_get_each`.
MEASURED:    not measured: killed before any run (owner direction: the old approach).
OUTCOME:     abandoned-before-measure
WHY:         the base container cannot admit the registered population (FAIL-3b); integration on a
             non-admitting container cannot produce a green claim, and the domain repair is work on a
             container the subsumption replaces.
BASE:        the derived container.
MODEL-GAP:   "integrate the new container." The subsumption makes the container unnecessary.
DISPOSITION: canceled; rows parked.
EVIDENCE:    /home/socol/oc-tmp/escalation/APEIRON-PARITY-PATH2.md.verdict.md.
```

### [FAIL-5] `RESIDENT-GROWTH` · `SPILL-PROVISION` · `COUNT-READBACK`

```text
ATTEMPTED:   transactional resident-extent growth (the fresh-insert cell, rmax 5.1144); load-sized
             shard/slot provisioning (the spatial/n1000 cell, 66% grow_spill); a workload-preserving
             readback change for `get_each`.
HYPOTHESIS:  the worst registered pair's cost is the resident rebuild / the over-provision / the
             readback, so removing each moves its pair.
MEASURED:    not measured: killed before any run (owner direction: the old approach).
OUTCOME:     abandoned-before-measure
WHY:         all three iterate from the generic Table — the primitive the subsumption replaces. The
             fresh-insert rebuild and the over-provision are *symptoms* of the generic Table's
             representation, not independent levers.
BASE:        the generic Table.
MODEL-GAP:   "optimize the generic Table's expensive paths." If the timed path carries the subsumed
             structure, those paths do not exist to optimize.
DISPOSITION: canceled; rows parked.
EVIDENCE:    /home/socol/oc-tmp/escalation/APEIRON-PARITY-PATH2.md.verdict.md (RESIDENT-GROWTH was
             ranked #1 on the generic-Table base).
```

### [FAIL-6] The hygiene batch — eight lanes, all green, zero contribution

```text
ATTEMPTED:   CITATION-CONSTRUCT, SUPERSESSION-GATE, ROOTING-D2, MEASURE-GUARD, BOARD-LENS-GATE,
             OPERATELY-SDK-GAP, ONTO-REPAIR, BEDROCK-FIX.
HYPOTHESIS:  n/a — none claimed a speed or correctness improvement to the measured pairs.
MEASURED:    each gate passed; none touched the measured path. CITATION-CONSTRUCT would have reddened
             the live tree (+30 in the epic); the rest were held uninstalled.
OUTCOME:     no-move (by construction)
WHY:         AGENTS.md §3a avoidance: easy, safe, locally-green work while a quantitative gate is red.
BASE:        n/a.
MODEL-GAP:   "green local gates are progress." They are not, when the objective is unmoved.
DISPOSITION: parked (uninstalled); the rule that forbids them is now the §3a subsumption insertion.
EVIDENCE:    the lane returns; META-ANALYSIS.md §2(e).
```

### [FAIL-7] The dispatch regime itself — the meta-failure

```text
ATTEMPTED:   ~30 lanes, ~6 hours, dispatched by the standing loop (hardest-first, one gate + one
             control, board checkpointed each cycle).
HYPOTHESIS:  that the loop reaches the objective because every lane is correct and well-gated.
MEASURED:    hashtrinity_table 1.7504 -> 1.7541; count_par 3.4508 -> 3.2333; get_each 3.4003 ->
             3.2889 — every delta inside its own spread. full_acceptance=false throughout.
OUTCOME:     no-move
WHY:         the loop has no counterfactual and no objective predicate. It grades the artifact, never
             the objective; "hardest first" became "deepest new primitive first"; the evidence regime
             rewards reproducing the number over changing it; §3a named the failure with no force
             behind it; §9.3's port rule was unenforced.
BASE:        the whole queue was on the generic Table's branch.
MODEL-GAP:   "a correct lane is progress." A correct, off-path lane is a *negative* result.
DISPOSITION: retained; the fixes are META-ANALYSIS.md §4 and the AGENTS.md §3a insertion.
EVIDENCE:    /home/socol/oc-tmp/escalation/META-ANALYSIS.md; the five-run campaign logs
             /tmp/opencode/ckpt{2,3}.log.
```

### [FAIL-8] The directory-differential hypothesis — "drop the locator registry"

```text
ATTEMPTED:   name the largest removable differential as the excluded per-key locator registry
             (`src/table/locator.rs:961-984` reached at `src/table/parallel.rs:658`), on the claim
             that it was a large part of the ratio.
HYPOTHESIS:  a cost that dominates *absolute* elapsed time dominates the *ratio*.
MEASURED:    the 44e6/312e6/268e6 triple is `spatial/n1000000`, whose CHARGED ratio is 2.398 (vs
             `hashtrinity_count_par`), not the 1.7728 serial mean it was quoted against. Removing the
             registry leaves 43,103,531 / 17,972,019 = 2.398 UNCHANGED; it changes absolute build wall
             clock 310e6 -> ~43e6 ns (7.19x). The harness already subtracts the bucket
             (`examples/reg1/adapters/spatial.rs:688`).
OUTCOME:     no-move (ratio), despite a 7.19x absolute effect.
WHY:         the registered ratio is charged-time; the harness excludes `assemble_directory` /
             `finalize_recorded_locations` from the charged region by construction. An excluded cost
             cannot move a charged ratio.
BASE:        n/a — a model error, not a base error.
MODEL-GAP:   "a big absolute cost is a big ratio cost." False whenever the harness excludes it. The
             charged region's cost is the per-key path; the excluded bucket is definitionally outside
             the comparison.
DISPOSITION: the claim is retracted; the removal is still worth doing for absolute time and for
             removing the campaign's only structural asymmetry, but it is NOT a ratio lever.
EVIDENCE:    /home/socol/oc-tmp/escalation/DIFF-HT-AP.md; /tmp/opencode/ckpt3-run5.out pair table.
```

### [FAIL-9] `PROF-BREAKDOWN` — the per-function breakdown lane returned nothing

```text
ATTEMPTED:   a per-function/per-phase `perf` breakdown of the worst hashtrinity pair.
HYPOTHESIS:  the coarse phase percentages can be replaced by a per-function attribution.
MEASURED:    not measured: the lane ended with no return text. Its worktree
             `/home/socol/oc-tmp/prof-breakdown` holds a 420-line diff (spatial.rs, main.rs,
             strata.rs) + `tools/prof/`, with no gate output, no command, no numbers.
OUTCOME:     abandoned-before-measure
WHY:         the lane produced code but no return; there is no evidence the diff runs, and it touches
             files a concurrent lane owns (`spatial.rs`), so it cannot be installed as-is.
BASE:        n/a.
MODEL-GAP:   "a running lane produces a result." A lane's artifact without its gate output and numbers
             is not a result — the return is the deliverable, not the diff.
DISPOSITION: held uninstalled; worktree retained for reuse or removal. `perf 7.2.8` is available on the
             box, so the instrument was not the blocker.
EVIDENCE:    /home/socol/oc-tmp/prof-breakdown (uncommitted); no receipt.
```

## 3 · The model of speed, corrected

Every failure above shares one wrong belief: **that the speed lives in the generic Table's overhead —
its provisioning, its locator, its directory, its construction.** The evidence says otherwise.

- **The charged region excludes the directory.** `examples/reg1/adapters/spatial.rs:688-694`: the
  excluded component is `assemble_directory` / `finalize_recorded_locations`. The charged cost is the
  per-key path: qualify, partition, schedule, reserve, kernel, probe, query-order fold.
- **The incumbent's speed is the speed of doing less.** Its counting path is keyless (`FreqLine<V>` has
  no key column, `freq.rs:275`), its identity is a 64-bit fold (`algebra.rs:358-363`), it carries no
  checked arithmetic, and its emitted plain loop is **19 insns/key, 3 loads, 0 stores**
  (`ASM-HT.md`). Apeiron's corresponding path does the identity, the checked arithmetic and the walk's
  stores (0 vs 16 in the decision walk).
- **Therefore the lever is the per-key representation**, not the container's construction: how many
  loads per key, whether the identity is carried in the word or re-derived, whether the hot walk
  stores, and whether the merge is a bare ALU op over a disjoint partition.

**The proposition to formalize in apeiron:** *the ratio on any apeiron arm is dominated by its per-key
instruction count in the charged region; any change that does not reduce per-key instructions
(identity addressing, slot loads, checked arithmetic, walk stores) cannot move the ratio, however much
it improves construction, provisioning or cleanup.* The subsumption (port the winner) is the operation
that changes the per-key representation; everything else is a tax on the inferior base.
