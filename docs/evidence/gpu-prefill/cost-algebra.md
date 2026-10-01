# Cost algebra: composition, dominance and one nonvacuous sharing choice

Grade `proved` for the algebra, the dominance order and both witnesses. Grade `model` for the
per-stage counts, which are this module's declared example inputs. Closes Plan row 3 of
[`docs/epics/resource-guided-evolution.md`](../../epics/resource-guided-evolution.md) only in its
named form: a composition/dominance algebra with a nonvacuous witness, a duplicated-effect and an
omitted-packing defect rejected, and a non-strict improvement rejected.

Artifacts: `proof/resource/CostAlgebra.lean`, `proof/resource/CostAlgebraGate.py`, this receipt.
No other file is read for the claim except `proof/DispatchAdmission.lean`.

## Resource semantics

Seven fields, one resource each, in that field's own unit. The list is the epic's Specification
row "Resource semantics"; a field never mixes two resources, so a regression in one is visible
instead of traded away.

| field | unit | counts |
|---|---|---|
| `ops` | operations | arithmetic, one multiply-accumulate or scale unit per operation |
| `pack` | bytes | packing and copy between layouts |
| `alloc` | allocations | buffer allocations |
| `xfer` | bytes | uploads and readbacks |
| `dispatch` | dispatches | kernel dispatches and queue drains |
| `peak` | bytes | peak-live storage held at one instant |
| `path` | stage units | dependency critical path, the longest dependent chain |

## Composition

| operator | domain | law |
|---|---|---|
| `+` (`costAdd`) | dependent stages | every registered resource accumulates: the schedule-independent sum the epic takes for disjoint dependent work |
| `overlap` | independent, concurrent stages | additive resources still total; `path` is `max` of the two branches; `peak` adds under the premise that both branches' buffers are live at the same instant |
| `seq` | `List Cost` | `foldr (· + ·) zero` |
| `layerCost` | `Layer` | `foldr (fun b acc => overlap b.cost acc) zero` |
| `planCost` | `Plan` | `foldr (· + ·) zero` over `layers.map layerCost`, then the shared effect added once |

`Scheduled p := (p.layers.flatMap id).map Prod.fst = p.stages` binds one registered cost to each
stage of the computation, each exactly once and in order, so the composition can neither copy nor
eliminate stateful work. `observe p := p.stages` is the observation two schedules must agree on.

## Laws (all kernel-checked, all audited)

| claim | theorem |
|---|---|
| `+` associative, so a chain may be grouped either way | `add_assoc` |
| `+` commutative | `add_comm` |
| the `+` fold is invariant under a permutation of its arguments: disjoint work sums the same in any order | `seq_perm` |
| the `+` fold splits at an append | `seq_append` |
| `+` monotone in each operand | `add_monotone`, `add_monotone_left` |
| sequence composition monotone in every stage | `seq_monotone` |
| layer composition monotone in every stage | `layerCost_monotone` |
| `overlap` associative | `overlap_assoc` |
| `overlap` changes no additive resource | `overlap_fields` |
| the critical path of two concurrent branches is the `max` | `overlap_path` |
| the critical path of a layer is the `max` over its stages | `layer_critical_path` |
| two branches with positive paths overlap to a strictly shorter chain | `overlap_path_lt_add` |
| concurrency never exceeds sequence | `overlap_le_add` |
| concurrency monotone in each branch | `overlap_monotone`, `overlap_monotone_right` |
| plan cost monotone in the shared effect | `planCost_mono_in_shared` |
| the shared effect enters the composition exactly once | `planCost_shared_add` |

## Dominance

`a ≤ b` is fieldwise: a candidate dominates the baseline when no registered resource grows.
`strictlyCheaper a b := a ≤ b ∧ a ≠ b` is the registered improvement.

| claim | theorem |
|---|---|
| dominance is a partial order | `Std.IsPartialOrder Cost`, `Std.LawfulOrderLT Cost` |
| a tie is not an improvement | `tie_not_improvement` |
| growth in any registered resource is rejected, whatever is saved elsewhere | `regression_not_improvement` |
| dominance yields the strict `Nat` reduction the certificate demands | `dominates_improves` |
| the weighted scalar is not a substitute for the order | `scalar_not_sufficient` |

`registeredCost c = 8*ops + 64*pack + 16*alloc + 64*xfer + 4*dispatch + 64*peak + path` is the
`Nat` cost, all weights positive. `dominates_improves` proves `strictlyCheaper a b →
registeredCost a < registeredCost b`, which is the comparison
`proof/DispatchAdmission.lean:15` states as
`improves : ∀ x, reachable x → admissible x → afterCost x < beforeCost x`.
`scalar_not_sufficient` exhibits `⟨0,0,0,0,0,0,4⟩ <registeredCost ⟨1,0,0,0,0,0,0⟩` while
`¬ a ≤ b`: selecting on the scalar alone would accept a candidate that lengthens the critical
path, which is why the registered order is componentwise.

The link to the certificate is an interface binding, not an elaboration: the gate asserts that the
quoted `improves` line is present in `proof/DispatchAdmission.lean` and records its digest.

## Nonvacuous witness

Two candidates of one computation, equal observations (`equal_outputs : observe residentPlan =
observe perProjectionPlan`, by `rfl`), the mechanism being one resident activation pack shared
across three projections against one pack per projection.

| plan | ops | pack | alloc | xfer | dispatch | peak | path | `registeredCost` |
|---|---|---|---|---|---|---|---|---|
| `perProjectionPlan` (baseline) | 36 | 9216 | 3 | 9216 | 3 | 9216 | 12 | 1769832 |
| `residentPlan` (candidate) | 36 | 3072 | 1 | 3072 | 3 | 3072 | 4 | 590144 |

`perProjection_cost`, `resident_cost` pin both records; `resident_dominates` is the strict
dominance; `resident_improves` is the strict `Nat` reduction; `packing_reduction` isolates the
packing bytes; the four `*_scheduled` theorems check the one-cost-per-stage binding. All five
plans' `Scheduled` obligations hold by `rfl`.

Second witness, isolating the `max` law with packing excluded: `sequentialPlan` (three dependent
layers) composes to `⟨36,0,0,0,3,0,9⟩` and `forkedPlan` (one concurrent layer) to `⟨36,0,0,0,3,0,3⟩`
with `observations_equal` and `forked_critical_path_reduced`. Only the critical path differs, which
is the whole content of `overlap_path` and `overlap_le_add`.

The counts are declared inputs of this module, not a measurement. The census and its retained
receipts are the source of the mechanism claim; nothing here measures a kernel.

## Gate

```sh
cd /home/socol/Workspace/apeiron
python3 proof/resource/CostAlgebraGate.py --check --controls
```

Exit status 0. The gate re-execs itself under `timeout 540s flock -w 530
/tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0
timeout 525s env LEAN_NUM_THREADS=1`; the whole invocation, lane wait included, took 10.2 s
against the 540 s ceiling. A lane held past that budget yields exit 124 with no build run (LANE_BUSY). The
module is copied into a private Lake package under `/home/socol/oc-tmp/cost-algebra` that resolves
`leanprover/lean4:v4.31.0` from `target/aeneas-model/lean`; only the copy is ever mutated. Policy
preflight rejects `sorry`, `axiom`, `admit` and `set_option maxHeartbeats 0` in the module.

| stage | exit | seconds | result |
|---|---|---|---|
| `lake --version` | 0 | 0.03 | — |
| `fresh-proof` | 0 | 2.27 | 44 `#resource_audit` lines, all axiom sets inside `{propext, Classical.choice, Quot.sound}`, `COST_ALGEBRA_KERNEL_OK` |
| `mutant-duplicated-effect` | 1 | 1.37 | `COST_ALGEBRA_AXIOM_REJECT CostAlgebra.planCost_shared_add`, `... CostAlgebra.resident_cost` |
| `restored-duplicated-effect` | 0 | 1.36 | `COST_ALGEBRA_KERNEL_OK` |
| `mutant-omitted-packing` | 1 | 1.27 | `COST_ALGEBRA_AXIOM_REJECT CostAlgebra.packing_reduction`, `... CostAlgebra.perProjection_cost`, `... CostAlgebra.resident_cost` |
| `restored-omitted-packing` | 0 | 1.37 | `COST_ALGEBRA_KERNEL_OK` |
| `mutant-non-strict-improvement` | 1 | 1.17 | `COST_ALGEBRA_AXIOM_REJECT CostAlgebra.tie_not_improvement`, `... CostAlgebra.resident_dominates` |
| `restored-non-strict-improvement` | 0 | 1.37 | `COST_ALGEBRA_KERNEL_OK` |

Every `theorem` in the module appears in the audit list, and the gate fails unless each requested
`#resource_audit` name produces its own `COST_ALGEBRA_AUDIT` line and no `COST_ALGEBRA_AXIOM_REJECT`
appears. The build exit status is checked as well as the marker, because `run_cmd` prints the
marker even when earlier commands in the file failed.

Each control is a single-needle text defect in the throwaway copy, and each must name the headline
it broke, not merely exit non-zero:

| control | seeded defect | required red headline |
|---|---|---|
| duplicated-effect | `planCost` adds `p.shared + p.shared` | `planCost_shared_add`, `resident_cost` |
| omitted-packing | `costAdd` sets the `pack` field to `0` | `packing_reduction`, `perProjection_cost`, `resident_cost` |
| non-strict-improvement | `strictlyCheaper` drops the `a ≠ b` component | `tie_not_improvement`, `resident_dominates` |

Receipt `/tmp/opencode/cost-algebra/gate-l8b2rti2-receipt.json`, with every stage log beside it,
records the table above at exit 0 in 10.20 s. An earlier run of the same gate on the 40-headline
predecessor of the module (`gate-y9b75jnj-receipt.json`) exited 0 in 9.85 s with the same three
red/restored verdicts; `gate-htrs3ch_` and `gate-_tb6wzf6-` are the two earlier runs of the same
gate, the first aborting on a policy false positive and the second on a control needle that was not
unique. Source identities of the final module:

| sha256 (first 16) | path |
|---|---|
| `c362ec455954e512` | `proof/resource/CostAlgebra.lean` |
| `75d8ed218d752250` | `proof/resource/CostAlgebraGate.py` |
| `7f2ca1e195e56c00` | `proof/DispatchAdmission.lean` |
| `efac0b94923b2d8b` | `target/aeneas-model/lean/lean-toolchain` (`leanprover/lean4:v4.31.0`) |
| `9ccd1a96f2830caa` | `proof/dispatch/lake-manifest.json` |

The scalar values 1769832 and 590144 were read from a throwaway package that imported the module
and evaluated `registeredCost`; they are not asserted by any theorem.

## Boundary: not claimed

- No hardware speed, latency, bandwidth, device dispatch or occupancy measurement. Nothing here
  bounds the latency of any implementation, and no removed work is shown to be removed time.
- No compiler, codegen, register or occupancy cost. `ops` counts declared arithmetic units, not
  instructions.
- No cost record for any specific kernel. `dotCost`, `packDotCost` and `residentPack` are example
  inputs; a real candidate must supply counts derived from its own source loop.
- No claim about any Rust source, extraction or refinement. The module reads no `src/`, and its
  costs are not connected to `ActScaleResource` or to the census artifacts.
- The certificate link is a text-and-digest binding of `proof/DispatchAdmission.lean`, not a Lean
  elaboration of `ProofDispatch.Admission`; the universal preservation and reachability premises of
  that structure are supplied by the production contract, not here.
- The graph model is two-level: a list of dependency layers, each a list of concurrent stages. No
  partial-order dependency between arbitrary stage pairs is represented.
- `peak` adds under the premise that concurrent branches hold their buffers simultaneously; a
  schedule with exclusive buffer lifetimes is not represented and would need a different
  composition for that field.
- `residentPack.dispatch = 0` is a premise of the example (the pack rides in the consuming
  launch), not a measured dispatch count.
- This slice runs its own named gate only. The `AGENTS.md` §6 epic barrier (`cargo test`,
  `proof/regen.sh --check`, `lake build` in `proof/` and `proof/corpus/`) and the ActScale
  prerequisite were not re-run here and are unaffected by these three new files.
