# Cost bridge: source-derived counts into `CostAlgebra.Cost`

Grade `proved` for the map, its monotonicity, the composition law and the two schedules, under the
closed logical cone `{propext, Classical.choice, Quot.sound}` checked by `#resource_audit`. Grade
`model` for the element-read count `2n` and for the critical-path convention `path := visits`. No
hardware speed follows from any of it.

This closes the connecting row the [cost algebra](cost-algebra.md) receipt leaves open: that module
declared its per-stage counts as example inputs and said a real candidate must supply counts derived
from its own source loop. This module supplies them, from `proof/resource/RowDotResource.lean`'s
closed form for the actual extracted `matvec::row_dot_fixed` body, and carries them through
`CostAlgebra.costAdd` into the componentwise order.

Artifacts: `proof/resource/CostBridge.lean`, `proof/resource/CostBridgeGate.py`, this receipt. No
other file is edited.

## The count record

`Perf` carries six counts, each in its own unit, none traded against another:

| field | unit | source |
|---|---|---|
| `visits` | invocations of the body | `RowDotResource.entry_visits_exact`: `n + 1` for a width-`n` row |
| `mulAdds` | checked multiply-adds | the same theorem: `n`, and every certificate of it agrees |
| `loads` | element reads | `RowDotResource.body_cont`, one `w_row[i]` and one `x[i]` per visit |
| `stores` | element writes | 0: the body carries its accumulator, index and flag as values |
| `bytes` | bytes copied between layouts | 0: `src/engine/matvec.rs` takes `&[i64]` by reference |
| `allocs` | buffer allocations | 0, same declaration |

`rowDotPerf w _x = ⟨n + 1, n, 2n, 0, 0, 0⟩`. The link to the extracted body is a theorem, not a
citation:

| claim | theorem |
|---|---|
| the map reads the body's own certificate: any `RowTrace` of the entry state reads as `rowDotPerf` | `tracePerf_rowDot` |
| a certificate of exactly those counts exists | `rowDotPerf_trace` |

`tracePerf_rowDot` quantifies over *every* certificate `result visits steps` of the entry state and
concludes `tracePerf visits steps = rowDotPerf w x`, from the uniqueness half of
`entry_visits_exact`. It cannot hold for one derivation's counts alone.

## The map

`toCost : Perf → CostAlgebra.Cost` assigns each count to a registered field. `xfer`, `dispatch` and
`peak` are set to `0`, each for a stated reason in the source: a scalar source theorem transfers
nothing to a device, dispatches no kernel, and a source-level count carries no buffer width, so no
peak-live byte count can be asserted from it.

| `Cost` field | from | |
|---|---|---|
| `ops` | `mulAdds` | one unit per checked multiply-add |
| `pack` | `bytes` | one unit per copied byte |
| `alloc` | `allocs` | one unit per allocation |
| `xfer` | `0` | no device premise exists here |
| `dispatch` | `0` | no kernel is dispatched |
| `peak` | `0` | no buffer width premise exists here |
| `path` | `visits` | the primitive's own dependent step chain, `n + 1`, so an upper bound on the `n` dependent accumulate steps |

## Laws (all kernel-checked, all audited)

Monotonicity is stated as "raise one count, every other field held fixed", which is the order's
statement about that count alone.

| claim | theorem |
|---|---|
| monotone in the visit count | `toCost_monotone_visits` |
| monotone in the checked multiply-add count | `toCost_monotone_mulAdds` |
| monotone in the copied-byte count | `toCost_monotone_bytes` |
| monotone in the allocation count | `toCost_monotone_allocs` |
| monotone in the element-read count | `toCost_monotone_loads` |
| monotone in the element-write count | `toCost_monotone_stores` |
| monotone in all six at once | `toCost_monotone` |
| the registered order cannot see element traffic: `k` extra reads and `k` extra writes tie | `loads_invisible_to_order` |
| doing nothing maps to the cost of doing nothing | `toCost_zero` |
| the map preserves dependent composition: `toCost (a + b) = toCost a + toCost b` | `toCost_add`, `toCost_add_eq` |
| composition of count records is monotone in each record | `Perf.add_monotone` |
| a schedule's composed cost is the map of its composed counts | `runCost_eq_toCost` |
| composing two schedules composes their costs | `runCost_append` |
| the composition law read at schedule granularity | `runCost_append_toCost_add` |
| the cost of a run of one invocation | `runCost_singleton` |
| the cost of a run of two invocations | `runCost_pair` |
| a schedule's cost is monotone in every invocation | `runCost_mono` |

`loads_invisible_to_order` is the honest boundary of the registered vocabulary: `CostAlgebra.Cost`
has no field for host element traffic, so `toCost {p with loads := p.loads + k, stores :=
p.stores + k} = toCost p` and the pair is *not* `strictlyCheaper` in either direction. `loads` and
`stores` are therefore carried in `Perf` and unmapped, and the two monotonicity statements about
them hold vacuously. Closing this would need a new field in `CostAlgebra.Cost`, which is that
module's contract and not this one's.

## Nonvacuous witness

One computation: the dot product of the weight row `[1, 1]` against the activation `[1, 1]` in
Q32.32, as the actual `matvec::row_dot_fixed` computes it. Two schedules, differing only in whether
the rows are padded to the tile width four with zeros.

| schedule | visits | mulAdds | loads | ops | pack | alloc | xfer | dispatch | peak | path | `registeredCost` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `exactSchedule` (candidate) | 3 | 2 | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 3 | 19 |
| `paddedSchedule` (baseline) | 5 | 4 | 8 | 4 | 0 | 0 | 0 | 0 | 0 | 5 | 37 |

Equal observations, proved at the extracted entry point rather than at a schedule identifier:
`observations_equal : run exactSchedule = run paddedSchedule`, from `exact_run` and `padded_success`,
both of which state `row_dot_source.matvec.row_dot_fixed … = ok (core.result.Result.Ok 0#i64)`.

| claim | theorem |
|---|---|
| the padded rows satisfy the row-dot window | `pad_fit` |
| the padded loop is reachable and its closed form is 5 visits and 4 checked multiply-adds | `pad_visits` |
| the padded entry point returns the same value as the exact one | `padded_success` |
| the exact run returns that value | `exact_run` |
| the two schedules compute the same thing | `observations_equal` |
| the exact cost is `⟨2,0,0,0,0,0,3⟩` | `exact_run_cost` |
| the padded cost is `⟨4,0,0,0,0,0,5⟩` | `padded_run_cost` |
| no registered resource grows and two shrink | `exact_dominates_padded` |
| dominance yields the strict `Nat` reduction the certificate demands | `exact_improves_padded` |
| the padding costs two more multiply-adds | `padding_costs_two_more_mulAdds` |
| the padding costs two more dependent steps | `padding_costs_two_more_visits` |

`padded_success` does not recompute the epilogue. `pad_visits` and `RowDotResource.wit_visits` put
both loops at the same accumulator value `2`; `IScalar.eq_of_val_eq` makes the accumulators equal,
and `RowDotResource.row_dot_fixed_unfold` then shows the epilogues and the returned values agree.
The padding adds the products `0 · 0`, so the whole difference is two wasted dependent steps and two
wasted checked multiply-adds.

Composed to two dependent invocations, the same comparison holds and exercises the composition law:

| schedule | ops | path | `registeredCost` |
|---|---|---|---|
| `exactPair` | 4 | 6 | 38 |
| `paddedPair` | 8 | 10 | 74 |

`composed_observations_equal`, `composed_exact_cost`, `composed_padded_cost`,
`composed_exact_dominates_padded`, `composed_exact_improves_padded`, and `composed_is_sum_of_singles`
(which is `runCost_append` read as "the two-stage cost is the registered sum of the one-stage
costs"). The four `registeredCost` figures and the two `rowDotPerf` records were read off a
throwaway `Eval.lean` in a scratch package by `decide`; they are not asserted by any theorem here,
which asserts the inequalities.

## Gate

```sh
cd /home/socol/Workspace/apeiron
python3 proof/resource/CostBridgeGate.py --check --controls
```

Exit status 0. The gate re-execs itself under `timeout 540s flock -w 530
/tmp/opencode/apeiron-lane-3.lock systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0
timeout 525s env LEAN_NUM_THREADS=1`; the whole invocation, lane wait included, took 13.7 s against
the 540 s ceiling. A lane held past the 530 s wait yields exit 124 with no build run (LANE_BUSY).

The module is copied into a private Lake package under `/home/socol/oc-tmp/cost-bridge` that resolves
`leanprover/lean4:v4.31.0` from `target/aeneas-model/lean`; only the bridge copy is ever mutated.
Policy preflight rejects `sorry`, `axiom`, `admit`, `set_option maxHeartbeats 0`, and any
`maxRecDepth`/`maxHeartbeats` option in the bridge module. It also pins the module's two imports,
the single `toCost` needle, and the count of the kernel marker.

**The two-lane rename.** `CostAlgebra` and `RowDotResource` each register an elaborator named
`#resource_audit`, so one file cannot import both. The gate builds verbatim copies of the two with
that single identifier renamed (`#cost_algebra_audit`, `#row_dot_resource_audit`) and refuses to run
unless each copy is byte-equal to the committed file modulo exactly that rename, checked by a
round-trip substitution. No theorem statement or proof changes; both modules' own audits still run,
which is why the fresh build also requires `COST_ALGEBRA_KERNEL_OK` and `ROW_DOT_KERNEL_OK`. This is
a gate-level accommodation for an elaborator name collision, not a relaxation.

| stage | exit | seconds | result |
|---|---|---|---|
| `lake --version` | 0 | 0.03 | — |
| `fresh-proof` | 0 | 6.37 | 39 `#resource_audit` lines, all axiom sets inside `{propext, Classical.choice, Quot.sound}`, `COST_ALGEBRA_KERNEL_OK`, `ROW_DOT_KERNEL_OK`, `COST_BRIDGE_KERNEL_OK`, no `error:` |
| `mutant-dropped-multiply-add` | 1 | 1.77 | `COST_BRIDGE_AXIOM_REJECT` naming `exact_run_cost`, `padded_run_cost`, `composed_exact_cost`, `composed_padded_cost` |
| `restored-dropped-multiply-add` | 0 | 2.02 | `COST_BRIDGE_KERNEL_OK` |
| `mutant-non-monotone-critical-path` | 1 | 1.67 | `COST_BRIDGE_AXIOM_REJECT` naming `toCost_monotone_visits`, `exact_run_cost` |
| `restored-non-monotone-critical-path` | 0 | 1.87 | `COST_BRIDGE_KERNEL_OK` |

The build exit status is checked as well as the marker, and the green path additionally refuses any
output containing `error:`. That is what makes a rejection observable as red: a bare `throwError` in
a `Lean.Elab.Command` is reported and the command sequence continues, so `COST_BRIDGE_KERNEL_OK` is
a completion marker, not a cleanliness claim.

Each control is a single-needle text defect in the throwaway copy, applied to the one line of
`toCost`. Each must name every headline it broke and place a diagnostic inside that headline's own
declaration, so a control that fails elsewhere is not evidence about this module:

| control | seeded defect | required red headline | diagnostic line |
|---|---|---|---|
| dropped-multiply-add | `toCost` sets `ops` to `0` | `exact_run_cost`, `padded_run_cost`, `composed_exact_cost`, `composed_padded_cost` | 443, 451, 508, 516 |
| non-monotone-critical-path | `toCost` sets `path` to `p.visits - p.mulAdds` | `toCost_monotone_visits`, `exact_run_cost` | 113, 443 |

The two defects are complementary. Dropping `mulAdds` keeps the map a homomorphism but destroys the
information, so the pinned cost records break and no composition theorem does; making `path`
non-monotone keeps every pinned record reachable but destroys the order, so the monotonicity
headline breaks at its own proof. Neither defect is caught by the other.

Receipt `/tmp/opencode/cost-bridge/gate-qgu6xx0z-receipt.json`, with every stage log beside it,
records the table above at exit 0 in 13.74 s. Source identities:

| sha256 (first 16) | path |
|---|---|
| `031b4aab813c5074` | `proof/resource/CostBridge.lean` |
| `2de71de86a39bad7` | `proof/resource/CostBridgeGate.py` |
| `c362ec455954e512` | `proof/resource/CostAlgebra.lean` |
| `7c561036745c40eb` | `proof/resource/RowDotResource.lean` |
| `3dabece20cb5c7e3` | `proof/resource/RowDotSource.lean` |
| `efac0b94923b2d8b` | `target/aeneas-model/lean/lean-toolchain` (`leanprover/lean4:v4.31.0`) |
| `9ccd1a96f2830caa` | `proof/dispatch/lake-manifest.json` |

Evidence is bound to these digests, not to `HEAD`. The working tree carried concurrent edits from
other slices at the time of the run, including `src/engine/mixer.rs` and `src/engine/step.rs`; none
is an input to this gate.

## Boundary: not claimed

- No hardware speed, latency, bandwidth, device dispatch, occupancy or cache measurement. The
  counts are source-level invocations and checked arithmetic on the extracted body; nothing here
  bounds the time of any implementation, and no removed work is shown to be removed time.
- `ops` is a lower bound on the arithmetic of one invocation. The epilogue of
  `row_dot_fixed` (`round_shift_32`, two range comparisons, one `i64` narrowing) is not counted.
- Grade `model`: the element-read count `2n` is read off `body_cont`'s two indexed reads per visit,
  not proved against a memory model, and `stores = 0` is a property of the borrowed-slice
  declaration plus the fact that the loop carries its state as values. Neither is a hardware
  measurement.
- Grade `model`: the convention `path := visits`. The critical path of one primitive is taken to be
  its own dependent step chain, which for the row loop is `n + 1` including the terminating guard
  visit, so `path` over-reports by one relative to the `n` dependent accumulate steps. A different
  convention changes the `path` figures, not the `ops` figures or the strict dominance.
- The registered order cannot see host element traffic (`loads_invisible_to_order`), and cannot see
  `xfer`, `dispatch` or `peak` for a scalar primitive. A candidate that trades device traffic or peak
  storage against arithmetic is invisible to this comparison.
- Freshness of the extraction is `proof/resource/RowDotGate.py`'s obligation. This gate consumes the
  committed `RowDotSource.lean` as given and records its digest; it does not re-run Aeneas/Charon.
- `path` and `ops` agree for this primitive family because `visits = mulAdds + 1`. For a primitive
  whose internal chain length and arithmetic count differ, the two registered fields separate and
  the map is untested on it.
- This slice runs its own named gate only. The `AGENTS.md` §6 epic barrier (`cargo test`,
  `proof/regen.sh --check`, `lake build` in `proof/` and `proof/corpus/`) was not re-run here and is
  unaffected by these three new files.