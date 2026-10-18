# Apeiron–Intent improvement loop

One checked cycle: `objective → observed deficiency → bounded proposal → admission → implementation
→ independent verification → measured result → next proposal`. Objective, acceptance criteria and
checking authority are fixed for the cycle; changing them is a separate, reviewed task. A candidate
cannot rewrite its own gate.

Division. **Apeiron** owns canonical semantics, reusable primitives, executing-source refinements,
resource models and optimization admission. **Intent** turns objectives and observed failures into
bounded proposals, routes them through acceptance, and selects the next task from the resulting
evidence.

## Transitions, artifact and owner

| # | transition | artifact (where it resolves) | owner |
|---|---|---|---|
| 1 | objective | one measured deficiency, named machine/metric/interval in `docs/evidence/gpu-prefill/*.md`, or an Evolver round deficiency in `../intent/runs/evolve-*/eval.json` | lead (objective selection) |
| 2 | observed deficiency | the receipt/cost model that quantifies it (Apeiron `proof/resource/<Name>Resource.lean`; Intent round receipt) | measurement owner |
| 3 | bounded proposal | Intent edit `η`, `|η| ≤ b_t`, `θ'|Φ = θ|Φ` (`../intent/docs/EVOLVER.md` §0); recorded in `../intent/artifacts/<round>/` | Intent proposer (Evolver) |
| 4 | admission | Apeiron `ProofDispatch.Admission` (witness + joint + preserves + improves) under a registered contract in `tools/dispatch/contracts.json`; launched by `tools/dispatch/admit.py --launch` | lead + independent verifier |
| 5 | implementation | the registered worker argv on the real source path (`src/…`), one writer per file | worker |
| 6 | independent verification | kernel check + real-source mutation red → restore green, by a different model than the writer | independent verifier (fresh model) |
| 7 | measured result | before/after on a named machine, metric and interval; `docs/evidence/` receipt | measurement owner |
| 8 | next proposal | Intent consumes the accepted receipt (`readjudicate` — pure replay, not re-measure) | Intent |

## Collected returns and what they unblock

- **Extraction-root repair.** The root resolves (`52cb47b`: `dequant_f32_row` moved into
  `src/engine/num/f32_convert.rs`, byte-identical; the extraction crate compiles) and the digest list
  covers all 38 `#[path]` targets (`6084cca`). What remains is the E0514 sysroot mismatch, so
  `proof/regen.sh --check` and the admission launcher's freshness gate are still blocked.
  **Transition 4 is blocked on that freshness gate for any production contract.**
- **`CostBridge`** (`05f1d72`). Supplies transitions 2→4's cost interface: a primitive's own counts →
  `CostAlgebra.Cost` → dominance. Carries the RowDot count.
- **Source-cost modules** (22 committed). Supply transition 2's receipts per primitive.
- **Intent pending returns**: `artifacts/space-bunny-evolution/` (untracked, 21 files). Local
  acceptance only; it unblocks Intent's proposal handling, not Apeiron admission. The worktrees
  `intent-sb-oracle` and `/tmp/opencode/vtree` no longer exist; the cleanup that removed them captured
  every diff to `/home/socol/oc-tmp/agent-trees-cleanup-1790860389/`, and `sb/oracle-relay` is merged
  into `main` per the campaign hand-back.

## Candidate: `row_dot_fixed` realization

| prerequisite | status |
|---|---|
| canonical cost model of the primitive | `RowDotResource` — proved (`n+1` visits, `n` checked `i128` mul-adds) |
| count → `Cost` → dominance | `CostBridge` — proved (39 headlines, two controls) |
| composition/dominance algebra | `CostAlgebra` — proved |
| admission schema | `ProofDispatch.Admission` — present |
| extraction freshness (launcher gate) | **open** — repair in flight |
| a concrete cheaper real body with preserved output | **open** — design decision (see below) |

Objective (transition 1): reduce the source-level work of the executing `matvec::row_dot_fixed` on a
declared, nonempty reachable-and-admissible domain of equal-length Q32.32 `i64` rows whose integer
prefix sums fit `i128`, preserving its returned value and refusal behaviour. The metric is extracted
loop-body invocations: exactly `n + 1` for width `n`, the terminating visit included, with `n` checked
multiply-add steps (`RowDotResource.entry_visits_exact`). That interval `[n + 1, n + 1]` is a count,
not a latency.

The `docs/evidence/gpu-prefill/` figures (`171.63–506.36` GMAC/s Vulkan `q8_tile`, `6387–52225` GMAC/s
CUDA MMQ) time other dispatch paths with different activation conventions and epilogues. They do not
isolate `row_dot_fixed` and are not attributed to it. Its isolated machine-time baseline is
**unmeasured**: `measured` here requires the microbenchmark of transition 2's receipt.

Deficiency (transition 2): `RowDotResource`'s full-row visit count at its stated premises. It does not
establish that fewer visits are achievable, or that removing visits removes elapsed time.

Acceptance: a source-connected before/after contract with one witness satisfying both reachability and
admissibility; universal equality of observations, refusals included, on the registered domain; and
universally strict reduction in the registered visit-cost model (`afterVisits(x) < n(x) + 1`), with the
boundary cases where strict improvement cannot hold handled or excluded explicitly. Where the
`CostBridge`/`CostAlgebra` order is used, require componentwise non-regression and strict dominance —
not a smaller weighted scalar. Rejecting control: substitute the unchanged baseline as the candidate;
equal observations at `afterVisits = beforeVisits = n + 1` must fail admission as a tie, and the
restored candidate must pass on the same identities. A separate accumulator-update mutation rejects
preservation attributably.

The one open design decision before transition 3: **which cheaper realization**, on which admissible
domain, with equal output. Candidates: a two-element unrolled step (halves visits on even widths); a
hoisted bounds check; a paired accumulate. Each must keep `preserves` true on a reachable joint domain.
Its gate is **MISSING**: `tools/dispatch/contracts.json` registers no target (`"contracts": {}`), so
transition 4 stays closed until a registration row binds the executing source root, the
observation/refusal contract, the joint witness, the source-derived before/after costs, the
certificate namespace and declarations, the owned files and the worker argv.

## Invariants held during the cycle

- The objective, acceptance criteria and checking authority are frozen for the cycle.
- A candidate cannot edit `Φ`, the register, the gate or the evaluator (Intent D1).
- Every claim names its grade: `proved` / `tested` / `measured` / `model`.
- Extracting, registering or measuring a candidate never substitutes for admission.
