# physical computation bounds

Merged epic record: `intent/physical-computation-bounds.md`, `spec/physical-computation-bounds.md`, `plan/physical-computation-bounds.md` (one file per epic; see `docs/README.md`).

## Intent — physical computation bounds

- **Owner:** lead; first-pass workers `deepseek-flash` (speed lane)
- **Source revision:** Apeiron HEAD `74c037b1cfc79b0d6be6d3907398fe926d78d12d`, working tree dirty (155 recorded paths, preserved); canonical corpus HEAD `880aa57e47e782ed7a9d9ee1974cfc87360b446b`, clean.
- **Status:** draft
- **Spec:** `docs/epics/physical-computation-bounds.md`
- **Plan:** `docs/epics/physical-computation-bounds.md`

### Problem

The execution model is resource-bounded, and four physical constraints on it are stated in design
discussion but carry no artifact: holographic state capacity, causal metric locality,
action-bounded transitions, and non-equilibrium dissipation budgets. Song executes the first half of
this picture — fixed caps (`FUEL_PASSES`, `NODE_CAP`, `STACK_CAP` in
`../song/src/prover/ring/prover0.slang`) and a `FUEL` verdict that reports exhaustion without a
false answer. No tree carries the second half: no counting core for a boundary-scaled capacity, no
latency lower bound, no first-orthogonality time, and no contraction accounting for an irreversible
map. The algebra, the elliptic metric and the fuel evaluator exist; the constraints are prose.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Keep the four as anchors in discussion | Each has a finite counting or real-analytic core that is provable today; leaving it unmapped keeps a gap with no gate. |
| Port the predecessor modules wholesale | `../sanguine-ref/proof/Substrate/Algebra/{Gravity,Matter,NullCell}` is pre-trichotomy and the corpus requires re-derivation on `Apeiron.Trichotomy`; transcription carries the old rooting. |
| State the physical bound as an axiom or hypothesis | Violates the fail-closed cone in `../sanguine/AGENTS.md` §3a; no new axioms, and an unproved identification is `anchor`, not `proved`. |
| Prove the full physical statement (Bekenstein number, `ħ`, `k_B`) | Those constants are unit conventions or empirical inputs; the corpus proves the structural core and names the substitution as a mapping. |

### Completion condition

Four rooted corpus modules compile with kernel-checked cores, carry `#audit` lines in
`proof/Apeiron/Audit.lean` with the cone inside `{propext, Classical.choice, Quot.sound}`, and the
corpus barrier is green.

```sh
cd /home/socol/Workspace/sanguine/proof && lake build
```

### Invariants

1. No `sorry`, `admit`, new axiom, `maxHeartbeats 0` or unbounded kernel reduction enters the corpus.
2. No existing declaration is weakened, renamed or removed; a restatement beside an authority is not an edge.
3. The four physical identifications (`k_B T`, `ħ`, `c`, `A/4ℓ_P²`) stay `anchor` readings until a definitional bridge exists; no theorem claims them.
4. One in-progress owner per file; the shared corpus `.lake` and `Audit.lean` are single lanes.

### Verification hypotheses

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| Finite contraction counting: fiber partition, image cardinality, `Bool`-erasure fiber of size `2` | proved | `Apeiron.Physics.Landauer`, `lake build` |
| `W ≥ k_B T ln(Ω_in/Ω_out)` as a physical bound | model/anchor | `landauerCost` definition; substitution named, not proved |
| First orthogonal time of the rotor: `π/2`, and `π/(2√E)` at energy scale `E` | proved | `Apeiron.Physics.ActionClock`, `lake build` |
| `Δt_op ≥ πħ/(2ΔE)` | model/anchor | reading of the rotor theorem; no `ħ` in the corpus |
| Access-time lower bound `τ ≥ d/c` on the elliptic face | proved, elliptic only | `Apeiron.Projective.Latency` from `log_height_le_arcLength` |
| Hyperbolic causal condition on a path | next | `exists_null_direction_iff_disc_nonneg`; `Projective/Length.lean`'s header records its absence |
| Boundary-linear capacity cannot carry a square region (`4n−4 < n²`) | proved | `Apeiron.Physics.Capacity`, `lake build` |
| Bekenstein `S = A/4` | model/anchor | predecessor `../sanguine-ref/proof/Substrate/Algebra/Gravity/BlackHoleThermodynamics.lean:101`; re-rooting required, not authority |

### Non-goals

- No claim that these theorems describe deployed hardware, physical spacetime or a measured machine.
- No SI constants: only counting, real analysis and named unit-scale parameters.
- No change to Song, Rust, extraction, Aeneas or the bridge pin.
- No migration of predecessor modules into the corpus; re-derivation is a later wave with its own lifecycle.

### Trust boundary

The claims rest on the Lean kernel, Mathlib, and the corpus's existing rooted definitions. The
physical reading — that these are the thermodynamics, causal structure or action of a real machine —
is not covered by any kernel check and is recorded as `anchor`. No executing system is refined, and
no compiler, runtime or hardware claim follows.

### Open questions

- Holographic pillar: counting core only, or a re-rooted Bekenstein statement in scope? Requires a corpus economics decision; owner: lead.
- Second wave order after first-pass results: full `arcLength` triangle inequality, or coupling the Landauer counting core to `CarrierDissipationAnyRing` transport? Owner: lead.

## Specification — physical computation bounds

- **Owner:** lead; first-pass workers `deepseek-flash`
- **Intent:** `docs/epics/physical-computation-bounds.md`
- **Source revision:** Apeiron HEAD `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; canonical corpus HEAD `880aa57e47e782ed7a9d9ee1974cfc87360b446b`
- **Status:** draft

### Scope and admissible domain

Four independent targets, one corpus module each. All are total over their stated types; none
introduces a pre-existing structure, an axiom or a measure.

| Item | Operation | Domain | Totality | Preconditions | Postcondition |
|---|---|---|---|---|---|
| Dissipation | finite-set contraction counting | finite types `α`, `β`, decidable equality on `β` | total | none | fiber partition sums to `card α`; image card `≤ card α`; a `Bool × α → α` fiber has card `2` |
| Action clock | first orthogonal time of `flow (-1)` | `t : ℝ`, scale `E > 0` | total | `0 < E` for the scaled form | least positive `t` with `cos t = 0` is `π/2`; with `cos (√E · t) = 0` is `π/(2√E)` |
| Causal latency | elliptic half-plane access time | heights `0 < p ≤ q`, speed `c > 0`, path `γ` with the `log_height_le_arcLength` hypotheses | total | `0 < c` | `Real.log (q/p) / c ≤ arcLength (-1) γ a b / c`; positivity and monotonicity |
| Capacity | square-region boundary count | `n : ℕ` | total | `2 ≤ n` (cardinality), `3 ≤ n` (strict bound) | boundary card `= 4n − 4`; `4n − 4 < n²` |

No silent domain reduction. The elliptic latency statement keeps its face qualifier: above `σ = 0`
the norm form takes negative values, the square root is not a length, and a causal condition on the
path is required first. That condition is named as `next`, not assumed.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `fiber_card_sum` | `∑ b, (univ.filter (f · = b)).card = Fintype.card α` | proved | `Apeiron.Physics.Landauer` |
| `card_image_le_card` | `(univ.image f).card ≤ Fintype.card α` | proved | same module |
| `log_card_image_le_log_card` | `log (image card) ≤ log (card α)` for nonempty `α` | proved | same module |
| `bool_erasure_fiber_card` | `(univ.filter (p : Bool × α => p.2 = a)).card = 2` | proved | same module |
| `landauerCost` | `k * T * Real.log (m / n)`; nonneg under `0 ≤ k, T`, `n ≤ m` | proved (definitional) | same module; physical bound `anchor` |
| `rotor_orthogonal_iff` | `Real.cos t = 0 ↔ ∃ k : ℤ, t = π/2 + k * π` | proved | `Apeiron.Physics.ActionClock` |
| `first_orthogonal_time` | `IsLeast {t | 0 < t ∧ Real.cos t = 0} (π/2)` | proved | same module |
| `first_orthogonal_time_scaled` | `0 < E → IsLeast {t | 0 < t ∧ Real.cos (√E * t) = 0} (π/(2√E))` | proved | same module; `Δt ≥ πħ/(2ΔE)` anchor |
| `accessTime` | `Real.log (q/p) / c` on `0 < p ≤ q`, `0 < c` | proved (definitional) | `Apeiron.Projective.Latency` |
| `accessTime_lower_bound` | under the `log_height_le_arcLength` hypotheses, `accessTime p q c ≤ arcLength (-1) γ a b / c` | proved | same module |
| `accessTime_pos` | `p < q → 0 < accessTime p q c` | proved | same module |
| `accessTime_mono_height` | `q ≤ r → accessTime p q c ≤ accessTime p r c` | proved | same module |
| `squareBoundary_card` | `2 ≤ n → (squareBoundary n).card = 4 * n − 4` | proved | `Apeiron.Physics.Capacity` |
| `area_exceeds_boundary` | `3 ≤ n → 4 * n − 4 < n * n` | proved | same module; area-law reading `anchor` |

### Representation relation

There is no Rust operation in this epic. Each contract refines an existing canonical corpus
definition or Mathlib statement: `flow` and `one` (`Apeiron/Trichotomy/Flow.lean`), `arcLength` and
`log_height_le_arcLength` (`Apeiron/Projective/Length.lean`), and Mathlib's `Finset`/`Fintype`
cardinality API. No extraction, Aeneas or bridge claim is made.

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| A target that does not kernel-check | not landed; reported `open` with the failing goal | module does not build |
| A physical identification without a definitional bridge | `anchor` in the module header, never `proved` | `../sanguine/AGENTS.md` §5 |
| A statement needing `sorry`, an axiom or `maxHeartbeats 0` | refused; task reports a blocker | fail-closed cone |
| A target that resists after three materially different approaches | report the smallest unresolved goal | apex `AGENTS.md` §10 budget |

### Evidence plan

1. **Positive build.** Bounded corpus module build per task:
   `cd /home/socol/Workspace/sanguine/proof && lake build Apeiron.Physics.<Module>`
   (or `Apeiron.Projective.Latency`), under the §10 runner: `flock /tmp/opencode/apeiron-lane.lock`,
   `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`, `LEAN_NUM_THREADS=1`,
   `timeout 540s`. A run reaching 600 s has failed.
2. **Negative control.** In a throwaway copy: weaken one statement (replace the fiber equality with
   `≤`, or drop `0 < p` from `accessTime_pos`) and confirm the kernel rejects; restore and confirm
   green. A control that has never gone red is not a control.
3. **Audit.** A separate serialized barrier task adds `#audit` lines to `proof/Apeiron/Audit.lean`
   for every headline declaration, cone inside `{propext, Classical.choice, Quot.sound}`.
4. **Linters.** `scripts/check-style.py`, `scripts/check-lints.py`, `scripts/check-imports.py` at
   the barrier, not per first-pass task.
5. **Measurement.** None; no performance claim is made.

### Trust boundary

The corpus proves counting and real-analytic identities over its rooted algebra. The substitution of
`k_B`, `ħ`, `c` and `A/4ℓ_P²` is a declared mapping onto units and geometry, not a kernel claim. The
predecessor modules in `../sanguine-ref` are historical reference; nothing here is refined against an
executing system, and Song's `FUEL` behaviour is cited, not re-proved.

### Open questions

- Is the hyperbolic-face causal condition (`exists_null_direction_iff_disc_nonneg`) part of this
  epic's closure or a separate intent? Decision needed before the latency wave closes.
- Does the capacity module stay pure counting, or may it name a cell-count-to-area map? The map is
  the only way the Bekenstein reading becomes more than `anchor`.

## Plan — physical computation bounds

- **Owner:** lead; first-pass workers `deepseek-flash` (speed lane, user-directed)
- **Intent:** `docs/epics/physical-computation-bounds.md`
- **Spec:** `docs/epics/physical-computation-bounds.md`
- **Source revision:** Apeiron HEAD `74c037b1cfc79b0d6be6d3907398fe926d78d12d`, dirty (preserved); canonical corpus HEAD `880aa57e47e782ed7a9d9ee1974cfc87360b446b`, clean.
- **Status:** W1 green 4/4 with red-then-restored controls; W2 audit registration landed and the corpus barrier plus the three linters are green; W3 (predecessor re-root) open.

### Arc Block — emitted before the first W1 edit

```text
SCOPE:      four new canonical corpus modules (Landauer, ActionClock, Latency, Capacity), one
            rooted counting/analytic core each; completion: each module kernel-builds green and
            the four physical identifications are recorded as anchor; audit registration W2.
REGISTER:   local algebra/analysis obligation plus planning; excludes Rust, extraction, Song,
            measurements and any completed physical claim.
CORPUS:     Apeiron/Trichotomy/Flow.lean: flow_neg_one, flow_add, flow_returns_iff_disc_neg
            (proved); Apeiron/Projective/Length.lean: arcLength_vertical,
            log_height_le_arcLength (proved, elliptic face only); Apeiron/Linear/Basic.lean:
            consumeAt_second_alias_is_nothing; Apeiron.Combinator.Comb.verdict_decided_exact.
REFERENCE:  /home/socol/Workspace/sanguine/proof/Apeiron/Trichotomy/Flow.lean;
            /home/socol/Workspace/sanguine/proof/Apeiron/Projective/Length.lean;
            /home/socol/Workspace/sanguine-ref/proof/Substrate/Algebra/Gravity/BlackHoleThermodynamics.lean;
            /home/socol/Workspace/sanguine-ref/proof/Substrate/Algebra/Matter/ThermalStatistics.lean.
GATE:       per W1 module: cd /home/socol/Workspace/sanguine/proof && lake build
            Apeiron.<module> under the §10 runner; observed red baseline: each module does not
            exist (lake: unknown target). MISSING at plan time: audit lines, linters, controls.
GRADE:      planning/anchor only; before: four absent items, no corpus target; after: these four
            modules are the first-pass targets, no theorem claimed yet.
```

### Ground rules

The corpus carries the proofs: Apeiron `AGENTS.md` §4.4 puts generic mathematics in the pinned
canonical corpus at `../sanguine/proof/Apeiron/`. `lakefile.lean` globs `Apeiron.*`, so a
new module enters the build the moment it exists; no root list is edited. Corpus writing,
rooting, satisfiability, one-authority, fail-closed and audit rules in
`../sanguine/AGENTS.md` §3a and `theory/WRITING.md` govern over this plan.

W1 runs in the supplied corpus checkout, recorded exception: the warm `.lake` is 7.7 GiB and cannot
be duplicated per task inside the §10 budget. Each worker edits exactly one declared new file, never
`proof/Apeiron/Audit.lean`, never any other path; no worktree, branch, commit or push. Builds
serialize on `flock /tmp/opencode/apeiron-lane.lock`:

```sh
cd /home/socol/Workspace/sanguine/proof && \
  flock /tmp/opencode/apeiron-lane.lock \
  systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0 \
  env LEAN_NUM_THREADS=1 timeout 540s lake build Apeiron.<module>
```

No `sorry`, new axiom, `set_option maxHeartbeats 0`, raised recursion depth or unbounded kernel
reduction. After three materially different failed approaches, stop and report the smallest
unresolved goal. Module headers follow the sibling `Apeiron/Physics/*.lean` conventions; `@grade`
is descriptive and no headline is `proved` until its `#audit` line exists.

### Portable inventory (read this pass)

Predecessor items are historical reference and re-derive on the trichotomy; none is authority.

| Pillar | Portable artifact (path: declaration) | Native grade / state |
|---|---|---|
| Capacity | corpus `Projective/Length.lean:179` `log_height_le_arcLength`; `:140` `arcLength_vertical` | proved; elliptic face only, causal condition absent |
| Capacity | predecessor `Substrate/Algebra/CellBoundaryFlux.lean:53,58,69` `cell_coupling_is_boundary_difference`, `cell_interval_boundary_balance`, `cell_boundary_flux_closure`; `Substrate/Algebra/FiniteCellBoundary.lean` defs `reverse`, `forwardEdge`, `IsForwardEdge` | historical; re-root owed |
| Capacity | predecessor `Substrate/Algebra/Gravity/BlackHoleThermodynamics.lean:101,128,137` `bekenstein_is_area_over_four`, `first_law`, `bekenstein_from_temperature`; `Substrate/Algebra/Gravity/HorizonInformation.lean`; `Substrate/Algebra/Gravity/BlackHoleCeiling.lean` | historical, pre-trichotomy; re-derivation required |
| Locality | corpus `Trichotomy/Flow.lean:328` `flow_returns_iff_disc_neg`, `:181` `flow_add` | proved; hyperbolic face is non-returning |
| Locality | predecessor `Substrate/Algebra/Gravity/GeodesicDeviation.lean`, `Substrate/Algebra/Gravity/ExactNullGeometry.lean`, `Substrate/Algebra/Gravity/EnergyConditions.lean` | filenames read; contents NOT read — retrieval required before citation |
| Action | corpus `Trichotomy/Flow.lean:117` `flow_neg_one`, `:221` `hasDerivAt_flow`; `Physics/Wick.lean` `flow_neg_one_eq_rot`, `smoothness_fixes_temperature` | proved |
| Action | predecessor `Substrate/Algebra/Gravity/ChaosBound.lean:129,136,148,172` `rotor_closes_fundamental`, `fundamental_rate`, `chaos_scale_is_monad_rotor`, `chaos_bound_is_monad_rotor`; `Substrate/Algebra/QuantumField/WickRotation.lean` `rotor_wick_to_boost` | historical; MST-type time bound, re-root owed |
| Dissipation | corpus `Physics/CarrierDissipationAnyRing.lean` `diffusiveUpdate_total_mass`, `diffusiveUpdate_continuity`, `slope_diffusiveUpdate` | proved; total mass conserved, no entropy production |
| Dissipation | corpus `Linear/Basic.lean` `consumeAt_second_alias_is_nothing`; `Combinator/Verdict.lean:420` `verdict_decided_exact` | proved; fuel is a counter, not a potential |
| Dissipation | predecessor `Substrate/Algebra/Matter/ThermalStatistics.lean:157,196,234` `maxent_forces_boltzmann`, `boltzmann_kB_is_units`, `thermal_statistics_characterization`; `Substrate/Algebra/NullCell/NullCellCoding.lean:92,167` `info_additive`, `null_cell_is_the_entropy_coder`; `Substrate/Algebra/Neural/CountLMFloor.lean:101,142` `cross_eq_ent_add_kl`, `cross_ge_entropy` | historical; entropy side of Landauer, re-root owed. The names resolve at these full paths; the earlier shorthand `cross_ge_ln` does not resolve anywhere in the provisioned checkout and is replaced by `cross_ge_entropy`. |

### Wave map

W1 tasks are file-disjoint and parallel. W2 is one owner on `Audit.lean`. W3 is design-blocked.

| Task | Files (one each) | Owner | Gate (bounded) | Negative control |
|---|---|---|---|---|
| `APEIRON-PHYS-W1-T1` | C: `proof/Apeiron/Physics/Landauer.lean` | `deepseek-flash` | `lake build Apeiron.Physics.Landauer` | scratch: fiber equality → `≤` must fail kernel; restore |
| `APEIRON-PHYS-W1-T2` | C: `proof/Apeiron/Physics/ActionClock.lean` | `deepseek-flash` | `lake build Apeiron.Physics.ActionClock` | scratch: replace `π/2` by `π`; `IsLeast` must fail; restore |
| `APEIRON-PHYS-W1-T3` | C: `proof/Apeiron/Projective/Latency.lean` | `deepseek-flash` | `lake build Apeiron.Projective.Latency` | scratch: drop `0 < c` or `0 < p`; statement must fail; restore |
| `APEIRON-PHYS-W1-T4` | C: `proof/Apeiron/Physics/Capacity.lean` | `deepseek-flash` | `lake build Apeiron.Physics.Capacity` | scratch: `4n−4` → `3n−4` must fail; restore |
| `APEIRON-PHYS-W2-T1` | C: `proof/Apeiron/Audit.lean` (serialized) | lead or assigned worker | `lake build` after adding `#audit` per headline; closed cone | scratch: insert an axiom-backed declaration and confirm the audit rejects; restore |
| `APEIRON-PHYS-W3-T1` | C: design task (no file yet) | lead | corpus economics decision recorded as an intent update | n/a — planning |

C = `/home/socol/Workspace/sanguine`. W3-T1 is the re-root of the portable predecessor
items (Bekenstein chain, thermal statistics, chaos bound) and does not open before W2 is green and
the lead resolves the spec's open questions.

### Task blocks

#### W1-T1 · finite contraction counting core

- **Scope:** the counting core of the Landauer reading, in one new module.
- **Files:** C `proof/Apeiron/Physics/Landauer.lean`
- **Depends on:** none
- **Grade:** proved for the counting statements; the `kT ln` bound is `anchor` in the header.
- **Gate:** bounded `lake build Apeiron.Physics.Landauer`.
- **Targets:** `fiber_card_sum`, `card_image_le_card`, `log_card_image_le_log_card`,
  `bool_erasure_fiber_card`, `landauerCost` + nonnegativity; a header sentence states the
  thermodynamic substitution as a mapping. Import only what the proofs use; keep the module
  Mathlib-minimal in the corpus's combinator spirit.
- **Evidence to record:** build command, exit status, elapsed seconds, module sha256.

#### W1-T2 · first orthogonal time of the rotor

- **Scope:** least positive time at which the elliptic flow is orthogonal to the unit.
- **Files:** C `proof/Apeiron/Physics/ActionClock.lean`
- **Depends on:** `Apeiron.Trichotomy.Flow` (`flow_neg_one`), local Euclidean `inner` on `Cell`.
- **Grade:** proved for the rotor statements; `Δt ≥ πħ/(2ΔE)` is `anchor` in the header.
- **Gate:** bounded `lake build Apeiron.Physics.ActionClock`.
- **Targets:** `rotor_orthogonal_iff`, `first_orthogonal_time`, `first_orthogonal_time_scaled`.
- **Evidence to record:** as W1-T1.

#### W1-T3 · elliptic access-time lower bound

- **Scope:** latency as arc length over speed `c`, lower-bounded on the elliptic face.
- **Files:** C `proof/Apeiron/Projective/Latency.lean` (imports `Apeiron.Projective.Length`)
- **Depends on:** `log_height_le_arcLength` and `arcLength_vertical`.
- **Grade:** proved, elliptic face only; the causal condition above `σ = 0` is named `next`.
- **Gate:** bounded `lake build Apeiron.Projective.Latency`.
- **Targets:** `accessTime`, `accessTime_lower_bound`, `accessTime_pos`, `accessTime_mono_height`.
- **Evidence to record:** as W1-T1; include the exact hypothesis list borrowed from `log_height_le_arcLength`.

#### W1-T4 · boundary-cardinality core of the area law

- **Scope:** the discrete counting fact behind "boundary, not volume".
- **Files:** C `proof/Apeiron/Physics/Capacity.lean`
- **Depends on:** Mathlib `Finset`/`Fintype` only.
- **Grade:** proved for the counts; the Bekenstein `A/4` reading is `anchor`.
- **Gate:** bounded `lake build Apeiron.Physics.Capacity`.
- **Targets:** `squareBoundary`, `squareBoundary_card` (`2 ≤ n`), `area_exceeds_boundary`
  (`3 ≤ n`). The 2×2 grid is the small witness; state it.
- **Evidence to record:** as W1-T1.

### W1/W2 results

Corpus revision `880aa57e47e782ed7a9d9ee1974cfc87360b446b`, supplied-checkout exception; full
barrier `cd proof && lake build` 3530 jobs exit 0 under the §10 runner. Linters from the repository
root: `scripts/check-style.py` clean (191 files), `scripts/check-imports.py` clean,
`scripts/check-lints.py` clean (build green, no apeiron linter warnings). `proof/Apeiron/Audit.lean`
grew 31 lines (four imports, 21 `#audit` lines).

| Task | Artifact (sha256) | Focused gate | Control | Status |
|---|---|---|---|---|
| W1-T1 | `proof/Apeiron/Physics/Landauer.lean` `d8e6e92b…` | `lake build Apeiron.Physics.Landauer` 1918 jobs, exit 0 | fiber sum `=`→`≤` rejected; restore green | green, audited `Audit.lean:1425-1430` |
| W1-T2 | `proof/Apeiron/Physics/ActionClock.lean` `5bfc2164…` | `lake build Apeiron.Physics.ActionClock` 3084 jobs, exit 0 | `π/2`→`π` rejected at three sites; restore green | green, audited `:512-516` |
| W1-T3 | `proof/Apeiron/Projective/Latency.lean` `f3ea430c…` | `lake build Apeiron.Projective.Latency` 3360 jobs, exit 0 | dropping `hp` rejected; restore green | green, audited `:281-284` |
| W1-T4 | `proof/Apeiron/Physics/Capacity.lean` `44493912…` | `lake build Apeiron.Physics.Capacity` 2966 jobs, exit 0 | `4n−4`→`3n−4` rejected; restore green | green, audited `:1431-1436` |
| W2-T1 | `proof/Apeiron/Audit.lean` +31 lines | `lake build Apeiron.Audit` 3528 jobs, exit 0; full `lake build` 3530 jobs | audit command is itself the check; a documentary registration has no semantic mutant | landed |

The audit output reports `{propext, Classical.choice, Quot.sound}` for every new headline except
`area_exceeds_boundary`, which reports `{propext, Quot.sound}` only.

Open after W1/W2: W3 predecessor re-root (Bekenstein chain, thermal statistics, chaos bound), the
corpus `theory/ROADMAP.md` row, and any corpus publication or pin move, which this plan does not
authorize.

### Barrier checklist

- [x] Each W1 module kernel-builds green on the recorded corpus revision; elapsed within budget.
- [x] W1 touched only its four new files; W2 touched only `Audit.lean` (31 added lines); no other path changed.
- [x] Negative controls run red and restore green, with output retained.
- [x] W2 `#audit` lines added, cone closed; corpus `lake build` and the three linter scripts green.
- [x] Headline claims registered at their native grade; physical readings marked `anchor`.
- [x] Predecessor items remain historical until a re-root wave lands; no authority transferred by citation.

### Stop and rollback conditions

- A module exceeds 540 s of Lean campaign: stop, record elapsed and stage, redesign the target
  (smaller core), never raise the budget.
- A task cannot state its target without an axiom or `sorry`: report the blocker; the statement goes
  back to the spec, not into the corpus.
- Two W1 tasks need the same declaration or file: they are one task; stop and report the collision.
