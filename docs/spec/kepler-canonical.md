# Specification — canonical elliptic Kepler solve

- **Task/owner:** `APEIRON-KEPLER-W0-T1`, Sol (contract). [Intent](../intent/kepler-canonical.md); [plan](../plan/kepler-canonical.md).
- **Identity/status:** Apeiron HEAD `262bb0a94eff2cc7d5655122a954f44d74de96b6` (dirty shared checkout), canonical corpus pin `3d66ea3aa2fe1de5997b6b757247454a2f771358` (dirty working copy is not the pin); proposal, **model/anchor** unless stated otherwise. No new Lean build, Rust test or numeric measurement is claimed here.

## Exact equation, normalization and service domain

Write `F_{m,e}(x) = x − e * Real.sin x − m` over `ℝ`. The canonical existence, uniqueness, fixed-point and absolute-defect enclosure MUST retain the **general** `m e : ℝ`, `|e| < 1` domain; there is no nonnegative-e assumption in that theorem's domain. For physical service specialize to `0 ≤ e < 1`. Neither mathematical domain implies all such real inputs are representable as binary64.

The proposed `solve_kepler(face: Sigma, mean_anomaly: f64, eccentricity: f64) -> Result<KeplerSolution, KeplerError>` accepts **only** `face == Sigma::Elliptic`, finite inputs denoted exactly as reals `m = ⟦mean_anomaly⟧`, `e = ⟦eccentricity⟧`, `−π ≤ m ≤ π` over the exact reals, and `0 ≤ e ≤ 0.95` (comparison to the binary64 threshold must be specified consistently; an exactly represented `0.95_f64` is the service endpoint, not the unrepresentable decimal real 0.95). The mathematical target is the unique **unwrapped normalized** real `E*(m,e)` solving `F_{m,e}(E*) = 0`; monotonicity and `F(−π) ≤ 0 ≤ F(π)` put it in `[−π,π]`. Input `m` is already reduced; this API carries **no turn count** and MUST NOT silently reduce an arbitrary large finite `f64`. Endpoints satisfying the exact inequality are admitted. `−0.0` is admitted with its sign preserved for circular bit identity.

Penglai MUST own a separate horizon-stable representation `(k, m)` with integer turn count `k`, normalized `m` and the chosen epoch/time-unit mapping. The mathematical reconstruction is `M = 2πk + m`, `E = 2πk + E*(m,e)`. The API bounds **only** `|⟦solution.anomaly⟧ − E*(m,e)|`, not the reconstructed absolute `E`, input phase reduction, or long-horizon orbital phase. The integer width/overflow, epoch representation, modulo-`2π` error and conversion from a large finite time or anomaly MUST be specified, measured and checked in the Penglai adapter; a large finite raw `M` outside `[-π,π]` returns `UnsupportedDomain`, including cases where forming or subtracting `2πk` in `f64` loses low bits. No assertion about drift follows from the Kepler enclosure alone. The decimal/`f64` endpoint choice and phase protocol require the SPIKE slot below before implementation.

## Enclosure and certificate interpretation

**NEW canonical proof obligation (not a predecessor theorem).** For `m e x d : ℝ`, `|e| < 1`, `0 ≤ d`, define `E*` by the *constructed* unique real solution of `F_{m,e}(E*) = 0`. Prove

```text
|F_{m,e}(x)| ≤ d  →  |x − E*| ≤ d / (1 − |e|).
```

The proof uses `|sin x − sin y| ≤ |x−y|` and `|e| < 1`: `(1−|e|)|x−E*| ≤ |F(x)−F(E*)| = |F(x)|`. For physical `0 ≤ e < 1`, `1−|e| = 1−e`, so the requested enclosure is `|F(Ê)| ≤ d ⇒ |Ê−E*| ≤ d/(1−e)`. The left premise is **not** the rounded computation of `x − e * sin(x) − m`.

The bridge certificate interpretation MUST state a denotation relation for the **exact rational values** of the input/output binary64 bit patterns (including signed zero and finite-only admission), an outward-rounded nonnegative finite `d` satisfying `|⟦Ê⟧ − ⟦e⟧ sin(⟦Ê⟧) − ⟦m⟧| ≤ ⟦d⟧` in `ℝ`, and a sound outward-rounded finite returned error upper bound `r ≥ d/(1−⟦e⟧)` (or refuse `CertificateFailure` if it cannot produce finite bounds). A returned interval `[lo,hi]` MAY additionally enclose `E*` provided endpoint arithmetic is outward and `lo ≤ E* ≤ hi` is justified; the mandatory error enclosure is an absolute-error upper bound `r`. Checking `|fl(Ê−e*fl(sin Ê)−m)| ≤ d` is **insufficient**: an independent bound on argument reduction and transcendental `sin`, every arithmetic rounding and cancellation, underflow/subnormal behavior, overflow, directed rounding or a rigorously bounded substitute, and outward conversion into the reported `d`/`r` is required. A bridge theorem conditional on a supplied bound does not establish that Rust constructed the bound. No Aeneas `f64` source refinement exists for this path; `F64 -> Float` adapter naming fixes a name, not the semantics of IEEE or transcendental operations.

Set `ε = 2^-52` (binary64 machine epsilon here, **not** `2^-53`). Successful service MUST deliver `d ≤ 4ε` and `r ≤ 4 κₑ ε`, where `κₑ = 1/(1−e)`, and `|Ê−E*| ≤ r`. At `e = 0`, return the exact original anomaly bit pattern (including `−0.0`) with zero iterations, `d = 0`, `r = 0`; after finiteness, face and domain checks, no trigonometric evaluation is needed. At `e = 0.95_f64`, use `κ` computed from its exact denoted value, not from decimal 0.95. The resulting analytic upper bound is below `10^-9 rad` throughout the admitted `e` domain, conditional on the sound certificate; the useful-science threshold is not itself a proof of the runtime certificate.

Required non-vacuity: one joint Lean witness for `m = 0, e = 0, E* = 0, d = 0` and another for positive admitted `e`, with the stated prerequisites **at the same real types**; Rust constructor/test witnesses for circular, noncircular, endpoint, near-boundary and each refusal; preservation of the certificate invariant for every successful return. Existing `Sanguine.KeplerCollapse.kepler_floor_exists_unique` in `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Cosmology/KeplerCollapse.lean:174` constructs `∃! E : ℝ, M = E − e * Real.sin E` under `|e| < 1`; it is a predecessor pointer, not a built canonical theorem or an audited axiom-cone receipt for this task.

## Executing interface and refusal

`src/kepler.rs` MUST reuse `crate::cell::Sigma` from `src/cell.rs:16–45`: `Elliptic.value() == -1`, `Parabolic.value() == 0`, `Hyperbolic.value() == 1`. `Sigma` derives `Default` as `Parabolic`: NEVER use `Sigma::default()` as a solver face. The host maps physical bound-orbit eccentricity to `Sigma::Elliptic` explicitly; the enum by itself proves no physical classification. `src/kepler.rs` is the authority for the solution, defect, bound and iteration count; Penglai readouts/caches are not authorities.

```rust
solve_kepler(face: Sigma, mean_anomaly: f64, eccentricity: f64)
    -> Result<KeplerSolution, KeplerError>
```

`KeplerSolution` MUST carry the input `face`, the **normalized** eccentric anomaly (`f64`), an outward certified absolute defect upper bound (`f64`), an outward absolute-error enclosure (`f64`), and correction-iteration count (`u8` or other explicitly bounded integer). Exact field spelling is an implementation choice; semantics are not. No partial result or readout is committed on `Err`; the pure kernel mutates no caller state, has no filesystem/rendering/clock input, and never panics on admitted input. Penglai MUST commit any orbit/season/tide state transactionally only after `Ok`; all `Err` paths preserve prior gameplay state, with save/replay version and determinism checked if script-visible/persisted state changes.

| Refusal (`KeplerError`, typed variant required) | Input / execution condition |
|---|---|
| `NonFiniteInput` | Either numerical input is NaN or ±infinity; checked before comparison/iteration. |
| `InvalidEccentricity` | Finite `e < 0` or `e ≥ 1`; no clamping or substitution. |
| `UnsupportedFace` | `Sigma::Parabolic` or `Sigma::Hyperbolic`; checked explicitly, even at `e = 0`. |
| `UnsupportedDomain` | Finite `m ∉ [−π,π]` or physical `0 ≤ e < 1` outside the service limit `e ≤ 0.95_f64`; no implicit time reduction. |
| `IterationExhaustion` | Solver cannot produce a candidate in the allocated count, not a routine escape for the promised admitted population. |
| `CertificateFailure` | Cannot form/validate the outward real-residual and error bounds, including nonfinite/unsound enclosure arithmetic; never return an uncertified candidate. |

Checks have deterministic precedence `NonFiniteInput` → `UnsupportedFace` → `InvalidEccentricity` → `UnsupportedDomain` → execution/certificate. Reject unrepresentable/enclosure-overflow cases with a typed failure, never zero/default substitutes. The defined algebraic service faces do **not** follow from arbitrary real `|e| < 1` in the mathematics; the host chooses the face.

## Iteration, phase and open measurement slots

Count one iteration for each complete candidate correction (including Newton, Halley or safeguarded fallback); certificate evaluation/rechecks and reduction MUST be recorded separately as cost, not hidden inside an iteration. Circular shortcut takes zero iterations. A valid SLA means **every** admitted `(face, m, e)` receives `Ok` with sound `d ≤ 4ε`, `r ≤ 4κₑε`, and count `≤ 5` for `0 ≤ e ≤ 0.45_f64` or `≤ 8` for `0.45_f64 < e ≤ 0.95_f64`; a loop capped at those numbers that returns arbitrary `IterationExhaustion`/`CertificateFailure` does **not** satisfy service. Tests over samples cannot prove a universal bound: a checked analytic bound or a clearly finite exhaustively covered domain is required for universal language. Until demonstrated, the limits are **SLOT-SPIKE** design targets, with explicit typed refusal and the SLA row OPEN. No silent threshold relaxation, domain narrowing, or promotion of sample tests to universality.

| Slot (all `model/anchor`, not measurements here) | Required source / decision and fallback |
|---|---|
| `SLOT-SPIKE-ITER` 5/8 corrections at `0.45_f64`/`0.95_f64` | Penglai iteration-kernel spike: record command, machine, oracle, metric, interval, coverage, version and instrument control. If infeasible, obtain an approved new contract or retain typed refusal **without** claiming promised-domain service. |
| `SLOT-SPIKE-CERT` `4ε` and `4κₑε` | Same spike + sound independent interval/transcendental-error study on selected platform; demonstrate bound construction and semantic inward-rounding control. If unavailable, return `CertificateFailure` and leave SLA open. |
| `SLOT-SPIKE-PHASE` `10^-12 rad` over `10^6` orbits | Penglai spike + horizon-stable upstream phase/time harness, before/after naive-accumulation falsifier, command/machine/metric/interval and oracle validation. If unavailable, refuse upstream unrepresentable/unbounded phase input; solver's normalized-domain guarantee cannot substitute. Supplied ~`2.5e-5 rad` naive-growth number is unvalidated context. |
| `SLOT-ADAPTER` turn-count integer range, stable modulo error, gameplay gate and save/replay transition | Penglai owner supplies concrete call-site/format inventory and acceptance command; until then no integration dispatch or claim. |

## Predecessor attribution and migrations

All names below resolve in `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Cosmology/KeplerCollapse.lean`, namespace `Sanguine.KeplerCollapse`, over `ℝ`; **source-read pointers**, not re-run `proved` claims. The old imports `Substrate.StationaryPhase.SpinorLoop` and `Substrate.StationaryPhase.BiasInjectedRecovery` are **not** migrated as dependencies. No predecessor file is removed. A separately authorized retirement would require its other consumers' inventory and reachability gate; that inventory is not supplied here.

| Exact predecessor declaration | Exact premise / conclusion and disposition |
|---|---|
| `keplerMap` (def) | `M + e * Real.sin E`; **re-derive** fixed-point map in canonical real module. |
| `kepler_iff_fixedpoint` | All `M e E : ℝ`; `M = E − e*sin E ↔ keplerMap M e E = E`; **re-derive**. |
| `keplerMap_lipschitz` | All real inputs; `|g(E₁)−g(E₂)| ≤ |e|*|E₁−E₂|`; **re-derive**, retain `|e|`. |
| `keplerMap_strictly_contracts` | `|e| < 1`, `E₁ ≠ E₂`; strict distance decrease; **re-derive**. |
| `keplerMap_contractingWith` | `|e| < 1`; `ContractingWith ⟨|e|,abs_nonneg e⟩ (keplerMap M e)`; **re-derive** with the actual complete `ℝ` space. |
| `kepler_unique` | `|e| < 1` and **both** fixed-point hypotheses; `E₁ = E₂`; **re-derive** without assuming existence. |
| `eccentricAnomaly` (noncomputable def) | `|e| < 1`; `ContractingWith.fixedPoint`; **re-derive** constructed unique witness, not a runtime algorithm. |
| `eccentricAnomaly_isFixedPt`, `eccentricAnomaly_solves` | `|e| < 1`; constructed point fixed / solves `M = E − e*sin E`; **re-derive** both. |
| `kepler_floor_exists_unique` | `|e| < 1`; `∃! E : ℝ, M = E − e*sin E`; **re-derive** and independently add new residual/error enclosure. |
| `circular_is_monad` | All `M : ℝ`: constant map at `e=0`, `∃!` real solution, `∀ E, M = E−0*sin E ↔ E=M`; **specialize** to canonical circular theorem; binary64 signed-zero bit identity remains a runtime test obligation. |
| `kepler_collapse_trinity` | `|e| < 1`; conjunction: fixed-point iff at `E₁`, strict decrease if `E₁ ≠ E₂`, uniqueness *if both points are fixed*; **inventory-only**, reuse components above, no second headline authority. |
| `kepler_is_spinor_loop_collapse` | `hA : A ≠ 1`; conjunction of Kepler iff in `M,e,E` and `SpinorLoop.constantStep_fixed_point` in unrelated `A,B,x`; **not-reproduced**. Source header explicitly calls the same-object reading open; no affine nonlinear transport follows. |
| `keplerResidual` | `(keplerMap M e E − E)^2`; **re-derive** the squared-residual definition only as useful legacy inventory; it is `F(E)^2`, not `|F(E)|`. |
| `keplerRecovery` | `|e| < 1`; `BiasInjectedRecovery ℝ` with square nonnegative, zero at constructed anomaly, strictly positive off it; **not-reproduced as hierarchy**. Preserve its mathematical content by a new unique-zero result on squared residual; no SAT/folding equivalence claim. |
| `keplerRecovery_unique_floor` | `|e| < 1`; squared residual zero iff `E = eccentricAnomaly M e he`; **re-derive** from uniqueness without the old hierarchy. |
| `kepler_is_bias_injected_recovery` | `|e| < 1`; conjunction of fixed-point iff and squared-residual unique-zero iff; **inventory-only** as conjunction, preserve both components but do not migrate the `BiasInjectedRecovery` structure. |

`Combinator/Linear/Core/Contraction.lean` concerns combinatory duplication (`W x y = x y y`), **not** a Banach convergence result; `AnyRing.ringAffineCompose_apply` is affine composition and cannot solve nonlinear `M+e sin E`. `Numeric.Accumulator.lossy_residue_bounded` assumes a rounding bound, not a proof of binary64 sine or range reduction. These are excluded, not cited as solver facts.

## Trichotomy connection and evidence boundary

`/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Trichotomy/Flow.lean:117`, `Apeiron.Trichotomy.flow_neg_one (t : ℝ)`, states `flow (-1) t = (Real.cos t, Real.sin t)`; therefore the **new target canonical equation** is exactly `m = E − e * (Trichotomy.flow (-1) E).2` iff `F_{m,e}(E) = 0`. The new module MUST prove this projection rewriting and tie the unique constructed `E*` to the same projection, rather than importing `flow` unused. `Physics/Quadratic.lean:608`, `Apeiron.Physics.flow_is_one_param (σ : ℝ) (x0 : Cell) (t : ℝ)`, proves the derivative of `u ↦ mul σ (flowPoint σ u) x0` is generator multiplication at each `t`; `flow_is_one_param_unique` at :683 additionally requires `∀ t, HasDerivAt g (mul σ (0,1) (g t)) t` and `g 0 = x0`, then identifies `g` with that flow. The `flowPoint` in Track A2 is a **different definition** from `Trichotomy.flow`: equality at `σ=-1` must be proved through their respective negative-one formulas before using A2 as a direct link. `Trichotomy/Derived/Elliptic.lean:236`, `Apeiron.Trichotomy.ellipticEquiv_flow (σ : ℝ) (hσ : σ < 0) (t : ℝ)`, maps `flow σ t` to `Complex.exp ((√(-σ)*t : ℝ) * Complex.I)`; it supplies a useful real-to-complex flow map, **not** Kepler existence or an algorithm. General negative `σ` uses the `√(-σ)` angular rescaling; only normalized `σ=-1` has the literal second projection `sin E`. The physical orbit and its seasons/tides are **model/anchor** readings; the requested projection/enclosure and equality-to-A2 link are **open proof targets**. Current dirty corpus revisions and unaudited new bridge carry no fresh `proved` grade here.

## Evidence and trust boundary

| Claim | Current grade / required discharge |
|---|---|
| Exact predecessor statements and flow declarations | `model/anchor` source-read pointers in this contract; `proved` for NEW canonical theorems only after `cd <corpus>/proof && lake build`, joint witnesses and `Audit.lean`/`AuditCarrier.lean` audit within `{propext, Classical.choice, Quot.sound}`. |
| Revision-pinned certificate interpretation | OPEN; `proved` may cover the **conditional implication** from a sound outward bound on exact-real residual, after `proof/corpus/AxiomAudit.lean` closed-cone audit on the resolved published pin. No executable `f64` conclusion from that alone. |
| Runtime soundness, refusal and iterations | OPEN; positive Rust tests, independent oracle and attributable baseline/red/restore mutation controls yield `tested` at their sampled scope. An actual source-linked f64 refinement would require new extraction and appropriate semantics, not a name fix. |
| Iteration, certificate cost and million-orbit drift | OPEN `SLOT-SPIKE`; `measured` only after instrument controls plus command, named machine, metric and interval; samples do not prove universal service. |

Rust's host/floating-point transcendental implementation, compiler/LLVM, ABI, hardware rounding, overflow/subnormal semantics and any libm dependency are external trust; a Luau script or sandbox MUST NOT manufacture or silently replace `d`, `r` or the face. Penglai's clock, orbit policy, transactional state, renderer, save and replay are external. No `proof/regen.sh` / Aeneas `f64` refinement is claimed, and a bridge input-premise theorem must not be reported as a constructed runtime certificate.

## Additive per-orbit plan API (APEIRON-KEPLER-W8-T2)

The frozen `solve_kepler` contract above is unchanged. An additive, optional entry point folds every eccentricity-dependent constant once per orbit and MUST return results bit-identical to `solve_kepler` (tested over the sampled population):

- `KeplerPlan::new(eccentricity: f64) -> Result<KeplerPlan, KeplerError>` validates the eccentricity alone (`NonFiniteInput`, `InvalidEccentricity`, `UnsupportedDomain` above the service endpoint, the same thresholds as `solve_kepler`) and folds the schedule band (both correction schedules and the construction cap), the e-powers of the series starter, the exact Knuth two-sum split `1 - e = denom + denom_tail` and the e-dependent constant allowances (the conservative service error ceiling). Fields are private and construction is validated-only, so a caller cannot forge constants or bounds; a plan is not an authority for `d` or `r`, which are still constructed per candidate.
- `solve_with_plan(face: Sigma, mean_anomaly: f64, plan: &KeplerPlan) -> Result<KeplerSolution, KeplerError>` re-checks what it receives (`NonFiniteInput` for the mean anomaly, `UnsupportedFace`, `UnsupportedDomain` for `m` outside `[-pi, pi]`), then runs the same circular monad, arms, certificate and refusal semantics as `solve_kepler` against the shared plan. The plan outlives many solves; face and mean anomaly are per-solve checks by design.
- `KeplerPlan::denominator_split()` reads out the exact split with its documented slack (`|denom_tail| <= 2^-53 denom`, a pure readout that manufactures no bound).
- REQUIRED invariant (tested): for every `(face, m, e)`, `solve_with_plan(face, m, plan)` with a plan folded from `e` returns the bit-identical `KeplerSolution` (anomaly, defect and error bound patterns, rounds) or the same typed refusal as `solve_kepler(face, m, e)`. The combined refusal precedence stays `NonFiniteInput`, `UnsupportedFace`, `InvalidEccentricity`, `UnsupportedDomain`, enforced by `solve_kepler` across its three arguments; the additive entries check the arguments each receives.
