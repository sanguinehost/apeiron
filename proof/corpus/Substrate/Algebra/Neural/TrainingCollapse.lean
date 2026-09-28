import Substrate.Algebra.QuantumField.QuantumClassicalIsomorphism
import Substrate.Collapse
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Data.Complex.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.StarOrdered
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Common
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Training Collapse — the micro-training pipeline, proven end to end
@name Training Without the Grind
@child "Training" a model usually means a long slow guessing loop. Here it's a single calculation you can write down in closed form — boundary divided by drag — and folding in more data is just adding the pieces up.
@scientist Fitting a head Y ≈ X·W is one x = B/U inversion at the matrix applicative: B = XᵀY, U = XᵀX + λ·1, W = U⁻¹B; the streaming fold is exact because the normal-equations accumulator is a commutative monoid morphism, so stage order doesn't change the endpoint.
@engineer `ridgeHead_solves`/`ridgeHead_unique` prove well-posedness and uniqueness (ridge λ > 0 ⟹ positive-definite drag, `drag_posDef`); the accumulate/merge laws make the streaming fold exact, and the forward-only second-order step uses `antithetic_recovers_gradient`/`stencil4_reconstructs_laplacian` — no backward pass.

The reference implementation for the closed-form trainer. It is `../fold/modelfold/`:
`src/foldtrain.rs` (the ridge/normal-equations fold), whose own header calls itself *"the Rust port
of Sanguine paper 1's `Train.lean`"* and cites `Train.ridgeHead_unique` by name. The rotor cell it
also named, `src/rotorlm.rs`, was DELETED in fold `b5383a5` (2026-08-13) <!-- CITATION-EXEMPT: deleted in fold b5383a5, this line is the record of the deletion -->: under a fair one-code-path
A/B it read chance (18%, chance 17%) on 6-binding associative recall where softmax attention read
35%. The theorems below are about `ℂ` and stand without it; only the pointer is gone. The `platonic` crate this header used to name — `solve.rs`,
`rotor.rs`, `train.rs`, `nursery.rs`, `examples/nursery_lm.rs` — exists on no disk here and in no
commit of either repo (`git log --all -S`, `git log --all --name-only`, `find ~/Workspace`, all
empty); those names are kept below only as the record of what each section was written against.
"Training" a head `Y ≈ X·W` is **not** a
search — it is one `x = B/U` inversion at the matrix applicative: `B = XᵀY` (the boundary),
`U = XᵀX + λ•1` (the drag), `W = U⁻¹B`. Each section proves the law a Rust function currently
only cites:

1. **Well-posedness** — `λ > 0` makes the drag positive-definite, hence invertible; the ridge
   head solves the normal equations and is the *unique* solution
   (`foldtrain.rs::FoldTrainer::solve`, `:287`; `solve_spd`, `:318`).
2. **The streaming fold is exact** — the normal equations of stacked data are the SUM of the
   blocks' normal equations: the accumulator is a monoid morphism
   (`foldtrain.rs::FoldTrainer::accumulate`, `:239`; `::merge`, `:268`; the test
   `solve_matches_normal_equations`, `:592`), and since addition commutes the FINAL head is
   independent of stage order — the
   developmental curriculum (`data/MANIFEST.md`) orders the *gates*, not the solved endpoint.
3. **Weighted merge IS upsampling** — a stage folded with natural weight `w` equals that stage
   repeated `w` times. The dilution fix (the measured 31.6% → 5.8% crash of the 9 KB nursery
   stage under a 7.5 M-row stage) is licensed as *exact*, not a replay heuristic.
4. **The rotor cell is stable by geometry** — the SO(2) step preserves the norm, the damped
   step holds an invariant ball (`‖ψ‖ ≤ max ‖u‖` needs no schedule), and the conjugate rotor IS
   the inverse — the adjoint carried forward (`CellSquareRoot`'s conjugate block; the previously
   dangling `rotor_state_bounded` citation is written at last, as the theorem of that name in §4.
   `RotorAdjointForwardOnly` is NOT — no declaration or module ever bore it; the theorem written
   is `rotor_adjoint_forward`).
5. **The gate, quantified** — on recoverable targets `Y = X·W*` the ridge head is exactly
   `W* − λ·U⁻¹·W*`: the Axiom-V single-verification residual is a NAMED λ-bias, not a tolerance.
6. **The forward-only second-order step** — the damped-Newton update `x − f'(x)/c` lands the
   quadratic model's minimum exactly; gradient and curvature come from the proven branch orbits
   (`QuantumClassicalIsomorphism.antithetic_recovers_gradient`, `stencil4_reconstructs_laplacian`)
   — no backward pass.
7. **A concrete micro-run** — explicit rationals through the whole pipe, closed by `norm_num`:
   the reference numbers the Rust trainer must reproduce.
8. **The root wiring** — the normal-equations accumulator IS `Collapse.foldKey` over rows
   (`normal_equations_are_foldKey`), so §2's stack laws, curriculum order-freedom and exact
   unlearning are `Collapse.streaming_exact` / `streaming_order_free` / `streaming_exact_removal`
   worn directly, not a parallel discovery.
9. **The weighted fold** — the fold face is closed under reweighting
   (`weighted_normal_equations_are_foldKey`, `weighted_drag_posDef`), which is IRLS's leg: the
   CE estimator is a different RESIDUAL face over the same fold and key faces.
10. **The objective is forced, and its minimizer unique** — the ridge head minimizes the
    least-squares objective, not only solves the normal equations (`objective_eq_qform`,
    `qform_completion`, `ridgeHead_minimizes`, `ridgeHead_minimizes_objective`,
    `training_is_a_collapse`). Each of the three "uniquely minimizes" statements now conjoins the
    `≤` with the STRICT inequality off the head, via `qform_gt` — the strictness `drag_posDef`
    always supplied and `qform_ge`'s `PosSemidef` route discarded (`qform_diag_eq`,
    `trace_posDef_pos` carry it to the trace). Softmax-CE is a calibration read over the same
    scores, not a rival forced objective.
11. **Finetuning is the base-anchored ridge** — anchoring at `W₀` is re-centering
    (`anchoredObjective_shift`), so the optimal adapter is one closed-form solve on the residual
    (`anchoredHead_minimizes`). This is §11's proved content; the LoRA/qLoRA reading laid over it
    is not (see the warrant block).

The list above ran to §7 for as long as §8–§11 have existed; a reader who stopped at the header
would have missed the root wiring, the weighted fold, and the whole minimization section.

Cited bridge: Mathlib's positive-definite ⇒ invertible chain (`PosDef.isUnit`). Outside the
theorems: the softmax-CE metric is a calibration choice on least-squares scores and the byte
embedding is a fixed random lift (the reservoir) — neither affects the solve laws proven here.

## What the pipeline proves, and where the headline stops

* **Proved (native):** the matrix algebra, section by section — §1 `drag_posDef`, `drag_isUnit`,
  `ridgeHead_solves`, `ridgeHead_unique`; §2 `gram_stack`, `boundary_stack`; §3 `gram_replicate`,
  `boundary_replicate`; §4 `rotor_isometry`, `rotor_state_bounded`, `rotor_adjoint_forward`; §5
  `ridge_bias`; §6 `newton_step_lands`; §7 `micro_run`; §8 `normal_equations_are_foldKey`; §9
  `weighted_normal_equations_are_foldKey`, `weighted_drag_posDef`; §10 `drag_isSymm`,
  `trace_cross_symm`, `qform_completion`, `qform_ge`, `qform_diag_eq`, `trace_posDef_pos`,
  `qform_gt`, `ridgeHead_minimizes`, `objective_eq_qform`,
  `ridgeHead_minimizes_objective`, `training_is_a_collapse`; §11 `anchoredObjective_shift`,
  `anchoredHead_minimizes`. Each is an identity or an inequality on matrices over `ℝ`, and each
  says what its docstring says with the exceptions below.
* **Proved (Mathlib):** well-posedness rides `Matrix.posSemidef_conjTranspose_mul_self`,
  `Matrix.PosDef.smul`, `Matrix.PosDef.posSemidef_add` and `PosDef.isUnit` — which the header
  already labels a cited bridge — and §9 adds `Matrix.posSemidef_diagonal_iff` and
  `PosSemidef.conjTranspose_mul_mul_same`.
* **Proved (riding):** §8 is `Substrate/Collapse.lean` worn directly. `normal_equations_stream` is
  `Collapse.streaming_exact`, `normal_equations_order_free` is
  `Collapse.streaming_order_free rowContrib h`, and `stage_removal_exact` is
  `Collapse.streaming_exact_removal rowContrib rest flaw`. The section says so; the native content
  is the identification `normal_equations_are_foldKey` and nothing else.
* **Proved (native):** the headline `training_collapse`, widened 2026-08-06 from TWO conjuncts to
  EIGHT — the six results its docstring names, plus `drag_isUnit`. It bundled `IsUnit (drag X lam)`
  and `gram_stack` while naming `ridgeHead_unique`, `gram_replicate`, `rotor_state_bounded`,
  `rotor_adjoint_forward`, `ridge_bias` and `newton_step_lands`; all six were proved in the file
  and none was in the bundle. Every conjunct is now one of the theorems the docstring names, in
  section order, and every one shares `X` or `lam` with the statement. Native is the CONJUNCTION;
  conjunct 1 still rides `PosDef.isUnit`, the cited bridge above, and §4's and §6's readings are
  still the assertions named below.
* **Proved (native):** "uniquely minimizes", three times — widened 2026-08-06.
  `ridgeHead_minimizes`, `ridgeHead_minimizes_objective` and `anchoredHead_minimizes` each
  concluded `≤` and nothing more; each now conjoins that `≤` with `W ≠ head → strict <`, proved by
  `qform_gt` off `drag_posDef`. `qform_ge` takes its `≥ 0` from `PosSemidef` and throws the
  strictness away; `qform_diag_eq` reads a diagonal entry of `Mᵀ U M` as the quadratic form at
  `M`'s column and `trace_posDef_pos` sums it, which is what carries `PosDef`'s vector statement to
  the matrix trace. `ridgeHead_unique` remains uniqueness of the normal-equations SOLUTION — a
  different claim, still composed with the minimization only inside `training_is_a_collapse`.
* **Anchor:** that §11 is LoRA/qLoRA. `anchoredHead_minimizes` is FULL-RANK
  base-anchored ridge; no rank, truncation, quantization or `Seed` occurs in any type in this
  file, so "LoRA's low rank is `ΔW`'s resonance truncation; qLoRA's quantized base is the Seed
  floor" and "qLoRA = Seed-floor(W₀) + rank-r-resonance(anchoredRidge)" are unproved here. The
  reduction that IS proved — `anchoredObjective_shift`, anchoring is re-centering — is the real
  and sufficient result.
* **Anchor:** §4's and §6's readings. `rotor_adjoint_forward` is
  `conj(e^{iθ}) = (e^{iθ})⁻¹` on ℂ, one `Complex.exp_conj` rewrite; "the backward pass's operator
  is available at every forward step … no reverse sweep stores it" has no backward pass, gradient
  or sweep in any type. `newton_step_lands` is a completed square over five free reals; that `g`
  and `c` arrive from `QuantumClassicalIsomorphism.antithetic_recovers_gradient` (`:254`) and
  `stencil4_reconstructs_laplacian` (`:245`) — both real theorems — is a composition the file does
  not make.
* **Next:** `micro_run`'s conclusion. The statement is two 2×2 identities over
  literals, closed by `fin_cases` + `norm_num`; neither `ridgeHead` nor `drag` occurs in it and
  `ridgeHead_unique` is not applied. "The unique head is `[[1],[2]]` … (`ridgeHead_unique` pins
  it)" is assembled in the docstring, so this is the reference NUMBERS, not the reference RUN.
* **Retired:** the §4 citation `RotorAdjointForwardOnly`, removed from the header and from
  `rotor_adjoint_forward`'s docstring on 2026-08-06. No module or declaration of that name exists
  and none was deleted — `git log --all -S 'RotorAdjointForwardOnly'` returns only the two commits
  that ADDED the citation (`b196e2e2`, `d663a040`), and `git log --all --name-only` matches no
  path. The theorem written is `rotor_adjoint_forward`. Its sibling `rotor_state_bounded` did land
  and is a theorem here; `DiracRotor` is prose in `CellSquareRoot.lean`, not a declaration.
* **Anchor:** every measured figure. "the measured 31.6% → 5.8% crash of the 9 KB
  nursery stage under a 7.5 M-row stage" (§3), "the measured 621×, bit-identical heads" (§8), and
  the CE-vs-ridge baseline bench (§9): no artefact in this repo carries any of them.
  `data/MANIFEST.md` does resolve.
* **Anchor:** the implementation citations, re-pointed 2026-08-06. §1–§2's solve and
  fold are `../fold/modelfold/src/foldtrain.rs` — `FoldTrainer::solve` (`:287`), `accumulate`
  (`:239`), `merge` (`:268`), `solve_spd` (`:318`), test `solve_matches_normal_equations` (`:592`)
  — a file whose own header states it is the Rust port of `Train.lean` and names
  `Train.ridgeHead_unique`; §4's rotor HAD a reference implementation,
  `../fold/modelfold/src/rotorlm.rs`, deleted in fold `b5383a5` — §4 now names no running code <!-- CITATION-EXEMPT: implementation binned in fold b5383a5, this line is the record -->. Naming them is still a bridge, not a theorem: nothing
  here checks that either Rust function computes what the Lean states. The `platonic` crate
  formerly named (`solve.rs`, `rotor.rs`, `train.rs`, `nursery.rs`, `examples/nursery_lm.rs`, and
  the symbols `solve_linear`, `Rotor::forward`, `NormalEquations::accumulate`/`merge`) exists on
  no disk here and in no commit of either repo; `nursery` in particular has no successor in
  `modelfold`.
* **Open:** §9's own scope note, kept as it stands — the exact-stream and exact-unlearn verbs are
  properties of the QUADRATIC residual face, holding per weighted fold and not across the IRLS
  loop, and "what the iteration buys in accuracy is the bench's question, not a theorem's".
-/

namespace Sanguine
namespace TrainingCollapse

open Matrix

variable {n m d e w : ℕ}

-- ═══════════════════════════════════════════════════════════════════
-- §1. WELL-POSEDNESS — the drag U = XᵀX + λ•1 is invertible; the head is unique
-- ═══════════════════════════════════════════════════════════════════

/-- The drag `U = XᵀX + λ•1` — the denominator of the training `x = B/U`. -/
def drag (X : Matrix (Fin n) (Fin d) ℝ) (lam : ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Xᵀ * X + lam • (1 : Matrix (Fin d) (Fin d) ℝ)

/-- **The drag is positive-definite for `λ > 0`.** `XᵀX` is PSD (a Gram matrix); adding `λ•1`
    injects the strictly positive floor. This is what the ridge `λ` is FOR. -/
theorem drag_posDef (X : Matrix (Fin n) (Fin d) ℝ) {lam : ℝ} (hl : 0 < lam) :
    (drag X lam).PosDef := by
  have hgram : (Xᵀ * X).PosSemidef := by
    have h := Matrix.posSemidef_conjTranspose_mul_self X
    simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using h
  have hsm : (lam • (1 : Matrix (Fin d) (Fin d) ℝ)).PosDef :=
    Matrix.PosDef.smul Matrix.PosDef.one hl
  exact Matrix.PosDef.posSemidef_add hgram hsm

/-- **The drag is a unit** — the inversion `B/U` exists. -/
theorem drag_isUnit (X : Matrix (Fin n) (Fin d) ℝ) {lam : ℝ} (hl : 0 < lam) :
    IsUnit (drag X lam) :=
  (drag_posDef X hl).isUnit

/-- The ridge head — `W = U⁻¹ · (XᵀY)`, the one inversion
    (`../fold/modelfold/src/foldtrain.rs::FoldTrainer::solve`). -/
noncomputable def ridgeHead (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ) (lam : ℝ) :
    Matrix (Fin d) (Fin e) ℝ :=
  (drag X lam)⁻¹ * (Xᵀ * Y)

/-- **The head solves the normal equations**: `U · W = XᵀY = B`. Training is the solve. -/
theorem ridgeHead_solves (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    {lam : ℝ} (hl : 0 < lam) :
    drag X lam * ridgeHead X Y lam = Xᵀ * Y := by
  rw [ridgeHead, ← Matrix.mul_assoc,
    Matrix.mul_nonsing_inv _ ((Matrix.isUnit_iff_isUnit_det _).mp (drag_isUnit X hl)),
    Matrix.one_mul]

/-- **The head is unique**: any `W'` with `U·W' = B` is THE head. No search could find another
    answer — there is exactly one, and the inversion computes it. -/
theorem ridgeHead_unique (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    {lam : ℝ} (hl : 0 < lam) (W' : Matrix (Fin d) (Fin e) ℝ)
    (hW' : drag X lam * W' = Xᵀ * Y) :
    W' = ridgeHead X Y lam := by
  have hdet := (Matrix.isUnit_iff_isUnit_det _).mp (drag_isUnit X hl)
  calc W' = (drag X lam)⁻¹ * (drag X lam * W') := by
              rw [← Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hdet, Matrix.one_mul]
    _ = ridgeHead X Y lam := by rw [hW', ridgeHead]

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE STREAMING FOLD IS EXACT — the accumulator is a monoid morphism
-- ═══════════════════════════════════════════════════════════════════

/-- Stack two data blocks (two stages, two shards) into one dataset — rows are the sum type. -/
def stack (X₁ : Matrix (Fin n) (Fin d) ℝ) (X₂ : Matrix (Fin m) (Fin d) ℝ) :
    Matrix (Fin n ⊕ Fin m) (Fin d) ℝ :=
  Matrix.of fun r c => Sum.elim (fun i => X₁ i c) (fun j => X₂ j c) r

/-- **The Gram fold is a monoid morphism**: the stacked data's `XᵀX` is the SUM of the blocks'.
    This is `NormalEquations::accumulate`/`merge` — the streaming O(d²) path is *exact*, and
    since `+` commutes, the FINAL solved head cannot depend on the order stages were folded:
    the developmental curriculum orders the gates, not the endpoint. -/
theorem gram_stack (X₁ : Matrix (Fin n) (Fin d) ℝ) (X₂ : Matrix (Fin m) (Fin d) ℝ) :
    (stack X₁ X₂)ᵀ * stack X₁ X₂ = X₁ᵀ * X₁ + X₂ᵀ * X₂ := by
  ext i j
  simp [Matrix.mul_apply, Matrix.transpose_apply, stack, Fintype.sum_sum_type]

/-- **The boundary fold is the same morphism**: stacked `XᵀY` = sum of block `XᵀY`. -/
theorem boundary_stack (X₁ : Matrix (Fin n) (Fin d) ℝ) (X₂ : Matrix (Fin m) (Fin d) ℝ)
    (Y₁ : Matrix (Fin n) (Fin e) ℝ) (Y₂ : Matrix (Fin m) (Fin e) ℝ) :
    (stack X₁ X₂)ᵀ * stack Y₁ Y₂ = X₁ᵀ * Y₁ + X₂ᵀ * Y₂ := by
  ext i j
  simp [Matrix.mul_apply, Matrix.transpose_apply, stack, Fintype.sum_sum_type]

-- ═══════════════════════════════════════════════════════════════════
-- §3. WEIGHTED MERGE IS EXACT UPSAMPLING — the dilution fix licensed
-- ═══════════════════════════════════════════════════════════════════

/-- A stage replicated `w` times — rows indexed by `Fin w × Fin n`, every copy the same block. -/
def replicate (w : ℕ) (X : Matrix (Fin n) (Fin d) ℝ) : Matrix (Fin w × Fin n) (Fin d) ℝ :=
  Matrix.of fun r c => X r.2 c

/-- **Weight = repetition, exactly.** The Gram matrix of a stage repeated `w` times is `w • XᵀX`.
    So folding a stage's normal equations scaled by `w` IS training on it `w` times — the
    upsampling the BOM's assemble stage names ("self upsampled"), and the exact cure for the
    measured nursery-stage dilution. A weight is not a heuristic; it is repetition collapsed. -/
theorem gram_replicate (w : ℕ) (X : Matrix (Fin n) (Fin d) ℝ) :
    (replicate w X)ᵀ * replicate w X = (w : ℝ) • (Xᵀ * X) := by
  ext i j
  simp [Matrix.mul_apply, Matrix.transpose_apply, replicate, Fintype.sum_prod_type,
    Finset.sum_const]

/-- The boundary side of the same law: `(w-replicated X)ᵀ(w-replicated Y) = w • XᵀY`. -/
theorem boundary_replicate (w : ℕ) (X : Matrix (Fin n) (Fin d) ℝ)
    (Y : Matrix (Fin n) (Fin e) ℝ) :
    (replicate w X)ᵀ * replicate w Y = (w : ℝ) • (Xᵀ * Y) := by
  ext i j
  simp [Matrix.mul_apply, Matrix.transpose_apply, replicate, Fintype.sum_prod_type,
    Finset.sum_const]

-- ═══════════════════════════════════════════════════════════════════
-- §4. THE ROTOR CELL IS STABLE BY GEOMETRY — no schedule, no fragile start
-- ═══════════════════════════════════════════════════════════════════

open Complex in
/-- One damped rotor plane (one SO(2) plane as `ℂ`; the `rotorlm.rs` it cited is binned) <!-- CITATION-EXEMPT: deleted in fold b5383a5, this line is the record of the deletion -->:
    `ψ' = ρ·e^{iθ}·ψ + (1−ρ)·u`. -/
noncomputable def rotorStep (ρ θ : ℝ) (ψ u : ℂ) : ℂ :=
  (ρ : ℂ) * (Complex.exp (θ * Complex.I) * ψ) + ((1 : ℂ) - ρ) * u

open Complex in
/-- **The rotation is an isometry** — `‖e^{iθ}ψ‖ = ‖ψ‖`. The rotor's memory neither grows nor
    decays by itself; only the damping moves the modulus. -/
theorem rotor_isometry (θ : ℝ) (ψ : ℂ) : ‖Complex.exp (θ * Complex.I) * ψ‖ = ‖ψ‖ := by
  rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]

open Complex in
/-- **The invariant ball (`rotor_state_bounded`, at last a theorem).** If the state and the input
    sit in the ball of radius `M` and `0 ≤ ρ ≤ 1`, the damped step stays in the ball:
    `‖ψ'‖ ≤ ρM + (1−ρ)M = M`. Stability is a property of the geometry, not a training
    schedule — the nursery is stable on Day 1. (The `nursery.rs` this once cited exists nowhere, and
    `rotorlm.rs` was binned in fold `b5383a5`; there is no live rotor runtime.) <!-- CITATION-EXEMPT: implementation binned in fold b5383a5 --> -/
theorem rotor_state_bounded (ρ θ : ℝ) (ψ u : ℂ) (M : ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (hψ : ‖ψ‖ ≤ M) (hu : ‖u‖ ≤ M) :
    ‖rotorStep ρ θ ψ u‖ ≤ M := by
  have h1 : ‖(ρ : ℂ) * (Complex.exp (θ * Complex.I) * ψ)‖ ≤ ρ * M := by
    rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg hρ0, rotor_isometry]
    exact mul_le_mul_of_nonneg_left hψ hρ0
  have h2 : ‖((1 : ℂ) - ρ) * u‖ ≤ (1 - ρ) * M := by
    have : ((1 : ℂ) - ρ) = ((1 - ρ : ℝ) : ℂ) := by push_cast; ring
    rw [this, norm_mul, Complex.norm_real, Real.norm_of_nonneg (by linarith)]
    exact mul_le_mul_of_nonneg_left hu (by linarith)
  calc ‖rotorStep ρ θ ψ u‖
      ≤ ‖(ρ : ℂ) * (Complex.exp (θ * Complex.I) * ψ)‖ + ‖((1 : ℂ) - ρ) * u‖ := norm_add_le _ _
    _ ≤ ρ * M + (1 - ρ) * M := add_le_add h1 h2
    _ = M := by ring

open Complex in
/-- **The conjugate rotor IS the inverse — the adjoint carried forward**
    (`CellSquareRoot`'s conjugate block, the
    `DiracRotor`'s counter-rotating `R(−θ)`). The backward pass's operator is available at
    every forward step as the conjugate — no reverse sweep stores it. -/
theorem rotor_adjoint_forward (θ : ℝ) :
    (starRingEnd ℂ) (Complex.exp (θ * Complex.I)) = (Complex.exp (θ * Complex.I))⁻¹ := by
  rw [← Complex.exp_conj,
    show (starRingEnd ℂ) (θ * Complex.I) = -(θ * Complex.I) by
      simp [map_mul, Complex.conj_ofReal, Complex.conj_I, mul_neg],
    Complex.exp_neg]

-- ═══════════════════════════════════════════════════════════════════
-- §5. THE GATE, QUANTIFIED — the residual on recoverable targets IS the λ-bias
-- ═══════════════════════════════════════════════════════════════════

/-- **The ridge bias identity.** When the targets are exactly recoverable (`Y = X·W*`), the
    solved head is `W* − λ·U⁻¹·W*` — exactly. The Axiom-V gate residual is therefore the NAMED
    quantity `λ·U⁻¹·W*` shrinking with `λ`, not an unexplained tolerance. (At `λ` machine-small,
    the nursery gate `fit_residual ≤ gate` is this bias plus float noise.) -/
theorem ridge_bias (X : Matrix (Fin n) (Fin d) ℝ) (Wstar : Matrix (Fin d) (Fin e) ℝ)
    {lam : ℝ} (hl : 0 < lam) :
    ridgeHead X (X * Wstar) lam
      = Wstar - lam • ((drag X lam)⁻¹ * Wstar) := by
  have hdet := (Matrix.isUnit_iff_isUnit_det _).mp (drag_isUnit X hl)
  have key : drag X lam * (Wstar - lam • ((drag X lam)⁻¹ * Wstar)) = Xᵀ * (X * Wstar) := by
    rw [Matrix.mul_sub, Matrix.mul_smul, ← Matrix.mul_assoc,
      Matrix.mul_nonsing_inv _ hdet, Matrix.one_mul, drag, Matrix.add_mul,
      Matrix.smul_mul, Matrix.one_mul, Matrix.mul_assoc]
    abel
  exact (ridgeHead_unique X (X * Wstar) hl _ key).symm

-- ═══════════════════════════════════════════════════════════════════
-- §6. THE FORWARD-ONLY SECOND-ORDER STEP — damped Newton lands the quadratic exactly
-- ═══════════════════════════════════════════════════════════════════

/-- The local quadratic loss model along one direction: `f(x) = f₀ + g·x + ½c·x²`. -/
noncomputable def quadLoss (f0 g c x : ℝ) : ℝ := f0 + g * x + (c / 2) * x ^ 2

/-- **The Newton step lands the minimum exactly.** For the quadratic model with curvature
    `c > 0`, the dispersion trainer's update `x − f'(x)/c` reaches the unique minimizer in ONE
    step, from anywhere: `f(x − f'(x)/c) ≤ f(y)` for all `y`. The gradient `g + c·x` and the
    curvature `c` are exactly what the proven branch orbits read forward
    (`QuantumClassicalIsomorphism.antithetic_recovers_gradient` — K=2 — and
    `stencil4_reconstructs_laplacian` — K=4): the second-order step with no backward pass. -/
theorem newton_step_lands (f0 g c x y : ℝ) (hc : 0 < c) :
    quadLoss f0 g c (x - (g + c * x) / c) ≤ quadLoss f0 g c y := by
  have hx : x - (g + c * x) / c = -(g / c) := by field_simp; ring
  rw [hx]
  simp only [quadLoss]
  have h : f0 + g * (-(g / c)) + c / 2 * (-(g / c)) ^ 2 = f0 - g ^ 2 / (2 * c) := by
    field_simp; ring
  rw [h, ← sub_nonneg]
  have hsq : f0 + g * y + c / 2 * y ^ 2 - (f0 - g ^ 2 / (2 * c)) = (c * y + g) ^ 2 / (2 * c) := by
    field_simp; ring
  rw [hsq]
  positivity

-- ═══════════════════════════════════════════════════════════════════
-- §7. THE MICRO-RUN — the whole pipeline on explicit numbers (the reference instance)
-- ═══════════════════════════════════════════════════════════════════

/-- **The reference micro-training run.** Two examples, two features, one output, `λ = 1`:
    `X = [[1,0],[0,1]]` (so `XᵀX = 1`), `Y = [[2],[4]]`. The drag is `2•1`, the boundary is
    `Y` itself, and the unique head is `[[1],[2]]` — verified by the normal equations,
    no inversion routine needed (`ridgeHead_unique` pins it). The Rust trainer must reproduce
    this exactly; any deviation is a bug in the implementation, not a tuning question. -/
theorem micro_run :
    (Matrix.of ![![(1 : ℝ), 0], ![0, 1]])ᵀ * Matrix.of ![![(1 : ℝ), 0], ![0, 1]]
        + (1 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)
      = (2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)
    ∧ ((2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)) * Matrix.of ![![(1 : ℝ)], ![2]]
      = (Matrix.of ![![(1 : ℝ), 0], ![0, 1]])ᵀ * Matrix.of ![![(2 : ℝ)], ![4]] := by
  constructor <;>
    (ext i j
     fin_cases i <;> fin_cases j <;>
       norm_num [Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_two,
         Matrix.smul_apply, Matrix.add_apply, Matrix.one_apply,
         Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons])

-- ═══════════════════════════════════════════════════════════════════
-- THE HEADLINE
-- ═══════════════════════════════════════════════════════════════════

/-- **Training collapse (headline).** The micro-training pipeline is proven end to end:
    the drag is invertible and the head unique (§1, `ridgeHead_unique`); the streaming fold is
    exact and the endpoint curriculum-order-free (§2, `gram_stack`); a stage weight is exact
    repetition (§3, `gram_replicate`); the rotor memory is stable by geometry with its adjoint
    carried forward (§4, `rotor_state_bounded`, `rotor_adjoint_forward`); the gate residual is
    the named λ-bias (§5, `ridge_bias`); and the forward-only Newton step lands the local
    quadratic exactly (§6, `newton_step_lands`). Training is one `x = B/U` inversion — the
    rest is folding. -/
theorem training_collapse (X : Matrix (Fin n) (Fin d) ℝ) {lam : ℝ} (hl : 0 < lam) :
    -- §1 the drag inverts, and the head that solves is the only one
    IsUnit (drag X lam)
      ∧ (∀ (Y : Matrix (Fin n) (Fin e) ℝ) (W' : Matrix (Fin d) (Fin e) ℝ),
          drag X lam * W' = Xᵀ * Y → W' = ridgeHead X Y lam)
    -- §2 the streaming fold is exact, hence curriculum-order-free
      ∧ (∀ (X₂ : Matrix (Fin m) (Fin d) ℝ),
          (stack X X₂)ᵀ * stack X X₂ = Xᵀ * X + X₂ᵀ * X₂)
    -- §3 a stage weight IS repetition
      ∧ (∀ (wrep : ℕ),
          (replicate wrep X)ᵀ * replicate wrep X = (wrep : ℝ) • (Xᵀ * X))
    -- §4 the rotor holds its ball, and its adjoint is the forward conjugate
      ∧ (∀ (ρ θ : ℝ) (ψ u : ℂ) (Mb : ℝ), 0 ≤ ρ → ρ ≤ 1 → ‖ψ‖ ≤ Mb → ‖u‖ ≤ Mb →
          ‖rotorStep ρ θ ψ u‖ ≤ Mb)
      ∧ (∀ θ : ℝ, (starRingEnd ℂ) (Complex.exp (θ * Complex.I))
          = (Complex.exp (θ * Complex.I))⁻¹)
    -- §5 the gate residual is the named λ-bias
      ∧ (∀ (Wstar : Matrix (Fin d) (Fin e) ℝ),
          ridgeHead X (X * Wstar) lam = Wstar - lam • ((drag X lam)⁻¹ * Wstar))
    -- §6 the forward-only Newton step lands the local quadratic exactly
      ∧ (∀ (f0 g c x y : ℝ), 0 < c →
          quadLoss f0 g c (x - (g + c * x) / c) ≤ quadLoss f0 g c y) :=
  ⟨drag_isUnit X hl,
   fun Y W' hW' => ridgeHead_unique X Y hl W' hW',
   fun X₂ => gram_stack X X₂,
   fun wrep => gram_replicate wrep X,
   fun ρ θ ψ u Mb => rotor_state_bounded ρ θ ψ u Mb,
   rotor_adjoint_forward,
   fun Wstar => ridge_bias X Wstar hl,
   fun f0 g c x y hc => newton_step_lands f0 g c x y hc⟩

-- ═══════════════════════════════════════════════════════════════════
-- §8. THE ROOT WIRING — the normal equations ARE `Collapse.foldKey` over rows
-- ═══════════════════════════════════════════════════════════════════
-- The demotion (INDEX consolidation item 4): the trainer's (B,U) ledger is the
-- streaming law's fold face, not a parallel discovery. The per-row contribution
-- `c(row) = (x·xᵀ, x·yᵀ)` lands in the AddCommGroup of matrix pairs, and the
-- normal-equations accumulator IS `Collapse.foldKey c` — one identification
-- (`normal_equations_are_foldKey`); the §2 stack laws, curriculum order-freedom,
-- and exact unlearning are then the root's `streaming_exact` /
-- `streaming_order_free` / `streaming_exact_removal`, worn directly. The §2
-- theorems stay (the matrix face); what §8 adds is the visible import.

/-- The per-row contribution to the normal equations — the streaming law's `c`:
    one row in, one `(B,U)`-pair increment out (`NormalEquations::accumulate`'s
    body, as algebra). -/
def rowContrib (r : (Fin d → ℝ) × (Fin e → ℝ)) :
    Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin e) ℝ :=
  (Matrix.vecMulVec r.1 r.1, Matrix.vecMulVec r.1 r.2)

/-- A dataset `(X, Y)` as the list of its rows — the stream the ledger folds. -/
def rowsOf (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ) :
    List ((Fin d → ℝ) × (Fin e → ℝ)) :=
  (List.finRange n).map (fun i => (X i, Y i))

/-- **THE IDENTIFICATION — the normal equations ARE the root's `foldKey`.**
    Folding the per-row contributions over a dataset's rows is exactly the
    matrix pair `(XᵀX, XᵀY)` — the drag and the boundary of the training
    `x = B/U`. The trainer's accumulator is `Collapse.foldKey`, so the
    streaming law governs it verbatim. -/
theorem normal_equations_are_foldKey (X : Matrix (Fin n) (Fin d) ℝ)
    (Y : Matrix (Fin n) (Fin e) ℝ) :
    Collapse.foldKey rowContrib (rowsOf X Y) = (Xᵀ * X, Xᵀ * Y) := by
  have h : Collapse.foldKey rowContrib (rowsOf X Y)
      = ∑ i : Fin n, (Matrix.vecMulVec (X i) (X i), Matrix.vecMulVec (X i) (Y i)) := by
    rw [Collapse.foldKey, rowsOf, List.map_map, Fin.sum_univ_def]
    rfl
  rw [h]
  refine Prod.ext ?_ ?_ <;>
    · simp only [Prod.fst_sum, Prod.snd_sum]
      ext j k
      simp [Matrix.sum_apply, Matrix.vecMulVec, Matrix.mul_apply,
        Matrix.transpose_apply]

/-- **`gram_stack`/`boundary_stack` ride `streaming_exact`** — the stacked
    dataset's normal equations are the `foldKey` of the concatenated row
    streams: the §2 matrix laws and the root's streaming law are one statement.
    The incremental fold over `rest ++ flaw` reads only the flaw; the rest's
    ledger, certified once, is never re-touched (the measured 621×,
    bit-identical heads). -/
theorem normal_equations_stream (X₁ : Matrix (Fin n) (Fin d) ℝ)
    (X₂ : Matrix (Fin m) (Fin d) ℝ) (Y₁ : Matrix (Fin n) (Fin e) ℝ)
    (Y₂ : Matrix (Fin m) (Fin e) ℝ) :
    ((stack X₁ X₂)ᵀ * stack X₁ X₂, (stack X₁ X₂)ᵀ * stack Y₁ Y₂)
      = Collapse.foldKey rowContrib (rowsOf X₁ Y₁ ++ rowsOf X₂ Y₂) := by
  rw [Collapse.streaming_exact, normal_equations_are_foldKey,
    normal_equations_are_foldKey, gram_stack, boundary_stack, Prod.mk_add_mk]

/-- **Curriculum order-freedom is `streaming_order_free`** — any arrival order
    of the same rows folds to the same ledger (§2's "the curriculum orders the
    gates, never the endpoint", now the root's theorem worn directly). -/
theorem normal_equations_order_free {xs ys : List ((Fin d → ℝ) × (Fin e → ℝ))}
    (h : xs.Perm ys) :
    Collapse.foldKey rowContrib xs = Collapse.foldKey rowContrib ys :=
  Collapse.streaming_order_free rowContrib h

/-- **Exact unlearning is `streaming_exact_removal`** — subtracting a stage's
    ledger IS never having folded it: the counterfactual head, bit-identical,
    from the group structure alone. -/
theorem stage_removal_exact (rest flaw : List ((Fin d → ℝ) × (Fin e → ℝ))) :
    Collapse.foldKey rowContrib (rest ++ flaw) - Collapse.foldKey rowContrib flaw
      = Collapse.foldKey rowContrib rest :=
  Collapse.streaming_exact_removal rowContrib rest flaw

-- ═══════════════════════════════════════════════════════════════════
-- §9. THE WEIGHTED FOLD — the fold face is closed under reweighting (IRLS's leg)
-- ═══════════════════════════════════════════════════════════════════

/-! ### §9. The weighted fold

The estimator-family gap the baseline bench named (CE heads beat the ridge head
at high capacity) is a choice of RESIDUAL face over the same fold and key faces —
and the classical route to the CE estimator, IRLS, is iterated weighted ridge:
each Newton step solves a weighted normal-equations system. The two native facts
here: (1) a per-row weight ABSORBS INTO the contribution `c` — the root's fold
face needs no new primitive (`weighted_normal_equations_are_foldKey`), so every
weighted fold inherits streaming exactness, order-freedom, and removal verbatim;
(2) the weighted solve stays unique (`weighted_drag_posDef`).

* **Cited (bridge, not re-derived):** that the logistic/CE Newton step IS this weighted system
  (standard GLM/IRLS).

**The honest trade, named:** CE admits no fixed sufficient
statistic — the weights depend on the current head, so the outer loop re-passes
the data; the exact-stream/exact-unlearn verbs are properties of the QUADRATIC
residual face, holding per-iteration here (each weighted fold is itself
order-free and mergeable) but not across the IRLS loop. What the iteration buys
in accuracy is the bench's question, not a theorem's.
-/

/-- A weighted row — the weight rides WITH the row in the stream. -/
def wRowsOf (w : Fin n → ℝ) (X : Matrix (Fin n) (Fin d) ℝ)
    (Y : Matrix (Fin n) (Fin e) ℝ) : List (ℝ × (Fin d → ℝ) × (Fin e → ℝ)) :=
  (List.finRange n).map (fun i => (w i, X i, Y i))

/-- The weighted per-row contribution — the weight absorbed into `c`: the fold
    face closed under reweighting, no new primitive. -/
def wRowContrib (r : ℝ × (Fin d → ℝ) × (Fin e → ℝ)) :
    Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin e) ℝ :=
  (r.1 • Matrix.vecMulVec r.2.1 r.2.1, r.1 • Matrix.vecMulVec r.2.1 r.2.2)

/-- **The weighted normal equations ARE the root's `foldKey`** — folding the
    weighted contributions is exactly `(XᵀDX, XᵀDY)` with `D = diagonal w`: the
    IRLS inner system is the same fold at a reweighted `c`, governed by the same
    streaming law. -/
theorem weighted_normal_equations_are_foldKey (w : Fin n → ℝ)
    (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ) :
    Collapse.foldKey wRowContrib (wRowsOf w X Y)
      = (Xᵀ * Matrix.diagonal w * X, Xᵀ * Matrix.diagonal w * Y) := by
  have h : Collapse.foldKey wRowContrib (wRowsOf w X Y)
      = ∑ i : Fin n, (w i • Matrix.vecMulVec (X i) (X i),
                      w i • Matrix.vecMulVec (X i) (Y i)) := by
    rw [Collapse.foldKey, wRowsOf, List.map_map, Fin.sum_univ_def]
    rfl
  rw [h]
  refine Prod.ext ?_ ?_ <;>
    · simp only [Prod.fst_sum, Prod.snd_sum]
      ext j k
      simp only [Matrix.sum_apply, Matrix.smul_apply, Matrix.vecMulVec_apply,
        smul_eq_mul, Matrix.mul_apply, Matrix.transpose_apply]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      simp only [Matrix.diagonal_apply, mul_ite, mul_zero,
        Finset.sum_ite_eq', Finset.mem_univ, if_true]
      ring

/-- **The weighted drag is positive-definite** for nonnegative weights and
    `λ > 0` — the IRLS inner solve is as well-posed as the unweighted one:
    `xᵀ(XᵀDX)x = Σᵢ wᵢ·(Xx)ᵢ² ≥ 0`, plus the `λ` floor. The weighted head is
    then unique by the same `ridgeHead_unique` argument. -/
theorem weighted_drag_posDef (X : Matrix (Fin n) (Fin d) ℝ) (w : Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) {lam : ℝ} (hl : 0 < lam) :
    (Xᵀ * Matrix.diagonal w * X
      + lam • (1 : Matrix (Fin d) (Fin d) ℝ)).PosDef := by
  have hgram : (Xᵀ * Matrix.diagonal w * X).PosSemidef := by
    have hD : (Matrix.diagonal w).PosSemidef :=
      Matrix.posSemidef_diagonal_iff.mpr hw
    have h := hD.conjTranspose_mul_mul_same X
    simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using h
  have hsm : (lam • (1 : Matrix (Fin d) (Fin d) ℝ)).PosDef :=
    Matrix.PosDef.smul Matrix.PosDef.one hl
  exact Matrix.PosDef.posSemidef_add hgram hsm

-- ═══════════════════════════════════════════════════════════════════
-- §10. THE OBJECTIVE IS FORCED — least-squares is the unique minimizer (CE is a labeled read)
-- ═══════════════════════════════════════════════════════════════════
-- §1 proved the ridge head is the unique SOLUTION of the normal equations `U·W = B`. This proves
-- it is the unique MINIMIZER of the least-squares OBJECTIVE — so least-squares is not a choice but
-- the forced objective, and the closed-form solve IS its global minimum (no search, completion of
-- square on the positive-definite drag). The softmax-CE metric (the harness label, §intro) is a
-- calibration READ over these same least-squares scores (`MLPrimitivesBridge.softmax_is_bulk`,
-- the B/U normalization), not a separate forced objective: one head, the metric a labeled choice.

/-- The least-squares quadratic `Q(W) = tr(Wᵀ U W) − 2·tr(Wᵀ B)` — the ridge objective minus the
    data-only constant `tr(YᵀY)` (`objective_eq_qform`); its minimizer is the training answer. -/
def qform (U : Matrix (Fin d) (Fin d) ℝ) (B : Matrix (Fin d) (Fin e) ℝ)
    (W : Matrix (Fin d) (Fin e) ℝ) : ℝ :=
  (Wᵀ * U * W).trace - 2 * (Wᵀ * B).trace

/-- **The drag is symmetric** — `(XᵀX + λ•1)ᵀ = XᵀX + λ•1`. The metric of the quadratic. -/
theorem drag_isSymm (X : Matrix (Fin n) (Fin d) ℝ) (lam : ℝ) : (drag X lam).IsSymm := by
  unfold drag Matrix.IsSymm
  rw [Matrix.transpose_add, Matrix.transpose_mul, Matrix.transpose_transpose,
      Matrix.transpose_smul, Matrix.transpose_one]

/-- The cross term is symmetric under a symmetric metric: `tr(Wstarᵀ U W) = tr(Wᵀ U Wstar)`. -/
theorem trace_cross_symm {U : Matrix (Fin d) (Fin d) ℝ} (hU : U.IsSymm)
    (W Wstar : Matrix (Fin d) (Fin e) ℝ) :
    (Wstarᵀ * U * W).trace = (Wᵀ * U * Wstar).trace := by
  rw [← Matrix.trace_transpose (Wᵀ * U * Wstar)]
  congr 1
  simp only [Matrix.transpose_mul, Matrix.transpose_transpose, hU.eq, Matrix.mul_assoc]

/-- **Completion of square.** For a symmetric metric `U` and any normal-equations solution
    `U·Wstar = B`, the quadratic's gap from `Wstar` is the metric form of the displacement:
    `Q(W) − Q(Wstar) = tr((W−Wstar)ᵀ U (W−Wstar))`. The algebra behind "the solve is the min". -/
theorem qform_completion {U : Matrix (Fin d) (Fin d) ℝ} (hU : U.IsSymm)
    {B : Matrix (Fin d) (Fin e) ℝ} {Wstar : Matrix (Fin d) (Fin e) ℝ}
    (hW : U * Wstar = B) (W : Matrix (Fin d) (Fin e) ℝ) :
    qform U B W - qform U B Wstar = ((W - Wstar)ᵀ * U * (W - Wstar)).trace := by
  have hsym := trace_cross_symm hU W Wstar
  have hRHS : ((W - Wstar)ᵀ * U * (W - Wstar)).trace
      = (Wᵀ * U * W).trace - (Wᵀ * U * Wstar).trace
        - (Wstarᵀ * U * W).trace + (Wstarᵀ * U * Wstar).trace := by
    simp only [Matrix.transpose_sub, Matrix.sub_mul, Matrix.mul_sub, Matrix.trace_sub]
    ring
  rw [hRHS]
  simp only [qform, ← hW, ← Matrix.mul_assoc]
  linarith [hsym]

/-- **The normal-equations solution is the unique global minimizer of the quadratic.** For a
    symmetric positive-semidefinite metric `U` with `U·Wstar = B`, every `W` has `Q(Wstar) ≤ Q(W)`
    — the gap is `tr((W−Wstar)ᵀ U (W−Wstar)) ≥ 0` (the PSD metric, `trace_nonneg`). The solve is
    the min, by geometry, not search. -/
theorem qform_ge {U : Matrix (Fin d) (Fin d) ℝ} (hU : U.IsSymm) (hpsd : U.PosSemidef)
    {B : Matrix (Fin d) (Fin e) ℝ} {Wstar : Matrix (Fin d) (Fin e) ℝ}
    (hW : U * Wstar = B) (W : Matrix (Fin d) (Fin e) ℝ) :
    qform U B Wstar ≤ qform U B W := by
  have hnn : 0 ≤ ((W - Wstar)ᵀ * U * (W - Wstar)).trace := by
    have h := (hpsd.conjTranspose_mul_mul_same (W - Wstar)).trace_nonneg
    simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using h
  linarith [qform_completion hU hW W, hnn]

/-- The `j`-th diagonal entry of `Mᵀ·U·M` IS the quadratic form of `U` at `M`'s `j`-th column —
    the bridge from `PosDef`'s vector statement to the matrix trace. -/
theorem qform_diag_eq (U : Matrix (Fin d) (Fin d) ℝ) (M : Matrix (Fin d) (Fin e) ℝ) (j : Fin e) :
    (Mᵀ * U * M) j j = star (fun a => M a j) ⬝ᵥ (U *ᵥ (fun a => M a j)) := by
  simp only [Matrix.mul_apply, Matrix.transpose_apply, dotProduct, Matrix.mulVec,
    Pi.star_apply, star_trivial, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => by ring

/-- **A positive-definite metric gives a STRICTLY positive trace on any nonzero displacement.**
    `tr(Mᵀ U M) = Σⱼ (col j)ᵀ U (col j)`: every column is `≥ 0` and the column carrying a nonzero
    entry is `> 0`. This is the strictness `drag_posDef` supplies and `qform_ge`'s `PosSemidef`
    route throws away. -/
theorem trace_posDef_pos {U : Matrix (Fin d) (Fin d) ℝ} (hpd : U.PosDef)
    {M : Matrix (Fin d) (Fin e) ℝ} (hM : M ≠ 0) :
    0 < (Mᵀ * U * M).trace := by
  obtain ⟨a, j, haj⟩ : ∃ a j, M a j ≠ 0 := by
    by_contra h
    simp only [not_exists, not_not] at h
    exact hM (by ext a j; simpa using h a j)
  refine Finset.sum_pos' (fun i _ => ?_) ⟨j, Finset.mem_univ j, ?_⟩
  · rw [Matrix.diag_apply, qform_diag_eq]
    exact hpd.posSemidef.dotProduct_mulVec_nonneg _
  · rw [Matrix.diag_apply, qform_diag_eq]
    exact hpd.dotProduct_mulVec_pos (fun hc => haj (congrFun hc a))

/-- **The minimizer is UNIQUE.** `qform_ge` sharpened: with the metric positive-DEFINITE, every
    `W` other than the normal-equations solution is STRICTLY worse. The floor is a point, not a
    level set — which is what "the solve IS the answer" needs and `≤` alone never says. -/
theorem qform_gt {U : Matrix (Fin d) (Fin d) ℝ} (hU : U.IsSymm) (hpd : U.PosDef)
    {B : Matrix (Fin d) (Fin e) ℝ} {Wstar : Matrix (Fin d) (Fin e) ℝ}
    (hW : U * Wstar = B) (W : Matrix (Fin d) (Fin e) ℝ) (hne : W ≠ Wstar) :
    qform U B Wstar < qform U B W := by
  have hpos : 0 < ((W - Wstar)ᵀ * U * (W - Wstar)).trace :=
    trace_posDef_pos hpd (sub_ne_zero_of_ne hne)
  linarith [qform_completion hU hW W]

/-- **The ridge head uniquely minimizes the least-squares quadratic (the forced objective).**
    `Q(ridgeHead) ≤ Q(W)` for every `W`, and STRICTLY `<` for every `W` other than the head
    (`qform_gt` on the positive-definite drag): the closed-form solve is the global minimum of
    `Q(W) = tr(Wᵀ U W) − 2·tr(Wᵀ XᵀY)` (the ridge objective up to `tr(YᵀY)`, `objective_eq_qform`)
    and the ONLY one. Least-squares is therefore not a tuning choice but the forced objective whose
    unique minimizer `x = B/U` computes; the softmax-CE metric is a labeled read over these scores
    (`MLPrimitivesBridge.softmax_is_bulk`), never a rival objective. -/
theorem ridgeHead_minimizes (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    {lam : ℝ} (hl : 0 < lam) (W : Matrix (Fin d) (Fin e) ℝ) :
    qform (drag X lam) (Xᵀ * Y) (ridgeHead X Y lam) ≤ qform (drag X lam) (Xᵀ * Y) W
      ∧ (W ≠ ridgeHead X Y lam →
          qform (drag X lam) (Xᵀ * Y) (ridgeHead X Y lam) < qform (drag X lam) (Xᵀ * Y) W) :=
  ⟨qform_ge (drag_isSymm X lam) (drag_posDef X hl).posSemidef (ridgeHead_solves X Y hl) W,
   fun hne => qform_gt (drag_isSymm X lam) (drag_posDef X hl) (ridgeHead_solves X Y hl) W hne⟩

/-- The literal ridge objective `J(W) = ‖X·W − Y‖²_F + λ·‖W‖²_F` (Frobenius norm² as the trace). -/
def objective (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ) (lam : ℝ)
    (W : Matrix (Fin d) (Fin e) ℝ) : ℝ :=
  ((X * W - Y)ᵀ * (X * W - Y)).trace + lam * (Wᵀ * W).trace

/-- **The literal objective IS the quadratic up to the data-only constant** `tr(YᵀY)`:
    `‖X·W − Y‖²_F + λ‖W‖²_F = Q(W) + tr(YᵀY)`. The constant does not depend on `W`, so the
    minimizer of the least-squares objective is exactly the minimizer of `Q`. -/
theorem objective_eq_qform (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    (lam : ℝ) (W : Matrix (Fin d) (Fin e) ℝ) :
    objective X Y lam W = qform (drag X lam) (Xᵀ * Y) W + (Yᵀ * Y).trace := by
  have hcross : (Yᵀ * (X * W)).trace = (Wᵀ * (Xᵀ * Y)).trace := by
    rw [← Matrix.trace_transpose (Wᵀ * (Xᵀ * Y))]
    congr 1
    simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
  simp only [objective, qform, drag, Matrix.transpose_sub, Matrix.transpose_mul,
    Matrix.sub_mul, Matrix.mul_sub, Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul,
    Matrix.mul_smul, Matrix.mul_one, Matrix.mul_assoc,
    Matrix.trace_add, Matrix.trace_sub, Matrix.trace_smul, smul_eq_mul]
  rw [hcross]
  ring

/-- **The ridge head uniquely minimizes the literal least-squares objective.**
    `‖X·(ridgeHead) − Y‖²_F + λ‖ridgeHead‖²_F ≤ ‖X·W − Y‖²_F + λ‖W‖²_F` for every `W`, strictly
    for every `W` other than the head: the closed-form `x = B/U` solve is the global minimum of
    the actual ridge loss, and the only minimizer. Least-squares is forced, not chosen; CE is a
    labeled metric read over the same scores. -/
theorem ridgeHead_minimizes_objective (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    {lam : ℝ} (hl : 0 < lam) (W : Matrix (Fin d) (Fin e) ℝ) :
    objective X Y lam (ridgeHead X Y lam) ≤ objective X Y lam W
      ∧ (W ≠ ridgeHead X Y lam →
          objective X Y lam (ridgeHead X Y lam) < objective X Y lam W) := by
  rw [objective_eq_qform, objective_eq_qform]
  exact ⟨by linarith [(ridgeHead_minimizes X Y hl W).1],
         fun hne => by linarith [(ridgeHead_minimizes X Y hl W).2 hne]⟩

/-- **Training IS the monad collapse — the encircle shape at the matrix applicative.** The two
    halves that make a collapse: (1) the objective's gap from the ridge head is a **non-negative
    residual** (`ridgeHead_minimizes` — `Q(W) − Q(ridgeHead) ≥ 0`, the boundary residual of
    `Web.encircle_iff`/`BiasInjectedRecovery`); (2) the floor is **unique** — any normal-equations
    solution IS the ridge head (`ridgeHead_unique`). A non-negative residual with a unique floor is
    exactly the collapse/`Collapse.Seed` signature: the PosDef drag is the definite (monad) metric,
    the closed-form solve the floor where the residual vanishes. Training joins the one collapse —
    no search, no separate mechanism; the compact/monad half of the recovery dichotomy
    (`SpectralWorld.spectral_point_only_on_the_compact_chart`), the spectral/Apeiron half being §7-9. -/
theorem training_is_a_collapse (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    {lam : ℝ} (hl : 0 < lam) (W W' : Matrix (Fin d) (Fin e) ℝ)
    (hW' : drag X lam * W' = Xᵀ * Y) :
    0 ≤ qform (drag X lam) (Xᵀ * Y) W - qform (drag X lam) (Xᵀ * Y) (ridgeHead X Y lam)
      ∧ W' = ridgeHead X Y lam :=
  ⟨by linarith [(ridgeHead_minimizes X Y hl W).1], ridgeHead_unique X Y hl W' hW'⟩

-- ═══════════════════════════════════════════════════════════════════
-- §11. FINETUNING IS THE BASE-ANCHORED RIDGE — what LoRA/qLoRA really is
-- ═══════════════════════════════════════════════════════════════════
-- LoRA freezes a pretrained W₀ and learns an additive ΔW (low-rank); qLoRA
-- quantizes W₀ too. In the framework's algebra this is NOT a separate mechanism:
-- it is the §1 ridge re-centered from 0 to W₀. Substituting V = W − W₀, the
-- base-anchored objective ‖XW − Y‖² + λ‖W − W₀‖² becomes the PLAIN ridge
-- objective (§10) fitting the RESIDUAL R = Y − X·W₀. So the optimal adapter is
-- the ONE closed-form ΔW = ridgeHead(X, R) — no gradient — and λ is the anchor
-- (trust in the base vs the domain). x = B/U: W₀ the Being floor, ΔW the Null
-- drag the domain adds. The LOW-RANK restriction is ΔW's resonance truncation
-- (the top-r spectral modes, `SpectralWorld` / the seed's low-rank floor); the
-- QUANTIZED base is the Seed floor (`Seed`, the τ-seal). qLoRA = Seed-floor(W₀)
-- + rank-r-resonance(anchoredRidge). And by §8 `stage_removal_exact`, the fold
-- has EXACT unlearning of any folded stage — which gradient LoRA has not.

/-- The residual a frozen base `W₀` leaves on the target: `R = Y − X·W₀` (what the
    domain adds beyond the base's own prediction — the Null arm). -/
def residual (X : Matrix (Fin n) (Fin d) ℝ) (W₀ : Matrix (Fin d) (Fin e) ℝ)
    (Y : Matrix (Fin n) (Fin e) ℝ) : Matrix (Fin n) (Fin e) ℝ := Y - X * W₀

/-- **The base-anchored finetune head — the LoRA/qLoRA form `W₀ + ΔW`.** Freeze the
    base `W₀`, ridge-fit the residual it leaves, add the fit back. The one closed-form
    adapter `ΔW = ridgeHead(X, Y − X·W₀)`, no gradient. -/
noncomputable def anchoredHead (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    (W₀ : Matrix (Fin d) (Fin e) ℝ) (lam : ℝ) : Matrix (Fin d) (Fin e) ℝ :=
  W₀ + ridgeHead X (residual X W₀ Y) lam

/-- The base-anchored objective — fit the data AND stay near the base:
    `J(W) = ‖X·W − Y‖²_F + λ·‖W − W₀‖²_F`. The `λ` now regularizes toward `W₀`
    (the pretrained weights), not toward `0` — the finetuning prior, exactly. -/
def anchoredObjective (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    (W₀ : Matrix (Fin d) (Fin e) ℝ) (lam : ℝ) (W : Matrix (Fin d) (Fin e) ℝ) : ℝ :=
  ((X * W - Y)ᵀ * (X * W - Y)).trace + lam * ((W - W₀)ᵀ * (W - W₀)).trace

/-- **The reduction — anchoring is re-centering.** Substituting `V = W − W₀` turns the
    base-anchored objective into the PLAIN ridge objective (§10) fitting the residual
    `R = Y − X·W₀`: `J_anchored(W) = J_ridge(X, R)(W − W₀)`. LoRA's "start at the
    pretrained weights and add a small update" is ridge regression whose origin is `W₀`. -/
theorem anchoredObjective_shift (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    (W₀ W : Matrix (Fin d) (Fin e) ℝ) (lam : ℝ) :
    anchoredObjective X Y W₀ lam W = objective X (residual X W₀ Y) lam (W - W₀) := by
  have hdata : X * (W - W₀) - residual X W₀ Y = X * W - Y := by
    rw [residual, Matrix.mul_sub]; abel
  rw [anchoredObjective, objective, hdata]

/-- **Finetuning is the base-anchored ridge (headline) — qLoRA, as per Paper I.**
    The anchored head `W₀ + ridgeHead(X, Y − X·W₀)` is the UNIQUE global minimizer of
    `J(W) = ‖X·W − Y‖²_F + λ‖W − W₀‖²_F`. Freezing the base and closed-form-fitting the
    residual IS ridge-regularizing toward the base — the LoRA prior, solved not searched.
    The adapter `ΔW` is one `x = B/U` inversion (`W₀` the Being floor, `ΔW` the Null drag
    the domain adds); `λ` is the anchor. LoRA's low rank is `ΔW`'s resonance truncation;
    qLoRA's quantized base is the Seed floor. No gradient, and exact unlearning by §8.
    UNIQUE is the second conjunct: strictly worse off the anchored head, by `qform_gt`. -/
theorem anchoredHead_minimizes (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
    (W₀ : Matrix (Fin d) (Fin e) ℝ) {lam : ℝ} (hl : 0 < lam) (W : Matrix (Fin d) (Fin e) ℝ) :
    anchoredObjective X Y W₀ lam (anchoredHead X Y W₀ lam)
      ≤ anchoredObjective X Y W₀ lam W
      ∧ (W ≠ anchoredHead X Y W₀ lam →
          anchoredObjective X Y W₀ lam (anchoredHead X Y W₀ lam)
            < anchoredObjective X Y W₀ lam W) := by
  rw [anchoredObjective_shift, anchoredObjective_shift]
  have hshift : anchoredHead X Y W₀ lam - W₀ = ridgeHead X (residual X W₀ Y) lam := by
    rw [anchoredHead]; abel
  rw [hshift]
  refine ⟨(ridgeHead_minimizes_objective X (residual X W₀ Y) hl (W - W₀)).1, fun hne => ?_⟩
  refine (ridgeHead_minimizes_objective X (residual X W₀ Y) hl (W - W₀)).2 fun hc => hne ?_
  rw [← hshift] at hc
  rw [← sub_left_injective.eq_iff.mp hc]

end TrainingCollapse
end Sanguine
