import Substrate.Algebra.Bulk.UniversalEquation
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Common
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# The Madelung Phase Cell — the Quantum Potential has the form `x = B/U` at `B=∇²R, U=R`
@name Where Quantum Physics Is Our Rule
@child A quantum wave has a size and a twist. Differentiate it and — exactly — out falls our one rule, boundary over drag, with the wave's curvature as the boundary and the wave itself as the drag. The physics isn't bolted on; it's the same equation.
@scientist Differentiating the phase cell ψ = R·e^{iφ} twice splits, channel by channel, into the two Madelung equations, and the quantum potential that appears IS the bulk ratio x = B/U with B = ∇²R (the amplitude's curvature) and U = R (the amplitude itself).
@engineer `second_deriv_cell`: ψ'' = cellCurv · e^{iφ} via a genuine `HasDerivAt` chain; the amplitude channel gives the quantum-potential term ∇²R/R = x at B = ∇²R, U = R. One phase = one rotor plane = one cell.

The rotor moves, and the bulk operator is read off its curvature. The second
derivative of the **phase cell**

  `ψ(y) = R(y) · exp(i·φ(y))`        (`R ≥ 0` amplitude, `φ` phase)

splits, channel by channel, into the **two Madelung equations**, and the quantum
potential that appears is the bulk ratio `x = B/U` with `B = ∇²R` (the amplitude's
curvature) and `U = R` (the amplitude itself).

## The decomposition (1D — one phase = one rotor plane = one cell)

`second_deriv_cell` differentiates `ψ = R·e^{iφ}` twice (a genuine `HasDerivAt`
chain) to `ψ'' = cellCurv · e^{iφ}` with

  `cellCurv = (R'' − R·(φ')²)  +  i·(2R'φ' + R·φ'')`.

* **Real channel ÷ R** (`quantum_potential_is_bulk_ratio`):
  `Re(cellCurv)/R = R''/R − (φ')²`, the **Quantum Hamilton–Jacobi** equation —
  kinetic term `(φ')²` plus quantum potential `R''/R`. The quantum potential has
  the form `x = B/U` under the *assignment* `B := ∇²R`, `U := R`: a chosen
  labeling, **not forced** by a theorem (any ratio matches `B/U`, which is just
  division). The proof is the division `(R'' − R(φ')²)/R = R''/R − (φ')²`.

* **Imaginary channel × R** (`continuity_is_current_divergence`):
  `R·Im(cellCurv) = (R²·φ')'`, the **continuity equation** — conservation of the
  Born density `ρ = R²` carried by the phase-gradient current `v = φ'`.

One differentiation of the cell, split real/imaginary, is the pair (Quantum
Hamilton–Jacobi, continuity), with `x = B/U` in the quantum-potential slot.

## What is proved vs interpreted

* **Proved (native):** `second_deriv_cell` (`ψ'' = cellCurv·e^{iφ}`, full
  `HasDerivAt` calculus for twice-differentiable `R, φ`); the channel extractions
  `cellCurv_re`, `cellCurv_im`; `quantum_potential_is_bulk_ratio`;
  `continuity_is_current_divergence`.
* **Anchor:** that `ψ = R·e^{iφ}` is *the* state and the
  Schrödinger split is read off `ψ''` this way (the `−ħ²/2m`, `V` prefactors
  cited); that `ρ = |ψ|²` is the Born density. The algebra — quantum potential
  `= R''/R`, continuity `= (R²φ')'` — is native; the `B/U` name on the ratio is
  the assignment.

-/

noncomputable section

open Complex

namespace Sanguine
namespace MadelungCell

-- ═══════════════════════════════════════════════════════════════════
-- THE PHASE CELL ψ = R·e^{iφ} AND ITS FIRST DERIVATIVE
-- ═══════════════════════════════════════════════════════════════════

/-- **The Madelung phase cell** `ψ(y) = R(y)·exp(i·φ(y))` — one rotor plane.
    `R` is the amplitude (the drag `U`), `φ` the phase (the rotor). -/
def cell (R φ : ℝ → ℝ) : ℝ → ℂ :=
  fun y => (R y : ℂ) * Complex.exp (Complex.I * (φ y : ℂ))

/-- **The first derivative of the cell.** `ψ' = (R' + i·R·φ')·e^{iφ}` — amplitude
    growth plus phase-rotation drift. -/
def cellD1 (R φ dR dφ : ℝ → ℝ) : ℝ → ℂ :=
  fun y => ((dR y : ℂ) + Complex.I * ((R y : ℂ) * (dφ y : ℂ)))
            * Complex.exp (Complex.I * (φ y : ℂ))

/-- **The curvature coefficient** of the cell: `ψ'' = cellCurv·e^{iφ}`. Its real
    part carries Quantum Hamilton–Jacobi, its imaginary part carries continuity. -/
def cellCurv (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) : ℂ :=
  ((ddR y - R y * (dφ y) ^ 2 : ℝ) : ℂ)
    + Complex.I * ((2 * dR y * dφ y + R y * ddφ y : ℝ) : ℂ)

/-- The cell's first derivative: `HasDerivAt (cell R φ) (ψ' value) y`, for
    differentiable `R, φ`. Product rule with the chain rule through `exp`. -/
theorem hasDerivAt_cell (R φ dR dφ : ℝ → ℝ)
    (hR : ∀ y, HasDerivAt R (dR y) y) (hφ : ∀ y, HasDerivAt φ (dφ y) y) (y : ℝ) :
    HasDerivAt (cell R φ)
      (((dR y : ℂ) + Complex.I * ((R y : ℂ) * (dφ y : ℂ)))
        * Complex.exp (Complex.I * (φ y : ℂ))) y := by
  have hA : HasDerivAt (fun z => (R z : ℂ)) (dR y : ℂ) y := (hR y).ofReal_comp
  have hE : HasDerivAt (fun z => Complex.exp (Complex.I * (φ z : ℂ)))
      (Complex.exp (Complex.I * (φ y : ℂ)) * (Complex.I * (dφ y : ℂ))) y :=
    (((hφ y).ofReal_comp).const_mul Complex.I).cexp
  have h := hA.mul hE
  rw [show (((dR y : ℂ) + Complex.I * ((R y : ℂ) * (dφ y : ℂ)))
            * Complex.exp (Complex.I * (φ y : ℂ)))
        = (dR y : ℂ) * Complex.exp (Complex.I * (φ y : ℂ))
          + (R y : ℂ) * (Complex.exp (Complex.I * (φ y : ℂ)) * (Complex.I * (dφ y : ℂ)))
        from by ring]
  exact h

-- ═══════════════════════════════════════════════════════════════════
-- THE SECOND DERIVATIVE: ψ'' = cellCurv · e^{iφ}
-- ═══════════════════════════════════════════════════════════════════

/-- The cell's `ψ'` has derivative `cellCurv·e^{iφ}`. The rotor's `i² = −1` turns
    the phase-drift derivative into the real `−R(φ')²` quantum-potential term;
    `Complex.I_sq` is the one nontrivial step. -/
theorem hasDerivAt_cellD1 (R φ dR dφ ddR ddφ : ℝ → ℝ)
    (hR : ∀ y, HasDerivAt R (dR y) y) (hdR : ∀ y, HasDerivAt dR (ddR y) y)
    (hφ : ∀ y, HasDerivAt φ (dφ y) y) (hdφ : ∀ y, HasDerivAt dφ (ddφ y) y) (y : ℝ) :
    HasDerivAt (cellD1 R φ dR dφ)
      (cellCurv R dR ddR dφ ddφ y * Complex.exp (Complex.I * (φ y : ℂ))) y := by
  have hF : HasDerivAt (fun z => (dR z : ℂ) + Complex.I * ((R z : ℂ) * (dφ z : ℂ)))
      ((ddR y : ℂ) + Complex.I * ((dR y : ℂ) * (dφ y : ℂ) + (R y : ℂ) * (ddφ y : ℂ))) y :=
    ((hdR y).ofReal_comp).add
      ((((hR y).ofReal_comp).mul ((hdφ y).ofReal_comp)).const_mul Complex.I)
  have hE : HasDerivAt (fun z => Complex.exp (Complex.I * (φ z : ℂ)))
      (Complex.exp (Complex.I * (φ y : ℂ)) * (Complex.I * (dφ y : ℂ))) y :=
    (((hφ y).ofReal_comp).const_mul Complex.I).cexp
  have h := hF.mul hE
  rw [show cellCurv R dR ddR dφ ddφ y * Complex.exp (Complex.I * (φ y : ℂ))
        = ((ddR y : ℂ) + Complex.I * ((dR y : ℂ) * (dφ y : ℂ) + (R y : ℂ) * (ddφ y : ℂ)))
            * Complex.exp (Complex.I * (φ y : ℂ))
          + ((dR y : ℂ) + Complex.I * ((R y : ℂ) * (dφ y : ℂ)))
            * (Complex.exp (Complex.I * (φ y : ℂ)) * (Complex.I * (dφ y : ℂ)))
        from by
          simp only [cellCurv]
          push_cast
          linear_combination
            (-(R y : ℂ) * (dφ y : ℂ) ^ 2 * Complex.exp (Complex.I * (φ y : ℂ)))
              * Complex.I_sq]
  exact h

/-- **The cell's second derivative is the curvature coefficient times the phase.**
    `ψ'' = cellCurv·e^{iφ}`, for twice-differentiable `R, φ` — the Madelung cell's
    Laplacian in one phase dimension, genuine `deriv (deriv …)`. -/
theorem second_deriv_cell (R φ dR dφ ddR ddφ : ℝ → ℝ)
    (hR : ∀ y, HasDerivAt R (dR y) y) (hdR : ∀ y, HasDerivAt dR (ddR y) y)
    (hφ : ∀ y, HasDerivAt φ (dφ y) y) (hdφ : ∀ y, HasDerivAt dφ (ddφ y) y) (y : ℝ) :
    deriv (deriv (cell R φ)) y
      = cellCurv R dR ddR dφ ddφ y * Complex.exp (Complex.I * (φ y : ℂ)) := by
  have hderiv : deriv (cell R φ) = cellD1 R φ dR dφ :=
    funext fun z => (hasDerivAt_cell R φ dR dφ hR hφ z).deriv
  rw [hderiv]
  exact (hasDerivAt_cellD1 R φ dR dφ ddR ddφ hR hdR hφ hdφ y).deriv

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO CHANNELS
-- ═══════════════════════════════════════════════════════════════════

/-- The real channel of the curvature: `Re(ψ''/e^{iφ}) = R'' − R·(φ')²`. -/
theorem cellCurv_re (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) :
    (cellCurv R dR ddR dφ ddφ y).re = ddR y - R y * (dφ y) ^ 2 := by
  simp only [cellCurv, Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.I_re,
    Complex.I_im, Complex.ofReal_im]
  ring

/-- The imaginary channel of the curvature: `Im(ψ''/e^{iφ}) = 2R'φ' + R·φ''`. -/
theorem cellCurv_im (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) :
    (cellCurv R dR ddR dφ ddφ y).im = 2 * dR y * dφ y + R y * ddφ y := by
  simp only [cellCurv, Complex.add_im, Complex.ofReal_im, Complex.mul_im, Complex.I_re,
    Complex.I_im, Complex.ofReal_re]
  ring

-- ═══════════════════════════════════════════════════════════════════
-- REAL CHANNEL: QUANTUM HAMILTON–JACOBI, QUANTUM POTENTIAL = x = B/U
-- ═══════════════════════════════════════════════════════════════════

/-- **The amplitude bulk equation** `x = B/U` with `B = ∇²R` (the amplitude's
    curvature) and `U = R` (the amplitude). Its state `x = R''/R` is the quantum
    potential, up to the cited `−ħ²/2m`. -/
def amplitudeBulk (R ddR : ℝ → ℝ) : Sanguine.UniversalEquation := ⟨ddR, R⟩

/-- **The quantum potential has the bulk form `x = B/U` (headline).** The real
    channel, normalized by `R`, is the Quantum Hamilton–Jacobi form `x − (φ')²`,
    where `x = (amplitudeBulk).state = R''/R` under the *assignment* `B := ∇²R`,
    `U := R`. The proof is the division `(R'' − R(φ')²)/R = R''/R − (φ')²`. The
    `B/U` form is a chosen labeling, **not forced** — no theorem says `B` must be
    `∇²R`. -/
theorem quantum_potential_is_bulk_ratio (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ)
    (hR : R y ≠ 0) :
    (cellCurv R dR ddR dφ ddφ y).re / R y
      = (amplitudeBulk R ddR).state y - (dφ y) ^ 2 := by
  rw [cellCurv_re]
  simp only [amplitudeBulk, Sanguine.UniversalEquation.state]
  rw [sub_div]
  congr 1
  rw [mul_comm, mul_div_assoc, div_self hR, mul_one]

-- ═══════════════════════════════════════════════════════════════════
-- IMAGINARY CHANNEL: CONTINUITY OF THE BORN DENSITY ρ = |ψ|² = R²
-- ═══════════════════════════════════════════════════════════════════

/-- **The probability current** `ρ·v = R²·φ'` — the Born density `ρ = R²`
    transported by the phase-gradient velocity `v = φ'`. -/
def current (R dφ : ℝ → ℝ) : ℝ → ℝ := fun y => R y * R y * dφ y

/-- The current's divergence `(R²φ')' = 2R R'φ' + R²φ''`. -/
theorem hasDerivAt_current (R dφ dR ddφ : ℝ → ℝ)
    (hR : ∀ y, HasDerivAt R (dR y) y) (hdφ : ∀ y, HasDerivAt dφ (ddφ y) y) (y : ℝ) :
    HasDerivAt (current R dφ)
      ((dR y * R y + R y * dR y) * dφ y + R y * R y * ddφ y) y := by
  change HasDerivAt (R * R * dφ) _ _
  simpa only [Pi.mul_apply] using ((hR y).mul (hR y)).mul (hdφ y)

/-- **The imaginary channel is the continuity equation (headline).** `R·Im(cellCurv)`
    equals the current divergence `(R²·φ')'` — conservation of the Born density
    `ρ = |ψ|²`. -/
theorem continuity_is_current_divergence (R dR ddR dφ ddφ : ℝ → ℝ)
    (hR : ∀ y, HasDerivAt R (dR y) y) (hdφ : ∀ y, HasDerivAt dφ (ddφ y) y) (y : ℝ) :
    R y * (cellCurv R dR ddR dφ ddφ y).im = deriv (current R dφ) y := by
  rw [(hasDerivAt_current R dφ dR ddφ hR hdφ y).deriv, cellCurv_im]
  ring

-- ═══════════════════════════════════════════════════════════════════
-- THE BUNDLE: ONE DIFFERENTIATION = THE TWO MADELUNG EQUATIONS
-- ═══════════════════════════════════════════════════════════════════

/-- **Madelung from the bulk (headline bundle).** One differentiation of the phase
    cell `ψ = R·e^{iφ}`, split into two channels, is the Madelung pair: the real
    channel is Quantum Hamilton–Jacobi with quantum potential `x = B/U`
    (`B = ∇²R`, `U = R`); the imaginary channel is continuity of the Born
    density. -/
theorem madelung_from_bulk (R dR ddR dφ ddφ : ℝ → ℝ)
    (hR : ∀ y, HasDerivAt R (dR y) y) (hdφ : ∀ y, HasDerivAt dφ (ddφ y) y) (y : ℝ)
    (hRy : R y ≠ 0) :
    (cellCurv R dR ddR dφ ddφ y).re / R y
        = (amplitudeBulk R ddR).state y - (dφ y) ^ 2
      ∧ R y * (cellCurv R dR ddR dφ ddφ y).im = deriv (current R dφ) y :=
  ⟨quantum_potential_is_bulk_ratio R dR ddR dφ ddφ y hRy,
   continuity_is_current_divergence R dR ddR dφ ddφ hR hdφ y⟩

end MadelungCell
end Sanguine

end
