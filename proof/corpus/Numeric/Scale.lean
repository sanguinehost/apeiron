/-
Shared Q32.32 scale, denotation and the proved error target for the numeric refinements
(-*`; spec the inference-parity note).

Kept separate from the per-body modules so each refinement binds only its own body's evidence; the
window predicates and the per-body statements stay with their bodies.
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import ApeironWord

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-- `2^32`, the Q32.32 scale. -/
def q32 : ℝ := (2 : ℝ) ^ (32 : ℕ)

/-- The signed real value of a Q32.32 `i64`: `v.val / 2^32`. -/
def toReal (v : I64) : ℝ := (v.val : ℝ) / q32

/-- The signed real value of a Q32.32 `i128`: `v.val / 2^32`. -/
def toReal128 (v : I128) : ℝ := (v.val : ℝ) / q32

/-- `2^-20`, the proved relative-error target. -/
def epsExp : ℝ := (2 : ℝ) ^ (-20 : ℤ)

theorem q32_pos : (0 : ℝ) < q32 := by
  unfold q32
  positivity

theorem q32_ne_zero : q32 ≠ 0 := ne_of_gt q32_pos

end

end apeiron_word
