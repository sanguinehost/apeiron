import Mathlib

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.IdempotentsAnyRing

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-!
# Two composed elliptic cells: `ℂ ⊗_ℝ ℂ` and its hyperbolic idempotent

`APEIRON-QARCH-L2`. The canonical object is `Apeiron.Trichotomy.AnyRing.RingCell R = R × R`
with `ringMul σ z w = (z.1*w.1 + σ*z.2*w.2, z.1*w.2 + z.2*w.1)` — the ring
`R[x]/(x² − σ)` in two coordinates. Take `R = ℂ` (the first elliptic cell, where
`Complex.I² = −1`) and `σ = −1` (the second elliptic cell). The result is the real tensor
square `ℂ ⊗_ℝ ℂ`, presented by its two generators `iL = i ⊗ 1` and `iR = 1 ⊗ i`.

* `iL_sq`, `iR_sq`: each factor is elliptic, `iL² = iR² = (−1, 0) = σ`.
* `j_eq`, `j_sq`, `j_ne_one`, `j_ne_neg_one`: the product `j = iL * iR = i ⊗ i` is
  *hyperbolic*, `j² = +1` and `j ≠ ±1`: the hyperbolic face `x² = +1, x ≠ ±1` sits inside two
  composed elliptic cells. This is the algebraic content of the "real quantum theory" part of
  Renou et al., *Quantum theory based on real numbers can be experimentally falsified*,
  arXiv:2101.10873, whose tensor rule is the statement that the two factors agree on `e`.
* `e_idem`, `e'_idem`, `e_complementary`: `e = (1 − i⊗i)/2` and `e' = 1 − e` are the two
  complementary idempotents of `ℂ ⊗_ℝ ℂ` in the canonical sense `RingComplementary σ e e'`.
* `tensor_rule`, `anti_rule`: on `e` the two factors act identically (`iL * e = iR * e`),
  on `e'` they act oppositely (`iL * e' = −(iR * e')`): the complex tensor rule of quantum
  mechanics and its conjugate copy.
* `split`, `split_mul`, `split_one`, `split_e`, `split_e'`: `φ z = (z.1 + I·z.2, z.1 − I·z.2)`
  is a ring isomorphism `ℂ ⊗_ℝ ℂ ≅ ℂ × ℂ` with inverse
  `ψ (a, b) = ((a+b)/2, (a−b)/(2I))`; `e` is carried to `(1, 0)`.

Everything here is arithmetic in `ℂ` and `RingCell ℂ`; the `CommRing` instance on `ℂ × ℂ` is
Mathlib's componentwise one, so `*` on `RingCell ℂ` is never used — only `ringMul`.
-/

namespace Quantum
namespace EllipticTensor

open Apeiron.Trichotomy.AnyRing

noncomputable section

/-- `iL = i ⊗ 1`, the first elliptic cell's unit imaginary. -/
def iL : RingCell ℂ := (Complex.I, 0)

/-- `iR = 1 ⊗ i`, the second elliptic cell's unit imaginary. -/
def iR : RingCell ℂ := (0, 1)

/-- `j = i ⊗ i = iL * iR`, the product of the two imaginary units. -/
def j : RingCell ℂ := ringMul (-1) iL iR

/-- `e = (1 − i⊗i)/2`, the idempotent the tensor rule singles out. -/
def e : RingCell ℂ := ((1 : ℂ) / 2, -Complex.I / 2)

/-- `e' = (1 + i⊗i)/2`, the complement `1 − e`. -/
def e' : RingCell ℂ := ((1 : ℂ) / 2, Complex.I / 2)

/-- The forward coordinate map `φ z = (z.1 + I·z.2, z.1 − I·z.2)`. -/
def phi (z : RingCell ℂ) : ℂ × ℂ := (z.1 + Complex.I * z.2, z.1 - Complex.I * z.2)

/-- The inverse coordinate map `ψ (a, b) = ((a+b)/2, (a−b)/(2I))`. -/
def psi (w : ℂ × ℂ) : RingCell ℂ :=
  ((w.1 + w.2) / 2, (w.1 - w.2) / (2 * Complex.I))

/-- `ℂ ⊗_ℝ ℂ ≅ ℂ × ℂ` as rings, by `φ` with explicit inverse `ψ`. -/
def split : RingCell ℂ ≃ ℂ × ℂ where
  toFun := phi
  invFun := psi
  left_inv := by
    intro z
    apply Prod.ext
    · show ((z.1 + Complex.I * z.2) + (z.1 - Complex.I * z.2)) / 2 = z.1
      ring
    · show ((z.1 + Complex.I * z.2) - (z.1 - Complex.I * z.2)) / (2 * Complex.I) = z.2
      have h : (z.1 + Complex.I * z.2) - (z.1 - Complex.I * z.2) = (2 * Complex.I) * z.2 := by
        ring
      rw [h, mul_div_cancel_left₀ _ (by simp [Complex.I_ne_zero])]
  right_inv := by
    intro w
    have h : Complex.I * ((w.1 - w.2) / (2 * Complex.I)) = (w.1 - w.2) / 2 := by
      field_simp
    apply Prod.ext
    · show (w.1 + w.2) / 2 + Complex.I * ((w.1 - w.2) / (2 * Complex.I)) = w.1
      rw [h]
      ring
    · show (w.1 + w.2) / 2 - Complex.I * ((w.1 - w.2) / (2 * Complex.I)) = w.2
      rw [h]
      ring

/-- Both factors of the tensor square are elliptic: `iL² = σ`. -/
theorem iL_sq : ringMul (-1) iL iL = (-1, 0) := by
  ext <;> simp [iL]

/-- Both factors of the tensor square are elliptic: `iR² = σ`. -/
theorem iR_sq : ringMul (-1) iR iR = (-1, 0) := by
  ext <;> simp [iR]

/-- `j = i ⊗ i` has the claimed coordinates. -/
theorem j_eq : j = (0, Complex.I) := by
  ext <;> simp [j, iL, iR]

/-- The hyperbolic face inside the composed cells: `j² = +1`. -/
theorem j_sq : ringMul (-1) j j = ringOne := by
  rw [j_eq]
  ext <;> simp

/-- `j ≠ 1`: the hyperbolic element is not the unit. -/
theorem j_ne_one : j ≠ ringOne := by
  intro h
  have h2 : Complex.I = 0 := by simpa [j, iL, iR] using congrArg Prod.snd h
  exact Complex.I_ne_zero h2

/-- `j ≠ −1`: the hyperbolic element is not minus the unit. -/
theorem j_ne_neg_one : j ≠ (-1, 0) := by
  intro h
  have h2 : Complex.I = 0 := by simpa [j, iL, iR] using congrArg Prod.snd h
  exact Complex.I_ne_zero h2

/-- `e` is idempotent under `ringMul (−1)`. -/
theorem e_idem : RingIdempotent (-1) e := by
  unfold RingIdempotent
  apply Prod.ext
  · simp [e]
    ring_nf
    rw [Complex.I_sq]
    ring
  · simp [e]
    ring

/-- `e'` is idempotent under `ringMul (−1)`. -/
theorem e'_idem : RingIdempotent (-1) e' := by
  unfold RingIdempotent
  apply Prod.ext
  · simp [e']
    ring_nf
    rw [Complex.I_sq]
    ring
  · simp [e']
    ring

/-- `e` and `e'` are complementary idempotents: `e + e' = 1` and `e * e' = 0`. -/
theorem e_complementary : RingComplementary (-1) e e' := by
  refine ⟨⟨e_idem, e'_idem, ?_⟩, ?_⟩
  · apply Prod.ext
    · simp [e, e']
      ring_nf
      rw [Complex.I_sq]
      ring
    · simp [e, e']
      ring
  · apply Prod.ext
    · simp [e, e']
      ring
    · simp [e, e']
      ring

/-- The tensor rule: on `e`, the two factors `i ⊗ 1` and `1 ⊗ i` act identically. -/
theorem tensor_rule : ringMul (-1) iL e = ringMul (-1) iR e := by
  apply Prod.ext
  · simp [e, iL, iR]
    ring
  · simp [e, iL, iR]
    ring_nf
    rw [Complex.I_sq]
    ring

/-- The conjugate copy: on `e' = 1 − e`, the two factors act oppositely. -/
theorem anti_rule : ringMul (-1) iL e' = -(ringMul (-1) iR e') := by
  apply Prod.ext
  · simp [e', iL, iR]
    ring
  · simp [e', iL, iR]
    ring_nf
    rw [Complex.I_sq]
    ring

/-- `φ` is multiplicative: `ℂ ⊗_ℝ ℂ ≅ ℂ × ℂ` as rings, componentwise. -/
theorem split_mul (z w : RingCell ℂ) :
    phi (ringMul (-1) z w) = ((phi z).1 * (phi w).1, (phi z).2 * (phi w).2) := by
  apply Prod.ext
  · simp [phi]
    ring_nf
    rw [Complex.I_sq]
    ring
  · simp [phi]
    ring_nf
    rw [Complex.I_sq]
    ring

/-- `φ` carries the unit to `(1, 1)`. -/
theorem split_one : phi ringOne = (1, 1) := by
  apply Prod.ext
  · simp [phi]
  · simp [phi]

/-- `φ` carries the idempotent `e` to `(1, 0)`. -/
theorem split_e : phi e = (1, 0) := by
  apply Prod.ext
  · simp [phi, e]
    ring_nf
    rw [Complex.I_sq]
    ring
  · simp [phi, e]
    ring_nf
    rw [Complex.I_sq]
    ring

/-- `φ` carries the complement `e'` to `(0, 1)`. -/
theorem split_e' : phi e' = (0, 1) := by
  apply Prod.ext
  · simp [phi, e']
    ring_nf
    rw [Complex.I_sq]
    ring
  · simp [phi, e']
    ring_nf
    rw [Complex.I_sq]
    ring

/-- The unit's verdict: the hyperbolic face lies inside the two composed elliptic cells, they
    carry a complementary idempotent pair, the tensor and anti-tensor rules hold there, and
    `ℂ ⊗_ℝ ℂ ≅ ℂ × ℂ` is the componentwise product ring with `e ↦ (1, 0)`. -/
theorem verdict_l2 :
    (ringMul (-1) j j = ringOne ∧ j ≠ ringOne ∧ j ≠ (-1, 0))
      ∧ (RingIdempotent (-1) e ∧ RingIdempotent (-1) e' ∧ RingComplementary (-1) e e')
      ∧ (ringMul (-1) iL e = ringMul (-1) iR e
          ∧ ringMul (-1) iL e' = -(ringMul (-1) iR e'))
      ∧ (∀ z w : RingCell ℂ,
          phi (ringMul (-1) z w) = ((phi z).1 * (phi w).1, (phi z).2 * (phi w).2))
      ∧ phi e = (1, 0) :=
  ⟨⟨j_sq, j_ne_one, j_ne_neg_one⟩, ⟨e_idem, e'_idem, e_complementary⟩,
   ⟨tensor_rule, anti_rule⟩, split_mul, split_e⟩

end

end EllipticTensor
end Quantum
