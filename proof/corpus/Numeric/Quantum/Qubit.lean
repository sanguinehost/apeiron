import Mathlib.Tactic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Complex.Trigonometric

/-!
# Qubit Bloch geometry and the spin-wave rotations (`QREG-QUBIT-BLOCH`)

Epic the quantum-register note, rows 11–13. A qubit is `ψ = (α, β) : ℂ × ℂ`, not
normalised. Its Bloch vector is `bloch ψ = (2 Re(ᾱβ), 2 Im(ᾱβ), |α|² − |β|²)`.

## Landed here

* `bloch_norm`: `bx² + by² + bz² = (|α|² + |β|²)²`. This is the Kustaanheimo–Stiefel/Hopf norm
  identity `ks_norm` (`|x| = |u|²`) of `../sanguine/proof/Apeiron/Physics/KustaanheimoStiefel.lean`
  (sanguine rev `abcf09e3`, line 116), restated over `ℂ²`. The coordinate map is
  `u = (Re α, Im β, Re β, Im α)`: `bloch_eq_ks` shows `bloch (α, β) = (ksX3 u, ksX2 u, ksX1 u)`
  with the defs of that file (lines 53–71) copied here, and `ks_norm` is proved locally.
* Generators (Lohmiller–Slotine, arXiv 2405.06328, §4.2, eq. 36–39; spin wave
  `e^{iΣ·n sγ/2}`):
  * `bloch_rz`: `Rz θ ψ = (e^{−iθ/2} α, e^{iθ/2} β)` rotates the Bloch vector about `z` by `+θ`
    (right-handed: `bx' = cos θ·bx − sin θ·by`, `by' = sin θ·bx + cos θ·by`, `bz' = bz`).
  * `bloch_ry`: `Ry θ ψ = (cos(θ/2) α − sin(θ/2) β, sin(θ/2) α + cos(θ/2) β)` rotates about `y` by
    `+θ` (`bx' = cos θ·bx + sin θ·bz`, `by' = by`, `bz' = cos θ·bz − sin θ·bx`).
* General SU(2): `suU a b`, `U = [[a, −b̄], [b, ā]]`, acts as `blochVec (suU a b ψ) =
  suR a b *ᵥ blochVec ψ` with the explicit real matrix `suR a b` (`suR_blochVec`, no hypothesis).
  `suR_orthogonal`: `(suR a b)ᵀ * suR a b = 1` under `|a|² + |b|² = 1`.
* Pauli combination (Pauli 1927): `sigmaN n1 n2 n3 = !![n3, n1 − i n2; n1 + i n2, −n3]` (`Σ·n`).
  `sigmaN_sq`: `(Σ·n)² = |n|² · 1`. For `|n| = 1`: `projP_idem`, `projM_idem` (`P± = (1 ± Σ·n)/2`
  are idempotent), `projP_add_projM` (`P₊ + P₋ = 1`), `projP_sub_projM` (`P₊ − P₋ = Σ·n`),
  `projP_mul_projM` and `projM_mul_projP` (`P₊P₋ = P₋P₊ = 0`).
* Real meridian `n = (sin t, 0, cos t)`: `sigmaN_meridian` is `!![cos t, sin t; sin t, −cos t]`, the
  butterfly `(u, v) ↦ (c u + s v, s u − c v)` of `tools/onebit/kan_probe.py` (`register`);
  `sigmaN_meridian_sq`: it squares to `1`; `sigmaN_meridian_pi_four`: at `t = π/4` it equals
  `(1/√2) !![1, 1; 1, −1]`, the normalised Hadamard.

## Not claimed

* Nothing here is a statement about the algebra `R[x]/(x² − σ)` at `σ ≠ −1`, or about the
  register of `tools/onebit`; the meridian identity is a matrix identity over `ℂ`.
* The Bloch map is a map on `ℂ²` with its own normalisation; no quantum-measurement, Born-rule or
  density-matrix claim is made. The Kustaanheimo–Stiefel source is not imported; its identity is
  reproved here, and the correspondence to it is stated, not checked by a shared module.
* Rotations are stated for real angles and the unit-hypothesis orthogonality; no claim about
  floating-point implementations, codegen or the Rust crate.

Axioms: `{propext, Classical.choice, Quot.sound}` (no `sorry`, `axiom`, `admit`, `native_decide`,
large `decide`, or `maxHeartbeats 0`).
-/

namespace Numeric.Qubit

open Complex ComplexConjugate Matrix

noncomputable section

/-- The phase `e^{i x}` of a real angle `x`. -/
def phase (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

theorem phase_re (x : ℝ) : (phase x).re = Real.cos x := Complex.exp_ofReal_mul_I_re x

theorem phase_im (x : ℝ) : (phase x).im = Real.sin x := Complex.exp_ofReal_mul_I_im x

/-- Bloch vector `(bx, by, bz)` of `ψ = (α, β)`, unnormalised. -/
def bloch (ψ : ℂ × ℂ) : ℝ × ℝ × ℝ :=
  (2 * (conj ψ.1 * ψ.2).re, 2 * (conj ψ.1 * ψ.2).im, Complex.normSq ψ.1 - Complex.normSq ψ.2)

/-- The Bloch vector as a `Fin 3`-indexed column. -/
def blochVec (ψ : ℂ × ℂ) : Fin 3 → ℝ :=
  ![(bloch ψ).1, (bloch ψ).2.1, (bloch ψ).2.2]

/-! ## Hopf / Kustaanheimo–Stiefel norm identity -/

/-- `ksX1` of `../sanguine/proof/Apeiron/Physics/KustaanheimoStiefel.lean` (line 53), copied. -/
def ksX1 (u1 u2 u3 u4 : ℝ) : ℝ := u1 ^ 2 - u2 ^ 2 - u3 ^ 2 + u4 ^ 2

/-- `ksX2` of the sanguine KS file (line 60), copied. -/
def ksX2 (u1 u2 u3 u4 : ℝ) : ℝ := 2 * (u1 * u2 - u3 * u4)

/-- `ksX3` of the sanguine KS file (line 67), copied. -/
def ksX3 (u1 u2 u3 u4 : ℝ) : ℝ := 2 * (u1 * u3 + u2 * u4)

/-- `ksR` of the sanguine KS file (line 71), copied. -/
def ksR (u1 u2 u3 u4 : ℝ) : ℝ := u1 ^ 2 + u2 ^ 2 + u3 ^ 2 + u4 ^ 2

/-- The KS identity `ks_norm` (sanguine line 116), reproved locally: `|x| = |u|²`. -/
theorem ks_norm (u1 u2 u3 u4 : ℝ) :
    ksX1 u1 u2 u3 u4 ^ 2 + ksX2 u1 u2 u3 u4 ^ 2 + ksX3 u1 u2 u3 u4 ^ 2
      = ksR u1 u2 u3 u4 ^ 2 := by
  simp only [ksX1, ksX2, ksX3, ksR]
  ring

/-- The Bloch vector is the KS image of `u = (Re α, Im β, Re β, Im α)`. -/
theorem bloch_eq_ks (α β : ℂ) :
    bloch (α, β) = (ksX3 α.re β.im β.re α.im, ksX2 α.re β.im β.re α.im,
      ksX1 α.re β.im β.re α.im) := by
  simp only [bloch, ksX1, ksX2, ksX3, Complex.mul_re, Complex.mul_im, Complex.conj_re,
    Complex.conj_im, Complex.normSq_apply, Prod.mk.injEq]
  refine ⟨?_, ?_, ?_⟩ <;> ring

/-- `bx² + by² + bz² = (|α|² + |β|²)²`, the Hopf norm identity over `ℂ²`. -/
theorem bloch_norm (ψ : ℂ × ℂ) :
    (bloch ψ).1 ^ 2 + (bloch ψ).2.1 ^ 2 + (bloch ψ).2.2 ^ 2
      = (Complex.normSq ψ.1 + Complex.normSq ψ.2) ^ 2 := by
  obtain ⟨α, β⟩ := ψ
  simp only [bloch, Complex.mul_re, Complex.mul_im, Complex.conj_re, Complex.conj_im,
    Complex.normSq_apply]
  ring

/-! ## Generators: `Rz`, `Ry` -/

/-- `Rz θ` acts as `diag(e^{−iθ/2}, e^{iθ/2})`. -/
def rzU (θ : ℝ) (ψ : ℂ × ℂ) : ℂ × ℂ := (phase (-(θ / 2)) * ψ.1, phase (θ / 2) * ψ.2)

/-- `Ry θ` acts as the real matrix `[[cos(θ/2), −sin(θ/2)], [sin(θ/2), cos(θ/2)]]`. -/
def ryU (θ : ℝ) (ψ : ℂ × ℂ) : ℂ × ℂ :=
  ((Real.cos (θ / 2) : ℂ) * ψ.1 - (Real.sin (θ / 2) : ℂ) * ψ.2,
    (Real.sin (θ / 2) : ℂ) * ψ.1 + (Real.cos (θ / 2) : ℂ) * ψ.2)

/-- `Rz θ` rotates the Bloch vector about `z` by `+θ`. -/
theorem bloch_rz (θ : ℝ) (ψ : ℂ × ℂ) :
    (bloch (rzU θ ψ)).1 = Real.cos θ * (bloch ψ).1 - Real.sin θ * (bloch ψ).2.1 ∧
    (bloch (rzU θ ψ)).2.1 = Real.sin θ * (bloch ψ).1 + Real.cos θ * (bloch ψ).2.1 ∧
    (bloch (rzU θ ψ)).2.2 = (bloch ψ).2.2 := by
  obtain ⟨α, β⟩ := ψ
  have hc : Real.cos θ = Real.cos (θ / 2) ^ 2 - Real.sin (θ / 2) ^ 2 := by
    have h := Real.cos_two_mul (θ / 2)
    rw [show 2 * (θ / 2) = θ by ring] at h
    linarith [Real.sin_sq_add_cos_sq (θ / 2)]
  have hs : Real.sin θ = 2 * Real.sin (θ / 2) * Real.cos (θ / 2) := by
    have h := Real.sin_two_mul (θ / 2)
    rw [show 2 * (θ / 2) = θ by ring] at h
    exact h
  have H : Real.sin (θ / 2) ^ 2 + Real.cos (θ / 2) ^ 2 = 1 := Real.sin_sq_add_cos_sq (θ / 2)
  simp only [bloch, rzU, Complex.mul_re, Complex.mul_im, Complex.conj_re, Complex.conj_im,
    Complex.normSq_apply, phase_re, phase_im, Real.cos_neg, Real.sin_neg, hc, hs]
  refine ⟨?_, ?_, ?_⟩
  · ring
  · ring
  · linear_combination ((α.re ^ 2 + α.im ^ 2) - (β.re ^ 2 + β.im ^ 2)) * H

/-- `Ry θ` rotates the Bloch vector about `y` by `+θ`. -/
theorem bloch_ry (θ : ℝ) (ψ : ℂ × ℂ) :
    (bloch (ryU θ ψ)).1 = Real.cos θ * (bloch ψ).1 + Real.sin θ * (bloch ψ).2.2 ∧
    (bloch (ryU θ ψ)).2.1 = (bloch ψ).2.1 ∧
    (bloch (ryU θ ψ)).2.2 = Real.cos θ * (bloch ψ).2.2 - Real.sin θ * (bloch ψ).1 := by
  obtain ⟨α, β⟩ := ψ
  have hc : Real.cos θ = Real.cos (θ / 2) ^ 2 - Real.sin (θ / 2) ^ 2 := by
    have h := Real.cos_two_mul (θ / 2)
    rw [show 2 * (θ / 2) = θ by ring] at h
    linarith [Real.sin_sq_add_cos_sq (θ / 2)]
  have hs : Real.sin θ = 2 * Real.sin (θ / 2) * Real.cos (θ / 2) := by
    have h := Real.sin_two_mul (θ / 2)
    rw [show 2 * (θ / 2) = θ by ring] at h
    exact h
  have H : Real.sin (θ / 2) ^ 2 + Real.cos (θ / 2) ^ 2 = 1 := Real.sin_sq_add_cos_sq (θ / 2)
  simp only [bloch, ryU, Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im,
    Complex.sub_re, Complex.sub_im, Complex.conj_re, Complex.conj_im, Complex.normSq_apply,
    hc, hs, Complex.ofReal_re, Complex.ofReal_im]
  refine ⟨?_, ?_, ?_⟩
  · ring
  · linear_combination (2 * (α.re * β.im - α.im * β.re)) * H
  · ring

/-! ## General SU(2) -/

/-- `U = [[a, −b̄], [b, ā]]` acting on `ψ = (α, β)`. -/
def suU (a b : ℂ) (ψ : ℂ × ℂ) : ℂ × ℂ := (a * ψ.1 - conj b * ψ.2, b * ψ.1 + conj a * ψ.2)

/-- The real `3 × 3` matrix of `U(a, b)` on `(bx, by, bz)`, with `a = p + iq`, `b = r + is`. -/
def suR (a b : ℂ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![a.re ^ 2 - a.im ^ 2 - b.re ^ 2 + b.im ^ 2, 2 * (a.re * a.im - b.re * b.im),
      2 * (a.re * b.re + a.im * b.im);
    -2 * (a.re * a.im + b.re * b.im), a.re ^ 2 - a.im ^ 2 + b.re ^ 2 - b.im ^ 2,
      2 * (a.re * b.im - a.im * b.re);
    -2 * (a.re * b.re - a.im * b.im), -2 * (a.re * b.im + a.im * b.re),
      a.re ^ 2 + a.im ^ 2 - b.re ^ 2 - b.im ^ 2]

/-- `U(a, b)` maps the Bloch vector by `suR a b`; no normalisation is assumed. -/
theorem suR_blochVec (a b : ℂ) (ψ : ℂ × ℂ) :
    blochVec (suU a b ψ) = suR a b *ᵥ blochVec ψ := by
  obtain ⟨α, β⟩ := ψ
  ext i
  fin_cases i <;>
    simp [blochVec, bloch, suU, suR, mulVec, dotProduct, Fin.sum_univ_three,
      Complex.mul_re, Complex.mul_im, Complex.conj_re, Complex.conj_im,
      Complex.normSq_apply] <;>
    ring

/-- Under `|a|² + |b|² = 1`, `suR a b` is orthogonal. -/
theorem suR_orthogonal (a b : ℂ) (h : Complex.normSq a + Complex.normSq b = 1) :
    (suR a b).transpose * suR a b = 1 := by
  have hN : a.re ^ 2 + a.im ^ 2 + b.re ^ 2 + b.im ^ 2 = 1 := by
    rw [Complex.normSq_apply, Complex.normSq_apply] at h
    linear_combination h
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [suR, Matrix.mul_apply, Fin.sum_univ_three,
      Matrix.transpose_apply] <;>
    first
    | ring1
    | linear_combination (a.re ^ 2 + a.im ^ 2 + b.re ^ 2 + b.im ^ 2 + 1) * hN

/-! ## Pauli combination `Σ·n` -/

/-- `Σ·n = n₁σₓ + n₂σ_y + n₃σ_z` for real `n`. -/
def sigmaN (n1 n2 n3 : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(n3 : ℂ), (n1 : ℂ) - (n2 : ℂ) * I; (n1 : ℂ) + (n2 : ℂ) * I, -(n3 : ℂ)]

/-- `(Σ·n)² = |n|² · 1`. -/
theorem sigmaN_sq (n1 n2 n3 : ℝ) :
    sigmaN n1 n2 n3 * sigmaN n1 n2 n3
      = ((n1 ^ 2 + n2 ^ 2 + n3 ^ 2 : ℝ) : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [sigmaN, Matrix.mul_apply, Fin.sum_univ_two] <;>
    first
    | ring1
    | linear_combination (-(n2 : ℂ) ^ 2) * Complex.I_sq

/-- `P₊ = (1 + Σ·n)/2`. -/
def projP (n1 n2 n3 : ℝ) : Matrix (Fin 2) (Fin 2) ℂ := (1 / 2 : ℂ) • (1 + sigmaN n1 n2 n3)

/-- `P₋ = (1 − Σ·n)/2`. -/
def projM (n1 n2 n3 : ℝ) : Matrix (Fin 2) (Fin 2) ℂ := (1 / 2 : ℂ) • (1 - sigmaN n1 n2 n3)

/-- For `|n| = 1`, `(Σ·n)² = 1`, the common input of the projector identities. -/
theorem sigmaN_sq_unit {n1 n2 n3 : ℝ} (h : n1 ^ 2 + n2 ^ 2 + n3 ^ 2 = 1) :
    sigmaN n1 n2 n3 * sigmaN n1 n2 n3 = 1 := by
  rw [sigmaN_sq, h]
  simp

theorem projP_idem {n1 n2 n3 : ℝ} (h : n1 ^ 2 + n2 ^ 2 + n3 ^ 2 = 1) :
    projP n1 n2 n3 * projP n1 n2 n3 = projP n1 n2 n3 := by
  have hC : (n1 : ℂ) ^ 2 + (n2 : ℂ) ^ 2 + (n3 : ℂ) ^ 2 = 1 := by exact_mod_cast h
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [projP, sigmaN, Matrix.mul_apply, Fin.sum_univ_two] <;>
    first
    | ring1
    | linear_combination (1 / 4 : ℂ) * hC - ((n2 : ℂ) ^ 2 / 4) * Complex.I_sq

theorem projM_idem {n1 n2 n3 : ℝ} (h : n1 ^ 2 + n2 ^ 2 + n3 ^ 2 = 1) :
    projM n1 n2 n3 * projM n1 n2 n3 = projM n1 n2 n3 := by
  have hC : (n1 : ℂ) ^ 2 + (n2 : ℂ) ^ 2 + (n3 : ℂ) ^ 2 = 1 := by exact_mod_cast h
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [projM, sigmaN, Matrix.mul_apply, Fin.sum_univ_two] <;>
    first
    | ring1
    | linear_combination (1 / 4 : ℂ) * hC - ((n2 : ℂ) ^ 2 / 4) * Complex.I_sq

theorem projP_add_projM (n1 n2 n3 : ℝ) :
    projP n1 n2 n3 + projM n1 n2 n3 = 1 := by
  unfold projP projM
  module

theorem projP_sub_projM (n1 n2 n3 : ℝ) :
    projP n1 n2 n3 - projM n1 n2 n3 = sigmaN n1 n2 n3 := by
  unfold projP projM
  module

theorem projP_mul_projM {n1 n2 n3 : ℝ} (h : n1 ^ 2 + n2 ^ 2 + n3 ^ 2 = 1) :
    projP n1 n2 n3 * projM n1 n2 n3 = 0 := by
  have hC : (n1 : ℂ) ^ 2 + (n2 : ℂ) ^ 2 + (n3 : ℂ) ^ 2 = 1 := by exact_mod_cast h
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [projP, projM, sigmaN, Matrix.mul_apply, Fin.sum_univ_two] <;>
    first
    | ring1
    | linear_combination (-1 / 4 : ℂ) * hC + ((n2 : ℂ) ^ 2 / 4) * Complex.I_sq

theorem projM_mul_projP {n1 n2 n3 : ℝ} (h : n1 ^ 2 + n2 ^ 2 + n3 ^ 2 = 1) :
    projM n1 n2 n3 * projP n1 n2 n3 = 0 := by
  have hC : (n1 : ℂ) ^ 2 + (n2 : ℂ) ^ 2 + (n3 : ℂ) ^ 2 = 1 := by exact_mod_cast h
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [projP, projM, sigmaN, Matrix.mul_apply, Fin.sum_univ_two] <;>
    first
    | ring1
    | linear_combination (-1 / 4 : ℂ) * hC + ((n2 : ℂ) ^ 2 / 4) * Complex.I_sq

/-! ## Real meridian `n = (sin t, 0, cos t)` -/

/-- On the real meridian `Σ·n` is the butterfly `[[c, s], [s, −c]]`. -/
theorem sigmaN_meridian (t : ℝ) :
    sigmaN (Real.sin t) 0 (Real.cos t)
      = !![(Real.cos t : ℂ), (Real.sin t : ℂ); (Real.sin t : ℂ), -(Real.cos t : ℂ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [sigmaN]

theorem sigmaN_meridian_sq (t : ℝ) :
    sigmaN (Real.sin t) 0 (Real.cos t) * sigmaN (Real.sin t) 0 (Real.cos t) = 1 := by
  apply sigmaN_sq_unit
  linear_combination Real.sin_sq_add_cos_sq t

/-- At `t = π/4`, `Σ·n = (1/√2) [[1, 1], [1, −1]]`. -/
theorem sigmaN_meridian_pi_four :
    sigmaN (Real.sin (Real.pi / 4)) 0 (Real.cos (Real.pi / 4))
      = (((Real.sqrt 2)⁻¹ : ℝ) : ℂ) • !![(1 : ℂ), 1; 1, -1] := by
  have hpos : (Real.sqrt 2 : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.2 (by norm_num : (0 : ℝ) < 2)).ne'
  have hsq : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
    exact_mod_cast Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 2)
  rw [sigmaN_meridian, Real.sin_pi_div_four, Real.cos_pi_div_four]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp <;>
    field_simp <;>
    linear_combination hsq

end

end Numeric.Qubit
