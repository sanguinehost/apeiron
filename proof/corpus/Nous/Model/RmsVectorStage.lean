/-
# The vector RMSNorm stage at the rounded input mean

`Nous.Model.BlockGrid` lands `rmsStage a ha B hB : Stage ℝ ℝ` as the **scalar** RMSNorm stage at an
externally fixed radicand `a`; the mean-square computation that produces `a` is a hypothesis of the
block, not derived from the block's input. `Numeric.RmsNormIdeal` lands the missing half: on
`rms_norm_domain row eps a` the executed `word_rms_norm_fixed` output is within
`rms_norm_refines_ideal` of the **ideal** reference `rmsIdeal row eps xᵢ` with the exact rational
mean `meanSqReal row + toReal eps`.

This module composes the two into a dimension-preserving `Stage (Fin n → ℝ) (Fin n → ℝ)` whose
radicand **is** the rounded mean of the input vector: the domain is the image of a `Slice I64` row of
length `n` whose computed radicand `rmsMean row + eps` lies in `rms_norm_domain`, the reference is
the ideal vector RMSNorm `rmsIdealVec (toReal eps)`, and the executed map is the choice-based output
of `rms_norm_refines_ideal`.

## Contents

* `rmsIdealVec` and `rmsIdealVec_lipschitz`: the ideal vector RMSNorm
  `x ↦ x / √(mean(x²) + eps)` is Lipschitz in the sup norm with explicit constant
  `L = (1 + n)/√eps` (the headline bound proved is `(1 + n)/√eps`, which dominates the sharper
  `(1 + √n)/√eps`; the proof uses the cruder `√n · |yᵢ|/Dy` route).
* `rmsVectorStage`: the staged refinement. `dom` carries the magnitude cap `|xᵢ| ≤ B` and the
  radicand floor `toReal a ≥ A0`, with `A0 > 1/(2·q32)` the hypothesis that makes the
  reciprocal-square-root Lipschitz factor in `rms_norm_refines_ideal` monotone in the radicand.
  `delta` is the resulting uniform per-coordinate bound; `L` is `(1 + n)/√(toReal eps)`.
* `rmsVectorStage_dom_witness`: a concrete 2-element row in the domain (non-vacuity).

## Trust boundary

The Lipschitz bound and the domain algebra are kernel-checked at the abstract real level. The defect
budget is grounded in the reused `rms_norm_refines_ideal` chain (`Numeric/RmsNormIdeal.lean`), whose
own trust boundary stands. No claim is made about rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V or
the `unsafe` container. Nothing here re-derives `meanSqReal`'s rounding; that is `rmsMean_real_bound`.
-/
import Nous.Model.BlockGrid
import Numeric.RmsNormIdeal

namespace Nous.Model

open apeiron_word Aeneas Aeneas.Std Result Classical

noncomputable section

/-! ## The ideal vector RMSNorm and its Lipschitz constant -/

/-- The ideal vector RMSNorm at `eps`: coordinate `i` of `x / √(mean(x²) + eps)`, the vector form of
`rmsIdeal` (`Numeric/RmsNormIdeal.lean:37`) with `meanSqReal` written as `(∑ xᵢ²)/n`. -/
def rmsIdealVec {n : ℕ} (eps : ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => x i / Real.sqrt ((∑ j, x j ^ 2) / (n : ℝ) + eps)

@[simp] theorem rmsIdealVec_apply {n : ℕ} (eps : ℝ) (x : Fin n → ℝ) (i : Fin n) :
    rmsIdealVec eps x i = x i / Real.sqrt ((∑ j, x j ^ 2) / (n : ℝ) + eps) := rfl

/-- **Ideal vector RMSNorm Lipschitz bound.** For `eps > 0` and `n ≥ 1` the map
`x ↦ x / √(mean(x²) + eps)` is `(1 + n)/√eps`-Lipschitz in the sup norm.

Derivation: with `D z = √(mean(z²) + eps)`, `|D x − D y| ≤ √n·‖x − y‖` (from
`|Sx − Sy| ≤ ‖x−y‖·n·(√Sx+√Sy)` and `√n(√Sx+√Sy) ≤ n(Dx+Dy)`), and `|yᵢ| ≤ √n·Dy`; splitting
`xᵢ/Dx − yᵢ/Dy` into `(xᵢ−yᵢ)/Dx + (yᵢ/Dy)((Dy−Dx)/Dx)` and bounding `1/Dx ≤ 1/√eps` gives
`(1 + n)‖x−y‖/√eps`. -/
theorem rmsIdealVec_lipschitz {n : ℕ} (hn : 1 ≤ n) {eps : ℝ} (heps : 0 < eps)
    (x y : Fin n → ℝ) :
    ‖rmsIdealVec eps x - rmsIdealVec eps y‖ ≤ ((1 + (n : ℝ)) / Real.sqrt eps) * ‖x - y‖ := by
  set Sx : ℝ := ∑ j, x j ^ 2 with hSx
  set Sy : ℝ := ∑ j, y j ^ 2 with hSy
  set Dx : ℝ := Real.sqrt (Sx / (n : ℝ) + eps) with hDx
  set Dy : ℝ := Real.sqrt (Sy / (n : ℝ) + eps) with hDy
  set A : ℝ := ‖x - y‖ with hA
  have hnR : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hSxnn : 0 ≤ Sx := by rw [hSx]; exact Finset.sum_nonneg fun j _ => sq_nonneg _
  have hSynn : 0 ≤ Sy := by rw [hSy]; exact Finset.sum_nonneg fun j _ => sq_nonneg _
  have hDxn : 0 < Dx := by
    rw [hDx]
    exact Real.sqrt_pos_of_pos (by linarith [div_nonneg hSxnn hnR.le])
  have hDyn : 0 < Dy := by
    rw [hDy]
    exact Real.sqrt_pos_of_pos (by linarith [div_nonneg hSynn hnR.le])
  have hAn : 0 ≤ A := by rw [hA]; positivity
  have hsqrtn : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos_of_pos hnR
  have hepsn : 0 < Real.sqrt eps := Real.sqrt_pos_of_pos heps
  -- coordinatewise magnitude bounds by the `ℓ²` sums
  have hxy : ∀ j : Fin n, |x j| ≤ Real.sqrt Sx := by
    intro j
    have h1 : x j ^ 2 ≤ Sx := by
      rw [hSx]
      exact Finset.single_le_sum (fun i _ => sq_nonneg (x i)) (Finset.mem_univ j)
    calc |x j| = Real.sqrt (x j ^ 2) := (Real.sqrt_sq_eq_abs (x j)).symm
      _ ≤ Real.sqrt Sx := Real.sqrt_le_sqrt h1
  have hyy : ∀ j : Fin n, |y j| ≤ Real.sqrt Sy := by
    intro j
    have h1 : y j ^ 2 ≤ Sy := by
      rw [hSy]
      exact Finset.single_le_sum (fun i _ => sq_nonneg (y i)) (Finset.mem_univ j)
    calc |y j| = Real.sqrt (y j ^ 2) := (Real.sqrt_sq_eq_abs (y j)).symm
      _ ≤ Real.sqrt Sy := Real.sqrt_le_sqrt h1
  have hAcoord : ∀ j : Fin n, |x j - y j| ≤ A := by
    intro j
    have h := norm_le_pi_norm (x - y) j
    rw [Pi.sub_apply, Real.norm_eq_abs] at h
    rwa [hA]
  -- lower bounds `√S ≤ √n · D`
  have hDxlb : Real.sqrt Sx ≤ Real.sqrt (n : ℝ) * Dx := by
    have h1 : Real.sqrt (Sx / (n : ℝ)) ≤ Dx := by
      rw [hDx]
      exact Real.sqrt_le_sqrt (by linarith [heps.le])
    rw [Real.sqrt_div' Sx hnR.le] at h1
    calc Real.sqrt Sx ≤ Dx * Real.sqrt (n : ℝ) := (div_le_iff₀ hsqrtn).mp h1
      _ = Real.sqrt (n : ℝ) * Dx := by ring
  have hDylb : Real.sqrt Sy ≤ Real.sqrt (n : ℝ) * Dy := by
    have h1 : Real.sqrt (Sy / (n : ℝ)) ≤ Dy := by
      rw [hDy]
      exact Real.sqrt_le_sqrt (by linarith [heps.le])
    rw [Real.sqrt_div' Sy hnR.le] at h1
    calc Real.sqrt Sy ≤ Dy * Real.sqrt (n : ℝ) := (div_le_iff₀ hsqrtn).mp h1
      _ = Real.sqrt (n : ℝ) * Dy := by ring
  have hP : Real.sqrt Sx + Real.sqrt Sy ≤ Real.sqrt (n : ℝ) * (Dx + Dy) := by linarith
  have hsqn : Real.sqrt (n : ℝ) * Real.sqrt (n : ℝ) = (n : ℝ) := by
    rw [← sq, Real.sq_sqrt hnR.le]
  have hnP : (n : ℝ) * (Real.sqrt Sx + Real.sqrt Sy)
      ≤ Real.sqrt (n : ℝ) * ((n : ℝ) * (Dx + Dy)) := by
    calc (n : ℝ) * (Real.sqrt Sx + Real.sqrt Sy)
        = Real.sqrt (n : ℝ) * (Real.sqrt (n : ℝ) * (Real.sqrt Sx + Real.sqrt Sy)) := by
          rw [← mul_assoc, hsqn]
      _ ≤ Real.sqrt (n : ℝ) * ((n : ℝ) * (Dx + Dy)) := by
          apply mul_le_mul_of_nonneg_left _ hsqrtn.le
          calc Real.sqrt (n : ℝ) * (Real.sqrt Sx + Real.sqrt Sy)
              ≤ Real.sqrt (n : ℝ) * (Real.sqrt (n : ℝ) * (Dx + Dy)) :=
                mul_le_mul_of_nonneg_left hP hsqrtn.le
            _ = (n : ℝ) * (Dx + Dy) := by rw [← mul_assoc, hsqn]
  -- `|Sx − Sy| ≤ A·n·(√Sx+√Sy)`
  have hSdiff : |Sx - Sy| ≤ A * ((n : ℝ) * (Real.sqrt Sx + Real.sqrt Sy)) := by
    have hsum : Sx - Sy = ∑ j, (x j - y j) * (x j + y j) := by
      rw [hSx, hSy, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl (fun j _ => by ring)
    rw [hsum]
    calc |∑ j, (x j - y j) * (x j + y j)|
        ≤ ∑ j, |(x j - y j) * (x j + y j)| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, |x j - y j| * |x j + y j| := by simp only [abs_mul]
      _ ≤ ∑ j, A * |x j + y j| :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hAcoord j) (abs_nonneg _)
      _ = A * ∑ j, |x j + y j| := (Finset.mul_sum _ _ _).symm
      _ ≤ A * ∑ _j : Fin n, (Real.sqrt Sx + Real.sqrt Sy) := by
          apply mul_le_mul_of_nonneg_left _ hAn
          apply Finset.sum_le_sum
          intro j _
          have h := abs_add_le (x j) (y j)
          linarith [hxy j, hyy j]
      _ = A * ((n : ℝ) * (Real.sqrt Sx + Real.sqrt Sy)) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  -- `|Dx − Dy| ≤ √n·A`
  have hDdiff : |Dx - Dy| ≤ Real.sqrt (n : ℝ) * A := by
    have hx2 : Dx ^ 2 = Sx / (n : ℝ) + eps := by
      rw [hDx, Real.sq_sqrt (by linarith [div_nonneg hSxnn hnR.le])]
    have hy2 : Dy ^ 2 = Sy / (n : ℝ) + eps := by
      rw [hDy, Real.sq_sqrt (by linarith [div_nonneg hSynn hnR.le])]
    have hden : 0 < (n : ℝ) * (Dx + Dy) := mul_pos hnR (add_pos hDxn hDyn)
    have hDsub : Dx - Dy = (Sx - Sy) / ((n : ℝ) * (Dx + Dy)) := by
      rw [eq_div_iff (ne_of_gt hden)]
      rw [show (Dx - Dy) * ((n : ℝ) * (Dx + Dy)) = ((Dx - Dy) * (Dx + Dy)) * (n : ℝ) by ring,
        show (Dx - Dy) * (Dx + Dy) = Dx ^ 2 - Dy ^ 2 by ring, hx2, hy2]
      field_simp
      ring
    have hkey : |Sx - Sy| ≤ (Real.sqrt (n : ℝ) * A) * ((n : ℝ) * (Dx + Dy)) := by
      calc |Sx - Sy| ≤ A * ((n : ℝ) * (Real.sqrt Sx + Real.sqrt Sy)) := hSdiff
        _ ≤ A * (Real.sqrt (n : ℝ) * ((n : ℝ) * (Dx + Dy))) :=
              mul_le_mul_of_nonneg_left hnP hAn
        _ = (Real.sqrt (n : ℝ) * A) * ((n : ℝ) * (Dx + Dy)) := by ring
    calc |Dx - Dy| = |Sx - Sy| / ((n : ℝ) * (Dx + Dy)) := by
          rw [hDsub, abs_div, abs_of_pos hden]
      _ ≤ Real.sqrt (n : ℝ) * A := by
          rw [div_le_iff₀ hden]
          exact hkey
  -- `|yᵢ| ≤ √n·Dy`
  have hyiD : ∀ i : Fin n, |y i| / Dy ≤ Real.sqrt (n : ℝ) := by
    intro i
    rw [div_le_iff₀ hDyn]
    calc |y i| ≤ Real.sqrt Sy := hyy i
      _ ≤ Real.sqrt (n : ℝ) * Dy := hDylb
  -- the per-coordinate split
  have hmain : ∀ i : Fin n,
      |rmsIdealVec eps x i - rmsIdealVec eps y i| ≤ ((1 + (n : ℝ)) / Real.sqrt eps) * A := by
    intro i
    have hDx' : Real.sqrt ((∑ j, x j ^ 2) / (n : ℝ) + eps) = Dx := by
      rw [← hSx]
    have hDy' : Real.sqrt ((∑ j, y j ^ 2) / (n : ℝ) + eps) = Dy := by
      rw [← hSy]
    rw [rmsIdealVec_apply, rmsIdealVec_apply, hDx', hDy']
    have hdecomp : x i / Dx - y i / Dy
        = (x i - y i) / Dx + (y i / Dy) * ((Dy - Dx) / Dx) := by
      field_simp
      ring
    rw [hdecomp]
    have h1 : |x i - y i| / Dx ≤ A / Dx := div_le_div_of_nonneg_right (hAcoord i) hDxn.le
    have h2 : |Dy - Dx| / Dx ≤ Real.sqrt (n : ℝ) * A / Dx := by
      rw [abs_sub_comm]
      exact div_le_div_of_nonneg_right hDdiff hDxn.le
    have h3 : (|y i| / Dy) * (|Dy - Dx| / Dx)
        ≤ Real.sqrt (n : ℝ) * (Real.sqrt (n : ℝ) * A / Dx) :=
      mul_le_mul (hyiD i) h2 (by positivity) (Real.sqrt_nonneg _)
    calc |(x i - y i) / Dx + (y i / Dy) * ((Dy - Dx) / Dx)|
        ≤ |(x i - y i) / Dx| + |(y i / Dy) * ((Dy - Dx) / Dx)| := abs_add_le _ _
      _ = |x i - y i| / Dx + (|y i| / Dy) * (|Dy - Dx| / Dx) := by
          rw [abs_div, abs_of_pos hDxn, abs_mul,
            show |y i / Dy| = |y i| / Dy by rw [abs_div, abs_of_pos hDyn],
            show |(Dy - Dx) / Dx| = |Dy - Dx| / Dx by rw [abs_div, abs_of_pos hDxn]]
      _ ≤ A / Dx + Real.sqrt (n : ℝ) * (Real.sqrt (n : ℝ) * A / Dx) := add_le_add h1 h3
      _ = (1 + (n : ℝ)) * A / Dx := by
          have hmid' : Real.sqrt (n : ℝ) * (Real.sqrt (n : ℝ) * A) = (n : ℝ) * A := by
            rw [← mul_assoc, hsqn]
          have hmid : Real.sqrt (n : ℝ) * (Real.sqrt (n : ℝ) * A / Dx) = (n : ℝ) * A / Dx := by
            rw [← mul_div_assoc, hmid']
          rw [hmid]; ring
      _ ≤ (1 + (n : ℝ)) * A / Real.sqrt eps := by
          apply div_le_div_of_nonneg_left (by positivity) hepsn
          rw [hDx]
          exact Real.sqrt_le_sqrt (by linarith [div_nonneg hSxnn hnR.le])
      _ = ((1 + (n : ℝ)) / Real.sqrt eps) * A := by ring
  apply (pi_norm_le_iff_of_nonneg (by positivity)).2
  intro i
  rw [Real.norm_eq_abs]
  exact hmain i

/-! ## The grid embedding, its mean square, and the ideal reference -/

/-- The real vector of a `Slice I64` read at width `n`; out-of-range indices read `0`. On rows of
length `n` this is the row's Q32.32 denotation, and `rmsVecAdm` below is its admissible predicate. -/
def rowVec (n : ℕ) (row : Slice I64) : Fin n → ℝ :=
  fun i => toReal (row.val.getD i.val (0#i64))

/-- Sum of a mapped list as a `Fin n`-indexed sum of `getD`s, for a list of length `n`. -/
theorem sum_map_getD {n : ℕ} {α : Type*} (l : List α) (d : α) (f : α → ℝ) (hlen : l.length = n) :
    (l.map f).sum = ∑ i : Fin n, f (l.getD i.val d) := by
  induction l generalizing n with
  | nil =>
      simp only [List.length_nil] at hlen
      subst n
      simp
  | cons a t ih =>
      simp only [List.length_cons] at hlen
      rw [← hlen, List.map_cons, List.sum_cons, Fin.sum_univ_succ]
      change f a + (List.map f t).sum = f a + ∑ i : Fin t.length, f (t.getD (i : ℕ) d)
      rw [ih rfl]

/-- `meanSqReal` of a row of length `n` is the exact mean square of its real vector. -/
theorem meanSqReal_eq_rowVec {n : ℕ} {row : Slice I64} (hlen : row.val.length = n) :
    meanSqReal row = (∑ j : Fin n, (rowVec n row j) ^ 2) / (n : ℝ) := by
  unfold meanSqReal rowVec
  rw [hlen, sum_map_getD row (0#i64) (fun x => (toReal x) ^ 2) hlen]

/-- The ideal scalar reference at a grid entry equals the ideal vector reference at the same index:
the radicand's mean is the mean of the row's real vector (`meanSqReal_eq_rowVec`). -/
theorem rmsIdeal_eq_rmsIdealVec {n : ℕ} {eps : I64} {row : Slice I64}
    (hlen : row.val.length = n) (i : Fin n) :
    rmsIdeal row eps (row.val.getD i.val (0#i64)) = rmsIdealVec (toReal eps) (rowVec n row) i := by
  unfold rmsIdeal rmsIdealVec
  rw [meanSqReal_eq_rowVec hlen]
  rfl

/-! ## The vector RMSNorm stage at the computed radicand -/

/-- The uniform per-coordinate defect budget of the vector RMSNorm stage: the
`rms_norm_refines_ideal` bound evaluated at the coordinate cap `B` and the radicand floor `A0`. -/
noncomputable def rmsVecDelta (B A0 : ℝ) : ℝ :=
  epsExp * B * (1 / Real.sqrt A0) + (4 / q32) * B + 1 / (2 * q32)
    + B * (1 / (2 * q32)) / (2 * (A0 - 1 / (2 * q32)) * Real.sqrt (A0 - 1 / (2 * q32)))

/-- `rmsVecDelta` is nonnegative above the rounding floor. -/
theorem rmsVecDelta_nonneg (B A0 : ℝ) (hB : 0 ≤ B) (hA0 : 1 / (2 * q32) < A0) :
    0 ≤ rmsVecDelta B A0 := by
  have hc : (0 : ℝ) < 1 / (2 * q32) := by unfold q32; positivity
  have h1 : 0 ≤ epsExp * B * (1 / Real.sqrt A0) := by
    have := epsExp_nonneg
    positivity
  have h2 : 0 ≤ (4 / q32) * B := mul_nonneg (by unfold q32; norm_num) hB
  have h3 : 0 ≤ B * (1 / (2 * q32))
      / (2 * (A0 - 1 / (2 * q32)) * Real.sqrt (A0 - 1 / (2 * q32))) := by
    have hpos : 0 < A0 - 1 / (2 * q32) := by linarith
    refine div_nonneg (mul_nonneg hB hc.le) (le_of_lt ?_)
    exact mul_pos (mul_pos (by norm_num) hpos) (Real.sqrt_pos_of_pos hpos)
  unfold rmsVecDelta
  linarith

/-- The `rms_norm_refines_ideal` per-coordinate bound is dominated by `rmsVecDelta B A0` whenever
`|x| ≤ B` and the radicand `a` is at or above `A0`. -/
theorem rmsVecDelta_spec (B A0 x a : ℝ) (hB : 0 ≤ B) (hx : |x| ≤ B)
    (hA0 : 1 / (2 * q32) < A0) (ha : A0 ≤ a) :
    epsExp * |x| * (1 / Real.sqrt a) + (4 / q32) * |x| + 1 / (2 * q32)
      + |x| * (1 / (2 * q32)) / (2 * (a - 1 / (2 * q32)) * Real.sqrt (a - 1 / (2 * q32)))
    ≤ rmsVecDelta B A0 := by
  have hc : (0 : ℝ) < 1 / (2 * q32) := by unfold q32; positivity
  have hE : 0 ≤ epsExp := epsExp_nonneg
  have hxnn : 0 ≤ |x| := abs_nonneg x
  have hA0pos : 0 < A0 := lt_trans hc hA0
  have hapos : 0 < a := lt_of_lt_of_le hA0pos ha
  have hsA0 : 0 < Real.sqrt A0 := Real.sqrt_pos_of_pos hA0pos
  have hsle : Real.sqrt A0 ≤ Real.sqrt a := Real.sqrt_le_sqrt ha
  have hinv : 1 / Real.sqrt a ≤ 1 / Real.sqrt A0 :=
    div_le_div_of_nonneg_left (by norm_num) hsA0 hsle
  have ht1 : epsExp * |x| * (1 / Real.sqrt a) ≤ epsExp * B * (1 / Real.sqrt A0) :=
    mul_le_mul (mul_le_mul_of_nonneg_left hx hE) hinv (by positivity) (by positivity)
  have ht2 : (4 / q32) * |x| ≤ (4 / q32) * B :=
    mul_le_mul_of_nonneg_left hx (by unfold q32; norm_num)
  have hd1 : 0 ≤ A0 - 1 / (2 * q32) := by linarith
  have hd2 : A0 - 1 / (2 * q32) ≤ a - 1 / (2 * q32) := by linarith
  have hs2 : Real.sqrt (A0 - 1 / (2 * q32)) ≤ Real.sqrt (a - 1 / (2 * q32)) :=
    Real.sqrt_le_sqrt hd2
  have hda : 0 ≤ a - 1 / (2 * q32) := by linarith
  have hdenmono : 2 * (A0 - 1 / (2 * q32)) * Real.sqrt (A0 - 1 / (2 * q32))
      ≤ 2 * (a - 1 / (2 * q32)) * Real.sqrt (a - 1 / (2 * q32)) := by
    calc 2 * (A0 - 1 / (2 * q32)) * Real.sqrt (A0 - 1 / (2 * q32))
        = 2 * ((A0 - 1 / (2 * q32)) * Real.sqrt (A0 - 1 / (2 * q32))) := by ring
      _ ≤ 2 * ((a - 1 / (2 * q32)) * Real.sqrt (a - 1 / (2 * q32))) :=
          mul_le_mul_of_nonneg_left (mul_le_mul hd2 hs2 (Real.sqrt_nonneg _) hda) (by norm_num)
      _ = 2 * (a - 1 / (2 * q32)) * Real.sqrt (a - 1 / (2 * q32)) := by ring
  have hdenA0pos : 0 < 2 * (A0 - 1 / (2 * q32)) * Real.sqrt (A0 - 1 / (2 * q32)) := by
    have hpos : 0 < A0 - 1 / (2 * q32) := by linarith
    exact mul_pos (mul_pos (by norm_num) hpos) (Real.sqrt_pos_of_pos hpos)
  have ht4 : |x| * (1 / (2 * q32)) / (2 * (a - 1 / (2 * q32)) * Real.sqrt (a - 1 / (2 * q32)))
      ≤ B * (1 / (2 * q32))
        / (2 * (A0 - 1 / (2 * q32)) * Real.sqrt (A0 - 1 / (2 * q32))) := by
    have hnum : |x| * (1 / (2 * q32)) ≤ B * (1 / (2 * q32)) :=
      mul_le_mul_of_nonneg_right hx hc.le
    have hdena : 0 < a - 1 / (2 * q32) := by linarith
    calc |x| * (1 / (2 * q32)) / (2 * (a - 1 / (2 * q32)) * Real.sqrt (a - 1 / (2 * q32)))
        ≤ B * (1 / (2 * q32)) / (2 * (a - 1 / (2 * q32)) * Real.sqrt (a - 1 / (2 * q32))) :=
          div_le_div_of_nonneg_right hnum
            (le_of_lt (mul_pos (mul_pos (by norm_num) hdena) (Real.sqrt_pos_of_pos hdena)))
      _ ≤ B * (1 / (2 * q32))
            / (2 * (A0 - 1 / (2 * q32)) * Real.sqrt (A0 - 1 / (2 * q32))) :=
          div_le_div_of_nonneg_left (by positivity) hdenA0pos hdenmono
  unfold rmsVecDelta
  linarith

/-- Admissibility of a grid row for the vector RMSNorm stage: width `n`, the coordinate cap `B`, and
a computed radicand `a = rmsMean row + eps` lying in `rms_norm_domain row eps a` at or above the
floor `A0`. -/
def rmsVecAdm (n : ℕ) (eps : I64) (B A0 : ℝ) (row : Slice I64) : Prop :=
  row.val.length = n ∧
  (∀ i : Fin n, |rowVec n row i| ≤ B) ∧
  (∃ a : I64, rms_norm_domain row eps a ∧ A0 ≤ toReal a)

/-- The executed output read at width `n` for a row at the fixed radicand `a`: the choice-based
`rms_norm_refines_ideal` output. -/
noncomputable def rmsVecOutAux (n : ℕ) (eps a : I64) (row : Slice I64)
    (hd : rms_norm_domain row eps a) : Fin n → ℝ :=
  let out := Classical.choose (rms_norm_refines_ideal row eps a hd)
  fun i => toReal (out.val.getD i.val (0#i64))

/-- Coordinatewise unfold of `rmsVecOutAux`. -/
theorem rmsVecOutAux_apply {n : ℕ} (eps a : I64) (row : Slice I64)
    (hd : rms_norm_domain row eps a) (i : Fin n) :
    rmsVecOutAux n eps a row hd i
      = toReal ((Classical.choose (rms_norm_refines_ideal row eps a hd)).val.getD i.val (0#i64)) :=
  rfl

/-- The executed output of the vector RMSNorm stage for a grid row: `rmsVecOutAux` at the computed
radicand when the row admits one, zero otherwise. -/
noncomputable def rmsVecOut (n : ℕ) (eps : I64) (A0 : ℝ) (row : Slice I64) : Fin n → ℝ :=
  if h : ∃ a : I64, rms_norm_domain row eps a ∧ A0 ≤ toReal a then
    rmsVecOutAux n eps (Classical.choose h) row (Classical.choose_spec h).1
  else fun _ => 0

/-- The defect bound of the executed output at the fixed radicand `a` against the ideal vector
reference, uniform over the domain. -/
theorem rmsVecOutAux_bound {n : ℕ} (eps a : I64) (row : Slice I64)
    (hd : rms_norm_domain row eps a) (B A0 : ℝ) (hB : 0 ≤ B)
    (hlen : row.val.length = n) (hcap : ∀ i : Fin n, |rowVec n row i| ≤ B)
    (hA0 : 1 / (2 * q32) < A0) (hA0a : A0 ≤ toReal a) :
    ‖rmsVecOutAux n eps a row hd - rmsIdealVec (toReal eps) (rowVec n row)‖ ≤ rmsVecDelta B A0 := by
  have hδ := rmsVecDelta_nonneg B A0 hB hA0
  obtain ⟨_hok, hlenout, hbnd⟩ := Classical.choose_spec (rms_norm_refines_ideal row eps a hd)
  apply (pi_norm_le_iff_of_nonneg hδ).2
  intro i
  rw [Pi.sub_apply, rmsVecOutAux_apply, Real.norm_eq_abs]
  have hi : i.val < row.val.length := by rw [hlen]; exact i.isLt
  obtain ⟨hi', hbi⟩ := hbnd i.val hi
  have hrowd : row.val.getD i.val (0#i64) = row.val[i.val]'hi :=
    List.getD_eq_getElem row.val (0#i64) hi
  have houtd : (Classical.choose (rms_norm_refines_ideal row eps a hd)).val.getD i.val (0#i64)
      = (Classical.choose (rms_norm_refines_ideal row eps a hd)).val[i.val]'hi' :=
    List.getD_eq_getElem _ (0#i64) hi'
  have hxrow : |toReal (row.val[i.val]'hi)| = |rowVec n row i| := by
    rw [← hrowd]; rfl
  rw [hxrow] at hbi
  rw [houtd, ← rmsIdeal_eq_rmsIdealVec hlen i, hrowd]
  exact le_trans hbi (rmsVecDelta_spec B A0 (rowVec n row i) (toReal a) hB (hcap i) hA0 hA0a)

/-- The defect bound of the executed output for any row admitting a computed radicand. -/
theorem rmsVecOut_bound {n : ℕ} (eps : I64) (A0 B : ℝ) (row : Slice I64) (hB : 0 ≤ B)
    (hlen : row.val.length = n) (hcap : ∀ i : Fin n, |rowVec n row i| ≤ B)
    (hA0 : 1 / (2 * q32) < A0)
    (hstrong : ∃ a : I64, rms_norm_domain row eps a ∧ A0 ≤ toReal a) :
    ‖rmsVecOut n eps A0 row - rmsIdealVec (toReal eps) (rowVec n row)‖ ≤ rmsVecDelta B A0 := by
  unfold rmsVecOut
  rw [dif_pos hstrong]
  exact rmsVecOutAux_bound eps (Classical.choose hstrong) row (Classical.choose_spec hstrong).1
    B A0 hB hlen hcap hA0 (Classical.choose_spec hstrong).2

/-- **The vector RMSNorm stage at the rounded input mean.** The reference is
`rmsIdealVec (toReal eps)`; the domain is the image of a `Slice I64` row of width `n` whose computed
radicand `rmsMean row + eps` lies in `rms_norm_domain` at or above the floor `A0`, under the
coordinate cap `B`; the executed map is the choice-based `rms_norm_refines_ideal` output; the defect
budget is `rmsVecDelta B A0` and the expansion factor is `(1 + n)/√(toReal eps)`. -/
noncomputable def rmsVectorStage (n : ℕ) (hn : 1 ≤ n) (eps : I64) (heps : 0 < toReal eps)
    (B A0 : ℝ) (hB : 0 ≤ B) (hA0 : 1 / (2 * q32) < A0) :
    Stage (Fin n → ℝ) (Fin n → ℝ) where
  ref := rmsIdealVec (toReal eps)
  exec := fun z =>
    if h : ∃ row : Slice I64, rmsVecAdm n eps B A0 row ∧ rowVec n row = z
    then rmsVecOut n eps A0 (Classical.choose h)
    else rmsIdealVec (toReal eps) z
  dom := {z | ∃ row : Slice I64, rmsVecAdm n eps B A0 row ∧ rowVec n row = z}
  delta := rmsVecDelta B A0
  L := (1 + (n : ℝ)) / Real.sqrt (toReal eps)
  delta_nonneg := rmsVecDelta_nonneg B A0 hB hA0
  L_nonneg := div_nonneg (by positivity) (Real.sqrt_nonneg _)
  defect := by
    intro z hz
    have hz' : ∃ row : Slice I64, rmsVecAdm n eps B A0 row ∧ rowVec n row = z := hz
    rw [dif_pos hz']
    obtain ⟨hadm, hrow⟩ := Classical.choose_spec hz'
    have hlen : (Classical.choose hz').val.length = n := hadm.1
    have hcap : ∀ i : Fin n, |rowVec n (Classical.choose hz') i| ≤ B := hadm.2.1
    have hstrong : ∃ a : I64, rms_norm_domain (Classical.choose hz') eps a ∧ A0 ≤ toReal a :=
      hadm.2.2
    have hgoal := rmsVecOut_bound eps A0 B (Classical.choose hz') hB hlen hcap hA0 hstrong
    simpa [hrow] using hgoal
  expansion := rmsIdealVec_lipschitz hn heps

/-! ## A concrete non-vacuity witness -/

/-- The concrete 2-element zero row. -/
def row2 : Slice I64 :=
  Slice.from [0#i64, 0#i64] (by
    have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    rcases System.Platform.numBits_eq with h | h <;> rw [hmax, h] <;> norm_num)

theorem row2_val : row2.val = [0#i64, 0#i64] := by
  simp [row2]

theorem row2_length : row2.val.length = 2 := by
  rw [row2_val]; rfl

theorem rmsMean_row2 : rmsMean row2 = 0 := by
  unfold rmsMean divRoundNearest
  rw [row2_val]
  norm_num

/-- The concrete witness row is admissible at `eps = a = 2^32`: the rounded mean of a zero row is
`0`, its radicand `2^32` is in `rsqrt_window`, the zero epilogue is representable for every root, and
the `i64`/`i128` accumulation windows hold. -/
theorem row2_domain : rms_norm_domain row2 (4294967296#i64) (4294967296#i64) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [row2_val]; simp
  · rw [rmsMean_row2]; norm_num
  · unfold rsqrt_window; norm_num
  · intro r _ i hi
    have hzi : row2.val[i]'hi = 0#i64 := by
      have hi2 : i < 2 := by rw [row2_length] at hi; exact hi
      interval_cases i <;> simp [row2_val]
    have h0 : (0#i64).val = 0 := by norm_num
    rw [hzi, h0]
    constructor <;> · unfold divRoundNearest; norm_num
  · unfold rms_arith_window; rw [row2_val]; norm_num
  · rw [row2_val]; norm_num
  · rw [row2_val]; norm_num

/-- **Non-vacuity.** The concrete 2-element zero row lies in `rmsVectorStage`'s domain at
`eps = 2^32` (`toReal eps = 1`), cap `B = 0` and floor `A0 = 1/2`. -/
theorem rmsVectorStage_dom_witness :
    rowVec 2 row2 ∈ (rmsVectorStage 2 (by norm_num) (4294967296#i64)
      (by unfold toReal q32; norm_num) (0 : ℝ) (1 / 2) (by norm_num)
      (by unfold q32; norm_num)).dom := by
  refine ⟨row2, ?_, rfl⟩
  refine ⟨row2_length, ?_, ?_⟩
  · intro i
    have hzi : row2.val.getD i.val (0#i64) = 0#i64 := by
      rw [row2_val]; fin_cases i <;> rfl
    show |toReal (row2.val.getD i.val (0#i64))| ≤ (0 : ℝ)
    rw [hzi, toReal_zero, abs_zero]
  · exact ⟨4294967296#i64, row2_domain, by unfold toReal q32; norm_num⟩

end

end Nous.Model
