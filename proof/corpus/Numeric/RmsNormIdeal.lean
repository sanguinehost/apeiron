/-
RMSNorm against the ideal reference (`APEIRON-ALGEBRA-G3-RMS-W3-B2`).

`Numeric.Engine.Norm` proves `rms_norm_refines_proved`: the extracted
`apeiron_word.word_rms_norm_fixed` tracks `rmsReal a x = toReal x / sqrt (toReal a)`, that is, the
reference divided by the square root of the *executed rounded* radicand `a`. The ideal RMSNorm is
`x / sqrt (meanSqReal row + toReal eps)` with the exact rational mean square
`meanSqReal row = (sum_i (toReal x_i)^2) / n`. This module lands the exact-rounding bound for the
mean (`rmsMean_real_bound`), the reciprocal-square-root Lipschitz bound
(`inv_sqrt_lipschitz`), and their composition against the executed refinement so the executed output
is bounded against `rmsIdeal` (`rms_norm_refines_ideal`), at grade `proved` under the closed axiom
cone `{propext, Classical.choice, Quot.sound}`.

`rmsMean row = round(Σ xᵢ² / (n·2³²))` accumulates `x.val²` in Q32.32 units, so the radicand real
value is the integer sum divided by `n·2⁶⁴`; the single rounding contributes `≤ 1/(2·2³²)`.

The residual obligation is the ideal-only face of the composition: this module bounds the executed
kernel against the ideal reference, not an extracted ideal kernel.
-/
import Numeric.Engine.Norm

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The ideal mean square and the ideal reference -/

/-- The ideal (exact rational) mean square of a row as a real: the mean of the squared Q32.32
denotations, `(Σ x ∈ row.val, (toReal x)^2) / row.val.length`. The length-zero case is the junk
value `0`; every use below is under `row.val ≠ []`. -/
def meanSqReal (row : Slice I64) : ℝ :=
  ((row.val.map (fun x => (toReal x) ^ 2)).sum) / (row.val.length : ℝ)

/-- The ideal RMSNorm coordinate `x / sqrt (meanSqReal row + toReal eps)`. -/
def rmsIdeal (row : Slice I64) (eps : I64) (x : I64) : ℝ :=
  toReal x / Real.sqrt (meanSqReal row + toReal eps)

/-- The squared-denotation sum as an integer sum: `Σ (toReal x)² = (Σ x.val²) / 2⁶⁴`. -/
theorem sum_sq_toReal (l : List I64) :
    (l.map (fun x => (toReal x) ^ 2)).sum =
      (((l.map (fun x => x.val * x.val)).sum : ℤ) : ℝ) / (2 : ℝ) ^ (64 : ℕ) := by
  induction l with
  | nil => simp
  | cons x xs ih =>
      simp only [List.map_cons, List.sum_cons]
      rw [ih]
      unfold toReal q32
      push_cast
      ring

/-- `meanSqReal` in terms of the integer sum computed by the kernel, for nonempty rows. -/
theorem meanSqReal_eq (row : Slice I64) (hne : row.val ≠ []) :
    meanSqReal row =
      (((row.val.map (fun x => x.val * x.val)).sum : ℤ) : ℝ) /
        ((row.val.length : ℝ) * (2 : ℝ) ^ (64 : ℕ)) := by
  unfold meanSqReal
  rw [sum_sq_toReal]
  have hn : (row.val.length : ℝ) ≠ 0 := by
    have h : 0 < row.val.length := List.length_pos_iff.mpr hne
    positivity
  field_simp

/-! ## The mean-rounding bound -/

/-- The scaled half-biased rounding bound: for `d > 0`, the Q32.32 quotient
`round(S/d)` is within `1/(2·2³²)` of the exact ratio `S/(d·2³²)`. -/
theorem divRoundNearest_scaled_bound (S d : ℤ) (hd : 0 < d) :
    |((divRoundNearest S d : ℤ) : ℝ) / (2 : ℝ) ^ (32 : ℕ) -
        (S : ℝ) / ((d : ℝ) * (2 : ℝ) ^ (32 : ℕ))| ≤ 1 / (2 * (2 : ℝ) ^ (32 : ℕ)) := by
  have hb := div_round_nearest_abs_bound S d hd
  have hbR : |((divRoundNearest S d : ℤ) : ℝ) * (d : ℝ) - (S : ℝ)| ≤ (d : ℝ) / 2 := by
    have h2Z : 2 * (d / 2) ≤ d := by
      have h := Int.mul_ediv_add_emod d 2
      have h0 : 0 ≤ d % 2 := Int.emod_nonneg d (by norm_num)
      omega
    have hbZ : 2 * |divRoundNearest S d * d - S| ≤ d := by
      unfold divRoundNearest at hb ⊢
      linarith
    have hcast := (Int.cast_le (R := ℝ)).mpr hbZ
    push_cast at hcast
    linarith
  have hdR : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  have hdenpos : (0 : ℝ) < (d : ℝ) * (2 : ℝ) ^ (32 : ℕ) := by positivity
  have hmain : ((divRoundNearest S d : ℤ) : ℝ) / (2 : ℝ) ^ (32 : ℕ) -
        (S : ℝ) / ((d : ℝ) * (2 : ℝ) ^ (32 : ℕ)) =
      (((divRoundNearest S d : ℤ) : ℝ) * (d : ℝ) - (S : ℝ)) /
        ((d : ℝ) * (2 : ℝ) ^ (32 : ℕ)) := by
    field_simp
  rw [hmain, abs_div, abs_of_pos hdenpos]
  calc |((divRoundNearest S d : ℤ) : ℝ) * (d : ℝ) - (S : ℝ)| /
          ((d : ℝ) * (2 : ℝ) ^ (32 : ℕ))
      ≤ ((d : ℝ) / 2) / ((d : ℝ) * (2 : ℝ) ^ (32 : ℕ)) :=
        div_le_div_of_nonneg_right hbR (le_of_lt hdenpos)
    _ = 1 / (2 * (2 : ℝ) ^ (32 : ℕ)) := by
        field_simp

/-- **Mean-rounding bound.** For a nonempty row, the executed Q32.32 mean square
`rmsMean row` is within `1/(2·q32)` of the exact rational mean square `meanSqReal row`. -/
theorem rmsMean_real_bound (row : Slice I64) (hne : row.val ≠ []) :
    |(rmsMean row : ℝ) / q32 - meanSqReal row| ≤ 1 / (2 * q32) := by
  have hnpos : 0 < (row.val.length : ℤ) := by
    exact_mod_cast (List.length_pos_iff.mpr hne : 0 < row.val.length)
  have hdpos : 0 < ((row.val.length : ℤ) * 2 ^ (32 : ℕ)) := by positivity
  have hgen := divRoundNearest_scaled_bound ((row.val.map (fun x => x.val * x.val)).sum)
      ((row.val.length : ℤ) * 2 ^ (32 : ℕ)) hdpos
  rw [show q32 = (2 : ℝ) ^ (32 : ℕ) from rfl]
  rw [show rmsMean row = divRoundNearest ((row.val.map (fun x => x.val * x.val)).sum)
        ((row.val.length : ℤ) * 2 ^ (32 : ℕ)) from rfl]
  rw [meanSqReal_eq row hne]
  have hden : (((row.val.length : ℤ) * 2 ^ (32 : ℕ) : ℤ) : ℝ) * (2 : ℝ) ^ (32 : ℕ) =
      (row.val.length : ℝ) * (2 : ℝ) ^ (64 : ℕ) := by
    push_cast
    ring
  rw [hden] at hgen
  exact hgen

/-! ## Reciprocal-square-root Lipschitz bound -/

/-- **Reciprocal-square-root Lipschitz bound.** For `0 < w ≤ u` and `w ≤ v`,
`|1/√u − 1/√v| ≤ |u − v| / (2·w·√w)`. -/
theorem inv_sqrt_lipschitz (u v w : ℝ) (hw : 0 < w) (hu : w ≤ u) (hv : w ≤ v) :
    |1 / Real.sqrt u - 1 / Real.sqrt v| ≤ |u - v| / (2 * w * Real.sqrt w) := by
  have hupos : 0 < u := lt_of_lt_of_le hw hu
  have hvpos : 0 < v := lt_of_lt_of_le hw hv
  have hwu : 0 < Real.sqrt u := Real.sqrt_pos_of_pos hupos
  have hwv : 0 < Real.sqrt v := Real.sqrt_pos_of_pos hvpos
  have hww : 0 < Real.sqrt w := Real.sqrt_pos_of_pos hw
  have hsum_pos : 0 < Real.sqrt v + Real.sqrt u := by linarith
  have hdiff : |Real.sqrt v - Real.sqrt u| ≤ |u - v| / (2 * Real.sqrt w) := by
    have hid : Real.sqrt v - Real.sqrt u = (v - u) / (Real.sqrt v + Real.sqrt u) := by
      rw [eq_div_iff (ne_of_gt hsum_pos)]
      nlinarith [Real.sq_sqrt (le_of_lt hvpos), Real.sq_sqrt (le_of_lt hupos)]
    rw [hid, abs_div, abs_of_pos hsum_pos]
    have hle : 2 * Real.sqrt w ≤ Real.sqrt v + Real.sqrt u := by
      have h1 : Real.sqrt w ≤ Real.sqrt u := Real.sqrt_le_sqrt hu
      have h2 : Real.sqrt w ≤ Real.sqrt v := Real.sqrt_le_sqrt hv
      linarith
    calc |v - u| / (Real.sqrt v + Real.sqrt u)
        ≤ |v - u| / (2 * Real.sqrt w) :=
          div_le_div_of_nonneg_left (abs_nonneg _) (by positivity) hle
      _ = |u - v| / (2 * Real.sqrt w) := by rw [abs_sub_comm]
  have hprod : w ≤ Real.sqrt u * Real.sqrt v := by
    have h1 : Real.sqrt w ≤ Real.sqrt u := Real.sqrt_le_sqrt hu
    have h2 : Real.sqrt w ≤ Real.sqrt v := Real.sqrt_le_sqrt hv
    have hm := mul_le_mul h1 h2 (le_of_lt hww) (le_of_lt hwu)
    have hsq : (Real.sqrt w) ^ 2 = w := Real.sq_sqrt (le_of_lt hw)
    nlinarith [hm, hsq]
  have h1 : 1 / Real.sqrt u - 1 / Real.sqrt v =
      (Real.sqrt v - Real.sqrt u) / (Real.sqrt u * Real.sqrt v) := by
    field_simp
  rw [h1, abs_div, abs_of_pos (mul_pos hwu hwv)]
  calc |Real.sqrt v - Real.sqrt u| / (Real.sqrt u * Real.sqrt v)
      ≤ (|u - v| / (2 * Real.sqrt w)) / (Real.sqrt u * Real.sqrt v) :=
        div_le_div_of_nonneg_right hdiff (le_of_lt (mul_pos hwu hwv))
    _ ≤ (|u - v| / (2 * Real.sqrt w)) / w :=
        div_le_div_of_nonneg_left (by positivity) hw hprod
    _ = |u - v| / (2 * w * Real.sqrt w) := by
        rw [div_div]
        ring

/-! ## Composition against the executed refinement -/

/-- The domain forces `toReal a > 1/(2·q32)`: `rsqrt_window a` gives `a.val ≥ 4096`, so
`toReal a ≥ 2⁻²⁰ > 2⁻³³ = 1/(2·q32)`. -/
theorem rms_domain_toReal_gt (row : Slice I64) (eps a : I64) (hdom : rms_norm_domain row eps a) :
    1 / (2 * q32) < toReal a := by
  obtain ⟨_, _, hwin, _, _, _, _⟩ := hdom
  rw [rsqrt_window] at hwin
  have h4096 : (4096 : ℝ) ≤ (a.val : ℝ) := by exact_mod_cast hwin.1
  rw [toReal, show q32 = (2 : ℝ) ^ (32 : ℕ) from rfl,
    lt_div_iff₀ (by positivity : (0 : ℝ) < (2 : ℝ) ^ (32 : ℕ))]
  have h1 : 1 / (2 * (2 : ℝ) ^ (32 : ℕ)) * (2 : ℝ) ^ (32 : ℕ) = 1 / 2 := by
    field_simp
  rw [h1]
  linarith

/-- **Ideal element bound.** If the executed radicand `toReal a` is within `1/(2·q32)` of the ideal
radicand `meanSqReal row + toReal eps` and `1/(2·q32) < toReal a`, then the executed per-coordinate
bound against `rmsReal a x` upgrades to a bound against `rmsIdeal row eps x`, at the cost of the
mean-rounding term scaled by the reciprocal-square-root Lipschitz constant. -/
theorem rms_ideal_element_bound (row : Slice I64) (eps : I64) (a x y : I64)
    (hδ : 1 / (2 * q32) < toReal a)
    (hmean : |toReal a - (meanSqReal row + toReal eps)| ≤ 1 / (2 * q32))
    (hrms : |toReal y - rmsReal a x| ≤
      epsExp * |toReal x| * (1 / Real.sqrt (toReal a)) + (4 / q32) * |toReal x| + 1 / (2 * q32)) :
    |toReal y - rmsIdeal row eps x| ≤
      (epsExp * |toReal x| * (1 / Real.sqrt (toReal a)) + (4 / q32) * |toReal x| + 1 / (2 * q32))
      + |toReal x| * (1 / (2 * q32)) /
          (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32))) := by
  have hcpos : 0 < 1 / (2 * q32) := by unfold q32; positivity
  have hwpos : 0 < toReal a - 1 / (2 * q32) := by linarith
  have hwu : toReal a - 1 / (2 * q32) ≤ toReal a := by linarith
  have hwv : toReal a - 1 / (2 * q32) ≤ meanSqReal row + toReal eps := by
    have h := (abs_le.mp hmean).2
    linarith
  have hdenpos : 0 < 2 * (toReal a - 1 / (2 * q32)) *
      Real.sqrt (toReal a - 1 / (2 * q32)) := by
    have hs := Real.sqrt_pos_of_pos hwpos
    positivity
  have hscale : |toReal x / Real.sqrt (toReal a) -
        toReal x / Real.sqrt (meanSqReal row + toReal eps)| ≤
      |toReal x| * (1 / (2 * q32)) /
        (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32))) := by
    have heq : toReal x / Real.sqrt (toReal a) -
          toReal x / Real.sqrt (meanSqReal row + toReal eps) =
        toReal x * (1 / Real.sqrt (toReal a) -
          1 / Real.sqrt (meanSqReal row + toReal eps)) := by ring
    rw [heq, abs_mul]
    have hlip := inv_sqrt_lipschitz (toReal a) (meanSqReal row + toReal eps)
      (toReal a - 1 / (2 * q32)) hwpos hwu hwv
    have h1 : |toReal x| * |1 / Real.sqrt (toReal a) -
          1 / Real.sqrt (meanSqReal row + toReal eps)| ≤
        |toReal x| * (|toReal a - (meanSqReal row + toReal eps)| /
          (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32)))) :=
      mul_le_mul_of_nonneg_left hlip (abs_nonneg _)
    have h2 : |toReal a - (meanSqReal row + toReal eps)| /
          (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32))) ≤
        (1 / (2 * q32)) /
          (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32))) :=
      div_le_div_of_nonneg_right hmean (le_of_lt hdenpos)
    calc |toReal x| * |1 / Real.sqrt (toReal a) -
          1 / Real.sqrt (meanSqReal row + toReal eps)|
        ≤ |toReal x| * (|toReal a - (meanSqReal row + toReal eps)| /
            (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32)))) := h1
      _ ≤ |toReal x| * ((1 / (2 * q32)) /
            (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32)))) :=
          mul_le_mul_of_nonneg_left h2 (abs_nonneg _)
      _ = |toReal x| * (1 / (2 * q32)) /
            (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32))) := by ring
  have hrms' : |toReal y - toReal x / Real.sqrt (toReal a)| ≤
      epsExp * |toReal x| * (1 / Real.sqrt (toReal a)) + (4 / q32) * |toReal x| + 1 / (2 * q32) := by
    have he : toReal x / Real.sqrt (toReal a) = rmsReal a x := by
      rw [rmsReal]
      ring
    rw [he]
    exact hrms
  have htri := abs_sub_le (toReal y) (toReal x / Real.sqrt (toReal a))
    (toReal x / Real.sqrt (meanSqReal row + toReal eps))
  have hmain : |toReal y - toReal x / Real.sqrt (meanSqReal row + toReal eps)| ≤
      (epsExp * |toReal x| * (1 / Real.sqrt (toReal a)) + (4 / q32) * |toReal x| + 1 / (2 * q32))
      + |toReal x| * (1 / (2 * q32)) /
          (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32))) := by
    linarith [htri, hrms', hscale]
  have hideal : rmsIdeal row eps x =
      toReal x / Real.sqrt (meanSqReal row + toReal eps) := by
    rw [rmsIdeal]
  rw [hideal]
  exact hmain

/-- **RMSNorm against the ideal reference.** On `rms_norm_domain row eps a`, the executed kernel's
output is within the `rms_norm_refines` bound plus the mean-rounding term
`|toReal xᵢ| · (1/(2·q32)) / (2·w·√w)`, `w = toReal a − 1/(2·q32)`, of the ideal reference
`rmsIdeal row eps xᵢ`. -/
theorem rms_norm_refines_ideal (row : Slice I64) (eps a : I64)
    (hdom : rms_norm_domain row eps a) :
    ∃ out : alloc.vec.Vec I64,
      word_rms_norm_fixed row eps = ok (core.result.Result.Ok out) ∧
      out.val.length = row.val.length ∧
      ∀ (i : ℕ) (hi : i < row.val.length),
        ∃ (hi' : i < out.val.length),
          |toReal (out.val[i]'hi') - rmsIdeal row eps (row.val[i]'hi)| ≤
            epsExp * |toReal row.val[i]| * (1 / Real.sqrt (toReal a))
              + (4 / q32) * |toReal row.val[i]| + 1 / (2 * q32)
              + |toReal row.val[i]| * (1 / (2 * q32)) /
                  (2 * (toReal a - 1 / (2 * q32)) * Real.sqrt (toReal a - 1 / (2 * q32))) := by
  have hne : row.val ≠ [] := hdom.1
  have ha : a.val = rmsMean row + eps.val := hdom.2.1
  have href : rms_norm_refines := rms_norm_refines_proved
  unfold rms_norm_refines at href
  obtain ⟨out, hout, hlen, hb⟩ := href row eps a hdom
  refine ⟨out, hout, hlen, ?_⟩
  intro i hi
  obtain ⟨hi', hbound⟩ := hb i hi
  refine ⟨hi', ?_⟩
  have hδ : 1 / (2 * q32) < toReal a := rms_domain_toReal_gt row eps a hdom
  have hmean : |toReal a - (meanSqReal row + toReal eps)| ≤ 1 / (2 * q32) := by
    have haR : toReal a = (rmsMean row : ℝ) / q32 + toReal eps := by
      unfold toReal
      rw [ha]
      push_cast
      ring
    rw [haR]
    rw [show (rmsMean row : ℝ) / q32 + toReal eps - (meanSqReal row + toReal eps) =
        (rmsMean row : ℝ) / q32 - meanSqReal row from by ring]
    exact rmsMean_real_bound row hne
  exact rms_ideal_element_bound row eps a (row.val[i]'hi) (out.val[i]'hi') hδ hmean hbound

end

end apeiron_word
