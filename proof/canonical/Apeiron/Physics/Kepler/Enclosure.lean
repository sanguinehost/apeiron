import Apeiron.Physics.Kepler.FixedPoint

/-!
# Absolute-defect enclosure for the real Kepler root

For m = 0 and e = 0, the constructed anomaly is zero and an exact zero
defect gives zero error. A positive coefficient also admits nonzero candidate
error: the absolute real defect bounds that error after division by `1 - |e|`.
This theorem does not construct a floating-point defect or a rounded quotient.
The squared fixed-point residual has a separate unique-zero theorem.

## Root

| from | here |
|---|---|
| `Trichotomy.flow_neg_one` via `Kepler.anomaly_solves_flow` | `absolute_defect_enclosure` |

The sine Lipschitz bound is a real-analysis fact; the flow projection fixes
which trichotomy face supplies the constructed root's sine.
-/

noncomputable section

namespace Apeiron.Physics.Kepler

/-- An exact-real absolute defect encloses the constructed Kepler solution.

    (m e x d : ℝ)                       [mean value, coefficient, candidate and defect limit]
    (he : |e| < 1)                      [the absolute coefficient is below one]
    (hd : 0 ≤ d)                        [the defect limit is nonnegative]
    |x - e * Real.sin x - m| ≤ d        [the exact real defect is within the limit]
    →                                    [if the defect is bounded, so is the error]
    |x - eccentricAnomaly m e he| ≤ d / (1 - |e|)
                                         [error from the constructed root is enclosed]
    ≤                                    [is smaller than or equal to]
    <                                    [is smaller than]
    ℝ                                   [the real numbers: every number on the number line] -/
theorem absolute_defect_enclosure (m e x d : ℝ) (he : |e| < 1) (hd : 0 ≤ d) :
    |x - e * Real.sin x - m| ≤ d →
      |x - eccentricAnomaly m e he| ≤ d / (1 - |e|) := by
  intro hdefect
  let EStar := eccentricAnomaly m e he
  have hstar : m = EStar - e * Real.sin EStar := by
    simpa only [flow_snd_neg_one] using anomaly_solves_flow m e he
  have hid : x - EStar = (x - e * Real.sin x - m) +
      e * (Real.sin x - Real.sin EStar) := by
    rw [hstar]
    ring
  have hsin := Real.abs_sin_sub_sin_le x EStar
  have hscaled : |e| * |Real.sin x - Real.sin EStar| ≤ |e| * |x - EStar| :=
    mul_le_mul_of_nonneg_left hsin (abs_nonneg e)
  have htriangle : |x - EStar| ≤ |x - e * Real.sin x - m| +
      |e| * |Real.sin x - Real.sin EStar| := by
    rw [hid]
    calc
      _ ≤ |x - e * Real.sin x - m| + |e * (Real.sin x - Real.sin EStar)| :=
        abs_add_le _ _
      _ = _ := by rw [abs_mul]
  have hpos : 0 < 1 - |e| := by linarith
  apply (le_div_iff₀ hpos).2
  dsimp [EStar] at *
  have hd' : 0 ≤ d := hd
  nlinarith

/-- Circular inputs jointly satisfy the exact-real enclosure at zero defect.

    ∃ he : |(0 : ℝ)| < 1            [the circular coefficient contracts]
    ∃ (_ : (0 : ℝ) ≤ 0)            [the zero defect limit is nonnegative]
    eccentricAnomaly 0 0 he = 0    [the constructed circular root is zero]
    ∧ |(0 : ℝ) - 0 * Real.sin 0 - 0| ≤ 0
                                      [the candidate's absolute defect is zero]
    ∧ |(0 : ℝ) - eccentricAnomaly 0 0 he| ≤ 0 / (1 - |(0 : ℝ)|)
                                      [the candidate's error has the promised limit]
    ∃                                 [there is at least one]
    ∧                                 [both are true]
    ≤                                 [is smaller than or equal to]
    <                                 [is smaller than]
    ℝ                                [the real numbers: every number on the number line] -/
theorem circular_enclosure_joint_witness :
    ∃ he : |(0 : ℝ)| < 1, ∃ _ : (0 : ℝ) ≤ 0,
      eccentricAnomaly 0 0 he = 0 ∧
        |(0 : ℝ) - 0 * Real.sin 0 - 0| ≤ 0 ∧
        |(0 : ℝ) - eccentricAnomaly 0 0 he| ≤ 0 / (1 - |(0 : ℝ)|) := by
  let he : |(0 : ℝ)| < 1 := by norm_num
  let hd : (0 : ℝ) ≤ 0 := le_refl 0
  have hroot : eccentricAnomaly 0 0 he = 0 := (circular_solution 0).1
  have hdefect : |(0 : ℝ) - 0 * Real.sin 0 - 0| ≤ 0 := by norm_num
  exact ⟨he, hd, hroot, hdefect,
    absolute_defect_enclosure 0 0 0 0 he hd hdefect⟩

/-- Positive eccentricity admits a nonzero candidate with a real defect bound.

    ∃ e : ℝ                          [a positive real coefficient]
    ∃ he : |e| < 1                   [the coefficient contracts]
    ∃ d : ℝ                          [an exact real defect limit]
    ∃ (_ : 0 ≤ d)                   [the limit is nonnegative]
    0 < e                             [the coefficient is positive]
    ∧ eccentricAnomaly 0 e he = 0    [the constructed solution is zero]
    ∧ (1 : ℝ) ≠ eccentricAnomaly 0 e he
                                      [the candidate is not the solution]
    ∧ |(1 : ℝ) - e * Real.sin 1 - 0| ≤ d
                                      [the candidate's exact absolute defect is bounded]
    ∧ |(1 : ℝ) - eccentricAnomaly 0 e he| ≤ d / (1 - |e|)
                                      [the error obeys the enclosure]
    ∃                                 [there is at least one]
    ∧                                 [both are true]
    ≠                                 [is not equal to]
    ≤                                 [is smaller than or equal to]
    <                                 [is smaller than]
    ℝ                                 [the real numbers: every number on the number line] -/
theorem positive_enclosure_joint_witness :
    ∃ e : ℝ, ∃ he : |e| < 1, ∃ d : ℝ, ∃ _ : 0 ≤ d,
      0 < e ∧ eccentricAnomaly 0 e he = 0 ∧
        (1 : ℝ) ≠ eccentricAnomaly 0 e he ∧
        |(1 : ℝ) - e * Real.sin 1 - 0| ≤ d ∧
        |(1 : ℝ) - eccentricAnomaly 0 e he| ≤ d / (1 - |e|) := by
  let e : ℝ := 1 / 2
  let he : |e| < 1 := by dsimp [e]; norm_num
  let d : ℝ := |(1 : ℝ) - e * Real.sin 1 - 0|
  let hd : 0 ≤ d := abs_nonneg _
  have hroot : eccentricAnomaly 0 e he = 0 := by
    apply fixed_unique 0 e _ _ he (anomaly_fixed 0 e he)
    simp [keplerMap]
  have hdefect : |(1 : ℝ) - e * Real.sin 1 - 0| ≤ d := le_refl _
  refine ⟨e, he, d, hd, ?_, hroot, ?_, hdefect, ?_⟩
  · dsimp [e]; norm_num
  · simp [hroot]
  · exact absolute_defect_enclosure 0 e 1 d he hd hdefect

end Apeiron.Physics.Kepler
