import Apeiron.Physics.Kepler.Root
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Tactic

/-!
# The constructed real solution of Kepler's equation

The map sending a candidate `x` to `m + e * sin x` is a contraction on the
complete real line when `|e| < 1`. Its fixed point exists without assuming a
solution of Kepler's equation. For a circle the fixed point is the mean value.
The squared fixed-point residual has the same unique zero; it is not an
absolute-residual error bound.

## Root

| from | here |
|---|---|
| `Trichotomy.flow_neg_one`, `Kepler.equation_iff_flow` | `map_eq_flow`, `anomaly_solves_flow` |
| `Kepler.flow_snd_neg_one`, real sine Lipschitz | `map_lipschitz` |
| `Kepler.equation_iff_flow`, `map_contracting` | `anomaly_exists_unique` |

The nonlinear sine Lipschitz bound requires `ℝ`, not a generic commutative ring.
The trichotomy edge is the second projection at the normalized elliptic face.
-/

noncomputable section

namespace Apeiron.Physics.Kepler

/-- The fixed-point map for real Kepler data.

    (m e x : ℝ)                   [mean value, coefficient and candidate]
    : ℝ                            [the next real fixed-point value] -/
def keplerMap (m e x : ℝ) : ℝ := m + e * Real.sin x

/-- The map's sine is the second slot of the elliptic flow.

    (m e x : ℝ)                                  [mean value, coefficient and candidate]
    keplerMap m e x                              [the fixed-point map's value]
    = m + e * (Trichotomy.flow (-1) x).2         [the same value read from the flow]
    flow                                          [the size-one flow of the algebra] -/
theorem map_eq_flow (m e x : ℝ) :
    keplerMap m e x = m + e * (Trichotomy.flow (-1) x).2 := by
  rw [keplerMap, flow_snd_neg_one]

/-- A solution is exactly a fixed point, at every real coefficient.

    (m e x : ℝ)                      [mean value, coefficient and candidate]
    m = x - e * Real.sin x          [the Kepler equation]
    ↔                               [exactly when, true in both directions]
    keplerMap m e x = x             [the fixed-point equation] -/
theorem equation_iff_fixed (m e x : ℝ) :
    m = x - e * Real.sin x ↔ keplerMap m e x = x := by
  rw [equation_iff_flow, map_eq_flow]
  constructor <;> intro h <;> linarith

/-- The map shrinks distances by at most the absolute coefficient.

    (m e x y : ℝ)                               [mean value, coefficient and two inputs]
    |keplerMap m e x - keplerMap m e y|         [the distance between outputs]
    ≤ |e| * |x - y|                             [the coefficient times input distance]
    ≤                                            [is smaller than or equal to] -/
theorem map_lipschitz (m e x y : ℝ) :
    |keplerMap m e x - keplerMap m e y| ≤ |e| * |x - y| := by
  have hs := Real.abs_sin_sub_sin_le x y
  calc
    |keplerMap m e x - keplerMap m e y|
        = |e| * |Real.sin x - Real.sin y| := by
            have hid : keplerMap m e x - keplerMap m e y =
                e * (Real.sin x - Real.sin y) := by simp [keplerMap]; ring
            rw [hid, abs_mul]
    _ ≤ |e| * |x - y| := mul_le_mul_of_nonneg_left hs (abs_nonneg e)

/-- Distinct inputs move strictly closer when the absolute coefficient is below one.

    (m e x y : ℝ)                 [mean value, coefficient and two inputs]
    (he : |e| < 1)               [the contraction factor is below one]
    (hxy : x ≠ y)                [the inputs differ]
    |keplerMap m e x - keplerMap m e y| < |x - y|
                                   [the outputs are strictly closer]
    ≠                              [is not equal to]
    <                              [is smaller than] -/
theorem map_strict (m e x y : ℝ) (he : |e| < 1) (hxy : x ≠ y) :
    |keplerMap m e x - keplerMap m e y| < |x - y| := by
  have hp : 0 < |x - y| := abs_pos.mpr (sub_ne_zero.mpr hxy)
  have hl := map_lipschitz m e x y
  nlinarith

/-- The complete real line carries this map as a contraction.

    (m e : ℝ)                                 [mean value and coefficient]
    (he : |e| < 1)                            [the absolute coefficient is below one]
    ContractingWith ⟨|e|, abs_nonneg e⟩
      (keplerMap m e)                         [the map is contracting]
    <                                         [is smaller than]
    ℝ                                        [the real numbers: every number on the number line] -/
theorem map_contracting (m e : ℝ) (he : |e| < 1) :
    ContractingWith ⟨|e|, abs_nonneg e⟩ (keplerMap m e) := by
  refine ⟨?_, ?_⟩
  · exact_mod_cast he
  · exact LipschitzWith.of_dist_le_mul fun x y => by
      change |keplerMap m e x - keplerMap m e y| ≤ |e| * |x - y|
      exact map_lipschitz m e x y

/-- The fixed point constructed from the contraction on the complete real line.

    (m e : ℝ)                [mean value and coefficient]
    (he : |e| < 1)           [the absolute coefficient is below one]
    : ℝ                       [the constructed real eccentric anomaly] -/
noncomputable def eccentricAnomaly (m e : ℝ) (he : |e| < 1) : ℝ :=
  (map_contracting m e he).fixedPoint (keplerMap m e)

/-- The constructed anomaly is a fixed point of the map.

    (m e : ℝ)                          [mean value and coefficient]
    (he : |e| < 1)                     [the absolute coefficient is below one]
    keplerMap m e (eccentricAnomaly m e he)
                                         [the map at the constructed anomaly]
    = eccentricAnomaly m e he           [the constructed anomaly itself] -/
theorem anomaly_fixed (m e : ℝ) (he : |e| < 1) :
    keplerMap m e (eccentricAnomaly m e he) = eccentricAnomaly m e he :=
  (map_contracting m e he).fixedPoint_isFixedPt

/-- The constructed fixed point solves the real Kepler equation.

    (m e : ℝ)                               [mean value and coefficient]
    (he : |e| < 1)                          [the absolute coefficient is below one]
    m = eccentricAnomaly m e he - e * Real.sin (eccentricAnomaly m e he)
                                             [the constructed anomaly solves the equation] -/
theorem anomaly_solves (m e : ℝ) (he : |e| < 1) :
    m = eccentricAnomaly m e he - e * Real.sin (eccentricAnomaly m e he) :=
  (equation_iff_fixed m e _).mpr (anomaly_fixed m e he)

/-- The constructed point also solves the equation through the elliptic flow.

    (m e : ℝ)                               [mean value and coefficient]
    (he : |e| < 1)                          [the absolute coefficient is below one]
    m = eccentricAnomaly m e he - e *
      (Trichotomy.flow (-1) (eccentricAnomaly m e he)).2
                                             [the equation read on the elliptic flow]
    flow                                     [the size-one flow of the algebra] -/
theorem anomaly_solves_flow (m e : ℝ) (he : |e| < 1) :
    m = eccentricAnomaly m e he - e *
      (Trichotomy.flow (-1) (eccentricAnomaly m e he)).2 :=
  (equation_iff_flow m e _).mp (anomaly_solves m e he)

/-- Two fixed points of a strict contraction coincide.

    (m e x y : ℝ)                  [mean value, coefficient and two candidates]
    (he : |e| < 1)                [the absolute coefficient is below one]
    (hx : keplerMap m e x = x)    [the first point is fixed]
    (hy : keplerMap m e y = y)    [the second point is fixed]
    x = y                         [they are the same point] -/
theorem fixed_unique (m e x y : ℝ) (he : |e| < 1)
    (hx : keplerMap m e x = x) (hy : keplerMap m e y = y) : x = y := by
  by_contra hxy
  have h := map_strict m e x y he hxy
  rw [hx, hy] at h
  exact (lt_irrefl _) h

/-- Kepler's equation has exactly one real solution at every `|e| < 1`.

    (m e : ℝ)                           [mean value and coefficient]
    (he : |e| < 1)                      [the absolute coefficient is below one]
    ∃! x : ℝ, m = x - e * Real.sin x   [one solution exists and every solution is it]
    ∃                                   [there is at least one]
    ℝ                                   [the real numbers: every number on the number line] -/
theorem anomaly_exists_unique (m e : ℝ) (he : |e| < 1) :
    ∃! x : ℝ, m = x - e * Real.sin x := by
  refine ⟨eccentricAnomaly m e he, anomaly_solves m e he, ?_⟩
  intro x hx
  exact fixed_unique m e x _ he ((equation_iff_fixed m e x).mp hx)
    (anomaly_fixed m e he)

/-- With zero coefficient the sole solution is the input mean value.

    (m : ℝ)                              [the real mean value]
    eccentricAnomaly m 0 (by norm_num) = m
                                         [the constructed circular solution]
    ∧                                    [both are true]
    ∀ x : ℝ, m = x - 0 * Real.sin x ↔ x = m
                                         [every solution is this point]
    ∀                                    [for every single one]
    ℝ                                    [the real numbers: every number on the number line]
    ↔                                    [exactly when, true in both directions] -/
theorem circular_solution (m : ℝ) :
    eccentricAnomaly m 0 (by norm_num) = m ∧
      (∀ x : ℝ, m = x - 0 * Real.sin x ↔ x = m) := by
  constructor
  · have hfix : keplerMap m 0 m = m := by simp [keplerMap]
    exact (fixed_unique m 0 _ _ (by norm_num) (anomaly_fixed m 0 (by norm_num)) hfix)
  · intro x
    constructor <;> intro h <;> simp_all

/-- A squared fixed-point defect, not an absolute residual bound.

    (m e x : ℝ)                       [mean value, coefficient and candidate]
    : ℝ                                 [the squared real fixed-point defect] -/
def squaredResidual (m e x : ℝ) : ℝ := (keplerMap m e x - x) ^ 2

/-- The squared defect vanishes exactly at the constructed real solution.

    (m e : ℝ)                          [mean value and coefficient]
    (he : |e| < 1)                     [the absolute coefficient is below one]
    (x : ℝ)                            [a candidate]
    squaredResidual m e x = 0         [the squared defect vanishes]
    ↔ x = eccentricAnomaly m e he      [the candidate is the constructed solution]
    ↔                                  [exactly when, true in both directions] -/
theorem squared_zero_iff (m e : ℝ) (he : |e| < 1) (x : ℝ) :
    squaredResidual m e x = 0 ↔ x = eccentricAnomaly m e he := by
  constructor
  · intro h
    have hz : keplerMap m e x - x = 0 := by
      dsimp [squaredResidual] at h
      nlinarith [sq_nonneg (keplerMap m e x - x)]
    exact fixed_unique m e x _ he (sub_eq_zero.mp hz) (anomaly_fixed m e he)
  · intro h
    rw [h]
    simp [squaredResidual, anomaly_fixed]

/-- The real circular inputs jointly inhabit the constructed solution and flow equation.

    ∃ he : |(0 : ℝ)| < 1              [the circular contraction premise holds]
    eccentricAnomaly 0 0 he = 0       [the constructed anomaly is zero]
    ∧                                 [both are true]
    (0 : ℝ) = 0 - 0 * (Trichotomy.flow (-1) 0).2
                                      [the elliptic flow equation also holds]
    ∃                                 [there is at least one]
    ℝ                                 [the real numbers: every number on the number line]
    flow                              [the size-one flow of the algebra] -/
theorem circular_joint_witness :
    ∃ he : |(0 : ℝ)| < 1,
      eccentricAnomaly 0 0 he = 0 ∧
      (0 : ℝ) = 0 - 0 * (Trichotomy.flow (-1) 0).2 := by
  refine ⟨by norm_num, ?_, ?_⟩
  · have hfix : keplerMap 0 0 0 = 0 := by simp [keplerMap]
    exact fixed_unique 0 0 _ _ (by norm_num) (anomaly_fixed 0 0 (by norm_num)) hfix
  · norm_num

/-- A positive coefficient has a joint real fixed-point and flow witness.

    ∃ e : ℝ                            [a positive real coefficient]
    ∃ he : |e| < 1                     [its strict contraction premise]
    0 < e                               [the coefficient is positive]
    ∧ eccentricAnomaly 0 e he = 0      [the constructed anomaly is zero]
    ∧ (0 : ℝ) = 0 - e * (Trichotomy.flow (-1) 0).2
                                         [the flow equation holds at that point]
    ∃                                    [there is at least one]
    ℝ                                    [the real numbers: every number on the number line]
    flow                                 [the size-one flow of the algebra] -/
theorem positive_joint_witness :
    ∃ e : ℝ, ∃ he : |e| < 1,
      0 < e ∧ eccentricAnomaly 0 e he = 0 ∧
        (0 : ℝ) = 0 - e * (Trichotomy.flow (-1) 0).2 := by
  refine ⟨1 / 2, by norm_num, by norm_num, ?_, ?_⟩
  · have hfix : keplerMap 0 (1 / 2) 0 = 0 := by simp [keplerMap]
    exact fixed_unique 0 (1 / 2) _ _ (by norm_num)
      (anomaly_fixed 0 (1 / 2) (by norm_num)) hfix
  · simp [flow_snd_neg_one]

end Apeiron.Physics.Kepler
