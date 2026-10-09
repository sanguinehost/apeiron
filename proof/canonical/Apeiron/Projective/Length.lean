import Apeiron.Projective.Bulk
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Length in the half-plane

@name Arc Length
@cite ramallo-2013
@cite kisil-2010
@cite ryu-takayanagi-2006
@grade proved

`Projective/Bulk.lean` gives the half-plane a line element and proves the Möbius action
preserves it. A line element measures one displacement at one point. It is not a length, and
Ryu-Takayanagi is a length: in `AdS₃` the minimal surface is a geodesic and the entropy is its
size. This file integrates the line element along a path, and proves the two facts that make
the result a geometry rather than a definition.

## What is here

`arcLength` is the integral of the square root of the line element along a path.
`arcLength_vertical` computes it for the straight climb from height `p` to height `q` and gets
`log (q / p)`, a number. `log_height_le_arcLength` says no path between those two heights is
shorter, so the straight climb is a geodesic and that number is a distance rather than one
path's score. `vertical_minimises` puts the two together.

`boundary_at_infinite_distance` is what Ramallo states after his equation (12), and it is the
reason holography needs a cutoff. The conformal boundary is at height zero. `log (q / p)` grows
past every bound as `p` falls towards it, so the boundary is infinitely far from every interior
point and any quantity measured out to it diverges.

## What is not here

The elliptic face only. At `σ` zero or above the norm form takes negative values, so the square
root is not a length on every displacement, and the Lorentzian reading needs a causal condition
on the path before an arc length exists at all. That condition is not written here, and
`exists_null_direction_iff_disc_nonneg` says exactly which faces need it.
-/

noncomputable section

namespace Apeiron
namespace Projective

open Trichotomy MeasureTheory

/-- How fast a path moves, measured by the line element.

    (σ : ℝ) (γ : ℝ → Cell) (t : ℝ)         [sigma, a path, a time]
    : ℝ                                    [the square root of the line element at that time]
    Real.sqrt                              [the number that squares to give this]
    lineElement                            [the displacement's size over the point's height
                                            squared]
    deriv                                  [how fast the path is changing, at that time]

    So: `√(lineElement σ (γ t) (γ' t))`. The square root is what turns a quadratic form into a
    length, and it is also why this file stays below zero: above zero the form takes negative
    values and the root is not a length. -/
def speed (σ : ℝ) (γ : ℝ → Cell) (t : ℝ) : ℝ :=
  Real.sqrt (lineElement σ (γ t) (deriv γ t))

/-- The length of a path between two times.

    (σ : ℝ) (γ : ℝ → Cell) (a b : ℝ)       [sigma, a path, a start time and an end time]
    : ℝ                                    [the speed, added up over the times between]
    ∫                                      [added up continuously over]
    ..                                     [from the first time to the second]
    speed                                  [the square root of the line element at that time]

    So: the usual arc length, with the usual metric replaced by this one. -/
def arcLength (σ : ℝ) (γ : ℝ → Cell) (a b : ℝ) : ℝ :=
  ∫ t in a..b, speed σ γ t

/-- The speed, written out where the derivative is known and the height is positive.

    {σ : ℝ} {γ : ℝ → Cell} {v : Cell} {t : ℝ}      [sigma, a path, a step, a time]
    (hd : HasDerivAt γ v t)                        [the path is changing at rate v at that time]
    (hy : 0 < (γ t).2)                             [the path is above the boundary there]
    speed σ γ t = Real.sqrt (normForm σ v) / (γ t).2   [the step's root size over the height]
    Real.sqrt                                      [the number that squares to give this]
    normForm                                       [the norm form: a point times its conjugate,
                                                    which can come out negative]
    <                                              [is smaller than]

    So: the height comes out of the square root because it is positive. Everything below uses
    this form rather than `speed` itself, because a derivative supplied as a hypothesis is
    easier to work with than one extracted from the path. -/
theorem speed_eq_of_hasDerivAt {σ : ℝ} {γ : ℝ → Cell} {v : Cell} {t : ℝ}
    (hd : HasDerivAt γ v t) (hy : 0 < (γ t).2) :
    speed σ γ t = Real.sqrt (normForm σ v) / (γ t).2 := by
  rw [speed, hd.deriv, lineElement, Real.sqrt_div' _ (sq_nonneg (γ t).2),
    Real.sqrt_sq hy.le]

/-- The straight climb: a path that holds its first slot and raises its height.

    (x : ℝ)                     [where the climb stands]
    : ℝ → Cell                  [the path that sends a time to that point at that height]

    So: `t ↦ (x, t)`. In the half-plane this is the vertical line through `x`, and the two
    theorems below say it is a geodesic. -/
def vertical (x : ℝ) : ℝ → Cell := fun t => (x, t)

/-- The straight climb rises at rate one.

    (x t : ℝ)                                  [where the climb stands, and a time]
    HasDerivAt (vertical x) (0, 1) t           [it is changing at rate (0, 1) at that time]
    vertical                                   [the path that holds its first slot and rises] -/
theorem hasDerivAt_vertical (x t : ℝ) : HasDerivAt (vertical x) (0, 1) t :=
  (hasDerivAt_const t x).prodMk (hasDerivAt_id t)

/-- The straight climb's speed is one over its height.

    {x t : ℝ} (ht : 0 < t)                             [where the climb stands, a positive time]
    speed (-1) (vertical x) t = 1 / t                  [its speed there is one over the height]
    speed                                              [the square root of the line element]
    vertical                                           [the path that holds its slot and rises]
    <                                                  [is smaller than]

    So: the climb moves at rate one and the metric divides by the height, so high up it is slow.
    That is the whole of the hyperbolic plane in one line. -/
theorem speed_vertical {x t : ℝ} (ht : 0 < t) : speed (-1) (vertical x) t = 1 / t := by
  rw [speed_eq_of_hasDerivAt (hasDerivAt_vertical x t) (by simpa [vertical] using ht)]
  simp only [vertical, normForm]
  norm_num

/-- The straight climb from one height to another has length the log of their ratio.

    {x p q : ℝ} (hp : 0 < p) (hpq : p ≤ q)             [where it stands, and two heights]
    arcLength (-1) (vertical x) p q = Real.log (q / p) [its length is the log of the ratio]
    arcLength                                          [the speed, added up over the times]
    vertical                                           [the path that holds its slot and rises]
    Real.log                                           [the power e must be raised to for this]
    <                                                  [is smaller than]
    ≤                                                  [is smaller than or equal to]

    So: a number, not a formula. Together with `log_height_le_arcLength` it is the distance
    between the two heights, and it is what diverges as the lower height falls to the
    boundary. -/
theorem arcLength_vertical {x p q : ℝ} (hp : 0 < p) (hpq : p ≤ q) :
    arcLength (-1) (vertical x) p q = Real.log (q / p) := by
  have hzero : (0 : ℝ) ∉ Set.uIcc p q := by
    rw [Set.uIcc_of_le hpq]
    exact fun h => absurd h.1 (not_le.mpr hp)
  have hcongr : Set.EqOn (speed (-1) (vertical x)) (fun t => 1 / t) (Set.uIcc p q) := by
    intro t ht
    rw [Set.uIcc_of_le hpq] at ht
    exact speed_vertical (lt_of_lt_of_le hp ht.1)
  rw [arcLength, intervalIntegral.integral_congr hcongr, integral_one_div hzero]

/-- No path between two heights is shorter than the log of their ratio.

    {γ γ' : ℝ → Cell} {a b : ℝ} (hab : a ≤ b)     [a path, its rate, a start and an end]
    (hd : ∀ t ∈ Set.uIcc a b, HasDerivAt γ (γ' t) t)   [the rate is the path's rate throughout]
    (hy : ∀ t ∈ Set.uIcc a b, 0 < (γ t).2)             [the path stays above the boundary]
    (hs : IntervalIntegrable (speed (-1) γ) volume a b)          [its speed can be added up]
    (hr : IntervalIntegrable (fun t => (γ' t).2 / (γ t).2) volume a b)  [so can its climb rate]
    |Real.log ((γ b).2 / (γ a).2)| ≤ arcLength (-1) γ a b   [it is at least the log of the ratio]
    ∀                                             [for every single one]
    ∈                                             [is one of the things in]
    Set.uIcc                                      [every number between the two, either order]
    HasDerivAt                                    [is changing at that rate at that time]
    IntervalIntegrable                            [can be added up over that stretch]
    volume                                        [ordinary length, as a way of adding up]
    speed                                         [the square root of the line element]
    arcLength                                     [the speed, added up over the times]
    Real.log                                      [the power e must be raised to for this]
    ≤                                             [is smaller than or equal to]
    <                                             [is smaller than]

    So: the argument is three inequalities. The speed is at least the climb rate, because the
    norm form below zero is a sum of two squares and dropping one shrinks it. The integral of a
    size is at least the size of the integral. And the integral of the climb rate is the
    difference of the logs, by the fundamental theorem, because the climb rate is exactly the
    derivative of the log of the height.

    Nothing here is about the straight climb, so the bound holds for every path, which is what
    makes `arcLength_vertical` a distance. -/
theorem log_height_le_arcLength {γ γ' : ℝ → Cell} {a b : ℝ} (hab : a ≤ b)
    (hd : ∀ t ∈ Set.uIcc a b, HasDerivAt γ (γ' t) t)
    (hy : ∀ t ∈ Set.uIcc a b, 0 < (γ t).2)
    (hs : IntervalIntegrable (speed (-1) γ) volume a b)
    (hr : IntervalIntegrable (fun t => (γ' t).2 / (γ t).2) volume a b) :
    |Real.log ((γ b).2 / (γ a).2)| ≤ arcLength (-1) γ a b := by
  have hmem : ∀ t ∈ Set.Icc a b, t ∈ Set.uIcc a b := by
    intro t ht
    rw [Set.uIcc_of_le hab]
    exact ht
  have hlog : ∀ t ∈ Set.uIcc a b,
      HasDerivAt (fun s => Real.log ((γ s).2)) ((γ' t).2 / (γ t).2) t := by
    intro t ht
    have h2 : HasDerivAt (fun s => (γ s).2) (γ' t).2 t :=
      ((ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt).comp_hasDerivAt t (hd t ht)
    exact h2.log (ne_of_gt (hy t ht))
  have hftc : (∫ t in a..b, (γ' t).2 / (γ t).2)
      = Real.log ((γ b).2) - Real.log ((γ a).2) :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt hlog hr
  have hle : ∀ t ∈ Set.Icc a b, |(γ' t).2 / (γ t).2| ≤ speed (-1) γ t := by
    intro t ht
    have htu := hmem t ht
    have hyt := hy t htu
    rw [speed_eq_of_hasDerivAt (hd t htu) hyt, abs_div, abs_of_pos hyt]
    have hnum : |(γ' t).2| ≤ Real.sqrt (normForm (-1) (γ' t)) := by
      rw [← Real.sqrt_sq_eq_abs]
      apply Real.sqrt_le_sqrt
      simp only [normForm]
      nlinarith [sq_nonneg (γ' t).1]
    gcongr
  calc |Real.log ((γ b).2 / (γ a).2)|
      = |Real.log ((γ b).2) - Real.log ((γ a).2)| := by
        rw [Real.log_div (ne_of_gt (hy b (Set.right_mem_uIcc)))
          (ne_of_gt (hy a (Set.left_mem_uIcc)))]
    _ = |∫ t in a..b, (γ' t).2 / (γ t).2| := by rw [hftc]
    _ ≤ ∫ t in a..b, |(γ' t).2 / (γ t).2| := intervalIntegral.abs_integral_le_integral_abs hab
    _ ≤ ∫ t in a..b, speed (-1) γ t := intervalIntegral.integral_mono_on hab hr.abs hs hle
    _ = arcLength (-1) γ a b := rfl

/-- The straight climb is a shortest path between its two heights.

    {γ γ' : ℝ → Cell} {x p q a b : ℝ}                  [a path, its rate, a climb, four numbers]
    (hp : 0 < p) (hpq : p ≤ q) (hab : a ≤ b)           [the heights and the times are in order]
    (hd : ∀ t ∈ Set.uIcc a b, HasDerivAt γ (γ' t) t)   [the rate is the path's rate throughout]
    (hy : ∀ t ∈ Set.uIcc a b, 0 < (γ t).2)             [the path stays above the boundary]
    (hs : IntervalIntegrable (speed (-1) γ) volume a b)          [its speed can be added up]
    (hr : IntervalIntegrable (fun t => (γ' t).2 / (γ t).2) volume a b)  [so can its climb rate]
    (ha : (γ a).2 = p) (hb : (γ b).2 = q)              [it runs between the two heights]
    arcLength (-1) (vertical x) p q ≤ arcLength (-1) γ a b   [the climb is no longer than it]
    ∀                                                  [for every single one]
    ∈                                                  [is one of the things in]
    Set.uIcc                                           [every number between the two]
    HasDerivAt                                         [is changing at that rate at that time]
    IntervalIntegrable                                 [can be added up over that stretch]
    volume                                             [ordinary length, as a way of adding up]
    speed                                              [the square root of the line element]
    arcLength                                          [the speed, added up over the times]
    vertical                                           [the path that holds its slot and rises]
    ≤                                                  [is smaller than or equal to]
    <                                                  [is smaller than]

    So: this is what makes the half-plane a geometry rather than a space with a form on it.
    A distance is the least length over all paths, and here that least is attained and computed.
    The first slot of the climb is free, which is the statement that only the heights matter. -/
theorem vertical_minimises {γ γ' : ℝ → Cell} {x p q a b : ℝ}
    (hp : 0 < p) (hpq : p ≤ q) (hab : a ≤ b)
    (hd : ∀ t ∈ Set.uIcc a b, HasDerivAt γ (γ' t) t)
    (hy : ∀ t ∈ Set.uIcc a b, 0 < (γ t).2)
    (hs : IntervalIntegrable (speed (-1) γ) volume a b)
    (hr : IntervalIntegrable (fun t => (γ' t).2 / (γ t).2) volume a b)
    (ha : (γ a).2 = p) (hb : (γ b).2 = q) :
    arcLength (-1) (vertical x) p q ≤ arcLength (-1) γ a b := by
  rw [arcLength_vertical hp hpq]
  have h := log_height_le_arcLength hab hd hy hs hr
  rw [ha, hb] at h
  exact le_trans (le_abs_self _) h

/-- The boundary is infinitely far away.

    (x q M : ℝ) (hq : 0 < q)                           [a climb, a height, a bound]
    ∃ p : ℝ, 0 < p ∧ p ≤ q ∧ M < arcLength (-1) (vertical x) p q
                                                       [some lower height puts the climb past M]
    ∃                                                  [there is at least one]
    ∧                                                  [both are true]
    arcLength                                          [the speed, added up over the times]
    vertical                                           [the path that holds its slot and rises]
    <                                                  [is smaller than]
    ≤                                                  [is smaller than or equal to]

    So: the conformal boundary of the half-plane is height zero, and no point of the half-plane
    is a finite distance from it. This is what Ramallo records after his equation (12) when he
    says the metric is singular at the boundary and needs regularising, and it is why a
    holographic entropy is computed to a cutoff rather than to the boundary. The witness is
    `q · e^(−(|M| + 1))`, so the bound is not approached, it is passed at a stated height. -/
theorem boundary_at_infinite_distance (x q M : ℝ) (hq : 0 < q) :
    ∃ p : ℝ, 0 < p ∧ p ≤ q ∧ M < arcLength (-1) (vertical x) p q := by
  have hneg : -(|M| + 1) < 0 := by linarith [abs_nonneg M]
  have hlt : Real.exp (-(|M| + 1)) < 1 := Real.exp_lt_one_iff.mpr hneg
  have hpos : (0 : ℝ) < Real.exp (-(|M| + 1)) := Real.exp_pos _
  have hle : q * Real.exp (-(|M| + 1)) ≤ q := by nlinarith
  refine ⟨q * Real.exp (-(|M| + 1)), by positivity, hle, ?_⟩
  · rw [arcLength_vertical (by positivity) hle]
    rw [show q / (q * Real.exp (-(|M| + 1))) = Real.exp (|M| + 1) by
      rw [Real.exp_neg]
      field_simp]
    rw [Real.log_exp]
    exact lt_of_le_of_lt (le_abs_self M) (by linarith)

end Projective
end Apeiron

end
