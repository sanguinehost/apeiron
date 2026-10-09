import Apeiron.Trichotomy.Basic
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# One flow, three faces

@name Flow
@cite kisil-2010
@cite gauss-1831
@cite cockle-1849
@grade proved

`Apeiron/Physics/Quadratic.lean` exhibits the size-one points of each face separately: the circle
below zero with the angle adding, a line at zero with the shear adding, the hyperbola above zero
with the rapidity adding. Three theorems, three faces, three proofs.

They are one theorem. Write

```
   flow σ t  =  (C, S)     with    C' = σ·S,   S' = C,   C(0) = 1,  S(0) = 0
```

and the size-one condition `C² − σS² = 1` and the group law `flow σ s · flow σ t = flow σ (s+t)`
hold at every sigma. The face only decides which pair of familiar functions the
solution is:

```
        σ < 0                     σ = 0                  σ > 0
   ─────────────────────     ───────────────────    ─────────────────────
   cos(√−σ · t)                    1                 cosh(√σ · t)
   sin(√−σ · t)/√−σ                t                 sinh(√σ · t)/√σ

   the circle                 the shear line          the hyperbola
   closes at 2π/√−σ           never returns           never returns
```

The sign of sigma is the whole difference, and it is a difference of one kind: whether the flow
comes back. Below zero it is periodic, at zero and above it is injective.
`flow_returns_iff_disc_neg` is that statement, and `Apeiron/Physics/Wick.lean` reads a
temperature off it.

## Root

Instantiates `Trichotomy.mul`, `Trichotomy.normForm` and `Trichotomy.one` from
`Apeiron/Trichotomy/Basic.lean`. The edges proved here from them:

* `normForm_flow` puts the whole flow inside the size-one locus, which `isUnit_iff_normForm_ne_zero`
  places off the cone at every setting.
* `flow_add` is the group law, and it specialises to `Physics.elliptic_one_param`,
  `Physics.parabolic_one_param` and `Physics.hyperbolic_one_param` by `flow_neg_one`,
  `flow_at_zero` and `flow_one`.
* `flow_returns_iff_disc_neg` is decided by the sign of sigma alone, the same three-way split
  `trichotomy_complete` makes on zero divisors, read on periodicity instead.
-/

namespace Apeiron
namespace Trichotomy

open Real

/-- The size-one flow of the algebra, at every sigma.

    (σ t : ℝ)                [sigma, and how far along the flow]
    : Cell                   [the point the flow has reached]
    Real.sqrt                [the number that squares to give this]

    So: below zero this turns on the circle, at zero it shears along a line, above zero it runs
    out along the hyperbola. One definition, and the sign of sigma picks the face. -/
noncomputable def flow (σ t : ℝ) : Cell :=
  if σ < 0 then
    (Real.cos (Real.sqrt (-σ) * t), Real.sin (Real.sqrt (-σ) * t) / Real.sqrt (-σ))
  else if 0 < σ then
    (Real.cosh (Real.sqrt σ * t), Real.sinh (Real.sqrt σ * t) / Real.sqrt σ)
  else ((1 : ℝ), t)

/-- Below zero the flow is the turn.

    {σ : ℝ} (hσ : σ < 0) (t : ℝ)                                   [sigma below zero, and a time]
    flow σ t = (Real.cos (Real.sqrt (-σ) * t),
      Real.sin (Real.sqrt (-σ) * t) / Real.sqrt (-σ))                [the point on the circle]
    flow                                                          [the size-one flow of the algebra]
    Real.sqrt                                                   [the number that squares to this] -/
theorem flow_neg {σ : ℝ} (hσ : σ < 0) (t : ℝ) :
    flow σ t = (Real.cos (Real.sqrt (-σ) * t),
      Real.sin (Real.sqrt (-σ) * t) / Real.sqrt (-σ)) := by
  unfold flow; rw [if_pos hσ]

/-- At zero the flow is the shear.

    (t : ℝ)                              [a time]
    flow 0 t = ((1 : ℝ), t)              [the point on the shear line]
    flow                                 [the size-one flow of the algebra] -/
theorem flow_at_zero (t : ℝ) : flow 0 t = ((1 : ℝ), t) := by
  unfold flow; rw [if_neg (by norm_num), if_neg (by norm_num)]

/-- Above zero the flow is the boost.

    {σ : ℝ} (hσ : 0 < σ) (t : ℝ)                                   [sigma above zero, and a time]
    flow σ t = (Real.cosh (Real.sqrt σ * t),
      Real.sinh (Real.sqrt σ * t) / Real.sqrt σ)                    [the point on the hyperbola]
    flow                                                          [the size-one flow of the algebra]
    Real.sqrt                                                   [the number that squares to this] -/
theorem flow_pos {σ : ℝ} (hσ : 0 < σ) (t : ℝ) :
    flow σ t = (Real.cosh (Real.sqrt σ * t),
      Real.sinh (Real.sqrt σ * t) / Real.sqrt σ) := by
  unfold flow; rw [if_neg (not_lt.mpr hσ.le), if_pos hσ]

/-- At minus one the flow is the plain turn.

    (t : ℝ)                                            [a time]
    flow (-1) t = (Real.cos t, Real.sin t)             [the point on the unit circle]
    flow                                               [the size-one flow of the algebra]

    So: the elliptic face's flow with sigma fixed at its standard value. -/
theorem flow_neg_one (t : ℝ) : flow (-1) t = (Real.cos t, Real.sin t) := by
  rw [flow_neg (by norm_num : (-1 : ℝ) < 0)]
  norm_num

/-- At plus one the flow is the plain boost.

    (t : ℝ)                                            [a time]
    flow 1 t = (Real.cosh t, Real.sinh t)              [the point on the unit hyperbola]
    flow                                               [the size-one flow of the algebra]

    So: the hyperbolic face's flow with sigma fixed at its standard value. -/
theorem flow_one (t : ℝ) : flow 1 t = (Real.cosh t, Real.sinh t) := by
  rw [flow_pos (by norm_num : (0 : ℝ) < 1)]
  norm_num

/-- The flow starts at the unit.

    (σ : ℝ)                     [the parameter]
    flow σ 0 = one              [at time zero the flow is at the unit]
    flow                        [the size-one flow of the algebra]
    one                         [the point multiplication leaves alone] -/
theorem flow_start (σ : ℝ) : flow σ 0 = one := by
  rcases lt_trichotomy σ 0 with h | h | h
  · rw [flow_neg h]; simp [one]
  · rw [h, flow_at_zero]
  · rw [flow_pos h]; simp [one]

/-- **The flow keeps size one, at every sigma.**

    (σ t : ℝ)                              [sigma and a time]
    normForm σ (flow σ t) = 1              [the point the flow has reached has size one]
    normForm                               [the norm form: a point times its conjugate]
    flow                                   [the size-one flow of the algebra]

    So: `cos² + sin² = 1`, `1 = 1` and `cosh² − sinh² = 1` are one statement, read at the three
    settings. Sigma enters as the coefficient that rescales the second slot's contribution,
    and the rescaling is exactly what the generator's square supplies. -/
theorem normForm_flow (σ t : ℝ) : normForm σ (flow σ t) = 1 := by
  rcases lt_trichotomy σ 0 with h | h | h
  · have hpos : (0 : ℝ) < -σ := by linarith
    have hsq : Real.sqrt (-σ) ^ 2 = -σ := Real.sq_sqrt hpos.le
    have hne : Real.sqrt (-σ) ≠ 0 := by positivity
    rw [flow_neg h]
    simp only [normForm]
    field_simp
    nlinarith [Real.sin_sq_add_cos_sq (Real.sqrt (-σ) * t), hsq]
  · rw [h, flow_at_zero]; simp [normForm]
  · have hsq : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt h.le
    have hne : Real.sqrt σ ≠ 0 := by positivity
    rw [flow_pos h]
    simp only [normForm]
    field_simp
    nlinarith [Real.cosh_sq_sub_sinh_sq (Real.sqrt σ * t), hsq]

/-- **The flow is a one-parameter group, at every sigma.**

    (σ s t : ℝ)                                          [sigma and two times]
    mul σ (flow σ s) (flow σ t) = flow σ (s + t)         [flowing then flowing adds the times]
    mul                                                  [multiply two points, the σ way]
    flow                                                 [the size-one flow of the algebra]

    So: the angle addition formula, plain addition, and the rapidity addition formula are one
    theorem. `Physics.elliptic_one_param`, `Physics.parabolic_one_param` and
    `Physics.hyperbolic_one_param` are its three instances. -/
theorem flow_add (σ s t : ℝ) : mul σ (flow σ s) (flow σ t) = flow σ (s + t) := by
  rcases lt_trichotomy σ 0 with h | h | h
  · have hpos : (0 : ℝ) < -σ := by linarith
    have hsq : Real.sqrt (-σ) ^ 2 = -σ := Real.sq_sqrt hpos.le
    have hne : Real.sqrt (-σ) ≠ 0 := by positivity
    rw [flow_neg h, flow_neg h, flow_neg h]
    have hd : Real.sqrt (-σ) * (s + t) = Real.sqrt (-σ) * s + Real.sqrt (-σ) * t := by ring
    rw [hd, Real.cos_add, Real.sin_add]
    simp only [mul, Prod.ext_iff]
    constructor
    · field_simp
      rw [hsq]; ring
    · field_simp
      ring
  · rw [h, flow_at_zero, flow_at_zero, flow_at_zero]
    simp only [mul, Prod.ext_iff]
    constructor <;> ring
  · have hsq : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt h.le
    have hne : Real.sqrt σ ≠ 0 := by positivity
    rw [flow_pos h, flow_pos h, flow_pos h]
    have hd : Real.sqrt σ * (s + t) = Real.sqrt σ * s + Real.sqrt σ * t := by ring
    rw [hd, Real.cosh_add, Real.sinh_add]
    simp only [mul, Prod.ext_iff]
    constructor
    · field_simp
      rw [hsq]; ring
    · field_simp
      ring

/-- **The flow's velocity is the flow multiplied by the generator.**

    (σ t : ℝ)                                                   [sigma and a time]
    HasDerivAt (flow σ) (mul σ (flow σ t) ((0 : ℝ), (1 : ℝ))) t  [its rate of change]
    HasDerivAt                                                  [is changing at that rate]
    flow                                                        [the size-one flow of the algebra]
    mul                                                         [multiply two points, the σ way]

    So: the flow solves `z' = z·x` with `x` the generator, which is the differential equation the
    exponential of the generator satisfies. Sigma enters only through the generator's square,
    which is what carries the first slot's derivative back onto the second. -/
theorem hasDerivAt_flow (σ t : ℝ) :
    HasDerivAt (flow σ) (mul σ (flow σ t) ((0 : ℝ), (1 : ℝ))) t := by
  rcases lt_trichotomy σ 0 with h | h | h
  · have hpos : (0 : ℝ) < -σ := by linarith
    have hsq : Real.sqrt (-σ) ^ 2 = -σ := Real.sq_sqrt hpos.le
    have hne : Real.sqrt (-σ) ≠ 0 := by positivity
    have hin : HasDerivAt (fun u : ℝ => Real.sqrt (-σ) * u) (Real.sqrt (-σ)) t := by
      simpa using (hasDerivAt_id t).const_mul (Real.sqrt (-σ))
    have hc : HasDerivAt (fun u : ℝ => Real.cos (Real.sqrt (-σ) * u))
        (-Real.sin (Real.sqrt (-σ) * t) * Real.sqrt (-σ)) t :=
      (Real.hasDerivAt_cos _).comp t hin
    have hs : HasDerivAt (fun u : ℝ => Real.sin (Real.sqrt (-σ) * u) / Real.sqrt (-σ))
        (Real.cos (Real.sqrt (-σ) * t) * Real.sqrt (-σ) / Real.sqrt (-σ)) t :=
      ((Real.hasDerivAt_sin _).comp t hin).div_const _
    have hfun : (fun u : ℝ => (Real.cos (Real.sqrt (-σ) * u),
        Real.sin (Real.sqrt (-σ) * u) / Real.sqrt (-σ))) = flow σ := by
      funext u; rw [flow_neg h]
    have hval : mul σ (flow σ t) ((0 : ℝ), (1 : ℝ))
        = (-Real.sin (Real.sqrt (-σ) * t) * Real.sqrt (-σ),
           Real.cos (Real.sqrt (-σ) * t) * Real.sqrt (-σ) / Real.sqrt (-σ)) := by
      rw [flow_neg h]
      simp only [mul, Prod.ext_iff]
      constructor
      · field_simp
        rw [hsq]; ring
      · field_simp
        ring
    rw [hval, ← hfun]
    exact hc.prodMk hs
  · subst h
    have hfun : (fun u : ℝ => ((1 : ℝ), u)) = flow 0 := by funext u; rw [flow_at_zero]
    have hval : mul 0 (flow 0 t) ((0 : ℝ), (1 : ℝ)) = ((0 : ℝ), (1 : ℝ)) := by
      rw [flow_at_zero]
      simp only [mul, Prod.ext_iff]
      constructor <;> ring
    rw [hval, ← hfun]
    exact (hasDerivAt_const t (1 : ℝ)).prodMk (hasDerivAt_id t)
  · have hsq : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt h.le
    have hne : Real.sqrt σ ≠ 0 := by positivity
    have hin : HasDerivAt (fun u : ℝ => Real.sqrt σ * u) (Real.sqrt σ) t := by
      simpa using (hasDerivAt_id t).const_mul (Real.sqrt σ)
    have hc : HasDerivAt (fun u : ℝ => Real.cosh (Real.sqrt σ * u))
        (Real.sinh (Real.sqrt σ * t) * Real.sqrt σ) t :=
      (Real.hasDerivAt_cosh _).comp t hin
    have hs : HasDerivAt (fun u : ℝ => Real.sinh (Real.sqrt σ * u) / Real.sqrt σ)
        (Real.cosh (Real.sqrt σ * t) * Real.sqrt σ / Real.sqrt σ) t :=
      ((Real.hasDerivAt_sinh _).comp t hin).div_const _
    have hfun : (fun u : ℝ => (Real.cosh (Real.sqrt σ * u),
        Real.sinh (Real.sqrt σ * u) / Real.sqrt σ)) = flow σ := by
      funext u; rw [flow_pos h]
    have hval : mul σ (flow σ t) ((0 : ℝ), (1 : ℝ))
        = (Real.sinh (Real.sqrt σ * t) * Real.sqrt σ,
           Real.cosh (Real.sqrt σ * t) * Real.sqrt σ / Real.sqrt σ) := by
      rw [flow_pos h]
      simp only [mul, Prod.ext_iff]
      constructor
      · field_simp
        rw [hsq]; ring
      · field_simp
        ring
    rw [hval, ← hfun]
    exact hc.prodMk hs

/-- The first slot's rate of change is sigma times the second slot.

    (σ t : ℝ)                                                          [sigma and a time]
    HasDerivAt (fun u => (flow σ u).1) (σ * (flow σ t).2) t             [its rate of change]
    HasDerivAt                                                          [is changing at that rate]
    flow                                                                [the size-one flow]

    So: `hasDerivAt_flow` read on the first slot. Below zero this is the cosine losing height at
    the sine's rate; above zero it is the hyperbolic cosine gaining at the hyperbolic sine's. -/
theorem hasDerivAt_flow_fst (σ t : ℝ) :
    HasDerivAt (fun u => (flow σ u).1) (σ * (flow σ t).2) t := by
  have h := (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t (hasDerivAt_flow σ t)
  convert h using 1 <;> first | rfl | simp [mul]

/-- The second slot's rate of change is the first slot.

    (σ t : ℝ)                                                          [sigma and a time]
    HasDerivAt (fun u => (flow σ u).2) ((flow σ t).1) t                 [its rate of change]
    HasDerivAt                                                          [is changing at that rate]
    flow                                                                [the size-one flow]

    So: the same reading on the second slot, and here sigma is absent, which is why the two
    slots are not symmetric: one carries sigma and the other does not. -/
theorem hasDerivAt_flow_snd (σ t : ℝ) :
    HasDerivAt (fun u => (flow σ u).2) ((flow σ t).1) t := by
  have h := (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t (hasDerivAt_flow σ t)
  convert h using 1 <;> first | rfl | simp [mul]

/-- **The flow returns to the unit exactly below zero.**

    (σ : ℝ)                                    [the parameter]
    ∃ t : ℝ                                    [there is at least one time]
    t ≠ 0                                      [not the starting time]
    ∧                                          [both are true]
    flow σ t = one                             [at which the flow is back at the unit]
    ↔                                          [exactly when]
    σ < 0                                      [sigma sits below zero]
    flow                                       [the size-one flow of the algebra]
    one                                        [the point multiplication leaves alone]

    So: periodicity is the elliptic face. At zero the flow shears away and above
    zero it runs out along the hyperbola, and neither ever comes back. `Apeiron/Physics/Wick.lean`
    turns this into a temperature: a horizon has one only because its turned section is
    elliptic. -/
theorem flow_returns_iff_disc_neg (σ : ℝ) :
    (∃ t : ℝ, t ≠ 0 ∧ flow σ t = one) ↔ σ < 0 := by
  constructor
  · rintro ⟨t, ht, hflow⟩
    by_contra hcon
    rcases eq_or_lt_of_le (not_lt.mp hcon) with h | h
    · rw [← h, flow_at_zero] at hflow
      exact ht (congrArg Prod.snd hflow)
    · have hne : Real.sqrt σ ≠ 0 := by positivity
      rw [flow_pos h] at hflow
      have hsnd : Real.sinh (Real.sqrt σ * t) / Real.sqrt σ = 0 := congrArg Prod.snd hflow
      have hz : Real.sinh (Real.sqrt σ * t) = 0 := by
        rcases div_eq_zero_iff.mp hsnd with h1 | h1
        · exact h1
        · exact absurd h1 hne
      have := Real.sinh_eq_zero.mp hz
      exact ht (by
        rcases mul_eq_zero.mp this with h1 | h1
        · exact absurd h1 hne
        · exact h1)
  · intro hσ
    have hpos : (0 : ℝ) < -σ := by linarith
    have hne : Real.sqrt (-σ) ≠ 0 := by positivity
    refine ⟨2 * π / Real.sqrt (-σ), ?_, ?_⟩
    · positivity
    · rw [flow_neg hσ]
      have hval : Real.sqrt (-σ) * (2 * π / Real.sqrt (-σ)) = 2 * π := by field_simp
      rw [hval]
      simp [one, Real.cos_two_pi, Real.sin_two_pi]

end Trichotomy
end Apeiron
