import Apeiron.Trichotomy.Flow
import Apeiron.Physics.Quadratic
import Apeiron.Physics.WaveAnyRing

/-!
# Imaginary time is σ negated, and closure is a temperature

@name Wick
@cite herranz-ortega-santander-2000
@cite baldazzi-percacci-skrinjar-2018
@cite gibbons-hawking-1977
@cite kisil-2010
@grade proved

Physics calls the substitution `t ↦ i·t` a Wick rotation, and uses it to trade a Lorentzian
interval `a² − b²` for a Euclidean one `a² + b²`. Written on the algebra, that substitution is `σ`
negated:

```
   a² − σ·(i·b)²  =  a² + σ·b²  =  normForm (−σ) (a, b)
```

The `i` and the `σ` meet in one place, the `σb²` term, and the `i²` there is the whole of the
rotation. `wick_flips_the_sign` is that line. It is an identity between a complexified norm form
at one setting and the real norm form at the negated setting, so the elliptic and hyperbolic
faces are two readings of one analytic family rather than two algebras that happen to look alike.

The same swap runs at the level of the flow. `Trichotomy.flow` is one function of `σ`, and
`flow_wick` says the elliptic flow at real time is the hyperbolic flow's own formula at imaginary
time: `cos θ = cosh(iθ)`, `sin θ · i = sinh(iθ)`. So the rotor and the boost are one flow, and
`Physics.elliptic_one_param` and `Physics.hyperbolic_one_param` are its two readings.

## What closure buys

There is one thing the two faces do not share, and it is the whole of what follows.
`Trichotomy.flow_returns_iff_disc_neg` says the flow comes back to the unit at a nonzero time
exactly below zero. Above zero and at zero it never returns.

A horizon of surface gravity `κ`, rotated, is a flow of rate `κ` in imaginary time. Its centre is
a regular point rather than a cone point exactly when the imaginary-time period `β` carries the
flow once around, and once around is the smallest positive `t` with `flow (−1) t = one`, which is
`2π`. So `κ·β = 2π`, `β = 2π/κ`, and the inverse period is `κ/2π`. The `2π` is read off the
elliptic face's closure, not written into a definition: `Real.cos_eq_one_iff` supplies it.

The predecessor recorded exactly this as an open seam.
`../sanguine-ref/Substrate/Algebra/Trinity/MonadRadiation.lean` wrote `2 * π` into the definition
of its smoothness condition and put the rotor's closure beside it, and its own warrant says "no
theorem joins the two". `smoothness_fixes_temperature` joins them.

## Root

The elliptic face, identified by `Trichotomy.exists_sqrt_neg_one_iff_disc_neg` and by
`Trichotomy.flow_returns_iff_disc_neg`, both of which hold exactly at `σ < 0`. The edges:

* `wick_flips_the_sign` specialises `Physics.ringNormForm_wick`
  (`Apeiron/Physics/WaveAnyRing.lean`) at `ℂ` and `Complex.I`, so the map between the faces is the
  generic sign flip of `normForm`, not a change of convention. The complex numbers and the `i`
  that supplies the squaring hypothesis are the analytic realization.
* `flow_wick` carries `Trichotomy.flow_neg_one` and `Trichotomy.flow_one` across that map, which is
  `Physics.elliptic_one_param` and `Physics.hyperbolic_one_param` joined.
* `smoothness_fixes_temperature` rests on `Trichotomy.flow_returns_iff_disc_neg` for existence and
  on `Real.cos_eq_one_iff` for the value, and `no_period_of_disc_nonneg` is the same theorem read
  on the other two faces, where there is no period and so no temperature.
-/

namespace Apeiron
namespace Physics

open Trichotomy Real

/-- **Imaginary time is σ negated.**

    (σ : ℝ) (z : Cell)                                             [the parameter and a point]
    ((z.1 : ℂ)) ^ 2 - (σ : ℂ) * ((z.2 : ℂ) * Complex.I) ^ 2        [the size, second slot turned]
    = ((normForm (-σ) z : ℝ) : ℂ)                                  [the size at σ negated]
    normForm                                                       [the norm form of a point]
    Cell                                                           [a point: two reals side by side]
    ℂ                                                              [pairs of reals, second squaring
                                                                    to −1]

    So: putting an `i` on the second slot and negating `σ` are the same operation. The
    Lorentzian interval and the Euclidean one are one function of `σ` read at two signs. -/
theorem wick_flips_the_sign (σ : ℝ) (z : Cell) :
    ((z.1 : ℂ)) ^ 2 - (σ : ℂ) * ((z.2 : ℂ) * Complex.I) ^ 2 = ((normForm (-σ) z : ℝ) : ℂ) := by
  have h := ringNormForm_wick (R := ℂ) Complex.I_sq (σ : ℂ) ((z.1 : ℂ), (z.2 : ℂ))
  have hcast : ((normForm (-σ) z : ℝ) : ℂ)
      = AnyRing.ringNormForm (-(σ : ℂ)) ((z.1 : ℂ), (z.2 : ℂ)) := by
    simp only [normForm, AnyRing.ringNormForm]
    push_cast
    ring
  rw [hcast]
  exact h

/-- **The turn is the boost at imaginary time.**

    {σ : ℝ} (hσ : 0 < σ) (t : ℝ)                                     [σ above zero, and a time]
    (((flow (-σ) t).1 : ℝ) : ℂ)                                      [the turn's first slot]
    = Complex.cosh ((Real.sqrt σ : ℂ) * ((t : ℂ) * Complex.I))      [the boost's, at imaginary time]
    ∧                                                                [both are true]
    (((flow (-σ) t).2 : ℝ) : ℂ) * Complex.I                         [the turn's second slot, turned]
    = Complex.sinh ((Real.sqrt σ : ℂ) * ((t : ℂ) * Complex.I))
        / (Real.sqrt σ : ℂ)                                         [the boost's, at imaginary time]
    flow                                                          [the size-one flow of the algebra]
    Real.sqrt                                                      [the number that squares to this]
    ℂ                                                               [pairs of reals, second squaring
                                                                      to −1]

    So: `cos θ = cosh(iθ)` and `i·sin θ = sinh(iθ)`, read as the statement that the elliptic and
    hyperbolic one-parameter subgroups are one flow at two arguments. The rotor and the boost are
    not analogues. -/
theorem flow_wick {σ : ℝ} (hσ : 0 < σ) (t : ℝ) :
    (((flow (-σ) t).1 : ℝ) : ℂ)
        = Complex.cosh ((Real.sqrt σ : ℂ) * ((t : ℂ) * Complex.I))
      ∧ (((flow (-σ) t).2 : ℝ) : ℂ) * Complex.I
        = Complex.sinh ((Real.sqrt σ : ℂ) * ((t : ℂ) * Complex.I)) / (Real.sqrt σ : ℂ) := by
  have hneg : (-σ : ℝ) < 0 := by linarith
  have hnn : -(-σ) = σ := by ring
  have hre : Real.sqrt (-(-σ)) = Real.sqrt σ := by rw [hnn]
  have harg : (Real.sqrt σ : ℂ) * ((t : ℂ) * Complex.I)
      = ((Real.sqrt σ * t : ℝ) : ℂ) * Complex.I := by push_cast; ring
  rw [flow_neg hneg t, hre]
  constructor
  · rw [harg, Complex.cosh_mul_I]
    push_cast
    ring
  · rw [harg, Complex.sinh_mul_I]
    have hne : (Real.sqrt σ : ℂ) ≠ 0 := by
      simpa using (Real.sqrt_ne_zero'.mpr hσ)
    push_cast
    field_simp

/-- The flow at minus one is the turn `Physics/Quadratic.lean` already uses.

    (t : ℝ)                          [a time]
    flow (-1) t = rot t              [the two names give the same point]
    flow                             [the size-one flow of the algebra]
    rot                              [the point on the unit circle at that angle]

    So: `elliptic_one_param` is `flow_add` at `σ = −1`, and the three one-parameter subgroups of
    `Physics/Quadratic.lean` are one theorem read at three settings. -/
theorem flow_neg_one_eq_rot (t : ℝ) : flow (-1) t = rot t := by
  rw [flow_neg_one, rot]

/-- **On the two faces at or above zero the flow never returns, so nothing fixes a period.**

    {σ : ℝ} (hσ : 0 ≤ σ) {t : ℝ} (ht : t ≠ 0)     [σ at or above zero, and a nonzero time]
    flow σ t ≠ one                                [the flow is not back at the unit]
    flow                                          [the size-one flow of the algebra]
    one                                           [the point multiplication leaves alone]

    So: the smoothness argument below has nothing to bite on at or above zero. A shear runs off
    and a boost runs out, and neither closes, so no period and no temperature. -/
theorem no_period_of_disc_nonneg {σ : ℝ} (hσ : 0 ≤ σ) {t : ℝ} (ht : t ≠ 0) :
    flow σ t ≠ one := by
  intro hflow
  exact absurd ((flow_returns_iff_disc_neg σ).mp ⟨t, ht, hflow⟩) (not_lt.mpr hσ)

/-- **Smoothness at the centre fixes the period, and the period is `2π` over the rate.**

    (κ β : ℝ) (hκ : κ ≠ 0)                       [a rate and a period, the rate not zero]
    (hpos : 0 < κ * β)                           [the flow has turned somewhere]
    (hlt : κ * β < 4 * π)                        [and not yet twice round]
    (hsmooth : flow (-1) (κ * β) = one)          [and it is back at the unit: no cone point]
    β = 2 * π / κ                                [the period is fixed]
    ∧                                            [both are true]
    1 / β = κ / (2 * π)                          [so the inverse period is the rate over 2π]
    flow                                         [the size-one flow of the algebra]
    one                                          [the point multiplication leaves alone]

    So: the Hawking temperature, with its `2π` coming out of `Real.cos_eq_one_iff` rather than
    being written into a definition. The rotated horizon is a flow on the elliptic face; a regular
    centre is that flow closing; closure is `cos = 1`; and the smallest positive solution is `2π`.
    The predecessor left this as an open seam and it is now a theorem. -/
theorem smoothness_fixes_temperature (κ β : ℝ) (hκ : κ ≠ 0) (hpos : 0 < κ * β)
    (hlt : κ * β < 4 * π) (hsmooth : flow (-1) (κ * β) = one) :
    β = 2 * π / κ ∧ 1 / β = κ / (2 * π) := by
  have hcos : Real.cos (κ * β) = 1 := by
    rw [flow_neg_one] at hsmooth
    exact congrArg Prod.fst hsmooth
  obtain ⟨n, hn⟩ := (Real.cos_eq_one_iff (κ * β)).mp hcos
  have hpi : (0 : ℝ) < π := Real.pi_pos
  have hn1 : (n : ℝ) = 1 := by
    have hlow : (0 : ℝ) < (n : ℝ) := by
      by_contra hc
      nlinarith [hn, hpos, not_lt.mp hc]
    have hhigh : (n : ℝ) < 2 := by
      by_contra hc
      nlinarith [hn, hlt, not_lt.mp hc]
    have : (0 : ℤ) < n := by exact_mod_cast hlow
    have : n < 2 := by exact_mod_cast hhigh
    have : n = 1 := by omega
    exact_mod_cast congrArg (fun m : ℤ => (m : ℝ)) this
  rw [hn1] at hn
  have hβ : β = 2 * π / κ := by
    field_simp
    linarith [hn]
  refine ⟨hβ, ?_⟩
  rw [hβ, one_div_div]

/-- **The flow solves the generator's equation of motion, acting on any point.**

    σ                                                        [the parameter]
    ℝ                                                        [the real numbers]
    HasDerivAt                                               [is changing at that rate]
    flow                                                     [the size-one flow of the algebra]
    mul                                                      [multiply two points, the σ way]
    Cell                                                     [a point: two reals side by side]

    So: multiplying by a point is the grid `rep σ` acting on a vector (`Physics.rep`,
    `Physics.rep_mul`), so this is the linear equation of motion `x' = A·x` with `A` the generator,
    solved by the one-parameter subgroup at every value of `σ`. `Physics.det_generator`
    fixes `σ = −ω²`, so the circle is the oscillator's flow, the line the free particle's
    and the hyperbola the inverted one's, and the reading `Physics/Quadratic.lean` left outside
    its theorems is now inside one. What is still outside is the same statement written on
    matrices rather than on the vector, which needs `Matrix.exp` and a norm on the grid space. -/
theorem flow_solves_the_generator_ode (σ : ℝ) (w : Cell) (t : ℝ) :
    HasDerivAt (fun u => mul σ (flow σ u) w)
      (mul σ (mul σ (flow σ t) ((0 : ℝ), (1 : ℝ))) w) t := by
  have hC := hasDerivAt_flow_fst σ t
  have hS := hasDerivAt_flow_snd σ t
  have h1 : HasDerivAt (fun u => (flow σ u).1 * w.1 + σ * (flow σ u).2 * w.2)
      (σ * (flow σ t).2 * w.1 + σ * (flow σ t).1 * w.2) t := by
    have := (hC.mul_const w.1).add (((hS.const_mul σ).mul_const w.2))
    convert this using 1 <;> rfl
  have h2 : HasDerivAt (fun u => (flow σ u).1 * w.2 + (flow σ u).2 * w.1)
      (σ * (flow σ t).2 * w.2 + (flow σ t).1 * w.1) t := by
    have := (hC.mul_const w.2).add (hS.mul_const w.1)
    convert this using 1 <;> rfl
  have hpair := h1.prodMk h2
  have hval : mul σ (mul σ (flow σ t) ((0 : ℝ), (1 : ℝ))) w
      = (σ * (flow σ t).2 * w.1 + σ * (flow σ t).1 * w.2,
         σ * (flow σ t).2 * w.2 + (flow σ t).1 * w.1) := by
    simp only [mul, Prod.ext_iff]
    constructor <;> ring
  rw [hval]
  exact hpair

end Physics
end Apeiron
