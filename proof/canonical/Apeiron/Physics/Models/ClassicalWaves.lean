import Apeiron.Physics.Madelung
import Apeiron.Physics.Wick
import Apeiron.Physics.WaveAnyRing
import Apeiron.Trichotomy.Derived.Differential
import Apeiron.Trichotomy.Roots

/-!
# The classical residue, and the four branches that carry it

@name Classical Waves
@cite schrodinger-1926
@cite madelung-1927
@cite bohm-1952
@cite van-vleck-1928
@cite morette-1951
@cite lohmiller-slotine-2024
@cite vattay-2026-comment
@cite vattay-2026-reply
@cite feynman-hibbs-1965
@grade proved

The classical action computes the wave up to one term. `Physics.Madelung.lean`'s
`hamilton_jacobi` is that statement at every `σ`: the first slot of the wave's curvature point,
divided by the amplitude, is the amplitude's own curvature ratio `ddR/R` plus the phase-kinetic
term `σ·(dφ)²`. This file reads that equation three ways and derives the structure the reading
needs, all inside `Cell`.

**The residue is amplitude curvature.** Subtract the kinetic term and what is left of the wave's
first slot is `ddR/R`, the term physics calls the quantum potential (`residue_is_curvature`).
The classical computation is the same equation with that term missing, and it is missing exactly
where the amplitude is affine.

**One branch carries no curvature.** A point of the degenerate face is a value alongside a
first-order change, and the expansion stops there (`Trichotomy/Differential.lean`,
`value_slot_iff_disc_zero`, `dual_taylor_two`). Read onto the wave: at `σ = 0` the kinetic term
vanishes and the first slot is the bare amplitude curvature, which a first-order point has no
slot to carry (`kinetic_vanishes_at_degenerate`). The curvature must be sampled, not carried.

**The four branches are the elliptic generator's orbit.** A point squaring to minus the unit has
empty first slot and non-zero second (`sq_neg_one_implies_*`), so it is neither unit nor its
negative, and its fourth power returns to the unit (`fourth_power_of_sq_neg_one`). Its powers
`{one, z, -one, -z}` are four distinct points, the branches. The sign of `σ` sets which face
carries them (`Trichotomy/Roots.lean`, `exists_sqrt_neg_one_iff_disc_neg`).

**The weld between the faces is a kinetic sign.** The elliptic face at `σ = −1` and the
hyperbolic at `σ = 1` are two readings of one wave, and the difference between their first-slot
ratios is twice the kinetic term (`kinetic_sign_flips`). No `ℂ` appears anywhere: the rotation
`Physics/Wick.lean` carries across the faces is the negation of `σ`, an identity of `normForm`
and `flow` already in the corpus.

**Four samples recover the second-order content.** The four branches are the elliptic flow's own
quarter-turn positions (`flow_quarter_turn_points`): at `σ = −1` the flow visits
`{one, (0,1), −one, (0,−1)}` at times `0, π/2, π, 3π/2`. Sampling the algebra's quadratic
`normForm σ` at those four offsets and taking the symmetric combination isolates the trace of its
Hessian exactly, at every step `h`, with `σ` in the value
(`stencil4_normForm`): the sum of the four samples minus four times the centre is
`2·(1−σ)·h²`, the trace `2 + (−2σ)` carried by the second differences. The two corollaries read
the sign: on the split face the value vanishes (`stencil4_normForm_split`), because
`a² − b²` has no isotropic second-order content; on the elliptic face it is maximal
(`stencil4_normForm_elliptic`). The single-branch half is `Physics/ClassicalWaves.lean`'s own
`kinetic_vanishes_at_degenerate` read with `Trichotomy.dual_taylor_two`: a first-order point has
no slot for this content, so it must be sampled, and the four samples are the flow's points.

## Root

* `Physics.hamilton_jacobi` (from `Physics.secondDeriv_wave*`, `Trichotomy.flow`'s slot
  derivatives) carries `residue_is_curvature`, `kinetic_vanishes_at_degenerate`, and
  `kinetic_sign_flips`.
* `Trichotomy.mul` and `Trichotomy.one` carry `fourth_power_of_sq_neg_one` and the two
  `sq_neg_one_implies_*` structure theorems.
* `Trichotomy.value_slot_iff_disc_zero` and `Trichotomy.dual_taylor_two`
  (`Trichotomy/Differential.lean`) carry the reading of `kinetic_vanishes_at_degenerate`
  as the 1-jet face.
* `Trichotomy.exists_sqrt_neg_one_iff_disc_neg` (`Trichotomy/Roots.lean`) fixes which face
  the four branches live on.
* `Physics.flow_wick` and `Physics.wick_flips_the_sign` (`Physics/Wick.lean`) carry the
  σ-negation weld that `kinetic_sign_flips` states in first slots.
* `Trichotomy.flow_neg_one` (`Trichotomy/Flow.lean`) carries `flow_quarter_turn_points`:
  the elliptic flow at the four quarter turns is the four branches.
* `Trichotomy.normForm` and the sign of `σ` carry `stencil4_normForm` and its two face
  corollaries: the symmetric four-sample combination is an identity of the form, and the form's
  own sign is what the recovered value `2·(1−σ)` reports. The identity itself is the real
  specialisation of `Physics.ringStencil4_normForm` (`Apeiron/Physics/WaveAnyRing.lean`), which
  holds in every commutative ring because it is a polynomial identity with whole-number
  coefficients.

What is *not* claimed: a reconstruction of an arbitrary smooth amplitude from finitely many
samples. The identity here is exact on the algebra's own quadratic `normForm σ` at every step
`h`, and the general smooth case would be a `h → 0` limit of the same arithmetic, which is
Mathlib analysis and is not built.
-/

namespace Apeiron
namespace Physics

open Trichotomy
open scoped Real

/-- **Subtract the kinetic term from the wave's first slot and the amplitude curvature is left.**

@cite bohm-1952

    (σ : ℝ)                        [the parameter]
    (R dR ddR dφ ddφ : ℝ → ℝ)      [the amplitude's four functions]
    (y : ℝ)                        [a place]
    (hR : R y ≠ 0)                 [the amplitude is not zero at that place]
    (curv σ R dR ddR dφ ddφ y).1 / R y   [the wave's curvature ratio, first slot]
    - σ * dφ y ^ 2                 [minus the phase-kinetic term]
    = ddR y / R y                  [equals the amplitude's own curvature ratio]
    curv                           [the wave's curvature point]
    R                              [the amplitude: how much wave there is at each place]

    So: `hamilton_jacobi` with the kinetic term moved. The residue of the classical construction
    is the amplitude curvature alone, which is the term `Physics.Madelung.lean`'s header calls the
    quantum potential. -/
theorem residue_is_curvature (σ : ℝ) (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) (hR : R y ≠ 0) :
    (curv σ R dR ddR dφ ddφ y).1 / R y - σ * dφ y ^ 2 = ddR y / R y := by
  rw [hamilton_jacobi σ R dR ddR dφ ddφ y hR]
  ring

/-- **A point squaring to minus the unit has an empty first slot.**

    (σ : ℝ) (z : Cell)             [the parameter and the point]
    mul σ z z = -one               [the point, squared, is minus the unit]
    z.1 = 0                        [its first slot is zero]
    mul                            [multiply two points, the σ way]
    one                            [the point multiplication leaves alone]

    So: the second slot of `z·z` is `2·z.1·z.2`, and it must vanish, and the first slot is
    `z.1² + σ·z.2² = −1`, so the second slot cannot vanish and the first must. This is the first
    half of `Trichotomy/Roots.lean`'s proof, stated as a fact any such point has. -/
theorem sq_neg_one_implies_zero_first (σ : ℝ) (z : Cell) (hz : mul σ z z = -one) : z.1 = 0 := by
  obtain ⟨h1, h2⟩ := Prod.ext_iff.mp hz
  simp only [one, Prod.fst_neg, Prod.snd_neg, neg_zero] at h1 h2
  have hz2 : z.2 ≠ 0 := by
    intro h
    rw [h] at h1
    nlinarith [mul_self_nonneg z.1]
  have hprod : z.1 * z.2 = 0 := by linarith
  rcases mul_eq_zero.mp hprod with h | h
  · exact h
  · exact absurd h hz2

/-- **A point squaring to minus the unit has a non-zero second slot.**

    (σ : ℝ) (z : Cell)             [the parameter and the point]
    mul σ z z = -one               [the point, squared, is minus the unit]
    z.2 ≠ 0                        [its second slot is not zero]
    mul                            [multiply two points, the σ way]
    one                            [the point multiplication leaves alone]

    So: were the second slot zero the first would square to minus one, impossible over the
    reals. The point is not `0`, and it is not `±one`, so its powers are distinct points. -/
theorem sq_neg_one_implies_nonzero_second (σ : ℝ) (z : Cell) (hz : mul σ z z = -one) : z.2 ≠ 0 := by
  intro h
  obtain ⟨h1, _⟩ := Prod.ext_iff.mp hz
  simp only [one, Prod.fst_neg] at h1
  rw [h] at h1
  nlinarith [mul_self_nonneg z.1]

/-- **A point squaring to minus the unit returns to the unit on its fourth power.**

    (σ : ℝ) (z : Cell)                        [the parameter and the point]
    mul σ z z = -one                          [the point, squared, is minus the unit]
    mul σ (mul σ z z) (mul σ z z) = one       [squared twice more, it is the unit]
    mul                                       [multiply two points, the σ way]
    one                                       [the point multiplication leaves alone]

    So: `z⁴ = (z²)² = (−one)² = one`. The generator of the elliptic face has order four, and its
    powers `{one, z, −one, −z}` are the four branches, the rotor's orbit under its own quarter
    turns. -/
theorem fourth_power_of_sq_neg_one (σ : ℝ) (z : Cell) (hz : mul σ z z = -one) :
    mul σ (mul σ z z) (mul σ z z) = one := by
  rw [hz]
  simp only [mul, one]
  ext <;> norm_num

/-- **A point squaring to minus the unit is not the unit.**

    (σ : ℝ) (z : Cell)              [the parameter and the point]
    mul σ z z = -one                [the point, squared, is minus the unit]
    z ≠ one                         [it is not the unit]
    one                             [the point multiplication leaves alone]

    So: the unit squares to itself and this point squares to its negative, so it is not the unit.
    The branch `z` is a new point. -/
theorem sq_neg_one_implies_not_unit (σ : ℝ) (z : Cell) (hz : mul σ z z = -one) : z ≠ one := by
  intro h
  have hz2 := sq_neg_one_implies_nonzero_second σ z hz
  rw [h] at hz2
  exact hz2 rfl

/-- **A point squaring to minus the unit is not minus the unit.**

    (σ : ℝ) (z : Cell)              [the parameter and the point]
    mul σ z z = -one                [the point, squared, is minus the unit]
    z ≠ -one                        [it is not minus the unit]
    one                             [the point multiplication leaves alone]

    So: `(−one)² = one`, again different from what this point squares to. The four powers
    `{one, z, −one, −z}` are four distinct points. -/
theorem sq_neg_one_implies_not_neg_unit (σ : ℝ) (z : Cell) (hz : mul σ z z = -one) : z ≠ -one := by
  intro h
  have hz2 := sq_neg_one_implies_nonzero_second σ z hz
  apply hz2
  rw [h]
  simp [one]

/-- **At the degenerate setting the kinetic term vanishes, and the first slot is bare curvature.**

    (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ)    [the amplitude's four functions and a place]
    (hR : R y ≠ 0)                       [the amplitude is not zero there]
    (curv 0 R dR ddR dφ ddφ y).1 / R y   [the wave's curvature ratio at σ = 0]
    = ddR y / R y                        [is the amplitude's own curvature ratio alone]
    curv                                 [the wave's curvature point]

    So: `hamilton_jacobi` at `σ = 0`. On the degenerate face a point is a value with a first-order
    change and no second-order slot (`Trichotomy/Differential.lean`), and here the phase-kinetic
    coupling is likewise absent: the whole first slot is the amplitude curvature, which no
    first-order point carries. The curvature must be sampled. -/
theorem kinetic_vanishes_at_degenerate (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) (hR : R y ≠ 0) :
    (curv 0 R dR ddR dφ ddφ y).1 / R y = ddR y / R y := by
  rw [hamilton_jacobi 0 R dR ddR dφ ddφ y hR]
  ring

/-- **The two faces differ by twice the kinetic term.**

    (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ)          [the amplitude's four functions and a place]
    (hR : R y ≠ 0)                             [the amplitude is not zero there]
    (curv 1 R dR ddR dφ ddφ y).1 / R y         [the hyperbolic face's curvature ratio]
    - (curv (-1) R dR ddR dφ ddφ y).1 / R y    [minus the elliptic face's]
    = 2 * dφ y ^ 2                             [is twice the phase-kinetic term]
    curv                                       [the wave's curvature point]

    So: `hamilton_jacobi` at `σ = 1` and at `σ = −1` together. The amplitude-curvature term
    `ddR/R` cancels and only the kinetic sign remains, which is the σ-negation weld
    (`Physics/Wick.lean`) read on the wave's first slot: the elliptic and hyperbolic faces are one
    wave with the kinetic sign flipped, and the rotation that flips it is the negation of `σ`. -/
theorem kinetic_sign_flips (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) (hR : R y ≠ 0) :
    (curv 1 R dR ddR dφ ddφ y).1 / R y - (curv (-1) R dR ddR dφ ddφ y).1 / R y
      = 2 * dφ y ^ 2 := by
  rw [hamilton_jacobi 1 R dR ddR dφ ddφ y hR]
  rw [hamilton_jacobi (-1) R dR ddR dφ ddφ y hR]
  ring

/-- **The four branches are the elliptic flow's quarter-turn positions.**

    (t : ℝ)                                [a time]
    flow (-1) (0) = one                    [at time zero the flow is at the unit]
    ∧ flow (-1) (π / 2) = ((0 : ℝ), (1 : ℝ))  [at a quarter turn it is the generator]
    ∧ flow (-1) π = -one                   [at a half turn it is minus the unit]
    ∧ flow (-1) (3 * π / 2) = ((0 : ℝ), (-1 : ℝ))  [at three quarter turns, the negative generator]
    flow                                    [the size-one flow of the algebra]
    one                                     [the point multiplication leaves alone]

    So: `Trichotomy.flow_neg_one` writes the elliptic flow as `(cos t, sin t)`, and the four
    values of `t` in one period are the four branches of `{one, (0,1), −one, (0,−1)}` proved in
    `fourth_power_of_sq_neg_one` and its neighbours. The branches are the flow's own points, and
    the return to the unit at `3π/2 + π/2 = 2π` is the closure `Wick.lean` reads a temperature
    off. -/
theorem flow_quarter_turn_points :
    flow (-1) (0) = one
      ∧ flow (-1) (π / 2) = ((0 : ℝ), (1 : ℝ))
      ∧ flow (-1) π = -one
      ∧ flow (-1) (3 * π / 2) = ((0 : ℝ), (-1 : ℝ)) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [flow_neg_one, one, Real.cos_zero, Real.sin_zero]
  · simp [flow_neg_one, Real.cos_pi_div_two, Real.sin_pi_div_two]
  · simp [flow_neg_one, one, Real.cos_pi, Real.sin_pi]
  · have h3 : 3 * π / 2 = π / 2 + π := by ring
    rw [flow_neg_one, h3]
    simp [Real.cos_add_pi, Real.sin_add_pi,
      Real.cos_pi_div_two, Real.sin_pi_div_two]

/-- **The symmetric four-sample combination of the algebra's quadratic is its Hessian trace.**

    (σ u v h : ℝ)                          [the parameter, a place's two slots, a step]
    normForm σ (u + h, v)                  [the form sampled one step along the first slot]
    + normForm σ (u - h, v)                [and one step the other way]
    + normForm σ (u, v + h)                [and one step along the second slot]
    + normForm σ (u, v - h)                [and one step the other way]
    - 4 * normForm σ (u, v)                [minus four times the centre]
    = 2 * (1 - σ) * h ^ 2                  [is twice one minus σ, times the step squared]
    normForm                               [the size of a point]
    Cell                                   [a point: two real numbers side by side]

    So: the four-sample symmetric difference of `normForm σ` is exact at every step `h`, with no
    `h → 0` limit taken. The four offsets are the branch points of `flow_quarter_turn_points`,
    and the recovered value `2·(1−σ)` is the trace of the form's Hessian, `2 + (−2σ)`: the two
    second differences in the first slot give `2`, the two in the second give `−2σ`, and the
    cross and first-order terms cancel by the symmetry of the four samples. -/
theorem stencil4_normForm (σ u v h : ℝ) :
    normForm σ (u + h, v) + normForm σ (u - h, v)
      + normForm σ (u, v + h) + normForm σ (u, v - h) - 4 * normForm σ (u, v)
        = 2 * (1 - σ) * h ^ 2 :=
  ringStencil4_normForm σ u v h

/-- **On the split face the isotropic combination vanishes.**

    (u v h : ℝ)                    [a place's two slots and a step]
    normForm 1 (u + h, v)          [the form at one step on the first slot]
    + normForm 1 (u - h, v)        [and the other way]
    + normForm 1 (u, v + h)        [and one step on the second slot]
    + normForm 1 (u, v - h)        [and the other way]
    - 4 * normForm 1 (u, v)        [minus four times the centre]
    = 0                            [is zero]
    normForm                       [the size of a point]

    So: `stencil4_normForm` at `σ = 1`. The split form `u² − v²` has no isotropic second-order
    content under the plain sum of second differences; its curvature lives in the difference of
    the two axes, which is the sign the wave operator carries. -/
theorem stencil4_normForm_split (u v h : ℝ) :
    normForm 1 (u + h, v) + normForm 1 (u - h, v)
      + normForm 1 (u, v + h) + normForm 1 (u, v - h) - 4 * normForm 1 (u, v) = 0 := by
  rw [stencil4_normForm]
  ring

/-- **On the elliptic face the isotropic combination is maximal, four steps squared.**

    (u v h : ℝ)                    [a place's two slots and a step]
    normForm (-1) (u + h, v)       [the form at one step on the first slot]
    + normForm (-1) (u - h, v)     [and the other way]
    + normForm (-1) (u, v + h)     [and one step on the second slot]
    + normForm (-1) (u, v - h)     [and the other way]
    - 4 * normForm (-1) (u, v)     [minus four times the centre]
    = 4 * h ^ 2                    [is four steps squared]
    normForm                       [the size of a point]

    So: `stencil4_normForm` at `σ = −1`. The definite form `u² + v²` has second differences of
    two in each slot, so the four samples report four steps squared: the second-order content
    that the elliptic face's branches are there to sample. -/
theorem stencil4_normForm_elliptic (u v h : ℝ) :
    normForm (-1) (u + h, v) + normForm (-1) (u - h, v)
      + normForm (-1) (u, v + h) + normForm (-1) (u, v - h) - 4 * normForm (-1) (u, v)
        = 4 * h ^ 2 := by
  rw [stencil4_normForm]
  ring

end Physics
end Apeiron
