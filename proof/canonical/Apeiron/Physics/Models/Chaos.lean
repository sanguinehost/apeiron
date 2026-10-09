import Apeiron.Physics.Quadratic
import Apeiron.Trichotomy.Flow
import Apeiron.Trichotomy.NullBasis
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.NumberTheory.Real.GoldenRatio

/-!
# Multipliers of a planar fixed point, and the renormalisation operator

@name Chaos
@cite feigenbaum-1978
@cite lanford-1982
@cite pomeau-manneville-1980
@cite shenker-1982
@cite kisil-2010
@grade proved

A fixed point of a smooth planar map is classified by the pair of eigenvalues of the Jacobian
there, its multipliers. For a `2 × 2` real matrix `M` the multipliers are the roots of
`l² − tr(M)·l + det(M)`, so the discriminant `tr² − 4·det` decides whether they are real and
distinct, real and equal, or a conjugate pair off the real line. The three cases are the three
local bifurcations a one-parameter family can meet: a real multiplier passing `+1` is the
saddle-node, whose intermittent route is Pomeau and Manneville 1980; a real multiplier passing
`−1` is period doubling, Feigenbaum 1978; a conjugate pair crossing the unit circle is the
Neimark-Sacker case, whose golden-rotation-number scaling is Shenker 1982.

The corpus already carries that classification, in another notation.
`Physics.rep σ z` is multiplication by `z` written as a matrix, `Physics.trace_rep` gives
`tr = 2·z.1` and `Physics.det_rep` gives `det = normForm σ z`, so

```
   tr² − 4·det = 4·z.1² − 4·(z.1² − σ·z.2²) = 4·σ·z.2²
```

and the sign of the characteristic discriminant is the sign of `σ`. `multiplier_eq_zero_iff`
states this as the factorisation `(l − z.1)² = σ·z.2²`, and the three named theorems below read
it at the three signs. So the trichotomy of the algebra and the trichotomy of a planar
linearisation are one quantity, not two facts that agree: `Physics.generator_trichotomy` is the
same statement restricted to the traceless generator.

## The two routes that close exactly

**Saddle-node, `σ = 0`.** `multiplier_is_one_at_the_parabolic_face` says the multiplier at
`z = (1, v)` is `+1` and repeated, which is the tangency. Its normal form is `bulk a x =
x/(1 − a·x)`: composition-closed (`bulk_comp`), quadratic tangency `bulk a x − x = a·x²/(1 − a·x)`
(`bulk_is_saddle_node`), and an exact fixed point of `renorm 2` (`tangency_fixed`). The reason
composition is easy is `bulk_reciprocal`: in the coordinate `1/x` the family is translation, which
is the `σ = 0` flow `Trichotomy.flow 0 t = (1, t)` and `Physics.parabolic_one_param`.
`tangency_multiplier` computes the renormalisation multiplier of the drift as `4` exactly, and
`laminar_invariant` is the conserved quantity `t·√ε` of the two scalings, which is the
`ε^{-1/2}` laminar length.

**Golden rotation number.** `gaussStep θ = 1/θ − 1` is the continued-fraction step, `golden_fixed`
its fixed point at `φ − 1`, and `golden_multiplier` its derivative there, `−φ²` exactly.

## Feigenbaum's ratio: what is defined and what is cited

`renorm α f x = α·f(f(x/α))`. Taking `α = 1/f(1)` this is Lanford's doubling operator
`Tψ(x) = −(1/a)·ψ(ψ(−a·x))` with `a = −ψ(1)`, section 1 of his 1982 announcement.
`IsRenormMultiplier` defines a multiplier of that operator at a fixed point as the pointwise
derivative in the direction of a perturbation, and `four_is_a_renorm_multiplier` exhibits one:
the identity is a fixed point of `renorm 2` and the constant perturbation has multiplier `4`.

Feigenbaum's `δ` is a multiplier of the **same operator at a different fixed point**, the even
unimodal `g` with a quadratic maximum. That this fixed point exists, and that the derivative there
is hyperbolic with a single simple expanding eigenvalue, is Lanford's Theorems 1 and 3. His proof
is computer assisted with interval arithmetic; Mathlib carries no interval arithmetic at that
scale, so no fixed point is constructed here and **no value is claimed**. His Estimate 1 reports
`|α − 4.669| < 0.148` for the leading entry of the derivative; even that bound is not reproduced.

What the corpus does prove about the operator is a limitation:
`no_polynomial_fixed_point_above_degree_one`. Any polynomial fixed point has degree at most one,
so the only polynomial chart renormalisation closes on is the degree-one family, which is `bulk`.
`polynomial_renorm_fixed_iff` is the step the predecessor corpus left open: it identifies the
polynomial identity `g = C α · (g ∘ g)(C α⁻¹ · X)` with `renorm α (eval g) = eval g` rather than
asserting they are the same condition. Non-polynomiality is not the absence of a closed form, and
nothing here decides whether `δ` is algebraic.

## The boost, read on the corpus's own null channels

`Trichotomy.upper 1` and `Trichotomy.lower 1` are the two isotropic coordinates at `σ = 1`, and on
the flow they are `exp t` and `exp (−t)` (`upper_flow_one`, `lower_flow_one`). So the stable and
unstable eigenvalues of a hyperbolic linearisation are the two null coordinates of
`Trichotomy.flow 1`, and their product being one is `Trichotomy.normForm_flow` read through
`Trichotomy.normForm_eq_upper_mul_lower`, not a separate computation. `lyapunov_rate` recovers the
rate from the unstable coordinate.

## Root

| from | here |
|---|---|
| `trace_rep`, `det_rep`, `normForm` | `multiplier_eq_zero_iff` |
| `multiplier_eq_zero_iff` | `exists_real_multiplier_iff_disc_nonneg` |
| `multiplier_eq_zero_iff` | `two_real_multipliers_iff_disc_pos` |
| `multiplier_eq_zero_iff` | `multiplier_is_one_at_the_parabolic_face` |
| `exists_real_multiplier_iff_disc_nonneg` | `multipliers_on_the_unit_circle` |
| `flow_one`, `upper`, `lower` | `upper_flow_one`, `lower_flow_one` |
| `normForm_flow`, `normForm_eq_upper_mul_lower` | `boost_exponents_multiply_to_one` |

The first two rows are the file's content: the discriminant of the characteristic polynomial of a
planar linearisation is `4·σ·z.2²`, so classifying a fixed point by its multipliers and
classifying the algebra by the sign of `σ` are one quantity read twice, and deleting `det_rep`
deletes
all four classification theorems. The fourth row is why the Lyapunov pair multiplies to one for a
reason and not by the exponential law.

**Where `σ` is a spectator, stated rather than left to be found.** `renorm`, `bulk`,
`bulk_reciprocal`, `bulk_comp`, `bulk_is_saddle_node`, `tangency_fixed`, `drift_quadruples`,
`laminar_invariant`, `four_is_a_renorm_multiplier`, `gaussStep`, `golden_fixed`,
`golden_multiplier`, `renorm_eval`, `polynomial_renorm_fixed_iff` and
`no_polynomial_fixed_point_above_degree_one` are identities in one real variable or in a
polynomial, with no `σ` in them. By `theory/NOVELTY.md` test A1 they are facts
about maps of the line, not results about this algebra. They are here because the classification
above is what says which face each of them belongs to, and that classification is a `σ` result.
-/

namespace Apeiron
namespace Physics

open Trichotomy Real Polynomial
open scoped goldenRatio

-- ═══════════════════════════════════════════════════════════════════
-- MULTIPLIERS OF A PLANAR LINEARISATION
-- ═══════════════════════════════════════════════════════════════════

/-- **The characteristic polynomial of `rep σ z` factors as `(l − z.1)² − σ·z.2²`.**

    (σ : ℝ) (z : Cell) (l : ℝ)                                  [the parameter, a point, a number]
    l ^ 2 - (rep σ z).trace * l + (rep σ z).det = 0
      ↔ (l - z.1) ^ 2 = σ * z.2 ^ 2                                  [the characteristic equation]
    rep                                                             [the grid a point multiplies by]
    Cell                                            [a point: two real numbers written side by side]
    ↔                                                        [exactly when, true in both directions]
    ℝ                                                                [the real numbers]

    So: `Physics.trace_rep` gives `tr = 2·z.1` and `Physics.det_rep` gives `det = normForm σ z`,
    so `tr² − 4·det = 4·σ·z.2²`. The discriminant of the characteristic polynomial of a planar
    linearisation is `σ`, times the square of the second coordinate. -/
theorem multiplier_eq_zero_iff (σ : ℝ) (z : Cell) (l : ℝ) :
    l ^ 2 - (rep σ z).trace * l + (rep σ z).det = 0 ↔ (l - z.1) ^ 2 = σ * z.2 ^ 2 := by
  rw [trace_rep, det_rep]
  simp only [normForm]
  constructor
  · intro h; linear_combination h
  · intro h; linear_combination h

/-- **A real multiplier exists exactly for `σ ≥ 0`.**

    (σ : ℝ) (z : Cell) (hz : z.2 ≠ 0)                              [the parameter and a point]
    (∃ l : ℝ, l ^ 2 - (rep σ z).trace * l + (rep σ z).det = 0) ↔ 0 ≤ σ
    rep                                                             [the grid a point multiplies by]
    Cell                                                           [a point: two reals side by side]
    ∃                                                                [there is at least one]
    ↔                                                        [exactly when, true in both directions]
    ≠                                                                [is not equal to]
    ≤                                                                [is smaller than or equal to]
    ℝ                                                                [the real numbers]

    So: the same shape as `Trichotomy.exists_normForm_zero_iff_disc_nonneg`, and for the same
    reason: a square equals `σ·z.2²`, and a square is non-negative. -/
theorem exists_real_multiplier_iff_disc_nonneg (σ : ℝ) (z : Cell) (hz : z.2 ≠ 0) :
    (∃ l : ℝ, l ^ 2 - (rep σ z).trace * l + (rep σ z).det = 0) ↔ 0 ≤ σ := by
  have hz2 : 0 < z.2 ^ 2 := by rcases lt_or_gt_of_ne hz with h | h <;> nlinarith
  constructor
  · rintro ⟨l, hl⟩
    rw [multiplier_eq_zero_iff] at hl
    nlinarith [sq_nonneg (l - z.1)]
  · intro hσ
    refine ⟨z.1 + Real.sqrt σ * z.2, ?_⟩
    rw [multiplier_eq_zero_iff,
      show (z.1 + Real.sqrt σ * z.2 - z.1) ^ 2 = Real.sqrt σ ^ 2 * z.2 ^ 2 by ring,
      Real.sq_sqrt hσ]

/-- **Two distinct real multipliers exist exactly above zero.**

    (σ : ℝ) (z : Cell) (hz : z.2 ≠ 0)                              [the parameter and a point]
    (∃ l m : ℝ, l ≠ m ∧ l ^ 2 - (rep σ z).trace * l + (rep σ z).det = 0
      ∧ m ^ 2 - (rep σ z).trace * m + (rep σ z).det = 0) ↔ 0 < σ
    rep                                                             [the grid a point multiplies by]
    Cell                                                           [a point: two reals side by side]
    ∃                                                                [there is at least one]
    ∧                                                                [both are true]
    ↔                                                        [exactly when, true in both directions]
    ≠                                                                [is not equal to]
    <                                                                [is smaller than]
    ℝ                                                                [the real numbers]

    So: the hyperbolic face. A saddle has two distinct real multipliers, one of which can pass
    `−1` and give period doubling; the corpus's `σ` being above zero is that condition. -/
theorem two_real_multipliers_iff_disc_pos (σ : ℝ) (z : Cell) (hz : z.2 ≠ 0) :
    (∃ l m : ℝ, l ≠ m ∧ l ^ 2 - (rep σ z).trace * l + (rep σ z).det = 0
      ∧ m ^ 2 - (rep σ z).trace * m + (rep σ z).det = 0) ↔ 0 < σ := by
  constructor
  · rintro ⟨l, m, hlm, hl, hm⟩
    rw [multiplier_eq_zero_iff] at hl hm
    by_contra hc
    rw [not_lt] at hc
    have h1 : (l - z.1) ^ 2 = 0 := by nlinarith [sq_nonneg (l - z.1), sq_nonneg z.2]
    have h2 : (m - z.1) ^ 2 = 0 := by nlinarith [sq_nonneg (m - z.1), sq_nonneg z.2]
    have hl1 : l = z.1 := by have h3 := sq_eq_zero_iff.mp h1; linarith
    have hm1 : m = z.1 := by have h3 := sq_eq_zero_iff.mp h2; linarith
    exact hlm (hl1.trans hm1.symm)
  · intro hσ
    have hs : 0 < Real.sqrt σ := Real.sqrt_pos.mpr hσ
    refine ⟨z.1 + Real.sqrt σ * z.2, z.1 - Real.sqrt σ * z.2, ?_, ?_, ?_⟩
    · intro h
      have hzero : Real.sqrt σ * z.2 = 0 := by linarith
      rcases mul_eq_zero.mp hzero with h1 | h1
      · exact absurd h1 (ne_of_gt hs)
      · exact hz h1
    · rw [multiplier_eq_zero_iff,
        show (z.1 + Real.sqrt σ * z.2 - z.1) ^ 2 = Real.sqrt σ ^ 2 * z.2 ^ 2 by ring,
        Real.sq_sqrt hσ.le]
    · rw [multiplier_eq_zero_iff,
        show (z.1 - Real.sqrt σ * z.2 - z.1) ^ 2 = Real.sqrt σ ^ 2 * z.2 ^ 2 by ring,
        Real.sq_sqrt hσ.le]

/-- **At the parabolic face the multiplier is `+1` and repeated.**

    (v l : ℝ)                                                        [a step and a number]
    l ^ 2 - (rep 0 ((1 : ℝ), v)).trace * l + (rep 0 ((1 : ℝ), v)).det = 0 ↔ l = 1
    rep                                                             [the grid a point multiplies by]
    ↔                                                        [exactly when, true in both directions]
    ℝ                                                                [the real numbers]

    So: the tangency. A real multiplier sitting at `+1`, which is the saddle-node bifurcation and
    the intermittent route of Pomeau and Manneville 1980. The `bulk` family below is its normal
    form, and `Physics.parabolic_one_param` is the group it generates. -/
theorem multiplier_is_one_at_the_parabolic_face (v l : ℝ) :
    l ^ 2 - (rep 0 ((1 : ℝ), v)).trace * l + (rep 0 ((1 : ℝ), v)).det = 0 ↔ l = 1 := by
  rw [multiplier_eq_zero_iff]
  constructor
  · intro h
    have h0 : (l - 1) ^ 2 = 0 := by simpa using h
    have h1 := sq_eq_zero_iff.mp h0
    linarith
  · intro h
    subst h
    norm_num

/-- **Below zero the multipliers are a conjugate pair on the unit circle.**

    {σ : ℝ} (hσ : σ < 0) (z : Cell) (hz : z.2 ≠ 0)             [σ in the elliptic face, a point]
    (hdet : normForm σ z = 1)                                        [the map preserves area]
    (l : ℂ) (hl : l ^ 2 - 2 * (z.1 : ℂ) * l + 1 = 0)                 [a complex multiplier]
    Complex.normSq l = 1                                             [it lies on the unit circle]
    normForm                                                         [the norm form of a point]
    Cell                                                           [a point: two reals side by side]
    Complex.normSq                                           [the size of a complex number, squared]
    ≠                                                                [is not equal to]
    <                                                                [is smaller than]
    ℝ                                                                [the real numbers]
    ℂ                           [the complex numbers: pairs of reals where the second squares to −1]

    So: the Neimark-Sacker case. An area-preserving elliptic linearisation has multipliers of
    modulus one, so a one-parameter family sits on the boundary of stability and the invariant
    curve appears. The proof uses that the coefficients are real, so the conjugate is the other
    root, and that the two roots multiply to the determinant. -/
theorem multipliers_on_the_unit_circle {σ : ℝ} (hσ : σ < 0) (z : Cell) (hz : z.2 ≠ 0)
    (hdet : normForm σ z = 1) (l : ℂ) (hl : l ^ 2 - 2 * (z.1 : ℂ) * l + 1 = 0) :
    Complex.normSq l = 1 := by
  have hnoreal : ∀ r : ℝ, r ^ 2 - 2 * z.1 * r + 1 ≠ 0 := by
    intro r hr
    have hex : ∃ m : ℝ, m ^ 2 - (rep σ z).trace * m + (rep σ z).det = 0 := by
      refine ⟨r, ?_⟩
      rw [trace_rep, det_rep, hdet]
      linarith
    exact absurd ((exists_real_multiplier_iff_disc_nonneg σ z hz).mp hex) (not_le.mpr hσ)
  have hconj_ne : (starRingEnd ℂ) l ≠ l := by
    intro h
    have him : l.im = 0 := Complex.conj_eq_iff_im.mp h
    have hre : l = ((l.re : ℝ) : ℂ) := by
      apply Complex.ext <;> simp [him]
    rw [hre] at hl
    have hcast : ((l.re ^ 2 - 2 * z.1 * l.re + 1 : ℝ) : ℂ) = 0 := by
      push_cast
      linear_combination hl
    exact hnoreal l.re (by exact_mod_cast hcast)
  have hlc : ((starRingEnd ℂ) l) ^ 2 - 2 * (z.1 : ℂ) * ((starRingEnd ℂ) l) + 1 = 0 := by
    have hmap := congrArg (starRingEnd ℂ) hl
    simpa [Complex.conj_ofReal, map_ofNat] using hmap
  have hsum : l + (starRingEnd ℂ) l = 2 * (z.1 : ℂ) := by
    have hd : (l - (starRingEnd ℂ) l) * (l + (starRingEnd ℂ) l - 2 * (z.1 : ℂ)) = 0 := by
      linear_combination hl - hlc
    have hne : l - (starRingEnd ℂ) l ≠ 0 := sub_ne_zero.mpr (Ne.symm hconj_ne)
    rcases mul_eq_zero.mp hd with h | h
    · exact absurd h hne
    · exact sub_eq_zero.mp h
  have hprod : l * (starRingEnd ℂ) l = 1 := by linear_combination -hl + l * hsum
  have hns : ((Complex.normSq l : ℝ) : ℂ) = 1 := by
    rw [← Complex.mul_conj]
    exact hprod
  exact_mod_cast hns

-- ═══════════════════════════════════════════════════════════════════
-- THE SADDLE-NODE NORMAL FORM
-- ═══════════════════════════════════════════════════════════════════

/-- The saddle-node normal form, as a one-parameter family of Mobius maps of the line.

    (a : ℝ) (x : ℝ)                    [a parameter and a point]
    x / (1 - a * x)                    [the family]
    bulk                               [the saddle-node normal form x over 1 minus a x]
    ℝ                                  [the real numbers]
    →                                  [takes the left and gives the right] -/
noncomputable def bulk (a : ℝ) : ℝ → ℝ := fun x => x / (1 - a * x)

/-- **In the reciprocal coordinate the family is a translation.**

    (a x : ℝ) (hx : x ≠ 0) (h : 1 - a * x ≠ 0)         [a parameter and a point off the pole]
    (bulk a x)⁻¹ = x⁻¹ - a                             [the family translates the reciprocal]
    bulk                                               [the saddle-node normal form]
    ≠                                                  [is not equal to]
    ⁻¹                                                 [one divided by]
    ℝ                                                  [the real numbers]

    So: the family is the parabolic one-parameter subgroup, which is the `σ = 0` face.
    `Trichotomy.flow 0 t = (1, t)` and `Physics.parabolic_one_param` are the same translation on
    the cell, so the saddle-node normal form is not a separate object. -/
theorem bulk_reciprocal (a x : ℝ) (hx : x ≠ 0) (h : 1 - a * x ≠ 0) :
    (bulk a x)⁻¹ = x⁻¹ - a := by
  unfold bulk
  rw [inv_div]
  field_simp

/-- **The family is closed under composition, adding the parameters.**

    (a b x : ℝ) (hb : 1 - b * x ≠ 0)                   [two parameters and a point off the pole]
    bulk a (bulk b x) = bulk (a + b) x                 [composing adds the parameters]
    bulk                                               [the saddle-node normal form]
    ≠                                                  [is not equal to]
    ℝ                                                  [the real numbers]

    So: the group law of the parabolic face, read in the chart of `bulk_reciprocal`. -/
theorem bulk_comp (a b x : ℝ) (hb : 1 - b * x ≠ 0) :
    bulk a (bulk b x) = bulk (a + b) x := by
  unfold bulk
  have h2 : 1 - a * (x / (1 - b * x)) = (1 - (a + b) * x) / (1 - b * x) := by
    rw [eq_div_iff hb, sub_mul, _root_.mul_assoc, div_mul_cancel₀ _ hb]
    ring
  rw [h2, div_div_eq_mul_div, div_mul_cancel₀ _ hb]

/-- **The tangency is exactly quadratic.**

    (a x : ℝ) (h : 1 - a * x ≠ 0)                      [a parameter and a point off the pole]
    bulk a x - x = a * x ^ 2 / (1 - a * x)             [the deviation from the identity]
    bulk                                               [the saddle-node normal form]
    ≠                                                  [is not equal to]
    ℝ                                                  [the real numbers]

    So: `bulk a x = x + a·x² + O(x³)`, the saddle-node normal form, with the pole at `x = 1/a`
    where the map stops being defined. -/
theorem bulk_is_saddle_node (a x : ℝ) (h : 1 - a * x ≠ 0) :
    bulk a x - x = a * x ^ 2 / (1 - a * x) := by
  unfold bulk
  rw [eq_div_iff h, sub_mul, div_mul_cancel₀ _ h]
  ring

/-- **The laminar length scales as the inverse square root of the drift.**

    (t ε : ℝ)                                          [a time and a drift]
    t / 2 * Real.sqrt (4 * ε) = t * Real.sqrt ε        [halving the time and quadrupling the drift]
    Real.sqrt                                          [the number that squares to give this]
    ℝ                                                  [the real numbers]

    So: `t·√ε` is invariant under one renormalisation step, which halves the time and multiplies
    the drift by four. That invariance is the `ε^{-1/2}` law of the intermittent route. -/
theorem laminar_invariant (t ε : ℝ) :
    t / 2 * Real.sqrt (4 * ε) = t * Real.sqrt ε := by
  rw [show (4 : ℝ) * ε = 2 ^ 2 * ε by ring, Real.sqrt_mul (by positivity),
    Real.sqrt_sq (by norm_num)]
  ring

-- ═══════════════════════════════════════════════════════════════════
-- THE RENORMALISATION OPERATOR
-- ═══════════════════════════════════════════════════════════════════

/-- The renormalisation operator: compose the map with itself and rescale by `α`.

    (α : ℝ) (f : ℝ → ℝ)                [a scaling and a map of the line]
    fun x => α * f (f (x / α))         [the rescaled second iterate]
    renorm                             [compose a map with itself and rescale]
    ℝ                                  [the real numbers]
    →                                  [takes the left and gives the right]

    So: at `α = 1/f(1)` this is Lanford's doubling operator `Tψ(x) = −(1/a)·ψ(ψ(−a·x))` with
    `a = −ψ(1)`, section 1 of his 1982 announcement. -/
noncomputable def renorm (α : ℝ) (f : ℝ → ℝ) : ℝ → ℝ := fun x => α * f (f (x / α))

/-- **The saddle-node family is an exact fixed point of renormalisation at scaling two.**

    (a x : ℝ) (h1 : 2 - a * x ≠ 0) (h2 : 1 - a * x ≠ 0)    [a parameter and a point off two poles]
    renorm 2 (bulk a) x = bulk a x                         [renormalising returns the same map]
    renorm                                                 [compose a map with itself and rescale]
    bulk                                                   [the saddle-node normal form]
    ≠                                                      [is not equal to]
    ℝ                                                      [the real numbers]

    So: the tangency route is solved by its normal form, with no limit taken. -/
theorem tangency_fixed (a x : ℝ) (h1 : 2 - a * x ≠ 0) (h2 : 1 - a * x ≠ 0) :
    renorm 2 (bulk a) x = bulk a x := by
  unfold renorm bulk
  have e1 : x / 2 / (1 - a * (x / 2)) = x / (2 - a * x) := by
    rw [div_div]
    congr 1
    ring
  rw [e1]
  have e2 : 1 - a * (x / (2 - a * x)) = (2 * (1 - a * x)) / (2 - a * x) := by
    field_simp
    ring
  rw [e2, div_div_eq_mul_div, div_mul_cancel₀ _ h1, eq_div_iff h2, ← div_div, _root_.mul_assoc,
    div_mul_cancel₀ _ h2]
  ring

/-- **Renormalising a drift multiplies it by four.**

    (ε x : ℝ)                                          [a drift and a point]
    renorm 2 (fun y => y + ε) x = x + 4 * ε            [the drift is quadrupled]
    renorm                                             [compose a map with itself and rescale]
    ℝ                                                  [the real numbers]

    So: one factor of two from composing and one from the rescaling. -/
theorem drift_quadruples (ε x : ℝ) : renorm 2 (fun y => y + ε) x = x + 4 * ε := by
  unfold renorm
  ring

/-- A multiplier of renormalisation at a fixed point: the derivative of the operator along a
    perturbation, read pointwise.

    (α : ℝ) (f v : ℝ → ℝ) (d : ℝ)      [a scaling, a map, a perturbation, a candidate multiplier]
    IsRenormMultiplier                 [a multiplier of renormalisation at a fixed point]
    : Prop                             [a statement that is true or false]
    renorm                             [compose a map with itself and rescale]
    HasDerivAt                         [is changing at that rate at that time]
    ∀                                  [for every single one]
    ℝ                                  [the real numbers]
    →                                  [takes the left and gives the right]

    So: the linearisation of the operator at `f` sends the direction `v` to `d·v`, checked at
    every point. This is the eigenvalue equation Lanford's Theorem 3 is about, written without a
    Banach space. -/
def IsRenormMultiplier (α : ℝ) (f v : ℝ → ℝ) (d : ℝ) : Prop :=
  ∀ x : ℝ, HasDerivAt (fun t : ℝ => renorm α (fun y => f y + t * v y) x) (d * v x) 0

/-- **The identity is a fixed point of `renorm 2` and the constant direction has multiplier four.**

    IsRenormMultiplier 2 (fun x => x) (fun _ => 1) 4     [the multiplier is four]
    IsRenormMultiplier                            [a multiplier of renormalisation at a fixed point]

    So: the operator has at least one hyperbolic fixed point with an expanding multiplier, and it
    is `4` at the saddle-node face. Feigenbaum's ratio is a multiplier of this same operator at a
    different fixed point, the even unimodal map with a quadratic maximum; that fixed point's
    existence and the hyperbolicity of the derivative there are Lanford's Theorems 1 and 3, cited,
    and no value is claimed here. -/
theorem four_is_a_renorm_multiplier : IsRenormMultiplier 2 (fun x => x) (fun _ => 1) 4 := by
  intro x
  have hfun : (fun t : ℝ => renorm 2 (fun y => y + t * 1) x) = fun t : ℝ => x + 4 * t := by
    funext t
    unfold renorm
    ring
  rw [hfun]
  have hd : HasDerivAt (fun t : ℝ => x + 4 * t) (0 + 4 * 1) 0 :=
    (hasDerivAt_const (0 : ℝ) x).add (HasDerivAt.const_mul (4 : ℝ) (hasDerivAt_id' (x := (0 : ℝ))))
  simpa using hd

/-- **Evaluating the polynomial renormalisation identity is applying `renorm`.**

    (g : Polynomial ℝ) (α x : ℝ)                                     [a polynomial and two reals]
    (C α * ((g.comp g).comp (C α⁻¹ * X))).eval x
      = renorm α (fun y => g.eval y) x                               [the two agree at every point]
    renorm                                                   [compose a map with itself and rescale]
    ⁻¹                                                               [one divided by]
    ℝ                                                                [the real numbers]

    So: this is the step the predecessor corpus left as an assertion. The polynomial identity and
    the operator equation are now the same condition, not two objects that resemble each other. -/
theorem renorm_eval (g : Polynomial ℝ) (α x : ℝ) :
    (C α * ((g.comp g).comp (C α⁻¹ * X))).eval x = renorm α (fun y => g.eval y) x := by
  simp [renorm, Polynomial.eval_comp, div_eq_inv_mul]

/-- **The polynomial identity is exactly the fixed point equation.**

    (g : Polynomial ℝ) (α : ℝ)                                       [a polynomial and a scaling]
    g = C α * ((g.comp g).comp (C α⁻¹ * X))
      ↔ renorm α (fun y => g.eval y) = fun y => g.eval y             [the two conditions agree]
    renorm                                                   [compose a map with itself and rescale]
    ↔                                                        [exactly when, true in both directions]
    ⁻¹                                                               [one divided by]
    ℝ                                                                [the real numbers]

    So: a real polynomial is determined by its values, so the two directions are `renorm_eval` and
    `Polynomial.funext`. -/
theorem polynomial_renorm_fixed_iff (g : Polynomial ℝ) (α : ℝ) :
    g = C α * ((g.comp g).comp (C α⁻¹ * X))
      ↔ renorm α (fun y => g.eval y) = fun y => g.eval y := by
  constructor
  · intro h
    funext x
    rw [← renorm_eval, ← h]
  · intro h
    apply Polynomial.funext
    intro x
    rw [renorm_eval]
    exact (congrFun h x).symm

/-- **A polynomial fixed point of renormalisation has degree at most one.**

    (g : Polynomial ℝ) (α : ℝ) (hα : α ≠ 0)                          [a polynomial and a scaling]
    (hfix : renorm α (fun y => g.eval y) = fun y => g.eval y)        [it is a fixed point]
    g.natDegree ≤ 1                                                  [its degree is at most one]
    renorm                                                   [compose a map with itself and rescale]
    ≠                                                                [is not equal to]
    ≤                                                                [is smaller than or equal to]
    ℝ                                                                [the real numbers]

    So: composition multiplies degree and the rescaling preserves it, so `d = d²` forces `d ≤ 1`.
    The only polynomial chart renormalisation closes on is the degree-one family, which is `bulk`.
    This says nothing about whether Feigenbaum's ratio has a closed form, and no theorem here
    decides that. -/
theorem no_polynomial_fixed_point_above_degree_one (g : Polynomial ℝ) (α : ℝ) (hα : α ≠ 0)
    (hfix : renorm α (fun y => g.eval y) = fun y => g.eval y) : g.natDegree ≤ 1 := by
  have h := (polynomial_renorm_fixed_iff g α).mpr hfix
  by_contra hc
  rw [not_le] at hc
  have hdeg := congrArg Polynomial.natDegree h
  rw [Polynomial.natDegree_C_mul hα, Polynomial.natDegree_comp, Polynomial.natDegree_comp,
    Polynomial.natDegree_C_mul_X _ (inv_ne_zero hα), mul_one] at hdeg
  nlinarith

-- ═══════════════════════════════════════════════════════════════════
-- THE GOLDEN ROTATION NUMBER
-- ═══════════════════════════════════════════════════════════════════

/-- The continued-fraction step on a rotation number.

    (θ : ℝ)                            [a rotation number]
    θ⁻¹ - 1                            [invert and translate]
    gaussStep                          [the continued-fraction step, one over theta less one]
    ⁻¹                                 [one divided by]
    ℝ                                  [the real numbers]
    →                                  [takes the left and gives the right] -/
noncomputable def gaussStep : ℝ → ℝ := fun θ => θ⁻¹ - 1

/-- The golden section inverts the golden ratio.

    φ * (φ - 1) = 1                    [the defining relation, rearranged] -/
theorem gold_mul_gold_sub_one : φ * (φ - 1) = 1 := by
  linear_combination Real.goldenRatio_sq

/-- **The golden section is the fixed point of the continued-fraction step.**

    gaussStep (φ - 1) = φ - 1          [the step leaves it alone]
    gaussStep                          [the continued-fraction step]

    So: the rotation number that no renormalisation step simplifies. -/
theorem golden_fixed : gaussStep (φ - 1) = φ - 1 := by
  unfold gaussStep
  rw [inv_eq_of_mul_eq_one_left gold_mul_gold_sub_one]

/-- **The multiplier of the step at its golden fixed point is minus the golden ratio squared.**

    deriv gaussStep (φ - 1) = -(φ ^ 2)     [the derivative there]
    gaussStep                              [the continued-fraction step]
    deriv                                  [how fast the thing is changing, at that point]

    So: the scaling of the critical circle map at golden rotation number, whose universal constant
    Shenker 1982 reports numerically. This computes the multiplier of the linearisable step and
    not that constant. -/
theorem golden_multiplier : deriv gaussStep (φ - 1) = -(φ ^ 2) := by
  unfold gaussStep
  rw [deriv_sub_const, deriv_inv]
  have h : φ - 1 = φ⁻¹ := (inv_eq_of_mul_eq_one_right gold_mul_gold_sub_one).symm
  rw [h, inv_pow, inv_inv]

-- ═══════════════════════════════════════════════════════════════════
-- THE HYPERBOLIC PAIR ON THE CORPUS'S NULL CHANNELS
-- ═══════════════════════════════════════════════════════════════════

/-- **On the hyperbolic flow the first null coordinate is the growing exponential.**

    (t : ℝ)                                     [a time]
    upper 1 (flow 1 t) = Real.exp t             [the unstable coordinate grows]
    upper                              [how far a point lies along the first direction of size zero]
    flow             [the size-one points reached by turning, shearing or boosting for a given time]
    ℝ                                           [the real numbers]

    So: the unstable eigenvalue of a hyperbolic linearisation is the first isotropic coordinate of
    `Trichotomy.flow 1`, not an analogy to it. -/
theorem upper_flow_one (t : ℝ) : upper 1 (flow 1 t) = Real.exp t := by
  rw [flow_one]
  unfold upper
  rw [Real.sqrt_one]
  simp [Real.cosh_add_sinh t]

/-- **The other null coordinate is the decaying exponential.**

    (t : ℝ)                                     [a time]
    lower 1 (flow 1 t) = Real.exp (-t)          [the stable coordinate decays]
    lower                              [how far a point lies along the other direction of size zero]
    flow             [the size-one points reached by turning, shearing or boosting for a given time]
    ℝ                                           [the real numbers] -/
theorem lower_flow_one (t : ℝ) : lower 1 (flow 1 t) = Real.exp (-t) := by
  rw [flow_one]
  unfold lower
  rw [Real.sqrt_one]
  simp [Real.cosh_sub_sinh t]

/-- **The two exponents multiply to one, because the flow has size one.**

    (t : ℝ)                                          [a time]
    Real.exp t * Real.exp (-t) = 1                   [the pair of eigenvalues multiplies to one]
    ℝ                                                [the real numbers]

    So: derived from `Trichotomy.normForm_flow` through
    `Trichotomy.normForm_eq_upper_mul_lower`, not from the exponential law. The Lyapunov exponents
    of an area-preserving hyperbolic linearisation sum to zero because the norm form of the flow
    is one. -/
theorem boost_exponents_multiply_to_one (t : ℝ) : Real.exp t * Real.exp (-t) = 1 := by
  rw [← upper_flow_one t, ← lower_flow_one t,
    ← normForm_eq_upper_mul_lower (by norm_num : (0 : ℝ) ≤ 1), normForm_flow]

/-- **The rate is recovered from the unstable coordinate.**

    (lam t : ℝ) (ht : t ≠ 0)                                 [a rate and a non-zero time]
    Real.log (upper 1 (flow 1 (lam * t))) / t = lam           [the log of the growth over the time]
    upper                              [how far a point lies along the first direction of size zero]
    flow                              [the size-one points reached by turning, shearing or boosting]
    Real.log                                                [the power e must be raised to for this]
    ≠                                                        [is not equal to]
    ℝ                                                        [the real numbers]

    So: the Lyapunov exponent of the hyperbolic face is the flow's own rate parameter. -/
theorem lyapunov_rate (lam t : ℝ) (ht : t ≠ 0) :
    Real.log (upper 1 (flow 1 (lam * t))) / t = lam := by
  rw [upper_flow_one, Real.log_exp]
  field_simp

end Physics
end Apeiron
