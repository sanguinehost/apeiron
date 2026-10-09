import Apeiron.Trichotomy.Basic
import Apeiron.Trichotomy.IdempotentsAnyRing

/-!
# Idempotents

@name Idempotents
@cite boole-1847
@cite peirce-1870
@cite stone-1936
@cite mathlib-sqrt
@grade proved

`0` and `1` are unchanged when multiplied by themselves. Among ordinary real numbers nothing
else is: `x² = x` forces `x(x − 1) = 0`, and the reals have no zero divisors, so `x` is `0`
or `1`. Boole took that equation as the defining law of logical classes in 1847.

In the two-dimensional algebra the same question has a different answer, and the answer
depends on the sign of sigma. Write a point as `a + b·x` and multiply it by itself:

```
   (a + b·x)(a + b·x) = (a² + σb²) + (2ab)·x
```

Setting that equal to `a + b·x` gives two equations:

```
   a² + σb² = a          the first slot
   2ab      = b          the second slot
```

The second one factors as `b(2a − 1) = 0`, so either `b = 0` or `a = 1/2`.

If `b = 0` the first equation is `a² = a`, which is Boole's equation again, so `a` is `0` or
`1`. Those are the two points every sigma already has.

If `a = 1/2` the first equation becomes `σb² = 1/4`. A square is never negative and `b` cannot
be zero here (`1/4 = 1/2` is false), so this needs `σ > 0`, and then `b = ±1/(2√σ)`.

So a point unchanged by squaring, other than `0` and the unit, exists exactly on the
hyperbolic face: `exists_nontrivial_idempotent_iff_disc_pos`.

Below zero there are no such points and no zero divisors either
(`noZeroDivisors_iff_disc_neg`), so both halves of `x(1 − x) = 0` are unavailable at once.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

/-- A point unchanged by multiplying it by itself.

@cite boole-1847

    (σ : ℝ)              [the parameter]
    (z : Cell)           [the point]
    mul σ z z = z        [multiplied by itself, it gives itself back]

    So: Boole's `x² = x`, read in this algebra. -/
def Idempotent (σ : ℝ) (z : Cell) : Prop := mul σ z z = z

/-- Being idempotent is two equations on the slots.

    (σ : ℝ) (z : Cell)                    [the parameter and the point]
    Idempotent σ z                        [z is unchanged by squaring]
    ↔                                     [exactly when]
    z.1 ^ 2 + σ * z.2 ^ 2 = z.1           [the first slot fixes itself]
    ∧                                     [and]
    2 * z.1 * z.2 = z.2                   [the second slot fixes itself]

    So: the definition, written out one slot at a time. This is the generic
    `AnyRing.ringIdempotent_iff` at `R = ℝ`, so the same statement holds in every commutative
    ring. Everything below reads off these. -/
theorem idempotent_iff (σ : ℝ) (z : Cell) :
    Idempotent σ z ↔ z.1 ^ 2 + σ * z.2 ^ 2 = z.1 ∧ 2 * z.1 * z.2 = z.2 :=
  AnyRing.ringIdempotent_iff σ z

/-- The origin is unchanged by squaring, at every sigma.

    (σ : ℝ)                [the parameter]
    Idempotent σ 0         [the origin squares to itself] -/
theorem idempotent_zero (σ : ℝ) : Idempotent σ 0 :=
  AnyRing.ringIdempotent_zero σ

/-- The unit is unchanged by squaring, at every sigma.

    (σ : ℝ)                  [the parameter]
    Idempotent σ one         [the unit squares to itself] -/
theorem idempotent_one (σ : ℝ) : Idempotent σ one :=
  AnyRing.ringIdempotent_one σ

/-- A product of points unchanged by squaring is unchanged by squaring.

    {σ : ℝ}                                  [the parameter, at any value]
    {z w : Cell}                             [two points]
    Idempotent σ z                           [the first is unchanged by squaring]
    Idempotent σ w                           [so is the second]
    →                                        [then]
    Idempotent σ (mul σ z w)                 [so is their product]

    So: squaring the product rearranges into the square of one times the square of the other,
    and each of those is itself. With the origin and the unit, the idempotents are therefore a
    commutative monoid. This is `AnyRing.ringIdempotent_mul` at `R = ℝ`. -/
theorem idempotent_mul {σ : ℝ} {z w : Cell}
    (hz : Idempotent σ z) (hw : Idempotent σ w) : Idempotent σ (mul σ z w) :=
  AnyRing.ringIdempotent_mul hz hw

-- ═══════════════════════════════════════════════════════════════════
-- ABOVE ZERO: THE TWO EXTRA POINTS, WRITTEN DOWN
-- ═══════════════════════════════════════════════════════════════════

/-- Halfway along the first slot, and out to `1/(2√σ)` along the second.

    (σ : ℝ)                             [the parameter]
    idemPlus σ                          [the name for this point]
    (1/2, 1 / (2 * Real.sqrt σ))        [one half, then one over twice the root of sigma]
    : Cell                              [a point of the algebra] -/
def idemPlus (σ : ℝ) : Cell := (1 / 2, 1 / (2 * Real.sqrt σ))

/-- Its mirror in the second slot.

    (σ : ℝ)                                [the parameter]
    idemMinus σ                            [the name for this point]
    (1/2, -(1 / (2 * Real.sqrt σ)))        [one half, then the negative of the same]
    : Cell                                 [a point of the algebra] -/
def idemMinus (σ : ℝ) : Cell := (1 / 2, -(1 / (2 * Real.sqrt σ)))

/-- Two points that are each unchanged by squaring, and whose product is zero.

    (σ : ℝ)                [the parameter]
    (e f : Cell)           [the two points]
    Orthogonal σ e f       [they are each unchanged by squaring and multiply to zero]

    So: Peirce's orthogonality, at `R = ℝ`. This is `AnyRing.RingOrthogonal`, so the same
    notion holds in every commutative ring. -/
abbrev Orthogonal (σ : ℝ) (e f : Cell) : Prop := AnyRing.RingOrthogonal σ e f

/-- An orthogonal pair whose sum is the unit.

    (σ : ℝ)                    [the parameter]
    (e f : Cell)               [the two points]
    Complementary σ e f        [they are orthogonal and their sum is the unit]

    So: Peirce's resolution of the unit, at `R = ℝ`. This is `AnyRing.RingComplementary`. -/
abbrev Complementary (σ : ℝ) (e f : Cell) : Prop := AnyRing.RingComplementary σ e f

/-- The generic construction at the real square root and the real half is `idemPlus`.

    (σ : ℝ)                    [the parameter]
    idemPlus σ                 [the real point (1/2, 1/(2√σ))]
    Real.sqrt σ                [the number that squares to give sigma]
    idemPlus σ = AnyRing.ringIdemPlus σ (Real.sqrt σ) (1 / Real.sqrt σ) (1 / 2)
                               [the real point is the generic point at these witnesses]

    So: the real witnesses are the root `√σ`, its inverse and the half `1/2`. This is what makes
    the real statements below specialisations rather than second constructions. -/
theorem idemPlus_eq_ringIdemPlus (σ : ℝ) :
    idemPlus σ = AnyRing.ringIdemPlus σ (Real.sqrt σ) (1 / Real.sqrt σ) (1 / 2) := by
  simp only [idemPlus, AnyRing.ringIdemPlus]
  congr 1
  ring

/-- The generic construction at the real square root and the real half is `idemMinus`.

    (σ : ℝ)                    [the parameter]
    idemMinus σ                [the real point (1/2, -(1/(2√σ)))]
    Real.sqrt σ                [the number that squares to give sigma]
    idemMinus σ = AnyRing.ringIdemMinus σ (Real.sqrt σ) (1 / Real.sqrt σ) (1 / 2)
                               [the real point is the generic point at these witnesses]

    So: the mirror of `idemPlus_eq_ringIdemPlus`, with the second slot negated on both sides. -/
theorem idemMinus_eq_ringIdemMinus (σ : ℝ) :
    idemMinus σ = AnyRing.ringIdemMinus σ (Real.sqrt σ) (1 / Real.sqrt σ) (1 / 2) := by
  simp only [idemMinus, AnyRing.ringIdemMinus]
  congr 1
  ring

/-- Sigma times the square of `1/(2√σ)` is a quarter.

    {σ : ℝ} (hσ : 0 < σ)                        [the parameter, set above zero]
    σ * (1 / (2 * Real.sqrt σ)) ^ 2             [sigma times that second slot, squared]
    = 1 / 4                                     [comes to a quarter]

    So: `(2√σ)² = 4σ`, so the product is `σ/(4σ)`. This is the one computation both
    witnesses below need, so it is stated once. -/
theorem sigma_mul_halfRoot_sq {σ : ℝ} (hσ : 0 < σ) :
    σ * (1 / (2 * Real.sqrt σ)) ^ 2 = 1 / 4 := by
  have hs : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt hσ.le
  rw [div_pow, one_pow, mul_pow, hs, mul_one_div,
      div_eq_div_iff (by positivity) (by norm_num : (4 : ℝ) ≠ 0)]
  ring

/-- Above zero, `idemPlus` is unchanged by squaring.

    (σ : ℝ) (hσ : 0 < σ)             [the parameter, set above zero]
    Idempotent σ (idemPlus σ)        [the point squares to itself]

    So: this is the generic `AnyRing.ringIdempotent_ringIdemPlus` at the root `√σ`, its inverse
    and the half `1/2`. The root squares to sigma by `Real.sq_sqrt`, and the two divisions are
    the two invertibility witnesses the generic construction consumes. -/
theorem idempotent_idemPlus {σ : ℝ} (hσ : 0 < σ) : Idempotent σ (idemPlus σ) := by
  rw [idemPlus_eq_ringIdemPlus]
  exact AnyRing.ringIdempotent_ringIdemPlus
    (by simpa [pow_two] using Real.sq_sqrt hσ.le)
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num)

/-- Above zero, `idemMinus` is unchanged by squaring.

    (σ : ℝ) (hσ : 0 < σ)              [the parameter, set above zero]
    Idempotent σ (idemMinus σ)        [the point squares to itself]

    So: the generic `AnyRing.ringIdempotent_ringIdemMinus` at the same three witnesses. -/
theorem idempotent_idemMinus {σ : ℝ} (hσ : 0 < σ) : Idempotent σ (idemMinus σ) := by
  rw [idemMinus_eq_ringIdemMinus]
  exact AnyRing.ringIdempotent_ringIdemMinus
    (by simpa [pow_two] using Real.sq_sqrt hσ.le)
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num)

/-- Above zero, the two extra idempotents are a complementary pair.

    (σ : ℝ) (hσ : 0 < σ)                       [the parameter, set above zero]
    Complementary σ (idemPlus σ) (idemMinus σ) [they are orthogonal and sum to the unit]

    So: `AnyRing.ringComplementary_idem` at the real root and half. This is Peirce's resolution
    of the unit on the hyperbolic face, and the generic theorem is what supplies it. -/
theorem complementary_idemPlus {σ : ℝ} (hσ : 0 < σ) :
    Complementary σ (idemPlus σ) (idemMinus σ) := by
  rw [idemPlus_eq_ringIdemPlus, idemMinus_eq_ringIdemMinus]
  exact AnyRing.ringComplementary_idem
    (by simpa [pow_two] using Real.sq_sqrt hσ.le)
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num)

/-- `idemPlus` is neither the origin nor the unit.

    (σ : ℝ)                    [the parameter]
    idemPlus σ ≠ 0             [its first slot is one half, not zero]
    ∧                          [and]
    idemPlus σ ≠ one           [its first slot is one half, not one] -/
theorem idemPlus_nontrivial (σ : ℝ) : idemPlus σ ≠ 0 ∧ idemPlus σ ≠ one := by
  constructor
  · intro h; have := congrArg Prod.fst h; norm_num [idemPlus] at this
  · intro h; have := congrArg Prod.fst h; norm_num [idemPlus, one] at this

-- ═══════════════════════════════════════════════════════════════════
-- THE HEADLINE
-- ═══════════════════════════════════════════════════════════════════

/-- A point unchanged by squaring, other than the origin and the unit, exists exactly when
sigma is positive.

    (σ : ℝ)                  [the parameter]
    ∃ z : Cell               [there is a point z]
    Idempotent σ z           [unchanged by squaring]
    z ≠ 0                    [not the origin]
    z ≠ one                  [not the unit]
    ∧                        [all of these together]
    ↔                        [exactly when]
    0 < σ                    [sigma is strictly positive]

    So: on the elliptic face and at the degenerate edge, the only points that survive
    squaring are the two every algebra has. Above zero there are two more, at
    `(1/2, ±1/(2√σ))`. -/
theorem exists_nontrivial_idempotent_iff_disc_pos (σ : ℝ) :
    (∃ z : Cell, Idempotent σ z ∧ z ≠ 0 ∧ z ≠ one) ↔ 0 < σ := by
  constructor
  · rintro ⟨z, hidem, hz0, hz1⟩
    obtain ⟨a, b⟩ := z
    rw [idempotent_iff] at hidem
    obtain ⟨h1, h2⟩ := hidem
    simp only at h1 h2
    -- the second slot forces `b = 0` or `a = 1/2`
    have hfac : b * (2 * a - 1) = 0 := by linear_combination h2
    rcases mul_eq_zero.mp hfac with hb | ha
    · -- `b = 0` collapses the first slot to Boole's equation, giving only the two trivial points
      subst hb
      have : a * (a - 1) = 0 := by linear_combination h1
      rcases mul_eq_zero.mp this with h | h
      · exact absurd (by simp [h] : ((a, (0 : ℝ)) : Cell) = 0) hz0
      · have ha1 : a = 1 := by linarith
        exact absurd (by simp [ha1, one] : ((a, (0 : ℝ)) : Cell) = one) hz1
    · -- `a = 1/2` puts `σ b² = 1/4`, and a square cannot be negative
      have ha : a = 1 / 2 := by linarith
      subst ha
      have hsb : σ * b ^ 2 = 1 / 4 := by linear_combination h1
      have hbne : b ≠ 0 := by
        intro h; rw [h] at hsb; norm_num at hsb
      have hb2 : 0 < b ^ 2 := by positivity
      nlinarith [hsb, hb2]
  · intro hσ
    exact ⟨idemPlus σ, idempotent_idemPlus hσ,
      (idemPlus_nontrivial σ).1, (idemPlus_nontrivial σ).2⟩

/-- At zero and below, nothing survives squaring except the origin and the unit.

    {σ : ℝ} (hσ : σ ≤ 0)         [the parameter, at zero or below]
    (z : Cell)                   [any point]
    Idempotent σ z               [unchanged by squaring]
    ↔                            [exactly when]
    z = 0 ∨ z = one              [it is the origin or the unit]

    So: the branch `a = 1/2` needs `σb² = 1/4`, and at zero or below `σb²` cannot be
    positive. Only the `b = 0` branch survives, and that is Boole's equation. -/
theorem idempotent_iff_trivial_of_disc_nonpos {σ : ℝ} (hσ : σ ≤ 0) (z : Cell) :
    Idempotent σ z ↔ (z = 0 ∨ z = one) := by
  constructor
  · intro h
    obtain ⟨a, b⟩ := z
    rw [idempotent_iff] at h
    obtain ⟨h1, h2⟩ := h
    simp only at h1 h2
    have hfac : b * (2 * a - 1) = 0 := by linear_combination h2
    rcases mul_eq_zero.mp hfac with hb | ha
    · subst hb
      have : a * (a - 1) = 0 := by linear_combination h1
      rcases mul_eq_zero.mp this with h0 | h0
      · exact Or.inl (by simp [h0])
      · exact Or.inr (by simp [one, show a = 1 by linarith])
    · exfalso
      have haa : a = 1 / 2 := by linarith
      subst haa
      have hsb : σ * b ^ 2 = 1 / 4 := by linear_combination h1
      nlinarith [sq_nonneg b]
  · rintro (rfl | rfl)
    · exact idempotent_zero σ
    · exact idempotent_one σ

/-- The quotient has an element unchanged by squaring, other than zero and one, exactly when
sigma is positive.

    (σ : ℝ)                                    [the parameter]
    ∃ p : AdjoinRoot (rel σ)                   [an element of the quotient ring]
    p * p = p                                  [unchanged by squaring, in the ring]
    p ≠ 0                                      [not zero]
    p ≠ 1                                      [not one]
    ∧                                          [all three together]
    ↔                                          [exactly when]
    0 < σ                                      [sigma is positive]

    So: Boole's `x² = x` has solutions beyond `0` and `1` in `ℝ[X]/(X² − σ)` exactly on the
    hyperbolic face. Stated about the ring, with the ring's own multiplication and unit. -/
theorem quot_exists_nontrivial_idempotent_iff_disc_pos (σ : ℝ) :
    (∃ p : AdjoinRoot (rel σ), p * p = p ∧ p ≠ 0 ∧ p ≠ 1) ↔ 0 < σ := by
  rw [← exists_nontrivial_idempotent_iff_disc_pos σ]
  constructor
  · rintro ⟨p, hpp, hp0, hp1⟩
    obtain ⟨z, rfl⟩ := toQuot_surjective σ p
    rw [mul_eq_quotient_mul] at hpp
    refine ⟨z, toQuot_injective σ hpp, fun h => hp0 ((toQuot_eq_zero_iff σ z).mpr h), ?_⟩
    intro h
    exact hp1 (by rw [h]; exact toQuot_one σ)
  · rintro ⟨z, hz, hz0, hz1⟩
    refine ⟨toQuot σ z, by rw [mul_eq_quotient_mul]; exact congrArg (toQuot σ) hz, ?_, ?_⟩
    · exact fun h => hz0 ((toQuot_eq_zero_iff σ z).mp h)
    · intro h
      exact hz1 (toQuot_injective σ (by rw [h]; exact (toQuot_one σ).symm))

end Trichotomy
end Apeiron

end
