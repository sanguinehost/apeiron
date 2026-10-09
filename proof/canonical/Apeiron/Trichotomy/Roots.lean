import Apeiron.Trichotomy.Basic

/-!
# Which roots of one the algebra holds

@name Roots
@cite cockle-1849
@cite gauss-1831
@grade proved

`Trichotomy/Idempotents.lean` asks which points are unchanged by squaring. This file asks the
neighbouring question: which points return to the unit after being multiplied by themselves a
fixed number of times. The two questions have different answers, and the answers point to
opposite signs of sigma.

Take a point `z` and ask for a square root of the unit other than the two every algebra has.
Writing `z = (a, b)`, the second slot of `z·z` is `2ab`, so one of `a` and `b` is zero. With
`b = 0` the first slot is `a² = 1` and `z` is `±1`, the pair every algebra holds. So `a = 0`, and
then `σb² = 1` needs `σ > 0`. **An extra square root of one is the hyperbolic face.**
It is the generator `j`, and squaring it is what the split-complex numbers are for.

Now ask for a square root of minus the unit. The same two slots give `a = 0` and `σb² = −1`,
which needs `σ < 0`. **A square root of minus one is the elliptic face**, and it is
the point physics writes `i`.

Now ask for a cube root of the unit other than the unit itself. The second slot of `z·z·z` is
`b(3a² + σb²)`, so either `b = 0`, which forces `a = 1` over the reals and returns the unit, or
`3a² + σb² = 0`. Substituting that into the first slot leaves `−8a³ = 1`, so `a = −1/2` with no
freedom left, and then `σb² = −3/4` needs `σ < 0` again. **A cube root of one is the
elliptic face.**

So three neighbouring questions separate the three faces:

```
   z·z = −1            needs   σ < 0        the elliptic face, the point i
   z·z·z = 1, z ≠ 1    needs   σ < 0        the elliptic face, three phases
   z·z = 1, z ≠ ±1     needs   σ > 0        the hyperbolic face, the point j
```

Sigma is not a spectator in any of them. Delete it and there is no statement left, and each
direction is decided by its sign alone.

What this buys downstream: a structure carrying three symmetric copies of one object, which is
what a three-generation pattern is, cannot sit on the hyperbolic or parabolic face, because
neither has an element of order three to carry it. `Apeiron/Physics/Flavour.lean` uses that, and
`Apeiron/Cosmology/Majorana.lean` uses the cube root itself.
-/

namespace Apeiron
namespace Trichotomy

/-- Squaring a point with an empty first slot.

    (σ b : ℝ)                                [the parameter and a step along the generator]
    mul σ (0, b) (0, b) = (σ * b ^ 2, 0)            [the square lands back on the first slot]
    mul                                             [multiply two points together, the σ way] -/
theorem mul_gen_self (σ b : ℝ) : mul σ ((0 : ℝ), b) ((0 : ℝ), b) = (σ * b ^ 2, 0) := by
  simp only [mul, Prod.ext_iff]
  constructor <;> ring

/-- Multiplying a point by itself twice more, slot by slot.

    (σ a b : ℝ)                                                  [the parameter and two slots]
    mul σ (mul σ (a, b) (a, b)) (a, b)                                [the point cubed]
    = (a ^ 3 + 3 * σ * a * b ^ 2, 3 * a ^ 2 * b + σ * b ^ 3)          [written out]
    mul                                                               [multiply two points] -/
theorem mul_cube (σ a b : ℝ) :
    mul σ (mul σ (a, b) (a, b)) (a, b)
      = (a ^ 3 + 3 * σ * a * b ^ 2, 3 * a ^ 2 * b + σ * b ^ 3) := by
  simp only [mul, Prod.ext_iff]
  constructor <;> ring

/-- **A square root of minus the unit exists exactly below zero.**

    (σ : ℝ)                            [the parameter]
    ∃ z : Cell                         [there is at least one point]
    mul σ z z = -one                   [multiplied by itself it gives minus the unit]
    ↔                                  [exactly when, true in both directions]
    σ < 0                              [sigma is negative]
    mul                                [multiply two points together, the σ way]
    one                                [the point that multiplication leaves alone]
    Cell                               [a point: two real numbers written side by side]

    So: the point physics writes `i` and uses to carry a phase lives on the elliptic face and on
    no other. A charge that acts by rotating needs somewhere to rotate, and the other two faces
    do not supply it. -/
theorem exists_sqrt_neg_one_iff_disc_neg (σ : ℝ) :
    (∃ z : Cell, mul σ z z = -one) ↔ σ < 0 := by
  constructor
  · rintro ⟨z, hz⟩
    obtain ⟨h1, h2⟩ := Prod.ext_iff.mp hz
    simp only [one, Prod.fst_neg, Prod.snd_neg, neg_zero] at h1 h2
    have hz2 : z.2 ≠ 0 := by
      intro h
      rw [h] at h1
      nlinarith [mul_self_nonneg z.1]
    have hz1 : z.1 = 0 := by
      have hprod : z.1 * z.2 = 0 := by linarith
      rcases mul_eq_zero.mp hprod with h | h
      · exact h
      · exact absurd h hz2
    rw [hz1] at h1
    nlinarith [mul_self_pos.mpr hz2]
  · intro hσ
    have hpos : (0 : ℝ) < -σ := by linarith
    have hne : σ ≠ 0 := ne_of_lt hσ
    refine ⟨((0 : ℝ), 1 / Real.sqrt (-σ)), ?_⟩
    have hval : σ * (1 / Real.sqrt (-σ)) ^ 2 = -1 := by
      rw [div_pow, one_pow, Real.sq_sqrt hpos.le]
      field_simp
    rw [mul_gen_self, hval]
    simp [one]

/-- **A square root of the unit beyond the two obvious ones exists exactly above zero.**

    (σ : ℝ)                                        [the parameter]
    ∃ z : Cell                                     [there is at least one point]
    mul σ z z = one                                [multiplied by itself it gives the unit]
    ∧                                              [and]
    z ≠ one                                        [it is not the unit]
    z ≠ -one                                       [and not minus the unit either]
    ↔                                              [exactly when]
    0 < σ                                          [sigma is positive]
    mul                                            [multiply two points together, the σ way]
    one                                            [the point that multiplication leaves alone]
    Cell                                           [a point: two real numbers side by side]

    So: the generator physics writes `j`, whose square is `+1`, is the hyperbolic face and no
    other. Order two and order three pull toward opposite signs of sigma. -/
theorem exists_extra_sqrt_one_iff_disc_pos (σ : ℝ) :
    (∃ z : Cell, mul σ z z = one ∧ z ≠ one ∧ z ≠ -one) ↔ 0 < σ := by
  constructor
  · rintro ⟨z, hz, hne1, hne2⟩
    obtain ⟨h1, h2⟩ := Prod.ext_iff.mp hz
    change z.1 * z.1 + σ * z.2 * z.2 = 1 at h1
    change z.1 * z.2 + z.2 * z.1 = 0 at h2
    have hz2 : z.2 ≠ 0 := by
      intro h
      rw [h] at h1
      have hfac : (z.1 - 1) * (z.1 + 1) = 0 := by nlinarith
      rcases mul_eq_zero.mp hfac with hc | hc
      · refine hne1 (Prod.ext_iff.mpr ⟨?_, ?_⟩)
        · show z.1 = (1 : ℝ); linarith
        · show z.2 = (0 : ℝ); exact h
      · refine hne2 (Prod.ext_iff.mpr ⟨?_, ?_⟩)
        · show z.1 = -(1 : ℝ); linarith
        · show z.2 = -(0 : ℝ); rw [h]; norm_num
    have hz1 : z.1 = 0 := by
      have hprod : z.1 * z.2 = 0 := by linarith
      rcases mul_eq_zero.mp hprod with h | h
      · exact h
      · exact absurd h hz2
    rw [hz1] at h1
    nlinarith [mul_self_pos.mpr hz2]
  · intro hσ
    have hne : σ ≠ 0 := ne_of_gt hσ
    refine ⟨((0 : ℝ), 1 / Real.sqrt σ), ?_, ?_, ?_⟩
    · have hval : σ * (1 / Real.sqrt σ) ^ 2 = 1 := by
        rw [div_pow, one_pow, Real.sq_sqrt hσ.le]
        field_simp
      rw [mul_gen_self, hval]
    · intro h
      have hfst := congrArg Prod.fst h
      norm_num [one] at hfst
    · intro h
      have hfst := congrArg Prod.fst h
      norm_num [one] at hfst

/-- **A cube root of the unit other than the unit exists exactly below zero.**

    (σ : ℝ)                                              [the parameter]
    ∃ z : Cell                                           [there is at least one point]
    mul σ (mul σ z z) z = one                            [cubed, it gives the unit]
    ∧ z ≠ one                                            [and it is not the unit]
    ↔                                                    [exactly when]
    σ < 0                                                [sigma is negative]
    mul                                                  [multiply two points, the σ way]
    one                                                  [the point multiplication leaves alone]
    Cell                                                 [a point: two real numbers side by side]

    So: a three-fold symmetric structure, three copies of one object carried by an element of
    order three, is available on the elliptic face and on neither of the others. The first slot
    is pinned to `−1/2` with no freedom, which is where the three phases spaced by a third of a
    turn come from. -/
theorem exists_cube_root_one_iff_disc_neg (σ : ℝ) :
    (∃ z : Cell, mul σ (mul σ z z) z = one ∧ z ≠ one) ↔ σ < 0 := by
  constructor
  · rintro ⟨z, hz, hne⟩
    rw [show z = (z.1, z.2) from rfl, mul_cube] at hz
    obtain ⟨h1, h2⟩ := Prod.ext_iff.mp hz
    have hz2 : z.2 ≠ 0 := by
      intro h
      rw [h] at h1
      have hfac : (z.1 - 1) * (z.1 ^ 2 + z.1 + 1) = 0 := by nlinarith
      rcases mul_eq_zero.mp hfac with hc | hc
      · refine hne (Prod.ext_iff.mpr ⟨?_, ?_⟩)
        · show z.1 = (1 : ℝ); linarith
        · show z.2 = (0 : ℝ); exact h
      · exfalso; nlinarith [sq_nonneg (2 * z.1 + 1)]
    have hkey : 3 * z.1 ^ 2 + σ * z.2 ^ 2 = 0 := by
      have hfac : z.2 * (3 * z.1 ^ 2 + σ * z.2 ^ 2) = 0 := by linear_combination h2
      rcases mul_eq_zero.mp hfac with h | h
      · exact absurd h hz2
      · exact h
    have ha : z.1 = -(1 / 2) := by
      have h8 : (2 * z.1 + 1) * (4 * z.1 ^ 2 - 2 * z.1 + 1) = 0 := by
        linear_combination (-1 : ℝ) * h1 + (3 * z.1) * hkey
      rcases mul_eq_zero.mp h8 with hc | hc
      · linarith
      · exfalso; nlinarith [sq_nonneg (2 * z.1 - 1 / 2)]
    rw [ha] at hkey
    nlinarith [pow_pos (abs_pos.mpr hz2) 2, sq_abs z.2, sq_nonneg z.2,
      mul_self_pos.mpr hz2]
  · intro hσ
    have hpos : (0 : ℝ) < 3 / (-(4 * σ)) := by
      apply div_pos (by norm_num)
      linarith
    have hne : σ ≠ 0 := ne_of_lt hσ
    refine ⟨(-(1 / 2), Real.sqrt (3 / (-(4 * σ)))), ?_, ?_⟩
    · have hsb : σ * Real.sqrt (3 / (-(4 * σ))) ^ 2 = -(3 / 4) := by
        rw [Real.sq_sqrt hpos.le]
        field_simp
      rw [mul_cube]
      simp only [one, Prod.ext_iff]
      constructor
      · linear_combination (-(3 : ℝ) / 2) * hsb
      · linear_combination (Real.sqrt (3 / (-(4 * σ)))) * hsb
    · intro h
      have hfst := congrArg Prod.fst h
      norm_num [one] at hfst

end Trichotomy
end Apeiron
