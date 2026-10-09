import Apeiron.Trichotomy.AnyRing

/-!
# The projective line over any commutative ring

@name The Projective Line Over Any Commutative Ring
@cite kisil-2010
@cite euclid-elements-ii-5
@grade proved

`Projective/Basic.lean` and `Projective/Bulk.lean` build the projective line and the bulk over the
reals, where `σ` has a sign and a non-zero real can be divided by. This file carries the algebraic
half of that construction to an arbitrary commutative ring `R`, so a finite ring such as `Z/2^n`
is an instance of the same statements rather than an analogy.

What crosses is exactly what is polynomial. The determinant `a·d − b·c`, the Möbius numerator and
denominator, the second slot of the numerator cleared by the denominator's conjugate, and the
cross-multiplied difference identity are identities between two polynomials with whole-number
coefficients, so they hold in every commutative ring or in none. This is the same observation that
makes `ringNormForm_mul` ring-universal.

What does not cross is division. Over a general ring the norm form of the denominator can be
divided by only when it is a unit, so every identity here that would divide is stated with the
denominator left in place. The reciprocal takes the inverse of the norm form as an explicit
witness, `ringRecip σ z u` with `u * ringNormForm σ z = 1`, rather than using a division that a
ring need not have. The projective point carries the unit condition as its field:
`RingProjPair σ` holds `u_unit : IsUnit (ringNormForm σ u)`.

The real theorems of `Projective/Bulk.lean` are the specialisations at `R = ℝ`, where `IsUnit` and
`≠ 0` coincide, so those two statements become the same condition.

## Root

| from | here |
|---|---|
| `AnyRing.ringMul`, `AnyRing.ringConj` | `ringMobiusNum`, `ringMobiusDen` |
| `AnyRing.ringNormForm` | `ringMobiusNum_conj_snd`, `ringMobiusNum_cross` |
| `AnyRing.ringNormForm_mul` | `ringMobiusDen_unit_sq` |
| `AnyRing.ringNormForm_smul` | `ringProjScale`, `ringNormForm_ringRecip` |
| `AnyRing.exists_inverse_iff_isUnit_ringNormForm` | `ringProjPair_invertible` |

`ringDet2`, the Möbius numerator and denominator, and the cross-multiplied difference identity are
polynomials with whole-number coefficients, so they hold in every commutative ring. What does not
transfer is division. The reciprocal `ringRecip σ z u` carries `u * ringNormForm σ z = 1` as an
explicit witness rather than dividing, and `RingProjPair` carries `IsUnit (ringNormForm σ u)` in
place of `u ≠ 0`. Over the reals those two conditions coincide, which is why
`Projective/Bulk.lean` reads these as its specialisation.
-/

namespace Apeiron
namespace Projective
namespace Generic

open Apeiron.Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- The determinant of a two-by-two grid over a commutative ring, written out.

    (a b c d : R)          [the four entries of the grid]
    a * d - b * c          [the diagonal product less the off-diagonal product]

    So: what the grid scales oriented areas by. This is `det2` from `Projective/Bulk.lean` with
    the reals replaced, and it is a polynomial in the four entries, so it crosses every
    commutative ring unchanged. -/
def ringDet2 (a b c d : R) : R := a * d - b * c

/-- The point a grid sends `z` to, before dividing.

    (a b : R) (z : RingCell R)    [two entries and a point]
    (a * z.1 + b, a * z.2)        [the first slot moved and shifted, the second only moved]

    So: the numerator of `(a·z + b) / (c·z + d)`, written with no division in it. This is
    `mobiusNum` from `Projective/Bulk.lean` over the ring. -/
def ringMobiusNum (a b : R) (z : RingCell R) : RingCell R := (a * z.1 + b, a * z.2)

/-- The point a grid divides by.

    (c d : R) (z : RingCell R)    [two entries and a point]
    (c * z.1 + d, c * z.2)        [the first slot moved and shifted, the second only moved]

    So: the denominator of `(a·z + b) / (c·z + d)`, again with no division in it. This is
    `mobiusDen` from `Projective/Bulk.lean` over the ring. -/
def ringMobiusDen (c d : R) (z : RingCell R) : RingCell R := (c * z.1 + d, c * z.2)

/-- The second slot of the cleared numerator is the height times the determinant.

    (σ a b c d : R) (z : RingCell R)                            [sigma, four entries, a point]
    (ringMul σ (ringMobiusNum a b z) (ringConj (ringMobiusDen c d z))).2
      = z.2 * ringDet2 a b c d                                  [the height times the determinant]
    `ringMul`                   [multiply two points together, the σ way]
    `ringConj`                  [the same point with its second slot negated]
    `ringMobiusNum`             [the numerator of a Möbius map, with no division]
    `ringMobiusDen`             [the denominator of a Möbius map, with no division]

    So: every other term cancels, and the cancellation is polynomial, so it needs no inverse.
    Sigma does not appear, which is why the numerator's height is the same in all three real
    faces and only the denominator distinguishes them. This is `mobiusNum_conj_snd` over the
    ring. -/
theorem ringMobiusNum_conj_snd (σ a b c d : R) (z : RingCell R) :
    (ringMul σ (ringMobiusNum a b z) (ringConj (ringMobiusDen c d z))).2
      = z.2 * ringDet2 a b c d := by
  simp only [ringMobiusNum, ringMobiusDen, ringDet2]
  ring

/-- Two images cross-multiplied differ by the determinant times the gap.

    (σ a b c d : R) (z w : RingCell R)                          [sigma, four entries, two points]
    ringMul σ (ringMobiusNum a b z) (ringMobiusDen c d w)
      - ringMul σ (ringMobiusNum a b w) (ringMobiusDen c d z)
      = ringDet2 a b c d • (z - w)                              [the determinant times the gap]
    `ringMul`                   [multiply two points together, the σ way]
    `ringMobiusNum`             [the numerator of a Möbius map, with no division]
    `ringMobiusDen`             [the denominator of a Möbius map, with no division]

    So: the difference quotient of a Möbius map with the denominators cleared. It is an identity
    of polynomials, so it holds at every sigma and in every commutative ring, and no limit is
    taken. This is `mobiusNum_cross` over the ring. -/
theorem ringMobiusNum_cross (σ a b c d : R) (z w : RingCell R) :
    ringMul σ (ringMobiusNum a b z) (ringMobiusDen c d w)
      - ringMul σ (ringMobiusNum a b w) (ringMobiusDen c d z)
      = ringDet2 a b c d • (z - w) := by
  simp only [ringMul, ringMobiusNum, ringMobiusDen, ringDet2, Prod.ext_iff, Prod.fst_sub,
    Prod.snd_sub, Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
  constructor <;> ring

/-- Conjugating a point does not change its size.

    (σ : R) (z : RingCell R)                    [sigma and a point]
    ringNormForm σ (ringConj z) = ringNormForm σ z   [the size is the same]

    So: negating the second slot leaves both squares alone. This is what lets a reciprocal keep
    the size of what it inverts, and it needs no order and no inverse. -/
theorem ringNormForm_ringConj (σ : R) (z : RingCell R) :
    ringNormForm σ (ringConj z) = ringNormForm σ z := by
  simp only [ringNormForm]
  ring

/-- The conjugate scaled by an explicit inverse of the size.

    (σ : R) (z : RingCell R) (u : R)    [sigma, a point, and a claimed inverse of its size]
    u • ringConj z                      [the conjugate, stretched by that inverse]

    So: `conj z` over `normForm σ z`, with the division replaced by an explicit inverse `u`.
    When `u * ringNormForm σ z = 1` this is the algebra's reciprocal; on the cone no such `u`
    exists, which is `exists_inverse_iff_isUnit_ringNormForm`. The name marks that it is the
    ring's construction and not Lean's `Inv`, which the algebra need not carry. -/
def ringRecip (_σ : R) (z : RingCell R) (u : R) : RingCell R := u • ringConj z

/-- An explicit inverse of the size makes the reciprocal cancel.

    (σ : R) (z : RingCell R) (u : R)          [sigma, a point, an inverse of its size]
    u * ringNormForm σ z = 1                  [the inverse is one]
    ringMul σ z (ringRecip σ z u) = ringOne   [the point times its reciprocal is the unit]

    So: `z · (u · conj z) = u · (z · conj z) = u · normForm σ z = 1`. No division appears, so
    this holds in every commutative ring. This is `mul_recip_cancel` over the ring, with the
    inverse carried rather than produced by a field. -/
theorem ringMul_ringRecip (σ : R) (z : RingCell R) (u : R)
    (hu : u * ringNormForm σ z = 1) :
    ringMul σ z (ringRecip σ z u) = ringOne := by
  simp only [ringRecip, ringMul, ringConj, ringOne, Prod.smul_fst, Prod.smul_snd,
    smul_eq_mul, Prod.mk.injEq]
  constructor
  · linear_combination hu
  · ring

/-- The size of the reciprocal is the inverse that was carried.

    (σ : R) (z : RingCell R) (u : R)          [sigma, a point, an inverse of its size]
    u * ringNormForm σ z = 1                  [the inverse is one]
    ringNormForm σ (ringRecip σ z u) = u      [the reciprocal's size is that inverse]

    So: the size is quadratic (`ringNormForm_smul`) and conjugation preserves it, so
    `normForm σ (u · conj z) = u² · normForm σ z`, which the witness collapses to `u`. This is
    `normForm_recip` over the ring, cleared of the division. -/
theorem ringNormForm_ringRecip (σ : R) (z : RingCell R) (u : R)
    (hu : u * ringNormForm σ z = 1) :
    ringNormForm σ (ringRecip σ z u) = u := by
  rw [ringRecip, ringNormForm_smul, ringNormForm_ringConj, pow_two, mul_assoc, hu, mul_one]

/-- The point a Möbius map multiplies displacements by, with the denominator left in place.

    (σ a b c d : R) (z : RingCell R) (v : R)   [sigma, four entries, a point, an inverse]
    ringDet2 a b c d • ringRecip σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z)) v
                                               [the determinant times the reciprocal of the
                                               squared denominator]

    So: `mobiusNum_cross` divided by the two denominators reads
    `M z − M w = det · (z − w) / ((c·z+d)(c·w+d))`, and this is its coefficient at `w = z`,
    with the reciprocal carrying the inverse of the squared denominator's size. No limit is
    taken: the cross identity is polynomial and setting the two points equal is substitution.
    This is `mobiusTangent` over the ring. -/
def ringMobiusTangent (σ a b c d : R) (z : RingCell R) (v : R) : RingCell R :=
  ringDet2 a b c d •
    ringRecip σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z)) v

/-- An inverse of the denominator's size squares into an inverse of the squared size.

    (σ c d : R) (z : RingCell R) (u : R)      [sigma, two entries, a point, an inverse]
    u * ringNormForm σ (ringMobiusDen c d z) = 1   [the inverse of the denominator's size]
    (u * u) * ringNormForm σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z)) = 1
                                              [the square is an inverse of the squared size]
    `ringMul`                   [multiply two points together, the σ way]

    So: `normForm_mul` turns the squared size into the size squared, and `(u·n)·(u·n) = 1·1`.
    This is what lets the tangent's reciprocal take `u * u` as its witness. -/
theorem ringMobiusDen_unit_sq (σ c d : R) (z : RingCell R) (u : R)
    (hu : u * ringNormForm σ (ringMobiusDen c d z) = 1) :
    (u * u) * ringNormForm σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z)) = 1 := by
  rw [ringNormForm_mul]
  calc (u * u) * (ringNormForm σ (ringMobiusDen c d z) * ringNormForm σ (ringMobiusDen c d z))
      = (u * ringNormForm σ (ringMobiusDen c d z))
          * (u * ringNormForm σ (ringMobiusDen c d z)) := by ring
    _ = 1 := by rw [hu, one_mul]

/-- The displacement factor has size the determinant squared times the carried inverse.

    (σ a b c d : R) (z : RingCell R) (v : R)     [sigma, four entries, a point, an inverse]
    v * ringNormForm σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z)) = 1
                                                 [the inverse of the squared denominator's size]
    ringNormForm σ (ringMobiusTangent σ a b c d z v)
      = ringDet2 a b c d ^ 2 * v                 [the determinant squared, times that inverse]
    `ringMul`                   [multiply two points together, the σ way]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]
    `ringMobiusDen`             [the denominator of a Möbius map, with no division]
    `ringMobiusTangent`         [the Möbius displacement factor, with the inverse carried]

    So: the size is quadratic (`ringNormForm_smul`) and the reciprocal's size is the carried
    inverse (`ringNormForm_ringRecip`), so the determinant comes out squared and nothing is
    divided. This is `normForm_mobiusTangent` over the ring, with the denominator left in place. -/
theorem ringNormForm_ringMobiusTangent (σ a b c d : R) (z : RingCell R) (v : R)
    (hv : v * ringNormForm σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z)) = 1) :
    ringNormForm σ (ringMobiusTangent σ a b c d z v) = ringDet2 a b c d ^ 2 * v := by
  rw [ringMobiusTangent, ringNormForm_smul, ringNormForm_ringRecip _ _ _ hv]

/-- Cleared of the inverse, the tangent's size times the squared denominator is the determinant.

    (σ a b c d : R) (z : RingCell R) (v : R)     [sigma, four entries, a point, an inverse]
    v * ringNormForm σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z)) = 1
                                                 [the inverse of the squared denominator's size]
    ringNormForm σ (ringMobiusTangent σ a b c d z v)
      * ringNormForm σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z))
      = ringDet2 a b c d ^ 2                     [the determinant squared]
    `ringMul`                   [multiply two points together, the σ way]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]
    `ringMobiusDen`             [the denominator of a Möbius map, with no division]
    `ringMobiusTangent`         [the Möbius displacement factor, with the inverse carried]

    So: this is the same identity with the inverse cancelled, so it names no division at all and
    can be read in any commutative ring. This is the form `mobius_isometry` consumes. -/
theorem ringNormForm_ringMobiusTangent_mul (σ a b c d : R) (z : RingCell R) (v : R)
    (hv : v * ringNormForm σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z)) = 1) :
    ringNormForm σ (ringMobiusTangent σ a b c d z v)
      * ringNormForm σ (ringMul σ (ringMobiusDen c d z) (ringMobiusDen c d z))
      = ringDet2 a b c d ^ 2 := by
  rw [ringNormForm_ringMobiusTangent σ a b c d z v hv, mul_assoc, hv, mul_one]

/-- A point of the projective line over a commutative ring.

    (σ : R)                              [the parameter, an element of the ring]
    RingProjPair σ                       [a point of the projective line over that ring]
    b u : RingCell R                     [the two slots]
    u_unit : IsUnit (ringNormForm σ u)   [the second slot's size can be divided by]

    So: the condition is not that `u` is non-zero, which a ring with zero divisors does not make
    invertible, but that its size is a unit, witnessed explicitly. Over the reals this is exactly
    `Projective.ProjPair`, because there `IsUnit` and `≠ 0` coincide. -/
structure RingProjPair (σ : R) where
  /-- The first slot. -/
  b : RingCell R
  /-- The second slot, whose size must be a unit. -/
  u : RingCell R
  /-- Its size can be divided by. -/
  u_unit : IsUnit (ringNormForm σ u)

/-- The second slot of a generic projective point can be divided by.

    {σ : R} (p : RingProjPair σ)             [a point of the projective line]
    ∃ w : RingCell R, ringMul σ p.u w = ringOne   [something multiplies with it to give the unit]

    So: the structure's own field, read through `exists_inverse_iff_isUnit_ringNormForm`, which
    turns the size's unit into an explicit point. Over the reals this is `u_invertible`. -/
theorem ringProjPair_invertible {σ : R} (p : RingProjPair σ) :
    ∃ w : RingCell R, ringMul σ p.u w = ringOne :=
  (exists_inverse_iff_isUnit_ringNormForm σ p.u).mpr p.u_unit

/-- Stretching both slots by a scalar that can be divided by gives a projective point.

    {σ : R} (p : RingProjPair σ)     [a point of the projective line]
    (c : R) (hc : IsUnit c)          [a scalar that can be divided by]
    : RingProjPair σ                 [the stretched point, still with a unit size]

    So: the size scales by `c ^ 2` (`ringNormForm_smul`), and a product of units is a unit, so
    the stretched second slot still lies off the cone in the only sense a ring has. Over the
    reals this is `scale`, where the hypothesis `c ≠ 0` is the same as `IsUnit c`. -/
def ringProjScale {σ : R} (p : RingProjPair σ) (c : R) (hc : IsUnit c) : RingProjPair σ where
  b := c • p.b
  u := c • p.u
  u_unit := by
    rw [ringNormForm_smul]
    have hc2 : IsUnit (c ^ 2) := by rw [pow_two]; exact hc.mul hc
    exact hc2.mul p.u_unit

end Generic
end Projective
end Apeiron
