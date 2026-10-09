import Apeiron.Trichotomy.AnyRing

/-!
# Idempotents over any commutative ring

@name Idempotents Over Any Commutative Ring
@cite boole-1847
@cite peirce-1870
@cite stone-1936

`Idempotents` asks Boole's `x² = x` over the reals and answers by the sign of `σ`. Two steps of
that answer never used the reals: the slot expansion of `x² = x`, and the fact that a product of
idempotents is idempotent. Both are identities with whole-number coefficients, so they hold over
every commutative ring or over none.

This file carries them, together with the two structures a pair of idempotents can form.
Idempotents `e` and `f` are orthogonal when `e·f = 0`, and a complementary pair when in addition
`e + f = 1`. Multiplication by an idempotent is a projection: doing it twice is doing it once.

The points `idemPlus` and `idemMinus` of `Idempotents` come from a square root and a division by
two. Neither exists in a general commutative ring, so the construction here takes them as
explicit witnesses: a root `r` with `r·r = σ`, an inverse root `s` with `r·s = 1`, and a half `h`
with `2·h = 1`. Where a ring supplies those, the two idempotents exist; where it does not, the
statement says so rather than assuming them. Over `Z/2^n` the half does not exist, and the
conclusion that there are no extra idempotents is the one `AnyRing.ringIdempotent_trivial`
already gives.

## Root

| from | here |
|---|---|
| `AnyRing.RingIdempotent`, `AnyRing.ringMul` | `ringIdempotent_iff`, `ringIdempotent_mul` |
| `AnyRing.ringMul_assoc` | `ringProj_proj` |
| `AnyRing.ringMul_comm` | `ringOrthogonal_comm`, `ringComplementary_comm` |
| `AnyRing.ringMul_one`, `AnyRing.ringOne` | `ringProj_one`, `ringComplementary_add` |
| `AnyRing.ringMul_zero` | `ringProj_zero` |

The slot expansion, the monoid of idempotents, orthogonality, complementarity and the projection
are the `ringMul` identities of `Trichotomy/AnyRing.lean`, so they hold over every commutative
ring. What does not transfer is the sign classification:
`Idempotents.exists_nontrivial_idempotent_iff_disc_pos` decides the count by `0 < σ`. Here
`ringIdemPlus` and `ringIdemMinus` take a root `r` with `r * r = σ`, an inverse root `s` with
`r * s = 1`, and a half `h` with `2 * h = 1` as explicit witnesses, because no ring axiom
supplies a square root or a half. A ring without them has no such pair; in `ZMod (2 ^ n)` the
half does not exist and the answer is `AnyRing.ringIdempotent_trivial`.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy
namespace AnyRing

variable {R : Type*} [CommRing R]

/-- A point is unchanged by squaring exactly when each slot fixes itself.

    (σ : R)                                      [the parameter]
    (z : RingCell R)                             [the point]
    RingIdempotent σ z                           [the point is unchanged by squaring]
    ↔                                            [exactly when]
    z.1 ^ 2 + σ * z.2 ^ 2 = z.1                  [the first slot fixes itself]
    ∧                                            [and]
    2 * z.1 * z.2 = z.2                          [the second slot fixes itself]

    So: the definition written out one slot at a time, by expanding `ringMul`. No order and no
    root appears, so this is the same statement over every commutative ring. -/
theorem ringIdempotent_iff (σ : R) (z : RingCell R) :
    RingIdempotent σ z ↔ z.1 ^ 2 + σ * z.2 ^ 2 = z.1 ∧ 2 * z.1 * z.2 = z.2 := by
  obtain ⟨a, b⟩ := z
  simp only [RingIdempotent, ringMul, Prod.ext_iff]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨by linear_combination h1, by linear_combination h2⟩
  · rintro ⟨h1, h2⟩; exact ⟨by linear_combination h1, by linear_combination h2⟩

/-- The origin is unchanged by squaring, at every parameter.

    (σ : R)                 [the parameter]
    RingIdempotent σ 0      [the origin squares to itself] -/
theorem ringIdempotent_zero (σ : R) : RingIdempotent σ 0 := by
  simp [RingIdempotent, ringMul]

/-- The unit is unchanged by squaring, at every parameter.

    (σ : R)                   [the parameter]
    RingIdempotent σ ringOne  [the unit squares to itself] -/
theorem ringIdempotent_one (σ : R) : RingIdempotent σ ringOne := by
  simp [RingIdempotent, ringMul, ringOne]

/-- A product of points unchanged by squaring is unchanged by squaring.

    {σ : R}                                  [the parameter, at any value]
    {z w : RingCell R}                       [two points]
    RingIdempotent σ z                       [the first is unchanged by squaring]
    RingIdempotent σ w                       [so is the second]
    →                                        [then]
    RingIdempotent σ (ringMul σ z w)         [so is their product]

    So: squaring the product rearranges into the square of one times the square of the other, and
    each of those is itself. With the origin and the unit, the idempotents are therefore a
    commutative monoid over any commutative ring. -/
theorem ringIdempotent_mul {σ : R} {z w : RingCell R}
    (hz : RingIdempotent σ z) (hw : RingIdempotent σ w) :
    RingIdempotent σ (ringMul σ z w) := by
  show ringMul σ (ringMul σ z w) (ringMul σ z w) = ringMul σ z w
  calc ringMul σ (ringMul σ z w) (ringMul σ z w)
      = ringMul σ (ringMul σ z z) (ringMul σ w w) := by
        simp only [ringMul, Prod.ext_iff]; constructor <;> ring
    _ = ringMul σ z w := by rw [hz, hw]

/-- Two points that are each unchanged by squaring, and whose product is zero.

    (σ : R)                    [the parameter]
    (e f : RingCell R)         [the two points]
    RingIdempotent σ e         [the first is unchanged by squaring]
    ∧                          [and]
    RingIdempotent σ f         [the second is unchanged by squaring]
    ∧                          [and]
    ringMul σ e f = 0          [their product is the origin]

    So: Peirce's orthogonality, written over a ring. The product is zero, so each point lies
    outside the other. -/
def RingOrthogonal (σ : R) (e f : RingCell R) : Prop :=
  RingIdempotent σ e ∧ RingIdempotent σ f ∧ ringMul σ e f = 0

/-- An orthogonal pair whose sum is the unit.

    (σ : R)                    [the parameter]
    (e f : RingCell R)         [the two points]
    RingOrthogonal σ e f       [they are each unchanged by squaring and multiply to zero]
    ∧                          [and]
    e + f = ringOne            [their sum is the unit]

    So: Peirce's resolution of the unit. The two idempotents split every point into one part in
    each channel, and adding the two parts gives the point back. -/
def RingComplementary (σ : R) (e f : RingCell R) : Prop :=
  RingOrthogonal σ e f ∧ e + f = ringOne

/-- Orthogonality does not depend on the order of the two points.

    (σ : R)                    [the parameter]
    (e f : RingCell R)         [the two points]
    RingOrthogonal σ e f       [the first pair is orthogonal]
    ↔                          [exactly when]
    RingOrthogonal σ f e       [the reversed pair is orthogonal]

    So: multiplication is commutative, so the product that must vanish is the same either way,
    and each point is idempotent regardless of order. -/
theorem ringOrthogonal_comm (σ : R) (e f : RingCell R) :
    RingOrthogonal σ e f ↔ RingOrthogonal σ f e := by
  unfold RingOrthogonal
  constructor
  · rintro ⟨he, hf, h⟩
    exact ⟨hf, he, by rw [ringMul_comm σ f e, h]⟩
  · rintro ⟨hf, he, h⟩
    exact ⟨he, hf, by rw [ringMul_comm σ e f, h]⟩

/-- Being a complementary pair does not depend on the order of the two points.

    (σ : R)                    [the parameter]
    (e f : RingCell R)         [the two points]
    RingComplementary σ e f    [the first pair resolves the unit]
    ↔                          [exactly when]
    RingComplementary σ f e    [the reversed pair does]

    So: orthogonality reverses by `ringOrthogonal_comm`, and a sum is the same either way. -/
theorem ringComplementary_comm (σ : R) (e f : RingCell R) :
    RingComplementary σ e f ↔ RingComplementary σ f e := by
  unfold RingComplementary
  constructor
  · rintro ⟨ho, hsum⟩
    exact ⟨(ringOrthogonal_comm σ e f).mp ho, by rw [add_comm]; exact hsum⟩
  · rintro ⟨ho, hsum⟩
    exact ⟨(ringOrthogonal_comm σ e f).mpr ho, by rw [add_comm]; exact hsum⟩

/-- The first point of a complementary pair is unchanged by squaring.

    (σ : R) (e f : RingCell R)         [the parameter and the two points]
    RingComplementary σ e f            [they resolve the unit]
    →                                  [gives]
    RingIdempotent σ e                 [the first is unchanged by squaring] -/
theorem ringComplementary_fst {σ : R} {e f : RingCell R}
    (h : RingComplementary σ e f) : RingIdempotent σ e := h.1.1

/-- The second point of a complementary pair is unchanged by squaring.

    (σ : R) (e f : RingCell R)         [the parameter and the two points]
    RingComplementary σ e f            [they resolve the unit]
    →                                  [gives]
    RingIdempotent σ f                 [the second is unchanged by squaring] -/
theorem ringComplementary_snd {σ : R} {e f : RingCell R}
    (h : RingComplementary σ e f) : RingIdempotent σ f := h.1.2.1

/-- The two points of a complementary pair multiply to the origin.

    (σ : R) (e f : RingCell R)         [the parameter and the two points]
    RingComplementary σ e f            [they resolve the unit]
    →                                  [gives]
    ringMul σ e f = 0                  [their product is the origin] -/
theorem ringComplementary_mul {σ : R} {e f : RingCell R}
    (h : RingComplementary σ e f) : ringMul σ e f = 0 := h.1.2.2

/-- The two points of a complementary pair add to the unit.

    (σ : R) (e f : RingCell R)         [the parameter and the two points]
    RingComplementary σ e f            [they resolve the unit]
    →                                  [gives]
    e + f = ringOne                    [their sum is the unit] -/
theorem ringComplementary_add {σ : R} {e f : RingCell R}
    (h : RingComplementary σ e f) : e + f = ringOne := h.2

/-- The two points of a complementary pair are both read off the first.

    (σ : R) (e f : RingCell R)         [the parameter and the two points]
    RingComplementary σ e f            [they resolve the unit]
    →                                  [gives]
    f = ringOne - e                    [the second is the unit less the first] -/
theorem ringComplementary_snd_eq {σ : R} {e f : RingCell R}
    (h : RingComplementary σ e f) : f = ringOne - e := by
  rw [← ringComplementary_add h, add_sub_cancel_left]

/-- The two points of a complementary pair are both read off the second.

    (σ : R) (e f : RingCell R)         [the parameter and the two points]
    RingComplementary σ e f            [they resolve the unit]
    →                                  [gives]
    e = ringOne - f                    [the first is the unit less the second] -/
theorem ringComplementary_fst_eq {σ : R} {e f : RingCell R}
    (h : RingComplementary σ e f) : e = ringOne - f := by
  rw [← ringComplementary_add h, add_sub_cancel_right]

/-- Multiplying a point by a fixed point, as an operation on points.

    (σ : R)                    [the parameter]
    (e z : RingCell R)         [the fixed point, and the point being projected]
    ringMul σ z e              [multiply them]
    : RingCell R               [a point of the ring]

    So: the operation a projection is built from, with the fixed point on the right. -/
def ringProj (σ : R) (e z : RingCell R) : RingCell R := ringMul σ z e

/-- Projecting twice is the same as projecting once.

    {σ : R} {e : RingCell R}                       [the parameter and the fixed point]
    (he : RingIdempotent σ e)                      [the fixed point is unchanged by squaring]
    (z : RingCell R)                               [any point]
    ringProj σ e (ringProj σ e z) = ringProj σ e z [the second projection changes nothing]

    So: the two multiplications by `e` collapse into one, because `e` squared is `e`. This is
    what makes the operation a projection rather than a multiplication. -/
theorem ringProj_proj {σ : R} {e : RingCell R} (he : RingIdempotent σ e) (z : RingCell R) :
    ringProj σ e (ringProj σ e z) = ringProj σ e z := by
  show ringMul σ (ringMul σ z e) e = ringMul σ z e
  rw [ringMul_assoc, he]

/-- Projecting by the unit changes nothing.

    (σ : R) (z : RingCell R)         [the parameter and any point]
    ringProj σ ringOne z = z         [multiplying by the unit returns the point] -/
theorem ringProj_one (σ : R) (z : RingCell R) : ringProj σ ringOne z = z := by
  show ringMul σ z ringOne = z
  rw [ringMul_one]

/-- Projecting by the origin sends everything to the origin.

    (σ : R) (z : RingCell R)         [the parameter and any point]
    ringProj σ 0 z = 0               [multiplying by the origin gives the origin] -/
theorem ringProj_zero (σ : R) (z : RingCell R) : ringProj σ 0 z = 0 := by
  show ringMul σ z 0 = 0
  exact ringMul_zero σ z

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO EXTRA IDEMPOTENTS, FROM A ROOT AND A HALF
-- ═══════════════════════════════════════════════════════════════════

/-- Halfway along the first slot, and out to `h·s` along the second.

    (σ : R)                  [the parameter: what the root squares to]
    (r s h : R)              [the root, its inverse, and the half]
    (h, h * s)               [the two slots]
    : RingCell R             [a point over the ring]

    So: `r·r = σ`, `r·s = 1` and `2·h = 1` make this the idempotent `idemPlus`. The witnesses
    are arguments rather than assumptions, so a ring without a half has nothing to apply it to.
    `σ` is carried for the parameter's sake and enters only through the root's equation. -/
def ringIdemPlus (_σ _r s h : R) : RingCell R := (h, h * s)

/-- The mirror of `ringIdemPlus` in the second slot.

    (σ : R)                  [the parameter: what the root squares to]
    (r s h : R)              [the root, its inverse, and the half]
    (h, -(h * s))            [the two slots]
    : RingCell R             [a point over the ring] -/
def ringIdemMinus (_σ _r s h : R) : RingCell R := (h, -(h * s))

/-- A root, its inverse and a half make `ringIdemPlus` unchanged by squaring.

    {σ r s h : R}                            [the parameter, the root, its inverse, the half]
    r * r = σ                                [the root squares to the parameter]
    r * s = 1                                [the root times its inverse is one]
    2 * h = 1                                [twice the half is one]
    →                                        [then]
    RingIdempotent σ (ringIdemPlus σ r s h)  [the point is unchanged by squaring]

    So: the second slot fixes itself by `2h = 1`, and the first slot is `h² + σh²s²`, which is
    `h² + h²` because `σs² = (rs)² = 1`. No order appears anywhere. -/
theorem ringIdempotent_ringIdemPlus {σ r s h : R}
    (hrr : r * r = σ) (hrs : r * s = 1) (hh : 2 * h = 1) :
    RingIdempotent σ (ringIdemPlus σ r s h) := by
  rw [ringIdempotent_iff]
  have hσ : σ * s * s = 1 := by
    calc σ * s * s = (r * r) * s * s := by rw [hrr]
      _ = (r * s) * (r * s) := by ring
      _ = 1 := by rw [hrs]; ring
  simp only [ringIdemPlus]
  refine ⟨?_, ?_⟩
  · linear_combination (h ^ 2) * hσ + h * hh
  · linear_combination (h * s) * hh

/-- A root, its inverse and a half make `ringIdemMinus` unchanged by squaring.

    {σ r s h : R}                             [the parameter, the root, its inverse, the half]
    r * r = σ                                 [the root squares to the parameter]
    r * s = 1                                 [the root times its inverse is one]
    2 * h = 1                                 [twice the half is one]
    →                                         [then]
    RingIdempotent σ (ringIdemMinus σ r s h)  [the point is unchanged by squaring] -/
theorem ringIdempotent_ringIdemMinus {σ r s h : R}
    (hrr : r * r = σ) (hrs : r * s = 1) (hh : 2 * h = 1) :
    RingIdempotent σ (ringIdemMinus σ r s h) := by
  rw [ringIdempotent_iff]
  have hσ : σ * s * s = 1 := by
    calc σ * s * s = (r * r) * s * s := by rw [hrr]
      _ = (r * s) * (r * s) := by ring
      _ = 1 := by rw [hrs]; ring
  simp only [ringIdemMinus]
  refine ⟨?_, ?_⟩
  · linear_combination (h ^ 2) * hσ + h * hh
  · linear_combination (-(h * s)) * hh

/-- The two constructed idempotents multiply to the origin.

    {σ r s h : R}                    [the parameter, the root, its inverse, the half]
    r * r = σ                        [the root squares to the parameter]
    r * s = 1                        [the root times its inverse is one]
    2 * h = 1                        [twice the half is one]
    →                                [then]
    ringMul σ (ringIdemPlus σ r s h) (ringIdemMinus σ r s h) = 0
                                     [their product is the origin]
    `ringMul`                   [multiply two points together, the σ way]
    `ringIdemPlus`              [the point h in the first slot and h·s in the second]
    `ringIdemMinus`             [the mirror of ringIdemPlus, second slot negated]

    So: the two channels carry opposite signs, and `σs² = 1` makes the first slot vanish. -/
theorem ringIdemPlus_mul_ringIdemMinus {σ r s h : R}
    (hrr : r * r = σ) (hrs : r * s = 1) (_hh : 2 * h = 1) :
    ringMul σ (ringIdemPlus σ r s h) (ringIdemMinus σ r s h) = 0 := by
  have hσ : σ * s * s = 1 := by
    calc σ * s * s = (r * r) * s * s := by rw [hrr]
      _ = (r * s) * (r * s) := by ring
      _ = 1 := by rw [hrs]; ring
  apply Prod.ext
  · show h * h + σ * (h * s) * (-(h * s)) = 0
    linear_combination (-(h ^ 2)) * hσ
  · show h * (-(h * s)) + (h * s) * h = 0
    ring

/-- The two constructed idempotents add to the unit.

    (σ r s h : R)                    [the parameter, the root, its inverse, the half]
    2 * h = 1                        [twice the half is one]
    →                                [then]
    ringIdemPlus σ r s h + ringIdemMinus σ r s h = ringOne
                                     [their sum is the unit]
    `ringIdemMinus`             [the mirror of ringIdemPlus, second slot negated]
    `ringIdemPlus`              [the point h in the first slot and h·s in the second]
    `ringOne`                   [the point (1, 0), which changes nothing] -/
theorem ringIdemPlus_add_ringIdemMinus (σ r s h : R) (hh : 2 * h = 1) :
    ringIdemPlus σ r s h + ringIdemMinus σ r s h = ringOne := by
  apply Prod.ext
  · show h + h = 1
    linear_combination hh
  · show h * s + -(h * s) = 0
    ring

/-- The two constructed idempotents are a complementary pair.

    {σ r s h : R}                    [the parameter, the root, its inverse, the half]
    r * r = σ                        [the root squares to the parameter]
    r * s = 1                        [the root times its inverse is one]
    2 * h = 1                        [twice the half is one]
    →                                [then]
    RingComplementary σ (ringIdemPlus σ r s h) (ringIdemMinus σ r s h)
                                     [they are orthogonal and sum to the unit]
    `ringIdemPlus`              [the point h in the first slot and h·s in the second]
    `ringIdemMinus`             [the mirror of ringIdemPlus, second slot negated]

    So: Peirce's resolution, assembled from the two idempotence results, the vanishing product
    and the sum. This is the generic shape `SplitComplex.ePlus` and `eMinus` instantiate. -/
theorem ringComplementary_idem {σ r s h : R}
    (hrr : r * r = σ) (hrs : r * s = 1) (hh : 2 * h = 1) :
    RingComplementary σ (ringIdemPlus σ r s h) (ringIdemMinus σ r s h) :=
  ⟨⟨ringIdempotent_ringIdemPlus hrr hrs hh, ringIdempotent_ringIdemMinus hrr hrs hh,
    ringIdemPlus_mul_ringIdemMinus hrr hrs hh⟩,
   ringIdemPlus_add_ringIdemMinus σ r s h hh⟩

end AnyRing
end Trichotomy
end Apeiron

end
