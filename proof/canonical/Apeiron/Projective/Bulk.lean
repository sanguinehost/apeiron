import Apeiron.Projective.Basic
import Apeiron.Projective.AnyRing

/-!
# The bulk

@name The Half-Plane
@cite kisil-2010
@cite kisil-2007
@cite maldacena-1997
@cite ramallo-2013
@grade proved

The projective line is a boundary. What it bounds is the half-plane: the points of the algebra
whose second slot is positive. That is Kisil's object, and at `σ = −1` it is the Poincaré
half-plane, which is the hyperbolic plane, which is Euclidean `AdS₂`.

Two-by-two real grids act on it by Möbius transformations, and the question that decides whether
the geometry survives its own symmetries is whether the action keeps a bulk point in the bulk.

## The computation that governs it

Write a point as `x + y·ι` with `ι² = σ`, so `x` is the first slot and `y` the second. For a grid
with entries `a b c d`, the Möbius image is `(a·z + b) / (c·z + d)`, and clearing the denominator
by the conjugate gives

```
   second slot of (a·z + b)·conj(c·z + d)   =   y · (a·d − b·c)
   the denominator                          =   normForm σ (c·z + d)
```

Everything else cancels. So `mobiusIm` is `y · det / normForm σ (c·z + d)`, and the bulk is
preserved exactly when the determinant is positive and the denominator lies off the cone on the
positive side.

**That is where sigma enters, and its sign decides the geometry.** At `σ < 0` the norm form is
`(cx+d)² + |σ|c²y²`, a sum of two squares, so it is positive unless both terms vanish: the
half-plane is always preserved and the geometry is complete under its isometries. At `σ ≥ 0` the
norm form can vanish or turn negative, so a grid can carry a point of the bulk out of the bulk,
and the locus where it does is the cone.

`bulk_preserved_of_disc_neg` and `bulk_not_preserved_of_disc_nonneg` are those two statements.

## What makes it anti-de Sitter rather than flat

A space with a group acting on it is not yet a geometry. Ramallo's equation (8) writes the most
general metric whose boundary carries Poincaré invariance as a conformal factor times a flat
bracket, and his equations (9) to (11) derive the factor by requiring the dual theory be scale
invariant: under `(t, x) ↦ λ(t, x)` and `z ↦ λz` the factor must go to `λ⁻¹` times itself, so it
is `L/z` and the line element is `(L²/z²)(−dt² + dx² + dz²)`. Without that factor the half-plane
is flat, and flat is not anti-de Sitter.

The corpus supplies the split exactly. `lineElement` is `normForm σ v` over the height squared:
the bracket is the norm form and the factor is the height, and sigma does not appear in the
factor. So sigma's only contribution is the signature, which is what
`lineElement_elliptic`, `lineElement_parabolic` and `lineElement_hyperbolic` say in the three
faces. `mobius_isometry` is the statement that the Möbius action preserves it, and its proof is
one line of arithmetic on top of `mobius_snd`: the height scales by the same factor the
displacement does, so the ratio is untouched. That factor has no sigma in it. `mobiusNum_cross`
gives the exact difference quotient of a Möbius map with no limit taken, and `mobiusTangent`
is that quotient at coincident points.
-/

noncomputable section

namespace Apeiron
namespace Projective

open Trichotomy

/-- The determinant of a two-by-two real grid, written out.

    (a b c d : ℝ)               [the four entries]
    : ℝ                         [what the grid scales areas by]

    So: the real determinant is the generic determinant at `R = ℝ`, and the alias makes that
    definitional rather than a second formula. -/
abbrev det2 (a b c d : ℝ) : ℝ := Generic.ringDet2 a b c d

/-- The point a grid sends `z` to, before dividing.

    (a b : ℝ) (z : Cell)        [two entries and a point]
    : Cell                      [the numerator of the Möbius image]

    So: the real numerator is the generic one at `R = ℝ`, and the alias makes that definitional. -/
abbrev mobiusNum (a b : ℝ) (z : Cell) : Cell := Generic.ringMobiusNum a b z

/-- The point a grid divides by.

    (c d : ℝ) (z : Cell)        [two entries and a point]
    : Cell                      [the denominator of the Möbius image]

    So: the real denominator is the generic one at `R = ℝ`, and the alias makes that
    definitional. -/
abbrev mobiusDen (c d : ℝ) (z : Cell) : Cell := Generic.ringMobiusDen c d z

/-- The real determinant is the generic determinant at `R = ℝ`.

    (a b c d : ℝ)                          [the four entries]
    det2 a b c d = Generic.ringDet2 a b c d   [the two definitions agree]

    So: `det2` is the generic construction specialised, and the bridge is definitional. -/
theorem det2_eq_ringDet2 (a b c d : ℝ) : det2 a b c d = Generic.ringDet2 a b c d := rfl

/-- The real Möbius numerator is the generic one at `R = ℝ`.

    (a b : ℝ) (z : Cell)                             [two entries and a point]
    mobiusNum a b z = Generic.ringMobiusNum a b z    [the two definitions agree]

    So: the numerator crosses to the ring unchanged, so the real statements about it are
    specialisations of the generic ones. -/
theorem mobiusNum_eq_ringMobiusNum (a b : ℝ) (z : Cell) :
    mobiusNum a b z = Generic.ringMobiusNum a b z := rfl

/-- The real Möbius denominator is the generic one at `R = ℝ`.

    (c d : ℝ) (z : Cell)                             [two entries and a point]
    mobiusDen c d z = Generic.ringMobiusDen c d z    [the two definitions agree]

    So: the denominator crosses to the ring unchanged too, and only the inverse of its size is
    where the real field is used. -/
theorem mobiusDen_eq_ringMobiusDen (c d : ℝ) (z : Cell) :
    mobiusDen c d z = Generic.ringMobiusDen c d z := rfl

/-- The second slot of the cleared numerator is the height times the determinant.

    (σ a b c d : ℝ) (z : Cell)                                    [sigma, four entries, a point]
    (mul σ (mobiusNum a b z) (conj (mobiusDen c d z))).2  [clear by the conjugate, take slot two]
      = z.2 * det2 a b c d                              [the height, times the determinant]

    So: every other term cancels. Sigma does not appear, which is why the numerator's height is
    the same in all three geometries and only the denominator distinguishes them. -/
theorem mobiusNum_conj_snd (σ a b c d : ℝ) (z : Cell) :
    (mul σ (mobiusNum a b z) (conj (mobiusDen c d z))).2 = z.2 * det2 a b c d :=
  Generic.ringMobiusNum_conj_snd σ a b c d z

/-- The height of a Möbius image.

    (σ a b c d : ℝ) (z : Cell)               [sigma, four entries, a point]
    : ℝ                                      [the second slot of the image]

    So: `y · det` over the denominator's norm form, which `mobiusNum_conj_snd` and
    `mul_conj` together say is what dividing produces. -/
def mobiusIm (σ a b c d : ℝ) (z : Cell) : ℝ :=
  z.2 * det2 a b c d / normForm σ (mobiusDen c d z)

/-- Below zero the denominator never vanishes unless the grid's bottom row and the point conspire.

    {σ : ℝ} (hσ : σ < 0) (c d : ℝ) (z : Cell)               [sigma below zero, entries, a point]
    (hy : 0 < z.2)                                          [the point is in the bulk]
    (hcd : c ≠ 0 ∨ d ≠ 0)                                   [the bottom row is not both zero]
    0 < normForm σ (mobiusDen c d z)                        [the denominator is off the cone]

    So: the norm form is `(c·x + d)² + |σ|·c²·y²`, a sum of two squares. It is zero only when both
    vanish, and with the height above zero that needs `c = 0` and then `d = 0`. This is why the
    elliptic face is the one whose half-plane is complete under its own isometries. -/
theorem den_normForm_pos_of_disc_neg {σ : ℝ} (hσ : σ < 0) (c d : ℝ) (z : Cell)
    (hy : 0 < z.2) (hcd : c ≠ 0 ∨ d ≠ 0) :
    0 < normForm σ (mobiusDen c d z) := by
  simp only [normForm, mobiusDen, Generic.ringMobiusDen]
  rcases eq_or_ne c 0 with hc | hc
  · subst hc
    have hd : d ≠ 0 := by tauto
    have hd2 : (0 : ℝ) < d ^ 2 := by positivity
    nlinarith
  · have hz : c * z.2 ≠ 0 := mul_ne_zero hc (ne_of_gt hy)
    have h1 : (0 : ℝ) < (c * z.2) ^ 2 := by positivity
    have h2 : (0 : ℝ) < -σ * (c * z.2) ^ 2 := mul_pos (neg_pos.mpr hσ) h1
    nlinarith [sq_nonneg (c * z.1 + d)]

/-- Below zero, a grid of positive determinant keeps the bulk in the bulk.

    {σ : ℝ} (hσ : σ < 0) (a b c d : ℝ)                      [sigma below zero, four entries]
    (hdet : 0 < det2 a b c d)                               [the grid does not flip orientation]
    (z : Cell) (hy : 0 < z.2)                               [a point of the bulk]
    (hcd : c ≠ 0 ∨ d ≠ 0)                                   [the bottom row is not both zero]
    0 < mobiusIm σ a b c d z                                [the image is in the bulk too]

    So: the elliptic half-plane is closed under its own Möbius action, which is the classical fact
    about the hyperbolic plane and the reason `AdS₂` has the isometries it does. At zero and above
    the denominator's norm form can vanish or turn negative, so this argument does not run and the
    bulk is not closed: the locus where it fails is the cone. -/
theorem bulk_preserved_of_disc_neg {σ : ℝ} (hσ : σ < 0) (a b c d : ℝ)
    (hdet : 0 < det2 a b c d) (z : Cell) (hy : 0 < z.2) (hcd : c ≠ 0 ∨ d ≠ 0) :
    0 < mobiusIm σ a b c d z :=
  div_pos (mul_pos hy hdet) (den_normForm_pos_of_disc_neg hσ c d z hy hcd)

/-- At zero and above, a grid of positive determinant can throw a bulk point off the bulk.

    {σ : ℝ} (hσ : 0 ≤ σ)                                    [sigma at zero or above]
    ∃ a b c d, ∃ z : Cell                                   [there are entries and a point]
    0 < z.2                                                 [the point is in the bulk]
    0 < det2 a b c d                                        [the grid does not flip orientation]
    normForm σ (mobiusDen c d z) = 0                        [and lands the denominator on the cone]
    ∧                                                       [all together]

    So: the witness is the point `(√σ, 1)`, which has height one and sits on the cone because its
    two channels are `√σ ± √σ`. The grid `a = 1, b = −1, c = 1, d = 0` has determinant one and
    leaves that point where it is, so the denominator is the point itself and its size is zero.

    With `bulk_preserved_of_disc_neg` this is the whole statement: the half-plane is closed under
    its own Möbius action on the elliptic face and on no other. Below zero the size of the
    denominator is a sum of two squares and cannot vanish; at zero and above the cone is reachable
    and reaching it is what ejects the point. Which geometry a value of sigma gives is therefore
    not a matter of taste, and neither is which one is complete. -/
theorem bulk_not_preserved_of_disc_nonneg {σ : ℝ} (hσ : 0 ≤ σ) :
    ∃ a b c d : ℝ, ∃ z : Cell,
      0 < z.2 ∧ 0 < det2 a b c d ∧ normForm σ (mobiusDen c d z) = 0 := by
  refine ⟨1, -1, 1, 0, (Real.sqrt σ, 1), by norm_num, by simp [det2, Generic.ringDet2], ?_⟩
  simp only [normForm, AnyRing.ringNormForm, mobiusDen, Generic.ringMobiusDen]
  rw [show (1 : ℝ) * Real.sqrt σ + 0 = Real.sqrt σ by ring,
      show (1 : ℝ) * (1 : ℝ) = 1 by ring, Real.sq_sqrt hσ]
  ring


-- ═══════════════════════════════════════════════════════════════════
-- THE MAP ITSELF, AND THE FACTOR IT SCALES BY
-- ═══════════════════════════════════════════════════════════════════

/-- The Möbius image of a point.

    (σ a b c d : ℝ) (z : Cell)         [sigma, four entries, a point]
    : Cell                             [the numerator times the denominator's reciprocal]

    So: `(a·z + b) / (c·z + d)`, written with `recip` because the algebra has no division at
    `σ ≥ 0`. On the cone `recip` returns the origin and every statement below assumes the
    denominator lies off it. -/
def mobius (σ a b c d : ℝ) (z : Cell) : Cell :=
  mul σ (mobiusNum a b z) (recip σ (mobiusDen c d z))

/-- The height of the image is the height `mobiusIm` computes.

    (σ a b c d : ℝ) (z : Cell)                    [sigma, four entries, a point]
    (mobius σ a b c d z).2 = mobiusIm σ a b c d z [the image's second slot is that formula]

    So: `mobiusIm` was a formula and this makes it the height of an actual point. Everything the
    bulk results say about `mobiusIm` is therefore about the map, not about an abbreviation.

    No hypothesis on the denominator is needed, and that is a convention rather than a fact about
    the geometry. On the cone both sides are zero: `recip` returns the origin because Lean divides
    by zero to zero, and `mobiusIm` divides by zero for the same reason. Read this off the cone. -/
theorem mobius_snd (σ a b c d : ℝ) (z : Cell) :
    (mobius σ a b c d z).2 = mobiusIm σ a b c d z := by
  simp only [mobius, mobiusIm, mul, recip, mobiusNum, mobiusDen, det2, normForm,
    Generic.ringMobiusNum, Generic.ringMobiusDen, Generic.ringDet2]
  field_simp
  ring

/-- Two images cross-multiplied differ by the determinant times the gap.

    (σ a b c d : ℝ) (z w : Cell)                              [sigma, four entries, two points]
    mul σ (mobiusNum a b z) (mobiusDen c d w)                 [one numerator, the other denominator]
      - mul σ (mobiusNum a b w) (mobiusDen c d z)             [minus the same the other way round]
      = det2 a b c d • (z - w)                                [the determinant times the gap]

    So: this is the difference quotient of a Möbius map with the denominators cleared, and it is
    an identity of polynomials, so it holds at every sigma and needs no limit.
    Dividing by the two denominators gives `(M z − M w) = det·(z − w) / ((c·z+d)(c·w+d))`, and
    setting `w = z` in that is what `mobiusTangent` is. -/
theorem mobiusNum_cross (σ a b c d : ℝ) (z w : Cell) :
    mul σ (mobiusNum a b z) (mobiusDen c d w) - mul σ (mobiusNum a b w) (mobiusDen c d z)
      = det2 a b c d • (z - w) :=
  Generic.ringMobiusNum_cross σ a b c d z w

/-- The factor a Möbius map scales heights and displacements by.

    (σ a b c d : ℝ) (z : Cell)                   [sigma, four entries, a point]
    : ℝ                                          [the determinant over the denominator's size]

    So: one number, `det / normForm σ (c·z + d)`. `mobiusIm_eq_scale` says the height is the
    old height times this, and `normForm_mobiusTangent` says displacements are scaled by this
    too. Those being the same number is the whole of `mobius_isometry`. -/
def mobiusScale (σ a b c d : ℝ) (z : Cell) : ℝ :=
  det2 a b c d / normForm σ (mobiusDen c d z)

/-- The image's height is the old height times the factor.

    (σ a b c d : ℝ) (z : Cell)                                  [sigma, four entries, a point]
    mobiusIm σ a b c d z = z.2 * mobiusScale σ a b c d z        [height, times the factor] -/
theorem mobiusIm_eq_scale (σ a b c d : ℝ) (z : Cell) :
    mobiusIm σ a b c d z = z.2 * mobiusScale σ a b c d z := by
  simp only [mobiusIm, mobiusScale]
  ring

/-- The point a Möbius map multiplies displacements by.

    (σ a b c d : ℝ) (z : Cell)                   [sigma, four entries, a point]
    : Cell                                       [the determinant over the denominator squared]

    So: `mobiusNum_cross` divided by the two denominators reads
    `M z − M w = det · (z − w) / ((c·z+d)(c·w+d))`, and this is its coefficient at `w = z`. It is
    read off that identity rather than chosen, and no limit is taken to get it: the identity is
    polynomial and setting the two points equal is substitution. It is the generic tangent at the
    explicit inverse of the squared denominator's size, so the generic file's results about the
    tangent apply here directly. -/
def mobiusTangent (σ a b c d : ℝ) (z : Cell) : Cell :=
  Generic.ringMobiusTangent σ a b c d z
    (normForm σ (mul σ (mobiusDen c d z) (mobiusDen c d z)))⁻¹

/-- The real reciprocal is the conjugate scaled by the inverse of the size.

    (σ : ℝ) (w : Cell)                              [sigma and a point]
    recip σ w = (normForm σ w)⁻¹ • conj w           [the two definitions agree]

    So: `recip` divides each slot by the size, and scaling the conjugate by the size's inverse
    does the same thing. This is the bridge from the real division to the generic
    `ringRecip`, which carries that inverse as an explicit witness. -/
theorem recip_eq_smul_conj (σ : ℝ) (w : Cell) :
    recip σ w = (normForm σ w)⁻¹ • conj w := by
  apply Prod.ext <;>
    simp only [recip, conj, AnyRing.ringConj, Prod.smul_fst, Prod.smul_snd, smul_eq_mul,
      div_eq_inv_mul]

/-- The real tangent is the generic one at the explicit inverse of the squared size.

    (σ a b c d : ℝ) (z : Cell)                       [sigma, four entries, a point]
    mobiusTangent σ a b c d z                        [the real tangent]
      = Generic.ringMobiusTangent σ a b c d z
          (normForm σ (mul σ (mobiusDen c d z) (mobiusDen c d z)))⁻¹
                                                     [the generic tangent at that inverse]

    So: the real tangent is the generic construction specialised, with the division replaced by
    the explicit inverse the field supplies. Everything the generic file proves about the
    tangent therefore applies to the real one.

    normForm   [the norm form: a point times its conjugate, which can come out negative]
    mul        [multiply two points together, the σ way]
    ⁻¹         [one divided by] -/
theorem mobiusTangent_eq_ringMobiusTangent (σ a b c d : ℝ) (z : Cell) :
    mobiusTangent σ a b c d z
      = Generic.ringMobiusTangent σ a b c d z
          (normForm σ (mul σ (mobiusDen c d z) (mobiusDen c d z)))⁻¹ := rfl

/-- The displacement factor has size the square of the height factor.

    (σ a b c d : ℝ) (z : Cell)                          [sigma, four entries, a point]
    (h : normForm σ (mobiusDen c d z) ≠ 0)              [the denominator lies off the cone]
    normForm σ (mobiusTangent σ a b c d z)              [measure the displacement factor]
      = mobiusScale σ a b c d z ^ 2                     [the height factor, squared]

    So: the size is quadratic (`normForm_smul`) and multiplicative (`normForm_mul`), so the
    determinant comes out squared and the denominator comes out squared, and what is left is the
    height factor squared. This is the second of the two facts the isometry rests on, and it is
    now `ringNormForm_ringMobiusTangent` at the field's explicit inverse. -/
theorem normForm_mobiusTangent (σ a b c d : ℝ) (z : Cell)
    (h : normForm σ (mobiusDen c d z) ≠ 0) :
    normForm σ (mobiusTangent σ a b c d z) = mobiusScale σ a b c d z ^ 2 := by
  have hv : (normForm σ (mul σ (mobiusDen c d z) (mobiusDen c d z)))⁻¹
      * normForm σ (mul σ (mobiusDen c d z) (mobiusDen c d z)) = 1 :=
    inv_mul_cancel₀ (by rw [normForm_mul]; exact mul_ne_zero h h)
  rw [mobiusTangent_eq_ringMobiusTangent]
  change AnyRing.ringNormForm σ
      (Generic.ringMobiusTangent σ a b c d z
        (normForm σ (mul σ (mobiusDen c d z) (mobiusDen c d z)))⁻¹) =
    mobiusScale σ a b c d z ^ 2
  rw [Generic.ringNormForm_ringMobiusTangent σ a b c d z _ hv]
  rw [normForm_mul]
  simp only [Generic.ringDet2, det2, mobiusScale]
  field_simp

-- ═══════════════════════════════════════════════════════════════════
-- THE LINE ELEMENT: A FLAT BRACKET OVER THE HEIGHT SQUARED
-- ═══════════════════════════════════════════════════════════════════

/-- The line element of the half-plane.

    (σ : ℝ) (z v : Cell)               [sigma, a point and a displacement at it]
    : ℝ                                [the displacement's size over the point's height squared]

    So: `normForm σ v / z.2 ^ 2`. This is Ramallo's equation (8) at two dimensions, a conformal
    factor times a flat bracket, with the factor derived from scale invariance in his equations
    (9) to (11) and the bracket left free. Here the bracket is the algebra's own norm form, so
    sigma sets the signature and appears nowhere in the factor. At `σ = −1` this is the
    Poincaré metric of the hyperbolic plane, which is Euclidean `AdS₂`.

    The length unit `L` is set to one. It is an overall constant and scales the whole element,
    so it cannot change which face is which. -/
def lineElement (σ : ℝ) (z v : Cell) : ℝ := normForm σ v / z.2 ^ 2

/-- Below zero the line element is the Poincaré metric.

    (z v : Cell)                                                [a point and a displacement]
    lineElement (-1) z v = (v.1 ^ 2 + v.2 ^ 2) / z.2 ^ 2        [a sum of squares over the height]

    So: `(dx² + dy²) / y²`, the hyperbolic plane. -/
theorem lineElement_elliptic (z v : Cell) :
    lineElement (-1) z v = (v.1 ^ 2 + v.2 ^ 2) / z.2 ^ 2 := by
  simp only [lineElement, normForm]
  ring

/-- At zero the line element sees only the first slot.

    (z v : Cell)                                        [a point and a displacement]
    lineElement 0 z v = v.1 ^ 2 / z.2 ^ 2               [the first slot over the height]

    So: degenerate, and that degeneracy is the parabolic face. A displacement in the second slot
    has length zero whatever its size, which is why the two channels coincide there
    (`upper_eq_lower_of_disc_zero`). -/
theorem lineElement_parabolic (z v : Cell) :
    lineElement 0 z v = v.1 ^ 2 / z.2 ^ 2 := by
  simp only [lineElement, normForm]
  ring

/-- Above zero the line element is Lorentzian.

    (z v : Cell)                                                [a point and a displacement]
    lineElement 1 z v = (v.1 ^ 2 - v.2 ^ 2) / z.2 ^ 2           [a difference over the height]

    So: `(dx² − dy²) / y²`, which is `AdS₂` with one time direction rather than its Euclidean
    form. The conformal factor is the same one, and the only thing that changed is the sign
    sigma puts on the second square.

    The sign convention is worth stating because it is not Ramallo's. His equation (8) puts the
    minus on time and the plus on the height, so his `AdS₂` is `(dz² − dt²)/z²`. Here the height
    is the second slot and carries the minus, so this is `−1` times his, the mostly-minus
    convention against his mostly-plus. Same space, opposite overall sign. An overall sign cannot
    move a face, change which displacements are null, or change which paths are shortest, so
    nothing downstream turns on it, but a reader comparing the two line by line will hit it. -/
theorem lineElement_hyperbolic (z v : Cell) :
    lineElement 1 z v = (v.1 ^ 2 - v.2 ^ 2) / z.2 ^ 2 := by
  simp only [lineElement, normForm]
  ring

/-- The Möbius action preserves the line element.

    (σ a b c d : ℝ) (z v : Cell)                            [sigma, four entries, point and step]
    (hden : normForm σ (mobiusDen c d z) ≠ 0)               [the denominator lies off the cone]
    (hy : z.2 ≠ 0)                                          [the point has a height]
    (hdet : det2 a b c d ≠ 0)                               [the grid is not degenerate]
    lineElement σ (mobius σ a b c d z)                      [measure at the image]
      (mul σ (mobiusTangent σ a b c d z) v)                 [the displacement the map carries v to]
      = lineElement σ z v                                   [the same as measuring at the source]

    So: the height is multiplied by `mobiusScale` and the displacement's size by its square, and
    the line element divides one by the square of the other, so the factor cancels. This holds at
    every sigma, which is the point: the three geometries share one metric shape
    and one isometry group, and differ only in the signature sigma puts on the bracket.

    What the sign of sigma does decide is whether the hypotheses can be met globally.
    `den_normForm_pos_of_disc_neg` says that below zero the denominator is off the cone for every
    bulk point and every non-degenerate bottom row, so the action is by isometries of the whole
    half-plane. At zero and above `bulk_not_preserved_of_disc_nonneg` exhibits a grid and a point
    where it cannot be. -/
theorem mobius_isometry (σ a b c d : ℝ) (z v : Cell)
    (hden : normForm σ (mobiusDen c d z) ≠ 0) (hy : z.2 ≠ 0) (hdet : det2 a b c d ≠ 0) :
    lineElement σ (mobius σ a b c d z) (mul σ (mobiusTangent σ a b c d z) v)
      = lineElement σ z v := by
  have hs : mobiusScale σ a b c d z ≠ 0 := div_ne_zero hdet hden
  have h2 : (mobius σ a b c d z).2 = z.2 * mobiusScale σ a b c d z := by
    rw [mobius_snd σ a b c d z, mobiusIm_eq_scale]
  simp only [lineElement, h2, normForm_mul, normForm_mobiusTangent σ a b c d z hden]
  field_simp

/-- Below zero every non-zero displacement has positive length.

    {σ : ℝ} (hσ : σ < 0) (z v : Cell)              [sigma below zero, a point and a step]
    (hy : z.2 ≠ 0) (hv : v ≠ 0)                    [the point has a height, the step is not zero]
    0 < lineElement σ z v                          [the step has positive length]

    So: the elliptic face is the Riemannian one. The bracket is `v.1² + |σ|v.2²`, positive unless
    both slots vanish, and the factor is a square. -/
theorem lineElement_pos_of_disc_neg {σ : ℝ} (hσ : σ < 0) (z v : Cell)
    (hy : z.2 ≠ 0) (hv : v ≠ 0) : 0 < lineElement σ z v := by
  have hn : normForm σ v ≠ 0 := fun h => hv ((normForm_eq_zero_iff_of_neg hσ v).mp h)
  have hnn : 0 ≤ normForm σ v := by
    simp only [normForm]
    nlinarith [sq_nonneg v.1, sq_nonneg v.2]
  have hy2 : 0 < z.2 ^ 2 := by positivity
  exact div_pos (lt_of_le_of_ne hnn (Ne.symm hn)) hy2

/-- A direction of zero length exists exactly at zero and above.

    (σ : ℝ)                                        [sigma]
    (∃ v : Cell, v ≠ 0 ∧ ∀ z : Cell,               [some non-zero step, at every point]
      lineElement σ z v = 0)                       [has length zero]
    ↔ 0 ≤ σ                                        [exactly when sigma is at zero or above]

    So: the null cone of the algebra is the light cone of the metric, and they are one locus
    rather than two. This is `exists_normForm_zero_iff_disc_nonneg` read as geometry: below zero
    the metric is definite and there is nothing null to point at, at zero and above there is. -/
theorem exists_null_direction_iff_disc_nonneg (σ : ℝ) :
    (∃ v : Cell, v ≠ 0 ∧ ∀ z : Cell, lineElement σ z v = 0) ↔ 0 ≤ σ := by
  constructor
  · rintro ⟨v, hv, h⟩
    refine (exists_normForm_zero_iff_disc_nonneg σ).mp ⟨v, hv, ?_⟩
    have := h (0, 1)
    simpa [lineElement] using this
  · intro hσ
    obtain ⟨v, hv, hn⟩ := (exists_normForm_zero_iff_disc_nonneg σ).mpr hσ
    exact ⟨v, hv, fun z => by simp [lineElement, hn]⟩


-- ═══════════════════════════════════════════════════════════════════
-- THE FACTOR IS FORCED, NOT CHOSEN
-- ═══════════════════════════════════════════════════════════════════

/-- A line element with the conformal factor left free.

    (Ω : ℝ → ℝ) (σ : ℝ) (z v : Cell)     [a factor, sigma, a point and a displacement]
    : ℝ                                  [the factor at that height, squared, times the size]
    Ω                                    [the conformal factor: what the flat bracket is
                                          multiplied by at each height]
    normForm                             [the norm form: a point times its conjugate, which can
                                          come out negative]

    So: Ramallo's equation (8) with nothing yet fixed. His equations (9) to (11) then fix the
    factor by requiring the dual theory be scale invariant, and `scaleInvariant_iff_inverse` is
    that derivation here. -/
def genLineElement (Ω : ℝ → ℝ) (σ : ℝ) (z v : Cell) : ℝ := Ω z.2 ^ 2 * normForm σ v

/-- Stretching everything by one factor.

    (lam : ℝ) (z : Cell)        [a stretch factor and a point]
    : Cell                      [both slots stretched by it]
    •                           [stretch by, a number times a point]

    So: Ramallo's equation (9), which sends the boundary coordinates and the height by the same
    factor. It is the transformation the dual theory's scale invariance is about. -/
def dilate (lam : ℝ) (z : Cell) : Cell := lam • z

/-- A line element that the stretch leaves alone.

    (Ω : ℝ → ℝ) (σ : ℝ)                         [a factor and sigma]
    : Prop                                      [a statement that may hold or fail]
    ∀                                           [for every single one]
    <                                           [is smaller than]
    genLineElement                              [the factor squared, times the size]
    dilate                                      [both slots stretched by the factor]
    Ω                                           [the conformal factor: what the flat bracket is
                                                 multiplied by at each height]

    So: measuring a stretched displacement at a stretched point gives what measuring the
    original at the original gave. This is what Ramallo requires of the metric in his equation
    (10), and the next theorem says which factors satisfy it. -/
def ScaleInvariant (Ω : ℝ → ℝ) (σ : ℝ) : Prop :=
  ∀ lam : ℝ, 0 < lam → ∀ z v : Cell, 0 < z.2 →
    genLineElement Ω σ (dilate lam z) (dilate lam v) = genLineElement Ω σ z v

/-- The stretch fixes the line element exactly when the factor is one over the height.

    (Ω : ℝ → ℝ) (σ : ℝ)                                 [a factor and sigma]
    ScaleInvariant Ω σ                                  [the stretch leaves the element alone]
    ↔                                                   [exactly when, in both directions]
    ∀ y : ℝ, 0 < y → Ω y ^ 2 = Ω 1 ^ 2 / y ^ 2          [the factor falls off as one over height]
    ∀                                                   [for every single one]
    →                                                   [if the left holds so does the right]
    <                                                   [is smaller than]
    Ω                                                   [the conformal factor: what the flat
                                                         bracket is multiplied by at each height]

    So: this is Ramallo's equations (9) to (11), derived rather than quoted. The bracket picks up
    the factor squared under the stretch, so the conformal factor must lose it, which leaves
    `Ω y = L / y` with `L = Ω 1` the only freedom. The forward direction takes the unit at height
    one and reads the answer off; the backward direction substitutes and cancels.

    Sigma appears nowhere in the argument, which is the point. Scale invariance fixes the
    factor and says nothing about the signature, so all three faces carry the same factor and
    differ only in the bracket. That is why `lineElement` has one shape and three readings. -/
theorem scaleInvariant_iff_inverse (Ω : ℝ → ℝ) (σ : ℝ) :
    ScaleInvariant Ω σ ↔ ∀ y : ℝ, 0 < y → Ω y ^ 2 = Ω 1 ^ 2 / y ^ 2 := by
  constructor
  · intro h y hy
    have := h y hy (0, 1) one (by norm_num)
    simp only [genLineElement, dilate, normForm, AnyRing.ringNormForm, one, AnyRing.ringOne,
      Prod.smul_fst, Prod.smul_snd, smul_eq_mul] at this
    field_simp at this ⊢
    linarith [this]
  · intro h lam hlam z v hz
    have hzl : (0 : ℝ) < lam * z.2 := mul_pos hlam hz
    have hv : normForm σ (lam • v) = lam ^ 2 * normForm σ v := normForm_smul σ lam v
    simp only [genLineElement, dilate, Prod.smul_snd, smul_eq_mul, hv]
    rw [h (lam * z.2) hzl, h z.2 hz]
    field_simp

/-- The corpus's line element is the factor one over the height.

    (σ : ℝ) (z v : Cell)                                            [sigma, point, step]
    lineElement σ z v = genLineElement (fun y => 1 / y) σ z v        [the two agree]
    lineElement                                                     [the displacement's size
                                                                     over the height squared]
    genLineElement                                                  [the factor squared, times
                                                                     the size]

    So: `lineElement` is not one choice among many. It is the scale-invariant family at the
    length unit one, and `scaleInvariant_iff_inverse` says that family is the whole solution
    set. -/
theorem lineElement_eq_genLineElement (σ : ℝ) (z v : Cell) :
    lineElement σ z v = genLineElement (fun y => 1 / y) σ z v := by
  simp only [lineElement, genLineElement]
  field_simp

/-- The length unit is an overall constant.

    (L σ : ℝ) (z v : Cell)                                          [a unit, sigma, two points]
    genLineElement (fun y => L / y) σ z v = L ^ 2 * lineElement σ z v   [it scales everything]
    genLineElement                                                  [the factor squared, times
                                                                     the size]
    lineElement                                                     [the displacement's size
                                                                     over the height squared]

    So: Ramallo carries `L`, the radius, and this says what carrying it costs here. It multiplies
    every length by `L` and every squared length by `L²`, so it cannot change which face is
    which, cannot change whether a displacement is null, and cannot change which path is
    shortest. Setting it to one is a choice of unit and not a loss of generality. -/
theorem genLineElement_scale (L σ : ℝ) (z v : Cell) :
    genLineElement (fun y => L / y) σ z v = L ^ 2 * lineElement σ z v := by
  simp only [genLineElement, lineElement]
  field_simp

-- ═══════════════════════════════════════════════════════════════════
-- THE BOUNDARY, AND WHAT THE ISOMETRIES DO TO IT
-- ═══════════════════════════════════════════════════════════════════

/-- On the boundary a Möbius map is the fractional linear map, at every sigma.

    (σ a b c d x : ℝ) (h : c * x + d ≠ 0)      [sigma, four entries, a boundary point]
    mobius σ a b c d (x, 0)                    [send a point of height zero]
      = ((a * x + b) / (c * x + d), 0)         [it lands at height zero, moved this way]
    mobius                                     [the numerator times the denominator's reciprocal]
    ≠                                          [is not equal to]

    So: two things at once. The boundary is preserved, because the image has height zero. And
    the map induced there is `x ↦ (a·x + b) / (c·x + d)` with no `σ` in it, which is the group of
    fractional linear maps of the line: the conformal group in one dimension.

    That is Ramallo's section 7.1 at the smallest case. The isometries of the bulk act on the
    boundary as its conformal group, so a symmetry of the geometry is a symmetry of whatever
    lives on the boundary. Sigma dropping out of the map is the corpus's own addition: the
    three faces have different bulk geometries and the same boundary action, so a boundary
    theory cannot tell which face it bounds. -/
theorem mobius_boundary (σ a b c d x : ℝ) (h : c * x + d ≠ 0) :
    mobius σ a b c d (x, 0) = ((a * x + b) / (c * x + d), 0) := by
  have hsq : normForm σ (mobiusDen c d ((x : ℝ), (0 : ℝ))) = (c * x + d) ^ 2 := by
    simp [normForm, mobiusDen, Generic.ringMobiusDen]
  have hD : normForm σ (mobiusDen c d ((x : ℝ), (0 : ℝ))) ≠ 0 := by
    rw [hsq]
    exact pow_ne_zero 2 h
  have hNT : mobiusNum a b ((x : ℝ), (0 : ℝ))
      = mul σ (mobiusDen c d ((x : ℝ), (0 : ℝ))) ((a * x + b) / (c * x + d), 0) := by
    simp only [mobiusNum, mobiusDen, mul, Prod.ext_iff, Generic.ringMobiusNum,
      Generic.ringMobiusDen]
    refine ⟨?_, by ring⟩
    rw [mul_comm (c * x + d) ((a * x + b) / (c * x + d)), div_mul_cancel₀ _ h]
    ring
  rw [mobius, hNT, Trichotomy.mul_assoc,
    Trichotomy.mul_comm σ ((a * x + b) / (c * x + d), (0 : ℝ))
      (recip σ (mobiusDen c d ((x : ℝ), (0 : ℝ)))),
    ← Trichotomy.mul_assoc, mul_recip_cancel σ _ hD, one_mul]

/-- Every bulk point is the image of one bulk point.

    {σ : ℝ} (z : Cell) (hz : 0 < z.2)              [sigma and a point above the boundary]
    ∃ a b c d : ℝ, 0 < det2 a b c d ∧              [some grid of positive determinant]
      mobius σ a b c d (0, 1) = z                  [carries the point at height one to it]
    ∃                                              [there is at least one]
    ∧                                              [both are true]
    <                                              [is smaller than]
    mobius                                         [the numerator times the denominator's
                                                    reciprocal]
    det2                                           [what the grid scales areas by]

    So: the half-plane is homogeneous under its own isometries. With `mobius_isometry` this says
    the geometry looks the same from every point, which is the property that makes a bulk a
    background rather than a landscape. The witness is the grid that stretches by the height and
    slides by the first slot, whose determinant is the height. -/
theorem mobius_transitive {σ : ℝ} (z : Cell) (hz : 0 < z.2) :
    ∃ a b c d : ℝ, 0 < det2 a b c d ∧ mobius σ a b c d (0, 1) = z := by
  refine ⟨z.2, z.1, 0, 1, by simpa [det2, Generic.ringDet2] using hz, ?_⟩
  simp only [mobius, mobiusNum, mobiusDen, recip, mul, normForm, Prod.ext_iff,
    Generic.ringMobiusNum, Generic.ringMobiusDen]
  norm_num

end Projective
end Apeiron

end
