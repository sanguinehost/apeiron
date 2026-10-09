import Apeiron.Doubling.Sedenion
import Apeiron.Doubling.SplitOctonion
import Apeiron.Physics.Ladder
import Apeiron.Physics.Wick

/-!
# Where the number eight in `d = 8e + 2u` comes from, and what it is not

@name Period
@cite bott-1959
@cite atiyah-bott-shapiro-1964
@cite hurwitz-1898
@cite conway-sloane-1999
@grade proved

`Apeiron.Physics.Ladder` counts dimensions as `8e + 2u` and writes both coefficients as
`Module.finrank ℝ Cell` and four times it. This file replaces the `4 ×` with a construction:
`Doubling.finrank_double` says each Cayley-Dickson step doubles the real dimension, so iterating
from `ℝ` gives `1, 2, 4, 8, 16`, and `finrank_octonion` is `2³ = 8` counted rather than asserted.
`ladder_period_is_octonion` then rewrites the ladder with `dim_ℝ 𝕆` in place of the literal.

## Three statements about dimension eight, and one that is cited

1. **`𝕆` has no zero divisors and `𝕊` does.** `Octonion.no_zero_divisors` and
   `Sedenion.mul_eq_zero`. The first follows from multiplicativity plus anisotropy, the second is
   a witness.
2. **The failure at dimension 16 is of composition, not of anisotropy.** `Sedenion.no_null_vector`
   says the sedenion form has no non-zero isotropic vector, so the break there is not a null
   cone. This corrects a reading in the predecessor corpus, whose `sedenion_break_is_null_cell`
   conjoined the sedenion zero divisor with the split-complex null pair and read the conjunction
   as one signature. The two are different: the split-complex zero divisor lies **on** its cone,
   and the sedenion zero divisor lies **off** its cone because there is no cone.
3. **The definite and split forms in a given dimension are one analytic family.** At `A = ℝ` that
   is `Physics.wick_flips_the_sign`, reused rather than restated, and
   `definite_form_is_the_split_form_at_imaginary_second_slot` is the reading of it in the
   doubling's own notation.

What is **not** proved anywhere here: that the `8` above is the period of the real Clifford
algebras, `Cl(0,8) ≅ M₁₆(ℝ)` (Atiyah, Bott and Shapiro 1964), or the period of real topological
K-theory (Bott 1959). No Clifford algebra, module classification or homotopy group occurs in any
statement in this file. Those are cited as the name of the coincidence of values, and the
coincidence is not a theorem of this corpus.

Likewise, that dimension eight is the largest with a positive definite composition algebra is
Hurwitz 1898, cited. The corpus proves the two instances it uses and not the classification.

## Root

| from | here |
|---|---|
| `finrank_double`, `Quaternion.finrank_eq_four` | `finrank_octonion`, `finrank_sedenion` |
| `finrank_double` at `σ = 1` | `finrank_splitOct` |
| `Physics.cell_finrank`, `Physics.dimOf` | `ladder_period_is_octonion` |
| `Physics.wick_flips_the_sign`, `real_norm` | `definite_form_is_the_split_form_at_...` |

The second row is what replaces a literal with a dimension: `Physics.dimOf`'s `8` becomes
`Module.finrank ℝ Octonion` and its `2` becomes `Module.finrank ℝ Cell`, so both coefficients name
algebras in one tower whose bottom is `Apeiron/Trichotomy`. The third row reuses the Wick theorem
rather than restating it, which is why nothing analytic is proved twice.
-/

namespace Apeiron
namespace Doubling

open Trichotomy

/-- **The quaternions have four real dimensions.**

    Module.finrank ℝ Quat = 4         [four independent real directions]
    Module.finrank                    [how many independent directions the space has]
    ℝ                                 [the real numbers: every number on the number line] -/
theorem finrank_quat : Module.finrank ℝ Quat = 4 := Quaternion.finrank_eq_four

/-- **The octonions have eight real dimensions.**

    Module.finrank ℝ Octonion = 8     [eight independent real directions]
    Module.finrank                    [how many independent directions the space has]
    ℝ                                 [the real numbers]

    So: `dim_ℝ CD(A, σ) = 2·dim_ℝ A` applied to the quaternions. The eight is counted, and the
    same count gives `1, 2, 4, 8, 16` along the tower from `ℝ`. -/
theorem finrank_octonion : Module.finrank ℝ Octonion = 8 := by
  rw [finrank_double, finrank_quat]

/-- **The sedenions have sixteen real dimensions.**

    Module.finrank ℝ Sedenion = 16    [sixteen independent real directions]
    Module.finrank                    [how many independent directions the space has]
    ℝ                                 [the real numbers] -/
theorem finrank_sedenion : Module.finrank ℝ Sedenion = 16 := by
  rw [finrank_double, finrank_octonion]

/-- **The split octonions have eight real dimensions too.**

    Module.finrank ℝ SplitOct = 8     [eight independent real directions]
    Module.finrank                    [how many independent directions the space has]
    ℝ                                 [the real numbers]

    So: the parameter changes the form and the zero divisors, not the dimension. -/
theorem finrank_splitOct : Module.finrank ℝ SplitOct = 8 := by
  rw [finrank_double, finrank_quat]

/-- **The ladder's two coefficients are the dimensions of two algebras in the tower.**

    (e u : ℕ)                                                        [two counts]
    Physics.dimOf e u
      = Module.finrank ℝ Octonion * e + Module.finrank ℝ Cell * u     [the ladder, in dimensions]
    dimOf                  [how many real directions so many definite blocks and so many cells make]
    Module.finrank                                   [how many independent directions the space has]
    Cell                                            [a point: two real numbers written side by side]
    ℕ                                                                [the counting numbers]
    ℝ                                                                [the real numbers]

    So: `d = 8e + 2u` with `8 = dim_ℝ 𝕆` and `2 = dim_ℝ ℝ[x]/(x² − σ)`, both counted. That this
    modulus is the period of the real Clifford algebras or of real K-theory is cited in the header
    and proved nowhere. -/
theorem ladder_period_is_octonion (e u : ℕ) :
    Physics.dimOf e u = Module.finrank ℝ Octonion * e + Module.finrank ℝ Cell * u := by
  rw [finrank_octonion, Physics.cell_finrank]
  unfold Physics.dimOf
  ring

/-- **The definite form is the split form with an imaginary second component.**

    (z : Cell)                                                       [a point]
    ((z.1 : ℂ)) ^ 2 - ((1 : ℝ) : ℂ) * ((z.2 : ℂ) * Complex.I) ^ 2  [the split form, second slot]
    = ((doubleNorm (fun a : ℝ => a ^ 2) (ofPair z : Double ℝ (-1)) : ℝ) : ℂ)
    doubleNorm                                                       [the doubled quadratic form]
    ofPair                                              [read a pair into the double]
    Double                                                         [the Cayley-Dickson double A + A]
    Cell                                                           [a point: two reals side by side]
    ℝ                                                                [the real numbers]
    ℂ                           [the complex numbers: pairs of reals where the second squares to −1]

    So: `Physics.wick_flips_the_sign` at `σ = 1`, transported by `Doubling.real_norm`. The two
    real forms of a given dimension are two values of one analytic family, so the split and
    definite towers are not two constructions. Nothing is restated here: the analytic content is
    the cited theorem and this is its reading in the doubling's notation. -/
theorem definite_form_is_the_split_form_at_imaginary_second_slot (z : Cell) :
    ((z.1 : ℂ)) ^ 2 - ((1 : ℝ) : ℂ) * ((z.2 : ℂ) * Complex.I) ^ 2
      = ((doubleNorm (fun a : ℝ => a ^ 2) (ofPair z : Double ℝ (-1)) : ℝ) : ℂ) := by
  rw [real_norm]
  exact Physics.wick_flips_the_sign 1 z

end Doubling
end Apeiron
