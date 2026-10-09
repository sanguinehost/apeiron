import Apeiron.Doubling.Octonion
import Apeiron.Trichotomy.SplitComplex

/-!
# `𝕆' = CD(ℍ, +1)`: the split real form, isotropic and with zero divisors

@name SplitOctonion
@cite zorn-1933
@cite dickson-1919
@grade proved

`SplitOct` is `Doubling.Double Quat 1`. It differs from `Octonion` in the value of one real
parameter, and every difference between the two algebras is a consequence of that value:

```
   Octonion = CD(ℍ, −1)     N(a, b) = |a|² + |b|²     anisotropic, no zero divisors
   SplitOct = CD(ℍ, +1)     N(a, b) = |a|² − |b|²     isotropic,   zero divisors
```

Both are composition algebras. `normS_mul` is the same sixteen-variable polynomial identity as
`Octonion.normO_mul` with one sign changed, so multiplicativity is not what distinguishes them,
and Hurwitz's cap at dimension 8 therefore does not distinguish them either.

What distinguishes them is isotropy, and isotropy is fixed by the parameter.
`Doubling.exists_null_iff_sigma_nonneg` gives a non-trivial zero of the form exactly when
`σ ≥ 0`, so `null_cone` is that theorem at `σ = 1` rather than an exhibited vector. The zero
divisor is likewise not exhibited here: `Doubling.split_mul_eq_zero` proves
`(1, 1)·(1, −1) = 0` for the double of any associative `*`-algebra at parameter one, so
`ℝ ⊕ ℝ`, `M₂(ℝ)` and `𝕆'` are three instances of one statement.
`splitComplex_is_the_real_double` records the first of those as an equation with the
corpus's own `σ = 1` algebra.

Zorn 1933 is the reference for this algebra. That it is the unique second real form of the
eight-dimensional composition algebra is not proved here and is not used.

## Root

| from | here |
|---|---|
| `exists_null_iff_sigma_nonneg` | `null_cone` |
| `split_mul_eq_zero` | `zero_divisor` |
| `split_left_ne_zero`, `split_right_ne_zero` | `zero_divisor_factors_ne_zero` |
| `SplitComplex.mul_eq_mul_one`, `real_mul` | `splitComplex_is_the_real_double` |

The first two rows are the cull: the null cone is the isotropy theorem at `σ = 1` and not an
exhibited vector, and the zero divisor is a theorem about any associative base and not a
computation in eight squares. The third row closes the loop, identifying the corpus's own `σ = 1`
algebra with the first member of this same family.
-/

namespace Apeiron
namespace Doubling

open Trichotomy

/-- The split octonions `𝕆' = CD(ℍ, +1)`: the double of the quaternions at parameter +1.

    SplitOct                      [the split octonions, an eight-dimensional real algebra]
    Double                        [the Cayley-Dickson double A + A] -/
abbrev SplitOct : Type := Double Quat 1

/-- The split octonion norm form: the doubled form at parameter +1.

    (x : SplitOct)                [a split octonion]
    : ℝ                           [a real number]
    doubleNorm                    [the doubled quadratic form]
    ℝ                             [the real numbers] -/
def normS (x : SplitOct) : ℝ := doubleNorm (fun a : Quat => Quaternion.normSq a) x

/-- **The form is the difference of the two quaternion norm forms.**

    (x : SplitOct)                                                        [a split octonion]
    normS x = Quaternion.normSq (fst x) - Quaternion.normSq (snd x)   [four squares less four]
    normS                                                             [the split norm form]

    So: the minus sign is the value `σ = 1` substituted into `N(a) − σ·N(b)`. -/
theorem normS_eq (x : SplitOct) :
    normS x = Quaternion.normSq (fst x) - Quaternion.normSq (snd x) := by
  unfold normS doubleNorm
  ring

/-- The first component of a product, with the parameter's plus sign carried out.

    (x y : SplitOct)                                                      [two split octonions]
    fst (x * y) = fst x * fst y + star (snd y) * snd x                    [the first component]
    Double                                                                [the double A + A] -/
theorem fst_mul_split (x y : SplitOct) :
    fst (x * y) = fst x * fst y + star (snd y) * snd x := by
  rw [fst_mul, one_smul]

/-- **The indefinite form is multiplicative too.**

    (x y : SplitOct)                                     [two split octonions]
    normS (x * y) = normS x * normS y                    [the form is multiplicative]
    normS                                                [the split octonion norm form]

    So: the parameter does not control composition. Both real forms in dimension eight are
    composition algebras, so Hurwitz's cap does not separate them. -/
theorem normS_mul (x y : SplitOct) : normS (x * y) = normS x * normS y := by
  simp only [normS_eq, fst_mul_split, snd_mul]
  simp [Quaternion.normSq_def', Quaternion.re_mul, Quaternion.imI_mul, Quaternion.imJ_mul,
    Quaternion.imK_mul]
  ring

/-- **The form is indefinite.**

    ∃ p q : SplitOct, 0 < normS p ∧ normS q < 0     [one vector of positive value, one negative]
    normS                                                [the split octonion norm form]
    ∃                                                    [there is at least one]
    ∧                                                    [both are true]
    <                                                    [is smaller than]

    So: the form takes both signs. Its signature is not computed here; the statement asserts two
    values of opposite sign and nothing more. -/
theorem normS_indefinite : ∃ p q : SplitOct, 0 < normS p ∧ normS q < 0 := by
  refine ⟨ofPair (1, 0), ofPair (0, 1), ?_, ?_⟩ <;>
    simp [normS_eq]

/-- **The form is isotropic, because the parameter is above zero.**

    ∃ v : SplitOct, v ≠ 0 ∧ normS v = 0                  [the form has a non-zero isotropic vector]
    normS                                                [the split octonion norm form]
    ∃                                                    [there is at least one]
    ∧                                                    [both are true]
    ≠                                                    [is not equal to]

    So: `Doubling.exists_null_iff_sigma_nonneg` at `σ = 1`. This null cone and the null cone of
    `ℝ[x]/(x² − σ)` above zero are one theorem at two base algebras. -/
theorem null_cone : ∃ v : SplitOct, v ≠ 0 ∧ normS v = 0 :=
  (exists_null_iff_sigma_nonneg (σ := (1 : ℝ)) (fun a : Quat => Quaternion.normSq a)
    (fun _ => Quaternion.normSq_nonneg) (fun _ => Quaternion.normSq_eq_zero)
    (map_one Quaternion.normSq) (fun r a => Quaternion.normSq_smul r a)).mpr (by norm_num)

/-- **The algebra has a zero divisor.**

    (ofPair (1, 1) : SplitOct) * ofPair (1, -1) = 0      [their product is zero]
    SplitOct                                             [the split octonions]
    Quat                                                 [the real quaternions]
    ofPair                                               [read a pair into the double]
    ×                                                    [paired with]

    So: `Doubling.split_mul_eq_zero` at the quaternion base. The first component evaluates to
    `1·1 + σ·(−1)·1`, which is `0` at `σ = 1` and `2` at `σ = −1`. -/
theorem zero_divisor : (ofPair ((1 : Quat), (1 : Quat)) : SplitOct)
    * ofPair ((1 : Quat), (-1 : Quat)) = 0 :=
  split_mul_eq_zero

/-- **Neither factor is zero.**

    (ofPair (1, 1) : SplitOct) ≠ 0 ∧ (ofPair (1, -1) : SplitOct) ≠ 0     [both factors are non-zero]
    SplitOct                                                             [the split octonions]
    Quat                                                                 [the real quaternions]
    ofPair                                                               [read a pair in]
    ∧                                                                    [both are true]
    ≠                                                                    [is not equal to]
    ×                                                                    [paired with] -/
theorem zero_divisor_factors_ne_zero :
    (ofPair ((1 : Quat), (1 : Quat)) : SplitOct) ≠ 0
      ∧ (ofPair ((1 : Quat), (-1 : Quat)) : SplitOct) ≠ 0 :=
  ⟨split_left_ne_zero, split_right_ne_zero⟩

/-- **The corpus's `σ = 1` algebra is `CD(ℝ, 1)`.**

    (z w : SC)                                                       [two split-complex points]
    (ofPair z : Double ℝ 1) * ofPair w = ofPair (SplitComplex.mul z w)
    Double                                                           [the double A + A]
    ofPair                                                   [read a pair into the double]
    SC                                             [a point in the σ = 1 algebra: two reals a + b·j]
    mul                                                    [multiply two points together, the σ way]
    ℝ                                                                [the real numbers]

    So: `ℝ ⊕ ℝ`, `M₂(ℝ)` and `𝕆'` are one construction at three bases, and the first of them is the
    algebra the rest of the corpus is written on. -/
theorem splitComplex_is_the_real_double (z w : SplitComplex.SC) :
    (ofPair z : Double ℝ 1) * ofPair w = ofPair (SplitComplex.mul z w) := by
  rw [real_mul, SplitComplex.mul_eq_mul_one]

end Doubling
end Apeiron
