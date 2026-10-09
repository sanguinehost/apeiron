import Apeiron.Doubling.CayleyDickson
import Mathlib.Algebra.Quaternion

/-!
# `𝕆 = CD(ℍ, −1)`: an eight-dimensional definite composition algebra

@name Octonion
@cite graves-cayley-1845
@cite dickson-1919
@cite hurwitz-1898
@grade proved

`Octonion` is `Doubling.Double Quat (-1)`, the Cayley-Dickson double of Mathlib's real
quaternions at parameter `−1`. The construction is not rebuilt here. What this file supplies is
the quadratic form the doubling does not fix on its own, and the two properties of that form that
decide whether the algebra is a division algebra:

```
   N(a, b) = |a|² + |b|²                       a positive definite form in eight real variables
   N(x·y)  = N(x)·N(y)                         Degen's eight-square identity (normO_mul)
   N(x) = 0 ↔ x = 0                            anisotropy (normO_eq_zero_iff)
```

Multiplicativity is a polynomial identity in sixteen real variables, closed by `ring`. It cannot
be obtained from associativity, since `not_associative` exhibits an associator that does not
vanish. Anisotropy is not proved here at all: it is
`Doubling.doubleNorm_eq_zero_iff_of_neg` at `σ = −1`, the same statement that gives the sedenion
form its anisotropy.

Multiplicativity together with anisotropy excludes zero divisors, which is
`Doubling.no_break_of_composition_of_anisotropy`; `no_zero_divisors` is that theorem applied
here. No inverse is constructed and no invertibility statement appears below, so `𝕆` is shown to
have no zero divisors rather than shown to be a division algebra. Hurwitz's theorem, that the
positive definite composition algebras over `ℝ` are exactly those of dimension 1, 2, 4 and 8, is
cited rather than proved.

Mathlib has quaternions, and as of the 2026-09-02 sweep no octonions and no Cayley-Dickson
construction. The doubling here is the corpus's own.

## Root

| from | here |
|---|---|
| `doubleNorm_eq_zero_iff_of_neg` | `normO_eq_zero_iff` |
| `no_break_of_composition_of_anisotropy` | `no_zero_divisors` |
| `fst_mul`, `snd_mul` at `σ = −1` | `fst_mul_octonion`, `normO_mul` |

The first row is the one that matters: anisotropy here is not proved from eight squares but is the
`σ = −1` case of the isotropy theorem, whose `A = ℝ` case is
`Trichotomy.exists_normForm_zero_iff_disc_nonneg`. The carrier reaches the same place through
`Doubling.real_mul`. The second row means the octonions have no zero divisors for a reason that
mentions no dimension, which is why the same theorem contraposes onto dimension sixteen.
-/

namespace Apeiron
namespace Doubling

/-- Mathlib's real quaternions, the base of this doubling.

    Quat                          [the real quaternions]
    ℝ                             [the real numbers: every number on the number line] -/
abbrev Quat : Type := Quaternion ℝ

/-- The octonions `𝕆 = CD(ℍ, −1)`: the Cayley-Dickson double of the quaternions at parameter −1.

    Octonion                      [the octonions, an eight-dimensional real algebra]
    Double                        [the Cayley-Dickson double A + A] -/
abbrev Octonion : Type := Double Quat (-1)

/-- The octonion norm form: the doubled form built from the quaternion norm form.

    (x : Octonion)                [an octonion]
    : ℝ                           [a real number]
    doubleNorm                    [the doubled quadratic form]
    ℝ                             [the real numbers] -/
def normO (x : Octonion) : ℝ := doubleNorm (fun a : Quat => Quaternion.normSq a) x

/-- **The form is the sum of the two quaternion norm forms.**

    (x : Octonion)                                                        [an octonion]
    normO x = Quaternion.normSq (fst x) + Quaternion.normSq (snd x)   [eight real squares]
    normO                                                                 [the octonion norm form]

    So: the plus sign is the value `σ = −1` substituted into `N(a) − σ·N(b)`, not a convention. -/
theorem normO_eq (x : Octonion) :
    normO x = Quaternion.normSq (fst x) + Quaternion.normSq (snd x) := by
  unfold normO doubleNorm
  ring

/-- The first component of a product, with the parameter's minus sign carried out.

    (x y : Octonion)                                                      [two octonions]
    fst (x * y) = fst x * fst y - star (snd y) * snd x                    [the first component]
    Double                                                                [the double A + A] -/
theorem fst_mul_octonion (x y : Octonion) :
    fst (x * y) = fst x * fst y - star (snd y) * snd x := by
  rw [fst_mul, neg_one_smul, ← sub_eq_add_neg]

/-- **The form is multiplicative: Degen's eight-square identity.**

    (x y : Octonion)                                     [two octonions]
    normO (x * y) = normO x * normO y                    [the form is multiplicative]
    normO                                                [the octonion norm form]

    So: `𝕆` is a composition algebra. The proof is a polynomial identity in the sixteen real
    components and uses no associativity, which `not_associative` shows it does not have. -/
theorem normO_mul (x y : Octonion) : normO (x * y) = normO x * normO y := by
  simp only [normO_eq, fst_mul_octonion, snd_mul]
  simp [Quaternion.normSq_def', Quaternion.re_mul, Quaternion.imI_mul, Quaternion.imJ_mul,
    Quaternion.imK_mul]
  ring

/-- **The form is positive semidefinite.**

    (x : Octonion)                    [an octonion]
    0 ≤ normO x                       [its form value is at zero or above]
    normO                             [the octonion norm form]
    ≤                                 [is smaller than or equal to] -/
theorem normO_nonneg (x : Octonion) : 0 ≤ normO x := by
  rw [normO_eq]
  exact add_nonneg Quaternion.normSq_nonneg Quaternion.normSq_nonneg

/-- **The form is anisotropic, because the parameter is below zero.**

    (x : Octonion)                    [an octonion]
    normO x = 0 ↔ x = 0               [the form vanishes only at zero]
    normO                             [the octonion norm form]
    ↔                                 [exactly when, true in both directions]

    So: `Doubling.doubleNorm_eq_zero_iff_of_neg` at `σ = −1`, not a computation in eight squares.
    The sedenion form is anisotropic by the same instantiation. -/
theorem normO_eq_zero_iff (x : Octonion) : normO x = 0 ↔ x = 0 :=
  doubleNorm_eq_zero_iff_of_neg (σ := (-1 : ℝ)) (fun a : Quat => Quaternion.normSq a)
    (by norm_num) (fun _ => Quaternion.normSq_nonneg) (fun _ => Quaternion.normSq_eq_zero) x

/-- The unit multiplies on the left without changing anything.

    (x : Octonion)                    [an octonion]
    (1 : Octonion) * x = x            [the unit is a two-sided identity]
    Octonion                          [the octonions, an eight-dimensional real algebra] -/
theorem one_mul_octonion (x : Octonion) : (1 : Octonion) * x = x := by
  refine ext ?_ ?_ <;> simp

/-- The unit multiplies on the right without changing anything.

    (x : Octonion)                    [an octonion]
    x * (1 : Octonion) = x            [the unit is a two-sided identity]
    Octonion                          [the octonions, an eight-dimensional real algebra] -/
theorem mul_one_octonion (x : Octonion) : x * (1 : Octonion) = x := by
  refine ext ?_ ?_
  · show fst x * 1 + (-1 : ℝ) • (star (0 : Quat) * snd x) = fst x
    rw [star_zero, zero_mul, smul_zero, add_zero, mul_one]
  · show (0 : Quat) * fst x + snd x * star (1 : Quat) = snd x
    rw [zero_mul, star_one, mul_one, zero_add]

/-- **The algebra has no zero divisors.**

    {x y : Octonion}                  [two octonions]
    (hx : x ≠ 0) (hy : y ≠ 0)         [neither one is zero]
    x * y ≠ 0                         [their product is not zero either]
    Octonion                          [the octonions, an eight-dimensional real algebra]
    ≠                                 [is not equal to]

    So: `Doubling.no_break_of_composition_of_anisotropy` applied to `normO_mul` and
    `normO_eq_zero_iff`. Multiplicativity together with anisotropy excludes a zero divisor. The
    sedenion form is anisotropic too, so what fails there is the other hypothesis. -/
theorem no_zero_divisors {x y : Octonion} (hx : x ≠ 0) (hy : y ≠ 0) : x * y ≠ 0 :=
  no_break_of_composition_of_anisotropy (fun u v : Octonion => u * v) normO 0
    ((normO_eq_zero_iff 0).mpr rfl) normO_mul normO_eq_zero_iff hx hy

/-- **The algebra is not associative.**

    ∃ a b c : Octonion, (a * b) * c ≠ a * (b * c)    [three octonions with non-zero associator]
    Octonion                                         [the octonions]
    ∃                                                     [there is at least one]
    ≠                                                     [is not equal to]

    So: the witness is `a = (i, 0)`, `b = (j, 0)`, `c = (0, 1)`, with `(ab)c = (0, k)` and
    `a(bc) = (0, −k)`, differing in one real coordinate. Nothing here connects the failure of
    associativity to the sedenion zero divisor: they are separate facts and the corpus asserts no
    implication between them. -/
theorem not_associative : ∃ a b c : Octonion, (a * b) * c ≠ a * (b * c) := by
  -- The two bracketings in closed form, for the fallback routes below. With `a = (i, 0)`,
  -- `b = (j, 0)`, `c = (0, 1)`: `a * b = (k, 0)`, so `(a * b) * c = (0, k)`; and `b * c = (0, j)`,
  -- so `a * (b * c) = (0, j * i) = (0, -k)`. Both first components vanish because the second slot
  -- of one factor is `0` in each product.
  --
  -- Route 1, below: read one real coordinate off the equation and let `norm_num` expand the
  -- quaternion products in place.
  --
  -- Route 2, if `norm_num` does not reach through `fst_mul` and `snd_mul`: name both sides first
  -- and compare the normal forms.
  --   have hL : (ofPair (⟨0, 1, 0, 0⟩, 0) * ofPair (⟨0, 0, 1, 0⟩, 0) : Octonion)
  --       * ofPair (0, 1) = ofPair (0, ⟨0, 0, 0, 1⟩) := by
  --     refine ext ?_ ?_ <;> refine QuaternionAlgebra.ext ?_ ?_ ?_ ?_ <;>
  --       simp [fst_mul, snd_mul, neg_one_smul, Quaternion.re_mul, Quaternion.imI_mul,
  --         Quaternion.imJ_mul, Quaternion.imK_mul]
  --   have hR : (ofPair (⟨0, 1, 0, 0⟩, 0) : Octonion)
  --       * (ofPair (⟨0, 0, 1, 0⟩, 0) * ofPair (0, 1)) = ofPair (0, ⟨0, 0, 0, -1⟩) := by
  --     refine ext ?_ ?_ <;> refine QuaternionAlgebra.ext ?_ ?_ ?_ ?_ <;>
  --       simp [fst_mul, snd_mul, neg_one_smul, Quaternion.re_mul, Quaternion.imI_mul,
  --         Quaternion.imJ_mul, Quaternion.imK_mul]
  --   rw [hL, hR] at h
  --   exact absurd (congrArg (fun q : Octonion => (snd q).imK) h) (by norm_num)
  --
  -- Route 3, if the structure `ext` lemma takes a different arity: drop to the component
  -- equivalence instead of applying `ext`.
  --   simp only [Doubling.fst, Doubling.snd, Doubling.ofPair] at hk
  --   norm_num [QuaternionAlgebra.ext_iff, Quaternion.re_mul, Quaternion.imI_mul,
  --     Quaternion.imJ_mul, Quaternion.imK_mul] at hk
  refine ⟨ofPair (⟨0, 1, 0, 0⟩, 0), ofPair (⟨0, 0, 1, 0⟩, 0), ofPair (0, 1), fun h => ?_⟩
  have hk := congrArg (fun q : Octonion => (snd q).imK) h
  norm_num [fst_mul, snd_mul, neg_one_smul, Quaternion.re_mul, Quaternion.imI_mul,
    Quaternion.imJ_mul, Quaternion.imK_mul] at hk

end Doubling
end Apeiron
