import Apeiron.Doubling.Octonion

/-!
# `𝕊 = CD(𝕆, −1)`: anisotropic, with zero divisors, and therefore not a composition algebra

@name Sedenion
@cite dickson-1919
@cite hurwitz-1898
@grade proved

`Sedenion` is `Doubling.Double Octonion (-1)`, dimension 16. The Cayley-Dickson rule reads seven
operations off its base and no law, so it applies over a non-associative base unchanged. The
algebra exists; the question is which property it loses.

Not isotropy. The parameter is `−1`, so `Doubling.doubleNorm_eq_zero_iff_of_neg` applies exactly
as it does for `𝕆`, and `no_null_vector` states that the sedenion form has no non-zero isotropic
vector. Whatever fails in dimension 16, it is not the appearance of a null cone, and the zero
divisors below lie off the cone rather than on it.

What fails is composition. `Doubling.no_composition_of_anisotropy_of_break` gives that a form
vanishing only at the origin cannot be multiplicative for an operation with a zero divisor. Since
the algebra is anisotropic for **every** such form, `no_anisotropic_composition_norm` states the
conclusion for an arbitrary anisotropic real-valued function on `𝕊`, not for one choice. The cap
at dimension 8 is therefore a statement about composition and not about a cone, and
`normSed_not_multiplicative` is a corollary rather than a sixteen-square computation.

The zero divisor itself is a witness and is the only computation in the file: `mul_eq_zero` gives
`(e₁ + e₁₀)·(e₅ + e₁₄) = 0`. It is an input to the argument above, not the argument.

Nothing here proves that the failure of associativity in `𝕆` causes the zero divisor in `𝕊`.
They are separate facts and the corpus asserts no implication between them.

## Root

| from | here |
|---|---|
| `doubleNorm_eq_zero_iff_of_neg` | `normSed_eq_zero_iff`, `no_null_vector` |
| `no_composition_of_anisotropy_of_break` | `no_anisotropic_composition_norm` |
| `no_anisotropic_composition_norm` | `normSed_not_multiplicative` |

The first row is what corrects the predecessor's reading. Anisotropy in dimension sixteen is the
same theorem as anisotropy in dimension eight, read at the same `σ = −1`, whose `A = ℝ` case is
`Trichotomy.exists_normForm_zero_iff_disc_nonneg`. So the zero divisors here are off the cone
because there is no cone, and the second row says what gives way instead.
-/

namespace Apeiron
namespace Doubling

/-- The sedenions `𝕊 = CD(𝕆, −1)`: the double of the octonions at parameter −1, dimension 16.

    Sedenion                      [the sedenions, a sixteen-dimensional real algebra]
    Double                        [the Cayley-Dickson double A + A] -/
abbrev Sedenion : Type := Double Octonion (-1)

/-- The sedenion norm form: the doubled form built from the octonion norm form.

    (x : Sedenion)                [a sedenion]
    : ℝ                           [a real number]
    doubleNorm                    [the doubled quadratic form]
    ℝ                             [the real numbers] -/
def normSed (x : Sedenion) : ℝ := doubleNorm normO x

/-- **The form is the sum of the two octonion norm forms.**

    (x : Sedenion)                                        [a sedenion]
    normSed x = normO (fst x) + normO (snd x)             [sixteen real squares]
    normSed                                               [the sedenion norm form]
    normO                                                 [the octonion norm form] -/
theorem normSed_eq (x : Sedenion) : normSed x = normO (fst x) + normO (snd x) := by
  unfold normSed doubleNorm
  ring

/-- **The form is anisotropic.**

    (x : Sedenion)                    [a sedenion]
    normSed x = 0 ↔ x = 0             [the form vanishes only at zero]
    normSed                           [the sedenion norm form]
    ↔                                 [exactly when, true in both directions]

    So: `Doubling.doubleNorm_eq_zero_iff_of_neg` at `σ = −1`, the same instantiation that makes the
    octonion form anisotropic. -/
theorem normSed_eq_zero_iff (x : Sedenion) : normSed x = 0 ↔ x = 0 :=
  doubleNorm_eq_zero_iff_of_neg (σ := (-1 : ℝ)) normO (by norm_num) normO_nonneg
    normO_eq_zero_iff x

/-- **The sedenion form has no non-zero isotropic vector.**

    ¬ ∃ x : Sedenion, x ≠ 0 ∧ normSed x = 0              [the form has no non-zero isotropic vector]
    normSed                                              [the sedenion norm form]
    ¬                                                    [this is not true]
    ∃                                                    [there is at least one]
    ∧                                                    [both are true]
    ≠                                                    [is not equal to]

    So: what fails in dimension 16 is not isotropy. This is
    `Doubling.exists_null_iff_sigma_nonneg` at `σ = −1`, where the right-hand side is false, so the
    zero divisors below are anisotropic vectors. -/
theorem no_null_vector : ¬ ∃ x : Sedenion, x ≠ 0 ∧ normSed x = 0 := by
  rintro ⟨x, hx, hn⟩
  exact hx ((normSed_eq_zero_iff x).mp hn)

/-- The first factor of the break: `e₁ + e₁₀`.

    breakLeft                     [the first factor of the zero divisor]
    Sedenion                      [the sedenions, a sixteen-dimensional real algebra] -/
noncomputable def breakLeft : Sedenion :=
  ofPair (ofPair (⟨0, 1, 0, 0⟩, 0), ofPair (⟨0, 0, 1, 0⟩, 0))

/-- The second factor of the break: `e₅ + e₁₄`.

    breakRight                    [the second factor of the zero divisor]
    Sedenion                      [the sedenions, a sixteen-dimensional real algebra] -/
noncomputable def breakRight : Sedenion :=
  ofPair (ofPair (0, ⟨0, 1, 0, 0⟩), ofPair (0, ⟨0, 0, 1, 0⟩))

open Quaternion in
/-- **The two factors have product zero.**

    breakLeft * breakRight = 0    [their product vanishes]
    breakLeft                     [the first factor of the zero divisor]
    breakRight                    [the second factor of the zero divisor]

    So: the witness. This is the only computation in the file; everything else is an instance of a
    theorem stated over an arbitrary carrier. -/
theorem mul_eq_zero : breakLeft * breakRight = 0 := by
  -- The four octonion products this reduces to, for the fallback routes below. Write
  -- `A = (i, 0)`, `B = (j, 0)`, `C = (0, i)`, `D = (0, j)` as octonions, so that
  -- `breakLeft = (A, B)` and `breakRight = (C, D)`. Then
  --   `A * C = (0, -1)`,  `star D * B = (0, -1)`,  `D * A = (0, k)`,  `B * star C = (0, -k)`,
  -- so the first component is `A * C - star D * B = 0` and the second is
  -- `D * A + B * star C = 0`. Both cancellations are between two equal quaternions, so any route
  -- that reaches the four quaternion coordinates closes the goal.
  --
  -- Route 1, taken below and confirmed by the build: split twice with `ext`, let `simp` expand
  -- the doubling to octonion and then quaternion arithmetic, and finish the two components that
  -- survive with the structure `ext` and `norm_num` on the quaternion product lemmas. The
  -- component lemmas `re_mul`, `imI_mul`, `imJ_mul`, `imK_mul` are not `simp` in Mathlib, which
  -- is why they are listed, and the projections of `0` and `1` are `scoped simp`, which is why
  -- the declaration opens `Quaternion`.
  --
  -- Route 2, if `simp` loops or stalls on the nested products: name the four octonion products
  -- first, then rewrite.
  --   have h1 : (ofPair (⟨0, 1, 0, 0⟩, 0) : Octonion) * ofPair (0, ⟨0, 1, 0, 0⟩)
  --       = ofPair (0, -1) := by
  --     refine ext ?_ ?_ <;> simp [fst_mul, snd_mul, neg_one_smul, QuaternionAlgebra.ext_iff]
  --   (and the three companions, then `refine ext ?_ ?_ <;> simp [breakLeft, breakRight, h1, ...]`)
  --
  -- Route 3, if the doubled `ext` does not fire: drop to the underlying pairs and compare
  -- coordinates in one step.
  --   simp only [breakLeft, breakRight, Doubling.fst, Doubling.snd, Doubling.ofPair,
  --     Prod.ext_iff, QuaternionAlgebra.ext_iff]
  --   norm_num [fst_mul, snd_mul, neg_one_smul, Quaternion.re_mul, Quaternion.imI_mul,
  --     Quaternion.imJ_mul, Quaternion.imK_mul]
  refine Doubling.ext ?_ ?_ <;> refine Doubling.ext ?_ ?_ <;>
    simp [breakLeft, breakRight] <;>
    refine QuaternionAlgebra.ext ?_ ?_ ?_ ?_ <;>
    norm_num [Quaternion.re_mul, Quaternion.imI_mul, Quaternion.imJ_mul, Quaternion.imK_mul]

/-- **The first factor is not zero.**

    breakLeft ≠ 0                 [it is not the zero sedenion]
    breakLeft                     [the first factor of the zero divisor]
    ≠                             [is not equal to] -/
theorem breakLeft_ne_zero : breakLeft ≠ 0 := by
  intro h
  have h1 := congrArg (fun x : Sedenion => (fst (fst x)).imI) h
  norm_num [breakLeft, Quaternion.imI_zero] at h1

/-- **The second factor is not zero.**

    breakRight ≠ 0                [it is not the zero sedenion]
    breakRight                    [the second factor of the zero divisor]
    ≠                             [is not equal to] -/
theorem breakRight_ne_zero : breakRight ≠ 0 := by
  intro h
  have h1 := congrArg (fun x : Sedenion => (snd (fst x)).imI) h
  norm_num [breakRight, Quaternion.imI_zero] at h1

/-- **No anisotropic real-valued form on `𝕊` is multiplicative.**

    (N : Sedenion → ℝ)                                   [any real-valued function on the algebra]
    (ha : ∀ x, N x = 0 ↔ x = 0)                          [it vanishes only at zero]
    ¬ ∀ x y, N (x * y) = N x * N y                       [it is not multiplicative]
    Sedenion                                             [the sedenions]
    ∀                                                    [for every single one]
    ¬                                                    [this is not true]
    ↔                                                    [exactly when, true in both directions]
    →                                                    [takes the left and gives the right]
    ℝ                                                    [the real numbers]

    So: the cap is not a fact about one choice of form. A zero divisor together with anisotropy
    excludes multiplicativity on any carrier (`Doubling.no_composition_of_anisotropy_of_break`),
    and `𝕊` is anisotropic for every such form. Hurwitz's theorem states the same boundary from
    the other side and is cited, not proved. -/
theorem no_anisotropic_composition_norm (N : Sedenion → ℝ) (ha : ∀ x, N x = 0 ↔ x = 0) :
    ¬ ∀ x y, N (x * y) = N x * N y :=
  no_composition_of_anisotropy_of_break (fun u v : Sedenion => u * v) N 0 ((ha 0).mpr rfl) ha
    breakLeft_ne_zero breakRight_ne_zero mul_eq_zero

/-- **The algebra's own norm form is not multiplicative.**

    ∃ x y : Sedenion, normSed (x * y) ≠ normSed x * normSed y      [multiplicativity fails]
    normSed                                                        [the sedenion norm form]
    Sedenion                                                       [the sedenions]
    ∃                                                              [there is at least one]
    ≠                                                              [is not equal to]

    So: the previous theorem at `N = normSed`, not a sixteen-square computation. -/
theorem normSed_not_multiplicative :
    ∃ x y : Sedenion, normSed (x * y) ≠ normSed x * normSed y := by
  by_contra h
  push Not at h
  exact no_anisotropic_composition_norm normSed normSed_eq_zero_iff h

end Doubling
end Apeiron
