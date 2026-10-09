import Apeiron.Trichotomy.Basic
import Apeiron.Doubling.AnyRing
import Mathlib.Algebra.Star.Basic
import Mathlib.Data.Real.Star
import Mathlib.LinearAlgebra.Dimension.Constructions

/-!
# The Cayley-Dickson doubling `CD(A, σ)`, reusing the trichotomy's σ as its parameter

@name Doubling
@cite graves-cayley-1845
@cite dickson-1919
@cite hurwitz-1898
@grade proved

Let `A` be a real algebra with an involution `a ↦ ā`. The Cayley-Dickson double `CD(A, σ)` is the
real vector space `A ⊕ A` with

```
   (a, b)·(c, d) = (a·c + σ·(d̄·b),  d·a + b·c̄)
   conj (a, b)   = (ā, −b)
   N(a, b)       = N(a) − σ·N(b)
```

`σ` is a real parameter of the construction. Dickson 1919 wrote the rule in this form; Graves
1843 and Cayley 1845 had the eight-dimensional case before the rule existed. Iterating from `ℝ`
gives `ℂ` and `ℝ ⊕ ℝ` at `σ = ∓1`, then `ℍ` and `M₂(ℝ)`, then `𝕆` and the split octonions, then
the sedenions.

## `CD(ℝ, σ)` is the corpus's root algebra

Over `A = ℝ` the involution is the identity, so the product collapses to
`(a·c + σ·b·d, a·d + b·c)`, which is `Trichotomy.mul σ` term for term (`real_mul`). The
two-dimensional algebra `ℝ[x]/(x² − σ)` that the rest of the corpus is written on is therefore
`CD(ℝ, σ)`, not a separate object with a coincidental resemblance. So the doubling opens no
second root. It is the operation that produces `ℂ`, `ℍ`, `𝕆` and their split forms from the
algebra `Apeiron.Trichotomy` already fixes, and the `E₈` block of `Apeiron.Lattice` reaches that
same algebra by this route.

`gen_sq` is why one symbol serves at every dimension: the adjoined generator `(0, 1)` satisfies
`(0,1)² = σ·1` in `CD(A, σ)` over every base `A`. So `σ` denotes the square of the adjoined unit
whether `A` is `ℝ` or `𝕆`, which is the meaning it already carries in `ℝ[x]/(x² − σ)`.

## The isotropy theorem, stated once

If `N` is positive definite on `A`, then the doubled form `N(a) − σ·N(b)` on `CD(A, σ)`
has a non-zero isotropic vector if and only if `σ ≥ 0` (`exists_null_iff_sigma_nonneg`). The proof
uses only positivity, anisotropy, and quadratic homogeneity of `N`: below zero the form is a sum
of two positive-semidefinite terms, and at or above zero `(√σ·1, 1)` is isotropic. The dimension
of `A` does not enter. `trichotomy_isotropy_from_doubling` recovers
`Trichotomy.exists_normForm_zero_iff_disc_nonneg` as the `A = ℝ` instance.

For the sedenions `CD(𝕆, −1)` the parameter is `−1`, so the theorem says their form is anisotropic
and the failure there is not the appearance of an isotropic vector. What fails is composition, by
`no_composition_of_anisotropy_of_break`: a quadratic form vanishing only at the origin cannot be
multiplicative for an operation with a zero divisor. That exclusion is stated for an arbitrary
binary operation and an arbitrary real-valued function on an arbitrary type, since `σ` is a
spectator in it (`theory/NOVELTY.md` test A1).

## Scope

Multiplicativity of the doubled form is not asserted in general and is false in general: it
survives while the base is associative and fails at `CD(𝕆, −1)`. Hurwitz's theorem, that a
multiplicative positive definite quadratic form over `ℝ` exists only in dimensions 1, 2, 4 and 8,
is cited rather than proved; the corpus proves the two instances it uses, `𝕆` composing and the
sedenions admitting no anisotropic composition norm.

## Root

| from | here |
|---|---|
| `Trichotomy.mul`, `Trichotomy.conj`, `Trichotomy.one` | `real_mul`, `real_star`, `real_one` |
| `Trichotomy.normForm` | `real_norm` |
| `Trichotomy.root_sq` (the relation `x² = σ`) | `gen_sq` |
| `Trichotomy.exists_normForm_zero_iff_disc_nonneg` | `trichotomy_isotropy_from_doubling` |

`real_mul` is the load-bearing one: it makes `CD(ℝ, σ)` the corpus's own algebra rather than a
second construction beside it, so every later file in `Apeiron/Doubling` and the `E₈` block of
`Apeiron/Lattice` reach `Apeiron/Trichotomy` through it. `gen_sq` carries the generator relation
to every base, which is what lets one letter name the parameter at every dimension.

**Where σ is a spectator, stated rather than left to be found.** The three exclusion
theorems `no_break_of_composition_of_anisotropy`, `no_composition_of_anisotropy_of_break` and
`isotropic_of_composition_of_break` quantify over an arbitrary binary operation and an arbitrary
real-valued function on an arbitrary type. No value of σ occurs in them and none could,
so by `theory/NOVELTY.md` test A1 they are facts about multiplicative functions and not results
about this algebra. They are stated that way on purpose, because each is applied at a different
dimension and a `Cell`-shaped version would have to be reproved three times. `finrank_double` is
likewise a fact about a product of modules.
-/

namespace Apeiron
namespace Doubling

open Trichotomy

set_option linter.unusedVariables false in
/-- The Cayley-Dickson double `CD(A, σ)`: the real vector space `A ⊕ A`, with the parameter carried
    in the type so that the two real forms of one dimension are two types.

    (A : Type*)              [any type at all]
    (σ : ℝ)                  [the parameter: what the adjoined generator squares to]
    A × A                    [the real vector space A + A]
    ℝ                        [the real numbers: every number on the number line]
    ×                        [paired with, so ℝ × ℝ means two real numbers side by side] -/
abbrev Double (A : Type*) (σ : ℝ) : Type _ := AnyRing.RingDouble A σ

variable {A : Type*} {σ : ℝ}

/-- Read a pair of base elements as one element of the double.

    (p : A × A)              [a pair of base elements]
    : Double A σ             [an element of the double at that parameter]
    Double                   [the Cayley-Dickson double A + A]
    σ                        [the parameter]
    ×                        [paired with, so ℝ × ℝ means two real numbers side by side] -/
abbrev ofPair (p : A × A) : Double A σ := AnyRing.ofPair p

/-- The first component of an element of the double.

    (x : Double A σ)         [an element of the double]
    : A                      [an element of the base]
    Double                   [the Cayley-Dickson double A + A]
    σ                        [the parameter] -/
abbrev fst (x : Double A σ) : A := AnyRing.fst x

/-- The second component of an element of the double.

    (x : Double A σ)         [an element of the double]
    : A                      [an element of the base]
    Double                   [the Cayley-Dickson double A + A]
    σ                        [the parameter] -/
abbrev snd (x : Double A σ) : A := AnyRing.snd x

/-- The first component of a pair read into the double is the pair's first entry.

    (p : A × A)                        [a pair of base elements]
    Double                             [the Cayley-Dickson double A + A]
    ofPair                             [read a pair of base elements as one element of the double]
    σ                                  [the parameter]
    ×                                  [paired with] -/
@[simp] theorem fst_ofPair (p : A × A) : fst (ofPair p : Double A σ) = p.1 := rfl

/-- The second component of a pair read into the double is the pair's second entry.

    (p : A × A)                        [a pair of base elements]
    Double                             [the Cayley-Dickson double A + A]
    ofPair                             [read a pair of base elements as one element of the double]
    σ                                  [the parameter]
    ×                                  [paired with] -/
@[simp] theorem snd_ofPair (p : A × A) : snd (ofPair p : Double A σ) = p.2 := rfl

/-- Two elements of the double with the same components are equal.

    {x y : Double A σ}                    [two elements of the double]
    (h1 : fst x = fst y)                  [their first components agree]
    (h2 : snd x = snd y)                  [their second components agree]
    x = y                                 [they are equal]
    Double                                [the Cayley-Dickson double A + A]
    σ                                     [the parameter] -/
theorem ext {x y : Double A σ} (h1 : fst x = fst y) (h2 : snd x = snd y) : x = y :=
  Prod.ext h1 h2

section Structure

variable [AddCommGroup A]

/-- A double of a finite dimensional base is finite dimensional.

    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter]
    ×                                  [paired with]
    ℝ                                  [the real numbers] -/
instance [Module ℝ A] [Module.Finite ℝ A] : Module.Finite ℝ (Double A σ) :=
  inferInstanceAs (Module.Finite ℝ (A × A))

/-- Zero has zero in the first component.

    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem fst_zero : fst (0 : Double A σ) = 0 := rfl

/-- Zero has zero in the second component.

    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem snd_zero : snd (0 : Double A σ) = 0 := rfl

/-- The unit has the base unit in the first component.

    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem fst_one [One A] : fst (1 : Double A σ) = 1 := rfl

/-- The unit has zero in the second component.

    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem snd_one [One A] : snd (1 : Double A σ) = 0 := rfl

/-- The first component of a product, as the construction defines it.

    (x y : Double A σ)                 [two elements of the double]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter]
    •                                  [stretch by, a number times a point]
    ℝ                                  [the real numbers] -/
@[simp] theorem fst_mul [Module ℝ A] [Mul A] [Star A] (x y : Double A σ) :
    fst (x * y) = fst x * fst y + σ • (star (snd y) * snd x) := rfl

/-- The second component of a product, in which the parameter does not appear.

    (x y : Double A σ)                 [two elements of the double]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter]
    ℝ                                  [the real numbers] -/
@[simp] theorem snd_mul [Module ℝ A] [Mul A] [Star A] (x y : Double A σ) :
    snd (x * y) = snd y * fst x + snd x * star (fst y) := rfl

/-- Addition acts on the first component.

    (x y : Double A σ)                 [two elements of the double]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem fst_add (x y : Double A σ) : fst (x + y) = fst x + fst y := rfl

/-- Addition acts on the second component.

    (x y : Double A σ)                 [two elements of the double]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem snd_add (x y : Double A σ) : snd (x + y) = snd x + snd y := rfl

/-- Negation acts on the first component.

    (x : Double A σ)                   [an element of the double]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem fst_neg (x : Double A σ) : fst (-x) = -fst x := rfl

/-- Negation acts on the second component.

    (x : Double A σ)                   [an element of the double]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem snd_neg (x : Double A σ) : snd (-x) = -snd x := rfl

/-- The lifted involution conjugates the first component.

    (x : Double A σ)                   [an element of the double]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem fst_star [Star A] (x : Double A σ) : fst (star x) = star (fst x) := rfl

/-- The lifted involution negates the second component.

    (x : Double A σ)                   [an element of the double]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter] -/
@[simp] theorem snd_star [Star A] (x : Double A σ) : snd (star x) = -snd x := rfl

/-- **The lifted involution is involutive.**

    [InvolutiveStar A]                    [the base involution is involutive]
    (x : Double A σ)                      [an element of the double]
    star (star x) = x                     [the lift is involutive too]
    Double                                [the Cayley-Dickson double A + A]
    σ                                     [the parameter]

    So: the involution on `CD(A, σ)` is the lift of the base involution and is involutive
    whenever the base one is. `Trichotomy.conj_conj` is the `A = ℝ` case. -/
theorem star_star_eq [InvolutiveStar A] (x : Double A σ) : star (star x) = x :=
  ext (by simp) (by simp)

/-- **The adjoined generator squares to the parameter.**

    [Module ℝ A] [Mul A] [Star A] [One A]                 [what the product needs]
    (hz : ∀ a : A, a * 0 = 0)                             [zero absorbs on the right]
    (hs : star (0 : A) = 0)                               [zero conjugates to zero]
    (hu : star (1 : A) * 1 = 1)                           [the unit conjugates to a unit]
    (ofPair (0, 1) : Double A σ) * ofPair (0, 1) = σ • 1  [the adjoined generator squares to it]
    ∀                                                     [for every single one]
    Double                                                [the Cayley-Dickson double A + A]
    σ                                                     [the parameter]
    •                                                     [stretch by, a number times a point]
    ×                                    [paired with, so ℝ × ℝ means two real numbers side by side]

    So: the generator relation `x² = σ` of `ℝ[x]/(x² − σ)` holds in `CD(A, σ)` over every base,
    with `x = (0, 1)`. `Trichotomy.root_sq` is the `A = ℝ` case. This is what licenses one symbol
    for the parameter at every height of the tower. -/
theorem gen_sq [Module ℝ A] [Mul A] [Star A] [One A] (hz : ∀ a : A, a * 0 = 0)
    (hs : star (0 : A) = 0) (hu : star (1 : A) * 1 = 1) :
    (ofPair ((0 : A), (1 : A)) : Double A σ) * ofPair ((0 : A), (1 : A)) = σ • 1 := by
  refine ext ?_ ?_
  · show (0 : A) * 0 + σ • (star (1 : A) * 1) = fst (σ • (1 : Double A σ))
    rw [hz, hu, zero_add]
    rfl
  · show (1 : A) * 0 + (1 : A) * star (0 : A) = snd (σ • (1 : Double A σ))
    rw [hz, hs, hz, add_zero]
    show (0 : A) = σ • (0 : A)
    rw [smul_zero]

end Structure

-- ═══════════════════════════════════════════════════════════════════
-- RUNG ONE IS THE TRICHOTOMY
-- ═══════════════════════════════════════════════════════════════════

/-- **The doubling of the reals is the trichotomy's multiplication.**

    (σ : ℝ)                                                        [the parameter]
    (z w : Cell)                                                   [two points]
    (ofPair z : Double ℝ σ) * (ofPair w : Double ℝ σ) = ofPair (mul σ z w)   [they agree]
    Double                                                         [the Cayley-Dickson double A + A]
    Cell                                            [a point: two real numbers written side by side]
    mul                                                    [multiply two points together, the σ way]
    ℝ                                                              [the real numbers]

    So: `CD(ℝ, σ) = ℝ[x]/(x² − σ)` as algebras, on the nose. The involution on `ℝ` is the
    identity, so the two conjugated factors in the doubling rule drop out and the product is the
    corpus's `Trichotomy.mul`. -/
theorem real_mul (σ : ℝ) (z w : Cell) :
    (ofPair z : Double ℝ σ) * (ofPair w : Double ℝ σ) = ofPair (mul σ z w) := by
  refine ext ?_ ?_
  · show z.1 * w.1 + σ • (star w.2 * z.2) = z.1 * w.1 + σ * z.2 * w.2
    rw [star_trivial, smul_eq_mul]
    ring
  · show w.2 * z.1 + z.2 * star w.1 = z.1 * w.2 + z.2 * w.1
    rw [star_trivial]
    ring

/-- **The real double is the generic doubling at `R = ℝ`.**

    [AddCommGroup A] [Module ℝ A] [Mul A] [Star A]   [a real base with a product and an involution]
    (x y : Double A σ)              [two elements of the real double]
    x * y = the generic product     [the two products are the same element]
    Double                          [the Cayley-Dickson double A + A]
    RingDouble                      [the Cayley-Dickson double A + A over a ring]
    σ                               [the parameter]
    ℝ                               [the real numbers]

    So: `Doubling.CayleyDickson.Double` is `Doubling.AnyRing.RingDouble` at `R = ℝ`, with the
    same product and the same involution. `AnyRing.ringDouble_mul_eq_ringMul` then reads the
    product at the base `ℝ` as `Trichotomy.AnyRing.ringMul`, which is `Trichotomy.mul`, so the
    trichotomy is the first rung of the real tower by direct specialization rather than by a
    second formula. The two sides are definitionally equal, which is why the proof is `rfl`. -/
theorem mul_eq_anyRing [AddCommGroup A] [Module ℝ A] [Mul A] [Star A] (x y : Double A σ) :
    x * y = (show AnyRing.RingDouble A σ from x) *
      (show AnyRing.RingDouble A σ from y) := rfl

/-- **The doubling's conjugation on the reals is the trichotomy's.**

    (σ : ℝ) (z : Cell)                                        [the parameter and a point]
    star (ofPair z : Double ℝ σ) = ofPair (conj z)            [the two conjugations agree]
    Double                                                    [the Cayley-Dickson double A + A]
    Cell                                                      [a point: two reals side by side]
    conj                                               [the same point with its second slot negated]
    ℝ                                                         [the real numbers] -/
theorem real_star (σ : ℝ) (z : Cell) :
    star (ofPair z : Double ℝ σ) = ofPair (conj z) := by
  refine ext ?_ ?_
  · show star z.1 = z.1
    exact star_trivial z.1
  · rfl

/-- **The doubling's unit on the reals is the trichotomy's.**

    (σ : ℝ)                                       [the parameter]
    (1 : Double ℝ σ) = ofPair one                 [the two units agree]
    Double                                        [the Cayley-Dickson double A + A]
    one                            [the point (1, 0), which changes nothing when you multiply by it]
    ℝ                                             [the real numbers] -/
theorem real_one (σ : ℝ) : (1 : Double ℝ σ) = ofPair one := rfl

-- ═══════════════════════════════════════════════════════════════════
-- THE RUNG NORM, AND THE ONE THEOREM THAT DOES NOT NAME A RUNG
-- ═══════════════════════════════════════════════════════════════════

/-- The doubled quadratic form `N(a) - σ·N(b)`, built from a form `N` on the base.

    (N : A → ℝ)                        [a quadratic form on the base]
    (x : Double A σ)                   [an element of the double]
    : ℝ                                [a real number]
    Double                             [the Cayley-Dickson double A + A]
    σ                                  [the parameter]
    →                                  [takes the left and gives the right]
    ℝ                                  [the real numbers] -/
def doubleNorm (N : A → ℝ) (x : Double A σ) : ℝ := N (fst x) - σ * N (snd x)

/-- The real doubled form is the generic doubled form at `R = ℝ`.

    (N : A → ℝ) (x : Double A σ)                    [a base form and an element of the double]
    doubleNorm N x = AnyRing.ringDoubleNorm σ N x   [the two agree]
    doubleNorm                                      [the doubled quadratic form]
    Double                                          [the Cayley-Dickson double A + A]
    RingDouble                                      [the Cayley-Dickson double A + A over a ring]
    σ                                               [the parameter]
    →                                               [takes the left and gives the right]
    ℝ                                               [the real numbers]

    So: the real form is the generic one at the reals. The bodies differ in notation (`*`
    against `•`) but the scalar action on `ℝ` is multiplication, so the two are the same
    function. It stays a theorem rather than an `abbrev` because downstream proofs unfold
    `doubleNorm` and close by `ring`, and those do not see through the generic definition. -/
theorem doubleNorm_eq_ringDoubleNorm (N : A → ℝ) (x : Double A σ) :
    doubleNorm N x = AnyRing.ringDoubleNorm σ N x := rfl

/-- **The doubled size on the reals is the trichotomy's norm form.**

    (σ : ℝ) (z : Cell)                                                [the parameter and a point]
    doubleNorm (fun a : ℝ => a ^ 2) (ofPair z : Double ℝ σ)           [the doubled form]
    = normForm σ z                                                    [the trichotomy's size]
    doubleNorm                                                        [the doubled quadratic form]
    Double                                            [the Cayley-Dickson double A + A]
    Cell                                                           [a point: two reals side by side]
    normForm                                            [the norm form: a point times its conjugate]
    ℝ                                                                 [the real numbers] -/
theorem real_norm (σ : ℝ) (z : Cell) :
    doubleNorm (fun a : ℝ => a ^ 2) (ofPair z : Double ℝ σ) = normForm σ z := rfl

section Null

variable [AddCommGroup A] [Module ℝ A]

omit [Module ℝ A] in
/-- **Below zero the doubled form is anisotropic.**

    (N : A → ℝ)                             [a quadratic form on the base]
    (hσ : σ < 0)                            [the parameter below zero]
    (hnn : ∀ a, 0 ≤ N a)                    [the base form is positive semidefinite]
    (hz : ∀ a, N a = 0 ↔ a = 0)             [the base form is anisotropic]
    (x : Double A σ)                        [an element of the double]
    doubleNorm N x = 0 ↔ x = 0              [the doubled form vanishes only at zero]
    doubleNorm                              [the doubled quadratic form]
    Double                                  [the Cayley-Dickson double A + A]
    σ                                       [the parameter]
    ∀                                       [for every single one]
    ↔                                       [exactly when, true in both directions]
    ≤                                       [is smaller than or equal to]
    <                                       [is smaller than]
    →                                       [takes the left and gives the right]
    ℝ                                       [the real numbers]

    So: for `σ < 0` the doubled form reads `N(a) + |σ|·N(b)`, which is positive definite
    whenever `N` is. `CD(𝕆, −1)` is this case, so its zero divisors lie off the null cone rather
    than on it. -/
theorem doubleNorm_eq_zero_iff_of_neg (N : A → ℝ) (hσ : σ < 0) (hnn : ∀ a, 0 ≤ N a)
    (hz : ∀ a, N a = 0 ↔ a = 0) (x : Double A σ) : doubleNorm N x = 0 ↔ x = 0 := by
  constructor
  · intro h
    have h1 : 0 ≤ N (fst x) := hnn _
    have h2 : 0 ≤ N (snd x) := hnn _
    have hkey : N (fst x) + (-σ) * N (snd x) = 0 := by
      unfold doubleNorm at h
      linarith
    have hp : 0 ≤ (-σ) * N (snd x) := mul_nonneg (by linarith) h2
    have hfst : N (fst x) = 0 := by linarith
    have hsnd : N (snd x) = 0 := by
      have hprod : (-σ) * N (snd x) = 0 := by linarith
      rcases mul_eq_zero.mp hprod with h3 | h3
      · exact absurd h3 (by linarith)
      · exact h3
    exact ext (by rw [(hz _).mp hfst]; rfl) (by rw [(hz _).mp hsnd]; rfl)
  · intro h
    subst h
    unfold doubleNorm
    rw [fst_zero, snd_zero, (hz 0).mpr rfl]
    ring

/-- **At or above zero the doubled form has a non-zero isotropic vector.**

    (N : A → ℝ) [One A]                                   [a size on the base, and a unit]
    (hσ : 0 ≤ σ)                                          [the parameter at zero or above]
    (hone : N 1 = 1)                                      [the unit has form value one]
    (hsm : ∀ (r : ℝ) (a : A), N (r • a) = r ^ 2 * N a)    [the form is homogeneous of degree two]
    ∃ x : Double A σ, x ≠ 0 ∧ doubleNorm N x = 0          [the form has a non-zero isotropic vector]
    doubleNorm                                            [the doubled quadratic form]
    Double                                                [the Cayley-Dickson double A + A]
    σ                                                     [the parameter]
    Real.sqrt                                             [the number that squares to give this]
    ∀                                                     [for every single one]
    ∃                                                     [there is at least one]
    ∧                                                     [both are true]
    ≠                                                     [is not equal to]
    ≤                                                     [is smaller than or equal to]
    •                                                     [stretch by, a number times a point]
    →                                                     [takes the left and gives the right]
    ℝ                                                     [the real numbers]

    So: the isotropic vector is `(√σ·1, 1)`, the image under the doubling of the vector
    `Trichotomy.exists_normForm_zero_iff_disc_nonneg` uses over `ℝ`. -/
theorem exists_null_of_nonneg (N : A → ℝ) [One A] (hσ : 0 ≤ σ) (hone : N 1 = 1)
    (hsm : ∀ (r : ℝ) (a : A), N (r • a) = r ^ 2 * N a) :
    ∃ x : Double A σ, x ≠ 0 ∧ doubleNorm N x = 0 := by
  have hN0 : N (0 : A) = 0 := by
    have := hsm 0 (1 : A)
    rwa [zero_smul, zero_pow (by norm_num), zero_mul] at this
  have hone_ne : (1 : A) ≠ 0 := by
    intro h
    rw [h, hN0] at hone
    norm_num at hone
  refine ⟨ofPair (Real.sqrt σ • (1 : A), (1 : A)), ?_, ?_⟩
  · intro h
    exact hone_ne (by simpa using congrArg snd h)
  · show N (Real.sqrt σ • (1 : A)) - σ * N (1 : A) = 0
    rw [hsm, hone, Real.sq_sqrt hσ]
    ring

/-- **The doubled form is isotropic exactly when the parameter is at zero or above.**

    (N : A → ℝ) [One A]                                   [a size on the base, and a unit]
    (hnn : ∀ a, 0 ≤ N a)                                  [the base form is positive semidefinite]
    (hz : ∀ a, N a = 0 ↔ a = 0)                           [it vanishes only at zero]
    (hone : N 1 = 1)                                      [the unit has form value one]
    (hsm : ∀ (r : ℝ) (a : A), N (r • a) = r ^ 2 * N a)    [the form is homogeneous of degree two]
    (∃ x : Double A σ, x ≠ 0 ∧ doubleNorm N x = 0)        [the form has a non-zero isotropic vector]
    ↔ 0 ≤ σ                                               [the parameter is at zero or above]
    doubleNorm                                            [the doubled quadratic form]
    Double                                                [the Cayley-Dickson double A + A]
    σ                                                     [the parameter]
    ∀                                                     [for every single one]
    ∃                                                     [there is at least one]
    ∧                                                     [both are true]
    ≠                                                     [is not equal to]
    ≤                                                     [is smaller than or equal to]
    •                                                     [stretch by, a number times a point]
    →                                                     [takes the left and gives the right]
    ℝ                                                     [the real numbers]

    So: given a positive definite base form, the sign of `σ` alone decides whether the doubled
    form is isotropic. The dimension of `A` does not enter the argument, so the statement holds at
    every height of the tower with one proof. -/
theorem exists_null_iff_sigma_nonneg (N : A → ℝ) [One A] (hnn : ∀ a, 0 ≤ N a)
    (hz : ∀ a, N a = 0 ↔ a = 0) (hone : N 1 = 1)
    (hsm : ∀ (r : ℝ) (a : A), N (r • a) = r ^ 2 * N a) :
    (∃ x : Double A σ, x ≠ 0 ∧ doubleNorm N x = 0) ↔ 0 ≤ σ := by
  constructor
  · rintro ⟨x, hx, hn⟩
    by_contra hc
    push Not at hc
    exact hx ((doubleNorm_eq_zero_iff_of_neg N hc hnn hz x).mp hn)
  · intro hσ
    exact exists_null_of_nonneg N hσ hone hsm

end Null

/-- **The corpus's own isotropy theorem is the `A = ℝ` instance of the last one.**

    (σ : ℝ)                                             [the parameter]
    (∃ z : Cell, z ≠ 0 ∧ normForm σ z = 0)              [the form has a non-zero isotropic vector]
    ↔ 0 ≤ σ                                             [the parameter is at zero or above]
    Cell                                                [a point: two reals side by side]
    normForm                                            [the norm form of a point]
    ∃                                                   [there is at least one]
    ∧                                                   [both are true]
    ≠                                                   [is not equal to]
    ≤                                                   [is smaller than or equal to]
    ℝ                                                   [the real numbers]

    So: `Trichotomy.exists_normForm_zero_iff_disc_nonneg` obtained as the `A = ℝ` instance of
    `exists_null_iff_sigma_nonneg`, rather than by a computation in two real variables. The two
    statements are the same proposition because `CD(ℝ, σ)` is that algebra. -/
theorem trichotomy_isotropy_from_doubling (σ : ℝ) :
    (∃ z : Cell, z ≠ 0 ∧ normForm σ z = 0) ↔ 0 ≤ σ :=
  exists_null_iff_sigma_nonneg (A := ℝ) (σ := σ) (fun a : ℝ => a ^ 2) (fun a => sq_nonneg a)
    (fun _ => sq_eq_zero_iff) (one_pow 2) (fun r a => by simp only [smul_eq_mul]; ring)

-- ═══════════════════════════════════════════════════════════════════
-- COMPOSITION AND THE BREAK: ONE EXCLUSION, THREE READINGS
-- ═══════════════════════════════════════════════════════════════════

section Exclusion

variable {B : Type*} (m : B → B → B) (N : B → ℝ) (e : B)

/-- **A multiplicative size that vanishes only at one point forbids a break there.**

    (m : B → B → B) (N : B → ℝ) (e : B)          [an operation, a form, a distinguished element]
    (he : N e = 0)                                         [the form vanishes at that element]
    (hc : ∀ u v, N (m u v) = N u * N v)                    [the form is multiplicative]
    (ha : ∀ u, N u = 0 ↔ u = e)                            [the form is anisotropic]
    {x y : B} (hx : x ≠ e) (hy : y ≠ e)                    [two elements other than it]
    m x y ≠ e                                              [their product is not it either]
    ∀                                                      [for every single one]
    ↔                                                      [exactly when, true in both directions]
    ≠                                                      [is not equal to]
    →                                                      [takes the left and gives the right]
    ℝ                                                      [the real numbers]

    So: the composition-algebra exclusion. Applied at `𝕆`, where `normO_mul` and
    `normO_eq_zero_iff` supply the two hypotheses, it gives that the octonions have no zero
    divisors. No parameter appears in the statement, which quantifies over an arbitrary binary
    operation on an arbitrary type, so `theory/NOVELTY.md` test A1 keeps it out of the `σ`
    results. -/
theorem no_break_of_composition_of_anisotropy (he : N e = 0)
    (hc : ∀ u v, N (m u v) = N u * N v) (ha : ∀ u, N u = 0 ↔ u = e) {x y : B}
    (hx : x ≠ e) (hy : y ≠ e) : m x y ≠ e := by
  intro h
  have hzero : N x * N y = 0 := by rw [← hc, h, he]
  rcases mul_eq_zero.mp hzero with h1 | h1
  · exact hx ((ha x).mp h1)
  · exact hy ((ha y).mp h1)

/-- **A size that vanishes only at one point cannot be multiplicative once there is a break.**

    (m : B → B → B) (N : B → ℝ) (e : B)          [an operation, a form, a distinguished element]
    (he : N e = 0)                                         [the form vanishes at that element]
    (ha : ∀ u, N u = 0 ↔ u = e)                            [the form is anisotropic]
    {x y : B} (hx : x ≠ e) (hy : y ≠ e) (hxy : m x y = e)  [two non-zero elements with zero product]
    ¬ ∀ u v, N (m u v) = N u * N v                         [the form is not multiplicative]
    ∀                                                      [for every single one]
    ¬                                                      [this is not true]
    ↔                                                      [exactly when, true in both directions]
    ≠                                                      [is not equal to]
    →                                                      [takes the left and gives the right]
    ℝ                                                      [the real numbers]

    So: the same exclusion contraposed. Applied at `CD(𝕆, −1)`, which is anisotropic and has a
    zero divisor, it gives that no anisotropic form there is multiplicative. -/
theorem no_composition_of_anisotropy_of_break (he : N e = 0) (ha : ∀ u, N u = 0 ↔ u = e)
    {x y : B} (hx : x ≠ e) (hy : y ≠ e) (hxy : m x y = e) :
    ¬ ∀ u v, N (m u v) = N u * N v := fun hc =>
  no_break_of_composition_of_anisotropy m N e he hc ha hx hy hxy

/-- **A multiplicative size with a break has a non-zero point of size zero.**

    (m : B → B → B) (N : B → ℝ) (e : B)          [an operation, a form, a distinguished element]
    (he : N e = 0)                                         [the form vanishes at that element]
    (hc : ∀ u v, N (m u v) = N u * N v)                    [the form is multiplicative]
    {x y : B} (hx : x ≠ e) (hy : y ≠ e) (hxy : m x y = e)  [two non-zero elements with zero product]
    ∃ u, u ≠ e ∧ N u = 0                                   [the form is isotropic]
    ∀                                                      [for every single one]
    ∃                                                      [there is at least one]
    ∧                                                      [both are true]
    ≠                                                      [is not equal to]
    →                                                      [takes the left and gives the right]
    ℝ                                                      [the real numbers]

    So: the exclusion contraposed the third way. For a composition algebra, isotropy follows
    from the existence of a zero divisor rather than being a second hypothesis, and combined with
    `exists_null_iff_sigma_nonneg` it forces `σ ≥ 0`. -/
theorem isotropic_of_composition_of_break (he : N e = 0) (hc : ∀ u v, N (m u v) = N u * N v)
    {x y : B} (hx : x ≠ e) (hy : y ≠ e) (hxy : m x y = e) : ∃ u, u ≠ e ∧ N u = 0 := by
  have hzero : N x * N y = 0 := by rw [← hc, hxy, he]
  rcases mul_eq_zero.mp hzero with h1 | h1
  · exact ⟨x, hx, h1⟩
  · exact ⟨y, hy, h1⟩

end Exclusion

-- ═══════════════════════════════════════════════════════════════════
-- THE BREAK AT THE SPLIT SIGN, ONCE, FOR EVERY ASSOCIATIVE BASE
-- ═══════════════════════════════════════════════════════════════════

/-- **At parameter one the doubling of any associative base has a zero divisor.**

    [Ring A] [StarRing A] [Module ℝ A]                          [an associative base]
    (ofPair (1, 1) : Double A 1) * ofPair (1, -1) = 0           [their product is zero]
    Double                                                      [the Cayley-Dickson double A + A]
    ×                                                           [paired with]
    ℝ                                                           [the real numbers]

    So: at `σ = 1` the doubling of any associative `*`-algebra has zero divisors, so
    `ℝ ⊕ ℝ`, `M₂(ℝ)` and the split octonions are one instance each of this statement rather than
    three computations. The first component evaluates to `1·1 + \overline{(-1)}·1 = 1 − 1`, and the
    parameter is the only thing that makes it vanish. -/
theorem split_mul_eq_zero [Ring A] [StarRing A] [Module ℝ A] :
    (ofPair ((1 : A), (1 : A)) : Double A 1) * ofPair ((1 : A), (-1 : A)) = 0 := by
  refine ext ?_ ?_
  · show (1 : A) * 1 + (1 : ℝ) • (star (-1 : A) * 1) = fst (0 : Double A 1)
    rw [one_smul, star_neg, star_one]
    show (1 : A) * 1 + (-1 : A) * 1 = (0 : A)
    simp
  · show (-1 : A) * 1 + (1 : A) * star (1 : A) = snd (0 : Double A 1)
    rw [star_one]
    show (-1 : A) * 1 + (1 : A) * 1 = (0 : A)
    simp

/-- **The two factors are not zero.**

    [Ring A] [StarRing A] [Module ℝ A] [Nontrivial A]      [an associative base with a unit]
    (ofPair (1, 1) : Double A 1) ≠ 0                       [the first factor is not zero]
    Double                                                 [the Cayley-Dickson double A + A]
    ≠                                                      [is not equal to]
    ×                                                      [paired with]
    ℝ                                                      [the real numbers] -/
theorem split_left_ne_zero [Ring A] [StarRing A] [Module ℝ A] [Nontrivial A] :
    (ofPair ((1 : A), (1 : A)) : Double A 1) ≠ 0 := by
  intro h
  have h1 : (1 : A) = 0 := by simpa using congrArg fst h
  exact one_ne_zero h1

/-- **The other factor is not zero either.**

    [Ring A] [StarRing A] [Module ℝ A] [Nontrivial A]      [an associative base with a unit]
    (ofPair (1, -1) : Double A 1) ≠ 0                      [the second factor is not zero]
    Double                                                 [the Cayley-Dickson double A + A]
    ≠                                                      [is not equal to]
    ×                                                      [paired with]
    ℝ                                                      [the real numbers] -/
theorem split_right_ne_zero [Ring A] [StarRing A] [Module ℝ A] [Nontrivial A] :
    (ofPair ((1 : A), (-1 : A)) : Double A 1) ≠ 0 := by
  intro h
  have h1 : (1 : A) = 0 := by simpa using congrArg fst h
  exact one_ne_zero h1

-- ═══════════════════════════════════════════════════════════════════
-- THE DOUBLING DOUBLES THE DIMENSION
-- ═══════════════════════════════════════════════════════════════════

/-- **The doubling doubles the real dimension.**

    [AddCommGroup A] [Module ℝ A] [Module.Finite ℝ A]                 [a finite real base]
    Module.finrank ℝ (Double A σ) = 2 * Module.finrank ℝ A            [the count doubles]
    Module.finrank                                   [how many independent directions the space has]
    Double                                            [the Cayley-Dickson double A + A]
    σ                                                                 [the parameter]
    ℝ                                                                 [the real numbers]

    So: `dim_ℝ CD(A, σ) = 2·dim_ℝ A`, so the dimensions along the tower from `ℝ` are `2ᵏ` and
    the value 8 for `𝕆` is counted rather than posited. -/
theorem finrank_double [AddCommGroup A] [Module ℝ A] [Module.Finite ℝ A] :
    Module.finrank ℝ (Double A σ) = 2 * Module.finrank ℝ A := by
  show Module.finrank ℝ (A × A) = 2 * Module.finrank ℝ A
  rw [Module.finrank_prod]
  ring

end Doubling
end Apeiron
