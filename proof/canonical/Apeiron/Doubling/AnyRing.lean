import Apeiron.Trichotomy.AnyRing
import Mathlib.Algebra.Star.Basic

/-!
# The Cayley-Dickson doubling over any commutative ring

@name CayleyDicksonAnyRing
@cite dickson-1919
@grade proved

`Doubling.CayleyDickson` fixes the scalar parameter to a real number and the base to a real
vector space. Nothing algebraic in the product rule needs either. Dickson's rule

```
   (a, b)·(c, d) = (a·c + σ·(d̄·b),  d·a + b·c̄)
```

takes `σ` from whatever ring the base is a module over, and `σ` enters the first slot only. This
file carries the rule over an arbitrary commutative ring `R`, so the reals are one instance of
the construction rather than its definition. The `Ring` prefix marks the general carrier.

## Root

| from | here |
|---|---|
| `AnyRing.ringMul` | `ringDouble_mul_eq_ringMul` |
| `AnyRing.ringNormForm` | `ringDoubleNorm_eq_ringNormForm` |

At the base `R` itself, with the involution the identity, the rule collapses to
`(a, b)·(c, d) = (a·c + σ·b·d, a·d + b·c)`, which is `Trichotomy.AnyRing.ringMul σ` term for
term. `ringDouble_mul_eq_ringMul` is that equation, and `ringDoubleNorm_eq_ringNormForm` is the
same collapse for the norm form. So `RingCell R` is `RingDouble R σ`, the first generic
doubling, and not a second object that happens to share the formulas. At `R = ℝ` the same two
theorems read as `Doubling.real_mul` and `Doubling.real_norm`, which is the bridge the later
rungs of the tower cross.

## What is generic and what is not

`gen_sq` crosses every commutative ring, and so do the product's two component lemmas, which are
definitional. The unit laws, the split-sign zero divisor, and the ring-level laws that need the
base commutative are not lifted here. Their component proofs need the scalar action and the
involution moved through sums and products, and the instances such a proof sees for `•` and
`star` are not syntactically the ones `fst_mul`, `star_add` and `star_mul` are stated for. Those
theorems remain at `A = R` in `Trichotomy.AnyRing`; lifting them is the next bounded step.

Multiplicativity of the doubled norm is not asserted here; the corpus proves it one rung at a
time in `Doubling.Octonion` and `Doubling.SplitOctonion`.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Doubling
namespace AnyRing

variable {R : Type*}

set_option linter.unusedVariables false in
/-- The Cayley-Dickson double `CD_R(A, σ)`: two base elements side by side, the parameter carried
    in the type.

    (A : Type*)                  [the base the doubling is built on]
    (σ : R)                      [the parameter: what the adjoined direction squares to]
    : Type _                     [the type of the doubled algebra]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
def RingDouble (A : Type*) (σ : R) : Type _ := A × A

variable {A : Type*}

/-- Read a pair of base elements as one element of the double.

    (p : A × A)                  [a pair of base elements]
    : RingDouble A σ             [an element of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    ×                            [paired with, so A × A means two base elements side by side]
    σ                            [the parameter] -/
def ofPair {σ : R} (p : A × A) : RingDouble A σ := p

/-- The first component of an element of the double.

    (x : RingDouble A σ)         [an element of the double]
    : A                          [an element of the base]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
def fst {σ : R} (x : RingDouble A σ) : A := Prod.fst x

/-- The second component of an element of the double.

    (x : RingDouble A σ)         [an element of the double]
    : A                          [an element of the base]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
def snd {σ : R} (x : RingDouble A σ) : A := Prod.snd x

/-- The first component of a pair read into the double is the pair's first entry.

    (p : A × A)                  [a pair of base elements]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    ofPair                       [read a pair of base elements as one element of the double]
    σ                            [the parameter]
    ×                            [paired with] -/
@[simp] theorem fst_ofPair {σ : R} (p : A × A) :
    fst (ofPair p : RingDouble A σ) = p.1 := rfl

/-- The second component of a pair read into the double is the pair's second entry.

    (p : A × A)                  [a pair of base elements]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    ofPair                       [read a pair of base elements as one element of the double]
    σ                            [the parameter]
    ×                            [paired with] -/
@[simp] theorem snd_ofPair {σ : R} (p : A × A) :
    snd (ofPair p : RingDouble A σ) = p.2 := rfl

/-- Two elements of the double with the same components are equal.

    {x y : RingDouble A σ}       [two elements of the double]
    (h1 : fst x = fst y)         [their first components agree]
    (h2 : snd x = snd y)         [their second components agree]
    x = y                        [they are equal]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[ext] theorem ext {σ : R} {x y : RingDouble A σ} (h1 : fst x = fst y)
    (h2 : snd x = snd y) : x = y :=
  Prod.ext h1 h2

section Structure

variable [AddCommGroup A]

/-- Addition and negation are those of the underlying pair, componentwise.

    (σ : R)                      [the parameter]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R] -/
instance instAddCommGroupRingDouble (σ : R) : AddCommGroup (RingDouble A σ) :=
  inferInstanceAs (AddCommGroup (A × A))

/-- The scalar action is componentwise.

    (σ : R)                      [the parameter]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R] -/
instance instModuleRingDouble (σ : R) [Semiring R] [Module R A] : Module R (RingDouble A σ) :=
  inferInstanceAs (Module R (A × A))

/-- The unit of the double is the base unit in the first component.

    (σ : R)                      [the parameter]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R] -/
instance instOneRingDouble (σ : R) [One A] : One (RingDouble A σ) := ⟨((1 : A), (0 : A))⟩

/-- The Cayley-Dickson product. The parameter multiplies the crossed term of the first
    component and is absent from the second.

    (σ : R)                      [the parameter]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    •                            [stretch by, a ring element times a base element] -/
instance instMulRingDouble (σ : R) [SMul R A] [Mul A] [Star A] : Mul (RingDouble A σ) :=
  ⟨fun x y => ofPair (fst x * fst y + σ • (star (snd y) * snd x),
    snd y * fst x + snd x * star (fst y))⟩

/-- The lifted involution: conjugate the first component, negate the second.

    (σ : R)                      [the parameter]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R] -/
instance instStarRingDouble (σ : R) [Star A] : Star (RingDouble A σ) :=
  ⟨fun x => ofPair (star (fst x), -snd x)⟩

variable {σ : R}

/-- Zero has zero in the first component.

    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem fst_zero : fst (0 : RingDouble A σ) = 0 := rfl

/-- Zero has zero in the second component.

    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem snd_zero : snd (0 : RingDouble A σ) = 0 := rfl

/-- The unit has the base unit in the first component.

    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem fst_one [One A] : fst (1 : RingDouble A σ) = 1 := rfl

/-- The unit has zero in the second component.

    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem snd_one [One A] : snd (1 : RingDouble A σ) = 0 := rfl

/-- The first component of a product, as the construction defines it.

    (x y : RingDouble A σ)       [two elements of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter]
    •                            [stretch by, a ring element times a base element] -/
@[simp] theorem fst_mul [SMul R A] [Mul A] [Star A] (x y : RingDouble A σ) :
    fst (x * y) = fst x * fst y + σ • (star (snd y) * snd x) := rfl

/-- The second component of a product, in which the parameter does not appear.

    (x y : RingDouble A σ)       [two elements of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem snd_mul [SMul R A] [Mul A] [Star A] (x y : RingDouble A σ) :
    snd (x * y) = snd y * fst x + snd x * star (fst y) := rfl

/-- Addition acts on the first component.

    (x y : RingDouble A σ)       [two elements of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem fst_add (x y : RingDouble A σ) : fst (x + y) = fst x + fst y := rfl

/-- Addition acts on the second component.

    (x y : RingDouble A σ)       [two elements of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem snd_add (x y : RingDouble A σ) : snd (x + y) = snd x + snd y := rfl

/-- Negation acts on the first component.

    (x : RingDouble A σ)         [an element of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem fst_neg (x : RingDouble A σ) : fst (-x) = -fst x := rfl

/-- Negation acts on the second component.

    (x : RingDouble A σ)         [an element of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem snd_neg (x : RingDouble A σ) : snd (-x) = -snd x := rfl

/-- The lifted involution conjugates the first component.

    (x : RingDouble A σ)         [an element of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem fst_star [Star A] (x : RingDouble A σ) : fst (star x) = star (fst x) := rfl

/-- The lifted involution negates the second component.

    (x : RingDouble A σ)         [an element of the double]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter] -/
@[simp] theorem snd_star [Star A] (x : RingDouble A σ) : snd (star x) = -snd x := rfl

/-- The lifted involution is involutive whenever the base one is.

    [InvolutiveStar A]           [the base involution is its own inverse]
    (x : RingDouble A σ)         [an element of the double]
    star (star x) = x            [conjugating twice gives the element back]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    σ                            [the parameter]

    So: the involution on `CD_R(A, σ)` is the lift of the base involution. At `A = R` it is the
    identity, which is `Trichotomy.AnyRing.ringConj` when the base involution is trivial. -/
theorem star_star_eq [InvolutiveStar A] (x : RingDouble A σ) : star (star x) = x :=
  ext (by simp) (by simp)

/-- **The adjoined generator squares to the parameter.**

    [Module R A] [Mul A] [Star A] [One A]     [what the product needs]
    (hz : ∀ a : A, a * 0 = 0)                 [zero absorbs on the right]
    (hs : star (0 : A) = 0)                   [zero conjugates to zero]
    (hu : star (1 : A) * 1 = 1)               [the unit conjugates to a unit]
    (ofPair (0, 1) : RingDouble A σ) * ofPair (0, 1) = σ • 1   [the generator squares to σ]
    ∀                                         [for every single one]
    RingDouble                                [the Cayley-Dickson double A + A over the ring R]
    ofPair                                    [read a pair of base elements as one element]
    σ                                         [the parameter]
    •                                         [stretch by, a ring element times a base element]
    →                                         [takes the left and gives the right]

    So: the relation `x² = σ` of `R[x]/(x² − σ)` holds in `CD_R(A, σ)` over every base, with
    `x = (0, 1)`. At `A = R` it is the relation `Trichotomy.AnyRing.ringMul` is written for, and
    `Doubling.gen_sq` is the `R = ℝ` case. -/
theorem gen_sq [Semiring R] [Module R A] [Mul A] [Star A] [One A] (hz : ∀ a : A, a * 0 = 0)
    (hs : star (0 : A) = 0) (hu : star (1 : A) * 1 = 1) :
    (ofPair ((0 : A), (1 : A)) : RingDouble A σ) * ofPair ((0 : A), (1 : A)) = σ • 1 := by
  ext
  · show (0 : A) * 0 + σ • (star (1 : A) * 1) = fst (σ • (1 : RingDouble A σ))
    rw [hz, hu, zero_add]
    rfl
  · show (1 : A) * 0 + (1 : A) * star (0 : A) = snd (σ • (1 : RingDouble A σ))
    rw [hz, hs, hz, add_zero]
    show (0 : A) = σ • (0 : A)
    rw [smul_zero]

end Structure

section Norm

/-- The doubled quadratic form `N(a) - σ·N(b)`, built from a form on the base.

    (σ : R)                      [the parameter]
    (N : A → R)                  [a quadratic form on the base]
    (x : RingDouble A σ)         [an element of the double]
    : R                          [an element of the ring]
    RingDouble                   [the Cayley-Dickson double A + A over the ring R]
    →                            [takes the left and gives the right]
    •                            [stretch by, a ring element times a base element] -/
def ringDoubleNorm [CommRing R] (σ : R) (N : A → R) (x : RingDouble A σ) : R :=
  N (fst x) - σ • N (snd x)

/-- **The doubled size on the ring itself is the trichotomy's norm form.**

    (σ : R) (x : RingDouble R σ)             [the parameter and an element of the first rung]
    ringDoubleNorm σ (fun a : R => a ^ 2) x  [the doubled form of the squaring form]
    = Trichotomy.AnyRing.ringNormForm σ x    [the trichotomy's norm form]
    ringDoubleNorm                           [the doubled quadratic form]
    RingDouble                               [the Cayley-Dickson double A + A over the ring R]
    σ                                        [the parameter]
    Trichotomy.AnyRing.ringNormForm          [the norm form: an element times its conjugate]

    So: the first rung's norm form is `a² − σb²`, which is `Doubling.real_norm` at `R = ℝ`. -/
theorem ringDoubleNorm_eq_ringNormForm [CommRing R] (σ : R) (x : RingDouble R σ) :
    ringDoubleNorm σ (fun a : R => a ^ 2) x = Trichotomy.AnyRing.ringNormForm σ x := by
  simp only [ringDoubleNorm, fst, snd, Trichotomy.AnyRing.ringNormForm, smul_eq_mul]

/-- **The trichotomy's multiplication is the first generic doubling.**

    (σ : R) (z w : RingDouble R σ)           [the parameter and two elements of the first rung]
    z * w = Trichotomy.AnyRing.ringMul σ z w [the generic product is the trichotomy's product]
    RingDouble                               [the Cayley-Dickson double A + A over the ring R]
    σ                                        [the parameter]
    Trichotomy.AnyRing.ringMul               [multiply two points together, the σ way]

    So: `RingCell R` is `RingDouble R σ`, the first generic doubling by direct specialization.
    `Trichotomy.AnyRing.ringMul` is what this product becomes when the base is the ring itself
    and the involution is the identity. At `R = ℝ` it is `Doubling.real_mul` through
    `Trichotomy.mul`, which is `ringMul` at the reals. -/
theorem ringDouble_mul_eq_ringMul [CommRing R] (σ : R) [Star R] [TrivialStar R]
    (z w : RingDouble R σ) : z * w = Trichotomy.AnyRing.ringMul σ z w := by
  refine Prod.ext ?_ ?_
  · change z.1 * w.1 + σ • (star w.2 * z.2) = z.1 * w.1 + σ * z.2 * w.2
    rw [star_trivial, smul_eq_mul]
    ring
  · change w.2 * z.1 + z.2 * star w.1 = z.1 * w.2 + z.2 * w.1
    rw [star_trivial]
    ring

end Norm

end AnyRing
end Doubling
end Apeiron
