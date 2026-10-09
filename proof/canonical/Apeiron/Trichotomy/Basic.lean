import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing
import Mathlib.Algebra.GroupWithZero.Units.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.RingTheory.AdjoinRoot
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.LinearAlgebra.Pi
import Mathlib.Tactic.LinearCombination

/-!
# The σ-Trichotomy

@name The σ-Trichotomy
@cite euclid-elements-ii-5
@cite pythagoras-sum-of-squares
@cite gauss-1831
@cite cockle-1849
@cite study-1903
@cite weierstrass-1884
@cite wedderburn-1908
@cite mathlib-findim
@cite mathlib-sqrt
@grade proved

Start with an identity from school algebra:

```
(1 + x)(1 − x) = 1 − x²
```

Suppose `x` squares to `1`. The right-hand side is then `0`, and the left-hand side is a
product of two factors, neither of which is zero.

Whether an algebra admits such a pair depends on one real number: `σ`, the value the
generator squares to. The carrier is a pair of reals `a + b·x`, and the size of a point is

```
N(a + b·x) = a² − σb²
```

The set of points of size zero changes with `σ`:

```
      σ < 0                    σ = 0                    σ > 0
   ───────────              ───────────              ───────────
   N = a² + |σ|b²           N = a²                   N = a² − σb²

         b                        b                       b
         │                        │                      ╲│╱
    ─────┼─────  a           ─────┼─────  a          ─────┼─────  a
         │                        │                      ╱│╲
         │                        │                       │

   size zero only           size zero on a           size zero on two
   at the origin            whole line               crossing lines

   no two non-zero          one point squares        a pair of points
   points multiply          to zero                  multiplies to zero
   to zero
```

Those three pictures are `ℂ`, the dual numbers, and split-`ℂ`. Classically: elliptic,
parabolic, hyperbolic. They were found separately over three centuries, `ℂ` through Bombelli
and Gauss, the split numbers by Cockle in 1849 (who recorded that division fails on them),
the dual numbers by Clifford and Study.

The middle picture is not a third species on equal footing. It is the boundary the other two
pass through, and `hasNullPair_iff_disc_pos` is where that shows up as a theorem.

## The three as one object

The three pictures are one construction with `σ` varying. Take polynomials over the
reals and impose a single relation, `x·x = σ`. What is left is `a + b·x`, the pair, and
multiplying two of them gives `(ac + σbd) + (ad + bc)·x`. That is `mul σ`, and
`mul_eq_quotient_mul` proves the formula is what the relation forces rather than something
chosen.

`ℝ[X] / (X² − σ)` is therefore the elliptic, parabolic and hyperbolic cases as a single
object. The relation is imposed the same way at every `σ`, so this construction is not tied
to any one of the three. Results later in the corpus that need `σ > 0`, such as the
null-basis coordinates and the idempotent count, sit downstream of it.

## Where the size function comes from

`N(a + b·x) = a² − σb²` interpolates two identities older than the algebra. At `σ = −1`
it is `a² + b²`, the Pythagorean form. At `σ = +1` it is `a² − b²`, which Euclid factors as
`(a+b)(a−b)` in *Elements* II.5. That factorization is the pair of non-zero points this file
multiplies to zero, so the opening identity above is Euclid II.5 at `σ = 1`.

That the list of three is *complete* follows from Weierstrass and Wedderburn structure
theory. `classification` proves it here directly, by completing the square, citing nothing.

## Root

This file imports no mathematics from the library, only Mathlib. It is the bottom of
`Apeiron` and every other module reaches it. The two library imports are the linters that
check this file's own glosses and citations; neither carries mathematical content.

## Reading the symbols

Every statement below is glossed line by line in brackets. `Apeiron.Linter.Registry` holds
one canonical meaning per symbol, and `Apeiron.Linter.Notation` checks each statement against
it during elaboration, reading the parsed `declSig` rather than the file text.
-/

noncomputable section

open Polynomial AdjoinRoot

namespace Apeiron
namespace Trichotomy

/-- A point is a pair of real numbers.

    Cell    [the name for one point]
    ℝ × ℝ   [a real number, then another real number]

    So: a point is two reals side by side, written `a + b·x`. `x` labels the second slot;
    the parameter `σ` fixes what `x·x` equals. -/
abbrev Cell : Type := AnyRing.RingCell ℝ

/-- How to multiply two points.

    (σ : ℝ)                        [the parameter, any real number]
    (z w : Cell)                   [the two points being multiplied]
    z.1 * w.1 + σ * z.2 * w.2      [the first slot of the answer]
    z.1 * w.2 + z.2 * w.1          [the second slot of the answer]

    So: `(a+bx)(c+dx) = (ac + σbd) + (ad+bc)x`. `σ` enters at one place only, the
    `σbd` term. -/
abbrev mul (σ : ℝ) (z w : Cell) : Cell :=
  AnyRing.ringMul σ z w

/-- Multiplication does not depend on the order of its two points.

    (σ : ℝ)                        [the parameter]
    (z w : Cell)                   [two points]
    mul σ z w = mul σ w z          [multiplying either way round gives the same point] -/
theorem mul_comm (σ : ℝ) (z w : Cell) : mul σ z w = mul σ w z :=
  AnyRing.ringMul_comm σ z w

/-- Multiplication does not depend on the bracketing.

    (σ : ℝ)                                          [the parameter]
    (z w v : Cell)                                   [three points]
    mul σ (mul σ z w) v = mul σ z (mul σ w v)        [either bracketing gives the same point] -/
theorem mul_assoc (σ : ℝ) (z w v : Cell) :
    mul σ (mul σ z w) v = mul σ z (mul σ w v) :=
  AnyRing.ringMul_assoc σ z w v

/-- The size of a point.

    (σ : ℝ)                 [the parameter]
    (z : Cell)              [the point being measured]
    z.1 ^ 2 - σ * z.2 ^ 2   [first slot squared, minus σ times second slot squared]
    : ℝ                     [the answer is a plain real number]

    So: `N(a + bx) = a² − σb²`. At `σ = −1` this is `a² + b²`, which is never negative.
    At `σ = +1` it is `a² − b²`, which can be negative, and can be zero at a point that is
    not itself zero. -/
abbrev normForm (σ : ℝ) (z : Cell) : ℝ := AnyRing.ringNormForm σ z

/-- How much two points pull on each other.

    (σ : ℝ)                      [the parameter]
    (z w : Cell)                 [the two points]
    z.1 * w.1 - σ * z.2 * w.2    [their pull]

    So: `normForm` opened out to take two different points. Given the same point twice it
    returns that point's size. -/
abbrev form (σ : ℝ) (z w : Cell) : ℝ := AnyRing.ringForm σ z w

/-- The point that multiplication leaves alone.

    one        [the name for this point]
    (1, 0)     [one in the first slot, zero in the second]
    : Cell     [a point of the algebra]

    So: `1 + 0·x`. Note that Lean's own `(1 : Cell)` is `(1, 1)`, because a pair carries
    componentwise operations by default. The unit of `mul σ` is this, not that. -/
abbrev one : Cell := AnyRing.ringOne

/-- Multiplying by `one` changes nothing.

    (σ : ℝ)             [the parameter]
    (z : Cell)          [any point]
    mul σ one z         [multiply the unit by z]
    = z                 [gives z back unchanged] -/
theorem one_mul (σ : ℝ) (z : Cell) : mul σ one z = z :=
  AnyRing.ringOne_mul σ z

/-- Multiplication spreads across a sum.

    (σ : ℝ)                                          [the parameter]
    (z w v : Cell)                                   [three points]
    mul σ z (w + v) = mul σ z w + mul σ z v          [multiply each half and add]

    So: the algebra's multiplication is bilinear, which the coordinate formula makes an
    identity in the four slots. With `mul_comm` this gives the other side too. -/
theorem mul_distrib_add (σ : ℝ) (z w v : Cell) :
    mul σ z (w + v) = mul σ z w + mul σ z v :=
  AnyRing.ringMul_distrib_add σ z w v

/-- Multiplication spreads across a difference.

    (σ : ℝ)                                          [the parameter]
    (z w v : Cell)                                   [three points]
    mul σ z (w - v) = mul σ z w - mul σ z v          [multiply each half and subtract]

    So: `mul_distrib_add` with a sign. This is what lets a loop's distance to its resting
    point be rewritten as one multiplication. -/
theorem mul_distrib_sub (σ : ℝ) (z w v : Cell) :
    mul σ z (w - v) = mul σ z w - mul σ z v :=
  AnyRing.ringMul_distrib_sub σ z w v

/-- Multiplying by the origin gives the origin.

    (σ : ℝ)                        [the parameter]
    (z : Cell)                     [any point]
    mul σ z 0 = 0                  [the origin absorbs] -/
theorem mul_origin (σ : ℝ) (z : Cell) : mul σ z 0 = 0 :=
  AnyRing.ringMul_zero σ z

-- ═══════════════════════════════════════════════════════════════════
-- WHERE THE FORMULA COMES FROM: THE RING QUOTIENT
-- ═══════════════════════════════════════════════════════════════════

/-- The defining polynomial: the one relation the quotient imposes.

    (σ : ℝ)              [the parameter]
    X ^ 2 - C σ          [the polynomial whose vanishing says the generator squares to σ]
    : ℝ[X]               [a polynomial with real coefficients] -/
def rel (σ : ℝ) : ℝ[X] := X ^ 2 - C σ

/-- The relation is monic, so the quotient has `1` and the generator as a basis.

    (σ : ℝ)              [the parameter]
    (rel σ).Monic        [its leading coefficient is one] -/
theorem rel_monic (σ : ℝ) : (rel σ).Monic := by
  have : rel σ = X ^ 2 - C σ := rfl
  rw [this]
  exact monic_X_pow_sub_C σ (by norm_num)

/-- In the quotient, the generator squares to `σ`.

    (σ : ℝ)                                       [the parameter]
    root (rel σ) ^ 2                              [the generator, squared]
    = algebraMap ℝ (AdjoinRoot (rel σ)) σ         [is σ, read into the quotient]

    So: this is the one relation, and every statement below is it applied twice. -/
theorem root_sq (σ : ℝ) :
    root (rel σ) ^ 2 = algebraMap ℝ (AdjoinRoot (rel σ)) σ := by
  have h : mk (rel σ) (rel σ) = 0 := mk_self
  have e : mk (rel σ) (rel σ) = root (rel σ) ^ 2 - algebraMap ℝ (AdjoinRoot (rel σ)) σ := by
    show mk (rel σ) (X ^ 2 - C σ) = _
    rw [map_sub, map_pow, AdjoinRoot.mk_X, AdjoinRoot.mk_C, AdjoinRoot.algebraMap_eq]
  rw [e] at h
  linear_combination h

/-- Send a pair across into the quotient.

    (σ : ℝ)                                   [the parameter]
    (z : Cell)                                [a point, a pair of reals]
    AdjoinRoot (rel σ)                        [the ring left after imposing the relation]
    algebraMap ℝ _ z.1                        [the first slot, read in as a constant]
    algebraMap ℝ _ z.2 * root (rel σ)         [the second slot, times the generator]

    So: the pair `(a, b)` becomes `a + b·x`. -/
def toQuot (σ : ℝ) (z : Cell) : AdjoinRoot (rel σ) :=
  algebraMap ℝ _ z.1 + algebraMap ℝ _ z.2 * root (rel σ)

/-- The formula is the quotient's multiplication.

    (σ : ℝ)                                       [the parameter]
    (z w : Cell)                                  [two points]
    toQuot σ z * toQuot σ w                       [multiplied inside the quotient]
    = toQuot σ (mul σ z w)                        [equals the formula's answer, sent across]

    So: `mul σ` is forced by the relation rather than chosen. This holds at every `σ`. -/
theorem mul_eq_quotient_mul (σ : ℝ) (z w : Cell) :
    toQuot σ z * toQuot σ w = toQuot σ (mul σ z w) := by
  simp only [toQuot, map_add, map_mul]
  have hr := root_sq σ
  ring_nf
  linear_combination (algebraMap ℝ (AdjoinRoot (rel σ)) z.2 *
    algebraMap ℝ (AdjoinRoot (rel σ)) w.2) * hr

/-- Sending a point across respects addition.

    (σ : ℝ)                                     [the parameter]
    (z w : Cell)                                [two points]
    toQuot σ (z + w) = toQuot σ z + toQuot σ w  [adding first or last gives the same] -/
theorem toQuot_add (σ : ℝ) (z w : Cell) :
    toQuot σ (z + w) = toQuot σ z + toQuot σ w := by
  simp only [toQuot, Prod.fst_add, Prod.snd_add, map_add]
  ring

/-- The unit goes to the unit.

    (σ : ℝ)                        [the parameter]
    toQuot σ one = 1               [the point (1,0) is the quotient's one] -/
theorem toQuot_one (σ : ℝ) : toQuot σ one = 1 := by
  simp [toQuot]

/-- The relation has degree two, so the quotient is two-dimensional over the reals.

    (σ : ℝ)                      [the parameter]
    (rel σ).natDegree = 2        [the relation is a quadratic] -/
theorem rel_natDegree (σ : ℝ) : (rel σ).natDegree = 2 := by
  show (X ^ 2 - C σ).natDegree = 2
  exact natDegree_X_pow_sub_C

/-- `1` and the generator are a basis of the quotient over the reals.

    (σ : ℝ)                                          [the parameter]
    Module.Basis (Fin 2) ℝ (AdjoinRoot (rel σ))      [two directions spanning the quotient]

    So: the relation is a monic quadratic, so remainders after dividing by it have degree
    below two, which is `a + b·x`. -/
def quotBasis (σ : ℝ) : Module.Basis (Fin 2) ℝ (AdjoinRoot (rel σ)) :=
  (powerBasis' (rel_monic σ)).basis.reindex (finCongr (rel_natDegree σ))

/-- The first basis direction is `1`.

    (σ : ℝ)                    [the parameter]
    quotBasis σ 0 = 1          [direction zero is the unit] -/
theorem quotBasis_zero (σ : ℝ) : quotBasis σ 0 = 1 := by
  simp only [quotBasis, Module.Basis.reindex_apply, PowerBasis.basis_eq_pow]
  show root (rel σ) ^ (0 : ℕ) = 1
  exact pow_zero _

/-- The second basis direction is the generator.

    (σ : ℝ)                              [the parameter]
    quotBasis σ 1 = root (rel σ)         [direction one is the generator] -/
theorem quotBasis_one (σ : ℝ) : quotBasis σ 1 = root (rel σ) := by
  simp only [quotBasis, Module.Basis.reindex_apply, PowerBasis.basis_eq_pow]
  show root (rel σ) ^ (1 : ℕ) = root (rel σ)
  exact pow_one _

/-- Sending a pair across is a linear map.

    (σ : ℝ)                                     [the parameter]
    Cell →ₗ[ℝ] AdjoinRoot (rel σ)               [a linear map from pairs into the quotient]

    So: `toQuot` again, packaged so that its bijectivity can be read off the basis. -/
def toQuotₗ (σ : ℝ) : Cell →ₗ[ℝ] AdjoinRoot (rel σ) where
  toFun := toQuot σ
  map_add' := toQuot_add σ
  map_smul' r z := by
    show toQuot σ (r • z) = r • toQuot σ z
    unfold toQuot
    rw [show (r • z).1 = r * z.1 from rfl, show (r • z).2 = r * z.2 from rfl,
      Algebra.smul_def, map_mul, map_mul]
    ring

/-- Sending a pair across is a bijection onto the quotient.

    (σ : ℝ)                              [the parameter]
    Cell ≃ₗ[ℝ] AdjoinRoot (rel σ)        [pairs and quotient elements match up, one to one]

    So: a pair is two coordinates against the two basis directions, and the quotient's
    elements are exactly the combinations of those two. -/
def toQuotEquiv (σ : ℝ) : Cell ≃ₗ[ℝ] AdjoinRoot (rel σ) :=
  (LinearEquiv.finTwoArrow ℝ ℝ).symm.trans (quotBasis σ).equivFun.symm

/-- The bijection is `toQuot`.

    (σ : ℝ)                              [the parameter]
    (z : Cell)                           [a point]
    toQuotEquiv σ z = toQuot σ z         [the bijection agrees with the map written earlier] -/
theorem toQuotEquiv_apply (σ : ℝ) (z : Cell) : toQuotEquiv σ z = toQuot σ z := by
  simp only [toQuotEquiv, LinearEquiv.trans_apply, Module.Basis.equivFun_symm_apply,
    Fin.sum_univ_two, quotBasis_zero, quotBasis_one]
  simp only [LinearEquiv.finTwoArrow_symm_apply, Matrix.cons_val_zero, Matrix.cons_val_one]
  simp only [toQuot, Algebra.smul_def, mul_one]

/-- `toQuot` is injective.

    (σ : ℝ)                              [the parameter]
    Function.Injective (toQuot σ)        [distinct pairs land on distinct elements] -/
theorem toQuot_injective (σ : ℝ) : Function.Injective (toQuot σ) := by
  intro a b hab
  have h : toQuotEquiv σ a = toQuotEquiv σ b := by
    rw [toQuotEquiv_apply, toQuotEquiv_apply]; exact hab
  exact (toQuotEquiv σ).injective h

/-- `toQuot` is surjective.

    (σ : ℝ)                               [the parameter]
    Function.Surjective (toQuot σ)        [every element of the quotient is some pair] -/
theorem toQuot_surjective (σ : ℝ) : Function.Surjective (toQuot σ) := by
  intro q
  obtain ⟨z, hz⟩ := (toQuotEquiv σ).surjective q
  exact ⟨z, by rw [← toQuotEquiv_apply]; exact hz⟩

/-- The origin is the only pair sent to zero.

    (σ : ℝ)                          [the parameter]
    (z : Cell)                       [a point]
    toQuot σ z = 0                   [it lands on zero in the quotient]
    ↔                                [exactly when]
    z = 0                            [it was the origin] -/
theorem toQuot_eq_zero_iff (σ : ℝ) (z : Cell) : toQuot σ z = 0 ↔ z = 0 := by
  constructor
  · intro h
    refine toQuot_injective σ ?_
    rw [h]
    show (0 : AdjoinRoot (rel σ)) = toQuot σ ((0, 0) : Cell)
    simp [toQuot]
  · rintro rfl
    show toQuot σ ((0, 0) : Cell) = 0
    simp [toQuot]

/-- A point's size is how hard it pulls on itself.

    (σ : ℝ) (z : Cell)   [the parameter, and one point]
    normForm σ z             [the size of z]
    = form σ z z         [z paired with itself]

    So: the two definitions agree when the second is given the same point twice. -/
theorem normForm_eq_form_self (σ : ℝ) (z : Cell) : normForm σ z = form σ z z :=
  AnyRing.ringForm_self σ z

/-- At `σ = −1` this is ordinary complex multiplication.

    mul (-1) z w                     [multiply with σ at −1]
    (z.1 * w.1 - z.2 * w.2, …)       [the answer `(ac − bd, ad + bc)`]
    (z w : Cell)                     [the two points]

    So: setting `σ` to −1 recovers `ℂ`, the number system where `i·i = −1`. -/
theorem mul_neg_one (z w : Cell) :
    mul (-1) z w = (z.1 * w.1 - z.2 * w.2, z.1 * w.2 + z.2 * w.1) := by
  apply Prod.ext <;> ring

/-- Size survives multiplication.

    (σ : ℝ)                    [the parameter, holding at every σ]
    (z w : Cell)               [two points]
    normForm σ (mul σ z w)         [multiply the points, then measure]
    = normForm σ z * normForm σ w      [measure each, then multiply the two sizes]

    So: measuring after multiplying gives the same answer as multiplying after measuring.
    A point of size zero therefore has no inverse: `0` times anything will not reach 1. -/
theorem normForm_mul (σ : ℝ) (z w : Cell) :
    normForm σ (mul σ z w) = normForm σ z * normForm σ w :=
  AnyRing.ringNormForm_mul σ z w

/-- Multiplying both arguments scales the pull by the multiplier's size.

    (σ : ℝ) (u z w : Cell)                          [the parameter and three points]
    form σ (mul σ u z) (mul σ u w)                  [multiply both, then measure the pull]
    = normForm σ u * form σ z w                     [the multiplier's size, times the old pull]

    So: `normForm_mul` opened out to two different points, and the statement that multiplication
    is a similarity of the form. At size one it is an isometry, which is what makes the size-one
    points a group of rotations, shears or boosts depending on the sign of `σ`. Taking `z = w`
    gives `normForm_mul` back. -/
theorem form_mul (σ : ℝ) (u z w : Cell) :
    form σ (mul σ u z) (mul σ u w) = normForm σ u * form σ z w :=
  AnyRing.ringForm_mul σ u z w

-- ═══════════════════════════════════════════════════════════════════
-- SIGMA BELOW ZERO: SIZE IS NEVER ZERO EXCEPT AT THE ORIGIN
-- ═══════════════════════════════════════════════════════════════════

/-- Below zero, only the origin has size zero.

    {σ : ℝ} (hσ : σ < 0)     [σ is below zero]
    (z : Cell)               [any point]
    normForm σ z = 0             [its size is zero]
    ↔                        [exactly when, in both directions]
    z = 0                    [the point is the origin]

    So: with `σ` below zero the size is `a² + |σ|b²`, a sum of two things that are
    never negative. The only way that total reaches zero is if both parts are zero. -/
theorem normForm_eq_zero_iff_of_neg {σ : ℝ} (hσ : σ < 0) (z : Cell) :
    normForm σ z = 0 ↔ z = 0 := by
  obtain ⟨a, b⟩ := z
  simp only [normForm, Prod.ext_iff, Prod.fst_zero, Prod.snd_zero]
  constructor
  · intro h
    have hba : 0 ≤ -σ * b ^ 2 :=
      mul_nonneg (neg_nonneg.mpr (le_of_lt hσ)) (sq_nonneg b)
    have ha2 : a ^ 2 = 0 := le_antisymm (by nlinarith) (sq_nonneg a)
    have hsb : σ * b ^ 2 = 0 := by linarith
    have hb2 : b ^ 2 = 0 := by
      rcases mul_eq_zero.mp hsb with h0 | h0
      · exact absurd h0 (ne_of_lt hσ)
      · exact h0
    exact ⟨pow_eq_zero_iff (by norm_num : (2 : ℕ) ≠ 0) |>.mp ha2,
           pow_eq_zero_iff (by norm_num : (2 : ℕ) ≠ 0) |>.mp hb2⟩
  · rintro ⟨ha, hb⟩; subst ha; subst hb; ring

/-- Below zero, a product reaches zero only if a factor was already zero.

    {σ : ℝ} (hσ : σ < 0)      [σ is below zero]
    (z w : Cell)              [two points]
    mul σ z w = 0             [their product is the origin]
    z = 0 ∨ w = 0             [then one of them was the origin to begin with]

    So: this is the property that makes division work. The product's size is zero, sizes
    multiply, so one factor had size zero, and by the previous result that factor was the
    origin. -/
theorem eq_zero_or_eq_zero_of_mul_eq_zero_of_neg {σ : ℝ} (hσ : σ < 0) (z w : Cell)
    (h : mul σ z w = 0) : z = 0 ∨ w = 0 := by
  have hn : normForm σ z * normForm σ w = 0 := by rw [← normForm_mul, h]; simp [normForm]
  rcases mul_eq_zero.mp hn with hz | hw
  · exact Or.inl ((normForm_eq_zero_iff_of_neg hσ z).mp hz)
  · exact Or.inr ((normForm_eq_zero_iff_of_neg hσ w).mp hw)

-- ═══════════════════════════════════════════════════════════════════
-- SIGMA AT ZERO OR ABOVE: TWO NON-ZERO THINGS THAT MULTIPLY TO NOTHING
-- ═══════════════════════════════════════════════════════════════════

/-- At zero or above, two such points, written down.

@cite euclid-elements-ii-5

    {σ : ℝ} (hσ : 0 ≤ σ)        [σ is at zero or above]
    ∃ z w : Cell                [there are two points]
    z ≠ 0 ∧ w ≠ 0               [neither one is the origin]
    mul σ z w = 0               [yet multiplying them gives the origin]

    So: the two points are `(√σ, 1)` and `(√σ, −1)`. `a² − σb²` factors as
    `(a + b√σ)(a − b√σ)` (Euclid II.5); those two factors are these points, and both are
    non-zero because their second slots are 1 and −1. -/
theorem exists_zeroDivisor_of_disc_nonneg {σ : ℝ} (hσ : 0 ≤ σ) :
    ∃ z w : Cell, z ≠ 0 ∧ w ≠ 0 ∧ mul σ z w = 0 := by
  refine ⟨(Real.sqrt σ, 1), (Real.sqrt σ, -1), ?_, ?_, ?_⟩
  · intro h; have h2 := congrArg Prod.snd h; simp at h2
  · intro h; have h2 := congrArg Prod.snd h; simp at h2
  · have hs : Real.sqrt σ * Real.sqrt σ = σ := Real.mul_self_sqrt hσ
    simp only [mul, Prod.ext_iff, Prod.fst_zero, Prod.snd_zero]
    exact ⟨by linear_combination hs, by ring⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE CLASSIFICATION BY SIGN
-- ═══════════════════════════════════════════════════════════════════

/-- A non-zero point of size zero exists exactly on the non-negative face.

    (σ : ℝ)                                        [the parameter]
    ∃ z : Cell, z ≠ 0 ∧ normForm σ z = 0               [some point other than zero has size zero]
    ↔                                              [exactly when]
    0 ≤ σ                                          [σ is at zero or above]

    So: below zero the size is a sum of two squares and vanishes only at the origin, so there is
    no point of size zero to fall onto. At or above zero there is one, and `(√σ, 1)` is it. -/
theorem exists_normForm_zero_iff_disc_nonneg (σ : ℝ) :
    (∃ z : Cell, z ≠ 0 ∧ normForm σ z = 0) ↔ 0 ≤ σ := by
  constructor
  · rintro ⟨z, hz, hn⟩
    by_contra hσ
    push Not at hσ
    have h1 : z.1 ^ 2 + (-σ) * z.2 ^ 2 = 0 := by simp only [normForm] at hn; linarith
    have hpos : 0 < -σ := by linarith
    have ha : z.1 ^ 2 = 0 := by nlinarith [sq_nonneg z.1, sq_nonneg z.2]
    have hb : z.2 ^ 2 = 0 := by nlinarith [sq_nonneg z.1, sq_nonneg z.2]
    exact hz (Prod.ext (sq_eq_zero_iff.mp ha) (sq_eq_zero_iff.mp hb))
  · intro hσ
    refine ⟨(Real.sqrt σ, 1), ?_, ?_⟩
    · intro h
      have : (1 : ℝ) = 0 := congrArg Prod.snd h
      norm_num at this
    · simp only [normForm, AnyRing.ringNormForm]
      rw [Real.sq_sqrt hσ]
      ring

/-- The conjugate: the same point with its second slot negated.

    (z : Cell)                  [a point]
    : Cell                      [the same point with the second slot negated]

    So: the involution every quadratic algebra carries. At `σ = −1` it is complex conjugation, at
    `σ = +1` it swaps the two null channels, and at `σ = 0` it negates the nilpotent part. -/
abbrev conj (z : Cell) : Cell := AnyRing.ringConj z

/-- Conjugating twice returns the point.

    (z : Cell)                  [a point]
    conj (conj z) = z           [conjugating twice changes nothing] -/
theorem conj_conj (z : Cell) : conj (conj z) = z :=
  AnyRing.ringConj_conj z

/-- A point times its conjugate is its size, in the first slot.

    (σ : ℝ) (z : Cell)                          [the parameter and a point]
    mul σ z (conj z) = (normForm σ z, 0)        [the product is the size, in the first slot]

    So: this is what makes the size multiplicative and what makes an inverse available whenever
    the size is not zero. -/
theorem mul_conj (σ : ℝ) (z : Cell) : mul σ z (conj z) = (normForm σ z, 0) :=
  AnyRing.ringMul_conj σ z

/-- Stretching a point squares into its size.

    (σ r : ℝ) (z : Cell)                        [the parameter, a stretch factor and a point]
    normForm σ (r • z)                          [stretch, then measure]
    = r ^ 2 * normForm σ z                      [the factor squared, times the size]

    So: the size is quadratic, so a stretch factor comes out squared. This is why the factor a
    Möbius map multiplies displacements by enters the line element as a square. -/
theorem normForm_smul (σ r : ℝ) (z : Cell) :
    normForm σ (r • z) = r ^ 2 * normForm σ z :=
  AnyRing.ringNormForm_smul σ r z

/-- The point that undoes a point lying off the cone.

    (σ : ℝ) (z : Cell)                          [the parameter and a point]
    : Cell                                      [the conjugate, divided by the size]

    So: `conj z` over `normForm σ z`. On the cone the size is zero and this returns the origin,
    which is the junk value Lean's division by zero carries; every statement about it below
    assumes the size is not zero. The name is `recip` rather than `inv` because the algebra is
    not a division algebra at `σ ≥ 0` and Lean's `Inv` class would promise that it is. -/
def recip (σ : ℝ) (z : Cell) : Cell := (z.1 / normForm σ z, -z.2 / normForm σ z)

/-- Off the cone, the reciprocal reciprocates.

    (σ : ℝ) (z : Cell)                          [the parameter and a point]
    (h : normForm σ z ≠ 0)                      [the point lies off the cone]
    mul σ z (recip σ z) = one                   [multiplying the two gives the unit] -/
theorem mul_recip_cancel (σ : ℝ) (z : Cell) (h : normForm σ z ≠ 0) :
    mul σ z (recip σ z) = one := by
  have hc := mul_conj σ z
  simp only [mul, recip, one, AnyRing.ringMul, AnyRing.ringOne,
    Prod.ext_iff] at hc ⊢
  obtain ⟨h1, h2⟩ := hc
  refine ⟨?_, ?_⟩
  · field_simp
    linarith [h1]
  · field_simp
    linear_combination h2

/-- On the cone the reciprocal is the origin.

    (σ : ℝ) (z : Cell)                          [the parameter and a point]
    (h : normForm σ z = 0)                      [the point lies on the cone]
    recip σ z = 0                               [the reciprocal is the origin]

    So: the junk value Lean's division by zero carries, named rather than left implicit, so
    that a statement about a formula which divides can say what the formula returns where the
    division is not available. `normForm_zero_not_invertible` is why no other value would
    serve: on the cone there is nothing to return. -/
theorem recip_on_cone (σ : ℝ) (z : Cell) (h : normForm σ z = 0) : recip σ z = 0 := by
  simp [recip, h]

/-- The size of a reciprocal is one divided by the size.

    (σ : ℝ) (z : Cell)                          [the parameter and a point]
    (h : normForm σ z ≠ 0)                      [the point lies off the cone]
    normForm σ (recip σ z)                      [measure the reciprocal]
    = (normForm σ z)⁻¹                          [one divided by the size]

    So: sizes multiply (`normForm_mul`) and the unit has size one, so the reciprocal's size is
    forced. Proved directly here because the direct proof is shorter than the forcing argument. -/
theorem normForm_recip (σ : ℝ) (z : Cell) (h : normForm σ z ≠ 0) :
    normForm σ (recip σ z) = (normForm σ z)⁻¹ := by
  simp only [normForm, recip] at h ⊢
  field_simp

/-- A point of size zero has no inverse, at every `σ`.

    (σ : ℝ) (z : Cell)                       [the parameter, and a point]
    (h : normForm σ z = 0)                       [the point has size zero]
    ¬ ∃ w : Cell, mul σ z w = one            [nothing multiplies with it to give the unit]

    So: the size is multiplicative (`normForm_mul`) and the unit has size one, so a product with a
    zero-size factor has size zero and cannot be the unit. No case split on `σ` is needed.
    Below zero this is not vacuous but it is empty: there the only point of size zero is zero
    itself (`noZeroDivisors_iff_disc_neg`). -/
theorem normForm_zero_not_invertible (σ : ℝ) (z : Cell) (h : normForm σ z = 0) :
    ¬ ∃ w : Cell, mul σ z w = one := by
  change ¬ ∃ w : AnyRing.RingCell ℝ, AnyRing.ringMul σ z w = AnyRing.ringOne
  rw [AnyRing.exists_inverse_iff_isUnit_ringNormForm]
  rw [show AnyRing.ringNormForm σ z = 0 from h]
  exact not_isUnit_zero

/-- A point can be divided by exactly when it is off the cone.

    (σ : ℝ) (z : Cell)                          [the parameter and a point]
    (∃ w : Cell, mul σ z w = one)               [something multiplies with it to give the unit]
    ↔                                           [exactly when]
    normForm σ z ≠ 0                            [its size is not zero]

    So: invertibility and the cone are one condition, not two. The forward direction is
    `normForm_zero_not_invertible` in contrapositive; the backward direction divides the
    conjugate by the size, which `mul_conj` says is exactly what is needed.

    This is the statement the projective coordinate rests on. Dehomogenising `[b : u]` needs `u`
    to be divisible by, and that is the same as `u` lying off the cone, which
    `norm_zero_iff_channel_zero` says is the same as neither channel reading zero. -/
theorem isUnit_iff_normForm_ne_zero (σ : ℝ) (z : Cell) :
    (∃ w : Cell, mul σ z w = one) ↔ normForm σ z ≠ 0 := by
  change (∃ w : AnyRing.RingCell ℝ, AnyRing.ringMul σ z w = AnyRing.ringOne) ↔
    AnyRing.ringNormForm σ z ≠ 0
  rw [AnyRing.exists_inverse_iff_isUnit_ringNormForm]
  exact isUnit_iff_ne_zero

/-- Two non-zero things multiply to zero exactly when `σ` is at zero or above.

    (σ : ℝ)                 [the parameter]
    ∃ z w : Cell            [there are two points, z and w]
    z ≠ 0 ∧ w ≠ 0           [neither one is zero]
    mul σ z w = 0           [multiplied the σ way, they give zero]
    ↔                       [exactly when, true in both directions]
    0 ≤ σ                   [σ sits at zero or above]

    So: two things that are not zero can still multiply to nothing, in exactly the
    algebras where `σ` is at zero or above. One direction is the pair written down
    above; the other is that below zero the sizes forbid it. -/
theorem hasZeroDivisor_iff_disc_nonneg (σ : ℝ) :
    (∃ z w : Cell, z ≠ 0 ∧ w ≠ 0 ∧ mul σ z w = 0) ↔ 0 ≤ σ := by
  constructor
  · rintro ⟨z, w, hz, hw, h⟩
    by_contra hσ
    push Not at hσ
    rcases eq_zero_or_eq_zero_of_mul_eq_zero_of_neg hσ z w h with h0 | h0
    · exact hz h0
    · exact hw h0
  · intro hσ; exact exists_zeroDivisor_of_disc_nonneg hσ

/-- Division works exactly when `σ` is below zero.

    (σ : ℝ)                              [the parameter]
    ∀ z w : Cell                         [for every pair of points]
    mul σ z w = 0 → z = 0 ∨ w = 0        [a zero product forces a zero factor]
    ↔                                    [exactly when]
    σ < 0                                [σ is below zero]

    So: the elliptic setting is the only one where division always works. This is the same
    fact as the previous theorem, said from the other side. -/
theorem noZeroDivisors_iff_disc_neg (σ : ℝ) :
    (∀ z w : Cell, mul σ z w = 0 → z = 0 ∨ w = 0) ↔ σ < 0 := by
  constructor
  · intro hdom
    by_contra hσ
    push Not at hσ
    obtain ⟨z, w, hz, hw, h⟩ := exists_zeroDivisor_of_disc_nonneg hσ
    rcases hdom z w h with h0 | h0
    · exact hz h0
    · exact hw h0
  · intro hσ z w h; exact eq_zero_or_eq_zero_of_mul_eq_zero_of_neg hσ z w h

/-- Two non-zero elements of the quotient multiply to zero exactly when `σ` is at zero
or above.

    (σ : ℝ)                                   [the parameter]
    ∃ p q : AdjoinRoot (rel σ)                [two elements of the quotient ring]
    p ≠ 0 ∧ q ≠ 0                             [neither is zero]
    p * q = 0                                 [their product, in the ring, is zero]
    ↔                                         [exactly when]
    0 ≤ σ                                     [σ sits at zero or above]

    So: `hasZeroDivisor_iff_disc_nonneg` on the ring. The multiplication is the quotient's
    own. -/
theorem quot_hasZeroDivisor_iff_disc_nonneg (σ : ℝ) :
    (∃ p q : AdjoinRoot (rel σ), p ≠ 0 ∧ q ≠ 0 ∧ p * q = 0) ↔ 0 ≤ σ := by
  rw [← hasZeroDivisor_iff_disc_nonneg σ]
  constructor
  · rintro ⟨p, q, hp, hq, hpq⟩
    obtain ⟨z, rfl⟩ := toQuot_surjective σ p
    obtain ⟨w, rfl⟩ := toQuot_surjective σ q
    rw [mul_eq_quotient_mul] at hpq
    exact ⟨z, w, fun h => hp ((toQuot_eq_zero_iff σ z).mpr h),
      fun h => hq ((toQuot_eq_zero_iff σ w).mpr h), (toQuot_eq_zero_iff σ _).mp hpq⟩
  · rintro ⟨z, w, hz, hw, hzw⟩
    exact ⟨toQuot σ z, toQuot σ w, fun h => hz ((toQuot_eq_zero_iff σ z).mp h),
      fun h => hw ((toQuot_eq_zero_iff σ w).mp h),
      by rw [mul_eq_quotient_mul, hzw]; exact (toQuot_eq_zero_iff σ 0).mpr rfl⟩

/-- The quotient is a domain exactly when `σ` is below zero.

    (σ : ℝ)                                     [the parameter]
    ∀ p q : AdjoinRoot (rel σ)                  [for every two elements of the ring]
    p * q = 0 → p = 0 ∨ q = 0                   [a zero product forces a zero factor]
    ↔                                           [exactly when]
    σ < 0                                       [σ is below zero]

    So: division works in the quotient on the elliptic face and nowhere else. -/
theorem quot_noZeroDivisors_iff_disc_neg (σ : ℝ) :
    (∀ p q : AdjoinRoot (rel σ), p * q = 0 → p = 0 ∨ q = 0) ↔ σ < 0 := by
  constructor
  · intro hdom
    by_contra hcon
    push Not at hcon
    obtain ⟨p, q, hp, hq, hpq⟩ := (quot_hasZeroDivisor_iff_disc_nonneg σ).mpr hcon
    rcases hdom p q hpq with h | h
    · exact hp h
    · exact hq h
  · intro hσ p q hpq
    by_contra hcon
    push Not at hcon
    exact absurd ((quot_hasZeroDivisor_iff_disc_nonneg σ).mp ⟨p, q, hcon.1, hcon.2, hpq⟩)
      (by linarith)

/-- At `σ = 0` exactly: one point squares to nothing.

    mul 0 (0, 1) (0, 1) = 0     [the point (0,1) times itself gives the origin]
    ((0, 1) : Cell) ≠ 0         [and yet (0,1) is not the origin]
    ∧                           [both of these hold]

    So: these are the dual numbers, the middle setting. One point that is not zero has
    square zero. Unlike the case above zero, it is a single point, not two distinct ones. -/
theorem nilpotent_of_disc_zero :
    mul 0 (0, 1) (0, 1) = 0 ∧ ((0, 1) : Cell) ≠ 0 := by
  refine ⟨?_, ?_⟩
  · simp only [mul, Prod.ext_iff, Prod.fst_zero, Prod.snd_zero]; norm_num
  · intro h; have := congrArg Prod.snd h; simp at this

/-- At `σ = 1` exactly: two distinct points with product zero.

    mul 1 (1, 1) (1, -1) = 0     [(1,1) times (1,−1) gives the origin]
    ((1, 1) : Cell) ≠ 0          [(1,1) is not the origin]
    ((1, -1) : Cell) ≠ 0         [(1,−1) is not the origin either]
    ∧                            [all three hold together]

    So: Euclid II.5 at `a = b = 1`. `1² − 1² = 0`, and the factors `(1+x)` and `(1−x)`
    are both non-zero. This setting is the split-complex numbers. -/
theorem zeroDivisorPair_of_disc_one :
    mul 1 (1, 1) (1, -1) = 0 ∧ ((1, 1) : Cell) ≠ 0 ∧ ((1, -1) : Cell) ≠ 0 := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [mul, Prod.ext_iff, Prod.fst_zero, Prod.snd_zero]; norm_num
  · intro h; have := congrArg Prod.fst h; simp at this
  · intro h; have := congrArg Prod.fst h; simp at this

-- ═══════════════════════════════════════════════════════════════════
-- COMPLETENESS: every two-direction algebra IS one of the three
-- ═══════════════════════════════════════════════════════════════════
-- Complete the square to kill the linear term, then rescale to drive the
-- generator's square to −1, 0, or 1. Nothing is cited: the table above is
-- a theorem about every two-direction real algebra there is or could be.

section Completeness

open Module

variable {A : Type*} [CommRing A] [Algebra ℝ A]

/-- A two-direction algebra contains an element that is not a scalar.

    (h2 : Module.finrank ℝ A = 2)     [the algebra has exactly two independent directions]
    ∃ u : A                           [there is an element u]
    ∀ a : ℝ, a • (1 : A) ≠ u          [no plain number, stretched, ever equals it]

    So: the plain numbers sit inside the algebra as a single line. Two directions means
    there is more than that line, so something lies off it. That something is the
    generator every argument below starts from. -/
theorem exists_nonscalar (h2 : Module.finrank ℝ A = 2) :
    ∃ u : A, ∀ a : ℝ, a • (1 : A) ≠ u := by
  have hnt : Nontrivial A := by
    rcases subsingleton_or_nontrivial A with hs | hn
    · haveI := hs
      rw [Module.finrank_zero_of_subsingleton] at h2; exact absurd h2 (by norm_num)
    · exact hn
  have hne : (Submodule.span ℝ {(1 : A)}) ≠ ⊤ := by
    intro h
    have h1 : Module.finrank ℝ (Submodule.span ℝ {(1 : A)}) = 1 :=
      finrank_span_singleton (one_ne_zero)
    rw [h, finrank_top, h2] at h1
    exact absurd h1 (by norm_num)
  obtain ⟨u, -, hu⟩ := SetLike.exists_of_lt (lt_top_iff_ne_top.mpr hne)
  refine ⟨u, fun a ha => hu ?_⟩
  rw [← ha]
  exact Submodule.smul_mem _ _ (Submodule.mem_span_singleton_self _)

/-- The grid: the number 1, and one direction off it.

    (h2 : Module.finrank ℝ A = 2)        [two independent directions]
    (ω : A)                              [a chosen element]
    (hω : ∀ a : ℝ, a • (1 : A) ≠ ω)      [which is not a stretched number]
    Basis (Fin 2) ℝ A                    [these two are a grid for the whole algebra]

    So: an element off the number line, together with 1, writes every element of the
    algebra exactly one way. `pairBasis` is that grid. -/
noncomputable def pairBasis (h2 : Module.finrank ℝ A = 2) (ω : A)
    (hω : ∀ a : ℝ, a • (1 : A) ≠ ω) : Basis (Fin 2) ℝ A := by
  have hnt : Nontrivial A := by
    rcases subsingleton_or_nontrivial A with hs | hn
    · haveI := hs; exact absurd (Subsingleton.elim ((0 : ℝ) • (1 : A)) ω) (hω 0)
    · exact hn
  exact basisOfLinearIndependentOfCardEqFinrank
    ((LinearIndependent.pair_iff' (one_ne_zero)).mpr hω) (by simp [h2])

/-- The grid's first direction is the number 1.

    (h2 : Module.finrank ℝ A = 2)       [two directions]
    (ω : A)                             [the chosen off-line element]
    (hω : ∀ a : ℝ, a • (1 : A) ≠ ω)     [it is off the line]
    pairBasis h2 ω hω 0 = 1             [slot 0 of the grid holds 1] -/
@[simp] theorem pairBasis_zero (h2 : Module.finrank ℝ A = 2) (ω : A)
    (hω : ∀ a : ℝ, a • (1 : A) ≠ ω) : pairBasis h2 ω hω 0 = 1 := by
  simp [pairBasis]

/-- The grid's second direction is the chosen element.

    (h2 : Module.finrank ℝ A = 2)       [two directions]
    (ω : A)                             [the chosen off-line element]
    (hω : ∀ a : ℝ, a • (1 : A) ≠ ω)     [it is off the line]
    pairBasis h2 ω hω 1 = ω             [slot 1 of the grid holds ω] -/
@[simp] theorem pairBasis_one (h2 : Module.finrank ℝ A = 2) (ω : A)
    (hω : ∀ a : ℝ, a • (1 : A) ≠ ω) : pairBasis h2 ω hω 1 = ω := by
  simp [pairBasis]

/-- Every two-direction algebra is one of the three.

    (h2 : Module.finrank ℝ A = 2)    [any algebra with exactly two independent directions]
    ∃ σ : ℝ                          [there is such a σ]
    σ = -1 ∨ σ = 0 ∨ σ = 1           [it is one of exactly three values]
    ∧                                [and, on top of that]
    ∃ e : A ≃ₗ[ℝ] ℝ × ℝ              [the algebra can be relabelled as pairs of reals]
    e 1 = (1, 0)                     [the relabelling sends 1 to (1,0)]
    ∀ z w : A                        [for every two elements]
    e (z * w) = mul σ (e z) (e w)    [multiply then relabel, same as relabel then multiply]

    So: the three species are the complete list, not a selection.

    Method, in two moves. Take the off-line element `u` and write `u·u` on the grid as
    `c₀ + c₁u`. Replace `u` by `u − c₁/2`, cancelling the `c₁u` term, as when completing
    the square in a quadratic; the new element's square is then a plain number `d`. Divide
    that element by `√|d|` and its square becomes −1, 0, or 1 according to the sign of
    `d`. -/
theorem classification (h2 : Module.finrank ℝ A = 2) :
    ∃ σ : ℝ, (σ = -1 ∨ σ = 0 ∨ σ = 1) ∧
      ∃ e : A ≃ₗ[ℝ] ℝ × ℝ, e 1 = (1, 0) ∧ ∀ z w : A, e (z * w) = mul σ (e z) (e w) := by
  obtain ⟨u, hu⟩ := exists_nonscalar h2
  set B := pairBasis h2 u hu with hB
  set c₀ := B.repr (u * u) 0 with hc₀
  set c₁ := B.repr (u * u) 1 with hc₁
  have hu2 : u * u = c₀ • (1 : A) + c₁ • u := by
    have := B.sum_repr (u * u)
    rw [Fin.sum_univ_two, pairBasis_zero, pairBasis_one] at this
    exact this.symm
  -- complete the square: v = u − t·1 has no linear term
  set t := c₁ / 2 with ht
  set d := c₀ + t ^ 2 with hd
  set v := u - t • (1 : A) with hv
  have hvv : v * v = algebraMap ℝ A d := by
    have hu2' : u * u = algebraMap ℝ A c₀ + 2 * algebraMap ℝ A t * u := by
      have h2t : c₁ = 2 * t := by rw [ht]; ring
      rw [hu2, h2t, Algebra.smul_def, Algebra.smul_def, mul_one, map_mul, map_ofNat]
    rw [hv, hd, map_add, map_pow, Algebra.smul_def, mul_one]
    linear_combination hu2'
  -- something off the number line stays off it under `r • u + q • 1`, `r ≠ 0`
  have hshift : ∀ r q : ℝ, r ≠ 0 → ∀ a : ℝ, a • (1 : A) ≠ r • u + q • (1 : A) := by
    intro r q hr a hcon
    refine hu ((a - q) / r) ?_
    have hru : r • u = (a - q) • (1 : A) := by
      have : r • u = a • (1 : A) - q • (1 : A) := by rw [hcon]; abel
      rw [this, sub_smul]
    calc ((a - q) / r) • (1 : A) = (r⁻¹ * (a - q)) • (1 : A) := by rw [div_eq_inv_mul]
      _ = r⁻¹ • (r • u) := by rw [hru, smul_smul]
      _ = u := by rw [smul_smul, inv_mul_cancel₀ hr, one_smul]
  -- rescale: drive the generator's square to a sign
  have main : ∀ (σ r q : ℝ), r ≠ 0 → (r • u + q • (1 : A)) * (r • u + q • (1 : A))
      = algebraMap ℝ A σ →
      ∃ e : A ≃ₗ[ℝ] ℝ × ℝ, e 1 = (1, 0) ∧ ∀ z w : A, e (z * w) = mul σ (e z) (e w) := by
    intro σ r q hr hsq
    set ω := r • u + q • (1 : A) with hω
    set C := pairBasis h2 ω (hshift r q hr) with hC
    refine ⟨C.equivFun.trans (LinearEquiv.finTwoArrow ℝ ℝ), ?_, ?_⟩
    · have : (1 : A) = C 0 := (pairBasis_zero h2 ω (hshift r q hr)).symm
      simp [this, Basis.equivFun_apply, LinearEquiv.finTwoArrow]
    · have hcoord : ∀ x y : ℝ,
          (C.equivFun.trans (LinearEquiv.finTwoArrow ℝ ℝ)) (x • (1 : A) + y • ω) = (x, y) := by
        intro x y
        have h1 : (1 : A) = C 0 := (pairBasis_zero h2 ω (hshift r q hr)).symm
        have h2' : ω = C 1 := (pairBasis_one h2 ω (hshift r q hr)).symm
        rw [h1, h2']
        simp [Basis.equivFun_apply, LinearEquiv.finTwoArrow]
      have hrepr : ∀ z : A,
          ((C.equivFun.trans (LinearEquiv.finTwoArrow ℝ ℝ)) z).1 • (1 : A)
            + ((C.equivFun.trans (LinearEquiv.finTwoArrow ℝ ℝ)) z).2 • ω = z := by
        intro z
        have := C.sum_repr z
        rw [Fin.sum_univ_two, pairBasis_zero, pairBasis_one] at this
        simpa [Basis.equivFun_apply, LinearEquiv.finTwoArrow] using this
      have hmul : ∀ x₁ x₂ y₁ y₂ : ℝ,
          (x₁ • (1 : A) + x₂ • ω) * (y₁ • (1 : A) + y₂ • ω)
            = (x₁ * y₁ + σ * x₂ * y₂) • (1 : A) + (x₁ * y₂ + x₂ * y₁) • ω := by
        intro x₁ x₂ y₁ y₂
        simp only [Algebra.smul_def, mul_one, map_add, map_mul]
        linear_combination (algebraMap ℝ A x₂ * algebraMap ℝ A y₂) * hsq
      intro z w
      conv_lhs => rw [← hrepr z, ← hrepr w, hmul]
      rw [hcoord]
  rcases lt_trichotomy d 0 with hlt | heq | hgt
  · -- d below zero: the elliptic setting
    refine ⟨-1, Or.inl rfl, main (-1) (Real.sqrt (-d))⁻¹ (-(t / Real.sqrt (-d))) ?_ ?_⟩
    · exact inv_ne_zero (Real.sqrt_ne_zero'.mpr (by linarith))
    · have hs : Real.sqrt (-d) * Real.sqrt (-d) = -d := Real.mul_self_sqrt (by linarith)
      have hsne : Real.sqrt (-d) ≠ 0 := Real.sqrt_ne_zero'.mpr (by linarith)
      have hsc : -(t / Real.sqrt (-d)) = -((Real.sqrt (-d))⁻¹ * t) := by rw [div_eq_inv_mul]
      have hsplit : (Real.sqrt (-d))⁻¹ • u + (-(t / Real.sqrt (-d))) • (1 : A)
          = (Real.sqrt (-d))⁻¹ • v := by
        rw [hv, smul_sub, smul_smul, hsc, neg_smul, ← sub_eq_add_neg]
      rw [hsplit, smul_mul_smul_comm, hvv, Algebra.smul_def, ← map_mul]
      congr 1
      field_simp
      linarith [hs]
  · -- d exactly zero: the degenerate setting
    refine ⟨0, Or.inr (Or.inl rfl), main 0 1 (-t) one_ne_zero ?_⟩
    have : (1 : ℝ) • u + (-t) • (1 : A) = v := by rw [hv, one_smul, neg_smul]; abel
    rw [this, hvv, heq, map_zero]
  · -- d above zero: the hyperbolic setting
    refine ⟨1, Or.inr (Or.inr rfl), main 1 (Real.sqrt d)⁻¹ (-(t / Real.sqrt d)) ?_ ?_⟩
    · exact inv_ne_zero (Real.sqrt_ne_zero'.mpr hgt)
    · have hs : Real.sqrt d * Real.sqrt d = d := Real.mul_self_sqrt hgt.le
      have hsne : Real.sqrt d ≠ 0 := Real.sqrt_ne_zero'.mpr hgt
      have hsc : -(t / Real.sqrt d) = -((Real.sqrt d)⁻¹ * t) := by rw [div_eq_inv_mul]
      have hsplit : (Real.sqrt d)⁻¹ • u + (-(t / Real.sqrt d)) • (1 : A)
          = (Real.sqrt d)⁻¹ • v := by
        rw [hv, smul_sub, smul_smul, hsc, neg_smul, ← sub_eq_add_neg]
      rw [hsplit, smul_mul_smul_comm, hvv, Algebra.smul_def, ← map_mul]
      congr 1
      field_simp
      linarith [hs]

/-- The zero-divisor law, read off any two-direction algebra at all.

    (h2 : Module.finrank ℝ A = 2)    [any algebra with exactly two directions]
    ∃ σ : ℝ                          [there is such a σ]
    σ = -1 ∨ σ = 0 ∨ σ = 1           [one of three values]
    ∃ z w : A, z ≠ 0 ∧ w ≠ 0         [two non-zero elements]
    z * w = 0                        [multiplying to zero]
    ↔                                [exactly when]
    0 ≤ σ                            [that σ is zero or above]

    So: with `hasZeroDivisor_iff_disc_nonneg`, the sign of `σ` decides the question
    for every two-direction algebra, not only the three written down here. -/
theorem trichotomy_complete (h2 : Module.finrank ℝ A = 2) :
    ∃ σ : ℝ, (σ = -1 ∨ σ = 0 ∨ σ = 1) ∧
      ((∃ z w : A, z ≠ 0 ∧ w ≠ 0 ∧ z * w = 0) ↔ 0 ≤ σ) := by
  obtain ⟨σ, hσ, e, -, hmul⟩ := classification h2
  refine ⟨σ, hσ, ?_⟩
  rw [← hasZeroDivisor_iff_disc_nonneg σ]
  constructor
  · rintro ⟨z, w, hz, hw, h⟩
    exact ⟨e z, e w, fun hc => hz (by simpa using congrArg e.symm hc),
      fun hc => hw (by simpa using congrArg e.symm hc), by rw [← hmul, h, map_zero]⟩
  · rintro ⟨z, w, hz, hw, h⟩
    refine ⟨e.symm z, e.symm w, fun hc => hz (by simpa using congrArg e hc),
      fun hc => hw (by simpa using congrArg e hc), ?_⟩
    have := hmul (e.symm z) (e.symm w)
    rw [e.apply_symm_apply, e.apply_symm_apply, h] at this
    simpa using congrArg e.symm this

/-- Every two-direction algebra is `ℝ[X]/(X² − σ)` for one of three values of `σ`.

    (h2 : Module.finrank ℝ A = 2)          [any algebra with exactly two directions]
    ∃ σ : ℝ                                [there is such a σ]
    σ = -1 ∨ σ = 0 ∨ σ = 1                 [one of exactly three values]
    Nonempty (A ≃+* AdjoinRoot (rel σ))    [and the algebra is that quotient, as a ring]
    ∧                                      [both together]

    So: this is the end of the algebra layer. `classification` puts every such algebra in the
    form `mul σ`, `mul_eq_quotient_mul` identifies that form with the quotient's own
    multiplication, and `toQuotEquiv` says the identification is a bijection. Composed, they
    say a two-direction algebra is a copy of `ℝ[X]/(X² − σ)`: addition, multiplication and
    unit alike, not just described by it.

    Nothing below this rests on a choice of `σ`. Results needing `σ > 0` are downstream. -/
theorem quotient_classification (h2 : Module.finrank ℝ A = 2) :
    ∃ σ : ℝ, (σ = -1 ∨ σ = 0 ∨ σ = 1) ∧ Nonempty (A ≃+* AdjoinRoot (rel σ)) := by
  obtain ⟨σ, hσ, e, he1, hemul⟩ := classification h2
  let φ : A →+* AdjoinRoot (rel σ) :=
    { toFun := fun z => toQuot σ (e z)
      map_one' := by rw [he1]; exact toQuot_one σ
      map_mul' := fun z w => by rw [hemul, mul_eq_quotient_mul]
      map_zero' := by
        rw [map_zero]
        show toQuot σ ((0, 0) : Cell) = 0
        simp [toQuot]
      map_add' := fun z w => by rw [map_add, toQuot_add] }
  have hinj : Function.Injective φ := fun a b hab =>
    e.injective (toQuot_injective σ hab)
  have hsurj : Function.Surjective φ := by
    intro q
    obtain ⟨z, hz⟩ := toQuot_surjective σ q
    exact ⟨e.symm z, by show toQuot σ (e (e.symm z)) = q; rw [e.apply_symm_apply]; exact hz⟩
  exact ⟨σ, hσ, ⟨RingEquiv.ofBijective φ ⟨hinj, hsurj⟩⟩⟩

end Completeness

-- ═══════════════════════════════════════════════════════════════════
-- THE NULL PAIR: sigma strictly above zero
-- ═══════════════════════════════════════════════════════════════════
-- Two non-zero things multiplying to zero starts at `0 ≤ σ`. Two *different*
-- directions of size zero that still pull on each other needs `0 < σ`
-- strictly. The middle setting has a direction of size zero, but only one,
-- and it pulls on itself with strength zero. That is its degeneracy.

/-- Two directions of size zero that still pull on each other.

    (σ : ℝ)                    [the parameter]
    ∃ z w : Cell               [there are two points]
    normForm σ z = 0               [the first has size zero]
    normForm σ w = 0               [so does the second]
    form σ z w ≠ 0             [yet they pull on each other with non-zero strength]

    So: a non-degenerate light cone. `HasNullPair` names the property; the next theorem
    says when it holds. -/
def HasNullPair (σ : ℝ) : Prop :=
  ∃ z w : Cell, normForm σ z = 0 ∧ normForm σ w = 0 ∧ form σ z w ≠ 0

/-- The first size-zero direction, written down.

    (σ : ℝ) (hσ : 0 ≤ σ)              [σ at zero or above]
    normForm σ (Real.sqrt σ, 1) = 0       [the point (√σ, 1) has size zero] -/
theorem normForm_nullGen_pos (σ : ℝ) (hσ : 0 ≤ σ) : normForm σ (Real.sqrt σ, 1) = 0 := by
  simp only [normForm, AnyRing.ringNormForm]; rw [Real.sq_sqrt hσ]; ring

/-- The second size-zero direction, written down.

    (σ : ℝ) (hσ : 0 ≤ σ)               [σ at zero or above]
    normForm σ (Real.sqrt σ, -1) = 0       [the point (√σ, −1) also has size zero] -/
theorem normForm_nullGen_neg (σ : ℝ) (hσ : 0 ≤ σ) : normForm σ (Real.sqrt σ, -1) = 0 := by
  simp only [normForm, AnyRing.ringNormForm]; rw [Real.sq_sqrt hσ]; ring

/-- The two size-zero directions pull on each other with strength `2σ`.

    (σ : ℝ) (hσ : 0 ≤ σ)                              [σ at zero or above]
    form σ (Real.sqrt σ, 1) (Real.sqrt σ, -1)         [pair up the two directions]
    = 2 * σ                                           [the pull is twice σ]

    So: the strength is `σ` doubled. It is zero at the middle setting, which is that
    setting's degeneracy, and non-zero above it. -/
theorem form_nullGen (σ : ℝ) (hσ : 0 ≤ σ) :
    form σ (Real.sqrt σ, 1) (Real.sqrt σ, -1) = 2 * σ := by
  have hs : Real.sqrt σ * Real.sqrt σ = σ := Real.mul_self_sqrt hσ
  simp only [form, AnyRing.ringForm]; rw [hs]; ring

/-- A genuine light cone exists exactly when `σ` is above zero.

    (σ : ℝ)                 [the parameter]
    HasNullPair σ           [two size-zero directions pulling on each other]
    ↔                       [exactly when]
    0 < σ                   [σ is strictly above zero]

    So: below zero only the origin has size zero. At exactly zero there is one size-zero
    direction, pulling on itself with strength zero. Above zero there are two directions
    at strength `2σ`. -/
theorem hasNullPair_iff_disc_pos (σ : ℝ) : HasNullPair σ ↔ 0 < σ := by
  constructor
  · rintro ⟨z, w, hz, hw, hp⟩
    rcases lt_trichotomy σ 0 with h | h | h
    · rw [(normForm_eq_zero_iff_of_neg h z).mp hz] at hp; simp [form, AnyRing.ringForm] at hp
    · subst h
      have hz1 : z.1 = 0 := by
        have h2 : z.1 ^ 2 = 0 := by simpa [normForm] using hz
        exact pow_eq_zero_iff (by norm_num : (2:ℕ) ≠ 0) |>.mp h2
      have hw1 : w.1 = 0 := by
        have h2 : w.1 ^ 2 = 0 := by simpa [normForm] using hw
        exact pow_eq_zero_iff (by norm_num : (2:ℕ) ≠ 0) |>.mp h2
      simp [form, AnyRing.ringForm, hz1, hw1] at hp
    · exact h
  · intro hσ
    exact ⟨_, _, normForm_nullGen_pos σ hσ.le, normForm_nullGen_neg σ hσ.le,
      by rw [form_nullGen σ hσ.le]; positivity⟩

end Trichotomy
end Apeiron

end
