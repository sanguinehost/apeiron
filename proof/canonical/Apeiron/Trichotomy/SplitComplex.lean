import Apeiron.Trichotomy.Idempotents
import Mathlib.Data.Complex.Basic
import Mathlib.LinearAlgebra.Dimension.Constructions

/-!
# The Split-Complex Algebra

@name The Split-Complex Algebra
@cite euclid-elements-ii-5
@cite cockle-1849
@cite clifford-1873
@cite minkowski-1908
@cite peirce-1870
@cite boole-1847
@cite mathlib-complex
@grade proved

Take two ordinary real number lines and set them side by side. To multiply two pairs, multiply
the first entries together and the second entries together, and let the two lines never talk to
each other:

```
   (3, 5) · (2, 7)  =  (6, 35)
```

That is the whole algebra. It is `ℝ × ℝ` under componentwise multiplication, and it is also
`ℝ[j]/(j² − 1)`, the `σ = 1` setting of `Apeiron.Trichotomy.Basic`. The two look nothing
alike until the change of basis:

```
   e₊ = (1 + j)/2          e₋ = (1 − j)/2
   e₊ · e₊ = e₊            e₋ · e₋ = e₋            e₊ · e₋ = 0
   e₊ + e₋ = 1
```

`e₊` and `e₋` are the two channels. Each one is unchanged by squaring, they annihilate each
other, and they add to the unit. Peirce wrote this down in 1870.

Because the channels are independent, a point lying in one channel is invisible to
the other, and multiplying the two together gives nothing. That is `(1+j)(1−j) = 0`, the
failure of division Cockle recorded in 1849, and it is Euclid's `a² − b² = (a+b)(a−b)`.

Seen the other way the same object is a plane with an indefinite metric, Minkowski's signature
`(1,1)`, where the set of points of size zero is a light cone rather than a single point.

`x² = x` has solutions here other than `0` and `1`, which
`Trichotomy.exists_nontrivial_idempotent_iff_disc_pos` shows happens exactly when sigma
is positive.

`disc_eq_one` is the direction that costs something. Rather than defining this algebra to sit
at `σ = 1`, it runs `classification` on the carrier and shows the value of sigma that comes back
is `1`: the two idempotents rule out `σ = −1`, the absence of a nilpotent rules out `σ = 0`.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

noncomputable section

open Module

namespace Apeiron
namespace Trichotomy
namespace SplitComplex

/-- A point here is a pair of real numbers.

    SC       [the name for one point of this algebra]
    ℝ × ℝ    [a real number, then another real number]

    So: written `a + b·j`, where `j` is the label for the second slot. This is the generic
    cell `AnyRing.RingCell` at `R = ℝ`, not a second copy of the pair. -/
abbrev SC : Type := AnyRing.RingCell ℝ

/-- How to multiply two points.

    (z w : SC)                   [the two points]
    z.1 * w.1 + z.2 * w.2        [the first slot of the answer]
    z.1 * w.2 + z.2 * w.1        [the second slot of the answer]

    So: `(a+bj)(c+dj) = (ac+bd) + (ad+bc)j`. Compare `ℂ`, which has `ac−bd` in the first
    slot. The sign is the only difference. This is `AnyRing.ringMul` at `R = ℝ`, `σ = 1`,
    not a separately authored formula. -/
abbrev mul (z w : SC) : SC := AnyRing.ringMul 1 z w

/-- The size of a point.

    (z : SC)             [the point being measured]
    z.1 ^ 2 - z.2 ^ 2    [first slot squared minus second slot squared]
    : ℝ                  [a plain real number]

    So: `N(a + bj) = a² − b²`, the left-hand side of Euclid II.5. It takes both signs. This
    is `AnyRing.ringNormForm` at `R = ℝ`, `σ = 1`. -/
abbrev normForm (z : SC) : ℝ := AnyRing.ringNormForm 1 z

/-- How much two points pull on each other.

    (z w : SC)                   [the two points]
    z.1 * w.1 - z.2 * w.2        [their pull]
    : ℝ                          [a plain real number]

    So: `normForm` opened out to two different points. This is `AnyRing.ringForm` at
    `R = ℝ`, `σ = 1`, so like `SC`, `mul`, `normForm` and `one` it is a specialization
    rather than a second formula. -/
abbrev form (z w : SC) : ℝ := AnyRing.ringForm 1 z w

/-- The second direction.

    j          [the name for this point]
    (0, 1)     [zero in the first slot, one in the second]
    : SC       [it is a point of this algebra]

    So: `j` squares to `+1`, where `ℂ`'s `i` squares to `−1`. -/
def j : SC := (0, 1)

/-- The point that changes nothing under multiplication.

    one        [the name for this point]
    (1, 0)     [one in the first slot, zero in the second]
    : SC       [a point of this algebra]

    So: `AnyRing.ringOne` at `R = ℝ`, not a separately authored pair. -/
abbrev one : SC := AnyRing.ringOne

/-- One of the two points of size zero.

    nullPlus     [the name for this point]
    (1, 1)       [one in each slot, that is `1 + j`]
    : SC         [a point of this algebra]

    So: its size is `1² − 1² = 0`, and it is not the origin. -/
def nullPlus : SC := (1, 1)

/-- The other point of size zero.

    nullMinus    [the name for this point]
    (1, -1)      [that is `1 − j`]
    : SC         [a point of this algebra] -/
def nullMinus : SC := (1, -1)

-- ═══════════════════════════════════════════════════════════════════
-- THIS ALGEBRA IS THE GENERAL CELL AT σ = 1
-- ═══════════════════════════════════════════════════════════════════

/-- The general multiplication at `σ = 1` is this multiplication.

    (z w : SC)                 [two points]
    Trichotomy.mul 1 z w       [the general rule, σ set to 1]
    = mul z w                  [this file's rule]

    So: not a resemblance. The two definitions are equal, because both are
    `AnyRing.ringMul 1`. -/
theorem mul_eq_mul_one (z w : SC) : Trichotomy.mul 1 z w = mul z w := rfl

/-- The general size at `σ = 1` is this size.

    (z : SC)                [a point]
    Trichotomy.normForm 1 z     [the general size, σ set to 1]
    = normForm z                [this file's size]

    So: the two are the same function, because both are `AnyRing.ringNormForm 1`. -/
theorem normForm_eq_normForm_one (z : SC) : Trichotomy.normForm 1 z = normForm z := rfl

/-- The general pull at `σ = 1` is this pull.

    (z w : SC)                  [two points]
    Trichotomy.form 1 z w       [the general pull, σ set to 1]
    = form z w                  [this file's pull]

    So: both sides are `AnyRing.ringForm 1`, so this too is a definitional identity. -/
theorem form_eq_form_one (z w : SC) : Trichotomy.form 1 z w = form z w := rfl

-- ═══════════════════════════════════════════════════════════════════
-- j² = 1, AND SIZE SURVIVES MULTIPLICATION
-- ═══════════════════════════════════════════════════════════════════

/-- `j` squared is 1.

    mul j j     [multiply the second direction by itself]
    = one       [the answer is the point (1,0)]

    So: the defining identity. `ℂ` has `i·i = −1`. -/
theorem j_sq : mul j j = one := by
  simp [mul, j, one]

/-- Size survives multiplication.

    (z w : SC)                  [two points]
    normForm (mul z w)              [multiply, then measure]
    = normForm z * normForm w           [measure each, then multiply the sizes]

    So: `AnyRing.ringNormForm_mul` at `R = ℝ`, `σ = 1`, used directly. -/
theorem normForm_mul (z w : SC) : normForm (mul z w) = normForm z * normForm w :=
  AnyRing.ringNormForm_mul 1 z w

/-- The unit has size 1.

    normForm one = 1     [measuring the point (1,0) gives 1] -/
@[simp] theorem normForm_one : normForm one = 1 := by simp [normForm]

/-- The second direction has size −1.

    normForm j = -1     [measuring the point (0,1) gives −1] -/
@[simp] theorem normForm_j : normForm j = -1 := by norm_num [normForm, j]

/-- Size takes both signs.

    ∃ z w : SC              [there are two points]
    normForm z > 0              [the first measures positive]
    normForm w < 0              [the second measures negative]
    ∧                       [both at once]

    So: signature `(1,1)`, Minkowski's. `ℂ`'s size is never negative. -/
theorem normForm_indefinite : ∃ z w : SC, normForm z > 0 ∧ normForm w < 0 :=
  ⟨one, j, by rw [normForm_one]; norm_num, by rw [normForm_j]; norm_num⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE POINTS OF SIZE ZERO
-- ═══════════════════════════════════════════════════════════════════

/-- `1 + j` has size zero.

    normForm nullPlus = 0     [measuring the point (1,1) gives 0] -/
theorem normForm_nullPlus : normForm nullPlus = 0 := by norm_num [normForm, nullPlus]

/-- `1 − j` has size zero.

    normForm nullMinus = 0     [measuring the point (1,−1) gives 0] -/
theorem normForm_nullMinus : normForm nullMinus = 0 := by norm_num [normForm, nullMinus]

/-- `1 + j` is not the origin.

    nullPlus ≠ (0, 0)     [the point (1,1) is not the origin]

    So: with the previous result, a point of size zero that is not zero. `ℂ` has none. -/
theorem nullPlus_ne_zero : nullPlus ≠ (0, 0) := by
  simp [nullPlus, Prod.ext_iff]

/-- The two size-zero points multiply to the origin.

    mul nullPlus nullMinus     [(1+j) times (1−j)]
    = (0, 0)                   [gives the origin]

    So: Euclid II.5 at `a = b = 1`. Both factors are non-zero; the product is not. A field
    cannot do this. This is `AnyRing.ringMul_conj` at `R = ℝ`, `σ = 1`, used directly:
    `nullMinus` is the generic conjugate of `nullPlus`. -/
theorem nullPair_mul_eq_zero : mul nullPlus nullMinus = (0, 0) := by
  have hn : AnyRing.ringNormForm 1 nullPlus = 0 := normForm_nullPlus
  have h := AnyRing.ringMul_conj 1 nullPlus
  rw [hn] at h
  simpa only [mul, nullPlus, nullMinus] using h

/-- `1 + j` has no inverse.

    ¬ ∃ w : SC              [there is no point w]
    mul nullPlus w = one    [with (1+j) times w equal to 1]

    So: `AnyRing.exists_inverse_iff_isUnit_ringNormForm` at `R = ℝ`, `σ = 1` reads this as
    the size `0` not being a unit. -/
theorem nullPlus_not_invertible : ¬ ∃ w : SC, mul nullPlus w = one := by
  rw [show (∃ w : SC, mul nullPlus w = one) ↔ IsUnit (normForm nullPlus) from
    AnyRing.exists_inverse_iff_isUnit_ringNormForm 1 nullPlus]
  rw [normForm_nullPlus]
  exact not_isUnit_zero

/-- The two size-zero points pull on each other with strength 2.

    form nullPlus nullMinus     [pair up (1+j) and (1−j)]
    = 2                         [the pull is 2]

    So: both are size zero, yet the pull between them is not zero. That makes the plane
    they span non-degenerate, signature `(1,1)`. -/
theorem form_nullPair : form nullPlus nullMinus = 2 := by
  norm_num [form, nullPlus, nullMinus]

/-- `1 + j` pulls on itself with strength zero.

    form nullPlus nullPlus = 0     [pairing (1,1) with itself gives 0] -/
theorem form_nullPlus_self : form nullPlus nullPlus = 0 := by
  change AnyRing.ringForm 1 nullPlus nullPlus = 0
  rw [← AnyRing.ringForm_self 1 nullPlus]
  exact normForm_nullPlus

/-- `1 − j` pulls on itself with strength zero.

    form nullMinus nullMinus = 0     [pairing (1,−1) with itself gives 0] -/
theorem form_nullMinus_self : form nullMinus nullMinus = 0 := by
  change AnyRing.ringForm 1 nullMinus nullMinus = 0
  rw [← AnyRing.ringForm_self 1 nullMinus]
  exact normForm_nullMinus

/-- Every point is built from the two size-zero directions.

    (z : SC)                             [any point]
    ∃ a b : ℝ                            [there are two real numbers]
    z = a • nullPlus + b • nullMinus     [stretching and adding the two gives z]

    So: light-cone coordinates. `a = (z.1 + z.2)/2` and `b = (z.1 − z.2)/2`. -/
theorem eq_nullPair_combination (z : SC) :
    ∃ a b : ℝ, z = a • nullPlus + b • nullMinus :=
  ⟨(z.1 + z.2) / 2, (z.1 - z.2) / 2, by
    simp only [nullPlus, nullMinus, Prod.smul_mk, smul_eq_mul, Prod.mk_add_mk, mul_one,
      mul_neg, Prod.ext_iff]
    constructor <;> ring⟩

/-- The general size-zero points at `σ = 1` are these two.

    ((Real.sqrt 1, 1) : Cell) = nullPlus       [the general first one is (1,1)]
    ((Real.sqrt 1, -1) : Cell) = nullMinus     [the general second one is (1,−1)]
    ∧                                          [both hold]

    So: the pair the general theorem produces at `σ = 1` is this pair, not a copy of it. -/
theorem nullGen_eq_nullPair :
    ((Real.sqrt 1, 1) : Cell) = nullPlus ∧ ((Real.sqrt 1, -1) : Cell) = nullMinus := by
  refine ⟨?_, ?_⟩ <;> simp [nullPlus, nullMinus, Real.sqrt_one]

/-- `σ = 1` has a non-degenerate light cone.

    HasNullPair 1     [two size-zero directions in non-zero tension, at σ = 1]

    So: witnessed by `nullPlus` and `nullMinus` at strength 2. -/
theorem hasNullPair_one : HasNullPair 1 :=
  ⟨nullPlus, nullMinus,
   by rw [normForm_eq_normForm_one, normForm_nullPlus],
   by rw [normForm_eq_normForm_one, normForm_nullMinus],
   by rw [form_eq_form_one, form_nullPair]; norm_num⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE CONTRAST: ℂ AT σ = −1
-- ═══════════════════════════════════════════════════════════════════

/-- The size of a complex number is never negative.

    (z : ℂ)                     [any complex number]
    0 ≤ Complex.normSq z        [its squared size is zero or above]

    So: contrast `normForm_indefinite`. -/
theorem complex_normSq_definite (z : ℂ) : 0 ≤ Complex.normSq z := Complex.normSq_nonneg z

/-- In `ℂ` only the origin has size zero.

    (z : ℂ)                       [any complex number]
    Complex.normSq z = 0          [its squared size is zero]
    ↔                             [exactly when]
    z = 0                         [it is the origin]

    So: contrast `nullPlus_ne_zero`. There is no light cone in `ℂ`. -/
theorem complex_only_origin_null (z : ℂ) : Complex.normSq z = 0 ↔ z = 0 :=
  Complex.normSq_eq_zero

/-- In `ℂ` a product reaches zero only if a factor was zero.

    (z w : ℂ)                  [two complex numbers]
    z * w = 0                  [their product is zero]
    z = 0 ∨ w = 0              [then one of them was zero]

    So: contrast `nullPair_mul_eq_zero`. -/
theorem complex_no_zeroDivisors (z w : ℂ) (h : z * w = 0) : z = 0 ∨ w = 0 :=
  mul_eq_zero.mp h

/-- In `ℂ` every non-zero number can be divided by.

    (z : ℂ)          [any complex number]
    (hz : z ≠ 0)     [which is not zero]
    IsUnit z         [has something that multiplies with it to give 1]

    So: contrast `nullPlus_not_invertible`. -/
theorem complex_invertible (z : ℂ) (hz : z ≠ 0) : IsUnit z :=
  isUnit_iff_ne_zero.mpr hz

-- ═══════════════════════════════════════════════════════════════════
-- σ = 1, BY CLASSIFICATION
-- ═══════════════════════════════════════════════════════════════════

/-- This algebra has exactly two independent directions.

    Module.finrank ℝ SC = 2     [counting directions over the reals gives 2] -/
theorem dim_two : Module.finrank ℝ SC = 2 := by
  rw [Module.finrank_prod, Module.finrank_self]

/-- This algebra's σ is `1`, and the classification proves it.

    ∃ e : SC ≃ₗ[ℝ] ℝ × ℝ                       [the algebra can be relabelled as pairs]
    e 1 = (1, 0)                               [the relabelling sends 1 to (1,0)]
    ∧                                          [and, on top of that]
    ∀ z w : SC                                 [for every two points]
    e (z * w) = Trichotomy.mul 1 (e z) (e w)   [multiply then relabel, same as relabel
                                                then multiply with σ set to 1]

    So: run `classification` on this carrier. It returns some setting in `{−1, 0, 1}`. The
    two channels `(1,0)` and `(0,1)` multiply to zero while neither is zero, so the setting
    is not `−1`. No non-zero point of `ℝ × ℝ` squares to zero, so it is not `0`. Hence `1`.
    The setting is a consequence of the classification, not of how the algebra was written
    down. -/
theorem disc_eq_one :
    ∃ e : SC ≃ₗ[ℝ] ℝ × ℝ, e 1 = (1, 0) ∧
      ∀ z w : SC, e (z * w) = Trichotomy.mul 1 (e z) (e w) := by
  obtain ⟨σ, hσ, e, he1, hemul⟩ := classification (A := SC) dim_two
  have hne : ∀ x : SC, x ≠ 0 → e x ≠ 0 := fun x hx hc => hx (by
    have := congrArg e.symm hc; simpa using this)
  have hz : ((1, 0) : SC) ≠ 0 := by simp [Prod.ext_iff]
  have hw : ((0, 1) : SC) ≠ 0 := by simp [Prod.ext_iff]
  have hprod : ((1, 0) : SC) * ((0, 1) : SC) = 0 := by simp [Prod.ext_iff]
  have hnonneg : (0 : ℝ) ≤ σ := by
    refine (hasZeroDivisor_iff_disc_nonneg σ).mp ⟨e (1, 0), e (0, 1), hne _ hz, hne _ hw, ?_⟩
    rw [← hemul, hprod, map_zero]
  have hne_zero : σ ≠ 0 := by
    intro h0
    set x : SC := e.symm (0, 1) with hx
    have hxne : x ≠ 0 := fun hc => by
      have : ((0, 1) : ℝ × ℝ) = 0 := by rw [← e.apply_symm_apply (0, 1), ← hx, hc, map_zero]
      simp [Prod.ext_iff] at this
    have hsq : x * x = 0 := by
      have := hemul x x
      rw [hx, e.apply_symm_apply, h0] at this
      have hc : Trichotomy.mul 0 ((0 : ℝ), (1 : ℝ)) (0, 1) = 0 := by
        simp [Trichotomy.mul, Prod.ext_iff]
      rw [hc] at this
      simpa using congrArg e.symm this
    apply hxne
    have h1 : x.1 * x.1 = 0 := congrArg Prod.fst hsq
    have h2 : x.2 * x.2 = 0 := congrArg Prod.snd hsq
    have e1 : x.1 = 0 := by nlinarith
    have e2 : x.2 = 0 := by nlinarith
    exact Prod.ext e1 e2
  rcases hσ with h | h | h
  · exact absurd (h ▸ hnonneg) (by norm_num)
  · exact absurd h hne_zero
  · exact ⟨e, he1, h ▸ hemul⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO IDEMPOTENTS
-- ═══════════════════════════════════════════════════════════════════

/-- Half of `1 + j`.

    ePlus                        [the name for this point]
    ((1 : ℝ) / 2, (1 : ℝ) / 2)   [one half in each slot]
    : SC                         [a point of this algebra] -/
def ePlus : SC := ((1 : ℝ) / 2, (1 : ℝ) / 2)

/-- Half of `1 − j`.

    eMinus                          [the name for this point]
    ((1 : ℝ) / 2, -((1 : ℝ) / 2))   [one half, then minus one half]
    : SC                            [a point of this algebra] -/
def eMinus : SC := ((1 : ℝ) / 2, -((1 : ℝ) / 2))

/-- `e₊` times itself is `e₊`.

    mul ePlus ePlus = ePlus     [squaring this point returns it unchanged]

    So: a solution of Boole's `x² = x` other than `0` and `1`. By
    `exists_nontrivial_idempotent_iff_disc_pos`, such solutions exist only above zero. -/
theorem ePlus_idem : mul ePlus ePlus = ePlus := by
  simp only [mul, ePlus, Prod.ext_iff]; constructor <;> ring

/-- `e₋` times itself is `e₋`.

    mul eMinus eMinus = eMinus     [squaring this point returns it unchanged] -/
theorem eMinus_idem : mul eMinus eMinus = eMinus := by
  simp only [mul, eMinus, Prod.ext_iff]; constructor <;> ring

/-- `e₊` times `e₋` is the origin.

    mul ePlus eMinus = (0, 0)     [multiplying the two gives the origin]

    So: the two are independent channels. Nothing in one reaches the other. -/
theorem e_orthogonal : mul ePlus eMinus = (0, 0) := by
  simp only [mul, ePlus, eMinus, Prod.ext_iff]; constructor <;> ring

/-- `e₊` plus `e₋` is 1.

    ePlus + eMinus = one     [the two halves add to the unit]

    So: with idempotence and orthogonality, this is Peirce's resolution of the unit, and it
    is the isomorphism `ℝ[j]/(j²−1) ≅ ℝ × ℝ`. -/
theorem e_add_eq_one : ePlus + eMinus = one := by
  simp only [ePlus, eMinus, one, Prod.mk_add_mk, Prod.ext_iff]
  constructor <;> ring

/-- The general idempotent at `σ = 1` is `e₊`.

    idemPlus 1        [the general witness, σ set to 1]
    = ePlus           [is this file's e₊]

    So: `√1 = 1`, so the second slot is `1/2`. The point the general theorem produces at
    `σ = 1` is this one, not a copy of it. -/
theorem idemPlus_one_eq_ePlus : idemPlus 1 = ePlus := by
  simp [idemPlus, ePlus, Real.sqrt_one]

/-- The general idempotent at `σ = 1`, mirrored, is `e₋`.

    idemMinus 1        [the general mirrored witness, σ set to 1]
    = eMinus           [is this file's e₋] -/
theorem idemMinus_one_eq_eMinus : idemMinus 1 = eMinus := by
  simp [idemMinus, eMinus, Real.sqrt_one]

/-- `e₊` has size zero.

    normForm ePlus = 0     [measuring this point gives 0]

    So: the idempotents sit on the light cone, which is why they multiply to zero. -/
theorem normForm_ePlus : normForm ePlus = 0 := by norm_num [normForm, ePlus]

/-- `e₋` has size zero.

    normForm eMinus = 0     [measuring this point gives 0] -/
theorem normForm_eMinus : normForm eMinus = 0 := by norm_num [normForm, eMinus]

end SplitComplex
end Trichotomy
end Apeiron

end
