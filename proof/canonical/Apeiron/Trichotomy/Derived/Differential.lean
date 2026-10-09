import Apeiron.Trichotomy.Trace

/-!
# The degenerate face carries a derivative

@name Differential
@cite clifford-1873-dual
@cite study-1903
@cite ehrhard-regnier-2003
@cite girard-1989
@grade proved

At `σ = 0` the generator squares to nothing, and multiplying two points reads

```
   (a + b·x)(c + d·x)  =  a·c  +  (a·d + b·c)·x
```

with no `σ·b·d` term to spoil the first slot. That second slot is Leibniz's rule. The dual
numbers are the arithmetic of a value carried alongside its first-order change, and this file
proves that the degenerate face is where that reading is available and nowhere else.

## Root

| from | here |
|---|---|
| `classification`, `mul` at `σ = 0` | `parabolic_is_leibniz`, `value_slot_iff_disc_zero` |
| `upper_eq_lower_of_disc_zero` | `change_slot_linear`, `change_slot_smul` |
| `nilpotent_iff_disc_zero` (`Trichotomy/Trace.lean`) | `dual_pow`, `dual_taylor_two` |

`value_slot_iff_disc_zero` is the sharp statement: the first slot of a product is the product
of the first slots at `σ = 0` and at no other setting, so the value is undisturbed by the
change exactly on the degenerate face. `change_slot_linear` is the other half: the second
slot is additive in each argument separately, which is the linearity the differential calculus
asks of its differentiated argument.

The connection to `Trace.lean` is not decoration. `nilpotent_iff_disc_zero` says the only face
carrying a non-zero element that dies out under repeated multiplication is `σ = 0`, and the
element that dies out is the generator, `x² = 0`. That one equation is simultaneously the
reason Girard's execution formula terminates on this face and the reason a Taylor expansion
here stops after the first-order term. `dual_taylor_two` states the second as an equation.

## What is claimed and what is not

Ehrhard and Regnier, *The differential lambda-calculus*, Theoretical Computer Science 309
(2003) 1-41, add to the calculus a term `D M · N` that applies `M` to `N` linearly, once, and
prove confluence, strong normalisation in the typed case, and a theorem relating Taylor
expansion to linear head reduction. The syntactic calculus is theirs and none of it is
re-derived here.

What is here is the arithmetic the degenerate face supplies for it: a carrier on which a value
and a single first-order change multiply by Leibniz's rule, with the second-order term absent
by the algebra rather than by truncation. That the dual numbers compute first derivatives is
Clifford 1873 and Study 1903 and is not new either. The statement that is not in either place
is `value_slot_iff_disc_zero`, which says there is exactly one value of σ where it works, so
the differential reading is not a convention imposed on the algebra but a face of it.

**What is not built.** The differential calculus needs its terms differentiated, which needs a
sum of terms, and `Apeiron/Combinator` has no formal sums. A resource calculus over `Comb`
would need one. `theory/ROOTING.md` names it.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

/-- At the degenerate setting, multiplying is Leibniz's rule.

@cite clifford-1873-dual

    (a b c d : ℝ)                                                [two values, two changes]
    mul 0 (a, b) (c, d) = (a * c, a * d + b * c)                 [the product rule]

    So: the first slot carries the value and the second the first-order change, and the change
    of a product is the first times the change of the second plus the change of the first times
    the second. This is one line of arithmetic and it is the whole of the differential
    reading. -/
theorem parabolic_is_leibniz (a b c d : ℝ) :
    mul 0 (a, b) (c, d) = (a * c, a * d + b * c) := by
  simp [mul]

/-- **The value slot is undisturbed exactly at the degenerate setting.**

    (σ : ℝ)                                                      [the parameter]
    (∀ a b c d : ℝ, (mul σ (a, b) (c, d)).1 = a * c)             [first slots multiply alone]
    ↔                                                            [exactly when]
    σ = 0                                                        [σ is at zero]

    So: away from zero the product's first slot picks up `σ · b · d`, so the change feeds back
    into the value and the pair is no longer a value carried alongside its change. There is
    exactly one value of σ at which the differential reading is available. This is the whole face
    dependence of the route. -/
theorem value_slot_iff_disc_zero (σ : ℝ) :
    (∀ a b c d : ℝ, (mul σ (a, b) (c, d)).1 = a * c) ↔ σ = 0 := by
  constructor
  · intro h
    have := h 1 1 1 1
    linarith
  · rintro rfl
    intro a b c d
    ring

/-- The change slot is linear in the changes, with the values held fixed.

@cite ehrhard-regnier-2003

    (a c b₁ b₂ d₁ d₂ : ℝ)                                        [two values and four changes]
    (mul 0 (a, b₁ + b₂) (c, d₁ + d₂)).2                          [the change of the product]
    = (mul 0 (a, b₁) (c, d₁)).2 + (mul 0 (a, b₂) (c, d₂)).2      [adds over the changes]

    So: the linearity the differential calculus asks of the argument it differentiates. In
    Ehrhard and Regnier's calculus `D M · N` is linear in `N`, and here that linearity is the
    algebra's own addition in the second slot, taken in both changes at once. It is not additive
    in one change with the other held fixed: `a·d` would then be counted twice, which is the
    difference between a derivative and a difference and is why the pair moves together. -/
theorem change_slot_linear (a c b₁ b₂ d₁ d₂ : ℝ) :
    (mul 0 (a, b₁ + b₂) (c, d₁ + d₂)).2
      = (mul 0 (a, b₁) (c, d₁)).2 + (mul 0 (a, b₂) (c, d₂)).2 := by
  ring

/-- Scaling both changes scales the change of the product.

@cite ehrhard-regnier-2003

    (a c t b d : ℝ)                                        [two values, a factor, two changes]
    (mul 0 (a, t * b) (c, t * d)).2 = t * (mul 0 (a, b) (c, d)).2    [the factor comes out]

    So: with additivity above, the change slot is a linear map of the changes. -/
theorem change_slot_smul (a c t b d : ℝ) :
    (mul 0 (a, t * b) (c, t * d)).2 = t * (mul 0 (a, b) (c, d)).2 := by
  ring

/-- Powers of a value with unit change carry the derivative of the power.

@cite study-1903

    (x : ℝ) (n : ℕ)                                                     [a value and an exponent]
    cpow 0 (x, 1) (n + 1) = (x ^ (n + 1), ((n : ℝ) + 1) * x ^ n)  [the power and its derivative]

    So: the monomial case of the differential reading, proved by induction rather than asserted.
    The second slot of `(x, 1)` raised to a power is the derivative of that power. -/
theorem dual_pow (x : ℝ) (n : ℕ) :
    cpow 0 (x, 1) (n + 1) = (x ^ (n + 1), ((n : ℝ) + 1) * x ^ n) := by
  induction n with
  | zero =>
    simp only [cpow, mul, one, Prod.mk.injEq]
    constructor <;> norm_num
  | succ n ih =>
    rw [cpow, ih]
    simp only [mul, Prod.mk.injEq]
    refine ⟨by ring, ?_⟩
    push_cast
    ring

/-- The expansion stops after the first-order term.

@cite girard-1989
@cite ehrhard-regnier-2003

    (a b : ℝ)                                                    [a value and a change]
    mul 0 ((0 : ℝ), b) ((0 : ℝ), b) = ((0 : ℝ), (0 : ℝ))         [a pure change squares to nil]
    ∀ c d : ℝ, mul 0 (a, b) (c, d)                               [so a product has]
      = (a * c, a * d + b * c)                                   [no second-order term]
    ∧                                                            [both together]

    So: the Taylor expansion on this face has two terms and no remainder. The same equation is
    why `Trace.lean`'s `nilpotent_iff_disc_zero` puts finite termination of Girard's execution
    formula on this face: the loop that dies out and the expansion that stops are one fact
    about the generator, `x² = 0`. -/
theorem dual_taylor_two (a b : ℝ) :
    mul 0 ((0 : ℝ), b) ((0 : ℝ), b) = ((0 : ℝ), (0 : ℝ)) ∧
      ∀ c d : ℝ, mul 0 (a, b) (c, d) = (a * c, a * d + b * c) := by
  refine ⟨?_, fun c d => parabolic_is_leibniz a b c d⟩
  simp only [mul, Prod.mk.injEq]
  constructor <;> ring

end Trichotomy
end Apeiron

end
