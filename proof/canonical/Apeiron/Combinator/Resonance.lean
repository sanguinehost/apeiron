import Apeiron.Combinator.Basic
import Apeiron.Combinator.Confluence

/-!
# Structural subterm hoisting and resonance cancellation

@name Resonance
@cite curry-1930
@cite barendregt-1984
@grade proved

Resonance analysis identifies structural subterms whose shape recurs across an expression.
Cancelling this redundancy consists of hoisting the recurring subterm `s` out of term `T`:
abstracting a fresh variable over `T` and applying the resulting abstraction to `s`.

This file proves the algebraic soundness of the hoist transformation:
1. `hoist_reduces`: `app (bracket i T) s` reduces to `subst i T s`.
2. `hoist_preserves_semantics`: when `subst i body s = T`, `hoist i body s` reduces to `T`.
3. `hoist_normalForm_unique`: by confluence, the normal form of the hoisted term equals
   the normal form of the original term.
-/

namespace Apeiron
namespace Combinator
namespace Comb

/-- The structural hoist of recurring subterm `s` at variable index `i` in term `body`:
    abstracts `i` from `body` and applies it to `s`.

    (i : ℕ) (body s : Comb)         [the variable, the term, and the recurring subterm]
    Comb                            [the abstracted term applied to the subterm] -/
def hoist (i : ℕ) (body : Comb) (s : Comb) : Comb :=
  app (bracket i body) s

/-- Hoisting reduces to substitution of the recurring subterm.

    (i : ℕ) (body s : Comb)                     [the variable, the term and the subterm]
    Reduces (hoist i body s) (subst i body s)   [the hoisted term reaches the substituted one] -/
theorem hoist_reduces (i : ℕ) (body s : Comb) :
    Reduces (hoist i body s) (subst i body s) :=
  bracket_correct i body s

/-- Semantic preservation: if `body` has variable `i` at occurrences of `s`
    such that substituting `s` into `body` recovers original term `T`,
    then `hoist i body s` reduces to `T`.

    {i : ℕ} {body s T : Comb}       [the variable, the term, the subterm and the recovered term]
    (h : subst i body s = T)        [substituting the subterm recovers the recovered term]
    Reduces (hoist i body s) T      [so the hoisted term reaches it] -/
theorem hoist_preserves_semantics {i : ℕ} {body s T : Comb} (h : subst i body s = T) :
    Reduces (hoist i body s) T := by
  have hr := hoist_reduces i body s
  rw [h] at hr
  exact hr

/-- Normal form equivalence: if the hoisted term reduces to a normal form `N`
    and the unhoisted original term `T` reduces to a normal form `N'`,
    confluence guarantees `N = N'`.

    {i : ℕ} {body s T N N' : Comb}             [the variable, terms and two normal forms]
    (h : subst i body s = T)                   [substituting the subterm recovers the original]
    (hn : NormalForm N)                        [the first form has no rule left to fire]
    (hn' : NormalForm N')                      [and so does the second]
    (hred_hoist : Reduces (hoist i body s) N)  [the hoisted term reaches the first]
    (hred_orig : Reduces T N')                 [the original reaches the second]
    N = N'                                     [so the two normal forms are the same] -/
theorem hoist_normalForm_unique {i : ℕ} {body s T N N' : Comb}
    (h : subst i body s = T)
    (hn : NormalForm N) (hn' : NormalForm N')
    (hred_hoist : Reduces (hoist i body s) N)
    (hred_orig : Reduces T N') :
    N = N' := by
  have hred_t : Reduces (hoist i body s) T := hoist_preserves_semantics h
  have hred_both : Reduces (hoist i body s) N' := Reduces.trans hred_t hred_orig
  exact normalForm_unique hred_hoist hred_both hn hn'

end Comb
end Combinator
end Apeiron
