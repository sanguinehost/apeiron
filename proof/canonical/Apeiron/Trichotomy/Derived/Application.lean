import Apeiron.Trichotomy.Channels

/-!
# Channel ratios and quotient trees

@name Application
@cite cockle-1849
@cite schonfinkel-1924
@cite palmieri-2026
@cite mathlib-sqrt
@grade proved

For `σ > 0`, `upper` and `lower` are the two real characters classified in `Channels`.
Define `ratio σ z = upper σ z / lower σ z`. Unlike `mul`, this operation is sensitive to
the order and bracketing of encoded operands:

- order matters: packing `2` over `1` reads as `2`, packing `1` over `2` reads as a half
  (`ratio_order`);
- bracketing matters: `(8/4)/2` and `8/(4/2)` pack to different points (`ratio_bracket`).

Mathlib defines division by zero as zero, making `ratio` total. The theorem `ratio_null`
records this case. For `σ ≥ 0`, `normForm_eq_upper_mul_lower` identifies its singular locus
with one component of the null cone.

## Recursive quotient syntax

`Tower` is the free binary tree with functions `ℝ → ℝ` at its leaves. At a node, `state`
evaluates both subtrees at the same input, encodes the results with `ofHalves`, and applies
`ratio`.

For `σ > 0`:

- `state_const` evaluates a constant numerator over denominator `1`.
- `state_node` reduces node evaluation to division of subtree evaluations.
- `node_faithful_iff_disc_pos` states that this reduction holds for every node exactly when
  `σ > 0`. For `σ ≤ 0`, every node evaluates to `0` or `1`.

At and below zero these statements read `Real.sqrt σ` as `0`, which is Mathlib's
convention for the square root of a nonpositive number. The convention-free statement for
the face below zero is `no_channel_of_disc_neg`: there are no channel reads there at all.

## What this does not say

`Tower` is not a combinator calculus, and this file proves no embedding of combinator terms
into `Cell`. The combinator calculus is defined separately in `Combinator.Basic`.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

/-- Quotient of the two channel coordinates.

    (σ : ℝ)                        [the parameter]
    (z : Cell)                     [the point]
    upper σ z / lower σ z          [first coordinate divided by second]
    : ℝ                            [a plain real number]

    Division by zero follows Mathlib's totalized field convention. -/
def ratio (σ : ℝ) (z : Cell) : ℝ := upper σ z / lower σ z

/-- On the second arm of the null cone the fraction reads zero.

    {σ : ℝ}                  [the parameter]
    (z : Cell)               [the point]
    lower σ z = 0            [its second coordinate is zero]
    →                        [then]
    ratio σ z = 0            [the fraction reads zero]

    This states the zero-denominator case explicitly. -/
theorem ratio_null {σ : ℝ} (z : Cell) (h : lower σ z = 0) : ratio σ z = 0 := by
  simp only [ratio, h, div_zero]

/-- Building a point from two coordinates and reading the fraction divides them.

    {σ : ℝ} (hσ : 0 < σ)                     [the parameter, above zero]
    (u v : ℝ)                                [two numbers]
    ratio σ (ofHalves σ u v) = u / v         [pack them, read the fraction: u over v]

    This follows from the inverse coordinate equations for `ofHalves`. -/
theorem ratio_ofHalves {σ : ℝ} (hσ : 0 < σ) (u v : ℝ) :
    ratio σ (ofHalves σ u v) = u / v := by
  simp only [ratio, upper_ofHalves hσ, lower_ofHalves hσ]

/-- The fraction of a product is the product of the fractions.

    {σ : ℝ} (hσ : 0 ≤ σ)                             [the parameter, at zero or above]
    (z w : Cell)                                     [two points]
    ratio σ (mul σ z w) = ratio σ z * ratio σ w      [multiply then read, or read then multiply]

    This follows from multiplicativity of both channels. -/
theorem ratio_mul {σ : ℝ} (hσ : 0 ≤ σ) (z w : Cell) :
    ratio σ (mul σ z w) = ratio σ z * ratio σ w := by
  simp only [ratio, upper_mul hσ, lower_mul hσ]
  exact (div_mul_div_comm _ _ _ _).symm

/-- The order of the two channels matters to the fraction.

    {σ : ℝ} (hσ : 0 < σ)              [the parameter, above zero]
    ratio σ (ofHalves σ 2 1)          [2 packed over 1 reads as 2]
    ≠                                 [is not equal to]
    ratio σ (ofHalves σ 1 2)          [1 packed over 2 reads as a half]

    The displayed values provide a counterexample to commutativity. -/
theorem ratio_order {σ : ℝ} (hσ : 0 < σ) :
    ratio σ (ofHalves σ 2 1) ≠ ratio σ (ofHalves σ 1 2) := by
  rw [ratio_ofHalves hσ, ratio_ofHalves hσ]
  norm_num

/-- The bracketing of packed points matters to the fraction.

    {σ : ℝ} (hσ : 0 < σ)                                [the parameter, above zero]
    ratio σ (ofHalves σ (ratio σ (ofHalves σ 8 4)) 2)   [(8 over 4) over 2 reads as 1]
    ≠                                                   [is not equal to]
    ratio σ (ofHalves σ 8 (ratio σ (ofHalves σ 4 2)))   [8 over (4 over 2) reads as 4]

    The displayed values provide a counterexample to associativity. -/
theorem ratio_bracket {σ : ℝ} (hσ : 0 < σ) :
    ratio σ (ofHalves σ (ratio σ (ofHalves σ 8 4)) 2) ≠
      ratio σ (ofHalves σ 8 (ratio σ (ofHalves σ 4 2))) := by
  simp only [ratio_ofHalves hσ]
  norm_num

/-- At zero or below, packing and reading forgets what was packed.

    {σ : ℝ} (hσ : σ ≤ 0)                  [the parameter, at zero or below]
    (u v : ℝ)                             [two numbers]
    ratio σ (ofHalves σ u v) = 1          [the read is one]
    ∨                                     [or]
    ratio σ (ofHalves σ u v) = 0          [the read is zero]

    For nonpositive `σ`, `Real.sqrt σ = 0`; both coordinates equal `(u + v) / 2`, so the
    totalized quotient is `1` when nonzero and `0` otherwise. -/
theorem ratio_ofHalves_of_disc_nonpos {σ : ℝ} (hσ : σ ≤ 0) (u v : ℝ) :
    ratio σ (ofHalves σ u v) = 1 ∨ ratio σ (ofHalves σ u v) = 0 := by
  have hs : Real.sqrt σ = 0 := Real.sqrt_eq_zero_of_nonpos hσ
  have hu : upper σ (ofHalves σ u v) = (u + v) / 2 := by
    simp [upper, ofHalves, hs]
  have hl : lower σ (ofHalves σ u v) = (u + v) / 2 := by
    simp [lower, ofHalves, hs]
  by_cases h : (u + v) / 2 = 0
  · right
    simp only [ratio, hu, hl, h, div_zero]
  · left
    simp only [ratio, hu, hl]
    exact div_self h

-- ═══════════════════════════════════════════════════════════════════
-- RECURSIVE QUOTIENT SYNTAX
-- ═══════════════════════════════════════════════════════════════════

/-- A binary tree with a real function at each leaf.

    Tower                        [the type of trees]
    leaf : (ℝ → ℝ) → Tower       [a single function is a tree]
    node : Tower → Tower → Tower [two trees packed into the two channels of one point] -/
inductive Tower : Type where
  /-- A single function. -/
  | leaf : (ℝ → ℝ) → Tower
  /-- Two subtrees, to be packed into the two channels of one point. -/
  | node : Tower → Tower → Tower

namespace Tower

/-- Evaluation by recursive quotient composition.

    (σ : ℝ)              [the parameter]
    Tower                [a tree]
    → ℝ → ℝ              [what it computes, as a function of one number]

    A node evaluates both children at the same input and applies `ratio` to their
    `ofHalves` encoding. -/
def state (σ : ℝ) : Tower → ℝ → ℝ
  | leaf f => f
  | node t s => fun x => ratio σ (ofHalves σ (state σ t x) (state σ s x))

/-- A leaf computes its own function.

    (σ : ℝ)                     [the parameter]
    (f : ℝ → ℝ)                 [a function]
    state σ (leaf f) = f        [the tree holding only f computes f] -/
theorem state_leaf (σ : ℝ) (f : ℝ → ℝ) : state σ (leaf f) = f := rfl

/-- Above zero, a branch computes the quotient of its subtrees, at one shared input.

    {σ : ℝ} (hσ : 0 < σ)         [the parameter, above zero]
    (t s : Tower)                [two trees]
    (x : ℝ)                      [one input]
    state σ (node t s) x         [what the packed point reads]
    = state σ t x / state σ s x  [the first tree's value over the second's, both at x]

    Both subtree evaluations use the same input `x`. -/
theorem state_node {σ : ℝ} (hσ : 0 < σ) (t s : Tower) (x : ℝ) :
    state σ (node t s) x = state σ t x / state σ s x := by
  simp only [state]
  exact ratio_ofHalves hσ _ _

/-- Above zero, a branch holding a constant over one computes that constant.

    {σ : ℝ} (hσ : 0 < σ)                                 [the parameter, above zero]
    (c x : ℝ)                                            [a constant, and any input]
    state σ (node (leaf fun _ => c) (leaf fun _ => 1)) x [pack the constant over 1, read]
    = c                                                  [the constant comes back]

    The numerator is the constant function and the denominator is `1`. -/
theorem state_const {σ : ℝ} (hσ : 0 < σ) (c x : ℝ) :
    state σ (node (leaf fun _ => c) (leaf fun _ => 1)) x = c := by
  rw [state_node hσ]
  show c / 1 = c
  exact div_one c

/-- Above zero, recursion can sit in the first channel.

    {σ : ℝ} (hσ : 0 < σ)                          [the parameter, above zero]
    (t : Tower)                                   [the inner tree]
    (g : ℝ → ℝ)                                   [the outer function]
    (x : ℝ)                                       [one input]
    state σ (node t (leaf g)) x                   [inner tree packed over the function]
    = state σ t x / g x                           [reads as inner value over outer value] -/
theorem state_node_left {σ : ℝ} (hσ : 0 < σ) (t : Tower) (g : ℝ → ℝ) (x : ℝ) :
    state σ (node t (leaf g)) x = state σ t x / g x :=
  state_node hσ t (leaf g) x

/-- Above zero, recursion can sit in the second channel.

    {σ : ℝ} (hσ : 0 < σ)                          [the parameter, above zero]
    (f : ℝ → ℝ)                                   [the outer function]
    (t : Tower)                                   [the inner tree]
    (x : ℝ)                                       [one input]
    state σ (node (leaf f) t) x                   [the function packed over the inner tree]
    = f x / state σ t x                           [reads as outer value over inner value]

    This is the right-recursive counterpart of `state_node_left`. -/
theorem state_node_right {σ : ℝ} (hσ : 0 < σ) (f : ℝ → ℝ) (t : Tower) (x : ℝ) :
    state σ (node (leaf f) t) x = f x / state σ t x :=
  state_node hσ (leaf f) t x

/-- At zero or below, every branch reads zero or one.

    {σ : ℝ} (hσ : σ ≤ 0)             [the parameter, at zero or below]
    (t s : Tower)                    [two trees]
    (x : ℝ)                          [any input]
    state σ (node t s) x = 1         [the branch reads one]
    ∨                                [or]
    state σ (node t s) x = 0         [the branch reads zero]

    This is the node-level specialization of `ratio_ofHalves_of_disc_nonpos`. -/
theorem state_node_of_disc_nonpos {σ : ℝ} (hσ : σ ≤ 0) (t s : Tower) (x : ℝ) :
    state σ (node t s) x = 1 ∨ state σ (node t s) x = 0 := by
  simp only [state]
  exact ratio_ofHalves_of_disc_nonpos hσ _ _

/-- Branches read faithfully exactly when sigma is positive.

    (σ : ℝ)                                 [the parameter]
    ∀ t s : Tower, ∀ x : ℝ                  [for every branch and every input]
    state σ (node t s) x                    [what the branch reads]
    = state σ t x / state σ s x             [is the quotient of what its subtrees compute]
    ↔                                       [exactly when]
    0 < σ                                   [sigma is positive]

    Sufficiency is `state_node`; necessity uses constant leaves to exclude `σ ≤ 0`. -/
theorem node_faithful_iff_disc_pos (σ : ℝ) :
    (∀ t s : Tower, ∀ x : ℝ, state σ (node t s) x = state σ t x / state σ s x) ↔
      0 < σ := by
  constructor
  · intro h
    by_contra hσ
    push Not at hσ
    have h2 := h (leaf fun _ => 2) (leaf fun _ => 1) 0
    rcases state_node_of_disc_nonpos hσ (leaf fun _ => 2) (leaf fun _ => 1) 0 with h1 | h0
    · rw [h2] at h1
      norm_num [state] at h1
    · rw [h2] at h0
      norm_num [state] at h0
  · intro hσ t s x
    exact state_node hσ t s x

end Tower

end Trichotomy
end Apeiron

end
