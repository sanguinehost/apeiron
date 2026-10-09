import Apeiron.Linter.Notation
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.List.Perm.Basic
import Mathlib.Tactic

/-!
# The exact numerics seam: reductions and additive fusion

@name Numeric Generic
@grade proved

This file is the exact algebra of the numerics seam, separated from the real exponential, square
root, relative-error and ordered-calibration layers in `Apeiron/Numeric/Accumulator.lean`. It
states each fact over the weakest structure that carries it, so `ℝ` and `ℤ` are instances of the
ordered reductions and `ZMod (2 ^ 64)` is an instance of the order-free fusion and calibration
identity, rather than separate restatements.

**Sized reductions.** A reduction over `n` terms each bounded by `M` is bounded by `n · M`
(`dot_abs_bounded`), over any commutative ordered ring. Choosing an integer width whose capacity
clears that bound is what makes the reduction never wrap (`accumulator_fits`). No rounding is
involved: overflow is a violation of the premise, not an error inside the model.

**Additive bookkeeping and fusion.** The rounded value plus its own dropped residue reconstructs
the exact sum in any additive commutative group (`lossy_plus_carry_is_exact`), and the composed
deviation of two merges is bounded by the sum of the two per-op residues in any commutative
ordered ring (`lossy_error_is_additive`). A list's total under an additive commutative monoid is
independent of order (`foldTotal_perm`) and of contiguous chunking (`foldTotal_chunks`), which is
the exact content the tile and online-softmax schedule arguments instantiate.

**The calibration identity.** `functionDistortion` and `output_error_is_function_distortion` are
polynomial identities over any commutative semiring. The order, the square root and the relative
error model they are read through are instance hypotheses in `Accumulator`.

## Root

This file instantiates nothing from `Apeiron/Trichotomy`. Every statement is over the weakest
setting that carries it, a commutative ordered ring, an additive commutative group or monoid,
or a commutative semiring, and no declaration mentions `Cell` or `AnyRing.RingCell`. By
`theory/NOVELTY.md`'s A1 spectator test the reductions, the additive fusion and the calibration
identity are facts about those structures rather than about this algebra, which is why
`Apeiron/Numeric/Accumulator.lean` reads them as its exact seam.

The primitive that would supply an edge is the order the algebra carries on a null channel,
`Trichotomy.le_iff_exists_sq` and `inCell_meet_iff`, which fix that order by the channel's own
ring. A reduction bound stated over that order rather than a bare `LinearOrder` would then be a
statement about the algebra. Not built.

## What is not here

No order is used beyond the commutative ordered ring the reduction bounds need, and no division,
square root, exponential or completeness appears. The real analytic bounds stay in
`Apeiron/Numeric/Accumulator.lean`.
-/

open scoped BigOperators

namespace Apeiron
namespace Numeric

/- ── §1. sized reductions over a commutative ordered ring ── -/

section Reductions

variable {R : Type*} [CommRing R] [LinearOrder R] [IsStrictOrderedRing R]

/-- The reduction is bounded by `n · M`.

    (terms : List R)                       [a finite list of ring elements]
    (M : R)                                [a common bound on their size]
    (hM : ∀ t ∈ terms, |t| ≤ M)            [every term stays within that bound]
    |terms.foldr (· + ·) 0| ≤ (terms.length : R) * M
                                           [the summed reduction is within length times the bound]

    `List` [a finite sequence of things, in order]
    `∀` [for every single one]
    `∈` [is one of the things in]
    `≤` [is smaller than or equal to]

    A GEMV contraction, an attention score `q · k`, and a weighted-value sum `Σ p · v` are all
    this fold; a finite accumulator has to respect this bound to stay faithful. -/
theorem dot_abs_bounded (terms : List R) (M : R) (hM : ∀ t ∈ terms, |t| ≤ M) :
    |terms.foldr (· + ·) 0| ≤ (terms.length : R) * M := by
  induction terms with
  | nil => simp
  | cons a rest ih =>
    have ha : |a| ≤ M := hM a (List.mem_cons_self)
    have hrest : ∀ t ∈ rest, |t| ≤ M := fun t ht => hM t (List.mem_cons_of_mem a ht)
    have step : |a + rest.foldr (· + ·) 0| ≤ |a| + |rest.foldr (· + ·) 0| := abs_add_le a _
    have hlen : ((a :: rest).length : R) * M = M + (rest.length : R) * M := by
      simp [List.length_cons]; ring
    calc |(a :: rest).foldr (· + ·) 0| = |a + rest.foldr (· + ·) 0| := by
          simp [List.foldr_cons]
      _ ≤ |a| + |rest.foldr (· + ·) 0| := step
      _ ≤ M + (rest.length : R) * M := by linarith [ha, ih hrest]
      _ = ((a :: rest).length : R) * M := hlen.symm

/-- No overflow: a bound below the chosen capacity means the reduction never wraps.

    (terms : List R) (M C : R)                       [terms, their common bound, a capacity]
    (hM : ∀ t ∈ terms, |t| ≤ M)                      [the bound holds]
    (hcap : (terms.length : R) * M < C)              [and the accumulator's capacity clears it]
    |terms.foldr (· + ·) 0| < C                      [so the reduction stays under capacity]

    `List` [a finite sequence of things, in order]
    `∀` [for every single one]
    `∈` [is one of the things in]
    `≤` [is smaller than or equal to]
    `<` [is smaller than]

    A width whose capacity `C` clears `n · M` is the width that never overflows on this
    reduction. Overflow is a violation of `hcap`, not a rounding error the bounds below can
    absorb. -/
theorem accumulator_fits (terms : List R) (M C : R)
    (hM : ∀ t ∈ terms, |t| ≤ M) (hcap : (terms.length : R) * M < C) :
    |terms.foldr (· + ·) 0| < C :=
  lt_of_le_of_lt (dot_abs_bounded terms M hM) hcap

end Reductions

/- ── §2. exact additive bookkeeping and fusion ── -/

section Additive

/-- The rounded value plus its own dropped residue reconstructs the exact sum.

    (rnd : G → G) (a b : G)                          [a rounding map, and two group elements]
    rnd (a + b) + ((a + b) - rnd (a + b)) = a + b     [rounded plus dropped equals exact]

    `→` [takes the left and gives the right; between statements, if the left holds so does
        the right]

    Arithmetic identity: the residue is exactly what `rnd` leaves out, nothing more, so carrying
    it (Kahan-style) is exact bookkeeping rather than a second approximation. No order, no
    multiplication and no rounding model enter, only the additive group. -/
theorem lossy_plus_carry_is_exact {G : Type*} [AddCommGroup G] (rnd : G → G) (a b : G) :
    rnd (a + b) + ((a + b) - rnd (a + b)) = a + b := by
  abel

variable {R : Type*} [CommRing R] [LinearOrder R] [IsStrictOrderedRing R]

/-- The composed error is additive, never multiplicative.

    {rnd : R → R} {ε : R}                                       [a rounding map and its rate]
    (hrel : ∀ z : R, |rnd z - z| ≤ ε * |z|)                     [the relative error bound]
    (a b c : R)                                                 [three elements, two merges deep]
    |rnd (rnd (a + b) + c) - (a + b + c)|
      ≤ ε * |rnd (a + b) + c| + ε * |a + b|                     [the two-merge error is at most the
                                                                 sum of the two per-op residues]

    `→` [takes the left and gives the right; between statements, if the left holds so does
        the right]
    `∀` [for every single one]
    `≤` [is smaller than or equal to]

    Splitting the two-merge deviation into its two one-merge pieces and bounding each with
    `hrel` gives a sum, not a product: the forward pass's numeric error is itself a bounded
    accumulator over the composition depth, additive at every step. -/
theorem lossy_error_is_additive {rnd : R → R} {ε : R}
    (hrel : ∀ z : R, |rnd z - z| ≤ ε * |z|) (a b c : R) :
    |rnd (rnd (a + b) + c) - (a + b + c)| ≤ ε * |rnd (a + b) + c| + ε * |a + b| := by
  have hsplit : rnd (rnd (a + b) + c) - (a + b + c)
      = (rnd (rnd (a + b) + c) - (rnd (a + b) + c)) + ((rnd (a + b) + c) - (a + b + c)) := by
    ring
  have h1 : |rnd (rnd (a + b) + c) - (rnd (a + b) + c)| ≤ ε * |rnd (a + b) + c| := hrel _
  have h2 : |(rnd (a + b) + c) - (a + b + c)| ≤ ε * |a + b| := by
    have hc : (rnd (a + b) + c) - (a + b + c) = rnd (a + b) - (a + b) := by ring
    rw [hc]; exact hrel (a + b)
  calc |rnd (rnd (a + b) + c) - (a + b + c)|
      = |(rnd (rnd (a + b) + c) - (rnd (a + b) + c)) + ((rnd (a + b) + c) - (a + b + c))| := by
        rw [hsplit]
    _ ≤ |rnd (rnd (a + b) + c) - (rnd (a + b) + c)| + |(rnd (a + b) + c) - (a + b + c)| :=
        abs_add_le _ _
    _ ≤ ε * |rnd (a + b) + c| + ε * |a + b| := by linarith [h1, h2]

end Additive

/- ── §3. additive fusion: order and chunking are schedules, not values ── -/

section Fusion

variable {M : Type*} [AddCommMonoid M]

/-- The total of a list, read through the additive monoid.

    (l : List M) : M                                 [a finite list, to its additive total]

    `List` [a finite sequence of things, in order]

    `foldTotal` [a list's total under the additive monoid, left to right from zero]. -/
def foldTotal (l : List M) : M := l.foldl (· + ·) 0

/-- Folding a list from an accumulator is the accumulator plus the list's total.

    (a : M) (l : List M)                             [an accumulator and a list]
    l.foldl (· + ·) a = a + l.foldl (· + ·) 0        [the fold from `a` is `a` plus the total]

    `List` [a finite sequence of things, in order]

    `foldTotal` [a list's total under the additive monoid, left to right from zero]. -/
theorem foldl_add_eq (a : M) (l : List M) :
    l.foldl (· + ·) a = a + l.foldl (· + ·) 0 := by
  simpa using
    List.foldl_assoc (op := (· + ·)) (ha := ⟨add_assoc⟩) (a₁ := a) (a₂ := 0) (l := l)

/-- The total of a cons is the head plus the tail's total.

    (a : M) (t : List M)                             [a head and a tail]
    foldTotal (a :: t) = a + foldTotal t             [the total adds the head in front]

    `List` [a finite sequence of things, in order]

    `foldTotal` [a list's total under the additive monoid, left to right from zero]. -/
theorem foldTotal_cons (a : M) (t : List M) : foldTotal (a :: t) = a + foldTotal t := by
  unfold foldTotal
  simp only [List.foldl_cons, zero_add]
  exact foldl_add_eq a t

/-- The total of a concatenation is the sum of the totals.

    (l₁ l₂ : List M)                                 [two lists]
    foldTotal (l₁ ++ l₂) = foldTotal l₁ + foldTotal l₂   [totals add across concatenation]

    `List` [a finite sequence of things, in order]

    `foldTotal` [a list's total under the additive monoid, left to right from zero]. -/
theorem foldTotal_append (l₁ l₂ : List M) :
    foldTotal (l₁ ++ l₂) = foldTotal l₁ + foldTotal l₂ := by
  induction l₁ with
  | nil => simp [foldTotal]
  | cons a t ih => simp only [List.cons_append, foldTotal_cons, ih, add_assoc]

/-- Any permutation of the list leaves the total unchanged.

    {l₁ l₂ : List M}                                 [two lists of elements]
    (hperm : l₁.Perm l₂)                             [one a rearrangement of the other]
    foldTotal l₁ = foldTotal l₂                      [the same total]

    `List` [a finite sequence of things, in order]
    `Perm` [is a rearrangement of]

    `foldTotal` [a list's total under the additive monoid, left to right from zero].

    So: addition is commutative and associative, so the order a workgroup folds its lanes in is a
    schedule rather than part of the value. -/
theorem foldTotal_perm {l₁ l₂ : List M} (hperm : l₁.Perm l₂) :
    foldTotal l₁ = foldTotal l₂ := by
  unfold foldTotal
  exact List.Perm.foldl_eq' hperm
    (fun x _ y _ z => by rw [add_assoc, add_assoc, add_comm y x]) 0

/-- Splitting the list into contiguous chunks, folding each and combining the partials, gives the
    same total as folding sequentially.

    (chunks : List (List M))                         [the list, split into contiguous chunks]
    (chunks.map foldTotal).foldl (· + ·) 0 = foldTotal chunks.flatten
                                                     [combining each chunk's own total matches
                                                      the total of the flattened list]

    `List` [a finite sequence of things, in order]
    `flatten` [the inner lists joined into one, in order]

    `foldTotal` [a list's total under the additive monoid, left to right from zero].

    So: a worker pool over disjoint contiguous chunks with an in-order combine of the partials,
    and equally a SIMD lane split, is a schedule of the one abstract fold, not a second
    operation. Associativity and identity are the whole cost here. -/
theorem foldTotal_chunks (chunks : List (List M)) :
    (chunks.map foldTotal).foldl (· + ·) 0 = foldTotal chunks.flatten := by
  induction chunks with
  | nil => rfl
  | cons c rest ih =>
    simp only [List.map_cons, List.flatten_cons, List.foldl_cons, zero_add]
    rw [foldl_add_eq (foldTotal c) (List.map foldTotal rest), ih, foldTotal_append]

end Fusion

/- ── §4. the exact calibration identity over a commutative semiring ── -/

section Calibration

variable {R : Type*} [CommSemiring R]

/-- The precision-weighted squared residual, over a per-channel energy `τ`.

    {n : ℕ}                                          [how many channels]
    (τ r : Fin n → R)                                [the channel energy and the residual]
    : R                                               [the distortion]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `→` [takes the left and gives the right; between statements, if the left holds so does
        the right]

    `functionDistortion` [the precision-weighted squared residual, over a per-channel energy].

    The measure a diagonal-Gram forward forces on a residual, stated before any order or square
    root is read into it. -/
def functionDistortion {n : ℕ} (τ r : Fin n → R) : R := ∑ j, τ j * r j ^ 2

/-- The output error is the function distortion, when the calibration channels decouple.

    {T n : ℕ}                                        [how many calibration rows, how many channels]
    (X : Fin T → Fin n → R)                          [the calibration stream]
    (r : Fin n → R)                                  [the per-channel residual]
    (horth : ∀ p q, p ≠ q → ∑ t, X t p * X t q = 0)  [the calibration Gram matrix is diagonal:
                                                       channels do not correlate]
    ∑ t, (∑ p, X t p * r p) ^ 2 = functionDistortion (fun p => ∑ t, X t p ^ 2) r
                                                       [the summed squared output error equals the
                                                        distortion at channel energy
                                                        τ p = Σ_t X t p²]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `→` [takes the left and gives the right; between statements, if the left holds so does
        the right]
    `∀` [for every single one]
    `≠` [is not equal to]

    `functionDistortion` [the precision-weighted squared residual, over a per-channel energy].

    So the calibration stream's per-channel energy, `τ`, is exactly what the forward pass feels
    through the residual; a floor that reads only `r`, or only `‖r‖`, is missing the weight the
    forward pass applies. The identity is polynomial with integer coefficients, so it holds over
    every commutative semiring or in none. -/
theorem output_error_is_function_distortion {T n : ℕ} (X : Fin T → Fin n → R)
    (r : Fin n → R) (horth : ∀ p q, p ≠ q → ∑ t, X t p * X t q = 0) :
    ∑ t, (∑ p, X t p * r p) ^ 2
      = functionDistortion (fun p => ∑ t, X t p ^ 2) r := by
  unfold functionDistortion
  calc ∑ t, (∑ p, X t p * r p) ^ 2
      = ∑ t, ∑ p, ∑ q, (X t p * r p) * (X t q * r q) := by
        refine Finset.sum_congr rfl fun t _ => ?_
        rw [sq, Finset.sum_mul_sum]
    _ = ∑ p, ∑ q, (r p * r q) * ∑ t, X t p * X t q := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun p _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun q _ => ?_
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun t _ => by ring
    _ = ∑ p, (∑ t, X t p ^ 2) * r p ^ 2 := by
        refine Finset.sum_congr rfl fun p _ => ?_
        rw [Finset.sum_eq_single p
          (fun q _ hk => by rw [horth p q (Ne.symm hk), mul_zero])
          (fun h => absurd (Finset.mem_univ p) h)]
        have hsq : ∑ t, X t p * X t p = ∑ t, X t p ^ 2 :=
          Finset.sum_congr rfl fun t _ => (sq (X t p)).symm
        rw [hsq]; ring

end Calibration

end Numeric
end Apeiron
