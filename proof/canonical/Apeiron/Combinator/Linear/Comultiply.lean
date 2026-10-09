import Apeiron.Combinator.Linear.Functorial

/-!
# Splitting a reusable copy in two

@name Comultiply
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@grade proved

`Combinator/Contraction.lean` proves `W x y = x y y` with the argument arriving as `rtensor y y`,
and `Combinator/Substitution.lean` proves `S x y z = x z (y z)` the same way. That is the tensor
form, which is the sharper linear fact: it says the two uses consume two copies and nothing more
is needed. Abramsky, Haghverdi and Scott state the contraction with the exponential instead,
`W x !y = x !y !y`, and the step between the two forms is the comultiplication, the map carrying
a reusable copy to a pair of them.

At this carrier that map is a relabelling and no more. A position of `bang y` is a tag and a
position of `y`; a position of `rtensor (bang y) (bang y)` is a side, a tag and a position of `y`.
`contractSplit` is the bijection which reads a tag as a side and a tag, and `conj_contract_bang`
says `bang y` read through it **is** `rtensor (bang y) (bang y)`, on the nose.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | inherited through `rapp`, none new here |

The bijection is arithmetic on the counting numbers, so `theory/NOVELTY.md`'s A1 spectator test
applies as it has throughout: the same splitting serves every setting.

## What is derived and what is not

The two banged equations below are **corollaries**. They are obtained by feeding `bang y` to the
tensor statements already proved and rewriting the argument through the comultiplication, not by
repeating either proof. The tensor statements stay as the primary facts because they are the
stronger ones: they hold for any argument presented as two copies, whether or not those copies
come from an exponential.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- A TAG READ AS A SIDE AND A TAG
-- ═══════════════════════════════════════════════════════════════════

/-- Read a side and a tagged position as a tagged position with the side inside the tag.

    (n : ℕ)                                        [a position of a pair of reusable copies]
    ℕ                                              [its place in one reusable copy] -/
def contractSplitFun (n : ℕ) : ℕ :=
  Sum.elim
    (fun c => Nat.pair (extPos (Nat.unpair c).1) (Nat.unpair c).2)
    (fun c => Nat.pair (argPos (Nat.unpair c).1) (Nat.unpair c).2)
    (Equiv.natSumNatEquivNat.symm n)

/-- Read a tag carrying a side as a side and a tag.

    (m : ℕ)                                        [a position of one reusable copy]
    ℕ                                              [its place in the pair] -/
def contractSplitInv (m : ℕ) : ℕ :=
  Sum.elim
    (fun t => extPos (Nat.pair t (Nat.unpair m).2))
    (fun t => argPos (Nat.pair t (Nat.unpair m).2))
    (Equiv.natSumNatEquivNat.symm (Nat.unpair m).1)

/-- The splitting on the first side.

    (t u : ℕ)                                                          [a tag and a position]
    contractSplitFun (extPos (Nat.pair t u))       [the first side, tagged t]
    = Nat.pair (extPos t) u                        [is the tag which records the first side] -/
theorem contractSplitFun_ext (t u : ℕ) :
    contractSplitFun (extPos (Nat.pair t u)) = Nat.pair (extPos t) u := by
  simp only [contractSplitFun, extPos, Equiv.symm_apply_apply, Sum.elim_inl, Nat.unpair_pair]

/-- The splitting on the second side.

    (t u : ℕ)                                                          [a tag and a position]
    contractSplitFun (argPos (Nat.pair t u))       [the second side, tagged t]
    = Nat.pair (argPos t) u                        [is the tag which records the second side] -/
theorem contractSplitFun_arg (t u : ℕ) :
    contractSplitFun (argPos (Nat.pair t u)) = Nat.pair (argPos t) u := by
  simp only [contractSplitFun, argPos, extPos, Equiv.symm_apply_apply, Sum.elim_inr,
    Nat.unpair_pair]

/-- The splitting read back on the first side.

    (t u : ℕ)                                                          [a tag and a position]
    contractSplitInv (Nat.pair (extPos t) u)       [a tag recording the first side]
    = extPos (Nat.pair t u)                        [is the first side, tagged t] -/
theorem contractSplitInv_ext (t u : ℕ) :
    contractSplitInv (Nat.pair (extPos t) u) = extPos (Nat.pair t u) := by
  simp only [contractSplitInv, extPos, Nat.unpair_pair, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The splitting read back on the second side.

    (t u : ℕ)                                                          [a tag and a position]
    contractSplitInv (Nat.pair (argPos t) u)       [a tag recording the second side]
    = argPos (Nat.pair t u)                        [is the second side, tagged t] -/
theorem contractSplitInv_arg (t u : ℕ) :
    contractSplitInv (Nat.pair (argPos t) u) = argPos (Nat.pair t u) := by
  simp only [contractSplitInv, argPos, extPos, Nat.unpair_pair, Equiv.symm_apply_apply,
    Sum.elim_inr]

/-- **The comultiplication of the exponential, as a bijection of the counting numbers.**

@cite girard-1989

    ℕ ≃ ℕ                                          [a side and a tag, or a tag recording both] -/
def contractSplit : ℕ ≃ ℕ where
  toFun := contractSplitFun
  invFun := contractSplitInv
  left_inv := by
    intro n
    rcases pos_split2 n with ⟨c, rfl⟩ | ⟨c, rfl⟩ <;>
      obtain ⟨t, u, rfl⟩ : ∃ t u, c = Nat.pair t u := ⟨_, _, (Nat.pair_unpair c).symm⟩
    · rw [contractSplitFun_ext, contractSplitInv_ext]
    · rw [contractSplitFun_arg, contractSplitInv_arg]
  right_inv := by
    intro m
    obtain ⟨s, u, rfl⟩ : ∃ s u, m = Nat.pair s u := ⟨_, _, (Nat.pair_unpair m).symm⟩
    rcases pos_split2 s with ⟨t, rfl⟩ | ⟨t, rfl⟩
    · rw [contractSplitInv_ext, contractSplitFun_ext]
    · rw [contractSplitInv_arg, contractSplitFun_arg]

/-- The bijection is the map.

    (n : ℕ)                                        [a position]
    contractSplit n = contractSplitFun n           [the same splitting] -/
theorem contractSplit_apply (n : ℕ) : contractSplit n = contractSplitFun n := rfl

/-- The inverse bijection is the inverse map.

    (n : ℕ)                                        [a position]
    contractSplit.symm n = contractSplitInv n      [the same splitting back] -/
theorem contractSplit_symm_apply (n : ℕ) : contractSplit.symm n = contractSplitInv n := rfl

/-- **A reusable copy read through the splitting is a pair of reusable copies.**

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (y : Route)                                    [a route]
    conjRoute contractSplit (bang y)               [the copies, with the tag read as a side]
    = rtensor (bang y) (bang y)                    [are two independent families of copies]

    So: this is the comultiplication of the exponential at this carrier, and it is a relabelling
    with no trace and no loop. The tags of `bang y` are countably many, so splitting them into
    two countable families costs nothing, which is the whole reason the contraction is available
    for a banged argument and not for a bare one. -/
theorem conj_contract_bang (y : Route) :
    conjRoute contractSplit (bang y) = rtensor (bang y) (bang y) := by
  funext i
  rcases pos_split2 i with ⟨c, rfl⟩ | ⟨c, rfl⟩ <;>
    obtain ⟨t, u, rfl⟩ : ∃ t u, c = Nat.pair t u := ⟨_, _, (Nat.pair_unpair c).symm⟩
  · rw [conjRoute, contractSplit_apply, contractSplitFun_ext, bang_tag, rtensor_ext, bang_tag]
    cases hy : y u with
    | none => rfl
    | some q =>
      rw [Option.map_some, Option.map_some, Option.map_some, Option.map_some,
        contractSplit_symm_apply, contractSplitInv_ext]
  · rw [conjRoute, contractSplit_apply, contractSplitFun_arg, bang_tag, rtensor_arg, bang_tag]
    cases hy : y u with
    | none => rfl
    | some q =>
      rw [Option.map_some, Option.map_some, Option.map_some, Option.map_some,
        contractSplit_symm_apply, contractSplitInv_arg]

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO EQUATIONS IN THE FORM THE LITERATURE STATES THEM
-- ═══════════════════════════════════════════════════════════════════

/-- **W x !y = x !y !y**, the contraction as Abramsky, Haghverdi and Scott write it.

@cite abramsky-haghverdi-scott-2002
@cite curry-1930

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    rapp σ (conjRoute wIdx x) (conjRoute contractSplit (bang y))   [the argument split in two]
    = rapp σ (rapp σ x (bang y)) (bang y)                          [is the argument used twice]

    So: a corollary of `Combinator/Contraction.lean`'s `contraction` and nothing more. The tensor
    statement is fed a banged argument and the comultiplication rewrites the two copies as one,
    which is exactly the step the literature's form hides inside the combinator. -/
theorem contraction_bang (σ : ℝ) (x y : Route) :
    rapp σ (conjRoute wIdx x) (conjRoute contractSplit (bang y))
      = rapp σ (rapp σ x (bang y)) (bang y) := by
  rw [conj_contract_bang]
  exact contraction σ x (bang y)

/-- **S x !z (y !z)**, the substitution with a banged argument.

@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924

    (σ : ℝ) (x y z : Route)                        [the parameter and three routes]
    rapp σ (sComb σ x y) (conjRoute contractSplit (bang z))   [the argument split in two]
    = rapp σ (rapp σ x (bang z)) (rapp σ y (bang z))          [is the argument used by each]

    So: the same corollary for `Combinator/Substitution.lean`'s `substitution`. Both banged forms
    come from the tensor forms by one rewrite, which is the honest relation between them: the
    tensor form is the theorem and the banged form is what it looks like once the exponential
    supplies the two copies. -/
theorem substitution_bang (σ : ℝ) (x y z : Route) :
    rapp σ (sComb σ x y) (conjRoute contractSplit (bang z))
      = rapp σ (rapp σ x (bang z)) (rapp σ y (bang z)) := by
  rw [conj_contract_bang]
  exact substitution σ x y (bang z)

end Combinator
end Apeiron

end
