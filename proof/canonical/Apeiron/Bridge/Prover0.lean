import Apeiron.Combinator.Omega
import Apeiron.Combinator.Pair
import Apeiron.Combinator.Verdict

/-!
# The bridge to song's prover

@name The Prover Bridge
@grade proved

`../song/src/prover/ring/prover0.slang` normalises two terms and compares them. It asserts its
combinator rules as `law` blocks carrying historical citations, and song's own `CAVEATS.md`
records that those laws "are decided by nothing" in its tree: they are non-`for` laws, skipped in
its first pass, so the register is certified for extraction and not for its equations.

This file states each of them in Lean and derives it from `Combinator/Basic.lean` and
`Combinator/Confluence.lean`, which prove all of them already.

Every law song states is covered, including `omega_fuel`, which asserts non-termination and is
proved at that strength in `Combinator/Omega.lean` rather than weakened to a loop.

`distinct_licence` closes the seam `docs/DESIGN.md` §11 names by hand: confluence alone gives
uniqueness of a normal form, never that two distinct floors separate their sources, and
`Combinator/Confluence.lean`'s own header says so. `Combinator/Verdict.lean` proves that other
half, and this file states it under song's own name for it.
-/

namespace Apeiron
namespace Bridge

open Combinator Combinator.Comb

/-- song's `k_select: k x y = x`.

    (x y : Comb)                                     [two terms]
    Reduces (app (app K x) y) x                      [the discarding rule fires] -/
theorem k_select (x y : Comb) : Reduces (app (app K x) y) x :=
  Reduces.single (Step.k x y)

/-- song's `s_rule_open: s x y z = x z (y z)`.

    (x y z : Comb)                                                     [three terms]
    Reduces (app (app (app S x) y) z) (app (app x z) (app y z))        [the duplicating rule] -/
theorem s_rule_open (x y z : Comb) :
    Reduces (app (app (app S x) y) z) (app (app x z) (app y z)) :=
  Reduces.single (Step.s x y z)

/-- song's `i_rule_open: i x = x`.

    (x : Comb)                            [any term]
    Reduces (app I x) x                   [the identity gives its argument back] -/
theorem i_rule_open (x : Comb) : Reduces (app I x) x := I_app x

/-- song's `iota_unfold: iota x = x s k`.

    (x : Comb)                                              [any term]
    Reduces (app iota x) (app (app x S) K)                  [the one constant hands over the two] -/
theorem iota_unfold (x : Comb) : Reduces (app iota x) (app (app x S) K) := iota_app x

/-- song's `iota_generates_s: iota k = s`.

    Reduces (app iota K) S                [the constant applied to K reaches the duplicator]

    So: `iota_app` at `K` gives `K S K`, and the discarding rule throws `K` away. The corpus's
    `iota_is_S` is the four-deep form; this is the one-step form song states. -/
theorem iota_generates_s : Reduces (app iota K) S :=
  Reduces.trans (iota_app K) (Reduces.single (Step.k S K))

/-- song's `church_pair_fst: pair x y k = x`.

    (x y : Comb)                                   [two stored terms]
    Reduces (app (pack x y) K) x                   [applying the pair to K reads the first] -/
theorem church_pair_fst (x y : Comb) : Reduces (app (pack x y) K) x := pack_fst x y

/-- song's `distinct_atoms: k is not s`.

    (K S : Comb)                [the two constants, as terms]
    K ≠ S                       [they are different terms] -/
theorem distinct_atoms : (K : Comb) ≠ S := by
  intro h
  nomatch h

/-- song's `distinct_vars: x is not y`.

    {i j : ℕ} (h : i ≠ j)             [two different variable names]
    (var i var j : Comb)              [the two variables, as terms]
    var i ≠ var j                     [they are different terms]

    So: song's law is stated on two fixed distinct variables; this is the general form. -/
theorem distinct_vars {i j : ℕ} (h : i ≠ j) : (var i : Comb) ≠ var j := by
  intro heq
  exact h (by injection heq)

-- ═══════════════════════════════════════════════════════════════════
-- THE TERM THAT DOES NOT SETTLE
-- ═══════════════════════════════════════════════════════════════════

/-- song's `omega_fuel: s i i (s i i) never lands`.

    {t : Comb}                         [any term]
    Reduces omega t                    [that the looping term reaches]
    ¬ NormalForm t                     [is not a normal form]

    So: no reduction path from `S I I (S I I)` reaches a normal form, whichever redex is chosen
    at each step. `Combinator/Omega.lean` carries the invariant that proves it. With
    `normalForm_unique` this is why song's normalise-and-compare is a sound partial procedure and
    not a decision procedure, and why its `FUEL` verdict is a third honest answer. -/
theorem omega_fuel {t : Comb} (h : Reduces omega t) : ¬ NormalForm t :=
  omega_no_normalForm h

-- ═══════════════════════════════════════════════════════════════════
-- THE DISTINCT LICENCE
-- ═══════════════════════════════════════════════════════════════════

/-- song's DISTINCT licence: a `.distinct` verdict never lies.

    {fuel : Nat} {a b : Comb}                              [a fuel budget, two terms]
    verdict fuel a b = .distinct → ¬ Joinable a b          [a `.distinct` verdict is a genuine
                                                            separation]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `verdict` [a fuel budget, two terms, to the three-valued answer]
    `¬` [this is not true]
    `→` [if the left holds so does the right]

    So: `docs/DESIGN.md` §11 records that `Confluence.lean` gives uniqueness but not
    completeness, "neither `Joinable` transitivity nor checker completeness is stated in this
    file" in that file's own words, and names `NullProver.lean`'s `verdict_distinct_sound` as
    what closes the seam on song's side. `Combinator/Verdict.lean` proves the same theorem
    against this corpus's own `Comb`, from `confluence` and `normalForm_unique`; this is that
    proof under song's name for it. `verdict_decided_exact` is the full exactness statement
    `docs/DESIGN.md` §11 quotes: `verdict fuel a b ≠ .fuel → (verdict fuel a b = .equal ↔
    Joinable a b)`. -/
theorem distinct_licence {fuel : Nat} {a b : Comb} (h : verdict fuel a b = .distinct) :
    ¬ Joinable a b :=
  verdict_distinct_sound h

end Bridge
end Apeiron
