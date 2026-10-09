import Apeiron.Combinator.Confluence

/-!
# The verdict: normalize-and-compare, exact on the fragment it decides

@name Verdict
@grade proved

`song`'s `prover0` decides two terms equal by normalizing both and comparing the results, with
a fuel budget standing in for a step count SKI's own Turing-completeness makes unbounded. This
file proves that procedure exact, at the strength the design notes §11 states it:

```
verdict fuel a b ≠ .fuel → (verdict fuel a b = .equal ↔ Joinable a b)
```

Confluence gives uniqueness of a normal form, never that one is reached: `Combinator/Omega.lean`
carries a term with none, so a three-valued verdict is forced, and `.fuel` is the honest report
that a side did not land, not a weakening of the other two answers.

## What is built, and on what it rests

`Apeiron.Combinator.Comb` already carries `Step`, `Reduces`, `NormalForm`, `confluence` and
`normalForm_unique`. Nothing here reproves any of that; §1 adds the one piece those files do not
supply, a computable evaluator: `hasRedex` decides `NormalForm` at a term
(`normal_of_hasRedex_false`), and `normalize fuel` iterates one bottom-up reduction pass `fuel`
times, sound by construction
(`reduces_normalize`). §2 is `Joinable`, `TermFloorComplete.lean`'s object read onto this corpus's
own `Comb`. §3 is `floor_classifies_defeq`, ported from
`../sanguine-ref/Substrate/Ski/TermFloorComplete.lean` at its own stated strength: the two
`NormalForm` hypotheses do not come off, since that some fuel reaches a floor is strong
normalisation, which neither corpus proves. §4 is `verdict` itself and the three exactness
theorems `../song-ref/lean/Song/proof/NullProver.lean` states over its own copy of this machinery:
`verdict_equal_sound`, `verdict_distinct_sound`, `verdict_complete`, bundled into the headline
`verdict_decided_exact`.

## What is not ported

`NullProver.lean`'s `Comb` carries `I` and `Iota` as primitive constructors with their own
reduction rules; `Apeiron.Combinator.Comb` has neither, since `Combinator/Basic.lean` derives `I`
and `iota` as terms built from `S` and `K` (`I_app`, `iota_app`). `hasRedex` and `headReduce`
below are written against the two rules `Apeiron.Combinator.Comb.Step` has, `k` and `s`,
and need no `I`/`Iota` cases; `iota`'s own reduction is already covered because it unfolds to a
term the `S`/`K` rules fire on. `verdict_fuel_iff`, the predecessor's separate biconditional
naming exactly when `.fuel` is returned, is not ported as its own theorem: `verdict_decided_exact`
below inlines the one direction the headline needs, and a standalone restatement would add no
licence beyond what the headline already carries.
-/

namespace Apeiron
namespace Combinator
namespace Comb

/- ── §1. a computable evaluator, sound with respect to `Step` and `Reduces` ── -/

/-- Does a rule fire at the head of this term?

    Comb → Bool                          [a term, to whether a rule fires right here] -/
def isHeadRedex : Comb → Bool
  | app (app K _) _ => true
  | app (app (app S _) _) _ => true
  | _ => false

/-- Does a rule fire anywhere inside this term?

    Comb → Bool                          [a term, to whether some rule fires somewhere in it] -/
def hasRedex : Comb → Bool
  | app f g => isHeadRedex (app f g) || hasRedex f || hasRedex g
  | _ => false

/-- Fire one head redex if there is one, else leave the term alone.

    Comb → Comb                          [a term, to one head step's result] -/
def headReduce : Comb → Comb
  | app (app K x) _ => x
  | app (app (app S f) g) x => app (app f x) (app g x)
  | c => c

/-- The head reducer only ever takes a real step.

    (c : Comb)                     [any term]
    Reduces c (headReduce c)       [it reaches what firing its head redex, if any, gives] -/
theorem headReduce_sound (c : Comb) : Reduces c (headReduce c) := by
  unfold headReduce
  split
  · exact Reduces.single (Step.k _ _)
  · exact Reduces.single (Step.s _ _ _)
  · exact Reduces.refl _

/-- One bottom-up pass: reduce both halves of an application, then fire the head.

    Comb → Comb                          [a term, to the result of one full pass over it] -/
def reduceDeep : Comb → Comb
  | app f g => headReduce (app (reduceDeep f) (reduceDeep g))
  | c => c

/-- Every reduction one full pass performs, anywhere in the term, is real.

    (c : Comb)                     [any term]
    Reduces c (reduceDeep c)       [it reaches what one bottom-up pass gives] -/
theorem reduceDeep_sound (c : Comb) : Reduces c (reduceDeep c) := by
  induction c with
  | app f g ihf ihg =>
    have h1 : Reduces (app f g) (app (reduceDeep f) (reduceDeep g)) :=
      (Reduces.left g ihf).trans (Reduces.right (reduceDeep f) ihg)
    exact h1.trans (headReduce_sound (app (reduceDeep f) (reduceDeep g)))
  | _ => exact Reduces.refl _

/-- The evaluator: `reduceDeep` iterated a fixed number of times.

    Nat → Comb → Comb                    [a fuel budget and a term, to the reached floor] -/
def normalize : Nat → Comb → Comb
  | 0, c => c
  | n + 1, c => normalize n (reduceDeep c)

/-- The floor a run reports is a genuine reduct of the input, never fabricated.

    (fuel : Nat) (c : Comb)              [a fuel budget and a term]
    Reduces c (normalize fuel c)         [the reached floor is reachable from it] -/
theorem reduces_normalize (fuel : Nat) (c : Comb) : Reduces c (normalize fuel c) := by
  induction fuel generalizing c with
  | zero => exact Reduces.refl c
  | succ n ih => exact (reduceDeep_sound c).trans (ih (reduceDeep c))

/-- A redex on the left half puts one on the whole application.

    {f g : Comb} (h : hasRedex f = true)     [a term whose left half already has a redex]
    hasRedex (app f g) = true                [so does the application] -/
theorem hasRedex_app_left {f g : Comb} (h : hasRedex f = true) :
    hasRedex (app f g) = true := by
  show (isHeadRedex (app f g) || hasRedex f || hasRedex g) = true
  rw [h]; simp

/-- A redex on the right half puts one on the whole application.

    {f g : Comb} (h : hasRedex g = true)     [a term whose right half already has a redex]
    hasRedex (app f g) = true                [so does the application] -/
theorem hasRedex_app_right {f g : Comb} (h : hasRedex g = true) :
    hasRedex (app f g) = true := by
  show (isHeadRedex (app f g) || hasRedex f || hasRedex g) = true
  rw [h]; simp

/-- A step witnesses a redex: whatever `Step` fires on, `hasRedex` already saw.

    {c c' : Comb}               [two terms]
    Step c c'                   [one rewrites to the other]
    hasRedex c = true           [so the detector reports a redex at the source] -/
theorem step_hasRedex {c c' : Comb} (h : Step c c') : hasRedex c = true := by
  induction h with
  | k x y => rfl
  | s f g x => rfl
  | left b _ ih => exact hasRedex_app_left ih
  | right a _ ih => exact hasRedex_app_right ih

/-- A clean detector report is a genuine floor: `hasRedex` decides `NormalForm`.

    {c : Comb}                   [a term]
    hasRedex c = false           [the detector finds nothing]
    NormalForm c                 [so nothing rewrites it] -/
theorem normal_of_hasRedex_false {c : Comb} (h : hasRedex c = false) : NormalForm c := by
  rintro ⟨c', hstep⟩
  rw [step_hasRedex hstep] at h
  exact Bool.noConfusion h

/- ── §2. `Joinable`: equal by computation ── -/

/-- Two terms share a reduct.

    (a b : Comb) : Prop                       [two terms, to whether they are equal by
                                               computation]

    `Comb` [a term: two constants, and one term applied to another] -/
def Joinable (a b : Comb) : Prop := ∃ d, Reduces a d ∧ Reduces b d

/-- A term shares a reduct with itself.

    (a : Comb)                                [any term]
    Joinable a a                              [it is equal to itself by computation] -/
theorem Joinable.refl (a : Comb) : Joinable a a := ⟨a, Reduces.refl a, Reduces.refl a⟩

/-- `I` is exactly the canonical `S K K` syntax.

    I = app (app S K) K                    [the identity is this exact application tree]

    `I` [the derived identity term]
    `=` [is definitionally equal to]
    `app` [forms an application]
    `S` [the substitution combinator]
    `K` [the constant combinator] -/
theorem I_eq_skk : I = app (app S K) K := rfl

/-- `I` and its canonical `S K K` syntax are joinable.

    Joinable I (app (app S K) K)           [the two exact terms share a reduct]

    `Joinable` [two terms share a reduct]
    `I` [the derived identity term]
    `app` [forms an application]
    `S` [the substitution combinator]
    `K` [the constant combinator] -/
theorem I_skk_joinable : Joinable I (app (app S K) K) := Joinable.refl I

/-- Sharing a reduct is symmetric.

    {a b : Comb} (h : Joinable a b)           [two terms, equal by computation]
    Joinable b a                              [so are they in the other order] -/
theorem Joinable.symm {a b : Comb} (h : Joinable a b) : Joinable b a :=
  let ⟨d, ha, hb⟩ := h; ⟨d, hb, ha⟩

/-- Transitivity, unconditional: confluence is what pays for it.

    {a b c : Comb}                            [three terms]
    Joinable a b → Joinable b c → Joinable a c
                                               [sharing a reduct with a common middle term
                                               is itself transitive]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `→` [if the left holds so does the right] -/
theorem Joinable.trans {a b c : Comb} (hab : Joinable a b) (hbc : Joinable b c) :
    Joinable a c := by
  obtain ⟨d, had, hbd⟩ := hab
  obtain ⟨e, hbe, hce⟩ := hbc
  obtain ⟨f, hdf, hef⟩ := confluence hbd hbe
  exact ⟨f, had.trans hdf, hce.trans hef⟩

/-- Two normal forms that share a reduct are the same term: each reduces only to itself
    (`eq_of_reduces_normalForm`), and `Joinable` hands them a common one.

    {a b : Comb}                              [two terms]
    NormalForm a → NormalForm b → Joinable a b → a = b
                                               [two genuine floors, equal by computation, are
                                               the same term]

    `Comb` [a term: two constants, and one term applied to another]
    `NormalForm` [no rule applies anywhere inside the term, so nothing rewrites it]
    `Joinable` [two terms, to whether they are equal by computation]
    `→` [if the left holds so does the right] -/
theorem normal_joinable_eq {a b : Comb} (ha : NormalForm a) (hb : NormalForm b)
    (hj : Joinable a b) : a = b := by
  obtain ⟨d, had, hbd⟩ := hj
  rw [eq_of_reduces_normalForm ha had, eq_of_reduces_normalForm hb hbd]

/- ── §3. classify by the floor, on the fragment where the fuel lands ── -/

/-- Joinable terms whose fuel reaches a floor on each side share that floor.

    (fuel : Nat) (a b : Comb)                              [a fuel budget, two terms]
    Joinable a b →                                         [equal by computation]
    NormalForm (normalize fuel a) → NormalForm (normalize fuel b) →
                                                            [and both reach a genuine floor]
    normalize fuel a = normalize fuel b                    [their floors coincide]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `NormalForm` [no rule applies anywhere inside the term, so nothing rewrites it]
    `normalize` [a fuel budget and a term, to the reached floor]
    `→` [if the left holds so does the right]

    The chain is `floor a ≃ a ≃ b ≃ floor b` in `Joinable`, closed by `normal_joinable_eq`. -/
theorem floor_eq_of_joinable (fuel : Nat) (a b : Comb) (hj : Joinable a b)
    (h₁ : NormalForm (normalize fuel a)) (h₂ : NormalForm (normalize fuel b)) :
    normalize fuel a = normalize fuel b := by
  have ja : Joinable a (normalize fuel a) := ⟨normalize fuel a, reduces_normalize fuel a,
    Reduces.refl _⟩
  have jb : Joinable b (normalize fuel b) := ⟨normalize fuel b, reduces_normalize fuel b,
    Reduces.refl _⟩
  exact normal_joinable_eq h₁ h₂ ((ja.symm.trans hj).trans jb)

/-- Equal floors are joinable: each reduces to the same reached term, by construction.

    (fuel : Nat) (a b : Comb)                              [a fuel budget, two terms]
    normalize fuel a = normalize fuel b → Joinable a b     [equal floors are equal by
                                                            computation]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `normalize` [a fuel budget and a term, to the reached floor]
    `→` [if the left holds so does the right] -/
theorem joinable_of_floor_eq (fuel : Nat) (a b : Comb) (h : normalize fuel a = normalize fuel b) :
    Joinable a b :=
  ⟨normalize fuel a, reduces_normalize fuel a, h ▸ reduces_normalize fuel b⟩

/-- **Classify by the floor.** Ported from
    `../sanguine-ref/Substrate/Ski/TermFloorComplete.lean` at its own stated strength: on the
    fragment where the fuel reaches a genuine floor for BOTH sides, comparing floors decides
    `Joinable` exactly. The two `NormalForm` hypotheses stay: that some fuel reaches a floor is
    strong normalisation, which SKI's Turing-completeness (`Combinator/Omega.lean`) rules out in
    general, so they are not dropped here either.

    (fuel : Nat) (a b : Comb)                              [a fuel budget, two terms]
    NormalForm (normalize fuel a) → NormalForm (normalize fuel b) →
                                                            [both sides reach a genuine floor]
    (normalize fuel a = normalize fuel b ↔ Joinable a b)   [comparing the floors decides being
                                                            equal by computation]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `NormalForm` [no rule applies anywhere inside the term, so nothing rewrites it]
    `normalize` [a fuel budget and a term, to the reached floor]
    `→` [if the left holds so does the right]
    `↔` [exactly when, true in both directions] -/
theorem floor_classifies_defeq (fuel : Nat) (a b : Comb)
    (h₁ : NormalForm (normalize fuel a)) (h₂ : NormalForm (normalize fuel b)) :
    normalize fuel a = normalize fuel b ↔ Joinable a b :=
  ⟨joinable_of_floor_eq fuel a b, fun hj => floor_eq_of_joinable fuel a b hj h₁ h₂⟩

/- ── §4. the verdict: three-valued, exact on the fragment it decides ── -/

/-- The checker's three answers. `.fuel` is the honest third value, forced by
    `Combinator/Omega.lean`'s witness that no fuel reaches a floor for every term. -/
inductive Verdict : Type where
  | equal
  | distinct
  | fuel
  deriving DecidableEq, Repr

/-- Normalize-and-compare: `song`'s `prover0` decision procedure, computed by this file's own
    evaluator.

    (fuel : Nat) (a b : Comb)                      [a fuel budget and two terms]
    Verdict                                        [the answer: equal, distinct, or out of fuel]

    `Comb` [a term: two constants, and one term applied to another] -/
def verdict (fuel : Nat) (a b : Comb) : Verdict :=
  if hasRedex (normalize fuel a) || hasRedex (normalize fuel b) then .fuel
  else if normalize fuel a = normalize fuel b then .equal else .distinct

/-- `.equal` never lies.

    {fuel : Nat} {a b : Comb}                              [a fuel budget, two terms]
    verdict fuel a b = .equal → Joinable a b               [an `.equal` verdict is a genuine
                                                            equality by computation]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `verdict` [a fuel budget, two terms, to the three-valued answer]
    `→` [if the left holds so does the right] -/
theorem verdict_equal_sound {fuel : Nat} {a b : Comb} (h : verdict fuel a b = .equal) :
    Joinable a b := by
  unfold verdict at h
  split at h
  · exact absurd h (by simp)
  · split at h
    · next heq => exact joinable_of_floor_eq fuel a b heq
    · exact absurd h (by simp)

/-- `.distinct` never lies: two reached, distinct floors separate the inputs. Were they
    joinable, `normal_joinable_eq` would force the floors equal.

    {fuel : Nat} {a b : Comb}                              [a fuel budget, two terms]
    verdict fuel a b = .distinct → ¬ Joinable a b          [a `.distinct` verdict is a genuine
                                                            separation]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `verdict` [a fuel budget, two terms, to the three-valued answer]
    `¬` [this is not true]
    `→` [if the left holds so does the right] -/
theorem verdict_distinct_sound {fuel : Nat} {a b : Comb} (h : verdict fuel a b = .distinct) :
    ¬ Joinable a b := by
  unfold verdict at h
  split at h
  · exact absurd h (by simp)
  · next hor =>
    split at h
    · exact absurd h (by simp)
    · next hne =>
      intro hj
      have hfa : hasRedex (normalize fuel a) = false := by
        cases hra : hasRedex (normalize fuel a) with
        | false => rfl
        | true => exact absurd (by simp [hra]) hor
      have hfb : hasRedex (normalize fuel b) = false := by
        cases hrb : hasRedex (normalize fuel b) with
        | false => rfl
        | true => exact absurd (by simp [hrb]) hor
      exact hne (floor_eq_of_joinable fuel a b hj
        (normal_of_hasRedex_false hfa) (normal_of_hasRedex_false hfb))

/-- Completeness on the decided fragment: a genuine equality by computation whose two floors
    both land is accepted.

    {fuel : Nat} {a b : Comb}                              [a fuel budget, two terms]
    Joinable a b →                                         [equal by computation]
    hasRedex (normalize fuel a) = false → hasRedex (normalize fuel b) = false →
                                                            [and both floors are reached]
    verdict fuel a b = .equal                              [the verdict reports equal]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `hasRedex` [a term, to whether some rule fires somewhere in it]
    `normalize` [a fuel budget and a term, to the reached floor]
    `verdict` [a fuel budget, two terms, to the three-valued answer]
    `→` [if the left holds so does the right] -/
theorem verdict_complete {fuel : Nat} {a b : Comb} (hj : Joinable a b)
    (hna : hasRedex (normalize fuel a) = false) (hnb : hasRedex (normalize fuel b) = false) :
    verdict fuel a b = .equal := by
  have heq := floor_eq_of_joinable fuel a b hj
    (normal_of_hasRedex_false hna) (normal_of_hasRedex_false hnb)
  unfold verdict
  simp [hnb, heq]

/-- **THE HEADLINE.** On the decided fragment the verdict IS `Joinable`, both directions:
    the design notes §11's exactness theorem: `verdict fuel a b ≠ .fuel → (verdict fuel a b =
    .equal ↔ Joinable a b)`. `.fuel` is the only third value: once it is ruled out, `verdict`
    reporting `.equal` and being joinable by computation coincide exactly.

    {fuel : Nat} {a b : Comb}                              [a fuel budget, two terms]
    verdict fuel a b ≠ .fuel →                             [given the verdict is decided]
    (verdict fuel a b = .equal ↔ Joinable a b)             [it reports equal exactly when the
                                                            terms are equal by computation]

    `Comb` [a term: two constants, and one term applied to another]
    `Joinable` [two terms, to whether they are equal by computation]
    `verdict` [a fuel budget, two terms, to the three-valued answer]
    `≠` [is not equal to]
    `→` [if the left holds so does the right]
    `↔` [exactly when, true in both directions] -/
theorem verdict_decided_exact {fuel : Nat} {a b : Comb} (h : verdict fuel a b ≠ .fuel) :
    verdict fuel a b = .equal ↔ Joinable a b := by
  constructor
  · exact verdict_equal_sound
  · intro hj
    have hor : ¬ ((hasRedex (normalize fuel a) || hasRedex (normalize fuel b)) = true) := by
      intro hc
      apply h
      unfold verdict
      rw [if_pos hc]
    have hfa : hasRedex (normalize fuel a) = false := by
      cases hra : hasRedex (normalize fuel a) with
      | false => rfl
      | true => exact absurd (by simp [hra]) hor
    have hfb : hasRedex (normalize fuel b) = false := by
      cases hrb : hasRedex (normalize fuel b) with
      | false => rfl
      | true => exact absurd (by simp [hrb]) hor
    exact verdict_complete hj hfa hfb

end Comb
end Combinator
end Apeiron
