import Apeiron.Linter.Notation
import Mathlib.Data.Nat.Notation
import Mathlib.Logic.Relation

/-!
# Combinators, and one constant that generates them

@name Combinators
@cite curry-1930
@cite barker-2001
@cite goldberg-2004
@cite schonfinkel-1924
@cite barendregt-1984
@cite vanoosten-2008
@cite curry-feys-1958
@cite tomita-2023
@grade proved

A term is one of two constants, a variable, or one term applied to another, and computation
is two rewriting rules:

```
   K x y      becomes   x                      throw the second away
   S f g x    becomes   (f x) (g x)            hand x to both, then apply
```

`K` discards. `S` duplicates its last argument and applies. Bracket abstraction is what forces
both. Its three cases are the variable, a term without it, and an application, and they need
the identity, `K`, and `S` respectively.

## Variables carry names, and abstraction iterates

Variables are numbered, so there are as many as there are counting numbers. This is not
decoration. Abstraction removes one variable at a time, and a calculus with a single variable
can abstract once and then has nothing left to abstract over, so it cannot state what makes
`S` and `K` a basis.

Three statements need a second name and are the substance of this file:

- `hasVar_bracket_of_ne`: abstracting one variable neither creates nor destroys occurrences of
  any other, so a second abstraction still has something to work on
- `bracket_subst`: substituting into an abstraction and then applying reaches the same term as
  substituting in the other order. Reaches, not equals: where the body is the substituted
  variable the two sides build `K a` and `bracket j a`, which differ as terms and agree as
  behaviour
- `bracket_correct_two`: combinatory completeness over two variables, which is
  `bracket_correct` followed by `bracket_subst`, and iterates by one more `bracket_subst` per
  variable

## What this file is, in the field's terms

This is combinatory logic, the term calculus. It is not a combinatory algebra, which is a
carrier set with an application operation satisfying the same equations. The corpus has no
combinatory algebra and no model of one, and `theory/DEAD_ENDS.md` records why no operation of
`Apeiron.Trichotomy` can be application. van Oosten's survey is the reference for the model
side.

## One constant instead of two

Define a single term `iota` whose only rule is that it hands its argument the two constants:

```
   iota M     becomes   M S K
```

Feeding `iota` to itself, and then feeding the results back in, walks through the constants:

```
   iota iota                          behaves as   I
   iota (iota (iota iota))            behaves as   K
   iota (iota (iota (iota iota)))     behaves as   S
```

`iota_is_I`, `iota_is_K` and `iota_is_S` are those three statements. Each says the iota-term
obeys the defining equation of the combinator it stands for, which is what standing for it
means. None of them claims the iota-term is syntactically identical to `S` or `K`; that is
false, and behaving alike is the useful notion.

`iota` is the two-slot pair holding `S` and `K`: `iota_eq_pack` in `Combinator.Pair` proves
`iota = pack S K` by `rfl`. Its one rule is the pair's hand-over rule.
-/

namespace Apeiron
namespace Combinator

/-- A term: two constants, and application.

    Comb             [the type of terms]
    S                [the duplicating constant]
    K                [the discarding constant]
    var : ℕ → Comb   [a variable, named by a counting number]
    app : Comb → Comb → Comb     [one term applied to another] -/
inductive Comb : Type where
  /-- The duplicating constant. -/
  | S : Comb
  /-- The discarding constant. -/
  | K : Comb
  /-- A variable, named by a counting number. Abstraction takes one name at a time, and
      distinct names are what lets it be iterated. -/
  | var : ℕ → Comb
  /-- One term applied to another. -/
  | app : Comb → Comb → Comb
  deriving DecidableEq, Repr

namespace Comb

/-- One rewriting step.

    Step : Comb → Comb → Prop        [one term rewrites to another in a single step]

    So: the two rules, plus permission to rewrite inside either half of an application. -/
inductive Step : Comb → Comb → Prop where
  /-- `K x y` throws the second argument away. -/
  | k (x y : Comb) : Step (app (app K x) y) x
  /-- `S f g x` hands `x` to both `f` and `g`, then applies one result to the other. -/
  | s (f g x : Comb) : Step (app (app (app S f) g) x) (app (app f x) (app g x))
  /-- Rewriting inside the left half of an application. -/
  | left {a a' : Comb} (b : Comb) : Step a a' → Step (app a b) (app a' b)
  /-- Rewriting inside the right half of an application. -/
  | right (a : Comb) {b b' : Comb} : Step b b' → Step (app a b) (app a b')

/-- Rewriting in zero or more steps.

    Reduces : Comb → Comb → Prop     [one term reaches another by any number of steps] -/
def Reduces : Comb → Comb → Prop := Relation.ReflTransGen Step

/-- Every term reaches itself.

    (a : Comb)            [any term]
    Reduces a a           [in zero steps] -/
theorem Reduces.refl (a : Comb) : Reduces a a := Relation.ReflTransGen.refl

/-- A single step is a reduction.

    {a b : Comb}          [two terms]
    Step a b              [one rewrites to the other in one step]
    Reduces a b           [so it reaches it] -/
theorem Reduces.single {a b : Comb} (h : Step a b) : Reduces a b :=
  Relation.ReflTransGen.single h

/-- Reductions chain.

    {a b c : Comb}        [three terms]
    Reduces a b           [the first reaches the second]
    Reduces b c           [the second reaches the third]
    Reduces a c           [so the first reaches the third] -/
theorem Reduces.trans {a b c : Comb} (h₁ : Reduces a b) (h₂ : Reduces b c) : Reduces a c :=
  Relation.ReflTransGen.trans h₁ h₂

/-- Rewriting is allowed inside the left half of an application.

    {a a' : Comb} (b : Comb)                 [two terms and a third]
    Reduces a a'                             [the first reaches the second]
    Reduces (app a b) (app a' b)             [so does the application] -/
theorem Reduces.left {a a' : Comb} (b : Comb) (h : Reduces a a') :
    Reduces (app a b) (app a' b) := by
  induction h with
  | refl => exact Reduces.refl _
  | tail _ hstep ih => exact ih.trans (Reduces.single (Step.left b hstep))

/-- Rewriting is allowed inside the right half of an application.

    (a : Comb) {b b' : Comb}                 [a term and two others]
    Reduces b b'                             [the second reaches the third]
    Reduces (app a b) (app a b')             [so does the application] -/
theorem Reduces.right (a : Comb) {b b' : Comb} (h : Reduces b b') :
    Reduces (app a b) (app a b') := by
  induction h with
  | refl => exact Reduces.refl _
  | tail _ hstep ih => exact ih.trans (Reduces.single (Step.right a hstep))

-- ═══════════════════════════════════════════════════════════════════
-- THE IDENTITY, AND THE ONE CONSTANT
-- ═══════════════════════════════════════════════════════════════════

/-- The term that returns its argument unchanged.

    I                          [the name for this term]
    app (app S K) K            [S applied to K, applied to K]
    : Comb                     [a term] -/
def I : Comb := app (app S K) K

/-- The identity returns its argument.

    (x : Comb)                 [any term]
    Reduces (app I x) x        [applying the identity gives the argument back]

    So: `S K K x` hands `x` to both copies of `K`, giving `K x (K x)`, and `K` throws the
    second away. -/
theorem I_app (x : Comb) : Reduces (app I x) x := by
  refine Reduces.trans (Reduces.single (Step.s K K x)) ?_
  exact Reduces.single (Step.k x (app K x))

/-- The single constant that hands its argument the other two.

    iota                                       [the name for this term]
    app (app S (app (app S I) (app K S))) (app K K)     [its construction from S and K]
    : Comb                                     [a term] -/
def iota : Comb := app (app S (app (app S I) (app K S))) (app K K)

/-- The one constant hands its argument the other two.

    (M : Comb)                                      [any term]
    Reduces (app iota M) (app (app M S) K)          [iota applied to M gives M S K]

    So: this is the only rule `iota` has, and everything below is got by feeding it to
    itself. -/
theorem iota_app (M : Comb) : Reduces (app iota M) (app (app M S) K) := by
  refine Reduces.trans (Reduces.single (Step.s _ _ M)) ?_
  refine Reduces.trans (Reduces.left _ (Reduces.single (Step.s I (app K S) M))) ?_
  refine Reduces.trans (Reduces.left _ (Reduces.left _ (I_app M))) ?_
  refine Reduces.trans (Reduces.left _ (Reduces.right _ (Reduces.single (Step.k S M)))) ?_
  exact Reduces.right _ (Reduces.single (Step.k K M))

-- ═══════════════════════════════════════════════════════════════════
-- FEEDING THE CONSTANT TO ITSELF
-- ═══════════════════════════════════════════════════════════════════

/-- Applied to itself, the constant behaves as the identity.

    (x : Comb)                                    [any term]
    Reduces (app (app iota iota) x) x             [iota iota returns its argument]

    So: `iota iota` reaches `S K (K K)`, and that applied to `x` hands `x` to both halves,
    giving `K x (K K x)`, whose `K` throws the second away. -/
theorem iota_is_I (x : Comb) : Reduces (app (app iota iota) x) x := by
  have h : Reduces (app iota iota) (app (app S K) (app K K)) := by
    refine Reduces.trans (iota_app iota) ?_
    refine Reduces.trans (Reduces.left K (iota_app S)) ?_
    exact Reduces.single (Step.s S K K)
  refine Reduces.trans (Reduces.left x h) ?_
  refine Reduces.trans (Reduces.single (Step.s K (app K K) x)) ?_
  exact Reduces.single (Step.k x (app (app K K) x))

/-- Three applications of the constant reach the discarding constant itself.

    Reduces (app iota (app iota (app iota iota))) K     [iota (iota (iota iota)) reaches K]

    So: `iota (iota iota)` reaches `S K`, so the outer application reaches `(S K) S K`, which
    steps to `(K K) (S K)` and then throws the second half away. The result is `K` itself,
    not a term that behaves like it. -/
theorem iota_is_K : Reduces (app iota (app iota (app iota iota))) K := by
  have hinner : Reduces (app iota (app iota iota)) (app S K) := by
    refine Reduces.trans (iota_app (app iota iota)) ?_
    exact Reduces.left K (iota_is_I S)
  refine Reduces.trans (iota_app (app iota (app iota iota))) ?_
  refine Reduces.trans (Reduces.left K (Reduces.left S hinner)) ?_
  refine Reduces.trans (Reduces.single (Step.s K S K)) ?_
  exact Reduces.single (Step.k K (app S K))

/-- Four applications of the constant reach the duplicating constant itself.

    Reduces (app iota (app iota (app iota (app iota iota)))) S    [four deep reaches S]

    So: the inner three reach `K`, and `iota K` is `K S K`, which throws `K` away and leaves
    `S`. -/
theorem iota_is_S : Reduces (app iota (app iota (app iota (app iota iota)))) S := by
  refine Reduces.trans (iota_app (app iota (app iota (app iota iota)))) ?_
  refine Reduces.trans (Reduces.left K (Reduces.left S iota_is_K)) ?_
  exact Reduces.single (Step.k S K)

/-- The one constant recovers all three.

    (x : Comb)                                                  [any term]
    Reduces (app (app iota iota) x) x                           [it gives the identity]
    Reduces (app iota (app iota (app iota iota))) K             [it gives K]
    Reduces (app iota (app iota (app iota (app iota iota)))) S  [it gives S]
    ∧                                                           [all three together]

    So: one constant and one rule are enough to reach both of the constants everything else is
    built from. -/
theorem iota_generates (x : Comb) :
    Reduces (app (app iota iota) x) x ∧
      Reduces (app iota (app iota (app iota iota))) K ∧
      Reduces (app iota (app iota (app iota (app iota iota)))) S :=
  ⟨iota_is_I x, iota_is_K, iota_is_S⟩

-- ═══════════════════════════════════════════════════════════════════
-- ABSTRACTION: WHY THESE TWO CONSTANTS AND NOT SOME OTHERS
-- ═══════════════════════════════════════════════════════════════════

/-- Replace one variable by a term.

    (i : ℕ)                 [which variable, named by a counting number]
    Comb → Comb → Comb      [takes the term and the replacement, gives a term]

    So: `S` and `K` hold no variable and are unchanged. The named variable becomes the
    argument, every other variable stays put, and an application substitutes into both
    halves. -/
def subst (i : ℕ) : Comb → Comb → Comb
  | S, _ => S
  | K, _ => K
  | var j, t => if j = i then t else var j
  | app f g, t => app (subst i f t) (subst i g t)

/-- Does the term hold this variable?

    (i : ℕ)                 [which variable, named by a counting number]
    Comb → Bool             [takes a term, gives true exactly when that variable occurs] -/
def hasVar (i : ℕ) : Comb → Bool
  | var j => j == i
  | app f g => hasVar i f || hasVar i g
  | _ => false

/-- Abstraction: turn a term into one that takes the named variable as an argument.

@cite schonfinkel-1924
@cite curry-1930
@cite tomita-2023

    (i : ℕ)                 [which variable to abstract, named by a counting number]
    Comb → Comb             [takes a term, gives one that rebuilds it from an argument]

    So: this is Curry and Schönfinkel's bracket algorithm in its unoptimised form. The named
    variable becomes the identity. Anything holding no occurrence of it is wrapped in `K`,
    which will discard the argument. An application becomes `S` over the two abstracted
    halves, which hands the argument to both. The three cases are what force `S` and `K` to
    exist: discarding is what a closed subterm needs and sharing is what an application
    needs. -/
def bracket (i : ℕ) : Comb → Comb
  | var j => if j = i then I else app K (var j)
  | app f g => app (app S (bracket i f)) (bracket i g)
  | e => app K e

/-- Abstraction removes the variable it names.

    (i : ℕ)                          [which variable, named by a counting number]
    (e : Comb)                       [any term]
    hasVar i (bracket i e) = false   [the abstracted term no longer holds that variable]

    So: the argument slot is empty, which is what makes it a slot. -/
theorem hasVar_bracket (i : ℕ) (e : Comb) : hasVar i (bracket i e) = false := by
  induction e with
  | S => rfl
  | K => rfl
  | var j => by_cases h : j = i <;> simp [bracket, hasVar, I, h]
  | app f g ihf ihg => simp [bracket, hasVar, ihf, ihg]

/-- Abstraction leaves every other variable exactly as it found it.

    {i j : ℕ}                                  [two variables, named by counting numbers]
    (h : j ≠ i)                                [they are different names]
    (e : Comb)                                 [any term]
    hasVar j (bracket i e) = hasVar j e        [abstracting one does not disturb the other]

    So: abstraction can be iterated. Bracketing `i` neither creates nor destroys occurrences
    of `j`, so bracketing `j` afterwards still has something to work on. The one-variable
    calculus cannot state this, because there is no second variable to state it about. -/
theorem hasVar_bracket_of_ne {i j : ℕ} (h : j ≠ i) (e : Comb) :
    hasVar j (bracket i e) = hasVar j e := by
  induction e with
  | S => rfl
  | K => rfl
  | var k =>
    by_cases hk : k = i
    · subst hk
      simp [bracket, hasVar, I, Ne.symm h]
    · simp [bracket, hasVar, hk]
  | app f g ihf ihg => simp [bracket, hasVar, ihf, ihg]

/-- Applying an abstraction to a term performs the substitution.

    (i : ℕ)                                          [which variable, by counting number]
    (e x : Comb)                                     [the abstracted term, and an argument]
    Reduces (app (bracket i e) x) (subst i e x)      [applying it puts x at that variable]

    So: this is combinatory completeness for `S` and `K`, over one variable at a time. -/
theorem bracket_correct (i : ℕ) (e x : Comb) :
    Reduces (app (bracket i e) x) (subst i e x) := by
  induction e with
  | S => exact Reduces.single (Step.k S x)
  | K => exact Reduces.single (Step.k K x)
  | var j =>
    by_cases h : j = i
    · simp only [bracket, subst, if_pos h]
      exact I_app x
    · simp only [bracket, subst, if_neg h]
      exact Reduces.single (Step.k (var j) x)
  | app f g ihf ihg =>
    refine Reduces.trans (Reduces.single (Step.s (bracket i f) (bracket i g) x)) ?_
    exact Reduces.trans (Reduces.left _ ihf) (Reduces.right _ ihg)

/-- Every term is a closed term applied to the variable it was abstracted over.

@cite tomita-2023

    (i : ℕ)                                     [which variable, by counting number]
    (e : Comb)                                    [any term]
    ∃ c : Comb                                    [there is a term]
    hasVar i c = false                            [holding no occurrence of that variable]
    ∀ j, j ≠ i → hasVar j c = hasVar j e          [and holding every other one exactly as e did]
    ∀ x, Reduces (app c x) (subst i e x)          [and applying it substitutes into e]
    ∧                                             [all three together]

    So: `S` and `K` suffice to express abstraction over any one variable. This is the
    statement the corpus previously took on citation from Schönfinkel and Curry.

    The middle clause is not decoration. Combinatory completeness is stated in the field as
    "the free variables of the result are those of the body, less the abstracted one", and
    without it the theorem permits an abstraction that quietly loses the other variables and
    so cannot be iterated. -/
theorem combinatory_completeness (i : ℕ) (e : Comb) :
    ∃ c : Comb, hasVar i c = false ∧ (∀ j, j ≠ i → hasVar j c = hasVar j e) ∧
      ∀ x : Comb, Reduces (app c x) (subst i e x) :=
  ⟨bracket i e, hasVar_bracket i e, fun _ hj => hasVar_bracket_of_ne hj e,
    bracket_correct i e⟩

-- ═══════════════════════════════════════════════════════════════════
-- ITERATED ABSTRACTION
-- ═══════════════════════════════════════════════════════════════════

/-- A term without a variable is unchanged by substituting for it.

    {i : ℕ}                        [a variable, named by a counting number]
    {e : Comb}                     [a term]
    (h : hasVar i e = false)       [it holds no occurrence of that variable]
    (x : Comb)                     [anything at all]
    subst i e x = e                [substituting changes nothing] -/
theorem subst_of_not_hasVar {i : ℕ} {e : Comb} (h : hasVar i e = false) (x : Comb) :
    subst i e x = e := by
  induction e with
  | S => rfl
  | K => rfl
  | var j =>
    have hj : ¬ j = i := by simpa [hasVar] using h
    simp [subst, hj]
  | app f g ihf ihg =>
    simp only [hasVar, Bool.or_eq_false_iff] at h
    simp [subst, ihf h.1, ihg h.2]

/-- Substitution passes through an abstraction over a different variable.

@cite curry-1930
@cite barendregt-1984

    {i j : ℕ}                      [two variables, named by counting numbers]
    (hij : i ≠ j)                  [they are different names]
    {a : Comb}                     [the term being substituted]
    (ha : hasVar j a = false)      [it holds no occurrence of the abstracted variable]
    (e x : Comb)                   [the body, and an argument for the abstraction]
    Reduces (app (subst i (bracket j e) a) x)    [substitute in, abstract, then apply]
      (subst j (subst i e a) x)                  [reaches substituting in the other order]

    So: substituting into an abstraction, then applying, reaches the same place as
    substituting first and applying after. The two are not the same term. Where the body is
    the substituted variable, the left builds `K a` and the right builds `bracket j a`, and
    those are different terms with the same behaviour. Behavioural agreement is the notion
    combinatory logic runs on, and equality is not available here.

    This is the lemma that makes abstraction iterable, and it is what a calculus with one
    variable cannot even state. -/
theorem bracket_subst {i j : ℕ} (hij : i ≠ j) {a : Comb} (ha : hasVar j a = false)
    (e x : Comb) :
    Reduces (app (subst i (bracket j e) a) x) (subst j (subst i e a) x) := by
  induction e with
  | S => exact Reduces.single (Step.k S x)
  | K => exact Reduces.single (Step.k K x)
  | var k =>
    by_cases hkj : k = j
    · subst hkj
      have hki : ¬ k = i := fun hh => hij (hh ▸ rfl)
      simp only [bracket, subst, if_true, if_neg hki]
      simpa only [I, subst] using I_app x
    · by_cases hki : k = i
      · subst hki
        have hb : bracket j (var k) = app K (var k) := by simp [bracket, hkj]
        have hs : subst k (bracket j (var k)) a = app K a := by simp [hb, subst]
        have hr : subst j (subst k (var k) a) x = a := by
          simp only [subst]
          exact subst_of_not_hasVar ha x
        rw [hs, hr]
        exact Reduces.single (Step.k a x)
      · simp only [bracket, subst, if_neg hki, if_neg hkj]
        exact Reduces.single (Step.k (var k) x)
  | app f g ihf ihg =>
    simp only [bracket, subst]
    refine Reduces.trans (Reduces.single (Step.s _ _ x)) ?_
    exact Reduces.trans (Reduces.left _ ihf) (Reduces.right _ ihg)

/-- Combinatory completeness over two variables.

@cite schonfinkel-1924
@cite curry-1930

    {i j : ℕ}                      [two variables, named by counting numbers]
    (hij : i ≠ j)                  [they are different names]
    (e : Comb)                     [any term]
    {a : Comb}                     [the first argument]
    (ha : hasVar j a = false)      [it holds no occurrence of the second variable]
    (x : Comb)                     [the second argument]
    Reduces (app (app (bracket i (bracket j e)) a) x)   [abstract twice, apply twice]
      (subst j (subst i e a) x)                         [substitutes both arguments]

    So: `S` and `K` express abstraction over two variables in turn, which is what makes them a
    basis rather than a pair that handles one variable. The same argument iterates, one
    application of `bracket_subst` per extra variable. -/
theorem bracket_correct_two {i j : ℕ} (hij : i ≠ j) (e : Comb) {a : Comb}
    (ha : hasVar j a = false) (x : Comb) :
    Reduces (app (app (bracket i (bracket j e)) a) x) (subst j (subst i e a) x) :=
  Reduces.trans (Reduces.left x (bracket_correct i (bracket j e) a))
    (bracket_subst hij ha e x)

/-- Two abstractions leave neither variable behind.

    {i j : ℕ}                                       [two variables, by counting number]
    (h : j ≠ i)                                     [they are different names]
    (e : Comb)                                      [any term]
    hasVar i (bracket i (bracket j e)) = false      [the first is gone]
    hasVar j (bracket i (bracket j e)) = false      [and so is the second]
    ∧                                               [both together] -/
theorem hasVar_bracket_two {i j : ℕ} (h : j ≠ i) (e : Comb) :
    hasVar i (bracket i (bracket j e)) = false ∧
      hasVar j (bracket i (bracket j e)) = false :=
  ⟨hasVar_bracket i _, by rw [hasVar_bracket_of_ne h, hasVar_bracket]⟩

/-- Abstracting two different variables in turn rebuilds the discarding constant.

    (x y : Comb)                                                       [two terms]
    Reduces (app (app (bracket 0 (bracket 1 (var 0))) x) y) x          [it behaves as K]

    So: `[0]([1] v0)` is `S (K K) I`, which takes `x`, drops to `K x'`, and then discards `y`.
    `K` is recovered from abstraction rather than assumed, which needs two variable names and
    so could not be stated before. -/
theorem bracket_two_is_K (x y : Comb) :
    Reduces (app (app (bracket 0 (bracket 1 (var 0))) x) y) x := by
  have hstep : Reduces (app (bracket 0 (bracket 1 (var 0))) x) (app K (app I x)) := by
    refine Reduces.trans (Reduces.single (Step.s (app K K) I x)) ?_
    exact Reduces.left _ (Reduces.single (Step.k K x))
  refine Reduces.trans (Reduces.left y hstep) ?_
  refine Reduces.trans (Reduces.single (Step.k (app I x) y)) ?_
  exact I_app x

end Comb
end Combinator
end Apeiron
