import Apeiron.Combinator.Basic
import Apeiron.Trichotomy.Implicative

/-!
# A model of the calculus, built over the points that survive squaring

@name Model
@cite engeler-1981
@cite scott-1976
@cite barendregt-1984
@cite vanoosten-2008
@cite palmieri-2026
@cite tomita-2023
@grade proved

`theory/LITERATURE.md` §B recorded that the corpus had no applicative structure and no model of
one. This file builds one, over the four points of the algebra that survive squaring, and says
exactly how much of the algebra it carries.

## Root

| from | here |
|---|---|
| `idempotent_classification` | `two_channelled_atom_iff_disc_pos`, `model_over_the_survivors` |
| `upper_mul`, `lower_mul`, `halves_idemPlus`, `halves_idemMinus` | `feed_forgets_the_survivors` |
| `two_boolean_channels_iff_disc_pos` | `two_channelled_atom_iff_disc_pos` |

The atoms this model is built over are the survivors of squaring, and σ's sign fixes which they
are: four above zero, two at or below. `two_channelled_atom_iff_disc_pos` puts the atom set on
the split face by theorem, since an atom whose two channels carry independent truth values
exists there and nowhere else.

**Combinatory completeness itself is free of σ, and that is a finding rather than an
omission.** `feed_kPoint`, `feed_sPoint` and `feed_meaning_bracket` are stated over an arbitrary
atom type and use no property of the atoms at all. By `theory/NOVELTY.md`'s A1 spectator test
they are therefore not results about this algebra: they are Engeler's theorem, which holds over
any set, and the corpus's contribution is the machine-checked proof rather than the statement.

What is about this algebra is the second half. `feed_forgets_the_survivors` says the model's
application cannot read the atoms: `mul σ` separates the survivors above zero, computing AND on
two independent channels, while feeding one atom to another gives the empty set for every pair.
So the algebra enters this construction as a set and leaves its multiplication at the door. That
is the exact strength of this route, and it is why the route is a model of the calculus rather
than a bridge to it.

The primitive that would close the gap is named in `theory/ROOTING.md`: an arrow whose source is
a point of the algebra rather than a finite set of tokens, so that feeding could consult
`mul σ`. Not built.

## The construction

@cite engeler-1981

Engeler's graph model. From any set of atoms, build tokens: an atom is a token, and a finite
list of tokens together with a token is a token, written as an arrow. An element of the model is
a set of tokens. Feeding `X` to `Y` collects the target of every arrow in `X` whose whole source
list lies in `Y`:

```
   X · Y   =   { c : arrow l c ∈ X, and every b in l is in Y }
```

Scott's `P(ω)` is the same construction with the tokens coded as integers. Application is not an
operation of the atoms, and that is the point: the two constants are sets of arrows, so nothing
here needs the atoms to multiply. `theory/DEAD_ENDS.md` rules out application as an operation of
the algebra, and this route obeys that ruling by not asking the algebra for one.

`kPoint` holds every arrow `l → (m → d)` with `d` in `l`: feeding it `X` keeps a token of `X`
and then discards whatever comes next, because the second source list may be empty. `sPoint`
holds every arrow whose three source lists supply, in order, one arrow of the first argument,
one arrow per token the second argument must produce, and one shared list feeding both. Those
two equations are `feed_kPoint` and `feed_sPoint`, and they are equalities of sets rather than
inequalities, so this is a model in the strict sense and not an ordered approximation of one.

## What is proved about the calculus

`meaning` sends a term to an element: `S` and `K` to the two constants, a variable to whatever
the environment holds, and an application to a feeding. `meaning_reduces` is soundness, that a
term and anything it reaches have the same meaning, so reduction is sent to equality of
elements. That is the first of the three shapes `theory/DEAD_ENDS.md` left open.

`feed_meaning_bracket` is combinatory completeness: feeding an argument to the meaning of a
bracket abstraction is the meaning of the body with that argument bound, for arbitrary elements
of the model rather than for meanings of terms. That is the statement Tomita's Propositions 1
and 2 identify with being a partial combinatory algebra, here total.
-/

namespace Apeiron
namespace Combinator

open Trichotomy

universe u

/-- A token of the graph model: an atom, or a finite list of tokens pointing at one token.

@cite engeler-1981

    (A : Type u)                                  [the atoms the model is built over]
    atom : A → Engeler A                          [every atom is a token]
    arrow : List (Engeler A) → Engeler A → Engeler A   [a finite list of tokens, and a target]

    So: the least set closed under both rules, which is what an inductive type is. The arrow is
    a single approximation step of a function: given at least the tokens in the list, produce
    the target. -/
inductive Engeler (A : Type u) : Type u where
  /-- An atom of the underlying set, taken as a token. -/
  | atom : A → Engeler A
  /-- A finite list of tokens, and the token they yield. -/
  | arrow : List (Engeler A) → Engeler A → Engeler A

/-- Feeding one element of the model to another.

@cite engeler-1981
@cite scott-1976

    {A : Type u}                                                [the atoms]
    (X Y : Set (Engeler A))                                     [two elements of the model]
    ∃ l : List (Engeler A)                              [some finite list of tokens]
    ∀ b ∈ l, b ∈ Y                                      [all of them held by the right one]
    Engeler.arrow l c ∈ X                               [arrowing to c inside the left one]
    : Set (Engeler A)                                   [the set of all such c]

    So: this is application. It is neither commutative nor associative, and it is not an
    operation of the atoms, which is how it escapes the obstruction in `theory/DEAD_ENDS.md`. -/
def feed {A : Type u} (X Y : Set (Engeler A)) : Set (Engeler A) :=
  {c | ∃ l : List (Engeler A), (∀ b ∈ l, b ∈ Y) ∧ Engeler.arrow l c ∈ X}

/-- The discarding constant, as a set of arrows.

@cite engeler-1981

    (A : Type u)                                                     [the atoms]
    {c | ∃ (l m : List (Engeler A)) (d : Engeler A),                 [the arrows l → (m → d)]
      c = Engeler.arrow l (Engeler.arrow m d) ∧ d ∈ l}               [whose target lies in l]
    : Set (Engeler A)                                                [an element of the model]

    So: feeding it something keeps that thing's tokens, and the second source list may be
    empty, so the next argument is discarded. -/
def kPoint (A : Type u) : Set (Engeler A) :=
  {c | ∃ (l m : List (Engeler A)) (d : Engeler A),
    c = Engeler.arrow l (Engeler.arrow m d) ∧ d ∈ l}

/-- The duplicating constant, as a set of arrows.

@cite engeler-1981

    (A : Type u)                                                  [the atoms]
    c = Engeler.arrow f (Engeler.arrow g (Engeler.arrow x d))     [an arrow taking three lists]
    Engeler.arrow n (Engeler.arrow p d) ∈ f                       [the first supplies one arrow]
    ∀ b ∈ n, b ∈ x                                                [whose source is in the third]
    ∀ q ∈ p, ∃ nq, Engeler.arrow nq q ∈ g ∧ ∀ b ∈ nq, b ∈ x       [the second supplies the rest]
    : Set (Engeler A)                                             [an element of the model]

    So: the third list is fed to both of the first two, which is what duplication is. -/
def sPoint (A : Type u) : Set (Engeler A) :=
  {c | ∃ (f g x n p : List (Engeler A)) (d : Engeler A),
    c = Engeler.arrow f (Engeler.arrow g (Engeler.arrow x d)) ∧
    Engeler.arrow n (Engeler.arrow p d) ∈ f ∧
    (∀ b ∈ n, b ∈ x) ∧
    ∀ q ∈ p, ∃ nq : List (Engeler A), Engeler.arrow nq q ∈ g ∧ ∀ b ∈ nq, b ∈ x}

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO EQUATIONS, AS EQUALITIES OF SETS
-- ═══════════════════════════════════════════════════════════════════

/-- The discarding constant discards.

@cite engeler-1981

    (A : Type u)                                       [the atoms]
    (X Y : Set (Engeler A))                            [two elements of the model]
    feed (feed (kPoint A) X) Y = X                     [feeding it two things returns the first]

    So: an equality of sets, not an inequality. The second source list is taken empty, which is
    why the second argument leaves no trace. -/
theorem feed_kPoint (A : Type u) (X Y : Set (Engeler A)) :
    feed (feed (kPoint A) X) Y = X := by
  ext c
  simp only [feed, kPoint, Set.mem_setOf_eq]
  constructor
  · rintro ⟨m, -, l, hl, l', m', d', heq, hd⟩
    injection heq with h1 h2
    injection h2 with _ h4
    subst h1
    subst h4
    exact hl _ hd
  · intro hc
    exact ⟨[], by simp, [c], by simpa using hc, [c], [], c, rfl, by simp⟩

/-- Finitely many demands on two elements are met by one finite list from each.

    {A : Type u}                                                     [the atoms]
    (G X : Set (Engeler A))                                          [two elements of the model]
    (p : List (Engeler A))                                           [a list of demanded tokens]
    ∀ q ∈ p, ∃ nq, (∀ b ∈ nq, b ∈ X) ∧ Engeler.arrow nq q ∈ G       [G arrows to each of them]
    →                                                               [then, if that holds]
    ∃ g xs : List (Engeler A)                                       [then two finite lists]
    (∀ b ∈ g, b ∈ G) ∧ (∀ b ∈ xs, b ∈ X)                            [one inside each element]
    ∀ q ∈ p, ∃ nq, Engeler.arrow nq q ∈ g ∧ ∀ b ∈ nq, b ∈ xs        [do for every demand]

    So: the witnesses are collected by concatenation, one step per demanded token. This is the
    only place finiteness does any work, and it is what makes the duplicating equation hold. -/
theorem collect_witnesses {A : Type u} (G X : Set (Engeler A)) :
    ∀ p : List (Engeler A),
      (∀ q ∈ p, ∃ nq : List (Engeler A), (∀ b ∈ nq, b ∈ X) ∧ Engeler.arrow nq q ∈ G) →
      ∃ g xs : List (Engeler A), (∀ b ∈ g, b ∈ G) ∧ (∀ b ∈ xs, b ∈ X) ∧
        ∀ q ∈ p, ∃ nq : List (Engeler A), Engeler.arrow nq q ∈ g ∧ ∀ b ∈ nq, b ∈ xs := by
  intro p
  induction p with
  | nil => intro _; exact ⟨[], [], by simp, by simp, by simp⟩
  | cons q p ih =>
    intro h
    obtain ⟨nq, hnqX, hnqG⟩ := h q (by simp)
    obtain ⟨g, xs, hg, hxs, hall⟩ := ih fun r hr => h r (by simp [hr])
    refine ⟨Engeler.arrow nq q :: g, nq ++ xs, ?_, ?_, ?_⟩
    · intro b hb
      rcases List.mem_cons.mp hb with rfl | hb
      · exact hnqG
      · exact hg b hb
    · intro b hb
      rcases List.mem_append.mp hb with hb | hb
      · exact hnqX b hb
      · exact hxs b hb
    · intro r hr
      rcases List.mem_cons.mp hr with rfl | hr
      · exact ⟨nq, by simp, fun b hb => List.mem_append.mpr (Or.inl hb)⟩
      · obtain ⟨nr, hnr1, hnr2⟩ := hall r hr
        exact ⟨nr, by simp [hnr1], fun b hb => List.mem_append.mpr (Or.inr (hnr2 b hb))⟩

/-- The duplicating constant duplicates.

@cite engeler-1981

    {A : Type u}                                                       [the atoms]
    (F G X : Set (Engeler A))                                          [three elements]
    feed (feed (feed (sPoint A) F) G) X                                [fed all three in turn]
    = feed (feed F X) (feed G X)                                       [hands the third to both]

    So: an equality of sets again. Reading forward is one unpacking; reading back needs
    `collect_witnesses`, because the second argument is consulted once for every token the
    first argument's arrow demands. -/
theorem feed_sPoint {A : Type u} (F G X : Set (Engeler A)) :
    feed (feed (feed (sPoint A) F) G) X = feed (feed F X) (feed G X) := by
  ext c
  simp only [feed, sPoint, Set.mem_setOf_eq]
  constructor
  · rintro ⟨x, hx, g, hg, f, hf, f', g', x', n, p, d, heq, hnp, hnx, hpg⟩
    injection heq with e1 e2
    injection e2 with e3 e4
    injection e4 with e5 e6
    subst e1
    subst e3
    subst e5
    subst e6
    refine ⟨p, ?_, n, fun b hb => hx b (hnx b hb), hf _ hnp⟩
    intro q hq
    obtain ⟨nq, hnq1, hnq2⟩ := hpg q hq
    exact ⟨nq, fun b hb => hx b (hnq2 b hb), hg _ hnq1⟩
  · rintro ⟨p, hp, n, hn, hF⟩
    obtain ⟨g, xs, hg, hxs, hall⟩ :=
      collect_witnesses G X p fun q hq => by
        obtain ⟨nq, hnq1, hnq2⟩ := hp q hq
        exact ⟨nq, hnq1, hnq2⟩
    refine ⟨n ++ xs, ?_, g, hg, [Engeler.arrow n (Engeler.arrow p c)], ?_,
      [Engeler.arrow n (Engeler.arrow p c)], g, n ++ xs, n, p, c, rfl, by simp,
      fun b hb => List.mem_append.mpr (Or.inl hb), ?_⟩
    · intro b hb
      rcases List.mem_append.mp hb with hb | hb
      · exact hn b hb
      · exact hxs b hb
    · intro b hb
      rw [List.mem_singleton.mp hb]
      exact hF
    · intro q hq
      obtain ⟨nq, hnq1, hnq2⟩ := hall q hq
      exact ⟨nq, hnq1, fun b hb => List.mem_append.mpr (Or.inr (hnq2 b hb))⟩

-- ═══════════════════════════════════════════════════════════════════
-- TERMS TO ELEMENTS, AND REDUCTION TO EQUALITY
-- ═══════════════════════════════════════════════════════════════════

/-- What a term means in the model.

    (A : Type u)                                            [the atoms]
    Comb                                                    [a term of the calculus]
    ℕ → Set (Engeler A)                                     [an environment for the variables]
    →                                                       [together give]
    Set (Engeler A)                                         [an element of the model]

    So: the two constants go to the two sets of arrows, a variable to whatever the environment
    holds for it, and an application to a feeding. -/
def meaning (A : Type u) : Comb → (ℕ → Set (Engeler A)) → Set (Engeler A)
  | Comb.S, _ => sPoint A
  | Comb.K, _ => kPoint A
  | Comb.var i, ρ => ρ i
  | Comb.app f g, ρ => feed (meaning A f ρ) (meaning A g ρ)

/-- One rewriting step does not change what a term means.

    {A : Type u}                                        [the atoms]
    {t u : Comb}                                        [two terms]
    (h : Comb.Step t u)                                 [the first rewrites to the second]
    (ρ : ℕ → Set (Engeler A))                           [an environment]
    meaning A t ρ = meaning A u ρ                       [they mean the same element] -/
theorem meaning_step {A : Type u} {t u : Comb} (h : Comb.Step t u)
    (ρ : ℕ → Set (Engeler A)) : meaning A t ρ = meaning A u ρ := by
  induction h with
  | k x y => exact feed_kPoint A _ _
  | s f g x => exact feed_sPoint _ _ _
  | left b _ ih => exact congrArg (fun Z => feed Z (meaning A b ρ)) ih
  | right a _ ih => exact congrArg (fun Z => feed (meaning A a ρ) Z) ih

/-- Reduction is sent to equality of elements.

@cite vanoosten-2008

    {A : Type u}                                        [the atoms]
    {t u : Comb}                                        [two terms]
    (h : Comb.Reduces t u)                              [the first reaches the second]
    (ρ : ℕ → Set (Engeler A))                           [an environment]
    meaning A t ρ = meaning A u ρ                       [they mean the same element]

    So: soundness. A term and everything it reaches denote one element, so this is a
    denotational semantics of the calculus and not an ordered approximation of one. It is the
    first of the three shapes `theory/DEAD_ENDS.md` left open, terms interpreted as elements
    with reduction sent to an equality. -/
theorem meaning_reduces {A : Type u} {t u : Comb} (h : Comb.Reduces t u)
    (ρ : ℕ → Set (Engeler A)) : meaning A t ρ = meaning A u ρ := by
  induction h with
  | refl => rfl
  | tail _ hstep ih => exact ih.trans (meaning_step hstep ρ)

/-- Feeding an argument to an abstraction binds that argument.

@cite tomita-2023
@cite schonfinkel-1924

    {A : Type u}                                                  [the atoms]
    (i : ℕ)                                                       [which variable]
    (e : Comb)                                                    [the body]
    (ρ : ℕ → Set (Engeler A))                                     [an environment]
    (X : Set (Engeler A))                                         [any element of the model]
    feed (meaning A (Comb.bracket i e) ρ) X                       [feed the abstraction that]
    = meaning A e (Function.update ρ i X)                         [means the body with it bound]

    So: combinatory completeness, for arbitrary elements rather than for meanings of terms.
    The proof is `bracket_correct`'s, with the reductions replaced by the two equations above,
    which is why the conclusion is an equality. -/
theorem feed_meaning_bracket {A : Type u} (i : ℕ) (e : Comb) (ρ : ℕ → Set (Engeler A))
    (X : Set (Engeler A)) :
    feed (meaning A (Comb.bracket i e) ρ) X = meaning A e (Function.update ρ i X) := by
  induction e with
  | S => exact feed_kPoint A _ _
  | K => exact feed_kPoint A _ _
  | var j =>
    by_cases h : j = i
    · subst h
      have hb : Comb.bracket j (Comb.var j) = Comb.I := by simp [Comb.bracket]
      rw [hb]
      simp only [Comb.I, meaning]
      rw [feed_sPoint, feed_kPoint]
      simp
    · have hb : Comb.bracket i (Comb.var j) = Comb.app Comb.K (Comb.var j) := by
        simp [Comb.bracket, h]
      rw [hb]
      simp only [meaning]
      rw [feed_kPoint]
      simp [Function.update, h]
  | app f g ihf ihg =>
    show feed (feed (feed (sPoint A) (meaning A (Comb.bracket i f) ρ))
      (meaning A (Comb.bracket i g) ρ)) X = _
    rw [feed_sPoint, ihf, ihg]
    rfl

/-- The model is combinatorially complete.

@cite engeler-1981
@cite tomita-2023

    {A : Type u}                                                  [the atoms]
    (i : ℕ)                                                       [which variable]
    (e : Comb)                                                    [any term]
    (ρ : ℕ → Set (Engeler A))                                     [an environment]
    ∃ c : Set (Engeler A)                                         [there is an element]
    ∀ X : Set (Engeler A),                                        [such that for every element]
      feed c X = meaning A e (Function.update ρ i X)              [feeding it binds the variable]

    So: Engeler's theorem. The atoms are arbitrary, so no property of them is used, and
    `theory/NOVELTY.md`'s A1 spectator test says this is therefore a result about the
    construction rather than about the algebra the atoms come from. -/
theorem model_combinatory_completeness {A : Type u} (i : ℕ) (e : Comb)
    (ρ : ℕ → Set (Engeler A)) :
    ∃ c : Set (Engeler A), ∀ X : Set (Engeler A),
      feed c X = meaning A e (Function.update ρ i X) :=
  ⟨meaning A (Comb.bracket i e) ρ, fun X => feed_meaning_bracket i e ρ X⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE ATOMS ARE THE ALGEBRA'S SURVIVORS, AND THE MODEL CANNOT READ THEM
-- ═══════════════════════════════════════════════════════════════════

/-- The atoms: the points of the algebra that survive squaring.

    (σ : ℝ)                                     [the parameter]
    {z : Cell // Idempotent σ z}                [the points unchanged by squaring]
    : Type                                      [the atom set the model is built over]

    So: `idempotent_classification` fixes how many there are, four above zero and two at or
    below. -/
abbrev Survivor (σ : ℝ) : Type := {z : Cell // Idempotent σ z}

/-- An atom whose two channels disagree exists exactly above zero.

    (σ : ℝ)                                                       [the parameter]
    ∃ a : Survivor σ                                              [there is an atom]
    upper σ a.1 ≠ lower σ a.1                                     [whose two readings differ]
    ↔                                                             [exactly when]
    0 < σ                                                         [σ is above zero]

    So: the atom set this model is built over carries two independent truth values on the split
    face and one everywhere else. That is `two_boolean_channels_iff_disc_pos` read as a
    statement about the model's atoms. -/
theorem two_channelled_atom_iff_disc_pos (σ : ℝ) :
    (∃ a : Survivor σ, upper σ a.1 ≠ lower σ a.1) ↔ 0 < σ := by
  constructor
  · rintro ⟨a, ha⟩
    exact (two_boolean_channels_iff_disc_pos σ).mp ⟨a.1, a.2, ha⟩
  · intro hσ
    obtain ⟨z, hz, hne⟩ := (two_boolean_channels_iff_disc_pos σ).mpr hσ
    exact ⟨⟨z, hz⟩, hne⟩

/-- Feeding one atom to another gives nothing.

    {A : Type u}                                                [the atoms]
    (a b : A)                                                   [two of them]
    feed {Engeler.atom a} {Engeler.atom b} = ∅                  [feeding gives the empty set]

    So: an atom is not an arrow, so it approximates no function. The atoms sit inside the model
    as inert constants. -/
theorem atom_feed_atom {A : Type u} (a b : A) :
    feed {Engeler.atom a} {Engeler.atom b} = (∅ : Set (Engeler A)) := by
  ext c
  simp only [feed, Set.mem_setOf_eq, Set.mem_singleton_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨l, -, h⟩
  simp at h

/-- Above zero the algebra separates its survivors and the model's application does not.

@cite palmieri-2026

    {σ : ℝ} (hσ : 0 < σ)                                        [the parameter, above zero]
    ∃ a b : Survivor σ                                          [there are two atoms]
    mul σ a.1 b.1 ≠ mul σ a.1 a.1                               [multiplying tells them apart]
    ∀ a b : Survivor σ                                          [while for every two atoms]
    feed {Engeler.atom a} {Engeler.atom b} = ∅                  [feeding gives the empty set]
    ∧                                                           [both together]

    So: the exact strength of this route. The algebra supplies the atoms and σ's sign fixes how
    many there are, but the model's application is blind to everything the algebra does with
    them. `mul σ` on the survivors is AND on two independent channels, and `feed` on the same
    two atoms is the empty set whichever pair is chosen. A construction that read `mul σ` would
    need an arrow whose source is a point of the algebra rather than a finite list of tokens. -/
theorem feed_forgets_the_survivors {σ : ℝ} (hσ : 0 < σ) :
    (∃ a b : Survivor σ, mul σ a.1 b.1 ≠ mul σ a.1 a.1) ∧
      ∀ a b : Survivor σ,
        feed {Engeler.atom a} {Engeler.atom b} = (∅ : Set (Engeler (Survivor σ))) := by
  refine ⟨⟨⟨idemPlus σ, idempotent_idemPlus hσ⟩, ⟨idemMinus σ, idempotent_idemMinus hσ⟩, ?_⟩,
    fun a b => atom_feed_atom a b⟩
  have h0 : mul σ (idemPlus σ) (idemMinus σ) = 0 := by
    refine halves_ext hσ ?_ ?_
    · rw [upper_mul hσ.le, (halves_idemPlus hσ).1, (halves_idemMinus hσ).1, (halves_zero σ).1]
      ring
    · rw [lower_mul hσ.le, (halves_idemPlus hσ).2, (halves_idemMinus hσ).2, (halves_zero σ).2]
      ring
  show mul σ (idemPlus σ) (idemMinus σ) ≠ mul σ (idemPlus σ) (idemPlus σ)
  rw [h0, idempotent_idemPlus hσ]
  exact fun h => (idemPlus_nontrivial σ).1 h.symm

/-- The calculus has a model over the algebra's survivors, at every σ.

@cite engeler-1981

    (σ : ℝ)                                                        [the parameter]
    (i : ℕ)                                                        [which variable]
    (e : Comb)                                                     [any term]
    (ρ : ℕ → Set (Engeler (Survivor σ)))                           [an environment]
    ∃ c : Set (Engeler (Survivor σ))                               [there is an element]
    ∀ X, feed c X = meaning (Survivor σ) e (Function.update ρ i X) [binding the variable]

    So: the model exists on all three faces, because the construction needs a set of atoms and
    nothing more. `feed_forgets_the_survivors` is the companion statement saying what that
    costs. -/
theorem model_over_the_survivors (σ : ℝ) (i : ℕ) (e : Comb)
    (ρ : ℕ → Set (Engeler (Survivor σ))) :
    ∃ c : Set (Engeler (Survivor σ)), ∀ X : Set (Engeler (Survivor σ)),
      feed c X = meaning (Survivor σ) e (Function.update ρ i X) :=
  model_combinatory_completeness i e ρ

end Combinator
end Apeiron
