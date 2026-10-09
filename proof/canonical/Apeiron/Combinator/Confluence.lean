import Apeiron.Combinator.Basic

/-!
# Confluence

@name Confluence
@cite curry-1930
@cite barendregt-1984
@cite vanoosten-2008
@grade proved

Two rewriting rules can apply to the same term at once, and nothing so far says the choice does
not matter. `confluence` says it does not: if a term reaches two others, both reach a common
one. `normalForm_unique` is the consequence anything downstream wants, that a term has at most
one normal form, so "the normal form of `t`" names a single thing.

`../song/the design notes cites this as what makes `prover0`'s normal forms unique.

## Why one step at a time will not do it

Mathlib's `Relation.church_rosser` asks for a strip: if `a` steps to `b` and to `c`, then `b`
reaches some `d` in at most one step while `c` reaches it in any number. `Step` fails that, and
the duplicating rule is why. From `S f g x` with `x` stepping to `x'`:

```
   S f g x  ->  (f x) (g x)          the rule fired
   S f g x  ->  S f g x'  ->  (f x') (g x')
```

Closing the square from `(f x) (g x)` needs two steps, one per copy of `x`, and the strip
allows one. So the relation fed to `church_rosser` is not `Step`.

## Parallel reduction

`Par` contracts any number of redexes at once, each argument reducing on its own. It sits
between one step and many:

```
   Step a b  ->  Par a b  ->  Reduces a b
```

`Par` has the diamond property outright (`Par.diamond`), because the duplicating case now
contracts both copies in a single parallel step. Feeding that to `church_rosser` and collapsing
`ReflTransGen Par` back to `Reduces` gives the result for `Step`.

The base cases are the constants and the variables rather than a general reflexivity rule.
`Par.refl` is then a theorem, and inversion stays cheap: `Par K u` has one constructor that can
produce it, so `u = K` falls out by cases.
-/

namespace Apeiron
namespace Combinator
namespace Comb

/-- Contracting any number of redexes in one go.

    Par : Comb → Comb → Prop      [one term reduces to another, all at once]

    So: every argument reduces on its own, and a redex may be contracted in the same breath as
    the terms inside it. The duplicating rule contracts both copies of its last argument
    together, which is what one step cannot do. -/
inductive Par : Comb → Comb → Prop where
  /-- The duplicating constant reduces to itself. -/
  | S : Par S S
  /-- The discarding constant reduces to itself. -/
  | K : Par K K
  /-- A variable reduces to itself. -/
  | var (i : ℕ) : Par (var i) (var i)
  /-- Both halves of an application reduce. -/
  | app {f f' g g' : Comb} : Par f f' → Par g g' → Par (app f g) (app f' g')
  /-- The discarding rule, with the kept argument reducing too. -/
  | k {x x' : Comb} (y : Comb) : Par x x' → Par (app (app K x) y) x'
  /-- The duplicating rule, with all three arguments reducing. -/
  | s {f f' g g' x x' : Comb} : Par f f' → Par g g' → Par x x' →
      Par (app (app (app S f) g) x) (app (app f' x') (app g' x'))

/-- Every term reduces to itself.

    (a : Comb)          [any term]
    Par a a             [it reduces to itself in a parallel step] -/
theorem Par.refl (a : Comb) : Par a a := by
  induction a with
  | S => exact Par.S
  | K => exact Par.K
  | var i => exact Par.var i
  | app f g ihf ihg => exact Par.app ihf ihg

/-- The discarding constant reduces only to itself.

    {u : Comb}          [any term]
    Par K u        [the discarding constant reduces to it]
    u = K          [so it is the discarding constant] -/
theorem par_K_inv {u : Comb} (h : Par K u) : u = K := by cases h; rfl

/-- The duplicating constant reduces only to itself.

    {u : Comb}          [any term]
    Par S u        [the duplicating constant reduces to it]
    u = S          [so it is the duplicating constant] -/
theorem par_S_inv {u : Comb} (h : Par S u) : u = S := by cases h; rfl

/-- Reducing `K x` gives `K` applied to something `x` reduces to.

    {x u : Comb}                              [two terms]
    Par (app K x) u                      [the partial application reduces to u]
    ∃ x', u = app K x' ∧ Par x x'        [so u has that shape] -/
theorem par_appK_inv {x u : Comb} (h : Par (app K x) u) :
    ∃ x', u = app K x' ∧ Par x x' := by
  cases h with
  | app hf hg => exact ⟨_, by rw [par_K_inv hf], hg⟩

/-- Reducing `S f` gives `S` applied to something `f` reduces to.

    {f u : Comb}                              [two terms]
    Par (app S f) u                      [the partial application reduces to u]
    ∃ f', u = app S f' ∧ Par f f'        [so u has that shape] -/
theorem par_appS_inv {f u : Comb} (h : Par (app S f) u) :
    ∃ f', u = app S f' ∧ Par f f' := by
  cases h with
  | app hf hg => exact ⟨_, by rw [par_S_inv hf], hg⟩

/-- Reducing `S f g` gives `S` applied to reducts of both.

    {f g u : Comb}                                              [three terms]
    Par (app (app S f) g) u                                [it reduces to u]
    ∃ f' g', u = app (app S f') g' ∧ Par f f' ∧ Par g g'   [so u has that shape] -/
theorem par_appSS_inv {f g u : Comb} (h : Par (app (app S f) g) u) :
    ∃ f' g', u = app (app S f') g' ∧ Par f f' ∧ Par g g' := by
  cases h with
  | app hf hg =>
    obtain ⟨f', hf', hff⟩ := par_appS_inv hf
    exact ⟨f', _, by rw [hf'], hff, hg⟩

/-- Parallel reduction has the diamond property.

    {a b : Comb} (h1 : Par a b)          [a term and something it reduces to]
    ∀ c : Comb, Par a c → ...            [for every other reduct, if it is one, then]
    ∃ d, Par b d ∧ Par c d               [the two reduce to a common one]

    So: the square closes in one parallel step on each side. This is the property `Step` does
    not have, and having it is the whole reason `Par` exists. The two cases that carry the
    proof are the ones where a rule fires against a congruence: the discarding rule against an
    application, and the duplicating rule against one, where both copies of the shared argument
    must be contracted at once. -/
theorem par_diamond_forall {a b : Comb} (h1 : Par a b) :
    ∀ c : Comb, Par a c → ∃ d, Par b d ∧ Par c d := by
  induction h1 with
  | S => intro c h2; cases h2; exact ⟨S, Par.S, Par.S⟩
  | K => intro c h2; cases h2; exact ⟨K, Par.K, Par.K⟩
  | var i => intro c h2; cases h2; exact ⟨var i, Par.var i, Par.var i⟩
  | @app f f' g g' hf hg ihf ihg =>
    intro c h2
    cases h2 with
    | app hf2 hg2 =>
      obtain ⟨df, hdf1, hdf2⟩ := ihf _ hf2
      obtain ⟨dg, hdg1, hdg2⟩ := ihg _ hg2
      exact ⟨app df dg, Par.app hdf1 hdg1, Par.app hdf2 hdg2⟩
    | @k x x' y hx =>
      obtain ⟨x1, hf'eq, hxx1⟩ := par_appK_inv hf
      subst hf'eq
      obtain ⟨e, he1, he2⟩ := ihf _ (Par.app Par.K hx)
      obtain ⟨x2, heq2, hx1x2⟩ := par_appK_inv he1
      obtain ⟨x3, heq3, hx'x3⟩ := par_appK_inv he2
      have hxe : x2 = x3 := by
        have h := heq2.symm.trans heq3
        injection h
      exact ⟨x2, Par.k g' hx1x2, by rw [hxe]; exact hx'x3⟩
    | @s f0 f0' g0 g0' x0 x0' hf2 hg2 hx2 =>
      obtain ⟨f1, g1, hf'eq, hff1, hgg1⟩ := par_appSS_inv hf
      subst hf'eq
      obtain ⟨e, he1, he2⟩ := ihf _ (Par.app (Par.app Par.S hf2) hg2)
      obtain ⟨fd, gd, heqd, hf1d, hg1d⟩ := par_appSS_inv he1
      obtain ⟨fe, ge, heqe, hf0e, hg0e⟩ := par_appSS_inv he2
      have h := heqd.symm.trans heqe
      injection h with hleft hg'eq
      injection hleft with _ hf'eq2
      subst hf'eq2
      subst hg'eq
      obtain ⟨xd, hxd1, hxd2⟩ := ihg _ hx2
      exact ⟨app (app fd xd) (app gd xd), Par.s hf1d hg1d hxd1,
        Par.app (Par.app hf0e hxd2) (Par.app hg0e hxd2)⟩
  | @k x x' y hx ih =>
    intro c h2
    cases h2 with
    | app hf2 hg2 =>
      obtain ⟨x2, hx2eq, hxx2⟩ := par_appK_inv hf2
      obtain ⟨d, hd1, hd2⟩ := ih _ hxx2
      exact ⟨d, hd1, by rw [hx2eq]; exact Par.k _ hd2⟩
    | k y2 hx2 => exact ih _ hx2
  | @s f f' g g' x x' hf hg hx ihf ihg ihx =>
    intro c h2
    cases h2 with
    | app hf2 hg2 =>
      obtain ⟨f1, g1, hfeq, hff1, hgg1⟩ := par_appSS_inv hf2
      obtain ⟨fd, hfd1, hfd2⟩ := ihf _ hff1
      obtain ⟨gd, hgd1, hgd2⟩ := ihg _ hgg1
      obtain ⟨xd, hxd1, hxd2⟩ := ihx _ hg2
      refine ⟨app (app fd xd) (app gd xd),
        Par.app (Par.app hfd1 hxd1) (Par.app hgd1 hxd1), ?_⟩
      rw [hfeq]
      exact Par.s hfd2 hgd2 hxd2
    | s hf2 hg2 hx2 =>
      obtain ⟨fd, hfd1, hfd2⟩ := ihf _ hf2
      obtain ⟨gd, hgd1, hgd2⟩ := ihg _ hg2
      obtain ⟨xd, hxd1, hxd2⟩ := ihx _ hx2
      exact ⟨app (app fd xd) (app gd xd),
        Par.app (Par.app hfd1 hxd1) (Par.app hgd1 hxd1),
        Par.app (Par.app hfd2 hxd2) (Par.app hgd2 hxd2)⟩

/-- The diamond property, with both reductions given.

    {a b c : Comb}                       [three terms]
    (h1 : Par a b) (h2 : Par a c)        [the first reduces to each of the others]
    ∃ d, Par b d ∧ Par c d               [and those two reduce to a common one] -/
theorem par_diamond {a b c : Comb} (h1 : Par a b) (h2 : Par a c) :
    ∃ d, Par b d ∧ Par c d :=
  par_diamond_forall h1 c h2

-- ═══════════════════════════════════════════════════════════════════
-- BACK TO ONE STEP AT A TIME
-- ═══════════════════════════════════════════════════════════════════

/-- One step is a parallel step.

    {a b : Comb}            [two terms]
    Step a b                [one rewrites to the other in a single step]
    Par a b                 [so it reduces to it in one parallel step] -/
theorem par_of_step {a b : Comb} (h : Step a b) : Par a b := by
  induction h with
  | k x y => exact Par.k y (Par.refl x)
  | s f g x => exact Par.s (Par.refl f) (Par.refl g) (Par.refl x)
  | left b _ ih => exact Par.app ih (Par.refl b)
  | right a _ ih => exact Par.app (Par.refl a) ih

/-- A parallel step is some number of ordinary ones.

    {a b : Comb}            [two terms]
    Par a b                 [one reduces to the other in a parallel step]
    Reduces a b             [so it reaches it one step at a time] -/
theorem reduces_of_par {a b : Comb} (h : Par a b) : Reduces a b := by
  induction h with
  | S => exact Reduces.refl _
  | K => exact Reduces.refl _
  | var i => exact Reduces.refl _
  | @app f f' g g' _ _ ihf ihg =>
    exact (Reduces.left g ihf).trans (Reduces.right f' ihg)
  | @k x x' y _ ih => exact (Reduces.single (Step.k x y)).trans ih
  | @s f f' g g' x x' _ _ _ ihf ihg ihx =>
    refine (Reduces.single (Step.s f g x)).trans ?_
    refine (Reduces.left (app g x) ((Reduces.left x ihf).trans (Reduces.right f' ihx))).trans ?_
    exact Reduces.right (app f' x') ((Reduces.left x ihg).trans (Reduces.right g' ihx))

/-- Reaching by ordinary steps and reaching by parallel ones are the same relation.

    {a b : Comb}                              [two terms]
    Reduces a b                               [one reaches the other by single steps]
    ↔                                         [exactly when]
    Relation.ReflTransGen Par a b             [it reaches it by parallel ones] -/
theorem reduces_iff_parStar {a b : Comb} :
    Reduces a b ↔ Relation.ReflTransGen Par a b := by
  constructor
  · exact fun h => h.mono (fun _ _ => par_of_step)
  · intro h
    induction h with
    | refl => exact Reduces.refl _
    | tail _ hstep ih => exact ih.trans (reduces_of_par hstep)

-- ═══════════════════════════════════════════════════════════════════
-- CONFLUENCE, AND WHAT IT BUYS
-- ═══════════════════════════════════════════════════════════════════

/-- The order the rules fire in does not matter.

    {a b c : Comb}                            [three terms]
    Reduces a b                               [the first reaches the second]
    Reduces a c                               [and also the third]
    ∃ d, Reduces b d ∧ Reduces c d            [and those two reach a common fourth]

    So: Church-Rosser for these two rules. Two rewriting orders that diverge can always be
    brought back together. -/
theorem confluence {a b c : Comb} (hab : Reduces a b) (hac : Reduces a c) :
    ∃ d, Reduces b d ∧ Reduces c d := by
  have hj : Relation.Join (Relation.ReflTransGen Par) b c :=
    Relation.church_rosser
      (fun _ _ _ h1 h2 => by
        obtain ⟨d, hd1, hd2⟩ := par_diamond h1 h2
        exact ⟨d, Relation.ReflGen.single hd1, Relation.ReflTransGen.single hd2⟩)
      (reduces_iff_parStar.mp hab) (reduces_iff_parStar.mp hac)
  obtain ⟨d, hbd, hcd⟩ := hj
  exact ⟨d, reduces_iff_parStar.mpr hbd, reduces_iff_parStar.mpr hcd⟩

/-- A term nothing rewrites: no rule applies anywhere inside it.

    (a : Comb)                   [a term]
    ¬ ∃ b, Step a b              [nothing it rewrites to] -/
def NormalForm (a : Comb) : Prop := ¬ ∃ b, Step a b

/-- A term nothing rewrites reaches only itself.

    {a b : Comb}                 [two terms]
    NormalForm a                 [nothing rewrites the first]
    Reduces a b                  [and it reaches the second]
    a = b                        [so they are the same term] -/
theorem eq_of_reduces_normalForm {a b : Comb} (hn : NormalForm a) (h : Reduces a b) : a = b := by
  rcases Relation.ReflTransGen.cases_head h with heq | ⟨c, hstep, _⟩
  · exact heq
  · exact absurd ⟨c, hstep⟩ hn

/-- A term has at most one normal form.

    {t a b : Comb}               [a term and two others]
    Reduces t a                  [it reaches the first]
    Reduces t b                  [and the second]
    NormalForm a                 [nothing rewrites the first]
    NormalForm b                 [nothing rewrites the second]
    a = b                        [so they are the same term]

    So: "the normal form of `t`" names one thing when it names anything. This is what
    `../song/the design notes cites confluence for. -/
theorem normalForm_unique {t a b : Comb} (ha : Reduces t a) (hb : Reduces t b)
    (hna : NormalForm a) (hnb : NormalForm b) : a = b := by
  obtain ⟨d, had, hbd⟩ := confluence ha hb
  rw [eq_of_reduces_normalForm hna had, eq_of_reduces_normalForm hnb hbd]

end Comb
end Combinator
end Apeiron
