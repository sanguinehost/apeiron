import Apeiron.Combinator.Confluence

/-!
# The term that does not settle

@name Non-termination
@cite curry-1930
@cite barendregt-1984
@grade proved

`S I I (S I I)` has no normal form. Confluence says a term has *at most* one normal form; this
says one particular term has none, so the two together are why a normalise-and-compare procedure
is partial rather than a decision procedure, and why an exhaustion verdict is honest rather than
a failure.

`../song/src/prover/ring/prover0.slang` asserts this as `law omega_fuel: s i i (s i i) never
lands`, and its `FUEL` verdict is the third outcome its `normalize` reports.

## How it is proved

Not by exhibiting a loop. A loop is a reduction that returns, and on its own it leaves open that
some *other* reduction path terminates. What is needed is an invariant carried by every reachable
term, and the one used here is:

```
   Loops t   :=   t = app X Y  with  X ⟶* S I I  and  Y ⟶* S I I
```

Three facts close it. `S I I` is a normal form, since the duplicating rule needs three arguments
and has two. `Loops` is preserved by every single step. And no term satisfying `Loops` is a normal
form. So every term reachable from the looping term satisfies `Loops`, and none of them is normal.

The step-preservation case for the duplicating rule is where the shape lemmas earn their place: a
reduct of `S f g` is again `S f' g'`, so a head redex can only fire with `f` and `g` already on
their way to the identity.
-/

namespace Apeiron
namespace Combinator
namespace Comb

/-- Reducing `K x` leaves the head alone.

    {x t : Comb}                      [a term and something the application reaches]
    Reduces (app K x) t               [K applied to it reaches t]
    ∃ x', t = app K x'                [so t is K applied to something]

    So: `K` with one argument has no redex at the head, so every step happens inside the
    argument. -/
theorem reduces_appK {x t : Comb} (h : Reduces (app K x) t) : ∃ x', t = app K x' := by
  induction h with
  | refl => exact ⟨x, rfl⟩
  | @tail b c _ hbc ih =>
    obtain ⟨x', rfl⟩ := ih
    cases hbc with
    | left _ hs => nomatch hs
    | right _ hs => exact ⟨_, rfl⟩

/-- Reducing `S f g` leaves the head and the shape alone.

    {f g t : Comb}                                       [two terms and a reduct]
    Reduces (app (app S f) g) t                          [S applied to two things reaches t]
    ∃ f' g', t = app (app S f') g'                       [there is at least one such pair]
      ∧ Reduces f f' ∧ Reduces g g'                      [and both are reached from the originals]

    So: the duplicating rule needs three arguments and this has two, so no redex fires at the
    head and both arguments reduce on their own. -/
theorem reduces_appSS {f g t : Comb} (h : Reduces (app (app S f) g) t) :
    ∃ f' g', t = app (app S f') g' ∧ Reduces f f' ∧ Reduces g g' := by
  induction h with
  | refl => exact ⟨f, g, rfl, Reduces.refl _, Reduces.refl _⟩
  | @tail b c _ hbc ih =>
    obtain ⟨f', g', rfl, hf, hg⟩ := ih
    cases hbc with
    | left _ hs =>
      cases hs with
      | left _ hs' => nomatch hs'
      | right _ hs' =>
        exact ⟨_, g', rfl, hf.trans (Reduces.single hs'), hg⟩
    | right _ hs =>
      exact ⟨f', _, rfl, hf, hg.trans (Reduces.single hs)⟩

/-- `S I I`, the half of the looping term.

    selfApp                         [the term S I I]
    : Comb                          [a term] -/
def selfApp : Comb := app (app S I) I

/-- `S I I (S I I)`, the looping term.

    omega                           [the term that does not settle]
    : Comb                          [a term] -/
def omega : Comb := app selfApp selfApp

/-- `S I I` is a normal form.

    NormalForm selfApp              [nothing rewrites S I I]

    So: the duplicating rule needs three arguments and this has two, and `I` is `S K K`, which is
    also two short. -/
theorem selfApp_normalForm : NormalForm selfApp := by
  rintro ⟨t, hs⟩
  cases hs with
  | left _ hs' =>
    cases hs' with
    | left _ hs'' => nomatch hs''
    | right _ hs'' => nomatch hs''
  | right _ hs' => nomatch hs'

/-- The invariant every term reachable from the looping term carries.

    (t : Comb)                                                   [a term]
    ∃ X Y, t = app X Y ∧ Reduces X selfApp ∧ Reduces Y selfApp   [both halves head for S I I] -/
def Loops (t : Comb) : Prop :=
  ∃ X Y, t = app X Y ∧ Reduces X selfApp ∧ Reduces Y selfApp

/-- No term carrying the invariant is a normal form.

    {t : Comb} (h : Loops t)          [a term carrying the invariant]
    ¬ NormalForm t                    [something rewrites it]

    So: either the left half is already `S I I`, and then the whole term is the duplicating rule
    with three arguments, or the left half still has a step to take and that step is a step of the
    whole. -/
theorem Loops_not_normalForm {t : Comb} (h : Loops t) : ¬ NormalForm t := by
  obtain ⟨X, Y, rfl, hX, _⟩ := h
  rcases Relation.ReflTransGen.cases_head hX with heq | ⟨c, hstep, _⟩
  · subst heq
    exact fun hn => hn ⟨_, Step.s I I Y⟩
  · exact fun hn => hn ⟨_, Step.left Y hstep⟩

/-- The invariant survives every single step.

    {t t' : Comb} (h : Loops t) (hs : Step t t')     [a term carrying it, and a step]
    Loops t'                                         [the result carries it too]

    So: a congruence step keeps both halves heading for `S I I`, because `S I I` is a normal form
    and confluence then forces the reduct back onto the same target. The discarding rule cannot
    fire, because a reduct of `K x` is `K` applied to something and never `S I I`. The duplicating
    rule can fire only when both of its first two arguments already head for the identity, by the
    shape lemma, and then both halves of the result head for `S I I` through the third. -/
theorem Loops_step {t t' : Comb} (h : Loops t) (hs : Step t t') : Loops t' := by
  obtain ⟨X, Y, rfl, hX, hY⟩ := h
  cases hs with
  | @left a a' b hstep =>
    obtain ⟨d, hd1, hd2⟩ := confluence (Reduces.single hstep) hX
    rw [← eq_of_reduces_normalForm selfApp_normalForm hd2] at hd1
    exact ⟨a', Y, rfl, hd1, hY⟩
  | @right a b b' hstep =>
    obtain ⟨d, hd1, hd2⟩ := confluence (Reduces.single hstep) hY
    rw [← eq_of_reduces_normalForm selfApp_normalForm hd2] at hd1
    exact ⟨X, b', rfl, hX, hd1⟩
  | k x y =>
    obtain ⟨x', hx'⟩ := reduces_appK hX
    simp only [selfApp] at hx'
    nomatch hx'
  | s f g _ =>
    obtain ⟨f', g', heq, hf, hg⟩ := reduces_appSS hX
    simp only [selfApp] at heq
    obtain ⟨hl, hr⟩ := Comb.app.inj heq
    obtain ⟨_, hfI⟩ := Comb.app.inj hl
    subst hfI
    subst hr
    refine ⟨app f Y, app g Y, rfl, ?_, ?_⟩
    · exact ((Reduces.left Y hf).trans (I_app Y)).trans hY
    · exact ((Reduces.left Y hg).trans (I_app Y)).trans hY

/-- The looping term carries the invariant.

    Loops omega                     [both its halves are already S I I] -/
theorem Loops_omega : Loops omega :=
  ⟨selfApp, selfApp, rfl, Reduces.refl _, Reduces.refl _⟩

/-- Everything the looping term reaches carries the invariant.

    {t : Comb} (h : Reduces omega t)     [anything the looping term reaches]
    Loops t                              [carries the invariant] -/
theorem Loops_of_reduces {t : Comb} (h : Reduces omega t) : Loops t := by
  induction h with
  | refl => exact Loops_omega
  | tail _ hstep ih => exact Loops_step ih hstep

/-- The looping term has no normal form.

    {t : Comb}                          [any term]
    Reduces omega t                     [that the looping term reaches]
    ¬ NormalForm t                      [is not a normal form]

    So: `../song/src/prover/ring/prover0.slang`'s `law omega_fuel`, at the strength it states.
    Not a loop exhibited but an invariant carried: no reduction path from this term reaches a
    normal form, whichever redex is chosen at each step. With `normalForm_unique`, which says a
    term has at most one normal form, this is why normalise-and-compare is a sound partial
    procedure and not a decision procedure, and why an exhaustion verdict is a third honest
    answer rather than a failure. -/
theorem omega_no_normalForm {t : Comb} (h : Reduces omega t) : ¬ NormalForm t :=
  Loops_not_normalForm (Loops_of_reduces h)

end Comb
end Combinator
end Apeiron
