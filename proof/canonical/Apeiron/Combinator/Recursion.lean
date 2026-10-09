import Apeiron.Combinator.Verdict
import Apeiron.Combinator.Omega

/-!
# Two facts about the FUEL semantics: a false floor, and a discarding fixed point

@name Recursion
@grade proved

Two facts about `song`'s `prover0` FUEL semantics, ported from
`../sanguine-ref/Substrate/StationaryPhase/RecursionFloor.lean`. Both are about the bottom-up
evaluator `Combinator/Verdict.lean` adds, `reduceDeep` and its iterate `normalize`, and neither
reads anything into recursion depth beyond what the evaluator itself computes; the predecessor's
file carries a reading about a universe recursion depth of two, built on a quaternion double
cover, and none of that is ported here.

`fake_floor` says the bottom-up evaluator can stabilise on a term that still holds a redex, so
`hasRedex`, not "did the output change", is what a host must check to know it has a genuine
floor. `fix_normalises_for_const` says the SKI fixed-point combinator does not diverge for every
function fed to it: a discarding function closes.

## Why `fake_floor` is a two-pass fact here, not a one-pass one

The predecessor's `Comb` carries `I` as a primitive constructor with its own one-step rule,
`Step (App I x) x`. `Apeiron.Combinator.Comb` has no such constructor: `Combinator/Basic.lean`
defines `I := app (app S K) K`, so `I x` is itself an `S`-redex that a single `reduceDeep` pass
only partly discharges, landing on `app (app K x) (app K x)` rather than `x` directly. Chasing
this by hand through `omega`'s one head step gives a genuine period-two orbit rather than a
period-one fixed point: `reduceDeep fakeFloor` is a different term from `fakeFloor`, and
`reduceDeep` of THAT is `fakeFloor` again. `fake_floor` below states the fact this evaluator
has, checked by computation (`rfl` on both sides), rather than forcing the predecessor's
one-pass shape onto a term where it does not hold.
-/

namespace Apeiron
namespace Combinator
namespace Comb

/- ── a bottom-up reducer's fixed point that still holds a live redex ── -/

/-- `I` applied to the looping term's own duplicator: the left half of `omega`'s one head step.

    fakeFloorArg                 [the name for this term]
    app I selfApp                [the identity applied to `S I I`]
    : Comb                       [a term] -/
def fakeFloorArg : Comb := app I selfApp

/-- `omega`'s one head step: both halves the same term.

    fakeFloor                              [the name for this term]
    app fakeFloorArg fakeFloorArg          [the same term applied to itself]
    : Comb                                 [a term] -/
def fakeFloor : Comb := app fakeFloorArg fakeFloorArg

/-- One `I selfApp` redex, half-fired: the discarding rule's own two halves, one step short of
    `selfApp` since `I` is compound here.

    fakeFloorHalf                                          [the name for this term]
    app (app K selfApp) (app K selfApp)                    [the discarding rule applied to
                                                            `selfApp` twice over]
    : Comb                                                 [a term] -/
def fakeFloorHalf : Comb := app (app K selfApp) (app K selfApp)

/-- The one bottom-up pass's own output on `fakeFloor`: both halves half-fired.

    fakeFloorNext                              [the name for this term]
    app fakeFloorHalf fakeFloorHalf            [`fakeFloorHalf` applied to itself]
    : Comb                                     [a term] -/
def fakeFloorNext : Comb := app fakeFloorHalf fakeFloorHalf

/-- The bottom-up reducer's first output on the non-terminating term is `fakeFloor`, by
    computation.

    reduceDeep omega = fakeFloor      [one bottom-up pass over `omega` lands here]

    `Comb` [a term: two constants, and one term applied to another] -/
theorem reduceDeep_omega_eq_fakeFloor : reduceDeep omega = fakeFloor := rfl

/-- One more pass over `fakeFloor` reaches `fakeFloorNext`, and one more after that returns to
    `fakeFloor`: the period-two orbit, both by computation.

    reduceDeep fakeFloor = fakeFloorNext ∧ reduceDeep fakeFloorNext = fakeFloor
                                                           [each is the other's next pass]

    `Comb` [a term: two constants, and one term applied to another]
    `∧` [both are true] -/
theorem reduceDeep_fakeFloor_cycle :
    reduceDeep fakeFloor = fakeFloorNext ∧ reduceDeep fakeFloorNext = fakeFloor :=
  ⟨rfl, rfl⟩

/-- **The fake floor.** Two bottom-up passes over `fakeFloor` return `fakeFloor`, so a host
    comparing successive outputs at even fuel would see no change and could mistake it for a
    genuine floor. It is not one: a real contextual step still fires inside it, one of the two
    `I selfApp` redexes the first pass only partly discharged.

    reduceDeep (reduceDeep fakeFloor) = fakeFloor        [stable after two passes]
    ∧                                                     [and, in the same breath]
    Step fakeFloor (app fakeFloorHalf fakeFloorArg)       [a genuine step still fires: the
                                                           left half's own redex]

    `Comb` [a term: two constants, and one term applied to another]
    `Step` [one term rewrites to another in a single step]
    `app` [one term applied to another]
    `∧` [both are true]

    So: `hasRedex`, not "the output stopped changing across one pass" and not even "across two",
    is what a host must check to know it has reached a genuine floor. This is why
    `Combinator/Verdict.lean`'s `verdict` checks `hasRedex (normalize fuel a)` rather than
    comparing consecutive `normalize` outputs. -/
theorem fake_floor :
    reduceDeep (reduceDeep fakeFloor) = fakeFloor ∧
      Step fakeFloor (app fakeFloorHalf fakeFloorArg) :=
  ⟨rfl, Step.left fakeFloorArg (Step.s K K selfApp)⟩

/-- `fakeFloor` is never a normal form, at either phase of the two-cycle: the step witness in
    `fake_floor` is a redex at `fakeFloor` itself, and it recurs every two passes forever, so
    `normalize` never lands `omega` on a genuine floor.

    ¬ NormalForm fakeFloor          [something still rewrites it]

    `Comb` [a term: two constants, and one term applied to another]
    `NormalForm` [no rule applies anywhere inside the term, so nothing rewrites it]
    `¬` [this is not true] -/
theorem fakeFloor_not_normalForm : ¬ NormalForm fakeFloor :=
  fun h => h ⟨_, fake_floor.2⟩

/- ── the fixed-point combinator: a discarding function closes, a reusing one loops ── -/

/-- `λx. f (x x)`, bracket-abstracted over the variable named `0`.

    (f : Comb) : Comb                [a term, to the term that applies f to its argument's
                                      self-application]

    `Comb` [a term: two constants, and one term applied to another] -/
def selfDup (f : Comb) : Comb := bracket 0 (app f (app (var 0) (var 0)))

/-- The SKI fixed-point combinator applied to `f`: `selfDup f` fed to itself.

    (f : Comb) : Comb                [a term, to `Y f`] -/
def fix (f : Comb) : Comb := app (selfDup f) (selfDup f)

/-- `fix f` regenerates its own call, Curry's `Y f = f (Y f)`, for any `f` free of the
    variable `selfDup` abstracts over.

    (f : Comb) (hf : hasVar 0 f = false)          [a term holding no occurrence of variable 0]
    Reduces (fix f) (app f (fix f))                [`fix f` reaches `f` applied to `fix f`]

    `Comb` [a term: two constants, and one term applied to another] -/
theorem fix_unfolds (f : Comb) (hf : hasVar 0 f = false) :
    Reduces (fix f) (app f (fix f)) := by
  have h := bracket_correct 0 (app f (app (var 0) (var 0))) (selfDup f)
  have hs : subst 0 (app f (app (var 0) (var 0))) (selfDup f) = app f (fix f) := by
    simp [subst, subst_of_not_hasVar hf, fix]
  rw [hs] at h
  exact h

/-- **A discarding `f` closes `fix f`.** `f := app K K`, the function that ignores its
    argument and always returns `K`, is `Var`-free, so `fix_unfolds` applies: `fix f` reaches
    `f (fix f)`, and `f (fix f) = (K K)(fix f)` fires the discarding rule immediately, landing
    on the genuine normal form `K` in one more step. So "the fixed-point combinator diverges
    for every `f`" is false as stated: divergence needs `f` to REUSE its argument (the `f = I`
    case, already `Combinator/Omega.lean`'s `omega`, structurally), not the fixed-point shape
    on its own.

    Reduces (fix (app K K)) K ∧ NormalForm K       [the discarding case reaches a genuine
                                                     floor, which holds no further redex]

    `Comb` [a term: two constants, and one term applied to another]
    `NormalForm` [no rule applies anywhere inside the term, so nothing rewrites it]
    `Reduces` [one term reaches another by any number of steps]
    `app` [one term applied to another]
    `∧` [both are true] -/
theorem fix_normalises_for_const : Reduces (fix (app K K)) K ∧ NormalForm K := by
  refine ⟨?_, fun ⟨_, hstep⟩ => nomatch hstep⟩
  have h1 := fix_unfolds (app K K) rfl
  have h2 : Step (app (app K K) (fix (app K K))) K := Step.k K (fix (app K K))
  exact h1.trans (Reduces.single h2)

end Comb
end Combinator
end Apeiron
