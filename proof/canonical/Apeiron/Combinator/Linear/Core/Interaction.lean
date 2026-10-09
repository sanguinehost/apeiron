import Apeiron.Trichotomy.Trace
import Apeiron.Combinator.RouteAnyRing
import Mathlib.Data.Nat.Pairing
import Mathlib.Logic.Equiv.Nat

/-!
# The concrete interaction situation: wires on the counting numbers, weighted by the algebra

@name Interaction
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite haghverdi-scott-2010
@cite joyal-street-verity-1996
@grade proved

`Combinator/Wires.lean` shows the calculus cannot live on a two-dimensional carrier and builds
one that can. This file supplies the rest of Abramsky, Haghverdi and Scott's concrete situation
over that carrier: partial injections on the counting numbers, weighted by points of the
algebra, with both retractions built from pairings rather than assumed.

The two things a Geometry of Interaction situation needs beyond a traced category are a
reflexive object and an exponential. Neither is taken on trust here:

- `U ⊗ U ◁ U` is the even and odd positions, `Equiv.natSumNatEquivNat`, and the four equations
  `outL_inL`, `outR_inR`, `outL_inR` and `outR_inL` are its retraction.
- `T U ◁ U` with `T X = ℕ × X` is Cantor's pairing, `Nat.pair`, and `uncopy_copy` with
  `uncopy_copy_of_ne` is its retraction. That is the exponential, as countably many disjoint
  copies of the wire inside the wire. It is a construction and not a library object.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_assoc`, `mul_comm` | `rcomp_assoc`, `rcomp_rid_left`, `rcomp_rid_right` |
| `divergent_loop_of_disc_pos` | `weight_blocks_the_loop_iff_disc_nonneg` |
| `normForm_eq_zero_iff_of_neg` | `weight_blocks_the_loop_iff_disc_nonneg` |
| `isUnit_iff_normForm_ne_zero` | `weight_blocks_the_loop_iff_disc_nonneg` |

The wiring is free of σ and says so: a partial injection carries at most one path out of each
position, so composing needs no sum and the retractions are equalities of routes at every
setting. What the sign of σ decides is whether the **weight** along a loop lets the feedback close.
`weight_blocks_the_loop_iff_disc_nonneg` is that statement: there is a loop weight, other than
the unit, that neither dies out nor closes exactly when σ is at zero or above. Below zero
every weight but the unit closes, so the weighted situation is total there; at zero and above it
is a partial trace in Haghverdi and Scott's sense, and the wiring has to supply the termination
the weight will not.

## Why there are no sums here

@cite abramsky-haghverdi-scott-2002

Abramsky, Haghverdi and Scott's particle-style trace is a union over paths. Their carrier is
partial injections, and a partial injection does not branch, so at most one path leaves each
position and the union has at most one term. That is why `rcomp` below needs no sum: the weight
of a composite is one product along one path. `Trichotomy/Trace.lean`'s `feedback` supplies the
other half, the sum over how many times a path goes round a loop, and `loopSum_eq_sum` writes it
out.

## What is built, and what is left

Built: the routes and their monoid laws, the tensor with its two projections, the exponential's
countably many copies, the loop-weight classification, the trace, and two of the three
combinators.

`traces_functional` is the theorem an earlier draft named as owed and did not have. The
particle-style trace is a union over paths of unbounded length, and the question was whether
that union is single-valued. It is, and three disjointness facts are why: a position on the way
in either crosses straight over or enters the loop (`direct_entry_disjoint`), a position in the
loop either goes round again or leaves (`body_leave_disjoint`), and a path therefore leaves at
one depth only (`exit_depth_unique`). All three come from a partial map having one value where
it has any, so the trace is a route in its own right at every setting of the parameter.

With the trace in hand, `plug` is application: run the applied route, leave the even track
alone, send the odd track through the argument, and close the loop. Then:

- `iRoute_app`: `I x = x`. The identity combinator is the swap of the two tracks, so everything
  enters the loop, goes round no times, and leaves. This is `feedback_yank` with an argument
  threaded through it.
- `kRoute_discards`: `K x y = x`. The first application wires `x` onto the even track and leaves
  the odd track empty (`kRoute_app`); the second finds no wire into its own argument and hands
  back what the first stored (`emb_app`).

Neither equation uses a property of the parameter. The weights that arise are the argument's own
and the unit, so they cancel by `one_mul` and `mul_one_right` on every face. What the parameter
decides is `weight_blocks_the_loop_iff_disc_nonneg`, which is about loops that do not close, and
neither `I` nor `K` makes one: both have `body σ ... = rnone`.

**`S` is not built.** It is the combinator that needs its argument twice, so unlike `I` and `K`
it has to use the exponential rather than only the two-track retraction: the third argument goes
into `copy 0` and `copy 1` and the two results are recombined. The routing is writable with what
is here, and the obstacle is not structural: `S x y z = x z (y z)` unfolds into three nested
applications, so its verification is three traces composed rather than one, and the loop for the
inner application is not empty the way `I`'s and `K`'s are. That is the next piece of work and
this file does not claim it.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

/-- A weighted wire map: at most one target for each position, carrying a point as its weight.

@cite abramsky-haghverdi-scott-2002

    ℕ → Option (ℕ × Cell)         [each position goes nowhere, or to one with one weight]

    So: a partial injection on the counting numbers with a weight on each wire. Partial
    injections do not branch, which is what lets composition below multiply one weight along one
    path rather than summing over branches. -/
abbrev Route : Type := RouteAnyRing.Route ℝ

-- The generic route core at the real carrier. `RouteAnyRing.lean` is the single authority for
-- these formulas and their docstrings. The six nullary wires are specialised abbreviations,
-- because their statements carry no parameter for the ring to be inferred from; the operations
-- that take `σ` or a route are exported, so `rw`/`simp` reduce exactly as the generic bodies do.
/-- The route that connects nothing.

    Route    [a weighted wire map]
    rnone    [the route that joins nothing] -/
abbrev rnone : Route := RouteAnyRing.rnone (R := ℝ)

/-- The route that connects every position to itself, with the unit weight.

    Route    [a weighted wire map]
    rid      [the route joining every position to itself] -/
abbrev rid : Route := RouteAnyRing.rid (R := ℝ)

/-- Place a wire in the even positions.

    Route    [a weighted wire map]
    inL      [place a wire in the even positions] -/
abbrev inL : Route := RouteAnyRing.inL (R := ℝ)

/-- Place a wire in the odd positions.

    Route    [a weighted wire map]
    inR      [place a wire in the odd positions] -/
abbrev inR : Route := RouteAnyRing.inR (R := ℝ)

/-- Read the even positions back.

    Route    [a weighted wire map]
    outL     [read the even positions back] -/
abbrev outL : Route := RouteAnyRing.outL (R := ℝ)

/-- Read the odd positions back.

    Route    [a weighted wire map]
    outR     [read the odd positions back] -/
abbrev outR : Route := RouteAnyRing.outR (R := ℝ)
export RouteAnyRing (rcomp rtensor rpow direct entry leave body Traces)

/-- The unit weight on the right leaves a point alone.

    (σ : ℝ) (z : Cell)                    [the parameter and a point]
    mul σ z one = z                       [multiplying by the unit changes nothing] -/
theorem mul_one_right (σ : ℝ) (z : Cell) : mul σ z one = z := by
  rw [mul_comm]
  exact one_mul σ z

/-- The identity route is a left unit for composing.

    (σ : ℝ) (a : Route)                   [the parameter and a route]
    rcomp σ rid a = a                     [going nowhere first changes nothing] -/
theorem rcomp_rid_left (σ : ℝ) (a : Route) : rcomp σ rid a = a := by
  funext i
  cases h : a i with
  | none => simp [rcomp, rid, h]
  | some p => simp [rcomp, rid, h]

/-- The identity route is a right unit for composing.

    (σ : ℝ) (a : Route)                   [the parameter and a route]
    rcomp σ a rid = a                     [going nowhere after changes nothing] -/
theorem rcomp_rid_right (σ : ℝ) (a : Route) : rcomp σ a rid = a := by
  funext i
  cases h : a i with
  | none => simp [rcomp, rid, h]
  | some p => simp [rcomp, rid, h]

/-- Composing routes does not depend on the bracketing.

    (σ : ℝ) (a b c : Route)                                  [the parameter and three routes]
    rcomp σ (rcomp σ a b) c = rcomp σ a (rcomp σ b c)        [either bracketing gives the same]

    So: the weights multiply along the path and `mul_assoc` regroups them. -/
theorem rcomp_assoc (σ : ℝ) (a b c : Route) :
    rcomp σ (rcomp σ a b) c = rcomp σ a (rcomp σ b c) := by
  funext i
  cases h : a i with
  | none => simp [rcomp, h]
  | some p =>
    cases h2 : b p.1 with
    | none => simp [rcomp, h, h2]
    | some q =>
      cases h3 : c q.1 with
      | none => simp [rcomp, h, h2, h3]
      | some r => simp [rcomp, h, h2, h3]; constructor <;> ring

/-- The empty route absorbs on the left.

    (σ : ℝ) (a : Route)                   [the parameter and a route]
    rcomp σ rnone a = rnone               [nothing in, nothing out] -/
theorem rcomp_rnone_left (σ : ℝ) (a : Route) : rcomp σ rnone a = rnone := by
  funext i
  simp [rcomp, rnone]

/-- The empty route absorbs on the right.

    (σ : ℝ) (a : Route)                   [the parameter and a route]
    rcomp σ a rnone = rnone               [nothing out, nothing in] -/
theorem rcomp_rnone_right (σ : ℝ) (a : Route) : rcomp σ a rnone = rnone := by
  funext i
  cases h : a i with
  | none => simp [rcomp, rnone, h]
  | some p => simp [rcomp, rnone, h]

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO WIRES INSIDE ONE WIRE
-- ═══════════════════════════════════════════════════════════════════

/-- Placing in the even positions and reading them back is the identity.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ)                            [the parameter]
    rcomp σ inL outL = rid             [the round trip changes nothing] -/
theorem outL_inL (σ : ℝ) : rcomp σ inL outL = rid := by
  funext i
  simp [rcomp, inL, outL, rid]

/-- Placing in the odd positions and reading them back is the identity.

    (σ : ℝ)                            [the parameter]
    rcomp σ inR outR = rid             [the round trip changes nothing] -/
theorem outR_inR (σ : ℝ) : rcomp σ inR outR = rid := by
  funext i
  simp [rcomp, inR, outR, rid]

/-- The even positions carry nothing that was placed in the odd ones.

    (σ : ℝ)                            [the parameter]
    rcomp σ inR outL = rnone           [no wire crosses] -/
theorem outL_inR (σ : ℝ) : rcomp σ inR outL = rnone := by
  funext i
  simp [rcomp, inR, outL, rnone]

/-- The odd positions carry nothing that was placed in the even ones.

    (σ : ℝ)                            [the parameter]
    rcomp σ inL outR = rnone           [no wire crosses] -/
theorem outR_inL (σ : ℝ) : rcomp σ inL outR = rnone := by
  funext i
  simp [rcomp, inL, outR, rnone]

/-- Reading the first of two parallel routes returns it, and discards the second.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (a b : Route)                                   [the parameter and two routes]
    rcomp σ inL (rcomp σ (rtensor a b) outL) = a            [the left projection is a]

    So: this is the discarding `K` is made of, at the linear level. No feedback is involved and
    no hypothesis on σ is needed: the second route contributes nothing because no wire
    reaches it. -/
theorem tensor_outL (σ : ℝ) (a b : Route) :
    rcomp σ inL (rcomp σ (rtensor a b) outL) = a := by
  funext i
  cases h : a i with
  | none => simp [rcomp, inL, rtensor, outL, h]
  | some p => simp [rcomp, inL, rtensor, outL, h]

/-- Reading the second of two parallel routes returns it, and discards the first.

    (σ : ℝ) (a b : Route)                                   [the parameter and two routes]
    rcomp σ inR (rcomp σ (rtensor a b) outR) = b            [the right projection is b] -/
theorem tensor_outR (σ : ℝ) (a b : Route) :
    rcomp σ inR (rcomp σ (rtensor a b) outR) = b := by
  funext i
  cases h : b i with
  | none => simp [rcomp, inR, rtensor, outR, h]
  | some p => simp [rcomp, inR, rtensor, outR, h]

/-- Nothing crosses between the two parallel routes.

    (σ : ℝ) (a b : Route)                                       [the parameter and two routes]
    rcomp σ inL (rcomp σ (rtensor a b) outR) = rnone            [left in, right out: nothing]
    rcomp σ inR (rcomp σ (rtensor a b) outL) = rnone            [right in, left out: nothing]
    ∧                                                           [both together] -/
theorem tensor_no_crossing (σ : ℝ) (a b : Route) :
    rcomp σ inL (rcomp σ (rtensor a b) outR) = rnone ∧
      rcomp σ inR (rcomp σ (rtensor a b) outL) = rnone := by
  constructor
  · funext i
    cases h : a i with
    | none => simp [rcomp, inL, rtensor, outR, rnone, h]
    | some p => simp [rcomp, inL, rtensor, outR, rnone, h]
  · funext i
    cases h : b i with
    | none => simp [rcomp, inR, rtensor, outL, rnone, h]
    | some p => simp [rcomp, inR, rtensor, outL, rnone, h]

-- ═══════════════════════════════════════════════════════════════════
-- THE EXPONENTIAL: COUNTABLY MANY COPIES, BY CANTOR'S PAIRING
-- ═══════════════════════════════════════════════════════════════════

/-- Place a wire in the nth of countably many copies.

@cite abramsky-haghverdi-scott-2002

    (n : ℕ)                            [which copy]
    Route                              [the wire moved into that copy]

    So: `T X = ℕ × X` made concrete by Cantor's pairing. The exponential of linear logic is
    countably many disjoint copies of the wire sitting inside the wire, and this is the map
    that puts one there. -/
def copy (n : ℕ) : Route := fun i => some (Nat.pair n i, one)

/-- Read the nth copy back.

    (n : ℕ)                            [which copy]
    Route                              [what sits in that copy] -/
def uncopy (n : ℕ) : Route := fun j =>
  if (Nat.unpair j).1 = n then some ((Nat.unpair j).2, one) else none

/-- **The exponential's retraction.** Placing in a copy and reading it back is the identity.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (n : ℕ)                          [the parameter and which copy]
    rcomp σ (copy n) (uncopy n) = rid        [the round trip changes nothing]

    So: `T U ◁ U`, the retraction a Geometry of Interaction situation asks for on top of the
    reflexive object. Cantor's pairing supplies it, so the exponential here is a construction
    rather than an assumed comonad. -/
theorem uncopy_copy (σ : ℝ) (n : ℕ) : rcomp σ (copy n) (uncopy n) = rid := by
  funext i
  simp [rcomp, copy, uncopy, Nat.unpair_pair, rid]

/-- Distinct copies do not overhear each other.

    (σ : ℝ) {n m : ℕ} (h : n ≠ m)                 [the parameter and two different copies]
    rcomp σ (copy n) (uncopy m) = rnone           [reading the wrong copy gives nothing]

    So: the copies are disjoint, which is what makes them countably many separate resources
    rather than one shared one. -/
theorem uncopy_copy_of_ne (σ : ℝ) {n m : ℕ} (h : n ≠ m) :
    rcomp σ (copy n) (uncopy m) = rnone := by
  funext i
  simp [rcomp, copy, uncopy, Nat.unpair_pair, rnone, h]

-- ═══════════════════════════════════════════════════════════════════
-- GOING ROUND THE LOOP, AND WHAT THE WEIGHT DOES TO IT
-- ═══════════════════════════════════════════════════════════════════

/-- A wiring that runs out stays run out.

    (σ : ℝ) (a : Route) {n : ℕ}                      [the parameter, a route, a depth]
    (h : rpow σ a n = rnone)                         [n trips connect nothing]
    rpow σ a (n + 1) = rnone                         [neither does one more]

    So: structural termination. A loop can die because the wiring runs out, at every setting of
    σ, and this is the induction step saying it stays dead. -/
theorem rpow_rnone_succ (σ : ℝ) (a : Route) {n : ℕ} (h : rpow σ a n = rnone) :
    rpow σ a (n + 1) = rnone := by
  rw [rpow, h, rcomp_rnone_right]

/-- **The weight can block a loop exactly at zero and above.**

    (σ : ℝ)                                                   [the parameter]
    ∃ d : Cell                                                [there is a weight]
    d ≠ one                                                   [other than the unit]
    (∀ n : ℕ, cpow σ d n ≠ 0)                                 [that never dies out]
    ¬ ∃ u : Cell, mul σ (one - d) u = one                     [and never closes]
    ∧                                                         [all three together]
    ↔                                                         [exactly when]
    0 ≤ σ                                                     [σ is at zero or above]

    So: the wiring carries the trace at every setting, because a partial injection has at most
    one path and the union over paths has at most one term. σ's sign decides the weight.
    Below zero every weight but the unit closes, so a weighted situation is total there. At zero
    and above there is a weight that neither dies out nor closes, so the weighted trace is
    partial in Haghverdi and Scott's sense and the wiring has to supply the termination the
    weight will not. -/
theorem weight_blocks_the_loop_iff_disc_nonneg (σ : ℝ) :
    (∃ d : Cell, d ≠ one ∧ (∀ n : ℕ, cpow σ d n ≠ 0) ∧
      ¬ ∃ u : Cell, mul σ (one - d) u = one) ↔ 0 ≤ σ := by
  constructor
  · rintro ⟨d, hd1, -, hd3⟩
    by_contra hσ
    refine hd3 ((isUnit_iff_normForm_ne_zero σ (one - d)).mpr fun hn => hd1 ?_)
    exact (sub_eq_zero.mp ((normForm_eq_zero_iff_of_neg (not_le.mp hσ) (one - d)).mp hn)).symm
  · intro hσ
    rcases eq_or_lt_of_le hσ with h | hpos
    · subst h
      have hpt : (one : Cell) - ((1 : ℝ), (-1 : ℝ)) = ((0 : ℝ), (1 : ℝ)) := by
        simp [one]
      refine ⟨((1 : ℝ), (-1 : ℝ)), ?_, ?_, ?_⟩
      · intro hcon
        have h2 : (-1 : ℝ) = 0 := congrArg Prod.snd hcon
        norm_num at h2
      · refine fun n => cpow_ne_zero_of_off_cone ?_ n
        show ((1 : ℝ)) ^ 2 - (0 : ℝ) * (-1 : ℝ) ^ 2 ≠ 0
        norm_num
      · rintro ⟨u, hu⟩
        refine (isUnit_iff_normForm_ne_zero 0 (one - ((1 : ℝ), (-1 : ℝ)))).mp ⟨u, hu⟩ ?_
        rw [hpt]
        show ((0 : ℝ)) ^ 2 - (0 : ℝ) * (1 : ℝ) ^ 2 = 0
        norm_num
    · obtain ⟨d, hne, hcone, hnil⟩ := divergent_loop_of_disc_pos hpos
      exact ⟨d, hne, hnil,
        fun h => (isUnit_iff_normForm_ne_zero σ (one - d)).mp h hcone⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE TRACE: ONE PATH, WHATEVER ITS LENGTH
-- ═══════════════════════════════════════════════════════════════════

/-- What one trip round the loop does, written out.

    (σ : ℝ) (f : Route) (u : ℕ)                    [the parameter, a route, a loop position]
    body σ f u                                     [one trip round]
    outR                                           [kept only if the read lands odd]
    mul σ one                                      [with the weights multiplied]

    So: the definition unfolded once, so that the value of `f` is a subterm the proofs below can
    case on. -/
theorem body_apply (σ : ℝ) (f : Route) (u : ℕ) :
    body σ f u = ((f (Equiv.natSumNatEquivNat (Sum.inr u))).bind fun p =>
      (outR p.1).map fun q => (q.1, mul σ p.2 q.2)).map fun q => (q.1, mul σ one q.2) := rfl

/-- What leaving the loop does, written out.

    (σ : ℝ) (f : Route) (u : ℕ)                    [the parameter, a route, a loop position]
    leave σ f u                                    [the way out]
    outL                                           [kept only if the read lands even]
    mul σ one                                      [with the weights multiplied] -/
theorem leave_apply (σ : ℝ) (f : Route) (u : ℕ) :
    leave σ f u = ((f (Equiv.natSumNatEquivNat (Sum.inr u))).bind fun p =>
      (outL p.1).map fun q => (q.1, mul σ p.2 q.2)).map fun q => (q.1, mul σ one q.2) := rfl

/-- What crossing straight over does, written out.

    (σ : ℝ) (f : Route) (i : ℕ)                    [the parameter, a route, a position]
    direct σ f i                                   [the straight crossing]
    outL                                           [kept only if the read lands even]
    mul σ one                                      [with the weights multiplied] -/
theorem direct_apply (σ : ℝ) (f : Route) (i : ℕ) :
    direct σ f i = ((f (Equiv.natSumNatEquivNat (Sum.inl i))).bind fun p =>
      (outL p.1).map fun q => (q.1, mul σ p.2 q.2)).map fun q => (q.1, mul σ one q.2) := rfl

/-- What entering the loop does, written out.

    (σ : ℝ) (f : Route) (i : ℕ)                    [the parameter, a route, a position]
    entry σ f i                                    [the way in]
    outR                                           [kept only if the read lands odd]
    mul σ one                                      [with the weights multiplied] -/
theorem entry_apply (σ : ℝ) (f : Route) (i : ℕ) :
    entry σ f i = ((f (Equiv.natSumNatEquivNat (Sum.inl i))).bind fun p =>
      (outR p.1).map fun q => (q.1, mul σ p.2 q.2)).map fun q => (q.1, mul σ one q.2) := rfl

/-- A position lands in one half of the wire, so one of the two reads gives nothing.

    (j : ℕ)                                     [a position of the doubled wire]
    ℝ                                           [the real numbers: every number on the line]
    outL j = none ∨ outR j = none               [it is odd, or it is even] -/
theorem outL_outR_disjoint (j : ℕ) : outL j = none ∨ outR j = none := by
  cases hs : Equiv.natSumNatEquivNat.symm j with
  | inl k => right; simp only [outR, RouteAnyRing.outR, hs, Sum.elim_inl]
  | inr k => left; simp only [outL, RouteAnyRing.outL, hs, Sum.elim_inr]

/-- A position in the loop either goes round again or leaves, never both.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (f : Route) (u : ℕ)                            [the parameter, a route, a loop position]
    body σ f u = none ∨ leave σ f u = none                 [at most one of the two is defined]

    So: both read the same single value of `f` at that position, and that value lands either in
    the even half or in the odd half. This is why a partial injection's path does not branch,
    and it is the fact the trace rests on. -/
theorem body_leave_disjoint (σ : ℝ) (f : Route) (u : ℕ) :
    body σ f u = none ∨ leave σ f u = none := by
  rw [body_apply, leave_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inr u)) with
  | none => left; simp
  | some p =>
    rcases outL_outR_disjoint p.1 with h | h
    · right; simp [h]
    · left; simp [h]

/-- A position on the way in either crosses straight over or enters the loop, never both.

    (σ : ℝ) (f : Route) (i : ℕ)                            [the parameter, a route, a position]
    direct σ f i = none ∨ entry σ f i = none               [at most one of the two is defined]

    So: the same argument one step earlier. The straight crossing and the entry into the loop
    read the same single value of `f`. -/
theorem direct_entry_disjoint (σ : ℝ) (f : Route) (i : ℕ) :
    direct σ f i = none ∨ entry σ f i = none := by
  rw [direct_apply, entry_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inl i)) with
  | none => left; simp
  | some p =>
    rcases outL_outR_disjoint p.1 with h | h
    · left; simp [h]
    · right; simp [h]

/-- Going round n plus m times is going round n times and then m times.

    (σ : ℝ) (a : Route) (n m : ℕ)                                     [σ, a route, two counts]
    rpow σ a (n + m) = rcomp σ (rpow σ a n) (rpow σ a m)         [the trips split]

    So: the induction that lets a path be cut at any depth. -/
theorem rpow_split (σ : ℝ) (a : Route) (n m : ℕ) :
    rpow σ a (n + m) = rcomp σ (rpow σ a n) (rpow σ a m) := by
  induction n with
  | zero => rw [Nat.zero_add, rpow, rcomp_rid_left]
  | succ n ih =>
    rw [Nat.succ_add, rpow, ih, rpow, rcomp_assoc]

/-- One trip of a route that connects nothing connects nothing.

    (σ : ℝ) (a : Route) (n i : ℕ)                               [σ, a route, a count, a start]
    (h : a i = none)                                             [the route stops here]
    rpow σ a (n + 1) i = none                                    [so no trip from here lands] -/
theorem rpow_succ_none (σ : ℝ) (a : Route) (n i : ℕ) (h : a i = none) :
    rpow σ a (n + 1) i = none := by
  simp [rpow, rcomp, h]

/-- A composite that lands somewhere came through a first leg that landed somewhere.

    {σ : ℝ} {a b : Route} {i : ℕ}                          [the parameter, two routes, a position]
    (h : rcomp σ a b i ≠ none)                             [the composite goes somewhere]
    ∃ p, a i = some p ∧ b p.1 ≠ none                       [so each leg does] -/
theorem rcomp_ne_none {σ : ℝ} {a b : Route} {i : ℕ} (h : rcomp σ a b i ≠ none) :
    ∃ p, a i = some p ∧ b p.1 ≠ none := by
  cases ha : a i with
  | none => exact absurd (by simp [rcomp, ha]) h
  | some p =>
    refine ⟨p, rfl, fun hb => h ?_⟩
    simp [rcomp, ha, hb]

/-- A path cannot leave the loop at two different depths.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (f : Route) (u : ℕ) {n k : ℕ}                        [the parameter, a route, a start]
    (hn : rcomp σ (rpow σ (body σ f) n) (leave σ f) u ≠ none)    [it leaves after n trips]
    (hm : rcomp σ (rpow σ (body σ f) (n + k + 1))                [and also after more]
      (leave σ f) u ≠ none)
    False                                                        [which cannot happen]

    So: leaving after `n` trips means the loop is undefined at the position reached, and going
    round again means it is defined there. `body_leave_disjoint` says both cannot hold. -/
theorem no_two_exit_depths (σ : ℝ) (f : Route) (u : ℕ) {n k : ℕ}
    (hn : rcomp σ (rpow σ (body σ f) n) (leave σ f) u ≠ none)
    (hm : rcomp σ (rpow σ (body σ f) (n + k + 1)) (leave σ f) u ≠ none) : False := by
  obtain ⟨p, hp, hl⟩ := rcomp_ne_none hn
  obtain ⟨q, hq, -⟩ := rcomp_ne_none hm
  have hsplit : rcomp σ (rpow σ (body σ f) n) (rpow σ (body σ f) (k + 1)) u = some q := by
    rw [← rpow_split]; exact hq
  have hne : rcomp σ (rpow σ (body σ f) n) (rpow σ (body σ f) (k + 1)) u ≠ none := by
    rw [hsplit]; simp
  obtain ⟨p', hp', hb⟩ := rcomp_ne_none hne
  have hpp : p = p' := by rw [hp] at hp'; exact Option.some.inj hp'
  subst hpp
  rw [rpow] at hb
  obtain ⟨r, hr, -⟩ := rcomp_ne_none hb
  rcases body_leave_disjoint σ f p.1 with h | h
  · simp [h] at hr
  · exact hl h

/-- The depth at which a path leaves the loop is unique.

    (σ : ℝ) (f : Route) (u : ℕ) {n m : ℕ}                        [the parameter, a route, a start]
    (hn : rcomp σ (rpow σ (body σ f) n) (leave σ f) u ≠ none)    [it leaves after n trips]
    (hm : rcomp σ (rpow σ (body σ f) m) (leave σ f) u ≠ none)    [and after m trips]
    n = m                                                        [so the two counts agree] -/
theorem exit_depth_unique (σ : ℝ) (f : Route) (u : ℕ) {n m : ℕ}
    (hn : rcomp σ (rpow σ (body σ f) n) (leave σ f) u ≠ none)
    (hm : rcomp σ (rpow σ (body σ f) m) (leave σ f) u ≠ none) : n = m := by
  by_contra hne
  rcases Nat.lt_or_ge n m with h | h
  · obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_lt h
    exact no_two_exit_depths σ f u hn (by rw [← hk]; exact hm)
  · obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_lt (Nat.lt_of_le_of_ne h fun hc => hne hc.symm)
    exact no_two_exit_depths σ f u hm (by rw [← hk]; exact hn)

/-- **The union over path lengths is single-valued.**

@cite abramsky-haghverdi-scott-2002
@cite haghverdi-scott-2010

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    {i j₁ j₂ : ℕ} {w₁ w₂ : Cell}                   [one start, two results, two weights]
    (h₁ : Traces σ f i j₁ w₁)                      [the trace relates one]
    (h₂ : Traces σ f i j₂ w₂)                      [and the other]
    (j₁, w₁) = (j₂, w₂)                            [so they are the same]

    So: the trace is a partial function, which is what makes it a route in its own right and
    what `theory/ROOTING.md` named as owed. Three facts close it, and each is a disjointness:
    the straight crossing and the way in cannot both be defined, the way out and one more trip
    cannot both be defined, and a partial map has one value where it has any. No weight is used,
    so this holds at every σ: the wiring alone carries the trace, and
    `weight_blocks_the_loop_iff_disc_nonneg` is the separate question of what the weight
    does. -/
theorem traces_functional (σ : ℝ) (f : Route) {i j₁ j₂ : ℕ} {w₁ w₂ : Cell}
    (h₁ : Traces σ f i j₁ w₁) (h₂ : Traces σ f i j₂ w₂) : (j₁, w₁) = (j₂, w₂) := by
  rcases h₁ with hd₁ | ⟨n, hn⟩
  · rcases h₂ with hd₂ | ⟨m, hm⟩
    · exact Option.some.inj (hd₁.symm.trans hd₂)
    · exfalso
      have hmn : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) m) (leave σ f)) i ≠ none := by rw [hm]; simp
      obtain ⟨p, hp, -⟩ := rcomp_ne_none hmn
      rcases direct_entry_disjoint σ f i with h | h
      · simp [h] at hd₁
      · simp [h] at hp
  · rcases h₂ with hd₂ | ⟨m, hm⟩
    · exfalso
      have hnn : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) n) (leave σ f)) i ≠ none := by rw [hn]; simp
      obtain ⟨p, hp, -⟩ := rcomp_ne_none hnn
      rcases direct_entry_disjoint σ f i with h | h
      · simp [h] at hd₂
      · simp [h] at hp
    · have hnn : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) n) (leave σ f)) i ≠ none := by rw [hn]; simp
      have hmn : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) m) (leave σ f)) i ≠ none := by rw [hm]; simp
      obtain ⟨p, hp, hne1⟩ := rcomp_ne_none hnn
      obtain ⟨p', hp', hne2⟩ := rcomp_ne_none hmn
      have hpp : p = p' := by rw [hp] at hp'; exact Option.some.inj hp'
      subst hpp
      have hnm : n = m := exit_depth_unique σ f p.1 hne1 hne2
      subst hnm
      exact Option.some.inj (hn.symm.trans hm)

-- ═══════════════════════════════════════════════════════════════════
-- READING THE PIECES AT A NAMED POSITION
-- ═══════════════════════════════════════════════════════════════════

/-- The identity route at a position.

    (i : ℕ)                            [a position]
    ℝ                                  [the real numbers: every number on the number line]
    rid i = some (i, one)              [it goes to itself with the unit weight] -/
theorem rid_apply (i : ℕ) : rid i = some (i, one) := rfl

/-- Reading the even half at an even position gives the first route.

    (a b : Route) (k : ℕ)                                        [two routes and a position]
    rtensor a b (Equiv.natSumNatEquivNat (Sum.inl k))            [the pair at an even position]
    = (a k).map fun q => (Equiv.natSumNatEquivNat (Sum.inl q.1), q.2)   [is the first route] -/
theorem rtensor_inl (a b : Route) (k : ℕ) :
    rtensor a b (Equiv.natSumNatEquivNat (Sum.inl k))
      = (a k).map fun q => (Equiv.natSumNatEquivNat (Sum.inl q.1), q.2) := by
  simp [rtensor]

/-- Reading the odd half at an odd position gives the second route.

    (a b : Route) (k : ℕ)                                        [two routes and a position]
    rtensor a b (Equiv.natSumNatEquivNat (Sum.inr k))            [the pair at an odd position]
    = (b k).map fun q => (Equiv.natSumNatEquivNat (Sum.inr q.1), q.2)   [is the second route] -/
theorem rtensor_inr (a b : Route) (k : ℕ) :
    rtensor a b (Equiv.natSumNatEquivNat (Sum.inr k))
      = (b k).map fun q => (Equiv.natSumNatEquivNat (Sum.inr q.1), q.2) := by
  simp [rtensor]

/-- The even reader at an even position.

    (k : ℕ)                                                    [a position]
    ℝ                                                          [the real numbers]
    outL (Equiv.natSumNatEquivNat (Sum.inl k)) = some (k, one) [it reads back] -/
theorem outL_inl (k : ℕ) : outL (Equiv.natSumNatEquivNat (Sum.inl k)) = some (k, one) := by
  simp [outL]

/-- The even reader at an odd position.

    (k : ℕ)                                              [a position]
    ℝ                                                    [the real numbers]
    outL (Equiv.natSumNatEquivNat (Sum.inr k)) = none     [it reads nothing] -/
theorem outL_inr (k : ℕ) : outL (Equiv.natSumNatEquivNat (Sum.inr k)) = none := by
  simp [outL]

/-- The odd reader at an odd position.

    (k : ℕ)                                                    [a position]
    ℝ                                                          [the real numbers]
    outR (Equiv.natSumNatEquivNat (Sum.inr k)) = some (k, one) [it reads back] -/
theorem outR_inr (k : ℕ) : outR (Equiv.natSumNatEquivNat (Sum.inr k)) = some (k, one) := by
  simp [outR]

/-- The odd reader at an even position.

    (k : ℕ)                                              [a position]
    ℝ                                                    [the real numbers]
    outR (Equiv.natSumNatEquivNat (Sum.inl k)) = none     [it reads nothing] -/
theorem outR_inl (k : ℕ) : outR (Equiv.natSumNatEquivNat (Sum.inl k)) = none := by
  simp [outR]

-- ═══════════════════════════════════════════════════════════════════
-- APPLYING ONE ROUTE TO ANOTHER
-- ═══════════════════════════════════════════════════════════════════

/-- Plug the argument into the applied route's loop.

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (a b : Route)                                  [the parameter and two routes]
    rcomp σ a (rtensor rid b)                              [run a, then leave the even track
    alone and send the odd track through b]                [which is the feedback wiring]
    : Route                                                [the route whose trace is a applied
    to b]                                                  [before the loop is closed]

    So: application in the Geometry of Interaction is composition with feedback, and this is the
    composition. `Traces` closes the loop. -/
def plug (σ : ℝ) (a b : Route) : Route := rcomp σ a (rtensor rid b)

/-- A route sitting on the even track alone crosses straight over, whatever it is fed.

    (σ : ℝ) (a b : Route) (i : ℕ)                  [the parameter, two routes, a position]
    direct σ (plug σ (rtensor a rnone) b) i = a i  [the straight crossing is a] -/
theorem emb_direct (σ : ℝ) (a b : Route) (i : ℕ) :
    direct σ (plug σ (rtensor a rnone) b) i = a i := by
  rw [direct_apply]
  simp only [plug, rcomp, rtensor_inl]
  cases ha : a i with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, rtensor_inl, rid_apply, outL_inl,
      one_mul, mul_one_right, Prod.mk.eta]

/-- Such a route never enters the loop.

    (σ : ℝ) (a b : Route) (i : ℕ)                  [the parameter, two routes, a position]
    entry σ (plug σ (rtensor a rnone) b) i = none  [nothing goes in] -/
theorem emb_entry (σ : ℝ) (a b : Route) (i : ℕ) :
    entry σ (plug σ (rtensor a rnone) b) i = none := by
  rw [entry_apply]
  simp only [plug, rcomp, rtensor_inl]
  cases ha : a i with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, rtensor_inl, rid_apply, outR_inl,
      Option.map_none]

/-- **The discarding equation.** A route on the even track alone ignores its argument.

@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ (plug σ (rtensor x rnone) y) i j w                  [applying it to y]
    ↔                                                            [exactly when]
    x i = some (j, w)                                            [gives x back]

    So: `K x y = x`, with application being composition with feedback. The second argument is
    discarded because no wire reaches it: `emb_entry` says the loop is never entered, so the
    trace is the straight crossing, and `emb_direct` says the straight crossing is `x`. -/
theorem emb_app (σ : ℝ) (x y : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ (rtensor x rnone) y) i j w ↔ x i = some (j, w) := by
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · rwa [emb_direct] at hd
    · exfalso
      rw [rcomp, emb_entry] at hn
      simp at hn
  · intro h
    exact Or.inl (by rw [emb_direct]; exact h)

-- ═══════════════════════════════════════════════════════════════════
-- THE IDENTITY COMBINATOR IS THE SWAP
-- ═══════════════════════════════════════════════════════════════════

/-- The route that exchanges the two tracks.

@cite joyal-street-verity-1996

    Route                              [a weighted wire map]

    So: the braiding. `feedback_yank` says tracing it gives the identity, and `iRoute_app` below
    is that statement with the argument carried through: applying the swap to a route returns
    the route. -/
def iRoute : Route := fun j =>
  Sum.elim (fun k => some (Equiv.natSumNatEquivNat (Sum.inr k), one))
    (fun k => some (Equiv.natSumNatEquivNat (Sum.inl k), one))
    (Equiv.natSumNatEquivNat.symm j)

/-- The swap sends an even position to the matching odd one.

    (k : ℕ)                                                            [a position]
    iRoute (Equiv.natSumNatEquivNat (Sum.inl k))                       [the swap there]
    = some (Equiv.natSumNatEquivNat (Sum.inr k), one)                  [crosses to the odd side] -/
theorem iRoute_inl (k : ℕ) :
    iRoute (Equiv.natSumNatEquivNat (Sum.inl k))
      = some (Equiv.natSumNatEquivNat (Sum.inr k), one) := by
  simp [iRoute]

/-- The swap sends an odd position to the matching even one.

    (k : ℕ)                                                            [a position]
    iRoute (Equiv.natSumNatEquivNat (Sum.inr k))                       [the swap there]
    = some (Equiv.natSumNatEquivNat (Sum.inl k), one)                  [crosses to the even side] -/
theorem iRoute_inr (k : ℕ) :
    iRoute (Equiv.natSumNatEquivNat (Sum.inr k))
      = some (Equiv.natSumNatEquivNat (Sum.inl k), one) := by
  simp [iRoute]

/-- The swap sends everything into the loop, so nothing crosses straight over.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    direct σ (plug σ iRoute x) i = none                [no straight crossing] -/
theorem i_direct (σ : ℝ) (x : Route) (i : ℕ) : direct σ (plug σ iRoute x) i = none := by
  rw [direct_apply]
  simp only [plug, rcomp, iRoute_inl, Option.bind_some, rtensor_inr]
  cases hx : x i with
  | none => simp
  | some p => simp only [Option.map_some, Option.bind_some, outL_inr, Option.map_none]

/-- Entering the loop reads the argument.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    entry σ (plug σ iRoute x) i = x i                  [the way in is the argument itself] -/
theorem i_entry (σ : ℝ) (x : Route) (i : ℕ) : entry σ (plug σ iRoute x) i = x i := by
  rw [entry_apply]
  simp only [plug, rcomp, iRoute_inl, Option.bind_some, rtensor_inr]
  cases hx : x i with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, outR_inr, one_mul, mul_one_right,
      Prod.mk.eta]

/-- The loop is never taken twice.

    (σ : ℝ) (x : Route) (u : ℕ)                        [the parameter, a route, a position]
    body σ (plug σ iRoute x) u = none                  [one trip and no more] -/
theorem i_body (σ : ℝ) (x : Route) (u : ℕ) : body σ (plug σ iRoute x) u = none := by
  rw [body_apply]
  simp only [plug, rcomp, iRoute_inr, Option.bind_some, rtensor_inl, rid_apply,
    Option.map_some, outR_inl, Option.map_none]

/-- Leaving the loop returns the position untouched.

    (σ : ℝ) (x : Route) (u : ℕ)                        [the parameter, a route, a position]
    leave σ (plug σ iRoute x) u = some (u, one)        [the way out changes nothing] -/
theorem i_leave (σ : ℝ) (x : Route) (u : ℕ) :
    leave σ (plug σ iRoute x) u = some (u, one) := by
  rw [leave_apply]
  simp only [plug, rcomp, iRoute_inr, Option.bind_some, rtensor_inl, rid_apply,
    Option.map_some, outL_inl, one_mul]

/-- **The identity equation.** Applying the swap to a route returns that route.

@cite joyal-street-verity-1996
@cite schonfinkel-1924

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ (plug σ iRoute x) i j w               [applying the swap to x]
    ↔                                              [exactly when]
    x i = some (j, w)                              [gives x back]

    So: `I x = x`, with application being composition with feedback. Everything enters the loop,
    goes round no times, and leaves, which is `feedback_yank` with an argument threaded through
    it. The weights collapse by the two unit laws, so this holds at every setting of the
    parameter. -/
theorem iRoute_app (σ : ℝ) (x : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ iRoute x) i j w ↔ x i = some (j, w) := by
  have hbody : body σ (plug σ iRoute x) = rnone := funext (i_body σ x)
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · rw [i_direct] at hd; exact absurd hd (by simp)
    · cases n with
      | zero =>
        rw [rpow, rcomp_rid_left] at hn
        rw [rcomp, i_entry] at hn
        cases hx : x i with
        | none => rw [hx] at hn; simp at hn
        | some p =>
          rw [hx] at hn
          simp only [Option.bind_some, i_leave, Option.map_some, mul_one_right,
            Prod.mk.eta] at hn
          exact hn
      | succ n =>
        rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
        simp [rnone] at hn
  · intro h
    refine Or.inr ⟨0, ?_⟩
    rw [rpow, rcomp_rid_left, rcomp, i_entry, h]
    simp only [Option.bind_some, i_leave, Option.map_some, mul_one_right]

-- ═══════════════════════════════════════════════════════════════════
-- THE DISCARDING COMBINATOR AS AN EXPLICIT ROUTING
-- ═══════════════════════════════════════════════════════════════════

/-- The route that keeps its first argument and wires nothing to its second.

@cite schonfinkel-1924
@cite abramsky-haghverdi-scott-2002

    Route                              [a weighted wire map]

    So: the even half of an external position goes into the loop, so it reaches the argument;
    what comes back out of the loop is sent to the even half of the result. The odd half of an
    external position goes nowhere, which is the discarding. -/
def kRoute : Route := fun j =>
  Sum.elim
    (fun a => Sum.elim (fun k => some (Equiv.natSumNatEquivNat (Sum.inr k), one))
      (fun _ => (none : Option (ℕ × Cell))) (Equiv.natSumNatEquivNat.symm a))
    (fun m => some (Equiv.natSumNatEquivNat
      (Sum.inl (Equiv.natSumNatEquivNat (Sum.inl m))), one))
    (Equiv.natSumNatEquivNat.symm j)

/-- The first argument's wire goes into the loop.

    (k : ℕ)                                                    [a position]
    kRoute at the even copy of the even copy of k              [the first argument's slot]
    = some (Equiv.natSumNatEquivNat (Sum.inr k), one)          [enters the loop] -/
theorem kRoute_inl_inl (k : ℕ) :
    kRoute (Equiv.natSumNatEquivNat (Sum.inl (Equiv.natSumNatEquivNat (Sum.inl k))))
      = some (Equiv.natSumNatEquivNat (Sum.inr k), one) := by
  simp [kRoute]

/-- The second argument's wire goes nowhere.

    (k : ℕ)                                                    [a position]
    kRoute at the even copy of the odd copy of k = none        [the second argument's slot is
    unwired]                                                   [which is the discarding] -/
theorem kRoute_inl_inr (k : ℕ) :
    kRoute (Equiv.natSumNatEquivNat (Sum.inl (Equiv.natSumNatEquivNat (Sum.inr k))))
      = none := by
  simp [kRoute]

/-- What comes back out of the loop is the result's first half.

    (m : ℕ)                                                    [a position]
    kRoute at the odd copy of m                    [coming back from the argument]
    = some ... one                                 [is the result, with the unit weight] -/
theorem kRoute_inr (m : ℕ) :
    kRoute (Equiv.natSumNatEquivNat (Sum.inr m))
      = some (Equiv.natSumNatEquivNat
        (Sum.inl (Equiv.natSumNatEquivNat (Sum.inl m))), one) := by
  simp [kRoute]

/-- Nothing crosses the discarding routing straight over.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    direct σ (plug σ kRoute x) i = none                [no straight crossing] -/
theorem k_direct (σ : ℝ) (x : Route) (i : ℕ) : direct σ (plug σ kRoute x) i = none := by
  rw [direct_apply]
  cases hs : Equiv.natSumNatEquivNat.symm i with
  | inl k =>
    have hi : i = Equiv.natSumNatEquivNat (Sum.inl k) := by
      rw [← hs, Equiv.apply_symm_apply]
    rw [hi]
    simp only [plug, rcomp, kRoute_inl_inl, Option.bind_some, rtensor_inr]
    cases hx : x k with
    | none => simp
    | some p => simp only [Option.map_some, Option.bind_some, outL_inr, Option.map_none]
  | inr k =>
    have hi : i = Equiv.natSumNatEquivNat (Sum.inr k) := by
      rw [← hs, Equiv.apply_symm_apply]
    rw [hi]
    simp only [plug, rcomp, kRoute_inl_inr, Option.bind_none, Option.map_none]

/-- The discarding routing never takes the loop twice.

    (σ : ℝ) (x : Route) (u : ℕ)                        [the parameter, a route, a position]
    body σ (plug σ kRoute x) u = none                  [one trip and no more] -/
theorem k_body (σ : ℝ) (x : Route) (u : ℕ) : body σ (plug σ kRoute x) u = none := by
  rw [body_apply]
  simp only [plug, rcomp, kRoute_inr, Option.bind_some, rtensor_inl, rid_apply,
    Option.map_some, outR_inl, Option.map_none]

/-- Leaving the loop lands in the first half of the result.

    (σ : ℝ) (x : Route) (u : ℕ)                        [the parameter, a route, a position]
    leave σ (plug σ kRoute x) u                        [the way out]
    = some (Equiv.natSumNatEquivNat (Sum.inl u), one)  [is the even copy of u] -/
theorem k_leave (σ : ℝ) (x : Route) (u : ℕ) :
    leave σ (plug σ kRoute x) u = some (Equiv.natSumNatEquivNat (Sum.inl u), one) := by
  rw [leave_apply]
  simp only [plug, rcomp, kRoute_inr, Option.bind_some, rtensor_inl, rid_apply,
    Option.map_some, outL_inl, one_mul]

/-- The first argument's slot enters the loop carrying the argument's own reading.

    (σ : ℝ) (x : Route) (k : ℕ)                    [the parameter, a route, a position]
    entry σ (plug σ kRoute x) at the even copy of k = x k    [reads x there] -/
theorem k_entry_inl (σ : ℝ) (x : Route) (k : ℕ) :
    entry σ (plug σ kRoute x) (Equiv.natSumNatEquivNat (Sum.inl k)) = x k := by
  rw [entry_apply]
  simp only [plug, rcomp, kRoute_inl_inl, Option.bind_some, rtensor_inr]
  cases hx : x k with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, outR_inr, one_mul, mul_one_right,
      Prod.mk.eta]

/-- The second argument's slot never enters the loop.

    (σ : ℝ) (x : Route) (k : ℕ)                    [the parameter, a route, a position]
    entry σ (plug σ kRoute x) at the odd copy of k = none     [nothing goes in] -/
theorem k_entry_inr (σ : ℝ) (x : Route) (k : ℕ) :
    entry σ (plug σ kRoute x) (Equiv.natSumNatEquivNat (Sum.inr k)) = none := by
  rw [entry_apply]
  simp only [plug, rcomp, kRoute_inl_inr, Option.bind_none, Option.map_none]

/-- **The discarding combinator, applied once, is the embedding.**

@cite schonfinkel-1924
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ (plug σ kRoute x) i j w               [applying the routing to x]
    ↔                                              [exactly when]
    rtensor x rnone i = some (j, w)                [gives x beside the empty route]

    So: with `emb_app`, this is `K x y = x`. The first application wires the argument onto the
    even track and leaves the odd track empty; the second application then finds no wire into
    its own argument and hands back what the first one stored. Both halves hold at every setting
    of the parameter, because the weights that arise are the argument's own and the unit. -/
theorem kRoute_app (σ : ℝ) (x : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ kRoute x) i j w ↔ rtensor x rnone i = some (j, w) := by
  have hbody : body σ (plug σ kRoute x) = rnone := funext (k_body σ x)
  cases hs : Equiv.natSumNatEquivNat.symm i with
  | inl k =>
    have hi : i = Equiv.natSumNatEquivNat (Sum.inl k) := by
      rw [← hs, Equiv.apply_symm_apply]
    subst hi
    rw [rtensor_inl]
    constructor
    · rintro (hd | ⟨n, hn⟩)
      · rw [k_direct] at hd; exact absurd hd (by simp)
      · cases n with
        | zero =>
          rw [rpow, rcomp_rid_left, rcomp, k_entry_inl] at hn
          cases hx : x k with
          | none => rw [hx] at hn; simp at hn
          | some p =>
            rw [hx] at hn
            simp only [Option.bind_some, k_leave, Option.map_some, mul_one_right] at hn
            simp only [Option.map_some]
            exact hn
        | succ n =>
          rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
          simp [rnone] at hn
    · intro h
      refine Or.inr ⟨0, ?_⟩
      rw [rpow, rcomp_rid_left, rcomp, k_entry_inl]
      cases hx : x k with
      | none => rw [hx] at h; simp at h
      | some p =>
        rw [hx] at h
        simp only [Option.map_some] at h
        simp only [Option.bind_some, k_leave, Option.map_some, mul_one_right]
        exact h
  | inr k =>
    have hi : i = Equiv.natSumNatEquivNat (Sum.inr k) := by
      rw [← hs, Equiv.apply_symm_apply]
    subst hi
    rw [rtensor_inr]
    simp only [rnone, Option.map_none]
    constructor
    · rintro (hd | ⟨n, hn⟩)
      · rw [k_direct] at hd; exact absurd hd (by simp)
      · rw [rcomp, k_entry_inr] at hn; simp at hn
    · intro h
      exact absurd h (by simp)

/-- **K x y = x**, with an explicit routing and application by feedback.

@cite schonfinkel-1924
@cite curry-1930
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    ∃ z : Route                                    [there is a route]
    ∀ i j w, Traces σ (plug σ kRoute x) i j w      [the routing applied to x]
    ↔ z i = some (j, w)                            [is exactly that route]
    ∀ i j w, Traces σ (plug σ z y) i j w           [and that applied to y]
    ↔ x i = some (j, w)                            [is exactly x again]
    ∧                                              [both together]

    So: the discarding combinator, built from the pairings alone, with application
    being composition with feedback. The intermediate route is `x` beside the empty route, and
    `y` is discarded because no wire reaches it. Neither half uses a property of the parameter:
    the weights that arise are the argument's own and the unit, so the equation holds on every
    face. What the parameter decides is elsewhere, in
    `weight_blocks_the_loop_iff_disc_nonneg`. -/
theorem kRoute_discards (σ : ℝ) (x y : Route) :
    ∃ z : Route,
      (∀ i j w, Traces σ (plug σ kRoute x) i j w ↔ z i = some (j, w)) ∧
      (∀ i j w, Traces σ (plug σ z y) i j w ↔ x i = some (j, w)) :=
  ⟨rtensor x rnone, kRoute_app σ x, emb_app σ x y⟩

end Combinator
end Apeiron

end
