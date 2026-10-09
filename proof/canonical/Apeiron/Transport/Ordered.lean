import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Mathlib.Tactic

/-!
# Transport spread over an ordered ring and field

@name Transport Spread
@grade proved

Every place keeps some fraction of what arrived from its neighbours and adds its own source. A
field at rest under that law cannot be more uneven than what feeds it. `spread_ratio` is the
statement: the ratio between the largest and smallest values is at most the **source** ratio
times the **drag** ratio, and there is no third factor.

This file is the exact algebra, over a general scalar. The convex mixing needs only an ordered
commutative ring (`mix_le_max`, `min_le_mix`, `rest_nonneg`). The two bounds that divide and the
ratio they give need an ordered field (`rest_le_max_source`, `rest_ge_min_source`,
`spread_ratio`). `ℝ` is the instance the exponential model in `Apeiron/Transport/Basic.lean`
uses. No completeness and no exponential is read here.

The mechanism is one line. Every transport term is a convex mixture, a mixture is an average,
and an average lies inside what it averages. Reading the law at the largest and smallest places
bounds each one against the sources, with no series summed.

The mixing weight `w` occurs in no bound. `mixing_cannot_widen` says so by instantiating the
headline at a changed weight, which is why it holds: a quantity absent from a bound cannot move
it. A second transport face, a longer fetch, more sweeps, none of them enter.

## What the bound assumes

The source is exogenous: `source : ι → K` does not read the field. A source that grows with what
a place already holds is outside every theorem here, and that is the whole of the remaining
freedom. Nothing below quantifies over such a source or bounds one.

## Root

This file instantiates nothing from `Apeiron/Trichotomy`. It is convexity over an ordered
commutative ring and an ordered field, and no declaration mentions `Cell` or `RingCell`. By
`theory/NOVELTY.md`'s A1 spectator test `spread_ratio` and its companions are facts about those
structures rather than about this algebra, which is why `Transport/Basic.lean` reads them as its
exact algebra at `K = ℝ`.

The primitive that would supply an edge is the order the algebra carries on a null channel,
`Trichotomy.le_iff_exists_sq` and `inCell_meet_iff`, which fix that order by the channel's own
ring. A transport law whose scalar is a channel rather than a bare ordered field would then be a
statement about the algebra. Not built.

## A correction carried forward

The predecessor's `variety_is_exponential_in_relief` states
`src * exp (k * (g * (2 * dH))) = src * (exp (k * (g * dH))) ^ 2`, and its own header grades it a
true exponential identity that "establishes nothing about relief", with the planetary content
carried by the binder names `k g dH src`. That is right, and
`../whispering-tides/WORKLOG.md` cites it as the payoff anyway. It appears here as
`exp_double_is_square`, named for what it proves, in `Apeiron/Transport/Basic.lean`. The result
that carries the reading is `drag_ratio_of_exponential_response`, where the retention cancels
and the span is what is left.
-/

noncomputable section

namespace Apeiron
namespace Spread

/-- A transport law: what each place keeps, how it mixes its two neighbours, and what it adds.

    Transport K ι          [a transport law over a scalar K and places indexed by ι]
    a : ι → K              [what each place keeps]
    aLo aHi : K            [the least and greatest retention]
    w : K                  [how the two neighbours are mixed]
    pred cross : ι → ι     [the two neighbours]
    source : ι → K         [what each place adds]

    The scalars are an ordered commutative ring, so the mixing is an average and the order
    bounds it. Division is not asked for here; the bounds that divide live below. -/
structure Transport (K : Type*) [CommRing K] [LinearOrder K] [IsStrictOrderedRing K]
    (ι : Type*) where
  /-- What each place keeps of what arrived. -/
  a : ι → K
  /-- The least retention. -/
  aLo : K
  /-- The greatest retention. -/
  aHi : K
  /-- How the two neighbours are mixed. -/
  w : K
  /-- One neighbour. -/
  pred : ι → ι
  /-- The other. -/
  cross : ι → ι
  /-- What each place adds of its own. -/
  source : ι → K
  /-- Retention is not negative. -/
  aLo_nonneg : 0 ≤ aLo
  /-- Every place keeps at least the least. -/
  a_ge : ∀ i, aLo ≤ a i
  /-- And at most the greatest. -/
  a_le : ∀ i, a i ≤ aHi
  /-- No place keeps everything, or nothing reaches rest. -/
  aHi_lt_one : aHi < 1
  /-- The mixing weight is a weight. -/
  w_nonneg : 0 ≤ w
  /-- And no more than one. -/
  w_le_one : w ≤ 1

section Ring

variable {K : Type*} [CommRing K] [LinearOrder K] [IsStrictOrderedRing K]
variable {ι : Type*} [Fintype ι] [Nonempty ι]

namespace Transport

omit [Fintype ι] [Nonempty ι] in
/-- Every place keeps a non-negative fraction.

    (T : Transport K ι) (i : ι)     [a transport law and a place]
    0 ≤ T.a i                        [what it keeps is not negative] -/
theorem a_nonneg (T : Transport K ι) (i : ι) : 0 ≤ T.a i :=
  le_trans T.aLo_nonneg (T.a_ge i)

omit [Fintype ι] [Nonempty ι] in
/-- No place keeps everything.

    (T : Transport K ι) (i : ι)     [a transport law and a place]
    T.a i < 1                        [what it keeps is less than all of it] -/
theorem a_lt_one (T : Transport K ι) (i : ι) : T.a i < 1 :=
  lt_of_le_of_lt (T.a_le i) T.aHi_lt_one

/-- The field is at rest when every place reproduces itself under the law.

    (T : Transport K ι) (m : ι → K)     [a transport law and a field]
    ∀ i, m i = ...                       [for every single place, it reproduces itself] -/
def IsRest (T : Transport K ι) (m : ι → K) : Prop :=
  ∀ i, m i = T.a i * ((1 - T.w) * m (T.pred i) + T.w * m (T.cross i)) + T.source i

end Transport

omit [Fintype ι] [Nonempty ι] in
/-- A mixture of two values, each below the largest, is below the largest.

    (T : Transport K ι) (m : ι → K)               [a transport law and a field]
    (i₀ : ι) (hi₀ : ∀ x, m x ≤ m i₀)              [a place no other exceeds]
    (1 - T.w) * m (T.pred i₀) + T.w * m (T.cross i₀) ≤ m i₀   [what arrives there is no larger]

    So: this is the whole mechanism, and everything below is bookkeeping around it. -/
theorem mix_le_max (T : Transport K ι) (m : ι → K) (i₀ : ι) (hi₀ : ∀ x, m x ≤ m i₀) :
    (1 - T.w) * m (T.pred i₀) + T.w * m (T.cross i₀) ≤ m i₀ := by
  have hw1 : (0 : K) ≤ 1 - T.w := by linarith [T.w_le_one]
  nlinarith [hi₀ (T.pred i₀), hi₀ (T.cross i₀), T.w_nonneg]

omit [Fintype ι] [Nonempty ι] in
/-- A mixture of two values, each above the smallest, is above the smallest.

    (T : Transport K ι) (m : ι → K)               [a transport law and a field]
    (j₀ : ι) (hj₀ : ∀ x, m j₀ ≤ m x)              [a place no other falls below]
    m j₀ ≤ (1 - T.w) * m (T.pred j₀) + T.w * m (T.cross j₀)   [what arrives there is no smaller] -/
theorem min_le_mix (T : Transport K ι) (m : ι → K) (j₀ : ι) (hj₀ : ∀ x, m j₀ ≤ m x) :
    m j₀ ≤ (1 - T.w) * m (T.pred j₀) + T.w * m (T.cross j₀) := by
  have hw1 : (0 : K) ≤ 1 - T.w := by linarith [T.w_le_one]
  nlinarith [hj₀ (T.pred j₀), hj₀ (T.cross j₀), T.w_nonneg]

/-- A field at rest under non-negative sources is nowhere negative.

    (T : Transport K ι) (m : ι → K) (h : T.IsRest m)     [a law and a field at rest under it]
    (hsrc : ∀ j, 0 ≤ T.source j) (i : ι)                  [sources are not negative, and a place]
    0 ≤ m i                                               [the field there is not negative]

    So: read the law at the smallest place. What arrives there is no smaller than it
    (`min_le_mix`), so the source alone already puts it above zero. -/
theorem rest_nonneg (T : Transport K ι) (m : ι → K) (h : T.IsRest m)
    (hsrc : ∀ j, 0 ≤ T.source j) (i : ι) : 0 ≤ m i := by
  obtain ⟨j₀, hj₀⟩ := Finite.exists_min m
  have hkey : T.a j₀ * m j₀ + T.source j₀ ≤ m j₀ := by
    have hmul := mul_le_mul_of_nonneg_left (min_le_mix T m j₀ hj₀) (T.a_nonneg j₀)
    have hr := h j₀
    linarith
  have h1 : 0 < 1 - T.a j₀ := by linarith [T.a_lt_one j₀]
  have hj : 0 ≤ m j₀ := by nlinarith [hsrc j₀]
  exact le_trans hj (hj₀ i)

end Ring

section Field

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- No place holds more than the largest source over the least drag.

    (T : Transport K ι) (m : ι → K) (h : T.IsRest m)     [a law and a field at rest under it]
    (hsrc : ∀ j, 0 ≤ T.source j) (hi : K)                 [sources are not negative, and a ceiling]
    (his : ∀ j, T.source j ≤ hi) (i : ι)                  [which they respect, and a place]
    m i ≤ hi / (1 - T.aHi)                                [the field there is under that] -/
theorem rest_le_max_source (T : Transport K ι) (m : ι → K) (h : T.IsRest m)
    (hsrc : ∀ j, 0 ≤ T.source j) (hi : K) (his : ∀ j, T.source j ≤ hi) (i : ι) :
    m i ≤ hi / (1 - T.aHi) := by
  obtain ⟨i₀, hi₀⟩ := Finite.exists_max m
  have hposHi : 0 < 1 - T.aHi := by linarith [T.aHi_lt_one]
  have hnn : 0 ≤ m i₀ := rest_nonneg T m h hsrc i₀
  have hkey : m i₀ ≤ T.a i₀ * m i₀ + T.source i₀ := by
    have hmul := mul_le_mul_of_nonneg_left (mix_le_max T m i₀ hi₀) (T.a_nonneg i₀)
    have hr := h i₀
    linarith
  have hb : m i₀ * (1 - T.aHi) ≤ hi := by nlinarith [T.a_le i₀, his i₀]
  calc m i ≤ m i₀ := hi₀ i
    _ ≤ hi / (1 - T.aHi) := by rw [le_div_iff₀ hposHi]; linarith

/-- No place holds less than the smallest source over the greatest drag.

    (T : Transport K ι) (m : ι → K) (h : T.IsRest m)     [a law and a field at rest under it]
    (hsrc : ∀ j, 0 ≤ T.source j) (lo : K)                 [sources are not negative, and a floor]
    (hlos : ∀ j, lo ≤ T.source j) (i : ι)                 [which they respect, and a place]
    lo / (1 - T.aLo) ≤ m i                                [the field there is above that] -/
theorem rest_ge_min_source (T : Transport K ι) (m : ι → K) (h : T.IsRest m)
    (hsrc : ∀ j, 0 ≤ T.source j) (lo : K) (hlos : ∀ j, lo ≤ T.source j) (i : ι) :
    lo / (1 - T.aLo) ≤ m i := by
  obtain ⟨j₀, hj₀⟩ := Finite.exists_min m
  have hposLo : 0 < 1 - T.aLo := by
    have h2 := T.a_lt_one (Classical.arbitrary ι)
    have h3 := T.a_ge (Classical.arbitrary ι)
    linarith
  have hkey : T.a j₀ * m j₀ + T.source j₀ ≤ m j₀ := by
    have hmul := mul_le_mul_of_nonneg_left (min_le_mix T m j₀ hj₀) (T.a_nonneg j₀)
    have hr := h j₀
    linarith
  have hnn : 0 ≤ m j₀ := rest_nonneg T m h hsrc j₀
  have hb : lo ≤ m j₀ * (1 - T.aLo) := by nlinarith [T.a_ge j₀, hlos j₀]
  calc lo / (1 - T.aLo) ≤ m j₀ := by rw [div_le_iff₀ hposLo]; linarith
    _ ≤ m i := hj₀ i

/-- The range of a field at rest is at most its source range times its drag range.

    (T : Transport K ι) (m : ι → K) (h : T.IsRest m)     [a law and a field at rest under it]
    (lo hi : K) (hlo : 0 < lo)                            [a floor above zero, and a ceiling]
    (hlos : ∀ j, lo ≤ T.source j)                         [every source is above the floor]
    (his : ∀ j, T.source j ≤ hi) (i k : ι)                [and under the ceiling, and two places]
    m i / m k ≤ (hi / lo) * ((1 - T.aLo) / (1 - T.aHi))   [the range is at most that product]

    So: how uneven what is put in is, times how unevenly places hold on to it. Those are the two
    channels and there is no third. The mixing weight does not appear, so no transport face, no
    fetch and no number of sweeps enters. -/
theorem spread_ratio (T : Transport K ι) (m : ι → K) (h : T.IsRest m)
    (lo hi : K) (hlo : 0 < lo) (hlos : ∀ j, lo ≤ T.source j) (his : ∀ j, T.source j ≤ hi)
    (i k : ι) :
    m i / m k ≤ (hi / lo) * ((1 - T.aLo) / (1 - T.aHi)) := by
  have hsrc : ∀ j, 0 ≤ T.source j := fun j => le_trans (le_of_lt hlo) (hlos j)
  have hposHi : 0 < 1 - T.aHi := by linarith [T.aHi_lt_one]
  have hposLo : 0 < 1 - T.aLo := by
    have h2 := T.a_lt_one (Classical.arbitrary ι)
    have h3 := T.a_ge (Classical.arbitrary ι)
    linarith
  have hik : m i ≤ hi / (1 - T.aHi) := rest_le_max_source T m h hsrc hi his i
  have hkl : lo / (1 - T.aLo) ≤ m k := rest_ge_min_source T m h hsrc lo hlos k
  have hlodiv : 0 < lo / (1 - T.aLo) := div_pos hlo hposLo
  have hkpos : 0 < m k := lt_of_lt_of_le hlodiv hkl
  rw [div_le_iff₀ hkpos]
  have hhi : 0 < hi :=
    lt_of_lt_of_le hlo
      (le_trans (hlos (Classical.arbitrary ι)) (his (Classical.arbitrary ι)))
  have hcoef : 0 ≤ (hi / lo) * ((1 - T.aLo) / (1 - T.aHi)) :=
    mul_nonneg (div_nonneg hhi.le hlo.le) (div_nonneg hposLo.le hposHi.le)
  calc m i ≤ hi / (1 - T.aHi) := hik
    _ = (hi / lo) * ((1 - T.aLo) / (1 - T.aHi)) * (lo / (1 - T.aLo)) := by
        field_simp
    _ ≤ (hi / lo) * ((1 - T.aLo) / (1 - T.aHi)) * m k :=
        mul_le_mul_of_nonneg_left hkl hcoef

/-- A constant retention leaves only the source channel.

    (T : Transport K ι) (m : ι → K) (h : T.IsRest m)     [a law and a field at rest under it]
    (lo hi : K) (hlo : 0 < lo)                            [a floor above zero, and a ceiling]
    (hlos : ∀ j, lo ≤ T.source j)                         [every source is above the floor]
    (his : ∀ j, T.source j ≤ hi)                          [and under the ceiling]
    (huni : T.aLo = T.aHi) (i k : ι)                      [retention is the same everywhere]
    m i / m k ≤ hi / lo                                   [only the source range is left] -/
theorem uniform_retention_ratio (T : Transport K ι) (m : ι → K) (h : T.IsRest m)
    (lo hi : K) (hlo : 0 < lo) (hlos : ∀ j, lo ≤ T.source j) (his : ∀ j, T.source j ≤ hi)
    (huni : T.aLo = T.aHi) (i k : ι) :
    m i / m k ≤ hi / lo := by
  have hposHi : 0 < 1 - T.aHi := by linarith [T.aHi_lt_one]
  have := spread_ratio T m h lo hi hlo hlos his i k
  rw [huni, div_self (ne_of_gt hposHi), mul_one] at this
  exact this

/-- Changing how the two neighbours are mixed cannot widen the range.

    (T : Transport K ι) (w' : K)                         [a law, and another mixing weight]
    (hw0 : 0 ≤ w') (hw1 : w' ≤ 1)                        [which is a weight]
    (m : ι → K) (h : ({T with w := w', w_nonneg := hw0, w_le_one := hw1} : Transport K ι).IsRest m)
                                                                    [a field at rest under that]
    →                                                     [takes the left and gives the right]
    (lo hi : K) (hlo : 0 < lo)                           [a floor above zero, and a ceiling]
    (hlos : ∀ j, lo ≤ T.source j)                        [every source is above the floor]
    (his : ∀ j, T.source j ≤ hi) (i k : ι)               [and under the ceiling, and two places]
    m i / m k ≤ (hi / lo) * ((1 - T.aLo) / (1 - T.aHi))  [the same bound, unchanged]

    So: the weight occurs in no bound, so no weight can move one. A second transport face is not
    an exception to the headline, it is an instance of it. -/
theorem mixing_cannot_widen (T : Transport K ι) (w' : K) (hw0 : 0 ≤ w') (hw1 : w' ≤ 1)
    (m : ι → K) (h : ({T with w := w', w_nonneg := hw0, w_le_one := hw1} : Transport K ι).IsRest m)
    (lo hi : K) (hlo : 0 < lo) (hlos : ∀ j, lo ≤ T.source j) (his : ∀ j, T.source j ≤ hi)
    (i k : ι) :
    m i / m k ≤ (hi / lo) * ((1 - T.aLo) / (1 - T.aHi)) :=
  spread_ratio { T with w := w', w_nonneg := hw0, w_le_one := hw1 } m h lo hi hlo hlos his i k

end Field

end Spread
end Apeiron

end
