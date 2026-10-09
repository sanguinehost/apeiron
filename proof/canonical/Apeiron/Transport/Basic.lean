import Apeiron.Transport.Ordered
import Apeiron.Trichotomy.NullBasis
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# Transport spread in exponential form

@name Transport Spread In Exponential Form
@grade proved

`Apeiron/Transport/Ordered.lean` carries the exact algebra: a field at rest is no more uneven
than its source range times its drag range, over any ordered field. This file is the real model
layer on top of it. The drag ratio `(1 - aLo) / (1 - aHi)` is written as an exponential of a span
in the reading `h`, and the exact bound becomes source range times that exponential.

`drag_ratio_of_exponential_response` is the identity that carries the reading: a retention that
falls off exponentially in the reading cancels in the ratio, leaving the span alone. The bound
`spread_le_source_times_exponential_range` takes the exponential form as a hypothesis, so the
model enters at one named place.

`exp_double_is_square` is `exp (2 * x) = (exp x) ^ 2` and nothing more. The predecessor stated it
with four binders named for a rate, a lapse, a relief and a source, and read it as "doubling the
relief squares the rainfall range a planet can carry". Its own header grades that reading a true
exponential identity that "establishes nothing about relief", with the planetary content carried
by the binder names. It is named here for what it proves. The theorem that carries the reading is
`drag_ratio_of_exponential_response`, where a span appears because a retention cancelled.

## What the model assumes

The exponential form of the drag is a hypothesis, not a conclusion. The source is exogenous, and
a source that grows with what a place already holds is outside every theorem here.
-/

noncomputable section

namespace Apeiron
namespace Spread

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- When what a place lets go falls off exponentially in a reading, only the span survives.

    (A k hLo hHi : ℝ) (hA : A < 1)                     [a retention, a rate, and two readings]
    ((1 - A) * Real.exp (-k * hLo)) / ((1 - A) * Real.exp (-k * hHi))
      = Real.exp (k * (hHi - hLo))                     [the drag ratio is the span alone]

    So: the retention cancels. The drag channel depends on the range of the reading and on
    nothing else, not on the level, not on the shape of the reading across places, and not on how
    the places connect. This is the result that carries the reading, and the identity below is
    not. -/
theorem drag_ratio_of_exponential_response (A k hLo hHi : ℝ) (hA : A < 1) :
    ((1 - A) * Real.exp (-k * hLo)) / ((1 - A) * Real.exp (-k * hHi))
      = Real.exp (k * (hHi - hLo)) := by
  have hne : (1 - A) ≠ 0 := by linarith
  rw [mul_div_mul_left _ _ hne, ← Real.exp_sub]
  congr 1
  ring

/-- Doubling the argument of an exponential squares it.

    (x : ℝ)                                            [any real number]
    Real.exp (2 * x) = (Real.exp x) ^ 2                [doubling squares]

    So: this is `exp (2 * x) = (exp x) ^ 2` and nothing more. The predecessor stated it with four
    binders named for a rate, a lapse, a relief and a source, and read it as "doubling the relief
    squares the rainfall range a planet can carry". Its own header grades that reading a true
    exponential identity that "establishes nothing about relief", with the planetary content
    carried by the binder names. It is named here for what it proves. The theorem that carries
    the reading is `drag_ratio_of_exponential_response`, where a span appears because a retention
    cancelled. -/
theorem exp_double_is_square (x : ℝ) : Real.exp (2 * x) = (Real.exp x) ^ 2 := by
  rw [two_mul, Real.exp_add, sq]

/-- The bound with the drag channel in exponential form.

    (T : Transport ℝ ι) (m : ι → ℝ) (h : T.IsRest m)     [a law and a field at rest under it]
    (lo hi : ℝ) (hlo : 0 < lo)                            [a floor above zero, and a ceiling]
    (hlos : ∀ j, lo ≤ T.source j)                         [every source is above the floor]
    (his : ∀ j, T.source j ≤ hi) (k hLo hHi : ℝ)          [and under it, a rate and two readings]
    (hdrag : (1 - T.aLo) / (1 - T.aHi) = Real.exp (k * (hHi - hLo)))   [the drag has that form]
    (i n : ι)                                             [two places]
    m i / m n ≤ (hi / lo) * Real.exp (k * (hHi - hLo))    [the range is the source times the span]

    So: the exponential form is a hypothesis here, not a conclusion.
    `drag_ratio_of_exponential_response` is what supplies it. -/
theorem spread_le_source_times_exponential_range (T : Transport ℝ ι) (m : ι → ℝ) (h : T.IsRest m)
    (lo hi : ℝ) (hlo : 0 < lo) (hlos : ∀ j, lo ≤ T.source j) (his : ∀ j, T.source j ≤ hi)
    (k hLo hHi : ℝ) (hdrag : (1 - T.aLo) / (1 - T.aHi) = Real.exp (k * (hHi - hLo)))
    (i n : ι) : m i / m n ≤ (hi / lo) * Real.exp (k * (hHi - hLo)) := by
  have hb := spread_ratio T m h lo hi hlo hlos his i n
  rwa [hdrag] at hb

/-- The canonical cell carrying the established lower and upper rest-state source bounds.

    (σ : ℝ)                                        [the trichotomy parameter]
    (T : Transport ℝ ι)                            [the finite real transport law]
    (lo hi : ℝ)                                    [the source floor and ceiling]
    Trichotomy.Cell                                [the two-channel interval cell]
    Trichotomy.ofHalves σ (hi / (1 - T.aHi)) (lo / (1 - T.aLo))
                                                   [the reconstructed source-bound cell]
    /                                              [division in the real field]
    -                                              [subtraction in the real field]

    So: this packages the existing scalar transport bounds and does not strengthen either one. -/
noncomputable def restEnvelope (σ : ℝ) (T : Transport ℝ ι) (lo hi : ℝ) : Trichotomy.Cell :=
  Trichotomy.ofHalves σ (hi / (1 - T.aHi)) (lo / (1 - T.aLo))

/-- A rest-state value lies in the canonical cell reconstructed from its source bounds.

    (σ : ℝ) (hσ : 0 < σ)                           [the essential split-face domain]
    (T : Transport ℝ ι) (m : ι → ℝ)                [a transport law and rest-state field]
    (h : T.IsRest m)                               [every place reproduces under the law]
    (lo hi : ℝ) (hlo : 0 < lo)                     [a positive source floor and ceiling]
    (hlos : ∀ j, lo ≤ T.source j)                  [every source is above the floor]
    (his : ∀ j, T.source j ≤ hi) (i : ι)           [every source is below the ceiling and a place]
    Trichotomy.InCell σ (m i) (restEnvelope σ T lo hi)
                                                   [the rest value lies in the envelope]
    Trichotomy.InCell                              [the number lies between the cell channels]
    restEnvelope                                   [the reconstructed transport-bound cell]
    ∀                                              [for every indexed place]
    ≤                                              [is no greater than]

    So: `rest_ge_min_source` and `rest_le_max_source` are the two channels of this one cell. -/
theorem inCell_restEnvelope (σ : ℝ) (hσ : 0 < σ) (T : Transport ℝ ι) (m : ι → ℝ) (h : T.IsRest m)
    (lo hi : ℝ) (hlo : 0 < lo) (hlos : ∀ j, lo ≤ T.source j) (his : ∀ j, T.source j ≤ hi)
    (i : ι) : Trichotomy.InCell σ (m i) (restEnvelope σ T lo hi) := by
  have hsrc : ∀ j, 0 ≤ T.source j := fun j => le_trans hlo.le (hlos j)
  unfold Trichotomy.InCell restEnvelope
  rw [Trichotomy.lower_ofHalves hσ, Trichotomy.upper_ofHalves hσ]
  exact ⟨rest_ge_min_source T m h hsrc lo hlos i, rest_le_max_source T m h hsrc hi his i⟩

/-- Meeting a transport envelope with another cell preserves exactly the other cell's constraint.

    (σ : ℝ) (hσ : 0 < σ)                           [the essential split-face domain]
    (T : Transport ℝ ι) (m : ι → ℝ)                [a transport law and rest-state field]
    (h : T.IsRest m)                               [every place reproduces under the law]
    (lo hi : ℝ) (hlo : 0 < lo)                     [a positive source floor and ceiling]
    (hlos : ∀ j, lo ≤ T.source j)                  [every source is above the floor]
    (his : ∀ j, T.source j ≤ hi) (i : ι)           [every source is below the ceiling and a place]
    (z : Trichotomy.Cell)                          [another canonical constraint cell]
    Trichotomy.InCell σ (m i) (Trichotomy.meet σ (restEnvelope σ T lo hi) z)
      ↔ Trichotomy.InCell σ (m i) z                [meeting keeps exactly the other constraint]
    Trichotomy.meet                                [channelwise interval intersection]
    ↔                                              [exactly when, true in both directions]

    So: meeting composes constraints; it does not improve the transport estimate. -/
theorem inCell_meet_restEnvelope_iff (σ : ℝ) (hσ : 0 < σ) (T : Transport ℝ ι) (m : ι → ℝ)
    (h : T.IsRest m) (lo hi : ℝ) (hlo : 0 < lo) (hlos : ∀ j, lo ≤ T.source j)
    (his : ∀ j, T.source j ≤ hi) (i : ι) (z : Trichotomy.Cell) :
    Trichotomy.InCell σ (m i) (Trichotomy.meet σ (restEnvelope σ T lo hi) z) ↔
      Trichotomy.InCell σ (m i) z := by
  rw [Trichotomy.inCell_meet_iff hσ]
  constructor
  · exact And.right
  · intro hz
    exact ⟨inCell_restEnvelope σ hσ T m h lo hi hlo hlos his i, hz⟩

/-- A finite nonempty transport law, its rest state, and its envelope inhabit the joint domain.

    ∃ T : Transport ℝ (Fin 1)                      [there is a one-place real transport law]
    ∃ m : Fin 1 → ℝ                                [there is a real field on that finite place]
    T.IsRest m                                     [the field is at rest]
    ∧ 0 < (1 : ℝ)                                 [the envelope parameter is positive]
    ∧ (∀ j, 1 ≤ T.source j)                       [unit is a source floor]
    ∧ (∀ j, T.source j ≤ 1)                       [unit is a source ceiling]
    ∧ Trichotomy.InCell 1 (m 0) (restEnvelope 1 T 1 1)
                                                   [the rest value lies in its envelope]
    ∃                                              [there exists]
    ∧                                              [all witness facts hold]
    ℝ                                              [the real numbers]
    Fin                                            [a finite index type]
    ∀                                              [for every indexed place]
    ≤                                              [is no greater than]
    Trichotomy.InCell                              [the number lies between the cell channels]

    So: the finite witness inhabits the transport, domain, source-bound, and envelope facts.
    They hold together. -/
theorem restEnvelope_joint_witness :
    ∃ T : Transport ℝ (Fin 1), ∃ m : Fin 1 → ℝ, T.IsRest m ∧ 0 < (1 : ℝ) ∧
      (∀ j, 1 ≤ T.source j) ∧ (∀ j, T.source j ≤ 1) ∧
      Trichotomy.InCell 1 (m 0) (restEnvelope 1 T 1 1) := by
  let T : Transport ℝ (Fin 1) :=
    { a := fun _ => 0
      aLo := 0
      aHi := 0
      w := 0
      pred := id
      cross := id
      source := fun _ => 1
      aLo_nonneg := by norm_num
      a_ge := by intro i; norm_num
      a_le := by intro i; norm_num
      aHi_lt_one := by norm_num
      w_nonneg := by norm_num
      w_le_one := by norm_num }
  have hrest : T.IsRest (fun _ => (1 : ℝ)) := by
    intro i
    simp [T]
  have hfloor : ∀ j, (1 : ℝ) ≤ T.source j := by
    intro j
    simp [T]
  have hceiling : ∀ j, T.source j ≤ (1 : ℝ) := by
    intro j
    simp [T]
  refine ⟨T, fun _ => 1, hrest, by norm_num, hfloor, hceiling, ?_⟩
  exact inCell_restEnvelope 1 (by norm_num) T (fun _ => 1) hrest 1 1 (by norm_num)
    hfloor hceiling 0

/-- A rest-state range is bounded by the exponential source range of a canonical cell meet.

    (σ : ℝ) (hσ : 0 < σ)                           [the essential split-face domain]
    (T : Transport ℝ ι) (m : ι → ℝ)                [a transport law and rest-state field]
    (hrest : T.IsRest m)                            [every place reproduces under the law]
    (z w : Trichotomy.Cell)                         [two source constraint cells]
    (hlo : 0 < Trichotomy.lower σ (Trichotomy.meet σ z w))
                                                   [the joined source floor is positive]
    (hsrc : ∀ j, Trichotomy.InCell σ (T.source j) z ∧ Trichotomy.InCell σ (T.source j) w)
                                                   [every source satisfies both cells]
    (k hLo hHi : ℝ)                                 [the exponential rate and reading bounds]
    (hdrag : (1 - T.aLo) / (1 - T.aHi) = Real.exp (k * (hHi - hLo)))
                                                   [the specified drag response]
    (i n : ι)                                       [two places]
    m i / m n ≤ (Trichotomy.upper σ (Trichotomy.meet σ z w) /
      Trichotomy.lower σ (Trichotomy.meet σ z w)) * Real.exp (k * (hHi - hLo))
                                                   [the source-meet exponential range bound]
    Trichotomy.InCell                               [the number lies between the cell channels]
    Trichotomy.meet                                 [channelwise interval intersection]
    Real.exp                                        [the real exponential]
    ∀                                              [for every indexed place]
    j                                              [an indexed place]
    ∧                                              [both constraints hold]
    /                                              [division in the real field]
    ≤                                              [is no greater than]

    So: the meet supplies the scalar source floor and ceiling consumed by the established bound. -/
theorem spread_le_source_meet_exponential_range {σ : ℝ} (hσ : 0 < σ) (T : Transport ℝ ι)
    (m : ι → ℝ) (hrest : T.IsRest m) (z w : Trichotomy.Cell)
    (hlo : 0 < Trichotomy.lower σ (Trichotomy.meet σ z w))
    (hsrc : ∀ j, Trichotomy.InCell σ (T.source j) z ∧ Trichotomy.InCell σ (T.source j) w)
    (k hLo hHi : ℝ) (hdrag : (1 - T.aLo) / (1 - T.aHi) = Real.exp (k * (hHi - hLo)))
    (i n : ι) :
    m i / m n ≤ (Trichotomy.upper σ (Trichotomy.meet σ z w) /
      Trichotomy.lower σ (Trichotomy.meet σ z w)) * Real.exp (k * (hHi - hLo)) := by
  have hsource : ∀ j, Trichotomy.InCell σ (T.source j) (Trichotomy.meet σ z w) := fun j =>
    (Trichotomy.inCell_meet_iff hσ (T.source j) z w).mpr (hsrc j)
  exact spread_le_source_times_exponential_range T m hrest
    (Trichotomy.lower σ (Trichotomy.meet σ z w)) (Trichotomy.upper σ (Trichotomy.meet σ z w)) hlo
    (fun j => (hsource j).1) (fun j => (hsource j).2) k hLo hHi hdrag i n

end Spread
end Apeiron

end
