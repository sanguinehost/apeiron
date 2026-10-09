import Apeiron.Trichotomy.NullBasis
import Apeiron.Trichotomy.IdempotentsAnyRing

/-!
# Projection, and what it costs to discard

@name Projection
@cite peirce-1870
@cite boole-1847
@cite schonfinkel-1924
@cite palmieri-2026
@cite miquel-2018
@grade proved

Multiplying by a point that is unchanged by squaring is a projection: do it twice and nothing
further happens (`proj_proj`). Above zero the two extra idempotents project onto one channel
each, keeping everything in that channel and sending the other channel to zero
(`proj_idemPlus_halves`).

Keeping one thing and throwing the other away is what a projection does, and it is not free.
To throw something away the algebra must contain a non-zero point that multiplies to zero,
because that is what being thrown away means. Below zero no such point exists
(`noZeroDivisors_iff_disc_neg`), so every projection available is either the constant zero
map or the identity, and neither discards anything.

`discards_iff_disc_pos` is that statement: a projection that kills some non-zero point while
preserving some other non-zero point exists exactly when sigma is positive.

## What this supplies to a combinator layer, and what it does not

Discarding an argument is one feature of `K`, which takes two things and returns the first.
Above zero the algebra discards: pack two numbers into the two channels of a point, project,
and read a channel (`proj_keeps_first_channel`). The unit leaves a point alone (`proj_one`).

**It supplies no combinator.** `K` is defined by its behaviour under application, for arguments
of every kind, and no operation of this algebra is application: `mul σ` is commutative and
associative, and no commutative associative operation admits a `K` on a carrier with more than
one element. That obstruction, its published form in Palmieri arXiv:2603.27007, and the
order-based route that does carry application are in `theory/DEAD_ENDS.md` and
`theory/LITERATURE.md`. `proj_keeps_first_channel` and `proj_keeps_second_channel` are named for
what they prove, keeping a channel under a fixed linear projection on reals, since the earlier
names `proj_realises_K` and `proj_realises_K_swapped` claimed the combinator this file does not
supply.

`S` is further still. `S f g x` applies two functions to one duplicated argument, so it uses
an argument twice, and `mul σ z e` is bilinear and uses `z` once.

What this file proves is a fact about projections: which ones exist, and at which sigma.
That fact is worth having on its own.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

/-- Multiply a point by a fixed point, as an operation on points.

    (σ : ℝ)                   [the parameter]
    (e z : Cell)              [the fixed point, and the point being projected]
    mul σ z e                 [multiply them]
    : Cell                    [a point of the algebra]

    So: this is `AnyRing.ringProj` at `R = ℝ`, so the operation is the generic one rather than a
    second definition. -/
def proj (σ : ℝ) (e z : Cell) : Cell := AnyRing.ringProj σ e z

/-- Projecting twice is the same as projecting once.

    {σ : ℝ} {e : Cell}                     [the parameter and the fixed point]
    (he : Idempotent σ e)                  [the fixed point is unchanged by squaring]
    (z : Cell)                             [any point]
    proj σ e (proj σ e z) = proj σ e z     [the second projection changes nothing]

    So: the two multiplications by `e` collapse into one, because `e` squared is `e`. This is
    `AnyRing.ringProj_proj`, so it holds in every commutative ring. -/
theorem proj_proj {σ : ℝ} {e : Cell} (he : Idempotent σ e) (z : Cell) :
    proj σ e (proj σ e z) = proj σ e z :=
  AnyRing.ringProj_proj he z

/-- Projecting by the unit changes nothing.

    (σ : ℝ) (z : Cell)               [the parameter and any point]
    proj σ one z = z                 [multiplying by the unit returns the point]

    So: `AnyRing.ringProj_one`. -/
theorem proj_one (σ : ℝ) (z : Cell) : proj σ one z = z :=
  AnyRing.ringProj_one σ z

/-- Projecting by the origin sends everything to the origin.

    (σ : ℝ) (z : Cell)               [the parameter and any point]
    proj σ 0 z = 0                   [multiplying by the origin gives the origin]

    So: `AnyRing.ringProj_zero`. -/
theorem proj_zero (σ : ℝ) (z : Cell) : proj σ 0 z = 0 :=
  AnyRing.ringProj_zero σ z

/-- Above zero, projecting by `idemPlus` keeps the first channel and empties the second.

    {σ : ℝ} (hσ : 0 < σ)                                 [the parameter, above zero]
    (z : Cell)                                           [any point]
    upper σ (proj σ (idemPlus σ) z) = upper σ z          [the first channel survives intact]
    ∧                                                    [and]
    lower σ (proj σ (idemPlus σ) z) = 0                  [the second channel is gone]

    So: the coordinates multiply, and `idemPlus` has coordinates `1` and `0`. Multiplying by
    one leaves a channel alone; multiplying by zero empties it. -/
theorem proj_idemPlus_halves {σ : ℝ} (hσ : 0 < σ) (z : Cell) :
    upper σ (proj σ (idemPlus σ) z) = upper σ z ∧
      lower σ (proj σ (idemPlus σ) z) = 0 := by
  obtain ⟨hp1, hp0⟩ := halves_idemPlus hσ
  constructor
  · show upper σ (mul σ z (idemPlus σ)) = upper σ z
    rw [upper_mul hσ.le, hp1, mul_one]
  · show lower σ (mul σ z (idemPlus σ)) = 0
    rw [lower_mul hσ.le, hp0, mul_zero]

/-- A projection that throws information away: it kills something that was not already
nothing, and it keeps something that is not nothing either.

    (σ : ℝ)                                    [the parameter]
    ∃ e : Cell                                 [there is a point e to project by]
    Idempotent σ e                             [e is unchanged by squaring]
    (∃ z, z ≠ 0 ∧ proj σ e z = 0)              [some non-zero point is sent to the origin]
    (∃ w, w ≠ 0 ∧ proj σ e w = w)              [some non-zero point survives untouched]

    So: the first condition rules out the identity, which discards nothing. The second rules
    out the constant-zero map, which keeps nothing. Together they say the projection is a
    genuine choice of what to keep. -/
def Discards (σ : ℝ) : Prop :=
  ∃ e : Cell, Idempotent σ e ∧
    (∃ z : Cell, z ≠ 0 ∧ proj σ e z = 0) ∧ (∃ w : Cell, w ≠ 0 ∧ proj σ e w = w)

/-- A projection that throws information away exists exactly when sigma is positive.

    (σ : ℝ)                     [the parameter]
    Discards σ                  [some projection both keeps and discards]
    ↔                           [exactly when]
    0 < σ                       [sigma is positive]

    So: above zero, projecting by `idemPlus` kills `idemMinus` and keeps `idemPlus`. At zero
    and below the only points unchanged by squaring are the origin and the unit, and
    projecting by those gives the constant-zero map and the identity, neither of which both
    keeps and discards. Throwing an argument away is available on the hyperbolic face and
    nowhere else. -/
theorem discards_iff_disc_pos (σ : ℝ) : Discards σ ↔ 0 < σ := by
  constructor
  · rintro ⟨e, he, ⟨z, hz, hzp⟩, ⟨w, hw, hwp⟩⟩
    by_contra hcon
    push Not at hcon
    rcases (idempotent_iff_trivial_of_disc_nonpos hcon e).mp he with rfl | rfl
    · exact hw (by rw [← hwp, proj_zero])
    · exact hz (by rw [← hzp, proj_one])
  · intro hσ
    refine ⟨idemPlus σ, idempotent_idemPlus hσ, ⟨idemMinus σ, ?_, ?_⟩,
      ⟨idemPlus σ, ?_, ?_⟩⟩
    · intro h
      have := (halves_idemMinus hσ).2
      rw [h] at this
      simp [(halves_zero σ).2] at this
    · show mul σ (idemMinus σ) (idemPlus σ) = 0
      refine halves_ext hσ ?_ ?_
      · rw [upper_mul hσ.le, (halves_idemMinus hσ).1, (halves_zero σ).1, zero_mul]
      · rw [lower_mul hσ.le, (halves_idemPlus hσ).2, (halves_zero σ).2, mul_zero]
    · exact (idemPlus_nontrivial σ).1
    · show mul σ (idemPlus σ) (idemPlus σ) = idemPlus σ
      exact idempotent_idemPlus hσ

/-- Packing two numbers into the two channels, projecting, and reading gives the first back
and the second gone.

    {σ : ℝ} (hσ : 0 < σ)                                     [the parameter, above zero]
    (u v : ℝ)                                                [two numbers]
    upper σ (proj σ (idemPlus σ) (ofHalves σ u v)) = u       [the first survives]
    ∧                                                        [and]
    lower σ (proj σ (idemPlus σ) (ofHalves σ u v)) = 0       [the second is gone]

    So: the two numbers travel as the two coordinates of one point, and multiplying by an
    idempotent keeps one and empties the other. Nothing outside the algebra is used.

    Not `K`. `K` discards an argument under application, for arguments of every kind including
    functions, and multiplication is not application (`theory/DEAD_ENDS.md`). Here `u` and `v`
    are reals and the operation is a fixed linear projection. The shared feature is discarding
    and that is all. -/
theorem proj_keeps_first_channel {σ : ℝ} (hσ : 0 < σ) (u v : ℝ) :
    upper σ (proj σ (idemPlus σ) (ofHalves σ u v)) = u ∧
      lower σ (proj σ (idemPlus σ) (ofHalves σ u v)) = 0 := by
  obtain ⟨hu, hl⟩ := proj_idemPlus_halves hσ (ofHalves σ u v)
  exact ⟨by rw [hu, upper_ofHalves hσ], hl⟩

/-- Projecting by the other idempotent keeps the second number instead.

    {σ : ℝ} (hσ : 0 < σ)                                      [the parameter, above zero]
    (u v : ℝ)                                                 [two numbers]
    lower σ (proj σ (idemMinus σ) (ofHalves σ u v)) = v       [the second survives]
    ∧                                                         [and]
    upper σ (proj σ (idemMinus σ) (ofHalves σ u v)) = 0       [the first is gone]

    So: the mirror of `proj_keeps_first_channel`, keeping the other number. Which one is discarded
    is a choice of idempotent. -/
theorem proj_keeps_second_channel {σ : ℝ} (hσ : 0 < σ) (u v : ℝ) :
    lower σ (proj σ (idemMinus σ) (ofHalves σ u v)) = v ∧
      upper σ (proj σ (idemMinus σ) (ofHalves σ u v)) = 0 := by
  obtain ⟨hm0, hm1⟩ := halves_idemMinus hσ
  constructor
  · show lower σ (mul σ (ofHalves σ u v) (idemMinus σ)) = v
    rw [lower_mul hσ.le, hm1, mul_one, lower_ofHalves hσ]
  · show upper σ (mul σ (ofHalves σ u v) (idemMinus σ)) = 0
    rw [upper_mul hσ.le, hm0, mul_zero]

/-- A projection in the quotient that throws information away.

    (σ : ℝ)                                     [the parameter]
    ∃ e : AdjoinRoot (rel σ)                    [an element of the quotient ring]
    e * e = e                                   [unchanged by squaring]
    (∃ p, p ≠ 0 ∧ p * e = 0)                    [some non-zero element is sent to zero]
    (∃ q, q ≠ 0 ∧ q * e = q)                    [some non-zero element survives untouched]
    ∧                                           [all of these together] -/
def QuotDiscards (σ : ℝ) : Prop :=
  ∃ e : AdjoinRoot (rel σ), e * e = e ∧
    (∃ p : AdjoinRoot (rel σ), p ≠ 0 ∧ p * e = 0) ∧
    (∃ q : AdjoinRoot (rel σ), q ≠ 0 ∧ q * e = q)

/-- A projection in the quotient throws information away exactly above zero.

    (σ : ℝ)                     [the parameter]
    QuotDiscards σ              [the quotient has a projection that keeps and discards]
    ↔                           [exactly when]
    0 < σ                       [sigma is positive]

    So: `discards_iff_disc_pos` on the ring. Multiplication and zero here are the quotient's
    own, so throwing an argument away is a fact about `ℝ[X]/(X² − σ)` rather than about a
    choice of coordinates on it. -/
theorem quot_discards_iff_disc_pos (σ : ℝ) : QuotDiscards σ ↔ 0 < σ := by
  rw [← discards_iff_disc_pos σ]
  constructor
  · rintro ⟨e, hee, ⟨p, hp, hpe⟩, ⟨q, hq, hqe⟩⟩
    obtain ⟨ez, rfl⟩ := toQuot_surjective σ e
    obtain ⟨pz, rfl⟩ := toQuot_surjective σ p
    obtain ⟨qz, rfl⟩ := toQuot_surjective σ q
    rw [mul_eq_quotient_mul] at hee hpe hqe
    exact ⟨ez, toQuot_injective σ hee,
      ⟨pz, fun h => hp ((toQuot_eq_zero_iff σ pz).mpr h), (toQuot_eq_zero_iff σ _).mp hpe⟩,
      ⟨qz, fun h => hq ((toQuot_eq_zero_iff σ qz).mpr h), toQuot_injective σ hqe⟩⟩
  · rintro ⟨e, he, ⟨z, hz, hze⟩, ⟨w, hw, hwe⟩⟩
    refine ⟨toQuot σ e, by rw [mul_eq_quotient_mul]; exact congrArg (toQuot σ) he,
      ⟨toQuot σ z, fun h => hz ((toQuot_eq_zero_iff σ z).mp h), ?_⟩,
      ⟨toQuot σ w, fun h => hw ((toQuot_eq_zero_iff σ w).mp h), ?_⟩⟩
    · rw [mul_eq_quotient_mul]
      exact (toQuot_eq_zero_iff σ _).mpr hze
    · rw [mul_eq_quotient_mul]
      exact congrArg (toQuot σ) hwe

end Trichotomy
end Apeiron

end
