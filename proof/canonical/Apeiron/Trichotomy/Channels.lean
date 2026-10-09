import Apeiron.Trichotomy.NullBasis
import Apeiron.Trichotomy.ChannelsAnyRing

/-!
# Channels: classification of real characters

@name Channels
@cite weierstrass-1884
@cite peirce-1870
@cite mathlib-sqrt
@grade proved

A channel is a unital real-algebra homomorphism from `Cell` to `ℝ`, stated componentwise to
avoid additional structure. The sign of `σ` determines the number of channels:

```
   σ < 0        none
   σ = 0        one           the first slot
   σ > 0        two           upper and lower
```

Every `z` has the form `a + b·x` for `x = (0, 1)`, so a channel `φ` is determined by `φ x`.
Multiplicativity gives `(φ x)² = σ`. Hence no channel exists for `σ < 0`; the first projection
is the unique channel for `σ = 0`; and `upper` and `lower` are the two channels for `σ > 0`.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

/-- A componentwise definition of a unital real-algebra homomorphism.

    (σ : ℝ)                                    [the parameter]
    (φ : Cell → ℝ)                             [a map from points to plain numbers]
    φ (z + w) = φ z + φ w                      [adding points adds the outputs]
    φ (c • z) = c * φ z                        [stretching a point stretches the output]
    φ (mul σ z w) = φ z * φ w                  [multiplying points multiplies the outputs]
    φ one = 1                                  [the unit goes to one]

    This is `AnyRing.RingChannel` specialized to `R = ℝ`. -/
def Channel (σ : ℝ) (φ : Cell → ℝ) : Prop := AnyRing.RingChannel σ φ

/-- A channel is fixed by its value on the generator.

    {σ : ℝ}                                    [the parameter]
    {φ : Cell → ℝ}                             [a map from points to plain numbers]
    (h : Channel σ φ)                          [which is a read of the algebra]
    (z : Cell)                                 [any point]
    φ z = z.1 + z.2 * φ ((0, 1) : Cell)        [its read is fixed by its value at (0,1)]

    This is `AnyRing.ringChannel_apply`, using the decomposition into the unit and generator. -/
theorem channel_apply {σ : ℝ} {φ : Cell → ℝ} (h : Channel σ φ) (z : Cell) :
    φ z = z.1 + z.2 * φ ((0, 1) : Cell) :=
  AnyRing.ringChannel_apply h z

/-- What a channel gives the generator, squared, is sigma.

    {σ : ℝ}                                              [the parameter]
    {φ : Cell → ℝ}                                       [a map from points to plain numbers]
    (h : Channel σ φ)                                    [which is a read of the algebra]
    φ ((0, 1) : Cell) * φ ((0, 1) : Cell) = σ            [its value at (0,1), squared, is σ]

    This is `AnyRing.ringChannel_gen_sq`; it supplies the equation used by the classification. -/
theorem channel_gen_sq {σ : ℝ} {φ : Cell → ℝ} (h : Channel σ φ) :
    φ ((0, 1) : Cell) * φ ((0, 1) : Cell) = σ :=
  AnyRing.ringChannel_gen_sq h

/-- Below zero there is no channel.

    {σ : ℝ} (hσ : σ < 0)          [the parameter, below zero]
    (φ : Cell → ℝ)                [any map from points to plain numbers]
    ¬ Channel σ φ                 [it is not a read of the algebra]

    A channel would provide a real solution of `x² = σ`, contradicting nonnegativity of
    real squares. -/
theorem no_channel_of_disc_neg {σ : ℝ} (hσ : σ < 0) (φ : Cell → ℝ) : ¬ Channel σ φ := by
  intro h
  have hsq := channel_gen_sq h
  nlinarith [mul_self_nonneg (φ ((0, 1) : Cell))]

/-- At zero the only channel is the first slot.

    (φ : Cell → ℝ)                [a map from points to plain numbers]
    Channel 0 φ                   [it is a read of the algebra when sigma is zero]
    ↔                             [exactly when]
    ∀ z : Cell, φ z = z.1         [it returns the first slot of every point]

    The generator maps to zero, so every channel equals the first projection. -/
theorem channel_iff_fst_of_disc_zero (φ : Cell → ℝ) :
    Channel 0 φ ↔ ∀ z : Cell, φ z = z.1 := by
  constructor
  · intro h z
    have hgen : φ ((0, 1) : Cell) = 0 := mul_self_eq_zero.mp (channel_gen_sq h)
    rw [channel_apply h z, hgen, mul_zero, add_zero]
  · intro hφ
    refine ⟨fun z w => ?_, fun c z => ?_, fun z w => ?_, ?_⟩
    · rw [hφ, hφ, hφ, Prod.fst_add]
    · rw [hφ, hφ, Prod.smul_fst, smul_eq_mul]
    · rw [hφ, hφ, hφ]
      ring
    · rw [hφ one]

/-- At zero or above, `upper` is a channel.

    {σ : ℝ} (hσ : 0 ≤ σ)          [the parameter, at zero or above]
    Channel σ (upper σ)           [the first coordinate is a read of the algebra]

    Specialization of `AnyRing.ringUpper_channel` using `Real.sq_sqrt`. -/
theorem channel_upper {σ : ℝ} (hσ : 0 ≤ σ) : Channel σ (upper σ) :=
  AnyRing.ringUpper_channel (Real.sq_sqrt hσ)

/-- At zero or above, `lower` is a channel.

    {σ : ℝ} (hσ : 0 ≤ σ)          [the parameter, at zero or above]
    Channel σ (lower σ)           [the second coordinate is a read of the algebra]

    Specialization of `AnyRing.ringLower_channel` using `Real.sq_sqrt`. -/
theorem channel_lower {σ : ℝ} (hσ : 0 ≤ σ) : Channel σ (lower σ) :=
  AnyRing.ringLower_channel (Real.sq_sqrt hσ)

/-- Above zero the channels are `upper` and `lower`, and no other.

    {σ : ℝ} (hσ : 0 < σ)                     [the parameter, above zero]
    (φ : Cell → ℝ)                           [a map from points to plain numbers]
    Channel σ φ                              [it is a read of the algebra]
    ↔                                        [exactly when]
    (∀ z : Cell, φ z = upper σ z)            [it is the first coordinate]
    ∨                                        [or]
    (∀ z : Cell, φ z = lower σ z)            [it is the second]

    The equation on the generator has the two roots `±√σ`; `channel_apply` then determines
    the complete map. -/
theorem channel_iff_of_disc_pos {σ : ℝ} (hσ : 0 < σ) (φ : Cell → ℝ) :
    Channel σ φ ↔ (∀ z : Cell, φ z = upper σ z) ∨ (∀ z : Cell, φ z = lower σ z) := by
  constructor
  · intro h
    have hs : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt hσ.le
    have hsq := channel_gen_sq h
    have hfac : (φ ((0, 1) : Cell) - Real.sqrt σ) * (φ ((0, 1) : Cell) + Real.sqrt σ) = 0 := by
      linear_combination hsq - hs
    rcases mul_eq_zero.mp hfac with hgen | hgen
    · left
      intro z
      have hval : φ ((0, 1) : Cell) = Real.sqrt σ := by linarith
      rw [channel_apply h z, hval]
      simp only [upper]
      ring
    · right
      intro z
      have hval : φ ((0, 1) : Cell) = -Real.sqrt σ := by linarith
      rw [channel_apply h z, hval]
      simp only [lower]
      ring
  · rintro (hφ | hφ)
    · obtain ⟨ha, hst, hm, ho⟩ := channel_upper (σ := σ) hσ.le
      exact ⟨fun z w => by rw [hφ, hφ, hφ]; exact ha z w,
        fun c z => by rw [hφ, hφ]; exact hst c z,
        fun z w => by rw [hφ, hφ, hφ]; exact hm z w,
        by rw [hφ]; exact ho⟩
    · obtain ⟨ha, hst, hm, ho⟩ := channel_lower (σ := σ) hσ.le
      exact ⟨fun z w => by rw [hφ, hφ, hφ]; exact ha z w,
        fun c z => by rw [hφ, hφ]; exact hst c z,
        fun z w => by rw [hφ, hφ, hφ]; exact hm z w,
        by rw [hφ]; exact ho⟩

/-- Two distinct channels exist exactly when sigma is positive.

    (σ : ℝ)                             [the parameter]
    ∃ φ ψ : Cell → ℝ                    [there are two maps from points to plain numbers]
    Channel σ φ ∧ Channel σ ψ           [each is a read of the algebra]
    φ ≠ ψ                               [and they disagree somewhere]
    ↔                                   [exactly when]
    0 < σ                               [sigma is positive]

    The forward implication excludes the negative and zero cases. For `σ > 0`, `upper` and
    `lower` differ on the generator. -/
theorem two_channels_iff_disc_pos (σ : ℝ) :
    (∃ φ ψ : Cell → ℝ, Channel σ φ ∧ Channel σ ψ ∧ φ ≠ ψ) ↔ 0 < σ := by
  constructor
  · rintro ⟨φ, ψ, hφ, hψ, hne⟩
    rcases lt_trichotomy σ 0 with hσ | hσ | hσ
    · exact absurd hφ (no_channel_of_disc_neg hσ φ)
    · subst hσ
      refine absurd (funext fun z => ?_) hne
      rw [(channel_iff_fst_of_disc_zero φ).mp hφ z, (channel_iff_fst_of_disc_zero ψ).mp hψ z]
    · exact hσ
  · intro hσ
    refine ⟨upper σ, lower σ, channel_upper hσ.le, channel_lower hσ.le, fun h => ?_⟩
    have h1 : upper σ ((0, 1) : Cell) = lower σ ((0, 1) : Cell) := by rw [h]
    have hs : 0 < Real.sqrt σ := Real.sqrt_pos.mpr hσ
    simp only [upper, lower] at h1
    norm_num at h1
    linarith

end Trichotomy
end Apeiron

end
