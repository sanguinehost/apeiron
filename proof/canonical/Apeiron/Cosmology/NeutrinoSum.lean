import Apeiron.Physics.Flavour

/-!
# The clean-sector rule, named, and the number it produces

@name NeutrinoSum
@cite koide-2000
@cite rodejohann-zhang-2011
@cite nufit-6-2024
@cite desi-dr2-2025
@grade proved

The predecessor's sharpest registered prediction is a neutrino mass sum of `70.1 ± 0.8` meV with
normal ordering. Its own campaign document says the number "is carried by no theorem", and that
the step producing it is a posited rule: that the chargeless, colourless sector sits at the unit
of a per-sector quantity `3δ/Q`, where `δ` is the azimuth of the mass cone and `Q` its quotient.

That rule is `OnTheMonad`, stated here as a named hypothesis on the algebra's own two cone
coordinates, so nothing is hidden. `clean_spectrum_cubic` is what the rule buys: given it, all
three samples of the cone solve one cubic whose right-hand side is the quotient alone, so a sector
obeying the rule has its spectrum fixed by one number. That is a theorem, and it is conditional on
the rule in the type.

What the rule does **not** do is say which sector obeys it. `theory/DEAD_ENDS.md` records the three
routes tried here to derive it and why each failed; the shortest statement of the obstruction is
that `koide_of_trinity` fixes the quotient from the amplitude alone and `trinity_root` fixes the
spectrum from the azimuth alone, so the two cone coordinates are independent and no theorem in
this corpus relates them.

The algebraic order-three content of the chargeless sector is carried over any commutative ring in
`Apeiron/Cosmology/AnyRing.lean` and specialized to the reals in `Majorana.lean`. This file is the
real and calibration side of the subject. `clean_spectrum_cubic` is a trigonometric identity, so it
is a real instance of `Physics.trinity_root` and has no ring-generic content beyond the order-three
relation the trig functions encode. The numerical theorems below are calibration instances: the
measured splittings, masses and bounds are inputs, and no decimal here follows from the ring
axioms.

## Where the number stands

The arithmetic below is checked rather than derived: the registered triple is compared against the
measured splittings, summed, and placed against the current bounds.

* It reproduces the measured splittings to well inside their errors
  (`registered_spectrum_matches_the_splittings`).
* It sums to `70.408` meV (`registered_spectrum_sums`).
* That is above the floor oscillations alone impose in normal ordering, about `58.99` meV
  (`registered_sum_is_above_the_oscillation_floor`), so it is not the minimal spectrum.
* It is above the `64.2` meV bound DESI DR2 with the microwave background gives at 95 per cent
  in frozen-Λ cosmology (`registered_sum_is_above_the_standard_bound`). That is not the registered
  kill. The predecessor pre-registered that a bound taken in the frozen-Λ class does not count when
  the same data reject that class, because its own prediction is that `w(z)` crosses `−1`
  (`DARK_ENERGY_CROSSING`, phantom past to quintessence today), so the sum and the crossing stand or
  fall together in the running class.
* It is below the `163` meV bound the same data give in the running class
  (`registered_sum_is_below_the_running_bound`), the class the framework predicts and DESI DR2
  prefers, so the prediction stands there.

The kill criterion, pre-registered so it cannot migrate: a measured sum outside `[68, 72]` meV in
the model class the data prefer, or a frozen-Λ world, which kills the crossing with it. Neither has
fired. The theorems state the two bounds flatly; which class counts is the registration's call.

## Root

Instantiates `Physics.trinity_root` from `Apeiron/Physics/Flavour.lean`, which rests on the orbit
of the order-three element `Trichotomy.exists_cube_root_one_iff_disc_neg` supplies, so everything
here is on the elliptic face. The edge:

* `clean_spectrum_cubic` is `trinity_root` with the rule substituted, so the rule's whole content
  is that it collapses the cubic's right-hand side onto the quotient. It is a real instance: the
  cosine and the triple angle are real, and the ring-generic order-three content lives in
  `Apeiron/Cosmology/AnyRing.lean` instead.

`OnTheMonad` is a posited real input and instantiates nothing. The numerical theorems are rational
arithmetic on measured inputs, so they are calibration instances and instantiate nothing either.
-/

namespace Apeiron
namespace Cosmology

open Trichotomy Physics Real

/-- The posited rule: three times the cone's azimuth is its quotient.

    ℝ                                          [the real numbers: every number on the line]
    3 * δ = Q                                  [three times the azimuth is the quotient]
    OnTheMonad                                 [the sector sits at the unit of the ratio]

    So: the rule, named, with nothing under it. No theorem in this corpus says any sector obeys
    it, and `theory/DEAD_ENDS.md` records what was tried. -/
def OnTheMonad (δ Q : ℝ) : Prop := 3 * δ = Q

/-- **Under the rule, the whole spectrum solves one cubic set by the quotient alone.**

    ℝ                                                            [the real numbers]
    ℤ                                                            [whole numbers, either sign]
    OnTheMonad                                                   [the sector sits at the unit]
    cos                                                          [how far a turn goes across]

    So: this is what the rule buys and all it buys. Given it, the three samples of the cone are
    the three roots of `4c³ − 3c = cos Q`, so the spectrum is one number modulo the size and the
    amplitude. The rule itself is derived nowhere. This is the real instance of
    `Physics.trinity_root`: the cosine and the triple angle are real, and the identity is
    trigonometric rather than a consequence of the ring axioms. -/
theorem clean_spectrum_cubic (δ Q : ℝ) (k : ℤ) (h : OnTheMonad δ Q) :
    4 * Real.cos (δ + 2 * π * (k : ℝ) / 3) ^ 3 - 3 * Real.cos (δ + 2 * π * (k : ℝ) / 3)
      = Real.cos Q := by
  rw [trinity_root]
  unfold OnTheMonad at h
  rw [h]

/-- **The registered triple reproduces the measured splittings.**

    ∧                                                            [both are true]
    <                                                            [is smaller than]
    ℝ                                                            [the real numbers]

    So: the two differences of squares match the measured mass-squared splittings, `75.0` and
    `2510` in units of meV squared, to well inside their errors. A calibration instance: a
    consistency check on the registered numbers, not a derivation of them. -/
theorem registered_spectrum_matches_the_splittings :
    |(11.747 : ℝ) ^ 2 - 7.936 ^ 2 - 75.0| < 0.2
      ∧ |(50.725 : ℝ) ^ 2 - 7.936 ^ 2 - 2510| < 5 := by
  constructor <;> rw [abs_lt] <;> constructor <;> norm_num

/-- **The registered triple sums to 70.408 meV.**

    <                                                            [is smaller than]
    ℝ                                                            [the real numbers]

    So: the arithmetic of the registered prediction, stated once so the comparisons below have
    something exact to run against. A calibration instance: the number is an input. -/
theorem registered_spectrum_sums : |(7.936 : ℝ) + 11.747 + 50.725 - 70.408| < 0.001 := by
  rw [abs_lt]
  constructor <;> norm_num

/-- **The registered sum is above the floor oscillations alone impose.**

    <                                                            [is smaller than]
    ℝ                                                            [the real numbers]

    So: with the lightest mass sent to zero, normal ordering still forces about `58.99` meV. The
    registered triple sits above that, so it is not the minimal spectrum and carries a lightest
    mass. A calibration instance: the floor is a measured input. -/
theorem registered_sum_is_above_the_oscillation_floor : (58.99 : ℝ) < 70.408 := by norm_num

/-- **The registered sum is above the bound the standard cosmology now gives.**

    <                                                            [is smaller than]
    ℝ                                                            [the real numbers]

    So: DESI DR2 with the microwave background bounds the sum below `64.2` meV at 95 per cent in
    frozen-Λ cosmology, and `70.408` is above it. This is not the registered falsifier: the
    registration says a frozen-Λ bound does not count when the same data reject frozen Λ, which
    DESI DR2 does in favour of the running class the framework itself predicts. On record, not a
    kill. A calibration instance: the bound is a measured input. -/
theorem registered_sum_is_above_the_standard_bound : (64.2 : ℝ) < 70.408 := by norm_num

/-- **The registered sum is below the bound the wider class gives.**

    <                                                            [is smaller than]
    ℝ                                                            [the real numbers]

    So: in the running class, the one the framework predicts (`w(z)` crosses `−1`) and the data
    prefer, the same data bound the sum below about `163` meV and `70.408` is inside. This is the
    registered class, and the prediction stands in it. A calibration instance: the bound is a
    measured input. -/
theorem registered_sum_is_below_the_running_bound : (70.408 : ℝ) < 163 := by norm_num

end Cosmology
end Apeiron
