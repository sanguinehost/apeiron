import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.NullBasis

/-!
# Executable basis authority over any commutative ring

@name Executable Carrier Basis Authority
@grade proved

The carrier modules read every amount from the trichotomy norm form, and
`Apeiron.Physics.CarrierAuthority` realizes a nonnegative real amount by
`Real.sqrt d • one`. `Real.sqrt` is not computable, so that realization cannot be the
construction an exact integer or machine-word execution closes over: it proves the real
existence of a preimage and names no operation a compiled program can run.

This module carries the executable part on its own. A basis point `ringBasis a` is the pair
`(a, 0)`, the amplitude in the first slot and zero in the second, and its norm form is `a ^ 2`
by ring algebra alone. The amount is realized when an amplitude and a proof that it squares to
the amount are supplied; the proof is a hypothesis, never a search. An already-present
`RingCell` whose second slot is zero is the same data, and its norm form is its first slot
squared.

## The obstruction, as a hypothesis

A root is not automatic. `no_sq_two_int` proves that no whole number squares to two, so over
the integers the amount two has no amplitude witness at all. The interface therefore takes the
squaring proof as an argument. Nonnegativity of a real amount is not available in a general
ring, and even where it is, it does not supply a root.

## Root

| from | here |
|---|---|
| `Trichotomy.AnyRing.ringNormForm` | `ringNormForm_ringBasis`, `ringNormForm_of_snd_eq_zero` |
| `Trichotomy.two_not_sq` | `no_sq_two_int` |

The first row is the ring identity that a basis point measures to its amplitude squared, which
needs no square root and no order. The second row fixes the scope: the root hypothesis is not
vacuous and is not discharged here.
-/

namespace Apeiron
namespace Physics

open Trichotomy
open Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- The basis point of an explicit amplitude: the amplitude first, zero second.

    (a : R)                  [the amplitude, an element of the ring]
    R                        [the commutative ring the point is built over]
    RingCell R               [a point: two ring elements side by side, a + b·x]
    (a, 0)                   [the amplitude, then zero]

    So: the basis direction itself, stretched by the amplitude, with no root taken. -/
def ringBasis (a : R) : RingCell R := (a, 0)

/-- The norm form of a basis point is its amplitude squared.

    (σ : R)                  [the parameter: what the adjoined direction squares to]
    (a : R)                  [the amplitude, an element of the ring]
    R                        [the commutative ring the point is built over]
    ringNormForm σ (ringBasis a) = a ^ 2
                             [measuring the basis point returns the amplitude squared]
    `ringBasis`                 [the amplitude in the first slot, zero in the second]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    So: the second slot is zero, so the σ term drops and the first slot's square remains. No
    square root and no order enter. -/
theorem ringNormForm_ringBasis (σ : R) (a : R) :
    ringNormForm σ (ringBasis a) = a ^ 2 := by
  simp [ringBasis, ringNormForm]

/-- A supplied squaring proof turns a basis point into a realization of the amount.

    (σ : R)                  [the parameter: what the adjoined direction squares to]
    (a d : R)                [the amplitude and the amount it is meant to realize]
    (ha : a ^ 2 = d)         [the amplitude squares to the amount]
    R                        [the commutative ring the point is built over]
    ringNormForm σ (ringBasis a) = d
                             [measuring the basis point returns the amount]
    `ringBasis`                 [the amplitude in the first slot, zero in the second]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    So: the readback is the supplied proof, not a search for a root. The amount is whatever the
    caller's amplitude squares to. -/
theorem ringNormForm_ringBasis_of_sq (σ : R) {a d : R} (ha : a ^ 2 = d) :
    ringNormForm σ (ringBasis a) = d := by
  rw [ringNormForm_ringBasis, ha]

/-- An already-present point with second slot zero has norm form its first slot squared.

    (σ : R)                  [the parameter: what the adjoined direction squares to]
    (z : RingCell R)         [the point already present]
    (hz : z.2 = 0)           [its second slot is zero]
    R                        [the commutative ring the point is built over]
    ringNormForm σ z = z.1 ^ 2
                             [measuring the point returns its first slot squared]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    So: a state already on the basis direction needs no reconstruction, only the observation
    that its σ term vanishes. -/
theorem ringNormForm_of_snd_eq_zero (σ : R) {z : RingCell R} (hz : z.2 = 0) :
    ringNormForm σ z = z.1 ^ 2 := by
  simp [ringNormForm, hz]

/-- An already-present basis point realizes an amount when its first slot squares to it.

    (σ : R)                  [the parameter: what the adjoined direction squares to]
    (z : RingCell R)         [the point already present]
    (d : R)                  [the amount it is meant to realize]
    (hz : z.2 = 0)           [its second slot is zero]
    (hd : z.1 ^ 2 = d)       [its first slot squares to the amount]
    R                        [the commutative ring the point is built over]
    ringNormForm σ z = d     [measuring the point returns the amount]

    So: the already-present state and the supplied squaring proof together are the realization,
    with no construction step. -/
theorem ringNormForm_of_snd_eq_zero_of_sq (σ : R) {z : RingCell R} {d : R}
    (hz : z.2 = 0) (hd : z.1 ^ 2 = d) : ringNormForm σ z = d := by
  rw [ringNormForm_of_snd_eq_zero σ hz, hd]

/-- No whole number squares to two.

    ¬ ∃ a : ℤ, a ^ 2 = (2 : ℤ)   [there is no whole number whose square is two]
    ℤ                            [the whole numbers: the counting numbers and their negatives]
    ∃                            [there is at least one]
    ¬                            [this is not true]

    So: the squaring hypothesis is not vacuous. Over the integers the amount two has no
    amplitude, so a caller cannot conjure one and must supply it. This is the obstruction the
    interface carries as a hypothesis rather than a claim that roots always exist. -/
theorem no_sq_two_int : ¬ ∃ a : ℤ, a ^ 2 = (2 : ℤ) := by
  rintro ⟨a, ha⟩
  exact two_not_sq ⟨a, ha.symm⟩

end Physics
end Apeiron
