/-
One-bit unit actions on the two ring faces (`APEIRON-ONEBIT-W6-T1`).

The two-slot algebra `R[x]/(x² − σ)` (`Apeiron.Trichotomy.AnyRing.RingCell`) has a channel
projection `Nous.Model.channelPair r z = (ringUpper r z, ringLower r z)` at a root `r` of `σ`. On the
hyperbolic face `σ = 1` with `r = 1`, `channelPair 1` is the butterfly `(z₁ + z₂, z₁ − z₂)`, and the
unit group `{1, −1, j, −j}` (`j = (0, 1)`, `j² = 1`) acts on a point as an independent sign on each
channel: multiplication by a unit contributes the componentwise sign pair `(1, 1)`, `(−1, −1)`,
`(1, −1)`, `(−1, 1)` (`channelPair_one_mul_one`, `..._negOne`, `..._j`, `..._negJ`). The four units
form `ℤ/2 × ℤ/2` (`hyper_mul_table`).

On the elliptic face `σ = −1` with `i = (0, 1)`, `i² = −1`, multiplication by `i` is the quarter
turn `(a, b) ↦ (−b, a)` (`mul_i_components`); `i⁴ = 1`, and `−1` is the only element of order two
among `{1, −1, i, −i}` (`elliptic_order_two_unique`), so the units form `ℤ/4`.

Every statement is over `RingCell ℤ` and names the canonical corpus declarations
`ringMul`, `ringOne`, `ringUpper_mul`, `ringLower_mul`, `Nous.Model.channelPair`; the closure table
is closed by `decide`. No `sorry`, `axiom` or `maxHeartbeats 0` appears.
-/
import Nous.Model.Decomposition
import Mathlib.Tactic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Apeiron.Trichotomy.AnyRing
open Nous.Model

namespace OneBitUnits

/-! ## A unit acts on the channel coordinates by a sign pair -/

/-- Left multiplication by a unit `u` intertwines with `channelPair 1`: on the two channel
coordinates it is the componentwise product with `(ringUpper 1 u, ringLower 1 u)`. -/
theorem channelPair_ringMul_left (u z : RingCell ℤ) :
    channelPair (1 : ℤ) (ringMul 1 u z) =
      (ringUpper (1 : ℤ) u * (channelPair (1 : ℤ) z).1,
       ringLower (1 : ℤ) u * (channelPair (1 : ℤ) z).2) := by
  have h1 : (1 : ℤ) ^ 2 = 1 := by norm_num
  apply Prod.ext
  · simp only [channelPair]
    exact ringUpper_mul h1 u z
  · simp only [channelPair]
    exact ringLower_mul h1 u z

/-! ## The four units at `σ = 1` -/

/-- The generator `j = (0, 1)`, with `j² = 1` at `σ = 1`. -/
def j : RingCell ℤ := (0, 1)

/-- `−1` as a ring point. -/
def negOne : RingCell ℤ := (-1, 0)

/-- `−j`. -/
def negJ : RingCell ℤ := (0, -1)

/-- Multiplication by the unit `1` acts as the sign pair `(1, 1)`. -/
theorem channelPair_one_mul_one (z : RingCell ℤ) :
    channelPair (1 : ℤ) (ringMul 1 ringOne z) =
      ((1 : ℤ) * (channelPair (1 : ℤ) z).1, (1 : ℤ) * (channelPair (1 : ℤ) z).2) := by
  rw [channelPair_ringMul_left]
  ext <;> simp only [ringUpper, ringLower] <;> ring

/-- Multiplication by the unit `−1` acts as the sign pair `(−1, −1)`. -/
theorem channelPair_one_mul_negOne (z : RingCell ℤ) :
    channelPair (1 : ℤ) (ringMul 1 negOne z) =
      ((-1 : ℤ) * (channelPair (1 : ℤ) z).1, (-1 : ℤ) * (channelPair (1 : ℤ) z).2) := by
  rw [channelPair_ringMul_left]
  ext <;> simp only [negOne, ringUpper, ringLower] <;> ring

/-- Multiplication by the unit `j` acts as the sign pair `(1, −1)`. -/
theorem channelPair_one_mul_j (z : RingCell ℤ) :
    channelPair (1 : ℤ) (ringMul 1 j z) =
      ((1 : ℤ) * (channelPair (1 : ℤ) z).1, (-1 : ℤ) * (channelPair (1 : ℤ) z).2) := by
  rw [channelPair_ringMul_left]
  ext <;> simp only [j, ringUpper, ringLower] <;> ring

/-- Multiplication by the unit `−j` acts as the sign pair `(−1, 1)`. -/
theorem channelPair_one_mul_negJ (z : RingCell ℤ) :
    channelPair (1 : ℤ) (ringMul 1 negJ z) =
      ((-1 : ℤ) * (channelPair (1 : ℤ) z).1, (1 : ℤ) * (channelPair (1 : ℤ) z).2) := by
  rw [channelPair_ringMul_left]
  ext <;> simp only [negJ, ringUpper, ringLower] <;> ring

/-! ## The unit group is `ℤ/2 × ℤ/2` -/

/-- The multiplication table of `{1, −1, j, −j}`: closed, with each non-identity element of order
two. Every entry is a literal computation on `RingCell ℤ`. -/
theorem hyper_mul_table :
    ringMul (1 : ℤ) negOne negOne = ringOne ∧
    ringMul (1 : ℤ) negOne j = negJ ∧
    ringMul (1 : ℤ) negOne negJ = j ∧
    ringMul (1 : ℤ) j negOne = negJ ∧
    ringMul (1 : ℤ) j j = ringOne ∧
    ringMul (1 : ℤ) j negJ = negOne ∧
    ringMul (1 : ℤ) negJ negOne = j ∧
    ringMul (1 : ℤ) negJ j = negOne ∧
    ringMul (1 : ℤ) negJ negJ = ringOne := by
  decide

/-! ## The elliptic face `σ = −1` is `ℤ/4` -/

/-- The generator `i = (0, 1)` at `σ = −1`, with `i² = −1`. -/
def i : RingCell ℤ := (0, 1)

/-- Multiplication by `i` is the quarter turn `(a, b) ↦ (−b, a)`. -/
theorem mul_i_components (z : RingCell ℤ) :
    ringMul (-1 : ℤ) i z = (-(z.2), z.1) := by
  simp only [i, ringMul, Prod.mk.injEq]
  constructor <;> ring

/-- `i² = −1`. -/
theorem i_sq : ringMul (-1 : ℤ) i i = -ringOne := by
  decide

/-- `i⁴ = 1`, i.e. `(i²)² = 1`. -/
theorem i_four :
    ringMul (-1 : ℤ) (ringMul (-1 : ℤ) i i) (ringMul (-1 : ℤ) i i) = ringOne := by
  decide

/-- `i` is a unit, with inverse `−i`. -/
theorem i_mul_negI : ringMul (-1 : ℤ) i (-i) = ringOne := by
  decide

/-- Among `{1, −1, i, −i}`, the only element of order two is `−1`. -/
theorem elliptic_order_two_unique :
    ringMul (-1 : ℤ) (-ringOne) (-ringOne) = ringOne ∧
    (-ringOne : RingCell ℤ) ≠ ringOne ∧
    ringMul (-1 : ℤ) i i ≠ ringOne ∧
    ringMul (-1 : ℤ) (-i) (-i) ≠ ringOne := by
  decide

end OneBitUnits

end apeiron_word
