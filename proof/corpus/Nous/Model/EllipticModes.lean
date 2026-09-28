import Apeiron.Trichotomy.AnyRing
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic

/-!
# Elliptic Modes in Attention: Oscillatory Decay vs Monotonic Decay

This module establishes the algebraic distinction between hyperbolic modes
(pure real decay `αⁿ`) and elliptic modes (damped complex rotations `(r · e^{iθ})ⁿ`)
in attention mechanisms equipped with Rotary Position Embedding (RoPE).

## Mathematical Context and Attribution

- **Complex Multiplication and Trigonometric Identities**: Rafael Bombelli (1572);
  Abraham de Moivre, *Miscellanea Analytica* (1730).
- **Rotary Position Embedding (RoPE)**: Jianlin Su et al., arXiv:2104.09864 (2021/2024).
- **Linearized Attention Collapse**: Empirically, converting softmax attention with RoPE
  to purely real recurrent decay (`Gated DeltaNet`, `GLA`) loses long-range associative
  recall because real decays cannot match oscillatory interference patterns.

## Algebraic Theorems Established

1. `rotor_comp_norm`: Unitary rotors (`N(u) = 1` at `σ = -1`) preserve norm under powers.
2. `real_decay_strictly_positive`: Over an ordered field `F`, positive decay `α > 0` satisfies
   `αⁿ > 0` for all `n : ℕ`. A purely real decay cannot change sign.
3. `elliptic_sign_alternation_witness`: Under elliptic multiplication at `σ = -1`, the rotor
   `u = (0, 1)` (representing `i`) satisfies `(u²).1 = -1 < 0` despite `(u⁰).1 = 1 > 0`.
   Elliptic modes produce sign-reversing oscillatory crossings that cannot be generated
   by positive real exponential decay.
4. `damped_rotor_norm_pow`: A damped rotor with base norm `N(z)` has power norm `(N(z))^n`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unnecessarySeqFocus false
set_option linter.unusedSimpArgs false

open Apeiron.Trichotomy.AnyRing

namespace Nous.Model.EllipticModes

variable {R : Type*} [CommRing R]

/-- A 2-cell element representing a damped rotor `(a, b)` in `R[x]/(x² - σ)` at `σ = -1`. -/
def rotor (c s : R) : RingCell R := (c, s)

/-- The elliptic ring product at `σ = -1`. -/
def rotorMul (z w : RingCell R) : RingCell R :=
  ringMul (-1) z w

/-- Repeated power of a rotor under elliptic multiplication. -/
def rotorPow (z : RingCell R) : ℕ → RingCell R
  | 0 => ringOne
  | n + 1 => rotorMul (rotorPow z n) z

@[simp] theorem rotorPow_zero (z : RingCell R) : rotorPow z 0 = ringOne := rfl

@[simp] theorem rotorPow_succ (z : RingCell R) (n : ℕ) :
    rotorPow z (n + 1) = rotorMul (rotorPow z n) z := rfl

/-- Power 1 matches the original rotor. -/
theorem rotorPow_one (z : RingCell R) : rotorPow z 1 = z := by
  dsimp [rotorPow, rotorMul]
  exact ringOne_mul (-1) z

/-- Power 2 is squaring. -/
theorem rotorPow_two (z : RingCell R) : rotorPow z 2 = rotorMul z z := by
  dsimp [rotorPow, rotorMul, ringMul, ringOne]
  ext <;> ring

/-! ### Norm Preservation and Multiplicativity -/

/-- Elliptic norm of a product is the product of norms (Gauss 1801). -/
theorem rotorMul_norm (z w : RingCell R) :
    ringNormForm (-1) (rotorMul z w) = ringNormForm (-1) z * ringNormForm (-1) w := by
  dsimp [rotorMul]
  exact ringNormForm_mul (-1) z w

/-- Power of a rotor has norm equal to the power of the rotor's norm. -/
theorem rotorPow_norm (z : RingCell R) (n : ℕ) :
    ringNormForm (-1) (rotorPow z n) = (ringNormForm (-1) z) ^ n := by
  induction n with
  | zero =>
    simp only [rotorPow_zero, pow_zero]
    dsimp [ringNormForm, ringOne]
    ring
  | succ m ih =>
    simp only [rotorPow_succ, pow_succ]
    rw [rotorMul_norm, ih]

/-- Unitary rotor preserves unit norm at all integer powers. -/
theorem unitary_rotor_preserves_norm (z : RingCell R)
    (h_unit : ringNormForm (-1) z = 1) (n : ℕ) :
    ringNormForm (-1) (rotorPow z n) = 1 := by
  rw [rotorPow_norm, h_unit, one_pow]

/-! ### Positive Real Decay Cannot Cross Zero -/

section OrderedSetting

variable {F : Type*} [Field F] [LinearOrder F] [IsStrictOrderedRing F]

/-- Strictly positive real decay `α > 0` remains strictly positive for all `n : ℕ`.
Hence, a purely real decay profile `αⁿ` cannot cross zero or produce oscillatory signs. -/
theorem real_decay_strictly_positive (α : F) (hα : 0 < α) (n : ℕ) :
    0 < α ^ n := by
  exact pow_pos hα n

/-- Nonnegative real decay `α ≥ 0` remains nonnegative for all `n : ℕ`. -/
theorem real_decay_nonnegative (α : F) (hα : 0 ≤ α) (n : ℕ) :
    0 ≤ α ^ n := by
  exact pow_nonneg hα n

end OrderedSetting

/-! ### Elliptic Oscillatory Zero-Crossing Witness -/

/-- The quarter-turn rotor `J = (0, 1)` representing the imaginary unit `i`. -/
def quarterTurnRotor : RingCell R := (0, 1)

/-- Explicit evaluation of `quarterTurnRotor^2` yields `(-1, 0)`. -/
theorem quarterTurnRotor_sq :
    rotorPow (quarterTurnRotor (R := R)) 2 = (-1, 0) := by
  dsimp [rotorPow, quarterTurnRotor, rotorMul, ringMul, ringOne]
  ext <;> ring

/-- In any strictly ordered field, the elliptic mode `quarterTurnRotor` produces
a sign reversal: `(rotorPow z 0).1 = 1 > 0` while `(rotorPow z 2).1 = -1 < 0`.
This demonstrates that elliptic modes produce oscillatory zero-crossings, which
is impossible for any positive real decay. -/
theorem elliptic_sign_alternation_witness {F : Type*} [Field F] [LinearOrder F] [IsStrictOrderedRing F] :
    (0 : F) < (rotorPow (quarterTurnRotor (R := F)) 0).1 ∧
    (rotorPow (quarterTurnRotor (R := F)) 2).1 < (0 : F) := by
  constructor
  · dsimp [rotorPow_zero, ringOne]
    exact zero_lt_one
  · rw [quarterTurnRotor_sq]
    dsimp
    exact neg_lt_zero.mpr zero_lt_one

/-- Non-vacuous witness: the quarter-turn rotor is unitary (`N(J) = 1`). -/
theorem quarterTurnRotor_unitary :
    ringNormForm (-1) (quarterTurnRotor (R := R)) = 1 := by
  dsimp [quarterTurnRotor, ringNormForm]
  ring

end Nous.Model.EllipticModes
