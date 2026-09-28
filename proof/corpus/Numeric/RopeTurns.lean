/-
# RoPE on the turn ring: the angle as an element of `Z/2^64`

`src/numeric/fixed_trig.rs` is total only on `[-4096, 4096]` Q32.32 radians, and
`RopeTable` forms the angle as `pos · freq` in radians. A frequency stored as a `u64`
fraction of a turn makes the phase `(f · p) mod 2^64`, so the turn ring has no window.

This module is the Mathlib-level integer/real specification of that construction:
`phase` and its homomorphism laws, the signed `lift` into `[-2^63, 2^63)`, the Q32.32
rounding `toRad`, its window bound, and the error of `toRad (lift (phase f p))` against
the ideal angle `p · θ` on the circle. It is the reference the extracted
`turn_to_radians` is refined against; it contains no extracted code.

## Trust boundary

The statements are about integers and the real numbers. They do not by themselves link
the Rust `turn_to_radians`; that refinement is a later task. They do not reach
`rustc`/LLVM, the kernel ABI or any measured quantity.
-/
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

namespace Numeric.RopeTurns

/-- `round(2π · 2^32)`, the Q32.32 radian-per-turn scale. Read from
`src/numeric/fixed_trig.rs` (`TWO_PI_Q32`), not recomputed. -/
def TWO_PI_Q32 : ℤ := 26986075409

/-- The turn-ring modulus `2^64`: one full turn. -/
def M : ℕ := 2 ^ 64

theorem M_eq : M = 2 ^ 64 := rfl

/-- The phase of position `p` at frequency `f`, the `u64` fraction of a turn
`(f · p) mod 2^64`. -/
def phase (f p : ℕ) : ℕ := (f * p) % 2 ^ 64

theorem phase_lt (f p : ℕ) : phase f p < 2 ^ 64 :=
  Nat.mod_lt _ (by norm_num)

/-- `phase` is a homomorphism `ℕ → Z/2^64` under addition of positions. -/
theorem phase_add (f p q : ℕ) :
    phase f (p + q) = (phase f p + phase f q) % 2 ^ 64 := by
  unfold phase
  rw [Nat.mul_add, Nat.add_mod]

/-- `phase` is a homomorphism under multiplication of the frequency: the `rope_orbit`
analogue for the turn ring (`proof/corpus/Nous/Model/RopeOrbit.lean`, `rope_orbit`). -/
theorem phase_mul (f k p : ℕ) :
    phase f (k * p) = (k * phase f p) % 2 ^ 64 := by
  unfold phase
  rw [show f * (k * p) = k * (f * p) by ring]
  conv_lhs => rw [Nat.mul_mod]
  conv_rhs => rw [Nat.mul_mod, Nat.mod_mod]

/-- The signed lift `φ ↦ φ - 2^64 · [φ ≥ 2^63]`, the `phase as i64` of the Rust path. -/
def lift (φ : ℕ) : ℤ := if φ < 2 ^ 63 then (φ : ℤ) else (φ : ℤ) - 2 ^ 64

/-- The lift of a reduced phase lands in the signed `i64` range. -/
theorem lift_range (φ : ℕ) (h : φ < 2 ^ 64) :
    -(2 ^ 63 : ℤ) ≤ lift φ ∧ lift φ < (2 ^ 63 : ℤ) := by
  unfold lift
  split_ifs with hlt
  · have h0 : (0 : ℤ) ≤ (φ : ℤ) := Int.natCast_nonneg φ
    have hc : (φ : ℤ) < (2 : ℤ) ^ 63 := by exact_mod_cast hlt
    constructor <;> omega
  · have hc : (2 : ℤ) ^ 63 ≤ (φ : ℤ) := by
      exact_mod_cast (not_lt.mp hlt)
    have hc2 : (φ : ℤ) < (2 : ℤ) ^ 64 := by exact_mod_cast h
    constructor <;> omega

/-- The lift is congruent to the phase modulo `2^64`. -/
theorem lift_congr (φ : ℕ) : lift φ ≡ (φ : ℤ) [ZMOD (2 ^ 64 : ℤ)] := by
  unfold lift
  split_ifs with hlt
  · exact Int.ModEq.refl _
  · rw [Int.modEq_iff_dvd]
    exact ⟨1, by ring⟩

/-- `r = (s · TWO_PI_Q32 + 2^63) / 2^64`: the Q32.32 rounding of `s` turns, computed in
`i128` in Rust. `Int` division by the positive `2^64` is floor division. -/
def toRad (s : ℤ) : ℤ := (s * TWO_PI_Q32 + 2 ^ 63) / 2 ^ 64

/-- The rounding error of `toRad` against the exact turn value is at most half a Q32.32
unit. -/
theorem toRad_round (s : ℤ) :
    |(toRad s : ℝ) - (s : ℝ) * TWO_PI_Q32 / (2 : ℝ) ^ 64| ≤ 1 / 2 := by
  have hD : (0 : ℤ) < 2 ^ 64 := by norm_num
  have hDpos : (0 : ℝ) < (2 : ℝ) ^ 64 := by positivity
  have hleZ : toRad s * 2 ^ 64 ≤ s * TWO_PI_Q32 + 2 ^ 63 := by
    unfold toRad
    exact Int.ediv_mul_le _ (ne_of_gt hD)
  have hltZ : s * TWO_PI_Q32 + 2 ^ 63 < (toRad s + 1) * 2 ^ 64 := by
    unfold toRad
    exact Int.lt_ediv_add_one_mul_self _ hD
  have hleR : (toRad s : ℝ) * (2 : ℝ) ^ 64 ≤ (s : ℝ) * TWO_PI_Q32 + (2 : ℝ) ^ 63 := by
    exact_mod_cast hleZ
  have hltR : (s : ℝ) * TWO_PI_Q32 + (2 : ℝ) ^ 63 < ((toRad s : ℝ) + 1) * (2 : ℝ) ^ 64 := by
    exact_mod_cast hltZ
  rw [abs_le]
  constructor
  · have h : (-(1 / 2 : ℝ)) * (2 : ℝ) ^ 64
        ≤ ((toRad s : ℝ) - (s : ℝ) * TWO_PI_Q32 / (2 : ℝ) ^ 64) * (2 : ℝ) ^ 64 := by
      rw [sub_mul, div_mul_cancel₀ _ (ne_of_gt hDpos)]
      nlinarith [hltR]
    exact le_of_mul_le_mul_right h hDpos
  · have h : ((toRad s : ℝ) - (s : ℝ) * TWO_PI_Q32 / (2 : ℝ) ^ 64) * (2 : ℝ) ^ 64
        ≤ (1 / 2 : ℝ) * (2 : ℝ) ^ 64 := by
      rw [sub_mul, div_mul_cancel₀ _ (ne_of_gt hDpos)]
      nlinarith [hleR]
    exact le_of_mul_le_mul_right h hDpos

/-- `TWO_PI_Q32 = round(2π · 2^32)` is within half a unit of the exact value. -/
theorem two_pi_q32_defect : |(TWO_PI_Q32 : ℝ) - 2 * Real.pi * (2 : ℝ) ^ 32| ≤ 1 / 2 := by
  have h1 : (3.14159265358979323846 : ℝ) < Real.pi := Real.pi_gt_d20
  have h2 : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  rw [abs_le]
  constructor <;> push_cast [TWO_PI_Q32] <;> nlinarith [h1, h2]

/-- Sharp window bound: for `s` in the signed `i64` range, `toRad s` lies within
`TWO_PI_Q32 / 2 + 1` of zero. -/
theorem turn_to_radians_window_sharp (s : ℤ)
    (h1 : -(2 ^ 63 : ℤ) ≤ s) (h2 : s < (2 ^ 63 : ℤ)) :
    |toRad s| ≤ TWO_PI_Q32 / 2 + 1 := by
  unfold toRad
  rw [abs_le]
  have hD : (0 : ℤ) < 2 ^ 64 := by norm_num
  have hTpos : (0 : ℤ) < TWO_PI_Q32 := by norm_num [TWO_PI_Q32]
  constructor
  · rw [Int.le_ediv_iff_mul_le hD]
    have hmul : -(2 ^ 63) * TWO_PI_Q32 ≤ s * TWO_PI_Q32 :=
      Int.mul_le_mul_of_nonneg_right h1 (le_of_lt hTpos)
    have hT : TWO_PI_Q32 = 26986075409 := rfl
    rw [hT] at hmul ⊢
    omega
  · apply le_of_lt
    rw [Int.ediv_lt_iff_lt_mul hD]
    have hs : s ≤ 2 ^ 63 - 1 := by omega
    have hmul : s * TWO_PI_Q32 ≤ (2 ^ 63 - 1) * TWO_PI_Q32 :=
      Int.mul_le_mul_of_nonneg_right hs (by norm_num [TWO_PI_Q32])
    have hT : TWO_PI_Q32 = 26986075409 := rfl
    rw [hT] at hmul ⊢
    omega

/-- The Q32.32 window of `fixed_sin_cos` (`src/numeric/fixed_trig.rs:149-150`): for `s` in
the signed `i64` range, `toRad s` stays inside `[-4096 · 2^32, 4096 · 2^32]`. -/
theorem turn_to_radians_window (s : ℤ)
    (h1 : -(2 ^ 63 : ℤ) ≤ s) (h2 : s < (2 ^ 63 : ℤ)) :
    |toRad s| ≤ 4096 * (2 : ℤ) ^ 32 := by
  have h := turn_to_radians_window_sharp s h1 h2
  have hw : TWO_PI_Q32 / 2 + 1 ≤ 4096 * (2 : ℤ) ^ 32 := by norm_num [TWO_PI_Q32]
  linarith

/-- The real angle error of `toRad` against the exact turn value, for `s` in the signed
`i64` range: at most `2^-32` radian. -/
theorem turn_angle_error (s : ℤ) (h1 : -(2 ^ 63 : ℤ) ≤ s) (h2 : s < (2 ^ 63 : ℤ)) :
    |(toRad s : ℝ) / (2 : ℝ) ^ 32 - 2 * Real.pi * (s : ℝ) / (2 : ℝ) ^ 64|
      ≤ (2 : ℝ) ^ (-32 : ℤ) := by
  have hround := toRad_round s
  have hdef := two_pi_q32_defect
  have hsabs : |(s : ℝ)| ≤ (2 : ℝ) ^ 63 := by
    rw [abs_le]
    exact ⟨by exact_mod_cast h1, le_of_lt (by exact_mod_cast h2)⟩
  have hkey : (toRad s : ℝ) / (2 : ℝ) ^ 32 - 2 * Real.pi * (s : ℝ) / (2 : ℝ) ^ 64
      = ((toRad s : ℝ) - (s : ℝ) * TWO_PI_Q32 / (2 : ℝ) ^ 64) / (2 : ℝ) ^ 32
        + (s : ℝ) * ((TWO_PI_Q32 : ℝ) - 2 * Real.pi * (2 : ℝ) ^ 32) / (2 : ℝ) ^ 96 := by
    field_simp
    ring
  rw [hkey]
  calc |((toRad s : ℝ) - (s : ℝ) * TWO_PI_Q32 / (2 : ℝ) ^ 64) / (2 : ℝ) ^ 32
        + (s : ℝ) * ((TWO_PI_Q32 : ℝ) - 2 * Real.pi * (2 : ℝ) ^ 32) / (2 : ℝ) ^ 96|
      ≤ |((toRad s : ℝ) - (s : ℝ) * TWO_PI_Q32 / (2 : ℝ) ^ 64) / (2 : ℝ) ^ 32|
        + |(s : ℝ) * ((TWO_PI_Q32 : ℝ) - 2 * Real.pi * (2 : ℝ) ^ 32) / (2 : ℝ) ^ 96| :=
        abs_add_le _ _
    _ ≤ (1 / 2) / (2 : ℝ) ^ 32 + (2 : ℝ) ^ 63 * (1 / 2) / (2 : ℝ) ^ 96 := by
        apply add_le_add
        · rw [abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ 32)]
          exact div_le_div_of_nonneg_right hround (by positivity)
        · rw [abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ 96), abs_mul]
          apply div_le_div_of_nonneg_right _ (by positivity)
          exact mul_le_mul hsabs hdef (abs_nonneg _) (by positivity)
    _ ≤ (2 : ℝ) ^ (-32 : ℤ) := by norm_num

/-- The lifted phase denotes the same point of the circle as `f · p` turns: the two real
angles differ by an integer multiple of `2π`. -/
theorem turn_angle_congr (f p : ℕ) :
    ∃ k : ℤ, 2 * Real.pi * (lift (phase f p) : ℝ) / (2 : ℝ) ^ 64
      = 2 * Real.pi * ((f * p : ℕ) : ℝ) / (2 : ℝ) ^ 64 - 2 * Real.pi * k := by
  have hmod : lift (phase f p) ≡ ((f * p : ℕ) : ℤ) [ZMOD (2 ^ 64 : ℤ)] := by
    apply Int.ModEq.trans (lift_congr (phase f p))
    unfold phase
    rw [Int.natCast_mod]
    exact Int.mod_modEq _ _
  obtain ⟨m, hm⟩ := Int.modEq_iff_dvd.mp hmod
  have hlift : (lift (phase f p) : ℝ) = ((f * p : ℕ) : ℝ) - (2 : ℝ) ^ 64 * (m : ℝ) := by
    have h64 : (2 : ℝ) ^ 64 = 18446744073709551616 := by norm_num
    have := congrArg (fun z : ℤ => (z : ℝ)) hm
    push_cast at this ⊢
    rw [h64]
    linarith
  refine ⟨m, ?_⟩
  rw [hlift]
  field_simp

/-- **The headline**: if the frequency `f` denotes a position `f / 2^64` within `δ` of
`θ / (2π)` turns, then the executed angle `toRad (lift (phase f p)) / 2^32` is within
`2π · p · δ + 2^-32` of the ideal angle `p · θ` on the circle, for every position `p`
and with no window restriction. -/
theorem turn_total_error (f p : ℕ) (θ δ : ℝ)
    (hδ : |(f : ℝ) / (2 : ℝ) ^ 64 - θ / (2 * Real.pi)| ≤ δ) :
    ∃ k : ℤ,
      |(toRad (lift (phase f p)) : ℝ) / (2 : ℝ) ^ 32 - ((p : ℝ) * θ - 2 * Real.pi * k)|
        ≤ 2 * Real.pi * (p : ℝ) * δ + (2 : ℝ) ^ (-32 : ℤ) := by
  obtain ⟨k, hk⟩ := turn_angle_congr f p
  obtain ⟨hr1, hr2⟩ := lift_range (phase f p) (phase_lt f p)
  refine ⟨k, ?_⟩
  have hround := turn_angle_error (lift (phase f p)) hr1 hr2
  have hp : (0 : ℝ) ≤ (p : ℝ) := by positivity
  have hfp : ((f * p : ℕ) : ℝ) = (f : ℝ) * (p : ℝ) := by push_cast; ring
  have hsecond : |2 * Real.pi * ((f * p : ℕ) : ℝ) / (2 : ℝ) ^ 64 - (p : ℝ) * θ|
      ≤ 2 * Real.pi * (p : ℝ) * δ := by
    have hinner : |2 * Real.pi * ((f : ℝ) / (2 : ℝ) ^ 64) - θ| ≤ 2 * Real.pi * δ := by
      have he : 2 * Real.pi * ((f : ℝ) / (2 : ℝ) ^ 64) - θ
          = 2 * Real.pi * ((f : ℝ) / (2 : ℝ) ^ 64 - θ / (2 * Real.pi)) := by
        field_simp
      rw [he, abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 2 * Real.pi)]
      exact mul_le_mul_of_nonneg_left hδ (by positivity)
    have he2 : 2 * Real.pi * ((f * p : ℕ) : ℝ) / (2 : ℝ) ^ 64 - (p : ℝ) * θ
        = (p : ℝ) * (2 * Real.pi * ((f : ℝ) / (2 : ℝ) ^ 64) - θ) := by
      rw [hfp]; ring
    rw [he2, abs_mul, abs_of_nonneg hp]
    calc (p : ℝ) * |2 * Real.pi * ((f : ℝ) / (2 : ℝ) ^ 64) - θ|
        ≤ (p : ℝ) * (2 * Real.pi * δ) := mul_le_mul_of_nonneg_left hinner hp
      _ = 2 * Real.pi * (p : ℝ) * δ := by ring
  have hsplit : (toRad (lift (phase f p)) : ℝ) / (2 : ℝ) ^ 32
        - ((p : ℝ) * θ - 2 * Real.pi * k)
      = ((toRad (lift (phase f p)) : ℝ) / (2 : ℝ) ^ 32
          - 2 * Real.pi * (lift (phase f p) : ℝ) / (2 : ℝ) ^ 64)
        + (2 * Real.pi * ((f * p : ℕ) : ℝ) / (2 : ℝ) ^ 64 - (p : ℝ) * θ) := by
    rw [hk]
    ring
  rw [hsplit]
  calc |((toRad (lift (phase f p)) : ℝ) / (2 : ℝ) ^ 32
        - 2 * Real.pi * (lift (phase f p) : ℝ) / (2 : ℝ) ^ 64)
        + (2 * Real.pi * ((f * p : ℕ) : ℝ) / (2 : ℝ) ^ 64 - (p : ℝ) * θ)|
      ≤ |(toRad (lift (phase f p)) : ℝ) / (2 : ℝ) ^ 32
          - 2 * Real.pi * (lift (phase f p) : ℝ) / (2 : ℝ) ^ 64|
        + |2 * Real.pi * ((f * p : ℕ) : ℝ) / (2 : ℝ) ^ 64 - (p : ℝ) * θ| := abs_add_le _ _
    _ ≤ (2 : ℝ) ^ (-32 : ℤ) + 2 * Real.pi * (p : ℝ) * δ := add_le_add hround hsecond
    _ = 2 * Real.pi * (p : ℝ) * δ + (2 : ℝ) ^ (-32 : ℤ) := by ring

/- Small closed-literal checks of the definitions, including `toRad` of a quarter turn
against `round(π/2 · 2^32) = 6746518852` (`src/numeric/fixed_trig.rs`, `HALF_PI_Q32`). -/
namespace Witness

#guard phase 3 4 = 12
#guard phase 3 (4 + 5) = (phase 3 4 + phase 3 5) % 2 ^ 64
#guard phase 2 (3 * 7) = (3 * phase 2 7) % 2 ^ 64
#guard lift (2 ^ 63) = -(2 ^ 63)
#guard lift (2 ^ 63 - 1) = 2 ^ 63 - 1
#guard toRad (-(2 ^ 63)) = -13493037704
#guard toRad (2 ^ 63) = 13493037705
#guard toRad (2 ^ 62) = 6746518852
#guard (toRad (2 ^ 62) - 6746518852).natAbs ≤ 1

end Witness

end Numeric.RopeTurns
