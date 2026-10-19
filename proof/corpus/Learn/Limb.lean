import Mathlib
import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The block-floating limb-product error bound

`src/engine/kern/limb.rs` `gemm_q32` splits each 32-word block of each operand into `limbs` balanced
signed-byte digits and carries every kept limb pair on `gemm_q8_0`. Its `# Error bound` section
documents

```text
G = ⌈P/2⌉ + 1 + B·(1 + ⌈C_L·2^{Ea+Eb−32}⌉),   C_L = (33 + 16(L−1))·2^{−7L},
```

with `P = L(L+1)/2` kept pairs and `B = ⌈n/32⌉` blocks; `bound_ulps` (`:1234`) returns that integer.
This module proves the analytic bound `G` bounds, at the model of one block: the balanced digits
`|d_s| ≤ 64` with residual `|ρ| ≤ 2^{e−7L}/2` that `split_digits` (`:448`) emits, the exact integer
accumulation of the live kept pairs, and the single `round_shift_32`
(`src/engine/num/value.rs:10`) narrowing. It specializes at `L = 5`, `B = 1` and the measured `Down`
peaks to the documented `1562` and `24842` word bounds.

## The model, and what it is not

One block: 32 lanes at block exponents `ea` (weight side) and `eb` (activation side), `L` limbs,
digits `α i s`, `β i s` with `|α|, |β| ≤ 64`, residuals `ρa i`, `ρb i` with `|ρ| ≤ step^L/2` where
`step = 2^{−7}`, and the operand words `x i = 2^{ea}·(Σ_{s<L} α i s·step^{s+1} + ρa i)` (and `y`
likewise). The integer accumulator is

`T = 2^{ea+eb}·Σ_i Σ_{s+t<L, live(s,t)} α i s·β i t·step^{s+t+2}`,

with `live(s,t) := 7(s+t+2) + 8 ≤ ea + eb`, i.e. `T_pair = ea+eb−7(s+t+2) ≥ 8`: the `MIN_PAIR_EXP`
below which `Layout::placement` drops the pair. `round_shift_32` is `rshift32 v = (v + 2^31) / 2^32`
on `ℤ` (Lean floor division is the arithmetic shift).

Two inputs are hypotheses here, not theorems, and are the residual gap this module names:

* `hα`/`hβ`/`hρa`/`hρb` — the digit and residual bounds of `split_digits`, whose closed form
  `d_s = ⌊(u_s·2^7 + 2^{e−1})/2^e⌋` on the running residual `u_{s+1} = u_s − d_s·2^{e−7(s+1)}`
  gives `|d_s| ≤ 64` and `|ρ| ≤ 2^{e−7L}/2`. That division induction is **not** in this module.
* `hT` — that the `i64`/`i128` accumulator is exactly `T`. The AVX-VNNI core is `tested` against
  `exact_dot`; it is not extracted.
* `εf` — the `f_blk` word error, `dots`' input error. `fblk_error_le` proves its one-step part,
  `εres + 3114/2^32` (one `s·H` call scaled by `|a| ≤ 1` plus the residual); the sweep recurrence's
  state-error invariant for `f_res` is not in this module.

Not claimed: a refinement of the AVX-VNNI body, the `i64` non-overflow side condition on `T`, or
that `bound_ulps`'s per-block `+1` covers more than the `L ≤ 8` dropped mass bounded below.

The two specialized constants bound **one kernel call** at the brief's operand pairs, and the
shipped values are not those calls. `q_all` is `block_quads_q` (`icm.rs:785`), two calls:
`nt_q(codebook, H_rotᵀ)` at `(1.21e10, 6.05e10)`, i.e. `Ea+Eb = 72` and `bound_ulps = 3114`, then
`nt_q(t, codebook)` with `|t| ≤ 8·1.21e10·6.05e10/2^32` (`Ea+Eb ≤ 77`, `bound_ulps = 99338`), plus
the first call's error carried by the second (`≤ 8·3114·1.21e10/2^32`). `dots` is one call at
`(5.30e11, 1.21e10)`, but on an `f_blk` that already carries the `s·H` call's error. So `1562` and
`24842` are the per-call rounding and truncation bounds at the named peaks, not end-to-end bounds of
the shipped `q` and `dot`. The end-to-end constants are `q_all_end2end` and `dots_end2end` below:
`εq = 99338/2^32 + 8·(3114/2^32)·(12100000000/2^32) = 728088661250048/2^64` and
`εd = 24842/2^32 + 8·εf·(12100000000/2^32)`, `εf` the `f_blk` word error. The `s_blk ≤ 2^32` peak is
also the pre-sweep state: after one block is coded `s_blk` holds codebook rows (`1.21e10`,
`Ea = 35`).
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable section

namespace Learn.Limb

/-! ## The limb step and its geometric series -/

/-- One limb step, `2^{-7}`: `LIMB_BITS` in `src/engine/kern/limb.rs`. -/
def step : ℝ := 1 / 128

lemma step_pos : (0 : ℝ) < step := by unfold step; norm_num

lemma step_le_one : step ≤ 1 := by unfold step; norm_num

lemma step_pow_pos (k : ℕ) : (0 : ℝ) < step ^ k := pow_pos step_pos k

lemma step_pow_le_one (k : ℕ) : step ^ k ≤ 1 :=
  pow_le_one₀ step_pos.le step_le_one

/-- `Σ_{i<m} (1/2)^i ≤ 2`, by the invariant `Σ ≤ 2 - 2·(1/2)^m`. -/
lemma geom_half_aux (m : ℕ) :
    (∑ i ∈ Finset.range m, ((1 : ℝ) / 2) ^ i) ≤ 2 - 2 * ((1 : ℝ) / 2) ^ m := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [Finset.sum_range_succ, pow_succ]
    have hp : (0 : ℝ) < ((1 : ℝ) / 2) ^ m := by positivity
    nlinarith

lemma geom_half_le (m : ℕ) : (∑ i ∈ Finset.range m, ((1 : ℝ) / 2) ^ i) ≤ 2 := by
  have h := geom_half_aux m
  have hp : (0 : ℝ) ≤ ((1 : ℝ) / 2) ^ m := by positivity
  linarith

lemma geom_step_le (m : ℕ) : (∑ i ∈ Finset.range m, step ^ i) ≤ 2 := by
  refine le_trans (Finset.sum_le_sum (fun i _ => ?_)) (geom_half_le m)
  exact pow_le_pow_left₀ step_pos.le (by unfold step; norm_num) i

/-- `2^{7k}·step^k = 1`: the exact power-of-two cancellation behind the dropped-pair mass. -/
lemma two_pow_mul_step (k : ℕ) : (2 : ℝ) ^ (7 * k) * step ^ k = 1 := by
  induction k with
  | zero => norm_num
  | succ k ih =>
    have h7 : (2 : ℝ) ^ (7 * (k + 1)) = (2 : ℝ) ^ (7 * k) * 2 ^ 7 := by
      rw [show 7 * (k + 1) = 7 * k + 7 by ring, pow_add]
    calc (2 : ℝ) ^ (7 * (k + 1)) * step ^ (k + 1)
        = ((2 : ℝ) ^ (7 * k) * step ^ k) * ((2 : ℝ) ^ 7 * step) := by
          rw [h7, pow_succ]; ring
      _ = 1 := by rw [ih]; norm_num [step]

/-- `step^L ≤ 1/8` for `L ≥ 1`. -/
lemma step_pow_le_eighth (L : ℕ) (hL : 1 ≤ L) : step ^ L ≤ 1 / 8 := by
  obtain ⟨k, rfl⟩ : ∃ k, L = k + 1 := ⟨L - 1, by omega⟩
  rw [pow_succ]
  have hk : step ^ k ≤ 1 := step_pow_le_one k
  have h1 : step ^ k * step ≤ 1 * step := mul_le_mul_of_nonneg_right hk step_pos.le
  have h2 : step ≤ 1 / 8 := by unfold step; norm_num
  nlinarith

/-! ## The block: digits, pair masses, and the two accumulators -/

/-- `Σ_{s<L} d_s·2^{−7(s+1)}`, the kept digits at the block scale. -/
def digitSum (d : ℕ → ℤ) (L : ℕ) : ℝ :=
  ∑ s ∈ Finset.range L, (d s : ℝ) * step ^ (s + 1)

/-- `Σ_{s,t<L} α_s·β_t·2^{−7(s+t+2)}`, the full digit product. -/
def allMass (α β : ℕ → ℤ) (L : ℕ) : ℝ :=
  ∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
    (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2)

/-- The kept pairs `s + t < L`. -/
def keptMass (α β : ℕ → ℤ) (L : ℕ) : ℝ :=
  ∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
    if s + t + 1 ≤ L then (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) else 0

/-- The pair exponent `T = ea + eb − 7(s+t+2)` is at least `MIN_PAIR_EXP = 8`. -/
abbrev livePair (ea eb s t : ℕ) : Prop := 7 * (s + t + 2) + 8 ≤ ea + eb

/-- The kept pairs that are placed (live). -/
def liveMass (ea eb : ℕ) (α β : ℕ → ℤ) (L : ℕ) : ℝ :=
  ∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
    if s + t + 1 ≤ L ∧ livePair ea eb s t then
      (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) else 0

/-- The Q32.32 operand word of the digit model. -/
def bword (L ea : ℕ) (d : ℕ → ℤ) (ρ : ℝ) : ℝ := (2 : ℝ) ^ ea * (digitSum d L + ρ)

/-- The exact pre-shift block dot `Σ_i x_i·y_i` (Q64.64, as a real). -/
def exactPre (L ea eb : ℕ) (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) : ℝ :=
  ∑ i ∈ Finset.range 32, bword L ea (α i) (ρa i) * bword L eb (β i) (ρb i)

/-- The limb kernel's pre-shift accumulator, in the same units as `exactPre`. -/
def limbPre (L ea eb : ℕ) (α β : ℕ → ℕ → ℤ) : ℝ :=
  (2 : ℝ) ^ (ea + eb) * ∑ i ∈ Finset.range 32, liveMass ea eb (α i) (β i) L

/-- `round_shift_32` (`src/engine/num/value.rs:10`): `(v + 2^31) >> 32` on `i128`. -/
def rshift32 (v : ℤ) : ℤ := (v + 2 ^ 31) / 2 ^ 32

lemma cast_abs_64 {z : ℤ} (h : |z| ≤ 64) : |((z : ℝ))| ≤ 64 := by
  calc |((z : ℤ) : ℝ)| = ((|z| : ℤ) : ℝ) := by rw [Int.cast_abs]
    _ ≤ ((64 : ℤ) : ℝ) := by exact_mod_cast h
    _ = 64 := by norm_num

/-- `|d_s| ≤ 64` digits sum to at most `1` at the block scale. -/
lemma abs_digitSum_le (d : ℕ → ℤ) (L : ℕ) (hd : ∀ s ∈ Finset.range L, |d s| ≤ 64) :
    |digitSum d L| ≤ 1 := by
  have h1 : |digitSum d L| ≤ ∑ s ∈ Finset.range L, 64 * step ^ (s + 1) := by
    rw [digitSum]
    refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum (fun s hs => ?_))
    rw [abs_mul, abs_of_nonneg (step_pow_pos _).le]
    have hc : |((d s : ℤ) : ℝ)| ≤ 64 := cast_abs_64 (hd s hs)
    nlinarith [step_pow_pos (s + 1)]
  have h3 : (∑ s ∈ Finset.range L, 64 * step ^ (s + 1))
      = 64 * step * ∑ s ∈ Finset.range L, step ^ s := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro s _
    rw [pow_succ]
    ring
  have hstep : 64 * step = 1 / 2 := by unfold step; norm_num
  have hg := geom_step_le L
  have h2 : (∑ s ∈ Finset.range L, 64 * step ^ (s + 1)) ≤ 1 := by
    rw [h3]; nlinarith
  linarith

/-- The product of the two digit sums is the full pair mass. -/
lemma digitSum_mul (α β : ℕ → ℤ) (L : ℕ) :
    digitSum α L * digitSum β L = allMass α β L := by
  rw [digitSum, digitSum, allMass, Finset.sum_mul_sum]
  apply Finset.sum_congr rfl
  intro s _
  apply Finset.sum_congr rfl
  intro t _
  calc (α s : ℝ) * step ^ (s + 1) * ((β t : ℝ) * step ^ (t + 1))
      = ((α s : ℝ) * (β t : ℝ)) * (step ^ (s + 1) * step ^ (t + 1)) := by ring
    _ = ((α s : ℝ) * (β t : ℝ)) * step ^ (s + t + 2) := by
        rw [← pow_add, show s + 1 + (t + 1) = s + t + 2 by omega]
    _ = (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) := by ring

/-- One `s`'s truncated tail is at most `2·step^{L+2}`. The qualifying `t` are `[L−s, L)`;
reindexing `t = (L−s) + j` gives `step^{L+2}·Σ_{j<s} step^j`. -/
lemma trunc_tail (L s : ℕ) (hs : s < L) :
    (∑ t ∈ Finset.range L, (if s + t + 1 ≤ L then (0 : ℝ) else step ^ (s + t + 2)))
      ≤ 2 * step ^ (L + 2) := by
  have hset : (Finset.range L).filter (fun t => ¬ (s + t + 1 ≤ L)) = Finset.Ico (L - s) L := by
    ext t
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
    omega
  have hif : (∑ t ∈ Finset.range L, (if s + t + 1 ≤ L then (0 : ℝ) else step ^ (s + t + 2)))
      = ∑ t ∈ (Finset.range L).filter (fun t => ¬ (s + t + 1 ≤ L)), step ^ (s + t + 2) := by
    rw [Finset.sum_filter]
    apply Finset.sum_congr rfl
    intro t _
    by_cases h : s + t + 1 ≤ L <;> simp [h]
  rw [hif, hset, Finset.sum_Ico_eq_sum_range, show L - (L - s) = s by omega]
  have hexp : ∀ k ∈ Finset.range s, step ^ (s + (L - s + k) + 2) = step ^ (L + 2) * step ^ k := by
    intro k _
    rw [← pow_add]
    congr 1
    omega
  rw [Finset.sum_congr rfl hexp, ← Finset.mul_sum]
  have hg := geom_step_le s
  have hp := step_pow_pos (L + 2)
  nlinarith

/-- A sum over `range L` whose `0` term vanishes is at most `L−1` times a uniform bound. This is
where `abs_truncMass_le`'s `L−1` comes from: the `s = 0` truncated tail is empty. -/
lemma sum_range_le_pred_mul {g : ℕ → ℝ} {C : ℝ} {L : ℕ} (hL : 1 ≤ L)
    (hg : ∀ s ∈ Finset.range L, g s ≤ C) (h0 : g 0 = 0) :
    ∑ s ∈ Finset.range L, g s ≤ ((L : ℝ) - 1) * C := by
  obtain ⟨L', rfl⟩ : ∃ L', L = L' + 1 := ⟨L - 1, by omega⟩
  have hL' : 1 ≤ L' + 1 := hL
  rw [Finset.sum_range_succ', h0, add_zero]
  calc ∑ s ∈ Finset.range L', g (s + 1) ≤ ∑ _s ∈ Finset.range L', C := by
        apply Finset.sum_le_sum
        intro s hs
        exact hg (s + 1) (Finset.mem_range.mpr (by
          have hs' := Finset.mem_range.mp hs
          omega))
    _ = (L' : ℝ) * C := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    _ = (((L' + 1 : ℕ) : ℝ) - 1) * C := by
        push_cast
        ring

/-- The truncated pair mass (`s + t ≥ L`) is `4096·(L−1)·2·step^{L+2}`. The `s = 0` tail is empty,
which is where the `L−1` comes from. -/
lemma abs_truncMass_le (α β : ℕ → ℤ) (L : ℕ) (hL : 1 ≤ L)
    (hα : ∀ s ∈ Finset.range L, |α s| ≤ 64)
    (hβ : ∀ s ∈ Finset.range L, |β s| ≤ 64) :
    |allMass α β L - keptMass α β L|
      ≤ 4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2)) := by
  have hEq : allMass α β L - keptMass α β L
      = ∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
          (if s + t + 1 ≤ L then (0 : ℝ) else (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2)) := by
    rw [allMass, keptMass, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro s _
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro t _
    by_cases h : s + t + 1 ≤ L <;> simp [h]
  rw [hEq]
  have hterm : ∀ s ∈ Finset.range L, ∀ t ∈ Finset.range L,
      |(if s + t + 1 ≤ L then (0 : ℝ) else (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2))|
        ≤ (if s + t + 1 ≤ L then (0 : ℝ) else 4096 * step ^ (s + t + 2)) := by
    intro s hs t ht
    by_cases h : s + t + 1 ≤ L
    · simp [h]
    · simp only [h, if_false]
      have ha : |(α s : ℝ)| ≤ 64 := cast_abs_64 (hα s hs)
      have hb : |(β t : ℝ)| ≤ 64 := cast_abs_64 (hβ t ht)
      rw [abs_mul, abs_mul, abs_of_nonneg (step_pow_pos _).le]
      have hprod : |(α s : ℝ)| * |(β t : ℝ)| ≤ 64 * 64 :=
        mul_le_mul ha hb (abs_nonneg _) (by norm_num)
      have hp := step_pow_pos (s + t + 2)
      nlinarith
  have hinner : ∀ s ∈ Finset.range L,
      (∑ t ∈ Finset.range L,
        (if s + t + 1 ≤ L then (0 : ℝ) else 4096 * step ^ (s + t + 2)))
      ≤ 4096 * (2 * step ^ (L + 2)) := by
    intro s hs
    have htail := trunc_tail L s (Finset.mem_range.mp hs)
    have hfac : (∑ t ∈ Finset.range L,
        (if s + t + 1 ≤ L then (0 : ℝ) else 4096 * step ^ (s + t + 2)))
        = 4096 * ∑ t ∈ Finset.range L,
            (if s + t + 1 ≤ L then (0 : ℝ) else step ^ (s + t + 2)) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro t _
      by_cases h : s + t + 1 ≤ L <;> simp [h]
    rw [hfac]
    nlinarith
  have h0 : (∑ t ∈ Finset.range L,
      (if (0 : ℕ) + t + 1 ≤ L then (0 : ℝ) else 4096 * step ^ ((0 : ℕ) + t + 2))) = 0 := by
    apply Finset.sum_eq_zero
    intro t ht
    have ht' : t < L := Finset.mem_range.mp ht
    simp [show t + 1 ≤ L by omega]
  calc |∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
        (if s + t + 1 ≤ L then (0 : ℝ) else (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2))|
      ≤ ∑ s ∈ Finset.range L, |∑ t ∈ Finset.range L,
          (if s + t + 1 ≤ L then (0 : ℝ) else (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
          (if s + t + 1 ≤ L then (0 : ℝ) else 4096 * step ^ (s + t + 2)) := by
        apply Finset.sum_le_sum
        intro s hs
        refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum (fun t ht => ?_))
        exact hterm s hs t ht
    _ ≤ ((L : ℝ) - 1) * (4096 * (2 * step ^ (L + 2))) :=
        sum_range_le_pred_mul hL hinner h0
    _ = 4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2)) := by ring

/-- One dropped pair's scaled magnitude: `|2^{ea+eb}·α·β·step^{s+t+2}| ≤ 2^19` once the pair is
placed, from `|α|,|β| ≤ 64` and `2^{ea+eb}·step^{s+t+2} ≤ 2^7`. -/
lemma abs_scaled_term_le {ea eb s t : ℕ} {α β : ℕ → ℤ}
    (ha : |α s| ≤ 64) (hb : |β t| ≤ 64)
    (hsc : (2 : ℝ) ^ (ea + eb) * step ^ (s + t + 2) ≤ 2 ^ 7) :
    |(2 : ℝ) ^ (ea + eb) * ((α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2))| ≤ 2 ^ 19 := by
  have ha' : |(α s : ℝ)| ≤ 64 := cast_abs_64 ha
  have hb' : |(β t : ℝ)| ≤ 64 := cast_abs_64 hb
  have hprod : |(α s : ℝ)| * |(β t : ℝ)| ≤ 64 * 64 :=
    mul_le_mul ha' hb' (abs_nonneg _) (by norm_num)
  have hp : (0 : ℝ) < (2 : ℝ) ^ (ea + eb) * step ^ (s + t + 2) :=
    mul_pos (pow_pos (by norm_num) _) (step_pow_pos _)
  have hrew : (2 : ℝ) ^ (ea + eb) * ((α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2))
      = ((α s : ℝ) * (β t : ℝ)) * ((2 : ℝ) ^ (ea + eb) * step ^ (s + t + 2)) := by ring
  rw [hrew, abs_mul, abs_of_pos hp]
  have h1 : |(α s : ℝ) * (β t : ℝ)| ≤ 64 * 64 := by rw [abs_mul]; exact hprod
  calc |(α s : ℝ) * (β t : ℝ)| * ((2 : ℝ) ^ (ea + eb) * step ^ (s + t + 2))
      ≤ (64 * 64) * 2 ^ 7 := mul_le_mul h1 hsc (le_of_lt hp) (by norm_num)
    _ = 2 ^ 19 := by norm_num

/-- The dropped (`T < 8`) kept pairs, scaled by `2^{ea+eb}`: at most `L²·2^19`. Each dropped pair is
`2^{ea+eb}·step^{s+t+2} ≤ 2^7` by `two_pow_mul_step`, so `4096·2^7 = 2^19` per pair, and the grid
has `L²` slots. -/
lemma scaled_dropped_le (ea eb : ℕ) (α β : ℕ → ℤ) (L : ℕ)
    (hα : ∀ s ∈ Finset.range L, |α s| ≤ 64)
    (hβ : ∀ s ∈ Finset.range L, |β s| ≤ 64) :
    (2 : ℝ) ^ (ea + eb) * |keptMass α β L - liveMass ea eb α β L| ≤ (L : ℝ) ^ 2 * 2 ^ 19 := by
  have hc : (0 : ℝ) ≤ (2 : ℝ) ^ (ea + eb) := by positivity
  have hpair : ∀ s t, ¬ livePair ea eb s t →
      (2 : ℝ) ^ (ea + eb) * step ^ (s + t + 2) ≤ 2 ^ 7 := by
    intro s t h
    have hge : ¬ (7 * (s + t + 2) + 8 ≤ ea + eb) := h
    have hle : ea + eb ≤ 7 * (s + t + 2) + 7 := by omega
    have h1 : (2 : ℝ) ^ (ea + eb) ≤ (2 : ℝ) ^ (7 * (s + t + 2) + 7) :=
      pow_le_pow_right₀ (by norm_num) hle
    have h2 : (2 : ℝ) ^ (7 * (s + t + 2) + 7) * step ^ (s + t + 2) = 2 ^ 7 := by
      rw [pow_add]
      calc (2 : ℝ) ^ (7 * (s + t + 2)) * 2 ^ 7 * step ^ (s + t + 2)
          = ((2 : ℝ) ^ (7 * (s + t + 2)) * step ^ (s + t + 2)) * 2 ^ 7 := by ring
        _ = 2 ^ 7 := by rw [two_pow_mul_step (s + t + 2)]; ring
    calc (2 : ℝ) ^ (ea + eb) * step ^ (s + t + 2)
        ≤ (2 : ℝ) ^ (7 * (s + t + 2) + 7) * step ^ (s + t + 2) :=
          mul_le_mul_of_nonneg_right h1 (step_pow_pos _).le
      _ = 2 ^ 7 := h2
  have hEq : keptMass α β L - liveMass ea eb α β L
      = ∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
          (if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ)
            else if s + t + 1 ≤ L then (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) else 0) := by
    rw [keptMass, liveMass, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro s _
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro t _
    by_cases hp : s + t + 1 ≤ L ∧ livePair ea eb s t
    · rw [if_pos hp, if_pos hp, if_pos hp.1]
      ring
    · rw [if_neg hp, if_neg hp]
      ring
  have hEqc : (2 : ℝ) ^ (ea + eb) * (keptMass α β L - liveMass ea eb α β L)
      = ∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
          (if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ)
            else (2 : ℝ) ^ (ea + eb) *
              (if s + t + 1 ≤ L then (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) else 0)) := by
    rw [hEq, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro s _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro t _
    by_cases hp : s + t + 1 ≤ L ∧ livePair ea eb s t
    · rw [if_pos hp, if_pos hp]
      ring
    · rw [if_neg hp, if_neg hp]
  have habs : (2 : ℝ) ^ (ea + eb) * |keptMass α β L - liveMass ea eb α β L|
      = |(2 : ℝ) ^ (ea + eb) * (keptMass α β L - liveMass ea eb α β L)| := by
    rw [abs_mul, abs_of_nonneg hc]
  rw [habs, hEqc]
  have hterm : ∀ s ∈ Finset.range L, ∀ t ∈ Finset.range L,
      |(if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ)
          else (2 : ℝ) ^ (ea + eb) *
            (if s + t + 1 ≤ L then (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) else 0))|
        ≤ (if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ) else 2 ^ 19) := by
    intro s hs t ht
    by_cases hp : s + t + 1 ≤ L ∧ livePair ea eb s t
    · rw [if_pos hp, if_pos hp]
      simp
    · rw [if_neg hp, if_neg hp]
      by_cases hq : s + t + 1 ≤ L
      · rw [if_pos hq]
        have hnl : ¬ livePair ea eb s t := fun hl => hp ⟨hq, hl⟩
        exact abs_scaled_term_le (hα s hs) (hβ t ht) (hpair s t hnl)
      · rw [if_neg hq]
        simp
  have hsum : ∀ s ∈ Finset.range L,
      |∑ t ∈ Finset.range L,
        (if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ)
          else (2 : ℝ) ^ (ea + eb) *
            (if s + t + 1 ≤ L then (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) else 0))|
      ≤ ∑ t ∈ Finset.range L,
          (if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ) else 2 ^ 19) :=
    fun s hs => le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum (fun t ht => hterm s hs t ht))
  calc |∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
        (if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ)
          else (2 : ℝ) ^ (ea + eb) *
            (if s + t + 1 ≤ L then (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) else 0))|
      ≤ ∑ s ∈ Finset.range L, |∑ t ∈ Finset.range L,
          (if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ)
            else (2 : ℝ) ^ (ea + eb) *
              (if s + t + 1 ≤ L then (α s : ℝ) * (β t : ℝ) * step ^ (s + t + 2) else 0))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ s ∈ Finset.range L, ∑ t ∈ Finset.range L,
          (if s + t + 1 ≤ L ∧ livePair ea eb s t then (0 : ℝ) else 2 ^ 19) :=
        Finset.sum_le_sum hsum
    _ ≤ ∑ _s ∈ Finset.range L, ∑ _t ∈ Finset.range L, (2 : ℝ) ^ 19 := by
        apply Finset.sum_le_sum
        intro s _
        apply Finset.sum_le_sum
        intro t _
        by_cases hp : s + t + 1 ≤ L ∧ livePair ea eb s t
        · rw [if_pos hp]
          norm_num
        · rw [if_neg hp]

    _ = (L : ℝ) ^ 2 * 2 ^ 19 := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, Finset.sum_const,
          Finset.card_range, nsmul_eq_mul]
        ring

/-- `32·4096·2(L−1)·step^{L+2} = 16(L−1)·step^L`, and the residual `32(step^L + step^{2L}/4)` fits
in `33·step^L` because `step^L ≤ 1/8`. -/
lemma coeff_le (L : ℕ) (hL : 1 ≤ L) (c : ℝ) (hc : 0 ≤ c) :
    32 * (c * (4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2))))
      + 32 * (c * (step ^ L + step ^ (2 * L) / 4))
      + 32 * ((L : ℝ) ^ 2 * 2 ^ 19)
    ≤ c * ((16 * (L : ℝ) + 17) * step ^ L) + (L : ℝ) ^ 2 * 2 ^ 24 := by
  have hS : (0 : ℝ) < step ^ L := step_pow_pos L
  have hS8 : step ^ L ≤ 1 / 8 := step_pow_le_eighth L hL
  have h2 : step ^ (2 * L) = step ^ L * step ^ L := by
    rw [← pow_add, show L + L = 2 * L by ring]
  have h1 : step ^ (L + 2) = step ^ L * step ^ 2 := by rw [pow_add]
  have hs2 : step ^ 2 = 1 / 16384 := by unfold step; norm_num
  have h8 : 8 * step ^ (2 * L) ≤ step ^ L := by rw [h2]; nlinarith
  have hA : 32 * (c * (4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2))))
      = 16 * ((L : ℝ) - 1) * c * step ^ L := by
    rw [h1, hs2]
    ring
  rw [hA]
  have hfac : 0 ≤ c * step ^ L * (1 - 8 * step ^ L) :=
    mul_nonneg (mul_nonneg hc hS.le) (by linarith)
  nlinarith [hfac, mul_nonneg hc (mul_nonneg hS.le hS.le)]

/-- The per-lane error: the placed-pair mass against the exact digit product, scaled. -/
lemma lane_le (L ea eb : ℕ) (hL : 1 ≤ L)
    (α β : ℕ → ℤ) (ρa ρb : ℝ)
    (hα : ∀ s ∈ Finset.range L, |α s| ≤ 64)
    (hβ : ∀ s ∈ Finset.range L, |β s| ≤ 64)
    (hρa : |ρa| ≤ step ^ L / 2) (hρb : |ρb| ≤ step ^ L / 2) :
    |(2 : ℝ) ^ (ea + eb) * liveMass ea eb α β L - bword L ea α ρa * bword L eb β ρb|
      ≤ (2 : ℝ) ^ (ea + eb) * (4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2)))
        + (2 : ℝ) ^ (ea + eb) * (step ^ L + step ^ (2 * L) / 4)
        + (L : ℝ) ^ 2 * 2 ^ 19 := by
  have hA : |digitSum α L| ≤ 1 := abs_digitSum_le α L hα
  have hB : |digitSum β L| ≤ 1 := abs_digitSum_le β L hβ
  have hc : (0 : ℝ) ≤ (2 : ℝ) ^ (ea + eb) := by positivity
  have htrunc := abs_truncMass_le α β L hL hα hβ
  have hdrop := scaled_dropped_le ea eb α β L hα hβ
  have hres : |digitSum α L * ρb + digitSum β L * ρa + ρa * ρb|
      ≤ step ^ L + step ^ (2 * L) / 4 := by
    have hp : (0 : ℝ) ≤ step ^ L := step_pow_pos L |>.le
    have h1 : |digitSum α L * ρb| ≤ step ^ L / 2 := by
      rw [abs_mul]
      calc |digitSum α L| * |ρb| ≤ 1 * (step ^ L / 2) :=
            mul_le_mul hA hρb (abs_nonneg _) (by norm_num)
        _ = step ^ L / 2 := by ring
    have h2 : |digitSum β L * ρa| ≤ step ^ L / 2 := by
      rw [abs_mul]
      calc |digitSum β L| * |ρa| ≤ 1 * (step ^ L / 2) :=
            mul_le_mul hB hρa (abs_nonneg _) (by norm_num)
        _ = step ^ L / 2 := by ring
    have h3 : |ρa * ρb| ≤ step ^ (2 * L) / 4 := by
      rw [abs_mul]
      have hp2 : (0 : ℝ) ≤ step ^ L / 2 := by linarith
      have hsq : step ^ L * step ^ L = step ^ (2 * L) := by
        rw [← pow_add, show L + L = 2 * L by ring]
      calc |ρa| * |ρb| ≤ (step ^ L / 2) * (step ^ L / 2) :=
            mul_le_mul hρa hρb (abs_nonneg _) hp2
        _ = step ^ L * step ^ L / 4 := by ring
        _ = step ^ (2 * L) / 4 := by rw [hsq]
    calc |digitSum α L * ρb + digitSum β L * ρa + ρa * ρb|
        ≤ |digitSum α L * ρb + digitSum β L * ρa| + |ρa * ρb| := abs_add_le _ _
      _ ≤ (|digitSum α L * ρb| + |digitSum β L * ρa|) + |ρa * ρb| := by
          linarith [abs_add_le (digitSum α L * ρb) (digitSum β L * ρa)]
      _ ≤ step ^ L + step ^ (2 * L) / 4 := by linarith
  have hbw : bword L ea α ρa * bword L eb β ρb
      = (2 : ℝ) ^ (ea + eb) * ((digitSum α L + ρa) * (digitSum β L + ρb)) := by
    rw [bword, bword]
    calc (2 : ℝ) ^ ea * (digitSum α L + ρa) * ((2 : ℝ) ^ eb * (digitSum β L + ρb))
        = ((2 : ℝ) ^ ea * (2 : ℝ) ^ eb) * ((digitSum α L + ρa) * (digitSum β L + ρb)) := by
          ring
      _ = (2 : ℝ) ^ (ea + eb) * ((digitSum α L + ρa) * (digitSum β L + ρb)) := by
          rw [← pow_add]
  have hdec : (digitSum α L + ρa) * (digitSum β L + ρb)
      = liveMass ea eb α β L + ((allMass α β L - keptMass α β L)
        + (keptMass α β L - liveMass ea eb α β L)
        + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb)) := by
    calc (digitSum α L + ρa) * (digitSum β L + ρb)
        = digitSum α L * digitSum β L
          + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb) := by ring
      _ = allMass α β L + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb) := by
          rw [digitSum_mul]
      _ = liveMass ea eb α β L + ((allMass α β L - keptMass α β L)
          + (keptMass α β L - liveMass ea eb α β L)
          + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb)) := by ring
  rw [hbw, hdec]
  have hstep1 : (2 : ℝ) ^ (ea + eb) * liveMass ea eb α β L
        - (2 : ℝ) ^ (ea + eb) * (liveMass ea eb α β L
          + ((allMass α β L - keptMass α β L)
            + (keptMass α β L - liveMass ea eb α β L)
            + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb)))
      = (2 : ℝ) ^ (ea + eb) * (liveMass ea eb α β L
          - (liveMass ea eb α β L
            + ((allMass α β L - keptMass α β L) + (keptMass α β L - liveMass ea eb α β L)
              + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb)))) := by ring
  rw [hstep1, abs_mul, abs_of_nonneg hc]
  have hneg : liveMass ea eb α β L
        - (liveMass ea eb α β L
          + ((allMass α β L - keptMass α β L) + (keptMass α β L - liveMass ea eb α β L)
            + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb)))
      = -(((allMass α β L - keptMass α β L) + (keptMass α β L - liveMass ea eb α β L))
          + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb)) := by ring
  rw [hneg, abs_neg]
  have htri : |((allMass α β L - keptMass α β L) + (keptMass α β L - liveMass ea eb α β L))
        + (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb)|
      ≤ |allMass α β L - keptMass α β L| + |keptMass α β L - liveMass ea eb α β L|
        + |digitSum α L * ρb + digitSum β L * ρa + ρa * ρb| := by
    have h1 := abs_add_le ((allMass α β L - keptMass α β L) + (keptMass α β L - liveMass ea eb α β L))
      (digitSum α L * ρb + digitSum β L * ρa + ρa * ρb)
    have h2 := abs_add_le (allMass α β L - keptMass α β L) (keptMass α β L - liveMass ea eb α β L)
    linarith
  calc (2 : ℝ) ^ (ea + eb) * |_|
      ≤ (2 : ℝ) ^ (ea + eb) * (|allMass α β L - keptMass α β L|
          + |keptMass α β L - liveMass ea eb α β L|
          + |digitSum α L * ρb + digitSum β L * ρa + ρa * ρb|) := by
        exact mul_le_mul_of_nonneg_left htri hc
    _ = (2 : ℝ) ^ (ea + eb) * |allMass α β L - keptMass α β L|
          + (2 : ℝ) ^ (ea + eb) * |keptMass α β L - liveMass ea eb α β L|
          + (2 : ℝ) ^ (ea + eb) * |digitSum α L * ρb + digitSum β L * ρa + ρa * ρb| := by ring
    _ ≤ (2 : ℝ) ^ (ea + eb) * (4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2)))
          + (L : ℝ) ^ 2 * 2 ^ 19
          + (2 : ℝ) ^ (ea + eb) * (step ^ L + step ^ (2 * L) / 4) := by
        have hc0 : (0 : ℝ) ≤ (2 : ℝ) ^ (ea + eb) := hc
        have h1 := mul_le_mul_of_nonneg_left htrunc hc0
        have h3 := mul_le_mul_of_nonneg_left hres hc0
        linarith
    _ = (2 : ℝ) ^ (ea + eb) * (4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2)))
          + (2 : ℝ) ^ (ea + eb) * (step ^ L + step ^ (2 * L) / 4)
          + (L : ℝ) ^ 2 * 2 ^ 19 := by ring

/-- The one-block analytic bound: the exact pre-shift dot and the limb accumulator differ by at most
`2^{E}·C_L + L²·2^24` with `C_L = (33 + 16(L−1))·2^{−7L} = (16L+17)·step^L`. -/
theorem abs_limbPre_sub_exactPre_le (L ea eb : ℕ) (hL : 1 ≤ L)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range L, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range L, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ step ^ L / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ step ^ L / 2) :
    |limbPre L ea eb α β - exactPre L ea eb α β ρa ρb|
      ≤ (2 : ℝ) ^ (ea + eb) * ((16 * (L : ℝ) + 17) * step ^ L) + (L : ℝ) ^ 2 * 2 ^ 24 := by
  have hc : (0 : ℝ) ≤ (2 : ℝ) ^ (ea + eb) := by positivity
  have hEq : limbPre L ea eb α β - exactPre L ea eb α β ρa ρb
      = ∑ i ∈ Finset.range 32,
          ((2 : ℝ) ^ (ea + eb) * liveMass ea eb (α i) (β i) L
            - bword L ea (α i) (ρa i) * bword L eb (β i) (ρb i)) := by
    rw [limbPre, exactPre, Finset.mul_sum, ← Finset.sum_sub_distrib]
  rw [hEq]
  have hlane : ∀ i ∈ Finset.range 32,
      |(2 : ℝ) ^ (ea + eb) * liveMass ea eb (α i) (β i) L
        - bword L ea (α i) (ρa i) * bword L eb (β i) (ρb i)|
      ≤ (2 : ℝ) ^ (ea + eb) * (4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2)))
        + (2 : ℝ) ^ (ea + eb) * (step ^ L + step ^ (2 * L) / 4) + (L : ℝ) ^ 2 * 2 ^ 19 := by
    intro i hi
    exact lane_le L ea eb hL (α i) (β i) (ρa i) (ρb i) (hα i hi) (hβ i hi) (hρa i hi) (hρb i hi)
  calc |∑ i ∈ Finset.range 32, _|
      ≤ ∑ i ∈ Finset.range 32,
          |(2 : ℝ) ^ (ea + eb) * liveMass ea eb (α i) (β i) L
            - bword L ea (α i) (ρa i) * bword L eb (β i) (ρb i)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i ∈ Finset.range 32,
          ((2 : ℝ) ^ (ea + eb) * (4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2)))
            + (2 : ℝ) ^ (ea + eb) * (step ^ L + step ^ (2 * L) / 4) + (L : ℝ) ^ 2 * 2 ^ 19) :=
        Finset.sum_le_sum hlane
    _ = 32 * ((2 : ℝ) ^ (ea + eb) * (4096 * (2 * ((L : ℝ) - 1) * step ^ (L + 2)))
          + (2 : ℝ) ^ (ea + eb) * (step ^ L + step ^ (2 * L) / 4) + (L : ℝ) ^ 2 * 2 ^ 19) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        norm_num
    _ ≤ (2 : ℝ) ^ (ea + eb) * ((16 * (L : ℝ) + 17) * step ^ L) + (L : ℝ) ^ 2 * 2 ^ 24 := by
        have h := coeff_le L hL ((2 : ℝ) ^ (ea + eb)) hc
        linarith

/-- `round_shift_32` is a half-word rounding of `v/2^32`. -/
lemma rshift32_spec (v : ℤ) : |((rshift32 v : ℤ) : ℝ) - (v : ℝ) / 2 ^ 32| ≤ 1 / 2 := by
  have hd0 : (2 : ℤ) ^ 32 ≠ 0 := by norm_num
  have hdpos : (0 : ℤ) < 2 ^ 32 := by norm_num
  have hdiv : (2 : ℤ) ^ 32 * ((v + 2 ^ 31) / 2 ^ 32) + (v + 2 ^ 31) % 2 ^ 32 = v + 2 ^ 31 :=
    Int.mul_ediv_add_emod (v + 2 ^ 31) _
  have hcast : (2 : ℝ) ^ 32 * (((v + 2 ^ 31) / 2 ^ 32 : ℤ) : ℝ)
      + (((v + 2 ^ 31) % 2 ^ 32 : ℤ) : ℝ) = (v : ℝ) + 2 ^ 31 := by
    have := congrArg (fun z : ℤ => (z : ℝ)) hdiv
    push_cast at this ⊢
    linarith
  have hr0 : (0 : ℝ) ≤ (((v + 2 ^ 31) % 2 ^ 32 : ℤ) : ℝ) := by
    have := Int.emod_nonneg (v + 2 ^ 31) hd0
    exact_mod_cast this
  have hr1 : (((v + 2 ^ 31) % 2 ^ 32 : ℤ) : ℝ) < (2 : ℝ) ^ 32 := by
    have h1 : ((v + 2 ^ 31) % 2 ^ 32 : ℤ) < 2 ^ 32 := Int.emod_lt_of_pos _ hdpos
    have h2 : (((v + 2 ^ 31) % 2 ^ 32 : ℤ) : ℝ) < ((2 ^ 32 : ℤ) : ℝ) := by exact_mod_cast h1
    have h3 : ((2 ^ 32 : ℤ) : ℝ) = (2 : ℝ) ^ 32 := by norm_num
    linarith
  unfold rshift32
  have hkey : (((v + 2 ^ 31) / 2 ^ 32 : ℤ) : ℝ) - (v : ℝ) / 2 ^ 32
      = ((2 : ℝ) ^ 31 - (((v + 2 ^ 31) % 2 ^ 32 : ℤ) : ℝ)) / 2 ^ 32 := by
    field_simp
    linarith
  rw [hkey, abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ 32)]
  have habs : |(2 : ℝ) ^ 31 - (((v + 2 ^ 31) % 2 ^ 32 : ℤ) : ℝ)| ≤ 2 ^ 31 := by
    have hp32 : (2 : ℝ) ^ 32 = 2 * 2 ^ 31 := by norm_num
    rw [abs_le]
    constructor <;> linarith [hr0, hr1, hp32]
  calc |(2 : ℝ) ^ 31 - (((v + 2 ^ 31) % 2 ^ 32 : ℤ) : ℝ)| / 2 ^ 32
      ≤ (2 : ℝ) ^ 31 / 2 ^ 32 := div_le_div_of_nonneg_right habs (by norm_num)
    _ = 1 / 2 := by norm_num

/-! ## The error bound, and the `Down`-peak specializations -/

/-- **The one-block limb error.** The kernel's word `rshift32 T` approximates the exact block dot
`exactPre/2^64` within half a word plus the model mass divided by `2^32`; `C_L = (16L+17)·step^L`
and the dropped mass is `L²·2^24/2^32 = L²/256`. -/
theorem limb_error_le (L ea eb : ℕ) (hL : 1 ≤ L)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) (T : ℤ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range L, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range L, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ step ^ L / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ step ^ L / 2)
    (hT : (T : ℝ) = limbPre L ea eb α β) :
    |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre L ea eb α β ρa ρb / 2 ^ 64|
      ≤ (1 / 2 + ((2 : ℝ) ^ (ea + eb) * ((16 * (L : ℝ) + 17) * step ^ L)
          + (L : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32) / 2 ^ 32 := by
  have hpre := abs_limbPre_sub_exactPre_le L ea eb hL α β ρa ρb hα hβ hρa hρb
  have hrs := rshift32_spec T
  have hkey : |((rshift32 T : ℤ) : ℝ) - exactPre L ea eb α β ρa ρb / 2 ^ 32|
      ≤ 1 / 2 + |limbPre L ea eb α β - exactPre L ea eb α β ρa ρb| / 2 ^ 32 := by
    have hsplit : ((rshift32 T : ℤ) : ℝ) - exactPre L ea eb α β ρa ρb / 2 ^ 32
        = (((rshift32 T : ℤ) : ℝ) - (T : ℝ) / 2 ^ 32)
          + ((limbPre L ea eb α β - exactPre L ea eb α β ρa ρb) / 2 ^ 32) := by
      rw [hT]; ring
    rw [hsplit]
    refine le_trans (abs_add_le _ _) ?_
    have h2 : |(limbPre L ea eb α β - exactPre L ea eb α β ρa ρb) / 2 ^ 32|
        = |limbPre L ea eb α β - exactPre L ea eb α β ρa ρb| / 2 ^ 32 := by
      rw [abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ 32)]
    rw [h2]
    linarith
  have hq : |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre L ea eb α β ρa ρb / 2 ^ 64|
      = |((rshift32 T : ℤ) : ℝ) - exactPre L ea eb α β ρa ρb / 2 ^ 32| / 2 ^ 32 := by
    have : ((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre L ea eb α β ρa ρb / 2 ^ 64
        = (((rshift32 T : ℤ) : ℝ) - exactPre L ea eb α β ρa ρb / 2 ^ 32) / 2 ^ 32 := by
      ring
    rw [this, abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ 32)]
  rw [hq]
  apply div_le_div_of_nonneg_right _ (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ 32)
  have hd : |limbPre L ea eb α β - exactPre L ea eb α β ρa ρb| / 2 ^ 32
      ≤ ((2 : ℝ) ^ (ea + eb) * ((16 * (L : ℝ) + 17) * step ^ L)
          + (L : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32 := by
    apply div_le_div_of_nonneg_right hpre (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ 32)
  linarith

/-- `(2:ℝ)^k` is monotone, so a peak sum `ea + eb ≤ E` bounds the placement factor. -/
lemma two_pow_sum_le {ea eb E : ℕ} (h : ea + eb ≤ E) : (2 : ℝ) ^ (ea + eb) ≤ 2 ^ E :=
  pow_le_pow_right₀ (by norm_num) h

/-- **`q` at the `Down` peaks.** `L = 5`, `ea ≤ 34`, `eb ≤ 37`: `G = 1562` words. The peak data are
`block_exp(2^32) = 34` and `block_exp(6.05e10) = 37`; `C_L·2^{Ea+Eb−32} = 97·2^4 = 1552` is an
integer, so `bound_ulps`'s ceiling adds only its `+1`, and `G = 7 + 1 + 1 + 1553 = 1562`. The
operand pair is the brief's; the shipped `q_all` is two calls and the module doc names the gap. -/
theorem limb_q_error_down (ea eb : ℕ) (hea : ea ≤ 34) (heb : eb ≤ 37)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) (T : ℤ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ step ^ 5 / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ step ^ 5 / 2)
    (hT : (T : ℝ) = limbPre 5 ea eb α β) :
    |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre 5 ea eb α β ρa ρb / 2 ^ 64| ≤ 1562 / 2 ^ 32 := by
  have h := limb_error_le 5 ea eb (by norm_num) α β ρa ρb T hα hβ hρa hρb hT
  have hE : (2 : ℝ) ^ (ea + eb) ≤ 2 ^ 71 := two_pow_sum_le (by omega)
  have hC : (2 : ℝ) ^ (ea + eb) * ((16 * ((5 : ℕ) : ℝ) + 17) * step ^ 5) ≤ 1552 * 2 ^ 32 := by
    have hs5 : step ^ 5 = 1 / 2 ^ 35 := by unfold step; norm_num
    have hc97 : (16 * ((5 : ℕ) : ℝ) + 17) * (1 / 2 ^ 35) = 97 / 2 ^ 35 := by norm_num
    rw [hs5, hc97]
    have hmono : (2 : ℝ) ^ (ea + eb) * (97 / 2 ^ 35) ≤ 2 ^ 71 * (97 / 2 ^ 35) :=
      mul_le_mul_of_nonneg_right hE (by positivity)
    have heq : (2 : ℝ) ^ 71 * (97 / 2 ^ 35) = 1552 * 2 ^ 32 := by norm_num
    linarith
  have h' : |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre 5 ea eb α β ρa ρb / 2 ^ 64|
      ≤ (1 / 2 + ((2 : ℝ) ^ (ea + eb) * ((16 * ((5 : ℕ) : ℝ) + 17) * step ^ 5)
          + ((5 : ℕ) : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32) / 2 ^ 32 := by
    have h25 : (((5 : ℕ) : ℝ) ^ 2 * 2 ^ 24) = 25 * 2 ^ 24 := by norm_num
    simpa only [h25] using h
  have hbound : 1 / 2 + ((2 : ℝ) ^ (ea + eb) * ((16 * ((5 : ℕ) : ℝ) + 17) * step ^ 5)
      + ((5 : ℕ) : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32 ≤ 1562 := by
    have h25 : (((5 : ℕ) : ℝ) ^ 2 * 2 ^ 24) = 25 * 2 ^ 24 := by norm_num
    rw [h25]
    have hpos : (0 : ℝ) < 2 ^ 32 := by norm_num
    have h1 : ((2 : ℝ) ^ (ea + eb) * ((16 * ((5 : ℕ) : ℝ) + 17) * step ^ 5) + 25 * 2 ^ 24) / 2 ^ 32
        ≤ (1552 * 2 ^ 32 + 25 * 2 ^ 24) / 2 ^ 32 := by
      apply div_le_div_of_nonneg_right _ hpos.le
      linarith [hC]
    have h2 : ((1552 : ℝ) * 2 ^ 32 + 25 * 2 ^ 24) / 2 ^ 32 = 1552 + 25 / 256 := by
      field_simp
      norm_num
    linarith
  refine le_trans h' ?_
  exact div_le_div_of_nonneg_right hbound (by norm_num : (0 : ℝ) ≤ 2 ^ 32)

/-- **`dot` at the `Down` peaks.** `L = 5`, `ea ≤ 40`, `eb ≤ 35`: `G = 24842` words. The peak data
are `block_exp(5.30e11) = 40` and `block_exp(1.21e10) = 35`; `C_L·2^{Ea+Eb−32} = 97·2^8 = 24832`,
`G = 7 + 1 + 1 + 24833 = 24842`. -/
theorem limb_d_error_down (ea eb : ℕ) (hea : ea ≤ 40) (heb : eb ≤ 35)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) (T : ℤ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ step ^ 5 / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ step ^ 5 / 2)
    (hT : (T : ℝ) = limbPre 5 ea eb α β) :
    |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre 5 ea eb α β ρa ρb / 2 ^ 64| ≤ 24842 / 2 ^ 32 := by
  have h := limb_error_le 5 ea eb (by norm_num) α β ρa ρb T hα hβ hρa hρb hT
  have hE : (2 : ℝ) ^ (ea + eb) ≤ 2 ^ 75 := two_pow_sum_le (by omega)
  have hC : (2 : ℝ) ^ (ea + eb) * ((16 * ((5 : ℕ) : ℝ) + 17) * step ^ 5) ≤ 24832 * 2 ^ 32 := by
    have hs5 : step ^ 5 = 1 / 2 ^ 35 := by unfold step; norm_num
    have hc97 : (16 * ((5 : ℕ) : ℝ) + 17) * (1 / 2 ^ 35) = 97 / 2 ^ 35 := by norm_num
    rw [hs5, hc97]
    have hmono : (2 : ℝ) ^ (ea + eb) * (97 / 2 ^ 35) ≤ 2 ^ 75 * (97 / 2 ^ 35) :=
      mul_le_mul_of_nonneg_right hE (by positivity)
    have heq : (2 : ℝ) ^ 75 * (97 / 2 ^ 35) = 24832 * 2 ^ 32 := by norm_num
    linarith
  have h' : |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre 5 ea eb α β ρa ρb / 2 ^ 64|
      ≤ (1 / 2 + ((2 : ℝ) ^ (ea + eb) * ((16 * ((5 : ℕ) : ℝ) + 17) * step ^ 5)
          + ((5 : ℕ) : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32) / 2 ^ 32 := by
    have h25 : (((5 : ℕ) : ℝ) ^ 2 * 2 ^ 24) = 25 * 2 ^ 24 := by norm_num
    simpa only [h25] using h
  have hbound : 1 / 2 + ((2 : ℝ) ^ (ea + eb) * ((16 * ((5 : ℕ) : ℝ) + 17) * step ^ 5)
      + ((5 : ℕ) : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32 ≤ 24842 := by
    have h25 : (((5 : ℕ) : ℝ) ^ 2 * 2 ^ 24) = 25 * 2 ^ 24 := by norm_num
    rw [h25]
    have hpos : (0 : ℝ) < 2 ^ 32 := by norm_num
    have h1 : ((2 : ℝ) ^ (ea + eb) * ((16 * ((5 : ℕ) : ℝ) + 17) * step ^ 5) + 25 * 2 ^ 24) / 2 ^ 32
        ≤ (24832 * 2 ^ 32 + 25 * 2 ^ 24) / 2 ^ 32 := by
      apply div_le_div_of_nonneg_right _ hpos.le
      linarith [hC]
    have h2 : ((24832 : ℝ) * 2 ^ 32 + 25 * 2 ^ 24) / 2 ^ 32 = 24832 + 25 / 256 := by
      field_simp
      norm_num
    linarith
  refine le_trans h' ?_
  exact div_le_div_of_nonneg_right hbound (by norm_num : (0 : ℝ) ≤ 2 ^ 32)

/-- The documented integer `G` at `B = 1`, `bound_ulps`'s arithmetic:
`G = P/2 + 1 + (1 + (⌈C_L·2^{Ea+Eb−32}⌉ + 1))` in `ℕ` division, with `P = 15`. -/
def Gwords (P C : ℕ) : ℕ := P / 2 + 1 + (1 + (C + 1))

/-- `G` at the `Down` `q` peak: `97·2^4 = 1552`, `G = 1562`. -/
theorem Gwords_q_down : Gwords 15 1552 = 1562 := by norm_num [Gwords]

/-- `G` at the `Down` `dot` peak: `97·2^8 = 24832`, `G = 24842`. -/
theorem Gwords_d_down : Gwords 15 24832 = 24842 := by norm_num [Gwords]

/-- **`limb_error_le` at an arbitrary peak pair.** `ea ≤ Ea`, `eb ≤ Eb`, and the peak mass
`2^{Ea+Eb}·C_L + L²·2^24` over `2^32` under `G`: the one-block error is at most `G/2^32`. -/
theorem limb_error_le_G {L Ea Eb : ℕ} {G : ℝ} (hL : 1 ≤ L)
    (hGb : 1 / 2 + ((2 : ℝ) ^ (Ea + Eb) * ((16 * (L : ℝ) + 17) * step ^ L)
        + (L : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32 ≤ G)
    (ea eb : ℕ) (hea : ea ≤ Ea) (heb : eb ≤ Eb)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) (T : ℤ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range L, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range L, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ step ^ L / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ step ^ L / 2)
    (hT : (T : ℝ) = limbPre L ea eb α β) :
    |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre L ea eb α β ρa ρb / 2 ^ 64| ≤ G / 2 ^ 32 := by
  have h := limb_error_le L ea eb hL α β ρa ρb T hα hβ hρa hρb hT
  have hE : (2 : ℝ) ^ (ea + eb) ≤ 2 ^ (Ea + Eb) := two_pow_sum_le (by omega)
  have hCL : (0 : ℝ) ≤ (16 * (L : ℝ) + 17) * step ^ L := by
    have hL1 : (1 : ℝ) ≤ (L : ℝ) := by exact_mod_cast hL
    have hpos : (0 : ℝ) ≤ 16 * (L : ℝ) + 17 := by linarith
    exact mul_nonneg hpos (step_pow_pos L).le
  have hp : (0 : ℝ) < 2 ^ 32 := by norm_num
  have hmass : (2 : ℝ) ^ (ea + eb) * ((16 * (L : ℝ) + 17) * step ^ L)
      ≤ (2 : ℝ) ^ (Ea + Eb) * ((16 * (L : ℝ) + 17) * step ^ L) :=
    mul_le_mul_of_nonneg_right hE hCL
  have hpeak : 1 / 2 + ((2 : ℝ) ^ (ea + eb) * ((16 * (L : ℝ) + 17) * step ^ L)
      + (L : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32
      ≤ 1 / 2 + ((2 : ℝ) ^ (Ea + Eb) * ((16 * (L : ℝ) + 17) * step ^ L)
      + (L : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32 := by
    have hdiv : ((2 : ℝ) ^ (ea + eb) * ((16 * (L : ℝ) + 17) * step ^ L)
        + (L : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32
        ≤ ((2 : ℝ) ^ (Ea + Eb) * ((16 * (L : ℝ) + 17) * step ^ L)
        + (L : ℝ) ^ 2 * 2 ^ 24) / 2 ^ 32 :=
      div_le_div_of_nonneg_right (by linarith) hp.le
    linarith
  exact le_trans h (div_le_div_of_nonneg_right (le_trans hpeak hGb) hp.le)

/-- **`nt_q(codebook, H_rotᵀ)`: the codebook/H peak.** `L = 5`, `ea ≤ 35`, `eb ≤ 37`:
`block_exp(1.21e10) = 35`, `block_exp(6.05e10) = 37`, `97·2^{72−32−35} = 3104`,
`G = 7 + 1 + 1 + 3105 = 3114` words. -/
theorem limb_q_error_codebook (ea eb : ℕ) (hea : ea ≤ 35) (heb : eb ≤ 37)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) (T : ℤ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ step ^ 5 / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ step ^ 5 / 2)
    (hT : (T : ℝ) = limbPre 5 ea eb α β) :
    |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre 5 ea eb α β ρa ρb / 2 ^ 64| ≤ 3114 / 2 ^ 32 :=
  limb_error_le_G (L := 5) (Ea := 35) (Eb := 37) (G := 3114)
    (by norm_num) (by unfold step; norm_num) ea eb hea heb α β ρa ρb T hα hβ hρa hρb hT

/-- **`nt_q(t, codebook)` and `nt_q(f_blk, codebook)`: the second-call peak.** `L = 5`, `ea ≤ 42`,
`eb ≤ 35`: `97·2^{77−32−35} = 97·2^10 = 99328`, `G = 7 + 1 + 1 + 99329 = 99338` words. `ea = 42`
covers `block_exp(5.30e11) = 40` and the intermediate `|t| ≤ 8·1.21e10·6.05e10/2^32` (`Ea+Eb ≤ 77`). -/
theorem limb_t_error (ea eb : ℕ) (hea : ea ≤ 42) (heb : eb ≤ 35)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) (T : ℤ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ step ^ 5 / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ step ^ 5 / 2)
    (hT : (T : ℝ) = limbPre 5 ea eb α β) :
    |((rshift32 T : ℤ) : ℝ) / 2 ^ 32 - exactPre 5 ea eb α β ρa ρb / 2 ^ 64| ≤ 99338 / 2 ^ 32 :=
  limb_error_le_G (L := 5) (Ea := 42) (Eb := 35) (G := 99338)
    (by norm_num) (by unfold step; norm_num) ea eb hea heb α β ρa ρb T hα hβ hρa hρb hT

/-- `C_L·2^{Ea+Eb−32}` is an integer at both `Down` peaks, so the `⌈·⌉` in `G` is the value
itself: `97·2^4 = 1552` and `97·2^8 = 24832`. -/
theorem cL_peak_values :
    (33 + 16 * ((5 : ℝ) - 1)) * step ^ 5 * ((2 : ℝ) ^ 71 / 2 ^ 32) = 1552
      ∧ (33 + 16 * ((5 : ℝ) - 1)) * step ^ 5 * ((2 : ℝ) ^ 75 / 2 ^ 32) = 24832 := by
  constructor <;> unfold step <;> norm_num

/-! ## The two-call composition: `q_all` and `dots` end to end

`block_quads_q` (`src/learn/fold/fit/icm.rs:785`) computes `q = nt_q(nt_q(codebook, H_rotᵀ), codebook)`
and `sweep_rows` (`:904`) computes `dots = nt_q(f_blk, codebook)`. One `nt_q` entry is an eight-term
inner product of Q32.32 values, so the end-to-end error of one entry is the last call's own bound plus
the intermediate operand's word error carried through the codebook column. `dot8_comp_le` is that
composition; `q_all_end2end` and `dots_end2end` are the two shipped calls at their measured peaks, in
real units (the `G` of `limb_error_le` divided by `2^32`). -/

/-- One `nt_q` entry, in real units: the eight-term inner product of the block. -/
def dot8 (a b : Fin 8 → ℝ) : ℝ := ∑ k, a k * b k

/-- A perturbation of one operand moves the eight-term inner product by at most `8·εa·Mb`. -/
lemma dot8_error_le {a a' b : Fin 8 → ℝ} {εa Mb : ℝ}
    (ha : ∀ k, |a' k - a k| ≤ εa) (hb : ∀ k, |b k| ≤ Mb) (hεa : 0 ≤ εa) :
    |dot8 a' b - dot8 a b| ≤ 8 * εa * Mb := by
  have hterm : ∀ k : Fin 8, |a' k * b k - a k * b k| ≤ εa * Mb := by
    intro k
    have h1 : a' k * b k - a k * b k = (a' k - a k) * b k := by ring
    rw [h1, abs_mul]
    exact mul_le_mul (ha k) (hb k) (abs_nonneg _) hεa
  have hsum : |dot8 a' b - dot8 a b|
      = |∑ k : Fin 8, (a' k * b k - a k * b k)| := by
    rw [dot8, dot8, ← Finset.sum_sub_distrib]
  rw [hsum]
  calc |∑ k : Fin 8, (a' k * b k - a k * b k)|
      ≤ ∑ k : Fin 8, |a' k * b k - a k * b k| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _k : Fin 8, εa * Mb := Finset.sum_le_sum (fun k _ => hterm k)
    _ = 8 * (εa * Mb) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        norm_num
    _ = 8 * εa * Mb := by ring

/-- **The composition.** The computed entry `ĉ` is within `εc` of the inner product of the *computed*
intermediate operand; the intermediate's error `εa` then carries through `b`'s magnitude `Mb`. -/
lemma dot8_comp_le {a a' b : Fin 8 → ℝ} {ĉ εc εa Mb : ℝ}
    (hcall : |ĉ - dot8 a' b| ≤ εc)
    (ha : ∀ k, |a' k - a k| ≤ εa) (hb : ∀ k, |b k| ≤ Mb) (hεa : 0 ≤ εa) :
    |ĉ - dot8 a b| ≤ εc + 8 * εa * Mb := by
  have h := dot8_error_le ha hb hεa
  have hsplit : ĉ - dot8 a b = (ĉ - dot8 a' b) + (dot8 a' b - dot8 a b) := by ring
  rw [hsplit]
  exact le_trans (abs_add_le _ _) (add_le_add hcall h)

/-- **`q_all` end to end.** `block_quads_q` is two `nt_q` calls: `t = nt_q(codebook, H_rotᵀ)` at
`(1.21e10, 6.05e10)` (`Ea+Eb = 72`, `G = 3114`), then `q = nt_q(t, codebook)` with
`|t| ≤ 8·1.21e10·6.05e10/2^32` (`Ea+Eb ≤ 77`, `G = 99338`). The end-to-end entry error is the second
call's bound plus the first call's word error through the codebook column: `8·(3114/2^32)·(1.21e10/2^32)`.
The `8·εa·Mb` term is the triangle sum over the eight coordinates and is sharp for these hypotheses: each
term attains `εa·Mb` when `|t̂ − t| = εa` and `|C| = Mb` at all eight. -/
theorem q_all_end2end (t th C : Fin 8 → ℝ) (qh : ℝ)
    (ht : ∀ k, |th k - t k| ≤ 3114 / 2 ^ 32)
    (hC : ∀ k, |C k| ≤ 12100000000 / 2 ^ 32)
    (hcall : |qh - dot8 th C| ≤ 99338 / 2 ^ 32) :
    |qh - dot8 t C|
      ≤ 99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32) :=
  dot8_comp_le hcall ht hC (by norm_num)

/-- `q_all`'s end-to-end constant as one exact rational: `728088661250048/2^64 ≈ 3.947e-5`. -/
theorem epsilon_q_end2end :
    99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32) = 728088661250048 / 2 ^ 64 := by
  norm_num

/-- **`dots` end to end.** `dots = nt_q(f_blk, codebook)` at `(5.30e11, 1.21e10)` (`G = 24842`),
plus the `f_blk` word error `εf` through the codebook column. `εf` is the state error: see
`fblk_error_le` for its one-step part and the missing recurrence named in the module doc. -/
theorem dots_end2end (f fh C : Fin 8 → ℝ) (dh εf : ℝ) (hεf : 0 ≤ εf)
    (hf : ∀ k, |fh k - f k| ≤ εf)
    (hC : ∀ k, |C k| ≤ 12100000000 / 2 ^ 32)
    (hcall : |dh - dot8 fh C| ≤ 24842 / 2 ^ 32) :
    |dh - dot8 f C| ≤ 24842 / 2 ^ 32 + 8 * εf * (12100000000 / 2 ^ 32) :=
  dot8_comp_le hcall hf hC hεf

/-- **`f_blk`'s error, one step.** `icm.rs:960-965` is `f_blk = add_q(f_res, mul_q(a, v))` with
`v = nt_q(s_blk, cols)` at `(2^32, 6.05e10)` (`G = 3114`) and `|a| ≤ 1`: the block's error is at most
`εres + εv`, where `|f̂_res − f_res| ≤ εres` is the sweep's residual-state invariant — the named
missing model lemma (the `sub_q` recurrence at `:1045`). The `mul_q` product's own half-word rounding
and `add_q`'s overflow branch are excluded from this model. -/
theorem fblk_error_le (v vh fr frh : Fin 8 → ℝ) (a εv εres : ℝ)
    (hv : ∀ k, |vh k - v k| ≤ εv) (hr : ∀ k, |frh k - fr k| ≤ εres)
    (ha : |a| ≤ 1) :
    ∀ k, |(frh k + a * vh k) - (fr k + a * v k)| ≤ εres + εv := by
  intro k
  have h1 : (frh k + a * vh k) - (fr k + a * v k) = (frh k - fr k) + a * (vh k - v k) := by ring
  rw [h1]
  have hmul : |a| * |vh k - v k| ≤ 1 * εv :=
    mul_le_mul ha (hv k) (abs_nonneg _) (by norm_num)
  calc |(frh k - fr k) + a * (vh k - v k)|
      ≤ |frh k - fr k| + |a * (vh k - v k)| := abs_add_le _ _
    _ = |frh k - fr k| + |a| * |vh k - v k| := by rw [abs_mul]
    _ ≤ εres + 1 * εv := add_le_add (hr k) hmul
    _ = εres + εv := by ring

/-- **`dots` end to end, with the residual state `εres` explicit.** `εf = εres + 3114/2^32` at
`|a| ≤ 1`. The `εres` term is the missing sweep-recurrence invariant; the rest is derived here. -/
theorem dots_end2end_state (v vh fr frh C : Fin 8 → ℝ) (a dh εv εres : ℝ)
    (hv : ∀ k, |vh k - v k| ≤ εv) (hr : ∀ k, |frh k - fr k| ≤ εres)
    (ha : |a| ≤ 1) (hεres : 0 ≤ εres)
    (hC : ∀ k, |C k| ≤ 12100000000 / 2 ^ 32)
    (hcall : |dh - dot8 (fun k => frh k + a * vh k) C| ≤ 24842 / 2 ^ 32) :
    |dh - dot8 (fun k => fr k + a * v k) C|
      ≤ 24842 / 2 ^ 32 + 8 * (εres + εv) * (12100000000 / 2 ^ 32) := by
  have hεv : 0 ≤ εv := le_trans (abs_nonneg _) (hv 0)
  exact dots_end2end (fun k => fr k + a * v k) (fun k => frh k + a * vh k) C dh (εres + εv)
    (by linarith) (fblk_error_le v vh fr frh a εv εres hv hr ha) hC hcall

/-- `dots`' end-to-end constant with an exact residual state (`εres = 0`, `εv = 3114/2^32`):
`408130777567232/2^64 ≈ 2.213e-5`. -/
theorem epsilon_d_end2end_local :
    24842 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32) = 408130777567232 / 2 ^ 64 := by
  norm_num

end Learn.Limb
