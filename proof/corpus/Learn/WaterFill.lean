import ApeironWord
import Perf.CostCorrespondence
import Learn.Grad.DenseLinearSpec
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted reverse water-filling solver: integer specification and Gaussian bridge

`fold_port.word_stats_basis_solve_water_fill` (`proof/extract/fold_port.rs:363`) calls
`fold_port.stats.basis.solve_water_fill` (`src/learn/fold/stats/basis.rs:1138`), the infallible entry
point over `water_fill_words` (`:965`). This module states **what** that body returns, in the
extraction's own `Q32.32` words, and links the integer formulas to the real Gaussian
rate–distortion theorem proved in `Perf.CostCorrespondence`.

* **The returned triple.** After the bisection on the log-domain level `t`, the body returns
  `theta = exp2 t`, `distortion = Σ min(λ, θ)` and `rate = ½ Σ_{λ > θ} (log₂ λ − log₂ θ)`
  (`distWords`, `rateWords`); `WaterFillSpec` pins the three fields and the level bracket, and
  `earlyReturn` pins the two early returns.
* **The integer invariants.** `distWords` is nonnegative, monotone in `θ`, bounded above by `Σ λ`
  and by `n·θ`, and splits on the active set (`distWords_active_set`); `rateWords` is nonnegative
  and antitone in the level (`rateWords_anti`). The budget invariant is
  `|rate − (budget − side)| ≤ 4n` (`RATE_TOL_Q · n`).
* **The bridge.** `distWords` denotes the Gaussian floor `Σ min(λ_i, θ)` exactly, because `min`
  commutes with the positive scaling `w ↦ w / 2³²` (`distWords_denote_scale`). The real
  rate–distortion statement is `gaussian_floor` (the in-tree `waterfill_floor` at `c = 2 ln 2`),
  and `gaussian_floor_words` replaces its left side with the integer fold.
* **The distortion loop.** `distortion_words_loop_spec` refines the extracted
  `distortion_words_loop` to the integer fold by `loop.spec_decr_nat`, and
  `distortion_words_spec` carries it to `distortion_words` (`Σ min(λ, θ)` as the `i64` result),
  through `narrow` and `ok_or`, on the stated `i128` window `[0, 2¹²⁶]`; `distortion_words_post`
  collects the three branches (non-positive level, in-window fold, `OutOfRange` refusal).
* **The rate body.** `rate_words` (`basis.rs:931`) is refined on both level branches: the flag loop
  `rate_words_loop0` gives `Ok i64::MAX` / `Ok 0` (`rate_words_nonpos`), and `rate_words_loop1` is
  refined to `rateAt` (the active-log sum) plus `rateBad` (a `checked_sub` outside the `i64` window),
  with `rate_words_pos_spec` the half-sum through `>>> 1`/`narrow`, `rate_words_pos_fail` the
  `OutOfRange` of a failing difference and `rate_words_pos_none` the `OutOfRange` of an absent
  `log2 theta`. The per-word `log2` values stay the caller's `hlog`.
* **The bisection's sum loop.** `water_fill_words_loop1` (the `sum_lambda` accumulation at
  `basis.rs:1000`) is the one word loop of `water_fill_words` that feeds the `target ≤ 0` early
  return's distortion field. `water_fill_words_loop1_spec` refines it to the prefix sum
  `Σ λ` on the same `i128` window, and `water_fill_words_loop1_distWords` reads that sum as
  `distWords` at the maximum level (`distWords_maxOf`), under the body's `max > 0` guard.
* **The bisection's doubling walk.** `water_fill_words_loop2` (the `t_lo` walk at `basis.rs:1021`)
  is refined by `water_fill_words_loop2_spec` to `feasible ∨ error ∨ exhausted`; the exhausted
  branch carries the closed form of the returned level and the infeasibility of every probed level,
  because the loop's result type drops `walked` and an exhausted walk has not established
  feasibility. `water_fill_words_loop2_converged` is the caller-facing form under an explicit
  convergence hypothesis.
* **The bisection.** `water_fill_words_loop3` (the `t_hi`/`t_lo` bisection at `basis.rs:1049`) is
  refined by `water_fill_words_loop3_spec` to `converged ∨ error ∨ (exhausted with a rejected
  level)`, with the bracket in `BisectBracket`, the checked `i128` midpoint in three body shapes
  (`l3_body_eq_probe_pos`, `l3_body_eq_probe_neg_even`, `l3_body_eq_probe_odd`) and the probe tail
  in `l3_tail_run`. The whole-body `water_fill_words` composition still has no refinement here.
* **The log pre-pass.** `water_fill_words_loop0_spec` refines the extracted `water_fill_words_loop0`
  to the pre-bisection maxima: `max = maxOf (vals λ)`, `min_pos = minPosOf (vals λ)` (the running
  positive minimum from the `i64::MAX` sentinel) and `logs = map water_log`, including the
  `Vec.index_mut` back-function step (`logsOf_set`, `Vec.index_mut_usize_spec`). The per-word
  `log2` value is the caller's hypothesis `hlog` (pinned by `wv`), and the zero `logs` vector is
  the caller's `hlg` (`replicate len 0`), not constructed here.

**Model separation.** `gaussian_floor` is a statement about the Gaussian model: variance `λ_i`,
coding at rate `r_i` leaves distortion `λ_i · exp(−2 ln 2 · r_i)`. It is not a statement about the
E8/O8 codebook (`Learn.E8Code` refines that codebook separately) and it does not identify the two.
`basis.rs` records the same separation: the exponential distortion law is a model for real weights
and exact for a Gaussian source; the codebook, the coefficient variances and the measured held-out
KL are anchors outside these theorems.

**Fixed-point approximation, stated exactly.** Two approximations separate the extracted integers
from the real model, and neither is discharged here:

1. `log2` and `exp2` are the `Q32.32` series `fixed_ln`/`fixed_exp`, so `theta` and `rate` carry
   their rounding. `rateWords_floor` states the only gap proved here: the returned rate is the
   floor of half the exact integer log-sum, so its distance from that half-sum is below `2⁻³²` in
   value.
2. The bisection stops at `|rate − (budget − side)| ≤ 4n` rather than at equality
   (`budgetTolerance`). No theorem here bounds the induced distortion error; that bound is open.

**What is fired and what is proved.** The `#guard` block runs the extracted body under Lean's
evaluator (`Result.match`); the kernel does not reduce the `partial_fixpoint` monadic term, so
`by decide` on the same call does not reduce, and the guard is an evaluator reading, not a kernel
check. The
theorems below are kernel-checked and independent of it. The refinement below covers five steps of
the body: `distortion_words` and its single loop, `rate_at_logs` with its loop and its
`>>> 1`/`narrow` tail (`rate_at_logs_spec`), the `sum_lambda` word loop of `water_fill_words`
(`water_fill_words_loop1_spec`), the first bisection's doubling walk
(`water_fill_words_loop2_spec`, `water_fill_words_loop2_converged`), the second bisection loop
`water_fill_words_loop3` (`water_fill_words_loop3_spec`), and the log pre-pass
`water_fill_words_loop0` (`water_fill_words_loop0_spec`). The whole-body `water_fill_words`
composition has no refinement here, so `WaterFillSpec` remains the specification of that body
rather than a proved consequence of it.

Trust boundary: rustc/LLVM codegen, the `Vec`/`Slice` layout and the `log2`/`exp2` series accuracy
are not covered. The `i128` window and the nonnegative-word hypotheses are on the input, not derived.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Learn.WaterFill

open Aeneas Aeneas.Std Result
open apeiron_word

/-! ## 1 · The integer specification -/

/-- The `Q32.32` scale, `2³²`. -/
def SCALE : ℤ := 4294967296

/-- `i64::MIN`, the body's sentinel log for a non-positive variance (`water_log`). -/
def LOG_NONE : ℤ := -9223372036854775808

/-- The extraction's per-coefficient rate threshold `RATE_TOL_Q` (`basis.rs:135`). -/
def RATE_TOL : ℤ := 4

/-- The extraction's bisection count `BISECTION_ITERS` (`basis.rs:132`). -/
def BISECTION_ITERS : ℕ := 200

/-- The three `i64` fields of the extracted `WaterFill` (`basis.rs:908`), as integers. -/
structure Out where
  /-- The water level `θ`, in `Q32.32` words. -/
  theta : ℤ
  /-- `D(θ) = Σ min(λ, θ)`, in `Q32.32` words. -/
  distortion : ℤ
  /-- `R(θ) = Σ max(0, ½ log₂(λ/θ))`, in `Q32.32` bits. -/
  rate : ℤ
deriving DecidableEq

/-- The extracted struct read as an `Out` triple. -/
def outOf (w : fold_port.stats.basis.WaterFill) : Out :=
  ⟨w.theta.val, w.distortion.val, w.rate.val⟩

/-- The value list of a `Slice I64` of `Q32.32` words. -/
def vals (L : List Std.I64) : List ℤ := L.map (fun w => w.val)

/-- `D(θ) = Σ min(λ, θ)`; `θ ≤ 0` gives `0`. This is the arithmetic of
`fold_port.stats.basis.distortion_words` (`basis.rs:918`) with the `i128` accumulator collapsed. -/
def distWords (L : List ℤ) (theta : ℤ) : ℤ :=
  if theta ≤ 0 then 0 else (L.map (fun l => min l theta)).sum

/-- One term of the rate accumulator. -/
def rateTerm (l : ℤ) (t : ℤ) : ℤ := if l ≠ LOG_NONE ∧ t < l then l - t else 0

/-- The accumulator of `fold_port.stats.basis.rate_at_logs` (`basis.rs:1104`): the sum of
`log₂ λ_i − t` over the entries above the level, with the `i64::MIN` sentinel skipped. -/
def rateSum (logs : List ℤ) (t : ℤ) : ℤ := (logs.map (fun l => rateTerm l t)).sum

/-- `R(t) = ½ · rateSum`; the body's `>>> 1` on a nonnegative accumulator is floor division. -/
def rateWords (logs : List ℤ) (t : ℤ) : ℤ := rateSum logs t / 2

/-- `theta` is the extracted `exp2` of the level `t`. -/
def ThetaIsExp (t : Std.I64) (theta : Std.I64) : Prop :=
  fold_port.stats.basis.exp2 t = ok (some theta)

/-- `t` is the extracted `log2` of the word `x`. -/
def LogOf (x : Std.I64) (t : ℤ) : Prop :=
  ∃ g : Std.I64, fold_port.stats.basis.log2 x = ok (some g) ∧ g.val = t

/-- One entry of the body's `water_log` (`basis.rs:1096`): the extracted `log2` on a positive
variance, and the `i64::MIN` sentinel otherwise. -/
def WaterLogOf (l : Std.I64) (g : ℤ) : Prop :=
  (0 < l.val ∧ LogOf l g) ∨ (l.val ≤ 0 ∧ g = LOG_NONE)

/-- The log vector the body builds, elementwise. -/
def LogsAreWaterLogs : List Std.I64 → List ℤ → Prop
  | [], [] => True
  | l :: L, g :: G => WaterLogOf l g ∧ LogsAreWaterLogs L G
  | _, _ => False

/-- The budget invariant: the returned rate is within `RATE_TOL_Q · n = 4n` of the target
`budget − side`, the threshold the extraction's `converged` uses. -/
def budgetTolerance (n rate budget side : ℤ) : Prop :=
  |rate - (budget - side)| ≤ RATE_TOL * n

/-- The level bracket the bisection maintains. -/
def levelBracket (logs : List ℤ) (budget side tLo tHi t : ℤ) : Prop :=
  tLo ≤ t ∧ t ≤ tHi ∧ rateWords logs tHi ≤ budget - side ∧ budget - side ≤ rateWords logs tLo

/-- **The exact integer postcondition of `word_stats_basis_solve_water_fill`**, on the bisection
path: the returned triple is the extracted `exp2` of the level, the `min` distortion fold at that
level, and the half-log rate fold at the level the body re-reads from the rounded `theta`
(`rate_words`, `basis.rs:936`); the canonical level `t` sits in the maintained bracket, and the
returned rate satisfies the budget tolerance. -/
def WaterFillSpec (L : List Std.I64) (logs : List ℤ) (budget side : ℤ)
    (tLo tHi t theta : Std.I64) (o : Out) : Prop :=
  LogsAreWaterLogs L logs ∧ ThetaIsExp t theta ∧
    o.theta = theta.val ∧ o.distortion = distWords (vals L) theta.val ∧
      (∃ tRate : ℤ, LogOf theta tRate ∧ o.rate = rateWords logs tRate) ∧
        levelBracket logs budget side tLo.val tHi.val t.val ∧
          budgetTolerance (vals L).length o.rate budget side

/-- The maximum of a word list, `0` on the empty list (the body's `max` accumulator). -/
def maxOf (L : List ℤ) : ℤ := L.foldr max 0

/-- **The two early returns of `water_fill_words`**: an empty or non-positive input returns the
zero triple, and a non-positive rate target returns the level `max λ`, the full distortion and zero
rate (`basis.rs:975`, `:1004`). -/
def earlyReturn (L : List ℤ) (budget side : ℤ) (o : Out) : Prop :=
  ((L = [] ∨ ∀ l ∈ L, l ≤ 0) ∧ o = ⟨0, 0, 0⟩) ∨
    (budget - side ≤ 0 ∧ o = ⟨maxOf L, L.sum, 0⟩)

/-! ## 2 · The integer invariants -/

/-- A constant list sums to `|L|` times the constant. -/
theorem sum_map_const (L : List ℤ) (c : ℤ) : (L.map (fun _ => c)).sum = (L.length : ℤ) * c := by
  induction L with
  | nil => simp
  | cons l ls ih =>
    rw [List.map_cons, List.sum_cons, ih, List.length_cons]
    push_cast
    ring

/-- Pointwise domination maps to sum domination. -/
theorem list_sum_le_map {f g : ℤ → ℤ} (L : List ℤ) (h : ∀ l ∈ L, f l ≤ g l) :
    (L.map f).sum ≤ (L.map g).sum := by
  induction L with
  | nil => simp
  | cons l ls ih =>
    simp only [List.map_cons, List.sum_cons]
    exact add_le_add (h l (by simp)) (ih (fun x hx => h x (by simp [hx])))

/-- The body's `theta ≤ 0` branch returns distortion zero. -/
theorem distWords_zero_of_nonpos (L : List ℤ) {theta : ℤ} (h : theta ≤ 0) :
    distWords L theta = 0 := by
  unfold distWords
  rw [if_pos h]

/-- `D(θ) ≥ 0` on nonnegative variances. -/
theorem distWords_nonneg (L : List ℤ) {theta : ℤ} (hL : ∀ l ∈ L, 0 ≤ l) :
    0 ≤ distWords L theta := by
  unfold distWords
  by_cases h : theta ≤ 0
  · rw [if_pos h]
  · rw [if_neg h]
    apply List.sum_nonneg
    intro z hz
    obtain ⟨l, hl, rfl⟩ := List.mem_map.mp hz
    exact le_min (hL l hl) (le_of_lt (lt_of_not_ge h))

/-- `D(θ) ≤ Σ λ`: the distortion never exceeds the source variance. -/
theorem distWords_le_sum (L : List ℤ) {theta : ℤ} (hL : ∀ l ∈ L, 0 ≤ l) :
    distWords L theta ≤ L.sum := by
  unfold distWords
  by_cases h : theta ≤ 0
  · rw [if_pos h]
    exact List.sum_nonneg hL
  · rw [if_neg h]
    simpa using list_sum_le_map L (g := id) (fun l _ => min_le_left l theta)

/-- `D(θ) ≤ n·θ` at a positive level: every term is capped at the level. -/
theorem distWords_le_card_mul (L : List ℤ) {theta : ℤ} (h : 0 < theta) :
    distWords L theta ≤ (L.length : ℤ) * theta := by
  unfold distWords
  rw [if_neg (not_le.mpr h)]
  have hle : (L.map (fun l => min l theta)).sum ≤ (L.map (fun _ => theta)).sum :=
    list_sum_le_map L (fun l _ => min_le_right l theta)
  rw [sum_map_const] at hle
  exact hle

/-- `D` is monotone in the level. -/
theorem distWords_mono (L : List ℤ) {theta theta' : ℤ} (hL : ∀ l ∈ L, 0 ≤ l)
    (h : theta ≤ theta') : distWords L theta ≤ distWords L theta' := by
  by_cases h0 : theta ≤ 0
  · rw [distWords_zero_of_nonpos L h0]
    exact distWords_nonneg L hL
  · have hθ : 0 < theta := lt_of_not_ge h0
    have hθ' : 0 < theta' := lt_of_lt_of_le hθ h
    rw [show distWords L theta = (L.map (fun l => min l theta)).sum from by
          unfold distWords; rw [if_neg h0],
        show distWords L theta' = (L.map (fun l => min l theta')).sum from by
          unfold distWords; rw [if_neg (not_le.mpr hθ')]]
    exact list_sum_le_map L (fun l _ => min_le_min le_rfl h)

/-- The active-set split: the entries below the level contribute their variance, the entries at or
above it contribute the level. -/
theorem distWords_active_set (L : List ℤ) {theta : ℤ} (h : 0 < theta) :
    distWords L theta =
      (L.map (fun l => if l < theta then l else 0)).sum +
        (L.map (fun l => if l < theta then 0 else theta)).sum := by
  unfold distWords
  rw [if_neg (not_le.mpr h), ← List.sum_map_add]
  congr 1
  apply List.map_congr_left
  intro l _
  by_cases hl : l < theta
  · simp [hl, min_eq_left hl.le]
  · simp [hl, min_eq_right (le_of_not_gt hl)]

/-- Every term of the rate accumulator is nonnegative. -/
theorem rateTerm_nonneg (l : ℤ) (t : ℤ) : 0 ≤ rateTerm l t := by
  unfold rateTerm
  split_ifs with h
  · omega
  · exact le_rfl

/-- `rateSum` is a sum of nonnegative terms. -/
theorem rateSum_nonneg (logs : List ℤ) (t : ℤ) : 0 ≤ rateSum logs t := by
  unfold rateSum
  exact List.sum_nonneg (fun z hz => by
    obtain ⟨l, _, rfl⟩ := List.mem_map.mp hz
    exact rateTerm_nonneg l t)

/-- `rateWords ≥ 0`. -/
theorem rateWords_nonneg (logs : List ℤ) (t : ℤ) : 0 ≤ rateWords logs t := by
  unfold rateWords
  exact Int.ediv_nonneg (rateSum_nonneg logs t) (by norm_num)

/-- One term falls as the level rises. -/
theorem rateTerm_anti (l : ℤ) {t t' : ℤ} (htt : t ≤ t') : rateTerm l t' ≤ rateTerm l t := by
  unfold rateTerm
  by_cases h1 : l ≠ LOG_NONE ∧ t' < l
  · rw [if_pos h1, if_pos ⟨h1.1, lt_of_le_of_lt htt h1.2⟩]
    omega
  · rw [if_neg h1]
    by_cases h2 : l ≠ LOG_NONE ∧ t < l
    · rw [if_pos h2]
      omega
    · rw [if_neg h2]

/-- The rate falls as the level rises: the defining property of reverse water-filling. -/
theorem rateSum_anti (logs : List ℤ) {t t' : ℤ} (htt : t ≤ t') :
    rateSum logs t' ≤ rateSum logs t := by
  induction logs with
  | nil => simp [rateSum]
  | cons l ls ih => exact add_le_add (rateTerm_anti l htt) ih

/-- `rateWords` is antitone in the level. -/
theorem rateWords_anti (logs : List ℤ) {t t' : ℤ} (htt : t ≤ t') :
    rateWords logs t' ≤ rateWords logs t := by
  unfold rateWords
  exact Int.ediv_le_ediv (by norm_num) (rateSum_anti logs htt)

/-- The floor gap: the returned rate is floor half the log-sum, so twice it is at most the sum and
within two of it. This is the exact fixed-point content of the `>>> 1` step. -/
theorem rateWords_floor (logs : List ℤ) (t : ℤ) :
    2 * rateWords logs t ≤ rateSum logs t ∧ rateSum logs t < 2 * rateWords logs t + 2 := by
  unfold rateWords
  have h := Int.emod_add_mul_ediv (rateSum logs t) 2
  have h0 := Int.emod_nonneg (rateSum logs t) (by norm_num : (2 : ℤ) ≠ 0)
  have h1 := Int.emod_lt_of_pos (rateSum logs t) (by norm_num : (0 : ℤ) < 2)
  constructor <;> omega

/-! ## 3 · The Gaussian bridge -/

/-- The distortion fold denotes the Gaussian floor exactly.** On a positive level,
`D(θ) = Σ min(λ_i, θ)` holds after scaling by `2³²`, because `min` commutes with a positive scale
and the integer cast commutes with `min`. No approximation enters this identity. -/
theorem distWords_denote (L : List ℤ) {theta : ℤ} (h : 0 < theta) :
    ((distWords L theta : ℤ) : ℝ) = (L.map (fun (l : ℤ) => min (l : ℝ) (theta : ℝ))).sum := by
  unfold distWords
  rw [if_neg (not_le.mpr h)]
  induction L with
  | nil => simp
  | cons l ls ih =>
    rw [List.map_cons, List.sum_cons, Int.cast_add, List.map_cons, List.sum_cons, ih]
    congr 1
    by_cases hl : l ≤ theta
    · rw [min_eq_left hl, min_eq_left (by exact_mod_cast hl)]
    · have hlt : theta < l := not_le.mp hl
      rw [min_eq_right (le_of_lt hlt), min_eq_right (by exact_mod_cast (le_of_lt hlt))]

/-- Conversion between a list sum and the `Fin`-indexed sum the `Finset` theorems use. -/
theorem sum_get_eq_map (L : List ℤ) (f : ℤ → ℝ) :
    (∑ i : Fin L.length, f (L.get i)) = (L.map f).sum := by
  rw [← List.sum_ofFn]
  congr 1
  simp only [List.get_eq_getElem]
  exact List.ofFn_getElem_eq_map L f

/-- The same identity in value units: the integer fold divided by `2³²` is the real `min` fold at
the real variances and level. -/
theorem distWords_denote_scale (L : List ℤ) {theta : ℤ} (h : 0 < theta) :
    ((distWords L theta : ℤ) : ℝ) / (SCALE : ℝ) =
      (L.map (fun (l : ℤ) => min ((l : ℝ) / (SCALE : ℝ)) ((theta : ℝ) / (SCALE : ℝ)))).sum := by
  have hS : (0 : ℝ) < (SCALE : ℝ) := by unfold SCALE; norm_num
  rw [distWords_denote L h]
  induction L with
  | nil => simp
  | cons l ls ih =>
    rw [List.map_cons, List.sum_cons, add_div, List.map_cons, List.sum_cons, ih]
    congr 1
    exact (min_div_div_right (a := (l : ℝ)) (b := (theta : ℝ)) hS.le).symm

/-- The `c = 2 ln 2` Gaussian rate–distortion floor: `Perf.CostCorrespondence.waterfill_floor` at
`c = 2 ln 2`, the form that matches the quantiser's bit units. The sibling folding corpus states
the same theorem as `waterfill_bits_optimal`
(`sanguine` `proof/Apeiron/Folding/Quantisation.lean`); that file is a pointer, this proof is the
evidence. -/
theorem gaussian_floor {ι : Type*} [Fintype ι] (lam r : ι → ℝ) {theta : ℝ} (hθ : 0 < theta)
    (hlam : ∀ i, 0 < lam i) (hr : ∀ i, 0 ≤ r i)
    (hbud : ∑ i, r i ≤ ∑ i, max 0 (Real.log (lam i / theta) / (2 * Real.log 2))) :
    ∑ i, min (lam i) theta ≤ ∑ i, lam i * Real.exp (-(2 * Real.log 2 * r i)) := by
  have hc : 0 < 2 * Real.log 2 := by
    have h := Real.log_pos (by norm_num : (1 : ℝ) < 2)
    linarith
  have h := Perf.CostCorrespondence.waterfill_floor (Finset.univ : Finset ι) lam r hc hθ
    (fun i _ => hlam i) (fun i _ => hr i) (by simpa using hbud)
  simpa [mul_assoc] using h

/-- **The bridge.** Every real allocation whose total rate is at most the water-filling rate at the
denoted level leaves at least the integer distortion fold, in value units. This is the Gaussian
theorem with the left side replaced by `distWords`; the remaining gap is the rounding of the integer
level and the budget tolerance, both stated above and neither discharged here. -/
theorem gaussian_floor_words (L : List ℤ) {theta : ℤ} (hθ : 0 < theta)
    (hL : ∀ l ∈ L, 0 < l) (r : Fin L.length → ℝ) (hr : ∀ i, 0 ≤ r i)
    (hbud : ∑ i, r i ≤
      ∑ i, max 0 (Real.log (((L.get i : ℤ) : ℝ) / (SCALE : ℝ) / ((theta : ℝ) / (SCALE : ℝ))) /
        (2 * Real.log 2))) :
    ((distWords L theta : ℤ) : ℝ) / (SCALE : ℝ) ≤
      ∑ i, (((L.get i : ℤ) : ℝ) / (SCALE : ℝ)) * Real.exp (-(2 * Real.log 2 * r i)) := by
  have hS : (0 : ℝ) < (SCALE : ℝ) := by unfold SCALE; norm_num
  have hθr : 0 < (theta : ℝ) / (SCALE : ℝ) := div_pos (by exact_mod_cast hθ) hS
  have hlam : ∀ i : Fin L.length, 0 < ((L.get i : ℤ) : ℝ) / (SCALE : ℝ) := by
    intro i
    have hi : L.get i ∈ L := List.get_mem L i
    exact div_pos (by exact_mod_cast hL _ hi) hS
  have h := gaussian_floor (fun i : Fin L.length => ((L.get i : ℤ) : ℝ) / (SCALE : ℝ)) r hθr
    hlam hr hbud
  rw [distWords_denote_scale L hθ]
  rw [← sum_get_eq_map L (fun (l : ℤ) => min ((l : ℝ) / (SCALE : ℝ)) ((theta : ℝ) / (SCALE : ℝ)))]
  exact gaussian_floor (fun i : Fin L.length => ((L.get i : ℤ) : ℝ) / (SCALE : ℝ)) r hθr
    hlam hr hbud

/-! ## 4 · Witnesses and the fired body -/

/-- Read the `Some` field of an extracted call, for the evaluator. -/
def runField {α : Type} (r : Result α) (f : α → ℤ) : Option ℤ :=
  match r.match with
  | .ok a => some (f a)
  | _ => none

/-- The witness variances `λ = [4, 1]` in `Q32.32` words. -/
def lam4 : List Std.I64 := [17179869184#i64, 4294967296#i64]

/-- The witness variances `λ = [1, 1]` in `Q32.32` words. -/
def lam1 : List Std.I64 := [4294967296#i64, 4294967296#i64]

/-- The witness slices. -/
def slice4 : Slice Std.I64 :=
  Slice.from lam4 (by have h := dense_usize_le_max (2#usize); simpa [lam4] using h)

/-- The witness slices. -/
def slice1 : Slice Std.I64 :=
  Slice.from lam1 (by have h := dense_usize_le_max (2#usize); simpa [lam1] using h)

theorem lam4_vals : vals lam4 = [17179869184, 4294967296] := rfl

theorem lam1_vals : vals lam1 = [4294967296, 4294967296] := rfl

/-- The pure spec fields at the `λ = [4, 1]` witness: level `2²⁵`, distortion `2²⁶`, rate `8` bits.
Kernel-checked; it records the arithmetic, not the body's run. -/
theorem witness_fields_4 :
    distWords (vals lam4) 33554432 = 67108864 ∧
      rateWords [146028888064, 137438953472] (25 * 4294967296) = 34359738368 ∧
        budgetTolerance 2 34359738368 34359738368 0 := by
  refine ⟨by decide, by decide, by unfold budgetTolerance RATE_TOL; decide⟩

/-- The pure spec fields at the `λ = [1, 1]` witness: level `2³⁰ + 1`, rate `2` bits less the
bisection's six-word shortfall, inside the `4n = 8` tolerance. The tolerance is live here, not
vacuous: the shortfall is `6 ≠ 0`. -/
theorem witness_fields_1 :
    distWords (vals lam1) 1073741825 = 2147483650 ∧
      rateWords [137438953472, 137438953472] (30 * 4294967296) = 8589934592 ∧
        budgetTolerance 2 8589934586 8589934592 0 ∧
          ((8589934586 : ℤ) - 8589934592) = -6 := by
  refine ⟨by decide, by decide, by unfold budgetTolerance RATE_TOL; decide, by norm_num⟩

/-- The two early returns are not vacuous: the zero triple is the empty-input value, and the
full-distortion triple is the `target ≤ 0` value at `λ = [4, 1]`. -/
theorem early_witness :
    earlyReturn [] 0 0 ⟨0, 0, 0⟩ ∧ earlyReturn (vals lam4) 0 0 ⟨17179869184, 21474836480, 0⟩ := by
  refine ⟨Or.inl ⟨Or.inl rfl, rfl⟩, Or.inr ⟨by norm_num, by decide⟩⟩

/-- `distWords` splits on the active set at the witness: one entry below the level, one at or
above it. -/
theorem witness_active_set :
    distWords (vals lam4) 33554432 =
      ((vals lam4).map (fun l => if l < 33554432 then l else 0)).sum +
        ((vals lam4).map (fun l => if l < 33554432 then 0 else 33554432)).sum :=
  distWords_active_set (vals lam4) (by norm_num)

/-! ## 5 · The distortion loop refinement

`distortion_words` (`basis.rs:918`) has one loop, `distortion_words_loop`, over the `Slice` iterator.
`distortion_words_loop_spec` refines it by `loop.spec_decr_nat` to the exact integer fold
`Σ min(λ, θ)` (`distList`, and `distWords` at a positive level by `distList_eq_distWords`), with the
running-sum window `[0, 2¹²⁶]` stated as a hypothesis so the `i128` accumulator is total.
`distortion_words_spec` carries that to the named body's `Result`, through `narrow` and `ok_or`.

The loop invariant is the remaining-suffix equation `acc + D(rest) = acc₀ + D(rest₀)`; the terminating
step uses `List.drop_eq_getElem_cons`. The body's `if l < θ then l else θ` is `min` (`ite_min`). The
first step of the loop is the extracted `IteratorSliceIter.next` (`slice_next_eq`), which the model
defines by `if i < s.len`; no separate modelling of the iterator is introduced.

The window `2¹²⁶` is a hypothesis, not a derived bound: `distWords_le_card_mul` gives
`D ≤ n·θ`, so the caller supplies the product bound. An `i128` overflow therefore does not occur
under the hypotheses, and `spec` (total correctness) reports it: the theorem has no failure branch.
-/

/-- The integer sum of `min l θ` over a word list. -/
def distList (theta : ℤ) (L : List Std.I64) : ℤ := (L.map (fun w => min w.val theta)).sum

theorem distList_nil (theta : ℤ) : distList theta [] = 0 := rfl

theorem distList_cons (theta : ℤ) (w : Std.I64) (L : List Std.I64) :
    distList theta (w :: L) = min w.val theta + distList theta L := rfl

/-- One `next` step of the slice iterator. -/
theorem slice_next_eq (it : core.slice.iter.Iter Std.I64) :
    core.slice.iter.IteratorSliceIter.next it =
      if h : it.i < it.slice.val.length then
        ok (some it.slice.val[it.i], { it with i := it.i + 1 })
      else ok (none, it) := by
  unfold core.slice.iter.IteratorSliceIter.next
  by_cases h : it.i < it.slice.val.length
  · have h' : it.i < it.slice.len := by simpa using h
    rw [dif_pos h', dif_pos h]
    rfl
  · have h' : ¬ it.i < it.slice.len := by simpa using h
    rw [dif_neg h', dif_neg h]

/-- `distList` is `distWords` on the value list, at a positive level. -/
theorem distList_eq_distWords (theta : ℤ) (h : 0 < theta) (L : List Std.I64) :
    distList theta L = distWords (vals L) theta := by
  unfold distList distWords vals
  rw [if_neg (not_le.mpr h)]
  rw [List.map_map]
  rfl

/-- `if l < θ then l else θ` is `min l θ` over the integers. -/
theorem ite_min (l theta : ℤ) : (if l < theta then l else theta) = min l theta := by
  by_cases h : l < theta
  · rw [if_pos h, min_eq_left (le_of_lt h)]
  · rw [if_neg h, min_eq_right (by omega)]

/-- The loop body at an index inside the slice: it emits the current element and advances. -/
theorem body_eq_cont (theta : Std.I64) (it : core.slice.iter.Iter Std.I64) (a : Std.I128)
    (hlt : it.i < it.slice.val.length) :
    fold_port.stats.basis.distortion_words_loop.body theta it a =
      (do
        let l := it.slice.val[it.i]
        let l1 ← if l < theta then ok l else ok theta
        let i ← lift (core.convert.num.FromI128I64.from l1)
        let acc1 ← a + i
        ok (ControlFlow.cont ({ it with i := it.i + 1 }, acc1))) := by
  unfold fold_port.stats.basis.distortion_words_loop.body
  rw [slice_next_eq, dif_pos hlt]
  simp only [bind_tc_ok]
  rfl

/-- The loop body at the end of the slice: it finishes with the accumulator. -/
theorem body_eq_done (theta : Std.I64) (it : core.slice.iter.Iter Std.I64) (a : Std.I128)
    (hge : it.slice.val.length ≤ it.i) :
    fold_port.stats.basis.distortion_words_loop.body theta it a
      = ok (ControlFlow.done a) := by
  unfold fold_port.stats.basis.distortion_words_loop.body
  rw [slice_next_eq, dif_neg (by omega)]
  simp only [bind_tc_ok]
  rfl

/-- The raw fold is nonnegative at a nonnegative level and nonnegative words. -/
theorem distList_nonneg (theta : ℤ) (L : List Std.I64) (hθ : 0 ≤ theta)
    (hL : ∀ l ∈ L, 0 ≤ l.val) : 0 ≤ distList theta L := by
  induction L with
  | nil => simp [distList]
  | cons w ws ih =>
    rw [distList_cons]
    have hw : 0 ≤ w.val := hL w (by simp)
    have hws : 0 ≤ distList theta ws := ih (fun l hl => hL l (by simp [hl]))
    have hm : 0 ≤ min w.val theta := le_min hw hθ
    omega

/-- The extracted `distortion_words_loop` sums `min l θ` over the iterator's remaining suffix.
The `i128` accumulator is total on the stated window: the running sum stays in `[0, 2¹²⁶]`. -/
theorem distortion_words_loop_spec (theta : Std.I64) {hθ : 0 < theta.val}
    (s : Slice Std.I64) (i : ℕ) (hi : i ≤ s.val.length) (acc : Std.I128)
    (hL : ∀ l ∈ s.val, 0 ≤ l.val) (hacc : 0 ≤ acc.val)
    (htot : acc.val + distList theta.val (s.val.drop i) ≤ 2 ^ 126) :
    fold_port.stats.basis.distortion_words_loop ⟨s, i⟩ theta acc ⦃ r =>
      r.val = acc.val + distList theta.val (s.val.drop i) ⦄ := by
  unfold fold_port.stats.basis.distortion_words_loop
  apply loop.spec_decr_nat
    (measure := fun st : core.slice.iter.Iter Std.I64 × Std.I128 =>
      st.1.slice.val.length - st.1.i)
    (inv := fun st => st.1.slice = s ∧ st.1.i ≤ s.val.length ∧
      0 ≤ st.2.val ∧
      st.2.val + distList theta.val (s.val.drop st.1.i)
        = acc.val + distList theta.val (s.val.drop i))
    (post := fun r : Std.I128 => r.val = acc.val + distList theta.val (s.val.drop i))
  · rintro ⟨it, a⟩ ⟨hslice, hit, ha0, ha⟩
    dsimp only at hslice hit ha0 ha ⊢
    by_cases hlt : it.i < it.slice.val.length
    · rw [body_eq_cont theta it a hlt]
      subst hslice
      have hx0 : 0 ≤ it.slice.val[it.i].val := hL _ (List.getElem_mem hlt)
      have hcons := List.drop_eq_getElem_cons hlt
      have hrest0 : 0 ≤ distList theta.val (it.slice.val.drop (it.i + 1)) :=
        distList_nonneg _ _ (le_of_lt hθ) (fun l hl => hL l (List.mem_of_mem_drop hl))
      have hsplit : distList theta.val (it.slice.val.drop it.i) =
          min it.slice.val[it.i].val theta.val
            + distList theta.val (it.slice.val.drop (it.i + 1)) := by
        rw [hcons, distList_cons]
      have htot' : a.val + distList theta.val (it.slice.val.drop it.i) ≤ 2 ^ 126 := by
        rw [ha]; exact htot
      have hbound : a.val + min it.slice.val[it.i].val theta.val ≤ 2 ^ 126 := by
        rw [hsplit] at htot'; omega
      have hm0 : 0 ≤ min it.slice.val[it.i].val theta.val := le_min hx0 (le_of_lt hθ)
      by_cases hl : it.slice.val[it.i] < theta
      · simp only [hl, if_true, bind_tc_ok, lift]
        step*
      · simp only [hl, if_false, bind_tc_ok, lift]
        step*
    · rw [body_eq_done theta it a (by omega)]
      subst hslice
      have hlen : it.i = it.slice.val.length := by omega
      have hd : it.slice.val.drop it.i = [] := by rw [hlen]; simp
      rw [hd, distList_nil, add_zero] at ha
      simpa using ha
  · refine ⟨rfl, hi, hacc, ?_⟩
    simp


/-- The `i64` window bounds of the extracted cast. -/
theorem fromI64_MAX_val :
    (core.convert.num.FromI128I64.from core.num.I64.MAX).val = (2 : ℤ) ^ 63 - 1 := by
  rw [dense_fromI64_val]; decide

/-- The `i64` lower bound. -/
theorem fromI64_MIN_val :
    (core.convert.num.FromI128I64.from core.num.I64.MIN).val = -((2 : ℤ) ^ 63) := by
  rw [dense_fromI64_val]; decide

/-- The extracted `distortion_words` body at a positive level, with the iterator and accumulator
explicit. -/
theorem distortion_words_eq (lambda : Slice Std.I64) {theta : Std.I64} (hθ : 0 < theta.val) :
    fold_port.stats.basis.distortion_words lambda theta =
      (do
        let acc ← fold_port.stats.basis.distortion_words_loop ⟨lambda, 0⟩ theta 0#i128
        let o ← solve_port.q32.narrow acc
        core.option.Option.ok_or o fold_port.stats.basis.BasisError.OutOfRange) := by
  unfold fold_port.stats.basis.distortion_words
  have h0 : (0#i64).val = (0 : ℤ) := by simp
  rw [if_neg (by rw [IScalar.le_equiv]; omega)]
  simp only [SharedSlice.Insts.CoreIterTraitsCollectIntoIteratorSharedIter.into_iter,
    bind_tc_ok]

/-- `distortion_words` refines `distWords`: the extracted body returns the `i64` image of the
integer fold `Σ min(λ, θ)` on the positive branch, inside the stated `i64` window. -/
theorem distortion_words_spec (lambda : Slice Std.I64) {theta : Std.I64} (hθ : 0 < theta.val)
    (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (hi64 : distList theta.val lambda.val ≤ 2 ^ 63 - 1) :
    fold_port.stats.basis.distortion_words lambda theta ⦃ r =>
      ∃ v : Std.I64, r = core.result.Result.Ok v
        ∧ v.val = distList theta.val lambda.val ⦄ := by
  have hD0 : 0 ≤ distList theta.val lambda.val := distList_nonneg _ _ (le_of_lt hθ) hL
  have htot : (0 : ℤ) + distList theta.val (lambda.val.drop 0) ≤ 2 ^ 126 := by
    rw [List.drop_zero]; omega
  have hloop := distortion_words_loop_spec (theta := theta) (hθ := hθ) lambda 0 (Nat.zero_le _)
    0#i128 hL (by simp) htot
  rw [distortion_words_eq lambda hθ]
  obtain ⟨acc, hacc, hval⟩ := (WP.spec_equiv_exists _ _).1 hloop
  rw [hacc]
  simp only [bind_tc_ok]
  have haccval : acc.val = distList theta.val lambda.val := by
    rw [hval, List.drop_zero]
    have h0 : (0#i128).val = (0 : ℤ) := by simp
    omega
  have hnotmax : ¬ (acc > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    rw [gt_iff_lt, IScalar.lt_equiv, fromI64_MAX_val]; omega
  have hnotmin : ¬ (acc < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    rw [IScalar.lt_equiv, fromI64_MIN_val]; omega
  refine (WP.spec_equiv_exists _ _).2
    ⟨core.result.Result.Ok (IScalar.cast .I64 acc), ?_, IScalar.cast .I64 acc, rfl, ?_⟩
  · simp only [solve_port.q32.narrow, lift, bind_tc_ok, hnotmax, hnotmin, ↓reduceIte]
    simp
  · rw [dense_cast_i64_val acc ⟨by omega, by omega⟩]
    exact haccval

/-! ## 6 · The rate loop refinement

`rate_at_logs` (`src/learn/fold/stats/basis.rs:1104`) runs one `Range<usize>` loop,
`rate_at_logs_loop`, over the log vector and accumulates `rateTerm (log_k) t = log_k − t` at every
entry above the level, leaving `failed` false. `rate_at_logs_loop_spec` refines the loop body by
`loop.spec_decr_nat` to `rateSum`: the measure is the remaining range length `end − start`, and the
invariant carries `a = acc + (S start − S i)` with the suffix sum `S n = Σ_{k ≥ n} rateTerm (log_k) t`.
`failed` is the body's `OutOfRange` flag; the loop sets it only when `log_k − t` leaves the `i64`
window, which `hno` excludes. The accumulator window `[0, 2¹²⁶]` is a caller hypothesis, as for the
distortion loop.

`rate_at_logs_loop_zero` instantiates the loop at `[0, len)` from a zero accumulator, and
`rate_at_logs_spec` adds the tail: `>>> 1` is floor division by two (`rateWords`) and `narrow` is
the identity on a rate inside `[0, 2⁶³−1]`.

The model is the Gaussian rate–distortion sum; nothing here identifies it with the `E8`/`O8`
codebook rate.
-/

/-- `CloneUsize.clone` is the identity. -/
theorem cloneUsize_eq (x : Std.Usize) : core.clone.CloneUsize.clone x = ok x := rfl

/-- `PartialOrdUsize.lt` is the value comparison: `<` [is smaller than]. -/
theorem partialOrdUsize_lt (a b : Std.Usize) :
    core.cmp.PartialOrdUsize.lt a b = ok (decide (a.val < b.val)) := rfl

/-- The suffix rate sum. -/
def rateSuffix (logs : List Std.I64) (t : ℤ) (n : ℕ) : ℤ :=
  rateSum (vals (logs.drop n)) t

/-- The suffix sum at the end of the list is empty. -/
theorem rateSuffix_length (logs : List Std.I64) (t : ℤ) :
    rateSuffix logs t logs.length = 0 := by
  unfold rateSuffix
  rw [List.drop_length]
  rfl

/-- One step of the suffix sum. -/
theorem rateSuffix_cons (logs : List Std.I64) (t : ℤ) {n : ℕ} (h : n < logs.length) :
    rateSuffix logs t n = rateTerm logs[n].val t + rateSuffix logs t (n + 1) := by
  unfold rateSuffix
  rw [List.drop_eq_getElem_cons h]
  rfl

/-- The suffix sum is empty past the end of the list. -/
theorem rateSuffix_nil_of_le (logs : List Std.I64) (t : ℤ) {n : ℕ} (h : logs.length ≤ n) :
    rateSuffix logs t n = 0 := by
  unfold rateSuffix
  rw [List.drop_eq_nil_of_le h]
  rfl

/-- The suffix sum decreases with the start index. -/
theorem rateSuffix_anti (logs : List Std.I64) (t : ℤ) {n m : ℕ} (h : n ≤ m) :
    rateSuffix logs t m ≤ rateSuffix logs t n := by
  induction m, h using Nat.le_induction with
  | base => exact le_rfl
  | succ k hnk ih =>
    refine le_trans ?_ ih
    by_cases hk : k < logs.length
    · rw [rateSuffix_cons logs t hk]
      have h0 := rateTerm_nonneg logs[k].val t
      omega
    · have hge : logs.length ≤ k := by omega
      rw [rateSuffix_nil_of_le logs t (n := k + 1) (by omega),
        rateSuffix_nil_of_le logs t (n := k) hge]

/-- The suffix sum is nonnegative. -/
theorem rateSuffix_nonneg (logs : List Std.I64) (t : ℤ) (n : ℕ) :
    0 ≤ rateSuffix logs t n := by
  unfold rateSuffix
  exact rateSum_nonneg _ t

/-- `rateTerm` at an entry above the level. -/
theorem rateTerm_active (l t : ℤ) (hl : l ≠ LOG_NONE) (ht : t < l) : rateTerm l t = l - t := by
  unfold rateTerm
  rw [if_pos ⟨hl, ht⟩]

/-- `rateTerm` at an entry not above the level. -/
theorem rateTerm_zero (l t : ℤ) (ht : ¬ t < l) : rateTerm l t = 0 := by
  unfold rateTerm
  rw [if_neg]
  omega

/-- `Slice.index_usize` at an index inside the slice. -/
theorem index_usize_eq {α : Type} (v : Slice α) (i : Std.Usize) (h : i.val < v.val.length) :
    Slice.index_usize v i = ok v.val[i.val] := by
  obtain ⟨x, hx, hpost⟩ := (WP.spec_equiv_exists _ _).1 (Slice.index_usize_spec v i h)
  rw [hx, hpost]

/-- A `checked_sub` whose value is inside the `i64` window returns `some` with that value. -/
theorem checked_sub_some_of_window (l t : Std.I64)
    (h0 : 0 ≤ l.val - t.val) (hmax : l.val - t.val ≤ 2 ^ 63 - 1) :
    ∃ d : Std.I64, core.num.checked_sub_IScalar l t = some d ∧ d.val = l.val - t.val := by
  have hspec := I64.checked_sub_bv_spec l t
  match h : core.num.checked_sub_IScalar l t with
  | some d =>
    refine ⟨d, rfl, ?_⟩
    rw [h] at hspec
    exact hspec.2.2.1
  | none =>
    exfalso
    rw [h] at hspec
    rw [I64.min_eq, I64.max_eq] at hspec
    omega

/-- The loop body with `failed = false` at a live index. -/
theorem rate_body_eq (logs : Slice Std.I64) (t : Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (acc : Std.I128)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.rate_at_logs_loop.body logs t it acc false =
      (do
        let l ← Slice.index_usize logs it.start
        if l != core.num.I64.MIN then
          if l > t then
            let o1 ← lift (core.num.checked_sub_IScalar l t)
            match o1 with
            | none => ok (ControlFlow.cont (it1, acc, true))
            | some d =>
              let i ← lift (core.convert.num.FromI128I64.from d)
              let acc1 ← acc + i
              ok (ControlFlow.cont (it1, acc1, false))
          else ok (ControlFlow.cont (it1, acc, false))
        else ok (ControlFlow.cont (it1, acc, false))) := by
  unfold fold_port.stats.basis.rate_at_logs_loop.body
  rw [hnext]
  simp only [bind_tc_ok]
  rfl

/-- The loop body at the end of the range. -/
theorem rate_body_eq_done (logs : Slice Std.I64) (t : Std.I64)
    (it : core.ops.range.Range Std.Usize) (acc : Std.I128)
    (hge : it.end.val ≤ it.start.val) :
    fold_port.stats.basis.rate_at_logs_loop.body logs t it acc false =
      ok (ControlFlow.done (acc, false)) := by
  unfold fold_port.stats.basis.rate_at_logs_loop.body
  obtain ⟨p, hp, hopt, -⟩ := (WP.spec_equiv_exists _ _).1
    (core.iter.range.IteratorRange.next_UScalar_none_spec (ty := .Usize)
      (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
      partialOrdUsize_lt it hge)
  obtain ⟨o, it1⟩ := p
  change o = none at hopt
  rw [hopt] at hp
  rw [hp]
  simp only [bind_tc_ok]
  rfl

/-- The `i128` accumulator add of a widened `i64` term. -/
theorem i128_add_from (a : Std.I128) (d : Std.I64) (ha0 : 0 ≤ a.val) (hd0 : 0 ≤ d.val)
    (h : a.val + d.val ≤ 2 ^ 126) :
    ∃ acc1 : Std.I128, a + core.convert.num.FromI128I64.from d = ok acc1 ∧
      acc1.val = a.val + d.val := by
  have hmin : I128.min ≤ a.val + (core.convert.num.FromI128I64.from d).val := by
    rw [core.convert.num.FromI128I64.from_val_eq, I128.min_eq]
    omega
  have hmax : a.val + (core.convert.num.FromI128I64.from d).val ≤ I128.max := by
    rw [core.convert.num.FromI128I64.from_val_eq, I128.max_eq]
    omega
  obtain ⟨x, hx, hpost⟩ := (WP.spec_equiv_exists _ _).1 (I128.add_spec hmin hmax)
  refine ⟨x, hx, ?_⟩
  rw [core.convert.num.FromI128I64.from_val_eq] at hpost
  exact hpost

/-- `rate_at_logs_loop` refines `rateSum` over the range `[start, end)`: the accumulator gains
`rateTerm (log_k) t` at every index and `failed` stays `false` on the stated window. -/
theorem rate_at_logs_loop_spec (logs : Slice Std.I64) (t : Std.I64)
    (r : core.ops.range.Range Std.Usize) (acc : Std.I128)
    (hend : r.end.val ≤ logs.val.length) (hstart : r.start.val ≤ r.end.val)
    (hno : ∀ (k : ℕ) (hk : k < r.end.val), t.val < (logs.val[k]'(by omega)).val →
      (logs.val[k]'(by omega)).val - t.val ≤ 2 ^ 63 - 1)
    (hacc : 0 ≤ acc.val)
    (htot : acc.val + (rateSuffix logs.val t.val r.start.val
      - rateSuffix logs.val t.val r.end.val) ≤ 2 ^ 126) :
    fold_port.stats.basis.rate_at_logs_loop r logs t acc false ⦃ x =>
      x.2 = false ∧
      x.1.val = acc.val + (rateSuffix logs.val t.val r.start.val
        - rateSuffix logs.val t.val r.end.val) ⦄ := by
  unfold fold_port.stats.basis.rate_at_logs_loop
  apply loop.spec_decr_nat
    (measure := fun st : core.ops.range.Range Std.Usize × Std.I128 × Bool =>
      st.1.end.val - st.1.start.val)
    (inv := fun st => st.1.end = r.end ∧ st.1.start.val ≤ r.end.val ∧ st.2.2 = false ∧
      0 ≤ st.2.1.val ∧
      st.2.1.val = acc.val + (rateSuffix logs.val t.val r.start.val
        - rateSuffix logs.val t.val st.1.start.val))
    (post := fun x : Std.I128 × Bool => x.2 = false ∧
      x.1.val = acc.val + (rateSuffix logs.val t.val r.start.val
        - rateSuffix logs.val t.val r.end.val))
  · rintro ⟨it, a, f⟩ ⟨hend', hit, hf, ha0, ha⟩
    dsimp only at hend' hit hf ha0 ha ⊢
    rw [hf]
    by_cases hlt : it.start.val < it.end.val
    · obtain ⟨p, hp, hopt, hstart1, hend1⟩ := (WP.spec_equiv_exists _ _).1
        (core.iter.range.IteratorRange.next_UScalar_some_spec (ty := .Usize)
          (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
          cloneUsize_eq partialOrdUsize_lt it hlt)
      obtain ⟨o, it1⟩ := p
      change o = some it.start at hopt
      change it1.start.val = it.start.val + 1 at hstart1
      change it1.end = it.end at hend1
      rw [hopt] at hp
      rw [rate_body_eq logs t it it1 a hp]
      have hlen : it.start.val < logs.val.length :=
        lt_of_lt_of_le hlt (by rw [hend']; exact hend)
      rw [index_usize_eq logs it.start hlen]
      simp only [bind_tc_ok]
      have hsplit := rateSuffix_cons logs.val t.val hlen
      have hkend : it.start.val < r.end.val := by
        rw [← hend']; exact hlt
      by_cases hbne : (logs.val[it.start.val] != core.num.I64.MIN) = true
      · by_cases hgtP : logs.val[it.start.val] > t
        · have hgt : t.val < logs.val[it.start.val].val := by
            rw [gt_iff_lt, IScalar.lt_equiv] at hgtP
            exact hgtP
          have hmax : logs.val[it.start.val].val - t.val ≤ 2 ^ 63 - 1 :=
            hno it.start.val hkend hgt
          obtain ⟨d, hsub, hd⟩ := checked_sub_some_of_window _ _ (by omega) hmax
          have hlne : logs.val[it.start.val].val ≠ LOG_NONE := by
            have hb := t.hBounds
            simp only [IScalarTy.I64_numBits_eq] at hb
            norm_num at hb
            unfold LOG_NONE
            omega
          have hrt : rateTerm logs.val[it.start.val].val t.val
              = logs.val[it.start.val].val - t.val := rateTerm_active _ _ hlne hgt
          have hmono : rateSuffix logs.val t.val r.end.val
              ≤ rateSuffix logs.val t.val (it.start.val + 1) :=
            rateSuffix_anti logs.val t.val (by omega)
          have heq : logs.val[it.start.val].val - t.val
              = rateSuffix logs.val t.val it.start.val
                - rateSuffix logs.val t.val (it.start.val + 1) := by
            rw [hsplit, hrt]
            ring
          have hbudget : a.val + (logs.val[it.start.val].val - t.val) ≤ 2 ^ 126 := by
            rw [heq, ha]
            omega
          have hbud : a.val + d.val ≤ 2 ^ 126 := by
            rw [hd]
            exact hbudget
          have hd0 : 0 ≤ d.val := by rw [hd]; omega
          obtain ⟨acc1, hadd, hacc1⟩ := i128_add_from a d ha0 hd0 hbud
          simp only [hbne, hgtP, hsub, if_true, bind_tc_ok, lift, hadd, WP.spec_ok]
          repeat' apply And.intro
          · rw [hend1]; exact hend'
          · rw [hstart1]; omega
          · trivial
          · rw [hacc1, hd]; omega
          · rw [hacc1, hd, hstart1, heq, ha]
            ring
          · rw [hstart1, hend1]; omega
        · have hgt0 : ¬ t.val < logs.val[it.start.val].val := by
            intro hc
            exact hgtP (by rw [gt_iff_lt, IScalar.lt_equiv]; exact hc)
          have hrt : rateTerm logs.val[it.start.val].val t.val = 0 := rateTerm_zero _ _ hgt0
          simp only [hbne, hgtP, if_true, if_false, bind_tc_ok, lift, WP.spec_ok]
          repeat' apply And.intro
          · rw [hend1]; exact hend'
          · rw [hstart1]; omega
          · trivial
          · exact ha0
          · have hS : rateSuffix logs.val t.val it.start.val
                = rateSuffix logs.val t.val (it.start.val + 1) := by
              rw [hsplit, hrt]
              ring
            rw [ha, hstart1, hS]
          · rw [hstart1, hend1]; omega
      · have hbne_f : (logs.val[it.start.val] != core.num.I64.MIN) = false := by
          cases h : (logs.val[it.start.val] != core.num.I64.MIN) <;> simp_all
        have hlmin : logs.val[it.start.val].val = LOG_NONE := by
          have h : logs.val[it.start.val] = core.num.I64.MIN := by
            by_contra hc
            rw [bne_iff_ne.mpr hc] at hbne_f
            exact Bool.noConfusion hbne_f
          rw [h]
          rfl
        have hrt : rateTerm logs.val[it.start.val].val t.val = 0 := by
          rw [hlmin]
          exact rateTerm_zero _ _ (by
            have hb := t.hBounds
            simp only [IScalarTy.I64_numBits_eq] at hb
            norm_num at hb
            unfold LOG_NONE
            omega)
        simp only [hbne_f, Bool.false_eq_true, if_false, bind_tc_ok, lift, WP.spec_ok]
        repeat' apply And.intro
        · rw [hend1]; exact hend'
        · rw [hstart1]; omega
        · trivial
        · exact ha0
        · have hS : rateSuffix logs.val t.val it.start.val
              = rateSuffix logs.val t.val (it.start.val + 1) := by
            rw [hsplit, hrt]
            ring
          rw [ha, hstart1, hS]
        · rw [hstart1, hend1]; omega
    · rw [rate_body_eq_done logs t it a (by omega)]
      have hendv : it.end.val = r.end.val := by rw [hend']
      have hstart_eq : it.start.val = r.end.val := by omega
      rw [hstart_eq] at ha
      simp only [WP.spec_ok]
      exact ⟨trivial, ha⟩
  · refine ⟨rfl, hstart, rfl, hacc, ?_⟩
    rw [sub_self]
    ring

/-- The extracted `rate_at_logs` body with the range end and the `>>> 1` tail explicit. -/
theorem rate_at_logs_eq (logs : Slice Std.I64) (t : Std.I64) :
    fold_port.stats.basis.rate_at_logs logs t =
      (do
        let (acc, failed) ←
          fold_port.stats.basis.rate_at_logs_loop
            { start := 0#usize, «end» := Slice.len logs } logs t 0#i128 false
        if failed
        then ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange)
        else
          let i1 ← acc >>> 1#i32
          let o ← solve_port.q32.narrow i1
          core.option.Option.ok_or o fold_port.stats.basis.BasisError.OutOfRange) := rfl

/-- The rate loop on the whole vector from a zero accumulator returns `rateSum`. -/
theorem rate_at_logs_loop_zero (logs : Slice Std.I64) (t : Std.I64)
    (hno : ∀ (k : ℕ) (hk : k < logs.val.length), t.val < (logs.val[k]'(by omega)).val →
      (logs.val[k]'(by omega)).val - t.val ≤ 2 ^ 63 - 1)
    (hbound : rateSum (vals logs.val) t.val ≤ 2 ^ 126) :
    fold_port.stats.basis.rate_at_logs_loop { start := 0#usize, «end» := Slice.len logs }
      logs t 0#i128 false ⦃ x => x.2 = false ∧ x.1.val = rateSum (vals logs.val) t.val ⦄ := by
  have hlen : (Slice.len logs).val = logs.val.length := Slice.len_val logs
  have h0 : (0#i128).val = (0 : ℤ) := by simp
  have h0u : (0#usize).val = (0 : ℕ) := by simp
  have hS0 : rateSuffix logs.val t.val 0 = rateSum (vals logs.val) t.val := by
    unfold rateSuffix
    rw [List.drop_zero]
  have hSe : rateSuffix logs.val t.val (Slice.len logs).val = 0 := by
    rw [hlen]
    exact rateSuffix_length _ _
  have htot : (0#i128).val + (rateSuffix logs.val t.val (0#usize).val
      - rateSuffix logs.val t.val (Slice.len logs).val) ≤ 2 ^ 126 := by
    rw [h0, h0u, hS0, hSe, sub_zero, zero_add]
    exact hbound
  have h := rate_at_logs_loop_spec logs t { start := 0#usize, «end» := Slice.len logs } 0#i128
    (by simp) (by simp)
    (fun k hk => hno k (by rw [hlen] at hk; exact hk)) (by simp) htot
  obtain ⟨x, hx, hf, hv⟩ := (WP.spec_equiv_exists _ _).1 h
  refine (WP.spec_equiv_exists _ _).2 ⟨x, hx, hf, ?_⟩
  rw [hv, h0, h0u, hS0, hSe, sub_zero, zero_add]

/-- `rate_at_logs` refines `rateWords`: the extracted body returns the `i64` image of the floor
half-sum of the active terms, for a nonnegative accumulator inside the `i64` window. The `>>> 1`
step is floor division by two and `narrow` is the identity inside that window. -/
theorem rate_at_logs_spec (logs : Slice Std.I64) (t : Std.I64)
    (hno : ∀ (k : ℕ) (hk : k < logs.val.length), t.val < (logs.val[k]'(by omega)).val →
      (logs.val[k]'(by omega)).val - t.val ≤ 2 ^ 63 - 1)
    (hbound : rateSum (vals logs.val) t.val ≤ 2 ^ 126)
    (hi64 : rateWords (vals logs.val) t.val ≤ 2 ^ 63 - 1) :
    fold_port.stats.basis.rate_at_logs logs t ⦃ r =>
      ∃ v : Std.I64, r = core.result.Result.Ok v
        ∧ v.val = rateWords (vals logs.val) t.val ⦄ := by
  rw [rate_at_logs_eq]
  obtain ⟨xy, hxy, hpost⟩ :=
    (WP.spec_equiv_exists _ _).1 (rate_at_logs_loop_zero logs t hno hbound)
  obtain ⟨acc, failed⟩ := xy
  obtain ⟨hf, hval⟩ := hpost
  rw [hxy]
  simp only [bind_tc_ok]
  subst hf
  show (do
    let i1 ← acc >>> 1#i32
    let o ← solve_port.q32.narrow i1
    core.option.Option.ok_or o fold_port.stats.basis.BasisError.OutOfRange) ⦃ r =>
      ∃ v : Std.I64, r = core.result.Result.Ok v
        ∧ v.val = rateWords (vals logs.val) t.val ⦄
  obtain ⟨s, hsh, hsval, -⟩ := (WP.spec_equiv_exists _ _).1
    (IScalar.ShiftRight_IScalar_spec (ty0 := .I128) (ty1 := .I32) acc (1#i32)
      (by decide) (by decide))
  rw [hsh]
  simp only [bind_tc_ok]
  have hs : s.val = rateWords (vals logs.val) t.val := by
    rw [hsval]
    have h1 : (1#i32).toNat = 1 := by simp
    rw [h1, Int.shiftRight_eq_div_pow]
    norm_num
    unfold rateWords
    rw [hval]
  have hs0 : 0 ≤ s.val := by rw [hs]; exact rateWords_nonneg _ _
  have hnotmax : ¬ (s > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    rw [gt_iff_lt, IScalar.lt_equiv, fromI64_MAX_val]
    omega
  have hnotmin : ¬ (s < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    rw [IScalar.lt_equiv, fromI64_MIN_val]
    omega
  refine (WP.spec_equiv_exists _ _).2
    ⟨core.result.Result.Ok (IScalar.cast .I64 s), ?_, IScalar.cast .I64 s, rfl, ?_⟩
  · simp only [solve_port.q32.narrow, lift, bind_tc_ok, hnotmax, hnotmin, ↓reduceIte]
    simp
  · rw [dense_cast_i64_val s ⟨by omega, by omega⟩]
    exact hs

/-! ## 7 · The bisection's sum loop refinement

`water_fill_words` (`basis.rs:965`) sums the word list into `sum_lambda` at `basis.rs:1000`; the
extracted loop is `fold_port.stats.basis.water_fill_words_loop1`, a `Range<usize>` loop like
`rate_at_logs_loop`. `water_fill_words_loop1_spec` refines it by `loop.spec_decr_nat` to the prefix
sum `Σ λ` over `[start, end)`, on the same `i128` window `[0, 2¹²⁶]` the distortion loop uses and
under the nonnegative-word hypothesis (`hL`, as `distortion_words_spec`). The analogues of the
`rate_at_logs_spec` caller hypotheses are stated explicitly:

* `hstart` / `hend` — the range lies inside the slice;
* `hL` — the words are nonnegative, so both the prefix sums and every widened `i64` term are;
* `hacc` / `htot` — the accumulator starts nonnegative and `acc + Σλ` stays at most `2¹²⁶`, the
  `hbound` analogue;
* `hmax` — the body's `max > 0` guard, needed to read `Σ λ` as `distWords` at the maximum level.

The `hi64` analogue is **not** a hypothesis of this loop: the loop returns the `i128` accumulator,
and the `narrow` that turns it into the returned `distortion` word is a later step of
`water_fill_words`, so its `i64` window remains an obligation on that step.

The loop is `water_fill_words_loop1`, not the bisection: `water_fill_words_loop2` and
`water_fill_words_loop3` (the two search loops, which call the already-refined `rate_at_logs`) and
the log pre-pass `water_fill_words_loop0` are **not** refined, and nothing here proves
`water_fill_words` or `solve_water_fill`.
-/

/-- The suffix sum of the widened word values, the analogue of `rateSuffix`. -/
def lamSuffix (L : List Std.I64) (n : ℕ) : ℤ := (vals (L.drop n)).sum

/-- The widened word sum of a nonnegative list is nonnegative. -/
theorem vals_sum_nonneg (L : List Std.I64) (hL : ∀ l ∈ L, 0 ≤ l.val) :
    0 ≤ (vals L).sum := by
  unfold vals
  induction L with
  | nil => simp
  | cons w ws ih =>
    rw [List.map_cons]
    show 0 ≤ w.val + (ws.map (fun w => w.val)).sum
    have hw : 0 ≤ w.val := hL w (by simp)
    have hws : 0 ≤ (ws.map (fun w => w.val)).sum := ih (fun l hl => hL l (by simp [hl]))
    omega

/-- The suffix sum is empty at the end of the list. -/
theorem lamSuffix_length (L : List Std.I64) :
    lamSuffix L L.length = 0 := by
  unfold lamSuffix
  rw [List.drop_length]
  rfl

/-- One step of the suffix sum. -/
theorem lamSuffix_cons (L : List Std.I64) {n : ℕ} (h : n < L.length) :
    lamSuffix L n = L[n].val + lamSuffix L (n + 1) := by
  unfold lamSuffix
  rw [List.drop_eq_getElem_cons h]
  rfl

/-- The suffix sum is empty past the end of the list. -/
theorem lamSuffix_nil_of_le (L : List Std.I64) {n : ℕ} (h : L.length ≤ n) :
    lamSuffix L n = 0 := by
  unfold lamSuffix
  rw [List.drop_eq_nil_of_le h]
  rfl

/-- The suffix sum is nonnegative on a nonnegative list. -/
theorem lamSuffix_nonneg (L : List Std.I64) (hL : ∀ l ∈ L, 0 ≤ l.val) (n : ℕ) :
    0 ≤ lamSuffix L n := by
  unfold lamSuffix
  exact vals_sum_nonneg _ (fun l hl => hL l (List.mem_of_mem_drop hl))

/-- The loop body at a live index: it widens the word and adds it to the accumulator. -/
theorem sum_body_eq (lambda : Slice Std.I64) (it it1 : core.ops.range.Range Std.Usize)
    (acc : Std.I128)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.water_fill_words_loop1.body lambda it acc =
      (do
        let i ← Slice.index_usize lambda it.start
        let i1 ← lift (core.convert.num.FromI128I64.from i)
        let acc1 ← acc + i1
        ok (ControlFlow.cont (it1, acc1))) := by
  unfold fold_port.stats.basis.water_fill_words_loop1.body
  rw [hnext]
  simp only [bind_tc_ok]
  rfl

/-- The loop body at the end of the range. -/
theorem sum_body_eq_done (lambda : Slice Std.I64) (it : core.ops.range.Range Std.Usize)
    (acc : Std.I128) (hge : it.end.val ≤ it.start.val) :
    fold_port.stats.basis.water_fill_words_loop1.body lambda it acc
      = ok (ControlFlow.done acc) := by
  unfold fold_port.stats.basis.water_fill_words_loop1.body
  obtain ⟨p, hp, hopt, -⟩ := (WP.spec_equiv_exists _ _).1
    (core.iter.range.IteratorRange.next_UScalar_none_spec (ty := .Usize)
      (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
      partialOrdUsize_lt it hge)
  obtain ⟨o, it1⟩ := p
  change o = none at hopt
  rw [hopt] at hp
  rw [hp]
  simp only [bind_tc_ok]
  rfl

/-- `water_fill_words_loop1` refines the prefix sum over `[start, end)`: the accumulator gains
`λ_k` at every index and stays inside the `i128` window `[0, 2¹²⁶]` under the stated hypotheses. -/
theorem water_fill_words_loop1_spec (lambda : Slice Std.I64)
    (r : core.ops.range.Range Std.Usize) (acc : Std.I128)
    (hend : r.end.val ≤ lambda.val.length) (hstart : r.start.val ≤ r.end.val)
    (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (hacc : 0 ≤ acc.val)
    (htot : acc.val + lamSuffix lambda.val r.start.val ≤ 2 ^ 126) :
    fold_port.stats.basis.water_fill_words_loop1 r lambda acc ⦃ x =>
      x.val = acc.val + (lamSuffix lambda.val r.start.val
        - lamSuffix lambda.val r.end.val) ⦄ := by
  unfold fold_port.stats.basis.water_fill_words_loop1
  apply loop.spec_decr_nat
    (measure := fun st : core.ops.range.Range Std.Usize × Std.I128 =>
      st.1.end.val - st.1.start.val)
    (inv := fun st => st.1.end = r.end ∧ st.1.start.val ≤ r.end.val ∧
      0 ≤ st.2.val ∧
      st.2.val = acc.val + (lamSuffix lambda.val r.start.val
        - lamSuffix lambda.val st.1.start.val))
    (post := fun x : Std.I128 =>
      x.val = acc.val + (lamSuffix lambda.val r.start.val
        - lamSuffix lambda.val r.end.val))
  · rintro ⟨it, a⟩ ⟨hend', hit, ha0, ha⟩
    dsimp only at hend' hit ha0 ha ⊢
    by_cases hlt : it.start.val < it.end.val
    · obtain ⟨p, hp, hopt, hstart1, hend1⟩ := (WP.spec_equiv_exists _ _).1
        (core.iter.range.IteratorRange.next_UScalar_some_spec (ty := .Usize)
          (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
          cloneUsize_eq partialOrdUsize_lt it hlt)
      obtain ⟨o, it1⟩ := p
      change o = some it.start at hopt
      change it1.start.val = it.start.val + 1 at hstart1
      change it1.end = it.end at hend1
      rw [hopt] at hp
      rw [sum_body_eq lambda it it1 a hp]
      have hlen : it.start.val < lambda.val.length :=
        lt_of_lt_of_le hlt (by rw [hend']; exact hend)
      rw [index_usize_eq lambda it.start hlen]
      simp only [bind_tc_ok]
      have hsplit := lamSuffix_cons lambda.val hlen
      have hd0 : 0 ≤ lambda.val[it.start.val].val := hL _ (List.getElem_mem hlen)
      have hS1 : 0 ≤ lamSuffix lambda.val (it.start.val + 1) :=
        lamSuffix_nonneg lambda.val hL _
      have heq : lambda.val[it.start.val].val
          = lamSuffix lambda.val it.start.val
            - lamSuffix lambda.val (it.start.val + 1) := by
        rw [hsplit]; ring
      have hbudget : a.val + lambda.val[it.start.val].val ≤ 2 ^ 126 := by
        rw [heq, ha]
        omega
      obtain ⟨acc1, hadd, hacc1⟩ := i128_add_from a lambda.val[it.start.val] ha0 hd0 hbudget
      simp only [lift, bind_tc_ok, hadd, WP.spec_ok]
      have hendv : it.end.val = r.end.val := by rw [hend']
      repeat' apply And.intro
      · rw [hend1]; exact hend'
      · rw [hstart1]; omega
      · rw [hacc1]; omega
      · rw [hacc1, hstart1, heq, ha]; ring
      · rw [hstart1, hend1]; omega
    · rw [sum_body_eq_done lambda it a (by omega)]
      have hendv : it.end.val = r.end.val := by rw [hend']
      have hstart_eq : it.start.val = r.end.val := by omega
      rw [hstart_eq] at ha
      simp only [WP.spec_ok]
      exact ha
  · refine ⟨rfl, hstart, hacc, ?_⟩
    rw [sub_self]
    ring

/-- The sum loop on the whole slice from a zero accumulator returns `Σ λ`. -/
theorem water_fill_words_loop1_zero (lambda : Slice Std.I64)
    (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (hbound : (vals lambda.val).sum ≤ 2 ^ 126) :
    fold_port.stats.basis.water_fill_words_loop1
      { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 ⦃ x =>
        x.val = (vals lambda.val).sum ⦄ := by
  have hlen : (Slice.len lambda).val = lambda.val.length := Slice.len_val lambda
  have h0 : (0#i128).val = (0 : ℤ) := by simp
  have h0u : (0#usize).val = (0 : ℕ) := by simp
  have hS0 : lamSuffix lambda.val 0 = (vals lambda.val).sum := by
    unfold lamSuffix
    rw [List.drop_zero]
  have hSe : lamSuffix lambda.val (Slice.len lambda).val = 0 := by
    rw [hlen]
    exact lamSuffix_nil_of_le _ (le_refl _)
  have htot : (0#i128).val + lamSuffix lambda.val (0#usize).val ≤ 2 ^ 126 := by
    rw [h0, h0u, hS0]
    omega
  have h := water_fill_words_loop1_spec lambda
    { start := 0#usize, «end» := Slice.len lambda } 0#i128
    (by simp) (by simp) hL (by simp) htot
  obtain ⟨x, hx, hv⟩ := (WP.spec_equiv_exists _ _).1 h
  refine (WP.spec_equiv_exists _ _).2 ⟨x, hx, ?_⟩
  rw [hv, h0, h0u, hS0, hSe, sub_zero, zero_add]

/-- The maximum of a nonempty list bounds every member. -/
theorem le_maxOf (L : List ℤ) {l : ℤ} (hl : l ∈ L) : l ≤ maxOf L := by
  induction L with
  | nil => simp at hl
  | cons w ws ih =>
    show l ≤ max w (maxOf ws)
    rcases List.mem_cons.mp hl with hEq | hmem
    · rw [hEq]; exact le_max_left w (maxOf ws)
    · exact le_trans (ih hmem) (le_max_right w (maxOf ws))

/-- A `min` against a level every member is below is the identity. -/
theorem sum_min_eq (L : List ℤ) {m : ℤ} (hm : ∀ l ∈ L, l ≤ m) :
    (L.map (fun l => min l m)).sum = L.sum := by
  induction L with
  | nil => rfl
  | cons w ws ih =>
    show min w m + (ws.map (fun l => min l m)).sum = w + ws.sum
    rw [min_eq_left (hm w (by simp)), ih (fun l hl => hm l (by simp [hl]))]

/-- `Σ λ` is `distWords` at the maximum level, on the body's `max > 0` guard. -/
theorem distWords_maxOf (L : List ℤ) (h : 0 < maxOf L) :
    distWords L (maxOf L) = L.sum := by
  unfold distWords
  rw [if_neg (by omega : ¬ maxOf L ≤ 0)]
  exact sum_min_eq L (fun l hl => le_maxOf L hl)

/-- The `sum_lambda` loop read as `distWords`: its `i128` value is the distortion of the
`target ≤ 0` early return at the maximum level, before that return's `narrow`. -/
theorem water_fill_words_loop1_distWords (lambda : Slice Std.I64)
    (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (hmax : 0 < maxOf (vals lambda.val))
    (hbound : (vals lambda.val).sum ≤ 2 ^ 126) :
    fold_port.stats.basis.water_fill_words_loop1
      { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 ⦃ x =>
        x.val = distWords (vals lambda.val) (maxOf (vals lambda.val)) ⦄ := by
  obtain ⟨x, hx, hv⟩ := (WP.spec_equiv_exists _ _).1
    (water_fill_words_loop1_zero lambda hL hbound)
  refine (WP.spec_equiv_exists _ _).2 ⟨x, hx, ?_⟩
  rw [hv, distWords_maxOf (vals lambda.val) hmax]

/-! ## 7a · The bisection's doubling walk refinement

`water_fill_words` (`basis.rs:1021`) walks the lower bracket down for `BISECTION_ITERS` steps
(`fold_port.stats.basis.water_fill_words_loop2`): at each live step it probes
`rate_at_logs logs t_lo`; a rate at or above `target` sets `walked` and freezes the level, a rate
below it lowers the level by `gap` and doubles `gap`, and a failed probe or checked operation sets
`err`. After `k` steps the level is `walkLevel` and the gap is `gap0·2ᵏ`.

`water_fill_words_loop2_spec` refines the loop by `loop.spec_decr_nat` to the honest exit: the
returned level is feasible (`rate_at_logs` at it is at least `target`), or the error accumulator is
set, or the walk is exhausted with every probed level below `target`. The exhausted disjunct
carries the closed form of the returned level and the recorded infeasibility of every level it
probed; it does **not** claim feasibility, which an exhausted walk does not establish.
`water_fill_words_loop2_converged` is the caller-facing form: a feasible probe inside the iteration
budget makes an error-free exit feasible. -/

/-- The doubling walk's level after `k` steps: `t₀ − g₀·(2ᵏ − 1)`. -/
def walkLevel (t0 gap0 : ℤ) (k : ℕ) : ℤ := t0 - gap0 * (2 ^ k - 1)

/-- The walk state after `k` steps: the level and the doubling gap. -/
def walkForm (t_lo gap : Std.I64) (t0 gap0 : ℤ) (k : ℕ) : Prop :=
  t_lo.val = walkLevel t0 gap0 k ∧ gap.val = gap0 * 2 ^ k

/-- Every level probed by the walk so far is below `target`. Step `j` probes level `j`, so a walk
that has consumed `steps` steps has probed levels `0 … steps-1`. -/
def walkProbesBelow (logs : alloc.vec.Vec Std.I64) (target : Std.I64) (t0 gap0 : ℤ)
    (steps : ℕ) : Prop :=
  ∀ (j : ℕ), j < steps → ∀ (t : Std.I64), t.val = walkLevel t0 gap0 j →
    ∀ (v : Std.I64), fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) t
      = ok (core.result.Result.Ok v) → v.val < target.val

/-- One step of the walk level: `level k − g₀·2ᵏ = level (k+1)`. -/
theorem walkLevel_succ (t0 gap0 : ℤ) (k : ℕ) :
    walkLevel t0 gap0 k - gap0 * 2 ^ k = walkLevel t0 gap0 (k + 1) := by
  unfold walkLevel
  rw [pow_succ]
  ring

/-- One step of the doubling gap: `g₀·2ᵏ · 2 = g₀·2^(k+1)`. -/
theorem walkGap_succ (gap0 : ℤ) (k : ℕ) : gap0 * 2 ^ k * 2 = gap0 * 2 ^ (k + 1) := by
  rw [pow_succ]
  ring

/-- A `checked_sub` that returns `some` returns the exact difference. -/
theorem checked_sub_some_val (a b c : Std.I64)
    (h : core.num.checked_sub_IScalar a b = some c) : c.val = a.val - b.val := by
  have hspec := I64.checked_sub_bv_spec a b
  rw [h] at hspec
  exact hspec.2.2.1

/-- A `checked_mul` that returns `some` returns the exact product. -/
theorem checked_mul_some_val (a b c : Std.I64)
    (h : core.num.checked_mul_IScalar a b = some c) : c.val = a.val * b.val := by
  have hspec := I64.checked_mul_bv_spec a b
  rw [h] at hspec
  exact hspec.2.2.1

/-- The loop body at the end of the range: the exit returns the level and the error. -/
theorem l2_body_eq_done (target : Std.I64) (logs : alloc.vec.Vec Std.I64)
    (it : core.ops.range.Range Std.Usize) (t_lo gap : Std.I64)
    (err : Option fold_port.stats.basis.BasisError) (walked : Bool)
    (hge : it.end.val ≤ it.start.val) :
    fold_port.stats.basis.water_fill_words_loop2.body target logs it t_lo gap err walked
      = ok (ControlFlow.done (t_lo, err)) := by
  unfold fold_port.stats.basis.water_fill_words_loop2.body
  obtain ⟨p, hp, hopt, -⟩ := (WP.spec_equiv_exists _ _).1
    (core.iter.range.IteratorRange.next_UScalar_none_spec (ty := .Usize)
      (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
      partialOrdUsize_lt it hge)
  obtain ⟨o, it1⟩ := p
  change o = none at hopt
  rw [hopt] at hp
  rw [hp]
  simp only [bind_tc_ok]
  rfl

/-- The loop body at a live step with `walked` set: the level, the gap and the error are frozen. -/
theorem l2_body_eq_walked (target : Std.I64) (logs : alloc.vec.Vec Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (t_lo gap : Std.I64)
    (err : Option fold_port.stats.basis.BasisError)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.water_fill_words_loop2.body target logs it t_lo gap err true
      = ok (ControlFlow.cont (it1, t_lo, gap, err, true)) := by
  unfold fold_port.stats.basis.water_fill_words_loop2.body
  rw [hnext]
  simp only [bind_tc_ok]
  rfl

/-- The loop body at a live step with the error set and `walked` clear: the error is kept. -/
theorem l2_body_eq_err (target : Std.I64) (logs : alloc.vec.Vec Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (t_lo gap : Std.I64)
    (e : fold_port.stats.basis.BasisError)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.water_fill_words_loop2.body target logs it t_lo gap (some e) false
      = ok (ControlFlow.cont (it1, t_lo, gap, some e, false)) := by
  unfold fold_port.stats.basis.water_fill_words_loop2.body
  rw [hnext]
  simp only [bind_tc_ok]
  rfl

/-- The loop body at a live step with `walked` clear and no error: it probes `rate_at_logs`. -/
theorem l2_body_eq_probe (target : Std.I64) (logs : alloc.vec.Vec Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (t_lo gap : Std.I64)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.water_fill_words_loop2.body target logs it t_lo gap none false =
      (do
        let r ← fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) t_lo
        match r with
        | core.result.Result.Ok r1 =>
          if r1 >= target
          then ok (ControlFlow.cont (it1, t_lo, gap, none, true))
          else
            let o1 ← lift (I64.checked_sub t_lo gap)
            let (t_lo1, err1) ←
              match o1 with
              | none => ok (t_lo, some fold_port.stats.basis.BasisError.OutOfRange)
              | some lo => ok (lo, none)
            let o2 ← lift (I64.checked_mul gap 2#i64)
            match o2 with
            | none => ok (ControlFlow.cont (it1, t_lo1, gap,
                some fold_port.stats.basis.BasisError.OutOfRange, false))
            | some g => ok (ControlFlow.cont (it1, t_lo1, g, err1, false))
        | core.result.Result.Err e =>
          ok (ControlFlow.cont (it1, t_lo, gap, some e, false))) := by
  unfold fold_port.stats.basis.water_fill_words_loop2.body
  rw [hnext]
  simp only [bind_tc_ok]
  rfl

/-- `water_fill_words_loop2` refines the doubling walk to its honest exit: the returned level is
feasible, or the error accumulator is set, or the walk is exhausted with every probed level below
`target`. `hprobe` is the `rate_at_logs` applicability at every level the walk can reach (one
`rate_at_logs_spec` instantiation per level); the loop itself needs only that the probe returns
`Ok`. -/
theorem water_fill_words_loop2_spec (r : core.ops.range.Range Std.Usize)
    (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (t0 gap0 : Std.I64)
    (hstart : r.start.val ≤ r.end.val)
    (hprobe : ∀ (k : ℕ), k ≤ r.end.val - r.start.val → ∀ (t : Std.I64),
      t.val = walkLevel t0.val gap0.val k →
      ∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) t
        = ok (core.result.Result.Ok v)) :
    fold_port.stats.basis.water_fill_words_loop2 r target logs t0 gap0 none false ⦃ x =>
      (∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) x.1
          = ok (core.result.Result.Ok v) ∧ target.val ≤ v.val)
      ∨ x.2 ≠ none
      ∨ (x.2 = none ∧ x.1.val = walkLevel t0.val gap0.val (r.end.val - r.start.val) ∧
          walkProbesBelow logs target t0.val gap0.val (r.end.val - r.start.val)) ⦄ := by
  unfold fold_port.stats.basis.water_fill_words_loop2
  apply loop.spec_decr_nat
    (measure := fun st : core.ops.range.Range Std.Usize × Std.I64 × Std.I64
        × Option fold_port.stats.basis.BasisError × Bool => st.1.end.val - st.1.start.val)
    (inv := fun st =>
      st.1.end = r.end ∧ r.start.val ≤ st.1.start.val ∧ st.1.start.val ≤ st.1.end.val ∧
      ((st.2.2.2.2 = true ∧ st.2.2.2.1 = none ∧
          ∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) st.2.1
            = ok (core.result.Result.Ok v) ∧ target.val ≤ v.val)
        ∨ st.2.2.2.1 ≠ none
        ∨ (st.2.2.2.2 = false ∧ st.2.2.2.1 = none ∧
            walkForm st.2.1 st.2.2.1 t0.val gap0.val (st.1.start.val - r.start.val) ∧
            walkProbesBelow logs target t0.val gap0.val (st.1.start.val - r.start.val))))
    (post := fun x : Std.I64 × Option fold_port.stats.basis.BasisError =>
      (∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) x.1
          = ok (core.result.Result.Ok v) ∧ target.val ≤ v.val)
      ∨ x.2 ≠ none
      ∨ (x.2 = none ∧ x.1.val = walkLevel t0.val gap0.val (r.end.val - r.start.val) ∧
          walkProbesBelow logs target t0.val gap0.val (r.end.val - r.start.val)))
  · rintro ⟨it, tl, g, e, w⟩ ⟨hend', hrst, hle, hinv⟩
    dsimp only at hend' hrst hle hinv ⊢
    by_cases hlt : it.start.val < it.end.val
    · obtain ⟨p, hp, hopt, hstart1, hend1⟩ := (WP.spec_equiv_exists _ _).1
        (core.iter.range.IteratorRange.next_UScalar_some_spec (ty := .Usize)
          (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
          cloneUsize_eq partialOrdUsize_lt it hlt)
      obtain ⟨o, it1⟩ := p
      change o = some it.start at hopt
      change it1.start.val = it.start.val + 1 at hstart1
      change it1.end = it.end at hend1
      rw [hopt] at hp
      rcases hinv with ⟨hwt, hwe, hvfeas⟩ | hne | ⟨hwf, hwe, hwfrm, hpbelow⟩
      · -- `walked` set: the level is frozen and feasible
        rw [hwt, hwe]
        rw [l2_body_eq_walked target logs it it1 tl g none hp]
        simp only [WP.spec_ok]
        refine ⟨⟨?_, ?_, ?_, Or.inl ⟨trivial, trivial, hvfeas⟩⟩, ?_⟩
        · rw [hend1]; exact hend'
        · rw [hstart1]; omega
        · rw [hstart1, hend1]; omega
        · rw [hstart1, hend1]; omega
      · -- an error is set: it is kept and the level is frozen
        cases w with
        | true =>
          rw [l2_body_eq_walked target logs it it1 tl g e hp]
          simp only [WP.spec_ok]
          refine ⟨⟨?_, ?_, ?_, Or.inr (Or.inl hne)⟩, ?_⟩
          · rw [hend1]; exact hend'
          · rw [hstart1]; omega
          · rw [hstart1, hend1]; omega
          · rw [hstart1, hend1]; omega
        | false =>
          cases e with
          | none => exact absurd rfl hne
          | some e' =>
            rw [l2_body_eq_err target logs it it1 tl g e' hp]
            simp only [WP.spec_ok]
            refine ⟨⟨?_, ?_, ?_, Or.inr (Or.inl hne)⟩, ?_⟩
            · rw [hend1]; exact hend'
            · rw [hstart1]; omega
            · rw [hstart1, hend1]; omega
            · rw [hstart1, hend1]; omega
      · -- walking: probe the level, then freeze or step down
        rw [hwf, hwe]
        rw [l2_body_eq_probe target logs it it1 tl g hp]
        have hendv : it.end.val = r.end.val := by rw [hend']
        obtain ⟨rp, hrp⟩ := hprobe (it.start.val - r.start.val) (by omega) tl hwfrm.1
        rw [hrp]
        simp only [bind_tc_ok]
        by_cases hfv : rp >= target
        · -- the probe is feasible: `walked` is set
          rw [if_pos hfv]
          simp only [WP.spec_ok]
          refine ⟨⟨?_, ?_, ?_, Or.inl ⟨trivial, trivial, ⟨rp, hrp, hfv⟩⟩⟩, ?_⟩
          · rw [hend1]; exact hend'
          · rw [hstart1]; omega
          · rw [hstart1, hend1]; omega
          · rw [hstart1, hend1]; omega
        · -- the probe is below target: lower the level and double the gap
          rw [if_neg hfv]
          simp only [lift, bind_tc_ok, I64.checked_sub, I64.checked_mul]
          have hks : it1.start.val - r.start.val = (it.start.val - r.start.val) + 1 := by
            rw [hstart1]; omega
          cases hsub : core.num.checked_sub_IScalar tl g with
          | none =>
            cases hmul : core.num.checked_mul_IScalar g 2#i64 with
            | none =>
              simp only [bind_tc_ok]
              refine ⟨⟨?_, ?_, ?_, Or.inr (Or.inl ?_)⟩, ?_⟩
              · rw [hend1]; exact hend'
              · rw [hstart1]; omega
              · rw [hstart1, hend1]; omega
              · simp
              · rw [hstart1, hend1]; omega
            | some g2 =>
              simp only [bind_tc_ok]
              refine ⟨⟨?_, ?_, ?_, Or.inr (Or.inl ?_)⟩, ?_⟩
              · rw [hend1]; exact hend'
              · rw [hstart1]; omega
              · rw [hstart1, hend1]; omega
              · simp
              · rw [hstart1, hend1]; omega
          | some lo =>
            cases hmul : core.num.checked_mul_IScalar g 2#i64 with
            | none =>
              simp only [bind_tc_ok]
              refine ⟨⟨?_, ?_, ?_, Or.inr (Or.inl ?_)⟩, ?_⟩
              · rw [hend1]; exact hend'
              · rw [hstart1]; omega
              · rw [hstart1, hend1]; omega
              · simp
              · rw [hstart1, hend1]; omega
            | some g2 =>
              simp only [bind_tc_ok]
              refine WP.spec.ret ?_
              simp only []
              have h2 : (2#i64).val = 2 := by simp
              have hlv := checked_sub_some_val tl g lo hsub
              have hgv := checked_mul_some_val g 2#i64 g2 hmul
              have hg2 : g2.val = g.val * 2 := by rw [hgv, h2]
              rw [hks]
              refine ⟨⟨?_, ?_, ?_, Or.inr (Or.inr ⟨trivial, trivial, ?_, ?_⟩)⟩, ?_⟩
              · rw [hend1]; exact hend'
              · rw [hstart1]; omega
              · rw [hstart1, hend1]; omega
              · unfold walkForm
                refine ⟨?_, ?_⟩
                · rw [hlv, hwfrm.1, hwfrm.2]
                  exact walkLevel_succ t0.val gap0.val (it.start.val - r.start.val)
                · rw [hg2, hwfrm.2]
                  exact walkGap_succ gap0.val (it.start.val - r.start.val)
              · intro j hj t ht v hrv
                have hjk : j < it.start.val - r.start.val ∨ j = it.start.val - r.start.val := by
                  omega
                rcases hjk with hjk | rfl
                · exact hpbelow j hjk t ht v hrv
                · have hteq : t = tl := IScalar.val_eq_imp t tl (by rw [ht, hwfrm.1])
                  rw [hteq, hrp] at hrv
                  have hveq : rp = v := by simpa using hrv
                  have hlt' : ¬ (target.val ≤ rp.val) := hfv
                  rw [← hveq]
                  omega
              · rw [hstart1, hend1]; omega
    · -- the range is exhausted: the exit state is the invariant, discharged to the post
      have hge : it.end.val ≤ it.start.val := by omega
      rw [l2_body_eq_done target logs it tl g e w hge]
      simp only [WP.spec_ok]
      have hendv : it.end.val = r.end.val := by rw [hend']
      have hkn : it.start.val - r.start.val = r.end.val - r.start.val := by omega
      rcases hinv with ⟨-, -, hvfeas⟩ | hne | ⟨-, hwe, hwfrm, hpbelow⟩
      · exact Or.inl hvfeas
      · exact Or.inr (Or.inl hne)
      · refine Or.inr (Or.inr ⟨hwe, ?_, ?_⟩)
        · rw [hwfrm.1, hkn]
        · rwa [hkn] at hpbelow
  · dsimp only
    refine ⟨rfl, le_rfl, hstart, Or.inr (Or.inr ⟨rfl, rfl, ?_, ?_⟩)⟩
    · simp [walkForm, walkLevel]
    · intro j hj
      omega

/-- The caller-facing form: a feasible probe at a level inside the iteration budget makes an
error-free exit's level feasible. This is what loop 3 needs to read the returned level as a
feasible lower bracket. -/
theorem water_fill_words_loop2_converged (r : core.ops.range.Range Std.Usize)
    (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (t0 gap0 : Std.I64)
    (hstart : r.start.val ≤ r.end.val)
    (hprobe : ∀ (k : ℕ), k ≤ r.end.val - r.start.val → ∀ (t : Std.I64),
      t.val = walkLevel t0.val gap0.val k →
      ∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) t
        = ok (core.result.Result.Ok v))
    (hconv : ∃ (k : ℕ) (t v : Std.I64), k < r.end.val - r.start.val ∧
      t.val = walkLevel t0.val gap0.val k ∧
      fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) t
        = ok (core.result.Result.Ok v) ∧ target.val ≤ v.val) :
    fold_port.stats.basis.water_fill_words_loop2 r target logs t0 gap0 none false ⦃ x =>
      x.2 = none → ∃ v : Std.I64,
        fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) x.1
          = ok (core.result.Result.Ok v) ∧ target.val ≤ v.val ⦄ := by
  obtain ⟨x, hx, hpost⟩ := (WP.spec_equiv_exists _ _).1
    (water_fill_words_loop2_spec r target logs t0 gap0 hstart hprobe)
  refine (WP.spec_equiv_exists _ _).2 ⟨x, hx, ?_⟩
  intro hxe
  rcases hpost with hv | hne | ⟨-, -, hbelow⟩
  · exact hv
  · exact absurd hxe hne
  · obtain ⟨k, t, v, hk, ht, hrate, hle⟩ := hconv
    exact absurd hle (not_le.mpr (hbelow k hk t ht v hrate))

/-! ## 8 · The bisection refinement

`water_fill_words_loop3` (`basis.rs:1049`) is the second loop of the bisection: it carries the
bracket `(t_hi, t_lo)` and the returned level `t`, takes the checked midpoint of the two widened
`i64` ends, probes the rate there, and moves one end to the midpoint. The refinement below covers
the body's four exit shapes and the loop by `loop.spec_decr_nat` on the remaining range width.

* **The midpoint.** On the widened `i128` sum `sm = t_lo + t_hi`, the extracted midpoint is
  `sm >>> 1`, plus `1` when `sm < 0` and `sm` is odd (`shiftRight_one_spec`, `i128_and_one_val`,
  `le_div2_succ`, `div2_succ_le_of_odd`); the three body shapes are `l3_body_eq_probe_pos`,
  `l3_body_eq_probe_neg_even` and `l3_body_eq_probe_odd`. `narrow` then maps the `i128` midpoint
  back to an `i64` inside the widened window (`dense_narrow_spec`).
* **The probe tail.** `l3_tail_run` characterises the `narrow`/`rate_at_logs`/`converged` tail at a
  single midpoint: every exit is a `cont` and either reports convergence at the probed level, or
  carries an error, or keeps a bracket with a rejected level.
* **The invariant.** `BisectBracket logs target t_lo t_hi` (`FeasibleAt` at `t_lo` and
  `t_lo ≤ t_hi`) is preserved by both bracket updates, and every exit of the loop lands in
  `ConvergedAt ∨ error ∨ (exhausted with a rejected level)`. `water_fill_words_loop3_spec` is the
  resulting refinement; it does **not** claim convergence on exhaustion, and it does not prove the
  bracket's initial infeasibility (`t_hi` is not certified infeasible here).

The two applicability hypotheses are the caller's: `hprobe` is `rate_at_logs` returning a value at
every probed level (the same hypothesis shape as `water_fill_words_loop2_spec`), and `hconv` is
`converged` returning a Boolean at every probed value — where its `checked_sub` leaves the `i64`
range the extracted body visits the failure effect and there is no exit to refine. Neither is
derived here, and the whole-body `water_fill_words` composition remains unrefined.
-/

/-- The level `t` is feasible: `rate_at_logs` returns a value at least `target` there. -/
def FeasibleAt (logs : alloc.vec.Vec Std.I64) (target t : Std.I64) : Prop :=
  ∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) t
    = ok (core.result.Result.Ok v) ∧ target.val ≤ v.val

/-- The probe at `t` converges: `converged` returns `true` for the probed value. -/
def ConvergedAt (logs : alloc.vec.Vec Std.I64) (target tol t : Std.I64) : Prop :=
  ∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) t
    = ok (core.result.Result.Ok v) ∧ fold_port.stats.basis.converged v target tol
      = ok (core.result.Result.Ok true)

/-- The probe at `t` is rejected: `converged` returns `false` for the probed value. -/
def RejectedAt (logs : alloc.vec.Vec Std.I64) (target tol t : Std.I64) : Prop :=
  ∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) t
    = ok (core.result.Result.Ok v) ∧ fold_port.stats.basis.converged v target tol
      = ok (core.result.Result.Ok false)

/-- The loop state: the range cursor, the two bracket ends, the returned level, the error
accumulator and the convergence flag. -/
abbrev L3State := core.ops.range.Range Std.Usize × Std.I64 × Std.I64 × Std.I64
  × Option fold_port.stats.basis.BasisError × Bool

/-- `lo` is feasible and does not exceed `hi`. -/
def BisectBracket (logs : alloc.vec.Vec Std.I64) (target : Std.I64) (lo hi : Std.I64) : Prop :=
  FeasibleAt logs target lo ∧ lo.val ≤ hi.val

theorem I64_min_val (x : Std.I64) : -((2 : ℤ) ^ 63) ≤ x.val := by scalar_tac

theorem I64_max_val (x : Std.I64) : x.val ≤ 2 ^ 63 - 1 := by scalar_tac

theorem I128_min_val (x : Std.I128) : -((2 : ℤ) ^ 127) ≤ x.val := by scalar_tac

theorem bmod_small_two_w (w n : ℕ) (hw : 2 ≤ w) :
    ((n % 2 : ℕ) : ℤ).bmod (2 ^ w) = ((n % 2 : ℕ) : ℤ) := by
  have hcast : ((2^w : ℕ) : ℤ) = (2:ℤ)^w := by push_cast; ring
  have hpowZ : (2:ℤ)^w = 2 * (2:ℤ)^(w-1) := by
    rw [← pow_succ']; congr 1; omega
  have hnat : ((2^w : ℕ) : ℤ) = 2 * ((2^(w-1) : ℕ) : ℤ) := by
    rw [hcast, hpowZ]; push_cast; ring
  have hhalf : ((2^w : ℕ) : ℤ) / 2 = ((2^(w-1) : ℕ) : ℤ) := by
    rw [hnat, Int.mul_ediv_cancel_left _ (by norm_num)]
  have hge2 : (2:ℤ) ≤ ((2^(w-1) : ℕ) : ℤ) := by
    have : ((2:ℤ))^(w-1) = ((2^(w-1) : ℕ) : ℤ) := by push_cast; ring
    rw [← this]
    calc (2:ℤ) = 2^1 := by norm_num
      _ ≤ 2^(w-1) := by
          apply pow_le_pow_right₀ <;> norm_num; omega
  have hk0 : (0:ℤ) ≤ ((n % 2 : ℕ) : ℤ) := by positivity
  have hk1 : ((n % 2 : ℕ) : ℤ) ≤ 1 := by
    have h := Nat.mod_lt n (by norm_num : 0 < 2); omega
  apply Int.bmod_eq_of_le
  · rw [hhalf]; omega
  · have hle : ((2^(w-1) : ℕ) : ℤ) ≤ (((2^w : ℕ) : ℤ) + 1) / 2 := by
      apply Int.le_ediv_of_mul_le (by norm_num : (0:ℤ) < 2)
      rw [hnat]; omega
    omega

theorem bmod2_parity (n : ℤ) : (n.bmod (2 ^ 128)) % 2 = n % 2 := by
  have hdvd : (2 : ℤ) ∣ 2 ^ 128 := by norm_num
  have hsplit : n.bmod (2 ^ 128) = n % (2 ^ 128) ∨
      n.bmod (2 ^ 128) = n % (2 ^ 128) - 2 ^ 128 := by
    rw [Int.bmod_eq_emod]
    split_ifs <;> simp
  rcases hsplit with h | h
  · rw [h, Int.emod_emod_of_dvd n hdvd]
  · rw [h]
    have h2 : (2 : ℤ) ^ 128 % 2 = 0 := by norm_num
    omega

/-- The odd/even bit of an `i128`: the extracted `&&&` with `1` is zero exactly on the even words. -/
theorem i128_and_one_val (x : Std.I128) :
    (x &&& 1#i128).val = 0 ↔ x.val % 2 = 0 := by
  have hb1 : (1#i128 : Std.I128).bv = 1#128 := rfl
  have hb : (x &&& 1#i128).bv = x.bv &&& 1#128 := by rw [IScalar.bv_and, hb1]
  have hzv : (x &&& 1#i128).val = ((x.bv.toNat % 2 : ℕ) : ℤ) := by
    rw [show (x &&& 1#i128).val = (x &&& 1#i128).bv.toInt from rfl, hb,
      BitVec.toInt_and]
    have hn1 : (1#128).toNat = 1 := BitVec.toNat_one (by norm_num)
    rw [hn1, Nat.and_one_is_mod, bmod_small_two_w IScalarTy.I128.numBits _ (by decide)]
  have hxv : x.val % 2 = ((x.bv.toNat % 2 : ℕ) : ℤ) := by
    rw [show x.val = x.bv.toInt from rfl, BitVec.toInt_eq_toNat_bmod]
    rw [bmod2_parity]
    omega
  rw [hzv, hxv]

theorem div2_bounds (lo hi : ℤ) (h : lo ≤ hi) : lo ≤ (lo + hi) / 2 ∧ (lo + hi) / 2 ≤ hi := by
  constructor <;> omega

theorem le_div2_succ (lo hi : ℤ) (h : lo ≤ hi) : lo ≤ (lo + hi) / 2 + 1 := by omega

theorem div2_succ_le_of_odd (lo hi : ℤ) (h : lo ≤ hi) (hodd : (lo + hi) % 2 ≠ 0) :
    (lo + hi) / 2 + 1 ≤ hi := by omega

/-- The extracted `>>> 1` on `i128` is floor division by two. -/
theorem shiftRight_one_spec (a : Std.I128) :
    ∃ z : Std.I128, a >>> 1#i32 = ok z ∧ z.val = a.val / 2 := by
  obtain ⟨z, hz, hzv, -⟩ := (WP.spec_equiv_exists _ _).1
    (IScalar.ShiftRight_IScalar_spec (ty0 := .I128) (ty1 := .I32) a (1#i32)
      (by decide) (by decide))
  refine ⟨z, hz, ?_⟩
  rw [hzv]
  have h1 : (1#i32).toNat = 1 := by simp
  rw [h1, Int.shiftRight_eq_div_pow]
  norm_num

/-- An `i128` addition by one inside the upper bound is the exact successor. -/
theorem i128_add_one_spec (a : Std.I128) (hmax : a.val + 1 ≤ I128.max) :
    ∃ z : Std.I128, a + 1#i128 = ok z ∧ z.val = a.val + 1 := by
  have h1 : (1#i128).val = (1 : ℤ) := by simp
  have hmin : I128.min ≤ a.val + (1#i128).val := by
    rw [h1, I128.min_eq]
    have := I128_min_val a
    omega
  have hmax' : a.val + (1#i128).val ≤ I128.max := by rw [h1]; exact hmax
  obtain ⟨z, hz, hzv⟩ := (WP.spec_equiv_exists _ _).1
    (I128.add_spec (x := a) (y := 1#i128) hmin hmax')
  refine ⟨z, hz, ?_⟩
  rw [h1] at hzv
  exact hzv
theorem l3_body_eq_done (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (tol : Std.I64)
    (it : core.ops.range.Range Std.Usize) (t_hi t_lo t : Std.I64)
    (err : Option fold_port.stats.basis.BasisError) (done1 : Bool)
    (hge : it.end.val ≤ it.start.val) :
    fold_port.stats.basis.water_fill_words_loop3.body target logs tol it t_hi t_lo t err done1
      = ok (ControlFlow.done (t, err)) := by
  unfold fold_port.stats.basis.water_fill_words_loop3.body
  obtain ⟨p, hp, hopt, -⟩ := (WP.spec_equiv_exists _ _).1
    (core.iter.range.IteratorRange.next_UScalar_none_spec (ty := .Usize)
      (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
      partialOrdUsize_lt it hge)
  obtain ⟨o, it1⟩ := p
  change o = none at hopt
  rw [hopt] at hp
  rw [hp]
  simp only [bind_tc_ok]
  try dsimp only
  rfl

theorem l3_body_eq_doneFlag (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (tol : Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (t_hi t_lo t : Std.I64)
    (err : Option fold_port.stats.basis.BasisError)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.water_fill_words_loop3.body target logs tol it t_hi t_lo t err true
      = ok (ControlFlow.cont (it1, t_hi, t_lo, t, err, true)) := by
  unfold fold_port.stats.basis.water_fill_words_loop3.body
  rw [hnext]
  simp only [bind_tc_ok]
  try dsimp only
  rfl

theorem l3_body_eq_err (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (tol : Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (t_hi t_lo t : Std.I64)
    (e : fold_port.stats.basis.BasisError)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.water_fill_words_loop3.body target logs tol it t_hi t_lo t (some e) false
      = ok (ControlFlow.cont (it1, t_hi, t_lo, t, some e, false)) := by
  unfold fold_port.stats.basis.water_fill_words_loop3.body
  rw [hnext]
  simp only [bind_tc_ok, core.option.Option.is_none, Option.isNone_some, Bool.false_eq_true,
    if_false]
  rfl

theorem l3_body_eq_probe (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (tol : Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (t_hi t_lo t : Std.I64)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.water_fill_words_loop3.body target logs tol it t_hi t_lo t none false =
      (do
        let i ← lift (core.convert.num.FromI128I64.from t_lo)
        let i1 ← lift (core.convert.num.FromI128I64.from t_hi)
        let sum ← i + i1
        let (iter2, mid) ←
          if sum < 0#i128
          then
            do
            let i2 ← lift (sum &&& 1#i128)
            let i3 ←
              if i2 != 0#i128
              then do
                   let i4 ← sum >>> 1#i32
                   i4 + 1#i128
              else sum >>> 1#i32
            ok (it1, i3)
          else do
               let mid1 ← sum >>> 1#i32
               ok (it1, mid1)
        let o1 ← solve_port.q32.narrow mid
        match o1 with
        | none =>
          ok (ControlFlow.cont (iter2, t_hi, t_lo, t, some
            fold_port.stats.basis.BasisError.OutOfRange, false))
        | some mid1 =>
          let s := alloc.vec.Vec.deref logs
          let r ← fold_port.stats.basis.rate_at_logs s mid1
          match r with
          | core.result.Result.Ok r1 =>
            let r2 ← fold_port.stats.basis.converged r1 target tol
            match r2 with
            | core.result.Result.Ok b1 =>
              if b1
              then ok (ControlFlow.cont (iter2, t_hi, t_lo, mid1, none, true))
              else
                if r1 > target
                then ok (ControlFlow.cont (iter2, t_hi, mid1, mid1, none, false))
                else ok (ControlFlow.cont (iter2, mid1, t_lo, mid1, none, false))
            | core.result.Result.Err e =>
              ok (ControlFlow.cont (iter2, t_hi, t_lo, mid1, some e, false))
          | core.result.Result.Err e =>
            ok (ControlFlow.cont (iter2, t_hi, t_lo, mid1, some e, false))) := by
  unfold fold_port.stats.basis.water_fill_words_loop3.body
  rw [hnext]
  simp only [bind_tc_ok]
  try dsimp only
  rfl


/-- The probe step at a nonnegative sum: the body is the tail at the plain midpoint `sm >>> 1`. -/
theorem l3_body_eq_probe_pos (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (tol : Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (t_hi t_lo t : Std.I64) (sm mid : Std.I128)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1))
    (hsum : core.convert.num.FromI128I64.from t_lo + core.convert.num.FromI128I64.from t_hi
      = ok sm)
    (hneg : ¬ sm < 0#i128) (hq : sm >>> 1#i32 = ok mid) :
    fold_port.stats.basis.water_fill_words_loop3.body target logs tol it t_hi t_lo t none false =
      (do
        let o1 ← solve_port.q32.narrow mid
        match o1 with
        | none =>
          ok (ControlFlow.cont (it1, t_hi, t_lo, t, some
            fold_port.stats.basis.BasisError.OutOfRange, false))
        | some mid1 =>
          let s := alloc.vec.Vec.deref logs
          let r ← fold_port.stats.basis.rate_at_logs s mid1
          match r with
          | core.result.Result.Ok r1 =>
            let r2 ← fold_port.stats.basis.converged r1 target tol
            match r2 with
            | core.result.Result.Ok b1 =>
              if b1
              then ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, none, true))
              else
                if r1 > target
                then ok (ControlFlow.cont (it1, t_hi, mid1, mid1, none, false))
                else ok (ControlFlow.cont (it1, mid1, t_lo, mid1, none, false))
            | core.result.Result.Err e =>
              ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, some e, false))
          | core.result.Result.Err e =>
            ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, some e, false))) := by
  rw [l3_body_eq_probe target logs tol it it1 t_hi t_lo t hnext]
  simp only [lift, bind_tc_ok]
  rw [hsum]
  simp only [bind_tc_ok]
  rw [if_neg hneg]
  rw [hq]
  simp only [bind_tc_ok]
  rfl

/-- The probe step at a negative even sum: the body is the tail at the plain midpoint `sm >>> 1`. -/
theorem l3_body_eq_probe_neg_even (target : Std.I64) (logs : alloc.vec.Vec Std.I64)
    (tol : Std.I64) (it it1 : core.ops.range.Range Std.Usize) (t_hi t_lo t : Std.I64)
    (sm mid : Std.I128)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1))
    (hsum : core.convert.num.FromI128I64.from t_lo + core.convert.num.FromI128I64.from t_hi
      = ok sm)
    (hneg : sm < 0#i128) (hbit : ¬ (((sm &&& 1#i128) != 0#i128) = true))
    (hq : sm >>> 1#i32 = ok mid) :
    fold_port.stats.basis.water_fill_words_loop3.body target logs tol it t_hi t_lo t none false =
      (do
        let o1 ← solve_port.q32.narrow mid
        match o1 with
        | none =>
          ok (ControlFlow.cont (it1, t_hi, t_lo, t, some
            fold_port.stats.basis.BasisError.OutOfRange, false))
        | some mid1 =>
          let s := alloc.vec.Vec.deref logs
          let r ← fold_port.stats.basis.rate_at_logs s mid1
          match r with
          | core.result.Result.Ok r1 =>
            let r2 ← fold_port.stats.basis.converged r1 target tol
            match r2 with
            | core.result.Result.Ok b1 =>
              if b1
              then ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, none, true))
              else
                if r1 > target
                then ok (ControlFlow.cont (it1, t_hi, mid1, mid1, none, false))
                else ok (ControlFlow.cont (it1, mid1, t_lo, mid1, none, false))
            | core.result.Result.Err e =>
              ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, some e, false))
          | core.result.Result.Err e =>
            ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, some e, false))) := by
  rw [l3_body_eq_probe target logs tol it it1 t_hi t_lo t hnext]
  simp only [lift, bind_tc_ok]
  rw [hsum]
  simp only [bind_tc_ok]
  rw [if_pos hneg]
  rw [if_neg hbit]
  rw [hq]
  simp only [bind_tc_ok]
  rfl

/-- The probe step at a negative odd sum: the body is the tail at the rounded midpoint
`(sm >>> 1) + 1`. -/
theorem l3_body_eq_probe_odd (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (tol : Std.I64)
    (it it1 : core.ops.range.Range Std.Usize) (t_hi t_lo t : Std.I64) (sm mid q : Std.I128)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1))
    (hsum : core.convert.num.FromI128I64.from t_lo + core.convert.num.FromI128I64.from t_hi
      = ok sm)
    (hneg : sm < 0#i128) (hbit : ((sm &&& 1#i128) != 0#i128) = true)
    (hq : sm >>> 1#i32 = ok q) (hq1 : q + 1#i128 = ok mid) :
    fold_port.stats.basis.water_fill_words_loop3.body target logs tol it t_hi t_lo t none false =
      (do
        let o1 ← solve_port.q32.narrow mid
        match o1 with
        | none =>
          ok (ControlFlow.cont (it1, t_hi, t_lo, t, some
            fold_port.stats.basis.BasisError.OutOfRange, false))
        | some mid1 =>
          let s := alloc.vec.Vec.deref logs
          let r ← fold_port.stats.basis.rate_at_logs s mid1
          match r with
          | core.result.Result.Ok r1 =>
            let r2 ← fold_port.stats.basis.converged r1 target tol
            match r2 with
            | core.result.Result.Ok b1 =>
              if b1
              then ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, none, true))
              else
                if r1 > target
                then ok (ControlFlow.cont (it1, t_hi, mid1, mid1, none, false))
                else ok (ControlFlow.cont (it1, mid1, t_lo, mid1, none, false))
            | core.result.Result.Err e =>
              ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, some e, false))
          | core.result.Result.Err e =>
            ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, some e, false))) := by
  rw [l3_body_eq_probe target logs tol it it1 t_hi t_lo t hnext]
  simp only [lift, bind_tc_ok]
  rw [hsum]
  simp only [bind_tc_ok]
  rw [if_pos hneg]
  rw [if_pos hbit]
  rw [hq]
  simp only [bind_tc_ok]
  rw [hq1]
  simp only [bind_tc_ok]
  rfl
/-- The probe tail: `narrow`, the probe, the convergence test and the bracket update. Every exit is
a `cont` that leaves `it1` in place; the exits either report convergence, carry an error, or keep a
bracket with a rejected level. -/
theorem l3_tail_run (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (tol : Std.I64)
    (it1 : core.ops.range.Range Std.Usize) (t_hi t_lo t : Std.I64) (mid : Std.I128)
    (m : Std.I64)
    (hnarrow : solve_port.q32.narrow mid = ok (some m))
    (hconv : ∀ (v : Std.I64), ∃ b : Bool,
      fold_port.stats.basis.converged v target tol = ok (core.result.Result.Ok b))
    (hprobe : ∀ (s : Std.I64), ∃ v : Std.I64,
      fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) s
        = ok (core.result.Result.Ok v))
    (hloFeas : FeasibleAt logs target t_lo) (hlo : t_lo.val ≤ m.val) (hhi : m.val ≤ t_hi.val) :
    ∃ st : L3State,
      (do
      let o1 ← solve_port.q32.narrow mid
      match o1 with
      | none =>
        ok (ControlFlow.cont (it1, t_hi, t_lo, t, some
          fold_port.stats.basis.BasisError.OutOfRange, false))
      | some mid1 =>
        let s := alloc.vec.Vec.deref logs
        let r ← fold_port.stats.basis.rate_at_logs s mid1
        match r with
        | core.result.Result.Ok r1 =>
          let r2 ← fold_port.stats.basis.converged r1 target tol
          match r2 with
          | core.result.Result.Ok b1 =>
            if b1
            then ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, none, true))
            else
              if r1 > target
              then ok (ControlFlow.cont (it1, t_hi, mid1, mid1, none, false))
              else ok (ControlFlow.cont (it1, mid1, t_lo, mid1, none, false))
          | core.result.Result.Err e =>
            ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, some e, false))
        | core.result.Result.Err e =>
          ok (ControlFlow.cont (it1, t_hi, t_lo, mid1, some e, false))) = ok (ControlFlow.cont (α := L3State)
        (β := Std.I64 × Option fold_port.stats.basis.BasisError) st) ∧
      st.1 = it1 ∧
      ((st.2.2.2.2.2 = true ∧ st.2.2.2.2.1 = none ∧ ConvergedAt logs target tol st.2.2.2.1)
        ∨ st.2.2.2.2.1 ≠ none
        ∨ (st.2.2.2.2.2 = false ∧ st.2.2.2.2.1 = none ∧
            BisectBracket logs target st.2.2.1 st.2.1 ∧
            RejectedAt logs target tol st.2.2.2.1)) := by
  rw [hnarrow]
  simp only [bind_tc_ok]
  obtain ⟨v, hv⟩ := hprobe m
  rw [hv]
  simp only [bind_tc_ok]
  obtain ⟨b, hb⟩ := hconv v
  rw [hb]
  simp only [bind_tc_ok]
  by_cases hbtrue : b = true
  · rw [if_pos hbtrue]
    rw [hbtrue] at hb
    exact ⟨(it1, t_hi, t_lo, m, none, true), rfl, rfl, Or.inl ⟨rfl, rfl, ⟨v, hv, hb⟩⟩⟩
  · rw [if_neg hbtrue]
    have hbf : b = false := by cases b <;> simp_all
    rw [hbf] at hb
    by_cases hgt : target.val < v.val
    · rw [if_pos (by rw [gt_iff_lt, IScalar.lt_equiv]; exact hgt)]
      exact ⟨(it1, t_hi, m, m, none, false), rfl, rfl,
        Or.inr (Or.inr ⟨rfl, rfl, ⟨⟨v, hv, by omega⟩, by omega⟩, ⟨v, hv, hb⟩⟩)⟩
    · rw [if_neg (by rw [gt_iff_lt, IScalar.lt_equiv]; omega)]
      exact ⟨(it1, m, t_lo, m, none, false), rfl, rfl,
        Or.inr (Or.inr ⟨rfl, rfl, ⟨hloFeas, by omega⟩, ⟨v, hv, hb⟩⟩)⟩

/-- `water_fill_words_loop3` refines the bisection: every exit either converges at the returned
level, or carries an error, or is an exhausted walk whose returned level was rejected. `hprobe` and
`hconv` are the two applicability facts the loop needs: `rate_at_logs` returns a value at every
level, and `converged` returns a Boolean at every probed value. Neither is claimed infeasible-free:
where `converged` walks out of the range the loop has no exit to refine. -/
theorem water_fill_words_loop3_spec (r : core.ops.range.Range Std.Usize)
    (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (t_hi t_lo tol t0 : Std.I64)
    (hstart : r.start.val ≤ r.end.val) (hbr : BisectBracket logs target t_lo t_hi)
    (hprobe : ∀ (s : Std.I64), ∃ v : Std.I64,
      fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) s
        = ok (core.result.Result.Ok v))
    (hconv : ∀ (v : Std.I64), ∃ b : Bool,
      fold_port.stats.basis.converged v target tol = ok (core.result.Result.Ok b)) :
    fold_port.stats.basis.water_fill_words_loop3 r target logs t_hi t_lo tol t0 none false ⦃ x =>
      ConvergedAt logs target tol x.1 ∨ x.2 ≠ none ∨
        (x.2 = none ∧ (RejectedAt logs target tol x.1 ∨
          (r.start.val = r.end.val ∧ x.1 = t0))) ⦄ := by
  unfold fold_port.stats.basis.water_fill_words_loop3
  apply loop.spec_decr_nat
    (measure := fun st : core.ops.range.Range Std.Usize × Std.I64 × Std.I64 × Std.I64
      × Option fold_port.stats.basis.BasisError × Bool => st.1.end.val - st.1.start.val)
    (inv := fun st => st.1.end = r.end ∧ r.start.val ≤ st.1.start.val ∧
      st.1.start.val ≤ st.1.end.val ∧
      ((st.2.2.2.2.2 = true ∧ st.2.2.2.2.1 = none ∧
          ConvergedAt logs target tol st.2.2.2.1)
        ∨ st.2.2.2.2.1 ≠ none
        ∨ (st.2.2.2.2.2 = false ∧ st.2.2.2.2.1 = none ∧
            BisectBracket logs target st.2.2.1 st.2.1 ∧
            (RejectedAt logs target tol st.2.2.2.1 ∨
              (st.1.start.val = r.start.val ∧ st.2.2.2.1 = t0)))))
    (post := fun x : Std.I64 × Option fold_port.stats.basis.BasisError =>
      ConvergedAt logs target tol x.1 ∨ x.2 ≠ none ∨
        (x.2 = none ∧ (RejectedAt logs target tol x.1 ∨
          (r.start.val = r.end.val ∧ x.1 = t0))))
  · rintro ⟨it, th, tl, t, e, dn⟩ ⟨hend', hrst, hle, hinv⟩
    dsimp only at hend' hrst hle hinv ⊢
    by_cases hlt : it.start.val < it.end.val
    · obtain ⟨p, hp, hopt, hstart1, hend1⟩ := (WP.spec_equiv_exists _ _).1
        (core.iter.range.IteratorRange.next_UScalar_some_spec (ty := .Usize)
          (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
          cloneUsize_eq partialOrdUsize_lt it hlt)
      obtain ⟨o, it1⟩ := p
      change o = some it.start at hopt
      change it1.start.val = it.start.val + 1 at hstart1
      change it1.end = it.end at hend1
      rw [hopt] at hp
      rcases hinv with ⟨hwt, hwe, hconvAt⟩ | hne | ⟨hwf, hwe, hbracket, hprov⟩
      · rw [hwt, hwe]
        rw [l3_body_eq_doneFlag target logs tol it it1 th tl t none hp]
        simp only [WP.spec_ok]
        refine ⟨⟨?_, ?_, ?_, Or.inl ⟨trivial, trivial, hconvAt⟩⟩, ?_⟩
        · rw [hend1]; exact hend'
        · rw [hstart1]; omega
        · rw [hstart1, hend1]; omega
        · rw [hstart1, hend1]; omega
      · cases dn with
        | true =>
          rw [l3_body_eq_doneFlag target logs tol it it1 th tl t e hp]
          simp only [WP.spec_ok]
          refine ⟨⟨?_, ?_, ?_, Or.inr (Or.inl hne)⟩, ?_⟩
          · rw [hend1]; exact hend'
          · rw [hstart1]; omega
          · rw [hstart1, hend1]; omega
          · rw [hstart1, hend1]; omega
        | false =>
          cases e with
          | none => exact absurd rfl hne
          | some e' =>
            rw [l3_body_eq_err target logs tol it it1 th tl t e' hp]
            simp only [WP.spec_ok]
            refine ⟨⟨?_, ?_, ?_, Or.inr (Or.inl hne)⟩, ?_⟩
            · rw [hend1]; exact hend'
            · rw [hstart1]; omega
            · rw [hstart1, hend1]; omega
            · rw [hstart1, hend1]; omega
      · rw [hwf, hwe]
        -- The widened sum of the two bracket ends.
        obtain ⟨sm, hsumEq, hsumVal⟩ : ∃ s : Std.I128,
            core.convert.num.FromI128I64.from tl + core.convert.num.FromI128I64.from th = ok s ∧
            s.val = tl.val + th.val := by
          have hmin : I128.min ≤ (core.convert.num.FromI128I64.from tl).val
              + (core.convert.num.FromI128I64.from th).val := by
            rw [core.convert.num.FromI128I64.from_val_eq,
              core.convert.num.FromI128I64.from_val_eq, I128.min_eq]
            have h1 := I64_min_val tl
            have h2 := I64_min_val th
            omega
          have hmax : (core.convert.num.FromI128I64.from tl).val
              + (core.convert.num.FromI128I64.from th).val ≤ I128.max := by
            rw [core.convert.num.FromI128I64.from_val_eq,
              core.convert.num.FromI128I64.from_val_eq, I128.max_eq]
            have h1 := I64_max_val tl
            have h2 := I64_max_val th
            omega
          obtain ⟨s, hs, hpost⟩ := (WP.spec_equiv_exists _ _).1 (I128.add_spec hmin hmax)
          refine ⟨s, hs, ?_⟩
          rw [core.convert.num.FromI128I64.from_val_eq,
            core.convert.num.FromI128I64.from_val_eq] at hpost
          exact hpost
        obtain ⟨hloFeas, hlohi⟩ := hbracket
        by_cases hneg : sm.val < 0
        · obtain ⟨q, hq, hqv⟩ := shiftRight_one_spec sm
          by_cases hbit : ((sm &&& 1#i128) != 0#i128) = true
          · -- the odd-negative midpoint: `(sm >>> 1) + 1`
            obtain ⟨q1, hq1, hq1v⟩ := i128_add_one_spec q (by
              rw [hqv]
              have h1 : sm.val / 2 ≤ 2 ^ 126 := by omega
              have h2 : I128.max = 2 ^ 127 - 1 := by rw [I128.max_eq]; norm_num
              omega)
            have hne0 : (sm &&& 1#i128) ≠ 0#i128 := by
              intro h
              rw [h] at hbit
              simp at hbit
            have hvalne : (sm &&& 1#i128).val ≠ 0 := by
              intro h
              exact hne0 (IScalar.val_eq_imp _ _ (by rw [h]; rfl))
            have hodd : sm.val % 2 ≠ 0 := fun h => hvalne ((i128_and_one_val sm).mpr h)
            have hodd' : (tl.val + th.val) % 2 ≠ 0 := by rw [← hsumVal]; exact hodd
            have hq1lo : tl.val ≤ q1.val := by
              rw [hq1v, hqv, hsumVal]
              exact le_div2_succ tl.val th.val hlohi
            have hq1hi : q1.val ≤ th.val := by
              rw [hq1v, hqv, hsumVal]
              exact div2_succ_le_of_odd tl.val th.val hlohi hodd'
            obtain ⟨m, hm, hmv⟩ := dense_narrow_spec q1
              ⟨by have h := I64_min_val tl; omega,
               by have h := I64_max_val th; omega⟩
            rw [l3_body_eq_probe_odd target logs tol it it1 th tl t sm q1 q hp hsumEq
              (by rw [IScalar.lt_equiv]; exact hneg) hbit hq hq1]
            obtain ⟨st, hst, hit0, hdisj⟩ := l3_tail_run target logs tol it1 th tl t q1 m hm hconv
              hprobe hloFeas (by omega) (by omega)
            rw [hst]
            simp only [WP.spec_ok]
            refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
            · rw [hit0, hend1]; exact hend'
            · rw [hit0, hstart1]; omega
            · rw [hit0, hstart1, hend1]; omega
            · rcases hdisj with ⟨h1, h2, h3⟩ | hne' | ⟨h1, h2, hb, hrej⟩
              · exact Or.inl ⟨h1, h2, h3⟩
              · exact Or.inr (Or.inl hne')
              · exact Or.inr (Or.inr ⟨h1, h2, hb, Or.inl hrej⟩)
            · rw [hit0, hstart1, hend1]; omega
          · -- the negative even midpoint: `sm >>> 1`
            have hqlo : tl.val ≤ q.val := by
              rw [hqv, hsumVal]
              exact (div2_bounds tl.val th.val hlohi).1
            have hqhi : q.val ≤ th.val := by
              rw [hqv, hsumVal]
              exact (div2_bounds tl.val th.val hlohi).2
            obtain ⟨m, hm, hmv⟩ := dense_narrow_spec q
              ⟨by have h := I64_min_val tl; omega,
               by have h := I64_max_val th; omega⟩
            rw [l3_body_eq_probe_neg_even target logs tol it it1 th tl t sm q hp hsumEq
              (by rw [IScalar.lt_equiv]; exact hneg) hbit hq]
            obtain ⟨st, hst, hit0, hdisj⟩ := l3_tail_run target logs tol it1 th tl t q m hm hconv
              hprobe hloFeas (by omega) (by omega)
            rw [hst]
            simp only [WP.spec_ok]
            refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
            · rw [hit0, hend1]; exact hend'
            · rw [hit0, hstart1]; omega
            · rw [hit0, hstart1, hend1]; omega
            · rcases hdisj with ⟨h1, h2, h3⟩ | hne' | ⟨h1, h2, hb, hrej⟩
              · exact Or.inl ⟨h1, h2, h3⟩
              · exact Or.inr (Or.inl hne')
              · exact Or.inr (Or.inr ⟨h1, h2, hb, Or.inl hrej⟩)
            · rw [hit0, hstart1, hend1]; omega
        · -- the nonnegative midpoint: `sm >>> 1`
          obtain ⟨q, hq, hqv⟩ := shiftRight_one_spec sm
          have hqlo : tl.val ≤ q.val := by
            rw [hqv, hsumVal]
            exact (div2_bounds tl.val th.val hlohi).1
          have hqhi : q.val ≤ th.val := by
            rw [hqv, hsumVal]
            exact (div2_bounds tl.val th.val hlohi).2
          obtain ⟨m, hm, hmv⟩ := dense_narrow_spec q
            ⟨by have h := I64_min_val tl; omega,
             by have h := I64_max_val th; omega⟩
          rw [l3_body_eq_probe_pos target logs tol it it1 th tl t sm q hp hsumEq
            (by rw [IScalar.lt_equiv]; exact hneg) hq]
          obtain ⟨st, hst, hit0, hdisj⟩ := l3_tail_run target logs tol it1 th tl t q m hm hconv
            hprobe hloFeas (by omega) (by omega)
          rw [hst]
          simp only [WP.spec_ok]
          refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
          · rw [hit0, hend1]; exact hend'
          · rw [hit0, hstart1]; omega
          · rw [hit0, hstart1, hend1]; omega
          · rcases hdisj with ⟨h1, h2, h3⟩ | hne' | ⟨h1, h2, hb, hrej⟩
            · exact Or.inl ⟨h1, h2, h3⟩
            · exact Or.inr (Or.inl hne')
            · exact Or.inr (Or.inr ⟨h1, h2, hb, Or.inl hrej⟩)
          · rw [hit0, hstart1, hend1]; omega
    · -- the range is exhausted: the exit state is the invariant, discharged to the post
      have hge : it.end.val ≤ it.start.val := by omega
      rw [l3_body_eq_done target logs tol it th tl t e dn hge]
      simp only [WP.spec_ok]
      rcases hinv with ⟨-, -, hconvAt⟩ | hne | ⟨-, hwe, -, hprov⟩
      · exact Or.inl hconvAt
      · exact Or.inr (Or.inl hne)
      · rcases hprov with hrej | ⟨hs, ht⟩
        · exact Or.inr (Or.inr ⟨hwe, Or.inl hrej⟩)
        · refine Or.inr (Or.inr ⟨hwe, Or.inr ⟨?_, ht⟩⟩)
          have h1 : it.start.val = it.end.val := le_antisymm hle hge
          rw [← hs, h1, hend']
  · dsimp only
    refine ⟨rfl, le_rfl, hstart, Or.inr (Or.inr ⟨rfl, rfl, hbr, Or.inr ⟨rfl, rfl⟩⟩)⟩

/-- The caller-facing form: a nonempty range, a feasible lower bracket and no rejected probe make
an error-free exit converge at the returned level. The two exclusions are explicit: the range is
nonempty (`hsne`, so the zero-iteration `t0` exit is unreachable) and every probe is unrejected
(`hexh`). Neither is derived here — an exhausted walk still says nothing about the level it probed
last, and the bracket's upper end is not certified infeasible. -/
theorem water_fill_words_loop3_converged (r : core.ops.range.Range Std.Usize)
    (target : Std.I64) (logs : alloc.vec.Vec Std.I64) (t_hi t_lo tol t0 : Std.I64)
    (hstart : r.start.val ≤ r.end.val) (hsne : r.start.val < r.end.val)
    (hbr : BisectBracket logs target t_lo t_hi)
    (hprobe : ∀ (s : Std.I64), ∃ v : Std.I64,
      fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref logs) s
        = ok (core.result.Result.Ok v))
    (hconv : ∀ (v : Std.I64), ∃ b : Bool,
      fold_port.stats.basis.converged v target tol = ok (core.result.Result.Ok b))
    (hexh : ∀ (s : Std.I64), ¬ RejectedAt logs target tol s) :
    fold_port.stats.basis.water_fill_words_loop3 r target logs t_hi t_lo tol t0 none false ⦃ x =>
      x.2 = none → ConvergedAt logs target tol x.1 ⦄ := by
  obtain ⟨x, hx, hpost⟩ := (WP.spec_equiv_exists _ _).1
    (water_fill_words_loop3_spec r target logs t_hi t_lo tol t0 hstart hbr hprobe hconv)
  refine (WP.spec_equiv_exists _ _).2 ⟨x, hx, ?_⟩
  intro hxe
  rcases hpost with hcv | hne | ⟨-, hrej | ⟨hs, -⟩⟩
  · exact hcv
  · exact absurd hxe hne
  · exact absurd hrej (hexh x.1)
  · omega

/-! ## 9 · The log pre-pass refinement

`water_fill_words` (`basis.rs:965`) precomputes the per-word log at `basis.rs:974` into three
running quantities: `max` (the running maximum, there only to decide the `theta = 0` early return),
`min_pos` (the smallest positive word, the lower end of the bisection bracket) and `logs` (the log
domain of each word). The extracted loop is `fold_port.stats.basis.water_fill_words_loop0`, a
`Range<usize>` loop whose accumulator is the 4-tuple `(max, min_pos, logs, err)`; the `logs` update
is the `alloc.vec.Vec.index_mut` step, whose back-function is `Vec.set` on the extracted vector
(`water_fill_words_loop0_spec`, `Vec.index_mut_usize_spec`).

The spec states the three values at the loop's end as the pre-bisection maxima: `max` is
`maxOf (vals lambda)`, `min_pos` is `minPosOf (vals lambda)` (the running positive minimum from the
`i64::MAX` sentinel) and `logs` is `map water_log` (after `vals`). The caller hypotheses are stated
explicitly:

* `hstart` / `hend` — the range lies inside the slice;
* `hlog` — every word in the range has a successful `water_log`, with its value pinned by `wv`;
  this is the positive-word condition (a positive word needs `log2` to return, a nonpositive word
  returns `i64::MIN`), and it is the caller's refinement hypothesis for the `log2` series, not a
  theorem here;
* `hmx` / `hmn` / `hlg` — the initial accumulator is the running summary of the prefix already
  scanned, i.e. the zero-state hypothesis for the `[0, len)` instantiation (`water_fill_words_loop0_zero`);
* the only window condition is the slice bound `r.end ≤ lambda.length`; the `i128` window does not
  appear because this loop accumulates no `i128`.

The initial vector is a caller hypothesis (`hlg` at `start = 0` is `replicate len 0`, what
`alloc.vec.from_elem` produces), not constructed here. The loop is the pre-pass only:
`water_fill_words_loop1`, the bisection loops and the whole-body `water_fill_words`/`solve_water_fill`
composition are **not** proved by this section.
-/

/-- `i64::MAX`, the `min_pos` sentinel of `water_fill_words`. -/
def I64_MAX : ℤ := 2 ^ 63 - 1

/-- One `min_pos` update: a positive word below the running minimum replaces it. -/
def minPosStep (l m : ℤ) : ℤ := if 0 < l ∧ l < m then l else m

/-- The running positive minimum over a word list, from the sentinel; `foldl` matches the loop. -/
def minPosOf (L : List ℤ) : ℤ := L.foldl (fun m l => minPosStep l m) I64_MAX

/-- The `logs` value after a prefix: the stated log value of every scanned word, zero elsewhere. -/
def logsOf (wv : ℤ → ℤ) (L : List Std.I64) (k : ℕ) : List ℤ :=
  (L.take k).map (fun w => wv w.val) ++ List.replicate (L.length - k) 0

/-- `vals` of a concatenation. -/
theorem vals_append (l1 l2 : List Std.I64) :
    vals (l1 ++ l2) = vals l1 ++ vals l2 := by
  unfold vals
  rw [List.map_append]

/-- `vals` of a list update. -/
theorem vals_set (L : List Std.I64) (i : ℕ) (x : Std.I64) :
    vals (L.set i x) = (vals L).set i x.val := by
  unfold vals
  exact List.map_set

/-- The running maximum over an appended element. -/
theorem maxOf_append_singleton (l : List ℤ) (x : ℤ) :
    maxOf (l ++ [x]) = max (maxOf l) x := by
  induction l with
  | nil => show max x 0 = max 0 x; rw [max_comm]
  | cons a bs ih =>
    show max a (maxOf (bs ++ [x])) = max (max a (maxOf bs)) x
    rw [ih, max_assoc]

/-- `vals` of a singleton. -/
theorem vals_singleton (x : Std.I64) : vals [x] = [x.val] := rfl

/-- The extracted `max` step at an index inside the slice. -/
theorem maxOf_take_succ (L : List Std.I64) {k : ℕ} (h : k < L.length) :
    maxOf (vals (L.take (k + 1))) = max (maxOf (vals (L.take k))) L[k].val := by
  rw [List.take_succ_eq_append_getElem h, vals_append, vals_singleton,
    maxOf_append_singleton]

/-- The running positive minimum over an appended element. -/
theorem minPosOf_append_singleton (l : List ℤ) (x : ℤ) :
    minPosOf (l ++ [x]) = minPosStep x (minPosOf l) := by
  unfold minPosOf
  rw [List.foldl_append]
  rfl

/-- The extracted `min_pos` step at an index inside the slice. -/
theorem minPosOf_take_succ (L : List Std.I64) {k : ℕ} (h : k < L.length) :
    minPosOf (vals (L.take (k + 1)))
      = minPosStep L[k].val (minPosOf (vals (L.take k))) := by
  rw [List.take_succ_eq_append_getElem h, vals_append, vals_singleton,
    minPosOf_append_singleton]

/-- The `logs` value at an empty prefix is the zero vector. -/
theorem logsOf_zero (wv : ℤ → ℤ) (L : List Std.I64) :
    logsOf wv L 0 = List.replicate L.length 0 := by
  unfold logsOf
  simp

/-- The `logs` value at the full prefix is the mapped word list. -/
theorem logsOf_full (wv : ℤ → ℤ) (L : List Std.I64) :
    logsOf wv L L.length = L.map (fun w => wv w.val) := by
  unfold logsOf
  simp

/-- The `index_mut` step: writing the stated log value at `k` advances the prefix summary. -/
theorem logsOf_set (wv : ℤ → ℤ) (L : List Std.I64) {k : ℕ} (h : k < L.length) :
    (logsOf wv L k).set k (wv L[k].val) = logsOf wv L (k + 1) := by
  unfold logsOf
  have hk : ((L.take k).map (fun w => wv w.val)).length = k := by
    rw [List.length_map, List.length_take_of_le (Nat.le_of_lt h)]
  rw [List.set_append_right k (wv L[k].val) (by omega)]
  have hsub : k - ((L.take k).map (fun w => wv w.val)).length = 0 := by omega
  rw [hsub]
  have hr : List.replicate (L.length - k) (0 : ℤ)
      = (0 : ℤ) :: List.replicate (L.length - k - 1) 0 := by
    conv_lhs => rw [show L.length - k = (L.length - k - 1) + 1 by omega]
    rw [List.replicate_succ]
  simp only [hr, List.set_cons_zero]
  have hnk : L.length - (k + 1) = L.length - k - 1 := by omega
  rw [List.take_succ_eq_append_getElem h, List.map_append, hnk]
  simp

/-- The extracted `max` step as a value: `if l > m then l else m` is `max`. -/
theorem maxStep_val (l m : Std.I64) :
    (if l > m then l else m).val = max l.val m.val := by
  by_cases h : l > m
  · rw [if_pos h]
    rw [gt_iff_lt, IScalar.lt_equiv] at h
    rw [max_eq_left (le_of_lt h)]
  · rw [if_neg h]
    have hle : l.val ≤ m.val := by
      by_contra hc
      exact h (by rw [gt_iff_lt, IScalar.lt_equiv]; omega)
    rw [max_eq_right hle]

/-- The extracted inner `min` step as a value: `if l < m then l else m` is `min`. -/
theorem minInner_val (l m : Std.I64) :
    (if l < m then l else m).val = min l.val m.val := by
  by_cases h : l < m
  · rw [if_pos h]
    rw [IScalar.lt_equiv] at h
    rw [min_eq_left (le_of_lt h)]
  · rw [if_neg h]
    have hle : m.val ≤ l.val := by
      by_contra hc
      exact h (by rw [IScalar.lt_equiv]; omega)
    rw [min_eq_right hle]

/-- `minPosStep` is `min` on a positive word. -/
theorem minPosStep_of_pos (l m : ℤ) (h : 0 < l) : minPosStep l m = min l m := by
  unfold minPosStep
  by_cases hlt : l < m
  · rw [if_pos ⟨h, hlt⟩, min_eq_left (le_of_lt hlt)]
  · rw [if_neg (fun hc => hlt hc.2), min_eq_right (by omega)]

/-- `minPosStep` is the identity on a nonpositive word. -/
theorem minPosStep_of_nonpos (l m : ℤ) (h : ¬ 0 < l) : minPosStep l m = m := by
  unfold minPosStep
  rw [if_neg (fun hc => h hc.1)]

/-- The extracted `min_pos` step as a value: the nested `if` is `minPosStep`. -/
theorem minStep_val (l m : Std.I64) :
    (if l > 0#i64 then (if l < m then l else m) else m).val = minPosStep l.val m.val := by
  unfold minPosStep
  by_cases h0 : l > 0#i64
  · rw [if_pos h0]
    have h0' : 0 < l.val := by
      rw [gt_iff_lt, IScalar.lt_equiv] at h0; exact h0
    by_cases hl : l < m
    · rw [if_pos hl]
      have hl' : l.val < m.val := by
        rw [IScalar.lt_equiv] at hl; exact hl
      rw [if_pos ⟨h0', hl'⟩]
    · rw [if_neg hl]
      have hl' : ¬ l.val < m.val := by
        intro hc
        exact hl (by rw [IScalar.lt_equiv]; exact hc)
      rw [if_neg (fun hc => hl' hc.2)]
  · rw [if_neg h0]
    have h0' : ¬ 0 < l.val := by
      intro hc
      exact h0 (by rw [gt_iff_lt, IScalar.lt_equiv]; exact hc)
    rw [if_neg (fun hc => h0' hc.1)]

/-- The loop body at a live index (the error accumulator is `none`). -/
theorem prep_body_eq (lambda : Slice Std.I64) (it it1 : core.ops.range.Range Std.Usize)
    (mx mn : Std.I64) (lg : alloc.vec.Vec Std.I64)
    (hnext : core.iter.range.IteratorRange.next core.iter.range.StepUsize it
      = ok (some it.start, it1)) :
    fold_port.stats.basis.water_fill_words_loop0.body lambda it mx mn lg none =
      (do
        let l ← Slice.index_usize lambda it.start
        let max1 ← if l > mx then ok l else ok mx
        let min_pos1 ←
          if l > 0#i64 then (if l < mn then ok l else ok mn) else ok mn
        let r ← fold_port.stats.basis.water_log l
        match r with
        | core.result.Result.Ok t =>
          let (_, back) ← alloc.vec.Vec.index_mut
            (core.slice.index.SliceIndexUsizeSlice Std.I64) lg it.start
          let lg1 := back t
          ok (ControlFlow.cont (it1, max1, min_pos1, lg1, none))
        | core.result.Result.Err e =>
          ok (ControlFlow.cont (it1, max1, min_pos1, lg, some e))) := by
  unfold fold_port.stats.basis.water_fill_words_loop0.body
  rw [hnext]
  simp only [bind_tc_ok]
  rfl

/-- The `Vec.index_mut` value at an in-bounds index: the pair is the read value and the
back-function `Vec.set`. -/
theorem index_mut_usize_eq (v : alloc.vec.Vec Std.I64) (i : Std.Usize)
    (h : i.val < v.val.length) :
    alloc.vec.Vec.index_mut_usize v i
      = ok (v.val[i.val], alloc.vec.Vec.set v i) := by
  unfold alloc.vec.Vec.index_mut_usize
  have hx : alloc.vec.Vec.index_usize v i = ok v.val[i.val] := by
    obtain ⟨x, hx, hxe⟩ := (WP.spec_equiv_exists _ _).1
      (alloc.vec.Vec.index_usize_spec v i h)
    rw [hx, hxe]
  rw [hx]
  simp only [bind_tc_ok]

/-- The loop body at the end of the range. -/
theorem prep_body_eq_done (lambda : Slice Std.I64) (it : core.ops.range.Range Std.Usize)
    (mx mn : Std.I64) (lg : alloc.vec.Vec Std.I64) (hge : it.end.val ≤ it.start.val) :
    fold_port.stats.basis.water_fill_words_loop0.body lambda it mx mn lg none
      = ok (ControlFlow.done (mx, mn, lg, none)) := by
  unfold fold_port.stats.basis.water_fill_words_loop0.body
  obtain ⟨p, hp, hopt, -⟩ := (WP.spec_equiv_exists _ _).1
    (core.iter.range.IteratorRange.next_UScalar_none_spec (ty := .Usize)
      (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
      partialOrdUsize_lt it hge)
  obtain ⟨o, it1⟩ := p
  change o = none at hopt
  rw [hopt] at hp
  rw [hp]
  simp only [bind_tc_ok]
  rfl

/-- `water_fill_words_loop0` refines the pre-bisection maxima over `[start, end)`: the running
maximum is `maxOf`, the running positive minimum is `minPosOf`, and `logs` is `map water_log`
(after `vals`), with the error accumulator still `none` on the stated `hlog` window. -/
theorem water_fill_words_loop0_spec (lambda : Slice Std.I64) (wv : ℤ → ℤ)
    (r : core.ops.range.Range Std.Usize) (mx mn : Std.I64)
    (lg : alloc.vec.Vec Std.I64)
    (hend : r.end.val ≤ lambda.val.length) (hstart : r.start.val ≤ r.end.val)
    (hlog : ∀ (k : ℕ) (hk : k < r.end.val),
      ∃ t : Std.I64, fold_port.stats.basis.water_log (lambda.val[k]'(by omega))
        = ok (core.result.Result.Ok t)
        ∧ t.val = wv (lambda.val[k]'(by omega)).val)
    (hmx : mx.val = maxOf (vals (lambda.val.take r.start.val)))
    (hmn : mn.val = minPosOf (vals (lambda.val.take r.start.val)))
    (hlg : vals lg.val = logsOf wv lambda.val r.start.val) :
    fold_port.stats.basis.water_fill_words_loop0 r lambda mx mn lg none ⦃ x =>
      x.1.val = maxOf (vals (lambda.val.take r.end.val)) ∧
      x.2.1.val = minPosOf (vals (lambda.val.take r.end.val)) ∧
      x.2.2.2 = none ∧
      vals x.2.2.1.val = logsOf wv lambda.val r.end.val ⦄ := by
  unfold fold_port.stats.basis.water_fill_words_loop0
  apply loop.spec_decr_nat
    (measure := fun st : core.ops.range.Range Std.Usize × Std.I64 × Std.I64
        × alloc.vec.Vec Std.I64 × Option fold_port.stats.basis.BasisError =>
      st.1.end.val - st.1.start.val)
    (inv := fun st => st.1.end = r.end ∧ st.1.start.val ≤ r.end.val ∧
      st.2.1.val = maxOf (vals (lambda.val.take st.1.start.val)) ∧
      st.2.2.1.val = minPosOf (vals (lambda.val.take st.1.start.val)) ∧
      st.2.2.2.2 = none ∧
      vals st.2.2.2.1.val = logsOf wv lambda.val st.1.start.val)
    (post := fun x : Std.I64 × Std.I64 × alloc.vec.Vec Std.I64
        × Option fold_port.stats.basis.BasisError =>
      x.1.val = maxOf (vals (lambda.val.take r.end.val)) ∧
      x.2.1.val = minPosOf (vals (lambda.val.take r.end.val)) ∧
      x.2.2.2 = none ∧
      vals x.2.2.1.val = logsOf wv lambda.val r.end.val)
  · rintro ⟨it, mx1, mn1, lg1, er1⟩ ⟨hend', hit, hmx', hmn', her, hlg'⟩
    dsimp only at hend' hit hmx' hmn' her hlg' ⊢
    by_cases hlt : it.start.val < it.end.val
    · obtain ⟨p, hp, hopt, hstart1, hend1⟩ := (WP.spec_equiv_exists _ _).1
        (core.iter.range.IteratorRange.next_UScalar_some_spec (ty := .Usize)
          (cloneInst := core.clone.CloneUsize) (partialOrdInst := core.cmp.PartialOrdUsize)
          cloneUsize_eq partialOrdUsize_lt it hlt)
      obtain ⟨o, it1⟩ := p
      change o = some it.start at hopt
      change it1.start.val = it.start.val + 1 at hstart1
      change it1.end = it.end at hend1
      rw [hopt] at hp
      subst her
      rw [prep_body_eq lambda it it1 mx1 mn1 lg1 hp]
      have hlen : it.start.val < lambda.val.length :=
        lt_of_lt_of_le hlt (by rw [hend']; exact hend)
      rw [index_usize_eq lambda it.start hlen]
      simp only [bind_tc_ok]
      have hendv : it.end.val = r.end.val := by rw [hend']
      have hkend : it.start.val < r.end.val := by omega
      obtain ⟨t, hwater, htval⟩ := hlog it.start.val hkend
      have hlen1 : lg1.val.length = lambda.val.length := by
        have h := congrArg List.length hlg'
        simp only [vals, List.length_map] at h
        unfold logsOf at h
        rw [List.length_append, List.length_map, List.length_replicate,
          List.length_take_of_le (by omega)] at h
        omega
      have hlenlog : it.start.val < lg1.val.length := by
        rw [hlen1]; omega
      by_cases hmax : lambda.val[it.start.val] > mx1
      · rw [if_pos hmax]
        simp only [bind_tc_ok]
        by_cases h0 : lambda.val[it.start.val] > 0#i64
        · rw [if_pos h0]
          by_cases hmin : lambda.val[it.start.val] < mn1
          · rw [if_pos hmin]
            simp only [bind_tc_ok]
            rw [hwater]
            simp only [bind_tc_ok]
            rw [alloc.vec.Vec.index_mut_slice_index]
            apply WP.spec_bind (alloc.vec.Vec.index_mut_usize_spec lg1 it.start hlenlog)
            rintro ⟨q1, q2⟩ ⟨hx, hback⟩
            apply WP.spec.ret
            dsimp only
            rw [hback]
            repeat' apply And.intro
            · rw [hend1]; exact hend'
            · rw [hstart1]; omega
            · rw [hstart1, maxOf_take_succ lambda.val hlen, ← hmx', max_comm]
              have hle : mx1.val ≤ lambda.val[it.start.val].val := by
                rw [gt_iff_lt, IScalar.lt_equiv] at hmax; omega
              rw [max_eq_left hle]
            · rw [hstart1, minPosOf_take_succ lambda.val hlen, ← hmn']
              unfold minPosStep
              have h0' : 0 < lambda.val[it.start.val].val := by
                rw [gt_iff_lt, IScalar.lt_equiv] at h0; exact h0
              have hmin' : lambda.val[it.start.val].val < mn1.val := by
                rw [IScalar.lt_equiv] at hmin; exact hmin
              rw [if_pos ⟨h0', hmin'⟩]
            · rfl
            · rw [hstart1, alloc.vec.Vec.set_val_eq, vals_set, htval, hlg',
                logsOf_set wv lambda.val hlen]
            · rw [hstart1, hend1]; omega
          · rw [if_neg hmin]
            simp only [bind_tc_ok]
            rw [hwater]
            simp only [bind_tc_ok]
            rw [alloc.vec.Vec.index_mut_slice_index]
            apply WP.spec_bind (alloc.vec.Vec.index_mut_usize_spec lg1 it.start hlenlog)
            rintro ⟨q1, q2⟩ ⟨hx, hback⟩
            apply WP.spec.ret
            dsimp only
            rw [hback]
            repeat' apply And.intro
            · rw [hend1]; exact hend'
            · rw [hstart1]; omega
            · rw [hstart1, maxOf_take_succ lambda.val hlen, ← hmx', max_comm]
              have hle : mx1.val ≤ lambda.val[it.start.val].val := by
                rw [gt_iff_lt, IScalar.lt_equiv] at hmax; omega
              rw [max_eq_left hle]
            · rw [hstart1, minPosOf_take_succ lambda.val hlen, ← hmn']
              unfold minPosStep
              have hmin' : ¬ lambda.val[it.start.val].val < mn1.val := by
                intro hc
                exact hmin (by rw [IScalar.lt_equiv]; exact hc)
              rw [if_neg (fun hc => hmin' hc.2)]
            · rfl
            · rw [hstart1, alloc.vec.Vec.set_val_eq, vals_set, htval, hlg',
                logsOf_set wv lambda.val hlen]
            · rw [hstart1, hend1]; omega
        · rw [if_neg h0]
          simp only [bind_tc_ok]
          rw [hwater]
          simp only [bind_tc_ok]
          rw [alloc.vec.Vec.index_mut_slice_index]
          apply WP.spec_bind (alloc.vec.Vec.index_mut_usize_spec lg1 it.start hlenlog)
          rintro ⟨q1, q2⟩ ⟨hx, hback⟩
          apply WP.spec.ret
          dsimp only
          rw [hback]
          repeat' apply And.intro
          · rw [hend1]; exact hend'
          · rw [hstart1]; omega
          · rw [hstart1, maxOf_take_succ lambda.val hlen, ← hmx', max_comm]
            have hle : mx1.val ≤ lambda.val[it.start.val].val := by
              rw [gt_iff_lt, IScalar.lt_equiv] at hmax; omega
            rw [max_eq_left hle]
          · rw [hstart1, minPosOf_take_succ lambda.val hlen, ← hmn']
            unfold minPosStep
            have h0' : ¬ 0 < lambda.val[it.start.val].val := by
              intro hc
              exact h0 (by rw [gt_iff_lt, IScalar.lt_equiv]; exact hc)
            rw [if_neg (fun hc => h0' hc.1)]
          · rfl
          · rw [hstart1, alloc.vec.Vec.set_val_eq, vals_set, htval, hlg',
              logsOf_set wv lambda.val hlen]
          · rw [hstart1, hend1]; omega
      · rw [if_neg hmax]
        simp only [bind_tc_ok]
        by_cases h0 : lambda.val[it.start.val] > 0#i64
        · rw [if_pos h0]
          by_cases hmin : lambda.val[it.start.val] < mn1
          · rw [if_pos hmin]
            simp only [bind_tc_ok]
            rw [hwater]
            simp only [bind_tc_ok]
            rw [alloc.vec.Vec.index_mut_slice_index]
            apply WP.spec_bind (alloc.vec.Vec.index_mut_usize_spec lg1 it.start hlenlog)
            rintro ⟨q1, q2⟩ ⟨hx, hback⟩
            apply WP.spec.ret
            dsimp only
            rw [hback]
            repeat' apply And.intro
            · rw [hend1]; exact hend'
            · rw [hstart1]; omega
            · rw [hstart1, maxOf_take_succ lambda.val hlen, ← hmx']
              have hle : lambda.val[it.start.val].val ≤ mx1.val := by
                by_contra hc
                exact hmax (by rw [gt_iff_lt, IScalar.lt_equiv]; omega)
              rw [max_eq_left hle]
            · rw [hstart1, minPosOf_take_succ lambda.val hlen, ← hmn']
              unfold minPosStep
              have h0' : 0 < lambda.val[it.start.val].val := by
                rw [gt_iff_lt, IScalar.lt_equiv] at h0; exact h0
              have hmin' : lambda.val[it.start.val].val < mn1.val := by
                rw [IScalar.lt_equiv] at hmin; exact hmin
              rw [if_pos ⟨h0', hmin'⟩]
            · rfl
            · rw [hstart1, alloc.vec.Vec.set_val_eq, vals_set, htval, hlg',
                logsOf_set wv lambda.val hlen]
            · rw [hstart1, hend1]; omega
          · rw [if_neg hmin]
            simp only [bind_tc_ok]
            rw [hwater]
            simp only [bind_tc_ok]
            rw [alloc.vec.Vec.index_mut_slice_index]
            apply WP.spec_bind (alloc.vec.Vec.index_mut_usize_spec lg1 it.start hlenlog)
            rintro ⟨q1, q2⟩ ⟨hx, hback⟩
            apply WP.spec.ret
            dsimp only
            rw [hback]
            repeat' apply And.intro
            · rw [hend1]; exact hend'
            · rw [hstart1]; omega
            · rw [hstart1, maxOf_take_succ lambda.val hlen, ← hmx']
              have hle : lambda.val[it.start.val].val ≤ mx1.val := by
                by_contra hc
                exact hmax (by rw [gt_iff_lt, IScalar.lt_equiv]; omega)
              rw [max_eq_left hle]
            · rw [hstart1, minPosOf_take_succ lambda.val hlen, ← hmn']
              unfold minPosStep
              have hmin' : ¬ lambda.val[it.start.val].val < mn1.val := by
                intro hc
                exact hmin (by rw [IScalar.lt_equiv]; exact hc)
              rw [if_neg (fun hc => hmin' hc.2)]
            · rfl
            · rw [hstart1, alloc.vec.Vec.set_val_eq, vals_set, htval, hlg',
                logsOf_set wv lambda.val hlen]
            · rw [hstart1, hend1]; omega
        · rw [if_neg h0]
          simp only [bind_tc_ok]
          rw [hwater]
          simp only [bind_tc_ok]
          rw [alloc.vec.Vec.index_mut_slice_index]
          apply WP.spec_bind (alloc.vec.Vec.index_mut_usize_spec lg1 it.start hlenlog)
          rintro ⟨q1, q2⟩ ⟨hx, hback⟩
          apply WP.spec.ret
          dsimp only
          rw [hback]
          repeat' apply And.intro
          · rw [hend1]; exact hend'
          · rw [hstart1]; omega
          · rw [hstart1, maxOf_take_succ lambda.val hlen, ← hmx']
            have hle : lambda.val[it.start.val].val ≤ mx1.val := by
              by_contra hc
              exact hmax (by rw [gt_iff_lt, IScalar.lt_equiv]; omega)
            rw [max_eq_left hle]
          · rw [hstart1, minPosOf_take_succ lambda.val hlen, ← hmn']
            unfold minPosStep
            have h0' : ¬ 0 < lambda.val[it.start.val].val := by
              intro hc
              exact h0 (by rw [gt_iff_lt, IScalar.lt_equiv]; exact hc)
            rw [if_neg (fun hc => h0' hc.1)]
          · rfl
          · rw [hstart1, alloc.vec.Vec.set_val_eq, vals_set, htval, hlg',
              logsOf_set wv lambda.val hlen]
          · rw [hstart1, hend1]; omega
    · subst her
      rw [prep_body_eq_done lambda it mx1 mn1 lg1 (by omega)]
      have hendv : it.end.val = r.end.val := by rw [hend']
      have hstart_eq : it.start.val = r.end.val := by omega
      rw [hstart_eq] at hmx' hmn' hlg'
      simp only [WP.spec_ok]
      exact ⟨hmx', hmn', trivial, hlg'⟩
  · refine ⟨rfl, hstart, hmx, hmn, rfl, hlg⟩

/-- The pre-pass on the whole slice from the zero state: `maxOf`, `minPosOf` and `map water_log`
at `len`. The zero `logs` vector is the caller's `hlg` (`replicate len 0`, what
`alloc.vec.from_elem` builds), not constructed here. -/
theorem water_fill_words_loop0_zero (lambda : Slice Std.I64) (wv : ℤ → ℤ)
    (lg : alloc.vec.Vec Std.I64)
    (hlog : ∀ (k : ℕ) (hk : k < lambda.val.length),
      ∃ t : Std.I64, fold_port.stats.basis.water_log lambda.val[k]
        = ok (core.result.Result.Ok t) ∧ t.val = wv lambda.val[k].val)
    (hlg : vals lg.val = List.replicate lambda.val.length 0) :
    fold_port.stats.basis.water_fill_words_loop0
      { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX lg none ⦃ x =>
        x.1.val = maxOf (vals lambda.val) ∧
        x.2.1.val = minPosOf (vals lambda.val) ∧
        x.2.2.2 = none ∧
        vals x.2.2.1.val = lambda.val.map (fun w => wv w.val) ⦄ := by
  have hlen : (Slice.len lambda).val = lambda.val.length := Slice.len_val lambda
  have h0i : (0#i64).val = (0 : ℤ) := by simp
  have h0u : (0#usize).val = (0 : ℕ) := by simp
  have hmaxi : (core.num.I64.MAX).val = I64_MAX := by
    unfold I64_MAX
    norm_num [core.num.I64.MAX, I64.rMax]
  have h := water_fill_words_loop0_spec lambda wv
    { start := 0#usize, «end» := Slice.len lambda } 0#i64 core.num.I64.MAX lg
    (by rw [hlen]) (by simp)
    (fun k hk => hlog k (by rw [hlen] at hk; exact hk))
    (by
      rw [h0i, h0u]
      show (0 : ℤ) = maxOf (vals (lambda.val.take 0))
      simp [maxOf, vals])
    (by
      rw [hmaxi, h0u]
      show I64_MAX = minPosOf (vals (lambda.val.take 0))
      simp [minPosOf, vals])
    (by
      rw [h0u, hlg]
      exact (logsOf_zero wv lambda.val).symm)
  obtain ⟨x, hx, h1, h2, h3, h4⟩ := (WP.spec_equiv_exists _ _).1 h
  rw [hlen] at h1 h2 h4
  simp only [List.take_length] at h1 h2
  refine (WP.spec_equiv_exists _ _).2 ⟨x, hx, h1, h2, h3, ?_⟩
  rw [h4, logsOf_full]

/-! ## 10 · Firing the extracted body (evaluator reading, not a kernel check) -/

#guard runField (fold_port.stats.basis.solve_water_fill slice4 34359738368#i64 0#i64)
  (fun w => w.theta.val) == some 33554432
#guard runField (fold_port.stats.basis.solve_water_fill slice4 34359738368#i64 0#i64)
  (fun w => w.distortion.val) == some 67108864
#guard runField (fold_port.stats.basis.solve_water_fill slice4 34359738368#i64 0#i64)
  (fun w => w.rate.val) == some 34359738368
#guard runField (fold_port.stats.basis.solve_water_fill slice4 0#i64 0#i64)
  (fun w => w.theta.val) == some 17179869184
#guard runField (fold_port.stats.basis.solve_water_fill slice4 0#i64 0#i64)
  (fun w => w.distortion.val) == some 21474836480
#guard runField (fold_port.stats.basis.solve_water_fill slice1 8589934592#i64 0#i64)
  (fun w => w.theta.val) == some 1073741825
#guard runField (fold_port.stats.basis.solve_water_fill slice1 8589934592#i64 0#i64)
  (fun w => w.rate.val) == some 8589934586
#guard runField (fold_port.stats.basis.solve_water_fill slice1 8589934592#i64 0#i64)
  (fun w => w.distortion.val) == some 2147483650



/-! ## 8 · The extracted `rate_words` body and the total `distortion_words` post

`fold_port.stats.basis.distortion_words` (`basis.rs:918`) returns `Ok 0` at a non-positive level and
`narrow(Σ min(λ, θ))` otherwise; `distortion_words_post` collects the three branches: the
non-positive level, the fold inside the `i64` window, and the `OutOfRange` refusal of a fold outside
it.

`fold_port.stats.basis.rate_words` (`basis.rs:931`) is the body `water_fill_words` calls on the
rounded level word; it is **not** `rate_at_logs` (the model's rate function, section 6). This section
refines the extracted body itself:

* the non-positive branch runs the flag loop `rate_words_loop0` and returns `Ok i64::MAX` when some
  word is positive, `Ok 0` otherwise (`rate_words_nonpos`);
* the positive branch runs `log2 theta` (`Err → OutOfRange`, `rate_words_pos_none`), the slice
  loop `rate_words_loop1` and
  the `>>> 1`/`narrow` tail. The loop is refined to two total functions on the word list: `rateAt`,
  the active-log sum (`Σ_{λᵢ > θ} (logv λᵢ − t)`), and `rateBad`, a witness that some active word's
  difference leaves the `i64` window and freezes the accumulator at the flag.
  `rate_words_loop1_spec` states the `(acc, failed)` pair on both flags; `rate_words_pos_spec` gives
  the exact half-sum on the no-failure window, and `rate_words_pos_fail` the `OutOfRange` of a
  failing `checked_sub`.

The `log2` per-word values stay the caller's hypothesis `hlog` (`rate_words` reads the extracted
`log2`, whose `fixed_ln` series is not refined here), so no theorem below states the real value
`Σ max(0, ½ log₂(λ/θ))`.
-/

/-- The `log2`-domain rate accumulator of `rate_words`: the sum of `log₂ λ_i − t` over the words
above the level, zero on the others. -/
def rateAt (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ) (L : List Std.I64) : ℤ :=
  (L.map (fun l => if theta.val < l.val then (logv l).val - t else 0)).sum

theorem rateAt_nil (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ) :
    rateAt logv theta t [] = 0 := rfl

theorem rateAt_cons (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ)
    (l : Std.I64) (ls : List Std.I64) :
    rateAt logv theta t (l :: ls)
      = (if theta.val < l.val then (logv l).val - t else 0) + rateAt logv theta t ls := rfl

/-- The accumulator is nonnegative when every active term is. -/
theorem rateAt_nonneg (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ) (L : List Std.I64)
    (hmono : ∀ l, theta.val < l.val → 0 ≤ (logv l).val - t) : 0 ≤ rateAt logv theta t L := by
  induction L with
  | nil => simp [rateAt]
  | cons l ls ih =>
    rw [rateAt_cons]
    by_cases hgt : theta.val < l.val
    · rw [if_pos hgt]
      have h := hmono l hgt
      have h' := ih
      omega
    · rw [if_neg hgt]
      simpa using ih

theorem rateAt_append (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ)
    (L1 L2 : List Std.I64) :
    rateAt logv theta t (L1 ++ L2) = rateAt logv theta t L1 + rateAt logv theta t L2 := by
  unfold rateAt
  rw [List.map_append, List.sum_append]

/-- The accumulator of a shorter suffix bounds the longer one. -/
theorem rateAt_drop_le (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ)
    (L : List Std.I64) (hmono : ∀ l, theta.val < l.val → 0 ≤ (logv l).val - t) {n m : ℕ}
    (h : n ≤ m) : rateAt logv theta t (L.drop m) ≤ rateAt logv theta t (L.drop n) := by
  have hsplit : L.drop n = (L.drop n).take (m - n) ++ L.drop m := by
    conv_lhs => rw [← List.take_append_drop (m - n) (L.drop n)]
    rw [List.drop_drop, Nat.add_sub_of_le h]
  rw [hsplit, rateAt_append]
  have := rateAt_nonneg logv theta t ((L.drop n).take (m - n)) hmono
  omega

/-- The `Option::and_then` of the loop body's closure is the `i64` checked subtraction. -/
theorem rate_and_then (x t : Std.I64) :
    core.option.Option.and_then
      fold_port.stats.basis.rate_words.closure.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
      (some x) t = ok (I64.checked_sub x t) := rfl

/-- A `checked_sub` inside the nonnegative `i64` window returns the exact difference. -/
theorem checked_sub_some_window (l t : Std.I64)
    (h0 : 0 ≤ l.val - t.val) (hmax : l.val - t.val ≤ 2 ^ 63 - 1) :
    ∃ d : Std.I64, I64.checked_sub l t = some d ∧ d.val = l.val - t.val := by
  have hspec := I64.checked_sub_bv_spec l t
  match h : core.num.checked_sub_IScalar l t with
  | some d =>
    refine ⟨d, h, ?_⟩
    rw [h] at hspec
    exact hspec.2.2.1
  | none =>
    exfalso
    rw [h] at hspec
    rw [I64.min_eq, I64.max_eq] at hspec
    omega

/-- An `i128` accumulator add of a widened nonnegative `i64`, on the stated window. -/
theorem i128_add_from' (a : Std.I128) (d : Std.I64) (hd0 : 0 ≤ d.val)
    (h : a.val + d.val ≤ 2 ^ 126) :
    ∃ acc1 : Std.I128, a + core.convert.num.FromI128I64.from d = ok acc1 ∧
      acc1.val = a.val + d.val := by
  have hmin : I128.min ≤ a.val + (core.convert.num.FromI128I64.from d).val := by
    rw [core.convert.num.FromI128I64.from_val_eq, I128.min_eq]
    have := I128_min_val a
    omega
  have hmax : a.val + (core.convert.num.FromI128I64.from d).val ≤ I128.max := by
    rw [core.convert.num.FromI128I64.from_val_eq, I128.max_eq]
    omega
  obtain ⟨x, hx, hpost⟩ := (WP.spec_equiv_exists _ _).1 (I128.add_spec hmin hmax)
  refine ⟨x, hx, ?_⟩
  rw [core.convert.num.FromI128I64.from_val_eq] at hpost
  exact hpost

/-- The `distortion_words` non-positive branch: the level `θ ≤ 0` returns `Ok 0`. -/
theorem distortion_words_nonpos (lambda : Slice Std.I64) (theta : Std.I64)
    (hθ : theta.val ≤ 0) :
    fold_port.stats.basis.distortion_words lambda theta
      = ok (core.result.Result.Ok 0#i64) := by
  unfold fold_port.stats.basis.distortion_words
  rw [if_pos (by rw [IScalar.le_equiv]; simpa using hθ)]

/-- The `distortion_words` refusal: a fold above the `i64` window returns `OutOfRange`. -/
theorem distortion_words_overflow (lambda : Slice Std.I64) {theta : Std.I64}
    (hθ : 0 < theta.val) (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (htot : distList theta.val lambda.val ≤ 2 ^ 126)
    (hbig : 2 ^ 63 - 1 < distList theta.val lambda.val) :
    fold_port.stats.basis.distortion_words lambda theta
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  have hD0 : 0 ≤ distList theta.val lambda.val := distList_nonneg _ _ (le_of_lt hθ) hL
  have htot' : (0 : ℤ) + distList theta.val (lambda.val.drop 0) ≤ 2 ^ 126 := by
    rw [List.drop_zero]; simpa using htot
  have hloop := distortion_words_loop_spec (theta := theta) (hθ := hθ) lambda 0 (Nat.zero_le _)
    0#i128 hL (by simp) htot'
  rw [distortion_words_eq lambda hθ]
  obtain ⟨acc, hacc, hval⟩ := (WP.spec_equiv_exists _ _).1 hloop
  rw [hacc]
  simp only [bind_tc_ok]
  have haccval : acc.val = distList theta.val lambda.val := by
    rw [hval, List.drop_zero]
    have h0 : (0#i128).val = (0 : ℤ) := by simp
    omega
  have hpos : acc > core.convert.num.FromI128I64.from core.num.I64.MAX := by
    rw [gt_iff_lt, IScalar.lt_equiv, fromI64_MAX_val]; omega
  simp only [solve_port.q32.narrow, lift, bind_tc_ok, hpos, if_true]
  simp

/-- **The exact post of `distortion_words`**: the non-positive level returns `Ok 0`; a positive
level returns the `i64` image of `Σ min(λ, θ)` inside the `i64` window, and `OutOfRange` outside
it. -/
theorem distortion_words_post (lambda : Slice Std.I64) (theta : Std.I64)
    (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (htot : distList theta.val lambda.val ≤ 2 ^ 126) :
    (theta.val ≤ 0 →
      fold_port.stats.basis.distortion_words lambda theta
        = ok (core.result.Result.Ok 0#i64)) ∧
    (0 < theta.val →
      ((distList theta.val lambda.val ≤ 2 ^ 63 - 1 →
        ∃ v : Std.I64, fold_port.stats.basis.distortion_words lambda theta
          = ok (core.result.Result.Ok v) ∧ v.val = distList theta.val lambda.val) ∧
       (2 ^ 63 - 1 < distList theta.val lambda.val →
        fold_port.stats.basis.distortion_words lambda theta
          = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange)))) := by
  refine ⟨fun h => distortion_words_nonpos lambda theta h, fun hθ =>
    ⟨fun hfit => ?_, fun hbig => distortion_words_overflow lambda hθ hL htot hbig⟩⟩
  obtain ⟨a, ha, v, hv, hvval⟩ :=
    (WP.spec_equiv_exists _ _).1 (distortion_words_spec lambda hθ hL hfit)
  exact ⟨v, by rw [ha, hv], hvval⟩

/-- The active-log sum is exceeded by some word: the loop 1 failure condition. -/
def rateBad (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ) (L : List Std.I64) : Prop :=
  ∃ l ∈ L, theta.val < l.val ∧ 2 ^ 63 - 1 < (logv l).val - t

theorem rateBad_append_iff (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ)
    (L1 L2 : List Std.I64) :
    rateBad logv theta t (L1 ++ L2) ↔ rateBad logv theta t L1 ∨ rateBad logv theta t L2 := by
  simp only [rateBad, List.mem_append]
  constructor
  · rintro ⟨l, hl | hl, h1, h2⟩
    · exact Or.inl ⟨l, hl, h1, h2⟩
    · exact Or.inr ⟨l, hl, h1, h2⟩
  · rintro (⟨l, hl, h1, h2⟩ | ⟨l, hl, h1, h2⟩)
    · exact ⟨l, Or.inl hl, h1, h2⟩
    · exact ⟨l, Or.inr hl, h1, h2⟩

theorem rateBad_append_left (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ)
    (L1 L2 : List Std.I64) (h : rateBad logv theta t L1) :
    rateBad logv theta t (L1 ++ L2) := (rateBad_append_iff logv theta t L1 L2).2 (Or.inl h)

theorem rateBad_append_right (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ)
    (L1 L2 : List Std.I64) (h : rateBad logv theta t L2) :
    rateBad logv theta t (L1 ++ L2) := (rateBad_append_iff logv theta t L1 L2).2 (Or.inr h)

theorem rateBad_singleton (logv : Std.I64 → Std.I64) (theta : Std.I64) (t : ℤ)
    (l : Std.I64) :
    rateBad logv theta t [l] ↔ theta.val < l.val ∧ 2 ^ 63 - 1 < (logv l).val - t := by
  simp [rateBad]

/-- The loop 1 body at a live index with the flag already set. -/
theorem rw_body_eq_set (theta t : Std.I64) (it : core.slice.iter.Iter Std.I64)
    (acc : Std.I128) (hlt : it.i < it.slice.val.length) :
    fold_port.stats.basis.rate_words_loop1.body theta t it acc true
      = ok (ControlFlow.cont ({ it with i := it.i + 1 }, acc, true)) := by
  unfold fold_port.stats.basis.rate_words_loop1.body
  rw [slice_next_eq, dif_pos hlt]
  simp only [bind_tc_ok]
  simp

/-- The loop 1 body at a live index on an element at or below the level. -/
theorem rw_body_eq_idle (theta t : Std.I64) (it : core.slice.iter.Iter Std.I64)
    (acc : Std.I128) (hlt : it.i < it.slice.val.length)
    (hle : ¬ it.slice.val[it.i] > theta) :
    fold_port.stats.basis.rate_words_loop1.body theta t it acc false
      = ok (ControlFlow.cont ({ it with i := it.i + 1 }, acc, false)) := by
  unfold fold_port.stats.basis.rate_words_loop1.body
  rw [slice_next_eq, dif_pos hlt]
  simp only [bind_tc_ok]
  simp
  intro h
  exact absurd h hle

/-- The loop 1 body at a live index on an element above the level, inside the `i64` window. -/
theorem rw_body_eq_success (theta t : Std.I64) (it : core.slice.iter.Iter Std.I64)
    (acc : Std.I128) (logv : Std.I64 → Std.I64) (d : Std.I64) (acc1 : Std.I128)
    (hlt : it.i < it.slice.val.length) (hgt : it.slice.val[it.i] > theta)
    (hlog : fold_port.stats.basis.log2 it.slice.val[it.i]
      = ok (some (logv it.slice.val[it.i])))
    (hcs : I64.checked_sub (logv it.slice.val[it.i]) t = some d)
    (hadd : acc + core.convert.num.FromI128I64.from d = ok acc1) :
    fold_port.stats.basis.rate_words_loop1.body theta t it acc false
      = ok (ControlFlow.cont ({ it with i := it.i + 1 }, acc1, false)) := by
  unfold fold_port.stats.basis.rate_words_loop1.body
  rw [slice_next_eq, dif_pos hlt]
  simp only [bind_tc_ok]
  simp [hlog, rate_and_then, hcs, lift, hadd]
  intro h
  exfalso
  rw [gt_iff_lt, IScalar.lt_equiv] at hgt
  omega

/-- The loop 1 body at a live index on an element above the level, outside the `i64` window. -/
theorem rw_body_eq_overflow (theta t : Std.I64) (it : core.slice.iter.Iter Std.I64)
    (acc : Std.I128) (logv : Std.I64 → Std.I64)
    (hlt : it.i < it.slice.val.length) (hgt : it.slice.val[it.i] > theta)
    (hlog : fold_port.stats.basis.log2 it.slice.val[it.i]
      = ok (some (logv it.slice.val[it.i])))
    (hcs : I64.checked_sub (logv it.slice.val[it.i]) t = none) :
    fold_port.stats.basis.rate_words_loop1.body theta t it acc false
      = ok (ControlFlow.cont ({ it with i := it.i + 1 }, acc, true)) := by
  unfold fold_port.stats.basis.rate_words_loop1.body
  rw [slice_next_eq, dif_pos hlt]
  simp only [bind_tc_ok]
  simp [hlog, rate_and_then, hcs, lift]
  exact hgt

/-- The loop 1 body at the end of the slice. -/
theorem rw_body_eq_done (theta t : Std.I64) (it : core.slice.iter.Iter Std.I64)
    (acc : Std.I128) (f : Bool) (hge : it.slice.val.length ≤ it.i) :
    fold_port.stats.basis.rate_words_loop1.body theta t it acc f
      = ok (ControlFlow.done (acc, f)) := by
  unfold fold_port.stats.basis.rate_words_loop1.body
  rw [slice_next_eq, dif_neg (by omega)]
  simp only [bind_tc_ok]
  simp

/-- **The `rate_words` loop 1 refinement**: the flag is on exactly when some active log
difference leaves the `i64` window, and off the accumulator is the exact active-log sum. -/
theorem rate_words_loop1_spec (s : Slice Std.I64) (theta t : Std.I64)
    (logv : Std.I64 → Std.I64) (acc0 : Std.I128)
    (hlog : ∀ l, fold_port.stats.basis.log2 l = ok (some (logv l)))
    (hmono : ∀ l, theta.val < l.val → 0 ≤ (logv l).val - t.val)
    (hbound : acc0.val + rateAt logv theta t.val s.val ≤ 2 ^ 126) :
    fold_port.stats.basis.rate_words_loop1 ⟨s, 0⟩ theta t acc0 false ⦃ x =>
      (x.2 = true ↔ rateBad logv theta t.val s.val) ∧
      (x.2 = false → x.1.val = acc0.val + rateAt logv theta t.val s.val) ⦄ := by
  unfold fold_port.stats.basis.rate_words_loop1
  apply loop.spec_decr_nat
    (measure := fun st : core.slice.iter.Iter Std.I64 × Std.I128 × Bool =>
      st.1.slice.val.length - st.1.i)
    (inv := fun st => st.1.slice = s ∧ st.1.i ≤ s.val.length ∧
      (st.2.2 = true ↔ rateBad logv theta t.val (s.val.take st.1.i)) ∧
      st.2.1.val ≤ acc0.val + rateAt logv theta t.val s.val ∧
      (st.2.2 = false →
        st.2.1.val = acc0.val + (rateAt logv theta t.val s.val
          - rateAt logv theta t.val (s.val.drop st.1.i))))
    (post := fun x : Std.I128 × Bool =>
      (x.2 = true ↔ rateBad logv theta t.val s.val) ∧
      (x.2 = false → x.1.val = acc0.val + rateAt logv theta t.val s.val))
  · rintro ⟨it, a, f⟩ ⟨hslice, hit, hflag, hcap, hsum⟩
    dsimp only at hslice hit hflag hcap hsum ⊢
    by_cases hlt : it.i < it.slice.val.length
    · have hd : it.slice.val.drop it.i = it.slice.val[it.i] :: it.slice.val.drop (it.i + 1) :=
        List.drop_eq_getElem_cons hlt
      have htk : it.slice.val.take (it.i + 1)
          = it.slice.val.take it.i ++ [it.slice.val[it.i]] :=
        List.take_succ_eq_append_getElem hlt
      subst hslice
      cases f with
      | true =>
        rw [rw_body_eq_set theta t it a hlt, WP.spec_ok]
        dsimp only
        have hrb : rateBad logv theta t.val (it.slice.val.take it.i) := hflag.1 rfl
        refine ⟨⟨rfl, by omega, ?_, hcap, ?_⟩, by omega⟩
        · refine ⟨fun _ => ?_, fun _ => rfl⟩
          rw [htk]
          exact rateBad_append_left _ _ _ _ _ hrb
        · intro h; exact absurd h (by simp)
      | false =>
        have hnr : ¬ rateBad logv theta t.val (it.slice.val.take it.i) := by
          intro h
          exact Bool.noConfusion (hflag.2 h)
        have hsum' : a.val = acc0.val + (rateAt logv theta t.val it.slice.val
            - rateAt logv theta t.val (it.slice.val.drop it.i)) := hsum rfl
        by_cases hgtP : it.slice.val[it.i] > theta
        · have hgt : theta.val < it.slice.val[it.i].val := by
            rw [gt_iff_lt, IScalar.lt_equiv] at hgtP
            exact hgtP
          have hsplit : rateAt logv theta t.val (it.slice.val.drop it.i)
              = ((logv it.slice.val[it.i]).val - t.val)
                + rateAt logv theta t.val (it.slice.val.drop (it.i + 1)) := by
            rw [hd, rateAt_cons, if_pos hgt]
          by_cases hwin : 0 ≤ (logv it.slice.val[it.i]).val - t.val ∧
              (logv it.slice.val[it.i]).val - t.val ≤ 2 ^ 63 - 1
          · obtain ⟨d, hcs, hdval⟩ := checked_sub_some_window _ _ hwin.1 hwin.2
            have hbud : a.val + d.val ≤ 2 ^ 126 := by
              have h1 : 0 ≤ rateAt logv theta t.val (it.slice.val.drop (it.i + 1)) :=
                rateAt_nonneg _ _ _ _ hmono
              rw [hsplit] at hsum'
              rw [hdval]
              omega
            obtain ⟨acc1, hadd, hacc1⟩ := i128_add_from' a d (by rw [hdval]; exact hwin.1) hbud
            rw [rw_body_eq_success theta t it a logv d acc1 hlt hgtP (hlog _) hcs hadd,
              WP.spec_ok]
            dsimp only
            have hnr' : ¬ rateBad logv theta t.val (it.slice.val.take (it.i + 1)) := by
              intro h
              rw [htk] at h
              rcases (rateBad_append_iff _ _ _ _ _).1 h with h' | h'
              · exact hnr h'
              · rw [rateBad_singleton] at h'
                omega
            have hR0 : 0 ≤ rateAt logv theta t.val (it.slice.val.drop (it.i + 1)) :=
              rateAt_nonneg _ _ _ _ hmono
            refine ⟨⟨rfl, by omega, ?_, ?_, ?_⟩, by omega⟩
            · exact ⟨fun h => Bool.noConfusion h, fun h => absurd h hnr'⟩
            · rw [hacc1, hdval]; omega
            · rw [hacc1, hdval]; omega
          · have hcs : I64.checked_sub (logv it.slice.val[it.i]) t = none := by
              cases hc : core.num.checked_sub_IScalar (logv it.slice.val[it.i]) t with
              | none => exact hc
              | some d =>
                have hspec := I64.checked_sub_bv_spec (logv it.slice.val[it.i]) t
                rw [hc] at hspec
                rw [I64.min_eq, I64.max_eq] at hspec
                exact absurd ⟨hmono _ hgt, hspec.2.1⟩ hwin
            have hbig : 2 ^ 63 - 1 < (logv it.slice.val[it.i]).val - t.val := by
              by_contra hc
              exact hwin ⟨hmono _ hgt, by omega⟩
            rw [rw_body_eq_overflow theta t it a logv hlt hgtP (hlog _) hcs, WP.spec_ok]
            dsimp only
            have hrb : rateBad logv theta t.val (it.slice.val.take (it.i + 1)) := by
              rw [htk]
              exact rateBad_append_right _ _ _ _ _
                ⟨it.slice.val[it.i], List.mem_singleton_self _, hgt, hbig⟩
            refine ⟨⟨rfl, by omega, ?_, hcap, ?_⟩, by omega⟩
            · exact ⟨fun _ => hrb, fun _ => rfl⟩
            · intro h; exact absurd h (by simp)
        · rw [rw_body_eq_idle theta t it a hlt hgtP, WP.spec_ok]
          dsimp only
          have hle' : ¬ theta.val < (it.slice.val[it.i]).val := by
            intro h
            exact hgtP (by rw [gt_iff_lt, IScalar.lt_equiv]; exact h)
          have hsplit : rateAt logv theta t.val (it.slice.val.drop it.i)
              = rateAt logv theta t.val (it.slice.val.drop (it.i + 1)) := by
            rw [hd, rateAt_cons, if_neg hle']
            omega
          have hnr' : ¬ rateBad logv theta t.val (it.slice.val.take (it.i + 1)) := by
            intro h
            rw [htk] at h
            rcases (rateBad_append_iff _ _ _ _ _).1 h with h' | h'
            · exact hnr h'
            · rw [rateBad_singleton] at h'
              omega
          refine ⟨⟨rfl, by omega, ?_, hcap, ?_⟩, by omega⟩
          · exact ⟨fun h => Bool.noConfusion h, fun h => absurd h hnr'⟩
          · simpa [hsplit] using hsum'
    · subst hslice
      rw [rw_body_eq_done theta t it a f (Nat.le_of_not_lt hlt), WP.spec_ok]
      have htake : it.slice.val.take it.i = it.slice.val := by
        rw [show it.i = it.slice.val.length by omega, List.take_of_length_le (le_refl _)]
      have hdrop : it.slice.val.drop it.i = [] := List.drop_eq_nil_of_le (by omega)
      refine ⟨?_, ?_⟩
      · rw [htake] at hflag
        exact hflag
      · intro hf
        rw [hdrop] at hsum
        rw [rateAt_nil, sub_zero] at hsum
        exact hsum hf
  · refine ⟨rfl, Nat.zero_le _, ?_, ?_, ?_⟩
    · refine ⟨fun h => Bool.noConfusion h, fun h => ?_⟩
      simp [rateBad] at h
    · have h0 := rateAt_nonneg logv theta t.val s.val hmono
      show acc0.val ≤ acc0.val + rateAt logv theta t.val s.val
      omega
    · intro _
      rw [List.drop_zero, sub_self, add_zero]

/-! ### 8.1 · The non-positive branch: `rate_words_loop0`

At a non-positive level the body scans the words for a positive one and returns `i64::MAX` there,
`0` otherwise. `anyPos` is that predicate, `rate_words_loop0_spec` refines the flag loop to it, and
`rate_words_nonpos` carries the loop to the extracted body's two results.
-/

/-- Some word of the list is positive. -/
def anyPos (L : List Std.I64) : Prop := ∃ l ∈ L, 0 < l.val

theorem anyPos_nil : ¬ anyPos [] := by simp [anyPos]

theorem anyPos_append_singleton (L : List Std.I64) (l : Std.I64) :
    anyPos (L ++ [l]) ↔ anyPos L ∨ 0 < l.val := by
  constructor
  · rintro ⟨x, hx, h0⟩
    rcases List.mem_append.1 hx with hx | hx
    · exact Or.inl ⟨x, hx, h0⟩
    · exact Or.inr (by rw [List.mem_singleton.1 hx] at h0; exact h0)
  · rintro (⟨x, hx, h0⟩ | h0)
    · exact ⟨x, List.mem_append_left _ hx, h0⟩
    · exact ⟨l, List.mem_append_right _ (List.mem_singleton_self l), h0⟩

/-- The loop 0 body at a live positive word. -/
theorem l0_body_eq_cont_pos (it : core.slice.iter.Iter Std.I64) (any : Bool)
    (hlt : it.i < it.slice.val.length) (h : it.slice.val[it.i] > 0#i64) :
    fold_port.stats.basis.rate_words_loop0.body it any
      = ok (ControlFlow.cont ({ it with i := it.i + 1 }, true)) := by
  unfold fold_port.stats.basis.rate_words_loop0.body
  rw [slice_next_eq, dif_pos hlt]
  simp only [bind_tc_ok]
  simp
  intro hc
  exfalso
  have hz : (0#i64).val = (0 : ℤ) := by simp
  rw [gt_iff_lt, IScalar.lt_equiv, hz] at h
  omega

/-- The loop 0 body at a live non-positive word. -/
theorem l0_body_eq_cont_neg (it : core.slice.iter.Iter Std.I64) (any : Bool)
    (hlt : it.i < it.slice.val.length) (h : ¬ it.slice.val[it.i] > 0#i64) :
    fold_port.stats.basis.rate_words_loop0.body it any
      = ok (ControlFlow.cont ({ it with i := it.i + 1 }, any)) := by
  unfold fold_port.stats.basis.rate_words_loop0.body
  rw [slice_next_eq, dif_pos hlt]
  simp only [bind_tc_ok]
  simp
  intro hc
  exfalso
  have hz : (0#i64).val = (0 : ℤ) := by simp
  rw [gt_iff_lt, IScalar.lt_equiv, hz] at h
  exact h hc

/-- The loop 0 body at the end of the slice. -/
theorem l0_body_eq_done (it : core.slice.iter.Iter Std.I64) (any : Bool)
    (hge : it.slice.val.length ≤ it.i) :
    fold_port.stats.basis.rate_words_loop0.body it any = ok (ControlFlow.done any) := by
  unfold fold_port.stats.basis.rate_words_loop0.body
  rw [slice_next_eq, dif_neg (Nat.not_lt.mpr hge)]
  simp only [bind_tc_ok]
  simp

/-- **The `rate_words` loop 0 refinement**: the flag is on exactly when some word is positive. -/
theorem rate_words_loop0_spec (s : Slice Std.I64) :
    fold_port.stats.basis.rate_words_loop0 ⟨s, 0⟩ false ⦃ x =>
      (x = true ↔ anyPos s.val) ⦄ := by
  unfold fold_port.stats.basis.rate_words_loop0
  apply loop.spec_decr_nat
    (measure := fun st : core.slice.iter.Iter Std.I64 × Bool =>
      st.1.slice.val.length - st.1.i)
    (inv := fun st => st.1.slice = s ∧ st.1.i ≤ s.val.length ∧
      (st.2 = true ↔ anyPos (s.val.take st.1.i)))
    (post := fun x : Bool => (x = true ↔ anyPos s.val))
  · rintro ⟨it, any⟩ ⟨hslice, hit, hflag⟩
    dsimp only at hslice hit hflag ⊢
    by_cases hlt : it.i < it.slice.val.length
    · subst hslice
      have htk : it.slice.val.take (it.i + 1)
          = it.slice.val.take it.i ++ [it.slice.val[it.i]] :=
        List.take_succ_eq_append_getElem hlt
      by_cases hpos : it.slice.val[it.i] > 0#i64
      · rw [l0_body_eq_cont_pos it any hlt hpos, WP.spec_ok]
        dsimp only
        refine ⟨⟨rfl, by omega, ?_⟩, by omega⟩
        refine ⟨fun _ => ?_, fun _ => rfl⟩
        rw [htk]
        exact (anyPos_append_singleton _ _).2
          (Or.inr (by rw [gt_iff_lt, IScalar.lt_equiv] at hpos; exact hpos))
      · rw [l0_body_eq_cont_neg it any hlt hpos, WP.spec_ok]
        dsimp only
        refine ⟨⟨rfl, by omega, ?_⟩, by omega⟩
        rw [htk, anyPos_append_singleton]
        refine ⟨fun h => Or.inl (hflag.1 h), ?_⟩
        rintro (h | h)
        · exact hflag.2 h
        · exfalso
          rw [gt_iff_lt, IScalar.lt_equiv] at hpos
          exact hpos h
    · subst hslice
      rw [l0_body_eq_done it any (Nat.le_of_not_lt hlt), WP.spec_ok]
      dsimp only
      have htake : it.slice.val.take it.i = it.slice.val := by
        rw [show it.i = it.slice.val.length by omega, List.take_of_length_le (le_refl _)]
      rw [htake] at hflag
      exact hflag
  · refine ⟨rfl, Nat.zero_le _, ?_⟩
    exact ⟨fun h => Bool.noConfusion h, fun h => absurd h anyPos_nil⟩

/-- **The non-positive branch of `rate_words`**: `Ok i64::MAX` when some word is positive, `Ok 0`
otherwise. -/
theorem rate_words_nonpos (lambda : Slice Std.I64) (theta : Std.I64) (hθ : theta.val ≤ 0) :
    (anyPos lambda.val → fold_port.stats.basis.rate_words lambda theta
        = ok (core.result.Result.Ok core.num.I64.MAX)) ∧
    (¬ anyPos lambda.val → fold_port.stats.basis.rate_words lambda theta
        = ok (core.result.Result.Ok 0#i64)) := by
  unfold fold_port.stats.basis.rate_words
  rw [if_pos (by rw [IScalar.le_equiv]; simpa using hθ)]
  simp only [SharedSlice.Insts.CoreIterTraitsCollectIntoIteratorSharedIter.into_iter,
    bind_tc_ok]
  obtain ⟨any, hany, hflag⟩ := (WP.spec_equiv_exists _ _).1 (rate_words_loop0_spec lambda)
  rw [hany]
  simp only [bind_tc_ok]
  constructor
  · intro hp
    rw [if_pos (hflag.2 hp)]
  · intro hp
    rw [if_neg (fun h => hp (hflag.1 h))]

/-! ### 8.2 · The positive branch of `rate_words`

With `hlog : ∀ l, log2 l = ok (some (L l))` and `ht : L θ = t`, the body reads
`t = log2 θ`, runs `rate_words_loop1`, and reduces to the `>>> 1`/`narrow` tail. `rate_words_pos_eq`
is that reduction: the tail is the extracted `narrow`/`ok_or` composition on the exact half-sum.
`rate_words_pos_spec` states the `Ok`/`Err` dichotomy of that composition, and
`rate_words_pos_fail` the `OutOfRange` of a failing `checked_sub`.

The per-word `log2` values stay the caller's hypothesis: `fixed_ln`'s series is not refined here, so
no theorem states the real value `Σ max(0, ½ log₂(λ/θ))`.
-/

/-- **The positive branch of `rate_words`**, reduced: the loop either sets the failure flag, and the
body returns `OutOfRange`, or it returns the exact `>>> 1`/`narrow` tail of the active-log sum. -/
theorem rate_words_pos_eq (lambda : Slice Std.I64) (theta : Std.I64)
    (logv : Std.I64 → Std.I64) (t : Std.I64)
    (hθ : 0 < theta.val)
    (hlog : ∀ l, fold_port.stats.basis.log2 l = ok (some (logv l)))
    (ht : logv theta = t)
    (hmono : ∀ l, theta.val < l.val → 0 ≤ (logv l).val - t.val)
    (hnbad : ¬ rateBad logv theta t.val lambda.val)
    (hbound : rateAt logv theta t.val lambda.val ≤ 2 ^ 126) :
    ∃ z : Std.I128, z.val = rateAt logv theta t.val lambda.val / 2 ∧
      fold_port.stats.basis.rate_words lambda theta
        = (do
            let o ← solve_port.q32.narrow z
            core.option.Option.ok_or o fold_port.stats.basis.BasisError.OutOfRange) := by
  have h0i : (0#i128).val = (0 : ℤ) := by simp
  have hbound' : (0#i128).val + rateAt logv theta t.val lambda.val ≤ 2 ^ 126 := by
    rw [h0i]; simpa using hbound
  obtain ⟨p, hp, hpost⟩ := (WP.spec_equiv_exists _ _).1
    (rate_words_loop1_spec lambda theta t logv 0#i128 hlog hmono hbound')
  obtain ⟨acc, failed⟩ := p
  dsimp only at hp hpost
  have hf : failed = false := by
    cases failed with
    | false => rfl
    | true => exact absurd (hpost.1.1 rfl) hnbad
  have hacc : acc.val = rateAt logv theta t.val lambda.val := by
    have h := hpost.2 (by rw [hf])
    simpa using h
  obtain ⟨z, hz, hzval⟩ := shiftRight_one_spec acc
  refine ⟨z, ?_, ?_⟩
  · rw [hzval, hacc]
  · unfold fold_port.stats.basis.rate_words
    have h0 : (0#i64).val = (0 : ℤ) := by simp
    rw [if_neg (by rw [IScalar.le_equiv]; omega), hlog theta, ht]
    simp only [core.option.Option.ok_or_some, bind_tc_ok]
    have hb : core.result.Result.Insts.CoreOpsTry.branch
          (core.result.Result.Ok t)
        = ok (core.ops.control_flow.ControlFlow.Continue
            (B := core.result.Result core.convert.Infallible
              fold_port.stats.basis.BasisError) t) := rfl
    rw [hb]
    simp only [bind_tc_ok]
    rw [SharedSlice.Insts.CoreIterTraitsCollectIntoIteratorSharedIter.into_iter]
    simp only [bind_tc_ok]
    rw [hp]
    simp only [bind_tc_ok]
    rw [hf]
    simp
    rw [hz]
    simp only [bind_tc_ok]

/-- **The positive branch on the no-failure window**: the result is `Ok v` exactly when `narrow`
returns `v` on the half-sum, and `OutOfRange` when it refuses. -/
theorem rate_words_pos_spec (lambda : Slice Std.I64) (theta : Std.I64)
    (logv : Std.I64 → Std.I64) (t : Std.I64)
    (hθ : 0 < theta.val)
    (hlog : ∀ l, fold_port.stats.basis.log2 l = ok (some (logv l)))
    (ht : logv theta = t)
    (hmono : ∀ l, theta.val < l.val → 0 ≤ (logv l).val - t.val)
    (hnbad : ¬ rateBad logv theta t.val lambda.val)
    (hbound : rateAt logv theta t.val lambda.val ≤ 2 ^ 126) :
    ∃ z : Std.I128, z.val = rateAt logv theta t.val lambda.val / 2 ∧
      (∀ v : Std.I64, solve_port.q32.narrow z = ok (some v) →
        fold_port.stats.basis.rate_words lambda theta
          = ok (core.result.Result.Ok v)) ∧
      (solve_port.q32.narrow z = ok none →
        fold_port.stats.basis.rate_words lambda theta
          = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange)) := by
  obtain ⟨z, hz, heq⟩ :=
    rate_words_pos_eq lambda theta logv t hθ hlog ht hmono hnbad hbound
  refine ⟨z, hz, ?_, ?_⟩
  · intro v hn
    rw [heq, hn]
    simp
  · intro hn
    rw [heq, hn]
    simp

/-- **The positive branch on a failing `checked_sub`**: any active word whose log difference leaves
the `i64` window makes the body return `OutOfRange`. -/
theorem rate_words_pos_fail (lambda : Slice Std.I64) (theta : Std.I64)
    (logv : Std.I64 → Std.I64) (t : Std.I64)
    (hθ : 0 < theta.val)
    (hlog : ∀ l, fold_port.stats.basis.log2 l = ok (some (logv l)))
    (ht : logv theta = t)
    (hmono : ∀ l, theta.val < l.val → 0 ≤ (logv l).val - t.val)
    (hbad : rateBad logv theta t.val lambda.val)
    (hbound : rateAt logv theta t.val lambda.val ≤ 2 ^ 126) :
    fold_port.stats.basis.rate_words lambda theta
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  have h0i : (0#i128).val = (0 : ℤ) := by simp
  have hbound' : (0#i128).val + rateAt logv theta t.val lambda.val ≤ 2 ^ 126 := by
    rw [h0i]; simpa using hbound
  obtain ⟨p, hp, hpost⟩ := (WP.spec_equiv_exists _ _).1
    (rate_words_loop1_spec lambda theta t logv 0#i128 hlog hmono hbound')
  obtain ⟨acc, failed⟩ := p
  dsimp only at hp hpost
  have hf : failed = true := hpost.1.2 hbad
  unfold fold_port.stats.basis.rate_words
  have h0 : (0#i64).val = (0 : ℤ) := by simp
  rw [if_neg (by rw [IScalar.le_equiv]; omega), hlog theta, ht]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  have hb : core.result.Result.Insts.CoreOpsTry.branch
        (core.result.Result.Ok t)
      = ok (core.ops.control_flow.ControlFlow.Continue
          (B := core.result.Result core.convert.Infallible
            fold_port.stats.basis.BasisError) t) := rfl
  rw [hb]
  simp only [bind_tc_ok]
  rw [SharedSlice.Insts.CoreIterTraitsCollectIntoIteratorSharedIter.into_iter]
  simp only [bind_tc_ok]
  rw [hp]
  simp only [bind_tc_ok]
  rw [hf]
  simp

/-- **The positive branch where `log2 theta` refuses**: an absent log word makes the body return
`OutOfRange` before the slice loop runs. -/
theorem rate_words_pos_none (lambda : Slice Std.I64) (theta : Std.I64)
    (hθ : 0 < theta.val) (hnone : fold_port.stats.basis.log2 theta = ok none) :
    fold_port.stats.basis.rate_words lambda theta
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  unfold fold_port.stats.basis.rate_words
  have h0 : (0#i64).val = (0 : ℤ) := by simp
  rw [if_neg (by rw [IScalar.le_equiv]; omega), hnone]
  simp only [bind_tc_ok, core.option.Option.ok_or_none]
  have hb : core.result.Result.Insts.CoreOpsTry.branch (T := Std.I64)
        (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange)
      = ok (core.ops.control_flow.ControlFlow.Break
          (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange)) := rfl
  rw [hb]
  simp only [bind_tc_ok,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rfl

/-! ## 9 · The whole `water_fill_words` body, branch by branch

`fold_port.stats.basis.water_fill_words` (`basis.rs:965`) is the composition of the two guards
(`budget − side` and `|lambda|` as `i64`), loop 0, the two early zeros, loop 1 and the `t ≤ 0`
return, the two `log2` closures, loop 2, the tolerance product, loop 3, `exp2`, then
`distortion_words` and `rate_words` on the returned level. Each branch below states the exact
`Result` that branch returns; `water_fill_words_spec` collects them all and
`solve_water_fill_spec` is the public wrapper.

The `log2`/`exp2` values and the loop 2/loop 3 hypotheses stay the caller's: no real
`D(θ)`/`R(θ)` value, no `exp2`/`fixed_ln` accuracy and no allocation claim is made here.
-/

/-- `Result::branch` on `Ok` is `Continue`, definitionally. -/
theorem branch_ok {T E : Type} (v : T) :
    core.result.Result.Insts.CoreOpsTry.branch (E := E) (core.result.Result.Ok v)
      = ok (core.ops.control_flow.ControlFlow.Continue v) := rfl

/-- `Result::branch` on `Err` is `Break (Err e)`, definitionally. -/
theorem branch_err {T E : Type} (e : E) :
    core.result.Result.Insts.CoreOpsTry.branch (T := T) (core.result.Result.Err e)
      = ok (core.ops.control_flow.ControlFlow.Break (core.result.Result.Err e)) := rfl

/-- The `branch` of an `Err` on the body's intermediate `i64` results: `Break (Err e)`. -/
theorem branch_err_i64 (e : fold_port.stats.basis.BasisError) :
    core.result.Result.Insts.CoreOpsTry.branch (T := Std.I64)
        (core.result.Result.Err e)
      = ok (core.ops.control_flow.ControlFlow.Break (core.result.Result.Err e)) := rfl

/-- The body's `from_residual` on an `Err` residual with the identity conversion returns
`Ok (Err e)`. -/
theorem from_residual_same (T E : Type) (e : E) :
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual T
        (core.convert.FromSame E) (core.result.Result.Err e)
      = ok (core.result.Result.Err e) := by
  simp only [
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    bind_tc_ok]
  rfl

/-- **Branch 1**: a failing `budget − side` returns `OutOfRange`. -/
theorem water_fill_words_sub_none (lambda : Slice Std.I64) (budget side : Std.I64)
    (hsub : I64.checked_sub budget side = none) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_none, bind_tc_ok, branch_err,
    from_residual_same]

/-- **Branch 2**: a `|lambda|` outside the `i64` window returns `Overflow`. -/
theorem water_fill_words_len_overflow (lambda : Slice Std.I64) (budget side t : Std.I64)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Err ())) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.Overflow) := by
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once,
    branch_err, from_residual_same]

/-- **Branch 3**: an error inside loop 0 is returned unchanged. -/
theorem water_fill_words_loop0_err (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64)
    (e : fold_port.stats.basis.BasisError)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, some e)) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e) := by
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp

/-- **Branch 4**: an empty word list returns the zero triple. -/
theorem water_fill_words_empty (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hempty : lambda.val = []) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := 0#i64, distortion := 0#i64, rate := 0#i64 }) := by
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [bind_tc_ok]
  simp [core.slice.Slice.is_empty, Slice.length, hempty]

/-- **Branch 5**: a non-positive maximum returns the zero triple. -/
theorem water_fill_words_max_nonpos (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ []) (hmax : mx.val ≤ 0) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := 0#i64, distortion := 0#i64, rate := 0#i64 }) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  intro h
  omega

/-- **Branch 6a**: a non-positive `t` returns the maximum level with the rounded sum, at rate
zero, when `narrow` accepts. -/
theorem water_fill_words_t_nonpos_some
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (d : Std.I64)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : t.val ≤ 0) (hnarrow : solve_port.q32.narrow sum = ok (some d)) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := mx, distortion := d, rate := 0#i64 }) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_pos (by omega)]
  rw [hnarrow]
  simp only [bind_tc_ok, core.option.Option.unwrap_or_some]

/-- **Branch 6b**: the same return with `i64::MAX` when `narrow` refuses. -/
theorem water_fill_words_t_nonpos_none
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : t.val ≤ 0) (hnarrow : solve_port.q32.narrow sum = ok none) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := mx, distortion := core.num.I64.MAX, rate := 0#i64 }) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_pos (by omega)]
  rw [hnarrow]
  simp only [bind_tc_ok, core.option.Option.unwrap_or_none]

/-- **Branch 7**: an absent `log2 max` returns `OutOfRange`. -/
theorem water_fill_words_logmax_none     (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (hlogmax : fold_port.stats.basis.log2 mx = ok none) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  simp [core.option.Option.and_then, core.option.Option.ok_or_none, branch_err,
    from_residual_same]

/-- **Branch 8**: a failing `log2 max + 2^32` returns `OutOfRange`. -/
theorem water_fill_words_logmax_add_none     (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok none) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_none, branch_err, from_residual_same]

/-- **Branch 9**: an absent `log2 min_pos` returns `OutOfRange`. -/
theorem water_fill_words_logmin_none     (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok none) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  simp [core.option.Option.and_then, core.option.Option.ok_or_none, branch_err,
    from_residual_same]

/-- **Branch 10**: a failing `log2 min_pos - 2^32` returns `OutOfRange`. -/
theorem water_fill_words_logmin_sub_none     (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi lmin : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hand2 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok none) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  rw [hand2]
  simp only [bind_tc_ok, core.option.Option.ok_or_none, branch_err, from_residual_same]

/-- **Branch 11**: a loop 2 error is returned unchanged. -/
theorem water_fill_words_loop2_err     (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi lmin lsub one tlo : Std.I64)
    (e1 : fold_port.stats.basis.BasisError)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hand2 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub))
    (hgap : fixed_ln.ONE = ok one)
    (hloop2 : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, some e1))
    :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e1) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  rw [hand2]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hgap]
  simp only [bind_tc_ok]
  rw [hloop2]
  simp only [bind_tc_ok]
  simp

-- Branch 12: a failing `RATE_TOL_Q * |lambda|` returns `Overflow`.
theorem water_fill_words_mul_none     (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi lmin lsub one tlo : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hand2 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub))
    (hgap : fixed_ln.ONE = ok one)
    (hloop2 : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none))
    (hmul : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = none) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.Overflow) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  rw [hand2]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hgap]
  simp only [bind_tc_ok]
  rw [hloop2]
  simp only [bind_tc_ok]
  rw [hmul]
  simp only [bind_tc_ok, core.option.Option.ok_or_none]
  rw [branch_err_i64]
  simp only [bind_tc_ok,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rfl

/-- **Branch 13**: a loop 3 error is returned unchanged. -/
theorem water_fill_words_loop3_err
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi lmin lsub one tlo : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hand2 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub))
    (hgap : fixed_ln.ONE = ok one)
    (target tt : Std.I64) (e2 : fold_port.stats.basis.BasisError)
    (hloop2 : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none))
    (hmul : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target)
    (hloop3 : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, some e2)) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e2) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  rw [hand2]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hgap]
  simp only [bind_tc_ok]
  rw [hloop2]
  simp
  rw [hmul]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hloop3]
  simp

/-- **Branch 14**: an absent `exp2 t` returns `OutOfRange`. -/
theorem water_fill_words_exp2_none
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi lmin lsub one tlo : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hand2 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub))
    (hgap : fixed_ln.ONE = ok one)
    (target tt : Std.I64)
    (hloop2 : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none))
    (hmul : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target)
    (hloop3 : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none))
    (hexp : fold_port.stats.basis.exp2 tt = ok none) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  rw [hand2]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hgap]
  simp only [bind_tc_ok]
  rw [hloop2]
  simp
  rw [hmul]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hloop3]
  simp
  rw [hexp]
  simp only [bind_tc_ok, core.option.Option.ok_or_none]
  rw [branch_err_i64]
  simp only [bind_tc_ok,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rfl

/-- **Branch 15**: a `distortion_words` error is returned unchanged. -/
theorem water_fill_words_dist_err
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi lmin lsub one tlo : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hand2 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub))
    (hgap : fixed_ln.ONE = ok one)
    (target tt : Std.I64) (e3 : fold_port.stats.basis.BasisError)
    (hloop2 : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none))
    (hmul : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target)
    (hloop3 : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none))
    (hexp : fold_port.stats.basis.exp2 tt = ok (some one))
    (hdist : fold_port.stats.basis.distortion_words lambda one
      = ok (core.result.Result.Err e3)) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e3) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  rw [hand2]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hgap]
  simp only [bind_tc_ok]
  rw [hloop2]
  simp
  rw [hmul]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hloop3]
  simp
  rw [hexp]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hdist]
  simp only [bind_tc_ok]
  rw [branch_err_i64]
  simp only [bind_tc_ok,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rfl


/-- **Branch 16**: a `rate_words` error is returned unchanged. -/
theorem water_fill_words_rate_err
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi lmin lsub one tlo : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hand2 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub))
    (hgap : fixed_ln.ONE = ok one)
    (th : Std.I64)
    (target tt : Std.I64) (r : Std.I64)
    (hloop2 : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none))
    (hmul : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target)
    (hloop3 : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none))
    (hexp : fold_port.stats.basis.exp2 tt = ok (some th))
    (e4 : fold_port.stats.basis.BasisError)
    (hdist : fold_port.stats.basis.distortion_words lambda th
      = ok (core.result.Result.Ok r))
    (hrate : fold_port.stats.basis.rate_words lambda th
      = ok (core.result.Result.Err e4)) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e4) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  rw [hand2]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hgap]
  simp only [bind_tc_ok]
  rw [hloop2]
  simp
  rw [hmul]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hloop3]
  simp
  rw [hexp]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hdist]
  simp only [bind_tc_ok]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hrate]
  simp only [bind_tc_ok]
  rw [branch_err_i64]
  simp only [bind_tc_ok,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rfl

/-- **Branch 17**: the success branch returns `{theta, distortion, rate}` from the two word
bodies at the `exp2` level. -/
theorem water_fill_words_ok
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hnempty : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (lm t_hi lmin lsub one tlo : Std.I64)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hand1 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hand2 : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub))
    (hgap : fixed_ln.ONE = ok one)
    (th : Std.I64)
    (target tt d r : Std.I64)
    (hloop2 : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none))
    (hmul : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target)
    (hloop3 : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none))
    (hexp : fold_port.stats.basis.exp2 tt = ok (some th))
    (hdist : fold_port.stats.basis.distortion_words lambda th
      = ok (core.result.Result.Ok d))
    (hrate : fold_port.stats.basis.rate_words lambda th
      = ok (core.result.Result.Ok r)) :
    fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := th, distortion := d, rate := r }) := by
  have his : core.slice.Slice.is_empty lambda = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hnempty]
  unfold fold_port.stats.basis.water_fill_words
  rw [hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok, branch_ok]
  rw [hlen]
  simp only [bind_tc_ok, core.result.Result.map_err,
    fold_port.stats.basis.water_fill_words.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorBasisError.call_once]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogs]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [his, bind_tc_ok]
  simp
  rw [if_neg (by omega)]
  rw [hloop1]
  simp only [bind_tc_ok]
  rw [if_neg (by omega)]
  rw [hlogmax]
  simp only [bind_tc_ok]
  rw [hand1]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hlogmin]
  simp only [bind_tc_ok]
  rw [hand2]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hgap]
  simp only [bind_tc_ok]
  rw [hloop2]
  simp
  rw [hmul]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hloop3]
  simp
  rw [hexp]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hdist]
  simp only [bind_tc_ok]
  rw [branch_ok]
  simp only [bind_tc_ok]
  rw [hrate]
  simp only [bind_tc_ok]
  rw [branch_ok]
  simp only [bind_tc_ok]

/-! ### `water_fill_words_spec` — every branch of the body

The eighteen branches below are the guards (`budget − side`, `|lambda|`), loop 0's error, the two
early zeros, the `t ≤ 0` return with `narrow` accepting or refusing, the two `log2` closures
(absent input and failing `checked_add`/`checked_sub`), loop 2's error, the tolerance product's
overflow, loop 3's error, `exp2`'s absence, the `distortion_words` and `rate_words` errors, and the
success return. Each conjunct is the corresponding branch theorem; the hypothesis list is the step
value the branch consumes, so a caller that supplies the extracted step values reads off the exact
`Result`. -/
theorem water_fill_words_spec (lambda : Slice Std.I64) (budget side : Std.I64) :
    (∀ (_ : I64.checked_sub budget side = none), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange))
  ∧
    (∀ (t : Std.I64) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Err ())), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.Overflow))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (e : fold_port.stats.basis.BasisError) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, some e)), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val = []), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := 0#i64, distortion := 0#i64, rate := 0#i64 }))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : mx.val ≤ 0), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := 0#i64, distortion := 0#i64, rate := 0#i64 }))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (d : Std.I64) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : t.val ≤ 0) (_ : solve_port.q32.narrow sum = ok (some d)), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := mx, distortion := d, rate := 0#i64 }))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : t.val ≤ 0) (_ : solve_port.q32.narrow sum = ok none), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := mx, distortion := core.num.I64.MAX, rate := 0#i64 }))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (_ : fold_port.stats.basis.log2 mx = ok none), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok none), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok none), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi lmin : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok (some lmin)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok none), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi lmin lsub one tlo : Std.I64) (e1 : fold_port.stats.basis.BasisError) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok (some lmin)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub)) (_ : fixed_ln.ONE = ok one) (_ : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, some e1)), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e1))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi lmin lsub one tlo : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok (some lmin)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub)) (_ : fixed_ln.ONE = ok one) (_ : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none)) (_ : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = none), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.Overflow))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi lmin lsub one tlo : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok (some lmin)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub)) (_ : fixed_ln.ONE = ok one) (target tt : Std.I64) (e2 : fold_port.stats.basis.BasisError) (_ : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none)) (_ : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target) (_ : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, some e2)), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e2))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi lmin lsub one tlo : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok (some lmin)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub)) (_ : fixed_ln.ONE = ok one) (target tt : Std.I64) (_ : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none)) (_ : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target) (_ : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none)) (_ : fold_port.stats.basis.exp2 tt = ok none), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err fold_port.stats.basis.BasisError.OutOfRange))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi lmin lsub one tlo : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok (some lmin)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub)) (_ : fixed_ln.ONE = ok one) (target tt : Std.I64) (e3 : fold_port.stats.basis.BasisError) (_ : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none)) (_ : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target) (_ : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none)) (_ : fold_port.stats.basis.exp2 tt = ok (some one)) (_ : fold_port.stats.basis.distortion_words lambda one
      = ok (core.result.Result.Err e3)), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e3))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi lmin lsub one tlo : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok (some lmin)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub)) (_ : fixed_ln.ONE = ok one) (th : Std.I64) (target tt : Std.I64) (r : Std.I64) (_ : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none)) (_ : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target) (_ : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none)) (_ : fold_port.stats.basis.exp2 tt = ok (some th)) (e4 : fold_port.stats.basis.BasisError) (_ : fold_port.stats.basis.distortion_words lambda th
      = ok (core.result.Result.Ok r)) (_ : fold_port.stats.basis.rate_words lambda th
      = ok (core.result.Result.Err e4)), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Err e4))
  ∧
    (∀ (t n : Std.I64) (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128) (_ : I64.checked_sub budget side = some t) (_ : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n)) (_ : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg) (_ : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none)) (_ : lambda.val ≠ []) (_ : 0 < mx.val) (_ : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum) (_ : 0 < t.val) (lm t_hi lmin lsub one tlo : Std.I64) (_ : fold_port.stats.basis.log2 mx = ok (some lm)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi)) (_ : fold_port.stats.basis.log2 mp = ok (some lmin)) (_ : core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub)) (_ : fixed_ln.ONE = ok one) (th : Std.I64) (target tt d r : Std.I64) (_ : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none)) (_ : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target) (_ : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none)) (_ : fold_port.stats.basis.exp2 tt = ok (some th)) (_ : fold_port.stats.basis.distortion_words lambda th
      = ok (core.result.Result.Ok d)) (_ : fold_port.stats.basis.rate_words lambda th
      = ok (core.result.Result.Ok r)), fold_port.stats.basis.water_fill_words lambda budget side
      = ok (core.result.Result.Ok
          { theta := th, distortion := d, rate := r })) := by
  exact ⟨water_fill_words_sub_none lambda budget side, water_fill_words_len_overflow lambda budget side, water_fill_words_loop0_err lambda budget side, water_fill_words_empty lambda budget side, water_fill_words_max_nonpos lambda budget side, water_fill_words_t_nonpos_some lambda budget side, water_fill_words_t_nonpos_none lambda budget side, water_fill_words_logmax_none lambda budget side, water_fill_words_logmax_add_none lambda budget side, water_fill_words_logmin_none lambda budget side, water_fill_words_logmin_sub_none lambda budget side, water_fill_words_loop2_err lambda budget side, water_fill_words_mul_none lambda budget side, water_fill_words_loop3_err lambda budget side, water_fill_words_exp2_none lambda budget side, water_fill_words_dist_err lambda budget side, water_fill_words_rate_err lambda budget side, water_fill_words_ok lambda budget side⟩


/-! ### The public wrapper `solve_water_fill` -/

/-- A successful `water_fill_words` passes its triple through `solve_water_fill`. -/
theorem solve_water_fill_ok (lambda : Slice Std.I64) (budget_bits side_bits : Std.I64)
    (w : fold_port.stats.basis.WaterFill)
    (h : fold_port.stats.basis.water_fill_words lambda budget_bits side_bits
      = ok (core.result.Result.Ok w)) :
    fold_port.stats.basis.solve_water_fill lambda budget_bits side_bits = ok w := by
  unfold fold_port.stats.basis.solve_water_fill
  rw [h]
  simp only [bind_tc_ok]

/-- Every `water_fill_words` error becomes the zero triple in `solve_water_fill`. -/
theorem solve_water_fill_err (lambda : Slice Std.I64) (budget_bits side_bits : Std.I64)
    (e : fold_port.stats.basis.BasisError)
    (h : fold_port.stats.basis.water_fill_words lambda budget_bits side_bits
      = ok (core.result.Result.Err e)) :
    fold_port.stats.basis.solve_water_fill lambda budget_bits side_bits
      = ok { theta := 0#i64, distortion := 0#i64, rate := 0#i64 } := by
  unfold fold_port.stats.basis.solve_water_fill
  rw [h]
  simp only [bind_tc_ok]

/-! ### `water_fill_words_ok_composed` — branch 17 with the four step equalities derived

`water_fill_words_ok` takes loop 0's, loop 1's, `distortion_words`' and `rate_words`' results as
step equalities. Here they are derived instead: loop 0 from `water_fill_words_loop0_spec` at
`{0,|lambda|}`, loop 1 from `water_fill_words_loop1_spec` at the same range and the zero `i128`
accumulator, the distortion from the `0 < theta` window of `distortion_words_post`, and the rate from
`rate_words_pos_spec` at `theta` with `0 < theta` and the rounding step's `narrow` result as data.

Loop 0's output vector is not determined by the spec's post (`alloc.vec.Vec` carries a `cap` field
that `vals` does not see), so the later calls are supplied as one caller fact quantified over loop 0's
own output rather than over a caller-named vector. The `log2`/`fixed_ln`/`exp2` values stay the
caller's; no accuracy claim is made. -/
theorem water_fill_words_ok_composed
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg : alloc.vec.Vec Std.I64) (wv : ℤ → ℤ)
    (lm t_hi lmin lsub one tlo target tt th r : Std.I64)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hlog0 : ∀ (k : ℕ) (hk : k < lambda.val.length),
      ∃ t' : Std.I64, fold_port.stats.basis.water_log lambda.val[k]
        = ok (core.result.Result.Ok t') ∧ t'.val = wv lambda.val[k].val)
    (hlg0 : vals lg.val = List.replicate lambda.val.length 0)
    (hne : lambda.val ≠ [])
    (hmaxm : 0 < maxOf (vals lambda.val))
    (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (htot1 : lamSuffix lambda.val 0 ≤ 2 ^ 126)
    (ht : 0 < t.val)
    (hdown : ∀ (mx mp : Std.I64) (lg1 : alloc.vec.Vec Std.I64),
      fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none) →
      fold_port.stats.basis.log2 mx = ok (some lm) ∧
      core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lm) () = ok (some t_hi) ∧
      fold_port.stats.basis.log2 mp = ok (some lmin) ∧
      core.option.Option.and_then
        fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
        (some lmin) () = ok (some lsub) ∧
      fixed_ln.ONE = ok one ∧
      fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none) ∧
      I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target ∧
      fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none) ∧
      fold_port.stats.basis.exp2 tt = ok (some th))
    (hθ : 0 < th.val)
    (hfit : distList th.val lambda.val ≤ 2 ^ 63 - 1)
    (htotd : distList th.val lambda.val ≤ 2 ^ 126)
    (logv : Std.I64 → Std.I64) (tr : Std.I64)
    (hlogr : ∀ l, fold_port.stats.basis.log2 l = ok (some (logv l)))
    (htr : logv th = tr)
    (hmono : ∀ l, th.val < l.val → 0 ≤ (logv l).val - tr.val)
    (hnbad : ¬ rateBad logv th tr.val lambda.val)
    (hbound : rateAt logv th tr.val lambda.val ≤ 2 ^ 126)
    (z : Std.I128) (hz : z.val = rateAt logv th tr.val lambda.val / 2)
    (hrn : solve_port.q32.narrow z = ok (some r)) :
    ∃ (sum : Std.I128) (d : Std.I64),
      fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum ∧
      fold_port.stats.basis.distortion_words lambda th = ok (core.result.Result.Ok d) ∧
      fold_port.stats.basis.water_fill_words lambda budget side
        = ok (core.result.Result.Ok { theta := th, distortion := d, rate := r }) := by
  have hlenv : (Slice.len lambda).val = lambda.val.length := Slice.len_val lambda
  have h0i : (0#i64).val = (0 : ℤ) := by simp
  have h0u : (0#usize).val = (0 : ℕ) := by simp
  have hmaxi : (core.num.I64.MAX).val = I64_MAX := by
    unfold I64_MAX
    norm_num [core.num.I64.MAX, I64.rMax]
  obtain ⟨x, hx, h1, h2, h3, h4⟩ :=
    (WP.spec_equiv_exists _ _).1
      (water_fill_words_loop0_spec lambda wv
        { start := 0#usize, «end» := Slice.len lambda } 0#i64 core.num.I64.MAX lg
        (by rw [hlenv])
        (by simp)
        (fun k hk => hlog0 k (by rw [hlenv] at hk; exact hk))
        (by
          rw [h0i, h0u]
          show (0 : ℤ) = maxOf (vals (lambda.val.take 0))
          simp [maxOf, vals])
        (by
          rw [hmaxi, h0u]
          show I64_MAX = minPosOf (vals (lambda.val.take 0))
          simp [minPosOf, vals])
        (by
          rw [h0u, hlg0]
          exact (logsOf_zero wv lambda.val).symm))
  rcases x with ⟨mx, mp, lg1, er⟩
  have hver : er = none := h3
  have hloop0 : fold_port.stats.basis.water_fill_words_loop0
      { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
      lg none = ok (mx, mp, lg1, none) := by
    simpa [hver] using hx
  obtain ⟨hlm, hd1, hlm2, hd2, hgap1, hloop2, hmul, hloop3, hexp⟩ :=
    hdown mx mp lg1 hloop0
  rw [hlenv] at h1
  simp only [List.take_length] at h1
  have hmax : 0 < mx.val := by rw [h1]; exact hmaxm
  have hacc : 0 ≤ (0#i128).val := by simp
  have htot : (0#i128).val + lamSuffix lambda.val 0 ≤ 2 ^ 126 := by
    have h0 : (0#i128).val = 0 := by simp
    rw [h0]
    simpa using htot1
  obtain ⟨sum, hsum, hsumv⟩ :=
    (WP.spec_equiv_exists _ _).1
      (water_fill_words_loop1_spec lambda
        { start := 0#usize, «end» := Slice.len lambda } 0#i128
        (by rw [hlenv]) (by simp) hL hacc htot)
  obtain ⟨d, hd, hdv⟩ :=
    ((distortion_words_post lambda th hL htotd).2 hθ).1 hfit
  obtain ⟨z', hz', hzr, hzn⟩ :=
    rate_words_pos_spec lambda th logv tr hθ hlogr htr hmono hnbad hbound
  have hzz : z' = z := IScalar.eq_of_val_eq (by rw [hz', hz])
  have hrate : fold_port.stats.basis.rate_words lambda th
      = ok (core.result.Result.Ok r) := by
    have hrn' : solve_port.q32.narrow z' = ok (some r) := by rw [hzz]; exact hrn
    exact hzr r hrn'
  exact ⟨sum, d, hsum, hd,
    water_fill_words_ok lambda budget side t n lg lg1 mx mp sum
      hsub hlen hlogs hloop0 hne hmax hsum ht
      lm t_hi lmin lsub one tlo hlm hd1 hlm2 hd2 hgap1
      th target tt d r hloop2 hmul hloop3 hexp hd hrate⟩

/-! ### `closure_1` and `closure_2` as derived steps

The body calls `log2 max`, then `and_then closure_1` on its `Option`, then `ok_or`/`branch` on the
result. `closure_1` adds `fixed_ln.ONE`, `closure_2` subtracts it. These two lemmas derive the
`and_then` step from the `fixed_ln.ONE` result and the `checked_add`/`checked_sub` result, so the
composition no longer carries the `and_then` equations as hypotheses. -/
theorem closure_1_step (lm one t_hi : Std.I64) (hg : fixed_ln.ONE = ok one)
    (hadd : I64.checked_add lm one = some t_hi) :
    core.option.Option.and_then
      fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
      (some lm) () = ok (some t_hi) := by
  simp only [core.option.Option.and_then,
    fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64.call_once,
    hg, bind_tc_ok, hadd]

theorem closure_2_step (lmin one lsub : Std.I64) (hg : fixed_ln.ONE = ok one)
    (hsub : I64.checked_sub lmin one = some lsub) :
    core.option.Option.and_then
      fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
      (some lmin) () = ok (some lsub) := by
  simp only [core.option.Option.and_then,
    fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64.call_once,
    hg, bind_tc_ok, hsub]

/-! ### The two bisection loops, derived from their specs

`water_fill_words_loop2_spec` and `water_fill_words_loop3_spec` are `WP` triples. A `WP` triple
does not name the value the loop returns, so the step equality the body needs is the existence
form `WP.spec_equiv_exists` produces: the loop returns some pair, and the post holds of it. The
two lemmas below state exactly that at the range `{0, 200}` the body uses, with the spec's own
preconditions named. Neither lemma claims anything about `log2`, `exp2` or `fixed_ln.ONE`
accuracy; they carry the spec's post verbatim.
-/

/-- The loop-2 step at `{0, 200}`: from the walk's probe hypothesis, the loop returns some
`(tlo, e)` and the spec's post holds of it. -/
theorem water_fill_words_loop2_at (t : Std.I64) (lg1 : alloc.vec.Vec Std.I64)
    (t0 gap0 : Std.I64)
    (hstart : (0#usize).val ≤ fold_port.stats.basis.BISECTION_ITERS.val)
    (hprobe : ∀ (k : ℕ),
      k ≤ fold_port.stats.basis.BISECTION_ITERS.val - (0#usize).val →
      ∀ (s : Std.I64), s.val = walkLevel t0.val gap0.val k →
      ∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref lg1) s
        = ok (core.result.Result.Ok v)) :
    ∃ y : Std.I64 × Option fold_port.stats.basis.BasisError,
      fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t0 gap0 none false = ok y ∧
      ((∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref lg1) y.1
          = ok (core.result.Result.Ok v) ∧ t.val ≤ v.val) ∨
        y.2 ≠ none ∨
        (y.2 = none ∧
          y.1.val = walkLevel t0.val gap0.val
            (fold_port.stats.basis.BISECTION_ITERS.val - (0#usize).val) ∧
          walkProbesBelow lg1 t t0.val gap0.val
            (fold_port.stats.basis.BISECTION_ITERS.val - (0#usize).val))) :=
  (WP.spec_equiv_exists _ _).1
    (water_fill_words_loop2_spec
      { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
      t lg1 t0 gap0 hstart hprobe)

/-- The loop-3 step at `{0, 200}`: from the bracket, probe and convergence hypotheses, the loop
returns some `(tt, e)` and the spec's post holds of it. -/
theorem water_fill_words_loop3_at (t : Std.I64) (lg1 : alloc.vec.Vec Std.I64)
    (t_hi t_lo tol t0 : Std.I64)
    (hstart : (0#usize).val ≤ fold_port.stats.basis.BISECTION_ITERS.val)
    (hbr : BisectBracket lg1 t t_lo t_hi)
    (hprobe : ∀ (s : Std.I64), ∃ v : Std.I64,
      fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref lg1) s
        = ok (core.result.Result.Ok v))
    (hconv : ∀ (v : Std.I64), ∃ b : Bool,
      fold_port.stats.basis.converged v t tol = ok (core.result.Result.Ok b)) :
    ∃ y : Std.I64 × Option fold_port.stats.basis.BasisError,
      fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi t_lo tol t0 none false = ok y ∧
      (ConvergedAt lg1 t tol y.1 ∨ y.2 ≠ none ∨
        (y.2 = none ∧
          (RejectedAt lg1 t tol y.1 ∨
            ((0#usize).val = fold_port.stats.basis.BISECTION_ITERS.val ∧ y.1 = t0)))) :=
  (WP.spec_equiv_exists _ _).1
    (water_fill_words_loop3_spec
      { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
      t lg1 t_hi t_lo tol t0 hstart hbr hprobe hconv)

/-! ### `water_fill_words_ok_loops` — the composition with both loops and both closures

`water_fill_words_ok_composed` carried the two `log2` closures and `fixed_ln.ONE` as one caller
fact. Here the closures are derived by `closure_1_step`/`closure_2_step` from `fixed_ln.ONE` and
the `checked_add`/`checked_sub` arithmetic, and the loop-2 and loop-3 steps come from
`water_fill_words_loop2_at`/`water_fill_words_loop3_at`. What stays a caller fact is the error
component of each bisection loop (`e = none`): the specs' posts do not determine it, since their
first disjunct leaves `x.2` free. `exp2` stays a call hypothesis; no accuracy claim is made about
`exp2`, `log2` or `fixed_ln.ONE`.
-/
theorem water_fill_words_ok_loops
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (lm t_hi lmin lsub one tlo tt th r target : Std.I64)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hne : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hg : fixed_ln.ONE = ok one)
    (hadd : I64.checked_add lm one = some t_hi)
    (hsubc : I64.checked_sub lmin one = some lsub)
    (hloop2 : fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none))
    (hmul : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target)
    (hloop3 : fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none))
    (hexp : fold_port.stats.basis.exp2 tt = ok (some th))
    (hθ : 0 < th.val)
    (hfit : distList th.val lambda.val ≤ 2 ^ 63 - 1)
    (htotd : distList th.val lambda.val ≤ 2 ^ 126)
    (logv : Std.I64 → Std.I64) (tr : Std.I64)
    (hlogr : ∀ l, fold_port.stats.basis.log2 l = ok (some (logv l)))
    (htr : logv th = tr)
    (hmono : ∀ l, th.val < l.val → 0 ≤ (logv l).val - tr.val)
    (hnbad : ¬ rateBad logv th tr.val lambda.val)
    (hbound : rateAt logv th tr.val lambda.val ≤ 2 ^ 126)
    (z : Std.I128) (hz : z.val = rateAt logv th tr.val lambda.val / 2)
    (hrn : solve_port.q32.narrow z = ok (some r)) :
    ∃ (d : Std.I64),
      fold_port.stats.basis.water_fill_words lambda budget side
        = ok (core.result.Result.Ok (fold_port.stats.basis.WaterFill.mk th d r)) := by
  have hd1 : core.option.Option.and_then
      fold_port.stats.basis.water_fill_words.closure_1.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
      (some lm) () = ok (some t_hi) := closure_1_step lm one t_hi hg hadd
  have hd2 : core.option.Option.and_then
      fold_port.stats.basis.water_fill_words.closure_2.Insts.CoreOpsFunctionFnOnceTupleI64OptionI64
      (some lmin) () = ok (some lsub) := closure_2_step lmin one lsub hg hsubc
  obtain ⟨d, hd, _⟩ := ((distortion_words_post lambda th hL htotd).2 hθ).1 hfit
  obtain ⟨z', hz', hzr, _⟩ :=
    rate_words_pos_spec lambda th logv tr hθ hlogr htr hmono hnbad hbound
  have hzz : z' = z := IScalar.eq_of_val_eq (by rw [hz', hz])
  have hrate : fold_port.stats.basis.rate_words lambda th
      = ok (core.result.Result.Ok r) := by
    have hrn' : solve_port.q32.narrow z' = ok (some r) := by rw [hzz]; exact hrn
    exact hzr r hrn'
  exact ⟨d, water_fill_words_ok lambda budget side t n lg lg1 mx mp sum
    hsub hlen hlogs hloop0 hne hmax hloop1 ht
    lm t_hi lmin lsub one tlo hlogmax hd1 hlogmin hd2 hg
    th target tt d r hloop2 hmul hloop3 hexp hd hrate⟩

/-! ### `water_fill_words_ok_loops_derived` — the loop equations extracted inside the theorem

`water_fill_words_ok_loops` still takes the loop-2 and loop-3 step equations as hypotheses. Here
the caller supplies `water_fill_words_loop2_at`'s conclusion and `water_fill_words_loop3_at`'s
conclusion instead, and the two step equations are extracted from them inside the proof. The
error component stays an explicit hypothesis: the specs' first post disjunct leaves `x.2` free, so
`ok (tlo, none)` is not derivable from the post alone, and inventing it would be unsound.
`exp2` is quantified over the level loop 3 returns, so it stays a call hypothesis.
-/
theorem water_fill_words_ok_loops_derived
    (lambda : Slice Std.I64) (budget side t n : Std.I64)
    (lg lg1 : alloc.vec.Vec Std.I64) (mx mp : Std.I64) (sum : Std.I128)
    (lm t_hi lmin lsub one th r target : Std.I64)
    (hsub : I64.checked_sub budget side = some t)
    (hlen : I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from (Slice.len lambda)
      = ok (core.result.Result.Ok n))
    (hlogs : alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len lambda) = ok lg)
    (hloop0 : fold_port.stats.basis.water_fill_words_loop0
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i64 core.num.I64.MAX
        lg none = ok (mx, mp, lg1, none))
    (hne : lambda.val ≠ [])
    (hmax : 0 < mx.val)
    (hL : ∀ l ∈ lambda.val, 0 ≤ l.val)
    (hloop1 : fold_port.stats.basis.water_fill_words_loop1
        { start := 0#usize, «end» := Slice.len lambda } lambda 0#i128 = ok sum)
    (ht : 0 < t.val)
    (hlogmax : fold_port.stats.basis.log2 mx = ok (some lm))
    (hlogmin : fold_port.stats.basis.log2 mp = ok (some lmin))
    (hg : fixed_ln.ONE = ok one)
    (hadd : I64.checked_add lm one = some t_hi)
    (hsubc : I64.checked_sub lmin one = some lsub)
    (h2at : ∃ y : Std.I64 × Option fold_port.stats.basis.BasisError,
      fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok y ∧
      ((∃ v : Std.I64, fold_port.stats.basis.rate_at_logs (alloc.vec.Vec.deref lg1) y.1
          = ok (core.result.Result.Ok v) ∧ t.val ≤ v.val) ∨
        y.2 ≠ none ∨
        (y.2 = none ∧
          y.1.val = walkLevel lsub.val one.val
            (fold_port.stats.basis.BISECTION_ITERS.val - (0#usize).val) ∧
          walkProbesBelow lg1 t lsub.val one.val
            (fold_port.stats.basis.BISECTION_ITERS.val - (0#usize).val))))
    (h2none : ∀ (y : Std.I64 × Option fold_port.stats.basis.BasisError),
      fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok y → y.2 = none)
    (hmul : I64.checked_mul fold_port.stats.basis.RATE_TOL_Q n = some target)
    (h3at : ∀ (tlo : Std.I64),
      fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none) →
      ∃ y : Std.I64 × Option fold_port.stats.basis.BasisError,
        fold_port.stats.basis.water_fill_words_loop3
          { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
          t lg1 t_hi tlo target t_hi none false = ok y ∧
        (ConvergedAt lg1 t target y.1 ∨ y.2 ≠ none ∨
          (y.2 = none ∧
            (RejectedAt lg1 t target y.1 ∨
              ((0#usize).val = fold_port.stats.basis.BISECTION_ITERS.val ∧ y.1 = t_hi)))))
    (h3none : ∀ (tlo : Std.I64) (y : Std.I64 × Option fold_port.stats.basis.BasisError),
      fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none) →
      fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok y → y.2 = none)
    (hexp : ∀ (tlo tt : Std.I64),
      fold_port.stats.basis.water_fill_words_loop2
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 lsub one none false = ok (tlo, none) →
      fold_port.stats.basis.water_fill_words_loop3
        { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
        t lg1 t_hi tlo target t_hi none false = ok (tt, none) →
      fold_port.stats.basis.exp2 tt = ok (some th))
    (hθ : 0 < th.val)
    (hfit : distList th.val lambda.val ≤ 2 ^ 63 - 1)
    (htotd : distList th.val lambda.val ≤ 2 ^ 126)
    (logv : Std.I64 → Std.I64) (tr : Std.I64)
    (hlogr : ∀ l, fold_port.stats.basis.log2 l = ok (some (logv l)))
    (htr : logv th = tr)
    (hmono : ∀ l, th.val < l.val → 0 ≤ (logv l).val - tr.val)
    (hnbad : ¬ rateBad logv th tr.val lambda.val)
    (hbound : rateAt logv th tr.val lambda.val ≤ 2 ^ 126)
    (z : Std.I128) (hz : z.val = rateAt logv th tr.val lambda.val / 2)
    (hrn : solve_port.q32.narrow z = ok (some r)) :
    ∃ (d : Std.I64),
      fold_port.stats.basis.water_fill_words lambda budget side
        = ok (core.result.Result.Ok (fold_port.stats.basis.WaterFill.mk th d r)) := by
  obtain ⟨y2, h2eq, -⟩ := h2at
  obtain ⟨tlo, e2⟩ := y2
  have he2 : e2 = none := h2none (tlo, e2) h2eq
  have hloop2 : fold_port.stats.basis.water_fill_words_loop2
      { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
      t lg1 lsub one none false = ok (tlo, none) := by
    simpa [he2] using h2eq
  obtain ⟨y3, h3eq, -⟩ := h3at tlo hloop2
  obtain ⟨tt, e3⟩ := y3
  have he3 : e3 = none := h3none tlo (tt, e3) hloop2 h3eq
  have hloop3 : fold_port.stats.basis.water_fill_words_loop3
      { start := 0#usize, «end» := fold_port.stats.basis.BISECTION_ITERS }
      t lg1 t_hi tlo target t_hi none false = ok (tt, none) := by
    simpa [he3] using h3eq
  exact water_fill_words_ok_loops lambda budget side t n lg lg1 mx mp sum
    lm t_hi lmin lsub one tlo tt th r target
    hsub hlen hlogs hloop0 hne hmax hL hloop1 ht
    hlogmax hlogmin hg hadd hsubc hloop2 hmul hloop3
    (hexp tlo tt hloop2 hloop3)
    hθ hfit htotd logv tr hlogr htr hmono hnbad hbound z hz hrn

end Learn.WaterFill

