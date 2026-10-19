import Mathlib
import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Learn.Limb
import Learn.FitIcm

/-!
# The `sub_q` residual-state error across `sweep_rows`

`src/learn/fold/fit/icm.rs` `sweep_rows` keeps the residual state `f_res` and updates it once per
moved `(row, block)` step:

```text
let w = mul_q(a, e[i*n + col]);          // :1044  round(a·e / 2^32)
f_res[row*n + col] = sub_q(f_res[...], w); // :1045  checked i64 subtraction
```

with `e = nt_q(chg, cols)` (`:1034`) on the code change `chg = C_best − s_cur` and the metric block
`cols = H_rot`. `Learn.Limb.fblk_error_le` takes `|fh_res − f_res| ≤ εres` as a hypothesis; this
module derives that invariant and its closed form, then re-pins the end-to-end word bound at the
derived value.

## The model

Real values of Q32.32 words throughout. One step, at the entry's coefficient `a` (a Q32.32 word,
`|a| ≤ 1`) and the computed metric image `eh` (`|eh − e| ≤ εe`, `e` the exact `H·ΔC`):

* `mul_q` is `round_shift(a·e, 32)` = `rshift_rne(·, 32)` (`src/learn/solve/q32.rs:183`), modelled
  exactly as `rne32` and **proved** within half a word (`rne32_int_spec`, `mulQ_spec`);
* `sub_q`/`add_q` are checked `i64` add/sub, exact when they do not overflow: the state step is
  `fh' = fh − w` and the exact counterpart `f' = f − a·e`;
* the step error is `|Δ'| ≤ |Δ| + εe + ½/2^32` (`resid_step_le`), and `N` steps give
  `εres₀ + N·(εe + ½/2^32)` (`resid_iter_le`).

`εe` is the `nt_q(chg, cols)` entry bound: `Learn.Limb.limb_error_le_G` at the codebook-peak `chg`
(`2·1.21e10`, `Ea = 36`) and the `H_rot` peak (`6.05e10`, `Eb = 37`), one block, `G = 6218`
(`limb_error_chg`), and at the measured `Down` `chg` peak (`1.6443e10`, `Ea = 35`), `G = 3114`
(`limb_error_chg_Ea35`).

## Model boundary

Not claimed: the AVX-VNNI body, `split_digits`' digit/residual bounds, the `i64`/`i128` accumulator
identity `hT` (all three inherited as hypotheses from `Learn.Limb`); `add_q`/`sub_q` overflow and
`mul_q`'s `i64` narrowing, both of which return `Err` and abort the pass (the invariant is on the
success branch); and the step count `N`, which is data-dependent — per entry it is at most
`sweeps · blocks`.

The derived worst-case bound is large. Under the codebook-peak fallback `Ea ≤ 36`, at `N = 128`
the state error is `εres = 128·12437/2^33 ≈ 1.85e-4` and the margin is
`311287510032769024/2^64 ≈ 1.6875e-2`. At the measured `Ea ≤ 35` step the margin at `N = 128` is
`157448303632769024/2^64 ≈ 8.535e-3`, and at the structural maximum `N = 1216 = 2 · 608`
(`524.7` measured moves per entry per sweep, `608` blocks per sweep at `sweeps = 2`) it is
`1469504930832769024/2^64 ≈ 7.966e-2`. All are far above the measured
argmin margin `1.75e-7`: the worst-case state recurrence, not the `q`/`dots` rounding, dominates, and
the `εres = 0` constant `3088700432769024/2^64` remains the informative bound for the rounding alone.
The `Ea ≤ 35` hypothesis is measured on the `Down` witness (`max |chg| = 16,442,969,296` words at
`o = 896, n = 4864, sweeps = 1`); the `Ea ≤ 36` versions are the sound general fallback. The step
count `N` is at most `sweeps · blocks` (`Learn.IcmCount.stepCount_sweeps_le`), so `N = 128`
(`sweeps = 2`, `blocks = n/8 = 64`) and `N = 1216` (`2 · 608`) are structural maxima; only `Ea ≤ 35`
remains a measured hypothesis.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable section

namespace Learn.IcmError

/-! ## `mul_q`: round-to-nearest-even, proved within half a word -/

/-- The rounding step of `rshift_rne`: `0` when the truncating remainder is zero or below half in
magnitude, `sign r` when it is above half, and on the tie `|r| = 2^31` `0` when `q` is even and
`sign r` when it is odd. -/
def rneStep (q r : ℤ) : ℤ :=
  if r = 0 then 0
  else if |r| > 2 ^ 31 then (if 0 < r then 1 else -1)
  else if |r| < 2 ^ 31 then 0
  else if q % 2 = 0 then 0
  else (if 0 < r then 1 else -1)

/-- `v / 2^32` rounded to nearest, ties to even: `rshift_rne(v, 32)` (`src/learn/solve/q32.rs:183`),
with `q = v.tdiv 2^32` truncating toward zero and `r = v.tmod 2^32` carrying the dividend's sign. -/
def rne32 (v : ℤ) : ℤ :=
  v.tdiv (2 ^ 32) + rneStep (v.tdiv (2 ^ 32)) (v.tmod (2 ^ 32))

/-- The truncating remainder is less than the divisor in magnitude. -/
lemma abs_tmod_lt (v : ℤ) : |v.tmod (2 ^ 32)| < 2 ^ 32 := by
  have hb : (0 : ℤ) < 2 ^ 32 := by norm_num
  by_cases hdv : (2 ^ 32 : ℤ) ∣ v
  · rw [Int.tmod_eq_zero_of_dvd hdv]; norm_num
  · by_cases h0 : 0 ≤ v
    · have h1 : 0 ≤ v.tmod (2 ^ 32) := Int.tmod_nonneg _ h0
      have h2 : v.tmod (2 ^ 32) < 2 ^ 32 := Int.tmod_lt_of_pos v hb
      rw [abs_of_nonneg h1]; exact h2
    · have hneg : v < 0 := by omega
      rw [Int.tmod_eq_emod]
      have hc : ¬ (0 ≤ v ∨ (2 ^ 32 : ℤ) ∣ v) := by
        intro h; exact h.elim h0 hdv
      rw [if_neg hc]
      have hnum : ((2 ^ 32 : ℤ).natAbs : ℤ) = 2 ^ 32 := by norm_num
      rw [hnum]
      have hv : v % 2 ^ 32 < 2 ^ 32 := Int.emod_lt_of_pos v hb
      have hv0 : 0 ≤ v % 2 ^ 32 := Int.emod_nonneg v (by norm_num)
      rw [abs_lt]
      omega

/-- The rounding step moves the truncating quotient by at most half a word. -/
lemma rneStep_abs_le (q r : ℤ) (h : |r| < 2 ^ 32) : |2 ^ 32 * rneStep q r - r| ≤ 2 ^ 31 := by
  have hpow : (2 : ℤ) ^ 32 = 4294967296 := by norm_num
  have hpow31 : (2 : ℤ) ^ 31 = 2147483648 := by norm_num
  rw [hpow] at h ⊢
  rw [hpow31]
  obtain ⟨hlo, hhi⟩ := abs_lt.mp h
  dsimp only [rneStep]
  rcases lt_trichotomy r 0 with hneg | hzero | hpos
  · rw [abs_of_neg hneg]
    simp only [if_neg (by omega : ¬ (r = 0)), if_neg (by omega : ¬ (0 < r))]
    split_ifs <;> (rw [abs_le]; constructor <;> omega)
  · subst hzero
    norm_num
  · rw [abs_of_pos hpos]
    simp only [if_pos hpos]
    split_ifs <;> (rw [abs_le]; constructor <;> omega)

/-- **The `mul_q` rounding.** `round_shift_32(a·b)` is within half a word of the exact product. -/
theorem rne32_int_spec (v : ℤ) : |2 ^ 32 * rne32 v - v| ≤ 2 ^ 31 := by
  have hq : rne32 v = v.tdiv (2 ^ 32) + rneStep (v.tdiv (2 ^ 32)) (v.tmod (2 ^ 32)) := rfl
  have hdecomp : v.tdiv (2 ^ 32) * 2 ^ 32 + v.tmod (2 ^ 32) = v := Int.tdiv_mul_add_tmod v _
  have hlt : |v.tmod (2 ^ 32)| < 2 ^ 32 := abs_tmod_lt v
  have hs := rneStep_abs_le (v.tdiv (2 ^ 32)) (v.tmod (2 ^ 32)) hlt
  rw [hq]
  have heq : 2 ^ 32 * (v.tdiv (2 ^ 32) + rneStep (v.tdiv (2 ^ 32)) (v.tmod (2 ^ 32))) - v
      = 2 ^ 32 * rneStep (v.tdiv (2 ^ 32)) (v.tmod (2 ^ 32)) - v.tmod (2 ^ 32) := by
    have h32 : (2 : ℤ) ^ 32 = 4294967296 := by norm_num
    rw [h32] at hdecomp ⊢
    linarith
  rw [heq]
  exact hs

/-- `mul_q`: the Q32.32 word product `round_shift(a·b, 32)` (`icm.rs:130`). -/
def mulQ (a b : ℤ) : ℤ := rne32 (a * b)

/-- **`mul_q` in real units.** The rounded word's real value is within half a word of the exact
product's real value: `|w/2^32 − a·b/2^64| ≤ ½/2^32`. -/
theorem mulQ_spec (a b : ℤ) :
    |((mulQ a b : ℤ) : ℝ) / 2 ^ 32 - (a : ℝ) * (b : ℝ) / 2 ^ 64| ≤ 1 / 2 / 2 ^ 32 := by
  have h := rne32_int_spec (a * b)
  have hcast : (2 : ℝ) ^ 32 * ((mulQ a b : ℤ) : ℝ) - (a : ℝ) * (b : ℝ)
      = (((2 ^ 32 * rne32 (a * b) - a * b : ℤ)) : ℝ) := by
    unfold mulQ
    push_cast
    norm_num
  have hbound : |(2 : ℝ) ^ 32 * ((mulQ a b : ℤ) : ℝ) - (a : ℝ) * (b : ℝ)| ≤ ((2 ^ 31 : ℤ) : ℝ) := by
    rw [hcast, ← Int.cast_abs]
    exact_mod_cast h
  have h64 : (0 : ℝ) < 2 ^ 64 := by norm_num
  have hkey : ((mulQ a b : ℤ) : ℝ) / 2 ^ 32 - (a : ℝ) * (b : ℝ) / 2 ^ 64
      = ((2 : ℝ) ^ 32 * ((mulQ a b : ℤ) : ℝ) - (a : ℝ) * (b : ℝ)) / 2 ^ 64 := by
    field_simp
  rw [hkey, abs_div, abs_of_pos h64]
  calc |(2 : ℝ) ^ 32 * ((mulQ a b : ℤ) : ℝ) - (a : ℝ) * (b : ℝ)| / 2 ^ 64
      ≤ (((2 ^ 31 : ℤ)) : ℝ) / 2 ^ 64 := div_le_div_of_nonneg_right hbound h64.le
    _ = 1 / 2 / 2 ^ 32 := by push_cast; norm_num

/-! ## The state step and its closed form -/

/-- **One `sub_q` state step.** With `|fh − f| ≤ εr`, `|a| ≤ 1`, `|eh − e| ≤ εe` and the `mul_q`
rounding `|ρ| ≤ ½/2^32`, the step error grows by `εe + ½/2^32`. -/
theorem resid_step_le {fh f a eh e εr εe ρ : ℝ}
    (hr : |fh - f| ≤ εr) (ha : |a| ≤ 1) (he : |eh - e| ≤ εe)
    (hρ : |ρ| ≤ 1 / 2 / 2 ^ 32) :
    |(fh - (a * eh + ρ)) - (f - a * e)| ≤ εr + εe + 1 / 2 / 2 ^ 32 := by
  have hsplit : (fh - (a * eh + ρ)) - (f - a * e) = (fh - f) - a * (eh - e) - ρ := by ring
  rw [hsplit]
  have h1 : |a * (eh - e)| ≤ εe := by
    rw [abs_mul]
    calc |a| * |eh - e| ≤ 1 * εe := mul_le_mul ha he (abs_nonneg _) (by norm_num)
      _ = εe := by ring
  have h2 := abs_sub ((fh - f) - a * (eh - e)) ρ
  have h3 := abs_sub (fh - f) (a * (eh - e))
  calc |(fh - f) - a * (eh - e) - ρ| ≤ |(fh - f) - a * (eh - e)| + |ρ| := h2
    _ ≤ (|fh - f| + |a * (eh - e)|) + 1 / 2 / 2 ^ 32 := by linarith
    _ ≤ εr + εe + 1 / 2 / 2 ^ 32 := by linarith

/-- **The pass recurrence, closed form.** `N` state steps grow the residual error by
`N·(εe + ½/2^32)`; the recurrence is `ε(k+1) = ε(k) + εe + ½/2^32` (`resid_step_le`). -/
theorem resid_iter_le (N : ℕ) (εr εe : ℝ) (a eh e fh f ρ : ℕ → ℝ)
    (h0 : |fh 0 - f 0| ≤ εr) (ha : ∀ k, |a k| ≤ 1) (he : ∀ k, |eh k - e k| ≤ εe)
    (hρ : ∀ k, |ρ k| ≤ 1 / 2 / 2 ^ 32)
    (hW : ∀ k, fh (k + 1) = fh k - (a k * eh k + ρ k))
    (hX : ∀ k, f (k + 1) = f k - a k * e k) :
    |fh N - f N| ≤ εr + (N : ℝ) * (εe + 1 / 2 / 2 ^ 32) := by
  induction N with
  | zero => simpa using h0
  | succ n ih =>
    have hstep := resid_step_le (fh := fh n) (f := f n) (a := a n) (eh := eh n) (e := e n) (ρ := ρ n)
      ih (ha n) (he n) (hρ n)
    rw [hW n, hX n]
    have harith : εr + (n : ℝ) * (εe + 1 / 2 / 2 ^ 32) + (εe + 1 / 2 / 2 ^ 32)
        = εr + ((n + 1 : ℕ) : ℝ) * (εe + 1 / 2 / 2 ^ 32) := by push_cast; ring
    linarith [hstep, harith]

/-! ## The `chg` call's entry bound and the per-step constant -/

/-- **`nt_q(chg, cols)`: the state-update peak.** `L = 5`, `ea ≤ 36`, `eb ≤ 37`: `chg = C_best −
s_cur` with `|chg| ≤ 2·1.21e10` (`Ea = 36`) and `H_rot` `6.05e10` (`Eb = 37`); one block,
`97·2^{73−32−35} = 97·2^6 = 6208`, `G = 7 + 1 + 1 + 6209 = 6218` words. -/
theorem limb_error_chg (ea eb : ℕ) (hea : ea ≤ 36) (heb : eb ≤ 37)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) (T : ℤ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ Learn.Limb.step ^ 5 / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ Learn.Limb.step ^ 5 / 2)
    (hT : (T : ℝ) = Learn.Limb.limbPre 5 ea eb α β) :
    |((Learn.Limb.rshift32 T : ℤ) : ℝ) / 2 ^ 32
        - Learn.Limb.exactPre 5 ea eb α β ρa ρb / 2 ^ 64| ≤ 6218 / 2 ^ 32 :=
  Learn.Limb.limb_error_le_G (L := 5) (Ea := 36) (Eb := 37) (G := 6218)
    (by norm_num) (by unfold Learn.Limb.step; norm_num)
    ea eb hea heb α β ρa ρb T hα hβ hρa hρb hT

/-- The per-step state constant: `G/2^32 + ½/2^32 = (6218 + ½)/2^32 = 12437/2^33`. -/
def epsStep : ℝ := 12437 / 2 ^ 33

theorem eps_step_eq : 6218 / 2 ^ 32 + 1 / 2 / 2 ^ 32 = epsStep := by
  unfold epsStep; norm_num

/-- **The state invariant, derived.** `N` `sub_q` steps whose metric image is within `6218/2^32`
(`limb_error_chg`) and whose `mul_q` rounding is within half a word give
`|fh N − f N| ≤ εr + N·epsStep`; this is the `εres` of `epsF`. -/
theorem resid_le_derived (N : ℕ) (εr : ℝ) (a eh e frh fr ρ : ℕ → ℝ)
    (h0 : |frh 0 - fr 0| ≤ εr) (ha : ∀ k, |a k| ≤ 1) (he : ∀ k, |eh k - e k| ≤ 6218 / 2 ^ 32)
    (hρ : ∀ k, |ρ k| ≤ 1 / 2 / 2 ^ 32)
    (hW : ∀ k, frh (k + 1) = frh k - (a k * eh k + ρ k))
    (hX : ∀ k, fr (k + 1) = fr k - a k * e k) :
    |frh N - fr N| ≤ εr + (N : ℝ) * epsStep := by
  have h := resid_iter_le N εr (6218 / 2 ^ 32) a eh e frh fr ρ h0 ha he hρ hW hX
  rw [eps_step_eq] at h
  exact h

/-! ## The end-to-end constant at the derived state error -/

/-- `q_all`'s end-to-end entry error (`Learn.Limb.q_all_end2end`). -/
def epsQEnd : ℝ := 99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32)

/-- `dots`' end-to-end entry error given the `f_blk` error `εf` (`Learn.Limb.dots_end2end`). -/
def epsDEnd (εf : ℝ) : ℝ := 24842 / 2 ^ 32 + 8 * εf * (12100000000 / 2 ^ 32)

theorem epsQEnd_eq : epsQEnd = 728088661250048 / 2 ^ 64 := by unfold epsQEnd; norm_num

theorem epsDEnd_local_eq : epsDEnd (3114 / 2 ^ 32) = 408130777567232 / 2 ^ 64 := by
  unfold epsDEnd; norm_num

/-- The `f_blk` error from the state recurrence: the `εres` of `resid_le_derived` plus the one-call
`s·H` term `3114/2^32` (`Learn.Limb.limb_error_codebook`), matching `fblk_error_le`'s `εres + εv`. -/
def epsF (εres0 : ℝ) (N : ℕ) : ℝ := εres0 + (N : ℝ) * epsStep + 3114 / 2 ^ 32

/-- **The margin at the derived state error, exactly.** `2·(εq + 2·εd)` at `εf = epsF εres0 N`:
the `εres = 0` constant plus `N·2407803200000000/2^64` plus `εres0·387200000000/2^32`. -/
theorem margin_derived_eq (N : ℕ) (εres0 : ℝ) :
    2 * ((1 : ℝ) * (99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32))
        + 2 * (24842 / 2 ^ 32 + 8 * epsF εres0 N * (12100000000 / 2 ^ 32)))
      = 3088700432769024 / 2 ^ 64 + (N : ℝ) * (2407803200000000 / 2 ^ 64)
        + εres0 * (387200000000 / 2 ^ 32) := by
  unfold epsF epsStep
  ring

/-- **`dots` at the derived state error.** `Learn.Limb.dots_end2end_state` composed with
`resid_le_derived`'s `εres = εres0 + N·epsStep` and `εv = 3114/2^32`: this is the `εf = epsF εres0 N`
that `word_argmin_cost_bound_end2end` consumes. -/
theorem dots_derived (v vh fr frh C : Fin 8 → ℝ) (a dh : ℝ) (N : ℕ) (εres0 : ℝ)
    (hv : ∀ k, |vh k - v k| ≤ 3114 / 2 ^ 32)
    (hr : ∀ k, |frh k - fr k| ≤ εres0 + (N : ℝ) * epsStep)
    (ha : |a| ≤ 1) (hε : 0 ≤ εres0 + (N : ℝ) * epsStep)
    (hC : ∀ k, |C k| ≤ 12100000000 / 2 ^ 32)
    (hcall : |dh - Learn.Limb.dot8 (fun k => frh k + a * vh k) C| ≤ 24842 / 2 ^ 32) :
    |dh - Learn.Limb.dot8 (fun k => fr k + a * v k) C| ≤ epsDEnd (epsF εres0 N) := by
  have h := Learn.Limb.dots_end2end_state v vh fr frh C a dh (3114 / 2 ^ 32)
    (εres0 + (N : ℝ) * epsStep) hv hr ha hε hC hcall
  unfold epsDEnd epsF
  exact h

/-! ## The end-to-end word bound at the derived state error -/

/-- **The end-to-end bound with the derived state recurrence.** `εf = epsF εres0 N`, `N ≤ sweeps ·
blocks` the number of `sub_q` steps on the entry; the exact margin is the `εres = 0` constant plus
`N·2407803200000000/2^64` plus `εres0·387200000000/2^32`. -/
theorem word_argmin_cost_bound_end2end_derived (q d qh dh : Learn.FitIcm.CodeIndex → ℝ)
    (N : ℕ) (εres0 : ℝ)
    (hq : ∀ c, |qh c - q c| ≤ epsQEnd)
    (hd : ∀ c, |dh c - d c| ≤ epsDEnd (epsF εres0 N))
    (cstar cold : Learn.FitIcm.CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar
      ≤ q cold - 2 * d cold
        + (3088700432769024 / 2 ^ 64 + (N : ℝ) * (2407803200000000 / 2 ^ 64)
            + εres0 * (387200000000 / 2 ^ 32)) := by
  have h := Learn.FitIcm.word_argmin_cost_bound_end2end q d qh dh (epsF εres0 N) hq hd
    cstar cold hmin
  rw [margin_derived_eq N εres0] at h
  exact h

/-- `N = 128`, `εres0 = 0`: the witness shape's `sweeps = 2`, `blocks = n/8 = 64`. -/
theorem derived_const_value :
    3088700432769024 / 2 ^ 64 + ((128 : ℕ) : ℝ) * (2407803200000000 / 2 ^ 64)
        + 0 * (387200000000 / 2 ^ 32)
      = 311287510032769024 / 2 ^ 64 := by
  norm_num

/-- **The witness-shape bound.** `N = 128`, `εres0 = 0`: `311287510032769024/2^64 ≈ 1.69e-2`. That
is five orders above the measured argmin margin `1.75e-7`, so at this `N` the worst-case state
recurrence does not certify the decision; the `εres = 0` constant `3088700432769024/2^64` is the
informative bound for the rounding alone. -/
theorem word_argmin_cost_bound_end2end_derived_const (q d qh dh : Learn.FitIcm.CodeIndex → ℝ)
    (hq : ∀ c, |qh c - q c| ≤ epsQEnd)
    (hd : ∀ c, |dh c - d c| ≤ epsDEnd (epsF 0 128))
    (cstar cold : Learn.FitIcm.CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar ≤ q cold - 2 * d cold + 311287510032769024 / 2 ^ 64 := by
  have h := word_argmin_cost_bound_end2end_derived q d qh dh 128 0 hq hd cstar cold hmin
  rw [derived_const_value] at h
  exact h

/-! ## The measured `Ea ≤ 35` step (Down witness)

The `Down` witness (`o = 896`, `n = 4864`, `sweeps = 1`, `r = c = d = 1`) measures
`max |chg| = 16,442,969,296` words, so `Ea = 35` rather than the codebook-peak `36`: the
`nt_q(chg, cols)` entry bound halves to `3114/2^32` and the per-step state constant becomes
`epsStep35 = 6229/2^33`. The `Ea ≤ 36` chain above stays as the sound general fallback, whose
hypothesis holds for every input.
-/

/-- **`nt_q(chg, cols)` at the measured Down `chg` peak.** `ea ≤ 35` (`|chg| ≤ 16,442,969,296`
words), `eb ≤ 37` (`|H_rot| ≤ 60,466,383,651`): `97·2^{72−32−35} = 97·2^5 = 3104` words of mass,
`⌈15/2⌉ + 1 + (1 + 3104) = 3114`. Conditional on `ea ≤ 35`: measured at the Down witness, not proved
for every input. -/
theorem limb_error_chg_Ea35 (ea eb : ℕ) (hea : ea ≤ 35) (heb : eb ≤ 37)
    (α β : ℕ → ℕ → ℤ) (ρa ρb : ℕ → ℝ) (T : ℤ)
    (hα : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |α i s| ≤ 64)
    (hβ : ∀ i ∈ Finset.range 32, ∀ s ∈ Finset.range 5, |β i s| ≤ 64)
    (hρa : ∀ i ∈ Finset.range 32, |ρa i| ≤ Learn.Limb.step ^ 5 / 2)
    (hρb : ∀ i ∈ Finset.range 32, |ρb i| ≤ Learn.Limb.step ^ 5 / 2)
    (hT : (T : ℝ) = Learn.Limb.limbPre 5 ea eb α β) :
    |((Learn.Limb.rshift32 T : ℤ) : ℝ) / 2 ^ 32
        - Learn.Limb.exactPre 5 ea eb α β ρa ρb / 2 ^ 64| ≤ 3114 / 2 ^ 32 :=
  Learn.Limb.limb_error_le_G (L := 5) (Ea := 35) (Eb := 37) (G := 3114)
    (by norm_num) (by unfold Learn.Limb.step; norm_num)
    ea eb hea heb α β ρa ρb T hα hβ hρa hρb hT

/-- The per-step state constant at the measured `Ea ≤ 35`: `3114/2^32 + ½/2^32 = 6229/2^33`. -/
def epsStep35 : ℝ := 6229 / 2 ^ 33

theorem eps_step_eq_Ea35 : 3114 / 2 ^ 32 + 1 / 2 / 2 ^ 32 = epsStep35 := by
  unfold epsStep35; norm_num

/-- **The state invariant at the measured step.** `N` `sub_q` steps whose metric image is within
`3114/2^32` (`limb_error_chg_Ea35`) and whose `mul_q` rounding is within half a word give
`|fh N − f N| ≤ εr + N·epsStep35`. -/
theorem resid_le_derived_Ea35 (N : ℕ) (εr : ℝ) (a eh e frh fr ρ : ℕ → ℝ)
    (h0 : |frh 0 - fr 0| ≤ εr) (ha : ∀ k, |a k| ≤ 1) (he : ∀ k, |eh k - e k| ≤ 3114 / 2 ^ 32)
    (hρ : ∀ k, |ρ k| ≤ 1 / 2 / 2 ^ 32)
    (hW : ∀ k, frh (k + 1) = frh k - (a k * eh k + ρ k))
    (hX : ∀ k, fr (k + 1) = fr k - a k * e k) :
    |frh N - fr N| ≤ εr + (N : ℝ) * epsStep35 := by
  have h := resid_iter_le N εr (3114 / 2 ^ 32) a eh e frh fr ρ h0 ha he hρ hW hX
  rw [eps_step_eq_Ea35] at h
  exact h

/-- The `f_blk` error at the measured step: `εres0 + N·epsStep35` plus the one-call `s·H` term
`3114/2^32` (`Learn.Limb.limb_error_codebook`, the same integer as the `chg` word bound by
coincidence; they are different quantities). -/
def epsF35 (εres0 : ℝ) (N : ℕ) : ℝ := εres0 + (N : ℝ) * epsStep35 + 3114 / 2 ^ 32

/-- **`dots` at the measured step.** `Learn.Limb.dots_end2end_state` at `εres = εres0 + N·epsStep35`,
`εv = 3114/2^32`: the `εf = epsF35 εres0 N` of the `Ea ≤ 35` chain. -/
theorem dots_derived_Ea35 (v vh fr frh C : Fin 8 → ℝ) (a dh : ℝ) (N : ℕ) (εres0 : ℝ)
    (hv : ∀ k, |vh k - v k| ≤ 3114 / 2 ^ 32)
    (hr : ∀ k, |frh k - fr k| ≤ εres0 + (N : ℝ) * epsStep35)
    (ha : |a| ≤ 1) (hε : 0 ≤ εres0 + (N : ℝ) * epsStep35)
    (hC : ∀ k, |C k| ≤ 12100000000 / 2 ^ 32)
    (hcall : |dh - Learn.Limb.dot8 (fun k => frh k + a * vh k) C| ≤ 24842 / 2 ^ 32) :
    |dh - Learn.Limb.dot8 (fun k => fr k + a * v k) C| ≤ epsDEnd (epsF35 εres0 N) := by
  have h := Learn.Limb.dots_end2end_state v vh fr frh C a dh (3114 / 2 ^ 32)
    (εres0 + (N : ℝ) * epsStep35) hv hr ha hε hC hcall
  unfold epsDEnd epsF35
  exact h

/-- **The margin at the measured step, exactly.** `2·(εq + 2·εd)` at `εf = epsF35 εres0 N`: the
`εres = 0` constant plus `N·1205934400000000/2^64` plus `εres0·387200000000/2^32`. -/
theorem margin_derived_eq_Ea35 (N : ℕ) (εres0 : ℝ) :
    2 * ((1 : ℝ) * (99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32))
        + 2 * (24842 / 2 ^ 32 + 8 * epsF35 εres0 N * (12100000000 / 2 ^ 32)))
      = 3088700432769024 / 2 ^ 64 + (N : ℝ) * (1205934400000000 / 2 ^ 64)
        + εres0 * (387200000000 / 2 ^ 32) := by
  unfold epsF35 epsStep35
  ring

/-- **The end-to-end bound at the measured step.** `εf = epsF35 εres0 N`; the exact margin is the
`εres = 0` constant plus `N·1205934400000000/2^64` plus `εres0·387200000000/2^32`. -/
theorem word_argmin_cost_bound_end2end_derived_Ea35 (q d qh dh : Learn.FitIcm.CodeIndex → ℝ)
    (N : ℕ) (εres0 : ℝ)
    (hq : ∀ c, |qh c - q c| ≤ epsQEnd)
    (hd : ∀ c, |dh c - d c| ≤ epsDEnd (epsF35 εres0 N))
    (cstar cold : Learn.FitIcm.CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar
      ≤ q cold - 2 * d cold
        + (3088700432769024 / 2 ^ 64 + (N : ℝ) * (1205934400000000 / 2 ^ 64)
            + εres0 * (387200000000 / 2 ^ 32)) := by
  have h := Learn.FitIcm.word_argmin_cost_bound_end2end q d qh dh (epsF35 εres0 N) hq hd
    cstar cold hmin
  rw [margin_derived_eq_Ea35 N εres0] at h
  exact h

/-- `N = 128`, `εres0 = 0`: the witness shape's `sweeps = 2`, `blocks = n/8 = 64`. -/
theorem derived_const_value_Ea35_N128 :
    3088700432769024 / 2 ^ 64 + ((128 : ℕ) : ℝ) * (1205934400000000 / 2 ^ 64)
        + 0 * (387200000000 / 2 ^ 32)
      = 157448303632769024 / 2 ^ 64 := by
  norm_num

/-- `N = 1216 = 2 · 608`, the structural maximum per entry at the Down witness (`sweeps = 2`,
`blocks = n/8 = 608` at `n = 4864`; `Learn.IcmCount.stepCount_sweeps_le`). The measured average is
`524.7` moves per entry per sweep, below the maximum. -/
theorem derived_const_value_Ea35_N1216 :
    3088700432769024 / 2 ^ 64 + ((1216 : ℕ) : ℝ) * (1205934400000000 / 2 ^ 64)
        + 0 * (387200000000 / 2 ^ 32)
      = 1469504930832769024 / 2 ^ 64 := by
  norm_num

/-- **The witness-shape bound at the measured step.** `N = 128`, `εres0 = 0`:
`157448303632769024/2^64 ≈ 8.535e-3`, still four orders above the measured argmin margin `1.75e-7`. -/
theorem word_argmin_cost_bound_end2end_derived_Ea35_const (q d qh dh : Learn.FitIcm.CodeIndex → ℝ)
    (hq : ∀ c, |qh c - q c| ≤ epsQEnd)
    (hd : ∀ c, |dh c - d c| ≤ epsDEnd (epsF35 0 128))
    (cstar cold : Learn.FitIcm.CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar ≤ q cold - 2 * d cold + 157448303632769024 / 2 ^ 64 := by
  have h := word_argmin_cost_bound_end2end_derived_Ea35 q d qh dh 128 0 hq hd cstar cold hmin
  rw [derived_const_value_Ea35_N128] at h
  exact h

/-- **The Down-shape bound at the structural maximum.** `N = 1216 = 2 · 608` is the per-entry maximum
of the move structure (`sweeps = 2`, `blocks = n/8 = 608` at `o = 896`, `n = 4864`); `ea ≤ 35` in
`limb_error_chg_Ea35` remains measured. `εres0 = 0`: the margin is
`1469504930832769024/2^64 ≈ 7.966e-2`. `Learn.IcmCount.word_argmin_cost_bound_end2end_derived_Ea35_count_down`
states the same bound with `N` taken from the loop count. -/
theorem word_argmin_cost_bound_end2end_derived_Ea35_measured (q d qh dh : Learn.FitIcm.CodeIndex → ℝ)
    (hq : ∀ c, |qh c - q c| ≤ epsQEnd)
    (hd : ∀ c, |dh c - d c| ≤ epsDEnd (epsF35 0 1216))
    (cstar cold : Learn.FitIcm.CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar ≤ q cold - 2 * d cold + 1469504930832769024 / 2 ^ 64 := by
  have h := word_argmin_cost_bound_end2end_derived_Ea35 q d qh dh 1216 0 hq hd cstar cold hmin
  rw [derived_const_value_Ea35_N1216] at h
  exact h

end Learn.IcmError
