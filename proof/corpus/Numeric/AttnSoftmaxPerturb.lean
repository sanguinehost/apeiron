/-
Softmax perturbation and the attention stage composition (`APEIRON-ALGEBRA-G3-ATTN-W1-T1`).

`Numeric.EngineAttnRefinement` states the end-to-end target `attn_head_refines` of the extracted
`word_attn_head` (`src/engine/attn.rs:432-572`). This module proves
the real composition of the score, softmax, weight and output stages against the dense softmax
reference `denseAttnReal`, and states the corrected target. The refutation of `attn_head_refines`
as stated was removed against the live extraction; see "Statements dropped and restated" below.

## What is proved here

* `reweight_mean_perturb`: for reference weights `e ≥ 0` with `Σ e > 0` and any reweighting `e'`
  with `Σ e' > 0`, values in `[lo, hi]` move the weighted mean by at most
  `(hi - lo) · Σ|e' - e| / Σ e'`. Proof: centre at `μ = Σ e v / Σ e`, so `Σ e (v - μ) = 0`.
* the per-source numerator perturbations: score shift (`score_shift_perturb`, factor `exp δ - 1`),
  `fixed_exp` error (`fixed_exp_numerator_perturb`, `epsExp·e + 4/2^32`), underflow saturation
  (`saturation_numerator_perturb`, `exp(-16)` per saturated key), and the extracted per-key case
  split `exp_at_numerator_cases` over `exp_at_real_bound` / `exp_at_underflow`;
* `attn_output_perturb`: the weight division `divRoundNearest (E_j·2^32) T` and the output
  epilogue `divRoundNearest (Σ W_j·V_j) 2^32` composed with `reweight_mean_perturb` for any
  per-numerator bound `η_j`;
* `attn_stage_compose`: the stage composition against `denseAttnReal` of the ideal scores, with
  executed scores within `δ`, numerators in the `exp_at` window-or-saturated form, and the bound
  `1/(2·2^32) + n·V/(2·2^32) + (hi - lo)·((epsExp + exp δ - 1)·U + n·(4/2^32 + exp(-16)))/S'`,
  `U = Σ exp(shift_j)`, `S' = T/2^32 ≥ 1 - epsExp - 4/2^32`;
* `attn_stage_compose_closed`: the same composition with `U` and `S'` eliminated, the bound
  `1/(2·2^32) + n·V/(2·2^32) + (hi - lo)·softmaxPerturbBound n δ`.

All of the above are pure `ℝ`/`ℤ` statements over the stage facts, taken as hypotheses; none
reads the extracted loops.

## The corrected target

`attn_domain_v2` and `attn_head_refines_v2` are the corrected end-to-end target: the
reachable-state invariant (`pos < occupied[layer]`, `scale` the `fixed_rsqrt` result of
`head_dim·2^32`, plane lengths, per-key `row_dot_domain`, the score and value windows) and the
closed bound `attnErrBoundV2`. `attn_domain_v2_witness` checks the joint domain on a one-layer,
one-position state. `attn_head_refines_v2` is **not proved in the corpus**: its proof module
`Numeric/AttnHeadRefines.lean` is in `work/wip/attn-downstream/` (it closes through the test-only
`attn.value_component`), together with the loop specifications (`Numeric/AttnLoopScore.lean`,
`Numeric/AttnLoopOutput.lean`) and the window derivations (`Numeric/AttnHeadWindowsScore.lean`,
`Numeric/AttnHeadWindowsValue.lean`). The score-stage bound `Numeric.AttnScoreStage.attn_score_stage`
is in the corpus and green.

## Statements dropped and restated against the live extraction

The live `attn.KvState` is `{ carrier : mixer.Carrier, …, storage : attn.KvStorage }`; `k` and `v`
are no longer fields. Restated in this file:

* `attn_domain_v2` takes the plane tables `k v` as parameters and states
  `state.storage = attn.KvStorage.Q32 k v` before `k.val[layer.val]? = some kplane` and
  `v.val[layer.val]? = some vplane`, the form `Numeric.EngineAttnRefinement`'s `attn_domain`-based
  `attn_head_refines` already uses. The window conjuncts are unchanged.
* `attn_head_refines_v2` takes `k v` and assumes `carrierQ32 state` and
  `state.storage = attn.KvStorage.Q32 k v`, matching `attn_head_refines`.
* `attnOneState` is built with `carrier := mixer.Carrier.Q32` and `storage := attn.KvStorage.Q32 …`,
  the construction form `Numeric.EngineSamplerRefinement` already uses.

Removed, with the arithmetic they carried kept:

* `attnEmptyState`, `attnEmptyState_domain` and `not_attn_head_refines` (the refutation of
  `attn_head_refines`). `not_attn_head_refines` was proved from `word_attn_head_pos_refuses` on a
  concrete state, and that lemma takes `carrierQ32 state` as a hypothesis. `carrierQ32` is
  `¬ (decide (state.carrier.read_discriminant = mixer.Carrier.F32.read_discriminant) = true)` and
  `Std.Discriminant.read_discriminant` is an opaque instance projection
  (`target/aeneas-model/lean/Aeneas/Data/Discriminant.lean:14`), so no concrete state discharges it
  — the blocker `Numeric.EngineAttnRefinement`'s header already records. No statement here depends
  on the removed declarations. The refutation of `attn_head_refines` remains open at `anchor`; its
  argument (`attn_domain` admits `pos ≥ occupied[layer]`, where the body returns
  `Err PositionOutOfRange` by `src/engine/attn.rs:465-467`) is recorded at the definition.

## Trust boundary

Pure real/integer mathematics over the Q32.32 denotation in `Numeric.Scale`, plus the joint
witness `attn_domain_v2_witness`, which reaches the extracted `KvState` structure of
`proof/ApeironWord.lean` (not the `attn_head` body: no theorem here unfolds it after the removed
refutation). It does not cover rustc/LLVM codegen of `src/engine/attn.rs`, the kernel ABI,
GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.EngineAttnRefinement
import Mathlib.Algebra.BigOperators.Field

namespace apeiron_word

open Aeneas Aeneas.Std Result
open Finset

noncomputable section

/-! ## The refutation of `attn_head_refines` -/

/-- `1 ≤ Usize.max` on either platform word size. -/
theorem attn_one_le_usize_max : 1 ≤ Usize.max := by
  have h : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  rcases System.Platform.numBits_eq with h' | h' <;> simp [h, h']

/-! ## The reweighted-mean perturbation -/

/-- **Reweighted-mean perturbation (centred form).** For nonnegative weights `e` with positive
total, any positive-total reweighting `e'` moves the weighted mean of values in `[lo, hi]` by at
most `(hi - lo) · Σ|e' - e| / Σ e'`. Proof: centre at `μ = Σ e v / Σ e`; `Σ e (v - μ) = 0`. -/
theorem reweight_mean_perturb {ι : Type*} (s : Finset ι) (e e' v : ι → ℝ) (lo hi : ℝ)
    (he : ∀ j ∈ s, 0 ≤ e j) (hS : 0 < ∑ j ∈ s, e j) (hS' : 0 < ∑ j ∈ s, e' j)
    (hv : ∀ j ∈ s, lo ≤ v j ∧ v j ≤ hi) :
    |(∑ j ∈ s, e' j * v j) / (∑ j ∈ s, e' j) - (∑ j ∈ s, e j * v j) / (∑ j ∈ s, e j)|
      ≤ (hi - lo) * (∑ j ∈ s, |e' j - e j|) / (∑ j ∈ s, e' j) := by
  set S := ∑ j ∈ s, e j with hSdef
  set S' := ∑ j ∈ s, e' j with hS'def
  set μ := (∑ j ∈ s, e j * v j) / S with hμ
  have hcentre : ∑ j ∈ s, e j * (v j - μ) = 0 := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul]
    rw [hμ, ← hSdef]; field_simp; ring
  have hμlo : lo ≤ μ := by
    rw [hμ, le_div_iff₀ hS, hSdef, Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by nlinarith [he j hj, (hv j hj).1]
  have hμhi : μ ≤ hi := by
    rw [hμ, div_le_iff₀ hS, hSdef, Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by nlinarith [he j hj, (hv j hj).2]
  have hkey : (∑ j ∈ s, e' j * v j) / S' - μ
      = (∑ j ∈ s, (e' j - e j) * (v j - μ)) / S' := by
    have h1 : ∑ j ∈ s, (e' j - e j) * (v j - μ)
        = ∑ j ∈ s, e' j * v j - μ * S' - ∑ j ∈ s, e j * (v j - μ) := by
      rw [hS'def, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [h1, hcentre, sub_zero]
    field_simp
  rw [hkey, abs_div, abs_of_pos hS', div_le_div_iff_of_pos_right hS']
  calc |∑ j ∈ s, (e' j - e j) * (v j - μ)|
      ≤ ∑ j ∈ s, |(e' j - e j) * (v j - μ)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ s, |e' j - e j| * (hi - lo) := by
        apply Finset.sum_le_sum
        intro j hj
        rw [abs_mul]
        apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
        rw [abs_le]
        constructor <;> linarith [(hv j hj).1, (hv j hj).2]
    _ = (hi - lo) * ∑ j ∈ s, |e' j - e j| := by rw [← Finset.sum_mul, mul_comm]

/-! ## Per-source numerator perturbations -/

/-- `|exp t - 1| ≤ exp |t| - 1`: equality for `t ≥ 0`; for `t < 0`, `x + 1/x ≥ 2` at
`x = exp t`. -/
theorem abs_exp_sub_one_le (t : ℝ) : |Real.exp t - 1| ≤ Real.exp |t| - 1 := by
  rcases le_total 0 t with h | h
  · rw [abs_of_nonneg h, abs_of_nonneg (by linarith [Real.add_one_le_exp t])]
  · have hle : Real.exp t ≤ 1 := Real.exp_le_one_iff.2 h
    rw [abs_of_nonpos h, abs_of_nonpos (by linarith)]
    have hxy : Real.exp t * Real.exp (-t) = 1 := by rw [← Real.exp_add]; simp
    nlinarith [mul_nonneg (sq_nonneg (Real.exp t - 1)) (le_of_lt (Real.exp_pos (-t))),
      Real.exp_pos t]

/-- `|exp a - exp b| ≤ exp b · (exp |a - b| - 1)`. -/
theorem abs_exp_sub_exp_le (a b : ℝ) :
    |Real.exp a - Real.exp b| ≤ Real.exp b * (Real.exp |a - b| - 1) := by
  have hsplit : Real.exp a - Real.exp b = Real.exp b * (Real.exp (a - b) - 1) := by
    rw [mul_sub, ← Real.exp_add]; ring_nf
  rw [hsplit, abs_mul, abs_of_pos (Real.exp_pos b)]
  exact mul_le_mul_of_nonneg_left (abs_exp_sub_one_le _) (le_of_lt (Real.exp_pos b))

/-- **(a) Score shift.** A multiplicative shift `e'_j = e_j · exp δ_j` with `|δ_j| ≤ δ` moves the
numerators by at most `(exp δ - 1) · Σ e`. -/
theorem score_shift_perturb {ι : Type*} (s : Finset ι) (e δs : ι → ℝ) (δ : ℝ)
    (he : ∀ j ∈ s, 0 ≤ e j) (hδ : ∀ j ∈ s, |δs j| ≤ δ) :
    ∑ j ∈ s, |e j * Real.exp (δs j) - e j| ≤ (Real.exp δ - 1) * ∑ j ∈ s, e j := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro j hj
  have h1 : e j * Real.exp (δs j) - e j = e j * (Real.exp (δs j) - 1) := by ring
  rw [h1, abs_mul, abs_of_nonneg (he j hj), mul_comm]
  apply mul_le_mul_of_nonneg_right _ (he j hj)
  calc |Real.exp (δs j) - 1| ≤ Real.exp |δs j| - 1 := abs_exp_sub_one_le _
    _ ≤ Real.exp δ - 1 := by linarith [Real.exp_le_exp.2 (hδ j hj)]

/-- **(b) `fixed_exp` numerator.** Per-numerator error `epsExp·e_j + 4/2^32` sums to
`epsExp · Σ e + |s| · 4/2^32`. -/
theorem fixed_exp_numerator_perturb {ι : Type*} (s : Finset ι) (e e' : ι → ℝ)
    (h : ∀ j ∈ s, |e' j - e j| ≤ epsExp * e j + 4 / q32) :
    ∑ j ∈ s, |e' j - e j| ≤ epsExp * ∑ j ∈ s, e j + (s.card : ℝ) * (4 / q32) := by
  calc ∑ j ∈ s, |e' j - e j| ≤ ∑ j ∈ s, (epsExp * e j + 4 / q32) := Finset.sum_le_sum h
    _ = epsExp * ∑ j ∈ s, e j + (s.card : ℝ) * (4 / q32) := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul]

/-- **(c) Saturation.** A saturated numerator `e'_j = 0` against a reference `0 ≤ e_j ≤ exp(-16)`
costs at most `exp(-16)` per key. -/
theorem saturation_numerator_perturb {ι : Type*} (s : Finset ι) (e e' : ι → ℝ)
    (h : ∀ j ∈ s, e' j = 0 ∧ 0 ≤ e j ∧ e j ≤ Real.exp (-16)) :
    ∑ j ∈ s, |e' j - e j| ≤ (s.card : ℝ) * Real.exp (-16) := by
  calc ∑ j ∈ s, |e' j - e j| ≤ ∑ _j ∈ s, Real.exp (-16) := by
        apply Finset.sum_le_sum
        intro j hj
        obtain ⟨h0, hn, hle⟩ := h j hj
        rw [h0, zero_sub, abs_neg, abs_of_nonneg hn]
        exact hle
    _ = (s.card : ℝ) * Real.exp (-16) := by rw [Finset.sum_const, nsmul_eq_mul]

/-- **Extracted per-key numerator.** With the checked shift `shifted = score - mx ≤ 0` (the exact
maximum shift), `exp_at` returns `Ok v` in exactly one of two forms: the `fixed_exp` window with
the proved error `epsExp·exp + 4/2^32` (`exp_at_real_bound`), or the saturation `v = 0` below
`-16·2^32` (`exp_at_underflow`). This is the numerator hypothesis of `attn_stage_compose`. -/
theorem exp_at_numerator_cases (score mx shifted : I64)
    (hsub : I64.checked_sub score mx = some shifted) (hup : shifted.val ≤ 0) :
    ∃ v : I64, attn.exp_at score mx = ok (core.result.Result.Ok v) ∧
      (((-16) * (2 : ℤ) ^ (32 : ℕ) ≤ shifted.val ∧
          |toReal v - Real.exp (toReal shifted)| ≤
            epsExp * Real.exp (toReal shifted) + 4 / q32) ∨
        (shifted.val < (-16) * (2 : ℤ) ^ (32 : ℕ) ∧ v.val = 0)) := by
  by_cases hlt : shifted.val < (-68719476736 : ℤ)
  · refine ⟨0#i64, exp_at_underflow score mx shifted hsub hlt, Or.inr ⟨?_, by simp⟩⟩
    norm_num; exact hlt
  · have hge : (-68719476736 : ℤ) ≤ shifted.val := by omega
    obtain ⟨v, hv, hb⟩ := exp_at_real_bound score mx shifted hsub hge
      ⟨hge, by linarith⟩
    exact ⟨v, hv, Or.inl ⟨by norm_num; exact hge, hb⟩⟩

/-! ## Integer rounding in real form -/

/-- `divRoundNearest a d · d` is within `d/2` of `a`, over `ℝ`. -/
theorem divRoundNearest_real_bound (a d : ℤ) (hd : 0 < d) :
    |((divRoundNearest a d : ℤ) : ℝ) * d - a| ≤ (d : ℝ) / 2 := by
  have hZ := div_round_nearest_abs_bound a d hd
  have hZ' : |divRoundNearest a d * d - a| ≤ d / 2 := by simpa [divRoundNearest] using hZ
  have hdm : d / 2 * 2 ≤ d := by omega
  have hnum : |((divRoundNearest a d * d - a : ℤ) : ℝ)| ≤ ((d / 2 : ℤ) : ℝ) := by
    exact_mod_cast hZ'
  have h2 : ((d / 2 : ℤ) : ℝ) * 2 ≤ (d : ℝ) := by exact_mod_cast hdm
  push_cast at hnum
  linarith

/-- The weight division `W = divRoundNearest (E·2^32) T` is within `1/(2·2^32)` of `E/T`. -/
theorem weight_div_real_bound (E T W : ℤ) (hT : 0 < T)
    (hW : W = divRoundNearest (E * (2 : ℤ) ^ (32 : ℕ)) T) :
    |(W : ℝ) / q32 - (E : ℝ) / T| ≤ 1 / (2 * q32) := by
  have h := divRoundNearest_real_bound (E * (2 : ℤ) ^ (32 : ℕ)) T hT
  rw [← hW] at h
  push_cast at h
  have hTR : (0 : ℝ) < T := by exact_mod_cast hT
  have hq : q32 = (2 : ℝ) ^ (32 : ℕ) := rfl
  rw [div_sub_div _ _ q32_ne_zero (ne_of_gt hTR), abs_div, abs_of_pos (mul_pos q32_pos hTR),
    div_le_iff₀ (mul_pos q32_pos hTR)]
  have hr : 1 / (2 * q32) * (q32 * (T : ℝ)) = (T : ℝ) / 2 := by
    field_simp [q32_ne_zero]
  rw [hr, hq]
  obtain ⟨h1, h2⟩ := abs_le.1 h
  rw [abs_le]
  constructor <;> linarith

/-- The output epilogue `out = divRoundNearest acc 2^32` is within `1/(2·2^32)` of `acc/2^64`. -/
theorem out_epilogue_real_bound (acc out : ℤ)
    (hout : out = divRoundNearest acc ((2 : ℤ) ^ (32 : ℕ))) :
    |(out : ℝ) / q32 - (acc : ℝ) / q32 ^ 2| ≤ 1 / (2 * q32) := by
  have h := divRoundNearest_real_bound acc ((2 : ℤ) ^ (32 : ℕ)) (by norm_num)
  rw [← hout] at h
  push_cast at h
  have hq : q32 = (2 : ℝ) ^ (32 : ℕ) := rfl
  have key : (out : ℝ) / q32 - (acc : ℝ) / q32 ^ 2 = ((out : ℝ) * q32 - acc) / q32 ^ 2 := by
    field_simp [q32_ne_zero]
  have hq2 : (0 : ℝ) < q32 ^ 2 := pow_pos q32_pos 2
  rw [key, abs_div, abs_of_pos hq2, div_le_iff₀ hq2]
  have hr : 1 / (2 * q32) * q32 ^ 2 = q32 / 2 := by
    field_simp [q32_ne_zero]
  rw [hr, hq]
  obtain ⟨h1, h2⟩ := abs_le.1 h
  rw [abs_le]
  constructor <;> norm_num <;> linarith

/-! ## List/Finset bridge for the dense reference -/

/-- A mapped `List.range` sum is the `Finset.range` sum. -/
theorem list_sum_range_map (f : ℕ → ℝ) (n : ℕ) :
    ((List.range n).map f).sum = ∑ j ∈ range n, f j := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]
    simp

/-- Zipping two maps of one list and multiplying is the map of the product. -/
theorem zip_map_mul_eq (L : List ℕ) (a b : ℕ → ℝ) :
    ((L.map a).zip (L.map b)).map (fun p => p.1 * p.2) = L.map (fun j => a j * b j) := by
  induction L with
  | nil => simp
  | cons h t ih => simp [ih]

/-- **Dense reference, `Finset` form.** On range-indexed score and value lists the dense softmax
attention is `Σ exp(x_j - m)·v_j / Σ exp(x_j - m)`. -/
theorem denseAttnReal_range (m : ℝ) (n : ℕ) (x v : ℕ → ℝ) :
    denseAttnReal m ((List.range n).map x) ((List.range n).map v)
      = (∑ j ∈ range n, Real.exp (x j - m) * v j) / ∑ j ∈ range n, Real.exp (x j - m) := by
  unfold denseAttnReal softmaxWeights
  simp only [List.map_map]
  rw [zip_map_mul_eq, list_sum_range_map, list_sum_range_map, Finset.sum_div]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  simp only [Function.comp]
  ring

/-- The dense reference does not depend on the shift (`softmaxWeights_shift_invariant`). -/
theorem denseAttnReal_shift_invariant (m m' : ℝ) (sc vs : List ℝ) :
    denseAttnReal m sc vs = denseAttnReal m' sc vs := by
  unfold denseAttnReal
  rw [softmaxWeights_shift_invariant m m']

/-! ## The output stage composed with the perturbation -/

/-- **Weight and output stages under a numerator perturbation.** For executed numerators `E_j`
with checked total `T = Σ E_j > 0`, weights `W_j = divRoundNearest (E_j·2^32) T`, output
`out = divRoundNearest (Σ W_j·V_j) 2^32`, values `v_j = V_j/2^32 ∈ [lo, hi]` with `|v_j| ≤ V`,
and any reference numerators `e_j ≥ 0` with `Σ e > 0` and `|E_j/2^32 - e_j| ≤ η_j`, the output is
within `1/(2·2^32) + n·V/(2·2^32) + (hi - lo)·Σ η / (T/2^32)` of the reference mean
`Σ e_j v_j / Σ e_j`. Terms: output epilogue, `n` weight roundings, `reweight_mean_perturb`. -/
theorem attn_output_perturb (n : ℕ) (E W Vi : ℕ → ℤ) (T acc out : ℤ) (e η : ℕ → ℝ)
    (lo hi V : ℝ)
    (hT : T = ∑ j ∈ range n, E j) (hTpos : 0 < T)
    (hW : ∀ j ∈ range n, W j = divRoundNearest (E j * (2 : ℤ) ^ (32 : ℕ)) T)
    (hacc : acc = ∑ j ∈ range n, W j * Vi j)
    (hout : out = divRoundNearest acc ((2 : ℤ) ^ (32 : ℕ)))
    (hvb : ∀ j ∈ range n, lo ≤ (Vi j : ℝ) / q32 ∧ (Vi j : ℝ) / q32 ≤ hi)
    (hV : ∀ j ∈ range n, |(Vi j : ℝ) / q32| ≤ V)
    (he : ∀ j ∈ range n, 0 ≤ e j) (hS : 0 < ∑ j ∈ range n, e j)
    (hη : ∀ j ∈ range n, |(E j : ℝ) / q32 - e j| ≤ η j) :
    |(out : ℝ) / q32 - (∑ j ∈ range n, e j * ((Vi j : ℝ) / q32)) / (∑ j ∈ range n, e j)|
      ≤ 1 / (2 * q32) + n * V / (2 * q32)
          + (hi - lo) * (∑ j ∈ range n, η j) / ((T : ℝ) / q32) := by
  set v : ℕ → ℝ := fun j => (Vi j : ℝ) / q32 with hvdef
  have hTR : (0 : ℝ) < T := by exact_mod_cast hTpos
  have hS'pos : (0 : ℝ) < (T : ℝ) / q32 := div_pos hTR q32_pos
  have hTsum : (T : ℝ) / q32 = ∑ j ∈ range n, (E j : ℝ) / q32 := by
    rw [hT]; push_cast; rw [Finset.sum_div]
  -- output epilogue
  have h1 := out_epilogue_real_bound acc out hout
  have hacc' : (acc : ℝ) / q32 ^ 2 = ∑ j ∈ range n, (W j : ℝ) / q32 * v j := by
    rw [hacc]; push_cast; rw [Finset.sum_div]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    rw [hvdef]; field_simp [q32_ne_zero]
  -- weight roundings
  have h2 : |∑ j ∈ range n, (W j : ℝ) / q32 * v j - ∑ j ∈ range n, (E j : ℝ) / T * v j|
      ≤ n * V / (2 * q32) := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ j ∈ range n, ((W j : ℝ) / q32 * v j - (E j : ℝ) / T * v j)|
        ≤ ∑ j ∈ range n, |(W j : ℝ) / q32 * v j - (E j : ℝ) / T * v j| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _j ∈ range n, 1 / (2 * q32) * V := by
          apply Finset.sum_le_sum
          intro j hj
          rw [← sub_mul, abs_mul]
          exact mul_le_mul (weight_div_real_bound (E j) T (W j) hTpos (hW j hj)) (hV j hj)
            (abs_nonneg _) (div_nonneg zero_le_one (by linarith [q32_pos]))
      _ = n * V / (2 * q32) := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; ring
  -- the executed exact-ratio mean is the reweighted mean
  have hmean : ∑ j ∈ range n, (E j : ℝ) / T * v j
      = (∑ j ∈ range n, (E j : ℝ) / q32 * v j) / (∑ j ∈ range n, (E j : ℝ) / q32) := by
    rw [← hTsum, Finset.sum_div]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    field_simp [q32_ne_zero, ne_of_gt hTR]
  have hS' : 0 < ∑ j ∈ range n, (E j : ℝ) / q32 := by rw [← hTsum]; exact hS'pos
  have h3 := reweight_mean_perturb (range n) e (fun j => (E j : ℝ) / q32) v lo hi he hS hS' hvb
  have hne : (range n).Nonempty := by
    by_contra hc
    rw [Finset.not_nonempty_iff_eq_empty] at hc
    rw [hc, Finset.sum_empty] at hS
    exact lt_irrefl _ hS
  obtain ⟨j0, hj0⟩ := hne
  have hlohi : 0 ≤ hi - lo := by linarith [(hvb j0 hj0).1, (hvb j0 hj0).2]
  have hsumη : ∑ j ∈ range n, |(E j : ℝ) / q32 - e j| ≤ ∑ j ∈ range n, η j :=
    Finset.sum_le_sum hη
  have h3' : (hi - lo) * (∑ j ∈ range n, |(E j : ℝ) / q32 - e j|) / (∑ j ∈ range n, (E j : ℝ) / q32)
      ≤ (hi - lo) * (∑ j ∈ range n, η j) / ((T : ℝ) / q32) := by
    rw [← hTsum]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hsumη hlohi) (le_of_lt hS'pos)
  rw [hacc'] at h1
  rw [hmean] at h2
  try simp only at h3
  have hA := abs_sub_le ((out : ℝ) / q32) (∑ j ∈ range n, (W j : ℝ) / q32 * v j)
    ((∑ j ∈ range n, e j * v j) / (∑ j ∈ range n, e j))
  have hB := abs_sub_le (∑ j ∈ range n, (W j : ℝ) / q32 * v j)
    ((∑ j ∈ range n, (E j : ℝ) / q32 * v j) / (∑ j ∈ range n, (E j : ℝ) / q32))
    ((∑ j ∈ range n, e j * v j) / (∑ j ∈ range n, e j))
  linarith

/-! ## Numerator facts in the `exp_at` form -/

/-- The `exp_at` numerator hypothesis for one key at integer shift `sh = score - mx` and executed
numerator `E`: the `fixed_exp` window with its proved error, or the saturation `E = 0` below
`-16·2^32` (`src/engine/attn.rs:127-134`). -/
def attnNumeratorCase (sh E : ℤ) : Prop :=
  ((-16) * (2 : ℤ) ^ (32 : ℕ) ≤ sh ∧
      |(E : ℝ) / q32 - Real.exp ((sh : ℝ) / q32)| ≤ epsExp * Real.exp ((sh : ℝ) / q32) + 4 / q32) ∨
    (sh < (-16) * (2 : ℤ) ^ (32 : ℕ) ∧ E = 0)

/-- **Extracted per-key numerator, stage form.** `exp_at_numerator_cases` in the shape of
`attnNumeratorCase`. -/
theorem exp_at_numerator_case (score mx shifted : I64)
    (hsub : I64.checked_sub score mx = some shifted) (hup : shifted.val ≤ 0) :
    ∃ v : I64, attn.exp_at score mx = ok (core.result.Result.Ok v) ∧
      attnNumeratorCase shifted.val v.val := by
  obtain ⟨v, hv, hc⟩ := exp_at_numerator_cases score mx shifted hsub hup
  exact ⟨v, hv, hc⟩

theorem epsExp_pos : 0 < epsExp := by unfold epsExp; positivity

theorem epsExp_lt_one : epsExp < 1 := by norm_num [epsExp]

/-- `1 - epsExp - 4/2^32 > 0`: the lower bound of the executed total. -/
theorem attn_total_floor_pos : 0 < 1 - epsExp - 4 / q32 := by norm_num [epsExp, q32]

/-- `4/2^32 < (1 - epsExp)·exp(-16)`: an in-window numerator is strictly positive. From
`exp 1 ≤ 3`, `exp 16 ≤ 3^16`. -/
theorem attn_window_numerator_floor : 4 / q32 < (1 - epsExp) * Real.exp (-16) := by
  have he1 : Real.exp 1 ≤ 3 := by linarith [Real.exp_one_lt_d9]
  have h16 : Real.exp 16 = Real.exp 1 ^ 16 := by rw [← Real.exp_nat_mul]; norm_num
  have hle : Real.exp 16 ≤ 43046721 := by
    rw [h16]
    calc Real.exp 1 ^ 16 ≤ (3 : ℝ) ^ 16 := pow_le_pow_left₀ (Real.exp_pos 1).le he1 16
      _ = 43046721 := by norm_num
  have h1 : (1 : ℝ) / 43046721 ≤ Real.exp (-16) := by
    rw [Real.exp_neg, one_div]
    exact inv_anti₀ (Real.exp_pos 16) hle
  calc 4 / q32 < (1 - epsExp) * (1 / 43046721) := by norm_num [epsExp, q32]
    _ ≤ (1 - epsExp) * Real.exp (-16) :=
        mul_le_mul_of_nonneg_left h1 (by linarith [epsExp_lt_one])

/-- **Uniform numerator error and sign.** Both `exp_at` branches give
`|E/2^32 - exp(sh/2^32)| ≤ epsExp·exp(sh/2^32) + (4/2^32 + exp(-16))` and `E ≥ 0`. -/
theorem attn_numerator_facts (sh E : ℤ) (h : attnNumeratorCase sh E) :
    |(E : ℝ) / q32 - Real.exp ((sh : ℝ) / q32)|
        ≤ epsExp * Real.exp ((sh : ℝ) / q32) + (4 / q32 + Real.exp (-16)) ∧
      0 ≤ (E : ℝ) / q32 := by
  have hq : q32 = (2 : ℝ) ^ (32 : ℕ) := rfl
  have hepos := Real.exp_pos ((sh : ℝ) / q32)
  have h16 := Real.exp_pos (-16)
  rcases h with ⟨hge, hb⟩ | ⟨hlt, hE⟩
  · refine ⟨by linarith, ?_⟩
    have hu : (-16 : ℝ) ≤ (sh : ℝ) / q32 := by
      rw [le_div_iff₀ q32_pos, hq]
      have : (((-16) * (2 : ℤ) ^ (32 : ℕ) : ℤ) : ℝ) ≤ (sh : ℝ) := by exact_mod_cast hge
      push_cast at this
      linarith
    have hexp : Real.exp (-16) ≤ Real.exp ((sh : ℝ) / q32) := Real.exp_le_exp.2 hu
    have hlo := (abs_le.1 hb).1
    have hm : (1 - epsExp) * Real.exp (-16) ≤ (1 - epsExp) * Real.exp ((sh : ℝ) / q32) :=
      mul_le_mul_of_nonneg_left hexp (by linarith [epsExp_lt_one])
    linarith [attn_window_numerator_floor]
  · have hu : (sh : ℝ) / q32 < -16 := by
      rw [div_lt_iff₀ q32_pos, hq]
      have : (sh : ℝ) < (((-16) * (2 : ℤ) ^ (32 : ℕ) : ℤ) : ℝ) := by exact_mod_cast hlt
      push_cast at this
      linarith
    have hexp : Real.exp ((sh : ℝ) / q32) ≤ Real.exp (-16) := Real.exp_le_exp.2 hu.le
    have hE' : (E : ℝ) / q32 = 0 := by rw [hE]; simp
    rw [hE', zero_sub, abs_neg, abs_of_pos hepos]
    refine ⟨?_, le_refl _⟩
    have := mul_nonneg epsExp_pos.le hepos.le
    have : (0 : ℝ) < 4 / q32 := div_pos (by norm_num) q32_pos
    linarith

/-- **Max key.** At shift `0` the numerator is in the window and `E/2^32 ≥ 1 - epsExp - 4/2^32`. -/
theorem attn_numerator_max (E : ℤ) (h : attnNumeratorCase 0 E) :
    1 - epsExp - 4 / q32 ≤ (E : ℝ) / q32 := by
  rcases h with ⟨-, hb⟩ | ⟨hlt, -⟩
  · simp only [Int.cast_zero, zero_div, Real.exp_zero, mul_one] at hb
    linarith [(abs_le.1 hb).1]
  · norm_num at hlt

/-! ## The stage composition -/

/-- **Stage composition (proved, pure `ℝ`/`ℤ`).** Hypotheses, one per stage over the causal prefix
`j < n`:

* score: executed Q32.32 scores `y_j` within `δ` of the ideal real scores `x_j`, and `mx` attained
  by some key (`hmax`);
* softmax: each executed numerator `E_j` in the `exp_at` form at shift `y_j - mx`
  (`attnNumeratorCase`, supplied per key by `exp_at_numerator_case`);
* normaliser: `T = Σ E_j`;
* weight: `W_j = divRoundNearest (E_j·2^32) T` (`weight_at_spec`);
* output: `out = divRoundNearest (Σ W_j·V_j) 2^32` (`value_component_epilogue`), values
  `V_j/2^32 ∈ [lo, hi]`, `|V_j/2^32| ≤ V`.

Conclusions: `T > 0` (the `ZeroSum` arm is unreachable), `T/2^32 ≥ 1 - epsExp - 4/2^32`, and
the output is within
`1/(2·2^32) + n·V/(2·2^32) + (hi - lo)·((epsExp + exp δ - 1)·U + n·(4/2^32 + exp(-16)))/(T/2^32)`
of `denseAttnReal m` of the ideal scores for every shift `m`, with `U = Σ exp((y_j - mx)/2^32)`.
The `n·exp(-16)` term is the saturation cost. -/
theorem attn_stage_compose (n : ℕ) (x : ℕ → ℝ) (y E W Vi : ℕ → ℤ) (mx T acc out : ℤ)
    (m lo hi V δ : ℝ)
    (hδ : ∀ j ∈ range n, |(y j : ℝ) / q32 - x j| ≤ δ)
    (hmax : ∃ j ∈ range n, y j = mx)
    (hnum : ∀ j ∈ range n, attnNumeratorCase (y j - mx) (E j))
    (hT : T = ∑ j ∈ range n, E j)
    (hW : ∀ j ∈ range n, W j = divRoundNearest (E j * (2 : ℤ) ^ (32 : ℕ)) T)
    (hacc : acc = ∑ j ∈ range n, W j * Vi j)
    (hout : out = divRoundNearest acc ((2 : ℤ) ^ (32 : ℕ)))
    (hvb : ∀ j ∈ range n, lo ≤ (Vi j : ℝ) / q32 ∧ (Vi j : ℝ) / q32 ≤ hi)
    (hV : ∀ j ∈ range n, |(Vi j : ℝ) / q32| ≤ V) :
    0 < T ∧ 1 - epsExp - 4 / q32 ≤ (T : ℝ) / q32 ∧
    |(out : ℝ) / q32 - denseAttnReal m ((List.range n).map x)
        ((List.range n).map (fun j => (Vi j : ℝ) / q32))|
      ≤ 1 / (2 * q32) + n * V / (2 * q32)
        + (hi - lo) * ((epsExp + (Real.exp δ - 1))
              * (∑ j ∈ range n, Real.exp (((y j - mx : ℤ) : ℝ) / q32))
            + n * (4 / q32 + Real.exp (-16))) / ((T : ℝ) / q32) := by
  obtain ⟨j0, hj0, hy0⟩ := hmax
  have hfacts := fun j hj => attn_numerator_facts (y j - mx) (E j) (hnum j hj)
  have hE0 : 1 - epsExp - 4 / q32 ≤ (E j0 : ℝ) / q32 := by
    have h := hnum j0 hj0
    rw [hy0, sub_self] at h
    exact attn_numerator_max (E j0) h
  have hTsum : (T : ℝ) / q32 = ∑ j ∈ range n, (E j : ℝ) / q32 := by
    rw [hT]; push_cast; rw [Finset.sum_div]
  have hS'lb : 1 - epsExp - 4 / q32 ≤ (T : ℝ) / q32 := by
    rw [hTsum]
    exact le_trans hE0
      (Finset.single_le_sum (f := fun j => (E j : ℝ) / q32) (fun j hj => (hfacts j hj).2) hj0)
  have hTR : (0 : ℝ) < T := by
    have h := mul_pos (lt_of_lt_of_le attn_total_floor_pos hS'lb) q32_pos
    rwa [div_mul_cancel₀ _ q32_ne_zero] at h
  have hTpos : 0 < T := by exact_mod_cast hTR
  refine ⟨hTpos, hS'lb, ?_⟩
  set c : ℝ := (mx : ℝ) / q32 with hc
  have he : ∀ j ∈ range n, 0 ≤ Real.exp (x j - c) := fun j _ => (Real.exp_pos _).le
  have hS : 0 < ∑ j ∈ range n, Real.exp (x j - c) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨j0, hj0⟩
  have hη : ∀ j ∈ range n, |(E j : ℝ) / q32 - Real.exp (x j - c)|
      ≤ epsExp * Real.exp (((y j - mx : ℤ) : ℝ) / q32) + (4 / q32 + Real.exp (-16))
        + Real.exp (((y j - mx : ℤ) : ℝ) / q32) * (Real.exp δ - 1) := by
    intro j hj
    have h1 := (hfacts j hj).1
    have h2 := abs_exp_sub_exp_le (x j - c) (((y j - mx : ℤ) : ℝ) / q32)
    have hxu : x j - c - ((y j - mx : ℤ) : ℝ) / q32 = -((y j : ℝ) / q32 - x j) := by
      rw [hc]; push_cast; ring
    rw [hxu, abs_neg] at h2
    have h3 : Real.exp |(y j : ℝ) / q32 - x j| ≤ Real.exp δ := Real.exp_le_exp.2 (hδ j hj)
    have h4 : Real.exp (((y j - mx : ℤ) : ℝ) / q32) * (Real.exp |(y j : ℝ) / q32 - x j| - 1)
        ≤ Real.exp (((y j - mx : ℤ) : ℝ) / q32) * (Real.exp δ - 1) :=
      mul_le_mul_of_nonneg_left (by linarith) (Real.exp_pos _).le
    have h5 := abs_sub_le ((E j : ℝ) / q32) (Real.exp (((y j - mx : ℤ) : ℝ) / q32))
      (Real.exp (x j - c))
    rw [abs_sub_comm (Real.exp (((y j - mx : ℤ) : ℝ) / q32))] at h5
    linarith
  have hmain := attn_output_perturb n E W Vi T acc out (fun j => Real.exp (x j - c))
    (fun j => epsExp * Real.exp (((y j - mx : ℤ) : ℝ) / q32) + (4 / q32 + Real.exp (-16))
        + Real.exp (((y j - mx : ℤ) : ℝ) / q32) * (Real.exp δ - 1))
    lo hi V hT hTpos hW hacc hout hvb hV he hS hη
  have hsumη : ∑ j ∈ range n, (epsExp * Real.exp (((y j - mx : ℤ) : ℝ) / q32)
        + (4 / q32 + Real.exp (-16)) + Real.exp (((y j - mx : ℤ) : ℝ) / q32) * (Real.exp δ - 1))
      = (epsExp + (Real.exp δ - 1)) * (∑ j ∈ range n, Real.exp (((y j - mx : ℤ) : ℝ) / q32))
        + n * (4 / q32 + Real.exp (-16)) := by
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul, Finset.sum_const,
      Finset.card_range, nsmul_eq_mul]
    ring
  rw [denseAttnReal_shift_invariant m c, denseAttnReal_range c n x (fun j => (Vi j : ℝ) / q32)]
  rw [hsumη] at hmain
  exact hmain

/-- The closed softmax-stage factor: with `a = epsExp + (exp δ - 1)`, `ρ = 4/2^32 + exp(-16)`
and `s0 = 1 - epsExp - 4/2^32`, `a/(1 - epsExp) + (a/(1 - epsExp) + 1)·n·ρ/s0`. -/
def softmaxPerturbBound (n : ℕ) (δ : ℝ) : ℝ :=
  (epsExp + (Real.exp δ - 1)) / (1 - epsExp)
    + ((epsExp + (Real.exp δ - 1)) / (1 - epsExp) + 1) * ((n : ℝ) * (4 / q32 + Real.exp (-16)))
      / (1 - epsExp - 4 / q32)

/-- **Stage composition, closed form (proved).** The hypotheses of `attn_stage_compose`; the
executed-total and `U` terms are eliminated through `(1 - epsExp)·U - n·ρ ≤ T/2^32` and
`T/2^32 ≥ 1 - epsExp - 4/2^32`, giving
`1/(2·2^32) + n·V/(2·2^32) + (hi - lo)·softmaxPerturbBound n δ`. -/
theorem attn_stage_compose_closed (n : ℕ) (x : ℕ → ℝ) (y E W Vi : ℕ → ℤ) (mx T acc out : ℤ)
    (m lo hi V δ : ℝ)
    (hδ : ∀ j ∈ range n, |(y j : ℝ) / q32 - x j| ≤ δ)
    (hmax : ∃ j ∈ range n, y j = mx)
    (hnum : ∀ j ∈ range n, attnNumeratorCase (y j - mx) (E j))
    (hT : T = ∑ j ∈ range n, E j)
    (hW : ∀ j ∈ range n, W j = divRoundNearest (E j * (2 : ℤ) ^ (32 : ℕ)) T)
    (hacc : acc = ∑ j ∈ range n, W j * Vi j)
    (hout : out = divRoundNearest acc ((2 : ℤ) ^ (32 : ℕ)))
    (hvb : ∀ j ∈ range n, lo ≤ (Vi j : ℝ) / q32 ∧ (Vi j : ℝ) / q32 ≤ hi)
    (hV : ∀ j ∈ range n, |(Vi j : ℝ) / q32| ≤ V) :
    |(out : ℝ) / q32 - denseAttnReal m ((List.range n).map x)
        ((List.range n).map (fun j => (Vi j : ℝ) / q32))|
      ≤ 1 / (2 * q32) + n * V / (2 * q32) + (hi - lo) * softmaxPerturbBound n δ := by
  obtain ⟨-, hS'lb, hb⟩ := attn_stage_compose n x y E W Vi mx T acc out m lo hi V δ hδ hmax hnum
    hT hW hacc hout hvb hV
  obtain ⟨j0, hj0, -⟩ := hmax
  have hfacts := fun j hj => attn_numerator_facts (y j - mx) (E j) (hnum j hj)
  have hTsum : (T : ℝ) / q32 = ∑ j ∈ range n, (E j : ℝ) / q32 := by
    rw [hT]; push_cast; rw [Finset.sum_div]
  set U := ∑ j ∈ range n, Real.exp (((y j - mx : ℤ) : ℝ) / q32) with hU
  set S' := (T : ℝ) / q32 with hS'
  set a := epsExp + (Real.exp δ - 1) with ha
  set ρ := 4 / q32 + Real.exp (-16) with hρ
  set s0 := 1 - epsExp - 4 / q32 with hs0
  have hs0pos : 0 < s0 := attn_total_floor_pos
  have hS'pos : 0 < S' := lt_of_lt_of_le hs0pos hS'lb
  have h1ε : 0 < 1 - epsExp := by linarith [epsExp_lt_one]
  have hlower : (1 - epsExp) * U - n * ρ ≤ S' := by
    rw [hTsum, hU, Finset.mul_sum]
    have hsum : ∑ j ∈ range n, ((1 - epsExp) * Real.exp (((y j - mx : ℤ) : ℝ) / q32) - ρ)
        ≤ ∑ j ∈ range n, (E j : ℝ) / q32 := by
      apply Finset.sum_le_sum
      intro j hj
      have := (abs_le.1 (hfacts j hj).1).1
      linarith
    rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul] at hsum
    exact hsum
  have hδ0 : 0 ≤ δ := le_trans (abs_nonneg _) (hδ j0 hj0)
  have ha0 : 0 ≤ a := by
    have := Real.one_le_exp hδ0
    linarith [epsExp_pos]
  have hlohi : 0 ≤ hi - lo := by linarith [(hvb j0 hj0).1, (hvb j0 hj0).2]
  set A := a / (1 - epsExp) with hA
  have hA0 : 0 ≤ A := div_nonneg ha0 h1ε.le
  have hAe : A * (1 - epsExp) = a := div_mul_cancel₀ a (ne_of_gt h1ε)
  have hnρ : 0 ≤ (n : ℝ) * ρ := mul_nonneg (Nat.cast_nonneg n)
    (add_nonneg (div_pos (by norm_num) q32_pos).le (Real.exp_pos _).le)
  have hkey : a * U + n * ρ ≤ A * S' + (A + 1) * (n * ρ) := by
    have h := mul_le_mul_of_nonneg_left (show (1 - epsExp) * U ≤ S' + n * ρ by linarith) hA0
    have e1 : A * ((1 - epsExp) * U) = a * U := by rw [← mul_assoc, hAe]
    nlinarith
  have hdiv : (a * U + n * ρ) / S' ≤ A + (A + 1) * (n * ρ) / s0 := by
    calc (a * U + n * ρ) / S' ≤ (A * S' + (A + 1) * (n * ρ)) / S' :=
          div_le_div_of_nonneg_right hkey hS'pos.le
      _ = A + (A + 1) * (n * ρ) / S' := by
          rw [add_div, mul_div_assoc, div_self (ne_of_gt hS'pos), mul_one]
      _ ≤ A + (A + 1) * (n * ρ) / s0 := by
          have := div_le_div_of_nonneg_left
            (mul_nonneg (by linarith : (0 : ℝ) ≤ A + 1) hnρ) hs0pos hS'lb
          linarith
  have hfin : (hi - lo) * (a * U + n * ρ) / S' ≤ (hi - lo) * softmaxPerturbBound n δ := by
    rw [mul_div_assoc]
    apply mul_le_mul_of_nonneg_left _ hlohi
    refine le_trans hdiv (le_of_eq ?_)
    rw [softmaxPerturbBound]
  linarith

/-! ## The corrected end-to-end target (`anchor`) -/

/-- The score-stage error of one key against `idealScoreList`, `anchor` derivation, with the
row-dot magnitude expressed through the ideal score: `|dot| = √hdim · |ideal|`. Terms: row-dot
rounding `1/(2·2^32)` scaled by `|scale|`; the `fixed_rsqrt` error `epsExp/√hdim + 4/2^32`
scaled by `√hdim·idealAbs`; the scale epilogue `1/(2·2^32)`. (`scoreErr` in
`Numeric.EngineAttnRefinement` takes the row-dot magnitude; `attn_head_refines` passes the
scaled ideal score `listAbsMax sc` there, which is `√hdim` smaller.) -/
def scoreErrV2 (hdim idealAbs scaleAbs : ℝ) : ℝ :=
  scaleAbs * (1 / (2 * q32))
    + Real.sqrt hdim * idealAbs * (epsExp * (1 / Real.sqrt hdim) + 4 / q32)
    + 1 / (2 * q32)

/-- The corrected in-domain error target: `attn_stage_compose_closed` at `lo = -V`, `hi = V`. -/
def attnErrBoundV2 (n : ℕ) (V δ : ℝ) : ℝ :=
  1 / (2 * q32) + (n : ℝ) * V / (2 * q32) + 2 * V * softmaxPerturbBound n δ

/-- The corrected in-domain precondition: the reachable-state invariant of a `KvState::new` +
`append` state and the windows that keep every checked step of `attn_head` on its `Ok` arm. The
structural conjuncts hold on every reachable state (`attn_domain_v2_structural`,
`Numeric/KvReachable.lean`); the data windows remain hypotheses on the stored values.

* `attn_domain`;
* occupancy: `pos < occupied[layer]` (`src/engine/attn.rs:465-467`);
* scale: `state.scale` is the `fixed_rsqrt` result of `head_dim·2^32` (`KvState::new`);
* planes: `storage = attn.KvStorage.Q32 k v` with `k[layer] = kplane`, `v[layer] = vplane`, each
  `n_head_kv·cap·head_dim` long;
* per-key `row_dot_domain` of every causal key row against `q`;
* score window `|score_j| ≤ 2^62` (so `score_j - mx` is a checked `i64` subtraction);
* value window `|v_{j,i}| ≤ 2^62` and `pos + 1 ≤ 2^30` (so `|W_j·v_{j,i}| ≤ 2^95`, the `i128`
  accumulation and the `i64` output narrowing stay on their `Ok` arms).

The windows suffice: `Numeric/AttnHeadWindowsScore.lean` and `Numeric/AttnHeadWindowsValue.lean`
derive every loop window from them. -/
def attn_domain_v2 (state : attn.KvState) (layer q_head pos : Usize) (q out : Slice I64)
    (k v : alloc.vec.Vec (alloc.vec.Vec I64)) (kplane vplane : alloc.vec.Vec I64) : Prop :=
  attn_domain state layer q_head pos q out ∧
  (∃ len : Usize, state.occupied.val[layer.val]? = some len ∧ pos.val < len.val) ∧
  (∃ r : I64, r.val = (state.head_dim.val : ℤ) * (2 : ℤ) ^ (32 : ℕ) ∧
    word_fixed_rsqrt r = ok (core.result.Result.Ok state.scale)) ∧
  state.storage = attn.KvStorage.Q32 k v ∧
  k.val[layer.val]? = some kplane ∧ v.val[layer.val]? = some vplane ∧
  kplane.val.length = state.n_head_kv.val * state.cap.val * state.head_dim.val ∧
  vplane.val.length = state.n_head_kv.val * state.cap.val * state.head_dim.val ∧
  (∀ s, s ≤ pos.val → ∀ krow : Slice I64,
    krow.val = planeRow kplane.val state.n_head_kv.val state.head_dim.val
      (q_head.val / (state.n_head.val / state.n_head_kv.val)) s →
    row_dot_domain krow q) ∧
  (∀ sc ∈ scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
      (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val,
    |sc| ≤ (2 : ℤ) ^ (62 : ℕ)) ∧
  (∀ s, s ≤ pos.val → ∀ i, i < state.head_dim.val →
    |((planeRow vplane.val state.n_head_kv.val state.head_dim.val
      (q_head.val / (state.n_head.val / state.n_head_kv.val)) s).getD i 0#i64).val|
      ≤ (2 : ℤ) ^ (62 : ℕ)) ∧
  pos.val + 1 ≤ 2 ^ 30

/-- **Corrected end-to-end target.** On `attn_domain_v2`, with the `Q32` carrier guard
`carrierQ32 state` and `Q32` storage, `word_attn_head`
returns `Ok` and each output component is within `attnErrBoundV2` of the dense real softmax
attention of the ideal scores, at score error `scoreErrV2`. The bound is the conclusion of
`attn_stage_compose_closed`, whose hypotheses the four `attn_head_loop0..3` specifications supply
and whose `δ` the score-stage bound `attn_score_stage` supplies. The carrier guard and the storage
equation are the two hypotheses `Numeric.EngineAttnRefinement`'s `attn_head_refines` also carries
(`read_discriminant` is opaque, so neither is dischargeable from a concrete state). -/
def attn_head_refines_v2 : Prop :=
  ∀ (state : attn.KvState) (layer q_head pos : Usize) (q out : Slice I64)
    (k v : alloc.vec.Vec (alloc.vec.Vec I64)) (kplane vplane : alloc.vec.Vec I64),
    attn_domain_v2 state layer q_head pos q out k v kplane vplane →
    carrierQ32 state →
    state.storage = attn.KvStorage.Q32 k v →
    ∃ out' : Slice I64,
      word_attn_head state layer q_head q pos out = ok (core.result.Result.Ok (), out') ∧
      out'.val.length = out.val.length ∧
      ∀ i : ℕ, i < state.head_dim.val →
        let kvh := q_head.val / (state.n_head.val / state.n_head_kv.val)
        let hdim : ℝ := (state.head_dim.val : ℝ)
        let sc := idealScoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
          kvh hdim pos.val
        let vs := valueList vplane.val state.n_head_kv.val state.head_dim.val kvh pos.val i
        |toReal (out'.val.getD i 0#i64) - denseAttnReal (listAbsMax sc) sc (toRealList vs)|
          ≤ attnErrBoundV2 (pos.val + 1) (listAbsMax (toRealList vs))
              (scoreErrV2 hdim (listAbsMax sc) |toReal state.scale|)

/-! ## Joint non-vacuity of `attn_domain_v2` -/

/-- `2^32` is in the `fixed_rsqrt` window. -/
theorem attn_unit_radicand_window : rsqrt_window (4294967296#i64) := by
  unfold rsqrt_window
  constructor <;> simp

/-- The load-time scale of `head_dim = 1`: the `fixed_rsqrt` result at `1·2^32`, taken from
`fixed_rsqrt_refines_proved` (no evaluation of the extracted body). -/
def attnUnitScale : I64 :=
  Classical.choose (fixed_rsqrt_refines_proved (4294967296#i64) attn_unit_radicand_window)

/-- The one-wide zero row. -/
def attnZeroRow : Slice I64 := Slice.from [0#i64] (by simpa using attn_one_le_usize_max)

/-- The one-layer, one-head, one-position plane tables: one appended zero key row and one zero
value row. -/
def attnOnePlane : alloc.vec.Vec I64 :=
  alloc.vec.Vec.from [0#i64] (by simpa using attn_one_le_usize_max)

def attnOnePlanes : alloc.vec.Vec (alloc.vec.Vec I64) :=
  alloc.vec.Vec.from [attnOnePlane] (by simpa using attn_one_le_usize_max)

/-- `attnOnePlanes` at layer 0 is the zero plane. -/
theorem attnOnePlanes_index : attnOnePlanes.val[0]? = some attnOnePlane := rfl

/-- A one-layer, one-head, one-position state with one appended zero key/value and the
`fixed_rsqrt` scale. -/
def attnOneState : attn.KvState :=
  { carrier := mixer.Carrier.Q32, n_layer := 1#usize, n_head := 1#usize, n_head_kv := 1#usize,
    head_dim := 1#usize, cap := 1#usize, scale := attnUnitScale,
    occupied := alloc.vec.Vec.from [1#usize] (by simpa using attn_one_le_usize_max),
    storage := attn.KvStorage.Q32 attnOnePlanes attnOnePlanes }

/-- **Joint witness.** `attnOneState` at layer 0, head 0, position 0 satisfies every conjunct
of `attn_domain_v2` at once. -/
theorem attn_domain_v2_witness :
    attn_domain_v2 attnOneState 0#usize 0#usize 0#usize attnZeroRow attnZeroRow
      attnOnePlanes attnOnePlanes attnOnePlane attnOnePlane := by
  have hspec := Classical.choose_spec
    (fixed_rsqrt_refines_proved (4294967296#i64) attn_unit_radicand_window)
  refine ⟨?_, ⟨1#usize, by simp [attnOneState], by simp⟩, ⟨4294967296#i64, by simp [attnOneState],
    hspec.1⟩, by simp [attnOneState], attnOnePlanes_index, attnOnePlanes_index,
    by simp [attnOneState, attnOnePlane], by simp [attnOneState, attnOnePlane], ?_, ?_, ?_, by simp⟩
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      simp [attnOneState, attnZeroRow, Slice.from_val]
  · intro s hs krow hk
    simp at hs
    subst hs
    simp [attnOneState, attnOnePlanes, planeRow] at hk
    refine ⟨?_, ?_, ?_, ?_⟩
    · simp [hk, attnZeroRow, Slice.from_val]
    · intro k hk1 hk2
      simp [hk] at hk2
      obtain rfl : k = 1 := by omega
      simp [dotPrefix, hk, attnZeroRow, Slice.from_val]
      norm_num [I128.min, IScalar.min, I128.numBits, I128.max, IScalar.max]
    · simp [dotPrefix, hk, attnZeroRow, Slice.from_val, divRoundNearest]
    · simp [dotPrefix, hk, attnZeroRow, Slice.from_val, divRoundNearest]
  · intro sc hsc
    simp [scoreList, planeRow, attnOneState, attnZeroRow, Slice.from_val, attnScoreInt,
      divRoundNearest] at hsc
    subst hsc
    norm_num
  · intro s hs i hi
    simp [attnOneState] at hs hi
    subst hs; subst hi
    simp [planeRow, attnOneState, attnOnePlane]

end

end apeiron_word
