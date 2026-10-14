/-
Attention score-stage error (`APEIRON-ALGEBRA-G3-ATTN-W3-T1b`).

The per-key score error `δ` of `attn_head_refines_v2` (`Numeric.AttnSoftmaxPerturb`): the score
stage `raw = round(Σ q·k / 2^32)`, `score = round(raw·scale / 2^32)` (`attnScoreInt`) against the
ideal real score `<q, k> / (2^64 · √head_dim)` (`idealScoreList`), with the load-time scale
`fixed_rsqrt(head_dim·2^32)` bounded by `fixed_rsqrt_refines_proved`.

Landed here:

* `attn_scale_bound`: on `word_fixed_rsqrt r = ok (Ok scale)` at `r.val = head_dim·2^32`, the scale
  satisfies `|toReal scale − 1/√head_dim| ≤ epsExp·(1/√head_dim) + 4/2^32`
  (`fixed_rsqrt_refines_proved`, with the window forced by the success branch);
* `listAbsMax_nonneg`/`listAbsMax_ge`: the fold `max_j |x_j|` is a nonnegative upper bound;
* `attnScoreInt_err`: the two roundings of the score stage against `(D/2^64)·r`;
* `attn_score_stage`: the composition of the above with `listAbsMax` to `scoreErrV2`.

The module is a pure `ℝ`/`ℤ` statement over the Q32.32 denotation in `Numeric.Scale`; it does not
read the extracted loops. Trust boundary: it does not cover rustc/LLVM codegen of
`src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.Attn.SoftmaxPerturb

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The load-time scale -/

/-- **Scale bound.** If the extracted `fixed_rsqrt` succeeds on the load-time radicand
`head_dim·2^32` with result `scale`, then `scale` is within the `fixed_rsqrt` window error of
`1/√head_dim`. The radicand is in the `rsqrt_window` whenever the success branch is taken
(`fixed_rsqrt_refuses_proved` refuses outside it), so `fixed_rsqrt_refines_proved` applies and the
success value is the same by determinism of `ok`. -/
theorem attn_scale_bound (hd : ℕ) (hhd : 0 < hd) (r scale : I64)
    (hr : r.val = (hd : ℤ) * (2 : ℤ) ^ (32 : ℕ))
    (hwr : word_fixed_rsqrt r = ok (core.result.Result.Ok scale)) :
    |toReal scale - 1 / Real.sqrt (hd : ℝ)| ≤ epsExp * (1 / Real.sqrt (hd : ℝ)) + 4 / q32 := by
  have hwin : rsqrt_window r := by
    by_contra hc
    have href := fixed_rsqrt_refuses_proved r hc
    have hEq : core.result.Result.Ok scale
        = core.result.Result.Err fixed_rsqrt.RsqrtError.OutOfRange := by
      simpa using hwr.symm.trans href
    exact nomatch hEq
  obtain ⟨v, hv, hb⟩ := fixed_rsqrt_refines_proved r hwin
  have hvs : v = scale := by
    have h := hv.symm.trans hwr
    simpa using h
  rw [hvs] at hb
  have hrR : toReal r = (hd : ℝ) := by
    unfold toReal q32
    rw [hr]
    push_cast
    field_simp
    norm_num
  rw [hrR] at hb
  exact hb

/-! ## The fold upper bound -/

/-- `listAbsMax l = max_j |x_j|` is nonnegative. -/
theorem listAbsMax_nonneg (l : List ℝ) : 0 ≤ listAbsMax l := by
  induction l with
  | nil => simp [listAbsMax]
  | cons a t ih =>
      rw [listAbsMax, List.foldr_cons]
      exact le_trans ih (le_max_right _ _)

/-- `listAbsMax l` dominates every `|x|` with `x ∈ l`. -/
theorem listAbsMax_ge (l : List ℝ) (x : ℝ) (hx : x ∈ l) : |x| ≤ listAbsMax l := by
  induction l with
  | nil => simp at hx
  | cons a t ih =>
      rw [List.mem_cons] at hx
      rw [listAbsMax, List.foldr_cons]
      rcases hx with rfl | hx
      · exact le_max_left _ _
      · exact le_trans (ih hx) (le_max_right _ _)

/-! ## The score-stage integer error -/

/-- `divRoundNearest n d` is within `1/2` of the real quotient `n/d`. -/
theorem attnScore_round_close (n d : ℤ) (hd : 0 < d) :
    |((divRoundNearest n d : ℤ) : ℝ) - (n : ℝ) / (d : ℝ)| ≤ 1 / 2 := by
  have h := divRoundNearest_real_bound n d hd
  have hdR : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  have hdeq : ((divRoundNearest n d : ℤ) : ℝ) - (n : ℝ) / (d : ℝ)
      = (((divRoundNearest n d : ℤ) : ℝ) * (d : ℝ) - (n : ℝ)) / (d : ℝ) := by
    field_simp
  rw [hdeq, abs_div, abs_of_pos hdR, div_le_iff₀ hdR]
  linarith

/-- **Score-stage integer error.** The two roundings `raw = round(D/2^32)`,
`attnScoreInt D scale = round(raw·scale/2^32)` differ from `(D/2^64)·r` by the raw-rounding term
`|scale/2^32|·(1/(2·2^32))`, the scale-error term `|D/2^64|·|scale/2^32 − r|` and the epilogue
`1/(2·2^32)`. -/
theorem attnScoreInt_err (D scale : ℤ) (r : ℝ) :
    |((attnScoreInt D scale : ℤ) : ℝ) / q32 - ((D : ℝ) / q32 ^ 2) * r|
      ≤ |(scale : ℝ) / q32| * (1 / (2 * q32))
        + |(D : ℝ) / q32 ^ 2| * |(scale : ℝ) / q32 - r| + 1 / (2 * q32) := by
  set Q : ℤ := (2 : ℤ) ^ (32 : ℕ) with hQ
  set a : ℤ := divRoundNearest D Q with ha
  have hQR : (Q : ℝ) = q32 := by rw [hQ]; norm_num [q32]
  have hA : |(a : ℝ) - (D : ℝ) / q32| ≤ 1 / 2 := by
    have h := attnScore_round_close D Q (by rw [hQ]; norm_num)
    simpa [← ha, hQR] using h
  have hb : attnScoreInt D scale = divRoundNearest (a * scale) Q := by
    rw [attnScoreInt]
  have hB : |((attnScoreInt D scale : ℤ) : ℝ) - (a : ℝ) * (scale : ℝ) / q32| ≤ 1 / 2 := by
    have h := attnScore_round_close (a * scale) Q (by rw [hQ]; norm_num)
    rw [hb]
    simpa [hQR, Int.cast_mul] using h
  set x1 : ℝ := (1 / q32)
      * (((attnScoreInt D scale : ℤ) : ℝ) - (a : ℝ) * (scale : ℝ) / q32) with hx1'
  set x2 : ℝ := ((scale : ℝ) / q32 ^ 2) * ((a : ℝ) - (D : ℝ) / q32) with hx2'
  set x3 : ℝ := ((D : ℝ) / q32 ^ 2) * ((scale : ℝ) / q32 - r) with hx3'
  have hkey : ((attnScoreInt D scale : ℤ) : ℝ) / q32 - ((D : ℝ) / q32 ^ 2) * r
      = x1 + x2 + x3 := by
    rw [hx1', hx2', hx3']
    field_simp [q32_ne_zero]
    ring
  have hx1 : |x1| ≤ 1 / (2 * q32) := by
    rw [hx1', abs_mul, abs_of_pos (show (0 : ℝ) < 1 / q32 by positivity)]
    calc (1 / q32)
          * |((attnScoreInt D scale : ℤ) : ℝ) - (a : ℝ) * (scale : ℝ) / q32|
        ≤ (1 / q32) * (1 / 2) := mul_le_mul_of_nonneg_left hB (by positivity)
      _ = 1 / (2 * q32) := by ring
  have hx2 : |x2| ≤ |(scale : ℝ) / q32| * (1 / (2 * q32)) := by
    rw [hx2', abs_mul]
    calc |(scale : ℝ) / q32 ^ 2| * |(a : ℝ) - (D : ℝ) / q32|
        ≤ |(scale : ℝ) / q32 ^ 2| * (1 / 2) := mul_le_mul_of_nonneg_left hA (abs_nonneg _)
      _ = |(scale : ℝ) / q32| * (1 / (2 * q32)) := by
          rw [abs_div, abs_of_pos (pow_pos q32_pos 2), abs_div, abs_of_pos q32_pos]
          field_simp [q32_ne_zero]
  have hx3 : |x3| ≤ |(D : ℝ) / q32 ^ 2| * |(scale : ℝ) / q32 - r| := by
    rw [hx3', abs_mul]
  have htri : |x1 + x2 + x3| ≤ |x1| + |x2| + |x3| := by
    have h1 := abs_add_le (x1 + x2) x3
    have h2 := abs_add_le x1 x2
    linarith
  rw [hkey]
  linarith [htri, hx1, hx2, hx3]

/-! ## The score-stage composition -/

/-- The integer dot product and its real cast agree, coordinatewise. -/
theorem attnScoreStage_dot_cast (l : List (I64 × I64)) :
    (((l.map (fun p => p.1.val * p.2.val)).sum : ℤ) : ℝ)
      = (l.map (fun p => (p.1.val : ℝ) * (p.2.val : ℝ))).sum := by
  induction l with
  | nil => simp
  | cons a t ih =>
      simp only [List.map_cons, List.sum_cons]
      rw [Int.cast_add, ih]
      push_cast
      ring

/-- The `s`-th realised score is `attnScoreInt` of the `s`-th integer dot. -/
theorem attnScoreStage_scoreList_getD (kp q : List I64) (kv hd kvh : ℕ) (scale : ℤ)
    (pos s : ℕ) (hs : s ≤ pos) :
    (scoreList kp q kv hd kvh scale pos).getD s 0
      = attnScoreInt (((planeRow kp kv hd kvh s).zip q).map (fun p => p.1.val * p.2.val)).sum
          scale := by
  have hslen : s < (scoreList kp q kv hd kvh scale pos).length := by
    rw [scoreList, List.length_map, List.length_range]
    omega
  rw [List.getD_eq_getElem (scoreList kp q kv hd kvh scale pos) (0 : ℤ) hslen]
  simp only [scoreList, List.getElem_map, List.getElem_range]

/-- The `s`-th ideal score is the real quotient of the `s`-th dot over `2^64·√head_dim`. -/
theorem attnScoreStage_idealScoreList_getD (kp q : List I64) (kv hd kvh : ℕ) (hdim : ℝ)
    (pos s : ℕ) (hs : s ≤ pos) :
    (idealScoreList kp q kv hd kvh hdim pos).getD s 0
      = ((((planeRow kp kv hd kvh s).zip q).map
            (fun p => (p.1.val : ℝ) * (p.2.val : ℝ))).sum / q32 ^ 2) / Real.sqrt hdim := by
  have hslen : s < (idealScoreList kp q kv hd kvh hdim pos).length := by
    rw [idealScoreList, List.length_map, List.length_range]
    omega
  rw [List.getD_eq_getElem (idealScoreList kp q kv hd kvh hdim pos) (0 : ℝ) hslen]
  simp only [idealScoreList, List.getElem_map, List.getElem_range]

/-- **Score-stage error.** At each causal key `s ≤ pos`, the realised Q32.32 score is within
`scoreErrV2` of the ideal real score: the raw-dot rounding scaled by `|toReal scale|`, the
`fixed_rsqrt` scale error scaled by `√head_dim·listAbsMax`, and the epilogue. The dot magnitude is
recovered from the ideal score through `|D/2^64| = √head_dim·|ideal| ≤ √head_dim·listAbsMax`. -/
theorem attn_score_stage (hd : ℕ) (hhd : 0 < hd) (r scale : I64)
    (hr : r.val = (hd : ℤ) * (2 : ℤ) ^ (32 : ℕ))
    (hwr : word_fixed_rsqrt r = ok (core.result.Result.Ok scale))
    (kp q : List I64) (kv kvh pos : ℕ) :
    ∀ s, s ≤ pos →
      |((scoreList kp q kv hd kvh scale.val pos).getD s 0 : ℝ) / q32
        - (idealScoreList kp q kv hd kvh (hd : ℝ) pos).getD s 0|
      ≤ scoreErrV2 (hd : ℝ)
          (listAbsMax (idealScoreList kp q kv hd kvh (hd : ℝ) pos)) |toReal scale| := by
  intro s hs
  set D : ℤ := (((planeRow kp kv hd kvh s).zip q).map (fun p => p.1.val * p.2.val)).sum with hD
  have hDcast : (D : ℝ)
      = (((planeRow kp kv hd kvh s).zip q).map
          (fun p => (p.1.val : ℝ) * (p.2.val : ℝ))).sum := by
    simpa [hD] using attnScoreStage_dot_cast ((planeRow kp kv hd kvh s).zip q)
  rw [attnScoreStage_scoreList_getD kp q kv hd kvh scale.val pos s hs,
    attnScoreStage_idealScoreList_getD kp q kv hd kvh (hd : ℝ) pos s hs]
  rw [← hDcast]
  rw [show ((D : ℝ) / q32 ^ 2) / Real.sqrt (hd : ℝ)
      = ((D : ℝ) / q32 ^ 2) * (1 / Real.sqrt (hd : ℝ)) by rw [div_eq_mul_one_div]]
  have hslen : s < (idealScoreList kp q kv hd kvh (hd : ℝ) pos).length := by
    rw [idealScoreList, List.length_map, List.length_range]
    omega
  have hmem : (idealScoreList kp q kv hd kvh (hd : ℝ) pos).getD s 0
      ∈ idealScoreList kp q kv hd kvh (hd : ℝ) pos := by
    rw [List.getD_eq_getElem _ (0 : ℝ) hslen]
    exact List.getElem_mem hslen
  have hge := listAbsMax_ge _ _ hmem
  have hid : (idealScoreList kp q kv hd kvh (hd : ℝ) pos).getD s 0
      = ((D : ℝ) / q32 ^ 2) / Real.sqrt (hd : ℝ) := by
    rw [attnScoreStage_idealScoreList_getD kp q kv hd kvh (hd : ℝ) pos s hs, ← hDcast]
  have hge' : |((D : ℝ) / q32 ^ 2) / Real.sqrt (hd : ℝ)|
      ≤ listAbsMax (idealScoreList kp q kv hd kvh (hd : ℝ) pos) := by
    rw [← hid]; exact hge
  have hsqrt_pos : 0 < Real.sqrt (hd : ℝ) := Real.sqrt_pos.mpr (by exact_mod_cast hhd)
  have hDbound : |(D : ℝ) / q32 ^ 2|
      ≤ Real.sqrt (hd : ℝ)
          * listAbsMax (idealScoreList kp q kv hd kvh (hd : ℝ) pos) := by
    have hid2 : Real.sqrt (hd : ℝ) * |((D : ℝ) / q32 ^ 2) / Real.sqrt (hd : ℝ)|
        = |(D : ℝ) / q32 ^ 2| := by
      rw [abs_div, abs_of_pos hsqrt_pos]
      rw [mul_div_cancel₀ _ (ne_of_gt hsqrt_pos)]
    rw [← hid2]
    exact mul_le_mul_of_nonneg_left hge' (le_of_lt hsqrt_pos)
  have hscale' : |(scale.val : ℝ) / q32 - 1 / Real.sqrt (hd : ℝ)|
      ≤ epsExp * (1 / Real.sqrt (hd : ℝ)) + 4 / q32 := by
    simpa [toReal] using attn_scale_bound hd hhd r scale hr hwr
  have hmid : |(D : ℝ) / q32 ^ 2| * |(scale.val : ℝ) / q32 - 1 / Real.sqrt (hd : ℝ)|
      ≤ (Real.sqrt (hd : ℝ)
            * listAbsMax (idealScoreList kp q kv hd kvh (hd : ℝ) pos))
          * (epsExp * (1 / Real.sqrt (hd : ℝ)) + 4 / q32) :=
    mul_le_mul hDbound hscale' (abs_nonneg _)
      (mul_nonneg (le_of_lt hsqrt_pos) (listAbsMax_nonneg _))
  have h3 := attnScoreInt_err D scale.val (1 / Real.sqrt (hd : ℝ))
  have hsq : |(scale.val : ℝ) / q32| = |toReal scale| := by rw [toReal]
  rw [hsq] at h3
  have h3' : |((attnScoreInt D scale.val : ℤ) : ℝ) / q32
        - ((D : ℝ) / q32 ^ 2) * (1 / Real.sqrt (hd : ℝ))|
      ≤ |toReal scale| * (1 / (2 * q32))
        + (Real.sqrt (hd : ℝ)
            * listAbsMax (idealScoreList kp q kv hd kvh (hd : ℝ) pos))
          * (epsExp * (1 / Real.sqrt (hd : ℝ)) + 4 / q32) + 1 / (2 * q32) := by
    linarith [h3, hmid]
  rw [scoreErrV2]
  exact h3'

end

end apeiron_word
