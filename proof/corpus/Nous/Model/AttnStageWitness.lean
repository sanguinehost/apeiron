/-
# Unconditional attention-stage witnesses

`Nous/Model/AttnStage.lean` assembles the single-head attention `Stage` (`attnHeadStage`) but leaves
the two uniform bounds it consumes as hypotheses: `hVb` (the value bound `V`) and `hδb` (the
score-stage bound `δ`). Its domain-nonvacuity theorem `attnHeadStage_dom_witness` is therefore
conditional on `hVb`/`hδb`.

This module discharges both hypotheses for the concrete one-layer, one-head, one-position state
`attnOneState` (`Numeric/AttnSoftmaxPerturb.lean:734`) and its appended value row:

* key plane `[0]`, query/output `attnZeroRow`, so every causal score is `0`; the score list collapses
  to `[0]` and `hδb` holds at `δ = scoreErrV2 1 0 |toReal attnUnitScale|` for every query on the
  domain;
* value plane `[0]`, so `hVb` holds at `V = 0` — the degenerate case;
* `attnHeadStage_witness`: the stage is constructible and its domain contains `gridVec attnZeroRow`,
  with no hypotheses.

A second, nondegenerate witness (`attnNZState`) keeps the key row zero but carries the value row
`2^32` in the value plane. Its `hVb` discharges at `V = 2^32 > 0`, and `attnNZState_witness` exhibits
the stage and a domain point. `attn_domain_v2` is re-established for this state (the changed value
window is the only new conjunct).

## Trust boundary

Kernel-checked `ℝ`/`ℤ` algebra over the same denotations as `Nous/Model/AttnStage.lean`, plus the
reused `attn_domain_v2_witness`. Does not cover rustc/LLVM codegen of `src/engine/attn.rs`, the
kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Nous.Model.AttnStage

namespace Nous.Model

open apeiron_word Aeneas Aeneas.Std Result
open MatvecBridge
open Classical

noncomputable section

/-! ## The one-wide witness planes -/

/-- The one-wide zero plane, the inner value row of `attnOneState`. -/
abbrev attnZeroPlane : alloc.vec.Vec I64 :=
  alloc.vec.Vec.from [0#i64] (by simpa using attn_one_le_usize_max)

/-- The one-wide value plane carrying `2^32`. -/
abbrev attnPow32Plane : alloc.vec.Vec I64 :=
  alloc.vec.Vec.from [4294967296#i64] (by simpa using attn_one_le_usize_max)

/-! ## Zero-plane geometry -/

/-- `getD` on the one-wide zero row is the zero scalar, for every index. -/
theorem attnZero_getD (s : ℕ) : (([0#i64] : List I64).getD s 0#i64) = 0#i64 := by
  cases s with
  | zero => simp
  | succ k => simp

/-- The zero row of width `n`, read at any index, is the zero scalar. -/
theorem replicate_getD_zero (n j : ℕ) :
    (List.replicate n 0#i64).getD j 0#i64 = 0#i64 := by
  induction n generalizing j with
  | zero => simp
  | succ k ih =>
      rw [List.replicate_succ]
      cases j with
      | zero => simp
      | succ m => rw [List.getD_cons_succ]; exact ih m

/-- The `s`-th row of the one-wide zero plane is the zero row of width `n`. -/
theorem planeRow_zeroRow (n kv s : ℕ) :
    planeRow [0#i64] kv n 0 s = List.replicate n 0#i64 := by
  unfold planeRow
  simp

/-- Every ideal score against the zero key row is `0`. -/
theorem keyRowScore_zeroRow (n kv s : ℕ) (hdim : ℝ) (y : Fin n → ℝ) :
    keyRowScore (hd := n) [0#i64] kv 0 s hdim y = 0 := by
  unfold keyRowScore
  have hrow : ∀ i : Fin n,
      toReal ((planeRow [0#i64] kv n 0 s).getD i.val 0#i64) = 0 := by
    intro i
    rw [planeRow_zeroRow n kv s, replicate_getD_zero n i.val]
    simp [toReal]
  have hterm : ∀ i : Fin n,
      toReal ((planeRow [0#i64] kv n 0 s).getD i.val 0#i64) * y i = 0 :=
    fun i => by rw [hrow i, zero_mul]
  rw [Finset.sum_eq_zero (fun i _ => hterm i), zero_div]

/-- The causal score list against the zero key row is a list of zeros. -/
theorem attnScoreList_zeroRow (n kv pos : ℕ) (hdim : ℝ) (y : Fin n → ℝ) :
    attnScoreList (hd := n) [0#i64] kv 0 pos hdim y = List.replicate (pos + 1) 0 := by
  unfold attnScoreList
  simp [keyRowScore_zeroRow]

/-- The one-key causal score list against the zero key row is `[0]`, for every query. -/
theorem attnScoreList_zeroRow_one (n kv : ℕ) (hdim : ℝ) (y : Fin n → ℝ) :
    attnScoreList (hd := n) [0#i64] kv 0 0 hdim y = [0] := by
  rw [attnScoreList_zeroRow n kv 0 hdim y]
  simp

/-- The one-key value column of the zero plane is the zero list, for every component. -/
theorem valueList_zeroRow (n kv i : ℕ) : valueList [0#i64] kv n 0 0 i = [0] := by
  simp [valueList, planeRow]

/-- The largest absolute value of the singleton zero list is `0`. -/
theorem listAbsMax_singleton_zero : listAbsMax ([0] : List ℝ) = 0 := by
  simp [listAbsMax]

/-- The Q32.32 denotation of the singleton zero list is the singleton zero list. -/
theorem toRealList_singleton_zero : toRealList ([0] : List ℤ) = [0] := by
  simp [toRealList]

/-! ## The degenerate witness: zero value row, `V = 0` -/

/-- The `0`-plane **value bound** for `attnOneState`: every appended value is zero. -/
theorem attnOneState_hVb : ∀ i : Fin attnOneState.head_dim.val,
    listAbsMax (toRealList (valueList attnZeroPlane.val attnOneState.n_head_kv.val
      attnOneState.head_dim.val (kvhOf attnOneState 0#usize) 0 i.val)) ≤ 0 := by
  intro i
  have hsub : valueList attnZeroPlane.val attnOneState.n_head_kv.val attnOneState.head_dim.val
      (kvhOf attnOneState 0#usize) 0 i.val = [0] := by
    simpa [attnOneState, attnZeroPlane, kvhOf] using
      valueList_zeroRow attnOneState.head_dim.val attnOneState.n_head_kv.val i.val
  rw [hsub, toRealList_singleton_zero, listAbsMax_singleton_zero]

/-- The `0`-plane **score bound** for `attnOneState` at `δ = scoreErrV2 1 0 |toReal attnUnitScale|`:
the one-key score list is `[0]` for every query on the domain. -/
theorem attnOneState_hδb : ∀ y ∈ attnStageDom attnOneState 0#usize 0#usize 0#usize
      attnZeroPlane attnZeroPlane attnOneState.head_dim.val 0,
    scoreErrV2 (attnOneState.head_dim.val : ℝ)
      (listAbsMax (attnScoreList (hd := attnOneState.head_dim.val) attnZeroPlane.val
        attnOneState.n_head_kv.val (kvhOf attnOneState 0#usize) 0
        (attnOneState.head_dim.val : ℝ) y))
      |toReal attnOneState.scale|
    ≤ scoreErrV2 1 0 |toReal attnUnitScale| := by
  intro y _
  have hsc : attnScoreList (hd := attnOneState.head_dim.val) attnZeroPlane.val
      attnOneState.n_head_kv.val (kvhOf attnOneState 0#usize) 0
      (attnOneState.head_dim.val : ℝ) y = [0] := by
    simpa [attnOneState, attnZeroPlane, kvhOf] using
      attnScoreList_zeroRow_one attnOneState.head_dim.val attnOneState.n_head_kv.val
        (attnOneState.head_dim.val : ℝ) y
  rw [hsc, listAbsMax_singleton_zero]
  simp [attnOneState]

/-- **Unconditional degenerate witness.** The one-head stage of `attnOneState` is constructible at
`V = 0` and `δ = scoreErrV2 1 0 |toReal attnUnitScale|`, and its domain contains the zero row. -/
theorem attnHeadStage_witness :
    MatvecBridge.gridVec (n := attnOneState.head_dim.val) attnZeroRow ∈
      (attnHeadStage attnOneState 0#usize 0#usize 0#usize attnZeroPlane attnZeroPlane
        (by simp [attnOneState]) 0 0 (scoreErrV2 1 0 |toReal attnUnitScale|)
        (le_refl (0 : ℝ)) (le_refl (0 : ℝ))
        (scoreErrV2_nonneg (le_refl (0 : ℝ)) (abs_nonneg _))
        attnOneState_hVb attnOneState_hδb).dom := by
  exact attnHeadStage_dom_witness (by simp [attnOneState]) 0 0
    (scoreErrV2 1 0 |toReal attnUnitScale|) (le_refl (0 : ℝ)) (le_refl (0 : ℝ))
    (scoreErrV2_nonneg (le_refl (0 : ℝ)) (abs_nonneg _)) attnOneState_hVb attnOneState_hδb

/-! ## The nondegenerate witness: nonzero value row, `V = 2^32` -/

/-- The one-key value column of the `2^32` plane at component `0` is `[2^32]`. -/
theorem attnPow32_valueList_zero :
    valueList [4294967296#i64] 1 1 0 0 0 = [4294967296] := by
  simp [valueList, planeRow]

/-- The Q32.32 denotation of the singleton `2^32` list is the singleton `1` list. -/
theorem toRealList_pow32 : toRealList ([4294967296] : List ℤ) = [1] := by
  norm_num [toRealList, toReal, q32]

/-- The largest absolute value of the singleton `1` list is `1`. -/
theorem listAbsMax_singleton_one : listAbsMax ([1] : List ℝ) = 1 := by
  simp [listAbsMax]

/-- A one-layer, one-head, one-position state with a zero key row and the nonzero value row `2^32`
and the `fixed_rsqrt` scale. -/
def attnNZState : attn.KvState :=
  { n_layer := 1#usize, n_head := 1#usize, n_head_kv := 1#usize, head_dim := 1#usize,
    cap := 1#usize, scale := attnUnitScale,
    occupied := alloc.vec.Vec.from [1#usize] (by simpa using attn_one_le_usize_max),
    k := alloc.vec.Vec.from [attnZeroPlane] (by simpa using attn_one_le_usize_max),
    v := alloc.vec.Vec.from [attnPow32Plane] (by simpa using attn_one_le_usize_max) }

/-- The nonzero value row is a valid `attn_domain_v2` point: every conjunct holds, with the value
window `|2^32| ≤ 2^62` the only conjunct changed from `attn_domain_v2_witness`. -/
theorem attnNZState_domain :
    attn_domain_v2 attnNZState 0#usize 0#usize 0#usize attnZeroRow attnZeroRow
      attnZeroPlane attnPow32Plane := by
  have hspec := Classical.choose_spec
    (fixed_rsqrt_refines_proved (4294967296#i64) attn_unit_radicand_window)
  refine ⟨?_, ⟨1#usize, by simp [attnNZState], by simp⟩, ⟨4294967296#i64,
    by simp [attnNZState], hspec.1⟩, by simp [attnNZState, attnZeroPlane],
    by simp [attnNZState, attnPow32Plane], by simp [attnNZState, attnZeroPlane],
    by simp [attnNZState, attnPow32Plane], ?_, ?_, ?_, by simp⟩
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      simp [attnNZState, attnZeroRow, Slice.from_val]
  · intro s hs krow hk
    simp at hs
    subst hs
    simp [attnNZState, planeRow, attnZeroPlane] at hk
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
    simp [scoreList, planeRow, attnNZState, attnZeroPlane, attnZeroRow, Slice.from_val,
      attnScoreInt, divRoundNearest] at hsc
    subst hsc
    norm_num
  · intro s hs i hi
    simp [attnNZState] at hs hi
    subst hs; subst hi
    simp [planeRow, attnNZState, attnPow32Plane]

/-- The `2^32` plane **value bound** for `attnNZState` at `V = 2^32 > 0`. -/
theorem attnNZState_hVb : ∀ i : Fin attnNZState.head_dim.val,
    listAbsMax (toRealList (valueList attnPow32Plane.val attnNZState.n_head_kv.val
      attnNZState.head_dim.val (kvhOf attnNZState 0#usize) 0 i.val)) ≤ 4294967296 := by
  intro i
  have hd : attnNZState.head_dim.val = 1 := by simp [attnNZState]
  have hi : i.val = 0 := by have := i.isLt; omega
  rw [hi]
  have hv : valueList attnPow32Plane.val attnNZState.n_head_kv.val attnNZState.head_dim.val
      (kvhOf attnNZState 0#usize) 0 0 = [4294967296] := by
    simpa [attnNZState, attnPow32Plane, kvhOf] using attnPow32_valueList_zero
  rw [hv, toRealList_pow32, listAbsMax_singleton_one]
  norm_num

/-- The `2^32` plane **score bound** for `attnNZState`: the key row is still zero, so the score list
collapses to `[0]` for every query on the domain. -/
theorem attnNZState_hδb : ∀ y ∈ attnStageDom attnNZState 0#usize 0#usize 0#usize
      attnZeroPlane attnPow32Plane attnNZState.head_dim.val 0,
    scoreErrV2 (attnNZState.head_dim.val : ℝ)
      (listAbsMax (attnScoreList (hd := attnNZState.head_dim.val) attnZeroPlane.val
        attnNZState.n_head_kv.val (kvhOf attnNZState 0#usize) 0
        (attnNZState.head_dim.val : ℝ) y))
      |toReal attnNZState.scale|
    ≤ scoreErrV2 1 0 |toReal attnUnitScale| := by
  intro y _
  have hsc : attnScoreList (hd := attnNZState.head_dim.val) attnZeroPlane.val
      attnNZState.n_head_kv.val (kvhOf attnNZState 0#usize) 0
      (attnNZState.head_dim.val : ℝ) y = [0] := by
    simpa [attnNZState, attnZeroPlane, kvhOf] using
      attnScoreList_zeroRow_one attnNZState.head_dim.val attnNZState.n_head_kv.val
        (attnNZState.head_dim.val : ℝ) y
  rw [hsc, listAbsMax_singleton_zero]
  simp [attnNZState]

/-- The nondegenerate value bound is positive. -/
theorem attnNZState_V_pos : 0 < (4294967296 : ℝ) := by norm_num

/-- **Unconditional nondegenerate witness.** The one-head stage of `attnNZState` is constructible at
`V = 2^32 > 0` and `δ = scoreErrV2 1 0 |toReal attnUnitScale|`, and its domain contains the zero
row. -/
theorem attnNZState_witness :
    MatvecBridge.gridVec (n := attnNZState.head_dim.val) attnZeroRow ∈
      (attnHeadStage attnNZState 0#usize 0#usize 0#usize attnZeroPlane attnPow32Plane
        (by simp [attnNZState]) 0 4294967296 (scoreErrV2 1 0 |toReal attnUnitScale|)
        (le_refl (0 : ℝ)) (by norm_num)
        (scoreErrV2_nonneg (le_refl (0 : ℝ)) (abs_nonneg _))
        attnNZState_hVb attnNZState_hδb).dom := by
  refine ⟨(attnZeroRow, attnZeroRow), rfl, attnNZState_domain, ?_⟩
  intro i
  have hz : attnZeroRow.val.getD i.val 0#i64 = 0#i64 := by
    cases i.val with
    | zero => simp [attnZeroRow, Slice.from_val]
    | succ k => simp [attnZeroRow, Slice.from_val]
  rw [hz]
  simp [toReal]

end

end Nous.Model
