/-
Attention weight and value-side windows (`APEIRON-ALGEBRA-G3-ATTN-W3-T1c`).

Discharges the weight-loop window (`AttnWeightWindow`), the value-loop window (`AttnValueWindow`)
and the per-component output composition from the numerator-loop output and `attn_domain_v2`,
with no new domain conjunct.

* `attn_weight_bounds`: the pure `ℤ` weight bound `0 ≤ W_j ≤ 2^32`, `Σ_{j<n} W_j ≤ 2^32 + n`
  for `W_j = divRoundNearest (E_j·2^32) T`, `T = Σ E_j`;
* `attn_weight_window_of_numerators`: `AttnWeightWindow` from the numerator-loop output;
* `attn_value_fit`: the `value_component_spec` narrowing (`hlo`/`hhi`/`hfit`) from the weight and
  value windows;
* `plane_at_spec`: the checked `usize` address identity of `attn.plane_at`;
* `attn_value_window_of_v2`: `AttnValueWindow` from `attn_domain_v2` and the weight bounds;
* `attn_value_components`: the per-component `value_component` result.

Trust boundary: reaches the extracted `attn.plane_at`, `attn.value_component`,
`attn.KvState.attn_head_loop2` / `_loop3` of `proof/ApeironWord.lean`. Does not cover
rustc/LLVM codegen of `src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe`
container.
-/
import Numeric.AttnLoopScore
import Numeric.AttnLoopOutput
import Numeric.AttnSoftmaxPerturb

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Weight bounds -/

/-- **Weight magnitude and total bound.** For `W_j = divRoundNearest (E_j·2^32) T` with all `E_j`
nonnegative and `T = Σ E_j > 0`: `0 ≤ W_j ≤ 2^32`, and the total weight `Σ_{j<n} W_j` is at most
`2^32 + n` (the `n/2` rounding slack, with `n` the integer margin). The `2^32` summand is the
bound of the sum being a probability total; the `n` summand is rounding up. -/
theorem attn_weight_bounds (n : ℕ) (E W : ℕ → ℤ) (T : ℤ)
    (hE : ∀ j ∈ Finset.range n, 0 ≤ E j)
    (hT : T = ∑ j ∈ Finset.range n, E j)
    (hTpos : 0 < T)
    (hW : ∀ j ∈ Finset.range n, W j = divRoundNearest (E j * (2 : ℤ) ^ (32 : ℕ)) T) :
    (∀ j ∈ Finset.range n, 0 ≤ W j ∧ W j ≤ (2 : ℤ) ^ (32 : ℕ)) ∧
    (∑ j ∈ Finset.range n, W j ≤ (2 : ℤ) ^ (32 : ℕ) + (n : ℤ)) := by
  have h32pos : 0 < (2 : ℤ) ^ (32 : ℕ) := by norm_num
  have hhalflt : T / 2 < T := by
    have h2 : 2 * (T / 2) ≤ T := by
      have := Int.mul_ediv_add_emod T 2
      have hmod : 0 ≤ T % 2 := Int.emod_nonneg _ (by norm_num)
      linarith
    by_contra hc
    push Not at hc
    nlinarith
  have hEle : ∀ j ∈ Finset.range n, E j ≤ T := by
    intro j hj
    rw [hT]
    exact Finset.single_le_sum (fun i hi => hE i hi) hj
  constructor
  · intro j hj
    have hjE : 0 ≤ E j := hE j hj
    have hjT : E j ≤ T := hEle j hj
    rw [hW j hj]
    constructor
    · rw [divRoundNearest]
      apply Int.ediv_nonneg _ hTpos.le
      have h1 : 0 ≤ E j * (2 : ℤ) ^ (32 : ℕ) := mul_nonneg hjE h32pos.le
      have h2 : 0 ≤ T / 2 := Int.ediv_nonneg hTpos.le (by norm_num)
      linarith
    · rw [divRoundNearest]
      by_contra hc
      push Not at hc
      have hq : ((E j * (2 : ℤ) ^ (32 : ℕ) + T / 2) / T) * T
          ≤ E j * (2 : ℤ) ^ (32 : ℕ) + T / 2 := by
        have := Int.mul_ediv_add_emod (E j * (2 : ℤ) ^ (32 : ℕ) + T / 2) T
        have hmod : 0 ≤ (E j * (2 : ℤ) ^ (32 : ℕ) + T / 2) % T :=
          Int.emod_nonneg _ (ne_of_gt hTpos)
        linarith
      have h2 : ((2 : ℤ) ^ (32 : ℕ) + 1) * T
          ≤ ((E j * (2 : ℤ) ^ (32 : ℕ) + T / 2) / T) * T :=
        mul_le_mul_of_nonneg_right (by omega) hTpos.le
      have h5 : E j * (2 : ℤ) ^ (32 : ℕ) ≤ T * (2 : ℤ) ^ (32 : ℕ) :=
        mul_le_mul_of_nonneg_right hjT h32pos.le
      have : (2 : ℤ) ^ (32 : ℕ) * T + T ≤ (2 : ℤ) ^ (32 : ℕ) * T + T / 2 := by
        calc (2 : ℤ) ^ (32 : ℕ) * T + T = ((2 : ℤ) ^ (32 : ℕ) + 1) * T := by ring
          _ ≤ ((E j * (2 : ℤ) ^ (32 : ℕ) + T / 2) / T) * T := h2
          _ ≤ E j * (2 : ℤ) ^ (32 : ℕ) + T / 2 := hq
          _ ≤ T * (2 : ℤ) ^ (32 : ℕ) + T / 2 := by linarith
          _ = (2 : ℤ) ^ (32 : ℕ) * T + T / 2 := by ring
      linarith
  · have hterm : ∀ j ∈ Finset.range n, W j * T ≤ E j * (2 : ℤ) ^ (32 : ℕ) + T / 2 := by
      intro j hj
      have hb := (abs_le.mp (div_round_nearest_abs_bound (E j * (2 : ℤ) ^ (32 : ℕ)) T hTpos)).2
      rw [hW j hj, divRoundNearest]
      linarith
    have hgoal : (∑ j ∈ Finset.range n, W j * T) ≤ ((2 : ℤ) ^ (32 : ℕ) + (n : ℤ)) * T := by
      calc ∑ j ∈ Finset.range n, W j * T
          ≤ ∑ j ∈ Finset.range n, (E j * (2 : ℤ) ^ (32 : ℕ) + T / 2) :=
            Finset.sum_le_sum hterm
        _ = (∑ j ∈ Finset.range n, E j) * (2 : ℤ) ^ (32 : ℕ) + (n : ℤ) * (T / 2) := by
              rw [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_const, Finset.card_range,
                nsmul_eq_mul]
        _ = T * (2 : ℤ) ^ (32 : ℕ) + (n : ℤ) * (T / 2) := by rw [← hT]
        _ ≤ T * (2 : ℤ) ^ (32 : ℕ) + (n : ℤ) * T := by
              have hnn : (0 : ℤ) ≤ (n : ℤ) := Nat.cast_nonneg n
              have := mul_le_mul_of_nonneg_left hhalflt.le hnn
              linarith
        _ = ((2 : ℤ) ^ (32 : ℕ) + (n : ℤ)) * T := by ring
    rw [← Finset.sum_mul] at hgoal
    exact Int.le_of_mul_le_mul_right hgoal hTpos

/-! ## The weight window from the numerator loop -/

/-- **Weight window.** From the numerator-loop output — `exps` of length `pos+1`, each numerator in
the `attnNumeratorCase` form at a nonpositive shift, one shift exactly `0`, and the executed total
equal to the sum of the numerators — the `AttnWeightWindow`. The total's positivity is the shift-`0`
numerator `E/2^32 ≥ 1 - epsExp - 4/2^32 > 0`; the headroom and `i64` fit come from
`attnNumeratorCase_le` (`E_j ≤ 4·2^32`) and `attn_weight_bounds`. No `attn_domain_v2` conjunct is
added. -/
theorem attn_weight_window_of_numerators (exps : alloc.vec.Vec I64) (total : I128)
    (pos : Usize) (sh : ℕ → ℤ)
    (hlen : exps.val.length = pos.val + 1)
    (hpos : pos.val + 1 ≤ 2 ^ (30 : ℕ))
    (hcase : ∀ j, j ≤ pos.val →
      sh j ≤ 0 ∧ attnNumeratorCase (sh j) (exps.val.getD j 0#i64).val)
    (hzero : ∃ j, j ≤ pos.val ∧ sh j = 0)
    (htot : total.val = ∑ j ∈ Finset.range (pos.val + 1), (exps.val.getD j 0#i64).val) :
    AttnWeightWindow exps total pos := by
  obtain ⟨j0, hj0le, hj0sh⟩ := hzero
  have hfloor : 0 < 1 - epsExp - 4 / q32 := attn_total_floor_pos
  have hmax0 : 1 - epsExp - 4 / q32 ≤ ((exps.val.getD j0 0#i64).val : ℝ) / q32 := by
    have hc0 : attnNumeratorCase 0 (exps.val.getD j0 0#i64).val := by
      have h := (hcase j0 hj0le).2
      rw [hj0sh] at h
      exact h
    exact attn_numerator_max _ hc0
  have hE0posR : 0 < ((exps.val.getD j0 0#i64).val : ℝ) := by
    have h2 : 0 < ((exps.val.getD j0 0#i64).val : ℝ) / q32 := lt_of_lt_of_le hfloor hmax0
    have := mul_pos h2 q32_pos
    rwa [div_mul_cancel₀ _ q32_ne_zero] at this
  have hE0pos : 0 < (exps.val.getD j0 0#i64).val := by exact_mod_cast hE0posR
  have hmem0 : j0 ∈ Finset.range (pos.val + 1) := Finset.mem_range.mpr (by omega)
  have hEnonneg : ∀ j ∈ Finset.range (pos.val + 1), 0 ≤ (exps.val.getD j 0#i64).val := by
    intro j hj
    have hjle : j ≤ pos.val := by simp only [Finset.mem_range] at hj; omega
    exact (attnNumeratorCase_le _ _ (hcase j hjle).1 (hcase j hjle).2).1
  have htotpos : 0 < total.val := by
    have hle : (exps.val.getD j0 0#i64).val ≤ total.val := by
      rw [htot]
      exact Finset.single_le_sum hEnonneg hmem0
    linarith
  have hcap : pos.val + 1 ≤ Usize.max := le_trans hpos usize_max_ge_2_pow_30
  have hbounds := attn_weight_bounds (pos.val + 1)
    (fun k => (exps.val.getD k 0#i64).val)
    (fun k => attnWeightAt exps.val total k) total.val hEnonneg htot htotpos (by
      intro k hk
      rfl)
  have htotle : total.val ≤ (2 : ℤ) ^ (64 : ℕ) := by
    rw [htot]
    calc ∑ k ∈ Finset.range (pos.val + 1), (exps.val.getD k 0#i64).val
        ≤ ∑ _k ∈ Finset.range (pos.val + 1), 4 * (2 : ℤ) ^ (32 : ℕ) := by
          apply Finset.sum_le_sum
          intro k hk
          have hkle : k ≤ pos.val := by simp only [Finset.mem_range] at hk; omega
          exact (attnNumeratorCase_le _ _ (hcase k hkle).1 (hcase k hkle).2).2
      _ = ((pos.val + 1 : ℕ) : ℤ) * (4 * (2 : ℤ) ^ (32 : ℕ)) := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      _ ≤ ((2 : ℕ) ^ 30 : ℤ) * (4 * (2 : ℤ) ^ (32 : ℕ)) := by
          have hp : (((pos.val + 1 : ℕ)) : ℤ) ≤ ((2 : ℕ) ^ 30 : ℤ) := by exact_mod_cast hpos
          exact mul_le_mul_of_nonneg_right hp (by norm_num)
      _ = (2 : ℤ) ^ (64 : ℕ) := by norm_num
  refine ⟨hlen, hcap, htotpos, ?_⟩
  intro j hj
  have hmem : j ∈ Finset.range (pos.val + 1) := Finset.mem_range.mpr (by omega)
  obtain ⟨hEnn, hEle4⟩ := attnNumeratorCase_le (sh j) _ (hcase j hj).1 (hcase j hj).2
  obtain ⟨hW0, hW32⟩ := hbounds.1 j hmem
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  refine ⟨hEnn, ?_, ?_, ?_⟩
  · rw [hI128max]
    have hEj : (exps.val.getD j 0#i64).val * (2 : ℤ) ^ (32 : ℕ) ≤ (2 : ℤ) ^ (66 : ℕ) := by
      calc (exps.val.getD j 0#i64).val * (2 : ℤ) ^ (32 : ℕ)
          ≤ (4 * (2 : ℤ) ^ (32 : ℕ)) * (2 : ℤ) ^ (32 : ℕ) :=
            mul_le_mul_of_nonneg_right hEle4 (by positivity)
        _ = (2 : ℤ) ^ (66 : ℕ) := by norm_num
    have hhalf : total.val / 2 ≤ (2 : ℤ) ^ (63 : ℕ) := by
      have h2 : total.val ≤ 2 * (2 : ℤ) ^ (63 : ℕ) := by
        have : (2 : ℤ) ^ (64 : ℕ) = 2 * (2 : ℤ) ^ (63 : ℕ) := by norm_num
        linarith [htotle]
      omega
    linarith
  · have hbig : -(2 : ℤ) ^ (63 : ℕ) ≤ 0 := by norm_num
    linarith
  · have hbig : (2 : ℤ) ^ (32 : ℕ) ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by norm_num
    linarith

/-! ## The checked plane address -/

/-- `checked_mul` succeeds with the exact product inside `usize`. -/
private theorem usize_checked_mul_some (x y : Usize) (h : x.val * y.val ≤ Usize.max) :
    ∃ z : Usize, Usize.checked_mul x y = some z ∧ z.val = x.val * y.val := by
  have hs := Usize.checked_mul_bv_spec x y
  cases hc : Usize.checked_mul x y with
  | none => rw [hc] at hs; simp only at hs; omega
  | some z => rw [hc] at hs; exact ⟨z, rfl, hs.2.1⟩

/-- `checked_add` succeeds with the exact sum inside `usize`. -/
private theorem usize_checked_add_some (x y : Usize) (h : x.val + y.val ≤ Usize.max) :
    ∃ z : Usize, Usize.checked_add x y = some z ∧ z.val = x.val + y.val := by
  have hs := Usize.checked_add_bv_spec x y
  cases hc : Usize.checked_add x y with
  | none => rw [hc] at hs; simp only at hs; omega
  | some z => rw [hc] at hs; exact ⟨z, rfl, hs.2.1⟩

/-- **Plane address.** With the four checked `usize` operations of `((s·kv+kvh)·hd+i)` in range,
`attn.plane_at kv kvh hd s i` returns `Ok idx` with `idx.val` exactly `(s·kv+kvh)·hd+i`
(`src/engine/attn.rs:86-91`). -/
theorem plane_at_spec (kv kvh hd s i : Usize)
    (hkv : s.val * kv.val ≤ Usize.max)
    (hadd : s.val * kv.val + kvh.val ≤ Usize.max)
    (hmul : (s.val * kv.val + kvh.val) * hd.val ≤ Usize.max)
    (hidx : (s.val * kv.val + kvh.val) * hd.val + i.val ≤ Usize.max) :
    ∃ idx : Usize, attn.plane_at kv kvh hd s i = ok (core.result.Result.Ok idx) ∧
      idx.val = (s.val * kv.val + kvh.val) * hd.val + i.val := by
  obtain ⟨base, hbase, hbasev⟩ := usize_checked_mul_some s kv hkv
  obtain ⟨val1, hval1, hval1v⟩ := usize_checked_add_some base kvh (by rw [hbasev]; exact hadd)
  obtain ⟨val2, hval2, hval2v⟩ := usize_checked_mul_some val1 hd (by rw [hval1v, hbasev]; exact hmul)
  obtain ⟨idx, hval3, hval3v⟩ := usize_checked_add_some val2 i (by rw [hval2v, hval1v, hbasev]; exact hidx)
  refine ⟨idx, ?_, ?_⟩
  · simp only [attn.plane_at, hbase, hval1, hval2, hval3, lift,
      core.option.Option.ok_or_some, bind_tc_ok, result_branch_ok]
  · rw [hval3v, hval2v, hval1v, hbasev]

/-! ## The value fit -/

/-- **Value fit.** With `0 ≤ W_s ≤ 2^32`, `Σ W_s ≤ 2^32 + n`, `|V_s| ≤ 2^62` and `n = pos+1 ≤ 2^30`,
the `value_component_spec` narrowing conditions hold for `acc = Σ_{s<n} W_s·V_s`:
`I128.min ≤ acc + 2^31 ≤ I128.max` and `-2^63 ≤ divRoundNearest acc 2^32 ≤ 2^63 − 1`
(since `|acc|/2^32 ≤ 2^62 + 2^60`). -/
theorem attn_value_fit (n : ℕ) (W V : ℕ → ℤ) (pos : Usize)
    (hn : n = pos.val + 1) (hpos : pos.val + 1 ≤ 2 ^ (30 : ℕ))
    (hW0 : ∀ s ∈ Finset.range n, 0 ≤ W s ∧ W s ≤ (2 : ℤ) ^ (32 : ℕ))
    (hWsum : ∑ s ∈ Finset.range n, W s ≤ (2 : ℤ) ^ (32 : ℕ) + (n : ℤ))
    (hV : ∀ s ∈ Finset.range n, |V s| ≤ (2 : ℤ) ^ (62 : ℕ)) :
    I128.min ≤ (∑ s ∈ Finset.range n, W s * V s) + (2 : ℤ) ^ (31 : ℕ) ∧
    (∑ s ∈ Finset.range n, W s * V s) + (2 : ℤ) ^ (31 : ℕ) ≤ I128.max ∧
    -((2 : ℤ) ^ (63 : ℕ)) ≤
      divRoundNearest (∑ s ∈ Finset.range n, W s * V s) ((2 : ℤ) ^ (32 : ℕ)) ∧
    divRoundNearest (∑ s ∈ Finset.range n, W s * V s) ((2 : ℤ) ^ (32 : ℕ))
      ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  have hDpos : (0 : ℤ) < (2 : ℤ) ^ (32 : ℕ) := by norm_num
  set acc : ℤ := ∑ s ∈ Finset.range n, W s * V s with hacc
  set A : ℤ := acc + (2 : ℤ) ^ (31 : ℕ) with hA
  have hn30 : n ≤ 2 ^ 30 := by omega
  have habs : |acc| ≤ (2 : ℤ) ^ (62 : ℕ) * ((2 : ℤ) ^ (32 : ℕ) + (n : ℤ)) := by
    rw [hacc]
    calc |∑ s ∈ Finset.range n, W s * V s|
        ≤ ∑ s ∈ Finset.range n, |W s * V s| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ s ∈ Finset.range n, W s * |V s| := by
            apply Finset.sum_congr rfl
            intro s hs
            rw [abs_mul, abs_of_nonneg (hW0 s hs).1]
      _ ≤ ∑ s ∈ Finset.range n, W s * (2 : ℤ) ^ (62 : ℕ) := by
            apply Finset.sum_le_sum
            intro s hs
            exact mul_le_mul_of_nonneg_left (hV s hs) (hW0 s hs).1
      _ = (∑ s ∈ Finset.range n, W s) * (2 : ℤ) ^ (62 : ℕ) := by rw [Finset.sum_mul]
      _ ≤ ((2 : ℤ) ^ (32 : ℕ) + (n : ℤ)) * (2 : ℤ) ^ (62 : ℕ) :=
            mul_le_mul_of_nonneg_right hWsum (by positivity)
      _ = (2 : ℤ) ^ (62 : ℕ) * ((2 : ℤ) ^ (32 : ℕ) + (n : ℤ)) := by ring
  have hB : (2 : ℤ) ^ (62 : ℕ) * ((2 : ℤ) ^ (32 : ℕ) + (n : ℤ))
      ≤ (2 : ℤ) ^ (94 : ℕ) + (2 : ℤ) ^ (92 : ℕ) := by
    have hnZ : (n : ℤ) ≤ (2 : ℤ) ^ (30 : ℕ) := by exact_mod_cast hn30
    have h1 : (2 : ℤ) ^ (32 : ℕ) + (n : ℤ) ≤ (2 : ℤ) ^ (32 : ℕ) + (2 : ℤ) ^ (30 : ℕ) := by
      linarith
    calc (2 : ℤ) ^ (62 : ℕ) * ((2 : ℤ) ^ (32 : ℕ) + (n : ℤ))
        ≤ (2 : ℤ) ^ (62 : ℕ) * ((2 : ℤ) ^ (32 : ℕ) + (2 : ℤ) ^ (30 : ℕ)) :=
          mul_le_mul_of_nonneg_left h1 (by positivity)
      _ = (2 : ℤ) ^ (94 : ℕ) + (2 : ℤ) ^ (92 : ℕ) := by norm_num
  have hAabs : |A| < (2 : ℤ) ^ (95 : ℕ) := by
    have h1 : |A| ≤ |acc| + (2 : ℤ) ^ (31 : ℕ) := by
      rw [hA]
      calc |acc + (2 : ℤ) ^ (31 : ℕ)| ≤ |acc| + |(2 : ℤ) ^ (31 : ℕ)| := abs_add_le _ _
        _ = |acc| + (2 : ℤ) ^ (31 : ℕ) := by
              have h31 : |(2 : ℤ) ^ (31 : ℕ)| = (2 : ℤ) ^ (31 : ℕ) :=
                abs_of_nonneg (by positivity)
              rw [h31]
    have h2 : |acc| ≤ (2 : ℤ) ^ (94 : ℕ) + (2 : ℤ) ^ (92 : ℕ) := le_trans habs hB
    have h3 : (2 : ℤ) ^ (94 : ℕ) + (2 : ℤ) ^ (92 : ℕ) + (2 : ℤ) ^ (31 : ℕ)
        < (2 : ℤ) ^ (95 : ℕ) := by norm_num
    linarith
  have hmod := Int.mul_ediv_add_emod A ((2 : ℤ) ^ (32 : ℕ))
  have hqmul : (A / (2 : ℤ) ^ (32 : ℕ)) * (2 : ℤ) ^ (32 : ℕ) ≤ A := by
    have hmodnn : 0 ≤ A % (2 : ℤ) ^ (32 : ℕ) := Int.emod_nonneg _ (ne_of_gt hDpos)
    linarith
  have hqsucc : A < (A / (2 : ℤ) ^ (32 : ℕ) + 1) * (2 : ℤ) ^ (32 : ℕ) := by
    have hmodlt : A % (2 : ℤ) ^ (32 : ℕ) < (2 : ℤ) ^ (32 : ℕ) := Int.emod_lt_of_pos _ hDpos
    linarith
  have hfitlo : -((2 : ℤ) ^ (63 : ℕ)) ≤ A / (2 : ℤ) ^ (32 : ℕ) := by
    by_contra hc
    push Not at hc
    have h1 : (A / (2 : ℤ) ^ (32 : ℕ) + 1) * (2 : ℤ) ^ (32 : ℕ)
        ≤ (-((2 : ℤ) ^ (63 : ℕ))) * (2 : ℤ) ^ (32 : ℕ) :=
      mul_le_mul_of_nonneg_right (by omega) (by positivity)
    have h2 : (-((2 : ℤ) ^ (63 : ℕ))) * (2 : ℤ) ^ (32 : ℕ) = -((2 : ℤ) ^ (95 : ℕ)) := by
      norm_num
    have h3 : A < -((2 : ℤ) ^ (95 : ℕ)) := by linarith
    have h4 : -((2 : ℤ) ^ (95 : ℕ)) < A := (abs_lt.mp hAabs).1
    linarith
  have hfithi : A / (2 : ℤ) ^ (32 : ℕ) ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    by_contra hc
    push Not at hc
    have h1 : (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (32 : ℕ)
        ≤ (A / (2 : ℤ) ^ (32 : ℕ)) * (2 : ℤ) ^ (32 : ℕ) :=
      mul_le_mul_of_nonneg_right (by omega) (by positivity)
    have h2 : (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (32 : ℕ) = (2 : ℤ) ^ (95 : ℕ) := by norm_num
    have h3 : (2 : ℤ) ^ (95 : ℕ) ≤ A := by linarith
    have h4 : A < (2 : ℤ) ^ (95 : ℕ) := (abs_lt.mp hAabs).2
    linarith
  have hdiv : divRoundNearest acc ((2 : ℤ) ^ (32 : ℕ)) = A / (2 : ℤ) ^ (32 : ℕ) := by
    rw [divRoundNearest, hA]
    norm_num
  have hlo : -((2 : ℤ) ^ (127 : ℕ)) ≤ A := by
    have h4 : -((2 : ℤ) ^ (95 : ℕ)) < A := (abs_lt.mp hAabs).1
    have : -((2 : ℤ) ^ (127 : ℕ)) < -((2 : ℤ) ^ (95 : ℕ)) := by norm_num
    linarith
  have hhi : A ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
    have h4 : A < (2 : ℤ) ^ (95 : ℕ) := (abs_lt.mp hAabs).2
    have : (2 : ℤ) ^ (95 : ℕ) < (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num
    linarith
  rw [hI128min, hI128max, hdiv]
  exact ⟨hlo, hhi, hfitlo, hfithi⟩

/-! ## The value window from `attn_domain_v2` -/

/-- **Value window.** `attn_domain_v2` gives the plane lengths, the value magnitude window and
`pos+1 ≤ 2^30`; `kvh = q_head/(n_head/n_head_kv)` makes `kvh < n_head_kv`; the weight window
(`length = pos+1`, `|W_s| ≤ 2^33`) is supplied. Then `AttnValueWindow` holds, its checked
range-index identity discharged by `plane_at_spec` from `pos < cap`, `kvh < kv`,
`i < head_dim`. -/
theorem attn_value_window_of_v2 (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (kplane vplane : alloc.vec.Vec I64) (kvh : Usize) (i : Usize)
    (weights : alloc.vec.Vec I64)
    (hdom : attn_domain_v2 state layer q_head pos q out kplane vplane)
    (hkvh : kvh.val = q_head.val / (state.n_head.val / state.n_head_kv.val))
    (hi : i.val < state.head_dim.val)
    (hwlen : weights.val.length = pos.val + 1)
    (hwbound : ∀ s, s ≤ pos.val → |(weights.val.getD s 0#i64).val| ≤ (2 : ℤ) ^ (33 : ℕ)) :
    AttnValueWindow (alloc.vec.Vec.deref vplane) (alloc.vec.Vec.deref weights)
      state.n_head_kv kvh state.head_dim pos i := by
  obtain ⟨hdom0, _hocc, _hscale, _hk, _hv, _hklen, hvlen, _hrowdot, _hscore, hvalu, hpos30⟩ := hdom
  obtain ⟨_hnlayer, _hnhead, hnkhead, hdhead, _hcap, hmod, _hl, hqhead, _hqlen, _houtlen,
    hposcap⟩ := hdom0
  have hdvd : state.n_head_kv.val ∣ state.n_head.val := Nat.dvd_of_mod_eq_zero hmod
  have hmul : state.n_head.val = (state.n_head.val / state.n_head_kv.val) * state.n_head_kv.val :=
    (Nat.div_mul_cancel hdvd).symm
  have hdpos : 0 < state.n_head.val / state.n_head_kv.val := by
    rcases Nat.eq_zero_or_pos (state.n_head.val / state.n_head_kv.val) with h | h
    · exfalso
      rw [h, Nat.zero_mul] at hmul
      omega
    · exact h
  have hkvhlt : kvh.val < state.n_head_kv.val := by
    rw [hkvh]
    exact Nat.div_lt_of_lt_mul (by rw [← hmul]; exact hqhead)
  have hkvpos : 0 < state.n_head_kv.val := hnkhead
  have hhdpos : 0 < state.head_dim.val := hdhead
  have hidxlt : ∀ s : Usize, s.val ≤ pos.val →
      (s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val + i.val
        < vplane.val.length := by
    intro s hs
    have hscap : s.val < state.cap.val := lt_of_le_of_lt hs hposcap
    have hA : s.val * state.n_head_kv.val + kvh.val
        < state.cap.val * state.n_head_kv.val := by
      have h1 : s.val * state.n_head_kv.val + kvh.val + 1
          ≤ state.cap.val * state.n_head_kv.val := by
        calc s.val * state.n_head_kv.val + kvh.val + 1
            ≤ s.val * state.n_head_kv.val + state.n_head_kv.val := by omega
          _ = (s.val + 1) * state.n_head_kv.val := by ring
          _ ≤ state.cap.val * state.n_head_kv.val :=
              Nat.mul_le_mul_right _ (by omega)
      omega
    have hAi : (s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val + i.val
        < state.cap.val * state.n_head_kv.val * state.head_dim.val := by
      have hA1 : s.val * state.n_head_kv.val + kvh.val + 1
          ≤ state.cap.val * state.n_head_kv.val := by omega
      calc (s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val + i.val
          < (s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val
              + state.head_dim.val := by omega
        _ = (s.val * state.n_head_kv.val + kvh.val + 1) * state.head_dim.val := by ring
        _ ≤ state.cap.val * state.n_head_kv.val * state.head_dim.val :=
            Nat.mul_le_mul_right _ hA1
    rw [hvlen]
    exact lt_of_lt_of_eq hAi (by ring)
  refine ⟨?_, hpos30, ?_, ?_, ?_⟩
  · show (alloc.vec.Vec.deref weights).val.length = pos.val + 1
    rw [alloc.vec.Vec.deref, Slice.from_val]
    exact hwlen
  · intro s hs
    have hmax : (s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val + i.val
        ≤ Usize.max := le_trans (le_of_lt (hidxlt s hs)) vplane.property
    have hle1 : s.val * state.n_head_kv.val + kvh.val
        ≤ (s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val := by
      simpa using Nat.mul_le_mul_left (s.val * state.n_head_kv.val + kvh.val) hhdpos
    have hA : s.val * state.n_head_kv.val + kvh.val ≤ Usize.max :=
      le_trans hle1
        (le_trans (Nat.le_add_right
          ((s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val) i.val) hmax)
    have hB : (s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val ≤ Usize.max :=
      le_trans (Nat.le_add_right
        ((s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val) i.val) hmax
    have hC : s.val * state.n_head_kv.val ≤ Usize.max :=
      le_trans (Nat.le_add_right (s.val * state.n_head_kv.val) kvh.val) hA
    obtain ⟨idx, hplane, hidxeq⟩ := plane_at_spec state.n_head_kv kvh state.head_dim s i hC hA hB hmax
    refine ⟨idx, hplane, hidxeq, ?_⟩
    have : (alloc.vec.Vec.deref vplane).val = vplane.val := by
      rw [alloc.vec.Vec.deref, Slice.from_val]
    rw [this, hidxeq]
    exact hidxlt s hs
  · intro s hs
    have h1 := hvalu s hs i.val hi
    rw [← hkvh] at h1
    rw [planeRow_getD vplane.val state.n_head_kv.val state.head_dim.val kvh.val s i.val hi] at h1
    have hval_eq : (alloc.vec.Vec.deref vplane).val = vplane.val := by
      rw [alloc.vec.Vec.deref, Slice.from_val]
    rw [hval_eq]
    exact h1
  · intro s hs
    have h1 := hwbound s hs
    have hval_eq : (alloc.vec.Vec.deref weights).val = weights.val := by
      rw [alloc.vec.Vec.deref, Slice.from_val]
    rw [hval_eq]
    exact h1

/-! ## Per-component output -/

/-- **Per-component output.** Combining `attn_value_window_of_v2`, `attn_value_fit` and
`value_component_spec`: on `attn_domain_v2`, weights of length `pos+1` whose values are `W` with
`0 ≤ W_s ≤ 2^32` and `Σ W ≤ 2^32 + (pos+1)`, every component `i < head_dim` is the exact half-biased
reduction `divRoundNearest (Σ_{s≤pos} W_s·V_{s,i}) 2^32` of the `valueList` entries. -/
theorem attn_value_components (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (kplane vplane : alloc.vec.Vec I64) (kvh : Usize)
    (weights : alloc.vec.Vec I64) (W : ℕ → ℤ)
    (hdom : attn_domain_v2 state layer q_head pos q out kplane vplane)
    (hkvh : kvh.val = q_head.val / (state.n_head.val / state.n_head_kv.val))
    (hwlen : weights.val.length = pos.val + 1)
    (hW : ∀ s, s ≤ pos.val → (weights.val.getD s 0#i64).val = W s)
    (hW0 : ∀ s ∈ Finset.range (pos.val + 1), 0 ≤ W s ∧ W s ≤ (2 : ℤ) ^ (32 : ℕ))
    (hWsum : ∑ s ∈ Finset.range (pos.val + 1), W s
      ≤ (2 : ℤ) ^ (32 : ℕ) + ((pos.val + 1 : ℕ) : ℤ)) :
    ∀ i : Usize, i.val < state.head_dim.val →
      ∃ y : I64,
        attn.value_component (alloc.vec.Vec.deref vplane) (alloc.vec.Vec.deref weights)
          state.n_head_kv kvh state.head_dim pos i = ok (core.result.Result.Ok y) ∧
        y.val = divRoundNearest
          (∑ s ∈ Finset.range (pos.val + 1), W s *
            ((valueList vplane.val state.n_head_kv.val state.head_dim.val kvh.val pos.val i.val).getD s 0))
          ((2 : ℤ) ^ (32 : ℕ)) := by
  have hdomc := hdom
  obtain ⟨_hdom0, _hocc, _hscale, _hk, _hv, _hklen, _hvlen, _hrowdot, _hscore, hvalu, hpos30⟩ := hdom
  have hvderef : (alloc.vec.Vec.deref vplane).val = vplane.val := by
    rw [alloc.vec.Vec.deref, Slice.from_val]
  have hwderef : (alloc.vec.Vec.deref weights).val = weights.val := by
    rw [alloc.vec.Vec.deref, Slice.from_val]
  intro i hi
  have hwbound : ∀ s, s ≤ pos.val → |(weights.val.getD s 0#i64).val| ≤ (2 : ℤ) ^ (33 : ℕ) := by
    intro s hs
    have h1 := (hW0 s (Finset.mem_range.mpr (by omega))).1
    have h2 := (hW0 s (Finset.mem_range.mpr (by omega))).2
    have h3 := hW s hs
    rw [h3]
    have : (2 : ℤ) ^ (32 : ℕ) ≤ (2 : ℤ) ^ (33 : ℕ) := by norm_num
    rw [abs_of_nonneg h1]
    linarith
  have hwin := attn_value_window_of_v2 state layer q_head pos q out kplane vplane kvh i weights
    hdomc hkvh hi hwlen hwbound
  set V : ℕ → ℤ := fun s =>
    ((alloc.vec.Vec.deref vplane).val.getD
      ((s * state.n_head_kv.val + kvh.val) * state.head_dim.val + i.val) 0#i64).val with hVdef
  have hV : ∀ s ∈ Finset.range (pos.val + 1), |V s| ≤ (2 : ℤ) ^ (62 : ℕ) := by
    intro s hs
    have hsle : s ≤ pos.val := by simp only [Finset.mem_range] at hs; omega
    rw [hVdef]
    rw [hvderef]
    have h1 := hvalu s hsle i.val hi
    rw [← hkvh] at h1
    rwa [planeRow_getD vplane.val state.n_head_kv.val state.head_dim.val kvh.val s i.val hi] at h1
  have fit := attn_value_fit (pos.val + 1) W V pos rfl hpos30 hW0 hWsum hV
  have hsum_eq : (∑ s ∈ Finset.range (pos.val + 1),
        ((alloc.vec.Vec.deref weights).val.getD s 0#i64).val *
          ((alloc.vec.Vec.deref vplane).val.getD
            ((s * state.n_head_kv.val + kvh.val) * state.head_dim.val + i.val) 0#i64).val)
      = ∑ s ∈ Finset.range (pos.val + 1), W s * V s := by
    apply Finset.sum_congr rfl
    intro s hs
    have hsle : s ≤ pos.val := by simp only [Finset.mem_range] at hs; omega
    rw [hVdef]
    rw [hwderef, hvderef, hW s hsle]
  obtain ⟨y, hy_eq, hy_val⟩ := value_component_spec (alloc.vec.Vec.deref vplane)
    (alloc.vec.Vec.deref weights) state.n_head_kv kvh state.head_dim pos i hwin
    (by rw [hsum_eq]; exact fit.1)
    (by rw [hsum_eq]; exact fit.2.1)
    (by rw [hsum_eq]; exact ⟨fit.2.2.1, fit.2.2.2⟩)
    hi
  refine ⟨y, hy_eq, ?_⟩
  rw [hy_val, hvderef]
  congr 1
  apply Finset.sum_congr rfl
  intro s hs
  have hsle : s ≤ pos.val := by simp only [Finset.mem_range] at hs; omega
  rw [hwderef, hW s hsle]

end

end apeiron_word
