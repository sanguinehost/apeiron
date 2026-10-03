/-
Attention weight and value-side windows.

Discharges the weight-loop window (`AttnWeightWindow`), the value-loop window (`AttnValueWindow`)
and the per-component output composition from the numerator-loop output and `attn_domain_v2`,
with no new `attn_domain_v2` conjunct.

* `attn_weight_bounds`: the pure `ℤ` weight bound `0 ≤ W_j ≤ 2^32`, `Σ_{j<n} W_j ≤ 2^32 + n`
  for `W_j = divRoundNearest (E_j·2^32) T`, `T = Σ E_j`;
* `attn_weight_window_of_numerators`: `AttnWeightWindow` from the numerator-loop output;
* `plane_at_spec`: the checked address identity of the folded value pass — for `s ≤ pos` the
  `head_dim`-wide block at `first + s·stride` lies inside the plane and its `i`-th entry is the
  plane's `(s·kv + kvh)·hd + i` entry. This is the predecessor's `attn.plane_at` address
  identity, restated over the extracted `alloc.vec.Vec.index` range read;
* `attn_value_fit`: the output epilogue's narrowing (`hlo`/`hhi`/`hfit`) from the weight and
  value windows;
* `attn_value_window_of_v2`: `AttnValueWindow` from `attn_domain_v2`, `plane_at_spec` and the
  `usize` headroom of the extracted block arithmetic;
* `attn_value_components`: the per-component result — the value pass followed by the output pass,
  every component `i < head_dim` the exact half-biased reduction
  `divRoundNearest (Σ_{s≤pos} W_s·V_{s,i}) 2^32` of the `valueList` entries.

## The reshaped extraction this module reads

The staged artifact deletes `attn.plane_at`, `attn.value_component`, `attn.value_component_loop`
and `attn.value_term`, and folds the weight division into the value pass: `attn_head_loop2`
resolves `W_s = divRoundNearest (E_s·2^32) T` itself, reads each key's `head_dim`-wide row once at
the hoisted `first + s·stride`, and folds it into a zeroed `i128` accumulator; `attn_head_loop3`
applies the single `round_shift_32` epilogue. The three theorems that named the deleted helpers
are therefore restated over the loops that exist, keeping their content — the per-index row
address, the per-component weighted sum and its one half-biased reduction:

* `plane_at_spec` is the row-read identity, the same claim the predecessor drew from
  `attn.plane_at`'s four checked `usize` operations;
* `attn_value_components` is the two executed loops, through `Numeric.AttnLoopOutput`'s
  `attn_head_loop2_spec`, `attn_head_loop2_component` and `value_component_spec` (reused, not
  restated), with the narrowing supplied by `attn_value_fit` exactly as the predecessor supplied
  it to its own `value_component_spec`.

`stride = n_head_kv·head_dim` and `first = kvh·head_dim` are the Rust `kv.checked_mul(hd)` and
`kvh.checked_mul(hd)` of `src/engine/attn.rs:1079-1080`; both hypotheses are named `hstride`,
`hfirst`, as `Numeric.AttnLoopOutput`'s `value_component_spec` names them.

`AttnValueWindow`'s first conjunct is `stride·pos + first + head_dim ≤ Usize.max`, the largest
value the executed block arithmetic takes: `src/engine/attn.rs:1183-1187` reads
`base = first + s·stride` and `end = base + head_dim` under the loop guard `s ≤ pos`
(`src/engine/attn.rs:1176`), so `s = pos` is the last key addressed and no key one step past it is.
The wider form `stride·(pos+1) + first + head_dim ≤ Usize.max` that the conjunct previously carried
is not a consequence of `attn_domain_v2`: on `n_head_kv = 2^63 - 1`, `head_dim = 1`, `cap = 2`,
`pos = 1`, `kvh = 2^63 - 2` (all of `attn_domain` and `attn_domain_v2` hold — the plane length is
`2^64 - 2 ≤ Usize.max` and `pos+1 = 2 ≤ 2^30`) the left side is `2^64 + 2^63 - 3 > Usize.max`, while
the executed arithmetic on that state refuses earlier, at `base > vplane.len().saturating_sub(hd)`.
The narrow form is derivable from `plane_at_spec`: at `s = pos` it gives
`first + pos·stride + head_dim = end_.val ≤ vplane.val.length ≤ Usize.max`, and
`first + pos·stride + head_dim ≤ n_head_kv·cap·head_dim = vplane.val.length` is the per-key bound
`src/engine/attn.rs:1186` checks. `attn_value_window_of_v2` and `attn_value_components` therefore
carry the conjunct as the premise `harith`, in this narrow form; no `attn_domain_v2` conjunct is
added.

Trust boundary: reaches the extracted `attn.KvState.attn_head_loop2`,
`attn.KvState.attn_head_loop2_loop0`, `attn.KvState.attn_head_loop3` and `attn.weight_at` of
`proof/ApeironWord.lean` at sha256
`ac563ff7c7398d05aeec3f8fe1b0c66bd4ec365cce621f4d34749949c8e1e1c9`. Does not cover
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
equal to the sum of the numerators — the `AttnWeightWindow`: the loop's room for the numerator
vector, the `weight_at_spec` premises at every index and the magnitude `0 ≤ W_j ≤ 2^32` of each
rounded weight. The total's positivity is the shift-`0` numerator `E/2^32 ≥ 1 - epsExp - 4/2^32 > 0`;
the headroom and `i64` fit come from `attnNumeratorCase_le` (`E_j ≤ 4·2^32`) and
`attn_weight_bounds`. No `attn_domain_v2` conjunct is added. -/
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
  refine ⟨hlen, hcap, htotpos, ?_, ?_⟩
  · intro j hj
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
  · intro j hj
    have hmem : j ∈ Finset.range (pos.val + 1) := Finset.mem_range.mpr (by omega)
    exact hbounds.1 j hmem

/-! ## The checked row address -/

/-- The value of the extracted `saturating_sub` on `usize`: the clamped difference. -/
private theorem usize_saturating_sub_val (x y : Usize) :
    (core.num.Usize.saturating_sub x y).val = max 0 (x.val - y.val) := by
  rw [core.num.Usize.saturating_sub, UScalar.saturating_sub, UScalar.val, BitVec.toNat_ofNat]
  have hle : max 0 (x.val - y.val) ≤ x.val :=
    max_le (Nat.zero_le _) (Nat.sub_le x.val y.val)
  rw [Nat.mod_eq_of_lt (lt_of_le_of_lt hle x.hBounds)]

/-- `getD` at an in-range index of a `List.slice` is the same entry of the original list. -/
private theorem getD_slice_eq {α : Type} (l : List α) (s e i : ℕ) (hi : i < e - s) (d : α) :
    (l.slice s e).getD i d = l.getD (s + i) d := by
  rw [List.getD_eq_getElem?_getD (l := l.slice s e), List.getD_eq_getElem?_getD (l := l)]
  rw [List.slice, List.getElem?_take, if_pos hi, List.getElem?_drop]

/-- `x ≤ Usize.max` is `x < 2^numBits`, the bound `UScalar.ofNatCore` needs. -/
private theorem usize_lt_pow_numBits (x : ℕ) (h : x ≤ Usize.max) :
    x < 2 ^ UScalarTy.Usize.numBits := by
  have hmax : (Usize.max : ℕ) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  rw [UScalarTy.Usize_numBits_eq]
  rw [hmax] at h
  have hp : (0 : ℕ) < 2 ^ System.Platform.numBits := pow_pos (by norm_num) _
  omega

/-- **Row-read address identity.** With `stride = n_head_kv·head_dim`, `first = kvh·head_dim`,
`kvh < n_head_kv`, `pos < cap`, both head dimensions positive and the plane length
`n_head_kv·cap·head_dim`, the `head_dim`-wide block the folded value pass reads at
`first + s·stride` for `s ≤ pos` is in range, and its `i`-th entry is the plane's
`(s·kv + kvh)·hd + i` entry. This is the predecessor's `attn.plane_at` address identity
(`src/engine/attn.rs:86-91`), restated over the extracted `alloc.vec.Vec.index` range read at
`src/engine/attn.rs:1190-1191`. -/
theorem plane_at_spec (state : attn.KvState) (pos kvh stride first : Usize)
    (vplane : alloc.vec.Vec I64)
    (hvlen : vplane.val.length = state.n_head_kv.val * state.cap.val * state.head_dim.val)
    (hfirst : first.val = kvh.val * state.head_dim.val)
    (hstride : stride.val = state.n_head_kv.val * state.head_dim.val)
    (hkvhlt : kvh.val < state.n_head_kv.val)
    (hwidth : 0 < state.head_dim.val)
    (hposcap : pos.val < state.cap.val) :
    ∀ s : Usize, s.val ≤ pos.val →
      ∃ (sat base end_ : Usize) (row : Slice I64),
        sat.val = max 0 (vplane.val.length - state.head_dim.val) ∧
        base.val = first.val + s.val * stride.val ∧
        end_.val = base.val + state.head_dim.val ∧
        end_.val ≤ vplane.val.length ∧
        ¬ (base > sat) ∧
        alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice Std.I64) vplane
            { start := base, «end» := end_ } = ok row ∧
        row.val.length = state.head_dim.val ∧
        (∀ i, i < state.head_dim.val →
          row.val.getD i 0#i64 =
            vplane.val.getD (s.val * stride.val + first.val + i) 0#i64) := by
  intro s hs
  -- the flat block index of key `s` is `(s·kv + kvh)·hd`, and one more key fits in the plane
  have hidx : first.val + s.val * stride.val
      = (s.val * state.n_head_kv.val + kvh.val) * state.head_dim.val := by
    rw [hfirst, hstride]
    ring
  have hA : s.val * state.n_head_kv.val + kvh.val + 1
      ≤ state.cap.val * state.n_head_kv.val := by
    calc s.val * state.n_head_kv.val + kvh.val + 1
        ≤ s.val * state.n_head_kv.val + state.n_head_kv.val := by omega
      _ = (s.val + 1) * state.n_head_kv.val := by ring
      _ ≤ state.cap.val * state.n_head_kv.val := Nat.mul_le_mul_right _ (by omega)
  have hbase_le : first.val + s.val * stride.val + state.head_dim.val
      ≤ vplane.val.length := by
    calc first.val + s.val * stride.val + state.head_dim.val
        = (s.val * state.n_head_kv.val + kvh.val + 1) * state.head_dim.val := by
            rw [hidx]
            ring
      _ ≤ state.cap.val * state.n_head_kv.val * state.head_dim.val :=
          Nat.mul_le_mul_right state.head_dim.val hA
      _ = vplane.val.length := by rw [hvlen]; ring
  -- every one of these addresses is inside `usize`
  have hbase_max : first.val + s.val * stride.val + state.head_dim.val ≤ Usize.max :=
    le_trans hbase_le vplane.property
  have hsat_lt : max 0 (vplane.val.length - state.head_dim.val)
      < 2 ^ UScalarTy.Usize.numBits :=
    usize_lt_pow_numBits _
      (le_trans (max_le (Nat.zero_le _) (Nat.sub_le _ _)) vplane.property)
  have hbase_lt : first.val + s.val * stride.val < 2 ^ UScalarTy.Usize.numBits := by
    have h1 : first.val + s.val * stride.val
        < first.val + s.val * stride.val + state.head_dim.val := by omega
    exact lt_of_lt_of_le h1 (le_of_lt (usize_lt_pow_numBits _ hbase_max))
  have hend_lt : first.val + s.val * stride.val + state.head_dim.val
      < 2 ^ UScalarTy.Usize.numBits :=
    usize_lt_pow_numBits _ hbase_max
  
  -- the three addresses, as `usize` scalars carrying the arithmetic above
  have hsatv : (UScalar.ofNatCore _ hsat_lt).val
      = max 0 (vplane.val.length - state.head_dim.val) := UScalar.ofNatCore_val_eq hsat_lt
  have hbasev : (UScalar.ofNatCore _ hbase_lt).val
      = first.val + s.val * stride.val := UScalar.ofNatCore_val_eq hbase_lt
  have hendv : (UScalar.ofNatCore _ hend_lt).val
      = first.val + s.val * stride.val + state.head_dim.val := UScalar.ofNatCore_val_eq hend_lt
  have hbe : (UScalar.ofNatCore _ hbase_lt).val ≤ (UScalar.ofNatCore _ hend_lt).val := by
    rw [hbasev, hendv]
    omega
  have hen : (UScalar.ofNatCore _ hend_lt).val ≤ vplane.val.length := by
    rw [hendv]
    exact hbase_le
  obtain ⟨row, hindex, hrowval, hrowlen⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_Range_spec vplane
      { start := UScalar.ofNatCore _ hbase_lt, «end» := UScalar.ofNatCore _ hend_lt } hbe hen)
  refine ⟨UScalar.ofNatCore _ hsat_lt, UScalar.ofNatCore _ hbase_lt,
    UScalar.ofNatCore _ hend_lt, row, ?_, ?_, ?_, ?_, ?_, hindex, ?_, ?_⟩
  · show (UScalar.ofNatCore _ hsat_lt).val
      = max 0 (vplane.val.length - state.head_dim.val)
    exact hsatv
  · show (UScalar.ofNatCore _ hbase_lt).val = first.val + s.val * stride.val
    exact hbasev
  · show (UScalar.ofNatCore _ hend_lt).val
      = (UScalar.ofNatCore _ hbase_lt).val + state.head_dim.val
    rw [hendv, hbasev]
  · show (UScalar.ofNatCore _ hend_lt).val ≤ vplane.val.length
    exact hen
  · show ¬ ((UScalar.ofNatCore _ hbase_lt) : Usize)
        > (UScalar.ofNatCore _ hsat_lt : Usize)
    intro hgt
    have hlt : (UScalar.ofNatCore _ hsat_lt).val
        < (UScalar.ofNatCore _ hbase_lt).val := (UScalar.lt_equiv _ _).1 hgt
    rw [hsatv, hbasev] at hlt
    have hmax : first.val + s.val * stride.val
        ≤ max 0 (vplane.val.length - state.head_dim.val) :=
      (Nat.le_sub_of_add_le hbase_le).trans (Nat.le_max_right _ _)
    omega
  · show Slice.length row = state.head_dim.val
    rw [show Slice.length row = (UScalar.ofNatCore _ hend_lt).val
        - (UScalar.ofNatCore _ hbase_lt).val from hrowlen, hendv, hbasev]
    omega
  · intro i hi
    rw [hrowval,
      getD_slice_eq vplane.val (UScalar.ofNatCore _ hbase_lt).val
        (UScalar.ofNatCore _ hend_lt).val i (by rw [hendv, hbasev]; omega) (0#i64)]
    have hidxn : (UScalar.ofNatCore _ hbase_lt).val + i
        = s.val * stride.val + first.val + i := by
      rw [hbasev]
      omega
    rw [hidxn]

/-! ## The value fit -/

/-- **Value fit.** With `0 ≤ W_s ≤ 2^32`, `Σ W_s ≤ 2^32 + n`, `|V_s| ≤ 2^62` and `n = pos+1 ≤ 2^30`,
the output epilogue's narrowing conditions hold for `acc = Σ_{s<n} W_s·V_s`:
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

/-- **Value window.** `attn_domain_v2` gives the plane length, the value magnitude window and
`pos+1 ≤ 2^30`; `kvh = q_head/(n_head/n_head_kv)` makes `kvh < n_head_kv`. Then `AttnValueWindow`
holds at `stride = n_head_kv·head_dim`, `first = kvh·head_dim`: the block arithmetic at the largest
key read is the premise `harith`, `stride·pos + first + head_dim ≤ Usize.max` (see the module
header), the row's range-index identity and its bounds are
`plane_at_spec` from `pos < cap` and `kvh < n_head_kv`, and the value magnitude is the
`attn_domain_v2` window read through `planeRow_getD`. -/
theorem attn_value_window_of_v2 (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (kplane vplane : alloc.vec.Vec I64) (kvh stride first : Usize)
    (hdom : attn_domain_v2 state layer q_head pos q out kplane vplane)
    (hkvh : kvh.val = q_head.val / (state.n_head.val / state.n_head_kv.val))
    (hstride : stride.val = state.n_head_kv.val * state.head_dim.val)
    (hfirst : first.val = kvh.val * state.head_dim.val)
    (harith : stride.val * pos.val + first.val + state.head_dim.val ≤ Usize.max) :
    AttnValueWindow state.head_dim stride first pos vplane := by
  obtain ⟨hdom0, _hocc, _hscale, _hk, _hklen, hvlen, _hrowdot, _hscore, hvalu, hpos30, _hcarrier⟩ :=
    hdom
  obtain ⟨_hnlayer, _hnhead, hnkhead, hdhead, _hcap, hmod, _hl, hqhead, _hqlen, houtlen,
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
  refine ⟨harith, hpos30, ?_, ?_⟩
  · intro s hs
    exact plane_at_spec state pos kvh stride first vplane hvlen hfirst hstride hkvhlt hdhead
      hposcap s hs
  · intro s hs i hi
    have h1 := hvalu s hs i hi
    rw [← hkvh] at h1
    rw [planeRow_getD vplane.val state.n_head_kv.val state.head_dim.val kvh.val s i hi] at h1
    show |(vplane.val.getD (s * stride.val + first.val + i) 0#i64).val|
      ≤ (2 : ℤ) ^ (62 : ℕ)
    have hidx : s * stride.val + first.val + i
        = (s * state.n_head_kv.val + kvh.val) * state.head_dim.val + i := by
      rw [hstride, hfirst]
      ring
    rw [hidx]
    exact h1

/-! ## Per-component output -/

/-- **Per-component output.** Combining `attn_weight_window_of_numerators`,
`attn_value_window_of_v2`, `attn_value_fit` and `Numeric.AttnLoopOutput`'s
`value_component_spec`: on `attn_domain_v2`, with the numerator-loop facts that make the weights
the rounded probabilities `W_s = divRoundNearest (E_s·2^32) T` (so `0 ≤ W_s ≤ 2^32` and
`Σ W ≤ 2^32 + (pos+1)`), the value pass followed by the output pass returns, clear, and every
component `i < head_dim` is the exact half-biased reduction `divRoundNearest (Σ_{s≤pos} W_s·V_{s,i})
2^32` of the `valueList` entries — the claim the predecessor stated over the deleted
`attn.value_component`. The output length `out'.val.length = out.val.length` is exported as a third
conjunct, after the two loop equations: it is the component `value_component_spec` now threads
(`Numeric/AttnLoopOutput.lean`), and `attn_head_refines_v2` requires it. Additive: every existing
conjunct is unchanged. -/
theorem attn_value_components (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (kplane vplane : alloc.vec.Vec I64) (kvh stride first : Usize)
    (exps : alloc.vec.Vec I64) (total : I128) (recip : Option U64)
    (acc0 : alloc.vec.Vec I128) (sh : ℕ → ℤ)
    (hdom : attn_domain_v2 state layer q_head pos q out kplane vplane)
    (hkvh : kvh.val = q_head.val / (state.n_head.val / state.n_head_kv.val))
    (hstride : stride.val = state.n_head_kv.val * state.head_dim.val)
    (hfirst : first.val = kvh.val * state.head_dim.val)
    (harith : stride.val * pos.val + first.val + state.head_dim.val ≤ Usize.max)
    (hacl0 : acc0.val.length = state.head_dim.val)
    (hacc0zero : ∀ i, i < state.head_dim.val → (acc0.val.getD i 0#i128).val = 0)
    (hwres : AttnWeightResolve exps total pos recip)
    (hlen : exps.val.length = pos.val + 1)
    (hcase : ∀ j, j ≤ pos.val →
      sh j ≤ 0 ∧ attnNumeratorCase (sh j) (exps.val.getD j 0#i64).val)
    (hzero : ∃ j, j ≤ pos.val ∧ sh j = 0)
    (htot : total.val = ∑ j ∈ Finset.range (pos.val + 1), (exps.val.getD j 0#i64).val) :
    ∃ (acc : alloc.vec.Vec I128) (out' : Slice I64),
      attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip
          acc0 0#usize true none = ok (acc, none) ∧
      attn.KvState.attn_head_loop3 state.head_dim out acc 0#usize true none = ok (out', none) ∧
      out'.val.length = out.val.length ∧
      ∀ i : Usize, i.val < state.head_dim.val →
        (out'.val.getD i.val 0#i64).val =
          divRoundNearest
            (∑ s ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total s *
              (valueList vplane.val state.n_head_kv.val state.head_dim.val kvh.val pos.val
                i.val).getD s 0) ((2 : ℤ) ^ (32 : ℕ)) := by
  have hdomc := hdom
  obtain ⟨hdom0, _hocc, _hscale, _hk, _hklen, _hvlen, _hrowdot, _hscore, hvalu, hpos30, _hcarrier⟩ :=
    hdom
  obtain ⟨_hnlayer, _hnhead, _hnkhead, _hdhead, _hcap, _hmod, _hl, _hqhead, _hqlen, houtlen,
    _hposcap⟩ := hdom0
  have hwin := attn_weight_window_of_numerators exps total pos sh hlen hpos30 hcase hzero htot
  have hhvwin := attn_value_window_of_v2 state layer q_head pos q out kplane vplane kvh stride
    first hdomc hkvh hstride hfirst harith
  obtain ⟨acc, haccloop, hinv⟩ := attn_head_loop2_spec state.head_dim pos vplane stride first
    exps total recip acc0 hwin hhvwin hwres hacl0 hacc0zero
  -- the accumulator's per-component content, in the `stride`/`first` addressing
  have hcc : ∀ i, i < state.head_dim.val →
      (acc.val.getD i 0#i128).val =
        ∑ j ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total j *
          attnValueAt vplane.val (state.n_head_kv.val * state.head_dim.val)
            (kvh.val * state.head_dim.val) j i := by
    intro i hi
    have hz := hinv.2.2.2.2 i hi
    rw [hstride, hfirst] at hz
    rw [hz, hacc0zero i hi, zero_add]
  have hcontent := attn_head_loop2_component state.head_dim pos vplane exps total acc
    state.n_head_kv.val kvh.val hcc
  -- the weight bounds, read off the window rather than restated
  have hW0 : ∀ s ∈ Finset.range (pos.val + 1), 0 ≤ attnWeightAt exps.val total s ∧
      attnWeightAt exps.val total s ≤ (2 : ℤ) ^ (32 : ℕ) := by
    intro s hs
    have hsle : s ≤ pos.val := by simp only [Finset.mem_range] at hs; omega
    exact hwin.2.2.2.2 s hsle
  have hWsum : ∑ s ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total s
      ≤ (2 : ℤ) ^ (32 : ℕ) + ((pos.val + 1 : ℕ) : ℤ) := by
    have hEnonneg : ∀ j ∈ Finset.range (pos.val + 1), 0 ≤ (exps.val.getD j 0#i64).val := by
      intro j hj
      have hjle : j ≤ pos.val := by simp only [Finset.mem_range] at hj; omega
      exact (hwin.2.2.2.1 j hjle).1
    have hwb := attn_weight_bounds (pos.val + 1)
      (fun k => (exps.val.getD k 0#i64).val) (fun k => attnWeightAt exps.val total k) total.val
      hEnonneg htot hwin.2.2.1 (by intro k _; rfl)
    exact hwb.2
  -- the epilogue narrowing at each component, from the weighted-sum fit
  have hnarrow : ∀ (acc' : alloc.vec.Vec I128),
      attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip
          acc0 0#usize true none = ok (acc', none) →
      ∀ j, j < state.head_dim.val →
        I128.min ≤ (acc'.val.getD j 0#i128).val + 2 ^ (31 : ℕ) ∧
        (acc'.val.getD j 0#i128).val + 2 ^ (31 : ℕ) ≤ I128.max ∧
        -((2 : ℤ) ^ (63 : ℕ)) ≤ attnOutAt acc' j ∧ attnOutAt acc' j ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    intro acc' hacc' j hj
    have hcc' : acc' = acc := by
      have hsame : (ok (acc', none) : Result (alloc.vec.Vec I128 × Option attn.AttnError))
          = ok (acc, none) := by
        rw [hacc'] at haccloop
        exact haccloop
      exact congrArg Prod.fst (ok.injEq.mp hsame)
    have hcol : (acc'.val.getD j 0#i128).val
        = ∑ s ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total s *
          (valueList vplane.val state.n_head_kv.val state.head_dim.val kvh.val pos.val j).getD s 0 := by
      rw [hcc']
      exact hcontent j hj
    have hVabs : ∀ s ∈ Finset.range (pos.val + 1),
        |(valueList vplane.val state.n_head_kv.val state.head_dim.val kvh.val pos.val j).getD s 0|
          ≤ (2 : ℤ) ^ (62 : ℕ) := by
      intro s hs
      have hsle : s ≤ pos.val := by simp only [Finset.mem_range] at hs; omega
      rw [valueList_getD vplane.val state.n_head_kv.val state.head_dim.val kvh.val pos.val j s
        hj hsle]
      have h1 := hvalu s hsle j hj
      rw [← hkvh] at h1
      rwa [planeRow_getD vplane.val state.n_head_kv.val state.head_dim.val kvh.val s j hj] at h1
    have fit := attn_value_fit (pos.val + 1) (fun s => attnWeightAt exps.val total s)
      (fun s => (valueList vplane.val state.n_head_kv.val state.head_dim.val kvh.val pos.val j).getD s 0)
      pos rfl hpos30 hW0 hWsum hVabs
    simp only [attnOutAt]
    rw [hcol]
    exact fit
  -- the `i64` values the output loop writes at each component: the epilogue's own reduction. The
  -- loop's accumulator is unique (it is the value the value loop returns), so one `V` serves.
  obtain ⟨Vout, hVout⟩ :
      ∃ V : ℕ → I64, ∀ j : ℕ, j < state.head_dim.val → (V j).val = attnOutAt acc j := by
    have hI64 : ∀ j : ℕ, j < state.head_dim.val →
        ∃ y : I64, (y : I64).val = attnOutAt acc j :=
      fun j hj => by
      obtain ⟨hlo, hhi, hfl, hfh⟩ := hnarrow acc haccloop j hj
      obtain ⟨y, _, hyv⟩ := round_shift_32_spec (acc.val.getD j 0#i128) hlo hhi
      have hnb : (2 : ℤ) ^ (IScalarTy.I64.numBits - 1) = (2 : ℤ) ^ 63 := by
        rw [IScalarTy.I64_numBits_eq]
      have hyv' : y.val
          = divRoundNearest (acc.val.getD j 0#i128).val ((2 : ℤ) ^ (32 : ℕ)) := hyv
      have hdiv : divRoundNearest (acc.val.getD j 0#i128).val ((2 : ℤ) ^ (32 : ℕ))
          = attnOutAt acc j := rfl
      have hfl' : -((2 : ℤ) ^ (IScalarTy.I64.numBits - 1)) ≤ y.val := by
        rw [hyv', hdiv, hnb]
        exact hfl
      have hfh' : y.val < (2 : ℤ) ^ (IScalarTy.I64.numBits - 1) := by
        rw [hyv', hdiv, hnb]
        exact lt_of_le_of_lt hfh (by norm_num)
      have hcast : (IScalar.cast IScalarTy.I64 y).val = y.val := by
        rw [IScalar.cast_val_eq]
        norm_num [IScalarTy.numBits]
        exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 y.val hfl' hfh'
      refine ⟨IScalar.cast IScalarTy.I64 y, ?_⟩
      rw [hcast, hyv', hdiv]
    refine ⟨fun j => if h : j < state.head_dim.val then Classical.choose (hI64 j h) else 0#i64, ?_⟩
    intro j hj
    show ((if h : j < state.head_dim.val then Classical.choose (hI64 j h)
      else 0#i64) : I64).val = attnOutAt acc j
    rw [dif_pos hj]
    exact Classical.choose_spec (hI64 j hj)
  have hV : ∀ (a : alloc.vec.Vec I128),
      attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip
          acc0 0#usize true none = ok (a, none) →
      ∀ j : ℕ, j < state.head_dim.val → (Vout j).val = attnOutAt a j := by
    intro a ha j hj
    have hcc : a = acc := by
      have hsame : (ok (a, none) : Result (alloc.vec.Vec I128 × Option attn.AttnError))
          = ok (acc, none) := by
        rw [ha] at haccloop
        exact haccloop
      exact congrArg Prod.fst (ok.injEq.mp hsame)
    subst hcc
    exact hVout j hj
  obtain ⟨acc', out', hacc, hout, houtvlen, hval⟩ := value_component_spec state.head_dim pos vplane exps
    total recip acc0 out stride first state.n_head_kv.val kvh.val hstride hfirst hwin hhvwin hwres
    hacl0 hacc0zero houtlen Vout hV hnarrow
  exact ⟨acc', out', hacc, hout, houtvlen, fun i hi => hval i.val hi⟩

end

end apeiron_word