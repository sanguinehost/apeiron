/-
Score-side windows for the attention head.

`Numeric.AttnLoopScore` states the score/numerator loop specifications (`attn_head_loop0_spec`,
`attn_head_loop1_spec`) against the window hypotheses `attnHeadLoop0Domain`/`attnHeadLoop1Domain`;
`Numeric.AttnSoftmaxPerturb` states the corrected end-to-end precondition `attn_domain_v2`
(`anchor`) and the proved stage composition `attn_stage_compose_closed`. This module derives, from
`attn_domain_v2`, every scalar/window hypothesis those specifications and
`word_attn_head_ok` (`Numeric.EngineAttnRefinement`) need:

* `attn_v2_prefix`: the guard/prefix facts of `word_attn_head_unfold` — `len`, `slots`, `group`,
  `kvh`, the four refusals cleared, the two plane indices, `slots.val = pos.val + 1`, the GQA head
  index `kvh.val = attnKvh state q_head` with `kvh.val < n_head_kv.val`, and `pos.val + 1 ≤ Usize.max`;
* `attn_v2_loop0_domain`: the per-position plane-row window of `attnHeadLoop0Domain`, from the
  `attn_domain_v2` geometry (plane length, occupancy, the `row_dot_domain` conjunct, the score
  window through `scoreList_getD_eq`);
* `attn_v2_loop1_domain`: `attnHeadLoop1Domain`, the checked `i64` subtraction `score_j - mx`
  succeeding with a nonpositive result, from the score window and the running maximum;
* `attn_loop0_ok`/`attn_loop1_ok`: the two loop specs at the actual initial vectors
  (`Vec.with_capacity Std.I64 slots`, `mx = I64.MIN`, `total = 0`) in equality form;
* `attn_loop_stage_facts`: the `Finset.range (pos + 1)` repackaging of the two loop postconditions
  — `hmax`, `hnum`, `hT` — that `attn_stage_compose_closed` takes, plus the shift nonpositivity.

## Trust boundary

Reaches the extracted `attn.KvState.attn_head_loop0`, `attn.KvState.attn_head_loop1`, the plane
window of `attn.score_at` and the Q32.32 denotation in `Numeric.Scale`. Derived only from
`attn_domain_v2`; no new hypothesis beyond it is introduced. Does not cover rustc/LLVM codegen of
`src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.AttnLoopScore
import Numeric.AttnSoftmaxPerturb

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Scalar helpers -/

/-- `2^30 ≤ Usize.max` on either platform word size. -/
private theorem attn_two_pow_30_le_usize_max : 2 ^ (30 : ℕ) ≤ Usize.max := by
  have h : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  rcases System.Platform.numBits_eq with h' | h' <;> rw [h, h'] <;> norm_num

/-- `Usize.checked_add` succeeds whenever the exact sum is representable. -/
private theorem usize_checked_add_some {x y : Usize} (h : x.val + y.val ≤ Usize.max) :
    ∃ z, Usize.checked_add x y = some z ∧ z.val = x.val + y.val := by
  have hspec := Usize.checked_add_bv_spec x y
  cases hc : Usize.checked_add x y with
  | none => rw [hc] at hspec; exact absurd h (not_le.mpr hspec)
  | some z => rw [hc] at hspec; exact ⟨z, rfl, hspec.2.1⟩

/-- `Usize.checked_mul` succeeds whenever the exact product is representable. -/
private theorem usize_checked_mul_some {x y : Usize} (h : x.val * y.val ≤ Usize.max) :
    ∃ z, Usize.checked_mul x y = some z ∧ z.val = x.val * y.val := by
  have hspec := Usize.checked_mul_bv_spec x y
  cases hc : Usize.checked_mul x y with
  | none => rw [hc] at hspec; exact absurd h (not_le.mpr hspec)
  | some z => rw [hc] at hspec; exact ⟨z, rfl, hspec.2.1⟩

/-- `n_head % n_head_kv = 0` with both positive forces `n_head / n_head_kv ≥ 1`. -/
private theorem nat_div_pos_of_mod_zero {m n : ℕ} (hm : 0 < m) (hn : 0 < n) (hmod : m % n = 0) :
    0 < m / n :=
  Nat.div_pos (Nat.le_of_dvd hm (Nat.dvd_of_mod_eq_zero hmod)) hn

/-- `getD` at an in-range index is the checked `getElem`. -/
private theorem getD_eq_get {l : List I64} {n : ℕ} (h : n < l.length) :
    l.getD n 0#i64 = l[n]'h := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

/-- The mapped-`List.sum` of a `Vec`'s `I64` values is the `Finset.range` sum of its `getD`
entries. -/
private theorem list_sum_map_val_eq_finset (l : List I64) :
    (l.map (fun x => x.val)).sum =
      ∑ j ∈ Finset.range l.length, (l.getD j 0#i64).val := by
  induction l with
  | nil => simp
  | cons a t ih =>
    rw [List.length_cons, Finset.sum_range_succ', List.map_cons, List.sum_cons, ih]
    simp only [List.getD_cons_zero, List.getD_cons_succ]
    omega

/-- A list slice `[s, s+n)` below the list length is the `getD` window. -/
private theorem list_slice_eq_map_getD (l : List I64) (s n : ℕ) (h : s + n ≤ l.length) :
    l.slice s (s + n) = (List.range n).map (fun i => l.getD (s + i) 0#i64) := by
  refine List.ext_getElem ?_ ?_
  · rw [List.slice_length, List.length_map, List.length_range]
    omega
  · intro k h1 h2
    have hlen' : (l.slice s (s + n)).length = n := by
      rw [List.slice_length]
      omega
    have hk : k < n := by omega
    rw [List.getElem_map, List.getElem_range]
    rw [List.getElem_slice s (s + n) k l ⟨h, by omega⟩]
    rw [getD_eq_get (l := l) (by omega)]

/-- `Vec.with_capacity` starts empty. -/
@[simp]
theorem vec_with_capacity_val (T : Type) (n : Usize) :
    (alloc.vec.Vec.with_capacity T n).val = [] := rfl

/-! ## GQA head index -/

/-- The KV head index of query head `q_head`: `q_head / (n_head / n_head_kv)`. -/
def attnKvh (state : attn.KvState) (q_head : Usize) : ℕ :=
  q_head.val / (state.n_head.val / state.n_head_kv.val)

/-! ## The `attn_domain_v2` prefix -/

/-- **Prefix facts of the extracted guard chain.** From `attn_domain_v2`, the four guard refusals
are cleared, the occupancy index and the two plane indices resolve, `slots = pos + 1`, the GQA head
index `kvh` is `attnKvh` and is strictly below `n_head_kv`, and `pos + 1` fits `usize` — exactly the
non-loop hypotheses of `word_attn_head_ok` (`Numeric.EngineAttnRefinement`). -/
theorem attn_v2_prefix (state : attn.KvState) (layer q_head pos : Usize) (q out : Slice I64)
    (kplane vplane : alloc.vec.Vec I64)
    (h : attn_domain_v2 state layer q_head pos q out kplane vplane) :
    ∃ (len slots group kvh : Usize),
      ¬ (layer ≥ state.n_layer) ∧
      ¬ (q_head ≥ state.n_head) ∧
      (Slice.len q != state.head_dim) = false ∧
      (Slice.len out != state.head_dim) = false ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize) state.occupied layer
        = ok len ∧
      ¬ (pos ≥ len) ∧
      Usize.checked_add pos 1#usize = some slots ∧
      slots.val = pos.val + 1 ∧
      state.n_head / state.n_head_kv = ok group ∧
      q_head / group = ok kvh ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64))
        state.k layer = ok kplane ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64))
        state.v layer = ok vplane ∧
      kvh.val = attnKvh state q_head ∧
      kvh.val < state.n_head_kv.val ∧
      pos.val + 1 ≤ Usize.max := by
  unfold attn_domain_v2 at h
  obtain ⟨hadom, ⟨len, hocc, hposlt⟩, _hscale, hk, hv, _hklen, _hvlen, _hrowdot, _hscore,
    _hval, hpos30⟩ := h
  unfold attn_domain at hadom
  obtain ⟨_hnlayer, hnhead, hnkv, _hhdim, _hcap, hmod, hlayer, hqhead, hqlen, houtlen,
    _hposcap⟩ := hadom
  have hlayer' : ¬ (layer ≥ state.n_layer) := by
    intro hc
    have := (UScalar.le_equiv state.n_layer layer).1 hc
    omega
  have hhead' : ¬ (q_head ≥ state.n_head) := by
    intro hc
    have := (UScalar.le_equiv state.n_head q_head).1 hc
    omega
  have hqeq : Slice.len q = state.head_dim :=
    UScalar.eq_of_val_eq (by rw [Slice.len_val]; exact hqlen)
  have houteq : Slice.len out = state.head_dim :=
    UScalar.eq_of_val_eq (by rw [Slice.len_val]; exact houtlen)
  have hq' : (Slice.len q != state.head_dim) = false := by simp [hqeq]
  have hout' : (Slice.len out != state.head_dim) = false := by simp [houteq]
  have hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize) state.occupied
      layer = ok len := by
    rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
      alloc.vec.Vec.getElem?_Nat_eq, hocc]
  have hpos' : ¬ (pos ≥ len) := by
    intro hc
    have := (UScalar.le_equiv len pos).1 hc
    omega
  have h1v : (1#usize).val = 1 := by norm_num
  have hsum : pos.val + (1#usize).val ≤ Usize.max := by
    rw [h1v]
    exact le_trans hpos30 attn_two_pow_30_le_usize_max
  obtain ⟨slots, hadd, hslots⟩ := usize_checked_add_some hsum
  have hslots' : slots.val = pos.val + 1 := by rw [hslots, h1v]
  have hnkv' : (state.n_head_kv : Nat) ≠ 0 := by
    show state.n_head_kv.val ≠ 0
    omega
  obtain ⟨group, hgroup, hgroupv⟩ :=
    UScalar.div_spec (x := state.n_head) (y := state.n_head_kv) hnkv'
  have hgrouppos : 0 < group.val := by
    rw [hgroupv]
    exact nat_div_pos_of_mod_zero hnhead hnkv hmod
  have hgnz : (group : Nat) ≠ 0 := by
    show group.val ≠ 0
    omega
  obtain ⟨kvh, hkvh, hkvhv⟩ := UScalar.div_spec (x := q_head) (y := group) hgnz
  have hdvd : state.n_head_kv.val ∣ state.n_head.val := Nat.dvd_of_mod_eq_zero hmod
  have hkvlt : kvh.val < state.n_head_kv.val := by
    rw [hkvhv]
    apply Nat.div_lt_of_lt_mul
    rw [hgroupv, Nat.div_mul_cancel hdvd]
    exact hqhead
  have hkvhKvh : kvh.val = attnKvh state q_head := by
    rw [hkvhv, hgroupv, attnKvh]
  refine ⟨len, slots, group, kvh, hlayer', hhead', hq', hout', hlen, hpos', hadd, hslots',
    hgroup, hkvh, ?_, ?_, hkvhKvh, hkvlt, ?_⟩
  · rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
      alloc.vec.Vec.getElem?_Nat_eq, hk]
  · rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
      alloc.vec.Vec.getElem?_Nat_eq, hv]
  · exact le_trans hpos30 attn_two_pow_30_le_usize_max

/-! ## Stage facts for `attn_stage_compose_closed` -/

/-- **Loop postconditions in the composition form.** From the two extracted loops' outputs — the
score loop's `scores`, `mx` (`mx` the running maximum of `scores`, attained), the numerator loop's
`exps`, `total` — the `Finset.range (pos + 1)` hypotheses `attn_stage_compose_closed` takes:
`hmax` (`mx` attained by a score), `hnum` (`attnNumeratorCase (scores[j] - mx) exps[j]`), `hT`
(`total` the sum of the numerators), and the shift nonpositivity `scores[j] - mx ≤ 0`. -/
theorem attn_loop_stage_facts (pos : Usize) (scores exps : alloc.vec.Vec I64) (mx : I64)
    (total : I128)
    (hslen : scores.val.length = pos.val + 1)
    (hexpslen : exps.val.length = pos.val + 1)
    (hmx : mx.val = intListMax core.num.I64.MIN.val (scores.val.map (fun x => x.val)))
    (hmax0 : ∃ j, j < scores.val.length ∧ scores.val.getD j 0#i64 = mx)
    (hcase : ∀ j, j < exps.val.length →
      attnNumeratorCase ((scores.val.getD j 0#i64).val - mx.val)
        (exps.val.getD j 0#i64).val)
    (htoteq : total.val = (exps.val.map (fun x => x.val)).sum) :
    (∃ j ∈ Finset.range (pos.val + 1), (scores.val.getD j 0#i64).val = mx.val) ∧
    (∀ j ∈ Finset.range (pos.val + 1),
      attnNumeratorCase ((scores.val.getD j 0#i64).val - mx.val)
        (exps.val.getD j 0#i64).val) ∧
    total.val = ∑ j ∈ Finset.range (pos.val + 1), (exps.val.getD j 0#i64).val ∧
    (∀ j ∈ Finset.range (pos.val + 1), (scores.val.getD j 0#i64).val - mx.val ≤ 0) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · obtain ⟨j, hj, hjeq⟩ := hmax0
    exact ⟨j, Finset.mem_range.mpr (by omega), by rw [hjeq]⟩
  · intro j hj
    rw [Finset.mem_range] at hj
    exact hcase j (by rw [hexpslen]; omega)
  · rw [htoteq]
    rw [list_sum_map_val_eq_finset exps.val, hexpslen]
  · intro j hj
    rw [Finset.mem_range] at hj
    have hjlt : j < scores.val.length := by omega
    have hmem : (scores.val.getD j 0#i64).val ∈ scores.val.map (fun x => x.val) := by
      rw [getD_eq_get hjlt]
      exact List.mem_map.mpr ⟨_, List.getElem_mem hjlt, rfl⟩
    have hle := intListMax_ge_mem core.num.I64.MIN.val (scores.val.map (fun x => x.val)) _
      hmem
    rw [← hmx] at hle
    omega

/-! ## The `attn_domain_v2` loop-1 window -/

/-- **Numerator-loop window from `attn_domain_v2`.** On the corrected domain, with the score loop's
output (`scores` the causal prefix scores, `mx` their running maximum, attained), every checked
`i64` subtraction `score_j - mx` succeeds with a nonpositive result, so `attnHeadLoop1Domain`
holds. The score window `|score_j| ≤ 2^62` and `mx ≥ I64.MIN` bound the shift below by `-2^63`. -/
theorem attn_v2_loop1_domain (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (kplane vplane : alloc.vec.Vec I64)
    (h : attn_domain_v2 state layer q_head pos q out kplane vplane)
    (scores : alloc.vec.Vec I64) (mx : I64)
    (hslen : scores.val.length = pos.val + 1)
    (hpt : ∀ j, j < scores.val.length →
      (scores.val.getD j 0#i64).val =
        (scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
          (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val
          pos.val).getD j 0)
    (hmx : mx.val = intListMax core.num.I64.MIN.val (scores.val.map (fun x => x.val)))
    (hmax : ∃ j, j < scores.val.length ∧ scores.val.getD j 0#i64 = mx) :
    attnHeadLoop1Domain pos scores mx := by
  unfold attn_domain_v2 at h
  obtain ⟨_hadom, _hocc, _hscale, _hk, _hv, _hklen, _hvlen, _hrowdot, hscoreWin, _hval,
    hpos30⟩ := h
  refine ⟨hpos30, ?_⟩
  intro j hj
  have hjlt : j < scores.val.length := by omega
  have hscoreAt : ∀ k, k < scores.val.length →
      |(scores.val.getD k 0#i64).val| ≤ (2 : ℤ) ^ (62 : ℕ) := by
    intro k hk
    have hmem : (scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
        (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val).getD k 0
        ∈ scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
          (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val := by
      have hidx : k < (scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
          (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val).length := by
        unfold scoreList
        simp only [List.length_map, List.length_range]
        omega
      rw [List.getD_eq_getElem _ _ hidx]
      exact List.getElem_mem hidx
    rw [hpt k hk]
    exact hscoreWin _ hmem
  have hmxle : mx.val ≤ (2 : ℤ) ^ (62 : ℕ) := by
    obtain ⟨j0, hj0, hj0eq⟩ := hmax
    rw [← hj0eq]
    exact (abs_le.mp (hscoreAt j0 hj0)).2
  have hmxge : -((2 : ℤ) ^ (63 : ℕ)) ≤ mx.val := by
    have hseed := intListMax_ge_seed core.num.I64.MIN.val (scores.val.map (fun x => x.val))
    rw [← hmx] at hseed
    have hMIN : core.num.I64.MIN.val = -((2 : ℤ) ^ (63 : ℕ)) := by decide
    rwa [hMIN] at hseed
  have hshift : (scores.val.getD j 0#i64).val - mx.val ≤ 0 := by
    have hmem : (scores.val.getD j 0#i64).val ∈ scores.val.map (fun x => x.val) := by
      rw [getD_eq_get hjlt]
      exact List.mem_map.mpr ⟨_, List.getElem_mem hjlt, rfl⟩
    have hle := intListMax_ge_mem core.num.I64.MIN.val (scores.val.map (fun x => x.val)) _ hmem
    rw [← hmx] at hle
    omega
  have hshiftlo : I64.min ≤ (scores.val.getD j 0#i64).val - mx.val := by
    have h1 := (abs_le.mp (hscoreAt j hjlt)).1
    have hMIN : I64.min = -((2 : ℤ) ^ (63 : ℕ)) := by
      norm_num [I64.min, IScalar.min, I64.numBits]
    rw [hMIN]
    have h2 : (2 : ℤ) ^ (62 : ℕ) + (2 : ℤ) ^ (62 : ℕ) = (2 : ℤ) ^ (63 : ℕ) := by norm_num
    linarith
  have hshifthi : (scores.val.getD j 0#i64).val - mx.val ≤ I64.max := by
    have hMAX : I64.max = (2 : ℤ) ^ (63 : ℕ) - 1 := by
      norm_num [I64.max, IScalar.max, I64.numBits]
    rw [hMAX]
    omega
  cases hc : I64.checked_sub (scores.val.getD j 0#i64) mx with
  | none =>
    have hspec := I64.checked_sub_bv_spec (scores.val.getD j 0#i64) mx
    rw [show core.num.checked_sub_IScalar (scores.val.getD j 0#i64) mx
        = I64.checked_sub (scores.val.getD j 0#i64) mx from rfl] at hspec
    rw [hc] at hspec
    exact absurd ⟨hshiftlo, hshifthi⟩ hspec
  | some z =>
    exact ⟨z, rfl, by rw [I64_checked_sub_val hc]; exact hshift⟩

/-! ## The loop specs at the actual initial vectors -/

/-- **Score loop, equality form.** At the actual extracted initial vector
(`Vec.with_capacity Std.I64 slots`, `mx = I64.MIN`), `attn_head_loop0` returns
`Ok (scores, mx, none)` with `scores` the whole `scoreList` and `mx` its attained maximum. -/
theorem attn_loop0_ok (state : attn.KvState) (kplane : alloc.vec.Vec I64) (q : Slice I64)
    (pos kvh slots : Usize)
    (hslots : pos.val + 1 ≤ Usize.max)
    (hdom : attnHeadLoop0Domain state kplane q state.scale state.n_head_kv kvh pos) :
    ∃ scores mx,
      attn.KvState.attn_head_loop0 state state.n_head_kv state.scale q pos kvh
          (alloc.vec.Vec.with_capacity Std.I64 slots) core.num.I64.MIN kplane 0#usize true none
        = ok (scores, mx, none) ∧
      scores.val.length = pos.val + 1 ∧
      (∀ (j : ℕ) (_hj : j < scores.val.length),
        (scores.val.getD j 0#i64).val =
          (scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val kvh.val
            state.scale.val pos.val).getD j 0) ∧
      mx.val = intListMax core.num.I64.MIN.val (scores.val.map (fun x => x.val)) ∧
      ∃ (j : ℕ) (_hj : j < scores.val.length), scores.val.getD j 0#i64 = mx := by
  obtain ⟨b, hb, scores, mx, hbeq, hlen, hpt, hmx, hj⟩ := WP.spec_imp_exists
    (attn_head_loop0_spec state kplane q state.scale state.n_head_kv kvh pos hslots hdom
      (alloc.vec.Vec.with_capacity Std.I64 slots) (vec_with_capacity_val I64 slots))
  exact ⟨scores, mx, by rw [hbeq] at hb; exact hb, hlen, hpt, hmx, hj⟩

/-- **Numerator loop, equality form.** At the actual extracted initial vector
(`Vec.with_capacity Std.I64 slots`, `total = 0`), `attn_head_loop1` returns
`Ok (exps, total, none)` with the `attnNumeratorCase` family and `total` their nonnegative sum. -/
theorem attn_loop1_ok (pos : Usize) (scores : alloc.vec.Vec I64) (mx : I64) (slots : Usize)
    (hslen : scores.val.length = pos.val + 1)
    (hslots : pos.val + 1 ≤ Usize.max)
    (hdom : attnHeadLoop1Domain pos scores mx) :
    ∃ exps total,
      attn.KvState.attn_head_loop1 pos scores mx
          (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
        = ok (exps, total, none) ∧
      exps.val.length = pos.val + 1 ∧
      (∀ j, j < exps.val.length →
        attnNumeratorCase ((scores.val.getD j 0#i64).val - mx.val)
          (exps.val.getD j 0#i64).val) ∧
      0 ≤ total.val ∧
      total.val = (exps.val.map (fun x => x.val)).sum := by
  obtain ⟨b, hb, exps, total, hbeq, hlen, hpt, htot, htoteq⟩ := WP.spec_imp_exists
    (attn_head_loop1_spec pos scores mx hslen hslots hdom
      (alloc.vec.Vec.with_capacity Std.I64 slots) (vec_with_capacity_val I64 slots))
  exact ⟨exps, total, by rw [hbeq] at hb; exact hb, hlen, hpt, htot, htoteq⟩

/-! ## GQA head index bound -/

/-- The KV head index `attnKvh` is strictly below `n_head_kv`: `n_head % n_head_kv = 0` makes
`n_head / n_head_kv ≥ 1`, and `q_head < n_head = (n_head / n_head_kv) · n_head_kv`. -/
private theorem attnKvh_lt_nkv (state : attn.KvState) (q_head : Usize)
    (hmod : state.n_head.val % state.n_head_kv.val = 0)
    (_hnhead : 0 < state.n_head.val) (_hnkv : 0 < state.n_head_kv.val)
    (hqhead : q_head.val < state.n_head.val) :
    attnKvh state q_head < state.n_head_kv.val := by
  unfold attnKvh
  have hdvd : state.n_head_kv.val ∣ state.n_head.val := Nat.dvd_of_mod_eq_zero hmod
  exact Nat.div_lt_of_lt_mul (by rw [Nat.div_mul_cancel hdvd]; exact hqhead)

/-! ## The `attn_domain_v2` loop-0 window -/

/-- **Plane-row window from `attn_domain_v2`.** On the corrected domain with the GQA head index
`kvh = attnKvh`, every causal position `s' ≤ pos` gives the four checked `usize` address steps
`((s'·kv + kvh)·hd, +hd)`, a range index into the plane returning the `planeRow` window, that row on
`row_dot_domain` against `q`, and the `attnScoreInt` result inside `[-2^62, 2^62]` — exactly
`attnHeadLoop0Domain`, hence the per-position hypothesis of `attn_head_loop0_spec`. The bound is
`(s'·kv + kvh + 1)·hd ≤ kv·cap·hd = kplane.val.length ≤ Usize.max`. -/
theorem attn_v2_loop0_domain (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (kplane vplane : alloc.vec.Vec I64) (kvh : Usize)
    (h : attn_domain_v2 state layer q_head pos q out kplane vplane)
    (hkvh : kvh.val = attnKvh state q_head) :
    attnHeadLoop0Domain state kplane q state.scale state.n_head_kv kvh pos := by
  unfold attnHeadLoop0Domain
  intro s' hs'
  unfold attn_domain_v2 at h
  obtain ⟨hadom, _hocc, _hscale, _hk, _hv, hklen, _hvlen, hrowdot, hscoreWin, _hval,
    _hpos30⟩ := h
  unfold attn_domain at hadom
  obtain ⟨_hnlayer, hnhead, hnkv, hhdim, _hcap, hmod, _hlayer, hqhead, _hqlen, _houtlen,
    hposcap⟩ := hadom
  have hplaneMax : kplane.val.length ≤ Usize.max := alloc.vec.Vec.property kplane
  have hcapKVhd : state.cap.val * state.n_head_kv.val * state.head_dim.val
      = kplane.val.length := by
    rw [hklen]; ring
  have hcapKV : state.cap.val * state.n_head_kv.val ≤ kplane.val.length := by
    calc state.cap.val * state.n_head_kv.val
        ≤ state.cap.val * state.n_head_kv.val * state.head_dim.val :=
          Nat.le_mul_of_pos_right _ hhdim
      _ = kplane.val.length := hcapKVhd
  have hkvlt : kvh.val < state.n_head_kv.val := by
    rw [hkvh]
    exact attnKvh_lt_nkv state q_head hmod hnhead hnkv hqhead
  have hkvh' : kvh.val = q_head.val / (state.n_head.val / state.n_head_kv.val) := hkvh
  have hs'lt : s'.val < state.cap.val := lt_of_le_of_lt hs' hposcap
  have hs'1 : s'.val + 1 ≤ state.cap.val := hs'lt
  have hbaseb : s'.val * state.n_head_kv.val ≤ Usize.max :=
    le_trans (le_trans (Nat.mul_le_mul_right _ (Nat.le_of_lt hs'lt)) hcapKV) hplaneMax
  obtain ⟨base, hbase, hbasev⟩ := usize_checked_mul_some hbaseb
  have hbkv1 : base.val + kvh.val + 1 ≤ state.cap.val * state.n_head_kv.val := by
    rw [hbasev]
    calc s'.val * state.n_head_kv.val + kvh.val + 1
        ≤ s'.val * state.n_head_kv.val + state.n_head_kv.val := by omega
      _ = (s'.val + 1) * state.n_head_kv.val := by ring
      _ ≤ state.cap.val * state.n_head_kv.val := Nat.mul_le_mul_right _ hs'1
  have hbkv : base.val + kvh.val ≤ state.cap.val * state.n_head_kv.val := by omega
  obtain ⟨val1, hval1, hval1v⟩ :=
    usize_checked_add_some (le_trans hbkv (le_trans hcapKV hplaneMax))
  have hv1le : val1.val * state.head_dim.val
      ≤ state.cap.val * state.n_head_kv.val * state.head_dim.val := by
    rw [hval1v]
    exact Nat.mul_le_mul_right _ hbkv
  obtain ⟨val2, hval2, hval2v⟩ :=
    usize_checked_mul_some (le_trans (le_trans hv1le (le_of_eq hcapKVhd)) hplaneMax)
  have hv23le : val2.val + state.head_dim.val ≤ kplane.val.length := by
    rw [hval2v]
    calc val1.val * state.head_dim.val + state.head_dim.val
        = (val1.val + 1) * state.head_dim.val := by ring
      _ ≤ (state.cap.val * state.n_head_kv.val) * state.head_dim.val :=
          Nat.mul_le_mul_right _ (by rw [hval1v]; exact hbkv1)
      _ = kplane.val.length := hcapKVhd
  obtain ⟨val3, hval3, hval3v⟩ :=
    usize_checked_add_some (le_trans hv23le hplaneMax)
  have hval23 : val2.val ≤ val3.val := by rw [hval3v]; omega
  have hval3len : val3.val ≤ kplane.val.length := by rw [hval3v]; exact hv23le
  have hslice : val2.val + state.head_dim.val ≤ kplane.val.length := by
    rw [← hval3v]; exact hval3len
  obtain ⟨krow, hidxRaw, hkrowval0, _⟩ := WP.spec_imp_exists
    (core.slice.index.SliceIndexRangeUsizeSlice.index.step_spec
      ({ start := val2, «end» := val3 } : core.ops.range.Range Usize)
      (alloc.vec.Vec.deref kplane) ((UScalar.le_equiv val2 val3).2 hval23)
      (by
        have hd : (alloc.vec.Vec.deref kplane).length = kplane.val.length := by
          simp [alloc.vec.Vec.deref]
        rw [hd]
        exact hval3len))
  have hidx : core.slice.index.Slice.index (core.slice.index.SliceIndexRangeUsizeSlice Std.I64)
      (alloc.vec.Vec.deref kplane) { start := val2, «end» := val3 } = ok krow := by
    rw [Slice.index_SliceIndexRangeUsizeSliceInst]
    exact hidxRaw
  have hkrowval : krow.val = planeRow kplane.val state.n_head_kv.val state.head_dim.val
      kvh.val s'.val := by
    rw [hkrowval0]
    have hderef : (alloc.vec.Vec.deref kplane).val = kplane.val := Slice.from_val kplane.val _
    rw [hderef, hval3v]
    rw [list_slice_eq_map_getD kplane.val val2.val state.head_dim.val hslice]
    unfold planeRow
    apply List.map_congr_left
    intro i _
    have harith : (s'.val * state.n_head_kv.val + kvh.val) * state.head_dim.val = val2.val := by
      rw [← hbasev, ← hval1v, ← hval2v]
    rw [← harith]
  have hdotdom : row_dot_domain krow q := by
    refine hrowdot s'.val hs' krow ?_
    rw [hkrowval, hkvh']
  have hkl : krow.val.length = q.val.length := hdotdom.1
  have hscEq := scoreList_getD_eq kplane.val q state.n_head_kv.val state.head_dim.val kvh.val
    state.scale.val pos.val s'.val hs' krow hkrowval hkl
  have hscEq2 : (scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
      (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val).getD
        s'.val 0
      = attnScoreInt (dotPrefix krow q krow.val.length) state.scale.val := by
    rw [← hscEq, ← hkvh']
  have hscMem : attnScoreInt (dotPrefix krow q krow.val.length) state.scale.val
      ∈ scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
          (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val := by
    rw [← hscEq2]
    have hidx2 : s'.val < (scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
        (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val).length := by
      unfold scoreList
      simp only [List.length_map, List.length_range]
      omega
    rw [List.getD_eq_getElem _ _ hidx2]
    exact List.getElem_mem hidx2
  have hb := abs_le.mp (hscoreWin _ hscMem)
  exact ⟨base, val1, val2, val3, krow, hbase, hval1, hval2, hval3, hidx, hkrowval, hdotdom,
    hb.1, hb.2⟩

end

end apeiron_word
