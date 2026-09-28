/-
Content of the weight and output loops of the extracted single-token attention kernel
.

`Numeric.EngineAttnRefinement` proves the per-call identities `weight_at_spec` (the weight
division) and `value_component_epilogue` (the output epilogue), leaving the four
`attn_head_loop0..3` loop specifications open (residual item 1 there and in
`Numeric.AttnSoftmaxPerturb`). This module discharges that residual for the weight loop
(`attn_head_loop2`), the per-component value accumulation (`attn.value_component_loop`, the body
of `value_component`) and the output loop (`attn_head_loop3`):

* `attn_head_loop2_spec`: on `AttnWeightWindow` (every numerator nonnegative, the total positive,
  and the per-index `weight_at_spec` headroom/fit), the weight loop returns, for every causal
  index `j ≤ pos`, the exact division `attnWeightAt exps total j = divRoundNearest (E_j·2^32) T`,
  clear;
* `value_component_spec`: on `AttnValueWindow` (weights/plane range-index identity, per-term
  magnitude), `value_component` returns the exact accumulation `Σ_{s≤pos} W_s·V_{s,i}` (`V` the
  `valueList` entries) reduced through `value_component_epilogue`'s output narrowing, clear;
* `attn_head_loop3_spec`: on a per-component `value_component` result function `V`, the output
  loop returns, for every component `i < head_dim`, exactly `V i` — the equational invariant
  `outPrefix`, the closed-form reference in the style of `Numeric.RopeHeadContent`'s `rotPrefix`.

## Windows not supplied by `attn_domain_v2`

`attn_domain_v2` (`Numeric.AttnSoftmaxPerturb`) states the value window (`|v_{j,i}| ≤ 2^62`) and
`pos + 1 ≤ 2^30`, and its own header already documents an implicit weight-magnitude assumption
(`"|W_j·v_{j,i}| ≤ 2^95"`) it never states as a conjunct. Concretely:

* `AttnWeightWindow`'s numerator sign (`0 ≤ E_j`) and total positivity (`0 < T`) are guard facts
  the extracted `word_attn_head` establishes upstream (`total ≤ 0` refuses `ZeroSum` before the
  weight loop runs; `fixed_exp`'s codomain is nonnegative) — no `attn_domain_v2` conjunct;
* `AttnWeightWindow`'s per-index `weight_at_spec` headroom (`e·2^32 + T/2 ≤ I128.max`) and `i64`
  fit are consequences of `0 ≤ E_j ≤ T` (a weight is a rounded probability), not stated by
  `attn_domain_v2`;
* `AttnValueWindow`'s weight-magnitude bound (`|W_s| ≤ 2^33`) is exactly the gap named above: no
  `attn_domain_v2` conjunct supplies it. It is supplied here as an explicit hypothesis, satisfiable
  because a `weight_at_spec` output is a rounded ratio in `[0, 2^32]`, not re-derived in this file;
* `AttnValueWindow`'s plane range-index identity (`attn.plane_at kv kvh hd s i = ok (Ok idx)`,
  `idx.val = (s·kv+kvh)·hd+i`, in bounds) is the "range-index identity tying `planeRow` to the
  plane contents" `Numeric.EngineAttnRefinement`'s header still lists as residual; `attn_domain_v2`
  gives the plane *lengths* (`kplane.val.length`/`vplane.val.length` conjuncts) but not this checked
  arithmetic, which is a consequence of those lengths and `s ≤ pos`, `i < hd` not derived here;
* `value_component_epilogue`'s final narrowing (`hlo`, `hhi`, `hfit`) is passed through unchanged
  (reused, not restated); it is the weighted-mean bound `|out_i| ≤ max_j |v_{j,i}|` (well inside
  `i64` given the value window), not derived in this file.

## Trust boundary

Reaches the extracted `attn.KvState.attn_head_loop2`, `attn.value_component_loop` and
`attn.KvState.attn_head_loop3` of `proof/ApeironWord.lean` at the digest
`Numeric.EngineAttnRefinement` binds to. Does not cover rustc/LLVM codegen of
`src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.AttnSoftmaxPerturb

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## List-index helpers -/

/-- `getD` at an in-range index is the checked `getElem`. -/
private theorem getD_eq_get {l : List I64} {n : ℕ} (h : n < l.length) :
    l.getD n 0#i64 = l[n]'h := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

/-- `(2:ℕ)^30 ≤ Usize.max` on either platform word size. -/
theorem usize_max_ge_2_pow_30 : (2 : ℕ) ^ 30 ≤ Usize.max := by
  have h : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  rcases System.Platform.numBits_eq with h' | h' <;> simp [h, h']

/-! ## Weight stage: the loop content -/

/-- The exact weight of causal index `j`: `divRoundNearest (E_j·2^32) T`
(`attn.weight_at`, `src/engine/attn.rs:143-152`). -/
def attnWeightAt (exps : List I64) (total : I128) (j : ℕ) : ℤ :=
  divRoundNearest ((exps.getD j 0#i64).val * (2 : ℤ) ^ (32 : ℕ)) total.val

/-- The weight-stage window: nonnegative numerators, positive total, room for the loop to grow the
weight vector without exceeding `usize`, and the per-index `weight_at_spec` premises. See the
module header for which conjuncts `attn_domain_v2` does and does not supply. -/
def AttnWeightWindow (exps : alloc.vec.Vec I64) (total : I128) (pos : Usize) : Prop :=
  exps.val.length = pos.val + 1 ∧
  pos.val + 1 ≤ Usize.max ∧
  0 < total.val ∧
  ∀ j, j ≤ pos.val →
    0 ≤ (exps.val.getD j 0#i64).val ∧
    (exps.val.getD j 0#i64).val * (2 : ℤ) ^ (32 : ℕ) + total.val / 2 ≤ I128.max ∧
    -((2 : ℤ) ^ (63 : ℕ)) ≤ attnWeightAt exps.val total j ∧
    attnWeightAt exps.val total j ≤ (2 : ℤ) ^ (63 : ℕ) - 1

/-- Weight-loop invariant: the flag clear, no error, `s` within one of `pos`, `s` causal weights
written, each exactly `attnWeightAt`. -/
def AttnWeightLoopInv (exps : alloc.vec.Vec I64) (total : I128) (pos : Usize)
    (weights : alloc.vec.Vec I64) (s : Usize) (ok1 : Bool) (err : Option attn.AttnError) : Prop :=
  ok1 = true ∧ err = none ∧ s.val ≤ pos.val + 1 ∧
  weights.val.length = s.val ∧
  ∀ j, j < weights.val.length → (weights.val.getD j 0#i64).val = attnWeightAt exps.val total j

/-- **One weight-loop step.** With `s ≤ pos` and the invariant holding, the body reads `exps[s]`,
computes `weight_at exps[s] total` via `weight_at_spec` (reused, not restated), pushes it and
continues at `s + 1` with the invariant restored. -/
theorem attn_head_loop2_body_step (exps : alloc.vec.Vec I64) (total : I128) (pos : Usize)
    (hwin : AttnWeightWindow exps total pos)
    (weights : alloc.vec.Vec I64) (s : Usize)
    (hs : s.val ≤ pos.val)
    (hinv : AttnWeightLoopInv exps total pos weights s true none) :
    WP.spec (attn.KvState.attn_head_loop2.body pos exps total weights s true none)
      (fun cf =>
        ∃ (weights2 : alloc.vec.Vec Std.I64) (s2 : Usize),
          cf = ControlFlow.cont (weights2, s2, true, none) ∧
          AttnWeightLoopInv exps total pos weights2 s2 true none ∧
          s2.val = s.val + 1) := by
  obtain ⟨hlen, hcap, htot, hjw⟩ := hwin
  obtain ⟨_, _, _, hwlen, hwval⟩ := hinv
  obtain ⟨hnn, hadd, hfl, hfh⟩ := hjw s.val hs
  have hslt : s.val < exps.val.length := by rw [hlen]; omega
  have hspos : s ≤ pos := (UScalar.le_equiv s pos).2 hs
  have hemax : weights.val.length < Usize.max := by rw [hwlen]; omega
  have hsmax : s.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]; omega
  obtain ⟨e, he_eq, hev⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_usize_spec exps s hslt)
  have heq : exps.val.getD s.val 0#i64 = e := by rw [getD_eq_get hslt]; exact hev.symm
  obtain ⟨w, hw_eq, hwv⟩ := weight_at_spec e total (by rw [← heq]; exact hnn) htot
    (by rw [← heq]; exact hadd)
    (by rw [← heq]; exact ⟨hfl, hfh⟩)
  obtain ⟨weights2, hpush, hpushv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec weights w hemax)
  obtain ⟨s2, hadd1, hs2⟩ := WP.spec_imp_exists (Usize.add_spec (x := s) (y := 1#usize) hsmax)
  have hs2' : s2.val = s.val + 1 := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hs2, h1]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (weights2, s2, true, none), ?_, ?_⟩
  · rw [attn.KvState.attn_head_loop2.body]
    simp only [if_pos hspos, if_true, alloc.vec.Vec.index_slice_index,
      he_eq, bind_tc_ok]
    try dsimp only
    rw [hw_eq]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hpush, hadd1]
    simp only [bind_tc_ok]
    rfl
  · refine ⟨weights2, s2, rfl, ⟨rfl, rfl, by omega, ?_, ?_⟩, hs2'⟩
    · rw [hpushv]; simp [hwlen, hs2']
    · intro j hj
      rw [hpushv] at hj ⊢
      by_cases hjw' : j < weights.val.length
      · rw [List.getD_append weights.val [w] 0#i64 j hjw', hwval j hjw']
      · have hje : j = weights.val.length := by
          simp only [List.length_append, List.length_cons, List.length_nil] at hj
          omega
        subst hje
        rw [List.getD_append_right weights.val [w] 0#i64 weights.val.length (le_refl _)]
        simp only [Nat.sub_self, List.getD_cons_zero, hwlen, attnWeightAt, heq]
        exact hwv

/-- **Weight-loop content.** On `AttnWeightWindow`, `attn_head_loop2` returns, clear, weights
exactly `attnWeightAt` at every causal index up to `pos`. -/
theorem attn_head_loop2_spec (exps : alloc.vec.Vec I64) (total : I128) (pos slots : Usize)
    (hwin : AttnWeightWindow exps total pos) :
    ∃ weights : alloc.vec.Vec I64,
      attn.KvState.attn_head_loop2 pos exps total
          (alloc.vec.Vec.with_capacity Std.I64 slots) 0#usize true none
        = ok (weights, none) ∧
      weights.val.length = pos.val + 1 ∧
      ∀ j, j ≤ pos.val →
        (weights.val.getD j 0#i64).val =
          divRoundNearest ((exps.val.getD j 0#i64).val * (2 : ℤ) ^ (32 : ℕ)) total.val := by
  have hinv0 : AttnWeightLoopInv exps total pos
      (alloc.vec.Vec.with_capacity Std.I64 slots) 0#usize true none := by
    refine ⟨rfl, rfl,
      by have h0 : (0#usize).val = 0 := by norm_num
         omega, ?_, ?_⟩
    · show (alloc.vec.Vec.with_capacity Std.I64 slots).val.length = (0#usize).val
      have h0 : (0#usize).val = 0 := by norm_num
      simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val, h0]
    · intro j hj
      simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val] at hj
  have hspec : WP.spec
      (loop (fun (s : (alloc.vec.Vec Std.I64) × Std.Usize × Bool × (Option attn.AttnError)) =>
          attn.KvState.attn_head_loop2.body pos exps total s.1 s.2.1 s.2.2.1 s.2.2.2)
        (alloc.vec.Vec.with_capacity Std.I64 slots, (0#usize : Std.Usize), true,
          (none : Option attn.AttnError)))
      (fun b : (alloc.vec.Vec Std.I64) × (Option attn.AttnError) =>
        ∃ weights : alloc.vec.Vec I64, b = (weights, none) ∧
          weights.val.length = pos.val + 1 ∧
          ∀ j, j ≤ pos.val →
            (weights.val.getD j 0#i64).val = attnWeightAt exps.val total j) := by
    apply loop.spec_decr_nat
      (measure := fun s : (alloc.vec.Vec Std.I64) × Std.Usize × Bool × (Option attn.AttnError) =>
        pos.val + 1 - s.2.1.val)
      (inv := fun s => AttnWeightLoopInv exps total pos s.1 s.2.1 s.2.2.1 s.2.2.2)
      (post := fun b : (alloc.vec.Vec Std.I64) × (Option attn.AttnError) =>
        ∃ weights : alloc.vec.Vec I64, b = (weights, none) ∧
          weights.val.length = pos.val + 1 ∧
          ∀ j, j ≤ pos.val →
            (weights.val.getD j 0#i64).val = attnWeightAt exps.val total j)
    · intro s hs
      rcases s with ⟨weights, s1, ok1, err⟩
      obtain ⟨hok, herr, hs1le, hlen, hval⟩ := hs
      dsimp only at hok herr hs1le hlen hval
      subst hok herr
      by_cases hlt : s1.val ≤ pos.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (attn_head_loop2_body_step exps total pos hwin weights s1 hlt
            ⟨rfl, rfl, hs1le, hlen, hval⟩)
        obtain ⟨weights2, s2, hcfg, hinv2, hs2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        refine ⟨hinv2, ?_⟩
        show pos.val + 1 - s2.val < pos.val + 1 - s1.val
        omega
      · have hnot : ¬ (s1 ≤ pos) := by
          intro hc
          exact hlt ((UScalar.le_equiv s1 pos).1 hc)
        have hbody : attn.KvState.attn_head_loop2.body pos exps total weights s1 true none
            = ok (ControlFlow.done (weights, none)) := by
          rw [attn.KvState.attn_head_loop2.body, if_neg hnot]
        simp only [hbody]
        refine ⟨weights, rfl, ?_, ?_⟩
        · omega
        · intro j hj
          exact hval j (by rw [hlen]; omega)
    · exact hinv0
  obtain ⟨b, hb, weights, hbeq, hlen, hval⟩ := WP.spec_imp_exists hspec
  rw [hbeq] at hb
  refine ⟨weights, ?_, hlen, hval⟩
  unfold attn.KvState.attn_head_loop2
  exact hb

/-! ## Output stage: the per-component accumulation loop -/

/-- `planeRow` at an in-range column is the plane's flat get: the checked-arithmetic index of
`attn.plane_at` (`(s·kv+kvh)·hd+i`) matches `planeRow`'s definition literally. -/
theorem planeRow_getD (plane : List I64) (kv hd kvh s i : ℕ) (hi : i < hd) :
    (planeRow plane kv hd kvh s).getD i 0#i64 = plane.getD ((s * kv + kvh) * hd + i) 0#i64 := by
  have h1 : (planeRow plane kv hd kvh s)[i]? = some (plane.getD ((s * kv + kvh) * hd + i) 0#i64) := by
    simp only [planeRow, List.getElem?_map, List.getElem?_range hi, Option.map_some]
  rw [List.getD_eq_getElem?_getD, h1, Option.getD_some]

/-- `valueList` at a causal index is the plane's flat get: the range-index identity tying
`valueList`/`planeRow` to the plane contents (`Numeric.EngineAttnRefinement`'s residual item 1,
narrowed to the value column consumed here). -/
theorem valueList_getD (plane : List I64) (kv hd kvh pos i s : ℕ) (hi : i < hd) (hs : s ≤ pos) :
    (valueList plane kv hd kvh pos i).getD s 0 =
      (plane.getD ((s * kv + kvh) * hd + i) 0#i64).val := by
  have hlt : s < pos + 1 := by omega
  have h1 : (valueList plane kv hd kvh pos i)[s]? =
      some ((plane.getD ((s * kv + kvh) * hd + i) 0#i64).val) := by
    simp only [valueList, List.getElem?_map, List.getElem?_range hlt, Option.map_some]
    rw [planeRow_getD plane kv hd kvh s i hi]
  rw [List.getD_eq_getElem?_getD, h1, Option.getD_some]

/-- The value-loop window: `weights` has exactly `pos+1` entries, the plane range-index identity
holds at each causal `s ≤ pos` (`attn.plane_at` succeeds, in bounds), the value magnitude is the
`attn_domain_v2` window (`|v_{j,i}| ≤ 2^62`), and `pos+1 ≤ 2^30` (also `attn_domain_v2`). The
weight-magnitude bound `|W_s| ≤ 2^33` is the gap the module header names: no `attn_domain_v2`
conjunct supplies it. -/
def AttnValueWindow (vplane weights : Slice I64) (kv kvh hd pos i : Usize) : Prop :=
  weights.val.length = pos.val + 1 ∧
  pos.val + 1 ≤ (2 : ℕ) ^ 30 ∧
  (∀ s : Usize, s.val ≤ pos.val →
    ∃ idx : Usize, attn.plane_at kv kvh hd s i = ok (core.result.Result.Ok idx) ∧
      idx.val = (s.val * kv.val + kvh.val) * hd.val + i.val ∧ idx.val < vplane.val.length) ∧
  (∀ s, s ≤ pos.val →
    |(vplane.val.getD ((s * kv.val + kvh.val) * hd.val + i.val) 0#i64).val| ≤ (2 : ℤ) ^ (62 : ℕ)) ∧
  (∀ s, s ≤ pos.val → |(weights.val.getD s 0#i64).val| ≤ (2 : ℤ) ^ (33 : ℕ))

/-- Value-loop invariant: clear flag/error, `s` within one of `pos`, `acc` the exact partial sum of
the first `s` weight/value products. -/
def AttnValueLoopInv (vplane weights : Slice I64) (kv kvh hd pos i : Usize)
    (acc : I128) (s : Usize) (ok1 : Bool) (err : Option attn.AttnError) : Prop :=
  ok1 = true ∧ err = none ∧ s.val ≤ pos.val + 1 ∧
  acc.val = ∑ s' ∈ Finset.range s.val, (weights.val.getD s' 0#i64).val *
    (vplane.val.getD ((s' * kv.val + kvh.val) * hd.val + i.val) 0#i64).val

/-- **One value-loop step.** With `s ≤ pos` and the invariant holding, the body reads the plane
entry through `attn.plane_at` (the window's range-index identity), the weight at `s`, multiplies
and accumulates, checked at `i128`, and continues at `s + 1` with the invariant restored. -/
theorem attn_value_component_loop_body_step (vplane weights : Slice I64) (kv kvh hd pos i : Usize)
    (hwin : AttnValueWindow vplane weights kv kvh hd pos i)
    (acc : I128) (s : Usize) (hs : s.val ≤ pos.val)
    (hinv : AttnValueLoopInv vplane weights kv kvh hd pos i acc s true none) :
    WP.spec (attn.value_component_loop.body vplane weights kv kvh hd pos i acc s true none)
      (fun cf =>
        ∃ (acc2 : I128) (s2 : Usize),
          cf = ControlFlow.cont (acc2, s2, true, none) ∧
          AttnValueLoopInv vplane weights kv kvh hd pos i acc2 s2 true none ∧
          s2.val = s.val + 1) := by
  obtain ⟨hwlen, hnpos, hidx, hvb, hwb⟩ := hwin
  obtain ⟨_, _, hsle, hacc⟩ := hinv
  obtain ⟨idx, hplane, hidxeq, hidxlt⟩ := hidx s hs
  have hwslt : s.val < weights.val.length := by rw [hwlen]; omega
  have hspos : s ≤ pos := (UScalar.le_equiv s pos).2 hs
  have hgetv : vplane[idx]? = some (vplane.val.getD idx.val 0#i64) := by
    rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hidxlt, getD_eq_get hidxlt]
  have hgetw : Slice.index_usize weights s = ok (weights.val.getD s.val 0#i64) := by
    unfold Slice.index_usize
    rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hwslt, getD_eq_get hwslt]
  have he1 : (core.convert.num.FromI128I64.from (weights.val.getD s.val 0#i64)).val
      = (weights.val.getD s.val 0#i64).val := core.convert.num.FromI128I64.from_val_eq _
  have he2 : (core.convert.num.FromI128I64.from (vplane.val.getD idx.val 0#i64)).val
      = (vplane.val.getD idx.val 0#i64).val := core.convert.num.FromI128I64.from_val_eq _
  have hvbnd0 := hvb s.val hs
  rw [← hidxeq] at hvbnd0
  have hwbnd := hwb s.val hs
  have hI128min : I128.min = -(2 : ℤ) ^ 127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ 127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  have hprodfit : |(weights.val.getD s.val 0#i64).val * (vplane.val.getD idx.val 0#i64).val|
      ≤ (2 : ℤ) ^ 95 := by
    calc |(weights.val.getD s.val 0#i64).val * (vplane.val.getD idx.val 0#i64).val|
        = |(weights.val.getD s.val 0#i64).val| * |(vplane.val.getD idx.val 0#i64).val| :=
          abs_mul _ _
      _ ≤ (2 : ℤ) ^ 33 * (2 : ℤ) ^ 62 := mul_le_mul hwbnd hvbnd0 (abs_nonneg _) (by positivity)
      _ = (2 : ℤ) ^ 95 := by norm_num
  obtain ⟨prod, hprod_eq, hprodv⟩ := I128_checked_mul_some
    (core.convert.num.FromI128I64.from (weights.val.getD s.val 0#i64))
    (core.convert.num.FromI128I64.from (vplane.val.getD idx.val 0#i64))
    (by rw [he1, he2, hI128min]; have := (abs_le.mp hprodfit).1; linarith)
    (by rw [he1, he2, hI128max]; have := (abs_le.mp hprodfit).2; linarith)
  have hterm_eq : attn.value_term vplane weights kv kvh hd s i
      = ok (core.result.Result.Ok prod) := by
    rw [attn.value_term, hplane]
    simp only [bind_tc_ok]
    try dsimp only
    rw [result_branch_ok]
    simp only [bind_tc_ok]
    try dsimp only
    simp only [core.slice.Slice.get, core.slice.index.Usize.get]
    rw [hgetv]
    simp only [bind_tc_ok]
    try dsimp only
    rw [core.option.Option.ok_or_some]
    simp only [bind_tc_ok]
    try dsimp only
    rw [result_branch_ok]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hgetw]
    simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hprod_eq]
    try dsimp only
    rw [core.option.Option.ok_or_some]
  have hterm_val : prod.val =
      (weights.val.getD s.val 0#i64).val * (vplane.val.getD idx.val 0#i64).val := by
    rw [hprodv, he1, he2]
  have hsum_bound : |acc.val| ≤ (s.val : ℤ) * (2 : ℤ) ^ 95 := by
    rw [hacc]
    calc |∑ s' ∈ Finset.range s.val, (weights.val.getD s' 0#i64).val *
        (vplane.val.getD ((s' * kv.val + kvh.val) * hd.val + i.val) 0#i64).val|
        ≤ ∑ s' ∈ Finset.range s.val, |(weights.val.getD s' 0#i64).val *
            (vplane.val.getD ((s' * kv.val + kvh.val) * hd.val + i.val) 0#i64).val| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _s' ∈ Finset.range s.val, (2 : ℤ) ^ 95 := by
          apply Finset.sum_le_sum
          intro s' hs'
          simp only [Finset.mem_range] at hs'
          have hs'le : s' ≤ pos.val := by omega
          have hvbnd' := hvb s' hs'le
          have hwbnd' := hwb s' hs'le
          calc |(weights.val.getD s' 0#i64).val *
              (vplane.val.getD ((s' * kv.val + kvh.val) * hd.val + i.val) 0#i64).val|
              = |(weights.val.getD s' 0#i64).val| *
                |(vplane.val.getD ((s' * kv.val + kvh.val) * hd.val + i.val) 0#i64).val| :=
                abs_mul _ _
            _ ≤ (2 : ℤ) ^ 33 * (2 : ℤ) ^ 62 :=
                mul_le_mul hwbnd' hvbnd' (abs_nonneg _) (by positivity)
            _ = (2 : ℤ) ^ 95 := by norm_num
      _ = (s.val : ℤ) * (2 : ℤ) ^ 95 := by rw [Finset.sum_const, Finset.card_range]; ring
  have hnbound : (s.val : ℤ) ≤ (2 : ℤ) ^ 30 := by
    have h30 : pos.val + 1 ≤ (2 : ℕ) ^ 30 := hnpos
    have : (s.val : ℤ) ≤ ((2 : ℕ) ^ 30 : ℤ) - 1 := by push_cast; omega
    push_cast at this
    linarith
  have haddfit : |acc.val + prod.val| ≤ (2 : ℤ) ^ 30 * (2 : ℤ) ^ 95 := by
    have h1 : |acc.val + prod.val| ≤ |acc.val| + |prod.val| := abs_add_le _ _
    have h3 : |prod.val| ≤ (2 : ℤ) ^ 95 := by rw [hterm_val]; exact hprodfit
    nlinarith [hsum_bound, hnbound]
  obtain ⟨a, hadd_eq, hav⟩ := I128_checked_add_some acc prod
    (by rw [hI128min]; have := (abs_le.mp haddfit).1; nlinarith)
    (by rw [hI128max]; have := (abs_le.mp haddfit).2; nlinarith)
  have hsmax : s.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have h30 := usize_max_ge_2_pow_30
    omega
  obtain ⟨s2, hadd1, hs2⟩ := WP.spec_imp_exists (Usize.add_spec (x := s) (y := 1#usize) hsmax)
  have hs2' : s2.val = s.val + 1 := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hs2, h1]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (a, s2, true, none), ?_, ?_⟩
  · rw [attn.value_component_loop.body]
    simp only [if_pos hspos, if_true, hterm_eq, bind_tc_ok]
    try dsimp only
    rw [hadd_eq]
    try dsimp only
    rw [hadd1]
    simp only [lift, bind_tc_ok]
    rfl
  · refine ⟨a, s2, rfl, ⟨rfl, rfl, by omega, ?_⟩, hs2'⟩
    show a.val = ∑ s' ∈ Finset.range s2.val, (weights.val.getD s' 0#i64).val *
        (vplane.val.getD ((s' * kv.val + kvh.val) * hd.val + i.val) 0#i64).val
    rw [hs2', Finset.sum_range_succ, ← hacc, hav, hterm_val, hidxeq]

/-- **Value-loop content.** On `AttnValueWindow`, `attn.value_component_loop` returns, clear, the
exact accumulation `Σ_{s≤pos} W_s·V_{s,i}`. -/
theorem attn_value_component_loop_spec (vplane weights : Slice I64) (kv kvh hd pos i : Usize)
    (hwin : AttnValueWindow vplane weights kv kvh hd pos i) :
    ∃ acc : I128,
      attn.value_component_loop vplane weights kv kvh hd pos i 0#i128 0#usize true none
        = ok (acc, none) ∧
      acc.val = ∑ s ∈ Finset.range (pos.val + 1), (weights.val.getD s 0#i64).val *
        (vplane.val.getD ((s * kv.val + kvh.val) * hd.val + i.val) 0#i64).val := by
  have hinv0 : AttnValueLoopInv vplane weights kv kvh hd pos i 0#i128 0#usize true none := by
    refine ⟨rfl, rfl,
      by have h0 : (0#usize).val = 0 := by norm_num
         omega, ?_⟩
    have h0 : (0#usize).val = 0 := by norm_num
    rw [h0]
    simp
  have hspec : WP.spec
      (loop (fun (st : Std.I128 × Std.Usize × Bool × (Option attn.AttnError)) =>
          attn.value_component_loop.body vplane weights kv kvh hd pos i st.1 st.2.1 st.2.2.1
            st.2.2.2)
        ((0#i128 : Std.I128), (0#usize : Std.Usize), true, (none : Option attn.AttnError)))
      (fun b : Std.I128 × (Option attn.AttnError) =>
        ∃ acc : I128, b = (acc, none) ∧
          acc.val = ∑ s ∈ Finset.range (pos.val + 1), (weights.val.getD s 0#i64).val *
            (vplane.val.getD ((s * kv.val + kvh.val) * hd.val + i.val) 0#i64).val) := by
    apply loop.spec_decr_nat
      (measure := fun st : Std.I128 × Std.Usize × Bool × (Option attn.AttnError) =>
        pos.val + 1 - st.2.1.val)
      (inv := fun st => AttnValueLoopInv vplane weights kv kvh hd pos i st.1 st.2.1 st.2.2.1
        st.2.2.2)
      (post := fun b : Std.I128 × (Option attn.AttnError) =>
        ∃ acc : I128, b = (acc, none) ∧
          acc.val = ∑ s ∈ Finset.range (pos.val + 1), (weights.val.getD s 0#i64).val *
            (vplane.val.getD ((s * kv.val + kvh.val) * hd.val + i.val) 0#i64).val)
    · intro st hst
      rcases st with ⟨acc, s1, ok1, err⟩
      obtain ⟨hok, herr, hs1le, hacc⟩ := hst
      dsimp only at hok herr hs1le hacc
      subst hok herr
      by_cases hlt : s1.val ≤ pos.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (attn_value_component_loop_body_step vplane weights kv kvh hd pos i hwin acc s1 hlt
            ⟨rfl, rfl, hs1le, hacc⟩)
        obtain ⟨acc2, s2, hcfg, hinv2, hs2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        refine ⟨hinv2, ?_⟩
        show pos.val + 1 - s2.val < pos.val + 1 - s1.val
        omega
      · have hnot : ¬ (s1 ≤ pos) := by
          intro hc; exact hlt ((UScalar.le_equiv s1 pos).1 hc)
        have hbody : attn.value_component_loop.body vplane weights kv kvh hd pos i acc s1 true
            none = ok (ControlFlow.done (acc, none)) := by
          rw [attn.value_component_loop.body, if_neg hnot]
        simp only [hbody]
        refine ⟨acc, rfl, ?_⟩
        have hs1eq : s1.val = pos.val + 1 := by omega
        rw [hs1eq] at hacc
        exact hacc
    · exact hinv0
  obtain ⟨b, hb, acc, hbeq, hacc⟩ := WP.spec_imp_exists hspec
  rw [hbeq] at hb
  refine ⟨acc, ?_, hacc⟩
  unfold attn.value_component_loop
  exact hb

/-- **Value-component content.** On `AttnValueWindow` and the output-epilogue narrowing
(`value_component_epilogue`'s own `hlo`/`hhi`/`hfit`, reused unchanged), `value_component` returns
`Ok` and the exact half-biased reduction of `Σ_{s≤pos} W_s·V_{s,i}`, `V` the `valueList` entries at
component `i`. -/
theorem value_component_spec (vplane weights : Slice I64) (kv kvh hd pos i : Usize)
    (hwin : AttnValueWindow vplane weights kv kvh hd pos i)
    (hlo : I128.min ≤ (∑ s ∈ Finset.range (pos.val + 1), (weights.val.getD s 0#i64).val *
        (vplane.val.getD ((s * kv.val + kvh.val) * hd.val + i.val) 0#i64).val) + (2 : ℤ) ^ 31)
    (hhi : (∑ s ∈ Finset.range (pos.val + 1), (weights.val.getD s 0#i64).val *
        (vplane.val.getD ((s * kv.val + kvh.val) * hd.val + i.val) 0#i64).val) + (2 : ℤ) ^ 31
      ≤ I128.max)
    (hfit : -((2 : ℤ) ^ 63) ≤ divRoundNearest (∑ s ∈ Finset.range (pos.val + 1),
        (weights.val.getD s 0#i64).val *
          (vplane.val.getD ((s * kv.val + kvh.val) * hd.val + i.val) 0#i64).val) ((2 : ℤ) ^ 32) ∧
      divRoundNearest (∑ s ∈ Finset.range (pos.val + 1), (weights.val.getD s 0#i64).val *
          (vplane.val.getD ((s * kv.val + kvh.val) * hd.val + i.val) 0#i64).val) ((2 : ℤ) ^ 32)
        ≤ (2 : ℤ) ^ 63 - 1)
    (hidim : i.val < hd.val) :
    ∃ y : I64,
      attn.value_component vplane weights kv kvh hd pos i = ok (core.result.Result.Ok y) ∧
      y.val = divRoundNearest
        (∑ s ∈ Finset.range (pos.val + 1), (weights.val.getD s 0#i64).val *
          ((valueList vplane.val kv.val hd.val kvh.val pos.val i.val).getD s 0)) ((2 : ℤ) ^ 32) := by
  obtain ⟨acc, hloop, haccval⟩ := attn_value_component_loop_spec vplane weights kv kvh hd pos i hwin
  obtain ⟨y, hy_eq, hyval⟩ := value_component_epilogue vplane weights kv kvh hd pos i acc hloop
    (by rw [haccval]; exact hlo) (by rw [haccval]; exact hhi) (by rw [haccval]; exact hfit)
  refine ⟨y, hy_eq, ?_⟩
  rw [hyval, haccval]
  congr 1
  apply Finset.sum_congr rfl
  intro s hs
  simp only [Finset.mem_range] at hs
  rw [valueList_getD vplane.val kv.val hd.val kvh.val pos.val i.val s hidim (by omega)]

/-! ## Output stage: the write-through loop -/

/-- The output slice after `j` written components: entry `idx < j` is `V idx`, the rest unchanged
(the closed-form reference, in the style of `Numeric.RopeHeadContent`'s `rotPrefix`). -/
def outPrefix (out0 : List I64) (V : ℕ → I64) (j : ℕ) : List I64 :=
  out0.mapIdx fun idx a => if idx < j then V idx else a

theorem outPrefix_length (out0 : List I64) (V : ℕ → I64) (j : ℕ) :
    (outPrefix out0 V j).length = out0.length := by simp [outPrefix]

theorem outPrefix_zero (out0 : List I64) (V : ℕ → I64) :
    outPrefix out0 V 0 = out0 := by
  apply List.ext_getElem
  · simp [outPrefix]
  · intro n h1 _
    simp only [outPrefix, List.getElem_mapIdx]
    rw [if_neg (by omega)]

/-- **One output-loop iteration.** For `j < out0.length`, iteration `j` writes `V j` at `j`. -/
theorem outPrefix_succ (out0 : List I64) (V : ℕ → I64) (j : ℕ) (_hj : j < out0.length) :
    outPrefix out0 V (j + 1) = (outPrefix out0 V j).set j (V j) := by
  apply List.ext_getElem
  · simp [outPrefix]
  · intro n h1 _
    simp only [outPrefix, List.getElem_mapIdx, List.getElem_set]
    by_cases hjn : j = n
    · subst hjn; rw [if_pos rfl, if_pos (by omega)]
    · rw [if_neg hjn]
      by_cases hlt2 : n < j
      · rw [if_pos (by omega), if_pos hlt2]
      · rw [if_neg (by omega), if_neg hlt2]

/-- Output-loop invariant: clear flag/error, `j` within one of `head_dim`, `out` denoting
`outPrefix out0 V j`. -/
def AttnOutLoopInv (out0 : Slice I64) (V : ℕ → I64) (hd : Usize)
    (out : Slice I64) (j : Usize) (ok1 : Bool) (err : Option attn.AttnError) : Prop :=
  ok1 = true ∧ err = none ∧ j.val ≤ hd.val ∧
  out.val.length = out0.val.length ∧
  out.val = outPrefix out0.val V j.val

/-- **One output-loop step.** With `j < head_dim` and the invariant holding, the body computes
`value_component` at `j` (via `hval`, an arbitrary per-component result function — the residual
composition through `value_component_spec` is the caller's), writes it at `j` and continues at
`j + 1` with the invariant restored. -/
theorem attn_head_loop3_body_step (self : attn.KvState) (kv pos kvh : Usize)
    (weights vplane : alloc.vec.Vec I64) (out0 : Slice I64) (V : ℕ → I64)
    (hout0len : out0.val.length = self.head_dim.val)
    (hval : ∀ j : Usize, j.val < self.head_dim.val →
      attn.value_component (alloc.vec.Vec.deref vplane) (alloc.vec.Vec.deref weights)
          kv kvh self.head_dim pos j
        = ok (core.result.Result.Ok (V j.val)))
    (out : Slice I64) (j : Usize) (hj : j.val < self.head_dim.val)
    (hinv : AttnOutLoopInv out0 V self.head_dim out j true none) :
    WP.spec (attn.KvState.attn_head_loop3.body self kv pos kvh weights vplane out j true none)
      (fun cf =>
        ∃ (out2 : Slice I64) (j2 : Usize),
          cf = ControlFlow.cont (out2, j2, true, none) ∧
          AttnOutLoopInv out0 V self.head_dim out2 j2 true none ∧
          j2.val = j.val + 1) := by
  obtain ⟨_, _, hjle, hlen, hmap⟩ := hinv
  have hjlt : j < self.head_dim := (UScalar.lt_equiv j self.head_dim).2 hj
  have hvj := hval j hj
  have hjout : j.val < out.val.length := by rw [hlen, hout0len]; exact hj
  obtain ⟨out2, hupd, hupdv⟩ := WP.spec_imp_exists (Slice.update_spec out j (V j.val) hjout)
  have hjmax : j.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hbw := UScalar.hBounds self.head_dim
    rw [UScalarTy.Usize_numBits_eq] at hbw
    have hm : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    omega
  obtain ⟨j2, hadd1, hj2⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := 1#usize) hjmax)
  have hj2' : j2.val = j.val + 1 := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hj2, h1]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out2, j2, true, none), ?_, ?_⟩
  · rw [attn.KvState.attn_head_loop3.body]
    simp only [if_pos hjlt, if_true, hvj, bind_tc_ok]
    try dsimp only
    rw [hupd]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hadd1]
    simp only [bind_tc_ok]
    rfl
  · have hjout0 : j.val < out0.val.length := by rw [hout0len]; exact hj
    have hupdv' : out2.val = out.val.set j.val (V j.val) := by
      rw [hupdv]; exact Slice.set_val_eq out j (V j.val)
    refine ⟨out2, j2, rfl, ⟨rfl, rfl, by omega, ?_, ?_⟩, hj2'⟩
    · rw [hupdv']
      simp [hlen]
    · rw [hupdv', hmap, hj2', outPrefix_succ out0.val V j.val hjout0]

/-- **Output-loop content.** On the closing `value_component` results `V`, `attn_head_loop3`
returns, clear, the output slice `outPrefix out0 V head_dim`: every component `j < head_dim` is
exactly `V j`. -/
theorem attn_head_loop3_spec (self : attn.KvState) (kv pos kvh : Usize)
    (weights vplane : alloc.vec.Vec I64) (out0 : Slice I64) (V : ℕ → I64)
    (hout0len : out0.val.length = self.head_dim.val)
    (hval : ∀ j : Usize, j.val < self.head_dim.val →
      attn.value_component (alloc.vec.Vec.deref vplane) (alloc.vec.Vec.deref weights)
          kv kvh self.head_dim pos j
        = ok (core.result.Result.Ok (V j.val))) :
    ∃ out' : Slice I64,
      attn.KvState.attn_head_loop3 self kv pos out0 kvh weights vplane 0#usize true none
        = ok (out', none) ∧
      out'.val.length = out0.val.length ∧
      ∀ j, j < self.head_dim.val → out'.val.getD j 0#i64 = V j := by
  have hinv0 : AttnOutLoopInv out0 V self.head_dim out0 0#usize true none := by
    refine ⟨rfl, rfl,
      by have h0 : (0#usize).val = 0 := by norm_num
         omega, rfl, ?_⟩
    have h0 : (0#usize).val = 0 := by norm_num
    rw [h0]
    exact (outPrefix_zero out0.val V).symm
  have hspec : WP.spec
      (loop (fun (st : Slice Std.I64 × Std.Usize × Bool × (Option attn.AttnError)) =>
          attn.KvState.attn_head_loop3.body self kv pos kvh weights vplane st.1 st.2.1 st.2.2.1
            st.2.2.2)
        (out0, (0#usize : Std.Usize), true, (none : Option attn.AttnError)))
      (fun b : Slice Std.I64 × (Option attn.AttnError) =>
        ∃ out' : Slice I64, b = (out', none) ∧
          out'.val.length = out0.val.length ∧
          ∀ j, j < self.head_dim.val → out'.val.getD j 0#i64 = V j) := by
    apply loop.spec_decr_nat
      (measure := fun st : Slice Std.I64 × Std.Usize × Bool × (Option attn.AttnError) =>
        self.head_dim.val - st.2.1.val)
      (inv := fun st => AttnOutLoopInv out0 V self.head_dim st.1 st.2.1 st.2.2.1 st.2.2.2)
      (post := fun b : Slice Std.I64 × (Option attn.AttnError) =>
        ∃ out' : Slice I64, b = (out', none) ∧
          out'.val.length = out0.val.length ∧
          ∀ j, j < self.head_dim.val → out'.val.getD j 0#i64 = V j)
    · intro st hst
      rcases st with ⟨out, j1, ok1, err⟩
      obtain ⟨hok, herr, hjle, hlen, hmap⟩ := hst
      dsimp only at hok herr hjle hlen hmap
      subst hok herr
      by_cases hlt : j1.val < self.head_dim.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (attn_head_loop3_body_step self kv pos kvh weights vplane out0 V hout0len hval out j1
            hlt ⟨rfl, rfl, hjle, hlen, hmap⟩)
        obtain ⟨out2, j2, hcfg, hinv2, hj2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        refine ⟨hinv2, ?_⟩
        show self.head_dim.val - j2.val < self.head_dim.val - j1.val
        omega
      · have hnot : ¬ (j1 < self.head_dim) := by
          intro hc; exact hlt ((UScalar.lt_equiv j1 self.head_dim).1 hc)
        have hbody : attn.KvState.attn_head_loop3.body self kv pos kvh weights vplane out j1 true
            none = ok (ControlFlow.done (out, none)) := by
          rw [attn.KvState.attn_head_loop3.body, if_neg hnot]
        simp only [hbody]
        refine ⟨out, rfl, hlen, ?_⟩
        intro j hj
        have hj1eq : j1.val = self.head_dim.val := by omega
        rw [hj1eq] at hmap
        have hjlt2 : j < out0.val.length := by rw [hout0len]; exact hj
        rw [hmap]
        simp only [outPrefix, List.getD_eq_getElem?_getD, List.getElem?_mapIdx,
          List.getElem?_eq_getElem hjlt2, Option.map_some, Option.getD_some]
        rw [if_pos hj]
    · exact hinv0
  obtain ⟨b, hb, out', hbeq, hlen, hval'⟩ := WP.spec_imp_exists hspec
  rw [hbeq] at hb
  refine ⟨out', ?_, hlen, hval'⟩
  unfold attn.KvState.attn_head_loop3
  exact hb

end

end apeiron_word
