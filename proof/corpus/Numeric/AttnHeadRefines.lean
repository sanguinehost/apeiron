/-
End-to-end refinement of the extracted single-token attention head
(`APEIRON-ALGEBRA-G3-ATTN-W3-T1d`).

`Numeric.AttnHeadWindowsScore` and `Numeric.AttnHeadWindowsValue` discharge the four window
predicates of the extracted loops from `attn_domain_v2`; `Numeric.AttnScoreStage` bounds the
realised Q32.32 score against `idealScoreList`. This module assembles them through
`word_attn_head_ok` (`Numeric.EngineAttnRefinement`) and `attn_stage_compose_closed`
(`Numeric.AttnSoftmaxPerturb`) into the corrected end-to-end target:

* `attn_head_refines_v2_proved`: `attn_head_refines_v2` — on `attn_domain_v2` the body returns
  `Ok` and every output component is within `attnErrBoundV2` of the dense real softmax attention
  of the ideal scores, at the score-stage error `scoreErrV2`.

No hypothesis beyond `attn_domain_v2` is introduced.

Trust boundary: reaches the extracted `word_attn_head`, `attn_head_loop0..3`, `attn.value_component`
and `attn.plane_at` of `proof/ApeironWord.lean`, the plane/score/value denotations of
`Numeric.EngineAttnRefinement`, and the proved stage composition of `Numeric.AttnSoftmaxPerturb`.
Does not cover rustc/LLVM codegen of `src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the
`unsafe` container.
-/
import Numeric.AttnHeadWindowsScore
import Numeric.AttnHeadWindowsValue
import Numeric.AttnScoreStage

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## List helpers -/

/-- A list is the `getD` window of its own range. -/
private theorem map_getD_range_eq {α : Type} (l : List α) (d : α) :
    (List.range l.length).map (fun j => l.getD j d) = l := by
  apply List.ext_getElem
  · simp [List.length_map]
  · intro k h1 h2
    rw [List.getElem_map, List.getElem_range (by simpa using h1), List.getD_eq_getElem l d h2]

/-- `toRealList` at a cons. -/
private theorem toRealList_cons (a : ℤ) (t : List ℤ) :
    toRealList (a :: t) = ((a : ℝ) / q32) :: toRealList t := by
  simp only [toRealList, List.map_eq_flatMap, List.pure_def]
  rw [show ((a :: t) >>= (fun b : ℤ => ([((b : ℝ))] : List ℝ)))
        = List.flatMap (fun b : ℤ => ([((b : ℝ))] : List ℝ)) (a :: t) from rfl]
  rw [List.flatMap_cons, List.flatMap_append]
  simp

/-- `toRealList` is `getD` pointwise. -/
private theorem toRealList_getD (l : List ℤ) (j : ℕ) :
    (toRealList l).getD j 0 = ((l.getD j 0 : ℤ) : ℝ) / q32 := by
  induction l generalizing j with
  | nil => simp [toRealList]
  | cons a t ih =>
    cases j with
    | zero => simp [toRealList_cons]
    | succ j => rw [List.getD_cons_succ, toRealList_cons, List.getD_cons_succ, ih]

/-- `toRealList` preserves length. -/
private theorem toRealList_length (l : List ℤ) : (toRealList l).length = l.length := by
  induction l with
  | nil => simp [toRealList]
  | cons a t ih => rw [toRealList_cons, List.length_cons, ih, List.length_cons]

/-- The `getD` window of `toRealList`. -/
private theorem map_range_getD_toRealList (l : List ℤ) (v : ℕ → ℤ)
    (hv : ∀ j, v j = l.getD j 0) :
    (List.range l.length).map (fun j => (v j : ℝ) / q32) = toRealList l := by
  calc (List.range l.length).map (fun j => (v j : ℝ) / q32)
      = (List.range l.length).map (fun j => ((l.getD j 0 : ℤ) : ℝ) / q32) := by
          apply List.map_congr_left
          intro j _
          rw [hv j]
    _ = (List.range l.length).map (fun j => (toRealList l).getD j 0) := by
          apply List.map_congr_left
          intro j _
          rw [toRealList_getD]
    _ = toRealList l := by
          rw [← toRealList_length]
          exact map_getD_range_eq (toRealList l) 0

/-! ## The end-to-end target -/

/-- **Corrected end-to-end refinement (proved).** `attn_head_refines_v2` holds: on `attn_domain_v2`
the extracted `word_attn_head` returns `Ok` and each output component is within `attnErrBoundV2` of
the dense real softmax attention of the ideal scores, at score error `scoreErrV2`. -/
theorem attn_head_refines_v2_proved : attn_head_refines_v2 := by
  intro state layer q_head pos q out kplane vplane h
  have hD := h
  have h0 := h
  obtain ⟨len, slots, group, kvh, hlayer, hhead, hq', hout', hlen, hposlen, hadd, _hslots,
    hgroup, hkvh, hkplane, hvplane, hkvhKvh, _hkvhlt, hposmax⟩ :=
    attn_v2_prefix state layer q_head pos q out kplane vplane hD
  obtain ⟨hadom, _hocc, hscale, _hk, _hv, _hklen, _hvlen, _hrowdot, _hscorewin, _hvalwin,
    hpos30⟩ := h0
  unfold attn_domain at hadom
  obtain ⟨_, _, _, hhdim, _, _, _, _, _, houtlen, _⟩ := hadom
  obtain ⟨r, hr, hwr⟩ := hscale
  have hkvhN : kvh.val = q_head.val / (state.n_head.val / state.n_head_kv.val) := by
    simpa [attnKvh] using hkvhKvh
  -- score loop
  have hdom0 := attn_v2_loop0_domain state layer q_head pos q out kplane vplane kvh hD hkvhKvh
  obtain ⟨scores, mx, hloop0, hslen, hpt, hmx, hmax0⟩ :=
    attn_loop0_ok state kplane q pos kvh slots hposmax hdom0
  have hmax0' : ∃ j, j < scores.val.length ∧ scores.val.getD j 0#i64 = mx := by
    obtain ⟨j, hj, hjeq⟩ := hmax0
    exact ⟨j, hj, hjeq⟩
  -- numerator loop
  have hdom1 := attn_v2_loop1_domain state layer q_head pos q out kplane vplane hD scores mx hslen
    (by intro j hj; simpa only [hkvhN] using hpt j hj) hmx hmax0'
  obtain ⟨exps, total, hloop1, hexpslen, hcase, _htot0, htoteq⟩ :=
    attn_loop1_ok pos scores mx slots hslen hposmax hdom1
  -- stage facts
  obtain ⟨hmaxf, hnumf, hTf, hnonposf⟩ :=
    attn_loop_stage_facts pos scores exps mx total hslen hexpslen hmx hmax0' hcase htoteq
  -- weight window
  let sh : ℕ → ℤ := fun j => (scores.val.getD j 0#i64).val - mx.val
  have hcaseW : ∀ j, j ≤ pos.val → sh j ≤ 0 ∧ attnNumeratorCase (sh j)
      (exps.val.getD j 0#i64).val := by
    intro j hj
    have hmem : j ∈ Finset.range (pos.val + 1) := Finset.mem_range.mpr (by omega)
    exact ⟨by simp only [sh]; exact hnonposf j hmem, hnumf j hmem⟩
  have hzeroW : ∃ j, j ≤ pos.val ∧ sh j = 0 := by
    obtain ⟨j, hj, hj0⟩ := hmaxf
    exact ⟨j, by simpa using Finset.mem_range.mp hj, by simp only [sh]; omega⟩
  have htw : total.val = ∑ j ∈ Finset.range (pos.val + 1),
      (exps.val.getD j 0#i64).val := hTf
  have hwin := attn_weight_window_of_numerators exps total pos sh hexpslen hpos30 hcaseW hzeroW htw
  have htotpos : 0 < total.val := hwin.2.2.1
  have htotnle : ¬ (total ≤ 0#i128) := by
    rw [IScalar.le_equiv]
    intro hle
    have h0 : (0#i128).val = 0 := by norm_num
    rw [h0] at hle
    omega
  -- weight loop
  obtain ⟨weights, hloop2, hwlen, hwval⟩ := attn_head_loop2_spec exps total pos slots hwin
  -- weights and their bound
  let W : ℕ → ℤ := fun j => attnWeightAt exps.val total j
  have hWdiv : ∀ j ∈ Finset.range (pos.val + 1),
      W j = divRoundNearest ((exps.val.getD j 0#i64).val * (2 : ℤ) ^ (32 : ℕ)) total.val := by
    intro j _
    simp only [W, attnWeightAt]
  have hEnonneg : ∀ j ∈ Finset.range (pos.val + 1), 0 ≤ (exps.val.getD j 0#i64).val := by
    intro j hj
    have hjle : j ≤ pos.val := by simp only [Finset.mem_range] at hj; omega
    exact (hwin.2.2.2 j hjle).1
  have hWbounds := attn_weight_bounds (pos.val + 1)
    (fun j => (exps.val.getD j 0#i64).val) W total.val hEnonneg htw htotpos hWdiv
  have hW0 : ∀ s ∈ Finset.range (pos.val + 1),
      0 ≤ W s ∧ W s ≤ (2 : ℤ) ^ (32 : ℕ) := hWbounds.1
  have hWsum : ∑ s ∈ Finset.range (pos.val + 1), W s
      ≤ (2 : ℤ) ^ (32 : ℕ) + ((pos.val + 1 : ℕ) : ℤ) := hWbounds.2
  have hWco : ∀ s, s ≤ pos.val → (weights.val.getD s 0#i64).val = W s := by
    intro s hs
    rw [hwval s hs]
    simp only [W, attnWeightAt]
  have hcomponents := attn_value_components state layer q_head pos q out kplane vplane kvh weights
    W hD hkvhN hwlen hWco hW0 hWsum
  -- the per-component output function
  have hbound : ∀ n : ℕ, n < state.head_dim.val → n < 2 ^ UScalarTy.Usize.numBits :=
    fun n hn => lt_of_lt_of_le hn (le_of_lt (UScalar.hBounds state.head_dim))
  have hvalp : ∀ (n : ℕ) (hn : n < state.head_dim.val),
      (Usize.ofNatCore n (hbound n hn)).val < state.head_dim.val :=
    fun n hn => by rw [Usize.ofNatCore_val_eq]; exact hn
  set V : ℕ → I64 := fun n =>
    if hn : n < state.head_dim.val then
      Classical.choose (hcomponents (Usize.ofNatCore n (hbound n hn)) (hvalp n hn))
    else 0#i64 with hVdef
  have hVspec : ∀ j : Usize, j.val < state.head_dim.val →
      attn.value_component (alloc.vec.Vec.deref vplane) (alloc.vec.Vec.deref weights)
        state.n_head_kv kvh state.head_dim pos j = ok (core.result.Result.Ok (V j.val)) := by
    intro j hj
    have hjeq : j = Usize.ofNatCore j.val (hbound j.val hj) :=
      (UScalar.eq_of_val_eq (by rw [Usize.ofNatCore_val_eq])).symm
    have hstep : attn.value_component (alloc.vec.Vec.deref vplane) (alloc.vec.Vec.deref weights)
        state.n_head_kv kvh state.head_dim pos j
        = attn.value_component (alloc.vec.Vec.deref vplane) (alloc.vec.Vec.deref weights)
            state.n_head_kv kvh state.head_dim pos
            (Usize.ofNatCore j.val (hbound j.val hj)) :=
      congrArg (fun i => attn.value_component (alloc.vec.Vec.deref vplane)
        (alloc.vec.Vec.deref weights) state.n_head_kv kvh state.head_dim pos i) hjeq
    have hVj : V j.val = Classical.choose
        (hcomponents (Usize.ofNatCore j.val (hbound j.val hj)) (hvalp j.val hj)) := by
      rw [hVdef]
      simp only [dif_pos hj]
    rw [hstep, hVj]
    exact (Classical.choose_spec
      (hcomponents (Usize.ofNatCore j.val (hbound j.val hj)) (hvalp j.val hj))).1
  -- output loop
  obtain ⟨out1, hloop3, hout1len, hout1val⟩ :=
    attn_head_loop3_spec state state.n_head_kv pos kvh weights vplane out V houtlen hVspec
  refine ⟨out1, ?_, hout1len, ?_⟩
  · exact word_attn_head_ok state layer q_head q out pos hlayer hhead hq' hout' len hlen hposlen
      slots hadd group kvh hgroup hkvh kplane hkplane scores mx hloop0 exps total hloop1 htotnle
      weights hloop2 vplane hvplane out1 hloop3
  · intro i hi
    dsimp only
    rw [← hkvhN]
    set sc : List ℝ := idealScoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
      kvh.val (state.head_dim.val : ℝ) pos.val with hsc
    set vs : List ℤ := valueList vplane.val state.n_head_kv.val state.head_dim.val kvh.val pos.val
      i with hvs
    have hvslen : vs.length = pos.val + 1 := by
      rw [hvs]
      simp only [valueList, List.length_map, List.length_range]
    have hsclen : sc.length = pos.val + 1 := by
      rw [hsc]
      simp only [idealScoreList, List.length_map, List.length_range]
    set Vb : ℝ := listAbsMax (toRealList vs) with hVb
    set δ : ℝ := scoreErrV2 (state.head_dim.val : ℝ) (listAbsMax sc) |toReal state.scale| with hδ
    set x : ℕ → ℝ := fun j => sc.getD j 0 with hx
    set y : ℕ → ℤ := fun j => (scores.val.getD j 0#i64).val with hy
    set E : ℕ → ℤ := fun j => (exps.val.getD j 0#i64).val with hE
    set Vi : ℕ → ℤ := fun j => vs.getD j 0 with hVi
    set accc : ℤ := ∑ j ∈ Finset.range (pos.val + 1), W j * Vi j with haccc
    have hδ' : ∀ j ∈ Finset.range (pos.val + 1), |(y j : ℝ) / q32 - x j| ≤ δ := by
      intro j hj
      have hjle : j ≤ pos.val := by simp only [Finset.mem_range] at hj; omega
      have hjlt : j < scores.val.length := by rw [hslen]; omega
      have hstage := attn_score_stage state.head_dim.val hhdim r state.scale hr hwr kplane.val
        q.val state.n_head_kv.val kvh.val pos.val j hjle
      simp only [hx, hy, hδ, hsc, hpt j hjlt]
      exact hstage
    have hmax' : ∃ j ∈ Finset.range (pos.val + 1), y j = mx.val := by
      obtain ⟨j, hj, hj0⟩ := hmaxf
      exact ⟨j, hj, by simp only [hy]; exact hj0⟩
    have hnum' : ∀ j ∈ Finset.range (pos.val + 1), attnNumeratorCase (y j - mx.val) (E j) := by
      intro j hj
      simpa only [hy, hE] using hnumf j hj
    have hT' : total.val = ∑ j ∈ Finset.range (pos.val + 1), E j := by
      rw [hTf]
    have hW' : ∀ j ∈ Finset.range (pos.val + 1),
        W j = divRoundNearest (E j * (2 : ℤ) ^ (32 : ℕ)) total.val := by
      intro j _
      simp only [W, hE, attnWeightAt]
    have hvit : ∀ j, (Vi j : ℝ) / q32 = (toRealList vs).getD j 0 := by
      intro j
      rw [hVi, toRealList_getD]
    obtain ⟨hVcomp_eq, hVcomp_val⟩ := Classical.choose_spec
      (hcomponents (Usize.ofNatCore i (hbound i hi)) (hvalp i hi))
    have hVi_choose : V i = Classical.choose
        (hcomponents (Usize.ofNatCore i (hbound i hi)) (hvalp i hi)) := by
      rw [hVdef]
      simp only [dif_pos hi]
    have houtval : (out1.val.getD i 0#i64).val = divRoundNearest accc ((2 : ℤ) ^ (32 : ℕ)) := by
      rw [hout1val i hi, hVi_choose, hVcomp_val]
      congr 1
    have houtmem : ∀ j ∈ Finset.range (pos.val + 1),
        (toRealList vs).getD j 0 ∈ toRealList vs := by
      intro j hj
      have hjle : j ≤ pos.val := by simp only [Finset.mem_range] at hj; omega
      have hjlt : j < (toRealList vs).length := by
        rw [toRealList_length, hvslen]
        omega
      rw [List.getD_eq_getElem _ _ hjlt]
      exact List.getElem_mem hjlt
    have hvb' : ∀ j ∈ Finset.range (pos.val + 1),
        -Vb ≤ (Vi j : ℝ) / q32 ∧ (Vi j : ℝ) / q32 ≤ Vb := by
      intro j hj
      have hge : |(toRealList vs).getD j 0| ≤ Vb := by
        rw [hVb]
        exact listAbsMax_ge (toRealList vs) ((toRealList vs).getD j 0) (houtmem j hj)
      have hle := abs_le.mp hge
      rw [← hvit j] at hle
      exact hle
    have hVb' : ∀ j ∈ Finset.range (pos.val + 1), |(Vi j : ℝ) / q32| ≤ Vb := by
      intro j hj
      rw [hVb, hvit j]
      exact listAbsMax_ge (toRealList vs) ((toRealList vs).getD j 0) (houtmem j hj)
    have hxsc : (List.range (pos.val + 1)).map x = sc := by
      rw [← hsclen, hx]
      exact map_getD_range_eq sc 0
    have hvt : (List.range (pos.val + 1)).map (fun j => (Vi j : ℝ) / q32) = toRealList vs := by
      rw [← hvslen]
      exact map_range_getD_toRealList vs Vi (fun j => by simp only [hVi])
    have hmain := attn_stage_compose_closed (pos.val + 1) x y E W Vi mx.val total.val accc
      (out1.val.getD i 0#i64).val (listAbsMax sc) (-Vb) Vb Vb δ hδ' hmax' hnum' hT' hW' haccc
      houtval hvb' hVb'
    rw [attnErrBoundV2]
    rw [hxsc, hvt] at hmain
    rw [show Vb - (-Vb) = 2 * Vb by ring] at hmain
    simpa only [toReal] using hmain

end

end apeiron_word
