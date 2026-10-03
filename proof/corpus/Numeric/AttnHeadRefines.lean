/-
End-to-end refinement of the extracted single-token attention head
(`APEIRON-ALGEBRA-G3-ATTN-W3-T1d`, re-proved against the reshaped kernel `APEIRON-T2F2F`).

`Numeric.AttnHeadWindowsScore` and `Numeric.AttnHeadWindowsValue` discharge the four window
predicates of the extracted loops from `attn_domain_v2`; `Numeric.AttnScoreStage` bounds the
realised Q32.32 score against `idealScoreList`; `Numeric.AttnWeightRecipSpec` supplies the
reciprocal-arm weight identity. This module assembles them through
`word_attn_head_ok` (`Numeric.EngineAttnRefinement`) and `attn_stage_compose_closed`
(`Numeric.AttnSoftmaxPerturb`) into the corrected end-to-end target:

* `attn_head_refines_v2_proved`: `attn_head_refines_v2` — on `attn_domain_v2` the body returns
  `Ok` and every output component is within `attnErrBoundV2` of the dense real softmax attention
  of the ideal scores, at the score-stage error `scoreErrV2`.

The weight stage is instantiated at the unquantified reciprocal arm `recip` the extracted body
carries (`attn.recip_u64 total`): `attnWeightResolve_of_recip_u64` discharges `AttnWeightResolve`
on **both** arms — `attn.weight_at` on `none` (`attnWeightResolve_none`) and `attn.weight_at_recip`
on `some r` (`weight_at_recip_spec` through `recip_u64_some_spec`). No arm is left open.

No hypothesis beyond `attn_domain_v2` is introduced.

Trust boundary: reaches the extracted `word_attn_head`, `attn_head_loop2`, `attn_head_loop3`,
`attn.weight_at`, `attn.weight_at_recip` and `attn.recip_u64` of `proof/ApeironWord.lean`, the
plane/score/value denotations of `Numeric.EngineAttnRefinement`, the proved stage composition of
`Numeric.AttnSoftmaxPerturb`, and the reciprocal-arm specification of `Numeric.AttnWeightRecipSpec`.
Does not cover rustc/LLVM codegen of `src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the
`unsafe` container.
-/
import Numeric.AttnHeadWindowsScore
import Numeric.AttnHeadWindowsValue
import Numeric.AttnScoreStage
import Numeric.AttnWeightRecipSpec

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

/-! ## The reciprocal arm at the extracted `recip_u64` call -/

/-- The extracted `attn.recip_u64` never fails: on the `total < 2` and oversized branches it
returns `ok none`, and on the admitted branch its two checked divisions have nonzero divisors
(`2` and the cast total `≥ 2`), so the whole body evaluates to `ok <u64 option>`. -/
theorem recip_u64_ok (total : I128) : ∃ recip : Option U64, attn.recip_u64 total = ok recip := by
  rw [attn.recip_u64]
  by_cases hlt : total < 2#i128
  · exact ⟨none, by rw [if_pos hlt]⟩
  · rw [if_neg hlt]
    obtain ⟨i, hi_eq, -⟩ :=
      WP.spec_imp_exists (U64.div_spec (x := core.num.U64.MAX) (y := 2#u64) (by norm_num))
    rw [hi_eq]
    simp only [bind_tc_ok]
    rw [show I128.Insts.CoreConvertFromU64.from i =
        ok (UScalar.hcast IScalarTy.I128 i) from rfl]
    simp only [bind_tc_ok]
    by_cases hgt : total > UScalar.hcast IScalarTy.I128 i
    · exact ⟨none, by rw [if_pos hgt]⟩
    · rw [if_neg hgt]
      obtain ⟨i2, hi2_eq, -, -⟩ :=
        WP.spec_imp_exists (U128.ShiftLeft_IScalar_spec (x := 1#u128) (y := 64#i32)
          (by norm_num) (by norm_num))
      rw [hi2_eq]
      simp only [lift, bind_tc_ok]
      have hge2 : 2 ≤ total.val := by
        have := (IScalar.le_equiv (2#i128) total).1 (le_of_not_gt hlt)
        simpa using this
      have hi3val : (IScalar.hcast UScalarTy.U128 total).val = total.val.toNat := by
        rw [IScalar.hcast_val_eq, UScalarTy.U128_numBits_eq]
        have hmod : total.val % (2 : ℤ) ^ 128 = total.val :=
          Int.emod_eq_of_lt (by omega) (by
            have hb := total.hBounds
            rw [IScalarTy.I128_numBits_eq] at hb
            omega)
        rw [hmod]
      have hi3pos : (IScalar.hcast UScalarTy.U128 total).val ≠ 0 := by
        rw [hi3val]; omega
      obtain ⟨i4, hi4_eq, -⟩ :=
        WP.spec_imp_exists (U128.div_spec (x := i2) (y := IScalar.hcast UScalarTy.U128 total) hi3pos)
      rw [hi4_eq]
      simp only [bind_tc_ok]
      exact ⟨U64.checked_add (UScalar.cast UScalarTy.U64 i4) 1#u64, rfl⟩

/-- **The whole `AttnWeightResolve`, at whichever arm the extracted loop carries.** For the
`Option U64` the extracted `attn.recip_u64 total` produces, every causal index resolves to
`Ok w` with `w = attnWeightAt`: the `none` arm is `attnWeightResolve_none`, the `some r` arm is
`weight_at_recip_spec` with its reciprocal pinned by `recip_u64_some_spec`. -/
theorem attnWeightResolve_of_recip_u64 (exps : alloc.vec.Vec I64) (total : I128) (pos : Usize)
    (recip : Option U64) (hwin : AttnWeightWindow exps total pos)
    (hrecip : attn.recip_u64 total = ok recip) :
    AttnWeightResolve exps total pos recip := by
  cases recip with
  | none => exact attnWeightResolve_none exps total pos hwin
  | some r =>
    intro j hj
    obtain ⟨h2, h3, hrval⟩ := recip_u64_some_spec total r hrecip
    have htot : 0 < total.val := hwin.2.2.1
    have hnum : 0 ≤ (exps.val.getD j 0#i64).val := (hwin.2.2.2.1 j hj).1
    have hreclo : 2 ≤ total.val.toNat := by
      have hcast : (total.val.toNat : ℤ) = total.val := Int.toNat_of_nonneg (by omega)
      omega
    have hrechi : total.val.toNat ≤ 2 ^ 63 - 1 := by
      have hcast : (total.val.toNat : ℤ) = total.val := Int.toNat_of_nonneg (by omega)
      omega
    have haddI : (exps.val.getD j 0#i64).val * (2 : ℤ) ^ 32 + total.val / 2 ≤ I128.max :=
      (hwin.2.2.2.1 j hj).2.1
    have hfit : -((2 : ℤ) ^ 63) ≤
          divRoundNearest ((exps.val.getD j 0#i64).val * (2 : ℤ) ^ 32) total.val ∧
        divRoundNearest ((exps.val.getD j 0#i64).val * (2 : ℤ) ^ 32) total.val ≤
          (2 : ℤ) ^ 63 - 1 :=
      ⟨(hwin.2.2.2.1 j hj).2.2.1, (hwin.2.2.2.1 j hj).2.2.2⟩
    obtain ⟨w, hw, hwv⟩ :=
      weight_at_recip_spec (exps.val.getD j 0#i64) total r htot hnum hreclo hrechi hrval haddI hfit
    simpa only [] using (show ∃ w : I64,
      attn.weight_at_recip (exps.val.getD j 0#i64) total r =
          ok (core.result.Result.Ok w) ∧ w.val = attnWeightAt exps.val total j
      from ⟨w, hw, hwv⟩)

/-! ## The end-to-end target -/

/-- **Corrected end-to-end refinement (proved).** `attn_head_refines_v2` holds: on `attn_domain_v2`
the extracted `word_attn_head` returns `Ok` and each output component is within `attnErrBoundV2` of
the dense real softmax attention of the ideal scores, at score error `scoreErrV2`. -/
theorem attn_head_refines_v2_proved : attn_head_refines_v2 := by
  intro state layer q_head pos q out kplane vplane h
  have hD := h
  have h0 := h
  obtain ⟨len, slots, group, kvh, stride, first, ktab, vtab, hlayer, hhead, hq', hout', hlen,
    hposlen, hadd, _hslots, hgroup, hkvh, hstorage, hkplane, hvplane, hstride, hfirst, hstridev,
    hfirstv, hkvhKvh, hkvhlt, hposmax⟩ :=
    attn_v2_prefix state layer q_head pos q out kplane vplane hD
  obtain ⟨hadom, _hocc, hscale, _hk, _hklen, hvlen, _hrowdot, _hscorewin, _hvalwin,
    hpos30, hcarrier⟩ := h0
  unfold attn_domain at hadom
  obtain ⟨_, _, _, hhdim, _hcap, _, _, _, _, _houtlen, hposcap⟩ := hadom
  obtain ⟨r, hr, hwr⟩ := hscale
  have hkvhN : kvh.val = q_head.val / (state.n_head.val / state.n_head_kv.val) := by
    simpa [attnKvh] using hkvhKvh
  -- score loop
  have hdom0 := attn_v2_loop0_domain state layer q_head pos q out kplane vplane kvh stride first
    hD hkvhKvh hstride hfirst
  obtain ⟨scores, mx, hloop0, hslen, hpt, hmx, hmax0⟩ :=
    attn_loop0_ok state kplane q pos kvh slots stride first hposmax hdom0
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
  -- the reciprocal arm the executed loop carries, on both arms
  obtain ⟨recip, hrecip⟩ := recip_u64_ok total
  have hwres : AttnWeightResolve exps total pos recip :=
    attnWeightResolve_of_recip_u64 exps total pos recip hwin hrecip
  -- the zero accumulator the extracted body builds
  have hcloneI128 : core.clone.CloneI128.clone (0#i128) = ok (0#i128) := by simp
  obtain ⟨acc0, hacc0_eq, hacc0v, hacc0len⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI128 0#i128 state.head_dim hcloneI128)
  have hacl0 : acc0.val.length = state.head_dim.val := hacc0len
  have hacc0zero : ∀ i, i < state.head_dim.val → (acc0.val.getD i 0#i128).val = 0 := by
    intro i hi
    have hget : acc0.val.getD i 0#i128 = (0#i128) := by
      rw [hacc0v, List.getD_eq_getElem?_getD, List.getElem?_getD_replicate_default_eq]
    rw [hget]; simp
  -- the narrow row-window arithmetic (`AttnValueWindow`'s first conjunct), from `plane_at_spec`
  have harith : stride.val * pos.val + first.val + state.head_dim.val ≤ Usize.max := by
    obtain ⟨_sat, base, end_, _row, _hsat, hbase, hend, hendin, _hb, _hrow, _hrl, _hrg⟩ :=
      plane_at_spec state pos kvh stride first vplane hvlen hfirstv hstridev hkvhlt hhdim hposcap
        pos (le_refl _)
    have h1 : first.val + pos.val * stride.val + state.head_dim.val ≤ Usize.max := by
      rw [← hbase, ← hend]
      exact le_trans hendin vplane.property
    calc stride.val * pos.val + first.val + state.head_dim.val
        = first.val + pos.val * stride.val + state.head_dim.val := by ring
      _ ≤ Usize.max := h1
  -- the two executed loops, with per-component content, at the unquantified arm
  obtain ⟨acc1, out1, hloop2, hloop3, hout1len, hval⟩ :=
    attn_value_components state layer q_head pos q out kplane vplane kvh stride first exps total
      recip acc0 sh hD hkvhN hstridev hfirstv harith hacl0 hacc0zero hwres hexpslen hcaseW hzeroW htw
  refine ⟨out1, ?_, hout1len, ?_⟩
  · exact word_attn_head_ok state layer q_head q out pos hcarrier hlayer hhead hq' hout'
      ktab vtab hstorage len hlen hposlen
      kplane vplane hkplane hvplane
      group kvh hgroup hkvh
      slots hadd
      stride first hstride hfirst
      scores mx hloop0 exps total hloop1 htotnle
      recip hrecip acc0 hacc0_eq acc1 hloop2 out1 hloop3
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
    let W : ℕ → ℤ := fun j => attnWeightAt exps.val total j
    set accc : ℤ := ∑ j ∈ Finset.range (pos.val + 1), W j * Vi j with haccc
    have hW' : ∀ j ∈ Finset.range (pos.val + 1),
        W j = divRoundNearest (E j * (2 : ℤ) ^ (32 : ℕ)) total.val := by
      intro j _
      simp only [W, hE, attnWeightAt]
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
    have hvit : ∀ j, (Vi j : ℝ) / q32 = (toRealList vs).getD j 0 := by
      intro j
      rw [hVi, toRealList_getD]
    have hbound : ∀ n : ℕ, n < state.head_dim.val → n < 2 ^ UScalarTy.Usize.numBits :=
      fun n hn => lt_of_lt_of_le hn (le_of_lt (UScalar.hBounds state.head_dim))
    have houtval : (out1.val.getD i 0#i64).val = divRoundNearest accc ((2 : ℤ) ^ (32 : ℕ)) := by
      have h := hval (Usize.ofNatCore i (hbound i hi))
        (by rw [Usize.ofNatCore_val_eq]; exact hi)
      rw [Usize.ofNatCore_val_eq] at h
      rw [h]
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
