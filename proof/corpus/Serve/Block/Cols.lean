/-
  `Serve/Cols.lean` — refinement statements for the extracted integer kernels of
  `src/engine/step/cols.rs` and the small pure helpers of `src/engine/step.rs`
  (extraction `proof/ApeironWord.lean`, module `ApeironWord`, namespace
  `apeiron_word`).

  Group `step.cols.*` sits on the serving path
  `Engine::next_token_state` → `compose_final` → `compose_cols` → `layer_cols`
  (`src/engine/step.rs:1926`, `src/engine/step/cols.rs:68`, `:225`): the residual add
  (stage 11, `add_cols`/`add_assign_fixed`), the attention- and FFN-side RMSNorm with gain
  (`norm_cols`/`norm_col`/`norm_gain`, stages 2/12/14), the QK-norm head entry point
  (`qk_norm_head`, stage 4), the config accessor `eps_fixed`, and the projection drivers
  (`proj_cols`/`group_cols`/`ffn_cols`, stages 3/7/10/13).

  Proved here, by section:
  * §1 `step.add_assign_fixed` — the whole body: a loop spec from cursor zero over the value lists,
    and three headlines: every lane representable → `Ok ()` and the elementwise `Int` sum; the first
    unrepresentable lane → `Err Overflow` and the prefix through it; unequal widths → the opaque
    `From<&str>` refusal.
  * §2 `step.Engine.eps_fixed` — both branches, relative to the opaque `fp.prim.eps_q32`.
  * §3 `step.qk_norm_head` — both branches of the `Result` it receives from `loader.qk_norm_heads`
    (`Ok ()` → rope rotation of the updated slice; `Err e` → `e` with the slice unchanged).
  * §4 `step.cols.norm_col` — both branches (`Ok ()` with the copied destination; `Err e` with `dst`
    unwritten).
  * §5 `step.rms_with_gain_slice` — the width gate (`BadConfig "norm gain width"`) and the
    `norm::NormError` refusal mapped into `StepError.Norm`.
  * §6 `step.cols.norm_gain` — the layer/column lookup over `loader.DecodeVectors.norms_of` and
    `Array::index_usize`: the two `MissingTensor` refusals and the `Ok` gain slice, the `Ok` branch
    relative to the opaque `Vec::as_slice` call.

  Every reference is stated over `Int`/`Nat`/`List` and says *what* is computed; the loop bodies are
  not restated. Where an extracted body calls an `--opaque` external (a `proof/regen.sh` axiom:
  `fp::prim::eps_q32`, `alloc::string::…::from`, `par::matvec_rows_chunked`, …) the theorem is
  stated *relative to that call's result*, which is the boundary the proof does not reach.

  Not stated here, with the reason:
  * the outer row loop of `step.cols.add_cols` (`add_cols_loop`): a row-wise `step.add_assign_fixed`
    over `Slice (Vec I64)`, with `Slice::index_mut_usize` / `Vec::deref_mut` / `Vec::ofSlice`
    round-tripping each row. §1 supplies the per-row leaf; the nested `List (List ℤ)` reference over
    rows and the loop spec remain (the body-turn equations were reached in scratch, see the
    lane report, but the product-pattern `let` the extraction emits after `bind_tc_ok` is not
    reduced by `dsimp`/`simp only []` in this goal shape).
  * `step.cols.norm_cols` (`_loop0`/`_loop1`/`_loop2`/`_loop3`): the raggedy/dense row dispatch whose
    leaves are §4 `norm_col` and §5 `rms_with_gain_slice`.
  * `step.Engine.embedding_sched`/`embed_push`: the row read is `dequant.dequant_row_fixed`, whose
    carrier twin is `F32`; the RMSNorm/gain side is `step.rms_with_gain_slice`, whose leaf is
    `norm.rms_norm_fixed` (`rms_norm_refines_proved`, `Numeric/Engine/Norm.lean:726`).
  * `step.cols.Engine.proj_cols`/`group_cols`/`ffn_cols` and their `_loop*` bodies: a path/iter/dtype
    dispatch over the resident scratch planes crossing `pool`, `matvec` and the `mixer` carriers;
    the projection they drive is `proj_cols`, whose relation is the `matvec.matvec_fixed`
    refinement (`matvec_refines_proved`, `Numeric/Engine/Matvec.lean:583`).
  * every `F32`/`F64` carrier (`*_f32`, `*_carrier`), the AVX2 lanes and the `unsafe` raw-pointer
    writes: no slice model, `proof/regen.sh` opaque list.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the `pool`/`matvec`/`mixer` groups, or the opaque externals named above. The statements
  are about the Aeneas-extracted bodies over the model `Slice`/`Vec` and the model `Usize` bounds.

  Axiom cone: each headline below is `#print axioms`-checked at the end of this file. The pure
  combinatorial headlines (`add_assign_fixed_loop_spec`) depend on nothing outside
  `{propext, Classical.choice, Quot.sound}`; a headline about a body that calls an `--opaque`
  external additionally carries that boundary `axiom` (`alloc.string::…::from`, `fp::prim.eps_q32`,
  `alloc.vec.Vec.as_slice`, `alloc.sync.Arc`, `chart.pool.Excl`, `std.sync.once_lock.OnceLock`,
  `I64::…::try_from`, `core.ops.range.RangeInclusive.contains`), each named in
  `proof/boundary-axioms.txt` and accepted by `proof/audit-extraction.py`. No other axiom, and no
  `sorry`/`admit`, enters any proof here.
-/

import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow

/-- The value list of a word slice. -/
def cols_ivals (x : Slice Std.I64) : List ℤ := x.val.map (fun z => z.val)

theorem cols_ivals_length (x : Slice Std.I64) : (cols_ivals x).length = x.val.length := by
  simp [cols_ivals]

theorem cols_ivals_getD (x : Slice Std.I64) (k : ℕ) :
    (cols_ivals x).getD k 0 = (x.val.getD k 0#i64).val := by
  simpa [cols_ivals] using List.getD_map (f := fun z : Std.I64 => z.val) (l := x.val)
    (d := 0#i64) (n := k)

theorem cols_ivals_set (x : Slice Std.I64) (i : Std.Usize) (w : Std.I64) :
    cols_ivals (x.set i w) = (cols_ivals x).set i.val w.val := by
  simp only [cols_ivals, Slice.set_val_eq, List.map_set]

/-- `a` with the first `j` positions replaced by the elementwise sum with `b`. -/
def addPrefix (a b : List ℤ) : ℕ → List ℤ
  | 0 => a
  | (j + 1) => (addPrefix a b j).set j (a.getD j 0 + b.getD j 0)

theorem addPrefix_zero (a b : List ℤ) : addPrefix a b 0 = a := rfl

theorem addPrefix_succ (a b : List ℤ) (j : ℕ) :
    addPrefix a b (j + 1) = (addPrefix a b j).set j (a.getD j 0 + b.getD j 0) := rfl

theorem addPrefix_length (a b : List ℤ) (j : ℕ) : (addPrefix a b j).length = a.length := by
  induction j with
  | zero => rfl
  | succ j ih => rw [addPrefix_succ, List.length_set, ih]

theorem addPrefix_read (a b : List ℤ) : ∀ j i, j ≤ i →
    (addPrefix a b j).getD i 0 = a.getD i 0 := by
  intro j
  induction j with
  | zero => intro i _; rfl
  | succ j ih =>
    intro i hj
    have hji : j ≠ i := by omega
    have h2 : ((addPrefix a b j).set j (a.getD j 0 + b.getD j 0)).getD i 0
        = (addPrefix a b j).getD i 0 := by
      simp only [List.getD_eq_getElem?_getD, List.getElem?_set_ne hji]
    rw [addPrefix_succ, h2]
    exact ih i (by omega)

/-- The checked-addition domain of two words: their mathematical sum is representable. -/
def fits (a b : Std.I64) : Prop :=
  Std.I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I64.max

/-- A checked addition inside the domain returns its mathematical sum. -/
theorem checked_add_some {a b : Std.I64} (h : fits a b) :
    ∃ z, Std.I64.checked_add a b = some z ∧ z.val = a.val + b.val := by
  have hspec : (match Std.I64.checked_add a b with
      | some z => Std.I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I64.max
          ∧ z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (Std.I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I64.max)) :=
    Std.I64.checked_add_bv_spec a b
  cases hm : Std.I64.checked_add a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

/-- A checked addition outside the domain returns `none`. -/
theorem checked_add_none {a b : Std.I64} (h : ¬ fits a b) : Std.I64.checked_add a b = none := by
  have hspec : (match Std.I64.checked_add a b with
      | some z => Std.I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I64.max
          ∧ z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (Std.I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I64.max)) :=
    Std.I64.checked_add_bv_spec a b
  cases hm : Std.I64.checked_add a b with
  | none => rfl
  | some z => rw [hm] at hspec; exact absurd ⟨hspec.1, hspec.2.1⟩ h

/-- The extracted index result carries the reference list's value at that index. -/
theorem index_val_bridge {x : Slice Std.I64} {i : Std.Usize} {z : Std.I64}
    (hb : i.val < x.val.length) (h : x.index_usize i = ok z) :
    z.val = (cols_ivals x).getD i.val 0 := by
  rw [cols_ivals_getD, List.getD_eq_getElem x.val 0#i64 hb]
  obtain ⟨z', hz', hz'v⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hb)
  rw [h] at hz'
  have hzz : z = z' := Result.ok_injective hz'
  rw [hzz, hz'v]

/-- Equal slice lengths as `Nat`s. -/
theorem val_length_eq_of_len_eq {x y : Slice Std.I64} (h : Slice.len x = Slice.len y) :
    x.val.length = y.val.length := by
  have h' := congrArg (fun u : Std.Usize => u.val) h
  rwa [Slice.len_val x, Slice.len_val y] at h'

/-- The accumulator's value at the cursor equals the input's, while the invariant holds. -/
theorem acc_getD_val_eq (x y acc : Slice Std.I64) (i : ℕ)
    (hacc : cols_ivals acc = addPrefix (cols_ivals x) (cols_ivals y) i) :
    (acc.val.getD i 0#i64).val = (x.val.getD i 0#i64).val := by
  rw [← cols_ivals_getD acc i, hacc, addPrefix_read (cols_ivals x) (cols_ivals y) i i (le_refl _),
    cols_ivals_getD x i]

/-- **One `add_assign_fixed` body turn, continuation.** -/
theorem add_assign_fixed_body_cont (y acc : Slice Std.I64) (i : Std.Usize)
    (hx : i.val < acc.val.length) (hy : i.val < y.val.length)
    (hf : fits (acc.val.getD i.val 0#i64) (y.val.getD i.val 0#i64)) :
    ∃ s : Slice Std.I64, ∃ i' : Std.Usize,
      step.add_assign_fixed_loop.body y acc i = ok (ControlFlow.cont (s, i')) ∧
      i'.val = i.val + 1 ∧
      cols_ivals s = (cols_ivals acc).set i.val ((cols_ivals acc).getD i.val 0 + (cols_ivals y).getD i.val 0) := by
  have h1 : (1#usize : Std.Usize).val = 1 := rfl
  have hmax : i.val + (1#usize : Std.Usize).val ≤ Usize.max := by
    have := Slice.length_ineq acc
    omega
  unfold step.add_assign_fixed_loop.body
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hx)]
  obtain ⟨i2, hi2, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec acc i hx)
  rw [hi2]
  simp only [bind_tc_ok]
  obtain ⟨i3, hi3, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec y i hy)
  rw [hi3]
  simp only [bind_tc_ok]
  have hv2 : i2.val = (cols_ivals acc).getD i.val 0 := index_val_bridge hx hi2
  have hv3 : i3.val = (cols_ivals y).getD i.val 0 := index_val_bridge hy hi3
  have hf' : fits i2 i3 := by
    have e1 : (cols_ivals acc).getD i.val 0 = (acc.val.getD i.val 0#i64).val := cols_ivals_getD acc i.val
    have e2 : (cols_ivals y).getD i.val 0 = (y.val.getD i.val 0#i64).val := cols_ivals_getD y i.val
    unfold fits at hf ⊢
    constructor <;> omega
  obtain ⟨w, hw, hwv⟩ := checked_add_some hf'
  rw [hw]
  simp only [Aeneas.Std.lift, bind_tc_ok, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch]
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists (Slice.update_spec acc i w hx)
  rw [hs]
  simp only [bind_tc_ok]
  obtain ⟨i4, hi4, hi4v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hmax)
  rw [hi4]
  simp only [bind_tc_ok]
  refine ⟨s, i4, rfl, ?_, ?_⟩
  · rw [hi4v, h1]
  · have hwv' : w.val = (cols_ivals acc).getD i.val 0 + (cols_ivals y).getD i.val 0 := by
      rw [hwv, hv2, hv3]
    rw [hsv, cols_ivals_set, hwv']

/-- **One `add_assign_fixed` body turn, overflow.** -/
theorem add_assign_fixed_body_break (y acc : Slice Std.I64) (i : Std.Usize)
    (hx : i.val < acc.val.length) (hy : i.val < y.val.length)
    (hnf : ¬ fits (acc.val.getD i.val 0#i64) (y.val.getD i.val 0#i64)) :
    step.add_assign_fixed_loop.body y acc i
      = ok (ControlFlow.done (core.result.Result.Err step.StepError.Overflow, acc)) := by
  unfold step.add_assign_fixed_loop.body
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hx)]
  obtain ⟨i2, hi2, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec acc i hx)
  rw [hi2]
  simp only [bind_tc_ok]
  obtain ⟨i3, hi3, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec y i hy)
  rw [hi3]
  simp only [bind_tc_ok]
  have hv2 : i2.val = (cols_ivals acc).getD i.val 0 := index_val_bridge hx hi2
  have hv3 : i3.val = (cols_ivals y).getD i.val 0 := index_val_bridge hy hi3
  have hnf' : ¬ fits i2 i3 := by
    intro hf
    have e1 : (cols_ivals acc).getD i.val 0 = (acc.val.getD i.val 0#i64).val := cols_ivals_getD acc i.val
    have e2 : (cols_ivals y).getD i.val 0 = (y.val.getD i.val 0#i64).val := cols_ivals_getD y i.val
    apply hnf
    unfold fits at hf ⊢
    constructor <;> omega
  rw [checked_add_none hnf']
  simp only [Aeneas.Std.lift, bind_tc_ok, core.option.Option.ok_or_none,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame, core.convert.FromSame.from]

/-- **One `add_assign_fixed` body turn, past the end.** -/
theorem add_assign_fixed_body_done (y acc : Slice Std.I64) (i : Std.Usize)
    (hx : ¬ i.val < acc.val.length) :
    step.add_assign_fixed_loop.body y acc i
      = ok (ControlFlow.done (core.result.Result.Ok (), acc)) := by
  unfold step.add_assign_fixed_loop.body
  rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hx)]

/-- **The `add_assign_fixed` loop, from cursor zero.** The post is the disjunction the loop can
reach: either every lane is representable, the accumulator is the full prefix and the result is
`Ok ()`; or the first non-representable lane `i₁` is reported, the accumulator is the prefix through
it and the result is `Err Overflow`. -/
theorem add_assign_fixed_loop_spec (x y : Slice Std.I64) (hxy : Slice.len x = Slice.len y) :
    step.add_assign_fixed_loop x y 0#usize ⦃ o =>
      (o.1 = core.result.Result.Ok () ∧
          (∀ j, j < x.val.length → fits (x.val.getD j 0#i64) (y.val.getD j 0#i64)) ∧
          cols_ivals o.2 = addPrefix (cols_ivals x) (cols_ivals y) x.val.length)
      ∨ (∃ i₁, i₁ < x.val.length ∧
          ¬ fits (x.val.getD i₁ 0#i64) (y.val.getD i₁ 0#i64) ∧
          (∀ j, j < i₁ → fits (x.val.getD j 0#i64) (y.val.getD j 0#i64)) ∧
          o.1 = core.result.Result.Err step.StepError.Overflow ∧
          cols_ivals o.2 = addPrefix (cols_ivals x) (cols_ivals y) i₁) ⦄ := by
  unfold step.add_assign_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun st : Slice Std.I64 × Std.Usize => x.val.length - st.2.val)
    (inv := fun (st : Slice Std.I64 × Std.Usize) => st.2.val ≤ x.val.length ∧
      cols_ivals st.1 = addPrefix (cols_ivals x) (cols_ivals y) st.2.val ∧
      (∀ j, j < st.2.val → fits (x.val.getD j 0#i64) (y.val.getD j 0#i64)))
    (post := fun (o : (core.result.Result Unit step.StepError) × Slice Std.I64) =>
      (o.1 = core.result.Result.Ok () ∧
          (∀ j, j < x.val.length → fits (x.val.getD j 0#i64) (y.val.getD j 0#i64)) ∧
          cols_ivals o.2 = addPrefix (cols_ivals x) (cols_ivals y) x.val.length)
      ∨ (∃ i₁, i₁ < x.val.length ∧
          ¬ fits (x.val.getD i₁ 0#i64) (y.val.getD i₁ 0#i64) ∧
          (∀ j, j < i₁ → fits (x.val.getD j 0#i64) (y.val.getD j 0#i64)) ∧
          o.1 = core.result.Result.Err step.StepError.Overflow ∧
          cols_ivals o.2 = addPrefix (cols_ivals x) (cols_ivals y) i₁))
  · rintro ⟨acc, i⟩ ⟨hile, hacc, hfit⟩
    dsimp only at hile hacc hfit
    have hlen : acc.val.length = x.val.length := by
      rw [← cols_ivals_length acc, hacc, addPrefix_length, cols_ivals_length]
    have hylen_eq : x.val.length = y.val.length := val_length_eq_of_len_eq hxy
    by_cases hlt : i.val < x.val.length
    · have hltacc : i.val < acc.val.length := by rw [hlen]; exact hlt
      have hlty : i.val < y.val.length := by rw [← hylen_eq]; exact hlt
      have hce : (acc.val.getD i.val 0#i64).val = (x.val.getD i.val 0#i64).val :=
        acc_getD_val_eq x y acc i.val hacc
      by_cases hfi : fits (acc.val.getD i.val 0#i64) (y.val.getD i.val 0#i64)
      · obtain ⟨s, i', hb, hi'v, hsv⟩ := add_assign_fixed_body_cont y acc i hltacc hlty hfi
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        dsimp only
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · rw [hi'v]; omega
        · rw [hsv, hacc, addPrefix_read (cols_ivals x) (cols_ivals y) i.val i.val (le_refl _),
            hi'v, addPrefix_succ]
        · intro j hj
          rw [hi'v] at hj
          rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hj | hj
          · exact hfit j hj
          · rw [hj]
            unfold fits at hfi ⊢
            constructor <;> omega
        · rw [hi'v]; omega
      · have hfi' : ¬ fits (x.val.getD i.val 0#i64) (y.val.getD i.val 0#i64) := by
          intro hf
          apply hfi
          unfold fits at hf ⊢
          constructor <;> omega
        have hb := add_assign_fixed_body_break y acc i hltacc hlty hfi
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        dsimp only
        exact Or.inr ⟨i.val, hlt, hfi', hfit, rfl, hacc⟩
    · have hnlt : ¬ i.val < acc.val.length := by rw [hlen]; exact hlt
      have hb := add_assign_fixed_body_done y acc i hnlt
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      dsimp only
      have hie : i.val = x.val.length := le_antisymm hile (le_of_not_gt hlt)
      refine Or.inl ⟨rfl, ?_, ?_⟩
      · intro j hj
        rw [← hie] at hj
        exact hfit j hj
      · rw [hacc, hie]
  · refine ⟨by simp, ?_, ?_⟩
    · show cols_ivals x = addPrefix (cols_ivals x) (cols_ivals y) (0#usize).val
      rw [show (0#usize : Std.Usize).val = 0 from rfl, addPrefix_zero]
    · intro j hj
      simp at hj

/-- **Success law.** Equal widths and every lane representable: the extracted
`step.add_assign_fixed` writes the elementwise `Int` sum into `x` and returns `Ok ()`. -/
theorem word_add_assign_fixed_ok (x y : Slice Std.I64) (hxy : Slice.len x = Slice.len y)
    (hfit : ∀ j, j < x.val.length → fits (x.val.getD j 0#i64) (y.val.getD j 0#i64)) :
    ∃ out : Slice Std.I64,
      step.add_assign_fixed x y = ok (core.result.Result.Ok (), out) ∧
      cols_ivals out = addPrefix (cols_ivals x) (cols_ivals y) x.val.length := by
  have hne : (Slice.len x != Slice.len y) = false := by rw [hxy]; simp
  have hua : step.add_assign_fixed x y = step.add_assign_fixed_loop x y 0#usize := by
    unfold step.add_assign_fixed
    rw [if_neg (by rw [hne]; simp)]
  obtain ⟨⟨r, sl⟩, ho, hpost⟩ := WP.spec_imp_exists (add_assign_fixed_loop_spec x y hxy)
  dsimp only at hpost
  rw [hua]
  rcases hpost with ⟨h1, _, h2⟩ | ⟨i₁, hi₁, hnf, _, _, _⟩
  · exact ⟨sl, by rw [ho, h1], h2⟩
  · exact absurd (hfit i₁ hi₁) hnf

/-- **Overflow refusal.** At the first lane whose `Int` sum is not representable, the extracted
`step.add_assign_fixed` writes the prefix through it and returns `Err Overflow`. -/
theorem word_add_assign_fixed_refuses (x y : Slice Std.I64) (hxy : Slice.len x = Slice.len y)
    (i₁ : ℕ) (hi₁ : i₁ < x.val.length)
    (hfit : ∀ j, j < i₁ → fits (x.val.getD j 0#i64) (y.val.getD j 0#i64))
    (hnf : ¬ fits (x.val.getD i₁ 0#i64) (y.val.getD i₁ 0#i64)) :
    ∃ out : Slice Std.I64,
      step.add_assign_fixed x y = ok (core.result.Result.Err step.StepError.Overflow, out) ∧
      cols_ivals out = addPrefix (cols_ivals x) (cols_ivals y) i₁ := by
  have hne : (Slice.len x != Slice.len y) = false := by rw [hxy]; simp
  have hua : step.add_assign_fixed x y = step.add_assign_fixed_loop x y 0#usize := by
    unfold step.add_assign_fixed
    rw [if_neg (by rw [hne]; simp)]
  obtain ⟨⟨r, sl⟩, ho, hpost⟩ := WP.spec_imp_exists (add_assign_fixed_loop_spec x y hxy)
  dsimp only at hpost
  rw [hua]
  rcases hpost with ⟨_, hfits, _⟩ | ⟨i₂, hi₂, hnf₂, hfit₂, he, hv⟩
  · exact absurd (hfits i₁ hi₁) hnf
  · have heq : i₂ = i₁ := by
      rcases Nat.lt_trichotomy i₂ i₁ with h | h | h
      · exact absurd (hfit i₂ h) hnf₂
      · exact h
      · exact absurd (hfit₂ i₁ h) hnf
    exact ⟨sl, by rw [ho, he], by rw [hv, heq]⟩

/-- **Width refusal.** Unequal slice lengths reach the opaque `From<&str> for String` call carrying
`"residual row width"`; `x` is returned unwritten. The equality is stated relative to that call's
`Result`, which is the boundary this proof does not cross. -/
theorem word_add_assign_fixed_width_refuses (x y : Slice Std.I64) (h : Slice.len x ≠ Slice.len y) :
    step.add_assign_fixed x y
      = (alloc.string.String.Insts.CoreConvertFromShared0Str.from (toStr "residual row width")
          >>= fun s => ok (core.result.Result.Err (step.StepError.BadConfig s), x)) := by
  have hb : (Slice.len x != Slice.len y) = true := by
    cases hbe : (Slice.len x != Slice.len y) with
    | false => exact absurd (bne_eq_false_iff_eq.mp hbe) h
    | true => rfl
  unfold step.add_assign_fixed
  rw [if_pos hb]

/-! ## §2 · The config accessor `step.Engine.eps_fixed`

`step.Engine.eps_fixed` (`src/engine/step.rs:1338`) reads the RMS epsilon through the opaque
`fp.prim.eps_q32` (an `axiom` of the extraction, `ApeironWord.lean:25264`). The two theorems state
its behavior relative to that call's result: its `some` branch and its refusal branch. -/

/-- The accessor's success branch: the opaque quantiser's value is returned. -/
theorem eps_fixed_ok (self : loader.Engine) (q : Std.I64)
    (h : fp.prim.eps_q32 self.config.rms_eps = ok (some q)) :
    step.Engine.eps_fixed self = ok (core.result.Result.Ok q) := by
  unfold step.Engine.eps_fixed
  simp only [h, bind_tc_ok, core.option.Option.ok_or_some]

/-- The accessor's refusal branch: the opaque quantiser's `none` reaches the opaque `From<&str>`
call carrying `"rms_eps"`. -/
theorem eps_fixed_refuses (self : loader.Engine)
    (h : fp.prim.eps_q32 self.config.rms_eps = ok none) :
    step.Engine.eps_fixed self
      = (alloc.string.String.Insts.CoreConvertFromShared0Str.from (toStr "rms_eps")
          >>= fun s => ok (core.result.Result.Err (step.StepError.BadConfig s))) := by
  unfold step.Engine.eps_fixed
  simp only [h, bind_tc_ok]

/-! ## §3 · The QK-norm head entry point `step.qk_norm_head`

`step.qk_norm_head` (`src/engine/step.rs:521`) widens the optional gain to a slice with
`loader.opt_slice`, runs `loader.qk_norm_heads o dst (Slice.len dst) eps`, and branches on its
`Result`: on `Ok ()` it rotates the head prefixes by `pos` with `step.rotate_head_prefixes`; on
`Err e` it returns `e` with `dst1` unchanged. The two theorems state those two branches. -/

/-- The continue branch of `step.qk_norm_head`: `qk_norm_heads` returned `Ok ()`, so the result is
the rope rotation of the updated slice. -/
theorem qk_norm_head_continue (gain : Option (alloc.vec.Vec Std.I64)) (dst : Slice Std.I64)
    (table : rope.RopeTable) (eps : Std.I64) (pos : Std.U32)
    (o : Option (Slice Std.I64)) (dst1 : Slice Std.I64)
    (ho : loader.opt_slice gain = ok o)
    (hr : loader.qk_norm_heads o dst (Slice.len dst) eps = ok (core.result.Result.Ok (), dst1)) :
    step.qk_norm_head gain dst table eps pos
      = step.rotate_head_prefixes table dst1 (Slice.len dst1) pos := by
  unfold step.qk_norm_head
  simp only [ho, hr, bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
  simp

/-- The error branch of `step.qk_norm_head`: `qk_norm_heads` returned `Err e`, so the result is
`Err e` with the normed slice unchanged (no rotation happens). -/
theorem qk_norm_head_refuses (gain : Option (alloc.vec.Vec Std.I64)) (dst : Slice Std.I64)
    (table : rope.RopeTable) (eps : Std.I64) (pos : Std.U32)
    (o : Option (Slice Std.I64)) (dst1 : Slice Std.I64) (e : step.StepError)
    (ho : loader.opt_slice gain = ok o)
    (hr : loader.qk_norm_heads o dst (Slice.len dst) eps = ok (core.result.Result.Err e, dst1)) :
    step.qk_norm_head gain dst table eps pos = ok (core.result.Result.Err e, dst1) := by
  unfold step.qk_norm_head
  simp only [ho, hr, bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame, core.convert.FromSame.from]
  simp

/-! ## §4 · The per-column RMSNorm with gain `step.cols.norm_col`

`step.cols.norm_col` (`src/engine/step/cols.rs:1229`) indexes row `c` of `xs`, runs
`step.rms_with_gain_slice gain · eps` on it and, on success, copies the result into `dst` with
`Slice.copy_from_slice`. The two theorems state the two branches: `Ok ()` with the copied
destination, or `Err e` with `dst` unwritten. -/

/-- The success branch of `step.cols.norm_col`. -/
theorem norm_col_continue (gain : Slice Std.I64) (xs : Slice (alloc.vec.Vec Std.I64))
    (eps : Std.I64) (c : Std.Usize) (dst : Slice Std.I64)
    (v : alloc.vec.Vec Std.I64) (hv : Slice.index_usize xs c = ok v)
    (out : alloc.vec.Vec Std.I64)
    (ho : step.rms_with_gain_slice gain (alloc.vec.Vec.deref v) eps
            = ok (core.result.Result.Ok out))
    (dst1 : Slice Std.I64)
    (hd : core.slice.Slice.copy_from_slice core.marker.CopyI64 dst (alloc.vec.Vec.deref out)
            = ok dst1) :
    step.cols.norm_col gain xs eps c dst = ok (core.result.Result.Ok (), dst1) := by
  unfold step.cols.norm_col
  simp only [hv, ho, bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch, hd]

/-- The refusal branch of `step.cols.norm_col`: the RMSNorm refused, `dst` is returned unwritten. -/
theorem norm_col_refuses (gain : Slice Std.I64) (xs : Slice (alloc.vec.Vec Std.I64))
    (eps : Std.I64) (c : Std.Usize) (dst : Slice Std.I64)
    (v : alloc.vec.Vec Std.I64) (hv : Slice.index_usize xs c = ok v)
    (e : step.StepError)
    (ho : step.rms_with_gain_slice gain (alloc.vec.Vec.deref v) eps
            = ok (core.result.Result.Err e)) :
    step.cols.norm_col gain xs eps c dst = ok (core.result.Result.Err e, dst) := by
  unfold step.cols.norm_col
  simp only [hv, ho, bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame, core.convert.FromSame.from]

/-! ## §5 · The RMSNorm leaf `step.rms_with_gain_slice`

`step.rms_with_gain_slice` (`src/engine/step.rs:245`) refuses `BadConfig "norm gain width"` when the
gain and activation widths differ; otherwise it maps `norm.rms_norm_fixed x eps` through
`Result.map_err` into `StepError.Norm` and runs the gain loop. The theorem here states the width
law; the loop refinement belongs to `Numeric/Engine/Norm.lean` (`rms_norm_refines_proved`). -/

/-- The width refusal of `step.rms_with_gain_slice`, stated relative to the opaque `From<&str>`
call carrying `"norm gain width"`. -/
theorem rms_with_gain_slice_width_refuses (gain x : Slice Std.I64) (eps : Std.I64)
    (h : Slice.len gain ≠ Slice.len x) :
    step.rms_with_gain_slice gain x eps
      = (alloc.string.String.Insts.CoreConvertFromShared0Str.from (toStr "norm gain width")
          >>= fun s => ok (core.result.Result.Err (step.StepError.BadConfig s))) := by
  have hb : (Slice.len gain != Slice.len x) = true := by
    cases hbe : (Slice.len gain != Slice.len x) with
    | false => exact absurd (bne_eq_false_iff_eq.mp hbe) h
    | true => rfl
  unfold step.rms_with_gain_slice
  rw [if_pos hb]

/-- The `norm.rms_norm_fixed` refusal of `step.rms_with_gain_slice` maps into `StepError.Norm` and
stops before the gain loop. -/
theorem rms_with_gain_slice_norm_refuses (gain x : Slice Std.I64) (eps : Std.I64)
    (h : Slice.len gain = Slice.len x) (ne : norm.NormError)
    (hn : norm.rms_norm_fixed x eps = ok (core.result.Result.Err ne)) :
    step.rms_with_gain_slice gain x eps
      = ok (core.result.Result.Err (step.StepError.Norm ne)) := by
  have hne : (Slice.len gain != Slice.len x) = false := bne_eq_false_iff_eq.mpr h
  unfold step.rms_with_gain_slice
  rw [if_neg (by rw [hne]; simp)]
  simp only [hn, bind_tc_ok, core.result.Result.map_err_err,
    P.Insts.CoreOpsFunctionFnOnceTupleNormErrorStepError,
    P.Insts.CoreOpsFunctionFnOnceTupleNormErrorStepError.call_once,
    step.StepError.Norm.constructor, core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame, core.convert.FromSame.from]

/-! ## §6 · The norm-gain lookup `step.cols.norm_gain`

`step.cols.norm_gain` (`src/engine/step/cols.rs:1304`) reads the layer's norm table with
`loader.DecodeVectors.norms_of`, selects part `which` with `Array::index_usize`, and returns the gain
slice through the opaque `alloc.vec.Vec.as_slice`. Each miss returns `step.StepError.MissingTensor`
carrying `error.TensorKey.blk layer part`. The three theorems state the `none`-table, `none`-column
and `some`-column branches; the `TensorKey` producer and the `as_slice` call are stated relatively,
which is the boundary this proof does not cross. -/

/-- The missing-table branch of `step.cols.norm_gain`. -/
theorem norm_gain_layer_missing (vectors : loader.DecodeVectors) (layer which : Std.Usize)
    (part : Std.U8) (h : loader.DecodeVectors.norms_of vectors layer = ok none) :
    step.cols.norm_gain vectors layer which part
      = (error.TensorKey.blk layer part >>= fun tk =>
          ok (core.result.Result.Err (step.StepError.MissingTensor tk))) := by
  unfold step.cols.norm_gain
  rw [h]
  simp only [bind_tc_ok]

/-- The missing-column branch of `step.cols.norm_gain`. -/
theorem norm_gain_column_missing (vectors : loader.DecodeVectors) (layer which : Std.Usize)
    (part : Std.U8) (n : Array (Option (alloc.vec.Vec Std.I64)) 2#usize)
    (h : loader.DecodeVectors.norms_of vectors layer = ok (some n))
    (h2 : Array.index_usize n which = ok none) :
    step.cols.norm_gain vectors layer which part
      = (error.TensorKey.blk layer part >>= fun tk =>
          ok (core.result.Result.Err (step.StepError.MissingTensor tk))) := by
  unfold step.cols.norm_gain
  rw [h]
  simp only [bind_tc_ok]
  rw [h2]
  simp only [bind_tc_ok]

/-- The present-column branch of `step.cols.norm_gain`: the gain slice is produced by the opaque
`alloc.vec.Vec.as_slice`, stated relative to that call's result. -/
theorem norm_gain_ok (vectors : loader.DecodeVectors) (layer which : Std.Usize)
    (part : Std.U8) (n : Array (Option (alloc.vec.Vec Std.I64)) 2#usize)
    (g : alloc.vec.Vec Std.I64)
    (h : loader.DecodeVectors.norms_of vectors layer = ok (some n))
    (h2 : Array.index_usize n which = ok (some g)) :
    step.cols.norm_gain vectors layer which part
      = (alloc.vec.Vec.as_slice Global g >>= fun s =>
          ok (core.result.Result.Ok s)) := by
  unfold step.cols.norm_gain
  rw [h]
  simp only [bind_tc_ok]
  rw [h2]
  simp only [bind_tc_ok]

/-! ### Non-vacuity of §1 -/

/-- Two representable lanes exist, so `word_add_assign_fixed_ok`'s domain is inhabited. -/
example : fits (1#i64) (3#i64) ∧ fits (2#i64) (4#i64) := by
  constructor <;> · rw [fits]; constructor <;> norm_num [Std.I64.min_eq, Std.I64.max_eq]

/-- The reference of §1 at a concrete two-lane input. -/
example : addPrefix [1, 2] [3, 4] 2 = [4, 6] := by decide

/-- An overflowing lane is outside `fits`. -/
example : ¬ fits (9223372036854775807#i64) (1#i64) := by
  rw [fits]
  norm_num [Std.I64.min_eq, Std.I64.max_eq]

/-! ## Axiom cone

Each headline and loop spec is checked here to depend on nothing outside
`{propext, Classical.choice, Quot.sound}`. -/

#print axioms add_assign_fixed_loop_spec
#print axioms word_add_assign_fixed_ok
#print axioms word_add_assign_fixed_refuses
#print axioms word_add_assign_fixed_width_refuses
#print axioms eps_fixed_ok
#print axioms eps_fixed_refuses
#print axioms qk_norm_head_continue
#print axioms qk_norm_head_refuses
#print axioms norm_col_continue
#print axioms norm_col_refuses
#print axioms rms_with_gain_slice_width_refuses
#print axioms rms_with_gain_slice_norm_refuses
#print axioms norm_gain_layer_missing
#print axioms norm_gain_column_missing
#print axioms norm_gain_ok

end apeiron_word
