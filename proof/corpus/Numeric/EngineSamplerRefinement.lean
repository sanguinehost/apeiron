/-
Refinement of the extracted engine sampler argmax and the attention KV-state
constructor / append.

The Rust bodies are `src/engine/sampler.rs` (`argmax_fixed`) and `src/engine/attn.rs`
(`KvState::new`, `KvState::append`); `i64` Q32.32 logits and Q32.32 KV planes. The public
entry points are `apeiron_word.word_argmax_fixed`, `apeiron_word.word_kv_state_new` and
`apeiron_word.word_kv_append`.

`Numeric.EngineDotRefinement` is imported for the Q32.32 denotation `toReal` and the
extraction package; the loop proof reuses the `loop.spec_decr_nat` decomposition shape of
the row-dot module.

## Statements

* `sampler.argmax_fixed` scans `logits` with `best := 0`, `i := 1` and updates `best` only on
  a strict `>`; the extracted body is `sampler.argmax_fixed_loop`. `argmaxAt logits best` is
  the exact law: `best` is in range, dominates every in-range logit (weakly) and every earlier
  logit (strictly). `argmax_fixed_loop_spec` proves the loop returns such a `best`; the
  `u32` narrowing either returns it as `Ok` or refuses with `LogitsTooLong`.
  The real reading follows from `toReal` being strictly order-preserving.
* `attn.KvState.new` is a pure constructor: each geometry guard is an exact refusal equality
  (`BadGeometry` for a zero parameter or a nonzero `n_head % n_head_kv` remainder), and
  `word_kv_state_new_geometry` projects the `Ok` record's five geometry fields onto the
  arguments. The `scale` field is the `fixed_rsqrt` output and is not pinned by the guards.
* `attn.KvState.append` tests the layer index first; the layer-out-of-range, both width
  (`RowLength`) and capacity (`CapacityExceeded`) refusals are exact equalities leaving the
  state unchanged, and `word_kv_append_ok` is the success law: the returned position is
  `occupied[layer]`, it increments that lane to `pos + 1`, and it writes the two
  `n_head_kv·head_dim`-wide row blocks at `[pos·width, (pos+1)·width)`.

The `toReal` denotation is the Q32.32 reading from `Numeric.Scale`; the argmax law is stated
over the integers and lifted to the reals by strict order preservation.
-/
import Numeric.EngineDotRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The safe logit read and the argmax law

`logitAt` reads a position with the default `0` outside the slice, so the predicates below
elaborate without a proof obligation; `logitAt_eq_get` is the bridge to the checked
`Slice` index the extracted body uses. -/

/-- The `j`-th Q32.32 logit, `0` outside the length. -/
def logitAt (logits : Slice I64) (j : Nat) : I64 := logits.val.getD j (0#i64)

/-- Inside the length, `logitAt` is the checked `Slice` index. -/
theorem logitAt_eq_get (logits : Slice I64) {j : Nat} (h : j < logits.val.length) :
    logitAt logits j = logits.val[j] := by
  simp only [logitAt, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h,
    Option.getD_some]

/-- The exact argmax law: `best` is in range, no in-range logit exceeds it, and every earlier
logit is strictly smaller. The strict part is the "ties resolved to the lowest index" law. -/
def argmaxAt (logits : Slice I64) (best : Nat) : Prop :=
  best < logits.val.length ∧
  (∀ j, j < logits.val.length → (logitAt logits j).val ≤ (logitAt logits best).val) ∧
  (∀ j, j < best → (logitAt logits j).val < (logitAt logits best).val)

/-- The loop invariant of `sampler.argmax_fixed_loop`: `i` is a prefix bound, `best` is the
least-index prefix argmax and is strictly below `i`. -/
def argmaxLoopInv (logits : Slice I64) (best i : Usize) : Prop :=
  i.val ≤ logits.val.length ∧
  best.val < i.val ∧
  (∀ j, j < i.val → (logitAt logits j).val ≤ (logitAt logits best.val).val) ∧
  (∀ j, j < best.val → (logitAt logits j).val < (logitAt logits best.val).val)

/-! ## One loop-body step -/

/-- One step of the argmax scan: with `i` inside the logits, the extracted body reads
`logits[i]` and `logits[best]`, keeps `i` only on a strict `>`, and advances `i` by one. -/
theorem argmax_loop_body_step (logits : Slice I64) (best i : Usize)
    (hi : i.val < logits.val.length) (hinv : argmaxLoopInv logits best i) :
    ∃ best' i',
      sampler.argmax_fixed_loop.body logits best i = ok (ControlFlow.cont (best', i')) ∧
      i'.val = i.val + 1 ∧
      argmaxLoopInv logits best' i' := by
  obtain ⟨hile, hblt, hmax, hstrict⟩ := hinv
  have hblen : best.val < logits.val.length := by omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Slice.index_usize_spec logits i hi)
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Slice.index_usize_spec logits best hblen)
  have h1 : (1#usize).val = 1 := by norm_num
  have hmaxadd : i.val + (1#usize).val ≤ Usize.max := by
    have hb := logits.property
    rw [h1]; omega
  obtain ⟨i4, hi4, hi4v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hmaxadd)
  have hi4v' : i4.val = i.val + 1 := by rw [hi4v, h1]
  have hlt : i < Slice.len logits :=
    (UScalar.lt_equiv i (Slice.len logits)).2 (by rw [Slice.len_val]; exact hi)
  have hi2l : i2 = logits.val[i.val] := hi2v
  have hi3l : i3 = logits.val[best.val] := hi3v
  by_cases hcmp : i2 > i3
  · have hcmpv : (logitAt logits best.val).val < (logitAt logits i.val).val := by
      have h := hcmp
      rw [gt_iff_lt] at h
      rw [IScalar.lt_equiv] at h
      rw [hi2l, hi3l] at h
      rwa [logitAt_eq_get logits hblen, logitAt_eq_get logits hi]
    refine ⟨i, i4, ?_, ?_, ?_⟩
    · rw [sampler.argmax_fixed_loop.body]
      rw [if_pos hlt, hi2, hi3]
      simp only [bind_tc_ok]
      rw [if_pos hcmp]
      simp only [bind_tc_ok]
      rw [hi4]
      simp only [bind_tc_ok]
    · exact hi4v'
    · refine ⟨by rw [hi4v']; omega, by rw [hi4v']; omega, ?_, ?_⟩
      · intro j hj
        rw [hi4v'] at hj
        by_cases hji : j < i.val
        · have h1' := hmax j hji
          have h2' := hcmpv
          omega
        · have hje : j = i.val := by omega
          subst hje
          exact le_refl _
      · intro j hj
        have h1' := hmax j hj
        have h2' := hcmpv
        omega
  · have hle : (logitAt logits i.val).val ≤ (logitAt logits best.val).val := by
      have h := hcmp
      rw [gt_iff_lt] at h
      rw [IScalar.lt_equiv] at h
      rw [hi2l, hi3l] at h
      rw [logitAt_eq_get logits hi, logitAt_eq_get logits hblen]
      omega
    refine ⟨best, i4, ?_, ?_, ?_⟩
    · rw [sampler.argmax_fixed_loop.body]
      rw [if_pos hlt, hi2, hi3]
      simp only [bind_tc_ok]
      rw [if_neg hcmp]
      simp only [bind_tc_ok]
      rw [hi4]
      simp only [bind_tc_ok]
    · exact hi4v'
    · refine ⟨by rw [hi4v']; omega, by rw [hi4v']; omega, ?_, ?_⟩
      · intro j hj
        rw [hi4v'] at hj
        by_cases hji : j < i.val
        · exact hmax j hji
        · have hje : j = i.val := by omega
          subst hje
          exact hle
      · intro j hj
        exact hstrict j hj

/-! ## The loop spec -/

/-- On nonempty logits the extracted loop returns an index satisfying `argmaxAt`. -/
theorem argmax_loop_spec (logits : Slice I64) (hne : 0 < logits.val.length) :
    sampler.argmax_fixed_loop logits 0#usize 1#usize ⦃ best => argmaxAt logits best.val ⦄ := by
  unfold sampler.argmax_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun s : Usize × Usize => logits.val.length - s.2.val)
    (inv := fun s : Usize × Usize => argmaxLoopInv logits s.1 s.2)
    (post := fun best : Usize => argmaxAt logits best.val)
  · intro s hs
    rcases s with ⟨best, i⟩
    dsimp only at hs ⊢
    obtain ⟨hile, hblt, hmax, hstrict⟩ := hs
    by_cases hlt : i.val < logits.val.length
    · obtain ⟨best', i', hbody, hi'v, hinv'⟩ :=
        argmax_loop_body_step logits best i hlt ⟨hile, hblt, hmax, hstrict⟩
      rw [hbody]
      refine ⟨hinv', ?_⟩
      show logits.val.length - i'.val < logits.val.length - i.val
      rw [hi'v]; omega
    · have hieq : i.val = logits.val.length := by
        have hge := Nat.le_of_not_lt hlt
        omega
      have hnot : ¬ i < Slice.len logits := by
        intro hc
        have hl := (UScalar.lt_equiv i (Slice.len logits)).1 hc
        rw [Slice.len_val] at hl
        exact hlt hl
      have hbody : sampler.argmax_fixed_loop.body logits best i =
          ok (ControlFlow.done best) := by
        rw [sampler.argmax_fixed_loop.body]
        rw [if_neg hnot]
      rw [hbody]
      refine ⟨?_, ?_, ?_⟩
      · rw [← hieq]; exact hblt
      · intro j hj; exact hmax j (by rw [hieq]; exact hj)
      · intro j hj; exact hstrict j hj
  · refine ⟨?_, ?_, ?_, ?_⟩
    · show (1#usize).val ≤ logits.val.length
      have h1 : (1#usize).val = 1 := by norm_num
      rw [h1]; omega
    · show (0#usize).val < (1#usize).val
      have h0 : (0#usize).val = 0 := by norm_num
      have h1 : (1#usize).val = 1 := by norm_num
      rw [h0, h1]; norm_num
    · intro j hj
      have h1 : (1#usize).val = 1 := by norm_num
      rw [h1] at hj
      have h0 : (0#usize).val = 0 := by norm_num
      rw [h0]
      have hje : j = 0 := by omega
      subst hje
      exact le_refl _
    · intro j hj
      have h0 : (0#usize).val = 0 := by norm_num
      rw [h0] at hj
      omega

/-- Existential form: the loop returns an in-range least-index argmax. -/
theorem argmax_fixed_loop_eq (logits : Slice I64) (hne : 0 < logits.val.length) :
    ∃ best : Usize, argmaxAt logits best.val ∧
      sampler.argmax_fixed_loop logits 0#usize 1#usize = ok best := by
  obtain ⟨b, hb, hpost⟩ := WP.spec_imp_exists (argmax_loop_spec logits hne)
  exact ⟨b, hpost, hb⟩

/-! ## The public refusal and success law -/

/-- **Empty-logit refusal.** An empty logit vector returns `Err EmptyLogits` exactly. -/
theorem word_argmax_fixed_empty (logits : Slice I64) (h : logits.val = []) :
    word_argmax_fixed logits =
      ok (core.result.Result.Err sampler.SamplerError.EmptyLogits) := by
  have hem : core.slice.Slice.is_empty logits = ok true := by
    simp [core.slice.Slice.is_empty, Slice.length, h]
  unfold word_argmax_fixed sampler.argmax_fixed
  rw [hem]
  simp

/-- **Argmax law.** On nonempty logits the scan determines the least-index argmax `best`; the
`u32` narrowing returns it when it fits and refuses `LogitsTooLong` otherwise. -/
theorem word_argmax_fixed_spec (logits : Slice I64) (hne : 0 < logits.val.length) :
    ∃ best : Nat, argmaxAt logits best ∧
      (best ≤ U32.max →
        ∃ u : U32, word_argmax_fixed logits = ok (core.result.Result.Ok u) ∧ u.val = best) ∧
      (U32.max < best →
        word_argmax_fixed logits =
          ok (core.result.Result.Err sampler.SamplerError.LogitsTooLong)) := by
  obtain ⟨b, hbloop, hbpost⟩ := WP.spec_imp_exists (argmax_loop_spec logits hne)
  have hne' : logits.val ≠ [] := by
    intro h; rw [h] at hne; simp at hne
  have hem : core.slice.Slice.is_empty logits = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, List.length_eq_zero_iff, hne']
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
    (core.convert.num.ptr_try_from_impls.TryFromU32Usize.try_from.step_spec b)
  have hunf : word_argmax_fixed logits =
      (do
        let best ← sampler.argmax_fixed_loop logits 0#usize 1#usize
        let r ← core.convert.num.ptr_try_from_impls.TryFromU32Usize.try_from best
        core.result.Result.map_err
          sampler.argmax_fixed.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorSamplerError
          r ()) := by
    unfold word_argmax_fixed sampler.argmax_fixed
    rw [hem]
    simp only [bind_tc_ok]
    rw [if_neg (by simp)]
  refine ⟨b.val, hbpost, ?_, ?_⟩
  · intro hle
    obtain ⟨u, hur, huv⟩ := hpost.1 hle
    refine ⟨u, ?_, huv⟩
    rw [hunf, hbloop]
    simp only [bind_tc_ok]
    rw [hr]
    simp only [bind_tc_ok]
    rw [hur]
    simp only [core.result.Result.map_err_ok]
  · intro hgt
    rw [hunf, hbloop]
    simp only [bind_tc_ok]
    rw [hr]
    simp only [bind_tc_ok]
    rw [hpost.2 hgt]
    simp only [core.result.Result.map_err_err]
    rw [sampler.argmax_fixed.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorSamplerError.call_once]
    simp only [bind_tc_ok]

/-! ## The real reading -/

/-- `toReal` preserves the integer order (weakly). -/
theorem toReal_le_toReal {a b : I64} (h : a.val ≤ b.val) : toReal a ≤ toReal b := by
  unfold toReal
  exact div_le_div_of_nonneg_right (by exact_mod_cast h) (le_of_lt q32_pos)

/-- `toReal` is strictly order-preserving. -/
theorem toReal_lt_toReal {a b : I64} (h : a.val < b.val) : toReal a < toReal b := by
  unfold toReal
  exact div_lt_div_of_pos_right (by exact_mod_cast h) q32_pos

/-- The argmax law read over the real logits: no in-range real logit exceeds the winner, and
every earlier real logit is strictly smaller. This is the strict-argmax conclusion of
`Nous.Model.strictWinner_strict_argmax`; the integer gaps are not uniformly bounded, so the
`StrictWinner` margin form is stated only when a margin is supplied. -/
theorem argmaxAt_real (logits : Slice I64) (best : Nat) (h : argmaxAt logits best) :
    (∀ j, j < logits.val.length → toReal (logitAt logits j) ≤ toReal (logitAt logits best)) ∧
    (∀ j, j < best → toReal (logitAt logits j) < toReal (logitAt logits best)) := by
  obtain ⟨_, hmax, hstrict⟩ := h
  exact ⟨fun j hj => toReal_le_toReal (hmax j hj),
    fun j hj => toReal_lt_toReal (hstrict j hj)⟩

/-! ## The attention KV-state constructor

`attn.KvState.new` is the port-map row 5 allocator: five zero-parameter guards, the GQA
divisibility guard, the `fixed_rsqrt` window, and the two checked plane products. The refusals
below are the guard equalities; `word_kv_state_new_geometry` reads the geometry directly off
the `Ok` record. -/

/-- **Zero `n_layer`.** Returns `Err BadGeometry` exactly. -/
theorem word_kv_state_new_layer_zero (n_head n_head_kv head_dim cap : Usize) :
    word_kv_state_new 0#usize n_head n_head_kv head_dim cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.new]

/-- **Zero `n_head`.** Returns `Err BadGeometry` exactly when `n_layer` is nonzero. -/
theorem word_kv_state_new_head_zero (n_layer n_head_kv head_dim cap : Usize)
    (h : n_layer ≠ 0#usize) :
    word_kv_state_new n_layer 0#usize n_head_kv head_dim cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.new, h]

/-- **Zero `n_head_kv`.** Returns `Err BadGeometry` exactly when the earlier guards pass. -/
theorem word_kv_state_new_kv_zero (n_layer n_head head_dim cap : Usize)
    (h1 : n_layer ≠ 0#usize) (h2 : n_head ≠ 0#usize) :
    word_kv_state_new n_layer n_head 0#usize head_dim cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.new, h1, h2]

/-- **Zero `head_dim`.** Returns `Err BadGeometry` exactly when the earlier guards pass. -/
theorem word_kv_state_new_head_dim_zero (n_layer n_head n_head_kv cap : Usize)
    (h1 : n_layer ≠ 0#usize) (h2 : n_head ≠ 0#usize) (h3 : n_head_kv ≠ 0#usize) :
    word_kv_state_new n_layer n_head n_head_kv 0#usize cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.new, h1, h2, h3]

/-- **Zero `cap`.** Returns `Err BadGeometry` exactly when the earlier guards pass. -/
theorem word_kv_state_new_cap_zero (n_layer n_head n_head_kv head_dim : Usize)
    (h1 : n_layer ≠ 0#usize) (h2 : n_head ≠ 0#usize) (h3 : n_head_kv ≠ 0#usize)
    (h4 : head_dim ≠ 0#usize) :
    word_kv_state_new n_layer n_head n_head_kv head_dim 0#usize =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.new, h1, h2, h3, h4]

/-- **GQA divisibility.** A nonzero remainder `n_head % n_head_kv` returns `Err BadGeometry`
exactly when the zero guards pass. -/
theorem word_kv_state_new_not_multiple (nl nh nk hd cap : Usize)
    (h1 : nl ≠ 0#usize) (h2 : nh ≠ 0#usize) (h3 : nk ≠ 0#usize)
    (h4 : hd ≠ 0#usize) (h5 : cap ≠ 0#usize) (h6 : nh.val % nk.val ≠ 0) :
    word_kv_state_new nl nh nk hd cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.new, h1, h2, h3, h4, h5,
    core.num.Usize.is_multiple_of, UScalar.is_multiple_of, h6]

/-- **`Ok`-arm geometry projection.** With the five zero guards and the GQA divisibility guard
passing, any `Ok` result of the extracted constructor carries the geometry of its inputs: the
five record fields `n_layer`, `n_head`, `n_head_kv`, `head_dim`, `cap` equal the arguments.
Each later guard (`UScalar.hcast`, `engine.ONE`, `I64.checked_mul`, `Option.ok_or`, both
`Try.branch`, `fixed_rsqrt`, `map_err`) and each allocation is peeled by case analysis on the
monadic `Result` and on the pure `Option`s; the arithmetic block's `scale` value is the
`fixed_rsqrt` output and is not pinned by the guards. -/
theorem word_kv_state_new_geometry (nl nh nk hd cap : Usize) (s : attn.KvState)
    (h1 : nl ≠ 0#usize) (h2 : nh ≠ 0#usize) (h3 : nk ≠ 0#usize)
    (h4 : hd ≠ 0#usize) (h5 : cap ≠ 0#usize) (h6 : nh.val % nk.val = 0)
    (h : word_kv_state_new nl nh nk hd cap = ok (core.result.Result.Ok s)) :
    s.n_layer = nl ∧ s.n_head = nh ∧ s.n_head_kv = nk ∧ s.head_dim = hd ∧ s.cap = cap := by
  unfold word_kv_state_new attn.KvState.new at h
  rw [if_neg h1, if_neg h2, if_neg h3, if_neg h4, if_neg h5] at h
  have hb : core.num.Usize.is_multiple_of nh nk = ok true := by
    simp [core.num.Usize.is_multiple_of, UScalar.is_multiple_of, h6]
  simp only [hb, bind_tc_ok, reduceIte] at h
  simp only [lift, bind_tc_ok] at h
  by_cases hneg : UScalar.hcast IScalarTy.I64 hd < 0#i64
  · rw [if_pos hneg] at h
    exfalso; have hc := congrArg Result.match h; simp at hc
  · rw [if_neg hneg] at h
    cases hONE : engine.ONE with
    | ret i =>
      rw [hONE, bind_tc_ok] at h
      cases hmul : I64.checked_mul (UScalar.hcast IScalarTy.I64 hd) i with
      | none =>
        rw [hmul] at h
        simp only [core.option.Option.ok_or_none, core.result.Result.Insts.CoreOpsTry.branch,
          bind_tc_ok,
          core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
          core.convert.FromSame.from] at h
        exfalso; have hc := congrArg Result.match h; simp at hc
      | some v =>
        rw [hmul] at h
        simp only [core.option.Option.ok_or_some, core.result.Result.Insts.CoreOpsTry.branch,
          bind_tc_ok] at h
        cases hr1 : fixed_rsqrt.fixed_rsqrt v with
        | ret w =>
          rw [hr1, bind_tc_ok] at h
          cases hw : w with
          | Ok val1 =>
            rw [hw] at h
            simp only [core.result.Result.map_err_ok, bind_tc_ok] at h
            cases hnk : nk.checked_mul cap with
            | none =>
              simp only [hnk] at h
              exfalso; have hc := congrArg Result.match h; simp at hc
            | some x =>
              simp only [hnk] at h
              cases hx : x.checked_mul hd with
              | none =>
                simp only [hx] at h
                exfalso; have hc := congrArg Result.match h; simp at hc
              | some plane =>
                simp only [hx] at h
                cases hblank : alloc.vec.from_elem core.clone.CloneI64 0#i64 plane with
                | ret blank =>
                  rw [hblank, bind_tc_ok] at h
                  cases hv : alloc.vec.CloneVec.clone core.clone.CloneI64 blank with
                  | ret vv =>
                    rw [hv, bind_tc_ok] at h
                    cases hk : alloc.vec.from_elem
                        (core.clone.CloneallocvecVec core.clone.CloneI64) vv nl with
                    | ret kk =>
                      rw [hk, bind_tc_ok] at h
                      cases hv1 : alloc.vec.from_elem
                          (core.clone.CloneallocvecVec core.clone.CloneI64) blank nl with
                      | ret vv1 =>
                        rw [hv1, bind_tc_ok] at h
                        cases hv2 : alloc.vec.from_elem core.clone.CloneUsize 0#usize nl with
                        | ret vv2 =>
                          rw [hv2, bind_tc_ok] at h
                          have hs : s = { n_layer := nl, n_head := nh, n_head_kv := nk, head_dim := hd, cap := cap, scale := val1, occupied := vv2, k := kk, v := vv1 } := by
                            apply (core.result.Result.Ok.injEq _ _).mp
                            exact (Result.ok_injective h).symm
                          subst hs
                          simp
                        | vis e k =>
                          rw [hv2] at h
                          exfalso; have hc := congrArg Result.match h; simp at hc
                        | div =>
                          rw [hv2] at h
                          exfalso; have hc := congrArg Result.match h; simp at hc
                      | vis e k =>
                        rw [hv1] at h
                        exfalso; have hc := congrArg Result.match h; simp at hc
                      | div =>
                        rw [hv1] at h
                        exfalso; have hc := congrArg Result.match h; simp at hc
                    | vis e k =>
                      rw [hk] at h
                      exfalso; have hc := congrArg Result.match h; simp at hc
                    | div =>
                      rw [hk] at h
                      exfalso; have hc := congrArg Result.match h; simp at hc
                  | vis e k =>
                    rw [hv] at h
                    exfalso; have hc := congrArg Result.match h; simp at hc
                  | div =>
                    rw [hv] at h
                    exfalso; have hc := congrArg Result.match h; simp at hc
                | vis e k =>
                  rw [hblank] at h
                  exfalso; have hc := congrArg Result.match h; simp at hc
                | div =>
                  rw [hblank] at h
                  exfalso; have hc := congrArg Result.match h; simp at hc
          | Err e =>
            rw [hw] at h
            simp only [core.result.Result.map_err_err, bind_tc_ok,
              attn.KvState.new.closure.Insts.CoreOpsFunctionFnOnceTupleRsqrtErrorAttnError.call_once,
              core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
              core.convert.FromSame.from] at h
            exfalso; have hc := congrArg Result.match h; simp at hc
        | vis e k =>
          rw [hr1] at h
          exfalso; have hc := congrArg Result.match h; simp at hc
        | div =>
          rw [hr1] at h
          exfalso; have hc := congrArg Result.match h; simp at hc
    | vis e k =>
      rw [hONE] at h
      exfalso; have hc := congrArg Result.match h; simp at hc
    | div =>
      rw [hONE] at h
      exfalso; have hc := congrArg Result.match h; simp at hc

/-! ## The attention KV-state append -/

/-- **Layer refusal.** A layer index at or past the layer count returns `Err LayerOutOfRange`
exactly and leaves the state unchanged. -/
theorem word_kv_append_layer_refusal (self : attn.KvState) (layer : Usize)
    (k_rows v_rows : Slice I64) (h : self.n_layer ≤ layer) :
    word_kv_append self layer k_rows v_rows =
      ok (core.result.Result.Err (attn.AttnError.LayerOutOfRange layer self.n_layer), self) := by
  unfold word_kv_append attn.KvState.append
  rw [if_pos h]

/-- **Width refusal (keys).** With an in-range layer and the checked width, a `k_rows` block that
is not `n_head_kv · head_dim` wide returns `Err RowLength width got` exactly and leaves the state
unchanged. -/
theorem word_kv_append_row_length_k (self : attn.KvState) (layer : Usize)
    (k_rows v_rows : Slice I64) (w : Usize)
    (hlayer : ¬ self.n_layer ≤ layer)
    (hw_eq : self.n_head_kv.checked_mul self.head_dim = some w)
    (hk : ¬ Slice.len k_rows = w) :
    word_kv_append self layer k_rows v_rows =
      ok (core.result.Result.Err (attn.AttnError.RowLength w (Slice.len k_rows)), self) := by
  have hne : (Slice.len k_rows != w) = true := by simpa using hk
  unfold word_kv_append attn.KvState.append
  rw [if_neg hlayer, hw_eq]
  simp only [lift, bind_tc_ok, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch]
  rw [hne]
  simp

/-- **Width refusal (values).** Same as the key arm with matching `k_rows` and short `v_rows`. -/
theorem word_kv_append_row_length_v (self : attn.KvState) (layer : Usize)
    (k_rows v_rows : Slice I64) (w : Usize)
    (hlayer : ¬ self.n_layer ≤ layer)
    (hw_eq : self.n_head_kv.checked_mul self.head_dim = some w)
    (hk : Slice.len k_rows = w)
    (hv : ¬ Slice.len v_rows = w) :
    word_kv_append self layer k_rows v_rows =
      ok (core.result.Result.Err (attn.AttnError.RowLength w (Slice.len v_rows)), self) := by
  have hnek : (Slice.len k_rows != w) = false := by simp [hk]
  have hnev : (Slice.len v_rows != w) = true := by simpa using hv
  unfold word_kv_append attn.KvState.append
  rw [if_neg hlayer, hw_eq]
  simp only [lift, bind_tc_ok, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch]
  rw [hnek, hnev]
  simp

/-- **Capacity refusal.** With an in-range layer, the checked width, both row blocks at that
width and a full layer (`cap ≤ occupied[layer]`), append returns `Err CapacityExceeded cap`
exactly and leaves the state unchanged. The guards precede every allocation. -/
theorem word_kv_append_capacity_refusal (self : attn.KvState) (layer : Usize)
    (k_rows v_rows : Slice I64) (w : Usize)
    (hlayer : ¬ self.n_layer ≤ layer)
    (hw_eq : self.n_head_kv.checked_mul self.head_dim = some w)
    (hk : Slice.len k_rows = w) (hv : Slice.len v_rows = w)
    (hbnd : layer.val < self.occupied.length)
    (hpos : self.cap.val ≤ (self.occupied.val[layer.val]).val) :
    word_kv_append self layer k_rows v_rows =
      ok (core.result.Result.Err (attn.AttnError.CapacityExceeded self.cap), self) := by
  have hnek : (Slice.len k_rows != w) = false := by simp [hk]
  have hnev : (Slice.len v_rows != w) = false := by simp [hv]
  obtain ⟨pos, hpos_eq, hpos_val⟩ :=
    WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec self.occupied layer hbnd)
  have hge : self.cap ≤ pos := by
    rw [UScalar.le_equiv, hpos_val]
    exact hpos
  unfold word_kv_append attn.KvState.append
  rw [if_neg hlayer, hw_eq]
  simp only [lift, bind_tc_ok, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch]
  rw [hnek, hnev]
  simp only [Bool.false_eq_true, if_false]
  rw [alloc.vec.Vec.index_slice_index, hpos_eq]
  simp only [bind_tc_ok]
  rw [if_pos hge]

/-- **Success law.** With an in-range layer, the checked width, both row blocks at that width, a
non-full layer and the checked position arithmetic succeeding, `append` returns `pos = occupied[layer]`,
increments `occupied[layer]` to `v3 = pos + 1`, and writes the two `n_head_kv·head_dim`-wide row
blocks at `[pos·width, (pos+1)·width)` into `self.k` and `self.v`. The two range writes are the
`setSlice!` at the checked product `v1`; the five extracted `index_mut` threads
(`index_mut_back`…`index_mut_back4`) are reduced by their `@[step]` specs (`Vec.set` for the
addressable lane, `Vec.setSlice!` for the range) before the record is reassembled. -/
theorem word_kv_append_ok (self : attn.KvState) (layer : Usize)
    (k_rows v_rows : Slice I64) (w v1 v2 v3 : Usize)
    (hlayer : ¬ self.n_layer ≤ layer)
    (hw_eq : self.n_head_kv.checked_mul self.head_dim = some w)
    (hk : Slice.len k_rows = w) (hv : Slice.len v_rows = w)
    (hbnd : layer.val < self.occupied.length)
    (hcap : ¬ self.cap ≤ (self.occupied.val[layer.val]).val)
    (hbndk : layer.val < self.k.length) (hbndv : layer.val < self.v.length)
    (hmul : (self.occupied.val[layer.val]).checked_mul w = some v1)
    (hadd : v1.checked_add w = some v2)
    (hadd3 : (self.occupied.val[layer.val]).checked_add 1#usize = some v3)
    (hkend : v2.val ≤ (self.k.val[layer.val]).length)
    (hvend : v2.val ≤ (self.v.val[layer.val]).length) :
    word_kv_append self layer k_rows v_rows =
      ok (core.result.Result.Ok (self.occupied.val[layer.val]),
        { self with
            occupied := self.occupied.set layer v3,
            k := self.k.set layer
              (alloc.vec.Vec.setSlice! (self.k.val[layer.val]) v1.val k_rows.val),
            v := self.v.set layer
              (alloc.vec.Vec.setSlice! (self.v.val[layer.val]) v1.val v_rows.val) }) ∧
      v3.val = (self.occupied.val[layer.val]).val + 1 := by
  constructor
  · unfold word_kv_append attn.KvState.append
    rw [if_neg hlayer, hw_eq]
    have hnek : (Slice.len k_rows != w) = false := by simp [hk]
    have hnev : (Slice.len v_rows != w) = false := by simp [hv]
    simp only [lift, bind_tc_ok, core.option.Option.ok_or_some,
      core.result.Result.Insts.CoreOpsTry.branch]
    rw [hnek, hnev]
    simp only [Bool.false_eq_true, if_false]
    obtain ⟨pos, hpos_eq, hpos_val⟩ :=
      WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec self.occupied layer hbnd)
    rw [alloc.vec.Vec.index_slice_index, hpos_eq]
    simp only [bind_tc_ok]
    subst hpos_val
    have hcap' : ¬ self.cap ≤ self.occupied.val[layer.val] := by
      rw [UScalar.le_equiv]; exact hcap
    rw [if_neg hcap']
    have hv2 : v2.val = v1.val + w.val := by
      have h := Usize.checked_add_bv_spec v1 w
      simp only [hadd] at h
      exact h.2.1
    have hwv2 : v2.val - v1.val = w.val := by omega
    simp only [hmul, hadd, hadd3, core.option.Option.ok_or_some, bind_tc_ok]
    have hle : v1 ≤ v2 := by rw [UScalar.le_equiv]; omega
    have hklen : k_rows.length = w.val := by rw [← Slice.len_val, hk]
    have hvlen : v_rows.length = w.val := by rw [← Slice.len_val, hv]
    -- k: addressable lane
    obtain ⟨p1, hp1_eq, hp1_spec⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.index_mut_usize_spec self.k layer hbndk)
    rcases p1 with ⟨kplane, kback⟩
    rw [alloc.vec.Vec.index_mut_slice_index, hp1_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [hp1_spec.1]
    -- k: row range
    obtain ⟨p2, hp2_eq, hp2_spec⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.index_mut_Range_spec (self.k.val[layer.val])
        { start := v1, «end» := v2 } hle hkend)
    rcases p2 with ⟨kslice, kback1⟩
    rw [hp2_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
    have hlenk : kslice.length = k_rows.length := by
      rw [hp2_spec.2.1, hwv2, hklen]
    have hcsk : core.slice.Slice.copy_from_slice core.marker.CopyI64 kslice k_rows = ok k_rows := by
      obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists
        (core.slice.Slice.copy_from_slice.step_spec core.marker.CopyI64 kslice k_rows hlenk)
      rw [hy, hyv]
    rw [hcsk]
    simp only [bind_tc_ok]
    -- v: addressable lane
    obtain ⟨q1, hq1_eq, hq1_spec⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.index_mut_usize_spec self.v layer hbndv)
    rcases q1 with ⟨vplane, vback⟩
    rw [alloc.vec.Vec.index_mut_slice_index, hq1_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [hq1_spec.1]
    -- v: row range
    obtain ⟨q2, hq2_eq, hq2_spec⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.index_mut_Range_spec (self.v.val[layer.val])
        { start := v1, «end» := v2 } hle hvend)
    rcases q2 with ⟨vslice, vback3⟩
    rw [hq2_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
    have hlenv : vslice.length = v_rows.length := by
      rw [hq2_spec.2.1, hwv2, hvlen]
    have hcsv : core.slice.Slice.copy_from_slice core.marker.CopyI64 vslice v_rows = ok v_rows := by
      obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists
        (core.slice.Slice.copy_from_slice.step_spec core.marker.CopyI64 vslice v_rows hlenv)
      rw [hy, hyv]
    rw [hcsv]
    simp only [bind_tc_ok]
    -- occupied: addressable lane
    obtain ⟨p4, hp4_eq, hp4_spec⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.index_mut_usize_spec self.occupied layer hbnd)
    rcases p4 with ⟨occval, occback⟩
    rw [alloc.vec.Vec.index_mut_slice_index, hp4_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [hp2_spec.2.2 k_rows, hp1_spec.2, hq2_spec.2.2 v_rows, hq1_spec.2, hp4_spec.2]
  · have h := Usize.checked_add_bv_spec (self.occupied.val[layer.val]) 1#usize
    simp only [hadd3] at h
    exact h.2.1

/-! ## Residual obligations (not established here)

The following row of the task statement is a named, unproven obligation at grade `anchor`.

* `word_kv_state_new` `scale` field: `s.scale = fixed_rsqrt (hd·2^32)`. The five geometry fields
  are proved by `word_kv_state_new_geometry`; pinning `scale` also needs `UScalar.hcast .I64 hd`
  to preserve the value (which uses `hd.val ≤ I64.max`, not implied by `hd ≠ 0`) and
  `I64.checked_mul hd_signed engine.ONE = some (hd.val·2^32)`.
-/

end

end apeiron_word
