/-
Refinement of the extracted engine sampler argmax and the attention KV-state
constructor / append.

The Rust bodies are `src/engine/sampler.rs` (`argmax_fixed`) and `src/engine/attn.rs`
(`KvState::with_carrier`, `KvState::append`); `i64` Q32.32 logits and Q32.32 KV planes. The public
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
* The six `word_kv_state_new` guard theorems are exact equalities naming `Err BadGeometry` and
  `word_kv_state_new_geometry` reads the five geometry fields off the `Ok` record, both against
  the `def` body of `attn.KvState.with_carrier` in `ApeironWord.lean` at sha256 `c16a68f0…`. The
  `scale` and `storage` fields are not pinned, and `attn.alloc_f32_storage` is in the cone of
  every statement about `word_kv_state_new`; see the residual obligations below.
* `attn.KvState.append` tests the layer index first; the layer-out-of-range, both width
  (`RowLength`) and capacity (`CapacityExceeded`) refusals are exact equalities leaving the
  state unchanged, and `word_kv_append_ok` is the success law: the returned position is
  `occupied[layer]`, it increments that lane to `pos + 1`, and it writes the two
  `n_head_kv·head_dim`-wide row blocks at `[pos·width, (pos+1)·width)`. `KvState` stores its
  planes in `storage : attn.KvStorage`, so the success law is stated for the `Q32` branch under
  a hypothesis `self.storage = attn.KvStorage.Q32 kk vv` and writes through `kk`/`vv`.

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

`attn.KvState.with_carrier` is the port-map row 5 allocator: five zero-parameter guards, the GQA
divisibility guard, the `fixed_rsqrt` window, and the two checked plane products. The refusals
below are the guard equalities; `word_kv_state_new_geometry` reads the geometry directly off
the `Ok` record. -/

/-! ### The constructor is a `def`, not an axiom

`ApeironWord.lean` at sha256 `c16a68f0…` translates `attn.KvState.with_carrier`
(`src/engine/attn.rs` 766:4-826:5) as a `def` whose body is the `do`-block of the Rust
constructor, and `word_kv_state_new` is that `def` applied to
`mixer.Carrier.Q32`. Each statement below is therefore an equation about a
reducible term, not about an uninterpreted constant.

The `Carrier::Q32` arm — the only arm `word_kv_state_new` reaches — is translated: the
four `alloc.vec.from_elem` / `alloc.vec.CloneVec.clone` allocations of the zeroed plane
are in the block, and no statement below reduces the `F32` arm. The `Carrier::F32` arm is the
admitted boundary item `attn.alloc_f32_storage` (`src/engine/attn.rs` 118:1-123:2); it is a
free constant of `word_kv_state_new`, so it is in the axiom cone of every statement about
that function — see the cone row under residual obligations. -/

/-- **Zero `n_layer`.** Returns `Err BadGeometry` exactly. -/
theorem word_kv_state_new_layer_zero (n_head n_head_kv head_dim cap : Usize) :
    word_kv_state_new 0#usize n_head n_head_kv head_dim cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.with_carrier]

/-- **Zero `n_head`.** Returns `Err BadGeometry` exactly when `n_layer` is nonzero. -/
theorem word_kv_state_new_head_zero (n_layer n_head_kv head_dim cap : Usize)
    (h : n_layer ≠ 0#usize) :
    word_kv_state_new n_layer 0#usize n_head_kv head_dim cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.with_carrier, h]

/-- **Zero `n_head_kv`.** Returns `Err BadGeometry` exactly when the earlier guards pass. -/
theorem word_kv_state_new_kv_zero (n_layer n_head head_dim cap : Usize)
    (h1 : n_layer ≠ 0#usize) (h2 : n_head ≠ 0#usize) :
    word_kv_state_new n_layer n_head 0#usize head_dim cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.with_carrier, h1, h2]

/-- **Zero `head_dim`.** Returns `Err BadGeometry` exactly when the earlier guards pass. -/
theorem word_kv_state_new_head_dim_zero (n_layer n_head n_head_kv cap : Usize)
    (h1 : n_layer ≠ 0#usize) (h2 : n_head ≠ 0#usize) (h3 : n_head_kv ≠ 0#usize) :
    word_kv_state_new n_layer n_head n_head_kv 0#usize cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.with_carrier, h1, h2, h3]

/-- **Zero `cap`.** Returns `Err BadGeometry` exactly when the earlier guards pass. -/
theorem word_kv_state_new_cap_zero (n_layer n_head n_head_kv head_dim : Usize)
    (h1 : n_layer ≠ 0#usize) (h2 : n_head ≠ 0#usize) (h3 : n_head_kv ≠ 0#usize)
    (h4 : head_dim ≠ 0#usize) :
    word_kv_state_new n_layer n_head n_head_kv head_dim 0#usize =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.with_carrier, h1, h2, h3, h4]

/-- **GQA divisibility.** A nonzero remainder `n_head % n_head_kv` returns `Err BadGeometry`
exactly when the zero guards pass. The model reads `n.is_multiple_of(m)` as
`n.val % m.val == 0`, so `h6` decides the guard. -/
theorem word_kv_state_new_not_multiple (nl nh nk hd cap : Usize)
    (h1 : nl ≠ 0#usize) (h2 : nh ≠ 0#usize) (h3 : nk ≠ 0#usize)
    (h4 : hd ≠ 0#usize) (h5 : cap ≠ 0#usize) (h6 : nh.val % nk.val ≠ 0) :
    word_kv_state_new nl nh nk hd cap =
      ok (core.result.Result.Err attn.AttnError.BadGeometry) := by
  simp [word_kv_state_new, attn.KvState.with_carrier, h1, h2, h3, h4, h5,
    core.num.Usize.is_multiple_of, UScalar.is_multiple_of, h6]

/-- **`Ok`-arm geometry projection.** With the five zero guards and the GQA divisibility guard
passing, any `Ok` result of the extracted constructor carries the geometry of its inputs: the
five record fields `n_layer`, `n_head`, `n_head_kv`, `head_dim`, `cap` equal the arguments.

The proof peels the translated `do`-block from the outside in. `Result` is `ITree`, so a
bind whose continuation produces `ok y` can only have bound an `ok`: `peel` names that bound
value and refutes the `vis` and `div` alternatives (`bind_tc_vis`/`bind_tc_div` cannot
produce an `ok`). The `Option` binds are split on their two constructors; each `none` branch
yields `ok (Err _)` and is refuted by `h`. The `hd` sign guard is the one guard the hypotheses
do not decide, and its `Err ScaleOutOfRange` arm is refuted by `h` as well. What remains is the
`Ok` record literal, whose five geometry fields are the arguments. -/
theorem word_kv_state_new_geometry (nl nh nk hd cap : Usize) (s : attn.KvState)
    (h1 : nl ≠ 0#usize) (h2 : nh ≠ 0#usize) (h3 : nk ≠ 0#usize)
    (h4 : hd ≠ 0#usize) (h5 : cap ≠ 0#usize) (h6 : nh.val % nk.val = 0)
    (h : word_kv_state_new nl nh nk hd cap = ok (core.result.Result.Ok s)) :
    s.n_layer = nl ∧ s.n_head = nh ∧ s.n_head_kv = nk ∧ s.head_dim = hd ∧ s.cap = cap := by
  -- the GQA guard `n_head % n_head_kv == 0` is decided by `h6`
  have hmul : core.num.Usize.is_multiple_of nh nk = ok true := by
    simp [core.num.Usize.is_multiple_of, UScalar.is_multiple_of, h6]
  -- the `?` desugaring turns a `Result Err e` into `ok (Err e)`, never into an `Ok`
  have hres (e : attn.AttnError) :
      core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
        attn.KvState (core.convert.FromSame attn.AttnError)
        (core.result.Result.Err e) = ok (core.result.Result.Err e) := by
    simp [core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
      core.convert.FromSame.from]
  -- a `Result` bind inside a `do`-block that is `ok y` bound an `ok`; `bind_tc_vis` and
  -- `bind_tc_div` keep a `vis`/`div` bind a `vis`/`div`, so those cases contradict `hb`
  have peel {α β : Type} {m : Result α} {K : α → Result β} {y : β}
      (hb : (do let x ← m; K x) = ok y) : ∃ v, m = ok v := by
    cases hm : m.match with
    | ok v => exact ⟨v, Result.match.isOk.mp hm⟩
    | vis eff k => exact absurd hb (by rw [Result.match.isVis.mp hm] at hb; simp at hb)
    | div => exact absurd hb (by rw [Result.match.isDiv.mp hm] at hb; simp at hb)
  unfold word_kv_state_new attn.KvState.with_carrier at h
  rw [if_neg h1, if_neg h2, if_neg h3, if_neg h4, if_neg h5, hmul] at h
  simp only [bind_tc_ok, if_true] at h
  -- the sign guard on `UScalar.hcast .I64 hd`: its `Err` arm is not the `Ok` of `h`
  by_cases hs : UScalar.hcast IScalarTy.I64 hd < 0#i64
  · simp only [lift, bind_tc_ok] at h
    rw [if_pos hs] at h
    simp only [Result.ok.injEq] at h
    simp at h
  · simp only [lift, bind_tc_ok, engine.ONE] at h
    rw [if_neg hs] at h
    -- `engine.ONE` is the `Result`-valued shift `1#i64 <<< engine.FIXED_FRAC_BITS`
    obtain ⟨ione, hione⟩ := peel h
    rw [hione, bind_tc_ok] at h
    -- `I64.checked_mul (UScalar.hcast .I64 hd) 2^32`
    cases hcm : I64.checked_mul (UScalar.hcast IScalarTy.I64 hd) ione with
    | none =>
      simp only [hcm, core.option.Option.ok_or_none, bind_tc_ok,
        core.result.Result.Insts.CoreOpsTry.branch] at h
      rw [hres attn.AttnError.ScaleOutOfRange] at h
      simp at h
    | some v1 =>
      simp only [hcm, core.option.Option.ok_or_some, bind_tc_ok,
        core.result.Result.Insts.CoreOpsTry.branch] at h
      -- `fixed_rsqrt` returns `Result (Result I64 RsqrtError)`; peel both layers
      obtain ⟨v2, hv2⟩ := peel h
      rw [hv2, bind_tc_ok] at h
      cases hv2 : v2 with
      | Err e3 =>
        -- `map_err` runs the closure, which is `Err ScaleOutOfRange` on any `RsqrtError`
        simp only [hv2, core.result.Result.map_err, bind_tc_ok,
          attn.KvState.with_carrier.closure.Insts.CoreOpsFunctionFnOnceTupleRsqrtErrorAttnError.call_once] at h
        rw [hres attn.AttnError.ScaleOutOfRange] at h
        simp only [Result.ok.injEq] at h
        simp at h
      | Ok v3 =>
        simp only [hv2, core.result.Result.map_err, bind_tc_ok] at h
        -- the first checked plane product `n_head_kv · cap`
        cases hk : Usize.checked_mul nk cap with
        | none =>
          simp only [hk] at h
          simp only [Result.ok.injEq] at h
          simp at h
        | some x =>
          simp only [hk] at h
          -- the second checked plane product `(n_head_kv · cap) · head_dim`
          cases hp : Usize.checked_mul x hd with
          | none =>
            simp only [hp] at h
            simp only [Result.ok.injEq] at h
            simp at h
          | some plane =>
            simp only [hp] at h
            -- the `Carrier::Q32` allocation, then the occupancy vector
            obtain ⟨storage, hstorage⟩ := peel h
            rw [hstorage, bind_tc_ok] at h
            obtain ⟨occ, hocc⟩ := peel h
            rw [hocc, bind_tc_ok] at h
            -- the `Ok` record literal: its five geometry fields are the arguments
            simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at h
            cases h
            exact ⟨rfl, rfl, rfl, rfl, rfl⟩

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
blocks at `[pos·width, (pos+1)·width)` into the two `storage` row-block planes `kk` and `vv`. The
two range writes are the `setSlice!` at the checked product `v1`; the five extracted `index_mut`
threads (`index_mut_back`…`index_mut_back4`) are reduced by their `@[step]` specs (`Vec.set` for
the addressable lane, `Vec.setSlice!` for the range) before the record is reassembled. The
`match self.storage with` in the extracted body is reduced by `hstor` to the `Q32` branch, which
is the branch whose row copy is `Slice.copy_from_slice`; the `F32` branch converts elementwise
through `attn.KvState.append_loop0`/`append_loop1` and is not covered here. -/
theorem word_kv_append_ok (self : attn.KvState) (layer : Usize)
    (k_rows v_rows : Slice I64) (w v1 v2 v3 : Usize)
    (kk vv : alloc.vec.Vec (alloc.vec.Vec I64))
    (hlayer : ¬ self.n_layer ≤ layer)
    (hw_eq : self.n_head_kv.checked_mul self.head_dim = some w)
    (hk : Slice.len k_rows = w) (hv : Slice.len v_rows = w)
    (hbnd : layer.val < self.occupied.length)
    (hcap : ¬ self.cap ≤ (self.occupied.val[layer.val]).val)
    (hstor : self.storage = attn.KvStorage.Q32 kk vv)
    (hbndk : layer.val < kk.length) (hbndv : layer.val < vv.length)
    (hmul : (self.occupied.val[layer.val]).checked_mul w = some v1)
    (hadd : v1.checked_add w = some v2)
    (hadd3 : (self.occupied.val[layer.val]).checked_add 1#usize = some v3)
    (hkend : v2.val ≤ (kk.val[layer.val]).length)
    (hvend : v2.val ≤ (vv.val[layer.val]).length) :
    word_kv_append self layer k_rows v_rows =
      ok (core.result.Result.Ok (self.occupied.val[layer.val]),
        { self with
            occupied := self.occupied.set layer v3,
            storage := attn.KvStorage.Q32
              (kk.set layer
                (alloc.vec.Vec.setSlice! (kk.val[layer.val]) v1.val k_rows.val))
              (vv.set layer
                (alloc.vec.Vec.setSlice! (vv.val[layer.val]) v1.val v_rows.val)) }) ∧
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
    -- reduce `match self.storage with` to the Q32 branch, so the row-block planes are `kk`/`vv`
    simp only [hstor]
    have hle : v1 ≤ v2 := by rw [UScalar.le_equiv]; omega
    have hklen : k_rows.length = w.val := by rw [← Slice.len_val, hk]
    have hvlen : v_rows.length = w.val := by rw [← Slice.len_val, hv]
    -- k: addressable lane
    obtain ⟨p1, hp1_eq, hp1_spec⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.index_mut_usize_spec kk layer hbndk)
    rcases p1 with ⟨kplane, kback⟩
    rw [alloc.vec.Vec.index_mut_slice_index, hp1_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [hp1_spec.1]
    -- k: row range
    obtain ⟨p2, hp2_eq, hp2_spec⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.index_mut_Range_spec (kk.val[layer.val])
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
      (alloc.vec.Vec.index_mut_usize_spec vv layer hbndv)
    rcases q1 with ⟨vplane, vback⟩
    rw [alloc.vec.Vec.index_mut_slice_index, hq1_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [hq1_spec.1]
    -- v: row range
    obtain ⟨q2, hq2_eq, hq2_spec⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.index_mut_Range_spec (vv.val[layer.val])
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

The following rows are named, unproven obligations. None is discharged by this module.

* **Constructor `scale` field (grade `anchor`).** `word_kv_state_new_geometry` pins the five
  geometry fields; the `scale` field is `s.scale = fixed_rsqrt (hd·2^32)`, whose pinning also needs
  `UScalar.hcast .I64 hd` to preserve the value (which uses `hd.val ≤ I64.max`, not implied by
  `hd ≠ 0`) and `I64.checked_mul (UScalar.hcast .I64 hd) engine.ONE = some (hd.val·2^32)`. The
  guard order also matters: the `hd` sign guard precedes the `fixed_rsqrt` window, so a `scale`
  statement needs the guard discharged first.
* **Constructor `storage` field (grade `anchor`).** The `Carrier::Q32` allocation value
  (`alloc.vec.from_elem` / `alloc.vec.CloneVec.clone` at the plane length) is not pinned here;
  `word_kv_state_new_geometry` peels it as an opaque `Result` because the geometry fields do not
  depend on it.
* **`attn.KvState.append` `F32` branch.** The elementwise conversion loops
  `attn.KvState.append_loop0`/`append_loop1` and their `bad` flags are unrefined, and
  `mixer.f32_carrier.q32_to_f32` / `core.f32.F32.is_finite` are admitted boundary items, so the
  F32 success law is outside the closed cone.
* **Axiom cone of the seven constructor theorems.** `#print axioms` on each of the six guard
  theorems and on `word_kv_state_new_geometry` reports
  `[propext, Classical.choice, Quot.sound, apeiron_word.attn.alloc_f32_storage]`. The extra item
  is the admitted `Carrier::F32` allocation: it is a free constant of `word_kv_state_new`
  itself, so it appears in the cone of any theorem whose statement mentions `word_kv_state_new`,
  including statements that never reduce the `F32` arm. `word_kv_append_ok` has the same shape
  through `mixer.f32_carrier.q32_to_f32` and `core.f32.F32.is_finite`. Whether these boundary
  items are inside the audited cone is `AxiomAudit/Remaining.lean`'s decision, not this module's.
-/

end

end apeiron_word
