/-
Reachability of `attn.KvState` from `KvState::new` + `KvState::append`
(`APEIRON-MODEL-W4-AI12`).

`attn_domain_v2` (`Numeric.AttnSoftmaxPerturb`) is labelled `anchor`: its structural conjuncts
(geometry, occupancy, the load-time scale, the plane lengths) are stated as hypotheses, and only a
hand-built witness (`attn_domain_v2_witness`) is exhibited. This module supplies the missing
*reachable-state* invariant:

* `KvReachable`: the inductive closure of the states `word_kv_state_new` returns and of the states
  `word_kv_append` returns from a reachable state (the success law `word_kv_append_ok`);
* `KvInv`: the structural invariant of a reachable state — nonzero geometry, the GQA divisibility,
  `occupied.length = n_layer`, `occupied[l] ≤ cap`, the two planes `n_head_kv·cap·head_dim` long
  per layer, and `scale = word_fixed_rsqrt (head_dim·2^32)`;
* `KvReachable_inv`: every reachable state satisfies `KvInv` (induction);
* `attn_domain_v2_structural`: on a reachable state the structural conjuncts of `attn_domain_v2`
  hold; the three data windows remain hypotheses, so `reachable ∧ data windows ⇒ attn_domain_v2`;
* `attn_head_refines_reachable`: the conclusion of `attn_head_refines_v2_proved` on reachable
  states.

The `append` constructor carries the preconditions of a successful append: the layer is in range,
the two blocks are `n_head_kv·head_dim` wide, and the layer holds a live position `pos < cap`.
`word_kv_append_ok` (`Numeric.EngineSamplerRefinement`) is the bridge from the extracted
`attn.KvState.append` to the state such a step yields; inverting the extracted do-block for an
arbitrary raw equality `word_kv_append … = ok (Ok pos, s')` is the residual obligation.

Trust boundary: reaches the extracted `word_kv_state_new` / `word_kv_append`, the proved success and
geometry laws of `Numeric.EngineSamplerRefinement`, and the denotations of
`Numeric.EngineAttnRefinement`. It does not cover rustc/LLVM codegen of `src/engine/attn.rs`, the
kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.AttnHeadRefines
import Numeric.EngineSamplerRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Extracted constants and checked arithmetic -/

/-- The extracted `engine.ONE` evaluates to `2^32`. -/
private theorem kv_engine_ONE_eq : engine.ONE = ok (4294967296#i64) := by
  have h : engine.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
    unfold engine.ONE
    rw [engine.FIXED_FRAC_BITS]
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- `Usize.checked_mul` succeeds with the exact product inside `usize`. -/
private theorem usize_checked_mul_some (x y : Usize) (h : x.val * y.val ≤ Usize.max) :
    ∃ z, Usize.checked_mul x y = some z ∧ z.val = x.val * y.val := by
  have hspec := Usize.checked_mul_bv_spec x y
  cases hc : Usize.checked_mul x y with
  | none => rw [hc] at hspec; exact absurd h (not_le.mpr hspec)
  | some z => rw [hc] at hspec; exact ⟨z, rfl, hspec.2.1⟩

/-- `Usize.checked_add` succeeds with the exact sum inside `usize`. -/
private theorem usize_checked_add_some (x y : Usize) (h : x.val + y.val ≤ Usize.max) :
    ∃ z, Usize.checked_add x y = some z ∧ z.val = x.val + y.val := by
  have hspec := Usize.checked_add_bv_spec x y
  cases hc : Usize.checked_add x y with
  | none => rw [hc] at hspec; exact absurd h (not_le.mpr hspec)
  | some z => rw [hc] at hspec; exact ⟨z, rfl, hspec.2.1⟩

/-- `x.checked_add 1#usize = some z` gives `z.val = x.val + 1`. -/
private theorem usize_checked_add_one_val {x z : Usize}
    (h : x.checked_add 1#usize = some z) : z.val = x.val + 1 := by
  have hs := Usize.checked_add_bv_spec x 1#usize
  simp only [h] at hs
  rw [hs.2.1]
  norm_num

/-- A nonzero `usize` has a positive value. -/
private theorem usize_pos_of_ne {x : Usize} (h : x ≠ 0#usize) : 0 < x.val := by
  by_contra hc
  have h0 : x.val = 0 := by omega
  exact h (UScalar.eq_of_val_eq (by rw [h0]; norm_num))

/-- The `i64` heterogeneous cast of a `usize` with nonnegative result is value-preserving. -/
private theorem hcast_I64_val_of_nonneg {x : Usize}
    (h : 0 ≤ (UScalar.hcast IScalarTy.I64 x).val) :
    (UScalar.hcast IScalarTy.I64 x).val = (x.val : ℤ) := by
  have hi64 : IScalarTy.I64.numBits = 64 := rfl
  rw [UScalar.hcast_val_eq, hi64] at h ⊢
  have hxlt : (x.val : ℤ) < (2 : ℤ) ^ 64 := by
    have hbnd := UScalar.hBounds x
    rw [show UScalarTy.Usize.numBits = System.Platform.numBits from rfl] at hbnd
    rcases System.Platform.numBits_eq with h32 | h64
    · rw [h32] at hbnd
      have : x.val < 2 ^ 64 := by omega
      exact_mod_cast this
    · rw [h64] at hbnd; exact_mod_cast hbnd
  have hmod : ((x.val : ℤ) % ((2 ^ 64 : ℕ) : ℤ)) = (x.val : ℤ) :=
    Int.emod_eq_of_lt (Int.natCast_nonneg _) (by exact_mod_cast hxlt)
  rw [Int.bmod_def, hmod] at h ⊢
  by_cases hhalf : (x.val : ℤ) < (((2 ^ 64 : ℕ) : ℤ) + 1) / 2
  · rw [if_pos hhalf] at h ⊢
  · rw [if_neg hhalf] at h ⊢
    exfalso
    omega

/-! ## Clone identities used by `KvState::new` -/

private theorem kv_clone_i64_zero : core.clone.CloneI64.clone (0#i64) = ok (0#i64) := by simp

private theorem kv_clone_usize_zero : core.clone.CloneUsize.clone (0#usize) = ok (0#usize) := by simp

/-- Cloning an `i64` vector with the identity element clone returns the same value list. -/
private theorem clonevec_val {v v' : alloc.vec.Vec I64}
    (h : alloc.vec.CloneVec.clone core.clone.CloneI64 v = ok v') : v'.val = v.val := by
  have hspec := Slice.clone_spec (clone := core.clone.CloneI64.clone) (s := v.slice)
    (by intro x _; simp)
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists hspec
  unfold alloc.vec.CloneVec.clone at h
  rw [hs, bind_tc_ok] at h
  have hvv : v' = { slice := s, cap := v.cap } := (Result.ok_injective h).symm
  rw [hvv]
  show Slice.val s = Slice.val v.slice
  rw [hsv]

/-- Cloning an `i64` vector with the identity element clone is the identity. -/
private theorem clonevec_self (v : alloc.vec.Vec I64) :
    alloc.vec.CloneVec.clone core.clone.CloneI64 v = ok v := by
  have hspec := Slice.clone_spec (clone := core.clone.CloneI64.clone) (s := v.slice)
    (by intro x _; simp)
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists hspec
  unfold alloc.vec.CloneVec.clone
  rw [hs, bind_tc_ok]
  have hvv : ({ slice := s, cap := v.cap } : alloc.vec.Vec I64) = v := by
    apply alloc.vec.Vec.ext
    · show Slice.val s = Slice.val v.slice
      rw [hsv]
    · rfl
  rw [hvv]

/-- `from_elem` of an identity-cloned element is a replicate. -/
private theorem from_elem_val {T : Type} (cloneInst : core.clone.Clone T) (x : T) (n : Usize)
    (hclone : cloneInst.clone x = ok x) {v' : alloc.vec.Vec T}
    (h : alloc.vec.from_elem cloneInst x n = ok v') :
    v'.val = List.replicate n.val x ∧ v'.val.length = n.val := by
  obtain ⟨y, hy, hyv, hylen⟩ := WP.spec_imp_exists (alloc.vec.from_elem_spec cloneInst x n hclone)
  have hye : y = v' := Result.ok_injective (hy.symm.trans h)
  rw [hye] at hyv hylen
  exact ⟨hyv, hylen⟩

/-! ## The structural invariant and reachability -/

/-- Every plane below `nl` is a vector of exactly `len` values. -/
def PlanesWF (planes : List (alloc.vec.Vec I64)) (len nl : ℕ) : Prop :=
  planes.length = nl ∧ ∀ l, l < nl → ∃ pl, planes[l]? = some pl ∧ pl.val.length = len

/-- The structural invariant of a reachable KV state. -/
def KvInv (s : attn.KvState) : Prop :=
  0 < s.n_layer.val ∧ 0 < s.n_head.val ∧ 0 < s.n_head_kv.val ∧ 0 < s.head_dim.val ∧
  0 < s.cap.val ∧
  s.n_head.val % s.n_head_kv.val = 0 ∧
  s.occupied.val.length = s.n_layer.val ∧
  (∀ l, l < s.n_layer.val → ∃ len, s.occupied.val[l]? = some len ∧ len.val ≤ s.cap.val) ∧
  PlanesWF s.k.val (s.n_head_kv.val * s.cap.val * s.head_dim.val) s.n_layer.val ∧
  PlanesWF s.v.val (s.n_head_kv.val * s.cap.val * s.head_dim.val) s.n_layer.val ∧
  (∃ r : I64, r.val = (s.head_dim.val : ℤ) * (2 : ℤ) ^ (32 : ℕ) ∧
    word_fixed_rsqrt r = ok (core.result.Result.Ok s.scale))

/-- Reachability: the states `word_kv_state_new` returns, closed under a successful
`word_kv_append` (`word_kv_append_ok`). -/
inductive KvReachable : attn.KvState → Prop
  | new (nl nh nk hd cap : Usize) (s : attn.KvState)
      (h1 : nl ≠ 0#usize) (h2 : nh ≠ 0#usize) (h3 : nk ≠ 0#usize)
      (h4 : hd ≠ 0#usize) (h5 : cap ≠ 0#usize) (h6 : nh.val % nk.val = 0)
      (h : word_kv_state_new nl nh nk hd cap = ok (core.result.Result.Ok s)) :
      KvReachable s
  | append (s s' : attn.KvState) (layer : Usize) (k_rows v_rows : Slice I64) (pos : Usize)
      (hr : KvReachable s)
      (hlayer : layer.val < s.n_layer.val)
      (hk : k_rows.val.length = s.n_head_kv.val * s.head_dim.val)
      (hv : v_rows.val.length = s.n_head_kv.val * s.head_dim.val)
      (hocc : s.occupied.val[layer.val]? = some pos)
      (hposcap : pos.val < s.cap.val)
      (h : word_kv_append s layer k_rows v_rows = ok (core.result.Result.Ok pos, s')) :
      KvReachable s'

/-! ## The constructor allocates exactly the invariant -/

/-- **`KvState::new` output.** The extracted constructor's `Ok` record allocates a `nl`-long
`occupied` of zeros and the two `nl`-long planes of `n_head_kv·cap·head_dim` zeros, and its `scale`
is `word_fixed_rsqrt` of `head_dim·2^32`. -/
private theorem word_kv_state_new_inv (nl nh nk hd cap : Usize) (s : attn.KvState)
    (h1 : nl ≠ 0#usize) (h2 : nh ≠ 0#usize) (h3 : nk ≠ 0#usize)
    (h4 : hd ≠ 0#usize) (h5 : cap ≠ 0#usize) (h6 : nh.val % nk.val = 0)
    (h : word_kv_state_new nl nh nk hd cap = ok (core.result.Result.Ok s)) :
    s.occupied.val = List.replicate nl.val 0#usize ∧
    PlanesWF s.k.val (nk.val * cap.val * hd.val) nl.val ∧
    PlanesWF s.v.val (nk.val * cap.val * hd.val) nl.val ∧
    (∃ r : I64, r.val = (hd.val : ℤ) * (2 : ℤ) ^ (32 : ℕ) ∧
      word_fixed_rsqrt r = ok (core.result.Result.Ok s.scale)) := by
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
    have hnn : 0 ≤ (UScalar.hcast IScalarTy.I64 hd).val := by
      have hlt := hneg
      rw [IScalar.lt_equiv] at hlt
      have h0 : (0#i64).val = 0 := by norm_num
      omega
    have hcastval : (UScalar.hcast IScalarTy.I64 hd).val = (hd.val : ℤ) :=
      hcast_I64_val_of_nonneg hnn
    cases hONE : engine.ONE with
    | ret i =>
      rw [hONE, bind_tc_ok] at h
      have hi_eq : i = 4294967296#i64 := by
        have hE := kv_engine_ONE_eq
        rw [hONE] at hE
        exact Result.ok_injective hE
      have hival : i.val = (2 : ℤ) ^ (32 : ℕ) := by
        rw [hi_eq]; norm_num
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
        have hvval : v.val = (hd.val : ℤ) * (2 : ℤ) ^ (32 : ℕ) := by
          have hs := I64.checked_mul_bv_spec (UScalar.hcast IScalarTy.I64 hd) i
          have hmul' : core.num.checked_mul_IScalar (UScalar.hcast IScalarTy.I64 hd) i = some v :=
            hmul
          rw [hmul'] at hs
          rw [hs.2.2.1, hcastval, hival]
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
                          obtain ⟨hblankv, hblanklen⟩ :=
                            from_elem_val core.clone.CloneI64 0#i64 plane kv_clone_i64_zero hblank
                          have hvvv : vv.val = blank.val := clonevec_val hv
                          obtain ⟨hkvv, _⟩ := from_elem_val
                            (core.clone.CloneallocvecVec core.clone.CloneI64) vv nl
                            (by
                              show alloc.vec.CloneVec.clone core.clone.CloneI64 vv = ok vv
                              exact clonevec_self vv) hk
                          obtain ⟨hv1v, _⟩ := from_elem_val
                            (core.clone.CloneallocvecVec core.clone.CloneI64) blank nl
                            (by
                              show alloc.vec.CloneVec.clone core.clone.CloneI64 blank = ok blank
                              exact clonevec_self blank) hv1
                          obtain ⟨hv2v, _⟩ :=
                            from_elem_val core.clone.CloneUsize 0#usize nl kv_clone_usize_zero hv2
                          have hxval : x.val = nk.val * cap.val := by
                            have hs := Usize.checked_mul_bv_spec nk cap
                            simp only [hnk] at hs
                            exact hs.2.1
                          have hplanval : plane.val = x.val * hd.val := by
                            have hs := Usize.checked_mul_bv_spec x hd
                            simp only [hx] at hs
                            exact hs.2.1
                          have hscale : word_fixed_rsqrt v =
                              ok (core.result.Result.Ok val1) := by
                            unfold word_fixed_rsqrt
                            rw [hr1, hw]
                          have hsr : s = { n_layer := nl, n_head := nh, n_head_kv := nk, head_dim := hd, cap := cap, scale := val1, occupied := vv2, k := kk, v := vv1 } := by
                            apply (core.result.Result.Ok.injEq _ _).mp
                            exact (Result.ok_injective h).symm
                          subst hsr
                          refine ⟨hv2v, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨v, hvval, hscale⟩⟩
                          · rw [hkvv]; simp [List.length_replicate]
                          · intro l hl
                            refine ⟨vv, ?_, ?_⟩
                            · rw [hkvv]; simp [hl]
                            · rw [hvvv, hblanklen, hplanval, hxval]
                          · rw [hv1v]; simp [List.length_replicate]
                          · intro l hl
                            refine ⟨blank, ?_, ?_⟩
                            · rw [hv1v]; simp [hl]
                            · rw [hblanklen, hplanval, hxval]
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

/-! ## Append: the shape and the preserved invariant -/

/-- **Successful append shape.** On a reachable state the preconditions of a successful append
carry the append's arithmetic, so `word_kv_append_ok` pins `s'` to the record update of `s`. -/
private theorem kv_append_shape (s s' : attn.KvState) (layer : Usize) (k_rows v_rows : Slice I64)
    (pos : Usize) (hinvX : KvInv s) (hlayer : layer.val < s.n_layer.val)
    (hbnd : layer.val < s.occupied.val.length)
    (hbndk : layer.val < s.k.val.length) (hbndv : layer.val < s.v.val.length)
    (hk : k_rows.val.length = s.n_head_kv.val * s.head_dim.val)
    (hv : v_rows.val.length = s.n_head_kv.val * s.head_dim.val)
    (hocc : s.occupied.val[layer.val]? = some pos) (hposcap : pos.val < s.cap.val)
    (h : word_kv_append s layer k_rows v_rows = ok (core.result.Result.Ok pos, s')) :
    ∃ v1 v3 : Usize, v3.val = pos.val + 1 ∧
      s' = { s with
        occupied := s.occupied.set layer v3,
        k := s.k.set layer (alloc.vec.Vec.setSlice! (s.k.val[layer.val]) v1.val k_rows.val),
        v := s.v.set layer (alloc.vec.Vec.setSlice! (s.v.val[layer.val]) v1.val v_rows.val) } := by
  obtain ⟨_hnl, _hnh, _hnkv, _hhd, hcap0, _hdiv, _hlenocc, _hoccwf, hkpl, hvpl, _hscale⟩ := hinvX
  obtain ⟨hkl, hkwf⟩ := hkpl
  obtain ⟨hvl, hvwf⟩ := hvpl
  obtain ⟨plk, hplk?, hplklen⟩ := hkwf layer.val hlayer
  have hplkget : s.k.val[layer.val] = plk := by
    have hg := List.getElem?_eq_getElem (l := s.k.val) hbndk
    rw [hg] at hplk?
    exact Option.some.inj hplk?
  obtain ⟨plv, hplv?, hplvlen⟩ := hvwf layer.val hlayer
  have hplvget : s.v.val[layer.val] = plv := by
    have hg := List.getElem?_eq_getElem (l := s.v.val) hbndv
    rw [hg] at hplv?
    exact Option.some.inj hplv?
  have hoccget : s.occupied.val[layer.val] = pos := by
    have hg := List.getElem?_eq_getElem (l := s.occupied.val) hbnd
    rw [hg] at hocc
    exact Option.some.inj hocc
  have hPle : s.n_head_kv.val * s.cap.val * s.head_dim.val ≤ Usize.max := by
    rw [← hplklen]
    exact plk.property
  have hnk_hd : s.n_head_kv.val * s.head_dim.val ≤ Usize.max := by
    have hle : s.n_head_kv.val ≤ s.n_head_kv.val * s.cap.val :=
      Nat.le_mul_of_pos_right s.n_head_kv.val hcap0
    have hm := Nat.mul_le_mul_right s.head_dim.val hle
    simpa [Nat.mul_assoc] using le_trans hm hPle
  obtain ⟨w, hw_eq, hwval⟩ := usize_checked_mul_some _ _ hnk_hd
  have hkslice : Slice.len k_rows = w := by
    apply UScalar.eq_of_val_eq
    show k_rows.val.length = w.val
    rw [hk, ← hwval]
  have hvslice : Slice.len v_rows = w := by
    apply UScalar.eq_of_val_eq
    show v_rows.val.length = w.val
    rw [hv, ← hwval]
  have hcap' : ¬ s.cap ≤ (s.occupied.val[layer.val]).val := by
    rw [hoccget]
    exact not_le.mpr hposcap
  have hposw : pos.val * w.val ≤ Usize.max := by
    rw [hwval]
    calc pos.val * (s.n_head_kv.val * s.head_dim.val)
        ≤ s.cap.val * (s.n_head_kv.val * s.head_dim.val) :=
          Nat.mul_le_mul_right _ hposcap.le
      _ = s.n_head_kv.val * s.cap.val * s.head_dim.val := by ring
      _ ≤ Usize.max := hPle
  obtain ⟨v1, hmul0, hv1val⟩ := usize_checked_mul_some pos w hposw
  have hmul : (s.occupied.val[layer.val]).checked_mul w = some v1 := by
    rw [hoccget]; exact hmul0
  have hv1w : v1.val + w.val ≤ Usize.max := by
    rw [hv1val, hwval]
    calc pos.val * (s.n_head_kv.val * s.head_dim.val) + s.n_head_kv.val * s.head_dim.val
        = (pos.val + 1) * (s.n_head_kv.val * s.head_dim.val) := by ring
      _ ≤ s.cap.val * (s.n_head_kv.val * s.head_dim.val) :=
          Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hposcap)
      _ = s.n_head_kv.val * s.cap.val * s.head_dim.val := by ring
      _ ≤ Usize.max := hPle
  obtain ⟨v2, hadd, hv2val⟩ := usize_checked_add_some v1 w hv1w
  have hkend : v2.val ≤ (s.k.val[layer.val]).length := by
    rw [hplkget]
    show v2.val ≤ plk.val.length
    rw [hplklen, hv2val, hv1val, hwval]
    calc pos.val * (s.n_head_kv.val * s.head_dim.val) + s.n_head_kv.val * s.head_dim.val
        = (pos.val + 1) * (s.n_head_kv.val * s.head_dim.val) := by ring
      _ ≤ s.cap.val * (s.n_head_kv.val * s.head_dim.val) :=
          Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hposcap)
      _ = s.n_head_kv.val * s.cap.val * s.head_dim.val := by ring
  have hvend : v2.val ≤ (s.v.val[layer.val]).length := by
    rw [hplvget]
    show v2.val ≤ plv.val.length
    rw [hplvlen, hv2val, hv1val, hwval]
    calc pos.val * (s.n_head_kv.val * s.head_dim.val) + s.n_head_kv.val * s.head_dim.val
        = (pos.val + 1) * (s.n_head_kv.val * s.head_dim.val) := by ring
      _ ≤ s.cap.val * (s.n_head_kv.val * s.head_dim.val) :=
          Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hposcap)
      _ = s.n_head_kv.val * s.cap.val * s.head_dim.val := by ring
  have hpos1 : pos.val + 1 ≤ Usize.max := by
    have h1 : pos.val + 1 ≤ s.cap.val := Nat.succ_le_of_lt hposcap
    have h2 : s.cap.val ≤ Usize.max := by
      have hb := UScalar.hBounds s.cap
      rw [Usize.max, Usize.numBits]
      omega
    omega
  obtain ⟨v3, hadd30, _⟩ := usize_checked_add_some pos 1#usize hpos1
  have hadd3 : (s.occupied.val[layer.val]).checked_add 1#usize = some v3 := by
    rw [hoccget]; exact hadd30
  have hv3 : v3.val = pos.val + 1 := usize_checked_add_one_val hadd30
  have hv3cap : v3.val ≤ s.cap.val := by rw [hv3]; omega
  obtain ⟨heq, _⟩ := word_kv_append_ok s layer k_rows v_rows w v1 v2 v3
    (by rw [UScalar.le_equiv]; exact not_le.mpr hlayer) hw_eq hkslice
    hvslice hbnd hcap' hbndk hbndv hmul hadd hadd3 hkend hvend
  refine ⟨v1, v3, hv3, ?_⟩
  have hpair := Result.ok_injective (heq.symm.trans h)
  exact (Prod.mk.injEq _ _ _ _).mp hpair |>.2.symm

/-- **Append preserves `KvInv`.** -/
private theorem kv_append_inv (s s' : attn.KvState) (layer : Usize) (k_rows v_rows : Slice I64)
    (pos : Usize) (hinvX : KvInv s) (hlayer : layer.val < s.n_layer.val)
    (hk : k_rows.val.length = s.n_head_kv.val * s.head_dim.val)
    (hv : v_rows.val.length = s.n_head_kv.val * s.head_dim.val)
    (hocc : s.occupied.val[layer.val]? = some pos) (hposcap : pos.val < s.cap.val)
    (h : word_kv_append s layer k_rows v_rows = ok (core.result.Result.Ok pos, s')) :
    KvInv s' := by
  have hInv' : KvInv s := hinvX
  obtain ⟨hnl, hnh, hnk, hhd, hcap0, hdiv, hlenocc, hoccwf, hkpl, hvpl, hscale⟩ := hinvX
  obtain ⟨hkl, hkwf⟩ := hkpl
  obtain ⟨hvl, hvwf⟩ := hvpl
  have hbnd : layer.val < s.occupied.val.length := by rw [hlenocc]; exact hlayer
  have hbndk : layer.val < s.k.val.length := by rw [hkl]; exact hlayer
  have hbndv : layer.val < s.v.val.length := by rw [hvl]; exact hlayer
  obtain ⟨plk, hplk?, hplklen⟩ := hkwf layer.val hlayer
  have hplkget : s.k.val[layer.val] = plk := by
    have hg := List.getElem?_eq_getElem (l := s.k.val) hbndk
    rw [hg] at hplk?
    exact Option.some.inj hplk?
  obtain ⟨plv, hplv?, hplvlen⟩ := hvwf layer.val hlayer
  have hplvget : s.v.val[layer.val] = plv := by
    have hg := List.getElem?_eq_getElem (l := s.v.val) hbndv
    rw [hg] at hplv?
    exact Option.some.inj hplv?
  have hoccget : s.occupied.val[layer.val] = pos := by
    have hg := List.getElem?_eq_getElem (l := s.occupied.val) hbnd
    rw [hg] at hocc
    exact Option.some.inj hocc
  obtain ⟨v1, v3, hv3, hs'⟩ := kv_append_shape s s' layer k_rows v_rows pos hInv' hlayer hbnd
    hbndk hbndv hk hv hocc hposcap h
  have hv3cap : v3.val ≤ s.cap.val := by rw [hv3]; omega
  subst hs'
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hnl
  · simpa using hnh
  · simpa using hnk
  · simpa using hhd
  · simpa using hcap0
  · simpa using hdiv
  · simpa [alloc.vec.Vec.set_val_eq, List.length_set] using hlenocc
  · intro l hl
    simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set]
    by_cases hlj : layer.val = l
    · rw [if_pos hlj, if_pos hbnd]
      exact ⟨v3, rfl, hv3cap⟩
    · rw [if_neg hlj]
      exact hoccwf l hl
  · refine ⟨?_, ?_⟩
    · simpa [List.length_set] using hkl
    · intro l hl
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set]
      by_cases hlj : layer.val = l
      · rw [if_pos hlj, if_pos hbndk]
        refine ⟨alloc.vec.Vec.setSlice! (s.k.val[layer.val]) v1.val k_rows.val, rfl, ?_⟩
        show (alloc.vec.Vec.setSlice! (s.k.val[layer.val]) v1.val k_rows.val).length
          = s.n_head_kv.val * s.cap.val * s.head_dim.val
        rw [alloc.vec.Vec.setSlice!_length, hplkget]
        exact hplklen
      · rw [if_neg hlj]
        exact hkwf l hl
  · refine ⟨?_, ?_⟩
    · simpa [List.length_set] using hvl
    · intro l hl
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set]
      by_cases hlj : layer.val = l
      · rw [if_pos hlj, if_pos hbndv]
        refine ⟨alloc.vec.Vec.setSlice! (s.v.val[layer.val]) v1.val v_rows.val, rfl, ?_⟩
        show (alloc.vec.Vec.setSlice! (s.v.val[layer.val]) v1.val v_rows.val).length
          = s.n_head_kv.val * s.cap.val * s.head_dim.val
        rw [alloc.vec.Vec.setSlice!_length, hplvget]
        exact hplvlen
      · rw [if_neg hlj]
        exact hvwf l hl
  · simpa using hscale

/-! ## The reachable-state invariant -/

/-- **Every reachable state satisfies `KvInv`.** -/
theorem KvReachable_inv (s : attn.KvState) (hr : KvReachable s) : KvInv s := by
  induction hr with
  | new nl nh nk hd cap s h1 h2 h3 h4 h5 h6 h =>
    obtain ⟨hgeom1, hgeom2, hgeom3, hgeom4, hgeom5⟩ :=
      word_kv_state_new_geometry nl nh nk hd cap s h1 h2 h3 h4 h5 h6 h
    obtain ⟨hoccv, hkpl, hvpl, hscale0⟩ :=
      word_kv_state_new_inv nl nh nk hd cap s h1 h2 h3 h4 h5 h6 h
    have hnz1 : 0 < s.n_layer.val := by rw [hgeom1]; exact usize_pos_of_ne h1
    have hnz2 : 0 < s.n_head.val := by rw [hgeom2]; exact usize_pos_of_ne h2
    have hnz3 : 0 < s.n_head_kv.val := by rw [hgeom3]; exact usize_pos_of_ne h3
    have hnz4 : 0 < s.head_dim.val := by rw [hgeom4]; exact usize_pos_of_ne h4
    have hnz5 : 0 < s.cap.val := by rw [hgeom5]; exact usize_pos_of_ne h5
    have hdiv : s.n_head.val % s.n_head_kv.val = 0 := by rw [hgeom2, hgeom3]; exact h6
    have hlen : s.occupied.val.length = s.n_layer.val := by
      rw [hoccv, List.length_replicate, hgeom1]
    have hoccwf : ∀ l, l < s.n_layer.val →
        ∃ len, s.occupied.val[l]? = some len ∧ len.val ≤ s.cap.val := by
      intro l hl
      refine ⟨0#usize, ?_, by norm_num⟩
      rw [hoccv]
      simp only [List.getElem?_replicate]
      rw [if_pos (by rw [← hgeom1]; exact hl)]
    have hkpl' : PlanesWF s.k.val (s.n_head_kv.val * s.cap.val * s.head_dim.val)
        s.n_layer.val := by
      rw [hgeom3, hgeom5, hgeom4, hgeom1]; exact hkpl
    have hvpl' : PlanesWF s.v.val (s.n_head_kv.val * s.cap.val * s.head_dim.val)
        s.n_layer.val := by
      rw [hgeom3, hgeom5, hgeom4, hgeom1]; exact hvpl
    have hscl : ∃ r : I64, r.val = (s.head_dim.val : ℤ) * (2 : ℤ) ^ (32 : ℕ) ∧
        word_fixed_rsqrt r = ok (core.result.Result.Ok s.scale) := by
      obtain ⟨r, hr, hwr⟩ := hscale0
      exact ⟨r, by rw [hgeom4]; exact hr, hwr⟩
    exact ⟨hnz1, hnz2, hnz3, hnz4, hnz5, hdiv, hlen, hoccwf, hkpl', hvpl', hscl⟩
  | append s s' layer k_rows v_rows pos hr hlayer hk hv hocc hposcap h ih =>
    exact kv_append_inv s s' layer k_rows v_rows pos ih hlayer hk hv hocc hposcap h

/-- **A successful append leaves the geometry unchanged.** The `n_layer`, `n_head`, `n_head_kv`,
`head_dim` and `cap` of the returned state are those of the input. -/
theorem KvReachable_append_geometry (s s' : attn.KvState) (layer : Usize)
    (k_rows v_rows : Slice I64) (pos : Usize)
    (hr : KvReachable s)
    (hlayer : layer.val < s.n_layer.val)
    (hk : k_rows.val.length = s.n_head_kv.val * s.head_dim.val)
    (hv : v_rows.val.length = s.n_head_kv.val * s.head_dim.val)
    (hocc : s.occupied.val[layer.val]? = some pos) (hposcap : pos.val < s.cap.val)
    (h : word_kv_append s layer k_rows v_rows = ok (core.result.Result.Ok pos, s')) :
    s'.n_layer = s.n_layer ∧ s'.n_head = s.n_head ∧ s'.n_head_kv = s.n_head_kv ∧
      s'.head_dim = s.head_dim ∧ s'.cap = s.cap := by
  have hInv : KvInv s := KvReachable_inv s hr
  have hInv2 : KvInv s := hInv
  obtain ⟨_hnl, _hnh, _hnkv, _hhd, _hcap, _hdiv, hlenocc, _hoccwf, hkpl, hvpl, _hscale⟩ := hInv2
  have hbnd : layer.val < s.occupied.val.length := by rw [hlenocc]; exact hlayer
  have hbndk : layer.val < s.k.val.length := by rw [hkpl.1]; exact hlayer
  have hbndv : layer.val < s.v.val.length := by rw [hvpl.1]; exact hlayer
  obtain ⟨v1, v3, _hv3, hs'⟩ := kv_append_shape s s' layer k_rows v_rows pos hInv hlayer hbnd
    hbndk hbndv hk hv hocc hposcap h
  subst hs'
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-! ## Structural conjuncts of `attn_domain_v2` on reachable states -/

/-- **Structural part of `attn_domain_v2` (proved).** On a reachable state, in-range layer and
query head, a live position and `head_dim`-wide `q`/`out`, every structural conjunct of
`attn_domain_v2` holds; the three data windows remain hypotheses. -/
theorem attn_domain_v2_structural (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (kplane vplane : alloc.vec.Vec I64)
    (hr : KvReachable state)
    (hlayer : layer.val < state.n_layer.val)
    (hqhead : q_head.val < state.n_head.val)
    (hocc : ∃ len, state.occupied.val[layer.val]? = some len ∧ pos.val < len.val)
    (hq : q.val.length = state.head_dim.val)
    (hout : out.val.length = state.head_dim.val)
    (hkplane : state.k.val[layer.val]? = some kplane)
    (hvplane : state.v.val[layer.val]? = some vplane)
    (hcap30 : state.cap.val ≤ 2 ^ 30)
    (hrowdot : ∀ s, s ≤ pos.val → ∀ krow : Slice I64,
      krow.val = planeRow kplane.val state.n_head_kv.val state.head_dim.val
        (q_head.val / (state.n_head.val / state.n_head_kv.val)) s → row_dot_domain krow q)
    (hscore : ∀ sc ∈ scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
        (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val,
      |sc| ≤ (2 : ℤ) ^ (62 : ℕ))
    (hvalwin : ∀ s, s ≤ pos.val → ∀ i, i < state.head_dim.val →
      |((planeRow vplane.val state.n_head_kv.val state.head_dim.val
        (q_head.val / (state.n_head.val / state.n_head_kv.val)) s).getD i 0#i64).val|
        ≤ (2 : ℤ) ^ (62 : ℕ)) :
    attn_domain_v2 state layer q_head pos q out kplane vplane := by
  obtain ⟨hnl, hnh, hnk, hhd, hcap0, hdiv, _hlenocc, hoccwf, hkpl, hvpl, hscale⟩ :=
    KvReachable_inv state hr
  obtain ⟨len, hlen?, hposlen⟩ := hocc
  obtain ⟨len', hlen'?, hlenle⟩ := hoccwf layer hlayer
  have hlen_eq : len = len' := Option.some.inj (hlen?.symm.trans hlen'?)
  have hposcap : pos.val < state.cap.val := by
    rw [hlen_eq] at hposlen
    exact lt_of_lt_of_le hposlen hlenle
  obtain ⟨plk, hplk?, hplklen⟩ := hkpl.2 layer.val hlayer
  have hkplane_len : kplane.val.length =
      state.n_head_kv.val * state.cap.val * state.head_dim.val := by
    rw [hkplane] at hplk?
    have hkpe : kplane = plk := Option.some.inj hplk?
    rw [hkpe, hplklen]
  obtain ⟨plv, hplv?, hplvlen⟩ := hvpl.2 layer.val hlayer
  have hvplane_len : vplane.val.length =
      state.n_head_kv.val * state.cap.val * state.head_dim.val := by
    rw [hvplane] at hplv?
    have hvpe : vplane = plv := Option.some.inj hplv?
    rw [hvpe, hplvlen]
  refine ⟨⟨hnl, hnh, hnk, hhd, hcap0, hdiv, hlayer, hqhead, hq, hout, hposcap⟩,
    ⟨len, hlen?, hposlen⟩,
    hscale, hkplane, hvplane, hkplane_len, hvplane_len, hrowdot, hscore, hvalwin, ?_⟩
  rw [show (2 : ℕ) ^ 30 = 1073741824 by norm_num] at hcap30 ⊢
  omega

/-- **End-to-end on reachable states.** On a reachable state and with the three data windows, the
conclusion of `attn_head_refines_v2_proved` holds. -/
theorem attn_head_refines_reachable (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (kplane vplane : alloc.vec.Vec I64)
    (hr : KvReachable state)
    (hlayer : layer.val < state.n_layer.val)
    (hqhead : q_head.val < state.n_head.val)
    (hocc : ∃ len, state.occupied.val[layer.val]? = some len ∧ pos.val < len.val)
    (hq : q.val.length = state.head_dim.val)
    (hout : out.val.length = state.head_dim.val)
    (hkplane : state.k.val[layer.val]? = some kplane)
    (hvplane : state.v.val[layer.val]? = some vplane)
    (hcap30 : state.cap.val ≤ 2 ^ 30)
    (hrowdot : ∀ s, s ≤ pos.val → ∀ krow : Slice I64,
      krow.val = planeRow kplane.val state.n_head_kv.val state.head_dim.val
        (q_head.val / (state.n_head.val / state.n_head_kv.val)) s → row_dot_domain krow q)
    (hscore : ∀ sc ∈ scoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
        (q_head.val / (state.n_head.val / state.n_head_kv.val)) state.scale.val pos.val,
      |sc| ≤ (2 : ℤ) ^ (62 : ℕ))
    (hvalwin : ∀ s, s ≤ pos.val → ∀ i, i < state.head_dim.val →
      |((planeRow vplane.val state.n_head_kv.val state.head_dim.val
        (q_head.val / (state.n_head.val / state.n_head_kv.val)) s).getD i 0#i64).val|
        ≤ (2 : ℤ) ^ (62 : ℕ)) :
    ∃ out' : Slice I64,
      word_attn_head state layer q_head q pos out = ok (core.result.Result.Ok (), out') ∧
      out'.val.length = out.val.length ∧
      ∀ i : ℕ, i < state.head_dim.val →
        let kvh := q_head.val / (state.n_head.val / state.n_head_kv.val)
        let hdim : ℝ := (state.head_dim.val : ℝ)
        let sc := idealScoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
          kvh hdim pos.val
        let vs := valueList vplane.val state.n_head_kv.val state.head_dim.val kvh pos.val i
        |toReal (out'.val.getD i 0#i64) - denseAttnReal (listAbsMax sc) sc (toRealList vs)|
          ≤ attnErrBoundV2 (pos.val + 1) (listAbsMax (toRealList vs))
              (scoreErrV2 hdim (listAbsMax sc) |toReal state.scale|) :=
  attn_head_refines_v2_proved state layer q_head pos q out kplane vplane
    (attn_domain_v2_structural state layer q_head pos q out kplane vplane hr hlayer hqhead hocc hq
      hout hkplane hvplane hcap30 hrowdot hscore hvalwin)

end

end apeiron_word
