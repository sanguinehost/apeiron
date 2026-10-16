/-
Serving-path refinement of the extracted single-token attention head, loop by loop
(map `serve3-map.md` stage 6).

The Rust body is `src/engine/attn.rs` (`KvState::attn_head` 1053-1263); the extracted kernel
is `attn.KvState.attn_head` in `proof/ApeironWord.lean:10448`, reached through the public
wrapper `apeiron_word.word_attn_head` (`ApeironWord.lean:102829`). This module adds the
per-loop specifications the composition needs: the four score/softmax/weight/output loops
(`attn_head_loop0`, `attn_head_loop1`, `attn_head_loop2_loop0`, `attn_head_loop2`,
`attn_head_loop3`) as `loop.spec_decr_nat` invariants whose postcondition is the exact
plain-Lean reference of that stage.

## The carrier

The Q32 path uses the `I64` Q32.32 carrier (`attn.KvStorage.Q32`, `q : Slice I64`,
`kplane vplane : Vec I64`, the score scale `state.scale : I64`). The `F32` twin
(`attn_head_f32`, `attn_head_f32_rows*`, `f32_core`) and the AVX2/`unsafe` bodies are the
extracted float boundary and are out of scope; nothing here mentions them.

## The stage references

* score (`loop0`): `score_j = round(round(<k_j, q>/2^32)·scale/2^32)`, both reductions the
  half-biased `divRoundNearest` (`attn2_Score`), over the causal prefix `0..=pos`; the
  running maximum is the `max`-fold from `I64::MIN` (`attn2_RunMax`);
* softmax (`loop1`): `e_j = exp_at(score_j - mx)` with the documented underflow saturation,
  accumulated in checked `i128` (`attn2_Exp`/`attn2_ExpSum`);
* weights (`loop2`): `w_j = round(e_j·2^32/Σe)` and `acc_i = Σ_j w_j·v_{j,i}` in checked
  `i128` (`attn2_AccSum`);
* output (`loop3`): `out_i = round(acc_i/2^32)` narrowed to `i64` (`attn2_Out`).

Trust boundary: the evidence reaches the extracted kernels above. It does not cover
rustc/LLVM codegen of `src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe`
container.
-/
import Serve.Block.Attn

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Loop 0: the score stage over the causal prefix -/

/-- The extracted key row `j` of the plane: the `hd` values at `first + j·stride`
(`src/engine/attn.rs`, `base = first + s·stride`, `&kplane[base..base + hd]`). -/
def attn2_Row (kplane : alloc.vec.Vec I64) (hd stride first : Usize) (j : ℕ) : List I64 :=
  kplane.val.slice (first.val + j * stride.val) (first.val + j * stride.val + hd.val)

/-- The exact `i128` prefix sum `Σ_{i<k} k_{j,i}·q_i` of key row `j` (`dotPrefix` of the
extracted row-dot kernel). -/
def attn2_DotPrefix (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first : Usize)
    (j k : ℕ) : ℤ :=
  ((((attn2_Row kplane hd stride first j).take k).zip (q.val.take k)).map
    (fun p => p.1.val * p.2.val)).sum

/-- The exact row dot `Σ_i k_{j,i}·q_i` of key row `j`: the prefix at the full row width `hd`.
Under `attn2_Loop0Dom` the extracted row and the query row both have length `hd`, so both `take`s
are the whole rows and this is the complete row dot. -/
def attn2_Dot (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first : Usize)
    (j : ℕ) : ℤ :=
  attn2_DotPrefix kplane q hd stride first j hd.val

/-- The exact Q32.32 score of key `j`: the row dot reduced once, then the load-time scale
epilogue reduced once (`attnScoreInt`, both half-biased `divRoundNearest`). -/
def attn2_Score (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first : Usize)
    (scale : ℤ) (j : ℕ) : ℤ :=
  attnScoreInt (attn2_Dot kplane q hd stride first j) scale

/-- The exact score list of the causal prefix `0..k-1`. -/
def attn2_Scores (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first : Usize)
    (scale : ℤ) (k : ℕ) : List ℤ :=
  (List.range k).map (fun j => attn2_Score kplane q hd stride first scale j)

/-- The running maximum of a prefix list, folded with `max` from `I64::MIN`. -/
def attn2_RunMax (l : List ℤ) : ℤ := l.foldl max core.num.I64.MIN.val

/-- The domain of the score loop: the query row is `hd` wide, every causal key row is inside
the plane, the position fits the `usize` width, each row dot's `i128` prefixes are in range,
and the per-key quantities are Q32.32-magnitude (dots within `2^95 - 2^32`, the scale within
`2^31`). These are exactly the conditions under which the checked arithmetic of the body
neither refuses nor narrows. -/
def attn2_Loop0Dom (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first pos : Usize)
    (scale : I64) : Prop :=
  q.val.length = hd.val ∧
  pos.val + 1 ≤ Usize.max ∧
  (∀ j : ℕ, j ≤ pos.val → first.val + j * stride.val + hd.val ≤ kplane.val.length) ∧
  (∀ j : ℕ, j ≤ pos.val → ∀ k : ℕ, 1 ≤ k → k ≤ hd.val →
      I128.min ≤ attn2_DotPrefix kplane q hd stride first j k ∧
      attn2_DotPrefix kplane q hd stride first j k ≤ I128.max) ∧
  (∀ j : ℕ, j ≤ pos.val → |attn2_Dot kplane q hd stride first j| ≤ (2:ℤ)^95 - (2:ℤ)^32) ∧
  |scale.val| ≤ (2:ℤ)^31

/-- The score-loop invariant: the failure flag is clear, the error slot is empty, the key index
is at most `pos + 1`, and the accumulators are the reference score prefix and its running
maximum. -/
def attn2_Loop0Inv (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first pos : Usize)
    (scale : I64)
    (st : alloc.vec.Vec I64 × I64 × Usize × Bool × Option attn.AttnError) : Prop :=
  st.2.2.2.1 = true ∧ st.2.2.2.2 = none ∧ st.2.2.1.val ≤ pos.val + 1 ∧
  st.1.val.map (fun x => x.val) =
    attn2_Scores kplane q hd stride first scale.val st.2.2.1.val ∧
  st.2.1.val = attn2_RunMax (attn2_Scores kplane q hd stride first scale.val st.2.2.1.val)

/-! ## Arithmetic helpers -/

/-- A Q32.32-magnitude integer, reduced once by the `round_shift_32` epilogue, stays in the
`i64` window. -/
theorem attn2_round_bound (n : ℤ) (h : |n| ≤ (2:ℤ)^95 - (2:ℤ)^32) :
    |divRoundNearest n ((2:ℤ)^32)| ≤ (2:ℤ)^63 - 1 := by
  have hhalf : (2:ℤ)^32 / 2 = (2:ℤ)^31 := by norm_num
  have hd : (0:ℤ) < (2:ℤ)^32 := by norm_num
  unfold divRoundNearest
  rw [hhalf]
  have hb := div_round_nearest_abs_bound n ((2:ℤ)^32) hd
  rw [hhalf] at hb
  obtain ⟨hb1, hb2⟩ := abs_le.mp hb
  obtain ⟨hn1, hn2⟩ := abs_le.mp h
  have hX : ((n + 2^31) / 2^32) * 2^32 = n + (((n + 2^31) / 2^32) * 2^32 - n) := by ring
  have hle : ((n + 2^31) / 2^32) * 2^32 ≤ (2:ℤ)^95 - 2^31 := by rw [hX]; linarith
  have hge : -((2:ℤ)^95 - 2^31) ≤ ((n + 2^31) / 2^32) * 2^32 := by rw [hX]; linarith
  have habs : |(n + 2^31) / 2^32| * (2:ℤ)^32 ≤ (2:ℤ)^95 - 2^31 := by
    have h2 : |((n + 2^31) / 2^32) * (2:ℤ)^32| ≤ (2:ℤ)^95 - 2^31 := by
      rw [abs_le]; exact ⟨hge, hle⟩
    rwa [abs_mul, abs_of_pos hd] at h2
  by_contra hc
  rw [not_le] at hc
  have h2 : ((2:ℤ)^63 - 1) * (2:ℤ)^32 ≤ |(n + 2^31) / 2^32| * (2:ℤ)^32 :=
    mul_le_mul_of_nonneg_right (le_of_lt hc) (by norm_num)
  have h3 : ((2:ℤ)^63 - 1) * (2:ℤ)^32 = (2:ℤ)^95 - (2:ℤ)^32 := by norm_num
  rw [h3] at h2
  linarith [habs, h2]

/-- The `x >>> 32#i32` shift and the checked multiply witnesses are the extracted ones; the
row-dot kernel `matvec.row_dot_fixed` returns `Ok raw` exactly when its epilogue and `i64`
narrowing are admitted, and then `raw` is the half-biased reduction of the exact dot. -/
theorem attn2_row_dot_ok (w_row : Slice I64) (q : Slice I64)
    (hq : w_row.val.length = q.val.length)
    (hwin : ∀ k : ℕ, 1 ≤ k → k ≤ w_row.val.length →
      I128.min ≤ dotPrefix w_row q k ∧ dotPrefix w_row q k ≤ I128.max)
    (hmag : |dotPrefix w_row q w_row.val.length| ≤ (2:ℤ)^95 - (2:ℤ)^32) :
    ∃ raw : I64, matvec.row_dot_fixed w_row q = ok (core.result.Result.Ok raw) ∧
      raw.val = divRoundNearest (dotPrefix w_row q w_row.val.length) ((2:ℤ)^32) := by
  obtain ⟨S, hSval, hloop⟩ := row_dot_fixed_loop_eq w_row q hq.symm hwin
  have hI128min : I128.min = -(2:ℤ)^127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2:ℤ)^127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  have hSb : -(2^95 - 2^32) ≤ S.val ∧ S.val ≤ (2:ℤ)^95 - (2:ℤ)^32 := by
    rw [hSval]; exact abs_le.mp hmag
  have hlo31 : I128.min ≤ S.val + 2^31 := by rw [hI128min]; linarith [hSb.1]
  have hhi31 : S.val + 2^31 ≤ I128.max := by rw [hI128max]; linarith [hSb.2]
  obtain ⟨rounded, hround_eq, hroundval⟩ := round_shift_32_spec S hlo31 hhi31
  have hrb : |rounded.val| ≤ (2:ℤ)^63 - 1 := by
    rw [hroundval, hSval]
    exact attn2_round_bound (dotPrefix w_row q w_row.val.length) hmag
  have hrle : rounded.val ≤ (2:ℤ)^63 - 1 := (abs_le.mp hrb).2
  have hrge : -(2:ℤ)^63 ≤ rounded.val := by have := (abs_le.mp hrb).1; linarith
  have hmaxv : (IScalar.cast .I128 core.num.I64.MAX).val = (2:ℤ)^63 - 1 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MAX.val = (2:ℤ)^63 - 1 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2:ℤ)^63 - 1)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hminv : (IScalar.cast .I128 core.num.I64.MIN).val = -(2:ℤ)^63 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MIN.val = -(2:ℤ)^63 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2:ℤ)^63)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hmax : ¬ (rounded > IScalar.cast .I128 core.num.I64.MAX) := by
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv]
    exact not_lt.mpr hrle
  have hmin : ¬ (rounded < IScalar.cast .I128 core.num.I64.MIN) := by
    rw [IScalar.lt_equiv, hminv]
    exact not_lt.mpr hrge
  have hcastval : (IScalar.cast .I64 rounded).val = rounded.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 rounded.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  refine ⟨IScalar.cast .I64 rounded, ?_, ?_⟩
  · rw [show matvec.row_dot_fixed w_row q = word_row_dot_fixed w_row q from rfl,
      word_row_dot_fixed_unfold w_row q S hq hloop, hround_eq]
    simp only [lift, bind_tc_ok]
    rw [if_neg hmax, if_neg hmin]
  · rw [hcastval, hroundval, hSval]

/-! ## Score-loop arithmetic helpers -/

/-- `saturating_sub` on `usize` is truncated subtraction floored at zero. -/
theorem attn2_usize_saturating_sub_val (x y : Usize) :
    (core.num.Usize.saturating_sub x y).val = max (x.val - y.val) 0 := by
  have hb : x.bv.toNat < 2 ^ UScalarTy.Usize.numBits := by
    simpa only [UScalar.val] using x.hBounds
  have hlt0 : max 0 (x.bv.toNat - y.bv.toNat) < 2 ^ UScalarTy.Usize.numBits :=
    lt_of_le_of_lt (Nat.max_le.mpr ⟨Nat.zero_le _, Nat.sub_le _ _⟩) hb
  unfold core.num.Usize.saturating_sub UScalar.saturating_sub
  simp only [UScalar.val]
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hlt0]
  exact Nat.max_comm _ _

/-- The extracted `Vec::len` value. -/
theorem attn2_vec_len_val (v : alloc.vec.Vec I64) :
    (alloc.vec.Vec.len v).val = v.val.length :=
  alloc.vec.Vec.len_val v

/-- The score prefix list has one entry per key. -/
theorem attn2_scores_length (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first : Usize)
    (scale : ℤ) (n : ℕ) : (attn2_Scores kplane q hd stride first scale n).length = n := by
  unfold attn2_Scores
  simp

/-- The score prefix extends by one key. -/
theorem attn2_scores_succ (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first : Usize)
    (scale : ℤ) (n : ℕ) :
    attn2_Scores kplane q hd stride first scale (n + 1) =
      attn2_Scores kplane q hd stride first scale n ++
        [attn2_Score kplane q hd stride first scale n] := by
  unfold attn2_Scores
  rw [← Nat.succ_eq_add_one, List.range_succ, List.map_append, List.map_singleton]

/-- The running maximum extends by one key. -/
theorem attn2_runmax_succ (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first : Usize)
    (scale : ℤ) (n : ℕ) :
    attn2_RunMax (attn2_Scores kplane q hd stride first scale (n + 1)) =
      max (attn2_RunMax (attn2_Scores kplane q hd stride first scale n))
        (attn2_Score kplane q hd stride first scale n) := by
  rw [attn2_scores_succ]
  unfold attn2_RunMax
  rw [List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- The extracted running-maximum step, as a term identity on the carrier: the `if score > mx`
of the body is `max` in the order the reference folds with. -/
theorem attn2_ite_max (a b : I64) : (if a > b then a else b) = max b a := by
  apply IScalar.eq_of_val_eq
  rw [IScalar.coe_max]
  by_cases h : a > b
  · have h' : b.val < a.val := by rwa [gt_iff_lt, IScalar.lt_equiv] at h
    rw [if_pos h, max_def, if_pos (le_of_lt h')]
  · have h' : a.val ≤ b.val := le_of_not_gt (by rwa [gt_iff_lt, IScalar.lt_equiv] at h)
    rw [if_neg h, max_def]
    split_ifs with hle
    · exact le_antisymm hle h'
    · rfl

/-- The same step on the `Result` monad: the two success arms collapse to the folded maximum. -/
theorem attn2_ite_ok_max (a b : I64) :
    (if a > b then ok a else ok b) = ok (max b a) := by
  have h : (if a > b then a else b) = max b a := attn2_ite_max a b
  rw [← h]
  by_cases hc : a > b <;> simp [hc]

theorem attn2_i128_min : I128.min = -((2:ℤ)^127) := by
  norm_num [I128.min, IScalar.min, I128.numBits]

theorem attn2_i128_max : I128.max = (2:ℤ)^127 - 1 := by
  norm_num [I128.max, IScalar.max, I128.numBits]

theorem attn2_i64_min_val : core.num.I64.MIN.val = -((2:ℤ)^63) := by decide

theorem attn2_i64_max_val : core.num.I64.MAX.val = (2:ℤ)^63 - 1 := by decide

theorem attn2_prod_bound_lo : -((2:ℤ)^127) ≤ -(((2:ℤ)^63 - 1) * (2:ℤ)^31) := by norm_num

theorem attn2_prod_bound_hi : ((2:ℤ)^63 - 1) * (2:ℤ)^31 ≤ (2:ℤ)^127 - 1 := by norm_num

/-! ## Loop 0: the score stage -/

/-- One score-loop body step. From the invariant at `s ≤ pos`, the extracted body reads key row `s`,
reduces its Q32.32 score, extends the prefix list and the running maximum, and advances the index;
under the domain every checked operation stays on its success arm. -/
theorem attn2_loop0_body_step
    (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first pos : Usize)
    (scale : I64) (scores : alloc.vec.Vec I64) (mx : I64) (s : Usize)
    (hdom : attn2_Loop0Dom kplane q hd stride first pos scale)
    (hsle : s.val ≤ pos.val)
    (hscores : scores.val.map (fun x => x.val) =
      attn2_Scores kplane q hd stride first scale.val s.val)
    (hmx : mx.val = attn2_RunMax (attn2_Scores kplane q hd stride first scale.val s.val)) :
    ∃ scores' mx' s',
      attn.KvState.attn_head_loop0.body hd scale q pos kplane stride first scores mx s true none
        = ok (ControlFlow.cont (scores', mx', s', true, none)) ∧
      s'.val = s.val + 1 ∧
      scores'.val.map (fun x => x.val) =
        attn2_Scores kplane q hd stride first scale.val s'.val ∧
      mx'.val = attn2_RunMax (attn2_Scores kplane q hd stride first scale.val s'.val) := by
  obtain ⟨hqlen, hposmax, hrow, hpre, hmag, hscale⟩ := hdom
  have hklen : kplane.val.length ≤ Usize.max := kplane.property
  have hrows := hrow s.val hsle
  have hsle' : s ≤ pos := (UScalar.le_equiv s pos).2 hsle
  obtain ⟨i1v, hi1eq, hi1val⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := s) (y := stride) (by omega))
  obtain ⟨basev, hbaseeq, hbaseval⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := first) (y := i1v) (by omega))
  have hi3val : (core.num.Usize.saturating_sub (alloc.vec.Vec.len kplane) hd).val
      = kplane.val.length - hd.val := by
    rw [attn2_usize_saturating_sub_val, attn2_vec_len_val]
    omega
  have hnotb : ¬ (basev > core.num.Usize.saturating_sub (alloc.vec.Vec.len kplane) hd) := by
    intro hc
    rw [gt_iff_lt, UScalar.lt_equiv, hi3val] at hc
    omega
  obtain ⟨endv, hendeq, hendval⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := basev) (y := hd) (by omega))
  have hstart : (⟨basev, endv⟩ : core.ops.range.Range Usize).start ≤
      (⟨basev, endv⟩ : core.ops.range.Range Usize).«end» := by
    show basev ≤ endv
    rw [UScalar.le_equiv, hendval]
    omega
  have hendle : ((⟨basev, endv⟩ : core.ops.range.Range Usize).«end»).val ≤
      kplane.val.length := by
    show endv.val ≤ kplane.val.length
    rw [hendval, hbaseval, hi1val]
    omega
  obtain ⟨s1v, hs1eq, hs1val, hs1len⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_Range_spec kplane ⟨basev, endv⟩ hstart hendle)
  have hs1val' : s1v.val = List.slice basev.val endv.val kplane.val := by simpa using hs1val
  have hs1len0 : s1v.val.length = endv.val - basev.val := by simpa using hs1len
  have hs1row : s1v.val = attn2_Row kplane hd stride first s.val := by
    rw [hs1val', hendval, hbaseval, hi1val]
    rfl
  have hs1len' : s1v.val.length = hd.val := by
    rw [hs1len0, hendval, hbaseval, hi1val]
    omega
  have hdotrow : ∀ k : ℕ,
      dotPrefix s1v q k = attn2_DotPrefix kplane q hd stride first s.val k := by
    intro k
    unfold dotPrefix attn2_DotPrefix
    rw [hs1row]
  have hwin : ∀ k : ℕ, 1 ≤ k → k ≤ s1v.val.length →
      I128.min ≤ dotPrefix s1v q k ∧ dotPrefix s1v q k ≤ I128.max := by
    intro k h1 h2
    rw [hdotrow k]
    exact hpre s.val hsle k h1 (by omega)
  have hdoteq : dotPrefix s1v q s1v.val.length = attn2_Dot kplane q hd stride first s.val := by
    rw [hs1len']
    exact hdotrow hd.val
  have hmag' : |dotPrefix s1v q s1v.val.length| ≤ (2:ℤ)^95 - (2:ℤ)^32 := by
    rw [hdoteq]
    exact hmag s.val hsle
  obtain ⟨raw, hraweq, hrawval⟩ :=
    attn2_row_dot_ok s1v q (by rw [hs1len', hqlen]) hwin hmag'
  have hfrom1 : (core.convert.num.FromI128I64.from raw).val = raw.val :=
    core.convert.num.FromI128I64.from_val_eq raw
  have hfrom2 : (core.convert.num.FromI128I64.from scale).val = scale.val :=
    core.convert.num.FromI128I64.from_val_eq scale
  have hrawb : |raw.val| ≤ (2:ℤ)^63 - 1 := by
    rw [hrawval]
    exact attn2_round_bound _ hmag'
  have hprodabs : |raw.val * scale.val| ≤ ((2:ℤ)^63 - 1) * (2:ℤ)^31 := by
    rw [abs_mul]
    exact mul_le_mul hrawb hscale (abs_nonneg scale.val) (by norm_num)
  have hmul1 := (abs_le.mp hprodabs).1
  have hmul2 := (abs_le.mp hprodabs).2
  have hmullo : I128.min ≤ (core.convert.num.FromI128I64.from raw).val *
      (core.convert.num.FromI128I64.from scale).val := by
    rw [hfrom1, hfrom2, attn2_i128_min]
    linarith [attn2_prod_bound_lo]
  have hmulhi : (core.convert.num.FromI128I64.from raw).val *
      (core.convert.num.FromI128I64.from scale).val ≤ I128.max := by
    rw [hfrom1, hfrom2, attn2_i128_max]
    linarith [attn2_prod_bound_hi]
  obtain ⟨prod, hprodeq, hprodval⟩ := I128_checked_mul_some _ _ hmullo hmulhi
  have hprodval' : prod.val = raw.val * scale.val := by
    rw [hprodval, hfrom1, hfrom2]
  have hprodb : |prod.val| ≤ (2:ℤ)^95 - (2:ℤ)^32 := by
    rw [hprodval']
    exact le_trans hprodabs (by norm_num)
  have hlo31 : I128.min ≤ prod.val + 2^31 := by
    rw [hprodval', attn2_i128_min]
    linarith
  have hhi31 : prod.val + 2^31 ≤ I128.max := by
    rw [hprodval', attn2_i128_max]
    linarith
  obtain ⟨rounded, hroundeq, hroundval⟩ := round_shift_32_spec prod hlo31 hhi31
  have hroundb : |rounded.val| ≤ (2:ℤ)^63 - 1 := by
    rw [hroundval]
    exact attn2_round_bound prod.val hprodb
  have hroundle : rounded.val ≤ (2:ℤ)^63 - 1 := (abs_le.mp hroundb).2
  have hroundge : -((2:ℤ)^63) ≤ rounded.val := by
    have := (abs_le.mp hroundb).1
    linarith
  have hgt6 : ¬ (rounded > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, core.convert.num.FromI128I64.from_val_eq,
      attn2_i64_max_val] at hc
    omega
  have hlt7 : ¬ (rounded < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, core.convert.num.FromI128I64.from_val_eq,
      attn2_i64_min_val] at hc
    omega
  have hcast : (IScalar.cast .I64 rounded).val = rounded.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 rounded.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  have hscoreval : (IScalar.cast .I64 rounded).val =
      attn2_Score kplane q hd stride first scale.val s.val := by
    rw [hcast, hroundval, hprodval', hrawval]
    unfold attn2_Score attnScoreInt attn2_Dot attn2_DotPrefix dotPrefix
    rw [hs1len', hs1row]
  have hsc : scores.val.length = s.val := by
    have h := congrArg List.length hscores
    rw [List.length_map, attn2_scores_length] at h
    exact h
  have hpushmax : scores.val.length < Usize.max := by
    rw [hsc]
    omega
  obtain ⟨scores2, hpusheq, hscores2val⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec scores (IScalar.cast .I64 rounded) hpushmax)
  have hsucmax : s.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
    omega
  obtain ⟨s1', hs1'eq, hs1'val⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := s) (y := 1#usize) hsucmax)
  refine ⟨scores2, max mx (IScalar.cast .I64 rounded), s1', ?_, ?_, ?_, ?_⟩
  · rw [attn.KvState.attn_head_loop0.body, if_pos hsle', if_pos rfl]
    rw [hi1eq]
    simp only [lift, bind_tc_ok]
    rw [hbaseeq]
    simp only [bind_tc_ok]
    rw [if_neg hnotb]
    rw [hendeq]
    simp only [bind_tc_ok]
    rw [hs1eq]
    simp only [bind_tc_ok]
    rw [hraweq]
    simp only [bind_tc_ok]
    rw [hprodeq]
    dsimp only
    rw [hroundeq]
    simp only [bind_tc_ok]
    rw [if_neg hgt6]
    rw [if_neg hlt7]
    rw [attn2_ite_ok_max]
    simp only [bind_tc_ok]
    rw [hpusheq]
    simp only [bind_tc_ok]
    rw [hs1'eq]
    simp only [bind_tc_ok]
    rfl
  · rw [hs1'val]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
  · rw [hscores2val]
    simp only [List.map_append, List.map_singleton]
    rw [hscores, hscoreval, ← attn2_scores_succ, hs1'val]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
  · rw [IScalar.coe_max, hmx, hscoreval, ← attn2_runmax_succ, hs1'val]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]

/-- The score loop at the causal prefix. From an empty score accumulator at key `0` with running
maximum `I64::MIN`, and no error, the extracted loop terminates on its success arm at key `pos + 1`
with the exact score list `attn2_Scores … (pos + 1)` and its running maximum, both are the plain-Lean
reference of the score stage. -/
theorem attn2_loop0_spec
    (kplane : alloc.vec.Vec I64) (q : Slice I64) (hd stride first pos : Usize) (scale : I64)
    (scores0 : alloc.vec.Vec I64) (hs0 : scores0.val = [])
    (hdom : attn2_Loop0Dom kplane q hd stride first pos scale) :
    attn.KvState.attn_head_loop0 hd scale q pos kplane stride first
      scores0 core.num.I64.MIN 0#usize true none ⦃ b =>
      ∃ sc : alloc.vec.Vec I64, ∃ mm : I64, b = (sc, mm, none) ∧
        sc.val.map (fun x => x.val) =
          attn2_Scores kplane q hd stride first scale.val (pos.val + 1) ∧
        mm.val = attn2_RunMax
          (attn2_Scores kplane q hd stride first scale.val (pos.val + 1)) ⦄ := by
  unfold attn.KvState.attn_head_loop0
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec I64 × I64 × Usize × Bool × Option attn.AttnError =>
      pos.val + 1 - st.2.2.1.val)
    (inv := attn2_Loop0Inv kplane q hd stride first pos scale)
    (post := fun b : alloc.vec.Vec I64 × I64 × Option attn.AttnError =>
      ∃ sc : alloc.vec.Vec I64, ∃ mm : I64, b = (sc, mm, none) ∧
        sc.val.map (fun x => x.val) =
          attn2_Scores kplane q hd stride first scale.val (pos.val + 1) ∧
        mm.val = attn2_RunMax
          (attn2_Scores kplane q hd stride first scale.val (pos.val + 1)))
  · intro st hst
    rcases st with ⟨sc, mm, si, ok1, err⟩
    obtain ⟨hok1, herr, hle, hsc, hmx⟩ := hst
    dsimp only at hok1 herr hle hsc hmx
    rw [hok1, herr]
    by_cases hlt : si.val ≤ pos.val
    · obtain ⟨sc2, mm2, si2, hbody, hsi2, hsc2, hmx2⟩ :=
        attn2_loop0_body_step kplane q hd stride first pos scale sc mm si hdom hlt hsc hmx
      dsimp only
      rw [hbody]
      refine ⟨⟨rfl, rfl, ?_, hsc2, hmx2⟩, ?_⟩
      · rw [hsi2]
        omega
      · show pos.val + 1 - si2.val < pos.val + 1 - si.val
        rw [hsi2]
        omega
    · have hne : ¬ (si ≤ pos) := by
        intro hc
        exact hlt ((UScalar.le_equiv si pos).1 hc)
      have hsi_eq : si.val = pos.val + 1 := by omega
      have hbody : attn.KvState.attn_head_loop0.body hd scale q pos kplane stride first
          sc mm si true none = ok (ControlFlow.done (sc, mm, none)) := by
        rw [attn.KvState.attn_head_loop0.body, if_neg hne]
      dsimp only
      rw [hbody]
      refine ⟨sc, mm, rfl, ?_, ?_⟩
      · rw [hsc, hsi_eq]
      · rw [hmx, hsi_eq]
  · have h0 : (0#usize).val = 0 := by norm_num
    refine ⟨rfl, rfl, ?_, ?_, ?_⟩
    · rw [h0]
      omega
    · rw [hs0, h0]
      simp [attn2_Scores]
    · rw [h0]
      simp [attn2_RunMax, attn2_Scores]

/-! ## Loop 1: the softmax numerator stage -/

/-- The exp-loop invariant, relational in the extracted `attn.exp_at`: the failure flag is clear, the
error slot is empty, the key index is at most `pos + 1`, the accumulator holds exactly the pushed
values `(List.range s).map f` for a function `f` that agrees with `attn.exp_at` at every score of the
causal prefix, and `total` is the `i64`-value sum of those. `f` is the value the extracted
`fixed_exp` returns on its success arm, so no closed form for the extracted exponential is needed. -/
def attn2_Loop1Inv (scores : alloc.vec.Vec I64) (mx : I64) (pos : Usize)
    (hlen : scores.val.length = pos.val + 1) (f : ℕ → I64)
    (st : alloc.vec.Vec I64 × I128 × Usize × Bool × Option attn.AttnError) : Prop :=
  st.2.2.2.1 = true ∧ st.2.2.2.2 = none ∧ st.2.2.1.val ≤ pos.val + 1 ∧
  (∀ (j : ℕ) (hj : j ≤ pos.val),
    attn.exp_at (scores.val[j]'(by omega)) mx = ok (core.result.Result.Ok (f j))) ∧
  st.1.val = (List.range st.2.2.1.val).map f ∧
  st.2.1.val = (((List.range st.2.2.1.val).map f).map (fun x => x.val)).sum

/-- One exp-loop body step. From the invariant at `s ≤ pos`, the extracted body reads the score
`scores[s]`, takes the `attn.exp_at` success arm (whose value the invariant's `f` records), folds it
into the checked `i128` total and pushes it; under the domain every checked operation stays on its
success arm. -/
theorem attn2_loop1_body_step
    (scores : alloc.vec.Vec I64) (mx : I64) (pos : Usize) (hlen : scores.val.length = pos.val + 1)
    (f : ℕ → I64) (exps : alloc.vec.Vec I64) (total : I128) (s : Usize)
    (hposmax : pos.val + 1 ≤ Usize.max)
    (hsle : s.val ≤ pos.val)
    (hexp : ∀ (j : ℕ) (hj : j ≤ pos.val),
      attn.exp_at (scores.val[j]'(by omega)) mx = ok (core.result.Result.Ok (f j)))
    (hadd : I128.min ≤ total.val + (f s.val).val ∧ total.val + (f s.val).val ≤ I128.max)
    (hexps : exps.val = (List.range s.val).map f)
    (htot : total.val = (((List.range s.val).map f).map (fun x => x.val)).sum) :
    ∃ exps' total' s',
      attn.KvState.attn_head_loop1.body pos scores mx exps total s true none
        = ok (ControlFlow.cont (exps', total', s', true, none)) ∧
      s'.val = s.val + 1 ∧
      exps'.val = (List.range s'.val).map f ∧
      total'.val = (((List.range s'.val).map f).map (fun x => x.val)).sum := by
  have hsid : s.val < scores.val.length := by omega
  obtain ⟨i1v, hi1eq, hi1val⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_usize_spec scores s hsid)
  have hexp_s : attn.exp_at i1v mx = ok (core.result.Result.Ok (f s.val)) := by
    rw [hi1val]
    exact hexp s.val hsle
  have hfromv : (core.convert.num.FromI128I64.from (f s.val)).val = (f s.val).val :=
    core.convert.num.FromI128I64.from_val_eq (f s.val)
  have haddlo : I128.min ≤ total.val + (core.convert.num.FromI128I64.from (f s.val)).val := by
    rw [hfromv]
    exact hadd.1
  have haddhi : total.val + (core.convert.num.FromI128I64.from (f s.val)).val ≤ I128.max := by
    rw [hfromv]
    exact hadd.2
  obtain ⟨t, haddeq, haddval⟩ :=
    I128_checked_add_some total (core.convert.num.FromI128I64.from (f s.val)) haddlo haddhi
  have hpushmax : exps.val.length < Usize.max := by
    rw [hexps]
    simp only [List.length_map, List.length_range]
    omega
  obtain ⟨exps2, hpusheq, hexps2val⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec exps (f s.val) hpushmax)
  have hsucmax : s.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
    omega
  obtain ⟨s1', hs1'eq, hs1'val⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := s) (y := 1#usize) hsucmax)
  have hsle' : s ≤ pos := (UScalar.le_equiv s pos).2 hsle
  refine ⟨exps2, t, s1', ?_, ?_, ?_, ?_⟩
  · rw [attn.KvState.attn_head_loop1.body, if_pos hsle', if_pos rfl]
    rw [alloc.vec.Vec.index_slice_index, hi1eq]
    simp only [lift, bind_tc_ok]
    rw [hexp_s]
    simp only [bind_tc_ok]
    rw [haddeq]
    dsimp only
    rw [hpusheq]
    simp only [bind_tc_ok]
    rw [hs1'eq]
    simp only [bind_tc_ok]
    rfl
  · rw [hs1'val]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
  · have h1 : (1#usize).val = 1 := by norm_num
    rw [hs1'val, h1, hexps2val, hexps, List.range_succ, List.map_append, List.map_singleton]
  · have h1 : (1#usize).val = 1 := by norm_num
    rw [hs1'val, h1, haddval, hfromv, htot, List.range_succ, List.map_append, List.map_singleton,
      List.map_append, List.map_singleton, List.sum_append, List.sum_singleton]

/-- The exp loop at the causal prefix. From an empty accumulator at key `0` with `i128` total `0`
and no error, when `attn.exp_at` succeeds at every score of the prefix and the checked `i128` sums
stay in range, the extracted loop terminates on its success arm at key `pos + 1` with the accumulator
`(List.range (pos+1)).map f` and total the `i64`-value sum of those. -/
theorem attn2_loop1_spec
    (scores : alloc.vec.Vec I64) (mx : I64) (pos : Usize) (hlen : scores.val.length = pos.val + 1)
    (f : ℕ → I64)
    (hposmax : pos.val + 1 ≤ Usize.max)
    (hexp : ∀ (j : ℕ) (_ : j ≤ pos.val),
      attn.exp_at (scores.val[j]'(by omega)) mx = ok (core.result.Result.Ok (f j)))
    (hadd : ∀ (j : ℕ) (_ : j ≤ pos.val),
      I128.min ≤ (((List.range j).map f).map (fun x => x.val)).sum + (f j).val ∧
      (((List.range j).map f).map (fun x => x.val)).sum + (f j).val ≤ I128.max)
    (exps0 : alloc.vec.Vec I64) (he0 : exps0.val = []) :
    attn.KvState.attn_head_loop1 pos scores mx exps0 0#i128 0#usize true none ⦃ b =>
      ∃ es : alloc.vec.Vec I64, ∃ tot : I128, b = (es, tot, none) ∧
        es.val = (List.range (pos.val + 1)).map f ∧
        tot.val = (((List.range (pos.val + 1)).map f).map (fun x => x.val)).sum ⦄ := by
  unfold attn.KvState.attn_head_loop1
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec I64 × I128 × Usize × Bool × Option attn.AttnError =>
      pos.val + 1 - st.2.2.1.val)
    (inv := attn2_Loop1Inv scores mx pos hlen f)
    (post := fun b : alloc.vec.Vec I64 × I128 × Option attn.AttnError =>
      ∃ es : alloc.vec.Vec I64, ∃ tot : I128, b = (es, tot, none) ∧
        es.val = (List.range (pos.val + 1)).map f ∧
        tot.val = (((List.range (pos.val + 1)).map f).map (fun x => x.val)).sum)
  · intro st hst
    rcases st with ⟨ex, tot, si, ok1, err⟩
    obtain ⟨hok1, herr, hle, hexp', hexps, htot⟩ := hst
    dsimp only at hok1 herr hle hexp' hexps htot
    rw [hok1, herr]
    by_cases hlt : si.val ≤ pos.val
    · have hsub : I128.min ≤ tot.val + (f si).val ∧ tot.val + (f si).val ≤ I128.max := by
        rw [htot]
        exact hadd si.val hlt
      obtain ⟨ex2, tot2, si2, hbody, hsi2, hexps2, htot2⟩ :=
        attn2_loop1_body_step scores mx pos hlen f ex tot si hposmax hlt hexp' hsub hexps htot
      dsimp only
      rw [hbody]
      refine ⟨⟨rfl, rfl, ?_, hexp', hexps2, htot2⟩, ?_⟩
      · rw [hsi2]
        omega
      · show pos.val + 1 - si2.val < pos.val + 1 - si.val
        rw [hsi2]
        omega
    · have hne : ¬ (si ≤ pos) := by
        intro hc
        exact hlt ((UScalar.le_equiv si pos).1 hc)
      have hsi_eq : si.val = pos.val + 1 := by omega
      have hbody : attn.KvState.attn_head_loop1.body pos scores mx ex tot si true none
          = ok (ControlFlow.done (ex, tot, none)) := by
        rw [attn.KvState.attn_head_loop1.body, if_neg hne]
      dsimp only
      rw [hbody]
      refine ⟨ex, tot, rfl, ?_, ?_⟩
      · rw [hexps, hsi_eq]
      · rw [htot, hsi_eq]
  · have h0 : (0#usize).val = 0 := by norm_num
    refine ⟨rfl, rfl, ?_, hexp, ?_, ?_⟩
    · rw [h0]
      omega
    · rw [he0, h0]
      rfl
    · have hz : (0#i128).val = 0 := by norm_num
      rw [hz, h0]
      rfl

/-! ## Loop 2 inner: the per-key column accumulation -/

/-- The inner column-accumulate loop invariant: the failure flag is clear, the error slot is empty,
the column index is at most `hd`, the accumulator keeps its width, and every component `j` of the
accumulator holds the input value `a0_j` plus `w·row_j` for each visited column `j < i`. -/
def attn2_Loop2Loop0Inv (hd : Usize) (row : Slice I64) (w : I128) (a0 : List ℤ)
    (st : alloc.vec.Vec I128 × Bool × Option attn.AttnError × Usize) : Prop :=
  st.2.1 = true ∧ st.2.2.1 = none ∧ st.2.2.2.val ≤ hd.val ∧
  st.1.val.length = a0.length ∧
  (∀ (j : ℕ) (_ : j < a0.length),
    (st.1.val.getD j (0#i128)).val =
      a0.getD j 0 + (if j < st.2.2.2.val then w.val * (row.val.getD j (0#i64)).val else 0))

/-- The domain of the inner column-accumulate loop: at every column the checked `i128` product
`w·row_j` and the checked sum `a0_j + w·row_j` stay in range, so neither `checked_mul` nor
`checked_add` takes its `none` arm. -/
def attn2_Loop2Loop0Dom (row : Slice I64) (w : I128) (a0 : List ℤ)
    (hrow : row.val.length = a0.length) : Prop :=
  ∀ (j : ℕ) (hj : j < a0.length),
    I128.min ≤ w.val * (row.val[j]'(by rw [hrow]; exact hj)).val ∧
    w.val * (row.val[j]'(by rw [hrow]; exact hj)).val ≤ I128.max ∧
    I128.min ≤ a0[j] + w.val * (row.val[j]'(by rw [hrow]; exact hj)).val ∧
    a0[j] + w.val * (row.val[j]'(by rw [hrow]; exact hj)).val ≤ I128.max

/-- `getD` after a list `set`: the written slot holds the new value, every other slot keeps its old
value. -/
theorem attn2_getD_set_val (l : List I128) (i j : ℕ) (a : I128) (hj : j < l.length) :
    (l.set i a).getD j (0#i128) = if j = i then a else l.getD j (0#i128) := by
  have hj' : j < (l.set i a).length := by rw [List.length_set]; exact hj
  rw [List.getD_eq_getElem (l.set i a) (0#i128) hj', List.getElem_set hj',
    List.getD_eq_getElem l (0#i128) hj]
  by_cases h : j = i
  · rw [if_pos h.symm, if_pos h]
  · rw [if_neg (fun hc => h hc.symm), if_neg h]

/-- One inner-loop body step. From the invariant at column `i < hd`, the extracted body reads
`row[i]`, forms the checked `i128` product `w·row_i`, adds it to `acc[i]` on the checked arm, writes
the result back and advances; under the domain every checked operation stays on its success arm. -/
theorem attn2_loop2_loop0_body_step
    (hd : Usize) (row : Slice I64) (w : I128) (a0 : List ℤ)
    (hrow : row.val.length = a0.length) (hhd : hd.val = a0.length)
    (hhdmax : hd.val + 1 ≤ Usize.max)
    (hdom : attn2_Loop2Loop0Dom row w a0 hrow)
    (acc : alloc.vec.Vec I128) (i : Usize)
    (hinv : attn2_Loop2Loop0Inv hd row w a0 (acc, true, none, i))
    (hlt : i.val < hd.val) :
    ∃ acc' i',
      attn.KvState.attn_head_loop2_loop0.body hd row w acc true none i
        = ok (ControlFlow.cont (acc', true, none, i')) ∧
      i'.val = i.val + 1 ∧
      attn2_Loop2Loop0Inv hd row w a0 (acc', true, none, i') := by
  obtain ⟨hok, herr, hile, hlen, hpoint⟩ := hinv
  dsimp only at hok herr hile hlen hpoint
  have hilt : i.val < a0.length := by rw [← hhd]; exact hlt
  have hirow : i.val < row.val.length := by rw [hrow]; exact hilt
  have hiacc : i.val < acc.val.length := by rw [hlen]; exact hilt
  have hslt : i < hd := (UScalar.lt_equiv i hd).2 hlt
  have h1 : (1#usize).val = 1 := by norm_num
  obtain ⟨r1, hr1eq, hr1val⟩ := WP.spec_imp_exists (Slice.index_usize_spec row i hirow)
  have hr1v : r1.val = (row.val[i.val]'hirow).val := by rw [hr1val]
  have hfrom : (core.convert.num.FromI128I64.from r1).val = r1.val :=
    core.convert.num.FromI128I64.from_val_eq r1
  obtain ⟨hdl, hdh, hdl2, hdh2⟩ := hdom i.val hilt
  have hmullo : I128.min ≤ w.val * (core.convert.num.FromI128I64.from r1).val := by
    rw [hfrom, hr1v]; exact hdl
  have hmulhi : w.val * (core.convert.num.FromI128I64.from r1).val ≤ I128.max := by
    rw [hfrom, hr1v]; exact hdh
  obtain ⟨term, htermeq, htermval⟩ :=
    I128_checked_mul_some w (core.convert.num.FromI128I64.from r1) hmullo hmulhi
  obtain ⟨i3, hi3eq, hi3val⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec acc i hiacc)
  have hi3v : i3.val = (acc.val.getD i.val (0#i128)).val := by
    rw [hi3val, List.getD_eq_getElem (acc.val) (0#i128) hiacc]
  have ha0i : (acc.val.getD i.val (0#i128)).val = a0[i.val] := by
    have h := hpoint i.val hilt
    rw [if_neg (by omega), add_zero] at h
    rw [List.getD_eq_getElem a0 (0 : ℤ) hilt] at h
    exact h
  have haddlo : I128.min ≤ i3.val + term.val := by
    rw [hi3v, ha0i, htermval, hfrom, hr1v]; exact hdl2
  have haddhi : i3.val + term.val ≤ I128.max := by
    rw [hi3v, ha0i, htermval, hfrom, hr1v]; exact hdh2
  obtain ⟨next, hnexteq, hnextval⟩ := I128_checked_add_some i3 term haddlo haddhi
  have hidxmut : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice Std.I128) acc i
      = ok (acc.val[i.val]'hiacc, acc.set i) := by
    rw [alloc.vec.Vec.index_mut_slice_index]
    unfold alloc.vec.Vec.index_mut_usize
    have hindex : alloc.vec.Vec.index_usize acc i = ok (acc.val[i.val]'hiacc) := by
      unfold alloc.vec.Vec.index_usize
      rw [alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem hiacc]
    rw [hindex]
    simp only [bind_tc_ok]
  have hsucmax : i.val + (1#usize).val ≤ Usize.max := by
    rw [h1]; omega
  obtain ⟨i', hi'eq, hi'val⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hsucmax)
  refine ⟨acc.set i next, i', ?_, ?_, ?_⟩
  · rw [attn.KvState.attn_head_loop2_loop0.body, if_pos hslt, if_pos rfl]
    simp only [alloc.vec.Vec.index_slice_index, hr1eq, htermeq, hi3eq, hnexteq, hidxmut, hi'eq,
      lift, bind_tc_ok, uncurry_apply_pair]
  · rw [hi'val, h1]
  · refine ⟨rfl, rfl, ?_, ?_, ?_⟩
    · rw [hi'val, h1]; omega
    · rw [alloc.vec.Vec.set_val_eq, List.length_set]; exact hlen
    · intro j hj
      have hjacc : j < acc.val.length := by rw [hlen]; exact hj
      rw [alloc.vec.Vec.set_val_eq, attn2_getD_set_val (acc.val) i.val j next hjacc]
      by_cases hji : j = i.val
      · rw [if_pos hji, hji]
        rw [hnextval, hi3v, ha0i, htermval, hfrom, hr1v,
          List.getD_eq_getElem (row.val) (0#i64) hirow, List.getD_eq_getElem a0 (0 : ℤ) hilt,
          hi'val, h1, if_pos (by omega)]
      · rw [if_neg hji, hpoint j hj]
        by_cases hltj : j < i.val
        · rw [if_pos hltj, hi'val, h1, if_pos (by omega)]
        · rw [if_neg hltj, hi'val, h1, if_neg (by omega)]

/-- The inner column-accumulate loop. From an accumulator of width `hd` whose component `j` starts
at `a0_j`, with `row` wide enough and every checked `i128` product and sum in range, the extracted
loop terminates on its success arm at column `hd` with every component incremented by `w·row_j`. -/
theorem attn2_loop2_loop0_spec
    (hd : Usize) (row : Slice I64) (w : I128) (a0 : List ℤ)
    (hrow : row.val.length = a0.length) (hhd : hd.val = a0.length)
    (hhdmax : hd.val + 1 ≤ Usize.max)
    (acc0 : alloc.vec.Vec I128) (hacc0len : acc0.val.length = a0.length)
    (ha0 : ∀ (j : ℕ) (_ : j < a0.length), (acc0.val.getD j (0#i128)).val = a0[j])
    (hdom : attn2_Loop2Loop0Dom row w a0 hrow) :
    attn.KvState.attn_head_loop2_loop0 hd acc0 true none row w 0#usize ⦃ b =>
      ∃ acc' : alloc.vec.Vec I128, b = (acc', true, none) ∧
        acc'.val.length = a0.length ∧
        (∀ (j : ℕ) (hj : j < a0.length),
          (acc'.val.getD j (0#i128)).val =
            a0[j] + w.val * (row.val[j]'(by rw [hrow]; exact hj)).val) ⦄ := by
  unfold attn.KvState.attn_head_loop2_loop0
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec I128 × Bool × Option attn.AttnError × Usize =>
      hd.val - st.2.2.2.val)
    (inv := attn2_Loop2Loop0Inv hd row w a0)
    (post := fun b : alloc.vec.Vec I128 × Bool × Option attn.AttnError =>
      ∃ acc' : alloc.vec.Vec I128, b = (acc', true, none) ∧
        acc'.val.length = a0.length ∧
        (∀ (j : ℕ) (hj : j < a0.length),
          (acc'.val.getD j (0#i128)).val =
            a0[j] + w.val * (row.val[j]'(by rw [hrow]; exact hj)).val))
  · intro st hst
    rcases st with ⟨a, ok1, err, si⟩
    obtain ⟨hok1, herr, hle, hlen, hpoint⟩ := hst
    dsimp only at hok1 herr hle hlen hpoint
    rw [hok1, herr]
    have hinv' : attn2_Loop2Loop0Inv hd row w a0 (a, true, none, si) :=
      ⟨rfl, rfl, hle, hlen, hpoint⟩
    by_cases hlt : si.val < hd.val
    · obtain ⟨a2, si2, hbody, hsi2, hinv2⟩ :=
        attn2_loop2_loop0_body_step hd row w a0 hrow hhd hhdmax hdom a si hinv' hlt
      dsimp only
      rw [hbody]
      refine ⟨hinv2, ?_⟩
      show hd.val - si2.val < hd.val - si.val
      rw [hsi2]; omega
    · have hne : ¬ (si < hd) := by
        intro hc
        exact hlt ((UScalar.lt_equiv si hd).1 hc)
      have hsi_eq : si.val = hd.val := by omega
      have hbody : attn.KvState.attn_head_loop2_loop0.body hd row w a true none si
          = ok (ControlFlow.done (a, true, none)) := by
        rw [attn.KvState.attn_head_loop2_loop0.body, if_neg hne]
      dsimp only
      rw [hbody]
      refine ⟨a, rfl, hlen, ?_⟩
      intro j hj
      have hjrow : j < row.val.length := by rw [hrow]; exact hj
      have hh := hpoint j hj
      rw [hsi_eq, if_pos (by rw [hhd]; exact hj)] at hh
      rw [hh]
      congr 1
      · rw [List.getD_eq_getElem a0 (0 : ℤ) hj]
      · rw [List.getD_eq_getElem (row.val) (0#i64) hjrow]
  · have h0 : (0#usize).val = 0 := by norm_num
    refine ⟨rfl, rfl, ?_, ?_, ?_⟩
    · rw [h0]; omega
    · rw [hacc0len]
    · intro j hj
      rw [h0, if_neg (by omega), add_zero, List.getD_eq_getElem a0 (0 : ℤ) hj]
      exact ha0 j hj

end

end apeiron_word

