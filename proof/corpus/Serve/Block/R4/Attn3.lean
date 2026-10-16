/-
Serving-path refinement of the extracted single-token attention head, stage 2
(`APEIRON-SERVE4-ATTN`). Continues `Serve.Block.Attn2`, which states the score loop
(`attn_head_loop0`), the softmax numerator loop (`attn_head_loop1`) and the inner
column-accumulate loop (`attn_head_loop2_loop0`). This module adds the two remaining
loops and the weight-stage reference they consume:

* `attn3_Weight` / `attn3_Out`: the plain-mathematical references of the weight
  (`round(e·2^32/total)`) and output (`round(acc_i/2^32)`) stages;
* `attn3_weight_at_ok`, `attn3_weight_at_recip_ok`: the extracted weight kernels refine
  the single reference `attn3_Weight` (re-exported from `Numeric.Engine.Attn`'s
  `weight_at_spec` and `Numeric.Attn.WeightRecipSpec`'s `weight_at_recip_spec`);
* `attn3_loop3_spec`: the output loop.

Trust boundary: the evidence reaches the extracted kernels and the integer references. It
does not cover rustc/LLVM codegen of `src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or
the `unsafe` container. The real-valued `attn_head_refines_v2` bound is not touched here.
-/
import Serve.Block.Attn2
import Numeric.Attn.WeightRecipSpec

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Stage references -/

/-- The weight reference: the fixed-point softmax weight `round(e·2^32 / total)` — one
half-biased `i128` division. This is the same single quantity `attn.weight_at` reaches
directly and `attn.weight_at_recip` reaches through the round-up `u64` reciprocal. -/
def attn3_Weight (e total : ℤ) : ℤ := divRoundNearest (e * (2 : ℤ) ^ (32 : ℕ)) total

/-- The output reference: `out_i = round(acc_i / 2^32)`, the `round_shift_32` epilogue,
read from the extracted accumulator. -/
def attn3_Out (accv : alloc.vec.Vec I128) (i : ℕ) : ℤ :=
  divRoundNearest ((accv.val.getD i 0#i128).val) ((2 : ℤ) ^ (32 : ℕ))

/-- **Weight reference, `weight_at` arm.** On the admitted window the extracted
`attn.weight_at e total` returns `Ok w` with `w.val = attn3_Weight e.val total.val`. -/
theorem attn3_weight_at_ok (e : I64) (total : I128)
    (hnum : 0 ≤ e.val) (htot : 0 < total.val)
    (hadd : e.val * ((2 : ℤ) ^ (32 : ℕ)) + total.val / 2 ≤ I128.max)
    (hfit : -((2 : ℤ) ^ (63 : ℕ)) ≤
        divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val ∧
      divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val ≤
        (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ w : I64, attn.weight_at e total = ok (core.result.Result.Ok w) ∧
      w.val = attn3_Weight e.val total.val :=
  weight_at_spec e total hnum htot hadd hfit

/-- **Weight reference, `weight_at_recip` arm.** On the admitted window (`total ≥ 2`,
`total ≤ 2^63-1`, the round-up reciprocal `recip = 2^64/total + 1`, and the `i128`
numerator window) the extracted `attn.weight_at_recip e total recip` returns `Ok w` with
the same `w.val = attn3_Weight e.val total.val`, whichever internal arm the numerator
narrowing selects. -/
theorem attn3_weight_at_recip_ok (e : I64) (total : I128) (recip : U64)
    (htot : 0 < total.val) (hnum : 0 ≤ e.val)
    (hreclo : 2 ≤ total.val.toNat) (hrechi : total.val.toNat ≤ 2 ^ 63 - 1)
    (hrecip : recip.val = 2 ^ 64 / total.val.toNat + 1)
    (haddI : e.val * ((2 : ℤ) ^ (32 : ℕ)) + total.val / 2 ≤ I128.max)
    (hfit : -((2 : ℤ) ^ (63 : ℕ)) ≤
        divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val ∧
      divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val ≤
        (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ w : I64, attn.weight_at_recip e total recip = ok (core.result.Result.Ok w) ∧
      w.val = attn3_Weight e.val total.val :=
  weight_at_recip_spec e total recip htot hnum hreclo hrechi hrecip haddI hfit

/-! ## Loop 3: the output stage -/

/-- The output-loop domain: the accumulator has the head width, the output slice is the head
width, and at every column the `round_shift_32` input and output stay in range. -/
def attn3_Loop3Dom (hd : Usize) (out : Slice I64) (accv : alloc.vec.Vec I128) : Prop :=
  accv.val.length = hd.val ∧ out.val.length = hd.val ∧ hd.val + 1 ≤ Usize.max ∧
  (∀ j : ℕ, j < hd.val →
    I128.min ≤ (accv.val.getD j 0#i128).val + (2 : ℤ) ^ (31 : ℕ) ∧
    (accv.val.getD j 0#i128).val + (2 : ℤ) ^ (31 : ℕ) ≤ I128.max ∧
    -((2 : ℤ) ^ (63 : ℕ)) ≤ attn3_Out accv j ∧ attn3_Out accv j ≤ (2 : ℤ) ^ (63 : ℕ) - 1)

/-- The output-loop invariant: the failure flag is clear, the error slot is empty, the column
index is at most `hd`, the slice keeps its width, and every written column `< i` holds
`attn3_Out` while every other column keeps the input value. -/
def attn3_Loop3Inv (hd : Usize) (out : Slice I64) (accv : alloc.vec.Vec I128)
    (st : Slice I64 × Usize × Bool × Option attn.AttnError) : Prop :=
  st.2.2.1 = true ∧ st.2.2.2 = none ∧ st.2.1.val ≤ hd.val ∧
  st.1.val.length = out.val.length ∧
  (∀ j : ℕ, j < st.1.val.length →
    (st.1.val.getD j 0#i64).val =
      if j < st.2.1.val then attn3_Out accv j else (out.val.getD j 0#i64).val)

/-- `getD` after a list `set` on `I64` lists. -/
theorem attn3_getD_set_val (l : List I64) (i j : ℕ) (a : I64) (hj : j < l.length) :
    (l.set i a).getD j (0#i64) = if j = i then a else l.getD j (0#i64) := by
  have hj' : j < (l.set i a).length := by rw [List.length_set]; exact hj
  rw [List.getD_eq_getElem (l.set i a) (0#i64) hj', List.getElem_set hj',
    List.getD_eq_getElem l (0#i64) hj]
  by_cases h : j = i
  · rw [if_pos h.symm, if_pos h]
  · rw [if_neg (fun hc => h hc.symm), if_neg h]

/-- One output-loop body step. From the invariant at column `i < hd`, the extracted body
reads `acc[i]`, rounds it by `round_shift_32`, narrows to `i64` on the success arm, writes
the result into `out[i]` and advances; under the domain every guard stays on its success
arm and the invariant is preserved. -/
theorem attn3_loop3_body_step (hd : Usize) (out : Slice I64) (accv : alloc.vec.Vec I128)
    (hdom : attn3_Loop3Dom hd out accv)
    (ost : Slice I64) (i : Usize)
    (hinv : attn3_Loop3Inv hd out accv (ost, i, true, none))
    (hlt : i.val < hd.val) :
    ∃ ost' i',
      attn.KvState.attn_head_loop3.body hd accv ost i true none
        = ok (ControlFlow.cont (ost', i', true, none)) ∧
      i'.val = i.val + 1 ∧
      attn3_Loop3Inv hd out accv (ost', i', true, none) := by
  obtain ⟨hacc, hout, hcap, hbounds⟩ := hdom
  obtain ⟨hok, herr, hile, hlen, hpoint⟩ := hinv
  dsimp only at hok herr hile hlen hpoint
  have hiacc : i.val < accv.val.length := by rw [hacc]; exact hlt
  have hiout : i.val < ost.val.length := by rw [hlen, hout]; exact hlt
  have hslt : i < hd := (UScalar.lt_equiv i hd).2 hlt
  have h1 : (1#usize).val = 1 := by norm_num
  have himax : i.val + (1#usize).val ≤ Usize.max := by rw [h1]; omega
  have hb := hbounds i.val hlt
  obtain ⟨hb31lo, hb31hi, hboutlo, hbouthi⟩ := hb
  -- the extracted accumulator read
  obtain ⟨r1, hr1eq, hr1val⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec accv i hiacc)
  have hr1v : r1.val = (accv.val.getD i.val 0#i128).val := by
    rw [hr1val, List.getD_eq_getElem accv.val (0#i128) hiacc]
  have hgetD : (accv.val.getD i.val 0#i128).val = r1.val := by rw [hr1v]
  -- `round_shift_32` on the read guard
  obtain ⟨rounded, hroundeq, hroundval⟩ :=
    round_shift_32_spec r1 (by rw [← hgetD]; exact hb31lo) (by rw [← hgetD]; exact hb31hi)
  have hrounded : rounded.val = attn3_Out accv i.val := by
    rw [hroundval]; unfold attn3_Out; rw [hgetD]
  have hmaxv : (core.convert.num.FromI128I64.from core.num.I64.MAX).val =
      (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
    rw [h]
  have hminv : (core.convert.num.FromI128I64.from core.num.I64.MIN).val =
      -((2 : ℤ) ^ (63 : ℕ)) := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MIN.val = -((2 : ℤ) ^ (63 : ℕ)) := by decide
    rw [h]
  have hmaxnot : ¬ (rounded > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc; rw [gt_iff_lt, IScalar.lt_equiv, hmaxv, hrounded] at hc; omega
  have hminnot : ¬ (rounded < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro hc; rw [IScalar.lt_equiv, hminv, hrounded] at hc; omega
  have hcast : (IScalar.cast .I64 rounded).val = rounded.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 rounded.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  have hcastval : (IScalar.cast .I64 rounded).val = attn3_Out accv i.val := by
    rw [hcast, hrounded]
  obtain ⟨s1, hs1eq, hs1val⟩ := WP.spec_imp_exists
    (Slice.update_spec ost i (IScalar.cast .I64 rounded) hiout)
  have hs1val' : s1.val = ost.val.set i.val (IScalar.cast .I64 rounded) := by
    rw [hs1val, Slice.set_val_eq]
  obtain ⟨i', hi'eq, hi'val⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) himax)
  refine ⟨s1, i', ?_, ?_, ?_⟩
  · rw [attn.KvState.attn_head_loop3.body, if_pos hslt, if_pos rfl]
    rw [alloc.vec.Vec.index_slice_index, hr1eq]
    simp only [lift, bind_tc_ok, hroundeq, if_neg hmaxnot, if_neg hminnot, hs1eq, hi'eq]
    rfl
  · rw [hi'val, h1]
  · refine ⟨rfl, rfl, ?_, ?_, ?_⟩
    · rw [hi'val, h1]; omega
    · rw [hs1val', List.length_set]; exact hlen
    · intro j hj
      have hjost : j < ost.val.length := by
        have h2 : j < (ost.val.set i.val (IScalar.cast .I64 rounded)).length := by
          rw [← hs1val']; exact hj
        rwa [List.length_set] at h2
      rw [hs1val', attn3_getD_set_val ost.val i.val j (IScalar.cast .I64 rounded) hjost]
      by_cases hji : j = i.val
      · rw [if_pos hji, hji, hcastval, hi'val, h1, if_pos (by omega)]
      · rw [if_neg hji, hpoint j hjost, hi'val, h1]
        by_cases hltj : j < i.val
        · rw [if_pos hltj, if_pos (by omega)]
        · rw [if_neg hltj, if_neg (by omega)]

/-- The output loop at the head width. From an output slice of width `hd`, an accumulator of
width `hd`, no error and the success flag set at column `0`, the extracted loop terminates on
its success arm at column `hd` with every component `round(acc_j/2^32)`. -/
theorem attn3_loop3_spec (hd : Usize) (out : Slice I64) (accv : alloc.vec.Vec I128)
    (hdom : attn3_Loop3Dom hd out accv)
    (ost0 : Slice I64) (host0 : ost0.val = out.val) :
    attn.KvState.attn_head_loop3 hd ost0 accv 0#usize true none ⦃ b =>
      ∃ out' : Slice I64, b = (out', none) ∧
        out'.val.length = out.val.length ∧
        (∀ j : ℕ, j < hd.val → (out'.val.getD j 0#i64).val = attn3_Out accv j) ⦄ := by
  obtain ⟨hacc, hout, hcap, hbounds⟩ := hdom
  unfold attn.KvState.attn_head_loop3
  apply loop.spec_decr_nat
    (measure := fun st : Slice I64 × Usize × Bool × Option attn.AttnError => hd.val - st.2.1.val)
    (inv := attn3_Loop3Inv hd out accv)
    (post := fun b : Slice I64 × Option attn.AttnError =>
      ∃ out' : Slice I64, b = (out', none) ∧
        out'.val.length = out.val.length ∧
        (∀ j : ℕ, j < hd.val → (out'.val.getD j 0#i64).val = attn3_Out accv j))
  · intro st hst
    rcases st with ⟨os, si, ok1, err⟩
    obtain ⟨hok1, herr, hle, hlen, hpoint⟩ := hst
    dsimp only at hok1 herr hle hlen hpoint
    rw [hok1, herr]
    by_cases hlt : si.val < hd.val
    · obtain ⟨os2, si2, hbody, hsi2, hinv2⟩ :=
        attn3_loop3_body_step hd out accv ⟨hacc, hout, hcap, hbounds⟩ os si
          ⟨rfl, rfl, hle, hlen, hpoint⟩ hlt
      dsimp only
      rw [hbody]
      refine ⟨hinv2, ?_⟩
      show hd.val - si2.val < hd.val - si.val
      rw [hsi2]; omega
    · have hne : ¬ (si < hd) := by
        intro hc; exact hlt ((UScalar.lt_equiv si hd).1 hc)
      have hsi_eq : si.val = hd.val := by omega
      have hbody : attn.KvState.attn_head_loop3.body hd accv os si true none
          = ok (ControlFlow.done (os, none)) := by
        rw [attn.KvState.attn_head_loop3.body, if_neg hne]
      dsimp only
      rw [hbody]
      refine ⟨os, rfl, hlen, ?_⟩
      intro j hj
      have hjlen : j < os.val.length := by rw [hlen, hout]; exact hj
      have := hpoint j hjlen
      rw [hsi_eq, if_pos hj] at this
      exact this
  · have h0 : (0#usize).val = 0 := by norm_num
    refine ⟨rfl, rfl, ?_, ?_, ?_⟩
    · rw [h0]; omega
    · rw [host0]
    · intro j hj
      rw [h0, if_neg (by omega), host0]

/-! ## Loop 2: the weight / value-accumulate stage -/

/-- The accumulated reference component `i` after the first `s` causal keys:
`a0_i + Σ_{j<s} f_j · v_{j,i}` where `v_{j,i}` is component `i` of the value row of key `j`.
Written as a left fold over `ℤ`, so the successor step is one addition and no list-length
invariant is needed. -/
def attn3_AccC (a0 : List ℤ) (f : ℕ → ℤ) (vplane : alloc.vec.Vec I64)
    (hd stride first : Usize) (s i : ℕ) : ℤ :=
  (List.range s).foldl
    (fun (a : ℤ) j => a + f j * ((attn2_Row vplane hd stride first j).getD i 0#i64).val)
    (a0.getD i 0)

/-- The accumulated reference extends by one key. -/
theorem attn3_accC_succ (a0 : List ℤ) (f : ℕ → ℤ) (vplane : alloc.vec.Vec I64)
    (hd stride first : Usize) (s i : ℕ) :
    attn3_AccC a0 f vplane hd stride first (s + 1) i =
      attn3_AccC a0 f vplane hd stride first s i
        + f s * ((attn2_Row vplane hd stride first s).getD i 0#i64).val := by
  unfold attn3_AccC
  rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- `List.getElem` respects list equality at a shared index, mapped through `f`. -/
theorem attn3_getElem_eq {α : Type} (l₁ l₂ : List α) (h : l₁ = l₂) (j : ℕ)
    (h₁ : j < l₁.length) (h₂ : j < l₂.length) (f : α → ℤ) :
    f (l₁[j]'h₁) = f (l₂[j]'h₂) := by
  cases h; rfl

/-- The domain of the weight/accumulate loop: the accumulator has the head width, the key index
and head width fit `usize`, every causal value row is inside the plane, and at every visited key
and column the checked `i128` product `f_j·v_{j,i}` and prefix sum stay in range. -/
def attn3_Loop2Dom (hd pos : Usize) (vplane : alloc.vec.Vec I64) (stride first : Usize)
    (f : ℕ → I64) (a0 : List ℤ) : Prop :=
  a0.length = hd.val ∧ pos.val + 1 ≤ Usize.max ∧ hd.val + 1 ≤ Usize.max ∧
  (∀ j : ℕ, j ≤ pos.val → first.val + j * stride.val + hd.val ≤ vplane.val.length) ∧
  (∀ (j : ℕ) (_ : j ≤ pos.val) (i : ℕ) (_ : i < hd.val),
     I128.min ≤ (f j).val * ((attn2_Row vplane hd stride first j).getD i 0#i64).val ∧
     (f j).val * ((attn2_Row vplane hd stride first j).getD i 0#i64).val ≤ I128.max ∧
     I128.min ≤ attn3_AccC a0 (fun j => (f j).val) vplane hd stride first j i
        + (f j).val * ((attn2_Row vplane hd stride first j).getD i 0#i64).val ∧
     attn3_AccC a0 (fun j => (f j).val) vplane hd stride first j i
        + (f j).val * ((attn2_Row vplane hd stride first j).getD i 0#i64).val ≤ I128.max)

/-- The weight/accumulate-loop invariant: the failure flag is clear, the error slot empty, the key
index at most `pos + 1`, the extracted weight of every causal key equals `f` (relational in the
`Some`/`None` reciprocal arm), the accumulator keeps its width, and every component is the
reference prefix `attn3_AccC` at the current key count. -/
def attn3_Loop2Inv (hd pos : Usize) (vplane : alloc.vec.Vec I64) (stride first : Usize)
    (exps : alloc.vec.Vec I64) (total : I128) (recip : Option U64)
    (a0 : List ℤ) (f : ℕ → I64)
    (st : alloc.vec.Vec I128 × Usize × Bool × Option attn.AttnError) : Prop :=
  st.2.2.1 = true ∧ st.2.2.2 = none ∧ st.2.1.val ≤ pos.val + 1 ∧
  (∀ (j : ℕ) (_ : j ≤ pos.val),
     (match recip with
      | none => attn.weight_at (exps.val.getD j 0#i64) total
      | some r => attn.weight_at_recip (exps.val.getD j 0#i64) total r)
       = ok (core.result.Result.Ok (f j))) ∧
  st.1.val.length = a0.length ∧
  (∀ (i : ℕ) (_ : i < a0.length),
     (st.1.val.getD i (0#i128)).val =
       attn3_AccC a0 (fun j => (f j).val) vplane hd stride first st.2.1.val i)

/-- One weight/accumulate body step. From the invariant at key `s ≤ pos`, the extracted body reads
`exps[s]`, forms its weight through the `weight_at`/`weight_at_recip` arm recorded by `f`, reads the
value row of key `s`, folds it into the accumulator with the inner column loop and advances; under
the domain every checked operation stays on its success arm and the invariant is preserved. -/
theorem attn3_loop2_body_step
    (hd pos : Usize) (vplane : alloc.vec.Vec I64) (stride first : Usize)
    (exps : alloc.vec.Vec I64) (total : I128) (recip : Option U64)
    (hlen : exps.val.length = pos.val + 1) (f : ℕ → I64) (a0 : List ℤ)
    (hdom : attn3_Loop2Dom hd pos vplane stride first f a0)
    (acc : alloc.vec.Vec I128) (s : Usize)
    (hinv : attn3_Loop2Inv hd pos vplane stride first exps total recip a0 f (acc, s, true, none))
    (hsle : s.val ≤ pos.val) :
    ∃ acc' s',
      attn.KvState.attn_head_loop2.body hd pos vplane stride first exps total recip acc s true none
        = ok (ControlFlow.cont (acc', s', true, none)) ∧
      s'.val = s.val + 1 ∧
      attn3_Loop2Inv hd pos vplane stride first exps total recip a0 f (acc', s', true, none) := by
  obtain ⟨ha0len, hposmax, hcap, hnowof, hbounds⟩ := hdom
  obtain ⟨hok, herr, hle, hW, hacclen, haccval⟩ := hinv
  dsimp only at hok herr hle hW hacclen haccval
  have hsle' : s ≤ pos := (UScalar.le_equiv s pos).2 hsle
  have hsid : s.val < exps.val.length := by rw [hlen]; omega
  have hvlen : vplane.val.length ≤ Usize.max := vplane.property
  -- read exps[s]
  obtain ⟨i1, hi1eq, hi1val⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec exps s hsid)
  have hi1getD : i1 = exps.val.getD s.val 0#i64 := by
    rw [hi1val]; exact (List.getD_eq_getElem exps.val (0#i64) hsid).symm
  -- the geometry of key `s`
  have hmul : s.val * stride.val ≤ Usize.max := by
    have := hnowof s.val hsle; omega
  obtain ⟨iv, hieq, hival⟩ := WP.spec_imp_exists (Usize.mul_spec (x := s) (y := stride) hmul)
  have hadd : first.val + iv.val ≤ Usize.max := by
    have := hnowof s.val hsle; omega
  obtain ⟨basev, hbaseeq, hbaseval⟩ := WP.spec_imp_exists (Usize.add_spec (x := first) (y := iv) hadd)
  have hi3val : (core.num.Usize.saturating_sub (alloc.vec.Vec.len vplane) hd).val
      = vplane.val.length - hd.val := by
    rw [attn2_usize_saturating_sub_val, attn2_vec_len_val]
    have := hnowof s.val hsle; omega
  have hnotb : ¬ (basev > core.num.Usize.saturating_sub (alloc.vec.Vec.len vplane) hd) := by
    intro hc
    rw [gt_iff_lt, UScalar.lt_equiv, hi3val] at hc
    have := hnowof s.val hsle; omega
  have hend : basev.val + hd.val ≤ Usize.max := by
    have := hnowof s.val hsle; omega
  obtain ⟨endv, hendeq, hendval⟩ := WP.spec_imp_exists (Usize.add_spec (x := basev) (y := hd) hend)
  have hstart : (⟨basev, endv⟩ : core.ops.range.Range Usize).start ≤
      (⟨basev, endv⟩ : core.ops.range.Range Usize).«end» := by
    show basev ≤ endv; rw [UScalar.le_equiv, hendval]; omega
  have hendle : ((⟨basev, endv⟩ : core.ops.range.Range Usize).«end»).val ≤ vplane.val.length := by
    show endv.val ≤ vplane.val.length
    rw [hendval, hbaseval, hival]
    exact hnowof s.val hsle
  obtain ⟨row, hroweq, hrowval, hrowlen⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_Range_spec vplane ⟨basev, endv⟩ hstart hendle)
  have hrowval_eq : row.val = List.slice basev.val endv.val vplane.val := by simpa using hrowval
  have hrowval' : row.val = attn2_Row vplane hd stride first s.val := by
    rw [hrowval_eq, hendval, hbaseval, hival]; rfl
  have hrowlen_eq : row.val.length = endv.val - basev.val := by simpa using hrowlen
  have hrowlen' : row.val.length = hd.val := by
    rw [hrowlen_eq, hendval, hbaseval, hival]; omega
  -- the inner column loop over this key's value row
  have hrouw : row.val.length = (acc.val.map (fun x => x.val)).length := by
    rw [hrowlen', List.length_map, hacclen, ha0len]
  have hhd' : hd.val = (acc.val.map (fun x => x.val)).length := by
    rw [List.length_map, hacclen, ha0len]
  have haccG : ∀ (i : ℕ) (hi : i < acc.val.length),
      (acc.val[i]'hi).val =
        attn3_AccC a0 (fun j => (f j).val) vplane hd stride first s.val i := by
    intro i hi
    rw [← List.getD_eq_getElem acc.val (0#i128) hi]
    exact haccval i (by rw [← hacclen]; exact hi)
  have hmapG : ∀ (i : ℕ) (hi : i < (acc.val.map (fun x => x.val)).length),
      (acc.val.map (fun x => x.val))[i] =
        attn3_AccC a0 (fun j => (f j).val) vplane hd stride first s.val i := by
    intro i hi
    have hi₁ : i < acc.val.length := by rwa [List.length_map] at hi
    rw [List.getElem_map]
    exact haccG i hi₁
  have hrowG : ∀ (i : ℕ) (hi : i < row.val.length),
      (row.val[i]'hi).val = ((attn2_Row vplane hd stride first s.val).getD i 0#i64).val := by
    intro i hi
    have hi' : i < (attn2_Row vplane hd stride first s.val).length := by rw [← hrowval']; exact hi
    rw [List.getD_eq_getElem (attn2_Row vplane hd stride first s.val) (0#i64) hi']
    exact attn3_getElem_eq row.val (attn2_Row vplane hd stride first s.val) hrowval' i hi hi'
      (fun x => x.val)
  have ha0map : ∀ (j : ℕ) (_ : j < (acc.val.map (fun x => x.val)).length),
      (acc.val.getD j (0#i128)).val = (acc.val.map (fun x => x.val))[j] := by
    intro j hj
    have hj' : j < acc.val.length := by rwa [List.length_map] at hj
    rw [List.getElem_map, List.getD_eq_getElem acc.val (0#i128) hj']
  have hdom0 : attn2_Loop2Loop0Dom row (core.convert.num.FromI128I64.from (f s.val))
      (acc.val.map (fun x => x.val)) hrouw := by
    intro j hj
    have hjlt : j < hd.val := by rw [hhd']; exact hj
    have hjrow : j < row.val.length := by rw [hrouw]; exact hj
    have hbnd := hbounds s.val hsle j hjlt
    obtain ⟨h1, h2, h3, h4⟩ := hbnd
    have hw1v : (core.convert.num.FromI128I64.from (f s.val)).val = (f s.val).val :=
      core.convert.num.FromI128I64.from_val_eq (f s.val)
    rw [hw1v, hrowG j hjrow, hmapG j hj]
    exact ⟨h1, h2, h3, h4⟩
  obtain ⟨b0, hb0eq, b0post⟩ := WP.spec_imp_exists
    (attn2_loop2_loop0_spec hd row (core.convert.num.FromI128I64.from (f s.val))
      (acc.val.map (fun x => x.val)) hrouw hhd' hcap acc (by rw [List.length_map]) ha0map hdom0)
  obtain ⟨acc1, hb0eq2, hacc1len, hacc1val⟩ := b0post
  -- advance the key index
  have hsucmax : s.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]; omega
  obtain ⟨s1, hs1eq, hs1val⟩ := WP.spec_imp_exists (Usize.add_spec (x := s) (y := 1#usize) hsucmax)
  refine ⟨acc1, s1, ?_, ?_, ?_⟩
  · cases recip with
    | none =>
      have hw0 : attn.weight_at (exps.val.getD s.val 0#i64) total =
          ok (core.result.Result.Ok (f s.val)) := by simpa using hW s.val hsle
      rw [attn.KvState.attn_head_loop2.body.eq_def, if_pos hsle', if_pos rfl]
      rw [alloc.vec.Vec.index_slice_index, hi1eq]
      simp only [lift, bind_tc_ok]
      rw [hi1getD, hw0]
      simp only [bind_tc_ok]
      rw [hieq]
      simp only [bind_tc_ok]
      rw [hbaseeq]
      simp only [bind_tc_ok]
      rw [if_neg hnotb]
      rw [hendeq]
      simp only [bind_tc_ok]
      rw [hroweq]
      simp only [bind_tc_ok]
      rw [hb0eq]
      rw [hb0eq2]
      simp only [bind_tc_ok]
      rw [hs1eq]
      simp only [bind_tc_ok]
      rfl
    | some rr =>
      have hw0 : attn.weight_at_recip (exps.val.getD s.val 0#i64) total rr =
          ok (core.result.Result.Ok (f s.val)) := by simpa using hW s.val hsle
      rw [attn.KvState.attn_head_loop2.body.eq_def, if_pos hsle', if_pos rfl]
      rw [alloc.vec.Vec.index_slice_index, hi1eq]
      simp only [lift, bind_tc_ok]
      rw [hi1getD, hw0]
      simp only [bind_tc_ok]
      rw [hieq]
      simp only [bind_tc_ok]
      rw [hbaseeq]
      simp only [bind_tc_ok]
      rw [if_neg hnotb]
      rw [hendeq]
      simp only [bind_tc_ok]
      rw [hroweq]
      simp only [bind_tc_ok]
      rw [hb0eq]
      rw [hb0eq2]
      simp only [bind_tc_ok]
      rw [hs1eq]
      simp only [bind_tc_ok]
      rfl
  · rw [hs1val]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
  · have h1 : (1#usize).val = 1 := by norm_num
    refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
    · rw [hs1val, h1]; omega
    · exact hW
    · rw [hacc1len, List.length_map, hacclen]
    · intro i hi
      rw [hs1val, h1, attn3_accC_succ]
      have hi'm : i < (acc.val.map (fun x => x.val)).length := by rw [List.length_map, hacclen]; exact hi
      have hpost := hacc1val i hi'm
      rw [hpost]
      have hw1v : (core.convert.num.FromI128I64.from (f s.val)).val = (f s.val).val :=
        core.convert.num.FromI128I64.from_val_eq (f s.val)
      rw [hmapG i hi'm, hw1v, hrowG i (by rw [hrouw]; exact hi'm)]

/-- The weight/accumulate loop at the causal prefix. From an accumulator of the head width holding
`a0` and no error, when the extracted weight of every causal key is `f` (the `weight_at` /
`weight_at_recip` success the domain admits) and the checked `i128` folds stay in range, the loop
terminates on its success arm at key `pos + 1` with every component the reference prefix `attn3_AccC`
over the causal keys. The weight arm is a hypothesis: it is the fixed-point boundary this stage
consumes, not an axiom. -/
theorem attn3_loop2_spec
    (hd pos : Usize) (vplane : alloc.vec.Vec I64) (stride first : Usize)
    (exps : alloc.vec.Vec I64) (total : I128) (recip : Option U64)
    (hlen : exps.val.length = pos.val + 1) (f : ℕ → I64) (a0 : List ℤ)
    (hdom : attn3_Loop2Dom hd pos vplane stride first f a0)
    (hW : ∀ (j : ℕ) (_ : j ≤ pos.val),
      (match recip with
       | none => attn.weight_at (exps.val.getD j 0#i64) total
       | some r => attn.weight_at_recip (exps.val.getD j 0#i64) total r)
        = ok (core.result.Result.Ok (f j)))
    (acc0 : alloc.vec.Vec I128) (hacc0len : acc0.val.length = a0.length)
    (hacc0 : ∀ (i : ℕ) (_ : i < a0.length),
      (acc0.val.getD i (0#i128)).val =
        attn3_AccC a0 (fun j => (f j).val) vplane hd stride first 0 i) :
    attn.KvState.attn_head_loop2 hd pos vplane stride first exps total recip acc0 0#usize true none ⦃ b =>
      ∃ acc' : alloc.vec.Vec I128, b = (acc', none) ∧
        acc'.val.length = a0.length ∧
        (∀ (i : ℕ) (_ : i < a0.length),
          (acc'.val.getD i (0#i128)).val =
            attn3_AccC a0 (fun j => (f j).val) vplane hd stride first (pos.val + 1) i) ⦄ := by
  unfold attn.KvState.attn_head_loop2
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec I128 × Usize × Bool × Option attn.AttnError =>
      pos.val + 1 - st.2.1.val)
    (inv := attn3_Loop2Inv hd pos vplane stride first exps total recip a0 f)
    (post := fun b : alloc.vec.Vec I128 × Option attn.AttnError =>
      ∃ acc' : alloc.vec.Vec I128, b = (acc', none) ∧
        acc'.val.length = a0.length ∧
        (∀ (i : ℕ) (_ : i < a0.length),
          (acc'.val.getD i (0#i128)).val =
            attn3_AccC a0 (fun j => (f j).val) vplane hd stride first (pos.val + 1) i))
  · intro st hst
    rcases st with ⟨ac, si, ok1, er⟩
    obtain ⟨hok1, herr, hle, hWs, hlens, haccv⟩ := hst
    dsimp only at hok1 herr hle hWs hlens haccv
    rw [hok1, herr]
    by_cases hlt : si.val ≤ pos.val
    · obtain ⟨ac2, si2, hbody, hsi2, hinv2⟩ :=
        attn3_loop2_body_step hd pos vplane stride first exps total recip hlen f a0 hdom ac si
          ⟨rfl, rfl, hle, hWs, hlens, haccv⟩ hlt
      dsimp only
      rw [hbody]
      refine ⟨hinv2, ?_⟩
      show pos.val + 1 - si2.val < pos.val + 1 - si.val
      rw [hsi2]; omega
    · have hne : ¬ (si ≤ pos) := by
        intro hc; exact hlt ((UScalar.le_equiv si pos).1 hc)
      have hsi_eq : si.val = pos.val + 1 := by omega
      have hbody : attn.KvState.attn_head_loop2.body hd pos vplane stride first exps total recip
          ac si true none = ok (ControlFlow.done (ac, none)) := by
        rw [attn.KvState.attn_head_loop2.body.eq_def, if_neg hne]
      dsimp only
      rw [hbody]
      refine ⟨ac, rfl, hlens, ?_⟩
      intro i hi
      have := haccv i hi
      rw [hsi_eq] at this
      exact this
  · have h0 : (0#usize).val = 0 := by norm_num
    refine ⟨rfl, rfl, ?_, hW, hacc0len, ?_⟩
    · rw [h0]; omega
    · intro i hi
      rw [h0]
      exact hacc0 i hi

/-! ## The driver -/

/-- **Driver composition.** The extracted `attn.KvState.attn_head` (through the `word_attn_head`
wrapper), past the carrier test, the four guard tests, the two plane reads, the two `usize` GQA
divisions and the three `checked`/`ok_or`/`branch` admissions, returns `Ok ()` exactly on the output
slice `out1` the output loop wrote, when the four loop bodies return their success arms with the
accumulators `scores1`/`mx`, `exps1`/`total`, `a1` and `out1`.

The statement is conditional, and that is the fixed-point boundary this stage consumes rather than
an axiom: the guard equations are the `fp.prim` facts the driver reads through `state.storage`,
`self.n_head / self.n_head_kv`, `q_head / group` and `Usize.checked_add`/`checked_mul`; the four loop
equations are the success-arm results of `attn_head_loop0..3`, each obtained from the corresponding
loop specification (`attn2_loop0_spec`, `attn2_loop1_spec`, `attn3_loop2_spec`, `attn3_loop3_spec`)
by `WP.spec_equiv_exists`. -/
theorem attn3_driver_spec (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcar : carrierQ32 state)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (k v : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstor : state.storage = attn.KvStorage.Q32 k v)
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) k layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) v layer = ok vplane)
    (grp kvh1 : Usize)
    (hgroup : state.n_head / state.n_head_kv = ok grp)
    (hkvh : q_head / grp = ok kvh1)
    (val : Usize) (hval : Usize.checked_add pos 1#usize = some val)
    (val1 : Usize) (hval1 : Usize.checked_mul state.n_head_kv state.head_dim = some val1)
    (val2 : Usize) (hval2 : Usize.checked_mul kvh1 state.head_dim = some val2)
    (scores1 : alloc.vec.Vec I64) (mx : I64)
    (hL0 : attn.KvState.attn_head_loop0 state.head_dim state.scale q pos kplane val1 val2
        (alloc.vec.Vec.with_capacity Std.I64 val) core.num.I64.MIN 0#usize true none
        = ok (scores1, mx, none))
    (exps1 : alloc.vec.Vec I64) (total : I128)
    (hL1 : attn.KvState.attn_head_loop1 pos scores1 mx
        (alloc.vec.Vec.with_capacity Std.I64 val) 0#i128 0#usize true none
        = ok (exps1, total, none))
    (htot : ¬ (total ≤ 0#i128))
    (recip : Option U64) (hrecip : attn.recip_u64 total = ok recip)
    (acc acc1 : alloc.vec.Vec I128)
    (hfrom : alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim = ok acc)
    (hL2 : attn.KvState.attn_head_loop2 state.head_dim pos vplane val1 val2 exps1 total recip
        acc 0#usize true none = ok (acc1, none))
    (out1 : Slice I64)
    (hL3 : attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
        = ok (out1, none)) :
    word_attn_head state layer q_head q pos out = ok (core.result.Result.Ok (), out1) := by
  rw [word_attn_head, attn.KvState.attn_head]
  simp only [mixer.Carrier.Insts.CoreCmpPartialEqCarrier.eq, bind_tc_ok]
  rw [if_neg hcar, if_neg hlayer, if_neg hhead, if_neg (ne_true_of_eq_false hq),
    if_neg (ne_true_of_eq_false hout)]
  rw [hlen]
  simp only [bind_tc_ok]
  rw [if_neg hpos]
  simp only [hstor]
  simp only [hkplane, hvplane, hgroup, hkvh, bind_tc_ok]
  rw [hval]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hval1]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hval2]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hL0]
  simp only [bind_tc_ok]
  show (do
    let (exps1, total, err1) ←
      attn.KvState.attn_head_loop1 pos scores1 mx (alloc.vec.Vec.with_capacity Std.I64 val)
        0#i128 0#usize true none
    match err1 with
    | none =>
      if total ≤ 0#i128 then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
      else do
        let recip ← attn.recip_u64 total
        let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
        let (acc1, err2) ←
          attn.KvState.attn_head_loop2 state.head_dim pos vplane val1 val2 exps1 total recip acc
            0#usize true none
        match err2 with
        | none => do
          let (out1, err3) ←
            attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
          match err3 with
          | none => ok (core.result.Result.Ok (), out1)
          | some e => ok (core.result.Result.Err e, out1)
        | some e => ok (core.result.Result.Err e, out)
    | some e => ok (core.result.Result.Err e, out)) =
    ok (core.result.Result.Ok (), out1)
  rw [hL1]
  simp only [bind_tc_ok]
  show (do
      if total ≤ 0#i128 then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
      else do
        let recip ← attn.recip_u64 total
        let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
        let (acc1, err2) ←
          attn.KvState.attn_head_loop2 state.head_dim pos vplane val1 val2 exps1 total recip acc
            0#usize true none
        match err2 with
        | none => do
          let (out1, err3) ←
            attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
          match err3 with
          | none => ok (core.result.Result.Ok (), out1)
          | some e => ok (core.result.Result.Err e, out1)
        | some e => ok (core.result.Result.Err e, out)) =
    ok (core.result.Result.Ok (), out1)
  rw [if_neg htot, hrecip, hfrom]
  simp only [bind_tc_ok]
  show (do
      let (acc1, err2) ←
        attn.KvState.attn_head_loop2 state.head_dim pos vplane val1 val2 exps1 total recip acc
          0#usize true none
      match err2 with
      | none => do
        let (out1, err3) ←
          attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
        match err3 with
        | none => ok (core.result.Result.Ok (), out1)
        | some e => ok (core.result.Result.Err e, out1)
      | some e => ok (core.result.Result.Err e, out)) =
    ok (core.result.Result.Ok (), out1)
  rw [hL2]
  simp only [bind_tc_ok]
  show (do
      let (out1, err3) ←
        attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
      match err3 with
      | none => ok (core.result.Result.Ok (), out1)
      | some e => ok (core.result.Result.Err e, out1)) =
    ok (core.result.Result.Ok (), out1)
  rw [hL3]
  simp only [bind_tc_ok]
  rfl

/-! ## Non-vacuity -/

/-- The weight reference is total and non-vacuous: a zero numerator rounds to zero. -/
example : attn3_Weight 0 1 = 0 := by
  unfold attn3_Weight
  norm_num [divRoundNearest]

/-- The accumulator reference at zero causal keys is the initial component, so the empty prefix is
not a vacuous identity. -/
example (a0 : List ℤ) (f : ℕ → ℤ) (vplane : alloc.vec.Vec I64) (hd stride first : Usize) (i : ℕ) :
    attn3_AccC a0 f vplane hd stride first 0 i = a0.getD i 0 := by
  unfold attn3_AccC
  simp

end

end apeiron_word
