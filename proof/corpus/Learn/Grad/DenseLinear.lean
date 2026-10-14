import ApeironWord
import Learn.Grad.DenseLinearSpec
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted `src/learn/grad/dense.rs` linear passes: refinement

`learn_port.dense.linear_fwd`, `linear_bwd_input` and `linear_bwd_weight` are the Q32.32 `i64`
linear forward and its two adjoints, extracted from the real source (`proof/extract/learn_port.rs`
roots `dense::linear_fwd` etc.). Each is refined, under an explicit admissible domain (the
`n × i` / `o × i` length equalities and the `i128` no-overflow bounds), to the independent `ℤ`
model in `Learn.Grad.DenseLinearSpec`: the exact accumulation plus the Q32.32 rounding
`(v + 2³¹) >>> 32`. The loop refinements are stated over the extracted `_loop` bodies via
`loop.spec_decr_nat`, not by kernel evaluation.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the vector container layout are not covered;
only the extracted integer body.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result
/-! ## 5. The `linear_fwd_acc` loops -/

theorem fwdK_succ (L R : List ℤ) (i r c k : ℕ) :
    fwdK L R i r c (k + 1) = fwdK L R i r c k + L.getD (r * i + k) 0 * R.getD (c * i + k) 0 := rfl

theorem fwdCols_zero (L R : List ℤ) (i r : ℕ) : fwdCols L R i r 0 = [] := rfl

theorem fwdCols_succ (L R : List ℤ) (i r c : ℕ) :
    fwdCols L R i r (c + 1) = fwdCols L R i r c ++ [fwdK L R i r c i] := by
  unfold fwdCols
  rw [pushN]

theorem fwdRows_succ (L R : List ℤ) (i o r : ℕ) :
    fwdRows L R i o (r + 1) = fwdRows L R i o r ++ fwdCols L R i r o := rfl

/-- The `linear_fwd_acc` inner-loop invariant (`linear_fwd_acc_loop0_loop0_loop0`). -/
def fwdInnerInv (x w : Slice I64) (i r c : Usize) (k : Usize) (acc : I128) (ok1 : Bool) : Prop :=
  ok1 = true ∧ k.val ≤ i.val ∧
    acc.val = fwdK (qvals x) (qvals w) i.val r.val c.val k.val

/-- The extracted `linear_fwd_acc` inner body on the succeeding path. -/
theorem linear_fwd_acc_loop0_loop0_loop0_body_exec
    (x w : Slice I64) (i r c : Usize) (k : Usize) (acc : I128)
    (ri ci i2v i6v : Usize) (p s : I128) (k1 : Usize)
    (hk : k < i)
    (hri : r * i = ok ri)
    (hi2 : ri + k = ok i2v)
    (hi2lt : i2v.val < x.val.length)
    (hci : c * i = ok ci)
    (hi6 : ci + k = ok i6v)
    (hi6lt : i6v.val < w.val.length)
    (hmul : I128.checked_mul (core.convert.num.FromI128I64.from (x.val.getD i2v.val default))
              (core.convert.num.FromI128I64.from (w.val.getD i6v.val default)) = some p)
    (hadd : I128.checked_add acc p = some s)
    (hk1 : k + 1#usize = ok k1) :
    learn_port.dense.linear_fwd_acc_loop0_loop0_loop0.body x w i r c true k acc =
      ok (ControlFlow.cont (true, k1, s)) := by
  have hx := dense_slice_index_getD x i2v hi2lt
  have hw := dense_slice_index_getD w i6v hi6lt
  rw [learn_port.dense.linear_fwd_acc_loop0_loop0_loop0.body]
  rw [if_pos hk, if_pos rfl]
  simp only [hri, hi2, hx, hci, hi6, hw, lift, hmul, hadd, hk1, bind_tc_ok]
  simp

/-- One iteration of the `linear_fwd_acc` inner loop advances the model. -/
theorem linear_fwd_acc_loop0_loop0_loop0_body_step
    (x w : Slice I64) (i r c : Usize) (k : Usize) (acc : I128) (ok1 : Bool)
    (hxd : r.val * i.val + i.val ≤ x.val.length)
    (hwd : c.val * i.val + i.val ≤ w.val.length)
    (hno : ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r.val c.val t ∧
      fwdK (qvals x) (qvals w) i.val r.val c.val t ≤ I128.max)
    (hkn : k.val < i.val)
    (hinv : fwdInnerInv x w i r c k acc ok1) :
    WP.spec (learn_port.dense.linear_fwd_acc_loop0_loop0_loop0.body x w i r c ok1 k acc)
      (fun cf => ∃ k' acc', cf = ControlFlow.cont (true, k', acc') ∧
        k'.val = k.val + 1 ∧ fwdInnerInv x w i r c k' acc' true) := by
  obtain ⟨hok, _hkle, hacc⟩ := hinv
  have hxmax : x.val.length ≤ Usize.max := Slice.property x
  have hwmax : w.val.length ≤ Usize.max := Slice.property w
  have hri_max : r.val * i.val ≤ Usize.max := by omega
  have hci_max : c.val * i.val ≤ Usize.max := by omega
  obtain ⟨ri, hri, hriv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := r) (y := i) hri_max)
  obtain ⟨i2v, hi2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := ri) (y := k) (by rw [hriv]; omega))
  obtain ⟨ci, hci, hciv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := c) (y := i) hci_max)
  obtain ⟨i6v, hi6, hi6v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := ci) (y := k) (by rw [hciv]; omega))
  obtain ⟨k1, hk1, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize)
      (by show k.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          have := dense_usize_le_max i
          omega))
  have hk1_val : k1.val = k.val + 1 := by rw [hk1v]; rfl
  have hidx2 : i2v.val = r.val * i.val + k.val := by rw [hi2v, hriv]
  have hidx6 : i6v.val = c.val * i.val + k.val := by rw [hi6v, hciv]
  have hi2lt : i2v.val < x.val.length := by rw [hidx2]; omega
  have hi6lt : i6v.val < w.val.length := by rw [hidx6]; omega
  have hmulb : I128.min ≤
        (core.convert.num.FromI128I64.from (x.val.getD i2v.val default)).val *
        (core.convert.num.FromI128I64.from (w.val.getD i6v.val default)).val ∧
      (core.convert.num.FromI128I64.from (x.val.getD i2v.val default)).val *
        (core.convert.num.FromI128I64.from (w.val.getD i6v.val default)).val ≤ I128.max := by
    rw [dense_fromI64_val, dense_fromI64_val]
    exact dense_i64_mul_fits _ _
  obtain ⟨p, hmul, hpv⟩ :=
    dense_checked_mul_some (a := core.convert.num.FromI128I64.from (x.val.getD i2v.val default))
      (b := core.convert.num.FromI128I64.from (w.val.getD i6v.val default)) hmulb
  have hLg : (x.val.getD i2v.val default).val = (qvals x).getD (r.val * i.val + k.val) 0 := by
    rw [← hidx2]; exact (qvals_getD x hi2lt).symm
  have hRg : (w.val.getD i6v.val default).val = (qvals w).getD (c.val * i.val + k.val) 0 := by
    rw [← hidx6]; exact (qvals_getD w hi6lt).symm
  have hpval : p.val =
      (qvals x).getD (r.val * i.val + k.val) 0 * (qvals w).getD (c.val * i.val + k.val) 0 := by
    rw [hpv, dense_fromI64_val, dense_fromI64_val, hLg, hRg]
  have hsum_in : I128.min ≤ acc.val + p.val ∧ acc.val + p.val ≤ I128.max := by
    have he : acc.val + p.val = fwdK (qvals x) (qvals w) i.val r.val c.val (k.val + 1) := by
      rw [hacc, hpval]; exact (fwdK_succ (qvals x) (qvals w) i.val r.val c.val k.val).symm
    rw [he]; exact hno (k.val + 1) (by omega)
  obtain ⟨s, hadd, hsv⟩ := dense_checked_add_some hsum_in
  have hsval : s.val = fwdK (qvals x) (qvals w) i.val r.val c.val k1.val := by
    rw [hsv, hacc, hpval, hk1_val]
    exact (fwdK_succ (qvals x) (qvals w) i.val r.val c.val k.val).symm
  have hkProp : k < i := (UScalar.lt_equiv k i).2 hkn
  have hexec := linear_fwd_acc_loop0_loop0_loop0_body_exec x w i r c k acc ri ci i2v i6v p s k1
    hkProp hri hi2 hi2lt hci hi6 hi6lt hmul hadd hk1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (true, k1, s), hexec, k1, s, rfl, hk1_val, ?_⟩
  exact ⟨rfl, by omega, hsval⟩

/-- Full specification of the `linear_fwd_acc` inner loop. -/
theorem linear_fwd_acc_loop0_loop0_loop0_spec
    (x w : Slice I64) (i r c : Usize) (k : Usize) (acc : I128)
    (hxd : r.val * i.val + i.val ≤ x.val.length)
    (hwd : c.val * i.val + i.val ≤ w.val.length)
    (hno : ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r.val c.val t ∧
      fwdK (qvals x) (qvals w) i.val r.val c.val t ≤ I128.max)
    (hinv0 : fwdInnerInv x w i r c k acc true) :
    WP.spec (learn_port.dense.linear_fwd_acc_loop0_loop0_loop0 x w i r true c k acc)
      (fun (ok', acc') => ok' = true ∧
        acc'.val = fwdK (qvals x) (qvals w) i.val r.val c.val i.val) := by
  unfold learn_port.dense.linear_fwd_acc_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Usize × I128 => i.val - s.2.1.val)
    (inv := fun s => fwdInnerInv x w i r c s.2.1 s.2.2 s.1)
    (post := fun b : Bool × I128 => b.1 = true ∧
      b.2.val = fwdK (qvals x) (qvals w) i.val r.val c.val i.val)
  · rintro ⟨ok', k', acc'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hk'le, hacc'⟩ := hinv
    rw [hok']
    by_cases hkn : k'.val < i.val
    · obtain ⟨cf, hstep, k'', acc'', hcf, hk''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_fwd_acc_loop0_loop0_loop0_body_step x w i r c k' acc' true hxd hwd hno hkn
          ⟨rfl, hk'le, hacc'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (k' < i) := fun hc => hkn ((UScalar.lt_equiv k' i).1 hc)
      have hbody : learn_port.dense.linear_fwd_acc_loop0_loop0_loop0.body x w i r c true k' acc' =
          ok (ControlFlow.done (true, acc')) := by
        rw [learn_port.dense.linear_fwd_acc_loop0_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      have hk'_eq : k'.val = i.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hacc', hk'_eq]
  · exact hinv0

/-! ### The middle loop (`c < o`) -/

/-- The `linear_fwd_acc` middle-loop invariant (`linear_fwd_acc_loop0_loop0`). -/
def fwdMidInv (x w : Slice I64) (i o r : Usize) (out : alloc.vec.Vec I128) (c : Usize)
    (ok1 : Bool) : Prop :=
  ok1 = true ∧ c.val ≤ o.val ∧
    ovals out = fwdRows (qvals x) (qvals w) i.val o.val r.val ++
      fwdCols (qvals x) (qvals w) i.val r.val c.val

/-- The extracted `linear_fwd_acc` middle body on the succeeding path. -/
theorem linear_fwd_acc_loop0_loop0_body_exec
    (x w : Slice I64) (i o r : Usize) (out : alloc.vec.Vec I128) (c : Usize)
    (acc : I128) (out1 : alloc.vec.Vec I128) (c1 : Usize)
    (hc : c < o)
    (hinner : learn_port.dense.linear_fwd_acc_loop0_loop0_loop0 x w i r true c 0#usize 0#i128
      = ok (true, acc))
    (hp : alloc.vec.Vec.push out acc = ok out1)
    (hc1 : c + 1#usize = ok c1) :
    learn_port.dense.linear_fwd_acc_loop0_loop0.body x w i o r out true c =
      ok (ControlFlow.cont (out1, true, c1)) := by
  rw [learn_port.dense.linear_fwd_acc_loop0_loop0.body]
  rw [if_pos hc, if_pos rfl]
  simp only [hinner]
  simp [hp, hc1]

/-- One iteration of the `linear_fwd_acc` middle loop advances the model. -/
theorem linear_fwd_acc_loop0_loop0_body_step
    (x w : Slice I64) (i o r : Usize) (out : alloc.vec.Vec I128) (c : Usize) (ok1 : Bool)
    (hxd : r.val * i.val + i.val ≤ x.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hcap : r.val * o.val + o.val ≤ Usize.max)
    (hno : ∀ c' < o.val, ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r.val c' t ∧
      fwdK (qvals x) (qvals w) i.val r.val c' t ≤ I128.max)
    (hco : c.val < o.val)
    (hinv : fwdMidInv x w i o r out c ok1) :
    WP.spec (learn_port.dense.linear_fwd_acc_loop0_loop0.body x w i o r out ok1 c)
      (fun cf => ∃ out' c', cf = ControlFlow.cont (out', true, c') ∧ c'.val = c.val + 1 ∧
        fwdMidInv x w i o r out' c' true) := by
  obtain ⟨hok, _hcle, hout⟩ := hinv
  have hwd' : c.val * i.val + i.val ≤ w.val.length := by
    calc c.val * i.val + i.val = (c.val + 1) * i.val := by rw [Nat.succ_mul]
      _ ≤ o.val * i.val := Nat.mul_le_mul_right i.val (Nat.succ_le_of_lt hco)
      _ ≤ w.val.length := hwo
  have hinner0 : fwdInnerInv x w i r c 0#usize 0#i128 true := by
    refine ⟨rfl, by show (0#usize).val ≤ i.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show (0#i128).val = fwdK (qvals x) (qvals w) i.val r.val c.val (0#usize).val
    rw [show (0#i128).val = 0 from rfl, show (0#usize).val = 0 from rfl]
    rfl
  have hinner := linear_fwd_acc_loop0_loop0_loop0_spec x w i r c 0#usize 0#i128 hxd hwd'
    (hno c.val hco) hinner0
  obtain ⟨⟨ok2, acc⟩, hinner_exec, hok2, haccv⟩ := (WP.spec_equiv_exists _ _).mp hinner
  subst hok2
  have hlenout : out.val.length < Usize.max := by
    have h1 : (ovals out).length = out.val.length := ovals_length out
    rw [← h1, hout, List.length_append, fwdRows_length, fwdCols_length]
    omega
  obtain ⟨out1, hp, hout1⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out acc hlenout)
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := c) (y := 1#usize)
      (by show c.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          omega))
  have hc1_val : c1.val = c.val + 1 := by rw [hc1v]; rfl
  have hcProp : c < o := (UScalar.lt_equiv c o).2 hco
  have hexec := linear_fwd_acc_loop0_loop0_body_exec x w i o r out c acc out1 c1 hcProp
    hinner_exec hp hc1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out1, true, c1), hexec, out1, c1, rfl, hc1_val, ?_⟩
  refine ⟨rfl, by omega, ?_⟩
  have hval1 : ovals out1 = ovals out ++ [acc.val] := by simp [ovals, hout1]
  rw [hval1, hout, haccv, hc1_val, fwdCols_succ, List.append_assoc]

/-- Full specification of the `linear_fwd_acc` middle loop. -/
theorem linear_fwd_acc_loop0_loop0_spec
    (x w : Slice I64) (i o r : Usize) (out : alloc.vec.Vec I128) (c : Usize)
    (hxd : r.val * i.val + i.val ≤ x.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hcap : r.val * o.val + o.val ≤ Usize.max)
    (hno : ∀ c' < o.val, ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r.val c' t ∧
      fwdK (qvals x) (qvals w) i.val r.val c' t ≤ I128.max)
    (hinv0 : fwdMidInv x w i o r out c true) :
    WP.spec (learn_port.dense.linear_fwd_acc_loop0_loop0 x w i o out r true c)
      (fun (out', ok') => ok' = true ∧
        ovals out' = fwdRows (qvals x) (qvals w) i.val o.val r.val ++
          fwdCols (qvals x) (qvals w) i.val r.val o.val) := by
  unfold learn_port.dense.linear_fwd_acc_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I128 × Bool × Usize => o.val - s.2.2.val)
    (inv := fun s => fwdMidInv x w i o r s.1 s.2.2 s.2.1)
    (post := fun b : alloc.vec.Vec I128 × Bool => b.2 = true ∧
      ovals b.1 = fwdRows (qvals x) (qvals w) i.val o.val r.val ++
        fwdCols (qvals x) (qvals w) i.val r.val o.val)
  · rintro ⟨out', ok', c'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hc'le, hout'⟩ := hinv
    rw [hok']
    by_cases hco : c'.val < o.val
    · obtain ⟨cf, hstep, out'', c'', hcf, hc''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_fwd_acc_loop0_loop0_body_step x w i o r out' c' true hxd hwo hcap hno hco
          ⟨rfl, hc'le, hout'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (c' < o) := fun hc => hco ((UScalar.lt_equiv c' o).1 hc)
      have hbody : learn_port.dense.linear_fwd_acc_loop0_loop0.body x w i o r out' true c' =
          ok (ControlFlow.done (out', true)) := by
        rw [learn_port.dense.linear_fwd_acc_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      have hc'_eq : c'.val = o.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hout', hc'_eq]
  · exact hinv0

/-! ### The outer loop (`r < n`) -/

/-- The `linear_fwd_acc` outer-loop invariant (`linear_fwd_acc_loop0`). -/
def fwdOuterInv (x w : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128)
    (r : Usize) (ok1 : Bool) : Prop :=
  ok1 = true ∧ r.val ≤ n.val ∧
    ovals out = fwdRows (qvals x) (qvals w) i.val o.val r.val

/-- The extracted `linear_fwd_acc` outer body on the succeeding path. -/
theorem linear_fwd_acc_loop0_body_exec
    (x w : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (r : Usize)
    (out1 : alloc.vec.Vec I128) (r1 : Usize)
    (hr : r < n)
    (hmid : learn_port.dense.linear_fwd_acc_loop0_loop0 x w i o out r true 0#usize
      = ok (out1, true))
    (hr1 : r + 1#usize = ok r1) :
    learn_port.dense.linear_fwd_acc_loop0.body x w n i o out r true =
      ok (ControlFlow.cont (out1, r1, true)) := by
  rw [learn_port.dense.linear_fwd_acc_loop0.body]
  rw [if_pos hr, if_pos rfl]
  simp only [hmid]
  simp [hr1]

/-- One iteration of the `linear_fwd_acc` outer loop advances the model. -/
theorem linear_fwd_acc_loop0_body_step
    (x w : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (r : Usize) (ok1 : Bool)
    (hxn : n.val * i.val ≤ x.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hcap : n.val * o.val ≤ Usize.max)
    (hno : ∀ r' < n.val, ∀ c' < o.val, ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r' c' t ∧
      fwdK (qvals x) (qvals w) i.val r' c' t ≤ I128.max)
    (hrn : r.val < n.val)
    (hinv : fwdOuterInv x w n i o out r ok1) :
    WP.spec (learn_port.dense.linear_fwd_acc_loop0.body x w n i o out r ok1)
      (fun cf => ∃ out' r', cf = ControlFlow.cont (out', r', true) ∧ r'.val = r.val + 1 ∧
        fwdOuterInv x w n i o out' r' true) := by
  obtain ⟨hok, _hrle, hout⟩ := hinv
  have hxd : r.val * i.val + i.val ≤ x.val.length := by
    calc r.val * i.val + i.val = (r.val + 1) * i.val := by rw [Nat.succ_mul]
      _ ≤ n.val * i.val := Nat.mul_le_mul_right i.val (Nat.succ_le_of_lt hrn)
      _ ≤ x.val.length := hxn
  have hcapr : r.val * o.val + o.val ≤ Usize.max := by
    calc r.val * o.val + o.val = (r.val + 1) * o.val := by rw [Nat.succ_mul]
      _ ≤ n.val * o.val := Nat.mul_le_mul_right o.val (Nat.succ_le_of_lt hrn)
      _ ≤ Usize.max := hcap
  have hno_r : ∀ c' < o.val, ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r.val c' t ∧
      fwdK (qvals x) (qvals w) i.val r.val c' t ≤ I128.max :=
    fun c' hc' => hno r.val hrn c' hc'
  have hmid0 : fwdMidInv x w i o r out 0#usize true := by
    refine ⟨rfl, by show (0#usize).val ≤ o.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show ovals out = fwdRows (qvals x) (qvals w) i.val o.val r.val ++
      fwdCols (qvals x) (qvals w) i.val r.val (0#usize).val
    rw [show (0#usize).val = 0 from rfl, fwdCols_zero, List.append_nil]
    exact hout
  have hmid := linear_fwd_acc_loop0_loop0_spec x w i o r out 0#usize hxd hwo hcapr hno_r hmid0
  obtain ⟨⟨out1, ok2⟩, hmid_exec, hok2, hout1⟩ := (WP.spec_equiv_exists _ _).mp hmid
  subst hok2
  obtain ⟨r1, hr1, hr1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := r) (y := 1#usize)
      (by show r.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          have := dense_usize_le_max n
          omega))
  have hr1_val : r1.val = r.val + 1 := by rw [hr1v]; rfl
  have hrProp : r < n := (UScalar.lt_equiv r n).2 hrn
  have hexec := linear_fwd_acc_loop0_body_exec x w n i o out r out1 r1 hrProp hmid_exec hr1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out1, r1, true), hexec, out1, r1, rfl, hr1_val, ?_⟩
  refine ⟨rfl, by omega, ?_⟩
  rw [hout1, hr1_val, fwdRows_succ]

/-- Full specification of the `linear_fwd_acc` outer loop. -/
theorem linear_fwd_acc_loop0_spec
    (x w : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (r : Usize)
    (hxn : n.val * i.val ≤ x.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hcap : n.val * o.val ≤ Usize.max)
    (hno : ∀ r' < n.val, ∀ c' < o.val, ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r' c' t ∧
      fwdK (qvals x) (qvals w) i.val r' c' t ≤ I128.max)
    (hinv0 : fwdOuterInv x w n i o out r true) :
    WP.spec (learn_port.dense.linear_fwd_acc_loop0 x w n i o out r true)
      (fun (out', ok') => ok' = true ∧
        ovals out' = fwdRows (qvals x) (qvals w) i.val o.val n.val) := by
  unfold learn_port.dense.linear_fwd_acc_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I128 × Usize × Bool => n.val - s.2.1.val)
    (inv := fun s => fwdOuterInv x w n i o s.1 s.2.1 s.2.2)
    (post := fun b : alloc.vec.Vec I128 × Bool => b.2 = true ∧
      ovals b.1 = fwdRows (qvals x) (qvals w) i.val o.val n.val)
  · rintro ⟨out', r', ok'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hr'le, hout'⟩ := hinv
    rw [hok']
    by_cases hrn : r'.val < n.val
    · obtain ⟨cf, hstep, out'', r'', hcf, hr''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_fwd_acc_loop0_body_step x w n i o out' r' true hxn hwo hcap hno hrn
          ⟨rfl, hr'le, hout'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (r' < n) := fun hc => hrn ((UScalar.lt_equiv r' n).1 hc)
      have hbody : learn_port.dense.linear_fwd_acc_loop0.body x w n i o out' r' true =
          ok (ControlFlow.done (out', true)) := by
        rw [learn_port.dense.linear_fwd_acc_loop0.body, if_neg hnot]
      simp only [hbody]
      have hr'_eq : r'.val = n.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hout', hr'_eq]
  · exact hinv0

/-- The empty `i128` vector has no values. -/
theorem ovals_with_capacity (n : Usize) :
    ovals (alloc.vec.Vec.with_capacity I128 n) = [] := by
  show (alloc.vec.Vec.with_capacity I128 n).val.map (fun z => z.val) = []
  rw [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]
  rfl

/-- Draining an `i128` vector to a slice preserves its values. -/
theorem ivals_deref (v : alloc.vec.Vec I128) : ivals (alloc.vec.Vec.deref v) = ovals v := by
  simp [ivals, ovals, alloc.vec.Vec.deref, alloc.vec.Vec.val]

theorem deref_length (v : alloc.vec.Vec I128) :
    (alloc.vec.Vec.deref v).val.length = v.val.length := by
  simp [alloc.vec.Vec.deref, alloc.vec.Vec.val]

/-- `linear_fwd_acc` computes the unrounded `i128` product `x·wᵀ`, row-major. -/
theorem linear_fwd_acc_spec
    (x w : Slice I64) (n i o : Usize)
    (hxn : x.val.length = n.val * i.val)
    (hwo : w.val.length = o.val * i.val)
    (hprod : n.val * o.val ≤ Usize.max)
    (hno : ∀ r' < n.val, ∀ c' < o.val, ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r' c' t ∧
      fwdK (qvals x) (qvals w) i.val r' c' t ≤ I128.max) :
    ∃ out, learn_port.dense.linear_fwd_acc x w n i o = ok (core.result.Result.Ok out) ∧
      ovals out = fwdRows (qvals x) (qvals w) i.val o.val n.val := by
  have hxn' : n.val * i.val ≤ x.val.length := by rw [hxn]
  have hwo' : o.val * i.val ≤ w.val.length := by rw [hwo]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := n) (y := o) hprod)
  have hinv0 : fwdOuterInv x w n i o (alloc.vec.Vec.with_capacity I128 i1) 0#usize true := by
    refine ⟨rfl, by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show ovals (alloc.vec.Vec.with_capacity I128 i1) =
      fwdRows (qvals x) (qvals w) i.val o.val (0#usize).val
    rw [show (0#usize).val = 0 from rfl, ovals_with_capacity]
    rfl
  have hspec := linear_fwd_acc_loop0_spec x w n i o (alloc.vec.Vec.with_capacity I128 i1) 0#usize
    hxn' hwo' hprod hno hinv0
  obtain ⟨⟨out1, ok1⟩, hloop, hok1, hout1⟩ := (WP.spec_equiv_exists _ _).mp hspec
  subst hok1
  refine ⟨out1, ?_, hout1⟩
  unfold learn_port.dense.linear_fwd_acc
  dsimp only
  rw [hi1]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  simp

/-! ## 6. `Try` reductions and the `linear_fwd` headline -/

lemma dense_branch_ok {α ε : Type} (v : α) :
    core.result.Result.Insts.CoreOpsTry.branch (T := α) (E := ε) (core.result.Result.Ok v)
      = ok (core.ops.control_flow.ControlFlow.Continue v) := rfl

lemma dense_from_residual_err {T ε : Type} (e : ε) :
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual T
      (core.convert.FromSame ε) (core.result.Result.Err e) = ok (core.result.Result.Err e) := by
  simp only [core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, bind_tc_ok]

/-- `linear_fwd` rounds the exact `i128` product `x·wᵀ` to Q32.32. -/
theorem linear_fwd_ok (x w : Slice I64) (n i o : Usize)
    (hn : n ≠ 0#usize) (hi : i ≠ 0#usize) (ho : o ≠ 0#usize)
    (hx : x.val.length = n.val * i.val)
    (hw : w.val.length = o.val * i.val)
    (hni : n.val * i.val ≤ Usize.max)
    (hoi : o.val * i.val ≤ Usize.max)
    (hno_prod : n.val * o.val ≤ Usize.max)
    (hbound : ∀ r' < n.val, ∀ c' < o.val, ∀ t ≤ i.val,
      I128.min ≤ fwdK (qvals x) (qvals w) i.val r' c' t ∧
      fwdK (qvals x) (qvals w) i.val r' c' t ≤ I128.max)
    (hround : ∀ t < n.val * o.val,
      I128.min ≤ (fwdRows (qvals x) (qvals w) i.val o.val n.val).getD t 0 + 2 ^ 31 ∧
      (fwdRows (qvals x) (qvals w) i.val o.val n.val).getD t 0 + 2 ^ 31 ≤ I128.max)
    (hnar : ∀ t < n.val * o.val,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((fwdRows (qvals x) (qvals w) i.val o.val n.val).getD t 0) ∧
      rnd32 ((fwdRows (qvals x) (qvals w) i.val o.val n.val).getD t 0) ≤ (2 : ℤ) ^ 63 - 1) :
    ∃ out, learn_port.dense.linear_fwd x w n i o = ok (core.result.Result.Ok out) ∧
      vvals out = rndAll (fwdRows (qvals x) (qvals w) i.val o.val n.val) (n.val * o.val) := by
  have hexp1 := dense_expect_len_ok n i (Slice.len x) hn hi hni (by rw [Slice.len_val]; exact hx)
  have hexp2 := dense_expect_len_ok o i (Slice.len w) ho hi hoi (by rw [Slice.len_val]; exact hw)
  obtain ⟨acc, hacc, haccv⟩ := linear_fwd_acc_spec x w n i o hx hw hno_prod hbound
  have hlenacc : acc.val.length = n.val * o.val := by
    rw [← ovals_length, haccv, fwdRows_length]
  have hlen_deref : (alloc.vec.Vec.deref acc).val.length = n.val * o.val := by
    rw [deref_length, hlenacc]
  have hround' : ∀ t < (alloc.vec.Vec.deref acc).val.length,
      I128.min ≤ (ivals (alloc.vec.Vec.deref acc)).getD t 0 + 2 ^ 31 ∧
      (ivals (alloc.vec.Vec.deref acc)).getD t 0 + 2 ^ 31 ≤ I128.max := by
    intro t ht
    rw [ivals_deref, haccv]
    exact hround t (by rw [hlen_deref] at ht; exact ht)
  have hnar' : ∀ t < (alloc.vec.Vec.deref acc).val.length,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((ivals (alloc.vec.Vec.deref acc)).getD t 0) ∧
      rnd32 ((ivals (alloc.vec.Vec.deref acc)).getD t 0) ≤ (2 : ℤ) ^ 63 - 1 := by
    intro t ht
    rw [ivals_deref, haccv]
    exact hnar t (by rw [hlen_deref] at ht; exact ht)
  obtain ⟨out, hra, hrav⟩ := round_all_spec (alloc.vec.Vec.deref acc) hround' hnar'
  have hrav' : vvals out =
      rndAll (fwdRows (qvals x) (qvals w) i.val o.val n.val) (n.val * o.val) := by
    rw [hrav, ivals_deref, haccv, deref_length, hlenacc]
  refine ⟨out, ?_, hrav'⟩
  unfold learn_port.dense.linear_fwd
  simp only [hexp1, hexp2, hacc, hra, dense_branch_ok, bind_tc_ok]


/-! ## 6. The `linear_bwd_input_acc` loops -/

/-- The `linear_bwd_input_acc` inner accumulator `Σ_{t<c} dy[r·o + t] · w[t·i + k]`, count-indexed. -/
def bwdIAcc (dy w : Slice I64) (i o : Usize) (r k : ℕ) : ℕ → ℤ :=
  bwdIK (qvals dy) (qvals w) i.val o.val r k

theorem bwdIAcc_succ (dy w : Slice I64) (i o : Usize) (r k c : ℕ) :
    bwdIAcc dy w i o r k (c + 1) =
      bwdIAcc dy w i o r k c + (qvals dy).getD (r * o.val + c) 0 *
        (qvals w).getD (c * i.val + k) 0 := rfl

/-- The `linear_bwd_input_acc` output row-major element model. -/
def bwdIElem (dy w : Slice I64) (i o : Usize) : ℕ → ℕ → ℤ :=
  fun r k => bwdIAcc dy w i o r k o.val

/-- The `linear_bwd_input_acc` inner-loop invariant (`..._loop0_loop0_loop0`). -/
def bwdIInnerInv (dy w : Slice I64) (i o r k : Usize) (c : Usize) (acc : I128)
    (ok1 : Bool) : Prop :=
  ok1 = true ∧ c.val ≤ o.val ∧ acc.val = bwdIAcc dy w i o r.val k.val c.val

/-- The extracted `linear_bwd_input_acc` inner body on the succeeding path. -/
theorem linear_bwd_input_acc_loop0_loop0_loop0_body_exec
    (dy w : Slice I64) (i o r k : Usize) (c : Usize) (acc : I128)
    (ro c2v ci k6v : Usize) (p s : I128) (c1 : Usize)
    (hc : c < o)
    (hro : r * o = ok ro)
    (hc2 : ro + c = ok c2v)
    (hc2lt : c2v.val < dy.val.length)
    (hci : c * i = ok ci)
    (hk6 : ci + k = ok k6v)
    (hk6lt : k6v.val < w.val.length)
    (hmul : I128.checked_mul (core.convert.num.FromI128I64.from (dy.val.getD c2v.val default))
              (core.convert.num.FromI128I64.from (w.val.getD k6v.val default)) = some p)
    (hadd : I128.checked_add acc p = some s)
    (hc1 : c + 1#usize = ok c1) :
    learn_port.dense.linear_bwd_input_acc_loop0_loop0_loop0.body dy w i o r k true c acc =
      ok (ControlFlow.cont (true, c1, s)) := by
  have hd := dense_slice_index_getD dy c2v hc2lt
  have hw := dense_slice_index_getD w k6v hk6lt
  rw [learn_port.dense.linear_bwd_input_acc_loop0_loop0_loop0.body]
  rw [if_pos hc, if_pos rfl]
  simp only [hro, hc2, hd, hci, hk6, hw, lift, hmul, hadd, hc1, bind_tc_ok]
  simp

/-- One iteration of the `linear_bwd_input_acc` inner loop advances the model. -/
theorem linear_bwd_input_acc_loop0_loop0_loop0_body_step
    (dy w : Slice I64) (i o r k : Usize) (c : Usize) (acc : I128) (ok1 : Bool)
    (hdy : r.val * o.val + o.val ≤ dy.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hki : k.val < i.val)
    (hno : ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r.val k.val t ∧ bwdIAcc dy w i o r.val k.val t ≤ I128.max)
    (hco : c.val < o.val)
    (hinv : bwdIInnerInv dy w i o r k c acc ok1) :
    WP.spec (learn_port.dense.linear_bwd_input_acc_loop0_loop0_loop0.body dy w i o r k ok1 c acc)
      (fun cf => ∃ c' acc', cf = ControlFlow.cont (true, c', acc') ∧
        c'.val = c.val + 1 ∧ bwdIInnerInv dy w i o r k c' acc' true) := by
  obtain ⟨hok, _hcle, hacc⟩ := hinv
  have hdymax : dy.val.length ≤ Usize.max := Slice.property dy
  have hwmax : w.val.length ≤ Usize.max := Slice.property w
  have hrc : r.val * o.val + c.val < Usize.max := by
    calc r.val * o.val + c.val < r.val * o.val + o.val := by omega
      _ ≤ dy.val.length := hdy
      _ ≤ Usize.max := hdymax
  have hck : c.val * i.val + k.val < Usize.max := by
    calc c.val * i.val + k.val < c.val * i.val + i.val := by omega
      _ = (c.val + 1) * i.val := (Nat.succ_mul c.val i.val).symm
      _ ≤ o.val * i.val := Nat.mul_le_mul_right i.val (Nat.succ_le_of_lt hco)
      _ ≤ w.val.length := hwo
      _ ≤ Usize.max := hwmax
  have hro_max : r.val * o.val ≤ Usize.max := by omega
  have hci_max : c.val * i.val ≤ Usize.max := by omega
  obtain ⟨ro, hro, hrov⟩ := WP.spec_imp_exists (Usize.mul_spec (x := r) (y := o) hro_max)
  obtain ⟨c2v, hc2, hc2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := ro) (y := c) (by rw [hrov]; omega))
  obtain ⟨ci, hci, hciv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := c) (y := i) hci_max)
  obtain ⟨k6v, hk6, hk6v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := ci) (y := k) (by rw [hciv]; omega))
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := c) (y := 1#usize)
      (by show c.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          have := dense_usize_le_max o
          omega))
  have hc1_val : c1.val = c.val + 1 := by rw [hc1v]; rfl
  have hidx2 : c2v.val = r.val * o.val + c.val := by rw [hc2v, hrov]
  have hidx6 : k6v.val = c.val * i.val + k.val := by rw [hk6v, hciv]
  have hc2lt : c2v.val < dy.val.length := by rw [hidx2]; omega
  have hk6lt : k6v.val < w.val.length := by
    rw [hidx6]
    calc c.val * i.val + k.val < c.val * i.val + i.val := by omega
      _ = (c.val + 1) * i.val := (Nat.succ_mul c.val i.val).symm
      _ ≤ o.val * i.val := Nat.mul_le_mul_right i.val (Nat.succ_le_of_lt hco)
      _ ≤ w.val.length := hwo
  have hmulb : I128.min ≤
        (core.convert.num.FromI128I64.from (dy.val.getD c2v.val default)).val *
        (core.convert.num.FromI128I64.from (w.val.getD k6v.val default)).val ∧
      (core.convert.num.FromI128I64.from (dy.val.getD c2v.val default)).val *
        (core.convert.num.FromI128I64.from (w.val.getD k6v.val default)).val ≤ I128.max := by
    rw [dense_fromI64_val, dense_fromI64_val]
    exact dense_i64_mul_fits _ _
  obtain ⟨p, hmul, hpv⟩ :=
    dense_checked_mul_some (a := core.convert.num.FromI128I64.from (dy.val.getD c2v.val default))
      (b := core.convert.num.FromI128I64.from (w.val.getD k6v.val default)) hmulb
  have hLg : (dy.val.getD c2v.val default).val = (qvals dy).getD (r.val * o.val + c.val) 0 := by
    rw [← hidx2]; exact (qvals_getD dy hc2lt).symm
  have hRg : (w.val.getD k6v.val default).val = (qvals w).getD (c.val * i.val + k.val) 0 := by
    rw [← hidx6]; exact (qvals_getD w hk6lt).symm
  have hpval : p.val =
      (qvals dy).getD (r.val * o.val + c.val) 0 * (qvals w).getD (c.val * i.val + k.val) 0 := by
    rw [hpv, dense_fromI64_val, dense_fromI64_val, hLg, hRg]
  have hsum_in : I128.min ≤ acc.val + p.val ∧ acc.val + p.val ≤ I128.max := by
    have he : acc.val + p.val = bwdIAcc dy w i o r.val k.val (c.val + 1) := by
      rw [hacc, hpval]; exact (bwdIAcc_succ dy w i o r.val k.val c.val).symm
    rw [he]; exact hno (c.val + 1) (by omega)
  obtain ⟨s, hadd, hsv⟩ := dense_checked_add_some hsum_in
  have hsval : s.val = bwdIAcc dy w i o r.val k.val c1.val := by
    rw [hsv, hacc, hpval, hc1_val]
    exact (bwdIAcc_succ dy w i o r.val k.val c.val).symm
  have hcProp : c < o := (UScalar.lt_equiv c o).2 hco
  have hexec := linear_bwd_input_acc_loop0_loop0_loop0_body_exec dy w i o r k c acc ro c2v ci k6v
    p s c1 hcProp hro hc2 hc2lt hci hk6 hk6lt hmul hadd hc1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (true, c1, s), hexec, c1, s, rfl, hc1_val, ?_⟩
  exact ⟨rfl, by omega, hsval⟩

/-- Full specification of the `linear_bwd_input_acc` inner loop. -/
theorem linear_bwd_input_acc_loop0_loop0_loop0_spec
    (dy w : Slice I64) (i o r k : Usize) (c : Usize) (acc : I128)
    (hdy : r.val * o.val + o.val ≤ dy.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hki : k.val < i.val)
    (hno : ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r.val k.val t ∧ bwdIAcc dy w i o r.val k.val t ≤ I128.max)
    (hinv0 : bwdIInnerInv dy w i o r k c acc true) :
    WP.spec (learn_port.dense.linear_bwd_input_acc_loop0_loop0_loop0 dy w i o r true k c acc)
      (fun (ok', acc') => ok' = true ∧ acc'.val = bwdIAcc dy w i o r.val k.val o.val) := by
  unfold learn_port.dense.linear_bwd_input_acc_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Usize × I128 => o.val - s.2.1.val)
    (inv := fun s => bwdIInnerInv dy w i o r k s.2.1 s.2.2 s.1)
    (post := fun b : Bool × I128 => b.1 = true ∧ b.2.val = bwdIAcc dy w i o r.val k.val o.val)
  · rintro ⟨ok', c', acc'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hc'le, hacc'⟩ := hinv
    rw [hok']
    by_cases hco : c'.val < o.val
    · obtain ⟨cf, hstep, c'', acc'', hcf, hc''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_bwd_input_acc_loop0_loop0_loop0_body_step dy w i o r k c' acc' true hdy hwo hki
          hno hco ⟨rfl, hc'le, hacc'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (c' < o) := fun hc => hco ((UScalar.lt_equiv c' o).1 hc)
      have hbody :
          learn_port.dense.linear_bwd_input_acc_loop0_loop0_loop0.body dy w i o r k true c' acc' =
            ok (ControlFlow.done (true, acc')) := by
        rw [learn_port.dense.linear_bwd_input_acc_loop0_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      have hc'_eq : c'.val = o.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hacc', hc'_eq]
  · exact hinv0

/-- The `linear_bwd_input_acc` middle-loop invariant (`..._loop0_loop0`). -/
def bwdIMidInv (dy w : Slice I64) (i o r : Usize) (out : alloc.vec.Vec I128) (k : Usize)
    (ok1 : Bool) : Prop :=
  ok1 = true ∧ k.val ≤ i.val ∧
    ovals out = rowsOf i.val (bwdIElem dy w i o) r.val ++
      pushN [] (fun k' => bwdIAcc dy w i o r.val k' o.val) k.val

/-- The extracted `linear_bwd_input_acc` middle body on the succeeding path. -/
theorem linear_bwd_input_acc_loop0_loop0_body_exec
    (dy w : Slice I64) (i o r : Usize) (out : alloc.vec.Vec I128) (k : Usize)
    (acc : I128) (out1 : alloc.vec.Vec I128) (k1 : Usize)
    (hk : k < i)
    (hinner : learn_port.dense.linear_bwd_input_acc_loop0_loop0_loop0 dy w i o r true k 0#usize
      0#i128 = ok (true, acc))
    (hp : alloc.vec.Vec.push out acc = ok out1)
    (hk1 : k + 1#usize = ok k1) :
    learn_port.dense.linear_bwd_input_acc_loop0_loop0.body dy w i o r out true k =
      ok (ControlFlow.cont (out1, true, k1)) := by
  rw [learn_port.dense.linear_bwd_input_acc_loop0_loop0.body]
  rw [if_pos hk, if_pos rfl]
  simp only [hinner]
  simp [hp, hk1]

/-- One iteration of the `linear_bwd_input_acc` middle loop advances the model. -/
theorem linear_bwd_input_acc_loop0_loop0_body_step
    (dy w : Slice I64) (i o r : Usize) (out : alloc.vec.Vec I128) (k : Usize) (ok1 : Bool)
    (hdy : r.val * o.val + o.val ≤ dy.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hcap : r.val * i.val + i.val ≤ Usize.max)
    (hno : ∀ k' < i.val, ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r.val k' t ∧ bwdIAcc dy w i o r.val k' t ≤ I128.max)
    (hki : k.val < i.val)
    (hinv : bwdIMidInv dy w i o r out k ok1) :
    WP.spec (learn_port.dense.linear_bwd_input_acc_loop0_loop0.body dy w i o r out ok1 k)
      (fun cf => ∃ out' k', cf = ControlFlow.cont (out', true, k') ∧ k'.val = k.val + 1 ∧
        bwdIMidInv dy w i o r out' k' true) := by
  obtain ⟨hok, _hkle, hout⟩ := hinv
  have hinner0 : bwdIInnerInv dy w i o r k 0#usize 0#i128 true := by
    refine ⟨rfl, by show (0#usize).val ≤ o.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show (0#i128).val = bwdIAcc dy w i o r.val k.val (0#usize).val
    rw [show (0#i128).val = 0 from rfl, show (0#usize).val = 0 from rfl]
    rfl
  have hinner := linear_bwd_input_acc_loop0_loop0_loop0_spec dy w i o r k 0#usize 0#i128
    hdy hwo hki (hno k.val hki) hinner0
  obtain ⟨⟨ok2, acc⟩, hinner_exec, hok2, haccv⟩ := (WP.spec_equiv_exists _ _).mp hinner
  subst hok2
  have hlenout : out.val.length < Usize.max := by
    have h1 : (ovals out).length = out.val.length := ovals_length out
    rw [← h1, hout, List.length_append, rowsOf_length, pushN_length, List.length_nil]
    omega
  obtain ⟨out1, hp, hout1⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out acc hlenout)
  obtain ⟨k1, hk1, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize)
      (by show k.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          omega))
  have hk1_val : k1.val = k.val + 1 := by rw [hk1v]; rfl
  have hkProp : k < i := (UScalar.lt_equiv k i).2 hki
  have hexec := linear_bwd_input_acc_loop0_loop0_body_exec dy w i o r out k acc out1 k1 hkProp
    hinner_exec hp hk1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out1, true, k1), hexec, out1, k1, rfl, hk1_val, ?_⟩
  refine ⟨rfl, by omega, ?_⟩
  have hval1 : ovals out1 = ovals out ++ [acc.val] := by simp [ovals, hout1]
  rw [hval1, hout, haccv, hk1_val, pushN, List.append_assoc]

/-- Full specification of the `linear_bwd_input_acc` middle loop. -/
theorem linear_bwd_input_acc_loop0_loop0_spec
    (dy w : Slice I64) (i o r : Usize) (out : alloc.vec.Vec I128) (k : Usize)
    (hdy : r.val * o.val + o.val ≤ dy.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hcap : r.val * i.val + i.val ≤ Usize.max)
    (hno : ∀ k' < i.val, ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r.val k' t ∧ bwdIAcc dy w i o r.val k' t ≤ I128.max)
    (hinv0 : bwdIMidInv dy w i o r out k true) :
    WP.spec (learn_port.dense.linear_bwd_input_acc_loop0_loop0 dy w i o out r true k)
      (fun (out', ok') => ok' = true ∧
        ovals out' = rowsOf i.val (bwdIElem dy w i o) r.val ++
          pushN [] (fun k' => bwdIAcc dy w i o r.val k' o.val) i.val) := by
  unfold learn_port.dense.linear_bwd_input_acc_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I128 × Bool × Usize => i.val - s.2.2.val)
    (inv := fun s => bwdIMidInv dy w i o r s.1 s.2.2 s.2.1)
    (post := fun b : alloc.vec.Vec I128 × Bool => b.2 = true ∧
      ovals b.1 = rowsOf i.val (bwdIElem dy w i o) r.val ++
        pushN [] (fun k' => bwdIAcc dy w i o r.val k' o.val) i.val)
  · rintro ⟨out', ok', k'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hk'le, hout'⟩ := hinv
    rw [hok']
    by_cases hki : k'.val < i.val
    · obtain ⟨cf, hstep, out'', k'', hcf, hk''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_bwd_input_acc_loop0_loop0_body_step dy w i o r out' k' true hdy hwo hcap hno hki
          ⟨rfl, hk'le, hout'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (k' < i) := fun hc => hki ((UScalar.lt_equiv k' i).1 hc)
      have hbody : learn_port.dense.linear_bwd_input_acc_loop0_loop0.body dy w i o r out' true k' =
          ok (ControlFlow.done (out', true)) := by
        rw [learn_port.dense.linear_bwd_input_acc_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      have hk'_eq : k'.val = i.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hout', hk'_eq]
  · exact hinv0

/-! ### The `linear_bwd_input_acc` outer loop (`r < n`) -/

/-- The `linear_bwd_input_acc` outer-loop invariant (`..._loop0`). -/
def bwdIOuterInv (dy w : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128)
    (r : Usize) (ok1 : Bool) : Prop :=
  ok1 = true ∧ r.val ≤ n.val ∧ ovals out = rowsOf i.val (bwdIElem dy w i o) r.val

/-- The extracted `linear_bwd_input_acc` outer body on the succeeding path. -/
theorem linear_bwd_input_acc_loop0_body_exec
    (dy w : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (r : Usize)
    (out1 : alloc.vec.Vec I128) (r1 : Usize)
    (hr : r < n)
    (hmid : learn_port.dense.linear_bwd_input_acc_loop0_loop0 dy w i o out r true 0#usize
      = ok (out1, true))
    (hr1 : r + 1#usize = ok r1) :
    learn_port.dense.linear_bwd_input_acc_loop0.body dy w n i o out r true =
      ok (ControlFlow.cont (out1, r1, true)) := by
  rw [learn_port.dense.linear_bwd_input_acc_loop0.body]
  rw [if_pos hr, if_pos rfl]
  simp only [hmid]
  simp [hr1]

/-- One iteration of the `linear_bwd_input_acc` outer loop advances the model. -/
theorem linear_bwd_input_acc_loop0_body_step
    (dy w : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (r : Usize) (ok1 : Bool)
    (hdy : n.val * o.val ≤ dy.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hprod : n.val * i.val ≤ Usize.max)
    (hno : ∀ r' < n.val, ∀ k' < i.val, ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r' k' t ∧ bwdIAcc dy w i o r' k' t ≤ I128.max)
    (hrn : r.val < n.val)
    (hinv : bwdIOuterInv dy w n i o out r ok1) :
    WP.spec (learn_port.dense.linear_bwd_input_acc_loop0.body dy w n i o out r ok1)
      (fun cf => ∃ out' r', cf = ControlFlow.cont (out', r', true) ∧ r'.val = r.val + 1 ∧
        bwdIOuterInv dy w n i o out' r' true) := by
  obtain ⟨hok, _hrle, hout⟩ := hinv
  have hdyr : r.val * o.val + o.val ≤ dy.val.length := by
    calc r.val * o.val + o.val = (r.val + 1) * o.val := (Nat.succ_mul r.val o.val).symm
      _ ≤ n.val * o.val := Nat.mul_le_mul_right o.val (Nat.succ_le_of_lt hrn)
      _ ≤ dy.val.length := hdy
  have hcapr : r.val * i.val + i.val ≤ Usize.max := by
    calc r.val * i.val + i.val = (r.val + 1) * i.val := (Nat.succ_mul r.val i.val).symm
      _ ≤ n.val * i.val := Nat.mul_le_mul_right i.val (Nat.succ_le_of_lt hrn)
      _ ≤ Usize.max := hprod
  have hno_r : ∀ k' < i.val, ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r.val k' t ∧ bwdIAcc dy w i o r.val k' t ≤ I128.max :=
    fun k' hk' => hno r.val hrn k' hk'
  have hmid0 : bwdIMidInv dy w i o r out 0#usize true := by
    refine ⟨rfl, by show (0#usize).val ≤ i.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show ovals out = rowsOf i.val (bwdIElem dy w i o) r.val ++
      pushN [] (fun k' => bwdIAcc dy w i o r.val k' o.val) (0#usize).val
    rw [show (0#usize).val = 0 from rfl,
      show pushN ([] : List ℤ) (fun k' => bwdIAcc dy w i o r.val k' o.val) 0 = [] from rfl,
      List.append_nil]
    exact hout
  have hmid := linear_bwd_input_acc_loop0_loop0_spec dy w i o r out 0#usize hdyr hwo hcapr hno_r
    hmid0
  obtain ⟨⟨out1, ok2⟩, hmid_exec, hok2, hout1⟩ := (WP.spec_equiv_exists _ _).mp hmid
  subst hok2
  obtain ⟨r1, hr1, hr1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := r) (y := 1#usize)
      (by show r.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          have := dense_usize_le_max n
          omega))
  have hr1_val : r1.val = r.val + 1 := by rw [hr1v]; rfl
  have hrProp : r < n := (UScalar.lt_equiv r n).2 hrn
  have hexec := linear_bwd_input_acc_loop0_body_exec dy w n i o out r out1 r1 hrProp hmid_exec hr1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out1, r1, true), hexec, out1, r1, rfl, hr1_val, ?_⟩
  refine ⟨rfl, by omega, ?_⟩
  rw [hout1, hr1_val]
  rfl

/-- Full specification of the `linear_bwd_input_acc` outer loop. -/
theorem linear_bwd_input_acc_loop0_spec
    (dy w : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (r : Usize)
    (hdy : n.val * o.val ≤ dy.val.length)
    (hwo : o.val * i.val ≤ w.val.length)
    (hprod : n.val * i.val ≤ Usize.max)
    (hno : ∀ r' < n.val, ∀ k' < i.val, ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r' k' t ∧ bwdIAcc dy w i o r' k' t ≤ I128.max)
    (hinv0 : bwdIOuterInv dy w n i o out r true) :
    WP.spec (learn_port.dense.linear_bwd_input_acc_loop0 dy w n i o out r true)
      (fun (out', ok') => ok' = true ∧ ovals out' = rowsOf i.val (bwdIElem dy w i o) n.val) := by
  unfold learn_port.dense.linear_bwd_input_acc_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I128 × Usize × Bool => n.val - s.2.1.val)
    (inv := fun s => bwdIOuterInv dy w n i o s.1 s.2.1 s.2.2)
    (post := fun b : alloc.vec.Vec I128 × Bool => b.2 = true ∧
      ovals b.1 = rowsOf i.val (bwdIElem dy w i o) n.val)
  · rintro ⟨out', r', ok'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hr'le, hout'⟩ := hinv
    rw [hok']
    by_cases hrn : r'.val < n.val
    · obtain ⟨cf, hstep, out'', r'', hcf, hr''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_bwd_input_acc_loop0_body_step dy w n i o out' r' true hdy hwo hprod hno hrn
          ⟨rfl, hr'le, hout'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (r' < n) := fun hc => hrn ((UScalar.lt_equiv r' n).1 hc)
      have hbody : learn_port.dense.linear_bwd_input_acc_loop0.body dy w n i o out' r' true =
          ok (ControlFlow.done (out', true)) := by
        rw [learn_port.dense.linear_bwd_input_acc_loop0.body, if_neg hnot]
      simp only [hbody]
      have hr'_eq : r'.val = n.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hout', hr'_eq]
  · exact hinv0

/-- `linear_bwd_input_acc` computes the unrounded `i128` input adjoint `dy·wᵀ`, row-major `n × i`. -/
theorem linear_bwd_input_acc_spec
    (dy w : Slice I64) (n i o : Usize)
    (hdy : dy.val.length = n.val * o.val)
    (hwo : w.val.length = o.val * i.val)
    (hprod : n.val * i.val ≤ Usize.max)
    (hno : ∀ r' < n.val, ∀ k' < i.val, ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r' k' t ∧ bwdIAcc dy w i o r' k' t ≤ I128.max) :
    ∃ out, learn_port.dense.linear_bwd_input_acc dy w n i o = ok (core.result.Result.Ok out) ∧
      ovals out = rowsOf i.val (bwdIElem dy w i o) n.val := by
  have hdy' : n.val * o.val ≤ dy.val.length := by rw [hdy]
  have hwo' : o.val * i.val ≤ w.val.length := by rw [hwo]
  obtain ⟨t, ht, _htv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := n) (y := i) hprod)
  have hinv0 : bwdIOuterInv dy w n i o (alloc.vec.Vec.with_capacity I128 t) 0#usize true := by
    refine ⟨rfl, by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show ovals (alloc.vec.Vec.with_capacity I128 t) =
      rowsOf i.val (bwdIElem dy w i o) (0#usize).val
    rw [show (0#usize).val = 0 from rfl, ovals_with_capacity]
    rfl
  have hspec := linear_bwd_input_acc_loop0_spec dy w n i o (alloc.vec.Vec.with_capacity I128 t)
    0#usize hdy' hwo' hprod hno hinv0
  obtain ⟨⟨out1, ok1⟩, hloop, hok1, hout1⟩ := (WP.spec_equiv_exists _ _).mp hspec
  subst hok1
  refine ⟨out1, ?_, hout1⟩
  unfold learn_port.dense.linear_bwd_input_acc
  dsimp only
  rw [ht]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  simp

/-- `linear_bwd_input` rounds the exact `i128` input adjoint to Q32.32. -/
theorem linear_bwd_input_ok (dy w : Slice I64) (n i o : Usize)
    (hn : n ≠ 0#usize) (hi : i ≠ 0#usize) (ho : o ≠ 0#usize)
    (hdy : dy.val.length = n.val * o.val)
    (hw : w.val.length = o.val * i.val)
    (hno_ : n.val * o.val ≤ Usize.max)
    (hoi : o.val * i.val ≤ Usize.max)
    (hprod : n.val * i.val ≤ Usize.max)
    (hbound : ∀ r' < n.val, ∀ k' < i.val, ∀ t ≤ o.val,
      I128.min ≤ bwdIAcc dy w i o r' k' t ∧ bwdIAcc dy w i o r' k' t ≤ I128.max)
    (hround : ∀ t < n.val * i.val,
      I128.min ≤ (rowsOf i.val (bwdIElem dy w i o) n.val).getD t 0 + 2 ^ 31 ∧
        (rowsOf i.val (bwdIElem dy w i o) n.val).getD t 0 + 2 ^ 31 ≤ I128.max)
    (hnar : ∀ t < n.val * i.val,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((rowsOf i.val (bwdIElem dy w i o) n.val).getD t 0) ∧
        rnd32 ((rowsOf i.val (bwdIElem dy w i o) n.val).getD t 0) ≤ (2 : ℤ) ^ 63 - 1) :
    ∃ out, learn_port.dense.linear_bwd_input dy w n i o = ok (core.result.Result.Ok out) ∧
      vvals out = rndAll (rowsOf i.val (bwdIElem dy w i o) n.val) (n.val * i.val) := by
  have hexp1 := dense_expect_len_ok n o (Slice.len dy) hn ho hno_ (by rw [Slice.len_val]; exact hdy)
  have hexp2 := dense_expect_len_ok o i (Slice.len w) ho hi hoi (by rw [Slice.len_val]; exact hw)
  obtain ⟨acc, hacc, haccv⟩ := linear_bwd_input_acc_spec dy w n i o hdy hw hprod hbound
  have hlenacc : acc.val.length = n.val * i.val := by
    rw [← ovals_length, haccv, rowsOf_length]
  have hlen_deref : (alloc.vec.Vec.deref acc).val.length = n.val * i.val := by
    rw [deref_length, hlenacc]
  have hround' : ∀ t < (alloc.vec.Vec.deref acc).val.length,
      I128.min ≤ (ivals (alloc.vec.Vec.deref acc)).getD t 0 + 2 ^ 31 ∧
      (ivals (alloc.vec.Vec.deref acc)).getD t 0 + 2 ^ 31 ≤ I128.max := by
    intro t ht
    rw [ivals_deref, haccv]
    exact hround t (by rw [hlen_deref] at ht; exact ht)
  have hnar' : ∀ t < (alloc.vec.Vec.deref acc).val.length,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((ivals (alloc.vec.Vec.deref acc)).getD t 0) ∧
      rnd32 ((ivals (alloc.vec.Vec.deref acc)).getD t 0) ≤ (2 : ℤ) ^ 63 - 1 := by
    intro t ht
    rw [ivals_deref, haccv]
    exact hnar t (by rw [hlen_deref] at ht; exact ht)
  obtain ⟨out, hra, hrav⟩ := round_all_spec (alloc.vec.Vec.deref acc) hround' hnar'
  have hrav' : vvals out = rndAll (rowsOf i.val (bwdIElem dy w i o) n.val) (n.val * i.val) := by
    rw [hrav, ivals_deref, haccv, deref_length, hlenacc]
  refine ⟨out, ?_, hrav'⟩
  unfold learn_port.dense.linear_bwd_input
  simp only [hexp1, hexp2, hacc, hra, dense_branch_ok, bind_tc_ok]

/-! ## 7. The `linear_bwd_weight_acc` loops -/

/-- The `linear_bwd_weight_acc` inner accumulator `Σ_{t<r} dy[t·o + c] · x[t·i + k]`. -/
def bwdWAcc (dy x : Slice I64) (i o : Usize) (c k : ℕ) : ℕ → ℤ :=
  bwdWK (qvals dy) (qvals x) i.val o.val c k

theorem bwdWAcc_succ (dy x : Slice I64) (i o : Usize) (c k r : ℕ) :
    bwdWAcc dy x i o c k (r + 1) =
      bwdWAcc dy x i o c k r + (qvals dy).getD (r * o.val + c) 0 *
        (qvals x).getD (r * i.val + k) 0 := rfl

/-- The `linear_bwd_weight_acc` output row-major element model. -/
def bwdWElem (dy x : Slice I64) (n i o : Usize) : ℕ → ℕ → ℤ :=
  fun c k => bwdWAcc dy x i o c k n.val

/-- The `linear_bwd_weight_acc` inner-loop invariant (`..._loop0_loop0_loop0`). -/
def bwdWInnerInv (dy x : Slice I64) (n i o c k : Usize) (r : Usize) (acc : I128)
    (ok1 : Bool) : Prop :=
  ok1 = true ∧ r.val ≤ n.val ∧ acc.val = bwdWAcc dy x i o c.val k.val r.val

/-- The extracted `linear_bwd_weight_acc` inner body on the succeeding path. -/
theorem linear_bwd_weight_acc_loop0_loop0_loop0_body_exec
    (dy x : Slice I64) (n i o c k : Usize) (r : Usize) (acc : I128)
    (ro d2v ri x6v : Usize) (p s : I128) (r1 : Usize)
    (hr : r < n)
    (hro : r * o = ok ro)
    (hd2 : ro + c = ok d2v)
    (hd2lt : d2v.val < dy.val.length)
    (hri : r * i = ok ri)
    (hx6 : ri + k = ok x6v)
    (hx6lt : x6v.val < x.val.length)
    (hmul : I128.checked_mul (core.convert.num.FromI128I64.from (dy.val.getD d2v.val default))
              (core.convert.num.FromI128I64.from (x.val.getD x6v.val default)) = some p)
    (hadd : I128.checked_add acc p = some s)
    (hr1 : r + 1#usize = ok r1) :
    learn_port.dense.linear_bwd_weight_acc_loop0_loop0_loop0.body dy x n i o c k true r acc =
      ok (ControlFlow.cont (true, r1, s)) := by
  have hd := dense_slice_index_getD dy d2v hd2lt
  have hx := dense_slice_index_getD x x6v hx6lt
  rw [learn_port.dense.linear_bwd_weight_acc_loop0_loop0_loop0.body]
  rw [if_pos hr, if_pos rfl]
  simp only [hro, hd2, hd, hri, hx6, hx, lift, hmul, hadd, hr1, bind_tc_ok]
  simp

/-- One iteration of the `linear_bwd_weight_acc` inner loop advances the model. -/
theorem linear_bwd_weight_acc_loop0_loop0_loop0_body_step
    (dy x : Slice I64) (n i o c k : Usize) (r : Usize) (acc : I128) (ok1 : Bool)
    (hdyn : n.val * o.val ≤ dy.val.length)
    (hxin : n.val * i.val ≤ x.val.length)
    (hco : c.val < o.val)
    (hki : k.val < i.val)
    (hno : ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c.val k.val t ∧ bwdWAcc dy x i o c.val k.val t ≤ I128.max)
    (hrn : r.val < n.val)
    (hinv : bwdWInnerInv dy x n i o c k r acc ok1) :
    WP.spec (learn_port.dense.linear_bwd_weight_acc_loop0_loop0_loop0.body dy x n i o c k ok1 r acc)
      (fun cf => ∃ r' acc', cf = ControlFlow.cont (true, r', acc') ∧
        r'.val = r.val + 1 ∧ bwdWInnerInv dy x n i o c k r' acc' true) := by
  obtain ⟨hok, _hrle, hacc⟩ := hinv
  have hd2b : r.val * o.val + c.val < dy.val.length := by
    calc r.val * o.val + c.val < r.val * o.val + o.val := by omega
      _ = (r.val + 1) * o.val := (Nat.succ_mul r.val o.val).symm
      _ ≤ n.val * o.val := Nat.mul_le_mul_right o.val (Nat.succ_le_of_lt hrn)
      _ ≤ dy.val.length := hdyn
  have hx6b : r.val * i.val + k.val < x.val.length := by
    calc r.val * i.val + k.val < r.val * i.val + i.val := by omega
      _ = (r.val + 1) * i.val := (Nat.succ_mul r.val i.val).symm
      _ ≤ n.val * i.val := Nat.mul_le_mul_right i.val (Nat.succ_le_of_lt hrn)
      _ ≤ x.val.length := hxin
  have hd2max : r.val * o.val + c.val ≤ Usize.max :=
    le_of_lt (lt_of_lt_of_le hd2b (Slice.property dy))
  have hx6max : r.val * i.val + k.val ≤ Usize.max :=
    le_of_lt (lt_of_lt_of_le hx6b (Slice.property x))
  have hro_max : r.val * o.val ≤ Usize.max := by omega
  have hri_max : r.val * i.val ≤ Usize.max := by omega
  obtain ⟨ro, hro, hrov⟩ := WP.spec_imp_exists (Usize.mul_spec (x := r) (y := o) hro_max)
  obtain ⟨d2v, hd2, hd2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := ro) (y := c) (by rw [hrov]; exact hd2max))
  obtain ⟨ri, hri, hriv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := r) (y := i) hri_max)
  obtain ⟨x6v, hx6, hx6v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := ri) (y := k) (by rw [hriv]; exact hx6max))
  obtain ⟨r1, hr1, hr1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := r) (y := 1#usize)
      (by show r.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          have := dense_usize_le_max n
          omega))
  have hr1_val : r1.val = r.val + 1 := by rw [hr1v]; rfl
  have hidx2 : d2v.val = r.val * o.val + c.val := by rw [hd2v, hrov]
  have hidx6 : x6v.val = r.val * i.val + k.val := by rw [hx6v, hriv]
  have hd2lt : d2v.val < dy.val.length := by rw [hidx2]; exact hd2b
  have hx6lt : x6v.val < x.val.length := by rw [hidx6]; exact hx6b
  have hmulb : I128.min ≤
        (core.convert.num.FromI128I64.from (dy.val.getD d2v.val default)).val *
        (core.convert.num.FromI128I64.from (x.val.getD x6v.val default)).val ∧
      (core.convert.num.FromI128I64.from (dy.val.getD d2v.val default)).val *
        (core.convert.num.FromI128I64.from (x.val.getD x6v.val default)).val ≤ I128.max := by
    rw [dense_fromI64_val, dense_fromI64_val]
    exact dense_i64_mul_fits _ _
  obtain ⟨p, hmul, hpv⟩ :=
    dense_checked_mul_some (a := core.convert.num.FromI128I64.from (dy.val.getD d2v.val default))
      (b := core.convert.num.FromI128I64.from (x.val.getD x6v.val default)) hmulb
  have hLg : (dy.val.getD d2v.val default).val = (qvals dy).getD (r.val * o.val + c.val) 0 := by
    rw [← hidx2]; exact (qvals_getD dy hd2lt).symm
  have hRg : (x.val.getD x6v.val default).val = (qvals x).getD (r.val * i.val + k.val) 0 := by
    rw [← hidx6]; exact (qvals_getD x hx6lt).symm
  have hpval : p.val =
      (qvals dy).getD (r.val * o.val + c.val) 0 * (qvals x).getD (r.val * i.val + k.val) 0 := by
    rw [hpv, dense_fromI64_val, dense_fromI64_val, hLg, hRg]
  have hsum_in : I128.min ≤ acc.val + p.val ∧ acc.val + p.val ≤ I128.max := by
    have he : acc.val + p.val = bwdWAcc dy x i o c.val k.val (r.val + 1) := by
      rw [hacc, hpval]; exact (bwdWAcc_succ dy x i o c.val k.val r.val).symm
    rw [he]; exact hno (r.val + 1) (by omega)
  obtain ⟨s, hadd, hsv⟩ := dense_checked_add_some hsum_in
  have hsval : s.val = bwdWAcc dy x i o c.val k.val r1.val := by
    rw [hsv, hacc, hpval, hr1_val]
    exact (bwdWAcc_succ dy x i o c.val k.val r.val).symm
  have hrProp : r < n := (UScalar.lt_equiv r n).2 hrn
  have hexec := linear_bwd_weight_acc_loop0_loop0_loop0_body_exec dy x n i o c k r acc ro d2v ri
    x6v p s r1 hrProp hro hd2 hd2lt hri hx6 hx6lt hmul hadd hr1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (true, r1, s), hexec, r1, s, rfl, hr1_val, ?_⟩
  exact ⟨rfl, by omega, hsval⟩

/-- Full specification of the `linear_bwd_weight_acc` inner loop. -/
theorem linear_bwd_weight_acc_loop0_loop0_loop0_spec
    (dy x : Slice I64) (n i o c k : Usize) (r : Usize) (acc : I128)
    (hdyn : n.val * o.val ≤ dy.val.length)
    (hxin : n.val * i.val ≤ x.val.length)
    (hco : c.val < o.val)
    (hki : k.val < i.val)
    (hno : ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c.val k.val t ∧ bwdWAcc dy x i o c.val k.val t ≤ I128.max)
    (hinv0 : bwdWInnerInv dy x n i o c k r acc true) :
    WP.spec (learn_port.dense.linear_bwd_weight_acc_loop0_loop0_loop0 dy x n i o c true k r acc)
      (fun (ok', acc') => ok' = true ∧ acc'.val = bwdWAcc dy x i o c.val k.val n.val) := by
  unfold learn_port.dense.linear_bwd_weight_acc_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Usize × I128 => n.val - s.2.1.val)
    (inv := fun s => bwdWInnerInv dy x n i o c k s.2.1 s.2.2 s.1)
    (post := fun b : Bool × I128 => b.1 = true ∧ b.2.val = bwdWAcc dy x i o c.val k.val n.val)
  · rintro ⟨ok', r', acc'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hr'le, hacc'⟩ := hinv
    rw [hok']
    by_cases hrn : r'.val < n.val
    · obtain ⟨cf, hstep, r'', acc'', hcf, hr''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_bwd_weight_acc_loop0_loop0_loop0_body_step dy x n i o c k r' acc' true hdyn hxin
          hco hki hno hrn ⟨rfl, hr'le, hacc'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (r' < n) := fun hc => hrn ((UScalar.lt_equiv r' n).1 hc)
      have hbody :
          learn_port.dense.linear_bwd_weight_acc_loop0_loop0_loop0.body dy x n i o c k true r' acc' =
            ok (ControlFlow.done (true, acc')) := by
        rw [learn_port.dense.linear_bwd_weight_acc_loop0_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      have hr'_eq : r'.val = n.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hacc', hr'_eq]
  · exact hinv0

/-- The `linear_bwd_weight_acc` middle-loop invariant (`..._loop0_loop0`). -/
def bwdWMidInv (dy x : Slice I64) (n i o c : Usize) (out : alloc.vec.Vec I128) (k : Usize)
    (ok1 : Bool) : Prop :=
  ok1 = true ∧ k.val ≤ i.val ∧
    ovals out = rowsOf i.val (bwdWElem dy x n i o) c.val ++
      pushN [] (fun k' => bwdWAcc dy x i o c.val k' n.val) k.val

/-- The extracted `linear_bwd_weight_acc` middle body on the succeeding path. -/
theorem linear_bwd_weight_acc_loop0_loop0_body_exec
    (dy x : Slice I64) (n i o c : Usize) (out : alloc.vec.Vec I128) (k : Usize)
    (acc : I128) (out1 : alloc.vec.Vec I128) (k1 : Usize)
    (hk : k < i)
    (hinner : learn_port.dense.linear_bwd_weight_acc_loop0_loop0_loop0 dy x n i o c true k 0#usize
      0#i128 = ok (true, acc))
    (hp : alloc.vec.Vec.push out acc = ok out1)
    (hk1 : k + 1#usize = ok k1) :
    learn_port.dense.linear_bwd_weight_acc_loop0_loop0.body dy x n i o c out true k =
      ok (ControlFlow.cont (out1, true, k1)) := by
  rw [learn_port.dense.linear_bwd_weight_acc_loop0_loop0.body]
  rw [if_pos hk, if_pos rfl]
  simp only [hinner]
  simp [hp, hk1]

/-- One iteration of the `linear_bwd_weight_acc` middle loop advances the model. -/
theorem linear_bwd_weight_acc_loop0_loop0_body_step
    (dy x : Slice I64) (n i o c : Usize) (out : alloc.vec.Vec I128) (k : Usize) (ok1 : Bool)
    (hdyn : n.val * o.val ≤ dy.val.length)
    (hxin : n.val * i.val ≤ x.val.length)
    (hco : c.val < o.val)
    (hcap : c.val * i.val + i.val ≤ Usize.max)
    (hno : ∀ k' < i.val, ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c.val k' t ∧ bwdWAcc dy x i o c.val k' t ≤ I128.max)
    (hki : k.val < i.val)
    (hinv : bwdWMidInv dy x n i o c out k ok1) :
    WP.spec (learn_port.dense.linear_bwd_weight_acc_loop0_loop0.body dy x n i o c out ok1 k)
      (fun cf => ∃ out' k', cf = ControlFlow.cont (out', true, k') ∧ k'.val = k.val + 1 ∧
        bwdWMidInv dy x n i o c out' k' true) := by
  obtain ⟨hok, _hkle, hout⟩ := hinv
  have hinner0 : bwdWInnerInv dy x n i o c k 0#usize 0#i128 true := by
    refine ⟨rfl, by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show (0#i128).val = bwdWAcc dy x i o c.val k.val (0#usize).val
    rw [show (0#i128).val = 0 from rfl, show (0#usize).val = 0 from rfl]
    rfl
  have hinner := linear_bwd_weight_acc_loop0_loop0_loop0_spec dy x n i o c k 0#usize 0#i128
    hdyn hxin hco hki (hno k.val hki) hinner0
  obtain ⟨⟨ok2, acc⟩, hinner_exec, hok2, haccv⟩ := (WP.spec_equiv_exists _ _).mp hinner
  subst hok2
  have hlenout : out.val.length < Usize.max := by
    have h1 : (ovals out).length = out.val.length := ovals_length out
    rw [← h1, hout, List.length_append, rowsOf_length, pushN_length, List.length_nil]
    omega
  obtain ⟨out1, hp, hout1⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out acc hlenout)
  obtain ⟨k1, hk1, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize)
      (by show k.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          omega))
  have hk1_val : k1.val = k.val + 1 := by rw [hk1v]; rfl
  have hkProp : k < i := (UScalar.lt_equiv k i).2 hki
  have hexec := linear_bwd_weight_acc_loop0_loop0_body_exec dy x n i o c out k acc out1 k1 hkProp
    hinner_exec hp hk1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out1, true, k1), hexec, out1, k1, rfl, hk1_val, ?_⟩
  refine ⟨rfl, by omega, ?_⟩
  have hval1 : ovals out1 = ovals out ++ [acc.val] := by simp [ovals, hout1]
  rw [hval1, hout, haccv, hk1_val, pushN, List.append_assoc]

/-- Full specification of the `linear_bwd_weight_acc` middle loop. -/
theorem linear_bwd_weight_acc_loop0_loop0_spec
    (dy x : Slice I64) (n i o c : Usize) (out : alloc.vec.Vec I128) (k : Usize)
    (hdyn : n.val * o.val ≤ dy.val.length)
    (hxin : n.val * i.val ≤ x.val.length)
    (hco : c.val < o.val)
    (hcap : c.val * i.val + i.val ≤ Usize.max)
    (hno : ∀ k' < i.val, ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c.val k' t ∧ bwdWAcc dy x i o c.val k' t ≤ I128.max)
    (hinv0 : bwdWMidInv dy x n i o c out k true) :
    WP.spec (learn_port.dense.linear_bwd_weight_acc_loop0_loop0 dy x n i o out c true k)
      (fun (out', ok') => ok' = true ∧
        ovals out' = rowsOf i.val (bwdWElem dy x n i o) c.val ++
          pushN [] (fun k' => bwdWAcc dy x i o c.val k' n.val) i.val) := by
  unfold learn_port.dense.linear_bwd_weight_acc_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I128 × Bool × Usize => i.val - s.2.2.val)
    (inv := fun s => bwdWMidInv dy x n i o c s.1 s.2.2 s.2.1)
    (post := fun b : alloc.vec.Vec I128 × Bool => b.2 = true ∧
      ovals b.1 = rowsOf i.val (bwdWElem dy x n i o) c.val ++
        pushN [] (fun k' => bwdWAcc dy x i o c.val k' n.val) i.val)
  · rintro ⟨out', ok', k'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hk'le, hout'⟩ := hinv
    rw [hok']
    by_cases hki : k'.val < i.val
    · obtain ⟨cf, hstep, out'', k'', hcf, hk''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_bwd_weight_acc_loop0_loop0_body_step dy x n i o c out' k' true hdyn hxin hco hcap
          hno hki ⟨rfl, hk'le, hout'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (k' < i) := fun hc => hki ((UScalar.lt_equiv k' i).1 hc)
      have hbody : learn_port.dense.linear_bwd_weight_acc_loop0_loop0.body dy x n i o c out' true k' =
          ok (ControlFlow.done (out', true)) := by
        rw [learn_port.dense.linear_bwd_weight_acc_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      have hk'_eq : k'.val = i.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hout', hk'_eq]
  · exact hinv0

/-! ### The `linear_bwd_weight_acc` outer loop (`c < o`) -/

/-- The `linear_bwd_weight_acc` outer-loop invariant (`..._loop0`). -/
def bwdWOuterInv (dy x : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128)
    (c : Usize) (ok1 : Bool) : Prop :=
  ok1 = true ∧ c.val ≤ o.val ∧ ovals out = rowsOf i.val (bwdWElem dy x n i o) c.val

/-- The extracted `linear_bwd_weight_acc` outer body on the succeeding path. -/
theorem linear_bwd_weight_acc_loop0_body_exec
    (dy x : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (c : Usize)
    (out1 : alloc.vec.Vec I128) (c1 : Usize)
    (hc : c < o)
    (hmid : learn_port.dense.linear_bwd_weight_acc_loop0_loop0 dy x n i o out c true 0#usize
      = ok (out1, true))
    (hc1 : c + 1#usize = ok c1) :
    learn_port.dense.linear_bwd_weight_acc_loop0.body dy x n i o out c true =
      ok (ControlFlow.cont (out1, c1, true)) := by
  rw [learn_port.dense.linear_bwd_weight_acc_loop0.body]
  rw [if_pos hc, if_pos rfl]
  simp only [hmid]
  simp [hc1]

/-- One iteration of the `linear_bwd_weight_acc` outer loop advances the model. -/
theorem linear_bwd_weight_acc_loop0_body_step
    (dy x : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (c : Usize) (ok1 : Bool)
    (hdyn : n.val * o.val ≤ dy.val.length)
    (hxin : n.val * i.val ≤ x.val.length)
    (hprod : o.val * i.val ≤ Usize.max)
    (hno : ∀ c' < o.val, ∀ k' < i.val, ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c' k' t ∧ bwdWAcc dy x i o c' k' t ≤ I128.max)
    (hco : c.val < o.val)
    (hinv : bwdWOuterInv dy x n i o out c ok1) :
    WP.spec (learn_port.dense.linear_bwd_weight_acc_loop0.body dy x n i o out c ok1)
      (fun cf => ∃ out' c', cf = ControlFlow.cont (out', c', true) ∧ c'.val = c.val + 1 ∧
        bwdWOuterInv dy x n i o out' c' true) := by
  obtain ⟨hok, _hcle, hout⟩ := hinv
  have hcapc : c.val * i.val + i.val ≤ Usize.max := by
    calc c.val * i.val + i.val = (c.val + 1) * i.val := (Nat.succ_mul c.val i.val).symm
      _ ≤ o.val * i.val := Nat.mul_le_mul_right i.val (Nat.succ_le_of_lt hco)
      _ ≤ Usize.max := hprod
  have hno_c : ∀ k' < i.val, ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c.val k' t ∧ bwdWAcc dy x i o c.val k' t ≤ I128.max :=
    fun k' hk' => hno c.val hco k' hk'
  have hmid0 : bwdWMidInv dy x n i o c out 0#usize true := by
    refine ⟨rfl, by show (0#usize).val ≤ i.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show ovals out = rowsOf i.val (bwdWElem dy x n i o) c.val ++
      pushN [] (fun k' => bwdWAcc dy x i o c.val k' n.val) (0#usize).val
    rw [show (0#usize).val = 0 from rfl,
      show pushN ([] : List ℤ) (fun k' => bwdWAcc dy x i o c.val k' n.val) 0 = [] from rfl,
      List.append_nil]
    exact hout
  have hmid := linear_bwd_weight_acc_loop0_loop0_spec dy x n i o c out 0#usize hdyn hxin hco
    hcapc hno_c hmid0
  obtain ⟨⟨out1, ok2⟩, hmid_exec, hok2, hout1⟩ := (WP.spec_equiv_exists _ _).mp hmid
  subst hok2
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := c) (y := 1#usize)
      (by show c.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          have := dense_usize_le_max o
          omega))
  have hc1_val : c1.val = c.val + 1 := by rw [hc1v]; rfl
  have hcProp : c < o := (UScalar.lt_equiv c o).2 hco
  have hexec := linear_bwd_weight_acc_loop0_body_exec dy x n i o out c out1 c1 hcProp hmid_exec hc1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out1, c1, true), hexec, out1, c1, rfl, hc1_val, ?_⟩
  refine ⟨rfl, by omega, ?_⟩
  rw [hout1, hc1_val]
  rfl

/-- Full specification of the `linear_bwd_weight_acc` outer loop. -/
theorem linear_bwd_weight_acc_loop0_spec
    (dy x : Slice I64) (n i o : Usize) (out : alloc.vec.Vec I128) (c : Usize)
    (hdyn : n.val * o.val ≤ dy.val.length)
    (hxin : n.val * i.val ≤ x.val.length)
    (hprod : o.val * i.val ≤ Usize.max)
    (hno : ∀ c' < o.val, ∀ k' < i.val, ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c' k' t ∧ bwdWAcc dy x i o c' k' t ≤ I128.max)
    (hinv0 : bwdWOuterInv dy x n i o out c true) :
    WP.spec (learn_port.dense.linear_bwd_weight_acc_loop0 dy x n i o out c true)
      (fun (out', ok') => ok' = true ∧ ovals out' = rowsOf i.val (bwdWElem dy x n i o) o.val) := by
  unfold learn_port.dense.linear_bwd_weight_acc_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I128 × Usize × Bool => o.val - s.2.1.val)
    (inv := fun s => bwdWOuterInv dy x n i o s.1 s.2.1 s.2.2)
    (post := fun b : alloc.vec.Vec I128 × Bool => b.2 = true ∧
      ovals b.1 = rowsOf i.val (bwdWElem dy x n i o) o.val)
  · rintro ⟨out', c', ok'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hc'le, hout'⟩ := hinv
    rw [hok']
    by_cases hco : c'.val < o.val
    · obtain ⟨cf, hstep, out'', c'', hcf, hc''_val, hinv''⟩ := WP.spec_imp_exists
        (linear_bwd_weight_acc_loop0_body_step dy x n i o out' c' true hdyn hxin hprod hno hco
          ⟨rfl, hc'le, hout'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (c' < o) := fun hc => hco ((UScalar.lt_equiv c' o).1 hc)
      have hbody : learn_port.dense.linear_bwd_weight_acc_loop0.body dy x n i o out' c' true =
          ok (ControlFlow.done (out', true)) := by
        rw [learn_port.dense.linear_bwd_weight_acc_loop0.body, if_neg hnot]
      simp only [hbody]
      have hc'_eq : c'.val = o.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hout', hc'_eq]
  · exact hinv0

/-- `linear_bwd_weight_acc` computes the unrounded `i128` weight adjoint `dyᵀ·x`, row-major `o × i`. -/
theorem linear_bwd_weight_acc_spec
    (dy x : Slice I64) (n i o : Usize)
    (hdy : dy.val.length = n.val * o.val)
    (hx : x.val.length = n.val * i.val)
    (hprod : o.val * i.val ≤ Usize.max)
    (hno : ∀ c' < o.val, ∀ k' < i.val, ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c' k' t ∧ bwdWAcc dy x i o c' k' t ≤ I128.max) :
    ∃ out, learn_port.dense.linear_bwd_weight_acc dy x n i o = ok (core.result.Result.Ok out) ∧
      ovals out = rowsOf i.val (bwdWElem dy x n i o) o.val := by
  have hdy' : n.val * o.val ≤ dy.val.length := by rw [hdy]
  have hx' : n.val * i.val ≤ x.val.length := by rw [hx]
  obtain ⟨t, ht, _htv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := o) (y := i) hprod)
  have hinv0 : bwdWOuterInv dy x n i o (alloc.vec.Vec.with_capacity I128 t) 0#usize true := by
    refine ⟨rfl, by show (0#usize).val ≤ o.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show ovals (alloc.vec.Vec.with_capacity I128 t) =
      rowsOf i.val (bwdWElem dy x n i o) (0#usize).val
    rw [show (0#usize).val = 0 from rfl, ovals_with_capacity]
    rfl
  have hspec := linear_bwd_weight_acc_loop0_spec dy x n i o (alloc.vec.Vec.with_capacity I128 t)
    0#usize hdy' hx' hprod hno hinv0
  obtain ⟨⟨out1, ok1⟩, hloop, hok1, hout1⟩ := (WP.spec_equiv_exists _ _).mp hspec
  subst hok1
  refine ⟨out1, ?_, hout1⟩
  unfold learn_port.dense.linear_bwd_weight_acc
  dsimp only
  rw [ht]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  simp

/-- `linear_bwd_weight` rounds the exact `i128` weight adjoint to Q32.32. -/
theorem linear_bwd_weight_ok (dy x : Slice I64) (n i o : Usize)
    (hn : n ≠ 0#usize) (hi : i ≠ 0#usize) (ho : o ≠ 0#usize)
    (hdy : dy.val.length = n.val * o.val)
    (hx : x.val.length = n.val * i.val)
    (hno_ : n.val * o.val ≤ Usize.max)
    (hni : n.val * i.val ≤ Usize.max)
    (hprod : o.val * i.val ≤ Usize.max)
    (hbound : ∀ c' < o.val, ∀ k' < i.val, ∀ t ≤ n.val,
      I128.min ≤ bwdWAcc dy x i o c' k' t ∧ bwdWAcc dy x i o c' k' t ≤ I128.max)
    (hround : ∀ t < o.val * i.val,
      I128.min ≤ (rowsOf i.val (bwdWElem dy x n i o) o.val).getD t 0 + 2 ^ 31 ∧
        (rowsOf i.val (bwdWElem dy x n i o) o.val).getD t 0 + 2 ^ 31 ≤ I128.max)
    (hnar : ∀ t < o.val * i.val,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((rowsOf i.val (bwdWElem dy x n i o) o.val).getD t 0) ∧
        rnd32 ((rowsOf i.val (bwdWElem dy x n i o) o.val).getD t 0) ≤ (2 : ℤ) ^ 63 - 1) :
    ∃ out, learn_port.dense.linear_bwd_weight dy x n i o = ok (core.result.Result.Ok out) ∧
      vvals out = rndAll (rowsOf i.val (bwdWElem dy x n i o) o.val) (o.val * i.val) := by
  have hexp1 := dense_expect_len_ok n o (Slice.len dy) hn ho hno_ (by rw [Slice.len_val]; exact hdy)
  have hexp2 := dense_expect_len_ok n i (Slice.len x) hn hi hni (by rw [Slice.len_val]; exact hx)
  obtain ⟨acc, hacc, haccv⟩ := linear_bwd_weight_acc_spec dy x n i o hdy hx hprod hbound
  have hlenacc : acc.val.length = o.val * i.val := by
    rw [← ovals_length, haccv, rowsOf_length]
  have hlen_deref : (alloc.vec.Vec.deref acc).val.length = o.val * i.val := by
    rw [deref_length, hlenacc]
  have hround' : ∀ t < (alloc.vec.Vec.deref acc).val.length,
      I128.min ≤ (ivals (alloc.vec.Vec.deref acc)).getD t 0 + 2 ^ 31 ∧
      (ivals (alloc.vec.Vec.deref acc)).getD t 0 + 2 ^ 31 ≤ I128.max := by
    intro t ht
    rw [ivals_deref, haccv]
    exact hround t (by rw [hlen_deref] at ht; exact ht)
  have hnar' : ∀ t < (alloc.vec.Vec.deref acc).val.length,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((ivals (alloc.vec.Vec.deref acc)).getD t 0) ∧
      rnd32 ((ivals (alloc.vec.Vec.deref acc)).getD t 0) ≤ (2 : ℤ) ^ 63 - 1 := by
    intro t ht
    rw [ivals_deref, haccv]
    exact hnar t (by rw [hlen_deref] at ht; exact ht)
  obtain ⟨out, hra, hrav⟩ := round_all_spec (alloc.vec.Vec.deref acc) hround' hnar'
  have hrav' : vvals out = rndAll (rowsOf i.val (bwdWElem dy x n i o) o.val) (o.val * i.val) := by
    rw [hrav, ivals_deref, haccv, deref_length, hlenacc]
  refine ⟨out, ?_, hrav'⟩
  unfold learn_port.dense.linear_bwd_weight
  simp only [hexp1, hexp2, hacc, hra, dense_branch_ok, bind_tc_ok]

end apeiron_word
