import ApeironWord
import Learn.Grad.HeadLoop
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted vocabulary head: the row, `W_head`-backward and fill loops

`src/learn/grad/head.rs` holds the head of the ported one-bit transfer. `Learn/Grad/Head.lean`
refines its non-loop bodies against the `ℤ` specifications, `Learn/Grad/HeadLoop.lean` the first
seven of its thirteen `_loop` bodies. This module refines the remaining ones: the `W_head`
backward (`dh_entry`, `dh_fill`), the TV gradient (`tv_dz`), the student logits (`logits_row`), the
softmax numerator and probability fills (`exp_fill`, `fill_lp_p`) and the row driver (`rows_loop`)
with the public entry points that compose them.

The accumulator models are the ones stated in `Learn/Grad/Head.lean`; they are not restated here.
`dhAux`/`dhEntryVal`/`dhVals` are the corrected `W_head` backward: `logits_row` forms
`z_k = Σ_j hidden[r·d+j]·w_head[k·d+j]` with `w_head` row-major, so `dh[dbase+j] = Σ_{k<v} dz[k]·w_head[k·d+j]`
sums over `k` at the fixed column `j`, which is what the extracted `dh_entry` computes.

`exp_entry` and `p_entry` reach the engine `fixed_exp`/`fixed_ln` through the head's own call
shape; their refinement is `Numeric.Engine.Attn`'s (`fixed_exp_refines_proved` and the
refusal/underflow lemmas), so the loops here are stated conditionally on the callee returning the
values the caller consumes, exactly as `log_softmax_row` is stated conditionally on `softmax_row`.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the container layout are not covered; only
the extracted integer bodies. The refusal arms carry the invariant's `fail` component: outside the
stated domain the body sets `fail` and leaves the accumulator alone.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_head

open apeiron_word
open Aeneas Aeneas.Std Result

/-! ## 10. `dh_entry` and `dh_fill`: the `W_head` backward

Specification (`Learn/Grad/Head.lean`): `dhAux dz w_head d j acc n = acc + Σ_{t<n} dz[t]·w_head[t·d+j]`,
`dhEntryVal dz w_head d j nscale = round(Σ_{k<v} dz[k]·w_head[k·d+j] / nscale)` with `v = dz.length`,
and `dhVals dz w_head d nscale base dh n` is `dh` with entry `base + t` set to `dhEntryVal ... t`.
The loop walks `k < v` accumulating the products; the domain carries the `usize` index fits, the
`w_head` read reach, the per-product `mul128` fit and the running-sum fit. -/

/-- `dhAux` is additive in its starting accumulator. -/
theorem dhAux_add (dz w_head : List ℤ) (d j : ℕ) (n : ℕ) (acc x : ℤ) :
    dhAux dz w_head d j (acc + x) n = dhAux dz w_head d j acc n + x := by
  induction n generalizing acc with
  | zero => rfl
  | succ k ih =>
    change dhAux dz w_head d j (acc + x + dz.getD k 0 * w_head.getD (k * d + j) 0) k =
      dhAux dz w_head d j (acc + dz.getD k 0 * w_head.getD (k * d + j) 0) k + x
    rw [show acc + x + dz.getD k 0 * w_head.getD (k * d + j) 0 =
        (acc + dz.getD k 0 * w_head.getD (k * d + j) 0) + x by ring]
    exact ih (acc + dz.getD k 0 * w_head.getD (k * d + j) 0)

/-- One `dhAux` step: the term at index `n` is added. -/
theorem dhAux_succ (dz w_head : List ℤ) (d j : ℕ) (acc : ℤ) (n : ℕ) :
    dhAux dz w_head d j acc (n + 1) =
      dhAux dz w_head d j acc n + dz.getD n 0 * w_head.getD (n * d + j) 0 := by
  change dhAux dz w_head d j (acc + dz.getD n 0 * w_head.getD (n * d + j) 0) n = _
  rw [dhAux_add]

/-- `dhVals` unfolded at a successor index. -/
theorem dhVals_succ (dz w_head : List ℤ) (d : ℕ) (nscale : ℤ) (base : ℕ) (dh : List ℤ) (n : ℕ) :
    dhVals dz w_head d nscale base dh (n + 1) =
      (dhVals dz w_head d nscale base dh n).set (base + n) (dhEntryVal dz w_head d n nscale) := rfl

theorem dhVals_length (dz w_head : List ℤ) (d : ℕ) (nscale : ℤ) (base : ℕ) (dh : List ℤ)
    (n : ℕ) : (dhVals dz w_head d nscale base dh n).length = dh.length := by
  induction n with
  | zero => rfl
  | succ k ih => rw [dhVals_succ, List.length_set, ih]

/-- The `dh_entry` loop invariant: `k` columns scanned, `acc` the exact prefix sum. -/
def dhEntryInv (dz w_head : Slice I64) (d j v : Usize) (acc0 : ℤ)
    (s : I128 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ v.val ∧
    s.1.val = dhAux (hvals dz) (hvals w_head) d.val j.val acc0 s.2.2.val

/-- The extracted `dh_entry` body: one column product is accumulated. -/
theorem dh_entry_loop_body_spec (dz w_head : Slice I64) (d j v : Usize) (acc0 : ℤ)
    (hdzl : ∀ t < v.val, t < dz.val.length)
    (hwm : ∀ t < v.val, t * d.val + j.val ≤ Usize.max)
    (hwr : ∀ t < v.val, t * d.val + j.val < w_head.val.length)
    (hprod : ∀ t < v.val, I128.min ≤
        (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + j.val) 0 ∧
      (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + j.val) 0 ≤ I128.max)
    (hsum : ∀ t ≤ v.val, I128.min ≤ dhAux (hvals dz) (hvals w_head) d.val j.val acc0 t ∧
      dhAux (hvals dz) (hvals w_head) d.val j.val acc0 t ≤ I128.max)
    (st : I128 × Option learn_port.head.HeadError × Usize)
    (hinv : dhEntryInv dz w_head d j v acc0 st) :
    learn_port.head.dh_entry_loop.body dz w_head d j v st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y =>
          y.1.val = dhAux (hvals dz) (hvals w_head) d.val j.val acc0 v.val ∧ y.2 = none
      | ControlFlow.cont st' => dhEntryInv dz w_head d j v acc0 st' ∧
          v.val - st'.2.2.val < v.val - st.2.2.val ⦄ := by
  obtain ⟨acc, fail, k⟩ := st
  simp only [dhEntryInv] at hinv
  obtain ⟨hfail, hle, hacc⟩ := hinv
  by_cases hlt : k < v
  · have hltv : k.val < v.val := (UScalar.lt_equiv k v).mp hlt
    have hklen : k.val < dz.val.length := hdzl k.val hltv
    have hidxz := slice_index_getD dz k hklen
    have hgdz : (hvals dz).getD k.val 0 = (dz.val.getD k.val default).val := hvals_getD dz hklen
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
      (Usize.mul_spec (x := k) (y := d) (by have := hwm k.val hltv; omega))
    have hi2vv : i2.val = k.val * d.val := hi2v
    obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := i2) (y := j) (by rw [hi2vv]; exact hwm k.val hltv))
    have hi3vv : i3.val = k.val * d.val + j.val := by rw [hi3v, hi2vv]
    have hi3len : i3.val < w_head.val.length := by rw [hi3vv]; exact hwr k.val hltv
    have hidxw := slice_index_getD w_head i3 hi3len
    have hgw : (hvals w_head).getD i3.val 0 = (w_head.val.getD i3.val default).val :=
      hvals_getD w_head hi3len
    have hfit : I128.min ≤
          (core.convert.num.FromI128I64.from (dz.val.getD k.val default)).val *
            (core.convert.num.FromI128I64.from (w_head.val.getD i3.val default)).val ∧
        (core.convert.num.FromI128I64.from (dz.val.getD k.val default)).val *
            (core.convert.num.FromI128I64.from (w_head.val.getD i3.val default)).val ≤
          I128.max := by
      rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
        ← hgdz, ← hgw, hi3vv]
      exact hprod k.val hltv
    obtain ⟨term, hmul, htermv⟩ :=
      mul128_ok (core.convert.num.FromI128I64.from (dz.val.getD k.val default))
        (core.convert.num.FromI128I64.from (w_head.val.getD i3.val default)) hfit
    have heq : acc.val + term.val =
        dhAux (hvals dz) (hvals w_head) d.val j.val acc0 (k.val + 1) := by
      rw [hacc, htermv, core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq, ← hgdz, ← hgw, hi3vv, dhAux_succ]
    obtain ⟨s, hadd, hsv⟩ := add128_ok acc term (by rw [heq]; exact hsum (k.val + 1) (by omega))
    obtain ⟨k1, hk1, hk1v⟩ := usize_succ_fits hltv (by scalar_tac)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.dh_entry_loop.body dz w_head d j v acc fail k =
        ok (ControlFlow.cont (s, none, k1)) := by
      rw [learn_port.head.dh_entry_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, lift, hidxz, hi2, hi3, hidxw, hmul, hadd, hk1]
      rw [hfail]
      rfl
    rw [hbody]
    simp only [WP.spec_ok, dhEntryInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hsv, heq, hk1v]
  · have hbody : learn_port.head.dh_entry_loop.body dz w_head d j v acc fail k =
        ok (ControlFlow.done (acc, fail)) := by
      rw [learn_port.head.dh_entry_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hkeq : k.val = v.val := by
      have hvle : v.val ≤ k.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv k v).mpr (by omega))
      omega
    exact ⟨by rw [← hkeq, hacc], hfail⟩

/-- **The `dh_entry` loop.** From the zero accumulator the loop returns the whole sum. -/
theorem dh_entry_loop_spec (dz w_head : Slice I64) (d j v : Usize) (acc : I128)
    (fail : Option learn_port.head.HeadError) (k : Usize) (acc0 : ℤ)
    (hdzl : ∀ t < v.val, t < dz.val.length)
    (hwm : ∀ t < v.val, t * d.val + j.val ≤ Usize.max)
    (hwr : ∀ t < v.val, t * d.val + j.val < w_head.val.length)
    (hprod : ∀ t < v.val, I128.min ≤
        (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + j.val) 0 ∧
      (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + j.val) 0 ≤ I128.max)
    (hsum : ∀ t ≤ v.val, I128.min ≤ dhAux (hvals dz) (hvals w_head) d.val j.val acc0 t ∧
      dhAux (hvals dz) (hvals w_head) d.val j.val acc0 t ≤ I128.max)
    (hinv0 : dhEntryInv dz w_head d j v acc0 (acc, fail, k)) :
    WP.spec (learn_port.head.dh_entry_loop dz w_head d j v acc fail k)
      (fun y : I128 × Option learn_port.head.HeadError =>
        y.1.val = dhAux (hvals dz) (hvals w_head) d.val j.val acc0 v.val ∧ y.2 = none) := by
  unfold learn_port.head.dh_entry_loop
  apply loop.spec_decr_nat
    (measure := fun s : I128 × Option learn_port.head.HeadError × Usize => v.val - s.2.2.val)
    (inv := dhEntryInv dz w_head d j v acc0)
    (post := fun y : I128 × Option learn_port.head.HeadError =>
      y.1.val = dhAux (hvals dz) (hvals w_head) d.val j.val acc0 v.val ∧ y.2 = none)
  · intro st hst
    convert dh_entry_loop_body_spec dz w_head d j v acc0 hdzl hwm hwr hprod hsum st hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-- **`dh_entry`.** The extracted body returns the column's rounded `W_head` backward value. -/
theorem dh_entry_spec (dz w_head : Slice I64) (d j : Usize) (nscale : I128)
    (hwm : ∀ t < dz.val.length, t * d.val + j.val ≤ Usize.max)
    (hwr : ∀ t < dz.val.length, t * d.val + j.val < w_head.val.length)
    (hprod : ∀ t < dz.val.length, I128.min ≤
        (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + j.val) 0 ∧
      (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + j.val) 0 ≤ I128.max)
    (hsum : ∀ t ≤ dz.val.length,
      I128.min ≤ dhAux (hvals dz) (hvals w_head) d.val j.val 0 t ∧
      dhAux (hvals dz) (hvals w_head) d.val j.val 0 t ≤ I128.max)
    (hpos : 0 < nscale.val)
    (h2 : 2 * absZ (Int.tmod
        (dhAux (hvals dz) (hvals w_head) d.val j.val 0 dz.val.length) nscale.val) ≤ I128.max)
    (hdom : rndDivDom (dhAux (hvals dz) (hvals w_head) d.val j.val 0 dz.val.length) nscale.val)
    (hfit : I64.min ≤ dhEntryVal (hvals dz) (hvals w_head) d.val j.val nscale.val ∧
      dhEntryVal (hvals dz) (hvals w_head) d.val j.val nscale.val ≤ I64.max) :
    ∃ g, learn_port.head.dh_entry dz w_head d j nscale = ok (core.result.Result.Ok g) ∧
      g.val = dhEntryVal (hvals dz) (hvals w_head) d.val j.val nscale.val := by
  unfold learn_port.head.dh_entry
  dsimp only
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (dh_entry_loop_spec dz w_head d j (Slice.len dz) 0#i128 none 0#usize 0
      (fun t ht => ht) hwm hwr hprod hsum (by refine ⟨rfl, by simp, rfl⟩))
  obtain ⟨acc, fail⟩ := y
  have hy1' : acc.val = dhAux (hvals dz) (hvals w_head) d.val j.val 0 dz.val.length := by
    rw [show (acc, fail).1 = acc from rfl] at hy1
    rw [hy1, Slice.len_val]
  have hfail : fail = none := hy2
  rw [hfail] at hy
  rw [hy]
  simp only [bind_tc_ok]
  change ∃ g, (do
      let i ← learn_port.head.round_div acc nscale
      learn_port.head.to64 i) = ok (core.result.Result.Ok g) ∧
      g.val = dhEntryVal (hvals dz) (hvals w_head) d.val j.val nscale.val
  obtain ⟨v, hv, hvv⟩ := round_div_dom_spec acc nscale hpos (by rw [hy1']; exact h2)
    (by rw [hy1']; exact hdom)
  rw [hv]
  simp only [bind_tc_ok]
  obtain ⟨g, hg, hgv⟩ := to64_ok v (by
    rw [hvv, hy1']
    simpa only [dhEntryVal, hvals_length] using hfit)
  exact ⟨g, hg, by
    simp only [dhEntryVal]
    rw [hgv, hvv, hy1', hvals_length]⟩

/-- The `dh_fill` loop invariant: `j` columns written, matched against `dhVals`. -/
def dhFillInv (dz w_head : Slice I64) (d : Usize) (nscale : I128) (dbase : Usize)
    (dh0 : List ℤ) (s : Slice I64 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ d.val ∧
    hvals s.1 = dhVals (hvals dz) (hvals w_head) d.val nscale.val dbase.val dh0 s.2.2.val

/-- The extracted `dh_fill` body: one column's gradient is written. -/
theorem dh_fill_loop_body_spec (dz w_head : Slice I64) (d : Usize) (nscale : I128)
    (dbase : Usize) (dh0 : List ℤ)
    (hwm : ∀ jj < d.val, ∀ t < dz.val.length, t * d.val + jj ≤ Usize.max)
    (hwr : ∀ jj < d.val, ∀ t < dz.val.length, t * d.val + jj < w_head.val.length)
    (hprod : ∀ jj < d.val, ∀ t < dz.val.length, I128.min ≤
        (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + jj) 0 ∧
      (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + jj) 0 ≤ I128.max)
    (hsum : ∀ jj < d.val, ∀ t ≤ dz.val.length,
      I128.min ≤ dhAux (hvals dz) (hvals w_head) d.val jj 0 t ∧
      dhAux (hvals dz) (hvals w_head) d.val jj 0 t ≤ I128.max)
    (h2 : ∀ jj < d.val, 2 * absZ (Int.tmod
        (dhAux (hvals dz) (hvals w_head) d.val jj 0 dz.val.length) nscale.val) ≤ I128.max)
    (hdom : ∀ jj < d.val,
      rndDivDom (dhAux (hvals dz) (hvals w_head) d.val jj 0 dz.val.length) nscale.val)
    (hfit : ∀ jj < d.val, I64.min ≤ dhEntryVal (hvals dz) (hvals w_head) d.val jj nscale.val ∧
      dhEntryVal (hvals dz) (hvals w_head) d.val jj nscale.val ≤ I64.max)
    (hpos : 0 < nscale.val)
    (hdb : ∀ t < d.val, dbase.val + t ≤ Usize.max)
    (hdi : ∀ t < d.val, dbase.val + t < dh0.length)
    (st : Slice I64 × Option learn_port.head.HeadError × Usize)
    (hinv : dhFillInv dz w_head d nscale dbase dh0 st) :
    learn_port.head.dh_fill_loop.body dz w_head d nscale dbase st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y =>
          hvals y.2 = dhVals (hvals dz) (hvals w_head) d.val nscale.val dbase.val dh0 d.val ∧
            y.1 = none
      | ControlFlow.cont st' => dhFillInv dz w_head d nscale dbase dh0 st' ∧
          d.val - st'.2.2.val < d.val - st.2.2.val ⦄ := by
  obtain ⟨dh, fail, j⟩ := st
  simp only [dhFillInv] at hinv
  obtain ⟨hfail, hle, hdh⟩ := hinv
  by_cases hlt : j < d
  · have hltv : j.val < d.val := (UScalar.lt_equiv j d).mp hlt
    obtain ⟨g, hgentry, hgv⟩ := dh_entry_spec dz w_head d j nscale
      (hwm j.val hltv) (hwr j.val hltv) (hprod j.val hltv) (hsum j.val hltv)
      hpos (h2 j.val hltv) (hdom j.val hltv) (hfit j.val hltv)
    have hdhl : dh.val.length = dh0.length := by
      rw [← hvals_length dh, hdh, dhVals_length]
    obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := dbase) (y := j) (by have := hdb j.val hltv; omega))
    have hilen : i.val < dh.val.length := by rw [hiv, hdhl]; exact hdi j.val hltv
    obtain ⟨dh1, hupd, hdh1⟩ := WP.spec_imp_exists (Slice.update_spec dh i g hilen)
    obtain ⟨j1, hj1, hj1v⟩ := usize_succ_fits hltv (by scalar_tac)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.dh_fill_loop.body dz w_head d nscale dbase dh fail j =
        ok (ControlFlow.cont (dh1, none, j1)) := by
      rw [learn_port.head.dh_fill_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, hgentry, hi, hupd, hj1]
      rw [hfail]
      rfl
    rw [hbody]
    simp only [WP.spec_ok, dhFillInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hdh1, hvals_set, hdh, hj1v, hiv, dhVals_succ, hgv]
  · have hbody : learn_port.head.dh_fill_loop.body dz w_head d nscale dbase dh fail j =
        ok (ControlFlow.done (fail, dh)) := by
      rw [learn_port.head.dh_fill_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hjeq : j.val = d.val := by
      have hvle : d.val ≤ j.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv j d).mpr (by omega))
      omega
    exact ⟨by rw [hdh, hjeq], hfail⟩

/-- **The `dh_fill` loop.** Over the `d`-block the loop writes every column's gradient. -/
theorem dh_fill_loop_spec (dz w_head : Slice I64) (d : Usize) (nscale : I128) (dbase : Usize)
    (dh : Slice I64) (fail : Option learn_port.head.HeadError) (j : Usize)
    (hwm : ∀ jj < d.val, ∀ t < dz.val.length, t * d.val + jj ≤ Usize.max)
    (hwr : ∀ jj < d.val, ∀ t < dz.val.length, t * d.val + jj < w_head.val.length)
    (hprod : ∀ jj < d.val, ∀ t < dz.val.length, I128.min ≤
        (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + jj) 0 ∧
      (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + jj) 0 ≤ I128.max)
    (hsum : ∀ jj < d.val, ∀ t ≤ dz.val.length,
      I128.min ≤ dhAux (hvals dz) (hvals w_head) d.val jj 0 t ∧
      dhAux (hvals dz) (hvals w_head) d.val jj 0 t ≤ I128.max)
    (h2 : ∀ jj < d.val, 2 * absZ (Int.tmod
        (dhAux (hvals dz) (hvals w_head) d.val jj 0 dz.val.length) nscale.val) ≤ I128.max)
    (hdom : ∀ jj < d.val,
      rndDivDom (dhAux (hvals dz) (hvals w_head) d.val jj 0 dz.val.length) nscale.val)
    (hfit : ∀ jj < d.val, I64.min ≤ dhEntryVal (hvals dz) (hvals w_head) d.val jj nscale.val ∧
      dhEntryVal (hvals dz) (hvals w_head) d.val jj nscale.val ≤ I64.max)
    (hpos : 0 < nscale.val)
    (hdb : ∀ t < d.val, dbase.val + t ≤ Usize.max)
    (hdi : ∀ t < d.val, dbase.val + t < dh.val.length)
    (hinv0 : dhFillInv dz w_head d nscale dbase (hvals dh) (dh, fail, j)) :
    WP.spec (learn_port.head.dh_fill_loop dz w_head d nscale dh dbase fail j)
      (fun y : Option learn_port.head.HeadError × Slice I64 =>
        hvals y.2 = dhVals (hvals dz) (hvals w_head) d.val nscale.val dbase.val (hvals dh) d.val ∧
          y.1 = none) := by
  unfold learn_port.head.dh_fill_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × Option learn_port.head.HeadError × Usize =>
      d.val - s.2.2.val)
    (inv := dhFillInv dz w_head d nscale dbase (hvals dh))
    (post := fun y : Option learn_port.head.HeadError × Slice I64 =>
      hvals y.2 = dhVals (hvals dz) (hvals w_head) d.val nscale.val dbase.val (hvals dh) d.val ∧
        y.1 = none)
  · intro st hst
    convert dh_fill_loop_body_spec dz w_head d nscale dbase (hvals dh) hwm hwr hprod hsum h2 hdom
      hfit hpos hdb (by rw [hvals_length]; exact hdi) st hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-- **`dh_fill`.** The extracted body writes every column of the block. -/
theorem dh_fill_spec (dz w_head : Slice I64) (d : Usize) (nscale : I128) (dh : Slice I64)
    (dbase : Usize)
    (hwm : ∀ jj < d.val, ∀ t < dz.val.length, t * d.val + jj ≤ Usize.max)
    (hwr : ∀ jj < d.val, ∀ t < dz.val.length, t * d.val + jj < w_head.val.length)
    (hprod : ∀ jj < d.val, ∀ t < dz.val.length, I128.min ≤
        (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + jj) 0 ∧
      (hvals dz).getD t 0 * (hvals w_head).getD (t * d.val + jj) 0 ≤ I128.max)
    (hsum : ∀ jj < d.val, ∀ t ≤ dz.val.length,
      I128.min ≤ dhAux (hvals dz) (hvals w_head) d.val jj 0 t ∧
      dhAux (hvals dz) (hvals w_head) d.val jj 0 t ≤ I128.max)
    (h2 : ∀ jj < d.val, 2 * absZ (Int.tmod
        (dhAux (hvals dz) (hvals w_head) d.val jj 0 dz.val.length) nscale.val) ≤ I128.max)
    (hdom : ∀ jj < d.val,
      rndDivDom (dhAux (hvals dz) (hvals w_head) d.val jj 0 dz.val.length) nscale.val)
    (hfit : ∀ jj < d.val, I64.min ≤ dhEntryVal (hvals dz) (hvals w_head) d.val jj nscale.val ∧
      dhEntryVal (hvals dz) (hvals w_head) d.val jj nscale.val ≤ I64.max)
    (hpos : 0 < nscale.val)
    (hdb : ∀ t < d.val, dbase.val + t ≤ Usize.max)
    (hdi : ∀ t < d.val, dbase.val + t < dh.val.length) :
    ∃ y, learn_port.head.dh_fill dz w_head d nscale dh dbase = ok (none, y) ∧
      hvals y = dhVals (hvals dz) (hvals w_head) d.val nscale.val dbase.val (hvals dh) d.val := by
  unfold learn_port.head.dh_fill
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (dh_fill_loop_spec dz w_head d nscale dbase dh none 0#usize hwm hwr hprod hsum h2 hdom hfit
      hpos hdb hdi (by refine ⟨rfl, by simp, rfl⟩))
  obtain ⟨a, b⟩ := y
  have hnone : a = none := hy2
  exact ⟨b, by rw [hy, hnone], hy1⟩

/-! ## 11. `tv_dz`: the TV gradient

Specification (`Learn/Grad/Head.lean`): `tvDzVals p_s sg signed dz n` is `dz` with entry `t < n` set
to `round(p_s[t]·(sg[t]·2^32 − signed) / (2·2^32))`. The loop walks `i < v` writing the gradient; the
domain carries the three checked `i128` fits (`sg[i]·one`, that minus `signed`, and `2·one`), the
`mul128` product fit, the `round_div` domain and the `to64` fit. -/

/-- The model's own numerator `p_s[t]·(sg[t]·2^32 − signed)`, as a rewrite target. -/
def tvDzNum (p_s sg : List ℤ) (signed : ℤ) (t : ℕ) : ℤ :=
  p_s.getD t 0 * (sg.getD t 0 * 2 ^ (32 : ℕ) - signed)

theorem tvDzNum_eq (p_s sg : List ℤ) (signed : ℤ) (t : ℕ) :
    tvDzNum p_s sg signed t = p_s.getD t 0 * (sg.getD t 0 * 2 ^ (32 : ℕ) - signed) := rfl

/-- `tvDzVals` unfolded at a successor index. -/
theorem tvDzVals_succ (p_s sg : List ℤ) (signed : ℤ) (dz : List ℤ) (t : ℕ) :
    tvDzVals p_s sg signed dz (t + 1) =
      (tvDzVals p_s sg signed dz t).set t (rndDiv (tvDzNum p_s sg signed t) (2 * 2 ^ (32 : ℕ))) :=
  rfl

theorem tvDzVals_length (p_s sg : List ℤ) (signed : ℤ) (dz : List ℤ) (t : ℕ) :
    (tvDzVals p_s sg signed dz t).length = dz.length := by
  induction t with
  | zero => rfl
  | succ k ih => rw [tvDzVals_succ, List.length_set, ih]

/-- The `tv_dz` loop invariant: `i` entries written, matched against `tvDzVals`. -/
def tvDzInv (p_s : Slice I64) (sg : Slice I128) (signed : I128) (v : Usize) (dz0 : List ℤ)
    (s : Slice I64 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ v.val ∧
    hvals s.1 = tvDzVals (hvals p_s) (svals sg) signed.val dz0 s.2.2.val

/-- The extracted `tv_dz` body: one gradient entry is written. -/
theorem tv_dz_loop_body_spec (p_s : Slice I64) (sg : Slice I128) (signed : I128) (v : Usize)
    (one : I128) (dz0 : List ℤ)
    (hone : one.val = 2 ^ (32 : ℕ))
    (hp : ∀ t < v.val, t < p_s.val.length)
    (hg : ∀ t < v.val, t < sg.val.length)
    (hdz : ∀ t < v.val, t < dz0.length)
    (hg1 : ∀ t < v.val, I128.min ≤ (svals sg).getD t 0 * one.val ∧
      (svals sg).getD t 0 * one.val ≤ I128.max)
    (hs1 : ∀ t < v.val, I128.min ≤ (svals sg).getD t 0 * one.val - signed.val ∧
      (svals sg).getD t 0 * one.val - signed.val ≤ I128.max)
    (hn : ∀ t < v.val, I128.min ≤ tvDzNum (hvals p_s) (svals sg) signed.val t ∧
      tvDzNum (hvals p_s) (svals sg) signed.val t ≤ I128.max)
    (h3 : I128.min ≤ 2 * one.val ∧ 2 * one.val ≤ I128.max)
    (hrnd : ∀ t < v.val,
      rndDivDom (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * one.val) ∧
      2 * absZ (Int.tmod (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * one.val)) ≤ I128.max)
    (hfit : ∀ t < v.val,
      I64.min ≤ rndDiv (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * one.val) ∧
      rndDiv (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * one.val) ≤ I64.max)
    (st : Slice I64 × Option learn_port.head.HeadError × Usize)
    (hinv : tvDzInv p_s sg signed v dz0 st) :
    learn_port.head.tv_dz_loop.body p_s sg signed v one st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y =>
          hvals y.2 = tvDzVals (hvals p_s) (svals sg) signed.val dz0 v.val ∧ y.1 = none
      | ControlFlow.cont st' => tvDzInv p_s sg signed v dz0 st' ∧
          v.val - st'.2.2.val < v.val - st.2.2.val ⦄ := by
  obtain ⟨dz, fail, i⟩ := st
  simp only [tvDzInv] at hinv
  obtain ⟨hfail, hle, hdzv⟩ := hinv
  by_cases hlt : i < v
  · have hltv : i.val < v.val := (UScalar.lt_equiv i v).mp hlt
    have hlen_p : i.val < p_s.val.length := hp i.val hltv
    have hlen_g : i.val < sg.val.length := hg i.val hltv
    have hlen_dz : i.val < dz.val.length := by
      rw [← hvals_length dz, hdzv, tvDzVals_length]; exact hdz i.val hltv
    have hidxp := slice_index_getD p_s i hlen_p
    have hgp : (hvals p_s).getD i.val 0 = (p_s.val.getD i.val default).val := hvals_getD p_s hlen_p
    have hidxg := slice_index_getD sg i hlen_g
    have hgg : (svals sg).getD i.val 0 = (sg.val.getD i.val default).val := svals_getD sg hlen_g
    have h2i : (↑(2#i128) : ℤ) = 2 := rfl
    obtain ⟨i4, hi4, hi4v, _⟩ := WP.spec_imp_exists (I128.mul_bv_spec
      (x := sg.val.getD i.val default) (y := one)
      (by rw [← hgg]; exact (hg1 i.val hltv).1)
      (by rw [← hgg]; exact (hg1 i.val hltv).2))
    have hi4v' : i4.val = (svals sg).getD i.val 0 * one.val := by rw [hi4v, ← hgg]
    obtain ⟨i5, hi5, hi5v, _⟩ := WP.spec_imp_exists (I128.sub_bv_spec (x := i4) (y := signed)
      (by rw [hi4v']; exact (hs1 i.val hltv).1)
      (by rw [hi4v']; exact (hs1 i.val hltv).2))
    have hi5v' : i5.val = (svals sg).getD i.val 0 * one.val - signed.val := by rw [hi5v, hi4v']
    obtain ⟨num, hnum, hnumv⟩ :=
      mul128_ok (core.convert.num.FromI128I64.from (p_s.val.getD i.val default)) i5 (by
        rw [core.convert.num.FromI128I64.from_val_eq, ← hgp, hi5v']
        simpa only [tvDzNum, hone] using hn i.val hltv)
    have hnum_eq : num.val = tvDzNum (hvals p_s) (svals sg) signed.val i.val := by
      rw [hnumv, core.convert.num.FromI128I64.from_val_eq, ← hgp, hi5v']
      simp only [tvDzNum, hone]
    obtain ⟨i6, hi6, hi6v, _⟩ := WP.spec_imp_exists (I128.mul_bv_spec (x := 2#i128) (y := one)
      (by rw [h2i]; exact h3.1) (by rw [h2i]; exact h3.2))
    have hi6v' : i6.val = 2 * one.val := by rw [hi6v, h2i]
    obtain ⟨i7, hi7, hi7v⟩ := round_div_dom_spec num i6 (by rw [hi6v', hone]; norm_num)
      (by rw [hnum_eq, hi6v']; exact (hrnd i.val hltv).2)
      (by rw [hnum_eq, hi6v']; exact (hrnd i.val hltv).1)
    obtain ⟨g, hg', hgv⟩ := to64_ok i7 (by rw [hi7v, hnum_eq, hi6v']; exact hfit i.val hltv)
    obtain ⟨dz1, hupd, hdz1⟩ := WP.spec_imp_exists (Slice.update_spec dz i g hlen_dz)
    obtain ⟨i9, hi9, hi9v⟩ := usize_succ_fits hltv (by scalar_tac)
    have hval_eq : g.val = rndDiv ((hvals p_s).getD i.val 0 *
        ((svals sg).getD i.val 0 * 2 ^ (32 : ℕ) - signed.val)) (2 * 2 ^ (32 : ℕ)) := by
      rw [hgv, hi7v, hnum_eq, hi6v', hone]
      simp only [tvDzNum]
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.tv_dz_loop.body p_s sg signed v one dz fail i =
        ok (ControlFlow.cont (dz1, none, i9)) := by
      rw [learn_port.head.tv_dz_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, lift, hidxp, hidxg, hi4, hi5, hnum, hi6, hi7, hg',
        hupd, hi9]
      rw [hfail]
      rfl
    rw [hbody]
    simp only [WP.spec_ok, tvDzInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hdz1, hvals_set, hdzv, hi9v, tvDzVals_succ]
    rw [hval_eq]
    rfl
  · have hbody : learn_port.head.tv_dz_loop.body p_s sg signed v one dz fail i =
        ok (ControlFlow.done (fail, dz)) := by
      rw [learn_port.head.tv_dz_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hieq : i.val = v.val := by
      have hvle : v.val ≤ i.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv i v).mpr (by omega))
      omega
    exact ⟨by rw [← hieq, hdzv], hfail⟩

/-- **The `tv_dz` loop.** Over the whole row the loop writes every gradient entry. -/
theorem tv_dz_loop_spec (p_s : Slice I64) (sg : Slice I128) (signed : I128) (dz : Slice I64)
    (v : Usize) (one : I128) (fail : Option learn_port.head.HeadError) (i : Usize)
    (hone : one.val = 2 ^ (32 : ℕ))
    (hp : ∀ t < v.val, t < p_s.val.length)
    (hg : ∀ t < v.val, t < sg.val.length)
    (hdz : ∀ t < v.val, t < dz.val.length)
    (hg1 : ∀ t < v.val, I128.min ≤ (svals sg).getD t 0 * one.val ∧
      (svals sg).getD t 0 * one.val ≤ I128.max)
    (hs1 : ∀ t < v.val, I128.min ≤ (svals sg).getD t 0 * one.val - signed.val ∧
      (svals sg).getD t 0 * one.val - signed.val ≤ I128.max)
    (hn : ∀ t < v.val, I128.min ≤ tvDzNum (hvals p_s) (svals sg) signed.val t ∧
      tvDzNum (hvals p_s) (svals sg) signed.val t ≤ I128.max)
    (h3 : I128.min ≤ 2 * one.val ∧ 2 * one.val ≤ I128.max)
    (hrnd : ∀ t < v.val,
      rndDivDom (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * one.val) ∧
      2 * absZ (Int.tmod (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * one.val)) ≤ I128.max)
    (hfit : ∀ t < v.val,
      I64.min ≤ rndDiv (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * one.val) ∧
      rndDiv (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * one.val) ≤ I64.max)
    (hinv0 : tvDzInv p_s sg signed v (hvals dz) (dz, fail, i)) :
    WP.spec (learn_port.head.tv_dz_loop p_s sg signed dz v one fail i)
      (fun y : Option learn_port.head.HeadError × Slice I64 =>
        hvals y.2 = tvDzVals (hvals p_s) (svals sg) signed.val (hvals dz) v.val ∧ y.1 = none) := by
  unfold learn_port.head.tv_dz_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × Option learn_port.head.HeadError × Usize =>
      v.val - s.2.2.val)
    (inv := tvDzInv p_s sg signed v (hvals dz))
    (post := fun y : Option learn_port.head.HeadError × Slice I64 =>
      hvals y.2 = tvDzVals (hvals p_s) (svals sg) signed.val (hvals dz) v.val ∧ y.1 = none)
  · intro st hst
    convert tv_dz_loop_body_spec p_s sg signed v one (hvals dz) hone hp hg
      (by rw [hvals_length]; exact hdz) hg1 hs1 hn h3 hrnd hfit st hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-- **`tv_dz`.** The extracted body writes every TV gradient entry, on the checked-`i128` domain. -/
theorem tv_dz_spec (p_s : Slice I64) (sg : Slice I128) (signed : I128) (dz : Slice I64)
    (hg : ∀ t < p_s.val.length, t < sg.val.length)
    (hdz : ∀ t < p_s.val.length, t < dz.val.length)
    (hg1 : ∀ t < p_s.val.length, I128.min ≤ (svals sg).getD t 0 * 2 ^ (32 : ℕ) ∧
      (svals sg).getD t 0 * 2 ^ (32 : ℕ) ≤ I128.max)
    (hs1 : ∀ t < p_s.val.length, I128.min ≤ (svals sg).getD t 0 * 2 ^ (32 : ℕ) - signed.val ∧
      (svals sg).getD t 0 * 2 ^ (32 : ℕ) - signed.val ≤ I128.max)
    (hn : ∀ t < p_s.val.length, I128.min ≤ tvDzNum (hvals p_s) (svals sg) signed.val t ∧
      tvDzNum (hvals p_s) (svals sg) signed.val t ≤ I128.max)
    (h3 : I128.min ≤ 2 * 2 ^ (32 : ℕ) ∧ 2 * 2 ^ (32 : ℕ) ≤ I128.max)
    (hrnd : ∀ t < p_s.val.length,
      rndDivDom (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * 2 ^ (32 : ℕ)) ∧
      2 * absZ (Int.tmod (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * 2 ^ (32 : ℕ))) ≤
        I128.max)
    (hfit : ∀ t < p_s.val.length,
      I64.min ≤ rndDiv (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * 2 ^ (32 : ℕ)) ∧
      rndDiv (tvDzNum (hvals p_s) (svals sg) signed.val t) (2 * 2 ^ (32 : ℕ)) ≤ I64.max) :
    ∃ y, learn_port.head.tv_dz p_s sg signed dz = ok (none, y) ∧
      hvals y = tvDzVals (hvals p_s) (svals sg) signed.val (hvals dz) p_s.val.length := by
  unfold learn_port.head.tv_dz
  dsimp only
  rw [engine_ONE_eq]
  simp only [lift, bind_tc_ok]
  have hone : (core.convert.num.FromI128I64.from (4294967296#i64)).val = 2 ^ (32 : ℕ) := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    norm_num
  have hv : (Slice.len p_s).val = p_s.val.length := Slice.len_val p_s
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (tv_dz_loop_spec p_s sg signed dz (Slice.len p_s)
      (core.convert.num.FromI128I64.from (4294967296#i64)) none 0#usize hone
      (fun t ht => by rw [hv] at ht; exact ht)
      (fun t ht => hg t (by rw [hv] at ht; exact ht))
      (fun t ht => hdz t (by rw [hv] at ht; exact ht))
      (fun t ht => by rw [hone]; exact hg1 t (by rw [hv] at ht; exact ht))
      (fun t ht => by rw [hone]; exact hs1 t (by rw [hv] at ht; exact ht))
      (fun t ht => hn t (by rw [hv] at ht; exact ht))
      (by rw [hone]; exact h3)
      (fun t ht => by rw [hone]; exact hrnd t (by rw [hv] at ht; exact ht))
      (fun t ht => by rw [hone]; exact hfit t (by rw [hv] at ht; exact ht))
      (by refine ⟨rfl, by simp, rfl⟩))
  obtain ⟨a, b⟩ := y
  have hnone : a = none := hy2
  exact ⟨b, by rw [hy, hnone], by rw [hy1, hv]⟩

/-! ## 12. `dot_round`: the rounded dot product (public entry)

`dot_round` runs `dot_round_loop` from the zero accumulator and divides the exact `i128` sum by
`2^32` (`engine.ONE`), narrowing once. The domain adds the `round_div`/`to64` fits of the sum to the
loop's per-product and prefix-sum fits. -/

theorem dot_round_spec (a : Slice I64) (abase : Usize) (b : Slice I64) (bbase : Usize) (d : Usize)
    (hab : ∀ t < d.val, abase.val + t ≤ Usize.max)
    (hbb : ∀ t < d.val, bbase.val + t ≤ Usize.max)
    (hla : ∀ t < d.val, abase.val + t < a.val.length)
    (hlb : ∀ t < d.val, bbase.val + t < b.val.length)
    (hprod : ∀ t < d.val,
      I128.min ≤ (hvals a).getD (abase.val + t) 0 * (hvals b).getD (bbase.val + t) 0 ∧
      (hvals a).getD (abase.val + t) 0 * (hvals b).getD (bbase.val + t) 0 ≤ I128.max)
    (hsum : ∀ t ≤ d.val, I128.min ≤ dotAux (hvals a) abase.val (hvals b) bbase.val 0 t ∧
      dotAux (hvals a) abase.val (hvals b) bbase.val 0 t ≤ I128.max)
    (h2 : 2 * absZ (Int.tmod (dotAux (hvals a) abase.val (hvals b) bbase.val 0 d.val)
      (2 ^ (32 : ℕ))) ≤ I128.max)
    (hrnd : rndDivDom (dotAux (hvals a) abase.val (hvals b) bbase.val 0 d.val) (2 ^ (32 : ℕ)))
    (hfit : I64.min ≤ rndDiv (dotAux (hvals a) abase.val (hvals b) bbase.val 0 d.val)
        (2 ^ (32 : ℕ)) ∧
      rndDiv (dotAux (hvals a) abase.val (hvals b) bbase.val 0 d.val) (2 ^ (32 : ℕ)) ≤ I64.max) :
    ∃ z, learn_port.head.dot_round a abase b bbase d = ok (core.result.Result.Ok z) ∧
      z.val = rndDiv (dotAux (hvals a) abase.val (hvals b) bbase.val 0 d.val) (2 ^ (32 : ℕ)) := by
  have hz0 : (0#i128).val = 0 := rfl
  have hsum0 : ∀ t ≤ d.val,
      I128.min ≤ dotAux (hvals a) abase.val (hvals b) bbase.val (0#i128).val t ∧
      dotAux (hvals a) abase.val (hvals b) bbase.val (0#i128).val t ≤ I128.max := by
    intro t ht; rw [hz0]; exact hsum t ht
  unfold learn_port.head.dot_round
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (dot_round_loop_spec a abase b bbase d 0#i128 none 0#usize hab hbb hla hlb hprod hsum0
      (by refine ⟨rfl, by simp, rfl⟩))
  obtain ⟨acc, fail⟩ := y
  have hfail : fail = none := hy2
  have hacc : acc.val = dotAux (hvals a) abase.val (hvals b) bbase.val 0 d.val := by
    rw [hy1, hz0]
  rw [hy, hfail]
  rw [engine_ONE_eq]
  simp only [lift, bind_tc_ok]
  have hone : (core.convert.num.FromI128I64.from (4294967296#i64)).val = 2 ^ (32 : ℕ) := by
    rw [core.convert.num.FromI128I64.from_val_eq]; norm_num
  obtain ⟨i2, hi2, hi2v⟩ :=
    round_div_dom_spec acc (core.convert.num.FromI128I64.from (4294967296#i64))
      (by rw [hone]; norm_num)
      (by rw [hacc, hone]; exact h2)
      (by rw [hacc, hone]; exact hrnd)
  obtain ⟨z, hz, hzv⟩ := to64_ok i2 (by rw [hi2v, hacc, hone]; exact hfit)
  refine ⟨z, ?_, ?_⟩
  · change (learn_port.head.round_div acc (core.convert.num.FromI128I64.from (4294967296#i64))
        >>= fun i2 => learn_port.head.to64 i2) = ok (core.result.Result.Ok z)
    rw [hi2]
    simp only [bind_tc_ok]
    exact hz
  · rw [hzv, hi2v, hacc, hone]

/-! ## 13. `logits_row`: the student logits of one row

Specification (`Learn/Grad/Head.lean`): `logitsVals hidden w_head r d zs n` is `zs` with entry `k < n`
set to `round(Σ_j hidden[r·d+j]·w_head[k·d+j] / 2^32)`. The loop walks `k < v`, computes one
`dot_round` per entry and writes it into the `alloc.vec.Vec` through `Vec.index_mut`. -/

theorem logitsVals_succ (hidden w_head : List ℤ) (r d : ℕ) (zs : List ℤ) (n : ℕ) :
    logitsVals hidden w_head r d zs (n + 1) =
      (logitsVals hidden w_head r d zs n).set n
        (rndDiv (dotAux hidden (r * d) w_head (n * d) 0 d) (2 ^ (32 : ℕ))) := rfl

theorem logitsVals_length (hidden w_head : List ℤ) (r d : ℕ) (zs : List ℤ) (n : ℕ) :
    (logitsVals hidden w_head r d zs n).length = zs.length := by
  induction n with
  | zero => rfl
  | succ k ih => rw [logitsVals_succ, List.length_set, ih]

theorem vvals_length (v : alloc.vec.Vec I64) : (vvals v).length = v.val.length := by
  simp [vvals]

/-- `Vec.index_mut` at a `usize` index: the returned setter is `Vec.set v i`. -/
theorem vec_index_mut_spec (v : alloc.vec.Vec I64) (i : Usize) (h : i.val < v.val.length) :
    alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice I64) v i =
      ok (v.val[i.val], v.set i) := by
  rw [alloc.vec.Vec.index_mut_slice_index]
  obtain ⟨p, hp, hp1, hp2⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_mut_usize_spec v i h)
  rw [hp, ← hp1, ← hp2]

/-- The `logits_row` loop invariant: `k` entries written, matched against `logitsVals`. -/
def logitsInv (hidden w_head : Slice I64) (r d v : Usize) (zs0 : List ℤ)
    (s : alloc.vec.Vec I64 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ v.val ∧
    vvals s.1 = logitsVals (hvals hidden) (hvals w_head) r.val d.val zs0 s.2.2.val

/-- The extracted `logits_row` body: one logit is written. -/
theorem logits_row_loop_body_spec (hidden w_head : Slice I64) (r d v : Usize) (zs0 : List ℤ)
    (hrd : r.val * d.val ≤ Usize.max)
    (hkd : ∀ kk < v.val, kk * d.val ≤ Usize.max)
    (hrmaxt : ∀ t < d.val, r.val * d.val + t ≤ Usize.max)
    (hkmaxt : ∀ kk < v.val, ∀ t < d.val, kk * d.val + t ≤ Usize.max)
    (hla : ∀ kk < v.val, ∀ t < d.val, r.val * d.val + t < hidden.val.length)
    (hlb : ∀ kk < v.val, ∀ t < d.val, kk * d.val + t < w_head.val.length)
    (hprod : ∀ kk < v.val, ∀ t < d.val,
      I128.min ≤ (hvals hidden).getD (r.val * d.val + t) 0 *
        (hvals w_head).getD (kk * d.val + t) 0 ∧
      (hvals hidden).getD (r.val * d.val + t) 0 *
        (hvals w_head).getD (kk * d.val + t) 0 ≤ I128.max)
    (hsum : ∀ kk < v.val, ∀ t ≤ d.val,
      I128.min ≤ dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 t ∧
      dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 t ≤ I128.max)
    (h2 : ∀ kk < v.val, 2 * absZ (Int.tmod
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ))) ≤ I128.max)
    (hrnd : ∀ kk < v.val, rndDivDom
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)))
    (hfit : ∀ kk < v.val, I64.min ≤ rndDiv
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)) ∧
      rndDiv (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)) ≤ I64.max)
    (hlen0 : v.val ≤ zs0.length)
    (st : alloc.vec.Vec I64 × Option learn_port.head.HeadError × Usize)
    (hinv : logitsInv hidden w_head r d v zs0 st) :
    learn_port.head.logits_row_loop.body hidden w_head r d v st.1 st.2.1 st.2.2 ⦃ r' =>
      match r' with
      | ControlFlow.done y =>
          vvals y.1 = logitsVals (hvals hidden) (hvals w_head) r.val d.val zs0 v.val ∧
            y.2 = none
      | ControlFlow.cont st' => logitsInv hidden w_head r d v zs0 st' ∧
          v.val - st'.2.2.val < v.val - st.2.2.val ⦄ := by
  obtain ⟨zs, fail, k⟩ := st
  simp only [logitsInv] at hinv
  obtain ⟨hfail, hle, hzs⟩ := hinv
  by_cases hlt : k < v
  · have hltv : k.val < v.val := (UScalar.lt_equiv k v).mp hlt
    obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := r) (y := d) hrd)
    obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := k) (y := d) (hkd k.val hltv))
    have hlen_zs : k.val < zs.val.length := by
      rw [← vvals_length, hzs, logitsVals_length]
      exact lt_of_lt_of_le hltv hlen0
    have hback := vec_index_mut_spec zs k hlen_zs
    obtain ⟨z, hz, hzv⟩ := dot_round_spec hidden i w_head i1 d
      (by intro t ht; rw [hiv]; exact hrmaxt t ht)
      (by intro t ht; rw [hi1v]; exact hkmaxt k.val hltv t ht)
      (by intro t ht; rw [hiv]; exact hla k.val hltv t ht)
      (by intro t ht; rw [hi1v]; exact hlb k.val hltv t ht)
      (by intro t ht; rw [hiv, hi1v]; exact hprod k.val hltv t ht)
      (by intro t ht; rw [hiv, hi1v]; exact hsum k.val hltv t ht)
      (by rw [hiv, hi1v]; exact h2 k.val hltv)
      (by rw [hiv, hi1v]; exact hrnd k.val hltv)
      (by rw [hiv, hi1v]; exact hfit k.val hltv)
    obtain ⟨i9, hi9, hi9v⟩ := usize_succ_fits hltv (by scalar_tac)
    have hzeq : z.val = rndDiv (dotAux (hvals hidden) (r.val * d.val)
        (hvals w_head) (k.val * d.val) 0 d.val) (2 ^ (32 : ℕ)) := by
      rw [hzv, hiv, hi1v]
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.logits_row_loop.body hidden w_head r d v zs fail k =
        ok (ControlFlow.cont (zs.set k z, none, i9)) := by
      rw [learn_port.head.logits_row_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, hi, hi1, hz, hback, hi9]
      rw [hfail]
      change (((ok (zs.set k z, none) : Result (alloc.vec.Vec I64 × Option learn_port.head.HeadError))
          >>= fun p => match p with
            | (zs1, fail2) => ok (ControlFlow.cont (zs1, fail2, i9))) =
          ok (ControlFlow.cont (zs.set k z, none, i9)))
      simp only [bind_tc_ok]
    rw [hbody]
    simp only [WP.spec_ok, logitsInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hi9v, vvals_set, hzs, logitsVals_succ, hzeq]
  · have hbody : learn_port.head.logits_row_loop.body hidden w_head r d v zs fail k =
        ok (ControlFlow.done (zs, fail)) := by
      rw [learn_port.head.logits_row_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hkeq : k.val = v.val := by
      have hvle : v.val ≤ k.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv k v).mpr (by omega))
      omega
    exact ⟨by rw [← hkeq, hzs], hfail⟩

/-- **The `logits_row` loop.** From the zero row the loop returns the whole logit vector. -/
theorem logits_row_loop_spec (hidden w_head : Slice I64) (r d v : Usize)
    (zs : alloc.vec.Vec I64) (fail : Option learn_port.head.HeadError) (k : Usize)
    (hrd : r.val * d.val ≤ Usize.max)
    (hkd : ∀ kk < v.val, kk * d.val ≤ Usize.max)
    (hrmaxt : ∀ t < d.val, r.val * d.val + t ≤ Usize.max)
    (hkmaxt : ∀ kk < v.val, ∀ t < d.val, kk * d.val + t ≤ Usize.max)
    (hla : ∀ kk < v.val, ∀ t < d.val, r.val * d.val + t < hidden.val.length)
    (hlb : ∀ kk < v.val, ∀ t < d.val, kk * d.val + t < w_head.val.length)
    (hprod : ∀ kk < v.val, ∀ t < d.val,
      I128.min ≤ (hvals hidden).getD (r.val * d.val + t) 0 *
        (hvals w_head).getD (kk * d.val + t) 0 ∧
      (hvals hidden).getD (r.val * d.val + t) 0 *
        (hvals w_head).getD (kk * d.val + t) 0 ≤ I128.max)
    (hsum : ∀ kk < v.val, ∀ t ≤ d.val,
      I128.min ≤ dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 t ∧
      dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 t ≤ I128.max)
    (h2 : ∀ kk < v.val, 2 * absZ (Int.tmod
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ))) ≤ I128.max)
    (hrnd : ∀ kk < v.val, rndDivDom
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)))
    (hfit : ∀ kk < v.val, I64.min ≤ rndDiv
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)) ∧
      rndDiv (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)) ≤ I64.max)
    (hlen0 : v.val ≤ (vvals zs).length)
    (hinv0 : logitsInv hidden w_head r d v (vvals zs) (zs, fail, k)) :
    WP.spec (learn_port.head.logits_row_loop hidden w_head r d v zs fail k)
      (fun y : alloc.vec.Vec I64 × Option learn_port.head.HeadError =>
        vvals y.1 = logitsVals (hvals hidden) (hvals w_head) r.val d.val (vvals zs) v.val ∧
          y.2 = none) := by
  unfold learn_port.head.logits_row_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Option learn_port.head.HeadError × Usize =>
      v.val - s.2.2.val)
    (inv := logitsInv hidden w_head r d v (vvals zs))
    (post := fun y : alloc.vec.Vec I64 × Option learn_port.head.HeadError =>
      vvals y.1 = logitsVals (hvals hidden) (hvals w_head) r.val d.val (vvals zs) v.val ∧
        y.2 = none)
  · intro st hst
    convert logits_row_loop_body_spec hidden w_head r d v (vvals zs) hrd hkd hrmaxt hkmaxt hla hlb
      hprod hsum h2 hrnd hfit hlen0 st hst using 1
    funext r'
    cases r' with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-- **`logits_row`.** The extracted body zero-fills `v` entries of a fresh `alloc.vec.Vec` and writes
one logit per entry, on the checked-`i64` dot/round domain. -/
theorem logits_row_spec (hidden w_head : Slice I64) (r d v : Usize)
    (hrd : r.val * d.val ≤ Usize.max)
    (hkd : ∀ kk < v.val, kk * d.val ≤ Usize.max)
    (hrmaxt : ∀ t < d.val, r.val * d.val + t ≤ Usize.max)
    (hkmaxt : ∀ kk < v.val, ∀ t < d.val, kk * d.val + t ≤ Usize.max)
    (hla : ∀ kk < v.val, ∀ t < d.val, r.val * d.val + t < hidden.val.length)
    (hlb : ∀ kk < v.val, ∀ t < d.val, kk * d.val + t < w_head.val.length)
    (hprod : ∀ kk < v.val, ∀ t < d.val,
      I128.min ≤ (hvals hidden).getD (r.val * d.val + t) 0 *
        (hvals w_head).getD (kk * d.val + t) 0 ∧
      (hvals hidden).getD (r.val * d.val + t) 0 *
        (hvals w_head).getD (kk * d.val + t) 0 ≤ I128.max)
    (hsum : ∀ kk < v.val, ∀ t ≤ d.val,
      I128.min ≤ dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 t ∧
      dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 t ≤ I128.max)
    (h2 : ∀ kk < v.val, 2 * absZ (Int.tmod
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ))) ≤ I128.max)
    (hrnd : ∀ kk < v.val, rndDivDom
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)))
    (hfit : ∀ kk < v.val, I64.min ≤ rndDiv
        (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)) ∧
      rndDiv (dotAux (hvals hidden) (r.val * d.val) (hvals w_head) (kk * d.val) 0 d.val)
        (2 ^ (32 : ℕ)) ≤ I64.max) :
    ∃ zs, learn_port.head.logits_row hidden w_head r d v = ok (core.result.Result.Ok zs) ∧
      vvals zs = logitsVals (hvals hidden) (hvals w_head) r.val d.val
        (List.replicate v.val 0) v.val := by
  unfold learn_port.head.logits_row
  obtain ⟨zs, hzs, hzsv, hzsl⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI64 0#i64 v (by rfl))
  have hvvals : vvals zs = List.replicate v.val 0 := by
    rw [vvals, hzsv]
    simp
  rw [hzs]
  simp only [bind_tc_ok]
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (logits_row_loop_spec hidden w_head r d v zs none 0#usize hrd hkd hrmaxt hkmaxt hla hlb
      hprod hsum h2 hrnd hfit
      (by rw [hvvals]; simp)
      (by refine ⟨rfl, by simp, rfl⟩))
  obtain ⟨zs1, fail1⟩ := y
  have hnone : fail1 = none := hy2
  refine ⟨zs1, ?_, ?_⟩
  · rw [hy]
    simp only [bind_tc_ok]
    rw [hnone]
    rfl
  · rw [hy1, hvvals]

/-! ## 14. `exp_fill`: the softmax numerator fill

`exp_entry` is `fixed_exp(z_i − m)` above the flush edge `−16·2^32` and `0` below it; `exp_fill`
writes `e[i] = exp_entry(z[i], m)` and accumulates `Σ e`. `FLUSH` is the extracted
`(-16) · (1 <<< 32)` threshold. -/

/-- The extracted `head::FLUSH` threshold evaluates to `-16·2^32`. -/
theorem head_FLUSH_eq : learn_port.head.FLUSH = ok ((-68719476736)#i128) := by
  have h : learn_port.head.FLUSH ⦃ z => z = ((-68719476736)#i128) ⦄ := by
    unfold learn_port.head.FLUSH
    step
    step
    apply IScalar.eq_imp
    rw [z_post, i_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

theorem flush_neg_val : ((-68719476736)#i128).val = (-68719476736 : ℤ) := rfl

/-- **`exp_entry`, flush arm.** A shift strictly below `−16·2^32` returns `Ok 0` exactly. -/
theorem exp_entry_flush (zi m : I64) (ui : I128)
    (hsub : core.convert.num.FromI128I64.from zi - core.convert.num.FromI128I64.from m = ok ui)
    (h : ui.val < (-68719476736 : ℤ)) :
    learn_port.head.exp_entry zi m = ok (core.result.Result.Ok 0#i64) := by
  unfold learn_port.head.exp_entry
  simp only [lift, bind_tc_ok]
  rw [hsub]
  simp only [bind_tc_ok]
  rw [head_FLUSH_eq]
  simp only [bind_tc_ok]
  rw [if_pos ((IScalar.lt_equiv ui _).2 (by rw [flush_neg_val]; exact h))]

/-- **`exp_entry`, `i64`-overflow arm.** A shift above `i64::MAX` is refused as `Overflow`. -/
theorem exp_entry_overflow (zi m : I64) (ui : I128)
    (hsub : core.convert.num.FromI128I64.from zi - core.convert.num.FromI128I64.from m = ok ui)
    (h : I64.max < ui.val) :
    learn_port.head.exp_entry zi m =
      ok (core.result.Result.Err learn_port.head.HeadError.Overflow) := by
  unfold learn_port.head.exp_entry
  simp only [lift, bind_tc_ok]
  rw [hsub]
  simp only [bind_tc_ok]
  rw [head_FLUSH_eq]
  simp only [bind_tc_ok]
  rw [if_neg (by
    intro hc
    have hc' := (IScalar.lt_equiv ui _).mp hc
    rw [flush_neg_val] at hc'
    rw [i64_max_eq] at h
    omega)]
  rw [if_pos ((IScalar.lt_equiv _ ui).2 (by rw [fromMAX_val]; exact h))]

/-- **`exp_entry`, engine-refusal arm.** A `fixed_exp` refusal is wrapped as `Exp`. -/
theorem exp_entry_exp_err (zi m : I64) (ui : I128) (ex : fixed_exp.ExpError)
    (hsub : core.convert.num.FromI128I64.from zi - core.convert.num.FromI128I64.from m = ok ui)
    (hflush : ¬ (ui.val < (-68719476736 : ℤ)))
    (hmax : ¬ (I64.max < ui.val))
    (he : fixed_exp.fixed_exp (IScalar.cast .I64 ui) = ok (core.result.Result.Err ex)) :
    learn_port.head.exp_entry zi m =
      ok (core.result.Result.Err (learn_port.head.HeadError.Exp ex)) := by
  unfold learn_port.head.exp_entry
  simp only [lift, bind_tc_ok]
  rw [hsub]
  simp only [bind_tc_ok]
  rw [head_FLUSH_eq]
  simp only [bind_tc_ok]
  rw [if_neg (by
    intro hc
    have hc' := (IScalar.lt_equiv ui _).mp hc
    rw [flush_neg_val] at hc'
    exact hflush hc')]
  rw [if_neg (not_gt_MAX ui (by omega))]
  rw [he]
  simp only [bind_tc_ok]

/-- **`exp_entry` success.** Above the flush edge and inside `i64`, the body returns the
`fixed_exp` value. -/
theorem exp_entry_ok (zi m : I64) (ui : I128) (ev : I64)
    (hsub : core.convert.num.FromI128I64.from zi - core.convert.num.FromI128I64.from m = ok ui)
    (hflush : ¬ (ui.val < (-68719476736 : ℤ)))
    (hmax : ¬ (I64.max < ui.val))
    (he : fixed_exp.fixed_exp (IScalar.cast .I64 ui) = ok (core.result.Result.Ok ev)) :
    learn_port.head.exp_entry zi m = ok (core.result.Result.Ok ev) := by
  unfold learn_port.head.exp_entry
  simp only [lift, bind_tc_ok]
  rw [hsub]
  simp only [bind_tc_ok]
  rw [head_FLUSH_eq]
  simp only [bind_tc_ok]
  rw [if_neg (by
    intro hc
    have hc' := (IScalar.lt_equiv ui _).mp hc
    rw [flush_neg_val] at hc'
    exact hflush hc')]
  rw [if_neg (not_gt_MAX ui (by omega))]
  rw [he]
  simp only [bind_tc_ok]

end apeiron_head
