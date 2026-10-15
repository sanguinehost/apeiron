import ApeironWord
import Learn.Grad.HeadRows
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted vocabulary head: the softmax numerator/probability fills and the row driver

`Learn/Grad/HeadRows.lean` refines `exp_entry` against the engine `fixed_exp` and the first
eleven of the thirteen `_loop` bodies of `learn_port.head`. This module refines the last two,
`exp_fill` and `fill_lp_p` (the softmax numerator and probability fills), and the row driver
`rows_loop` with the public entry points that compose them.

The callees `exp_entry`, `lp_entry` and `p_entry` enter as per-index hypotheses on the values
they return (the engine's refinement is `Numeric.Engine.Attn`'s); the model is the list of those
values, exactly as `log_softmax_row` is stated conditionally on `softmax_row`.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the container layout are not covered; only
the extracted integer bodies.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_head

open apeiron_word
open Aeneas Aeneas.Std Result

/-! ## 15. `exp_fill`: the weights `e_i = exp_entry(z_i, m)` and their running sum. -/

theorem expSum_succ (evs : List ℤ) (n : ℕ) (h : n < evs.length) :
    (evs.take (n + 1)).sum = (evs.take n).sum + evs.getD n 0 := by
  rw [List.sum_take_succ _ _ h, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

def expFillInv (z e : Slice I64) (evs : List ℤ) (sum : I128) (fail : Option learn_port.head.HeadError)
    (i : Usize) : Prop :=
  fail = none ∧ i.val ≤ z.val.length ∧ (hvals e).length = z.val.length ∧
    (∀ t < i.val, (hvals e).getD t 0 = evs.getD t 0) ∧ sum.val = (evs.take i.val).sum

/-- The extracted `exp_fill` body: one weight is written and added to the running sum. -/
theorem exp_fill_loop_body_spec (z e : Slice I64) (m : I64) (evs : List ℤ) (v : Usize)
    (hv : v.val = z.val.length) (hevs : evs.length = z.val.length)
    (hE : ∀ t < z.val.length, ∃ ev : I64,
        learn_port.head.exp_entry (z.val.getD t default) m = ok (core.result.Result.Ok ev) ∧
        ev.val = evs.getD t 0)
    (hsum : ∀ t ≤ z.val.length, I128.min ≤ (evs.take t).sum ∧ (evs.take t).sum ≤ I128.max)
    (sum : I128) (fail : Option learn_port.head.HeadError) (i : Usize)
    (hinv : expFillInv z e evs sum fail i) :
    learn_port.head.exp_fill_loop.body z m v e sum fail i ⦃ r =>
      match r with
      | ControlFlow.done y => hvals y.1 = evs ∧ y.2.1.val = evs.sum ∧ y.2.2 = none
      | ControlFlow.cont st' => expFillInv z st'.1 evs st'.2.1 st'.2.2.1 st'.2.2.2 ∧
          v.val - st'.2.2.2.val < v.val - i.val ⦄ := by
  obtain ⟨hfail, hle, hlen, hpre, hsv⟩ := hinv
  by_cases hlt : i < v
  · have hltv : i.val < z.val.length := by have := (UScalar.lt_equiv i v).mp hlt; omega
    have hltv' : i.val < v.val := (UScalar.lt_equiv i v).mp hlt
    obtain ⟨ev, hev, hevv⟩ := hE i.val hltv
    have hidx := slice_index_getD z i hltv
    have hilen : i.val < e.val.length := by rw [← hvals_length e]; omega
    obtain ⟨e1, hupd, he1⟩ := WP.spec_imp_exists (Slice.update_spec e i ev hilen)
    have hfit : I128.min ≤ sum.val + (core.convert.num.FromI128I64.from ev).val ∧
        sum.val + (core.convert.num.FromI128I64.from ev).val ≤ I128.max := by
      rw [core.convert.num.FromI128I64.from_val_eq, hsv, hevv,
        ← expSum_succ evs i.val (by omega)]
      exact hsum (i.val + 1) (by omega)
    obtain ⟨s1, hadd, hs1⟩ := add128_ok sum (core.convert.num.FromI128I64.from ev) hfit
    obtain ⟨i7, hi7, hi7v⟩ := usize_succ_fits hltv' (by scalar_tac)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.exp_fill_loop.body z m v e sum fail i =
        ok (ControlFlow.cont (e1, s1, none, i7)) := by
      rw [learn_port.head.exp_fill_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, lift, hidx, hev, hupd, hadd, hi7]
      rw [hfail]
      first | exact rfl | simp
    rw [hbody]
    simp only [WP.spec_ok, expFillInv]
    refine ⟨⟨trivial, by omega, ?_, ?_, ?_⟩, by omega⟩
    · rw [he1, hvals_set, List.length_set, hlen]
    · intro t ht
      rw [hi7v] at ht
      rw [he1, hvals_set]
      by_cases htk : t = i.val
      · rw [htk]
        have hlenlt : i.val < (hvals e).length := by rw [hlen]; omega
        rw [List.getD_eq_getElem?_getD, List.getElem?_set_self hlenlt, hevv]
        rfl
      · have hpt := hpre t (by omega)
        rw [List.getD_eq_getElem?_getD] at hpt ⊢
        rw [List.getElem?_set_ne (Ne.symm htk)]
        exact hpt
    · rw [hs1, core.convert.num.FromI128I64.from_val_eq, hsv, hevv, hi7v,
        ← expSum_succ evs i.val (by omega)]
  · have hbody : learn_port.head.exp_fill_loop.body z m v e sum fail i =
        ok (ControlFlow.done (e, sum, fail)) := by
      rw [learn_port.head.exp_fill_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hieq : i.val = v.val := by
      have hvle : v.val ≤ i.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv i v).mpr (by omega))
      omega
    refine ⟨?_, ?_, hfail⟩
    · refine List.ext_getElem (by rw [hlen, hevs]) (fun n h1 h2 => ?_)
      have h := hpre n (by omega)
      rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
        List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2] at h
      simpa using h
    · rw [hsv, hieq, hv, ← hevs, List.take_length]

/-- **The `exp_fill` loop.** From the zero sum the loop returns the whole weight vector and its sum. -/
theorem exp_fill_loop_spec (z e : Slice I64) (m : I64) (evs : List ℤ) (v : Usize)
    (hv : v.val = z.val.length) (hevs : evs.length = z.val.length)
    (hE : ∀ t < z.val.length, ∃ ev : I64,
        learn_port.head.exp_entry (z.val.getD t default) m = ok (core.result.Result.Ok ev) ∧
        ev.val = evs.getD t 0)
    (hsum : ∀ t ≤ z.val.length, I128.min ≤ (evs.take t).sum ∧ (evs.take t).sum ≤ I128.max)
    (hlen : (hvals e).length = z.val.length) :
    WP.spec (learn_port.head.exp_fill_loop z m e v 0#i128 none 0#usize)
      (fun y : Slice I64 × I128 × Option learn_port.head.HeadError =>
        hvals y.1 = evs ∧ y.2.1.val = evs.sum ∧ y.2.2 = none) := by
  unfold learn_port.head.exp_fill_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × I128 × Option learn_port.head.HeadError × Usize =>
      v.val - s.2.2.2.val)
    (inv := fun s => expFillInv z s.1 evs s.2.1 s.2.2.1 s.2.2.2)
    (post := fun y : Slice I64 × I128 × Option learn_port.head.HeadError =>
      hvals y.1 = evs ∧ y.2.1.val = evs.sum ∧ y.2.2 = none)
  · intro st hst
    obtain ⟨e', sum', fail', i'⟩ := st
    convert exp_fill_loop_body_spec z e' m evs v hv hevs hE hsum sum' fail' i' hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · refine ⟨rfl, by simp, hlen, by simp, by simp⟩

/-- **`exp_fill`.** Given the per-index `exp_entry` values `evs`, the function returns `Ok` of their
sum and the weight vector `evs`. -/
theorem exp_fill_spec (z e : Slice I64) (m : I64) (evs : List ℤ)
    (hevs : evs.length = z.val.length)
    (hE : ∀ t < z.val.length, ∃ ev : I64,
        learn_port.head.exp_entry (z.val.getD t default) m = ok (core.result.Result.Ok ev) ∧
        ev.val = evs.getD t 0)
    (hsum : ∀ t ≤ z.val.length, I128.min ≤ (evs.take t).sum ∧ (evs.take t).sum ≤ I128.max)
    (hlen : (hvals e).length = z.val.length) :
    ∃ s e1, learn_port.head.exp_fill z m e = ok (core.result.Result.Ok s, e1) ∧
      s.val = evs.sum ∧ hvals e1 = evs := by
  unfold learn_port.head.exp_fill
  obtain ⟨y, hy, hy1, hy2, hy3⟩ := WP.spec_imp_exists
    (exp_fill_loop_spec z e m evs (Slice.len z) (Slice.len_val z) hevs hE hsum hlen)
  obtain ⟨e1, s, f⟩ := y
  have hnone : f = none := hy3
  refine ⟨s, e1, ?_, hy2, hy1⟩
  show (learn_port.head.exp_fill_loop z m e (Slice.len z) 0#i128 none 0#usize >>= _) = _
  rw [hy, hnone]
  first | exact rfl | simp

/-! ## 16. `fill_lp_p`: the log-probability and probability fills

`lp[i] = lp_entry(z[i], m, l)` and `p[i] = p_entry(e[i], sum)`. The callees enter as per-index
hypotheses on the values they return; the model is the pair of lists `lps`, `ps` of those values.
A buffer `L` is `fillInv L L0 xs n` when it is the starting buffer `L0` with its first `n` entries
replaced by `xs`. -/

/-- `L` is `L0` with the first `n` entries replaced by `xs`. -/
def fillInv (L L0 xs : List ℤ) (n : ℕ) : Prop :=
  L.length = L0.length ∧ (∀ t < n, L.getD t 0 = xs.getD t 0) ∧ (∀ t, n ≤ t → L.getD t 0 = L0.getD t 0)

theorem fillInv_set (L L0 xs : List ℤ) (n : ℕ) (x : ℤ) (hn : n < L.length)
    (hx : x = xs.getD n 0) (h : fillInv L L0 xs n) : fillInv (L.set n x) L0 xs (n + 1) := by
  obtain ⟨hl, hlt, hge⟩ := h
  refine ⟨by rw [List.length_set, hl], ?_, ?_⟩
  · intro t ht
    by_cases htk : t = n
    · subst htk
      rw [List.getD_eq_getElem?_getD, List.getElem?_set_self hn]
      rw [hx]
      rfl
    · have hpt := hlt t (by omega)
      rw [List.getD_eq_getElem?_getD] at hpt ⊢
      rw [List.getElem?_set_ne (Ne.symm htk)]
      exact hpt
  · intro t ht
    have hpt := hge t (by omega)
    rw [List.getD_eq_getElem?_getD] at hpt ⊢
    rw [List.getElem?_set_ne (by omega)]
    exact hpt

/-- The `fill_lp_p` loop invariant. -/
def fillLpPInv (z lp p : Slice I64) (L0 P0 lps ps : List ℤ)
    (fail : Option learn_port.head.HeadError) (i : Usize) : Prop :=
  fail = none ∧ i.val ≤ z.val.length ∧ fillInv (hvals lp) L0 lps i.val ∧ fillInv (hvals p) P0 ps i.val

/-- The extracted `fill_lp_p` body: one `log p` and one `p` entry are written. -/
theorem fill_lp_p_loop_body_spec (z e lp p : Slice I64) (m l : I64) (sum : I128) (v : Usize)
    (L0 P0 lps ps : List ℤ)
    (hv : v.val = z.val.length) (hel : z.val.length ≤ e.val.length)
    (hLl : z.val.length ≤ L0.length) (hPl : z.val.length ≤ P0.length)
    (hLP : ∀ t < z.val.length, ∃ x : I64,
        learn_port.head.lp_entry (z.val.getD t default) m l = ok (core.result.Result.Ok x) ∧
        x.val = lps.getD t 0)
    (hP : ∀ t < z.val.length, ∃ x : I64,
        learn_port.head.p_entry (e.val.getD t default) sum = ok (core.result.Result.Ok x) ∧
        x.val = ps.getD t 0)
    (fail : Option learn_port.head.HeadError) (i : Usize)
    (hinv : fillLpPInv z lp p L0 P0 lps ps fail i) :
    learn_port.head.fill_lp_p_loop.body z m l e sum v lp p fail i ⦃ r =>
      match r with
      | ControlFlow.done y => y.1 = none ∧ fillInv (hvals y.2.1) L0 lps z.val.length ∧
          fillInv (hvals y.2.2) P0 ps z.val.length
      | ControlFlow.cont st' => fillLpPInv z st'.1 st'.2.1 L0 P0 lps ps st'.2.2.1 st'.2.2.2 ∧
          v.val - st'.2.2.2.val < v.val - i.val ⦄ := by
  obtain ⟨hfail, hle, hfl, hfp⟩ := hinv
  by_cases hlt : i < v
  · have hltz : i.val < z.val.length := by have := (UScalar.lt_equiv i v).mp hlt; omega
    have hltv' : i.val < v.val := (UScalar.lt_equiv i v).mp hlt
    obtain ⟨x, hx, hxv⟩ := hLP i.val hltz
    obtain ⟨y, hy, hyv⟩ := hP i.val hltz
    have hidx := slice_index_getD z i hltz
    have hidxe := slice_index_getD e i (by omega)
    have hlplen : i.val < lp.val.length := by rw [← hvals_length lp, hfl.1]; omega
    have hplen : i.val < p.val.length := by rw [← hvals_length p, hfp.1]; omega
    obtain ⟨lp1, hupd, hlp1⟩ := WP.spec_imp_exists (Slice.update_spec lp i x hlplen)
    obtain ⟨p1, hupdp, hp1⟩ := WP.spec_imp_exists (Slice.update_spec p i y hplen)
    obtain ⟨i7, hi7, hi7v⟩ := usize_succ_fits hltv' (by scalar_tac)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.fill_lp_p_loop.body z m l e sum v lp p fail i =
        ok (ControlFlow.cont (lp1, p1, none, i7)) := by
      rw [learn_port.head.fill_lp_p_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, hidx, hidxe, hx, hy, hupd, hupdp, hi7]
      rw [hfail]
      first | exact rfl | simp
    rw [hbody]
    simp only [WP.spec_ok, fillLpPInv]
    refine ⟨⟨trivial, by omega, ?_, ?_⟩, by omega⟩
    · rw [hlp1, hvals_set, hi7v]
      exact fillInv_set _ _ _ _ _ (by rw [hvals_length]; exact hlplen) hxv hfl
    · rw [hp1, hvals_set, hi7v]
      exact fillInv_set _ _ _ _ _ (by rw [hvals_length]; exact hplen) hyv hfp
  · have hbody : learn_port.head.fill_lp_p_loop.body z m l e sum v lp p fail i =
        ok (ControlFlow.done (fail, lp, p)) := by
      rw [learn_port.head.fill_lp_p_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hieq : i.val = z.val.length := by
      have hvle : v.val ≤ i.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv i v).mpr (by omega))
      omega
    rw [hieq] at hfl hfp
    exact ⟨hfail, hfl, hfp⟩

/-- **The `fill_lp_p` loop.** From the empty prefix the loop writes all `v` entries of both buffers. -/
theorem fill_lp_p_loop_spec (z e lp p : Slice I64) (m l : I64) (sum : I128) (v : Usize)
    (hv : v.val = z.val.length) (hel : z.val.length ≤ e.val.length)
    (hLl : z.val.length ≤ (hvals lp).length) (hPl : z.val.length ≤ (hvals p).length)
    (lps ps : List ℤ)
    (hLP : ∀ t < z.val.length, ∃ x : I64,
        learn_port.head.lp_entry (z.val.getD t default) m l = ok (core.result.Result.Ok x) ∧
        x.val = lps.getD t 0)
    (hP : ∀ t < z.val.length, ∃ x : I64,
        learn_port.head.p_entry (e.val.getD t default) sum = ok (core.result.Result.Ok x) ∧
        x.val = ps.getD t 0) :
    WP.spec (learn_port.head.fill_lp_p_loop z m l e sum lp p v none 0#usize)
      (fun y : Option learn_port.head.HeadError × Slice I64 × Slice I64 =>
        y.1 = none ∧ fillInv (hvals y.2.1) (hvals lp) lps z.val.length ∧
          fillInv (hvals y.2.2) (hvals p) ps z.val.length) := by
  unfold learn_port.head.fill_lp_p_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × Slice I64 × Option learn_port.head.HeadError × Usize =>
      v.val - s.2.2.2.val)
    (inv := fun s => fillLpPInv z s.1 s.2.1 (hvals lp) (hvals p) lps ps s.2.2.1 s.2.2.2)
    (post := fun y : Option learn_port.head.HeadError × Slice I64 × Slice I64 =>
      y.1 = none ∧ fillInv (hvals y.2.1) (hvals lp) lps z.val.length ∧
        fillInv (hvals y.2.2) (hvals p) ps z.val.length)
  · intro st hst
    obtain ⟨lp', p', fail', i'⟩ := st
    convert fill_lp_p_loop_body_spec z e lp' p' m l sum v (hvals lp) (hvals p) lps ps hv hel hLl hPl
      hLP hP fail' i' hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · refine ⟨rfl, by simp, ⟨rfl, by simp, by simp⟩, ⟨rfl, by simp, by simp⟩⟩

/-- **`fill_lp_p`.** Given the per-index `lp_entry` and `p_entry` values, the function returns no
failure and both buffers filled. -/
theorem fill_lp_p_spec (z e lp p : Slice I64) (m l : I64) (sum : I128) (lps ps : List ℤ)
    (hel : z.val.length ≤ e.val.length)
    (hLl : z.val.length ≤ (hvals lp).length) (hPl : z.val.length ≤ (hvals p).length)
    (hLP : ∀ t < z.val.length, ∃ x : I64,
        learn_port.head.lp_entry (z.val.getD t default) m l = ok (core.result.Result.Ok x) ∧
        x.val = lps.getD t 0)
    (hP : ∀ t < z.val.length, ∃ x : I64,
        learn_port.head.p_entry (e.val.getD t default) sum = ok (core.result.Result.Ok x) ∧
        x.val = ps.getD t 0) :
    ∃ lp1 p1, learn_port.head.fill_lp_p z m l e sum lp p = ok (none, lp1, p1) ∧
      fillInv (hvals lp1) (hvals lp) lps z.val.length ∧
        fillInv (hvals p1) (hvals p) ps z.val.length := by
  unfold learn_port.head.fill_lp_p
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (fill_lp_p_loop_spec z e lp p m l sum (Slice.len z) (Slice.len_val z) hel hLl hPl lps ps hLP hP)
  obtain ⟨f, lp1, p1⟩ := y
  have hnone : f = none := hy1
  refine ⟨lp1, p1, ?_, hy2⟩
  show learn_port.head.fill_lp_p_loop z m l e sum lp p (Slice.len z) none 0#usize = _
  rw [hy, hnone]

/-! ## 17. `rows_loop`: the row driver

`rows_loop` calls `row_step` once per row `r < n`, threading `(dh, dw, loss_acc)` and stopping at the
first refusal. The per-row callee enters as a contract: `Q r dh dw acc` is the state predicate after
`r` rows, and `hstep` says `row_step` from a state satisfying `Q r` returns `Ok` with a state satisfying
`Q (r + 1)`. The loop theorems derive `Q n` and the absence of a refusal from that contract by
`loop.spec_decr_nat`. -/

/-- The `rows_loop` invariant: no refusal yet, `r ≤ n` rows done, and the state predicate. -/
def rowsInv (Q : ℕ → Slice I64 → Slice I128 → I128 → Prop) (n : Usize)
    (dh : Slice I64) (dw : Slice I128) (acc : I128)
    (fail : Option learn_port.head.HeadError) (r : Usize) : Prop :=
  fail = none ∧ r.val ≤ n.val ∧ Q r.val dh dw acc

/-- The extracted `rows_loop` body: one `row_step` is run and its loss is accumulated. -/
theorem rows_loop_loop_body_spec (objective : learn_port.head.Objective)
    (hidden w_head z_t : Slice I64) (n i i1 : Usize) (n_scale : I128)
    (Q : ℕ → Slice I64 → Slice I128 → I128 → Prop)
    (hstep : ∀ r : Usize, r.val < n.val → ∀ (dh : Slice I64) (dw : Slice I128) (acc : I128),
      Q r.val dh dw acc → ∃ (a : I128) (dh1 : Slice I64) (dw1 : Slice I128),
        learn_port.head.row_step objective hidden w_head z_t { n, d := i, v := i1 } r n_scale acc dh dw =
          ok (core.result.Result.Ok a, dh1, dw1) ∧ Q (r.val + 1) dh1 dw1 a)
    (dh : Slice I64) (dw : Slice I128) (acc : I128) (fail : Option learn_port.head.HeadError)
    (r : Usize) (hinv : rowsInv Q n dh dw acc fail r) :
    learn_port.head.rows_loop_loop.body objective hidden w_head z_t n i i1 n_scale dh dw acc fail r ⦃ r' =>
      match r' with
      | ControlFlow.done y => Q n.val y.1 y.2.1 y.2.2.1 ∧ y.2.2.2 = none
      | ControlFlow.cont st' => rowsInv Q n st'.1 st'.2.1 st'.2.2.1 st'.2.2.2.1 st'.2.2.2.2 ∧
          n.val - st'.2.2.2.2.val < n.val - r.val ⦄ := by
  obtain ⟨hfail, hle, hQ⟩ := hinv
  by_cases hlt : r < n
  · have hltv : r.val < n.val := (UScalar.lt_equiv r n).mp hlt
    obtain ⟨a, dh1, dw1, hrs, hQ1⟩ := hstep r hltv dh dw acc hQ
    obtain ⟨r2, hr2, hr2v⟩ := usize_succ_fits hltv (by scalar_tac)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.rows_loop_loop.body objective hidden w_head z_t n i i1 n_scale dh dw
        acc fail r = ok (ControlFlow.cont (dh1, dw1, a, none, r2)) := by
      rw [learn_port.head.rows_loop_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, hrs, hr2]
      rw [hfail]
      first | exact rfl | simp
    rw [hbody]
    simp only [WP.spec_ok, rowsInv]
    refine ⟨⟨trivial, by omega, by rw [hr2v]; exact hQ1⟩, by omega⟩
  · have hbody : learn_port.head.rows_loop_loop.body objective hidden w_head z_t n i i1 n_scale dh dw
        acc fail r = ok (ControlFlow.done (dh, dw, acc, fail)) := by
      rw [learn_port.head.rows_loop_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hreq : r.val = n.val := by
      have hvle : n.val ≤ r.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv r n).mpr (by omega))
      omega
    rw [hreq] at hQ
    exact ⟨hQ, hfail⟩

/-- **The `rows_loop` loop.** From the state `Q 0` the loop returns the state `Q n` with no refusal. -/
theorem rows_loop_loop_spec (objective : learn_port.head.Objective)
    (hidden w_head z_t : Slice I64) (n i i1 : Usize) (n_scale : I128)
    (Q : ℕ → Slice I64 → Slice I128 → I128 → Prop)
    (hstep : ∀ r : Usize, r.val < n.val → ∀ (dh : Slice I64) (dw : Slice I128) (acc : I128),
      Q r.val dh dw acc → ∃ (a : I128) (dh1 : Slice I64) (dw1 : Slice I128),
        learn_port.head.row_step objective hidden w_head z_t { n, d := i, v := i1 } r n_scale acc dh dw =
          ok (core.result.Result.Ok a, dh1, dw1) ∧ Q (r.val + 1) dh1 dw1 a)
    (dh : Slice I64) (dw : Slice I128) (hQ0 : Q 0 dh dw 0#i128) :
    WP.spec (learn_port.head.rows_loop_loop objective hidden w_head z_t n i i1 n_scale dh dw 0#i128
        none 0#usize)
      (fun y : Slice I64 × Slice I128 × I128 × Option learn_port.head.HeadError =>
        Q n.val y.1 y.2.1 y.2.2.1 ∧ y.2.2.2 = none) := by
  unfold learn_port.head.rows_loop_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × Slice I128 × I128 × Option learn_port.head.HeadError × Usize =>
      n.val - s.2.2.2.2.val)
    (inv := fun s => rowsInv Q n s.1 s.2.1 s.2.2.1 s.2.2.2.1 s.2.2.2.2)
    (post := fun y : Slice I64 × Slice I128 × I128 × Option learn_port.head.HeadError =>
      Q n.val y.1 y.2.1 y.2.2.1 ∧ y.2.2.2 = none)
  · intro st hst
    obtain ⟨dh', dw', acc', fail', r'⟩ := st
    convert rows_loop_loop_body_spec objective hidden w_head z_t n i i1 n_scale Q hstep dh' dw' acc'
      fail' r' hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact ⟨rfl, by simp, hQ0⟩

/-- **`rows_loop`.** Given the per-row contract, the driver returns `Ok` of the accumulated loss
and the final `(dh, dw)` state. -/
theorem rows_loop_spec (objective : learn_port.head.Objective)
    (hidden w_head z_t : Slice I64) (dims : learn_port.head.Dims) (n_scale : I128)
    (Q : ℕ → Slice I64 → Slice I128 → I128 → Prop)
    (hstep : ∀ r : Usize, r.val < dims.n.val → ∀ (dh : Slice I64) (dw : Slice I128) (acc : I128),
      Q r.val dh dw acc → ∃ (a : I128) (dh1 : Slice I64) (dw1 : Slice I128),
        learn_port.head.row_step objective hidden w_head z_t
          { n := dims.n, d := dims.d, v := dims.v } r n_scale acc dh dw =
          ok (core.result.Result.Ok a, dh1, dw1) ∧ Q (r.val + 1) dh1 dw1 a)
    (dh : Slice I64) (dw : Slice I128) (hQ0 : Q 0 dh dw 0#i128) :
    ∃ acc dh1 dw1, learn_port.head.rows_loop objective hidden w_head z_t dims n_scale dh dw =
      ok (core.result.Result.Ok acc, dh1, dw1) ∧ Q dims.n.val dh1 dw1 acc := by
  unfold learn_port.head.rows_loop
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (rows_loop_loop_spec objective hidden w_head z_t dims.n dims.d dims.v n_scale Q hstep dh dw hQ0)
  obtain ⟨dh1, dw1, acc, f⟩ := y
  have hnone : f = none := hy2
  refine ⟨acc, dh1, dw1, ?_, hy1⟩
  rw [hy, hnone]
  first | exact rfl | simp

/-! ## 18. Function wrappers `kl_acc`, `dz_diff`, `tv_scan` -/

/-- `kl_acc` on the no-overflow domain returns the whole `klAccAux` sum. -/
theorem kl_acc_spec (p_t lp_t lp_s : Slice I64)
    (hlt : p_t.val.length ≤ lp_t.val.length) (hls : p_t.val.length ≤ lp_s.val.length)
    (hprod : ∀ t < p_t.val.length, I128.min ≤ klTerm (hvals p_t) (hvals lp_t) (hvals lp_s) t ∧
        klTerm (hvals p_t) (hvals lp_t) (hvals lp_s) t ≤ I128.max)
    (hsum : ∀ t ≤ p_t.val.length, I128.min ≤ klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 t ∧
        klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 t ≤ I128.max) :
    ∃ a, learn_port.head.kl_acc p_t lp_t lp_s = ok (core.result.Result.Ok a) ∧
      a.val = klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 p_t.val.length := by
  unfold learn_port.head.kl_acc
  have hv : (Slice.len p_t).val = p_t.val.length := Slice.len_val p_t
  dsimp only
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (kl_acc_loop_spec p_t lp_t lp_s (Slice.len p_t) 0#i128 none 0#usize
      (by rw [hv]; exact hlt) (by rw [hv]; exact hls) (by rw [hv])
      (fun t ht => hprod t (by rw [hv] at ht; exact ht))
      (fun t ht => hsum t (by rw [hv] at ht; exact ht))
      (by refine ⟨rfl, by simp, rfl⟩))
  obtain ⟨acc, fail⟩ := y
  have hnone : fail = none := hy2
  refine ⟨acc, ?_, by rw [hy1, hv]⟩
  rw [hy]
  simp only [bind_tc_ok]
  rw [hnone]
  exact rfl

/-- `dz_diff` on the no-overflow domain writes `p_s[i] − p_t[i]` into every entry. -/
theorem dz_diff_spec (p_s p_t dz : Slice I64)
    (htl : p_s.val.length ≤ p_t.val.length) (hdl : p_s.val.length ≤ dz.val.length)
    (hdiff : ∀ t < p_s.val.length, I64.min ≤ (hvals p_s).getD t 0 - (hvals p_t).getD t 0 ∧
        (hvals p_s).getD t 0 - (hvals p_t).getD t 0 ≤ I64.max) :
    ∃ y, learn_port.head.dz_diff p_s p_t dz = ok (none, y) ∧
      hvals y = dzVals (hvals p_s) (hvals p_t) (hvals dz) p_s.val.length := by
  unfold learn_port.head.dz_diff
  have hv : (Slice.len p_s).val = p_s.val.length := Slice.len_val p_s
  dsimp only
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (dz_diff_loop_spec p_s p_t dz (Slice.len p_s) none 0#usize
      (by rw [hv]) (by rw [hv]; exact htl) (by rw [hv]; exact hdl)
      (fun t ht => hdiff t (by rw [hv] at ht; exact ht))
      (by refine ⟨rfl, by simp, rfl⟩))
  obtain ⟨o, s⟩ := y
  have hnone : o = none := hy2
  refine ⟨s, ?_, by rw [hy1, hv]⟩
  rw [hy, hnone]

/-- `tv_scan` on the no-overflow domain returns the absolute and signed sums and the sign vector. -/
theorem tv_scan_spec (p_s p_t : Slice I64) (sg : Slice I128)
    (htl : p_s.val.length ≤ p_t.val.length) (hsgl : p_s.val.length ≤ sg.val.length)
    (habsfit : ∀ t ≤ p_s.val.length,
      I128.min ≤ tvAbsAux (hvals p_s) (hvals p_t) 0 t ∧
      tvAbsAux (hvals p_s) (hvals p_t) 0 t ≤ I128.max)
    (hsignedfit : ∀ t ≤ p_s.val.length,
      I128.min ≤ tvSignedAux (hvals p_s) (hvals p_t) 0 t ∧
      tvSignedAux (hvals p_s) (hvals p_t) 0 t ≤ I128.max)
    (hprod : ∀ t < p_s.val.length, I128.min ≤
        (hvals p_s).getD t 0 * sgnZ ((hvals p_s).getD t 0 - (hvals p_t).getD t 0) ∧
      (hvals p_s).getD t 0 * sgnZ ((hvals p_s).getD t 0 - (hvals p_t).getD t 0) ≤ I128.max) :
    ∃ a b sg1, learn_port.head.tv_scan p_s p_t sg = ok (core.result.Result.Ok (a, b), sg1) ∧
      svals sg1 = sgVals (hvals p_s) (hvals p_t) (svals sg) p_s.val.length ∧
      a.val = tvAbsAux (hvals p_s) (hvals p_t) 0 p_s.val.length ∧
      b.val = tvSignedAux (hvals p_s) (hvals p_t) 0 p_s.val.length := by
  unfold learn_port.head.tv_scan
  have hv : (Slice.len p_s).val = p_s.val.length := Slice.len_val p_s
  have hz0 : (0#i128).val = 0 := rfl
  dsimp only
  obtain ⟨y, hy, hy1, hy2, hy3, hy4⟩ := WP.spec_imp_exists
    (tv_scan_loop_spec p_s p_t sg (Slice.len p_s) 0#i128 0#i128 none 0#usize
      (by rw [hv]) (by rw [hv]; exact htl) (by rw [hv]; exact hsgl)
      (fun t ht => hz0 ▸ habsfit t (by rw [hv] at ht; exact ht))
      (fun t ht => hz0 ▸ hsignedfit t (by rw [hv] at ht; exact ht))
      (fun t ht => hprod t (by rw [hv] at ht; exact ht))
      (by refine ⟨rfl, by simp, rfl, rfl, rfl⟩))
  obtain ⟨sg1, a, b, fail⟩ := y
  have hnone : fail = none := hy4
  refine ⟨a, b, sg1, ?_, by rw [hy1, hv], by rw [hy2, hv], by rw [hy3, hv]⟩
  rw [hy]
  simp only [bind_tc_ok]
  rw [hnone]
  exact rfl

/-! ## 19. The row losses `kl_row` and `tv_row` (public)

`kl_row` and `tv_row` refuse unequal lengths with `Shape` unconditionally. Their success arms are
stated conditionally on the two `softmax_row` results (as `log_softmax_row` is) and on the
no-overflow domain of the callees; the loss is the `i128` sum divided by `2^32` (`engine.ONE`,
KL) or by `2` (TV) with round-half-away, narrowed once to `i64`. -/

/-- A length mismatch is refused with `Shape` before any softmax runs. -/
theorem kl_row_shape (z_s z_t : Slice I64) (h : z_s.val.length ≠ z_t.val.length) :
    learn_port.head.kl_row z_s z_t = ok (core.result.Result.Err learn_port.head.HeadError.Shape) := by
  unfold learn_port.head.kl_row
  have hne : (Slice.len z_s != Slice.len z_t) = true := by
    simp only [bne_iff_ne, ne_eq]
    intro he
    apply h
    have := congrArg (fun x : Usize => x.val) he
    simpa [Slice.len_val] using this
  dsimp only
  simp only [hne, ↓reduceIte]

theorem tv_row_shape (z_s z_t : Slice I64) (h : z_s.val.length ≠ z_t.val.length) :
    learn_port.head.tv_row z_s z_t = ok (core.result.Result.Err learn_port.head.HeadError.Shape) := by
  unfold learn_port.head.tv_row
  have hne : (Slice.len z_s != Slice.len z_t) = true := by
    simp only [bne_iff_ne, ne_eq]
    intro he
    apply h
    have := congrArg (fun x : Usize => x.val) he
    simpa [Slice.len_val] using this
  dsimp only
  simp only [hne, ↓reduceIte]

theorem kl_row_spec (z_s z_t : Slice I64) (lp_s p_s lp_t p_t : alloc.vec.Vec I64)
    (hlen : z_s.val.length = z_t.val.length)
    (hs : learn_port.head.softmax_row z_s = ok (core.result.Result.Ok (lp_s, p_s)))
    (ht : learn_port.head.softmax_row z_t = ok (core.result.Result.Ok (lp_t, p_t)))
    (hlps : lp_s.val.length = z_s.val.length) (hps : p_s.val.length = z_s.val.length)
    (hlpt : lp_t.val.length = z_s.val.length) (hpt : p_t.val.length = z_s.val.length)
    (hprod : ∀ t < z_s.val.length,
      I128.min ≤ klTerm (hvals (alloc.vec.Vec.deref p_t)) (hvals (alloc.vec.Vec.deref lp_t))
        (hvals (alloc.vec.Vec.deref lp_s)) t ∧
      klTerm (hvals (alloc.vec.Vec.deref p_t)) (hvals (alloc.vec.Vec.deref lp_t))
        (hvals (alloc.vec.Vec.deref lp_s)) t ≤ I128.max)
    (hsum : ∀ t ≤ z_s.val.length,
      I128.min ≤ klAccAux (hvals (alloc.vec.Vec.deref p_t)) (hvals (alloc.vec.Vec.deref lp_t))
        (hvals (alloc.vec.Vec.deref lp_s)) 0 t ∧
      klAccAux (hvals (alloc.vec.Vec.deref p_t)) (hvals (alloc.vec.Vec.deref lp_t))
        (hvals (alloc.vec.Vec.deref lp_s)) 0 t ≤ I128.max)
    (hdiff : ∀ t < z_s.val.length,
      I64.min ≤ (hvals (alloc.vec.Vec.deref p_s)).getD t 0 -
        (hvals (alloc.vec.Vec.deref p_t)).getD t 0 ∧
      (hvals (alloc.vec.Vec.deref p_s)).getD t 0 -
        (hvals (alloc.vec.Vec.deref p_t)).getD t 0 ≤ I64.max)
    (h2 : 2 * absZ (Int.tmod (klAccAux (hvals (alloc.vec.Vec.deref p_t))
        (hvals (alloc.vec.Vec.deref lp_t)) (hvals (alloc.vec.Vec.deref lp_s)) 0 z_s.val.length)
        (2 ^ (32 : ℕ))) ≤ I128.max)
    (hrnd : rndDivDom (klAccAux (hvals (alloc.vec.Vec.deref p_t))
        (hvals (alloc.vec.Vec.deref lp_t)) (hvals (alloc.vec.Vec.deref lp_s)) 0 z_s.val.length)
        (2 ^ (32 : ℕ)))
    (hfit : I64.min ≤ rndDiv (klAccAux (hvals (alloc.vec.Vec.deref p_t))
        (hvals (alloc.vec.Vec.deref lp_t)) (hvals (alloc.vec.Vec.deref lp_s)) 0 z_s.val.length)
        (2 ^ (32 : ℕ)) ∧
      rndDiv (klAccAux (hvals (alloc.vec.Vec.deref p_t))
        (hvals (alloc.vec.Vec.deref lp_t)) (hvals (alloc.vec.Vec.deref lp_s)) 0 z_s.val.length)
        (2 ^ (32 : ℕ)) ≤ I64.max) :
    ∃ loss dz, learn_port.head.kl_row z_s z_t = ok (core.result.Result.Ok (loss, dz)) ∧
      loss.val = rndDiv (klAccAux (hvals (alloc.vec.Vec.deref p_t))
        (hvals (alloc.vec.Vec.deref lp_t)) (hvals (alloc.vec.Vec.deref lp_s)) 0 z_s.val.length)
        (2 ^ (32 : ℕ)) ∧
      vvals dz = dzVals (hvals (alloc.vec.Vec.deref p_s)) (hvals (alloc.vec.Vec.deref p_t))
        (List.replicate z_s.val.length 0) z_s.val.length := by
  unfold learn_port.head.kl_row
  have hne : (Slice.len z_s != Slice.len z_t) = false := by
    simp only [bne_eq_false_iff_eq]
    apply UScalar.eq_of_val_eq
    simpa [Slice.len_val] using hlen
  dsimp only
  simp only [hne, Bool.false_eq_true, ↓reduceIte]
  rw [hs]
  simp only [bind_tc_ok]
  rw [ht]
  simp only [bind_tc_ok]
  have hder : ∀ v : alloc.vec.Vec I64, (alloc.vec.Vec.deref v).val = v.val := fun v => by
    unfold alloc.vec.Vec.deref; exact Slice.from_val _ _
  have hZ : (Slice.len z_s).val = z_s.val.length := Slice.len_val z_s
  obtain ⟨a, ha, hav⟩ := kl_acc_spec (alloc.vec.Vec.deref p_t) (alloc.vec.Vec.deref lp_t)
    (alloc.vec.Vec.deref lp_s)
    (by rw [hder, hder]; omega) (by rw [hder, hder]; omega)
    (fun t ht => hprod t (by rw [hder, hpt] at ht; exact ht))
    (fun t ht => hsum t (by rw [hder, hpt] at ht; exact ht))
  rw [hder, hpt] at hav
  obtain ⟨dz, hdz, hdzv, hdzl⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI64 0#i64 (Slice.len z_s) (by rfl))
  have hvv : hvals (alloc.vec.Vec.deref_mut dz).1 = vvals dz := rfl
  have hdzs : hvals (alloc.vec.Vec.deref_mut dz).1 = List.replicate z_s.val.length 0 := by
    rw [hvv, vvals, hdzv, hZ]; simp
  have hdzl' : (alloc.vec.Vec.deref_mut dz).1.val.length = z_s.val.length := by
    have h := congrArg List.length hdzs.symm
    rw [hvals_length] at h
    simpa using h.symm
  obtain ⟨y, hy, hyv⟩ := dz_diff_spec (alloc.vec.Vec.deref p_s) (alloc.vec.Vec.deref p_t)
    (alloc.vec.Vec.deref_mut dz).1
    (by rw [hder, hder]; omega)
    (by rw [hder, hdzl']; omega)
    (fun t ht => hdiff t (by rw [hder, hps] at ht; exact ht))
  have hone : (core.convert.num.FromI128I64.from (4294967296#i64)).val = 2 ^ (32 : ℕ) := by
    rw [core.convert.num.FromI128I64.from_val_eq]; norm_num
  obtain ⟨i2, hi2, hi2v⟩ :=
    round_div_dom_spec a (core.convert.num.FromI128I64.from (4294967296#i64))
      (by rw [hone]; norm_num)
      (by rw [hav, hone]; exact h2)
      (by rw [hav, hone]; exact hrnd)
  obtain ⟨z, hz, hzv⟩ := to64_ok i2 (by rw [hi2v, hav, hone]; exact hfit)
  refine ⟨z, alloc.vec.Vec.ofSlice y, ?_, ?_, ?_⟩
  · change (learn_port.head.kl_acc (alloc.vec.Vec.deref p_t) (alloc.vec.Vec.deref lp_t)
      (alloc.vec.Vec.deref lp_s) >>= _) = _
    rw [ha]
    simp only [bind_tc_ok]
    rw [hdz]
    simp only [lift, bind_tc_ok]
    change (learn_port.head.dz_diff (alloc.vec.Vec.deref p_s) (alloc.vec.Vec.deref p_t)
      (alloc.vec.Vec.deref_mut dz).1 >>= _) = _
    rw [hy]
    simp only [bind_tc_ok]
    rw [engine_ONE_eq]
    simp only [bind_tc_ok]
    rw [hi2]
    simp only [bind_tc_ok]
    rw [hz]
    first | exact rfl | simp
  · rw [hzv, hi2v, hav, hone]
  · have h1 : vvals (alloc.vec.Vec.ofSlice y) = hvals y := rfl
    rw [h1, hyv, hdzs, hder, hps]

/-- **`tv_row`, success arm.** The loss is `round(Σ|p_s − p_t| / 2)`; the gradient is `tvDzVals`
over the sign vector `S` and the signed sum `sgn`. -/
theorem tv_row_spec (z_s z_t : Slice I64) (lp_s p_s lp_t p_t : alloc.vec.Vec I64)
    (S : List ℤ) (sgn ab : ℤ)
    (hS : S = sgVals (hvals (alloc.vec.Vec.deref p_s)) (hvals (alloc.vec.Vec.deref p_t))
      (List.replicate z_s.val.length 0) z_s.val.length)
    (hsgn : sgn = tvSignedAux (hvals (alloc.vec.Vec.deref p_s)) (hvals (alloc.vec.Vec.deref p_t))
      0 z_s.val.length)
    (hab : ab = tvAbsAux (hvals (alloc.vec.Vec.deref p_s)) (hvals (alloc.vec.Vec.deref p_t))
      0 z_s.val.length)
    (hlen : z_s.val.length = z_t.val.length)
    (hs : learn_port.head.softmax_row z_s = ok (core.result.Result.Ok (lp_s, p_s)))
    (ht : learn_port.head.softmax_row z_t = ok (core.result.Result.Ok (lp_t, p_t)))
    (hps : p_s.val.length = z_s.val.length) (hpt : p_t.val.length = z_s.val.length)
    (habsfit : ∀ t ≤ z_s.val.length,
      I128.min ≤ tvAbsAux (hvals (alloc.vec.Vec.deref p_s)) (hvals (alloc.vec.Vec.deref p_t)) 0 t ∧
      tvAbsAux (hvals (alloc.vec.Vec.deref p_s)) (hvals (alloc.vec.Vec.deref p_t)) 0 t ≤ I128.max)
    (hsignedfit : ∀ t ≤ z_s.val.length,
      I128.min ≤ tvSignedAux (hvals (alloc.vec.Vec.deref p_s))
        (hvals (alloc.vec.Vec.deref p_t)) 0 t ∧
      tvSignedAux (hvals (alloc.vec.Vec.deref p_s)) (hvals (alloc.vec.Vec.deref p_t)) 0 t ≤
        I128.max)
    (hprod : ∀ t < z_s.val.length, I128.min ≤
        (hvals (alloc.vec.Vec.deref p_s)).getD t 0 *
          sgnZ ((hvals (alloc.vec.Vec.deref p_s)).getD t 0 -
            (hvals (alloc.vec.Vec.deref p_t)).getD t 0) ∧
      (hvals (alloc.vec.Vec.deref p_s)).getD t 0 *
          sgnZ ((hvals (alloc.vec.Vec.deref p_s)).getD t 0 -
            (hvals (alloc.vec.Vec.deref p_t)).getD t 0) ≤ I128.max)
    (hg1 : ∀ t < z_s.val.length, I128.min ≤ S.getD t 0 * 2 ^ (32 : ℕ) ∧
      S.getD t 0 * 2 ^ (32 : ℕ) ≤ I128.max)
    (hs1 : ∀ t < z_s.val.length, I128.min ≤ S.getD t 0 * 2 ^ (32 : ℕ) - sgn ∧
      S.getD t 0 * 2 ^ (32 : ℕ) - sgn ≤ I128.max)
    (hn : ∀ t < z_s.val.length,
      I128.min ≤ tvDzNum (hvals (alloc.vec.Vec.deref p_s)) S sgn t ∧
      tvDzNum (hvals (alloc.vec.Vec.deref p_s)) S sgn t ≤ I128.max)
    (h3 : I128.min ≤ 2 * 2 ^ (32 : ℕ) ∧ 2 * 2 ^ (32 : ℕ) ≤ I128.max)
    (hrnd : ∀ t < z_s.val.length,
      rndDivDom (tvDzNum (hvals (alloc.vec.Vec.deref p_s)) S sgn t) (2 * 2 ^ (32 : ℕ)) ∧
      2 * absZ (Int.tmod (tvDzNum (hvals (alloc.vec.Vec.deref p_s)) S sgn t)
        (2 * 2 ^ (32 : ℕ))) ≤ I128.max)
    (hfit : ∀ t < z_s.val.length,
      I64.min ≤ rndDiv (tvDzNum (hvals (alloc.vec.Vec.deref p_s)) S sgn t) (2 * 2 ^ (32 : ℕ)) ∧
      rndDiv (tvDzNum (hvals (alloc.vec.Vec.deref p_s)) S sgn t) (2 * 2 ^ (32 : ℕ)) ≤ I64.max)
    (h2 : 2 * absZ (Int.tmod ab 2) ≤ I128.max) (hrndab : rndDivDom ab 2)
    (hfitab : I64.min ≤ rndDiv ab 2 ∧ rndDiv ab 2 ≤ I64.max) :
    ∃ loss dz, learn_port.head.tv_row z_s z_t = ok (core.result.Result.Ok (loss, dz)) ∧
      loss.val = rndDiv ab 2 ∧
      vvals dz = tvDzVals (hvals (alloc.vec.Vec.deref p_s)) S sgn
        (List.replicate z_s.val.length 0) z_s.val.length := by
  unfold learn_port.head.tv_row
  have hne : (Slice.len z_s != Slice.len z_t) = false := by
    simp only [bne_eq_false_iff_eq]
    apply UScalar.eq_of_val_eq
    simpa [Slice.len_val] using hlen
  dsimp only
  simp only [hne, Bool.false_eq_true, ↓reduceIte]
  rw [hs]
  simp only [bind_tc_ok]
  rw [ht]
  simp only [bind_tc_ok]
  have hder : ∀ v : alloc.vec.Vec I64, (alloc.vec.Vec.deref v).val = v.val := fun v => by
    unfold alloc.vec.Vec.deref; exact Slice.from_val _ _
  have hZ : (Slice.len z_s).val = z_s.val.length := Slice.len_val z_s
  -- the sign vector buffer
  obtain ⟨sg, hsg, hsgv, hsgl⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI128 0#i128 (Slice.len z_s) (by rfl))
  have hsgs : svals (alloc.vec.Vec.deref_mut sg).1 = List.replicate z_s.val.length 0 := by
    have : svals (alloc.vec.Vec.deref_mut sg).1 = sg.val.map (fun z => z.val) := rfl
    rw [this, hsgv, hZ]; simp
  have hsgl' : (alloc.vec.Vec.deref_mut sg).1.val.length = z_s.val.length := by
    have h := congrArg List.length hsgs
    rw [svals_length] at h
    simpa using h
  obtain ⟨a, b, sg1, hscan, hsg1, hav, hbv⟩ := tv_scan_spec (alloc.vec.Vec.deref p_s)
    (alloc.vec.Vec.deref p_t) (alloc.vec.Vec.deref_mut sg).1
    (by rw [hder, hder]; omega) (by rw [hder, hsgl']; omega)
    (fun t ht => habsfit t (by rw [hder, hps] at ht; exact ht))
    (fun t ht => hsignedfit t (by rw [hder, hps] at ht; exact ht))
    (fun t ht => hprod t (by rw [hder, hps] at ht; exact ht))
  have hlenS : (alloc.vec.Vec.deref p_s).val.length = z_s.val.length := by rw [hder, hps]
  rw [hlenS, hsgs, ← hS] at hsg1
  rw [hlenS, ← hsgn] at hbv
  rw [hlenS, ← hab] at hav
  -- the gradient buffer
  obtain ⟨dz, hdz, hdzv, hdzl⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI64 0#i64 (Slice.len z_s) (by rfl))
  have hvv : hvals (alloc.vec.Vec.deref_mut dz).1 = vvals dz := rfl
  have hdzs : hvals (alloc.vec.Vec.deref_mut dz).1 = List.replicate z_s.val.length 0 := by
    rw [hvv, vvals, hdzv, hZ]; simp
  have hdzl' : (alloc.vec.Vec.deref_mut dz).1.val.length = z_s.val.length := by
    have h := congrArg List.length hdzs.symm
    rw [hvals_length] at h
    simpa using h.symm
  have hderS : ∀ s : Slice I128, (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice s)).val = s.val :=
    fun s => by unfold alloc.vec.Vec.deref; rw [Slice.from_val]; rfl
  have hsvS : svals (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice sg1)) = S := by
    rw [svals, hderS]; exact hsg1
  have hSl : (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice sg1)).val.length = z_s.val.length := by
    have h := congrArg List.length hsvS
    rw [svals_length, hS, sgVals_length] at h
    simpa using h
  obtain ⟨y, hy, hyv⟩ := tv_dz_spec (alloc.vec.Vec.deref p_s)
    (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice sg1)) b (alloc.vec.Vec.deref_mut dz).1
    (fun t ht => by rw [hlenS] at ht; rw [hSl]; exact ht)
    (fun t ht => by rw [hlenS] at ht; rw [hdzl']; exact ht)
    (fun t ht => by rw [hlenS] at ht; rw [hsvS]; simpa using hg1 t ht)
    (fun t ht => by rw [hlenS] at ht; rw [hsvS, hbv]; simpa using hs1 t ht)
    (fun t ht => by rw [hlenS] at ht; rw [hsvS, hbv]; exact hn t ht)
    h3
    (fun t ht => by rw [hlenS] at ht; rw [hsvS, hbv]; exact hrnd t ht)
    (fun t ht => by rw [hlenS] at ht; rw [hsvS, hbv]; exact hfit t ht)
  have h2v : (2#i128).val = 2 := rfl
  obtain ⟨i2, hi2, hi2v⟩ := round_div_dom_spec a 2#i128 (by rw [h2v]; norm_num)
    (by rw [hav]; exact h2) (by rw [hav]; exact hrndab)
  obtain ⟨z, hz, hzv⟩ := to64_ok i2 (by rw [hi2v, hav]; exact hfitab)
  refine ⟨z, alloc.vec.Vec.ofSlice y, ?_, ?_, ?_⟩
  · change (alloc.vec.from_elem core.clone.CloneI128 0#i128 (Slice.len z_s) >>= _) = _
    rw [hsg]
    simp only [lift, bind_tc_ok]
    change (learn_port.head.tv_scan (alloc.vec.Vec.deref p_s) (alloc.vec.Vec.deref p_t)
      (alloc.vec.Vec.deref_mut sg).1 >>= _) = _
    rw [hscan]
    simp only [bind_tc_ok]
    change (alloc.vec.from_elem core.clone.CloneI64 0#i64 (Slice.len z_s) >>= _) = _
    rw [hdz]
    simp only [bind_tc_ok]
    change (learn_port.head.tv_dz (alloc.vec.Vec.deref p_s)
      (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice sg1)) b (alloc.vec.Vec.deref_mut dz).1 >>= _)
      = _
    rw [hy]
    simp only [bind_tc_ok]
    rw [hi2]
    simp only [bind_tc_ok]
    rw [hz]
    first | exact rfl | simp
  · rw [hzv, hi2v, hav, h2v]
  · have h1 : vvals (alloc.vec.Vec.ofSlice y) = hvals y := rfl
    rw [h1, hyv, hsvS, hbv, hdzs, hlenS]

/-! ## 20. `dw_out_fill` and the public `head_chunked`

`head_chunked` refuses a zero dimension or chunk with `Shape` before any buffer is allocated
(unconditional theorems below). Its success arm is stated against the row loop's result: the
`rows_loop` call is the hypothesis `hrows` (discharged by `rows_loop_spec` for a state predicate
`Q`), and the loss, `dh` and `dw` are the rounded accumulator, the row-loop `dh` and the
`dwOutVals` rounding of the row-loop `dw`. -/

/-- `dw_out_fill` on its rounding domain writes `round(dw[t] / n_scale)` into every entry. -/
theorem dw_out_fill_spec (dw : Slice I128) (nscale : I128) (out : Slice I64)
    (hol : dw.val.length ≤ out.val.length) (hpos : 0 < nscale.val)
    (hrnd : ∀ t < dw.val.length, rndDivDom ((svals dw).getD t 0) nscale.val ∧
        2 * absZ (Int.tmod ((svals dw).getD t 0) nscale.val) ≤ I128.max)
    (hfit : ∀ t < dw.val.length, I64.min ≤ rndDiv ((svals dw).getD t 0) nscale.val ∧
        rndDiv ((svals dw).getD t 0) nscale.val ≤ I64.max) :
    ∃ y, learn_port.head.dw_out_fill dw nscale out = ok (none, y) ∧
      hvals y = dwOutVals (svals dw) nscale.val (hvals out) dw.val.length := by
  unfold learn_port.head.dw_out_fill
  have hv : (Slice.len dw).val = dw.val.length := Slice.len_val dw
  dsimp only
  obtain ⟨y, hy, hy1, hy2⟩ := WP.spec_imp_exists
    (dw_out_fill_loop_spec dw nscale out (Slice.len dw) none 0#usize
      (by rw [hv]) (by rw [hv]; exact hol) hpos
      (fun t ht => hrnd t (by rw [hv] at ht; exact ht))
      (fun t ht => hfit t (by rw [hv] at ht; exact ht))
      (by refine ⟨rfl, by simp, rfl⟩))
  obtain ⟨o, s⟩ := y
  have hnone : o = none := hy2
  refine ⟨s, ?_, by rw [hy1, hv]⟩
  rw [hy, hnone]

/-- A zero `n` is refused with `Shape`. -/
theorem head_chunked_shape_n (hidden w_head z_t : Slice I64) (dims : learn_port.head.Dims)
    (chunk : Usize) (objective : learn_port.head.Objective) (h : dims.n.val = 0) :
    learn_port.head.head_chunked hidden w_head z_t dims chunk objective =
      ok (core.result.Result.Err learn_port.head.HeadError.Shape) := by
  unfold learn_port.head.head_chunked
  have : dims.n = 0#usize := UScalar.eq_of_val_eq (by simpa using h)
  simp [this]

/-- A zero `d` is refused with `Shape`. -/
theorem head_chunked_shape_d (hidden w_head z_t : Slice I64) (dims : learn_port.head.Dims)
    (chunk : Usize) (objective : learn_port.head.Objective) (hn : dims.n.val ≠ 0)
    (h : dims.d.val = 0) :
    learn_port.head.head_chunked hidden w_head z_t dims chunk objective =
      ok (core.result.Result.Err learn_port.head.HeadError.Shape) := by
  unfold learn_port.head.head_chunked
  have hn' : ¬ dims.n = 0#usize := fun e => hn (by simpa using congrArg (fun x : Usize => x.val) e)
  have : dims.d = 0#usize := UScalar.eq_of_val_eq (by simpa using h)
  simp [hn', this]

/-- A zero `v` is refused with `Shape`. -/
theorem head_chunked_shape_v (hidden w_head z_t : Slice I64) (dims : learn_port.head.Dims)
    (chunk : Usize) (objective : learn_port.head.Objective) (hn : dims.n.val ≠ 0)
    (hd : dims.d.val ≠ 0) (h : dims.v.val = 0) :
    learn_port.head.head_chunked hidden w_head z_t dims chunk objective =
      ok (core.result.Result.Err learn_port.head.HeadError.Shape) := by
  unfold learn_port.head.head_chunked
  have hn' : ¬ dims.n = 0#usize := fun e => hn (by simpa using congrArg (fun x : Usize => x.val) e)
  have hd' : ¬ dims.d = 0#usize := fun e => hd (by simpa using congrArg (fun x : Usize => x.val) e)
  have : dims.v = 0#usize := UScalar.eq_of_val_eq (by simpa using h)
  simp [hn', hd', this]

/-- A zero `chunk` is refused with `Shape`. -/
theorem head_chunked_shape_chunk (hidden w_head z_t : Slice I64) (dims : learn_port.head.Dims)
    (chunk : Usize) (objective : learn_port.head.Objective) (hn : dims.n.val ≠ 0)
    (hd : dims.d.val ≠ 0) (hv : dims.v.val ≠ 0) (h : chunk.val = 0) :
    learn_port.head.head_chunked hidden w_head z_t dims chunk objective =
      ok (core.result.Result.Err learn_port.head.HeadError.Shape) := by
  unfold learn_port.head.head_chunked
  have hn' : ¬ dims.n = 0#usize := fun e => hn (by simpa using congrArg (fun x : Usize => x.val) e)
  have hd' : ¬ dims.d = 0#usize := fun e => hd (by simpa using congrArg (fun x : Usize => x.val) e)
  have hv' : ¬ dims.v = 0#usize := fun e => hv (by simpa using congrArg (fun x : Usize => x.val) e)
  have : chunk = 0#usize := UScalar.eq_of_val_eq (by simpa using h)
  simp [hn', hd', hv', this]

/-- **`head_chunked`, success arm.** On non-zero dimensions whose products fit `usize`, with
buffers of exactly `n·d`, `v·d` and `n·v` entries, the result is the rounded accumulator of the
row loop, the row loop's `dh`, and the `dwOutVals` rounding of its `dw`. The row loop enters as
`hrows`, given the zeroed `dh`/`dw` buffers and `n_scale = n·2^32`. -/
theorem head_chunked_spec (hidden w_head z_t : Slice I64) (dims : learn_port.head.Dims)
    (chunk : Usize) (objective : learn_port.head.Objective)
    (hn : dims.n.val ≠ 0) (hd : dims.d.val ≠ 0) (hv : dims.v.val ≠ 0) (hc : chunk.val ≠ 0)
    (hx : dims.n.val * dims.d.val ≤ Usize.max) (hx1 : dims.v.val * dims.d.val ≤ Usize.max)
    (hx2 : dims.n.val * dims.v.val ≤ Usize.max)
    (hh : hidden.val.length = dims.n.val * dims.d.val)
    (hw : w_head.val.length = dims.v.val * dims.d.val)
    (hz : z_t.val.length = dims.n.val * dims.v.val)
    (hn32 : I128.min ≤ (dims.n.val : ℤ) * 2 ^ (32 : ℕ) ∧
      (dims.n.val : ℤ) * 2 ^ (32 : ℕ) ≤ I128.max)
    (hrows : ∀ (x3 : I128) (dh0 : Slice I64) (dw0 : Slice I128),
      x3.val = (dims.n.val : ℤ) * 2 ^ (32 : ℕ) →
      hvals dh0 = List.replicate (dims.n.val * dims.d.val) 0 →
      svals dw0 = List.replicate (dims.v.val * dims.d.val) 0 →
      ∃ acc dh1 dw1, learn_port.head.rows_loop objective hidden w_head z_t dims x3 dh0 dw0 =
          ok (core.result.Result.Ok acc, dh1, dw1) ∧
        (2 * absZ (Int.tmod acc.val (dims.n.val : ℤ)) ≤ I128.max ∧
          rndDivDom acc.val (dims.n.val : ℤ) ∧
          I64.min ≤ rndDiv acc.val (dims.n.val : ℤ) ∧ rndDiv acc.val (dims.n.val : ℤ) ≤ I64.max) ∧
        dw1.val.length = dims.v.val * dims.d.val ∧
        (∀ t < dims.v.val * dims.d.val, rndDivDom ((svals dw1).getD t 0) x3.val ∧
          2 * absZ (Int.tmod ((svals dw1).getD t 0) x3.val) ≤ I128.max) ∧
        (∀ t < dims.v.val * dims.d.val, I64.min ≤ rndDiv ((svals dw1).getD t 0) x3.val ∧
          rndDiv ((svals dw1).getD t 0) x3.val ≤ I64.max)) :
    ∃ loss dh dwo, learn_port.head.head_chunked hidden w_head z_t dims chunk objective =
        ok (core.result.Result.Ok (loss, dh, dwo)) ∧
      ∃ (x3 : I128) (dh0 : Slice I64) (dw0 : Slice I128) (acc : I128) (dh1 : Slice I64)
        (dw1 : Slice I128),
        x3.val = (dims.n.val : ℤ) * 2 ^ (32 : ℕ) ∧
        hvals dh0 = List.replicate (dims.n.val * dims.d.val) 0 ∧
        svals dw0 = List.replicate (dims.v.val * dims.d.val) 0 ∧
        learn_port.head.rows_loop objective hidden w_head z_t dims x3 dh0 dw0 =
          ok (core.result.Result.Ok acc, dh1, dw1) ∧
        loss.val = rndDiv acc.val (dims.n.val : ℤ) ∧ vvals dh = hvals dh1 ∧
        vvals dwo = dwOutVals (svals dw1) x3.val (List.replicate (dims.v.val * dims.d.val) 0)
          (dims.v.val * dims.d.val) := by
  unfold learn_port.head.head_chunked
  have hn' : ¬ dims.n = 0#usize := fun e => hn (by simpa using congrArg (fun x : Usize => x.val) e)
  have hd' : ¬ dims.d = 0#usize := fun e => hd (by simpa using congrArg (fun x : Usize => x.val) e)
  have hv' : ¬ dims.v = 0#usize := fun e => hv (by simpa using congrArg (fun x : Usize => x.val) e)
  have hc' : ¬ chunk = 0#usize := fun e => hc (by simpa using congrArg (fun x : Usize => x.val) e)
  simp only [hn', hd', hv', hc', if_false]
  obtain ⟨X, hXe, hXv⟩ := mul_dims_ok dims.n dims.d hx
  obtain ⟨X1, hX1e, hX1v⟩ := mul_dims_ok dims.v dims.d hx1
  obtain ⟨X2, hX2e, hX2v⟩ := mul_dims_ok dims.n dims.v hx2
  rw [hXe]
  simp only [bind_tc_ok]
  rw [hX1e]
  simp only [bind_tc_ok]
  rw [hX2e]
  simp only [bind_tc_ok]
  have hl1 : (Slice.len hidden != X) = false := by
    simp only [bne_eq_false_iff_eq]
    apply UScalar.eq_of_val_eq
    show (Slice.len hidden).val = X.val
    rw [Slice.len_val, hXv]; exact hh
  have hl2 : (Slice.len w_head != X1) = false := by
    simp only [bne_eq_false_iff_eq]
    apply UScalar.eq_of_val_eq
    show (Slice.len w_head).val = X1.val
    rw [Slice.len_val, hX1v]; exact hw
  have hl3 : (Slice.len z_t != X2) = false := by
    simp only [bne_eq_false_iff_eq]
    apply UScalar.eq_of_val_eq
    show (Slice.len z_t).val = X2.val
    rw [Slice.len_val, hX2v]; exact hz
  simp only [hl1, hl2, hl3, Bool.false_eq_true, ↓reduceIte]
  obtain ⟨i3, hi3, hi3v⟩ := dim128_spec dims.n
  have hone : (core.convert.num.FromI128I64.from (4294967296#i64)).val = 2 ^ (32 : ℕ) := by
    rw [core.convert.num.FromI128I64.from_val_eq]; norm_num
  obtain ⟨x3, hx3, hx3v⟩ := mul128_ok i3 (core.convert.num.FromI128I64.from (4294967296#i64))
    (by rw [hi3v, hone]; exact hn32)
  have hx3n : x3.val = (dims.n.val : ℤ) * 2 ^ (32 : ℕ) := by rw [hx3v, hi3v, hone]
  obtain ⟨dh, hdh, hdhv, hdhl⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI64 0#i64 X (by rfl))
  obtain ⟨dw, hdw, hdwv, hdwl⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI128 0#i128 X1 (by rfl))
  have hs0 : hvals (alloc.vec.Vec.deref_mut dh).1 = List.replicate (dims.n.val * dims.d.val) 0 := by
    have : hvals (alloc.vec.Vec.deref_mut dh).1 = vvals dh := rfl
    rw [this, vvals, hdhv, hXv]; simp
  have hs1 : svals (alloc.vec.Vec.deref_mut dw).1 = List.replicate (dims.v.val * dims.d.val) 0 := by
    have : svals (alloc.vec.Vec.deref_mut dw).1 = dw.val.map (fun z => z.val) := rfl
    rw [this, hdwv, hX1v]; simp
  obtain ⟨acc, dh1, dw1, hr, ⟨h2, hrd, hf1, hf2⟩, hdwl1, hdr, hdf⟩ :=
    hrows x3 (alloc.vec.Vec.deref_mut dh).1 (alloc.vec.Vec.deref_mut dw).1 hx3n hs0 hs1
  have hipos : 0 < i3.val := by rw [hi3v]; omega
  obtain ⟨i6, hi6, hi6v⟩ := round_div_dom_spec acc i3 hipos (by rw [hi3v]; exact h2)
    (by rw [hi3v]; exact hrd)
  have hi6v' : i6.val = rndDiv acc.val (dims.n.val : ℤ) := by rw [hi6v, hi3v]
  obtain ⟨x4, hx4, hx4v⟩ := to64_ok i6 (by rw [hi6v']; exact ⟨hf1, hf2⟩)
  obtain ⟨dwo, hdwo, hdwov, hdwol⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI64 0#i64 X1 (by rfl))
  have hD : (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice dw1)).val = dw1.val := by
    unfold alloc.vec.Vec.deref; exact Slice.from_val _ _
  have hDs : svals (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice dw1)) = svals dw1 := by
    unfold svals; rw [hD]
  have hDl : (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice dw1)).val.length = dims.v.val * dims.d.val := by
    rw [hD]; exact hdwl1
  have hwol : (alloc.vec.Vec.deref_mut dwo).1.val.length = dims.v.val * dims.d.val := by
    have : (alloc.vec.Vec.deref_mut dwo).1.val = dwo.val := rfl
    rw [this, hdwov, hX1v]; simp
  have hwoh : hvals (alloc.vec.Vec.deref_mut dwo).1 = List.replicate (dims.v.val * dims.d.val) 0 := by
    have : hvals (alloc.vec.Vec.deref_mut dwo).1 = vvals dwo := rfl
    rw [this, vvals, hdwov, hX1v]; simp
  obtain ⟨y, hy, hyv⟩ := dw_out_fill_spec (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice dw1)) x3
    (alloc.vec.Vec.deref_mut dwo).1 (by rw [hDl, hwol])
    (by rw [hx3n]; positivity)
    (fun t ht => by rw [hDs, hx3n]; rw [hDl] at ht; rw [← hx3n]; exact hdr t ht)
    (fun t ht => by rw [hDs]; rw [hDl] at ht; exact hdf t ht)
  refine ⟨x4, alloc.vec.Vec.ofSlice dh1, alloc.vec.Vec.ofSlice y, ?_,
    x3, (alloc.vec.Vec.deref_mut dh).1, (alloc.vec.Vec.deref_mut dw).1, acc, dh1, dw1,
    hx3n, hs0, hs1, hr, ?_, rfl, ?_⟩
  · rw [hi3]
    simp only [bind_tc_ok]
    rw [engine_ONE_eq]
    simp only [lift, bind_tc_ok]
    rw [hx3]
    simp only [bind_tc_ok]
    rw [hdh]
    simp only [bind_tc_ok]
    rw [hdw]
    simp only [bind_tc_ok]
    change (learn_port.head.rows_loop objective hidden w_head z_t dims x3
      (alloc.vec.Vec.deref_mut dh).1 (alloc.vec.Vec.deref_mut dw).1 >>= _) = _
    rw [hr]
    simp only [bind_tc_ok]
    change (learn_port.head.round_div acc i3 >>= _) = _
    rw [hi6]
    simp only [bind_tc_ok]
    rw [hx4]
    simp only [bind_tc_ok]
    rw [hdwo]
    simp only [bind_tc_ok]
    change (learn_port.head.dw_out_fill (alloc.vec.Vec.deref (alloc.vec.Vec.ofSlice dw1)) x3
      (alloc.vec.Vec.deref_mut dwo).1 >>= _) = _
    rw [hy]
    first | exact rfl | simp
  · rw [hx4v, hi6v']
  · have h1 : vvals (alloc.vec.Vec.ofSlice y) = hvals y := rfl
    rw [h1, hyv, hDs, hx3n, hwoh, hDl]

/-! ## 21. Refusal and joint witnesses -/

/-- **`dot_round` refusal.** On the success domain of `dot_round_spec`, a rounded quotient outside
`i64` is refused with `Overflow`. -/
theorem dot_round_overflow (a : Slice I64) (abase : Usize) (b : Slice I64) (bbase : Usize)
    (d : Usize)
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
    (hfit : ¬ (I64.min ≤ rndDiv (dotAux (hvals a) abase.val (hvals b) bbase.val 0 d.val)
        (2 ^ (32 : ℕ)) ∧
      rndDiv (dotAux (hvals a) abase.val (hvals b) bbase.val 0 d.val) (2 ^ (32 : ℕ)) ≤ I64.max)) :
    learn_port.head.dot_round a abase b bbase d =
      ok (core.result.Result.Err learn_port.head.HeadError.Overflow) := by
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
  change (learn_port.head.round_div acc (core.convert.num.FromI128I64.from (4294967296#i64))
      >>= fun i2 => learn_port.head.to64 i2) = _
  rw [hi2]
  simp only [bind_tc_ok]
  rw [hacc, hone] at hi2v
  rcases not_and_or.mp hfit with h | h
  · exact to64_low i2 (by rw [hi2v]; exact not_le.mp h)
  · exact to64_high i2 (by rw [hi2v]; exact not_le.mp h)

/-- One-element `i64` slice holding `2^62`. -/
def witBig : Slice I64 := Slice.from [4611686018427387904#i64] (by scalar_tac)

/-- **Witness, `dot_round` `Overflow` arm.** `2^62 · 2^62 = 2^124` fits `i128` and divides by `2^32`
exactly to `2^92 > i64::MAX`, so the single-term dot product is refused. -/
theorem dot_round_overflow_witness :
    learn_port.head.dot_round witBig 0#usize witBig 0#usize 1#usize =
      ok (core.result.Result.Err learn_port.head.HeadError.Overflow) := by
  have hv : hvals witBig = [4611686018427387904] := by
    unfold hvals witBig; rw [Slice.from_val]; rfl
  have h0 : (0#usize).val = 0 := rfl
  have h1 : (1#usize).val = 1 := rfl
  have hl : witBig.val.length = 1 := by unfold witBig; rw [Slice.from_val]; rfl
  apply dot_round_overflow
  · intro t ht; rw [h0]; rw [h1] at ht; simp [Usize.max]; scalar_tac
  · intro t ht; rw [h0]; rw [h1] at ht; simp [Usize.max]; scalar_tac
  · intro t ht; rw [h0, hl]; rw [h1] at ht; omega
  · intro t ht; rw [h0, hl]; rw [h1] at ht; omega
  · intro t ht; rw [h0, hv]; rw [h1] at ht
    have : t = 0 := by omega
    subst this; rw [i128_min_eq, i128_max_eq]; norm_num
  · intro t ht; rw [h0, hv, h1] at *
    have ht2 : t = 0 ∨ t = 1 := by omega
    rcases ht2 with rfl | rfl <;> simp [dotAux, i128_min_eq, i128_max_eq]
  · rw [h0, hv, h1]; simp [dotAux, absZ, i128_max_eq]
  · rw [h0, hv, h1]; simp [dotAux, rndDivDom, i128_min_eq, i128_max_eq]
  · rw [h0, hv, h1]; simp [dotAux, rndDiv, absZ, i64_max_eq, i64_min_eq]

/-- **Witness, `head_chunked` `Shape` arm.** `n = 0` is refused. -/
theorem head_chunked_shape_witness :
    learn_port.head.head_chunked witBig witBig witBig ⟨0#usize, 1#usize, 1#usize⟩ 1#usize
      learn_port.head.Objective.Kl =
      ok (core.result.Result.Err learn_port.head.HeadError.Shape) :=
  head_chunked_shape_n _ _ _ _ _ _ rfl

/-- **Witness, `head_chunked` success arm: the structural domain is jointly satisfiable.** At
`n = d = v = chunk = 1` with one-element buffers, every hypothesis of `head_chunked_spec` other than
`hrows` holds. `hrows` (the row loop's success, which needs the `softmax_row` specification) is the
open obligation. -/
theorem head_chunked_domain_witness :
    ∃ (hidden w_head z_t : Slice I64) (dims : learn_port.head.Dims) (chunk : Usize),
      dims.n.val ≠ 0 ∧ dims.d.val ≠ 0 ∧ dims.v.val ≠ 0 ∧ chunk.val ≠ 0 ∧
      dims.n.val * dims.d.val ≤ Usize.max ∧ dims.v.val * dims.d.val ≤ Usize.max ∧
      dims.n.val * dims.v.val ≤ Usize.max ∧
      hidden.val.length = dims.n.val * dims.d.val ∧
      w_head.val.length = dims.v.val * dims.d.val ∧
      z_t.val.length = dims.n.val * dims.v.val ∧
      (I128.min ≤ (dims.n.val : ℤ) * 2 ^ (32 : ℕ) ∧
        (dims.n.val : ℤ) * 2 ^ (32 : ℕ) ≤ I128.max) := by
  refine ⟨witBig, witBig, witBig, ⟨1#usize, 1#usize, 1#usize⟩, 1#usize, ?_⟩
  have h1 : (1#usize).val = 1 := rfl
  have hl : witBig.val.length = 1 := by unfold witBig; rw [Slice.from_val]; rfl
  simp only [h1, hl]
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, ?_, ?_, ?_, by norm_num, by norm_num, by norm_num, ?_⟩
  all_goals (simp [Usize.max]; scalar_tac)

end apeiron_head
