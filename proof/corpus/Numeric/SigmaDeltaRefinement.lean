import ApeironWord
import Numeric.SigmaDelta
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The executed sigma-delta block refines `Numeric.SigmaDelta`

`sigma_delta.sd_binary_block` (`src/learn/sigma_delta.rs`) is the checked-`i64` first-order
sigma-delta over a fixed-point block `w` with scale `a`: `e_0 = 0`, `s_t = sign(w_t + e_t)`,
`e_{t+1} = w_t + e_t − a·s_t`. It returns `(some (e_n, peak), s')` on success and `(none, s')` on a
shape mismatch or a checked overflow; it never wraps and never fails.

`Numeric.SigmaDelta.sdState`/`sdSign` are stated over a field, so the `i64` values are read through
the cast to `ℚ`. Landed here:

* `sd_step_spec`: the checked step is total and, when it returns `some`, is the exact
  `(sign, e', |e'|)` triple;
* `sd_binary_block_spec`: on `(some (e, peak), s')` every emitted sign equals
  `Numeric.SigmaDelta.sdSign`, `e = sdState … n`, and `peak = sdPeak … n`
  (`= max_{1≤t≤n} |sdState … t|`); the call is total (`ok`, never `fail`);
* `sd_binary_block_shape`: a length mismatch returns `none`;
* `sd_binary_block_witness`: a concrete block (`w = [3]`, `a = 5`) hits the success branch with
  `e = −2`, `peak = 2`, sign `+1`, and a concrete block (`w = [i64::MAX]`, `a = i64::MIN`, where
  `w_0 − a·s_0` overflows) hits the `none` branch; with `sd_binary_block_shape` all three `none`/`some`
  causes are inhabited.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the `f64` weight-to-fixed-point rounding in
`sd1_row` (`src/learn/onebit.rs`) are not covered; only the integer body is.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result
open Numeric.SigmaDelta

/-! ## Reference quantities -/

/-- The block read as a `ℚ`-valued sequence (zero past the end). -/
def sdW (w : Slice I64) (t : ℕ) : ℚ := ((w.val.getD t 0#i64).val : ℚ)

/-- The running peak `max_{1 ≤ t ≤ n} |e_t|` of the reference state. -/
def sdPeak (W : ℕ → ℚ) (a : ℚ) : ℕ → ℚ
  | 0 => 0
  | t + 1 => max (sdPeak W a t) |sdState W a (t + 1)|

/-! ## Checked operations -/

theorem sd_chk_add_some (x y z : I64) (h : I64.checked_add x y = some z) :
    z.val = x.val + y.val := by
  have hs := I64.checked_add_bv_spec x y
  have h' : core.num.checked_add_IScalar x y = some z := by simpa only [I64.checked_add] using h
  rw [h'] at hs
  exact hs.2.2.1

theorem sd_chk_sub_some (x y z : I64) (h : I64.checked_sub x y = some z) :
    z.val = x.val - y.val := by
  have hs := I64.checked_sub_bv_spec x y
  have h' : core.num.checked_sub_IScalar x y = some z := by simpa only [I64.checked_sub] using h
  rw [h'] at hs
  exact hs.2.2.1

theorem sd_chk_mul_some (x y z : I64) (h : I64.checked_mul x y = some z) :
    z.val = x.val * y.val := by
  have hs := I64.checked_mul_bv_spec x y
  have h' : core.num.checked_mul_IScalar x y = some z := by simpa only [I64.checked_mul] using h
  rw [h'] at hs
  exact hs.2.2.1

/-! ## One step -/

/-- The checked step is total; on `some` it is the exact `(sign, e', |e'|)` triple. -/
theorem sd_step_spec (wt e a : I64) :
    ∃ r, sigma_delta.sd_step wt e a = ok r ∧
      ∀ sg e2 mag, r = some (sg, e2, mag) →
        (sg.val = if 0 ≤ wt.val + e.val then 1 else -1) ∧
        e2.val = wt.val + e.val - a.val * sg.val ∧ mag.val = |e2.val| := by
  unfold sigma_delta.sd_step
  simp only [lift, bind_tc_ok]
  cases h1 : I64.checked_add wt e with
  | none => exact ⟨none, by simp, by simp⟩
  | some u =>
    have hu := sd_chk_add_some wt e u h1
    simp only
    by_cases hge : u >= 0#i64
    · have hge' : 0 ≤ wt.val + e.val := by
        have := (IScalar.le_equiv (0#i64) u).mp hge
        simpa [hu] using this
      simp only [hge, if_true, bind_tc_ok]
      have hfrom : (core.convert.num.FromI64I8.from 1#i8).val = 1 := by
        rw [core.convert.num.FromI64I8.from_val_eq]; rfl
      cases h2 : I64.checked_mul a (core.convert.num.FromI64I8.from 1#i8) with
      | none => exact ⟨none, by simp, by simp⟩
      | some q =>
        have hq := sd_chk_mul_some _ _ q h2
        rw [hfrom] at hq
        simp only
        cases h3 : I64.checked_sub u q with
        | none => exact ⟨none, by simp, by simp⟩
        | some e2 =>
          have he2 := sd_chk_sub_some u q e2 h3
          simp only
          cases h4 : I64.checked_sub 0#i64 e2 with
          | none => exact ⟨none, by simp, by simp⟩
          | some neg =>
            have hneg := sd_chk_sub_some 0#i64 e2 neg h4
            simp only
            by_cases hlt : e2 < 0#i64
            · have hlt' : e2.val < 0 := by
                have := (IScalar.lt_equiv e2 (0#i64)).mp hlt
                simpa using this
              refine ⟨some (1#i8, e2, neg), by simp [hlt], ?_⟩
              intro sg e2' mag hr
              simp only [Option.some.injEq, Prod.mk.injEq] at hr
              obtain ⟨rfl, rfl, rfl⟩ := hr
              refine ⟨by simp [hge'], ?_, ?_⟩
              · rw [he2, hq, hu]; simp
              · rw [hneg, abs_of_neg hlt']; simp
            · have hlt' : 0 ≤ e2.val := by
                have := (IScalar.lt_equiv e2 (0#i64)).not.mp hlt
                simpa using this
              refine ⟨some (1#i8, e2, e2), by simp [hlt], ?_⟩
              intro sg e2' mag hr
              simp only [Option.some.injEq, Prod.mk.injEq] at hr
              obtain ⟨rfl, rfl, rfl⟩ := hr
              refine ⟨by simp [hge'], ?_, ?_⟩
              · rw [he2, hq, hu]; simp
              · rw [abs_of_nonneg hlt']
    · have hge' : ¬ 0 ≤ wt.val + e.val := by
        intro h
        apply hge
        have : (0#i64).val ≤ u.val := by simpa [hu] using h
        exact (IScalar.le_equiv (0#i64) u).mpr this
      simp only [hge, if_false, bind_tc_ok]
      have hfrom : (core.convert.num.FromI64I8.from (-1)#i8).val = -1 := by
        rw [core.convert.num.FromI64I8.from_val_eq]; rfl
      cases h2 : I64.checked_mul a (core.convert.num.FromI64I8.from (-1)#i8) with
      | none => exact ⟨none, by simp, by simp⟩
      | some q =>
        have hq := sd_chk_mul_some _ _ q h2
        rw [hfrom] at hq
        simp only
        cases h3 : I64.checked_sub u q with
        | none => exact ⟨none, by simp, by simp⟩
        | some e2 =>
          have he2 := sd_chk_sub_some u q e2 h3
          simp only
          cases h4 : I64.checked_sub 0#i64 e2 with
          | none => exact ⟨none, by simp, by simp⟩
          | some neg =>
            have hneg := sd_chk_sub_some 0#i64 e2 neg h4
            simp only
            by_cases hlt : e2 < 0#i64
            · have hlt' : e2.val < 0 := by
                have := (IScalar.lt_equiv e2 (0#i64)).mp hlt
                simpa using this
              refine ⟨some ((-1)#i8, e2, neg), by simp [hlt], ?_⟩
              intro sg e2' mag hr
              simp only [Option.some.injEq, Prod.mk.injEq] at hr
              obtain ⟨rfl, rfl, rfl⟩ := hr
              refine ⟨by simp [hge'], ?_, ?_⟩
              · rw [he2, hq, hu]; simp
              · rw [hneg, abs_of_neg hlt']; simp
            · have hlt' : 0 ≤ e2.val := by
                have := (IScalar.lt_equiv e2 (0#i64)).not.mp hlt
                simpa using this
              refine ⟨some ((-1)#i8, e2, e2), by simp [hlt], ?_⟩
              intro sg e2' mag hr
              simp only [Option.some.injEq, Prod.mk.injEq] at hr
              obtain ⟨rfl, rfl, rfl⟩ := hr
              refine ⟨by simp [hge'], ?_, ?_⟩
              · rw [he2, hq, hu]; simp
              · rw [abs_of_nonneg hlt']


/-! ## The loop -/

/-- The success-path invariant after `t` steps: state, peak and the signs emitted so far. -/
def SdGood (w : Slice I64) (a : I64) (t : ℕ) (s : Slice I8) (e peak : I64) : Prop :=
  (e.val : ℚ) = sdState (sdW w) (a.val : ℚ) t ∧
  (peak.val : ℚ) = sdPeak (sdW w) (a.val : ℚ) t ∧
  ∀ i, i < t → ((s.val.getD i 0#i8).val : ℚ) = sdSign (sdW w) (a.val : ℚ) i

theorem sd_slice_update_val {α : Type} (sl sl' : Slice α) (i : Usize) (x : α)
    (h : Slice.update sl i x = ok sl') : sl'.val = sl.val.set i.val x := by
  unfold Slice.update at h
  split at h
  · simp at h
  · have := (Result.ok_injective h).symm
    subst this
    simp

theorem sd_sign_cast (W : ℕ → ℚ) (a : ℚ) (t : ℕ) (xw ew sg : ℤ) (hW : W t = xw)
    (hE : sdState W a t = ew) (hsg : sg = if 0 ≤ xw + ew then 1 else -1) :
    (sg : ℚ) = sdSign W a t := by
  unfold sdSign sdSignAt
  rw [hW, hE]
  by_cases c : 0 ≤ xw + ew
  · have cq : (0 : ℚ) ≤ (xw : ℚ) + (ew : ℚ) := by exact_mod_cast c
    simp [hsg, c, cq]
  · have cq : ¬ (0 : ℚ) ≤ (xw : ℚ) + (ew : ℚ) := by exact_mod_cast c
    simp [hsg, c, cq]

/-- One loop step on the success path: the body continues with `t + 1` and either records the
overflow (`ok2 = false`) or extends the invariant. -/
theorem sd_body_step (w : Slice I64) (a : I64) (n : Usize) (s : Slice I8) (e peak : I64)
    (t : Usize) (hn : n.val = w.val.length) (hs : s.val.length = n.val)
    (ht : t.val < n.val) (hg : SdGood w a t.val s e peak) :
    ∃ s1 e1 peak1 t1 ok2, t1.val = t.val + 1 ∧
      sigma_delta.sd_binary_block_loop.body w a n s e peak t true
        = ok (ControlFlow.cont (s1, e1, peak1, t1, ok2)) ∧
      s1.val.length = n.val ∧ (ok2 = true → SdGood w a (t.val + 1) s1 e1 peak1) := by
  have htn : t < n := (UScalar.lt_equiv t n).mpr ht
  have htw : t.val < w.val.length := by omega
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (Slice.index_usize_spec w t htw)
  obtain ⟨r, hr, hspec⟩ := sd_step_spec x e a
  have hxW : (x.val : ℚ) = sdW w t.val := by
    rw [hxv, sdW, List.getD_eq_getElem _ _ htw]
  have hmax := Slice.length_ineq w
  obtain ⟨t1, hadd, ht1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := t) (y := 1#usize) (by simp; omega))
  have ht1v' : t1.val = t.val + 1 := by simpa using ht1v
  unfold sigma_delta.sd_binary_block_loop.body
  rw [if_pos htn]
  simp only [if_true]
  rw [hx]
  simp only [bind_tc_ok]
  rw [hr]
  simp only [bind_tc_ok]
  cases hrc : r with
  | none =>
    refine ⟨s, e, peak, t1, false, ht1v', ?_, hs, by simp⟩
    simp only [bind_tc_ok]
    rw [hadd]
    simp only [bind_tc_ok]
    rfl
  | some tr =>
    obtain ⟨sg, e2, mag⟩ := tr
    obtain ⟨hsg, he2, hmag⟩ := hspec sg e2 mag hrc
    have hsw : t.val < s.val.length := by omega
    obtain ⟨s2, hupd, hs2⟩ := WP.spec_imp_exists (Slice.update_spec s t sg hsw)
    have hs2v : s2.val = s.val.set t.val sg := by rw [hs2]; simp
    have hs2len : s2.val.length = n.val := by rw [hs2v, List.length_set]; exact hs
    have hsgQ : (sg.val : ℚ) = sdSign (sdW w) (a.val : ℚ) t.val :=
      sd_sign_cast (sdW w) (a.val : ℚ) t.val x.val e.val sg.val
        (by rw [← hxW]) hg.1.symm hsg
    have he2Q : (e2.val : ℚ) = sdState (sdW w) (a.val : ℚ) (t.val + 1) := by
      have h1 : (e2.val : ℚ) = x.val + e.val - a.val * sg.val := by exact_mod_cast he2
      rw [sdState_succ, h1, hxW, hg.1, hsgQ]
      ring
    have hmagQ : (mag.val : ℚ) = |sdState (sdW w) (a.val : ℚ) (t.val + 1)| := by
      rw [← he2Q, hmag]; push_cast; rfl
    have hsigns : ∀ i, i < t.val + 1 → ((s2.val.getD i 0#i8).val : ℚ) = sdSign (sdW w) (a.val : ℚ) i := by
      intro i hi
      rcases Nat.lt_succ_iff_lt_or_eq.mp hi with hlt | heq
      · have hne : t.val ≠ i := by omega
        rw [hs2v, List.getD_eq_getElem?_getD, List.getElem?_set_ne hne,
          ← List.getD_eq_getElem?_getD]
        exact hg.2.2 i hlt
      · subst heq
        rw [hs2v, List.getD_eq_getElem?_getD, List.getElem?_set_self hsw]
        simpa using hsgQ
    by_cases hm : mag > peak
    · refine ⟨s2, e2, mag, t1, true, ht1v', ?_, hs2len, fun _ => ⟨he2Q, ?_, hsigns⟩⟩
      · have hlt : peak.val < mag.val := (IScalar.lt_equiv peak mag).mp hm
        simp [hupd, hlt, hadd]
      · have hlt : peak.val < mag.val := (IScalar.lt_equiv peak mag).mp hm
        have hltQ : (peak.val : ℚ) < (mag.val : ℚ) := by exact_mod_cast hlt
        rw [sdPeak, ← hg.2.1, ← hmagQ]
        exact (max_eq_right hltQ.le).symm
    · refine ⟨s2, e2, peak, t1, true, ht1v', ?_, hs2len, fun _ => ⟨he2Q, ?_, hsigns⟩⟩
      · have hle : mag.val ≤ peak.val := by
          have := (IScalar.lt_equiv peak mag).not.mp hm
          omega
        simp [hupd, not_lt.mpr hle, hadd]
      · have hle : mag.val ≤ peak.val := by
          have := (IScalar.lt_equiv peak mag).not.mp hm
          omega
        have hleQ : (mag.val : ℚ) ≤ (peak.val : ℚ) := by exact_mod_cast hle
        rw [sdPeak, ← hg.2.1, ← hmagQ]
        exact (max_eq_left hleQ).symm

/-- The loop: it never fails, keeps the buffer length, and when it ends with `ok = true` the
invariant holds at `t = n`. -/
theorem sd_loop_spec (w : Slice I64) (a : I64) (s : Slice I8) (n : Usize)
    (hn : n.val = w.val.length) (hs : s.val.length = n.val) :
    WP.spec (sigma_delta.sd_binary_block_loop w a s n 0#i64 0#i64 0#usize true)
      (fun r => r.1.val.length = n.val ∧
        (r.2.2.2 = true → SdGood w a n.val r.1 r.2.1 r.2.2.1)) := by
  unfold sigma_delta.sd_binary_block_loop
  apply loop.spec_decr_nat
    (measure := fun st : Slice I8 × I64 × I64 × Usize × Bool => n.val - st.2.2.2.1.val)
    (inv := fun st => st.1.val.length = n.val ∧ st.2.2.2.1.val ≤ n.val ∧
      (st.2.2.2.2 = true → SdGood w a st.2.2.2.1.val st.1 st.2.1 st.2.2.1))
  · intro st hst
    rcases st with ⟨s0, e0, p0, t0, ok0⟩
    obtain ⟨hl0, ht0, hg0⟩ := hst
    dsimp only at hl0 ht0 hg0 ⊢
    by_cases hok : ok0 = true
    · subst hok
      have hg := hg0 rfl
      by_cases htn : t0.val < n.val
      · obtain ⟨s1, e1, p1, t1, ok2, ht1, hbody, hl1, hg1⟩ :=
          sd_body_step w a n s0 e0 p0 t0 hn hl0 htn hg
        rw [hbody]
        refine ⟨⟨hl1, ?_, ?_⟩, ?_⟩
        · show t1.val ≤ n.val
          omega
        · show ok2 = true → SdGood w a t1.val s1 e1 p1
          rw [ht1]; exact hg1
        · show n.val - t1.val < n.val - t0.val
          omega
      · have hnot : ¬ (t0 < n) := fun h => htn ((UScalar.lt_equiv t0 n).mp h)
        have hbody : sigma_delta.sd_binary_block_loop.body w a n s0 e0 p0 t0 true
            = ok (ControlFlow.done (s0, e0, p0, true)) := by
          rw [sigma_delta.sd_binary_block_loop.body, if_neg hnot]
        rw [hbody]
        have hteq : t0.val = n.val := by omega
        refine ⟨hl0, fun _ => ?_⟩
        rw [← hteq]; exact hg
    · have hf : ok0 = false := by simpa using hok
      subst hf
      have hbody : sigma_delta.sd_binary_block_loop.body w a n s0 e0 p0 t0 false
          = ok (ControlFlow.done (s0, e0, p0, false)) := by
        unfold sigma_delta.sd_binary_block_loop.body
        by_cases htn : t0 < n
        · simp [htn]
        · simp [htn]
      rw [hbody]
      exact ⟨hl0, fun h => absurd h (by simp)⟩
  · refine ⟨hs, by simp, fun _ => ⟨?_, ?_, ?_⟩⟩
    · simp [sdState]
    · simp [sdPeak]
    · intro i hi
      simp at hi

/-- A length mismatch returns `none` and leaves the buffer untouched. -/
theorem sd_binary_block_shape (w : Slice I64) (a : I64) (s : Slice I8)
    (h : s.val.length ≠ w.val.length) :
    sigma_delta.sd_binary_block w a s = ok (none, s) := by
  unfold sigma_delta.sd_binary_block
  have hne : (Slice.len s != Slice.len w) = true := by
    simp only [bne_iff_ne, ne_eq]
    intro heq
    apply h
    have := congrArg UScalar.val heq
    rwa [Slice.len_val, Slice.len_val] at this
  simp [hne]

/-- Refinement. The call is total. On `some (e, peak)` the shapes agree and the emitted signs, the
final state and the peak are the canonical `Numeric.SigmaDelta` ones, read through `ℚ`. -/
theorem sd_binary_block_spec (w : Slice I64) (a : I64) (s : Slice I8) :
    ∃ r s', sigma_delta.sd_binary_block w a s = ok (r, s') ∧
      s'.val.length = s.val.length ∧
      ∀ e peak, r = some (e, peak) →
        s.val.length = w.val.length ∧
        (e.val : ℚ) = sdState (sdW w) (a.val : ℚ) w.val.length ∧
        (peak.val : ℚ) = sdPeak (sdW w) (a.val : ℚ) w.val.length ∧
        ∀ i, i < w.val.length →
          ((s'.val.getD i 0#i8).val : ℚ) = sdSign (sdW w) (a.val : ℚ) i := by
  by_cases hl : s.val.length = w.val.length
  · have hn : (Slice.len w).val = w.val.length := Slice.len_val w
    have hlen : Slice.len s = Slice.len w := by
      apply UScalar.eq_of_val_eq
      rw [Slice.len_val, Slice.len_val]; exact hl
    obtain ⟨res, hres, hpost⟩ := WP.spec_imp_exists
      (sd_loop_spec w a s (Slice.len w) hn.symm (by rw [hn]; exact hl))
    obtain ⟨s1, e1, p1, ok1⟩ := res
    obtain ⟨hl1, hg1⟩ := hpost
    unfold sigma_delta.sd_binary_block
    simp only [hlen, bne_self_eq_false, Bool.false_eq_true, if_false]
    simp only [hres, bind_tc_ok]
    cases hok : ok1 with
    | true =>
      have hg := hg1 hok
      refine ⟨some (e1, p1), s1, by simp, by rw [hl1, hn, hl], ?_⟩
      intro e peak hr
      simp only [Option.some.injEq, Prod.mk.injEq] at hr
      obtain ⟨rfl, rfl⟩ := hr
      rw [hn] at hg
      exact ⟨hl, hg.1, hg.2.1, hg.2.2⟩
    | false =>
      refine ⟨none, s1, by simp, by rw [hl1, hn, hl], ?_⟩
      intro e peak hr
      exact absurd hr (by simp)
  · exact ⟨none, s, sd_binary_block_shape w a s hl, rfl, fun e peak hr => absurd hr (by simp)⟩

/-! ## Joint reachable witness -/

/-- The checked step on `(w, e, a) = (3, 0, 5)` returns sign `+1`, `e' = −2`, `|e'| = 2`. -/
theorem sd_step_witness_some :
    sigma_delta.sd_step 3#i64 0#i64 5#i64 = ok (some (1#i8, (-2)#i64, 2#i64)) := by
  have c1 : I64.checked_add 3#i64 0#i64 = some 3#i64 := by decide
  have c2 : I64.checked_mul 5#i64 (core.convert.num.FromI64I8.from 1#i8) = some 5#i64 := by decide
  have c3 : I64.checked_sub 3#i64 5#i64 = some (-2)#i64 := by decide
  have c4 : I64.checked_sub 0#i64 (-2)#i64 = some 2#i64 := by decide
  have c5 : (3#i64 >= 0#i64) = True := by decide
  have c6 : ((-2)#i64 < 0#i64) = True := by decide
  unfold sigma_delta.sd_step
  simp only [lift, bind_tc_ok, c1, c5, if_true, c2, c3, c4, c6]

/-- `w − a·s` overflows at `(i64::MAX, 0, i64::MIN)` with `s = +1`: the step returns `none`. -/
theorem sd_step_witness_none :
    sigma_delta.sd_step 9223372036854775807#i64 0#i64 (-9223372036854775808)#i64 = ok none := by
  have c1 : I64.checked_add 9223372036854775807#i64 0#i64 = some 9223372036854775807#i64 := by
    decide
  have c2 : I64.checked_mul (-9223372036854775808)#i64 (core.convert.num.FromI64I8.from 1#i8) =
      some (-9223372036854775808)#i64 := by decide
  have c3 : I64.checked_sub 9223372036854775807#i64 (-9223372036854775808)#i64 = none := by decide
  have c5 : (9223372036854775807#i64 >= 0#i64) = True := by decide
  unfold sigma_delta.sd_step
  simp only [lift, bind_tc_ok, c1, c5, if_true, c2, c3]

/-- `[3]`. -/
def sdWitW : Slice I64 := Slice.from [3#i64] (by scalar_tac)
/-- `[i64::MAX]`. -/
def sdWitWMax : Slice I64 := Slice.from [9223372036854775807#i64] (by scalar_tac)
/-- The zeroed output buffer `[0]`. -/
def sdWitS0 : Slice I8 := Slice.from [0#i8] (by scalar_tac)
/-- The output buffer after one `+1` sign, `[1]`. -/
def sdWitS1 : Slice I8 := Slice.from [1#i8] (by scalar_tac)

theorem sd_usize_succ_zero : (0#usize + 1#usize : Result Usize) = ok 1#usize := by
  obtain ⟨z, hz, hv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac))
  have : z = 1#usize := by apply UScalar.eq_of_val_eq; simp_all
  rw [hz, this]

theorem sd_len_witW : Slice.len sdWitW = 1#usize := by
  apply UScalar.eq_of_val_eq; rw [Slice.len_val]; simp [sdWitW]
theorem sd_len_witWMax : Slice.len sdWitWMax = 1#usize := by
  apply UScalar.eq_of_val_eq; rw [Slice.len_val]; simp [sdWitWMax]
theorem sd_len_witS0 : Slice.len sdWitS0 = 1#usize := by
  apply UScalar.eq_of_val_eq; rw [Slice.len_val]; simp [sdWitS0]

/-- Success branch: `w = [3]`, `a = 5` returns `(some (−2, 2), [1])`. -/
theorem sd_binary_block_witness_some :
    sigma_delta.sd_binary_block sdWitW 5#i64 sdWitS0 =
      ok (some ((-2)#i64, 2#i64), sdWitS1) := by
  have hl : sigma_delta.sd_binary_block_loop sdWitW 5#i64 sdWitS0 1#usize 0#i64 0#i64
      0#usize true = ok (sdWitS1, (-2)#i64, 2#i64, true) := by
    unfold sigma_delta.sd_binary_block_loop
    rw [loop.eq_1]
    simp [sigma_delta.sd_binary_block_loop.body, sd_step_witness_some, sdWitW, sdWitS0, sdWitS1,
      Slice.index_usize, Slice.update, sd_usize_succ_zero]
    rw [loop.eq_1]
    simp
  unfold sigma_delta.sd_binary_block
  simp [sd_len_witW, sd_len_witS0, hl]

/-- Overflow branch: `w = [i64::MAX]`, `a = i64::MIN` returns `(none, [0])`. -/
theorem sd_binary_block_witness_none :
    sigma_delta.sd_binary_block sdWitWMax (-9223372036854775808)#i64 sdWitS0 =
      ok (none, sdWitS0) := by
  have hl : sigma_delta.sd_binary_block_loop sdWitWMax (-9223372036854775808)#i64 sdWitS0
      1#usize 0#i64 0#i64 0#usize true = ok (sdWitS0, 0#i64, 0#i64, false) := by
    unfold sigma_delta.sd_binary_block_loop
    rw [loop.eq_1]
    simp [sigma_delta.sd_binary_block_loop.body, sd_step_witness_none, sdWitWMax, sdWitS0,
      Slice.index_usize, sd_usize_succ_zero]
    rw [loop.eq_1]
    simp
  unfold sigma_delta.sd_binary_block
  simp [sd_len_witWMax, sd_len_witS0, hl]

/-- Joint witness: both branches, and the shape-mismatch branch, are reachable on the executed
body, and the success block meets the canonical `Numeric.SigmaDelta` values via
`sd_binary_block_spec`. -/
theorem sd_binary_block_witness :
    (∃ e peak s', sigma_delta.sd_binary_block sdWitW 5#i64 sdWitS0 = ok (some (e, peak), s')) ∧
    (∃ s', sigma_delta.sd_binary_block sdWitWMax (-9223372036854775808)#i64 sdWitS0 = ok (none, s')) ∧
    (∃ s', sigma_delta.sd_binary_block sdWitW 5#i64
        (Slice.from [0#i8, 0#i8] (by scalar_tac)) = ok (none, s')) :=
  ⟨⟨_, _, _, sd_binary_block_witness_some⟩, ⟨_, sd_binary_block_witness_none⟩,
    ⟨_, sd_binary_block_shape _ _ _ (by simp [sdWitW])⟩⟩

end apeiron_word
