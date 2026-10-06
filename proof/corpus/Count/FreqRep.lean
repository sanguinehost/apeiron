-- Row: `p0-count-body-get-the-a-EF9n3OLyqnX2ouQQWL8w9C` (COUNT-BODY step 2d): the count table's representation and `count_aggregate`'s refinement.
import ApeironWord
import Count.Helpers
import Apeiron.Table.Basic
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
open Aeneas Aeneas.Std Result apeiron_word

/-!
# The count table's representation and group probe

This module proves, on the extracted Lean:
* Partial correctness fixpoint principle `loop_pc` and admissibility `pspec_admissible`.
* Bit-level lemmas for SSE2 masks (`u16_tz_spec`, `movemask_spec`, `group_tag_mask_spec`).
* Group probe specifications (`home_group_spec`, `descend_group_spec`, `match_empty_spec`, `match_tag_spec`).
* Line geometry `LineGeom` and probe specifications (`probe_inner_spec`, `probe_outer_spec`, `probe_tagged_spec`, `probe_spec`).
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.CountAggregate
open Lean.Order Aeneas.Data.Coinductive

unseal Result in
theorem pspec_admissible {α : Type} (P : α → Prop) :
    admissible (fun (x : Result α) => ∀ a, x = .ok a → P a) := by
  intro c hchain h a ha
  by_cases hex : ∃ t, c t ∧ t ≠ (Result.div : Result α)
  · obtain ⟨t, ct, tne⟩ := hex
    have hle : t ⊑ CCPO.csup hchain := le_csup hchain ct
    rw [ha] at hle
    have hu : t ⊑ (Result.ok a : Result α) := hle
    have hu' := ITree.le_unfold t (Result.ok a) ▸ hu
    rcases hu' with h1 | ⟨r, h1, h2⟩ | ⟨i, t1', t2', h1, h2, _⟩
    · exact absurd h1 tne
    · have : r = a := by
        exact (Aeneas.Data.Coinductive.ret_inj.mp h2).symm
      subst this
      exact h t ct r h1
    · exact absurd h2 (by intro hh; exact ok_not_vis hh)
  · have hbot : CCPO.csup hchain = (Result.div : Result α) := by
      have hle : CCPO.csup hchain ⊑ (Result.div : Result α) :=
        csup_le hchain (fun y cy => by
          have : y = (Result.div : Result α) := by
            by_contra hne
            exact hex ⟨y, cy, hne⟩
          rw [this]
          exact PartialOrder.rel_refl)
      exact ITree.le_div_is_div _ hle
    rw [hbot] at ha
    exact absurd ha div_not_ok


theorem bind_inv {α β : Type} {e : Result α} {f : α → Result β} {r : β}
    (h : (do let a ← e; f a) = ok r) : ∃ a, e = ok a ∧ f a = ok r := by
  induction e using Result.cases with
  | ret a => exact ⟨a, rfl, by simpa using h⟩
  | vis i k =>
    exfalso
    simp [Bind.bind, Aeneas.Std.bind_vis] at h
  | div => simp at h

/-- Partial correctness of `loop`: if the loop returns `ok r`, every state it visits satisfies `inv`
and `r` satisfies `post`.  No termination or non-failure is assumed. -/
theorem loop_pc {α β : Type} (inv : α → Prop) (post : β → Prop)
    (body : α → Result (ControlFlow α β))
    (hB : ∀ x, inv x → ∀ cf, body x = ok cf →
      match cf with | .done y => post y | .cont x' => inv x')
    (x : α) (hx : inv x) (r : β) (h : loop body x = ok r) : post r := by
  have key : ∀ x, ∀ r, loop body x = ok r → (inv x → post r) := by
    refine loop.fixpoint_induct body (fun f => ∀ x, ∀ r, f x = ok r → (inv x → post r)) ?_ ?_
    · apply admissible_pi_apply (fun x (y : Result β) => ∀ r, y = ok r → (inv x → post r))
      intro x
      exact pspec_admissible (fun r => inv x → post r)
    · intro g ih x r h hx
      obtain ⟨cf, hb, hk⟩ := bind_inv h
      have := hB x hx cf hb
      cases cf with
      | cont x' => exact ih x' r hk this
      | done y =>
        have : y = r := Result.ok_injective hk
        subst this
        exact ‹post y›
  exact key x r h hx


/-! ## Bit-level lemmas for the SSE2 masks -/

theorem tz_go_spec {w : Nat} (x : BitVec w) :
    ∀ fuel acc, (∃ k, k < fuel ∧ x.getLsbD (acc + k) = true) →
      acc ≤ Aeneas.Std.BitVec.trailingZeros.go x fuel acc ∧
      x.getLsbD (Aeneas.Std.BitVec.trailingZeros.go x fuel acc) = true ∧
      ∀ k, acc ≤ k → k < Aeneas.Std.BitVec.trailingZeros.go x fuel acc → x.getLsbD k = false := by
  intro fuel
  induction fuel with
  | zero => intro acc ⟨k, hk, _⟩; omega
  | succ fuel ih =>
    intro acc ⟨k, hk, hb⟩
    by_cases h : x.getLsbD acc = true
    · have e : Aeneas.Std.BitVec.trailingZeros.go x (fuel+1) acc = acc := by
        simp [Aeneas.Std.BitVec.trailingZeros.go, h]
      rw [e]
      exact ⟨le_refl _, h, fun k h1 h2 => by omega⟩
    · have e : Aeneas.Std.BitVec.trailingZeros.go x (fuel+1) acc
          = Aeneas.Std.BitVec.trailingZeros.go x fuel (acc+1) := by
        simp [Aeneas.Std.BitVec.trailingZeros.go, h]
      rw [e]
      have hk0 : k ≠ 0 := by
        intro h0; subst h0; simp at hb; exact h hb
      obtain ⟨a1, a2, a3⟩ := ih (acc + 1) ⟨k - 1, by omega, by
        have : acc + 1 + (k - 1) = acc + k := by omega
        rw [this]; exact hb⟩
      refine ⟨by omega, a2, ?_⟩
      intro j hj1 hj2
      by_cases hj : j = acc
      · subst hj; simpa using h
      · exact a3 j (by omega) hj2

theorem u16_tz_spec (m : U16) (hm : m.val ≠ 0) :
    (core.num.U16.trailing_zeros m) = ok ⟨Aeneas.Std.BitVec.trailingZeros m.bv⟩ ∧
    ∀ tz : Nat, tz = Aeneas.Std.BitVec.trailingZeros m.bv →
      tz < 16 ∧ m.val.testBit tz = true ∧ ∀ k, k < tz → m.val.testBit k = false := by
  refine ⟨rfl, ?_⟩
  intro tz htz
  have hlt : m.val < 2^16 := by have := m.hBounds; simpa [UScalarTy.numBits] using this
  have hex : ∃ k, k < 16 ∧ m.bv.getLsbD (0 + k) = true := by
    by_contra hne
    push Not at hne
    apply hm
    have : m.val = 0 := by
      apply Nat.eq_of_testBit_eq
      intro i
      by_cases hi : i < 16
      · have := hne i hi
        have h2 : m.bv.getLsbD i = false := by simpa using this
        have : m.val.testBit i = m.bv.getLsbD i := (BitVec.testBit_toNat m.bv)
        simp [this, h2]
      · rw [Nat.testBit_eq_false_of_lt (by
          calc m.val < 2^16 := hlt
            _ ≤ 2^i := Nat.pow_le_pow_right (by norm_num) (by omega))]
        simp
    exact this
  obtain ⟨a1, a2, a3⟩ := tz_go_spec m.bv 16 0 hex
  have e : tz = Aeneas.Std.BitVec.trailingZeros.go m.bv 16 0 := htz
  rw [← e] at a1 a2 a3
  have hbit : ∀ k, m.val.testBit k = m.bv.getLsbD k := fun k => BitVec.testBit_toNat m.bv
  refine ⟨?_, by rw [hbit]; exact a2, fun k hk => by rw [hbit]; exact a3 k (by omega) hk⟩
  by_contra hge
  have : m.val.testBit tz = false := Nat.testBit_eq_false_of_lt (by
    calc m.val < 2^16 := hlt
      _ ≤ 2^tz := Nat.pow_le_pow_right (by norm_num) (by omega))
  rw [hbit] at this
  simp [this] at a2

/-- Clearing the lowest set bit. -/
theorem clear_lowest (m t : Nat) (h1 : m.testBit t = true) (h2 : ∀ k, k < t → m.testBit k = false) (k : Nat) :
    (m &&& (m - 1)).testBit k = (m.testBit k && decide (k ≠ t)) := by
  have hmod : m % 2^(t+1) = 2^t := by
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_mod_two_pow, Nat.testBit_two_pow]
    by_cases hi : i < t
    · simp [h2 i hi]; omega
    · by_cases hit : i = t
      · subst hit; simp [h1]
      · have : ¬ i < t + 1 := by omega
        simp [this]; omega
  obtain ⟨q, hq⟩ : ∃ q, q = m / 2^(t+1) := ⟨_, rfl⟩
  have hm : m = 2^(t+1) * q + 2^t := by
    have := Nat.div_add_mod m (2^(t+1))
    rw [hmod, ← hq] at this; omega
  have hpos : 0 < 2^t := Nat.two_pow_pos t
  have hm1 : m - 1 = 2^(t+1) * q + (2^t - 1) := by omega
  have hb : 2^t - 1 < 2^(t+1) := by
    have : 2^(t+1) = 2 * 2^t := by ring
    omega
  have e1 : m.testBit k = if k < t+1 then (2^t).testBit k else q.testBit (k - (t+1)) := by
    conv_lhs => rw [hm]
    rw [Nat.testBit_two_pow_mul_add q (by
      have : 2^(t+1) = 2 * 2^t := by ring
      omega) k]
  have e2 : (m-1).testBit k = if k < t+1 then (2^t - 1).testBit k else q.testBit (k - (t+1)) := by
    rw [hm1, Nat.testBit_two_pow_mul_add q hb k]
  rw [Nat.testBit_and, e1, e2, Nat.testBit_two_pow_sub_one, Nat.testBit_two_pow]
  by_cases hk : k < t
  · have h' : k < t + 1 := by omega
    simp [h', hk]; intro h; omega
  · by_cases hkt : k = t
    · subst hkt; simp
    · have : ¬ k < t + 1 := by omega
      simp [this, hkt]


open apeiron_word

theorem group_val : hashtrinity.freq.GROUP.val = 16 := by
  unfold hashtrinity.freq.GROUP
  rfl

theorem mm_fold (P : Nat → Bool) :
    ∀ n, n ≤ 16 →
      ((List.range n).foldl (fun acc k => if k < 16 then (if P k then acc + 2 ^ k else acc) else acc) 0 < 2 ^ n ∧
      ∀ k, k < n → (((List.range n).foldl (fun acc k => if k < 16 then (if P k then acc + 2 ^ k else acc) else acc) 0).testBit k = P k)) := by
  intro n
  induction n with
  | zero => intro _; simp
  | succ n ih =>
    intro hn
    obtain ⟨ih1, ih2⟩ := ih (by omega)
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    have hn16 : n < 16 := by omega
    simp only [hn16, if_true]
    set F := (List.range n).foldl (fun acc k => if k < 16 then (if P k then acc + 2 ^ k else acc) else acc) 0 with hF
    by_cases hP : P n = true
    · simp only [hP, if_true]
      refine ⟨by have : 2^(n+1) = 2 * 2^n := by ring
                 omega, ?_⟩
      intro k hk
      by_cases hkn : k = n
      · subst hkn
        have : F + 2^k = 2^k + F := by omega
        rw [this, Nat.testBit_two_pow_add_eq, Nat.testBit_lt_two_pow ih1]; simp [hP]
      · have : F + 2^n = 2^n + F := by omega
        rw [this, Nat.testBit_two_pow_add_gt (by omega)]
        exact ih2 k (by omega)
    · have hP' : P n = false := by simpa using hP
      simp only [hP', Bool.false_eq_true, if_false]
      refine ⟨by have : 2^(n+1) = 2 * 2^n := by ring
                 omega, ?_⟩
      intro k hk
      by_cases hkn : k = n
      · subst hkn; rw [Nat.testBit_lt_two_pow ih1, hP']
      · exact ih2 k (by omega)


theorem movemask_spec (a : core.core_arch.x86.__m128i) (i : I32)
    (h : core.core_arch.x86.sse2._mm_movemask_epi8 a = ok i) :
    (IScalar.hcast .U16 i).val < 2^16 ∧
    ∀ k (hk : k < 16), (IScalar.hcast .U16 i).val.testBit k = decide (128 ≤ (a.lane ⟨k, hk⟩).val) := by
  unfold core.core_arch.x86.sse2._mm_movemask_epi8 at h
  have hi : i = ⟨BitVec.ofNat 32 ((List.range 16).foldl
    (fun acc k => if h : k < 16 then (if 128 ≤ (a.lane ⟨ k, h ⟩).val then acc + 2 ^ k else acc)
      else acc) 0)⟩ := (Result.ok_injective h).symm
  set P : Nat → Bool := fun k => if hk : k < 16 then decide (128 ≤ (a.lane ⟨k, hk⟩).val) else false with hP
  have hfun : (fun acc k => if h : k < 16 then (if 128 ≤ (a.lane ⟨ k, h ⟩).val then acc + 2 ^ k else acc)
      else acc) = (fun acc k => if k < 16 then (if P k then acc + 2 ^ k else acc) else acc) := by
    funext acc k
    by_cases hk : k < 16
    · simp [hk, hP]
    · simp [hk]
  rw [hfun] at hi
  obtain ⟨b1, b2⟩ := mm_fold P 16 (le_refl _)
  set S := (List.range 16).foldl (fun acc k => if k < 16 then (if P k then acc + 2 ^ k else acc) else acc) 0 with hS
  have hval : (IScalar.hcast .U16 i).val = S := by
    rw [hi]
    have : S < 2^16 := b1
    simp only [IScalar.hcast, UScalar.val]
    have e16 : UScalarTy.U16.numBits = 16 := rfl
    rw [e16]
    simp [BitVec.toNat_signExtend]
    omega
  rw [hval]
  refine ⟨b1, fun k hk => ?_⟩
  rw [b2 k hk, hP]
  simp [hk]


/-! ## Inversion lemmas: what `= ok` says about a checked scalar operation -/

theorem uadd_ok {ty} {x y z : UScalar ty} (h : x + y = ok z) :
    z.val = x.val + y.val ∧ x.val + y.val < 2^ty.numBits := by
  have := UScalar.add_equiv x y
  simp only [h, Result.match.ok] at this
  exact ⟨this.2.1, this.1⟩

theorem usub_ok {ty} {x y z : UScalar ty} (h : x - y = ok z) :
    x.val = z.val + y.val := by
  have := UScalar.sub_equiv x y
  simp only [h, Result.match.ok] at this
  exact this.2.1

theorem umul_ok {ty} {x y z : UScalar ty} (h : x * y = ok z) :
    z.val = x.val * y.val ∧ x.val * y.val ≤ UScalar.max ty := by
  have := UScalar.mul_equiv x y
  have h' : UScalar.mul x y = ok z := h
  simp only [h', Result.match.ok] at this
  exact ⟨this.2.1, this.1⟩

theorem udiv_ok {ty} {x y z : UScalar ty} (h : x / y = ok z) :
    y.val ≠ 0 ∧ z.val = x.val / y.val := by
  have h' : UScalar.div x y = ok z := h
  unfold UScalar.div at h'
  by_cases hy0 : y.bv = 0#ty.numBits
  · simp [hy0] at h'
  · have hy : (y.bv != 0) = true := by simpa using hy0
    simp only [hy, if_true] at h'
    have hz : z = ⟨BitVec.udiv x.bv y.bv⟩ := (Result.ok_injective h').symm
    refine ⟨?_, ?_⟩
    · intro h0
      apply hy0
      apply BitVec.eq_of_toNat_eq
      simpa using h0
    · rw [hz]
      show (BitVec.udiv x.bv y.bv).toNat = x.bv.toNat / y.bv.toNat
      exact BitVec.toNat_udiv

theorem ushr_ok {ty tys} {x : UScalar ty} {s : UScalar tys} {z : UScalar ty}
    (h : x >>> s = ok z) : s.val < ty.numBits ∧ z.val = x.val >>> s.val := by
  have h' : UScalar.shiftRight_UScalar x s = ok z := h
  unfold UScalar.shiftRight_UScalar UScalar.shiftRight at h'
  by_cases hs : s.val < ty.numBits
  · simp only [hs, if_true] at h'
    have hz : z = ⟨x.bv.ushiftRight s.val⟩ := (Result.ok_injective h').symm
    refine ⟨hs, ?_⟩
    rw [hz]
    show (x.bv.ushiftRight s.val).toNat = x.bv.toNat >>> s.val
    exact BitVec.toNat_ushiftRight _ _
  · simp [hs] at h'

theorem ushl_ok {ty tys} {x : UScalar ty} {s : UScalar tys} {z : UScalar ty}
    (h : x <<< s = ok z) : s.val < ty.numBits ∧ z.val = (x.val * 2^s.val) % 2^ty.numBits := by
  have h' : UScalar.shiftLeft_UScalar x s = ok z := h
  unfold UScalar.shiftLeft_UScalar UScalar.shiftLeft at h'
  by_cases hs : s.val < ty.numBits
  · simp only [hs, if_true] at h'
    have hz : z = ⟨x.bv.shiftLeft s.val⟩ := (Result.ok_injective h').symm
    refine ⟨hs, ?_⟩
    rw [hz]
    show (x.bv.shiftLeft s.val).toNat = (x.bv.toNat * 2^s.val) % 2^ty.numBits
    exact BitVec.toNat_shiftLeft.trans (by rw [Nat.shiftLeft_eq])
  · simp [hs] at h'

theorem lane_set1 (tag : U8) : IScalar.hcast .U8 (UScalar.hcast .I8 tag) = tag := by
  cases tag with | mk bv =>
  simp [IScalar.hcast, UScalar.hcast]

theorem as_ptr_u8 (s : Slice U8) : core.slice.Slice.as_ptr s = ok ⟨s.val, 0⟩ := by
  simp [core.slice.Slice.as_ptr]

theorem u8_255 : (⟨BitVec.ofNat 8 255⟩ : U8).val = 255 := by
  show (BitVec.ofNat 8 255).toNat = 255
  decide

theorem u8_0 : (⟨BitVec.ofNat 8 0⟩ : U8).val = 0 := by
  show (BitVec.ofNat 8 0).toNat = 0
  decide

theorem ptrElemSize_u8 : ptrElemSize U8 = 1 := by simp [ptrElemSize]

/-- `ptr.add` on the `u8` control plane: the byte offset advances by `n` (`ptrElemSize U8 = 1`). -/
theorem ptr_add_inv (p : ConstRawPtr U8) (n : Usize) (q : ConstRawPtr U8)
    (h : core.ptr.const_ptr.RawPtrConstT.add p n = ok q) :
    p.off + n.val ≤ p.bytes.length ∧ q = ⟨p.bytes, p.off + n.val⟩ := by
  unfold core.ptr.const_ptr.RawPtrConstT.add at h
  rw [ptrElemSize_u8, Nat.mul_one] at h
  by_cases hc : p.off + n.val ≤ p.bytes.length
  · rw [if_pos hc] at h
    exact ⟨hc, (Result.ok_injective h).symm⟩
  · rw [if_neg hc] at h
    exact absurd h (by intro hh; exact ok_not_fail hh.symm)

theorem group_tag_mask_spec (ctrl : Slice U8) (base : Usize) (tag : U8) (m : U16)
    (h : hashtrinity.freq.group_tag_mask ctrl base tag = ok m) :
    base.val + 16 ≤ ctrl.length ∧ m.val < 2^16 ∧
    ∀ k, k < 16 → ∃ b, ctrl.val[base.val + k]? = some b ∧
      (m.val.testBit k = true ↔ b.val = tag.val) := by
  unfold hashtrinity.freq.group_tag_mask at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨u, hu, h⟩ := bind_inv h
  rw [as_ptr_u8] at h
  simp only [bind_tc_ok] at h
  obtain ⟨ptr, hptr, h⟩ := bind_inv h
  obtain ⟨p1, hp1, h⟩ := bind_inv h
  obtain ⟨mi, hmi, h⟩ := bind_inv h
  obtain ⟨i2, hi2, h⟩ := bind_inv h
  obtain ⟨mi1, hmi1, h⟩ := bind_inv h
  obtain ⟨mi2, hmi2, h⟩ := bind_inv h
  obtain ⟨i3, hi3, h⟩ := bind_inv h
  have hm : IScalar.hcast .U16 i3 = m := Result.ok_injective h
  have hiv := (uadd_ok hi).1
  rw [group_val] at hiv
  have hle : i ≤ ctrl.len := by
    unfold massert at hu
    by_contra hc
    simp [hc] at hu
  have hle' : i.val ≤ ctrl.length := by
    have := (UScalar.le_equiv i ctrl.len).mp hle
    simpa [Slice.len_val] using this
  have hbl : base.val + 16 ≤ ctrl.length := by omega
  obtain ⟨hptr1, hptr2⟩ := ptr_add_inv _ _ _ hptr
  have hp1' : p1 = ⟨ptr.bytes, ptr.off⟩ := (Result.ok_injective hp1).symm
  have hmi' : mi = ⟨fun k => p1.bytes.getD (p1.off + k.val) (⟨BitVec.ofNat 8 0⟩ : U8)⟩ := by
    unfold core.core_arch.x86.sse2._mm_loadu_si128 at hmi
    have hc : p1.off + 16 ≤ p1.bytes.length := by
      rw [hp1', hptr2]; simp; omega
    rw [if_pos hc] at hmi
    exact (Result.ok_injective hmi).symm
  have hi2' : i2 = UScalar.hcast .I8 tag := (Result.ok_injective hi2).symm
  have hmi1' : mi1 = ⟨fun _ => IScalar.hcast .U8 i2⟩ := (Result.ok_injective hmi1).symm
  have hmi2' : mi2 = ⟨fun k => if (mi.lane k).val = (mi1.lane k).val then (⟨BitVec.ofNat 8 255⟩ : U8) else (⟨BitVec.ofNat 8 0⟩ : U8)⟩ :=
    (Result.ok_injective hmi2).symm
  obtain ⟨mm1, mm2⟩ := movemask_spec mi2 i3 hi3
  rw [hm] at mm1 mm2
  refine ⟨hbl, mm1, ?_⟩
  intro k hk
  have hkl : base.val + k < ctrl.length := by omega
  refine ⟨ctrl.val[base.val + k]'hkl, List.getElem?_eq_getElem hkl, ?_⟩
  rw [mm2 k hk]
  have hlane : (mi.lane ⟨k, hk⟩) = ctrl.val[base.val + k]'hkl := by
    rw [hmi']
    simp only [hp1', hptr2, Nat.zero_add]
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkl]
  have hl2 : (mi1.lane ⟨k, hk⟩) = tag := by rw [hmi1', hi2', lane_set1]
  rw [hmi2']
  simp only [hlane, hl2]
  by_cases he : (ctrl.val[base.val + k]'hkl).val = tag.val
  · simp [he, u8_255]
  · simp [he, u8_0]


theorem match_empty_spec {V : Type} (self : hashtrinity.freq.FreqLine V) (g : Usize) (m : U16)
    (h : hashtrinity.freq.FreqLine.match_empty self g = ok m) :
    g.val * 16 + 16 ≤ self.ctrl.length ∧ m.val < 2^16 ∧
    ∀ k, k < 16 → ∃ b, self.ctrl.val[g.val * 16 + k]? = some b ∧
      (m.val.testBit k = true ↔ 128 ≤ b.val) := by
  unfold hashtrinity.freq.FreqLine.match_empty at h
  obtain ⟨base, hbase, h⟩ := bind_inv h
  obtain ⟨p, hp, h⟩ := bind_inv h
  obtain ⟨ptr, hptr, h⟩ := bind_inv h
  obtain ⟨p1, hp1, h⟩ := bind_inv h
  obtain ⟨mi, hmi, h⟩ := bind_inv h
  obtain ⟨i, hi, h⟩ := bind_inv h
  have hm : IScalar.hcast .U16 i = m := Result.ok_injective h
  have hbv := (umul_ok hbase)
  rw [group_val] at hbv
  rw [as_ptr_u8] at hp
  have hp' : p = ⟨self.ctrl.val, 0⟩ := (Result.ok_injective hp).symm
  subst hp'
  obtain ⟨hptr1, hptr2⟩ := ptr_add_inv _ _ _ hptr
  have hp1' : p1 = ⟨ptr.bytes, ptr.off⟩ := (Result.ok_injective hp1).symm
  have hbl : g.val * 16 + 16 ≤ self.ctrl.length := by
    have hc : p1.off + 16 ≤ p1.bytes.length := by
      by_contra hc
      unfold core.core_arch.x86.sse2._mm_loadu_si128 at hmi
      rw [if_neg hc] at hmi
      exact absurd hmi (by intro hh; exact ok_not_fail hh.symm)
    rw [hp1', hptr2] at hc
    dsimp only at hc hptr1
    have e1 : (↑self.ctrl : List U8).length = self.ctrl.length := rfl
    omega
  have hmi' : mi = ⟨fun k => p1.bytes.getD (p1.off + k.val) (⟨BitVec.ofNat 8 0⟩ : U8)⟩ := by
    unfold core.core_arch.x86.sse2._mm_loadu_si128 at hmi
    have hc : p1.off + 16 ≤ p1.bytes.length := by
      rw [hp1', hptr2]
      dsimp only
      have e1 : (↑self.ctrl : List U8).length = self.ctrl.length := rfl
      omega
    rw [if_pos hc] at hmi
    exact (Result.ok_injective hmi).symm
  obtain ⟨mm1, mm2⟩ := movemask_spec mi i hi
  rw [hm] at mm1 mm2
  refine ⟨hbl, mm1, ?_⟩
  intro k hk
  have hkl : g.val * 16 + k < self.ctrl.length := by omega
  refine ⟨self.ctrl.val[g.val * 16 + k]'hkl, List.getElem?_eq_getElem hkl, ?_⟩
  rw [mm2 k hk]
  have hlane : (mi.lane ⟨k, hk⟩) = self.ctrl.val[g.val * 16 + k]'hkl := by
    rw [hmi']
    simp only [hp1', hptr2, Nat.zero_add, hbv.1]
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkl]
  rw [hlane]
  simp

theorem match_tag_spec {V : Type} (self : hashtrinity.freq.FreqLine V) (g : Usize) (tag : U8) (m : U16)
    (h : hashtrinity.freq.FreqLine.match_tag self g tag = ok m) :
    g.val * 16 + 16 ≤ self.ctrl.length ∧ m.val < 2^16 ∧
    ∀ k, k < 16 → ∃ b, self.ctrl.val[g.val * 16 + k]? = some b ∧
      (m.val.testBit k = true ↔ b.val = tag.val) := by
  unfold hashtrinity.freq.FreqLine.match_tag at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  have hbv := (umul_ok hi)
  rw [group_val] at hbv
  obtain ⟨a1, a2, a3⟩ := group_tag_mask_spec _ _ _ _ h
  rw [hbv.1] at a1 a3
  exact ⟨by omega, a2, a3⟩


/-! ## Semantic objects -/

def PHIN : Nat := 11400714819323198485
def EMPTYN : Nat := 2^64 - 1
def pN (f : Nat) : Nat := f * PHIN % 2^64
def tagN (f : Nat) : Nat := pN f / 2^33 % 128

/-- The group visited at probe step `j`: the PHI-descent read at group granularity
(`FreqLine::home_group` for `j = 0`, `FreqLine::descend_group` after), `gb = bits - 4`. -/
def gseq (gb p j : Nat) : Nat :=
  if gb = 0 then 0
  else if gb * (j + 1) ≤ 64 then (p >>> (64 - gb * (j + 1))) % 2^gb
  else ((p >>> (64 - gb * (64 / gb))) + (j - 64 / gb) + 1) % 2^gb

theorem cast_and_mask (x gb N : Nat) (hgb : gb ≤ N) : (x % 2^N) &&& (2^gb - 1) = x % 2^gb := by
  rw [Nat.and_two_pow_sub_one_eq_mod]
  exact Nat.mod_mod_of_dvd _ (Nat.pow_dvd_pow 2 hgb)

theorem usize_cast_val (x : U64) : (UScalar.cast .Usize x).val = x.val % 2^System.Platform.numBits := by
  rw [UScalar.cast_val_eq]
  rfl

theorem home_group_spec {V : Type} (self : hashtrinity.freq.FreqLine V) (p : U64) (g : Usize)
    (h : hashtrinity.freq.FreqLine.home_group self p = ok g) (gb : Nat)
    (hm : self.gmask.val = 2^gb - 1) (hgb : gb ≤ System.Platform.numBits) :
    g.val = (p.val >>> self.gshift.val) % 2^gb := by
  unfold hashtrinity.freq.FreqLine.home_group at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨i1, hi1, h⟩ := bind_inv h
  have hg : g = i1 &&& self.gmask := (Result.ok_injective h).symm
  have hi1' : i1 = UScalar.cast .Usize i := (Result.ok_injective hi1).symm
  have e1 := (ushr_ok hi).2
  rw [hg, UScalar.val_and, hm, hi1', usize_cast_val, e1]
  exact cast_and_mask _ _ _ hgb

theorem nb_ge_aux : 32 ≤ System.Platform.numBits := Apeiron.CountHelpers.nb_ge

theorem one_u32 : (1#u32).val = 1 := rfl

theorem four_val : (4#u32).val = 4 := rfl
theorem u64_64_val : (64#u32).val = 64 := rfl

theorem descend_group_spec {V : Type} (self : hashtrinity.freq.FreqLine V) (p : U64) (j : U32)
    (g : Usize) (h : hashtrinity.freq.FreqLine.descend_group self p j = ok g)
    (hm : self.gmask.val = 2^(self.bits.val - 4) - 1)
    (hgb : self.bits.val - 4 ≤ System.Platform.numBits) :
    g.val = gseq (self.bits.val - 4) p.val j.val := by
  unfold hashtrinity.freq.FreqLine.descend_group at h
  obtain ⟨gbits, hgbits, h⟩ := bind_inv h
  have hb := usub_ok hgbits
  rw [four_val] at hb
  have hgv : gbits.val = self.bits.val - 4 := by omega
  rw [← hgv] at hm hgb ⊢
  by_cases hz : gbits = 0#u32
  · rw [if_pos hz] at h
    have : g = 0#usize := (Result.ok_injective h).symm
    subst this
    have : gbits.val = 0 := by rw [hz]; rfl
    simp [gseq, this]
  · rw [if_neg hz] at h
    have hz' : gbits.val ≠ 0 := by
      intro h0; apply hz; apply UScalar.eq_of_val_eq; simpa using h0
    obtain ⟨i, hi, h⟩ := bind_inv h
    obtain ⟨used, hused, h⟩ := bind_inv h
    have hiv := (uadd_ok hi).1
    rw [one_u32] at hiv
    have huv := (umul_ok hused).1
    by_cases hu : used ≤ 64#u32
    · rw [if_pos hu] at h
      have hu' : used.val ≤ 64 := by
        have := (UScalar.le_equiv used 64#u32).mp hu
        simpa using this
      obtain ⟨i1, hi1, h⟩ := bind_inv h
      obtain ⟨i2, hi2, h⟩ := bind_inv h
      obtain ⟨i3, hi3, h⟩ := bind_inv h
      have hg : g = i3 &&& self.gmask := (Result.ok_injective h).symm
      have hi3' : i3 = UScalar.cast .Usize i2 := (Result.ok_injective hi3).symm
      have e1 := usub_ok hi1
      rw [u64_64_val] at e1
      have e2 := (ushr_ok hi2).2
      have hcond : gbits.val * (j.val + 1) ≤ 64 := by rw [← hiv, ← huv]; exact hu'
      rw [hg, UScalar.val_and, hm, hi3', usize_cast_val, e2]
      unfold gseq
      rw [if_neg hz', if_pos hcond]
      have : i1.val = 64 - gbits.val * (j.val + 1) := by rw [← hiv, ← huv]; omega
      rw [this]
      exact cast_and_mask _ _ _ hgb
    · rw [if_neg hu] at h
      have hu' : ¬ used.val ≤ 64 := by
        intro hc; apply hu
        exact (UScalar.le_equiv used 64#u32).mpr (by simpa using hc)
      have hcond : ¬ gbits.val * (j.val + 1) ≤ 64 := by rw [← hiv, ← huv]; exact hu'
      obtain ⟨full, hfull, h⟩ := bind_inv h
      obtain ⟨i1, hi1, h⟩ := bind_inv h
      obtain ⟨i2, hi2, h⟩ := bind_inv h
      obtain ⟨i3, hi3, h⟩ := bind_inv h
      obtain ⟨i4, hi4, h⟩ := bind_inv h
      obtain ⟨i5, hi5, h⟩ := bind_inv h
      obtain ⟨i6, hi6, h⟩ := bind_inv h
      obtain ⟨i7, hi7, h⟩ := bind_inv h
      obtain ⟨i8, hi8, h⟩ := bind_inv h
      have hg : g = i8 &&& self.gmask := (Result.ok_injective h).symm
      have fullv := (udiv_ok hfull).2
      rw [u64_64_val] at fullv
      have i1v := (umul_ok hi1).1
      have i2v := usub_ok hi2
      rw [u64_64_val] at i2v
      have i3v := (ushr_ok hi3).2
      have i4' : i4 = UScalar.cast .Usize i3 := (Result.ok_injective hi4).symm
      have i5v := usub_ok hi5
      have i6' : i6 = UScalar.cast .Usize i5 := (Result.ok_injective hi6).symm
      have i7v := uadd_ok hi7
      have i8v := uadd_ok hi8
      have hN : (1#usize).val = 1 := rfl
      rw [hN] at i8v
      rw [hg, UScalar.val_and, hm]
      unfold gseq
      rw [if_neg hz', if_neg hcond]
      have hbit : gbits.val * (64 / gbits.val) ≤ 64 := Nat.mul_div_le 64 gbits.val
      have hi2n : i2.val = 64 - gbits.val * (64 / gbits.val) := by
        rw [fullv] at i1v; omega
      have hj : j.val - 64 / gbits.val = i5.val := by
        rw [fullv] at i5v; omega
      have hjb : i5.val < 2^32 := by
        have := i5.hBounds; simpa [UScalarTy.numBits] using this
      have i6v : i6.val = i5.val := by
        rw [i6', UScalar.cast_val_eq]
        apply Nat.mod_eq_of_lt
        have := nb_ge_aux
        calc i5.val < 2^32 := hjb
          _ ≤ _ := Nat.pow_le_pow_right (by norm_num) this
      have i4v : i4.val = (p.val >>> i2.val) % 2^System.Platform.numBits := by
        rw [i4', usize_cast_val, i3v]
      have hmod : i8.val % 2^gbits.val = ((p.val >>> i2.val) + i5.val + 1) % 2^gbits.val := by
        rw [i8v.1, i7v.1, i6v, i4v]
        have hd : 2^gbits.val ∣ 2^System.Platform.numBits := Nat.pow_dvd_pow 2 hgb
        have hme := ((Nat.mod_modEq (p.val >>> i2.val) (2^System.Platform.numBits)).of_dvd hd).add_right (i5.val + 1)
        have e1 : (p.val >>> i2.val) % 2^System.Platform.numBits + i5.val + 1
            = (p.val >>> i2.val) % 2^System.Platform.numBits + (i5.val + 1) := by omega
        have e2 : (p.val >>> i2.val) + i5.val + 1 = (p.val >>> i2.val) + (i5.val + 1) := by omega
        rw [e1, e2]
        exact hme
      rw [Nat.and_two_pow_sub_one_eq_mod, hmod, hi2n, hj]


/-! ## Line accessors -/

def foN {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) : Nat :=
  match l.slots.val[i]? with
  | some s => s.1.val
  | none => EMPTYN

def ctN {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) : Nat :=
  match l.ctrl.val[i]? with
  | some b => b.val
  | none => 128

theorem foN_of_get {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) (x : U64 × core.mem.maybe_uninit.MaybeUninit V)
    (h : l.slots.val[i]? = some x) : foN l i = x.1.val := by
  simp [foN, h]

theorem ctN_of_get {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) (b : U8)
    (h : l.ctrl.val[i]? = some b) : ctN l i = b.val := by
  simp [ctN, h]

theorem slice_index_ok {α : Type} (s : Slice α) (i : Usize) (x : α) (h : s.index_usize i = ok x) :
    s.val[i.val]? = some x := by
  unfold Slice.index_usize at h
  cases hh : s[i]? with
  | none => simp [hh] at h
  | some y =>
    simp [hh] at h
    subst h
    rfl

theorem get_unchecked_ok {α : Type} (s : Slice α) (i : Usize) (x : α)
    (h : core.slice.Slice.get_unchecked (core.slice.index.SliceIndexUsizeSlice α) s i = ok x) :
    s.val[i.val]? = some x := by
  exact slice_index_ok s i x h

/-! ## The inner (tag-confirm) loop -/

theorem probe_inner_spec {V : Type} (self : hashtrinity.freq.FreqLine V) (f : U64) (base : Usize)
    (M : U16) (f0 : Bool) (s0 : Usize)
    (r : hashtrinity.freq.FreqLine V × Bool × Usize)
    (h : hashtrinity.freq.FreqLine.probe_tagged_loop0_loop0 self f f0 s0 base M = ok r) :
    r.1 = self ∧
    ((r.2.1 = f0 ∧ r.2.2 = s0 ∧
        ∀ k, k < 16 → M.val.testBit k = true → foN self (base.val + k) ≠ f.val) ∨
     (r.2.1 = true ∧
        ∃ k, k < 16 ∧ M.val.testBit k = true ∧ foN self (base.val + k) = f.val ∧
          r.2.2.val = base.val + k)) := by
  unfold hashtrinity.freq.FreqLine.probe_tagged_loop0_loop0 at h
  have hM : M.val < 2^16 := by have := M.hBounds; simpa [UScalarTy.numBits] using this
  refine loop_pc (fun (st : Bool × Usize × U16) =>
      (st.1 = f0 ∧ st.2.1 = s0 ∧
        (∀ k, st.2.2.val.testBit k = true → M.val.testBit k = true) ∧
        (∀ k, k < 16 → M.val.testBit k = true → st.2.2.val.testBit k = false →
          foN self (base.val + k) ≠ f.val)) ∨
      (st.1 = true ∧ st.2.2.val = 0 ∧
        ∃ k, k < 16 ∧ M.val.testBit k = true ∧ foN self (base.val + k) = f.val ∧
          st.2.1.val = base.val + k))
    (fun (y : hashtrinity.freq.FreqLine V × Bool × Usize) => y.1 = self ∧
    ((y.2.1 = f0 ∧ y.2.2 = s0 ∧
        ∀ k, k < 16 → M.val.testBit k = true → foN self (base.val + k) ≠ f.val) ∨
     (y.2.1 = true ∧
        ∃ k, k < 16 ∧ M.val.testBit k = true ∧ foN self (base.val + k) = f.val ∧
          y.2.2.val = base.val + k))) _ ?_ (f0, s0, M) (Or.inl ⟨rfl, rfl, fun k hk => hk, fun k _ hk => by simp [hk]⟩) r h
  intro st hst cf hcf
  obtain ⟨fo, sl, m⟩ := st
  simp only [] at hst
  unfold hashtrinity.freq.FreqLine.probe_tagged_loop0_loop0.body at hcf
  dsimp only at hcf
  by_cases hm0 : m != 0#u16
  · rw [if_pos hm0] at hcf
    have hmne : m.val ≠ 0 := by
      intro h0
      have : m = 0#u16 := UScalar.eq_of_val_eq (h0.trans rfl)
      simp [this] at hm0
    obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
    obtain ⟨i1, hi1, hcf⟩ := bind_inv hcf
    obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
    obtain ⟨⟨i3, mu⟩, hi3, hcf⟩ := bind_inv hcf
    obtain ⟨htz1, htz2⟩ := u16_tz_spec m hmne
    have htzv : tz.val = Aeneas.Std.BitVec.trailingZeros m.bv := by
      have : (core.num.U16.trailing_zeros m) = ok tz := htz
      rw [htz1] at this
      have := (Result.ok_injective this).symm
      rw [this]
      have hT := (htz2 _ rfl).1
      show (BitVec.ofNat 32 (Aeneas.Std.BitVec.trailingZeros m.bv)).toNat = _
      rw [BitVec.toNat_ofNat]
      exact Nat.mod_eq_of_lt (by omega)
    obtain ⟨hz16, hzbit, hzlow⟩ := htz2 _ htzv
    have hi1v : i1.val = tz.val := by
      have : i1 = UScalar.cast .Usize tz := (Result.ok_injective hi1).symm
      rw [this, UScalar.cast_val_eq]
      apply Nat.mod_eq_of_lt
      have := nb_ge_aux
      have h2 : tz.val < 2^32 := by omega
      calc tz.val < 2^32 := h2
        _ ≤ _ := Nat.pow_le_pow_right (by norm_num) this
    have hi2v := (uadd_ok hi2).1
    have hget := get_unchecked_ok _ _ _ hi3
    have hfo : foN self i2.val = i3.val := foN_of_get _ _ _ hget
    have hi2' : i2.val = base.val + tz.val := by omega
    have hfo' : foN self (base.val + tz.val) = i3.val := by rw [← hi2']; exact hfo
    rcases hst with ⟨e1, e3, hsub, hproc⟩ | ⟨e1, e3, _⟩
    · by_cases hf : i3 = f
      · change (if i3 = f then _ else _) = _ at hcf
        rw [if_pos hf] at hcf
        have hcf' : cf = ControlFlow.cont (true, i2, 0#u16) := (Result.ok_injective hcf).symm
        subst hcf'
        refine Or.inr ⟨rfl, by simp, tz.val, hz16, hsub _ hzbit, ?_, ?_⟩
        · rw [hfo', hf]
        · simp; omega
      · change (if i3 = f then _ else _) = _ at hcf
        rw [if_neg hf] at hcf
        obtain ⟨i4, hi4, hcf⟩ := bind_inv hcf
        simp only [lift, bind_tc_ok] at hcf
        have hcf' : cf = ControlFlow.cont (fo, sl, (m &&& i4)) := by
          have := Result.ok_injective hcf
          exact this.symm
        subst hcf'
        have hi4v := usub_ok hi4
        have hone : (1#u16).val = 1 := rfl
        rw [hone] at hi4v
        have hand : (m &&& i4).val = m.val &&& (m.val - 1) := by
          rw [UScalar.val_and]; congr 1; omega
        refine Or.inl ⟨e1, e3, ?_, ?_⟩
        · intro k hk
          rw [hand, clear_lowest m.val tz.val hzbit hzlow k] at hk
          simp at hk
          exact hsub k hk.1
        · intro k hk hMk hbit
          rw [hand, clear_lowest m.val tz.val hzbit hzlow k] at hbit
          by_cases hmk : m.val.testBit k = true
          · have hkt : k = tz.val := by
              by_contra hne
              simp [hmk, hne] at hbit
            subst hkt
            rw [hfo']
            intro hh
            apply hf
            apply UScalar.eq_of_val_eq
            exact hh
          · simp at hmk
            exact hproc k hk hMk hmk
    · exfalso
      simp [e3] at hmne
  · rw [if_neg hm0] at hcf
    have hcf' : cf = ControlFlow.done (self, fo, sl) := (Result.ok_injective hcf).symm
    subst hcf'
    have hm0' : m.val = 0 := by simpa using hm0
    refine ⟨rfl, ?_⟩
    rcases hst with ⟨e1, e3, hsub, hproc⟩ | ⟨e1, e3, k, hk1, hk2, hk3, hk4⟩
    · refine Or.inl ⟨e1, e3, fun k hk hMk => hproc k hk hMk (by rw [hm0']; simp)⟩
    · exact Or.inr ⟨e1, k, hk1, hk2, hk3, hk4⟩


/-! ## Geometry of a line and the probe -/

def fullG {V : Type} (l : hashtrinity.freq.FreqLine V) (g : Nat) : Prop :=
  ∀ k, k < 16 → ctN l (16 * g + k) < 128

/-- Structural facts the group probe reads: sizes, the shift/mask, control bytes agreeing with slots,
distinct folds, and first-fit placement along the group sequence. -/
structure LineGeom {V : Type} (l : hashtrinity.freq.FreqLine V) : Prop where
  len_eq : l.ctrl.length = l.slots.length
  size : l.slots.length = 2 ^ l.bits.val
  bits4 : 4 ≤ l.bits.val
  gshift : l.gshift.val = if l.bits.val = 4 then 0 else 64 - (l.bits.val - 4)
  gmask : l.gmask.val = 2 ^ (l.bits.val - 4) - 1
  empty_ctrl : ∀ i, foN l i = EMPTYN → ctN l i = 128
  full_ctrl : ∀ i, foN l i ≠ EMPTYN → ctN l i = tagN (foN l i)
  distinct : ∀ i j, foN l i ≠ EMPTYN → foN l i = foN l j → i = j
  placed : ∀ i, foN l i ≠ EMPTYN → ∃ j, j < 16 ∧ gseq (l.bits.val - 4) (pN (foN l i)) j = i / 16 ∧
    ∀ j', j' < j → fullG l (gseq (l.bits.val - 4) (pN (foN l i)) j')

theorem LineGeom.gb_le {V : Type} {l : hashtrinity.freq.FreqLine V} (h : LineGeom l) :
    l.bits.val - 4 ≤ System.Platform.numBits := by
  have h1 : l.slots.length ≤ Usize.max := l.slots.property
  rw [h.size] at h1
  by_contra hc
  have : 2 ^ System.Platform.numBits ≤ 2 ^ l.bits.val :=
    Nat.pow_le_pow_right (by norm_num) (by omega)
  have h2 : Usize.max < 2 ^ System.Platform.numBits := by
    have : Usize.size = 2 ^ System.Platform.numBits := by simp [Usize.size, Usize.numBits]
    scalar_tac
  omega

/-- What the capped probe returns, in terms of the line: a hit; a vacant first-fit lane within the
`PROBE_CAP = 16` groups; or `blocked` (the whole capped prefix is full and holds no lane of `f`). -/
def ProbePost {V : Type} (l : hashtrinity.freq.FreqLine V) (f : Nat) (fnd blk : Bool) (slot : Nat) : Prop :=
  (fnd = true → blk = false ∧ foN l slot = f ∧ foN l slot ≠ EMPTYN) ∧
  (fnd = false → blk = false → 128 ≤ ctN l slot ∧ slot < l.slots.length ∧
    (∃ j, j < 16 ∧ slot / 16 = gseq (l.bits.val - 4) (pN f) j ∧
      ∀ j', j' < j → fullG l (gseq (l.bits.val - 4) (pN f) j')) ∧
    ∀ i, foN l i ≠ EMPTYN → foN l i ≠ f) ∧
  (blk = true → fnd = false ∧
    (∀ j', j' < 16 → fullG l (gseq (l.bits.val - 4) (pN f) j')) ∧
    ∀ i, foN l i ≠ EMPTYN → foN l i ≠ f)

theorem PROBE_CAP_val : hashtrinity.PROBE_CAP.val = 16 := by
  unfold hashtrinity.PROBE_CAP; rfl


theorem tz_cast_spec (m : U16) (hm : m.val ≠ 0) (tz : U32) (i1 : Usize)
    (htz : core.num.U16.trailing_zeros m = ok tz) (hi1 : lift (UScalar.cast .Usize tz) = ok i1) :
    i1.val < 16 ∧ m.val.testBit i1.val = true := by
  obtain ⟨htz1, htz2⟩ := u16_tz_spec m hm
  have htzv : tz.val = Aeneas.Std.BitVec.trailingZeros m.bv := by
    rw [htz1] at htz
    have := (Result.ok_injective htz).symm
    rw [this]
    have hT := (htz2 _ rfl).1
    show (BitVec.ofNat 32 (Aeneas.Std.BitVec.trailingZeros m.bv)).toNat = _
    rw [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (by omega)
  obtain ⟨hz16, hzbit, _⟩ := htz2 _ htzv
  have hi1v : i1.val = tz.val := by
    have : i1 = UScalar.cast .Usize tz := (Result.ok_injective hi1).symm
    rw [this, UScalar.cast_val_eq]
    apply Nat.mod_eq_of_lt
    have := nb_ge_aux
    have h2 : tz.val < 2^32 := by omega
    calc tz.val < 2^32 := h2
      _ ≤ _ := Nat.pow_le_pow_right (by norm_num) this
  rw [hi1v]
  exact ⟨hz16, hzbit⟩

theorem tagN_lt (f : Nat) : tagN f < 128 := by
  unfold tagN; exact Nat.mod_lt _ (by norm_num)

theorem lane_group (i : Nat) : i = 16 * (i / 16) + i % 16 ∧ i % 16 < 16 := by omega

theorem ge_val {ty} (x y : UScalar ty) : (x >= y) ↔ y.val ≤ x.val := by
  exact UScalar.le_equiv y x

theorem probe_outer_spec {V : Type} (l : hashtrinity.freq.FreqLine V) (hG : LineGeom l)
    (f p : U64) (tag : U8) (limit : U32) (hlim : limit.val = 16)
    (hp : p.val = pN f.val) (htag : tag.val = tagN f.val)
    (g0 : Usize) (hg0 : g0.val = gseq (l.bits.val - 4) (pN f.val) 0)
    (r : Bool × Bool × Usize)
    (h : hashtrinity.freq.FreqLine.probe_tagged_loop0 l f p tag limit g0 0#u32 false false false 0#usize = ok r) :
    ProbePost l f.val r.1 r.2.1 r.2.2.val := by
  unfold hashtrinity.freq.FreqLine.probe_tagged_loop0 at h
  refine loop_pc (fun (st : hashtrinity.freq.FreqLine V × Usize × U32 × Bool × Bool × Bool × Usize) =>
      st.1 = l ∧
      ((st.2.2.2.1 = true ∧ ProbePost l f.val st.2.2.2.2.1 st.2.2.2.2.2.1 st.2.2.2.2.2.2.val) ∨
       (st.2.2.2.1 = false ∧ st.2.2.2.2.1 = false ∧ st.2.2.2.2.2.1 = false ∧
        st.2.1.val = gseq (l.bits.val - 4) (pN f.val) st.2.2.1.val ∧ st.2.2.1.val < 16 ∧
        (∀ j', j' < st.2.2.1.val → fullG l (gseq (l.bits.val - 4) (pN f.val) j')) ∧
        (∀ i, foN l i ≠ EMPTYN → foN l i = f.val →
          ∃ js, js < 16 ∧ gseq (l.bits.val - 4) (pN f.val) js = i / 16 ∧
            (∀ j', j' < js → fullG l (gseq (l.bits.val - 4) (pN f.val) j')) ∧ st.2.2.1.val ≤ js))))
    (fun (y : Bool × Bool × Usize) => ProbePost l f.val y.1 y.2.1 y.2.2.val) _ ?_ _ ?_ r h
  · intro st hst cf hcf
    obtain ⟨self0, g, j, d, fnd, blk, sl⟩ := st
    obtain ⟨hs1, hst⟩ := hst
    simp only at hs1
    have hs1' := hs1.symm
    subst hs1'
    unfold hashtrinity.freq.FreqLine.probe_tagged_loop0.body at hcf
    dsimp only at hcf
    cases d with
    | true =>
      simp only [if_true] at hcf
      have hcf' : cf = ControlFlow.done (fnd, blk, sl) := (Result.ok_injective hcf).symm
      subst hcf'
      rcases hst with ⟨_, hpost⟩ | ⟨hd, _⟩
      · exact hpost
      · simp at hd
    | false =>
      simp only [Bool.false_eq_true, if_false] at hcf
      rcases hst with ⟨hd, _⟩ | ⟨_, hfnd0, hblk0, hgj, hj16, hfull, hpres⟩
      · simp at hd
      dsimp only at hgj hj16 hfull hpres hfnd0 hblk0
      subst hfnd0
      subst hblk0
      obtain ⟨base, hbase, hcf⟩ := bind_inv hcf
      obtain ⟨M, hM, hcf⟩ := bind_inv hcf
      obtain ⟨⟨self2, found1, slot1⟩, hin, hcf⟩ := bind_inv hcf
      have hbv := (umul_ok hbase).1
      rw [group_val] at hbv
      obtain ⟨hb1, hb2, hb3⟩ := match_tag_spec _ _ _ _ hM
      obtain ⟨hin1, hin2⟩ := probe_inner_spec _ _ _ _ _ _ _ hin
      simp only at hin1
      have hin1' := hin1.symm
      subst hin1'
      have hMk : ∀ k, k < 16 → (M.val.testBit k = true ↔ ctN l (16 * g.val + k) = tag.val) := by
        intro k hk
        obtain ⟨b, hb, hiff⟩ := hb3 k hk
        have : ctN l (16 * g.val + k) = b.val := by
          have e : 16 * g.val + k = g.val * 16 + k := by omega
          rw [e]; exact ctN_of_get _ _ _ hb
        rw [this]; exact hiff
      rcases hin2 with ⟨hf2, hs2, hnomatch⟩ | ⟨hf2, k, hk16, hMkb, hfok, hslk⟩
      · -- no tag candidate confirmed in this group
        simp only at hf2 hs2
        change (if found1 = true then _ else _) = _ at hcf
        rw [if_neg (by simp [hf2])] at hcf
        obtain ⟨E, hE, hcf⟩ := bind_inv hcf
        obtain ⟨he1, he2, he3⟩ := match_empty_spec _ _ _ hE
        have hEk : ∀ k, k < 16 → (E.val.testBit k = true ↔ 128 ≤ ctN l (16 * g.val + k)) := by
          intro k hk
          obtain ⟨b, hb, hiff⟩ := he3 k hk
          have : ctN l (16 * g.val + k) = b.val := by
            have e : 16 * g.val + k = g.val * 16 + k := by omega
            rw [e]; exact ctN_of_get _ _ _ hb
          rw [this]; exact hiff
        -- a live lane of f in this group would have been confirmed
        have hnotin : ∀ i, foN l i ≠ EMPTYN → foN l i = f.val → i / 16 ≠ g.val := by
          intro i hi hfi hig
          have hct := hG.full_ctrl i hi
          rw [hfi] at hct
          have hk := lane_group i
          have hm1 := (hMk (i % 16) hk.2).mpr (by
            have e : 16 * g.val + i % 16 = i := by omega
            rw [e, hct, htag])
          have e : g.val * 16 + i % 16 = i := by omega
          have h2 := hnomatch (i % 16) hk.2 hm1
          rw [hbv, e] at h2
          exact h2 hfi
        by_cases hE0 : E != 0#u16
        · simp only [hE0, if_true] at hcf
          have hEne : E.val ≠ 0 := by
            intro h0
            have : E = 0#u16 := UScalar.eq_of_val_eq (h0.trans rfl)
            simp [this] at hE0
          obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
          obtain ⟨i1, hi1, hcf⟩ := bind_inv hcf
          obtain ⟨slot2, hslot2, hcf⟩ := bind_inv hcf
          have hcf' : cf = ControlFlow.cont (l, g, j, true, false, false, slot2) := (Result.ok_injective hcf).symm
          subst hcf'
          obtain ⟨hi1lt, hi1bit⟩ := tz_cast_spec E hEne tz i1 htz hi1
          have hs2v := (uadd_ok hslot2).1
          have hemp := (hEk i1.val hi1lt).mp hi1bit
          obtain ⟨b, hb, _⟩ := he3 i1.val hi1lt
          have hslt : slot2.val < l.slots.length := by
            rw [hs2v, hbv]
            have := (List.getElem?_eq_some_iff.mp hb).1
            have e1 : (↑l.ctrl : List U8).length = l.ctrl.length := rfl
            have := hG.len_eq
            omega
          have e : slot2.val = 16 * g.val + i1.val := by rw [hs2v, hbv]; omega
          refine ⟨rfl, Or.inl ⟨rfl, ?_⟩⟩
          refine ⟨fun htr => by simp at htr, fun _ _ => ?_, fun htr => by simp at htr⟩
          refine ⟨?_, hslt, ⟨j.val, hj16, ?_, hfull⟩, ?_⟩
          · rw [e]; exact hemp
          · rw [← hgj, e]; omega
          · intro i hi hfi
            obtain ⟨js, hjs0, hjs1, hjs2, hjs3⟩ := hpres i hi hfi
            by_cases hjj : js = j.val
            · subst hjj
              have hig : i / 16 = g.val := by rw [← hjs1, ← hgj]
              exact absurd hig (hnotin i hi hfi)
            · have hfg := hjs2 j.val (by omega)
              rw [← hgj] at hfg
              have := hfg i1.val hi1lt
              omega
        · rw [if_neg hE0] at hcf
          have hE0' : E.val = 0 := by simpa using hE0
          obtain ⟨j1, hj1, hcf⟩ := bind_inv hcf
          have hj1v : j1.val = j.val + 1 := by
            have := (uadd_ok hj1).1
            rw [one_u32] at this
            exact this
          have hfg : fullG l g.val := by
            intro k hk
            by_contra hc
            have h1 := (hEk k hk).mpr (by omega)
            rw [hE0'] at h1
            simp at h1
          by_cases hge : j1 >= limit
          · rw [if_pos hge] at hcf
            have hcf' : cf = ControlFlow.cont (l, g, j1, true, false, true, slot1) :=
              (Result.ok_injective hcf).symm
            subst hcf'
            have hge' : limit.val ≤ j1.val := (ge_val j1 limit).mp hge
            have hj16' : j.val + 1 = 16 := by omega
            have hall : ∀ j', j' < 16 → fullG l (gseq (l.bits.val - 4) (pN f.val) j') := by
              intro j' hj'
              by_cases hjj : j' = j.val
              · subst hjj; rw [← hgj]; exact hfg
              · exact hfull j' (by omega)
            refine ⟨rfl, Or.inl ⟨rfl, ?_⟩⟩
            refine ⟨fun htr => by simp at htr, fun _ hb => by simp at hb, fun _ => ⟨rfl, hall, ?_⟩⟩
            intro i hi hfi
            obtain ⟨js, hjs0, hjs1, hjs2, hjs3⟩ := hpres i hi hfi
            by_cases hjj : js = j.val
            · subst hjj
              have hig : i / 16 = g.val := by rw [← hjs1, ← hgj]
              exact absurd hig (hnotin i hi hfi)
            · omega
          · rw [if_neg hge] at hcf
            obtain ⟨g1, hg1, hcf⟩ := bind_inv hcf
            have hcf' : cf = ControlFlow.cont (l, g1, j1, false, false, false, slot1) :=
              (Result.ok_injective hcf).symm
            subst hcf'
            have hlt16 : j1.val < 16 := by
              have : ¬ limit.val ≤ j1.val := fun hh => hge ((ge_val j1 limit).mpr hh)
              omega
            have hg1v : g1.val = gseq (l.bits.val - 4) (pN f.val) j1.val := by
              have := descend_group_spec l p j1 g1 hg1 hG.gmask hG.gb_le
              rwa [hp] at this
            refine ⟨rfl, Or.inr ⟨rfl, rfl, rfl, hg1v, hlt16, ?_, ?_⟩⟩
            · intro j' hj'
              rw [hj1v] at hj'
              by_cases hjj : j' = j.val
              · subst hjj; rw [← hgj]; exact hfg
              · exact hfull j' (by omega)
            · intro i hi hfi
              obtain ⟨js, hjs0, hjs1, hjs2, hjs3⟩ := hpres i hi hfi
              refine ⟨js, hjs0, hjs1, hjs2, ?_⟩
              rw [hj1v]
              by_cases hjj : js = j.val
              · subst hjj
                have hig : i / 16 = g.val := by rw [← hjs1, ← hgj]
                exact absurd hig (hnotin i hi hfi)
              · omega
      · -- the confirm loop found the fold
        simp only at hf2
        change (if found1 = true then _ else _) = _ at hcf
        rw [if_pos hf2] at hcf
        have hcf' : cf = ControlFlow.cont (l, g, j, true, true, false, slot1) := (Result.ok_injective hcf).symm
        subst hcf'
        have e : base.val + k = 16 * g.val + k := by omega
        have hnotE : foN l (base.val + k) ≠ EMPTYN := by
          intro hE
          have h1 := hG.empty_ctrl _ hE
          have h2 := (hMk k hk16).mp hMkb
          rw [e] at h1
          have := tagN_lt f.val
          omega
        refine ⟨rfl, Or.inl ⟨rfl, ?_⟩⟩
        refine ⟨fun _ => ⟨rfl, ?_, ?_⟩, fun hh => by simp at hh, fun hh => by simp at hh⟩
        · rw [hslk]; exact hfok
        · rw [hslk]; exact hnotE
  · dsimp only
    refine ⟨rfl, Or.inr ⟨rfl, rfl, rfl, hg0, by norm_num, fun j' hj => absurd hj (by simp), ?_⟩⟩
    intro i hi hfi
    obtain ⟨js, hjs0, hjs1, hjs2⟩ := hG.placed i hi
    rw [hfi] at hjs1 hjs2
    exact ⟨js, hjs0, hjs1, hjs2, by simp⟩


theorem tag_of_p (i1 : U64) (fv : Nat) (hp : i1.val = pN fv / 2^33) :
    (UScalar.cast .U8 (i1 &&& 127#u64)).val = tagN fv := by
  rw [UScalar.cast_val_eq, UScalar.val_and, hp]
  have h127 : (127#u64).val = 127 := rfl
  have e7 : (127 : Nat) = 2^7 - 1 := by norm_num
  rw [h127, e7, Nat.and_two_pow_sub_one_eq_mod]
  simp only [tagN, UScalarTy.numBits]
  apply Nat.mod_eq_of_lt
  have := Nat.mod_lt (pN fv / 2^33) (show 0 < 2^7 by norm_num)
  omega

theorem pN_of_wrapping (f : U64) : (core.num.U64.wrapping_mul f hashtrinity.PHI).val = pN f.val := by
  rw [Apeiron.CountHelpers.wm_val]
  rfl

theorem probe_tagged_spec {V : Type} (l : hashtrinity.freq.FreqLine V) (hG : LineGeom l)
    (f : U64) (slot : Usize) (found blocked : Bool) (tag : U8)
    (h : hashtrinity.freq.FreqLine.probe_tagged l f = ok (slot, found, blocked, tag)) :
    tag.val = tagN f.val ∧ ProbePost l f.val found blocked slot.val := by
  unfold hashtrinity.freq.FreqLine.probe_tagged at h
  obtain ⟨p, hp, h⟩ := bind_inv h
  obtain ⟨sh, hsh, h⟩ := bind_inv h
  obtain ⟨i1, hi1, h⟩ := bind_inv h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨tag', htag, h⟩ := bind_inv h
  obtain ⟨limit, hlimit, h⟩ := bind_inv h
  obtain ⟨g, hg, h⟩ := bind_inv h
  obtain ⟨⟨found', blocked', slot'⟩, hloop, h⟩ := bind_inv h
  have hpv : p.val = pN f.val := by
    have : p = core.num.U64.wrapping_mul f hashtrinity.PHI := (Result.ok_injective hp).symm
    rw [this]; exact pN_of_wrapping f
  have hshv : sh = 33#u32 := by
    rw [Apeiron.CountHelpers.tag_shift_ok] at hsh; exact (Result.ok_injective hsh).symm
  have hi1v : i1.val = pN f.val / 2^33 := by
    rw [hshv] at hi1
    have hsp : (p >>> (33#u32)) ⦃ r => r.val = p.val >>> 33 ⦄ := by
      step*
    obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists hsp
    rw [hr] at hi1
    have : i1 = r := (Result.ok_injective hi1).symm
    rw [this, hrv, Nat.shiftRight_eq_div_pow, hpv]
  have hi' : i = i1 &&& 127#u64 := (Result.ok_injective hi).symm
  have htag' : tag' = UScalar.cast .U8 i := (Result.ok_injective htag).symm
  have htagv : tag'.val = tagN f.val := by rw [htag', hi']; exact tag_of_p i1 f.val hi1v
  have hlimv : limit.val = 16 := by
    have : limit = hashtrinity.PROBE_CAP := by
      unfold hashtrinity.freq.FreqLine.probe_limit at hlimit
      exact (Result.ok_injective hlimit).symm
    rw [this]; exact PROBE_CAP_val
  have hres : (slot', found', blocked', tag') = (slot, found, blocked, tag) := Result.ok_injective h
  simp only [Prod.mk.injEq] at hres
  obtain ⟨hs, hf, hb, ht⟩ := hres
  subst hs; subst hf; subst hb; subst ht
  have hg0 : g.val = gseq (l.bits.val - 4) (pN f.val) 0 := by
    have := home_group_spec l p g hg (l.bits.val - 4) hG.gmask hG.gb_le
    rw [this, hpv]
    have hb := hG.bits4
    have hgs := hG.gshift
    have hN : System.Platform.numBits = 32 ∨ System.Platform.numBits = 64 := System.Platform.numBits_eq
    have hle := hG.gb_le
    unfold gseq
    by_cases hz : l.bits.val - 4 = 0
    · simp [hz, Nat.mod_one]
    · have hne : l.bits.val ≠ 4 := by omega
      rw [if_neg hz, if_pos (by omega), hgs, if_neg hne]
      congr 2
      omega
  have := probe_outer_spec l hG f p tag' limit hlimv hpv htagv g hg0 _ hloop
  exact ⟨htagv, this⟩

end Apeiron.CountAggregate
