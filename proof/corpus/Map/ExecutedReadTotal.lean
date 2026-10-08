-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): totality of the executed read.
import Map.ExecutedRead
import Map.ExecutedReadUncond
import Map.ExecutedBuild
import Map.PrefixMasks
open Aeneas Aeneas.Std Result

/-!
# Totality of the executed read

Every lemma on the read path (`Map/ExecutedRead.lean`, `Map/ExecutedProbe.lean`) is partial
correctness: `body = ok r → P r`, and `word_hashtrinity_get` returns `ok` only under the premise
`hok`.  This module proves the read's failure-freedom: each arithmetic step and each of the executed
loops returns `ok`, by a strictly decreasing measure on the loop state (`loop_tot_inv`), so that
`get_ok_of_mapRep : MapRep s m live → ∀ k, ∃ r, word_hashtrinity_get s k = ok r` removes `hok`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-! ## Generic loop totality

The Aeneas `loop` is a `partial_fixpoint` (`Aeneas/Std/Primitives.lean:415`).  `loop_tot_inv` is its
termination principle by strong recursion on a measure: on the invariant the body returns some
`ControlFlow`, and every `cont` step preserves the invariant and strictly decreases the measure. -/

theorem loop_tot_inv {α β : Type} (body : α → Result (ControlFlow α β)) (inv : α → Prop)
    (measure : α → Nat)
    (hbody : ∀ a, inv a → ∃ cf, body a = ok cf)
    (hdec : ∀ a a', inv a → body a = ok (ControlFlow.cont a') → inv a' ∧ measure a' < measure a)
    (x : α) (hx : inv x) : ∃ r, loop body x = ok r := by
  have aux : ∀ n, ∀ a, measure a = n → inv a → ∃ r, loop body a = ok r := by
    intro n
    induction n using Nat.strongRecOn with
    | ind n ih =>
      intro a ha hinv
      rw [loop.eq_def]
      obtain ⟨cf, hcf⟩ := hbody a hinv
      rw [hcf]
      simp only [bind_ok]
      cases cf with
      | done y => exact ⟨y, rfl⟩
      | cont a' =>
        obtain ⟨hinv', hlt⟩ := hdec a a' hinv hcf
        exact ih (measure a') (by rw [← ha]; exact hlt) a' rfl hinv'
  exact aux (measure x) x rfl hx

/-- Totality when the body is total on every state. -/
theorem loop_tot {α β : Type} (body : α → Result (ControlFlow α β)) (measure : α → Nat)
    (hbody : ∀ a, ∃ cf, body a = ok cf)
    (hdec : ∀ a a', body a = ok (ControlFlow.cont a') → measure a' < measure a)
    (x : α) : ∃ r, loop body x = ok r :=
  loop_tot_inv body (fun _ => True) measure (fun a _ => hbody a)
    (fun a a' _ h => ⟨trivial, hdec a a' h⟩) x trivial

/-! ## Checked-scalar existence -/

theorem uadd_tot {ty} (x y : UScalar ty) (h : x.val + y.val < 2 ^ ty.numBits) :
    ∃ z, x + y = ok z ∧ z.val = x.val + y.val :=
  Apeiron.PrefixMasks.uadd_ex x y h

theorem usub_tot {ty} (x y : UScalar ty) (h : y.val ≤ x.val) :
    ∃ z, x - y = ok z ∧ z.val = x.val - y.val := by
  obtain ⟨z, hz, hv, _⟩ := WP.spec_imp_exists (UScalar.sub_spec (x := x) (y := y) h)
  exact ⟨z, hz, hv⟩

theorem ushr_tot {ty tys} (x : UScalar ty) (s : UScalar tys) (h : s.val < ty.numBits) :
    ∃ z, x >>> s = ok z ∧ z.val = x.val >>> s.val := by
  obtain ⟨z, hz, hv, _⟩ := WP.spec_imp_exists (UScalar.ShiftRight_spec x s h)
  exact ⟨z, hz, hv⟩

theorem umul_tot {ty} (x y : UScalar ty) (h : x.val * y.val ≤ UScalar.max ty) :
    ∃ z, x * y = ok z ∧ z.val = x.val * y.val := by
  obtain ⟨z, hz, hv⟩ := WP.spec_imp_exists (UScalar.mul_spec (x := x) (y := y) h)
  exact ⟨z, hz, hv⟩

theorem udiv_tot {ty} (x y : UScalar ty) (h : y.val ≠ 0) :
    ∃ z, x / y = ok z ∧ z.val = x.val / y.val := by
  obtain ⟨z, hz, hv⟩ := UScalar.div_spec x h
  exact ⟨z, hz, hv⟩

/-- `U64 → Usize` cast is the identity when the value fits the platform's word. -/
theorem cast_u64_usize_lt (x : U64) (h : x.val < 2 ^ System.Platform.numBits) :
    (UScalar.cast .Usize x).val = x.val := by
  rw [UScalar.cast_val_eq]
  apply Nat.mod_eq_of_lt
  have e : UScalarTy.Usize.numBits = System.Platform.numBits := rfl
  rwa [e]

/-- `x &&& (2^k - 1) < 2^k`: the masked probe index stays below the power-of-two slot count. -/
theorem and_mask_lt {ty} (x mask : UScalar ty) (k : Nat)
    (hmask : mask.val = 2 ^ k - 1) : (x &&& mask).val < 2 ^ k := by
  rw [UScalar.val_and, hmask]
  have h : (2:Nat)^k - 1 < 2^k := by
    have := Nat.one_le_two_pow (n := k); omega
  exact lt_of_le_of_lt Nat.and_le_right h

theorem val_1_u32 : (1#u32).val = 1 := rfl
theorem val_1_usize : (1#usize).val = 1 := rfl
theorem val_64_u32 : (64#u32).val = 64 := rfl
theorem val_65_usize : (65#usize).val = 65 := rfl

theorem u32_sum_lt (a b : Nat) (ha : a ≤ 64) (hb : b ≤ 64) :
    a + b < 2 ^ UScalarTy.U32.numBits := by
  have h : (2:Nat)^UScalarTy.U32.numBits = 4294967296 := by
    rw [show UScalarTy.U32.numBits = 32 from rfl]; norm_num
  omega

theorem usize_sum_lt (a b : Nat) (ha : a ≤ 64) (hb : b ≤ 8) :
    a + b < 2 ^ UScalarTy.Usize.numBits := by
  have h2 : 2 ^ 32 ≤ 2 ^ UScalarTy.Usize.numBits :=
    Nat.pow_le_pow_right (by norm_num) Apeiron.PrefixMasks.nb_ge
  have h32 : (2:Nat)^32 = 4294967296 := by norm_num
  rw [h32] at h2
  omega

theorem vec_index_ex {α : Type} (v : alloc.vec.Vec α) (i : Usize) (h : i.val < v.val.length) :
    ∃ x, alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice α) v i = ok x := by
  unfold alloc.vec.Vec.index
  exact Apeiron.PrefixMasks.slice_index_ex v.slice i
    (by simpa [alloc.vec.Vec.val, Slice.length] using h)

theorem array_index_usize_ex {α : Type} {n : Usize} (v : Array α n) (i : Usize)
    (h : i.val < n.val) : ∃ x, Array.index_usize v i = ok x := by
  obtain ⟨x, hx, _⟩ := WP.spec_imp_exists
    (Array.index_usize_spec v i (by show i.val < v.val.length; rw [Array.property]; exact h))
  exact ⟨x, hx⟩

theorem array_update_ex {α : Type} {n : Usize} (v : Array α n) (i : Usize) (x : α)
    (h : i.val < n.val) : ∃ v', Array.update v i x = ok v' := by
  obtain ⟨v', hv', _⟩ := WP.spec_imp_exists
    (Array.update_spec v i x (by show i.val < v.val.length; rw [Array.property]; exact h))
  exact ⟨v', hv'⟩

/-! ## The no-loop arithmetic -/

theorem bits_le_64 {KS : Type} (_ks : hashtrinity.algebra.KeyStore KS U64) (s : Tbl KS)
    (hlen : s.coords.length = 2 ^ s.bits.val) : s.bits.val ≤ 64 := by
  have hlt := slice_len_lt s.coords
  rw [hlen] at hlt
  have hpow := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp hlt
  rcases System.Platform.numBits_eq with h | h <;> omega

/-- `Table::home` returns a slot index, below the slot count. -/
theorem home_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) (s : Tbl KS) (c : U64)
    (hb : 0 < s.bits.val) (hb64 : s.bits.val ≤ 64) (hlen : s.coords.length = 2 ^ s.bits.val) :
    ∃ h : Usize, hashtrinity.Table.home core.cmp.EqU64 CI ks s c = ok h ∧
      h.val < s.coords.length := by
  unfold hashtrinity.Table.home
  simp only [lift, bind_tc_ok]
  by_cases hlt : UScalar.cast .Usize c < Slice.len s.coords
  · refine ⟨UScalar.cast .Usize c, by rw [if_pos hlt], ?_⟩
    have := (UScalar.lt_equiv _ _).mp hlt
    rwa [Slice.len_val] at this
  · rw [if_neg hlt]
    obtain ⟨i2, hi2, hi2v0⟩ := usub_tot (64#u32) s.bits (by simpa using hb64)
    have hi2v : i2.val = 64 - s.bits.val := by rw [hi2v0, val_64_u32]
    rw [hi2]
    simp only [bind_tc_ok]
    have hi2lt : i2.val < 64 := by rw [hi2v]; omega
    obtain ⟨i3, hi3, hi3v⟩ :=
      ushr_tot (core.num.U64.wrapping_mul c hashtrinity.PHI) i2
        (by simpa [UScalarTy.numBits] using hi2lt)
    rw [hi3]
    simp only [bind_tc_ok]
    refine ⟨UScalar.cast .Usize i3, rfl, ?_⟩
    have hbits_lt : s.bits.val < System.Platform.numBits := by
      have hlt' := slice_len_lt s.coords
      rw [hlen] at hlt'
      exact (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp hlt'
    have hi3lt : i3.val < 2 ^ s.bits.val := by
      rw [hi3v, hi2v, Nat.shiftRight_eq_div_pow]
      rw [Nat.div_lt_iff_lt_mul (by positivity : (0:Nat) < 2 ^ (64 - s.bits.val))]
      rw [← Nat.pow_add, Nat.add_sub_of_le hb64]
      exact (core.num.U64.wrapping_mul c hashtrinity.PHI).hBounds
    rw [cast_u64_usize_lt i3 (lt_trans hi3lt (Nat.pow_lt_pow_right (by norm_num) hbits_lt)), hlen]
    exact hi3lt

theorem window_steps_loop_total (t : Array U32 65#usize) (b : Usize) (hb1 : 1 ≤ b.val) :
    ∃ a, hashtrinity.overflow.WINDOW_STEPS_BY_BITS_loop t b = ok a := by
  unfold hashtrinity.overflow.WINDOW_STEPS_BY_BITS_loop
  refine loop_tot_inv
    (fun p : Array U32 65#usize × Usize =>
      hashtrinity.overflow.WINDOW_STEPS_BY_BITS_loop.body p.1 p.2)
    (fun p => 1 ≤ p.2.val) (fun p => 66 - p.2.val) ?_ ?_ (t, b) hb1
  · rintro ⟨t1, b1⟩ hb1
    change 1 ≤ b1.val at hb1
    unfold hashtrinity.overflow.WINDOW_STEPS_BY_BITS_loop.body
    simp only [lift, bind_tc_ok]
    by_cases hlt : b1 < 65#usize
    · rw [if_pos hlt]
      have hbv : b1.val < 65 := by simpa using hlt
      have hcast : (UScalar.cast .U32 b1).val = b1.val := by
        rw [UScalar.cast_val_eq]; apply Nat.mod_eq_of_lt
        simpa using u32_sum_lt b1.val 0 (by omega) (by norm_num)
      obtain ⟨i1, hi1, hi1v0⟩ := udiv_tot (64#u32) (UScalar.cast .U32 b1)
        (by rw [hcast]; omega)
      have hi1v : i1.val = 64 / b1.val := by rw [hi1v0, hcast, val_64_u32]
      have hi1le : i1.val ≤ 64 := by rw [hi1v]; exact Nat.div_le_self _ _
      rw [hi1]
      simp only [bind_tc_ok]
      obtain ⟨i2, hi2, hi2v0⟩ := uadd_tot hashtrinity.LINEAR_G i1
        (by rw [LINEAR_G_val]; exact u32_sum_lt 8 i1.val (by norm_num) hi1le)
      have hi2v : i2.val = 8 + i1.val := by rw [hi2v0, LINEAR_G_val]
      rw [hi2]
      simp only [bind_tc_ok]
      obtain ⟨i3, hi3, _⟩ := usub_tot i2 1#u32 (by rw [hi2v, val_1_u32]; omega)
      rw [hi3]
      simp only [bind_tc_ok]
      obtain ⟨a, ha⟩ := array_update_ex t1 b1 i3 (by rw [val_65_usize]; exact hbv)
      rw [ha]
      simp only [bind_tc_ok]
      obtain ⟨b2, hb2, _⟩ := uadd_tot b1 1#usize
        (by rw [val_1_usize]; exact usize_sum_lt b1.val 1 (by omega) (by norm_num))
      rw [hb2]
      simp only [bind_tc_ok]
      exact ⟨_, rfl⟩
    · rw [if_neg hlt]
      exact ⟨_, rfl⟩
  · rintro ⟨t1, b1⟩ ⟨t2, b2⟩ hb1 hcf
    change 1 ≤ b1.val at hb1
    unfold hashtrinity.overflow.WINDOW_STEPS_BY_BITS_loop.body at hcf
    simp only [lift, bind_tc_ok] at hcf
    by_cases hlt : b1 < 65#usize
    · rw [if_pos hlt] at hcf
      obtain ⟨i1, _, hcf⟩ := bind_inv hcf
      obtain ⟨i2, _, hcf⟩ := bind_inv hcf
      obtain ⟨i3, _, hcf⟩ := bind_inv hcf
      obtain ⟨a, _, hcf⟩ := bind_inv hcf
      obtain ⟨b3, hb3, hcf⟩ := bind_inv hcf
      have hbv : b1.val < 65 := by simpa using hlt
      have hb3v := (uadd_ok hb3).1
      rw [val_1_usize] at hb3v
      have hcont := Result.ok_injective hcf
      simp only [ControlFlow.cont.injEq, Prod.mk.injEq] at hcont
      obtain ⟨_, hb2eq⟩ := hcont
      have hle : 1 ≤ b3.val := by rw [hb3v]; omega
      have hlt' : 66 - b3.val < 66 - b1.val := by rw [hb3v]; omega
      subst hb2eq
      exact ⟨hle, hlt'⟩
    · rw [if_neg hlt] at hcf
      simp at hcf

theorem window_steps_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) (s : Tbl KS)
    (hb64 : s.bits.val ≤ 64) :
    ∃ w, hashtrinity.overflow.Table.window_steps core.cmp.EqU64 CI ks s = ok w := by
  unfold hashtrinity.overflow.Table.window_steps hashtrinity.overflow.WINDOW_STEPS_BY_BITS
  simp only [lift, bind_tc_ok]
  obtain ⟨a, ha⟩ := window_steps_loop_total (Array.repeat 65#usize 0#u32) 1#usize (by simp)
  rw [ha]
  simp only [bind_tc_ok]
  have hcast : (UScalar.cast .Usize s.bits).val = s.bits.val := by
    rw [UScalar.cast_val_eq]; apply Nat.mod_eq_of_lt
    have h2 : 2 ^ 32 ≤ 2 ^ UScalarTy.Usize.numBits :=
      Nat.pow_le_pow_right (by norm_num) Apeiron.PrefixMasks.nb_ge
    have h32 : (2:Nat)^32 = 4294967296 := by norm_num
    rw [h32] at h2; omega
  obtain ⟨w, hw⟩ := array_index_usize_ex a (UScalar.cast .Usize s.bits)
    (by rw [hcast, val_65_usize]; omega)
  exact ⟨w, hw⟩

/-- `Table::descend` returns a probe index below the slot count, for a step below the cap. -/
theorem descend_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) (s : Tbl KS) (c : U64)
    (j : U32) (hb : 0 < s.bits.val) (hb64 : s.bits.val ≤ 64)
    (hlen : s.coords.length = 2 ^ s.bits.val) (hj : j.val < limN s.bits.val) :
    ∃ i : Usize, Desc ks s c j i ∧ i.val < s.coords.length := by
  unfold Desc hashtrinity.Table.descend
  simp only [lift, bind_tc_ok]
  obtain ⟨mask, hmask, hmaskv⟩ :=
    usub_tot (Slice.len s.coords) 1#usize
      (by rw [val_1_usize, Slice.len_val, hlen]; exact Nat.one_le_two_pow)
  rw [hmask]
  simp only [bind_tc_ok]
  have hmv : mask.val = 2 ^ s.bits.val - 1 := by rw [hmaskv, Slice.len_val, hlen, val_1_usize]
  by_cases hj8 : j < hashtrinity.LINEAR_G
  · rw [if_pos hj8]
    obtain ⟨h, hhome, _⟩ := home_total ks s c hb hb64 hlen
    rw [hhome]
    simp only [bind_tc_ok]
    refine ⟨_, rfl, ?_⟩
    rw [hlen]
    exact and_mask_lt (core.num.Usize.wrapping_add h (UScalar.cast .Usize j)) mask
      (s.bits.val) hmv
  · rw [if_neg hj8]
    have hjge : 8 ≤ j.val := by
      have h : ¬ j.val < hashtrinity.LINEAR_G.val :=
        (UScalar.lt_equiv j hashtrinity.LINEAR_G).not.mp hj8
      rw [LINEAR_G_val] at h; omega
    have hlim16 : limN s.bits.val ≤ 16 := by unfold limN; omega
    obtain ⟨i1, hi1, hi1v0⟩ := usub_tot j hashtrinity.LINEAR_G (by rw [LINEAR_G_val]; omega)
    have hi1v : i1.val = j.val - 8 := by rw [hi1v0, LINEAR_G_val]
    rw [hi1]
    simp only [bind_tc_ok]
    obtain ⟨jj, hjj, hjjv0⟩ := uadd_tot i1 1#u32
      (by rw [hi1v, val_1_u32]; exact u32_sum_lt (j.val - 8) 1 (by have := hlim16; have := hj; omega) (by norm_num))
    have hjjv : jj.val = i1.val + 1 := by rw [hjjv0, val_1_u32]
    rw [hjj]
    simp only [bind_tc_ok]
    obtain ⟨i2, hi2, hi2v0⟩ := uadd_tot jj 1#u32
      (by rw [val_1_u32]; exact u32_sum_lt jj.val 1 (by have := hlim16; have := hj; have := hjjv; omega) (by norm_num))
    have hi2v : i2.val = jj.val + 1 := by rw [hi2v0, val_1_u32]
    rw [hi2]
    simp only [bind_tc_ok]
    have hi2val : i2.val = j.val - 6 := by rw [hi2v, hjjv, hi1v]; omega
    have hlimle : limN s.bits.val ≤ 64 / s.bits.val + 7 := by
      unfold limN
      refine le_trans (Nat.min_le_right _ _) ?_
      rw [Nat.sub_le_iff_le_add]
      omega
    have hi2ub : i2.val ≤ 64 / s.bits.val := by
      have := lt_of_lt_of_le hj hlimle; rw [hi2val]; omega
    have hmul64 : s.bits.val * i2.val ≤ 64 :=
      calc s.bits.val * i2.val ≤ s.bits.val * (64 / s.bits.val) := Nat.mul_le_mul_left _ hi2ub
        _ ≤ 64 := Nat.mul_div_le _ _
    have hprod : s.bits.val * i2.val ≤ UScalar.max UScalarTy.U32 := by
      have hmax : UScalar.max UScalarTy.U32 = 2 ^ 32 - 1 := by
        rw [← UScalar.rMax_eq_max, UScalar.rMax_eq_pow_numBits]; rfl
      rw [hmax]
      have h32 : (2:Nat)^32 = 4294967296 := by norm_num
      omega
    obtain ⟨used, hused, husedv⟩ := umul_tot s.bits i2 hprod
    rw [hused]
    simp only [bind_tc_ok]
    have husedle64 : used.val ≤ 64 := by rw [husedv]; exact hmul64
    rw [if_pos ((UScalar.le_equiv _ _).mpr husedle64)]
    obtain ⟨i3, hi3, hi3v0⟩ := usub_tot (64#u32) used husedle64
    have hi3v : i3.val = 64 - used.val := by rw [hi3v0, val_64_u32]
    rw [hi3]
    simp only [bind_tc_ok]
    have husedpos : 0 < used.val := by rw [husedv]; exact Nat.mul_pos hb (by omega)
    have hi3lt : i3.val < 64 := by rw [hi3v]; omega
    obtain ⟨i4, hi4, _⟩ :=
      ushr_tot (core.num.U64.wrapping_mul c hashtrinity.PHI) i3
        (by simpa [UScalarTy.numBits] using hi3lt)
    rw [hi4]
    simp only [bind_tc_ok]
    refine ⟨_, rfl, ?_⟩
    rw [hlen]
    exact and_mask_lt (UScalar.cast .Usize i4) mask (s.bits.val) hmv

/-! ## The executed loops -/

/-- `get_unchecked` on a `Slice` is its checked index on the in-range domain. -/
theorem get_unchecked_ex {α : Type} (s : Slice α) (i : Usize) (h : i.val < s.length) :
    ∃ x, core.slice.Slice.get_unchecked (core.slice.index.SliceIndexUsizeSlice α) s i = ok x := by
  unfold core.slice.Slice.get_unchecked
  exact Apeiron.PrefixMasks.slice_index_ex s i h

theorem limN_le_16 (b : Nat) : limN b ≤ 16 := Nat.min_le_left _ _

theorem vec_len_lt {α : Type} (v : alloc.vec.Vec α) :
    v.val.length < 2 ^ System.Platform.numBits := by
  have h := Slice.length_ineq v.slice
  have hm : Usize.max = 2 ^ System.Platform.numBits - 1 := by simp [Usize.max, Usize.numBits]
  rw [hm] at h
  exact lt_of_le_of_lt h (Nat.sub_lt (by positivity) one_pos)

/-- The `PartialEq` comparison on `u64` is `decide` of equality. -/
theorem impls_eqU64 (a b : U64) : core.cmp.impls.PartialEqU64.eq a b = decide (a = b) :=
  Result.ok_injective (eqU64_eq a b)

/-- `Usize → U32` cast is the identity below `2^32`. -/
theorem cast_usize_u32_of_lt (x : Usize) (h : x.val < 2 ^ 32) :
    (UScalar.cast .U32 x).val = x.val := by
  rw [UScalar.cast_val_eq, show UScalarTy.U32.numBits = 32 from rfl]
  exact Nat.mod_eq_of_lt h

/-- `SideStore::find`'s loop terminates: the index strictly increases toward the vector length. -/
theorem side_find_loop_total {K V KS : Type} (ks : hashtrinity.algebra.KeyStore KS K)
    (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (ss : hashtrinity.side.SideStore K V KS) (c : U64) (k : K) (i : Usize) :
    ∃ r, hashtrinity.side.SideStore.find_loop ks ss c k i = ok r := by
  unfold hashtrinity.side.SideStore.find_loop
  refine loop_tot
    (fun i => hashtrinity.side.SideStore.find_loop.body ks ss c k i)
    (fun i => ss.coords.val.length - i.val) ?_ ?_ i
  · intro i
    unfold hashtrinity.side.SideStore.find_loop.body
    by_cases hl : i < alloc.vec.Vec.len ss.coords
    · rw [if_pos hl]
      have hv : i.val < ss.coords.val.length := by
        have := (UScalar.lt_equiv _ _).mp hl; rwa [alloc.vec.Vec.len_val] at this
      obtain ⟨x, hx⟩ := vec_index_ex ss.coords i hv
      rw [hx]
      simp only [bind_tc_ok]
      by_cases hxc : x = c
      · rw [if_pos hxc]
        rw [hm]
        simp only [bind_tc_ok]
        rw [if_pos trivial]
        exact ⟨_, rfl⟩
      · rw [if_neg hxc]
        obtain ⟨i3, hi3, _⟩ := uadd_tot i 1#usize
          (by
            rw [val_1_usize, show UScalarTy.Usize.numBits = System.Platform.numBits from rfl]
            have hlt := vec_len_lt ss.coords
            omega)
        rw [hi3]
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
    · rw [if_neg hl]
      exact ⟨_, rfl⟩
  · intro i i' hcf
    unfold hashtrinity.side.SideStore.find_loop.body at hcf
    by_cases hl : i < alloc.vec.Vec.len ss.coords
    · rw [if_pos hl] at hcf
      obtain ⟨x, hx, hcf⟩ := bind_inv hcf
      by_cases hxc : x = c
      · rw [if_pos hxc] at hcf
        obtain ⟨b, hb, hcf⟩ := bind_inv hcf
        rw [hm] at hb
        have hb' : b = true := (Result.ok_injective hb).symm
        rw [if_pos hb'] at hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
      · rw [if_neg hxc] at hcf
        obtain ⟨i3, hi3, hcf⟩ := bind_inv hcf
        have hh := Result.ok_injective hcf
        simp only [ControlFlow.cont.injEq] at hh
        subst hh
        have hi3v := (uadd_ok hi3).1
        rw [val_1_usize] at hi3v
        have hv : i.val < ss.coords.val.length := by
          have := (UScalar.lt_equiv _ _).mp hl; rwa [alloc.vec.Vec.len_val] at this
        rw [hi3v]
        omega
    · rw [if_neg hl] at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)

/-- `prefix_masks` is total on a readable window; by `prefix_masks_probe_independent` this covers
both the scalar `prefix_masks_loop` and the AVX2 `prefix_masks_avx2` branches. -/
theorem prefix_masks_total (coords : Slice U64) (home : Usize) (coord : U64)
    (hlen : home.val + 8 ≤ coords.length) :
    ∃ r, hashtrinity.overflow.prefix_masks coords home coord = ok r := by
  obtain ⟨r, hr, _, _⟩ := Apeiron.PrefixMasks.prefix_masks_probe_independent coords home coord hlen
  exact ⟨r, hr⟩

/-- The capped-probe tail loop is total below the cap. -/
theorem probe_rest_with_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (s : Tbl KS) (c : U64) (j limit : U32) (q : hashtrinity.overflow.ByKey U64)
    (hb : 0 < s.bits.val) (hb64 : s.bits.val ≤ 64) (hlen : s.coords.length = 2 ^ s.bits.val)
    (hjl : j.val ≤ limit.val) (hlim : limit.val ≤ limN s.bits.val) :
    ∃ r, hashtrinity.overflow.Table.probe_rest_with core.cmp.EqU64 CI ks (ConfI ks) s c j limit q
      = ok r := by
  unfold hashtrinity.overflow.Table.probe_rest_with hashtrinity.overflow.Table.probe_rest_with_loop
  refine loop_tot_inv
    (fun j => hashtrinity.overflow.Table.probe_rest_with_loop.body core.cmp.EqU64 CI ks
      (ConfI ks) s c limit q j)
    (fun j => j.val ≤ limit.val) (fun j => limit.val - j.val) ?_ ?_ j hjl
  · intro j hj
    unfold hashtrinity.overflow.Table.probe_rest_with_loop.body
    by_cases hjl' : j < limit
    · rw [if_pos hjl']
      have hjlt : j.val < limit.val := (UScalar.lt_equiv _ _).mp hjl'
      obtain ⟨i, hid, hilt⟩ := descend_total ks s c j hb hb64 hlen (lt_of_lt_of_le hjlt hlim)
      rw [hid]
      simp only [bind_tc_ok]
      obtain ⟨sc, hsc⟩ := get_unchecked_ex s.coords i hilt
      rw [hsc]
      simp only [bind_tc_ok]
      by_cases hsce : sc = c
      · rw [if_pos hsce]
        rw [confirm_ok' ks hm]
        simp only [bind_tc_ok]
        rw [if_pos trivial]
        exact ⟨_, rfl⟩
      · rw [if_neg hsce]
        by_cases hse : sc = hashtrinity.EMPTY
        · rw [if_pos hse]
          exact ⟨_, rfl⟩
        · rw [if_neg hse]
          obtain ⟨j1, hj1, _⟩ := uadd_tot j 1#u32
            (by rw [val_1_u32]; exact u32_sum_lt j.val 1 (by have := limN_le_16 s.bits.val; omega) (by norm_num))
          rw [hj1]
          simp only [bind_tc_ok]
          exact ⟨_, rfl⟩
    · rw [if_neg hjl']
      exact ⟨_, rfl⟩
  · intro j j' hj hcf
    unfold hashtrinity.overflow.Table.probe_rest_with_loop.body at hcf
    by_cases hjl' : j < limit
    · rw [if_pos hjl'] at hcf
      obtain ⟨i, _, hcf⟩ := bind_inv hcf
      obtain ⟨sc, _, hcf⟩ := bind_inv hcf
      by_cases hsce : sc = c
      · rw [if_pos hsce] at hcf
        obtain ⟨b, hb, hcf⟩ := bind_inv hcf
        rw [confirm_ok ks hm] at hb
        have hb' : b = true := (Result.ok_injective hb).symm
        rw [if_pos hb'] at hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
      · rw [if_neg hsce] at hcf
        by_cases hse : sc = hashtrinity.EMPTY
        · rw [if_pos hse] at hcf
          exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
        · rw [if_neg hse] at hcf
          obtain ⟨j1, hj1, hcf⟩ := bind_inv hcf
          have hh := Result.ok_injective hcf
          simp only [ControlFlow.cont.injEq] at hh
          subst hh
          have hj1v := (uadd_ok hj1).1
          rw [val_1_u32] at hj1v
          have hjlt : j.val < limit.val := (UScalar.lt_equiv _ _).mp hjl'
          exact ⟨by omega, by omega⟩
    · rw [if_neg hjl'] at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)

/-! ## The capped-probe loops -/

theorem u32_tz_ex (m : U32) : ∃ t, core.num.U32.trailing_zeros m = ok t := ⟨_, rfl⟩

theorem u32_tz_le7 (m : U32) (hm : m.val < 256) (hm0 : m.val ≠ 0) (t : U32)
    (ht : core.num.U32.trailing_zeros m = ok t) : t.val ≤ 7 := by
  obtain ⟨_, htb, _⟩ := u32_tz_spec m hm0 t ht
  by_contra h
  have h8 : 8 ≤ t.val := by omega
  have hm8 : m.val < 2 ^ 8 := by
    have : (2:Nat)^8 = 256 := by norm_num
    omega
  have hlt : m.val < 2 ^ t.val := lt_of_lt_of_le hm8 (Nat.pow_le_pow_right (by norm_num) h8)
  rw [Nat.testBit_eq_false_of_lt hlt] at htb
  exact absurd htb (by simp)

/-- A `home`-anchored lane index plus a small offset stays in the word. -/
theorem add_lt_numBits (a k len b : Nat) (hlen : len = 2 ^ b)
    (hblt : b < System.Platform.numBits) (ha : a < len) (hk : k ≤ 8) :
    a + k < 2 ^ System.Platform.numBits := by
  have h1 : b ≤ System.Platform.numBits - 1 := by omega
  have h2 : (2:Nat) ^ b ≤ 2 ^ (System.Platform.numBits - 1) :=
    Nat.pow_le_pow_right (by norm_num) h1
  have h3 : (2:Nat) ^ (System.Platform.numBits - 1) < 2 ^ System.Platform.numBits :=
    Nat.pow_lt_pow_right (by norm_num) (by omega)
  have hnb : 32 ≤ System.Platform.numBits := Apeiron.PrefixMasks.nb_ge
  have h4 : 8 ≤ (2:Nat) ^ (System.Platform.numBits - 1) :=
    calc 8 = 2 ^ 3 := by norm_num
      _ ≤ 2 ^ (System.Platform.numBits - 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hnm : System.Platform.numBits - 1 + 1 = System.Platform.numBits := by omega
  have h5 : (2:Nat) ^ System.Platform.numBits = 2 ^ (System.Platform.numBits - 1) * 2 := by
    conv_lhs => rw [← hnm]
    rw [Nat.pow_succ]
  omega

theorem bits_lt_numBits (s : Slice U64) (i : U32) (hlen : s.length = 2 ^ i.val) :
    i.val < System.Platform.numBits := by
  have hlt := slice_len_lt s
  rw [hlen] at hlt
  exact (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp hlt

/-- The cold capped-probe loop is total: the prefix masks drive `Hit`/`Vacant`, and the no-hit,
no-empty tail defers to the total `probe_rest_with`. -/
theorem probe_capped_cold_loop_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (s : Slice U64) (s1 : Slice (core.mem.maybe_uninit.MaybeUninit I64)) (t : KS)
    (ss : hashtrinity.side.SideStore U64 I64 KS) (o : Option Ov) (i : U32) (i1 i2 : Usize)
    (pd : core.marker.PhantomData U64) (coord : U64) (home : Usize)
    (q : hashtrinity.overflow.ByKey U64) (limit : U32)
    (hbits : 0 < i.val) (hlen : s.length = 2 ^ i.val) (hhome8 : home.val + 8 ≤ s.length)
    (hlim8 : 8 ≤ limit.val) (hlim : limit.val ≤ limN i.val)
    (hits empties : U32) (hhits : hits.val < 256) (hemp : empties.val < 256) :
    ∃ r, hashtrinity.overflow.Table.probe_capped_cold_loop core.cmp.EqU64 CI ks (ConfI ks)
      s s1 t ss o i i1 i2 pd coord home q limit hits empties = ok r := by
  unfold hashtrinity.overflow.Table.probe_capped_cold_loop
  refine loop_tot_inv _ (fun h => h.val < 256) (fun h => h.val) ?_ ?_ hits hhits
  · intro h1 hlt256
    unfold hashtrinity.overflow.Table.probe_capped_cold_loop.body
    simp only [lift, bind_tc_ok]
    have hblt := bits_lt_numBits s i hlen
    by_cases hnz : (h1 != 0#u32) = true
    · rw [if_pos hnz]
      obtain ⟨tz, htz⟩ := u32_tz_ex h1
      rw [htz]
      simp only [bind_tc_ok]
      have htz7 : tz.val ≤ 7 := u32_tz_le7 h1 hlt256 (ne_zero_of_bne h1 hnz) tz htz
      obtain ⟨slot, hslot, _⟩ := uadd_tot home (UScalar.cast .Usize tz)
        (by rw [cast_usize_u32]; exact add_lt_numBits home.val tz.val s.length i.val hlen hblt (by omega) (by omega))
      rw [hslot]
      simp only [bind_tc_ok]
      rw [confirm_ok' ks hm]
      simp only [bind_tc_ok]
      rw [if_pos trivial]
      exact ⟨_, rfl⟩
    · rw [if_neg hnz]
      by_cases hne : (empties != 0#u32) = true
      · rw [if_pos hne]
        obtain ⟨tz, htz⟩ := u32_tz_ex empties
        rw [htz]
        simp only [bind_tc_ok]
        have htz7 : tz.val ≤ 7 := u32_tz_le7 empties hemp (ne_zero_of_bne empties hne) tz htz
        obtain ⟨slot, hslot, _⟩ := uadd_tot home (UScalar.cast .Usize tz)
          (by rw [cast_usize_u32]; exact add_lt_numBits home.val tz.val s.length i.val hlen hblt (by omega) (by omega))
        rw [hslot]
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
      · rw [if_neg hne]
        have hb64 : i.val ≤ 64 := by
          have := slice_len_lt s
          rw [hlen] at this
          have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp this
          rcases System.Platform.numBits_eq with h | h <;> omega
        obtain ⟨r, hr⟩ := probe_rest_with_total ks hm
          { coords := s, vals := s1, keys := t, side := ss, overflow := o, bits := i,
            len := i1, tombs := i2, _k := pd } coord hashtrinity.LINEAR_G limit q
          hbits hb64 hlen (by rw [LINEAR_G_val]; exact hlim8) hlim
        rw [hr]
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
  · intro h1 h1' hlt256 hcf
    exfalso
    unfold hashtrinity.overflow.Table.probe_capped_cold_loop.body at hcf
    simp only [lift, bind_tc_ok] at hcf
    by_cases hnz : (h1 != 0#u32) = true
    · rw [if_pos hnz] at hcf
      obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
      obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
      obtain ⟨b, hb, hcf⟩ := bind_inv hcf
      rw [confirm_ok' ks hm] at hb
      have hb' : b = true := (Result.ok_injective hb).symm
      rw [if_pos hb'] at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
    · rw [if_neg hnz] at hcf
      by_cases hne : (empties != 0#u32) = true
      · rw [if_pos hne] at hcf
        obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
        obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
      · rw [if_neg hne] at hcf
        obtain ⟨r, hr, hcf⟩ := bind_inv hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)

theorem probe_limit_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) (s : Tbl KS)
    (hb64 : s.bits.val ≤ 64) :
    ∃ l, hashtrinity.overflow.Table.probe_limit core.cmp.EqU64 CI ks s = ok l := by
  unfold hashtrinity.overflow.Table.probe_limit
  obtain ⟨w, hw⟩ := window_steps_total ks s hb64
  rw [hw]
  simp only [bind_tc_ok]
  by_cases hlt : w.val < hashtrinity.PROBE_CAP.val
  · refine ⟨w, ?_⟩
    unfold core.cmp.Ord.min.trait_default core.cmp.Ord.min.default core.cmp.Ord.min_body
    simp only [core.cmp.impls.PartialOrdU32.lt, liftFun2, bind_tc_ok]
    rw [if_pos (decide_eq_true_iff.mpr hlt)]
  · refine ⟨hashtrinity.PROBE_CAP, ?_⟩
    unfold core.cmp.Ord.min.trait_default core.cmp.Ord.min.default core.cmp.Ord.min_body
    simp only [core.cmp.impls.PartialOrdU32.lt, liftFun2, bind_tc_ok]
    rw [if_neg (fun hh => hlt (decide_eq_true_iff.mp hh))]

/-- The cold capped probe is total. -/
theorem probe_capped_cold_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64)
    (hbits : 0 < s.bits.val) (hb64 : s.bits.val ≤ 64) (hlen : s.coords.length = 2 ^ s.bits.val)
    (hhome : home.val < s.coords.length) :
    ∃ r, hashtrinity.overflow.Table.probe_capped_cold core.cmp.EqU64 CI ks (ConfI ks) s c home q
      = ok r := by
  unfold hashtrinity.overflow.Table.probe_capped_cold
  simp only [lift, bind_tc_ok]
  obtain ⟨lim, hlimok⟩ := probe_limit_total ks s hb64
  rw [hlimok]
  simp only [bind_tc_ok]
  have hlimv : lim.val = limN s.bits.val := probe_limit_val ks s lim hbits hlimok
  have hlim8 : 8 ≤ limN s.bits.val := by
    unfold limN
    have hd : 1 ≤ 64 / s.bits.val := Nat.div_pos hb64 hbits
    omega
  by_cases h8 : lim >= hashtrinity.LINEAR_G
  · rw [if_pos h8]
    have hblt := bits_lt_numBits s.coords s.bits hlen
    obtain ⟨i, hi, _⟩ := uadd_tot home (UScalar.cast .Usize hashtrinity.LINEAR_G)
      (by rw [cast_usize_u32, LINEAR_G_val]
          exact add_lt_numBits home.val 8 s.coords.length s.bits.val hlen hblt hhome (by norm_num))
    rw [hi]
    simp only [bind_tc_ok]
    by_cases hle : i <= Slice.len s.coords
    · rw [if_pos hle]
      have hivi : i.val = home.val + 8 := by
        rw [(uadd_ok hi).1, cast_usize_u32, LINEAR_G_val]
      have hle' : home.val + 8 ≤ s.coords.length := by
        rw [UScalar.le_equiv, Slice.len_val] at hle
        rw [hivi] at hle; exact hle
      obtain ⟨hits, empties, hpm, hh256, he256, _⟩ :=
        Apeiron.PrefixMasks.prefix_masks_spec s.coords home c hle'
      rw [hpm]
      simp only [bind_tc_ok]
      obtain ⟨r, hr⟩ := probe_capped_cold_loop_total ks hm s.coords s.vals s.keys s.side s.overflow
        s.bits s.len s.tombs s._k c home q lim hbits hlen hle'
        (by rw [hlimv]; exact hlim8) (by rw [hlimv]) hits empties hh256 he256
      exact ⟨r, hr⟩
    · rw [if_neg hle]
      obtain ⟨r, hr⟩ := probe_rest_with_total ks hm s c 0#u32 lim q hbits hb64 hlen
        (by simp) (by rw [hlimv])
      exact ⟨r, hr⟩
  · rw [if_neg h8]
    obtain ⟨r, hr⟩ := probe_rest_with_total ks hm s c 0#u32 lim q hbits hb64 hlen
      (by simp) (by rw [hlimv])
    exact ⟨r, hr⟩

/-- The hot capped-probe loop is total; its no-hit, no-empty tail defers to the cold probe. -/
theorem probe_capped_with_loop_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (s : Slice U64) (s1 : Slice (core.mem.maybe_uninit.MaybeUninit I64)) (t : KS)
    (ss : hashtrinity.side.SideStore U64 I64 KS) (o : Option Ov) (i : U32) (i1 i2 : Usize)
    (pd : core.marker.PhantomData U64) (coord : U64) (home : Usize)
    (q : hashtrinity.overflow.ByKey U64)
    (hbits : 0 < i.val) (hlen : s.length = 2 ^ i.val) (hhome : home.val < s.length)
    (hits empties : U32) (hhits : hits.val < 256) (hemp : empties.val < 256) :
    ∃ r, hashtrinity.overflow.Table.probe_capped_with_loop core.cmp.EqU64 CI ks (ConfI ks)
      s s1 t ss o i i1 i2 pd coord home q hits empties = ok r := by
  unfold hashtrinity.overflow.Table.probe_capped_with_loop
  refine loop_tot_inv _ (fun h => h.val < 256) (fun h => h.val) ?_ ?_ hits hhits
  · intro h1 hlt256
    unfold hashtrinity.overflow.Table.probe_capped_with_loop.body
    simp only [lift, bind_tc_ok]
    have hblt := bits_lt_numBits s i hlen
    have hb64 : i.val ≤ 64 := by
      have := slice_len_lt s
      rw [hlen] at this
      have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp this
      rcases System.Platform.numBits_eq with h | h <;> omega
    by_cases hnz : (h1 != 0#u32) = true
    · rw [if_pos hnz]
      obtain ⟨tz, htz⟩ := u32_tz_ex h1
      rw [htz]
      simp only [bind_tc_ok]
      have htz7 : tz.val ≤ 7 := u32_tz_le7 h1 hlt256 (ne_zero_of_bne h1 hnz) tz htz
      obtain ⟨slot, hslot, _⟩ := uadd_tot home (UScalar.cast .Usize tz)
        (by rw [cast_usize_u32]; exact add_lt_numBits home.val tz.val s.length i.val hlen hblt hhome (by omega))
      rw [hslot]
      simp only [bind_tc_ok]
      rw [confirm_ok' ks hm]
      simp only [bind_tc_ok]
      rw [if_pos trivial]
      exact ⟨_, rfl⟩
    · rw [if_neg hnz]
      by_cases hne : (empties != 0#u32) = true
      · rw [if_pos hne]
        obtain ⟨tz, htz⟩ := u32_tz_ex empties
        rw [htz]
        simp only [bind_tc_ok]
        have htz7 : tz.val ≤ 7 := u32_tz_le7 empties hemp (ne_zero_of_bne empties hne) tz htz
        obtain ⟨slot, hslot, _⟩ := uadd_tot home (UScalar.cast .Usize tz)
          (by rw [cast_usize_u32]; exact add_lt_numBits home.val tz.val s.length i.val hlen hblt hhome (by omega))
        rw [hslot]
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
      · rw [if_neg hne]
        obtain ⟨r, hr⟩ := probe_capped_cold_total ks hm
          { coords := s, vals := s1, keys := t, side := ss, overflow := o, bits := i,
            len := i1, tombs := i2, _k := pd } coord home q hbits hb64 hlen hhome
        rw [hr]
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
  · intro h1 h1' hlt256 hcf
    exfalso
    unfold hashtrinity.overflow.Table.probe_capped_with_loop.body at hcf
    simp only [lift, bind_tc_ok] at hcf
    by_cases hnz : (h1 != 0#u32) = true
    · rw [if_pos hnz] at hcf
      obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
      obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
      obtain ⟨b, hb, hcf⟩ := bind_inv hcf
      rw [confirm_ok' ks hm] at hb
      have hb' : b = true := (Result.ok_injective hb).symm
      rw [if_pos hb'] at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
    · rw [if_neg hnz] at hcf
      by_cases hne : (empties != 0#u32) = true
      · rw [if_pos hne] at hcf
        obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
        obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
      · rw [if_neg hne] at hcf
        obtain ⟨r, hr, hcf⟩ := bind_inv hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)

/-- The hot capped probe is total. -/
theorem probe_capped_with_total {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64)
    (hbits : 0 < s.bits.val) (hb64 : s.bits.val ≤ 64) (hlen : s.coords.length = 2 ^ s.bits.val)
    (hhome : home.val < s.coords.length) :
    ∃ r, hashtrinity.overflow.Table.probe_capped_with core.cmp.EqU64 CI ks (ConfI ks) s c home q
      = ok r := by
  unfold hashtrinity.overflow.Table.probe_capped_with
  simp only [lift, bind_tc_ok]
  have hblt := bits_lt_numBits s.coords s.bits hlen
  obtain ⟨i, hi, _⟩ := uadd_tot home (UScalar.cast .Usize hashtrinity.LINEAR_G)
    (by rw [cast_usize_u32, LINEAR_G_val]
        exact add_lt_numBits home.val 8 s.coords.length s.bits.val hlen hblt hhome (by norm_num))
  rw [hi]
  simp only [bind_tc_ok]
  by_cases hle : i <= Slice.len s.coords
  · rw [if_pos hle]
    have hivi : i.val = home.val + 8 := by
      rw [(uadd_ok hi).1, cast_usize_u32, LINEAR_G_val]
    have hle' : home.val + 8 ≤ s.coords.length := by
      rw [UScalar.le_equiv, Slice.len_val] at hle
      rw [hivi] at hle; exact hle
    obtain ⟨hits, empties, hpm, hh256, he256, _⟩ :=
      Apeiron.PrefixMasks.prefix_masks_spec s.coords home c hle'
    rw [hpm]
    simp only [bind_tc_ok]
    obtain ⟨r, hr⟩ := probe_capped_with_loop_total ks hm s.coords s.vals s.keys s.side s.overflow
      s.bits s.len s.tombs s._k c home q hbits hlen hhome hits empties hh256 he256
    exact ⟨r, hr⟩
  · rw [if_neg hle]
    exact probe_capped_cold_total ks hm s c home q hbits hb64 hlen hhome

/-! ## The overflow-tree loops -/

/-- `Overflow::find_node_loop` terminates on an embedded tree: each `cont` descends to a strictly
smaller subtree shape. -/
theorem find_node_loop_total (ns : alloc.vec.Vec ONode) (c : U64) :
    ∀ (T : OTree) (cur : U32), Emb ns.val cur T →
      ∃ r, hashtrinity.overflow.Overflow.find_node_loop ns c cur = ok r := by
  intro T
  induction T with
  | leaf =>
    intro cur hE
    have hcur : cur = hashtrinity.overflow.NIL := (NIL_eq_iff cur).mpr hE
    subst hcur
    refine ⟨hashtrinity.overflow.NIL, ?_⟩
    unfold hashtrinity.overflow.Overflow.find_node_loop
    rw [loop.eq_def]
    unfold hashtrinity.overflow.Overflow.find_node_loop.body
    by_cases hn : (hashtrinity.overflow.NIL != hashtrinity.overflow.NIL) = true
    · exfalso; simp at hn
    · rw [if_neg hn]
      simp only [bind_ok]
  | node l i r ihl ihr =>
    intro cur hE
    obtain ⟨hval, hine, nd, hnd, hl, hr⟩ := hE
    have hcurlt : cur.val < ns.val.length := by
      rw [hval]; exact (List.getElem?_eq_some_iff.mp hnd).1
    have hcastlt : (UScalar.cast .Usize cur).val < ns.val.length := by
      rw [cast_usize_u32]; exact hcurlt
    have nd_eq : ∀ n, alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) ns
        (UScalar.cast .Usize cur) = ok n → n = nd := by
      intro n hn
      have := vec_index_ok _ _ _ hn
      rw [cast_usize_u32, hval, hnd] at this
      exact (Option.some.inj this).symm
    unfold hashtrinity.overflow.Overflow.find_node_loop
    rw [loop.eq_def]
    unfold hashtrinity.overflow.Overflow.find_node_loop.body
    simp only [lift, bind_tc_ok]
    have hnz : (cur != hashtrinity.overflow.NIL) = true := by
      have h : cur.val ≠ nilN := by rw [hval]; exact hine
      have h2 : cur ≠ hashtrinity.overflow.NIL := fun hh => h ((NIL_eq_iff cur).mp hh)
      simpa using h2
    rw [if_pos hnz]
    obtain ⟨n, hn⟩ := vec_index_ex ns (UScalar.cast .Usize cur) hcastlt
    rw [hn]
    simp only [bind_tc_ok]
    by_cases heq : c = n.key
    · rw [if_pos heq]
      rw [bind_ok]
      exact ⟨cur, rfl⟩
    · rw [if_neg heq]
      by_cases hlt : c < n.key
      · rw [if_pos hlt]
        rw [bind_ok]
        rw [nd_eq n hn]
        simp only []
        simpa only [hashtrinity.overflow.Overflow.find_node_loop,
          hashtrinity.overflow.Overflow.find_node_loop.body, lift, bind_tc_ok] using ihl nd.left hl
      · rw [if_neg hlt]
        rw [bind_ok]
        rw [nd_eq n hn]
        simp only []
        simpa only [hashtrinity.overflow.Overflow.find_node_loop,
          hashtrinity.overflow.Overflow.find_node_loop.body, lift, bind_tc_ok] using ihr nd.right hr

/-- `Overflow::find_entry_loop` terminates, and a returned entry index is a valid index. -/
theorem find_entry_loop_total (k : U64) (n : U32) (entries : alloc.vec.Vec (U64 × Option I64))
    (e : Usize) (he : e.val ≤ entries.val.length) (hlen32 : entries.val.length ≤ 2 ^ 32) :
    ∃ r, hashtrinity.overflow.Overflow.find_entry_loop CI core.cmp.EqU64 k n entries e = ok r ∧
      (∀ n' e', r = some (n', e') → n' = n ∧ e'.val < entries.val.length) := by
  unfold hashtrinity.overflow.Overflow.find_entry_loop
  have hex : ∀ e1, ∃ cf, hashtrinity.overflow.Overflow.find_entry_loop.body CI core.cmp.EqU64 k n
      entries e1 = ok cf := by
    intro e1
    unfold hashtrinity.overflow.Overflow.find_entry_loop.body
    by_cases hl : e1 < alloc.vec.Vec.len entries
    · rw [if_pos hl]
      have hlt : e1.val < entries.val.length := by
        have := (UScalar.lt_equiv _ _).mp hl; rwa [alloc.vec.Vec.len_val] at this
      obtain ⟨tp, htp⟩ := vec_index_ex entries e1 hlt
      rw [htp]
      simp
      change ∃ cf, (if tp.1 = k then
          (do let i1 ← lift (UScalar.cast .U32 e1);
              ok (ControlFlow.done (some (n, i1))))
        else (do let e1 ← e1 + 1#usize; ok (ControlFlow.cont e1))) = ok cf
      split
      · simp only [lift, bind_tc_ok]
        exact ⟨_, rfl⟩
      · obtain ⟨e1', he1', _⟩ := uadd_tot e1 1#usize
          (by rw [val_1_usize, show UScalarTy.Usize.numBits = System.Platform.numBits from rfl]
              have := vec_len_lt entries; omega)
        rw [he1']
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
    · rw [if_neg hl]
      exact ⟨_, rfl⟩
  have hdec : ∀ e1 e1', hashtrinity.overflow.Overflow.find_entry_loop.body CI core.cmp.EqU64 k n
      entries e1 = ok (ControlFlow.cont e1') →
      entries.val.length - e1'.val < entries.val.length - e1.val := by
    intro e1 e1' hcf
    unfold hashtrinity.overflow.Overflow.find_entry_loop.body at hcf
    by_cases hl : e1 < alloc.vec.Vec.len entries
    · rw [if_pos hl] at hcf
      obtain ⟨t, _, hcf⟩ := bind_inv hcf
      obtain ⟨b, hb, hcf⟩ := bind_inv hcf
      rw [eqU64_eq] at hb
      by_cases hcase : t.1 = k
      · rw [if_pos (by rw [(Result.ok_injective hb).symm]; simp [hcase])] at hcf
        simp only [lift, bind_tc_ok] at hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
      · rw [if_neg (by rw [(Result.ok_injective hb).symm]; simp [hcase])] at hcf
        obtain ⟨e1'', he1'', hcf⟩ := bind_inv hcf
        have hcont := Result.ok_injective hcf
        simp only [ControlFlow.cont.injEq] at hcont
        subst hcont
        have he1''v := (uadd_ok he1'').1
        rw [val_1_usize] at he1''v
        have hlt : e1.val < entries.val.length := by
          have := (UScalar.lt_equiv _ _).mp hl; rwa [alloc.vec.Vec.len_val] at this
        rw [he1''v]; omega
    · rw [if_neg hl] at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
  obtain ⟨r, hr⟩ := loop_tot
    (fun (e1 : Usize) => hashtrinity.overflow.Overflow.find_entry_loop.body CI core.cmp.EqU64 k n
      entries e1)
    (fun e1 => entries.val.length - e1.val) hex hdec e
  refine ⟨r, hr, ?_⟩
  refine loop_pc (fun e1 : Usize => e1.val ≤ entries.val.length)
    (fun r => ∀ n' e', r = some (n', e') → n' = n ∧ e'.val < entries.val.length)
    (fun (e1 : Usize) => hashtrinity.overflow.Overflow.find_entry_loop.body CI core.cmp.EqU64 k n
      entries e1) ?_ e he r hr
  intro e1 he1 cf hcf
  unfold hashtrinity.overflow.Overflow.find_entry_loop.body at hcf
  by_cases hl : e1 < alloc.vec.Vec.len entries
  · rw [if_pos hl] at hcf
    obtain ⟨t, _, hcf⟩ := bind_inv hcf
    obtain ⟨b, hb, hcf⟩ := bind_inv hcf
    rw [eqU64_eq] at hb
    have hlt : e1.val < entries.val.length := by
      have := (UScalar.lt_equiv _ _).mp hl; rwa [alloc.vec.Vec.len_val] at this
    by_cases hcase : t.1 = k
    · rw [if_pos (by rw [(Result.ok_injective hb).symm]; simp [hcase])] at hcf
      simp only [lift, bind_tc_ok] at hcf
      have hcf' := (Result.ok_injective hcf).symm
      subst hcf'
      intro n' e' hsome
      simp only [Option.some.injEq, Prod.mk.injEq] at hsome
      obtain ⟨h1, h2⟩ := hsome
      exact ⟨h1.symm, by
        rw [← h2, cast_usize_u32_of_lt e1 (by omega)]
        exact hlt⟩
    · rw [if_neg (by rw [(Result.ok_injective hb).symm]; simp [hcase])] at hcf
      obtain ⟨e1'', he1'', hcf⟩ := bind_inv hcf
      have hcont := Result.ok_injective hcf
      subst hcont
      change e1''.val ≤ entries.val.length
      have he1''v := (uadd_ok he1'').1
      rw [val_1_usize] at he1''v
      rw [he1''v]; omega
  · rw [if_neg hl] at hcf
    have hcf' := (Result.ok_injective hcf).symm
    subst hcf'
    intro n' e' hsome
    simp at hsome

/-- `Overflow::find_entry` is total, and a returned entry index is valid at its node. -/
theorem find_entry_total (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T)
    (hB : BST o.nodes.val T) (hS : SingleEntries o.nodes.val T) (c : U64) :
    ∃ r, hashtrinity.overflow.Overflow.find_entry CI core.cmp.EqU64 o c c = ok r ∧
      (∀ n e, r = some (n, e) → ∃ nd, o.nodes.val[n.val]? = some nd ∧
        e.val < nd.entries.val.length) := by
  unfold hashtrinity.overflow.Overflow.find_entry hashtrinity.overflow.Overflow.find_node
  obtain ⟨n, hn⟩ := find_node_loop_total o.nodes c T o.root hE
  rw [hn]
  simp only [bind_tc_ok]
  by_cases hnil : n = hashtrinity.overflow.NIL
  · rw [if_pos hnil]
    exact ⟨none, rfl, by intro n' e' h; simp at h⟩
  · rw [if_neg hnil]
    have hspec := find_node_spec o.nodes c T o.root hE hB n hn
    have hnmem : n.val ∈ idxs T := by
      rcases hspec with ⟨hm, _⟩ | ⟨hm, _⟩
      · exact absurd hm (fun hh => hnil ((NIL_eq_iff n).mpr hh))
      · exact hm
    obtain ⟨nd, hnd, _⟩ := emb_mem hE n.val hnmem
    obtain ⟨nd', hnd', v, hent⟩ := hS n.val hnmem
    have hndnp : nd = nd' := Option.some.inj (hnd.symm.trans hnd')
    have hlen1 : nd.entries.val.length = 1 := by rw [hndnp, hent]; simp
    have hnlt : n.val < o.nodes.val.length := (List.getElem?_eq_some_iff.mp hnd).1
    obtain ⟨n1, hn1⟩ := vec_index_ex o.nodes (UScalar.cast .Usize n)
      (by rw [cast_usize_u32]; exact hnlt)
    simp only [lift, bind_tc_ok]
    rw [hn1]
    simp only [bind_tc_ok]
    have hn1_eq : n1 = nd := by
      have := vec_index_ok _ _ _ hn1
      rw [cast_usize_u32, hnd] at this
      exact (Option.some.inj this).symm
    rw [hn1_eq]
    obtain ⟨r, hr, hpost⟩ := find_entry_loop_total c n nd.entries 0#usize (by simp)
      (by rw [hlen1]; norm_num)
    refine ⟨r, hr, ?_⟩
    intro n' e' hsome
    obtain ⟨hn'n, he'⟩ := hpost n' e' hsome
    subst hn'n
    exact ⟨nd, hnd, he'⟩

/-- `Overflow::get` is total on a search tree of single entries. -/
theorem Overflow.get_total (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T)
    (hB : BST o.nodes.val T) (hS : SingleEntries o.nodes.val T) (c : U64) :
    ∃ r, hashtrinity.overflow.Overflow.get CI core.cmp.EqU64 o c c = ok r := by
  unfold hashtrinity.overflow.Overflow.get
  obtain ⟨fe, hfe, hpost⟩ := find_entry_total o T hE hB hS c
  rw [hfe]
  simp only [bind_tc_ok]
  cases fe with
  | none =>
    simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch,
      core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual,
      bind_tc_ok]
    exact ⟨_, rfl⟩
  | some ne =>
    obtain ⟨n, e⟩ := ne
    simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, lift, bind_tc_ok]
    change ∃ r, (do
        let n1 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) o.nodes
          (UScalar.cast .Usize n)
        let (_, o1) ← alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice (U64 × Option I64)) n1.entries
          (UScalar.cast .Usize e)
        core.option.Option.as_ref o1) = ok r
    obtain ⟨nd, hnd, he⟩ := hpost n e rfl
    have hnlt : n.val < o.nodes.val.length := (List.getElem?_eq_some_iff.mp hnd).1
    obtain ⟨n1, hn1⟩ := vec_index_ex o.nodes (UScalar.cast .Usize n)
      (by rw [cast_usize_u32]; exact hnlt)
    rw [hn1]
    simp only [bind_tc_ok]
    have hn1_eq : n1 = nd := by
      have := vec_index_ok _ _ _ hn1
      rw [cast_usize_u32, hnd] at this
      exact (Option.some.inj this).symm
    have he' : (UScalar.cast .Usize e).val < nd.entries.val.length := by
      rw [cast_usize_u32]; exact he
    obtain ⟨pair, hpair⟩ := vec_index_ex n1.entries (UScalar.cast .Usize e)
      (by rw [hn1_eq]; exact he')
    rw [hpair]
    obtain ⟨_, val⟩ := pair
    simp only [core.option.Option.as_ref, bind_tc_ok]
    exact ⟨val, rfl⟩

/-! ## Totality of the coordinate read -/

/-- `SideStore::find` is total. -/
theorem side_find_total (ss : hashtrinity.side.SideStore U64 I64 (hashtrinity.algebra.Stored U64))
    (c k : U64) :
    ∃ o, hashtrinity.side.SideStore.find CI SKS ss c k = ok o := by
  unfold hashtrinity.side.SideStore.find
  exact side_find_loop_total SKS SKS_matches ss c k 0#usize

/-- The main-slot value read is total whenever the slot holds an initialised value. -/
theorem read_val_total (s : Tbl (hashtrinity.algebra.Stored U64)) (i : Usize) (x : I64)
    (hv : s.vals.val[i.val]? = some (some x)) :
    ∃ r, (do
      let mu ← core.slice.Slice.get_unchecked (core.slice.index.SliceIndexUsizeSlice
        (core.mem.maybe_uninit.MaybeUninit I64)) s.vals i
      let t ← core.mem.maybe_uninit.MaybeUninit.assume_init_ref mu
      ok (some t)) = ok r := by
  obtain ⟨mu, hmu⟩ := get_unchecked_ex s.vals i (by
    have h1 : i.val < s.vals.val.length := (List.getElem?_eq_some_iff.mp hv).1
    simpa [Slice.length] using h1)
  rw [hmu]; simp only [bind_tc_ok]
  have hmu' := get_unchecked_ok s.vals i mu hmu
  have hmu_eq : mu = some x := (Option.some.inj (by rw [hv] at hmu'; exact hmu')).symm
  rw [hmu_eq]
  simp only [core.mem.maybe_uninit.MaybeUninit.assume_init_ref, bind_tc_ok]
  exact ⟨some x, rfl⟩

/-- The extracted coordinate read is total on a table representing `m`: every branch returns `ok`. -/
theorem get_coord_total (s : Tbl (hashtrinity.algebra.Stored U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : TableRep s m live) (k : U64) :
    ∃ r, hashtrinity.Table.get_coord core.cmp.EqU64 CI SKS s k = ok r := by
  have hb := hR.1.bits_pos
  have hlen := hR.1.len_pow
  have hb64 := bits_le_64 SKS s hlen
  unfold hashtrinity.Table.get_coord
  rw [coord_of_ok, bind_tc_ok, is_sentinel_ok, bind_tc_ok]
  by_cases hs : IsSent k
  · rw [if_pos (by simp [hs])]
    obtain ⟨o, ho⟩ := side_find_total s.side k k
    rw [ho]; simp only [bind_tc_ok]
    cases o with
    | none =>
      simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok,
        core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]
      exact ⟨none, rfl⟩
    | some i =>
      obtain ⟨hsome, _⟩ := side_find_spec SKS SKS_matches s.side k k (some i) ho
      simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok]
      have hci := hsome i rfl
      obtain ⟨_, _, hv⟩ := hR.1.side_val i.val k hci
      obtain ⟨mu, hmu⟩ := get_unchecked_ex (alloc.vec.Vec.deref s.side.vals) i
        (by
          have h1 : i.val < s.side.vals.val.length := (List.getElem?_eq_some_iff.mp hv).1
          simpa [alloc.vec.Vec.deref, Slice.length] using h1)
      rw [hmu]; simp only [bind_tc_ok]
      have hmu' := get_unchecked_ok _ _ _ hmu
      have hmu'' : s.side.vals.val[i.val]? = some mu := by
        simpa [alloc.vec.Vec.deref] using hmu'
      have hmu_eq : mu = some (m k) := (Option.some.inj (by rw [hv] at hmu''; exact hmu'')).symm
      rw [hmu_eq]
      simp only [core.mem.maybe_uninit.MaybeUninit.assume_init_ref, bind_tc_ok]
      exact ⟨some (m k), rfl⟩
  · rw [if_neg (by simp [hs])]
    obtain ⟨home, hhome_ok, hhome⟩ := home_total SKS s k hb hb64 hlen
    rw [hhome_ok]; simp only [bind_tc_ok]
    obtain ⟨sc, hsc⟩ := get_unchecked_ex s.coords home hhome
    rw [hsc]; simp only [bind_tc_ok]
    by_cases hc : sc = k
    · rw [if_pos hc]
      have hm : hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.matches
          CI core.cmp.EqU64 s.keys home k k = ok true := stored_matches_ok s.keys home k k
      rw [hm]; simp only [bind_tc_ok]
      rw [if_pos trivial]
      have hscv : s.coords.val[home.val]? = some k := by
        have := get_unchecked_ok _ _ _ hsc
        rw [hc] at this; exact this
      obtain ⟨_, hv⟩ := hR.1.main_val home.val k hscv hs
      exact read_val_total s home (m k) hv
    · rw [if_neg hc]
      simp only [lift, bind_tc_ok]
      by_cases hci : home = UScalar.cast .Usize k
      · rw [if_pos hci]
        by_cases hse : sc = hashtrinity.EMPTY
        · rw [if_pos hse]; exact ⟨none, rfl⟩
        · rw [if_neg hse]
          obtain ⟨c, hcp⟩ := probe_capped_with_total SKS SKS_matches s k home { coord := k, k := k }
            hb hb64 hlen hhome
          have hcok := pcw_spec SKS SKS_matches (fun _ => avx2_spec_holds) s k home { coord := k, k := k }
            hb hhome_ok hlen c hcp
          simp only [ConfI] at hcp
          rw [hcp]; simp only [bind_tc_ok]
          cases c with
          | Hit i1 =>
            have hcv : s.coords.val[i1.val]? = some k := hcok
            obtain ⟨_, hv⟩ := hR.1.main_val i1.val k hcv hs
            exact read_val_total s i1 (m k) hv
          | Vacant _ => exact ⟨none, rfl⟩
          | Blocked =>
            simp only [core.option.Option.as_ref, bind_tc_ok]
            cases hov : s.overflow with
            | none => exact ⟨none, rfl⟩
            | some o1 =>
              obtain ⟨T, hE, hB, _, _, hall⟩ := ovInv_some SKS hR.1 hov
              have hS : SingleEntries o1.nodes.val T := by
                intro i hi
                obtain ⟨nd, hnd, hent, _⟩ := hall i hi
                exact ⟨nd, hnd, m nd.key, hent⟩
              obtain ⟨r, hr⟩ := Overflow.get_total o1 T hE hB hS k
              exact ⟨r, hr⟩
      · rw [if_neg hci]
        obtain ⟨c, hcp⟩ := probe_capped_with_total SKS SKS_matches s k home { coord := k, k := k }
          hb hb64 hlen hhome
        have hcok := pcw_spec SKS SKS_matches (fun _ => avx2_spec_holds) s k home { coord := k, k := k }
          hb hhome_ok hlen c hcp
        simp only [ConfI] at hcp
        rw [hcp]; simp only [bind_tc_ok]
        cases c with
        | Hit i1 =>
          have hcv : s.coords.val[i1.val]? = some k := hcok
          obtain ⟨_, hv⟩ := hR.1.main_val i1.val k hcv hs
          exact read_val_total s i1 (m k) hv
        | Vacant _ => exact ⟨none, rfl⟩
        | Blocked =>
          simp only [core.option.Option.as_ref, bind_tc_ok]
          cases hov : s.overflow with
          | none => exact ⟨none, rfl⟩
          | some o1 =>
            obtain ⟨T, hE, hB, _, _, hall⟩ := ovInv_some SKS hR.1 hov
            have hS : SingleEntries o1.nodes.val T := by
              intro i hi
              obtain ⟨nd, hnd, hent, _⟩ := hall i hi
              exact ⟨nd, hnd, m nd.key, hent⟩
            obtain ⟨r, hr⟩ := Overflow.get_total o1 T hE hB hS k
            exact ⟨r, hr⟩

/-- **Totality of the executed read (`Stored<u64>`).**  On a `MapRep` state, the read returns `ok`
for every key: each executed loop terminates (`loop_tot_inv`), so `hok` is discharged. -/
theorem get_ok_of_mapRep (s : Tbl (hashtrinity.algebra.Stored U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : MapRep s m live) (k : U64) :
    ∃ r, word_hashtrinity_get s k = ok r := by
  obtain ⟨o, ho⟩ := get_coord_total s m live hR.1 k
  unfold word_hashtrinity_get hashtrinity.Table.get
  rw [ho]; simp only [bind_tc_ok]
  cases o with
  | none => exact ⟨none, rfl⟩
  | some v => exact ⟨some v, rfl⟩

end Apeiron.MapExecuted

