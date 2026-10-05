-- Row: `apeiron-ht-extract-DjjuLApKg3g7zaM9kRUkUE` (residual): the AVX2 and scalar branches of `prefix_masks` agree.
import ApeironWord
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases
open Aeneas Aeneas.Std Result

/-!
# `prefix_masks`: the AVX2 branch equals the scalar branch

`apeiron_word.hashtrinity.overflow.prefix_masks` (`proof/ApeironWord.lean`, source
`src/hashtrinity/overflow.rs:66-96`) branches on `std_detect...avx2_probe`, an opaque `Bool`: either
`prefix_masks_avx2` (the modelled `_mm256_*` intrinsics, `__m256i` = four `U64` lanes) or the scalar
`prefix_masks_loop`.  Both are shown to compute the same pair of 8-bit masks, so the result does not
depend on the probe.

Not covered: that LLVM emits the intrinsics the model describes; the model of `RawPtr` (a window into
the little-endian bytes of the slice); `unsafe` reads staying in the window (the model fails there).
-/

namespace Apeiron.PrefixMasks
open apeiron_word Lean.Order Aeneas.Data.Coinductive

/-! ## Loop reasoning (the fixpoint-induction principle of `Count/Aggregate.lean`) -/

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

/-! ## The pointer model on `[u64]` -/

theorem u64_ne_u8 : ¬ ((UScalar UScalarTy.U64 : Type) = UScalar UScalarTy.U8) := by
  intro h
  have e : ∀ ty : UScalarTy, Nat.card (UScalar ty) = 2 ^ ty.numBits := by
    intro ty
    have : UScalar ty ≃ BitVec ty.numBits :=
      ⟨fun x => x.bv, fun b => ⟨b⟩, fun x => by cases x; rfl, fun b => rfl⟩
    rw [Nat.card_congr (this.trans BitVec.equivFin.toEquiv)]
    simp
  have h1 := e .U64
  rw [h, e .U8] at h1
  have : (2:ℕ)^UScalarTy.U8.numBits = 2^UScalarTy.U64.numBits := h1
  simp [UScalarTy.numBits] at this

theorem as_ptr_u64 (s : Slice U64) :
    core.slice.Slice.as_ptr s = ok ⟨(s.val.map U64.leBytes).flatten, 0⟩ := by
  simp [core.slice.Slice.as_ptr, u64_ne_u8]

theorem ptrElemSize_u64 : ptrElemSize U64 = 8 := by
  simp [ptrElemSize, u64_ne_u8]

theorem ptr_add_u64 (p : ConstRawPtr U64) (n : Usize) :
    core.ptr.const_ptr.RawPtrConstT.add p n =
      if p.off + n.val * 8 ≤ p.bytes.length then ok ⟨p.bytes, p.off + n.val * 8⟩
      else fail .undef := by
  unfold core.ptr.const_ptr.RawPtrConstT.add
  rw [ptrElemSize_u64]

theorem leBytes_length (x : U64) : (U64.leBytes x).length = 8 := by
  simp [U64.leBytes]

theorem leBytes_getD (x : U64) (j : Nat) (hj : j < 8) (d : U8) :
    (U64.leBytes x).getD j d = ⟨BitVec.ofNat 8 (x.val / 2 ^ (8 * j))⟩ := by
  simp [U64.leBytes, hj, List.getD_eq_getElem?_getD]

theorem flatten_getD (l : List U64) (i j : Nat) (hi : i < l.length) (hj : j < 8) (d : U8) :
    (l.map U64.leBytes).flatten.getD (8 * i + j) d = (U64.leBytes (l[i]'hi)).getD j d := by
  induction l generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
    rw [List.map_cons, List.flatten_cons]
    cases i with
    | zero =>
      simp only [Nat.mul_zero, Nat.zero_add]
      rw [List.getD_append _ _ _ _ (by rw [leBytes_length]; exact hj)]
      rfl
    | succ i =>
      have : 8 * (i + 1) + j = (U64.leBytes x).length + (8 * i + j) := by
        rw [leBytes_length]; omega
      rw [this, List.getD_append_right _ _ _ _ (by omega)]
      simp only [Nat.add_sub_cancel_left]
      exact ih i (by simpa using hi)

theorem fold_congr (F G : Nat → Nat) (n : Nat) (h : ∀ j, j < n → F j = G j) :
    (List.range n).foldl (fun acc j => acc + F j * 2 ^ (8 * j)) 0 =
    (List.range n).foldl (fun acc j => acc + G j * 2 ^ (8 * j)) 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [ih (fun j hj => h j (by omega)), h n (by omega)]

theorem fold_mod (x : Nat) (n : Nat) :
    (List.range n).foldl (fun acc j => acc + (x / 2 ^ (8 * j) % 256) * 2 ^ (8 * j)) 0 =
      x % 2 ^ (8 * n) := by
  induction n with
  | zero => simp; exact Nat.mod_one x
  | succ n ih =>
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [ih]
    have h256 : (2:Nat) ^ (8 * (n + 1)) = 2 ^ (8 * n) * 256 := by
      rw [Nat.mul_succ, Nat.pow_add]
    rw [h256, Nat.mod_mul]
    ring

theorem u8_val (v : Nat) : (⟨BitVec.ofNat 8 v⟩ : U8).val = v % 256 := by
  show (BitVec.ofNat 8 v).toNat = v % 256
  simp [BitVec.toNat_ofNat]

theorem ofLeBytes_flat (l : List U64) (i : Nat) (hi : i < l.length) :
    U64.ofLeBytes (l.map U64.leBytes).flatten (8 * i) = l[i]'hi := by
  unfold U64.ofLeBytes
  have hF := fold_congr
    (fun j => (((l.map U64.leBytes).flatten.getD (8 * i + j) (⟨BitVec.ofNat 8 0⟩ : U8)).val))
    (fun j => (l[i]'hi).val / 2 ^ (8 * j) % 256) 8 (by
      intro j hj
      show ((l.map U64.leBytes).flatten.getD (8 * i + j) (⟨BitVec.ofNat 8 0⟩ : U8)).val = _
      rw [flatten_getD l i j hi hj, leBytes_getD _ _ hj, u8_val])
  try beta_reduce at hF
  rw [hF, fold_mod]
  have hlt : (l[i]'hi).val < 2 ^ 64 := (l[i]'hi).bv.isLt
  have : (l[i]'hi).val % 2 ^ (8 * 8) = (l[i]'hi).val := Nat.mod_eq_of_lt (by norm_num; exact hlt)
  apply UScalar.eq_of_val_eq
  change (BitVec.ofNat 64 _).toNat = _
  rw [BitVec.toNat_ofNat, this]
  exact Nat.mod_eq_of_lt hlt

theorem flat_length (l : List U64) : (l.map U64.leBytes).flatten.length = 8 * l.length := by
  induction l with
  | nil => simp
  | cons x xs ih => simp [List.flatten_cons, leBytes_length, ih]; omega

/-! ## `__m256i` lanes -/

abbrev M256 := core.core_arch.x86.__m256i

theorem cast_eq (U : Type) {T : Type} (p : ConstRawPtr T) :
    core.ptr.const_ptr.RawPtrConstT.cast U p = ok ⟨p.bytes, p.off⟩ := rfl

theorem loadu_ok (p : ConstRawPtr M256) (h : p.off + 32 ≤ p.bytes.length) :
    core.core_arch.x86.avx._mm256_loadu_si256 p =
      ok ⟨fun k => U64.ofLeBytes p.bytes (p.off + 8 * k.val)⟩ := by
  unfold core.core_arch.x86.avx._mm256_loadu_si256
  rw [if_pos h]

theorem loadu_fail (p : ConstRawPtr M256) (h : ¬ p.off + 32 ≤ p.bytes.length) :
    core.core_arch.x86.avx._mm256_loadu_si256 p = fail .undef := by
  unfold core.core_arch.x86.avx._mm256_loadu_si256
  rw [if_neg h]

/-- Lane `k` of the comparison result: all ones when the lanes are equal, else `0`. -/
def cmpv (a b : M256) : M256 :=
  ⟨fun k => if (a.lane k).val = (b.lane k).val
    then (⟨BitVec.ofNat 64 (2 ^ 64 - 1)⟩ : U64) else (⟨BitVec.ofNat 64 0⟩ : U64)⟩

/-- The `movemask_pd` result: bit `k` is the sign bit of lane `k`. -/
def mmI (f : Fin 4 → U64) : I32 :=
  ⟨BitVec.ofNat 32 ((List.range 4).foldl
    (fun acc k => if h : k < 4 then (if 2 ^ 63 ≤ (f ⟨k, h⟩).val then acc + 2 ^ k else acc)
      else acc) 0)⟩

def kv (x : U64) : M256 := ⟨fun _ => IScalar.hcast .U64 (UScalar.hcast .I64 x)⟩

def m4 (a b : M256) : U32 := IScalar.hcast .U32 (mmI (cmpv a b).lane)

def shl4v (x : U32) : U32 := ⟨x.bv.shiftLeft 4⟩

theorem set1_eq (x : I64) :
    core.core_arch.x86.avx._mm256_set1_epi64x x = ok ⟨fun _ => IScalar.hcast .U64 x⟩ := rfl

theorem cmpeq_eq (a b : M256) :
    core.core_arch.x86.avx2._mm256_cmpeq_epi64 a b = ok (cmpv a b) := rfl

theorem castsi_eq (a : M256) :
    core.core_arch.x86.avx._mm256_castsi256_pd a = ok ⟨a.lane⟩ := rfl

theorem movemask_eq (a : core.core_arch.x86.__m256d) :
    core.core_arch.x86.avx._mm256_movemask_pd a = ok (mmI a.lane) := rfl

theorem shl4_eq (x : U32) : (x <<< 4#i32 : Result U32) = ok (shl4v x) := by
  have h4 : (4#i32 : I32).val = 4 := rfl
  show UScalar.shiftLeft_IScalar x 4#i32 = _
  unfold UScalar.shiftLeft_IScalar UScalar.shiftLeft
  have h4' : (4#i32 : I32).toNat = 4 := by simp [h4]
  rw [if_pos (by omega), h4', if_pos (by decide)]
  rfl

/-- The tail of the extracted `prefix_masks_avx2`, after the two loads. -/
def avxTail (lo hi : M256) (coord : Std.U64) : Result (Std.U32 × Std.U32) := do
  let i ← lift (UScalar.hcast .I64 coord)
  let key ← core.core_arch.x86.avx._mm256_set1_epi64x i
  let i1 ← lift (UScalar.hcast .I64 hashtrinity.EMPTY)
  let emp ← core.core_arch.x86.avx._mm256_set1_epi64x i1
  let mi ← core.core_arch.x86.avx2._mm256_cmpeq_epi64 lo key
  let md ← core.core_arch.x86.avx._mm256_castsi256_pd mi
  let i2 ← core.core_arch.x86.avx._mm256_movemask_pd md
  let hits_lo ← lift (IScalar.hcast .U32 i2)
  let mi1 ← core.core_arch.x86.avx2._mm256_cmpeq_epi64 hi key
  let md1 ← core.core_arch.x86.avx._mm256_castsi256_pd mi1
  let i3 ← core.core_arch.x86.avx._mm256_movemask_pd md1
  let hits_hi ← lift (IScalar.hcast .U32 i3)
  let mi2 ← core.core_arch.x86.avx2._mm256_cmpeq_epi64 lo emp
  let md2 ← core.core_arch.x86.avx._mm256_castsi256_pd mi2
  let i4 ← core.core_arch.x86.avx._mm256_movemask_pd md2
  let empties_lo ← lift (IScalar.hcast .U32 i4)
  let mi3 ← core.core_arch.x86.avx2._mm256_cmpeq_epi64 hi emp
  let md3 ← core.core_arch.x86.avx._mm256_castsi256_pd mi3
  let i5 ← core.core_arch.x86.avx._mm256_movemask_pd md3
  let empties_hi ← lift (IScalar.hcast .U32 i5)
  let i6 ← hits_hi <<< 4#i32
  let hits ← lift (hits_lo ||| i6)
  let i7 ← empties_hi <<< 4#i32
  let empties ← lift (empties_lo ||| i7)
  ok (hits, empties)

theorem avx2_unfold (p : ConstRawPtr Std.U64) (coord : Std.U64) :
    hashtrinity.overflow.prefix_masks_avx2 p coord = (do
      let pc ← core.ptr.const_ptr.RawPtrConstT.cast M256 p
      let lo ← core.core_arch.x86.avx._mm256_loadu_si256 pc
      let p1 ← core.ptr.const_ptr.RawPtrConstT.add p 4#usize
      let p2 ← core.ptr.const_ptr.RawPtrConstT.cast M256 p1
      let hi ← core.core_arch.x86.avx._mm256_loadu_si256 p2
      avxTail lo hi coord) := by
  unfold hashtrinity.overflow.prefix_masks_avx2 avxTail
  rfl

theorem tail_eq (lo hi : M256) (coord : Std.U64) :
    avxTail lo hi coord =
      ok (m4 lo (kv coord) ||| shl4v (m4 hi (kv coord)),
          m4 lo (kv hashtrinity.EMPTY) ||| shl4v (m4 hi (kv hashtrinity.EMPTY))) := by
  unfold avxTail
  simp only [lift, bind_tc_ok, set1_eq, cmpeq_eq, castsi_eq, movemask_eq, shl4_eq]
  try rfl

/-! ## Mask lemmas -/

theorem mm_fold (N : Nat) (P : Nat → Bool) :
    ∀ n, n ≤ N →
      ((List.range n).foldl (fun acc k => if k < N then (if P k then acc + 2 ^ k else acc) else acc) 0 < 2 ^ n ∧
      ∀ k, k < n → (((List.range n).foldl (fun acc k => if k < N then (if P k then acc + 2 ^ k else acc) else acc) 0).testBit k = P k)) := by
  intro n
  induction n with
  | zero => intro _; simp
  | succ n ih =>
    intro hn
    obtain ⟨ih1, ih2⟩ := ih (by omega)
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    have hnN : n < N := by omega
    simp only [hnN, if_true]
    set F := (List.range n).foldl (fun acc k => if k < N then (if P k then acc + 2 ^ k else acc) else acc) 0 with hF
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

theorem mmI_spec (f : Fin 4 → U64) :
    (IScalar.hcast .U32 (mmI f)).val < 16 ∧
    ∀ k (hk : k < 4), (IScalar.hcast .U32 (mmI f)).val.testBit k = decide (2 ^ 63 ≤ (f ⟨k, hk⟩).val) := by
  set P : Nat → Bool := fun k => if hk : k < 4 then decide (2 ^ 63 ≤ (f ⟨k, hk⟩).val) else false with hP
  have hfun : (fun acc k => if h : k < 4 then (if 2 ^ 63 ≤ (f ⟨k, h⟩).val then acc + 2 ^ k else acc)
      else acc) = (fun acc k => if k < 4 then (if P k then acc + 2 ^ k else acc) else acc) := by
    funext acc k
    by_cases hk : k < 4
    · simp [hk, hP]
    · simp [hk]
  have hi : mmI f = ⟨BitVec.ofNat 32 ((List.range 4).foldl
      (fun acc k => if k < 4 then (if P k then acc + 2 ^ k else acc) else acc) 0)⟩ := by
    unfold mmI
    rw [hfun]
  obtain ⟨b1, b2⟩ := mm_fold 4 P 4 (le_refl _)
  set S := (List.range 4).foldl (fun acc k => if k < 4 then (if P k then acc + 2 ^ k else acc) else acc) 0 with hS
  have hval : (IScalar.hcast .U32 (mmI f)).val = S := by
    rw [hi]
    have : S < 2 ^ 4 := b1
    simp only [IScalar.hcast, UScalar.val]
    have e32 : UScalarTy.U32.numBits = 32 := rfl
    rw [e32]
    simp
    omega
  rw [hval]
  refine ⟨by omega, fun k hk => ?_⟩
  rw [b2 k hk, hP]
  simp [hk]

theorem ones_val : (⟨BitVec.ofNat 64 (2 ^ 64 - 1)⟩ : U64).val = 2 ^ 64 - 1 := by
  show (BitVec.ofNat 64 (2 ^ 64 - 1)).toNat = _
  simp [BitVec.toNat_ofNat]

theorem zero_val : (⟨BitVec.ofNat 64 0⟩ : U64).val = 0 := by
  show (BitVec.ofNat 64 0).toNat = 0
  simp [BitVec.toNat_ofNat]

theorem cmp_bit (a b : U64) :
    decide (2 ^ 63 ≤ ((if a.val = b.val then (⟨BitVec.ofNat 64 (2 ^ 64 - 1)⟩ : U64)
      else (⟨BitVec.ofNat 64 0⟩ : U64)).val)) = decide (a = b) := by
  by_cases h : a.val = b.val
  · have : a = b := UScalar.eq_of_val_eq h
    rw [if_pos h, ones_val]
    simp [this]
  · have : a ≠ b := fun e => h (by rw [e])
    simp [h, this, zero_val]

theorem m4_spec (a b : M256) :
    (m4 a b).val < 16 ∧ ∀ k (hk : k < 4), (m4 a b).val.testBit k = decide (a.lane ⟨k, hk⟩ = b.lane ⟨k, hk⟩) := by
  obtain ⟨h1, h2⟩ := mmI_spec (cmpv a b).lane
  refine ⟨h1, fun k hk => ?_⟩
  unfold m4
  rw [h2 k hk]
  exact cmp_bit _ _

theorem shl4v_val (x : U32) : (shl4v x).val = (x.val * 2 ^ 4) % 2 ^ 32 := by
  show (x.bv.shiftLeft 4).toNat = _
  exact BitVec.toNat_shiftLeft.trans (by rw [Nat.shiftLeft_eq]; rfl)

theorem or_val (x y : U32) : (x ||| y).val = x.val ||| y.val := by
  show (x.bv ||| y.bv).toNat = _
  exact BitVec.toNat_or _ _

theorem hcast_rt (x : U64) : IScalar.hcast .U64 (UScalar.hcast .I64 x) = x := by
  cases x with | mk bv =>
  have h1 : bv.setWidth 64 = bv := BitVec.setWidth_eq bv
  have h2 : (bv.setWidth 64).signExtend 64 = bv := by rw [h1]; exact BitVec.signExtend_eq bv
  exact congrArg UScalar.mk h2

theorem kv_lane (x : U64) (k : Fin 4) : (kv x).lane k = x := hcast_rt x

theorem tb_shl4 (b k : Nat) : (b * 2 ^ 4).testBit k = (decide (4 ≤ k) && b.testBit (k - 4)) := by
  have e : b * 2 ^ 4 = 2 ^ 4 * b + 0 := by ring
  rw [e, Nat.testBit_two_pow_mul_add b (by norm_num : (0:Nat) < 2 ^ 4) k]
  by_cases h : k < 4
  · simp [h]
  · have : 4 ≤ k := by omega
    simp [h, this]

/-- Combining the low and high four-lane masks. -/
theorem combine_spec (a b : U32) (ha : a.val < 16) (hb : b.val < 16) :
    (a ||| shl4v b).val < 256 ∧
    ∀ k, k < 4 → (a ||| shl4v b).val.testBit k = a.val.testBit k ∧
      (a ||| shl4v b).val.testBit (k + 4) = b.val.testBit k := by
  have hs : (shl4v b).val = b.val * 2 ^ 4 := by
    rw [shl4v_val]; exact Nat.mod_eq_of_lt (by omega)
  rw [or_val, hs]
  refine ⟨?_, fun k hk => ?_⟩
  · have := Nat.or_lt_two_pow (x := a.val) (y := b.val * 2 ^ 4) (n := 8) (by omega) (by omega)
    exact this
  · have hak : a.val.testBit (k + 4) = false :=
      Nat.testBit_lt_two_pow (lt_of_lt_of_le (by omega : a.val < 2 ^ 4)
        (Nat.pow_le_pow_right (by norm_num) (by omega)))
    constructor
    · rw [Nat.testBit_or, tb_shl4]
      have : ¬ (4 ≤ k) := by omega
      simp [this]
    · rw [Nat.testBit_or, tb_shl4, hak]
      simp

theorem tail_spec (lo hi : M256) (coord : Std.U64) :
    ∃ H E, avxTail lo hi coord = ok (H, E) ∧ H.val < 256 ∧ E.val < 256 ∧
    ∀ k (hk : k < 4),
      (H.val.testBit k = decide (lo.lane ⟨k, hk⟩ = coord) ∧
       H.val.testBit (k + 4) = decide (hi.lane ⟨k, hk⟩ = coord)) ∧
      (E.val.testBit k = decide (lo.lane ⟨k, hk⟩ = hashtrinity.EMPTY) ∧
       E.val.testBit (k + 4) = decide (hi.lane ⟨k, hk⟩ = hashtrinity.EMPTY)) := by
  rw [tail_eq]
  obtain ⟨a1, a2⟩ := m4_spec lo (kv coord)
  obtain ⟨b1, b2⟩ := m4_spec hi (kv coord)
  obtain ⟨c1, c2⟩ := m4_spec lo (kv hashtrinity.EMPTY)
  obtain ⟨d1, d2⟩ := m4_spec hi (kv hashtrinity.EMPTY)
  obtain ⟨e1, e2⟩ := combine_spec _ _ a1 b1
  obtain ⟨f1, f2⟩ := combine_spec _ _ c1 d1
  refine ⟨_, _, rfl, e1, f1, fun k hk => ?_⟩
  obtain ⟨g1, g2⟩ := e2 k hk
  obtain ⟨h1, h2⟩ := f2 k hk
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
  · rw [g1, a2 k hk, kv_lane]
  · rw [g2, b2 k hk, kv_lane]
  · rw [h1, c2 k hk, kv_lane]
  · rw [h2, d2 k hk, kv_lane]

/-! ## `prefix_masks_avx2` on a window -/

theorem four_val : (4#usize).val = 4 := by simp

theorem hadd4 (p : ConstRawPtr Std.U64) (h : p.off + 4 * 8 ≤ p.bytes.length) :
    core.ptr.const_ptr.RawPtrConstT.add p 4#usize = ok ⟨p.bytes, p.off + 4 * 8⟩ := by
  rw [ptr_add_u64, if_pos (by rw [four_val]; exact h), four_val]

theorem avx2_loads (p : ConstRawPtr Std.U64) (coord : Std.U64)
    (hc : p.off + 64 ≤ p.bytes.length) :
    hashtrinity.overflow.prefix_masks_avx2 p coord =
      avxTail ⟨fun k => U64.ofLeBytes p.bytes (p.off + 8 * k.val)⟩
              ⟨fun k => U64.ofLeBytes p.bytes (p.off + 4 * 8 + 8 * k.val)⟩ coord := by
  have hl1 : p.off + 32 ≤ p.bytes.length := by omega
  have hl2 : p.off + 4 * 8 + 32 ≤ p.bytes.length := by omega
  rw [avx2_unfold]
  simp only [cast_eq, bind_tc_ok]
  rw [loadu_ok ⟨p.bytes, p.off⟩ hl1, hadd4 p (by omega)]
  simp only [bind_tc_ok]
  rw [loadu_ok ⟨p.bytes, p.off + 4 * 8⟩ hl2]
  simp only [bind_tc_ok]

theorem avx2_fail (p : ConstRawPtr Std.U64) (coord : Std.U64)
    (hc : ¬ p.off + 64 ≤ p.bytes.length) :
    ∃ e, hashtrinity.overflow.prefix_masks_avx2 p coord = fail e := by
  rw [avx2_unfold]
  simp only [cast_eq, bind_tc_ok]
  by_cases h1 : p.off + 32 ≤ p.bytes.length
  · rw [loadu_ok ⟨p.bytes, p.off⟩ h1, hadd4 p (by omega)]
    simp only [bind_tc_ok]
    rw [loadu_fail ⟨p.bytes, p.off + 4 * 8⟩ (by simp only []; omega)]
    exact ⟨.undef, by simp⟩
  · rw [loadu_fail ⟨p.bytes, p.off⟩ h1]
    exact ⟨.undef, by simp⟩

theorem avx2_spec (p : ConstRawPtr Std.U64) (coord : Std.U64)
    (hc : p.off + 64 ≤ p.bytes.length) :
    ∃ H E, hashtrinity.overflow.prefix_masks_avx2 p coord = ok (H, E) ∧ H.val < 256 ∧ E.val < 256 ∧
    ∀ j, j < 8 →
      H.val.testBit j = decide (U64.ofLeBytes p.bytes (p.off + 8 * j) = coord) ∧
      E.val.testBit j = decide (U64.ofLeBytes p.bytes (p.off + 8 * j) = hashtrinity.EMPTY) := by
  obtain ⟨H, E, h1, h2, h3, h4⟩ := tail_spec
    ⟨fun k => U64.ofLeBytes p.bytes (p.off + 8 * k.val)⟩
    ⟨fun k => U64.ofLeBytes p.bytes (p.off + 4 * 8 + 8 * k.val)⟩ coord
  refine ⟨H, E, by rw [avx2_loads p coord hc]; exact h1, h2, h3, fun j hj => ?_⟩
  by_cases hj4 : j < 4
  · obtain ⟨⟨a, _⟩, ⟨c, _⟩⟩ := h4 j hj4
    exact ⟨a, c⟩
  · have hj' : j - 4 < 4 := by omega
    obtain ⟨⟨_, b⟩, ⟨_, d⟩⟩ := h4 (j - 4) hj'
    have e1 : j - 4 + 4 = j := by omega
    have e2 : p.off + 4 * 8 + 8 * (j - 4) = p.off + 8 * j := by omega
    rw [e1] at b d
    refine ⟨?_, ?_⟩
    · rw [b]
      show decide (U64.ofLeBytes p.bytes (p.off + 4 * 8 + 8 * (j - 4)) = coord) = _
      rw [e2]
    · rw [d]
      show decide (U64.ofLeBytes p.bytes (p.off + 4 * 8 + 8 * (j - 4)) = hashtrinity.EMPTY) = _
      rw [e2]

/-- The AVX2 arm of the extracted `prefix_masks`. -/
noncomputable def avxBranch (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64) :
    Result (Std.U32 × Std.U32) := do
  let p ← core.slice.Slice.as_ptr coords
  let p1 ← core.ptr.const_ptr.RawPtrConstT.add p home
  hashtrinity.overflow.prefix_masks_avx2 p1 coord

/-- The scalar arm of the extracted `prefix_masks`. -/
def scalarBranch (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64) :
    Result (Std.U32 × Std.U32) :=
  hashtrinity.overflow.prefix_masks_loop coords home coord 0#u32 0#u32 0#usize

theorem prefix_masks_eq (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64) :
    hashtrinity.overflow.prefix_masks coords home coord =
      if std_detect.detect.arch.x86.avx2_probe = true then avxBranch coords home coord
      else scalarBranch coords home coord := by
  unfold hashtrinity.overflow.prefix_masks std_detect.detect.arch.x86.__is_feature_detected.avx2
  simp only [bind_tc_ok]
  cases hP : std_detect.detect.arch.x86.avx2_probe <;> simp [avxBranch, scalarBranch]

def Spec (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64) (r : Std.U32 × Std.U32) : Prop :=
  r.1.val < 256 ∧ r.2.val < 256 ∧ ∀ j, j < 8 → ∃ x, coords.val[home.val + j]? = some x ∧
    r.1.val.testBit j = decide (x = coord) ∧ r.2.val.testBit j = decide (x = hashtrinity.EMPTY)

theorem avxBranch_spec (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hlen : home.val + 8 ≤ coords.length) :
    ∃ r, avxBranch coords home coord = ok r ∧ Spec coords home coord r := by
  have hfl := flat_length coords.val
  have hlen' : home.val + 8 ≤ coords.val.length := hlen
  have hc : (0 : Nat) + home.val * 8 + 64 ≤ ((coords.val.map U64.leBytes).flatten).length := by
    rw [hfl]; omega
  have hc0 : (0 : Nat) + home.val * 8 ≤ ((coords.val.map U64.leBytes).flatten).length := by
    omega
  have hadd : core.ptr.const_ptr.RawPtrConstT.add
      (⟨(coords.val.map U64.leBytes).flatten, 0⟩ : ConstRawPtr Std.U64) home =
      ok ⟨(coords.val.map U64.leBytes).flatten, 0 + home.val * 8⟩ := by
    rw [ptr_add_u64, if_pos hc0]
  obtain ⟨H, E, h1, h2, h3, h4⟩ := avx2_spec
    (⟨(coords.val.map U64.leBytes).flatten, 0 + home.val * 8⟩ : ConstRawPtr Std.U64) coord (by
      simpa using hc)
  refine ⟨(H, E), ?_, h2, h3, fun j hj => ?_⟩
  · unfold avxBranch
    rw [as_ptr_u64]
    simp only [bind_tc_ok]
    rw [hadd]
    simp only [bind_tc_ok]
    exact h1
  · have hx : home.val + j < coords.val.length := by omega
    refine ⟨coords.val[home.val + j]'hx, List.getElem?_eq_getElem hx, ?_⟩
    have e : 0 + home.val * 8 + 8 * j = 8 * (home.val + j) := by omega
    obtain ⟨a, b⟩ := h4 j hj
    simp only [e, ofLeBytes_flat coords.val (home.val + j) hx] at a b
    exact ⟨a, b⟩

theorem avxBranch_fail (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hlen : ¬ home.val + 8 ≤ coords.length) : ∃ e, avxBranch coords home coord = fail e := by
  have hfl := flat_length coords.val
  unfold avxBranch
  rw [as_ptr_u64]
  simp only [bind_tc_ok]
  by_cases hh : 0 + home.val * 8 ≤ ((coords.val.map U64.leBytes).flatten).length
  · rw [ptr_add_u64, if_pos hh]
    simp only [bind_tc_ok]
    obtain ⟨e, he⟩ := avx2_fail
      (⟨(coords.val.map U64.leBytes).flatten, 0 + home.val * 8⟩ : ConstRawPtr Std.U64) coord (by
        simp only []; rw [hfl]; have : ¬ home.val + 8 ≤ coords.val.length := hlen
        omega)
    exact ⟨e, he⟩
  · rw [ptr_add_u64, if_neg hh]
    exact ⟨.undef, by simp⟩

/-! ## The scalar loop -/

theorem nb_ge : 32 ≤ System.Platform.numBits := by
  rcases System.Platform.numBits_eq with h | h <;> omega

theorem LINEAR_G_val : hashtrinity.LINEAR_G.val = 8 := by
  unfold hashtrinity.LINEAR_G
  rfl

theorem usize_cast_val (x : U32) : (UScalar.cast .Usize x).val = x.val % 2 ^ System.Platform.numBits := by
  rw [UScalar.cast_val_eq]
  rfl

theorem cast_G : (UScalar.cast .Usize hashtrinity.LINEAR_G).val = 8 := by
  rw [usize_cast_val, LINEAR_G_val]
  exact Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le (by norm_num : 8 < 2 ^ 32)
    (Nat.pow_le_pow_right (by norm_num) nb_ge))

theorem uadd_ok {ty} {x y z : UScalar ty} (h : x + y = ok z) :
    z.val = x.val + y.val ∧ x.val + y.val < 2 ^ ty.numBits := by
  have := UScalar.add_equiv x y
  simp only [h, Result.match.ok] at this
  exact ⟨this.2.1, this.1⟩

theorem uadd_ex {ty} (x y : UScalar ty) (h : x.val + y.val < 2 ^ ty.numBits) :
    ∃ z, x + y = ok z ∧ z.val = x.val + y.val := by
  have hmx : UScalar.max ty = 2 ^ ty.numBits - 1 := by
    rw [← UScalar.rMax_eq_max, UScalar.rMax_eq_pow_numBits]
  have hsp := UScalar.add_spec (x := x) (y := y) (by rw [hmx]; omega)
  exact Aeneas.Std.WP.spec_imp_exists hsp

theorem ushl_ok {ty tys} {x : UScalar ty} {s : UScalar tys} {z : UScalar ty}
    (h : x <<< s = ok z) : s.val < ty.numBits ∧ z.val = (x.val * 2 ^ s.val) % 2 ^ ty.numBits := by
  have h' : UScalar.shiftLeft_UScalar x s = ok z := h
  unfold UScalar.shiftLeft_UScalar UScalar.shiftLeft at h'
  by_cases hs : s.val < ty.numBits
  · simp only [hs, if_true] at h'
    have hz : z = ⟨x.bv.shiftLeft s.val⟩ := (Result.ok_injective h').symm
    refine ⟨hs, ?_⟩
    rw [hz]
    show (x.bv.shiftLeft s.val).toNat = (x.bv.toNat * 2 ^ s.val) % 2 ^ ty.numBits
    exact BitVec.toNat_shiftLeft.trans (by rw [Nat.shiftLeft_eq])
  · simp [hs] at h'

theorem ushl_ex {ty tys} (x : UScalar ty) (s : UScalar tys) (h : s.val < ty.numBits) :
    ∃ z, x <<< s = ok z := by
  refine ⟨⟨x.bv.shiftLeft s.val⟩, ?_⟩
  show UScalar.shiftLeft_UScalar x s = _
  unfold UScalar.shiftLeft_UScalar UScalar.shiftLeft
  rw [if_pos h]

theorem slice_index_ok {α : Type} (s : Slice α) (i : Usize) (x : α) (h : s.index_usize i = ok x) :
    s.val[i.val]? = some x := by
  unfold Slice.index_usize at h
  cases hh : s[i]? with
  | none => simp [hh] at h
  | some y =>
    simp [hh] at h
    subst h
    rfl

theorem slice_index_ex {α : Type} (s : Slice α) (i : Usize) (hlt : i.val < s.length) :
    ∃ x, s.index_usize i = ok x := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hlt]
  exact ⟨_, rfl⟩

theorem one_usize : (1#usize).val = 1 := by simp

theorem from_val (b : Bool) : (core.convert.num.FromU32Bool.from b).val = if b then 1 else 0 := by
  cases b <;> rfl

theorem body_inv (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hits empties : Std.U32) (k : Std.Usize)
    (cf : ControlFlow (Std.U32 × Std.U32 × Std.Usize) (Std.U32 × Std.U32))
    (hb : hashtrinity.overflow.prefix_masks_loop.body coords home coord hits empties k = ok cf) :
    (8 ≤ k.val ∧ cf = .done (hits, empties)) ∨
    (k.val < 8 ∧ home.val + k.val < coords.length ∧
      ∃ (h1 e1 : Std.U32) (k1 : Std.Usize) (x : Std.U64), cf = .cont (h1, e1, k1) ∧
        k1.val = k.val + 1 ∧ coords.val[home.val + k.val]? = some x ∧
        h1.val = hits.val ||| (if x = coord then 2 ^ k.val else 0) ∧
        e1.val = empties.val ||| (if x = hashtrinity.EMPTY then 2 ^ k.val else 0)) := by
  unfold hashtrinity.overflow.prefix_masks_loop.body at hb
  obtain ⟨i, hi, hb⟩ := bind_inv hb
  have hi' : i = UScalar.cast .Usize hashtrinity.LINEAR_G := (Result.ok_injective hi).symm
  have hiv : i.val = 8 := by rw [hi']; exact cast_G
  by_cases hk : k < i
  · rw [if_pos hk] at hb
    right
    have hk8 : k.val < 8 := by have := (UScalar.lt_equiv k i).mp hk; omega
    obtain ⟨i1, hi1, hb⟩ := bind_inv hb
    obtain ⟨sc, hsc, hb⟩ := bind_inv hb
    obtain ⟨i2, hi2, hb⟩ := bind_inv hb
    obtain ⟨i3, hi3, hb⟩ := bind_inv hb
    obtain ⟨hits1, hh1, hb⟩ := bind_inv hb
    obtain ⟨i4, hi4, hb⟩ := bind_inv hb
    obtain ⟨i5, hi5, hb⟩ := bind_inv hb
    obtain ⟨empties1, he1, hb⟩ := bind_inv hb
    obtain ⟨k1, hk1, hb⟩ := bind_inv hb
    have hcf : cf = .cont (hits1, empties1, k1) := (Result.ok_injective hb).symm
    obtain ⟨hi1v, _⟩ := uadd_ok hi1
    obtain ⟨hk1v, _⟩ := uadd_ok hk1
    have hscv := slice_index_ok coords i1 sc hsc
    rw [hi1v] at hscv
    have hcl : coords.length = coords.val.length := rfl
    have hlt : home.val + k.val < coords.length := by
      by_contra hh
      rw [List.getElem?_eq_none (by omega)] at hscv
      exact absurd hscv (by simp)
    have hi2' : i2 = core.convert.num.FromU32Bool.from (decide (sc = coord)) :=
      (Result.ok_injective hi2).symm
    have hi4' : i4 = core.convert.num.FromU32Bool.from (decide (sc = hashtrinity.EMPTY)) :=
      (Result.ok_injective hi4).symm
    obtain ⟨hk32, hi3v⟩ := ushl_ok hi3
    obtain ⟨_, hi5v⟩ := ushl_ok hi5
    have e32 : UScalarTy.U32.numBits = 32 := rfl
    rw [e32] at hi3v hi5v
    have hh1' : hits1 = hits ||| i3 := (Result.ok_injective hh1).symm
    have he1' : empties1 = empties ||| i5 := (Result.ok_injective he1).symm
    have hp : 2 ^ k.val < 2 ^ 32 := Nat.pow_lt_pow_right (by norm_num) (by omega)
    have hp' : 2 ^ k.val % 4294967296 = 2 ^ k.val := Nat.mod_eq_of_lt (by simpa using hp)
    refine ⟨hk8, hlt, hits1, empties1, k1, sc, hcf, by rw [hk1v, one_usize], ?_, ?_, ?_⟩
    · exact hscv
    · rw [hh1', or_val, hi3v, hi2', from_val]
      by_cases hs : sc = coord
      · simp [hs, hp']
      · simp [hs]
    · rw [he1', or_val, hi5v, hi4', from_val]
      by_cases hs : sc = hashtrinity.EMPTY
      · simp [hs, hp']
      · simp [hs]
  · rw [if_neg hk] at hb
    left
    have : 8 ≤ k.val := by
      have h := (UScalar.lt_equiv k i)
      by_contra hh
      exact hk (h.mpr (by omega))
    exact ⟨this, (Result.ok_injective hb).symm⟩

theorem ex_bind {α β : Type} {e : Result α} {f : α → Result β} (Q : β → Prop) (a : α)
    (he : e = ok a) (hf : ∃ r, f a = ok r ∧ Q r) :
    ∃ r, (do let x ← e; f x) = ok r ∧ Q r := by
  subst he
  simpa using hf

theorem body_ex (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hits empties : Std.U32) (k : Std.Usize) (hk : k.val < 8) (hlt : home.val + k.val < coords.length) :
    ∃ r, hashtrinity.overflow.prefix_masks_loop.body coords home coord hits empties k = ok r ∧
      ∃ h1 e1 k1, r = .cont (h1, e1, k1) ∧ k1.val = k.val + 1 := by
  unfold hashtrinity.overflow.prefix_masks_loop.body
  have hlt8 : k < UScalar.cast .Usize hashtrinity.LINEAR_G :=
    (UScalar.lt_equiv _ _).mpr (by rw [cast_G]; exact hk)
  refine ex_bind _ (UScalar.cast .Usize hashtrinity.LINEAR_G) rfl ?_
  rw [if_pos hlt8]
  have hsz : home.val + k.val < 2 ^ UScalarTy.Usize.numBits := by
    have := Slice.length_ineq coords
    have h2 : Usize.max < 2 ^ UScalarTy.Usize.numBits := by
      simp [Usize.max, Usize.numBits]
    have hcl : coords.length = coords.val.length := rfl
    omega
  obtain ⟨i1, hi1, hi1v⟩ := uadd_ex home k hsz
  refine ex_bind _ i1 hi1 ?_
  obtain ⟨sc, hsc⟩ := slice_index_ex coords i1 (by rw [hi1v]; exact hlt)
  refine ex_bind _ sc hsc ?_
  refine ex_bind _ (core.convert.num.FromU32Bool.from (decide (sc = coord))) rfl ?_
  obtain ⟨i3, hi3⟩ := ushl_ex (core.convert.num.FromU32Bool.from (decide (sc = coord))) k
    (by have e32 : UScalarTy.U32.numBits = 32 := rfl
        rw [e32]; omega)
  refine ex_bind _ i3 hi3 ?_
  refine ex_bind _ (hits ||| i3) rfl ?_
  refine ex_bind _ (core.convert.num.FromU32Bool.from (decide (sc = hashtrinity.EMPTY))) rfl ?_
  obtain ⟨i5, hi5⟩ := ushl_ex (core.convert.num.FromU32Bool.from (decide (sc = hashtrinity.EMPTY))) k
    (by have e32 : UScalarTy.U32.numBits = 32 := rfl
        rw [e32]; omega)
  refine ex_bind _ i5 hi5 ?_
  refine ex_bind _ (empties ||| i5) rfl ?_
  have hsz1 : k.val + (1#usize).val < 2 ^ UScalarTy.Usize.numBits := by
    rw [one_usize]
    have h2 : 32 ≤ UScalarTy.Usize.numBits := nb_ge
    have : 2 ^ 32 ≤ 2 ^ UScalarTy.Usize.numBits := Nat.pow_le_pow_right (by norm_num) h2
    omega
  obtain ⟨k1, hk1, hk1v⟩ := uadd_ex k 1#usize hsz1
  refine ex_bind _ k1 hk1 ?_
  exact ⟨_, rfl, _, _, _, rfl, by rw [hk1v, one_usize]⟩

theorem body_done (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hits empties : Std.U32) (k : Std.Usize) (hk : 8 ≤ k.val) :
    hashtrinity.overflow.prefix_masks_loop.body coords home coord hits empties k =
      ok (.done (hits, empties)) := by
  unfold hashtrinity.overflow.prefix_masks_loop.body
  have hlt8 : ¬ k < UScalar.cast .Usize hashtrinity.LINEAR_G := by
    intro h
    have := (UScalar.lt_equiv _ _).mp h
    rw [cast_G] at this
    omega
  simp only [lift, bind_tc_ok]
  rw [if_neg hlt8]

/-- The extracted loop's step function. -/
def stepF (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64) :
    Std.U32 × Std.U32 × Std.Usize → Result (ControlFlow (Std.U32 × Std.U32 × Std.Usize) (Std.U32 × Std.U32)) :=
  fun (hits1, empties1, k1) =>
    hashtrinity.overflow.prefix_masks_loop.body coords home coord hits1 empties1 k1

theorem loop_unfold (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hits empties : Std.U32) (k : Std.Usize) :
    hashtrinity.overflow.prefix_masks_loop coords home coord hits empties k =
      loop (stepF coords home coord) (hits, empties, k) := by
  unfold hashtrinity.overflow.prefix_masks_loop stepF
  rfl

theorem loop_total (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hlen : home.val + 8 ≤ coords.length) :
    ∀ m (hits empties : Std.U32) (k : Std.Usize), k.val + m = 8 →
      ∃ r, loop (stepF coords home coord) (hits, empties, k) = ok r := by
  intro m
  induction m with
  | zero =>
    intro hits empties k hk
    rw [loop.eq_def]
    simp only [stepF]
    rw [body_done coords home coord hits empties k (by omega)]
    exact ⟨(hits, empties), by simp⟩
  | succ m ih =>
    intro hits empties k hk
    rw [loop.eq_def]
    simp only [stepF]
    obtain ⟨r, hr, h1, e1, k1, hr', hk1⟩ := body_ex coords home coord hits empties k (by omega) (by omega)
    rw [hr, hr']
    simp only [bind_ok]
    exact ih h1 e1 k1 (by omega)

/-- The loop invariant at state `(hits, empties, k)`. -/
def Inv (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (st : Std.U32 × Std.U32 × Std.Usize) : Prop :=
  st.2.2.val ≤ 8 ∧ (0 < st.2.2.val → home.val + st.2.2.val ≤ coords.length) ∧
  st.1.val < 2 ^ st.2.2.val ∧ st.2.1.val < 2 ^ st.2.2.val ∧
  ∀ j, j < st.2.2.val → ∃ x, coords.val[home.val + j]? = some x ∧
    st.1.val.testBit j = decide (x = coord) ∧ st.2.1.val.testBit j = decide (x = hashtrinity.EMPTY)

theorem or_pow_bit (h : Nat) (k j : Nat) (P : Prop) [Decidable P] (hh : h < 2 ^ k) (hj : j < k + 1) :
    (h ||| (if P then 2 ^ k else 0)).testBit j = (if j < k then h.testBit j else decide P) := by
  rw [Nat.testBit_or]
  by_cases hjk : j < k
  · by_cases hP : P
    · simp [hP, hjk, Nat.testBit_two_pow]
      omega
    · simp [hP, hjk]
  · have hjk' : j = k := by omega
    subst hjk'
    rw [Nat.testBit_lt_two_pow hh]
    by_cases hP : P <;> simp [hP]

theorem or_pow_lt (h : Nat) (k : Nat) (P : Prop) [Decidable P] (hh : h < 2 ^ k) :
    (h ||| (if P then 2 ^ k else 0)) < 2 ^ (k + 1) := by
  have h2 : 2 ^ k < 2 ^ (k + 1) := Nat.pow_lt_pow_right (by norm_num) (by omega)
  apply Nat.or_lt_two_pow
  · omega
  · by_cases hP : P
    · simp [hP]; exact h2
    · simp [hP]

theorem scalar_pc (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (H E : Std.U32) (h : scalarBranch coords home coord = ok (H, E)) :
    home.val + 8 ≤ coords.length ∧ Spec coords home coord (H, E) := by
  unfold scalarBranch at h
  rw [loop_unfold] at h
  have key := loop_pc (Inv coords home coord)
    (fun r : Std.U32 × Std.U32 => home.val + 8 ≤ coords.length ∧ Spec coords home coord r)
    (stepF coords home coord) ?_ (0#u32, 0#u32, 0#usize) ?_ (H, E) h
  · exact key
  · rintro ⟨hh, ee, kk⟩ hinv cf hb
    obtain ⟨i1, i2, i3, i4, i5⟩ := hinv
    simp only [stepF] at hb
    rcases body_inv coords home coord hh ee kk cf hb with ⟨hk8, rfl⟩ | ⟨hk8, hlt, h1, e1, k1, x, rfl, hk1, hx, hh1, he1⟩
    · simp only at i1 i2 i3 i4 i5 ⊢
      have hk : kk.val = 8 := by omega
      rw [hk] at i3 i4 i5
      refine ⟨i2 (by omega) |> fun h' => by rw [hk] at h'; exact h', i3, i4, fun j hj => ?_⟩
      · exact i5 j (by omega)
    · simp only at i1 i2 i3 i4 i5 ⊢
      have hcl : coords.length = coords.val.length := rfl
      refine ⟨show k1.val ≤ 8 by omega, fun _ => show home.val + k1.val ≤ coords.length by omega,
        show h1.val < 2 ^ k1.val by rw [hh1, hk1]; exact or_pow_lt _ _ _ i3,
        show e1.val < 2 ^ k1.val by rw [he1, hk1]; exact or_pow_lt _ _ _ i4, fun j hj => ?_⟩
      have hj' : j < kk.val + 1 := by
        have : (h1, e1, k1).2.2.val = k1.val := rfl
        omega
      show ∃ x', coords.val[home.val + j]? = some x' ∧ h1.val.testBit j = decide (x' = coord) ∧
        e1.val.testBit j = decide (x' = hashtrinity.EMPTY)
      by_cases hjk : j < kk.val
      · obtain ⟨y, hy, hy1, hy2⟩ := i5 j hjk
        refine ⟨y, hy, ?_, ?_⟩
        · rw [hh1, or_pow_bit _ _ _ _ i3 hj', if_pos hjk, hy1]
        · rw [he1, or_pow_bit _ _ _ _ i4 hj', if_pos hjk, hy2]
      · have hjk' : j = kk.val := by omega
        subst hjk'
        refine ⟨x, hx, ?_, ?_⟩
        · rw [hh1, or_pow_bit _ _ _ _ i3 hj', if_neg hjk]
        · rw [he1, or_pow_bit _ _ _ _ i4 hj', if_neg hjk]
  · refine ⟨by simp, fun h' => by simp at h', by simp, by simp, fun j hj => by simp at hj⟩

theorem scalar_ex (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hlen : home.val + 8 ≤ coords.length) :
    ∃ r, scalarBranch coords home coord = ok r ∧ Spec coords home coord r := by
  obtain ⟨r, hr⟩ := loop_total coords home coord hlen 8 0#u32 0#u32 0#usize (by simp)
  refine ⟨r, ?_, ?_⟩
  · unfold scalarBranch; rw [loop_unfold]; exact hr
  · have := scalar_pc coords home coord r.1 r.2 (by
      unfold scalarBranch; rw [loop_unfold]; exact hr)
    exact this.2

/-! ## Agreement -/

theorem eq_of_bits (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h : ∀ j, j < 8 → a.testBit j = b.testBit j) : a = b := by
  apply Nat.eq_of_testBit_eq
  intro j
  by_cases hj : j < 8
  · exact h j hj
  · have h8 : (256 : Nat) ≤ 2 ^ j := by
      have : 2 ^ 8 ≤ 2 ^ j := Nat.pow_le_pow_right (by norm_num) (by omega)
      simpa using this
    rw [Nat.testBit_lt_two_pow (by omega : a < 2 ^ j), Nat.testBit_lt_two_pow (by omega : b < 2 ^ j)]

theorem spec_unique (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (r r' : Std.U32 × Std.U32) (h : Spec coords home coord r) (h' : Spec coords home coord r') :
    r = r' := by
  obtain ⟨a1, a2, a3⟩ := h
  obtain ⟨b1, b2, b3⟩ := h'
  have e1 : r.1.val = r'.1.val := eq_of_bits _ _ a1 b1 (fun j hj => by
    obtain ⟨x, hx, p, _⟩ := a3 j hj
    obtain ⟨y, hy, q, _⟩ := b3 j hj
    have : x = y := Option.some.inj (hx.symm.trans hy)
    rw [p, q, this])
  have e2 : r.2.val = r'.2.val := eq_of_bits _ _ a2 b2 (fun j hj => by
    obtain ⟨x, hx, _, p⟩ := a3 j hj
    obtain ⟨y, hy, _, q⟩ := b3 j hj
    have : x = y := Option.some.inj (hx.symm.trans hy)
    rw [p, q, this])
  exact Prod.ext (UScalar.eq_of_val_eq e1) (UScalar.eq_of_val_eq e2)

/-- On every input where the scalar branch returns `ok`, the AVX2 branch returns the same pair; and
where the window is out of range neither branch returns `ok`. -/
theorem prefix_masks_branches_agree (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64) :
    (home.val + 8 ≤ coords.length →
      ∃ r, scalarBranch coords home coord = ok r ∧ avxBranch coords home coord = ok r ∧
        Spec coords home coord r) ∧
    (¬ home.val + 8 ≤ coords.length →
      (∃ e, avxBranch coords home coord = fail e) ∧ ∀ r, scalarBranch coords home coord ≠ ok r) := by
  refine ⟨fun hlen => ?_, fun hlen => ⟨avxBranch_fail coords home coord hlen, fun r hr => ?_⟩⟩
  · obtain ⟨r, hr, hs⟩ := scalar_ex coords home coord hlen
    obtain ⟨r', hr', hs'⟩ := avxBranch_spec coords home coord hlen
    have := spec_unique coords home coord r r' hs hs'
    subst this
    exact ⟨r, hr, hr', hs⟩
  · exact hlen (scalar_pc coords home coord r.1 r.2 hr).1

/-- The result of the extracted `prefix_masks` does not depend on `avx2_probe`. -/
theorem prefix_masks_probe_independent (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hlen : home.val + 8 ≤ coords.length) :
    ∃ r, hashtrinity.overflow.prefix_masks coords home coord = ok r ∧
      avxBranch coords home coord = ok r ∧ scalarBranch coords home coord = ok r := by
  obtain ⟨r, h1, h2, _⟩ := (prefix_masks_branches_agree coords home coord).1 hlen
  refine ⟨r, ?_, h2, h1⟩
  rw [prefix_masks_eq]
  cases std_detect.detect.arch.x86.avx2_probe <;> simp [h1, h2]

/-- Bit `j < 8` of the returned hit (empty) mask is set exactly when `coords[home + j]` equals
`coord` (`EMPTY`). -/
theorem prefix_masks_spec (coords : Slice Std.U64) (home : Std.Usize) (coord : Std.U64)
    (hlen : home.val + 8 ≤ coords.length) :
    ∃ H E, hashtrinity.overflow.prefix_masks coords home coord = ok (H, E) ∧
      H.val < 256 ∧ E.val < 256 ∧
      ∀ j, j < 8 → ∃ x, coords.val[home.val + j]? = some x ∧
        (H.val.testBit j = true ↔ x = coord) ∧ (E.val.testBit j = true ↔ x = hashtrinity.EMPTY) := by
  obtain ⟨r, hsc, hav, hs1, hs2, hs3⟩ := (prefix_masks_branches_agree coords home coord).1 hlen
  refine ⟨r.1, r.2, ?_, hs1, hs2, fun j hj => ?_⟩
  · rw [prefix_masks_eq]
    cases std_detect.detect.arch.x86.avx2_probe <;> simp [hsc, hav]
  · obtain ⟨x, hx, b1, b2⟩ := hs3 j hj
    exact ⟨x, hx, by rw [b1]; exact decide_eq_true_iff, by rw [b2]; exact decide_eq_true_iff⟩

end Apeiron.PrefixMasks
