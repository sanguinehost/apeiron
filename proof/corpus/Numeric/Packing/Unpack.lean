/-
Refinement of the extracted canonical packing decoder's bit-plane code reader
(`APEIRON-ALGEBRA-PACKING-W1-T1`).

The Rust body is `src/format/packing.rs:571-591` (`Packing::unpack_code`), extracted at
`proof/ApeironWord.lean:9821-9888` as `packing.Packing.unpack_code_loop` plus
`packing.Packing.unpack_code`. This file supplies the *bit-plane reference law* that
`Numeric/Packing/Refinement.lean` names as residual item 1: the generic `unpack_code` reads
`total` code bits of weight `i` from the little-endian byte stream `area` and places bit `k` at
plane shift `k` (`LsbFirst`) or `total-1-k` (`MsbFirst`), then two's-complement sign-extends
when `signed_code` and `total < 64`.

Proved here:

* `unpack_code_loop_body_step`: one loop iteration ORs code bit `k` into the accumulator at its
  plane shift and advances `k`, leaving the descriptor unchanged (kernel-checked, cone-clean);
* `unpack_code_loop_spec`: the loop returns the unchanged descriptor and an accumulator whose bit
  pattern is the bit-plane sum `codeBits` of all `total` code bits;
* `unpack_code_ref`: the top-level law — on the declared domain `packing.Packing.unpack_code self
  area i total = ok (unpackRef self area i total)`, the bit-plane sum `codeSum` reached by the loop
  and two's-complement sign-extended over its low `total` bits exactly when `signed_code` is set,
  `total < 64` and the sign bit is set.

## Domain and the excluded sign-extension face

`unpack_code` `<<<`-shifts an `i64` by up to `total-1`, so the loop requires `total ≤ 64`. The
sign-extension branch subtracts `1i64 << total` from a value whose bit pattern holds only the low
`total` bits. For `total ≤ 62` that checked subtraction is in range. For `total = 63` the literal
`1i64 << 63` *is* `i64::MIN`, so the extracted `i64` subtraction leaves range and `unpack_code`
fails with `integerOverflow`. `unpack_code_ref` therefore carries the two edges as hypotheses
(`hsigned`): a signed code needs `0 < total.val` (otherwise `total - 1#usize` underflows) and, when
`total.val < 64`, `total.val ≤ 62` (the `total = 63` sign-extension subtraction overflow face is
excluded, not absorbed). Total `total = 63` with the sign bit clear and unsigned codes skip the
subtraction and are covered.

## Complexity

`unpack_code_loop` runs `total` iterations of `O(1)` byte load, shift and OR; `unpack_code` adds
`O(1)`. Time and space `O(total)`, counted in loop iterations and `i64` words; `total ≤ 64` is the
declared domain.
-/
import Numeric.Packing.Refinement

namespace apeiron_word

open Aeneas Aeneas.Std Result WP
open scoped BigOperators

/-! ## The bit-plane reference primitives -/

/-- The bit-plane shift of code bit `k` under the plane order. -/
def planeShift (po : packing.PlaneOrder) (total k : Nat) : Nat :=
  match po with
  | .LsbFirst => k
  | .MsbFirst => total - 1 - k

/-- Bit `pos` of the little-endian bit stream `area` (LSB of byte 0 first). -/
def areaBit (area : List U8) (pos : Nat) : Nat :=
  ((area.getD (pos / 8) 0#u8).val / 2 ^ (pos % 8)) % 2

theorem self_with_po (self : packing.Packing) :
    { self with plane_order := self.plane_order } = self := by cases self; rfl

/-- **`unpack_code` loop-body step.** One iteration ORs code bit `k` (stream position
`i·total + k`) into `v` at its plane shift and advances `k`; the descriptor is unchanged. -/
theorem unpack_code_loop_body_step (area : Slice U8) (i total : Usize)
    (self : packing.Packing) (v : I64) (k : Usize)
    (hk : k.val < total.val) (htot : total.val ≤ 64)
    (hmul : i.val * total.val + total.val ≤ Usize.max)
    (harea : (i.val * total.val + k.val) / 8 < area.val.length) :
    packing.Packing.unpack_code_loop.body area i total self v k ⦃ cf =>
      ∃ (v1 : I64) (k1 : Usize), cf = ControlFlow.cont (self, v1, k1) ∧
        k1.val = k.val + 1 ∧
        v1.bv = v.bv ||| (BitVec.ofNat 64 (areaBit area.val (i.val * total.val + k.val))
          <<< planeShift self.plane_order total.val k.val) ⦄ := by
  unfold packing.Packing.unpack_code_loop.body
  have hlt : k < total := by scalar_tac
  simp only [hlt, if_true]
  cases hpo : self.plane_order <;> simp only [planeShift] <;> step* <;>
  · refine ⟨v1, k1, ?_, k1_post, ?_⟩
    · rw [← hpo]
    · refine v1_post1.trans ?_
      congr 1
      refine i7_post1.trans ?_
      congr 1
      · rw [bit_post]
        apply BitVec.eq_of_toNat_eq
        have ei2 : i2.val = (i.val * total.val + k.val) / 8 := by scalar_tac
        have ei4 : i4.val = (i.val * total.val + k.val) % 8 := by scalar_tac
        have hi3 : i3.val = (area.val.getD ((i.val * total.val + k.val) / 8) 0#u8).val := by
          rw [List.getD_eq_getElem _ _ harea]
          subst i3_post
          simp only [ei2]
        have e1 : i6.bv.toNat = i5.bv.toNat % 2 := by
          have h := congrArg BitVec.toNat i6_post1
          simpa [BitVec.toNat_and, Nat.and_one_is_mod] using h
        have e2 : i5.bv.toNat = i3.bv.toNat / 2 ^ i4.val := by
          have h := congrArg BitVec.toNat i5_post1
          simpa [BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow] using h
        have e3 : i3.bv.toNat = i3.val := UScalar.bv_toNat i3
        simp only [UScalar.hcast, BitVec.truncate_eq_setWidth, BitVec.toNat_ofNat, areaBit, ← hi3]
        simp [BitVec.toNat_setWidth, e1, e2, e3, ei4]
      all_goals scalar_tac

/-! ## The accumulating bit-plane sum -/

/-- The low `k` code bits of weight `i`, placed at their plane shifts and summed as a `Nat`. -/
def codeBits (po : packing.PlaneOrder) (area : Slice U8) (i total k : Nat) : Nat :=
  ∑ k' ∈ Finset.range k, areaBit area.val (i * total + k') * 2 ^ planeShift po total k'

/-- The full code bit-plane sum of weight `i`. -/
def codeSum (self : packing.Packing) (area : Slice U8) (i total : Usize) : Nat :=
  codeBits self.plane_order area i.val total.val total.val

lemma codeBits_succ (po : packing.PlaneOrder) (area : Slice U8) (i total k : Nat) :
    codeBits po area i total (k + 1) = codeBits po area i total k
      + areaBit area.val (i * total + k) * 2 ^ planeShift po total k := by
  unfold codeBits
  rw [Finset.sum_range_succ]

/-- A stream bit is `0` or `1`. -/
lemma areaBit_le_one (area : List U8) (pos : Nat) : areaBit area pos ≤ 1 := by
  unfold areaBit
  have := Nat.mod_lt ((area.getD (pos / 8) 0#u8).val / 2 ^ (pos % 8)) (by norm_num : 0 < 2)
  omega

lemma planeShift_lt {po : packing.PlaneOrder} {total k : Nat} (hk : k < total)
    (htot : total ≤ 64) : planeShift po total k < 64 := by
  cases po <;> simp only [planeShift] <;> omega

/-- The bit pattern of `areaBit · 2^s` as an `I64`, via the extracted `ofNat 64` then `<<< s`. -/
lemma toNat_ofNat_shiftLeft {b s : Nat} (hb : b ≤ 1) (hs : s < 64) :
    ((BitVec.ofNat 64 b) <<< s).toNat = b * 2 ^ s := by
  rw [BitVec.toNat_shiftLeft]
  have hb64 : (BitVec.ofNat 64 b).toNat = b := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  rw [hb64, Nat.shiftLeft_eq, Nat.mod_eq_of_lt]
  calc b * 2 ^ s ≤ 1 * 2 ^ s := Nat.mul_le_mul_right _ hb
    _ = 2 ^ s := one_mul _
    _ < 2 ^ 64 := Nat.pow_lt_pow_right (by norm_num) hs

/-- OR with a fresh high bit `w·2^k` is addition when `a < 2^k`. -/
lemma nat_or_eq_add_of_lt {a k w : Nat} (ha : a < 2 ^ k) :
    a ||| (w * 2 ^ k) = a + w * 2 ^ k := by
  have h := Nat.two_pow_add_eq_or_of_lt ha w
  rw [Nat.mul_comm (2 ^ k) w, Nat.or_comm, Nat.add_comm] at h
  exact h.symm

/-- OR with a fresh low bit `w·2^s` (`w ≤ 1`) is addition when `2^(s+1)` divides `a`. -/
lemma nat_or_eq_add_of_dvd {a s w : Nat} (h : 2 ^ (s + 1) ∣ a) (hw : w ≤ 1) :
    a ||| (w * 2 ^ s) = a + w * 2 ^ s := by
  obtain ⟨c, rfl⟩ := h
  have hlt : w * 2 ^ s < 2 ^ (s + 1) := by
    calc w * 2 ^ s ≤ 1 * 2 ^ s := Nat.mul_le_mul_right _ hw
      _ = 2 ^ s := one_mul _
      _ < 2 ^ (s + 1) := by rw [Nat.pow_succ]; have := Nat.two_pow_pos s; omega
  exact (Nat.two_pow_add_eq_or_of_lt (i := s + 1) (b := w * 2 ^ s) hlt c).symm

/-! ## The loop invariant and its step -/

/-- Loop invariant for `unpack_code_loop`, parameterized by the (never-changed) descriptor
`self0`: the descriptor is preserved, `k` is bounded, the accumulator's bit pattern is the
bit-plane sum of the first `k` code bits, and it has no bits outside the planes already visited
(below `k` for `LsbFirst`, a multiple of `2^(total-k)` for `MsbFirst`). -/
def unpackLoopInv (self0 : packing.Packing) (area : Slice U8) (i total : Usize)
    (self : packing.Packing) (v : I64) (k : Usize) : Prop :=
  self = self0 ∧ k.val ≤ total.val ∧
  v.bv.toNat = codeBits self0.plane_order area i.val total.val k.val ∧
  (self0.plane_order = packing.PlaneOrder.LsbFirst → v.bv.toNat < 2 ^ k.val) ∧
  (self0.plane_order = packing.PlaneOrder.MsbFirst →
      2 ^ (total.val - k.val) ∣ v.bv.toNat)

/-- One invariant-preserving iteration of the extracted loop: the accumulator gains exactly the
next code bit at its plane position, the descriptor is unchanged, and `k` advances. -/
theorem unpack_code_loop_step (self0 self : packing.Packing) (area : Slice U8)
    (i total : Usize) (v : I64) (k : Usize)
    (hinv : unpackLoopInv self0 area i total self v k)
    (hk : k.val < total.val) (htot : total.val ≤ 64)
    (hmul : i.val * total.val + total.val ≤ Usize.max)
    (harea : (i.val * total.val + k.val) / 8 < area.val.length) :
    WP.spec (packing.Packing.unpack_code_loop.body area i total self v k) (fun cf =>
      ∃ (v1 : I64) (k1 : Usize), cf = ControlFlow.cont (self, v1, k1) ∧
        unpackLoopInv self0 area i total self v1 k1 ∧ k1.val = k.val + 1) := by
  obtain ⟨hself, hkle, hsum, hlsb, hmsb⟩ := hinv
  obtain ⟨cf, hcf, v1, k1, hcfv, hk1, hv1⟩ :=
    WP.spec_imp_exists (unpack_code_loop_body_step area i total self v k hk htot hmul harea)
  rw [hcf, WP.spec_ok]
  rw [hself] at hv1 hcfv ⊢
  have hb : areaBit area.val (i.val * total.val + k.val) ≤ 1 := areaBit_le_one _ _
  refine ⟨v1, k1, hcfv, ?_, hk1⟩
  cases hpo : self0.plane_order
  · -- LsbFirst: the new bit is above all accumulated bits
    rw [hpo] at hv1 hsum hlsb hmsb
    simp only [planeShift] at hv1
    have hlsb' : v.bv.toNat < 2 ^ k.val := hlsb rfl
    have hv1n : v1.bv.toNat = v.bv.toNat
        ||| (areaBit area.val (i.val * total.val + k.val) * 2 ^ k.val) := by
      rw [hv1, BitVec.toNat_or,
        toNat_ofNat_shiftLeft (b := areaBit area.val (i.val * total.val + k.val))
          (s := k.val) hb (by omega)]
    have hadd := nat_or_eq_add_of_lt (w := areaBit area.val (i.val * total.val + k.val)) hlsb'
    unfold unpackLoopInv
    rw [hpo]
    refine ⟨rfl, by omega, ?_, ?_, ?_⟩
    · rw [hv1n, hadd, hsum, hk1, codeBits_succ]
      simp only [planeShift]
    · intro _
      rw [hv1n, hadd, hk1]
      have hbm : areaBit area.val (i.val * total.val + k.val) * 2 ^ k.val ≤ 2 ^ k.val := by
        simpa using Nat.mul_le_mul_right (2 ^ k.val) hb
      have hp : (2 : Nat) ^ (k.val + 1) = 2 ^ k.val + 2 ^ k.val := by
        rw [Nat.pow_succ]; ring
      omega
    · intro hc
      cases hc
  · -- MsbFirst: the new bit is below all accumulated bits
    rw [hpo] at hv1 hsum hlsb hmsb
    simp only [planeShift] at hv1
    have hmsb' : 2 ^ (total.val - k.val) ∣ v.bv.toNat := hmsb rfl
    have hv1n : v1.bv.toNat = v.bv.toNat
        ||| (areaBit area.val (i.val * total.val + k.val)
          * 2 ^ (total.val - 1 - k.val)) := by
      rw [hv1, BitVec.toNat_or,
        toNat_ofNat_shiftLeft (b := areaBit area.val (i.val * total.val + k.val))
          (s := total.val - 1 - k.val) hb (by omega)]
    have hdiv : 2 ^ (total.val - 1 - k.val + 1) ∣ v.bv.toNat := by
      have he : total.val - 1 - k.val + 1 = total.val - k.val := by omega
      rw [he]; exact hmsb'
    have hadd := nat_or_eq_add_of_dvd hdiv hb
    unfold unpackLoopInv
    rw [hpo]
    refine ⟨rfl, by omega, ?_, ?_, ?_⟩
    · rw [hv1n, hadd, hsum, hk1, codeBits_succ]
      simp only [planeShift]
    · intro hc
      cases hc
    · -- 2^(total-k) ∣ v.toNat stretches to 2^(total-(k+1)) ∣ v1.toNat
      intro _
      have hle : total.val - (k.val + 1) ≤ total.val - k.val := by omega
      have hdv : 2 ^ (total.val - (k.val + 1)) ∣ v.bv.toNat :=
        (Nat.pow_dvd_pow 2 hle).trans hmsb'
      have hdb : 2 ^ (total.val - (k.val + 1))
          ∣ areaBit area.val (i.val * total.val + k.val) * 2 ^ (total.val - 1 - k.val) := by
        have he : total.val - 1 - k.val = total.val - (k.val + 1) := by omega
        rw [he]
        exact dvd_mul_left _ _
      rw [hv1n, hadd, hk1]
      exact Nat.dvd_add hdv hdb

/-! ## The loop assembly -/

/-- The `unpack_code_loop` on the initial accumulator `0`: it returns the unchanged descriptor and
an accumulator whose bit pattern is the full bit-plane sum of weight `i`. -/
theorem unpack_code_loop_spec (self : packing.Packing) (area : Slice U8) (i total : Usize)
    (htot : total.val ≤ 64)
    (hmul : i.val * total.val + total.val ≤ Usize.max)
    (harea : (i.val * total.val + total.val - 1) / 8 < area.val.length) :
    WP.spec (packing.Packing.unpack_code_loop self area i total 0#i64 0#usize)
      (fun b => ∃ v', b = (self, v') ∧ v'.bv.toNat = codeSum self area i total) := by
  have hinv0 : unpackLoopInv self area i total self 0#i64 0#usize := by
    refine ⟨rfl, by simp, ?_, ?_, ?_⟩
    · simp [codeBits]
    · intro _; simp
    · intro _; simp
  unfold packing.Packing.unpack_code_loop
  apply loop.spec_decr_nat
    (measure := fun s : packing.Packing × Std.I64 × Std.Usize => total.val - s.2.2.val)
    (inv := fun s => unpackLoopInv self area i total s.1 s.2.1 s.2.2)
    (post := fun b : packing.Packing × Std.I64 =>
      ∃ v', b = (self, v') ∧ v'.bv.toNat = codeSum self area i total)
  · intro x hx
    rcases x with ⟨self', v, k⟩
    obtain ⟨hself, hkle, hsum, hlsb, hmsb⟩ := hx
    have hkle' : k.val ≤ total.val := hkle
    have hsum' : v.bv.toNat = codeBits self.plane_order area i.val total.val k.val := hsum
    have hlsb' : self.plane_order = packing.PlaneOrder.LsbFirst → v.bv.toNat < 2 ^ k.val := hlsb
    have hmsb' : self.plane_order = packing.PlaneOrder.MsbFirst →
        2 ^ (total.val - k.val) ∣ v.bv.toNat := hmsb
    by_cases hk : k.val < total.val
    · have harea' : (i.val * total.val + k.val) / 8 < area.val.length := by
        have hle : i.val * total.val + k.val ≤ i.val * total.val + total.val - 1 := by omega
        exact lt_of_le_of_lt (Nat.div_le_div_right hle) harea
      obtain ⟨cf, hcf, v1, k1, hcfv, hinv1, hk1⟩ :=
        WP.spec_imp_exists (unpack_code_loop_step self self' area i total v k
          ⟨hself, hkle', hsum', hlsb', hmsb'⟩ hk htot hmul harea')
      rw [hcfv] at hcf
      simp only [hcf]
      refine ⟨hinv1, ?_⟩
      show total.val - k1.val < total.val - k.val
      rw [hk1]
      omega
    · have hnot : ¬ (k < total) := by rw [UScalar.lt_equiv]; exact hk
      have hkeq : k.val = total.val := le_antisymm hkle' (not_lt.mp hk)
      have hbody : packing.Packing.unpack_code_loop.body area i total self' v k
          = ok (ControlFlow.done (self', v)) := by
        unfold packing.Packing.unpack_code_loop.body
        rw [if_neg hnot]
      simp only [hbody]
      refine ⟨v, ?_, ?_⟩
      · exact Prod.ext hself rfl
      · show v.bv.toNat = codeBits self.plane_order area i.val total.val total.val
        rw [show codeBits self.plane_order area i.val total.val total.val
              = codeBits self.plane_order area i.val total.val k.val from by rw [hkeq]]
        exact hsum'
  · exact hinv0

/-! ## The reference value

`unpackRef` is the top-level reference: the bit-plane sum `codeSum` of weight `i`,
two's-complement sign-extended over its low `total` bits when `signed_code`, `total < 64`, and the
sign bit is set. `unpack_code_ref` below proves `unpack_code self area i total = ok (unpackRef self
area i total)` on the declared domain. -/

/-- The reference code value of weight `i`: the bit-plane sum `codeSum`, two's-complement
sign-extended over `total` bits when `signed_code`, `total < 64`, and the sign bit is set. -/
def unpackRef (self : packing.Packing) (area : Slice U8) (i total : Usize) : I64 :=
  if self.signed_code = true ∧ total.val < 64
      ∧ (codeSum self area i total).testBit (total.val - 1) = true
  then ⟨BitVec.ofInt 64 ((codeSum self area i total : Int) - 2 ^ total.val)⟩
  else ⟨BitVec.ofNat 64 (codeSum self area i total)⟩

lemma I64.eq_of_toNat_eq {x y : I64} (h : x.bv.toNat = y.bv.toNat) : x = y :=
  congrArg (fun bv => (IScalar.mk bv : I64)) (BitVec.eq_of_toNat_eq h)

lemma I64.eq_of_toInt_eq {x y : I64} (h : x.bv.toInt = y.bv.toInt) : x = y :=
  congrArg (fun bv => (IScalar.mk bv : I64)) (BitVec.toInt_inj.mp h)

/-! ## The top-level law

`codeSum_lt` bounds the plane sum below `2^total` (the shifts are the distinct positions
`0..total-1`), so the loop accumulator is exactly `⟨BitVec.ofNat 64 codeSum⟩`; `signbit_iff` turns
the extracted `(v &&& sign_bit) != 0` into the reference test bit. `unpack_code_ref` then cases the
`self.signed_code` / `total < 64` / sign-bit branches, evaluates each `do` operation through its
`spec` (`UScalar.sub_spec`, `IScalar.ShiftLeft_spec`, `IScalar.sub_spec`), and matches the result to
`unpackRef`. -/

lemma sum_range_two_pow (n : Nat) : (∑ k ∈ Finset.range n, 2 ^ k : Nat) = 2 ^ n - 1 := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ, ih, Nat.pow_succ]
      have h : 0 < 2 ^ n := Nat.two_pow_pos n
      omega

/-- The bit-plane sum of weight `i` fits in `total` bits. -/
lemma codeSum_lt (self : packing.Packing) (area : Slice U8) (i total : Usize) :
    codeSum self area i total < 2 ^ total.val := by
  unfold codeSum codeBits
  cases hpo : self.plane_order
  · simp only [planeShift]
    calc (∑ k' ∈ Finset.range total.val,
            areaBit area.val (i.val * total.val + k') * 2 ^ k')
        ≤ ∑ k' ∈ Finset.range total.val, 2 ^ k' := by
          apply Finset.sum_le_sum
          intro k' _
          calc areaBit area.val (i.val * total.val + k') * 2 ^ k'
              ≤ 1 * 2 ^ k' := Nat.mul_le_mul_right _ (areaBit_le_one _ _)
            _ = 2 ^ k' := one_mul _
      _ = 2 ^ total.val - 1 := sum_range_two_pow total.val
      _ < 2 ^ total.val := by have := Nat.two_pow_pos total.val; omega
  · simp only [planeShift]
    calc (∑ k' ∈ Finset.range total.val,
            areaBit area.val (i.val * total.val + k') * 2 ^ (total.val - 1 - k'))
        ≤ ∑ k' ∈ Finset.range total.val, 2 ^ (total.val - 1 - k') := by
          apply Finset.sum_le_sum
          intro k' _
          calc areaBit area.val (i.val * total.val + k') * 2 ^ (total.val - 1 - k')
              ≤ 1 * 2 ^ (total.val - 1 - k') := Nat.mul_le_mul_right _ (areaBit_le_one _ _)
            _ = 2 ^ (total.val - 1 - k') := one_mul _
      _ = ∑ k' ∈ Finset.range total.val, 2 ^ k' :=
            Finset.sum_range_reflect (fun j => 2 ^ j) total.val
      _ = 2 ^ total.val - 1 := sum_range_two_pow total.val
      _ < 2 ^ total.val := by have := Nat.two_pow_pos total.val; omega

/-- The extracted `(v &&& sign_bit) != 0` is the reference sign-bit test `c.testBit k`. -/
lemma signbit_iff (v' sb : I64) (c k : Nat) (hv' : v'.bv.toNat = c)
    (hsb : sb.bv.toNat = 2 ^ k) :
    ((v' &&& sb) != 0#i64) = true ↔ c.testBit k = true := by
  have htn : (v' &&& sb).bv.toNat = c &&& 2 ^ k := by
    show (v'.bv &&& sb.bv).toNat = c &&& 2 ^ k
    rw [BitVec.toNat_and, hv', hsb]
  have hne : ((v' &&& sb) != 0#i64) = true ↔ (v' &&& sb).bv.toNat ≠ 0 := by
    rw [bne_iff_ne]
    constructor
    · intro h h0
      exact h (I64.eq_of_toNat_eq (by rw [h0]; rfl))
    · intro h hz
      exact h (by rw [congrArg (fun x : I64 => x.bv.toNat) hz]; rfl)
  rw [hne, htn, Nat.and_two_pow]
  cases hck : c.testBit k <;> simp

/-- **`unpack_code` top-level law.** On the declared domain the extracted decoder returns exactly
the reference value: the bit-plane sum, two's-complement sign-extended over its low `total` bits
when `signed_code`, `total < 64` and the sign bit is set. `hsigned` excludes the two signed edges:
`total = 0` (the `total - 1#usize` underflow) and, on `total < 64`, `total = 63` (the checked
sign-extension subtraction overflow). -/
theorem unpack_code_ref (self : packing.Packing) (area : Slice U8) (i total : Usize)
    (hsigned : self.signed_code = true → 0 < total.val ∧ (total.val < 64 → total.val ≤ 62))
    (htot : total.val ≤ 64)
    (hmul : i.val * total.val + total.val ≤ Usize.max)
    (harea : (i.val * total.val + total.val - 1) / 8 < area.val.length) :
    packing.Packing.unpack_code self area i total = ok (unpackRef self area i total) := by
  obtain ⟨y, hloop, v', hyv, hv'⟩ :=
    WP.spec_imp_exists (unpack_code_loop_spec self area i total htot hmul harea)
  subst hyv
  have hc64 : codeSum self area i total < 2 ^ 64 := by
    have h := v'.bv.isLt; rw [hv'] at h; exact h
  have hv'ofNat : v' = (⟨BitVec.ofNat 64 (codeSum self area i total)⟩ : I64) := by
    apply I64.eq_of_toNat_eq
    rw [hv', BitVec.toNat_ofNat, Nat.mod_eq_of_lt hc64]
  unfold packing.Packing.unpack_code
  rw [hloop]
  simp only [bind_tc_ok]
  change (if self.signed_code = true then
      if total < 64#usize then
        (do let i1 ← total - 1#usize
            let sign_bit ← 1#i64 <<< i1
            let i2 ← lift (v' &&& sign_bit)
            if (i2 != 0#i64) = true then
              (do let i3 ← 1#i64 <<< total
                  v' - i3)
            else ok v')
      else ok v'
    else ok v') = ok (unpackRef self area i total)
  by_cases hs : self.signed_code = true
  · rw [if_pos hs]
    by_cases ht : total < 64#usize
    · rw [if_pos ht]
      have htv : total.val < 64 := (UScalar.lt_equiv total 64#usize).mp ht
      have hpos : 0 < total.val := (hsigned hs).1
      have hs62 : total.val ≤ 62 := (hsigned hs).2 htv
      have hc := codeSum_lt self area i total
      have hp62 : 2 ^ total.val ≤ 2 ^ 62 :=
        Nat.pow_le_pow_right (by norm_num) hs62
      have h1u : (1#usize : Usize).val = 1 := by decide
      obtain ⟨u1, hu1eq, hu1v, -⟩ := WP.spec_imp_exists
        (UScalar.sub_spec (ty := .Usize) (x := total) (y := 1#usize) (by rw [h1u]; omega))
      have hu1val : u1.val = total.val - 1 := by rw [hu1v]; rfl
      rw [hu1eq]
      simp only [bind_tc_ok]
      obtain ⟨sb, hsbeq, _, hsbbv⟩ := WP.spec_imp_exists
        (IScalar.ShiftLeft_spec (x := 1#i64) (y := u1)
          (by simp only [IScalarTy.I64_numBits_eq, hu1val]; omega))
      rw [hsbeq]
      simp only [bind_tc_ok, Aeneas.Std.lift]
      have h1 : ((1#i64 : I64).bv).toNat = 1 := by decide
      have hsbv : sb.bv.toNat = 2 ^ (total.val - 1) := by
        rw [hsbbv, BitVec.toNat_shiftLeft, h1, Nat.shiftLeft_eq, Nat.one_mul, hu1val,
          IScalarTy.I64_numBits_eq]
        exact Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega : total.val - 1 < 64))
      by_cases hb : ((v' &&& sb) != 0#i64) = true
      · rw [if_pos hb]
        have hb' : (codeSum self area i total).testBit (total.val - 1) = true :=
          (signbit_iff v' sb (codeSum self area i total) (total.val - 1) hv' hsbv).mp hb
        have hv'val : v'.val = (codeSum self area i total : Int) := by
          have h2 : 2 * v'.bv.toNat < 2 ^ 64 := by
            rw [hv']; have := hp62; have := Nat.two_pow_pos total.val; omega
          show v'.bv.toInt = _
          rw [BitVec.toInt_eq_toNat_of_lt h2, hv']
        obtain ⟨i3, hi3eq, _, hi3bv⟩ := WP.spec_imp_exists
          (IScalar.ShiftLeft_spec (x := 1#i64) (y := total)
            (by simp only [IScalarTy.I64_numBits_eq]; exact htv))
        rw [hi3eq]
        simp only [bind_tc_ok]
        have hi3nat : i3.bv.toNat = 2 ^ total.val := by
          rw [hi3bv, BitVec.toNat_shiftLeft, h1, Nat.shiftLeft_eq, Nat.one_mul,
            IScalarTy.I64_numBits_eq]
          exact Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) htv)
        have hi3val' : i3.val = (2 ^ total.val : Int) := by
          have h2 : 2 * i3.bv.toNat < 2 ^ 64 := by
            rw [hi3nat]; have := hp62; have := Nat.two_pow_pos total.val; omega
          show i3.bv.toInt = (2 ^ total.val : Int)
          rw [BitVec.toInt_eq_toNat_of_lt h2, hi3nat]
          push_cast
          ring
        have hlo : IScalar.min IScalarTy.I64 ≤ v'.val - i3.val := by
          rw [IScalar.min_IScalarTy_I64_eq, I64.min_eq, hv'val, hi3val']
          have hpi : (2 : Int) ^ total.val ≤ 2 ^ 62 := by exact_mod_cast hp62
          have hci : (codeSum self area i total : Int) < 2 ^ total.val := by exact_mod_cast hc
          omega
        have hhi : v'.val - i3.val ≤ IScalar.max IScalarTy.I64 := by
          rw [IScalar.max_IScalarTy_I64_eq, I64.max_eq, hv'val, hi3val']
          have hci : (codeSum self area i total : Int) < 2 ^ total.val := by exact_mod_cast hc
          have h0 : (0 : Int) < 2 ^ total.val := by exact_mod_cast Nat.two_pow_pos total.val
          omega
        obtain ⟨z, hzeq, hzval⟩ := WP.spec_imp_exists (IScalar.sub_spec (x := v') (y := i3) hlo hhi)
        rw [hzeq]
        have href : unpackRef self area i total
            = (⟨BitVec.ofInt 64 ((codeSum self area i total : Int) - 2 ^ total.val)⟩ : I64) := by
          unfold unpackRef
          rw [if_pos]
          exact ⟨hs, htv, hb'⟩
        rw [href]
        simp only [Result.ok.injEq]
        apply I64.eq_of_toInt_eq
        show z.val = (BitVec.ofInt 64 ((codeSum self area i total : Int) - 2 ^ total.val)).toInt
        have hci : (codeSum self area i total : Int) < 2 ^ total.val := by exact_mod_cast hc
        have hpi : (2 : Int) ^ total.val ≤ 2 ^ 62 := by exact_mod_cast hp62
        have hr1 : -(2 : Int) ^ (64 - 1 : Nat) ≤ (codeSum self area i total : Int) - 2 ^ total.val := by
          have h0 : (0 : Int) ≤ (codeSum self area i total : Int) := Int.natCast_nonneg _
          have h63 : (2 : Int) ^ 62 ≤ 2 ^ (64 - 1 : Nat) := by norm_num
          linarith
        have hr2 : (codeSum self area i total : Int) - 2 ^ total.val < 2 ^ (64 - 1 : Nat) := by
          have hX : (0 : Int) < 2 ^ (64 - 1 : Nat) := by norm_num
          linarith
        have htoInt : (BitVec.ofInt 64 ((codeSum self area i total : Int) - 2 ^ total.val)).toInt
            = (codeSum self area i total : Int) - 2 ^ total.val :=
          BitVec.toInt_ofInt_eq_self (by norm_num) hr1 hr2
        rw [htoInt]
        show z.val = (codeSum self area i total : Int) - 2 ^ total.val
        rw [hzval, hv'val, hi3val']
      · rw [if_neg hb]
        have hb' : (codeSum self area i total).testBit (total.val - 1) ≠ true := by
          intro h
          exact hb ((signbit_iff v' sb (codeSum self area i total) (total.val - 1) hv' hsbv).mpr h)
        have href : unpackRef self area i total
            = (⟨BitVec.ofNat 64 (codeSum self area i total)⟩ : I64) := by
          unfold unpackRef
          rw [if_neg]
          rintro ⟨-, -, hh⟩
          exact hb' hh
        rw [href, hv'ofNat]
    · rw [if_neg ht]
      have htv : ¬ total.val < 64 := by
        intro h; exact ht ((UScalar.lt_equiv total 64#usize).mpr h)
      have href : unpackRef self area i total
          = (⟨BitVec.ofNat 64 (codeSum self area i total)⟩ : I64) := by
        unfold unpackRef
        rw [if_neg]
        rintro ⟨-, h, -⟩
        exact htv h
      rw [href, hv'ofNat]
  · rw [if_neg hs]
    have href : unpackRef self area i total
        = (⟨BitVec.ofNat 64 (codeSum self area i total)⟩ : I64) := by
      unfold unpackRef
      rw [if_neg]
      rintro ⟨h, -⟩
      exact hs h
    rw [href, hv'ofNat]

/-! ## Domain edges excluded from the law

`unpack_code_ref` carries the two failed faces of the sign-extension branch as `hsigned` hypotheses
rather than absorbing them: a signed code needs `0 < total.val`, and when `total.val < 64` needs
`total.val ≤ 62`. At `total = 0` the `total - 1#usize` subtraction underflows; at `total = 63`,
`1i64 <<< 63 = i64::MIN` and the checked subtraction `v - i64::MIN` leaves range, so `unpack_code`
returns `integerOverflow` and no `ok (unpackRef …)` law holds there. `total = 64` and unsigned codes
skip the branch and return the loop's `⟨BitVec.ofNat 64 (codeSum …)⟩`. -/

end apeiron_word

#print axioms apeiron_word.unpack_code_loop_body_step
#print axioms apeiron_word.unpack_code_loop_step
#print axioms apeiron_word.unpack_code_loop_spec
#print axioms apeiron_word.unpack_code_ref
