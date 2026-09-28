import ApeironWord
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases

/-!
# Exact scalar tag scan

The scalar SWAR kernel marks a lane after setting every byte's high bit.  Consequently subtraction
of the repeated one byte cannot borrow across a byte boundary.  The proofs below isolate the only
finite calculation at width eight; word composition is structural.
-/

namespace Apeiron.GrowableScan

open Aeneas Aeneas.Std Result
open apeiron_word

/-- `word_at` returns exactly the eight source bytes beginning at an in-bounds offset. -/
theorem word_at_exact (line : Slice U8) (off : Usize)
    (hwindow : off.val + 8 ≤ line.val.length) :
    ∃ bytes : Array U8 8#usize,
      bytes.val = line.val.slice off.val (off.val + 8) ∧
      table.scan.word_at line off =
        ok (core.num.U64.from_le_bytes bytes) := by
  have hsliceLen : (line.val.slice off.val (off.val + 8)).length = 8 := by
    rw [List.slice_length]
    omega
  let bytes : Array U8 8#usize :=
    Array.from (line.val.slice off.val (off.val + 8)) (by simpa using hsliceLen)
  refine ⟨bytes, by simp [bytes], ?_⟩
  have hadd : off.val + 8 ≤ Usize.max :=
    hwindow.trans (Slice.length_ineq line)
  have hspec : table.scan.word_at line off ⦃ word =>
      word = core.num.U64.from_le_bytes bytes ⦄ := by
    unfold table.scan.word_at table.scan.LANES_PER_WORD
    step as ⟨initial, back, hinitial, hback⟩
    step as ⟨stop, hstop⟩
    have hstart : off ≤ stop := by
      rw [UScalar.le_equiv, hstop]
      omega
    have hend : stop ≤ line.len := by
      rw [UScalar.le_equiv, hstop, Slice.len_val]
      exact hwindow
    step as ⟨window, hwindowVal, hwindowLen⟩
    have hlen : initial.length = window.length := by
      calc
        initial.length = (Array.repeat 8#usize 0#u8).val.length :=
          congrArg List.length hinitial
        _ = 8 := by simp
        _ = window.length := by rw [hwindowLen, hstop]; omega
    step as ⟨copied, hcopied⟩
    congr 1
    rw [Array.eq_iff]
    rw [hback, Array.from_slice_val]
    · simp [hcopied, hwindowVal, hstop, bytes]
    · simp [hcopied, hwindowLen, hstop]
  obtain ⟨word, hword, heq⟩ := WP.spec_imp_exists hspec
  rw [hword, heq]

/-- Asymmetric bytes distinguish the generated little-endian conversion from big-endian order. -/
def ascendingByteLine : Slice U8 :=
  Slice.from [1#u8, 2#u8, 3#u8, 4#u8, 5#u8, 6#u8, 7#u8, 8#u8] (by scalar_tac)

#guard match (table.scan.word_at ascendingByteLine 0#usize).match with
  | .ok word => word == 0x0807060504030201#u64
  | _ => false

/-- Exact successful normalization of the machine-width bound in generated `summary_index`.
The platform model admits both 32-bit and 64-bit `usize`; neither needs a caller premise. -/
theorem summary_max_shift_six :
    ∃ bound : Usize, core.num.Usize.MAX >>> 6#i32 = ok bound ∧
      bound.val = Usize.max / 64 := by
  have hwidth : 6 < System.Platform.numBits := by
    cases System.Platform.numBits_eq <;> omega
  have hval : (6#i32).val = 6 := by rfl
  have hnat : (6#i32).toNat = 6 := by rfl
  have hnonneg : (0 : Int) ≤ 6 := by omega
  refine ⟨⟨core.num.Usize.MAX.bv.ushiftRight 6⟩, ?_, ?_⟩
  · simp only [HShiftRight.hShiftRight, UScalar.shiftRight_IScalar, hval, hnat,
      UScalar.shiftRight, UScalarTy.numBits, hwidth, hnonneg, ↓reduceIte]
  · change (core.num.Usize.MAX.bv.ushiftRight 6).toNat = Usize.max / 64
    rw [BitVec.ushiftRight_eq, BitVec.toNat_ushiftRight]
    change core.num.Usize.MAX.val >>> 6 = Usize.max / 64
    rw [core.num.Usize.MAX, UScalar.ofNatCore_val_eq, Nat.shiftRight_eq_div_pow]

/-- Fixed-width trailing-zero semantics: zero returns the word width; a nonzero input returns the
least exponent whose corresponding bit is set. -/
def FixedWidthTrailingZeros (value : Usize) (zeros : U32) : Prop :=
  (value.val = 0 ∧ zeros.val = System.Platform.numBits) ∨
    (value.val ≠ 0 ∧ zeros.val < System.Platform.numBits ∧
      2 ^ zeros.val ∣ value.val ∧ ¬2 ^ (zeros.val + 1) ∣ value.val)

def TrailingZerosInv (original : Usize) (state : Usize × U32) : Prop :=
  state.2.val ≤ System.Platform.numBits ∧ original.val = state.1.val * 2 ^ state.2.val

theorem usize_and_one_val (value : Usize) :
    (value &&& 1#usize).val = value.val % 2 := by
  change (value.bv &&& (1#usize).bv).toNat = value.bv.toNat % 2
  rw [BitVec.toNat_and]
  have hone : (1#usize).bv.toNat = 1 := by rfl
  rw [hone, Nat.and_one_is_mod]

/-- The extracted bounded loop returns the word width at zero and the least set-bit index otherwise. -/
theorem summary_trailing_zeros_eq (value : Usize) :
    table.summary_trailing_zeros value ⦃ zeros => FixedWidthTrailingZeros value zeros ⦄ := by
  unfold table.summary_trailing_zeros table.summary_trailing_zeros_loop
  apply loop.spec_decr_nat
    (measure := fun state : Usize × U32 => System.Platform.numBits - state.2.val)
    (inv := TrailingZerosInv value)
    (post := FixedWidthTrailingZeros value)
  · intro state hstate
    rcases state with ⟨current, zeros⟩
    rcases hstate with ⟨hzeros, hvalue⟩
    dsimp only at hzeros hvalue ⊢
    unfold table.summary_trailing_zeros_loop.body
    by_cases hwidth : zeros < core.num.Usize.BITS
    · rw [if_pos hwidth]
      have hzerosLt : zeros.val < System.Platform.numBits := by simpa using hwidth
      simp only [lift, bind_tc_ok]
      by_cases heven : current &&& 1#usize = 0#usize
      · rw [if_pos heven]
        have hzeroAdd : zeros.val + 1 ≤ U32.max := by
          have := System.Platform.numBits_eq
          rw [U32.max_eq]
          omega
        step
        have honeWidth : 1 < System.Platform.numBits := by
          cases System.Platform.numBits_eq <;> omega
        step
        have hmod : current.val % 2 = 0 := by
          rw [← usize_and_one_val]
          exact congrArg UScalar.val heven
        have hcurrent : current.val = current.val / 2 * 2 := by omega
        refine ⟨⟨?_, ?_⟩, ?_⟩
        · simpa [zeros1_post] using Nat.succ_le_of_lt hzerosLt
        · rw [value1_post, Nat.shiftRight_eq_div_pow]
          simp only [pow_one]
          rw [zeros1_post, pow_succ]
          calc
            value.val = current.val * 2 ^ zeros.val := hvalue
            _ = (current.val / 2 * 2) * 2 ^ zeros.val := by rw [← hcurrent]
            _ = current.val / 2 * 2 ^ zeros.val * 2 := by ac_rfl
            _ = current.val / 2 * (2 ^ zeros.val * 2) :=
              Nat.mul_assoc (current.val / 2) (2 ^ zeros.val) 2
        · simp [zeros1_post]
          omega
      · rw [if_neg heven, Aeneas.Std.WP.spec_ok]
        right
        have hmod : current.val % 2 ≠ 0 := by
          intro hzero
          apply heven
          apply UScalar.eq_of_val_eq
          rw [usize_and_one_val]
          exact hzero
        refine ⟨?_, hzerosLt, ⟨current.val, by simpa [Nat.mul_comm] using hvalue⟩, ?_⟩
        · intro hzero
          rw [hzero] at hvalue
          simp at hvalue
          omega
        · rw [pow_succ]
          intro hdvd
          rw [hvalue] at hdvd
          have hevenCurrent : 2 ∣ current.val := by
            apply (Nat.mul_dvd_mul_iff_right (by positivity : 0 < 2 ^ zeros.val)).mp
            simpa only [Nat.mul_comm (2 ^ zeros.val) 2] using hdvd
          exact hmod (Nat.dvd_iff_mod_eq_zero.mp hevenCurrent)
    · rw [if_neg hwidth, Aeneas.Std.WP.spec_ok]
      left
      have hzerosEq : zeros.val = System.Platform.numBits := by
        have : ¬zeros.val < System.Platform.numBits := by simpa using hwidth
        omega
      refine ⟨?_, hzerosEq⟩
      rw [hzerosEq] at hvalue
      by_cases hcurrentZero : current.val = 0
      · rw [hcurrentZero] at hvalue
        simpa using hvalue
      · have hbound : value.val < 2 ^ System.Platform.numBits := by
          simpa [UScalar.size] using value.hBounds
        have hcurrent : 1 ≤ current.val := Nat.one_le_iff_ne_zero.mpr hcurrentZero
        have hpow : 2 ^ System.Platform.numBits ≤ current.val * 2 ^ System.Platform.numBits := by
          simpa using Nat.mul_le_mul_right (2 ^ System.Platform.numBits) hcurrent
        rw [hvalue] at hbound
        omega
  · exact ⟨Nat.zero_le _, by simp⟩

private theorem summary_cast_and_le (mixed mask : U64) :
    (UScalar.cast .Usize (mixed &&& mask)).val ≤ mask.val := by
  rw [UScalar.cast_val_eq, UScalar.val_and]
  exact (Nat.mod_le _ _).trans Nat.and_le_right

private theorem summary_finish_bound (place power : U64) (lenWords : Usize)
    (hpositive : 0 < power.val) (hbound : power.val ≤ lenWords.val * 64) :
    (do
      let mask ← power - 1#u64
      let high ← place >>> 8#i32
      let mixed ← lift (core.num.U64.wrapping_mul high 11400714819323198485#u64)
      let shifted ← mixed >>> 29#i32
      let mixed1 ← lift (mixed ^^^ shifted)
      let masked ← lift (mixed1 &&& mask)
      let index ← lift (UScalar.cast .Usize masked)
      ok (some index)) ⦃ out => ∀ index, out = some index → index.val / 64 < lenWords.val ⦄ := by
  step as ⟨mask, hmask⟩
  step as ⟨high, hhigh⟩
  simp only [lift, bind_tc_ok]
  step as ⟨shifted, hshifted⟩
  intro index hindex
  have heq := Option.some.inj hindex
  rw [← heq]
  have hle := summary_cast_and_le
    (core.num.U64.wrapping_mul high 11400714819323198485#u64 ^^^ shifted) mask
  omega

private theorem summary_fallback_bound (place : U64) (lenWords : Usize)
    (hnonzero : lenWords.val ≠ 0) :
    (do
      let zeros ← table.summary_trailing_zeros lenWords
      let bits ← zeros + 6#u32
      if bits >= core.num.U64.BITS then ok none
      else
        let power ← 1#u64 <<< bits
        let mask ← power - 1#u64
        let high ← place >>> 8#i32
        let mixed ← lift (core.num.U64.wrapping_mul high 11400714819323198485#u64)
        let shifted ← mixed >>> 29#i32
        let mixed1 ← lift (mixed ^^^ shifted)
        let masked ← lift (mixed1 &&& mask)
        let index ← lift (UScalar.cast .Usize masked)
        ok (some index)) ⦃ out => ∀ index, out = some index → index.val / 64 < lenWords.val ⦄ := by
  step with summary_trailing_zeros_eq lenWords as ⟨zeros, hzeros⟩
  rcases hzeros with hzero | ⟨_, hwidth, hdiv, _⟩
  · exact False.elim (hnonzero hzero.1)
  have hadd : zeros.val + 6 ≤ U32.max := by
    have := System.Platform.numBits_eq
    rw [U32.max_eq]
    omega
  step as ⟨bits, hbits⟩
  split
  · simp only [WP.spec_ok]
    intro index hindex
    cases hindex
  · rename_i hsmall
    have hbitsLt : bits.val < 64 := by simpa using hsmall
    step as ⟨power, hpower⟩
    have hpowLt : 2 ^ bits.val < 2 ^ 64 := Nat.pow_lt_pow_right (by omega) hbitsLt
    have hpowerVal : power.val = 2 ^ bits.val := by
      calc
        power.val = 2 ^ bits.val % 2 ^ 64 := by
          simpa [Nat.shiftLeft_eq, U64.size, U64.numBits, UScalarTy.numBits] using hpower
        _ = _ := Nat.mod_eq_of_lt hpowLt
    have hp : 0 < power.val := by rw [hpowerVal]; positivity
    have hb : power.val ≤ lenWords.val * 64 := by
      have hdivLe := Nat.le_of_dvd (Nat.pos_of_ne_zero hnonzero) hdiv
      rw [hpowerVal, hbits, pow_add]
      norm_num only [Nat.reducePow]
      exact Nat.mul_le_mul_right 64 hdivLe
    simpa only [lift, bind_tc_ok] using summary_finish_bound place power lenWords hp hb

private theorem summary_index_bound_spec (R V : Type) (place : U64) (lenWords : Usize) :
    table.Region.summary_index R V place lenWords ⦃ out =>
      ∀ index, out = some index → index.val / 64 < lenWords.val ⦄ := by
  unfold table.Region.summary_index
  split
  · simp only [WP.spec_ok]
    intro index hindex
    cases hindex
  · rename_i hnonzero
    have hn : lenWords.val ≠ 0 := by
      intro hz
      apply hnonzero
      apply UScalar.eq_of_val_eq
      simpa using hz
    simp only [core.num.Usize.is_power_of_two, bind_tc_ok]
    split
    · obtain ⟨bound, hshift, hbound⟩ := summary_max_shift_six
      rw [hshift, bind_tc_ok]
      split
      · rename_i hsmall
        have hsmallVal : lenWords.val ≤ Usize.max / 64 := by
          simpa only [UScalar.le_equiv, hbound] using hsmall
        have hmax : Usize.max ≤ 18446744073709551615 := by
          rcases Usize.bounds_eq with h | h <;> rw [h] <;> norm_num [U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
        have hlenLt : lenWords.val < 2 ^ 64 := by omega
        have hscaled : lenWords.val * 64 < 2 ^ 64 := by omega
        have hcast : (UScalar.cast .U64 lenWords).val = lenWords.val := by
          rw [UScalar.cast_val_eq]
          exact Nat.mod_eq_of_lt hlenLt
        simp only [lift, bind_tc_ok]
        step as ⟨power, hpower⟩
        have hp : power.val = lenWords.val * 64 := by
          calc
            power.val = lenWords.val * 64 % 2 ^ 64 := by
              simpa [hcast, Nat.shiftLeft_eq, U64.size, U64.numBits, UScalarTy.numBits] using hpower
            _ = _ := Nat.mod_eq_of_lt hscaled
        have hpos : 0 < power.val := by rw [hp]; omega
        have hle : power.val ≤ lenWords.val * 64 := by rw [hp]
        simpa only [lift, bind_tc_ok] using summary_finish_bound place power lenWords hpos hle
      · exact summary_fallback_bound place lenWords hn
    · exact summary_fallback_bound place lenWords hn

/-- Every successful generated summary index addresses an existing summary word. -/
theorem summary_index_word_lt_len (R V : Type) (place : U64)
    (lenWords index : Usize)
    (hindex : table.Region.summary_index R V place lenWords = ok (some index)) :
    index.val / 64 < lenWords.val := by
  have hspec := summary_index_bound_spec R V place lenWords
  rw [hindex, WP.spec_ok] at hspec
  exact hspec index rfl

/-- A machine-index shift below the U64 width produces a nonzero single-bit mask. -/
theorem u64_one_shift_nonzero (shift : Usize) (hshift : shift.val < 64) :
    ∃ mask : U64, (1#u64 <<< shift) = ok mask ∧ mask ≠ 0#u64 := by
  obtain ⟨mask, hresult, hvalue, _⟩ :=
    WP.spec_imp_exists (U64.ShiftLeft_spec (1#u64) shift hshift)
  have hpowLt : 2 ^ shift.val < 2 ^ 64 := Nat.pow_lt_pow_right (by omega) hshift
  have hmask : mask.val = 2 ^ shift.val := by
    calc
      mask.val = 2 ^ shift.val % 2 ^ 64 := by
        simpa [Nat.shiftLeft_eq, U64.size, U64.numBits, UScalarTy.numBits] using hvalue
      _ = _ := Nat.mod_eq_of_lt hpowLt
  refine ⟨mask, hresult, ?_⟩
  intro hzero
  have hz := congrArg UScalar.val hzero
  simp only [hmask] at hz
  have hp : 0 < 2 ^ shift.val := by positivity
  norm_num at hz

/-- OR with a nonzero mask makes the subsequent AND with that mask nonzero. -/
theorem u64_or_mask_and_mask (word mask : U64) (hmask : mask ≠ 0#u64) :
    (word ||| mask) &&& mask ≠ 0#u64 := by
  intro hzero
  apply hmask
  apply (UScalar.eq_equiv_bv_eq _ _).mpr
  have hbits := congrArg UScalar.bv hzero
  simp only [UScalar.bv_and, UScalar.bv_or, BitVec.and_or_distrib_right,
    BitVec.and_self] at hbits
  exact (BitVec.or_eq_zero_iff.mp hbits).2

/-- Successful evaluation of the generated remainder admits the U64 shift. -/
theorem summary_remainder_lt64 (index shift : Usize)
    (hrem : index % 64#usize = ok shift) : shift.val < 64 := by
  have hspec := Usize.rem_spec index (y := 64#usize) (by simp)
  rw [hrem, WP.spec_ok] at hspec
  have hv : shift.val = index.val % 64 := by simpa using hspec
  rw [hv]
  exact Nat.mod_lt _ (by omega)

/-- The actual remainder and shift operations always produce a nonzero summary mask. -/
theorem summary_mask_nonzero (index : Usize) :
    ∃ shift : Usize, ∃ mask : U64,
      index % 64#usize = ok shift ∧ (1#u64 <<< shift) = ok mask ∧ mask ≠ 0#u64 := by
  obtain ⟨shift, hrem, _⟩ :=
    WP.spec_imp_exists (Usize.rem_spec index (y := 64#usize) (by simp))
  obtain ⟨mask, hshift, hnonzero⟩ := u64_one_shift_nonzero shift (summary_remainder_lt64 index shift hrem)
  exact ⟨shift, mask, hrem, hshift, hnonzero⟩

/-- At index zero, a zero word acquires the first summary bit. -/
theorem summary_mask_zero_word_witness :
    (0#usize % 64#usize) = ok 0#usize ∧
      (1#u64 <<< 0#usize) = ok 1#u64 ∧
      (0#u64 ||| 1#u64) &&& 1#u64 ≠ 0#u64 := by
  refine ⟨?_, ?_, ?_⟩
  · rfl
  · rfl
  · exact u64_or_mask_and_mask 0#u64 1#u64 (by decide)

/-- Every successful generated summary write makes the same generated query positive. -/
theorem summary_set_implies_maybe {R V : Type} (before after : table.Region R V)
    (place : U64) (hset : table.Region.summary_set before place = ok after) :
    table.Region.summary_maybe after place = ok true := by
  obtain ⟨result, hindex, _⟩ := WP.spec_imp_exists
    (summary_index_bound_spec R V place (alloc.vec.Vec.len before.summary))
  cases result with
  | none =>
      simp only [table.Region.summary_set, hindex, bind_tc_ok] at hset
      have heq : before = after := Result.ok_injective hset
      subst after
      simp only [table.Region.summary_maybe, hindex, bind_tc_ok]
  | some index =>
      obtain ⟨shift, mask, hrem, hshift, hmask⟩ := summary_mask_nonzero index
      obtain ⟨wordIndex, hdiv, hwordIndex⟩ :=
        WP.spec_imp_exists (Usize.div_spec index (y := 64#usize) (by simp))
      have hbound : wordIndex.val < before.summary.length := by
        have h := summary_index_word_lt_len R V place
          (alloc.vec.Vec.len before.summary) index hindex
        simpa [hwordIndex] using h
      obtain ⟨⟨oldWord, back⟩, hmut, _, hback⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.index_mut_usize_spec before.summary wordIndex hbound)
      subst back
      simp only [table.Region.summary_set, hindex, bind_tc_ok, hrem, hshift, hdiv,
        alloc.vec.Vec.index_mut_slice_index, hmut, lift] at hset
      have heq := Result.ok_injective hset
      subst after
      have hlen : alloc.vec.Vec.len (before.summary.set wordIndex (oldWord ||| mask)) =
          alloc.vec.Vec.len before.summary := by
        apply UScalar.eq_of_val_eq
        simp only [alloc.vec.Vec.len_val, alloc.vec.Vec.set_length]
      have hupdated : wordIndex.val < (before.summary.set wordIndex (oldWord ||| mask)).length := by
        simpa only [alloc.vec.Vec.set_length] using hbound
      obtain ⟨newWord, hget, hnewWord⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.index_usize_spec (before.summary.set wordIndex (oldWord ||| mask))
          wordIndex hupdated)
      have hnew : newWord = oldWord ||| mask := by
        simpa using hnewWord
      subst newWord
      simp only [table.Region.summary_maybe, hlen, hindex, bind_tc_ok, hdiv,
        alloc.vec.Vec.index_slice_index, hget, hrem, hshift, lift]
      have hnonzero := u64_or_mask_and_mask oldWord mask hmask
      simp [hnonzero]

/-- Successful publication retains the summary bit written before the occupancy tag. -/
theorem publish_live_implies_summary_maybe {R V : Type}
    (before after : table.Region R V) (index : Usize)
    (payload : cell.Sigma × R × V) (place : U64)
    (hpub : table.Region.publish_live before index payload place = ok after) :
    table.Region.summary_maybe after place = ok true := by
  unfold table.Region.publish_live at hpub
  cases hslot : alloc.vec.Vec.index_mut
      (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) before.slots index with
  | vis eff k => simp [hslot] at hpub
  | div => simp [hslot] at hpub
  | ret slotResult =>
      rcases slotResult with ⟨_, slotBack⟩
      simp only [hslot, bind_tc_ok] at hpub
      simp at hpub
      generalize hset : table.Region.summary_set
          { before with slots := slotBack payload } place = summaryResult at hpub
      cases summaryResult with
      | vis eff k => simp at hpub
      | div => simp at hpub
      | ret interim =>
          simp at hpub
          generalize htag : table.tag place = tagResult at hpub
          cases tagResult with
          | vis eff k => simp at hpub
          | div => simp at hpub
          | ret tag =>
              simp at hpub
              generalize htagSlot : alloc.vec.Vec.index_mut_usize
                  interim.tags index = tagSlotResult at hpub
              cases tagSlotResult with
              | vis eff k => simp at hpub
              | div => simp at hpub
              | ret tagResult =>
                  rcases tagResult with ⟨_, tagBack⟩
                  simp at hpub
                  subst after
                  change table.Region.summary_maybe interim place = ok true
                  exact summary_set_implies_maybe
                    { before with slots := slotBack payload } interim place hset

private theorem summary_index_zero_one :
    table.Region.summary_index Unit Unit 0#u64 1#usize = ok (some 0#usize) := by
  have hn : (1#usize) ≠ 0#usize := by simp
  have hpow : (decide (1#usize).val.isPowerOfTwo) = true := by decide
  obtain ⟨bound, hshift, hbound⟩ := summary_max_shift_six
  have hle : 1#usize ≤ bound := by
    rw [UScalar.le_equiv, hbound]
    rcases Usize.bounds_eq with h | h <;>
      norm_num [h, U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
  have hcast : UScalar.cast .U64 (1#usize) = 1#u64 := by
    apply UScalar.eq_of_val_eq
    simp [UScalar.cast_val_eq]
  have hp : (1#u64 <<< 6#i32) = ok 64#u64 := by rfl
  have hm : (64#u64 - 1#u64) = ok 63#u64 := by rfl
  have hh : (0#u64 >>> 8#i32) = ok 0#u64 := by rfl
  have hmul : core.num.U64.wrapping_mul 0#u64 11400714819323198485#u64 =
      0#u64 := by rfl
  have hs : (0#u64 >>> 29#i32) = ok 0#u64 := by rfl
  have hx : (0#u64 ^^^ 0#u64) =
      0#u64 := by rfl
  have ha : (0#u64 &&& 63#u64) = 0#u64 := by rfl
  have hc : UScalar.cast .Usize (0#u64) = 0#usize := by
    apply UScalar.eq_of_val_eq
    cases System.Platform.numBits_eq <;> simp [UScalar.cast_val_eq, *]
  unfold table.Region.summary_index
  rw [if_neg hn]
  simp only [core.num.Usize.is_power_of_two, bind_tc_ok, hpow, ↓reduceIte,
    hshift, hle, lift, hcast, hp, hm, hh, hmul, hs, hx, ha, hc]

/-- Concrete nonempty summary state used to witness the generated write branch. -/
def oneWordZeroRegion : table.Region Unit Unit :=
  { tags := alloc.vec.Vec.new U8
    slots := alloc.vec.Vec.new (cell.Sigma × Unit × Unit)
    summary := alloc.vec.Vec.from [0#u64] (by
      rcases Usize.bounds_eq with h | h <;>
        norm_num [h, U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]) }

/-- A one-word zero summary takes the actual generated write and positive-query branches. -/
theorem summary_set_maybe_one_word_witness :
    ∃ after,
      table.Region.summary_set oneWordZeroRegion 0#u64 = ok after ∧
      table.Region.summary_maybe after 0#u64 = ok true := by
  have hlen : alloc.vec.Vec.len oneWordZeroRegion.summary = 1#usize := by
    apply UScalar.eq_of_val_eq
    simp [oneWordZeroRegion]
  have hbound : (0#usize).val < oneWordZeroRegion.summary.length := by
    simp [oneWordZeroRegion]
  obtain ⟨⟨oldWord, back⟩, hmut, hold, hback⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_mut_usize_spec oneWordZeroRegion.summary 0#usize hbound)
  have ho : oldWord = 0#u64 := by simpa [oneWordZeroRegion] using hold
  rw [ho, hback] at hmut
  have hr : (0#usize % 64#usize) = ok 0#usize := summary_mask_zero_word_witness.1
  have hs : (1#u64 <<< 0#usize) = ok 1#u64 := summary_mask_zero_word_witness.2.1
  have hd : (0#usize / 64#usize) = ok 0#usize := by rfl
  let after : table.Region Unit Unit :=
    { oneWordZeroRegion with summary := oneWordZeroRegion.summary.set 0#usize 1#u64 }
  have hset : table.Region.summary_set oneWordZeroRegion 0#u64 = ok after := by
    simp only [table.Region.summary_set, hlen, summary_index_zero_one, bind_tc_ok,
      hr, hs, hd, alloc.vec.Vec.index_mut_slice_index, hmut, lift]
    rfl
  exact ⟨after, hset, summary_set_implies_maybe oneWordZeroRegion after 0#u64 hset⟩

/-- One payload and tag slot with a zero summary word exercise the publication path. -/
def oneSlotZeroRegion : table.Region Unit Unit :=
  { tags := alloc.vec.Vec.from [0#u8] (by
      rcases Usize.bounds_eq with h | h <;>
        norm_num [h, U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits])
    slots := alloc.vec.Vec.from [(cell.Sigma.Parabolic, (), ())] (by
      rcases Usize.bounds_eq with h | h <;>
        norm_num [h, U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits])
    summary := oneWordZeroRegion.summary }

#guard match (table.Region.summary_maybe oneSlotZeroRegion 0#u64).match with
  | .ok false => true
  | _ => false

#guard match (table.Region.publish_live oneSlotZeroRegion 0#usize
    (cell.Sigma.Parabolic, (), ()) 0#u64).match with
  | .ok after =>
      match (table.Region.summary_maybe after 0#u64).match with
      | .ok true => true
      | _ => false
  | _ => false

/-- Every generated summary index that names a word stays inside the region's summary vector. -/
structure SpillSummaryShape {R V : Type} (region : table.Region R V) : Prop where
  word_lt : ∀ place index,
    table.Region.summary_index R V place (alloc.vec.Vec.len region.summary) =
        ok (some index) →
      index.val / 64 < region.summary.length

/-- `summary_index` establishes the scalar shape fact for every generated region value. -/
theorem spillSummaryShape {R V : Type} (region : table.Region R V) :
    SpillSummaryShape region := by
  constructor
  intro place index hindex
  simpa [alloc.vec.Vec.len_val] using
    summary_index_word_lt_len R V place (alloc.vec.Vec.len region.summary) index hindex

/-- The exact generated operations witnessing that the same-place summary bit is recorded. -/
def SummaryBitRecorded {R V : Type} (region : table.Region R V) (place : U64) : Prop :=
  ∃ index wordIndex shift mask word,
    table.Region.summary_index R V place (alloc.vec.Vec.len region.summary) =
        ok (some index) ∧
    index / 64#usize = ok wordIndex ∧
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
        region.summary wordIndex = ok word ∧
    index % 64#usize = ok shift ∧
    1#u64 <<< shift = ok mask ∧
    word &&& mask ≠ 0#u64

/-- A successful negative generated query excludes a recorded bit for that same place. -/
theorem summary_maybe_false_excludes_recorded {R V : Type}
    (region : table.Region R V) (place : U64)
    (hmiss : table.Region.summary_maybe region place = ok false) :
    ¬ SummaryBitRecorded region place := by
  rintro ⟨index, wordIndex, shift, mask, word, hindex, hdiv, hword, hrem, hshift, hbit⟩
  simp only [table.Region.summary_maybe, hindex, bind_tc_ok, hdiv, hword, hrem,
    hshift, lift] at hmiss
  simp [hbit] at hmiss

/-- A generated tag always contains the nonzero live bit. -/
theorem tag_ne_zero (place : U64) (tag : U8) (htag : table.tag place = ok tag) :
    tag ≠ 0#u8 := by
  unfold table.tag at htag
  cases hshift : place >>> table.TAG_FINGERPRINT_SHIFT with
  | vis eff k => simp [hshift] at htag
  | div => simp [hshift] at htag
  | ret shifted =>
      simp only [hshift, bind_tc_ok, lift] at htag
      have hbits := congrArg UScalar.bv (Result.ok_injective htag)
      intro hzero
      have hzeroBits := congrArg UScalar.bv hzero
      simp only [UScalar.bv_or] at hbits
      rw [hzeroBits] at hbits
      have hlive := (BitVec.or_eq_zero_iff.mp hbits).1
      have hliveScalar : table.TAG_LIVE = 0#u8 :=
        (UScalar.eq_equiv_bv_eq table.TAG_LIVE 0#u8).mpr hlive
      have hvalue := congrArg UScalar.val hliveScalar
      norm_num [table.TAG_LIVE] at hvalue

/-- Successful publication writes the generated nonzero tag at the published index. -/
theorem publish_live_tag_readback {R V : Type}
    (before after : table.Region R V) (index : Usize)
    (payload : cell.Sigma × R × V) (place : U64)
    (hpub : table.Region.publish_live before index payload place = ok after) :
    ∃ tag, table.tag place = ok tag ∧ tag ≠ 0#u8 ∧
      after.tags.val[index.val]? = some tag := by
  unfold table.Region.publish_live at hpub
  cases hslot : alloc.vec.Vec.index_mut
      (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) before.slots index with
  | vis eff k => simp [hslot] at hpub
  | div => simp [hslot] at hpub
  | ret slotResult =>
      rcases slotResult with ⟨_, slotBack⟩
      simp only [hslot, bind_tc_ok] at hpub
      simp at hpub
      generalize hset : table.Region.summary_set
          { before with slots := slotBack payload } place = summaryResult at hpub
      cases summaryResult with
      | vis eff k => simp at hpub
      | div => simp at hpub
      | ret interim =>
          simp at hpub
          generalize htag : table.tag place = tagResult at hpub
          cases tagResult with
          | vis eff k => simp at hpub
          | div => simp at hpub
          | ret tag =>
              simp at hpub
              generalize htagSlot : alloc.vec.Vec.index_mut_usize
                  interim.tags index = tagSlotResult at hpub
              cases tagSlotResult with
              | vis eff k => simp at hpub
              | div => simp at hpub
              | ret tagResult =>
                  rcases tagResult with ⟨_, tagBack⟩
                  simp at hpub
                  subst after
                  have htagBound : index.val < interim.tags.length := by
                    cases hget : interim.tags.val[index.val]? with
                    | none =>
                        simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize,
                          hget] at htagSlot
                    | some old => exact (List.getElem?_eq_some_iff.mp hget).1
                  have htagSpec :=
                    alloc.vec.Vec.index_mut_usize_spec interim.tags index htagBound
                  rw [htagSlot, WP.spec_ok] at htagSpec
                  have htagBack : tagBack = interim.tags.set index := htagSpec.2
                  refine ⟨tag, ?_, tag_ne_zero place tag htag, ?_⟩
                  · simp
                  change (tagBack tag).val[index.val]? = some tag
                  rw [htagBack, alloc.vec.Vec.set_val_eq]
                  exact List.getElem?_set_self htagBound

/-- The tag fingerprint obtained through a generated address computation. -/
def spillFingerprint {K : Type}
    (address : cell.Sigma → K → Result table.TableAddress)
    (identity : cell.Sigma × K) : Result U8 := do
  let a ← address identity.1 identity.2
  table.tag a.place

/-- Fingerprints respect equality of the complete `(Sigma, key)` identity. -/
structure SpillFingerprintExact {K : Type}
    (address : cell.Sigma → K → Result table.TableAddress) : Prop where
  eq_of_identity : ∀ left right : cell.Sigma × K, left = right →
    spillFingerprint address left = spillFingerprint address right

/-- Generated address/tag composition is fingerprint-exact at equal full identities. -/
theorem spillFingerprintExact {K : Type}
    (address : cell.Sigma → K → Result table.TableAddress) :
    SpillFingerprintExact address := by
  constructor
  intro left right heq
  subst right
  rfl

/-- Equal full identities therefore have equal generated tag fingerprints. -/
theorem spillFingerprint_eq_of_identity {K : Type}
    (address : cell.Sigma → K → Result table.TableAddress)
    (left right : cell.Sigma × K) (heq : left = right) :
    spillFingerprint address left = spillFingerprint address right :=
  (spillFingerprintExact address).eq_of_identity left right heq

/-- Distinct full keys can have the same generated seven-bit fingerprint. -/
theorem spillFingerprint_collision_witness :
    ∃ left right : cell.Sigma × U64,
      left ≠ right ∧
      spillFingerprint (fun _ key => ok { place := key, direct_word := none }) left =
        spillFingerprint (fun _ key => ok { place := key, direct_word := none }) right := by
  refine ⟨(cell.Sigma.Parabolic, 0#u64), (cell.Sigma.Parabolic, 1#u64), ?_, ?_⟩
  · intro heq
    have := congrArg (fun identity : cell.Sigma × U64 => identity.2.val) heq
    norm_num at this
  · have hzero : (0#u64 >>> 8#u32) = ok 0#u64 := by rfl
    have hone : (1#u64 >>> 8#u32) = ok 0#u64 := by rfl
    simp only [spillFingerprint, bind_tc_ok, table.tag, table.TAG_FINGERPRINT_SHIFT,
      hzero, hone]

/-- A lane selected by the generated tag scan.  This is only a candidate location. -/
def SpillTagHit (tags : Slice U8) (wanted : U8) (lane : Usize) : Prop :=
  ∃ bits : U64, table.scan.match_tag tags wanted = ok bits ∧
    bits ≠ 0#u64 ∧ table.scan.least_set_lane bits = ok lane

/-- Even after a tag hit, successful presence still traverses generated `table.matches`:
both `Sigma` equality and the carrier's full-key comparison must return true. -/
theorem fingerprint_hit_requires_full_matches {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (slotSigma : cell.Sigma)
    (carrier : R) (value : V) (querySigma : cell.Sigma) (key : K)
    (address : table.TableAddress) (fingerprint : U8) (tags : Slice U8) (lane : Usize)
    (_htag : table.tag address.place = ok fingerprint)
    (_hit : SpillTagHit tags fingerprint lane)
    (hmatch : table.matches keyRep (slotSigma, carrier, value) querySigma key address =
      ok true) :
    cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slotSigma querySigma = ok true ∧
      keyRep.«matches» querySigma carrier key address = ok true := by
  unfold table.matches at hmatch
  cases heq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slotSigma querySigma with
  | vis eff k => simp [heq] at hmatch
  | div => simp [heq] at hmatch
  | ret equal =>
      cases equal <;> simp_all

private theorem append_or {n m : Nat} (a c : BitVec n) (b d : BitVec m) :
    (a ++ b) ||| (c ++ d) = (a ||| c) ++ (b ||| d) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_or, BitVec.getLsbD_append]
  split <;> rfl

private theorem append_and {n m : Nat} (a c : BitVec n) (b d : BitVec m) :
    (a ++ b) &&& (c ++ d) = (a &&& c) ++ (b &&& d) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_append]
  split <;> rfl

private theorem append_not {n m : Nat} (a : BitVec n) (b : BitVec m) :
    ~~~(a ++ b) = (~~~a) ++ (~~~b) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_not, BitVec.getLsbD_append]
  split
  · simp_all
  · have : i - m < n := by omega
    simp_all

private theorem toNat_append_byte {n : Nat} (a : BitVec n) (b : BitVec 8) :
    (a ++ b).toNat = a.toNat * 256 + b.toNat := by
  rw [BitVec.toNat_append, ← Nat.shiftLeft_add_eq_or_of_lt b.isLt]
  simp [Nat.shiftLeft_eq]

/-- Subtraction splits at an eight-bit boundary when the low byte cannot borrow. -/
theorem append_sub_byte_noBorrow {n : Nat} (a c : BitVec n) (b d : BitVec 8)
    (h : d.toNat ≤ b.toNat) :
    (a ++ b) - (c ++ d) = (a - c) ++ (b - d) := by
  apply BitVec.toNat_inj.mp
  simp only [BitVec.toNat_sub, toNat_append_byte]
  rw [pow_add]
  norm_num
  have ha := a.isLt
  have hc := c.isLt
  have hb := b.isLt
  have hd := d.isLt
  have hlow : (256 - d.toNat + b.toNat) % 256 = b.toNat - d.toNat := by
    have heq : 256 - d.toNat + b.toNat = 256 + (b.toNat - d.toNat) := by omega
    rw [heq, Nat.add_mod]
    simp
    omega
  rw [hlow]
  have hnum :
      2 ^ n * 256 - (c.toNat * 256 + d.toNat) + (a.toNat * 256 + b.toNat) =
        (2 ^ n - c.toNat + a.toNat) * 256 + (b.toNat - d.toNat) := by omega
  rw [hnum]
  rw [Nat.add_mod, Nat.mul_mod_mul_right]
  have hr : b.toNat - d.toNat < 2 ^ n * 256 := by
    have hp : 0 < 2 ^ n := pow_pos (by omega) _
    omega
  rw [Nat.mod_eq_of_lt hr]
  apply Nat.mod_eq_of_lt
  have hmod : (2 ^ n - c.toNat + a.toNat) % 2 ^ n < 2 ^ n :=
    Nat.mod_lt _ (pow_pos (by omega) _)
  omega

/-- The complete eight-bit zero-marker identity; this is the only finite bit-vector base case. -/
theorem byte_zero_marker (x : BitVec 8) :
    (~~~(x ||| ((x ||| 128#8) - 1#8))) &&& 128#8 =
      if x = 0#8 then 128#8 else 0#8 := by
  rcases x with ⟨⟨x, hx⟩⟩
  interval_cases x <;> decide +revert

private abbrev byteWidth (n : Nat) : Nat := Nat.rec 0 (fun _ width => width + 8) n

private def repeatByte (byte : BitVec 8) : (n : Nat) → BitVec (byteWidth n)
  | 0 => 0#0
  | n + 1 => repeatByte byte n ++ byte

private def lowByte {n : Nat} (x : BitVec (byteWidth n + 8)) : BitVec 8 :=
  BitVec.extractLsb' 0 8 x

private def upperBytes {n : Nat} (x : BitVec (byteWidth n + 8)) : BitVec (byteWidth n) :=
  BitVec.extractLsb' 8 (byteWidth n) x

private theorem upper_low_decompose {n : Nat} (x : BitVec (byteWidth n + 8)) :
    upperBytes x ++ lowByte x = x := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_append]
  split
  · simp_all [lowByte, byteWidth]
  · have : i - 8 < byteWidth n := by omega
    simp_all [upperBytes, byteWidth]

private def mapBytes (f : BitVec 8 → BitVec 8) :
    (n : Nat) → BitVec (byteWidth n) → BitVec (byteWidth n)
  | 0, _ => 0#0
  | Nat.succ n, x => mapBytes f n (upperBytes x) ++ f (lowByte x)

/-- The byte map reads each source lane and writes exactly the same destination lane. -/
theorem mapBytes_lane (f : BitVec 8 → BitVec 8) (n i : Nat)
    (x : BitVec (byteWidth n)) (hi : i < n) :
    BitVec.extractLsb' (8 * i) 8 (mapBytes f n x) =
      f (BitVec.extractLsb' (8 * i) 8 x) := by
  induction n generalizing i with
  | zero => omega
  | succ n ih =>
      change BitVec (byteWidth n + 8) at x
      by_cases hzero : i = 0
      · subst i
        simp only [mapBytes, Nat.mul_zero, BitVec.extractLsb'_append_eq_right]
        rfl
      · have hprev : i - 1 < n := by omega
        have hbits : 8 * i = 8 + 8 * (i - 1) := by omega
        rw [hbits]
        simp only [mapBytes]
        have hleft (z : BitVec (byteWidth n)) (b : BitVec 8) :
            BitVec.extractLsb' (8 + 8 * (i - 1)) 8 (z ++ b) =
              BitVec.extractLsb' (8 * (i - 1)) 8 z := by
          apply BitVec.eq_of_getLsbD_eq
          intro j hj
          simp only [BitVec.getLsbD_extractLsb', hj, decide_true, Bool.true_and,
            BitVec.getLsbD_append]
          have hge : ¬ 8 + 8 * (i - 1) + j < 8 := by omega
          simp only [hge, ↓reduceIte]
          congr 1
          omega
        rw [hleft, ih (i - 1) (upperBytes x) hprev]
        have hx := hleft (upperBytes x) (lowByte x)
        rw [upper_low_decompose x] at hx
        exact congrArg f hx.symm

private def zeroMarkerByte (x : BitVec 8) : BitVec 8 :=
  if x = 0#8 then 128#8 else 0#8

/-- Repeated-byte zero detection is the independent composition of the eight-bit identity. -/
theorem zero_marker_decomposition (n : Nat) (x : BitVec (byteWidth n)) :
    (~~~(x ||| ((x ||| repeatByte 128#8 n) - repeatByte 1#8 n))) &&&
        repeatByte 128#8 n = mapBytes zeroMarkerByte n x := by
  induction n with
  | zero =>
      apply BitVec.eq_of_getLsbD_eq
      intro i hi
      simp [byteWidth] at hi
  | succ n ih =>
      change BitVec (byteWidth n + 8) at x
      rw [← upper_low_decompose x]
      simp only [repeatByte, append_or]
      rw [append_sub_byte_noBorrow]
      · rw [append_or, append_not, append_and, ih, byte_zero_marker]
        simp [mapBytes, zeroMarkerByte, upperBytes, lowByte]
        rfl
      · apply Nat.one_le_iff_ne_zero.mpr
        intro hzero
        have hor : (lowByte x ||| 128#8) = 0#8 :=
          BitVec.toNat_inj.mp (by simpa using hzero)
        have := (BitVec.or_eq_zero_iff.mp hor).2
        exact (by norm_num : (128#8).toNat ≠ 0) (congrArg BitVec.toNat this)

/-- The lane-wise marker word denoted by `zero_swar`. -/
def zeroMarkers (word : U64) : U64 := ⟨mapBytes zeroMarkerByte 8 word.bv⟩

/-- The extracted 64-bit zero marker is eight applications of `byte_zero_marker`. -/
theorem extracted_zero_swar_exact (word : U64) :
    table.scan.zero_swar word = ok (zeroMarkers word) := by
  cases word with
  | mk bv =>
      unfold table.scan.zero_swar zeroMarkers
      simp [lift, table.scan.HI, table.scan.LO, core.num.U64.wrapping_sub,
        UScalar.wrapping_sub]
      apply congrArg (fun bits : BitVec 64 => (⟨bits⟩ : U64))
      have hhi : repeatByte 128#8 8 = 9259542123273814144#64 := by rfl
      have hlo : repeatByte 1#8 8 = 72340172838076673#64 := by rfl
      simpa [hhi, hlo] using zero_marker_decomposition 8 bv

private def byteSeries : Nat → Nat
  | 0 => 0
  | n + 1 => byteSeries n * 256 + 1

private theorem repeatByte_toNat (n : Nat) (byte : BitVec 8) :
    (repeatByte byte n).toNat = byte.toNat * byteSeries n := by
  induction n with
  | zero => simp [repeatByte, byteSeries]
  | succ n ih =>
      rw [repeatByte, toNat_append_byte, ih]
      simp only [byteSeries]
      ring

/-- Multiplication by the extracted `LO` constant broadcasts one byte without overflow. -/
theorem lo_mul_broadcast (byte : U8) :
    core.num.U64.wrapping_mul table.scan.LO (UScalar.cast .U64 byte) =
      (⟨repeatByte byte.bv 8⟩ : U64) := by
  apply UScalar.eq_of_val_eq
  simp only [core.num.U64.wrapping_mul_val_eq, UScalar.cast_val_eq, table.scan.LO]
  simp only [UScalar.val, UScalar.size, UScalarTy.numBits]
  rw [repeatByte_toNat]
  norm_num [byteSeries]
  have hlo : (72340172838076673#(UScalarTy.U64.numBits)).toNat =
      72340172838076673 := by norm_num [UScalarTy.numBits]
  rw [hlo]
  have hbyte : byte.val < 256 := by
    simpa [UScalar.size, UScalarTy.numBits] using byte.hBounds
  omega

/-- The lane-wise equality markers denoted by `eq_swar`. -/
def tagMarkers (word : U64) (byte : U8) : U64 :=
  ⟨mapBytes zeroMarkerByte 8 (word.bv ^^^ repeatByte byte.bv 8)⟩

/-- The model's little-endian conversion exposes source byte `i` at word lane `i`. -/
theorem from_le_bytes_lane (bytes : Array U8 8#usize) (i : Nat) (hi : i < 8) :
    BitVec.extractLsb' (8 * i) 8 (core.num.U64.from_le_bytes bytes).bv =
      (bytes.val[i]'(by simpa using hi)).bv := by
  apply BitVec.eq_of_getLsbD_eq
  intro j hj
  have hbit : 8 * i + j < 64 := by omega
  have hindex : i < bytes.val.length := by simpa using hi
  simp only [BitVec.getLsbD_extractLsb', hj,
    BitVec.getLsbD_eq_getElem hbit]
  simp only [core.num.U64.from_le_bytes, BitVec.getLsbD_eq_getElem hj]
  rw [← BitVec.getElem!_eq_getElem (hi := hbit), BitVec.getElem!_cast,
    BitVec.fromLEBytes_getElem!]
  have hdiv : (8 * i + j) / 8 = i := by omega
  have hmod : (8 * i + j) % 8 = j := by omega
  simp only [hdiv, hmod]
  rw [← List.Inhabited_getElem_eq_getElem! (bytes.val.map U8.bv) i (by simpa using hindex)]
  simp only [decide_true, Bool.true_and, List.getElem_map, Byte.testBit,
    BitVec.getElem_eq_testBit_toNat]

/-- The end lanes of an asymmetric eight-byte input retain their original order. -/
theorem from_le_bytes_end_lanes :
    let bytes : Array U8 8#usize :=
      Array.from [1#u8, 2#u8, 3#u8, 4#u8, 5#u8, 6#u8, 7#u8, 8#u8] (by decide)
    BitVec.extractLsb' 0 8 (core.num.U64.from_le_bytes bytes).bv = 1#8 ∧
      BitVec.extractLsb' 56 8 (core.num.U64.from_le_bytes bytes).bv = 8#8 := by
  dsimp
  constructor
  · calc
      _ = (1#u8).bv := from_le_bytes_lane
        (Array.from [1#u8, 2#u8, 3#u8, 4#u8, 5#u8, 6#u8, 7#u8, 8#u8] (by decide)) 0 (by decide)
      _ = 1#8 := by decide
  · calc
      _ = (8#u8).bv := from_le_bytes_lane
        (Array.from [1#u8, 2#u8, 3#u8, 4#u8, 5#u8, 6#u8, 7#u8, 8#u8] (by decide)) 7 (by decide)
      _ = 8#8 := by decide

private theorem repeatByte_eq_mapBytes (byte : BitVec 8) (n : Nat)
    (x : BitVec (byteWidth n)) :
    repeatByte byte n = mapBytes (fun _ => byte) n x := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [repeatByte, mapBytes]
      rw [← ih (upperBytes x)]

private theorem repeatByte_lane (byte : BitVec 8) (n i : Nat) (hi : i < n) :
    BitVec.extractLsb' (8 * i) 8 (repeatByte byte n) = byte := by
  rw [repeatByte_eq_mapBytes byte n (0#(byteWidth n)), mapBytes_lane (hi := hi)]

/-- Equality marking reads precisely the original little-endian byte at each SWAR lane. -/
theorem tagMarkers_from_le_bytes_lane (bytes : Array U8 8#usize)
    (wanted : U8) (i : Nat) (hi : i < 8) :
    BitVec.extractLsb' (8 * i) 8
        (tagMarkers (core.num.U64.from_le_bytes bytes) wanted).bv =
      if bytes.val[i]'(by simpa using hi) = wanted then 128#8 else 0#8 := by
  simp only [tagMarkers, mapBytes_lane _ 8 i _ hi, BitVec.extractLsb'_xor,
    from_le_bytes_lane bytes i hi, repeatByte_lane wanted.bv 8 i hi,
    zeroMarkerByte]
  by_cases heq : bytes.val[i]'(by simpa using hi) = wanted
  · simp [heq]
  · have hne : (bytes.val[i]'(by simpa using hi)).bv ≠ wanted.bv := by
      intro h
      apply heq
      exact UScalar.eq_of_val_eq (BitVec.toNat_inj.mpr h)
    simp [heq, hne, BitVec.xor_eq_zero_iff]

/-- The extracted equality marker is byte broadcast, xor, then independent zero detection. -/
theorem extracted_eq_swar_exact (word : U64) (byte : U8) :
    table.scan.eq_swar word byte = ok (tagMarkers word byte) := by
  unfold table.scan.eq_swar tagMarkers
  have hbroadcast := congrArg UScalar.bv (lo_mul_broadcast byte)
  simp [lift, table.scan.HI, table.scan.LO,
    core.num.U64.wrapping_sub, UScalar.wrapping_sub]
  apply congrArg (fun bits : BitVec 64 => (⟨bits⟩ : U64))
  simp only [core.num.U64.wrapping_mul, UScalar.wrapping_mul, UScalar.cast] at hbroadcast
  have hbroadcast' :
      72340172838076673#64 * BitVec.setWidth 64 byte.bv = repeatByte byte.bv 8 := by
    simpa [table.scan.LO] using hbroadcast
  rw [hbroadcast']
  have hhi : repeatByte 128#8 8 = 9259542123273814144#64 := by rfl
  have hlo : repeatByte 1#8 8 = 72340172838076673#64 := by rfl
  simpa [hhi, hlo] using
    zero_marker_decomposition 8 (word.bv ^^^ repeatByte byte.bv 8)

private def markerNat (mask : BitVec 8) : Nat :=
  (mask.toNat % 2) * 128 +
  (mask.toNat / 2 % 2) * 32768 +
  (mask.toNat / 4 % 2) * 8388608 +
  (mask.toNat / 8 % 2) * 2147483648 +
  (mask.toNat / 16 % 2) * 549755813888 +
  (mask.toNat / 32 % 2) * 140737488355328 +
  (mask.toNat / 64 % 2) * 36028797018963968 +
  (mask.toNat / 128 % 2) * 9223372036854775808

/-- Spread the eight mask bits to bits `7 + 8i`, the marker position of lane `i`. -/
def markerWord (mask : BitVec 8) : U64 := ⟨BitVec.ofNat 64 (markerNat mask)⟩

/-- Each bit of the eight-bit mask contributes one high bit to the corresponding word byte. -/
private theorem markerWord_lane (mask : BitVec 8) (i : Nat) (hi : i < 8) :
    BitVec.extractLsb' (8 * i) 8 (markerWord mask).bv =
      if mask[i] then 128#8 else 0#8 := by
  rcases mask with ⟨⟨value, hvalue⟩⟩
  interval_cases value <;> interval_cases i <;> decide +revert

/-- Eight equal byte projections determine a 64-bit word, including non-marker bits. -/
private theorem word_eq_of_lanes (a b : BitVec 64)
    (h : ∀ i : Nat, i < 8 →
      BitVec.extractLsb' (8 * i) 8 a = BitVec.extractLsb' (8 * i) 8 b) :
    a = b := by
  apply BitVec.eq_of_getLsbD_eq
  intro bit hbit
  have hlane : bit / 8 < 8 := by omega
  have hslot : bit % 8 < 8 := Nat.mod_lt _ (by omega)
  have hposition : 8 * (bit / 8) + bit % 8 = bit := by omega
  have heq := congrArg (fun lane : BitVec 8 => lane.getLsbD (bit % 8))
    (h (bit / 8) hlane)
  simpa only [BitVec.getLsbD_extractLsb', hslot, decide_true, Bool.true_and,
    hposition] using heq

/-- The complete equality word retains exactly the high bits of matching source bytes. -/
theorem tagMarkers_from_le_bytes_word (bytes : Array U8 8#usize) (wanted : U8) :
    tagMarkers (core.num.U64.from_le_bytes bytes) wanted =
      markerWord (BitVec.ofFn fun i => decide (bytes.val[i.val] = wanted)) := by
  apply UScalar.eq_of_val_eq
  apply congrArg BitVec.toNat
  apply word_eq_of_lanes
  intro i hi
  rw [tagMarkers_from_le_bytes_lane bytes wanted i hi,
    markerWord_lane _ i hi, BitVec.getElem_ofFn]
  simp

/-- Asymmetric source bytes place a present tag at lane two and no absent tag anywhere. -/
theorem tagMarkers_asymmetric_witness :
    let bytes : Array U8 8#usize :=
      Array.from [1#u8, 2#u8, 3#u8, 4#u8, 5#u8, 6#u8, 7#u8, 8#u8] (by decide)
    tagMarkers (core.num.U64.from_le_bytes bytes) 3#u8 = ⟨8388608#64⟩ ∧
      tagMarkers (core.num.U64.from_le_bytes bytes) 9#u8 = ⟨0#64⟩ := by
  dsimp
  constructor
  · rw [tagMarkers_from_le_bytes_word]
    decide +revert
  · rw [tagMarkers_from_le_bytes_word]
    decide +revert

private theorem eight_bits (x : Nat) (h : x < 256) :
    x % 2 + 2 * (x / 2 % 2) + 4 * (x / 4 % 2) + 8 * (x / 8 % 2) +
      16 * (x / 16 % 2) + 32 * (x / 32 % 2) + 64 * (x / 64 % 2) +
        128 * (x / 128 % 2) = x := by
  omega

private theorem gather_low_four (d₀ d₁ d₂ d₃ : Nat)
    (h₀ : d₀ < 2) (h₁ : d₁ < 2) (h₂ : d₂ < 2) (h₃ : d₃ < 2) :
    ((d₀ * 128 + d₁ * 32768 + d₂ * 8388608 + d₃ * 2147483648) *
        567382630219905) % 18446744073709551616 =
      72057594037927936 * (d₀ + 2 * d₁ + 4 * d₂ + 8 * d₃) +
        (d₀ * 567382630219904 + d₁ * 1134765260439552 +
          d₂ * 2269530520813568 + d₃ * 4539061024849920) := by
  omega

private theorem gather_high_four (d₄ d₅ d₆ d₇ : Nat)
    (h₄ : d₄ < 2) (h₅ : d₅ < 2) (h₆ : d₆ < 2) (h₇ : d₇ < 2) :
    ((d₄ * 549755813888 + d₅ * 140737488355328 + d₆ * 36028797018963968 +
        d₇ * 9223372036854775808) * 567382630219905) % 18446744073709551616 =
      72057594037927936 * (16 * d₄ + 32 * d₅ + 64 * d₆ + 128 * d₇) +
        (d₄ * 9078117754732544 + d₅ * 18155135997837312 +
          d₆ * 36028797018963968) := by
  omega

private theorem gather_eight_digits (d₀ d₁ d₂ d₃ d₄ d₅ d₆ d₇ : Nat)
    (h₀ : d₀ < 2) (h₁ : d₁ < 2) (h₂ : d₂ < 2) (h₃ : d₃ < 2)
    (h₄ : d₄ < 2) (h₅ : d₅ < 2) (h₆ : d₆ < 2) (h₇ : d₇ < 2) :
    ((d₀ * 128 + d₁ * 32768 + d₂ * 8388608 + d₃ * 2147483648 +
        d₄ * 549755813888 + d₅ * 140737488355328 + d₆ * 36028797018963968 +
          d₇ * 9223372036854775808) * 567382630219905 % 18446744073709551616) /
      72057594037927936 =
        d₀ + 2 * d₁ + 4 * d₂ + 8 * d₃ + 16 * d₄ + 32 * d₅ + 64 * d₆ + 128 * d₇ := by
  let lowMarker := d₀ * 128 + d₁ * 32768 + d₂ * 8388608 + d₃ * 2147483648
  let highMarker := d₄ * 549755813888 + d₅ * 140737488355328 +
    d₆ * 36028797018963968 + d₇ * 9223372036854775808
  let lowBits := d₀ + 2 * d₁ + 4 * d₂ + 8 * d₃
  let highBits := 16 * d₄ + 32 * d₅ + 64 * d₆ + 128 * d₇
  let lowRemainder := d₀ * 567382630219904 + d₁ * 1134765260439552 +
    d₂ * 2269530520813568 + d₃ * 4539061024849920
  let highRemainder := d₄ * 9078117754732544 + d₅ * 18155135997837312 +
    d₆ * 36028797018963968
  have hlo := gather_low_four d₀ d₁ d₂ d₃ h₀ h₁ h₂ h₃
  have hhi := gather_high_four d₄ d₅ d₆ d₇ h₄ h₅ h₆ h₇
  have hmarker :
      d₀ * 128 + d₁ * 32768 + d₂ * 8388608 + d₃ * 2147483648 +
          d₄ * 549755813888 + d₅ * 140737488355328 + d₆ * 36028797018963968 +
            d₇ * 9223372036854775808 = lowMarker + highMarker := by
    dsimp only [lowMarker, highMarker]
    omega
  have hbits :
      d₀ + 2 * d₁ + 4 * d₂ + 8 * d₃ + 16 * d₄ + 32 * d₅ + 64 * d₆ + 128 * d₇ =
        lowBits + highBits := by
    dsimp only [lowBits, highBits]
    omega
  rw [hmarker, hbits]
  have hremainder : lowRemainder + highRemainder < 72057594037927936 := by
    dsimp only [lowRemainder, highRemainder]
    omega
  have htotal :
      72057594037927936 * lowBits + lowRemainder +
          (72057594037927936 * highBits + highRemainder) <
        18446744073709551616 := by
    dsimp only [lowBits, highBits, lowRemainder, highRemainder]
    omega
  rw [Nat.add_mul, Nat.add_mod, hlo, hhi, Nat.mod_eq_of_lt htotal]
  have hsplit :
      72057594037927936 * lowBits + lowRemainder +
          (72057594037927936 * highBits + highRemainder) =
        72057594037927936 * (lowBits + highBits) +
          (lowRemainder + highRemainder) := by omega
  rw [hsplit, Nat.mul_add_div (by norm_num), Nat.div_eq_of_lt hremainder, Nat.add_zero]

private theorem gather_markerNat (mask : BitVec 8) :
    markerNat mask * 567382630219905 % 18446744073709551616 /
        72057594037927936 = mask.toNat := by
  have hmask : mask.toNat < 256 := by simpa using mask.isLt
  rw [← eight_bits mask.toNat hmask]
  unfold markerNat
  apply gather_eight_digits <;> omega

private theorem bind_two_ok {α β γ : Type} (x : α) (f : α → β) (g : β → γ) :
    (do let y ← ok x; let z ← ok (f y); ok (g z)) = ok (g (f x)) := by
  simp

private theorem lowEightOfShift (word : BitVec 64) (mask : BitVec 8)
    (h : (word.toNat >>> 56) % 256 = mask.toNat) :
    BitVec.setWidth 8 (word.ushiftRight 56) = mask := by
  apply BitVec.toNat_inj.mp
  rw [BitVec.toNat_setWidth, BitVec.ushiftRight_eq, BitVec.toNat_ushiftRight]
  norm_num only [pow_succ, pow_zero, Nat.mul_one]
  exact h

private theorem gather_markerBits (mask : BitVec 8) :
    BitVec.setWidth 8
        ((BitVec.ofNat 64 (markerNat mask) * 567382630219905#64).ushiftRight 56) = mask := by
  apply lowEightOfShift
  have hmask : mask.toNat < 256 := by simpa using mask.isLt
  have hgather : 567382630219905 % 18446744073709551616 = 567382630219905 := by
    norm_num
  calc
    ((BitVec.ofNat 64 (markerNat mask) * 567382630219905#64).toNat >>> 56) % 256 =
        (markerNat mask * 567382630219905 % 18446744073709551616 /
          72057594037927936) % 256 := by
            rw [BitVec.toNat_mul, BitVec.toNat_ofNat, BitVec.toNat_ofNat]
            norm_num only [pow_succ, pow_zero, Nat.mul_one]
            rw [← hgather, ← Nat.mul_mod, Nat.shiftRight_eq_div_pow]
    _ = mask.toNat % 256 := by rw [gather_markerNat]
    _ = mask.toNat := Nat.mod_eq_of_lt hmask

/-- The extracted multiplier gather is inverse to spreading on the lane-marker subspace. -/
theorem gather_spreadMarkerBits (mask : BitVec 8) :
    table.scan.gather (markerWord mask) = ok (⟨mask⟩ : U8) := by
  unfold table.scan.gather markerWord
  have hval : (56#i32).val = 56 := by rfl
  have hnat : (56#i32).toNat = 56 := by rfl
  simp only [lift, core.num.U64.wrapping_mul, UScalar.wrapping_mul, table.scan.GATHER,
    HShiftRight.hShiftRight, UScalar.shiftRight_IScalar, hval, hnat, UScalar.shiftRight]
  norm_num only [UScalarTy.numBits, ↓reduceIte]
  simp only [if_true]
  rw [bind_two_ok]
  exact congrArg ok (congrArg (fun bits : BitVec 8 => (⟨bits⟩ : U8)) (gather_markerBits mask))

/-- Exactness facts for the three extracted scalar scan primitives. -/
structure ScanExactness : Prop where
  zero : ∀ word, table.scan.zero_swar word = ok (zeroMarkers word)
  eq : ∀ word byte, table.scan.eq_swar word byte = ok (tagMarkers word byte)
  gather : ∀ mask, table.scan.gather (markerWord mask) = ok (⟨mask⟩ : U8)

/-- The extracted scalar scan marks bytes independently and gathers every marker exactly. -/
theorem scan_exact : ScanExactness :=
  ⟨extracted_zero_swar_exact, extracted_eq_swar_exact, gather_spreadMarkerBits⟩

/-- The generated eight-byte path reads and compresses exactly the equality bits in its
    source window. This concerns one window, not either traversal loop or its mask shift. -/
theorem match_tag_window_exact (line : Slice U8) (wanted : U8) (off : Usize)
    (hwindow : off.val + 8 ≤ line.val.length) :
    ∃ bytes : Array U8 8#usize,
      bytes.val = line.val.slice off.val (off.val + 8) ∧
      (do
        let word ← table.scan.word_at line off
        let marked ← table.scan.eq_swar word wanted
        table.scan.gather marked) =
        ok (⟨BitVec.ofFn fun i => decide (bytes.val[i.val] = wanted)⟩ : U8) := by
  obtain ⟨bytes, hbytes, hword⟩ := word_at_exact line off hwindow
  refine ⟨bytes, hbytes, ?_⟩
  rw [hword, bind_tc_ok, extracted_eq_swar_exact, bind_tc_ok,
    tagMarkers_from_le_bytes_word, gather_spreadMarkerBits]
  rfl

/-- Equality bits accumulated strictly before the scan cursor, in their source positions. -/
private def tagPrefix (line : Slice U8) (wanted : U8) (cursor : Nat) : BitVec 64 :=
  BitVec.ofFn fun i => decide (i.val < cursor ∧ line.val[i.val]? = some wanted)

private theorem tagPrefix_zero (line : Slice U8) (wanted : U8) :
    tagPrefix line wanted 0 = 0#64 := by
  ext i hi
  simp [tagPrefix, BitVec.getElem_ofFn]

private theorem tagPrefix_step (line : Slice U8) (wanted : U8) (cursor : Nat)
    (hcursor : cursor < line.val.length) (_hword : cursor < 64) :
    tagPrefix line wanted (cursor + 1) =
      tagPrefix line wanted cursor |||
        (if line.val[cursor] = wanted then (1#64 <<< cursor) else 0#64) := by
  ext i hi
  by_cases heq : line.val[cursor] = wanted
  · by_cases hindex : i = cursor
    · subst i
      simp [tagPrefix, heq, hcursor, BitVec.getElem_or,
        BitVec.getElem_shiftLeft, BitVec.getElem_one]
    · by_cases hlt : i < cursor
      · have hle : i ≤ cursor := by omega
        simp [tagPrefix, heq, hlt, hle, BitVec.getElem_or,
          BitVec.getElem_shiftLeft, BitVec.getElem_one]
      · have hge : cursor + 1 ≤ i := by omega
        have hnle : ¬ i ≤ cursor := by omega
        have hsub : i - cursor ≠ 0 := by omega
        simp [tagPrefix, heq, hlt, hnle, hsub, BitVec.getElem_or,
          BitVec.getElem_shiftLeft, BitVec.getElem_one]
  · by_cases hindex : i = cursor
    · subst i
      simp [tagPrefix, heq, hcursor]
    · by_cases hlt : i < cursor
      · have hle : i ≤ cursor := by omega
        simp [tagPrefix, heq, hlt, hle]
      · have hnle : ¬ i ≤ cursor := by omega
        simp [tagPrefix, heq, hlt, hnle]

private theorem tagPrefix_word (line : Slice U8) (wanted : U8) (cursor : Nat)
    (hwindow : cursor + 8 ≤ line.val.length) (_hbound : line.val.length ≤ 64)
    (bytes : Array U8 8#usize)
    (hbytes : bytes.val = line.val.slice cursor (cursor + 8)) :
    tagPrefix line wanted (cursor + 8) =
      tagPrefix line wanted cursor |||
        ((BitVec.ofFn fun i : Fin 8 => decide (bytes.val[i.val] = wanted)).setWidth 64 <<< cursor) := by
  ext i hi
  by_cases hbefore : i < cursor
  · have hbeforeEnd : i < cursor + 8 := by omega
    simp [tagPrefix, hbefore, hbeforeEnd, BitVec.getElem_or, BitVec.getElem_shiftLeft]
  · by_cases hinside : i < cursor + 8
    · have hlocal : i - cursor < 8 := by omega
      have hslice : bytes.val[i - cursor]? = line.val[i]? := by
        rw [hbytes, List.getElem?_slice cursor (cursor + 8) (i - cursor) line.val (by omega)]
        congr 1
        omega
      have hl : i - cursor < bytes.val.length := by
        rw [Array.length_eq]
        exact hlocal
      have hr : i < line.val.length := by omega
      have hbyte : bytes.val[i - cursor] = line.val[i] := by
        simpa [List.getElem?_eq_getElem hl, List.getElem?_eq_getElem hr] using hslice
      have hshift : i - cursor < 8 := hlocal
      simp [tagPrefix, hbefore, hinside, BitVec.getElem_or,
        BitVec.getElem_shiftLeft, hshift, hbyte, List.getElem?_eq_getElem hr]
    · have hafter : cursor + 8 ≤ i := by omega
      have hlocal : ¬ i - cursor < 8 := by omega
      have hz : (BitVec.ofFn fun j : Fin 8 => decide (bytes.val[j.val] = wanted)).getLsbD
          (i - cursor) = false := by
        exact BitVec.getLsbD_of_ge _ _ (by omega)
      simp [tagPrefix, hbefore, BitVec.getElem_or,
        BitVec.getElem_shiftLeft, hz, hafter]

private def TagScanInv (line : Slice U8) (wanted : U8)
    (state : U64 × Usize × U32) : Prop :=
  state.2.1.val = state.2.2.val ∧ state.2.1.val ≤ line.val.length ∧
    state.1.bv = tagPrefix line wanted state.2.1.val

private theorem match_tag_words (line : Slice U8) (wanted : U8)
    (hlen : line.val.length ≤ 64) :
    table.scan.match_tag_loop0 line wanted 0#u64 0#usize 0#u32 ⦃ state =>
      TagScanInv line wanted state ⦄ := by
  unfold table.scan.match_tag_loop0
  apply loop.spec_decr_nat
    (measure := fun state : U64 × Usize × U32 => 64 - state.2.1.val)
    (inv := TagScanInv line wanted)
    (post := TagScanInv line wanted)
  · intro state hstate
    rcases state with ⟨mask, off, base⟩
    rcases hstate with ⟨hbase, hoff, hmask⟩
    dsimp only at hbase hoff hmask ⊢
    unfold table.scan.match_tag_loop0.body table.scan.LANES_PER_WORD
    have hnextCap : off.val + 8 ≤ Usize.max := by
      have hmax : 64 ≤ Usize.max := by
        rcases Usize.bounds_eq with h | h <;> rw [h] <;>
          norm_num [U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
      have hmax' : 256 ≤ Usize.max := by
        rcases Usize.bounds_eq with h | h <;> rw [h] <;>
          norm_num [U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
      omega
    step as ⟨next, hnext⟩
    by_cases hwindow : off.val + 8 ≤ line.val.length
    · have hle : next ≤ Slice.len line := by
        apply (UScalar.le_equiv _ _).mpr
        simpa [Slice.len_val, hnext] using hwindow
      rw [if_pos hle]
      have hbaseCap : base.val + 8 ≤ U32.max := by
        rw [U32.max_eq]
        omega
      step as ⟨nextBase, hnextBase⟩
      have hbaseLe : nextBase ≤ 64#u32 := by
        apply (UScalar.le_equiv _ _).mpr
        have h64 : (64#u32).val = 64 := by rfl
        rw [hnextBase, h64]
        omega
      rw [if_pos hbaseLe]
      obtain ⟨bytes, hbytes, hword⟩ := word_at_exact line off hwindow
      rw [hword, bind_tc_ok, extracted_eq_swar_exact, bind_tc_ok,
        tagMarkers_from_le_bytes_word, gather_spreadMarkerBits, bind_tc_ok]
      simp only [lift, bind_tc_ok]
      step as ⟨shifted, hshifted, hshiftbv⟩
      refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
      · change next.val = nextBase.val
        omega
      · change next.val ≤ line.val.length
        omega
      · change (mask ||| shifted).bv = tagPrefix line wanted next.val
        calc
          (mask ||| shifted).bv = mask.bv ||| shifted.bv := rfl
          _ = tagPrefix line wanted off.val |||
              ((BitVec.ofFn fun i : Fin 8 => decide (bytes.val[i.val] = wanted)).setWidth 64 <<<
                off.val) := by
                  simp only [hmask, hshiftbv, core.convert.num.FromU64U8.from_bv_eq,
                    hbase]
          _ = tagPrefix line wanted (off.val + 8) :=
            (tagPrefix_word line wanted off.val hwindow hlen bytes hbytes).symm
          _ = tagPrefix line wanted next.val := by rw [hnext]
      · omega
    · have hnot : ¬ next ≤ Slice.len line := by
        intro hle
        have := (UScalar.le_equiv _ _).mp hle
        simp [hnext] at this
        omega
      rw [if_neg hnot, WP.spec_ok]
      exact ⟨hbase, hoff, hmask⟩
  · simp [TagScanInv, tagPrefix_zero]

private theorem match_tag_tail (line : Slice U8) (wanted : U8)
    (mask : U64) (off : Usize) (base : U32)
    (hlen : line.val.length ≤ 64)
    (hinv : TagScanInv line wanted (mask, off, base)) :
    table.scan.match_tag_loop1 line wanted mask off base ⦃ bits =>
      bits.bv = tagPrefix line wanted line.val.length ⦄ := by
  unfold table.scan.match_tag_loop1
  apply loop.spec_decr_nat
    (measure := fun state : U64 × Usize × U32 => line.val.length - state.2.1.val)
    (inv := TagScanInv line wanted)
    (post := fun bits : U64 => bits.bv = tagPrefix line wanted line.val.length)
  · intro state hstate
    rcases state with ⟨current, cursor, position⟩
    rcases hstate with ⟨hposition, hcursor, hmask⟩
    dsimp only at hposition hcursor hmask ⊢
    unfold table.scan.match_tag_loop1.body
    by_cases hmore : cursor.val < line.val.length
    · have hlt : cursor < Slice.len line :=
        (UScalar.lt_equiv _ _).mpr (by simpa [Slice.len_val] using hmore)
      rw [if_pos hlt]
      have hpos : position < 64#u32 :=
        (UScalar.lt_equiv _ _).mpr (by norm_num; omega)
      rw [if_pos hpos]
      obtain ⟨byte, hread, hbyte⟩ :=
        WP.spec_imp_exists (Slice.index_usize_spec line cursor hmore)
      rw [hread]
      simp only [bind_tc_ok]
      split
      · rename_i heq
        have heqLine : line.val[cursor.val] = wanted := hbyte.symm.trans heq
        step as ⟨single, hsingle, hsinglebv⟩
        have hcursorCap : cursor.val + 1 ≤ Usize.max := by
          have hmax : 64 ≤ Usize.max := by
            rcases Usize.bounds_eq with h | h <;> rw [h] <;>
              norm_num [U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
          omega
        step as ⟨next, hnext⟩
        have hposCap : position.val + 1 ≤ U32.max := by
          rw [U32.max_eq]
          omega
        step as ⟨nextPosition, hnextPosition⟩
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · change next.val = nextPosition.val
          omega
        · change next.val ≤ line.val.length
          omega
        · change (current ||| single).bv = tagPrefix line wanted next.val
          calc
            (current ||| single).bv = current.bv ||| single.bv := rfl
            _ = tagPrefix line wanted cursor.val ||| (1#64 <<< cursor.val) := by
              simp only [hmask, hsinglebv, ← hposition]
              rfl
            _ = tagPrefix line wanted (cursor.val + 1) := by
              have hstep := tagPrefix_step line wanted cursor.val hmore (by omega)
              split at hstep
              · exact hstep.symm
              · rename_i hfalse
                exact False.elim (hfalse heqLine)
            _ = tagPrefix line wanted next.val := by rw [hnext]
        · omega
      · rename_i heq
        have heqLine : line.val[cursor.val] ≠ wanted := by
          intro h
          exact heq (hbyte.trans h)
        have hcursorCap : cursor.val + 1 ≤ Usize.max := by
          have hmax : 64 ≤ Usize.max := by
            rcases Usize.bounds_eq with h | h <;> rw [h] <;>
              norm_num [U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
          omega
        step as ⟨next, hnext⟩
        have hposCap : position.val + 1 ≤ U32.max := by
          rw [U32.max_eq]
          omega
        step as ⟨nextPosition, hnextPosition⟩
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · change next.val = nextPosition.val
          omega
        · change next.val ≤ line.val.length
          omega
        · change current.bv = tagPrefix line wanted next.val
          rw [hmask, hnext]
          have hstep := tagPrefix_step line wanted cursor.val hmore (by omega)
          split at hstep
          · rename_i htrue
            exact False.elim (heqLine htrue)
          · simpa using hstep.symm
        · omega
    · have hstop : ¬ cursor < Slice.len line := by
        intro hlt
        exact hmore ((UScalar.lt_equiv _ _).mp hlt)
      rw [if_neg hstop, WP.spec_ok]
      have hdone : cursor.val = line.val.length := by omega
      simpa only [hdone] using hmask
  · exact hinv

/-- The generated word loop and scalar tail mark precisely the matching byte positions.
    The admitted scan geometry has at most 64 bytes, one per output-mask bit. -/
theorem match_tag_exact (line : Slice U8) (wanted : U8)
    (hlen : line.val.length ≤ 64) :
    ∃ bits : U64, table.scan.match_tag line wanted = ok bits ∧
      ∀ i : Nat, bits.bv.getLsbD i =
        decide (i < line.val.length ∧ line.val[i]? = some wanted) := by
  have hspec : table.scan.match_tag line wanted ⦃ bits =>
      bits.bv = tagPrefix line wanted line.val.length ⦄ := by
    obtain ⟨state, hloop, hinv⟩ :=
      WP.spec_imp_exists (match_tag_words line wanted hlen)
    rcases state with ⟨mask, off, base⟩
    unfold table.scan.match_tag
    rw [hloop]
    have hreduce :
        (do let (m, o, b) ← (ok (mask, off, base) : Result (U64 × Usize × U32))
            table.scan.match_tag_loop1 line wanted m o b) =
          table.scan.match_tag_loop1 line wanted mask off base := by
      simp only [bind_tc_ok]
      rfl
    rw [hreduce]
    exact match_tag_tail line wanted mask off base hlen hinv
  obtain ⟨bits, hbits, hexact⟩ := WP.spec_imp_exists hspec
  refine ⟨bits, hbits, ?_⟩
  intro i
  rw [hexact]
  by_cases hi64 : i < 64
  · rw [BitVec.getLsbD_eq_getElem hi64]
    simp only [tagPrefix, BitVec.getElem_ofFn]
  · have hout : ¬ i < line.val.length := by omega
    have hz : (tagPrefix line wanted line.val.length).getLsbD i = false :=
      BitVec.getLsbD_of_ge _ _ (by omega)
    rw [hz]
    simp [hout]

/-- A nonzero hit word's generated clear expression succeeds and strictly decreases its
    numeric value. This fact alone does not identify the removed lane. -/
theorem generated_clear_hits_strict (hits : U64) (hne : hits ≠ 0#u64) :
    (do
      let previous ← hits - 1#u64
      lift (hits &&& previous)) ⦃ next => next.val < hits.val ⦄ := by
  have hpositive : 1 ≤ hits.val := by
    have hnot : hits.val ≠ 0 := by
      intro hz
      apply hne
      exact UScalar.eq_of_val_eq (by simpa using hz)
    omega
  step as ⟨previous, hprevious⟩
  simp only [lift, WP.spec_ok]
  rw [UScalar.val_and]
  have hle : hits.val &&& previous.val ≤ previous.val := Nat.and_le_right
  omega

/-- The halving state shifts away only empty low lanes; the low candidate window contains
at least one set bit. Higher bits need not be zero. -/
private def LaneHalvingInv (bits rest : U64) (lane : Usize) (width : Nat) : Prop :=
  bits.val = rest.val * 2 ^ lane.val ∧
  rest.val % 2 ^ width ≠ 0 ∧ lane.val + width ≤ 64

/-- One source-level halving stage: a zero low half shifts the candidate window; a nonzero
low half retains it. The numeric mask and signed shift are supplied by the generated stage. -/
private theorem lane_halving_stage (bits rest mask : U64) (lane widthToken : Usize)
    (shift : I32) (width : Nat)
    (hwidth : 0 < width) (hwidthBound : width ≤ 32)
    (hmask : mask.val = 2 ^ width - 1)
    (hshift : shift.val = width)
    (hwidthVal : widthToken.val = width)
    (hinv : LaneHalvingInv bits rest lane (2 * width)) :
    (do
      let low ← lift (rest &&& mask)
      if low = 0#u64 then
        do
          let rest' ← rest >>> shift
          let lane' ← lane + widthToken
          ok (lane', rest')
      else ok (lane, rest)) ⦃ out =>
        LaneHalvingInv bits out.2 out.1 width ⦄ := by
  rcases hinv with ⟨hbits, hcandidate, hlane⟩
  have hwidthWord : width < 64 := by omega
  have hmaskNat : (rest &&& mask).val = rest.val % 2 ^ width := by
    rw [UScalar.val_and, hmask, Nat.and_two_pow_sub_one_eq_mod]
  simp only [lift, bind_tc_ok]
  by_cases hlow : rest &&& mask = 0#u64
  · rw [if_pos hlow]
    have hmod : rest.val % 2 ^ width = 0 := by
      rw [← hmaskNat, hlow]
      rfl
    have hsum : lane.val + width ≤ Usize.max := by
      have hmax : 64 ≤ Usize.max := by
        rcases Usize.bounds_eq with h | h <;> rw [h] <;>
          norm_num [U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
      omega
    step as ⟨rest', hrest'⟩
    step as ⟨lane', hlane'⟩
    have hrestVal : rest'.val = rest.val / 2 ^ width := by
      simpa only [hshift, Int.toNat_natCast, Nat.shiftRight_eq_div_pow] using hrest'
    have hfactor : rest.val = (rest.val / 2 ^ width) * 2 ^ width := by
      simpa [hmod, Nat.mul_comm] using (Nat.mod_add_div rest.val (2 ^ width)).symm
    refine ⟨?_, ?_, ?_⟩
    · rw [hlane', hwidthVal, hrestVal]
      calc
        bits.val = rest.val * 2 ^ lane.val := hbits
        _ = (rest.val / 2 ^ width * 2 ^ width) * 2 ^ lane.val :=
          congrArg (fun v => v * 2 ^ lane.val) hfactor
        _ = rest.val / 2 ^ width * 2 ^ (lane.val + width) := by
          rw [pow_add]; ring
    · rw [hrestVal]
      have hpow : 2 ^ (2 * width) = 2 ^ width * 2 ^ width := by ring
      rw [hpow] at hcandidate
      exact fun hzero => hcandidate (by
        rw [hfactor, Nat.mul_mod_mul_right]
        simp [hzero])
    · rw [hlane', hwidthVal]
      omega
  · rw [if_neg hlow, WP.spec_ok]
    have hmod : rest.val % 2 ^ width ≠ 0 := by
      intro hz
      apply hlow
      apply UScalar.eq_of_val_eq
      rw [hmaskNat]
      exact hz
    change LaneHalvingInv bits rest lane width
    exact ⟨hbits, hmod, by omega⟩

private theorem lane_halving_branch (bits rest mask : U64) (lane widthToken : Usize)
    (shift : I32) (width : Nat)
    (hwidth : 0 < width) (hwidthBound : width ≤ 32)
    (hmask : mask.val = 2 ^ width - 1)
    (hshift : shift.val = width)
    (hwidthVal : widthToken.val = width)
    (hinv : LaneHalvingInv bits rest lane (2 * width)) :
    (if rest &&& mask = 0#u64 then
      do
        let rest' ← rest >>> shift
        let lane' ← lane + widthToken
        ok (lane', rest')
    else ok (lane, rest)) ⦃ out => LaneHalvingInv bits out.2 out.1 width ⦄ := by
  simpa only [lift, bind_tc_ok] using
    lane_halving_stage bits rest mask lane widthToken shift width hwidth hwidthBound hmask hshift hwidthVal hinv

private theorem lane_halving_branch32 (bits : U64)
    (hinv : LaneHalvingInv bits bits 0#usize 64) :
    (if bits &&& 4294967295#u64 = 0#u64 then
      do
        let rest ← bits >>> 32#i32
        let lane ← 0#usize + 32#usize
        ok (lane, rest)
    else ok (0#usize, bits)) ⦃ out => LaneHalvingInv bits out.2 out.1 32 ⦄ := by
  exact lane_halving_branch bits bits 4294967295#u64 0#usize 32#usize 32#i32 32
    (by omega) (by omega) (by norm_num) (by norm_num) (by norm_num) hinv

/-- The generated six-stage scan returns the first set bit on a nonzero input. -/
theorem least_set_lane_selects (bits : U64) (hne : bits ≠ 0#u64) :
    table.scan.least_set_lane bits ⦃ lane =>
      lane.val < 64 ∧ 2 ^ lane.val ∣ bits.val ∧ ¬ 2 ^ (lane.val + 1) ∣ bits.val ⦄ := by
  unfold table.scan.least_set_lane
  rw [if_neg hne]
  have hinit : LaneHalvingInv bits bits 0#usize 64 := by
    have hp : bits.val ≠ 0 := by
      intro hz
      apply hne
      exact UScalar.eq_of_val_eq hz
    refine ⟨by simp, ?_, by norm_num⟩
    have hlt : bits.val < 2 ^ 64 := bits.bv.isLt
    simpa only [Nat.mod_eq_of_lt hlt] using hp
  step
  have hi : i = bits &&& 4294967295#u64 := UScalar.eq_of_val_eq i_post
  rw [hi]
  step with lane_halving_branch32 bits hinit as ⟨lane0, rest0, hstate0⟩
  step
  have hi1 : i1 = rest0 &&& 65535#u64 := UScalar.eq_of_val_eq i1_post
  rw [hi1]
  step with lane_halving_branch bits rest0 65535#u64 lane0 16#usize 16#i32 16
    (by omega) (by omega) (by norm_num) (by norm_num) (by norm_num) hstate0
    as ⟨lane1, rest1, hstate1⟩
  step
  have hi2 : i2 = rest1 &&& 255#u64 := UScalar.eq_of_val_eq i2_post
  rw [hi2]
  step with lane_halving_branch bits rest1 255#u64 lane1 8#usize 8#i32 8
    (by omega) (by omega) (by norm_num) (by norm_num) (by norm_num) hstate1
    as ⟨lane2, rest2, hstate2⟩
  step
  have hi3 : i3 = rest2 &&& 15#u64 := UScalar.eq_of_val_eq i3_post
  rw [hi3]
  step with lane_halving_branch bits rest2 15#u64 lane2 4#usize 4#i32 4
    (by omega) (by omega) (by norm_num) (by norm_num) (by norm_num) hstate2
    as ⟨lane3, rest3, hstate3⟩
  step
  have hi4 : i4 = rest3 &&& 3#u64 := UScalar.eq_of_val_eq i4_post
  rw [hi4]
  step with lane_halving_branch bits rest3 3#u64 lane3 2#usize 2#i32 2
    (by omega) (by omega) (by norm_num) (by norm_num) (by norm_num) hstate3
    as ⟨lane4, rest4, hstate4⟩
  step
  have hi5 : i5 = rest4 &&& 1#u64 := UScalar.eq_of_val_eq i5_post
  rw [hi5]
  obtain ⟨hbits, hcandidate, hbound⟩ := hstate4
  have hone : (rest4 &&& 1#u64).val = rest4.val % 2 := by
    rw [UScalar.val_and]
    norm_num [Nat.and_one_is_mod]
  by_cases hlow : rest4 &&& 1#u64 = 0#u64
  · rw [if_pos hlow]
    have heven : rest4.val % 2 = 0 := by rw [← hone, hlow]; rfl
    have htwo : rest4.val % 4 = 2 := by
      change rest4.val % 4 ≠ 0 at hcandidate
      omega
    have hsum : lane4.val + 1 ≤ Usize.max := by
      have hmax : 64 ≤ Usize.max := by
        rcases Usize.bounds_eq with h | h <;> rw [h] <;>
          norm_num [U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
      omega
    step as ⟨out, hout⟩
    have hdiv2 : 2 ∣ rest4.val := Nat.dvd_iff_mod_eq_zero.mpr heven
    have hnot4 : ¬ 4 ∣ rest4.val := by
      intro hd
      exact (by omega : rest4.val % 4 ≠ 0) (Nat.dvd_iff_mod_eq_zero.mp hd)
    refine ⟨by omega, ?_, ?_⟩
    · rw [hout, pow_succ, hbits]
      rcases hdiv2 with ⟨q, hq⟩
      exact ⟨q, by rw [hq]; ring⟩
    · intro hd
      have hpow : 2 ^ (out.val + 1) = 4 * 2 ^ lane4.val := by
        rw [hout]
        simp only [pow_add, pow_one]
        ring
      rw [hpow, hbits] at hd
      have hd4 : 4 ∣ rest4.val := by
        apply (Nat.mul_dvd_mul_iff_right (by positivity : 0 < 2 ^ lane4.val)).mp
        exact hd
      exact hnot4 hd4
  · rw [if_neg hlow, WP.spec_ok]
    have hodd : rest4.val % 2 ≠ 0 := by
      intro hz
      apply hlow
      apply UScalar.eq_of_val_eq
      rw [hone]
      exact hz
    refine ⟨by omega, ⟨rest4.val, by simpa [Nat.mul_comm] using hbits⟩, ?_⟩
    intro hd
    have hd2 : 2 ∣ rest4.val := by
      apply (Nat.mul_dvd_mul_iff_right (by positivity : 0 < 2 ^ lane4.val)).mp
      simpa [hbits, pow_succ, Nat.mul_comm] using hd
    exact hodd (Nat.dvd_iff_mod_eq_zero.mp hd2)

/-- A generated hit scan succeeds within the 64-lane word; the result is the least set lane. -/
theorem least_set_lane_exact (bits : U64) (hne : bits ≠ 0#u64) :
    ∃ lane : Usize, table.scan.least_set_lane bits = ok lane ∧
      lane.val < 64 ∧ 2 ^ lane.val ∣ bits.val ∧ ¬ 2 ^ (lane.val + 1) ∣ bits.val := by
  obtain ⟨lane, hresult, hleast⟩ := WP.spec_imp_exists (least_set_lane_selects bits hne)
  exact ⟨lane, hresult, hleast⟩

/-- Every successful result on a nonzero hit word is in range and selects its least set bit. -/
theorem least_set_lane_in_range (bits : U64) (lane : Usize) (hne : bits ≠ 0#u64)
    (hresult : table.scan.least_set_lane bits = ok lane) :
    lane.val < 64 ∧ 2 ^ lane.val ∣ bits.val ∧ ¬ 2 ^ (lane.val + 1) ∣ bits.val := by
  have hspec := least_set_lane_selects bits hne
  rw [hresult, WP.spec_ok] at hspec
  exact hspec

/-- For an odd multiplier, clearing by predecessor removes its trailing unit bit. -/
private theorem odd_multiple_bit (k lane j : Nat) :
    ((2 * k + 1) * 2 ^ lane).testBit j =
      if j = lane then true else ((2 * k) * 2 ^ lane).testBit j := by
  rw [Nat.testBit_mul_two_pow, Nat.testBit_mul_two_pow]
  by_cases hlow : j < lane
  · simp [show j ≠ lane by omega, show ¬lane ≤ j by omega]
  · have hle : lane ≤ j := by omega
    by_cases heq : j = lane
    · subst j
      simp [Nat.testBit_eq_decide_div_mod_eq]
    · have hpos : 0 < j - lane := by omega
      have hb := Nat.testBit_two_pow_mul_add k (show 1 < 2 ^ 1 by norm_num) (j - lane)
      have heven := Nat.testBit_two_pow_mul (i := 1) (a := k) (j := j - lane)
      rw [show 2 * k + 1 = 2 ^ 1 * k + 1 by ring, hb,
        show 2 * k = 2 ^ 1 * k by ring, heven]
      simp [heq, hle, show ¬ j - lane < 1 by omega,
        show 1 ≤ j - lane by omega]

private theorem clear_odd_multiple (k lane : Nat) :
    ((2 * k + 1) * 2 ^ lane) &&& ((2 * k + 1) * 2 ^ lane - 1) =
      (2 * k) * 2 ^ lane := by
  have hpow : 0 < 2 ^ lane := by positivity
  have hprev : (2 * k + 1) * 2 ^ lane - 1 =
      (2 * k) * 2 ^ lane + (2 ^ lane - 1) := by
    rw [add_mul, one_mul]
    omega
  rw [hprev]
  apply Nat.eq_of_testBit_eq
  intro j
  rw [Nat.testBit_and]
  have hlow : 2 ^ lane - 1 < 2 ^ lane := by omega
  have hbit := Nat.testBit_two_pow_mul_add (2 * k) hlow j
  rw [show 2 ^ lane * (2 * k) + (2 ^ lane - 1) =
    2 * k * 2 ^ lane + (2 ^ lane - 1) by ring] at hbit
  rw [hbit]
  by_cases hj : j < lane
  · simp [hj, Nat.testBit_mul_two_pow, show ¬lane ≤ j by omega]
  · rw [odd_multiple_bit]
    by_cases heq : j = lane
    · subst j
      simp [Nat.testBit_eq_decide_div_mod_eq]
    · simp only [heq, ↓reduceIte, Nat.testBit_mul_two_pow,
        show decide (lane ≤ j) = true by simp [show lane ≤ j by omega],
        Bool.true_and, hj, Bool.and_self]

/-- The generated checked subtraction clears exactly the selected lowest bit. -/
theorem generated_clear_hits_exact (hits : U64) (lane : Usize)
    (hne : hits ≠ 0#u64)
    (hselect : table.scan.least_set_lane hits = ok lane) :
    ∃ next : U64,
      (do let previous ← hits - 1#u64
          lift (hits &&& previous)) = ok next ∧
      (∀ j : Nat, next.bv.getLsbD j =
        if j = lane.val then false else hits.bv.getLsbD j) := by
  obtain ⟨hlane, hdiv, hnot⟩ := least_set_lane_in_range hits lane hne hselect
  obtain ⟨q, hq⟩ := hdiv
  have hqodd : q % 2 = 1 := by
    have htwo : ¬ 2 ∣ q := by
      intro hd
      apply hnot
      rcases hd with ⟨k, hk⟩
      refine ⟨k, ?_⟩
      rw [hq, hk, pow_succ]
      ring
    omega
  obtain ⟨k, hk⟩ : ∃ k, q = 2 * k + 1 := ⟨q / 2, by omega⟩
  have hshape : hits.val = (2 * k + 1) * 2 ^ lane.val := by
    rw [hq, hk]
    ring
  have hpositive : 1 ≤ hits.val := by
    rw [hshape]
    have hp : 0 < 2 ^ lane.val := by positivity
    exact Nat.succ_le_of_lt (Nat.mul_pos (by omega) hp)
  obtain ⟨previous, hprevious, hprevval⟩ := WP.spec_imp_exists
    (show (hits - 1#u64) ⦃ previous => previous.val = hits.val - 1 ⦄ from by
      step
      omega)
  refine ⟨hits &&& previous, ?_, ?_⟩
  · simp [hprevious, lift]
  · intro j
    have hnext : (hits &&& previous).val = 2 * k * 2 ^ lane.val := by
      rw [UScalar.val_and, hprevval, hshape]
      exact clear_odd_multiple k lane.val
    change (hits &&& previous).bv.getLsbD j = _
    rw [← BitVec.testBit_toNat]
    change (hits &&& previous).val.testBit j =
      if j = lane.val then false else hits.val.testBit j
    rw [hnext, hshape]
    rw [odd_multiple_bit]
    by_cases heq : j = lane.val
    · subst j
      simp [Nat.testBit_eq_decide_div_mod_eq]
    · simp [heq]

/-- Counts live lanes in 64 bit inspections and constant extra space: native word popcount. -/
def liveLaneCount (hits : U64) : Nat :=
  hits.bv.cpop.toNat

private theorem cpopNatRec_eq_sum (bits : BitVec 64) (n : Nat) :
    bits.cpopNatRec n 0 =
      ∑ j ∈ Finset.range n, (bits.getLsbD j).toNat := by
  induction n with
  | zero => simp [BitVec.cpopNatRec_zero_self]
  | succ n ih =>
    rw [BitVec.cpopNatRec_succ, BitVec.cpopNatRec_eq, ih, Finset.sum_range_succ]
    omega

private theorem liveLaneCount_eq_sum (hits : U64) :
    liveLaneCount hits =
      ∑ j ∈ Finset.range 64, (hits.bv.getLsbD j).toNat := by
  exact (BitVec.toNat_cpop hits.bv).trans (cpopNatRec_eq_sum hits.bv 64)

/-- Each successful generated clear removes one live lane, with no other lane altered. -/
theorem generated_clear_hits_live_decrease (hits : U64) (lane : Usize)
    (hne : hits ≠ 0#u64)
    (hselect : table.scan.least_set_lane hits = ok lane) :
    ∃ next : U64,
      (do let previous ← hits - 1#u64
          lift (hits &&& previous)) = ok next ∧
      (∀ j : Nat, next.bv.getLsbD j =
        if j = lane.val then false else hits.bv.getLsbD j) ∧
      liveLaneCount next + 1 = liveLaneCount hits ∧
      liveLaneCount next < liveLaneCount hits := by
  obtain ⟨next, hresult, hexact⟩ := generated_clear_hits_exact hits lane hne hselect
  have hlane := (least_set_lane_in_range hits lane hne hselect).1
  have hset : hits.bv.getLsbD lane.val = true := by
    obtain ⟨_, hdiv, hnot⟩ := least_set_lane_in_range hits lane hne hselect
    obtain ⟨q, hq⟩ := hdiv
    have hodd : q % 2 = 1 := by
      have hn : ¬2 ∣ q := by
        intro hd
        apply hnot
        obtain ⟨k, hk⟩ := hd
        refine ⟨k, ?_⟩
        rw [hq, hk, pow_succ]
        ring
      omega
    rw [← BitVec.testBit_toNat]
    change hits.val.testBit lane.val = true
    rw [hq, Nat.mul_comm, Nat.testBit_mul_two_pow]
    simp [Nat.testBit_eq_decide_div_mod_eq, hodd]
  have hmem : lane.val ∈ Finset.range 64 := Finset.mem_range.mpr hlane
  have hother :
      ∑ j ∈ (Finset.range 64).erase lane.val, (next.bv.getLsbD j).toNat =
      ∑ j ∈ (Finset.range 64).erase lane.val, (hits.bv.getLsbD j).toNat := by
    apply Finset.sum_congr rfl
    intro j hj
    have hneLane : j ≠ lane.val := (Finset.mem_erase.mp hj).1
    rw [hexact j, if_neg hneLane]
  have hcount : liveLaneCount next + 1 = liveLaneCount hits := by
    rw [liveLaneCount_eq_sum, liveLaneCount_eq_sum]
    rw [← Finset.sum_erase_add (Finset.range 64)
      (fun j => (next.bv.getLsbD j).toNat) hmem,
      ← Finset.sum_erase_add (Finset.range 64)
      (fun j => (hits.bv.getLsbD j).toNat) hmem, hother, hexact, hset]
    simp
  exact ⟨next, hresult, hexact, hcount, by omega⟩

/-- The generated spill-read body executes the proved clear after a selected hit and an
in-bounds tuple lookup. Matching controls only the recorded slot, not the remaining mask. -/
theorem generated_spill_read_clears_selected {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (v : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (hits : U64) (lane slot : Usize) (entry : cell.Sigma × R × V) (matched : Bool)
    (hne : hits ≠ 0#u64)
    (hselect : table.scan.least_set_lane hits = ok lane)
    (hslot : base + lane = ok slot)
    (hindex : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) v slot = ok entry)
    (hmatch : table.matches keyRep entry sigma key
      { place := place, direct_word := direct } = ok matched) :
    ∃ next : U64,
      table.probe.spill_read_loop0_loop0.body keyRep v sigma place direct key base none hits =
        ok (ControlFlow.cont (if matched then some slot else none, next)) ∧
      (∀ j : Nat, next.bv.getLsbD j =
        if j = lane.val then false else hits.bv.getLsbD j) ∧
      liveLaneCount next + 1 = liveLaneCount hits := by
  obtain ⟨next, hclear, hexact, hcount, _⟩ :=
    generated_clear_hits_live_decrease hits lane hne hselect
  refine ⟨next, ?_, hexact, hcount⟩
  unfold table.probe.spill_read_loop0_loop0.body
  have hcond : (hits != 0#u64) = true := by
    simpa using hne
  simp only [hcond, if_true, core.option.Option.is_none, hselect, hslot,
    hindex, hmatch, bind_tc_ok]
  simp only [Option.isNone_none, if_true]
  cases matched <;> simp
  all_goals
    rw [← bind_assoc_eq]
    rw [hclear]
    simp only [bind_tc_ok]

-- The expression is the generated checked clear, including its subtraction failure branch.
#guard match (do let previous ← 32768#u64 - 1#u64
                 lift (32768#u64 &&& previous) : Result U64).match with
  | .ok next => next == 0#u64
  | _ => false

#guard match (do let previous ← 32770#u64 - 1#u64
                 lift (32770#u64 &&& previous) : Result U64).match with
  | .ok next => next == 32768#u64
  | _ => false

-- Lanes 1, 5, 10 and 15 are live; clearing the least retains all three upper lanes.
#guard match (do let previous ← 33826#u64 - 1#u64
                 lift (33826#u64 &&& previous) : Result U64).match with
  | .ok next => next == 33824#u64
  | _ => false

/-- The high-lane tag fixture follows from universal selection, not concrete kernel reduction. -/
theorem least_set_lane_high_fixture :
    table.scan.least_set_lane 32768#u64 = ok 15#usize := by
  obtain ⟨lane, hresult, _, hdiv, hnot⟩ :=
    least_set_lane_exact 32768#u64 (by decide)
  have hval : (32768#u64).val = 2 ^ 15 := by norm_num
  rw [hval] at hdiv hnot
  have hsmall : lane.val ≤ 15 :=
    (Nat.pow_dvd_pow_iff_le_right (by norm_num : 1 < (2 : Nat))).mp hdiv
  have hlarge : 15 < lane.val + 1 := by
    by_contra h
    exact hnot ((Nat.pow_dvd_pow_iff_le_right (by norm_num : 1 < (2 : Nat))).mpr
      (by omega))
  have heq : lane = 15#usize := UScalar.eq_of_val_eq (by norm_num; omega)
  simpa only [heq] using hresult

-- A two-hit word distinguishes the least set lane from the highest set lane.
#guard match (table.scan.least_set_lane 32770#u64).match with
  | .ok lane => lane == 1#usize
  | _ => false

/-- Asymmetric 16-lane line: the second eight-byte window contains a sole high-lane hit.
    The first window and the absent-tag case constrain both generated loops. -/
def highLaneLine : Slice U8 :=
  Slice.from [1#u8, 2#u8, 3#u8, 4#u8, 5#u8, 6#u8, 7#u8, 8#u8,
    9#u8, 10#u8, 11#u8, 12#u8, 13#u8, 14#u8, 15#u8, 42#u8] (by scalar_tac)

#guard match (table.scan.match_tag highLaneLine 42#u8).match with
  | .ok mask => mask == 32768#u64
  | _ => false

#guard match (table.scan.match_tag highLaneLine 100#u8).match with
  | .ok mask => mask == 0#u64
  | _ => false

/- The leading vacancy and the vacancy at the second window's start do not terminate the
   generated tag scan; the last lane still matches. -/
def highLaneAfterVacancies : Slice U8 :=
  Slice.from [0#u8, 2#u8, 3#u8, 4#u8, 5#u8, 6#u8, 7#u8, 8#u8,
    0#u8, 10#u8, 11#u8, 12#u8, 13#u8, 14#u8, 15#u8, 42#u8] (by scalar_tac)

#guard match (table.scan.match_tag highLaneAfterVacancies 42#u8).match with
  | .ok mask => mask == 32768#u64
  | _ => false

#guard match (table.scan.match_tag highLaneAfterVacancies 100#u8).match with
  | .ok mask => mask == 0#u64
  | _ => false

#guard match (table.scan.match_tag highLaneAfterVacancies 0#u8).match with
  | .ok mask => mask == 257#u64
  | _ => false

-- Lane seven is gathered from a word; lane eight is set by the scalar tail.
def wordTailBoundaryLine : Slice U8 :=
  Slice.from [0#u8, 2#u8, 3#u8, 4#u8, 5#u8, 6#u8, 7#u8, 42#u8, 42#u8]
    (by scalar_tac)

#guard match (table.scan.match_tag wordTailBoundaryLine 42#u8).match with
  | .ok mask => mask == 384#u64
  | _ => false

#guard match (table.scan.match_tag wordTailBoundaryLine 100#u8).match with
  | .ok mask => mask == 0#u64
  | _ => false

-- The scalar tail begins after both of the preceding full words, not after the first.
def secondWordTailLine : Slice U8 :=
  Slice.from (List.replicate 16 (0#u8) ++ [42#u8]) (by scalar_tac)

#guard match (table.scan.match_tag secondWordTailLine 42#u8).match with
  | .ok mask => mask == 65536#u64
  | _ => false

-- All preceding groups contain vacancies; the last lane of the eighth word still hits.
def lastGroupLine : Slice U8 :=
  Slice.from (List.replicate 63 (0#u8) ++ [42#u8]) (by scalar_tac)

#guard match (table.scan.match_tag lastGroupLine 42#u8).match with
  | .ok mask => mask == 9223372036854775808#u64
  | _ => false

#guard match (table.scan.least_set_lane 32768#u64).match with
  | .ok lane => lane == 15#usize
  | _ => false

#guard match (table.scan.least_set_lane 0#u64).match with
  | .ok lane => lane == 64#usize
  | _ => false

/-- For a non-overflowing grouped traversal, every tail step lies strictly below the
    generated `WINDOWS + groups` bound; in particular the last step is not omitted. -/
theorem generated_tail_steps_in_bound (groups : Usize)
    (hcap : 7 + groups.val ≤ Usize.max) :
    ∃ limit : Usize, table.WINDOWS + groups = ok limit ∧
      ∀ j : Nat, j < groups.val →
        ∃ step : Usize, step.val = 7 + j ∧ step < limit := by
  have hwindow : table.WINDOWS.val = 7 := by simp [table.WINDOWS]
  have hsum : table.WINDOWS.val + groups.val ≤ Usize.max := by
    rw [hwindow]
    exact hcap
  obtain ⟨limit, hlimit, hlimitVal⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := table.WINDOWS) (y := groups) hsum)
  refine ⟨limit, hlimit, ?_⟩
  intro j hj
  have hjCap : 7 + j < 2 ^ System.Platform.numBits := by
    have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    have hpow : 0 < 2 ^ System.Platform.numBits := by positivity
    omega
  let step : Usize := ⟨BitVec.ofNat System.Platform.numBits (7 + j)⟩
  have hstep : step.val = 7 + j := by
    simp [step, UScalar.val, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hjCap]
  refine ⟨step, hstep, ?_⟩
  apply (UScalar.lt_equiv step limit).mpr
  omega

/- Four successive tail steps visit groups 1, 2, 3, 0 when `groups = 4`.
   Group zero is reached only on the final step of this suffix. -/
#guard match (table.group 0#u64 7#usize 3#usize).match with
  | .ok group => group == 1#usize
  | _ => false
#guard match (table.group 0#u64 8#usize 3#usize).match with
  | .ok group => group == 2#usize
  | _ => false
#guard match (table.group 0#u64 9#usize 3#usize).match with
  | .ok group => group == 3#usize
  | _ => false
#guard match (table.group 0#u64 10#usize 3#usize).match with
  | .ok group => group == 0#usize
  | _ => false

end Apeiron.GrowableScan
