/-
Refinement of the extracted `f16` and `F16`-row dequantizers (`APEIRON-ALGEBRA-G3-DEQUANT-W1-T1`).

The Rust bodies are `dequant::f16_to_fixed`, `dequant::read_u16`, `dequant::dequant_f16_row` and
the `GGML_TYPE_F16` arm of `dequant::dequant_row_fixed` in `src/engine/dequant.rs`, extracted into
`proof/ApeironWord.lean`. The decode is exact integer arithmetic: every finite IEEE 754-2008
binary16 value is a multiple of `2^-24`, hence of the Q32.32 unit `2^-32`, so the law is an `ℤ`
equality, not an error bound.

## Proved

* `f16_to_fixed_exact`: for `f16Exp h ≠ 31`, `f16_to_fixed h = ok (Ok v)` with
  `v.val = f16Value h`; `f16_to_fixed_nonfinite`: `e = 31` refuses with `NonFiniteScale h`;
  `f16_to_fixed_real`: `toReal v = f16Real h`, the binary16 value `(-1)^s·m·2^-24` (subnormal) or
  `(-1)^s·(1+m/2^10)·2^(e-15)` (normal).
* `dequant_f16_row_exact`: on `f16RowDomain raw n` (`2·n` bytes, every half finite) and room for
  `n` pushes, `dequant_f16_row raw n out` returns `Ok ()` and appends exactly
  `f16RowValues raw n = [f16Value (le u16 at 2j) | j < n]`.
* `dequant_row_fixed_f16`: `dequant_row_fixed 1 = dequant_f16_row`.
* `matvec_domain_f16_witness`: a joint witness of every `matvec_domain` conjunct at `n_out = 1`,
  `n_in = 1`, `ggml_type = 1`, bytes `[0x00, 0x3C]` (`1.0`), activation `[1.0]`, including the
  dequant conjunct; `matvec_f16_witness_refines` feeds it to `matvec_refines_proved`.

## Residual

The row laws for `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`, `Q2_K` and `Q3_K` (the pattern is
`dequant_f16_row_exact` composed with the block-code equalities of `Numeric.FormatRefinement`,
e.g. `word_q4_k_codes_ok`/`word_q6_k_codes_ok`) and the `F16` row refusals (short payload,
non-finite half) stated at the row level are not proved here.

## Trust boundary

Kernel-checked against the Aeneas model of `i64::checked_shl`, `u16::from_le_bytes`, slice range
`get` and `Vec::push` (`target/aeneas-model/lean`); rustc/LLVM codegen and the GGUF file reader
upstream of the row bytes are outside this module.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts (`0x3C00` from two bytes, distinct `u32` match literals), never on an extracted body.
-/
import Numeric.EngineMatvecRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-- Sign bit `s = h >> 15` of an IEEE 754-2008 binary16 pattern. -/
def f16Sign (h : U16) : Nat := h.val / 2 ^ 15
/-- Biased exponent field `e = (h >> 10) mod 32`. -/
def f16Exp (h : U16) : Nat := (h.val / 2 ^ 10) % 32
/-- Trailing significand field `m = h mod 2^10`. -/
def f16Man (h : U16) : Nat := h.val % 2 ^ 10

/-- Exact Q32.32 magnitude of a finite pattern: subnormal `m·2^8` (`= m·2^-24·2^32`), normal
`(2^10+m)·2^(e+7)` (`= (1+m/2^10)·2^(e-15)·2^32`). Every finite binary16 value is a multiple of
`2^-24`, so the image is an integer. -/
def f16Mag (h : U16) : ℤ :=
  if f16Exp h = 0 then (f16Man h : ℤ) * 2 ^ 8
  else ((1024 + f16Man h : ℕ) : ℤ) * 2 ^ (f16Exp h + 7)

/-- Exact signed Q32.32 integer of a finite binary16 pattern (`e ≠ 31`). -/
def f16Value (h : U16) : ℤ := if f16Sign h = 1 then -f16Mag h else f16Mag h

/-- The modelled `i64::checked_shl` success value is the exact product when it fits `i64`. -/
theorem dequant_wrapping_shl_val (x : I64) (s : U32) (hx0 : 0 ≤ x.val) (hs : s.val < 64)
    (hfit : x.val * 2 ^ s.val < 2 ^ 63) :
    (IScalar.wrapping_shl x s).val = x.val * 2 ^ s.val := by
  simp only [IScalar.wrapping_shl, IScalar.val, IScalarTy.I64_numBits_eq, Nat.mod_eq_of_lt hs]
  have hxv : (x.val : ℤ) = x.bv.toInt := rfl
  have hsh : x.bv.shiftLeft s.val = x.bv <<< s.val := rfl
  rw [hsh, BitVec.toInt_shiftLeft]
  have hnat : (x.bv.toNat : ℤ) = x.bv.toInt := by
    have hc := BitVec.toInt_eq_toNat_cond x.bv
    split at hc
    · exact hc.symm
    · exfalso
      have := x.bv.isLt
      omega
  have hpos : 0 ≤ x.val * 2 ^ s.val := mul_nonneg hx0 (by positivity)
  rw [Nat.shiftLeft_eq]
  push_cast
  rw [hnat, ← hxv]
  apply Int.bmod_eq_of_le <;> norm_num <;> omega

/-- `h >> 15` is the sign field. -/
theorem f16_shr15 (h : U16) : ∃ s : U16, h >>> 15#i32 = ok s ∧ s.val = f16Sign h := by
  obtain ⟨s, hs, hsv, -⟩ := WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 15#i32 (by simp) (by simp))
  refine ⟨s, hs, ?_⟩
  rw [hsv, Nat.shiftRight_eq_div_pow]
  rfl

/-- `(h >> 10) & 0x1f` is the exponent field. -/
theorem f16_exp_eq (h i : U16) (hiv : i.val = h.val >>> 10) :
    (i &&& 31#u16).val = f16Exp h := by
  rw [UScalar.val_and, hiv, Nat.shiftRight_eq_div_pow]
  have h31 : (31#u16).val = 2 ^ 5 - 1 := by simp
  rw [h31, Nat.and_two_pow_sub_one_eq_mod]
  rfl

/-- `h & 0x3ff` is the significand field. -/
theorem f16_man_eq (h : U16) : (h &&& 1023#u16).val = f16Man h := by
  rw [UScalar.val_and]
  have h31 : (1023#u16).val = 2 ^ 10 - 1 := by simp
  rw [h31, Nat.and_two_pow_sub_one_eq_mod]
  rfl

theorem f16_exp_lt (h : U16) : f16Exp h < 32 := Nat.mod_lt _ (by norm_num)
theorem f16_man_lt (h : U16) : f16Man h < 1024 := Nat.mod_lt _ (by norm_num)

/-- **`f16` decode law.** For a finite pattern (`e ≠ 0x1f`) the extracted `f16_to_fixed` returns
`Ok v` with `v.val = f16Value h` exactly: no rounding, no overflow refusal. -/
theorem f16_to_fixed_exact (h : U16) (he : f16Exp h ≠ 31) :
    ∃ v : I64, dequant.f16_to_fixed h = ok (.Ok v) ∧ v.val = f16Value h := by
  obtain ⟨sign, hs, hsv⟩ := f16_shr15 h
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hiv' : i.val = h.val >>> 10 := by rw [hiv]; rfl
  have hE := f16_exp_eq h i hiv'
  have hM := f16_man_eq h
  have hEl := f16_exp_lt h
  have hMl := f16_man_lt h
  have hmcast : (UScalar.hcast .I64 (h &&& 1023#u16)).val = (f16Man h : ℤ) := by
    rw [UScalar.hcast_val_eq, hM]
    apply Int.bmod_eq_of_le
    all_goals norm_num
    all_goals omega
  rw [dequant.f16_to_fixed]
  simp only [hs, hi, lift, bind_tc_ok, I64.Insts.CoreConvertFromU16.from]
  by_cases he0 : (i &&& 31#u16) = 0#u16
  · have hE0 : f16Exp h = 0 := by rw [← hE, he0]; simp
    obtain ⟨mag, hmag, hmagv, -⟩ := WP.spec_imp_exists
      (I64.ShiftLeft_IScalar_spec (UScalar.hcast .I64 (h &&& 1023#u16)) 8#i32 (by simp) (by simp))
    have hmagv' : mag.val = (f16Man h : ℤ) * 2 ^ 8 := by
      have hsz : I64.size = 2 ^ 64 := by simp [I64.size, I64.numBits]
      have h8 : (8#i32).toNat = 8 := by simp
      simp only [hmagv, hmcast, Int.shiftLeft_eq, hsz, h8]
      apply Int.bmod_eq_of_le <;> norm_num <;> omega
    simp only [if_pos he0, hmag, bind_tc_ok]
    by_cases hs1 : sign = 1#u16
    · have hS : f16Sign h = 1 := by rw [← hsv, hs1]; simp
      obtain ⟨mag1, hneg, hnegv⟩ := WP.spec_imp_exists
        (HNeg.hNeg.step mag (by
          intro hc
          rw [hmagv', IScalar.min_IScalarTy_I64_eq, I64.min_eq] at hc
          omega))
      simp only [if_pos hs1, hneg, bind_tc_ok]
      refine ⟨mag1, rfl, ?_⟩
      rw [hnegv, hmagv', f16Value, if_pos hS, f16Mag, if_pos hE0]
    · have hS : f16Sign h ≠ 1 := by
        intro hc; apply hs1; apply UScalar.eq_of_val_eq; rw [hsv, hc]; simp
      simp only [if_neg hs1]
      refine ⟨mag, rfl, ?_⟩
      rw [hmagv', f16Value, if_neg hS, f16Mag, if_pos hE0]
  · have hE0 : f16Exp h ≠ 0 := by
      intro hc; apply he0; apply UScalar.eq_of_val_eq; rw [hE, hc]; simp
    have he31 : ¬ (i &&& 31#u16) = 31#u16 := by
      intro hc; apply he; rw [← hE, hc]; simp
    simp only [if_neg he0, if_neg he31]
    obtain ⟨sig, hsig, hsigv⟩ := WP.spec_imp_exists
      (I64.add_spec (x := UScalar.hcast .I64 (h &&& 1023#u16)) (y := 1024#i64)
        (by rw [hmcast, I64.min_eq]; simp; omega) (by rw [hmcast, I64.max_eq]; simp; omega))
    have hsigv' : sig.val = ((1024 + f16Man h : ℕ) : ℤ) := by
      rw [hsigv, hmcast]; push_cast; simp; ring
    have he2 : (core.convert.num.FromU32U16.from (i &&& 31#u16)).val = f16Exp h := by
      rw [core.convert.num.FromU32U16.from_val_eq, hE]
    obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
      (U32.add_spec (x := core.convert.num.FromU32U16.from (i &&& 31#u16)) (y := 7#u32)
        (by rw [he2, U32.max_eq]; simp; omega))
    have hi3v' : i3.val = f16Exp h + 7 := by rw [hi3v, he2]; simp
    have hshl : core.num.I64.checked_shl sig i3 =
        ok (some (IScalar.wrapping_shl sig i3)) := by
      rw [core.num.I64.checked_shl, if_pos (by rw [hi3v']; omega)]
    have hval : (IScalar.wrapping_shl sig i3).val = f16Mag h := by
      rw [dequant_wrapping_shl_val sig i3 (by rw [hsigv']; positivity) (by rw [hi3v']; omega)]
      · rw [hsigv', hi3v', f16Mag, if_neg hE0]
      · rw [hsigv', hi3v']
        have h1 : ((1024 + f16Man h : ℕ) : ℤ) ≤ 2047 := by push_cast; omega
        have h2 : (2 : ℤ) ^ (f16Exp h + 7) ≤ 2 ^ 37 := pow_le_pow_right₀ (by norm_num) (by omega)
        have h3 : ((1024 + f16Man h : ℕ) : ℤ) * 2 ^ (f16Exp h + 7) ≤ 2047 * 2 ^ 37 :=
          mul_le_mul h1 h2 (by positivity) (by norm_num)
        norm_num at h3 ⊢
        omega
    simp only [hsig, hi3, hshl, bind_tc_ok, core.option.Option.ok_or_some,
      core.result.Result.Insts.CoreOpsTry.branch]
    by_cases hs1 : sign = 1#u16
    · have hS : f16Sign h = 1 := by rw [← hsv, hs1]; simp
      have hmagpos : 0 ≤ f16Mag h := by rw [f16Mag, if_neg hE0]; positivity
      obtain ⟨mag1, hneg, hnegv⟩ := WP.spec_imp_exists
        (HNeg.hNeg.step (IScalar.wrapping_shl sig i3) (by
          intro hc
          rw [hval, IScalar.min_IScalarTy_I64_eq, I64.min_eq] at hc
          omega))
      simp only [if_pos hs1, hneg, bind_tc_ok]
      refine ⟨mag1, rfl, ?_⟩
      rw [hnegv, hval, f16Value, if_pos hS]
    · have hS : f16Sign h ≠ 1 := by
        intro hc; apply hs1; apply UScalar.eq_of_val_eq; rw [hsv, hc]; simp
      simp only [if_neg hs1]
      refine ⟨_, rfl, ?_⟩
      rw [hval, f16Value, if_neg hS]

/-- **`f16` refusal.** An `Inf`/`NaN` exponent (`e = 0x1f`) refuses with `NonFiniteScale h`. -/
theorem f16_to_fixed_nonfinite (h : U16) (he : f16Exp h = 31) :
    dequant.f16_to_fixed h = ok (.Err (error.EngineError.NonFiniteScale h)) := by
  obtain ⟨sign, hs, -⟩ := f16_shr15 h
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hiv' : i.val = h.val >>> 10 := by rw [hiv]; rfl
  have hE := f16_exp_eq h i hiv'
  have he0 : ¬ (i &&& 31#u16) = 0#u16 := by
    intro hc; have := congrArg UScalar.val hc; rw [hE, he] at this; simp at this
  have he31 : (i &&& 31#u16) = 31#u16 := by
    apply UScalar.eq_of_val_eq; rw [hE, he]; simp
  rw [dequant.f16_to_fixed]
  simp only [hs, hi, lift, bind_tc_ok, if_neg he0, if_pos he31]

/-- The little-endian `u16` at byte offset `i` of `raw` (bytes past the end read as `0`). -/
def f16LeHalf (raw : Slice U8) (i : Nat) : U16 :=
  core.num.U16.from_le_bytes (Array.make 2#usize [raw.val.getD i 0#u8, raw.val.getD (i + 1) 0#u8])

/-- A two-element window of a list, by index. -/
theorem dequant_slice_two (l : List U8) (i : Nat) (h : i + 2 ≤ l.length) :
    l.slice i (i + 2) = [l.getD i 0#u8, l.getD (i + 1) 0#u8] := by
  unfold List.slice
  have hd : l.drop i = l.getD i 0#u8 :: l.getD (i + 1) 0#u8 :: l.drop (i + 2) := by
    rw [List.drop_eq_getElem_cons (by omega), List.drop_eq_getElem_cons (by omega)]
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (show i < l.length by omega),
      List.getElem?_eq_getElem (show i + 1 < l.length by omega)]
  rw [hd]
  simp

/-- `read_u16` inside the payload returns the little-endian half at `i`. -/
theorem read_u16_ok (raw : Slice U8) (i : Usize) (hi : i.val + 2 ≤ raw.val.length) :
    dequant.read_u16 raw i = ok (.Ok (f16LeHalf raw i.val)) := by
  have hmax : raw.val.length ≤ Usize.max := raw.property
  obtain ⟨i1, hadd, hi1⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 2#usize) (by simp; omega))
  have hi1' : i1.val = i.val + 2 := by rw [hi1]; simp
  have hcond : i ≤ i1 ∧ i1.val ≤ raw.length := by
    refine ⟨?_, ?_⟩
    · rw [UScalar.le_equiv, hi1']; omega
    · rw [hi1']; exact hi
  have hsl := dequant_slice_two raw.val i.val hi
  rw [dequant.read_u16]
  simp only [hadd, bind_tc_ok, core.slice.Slice.get,
    core.slice.index.SliceIndexRangeUsizeSlice.get]
  rw [if_pos hcond]
  simp only [bind_tc_ok, core.option.Option.and_then,
    dequant.read_u16.closure.Insts.CoreOpsFunctionFnOnceTupleSharedSliceU8OptionArrayU82.call_once,
    core.array.TryFromArrayCopySlice.try_from]
  have hsl' : List.slice i.val i1.val raw.val = [raw.val.getD i.val 0#u8, raw.val.getD (i.val + 1) 0#u8] := by
    rw [hi1']; exact hsl
  have hlen : (Slice.from (List.slice i.val i1.val raw.val) (by
      have := List.slice_length_le i.val i1.val raw.val; omega) : Slice U8).length = (2#usize).val := by
    simp [Slice.length, hsl']
  rw [dif_pos hlen]
  simp only [bind_tc_ok, core.result.Result.ok, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch, lift]
  unfold f16LeHalf
  congr 3
  apply Aeneas.Std.Array.ext
  rw [Array.make_val, Array.from_val]
  rw [Slice.from_val, hsl']

/-- The `j`-th `f16` of an `F16` row: the little-endian `u16` at byte offset `2·j`. -/
def f16Half (raw : Slice U8) (j : Nat) : U16 := f16LeHalf raw (2 * j)

/-- The exact Q32.32 integers of the first `n` weights of an `F16` row. -/
def f16RowValues (raw : Slice U8) (n : Nat) : List ℤ :=
  (List.range n).map (fun j => f16Value (f16Half raw j))

/-- The `F16` row success domain: `2·n` payload bytes, every weight finite. -/
def f16RowDomain (raw : Slice U8) (n : Nat) : Prop :=
  2 * n ≤ raw.val.length ∧ ∀ j, j < n → f16Exp (f16Half raw j) ≠ 31

/-- The integer values of a list of `i64` words. -/
def dequantVals (l : List I64) : List ℤ := l.map (fun v : I64 => v.val)

/-- The `F16` row-loop invariant after `idx` decoded weights. -/
def f16LoopInv (raw : Slice U8) (n : Usize) (out0 : alloc.vec.Vec I64)
    (out : alloc.vec.Vec I64) (idx : Usize) (ok1 : Bool) (err : Option error.EngineError) :
    Prop :=
  ok1 = true ∧ err = none ∧ idx.val ≤ n.val ∧
  dequantVals out.val = dequantVals out0.val ++ f16RowValues raw idx.val

/-- One `F16` row-loop body step on the success domain: read, decode, push, advance. -/
theorem f16_loop_body_step (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (idx : Usize) (hdom : f16RowDomain raw n.val)
    (hcap : out0.val.length + n.val ≤ Usize.max) (hlt : idx.val < n.val)
    (hinv : f16LoopInv raw n out0 out idx true none) :
    ∃ (out2 : alloc.vec.Vec I64) (idx2 : Usize),
      dequant.dequant_f16_row_loop.body raw n out idx true none =
        ok (ControlFlow.cont (out2, idx2, true, none)) ∧
      f16LoopInv raw n out0 out2 idx2 true none ∧ idx2.val = idx.val + 1 := by
  obtain ⟨hlen2, hfin⟩ := hdom
  obtain ⟨-, -, -, hvals⟩ := hinv
  have hmax : raw.val.length ≤ Usize.max := raw.property
  obtain ⟨i, hmul, hiv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 2#usize) (y := idx) (by simp; omega))
  have hiv' : i.val = 2 * idx.val := by rw [hiv]; simp
  have hread := read_u16_ok raw i (by rw [hiv']; omega)
  obtain ⟨v, hv, hvv⟩ := f16_to_fixed_exact (f16LeHalf raw i.val) (by
    rw [hiv']; exact hfin idx.val hlt)
  have hlenout : out.val.length = out0.val.length + idx.val := by
    have := congrArg List.length hvals
    simp [dequantVals, f16RowValues] at this
    omega
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec out v (by rw [hlenout]; omega))
  obtain ⟨idx2, hadd, hidx2⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := idx) (y := 1#usize) (by simp; omega))
  have hidx2' : idx2.val = idx.val + 1 := by rw [hidx2]; simp
  refine ⟨out2, idx2, ?_, ?_, hidx2'⟩
  · rw [dequant.dequant_f16_row_loop.body]
    simp only [if_pos ((UScalar.lt_equiv idx n).2 hlt), if_true, hmul, hread, hv, hpush, hadd,
      bind_tc_ok, uncurry_apply_pair]
  · refine ⟨rfl, rfl, by omega, ?_⟩
    rw [hpushv, hidx2', dequantVals, List.map_append, ← dequantVals, hvals]
    simp only [f16RowValues, List.range_succ, List.map_append, List.append_assoc, List.map_cons,
      List.map_nil, hvv, f16Half, hiv']

/-- The `F16` row loop: `loop.spec_decr_nat` with `f16LoopInv` and measure `n - idx`.

Complexity: `n` iterations, each O(1) (one 2-byte read, one decode, one push); O(n) words appended. -/
theorem f16_row_loop_spec (raw : Slice U8) (n : Usize) (out0 : alloc.vec.Vec I64)
    (hdom : f16RowDomain raw n.val) (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.dequant_f16_row_loop raw n out0 0#usize true none)
      (fun b => ∃ out : alloc.vec.Vec I64, b = (out, none) ∧
        dequantVals out.val = dequantVals out0.val ++ f16RowValues raw n.val) := by
  have hinv0 : f16LoopInv raw n out0 out0 0#usize true none := by
    refine ⟨rfl, rfl, by simp, ?_⟩
    simp [f16RowValues]
  unfold dequant.dequant_f16_row_loop
  apply loop.spec_decr_nat
    (measure := fun s : (alloc.vec.Vec Std.I64) × Std.Usize × Bool × (Option error.EngineError) =>
      n.val - s.2.1.val)
    (inv := fun s => f16LoopInv raw n out0 s.1 s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, idx, ok1, err⟩
    obtain ⟨hok, herr, hidx, hvals⟩ := hs
    have hok' : ok1 = true := hok
    have herr' : err = none := herr
    have hidx' : idx.val ≤ n.val := hidx
    rw [hok', herr']
    by_cases hlt : idx.val < n.val
    · obtain ⟨out2, idx2, hbody, hinv2, hidx2⟩ :=
        f16_loop_body_step raw n out0 out idx hdom hcap hlt ⟨rfl, rfl, hidx', hvals⟩
      simp only [hbody]
      refine ⟨hinv2, ?_⟩
      show n.val - idx2.val < n.val - idx.val
      omega
    · have hnot : ¬ (idx < n) := by
        intro hc
        exact hlt ((UScalar.lt_equiv idx n).1 hc)
      have hbody : dequant.dequant_f16_row_loop.body raw n out idx true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.dequant_f16_row_loop.body, if_neg hnot]
      simp only [hbody]
      refine ⟨out, rfl, ?_⟩
      have hidxn : idx.val = n.val := by omega
      rw [← hidxn]
      exact hvals
  · exact hinv0

/-- **`F16` row law.** On `f16RowDomain` the extracted `dequant_f16_row` succeeds and appends
exactly the `n` Q32.32 integers `f16Value` of the little-endian halves at byte offsets `2·j`. -/
theorem dequant_f16_row_exact (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : f16RowDomain raw n.val) (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out' : alloc.vec.Vec I64,
      dequant.dequant_f16_row raw n out = ok (core.result.Result.Ok (), out') ∧
      dequantVals out'.val = dequantVals out.val ++ f16RowValues raw n.val := by
  obtain ⟨b, hb, out', hbout, hvals⟩ :=
    WP.spec_imp_exists (f16_row_loop_spec raw n out hdom hcap)
  rw [hbout] at hb
  refine ⟨out', ?_, hvals⟩
  rw [dequant.dequant_f16_row, hb]
  simp only [bind_tc_ok, uncurry_apply_pair]

/-- `dequant_row_fixed` at `ggml_type = 1` (`GGML_TYPE_F16`) is the `F16` row. -/
theorem dequant_row_fixed_f16 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 1#u32 raw n out = dequant.dequant_f16_row raw n out := by
  unfold dequant.dequant_row_fixed
  split
  · rfl
  all_goals first
    | (rename_i heq; have hv := congrArg UScalar.val heq; exact absurd hv (by decide))
    | (exfalso; exact ‹(1#u32 : U32) = 1#u32 → False› rfl)

/-- `checked_mul` succeeds with the exact product inside `usize`. -/
theorem dequant_usize_checked_mul_some (x y : Usize) (h : x.val * y.val ≤ Usize.max) :
    ∃ z : Usize, Usize.checked_mul x y = some z ∧ z.val = x.val * y.val := by
  have hs := Usize.checked_mul_bv_spec x y
  cases hc : Usize.checked_mul x y with
  | none => rw [hc] at hs; simp only at hs; omega
  | some z => rw [hc] at hs; exact ⟨z, rfl, hs.2.1⟩

/-- `block_geom` for `GGML_TYPE_F16`: one weight per 2-byte block. -/
theorem block_geom_f16 : quant.block_geom 1#u32 = ok (some (1#usize, 2#usize)) := by
  unfold quant.block_geom
  split
  all_goals first
    | rfl
    | (rename_i heq; have hv := congrArg UScalar.val heq; exact absurd hv (by decide))
    | (exfalso; exact ‹(1#u32 : U32) = 1#u32 → False› rfl)

/-- A one-word list is determined by its integer value. -/
theorem dequantVals_singleton (l : List I64) (v : I64) (h : dequantVals l = [v.val]) : l = [v] := by
  match l, h with
  | [a], h =>
    simp only [dequantVals, List.map_cons, List.map_nil, List.cons.injEq, and_true] at h
    rw [IScalar.eq_of_val_eq h]
  | [], h => simp [dequantVals] at h
  | _ :: _ :: _, h => simp [dequantVals] at h

/-- The witness row: the `f16` bit pattern `0x3C00` (`1.0`) in little-endian order. -/
def f16WitRaw : Slice U8 := Slice.from [0#u8, 60#u8] (by simp; scalar_tac)

/-- `1.0` in Q32.32. -/
def f16WitOne : I64 := 4294967296#i64

/-- The one-element activation / dequantized row `[1.0]`. -/
def f16WitVec : Slice I64 := Slice.from [f16WitOne] (by simp; scalar_tac)

theorem f16WitOne_val : f16WitOne.val = 2 ^ 32 := by simp [f16WitOne]

/-- The witness bytes `[0x00, 0x3C]` read as `0x3C00 = 15360`. -/
theorem f16LeHalf_wit (row : Slice U8) (hrow : row.val = [0#u8, 60#u8]) :
    (f16LeHalf row 0).val = 15360 := by
  have hlit : (core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8])).val = 15360 := by
    decide
  have heq : f16LeHalf row 0 = core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) := by
    unfold f16LeHalf
    congr 1
    apply Aeneas.Std.Array.ext
    simp [hrow]
  rw [heq, hlit]

/-- `0x3C00` decodes to `2^32`, i.e. `1.0` in Q32.32. -/
theorem f16Value_wit (row : Slice U8) (hrow : row.val = [0#u8, 60#u8]) :
    f16Value (f16Half row 0) = 2 ^ 32 := by
  have hv := f16LeHalf_wit row hrow
  have h0 : f16Half row 0 = f16LeHalf row 0 := rfl
  simp only [h0, f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

theorem f16WitRaw_val : f16WitRaw.val = [0#u8, 60#u8] := Slice.from_val _ _
theorem f16WitVec_val : f16WitVec.val = [f16WitOne] := Slice.from_val _ _

/-- `div_ceil 1 1 = 1`. -/
theorem dequant_div_ceil_one_one :
    ∃ q : Usize, core.num.Usize.div_ceil 1#usize 1#usize = ok q ∧ q.val = 1 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec (1#usize) (y := 1#usize) (by simp)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists (UScalar.rem_spec (1#usize) (y := 1#usize) (by simp))
  have hq' : UScalar.div (1#usize) (1#usize) = ok q := hq
  have hr' : UScalar.rem (1#usize) (1#usize) = ok r := hr
  have hr0 : r.val = 0 := by rw [hrv]; simp
  refine ⟨q, ?_, by rw [hqv]; simp⟩
  unfold core.num.Usize.div_ceil
  simp only [hq', hr', bind_tc_ok, if_pos hr0]

/-- **Joint non-vacuity of `matvec_domain` at `n_out = 1`, `F16`.** One `F16` weight `0x3C00`
(`1.0`, bytes `[0x00, 0x3C]`), activation `[1.0]` (`2^32` in Q32.32), `n_in = 1`: every conjunct
holds at once, including the dequant conjunct, discharged by `dequant_f16_row_exact` through
`dequant_row_fixed_f16`, with `deq 0 = [1.0]`. -/
theorem matvec_domain_f16_witness :
    ∃ (qk bb blocks row_bytes need : Usize),
      matvec_domain 1#u32 f16WitRaw 1#usize 1#usize f16WitVec qk bb blocks row_bytes need
        (fun _ => f16WitVec) := by
  have hraw := f16WitRaw_val
  have hx := f16WitVec_val
  obtain ⟨q, hdiv, hqv⟩ := dequant_div_ceil_one_one
  obtain ⟨rb, hrb, hrbv⟩ := dequant_usize_checked_mul_some q 2#usize (by rw [hqv]; simp; scalar_tac)
  have hrbv' : rb.val = 2 := by rw [hrbv, hqv]; simp
  obtain ⟨need, hneed, hneedv⟩ := dequant_usize_checked_mul_some 1#usize rb (by rw [hrbv']; simp; scalar_tac)
  have hneedv' : need.val = 2 := by rw [hneedv, hrbv']; simp
  refine ⟨1#usize, 2#usize, q, rb, need, block_geom_f16, ?_, hdiv, hrb, hneed, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hx]; simp
  · rw [hneedv', hraw]; simp
  · intro r _
    rw [hx]; simp
  · intro r _
    refine ⟨rfl, ?_, ?_, ?_⟩
    · intro k hk1 hk2
      have hk : k = 1 := by rw [hx] at hk2; simp at hk2; omega
      subst hk
      simp only [dotPrefix, hx, f16WitOne_val, List.take, List.zip_cons_cons, List.zip_nil_left,
        List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, I128.min_eq, I128.max_eq]
      norm_num
    · simp only [dotPrefix, hx, f16WitOne_val, divRoundNearest, List.length_cons, List.length_nil,
        List.take, List.zip_cons_cons, List.zip_nil_left, List.map_cons, List.map_nil,
        List.sum_cons, List.sum_nil]
      norm_num
    · simp only [dotPrefix, hx, f16WitOne_val, divRoundNearest, List.length_cons, List.length_nil,
        List.take, List.zip_cons_cons, List.zip_nil_left, List.map_cons, List.map_nil,
        List.sum_cons, List.sum_nil]
      norm_num
  · intro o ho
    have ho0 : o.val = 0 := by simp at ho; omega
    obtain ⟨start, hstart, hstartv⟩ := dequant_usize_checked_mul_some o rb (by rw [ho0]; simp)
    have hstartv' : start.val = 0 := by rw [hstartv, ho0]; simp
    obtain ⟨i, hadd, hiv⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := start) (y := rb) (by rw [hstartv', hrbv']; simp; scalar_tac))
    have hiv' : i.val = 2 := by rw [hiv, hstartv', hrbv']
    have hget : ∃ row : Slice U8,
        core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8) f16WitRaw
          { start := start, «end» := i } = ok (some row) := by
      simp only [core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get]
      rw [if_pos ⟨by rw [UScalar.le_equiv, hstartv', hiv']; omega,
        by simp [Slice.length, hiv', hraw]⟩]
      exact ⟨_, rfl⟩
    obtain ⟨row, hget⟩ := hget
    exact ⟨start, i, row, hstart, hadd, hget⟩
  · intro o start i row rbuf ho hstart hadd hget hrb0
    have ho0 : o.val = 0 := by simp at ho; omega
    have hsb := Usize.checked_mul_bv_spec o rb
    rw [hstart] at hsb
    have hstartv' : start.val = 0 := by rw [hsb.2.1, ho0]; simp
    obtain ⟨i', hadd', hiv⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := start) (y := rb) (by rw [hstartv', hrbv']; simp; scalar_tac))
    rw [hadd] at hadd'
    have hii : i = i' := by simpa using hadd'
    have hiv' : i.val = 2 := by rw [hii, hiv, hstartv', hrbv']
    have hrowv : row.val = [0#u8, 60#u8] := by
      simp only [core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get] at hget
      rw [if_pos ⟨by rw [UScalar.le_equiv, hstartv', hiv']; omega,
        by simp [Slice.length, hiv', hraw]⟩] at hget
      simp only [ok.injEq, Option.some.injEq] at hget
      rw [← hget, Slice.from_val, hstartv', hiv', hraw]
      simp [List.slice]
    have hdom : f16RowDomain row (1#usize).val := by
      refine ⟨by rw [hrowv]; simp, ?_⟩
      intro j hj
      have hj0 : j = 0 := by simp at hj; omega
      subst hj0
      have hv := f16LeHalf_wit row hrowv
      have h0 : f16Half row 0 = f16LeHalf row 0 := rfl
      simp only [h0, f16Exp, hv]
      norm_num
    obtain ⟨rb', hrow_eq, hvals⟩ :=
      dequant_f16_row_exact row 1#usize rbuf hdom (by rw [hrb0]; simp; scalar_tac)
    refine ⟨rb', ?_, ?_⟩
    · rw [dequant_row_fixed_f16]; exact hrow_eq
    · rw [hx]
      apply dequantVals_singleton
      rw [hvals, hrb0, f16WitOne_val]
      simp only [dequantVals, List.map_nil, List.nil_append, f16RowValues]
      simp [f16Value_wit row hrowv]

/-- The witness drives `matvec_refines_proved` at `n_out = 1` on a real `F16` row: the extracted
`word_matvec_fixed` succeeds with one output within `1/(2·q32)` of `rowDotReal [1.0] [1.0]`. -/
theorem matvec_f16_witness_refines :
    ∃ out : alloc.vec.Vec I64,
      word_matvec_fixed 1#u32 f16WitRaw 1#usize 1#usize f16WitVec = ok (core.result.Result.Ok out) ∧
      out.val.length = 1 ∧
      ∀ (r : Nat) (hr : r < out.val.length),
        |toReal (out.val[r]'hr) - rowDotReal f16WitVec f16WitVec| ≤ 1 / (2 * q32) := by
  obtain ⟨qk, bb, blocks, row_bytes, need, hdom⟩ := matvec_domain_f16_witness
  obtain ⟨out, hout, hlen, hb⟩ := matvec_refines_proved 1#u32 f16WitRaw 1#usize 1#usize f16WitVec
    qk bb blocks row_bytes need (fun _ => f16WitVec) hdom
  exact ⟨out, hout, by rw [hlen]; simp, hb⟩

/-- The IEEE 754-2008 binary16 value of a finite bit pattern `h`:
`(-1)^s · m · 2^-24` (subnormal, `e = 0`) or `(-1)^s · (1 + m/2^10) · 2^(e-15)` (normal). -/
noncomputable def f16Real (h : U16) : ℝ :=
  (if f16Sign h = 1 then -1 else 1) *
    (if f16Exp h = 0 then (f16Man h : ℝ) * (2 : ℝ) ^ (-24 : ℤ)
     else (1 + (f16Man h : ℝ) / 1024) * (2 : ℝ) ^ ((f16Exp h : ℤ) - 15))

/-- The Q32.32 integer over `2^32` is the binary16 real value. -/
theorem f16Value_real (h : U16) : (f16Value h : ℝ) / q32 = f16Real h := by
  unfold f16Value f16Mag f16Real q32
  have h24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by
    rw [zpow_neg]; norm_num
  have hez : ∀ e : ℕ, (2 : ℝ) ^ ((e : ℤ) - 15) = 2 ^ e / 2 ^ 15 := by
    intro e; rw [zpow_sub₀ two_ne_zero, zpow_natCast]; norm_num
  by_cases hs : f16Sign h = 1 <;> by_cases he : f16Exp h = 0 <;>
    simp only [hs, he, if_true, if_false, h24, hez] <;> push_cast <;> field_simp <;> ring

/-- **Real reading.** For a finite `f16` pattern the extracted `f16_to_fixed` returns the Q32.32
word whose real value `toReal v` is the IEEE 754-2008 binary16 value `f16Real h`. -/
theorem f16_to_fixed_real (h : U16) (he : f16Exp h ≠ 31) :
    ∃ v : I64, dequant.f16_to_fixed h = ok (.Ok v) ∧ toReal v = f16Real h := by
  obtain ⟨v, hv, hvv⟩ := f16_to_fixed_exact h he
  refine ⟨v, hv, ?_⟩
  rw [toReal, hvv, f16Value_real]

end apeiron_word
