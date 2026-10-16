/-
Integer, index and layout refinement of the extracted `kern::gemm` bodies.

Serving path: `Engine::next_token_state` -> `step::cols` -> `kern::gemm::gemm_q8_0_group` (and
`gemm_q8_0_swiglu`). This module refines the scalar bodies that group reaches directly: `task_rows`,
`spread_limit` and `part_refused`.

Each reference is plain Lean over `Nat`: it states WHAT the body computes (a pool task shape, a
saturating `i64` accumulation budget, a layout refusal predicate), not how any loop runs.
`word_<fn>_ok` is the refinement. None of the three bodies can return `Err`/panic (their scalar
operations are saturating or guarded), so no `word_<fn>_refuses` is stated. `top` is the word
maximum (`Usize.max` at the extracted call sites), carried as a parameter so the reference
arithmetic is independent of the `usize` width.

Boundaries, stated rather than proved here: the `F32`/`F64` bodies (`pow2.f16_split`, `pow2.finish`,
`round_shift_32`), the AVX2/AVX-VNNI core (`kern.gemm.x86.rows` / `rows_swiglu`, which the
extraction emits as an `axiom`) and the `unsafe`/raw-pointer container bodies (`column_ptrs`,
`ColPtrs::put`, `SwigluPtrs`). Not yet proved, and therefore not claimed: the slice/vector layout
bodies `part_row_starts`, `split_part`, `empty_columns`, `pack`, `pack_group`.
-/
import ApeironWord

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## References (plain Lean) -/

/-- `usize` saturating addition: the sum capped at the word maximum. -/
def satAddRef (top x y : Nat) : Nat := min top (x + y)

/-- `usize` saturating multiplication: the product capped at the word maximum. -/
def satMulRef (top x y : Nat) : Nat := min (x * y) top

/-- Weight rows per pool task: `4` for a single projection of at most `1024` rows (an even row
count, so every task keeps whole row pairs), `16` otherwise. -/
def taskRowsRef (n_out parts : Nat) : Nat := if parts = 1 ∧ n_out ≤ 1024 then 4 else 16

/-- Leading zeros of a `w`-bit word: `w` for the zero word, the high zero-bit count otherwise. -/
def leadingZerosRef (w x : Nat) : Nat := if x = 0 then w else w - Nat.log 2 x - 1

/-- The `i64` accumulation budget `33 - ⌈log₂(max blocks 1)⌉`, saturating at `0`. `blocks - 1` is
`blocks.saturating_sub 1`; `w - leadingZerosRef w (blocks - 1)` is
`usize::BITS - blocks.leading_zeros()`. -/
def spreadLimitRef (w blocks : Nat) : Nat := 33 - min 33 (w - leadingZerosRef w (blocks - 1))

/-- Whether the group path refuses `part`: a family other than `Q8_0`, no rows, an extent that
leaves the file, or fewer bytes than `n_out` rows of `row_bytes`. -/
def partRefusedRef (top fileLen ggmlType nOut off len rowBytes : Nat) : Prop :=
  ggmlType ≠ 8 ∨ nOut = 0 ∨ fileLen < satAddRef top off len ∨ len < satMulRef top nOut rowBytes

/-! ## Scalar helpers -/

lemma usize_max_eq : Usize.max = 2 ^ System.Platform.numBits - 1 := by
  rw [Usize.max_def, Usize.numBits_def, UScalarTy.Usize_numBits_eq]

lemma usize_bv_lt (x : Std.Usize) : x.bv.toNat < 2 ^ System.Platform.numBits := by
  have h := x.bv.isLt
  simpa [UScalarTy.Usize_numBits_eq] using h

lemma usize_le_max (x : Std.Usize) : x.val ≤ Usize.max := by
  have hlt := usize_bv_lt x
  have hmax := usize_max_eq
  show x.bv.toNat ≤ Usize.max
  omega

lemma le_usize_max {n : Nat} (h : n ≤ 2 ^ 32 - 1) : n ≤ Usize.max := by
  have hmax := usize_max_eq
  have hbits : System.Platform.numBits = 32 ∨ System.Platform.numBits = 64 :=
    System.Platform.numBits_eq
  rcases hbits with hb | hb <;> rw [hmax, hb] <;> omega

lemma usize_val_ofNat {n : Nat} (h : n ≤ Usize.max) :
    (⟨BitVec.ofNat _ n⟩ : Std.Usize).val = n := by
  show (BitVec.ofNat _ n).toNat = n
  rw [BitVec.toNat_ofNat, UScalarTy.Usize_numBits_eq]
  apply Nat.mod_eq_of_lt
  have hmax := usize_max_eq
  have hpos : 0 < 2 ^ System.Platform.numBits := Nat.two_pow_pos _
  omega

lemma u32_val_ofNat {n : Nat} (h : n < 2 ^ 32) :
    (⟨BitVec.ofNat _ n⟩ : Std.U32).val = n := by
  show (BitVec.ofNat _ n).toNat = n
  rw [BitVec.toNat_ofNat, UScalarTy.U32_numBits_eq]
  exact Nat.mod_eq_of_lt h

lemma u32_val_lt (x : Std.U32) : x.val < 2 ^ 32 :=
  x.bv.isLt

lemma val_sat_add (x y : Std.Usize) :
    (core.num.Usize.saturating_add x y).val = satAddRef Usize.max x.val y.val := by
  unfold core.num.Usize.saturating_add UScalar.saturating_add satAddRef
  rw [UScalar.max_USize_eq, usize_val_ofNat (min_le_left _ _)]

lemma sat_mul_eq (x y : Std.Usize) :
    core.num.Usize.saturating_mul x y =
      ok (⟨BitVec.ofNat _ (satMulRef Usize.max x.val y.val)⟩ : Std.Usize) := by
  unfold core.num.Usize.saturating_mul satMulRef
  rfl

lemma val_sat_sub_one (blocks : Std.Usize) :
    (core.num.Usize.saturating_sub blocks 1#usize).val = blocks.val - 1 := by
  unfold core.num.Usize.saturating_sub UScalar.saturating_sub
  show (⟨BitVec.ofNat _ (max 0 (blocks.val - (1#usize).val))⟩ : Std.Usize).val = blocks.val - 1
  have h1 : (1#usize : Std.Usize).val = 1 := rfl
  rw [h1, usize_val_ofNat (by have := usize_le_max blocks; omega)]
  omega

lemma usize_BITS_val : (core.num.Usize.BITS).val = System.Platform.numBits := rfl

lemma max_zero_sub (a b : Nat) : max 0 (a - b) = a - min a b := by omega

lemma lzRef_le (w x : Nat) : leadingZerosRef w x ≤ w := by
  unfold leadingZerosRef
  split <;> omega

lemma lz_le (x : Std.Usize) :
    Aeneas.Std.BitVec.leadingZeros x.bv ≤ System.Platform.numBits := by
  unfold Aeneas.Std.BitVec.leadingZeros
  by_cases h : x.bv = 0
  · rw [if_pos h]
  · rw [if_neg h]
    have hlt := usize_bv_lt x
    omega

lemma lz_lt_32 (x : Std.Usize) :
    Aeneas.Std.BitVec.leadingZeros x.bv < 2 ^ 32 := by
  have h := lz_le x
  rcases System.Platform.numBits_eq with hb | hb <;> omega

lemma lz_eq_zero (x : Std.Usize) (h : x.val = 0) : x.bv = 0 :=
  BitVec.eq_of_toNat_eq (by simpa using h)

lemma lz_ne_zero (x : Std.Usize) (h : x.val ≠ 0) : ¬ (x.bv = 0) := by
  intro hc
  apply h
  have := congrArg BitVec.toNat hc
  simpa using this

lemma val_leading_zeros (x : Std.Usize) :
    (core.num.Usize.leading_zeros x).val = leadingZerosRef System.Platform.numBits x.val := by
  unfold core.num.Usize.leading_zeros
  simp only [BitVec.natCast_eq_ofNat]
  rw [u32_val_ofNat (lz_lt_32 x)]
  unfold Aeneas.Std.BitVec.leadingZeros leadingZerosRef
  by_cases hz : x.val = 0
  · rw [if_pos (lz_eq_zero x hz), if_pos hz]
  · rw [if_neg (lz_ne_zero x hz), if_neg hz]
    rfl

lemma val_sat_sub_u32 (x y : Std.U32) :
    (core.num.U32.saturating_sub x y).val = max 0 (x.val - y.val) := by
  unfold core.num.U32.saturating_sub UScalar.saturating_sub
  rw [u32_val_ofNat (by have := u32_val_lt x; omega)]

/-! ## Non-vacuity of the references -/

example : taskRowsRef 1024 1 = 4 ∧ taskRowsRef 4096 1 = 16 ∧ taskRowsRef 512 3 = 16 := by
  decide

example : spreadLimitRef 64 512 = 24 := by decide

example : ¬ partRefusedRef 1000 1000 8 4 10 200 32 := by
  unfold partRefusedRef satAddRef satMulRef
  decide

example : partRefusedRef 1000 1000 8 4 0 200 64 := by
  unfold partRefusedRef satAddRef satMulRef
  decide

/-! ## `task_rows` -/

theorem word_task_rows_ok (n_out parts : Std.Usize) :
    kern.gemm.task_rows n_out parts ⦃ r => r.val = taskRowsRef n_out.val parts.val ⦄ := by
  unfold kern.gemm.task_rows
  by_cases hp : parts = 1#usize
  · rw [if_pos hp]
    have hpv : parts.val = 1 := by rw [hp]; rfl
    by_cases hn : n_out ≤ kern.gemm.NARROW_ROWS
    · rw [if_pos hn]
      have hnv : n_out.val ≤ 1024 := by
        have hh := hn
        rw [UScalar.le_equiv, kern.gemm.NARROW_ROWS] at hh
        simpa using hh
      rw [taskRowsRef, if_pos ⟨hpv, hnv⟩]
      simp [kern.gemm.NARROW_ROWS_PER_TASK]
    · rw [if_neg hn]
      have hnv : ¬ n_out.val ≤ 1024 := by
        intro hc
        apply hn
        rw [UScalar.le_equiv, kern.gemm.NARROW_ROWS]
        simpa using hc
      rw [taskRowsRef, if_neg (by rintro ⟨_, h2⟩; exact hnv h2)]
      simp [kern.gemm.ROWS_PER_TASK]
  · rw [if_neg hp]
    have hpv : parts.val ≠ 1 := by
      intro hc
      apply hp
      exact UScalar.eq_of_val_eq (by rw [hc]; rfl)
    rw [taskRowsRef, if_neg (by rintro ⟨h1, _⟩; exact hpv h1)]
    simp [kern.gemm.ROWS_PER_TASK]

/-- Non-vacuity: the extracted body returns that shape. -/
example : kern.gemm.task_rows 1024#usize 1#usize = ok (4#usize) := by
  unfold kern.gemm.task_rows
  rw [if_pos (rfl : (1#usize) = 1#usize)]
  rw [if_pos (by rw [UScalar.le_equiv, kern.gemm.NARROW_ROWS] :
    (1024#usize) ≤ kern.gemm.NARROW_ROWS)]
  unfold kern.gemm.NARROW_ROWS_PER_TASK
  rfl

/-! ## `spread_limit` -/

theorem word_spread_limit_ok (blocks : Std.Usize) :
    kern.gemm.spread_limit blocks ⦃ r =>
      r.val = spreadLimitRef System.Platform.numBits blocks.val ⦄ := by
  apply WP.exists_imp_spec
  unfold kern.gemm.spread_limit spreadLimitRef
  simp only [lift, bind_tc_ok]
  set s := core.num.Usize.saturating_sub blocks 1#usize with hs
  have hsv : s.val = blocks.val - 1 := by rw [hs]; exact val_sat_sub_one blocks
  have hlv : (core.num.Usize.leading_zeros s).val
      = leadingZerosRef System.Platform.numBits (blocks.val - 1) := by
    rw [val_leading_zeros s, hsv]
  have hBITS : (core.num.Usize.BITS).val = System.Platform.numBits := usize_BITS_val
  have hi1le : (core.num.Usize.leading_zeros s).val ≤ System.Platform.numBits := by
    rw [hlv]; exact lzRef_le _ _
  have hnumBits_le : System.Platform.numBits ≤ 64 := by
    rcases System.Platform.numBits_eq with h | h <;> omega
  have hle : (core.num.Usize.leading_zeros s).val ≤ (core.num.Usize.BITS).val := by
    rw [hBITS]; exact hi1le
  have hKv : (⟨BitVec.ofNat _ (System.Platform.numBits
      - (core.num.Usize.leading_zeros s).val)⟩ : Std.U32).val
      = System.Platform.numBits - (core.num.Usize.leading_zeros s).val :=
    u32_val_ofNat (by have h1 := hi1le; have h2 := hnumBits_le; omega)
  have hsub : (core.num.Usize.BITS - core.num.Usize.leading_zeros s)
      = ok (⟨BitVec.ofNat _ (System.Platform.numBits
          - (core.num.Usize.leading_zeros s).val)⟩ : Std.U32) := by
    show UScalar.sub core.num.Usize.BITS (core.num.Usize.leading_zeros s) = _
    unfold UScalar.sub
    rw [if_neg (not_lt.mpr hle)]
    rfl
  rw [hsub]
  simp only [bind_tc_ok]
  refine ⟨⟨BitVec.ofNat _ (max 0 (33 - (System.Platform.numBits
      - (core.num.Usize.leading_zeros s).val)))⟩, ?_, ?_⟩
  · congr 1
    unfold core.num.U32.saturating_sub UScalar.saturating_sub
    rw [show (33#u32).val = 33 from rfl, hKv]
  · rw [u32_val_ofNat (by omega), hlv]
    exact max_zero_sub 33 _

/-! ## `part_refused` -/

theorem word_part_refused_ok (file : Slice Std.U8) (part : kern.gemm.GemmPart)
    (row_bytes : Std.Usize) :
    kern.gemm.part_refused file part row_bytes ⦃ b =>
      (b = true ↔ partRefusedRef Usize.max (Slice.len file).val part.ggml_type.val
        part.n_out.val part.at.off.val part.at.len.val row_bytes.val) ⦄ := by
  apply WP.exists_imp_spec
  unfold kern.gemm.part_refused partRefusedRef quant.GGML_TYPE_Q8_0
  simp only [lift, bind_tc_ok]
  by_cases h1 : part.ggml_type = 8#u32
  · have hc1 : ¬ ((part.ggml_type != 8#u32) = true) := by simp [h1]
    rw [if_neg hc1]
    have h1v : part.ggml_type.val = 8 := by rw [h1]; rfl
    by_cases h2 : part.n_out = 0#usize
    · rw [if_pos h2]
      refine ⟨true, rfl, ?_⟩
      have h2v : part.n_out.val = 0 := by rw [h2]; rfl
      exact ⟨fun _ => Or.inr (Or.inl h2v), fun _ => rfl⟩
    · rw [if_neg h2]
      have h2v : part.n_out.val ≠ 0 := by
        intro hh; exact h2 (UScalar.eq_of_val_eq (by rw [hh]; rfl))
      by_cases h3 : core.num.Usize.saturating_add part.at.off part.at.len > Slice.len file
      · rw [if_pos h3]
        refine ⟨true, rfl, ?_⟩
        have hC : (Slice.len file).val
            < satAddRef Usize.max part.at.off.val part.at.len.val := by
          have hh := h3
          rw [gt_iff_lt, UScalar.lt_equiv, val_sat_add] at hh
          exact hh
        exact ⟨fun _ => Or.inr (Or.inr (Or.inl hC)), fun _ => rfl⟩
      · rw [if_neg h3]
        rw [sat_mul_eq]
        simp only [bind_tc_ok]
        refine ⟨decide (part.at.len
          < (⟨BitVec.ofNat _ (satMulRef Usize.max part.n_out.val row_bytes.val)⟩ : Std.Usize)),
          rfl, ?_⟩
        simp only [decide_eq_true_eq]
        have hN : (⟨BitVec.ofNat _ (satMulRef Usize.max part.n_out.val row_bytes.val)⟩
            : Std.Usize).val = satMulRef Usize.max part.n_out.val row_bytes.val :=
          usize_val_ofNat (min_le_right _ _)
        have h3v : ¬ ((Slice.len file).val
            < satAddRef Usize.max part.at.off.val part.at.len.val) := by
          intro hh
          apply h3
          rw [gt_iff_lt, UScalar.lt_equiv, val_sat_add]
          exact hh
        rw [UScalar.lt_equiv, hN]
        constructor
        · intro h; exact Or.inr (Or.inr (Or.inr h))
        · rintro (h | h | h | h)
          · exact absurd h1v h
          · exact absurd h h2v
          · exact absurd h h3v
          · exact h
  · have hc1 : (part.ggml_type != 8#u32) = true := by simpa using h1
    rw [if_pos hc1]
    refine ⟨true, rfl, ?_⟩
    have h1v : part.ggml_type.val ≠ 8 := by
      intro hh
      apply h1
      apply UScalar.eq_of_val_eq
      rw [hh]
      rfl
    exact ⟨fun _ => Or.inl h1v, fun _ => rfl⟩

/-- Non-vacuity: the extracted body refuses a zero-row part. -/
example :
    kern.gemm.part_refused (Slice.from ([] : List Std.U8) (by simp))
        { ggml_type := 8#u32, «at» := ⟨0#usize, 0#usize⟩, n_out := 0#usize } 0#usize
      = ok true := by
  unfold kern.gemm.part_refused quant.GGML_TYPE_Q8_0
  simp only [lift, bind_tc_ok]
  rw [if_neg (by decide : ¬ ((8#u32 != 8#u32) = true))]
  simp

/-! ## Axiom cone -/

#print axioms apeiron_word.word_task_rows_ok
#print axioms apeiron_word.word_spread_limit_ok
#print axioms apeiron_word.word_part_refused_ok

end apeiron_word
