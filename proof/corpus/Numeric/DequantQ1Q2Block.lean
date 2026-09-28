/-
Split of `Numeric.DequantQ1Q2`.

This module holds the reference definitions and the extracted low-bit block readers for the `Q1_0`,
`Q2_0` and `PQ2_0` dequantizers: the block-length constants, the raw reference scale/code/weight
definitions (`q1_0ScaleRef`/`q1_0CodeByte`/`q1_0Bit`/`q1_0BitRef`/`q1_0Weight`/`q1_0RowValues` and the
`q2_0`/`pq2_0` analogues), the block readers `quant::q1_q2_block` and `quant::pq2_0_block` with their
fill-loop specifications and laws (`q1_q2_block_ok`, `pq2_0_block_ok`), the shared integer bounds
helpers, and the `Q2_0` lane/block loop scaffolding with the row law `dequant_q2_0_row_spec` and its
`dequant_row_fixed_q2_0` specialization.

The Rust bodies are `dequant::lowbit::q1_q2_block`, `dequant::lowbit::pq2_0_block` and
`dequant::lowbit::dequant_q2_0_row` in `src/engine/dequant_lowbit.rs`; the decode is exact integer
arithmetic over Q32.32, with the block `d` a finite `f16` decoded by `f16_to_fixed_exact` and every
weight an integer multiple of `d`. The `PQ2_0` and `Q1_0` row laws, the raw-level lane references and
the one-block joint domain witnesses remain in `Numeric.DequantQ1Q2`, which imports this module.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.DequantQ8_0

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Block-length constants

`Q1_0_QK = 128`, `Q1_0_BLOCK_BYTES = 2 + 128/8 = 18`, `Q2_0_QK = 64` (the same 18-byte block),
`PQ2_0_QK = 128`, `PQ2_0_BLOCK_BYTES = 2 + 128/4 = 34`. All are `irreducible` extractions. -/

/-- `quant::Q1_0_QK` is the `usize` literal `128`. -/
theorem quant_Q1_0_QK_eq : quant.Q1_0_QK = 128#usize := by with_unfolding_all rfl

/-- `quant::Q2_0_QK` is the `usize` literal `64`. -/
theorem quant_Q2_0_QK_eq : quant.Q2_0_QK = 64#usize := by with_unfolding_all rfl

/-- `quant::PQ2_0_QK` is the `usize` literal `128`. -/
theorem quant_PQ2_0_QK_eq : quant.PQ2_0_QK = 128#usize := by with_unfolding_all rfl

/-- `quant::Q1_0_QK / 8` is `16`. -/
theorem quant_Q1_0_QK_div8 : quant.Q1_0_QK / 8#usize = ok (16#usize) := by
  rw [quant_Q1_0_QK_eq]
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec (128#usize) (y := 8#usize) (by decide)
  have hq16 : q = 16#usize := by
    apply UScalar.eq_of_val_eq
    rw [hqv]; norm_num
  rw [hq, hq16]

/-- `quant::Q1_0_BLOCK_BYTES` evaluates to 18 bytes. -/
theorem quant_Q1_0_BLOCK_BYTES_eq : quant.Q1_0_BLOCK_BYTES = ok (18#usize) := by
  have h : quant.Q1_0_BLOCK_BYTES ⦃ z => z = (18#usize) ⦄ := by
    unfold quant.Q1_0_BLOCK_BYTES
    rw [quant_Q1_0_QK_div8]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- `quant::PQ2_0_QK / 4` is `32`. -/
theorem quant_PQ2_0_QK_div4 : quant.PQ2_0_QK / 4#usize = ok (32#usize) := by
  rw [quant_PQ2_0_QK_eq]
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec (128#usize) (y := 4#usize) (by decide)
  have hq32 : q = 32#usize := by
    apply UScalar.eq_of_val_eq
    rw [hqv]; norm_num
  rw [hq, hq32]

/-- `quant::PQ2_0_BLOCK_BYTES` evaluates to 34 bytes. -/
theorem quant_PQ2_0_BLOCK_BYTES_eq : quant.PQ2_0_BLOCK_BYTES = ok (34#usize) := by
  have h : quant.PQ2_0_BLOCK_BYTES ⦃ z => z = (34#usize) ⦄ := by
    unfold quant.PQ2_0_BLOCK_BYTES
    rw [quant_PQ2_0_QK_div4]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted `div_ceil n Q1_0_QK` succeeds and its value is `(n + 127)/128`. -/
theorem dequant_div_ceil_128_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.Q1_0_QK = ok blocks ∧
      blocks.val = (n.val + 127) / 128 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.Q1_0_QK) (by rw [quant_Q1_0_QK_eq]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.Q1_0_QK) (by rw [quant_Q1_0_QK_eq]; decide))
  have hqv' : q.val = n.val / 128 := by rw [hqv, quant_Q1_0_QK_eq]; simp
  have hrv' : r.val = n.val % 128 := by rw [hrv, quant_Q1_0_QK_eq]; simp
  have hq' : UScalar.div n quant.Q1_0_QK = ok q := hq
  have hr' : UScalar.rem n quant.Q1_0_QK = ok r := hr
  unfold core.num.Usize.div_ceil
  rw [hq']
  simp only [bind_tc_ok]
  rw [hr']
  simp only [bind_tc_ok]
  by_cases hr0 : r.val = 0
  · refine ⟨q, ?_, ?_⟩
    · rw [if_pos hr0]
    · rw [hqv']
      have hr0' : n.val % 128 = 0 := by rw [← hrv']; exact hr0
      omega
  · obtain ⟨q1, hq1, hq1v⟩ := WP.spec_imp_exists
      (UScalar.add_spec (x := q) (y := (UScalar.one : Usize)) (by
        rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]
      exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 128 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-- The extracted `div_ceil n PQ2_0_QK` succeeds and its value is `(n + 127)/128`. -/
theorem dequant_div_ceil_128pq_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.PQ2_0_QK = ok blocks ∧
      blocks.val = (n.val + 127) / 128 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.PQ2_0_QK) (by rw [quant_PQ2_0_QK_eq]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.PQ2_0_QK) (by rw [quant_PQ2_0_QK_eq]; decide))
  have hqv' : q.val = n.val / 128 := by rw [hqv, quant_PQ2_0_QK_eq]; simp
  have hrv' : r.val = n.val % 128 := by rw [hrv, quant_PQ2_0_QK_eq]; simp
  have hq' : UScalar.div n quant.PQ2_0_QK = ok q := hq
  have hr' : UScalar.rem n quant.PQ2_0_QK = ok r := hr
  unfold core.num.Usize.div_ceil
  rw [hq']
  simp only [bind_tc_ok]
  rw [hr']
  simp only [bind_tc_ok]
  by_cases hr0 : r.val = 0
  · refine ⟨q, ?_, ?_⟩
    · rw [if_pos hr0]
    · rw [hqv']
      have hr0' : n.val % 128 = 0 := by rw [← hrv']; exact hr0
      omega
  · obtain ⟨q1, hq1, hq1v⟩ := WP.spec_imp_exists
      (UScalar.add_spec (x := q) (y := (UScalar.one : Usize)) (by
        rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]
      exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 128 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-- The extracted `div_ceil n Q2_0_QK` succeeds and its value is `(n + 63)/64`. -/
theorem dequant_div_ceil_64_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.Q2_0_QK = ok blocks ∧
      blocks.val = (n.val + 63) / 64 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.Q2_0_QK) (by rw [quant_Q2_0_QK_eq]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.Q2_0_QK) (by rw [quant_Q2_0_QK_eq]; decide))
  have hqv' : q.val = n.val / 64 := by rw [hqv, quant_Q2_0_QK_eq]; simp
  have hrv' : r.val = n.val % 64 := by rw [hrv, quant_Q2_0_QK_eq]; simp
  have hq' : UScalar.div n quant.Q2_0_QK = ok q := hq
  have hr' : UScalar.rem n quant.Q2_0_QK = ok r := hr
  unfold core.num.Usize.div_ceil
  rw [hq']
  simp only [bind_tc_ok]
  rw [hr']
  simp only [bind_tc_ok]
  by_cases hr0 : r.val = 0
  · refine ⟨q, ?_, ?_⟩
    · rw [if_pos hr0]
    · rw [hqv']
      have hr0' : n.val % 64 = 0 := by rw [← hrv']; exact hr0
      omega
  · obtain ⟨q1, hq1, hq1v⟩ := WP.spec_imp_exists
      (UScalar.add_spec (x := q) (y := (UScalar.one : Usize)) (by
        rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]
      exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 64 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-! ## References

| symbol | byte source (relative to the row) |
|---|---|
| `q1_0ScaleRef raw b` | finite `f16` at `18·b` |
| `q1_0CodeByte raw b j` | byte at `18·b + 2 + j`, `j < 16` |
| `q1_0Bit raw b l` | bit `l%8` of that byte, `l < 128` |
| `q2_0ScaleRef raw b` | finite `f16` at `18·b` (same block) |
| `q2_0Code raw b l` | two-bit code of byte `18·b + 2 + l/4`, `l < 64` |
| `pq2_0ScaleRef raw b` | finite `f16` at `34·b` |
| `pq2_0Code raw b l` | two-bit code of byte `34·b + 2 + l/4`, `l < 128` |
-/

/-- The finite `f16` scale `d` of Q1_0/Q2_0 block `b`, read at byte `18·b`. -/
def q1_0ScaleRef (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (18 * b))

/-- The unsigned `qs` byte `j` of Q1_0/Q2_0 block `b`, byte `18·b + 2 + j`. -/
def q1_0CodeByte (raw : Slice U8) (b j : Nat) : Nat := (raw.val.getD (18 * b + 2 + j) 0#u8).val

/-- Bit `l % 8` of `qs` byte `l / 8` of a Q1_0 block. -/
def q1_0Bit (raw : Slice U8) (b l : Nat) : Nat :=
  (q1_0CodeByte raw b (l / 8) >>> (l % 8)) &&& 1

/-- Base-3-free Q1_0 sign bit of weight index `i`. -/
def q1_0BitRef (raw : Slice U8) (i : Nat) : Nat := q1_0Bit raw (i / 128) (i % 128)

/-- The exact weight `i` of a Q1_0 row: `bit ? d : -d`. -/
def q1_0Weight (raw : Slice U8) (i : Nat) : ℤ :=
  if q1_0BitRef raw i = 1 then q1_0ScaleRef raw (i / 128) else -q1_0ScaleRef raw (i / 128)

/-- The first `n` exact Q32.32 integers of a Q1_0 row. -/
def q1_0RowValues (raw : Slice U8) (n : Nat) : List ℤ := (List.range n).map (q1_0Weight raw)

/-- The finite `f16` scale `d` of Q2_0 block `b`, read at byte `18·b`. -/
def q2_0ScaleRef (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (18 * b))

/-- The unsigned `qs` byte `j` of Q2_0 block `b`, byte `18·b + 2 + j`. -/
def q2_0CodeByte (raw : Slice U8) (b j : Nat) : Nat := (raw.val.getD (18 * b + 2 + j) 0#u8).val

/-- The two-bit code `l` (`l < 64`) of a Q2_0 block: `(qs[l/4] >> 2·(l%4)) & 3`. -/
def q2_0Code (raw : Slice U8) (b l : Nat) : Nat :=
  (q2_0CodeByte raw b (l / 4) >>> (2 * (l % 4))) &&& 3

/-- The exact weight `i` of a Q2_0 row: `(code - 1)·d`. -/
def q2_0Weight (raw : Slice U8) (i : Nat) : ℤ :=
  ((q2_0Code raw (i / 64) (i % 64) : ℤ) - 1) * q2_0ScaleRef raw (i / 64)

/-- The first `n` exact Q32.32 integers of a Q2_0 row. -/
def q2_0RowValues (raw : Slice U8) (n : Nat) : List ℤ := (List.range n).map (q2_0Weight raw)

/-- The finite `f16` scale `d` of PQ2_0 block `b`, read at byte `34·b`. -/
def pq2_0ScaleRef (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (34 * b))

/-- The unsigned `qs` byte `j` of PQ2_0 block `b`, byte `34·b + 2 + j`. -/
def pq2_0CodeByte (raw : Slice U8) (b j : Nat) : Nat := (raw.val.getD (34 * b + 2 + j) 0#u8).val

/-- The two-bit code `l` (`l < 128`) of a PQ2_0 block: `(qs[l/4] >> 2·(l%4)) & 3`. -/
def pq2_0Code (raw : Slice U8) (b l : Nat) : Nat :=
  (pq2_0CodeByte raw b (l / 4) >>> (2 * (l % 4))) &&& 3

/-- The exact weight `i` of a PQ2_0 row: `(code - 1)·d`. -/
def pq2_0Weight (raw : Slice U8) (i : Nat) : ℤ :=
  ((pq2_0Code raw (i / 128) (i % 128) : ℤ) - 1) * pq2_0ScaleRef raw (i / 128)

/-- The first `n` exact Q32.32 integers of a PQ2_0 row. -/
def pq2_0RowValues (raw : Slice U8) (n : Nat) : List ℤ := (List.range n).map (pq2_0Weight raw)

lemma low_and3_lt (x : Nat) : x &&& 3 < 4 := by
  have h := Nat.and_lt_two_pow x (y := 3) (n := 2) (by norm_num)
  simpa using h

/-- A `Q1_0` sign bit is `0` or `1`. -/
lemma q1_0Bit_lt (raw : Slice U8) (b l : Nat) : q1_0Bit raw b l < 2 := by
  unfold q1_0Bit
  have h := Nat.and_lt_two_pow (q1_0CodeByte raw b (l / 8) >>> (l % 8)) (y := 1) (n := 1)
    (by norm_num)
  simpa using h

/-- A `Q2_0` two-bit code is below `4`. -/
lemma q2_0Code_lt (raw : Slice U8) (b l : Nat) : q2_0Code raw b l < 4 := by
  unfold q2_0Code
  exact low_and3_lt _

/-- A `PQ2_0` two-bit code is below `4`. -/
lemma pq2_0Code_lt (raw : Slice U8) (b l : Nat) : pq2_0Code raw b l < 4 := by
  unfold pq2_0Code
  exact low_and3_lt _

/-- `f16LeHalf` of a block window at byte 0 is the raw half-word at the window start. -/
lemma low_f16LeHalf_slice (raw blk : Slice U8) (s e : Nat)
    (h : blk.val = raw.val.slice s e) (he : e ≤ raw.val.length) (hs2e : s + 2 ≤ e) :
    f16LeHalf blk 0 = f16LeHalf raw s := by
  unfold f16LeHalf
  have h0 : blk.val.getD 0 0#u8 = raw.val.getD s 0#u8 := by
    rw [h, getD_slice raw.val s e 0 he (by omega)]
    simp
  have h1 : blk.val.getD 1 0#u8 = raw.val.getD (s + 1) 0#u8 := by
    rw [h, getD_slice raw.val s e 1 he (by omega)]
  rw [h0, h1]

/-! ## Block readers

`q1_q2_block` checks the 18-byte window, reads the `f16` at byte 0 and copies the sixteen `qs`
bytes at 2..18 into an array. `pq2_0_block` is the same with 34 bytes and thirty-two `qs` bytes.
-/

/-- Fill-loop invariant for the 16-byte `qs` array: the first `k` bytes are copied, the rest zero. -/
def q1q2FillInv (blk : Slice U8) (qs : Array U8 16#usize) (k : Nat) : Prop :=
  k ≤ 16 ∧ ∀ j, j < 16 → (qs.val.getD j 0#u8).val =
    if j < k then (blk.val.getD (2 + j) 0#u8).val else 0

/-- One `q1_q2_block` fill-loop body step: read `blk[2+k]`, write entry `k`, advance. -/
theorem q1q2Fill_body_step (blk : Slice U8) (qs : Array U8 16#usize) (k : Usize)
    (hk : k.val < 16) (hblen : 18 ≤ blk.val.length)
    (hinv : q1q2FillInv blk qs k.val) :
    ∃ (qs2 : Array U8 16#usize) (k2 : Usize),
      dequant.lowbit.q1_q2_block_loop.body blk qs k = ok (ControlFlow.cont (qs2, k2)) ∧
      q1q2FillInv blk qs2 (k.val + 1) ∧ k2.val = k.val + 1 := by
  obtain ⟨i, hadd2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 2#usize) (y := k) (by scalar_tac))
  have hi2 : i.val = 2 + k.val := by rw [hi2v]; simp
  have hib : i.val < blk.val.length := by rw [hi2]; omega
  obtain ⟨i1, hidx, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec blk i hib)
  obtain ⟨a, hupd, hau⟩ := WP.spec_imp_exists
    (Array.update_spec qs k i1 (by show k.val < qs.val.length; rw [Array.property qs]; exact hk))
  obtain ⟨k1, hadd1, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by scalar_tac))
  have hk1 : k1.val = k.val + 1 := by rw [hk1v]; simp
  have hi1val : i1.val = (blk.val.getD i.val 0#u8).val := by
    rw [hi1v]
    exact congrArg (fun x : U8 => x.val) (getD_of_lt (l := blk.val) hib).symm
  refine ⟨a, k1, ?_, ?_, hk1⟩
  · rw [dequant.lowbit.q1_q2_block_loop.body]
    rw [if_pos (by scalar_tac : k < 16#usize)]
    simp only [hadd2, hidx, hupd, hadd1, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    intro j hj
    have hqslen : qs.val.length = 16 := Array.property qs
    rw [hau, Array.set_val_eq, getD_set]
    by_cases hjk : j = k.val
    · rw [if_pos ⟨hjk, by rw [hqslen]; omega⟩, hjk, if_pos (by omega)]
      exact congrArg (fun x => (blk.val.getD x 0#u8).val) hi2 ▸ hi1val
    · rw [if_neg (fun h => hjk h.1), hinv.2 j hj]
      by_cases hlt : j < k.val
      · rw [if_pos hlt, if_pos (by omega)]
      · rw [if_neg hlt, if_neg (by omega)]

/-- The 16-byte fill loop: `loop.spec_decr_nat` with `q1q2FillInv` and measure `16 - k`.

Complexity: 16 iterations, each O(1) (one byte read, one array write); the array is 16 words, never
resized. -/
theorem q1q2Fill_loop_spec (blk : Slice U8) (qs0 : Array U8 16#usize)
    (hblen : 18 ≤ blk.val.length) (hinv0 : q1q2FillInv blk qs0 0) :
    WP.spec (dequant.lowbit.q1_q2_block_loop blk qs0 0#usize)
      (fun qs => q1q2FillInv blk qs 16) := by
  unfold dequant.lowbit.q1_q2_block_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array U8 16#usize × Usize => 16 - s.2.val)
    (inv := fun s => q1q2FillInv blk s.1 s.2.val)
  · rintro ⟨qs, k⟩ hinv'
    dsimp only at hinv' ⊢
    by_cases hlt : k.val < 16
    · obtain ⟨qs2, k2, hbody, hinv2, hk2⟩ := q1q2Fill_body_step blk qs k hlt hblen hinv'
      simp only [hbody]
      refine ⟨by simpa [hk2] using hinv2, ?_⟩
      show 16 - k2.val < 16 - k.val
      omega
    · have hnot : ¬ (k < 16#usize) := fun hc => hlt ((UScalar.lt_equiv k 16#usize).1 hc)
      have hbody : dequant.lowbit.q1_q2_block_loop.body blk qs k = ok (ControlFlow.done qs) := by
        rw [dequant.lowbit.q1_q2_block_loop.body, if_neg hnot]
      simp only [hbody]
      have hk16 : k.val = 16 := by have := hinv'.1; omega
      simpa [hk16] using hinv'
  · exact hinv0

/-- **`Q1_0`/`Q2_0` block law.** On a whole 18-byte block with finite `d`, `q1_q2_block` returns
`(d, qs)` with `d` the exact Q32.32 `f16` value and `qs` the sixteen `qs` bytes. -/
theorem q1_q2_block_ok (raw : Slice U8) (block : Usize)
    (hlen : block.val * 18 + 18 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw (18 * block.val)) ≠ 31) :
    ∃ (d : I64) (qs : Array U8 16#usize),
      dequant.lowbit.q1_q2_block raw block = ok (.Ok (d, qs)) ∧
      d.val = q1_0ScaleRef raw block.val ∧
      (∀ j, j < 16 → (qs.val.getD j 0#u8).val = q1_0CodeByte raw block.val j) := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok raw block 18#usize (by simpa using hlen)
  have hslice' : blk.val = raw.val.slice (block.val * 18) (block.val * 18 + 18) := hslice
  have hblen : 18 ≤ blk.val.length := by
    rw [hslice']
    unfold List.slice
    rw [List.length_take, List.length_drop]
    omega
  have hhalf : f16LeHalf blk 0 = f16LeHalf raw (18 * block.val) := by
    rw [show 18 * block.val = block.val * 18 by omega]
    exact low_f16LeHalf_slice raw blk (block.val * 18) (block.val * 18 + 18) hslice' (by omega) (by omega)
  have hdfin : f16Exp (f16LeHalf blk 0) ≠ 31 := by rw [hhalf]; exact hd
  obtain ⟨d, hd_eq, hdv⟩ := read_scale_ok blk 0#usize (by simp; omega) hdfin
  have hinv0 : q1q2FillInv blk (Array.repeat 16#usize 0#u8) 0 := by
    refine ⟨by norm_num, ?_⟩
    intro j hj
    rw [Array.repeat_val]
    simp only [List.getD, List.getElem?_replicate]
    rw [if_pos (by simpa using hj), if_neg (by omega : ¬ j < 0)]
    rfl
  obtain ⟨qs, hqs_eq, hqs⟩ :=
    WP.spec_imp_exists (q1q2Fill_loop_spec blk (Array.repeat 16#usize 0#u8) hblen hinv0)
  refine ⟨d, qs, ?_, ?_, ?_⟩
  · simp only [dequant.lowbit.q1_q2_block, quant_Q1_0_BLOCK_BYTES_eq, hblk, hd_eq, hqs_eq,
      bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
  · rw [hdv, show (0#usize).val = 0 by decide, hhalf]
    rfl
  · intro j hj
    have h := hqs.2 j hj
    rw [if_pos hj] at h
    rw [h, hslice', getD_slice raw.val (block.val * 18) (block.val * 18 + 18) (2 + j)
      (by omega) (by omega)]
    exact congrArg (fun x : U8 => x.val) (congrArg (raw.val.getD · 0#u8) (by omega))

/-- Fill-loop invariant for the 32-byte `qs` array: the first `k` bytes are copied, the rest zero. -/
def pq2FillInv (blk : Slice U8) (qs : Array U8 32#usize) (k : Nat) : Prop :=
  k ≤ 32 ∧ ∀ j, j < 32 → (qs.val.getD j 0#u8).val =
    if j < k then (blk.val.getD (2 + j) 0#u8).val else 0

/-- One `pq2_0_block` fill-loop body step: read `blk[2+k]`, write entry `k`, advance. -/
theorem pq2Fill_body_step (blk : Slice U8) (qs : Array U8 32#usize) (k : Usize)
    (hk : k.val < 32) (hblen : 34 ≤ blk.val.length)
    (hinv : pq2FillInv blk qs k.val) :
    ∃ (qs2 : Array U8 32#usize) (k2 : Usize),
      dequant.lowbit.pq2_0_block_loop.body blk qs k = ok (ControlFlow.cont (qs2, k2)) ∧
      pq2FillInv blk qs2 (k.val + 1) ∧ k2.val = k.val + 1 := by
  obtain ⟨i, hadd2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 2#usize) (y := k) (by scalar_tac))
  have hi2 : i.val = 2 + k.val := by rw [hi2v]; simp
  have hib : i.val < blk.val.length := by rw [hi2]; omega
  obtain ⟨i1, hidx, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec blk i hib)
  obtain ⟨a, hupd, hau⟩ := WP.spec_imp_exists
    (Array.update_spec qs k i1 (by show k.val < qs.val.length; rw [Array.property qs]; exact hk))
  obtain ⟨k1, hadd1, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by scalar_tac))
  have hk1 : k1.val = k.val + 1 := by rw [hk1v]; simp
  have hi1val : i1.val = (blk.val.getD i.val 0#u8).val := by
    rw [hi1v]
    exact congrArg (fun x : U8 => x.val) (getD_of_lt (l := blk.val) hib).symm
  refine ⟨a, k1, ?_, ?_, hk1⟩
  · rw [dequant.lowbit.pq2_0_block_loop.body]
    rw [if_pos (by scalar_tac : k < 32#usize)]
    simp only [hadd2, hidx, hupd, hadd1, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    intro j hj
    have hqslen : qs.val.length = 32 := Array.property qs
    rw [hau, Array.set_val_eq, getD_set]
    by_cases hjk : j = k.val
    · rw [if_pos ⟨hjk, by rw [hqslen]; omega⟩, hjk, if_pos (by omega)]
      exact congrArg (fun x => (blk.val.getD x 0#u8).val) hi2 ▸ hi1val
    · rw [if_neg (fun h => hjk h.1), hinv.2 j hj]
      by_cases hlt : j < k.val
      · rw [if_pos hlt, if_pos (by omega)]
      · rw [if_neg hlt, if_neg (by omega)]

/-- The 32-byte fill loop: `loop.spec_decr_nat` with `pq2FillInv` and measure `32 - k`.

Complexity: 32 iterations, each O(1) (one byte read, one array write); the array is 32 words, never
resized. -/
theorem pq2Fill_loop_spec (blk : Slice U8) (qs0 : Array U8 32#usize)
    (hblen : 34 ≤ blk.val.length) (hinv0 : pq2FillInv blk qs0 0) :
    WP.spec (dequant.lowbit.pq2_0_block_loop blk qs0 0#usize)
      (fun qs => pq2FillInv blk qs 32) := by
  unfold dequant.lowbit.pq2_0_block_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array U8 32#usize × Usize => 32 - s.2.val)
    (inv := fun s => pq2FillInv blk s.1 s.2.val)
  · rintro ⟨qs, k⟩ hinv'
    dsimp only at hinv' ⊢
    by_cases hlt : k.val < 32
    · obtain ⟨qs2, k2, hbody, hinv2, hk2⟩ := pq2Fill_body_step blk qs k hlt hblen hinv'
      simp only [hbody]
      refine ⟨by simpa [hk2] using hinv2, ?_⟩
      show 32 - k2.val < 32 - k.val
      omega
    · have hnot : ¬ (k < 32#usize) := fun hc => hlt ((UScalar.lt_equiv k 32#usize).1 hc)
      have hbody : dequant.lowbit.pq2_0_block_loop.body blk qs k = ok (ControlFlow.done qs) := by
        rw [dequant.lowbit.pq2_0_block_loop.body, if_neg hnot]
      simp only [hbody]
      have hk32 : k.val = 32 := by have := hinv'.1; omega
      simpa [hk32] using hinv'
  · exact hinv0

/-- **`PQ2_0` block law.** On a whole 34-byte block with finite `d`, `pq2_0_block` returns `(d, qs)`
with `d` the exact Q32.32 `f16` value and `qs` the thirty-two `qs` bytes. -/
theorem pq2_0_block_ok (raw : Slice U8) (block : Usize)
    (hlen : block.val * 34 + 34 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw (34 * block.val)) ≠ 31) :
    ∃ (d : I64) (qs : Array U8 32#usize),
      dequant.lowbit.pq2_0_block raw block = ok (.Ok (d, qs)) ∧
      d.val = pq2_0ScaleRef raw block.val ∧
      (∀ j, j < 32 → (qs.val.getD j 0#u8).val = pq2_0CodeByte raw block.val j) := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok raw block 34#usize (by simpa using hlen)
  have hslice' : blk.val = raw.val.slice (block.val * 34) (block.val * 34 + 34) := hslice
  have hblen : 34 ≤ blk.val.length := by
    rw [hslice']
    unfold List.slice
    rw [List.length_take, List.length_drop]
    omega
  have hhalf : f16LeHalf blk 0 = f16LeHalf raw (34 * block.val) := by
    rw [show 34 * block.val = block.val * 34 by omega]
    exact low_f16LeHalf_slice raw blk (block.val * 34) (block.val * 34 + 34) hslice' (by omega) (by omega)
  have hdfin : f16Exp (f16LeHalf blk 0) ≠ 31 := by rw [hhalf]; exact hd
  obtain ⟨d, hd_eq, hdv⟩ := read_scale_ok blk 0#usize (by simp; omega) hdfin
  have hinv0 : pq2FillInv blk (Array.repeat 32#usize 0#u8) 0 := by
    refine ⟨by norm_num, ?_⟩
    intro j hj
    rw [Array.repeat_val]
    simp only [List.getD, List.getElem?_replicate]
    rw [if_pos (by simpa using hj), if_neg (by omega : ¬ j < 0)]
    rfl
  obtain ⟨qs, hqs_eq, hqs⟩ :=
    WP.spec_imp_exists (pq2Fill_loop_spec blk (Array.repeat 32#usize 0#u8) hblen hinv0)
  refine ⟨d, qs, ?_, ?_, ?_⟩
  · simp only [dequant.lowbit.pq2_0_block, quant_PQ2_0_BLOCK_BYTES_eq, hblk, hd_eq, hqs_eq,
      bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
  · rw [hdv, show (0#usize).val = 0 by decide, hhalf]
    rfl
  · intro j hj
    have h := hqs.2 j hj
    rw [if_pos hj] at h
    rw [h, hslice', getD_slice raw.val (block.val * 34) (block.val * 34 + 34) (2 + j)
      (by omega) (by omega)]
    exact congrArg (fun x : U8 => x.val) (congrArg (raw.val.getD · 0#u8) (by omega))

/-! ## Shared helpers -/

/-- An unsigned-`u8` code cast to `i64` keeps its value. -/
lemma low_hcast_u8_I64_val (c : U8) : (UScalar.hcast .I64 c).val = (c.val : ℤ) := by
  rw [UScalar.hcast_val_eq, IScalarTy.I64_numBits_eq]
  apply Int.bmod_eq_of_le
  · norm_num
  · have h := U8.lt_succ_max c
    norm_num
    omega

/-- The product of a scale bounded by `2^48` and a code in `[-1, 3]` fits `i64`. -/
lemma low_mul_fits3 {x y : ℤ} (hx : |x| ≤ 2 ^ 48) (hy : |y| ≤ 3) :
    I64.min ≤ x * y ∧ x * y ≤ I64.max := by
  have h2 : (2 : ℤ) ^ 48 * 3 < 2 ^ 63 := by norm_num
  have hxy : |x * y| < 2 ^ 63 := by
    rw [abs_mul]
    have h : |x| * |y| ≤ (2 : ℤ) ^ 48 * 3 :=
      mul_le_mul hx hy (abs_nonneg y) (by positivity)
    linarith
  have hb := abs_lt.mp hxy
  rw [I64.min_eq, I64.max_eq]
  constructor <;> omega

/-- A scale bounded by `2^48` has a representable negation. -/
lemma low_neg_fits {x : ℤ} (hx : |x| ≤ 2 ^ 48) :
    I64.min ≤ -x ∧ -x ≤ I64.max := by
  have h2 : (2 : ℤ) ^ 48 < 2 ^ 63 := by norm_num
  have hx' : |x| < 2 ^ 63 := by linarith
  have hb := abs_lt.mp hx'
  rw [I64.min_eq, I64.max_eq]
  constructor <;> omega

lemma lt_mul_of_lt_div_ceil_64 {b n : Nat} (hb : b < (n + 63) / 64) : 64 * b < n := by
  by_contra h
  have hle : n ≤ 64 * b := not_lt.mp h
  have h1 : (n + 63) / 64 ≤ (64 * b + 63) / 64 := Nat.div_le_div_right (by omega)
  have h2 : (64 * b + 63) / 64 = b := by omega
  omega

lemma mul_block_add_min_64 (b n : Nat) (h : 64 * b ≤ n) :
    64 * b + Nat.min 64 (n - 64 * b) = Nat.min (64 * (b + 1)) n := by
  change 64 * b + min 64 (n - 64 * b) = min (64 * (b + 1)) n
  by_cases h1 : n - 64 * b ≤ 64
  · rw [Nat.min_eq_right h1, Nat.min_eq_right (by omega)]
    omega
  · have h1' : 64 < n - 64 * b := not_le.mp h1
    rw [Nat.min_eq_left (le_of_lt h1'), Nat.min_eq_left (by omega)]
    omega

lemma le_mul_div_ceil_64 (n : Nat) : n ≤ 64 * ((n + 63) / 64) := by omega

lemma lt_mul_of_lt_div_ceil_128 {b n : Nat} (hb : b < (n + 127) / 128) : 128 * b < n := by
  by_contra h
  have hle : n ≤ 128 * b := not_lt.mp h
  have h1 : (n + 127) / 128 ≤ (128 * b + 127) / 128 := Nat.div_le_div_right (by omega)
  have h2 : (128 * b + 127) / 128 = b := by omega
  omega

lemma mul_block_add_min_128 (b n : Nat) (h : 128 * b ≤ n) :
    128 * b + Nat.min 128 (n - 128 * b) = Nat.min (128 * (b + 1)) n := by
  change 128 * b + min 128 (n - 128 * b) = min (128 * (b + 1)) n
  by_cases h1 : n - 128 * b ≤ 128
  · rw [Nat.min_eq_right h1, Nat.min_eq_right (by omega)]
    omega
  · have h1' : 128 < n - 128 * b := not_le.mp h1
    rw [Nat.min_eq_left (le_of_lt h1'), Nat.min_eq_left (by omega)]
    omega

lemma le_mul_div_ceil_128 (n : Nat) : n ≤ 128 * ((n + 127) / 128) := by omega

/-! ## `dequant_q2_0_row` -/

/-- The exact weight `l` of Q2_0 block `b` (`= q2_0Weight raw (64·b + l)`). -/
def q2_0LaneVal (raw : Slice U8) (b l : Nat) : ℤ :=
  ((q2_0Code raw b l : ℤ) - 1) * q2_0ScaleRef raw b

/-- `q2_0Weight` at lane `l` of block `b`. -/
lemma q2_0Weight_lane (raw : Slice U8) (b l : Nat) (hl : l < 64) :
    q2_0Weight raw (64 * b + l) = q2_0LaneVal raw b l := by
  unfold q2_0Weight q2_0LaneVal
  rw [show (64 * b + l) / 64 = b by omega, show (64 * b + l) % 64 = l by omega]

/-- Appending the next `take` raw weights advances the Q2_0 row prefix by `take`. -/
lemma q2_0RowValues_append (raw : Slice U8) (m take : Nat) :
    q2_0RowValues raw m ++ (List.range take).map (fun j => q2_0Weight raw (m + j)) =
      q2_0RowValues raw (m + take) := by
  unfold q2_0RowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-- Lane-loop invariant: the first `lane` values of block `b` are appended, success flag set. -/
def q2_0LaneInv (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64) (take lane : Usize)
    (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ lane.val ≤ take.val ∧
  dequantVals out.val = dequantVals out0.val ++
    (List.range lane.val).map (fun l => q2_0LaneVal raw b l)

/-- One Q2_0 lane-loop body step on the success path: decode code `lane`, multiply by the scale,
push, advance. -/
theorem q2_0Lane_body_step (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64)
    (scale : I64) (qs : Array U8 16#usize) (take lane : Usize)
    (hsc : scale.val = q2_0ScaleRef raw b)
    (hcd : ∀ j, j < 16 → (qs.val.getD j 0#u8).val = q2_0CodeByte raw b j)
    (htake : take.val ≤ 64) (hlane : lane.val < take.val)
    (hmul : I64.min ≤ scale.val * ((q2_0Code raw b lane.val : ℤ) - 1) ∧
      scale.val * ((q2_0Code raw b lane.val : ℤ) - 1) ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : q2_0LaneInv raw b out0 out take lane true none) :
    ∃ (out2 : alloc.vec.Vec I64) (lane2 : Usize),
      dequant.lowbit.dequant_q2_0_row_loop0_loop0.body scale qs take out true none lane
          = ok (ControlFlow.cont (out2, true, none, lane2)) ∧
      q2_0LaneInv raw b out0 out2 take lane2 true none ∧ lane2.val = lane.val + 1 := by
  have hlane64 : lane.val < 64 := by omega
  obtain ⟨i, hdiv, hiv⟩ := UScalar.div_spec lane (y := 4#usize) (by decide)
  have hi : i.val = lane.val / 4 := by simpa using hiv
  have hi16 : i.val < 16 := by rw [hi]; omega
  obtain ⟨i1, hidx, hi1v⟩ := WP.spec_imp_exists
    (Array.index_usize_spec qs i (by show i.val < qs.val.length; rw [Array.property qs]; exact hi16))
  obtain ⟨i2, hrem, hi2v⟩ := WP.spec_imp_exists
    (UScalar.rem_spec lane (y := 4#usize) (by decide))
  have hi2 : i2.val = lane.val % 4 := by simpa using hi2v
  obtain ⟨i3, hmul2, hi3v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 2#usize) (y := i2) (by rw [hi2]; scalar_tac))
  have hi3 : i3.val = 2 * (lane.val % 4) := by rw [hi3v, hi2]; simp
  obtain ⟨i4, hshr, hi4v, -⟩ := WP.spec_imp_exists
    (UScalar.ShiftRight_spec (x := i1) (y := i3)
      (by rw [hi3]; simp only [UScalarTy.U8_numBits_eq]; omega))
  let i5 : I64 := UScalar.hcast .I64 (i4 &&& 3#u8)
  have hcast : I64.Insts.CoreConvertFromU8.from (i4 &&& 3#u8) = ok i5 := rfl
  have hi1val : i1.val = q2_0CodeByte raw b (lane.val / 4) := by
    have h1 : i1.val = (qs.val.getD i.val 0#u8).val := by
      rw [hi1v]
      exact congrArg (fun x : U8 => x.val)
        (getD_of_lt (l := qs.val) (by rw [Array.property qs]; exact hi16)).symm
    rw [h1, hi, hcd (lane.val / 4) (by omega)]
  have hcodeval : (i4 &&& 3#u8).val = q2_0Code raw b lane.val := by
    rw [UScalar.val_and]
    have h3 : (3#u8).val = 3 := rfl
    rw [h3, hi4v, hi3, hi1val]
    rfl
  have hi5val : i5.val = (q2_0Code raw b lane.val : ℤ) := by
    dsimp only [i5]
    rw [low_hcast_u8_I64_val, hcodeval]
  have hcode_lt : q2_0Code raw b lane.val < 4 := by
    unfold q2_0Code
    exact low_and3_lt _
  have hone : (1#i64).val = 1 := by decide
  obtain ⟨i6, hsub, hi6v⟩ := WP.spec_imp_exists
    (IScalar.sub_spec (x := i5) (y := 1#i64)
      (by rw [hi5val, hone]
          simp only [IScalar.min_IScalarTy_I64_eq, I64.min_eq]
          omega)
      (by rw [hi5val, hone]
          simp only [IScalar.max_IScalarTy_I64_eq, I64.max_eq]
          omega))
  have hi6val : i6.val = (q2_0Code raw b lane.val : ℤ) - 1 := by
    have h1 := hi6v
    rw [hi5val, hone] at h1
    exact h1
  obtain ⟨v, hmulv, hvval⟩ := q6k_mul_ok scale i6
    (by rw [hi6val]; exact hmul.1) (by rw [hi6val]; exact hmul.2)
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out v hcap)
  obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
  have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
  have hvval' : v.val = q2_0LaneVal raw b lane.val := by
    rw [hvval, hsc, hi6val]
    unfold q2_0LaneVal
    ring
  refine ⟨out2, lane2, ?_, ?_, hlane2⟩
  · rw [dequant.lowbit.dequant_q2_0_row_loop0_loop0.body]
    rw [if_pos (by scalar_tac : lane < take)]
    rw [if_pos (by simp : (true : Bool) = true)]
    simp only [hdiv, hidx, hrem, hmul2, hshr, hcast, hsub, hmulv, hpush, haddl, lift, bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    rw [hpushv, hlane2, dequantVals, List.map_append, ← dequantVals, hinv.2.2.2,
      List.range_succ, List.map_append]
    simp only [List.map_cons, List.map_nil, hvval', List.append_assoc]

/-- The Q2_0 lane loop: `loop.spec_decr_nat` with `q2_0LaneInv` and measure `take - lane`.

Complexity: `take` iterations, each O(1) (one byte read, five bit operations, one checked multiply,
one push); `take` words appended. -/
theorem q2_0Lane_loop_spec (raw : Slice U8) (b : Nat) (out0 : alloc.vec.Vec I64) (scale : I64)
    (qs : Array U8 16#usize) (take : Usize)
    (hsc : scale.val = q2_0ScaleRef raw b)
    (hcd : ∀ j, j < 16 → (qs.val.getD j 0#u8).val = q2_0CodeByte raw b j)
    (htake : take.val ≤ 64)
    (hmul : ∀ l, l < take.val → I64.min ≤ scale.val * ((q2_0Code raw b l : ℤ) - 1) ∧
      scale.val * ((q2_0Code raw b l : ℤ) - 1) ≤ I64.max)
    (hcap : out0.val.length + take.val ≤ Usize.max)
    (hinv0 : q2_0LaneInv raw b out0 out0 take 0#usize true none) :
    WP.spec (dequant.lowbit.dequant_q2_0_row_loop0_loop0 out0 true none scale qs take 0#usize)
      (fun p => ∃ out : alloc.vec.Vec I64,
        p = (out, true, none) ∧ q2_0LaneInv raw b out0 out take take true none) := by
  unfold dequant.lowbit.dequant_q2_0_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize =>
      take.val - s.2.2.2.val)
    (inv := fun s => q2_0LaneInv raw b out0 s.1 take s.2.2.2 s.2.1 s.2.2.1)
  · intro s hs
    rcases s with ⟨out, ok1, err, lane⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : lane.val < take.val
    · have hcap' : out.val.length < Usize.max := by
        have hlen : out.val.length = out0.val.length + lane.val := by
          have h := congrArg List.length hrec
          simp [dequantVals, List.length_map, List.length_range] at h
          omega
        omega
      obtain ⟨out1, lane1, hbody, hinv2, hlane1⟩ :=
        q2_0Lane_body_step raw b out0 out scale qs take lane hsc hcd htake hlt
          (hmul lane.val hlt) hcap' ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hlane1] using hinv2, ?_⟩
      show take.val - lane1.val < take.val - lane.val
      omega
    · have hnot : ¬ (lane < take) := fun hc => hlt ((UScalar.lt_equiv lane take).1 hc)
      have hbody : dequant.lowbit.dequant_q2_0_row_loop0_loop0.body scale qs take out true none lane
          = ok (ControlFlow.done (out, true, none)) := by
        rw [dequant.lowbit.dequant_q2_0_row_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨out, rfl, ?_⟩
      have hlane_take : lane.val = take.val := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      rw [← hlane_take]
      exact hrec
  · exact hinv0

/-- The success domain of one Q2_0 block `b`: the block bytes are present, `d` is finite, and every
weight product is inside `i64`. -/
def q2_0BlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  b.val * 18 + 18 ≤ raw.val.length ∧
  f16Exp (f16LeHalf raw (18 * b.val)) ≠ 31 ∧
  (∀ l, l < 64 → I64.min ≤ q2_0ScaleRef raw b.val * ((q2_0Code raw b.val l : ℤ) - 1) ∧
    q2_0ScaleRef raw b.val * ((q2_0Code raw b.val l : ℤ) - 1) ≤ I64.max)

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (64·block) n` raw weights. -/
def q2_0RowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ q2_0RowValues raw (Nat.min (64 * block.val) n.val)

/-- One Q2_0 block-loop body step: decode block `block`, run the lane loop, append
`take = min 64 (n - 64·block)` values and advance. -/
theorem q2_0Block_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 63) / 64)
    (hdom : q2_0BlockDomain raw block)
    (hcap : out.val.length + (n.val - 64 * block.val) ≤ Usize.max)
    (hinv : q2_0RowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.lowbit.dequant_q2_0_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      q2_0RowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨hlen, hfin, hprod⟩ := hdom
  obtain ⟨d, qs, hq, hdv, hcodes⟩ := q1_q2_block_ok raw block hlen hfin
  have hltmul : 64 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_64 (by rw [← hblocks]; exact hblock)
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.Q2_0_QK) (by rw [quant_Q2_0_QK_eq]; scalar_tac))
  have hivv' : i.val = 64 * block.val := by rw [hivv, quant_Q2_0_QK_eq]; simp; omega
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 64 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.Q2_0_QK i1
  have htakev' : take.val = Nat.min 64 (n.val - 64 * block.val) := by
    rw [htakev, hi1v', quant_Q2_0_QK_eq]; simp
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 64 := by rw [htakev']; exact min_le_left _ _
  have hcap' : out.val.length + take.val ≤ Usize.max := by
    have htake_X : take.val ≤ n.val - 64 * block.val := by rw [htakev']; exact min_le_right _ _
    omega
  have hdv2 : d.val = q2_0ScaleRef raw block.val := hdv
  have hmul : ∀ l, l < take.val → I64.min ≤ d.val * ((q2_0Code raw block.val l : ℤ) - 1) ∧
      d.val * ((q2_0Code raw block.val l : ℤ) - 1) ≤ I64.max := by
    intro l hl
    have hl64 : l < 64 := by omega
    have h := hprod l hl64
    rw [← hdv2] at h
    exact h
  have hinv0 : q2_0LaneInv raw block.val out out take 0#usize true none := by
    refine ⟨rfl, rfl, by simp, ?_⟩
    simp
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q2_0Lane_loop_spec raw block.val out d qs take hdv2 hcodes htake_le hmul hcap' hinv0)
  obtain ⟨vals1, hp1, hlane⟩ := hp
  have hloop' : dequant.lowbit.dequant_q2_0_row_loop0_loop0 out true none d qs take 0#usize
      = ok (vals1, true, none) := by rw [hloop, hp1]
  have hmin : Nat.min (64 * block.val) n.val = 64 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 64 * block.val + take.val = Nat.min (64 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min_64 block.val n.val (le_of_lt hltmul)
  have hmap : (List.range take.val).map (fun l => q2_0LaneVal raw block.val l)
      = (List.range take.val).map (fun j => q2_0Weight raw (64 * block.val + j)) := by
    apply List.map_congr_left
    intro l hl
    exact (q2_0Weight_lane raw block.val l (by have := List.mem_range.mp hl; omega)).symm
  have hinv_new : dequantVals vals1.val =
      dequantVals out0.val ++ q2_0RowValues raw (Nat.min (64 * block1.val) n.val) := by
    rw [hblock1v', hlane.2.2.2, hinv.2.2.2, hmin, List.append_assoc, hmap,
      q2_0RowValues_append, hsum]
  refine ⟨vals1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.lowbit.dequant_q2_0_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq]
  simp only [bind_tc_ok]
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  change (do
    let (out1, ok2, err1) ←
      dequant.lowbit.dequant_q2_0_row_loop0_loop0 out true none d qs take 0#usize
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, ok2, err1)))
    = ok (ControlFlow.cont (vals1, block1, true, none))
  rw [hloop']
  simp only [bind_tc_ok]
  rw [hblock1a]
  simp only [bind_tc_ok]
  rfl

/-- The Q2_0 block loop: `loop.spec_decr_nat` with `q2_0RowInv` and measure `blocks - block`, on the
success domain for every block below `blocks`.

Complexity: `blocks` iterations, each an 18-byte block read and at most 64 pushes; the measure is
`blocks - block`. -/
theorem q2_0Row_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 63) / 64)
    (hdom : ∀ b : Usize, b.val < blocks.val → q2_0BlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.lowbit.dequant_q2_0_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ q2_0RowValues raw n.val) := by
  unfold dequant.lowbit.dequant_q2_0_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => q2_0RowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, block, ok1, err⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : block.val < blocks.val
    · have hltmul : 64 * block.val < n.val :=
        lt_mul_of_lt_div_ceil_64 (by rw [← hblocks]; exact hlt)
      have hmin : Nat.min (64 * block.val) n.val = 64 * block.val :=
        Nat.min_eq_left (le_of_lt hltmul)
      have hlen_out : out.val.length = out0.val.length + 64 * block.val := by
        have h := congrArg List.length hrec
        rw [hmin] at h
        simpa [dequantVals, q2_0RowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 64 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        q2_0Block_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.lowbit.dequant_q2_0_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.lowbit.dequant_q2_0_row_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨trivial, ?_⟩
      have hblock_eq : block.val = blocks.val := by omega
      rw [hrec, hblock_eq, hblocks]
      have hminend : Nat.min (64 * ((n.val + 63) / 64)) n.val = n.val :=
        Nat.min_eq_right (le_mul_div_ceil_64 n.val)
      rw [hminend]
  · refine ⟨rfl, rfl, by simp, ?_⟩
    have h0 : Nat.min (64 * (0#usize).val) n.val = 0 := by simp
    rw [h0]
    simp [q2_0RowValues]

/-- **`Q2_0` row law.** On the block success domain, the extracted `dequant_q2_0_row` succeeds with
`Ok ()` and appends exactly the first `n` raw weights `q2_0RowValues raw n`. -/
theorem dequant_q2_0_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 63) / 64 → q2_0BlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_q2_0_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ q2_0RowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_64_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q2_0Row_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.lowbit.dequant_q2_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- `dequant_row_fixed` at `ggml_type = 42` (`GGML_TYPE_Q2_0`) is `dequant_q2_0_row`. -/
theorem dequant_row_fixed_q2_0 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 42#u32 raw n out = dequant.lowbit.dequant_q2_0_row raw n out := rfl

end

end apeiron_word
