/-
Non-vacuity witnesses for the `Q4_K` and `Q6_K` row success domains
.

`dequant_q4_k_row_spec` (`Numeric.DequantQ4K`) and `dequant_q6_k_row_spec`
(`Numeric.DequantQ6K`) assume `q4kBlockDomain raw b` / `q6kBlockDomain raw b` for every block
below `(n + 255) / 256`. This module exhibits one concrete 144-byte `Q4_K` block and one concrete
210-byte `Q6_K` block that satisfy their domain at `n = 256` (a single block), instantiates both
row laws, and evaluates one output element.

## Proved

* `q4kWitRaw` / `q6kWitRaw`: concrete blocks with `d = 1.0` (`0x3C00` little-endian), a small
  scale table and a simple code pattern.
* `q4kBlockDomain_witness` / `q6kBlockDomain_witness`: the joint domain conjuncts at block `0`.
* `q4kWitness_row` / `q6kWitness_row`: the extracted rows at `n = 256` succeed on the empty vector,
  append exactly `q4kRowValues` / `q6kRowValues`, and element `0` is `2^32` (`1.0` in Q32.32).

## Trust boundary

The row laws themselves are the kernel-checked theorems of the imported modules; this module
supplies only the joint domain witness and the closed-`ℤ` element evaluation. `decide` is used on
closed literal byte facts only, never on an extracted body.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`.
-/
import Numeric.DequantQ4K
import Numeric.DequantQ6K

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Shared helpers -/

/-- `getD` of a list appended on the right. -/
lemma getD_append_left {α : Type} (l1 l2 : List α) (i : Nat) (h : i < l1.length) (d : α) :
    (l1 ++ l2).getD i d = l1.getD i d := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_append_left h, List.getD_eq_getElem?_getD]

/-- `getD` of a list appended on the left. -/
lemma getD_append_right {α : Type} (l1 l2 : List α) (i : Nat) (h : l1.length ≤ i) (d : α) :
    (l1 ++ l2).getD i d = l2.getD (i - l1.length) d := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_append_right h, List.getD_eq_getElem?_getD]

/-- `getD` of a `Vec` value is the `getD` of the underlying `I64` list. -/
lemma dequantVals_getD (l : List I64) (i : Nat) :
    (dequantVals l).getD i 0 = (l.getD i 0#i64).val := by
  unfold dequantVals
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases h : l[i]? with
  | none => simp
  | some v => simp

/-- The empty vector's payload is `[]`. -/
lemma vec_new_val_i64 : (alloc.vec.Vec.new I64).val = ([] : List I64) := by
  simp [alloc.vec.Vec.new, alloc.vec.Vec.from_val]

/-- The half-word `[0x00, 0x3C]` is the finite binary16 pattern `0x3C00 = 15360` (`1.0`). -/
lemma u16_le_0_60 :
    (core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8])).val = 15360 := by decide

/-- `f16LeHalf` at `0` of a list beginning `a :: b :: _`. -/
lemma f16LeHalf_zero_cons (raw : Slice U8) (a b : U8) (l : List U8)
    (h : raw.val = a :: b :: l) :
    f16LeHalf raw 0 = core.num.U16.from_le_bytes (Array.make 2#usize [a, b]) := by
  unfold f16LeHalf
  congr 1
  apply Aeneas.Std.Array.ext
  simp [h]

/-- `f16LeHalf` at `2` of a list beginning `p :: q :: a :: b :: _`. -/
lemma f16LeHalf_two_cons (raw : Slice U8) (p q a b : U8) (l : List U8)
    (h : raw.val = p :: q :: a :: b :: l) :
    f16LeHalf raw 2 = core.num.U16.from_le_bytes (Array.make 2#usize [a, b]) := by
  unfold f16LeHalf
  congr 1
  apply Aeneas.Std.Array.ext
  simp [h]

/-- `f16LeHalf` at `i` from the two bytes at `i` and `i+1`. -/
lemma f16LeHalf_eq_of_getD (raw : Slice U8) (i : Nat) (a b : U8)
    (h0 : raw.val.getD i 0#u8 = a) (h1 : raw.val.getD (i + 1) 0#u8 = b) :
    f16LeHalf raw i = core.num.U16.from_le_bytes (Array.make 2#usize [a, b]) := by
  unfold f16LeHalf
  congr 1
  apply Aeneas.Std.Array.ext
  simp only [Array.make_val]
  rw [h0, h1]

/-! ## `Q4_K` witness block -/

/-- The 144-byte `Q4_K` witness block: `d = 1.0` (`0x00 0x3C`), `dmin = 0.0`, the twelve-byte
scale table `[1,0,0,0,1,0,0,0,0,0,0,0]`, and `qs = [1] ++ 127·0`. -/
def q4kWitScale : List U8 :=
  [1#u8, 0#u8, 0#u8, 0#u8, 1#u8, 0#u8, 0#u8, 0#u8, 0#u8, 0#u8, 0#u8, 0#u8]

/-- The 128 `qs` bytes of the witness block: `1` then 127 zeros. -/
def q4kWitCodes : List U8 := 1#u8 :: List.replicate 127 0#u8

/-- The witness block, laid out so the first four bytes are definitionally the `f16` pair. -/
def q4kWitBytes : List U8 :=
  0#u8 :: 60#u8 :: 0#u8 :: 0#u8 :: (q4kWitScale ++ q4kWitCodes)

theorem q4kWitBytes_length : q4kWitBytes.length = 144 := by
  simp [q4kWitBytes, q4kWitScale, q4kWitCodes]

def q4kWitRaw : Slice U8 := Slice.from q4kWitBytes (by rw [q4kWitBytes_length]; scalar_tac)

theorem q4kWitRaw_val : q4kWitRaw.val = q4kWitBytes := Slice.from_val _ _

/-- The first `f16` byte pair of the witness block is `0x3C00`. -/
lemma q4kWit_f16LeHalf_0 : (f16LeHalf q4kWitRaw 0).val = 15360 := by
  have heq : f16LeHalf q4kWitRaw 0 =
      core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) := by
    apply f16LeHalf_zero_cons
    rw [q4kWitRaw_val]
    rfl
  rw [heq, u16_le_0_60]

/-- The `dmin` byte pair of the witness block is `0x0000`. -/
lemma q4kWit_f16LeHalf_2 : (f16LeHalf q4kWitRaw 2).val = 0 := by
  have heq : f16LeHalf q4kWitRaw 2 =
      core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 0#u8]) := by
    apply f16LeHalf_two_cons
    rw [q4kWitRaw_val]
    rfl
  rw [heq]
  decide

/-- `d = 1.0 = 2^32` in Q32.32. -/
lemma q4kWit_f16Value_0 : f16Value (f16LeHalf q4kWitRaw 0) = 2 ^ 32 := by
  have hv := q4kWit_f16LeHalf_0
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

/-- `dmin = 0.0`. -/
lemma q4kWit_f16Value_2 : f16Value (f16LeHalf q4kWitRaw 2) = 0 := by
  have hv := q4kWit_f16LeHalf_2
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

/-- The scale-table byte `m` of the witness block is the raw byte at `4 + m`. -/
lemma q4kWit_scList_getD (m : Nat) (hm : m < 12) :
    ((q4kScList q4kWitRaw 0).getD m 0#u8).val = (q4kWitBytes.getD (4 + m) 0#u8).val := by
  rw [q4kScList_getD q4kWitRaw 0 m hm]
  simp only [Nat.mul_zero, Nat.zero_add]
  rw [q4kWitRaw_val]

/-- `q4CodesRef` at index `0` reads byte `16`'s low nibble. -/
lemma q4CodesRef_zero (blk : List U8) : q4CodesRef blk 0 = (blk.getD 16 0#u8).val % 16 := by
  unfold q4CodesRef
  norm_num

/-- The first decoded code of the witness block is `1`. -/
lemma q4kWit_codes_getD0 : ((q4kCodesList q4kWitRaw 0).getD 0 0#u8).val = 1 := by
  rw [q4kCodesList_getD q4kWitRaw 0 0 (by norm_num), q4CodesRef_zero]
  have hblk : (q4kBlockBytes q4kWitRaw 0).getD 16 0#u8 = 1#u8 := by
    unfold q4kBlockBytes
    rw [Nat.mul_zero, List.drop_zero, q4kWitRaw_val]
    decide
  rw [hblk]
  norm_num

/-- The first scale/min pair of the witness block has scale factor `1`. -/
lemma q4kWit_scaleMin_fst : (q4kScaleMinRef (q4kScList q4kWitRaw 0) 0).1 = 1 := by
  have h0 : ((q4kScList q4kWitRaw 0).getD 0 0#u8).val = 1 := by
    rw [q4kWit_scList_getD 0 (by norm_num)]
    decide
  have hp : q4kScaleMinRef (q4kScList q4kWitRaw 0) 0 =
      (((q4kScList q4kWitRaw 0).getD 0 0#u8).val &&& 63,
       ((q4kScList q4kWitRaw 0).getD 4 0#u8).val &&& 63) := by
    unfold q4kScaleMinRef
    rw [if_pos (by norm_num : (0 : Nat) < 4)]
  rw [hp, h0]
  decide

/-- The first scale/min pair of the witness block has min factor `1`. -/
lemma q4kWit_scaleMin_snd : (q4kScaleMinRef (q4kScList q4kWitRaw 0) 0).2 = 1 := by
  have h4 : ((q4kScList q4kWitRaw 0).getD 4 0#u8).val = 1 := by
    rw [q4kWit_scList_getD 4 (by norm_num)]
    decide
  have hp : q4kScaleMinRef (q4kScList q4kWitRaw 0) 0 =
      (((q4kScList q4kWitRaw 0).getD 0 0#u8).val &&& 63,
       ((q4kScList q4kWitRaw 0).getD 4 0#u8).val &&& 63) := by
    unfold q4kScaleMinRef
    rw [if_pos (by norm_num : (0 : Nat) < 4)]
  rw [hp, h4]
  decide

/-- The first weight of the witness `Q4_K` block is `d·sc·q = 2^32`. -/
theorem q4kWit_weight_zero : q4kWeight q4kWitRaw 0 = 2 ^ 32 := by
  have hw : q4kWeight q4kWitRaw 0 = (q4kBlockVals q4kWitRaw 0).getD 0 0 := by
    rw [show (0 : Nat) = 256 * 0 + 0 by ring]
    exact q4kWeight_eq q4kWitRaw 0 0 (by norm_num)
  rw [hw, q4kBlockVals_getD q4kWitRaw 0 0 (by norm_num)]
  rw [show (144 * 0 : Nat) = 0 by ring, show (144 * 0 + 2 : Nat) = 2 by ring]
  unfold q4kBlockVal
  rw [if_pos (by norm_num : (0 : Nat) % 64 < 32)]
  simp only [Nat.zero_div, Nat.mul_zero]
  rw [q4kWit_f16Value_0, q4kWit_f16Value_2, q4kWit_scaleMin_fst, q4kWit_scaleMin_snd,
    q4kWit_codes_getD0]
  norm_num

/-- The head of the `256`-element `q4kRowValues` list. -/
lemma q4kRowValues_head (raw : Slice U8) :
    (q4kRowValues raw 256).getD 0 0 = q4kWeight raw 0 := by
  unfold q4kRowValues
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 256)[0]? = some 0 := by decide
  rw [h]
  rfl

/-- The head of the witness `Q4_K` row value list. -/
lemma q4kWit_rowValues_getD0 : (q4kRowValues q4kWitRaw 256).getD 0 0 = 2 ^ 32 := by
  rw [q4kRowValues_head]
  exact q4kWit_weight_zero

/-- **Joint non-vacuity of `q4kBlockDomain` at block `0`.** -/
theorem q4kBlockDomain_witness : q4kBlockDomain q4kWitRaw 0#usize := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok q4kWitRaw 0#usize 144#usize (by
    rw [q4kWitRaw_val, q4kWitBytes_length]; norm_num)
  refine ⟨blk, hblk, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hslice
  · rw [q4kWitRaw_val, q4kWitBytes_length]; norm_num
  · rw [hslice, List.slice_length]; simp [q4kWitRaw_val, q4kWitBytes_length]
  · intro hc
    have hs : f16LeHalf blk 0 = f16LeHalf q4kWitRaw 0 := by
      simpa using f16LeHalf_block q4kWitRaw 0 0 blk (by simpa using hslice)
        (by rw [q4kWitRaw_val, q4kWitBytes_length]) (by norm_num)
    rw [hs] at hc
    have hv := q4kWit_f16LeHalf_0
    unfold f16Exp at hc
    rw [hv] at hc
    norm_num at hc
  · intro hc
    have hs : f16LeHalf blk 2 = f16LeHalf q4kWitRaw 2 := by
      simpa using f16LeHalf_block q4kWitRaw 0 2 blk (by simpa using hslice)
        (by rw [q4kWitRaw_val, q4kWitBytes_length]) (by norm_num)
    rw [hs] at hc
    have hv := q4kWit_f16LeHalf_2
    unfold f16Exp at hc
    rw [hv] at hc
    norm_num at hc

/-- **`Q4_K` row witness at `n = 256`.** A single concrete block: the extracted row succeeds on
the empty vector, appends exactly the `q4kRowValues`, and element `0` is `2^32`. -/
theorem q4kWitness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q4_k_row q4kWitRaw 256#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = q4kRowValues q4kWitRaw 256 ∧
      (out1.val.getD 0 0#i64).val = 2 ^ 32 := by
  have hdom : ∀ b : Usize, b.val < ((256#usize).val + 255) / 256 → q4kBlockDomain q4kWitRaw b := by
    intro b hb
    have hb1 : b.val < 1 := by
      have hR : (((256#usize).val + 255) / 256) = 1 := by decide
      rwa [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact q4kBlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_q4_k_row_spec q4kWitRaw 256#usize (alloc.vec.Vec.new I64) hdom (by
      rw [vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = q4kRowValues q4kWitRaw 256 := by
    rw [vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  · have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
    rw [dequantVals_getD] at h0
    rw [h0]
    exact q4kWit_rowValues_getD0

/-! ## `Q6_K` witness block -/

/-- The 210-byte `Q6_K` witness block: byte `0 = 1` (low code nibble), byte `128 = 2` (high code
bits), the scale table `1 :: 15·0`, and `d = 1.0` at byte `208`. -/
def q6kWitBytes : List U8 :=
  ((1#u8 :: List.replicate 127 0#u8) ++ (2#u8 :: List.replicate 63 0#u8)) ++
    ((1#u8 :: List.replicate 15 0#u8) ++ [0#u8, 60#u8])

theorem q6kWitBytes_length : q6kWitBytes.length = 210 := by
  simp only [q6kWitBytes, List.length_append, List.length_cons, List.length_replicate,
    List.length_nil]

def q6kWitRaw : Slice U8 := Slice.from q6kWitBytes (by rw [q6kWitBytes_length]; scalar_tac)

theorem q6kWitRaw_val : q6kWitRaw.val = q6kWitBytes := Slice.from_val _ _

/-! ### `Q6_K` code bounds -/

/-- Every signed 6-bit code lies in `[-32, 31]`: the assembled `Nat` is `< 64`. -/
lemma q6CodesRef_bounds (blk : List U8) (i : Nat) :
    -32 ≤ q6CodesRef blk i ∧ q6CodesRef blk i ≤ 31 := by
  have key : ∀ N : Nat, N < 64 → -32 ≤ ((N : ℤ)) - 32 ∧ ((N : ℤ)) - 32 ≤ 31 := by
    intro N hN
    have hN' : ((N : ℤ)) < 64 := by exact_mod_cast hN
    omega
  unfold q6CodesRef
  apply key
  by_cases h : i % 128 / 32 < 2
  · rw [if_pos h]
    exact q6_combine_lt' _ _ _
  · rw [if_neg h]
    exact q6_high_lt _ (U8.lt_succ_max _) _ _

/-- The raw-level codes of the witness block are bounded. -/
lemma q6kWit_code_bounds (i : Nat) :
    -32 ≤ q6kCodeRef q6kWitRaw 0 i ∧ q6kCodeRef q6kWitRaw 0 i ≤ 31 := by
  unfold q6kCodeRef
  exact q6CodesRef_bounds _ i

/-! ### `Q6_K` concrete byte reads -/

/-- The scale-table entry `192 + k` of the witness block (`k < 16`). -/
lemma q6kWitBytes_getD_scales (k : Nat) (hk : k < 16) :
    (q6kWitBytes.getD (192 + k) 0#u8).val = if k = 0 then 1 else 0 := by
  unfold q6kWitBytes
  rw [getD_append_right
        (l1 := (1#u8 :: List.replicate 127 0#u8) ++ (2#u8 :: List.replicate 63 0#u8))
        (l2 := (1#u8 :: List.replicate 15 0#u8) ++ [0#u8, 60#u8])
        (i := 192 + k) (h := by
          simp only [List.length_append, List.length_cons, List.length_replicate]; omega)
        (d := 0#u8)]
  simp only [List.length_append, List.length_cons, List.length_replicate]
  rw [show 192 + k - 192 = k by omega]
  rw [getD_append_left
        (l1 := 1#u8 :: List.replicate 15 0#u8) (l2 := [0#u8, 60#u8])
        (i := k) (h := by
          simp only [List.length_cons, List.length_replicate]; omega)
        (d := 0#u8)]
  cases k with
  | zero => simp
  | succ j =>
    rw [List.getD_cons_succ]
    have hj : j < 15 := by omega
    rw [List.getD_eq_getElem?_getD, List.getElem?_replicate, if_pos hj]
    simp

/-- The `d` byte pair of the witness block is `0x3C00`. -/
lemma q6kWit_f16LeHalf_208 : (f16LeHalf q6kWitRaw 208).val = 15360 := by
  have h0 : q6kWitRaw.val.getD 208 0#u8 = 0#u8 := by rw [q6kWitRaw_val]; decide
  have h1 : q6kWitRaw.val.getD (208 + 1) 0#u8 = 60#u8 := by rw [q6kWitRaw_val]; decide
  have heq : f16LeHalf q6kWitRaw 208 =
      core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) :=
    f16LeHalf_eq_of_getD q6kWitRaw 208 0#u8 60#u8 h0 h1
  rw [heq, u16_le_0_60]

/-- `d = 1.0 = 2^32` in Q32.32. -/
lemma q6kWit_d : q6kD q6kWitRaw 0 = 2 ^ 32 := by
  have hv := q6kWit_f16LeHalf_208
  unfold q6kD
  rw [show 210 * 0 + 208 = 208 by ring]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

/-- The first sub-block scale is `1`. -/
lemma q6kWit_scale_ref_zero : q6kScaleRef q6kWitRaw 0 0 = 1 := by
  unfold q6kScaleRef
  simp only [Nat.mul_zero, Nat.zero_add]
  rw [q6kWitRaw_val, q6kWitBytes_getD_scales 0 (by norm_num)]
  decide

/-- Every later sub-block scale is `0`. -/
lemma q6kWit_scale_ref_succ (g : Nat) (hg : g < 15) : q6kScaleRef q6kWitRaw 0 (g + 1) = 0 := by
  unfold q6kScaleRef
  simp only [Nat.mul_zero, Nat.zero_add]
  rw [q6kWitRaw_val, q6kWitBytes_getD_scales (g + 1) (by omega)]
  rw [if_neg (Nat.succ_ne_zero g)]
  decide

/-- The first decoded code of the witness block is `1`. -/
lemma q6kWit_codeRef_zero : q6kCodeRef q6kWitRaw 0 0 = 1 := by
  unfold q6kCodeRef
  rw [show 210 * 0 = 0 by ring, List.drop_zero]
  have htake : q6kWitRaw.val.take 210 = q6kWitRaw.val := by
    apply List.take_of_length_le
    rw [q6kWitRaw_val, q6kWitBytes_length]
  rw [htake, q6kWitRaw_val]
  decide

/-- The first weight of the witness `Q6_K` block is `d·scale·code = 2^32`. -/
theorem q6kWit_weight_zero : q6kWeight q6kWitRaw 0 = 2 ^ 32 := by
  have hw : q6kWeight q6kWitRaw 0 =
      q6kD q6kWitRaw 0 * q6kScaleRef q6kWitRaw 0 (0 / 16) * q6kCodeRef q6kWitRaw 0 0 := by
    rw [show (0 : Nat) = 256 * 0 + 0 by ring]
    exact q6kWeight_block q6kWitRaw 0 0 (by norm_num)
  rw [hw, show (0 / 16 : Nat) = 0 by norm_num, q6kWit_d, q6kWit_scale_ref_zero,
    q6kWit_codeRef_zero]
  norm_num

/-- The head of the `256`-element `q6kRowValues` list. -/
lemma q6kRowValues_head (raw : Slice U8) :
    (q6kRowValues raw 256).getD 0 0 = q6kWeight raw 0 := by
  unfold q6kRowValues
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 256)[0]? = some 0 := by decide
  rw [h]
  rfl

/-- The head of the witness `Q6_K` row value list. -/
lemma q6kWit_rowValues_getD0 : (q6kRowValues q6kWitRaw 256).getD 0 0 = 2 ^ 32 := by
  rw [q6kRowValues_head]
  exact q6kWit_weight_zero

/-- **Joint non-vacuity of `q6kBlockDomain` at block `0`.** -/
theorem q6kBlockDomain_witness : q6kBlockDomain q6kWitRaw 0#usize := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok q6kWitRaw 0#usize 210#usize (by
    rw [q6kWitRaw_val, q6kWitBytes_length]; norm_num)
  refine ⟨blk, hblk, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hslice
  · rw [q6kWitRaw_val, q6kWitBytes_length]; norm_num
  · rw [hslice, List.slice_length]; simp [q6kWitRaw_val, q6kWitBytes_length]
  · intro hc
    have hs : f16LeHalf blk 208 = f16LeHalf q6kWitRaw 208 := by
      simpa using f16LeHalf_slice q6kWitRaw 0 blk (by simpa using hslice)
        (by rw [q6kWitRaw_val, q6kWitBytes_length])
    rw [hs] at hc
    have hv := q6kWit_f16LeHalf_208
    unfold f16Exp at hc
    rw [hv] at hc
    norm_num at hc
  · intro g hg
    have hb : (0#usize).val = 0 := by decide
    rw [hb, q6kWit_d]
    by_cases hg0 : g = 0
    · subst hg0
      rw [q6kWit_scale_ref_zero]
      norm_num [I64.min_eq, I64.max_eq]
    · obtain ⟨k, rfl⟩ : ∃ k, g = k + 1 := ⟨g - 1, by omega⟩
      rw [q6kWit_scale_ref_succ k (by omega)]
      norm_num [I64.min_eq, I64.max_eq]
  · intro g hg l hl
    have hb : (0#usize).val = 0 := by decide
    rw [hb, q6kWit_d]
    by_cases hg0 : g = 0
    · subst hg0
      rw [q6kWit_scale_ref_zero]
      obtain ⟨h1, h2⟩ := q6kWit_code_bounds l
      constructor
      · rw [I64.min_eq]
        simp only [Nat.mul_zero, Nat.zero_add]
        rw [show (2 : ℤ) ^ 32 = 4294967296 by norm_num]
        nlinarith [h1]
      · rw [I64.max_eq]
        simp only [Nat.mul_zero, Nat.zero_add]
        rw [show (2 : ℤ) ^ 32 = 4294967296 by norm_num]
        nlinarith [h2]
    · obtain ⟨k, rfl⟩ : ∃ k, g = k + 1 := ⟨g - 1, by omega⟩
      rw [q6kWit_scale_ref_succ k (by omega)]
      norm_num [I64.min_eq, I64.max_eq]

/-- **`Q6_K` row witness at `n = 256`.** A single concrete block: the extracted row succeeds on
the empty vector, appends exactly the `q6kRowValues`, and element `0` is `2^32`. -/
theorem q6kWitness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q6_k_row q6kWitRaw 256#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = q6kRowValues q6kWitRaw 256 ∧
      (out1.val.getD 0 0#i64).val = 2 ^ 32 := by
  have hdom : ∀ b : Usize, b.val < ((256#usize).val + 255) / 256 → q6kBlockDomain q6kWitRaw b := by
    intro b hb
    have hb1 : b.val < 1 := by
      have hR : (((256#usize).val + 255) / 256) = 1 := by decide
      rwa [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact q6kBlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_q6_k_row_spec q6kWitRaw 256#usize (alloc.vec.Vec.new I64) hdom (by
      rw [vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = q6kRowValues q6kWitRaw 256 := by
    rw [vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  · have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
    rw [dequantVals_getD] at h0
    rw [h0]
    exact q6kWit_rowValues_getD0

end apeiron_word
