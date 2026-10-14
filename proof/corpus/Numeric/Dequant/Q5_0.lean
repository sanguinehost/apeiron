/-
Refinement of the extracted `Q5_0` dequantizer.

The Rust bodies are `dequant::q5_0_block` and `dequant::dequant_q5_0_row` in
`src/engine/dequant.rs`; the code accessor `quant::q5_0_codes` and the scale reader `quant::q5_0_d`
are in `src/format/quant.rs`. The decode is exact integer arithmetic over Q32.32: the block's `d` is
a finite `f16` decoded exactly by `f16_to_fixed_exact`, and each weight is the checked `i64` product
`d·q_i` of the scale and one of the thirty-two signed five-bit codes.

The block width is 32 and a block is 22 bytes (`f16 d`, `qh[4]`, `qs[16]`). ggml order: output `j`
(`j < 16`) takes the low nibble of `qs[j]` and bit `j` of the little-endian `qh` word, output
`16 + j` takes the high nibble of `qs[j]` and bit `16 + j` (byte `2 + j/8`, bit `j % 8`). The
reference planes below use the same `&&&`/`|||`/`<<<`/`>>>` operations as the extracted body; the
reader `q5_0_codes` is proved here, and the row is `w_i = d_{i/32}·q_{i%32}`.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed literal
facts, never on an extracted body.
-/
import Numeric.Dequant.Q8_0
import Numeric.Dequant.K.Q6K

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Block-length constants

`quant::Q5_0_QK` is `quant::Q8_0_QK` (`32`) and `quant::Q5_0_BLOCK_BYTES` is `2 + 4 + 16 = 22`.
Both are `irreducible` extractions, so the equality is taken under full unfolding. -/

/-- `quant::Q5_0_QK` is the `usize` literal `32`. -/
theorem quant_Q5_0_QK_eq : quant.Q5_0_QK = 32#usize := by with_unfolding_all rfl

/-- `quant::Q5_0_BLOCK_BYTES` evaluates to 22 bytes. -/
theorem quant_Q5_0_BLOCK_BYTES_eq : quant.Q5_0_BLOCK_BYTES = ok (22#usize) := by
  have h : quant.Q5_0_BLOCK_BYTES ⦃ z => z = (22#usize) ⦄ := by
    unfold quant.Q5_0_BLOCK_BYTES
    rw [quant_Q5_0_QK_eq]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- `quant::Q5_0_QK / 2` is `16`: the number of `qs` lanes. -/
theorem quant_Q5_0_QK_div2 : quant.Q5_0_QK / 2#usize = ok (16#usize) := by
  rw [quant_Q5_0_QK_eq]
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec (32#usize) (y := 2#usize) (by decide)
  have hq16 : q = 16#usize := by
    apply UScalar.eq_of_val_eq
    rw [hqv]; norm_num
  rw [hq, hq16]

/-- The extracted `div_ceil n Q5_0_QK` succeeds and its value is `(n + 31)/32`. -/
theorem dequant_div_ceil_q5_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.Q5_0_QK = ok blocks ∧
      blocks.val = (n.val + 31) / 32 := by
  have h := dequant_div_ceil_32_ok n
  rw [show quant.Q8_0_QK = quant.Q5_0_QK from by with_unfolding_all rfl] at h
  exact h

/-! ## Q5_0 references

| symbol | byte source (relative to a block) |
|---|---|
| `q5_0Byte blk o` | unsigned byte `o` |
| `q5_0LoPlane blk j` | low nibble of `6 + j` plus bit `j` of `2 + j/8`, shifted to bit 4 |
| `q5_0HiPlane blk j` | high nibble of `6 + j` plus bit `j` of `4 + j/8`, shifted to bit 4 |
| `q5_0CodeRef blk i` | signed 5-bit code `i` (`i < 16` low plane, else high plane `i - 16`) |

Both planes assemble a value in `[0, 31]`; the code subtracts 16. -/

/-- The unsigned byte at offset `o` of a block list (zero past the end). -/
def q5_0Byte (blk : List U8) (o : Nat) : Nat := (blk.getD o 0#u8).val

/-- The five-bit value of the low-plane code at lane `j < 16`. -/
def q5_0LoPlane (blk : List U8) (j : Nat) : Nat :=
  (q5_0Byte blk (6 + j) &&& 15) ||| (((q5_0Byte blk (2 + j / 8) >>> (j % 8)) &&& 1) <<< 4)

/-- The five-bit value of the high-plane code at lane `j < 16`. -/
def q5_0HiPlane (blk : List U8) (j : Nat) : Nat :=
  (q5_0Byte blk (6 + j) >>> 4) ||| (((q5_0Byte blk (4 + j / 8) >>> (j % 8)) &&& 1) <<< 4)

/-- The signed five-bit code at output index `i` (`i < 32`) of a whole block. -/
def q5_0CodeRef (blk : List U8) (i : Nat) : ℤ :=
  if i < 16 then (q5_0LoPlane blk i : ℤ) - 16 else (q5_0HiPlane blk (i - 16) : ℤ) - 16

/-- The finite `f16` scale `d` of block `b`, read at byte `22·b`. -/
def q5_0D (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (22 * b))

/-- The signed five-bit code `i` of block `b`, on the block's 22-byte window. -/
def q5_0CodeRefRaw (raw : Slice U8) (b i : Nat) : ℤ :=
  q5_0CodeRef ((raw.val.drop (22 * b)).take 22) i

/-- The exact weight `j` of a Q5_0 row: `d_{j/32}·q_{j%32}`. -/
def q5_0Weight (raw : Slice U8) (j : Nat) : ℤ :=
  q5_0D raw (j / 32) * q5_0CodeRefRaw raw (j / 32) (j % 32)

/-- The first `n` exact Q32.32 integers of a Q5_0 row. -/
def q5_0RowValues (raw : Slice U8) (n : Nat) : List ℤ :=
  (List.range n).map (q5_0Weight raw)

lemma q5_0Byte_lt (blk : List U8) (o : Nat) : q5_0Byte blk o < 256 := by
  unfold q5_0Byte
  exact U8.lt_succ_max _

lemma q5_0LoPlane_lt (blk : List U8) (j : Nat) : q5_0LoPlane blk j < 32 := by
  unfold q5_0LoPlane
  rw [Nat.shiftLeft_eq]
  have ha : q5_0Byte blk (6 + j) &&& 15 < 2 ^ 5 := by
    have h := Nat.and_lt_two_pow (q5_0Byte blk (6 + j)) (n := 4) (by norm_num : (15 : Nat) < 2 ^ 4)
    omega
  have hb : (q5_0Byte blk (2 + j / 8) >>> (j % 8) &&& 1) * 2 ^ 4 < 2 ^ 5 := by
    have h := Nat.and_lt_two_pow (q5_0Byte blk (2 + j / 8) >>> (j % 8)) (n := 1)
      (by norm_num : (1 : Nat) < 2 ^ 1)
    nlinarith
  exact Nat.or_lt_two_pow ha hb

lemma q5_0HiPlane_lt (blk : List U8) (j : Nat) : q5_0HiPlane blk j < 32 := by
  unfold q5_0HiPlane
  rw [Nat.shiftLeft_eq]
  have ha : q5_0Byte blk (6 + j) >>> 4 < 2 ^ 5 := by
    have h := q5_0Byte_lt blk (6 + j)
    rw [Nat.shiftRight_eq_div_pow]
    omega
  have hb : (q5_0Byte blk (4 + j / 8) >>> (j % 8) &&& 1) * 2 ^ 4 < 2 ^ 5 := by
    have h := Nat.and_lt_two_pow (q5_0Byte blk (4 + j / 8) >>> (j % 8)) (n := 1)
      (by norm_num : (1 : Nat) < 2 ^ 1)
    nlinarith
  exact Nat.or_lt_two_pow ha hb

/-- Every assembled Q5_0 code lies in `[-16, 15]`. -/
lemma q5_0CodeRef_bounds (blk : List U8) (i : Nat) :
    -16 ≤ q5_0CodeRef blk i ∧ q5_0CodeRef blk i ≤ 15 := by
  unfold q5_0CodeRef
  split
  · have h := q5_0LoPlane_lt blk i; omega
  · have h := q5_0HiPlane_lt blk (i - 16); omega

lemma q5_0CodeRef_lo_eq (blk : List U8) (j : Nat) (hj : j < 16) :
    q5_0CodeRef blk j = (q5_0LoPlane blk j : ℤ) - 16 := by
  unfold q5_0CodeRef
  rw [if_pos hj]

lemma q5_0CodeRef_hi_eq (blk : List U8) (j : Nat) :
    q5_0CodeRef blk (16 + j) = (q5_0HiPlane blk j : ℤ) - 16 := by
  unfold q5_0CodeRef
  rw [if_neg (by omega : ¬ (16 + j < 16)), show 16 + j - 16 = j by omega]

/-- The `i8` cast of `plane - 16` is `plane - 16` for `plane < 32`. -/
lemma bmod_sub16 (n : Nat) (hn : n < 32) : Int.bmod ((n : Int) - 16) 256 = (n : Int) - 16 := by
  rw [Int.bmod_eq_emod]
  split <;> omega

/-- The signed-`i8` cast of the `i32` value `c - 16` is `c - 16` for `c < 32`. -/
lemma q5_0_cast_sub16 (c : Nat) (hc : c < 32) (q : I32) (x : I8)
    (hq : q.val = (c : ℤ) - 16) (hx : x = IScalar.cast IScalarTy.I8 q) :
    x.val = (c : ℤ) - 16 := by
  rw [hx, IScalar.cast_val_eq,
    show (2 : Nat) ^ (min IScalarTy.I8.numBits IScalarTy.I32.numBits) = 256 by
      norm_num [IScalarTy.numBits]]
  rw [hq]
  exact bmod_sub16 c hc

/-- The subtraction `x - 16` on the `i32` cast of a byte does not underflow. -/
lemma hcast_u8_I32_sub16_lower (x : U8) :
    I32.min ≤ (UScalar.hcast IScalarTy.I32 x).val - 16 := by
  rw [hcast_u8_I32_val]
  have h0 : (0 : ℤ) ≤ (x.val : ℤ) := Int.natCast_nonneg _
  norm_num [IScalar.min, IScalar.max, I32.min, I32.max, I32.numBits]
  omega

/-! ## `quant::q5_0_d`

`q5_0_d` checks `22 ≤ len blk` through `need` and reads the little-endian `f16` at byte 0. -/

/-- **`Q5_0` scale reader.** On a whole block `q5_0_d` returns the `f16` bits at byte 0. -/
theorem q5_0_d_ok (blk : Slice U8) (h22 : 22 ≤ blk.val.length) :
    quant.q5_0_d blk = ok (.Ok (f16LeHalf blk 0)) := by
  have hneed := quant_need_ok blk 22#usize (by simpa using h22)
  have hu := quant_u16le_ok blk 0#usize (by norm_num; omega)
  rw [quant.q5_0_d, quant_Q5_0_BLOCK_BYTES_eq]
  simp only [bind_tc_ok, hneed]
  simp only [core.result.Result.Insts.CoreOpsTry.branch]
  rw [hu]
  simp only [bind_tc_ok]
  rfl

/-! ## `quant::q5_0_codes`

`q5_0_codes` checks `22 ≤ len blk`, takes `qh = blk[2..6]` and `qs = blk[6..22]`, then runs the
sixteen-lane loop. Lane `j` writes output `j` (low plane) and output `16 + j` (high plane). -/

/-- After `j` lanes the accumulator holds the first `j` low-plane and high-plane codes, zero
elsewhere. -/
def q5_0CodesInv (blk : List U8) (out : Array I8 32#usize) (j : Nat) : Prop :=
  j ≤ 16 ∧ ∀ k, k < 32 → (out.val.getD k 0#i8).val =
    if k < j ∨ (16 ≤ k ∧ k < 16 + j) then q5_0CodeRef blk k else 0

/-- One lane-body step of `q5_0_codes`: assemble both five-bit planes, cast each to `i8`, write
output `j` and `16 + j`, advance. -/
theorem q5_0Codes_body_spec (blk : Slice U8) (qh qs : Slice U8) (out : Array I8 32#usize)
    (j : Usize) (hj : j.val < 16)
    (hqhl : qh.val.length = 4) (hqsl : qs.val.length = 16) (hblen : 22 ≤ blk.val.length)
    (hqh : qh.val = blk.val.slice 2 6) (hqs : qs.val = blk.val.slice 6 22)
    (hinv : q5_0CodesInv blk.val out j.val) :
    quant.q5_0_codes_loop.body 16#usize qh qs out j ⦃ r =>
      ∃ out2 j2, r = ControlFlow.cont (out2, j2) ∧
        q5_0CodesInv blk.val out2 (j.val + 1) ∧ j2.val = j.val + 1 ⦄ := by
  unfold quant.q5_0_codes_loop.body
  rw [if_pos (by scalar_tac : j < 16#usize)]
  simp only [I32.Insts.CoreConvertFromU8.from]
  step*
  · exact hcast_u8_I32_sub16_lower lo
  · exact hcast_u8_I32_sub16_lower hi
  · have hnib : nib.val = q5_0Byte blk.val (6 + j.val) := by
      have hlt : j.val < qs.val.length := by rw [hqsl]; omega
      have h1 : nib.val = (qs.val.getD j.val 0#u8).val := by
        rw [nib_post]; exact q4_getD_getElem qs j hlt
      rw [h1, hqs, getD_slice blk.val 6 22 j.val (by omega) (by omega)]
      rfl
    have hi2 : i2.val = q5_0Byte blk.val (2 + j.val / 8) := by
      have hlt : i1.val < qh.val.length := by rw [hqhl]; rw [i1_post]; omega
      have h1 : i2.val = (qh.val.getD i1.val 0#u8).val := by
        rw [i2_post]; exact q4_getD_getElem qh i1 hlt
      rw [h1, hqh, getD_slice blk.val 2 6 i1.val (by omega) (by omega), i1_post]
      rfl
    have hi6 : i6.val = q5_0Byte blk.val (4 + j.val / 8) := by
      have hlt : i5.val < qh.val.length := by rw [hqhl]; rw [i5_post, i1_post]; omega
      have h1 : i6.val = (qh.val.getD i5.val 0#u8).val := by
        rw [i6_post]; exact q4_getD_getElem qh i5 hlt
      rw [h1, hqh, getD_slice blk.val 2 6 i5.val (by omega) (by omega), i5_post, i1_post,
        show 2 + (2 + ↑j / 8) = 4 + ↑j / 8 by omega]
      rfl
    have hlo_plane : lo.val = q5_0LoPlane blk.val j.val := by
      rw [lo_post, UScalar.val_or, i8_post, UScalar.val_and, i9_post, lo_bit_post,
        UScalar.val_and, i4_post, hnib, hi2, i3_post]
      simp only [show (15#u8).val = 15 by rfl, show (1#u8).val = 1 by rfl, U8.size_eq]
      rw [shift4_mod (q5_0Byte blk.val (2 + j.val / 8) >>> (j.val % 8) &&& 1) (by
        have h := Nat.and_lt_two_pow (q5_0Byte blk.val (2 + j.val / 8) >>> (j.val % 8))
          (n := 1) (by norm_num : (1 : Nat) < 2 ^ 1)
        omega)]
      unfold q5_0LoPlane q5_0Byte
      rfl
    have hhi_plane : hi.val = q5_0HiPlane blk.val j.val := by
      rw [hi_post, UScalar.val_or, i10_post, i11_post, hi_bit_post, UScalar.val_and, i7_post,
        hi6, i3_post, hnib]
      simp only [show (1#u8).val = 1 by rfl, U8.size_eq]
      rw [shift4_mod (q5_0Byte blk.val (4 + j.val / 8) >>> (j.val % 8) &&& 1) (by
        have h := Nat.and_lt_two_pow (q5_0Byte blk.val (4 + j.val / 8) >>> (j.val % 8))
          (n := 1) (by norm_num : (1 : Nat) < 2 ^ 1)
        omega)]
      unfold q5_0HiPlane q5_0Byte
      rfl
    have hlo : i14.val = q5_0CodeRef blk.val j.val := by
      rw [q5_0CodeRef_lo_eq blk.val j.val hj]
      exact q5_0_cast_sub16 (q5_0LoPlane blk.val j.val) (q5_0LoPlane_lt blk.val j.val) i13 i14
        (by rw [i13_post, hcast_u8_I32_val, hlo_plane]) i14_post
    have hhi : i18.val = q5_0CodeRef blk.val (16 + j.val) := by
      rw [q5_0CodeRef_hi_eq blk.val j.val]
      refine q5_0_cast_sub16 (q5_0HiPlane blk.val j.val) (q5_0HiPlane_lt blk.val j.val) i16 i18
        ?_ i18_post
      rw [i16_post, hcast_u8_I32_val, hhi_plane]
    refine ⟨a, j1, rfl, ⟨by omega, ?_⟩, j1_post⟩
    intro k hk
    have hout_len : out.val.length = 32 := by
      rw [Array.property out]
      rfl
    have hlen : (out.val.set j.val i14).length = 32 := by
      rw [List.length_set, hout_len]
    rw [a_post, out1_post, Array.set_val_eq, Array.set_val_eq, getD_set]
    by_cases h1 : k = i17.val
    · rw [if_pos ⟨h1, by rw [hlen]; rw [i17_post]; omega⟩]
      rw [hhi, h1, i17_post]
      rw [if_pos (by omega)]
    · rw [if_neg (by rintro ⟨h, -⟩; exact h1 h), getD_set]
      by_cases h2 : k = j.val
      · rw [if_pos ⟨h2, by rw [hout_len]; omega⟩]
        rw [hlo, h2]
        rw [if_pos (by omega)]
      · rw [if_neg (by rintro ⟨h, -⟩; exact h2 h), hinv.2 k hk]
        by_cases hc : k < j.val ∨ (16 ≤ k ∧ k < 16 + j.val)
        · rw [if_pos hc, if_pos (by omega)]
        · rw [if_neg hc, if_neg (by
            rintro (h | ⟨h16, hlt⟩)
            · omega
            · omega)]

/-- The lane loop: `loop.spec_decr_nat` with `q5_0CodesInv` and measure `16 - j`.

Complexity: 16 iterations, each O(1) (two byte reads, five bit operations, two casts, two array
writes); the array is 32 words, never resized. -/
theorem q5_0Codes_loop_spec (blk : Slice U8) (qh qs : Slice U8) (out0 : Array I8 32#usize)
    (hqhl : qh.val.length = 4) (hqsl : qs.val.length = 16) (hblen : 22 ≤ blk.val.length)
    (hqh : qh.val = blk.val.slice 2 6) (hqs : qs.val = blk.val.slice 6 22)
    (hinv0 : q5_0CodesInv blk.val out0 0) :
    WP.spec (quant.q5_0_codes_loop 16#usize qh qs out0 0#usize)
      (fun out => q5_0CodesInv blk.val out 16) := by
  unfold quant.q5_0_codes_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array I8 32#usize × Usize => 16 - s.2.val)
    (inv := fun s => q5_0CodesInv blk.val s.1 s.2.val)
  · rintro ⟨out, j⟩ hinv'
    dsimp only at hinv' ⊢
    by_cases hlt : j.val < 16
    · obtain ⟨r, hr, out2, j2, hr2, hinv2, hj2⟩ := WP.spec_imp_exists
        (q5_0Codes_body_spec blk qh qs out j hlt hqhl hqsl hblen hqh hqs hinv')
      have hbody : quant.q5_0_codes_loop.body 16#usize qh qs out j
          = ok (ControlFlow.cont (out2, j2)) := by rw [hr, hr2]
      simp only [hbody]
      refine ⟨by simpa [hj2] using hinv2, ?_⟩
      show 16 - j2.val < 16 - j.val
      omega
    · have hnot : ¬ (j < 16#usize) := fun hc => hlt ((UScalar.lt_equiv j 16#usize).1 hc)
      have hbody : quant.q5_0_codes_loop.body 16#usize qh qs out j = ok (ControlFlow.done out) := by
        rw [quant.q5_0_codes_loop.body, if_neg hnot]
      simp only [hbody]
      have hj16 : j.val = 16 := by have := hinv'.1; omega
      simpa [hj16] using hinv'
  · exact hinv0

/-- **`Q5_0` code reader.** On a whole block (`22 ≤ len blk`) the extracted `q5_0_codes` succeeds
and returns the thirty-two signed codes `q5_0CodeRef blk i`. -/
theorem q5_0_codes_ok (blk : Slice U8) (h22 : 22 ≤ blk.val.length) :
    ∃ codes : Array I8 32#usize,
      quant.q5_0_codes blk = ok (.Ok codes) ∧
      ∀ i, i < 32 → (codes.val.getD i 0#i8).val = q5_0CodeRef blk.val i := by
  have hneed := quant_need_ok blk 22#usize (by simpa using h22)
  have hspec : quant.q5_0_codes blk ⦃ r => ∃ codes : Array I8 32#usize,
      r = (core.result.Result.Ok codes : core.result.Result (Array I8 32#usize) quant.QuantError) ∧
      ∀ i, i < 32 → (codes.val.getD i 0#i8).val = q5_0CodeRef blk.val i ⦄ := by
    unfold quant.q5_0_codes
    rw [quant_Q5_0_BLOCK_BYTES_eq, quant_Q5_0_QK_div2]
    simp only [bind_tc_ok, hneed]
    simp only [core.result.Result.Insts.CoreOpsTry.branch]
    step*
    · have hqhl : qh.val.length = 4 := by simpa using qh_post1
      have hi2 : i2 = 22#usize := UScalar.eq_of_val_eq (by rw [i2_post]; norm_num)
      have hqsl : qs.val.length = 16 := by
        have h := qs_post1
        rw [hi2] at h
        simpa [Slice.length] using h
      have hqh : qh.val = blk.val.slice 2 6 := by simpa using qh_post
      have hqs : qs.val = blk.val.slice 6 22 := by rw [qs_post, hi2]; simp [List.slice]
      refine WP.spec_bind
        (q5_0Codes_loop_spec blk qh qs (Array.repeat 32#usize 0#i8) hqhl hqsl h22 hqh hqs ?_) ?_
      · refine ⟨by norm_num, ?_⟩
        intro k hk
        rw [Array.repeat_val]
        simp only [List.getD, List.getElem?_replicate]
        rw [if_pos (by simpa using hk),
          if_neg (by omega : ¬ (k < 0 ∨ (16 ≤ k ∧ k < 16 + 0)))]
        rfl
      · intro out1 hout1
        refine ⟨out1, rfl, ?_⟩
        intro i hi
        rw [hout1.2 i hi, if_pos (by omega)]
  rcases WP.spec_imp_exists hspec with ⟨r, hr, hv⟩
  rcases hv with ⟨v, hrv, hpt⟩
  exact ⟨v, by rw [hr, hrv], hpt⟩

/-! ## `dequant::q5_0_block`

On a whole 22-byte block with finite `d` the block returns the exact Q32.32 scale and the
thirty-two signed codes. Mirrors `Numeric.DequantQ6K.q6_k_block_ok`: the block is passed by its
slice, so the references are stated over `blk`. -/

/-- **`Q5_0` block law.** Under the block window and finite `d`, `q5_0_block` returns
`(d, codes)` with `d` the exact Q32.32 `f16` value and the thirty-two signed codes
`q5_0CodeRef blk`. -/
theorem q5_0_block_ok (raw : Slice U8) (block : Usize) (blk : Slice U8)
    (hblk : dequant.block_slice raw block 22#usize = ok (.Ok blk))
    (h22 : 22 ≤ blk.val.length)
    (hd : f16Exp (f16LeHalf blk 0) ≠ 31) :
    ∃ (d : I64) (codes : Array I8 32#usize),
      dequant.q5_0_block raw block = ok (.Ok (d, codes)) ∧
      d.val = f16Value (f16LeHalf blk 0) ∧
      (∀ i, i < 32 → (codes.val.getD i 0#i8).val = q5_0CodeRef blk.val i) := by
  have hdok := q5_0_d_ok blk h22
  obtain ⟨d, hd_eq, hdv⟩ := f16_to_fixed_exact (f16LeHalf blk 0) hd
  obtain ⟨codes, hcodes_eq, hcodes⟩ := q5_0_codes_ok blk h22
  refine ⟨d, codes, ?_, hdv, hcodes⟩
  simp only [dequant.q5_0_block, quant_Q5_0_BLOCK_BYTES_eq, hblk, hdok, hd_eq, hcodes_eq,
    core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]

/-! ## Block-window byte reading

`block_slice_ok` returns `blk` with `blk.val = raw.val.slice s (s + 22)`. The `getD` of a
`List.slice` is the `getD` of the payload at the shifted index, which is what the raw-level
references need. -/

/-- `f16LeHalf` of a 22-byte window at byte 0 is the raw half-word at `s`. -/
lemma q5_0_f16LeHalf_slice (raw : Slice U8) (s : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice s (s + 22)) (he : s + 22 ≤ raw.val.length) :
    f16LeHalf blk 0 = f16LeHalf raw s := by
  unfold f16LeHalf
  have h0 : blk.val.getD 0 0#u8 = raw.val.getD s 0#u8 := by
    rw [h, getD_slice raw.val s (s + 22) 0 he (by omega)]
    simp
  have h1 : blk.val.getD 1 0#u8 = raw.val.getD (s + 1) 0#u8 := by
    rw [h, getD_slice raw.val s (s + 22) 1 he (by omega)]
  rw [h0, h1]

/-- The 22-byte window's bytes are `(drop (22·b)).take 22`, so `q5_0CodeRef` on the window is the
raw-level `q5_0CodeRefRaw`. -/
lemma q5_0CodeRef_slice (raw : Slice U8) (b : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice (22 * b) (22 * b + 22)) :
    ∀ i, q5_0CodeRef blk.val i = q5_0CodeRefRaw raw b i := by
  intro i
  have hsl : raw.val.slice (22 * b) (22 * b + 22) = (raw.val.drop (22 * b)).take 22 := by
    unfold List.slice
    congr 1
    omega
  unfold q5_0CodeRefRaw
  rw [h, hsl]

/-- The `j`-th entry of block `b`'s 32-value array is raw weight `32·b + j`. -/
lemma q5_0Weight_block (raw : Slice U8) (b j : Nat) (hj : j < 32) :
    q5_0Weight raw (32 * b + j) = q5_0D raw b * q5_0CodeRefRaw raw b j := by
  unfold q5_0Weight
  have h1 : (32 * b + j) / 32 = b := div32_mul_add b j hj
  have h2 : (32 * b + j) % 32 = j := mod32_mul_add b j hj
  rw [h1, h2]

/-- Appending the next `take` raw weights advances the row prefix by `take`. -/
lemma q5_0RowValues_append (raw : Slice U8) (m take : Nat) :
    q5_0RowValues raw m ++ (List.range take).map (fun j => q5_0Weight raw (m + j)) =
      q5_0RowValues raw (m + take) := by
  unfold q5_0RowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-! ## `dequant_q5_0_row` lane loop

The extract's inner loop pushes `scale·codes[lane]` into the output; the reference value is
`q5_0Weight raw (32·b + lane)`. `q5_0LaneInv` accumulates the first `lane` of them. -/

/-- Lane-loop invariant: the first `lane` values of block `b` are appended, with the success flag
set. -/
def q5_0LaneInv (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64) (take lane : Usize)
    (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ lane.val ≤ take.val ∧
  dequantVals out.val = dequantVals out0.val ++
    (List.range lane.val).map (fun l => q5_0Weight raw (32 * b + l))

/-- One lane-loop body step on the success path. -/
theorem q5_0Lane_body_step (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64)
    (scale : I64) (codes : Array I8 32#usize) (take lane : Usize)
    (hsc : scale.val = q5_0D raw b)
    (hcd : ∀ j, j < 32 → (codes.val.getD j 0#i8).val = q5_0CodeRefRaw raw b j)
    (htake : take.val ≤ 32) (hlane : lane.val < take.val)
    (hmul : I64.min ≤ scale.val * (codes.val.getD lane.val 0#i8).val ∧
      scale.val * (codes.val.getD lane.val 0#i8).val ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : q5_0LaneInv raw b out0 out take lane true none) :
    ∃ (out1 : alloc.vec.Vec I64) (lane1 : Usize),
      dequant.dequant_q5_0_row_loop0_loop0.body scale codes take out true none lane
          = ok (ControlFlow.cont (out1, true, none, lane1)) ∧
      q5_0LaneInv raw b out0 out1 take lane1 true none ∧ lane1.val = lane.val + 1 := by
  have hlane32 : lane.val < 32 := by omega
  obtain ⟨i, hidx, hiv⟩ := WP.spec_imp_exists
    (Array.index_usize_spec codes lane
      (by show lane.val < codes.val.length; rw [Array.property codes]; exact hlane32))
  have hival : i.val = (codes.val.getD lane.val 0#i8).val := by
    rw [hiv]
    exact congrArg (fun x : I8 => x.val)
      (getD_of_lt_i8 (l := codes.val)
        (by show lane.val < codes.val.length; rw [Array.property codes]; exact hlane32)).symm
  obtain ⟨v, hmulv, hvv⟩ := q6k_mul_ok scale (core.convert.num.FromI64I8.from i)
    (by rw [core.convert.num.FromI64I8.from_val_eq, hival]; exact hmul.1)
    (by rw [core.convert.num.FromI64I8.from_val_eq, hival]; exact hmul.2)
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out v hcap)
  obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
  have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
  have hvval : v.val = q5_0Weight raw (32 * b + lane.val) := by
    rw [hvv, core.convert.num.FromI64I8.from_val_eq, hival, hsc, hcd lane.val hlane32]
    unfold q5_0Weight
    rw [div32_mul_add b lane.val hlane32, mod32_mul_add b lane.val hlane32]
  refine ⟨out2, lane2, ?_, ?_, hlane2⟩
  · rw [dequant.dequant_q5_0_row_loop0_loop0.body]
    rw [if_pos (by scalar_tac : lane < take)]
    simp only [if_true, hidx, lift, hmulv, hpush, haddl, bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    rw [hpushv, hlane2, dequantVals, List.map_append, ← dequantVals, hinv.2.2.2,
      List.range_succ, List.map_append]
    simp only [List.map_cons, List.map_nil, hvval, List.append_assoc]

/-- The lane loop: `loop.spec_decr_nat` with `q5_0LaneInv` and measure `take - lane`, on the
no-overflow path.

Complexity: `take` iterations, each O(1) (one array read, one checked multiply, one vector push);
`take` words are appended. -/
theorem q5_0Lane_loop_spec (raw : Slice U8) (b : Nat) (out0 : alloc.vec.Vec I64) (scale : I64)
    (codes : Array I8 32#usize) (take : Usize)
    (hsc : scale.val = q5_0D raw b)
    (hcd : ∀ j, j < 32 → (codes.val.getD j 0#i8).val = q5_0CodeRefRaw raw b j)
    (htake : take.val ≤ 32)
    (hmul : ∀ l, l < take.val → I64.min ≤ scale.val * (codes.val.getD l 0#i8).val ∧
      scale.val * (codes.val.getD l 0#i8).val ≤ I64.max)
    (hcap : out0.val.length + take.val ≤ Usize.max)
    (hinv0 : q5_0LaneInv raw b out0 out0 take 0#usize true none) :
    WP.spec (dequant.dequant_q5_0_row_loop0_loop0 out0 true none scale codes take 0#usize)
      (fun p => ∃ out : alloc.vec.Vec I64,
        p = (out, true, none) ∧ q5_0LaneInv raw b out0 out take take true none) := by
  unfold dequant.dequant_q5_0_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize =>
      take.val - s.2.2.2.val)
    (inv := fun s => q5_0LaneInv raw b out0 s.1 take s.2.2.2 s.2.1 s.2.2.1)
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
        q5_0Lane_body_step raw b out0 out scale codes take lane hsc hcd htake hlt
          (hmul lane.val hlt) hcap' ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hlane1] using hinv2, ?_⟩
      show take.val - lane1.val < take.val - lane.val
      omega
    · have hnot : ¬ (lane < take) := fun hc => hlt ((UScalar.lt_equiv lane take).1 hc)
      have hbody : dequant.dequant_q5_0_row_loop0_loop0.body scale codes take out true none lane
          = ok (ControlFlow.done (out, true, none)) := by
        rw [dequant.dequant_q5_0_row_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨out, rfl, ?_⟩
      have hlane_take : lane.val = take.val := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      rw [← hlane_take]
      exact hrec
  · exact hinv0

/-! ## Block-loop scaffolding

`q5_0BlockDomain` is the per-block success domain; `q5_0RowInv` accumulates the reference values
`min (32·block) n` after `block` completed blocks. -/

/-- The success domain of one Q5_0 block `b`: the 22-byte window is the raw slice
`[22·b, 22·b+22)`, `d` is finite, and every weight product `q5_0D raw b · q5_0CodeRefRaw raw b l`
is inside `i64`. -/
def q5_0BlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  ∃ blk : Slice U8,
    dequant.block_slice raw b 22#usize = ok (.Ok blk) ∧
    blk.val = raw.val.slice (22 * b.val) (22 * b.val + 22) ∧
    22 * b.val + 22 ≤ raw.val.length ∧
    22 ≤ blk.val.length ∧
    f16Exp (f16LeHalf blk 0) ≠ 31 ∧
    (∀ l, l < 32 → I64.min ≤ q5_0D raw b.val * q5_0CodeRefRaw raw b.val l ∧
      q5_0D raw b.val * q5_0CodeRefRaw raw b.val l ≤ I64.max)

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (32·block) n` raw weights. -/
def q5_0RowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ q5_0RowValues raw (Nat.min (32 * block.val) n.val)

/-- One block-loop body step on the success domain: decode block `block`, run the lane loop,
append `take = min 32 (n - 32·block)` values and advance. -/
theorem q5_0Block_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 31) / 32)
    (hdom : q5_0BlockDomain raw block)
    (hcap : out.val.length + (n.val - 32 * block.val) ≤ Usize.max)
    (hinv : q5_0RowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.dequant_q5_0_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      q5_0RowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨blk, hblk, hslice, hbnd, h22, hfin, hprod⟩ := hdom
  obtain ⟨d, codes, hq5, hdv, hcodes⟩ := q5_0_block_ok raw block blk hblk h22 hfin
  have hdv' : d.val = q5_0D raw block.val := by
    rw [hdv, q5_0_f16LeHalf_slice raw (22 * block.val) blk hslice hbnd]
    rfl
  have hcodes' : ∀ i, i < 32 → (codes.val.getD i 0#i8).val = q5_0CodeRefRaw raw block.val i := by
    intro i hi
    rw [hcodes i hi]
    exact q5_0CodeRef_slice raw block.val blk hslice i
  have hltmul : 32 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_32 (by rw [← hblocks]; exact hblock)
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.Q5_0_QK) (by rw [quant_Q5_0_QK_eq]; scalar_tac))
  have hivv' : i.val = 32 * block.val := by
    rw [hivv, quant_Q5_0_QK_eq]; simp; omega
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 32 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.Q5_0_QK i1
  have htakev' : take.val = Nat.min 32 (n.val - 32 * block.val) := by
    rw [htakev, hi1v', quant_Q5_0_QK_eq]; simp
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 32 := by rw [htakev']; exact min_le_left _ _
  have hcap' : out.val.length + take.val ≤ Usize.max := by
    have htake_X : take.val ≤ n.val - 32 * block.val := by rw [htakev']; exact min_le_right _ _
    omega
  have hmul : ∀ l, l < take.val → I64.min ≤ d.val * (codes.val.getD l 0#i8).val ∧
      d.val * (codes.val.getD l 0#i8).val ≤ I64.max := by
    intro l hl
    have hl32 : l < 32 := by omega
    have h := hprod l hl32
    rw [← hdv', ← hcodes' l hl32] at h
    exact h
  have hinv0 : q5_0LaneInv raw block.val out out take 0#usize true none := by
    refine ⟨rfl, rfl, by simp, ?_⟩
    simp
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q5_0Lane_loop_spec raw block.val out d codes take hdv' hcodes' htake_le hmul hcap' hinv0)
  obtain ⟨vals1, hp1, hlane⟩ := hp
  have hloop' : dequant.dequant_q5_0_row_loop0_loop0 out true none d codes take 0#usize
      = ok (vals1, true, none) := by rw [hloop, hp1]
  have hmin : Nat.min (32 * block.val) n.val = 32 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 32 * block.val + take.val = Nat.min (32 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min_32 block.val n.val (le_of_lt hltmul)
  have hinv_new : dequantVals vals1.val =
      dequantVals out0.val ++ q5_0RowValues raw (Nat.min (32 * block1.val) n.val) := by
    rw [hblock1v', hlane.2.2.2, hinv.2.2.2, hmin, List.append_assoc, q5_0RowValues_append, hsum]
  refine ⟨vals1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.dequant_q5_0_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq5]
  simp only [bind_tc_ok]
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  change (do
    let (out1, ok2, err1) ← dequant.dequant_q5_0_row_loop0_loop0 out true none d codes take 0#usize
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, ok2, err1)))
    = ok (ControlFlow.cont (vals1, block1, true, none))
  rw [hloop']
  simp only [bind_tc_ok]
  rw [hblock1a]
  simp only [bind_tc_ok]
  rfl

/-- The block loop: `loop.spec_decr_nat` with `q5_0RowInv` and measure `blocks - block`, on the
success domain `q5_0BlockDomain` for every block below `blocks`.

Complexity: `blocks` iterations, each a 22-byte block read (scale, `qh`, `qs`) plus at most 32
pushes; the measure is `blocks - block`. -/
theorem q5_0Row_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 31) / 32)
    (hdom : ∀ b : Usize, b.val < blocks.val → q5_0BlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.dequant_q5_0_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ q5_0RowValues raw n.val) := by
  unfold dequant.dequant_q5_0_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => q5_0RowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, block, ok1, err⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : block.val < blocks.val
    · have hltmul : 32 * block.val < n.val :=
        lt_mul_of_lt_div_ceil_32 (by rw [← hblocks]; exact hlt)
      have hmin : Nat.min (32 * block.val) n.val = 32 * block.val :=
        Nat.min_eq_left (le_of_lt hltmul)
      have hlen_out : out.val.length = out0.val.length + 32 * block.val := by
        have h := congrArg List.length hrec
        rw [hmin] at h
        simpa [dequantVals, q5_0RowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 32 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        q5_0Block_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.dequant_q5_0_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.dequant_q5_0_row_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨trivial, ?_⟩
      have hblock_eq : block.val = blocks.val := by omega
      rw [hrec, hblock_eq, hblocks]
      have hminend : Nat.min (32 * ((n.val + 31) / 32)) n.val = n.val :=
        Nat.min_eq_right (le_mul_div_ceil_32 n.val)
      rw [hminend]
  · refine ⟨rfl, rfl, by simp, ?_⟩
    have h0 : Nat.min (32 * (0#usize).val) n.val = 0 := by simp
    rw [h0]
    simp [q5_0RowValues]

/-- **`Q5_0` row law.** On the block success domain, the extracted `dequant_q5_0_row` succeeds
with `Ok ()` and appends exactly the first `n` raw weights `q5_0RowValues raw n` to the output. -/
theorem dequant_q5_0_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 31) / 32 → q5_0BlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q5_0_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ q5_0RowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_q5_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q5_0Row_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.dequant_q5_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## `dequant_row_fixed` at `GGML_TYPE_Q5_0` -/

/-- `dequant_row_fixed` at `ggml_type = 6` (`GGML_TYPE_Q5_0`) is `dequant_q5_0_row`. -/
theorem dequant_row_fixed_q5_0 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 6#u32 raw n out = dequant.dequant_q5_0_row raw n out := rfl

/-! ## One-block joint domain witness

A concrete 22-byte Q5_0 block (`d = 1.0 = 2^32`, `qh = [1,0,0,0]`, `qs = [1] ++ 15·0`) satisfies
`q5_0BlockDomain` at block `0`, so the row law instantiates at `n = 32` and element `0` is
`d·q_0 = 2^32·1`. -/

/-- The 22-byte Q5_0 witness block. -/
def q5_0WitBytes : List U8 :=
  0#u8 :: 60#u8 :: 1#u8 :: 0#u8 :: 0#u8 :: 0#u8 :: (1#u8 :: List.replicate 15 0#u8)

theorem q5_0WitBytes_length : q5_0WitBytes.length = 22 := by
  simp [q5_0WitBytes]

def q5_0WitRaw : Slice U8 := Slice.from q5_0WitBytes (by rw [q5_0WitBytes_length]; scalar_tac)

theorem q5_0WitRaw_val : q5_0WitRaw.val = q5_0WitBytes := Slice.from_val _ _

/-- The empty vector's payload is `[]`. -/
lemma q5_0_vec_new_val : (alloc.vec.Vec.new I64).val = ([] : List I64) := by
  simp [alloc.vec.Vec.new, alloc.vec.Vec.from_val]

/-- `getD` of a `Vec` value is the `getD` of the underlying `I64` list. -/
lemma q5_0_dequantVals_getD (l : List I64) (i : Nat) :
    (dequantVals l).getD i 0 = (l.getD i 0#i64).val := by
  unfold dequantVals
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases h : l[i]? with
  | none => simp
  | some v => simp

/-- The half-word `[0x00, 0x3C]` is the finite binary16 pattern `0x3C00 = 15360` (`1.0`). -/
lemma q5_0_u16_le_0_60 :
    (core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8])).val = 15360 := by decide

/-- `f16LeHalf` at `i` from the two bytes at `i` and `i+1`. -/
lemma q5_0_f16LeHalf_eq_of_getD (raw : Slice U8) (i : Nat) (a b : U8)
    (h0 : raw.val.getD i 0#u8 = a) (h1 : raw.val.getD (i + 1) 0#u8 = b) :
    f16LeHalf raw i = core.num.U16.from_le_bytes (Array.make 2#usize [a, b]) := by
  unfold f16LeHalf
  congr 1
  apply Aeneas.Std.Array.ext
  simp only [Array.make_val]
  rw [h0, h1]

/-- The `d` byte pair of the witness block is `0x3C00`. -/
lemma q5_0Wit_f16LeHalf_0 : (f16LeHalf q5_0WitRaw 0).val = 15360 := by
  have h0 : q5_0WitRaw.val.getD 0 0#u8 = 0#u8 := by rw [q5_0WitRaw_val]; decide
  have h1 : q5_0WitRaw.val.getD (0 + 1) 0#u8 = 60#u8 := by rw [q5_0WitRaw_val]; decide
  have heq : f16LeHalf q5_0WitRaw 0 =
      core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) :=
    q5_0_f16LeHalf_eq_of_getD q5_0WitRaw 0 0#u8 60#u8 h0 h1
  rw [heq, q5_0_u16_le_0_60]

/-- `d = 1.0 = 2^32` in Q32.32. -/
lemma q5_0Wit_d : q5_0D q5_0WitRaw 0 = 2 ^ 32 := by
  have hv := q5_0Wit_f16LeHalf_0
  unfold q5_0D
  rw [show 22 * 0 = 0 by ring]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

/-- The first decoded code of the witness block is `1`. -/
lemma q5_0Wit_codeRef_zero : q5_0CodeRef q5_0WitBytes 0 = 1 := by
  have hb8 : q5_0Byte q5_0WitBytes (6 + 0) = 1 := by
    unfold q5_0Byte; rw [show 6 + 0 = 6 by ring]; decide
  have hb2 : q5_0Byte q5_0WitBytes (2 + 0 / 8) = 1 := by
    unfold q5_0Byte; rw [show 2 + 0 / 8 = 2 by ring]; decide
  rw [q5_0CodeRef_lo_eq q5_0WitBytes 0 (by norm_num)]
  unfold q5_0LoPlane
  rw [show (0 % 8 : Nat) = 0 by norm_num, hb8, hb2]
  decide

/-- The block window of the witness is the block list itself. -/
lemma q5_0CodeRefRaw_wit (i : Nat) :
    q5_0CodeRefRaw q5_0WitRaw 0 i = q5_0CodeRef q5_0WitBytes i := by
  unfold q5_0CodeRefRaw
  have htake : (q5_0WitRaw.val.drop (22 * 0)).take 22 = q5_0WitBytes := by
    rw [show (22 * 0 : Nat) = 0 by ring, List.drop_zero, q5_0WitRaw_val]
    apply List.take_of_length_le
    rw [q5_0WitBytes_length]
  rw [htake]

/-- The head of the `32`-element `q5_0RowValues` list. -/
lemma q5_0RowValues_head (raw : Slice U8) :
    (q5_0RowValues raw 32).getD 0 0 = q5_0Weight raw 0 := by
  unfold q5_0RowValues
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 32)[0]? = some 0 := by decide
  rw [h]
  rfl

/-- The head of the witness `Q5_0` row value list. -/
lemma q5_0Wit_rowValues_getD0 : (q5_0RowValues q5_0WitRaw 32).getD 0 0 = 2 ^ 32 := by
  rw [q5_0RowValues_head]
  rw [show (0 : Nat) = 32 * 0 + 0 by ring, q5_0Weight_block q5_0WitRaw 0 0 (by norm_num),
    q5_0Wit_d, q5_0CodeRefRaw_wit, q5_0Wit_codeRef_zero]
  norm_num

/-- **Joint non-vacuity of `q5_0BlockDomain` at block `0`.** -/
theorem q5_0BlockDomain_witness : q5_0BlockDomain q5_0WitRaw 0#usize := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok q5_0WitRaw 0#usize 22#usize (by
    rw [q5_0WitRaw_val, q5_0WitBytes_length]; norm_num)
  refine ⟨blk, hblk, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hslice
  · rw [q5_0WitRaw_val, q5_0WitBytes_length]; norm_num
  · rw [hslice, List.slice_length]; simp [q5_0WitRaw_val, q5_0WitBytes_length]
  · intro hc
    have hs : f16LeHalf blk 0 = f16LeHalf q5_0WitRaw 0 := by
      simpa using q5_0_f16LeHalf_slice q5_0WitRaw 0 blk (by simpa using hslice)
        (by rw [q5_0WitRaw_val, q5_0WitBytes_length])
    rw [hs] at hc
    have hv := q5_0Wit_f16LeHalf_0
    unfold f16Exp at hc
    rw [hv] at hc
    norm_num at hc
  · intro l hl
    have hb : (0#usize).val = 0 := by decide
    rw [hb, q5_0Wit_d]
    have hcode := q5_0CodeRef_bounds q5_0WitBytes l
    rw [q5_0CodeRefRaw_wit l]
    constructor
    · rw [I64.min_eq]
      rw [show (2 : ℤ) ^ 32 = 4294967296 by norm_num]
      nlinarith [hcode.1]
    · rw [I64.max_eq]
      rw [show (2 : ℤ) ^ 32 = 4294967296 by norm_num]
      nlinarith [hcode.2]

/-- **`Q5_0` row witness at `n = 32`.** A single concrete block: the extracted row succeeds on the
empty vector, appends exactly the `q5_0RowValues`, and element `0` is `2^32`. -/
theorem q5_0Witness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q5_0_row q5_0WitRaw 32#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = q5_0RowValues q5_0WitRaw 32 ∧
      (out1.val.getD 0 0#i64).val = 2 ^ 32 := by
  have hdom : ∀ b : Usize, b.val < ((32#usize).val + 31) / 32 → q5_0BlockDomain q5_0WitRaw b := by
    intro b hb
    have hb1 : b.val < 1 := by
      have hR : (((32#usize).val + 31) / 32) = 1 := by decide
      rwa [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact q5_0BlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_q5_0_row_spec q5_0WitRaw 32#usize (alloc.vec.Vec.new I64) hdom (by
      rw [q5_0_vec_new_val]; scalar_tac)
  have hvals' : dequantVals out1.val = q5_0RowValues q5_0WitRaw 32 := by
    rw [q5_0_vec_new_val] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  · have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
    rw [q5_0_dequantVals_getD] at h0
    rw [h0]
    exact q5_0Wit_rowValues_getD0

end

end apeiron_word
