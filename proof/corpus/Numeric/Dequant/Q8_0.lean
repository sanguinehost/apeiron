/-
Refinement of the extracted `Q8_0` dequantizer (`APEIRON-ALGEBRA-G3-DEQUANT-W3-b3b`).

The Rust bodies are `dequant::q8_0_block` and `dequant::dequant_q8_0_row` in
`src/engine/dequant.rs`; the code accessor `quant::q8_0_codes` is in `src/format/quant.rs`. The
decode is exact integer arithmetic over Q32.32: the block's `d` is a finite `f16` decoded exactly
by `f16_to_fixed_exact`, and each weight is the checked `i64` product `d·q_i` of the scale and one
of the thirty-two signed 8-bit codes, with no rounding.

The block width is 32: `dequant_q8_0_row` writes lane `l` of block `b` as `d_b · q_{32·b+l}`, so
weight `i` (of `n`) uses `scale_{i/32}` and `code_{i%32}`. The block-level reader is
`quant::q8_0_codes` proved here plus the scale reader `read_scale_ok` of `Numeric.DequantQ4K`.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.Dequant.Refinement
import Numeric.Dequant.K.Q4K
import Numeric.Dequant.K.Q6K

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Q8_0 references

| symbol | byte source (relative to the row) |
|---|---|
| `q8_0ScaleRef raw b` | finite `f16` at `34·b` |
| `q8_0CodeRef raw b j` | signed byte at `34·b + 2 + j`, `j < 32` |
-/

/-- The Q8_0 block length evaluates to 34 bytes. -/
theorem quant_Q8_0_BLOCK_BYTES_eq : quant.Q8_0_BLOCK_BYTES = ok (34#usize) := by
  have h : quant.Q8_0_BLOCK_BYTES ⦃ z => z = (34#usize) ⦄ := by
    unfold quant.Q8_0_BLOCK_BYTES
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The signed code written by the loop at entry `j` of a block starting at byte `o`. -/
def q8_0CodeAt (raw : Slice U8) (o j : Nat) : ℤ :=
  Int.bmod ((raw.val.getD (o + 2 + j) 0#u8).val : ℤ) 256

/-- The `j`-th signed 8-bit code of block `b`, read at byte `34·b + 2 + j`. -/
def q8_0CodeRef (raw : Slice U8) (b j : Nat) : ℤ := q8_0CodeAt raw (34 * b) j

/-- The exact Q32.32 scale `d` of block `b`. -/
def q8_0ScaleRef (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (34 * b))

/-! ## `quant::q8_0_codes`

`q8_0_codes` checks `b·34 + 34 ≤ len raw` through `checked_mul` and `need`, then fills the
thirty-two-entry array with `raw[b·34 + 2 + j] as i8`. The loop-body read is the value written at
entry `j`; the cast is the signed 8-bit reading `Int.bmod x 256`. -/

/-- After `j` steps the accumulator holds the first `j` signed codes and zeros elsewhere. -/
def q8_0CodesInv (raw : Slice U8) (o : Nat) (out : Array I8 32#usize) (j : Nat) : Prop :=
  j ≤ 32 ∧ ∀ k, k < 32 → (out.val.getD k 0#i8).val =
    if k < j then q8_0CodeAt raw o k else 0

/-- One `q8_0_codes` loop-body step: read `raw[o + 2 + j]`, signed-cast it, write entry `j`. -/
theorem q8_0Codes_body_step (raw : Slice U8) (o : Usize) (out : Array I8 32#usize) (j : Usize)
    (hbound : o.val + 2 + j.val < raw.val.length) (hj : j.val < 32)
    (hinv : q8_0CodesInv raw o.val out j.val) :
    ∃ (out2 : Array I8 32#usize) (j2 : Usize),
      quant.q8_0_codes_loop.body raw o out j = ok (ControlFlow.cont (out2, j2)) ∧
      q8_0CodesInv raw o.val out2 (j.val + 1) ∧ j2.val = j.val + 1 := by
  obtain ⟨i, hadd2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 2#usize) (by scalar_tac))
  have hi2 : i.val = o.val + 2 := by rw [hi2v]; simp
  obtain ⟨i1, haddj, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := j) (by rw [hi2]; scalar_tac))
  have hi1 : i1.val = o.val + 2 + j.val := by rw [hi1v, hi2]
  have hidxbound : i1.val < raw.val.length := by rw [hi1]; omega
  obtain ⟨i2, hidx, hi2v'⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw i1 hidxbound)
  have hi2val : i2.val = (raw.val.getD (o.val + 2 + j.val) 0#u8).val := by
    have h' : i2.val = (raw.val.getD i1.val 0#u8).val := by
      rw [hi2v']
      exact congrArg UScalar.val (getD_of_lt hidxbound).symm
    rwa [hi1] at h'
  obtain ⟨out2, hupd, hout2⟩ := WP.spec_imp_exists
    (Array.update_spec out j (UScalar.hcast .I8 i2)
      (by show j.val < out.val.length; rw [Array.property out]; exact hj))
  obtain ⟨j2, haddj1, hj2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
  have hj2 : j2.val = j.val + 1 := by rw [hj2v]; simp
  refine ⟨out2, j2, ?_, ?_, hj2⟩
  · rw [quant.q8_0_codes_loop.body]
    rw [if_pos (show j < quant.Q8_0_QK from by simp only [quant.Q8_0_QK]; scalar_tac)]
    simp only [hadd2, haddj, hidx, lift, hupd, haddj1, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    intro k hk
    have hout_prop : out.val.length = 32 := Array.property out
    rw [hout2, Array.set_val_eq, getD_set]
    by_cases hjk : k = j.val
    · rw [if_pos ⟨hjk, by rw [hout_prop]; exact hj⟩, hjk, if_pos (by omega)]
      unfold q8_0CodeAt
      rw [UScalar.hcast_val_eq, IScalarTy.I8_numBits_eq, hi2val,
        show (2 : ℕ) ^ 8 = 256 by norm_num]
    · rw [if_neg (fun h => hjk h.1), hinv.2 k hk]
      by_cases hlt : k < j.val
      · rw [if_pos hlt, if_pos (by omega)]
      · rw [if_neg hlt, if_neg (by omega)]

/-- The code loop: `loop.spec_decr_nat` with `q8_0CodesInv` and measure `32 - j`.

Complexity: 32 iterations, each O(1) (one byte read, one signed cast, one array write); the array is
32 words, never resized. -/
theorem q8_0Codes_loop_spec (raw : Slice U8) (o : Usize) (out0 : Array I8 32#usize)
    (hbound : o.val + 33 < raw.val.length) (hinv0 : q8_0CodesInv raw o.val out0 0) :
    WP.spec (quant.q8_0_codes_loop raw o out0 0#usize)
      (fun out => q8_0CodesInv raw o.val out 32) := by
  unfold quant.q8_0_codes_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array I8 32#usize × Usize => 32 - s.2.val)
    (inv := fun s => q8_0CodesInv raw o.val s.1 s.2.val)
  · rintro ⟨out, j⟩ hinv'
    dsimp only at hinv' ⊢
    by_cases hlt : j.val < 32
    · obtain ⟨out2, j2, hbody, hinv2, hj2⟩ :=
        q8_0Codes_body_step raw o out j (by omega) hlt hinv'
      simp only [hbody]
      refine ⟨by simpa [hj2] using hinv2, ?_⟩
      show 32 - j2.val < 32 - j.val
      omega
    · have hnot : ¬ (j < 32#usize) := fun hc => hlt ((UScalar.lt_equiv j 32#usize).1 hc)
      have hbody : quant.q8_0_codes_loop.body raw o out j = ok (ControlFlow.done out) := by
        rw [quant.q8_0_codes_loop.body]
        rw [if_neg (show ¬ (j < quant.Q8_0_QK) from by simp only [quant.Q8_0_QK]; exact hnot)]
      simp only [hbody]
      have hj32 : j.val = 32 := by have := hinv'.1; omega
      simpa [hj32] using hinv'
  · exact hinv0

/-- **`Q8_0` code reader.** On a whole block (`b·34 + 34 ≤ len raw`) the extracted `q8_0_codes`
succeeds and returns the thirty-two signed bytes at `34·b + 2 + j`. -/
theorem q8_0_codes_ok (raw : Slice U8) (b : Usize)
    (hb : b.val * 34 + 34 ≤ raw.val.length) :
    ∃ codes : Array I8 32#usize,
      quant.q8_0_codes raw b = ok (.Ok codes) ∧
      ∀ j, j < 32 → (codes.val.getD j 0#i8).val = q8_0CodeRef raw b.val j := by
  obtain ⟨o, ho, hov⟩ := dequant_usize_checked_mul_some b 34#usize (by
    have := raw.property; scalar_tac)
  have hov' : o.val = b.val * 34 := hov
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 34#usize) (by rw [hov']; scalar_tac))
  have hi1v' : i1.val = o.val + 34 := by rw [hi1v]; simp
  have hneed : quant.need raw i1 = ok (.Ok ()) :=
    quant_need_ok raw i1 (by rw [hi1v', hov']; exact hb)
  have hbound : o.val + 33 < raw.val.length := by rw [hov']; omega
  have hinv0 : q8_0CodesInv raw o.val (Array.repeat 32#usize 0#i8) 0 := by
    refine ⟨by norm_num, ?_⟩
    intro k hk
    rw [Array.repeat_val]
    simp only [List.getD, List.getElem?_replicate]
    rw [if_pos (by simpa using hk), if_neg (by omega : ¬ k < 0)]
    rfl
  obtain ⟨codes, hloop, hcodes⟩ := WP.spec_imp_exists
    (q8_0Codes_loop_spec raw o (Array.repeat 32#usize 0#i8) hbound hinv0)
  refine ⟨codes, ?_, ?_⟩
  · rw [quant.q8_0_codes, quant_Q8_0_BLOCK_BYTES_eq]
    simp only [bind_tc_ok, lift, ho, core.option.Option.ok_or_some,
      core.result.Result.Insts.CoreOpsTry.branch, hi1, hneed, hloop, bind_tc_ok]
  · intro j hj
    have h := hcodes.2 j hj
    rw [if_pos hj, hov'] at h
    simpa [q8_0CodeRef, Nat.mul_comm] using h

/-! ## `dequant::q8_0_block`

On a whole block with finite `d` the block returns the exact Q32.32 scale and the thirty-two signed
codes. The scale is read through `read_scale_ok` at byte `block·34`; the codes through
`q8_0_codes_ok`. -/

/-- **`Q8_0` block law.** Under `block·34 + 34 ≤ len raw` and finite `d`, `q8_0_block` returns
`(d, codes)` with `d` the exact Q32.32 `f16` value and the thirty-two signed codes
`q8_0CodeRef raw block`. -/
theorem q8_0_block_ok (raw : Slice U8) (block : Usize)
    (hlen : block.val * 34 + 34 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw (34 * block.val)) ≠ 31) :
    ∃ (d : I64) (codes : Array I8 32#usize),
      dequant.q8_0_block raw block = ok (.Ok (d, codes)) ∧
      d.val = q8_0ScaleRef raw block.val ∧
      (∀ j, j < 32 → (codes.val.getD j 0#i8).val = q8_0CodeRef raw block.val j) := by
  obtain ⟨i1, hmul, hi1v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := 34#usize) (by scalar_tac))
  have hi1v' : i1.val = block.val * 34 := by rw [hi1v]; simp
  obtain ⟨d, hscale_eq, hdv⟩ := read_scale_ok raw i1 (by rw [hi1v']; omega) (by
    rw [hi1v']; simpa [Nat.mul_comm] using hd)
  obtain ⟨codes, hcodes_eq, hcodes⟩ := q8_0_codes_ok raw block hlen
  have hdv' : d.val = q8_0ScaleRef raw block.val := by
    rw [hdv, hi1v']
    unfold q8_0ScaleRef
    rw [Nat.mul_comm]
  refine ⟨d, codes, ?_, hdv', hcodes⟩
  · rw [dequant.q8_0_block, quant_Q8_0_BLOCK_BYTES_eq]
    simp only [bind_tc_ok, hmul, hscale_eq, hcodes_eq,
      core.result.Result.Insts.CoreOpsTry.branch]

/-! ## The Q8_0 row values -/

/-- The exact weight `j` of a Q8_0 row: `d_{j/32}·q_{j%32}`. -/
def q8_0Weight (raw : Slice U8) (j : Nat) : ℤ :=
  q8_0ScaleRef raw (j / 32) * q8_0CodeRef raw (j / 32) (j % 32)

/-- The first `n` exact Q32.32 integers of a Q8_0 row. -/
def q8_0RowValues (raw : Slice U8) (n : Nat) : List ℤ :=
  (List.range n).map (q8_0Weight raw)

lemma div32_mul_add (g l : Nat) (hl : l < 32) : (32 * g + l) / 32 = g := by omega
lemma mod32_mul_add (g l : Nat) (hl : l < 32) : (32 * g + l) % 32 = l := by omega
lemma eq_of_div32 (j g : Nat) (h : j / 32 = g) : j = 32 * g + j % 32 := by
  have := Nat.div_add_mod j 32
  omega

/-! ## `dequant_q8_0_row` lane loop

The extract's inner loop pushes `scale·codes[lane]` into the output; the reference value is
`q8_0Weight raw (32·b + lane)`. `q8_0LaneInv` accumulates the first `lane` of them. -/

/-- Lane-loop invariant: the first `lane` values of block `b` are appended, with the success flag
set. -/
def q8_0LaneInv (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64) (take lane : Usize)
    (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ lane.val ≤ take.val ∧
  dequantVals out.val = dequantVals out0.val ++
    (List.range lane.val).map (fun l => q8_0Weight raw (32 * b + l))

/-- One lane-loop body step on the success path. -/
theorem q8_0Lane_body_step (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64)
    (scale : I64) (codes : Array I8 32#usize) (take lane : Usize)
    (hsc : scale.val = q8_0ScaleRef raw b)
    (hcd : ∀ j, j < 32 → (codes.val.getD j 0#i8).val = q8_0CodeRef raw b j)
    (htake : take.val ≤ 32) (hlane : lane.val < take.val)
    (hmul : I64.min ≤ scale.val * (codes.val.getD lane.val 0#i8).val ∧
      scale.val * (codes.val.getD lane.val 0#i8).val ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : q8_0LaneInv raw b out0 out take lane true none) :
    ∃ (out1 : alloc.vec.Vec I64) (lane1 : Usize),
      dequant.dequant_q8_0_row_loop0_loop0.body scale codes take out true none lane
          = ok (ControlFlow.cont (out1, true, none, lane1)) ∧
      q8_0LaneInv raw b out0 out1 take lane1 true none ∧ lane1.val = lane.val + 1 := by
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
  have hvval : v.val = q8_0Weight raw (32 * b + lane.val) := by
    rw [hvv, core.convert.num.FromI64I8.from_val_eq, hival, hsc,
      hcd lane.val hlane32]
    unfold q8_0Weight
    rw [div32_mul_add b lane.val hlane32, mod32_mul_add b lane.val hlane32]
  refine ⟨out2, lane2, ?_, ?_, hlane2⟩
  · rw [dequant.dequant_q8_0_row_loop0_loop0.body]
    rw [if_pos (by scalar_tac : lane < take)]
    simp only [if_true, hidx, lift, hmulv, hpush, haddl, bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    rw [hpushv, hlane2, dequantVals, List.map_append, ← dequantVals, hinv.2.2.2,
      List.range_succ, List.map_append]
    simp only [List.map_cons, List.map_nil, hvval, List.append_assoc]

/-- The lane loop: `loop.spec_decr_nat` with `q8_0LaneInv` and measure `take - lane`, on the
no-overflow path.

Complexity: `take` iterations, each O(1) (one array read, one checked multiply, one vector push);
`take` words are appended. -/
theorem q8_0Lane_loop_spec (raw : Slice U8) (b : Nat) (out0 : alloc.vec.Vec I64) (scale : I64)
    (codes : Array I8 32#usize) (take : Usize)
    (hsc : scale.val = q8_0ScaleRef raw b)
    (hcd : ∀ j, j < 32 → (codes.val.getD j 0#i8).val = q8_0CodeRef raw b j)
    (htake : take.val ≤ 32)
    (hmul : ∀ l, l < take.val → I64.min ≤ scale.val * (codes.val.getD l 0#i8).val ∧
      scale.val * (codes.val.getD l 0#i8).val ≤ I64.max)
    (hcap : out0.val.length + take.val ≤ Usize.max)
    (hinv0 : q8_0LaneInv raw b out0 out0 take 0#usize true none) :
    WP.spec (dequant.dequant_q8_0_row_loop0_loop0 out0 true none scale codes take 0#usize)
      (fun p => ∃ out : alloc.vec.Vec I64,
        p = (out, true, none) ∧ q8_0LaneInv raw b out0 out take take true none) := by
  unfold dequant.dequant_q8_0_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize =>
      take.val - s.2.2.2.val)
    (inv := fun s => q8_0LaneInv raw b out0 s.1 take s.2.2.2 s.2.1 s.2.2.1)
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
        q8_0Lane_body_step raw b out0 out scale codes take lane hsc hcd htake hlt
          (hmul lane.val hlt) hcap' ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hlane1] using hinv2, ?_⟩
      show take.val - lane1.val < take.val - lane.val
      omega
    · have hnot : ¬ (lane < take) := fun hc => hlt ((UScalar.lt_equiv lane take).1 hc)
      have hbody : dequant.dequant_q8_0_row_loop0_loop0.body scale codes take out true none lane
          = ok (ControlFlow.done (out, true, none)) := by
        rw [dequant.dequant_q8_0_row_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨out, rfl, ?_⟩
      have hlane_take : lane.val = take.val := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      rw [← hlane_take]
      exact hrec
  · exact hinv0

/-! ## Block-loop scaffolding

`q8_0BlockDomain` is the per-block success domain; `q8_0RowInv` accumulates the reference values
`min (32·block) n` after `block` completed blocks. -/

/-- The success domain of one Q8_0 block `b`: the block bytes are present, `d` is finite, and every
weight product `q8_0ScaleRef raw b · q8_0CodeRef raw b l` is inside `i64`. -/
def q8_0BlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  b.val * 34 + 34 ≤ raw.val.length ∧
  f16Exp (f16LeHalf raw (34 * b.val)) ≠ 31 ∧
  (∀ l, l < 32 → I64.min ≤ q8_0ScaleRef raw b.val * q8_0CodeRef raw b.val l ∧
    q8_0ScaleRef raw b.val * q8_0CodeRef raw b.val l ≤ I64.max)

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (32·block) n` raw weights. -/
def q8_0RowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ q8_0RowValues raw (Nat.min (32 * block.val) n.val)

/-- Appending the next `take` raw weights advances the row prefix by `take`. -/
lemma q8_0RowValues_append (raw : Slice U8) (m take : Nat) :
    q8_0RowValues raw m ++ (List.range take).map (fun j => q8_0Weight raw (m + j)) =
      q8_0RowValues raw (m + take) := by
  unfold q8_0RowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-- `block < (n + 31)/32` gives `32·block < n`: the block loop's divisibility bound. -/
lemma lt_mul_of_lt_div_ceil_32 {b n : Nat} (hb : b < (n + 31) / 32) : 32 * b < n := by
  by_contra h
  have hle : n ≤ 32 * b := not_lt.mp h
  have h1 : (n + 31) / 32 ≤ (32 * b + 31) / 32 := Nat.div_le_div_right (by omega)
  have h2 : (32 * b + 31) / 32 = b := by omega
  omega

/-- `32·b + min 32 (n - 32·b) = min (32·(b+1)) n` when `32·b ≤ n`. -/
lemma mul_block_add_min_32 (b n : Nat) (h : 32 * b ≤ n) :
    32 * b + Nat.min 32 (n - 32 * b) = Nat.min (32 * (b + 1)) n := by
  change 32 * b + min 32 (n - 32 * b) = min (32 * (b + 1)) n
  by_cases h1 : n - 32 * b ≤ 32
  · rw [Nat.min_eq_right h1, Nat.min_eq_right (by omega)]
    omega
  · have h1' : 32 < n - 32 * b := not_le.mp h1
    rw [Nat.min_eq_left (le_of_lt h1'), Nat.min_eq_left (by omega)]
    omega

/-- `n ≤ 32 · ceil(n/32)`. -/
lemma le_mul_div_ceil_32 (n : Nat) : n ≤ 32 * ((n + 31) / 32) := by omega

/-- The extracted `div_ceil n Q8_0_QK` succeeds and its value is `(n + 31)/32`. -/
theorem dequant_div_ceil_32_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.Q8_0_QK = ok blocks ∧
      blocks.val = (n.val + 31) / 32 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.Q8_0_QK) (by simp only [quant.Q8_0_QK]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.Q8_0_QK) (by simp only [quant.Q8_0_QK]; decide))
  have hqv' : q.val = n.val / 32 := by rw [hqv]; simp [quant.Q8_0_QK]
  have hrv' : r.val = n.val % 32 := by rw [hrv]; simp [quant.Q8_0_QK]
  have hq' : UScalar.div n quant.Q8_0_QK = ok q := hq
  have hr' : UScalar.rem n quant.Q8_0_QK = ok r := hr
  unfold core.num.Usize.div_ceil
  rw [hq']
  simp only [bind_tc_ok]
  rw [hr']
  simp only [bind_tc_ok]
  by_cases hr0 : r.val = 0
  · refine ⟨q, ?_, ?_⟩
    · rw [if_pos hr0]
    · rw [hqv']
      have hr0' : n.val % 32 = 0 := by rw [← hrv']; exact hr0
      omega
  · obtain ⟨q1, hq1, hq1v⟩ := WP.spec_imp_exists
      (UScalar.add_spec (x := q) (y := (UScalar.one : Usize)) (by
        rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]
      exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 32 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-- One block-loop body step on the success domain: decode block `block`, run the lane loop,
append `take = min 32 (n - 32·block)` values and advance. -/
theorem q8_0Block_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 31) / 32)
    (hdom : q8_0BlockDomain raw block)
    (hcap : out.val.length + (n.val - 32 * block.val) ≤ Usize.max)
    (hinv : q8_0RowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.dequant_q8_0_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      q8_0RowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨hlen, hfin, hprod⟩ := hdom
  obtain ⟨d, codes, hq8, hdv, hcodes⟩ := q8_0_block_ok raw block hlen hfin
  have hltmul : 32 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_32 (by rw [← hblocks]; exact hblock)
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.Q8_0_QK) (by simp only [quant.Q8_0_QK]; scalar_tac))
  have hivv' : i.val = 32 * block.val := by
    rw [hivv]; simp only [quant.Q8_0_QK]; scalar_tac
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 32 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.Q8_0_QK i1
  have htakev' : take.val = Nat.min 32 (n.val - 32 * block.val) := by
    rw [htakev, hi1v']; simp [quant.Q8_0_QK]
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 32 := by rw [htakev']; exact min_le_left _ _
  have htake_X : take.val ≤ n.val - 32 * block.val := by rw [htakev']; exact min_le_right _ _
  have hcap' : out.val.length + take.val ≤ Usize.max := by omega
  have hmul : ∀ l, l < take.val → I64.min ≤ d.val * (codes.val.getD l 0#i8).val ∧
      d.val * (codes.val.getD l 0#i8).val ≤ I64.max := by
    intro l hl
    have hl32 : l < 32 := by omega
    have h := hprod l hl32
    rw [← hdv, ← hcodes l hl32] at h
    exact h
  have hinv0 : q8_0LaneInv raw block.val out out take 0#usize true none := by
    refine ⟨rfl, rfl, by simp, ?_⟩
    simp
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q8_0Lane_loop_spec raw block.val out d codes take hdv hcodes htake_le hmul hcap' hinv0)
  obtain ⟨vals1, hp1, hlane⟩ := hp
  have hloop' : dequant.dequant_q8_0_row_loop0_loop0 out true none d codes take 0#usize
      = ok (vals1, true, none) := by rw [hloop, hp1]
  have hmin : Nat.min (32 * block.val) n.val = 32 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 32 * block.val + take.val = Nat.min (32 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min_32 block.val n.val (le_of_lt hltmul)
  have hinv_new : dequantVals vals1.val =
      dequantVals out0.val ++ q8_0RowValues raw (Nat.min (32 * block1.val) n.val) := by
    rw [hblock1v', hlane.2.2.2, hinv.2.2.2, hmin, List.append_assoc, q8_0RowValues_append, hsum]
  refine ⟨vals1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.dequant_q8_0_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq8]
  simp only [bind_tc_ok]
  change (do
    let i ← block * quant.Q8_0_QK
    let i1 ← n - i
    let take ← core.cmp.Ord.min.trait_default core.cmp.OrdUsize quant.Q8_0_QK i1
    let (out1, ok2, err1) ←
      dequant.dequant_q8_0_row_loop0_loop0 out true none d codes take 0#usize
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, ok2, err1)))
    = ok (ControlFlow.cont (vals1, block1, true, none))
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  rw [hloop']
  simp only [bind_tc_ok]
  rw [hblock1a]
  simp only [bind_tc_ok]
  rfl

/-- The block loop: `loop.spec_decr_nat` with `q8_0RowInv` and measure `blocks - block`, on the
success domain `q8_0BlockDomain` for every block below `blocks`.

Complexity: `blocks` iterations, each a 34-byte scale read, a 32-code read and at most 32 pushes;
the measure is `blocks - block`. -/
theorem q8_0Row_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 31) / 32)
    (hdom : ∀ b : Usize, b.val < blocks.val → q8_0BlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.dequant_q8_0_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ q8_0RowValues raw n.val) := by
  unfold dequant.dequant_q8_0_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => q8_0RowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
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
        simpa [dequantVals, q8_0RowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 32 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        q8_0Block_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.dequant_q8_0_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.dequant_q8_0_row_loop0.body, if_neg hnot]
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
    simp [q8_0RowValues]

/-- **`Q8_0` row law.** On the block success domain, the extracted `dequant_q8_0_row` succeeds with
`Ok ()` and appends exactly the first `n` raw weights `q8_0RowValues raw n` to the output. -/
theorem dequant_q8_0_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 31) / 32 → q8_0BlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q8_0_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ q8_0RowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_32_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q8_0Row_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.dequant_q8_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## `dequant_row_fixed` at `GGML_TYPE_Q8_0` -/

/-- `dequant_row_fixed` at `ggml_type = 8` (`GGML_TYPE_Q8_0`) is `dequant_q8_0_row`. -/
theorem dequant_row_fixed_q8_0 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 8#u32 raw n out = dequant.dequant_q8_0_row raw n out := rfl

/-! ## One-block joint domain witness

A concrete 34-byte Q8_0 block: `d = 1.0` (`0x3C00` little-endian), code `0` equal to `1` and the
other thirty-one codes `0`. The weight product is `2^32` or `0`, inside `i64`. -/

/-- `f16LeHalf` at `0` of a list beginning `a :: b :: _`. -/
lemma q8_0_f16LeHalf_zero_cons (raw : Slice U8) (a b : U8) (l : List U8)
    (h : raw.val = a :: b :: l) :
    f16LeHalf raw 0 = core.num.U16.from_le_bytes (Array.make 2#usize [a, b]) := by
  unfold f16LeHalf
  congr 1
  apply Aeneas.Std.Array.ext
  simp [h]

/-- The half-word `[0x00, 0x3C]` is the finite binary16 pattern `0x3C00 = 15360` (`1.0`). -/
lemma q8_0_u16_le_0_60 :
    (core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8])).val = 15360 := by decide

/-- The 34-byte witness block: `d = 1.0`, code `0` equal to `1`, the rest `0`. -/
def q8_0WitBytes : List U8 := 0#u8 :: 60#u8 :: (1#u8 :: List.replicate 31 0#u8)

theorem q8_0WitBytes_length : q8_0WitBytes.length = 34 := by
  simp [q8_0WitBytes]

def q8_0WitRaw : Slice U8 := Slice.from q8_0WitBytes (by rw [q8_0WitBytes_length]; scalar_tac)

theorem q8_0WitRaw_val : q8_0WitRaw.val = q8_0WitBytes := Slice.from_val _ _

/-- The witness scale is `0x3C00 = 15360`. -/
lemma q8_0Wit_f16LeHalf_0 : (f16LeHalf q8_0WitRaw 0).val = 15360 := by
  have heq : f16LeHalf q8_0WitRaw 0 =
      core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) := by
    apply q8_0_f16LeHalf_zero_cons
    rw [q8_0WitRaw_val]
    rfl
  rw [heq, q8_0_u16_le_0_60]

/-- `d = 1.0 = 2^32` in Q32.32 at block `0`. -/
lemma q8_0Wit_scale : q8_0ScaleRef q8_0WitRaw 0 = 2 ^ 32 := by
  have hv := q8_0Wit_f16LeHalf_0
  unfold q8_0ScaleRef
  rw [show 34 * 0 = 0 by ring]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

/-- `getD` at offset `k + 3` of a list beginning three elements is the tail's `getD`. -/
lemma getD_cons3 (a b c : U8) (L : List U8) (k : Nat) :
    (a :: b :: c :: L).getD (k + 3) 0#u8 = L.getD k 0#u8 := by
  rw [show k + 3 = (k + 2) + 1 by omega, List.getD_cons_succ]
  rw [show k + 2 = (k + 1) + 1 by omega, List.getD_cons_succ]
  rw [show k + 1 = k + 1 by rfl, List.getD_cons_succ]

/-- `getD` of a replicate of `0` is `0`. -/
lemma getD_replicate_zero (n k : Nat) : (List.replicate n 0#u8).getD k 0#u8 = 0#u8 := by
  rw [List.getD_eq_getElem?_getD]
  by_cases h : k < n
  · rw [List.getElem?_replicate, if_pos h]
    rfl
  · rw [List.getElem?_replicate, if_neg h]
    rfl

/-- The first code of the witness block is `1`. -/
lemma q8_0Wit_code_zero : q8_0CodeRef q8_0WitRaw 0 0 = 1 := by
  unfold q8_0CodeRef q8_0CodeAt
  rw [show 34 * 0 + 2 + 0 = 2 by ring, q8_0WitRaw_val]
  decide

/-- Every code of the witness block is `1` at lane `0` and `0` otherwise. -/
lemma q8_0Wit_code (l : Nat) (hl : l < 32) :
    q8_0CodeRef q8_0WitRaw 0 l = if l = 0 then 1 else 0 := by
  unfold q8_0CodeRef q8_0CodeAt
  rw [show 34 * 0 + 2 + l = 2 + l by ring, q8_0WitRaw_val]
  by_cases hl0 : l = 0
  · subst hl0
    decide
  · obtain ⟨k, rfl⟩ : ∃ k, l = k + 1 := ⟨l - 1, by omega⟩
    rw [show 2 + (k + 1) = k + 3 by omega,
      show q8_0WitBytes = 0#u8 :: 60#u8 :: 1#u8 :: List.replicate 31 0#u8 from rfl,
      getD_cons3, getD_replicate_zero]
    simp only [Nat.succ_ne_zero, if_false]
    decide

/-- **Joint non-vacuity of `q8_0BlockDomain` at block `0`.** -/
theorem q8_0BlockDomain_witness : q8_0BlockDomain q8_0WitRaw 0#usize := by
  have hb : (0#usize).val = 0 := by decide
  refine ⟨?_, ?_, ?_⟩
  · simp [hb, q8_0WitRaw_val, q8_0WitBytes_length]
  · have hv := q8_0Wit_f16LeHalf_0
    unfold f16Exp
    rw [hb]
    rw [show 34 * 0 = 0 by ring, hv]
    norm_num
  · intro l hl
    rw [hb, q8_0Wit_scale, q8_0Wit_code l hl]
    split <;> norm_num [I64.min_eq, I64.max_eq]

/-- The head of the `32`-element `q8_0RowValues` list. -/
lemma q8_0RowValues_head (raw : Slice U8) :
    (q8_0RowValues raw 32).getD 0 0 = q8_0Weight raw 0 := by
  unfold q8_0RowValues
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 32)[0]? = some 0 := by decide
  rw [h]
  rfl

/-- The first weight of the witness block is `d·q = 2^32`. -/
theorem q8_0Wit_weight_zero : q8_0Weight q8_0WitRaw 0 = 2 ^ 32 := by
  unfold q8_0Weight
  rw [show (0 / 32 : Nat) = 0 by norm_num, show (0 % 32 : Nat) = 0 by norm_num,
    q8_0Wit_scale, q8_0Wit_code_zero]
  norm_num

/-- The empty vector's payload is `[]`. -/
lemma q8_0_vec_new_val_i64 : (alloc.vec.Vec.new I64).val = ([] : List I64) := by
  simp [alloc.vec.Vec.new, alloc.vec.Vec.from_val]

/-- `getD` of a `Vec` value is the `getD` of the underlying `I64` list. -/
lemma q8_0_dequantVals_getD (l : List I64) (i : Nat) :
    (dequantVals l).getD i 0 = (l.getD i 0#i64).val := by
  unfold dequantVals
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases h : l[i]? with
  | none => simp
  | some v => simp

/-- **`Q8_0` row witness at `n = 32`.** A single concrete block: the extracted row succeeds on the
empty vector, appends exactly the `q8_0RowValues`, and element `0` is `2^32`. -/
theorem q8_0Witness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q8_0_row q8_0WitRaw 32#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = q8_0RowValues q8_0WitRaw 32 ∧
      (out1.val.getD 0 0#i64).val = 2 ^ 32 := by
  have hdom : ∀ b : Usize, b.val < ((32#usize).val + 31) / 32 → q8_0BlockDomain q8_0WitRaw b := by
    intro b hb
    have hb1 : b.val < 1 := by
      have hR : (((32#usize).val + 31) / 32) = 1 := by decide
      rwa [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact q8_0BlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_q8_0_row_spec q8_0WitRaw 32#usize (alloc.vec.Vec.new I64) hdom (by
      rw [q8_0_vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = q8_0RowValues q8_0WitRaw 32 := by
    rw [q8_0_vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  · have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
    rw [q8_0_dequantVals_getD] at h0
    rw [h0, q8_0RowValues_head]
    exact q8_0Wit_weight_zero

end

end apeiron_word
