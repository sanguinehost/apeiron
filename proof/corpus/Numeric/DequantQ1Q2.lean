/-
Row laws for the extracted `Q1_0`, `Q2_0` and `PQ2_0` dequantizers.

The reference definitions, the `quant::q1_q2_block`/`quant::pq2_0_block` readers and the `Q2_0` row
law are in `Numeric.DequantQ1Q2Block`. This module holds the `PQ2_0` and `Q1_0` lane/block loop
scaffolding and their row laws (`dequant_pq2_0_row_spec`, `dequant_q1_0_row_spec`), the
`dequant_row_fixed` specializations, and the one-block joint non-vacuity witnesses for all three
formats.

The Rust bodies are `dequant::lowbit::dequant_q1_0_row`, `dequant::lowbit::dequant_q2_0_row` and
`dequant::lowbit::dequant_pq2_0_row` in `src/engine/dequant_lowbit.rs`; the decode is exact integer
arithmetic over Q32.32, with the block `d` a finite `f16` decoded by `f16_to_fixed_exact` and every
weight an integer multiple of `d`. `Q1_0` (`ggml_type` 41, `Q1_0_QK` 128, 18-byte block) weight `i`
is `bit(qs[(i%128)/8], (i%128)%8) ? d : -d`; `PQ2_0` (`ggml_type` 142, `PQ2_0_QK` 128, 34-byte
block) uses `Q2_0` codes over 128 weights.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.DequantQ1Q2Block

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## `dequant_pq2_0_row` -/

/-- The exact weight `l` of PQ2_0 block `b` (`= pq2_0Weight raw (128·b + l)`). -/
def pq2_0LaneVal (raw : Slice U8) (b l : Nat) : ℤ :=
  ((pq2_0Code raw b l : ℤ) - 1) * pq2_0ScaleRef raw b

/-- `pq2_0Weight` at lane `l` of block `b`. -/
lemma pq2_0Weight_lane (raw : Slice U8) (b l : Nat) (hl : l < 128) :
    pq2_0Weight raw (128 * b + l) = pq2_0LaneVal raw b l := by
  unfold pq2_0Weight pq2_0LaneVal
  rw [show (128 * b + l) / 128 = b by omega, show (128 * b + l) % 128 = l by omega]

/-- Appending the next `take` raw weights advances the PQ2_0 row prefix by `take`. -/
lemma pq2_0RowValues_append (raw : Slice U8) (m take : Nat) :
    pq2_0RowValues raw m ++ (List.range take).map (fun j => pq2_0Weight raw (m + j)) =
      pq2_0RowValues raw (m + take) := by
  unfold pq2_0RowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-- Lane-loop invariant: the first `lane` values of block `b` are appended, success flag set. -/
def pq2_0LaneInv (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64) (take lane : Usize)
    (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ lane.val ≤ take.val ∧
  dequantVals out.val = dequantVals out0.val ++
    (List.range lane.val).map (fun l => pq2_0LaneVal raw b l)

/-- One PQ2_0 lane-loop body step on the success path: decode code `lane`, multiply by the scale,
push, advance. -/
theorem pq2_0Lane_body_step (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64)
    (scale : I64) (qs : Array U8 32#usize) (take lane : Usize)
    (hsc : scale.val = pq2_0ScaleRef raw b)
    (hcd : ∀ j, j < 32 → (qs.val.getD j 0#u8).val = pq2_0CodeByte raw b j)
    (htake : take.val ≤ 128) (hlane : lane.val < take.val)
    (hmul : I64.min ≤ scale.val * ((pq2_0Code raw b lane.val : ℤ) - 1) ∧
      scale.val * ((pq2_0Code raw b lane.val : ℤ) - 1) ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : pq2_0LaneInv raw b out0 out take lane true none) :
    ∃ (out2 : alloc.vec.Vec I64) (lane2 : Usize),
      dequant.lowbit.dequant_pq2_0_row_loop0_loop0.body scale qs take out true none lane
          = ok (ControlFlow.cont (out2, true, none, lane2)) ∧
      pq2_0LaneInv raw b out0 out2 take lane2 true none ∧ lane2.val = lane.val + 1 := by
  have hlane64 : lane.val < 128 := by omega
  obtain ⟨i, hdiv, hiv⟩ := UScalar.div_spec lane (y := 4#usize) (by decide)
  have hi : i.val = lane.val / 4 := by simpa using hiv
  have hi16 : i.val < 32 := by rw [hi]; omega
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
  have hi1val : i1.val = pq2_0CodeByte raw b (lane.val / 4) := by
    have h1 : i1.val = (qs.val.getD i.val 0#u8).val := by
      rw [hi1v]
      exact congrArg (fun x : U8 => x.val)
        (getD_of_lt (l := qs.val) (by rw [Array.property qs]; exact hi16)).symm
    rw [h1, hi, hcd (lane.val / 4) (by omega)]
  have hcodeval : (i4 &&& 3#u8).val = pq2_0Code raw b lane.val := by
    rw [UScalar.val_and]
    have h3 : (3#u8).val = 3 := rfl
    rw [h3, hi4v, hi3, hi1val]
    rfl
  have hi5val : i5.val = (pq2_0Code raw b lane.val : ℤ) := by
    dsimp only [i5]
    rw [low_hcast_u8_I64_val, hcodeval]
  have hcode_lt : pq2_0Code raw b lane.val < 4 := by
    unfold pq2_0Code
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
  have hi6val : i6.val = (pq2_0Code raw b lane.val : ℤ) - 1 := by
    have h1 := hi6v
    rw [hi5val, hone] at h1
    exact h1
  obtain ⟨v, hmulv, hvval⟩ := q6k_mul_ok scale i6
    (by rw [hi6val]; exact hmul.1) (by rw [hi6val]; exact hmul.2)
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out v hcap)
  obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
  have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
  have hvval' : v.val = pq2_0LaneVal raw b lane.val := by
    rw [hvval, hsc, hi6val]
    unfold pq2_0LaneVal
    ring
  refine ⟨out2, lane2, ?_, ?_, hlane2⟩
  · rw [dequant.lowbit.dequant_pq2_0_row_loop0_loop0.body]
    rw [if_pos (by scalar_tac : lane < take)]
    rw [if_pos (by simp : (true : Bool) = true)]
    simp only [hdiv, hidx, hrem, hmul2, hshr, hcast, hsub, hmulv, hpush, haddl, lift, bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    rw [hpushv, hlane2, dequantVals, List.map_append, ← dequantVals, hinv.2.2.2,
      List.range_succ, List.map_append]
    simp only [List.map_cons, List.map_nil, hvval', List.append_assoc]

/-- The PQ2_0 lane loop: `loop.spec_decr_nat` with `pq2_0LaneInv` and measure `take - lane`.

Complexity: `take` iterations, each O(1) (one byte read, five bit operations, one checked multiply,
one push); `take` words appended. -/
theorem pq2_0Lane_loop_spec (raw : Slice U8) (b : Nat) (out0 : alloc.vec.Vec I64) (scale : I64)
    (qs : Array U8 32#usize) (take : Usize)
    (hsc : scale.val = pq2_0ScaleRef raw b)
    (hcd : ∀ j, j < 32 → (qs.val.getD j 0#u8).val = pq2_0CodeByte raw b j)
    (htake : take.val ≤ 128)
    (hmul : ∀ l, l < take.val → I64.min ≤ scale.val * ((pq2_0Code raw b l : ℤ) - 1) ∧
      scale.val * ((pq2_0Code raw b l : ℤ) - 1) ≤ I64.max)
    (hcap : out0.val.length + take.val ≤ Usize.max)
    (hinv0 : pq2_0LaneInv raw b out0 out0 take 0#usize true none) :
    WP.spec (dequant.lowbit.dequant_pq2_0_row_loop0_loop0 out0 true none scale qs take 0#usize)
      (fun p => ∃ out : alloc.vec.Vec I64,
        p = (out, true, none) ∧ pq2_0LaneInv raw b out0 out take take true none) := by
  unfold dequant.lowbit.dequant_pq2_0_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize =>
      take.val - s.2.2.2.val)
    (inv := fun s => pq2_0LaneInv raw b out0 s.1 take s.2.2.2 s.2.1 s.2.2.1)
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
        pq2_0Lane_body_step raw b out0 out scale qs take lane hsc hcd htake hlt
          (hmul lane.val hlt) hcap' ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hlane1] using hinv2, ?_⟩
      show take.val - lane1.val < take.val - lane.val
      omega
    · have hnot : ¬ (lane < take) := fun hc => hlt ((UScalar.lt_equiv lane take).1 hc)
      have hbody : dequant.lowbit.dequant_pq2_0_row_loop0_loop0.body scale qs take out true none lane
          = ok (ControlFlow.done (out, true, none)) := by
        rw [dequant.lowbit.dequant_pq2_0_row_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨out, rfl, ?_⟩
      have hlane_take : lane.val = take.val := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      rw [← hlane_take]
      exact hrec
  · exact hinv0

/-- The success domain of one PQ2_0 block `b`: the block bytes are present, `d` is finite, and every
weight product is inside `i64`. -/
def pq2_0BlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  b.val * 34 + 34 ≤ raw.val.length ∧
  f16Exp (f16LeHalf raw (34 * b.val)) ≠ 31 ∧
  (∀ l, l < 128 → I64.min ≤ pq2_0ScaleRef raw b.val * ((pq2_0Code raw b.val l : ℤ) - 1) ∧
    pq2_0ScaleRef raw b.val * ((pq2_0Code raw b.val l : ℤ) - 1) ≤ I64.max)

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (128·block) n` raw weights. -/
def pq2_0RowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ pq2_0RowValues raw (Nat.min (128 * block.val) n.val)

/-- One PQ2_0 block-loop body step: decode block `block`, run the lane loop, append
`take = min 128 (n - 128·block)` values and advance. -/
theorem pq2_0Block_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 127) / 128)
    (hdom : pq2_0BlockDomain raw block)
    (hcap : out.val.length + (n.val - 128 * block.val) ≤ Usize.max)
    (hinv : pq2_0RowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.lowbit.dequant_pq2_0_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      pq2_0RowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨hlen, hfin, hprod⟩ := hdom
  obtain ⟨d, qs, hq, hdv, hcodes⟩ := pq2_0_block_ok raw block hlen hfin
  have hltmul : 128 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_128 (by rw [← hblocks]; exact hblock)
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.PQ2_0_QK) (by rw [quant_PQ2_0_QK_eq]; scalar_tac))
  have hivv' : i.val = 128 * block.val := by rw [hivv, quant_PQ2_0_QK_eq]; simp; omega
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 128 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.PQ2_0_QK i1
  have htakev' : take.val = Nat.min 128 (n.val - 128 * block.val) := by
    rw [htakev, hi1v', quant_PQ2_0_QK_eq]; simp
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 128 := by rw [htakev']; exact min_le_left _ _
  have hcap' : out.val.length + take.val ≤ Usize.max := by
    have htake_X : take.val ≤ n.val - 128 * block.val := by rw [htakev']; exact min_le_right _ _
    omega
  have hdv2 : d.val = pq2_0ScaleRef raw block.val := hdv
  have hmul : ∀ l, l < take.val → I64.min ≤ d.val * ((pq2_0Code raw block.val l : ℤ) - 1) ∧
      d.val * ((pq2_0Code raw block.val l : ℤ) - 1) ≤ I64.max := by
    intro l hl
    have hl64 : l < 128 := by omega
    have h := hprod l hl64
    rw [← hdv2] at h
    exact h
  have hinv0 : pq2_0LaneInv raw block.val out out take 0#usize true none := by
    refine ⟨rfl, rfl, by simp, ?_⟩
    simp
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (pq2_0Lane_loop_spec raw block.val out d qs take hdv2 hcodes htake_le hmul hcap' hinv0)
  obtain ⟨vals1, hp1, hlane⟩ := hp
  have hloop' : dequant.lowbit.dequant_pq2_0_row_loop0_loop0 out true none d qs take 0#usize
      = ok (vals1, true, none) := by rw [hloop, hp1]
  have hmin : Nat.min (128 * block.val) n.val = 128 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 128 * block.val + take.val = Nat.min (128 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min_128 block.val n.val (le_of_lt hltmul)
  have hmap : (List.range take.val).map (fun l => pq2_0LaneVal raw block.val l)
      = (List.range take.val).map (fun j => pq2_0Weight raw (128 * block.val + j)) := by
    apply List.map_congr_left
    intro l hl
    exact (pq2_0Weight_lane raw block.val l (by have := List.mem_range.mp hl; omega)).symm
  have hinv_new : dequantVals vals1.val =
      dequantVals out0.val ++ pq2_0RowValues raw (Nat.min (128 * block1.val) n.val) := by
    rw [hblock1v', hlane.2.2.2, hinv.2.2.2, hmin, List.append_assoc, hmap,
      pq2_0RowValues_append, hsum]
  refine ⟨vals1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.lowbit.dequant_pq2_0_row_loop0.body]
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
      dequant.lowbit.dequant_pq2_0_row_loop0_loop0 out true none d qs take 0#usize
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, ok2, err1)))
    = ok (ControlFlow.cont (vals1, block1, true, none))
  rw [hloop']
  simp only [bind_tc_ok]
  rw [hblock1a]
  simp only [bind_tc_ok]
  rfl

/-- The PQ2_0 block loop: `loop.spec_decr_nat` with `pq2_0RowInv` and measure `blocks - block`, on the
success domain for every block below `blocks`.

Complexity: `blocks` iterations, each an 34-byte block read and at most 128 pushes; the measure is
`blocks - block`. -/
theorem pq2_0Row_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 127) / 128)
    (hdom : ∀ b : Usize, b.val < blocks.val → pq2_0BlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.lowbit.dequant_pq2_0_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ pq2_0RowValues raw n.val) := by
  unfold dequant.lowbit.dequant_pq2_0_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => pq2_0RowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, block, ok1, err⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : block.val < blocks.val
    · have hltmul : 128 * block.val < n.val :=
        lt_mul_of_lt_div_ceil_128 (by rw [← hblocks]; exact hlt)
      have hmin : Nat.min (128 * block.val) n.val = 128 * block.val :=
        Nat.min_eq_left (le_of_lt hltmul)
      have hlen_out : out.val.length = out0.val.length + 128 * block.val := by
        have h := congrArg List.length hrec
        rw [hmin] at h
        simpa [dequantVals, pq2_0RowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 128 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        pq2_0Block_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.lowbit.dequant_pq2_0_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.lowbit.dequant_pq2_0_row_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨trivial, ?_⟩
      have hblock_eq : block.val = blocks.val := by omega
      rw [hrec, hblock_eq, hblocks]
      have hminend : Nat.min (128 * ((n.val + 127) / 128)) n.val = n.val :=
        Nat.min_eq_right (le_mul_div_ceil_128 n.val)
      rw [hminend]
  · refine ⟨rfl, rfl, by simp, ?_⟩
    have h0 : Nat.min (128 * (0#usize).val) n.val = 0 := by simp
    rw [h0]
    simp [pq2_0RowValues]

/-- **`PQ2_0` row law.** On the block success domain, the extracted `dequant_pq2_0_row` succeeds with
`Ok ()` and appends exactly the first `n` raw weights `pq2_0RowValues raw n`. -/
theorem dequant_pq2_0_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 127) / 128 → pq2_0BlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_pq2_0_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ pq2_0RowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_128pq_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (pq2_0Row_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.lowbit.dequant_pq2_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- `dequant_row_fixed` at `ggml_type = 42` (`GGML_TYPE_PQ2_0`) is `dequant_pq2_0_row`. -/
theorem dequant_row_fixed_pq2_0 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 142#u32 raw n out = dequant.lowbit.dequant_pq2_0_row raw n out := rfl

/-! ## `dequant_q1_0_row` -/

/-- The exact weight `l` of Q1_0 block `b` (`= q1_0Weight raw (128·b + l)`). -/
def q1_0LaneVal (raw : Slice U8) (b l : Nat) : ℤ :=
  if q1_0Bit raw b l = 1 then q1_0ScaleRef raw b else -q1_0ScaleRef raw b

/-- `q1_0Weight` at lane `l` of block `b`. -/
lemma q1_0Weight_lane (raw : Slice U8) (b l : Nat) (hl : l < 128) :
    q1_0Weight raw (128 * b + l) = q1_0LaneVal raw b l := by
  unfold q1_0Weight q1_0LaneVal q1_0BitRef
  rw [show (128 * b + l) / 128 = b by omega, show (128 * b + l) % 128 = l by omega]

/-- Appending the next `take` raw weights advances the Q1_0 row prefix by `take`. -/
lemma q1_0RowValues_append (raw : Slice U8) (m take : Nat) :
    q1_0RowValues raw m ++ (List.range take).map (fun j => q1_0Weight raw (m + j)) =
      q1_0RowValues raw (m + take) := by
  unfold q1_0RowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-- Lane-loop invariant: the first `lane` values of block `b` are appended, success flag set. -/
def q1_0LaneInv (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64) (take lane : Usize)
    (ok1 : Bool) : Prop :=
  ok1 = true ∧ lane.val ≤ take.val ∧
  dequantVals out.val = dequantVals out0.val ++
    (List.range lane.val).map (fun l => q1_0LaneVal raw b l)

/-- One Q1_0 lane-loop body step on the success path: read sign bit `lane`, push `scale` or `neg`,
advance. -/
theorem q1_0Lane_body_step (raw : Slice U8) (b : Nat) (out0 out : alloc.vec.Vec I64)
    (scale : I64) (qs : Array U8 16#usize) (neg : I64) (take lane : Usize)
    (hsc : scale.val = q1_0ScaleRef raw b)
    (hng : neg.val = -q1_0ScaleRef raw b)
    (hcd : ∀ j, j < 16 → (qs.val.getD j 0#u8).val = q1_0CodeByte raw b j)
    (htake : take.val ≤ 128) (hlane : lane.val < take.val)
    (hcap : out.val.length < Usize.max)
    (hinv : q1_0LaneInv raw b out0 out take lane true) :
    ∃ (out2 : alloc.vec.Vec I64) (lane2 : Usize),
      dequant.lowbit.dequant_q1_0_row_loop0_loop0.body scale qs neg take out true lane
          = ok (ControlFlow.cont (out2, true, lane2)) ∧
      q1_0LaneInv raw b out0 out2 take lane2 true ∧ lane2.val = lane.val + 1 := by
  have hlane128 : lane.val < 128 := by omega
  obtain ⟨i, hdiv, hiv⟩ := UScalar.div_spec lane (y := 8#usize) (by decide)
  have hi : i.val = lane.val / 8 := by simpa using hiv
  have hi16 : i.val < 16 := by rw [hi]; omega
  obtain ⟨i1, hidx, hi1v⟩ := WP.spec_imp_exists
    (Array.index_usize_spec qs i (by show i.val < qs.val.length; rw [Array.property qs]; exact hi16))
  obtain ⟨i2, hrem, hi2v⟩ := WP.spec_imp_exists
    (UScalar.rem_spec lane (y := 8#usize) (by decide))
  have hi2 : i2.val = lane.val % 8 := by simpa using hi2v
  obtain ⟨i3, hshr, hi3v, -⟩ := WP.spec_imp_exists
    (UScalar.ShiftRight_spec (x := i1) (y := i2)
      (by rw [hi2]; simp only [UScalarTy.U8_numBits_eq]; omega))
  have hi1val : i1.val = q1_0CodeByte raw b (lane.val / 8) := by
    have h1 : i1.val = (qs.val.getD i.val 0#u8).val := by
      rw [hi1v]
      exact congrArg (fun x : U8 => x.val)
        (getD_of_lt (l := qs.val) (by rw [Array.property qs]; exact hi16)).symm
    rw [h1, hi, hcd (lane.val / 8) (by omega)]
  have hbitval : (i3 &&& 1#u8).val = q1_0Bit raw b lane.val := by
    rw [UScalar.val_and]
    have h1 : (1#u8).val = 1 := rfl
    rw [h1, hi3v, hi2, hi1val]
    unfold q1_0Bit
    rfl
  obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
  have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
  by_cases hb : (i3 &&& 1#u8) = 1#u8
  · have hbitv : q1_0Bit raw b lane.val = 1 := by rw [← hbitval, hb]; decide
    obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out scale hcap)
    have hv : scale.val = q1_0LaneVal raw b lane.val := by
      rw [hsc]; unfold q1_0LaneVal; rw [if_pos hbitv]
    refine ⟨out2, lane2, ?_, ?_, hlane2⟩
    · rw [dequant.lowbit.dequant_q1_0_row_loop0_loop0.body]
      rw [if_pos (by scalar_tac : lane < take)]
      simp only [if_true]
      simp only [hdiv, hidx, hrem, hshr, lift, if_pos hb, hpush, haddl, bind_tc_ok]
      try rfl
    · refine ⟨rfl, by omega, ?_⟩
      rw [hpushv, hlane2, dequantVals, List.map_append, ← dequantVals, hinv.2.2,
        List.range_succ, List.map_append]
      simp only [List.map_cons, List.map_nil, hv, List.append_assoc]
  · have hbitv : q1_0Bit raw b lane.val ≠ 1 := by
      intro h
      apply hb
      apply UScalar.eq_of_val_eq
      rw [hbitval, h]
      decide
    obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out neg hcap)
    have hv : neg.val = q1_0LaneVal raw b lane.val := by
      rw [hng]; unfold q1_0LaneVal; rw [if_neg hbitv]
    refine ⟨out2, lane2, ?_, ?_, hlane2⟩
    · rw [dequant.lowbit.dequant_q1_0_row_loop0_loop0.body]
      rw [if_pos (by scalar_tac : lane < take)]
      simp only [if_true]
      simp only [hdiv, hidx, hrem, hshr, lift, if_neg hb, hpush, haddl, bind_tc_ok]
      try rfl
    · refine ⟨rfl, by omega, ?_⟩
      rw [hpushv, hlane2, dequantVals, List.map_append, ← dequantVals, hinv.2.2,
        List.range_succ, List.map_append]
      simp only [List.map_cons, List.map_nil, hv, List.append_assoc]

/-- The Q1_0 lane loop: `loop.spec_decr_nat` with `q1_0LaneInv` and measure `take - lane`.

Complexity: `take` iterations, each O(1) (one byte read, three bit operations, one push); `take`
words appended. -/
theorem q1_0Lane_loop_spec (raw : Slice U8) (b : Nat) (out0 : alloc.vec.Vec I64) (scale : I64)
    (qs : Array U8 16#usize) (neg : I64) (take : Usize)
    (hsc : scale.val = q1_0ScaleRef raw b)
    (hng : neg.val = -q1_0ScaleRef raw b)
    (hcd : ∀ j, j < 16 → (qs.val.getD j 0#u8).val = q1_0CodeByte raw b j)
    (htake : take.val ≤ 128)
    (hcap : out0.val.length + take.val ≤ Usize.max)
    (hinv0 : q1_0LaneInv raw b out0 out0 take 0#usize true) :
    WP.spec (dequant.lowbit.dequant_q1_0_row_loop0_loop0 out0 true scale qs neg take 0#usize)
      (fun p => ∃ out : alloc.vec.Vec I64,
        p = (out, true) ∧ q1_0LaneInv raw b out0 out take take true) := by
  unfold dequant.lowbit.dequant_q1_0_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Usize => take.val - s.2.2.val)
    (inv := fun s => q1_0LaneInv raw b out0 s.1 take s.2.2 s.2.1)
  · rintro ⟨out, ok1, lane⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hok, hle, hrec⟩ := hinv'
    rw [hok]
    by_cases hlt : lane.val < take.val
    · have hcap' : out.val.length < Usize.max := by
        have hlen : out.val.length = out0.val.length + lane.val := by
          have h := congrArg List.length hrec
          simp [dequantVals, List.length_map, List.length_range] at h
          omega
        omega
      obtain ⟨out1, lane1, hbody, hinv2, hlane1⟩ :=
        q1_0Lane_body_step raw b out0 out scale qs neg take lane hsc hng hcd htake hlt hcap'
          ⟨rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hlane1] using hinv2, ?_⟩
      show take.val - lane1.val < take.val - lane.val
      omega
    · have hnot : ¬ (lane < take) := fun hc => hlt ((UScalar.lt_equiv lane take).1 hc)
      have hbody : dequant.lowbit.dequant_q1_0_row_loop0_loop0.body scale qs neg take out true lane
          = ok (ControlFlow.done (out, true)) := by
        rw [dequant.lowbit.dequant_q1_0_row_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨out, rfl, ?_⟩
      have hlane_take : lane.val = take.val := by omega
      refine ⟨rfl, by omega, ?_⟩
      rw [← hlane_take]
      exact hrec
  · exact hinv0

/-- The success domain of one Q1_0 block `b`: the block bytes are present, `d` is finite, and the
negated scale fits `i64`. -/
def q1_0BlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  b.val * 18 + 18 ≤ raw.val.length ∧
  f16Exp (f16LeHalf raw (18 * b.val)) ≠ 31 ∧
  I64.min ≤ -q1_0ScaleRef raw b.val ∧ -q1_0ScaleRef raw b.val ≤ I64.max

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (128·block) n` raw weights. -/
def q1_0RowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ q1_0RowValues raw (Nat.min (128 * block.val) n.val)

/-- One Q1_0 block-loop body step: decode block `block`, negate the scale, run the lane loop,
append `take = min 128 (n - 128·block)` values and advance. -/
theorem q1_0Block_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 127) / 128)
    (hdom : q1_0BlockDomain raw block)
    (hcap : out.val.length + (n.val - 128 * block.val) ≤ Usize.max)
    (hinv : q1_0RowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.lowbit.dequant_q1_0_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      q1_0RowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨hlen, hfin, hneglow, hneghigh⟩ := hdom
  obtain ⟨d, qs, hq, hdv, hcodes⟩ := q1_q2_block_ok raw block hlen hfin
  have hdv2 : d.val = q1_0ScaleRef raw block.val := hdv
  have hzero : (0#i64).val = 0 := by decide
  obtain ⟨neg, hsub, hnegu⟩ := dequant_sub_ok 0#i64 d
    (by rw [hzero, hdv2, Int.zero_sub]; exact hneglow)
    (by rw [hzero, hdv2, Int.zero_sub]; exact hneghigh)
  have hnegu' : neg.val = -q1_0ScaleRef raw block.val := by
    rw [hnegu, hzero, Int.zero_sub, hdv2]
  have hltmul : 128 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_128 (by rw [← hblocks]; exact hblock)
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.Q1_0_QK) (by rw [quant_Q1_0_QK_eq]; scalar_tac))
  have hivv' : i.val = 128 * block.val := by rw [hivv, quant_Q1_0_QK_eq]; simp; omega
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 128 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.Q1_0_QK i1
  have htakev' : take.val = Nat.min 128 (n.val - 128 * block.val) := by
    rw [htakev, hi1v', quant_Q1_0_QK_eq]; simp
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 128 := by rw [htakev']; exact min_le_left _ _
  have hcap' : out.val.length + take.val ≤ Usize.max := by
    have htake_X : take.val ≤ n.val - 128 * block.val := by rw [htakev']; exact min_le_right _ _
    omega
  have hinv0 : q1_0LaneInv raw block.val out out take 0#usize true := by
    refine ⟨rfl, by simp, ?_⟩
    simp
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q1_0Lane_loop_spec raw block.val out d qs neg take hdv2 hnegu' hcodes htake_le hcap' hinv0)
  obtain ⟨vals1, hp1, hlane⟩ := hp
  have hloop' : dequant.lowbit.dequant_q1_0_row_loop0_loop0 out true d qs neg take 0#usize
      = ok (vals1, true) := by rw [hloop, hp1]
  have hmin : Nat.min (128 * block.val) n.val = 128 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 128 * block.val + take.val = Nat.min (128 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min_128 block.val n.val (le_of_lt hltmul)
  have hmap : (List.range take.val).map (fun l => q1_0LaneVal raw block.val l)
      = (List.range take.val).map (fun j => q1_0Weight raw (128 * block.val + j)) := by
    apply List.map_congr_left
    intro l hl
    exact (q1_0Weight_lane raw block.val l (by have := List.mem_range.mp hl; omega)).symm
  have hinv_new : dequantVals vals1.val =
      dequantVals out0.val ++ q1_0RowValues raw (Nat.min (128 * block1.val) n.val) := by
    rw [hblock1v', hlane.2.2, hinv.2.2.2, hmin, List.append_assoc, hmap,
      q1_0RowValues_append, hsum]
  refine ⟨vals1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.lowbit.dequant_q1_0_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq]
  simp only [bind_tc_ok]
  change (do
      let r1 ← dequant.sub 0#i64 d
      let (ok2, err1, neg) ← (match r1 with
        | core.result.Result.Ok v => ok (true, none, v)
        | core.result.Result.Err e => ok (false, some e, 0#i64))
      let i ← block * quant.Q1_0_QK
      let i1 ← n - i
      let take ← core.cmp.Ord.min.trait_default core.cmp.OrdUsize quant.Q1_0_QK i1
      let (out1, ok3) ←
        dequant.lowbit.dequant_q1_0_row_loop0_loop0 out ok2 d qs neg take 0#usize
      let block1 ← block + 1#usize
      ok (ControlFlow.cont (out1, block1, ok3, err1)))
    = ok (ControlFlow.cont (vals1, block1, true, none))
  rw [hsub]
  simp only [bind_tc_ok]
  change (do
    let i ← block * quant.Q1_0_QK
    let i1 ← n - i
    let take ← core.cmp.Ord.min.trait_default core.cmp.OrdUsize quant.Q1_0_QK i1
    let (out1, ok3) ←
      dequant.lowbit.dequant_q1_0_row_loop0_loop0 out true d qs neg take 0#usize
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, ok3, none)))
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

/-- The Q1_0 block loop: `loop.spec_decr_nat` with `q1_0RowInv` and measure `blocks - block`, on the
success domain for every block below `blocks`.

Complexity: `blocks` iterations, each an 18-byte block read and at most 128 pushes; the measure is
`blocks - block`. -/
theorem q1_0Row_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 127) / 128)
    (hdom : ∀ b : Usize, b.val < blocks.val → q1_0BlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.lowbit.dequant_q1_0_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ q1_0RowValues raw n.val) := by
  unfold dequant.lowbit.dequant_q1_0_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => q1_0RowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, block, ok1, err⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : block.val < blocks.val
    · have hltmul : 128 * block.val < n.val :=
        lt_mul_of_lt_div_ceil_128 (by rw [← hblocks]; exact hlt)
      have hmin : Nat.min (128 * block.val) n.val = 128 * block.val :=
        Nat.min_eq_left (le_of_lt hltmul)
      have hlen_out : out.val.length = out0.val.length + 128 * block.val := by
        have h := congrArg List.length hrec
        rw [hmin] at h
        simpa [dequantVals, q1_0RowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 128 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        q1_0Block_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.lowbit.dequant_q1_0_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.lowbit.dequant_q1_0_row_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨trivial, ?_⟩
      have hblock_eq : block.val = blocks.val := by omega
      rw [hrec, hblock_eq, hblocks]
      have hminend : Nat.min (128 * ((n.val + 127) / 128)) n.val = n.val :=
        Nat.min_eq_right (le_mul_div_ceil_128 n.val)
      rw [hminend]
  · refine ⟨rfl, rfl, by simp, ?_⟩
    have h0 : Nat.min (128 * (0#usize).val) n.val = 0 := by simp
    rw [h0]
    simp [q1_0RowValues]

/-- **`Q1_0` row law.** On the block success domain, the extracted `dequant_q1_0_row` succeeds with
`Ok ()` and appends exactly the first `n` raw weights `q1_0RowValues raw n`. -/
theorem dequant_q1_0_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 127) / 128 → q1_0BlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_q1_0_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ q1_0RowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_128_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q1_0Row_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.lowbit.dequant_q1_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- `dequant_row_fixed` at `ggml_type = 41` (`GGML_TYPE_Q1_0`) is `dequant_q1_0_row`. -/
theorem dequant_row_fixed_q1_0 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 41#u32 raw n out = dequant.lowbit.dequant_q1_0_row raw n out := rfl

/-! ## One-block joint witnesses -/

/-- `getD` at offset `k + 2` of a list beginning two elements. -/
lemma low_getD_cons2 (a b : U8) (L : List U8) (k : Nat) :
    (a :: b :: L).getD (k + 2) 0#u8 = L.getD k 0#u8 := by
  rw [show k + 2 = (k + 1) + 1 by omega, List.getD_cons_succ, List.getD_cons_succ]

/-- The head of a `List.range n`-mapped row value list. -/
lemma lowRowValues_head32 (f : Nat → ℤ) : ((List.range 32).map f).getD 0 0 = f 0 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 32)[0]? = some 0 := by decide
  rw [h]
  rfl

lemma lowRowValues_head64 (f : Nat → ℤ) : ((List.range 64).map f).getD 0 0 = f 0 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 64)[0]? = some 0 := by decide
  rw [h]
  rfl

lemma lowRowValues_head128 (f : Nat → ℤ) : ((List.range 128).map f).getD 0 0 = f 0 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 128)[0]? = some 0 := by decide
  rw [h]
  rfl

/-- The 18-byte witness block (shared by Q1_0 and Q2_0): `d = 1.0`, sixteen zero `qs` bytes. -/
def low18WitBytes : List U8 := 0#u8 :: 60#u8 :: List.replicate 16 0#u8

theorem low18WitBytes_length : low18WitBytes.length = 18 := by simp [low18WitBytes]

def low18WitRaw : Slice U8 := Slice.from low18WitBytes (by rw [low18WitBytes_length]; scalar_tac)

theorem low18WitRaw_val : low18WitRaw.val = low18WitBytes := Slice.from_val _ _

/-- The 34-byte witness block (`PQ2_0`): `d = 1.0`, thirty-two zero `qs` bytes. -/
def low34WitBytes : List U8 := 0#u8 :: 60#u8 :: List.replicate 32 0#u8

theorem low34WitBytes_length : low34WitBytes.length = 34 := by simp [low34WitBytes]

def low34WitRaw : Slice U8 := Slice.from low34WitBytes (by rw [low34WitBytes_length]; scalar_tac)

theorem low34WitRaw_val : low34WitRaw.val = low34WitBytes := Slice.from_val _ _

lemma low18Wit_qs (j : Nat) : (low18WitRaw.val.getD (2 + j) 0#u8).val = 0 := by
  rw [low18WitRaw_val]
  simp only [low18WitBytes]
  rw [show 2 + j = j + 2 by omega, low_getD_cons2, getD_replicate_zero]
  rfl

lemma low34Wit_qs (j : Nat) : (low34WitRaw.val.getD (2 + j) 0#u8).val = 0 := by
  rw [low34WitRaw_val]
  simp only [low34WitBytes]
  rw [show 2 + j = j + 2 by omega, low_getD_cons2, getD_replicate_zero]
  rfl

lemma low18Wit_half : (f16LeHalf low18WitRaw 0).val = 15360 := by
  have heq := q8_0_f16LeHalf_zero_cons low18WitRaw 0#u8 60#u8 (List.replicate 16 0#u8)
    (by rw [low18WitRaw_val]; rfl)
  rw [heq, q8_0_u16_le_0_60]

lemma low34Wit_half : (f16LeHalf low34WitRaw 0).val = 15360 := by
  have heq := q8_0_f16LeHalf_zero_cons low34WitRaw 0#u8 60#u8 (List.replicate 32 0#u8)
    (by rw [low34WitRaw_val]; rfl)
  rw [heq, q8_0_u16_le_0_60]

/-- The witness scale is `1.0 = 2^32`. -/
lemma low18Wit_scale : q1_0ScaleRef low18WitRaw 0 = 2 ^ 32 := by
  have hv := low18Wit_half
  unfold q1_0ScaleRef
  rw [show 18 * 0 = 0 by ring]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

lemma low34Wit_scale : pq2_0ScaleRef low34WitRaw 0 = 2 ^ 32 := by
  have hv := low34Wit_half
  unfold pq2_0ScaleRef
  rw [show 34 * 0 = 0 by ring]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

lemma low18Wit_bit_zero : q1_0Bit low18WitRaw 0 0 = 0 := by
  unfold q1_0Bit q1_0CodeByte
  rw [show 18 * 0 + 2 + 0 / 8 = 2 + 0 by omega, low18Wit_qs 0]
  simp

lemma low18Wit_code_zero : q2_0Code low18WitRaw 0 0 = 0 := by
  unfold q2_0Code q2_0CodeByte
  rw [show 18 * 0 + 2 + 0 / 4 = 2 + 0 by omega, low18Wit_qs 0]
  simp

lemma low18Wit_code (l : Nat) : q2_0Code low18WitRaw 0 l = 0 := by
  unfold q2_0Code q2_0CodeByte
  rw [show 18 * 0 + 2 + l / 4 = 2 + l / 4 by omega, low18Wit_qs (l / 4)]
  simp

lemma low34Wit_code (l : Nat) : pq2_0Code low34WitRaw 0 l = 0 := by
  unfold pq2_0Code pq2_0CodeByte
  rw [show 34 * 0 + 2 + l / 4 = 2 + l / 4 by omega, low34Wit_qs (l / 4)]
  simp

/-- The first Q1_0 weight of the witness is `-2^32`. -/
lemma q1_0Wit_weight_zero : q1_0Weight low18WitRaw 0 = -(2 ^ 32) := by
  unfold q1_0Weight q1_0BitRef
  rw [show (0 / 128 : Nat) = 0 by norm_num, show (0 % 128 : Nat) = 0 by norm_num,
    low18Wit_bit_zero]
  rw [if_neg (by decide : ¬ ((0 : Nat) = 1))]
  rw [low18Wit_scale]

/-- The first Q2_0 weight of the witness is `-2^32`. -/
lemma q2_0Wit_weight_zero : q2_0Weight low18WitRaw 0 = -(2 ^ 32) := by
  unfold q2_0Weight
  rw [show (0 / 64 : Nat) = 0 by norm_num, show (0 % 64 : Nat) = 0 by norm_num,
    low18Wit_code_zero]
  have hs : q2_0ScaleRef low18WitRaw 0 = 2 ^ 32 := low18Wit_scale
  rw [hs]
  norm_num

/-- The first PQ2_0 weight of the witness is `-2^32`. -/
lemma pq2_0Wit_weight_zero : pq2_0Weight low34WitRaw 0 = -(2 ^ 32) := by
  unfold pq2_0Weight
  rw [show (0 / 128 : Nat) = 0 by norm_num, show (0 % 128 : Nat) = 0 by norm_num]
  have hc : pq2_0Code low34WitRaw 0 0 = 0 := by
    unfold pq2_0Code pq2_0CodeByte
    rw [show 34 * 0 + 2 + 0 / 4 = 2 + 0 by omega, low34Wit_qs 0]
    simp
  rw [hc, low34Wit_scale]
  norm_num

/-- The finite `d` of the witness is `0x3C00`. -/
lemma low18Wit_exp : f16Exp (f16LeHalf low18WitRaw 0) ≠ 31 := by
  have hv := low18Wit_half
  simp only [f16Exp, hv]
  norm_num

lemma low34Wit_exp : f16Exp (f16LeHalf low34WitRaw 0) ≠ 31 := by
  have hv := low34Wit_half
  simp only [f16Exp, hv]
  norm_num

/-- **Joint non-vacuity of `q1_0BlockDomain` at block `0`.** -/
theorem q1_0BlockDomain_witness : q1_0BlockDomain low18WitRaw 0#usize := by
  have hb : (0#usize).val = 0 := by decide
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [hb, low18WitRaw_val, low18WitBytes_length]
  · rw [hb, show 18 * 0 = 0 by ring]
    exact low18Wit_exp
  · rw [hb, low18Wit_scale, I64.min_eq]
    norm_num
  · rw [hb, low18Wit_scale, I64.max_eq]
    norm_num

/-- **Joint non-vacuity of `q2_0BlockDomain` at block `0`.** -/
theorem q2_0BlockDomain_witness : q2_0BlockDomain low18WitRaw 0#usize := by
  have hb : (0#usize).val = 0 := by decide
  have hs : q2_0ScaleRef low18WitRaw 0 = 2 ^ 32 := low18Wit_scale
  refine ⟨?_, ?_, ?_⟩
  · simp [hb, low18WitRaw_val, low18WitBytes_length]
  · rw [hb, show 18 * 0 = 0 by ring]
    exact low18Wit_exp
  · intro l hl
    rw [hb, low18Wit_code l, hs]
    norm_num [I64.min_eq, I64.max_eq]

/-- **Joint non-vacuity of `pq2_0BlockDomain` at block `0`.** -/
theorem pq2_0BlockDomain_witness : pq2_0BlockDomain low34WitRaw 0#usize := by
  have hb : (0#usize).val = 0 := by decide
  refine ⟨?_, ?_, ?_⟩
  · simp [hb, low34WitRaw_val, low34WitBytes_length]
  · rw [hb, show 34 * 0 = 0 by ring]
    exact low34Wit_exp
  · intro l hl
    rw [hb, low34Wit_code l, low34Wit_scale]
    norm_num [I64.min_eq, I64.max_eq]

/-- **`Q1_0` one-block row witness.** A single witness block: the extracted row succeeds on the
empty vector, appends the `q1_0RowValues`, and element `0` is `-2^32`. -/
theorem q1_0Witness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_q1_0_row low18WitRaw 128#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = q1_0RowValues low18WitRaw 128 ∧
      (out1.val.getD 0 0#i64).val = -(2 ^ 32) := by
  have hdom : ∀ b : Usize, b.val < ((128#usize).val + 127) / 128 →
      q1_0BlockDomain low18WitRaw b := by
    intro b hb
    have hR : (((128#usize).val + 127) / 128) = 1 := by decide
    rw [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact q1_0BlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_q1_0_row_spec low18WitRaw 128#usize (alloc.vec.Vec.new I64) hdom (by
      rw [q8_0_vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = q1_0RowValues low18WitRaw 128 := by
    rw [q8_0_vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  · have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
    rw [q8_0_dequantVals_getD] at h0
    rw [h0, q1_0RowValues, lowRowValues_head128]
    exact q1_0Wit_weight_zero

/-- **`Q2_0` one-block row witness.** -/
theorem q2_0Witness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_q2_0_row low18WitRaw 64#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = q2_0RowValues low18WitRaw 64 ∧
      (out1.val.getD 0 0#i64).val = -(2 ^ 32) := by
  have hdom : ∀ b : Usize, b.val < ((64#usize).val + 63) / 64 →
      q2_0BlockDomain low18WitRaw b := by
    intro b hb
    have hR : (((64#usize).val + 63) / 64) = 1 := by decide
    rw [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact q2_0BlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_q2_0_row_spec low18WitRaw 64#usize (alloc.vec.Vec.new I64) hdom (by
      rw [q8_0_vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = q2_0RowValues low18WitRaw 64 := by
    rw [q8_0_vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  · have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
    rw [q8_0_dequantVals_getD] at h0
    rw [h0, q2_0RowValues, lowRowValues_head64]
    exact q2_0Wit_weight_zero

/-- **`PQ2_0` one-block row witness.** -/
theorem pq2_0Witness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_pq2_0_row low34WitRaw 128#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = pq2_0RowValues low34WitRaw 128 ∧
      (out1.val.getD 0 0#i64).val = -(2 ^ 32) := by
  have hdom : ∀ b : Usize, b.val < ((128#usize).val + 127) / 128 →
      pq2_0BlockDomain low34WitRaw b := by
    intro b hb
    have hR : (((128#usize).val + 127) / 128) = 1 := by decide
    rw [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact pq2_0BlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_pq2_0_row_spec low34WitRaw 128#usize (alloc.vec.Vec.new I64) hdom (by
      rw [q8_0_vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = pq2_0RowValues low34WitRaw 128 := by
    rw [q8_0_vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  · have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
    rw [q8_0_dequantVals_getD] at h0
    rw [h0, pq2_0RowValues, lowRowValues_head128]
    exact pq2_0Wit_weight_zero

end

end apeiron_word
