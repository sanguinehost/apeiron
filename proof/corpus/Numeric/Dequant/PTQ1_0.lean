/-
Row law for the ternary face `PTQ1_0` (`APEIRON-TERNARY-W3-T4d`).

`DequantTQ1_0.lean` proves the block laws and the order map
(`ptq1_0Byte`/`ptq1_0Plane` with the `lane16`/`lane8`/`tail` segments);
`DequantTQ1_0Row.lean` proves the `TQ1_0` row law. This file mirrors the `TQ1_0` composition at
block length 128 over the same 24-byte `qs` and 2-byte `qh`. The `PTQ1_0_STAGES` walk is three
stages `[32, 16, 8]`: the width-`32` stage is null over the 24-byte `qs` (base `0 → 0`, pos
`0 → 0`), width `16` walks `(0, 0)` to `(80, 16)`, width `8` walks `(80, 16)` to `(120, 24)`, and
the `qh` tail (4 planes × 2 lanes) reaches `128`. The scale `d` is read at byte `28·block + 26`.

`lt_mul_of_lt_div_ceil_128`, `le_mul_div_ceil_128` and `mul_block_add_min_128` are imported from
`Numeric.Dequant.Q1Q2Block` rather than restated.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed literal
facts (small array and byte reads), never on an extracted body.
-/
import Numeric.Dequant.TQ1_0Row
import Numeric.Dequant.Q1Q2Block

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section



/-! ## `PTQ1_0` innermost lane loop (24-byte `qs` walk) -/

lemma ptq1_0BlockVals_length (raw : Slice U8) (b : Nat) :
    (ptq1_0BlockVals raw b).length = 128 := by
  simp [ptq1_0BlockVals]

/-- Entry `r < 128` of the staged block stream. -/
lemma ptq1_0BlockVals_getD (raw : Slice U8) (b r : Nat) (hr : r < 128) :
    (ptq1_0BlockVals raw b).getD r 0 = ptq1_0Weight raw b r := by
  unfold ptq1_0BlockVals
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 128)[r]? = some r := by
    rw [List.getElem?_eq_getElem (by simpa using hr)]; simp
  rw [h]; rfl

def ptq1_0EmitInv (raw : Slice U8) (b : Nat) (take : Usize) (start out : alloc.vec.Vec I64)
    (pos : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧
  dequantVals out.val = dequantVals start.val ++
    (ptq1_0BlockVals raw b).take (Nat.min pos.val take.val)

/-- Lane-loop invariant: `ptq1_0EmitInv` and `pos = pos0 + lane`. -/
def ptq1_0LaneInv (raw : Slice U8) (b : Nat) (take : Usize) (pos0 : Nat) (width : Usize)
    (start out : alloc.vec.Vec I64) (pos : Usize) (lane : Usize)
    (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ptq1_0EmitInv raw b take start out pos ok1 err ∧ pos.val = pos0 + lane.val ∧
    lane.val ≤ width.val

theorem ptq1_0Lane_body_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 24#usize) (take : Usize) (base width plane pos0 : Usize)
    (start out : alloc.vec.Vec I64) (pos lane : Usize)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 24 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 28 + j) 0#u8).val)
    (hplane5 : plane.val < 5)
    (hplane : ptq1_0Plane pos.val = plane.val)
    (hbyte : ptq1_0Byte pos.val = base.val + lane.val)
    (hlane : lane.val < width.val)
    (hbl : base.val + lane.val < 24)
    (hpos128 : pos.val < 128)
    (hmul : I64.min ≤
        scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ∧
      scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : ptq1_0LaneInv raw b take pos0.val width start out pos lane true none) :
    ∃ (out2 : alloc.vec.Vec I64) (pos2 lane2 : Usize),
      dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0_loop0.body scale qs take base
          width plane out true none pos lane = ok (ControlFlow.cont (out2, true, none, pos2, lane2)) ∧
      ptq1_0LaneInv raw b take pos0.val width start out2 pos2 lane2 true none ∧
      pos2.val = pos.val + 1 ∧ lane2.val = lane.val + 1 := by
  obtain ⟨hemit, hposrel, hlanele⟩ := hinv
  obtain ⟨hok, herr, hrec⟩ := hemit
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := base) (y := lane) (by scalar_tac))
  have hbnd : i.val < qs.val.length := by rw [hiv, Array.property qs]; exact hbl
  obtain ⟨i1, hi1, hi1eq⟩ := WP.spec_imp_exists (Array.index_usize_spec qs i hbnd)
  have hi1val_i : i1.val = (qs.val.getD i.val 0#u8).val := by
    rw [hi1eq]; exact arr_getD_val_eq qs i hbnd
  have hi1val : i1.val = (qs.val.getD (base.val + lane.val) 0#u8).val := by
    rw [hi1val_i, hiv]
  obtain ⟨i2, htrit, hi2v⟩ := trit_spec i1 plane (by omega)
  have hi2val : i2.val = tritRef i1 plane.val := hi2v
  obtain ⟨i3, hsub, hi3v⟩ := WP.spec_imp_exists
    (I64.sub_spec (x := i2) (y := 1#i64)
      (by rw [hi2val, I64.min_eq, show (1#i64).val = 1 from rfl]
          have := tritRef_nonneg i1 plane.val; omega)
      (by rw [hi2val, I64.max_eq, show (1#i64).val = 1 from rfl]
          have := tritRef_le_two i1 plane.val; omega))
  have hi3val : i3.val = tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1 := by
    rw [hi3v, hi2val, tritRef_congr hi1val plane.val]; norm_num
  obtain ⟨v, hmulv, hvv⟩ := q6k_mul_ok scale i3 (by rw [hi3val]; exact hmul.1)
    (by rw [hi3val]; exact hmul.2)
  have hvval : v.val = (ptq1_0BlockVals raw b)[pos.val]'(by
      rw [ptq1_0BlockVals_length]; exact hpos128) := by
    have hcong := tritRef_congr (hqs (base.val + lane.val) hbl) plane.val
    rw [hvv, hi3val, hsc, hcong, ← hbyte]
    rw [list_getD_getElem (ptq1_0BlockVals raw b) pos.val
      (by rw [ptq1_0BlockVals_length]; exact hpos128) (0 : ℤ), ptq1_0BlockVals_getD raw b pos.val hpos128]
    unfold ptq1_0Weight
    rw [hplane]
  by_cases hlt : pos.val < take.val
  · obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.push_spec out v hcap)
    obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
    refine ⟨out2, pos2, lane2, ?_, ?_, hpos2, hlane2⟩
    · unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane), hok, if_pos rfl]
      simp only [hi, hi1, htrit, hsub, hmulv, bind_tc_ok]
      rw [if_pos ((UScalar.lt_equiv pos take).2 hlt)]
      simp only [hpush, haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, by rw [hpos2, hposrel]; omega, by rw [hlane2]; omega⟩
      have hmin1 : (pos.val + 1).min take.val = pos.val + 1 :=
        Nat.min_eq_left (by omega : pos.val + 1 ≤ take.val)
      have hmin2 : pos.val.min take.val = pos.val := Nat.min_eq_left (le_of_lt hlt)
      rw [hpushv, dequantVals_append, hrec]
      simp only [dequantVals, List.map_cons, List.map_nil, hvval]
      rw [hpos2, hmin1, hmin2,
        List.take_succ_eq_append_getElem (by rw [ptq1_0BlockVals_length]; exact hpos128),
        List.append_assoc]
  · obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
    refine ⟨out, pos2, lane2, ?_, ?_, hpos2, hlane2⟩
    · unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane), hok, if_pos rfl]
      simp only [hi, hi1, htrit, hsub, hmulv, bind_tc_ok]
      rw [if_neg (fun hc => hlt ((UScalar.lt_equiv pos take).1 hc))]
      simp only [haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, by rw [hpos2, hposrel]; omega, by rw [hlane2]; omega⟩
      have h1 : (pos.val + 1).min take.val = take.val :=
        Nat.min_eq_right (by omega : take.val ≤ pos.val + 1)
      have h2 : pos.val.min take.val = take.val :=
        Nat.min_eq_right (by omega : take.val ≤ pos.val)
      rw [hrec, hpos2, h1, h2]

/-- **Innermost lane loop.** From `lane = 0` at word offset `pos0` the loop emits `width` staged
values (truncated by `take`), leaves `ok = true`, and advances `pos` by `width`. -/
theorem ptq1_0Lane_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 24#usize) (take : Usize) (base width plane pos0 : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 24 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 28 + j) 0#u8).val)
    (hplane5 : plane.val < 5)
    (hrun : ∀ lane : Usize, lane.val < width.val →
      ptq1_0Byte (pos0.val + lane.val) = base.val + lane.val ∧
      ptq1_0Plane (pos0.val + lane.val) = plane.val)
    (hpos128 : pos0.val + width.val ≤ 128)
    (hbase : base.val + width.val ≤ 24)
    (hmul : ∀ lane : Usize, lane.val < width.val →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ∧
      scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ≤ I64.max)
    (hcap : out0.val.length + 128 < Usize.max)
    (hinv0 : ptq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0_loop0 out0 true none scale
        qs take pos0 base width plane 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none, pos') ∧ pos'.val = pos0.val + width.val ∧
        ptq1_0EmitInv raw b take start out pos' true none) := by
  have hinit : ptq1_0LaneInv raw b take pos0.val width start out0 pos0 0#usize true none :=
    ⟨hinv0, by simp, by simp⟩
  unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      width.val - s.2.2.2.2.val)
    (inv := fun s => ptq1_0LaneInv raw b take pos0.val width start s.1 s.2.2.2.1 s.2.2.2.2 s.2.1 s.2.2.1)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none, pos') ∧ pos'.val = pos0.val + width.val ∧
      ptq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, lane⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hemit, hposrel, hlanele⟩ := hinv'
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : lane.val < width.val
    · have hpos128' : pos.val < 128 := by
        have hpx : pos.val = pos0.val + lane.val := hposrel
        omega
      have houtlen : out.val.length ≤ start.val.length + 128 := by
        have h1 := congrArg List.length hrec
        simp only [dequantVals, List.length_append, List.length_map] at h1
        rw [h1]
        have h2 := (List.take_sublist (l := ptq1_0BlockVals raw b)
          (i := Nat.min pos.val take.val)).length_le
        simp only [ptq1_0BlockVals_length] at h2
        omega
      have hstart : start.val.length ≤ out0.val.length := by
        have h1 := congrArg List.length hinv0.2.2
        simp only [dequantVals, List.length_append, List.length_map] at h1
        rw [h1]
        have h2 := (List.take_sublist (l := ptq1_0BlockVals raw b)
          (i := Nat.min pos0.val take.val)).length_le
        simp only [ptq1_0BlockVals_length] at h2
        omega
      have hcap' : out.val.length < Usize.max := by
        have := houtlen; omega
      obtain ⟨out2, pos2, lane2, hbody, hinv2, hpos2, hlane2⟩ :=
        ptq1_0Lane_body_step raw b scale qs take base width plane pos0 start out pos lane
          hsc hqs hplane5 (by rw [hposrel]; exact (hrun lane hlt).2)
          (by rw [hposrel]; exact (hrun lane hlt).1) hlt (by omega) hpos128'
          (hmul lane hlt) hcap' ⟨⟨rfl, rfl, hrec⟩, hposrel, hlanele⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show width.val - lane2.val < width.val - lane.val
      omega
    · have hnot : ¬ (lane < width) := fun hc => hlt ((UScalar.lt_equiv lane width).1 hc)
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0_loop0.body scale qs
          take base width plane out true none pos lane =
          ok (ControlFlow.done (out, true, none, pos)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hlw : lane.val = width.val := by omega
      refine ⟨out, pos, rfl, by rw [hposrel, hlw], rfl, rfl, hrec⟩
  · exact hinit


/-! ## `PTQ1_0` staged `qs` walk -/

/-- `qs` segment predicate: output position `pos0 + p·width + l` reads byte `base + l` at plane `p`,
for every plane `p < 5` and lane `l < width`. -/
def ptq1_0QsOK (_raw : Slice U8) (_b : Nat) (pos0 base width : Nat) : Prop :=
  ∀ p, p < 5 → ∀ l, l < width →
    ptq1_0Byte (pos0 + p * width + l) = base + l ∧
    ptq1_0Plane (pos0 + p * width + l) = p

/-- The `qs` segment `0 … 79` of a `PTQ1_0` block: width `16`, base `0`. -/
lemma ptq1_0QsOK_zero16 (raw : Slice U8) (b : Nat) : ptq1_0QsOK raw b 0 0 16 := by
  intro p hp l hl
  exact ⟨by simpa using ptq1_0Byte_lane16 p l hp hl,
    by simpa using ptq1_0Plane_lane16 p l hp hl⟩

/-- The `qs` segment `80 … 119` of a `PTQ1_0` block: width `8`, base `16`. -/
lemma ptq1_0QsOK_lane8 (raw : Slice U8) (b : Nat) : ptq1_0QsOK raw b 80 16 8 := by
  intro p hp l hl
  exact ⟨ptq1_0Byte_lane8 p l hp hl, ptq1_0Plane_lane8 p l hp hl⟩

/-- `PTQ1_0_STAGES` has three entries. -/
lemma PTQ1_0_STAGES_len : dequant.lowbit.PTQ1_0_STAGES.length = 3 :=
  Array.length_eq dequant.lowbit.PTQ1_0_STAGES

/-- Reading `PTQ1_0_STAGES` at `i` is `32`, then `16`, then `8`. -/
lemma PTQ1_0_STAGES_val (i : Nat) (h : i < dequant.lowbit.PTQ1_0_STAGES.length) :
    (dequant.lowbit.PTQ1_0_STAGES.val[i]'h).val =
      if i = 0 then 32 else if i = 1 then 16 else 8 := by
  have hd : (dequant.lowbit.PTQ1_0_STAGES.val.getD i 0#usize).val =
      (if i = 0 then 32 else if i = 1 then 16 else 8) := by
    have h3 : i < 3 := by have := PTQ1_0_STAGES_len; omega
    have : i = 0 ∨ i = 1 ∨ i = 2 := by omega
    rcases this with h0 | h1 | h2
    · subst h0; unfold dequant.lowbit.PTQ1_0_STAGES; decide
    · subst h1; unfold dequant.lowbit.PTQ1_0_STAGES; decide
    · subst h2; unfold dequant.lowbit.PTQ1_0_STAGES; decide
  rwa [List.getD_eq_getElem (l := dequant.lowbit.PTQ1_0_STAGES.val) (d := 0#usize) h] at hd

/-! ## `qs` plane loop -/

/-- **`qs` plane loop.** From a segment `ptq1_0QsOK raw b pos0 base width`, the five planes emit the
staged values `pos0 … pos0 + 5·width - 1`, leaving the emission invariant. Capacity pad `512`. -/
theorem ptq1_0Plane_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 24#usize) (take : Usize) (base width pos0 : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 24 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 28 + j) 0#u8).val)
    (hOK : ptq1_0QsOK raw b pos0.val base.val width.val)
    (hpos128 : pos0.val + 5 * width.val ≤ 128)
    (hbase : base.val + width.val ≤ 24)
    (hprod : ∀ p, p < 5 → ∀ l, l < width.val →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (base.val + l) 0#u8) p - 1) ≤ I64.max)
    (hcap : out0.val.length + 512 < Usize.max)
    (hinv0 : ptq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0 out0 true none scale qs take
        pos0 base width 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none, pos') ∧ pos'.val = pos0.val + 5 * width.val ∧
        ptq1_0EmitInv raw b take start out pos' true none) := by
  unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      5 - s.2.2.2.2.val)
    (inv := fun s => ptq1_0EmitInv raw b take start s.1 s.2.2.2.1 s.2.1 s.2.2.1 ∧
      s.2.2.2.1.val = pos0.val + s.2.2.2.2.val * width.val ∧ s.2.2.2.2.val ≤ 5)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none, pos') ∧ pos'.val = pos0.val + 5 * width.val ∧
      ptq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, plane⟩ ⟨hemit, hposrel, hplanele⟩
    dsimp only at hemit hposrel hplanele ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : plane.val < 5
    · have hpos128' : pos.val + width.val ≤ 128 := by
        have he : pos.val + width.val = pos0.val + (plane.val + 1) * width.val := by
          rw [hposrel]; ring
        rw [he]
        have hpl : plane.val + 1 ≤ 5 := by omega
        calc pos0.val + (plane.val + 1) * width.val
            ≤ pos0.val + 5 * width.val := by
              exact Nat.add_le_add_left (Nat.mul_le_mul_right _ hpl) _
          _ ≤ 128 := hpos128
      have hrun : ∀ lane : Usize, lane.val < width.val →
          ptq1_0Byte (pos.val + lane.val) = base.val + lane.val ∧
          ptq1_0Plane (pos.val + lane.val) = plane.val := by
        intro lane hlane
        have h := hOK plane.val hlt lane.val hlane
        have hidx : pos.val + lane.val = pos0.val + plane.val * width.val + lane.val := by
          rw [hposrel]
        rw [hidx]
        exact h
      have hmul' : ∀ lane : Usize, lane.val < width.val →
          I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ∧
          scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ≤ I64.max :=
        fun lane hlane => hprod plane.val hlt lane.val hlane
      have hstart : start.val.length ≤ out0.val.length := by
        have h := congrArg List.length hinv0.2.2
        simp only [dequantVals, List.length_append, List.length_map] at h
        have h4 := (List.take_sublist (l := ptq1_0BlockVals raw b)
          (i := Nat.min pos0.val take.val)).length_le
        simp only [ptq1_0BlockVals_length] at h4
        omega
      have hlen : out.val.length ≤ out0.val.length + 128 := by
        have h := congrArg List.length hrec
        simp only [dequantVals, List.length_append, List.length_map] at h
        have h4 := (List.take_sublist (l := ptq1_0BlockVals raw b)
          (i := Nat.min pos.val take.val)).length_le
        simp only [ptq1_0BlockVals_length] at h4
        omega
      obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
        (ptq1_0Lane_loop_spec raw b scale qs take base width plane pos start out
          hsc hqs hlt hrun hpos128' hbase hmul' (by omega) ⟨rfl, rfl, hrec⟩)
      obtain ⟨out2, pos2, hp_eq, hpos2, hemit2⟩ := hp
      have hloop' : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0_loop0 out true none
          scale qs take pos base width plane 0#usize = ok (out2, true, none, pos2) := by
        rw [hloop, hp_eq]
      obtain ⟨plane1, haddp, hplane1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := plane) (y := 1#usize) (by scalar_tac))
      have hplane1 : plane1.val = plane.val + 1 := by rw [hplane1v]; simp
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0.body scale qs take
          base width out true none pos plane = ok (ControlFlow.cont (out2, true, none, pos2, plane1)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0.body
        rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), if_pos rfl]
        simp only [hloop', haddp, bind_tc_ok]
        rfl
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨hemit2, by rw [hpos2, hposrel, hplane1]; ring, by omega⟩, ?_⟩
      show 5 - plane1.val < 5 - plane.val
      omega
    · have hnot : ¬ (plane < 5#usize) := fun hc => hlt ((UScalar.lt_equiv plane 5#usize).1 hc)
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0.body scale qs take
          base width out true none pos plane = ok (ControlFlow.done (out, true, none, pos)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hplane5 : plane.val = 5 := by omega
      refine ⟨out, pos, rfl, by rw [hposrel, hplane5], rfl, rfl, hrec⟩
  · exact ⟨hinv0, by simp, by simp⟩

/-! ## `qs` base loop -/

/-- **`qs` base loop.** Runs the plane loop for successive base offsets `base0 + k·width` while
`base0 + k·width + width ≤ 24`, advancing `pos` by `5·width` each time. Capacity pad `768`. -/
theorem ptq1_0Base_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 24#usize) (take : Usize) (pos0 base0 width : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hwidth : 0 < width.val)
    (hwidth32 : width.val ≤ 32)
    (hbase0 : base0.val ≤ 24)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 24 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 28 + j) 0#u8).val)
    (hOK : ∀ k, base0.val + width.val * k + width.val ≤ 24 →
      ptq1_0QsOK raw b (pos0.val + 5 * width.val * k) (base0.val + width.val * k) width.val)
    (hpos128 : ∀ k, base0.val + width.val * k + width.val ≤ 24 →
      pos0.val + 5 * width.val * k + 5 * width.val ≤ 128)
    (hprod : ∀ k, base0.val + width.val * k + width.val ≤ 24 → ∀ p, p < 5 → ∀ l, l < width.val →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base0.val + width.val * k + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (base0.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max)
    (hcap : out0.val.length + 768 < Usize.max)
    (hinv0 : ptq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0 out0 true none scale qs take pos0
        base0 width)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos base : Usize),
        p = (out, true, none, pos, base) ∧ base.val ≤ 24 ∧ base.val + width.val > 24 ∧
        (∃ k, base.val = base0.val + width.val * k ∧ pos.val = pos0.val + 5 * width.val * k) ∧
        ptq1_0EmitInv raw b take start out pos true none) := by
  unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      24 - s.2.2.2.2.val)
    (inv := fun s => ptq1_0EmitInv raw b take start s.1 s.2.2.2.1 s.2.1 s.2.2.1 ∧
      s.2.2.2.2.val ≤ 24 ∧
      (∃ k, s.2.2.2.2.val = base0.val + width.val * k ∧
        s.2.2.2.1.val = pos0.val + 5 * width.val * k))
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos base : Usize),
      p = (out, true, none, pos, base) ∧ base.val ≤ 24 ∧ base.val + width.val > 24 ∧
      (∃ k, base.val = base0.val + width.val * k ∧ pos.val = pos0.val + 5 * width.val * k) ∧
      ptq1_0EmitInv raw b take start out pos true none)
  · rintro ⟨out, ok1, err, pos, base⟩ ⟨hemit, hbasele, hk⟩
    dsimp only at hemit hbasele hk ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    obtain ⟨k, hbasek, hposk⟩ := hk
    rw [hok, herr]
    obtain ⟨i, hiadd, hiv⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := base) (y := width) (by scalar_tac))
    have hiv' : i.val = base.val + width.val := by rw [hiv]
    have hlen : out.val.length ≤ out0.val.length + 128 := by
      have h := congrArg List.length hrec
      simp only [dequantVals, List.length_append, List.length_map] at h
      have h4 := (List.take_sublist (l := ptq1_0BlockVals raw b)
        (i := Nat.min pos.val take.val)).length_le
      simp only [ptq1_0BlockVals_length] at h4
      have hstart : start.val.length ≤ out0.val.length := by
        have h5 := congrArg List.length hinv0.2.2
        simp only [dequantVals, List.length_append, List.length_map] at h5
        have h6 := (List.take_sublist (l := ptq1_0BlockVals raw b)
          (i := Nat.min pos0.val take.val)).length_le
        simp only [ptq1_0BlockVals_length] at h6
        omega
      omega
    by_cases hlt : base.val + width.val ≤ 24
    · have hkcond : base0.val + width.val * k + width.val ≤ 24 := by
        rw [hbasek] at hlt; omega
      have hOKk : ptq1_0QsOK raw b pos.val base.val width.val := by
        have h := hOK k hkcond
        rw [hposk, hbasek]
        exact h
      have hpos128k : pos.val + 5 * width.val ≤ 128 := by
        have h := hpos128 k hkcond
        rw [← hposk] at h
        exact h
      have hprodk : ∀ p, p < 5 → ∀ l, l < width.val →
          I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + l) 0#u8) p - 1) ∧
          scale.val * (tritRef (qs.val.getD (base.val + l) 0#u8) p - 1) ≤ I64.max := by
        intro p hp l hl
        have hh := hprod k hkcond p hp l hl
        have hidx : base.val + l = base0.val + width.val * k + l := by rw [hbasek]
        rwa [hidx]
      obtain ⟨q, hplane_call, hq⟩ := WP.spec_imp_exists
        (ptq1_0Plane_loop_spec raw b scale qs take base width pos start out hsc hqs hOKk
          hpos128k hlt hprodk (by omega) ⟨rfl, rfl, hrec⟩)
      obtain ⟨out1, pos1, hq_eq, hpos1, hemit1⟩ := hq
      have hplane' : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0_loop0 out true none scale
          qs take pos base width 0#usize = ok (out1, true, none, pos1) := by rw [hplane_call, hq_eq]
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0.body scale qs take width out
          true none pos base = ok (ControlFlow.cont (out1, true, none, pos1, i)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0.body
        rw [hiadd]
        simp only [bind_tc_ok]
        rw [if_pos (by rw [UScalar.le_equiv, hiv']; exact hlt)]
        simp only [if_true, hplane', bind_tc_ok]
        rfl
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨hemit1, by rw [hiv']; omega, ⟨k + 1, ?_, ?_⟩⟩, ?_⟩
      · rw [hiv', hbasek]; ring
      · rw [hpos1, hposk]; ring
      · show 24 - i.val < 24 - base.val
        omega
    · have hnlt : ¬ (i ≤ 24#usize) := fun hc => hlt (by
        rw [UScalar.le_equiv] at hc; rw [hiv'] at hc; exact hc)
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0.body scale qs take width out
          true none pos base = ok (ControlFlow.done (out, true, none, pos, base)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0.body
        rw [hiadd]
        simp only [bind_tc_ok]
        rw [if_neg hnlt]
      have hibig : 24 < i.val := by
        rw [UScalar.le_equiv] at hnlt; omega
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨out, pos, base, rfl, hbasele, by omega, ⟨k, hbasek, hposk⟩, rfl, rfl, hrec⟩
  · exact ⟨hinv0, hbase0, ⟨0, by simp, by simp⟩⟩

/-! ## `qs` segment product bounds -/

/-- From a `ptq1_0QsOK raw b pos0 base width` segment and the block domain, the `i64` product bound
holds at every lane of the segment. -/
lemma ptq1_0QsOK_prod (raw : Slice U8) (b : Nat) (scale : I64) (qs : Array U8 24#usize)
    (pos0 base width : Nat)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 24 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 28 + j) 0#u8).val)
    (hOK : ptq1_0QsOK raw b pos0 base width)
    (hbound : ∀ r, r < 128 → I64.min ≤ ptq1_0Weight raw b r ∧ ptq1_0Weight raw b r ≤ I64.max)
    (hr128 : ∀ p, p < 5 → ∀ l, l < width → pos0 + p * width + l < 128)
    (hbase24 : base + width ≤ 24) :
    ∀ p, p < 5 → ∀ l, l < width →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (base + l) 0#u8) p - 1) ≤ I64.max := by
  intro p hp l hl
  have h := hOK p hp l hl
  obtain ⟨hbyte, hplane⟩ := h
  have hr := hr128 p hp l hl
  have hbl : base + l < 24 := by omega
  have hidx : b * 28 + ptq1_0Byte (pos0 + p * width + l) = b * 28 + (base + l) := by rw [hbyte]
  have hbyteEq : (qs.val.getD (base + l) 0#u8).val =
      (raw.val.getD (b * 28 + ptq1_0Byte (pos0 + p * width + l)) 0#u8).val := by
    rw [hidx]; exact hqs (base + l) hbl
  have htrit : tritRef (qs.val.getD (base + l) 0#u8) p =
      tritRef (raw.val.getD (b * 28 + ptq1_0Byte (pos0 + p * width + l)) 0#u8)
        (ptq1_0Plane (pos0 + p * width + l)) := by
    have h1 := tritRef_congr hbyteEq p
    rw [hplane]
    exact h1
  have hW : ptq1_0Weight raw b (pos0 + p * width + l) =
      ptq1_0ScaleRef raw b * (tritRef (qs.val.getD (base + l) 0#u8) p - 1) := by
    unfold ptq1_0Weight
    rw [htrit]
  have hb0 := hbound (pos0 + p * width + l) hr
  rw [hW, ← hsc] at hb0
  exact hb0

/-- `PTQ1_0_STAGES` read through `getD`. -/
lemma PTQ1_0_STAGES_getD (i : Nat) (h : i < 3) :
    (dequant.lowbit.PTQ1_0_STAGES.val.getD i 0#usize).val =
      if i = 0 then 32 else if i = 1 then 16 else 8 := by
  rcases (by omega : i = 0 ∨ i = 1 ∨ i = 2) with h0 | h1 | h2
  · subst h0; unfold dequant.lowbit.PTQ1_0_STAGES; decide
  · subst h1; unfold dequant.lowbit.PTQ1_0_STAGES; decide
  · subst h2; unfold dequant.lowbit.PTQ1_0_STAGES; decide

/-- The emitted output of an `emitInv` state exceeds its block start by at most `128`. -/
lemma ptq1_0EmitInv_length {raw : Slice U8} {b : Nat} {take : Usize}
    {start out : alloc.vec.Vec I64} {pos : Usize} {ok1 : Bool} {err : Option error.EngineError}
    (h : ptq1_0EmitInv raw b take start out pos ok1 err) :
    out.val.length ≤ start.val.length + 128 := by
  obtain ⟨-, -, he⟩ := h
  have h1 := congrArg List.length he
  simp only [dequantVals, List.length_append, List.length_map] at h1
  have h2 := (List.take_sublist (l := ptq1_0BlockVals raw b)
    (i := Nat.min pos.val take.val)).length_le
  simp only [ptq1_0BlockVals_length] at h2
  omega

/-- The block start of an `emitInv` state is no longer than its output. -/
lemma ptq1_0EmitInv_start_le {raw : Slice U8} {b : Nat} {take : Usize}
    {start out : alloc.vec.Vec I64} {pos : Usize} {ok1 : Bool} {err : Option error.EngineError}
    (h : ptq1_0EmitInv raw b take start out pos ok1 err) :
    start.val.length ≤ out.val.length := by
  obtain ⟨-, -, he⟩ := h
  have h1 := congrArg List.length he
  simp only [dequantVals, List.length_append, List.length_map] at h1
  omega

/-- The `PTQ1_0` staged phase: `(pos, base, stage)` after `stage` stages. Stage `0` (width `32`)
is null over the 24-byte `qs`; stage `1` (width `16`) emits `80` positions from `(0, 0)` to
`(80, 16)`; stage `2` (width `8`) emits `40` positions from `(80, 16)` to `(120, 24)`. -/
def ptq1_0StagePhase (pos base stage : Nat) : Prop :=
  (stage = 0 ∧ base = 0 ∧ pos = 0) ∨
  (stage = 1 ∧ base = 0 ∧ pos = 0) ∨
  (stage = 2 ∧ base = 16 ∧ pos = 80) ∨
  (stage = 3 ∧ base = 24 ∧ pos = 120)

/-- One `PTQ1_0` stage-loop body step: read the stage width, run the base loop, advance the stage. -/
theorem ptq1_0Stage_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 24#usize) (take : Usize) (stage : Usize)
    (start out : alloc.vec.Vec I64) (pos base width : Usize)
    (hstage : stage.val < 3)
    (hwidthread : Array.index_usize dequant.lowbit.PTQ1_0_STAGES stage = ok width)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 24 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 28 + j) 0#u8).val)
    (hOK : ∀ k, base.val + width.val * k + width.val ≤ 24 →
      ptq1_0QsOK raw b (pos.val + 5 * width.val * k) (base.val + width.val * k) width.val)
    (hpos128 : ∀ k, base.val + width.val * k + width.val ≤ 24 →
      pos.val + 5 * width.val * k + 5 * width.val ≤ 128)
    (hprod : ∀ k, base.val + width.val * k + width.val ≤ 24 → ∀ p, p < 5 → ∀ l, l < width.val →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max)
    (hw : 0 < width.val) (hwidth32 : width.val ≤ 32) (hbase : base.val ≤ 24)
    (hcap : out.val.length + 1024 < Usize.max)
    (hinv : ptq1_0EmitInv raw b take start out pos true none) :
    ∃ (out1 : alloc.vec.Vec I64) (pos1 base1 stage1 : Usize),
      dequant.lowbit.dequant_ptq1_0_row_loop0_loop0.body scale qs take out true none pos base stage
        = ok (ControlFlow.cont (out1, true, none, pos1, base1, stage1)) ∧
      ptq1_0EmitInv raw b take start out1 pos1 true none ∧
      base1.val ≤ 24 ∧ base1.val + width.val > 24 ∧
      (∃ k, base1.val = base.val + width.val * k ∧ pos1.val = pos.val + 5 * width.val * k) ∧
      stage1.val = stage.val + 1 := by
  obtain ⟨stage1, hstage1a, hstage1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := stage) (y := 1#usize) (by scalar_tac))
  have hstage1 : stage1.val = stage.val + 1 := by rw [hstage1v]; simp
  obtain ⟨q, hbase_call, hq⟩ := WP.spec_imp_exists
    (ptq1_0Base_loop_spec raw b scale qs take pos base width start out hw hwidth32 hbase hsc hqs hOK
      hpos128 hprod (by omega) hinv)
  obtain ⟨out1, pos1, base1, hq_eq, hb1le, hb1gt, hkq, hemit1⟩ := hq
  have hbase' : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0_loop0 out true none scale qs take pos
      base width = ok (out1, true, none, pos1, base1) := by rw [hbase_call, hq_eq]
  refine ⟨out1, pos1, base1, stage1, ?_, hemit1, hb1le, hb1gt, hkq, hstage1⟩
  unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hstage), if_pos rfl]
  rw [hwidthread]
  simp only [bind_tc_ok]
  rw [hbase']
  simp only [bind_tc_ok]
  rw [hstage1a]
  simp only [bind_tc_ok]
  rfl

/-- **`PTQ1_0` stage loop.** Three stages of `PTQ1_0_STAGES` walk `qs` from `(base, pos) = (0, 0)` to
`(24, 120)`. Capacity pad `1280`. -/
theorem ptq1_0Stage_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 24#usize) (take : Usize) (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 24 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 28 + j) 0#u8).val)
    (hbound : ∀ r, r < 128 → I64.min ≤ ptq1_0Weight raw b r ∧ ptq1_0Weight raw b r ≤ I64.max)
    (hcap : out0.val.length + 1280 < Usize.max)
    (hinv0 : ptq1_0EmitInv raw b take start out0 0#usize true none) :
    WP.spec (dequant.lowbit.dequant_ptq1_0_row_loop0_loop0 out0 true none scale qs take 0#usize
        0#usize 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none, pos') ∧ pos'.val = 120 ∧
        ptq1_0EmitInv raw b take start out pos' true none) := by
  have hprod1 : ∀ p, p < 5 → ∀ l, l < 16 →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (0 + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (0 + l) 0#u8) p - 1) ≤ I64.max :=
    ptq1_0QsOK_prod raw b scale qs 0 0 16 hsc hqs (ptq1_0QsOK_zero16 raw b) hbound
      (by intro p hp l hl; omega) (by norm_num)
  have hprod2 : ∀ p, p < 5 → ∀ l, l < 8 →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (16 + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (16 + l) 0#u8) p - 1) ≤ I64.max :=
    ptq1_0QsOK_prod raw b scale qs 80 16 8 hsc hqs (ptq1_0QsOK_lane8 raw b) hbound
      (by intro p hp l hl; omega) (by norm_num)
  unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize × Usize =>
      3 - s.2.2.2.2.2.val)
    (inv := fun s => ptq1_0EmitInv raw b take start s.1 s.2.2.2.1 s.2.1 s.2.2.1 ∧
      ptq1_0StagePhase s.2.2.2.1.val s.2.2.2.2.1.val s.2.2.2.2.2.val)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none, pos') ∧ pos'.val = 120 ∧
      ptq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, base, stage⟩ ⟨hemit, hphase⟩
    dsimp only at hemit hphase ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    have hcapOut : out.val.length + 1024 < Usize.max := by
      have h1 := ptq1_0EmitInv_length (⟨hok, herr, hrec⟩ :
        ptq1_0EmitInv raw b take start out pos ok1 err)
      have h2 := ptq1_0EmitInv_start_le hinv0
      omega
    by_cases hlt : stage.val < 3
    · have hidx : stage.val < dequant.lowbit.PTQ1_0_STAGES.length := by
        rw [PTQ1_0_STAGES_len]; omega
      obtain ⟨width, hwidthread, hwidthv⟩ := WP.spec_imp_exists
        (Array.index_usize_spec dequant.lowbit.PTQ1_0_STAGES stage hidx)
      have hwvg : width.val = (dequant.lowbit.PTQ1_0_STAGES.val.getD stage.val 0#usize).val := by
        rw [hwidthv]
        exact congrArg (fun u : Usize => u.val)
          (list_getD_getElem (dequant.lowbit.PTQ1_0_STAGES.val) stage.val hidx 0#usize)
      rcases hphase with ⟨hs, hb, hp⟩ | ⟨hs, hb, hp⟩ | ⟨hs, hb, hp⟩ | ⟨hs, hb, hp⟩
      · -- stage 0: width 32, base 0, pos 0; null over the 24-byte `qs`
        have hwv : width.val = 32 := by
          rw [hwvg, hs]; rw [PTQ1_0_STAGES_getD 0 (by omega)]; norm_num
        have hOK : ∀ k, base.val + width.val * k + width.val ≤ 24 →
            ptq1_0QsOK raw b (pos.val + 5 * width.val * k) (base.val + width.val * k) width.val := by
          intro k hk; omega
        have hpos128 : ∀ k, base.val + width.val * k + width.val ≤ 24 →
            pos.val + 5 * width.val * k + 5 * width.val ≤ 128 := by
          intro k hk; omega
        have hprod : ∀ k, base.val + width.val * k + width.val ≤ 24 → ∀ p, p < 5 →
            ∀ l, l < width.val →
            I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ∧
            scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max := by
          intro k hk p hp' l hl; omega
        obtain ⟨out1, pos1, base1, stage1, hbody, hemit1, hb1le, hb1gt, hkq, hstage1⟩ :=
          ptq1_0Stage_step raw b scale qs take stage start out pos base width hlt hwidthread hsc hqs
            hOK hpos128 hprod (by omega) (by omega) (by omega)
            hcapOut ⟨rfl, rfl, hrec⟩
        obtain ⟨k, hbk, hpk⟩ := hkq
        rw [hwv] at hbk hpk hb1gt
        have hk0 : k = 0 := by omega
        have hb1 : base1.val = 0 := by omega
        have hp1 : pos1.val = 0 := by omega
        rw [hbody]
        simp only [WP.spec_ok]
        refine ⟨⟨hemit1, ?_⟩, ?_⟩
        · right; left
          exact ⟨by omega, hb1, hp1⟩
        · show 3 - stage1.val < 3 - stage.val
          omega
      · -- stage 1: width 16, base 0, pos 0; emits 80
        have hwv : width.val = 16 := by
          rw [hwvg, hs]; rw [PTQ1_0_STAGES_getD 1 (by omega)]; norm_num
        have hOK : ∀ k, base.val + width.val * k + width.val ≤ 24 →
            ptq1_0QsOK raw b (pos.val + 5 * width.val * k) (base.val + width.val * k) width.val := by
          intro k hk
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0
          simpa [hb, hp, hwv] using ptq1_0QsOK_zero16 raw b
        have hpos128 : ∀ k, base.val + width.val * k + width.val ≤ 24 →
            pos.val + 5 * width.val * k + 5 * width.val ≤ 128 := by
          intro k hk
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0; rw [hp, hwv]; norm_num
        have hprod : ∀ k, base.val + width.val * k + width.val ≤ 24 → ∀ p, p < 5 →
            ∀ l, l < width.val →
            I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ∧
            scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max := by
          intro k hk p hp' l hl
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0
          have h := hprod1 p hp' l (by rw [hwv] at hl; omega)
          simpa [hb, hwv] using h
        obtain ⟨out1, pos1, base1, stage1, hbody, hemit1, hb1le, hb1gt, hkq, hstage1⟩ :=
          ptq1_0Stage_step raw b scale qs take stage start out pos base width hlt hwidthread hsc hqs
            hOK hpos128 hprod (by omega) (by omega) (by omega)
            hcapOut ⟨rfl, rfl, hrec⟩
        obtain ⟨k, hbk, hpk⟩ := hkq
        rw [hwv] at hbk hpk hb1gt
        have hk1 : k = 1 := by omega
        have hb1 : base1.val = 16 := by omega
        have hp1 : pos1.val = 80 := by omega
        rw [hbody]
        simp only [WP.spec_ok]
        refine ⟨⟨hemit1, ?_⟩, ?_⟩
        · right; right; left
          exact ⟨by omega, hb1, hp1⟩
        · show 3 - stage1.val < 3 - stage.val
          omega
      · -- stage 2: width 8, base 16, pos 80; emits 40
        have hwv : width.val = 8 := by
          rw [hwvg, hs]; rw [PTQ1_0_STAGES_getD 2 (by omega)]; norm_num
        have hOK : ∀ k, base.val + width.val * k + width.val ≤ 24 →
            ptq1_0QsOK raw b (pos.val + 5 * width.val * k) (base.val + width.val * k) width.val := by
          intro k hk
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0
          simpa [hb, hp, hwv] using ptq1_0QsOK_lane8 raw b
        have hpos128 : ∀ k, base.val + width.val * k + width.val ≤ 24 →
            pos.val + 5 * width.val * k + 5 * width.val ≤ 128 := by
          intro k hk
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0; rw [hp, hwv]; norm_num
        have hprod : ∀ k, base.val + width.val * k + width.val ≤ 24 → ∀ p, p < 5 →
            ∀ l, l < width.val →
            I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ∧
            scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max := by
          intro k hk p hp' l hl
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0
          have h := hprod2 p hp' l (by rw [hwv] at hl; omega)
          simpa [hb, hwv] using h
        obtain ⟨out1, pos1, base1, stage1, hbody, hemit1, hb1le, hb1gt, hkq, hstage1⟩ :=
          ptq1_0Stage_step raw b scale qs take stage start out pos base width hlt hwidthread hsc hqs
            hOK hpos128 hprod (by omega) (by omega) (by omega)
            hcapOut ⟨rfl, rfl, hrec⟩
        obtain ⟨k, hbk, hpk⟩ := hkq
        rw [hwv] at hbk hpk hb1gt
        have hk1 : k = 1 := by omega
        have hb1 : base1.val = 24 := by omega
        have hp1 : pos1.val = 120 := by omega
        rw [hbody]
        simp only [WP.spec_ok]
        refine ⟨⟨hemit1, ?_⟩, ?_⟩
        · right; right; right
          exact ⟨by omega, hb1, hp1⟩
        · show 3 - stage1.val < 3 - stage.val
          omega
      · omega
    · have hnot : ¬ (stage < 3#usize) := fun hc => hlt ((UScalar.lt_equiv stage 3#usize).1 hc)
      have hstage3 : stage.val = 3 := by
        rcases hphase with ⟨hs, _, _⟩ | ⟨hs, _, _⟩ | ⟨hs, _, _⟩ | ⟨hs, _, _⟩ <;> omega
      have hpos : pos.val = 120 := by
        rcases hphase with ⟨_, _, hp⟩ | ⟨_, _, hp⟩ | ⟨_, _, hp⟩ | ⟨_, _, hp⟩ <;> omega
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0.body scale qs take out true none
          pos base stage = ok (ControlFlow.done (out, true, none, pos)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨out, pos, rfl, hpos, rfl, rfl, hrec⟩
  · exact ⟨hinv0, Or.inl ⟨rfl, rfl, rfl⟩⟩

/-! ## `PTQ1_0` `qh` tail -/

/-- Tail-loop invariant: `emitInv` and `pos = pos0 + tail`, `tail ≤ 4`. -/
def ptq1_0TailInv (raw : Slice U8) (b : Nat) (take : Usize) (pos0 : Nat)
    (start out : alloc.vec.Vec I64) (pos tail : Usize) (ok1 : Bool)
    (err : Option error.EngineError) : Prop :=
  ptq1_0EmitInv raw b take start out pos ok1 err ∧ pos.val = pos0 + tail.val ∧ tail.val ≤ 2

/-- One `qh`-tail lane-loop body step: decode `qh[tail]` at plane `plane`, multiply and push. -/
theorem ptq1_0Tail_body_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qh : Array U8 2#usize) (take : Usize) (plane pos0 : Usize)
    (start out : alloc.vec.Vec I64) (pos tail : Usize)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqh : ∀ j, j < 2 → (qh.val.getD j 0#u8).val = (raw.val.getD (b * 28 + 24 + j) 0#u8).val)
    (hplane4 : plane.val < 4)
    (hbyte : ptq1_0Byte pos.val = 24 + tail.val)
    (hplane : ptq1_0Plane pos.val = plane.val)
    (htail : tail.val < 2)
    (hpos128 : pos.val < 128)
    (hmul : I64.min ≤ scale.val * (tritRef (qh.val.getD tail.val 0#u8) plane.val - 1) ∧
      scale.val * (tritRef (qh.val.getD tail.val 0#u8) plane.val - 1) ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : ptq1_0TailInv raw b take pos0.val start out pos tail true none) :
    ∃ (out2 : alloc.vec.Vec I64) (pos2 tail2 : Usize),
      dequant.lowbit.dequant_ptq1_0_row_loop0_loop1_loop0.body scale qh take plane out true none pos tail
        = ok (ControlFlow.cont (out2, true, none, pos2, tail2)) ∧
      ptq1_0TailInv raw b take pos0.val start out2 pos2 tail2 true none ∧
      pos2.val = pos.val + 1 ∧ tail2.val = tail.val + 1 := by
  obtain ⟨hemit, hposrel, htaille⟩ := hinv
  obtain ⟨hok, herr, hrec⟩ := hemit
  obtain ⟨i1, hi1, hi1eq⟩ := WP.spec_imp_exists
    (Array.index_usize_spec qh tail (by show tail.val < qh.val.length; rw [Array.property qh]; exact htail))
  have hi1val : i1.val = (qh.val.getD tail.val 0#u8).val := by
    rw [hi1eq]
    exact arr_getD_val_eq qh tail (by rw [Array.property qh]; exact htail)
  obtain ⟨i2, htrit, hi2v⟩ := trit_spec i1 plane (by omega)
  have hi2val : i2.val = tritRef i1 plane.val := hi2v
  obtain ⟨i3, hsub, hi3v⟩ := WP.spec_imp_exists
    (I64.sub_spec (x := i2) (y := 1#i64)
      (by rw [hi2val, I64.min_eq, show (1#i64).val = 1 from rfl]
          have := tritRef_nonneg i1 plane.val; omega)
      (by rw [hi2val, I64.max_eq, show (1#i64).val = 1 from rfl]
          have := tritRef_le_two i1 plane.val; omega))
  have hi3val : i3.val = tritRef (qh.val.getD tail.val 0#u8) plane.val - 1 := by
    rw [hi3v, hi2val, tritRef_congr hi1val plane.val]; norm_num
  obtain ⟨v, hmulv, hvv⟩ := q6k_mul_ok scale i3 (by rw [hi3val]; exact hmul.1)
    (by rw [hi3val]; exact hmul.2)
  have hcong : tritRef (qh.val.getD tail.val 0#u8) plane.val =
      tritRef (raw.val.getD (b * 28 + ptq1_0Byte pos.val) 0#u8) (ptq1_0Plane pos.val) := by
    rw [hplane]
    apply tritRef_congr
    rw [hqh tail.val htail, Nat.add_assoc, ← hbyte]
  have hvval : v.val = (ptq1_0BlockVals raw b)[pos.val]'(by
      rw [ptq1_0BlockVals_length]; exact hpos128) := by
    rw [hvv, hi3val, hsc, hcong]
    rw [list_getD_getElem (ptq1_0BlockVals raw b) pos.val
      (by rw [ptq1_0BlockVals_length]; exact hpos128) (0 : ℤ), ptq1_0BlockVals_getD raw b pos.val hpos128]
    unfold ptq1_0Weight
    rfl
  by_cases hlt : pos.val < take.val
  · obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out v hcap)
    obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨tail2, haddl, htail2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := tail) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have htail2 : tail2.val = tail.val + 1 := by rw [htail2v]; simp
    refine ⟨out2, pos2, tail2, ?_, ?_, hpos2, htail2⟩
    · unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop1_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact htail), if_pos rfl]
      simp only [hi1, htrit, hsub, hmulv, bind_tc_ok]
      rw [if_pos ((UScalar.lt_equiv pos take).2 hlt)]
      simp only [hpush, haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, by rw [hpos2, hposrel, htail2]; omega, by rw [htail2]; omega⟩
      have hmin1 : (pos.val + 1).min take.val = pos.val + 1 := Nat.min_eq_left (by omega)
      have hmin2 : pos.val.min take.val = pos.val := Nat.min_eq_left (le_of_lt hlt)
      rw [hpushv, dequantVals_append, hrec]
      simp only [dequantVals, List.map_cons, List.map_nil, hvval]
      rw [hpos2, hmin1, hmin2,
        List.take_succ_eq_append_getElem (by rw [ptq1_0BlockVals_length]; exact hpos128),
        List.append_assoc]
  · obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨tail2, haddl, htail2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := tail) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have htail2 : tail2.val = tail.val + 1 := by rw [htail2v]; simp
    refine ⟨out, pos2, tail2, ?_, ?_, hpos2, htail2⟩
    · unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop1_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact htail), if_pos rfl]
      simp only [hi1, htrit, hsub, hmulv, bind_tc_ok]
      rw [if_neg (fun hc => hlt ((UScalar.lt_equiv pos take).1 hc))]
      simp only [haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, by rw [hpos2, hposrel, htail2]; omega, by rw [htail2]; omega⟩
      have h1 : (pos.val + 1).min take.val = take.val := Nat.min_eq_right (by omega)
      have h2 : pos.val.min take.val = take.val := Nat.min_eq_right (by omega)
      rw [hrec, hpos2, h1, h2]

/-- **`PTQ1_0` `qh` tail loop.** From `tail = 0`, the two `qh` lanes emit `2` staged values. -/
theorem ptq1_0Tail_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qh : Array U8 2#usize) (take : Usize) (plane pos0 : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqh : ∀ j, j < 2 → (qh.val.getD j 0#u8).val = (raw.val.getD (b * 28 + 24 + j) 0#u8).val)
    (hplane4 : plane.val < 4)
    (hrun : ∀ lane : Usize, lane.val < 2 →
      ptq1_0Byte (pos0.val + lane.val) = 24 + lane.val ∧
      ptq1_0Plane (pos0.val + lane.val) = plane.val)
    (hpos128 : pos0.val + 2 ≤ 128)
    (hmul : ∀ lane : Usize, lane.val < 2 →
      I64.min ≤ scale.val * (tritRef (qh.val.getD lane.val 0#u8) plane.val - 1) ∧
      scale.val * (tritRef (qh.val.getD lane.val 0#u8) plane.val - 1) ≤ I64.max)
    (hcap : out0.val.length + 128 < Usize.max)
    (hinv0 : ptq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_ptq1_0_row_loop0_loop1_loop0 out0 true none scale qh take pos0
        plane 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none, pos') ∧ pos'.val = pos0.val + 2 ∧
        ptq1_0EmitInv raw b take start out pos' true none) := by
  have hinit : ptq1_0TailInv raw b take pos0.val start out0 pos0 0#usize true none :=
    ⟨hinv0, by simp, by simp⟩
  unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop1_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      2 - s.2.2.2.2.val)
    (inv := fun s => ptq1_0TailInv raw b take pos0.val start s.1 s.2.2.2.1 s.2.2.2.2 s.2.1 s.2.2.1)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none, pos') ∧ pos'.val = pos0.val + 2 ∧
      ptq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, tail⟩ ⟨hemit, hposrel, htaille⟩
    dsimp only at hemit hposrel htaille ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : tail.val < 2
    · have hpos128' : pos.val < 128 := by omega
      have hbyte : ptq1_0Byte pos.val = 24 + tail.val := by rw [hposrel]; exact (hrun tail hlt).1
      have hplane : ptq1_0Plane pos.val = plane.val := by rw [hposrel]; exact (hrun tail hlt).2
      have hcap' : out.val.length < Usize.max := by
        have h1 := ptq1_0EmitInv_length (⟨hok, herr, hrec⟩ :
          ptq1_0EmitInv raw b take start out pos ok1 err)
        have h2 := ptq1_0EmitInv_start_le hinv0
        omega
      obtain ⟨out2, pos2, tail2, hbody, hinv2, hpos2, htail2⟩ :=
        ptq1_0Tail_body_step raw b scale qh take plane pos0 start out pos tail hsc hqh hplane4 hbyte
          hplane hlt hpos128' (hmul tail hlt) hcap' ⟨⟨rfl, rfl, hrec⟩, hposrel, htaille⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨?_, ?_⟩
      · exact hinv2
      · show 2 - tail2.val < 2 - tail.val
        omega
    · have hnot : ¬ (tail < 2#usize) := fun hc => hlt ((UScalar.lt_equiv tail 2#usize).1 hc)
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop1_loop0.body scale qh take plane out
          true none pos tail = ok (ControlFlow.done (out, true, none, pos)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop1_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have htail4 : tail.val = 2 := by omega
      refine ⟨out, pos, rfl, by rw [hposrel, htail4], rfl, rfl, hrec⟩
  · exact hinit

/-- **`PTQ1_0` `qh` tail plane loop.** The four tail planes emit `8` staged values. Capacity pad
`512`. -/
theorem ptq1_0TailPlane_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qh : Array U8 2#usize) (take : Usize) (pos0 : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = ptq1_0ScaleRef raw b)
    (hqh : ∀ j, j < 2 → (qh.val.getD j 0#u8).val = (raw.val.getD (b * 28 + 24 + j) 0#u8).val)
    (hOK : ∀ p, p < 4 → ∀ l, l < 2 →
      ptq1_0Byte (pos0.val + p * 2 + l) = 24 + l ∧ ptq1_0Plane (pos0.val + p * 2 + l) = p)
    (hpos128 : pos0.val + 8 ≤ 128)
    (hprod : ∀ p, p < 4 → ∀ l, l < 2 →
      I64.min ≤ scale.val * (tritRef (qh.val.getD l 0#u8) p - 1) ∧
      scale.val * (tritRef (qh.val.getD l 0#u8) p - 1) ≤ I64.max)
    (hcap : out0.val.length + 512 < Usize.max)
    (hinv0 : ptq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_ptq1_0_row_loop0_loop1 out0 true none scale qh take pos0 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none) ∧ pos'.val = pos0.val + 8 ∧
        ptq1_0EmitInv raw b take start out pos' true none) := by
  unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop1
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      4 - s.2.2.2.2.val)
    (inv := fun s => ptq1_0EmitInv raw b take start s.1 s.2.2.2.1 s.2.1 s.2.2.1 ∧
      s.2.2.2.1.val = pos0.val + s.2.2.2.2.val * 2 ∧ s.2.2.2.2.val ≤ 4)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none) ∧ pos'.val = pos0.val + 8 ∧
      ptq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, plane⟩ ⟨hemit, hposrel, hplanele⟩
    dsimp only at hemit hposrel hplanele ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : plane.val < 4
    · have hpos128' : pos.val + 2 ≤ 128 := by
        have he : pos.val + 2 = pos0.val + (plane.val + 1) * 2 := by rw [hposrel]; ring
        rw [he]
        have hpl : plane.val + 1 ≤ 4 := by omega
        calc pos0.val + (plane.val + 1) * 2 ≤ pos0.val + 4 * 2 := by
              exact Nat.add_le_add_left (Nat.mul_le_mul_right _ hpl) _
          _ ≤ 128 := hpos128
      have hrun : ∀ lane : Usize, lane.val < 2 →
          ptq1_0Byte (pos.val + lane.val) = 24 + lane.val ∧
          ptq1_0Plane (pos.val + lane.val) = plane.val := by
        intro lane hlane
        have h := hOK plane.val hlt lane.val hlane
        have hidx : pos.val + lane.val = pos0.val + plane.val * 2 + lane.val := by rw [hposrel]
        rw [hidx]
        exact h
      have hmul' : ∀ lane : Usize, lane.val < 2 →
          I64.min ≤ scale.val * (tritRef (qh.val.getD lane.val 0#u8) plane.val - 1) ∧
          scale.val * (tritRef (qh.val.getD lane.val 0#u8) plane.val - 1) ≤ I64.max :=
        fun lane hlane => hprod plane.val hlt lane.val hlane
      have hlen : out.val.length ≤ out0.val.length + 128 := by
        have h1 := ptq1_0EmitInv_length (⟨hok, herr, hrec⟩ :
          ptq1_0EmitInv raw b take start out pos ok1 err)
        have h2 := ptq1_0EmitInv_start_le hinv0
        omega
      obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
        (ptq1_0Tail_loop_spec raw b scale qh take plane pos start out hsc hqh hlt hrun hpos128'
          hmul' (by omega) ⟨rfl, rfl, hrec⟩)
      obtain ⟨out2, pos2, hp_eq, hpos2, hemit2⟩ := hp
      have hloop' : dequant.lowbit.dequant_ptq1_0_row_loop0_loop1_loop0 out true none scale qh take
          pos plane 0#usize = ok (out2, true, none, pos2) := by rw [hloop, hp_eq]
      obtain ⟨plane1, haddp, hplane1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := plane) (y := 1#usize) (by scalar_tac))
      have hplane1 : plane1.val = plane.val + 1 := by rw [hplane1v]; simp
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop1.body scale qh take out true none pos
          plane = ok (ControlFlow.cont (out2, true, none, pos2, plane1)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop1.body
        rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), if_pos rfl]
        simp only [hloop', haddp, bind_tc_ok]
        rfl
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨hemit2, by rw [hpos2, hposrel, hplane1]; ring, by omega⟩, ?_⟩
      show 4 - plane1.val < 4 - plane.val
      omega
    · have hnot : ¬ (plane < 4#usize) := fun hc => hlt ((UScalar.lt_equiv plane 4#usize).1 hc)
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0_loop1.body scale qh take out true none pos
          plane = ok (ControlFlow.done (out, true, none)) := by
        unfold dequant.lowbit.dequant_ptq1_0_row_loop0_loop1.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hplane4 : plane.val = 4 := by omega
      refine ⟨out, pos, rfl, by rw [hposrel, hplane4], rfl, rfl, hrec⟩
  · exact ⟨hinv0, by simp, by simp⟩

/-! ## `PTQ1_0` row composition -/

/-- The `PTQ1_0` block success domain: the 28 bytes are present, `d` is finite, and every one of the
128 staged weights is inside `i64`. -/
def ptq1_0BlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  b.val * 28 + 28 ≤ raw.val.length ∧
  f16Exp (f16LeHalf raw (b.val * 28 + 26)) ≠ 31 ∧
  (∀ r, r < 128 → I64.min ≤ ptq1_0Weight raw b.val r ∧ ptq1_0Weight raw b.val r ≤ I64.max)

/-- The first `take ≤ 128` entries of a staged block stream. -/
lemma ptq1_0BlockVals_take (raw : Slice U8) (b take : Nat) (htake : take ≤ 128) :
    (ptq1_0BlockVals raw b).take take = (List.range take).map (fun j => ptq1_0Weight raw b j) := by
  unfold ptq1_0BlockVals
  rw [← List.map_take, List.take_range, Nat.min_eq_left htake]

/-- Appending the next `take` row weights advances the row prefix by `take`. -/
lemma ptq1_0RowValues_append (raw : Slice U8) (m take : Nat) :
    ptq1_0RowValues raw m ++
      (List.range take).map (fun j => ptq1_0Weight raw ((m + j) / 128) ((m + j) % 128)) =
    ptq1_0RowValues raw (m + take) := by
  unfold ptq1_0RowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

lemma div128_mul_add (g l : Nat) (hl : l < 128) : (128 * g + l) / 128 = g := by omega
lemma mod128_mul_add (g l : Nat) (hl : l < 128) : (128 * g + l) % 128 = l := by omega

/-- The extracted `div_ceil n PTQ1_0_QK` succeeds with `(n + 127)/128`. -/
theorem dequant_div_ceil_ptq1_0_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.PTQ1_0_QK = ok blocks ∧
      blocks.val = (n.val + 127) / 128 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.PTQ1_0_QK) (by rw [quant_PTQ1_0_QK_eq]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.PTQ1_0_QK) (by rw [quant_PTQ1_0_QK_eq]; decide))
  have hqv' : q.val = n.val / 128 := by rw [hqv, quant_PTQ1_0_QK_eq]; simp
  have hrv' : r.val = n.val % 128 := by rw [hrv, quant_PTQ1_0_QK_eq]; simp
  have hq' : UScalar.div n quant.PTQ1_0_QK = ok q := hq
  have hr' : UScalar.rem n quant.PTQ1_0_QK = ok r := hr
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
      (UScalar.add_spec (x := q) (y := (UScalar.one : Usize)) (by rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]
      exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 128 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (128·block) n` raw weights. -/
def ptq1_0RowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ ptq1_0RowValues raw (Nat.min (128 * block.val) n.val)

/-- One `PTQ1_0` block-loop body step: decode the block, run the `qs` stage loop and the `qh` tail
plane loop, append `take = min 128 (n - 128·block)` values and advance. -/
theorem ptq1_0Block_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 127) / 128)
    (hdom : ptq1_0BlockDomain raw block)
    (hcap : out.val.length + 1280 < Usize.max)
    (hinv : ptq1_0RowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.lowbit.dequant_ptq1_0_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      ptq1_0RowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨hlen, hfin, hbound⟩ := hdom
  obtain ⟨d, qs, qh, hq, hdv, hqsv, hqhv⟩ := ptq1_0_block_ok raw block hlen hfin
  have hltmul : 128 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_128 (by rw [← hblocks]; exact hblock)
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.PTQ1_0_QK) (by rw [quant_PTQ1_0_QK_eq]; scalar_tac))
  have hivv' : i.val = 128 * block.val := by rw [hivv, quant_PTQ1_0_QK_eq]; scalar_tac
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 128 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.PTQ1_0_QK i1
  have htakev' : take.val = Nat.min 128 (n.val - 128 * block.val) := by
    rw [htakev, hi1v', quant_PTQ1_0_QK_eq]; simp
  have htake_le : take.val ≤ 128 := by rw [htakev']; exact min_le_left _ _
  have hinv0 : ptq1_0EmitInv raw block.val take out out 0#usize true none := ⟨rfl, rfl, by simp⟩
  obtain ⟨p1, hstage_call, hp1⟩ := WP.spec_imp_exists
    (ptq1_0Stage_loop_spec raw block.val d qs take out out hdv hqsv hbound (by omega) hinv0)
  obtain ⟨out1, pos1, hp1_eq, hpos1, hemit1⟩ := hp1
  have hstage' : dequant.lowbit.dequant_ptq1_0_row_loop0_loop0 out true none d qs take 0#usize
      0#usize 0#usize = ok (out1, true, none, pos1) := by rw [hstage_call, hp1_eq]
  have hOKtail : ∀ p, p < 4 → ∀ l, l < 2 →
      ptq1_0Byte (pos1.val + p * 2 + l) = 24 + l ∧ ptq1_0Plane (pos1.val + p * 2 + l) = p := by
    intro p hp l hl
    rw [hpos1]
    exact ⟨ptq1_0Byte_tail p l hp hl, ptq1_0Plane_tail p l hp hl⟩
  have hprodtail : ∀ p, p < 4 → ∀ l, l < 2 →
      I64.min ≤ d.val * (tritRef (qh.val.getD l 0#u8) p - 1) ∧
      d.val * (tritRef (qh.val.getD l 0#u8) p - 1) ≤ I64.max := by
    intro p hp l hl
    have hr : 120 + p * 2 + l < 128 := by omega
    have hb := hbound (120 + p * 2 + l) hr
    have hbyte : ptq1_0Byte (120 + p * 2 + l) = 24 + l := ptq1_0Byte_tail p l hp hl
    have hplane : ptq1_0Plane (120 + p * 2 + l) = p := ptq1_0Plane_tail p l hp hl
    have hbyteEq : (qh.val.getD l 0#u8).val =
        (raw.val.getD (block.val * 28 + ptq1_0Byte (120 + p * 2 + l)) 0#u8).val := by
      rw [hbyte]
      rw [show block.val * 28 + (24 + l) = block.val * 28 + 24 + l by omega]
      exact hqhv l hl
    have htrit : tritRef (qh.val.getD l 0#u8) p =
        tritRef (raw.val.getD (block.val * 28 + ptq1_0Byte (120 + p * 2 + l)) 0#u8)
          (ptq1_0Plane (120 + p * 2 + l)) := by
      have h1 := tritRef_congr hbyteEq p
      rw [hplane]
      exact h1
    have hW : ptq1_0Weight raw block.val (120 + p * 2 + l) =
        ptq1_0ScaleRef raw block.val * (tritRef (qh.val.getD l 0#u8) p - 1) := by
      unfold ptq1_0Weight
      rw [htrit]
    have hb' := hb
    rw [hW, ← hdv] at hb'
    exact hb'
  have hlen1 : out1.val.length ≤ out.val.length + 128 :=
    ptq1_0EmitInv_length hemit1
  obtain ⟨p2, htail_call, hp2⟩ := WP.spec_imp_exists
    (ptq1_0TailPlane_loop_spec raw block.val d qh take pos1 out out1 hdv hqhv hOKtail
      (by rw [hpos1]) hprodtail (by omega) hemit1)
  obtain ⟨out2, pos2, hp2_eq, hpos2, hemit2⟩ := hp2
  have htail' : dequant.lowbit.dequant_ptq1_0_row_loop0_loop1 out1 true none d qh take pos1 0#usize
      = ok (out2, true, none) := by rw [htail_call, hp2_eq]
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1 : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have hmin : Nat.min (128 * block.val) n.val = 128 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 128 * block.val + take.val = Nat.min (128 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min_128 block.val n.val (le_of_lt hltmul)
  have hinv_new : dequantVals out2.val =
      dequantVals out0.val ++ ptq1_0RowValues raw (Nat.min (128 * block1.val) n.val) := by
    obtain ⟨-, -, hrec2⟩ := hemit2
    have hminpos : Nat.min pos2.val take.val = take.val := by
      rw [hpos2, hpos1]; exact Nat.min_eq_right htake_le
    rw [hrec2, hminpos, hinv.2.2.2, hmin, List.append_assoc,
      ptq1_0BlockVals_take raw block.val take.val htake_le]
    have hmap : (List.range take.val).map (fun j => ptq1_0Weight raw block.val j) =
        (List.range take.val).map
          (fun j => ptq1_0Weight raw ((128 * block.val + j) / 128) ((128 * block.val + j) % 128)) := by
      apply List.map_congr_left
      intro j hj
      have hj128 : j < 128 := by have := List.mem_range.mp hj; omega
      rw [div128_mul_add block.val j hj128, mod128_mul_add block.val j hj128]
    rw [hmap, ptq1_0RowValues_append, hsum, hblock1]
  refine ⟨out2, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1⟩
  rw [dequant.lowbit.dequant_ptq1_0_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq]
  simp only [bind_tc_ok]
  change (do
    let i ← block * quant.PTQ1_0_QK
    let i1 ← n - i
    let take ← core.cmp.Ord.min.trait_default core.cmp.OrdUsize quant.PTQ1_0_QK i1
    let (out1, ok2, err1, pos) ←
      dequant.lowbit.dequant_ptq1_0_row_loop0_loop0 out true none d qs take 0#usize 0#usize 0#usize
    let (out2, ok3, err2) ← dequant.lowbit.dequant_ptq1_0_row_loop0_loop1 out1 ok2 err1 d qh take pos 0#usize
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out2, block1, ok3, err2)))
    = ok (ControlFlow.cont (out2, block1, true, none))
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  rw [hstage']
  simp only [bind_tc_ok]
  show (do
    let (out2, ok3, err2) ←
      dequant.lowbit.dequant_ptq1_0_row_loop0_loop1 out1 true none d qh take pos1 0#usize
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out2, block1, ok3, err2)))
    = ok (ControlFlow.cont (out2, block1, true, none))
  rw [htail']
  simp only [bind_tc_ok]
  rw [hblock1a]
  simp only [bind_tc_ok]
  rfl

/-- The `PTQ1_0` block loop: `loop.spec_decr_nat` with `ptq1_0RowInv` and measure `blocks - block`. -/
theorem ptq1_0Row_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 127) / 128)
    (hdom : ∀ b : Usize, b.val < blocks.val → ptq1_0BlockDomain raw b)
    (hcap : out0.val.length + n.val + 1280 < Usize.max) :
    WP.spec (dequant.lowbit.dequant_ptq1_0_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ ptq1_0RowValues raw n.val) := by
  unfold dequant.lowbit.dequant_ptq1_0_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => ptq1_0RowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
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
        simpa [dequantVals, ptq1_0RowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + 1280 < Usize.max := by rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        ptq1_0Block_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.lowbit.dequant_ptq1_0_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.lowbit.dequant_ptq1_0_row_loop0.body, if_neg hnot]
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
    simp [ptq1_0RowValues]

/-- **`PTQ1_0` row law.** On the block success domain, the extracted `dequant_ptq1_0_row` succeeds
with `Ok ()` and appends exactly the first `n` raw weights `ptq1_0RowValues raw n`. -/
theorem dequant_ptq1_0_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 127) / 128 → ptq1_0BlockDomain raw b)
    (hcap : out.val.length + n.val + 1280 < Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_ptq1_0_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ ptq1_0RowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_ptq1_0_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (ptq1_0Row_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.lowbit.dequant_ptq1_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`PTQ1_0` dispatch row law.** `dequant_row_fixed` at `ggml_type = 143` (`GGML_TYPE_PTQ1_0`)
succeeds with the exact row weights. -/
theorem dequant_row_fixed_ptq1_0_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 127) / 128 → ptq1_0BlockDomain raw b)
    (hcap : out.val.length + n.val + 1280 < Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_row_fixed 143#u32 raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ ptq1_0RowValues raw n.val := by
  rw [dequant_row_fixed_ptq1_0]
  exact dequant_ptq1_0_row_spec raw n out hdom hcap

/-! ## `PTQ1_0` one-block witness -/

/-- The 28-byte `PTQ1_0` witness block: `d = 1.0` (`0x3C00` at bytes 26–27), all code bytes `0`. -/
def ptq1_0WitBytes : List U8 := List.replicate 26 0#u8 ++ [0#u8, 60#u8]

theorem ptq1_0WitBytes_length : ptq1_0WitBytes.length = 28 := by
  simp [ptq1_0WitBytes]

def ptq1_0WitRaw : Slice U8 := Slice.from ptq1_0WitBytes (by rw [ptq1_0WitBytes_length]; scalar_tac)

theorem ptq1_0WitRaw_val : ptq1_0WitRaw.val = ptq1_0WitBytes := Slice.from_val _ _

/-- `getD` of the witness bytes before byte 26 is `0`. -/
lemma ptq1_0Wit_getD_zero (k : Nat) (hk : k < 26) :
    (ptq1_0WitBytes.getD k 0#u8).val = 0 := by
  unfold ptq1_0WitBytes
  rw [List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_left (by rw [List.length_replicate]; exact hk)]
  rw [List.getElem?_replicate, if_pos hk]
  rfl

/-- The witness byte list's option at `26` is `0`. -/
lemma ptq1_0Wit_getElem?_26 : ptq1_0WitBytes[26]? = some 0#u8 := by
  unfold ptq1_0WitBytes
  rw [List.getElem?_append_right (by rw [List.length_replicate])]
  rw [List.length_replicate, show 26 - 26 = 0 by omega]
  rfl

/-- The witness byte list's option at `27` is `60`. -/
lemma ptq1_0Wit_getElem?_27 : ptq1_0WitBytes[27]? = some 60#u8 := by
  unfold ptq1_0WitBytes
  rw [List.getElem?_append_right (by rw [List.length_replicate]; omega)]
  rw [List.length_replicate, show 27 - 26 = 1 by omega]
  rfl

/-- The witness half-word is the binary16 pattern `0x3C00 = 15360` (`1.0`). -/
lemma ptq1_0Wit_f16LeHalf : f16LeHalf ptq1_0WitRaw 26 =
    core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) := by
  unfold f16LeHalf
  congr 1

/-- The half-word `[0x00, 0x3C]` is `15360`. -/
lemma ptq1_0_u16_le_0_60 :
    (core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8])).val = 15360 := by decide

lemma ptq1_0Wit_f16_val : (f16LeHalf ptq1_0WitRaw 26).val = 15360 := by
  rw [ptq1_0Wit_f16LeHalf, ptq1_0_u16_le_0_60]

/-- The witness scale is `1.0 = 2^32` in Q32.32. -/
lemma ptq1_0Wit_scale : ptq1_0ScaleRef ptq1_0WitRaw 0 = 2 ^ 32 := by
  have hv := ptq1_0Wit_f16_val
  unfold ptq1_0ScaleRef
  rw [show 0 * 28 + 26 = 26 by norm_num]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

/-- Every within-block byte index read by the walk is `< 26`, hence `0`. -/
lemma ptq1_0Byte_lt26 (r : Nat) : ptq1_0Byte r < 26 := by
  unfold ptq1_0Byte
  split
  · have := Nat.mod_lt r (by norm_num : 0 < 16); omega
  · split
    · have := Nat.mod_lt (r - 80) (by norm_num : 0 < 8); omega
    · have := Nat.mod_lt (r - 120) (by norm_num : 0 < 2); omega

/-- Every staged weight of the witness block is `-2^32`. -/
lemma ptq1_0Wit_weight (r : Nat) : ptq1_0Weight ptq1_0WitRaw 0 r = -(2 ^ 32) := by
  unfold ptq1_0Weight
  rw [ptq1_0Wit_scale]
  have hbyte : (ptq1_0WitRaw.val.getD (0 * 28 + ptq1_0Byte r) 0#u8).val = 0 := by
    rw [ptq1_0WitRaw_val]
    have hlt : 0 * 28 + ptq1_0Byte r < 26 := by
      have := ptq1_0Byte_lt26 r; omega
    exact ptq1_0Wit_getD_zero _ hlt
  have htrit : tritRef (ptq1_0WitRaw.val.getD (0 * 28 + ptq1_0Byte r) 0#u8) (ptq1_0Plane r) = 0 := by
    have heq : ptq1_0WitRaw.val.getD (0 * 28 + ptq1_0Byte r) 0#u8 = 0#u8 :=
      UScalar.eq_of_val_eq (by simpa using hbyte)
    rw [heq, tritRef_zero]
  rw [htrit]
  norm_num

/-- The witness block satisfies the `PTQ1_0` success domain. -/
theorem ptq1_0BlockDomain_witness : ptq1_0BlockDomain ptq1_0WitRaw 0#usize := by
  have hb : (0#usize).val = 0 := by decide
  refine ⟨?_, ?_, ?_⟩
  · simp [hb, ptq1_0WitRaw_val, ptq1_0WitBytes_length]
  · have hv := ptq1_0Wit_f16_val
    unfold f16Exp
    rw [hb, show 0 * 28 + 26 = 26 by norm_num, hv]
    norm_num
  · intro r hr
    rw [hb, ptq1_0Wit_weight r]
    norm_num [I64.min_eq, I64.max_eq]

/-- The head of the `128`-element row list. -/
lemma ptq1_0RowValues_head (raw : Slice U8) :
    (ptq1_0RowValues raw 128).getD 0 0 = ptq1_0Weight raw 0 0 := by
  unfold ptq1_0RowValues
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 128)[0]? = some 0 := by decide
  rw [h]; rfl

/-- **`PTQ1_0` one-block witness at `n = 128`.** The extracted row succeeds on the empty vector,
appends exactly the `128` weights of the witness block, and element `0` is `-2^32`. -/
theorem ptq1_0Witness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_ptq1_0_row ptq1_0WitRaw 128#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = ptq1_0RowValues ptq1_0WitRaw 128 ∧
      (out1.val.getD 0 0#i64).val = -(2 ^ 32) := by
  have hdom : ∀ b : Usize, b.val < ((128#usize).val + 127) / 128 → ptq1_0BlockDomain ptq1_0WitRaw b := by
    intro b hb
    have hb1 : b.val < 1 := by
      have hR : (((128#usize).val + 127) / 128) = 1 := by decide
      rwa [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact ptq1_0BlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_ptq1_0_row_spec ptq1_0WitRaw 128#usize (alloc.vec.Vec.new I64) hdom (by
      rw [tq1_0_vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = ptq1_0RowValues ptq1_0WitRaw 128 := by
    rw [tq1_0_vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
  rw [tq1_0_dequantVals_getD] at h0
  rw [h0, ptq1_0RowValues_head, ptq1_0Wit_weight]

end

end apeiron_word
