/-
Row laws for the ternary faces `TQ1_0` and `PTQ1_0`.

`DequantTQ1_0.lean` proves the block laws and the innermost `qs` lane loop. This file composes the
plane, base, stage and `qh`-tail loops into the block body, then the `QK`-block loop into the row
law `dequantVals out' = dequantVals out ++ tq1_0RowValues raw n`. The `PTQ1_0` face is the same
composition at block length 128 with stages `[32, 16, 8]`.

Capacity bookkeeping: the already-proved lane loop requires `out0.length + 256 < Usize.max` at each
call. Within one block at most `256` (resp. `128`) values are pushed, so a starting vector of length
`L` is only ever extended to length at most `L + 256`. Each loop therefore carries a static capacity
hypothesis with a pad covering the pushes of the enclosing loops: `512` for the `qs` plane loop,
`768` for the `qs` base loop, `1024` for the stage loop and block body, `n + 1024` for the row.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed literal
facts (small array reads), never on an extracted body.
-/
import Numeric.DequantTQ1_0

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## `TQ1_0` staged `qs` walk -/

/-- `qs` segment predicate: output position `pos0 + p·width + l` reads byte `base + l` at plane `p`,
for every plane `p < 5` and lane `l < width`. -/
def tq1_0QsOK (_raw : Slice U8) (_b : Nat) (pos0 base width : Nat) : Prop :=
  ∀ p, p < 5 → ∀ l, l < width →
    tq1_0Byte (pos0 + p * width + l) = base + l ∧
    tq1_0Plane (pos0 + p * width + l) = p

/-- The `qs` segment `0 … 159` of a `TQ1_0` block: width `32`, base `0`. -/
lemma tq1_0QsOK_zero32 (raw : Slice U8) (b : Nat) : tq1_0QsOK raw b 0 0 32 := by
  intro p hp l hl
  constructor
  · rw [show 0 + p * 32 + l = p * 32 + l by omega]
    simpa using tq1_0Byte_lane32 p l hp hl
  · rw [show 0 + p * 32 + l = p * 32 + l by omega]
    simpa using tq1_0Plane_lane32 p l hp hl

/-- The `qs` segment `160 … 239` of a `TQ1_0` block: width `16`, base `32`. -/
lemma tq1_0QsOK_lane16 (raw : Slice U8) (b : Nat) : tq1_0QsOK raw b 160 32 16 := by
  intro p hp l hl
  exact ⟨tq1_0Byte_lane16 p l hp hl, tq1_0Plane_lane16 p l hp hl⟩

/-- `TQ1_0_STAGES` has two entries. -/
lemma TQ1_0_STAGES_len : dequant.lowbit.TQ1_0_STAGES.length = 2 :=
  Array.length_eq dequant.lowbit.TQ1_0_STAGES

/-- Reading `TQ1_0_STAGES` at `i` is `32` then `16`. -/
lemma TQ1_0_STAGES_val (i : Nat) (h : i < dequant.lowbit.TQ1_0_STAGES.length) :
    (dequant.lowbit.TQ1_0_STAGES.val[i]'h).val = if i = 0 then 32 else 16 := by
  have hd : (dequant.lowbit.TQ1_0_STAGES.val.getD i 0#usize).val =
      (if i = 0 then 32 else 16) := by
    have h2 : i < 2 := by have := TQ1_0_STAGES_len; omega
    have : i = 0 ∨ i = 1 := by omega
    rcases this with h0 | h1
    · subst h0; unfold dequant.lowbit.TQ1_0_STAGES; decide
    · subst h1; unfold dequant.lowbit.TQ1_0_STAGES; decide
  rwa [List.getD_eq_getElem (l := dequant.lowbit.TQ1_0_STAGES.val) (d := 0#usize) h] at hd

/-! ## `qs` plane loop -/

/-- **`qs` plane loop.** From a segment `tq1_0QsOK raw b pos0 base width`, the five planes emit the
staged values `pos0 … pos0 + 5·width - 1`, leaving the emission invariant. Capacity pad `512`. -/
theorem tq1_0Plane_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 48#usize) (take : Usize) (base width pos0 : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 48 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 54 + j) 0#u8).val)
    (hOK : tq1_0QsOK raw b pos0.val base.val width.val)
    (hpos256 : pos0.val + 5 * width.val ≤ 256)
    (hbase : base.val + width.val ≤ 48)
    (hprod : ∀ p, p < 5 → ∀ l, l < width.val →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (base.val + l) 0#u8) p - 1) ≤ I64.max)
    (hcap : out0.val.length + 512 < Usize.max)
    (hinv0 : tq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0 out0 true none scale qs take
        pos0 base width 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none, pos') ∧ pos'.val = pos0.val + 5 * width.val ∧
        tq1_0EmitInv raw b take start out pos' true none) := by
  unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      5 - s.2.2.2.2.val)
    (inv := fun s => tq1_0EmitInv raw b take start s.1 s.2.2.2.1 s.2.1 s.2.2.1 ∧
      s.2.2.2.1.val = pos0.val + s.2.2.2.2.val * width.val ∧ s.2.2.2.2.val ≤ 5)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none, pos') ∧ pos'.val = pos0.val + 5 * width.val ∧
      tq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, plane⟩ ⟨hemit, hposrel, hplanele⟩
    dsimp only at hemit hposrel hplanele ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : plane.val < 5
    · have hpos256' : pos.val + width.val ≤ 256 := by
        have he : pos.val + width.val = pos0.val + (plane.val + 1) * width.val := by
          rw [hposrel]; ring
        rw [he]
        have hpl : plane.val + 1 ≤ 5 := by omega
        calc pos0.val + (plane.val + 1) * width.val
            ≤ pos0.val + 5 * width.val := by
              exact Nat.add_le_add_left (Nat.mul_le_mul_right _ hpl) _
          _ ≤ 256 := hpos256
      have hrun : ∀ lane : Usize, lane.val < width.val →
          tq1_0Byte (pos.val + lane.val) = base.val + lane.val ∧
          tq1_0Plane (pos.val + lane.val) = plane.val := by
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
        have h4 := (List.take_sublist (l := tq1_0BlockVals raw b)
          (i := Nat.min pos0.val take.val)).length_le
        simp only [tq1_0BlockVals_length] at h4
        omega
      have hlen : out.val.length ≤ out0.val.length + 256 := by
        have h := congrArg List.length hrec
        simp only [dequantVals, List.length_append, List.length_map] at h
        have h4 := (List.take_sublist (l := tq1_0BlockVals raw b)
          (i := Nat.min pos.val take.val)).length_le
        simp only [tq1_0BlockVals_length] at h4
        omega
      obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
        (tq1_0Lane_loop_spec raw b scale qs take base width plane pos start out
          hsc hqs hlt hrun hpos256' hbase hmul' (by omega) ⟨rfl, rfl, hrec⟩)
      obtain ⟨out2, pos2, hp_eq, hpos2, hemit2⟩ := hp
      have hloop' : dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0_loop0 out true none
          scale qs take pos base width plane 0#usize = ok (out2, true, none, pos2) := by
        rw [hloop, hp_eq]
      obtain ⟨plane1, haddp, hplane1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := plane) (y := 1#usize) (by scalar_tac))
      have hplane1 : plane1.val = plane.val + 1 := by rw [hplane1v]; simp
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0.body scale qs take
          base width out true none pos plane = ok (ControlFlow.cont (out2, true, none, pos2, plane1)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0.body
        rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), if_pos rfl]
        simp only [hloop', haddp, bind_tc_ok]
        rfl
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨hemit2, by rw [hpos2, hposrel, hplane1]; ring, by omega⟩, ?_⟩
      show 5 - plane1.val < 5 - plane.val
      omega
    · have hnot : ¬ (plane < 5#usize) := fun hc => hlt ((UScalar.lt_equiv plane 5#usize).1 hc)
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0.body scale qs take
          base width out true none pos plane = ok (ControlFlow.done (out, true, none, pos)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hplane5 : plane.val = 5 := by omega
      refine ⟨out, pos, rfl, by rw [hposrel, hplane5], rfl, rfl, hrec⟩
  · exact ⟨hinv0, by simp, by simp⟩

/-! ## `qs` base loop -/

/-- **`qs` base loop.** Runs the plane loop for successive base offsets `base0 + k·width` while
`base0 + k·width + width ≤ 48`, advancing `pos` by `5·width` each time. Capacity pad `768`. -/
theorem tq1_0Base_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 48#usize) (take : Usize) (pos0 base0 width : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hwidth : 0 < width.val)
    (hwidth48 : width.val ≤ 48)
    (hbase0 : base0.val ≤ 48)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 48 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 54 + j) 0#u8).val)
    (hOK : ∀ k, base0.val + width.val * k + width.val ≤ 48 →
      tq1_0QsOK raw b (pos0.val + 5 * width.val * k) (base0.val + width.val * k) width.val)
    (hpos256 : ∀ k, base0.val + width.val * k + width.val ≤ 48 →
      pos0.val + 5 * width.val * k + 5 * width.val ≤ 256)
    (hprod : ∀ k, base0.val + width.val * k + width.val ≤ 48 → ∀ p, p < 5 → ∀ l, l < width.val →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base0.val + width.val * k + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (base0.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max)
    (hcap : out0.val.length + 768 < Usize.max)
    (hinv0 : tq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0 out0 true none scale qs take pos0
        base0 width)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos base : Usize),
        p = (out, true, none, pos, base) ∧ base.val ≤ 48 ∧ base.val + width.val > 48 ∧
        (∃ k, base.val = base0.val + width.val * k ∧ pos.val = pos0.val + 5 * width.val * k) ∧
        tq1_0EmitInv raw b take start out pos true none) := by
  unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      48 - s.2.2.2.2.val)
    (inv := fun s => tq1_0EmitInv raw b take start s.1 s.2.2.2.1 s.2.1 s.2.2.1 ∧
      s.2.2.2.2.val ≤ 48 ∧
      (∃ k, s.2.2.2.2.val = base0.val + width.val * k ∧
        s.2.2.2.1.val = pos0.val + 5 * width.val * k))
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos base : Usize),
      p = (out, true, none, pos, base) ∧ base.val ≤ 48 ∧ base.val + width.val > 48 ∧
      (∃ k, base.val = base0.val + width.val * k ∧ pos.val = pos0.val + 5 * width.val * k) ∧
      tq1_0EmitInv raw b take start out pos true none)
  · rintro ⟨out, ok1, err, pos, base⟩ ⟨hemit, hbasele, hk⟩
    dsimp only at hemit hbasele hk ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    obtain ⟨k, hbasek, hposk⟩ := hk
    rw [hok, herr]
    obtain ⟨i, hiadd, hiv⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := base) (y := width) (by scalar_tac))
    have hiv' : i.val = base.val + width.val := by rw [hiv]
    have hlen : out.val.length ≤ out0.val.length + 256 := by
      have h := congrArg List.length hrec
      simp only [dequantVals, List.length_append, List.length_map] at h
      have h4 := (List.take_sublist (l := tq1_0BlockVals raw b)
        (i := Nat.min pos.val take.val)).length_le
      simp only [tq1_0BlockVals_length] at h4
      have hstart : start.val.length ≤ out0.val.length := by
        have h5 := congrArg List.length hinv0.2.2
        simp only [dequantVals, List.length_append, List.length_map] at h5
        have h6 := (List.take_sublist (l := tq1_0BlockVals raw b)
          (i := Nat.min pos0.val take.val)).length_le
        simp only [tq1_0BlockVals_length] at h6
        omega
      omega
    by_cases hlt : base.val + width.val ≤ 48
    · have hkcond : base0.val + width.val * k + width.val ≤ 48 := by
        rw [hbasek] at hlt; omega
      have hOKk : tq1_0QsOK raw b pos.val base.val width.val := by
        have h := hOK k hkcond
        rw [hposk, hbasek]
        exact h
      have hpos256k : pos.val + 5 * width.val ≤ 256 := by
        have h := hpos256 k hkcond
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
        (tq1_0Plane_loop_spec raw b scale qs take base width pos start out hsc hqs hOKk
          hpos256k hlt hprodk (by omega) ⟨rfl, rfl, hrec⟩)
      obtain ⟨out1, pos1, hq_eq, hpos1, hemit1⟩ := hq
      have hplane' : dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0 out true none scale
          qs take pos base width 0#usize = ok (out1, true, none, pos1) := by rw [hplane_call, hq_eq]
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0.body scale qs take width out
          true none pos base = ok (ControlFlow.cont (out1, true, none, pos1, i)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0.body
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
      · show 48 - i.val < 48 - base.val
        omega
    · have hnlt : ¬ (i ≤ 48#usize) := fun hc => hlt (by
        rw [UScalar.le_equiv] at hc; rw [hiv'] at hc; exact hc)
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0.body scale qs take width out
          true none pos base = ok (ControlFlow.done (out, true, none, pos, base)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0.body
        rw [hiadd]
        simp only [bind_tc_ok]
        rw [if_neg hnlt]
      have hibig : 48 < i.val := by
        rw [UScalar.le_equiv] at hnlt; omega
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨out, pos, base, rfl, hbasele, by omega, ⟨k, hbasek, hposk⟩, rfl, rfl, hrec⟩
  · exact ⟨hinv0, hbase0, ⟨0, by simp, by simp⟩⟩

/-! ## `qs` segment product bounds -/

/-- From a `tq1_0QsOK raw b pos0 base width` segment and the block domain, the `i64` product bound
holds at every lane of the segment. -/
lemma tq1_0QsOK_prod (raw : Slice U8) (b : Nat) (scale : I64) (qs : Array U8 48#usize)
    (pos0 base width : Nat)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 48 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 54 + j) 0#u8).val)
    (hOK : tq1_0QsOK raw b pos0 base width)
    (hbound : ∀ r, r < 256 → I64.min ≤ tq1_0Weight raw b r ∧ tq1_0Weight raw b r ≤ I64.max)
    (hr256 : ∀ p, p < 5 → ∀ l, l < width → pos0 + p * width + l < 256)
    (hbase48 : base + width ≤ 48) :
    ∀ p, p < 5 → ∀ l, l < width →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (base + l) 0#u8) p - 1) ≤ I64.max := by
  intro p hp l hl
  have h := hOK p hp l hl
  obtain ⟨hbyte, hplane⟩ := h
  have hr := hr256 p hp l hl
  have hbl : base + l < 48 := by omega
  have hidx : b * 54 + tq1_0Byte (pos0 + p * width + l) = b * 54 + (base + l) := by rw [hbyte]
  have hbyteEq : (qs.val.getD (base + l) 0#u8).val =
      (raw.val.getD (b * 54 + tq1_0Byte (pos0 + p * width + l)) 0#u8).val := by
    rw [hidx]; exact hqs (base + l) hbl
  have htrit : tritRef (qs.val.getD (base + l) 0#u8) p =
      tritRef (raw.val.getD (b * 54 + tq1_0Byte (pos0 + p * width + l)) 0#u8)
        (tq1_0Plane (pos0 + p * width + l)) := by
    have h1 := tritRef_congr hbyteEq p
    rw [hplane]
    exact h1
  have hW : tq1_0Weight raw b (pos0 + p * width + l) =
      tq1_0ScaleRef raw b * (tritRef (qs.val.getD (base + l) 0#u8) p - 1) := by
    unfold tq1_0Weight
    rw [htrit]
  have hb0 := hbound (pos0 + p * width + l) hr
  rw [hW, ← hsc] at hb0
  exact hb0

/-- `TQ1_0_STAGES` read through `getD`. -/
lemma TQ1_0_STAGES_getD (i : Nat) (h : i < 2) :
    (dequant.lowbit.TQ1_0_STAGES.val.getD i 0#usize).val = if i = 0 then 32 else 16 := by
  rcases (by omega : i = 0 ∨ i = 1) with h0 | h1
  · subst h0; unfold dequant.lowbit.TQ1_0_STAGES; decide
  · subst h1; unfold dequant.lowbit.TQ1_0_STAGES; decide

/-- The emitted output of an `emitInv` state exceeds its block start by at most `256`. -/
lemma tq1_0EmitInv_length {raw : Slice U8} {b : Nat} {take : Usize}
    {start out : alloc.vec.Vec I64} {pos : Usize} {ok1 : Bool} {err : Option error.EngineError}
    (h : tq1_0EmitInv raw b take start out pos ok1 err) :
    out.val.length ≤ start.val.length + 256 := by
  obtain ⟨-, -, he⟩ := h
  have h1 := congrArg List.length he
  simp only [dequantVals, List.length_append, List.length_map] at h1
  have h2 := (List.take_sublist (l := tq1_0BlockVals raw b)
    (i := Nat.min pos.val take.val)).length_le
  simp only [tq1_0BlockVals_length] at h2
  omega

/-- The block start of an `emitInv` state is no longer than its output. -/
lemma tq1_0EmitInv_start_le {raw : Slice U8} {b : Nat} {take : Usize}
    {start out : alloc.vec.Vec I64} {pos : Usize} {ok1 : Bool} {err : Option error.EngineError}
    (h : tq1_0EmitInv raw b take start out pos ok1 err) :
    start.val.length ≤ out.val.length := by
  obtain ⟨-, -, he⟩ := h
  have h1 := congrArg List.length he
  simp only [dequantVals, List.length_append, List.length_map] at h1
  omega

/-- The `TQ1_0` staged phase: `(pos, base, stage)` after `stage` stages. -/
def tq1_0StagePhase (pos base stage : Nat) : Prop :=
  (stage = 0 ∧ base = 0 ∧ pos = 0) ∨
  (stage = 1 ∧ base = 32 ∧ pos = 160) ∨
  (stage = 2 ∧ base = 48 ∧ pos = 240)

/-- One `TQ1_0` stage-loop body step: read the stage width, run the base loop, advance the stage. -/
theorem tq1_0Stage_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 48#usize) (take : Usize) (stage : Usize)
    (start out : alloc.vec.Vec I64) (pos base width : Usize)
    (hstage : stage.val < 2)
    (hwidthread : Array.index_usize dequant.lowbit.TQ1_0_STAGES stage = ok width)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 48 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 54 + j) 0#u8).val)
    (hOK : ∀ k, base.val + width.val * k + width.val ≤ 48 →
      tq1_0QsOK raw b (pos.val + 5 * width.val * k) (base.val + width.val * k) width.val)
    (hpos256 : ∀ k, base.val + width.val * k + width.val ≤ 48 →
      pos.val + 5 * width.val * k + 5 * width.val ≤ 256)
    (hprod : ∀ k, base.val + width.val * k + width.val ≤ 48 → ∀ p, p < 5 → ∀ l, l < width.val →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max)
    (hw : 0 < width.val) (hw48 : width.val ≤ 48) (hbase : base.val ≤ 48)
    (hcap : out.val.length + 1024 < Usize.max)
    (hinv : tq1_0EmitInv raw b take start out pos true none) :
    ∃ (out1 : alloc.vec.Vec I64) (pos1 base1 stage1 : Usize),
      dequant.lowbit.dequant_tq1_0_row_loop0_loop0.body scale qs take out true none pos base stage
        = ok (ControlFlow.cont (out1, true, none, pos1, base1, stage1)) ∧
      tq1_0EmitInv raw b take start out1 pos1 true none ∧
      base1.val ≤ 48 ∧ base1.val + width.val > 48 ∧
      (∃ k, base1.val = base.val + width.val * k ∧ pos1.val = pos.val + 5 * width.val * k) ∧
      stage1.val = stage.val + 1 := by
  obtain ⟨stage1, hstage1a, hstage1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := stage) (y := 1#usize) (by scalar_tac))
  have hstage1 : stage1.val = stage.val + 1 := by rw [hstage1v]; simp
  obtain ⟨q, hbase_call, hq⟩ := WP.spec_imp_exists
    (tq1_0Base_loop_spec raw b scale qs take pos base width start out hw hw48 hbase hsc hqs hOK
      hpos256 hprod (by omega) hinv)
  obtain ⟨out1, pos1, base1, hq_eq, hb1le, hb1gt, hkq, hemit1⟩ := hq
  have hbase' : dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0 out true none scale qs take pos
      base width = ok (out1, true, none, pos1, base1) := by rw [hbase_call, hq_eq]
  refine ⟨out1, pos1, base1, stage1, ?_, hemit1, hb1le, hb1gt, hkq, hstage1⟩
  unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hstage), if_pos rfl]
  rw [hwidthread]
  simp only [bind_tc_ok]
  rw [hbase']
  simp only [bind_tc_ok]
  rw [hstage1a]
  simp only [bind_tc_ok]
  rfl

/-- **`TQ1_0` stage loop.** Two stages of `TQ1_0_STAGES` walk `qs` from `(base, pos) = (0, 0)` to
`(48, 240)`, leaving the emission invariant. Capacity pad `1024`. -/
theorem tq1_0Stage_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 48#usize) (take : Usize) (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 48 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 54 + j) 0#u8).val)
    (hbound : ∀ r, r < 256 → I64.min ≤ tq1_0Weight raw b r ∧ tq1_0Weight raw b r ≤ I64.max)
    (hcap : out0.val.length + 1280 < Usize.max)
    (hinv0 : tq1_0EmitInv raw b take start out0 0#usize true none) :
    WP.spec (dequant.lowbit.dequant_tq1_0_row_loop0_loop0 out0 true none scale qs take 0#usize
        0#usize 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none, pos') ∧ pos'.val = 240 ∧
        tq1_0EmitInv raw b take start out pos' true none) := by
  have hprod0 : ∀ p, p < 5 → ∀ l, l < 32 →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (0 + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (0 + l) 0#u8) p - 1) ≤ I64.max :=
    tq1_0QsOK_prod raw b scale qs 0 0 32 hsc hqs (tq1_0QsOK_zero32 raw b) hbound
      (by intro p hp l hl; omega) (by norm_num)
  have hprod1 : ∀ p, p < 5 → ∀ l, l < 16 →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (32 + l) 0#u8) p - 1) ∧
      scale.val * (tritRef (qs.val.getD (32 + l) 0#u8) p - 1) ≤ I64.max :=
    tq1_0QsOK_prod raw b scale qs 160 32 16 hsc hqs (tq1_0QsOK_lane16 raw b) hbound
      (by intro p hp l hl; omega) (by norm_num)
  unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize × Usize =>
      2 - s.2.2.2.2.2.val)
    (inv := fun s => tq1_0EmitInv raw b take start s.1 s.2.2.2.1 s.2.1 s.2.2.1 ∧
      tq1_0StagePhase s.2.2.2.1.val s.2.2.2.2.1.val s.2.2.2.2.2.val)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none, pos') ∧ pos'.val = 240 ∧
      tq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, base, stage⟩ ⟨hemit, hphase⟩
    dsimp only at hemit hphase ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    have hcapOut : out.val.length + 1024 < Usize.max := by
      have h1 := tq1_0EmitInv_length (⟨hok, herr, hrec⟩ :
        tq1_0EmitInv raw b take start out pos ok1 err)
      have h2 := tq1_0EmitInv_start_le hinv0
      omega
    by_cases hlt : stage.val < 2
    · have hidx : stage.val < dequant.lowbit.TQ1_0_STAGES.length := by
        rw [TQ1_0_STAGES_len]; omega
      obtain ⟨width, hwidthread, hwidthv⟩ := WP.spec_imp_exists
        (Array.index_usize_spec dequant.lowbit.TQ1_0_STAGES stage hidx)
      have hwvg : width.val = (dequant.lowbit.TQ1_0_STAGES.val.getD stage.val 0#usize).val := by
        rw [hwidthv]
        exact congrArg (fun u : Usize => u.val)
          (list_getD_getElem (dequant.lowbit.TQ1_0_STAGES.val) stage.val hidx 0#usize)
      rcases hphase with ⟨hs, hb, hp⟩ | ⟨hs, hb, hp⟩ | ⟨hs, hb, hp⟩
      · -- stage 0: width 32, base 0, pos 0
        have hwv : width.val = 32 := by
          rw [hwvg, hs]; rw [TQ1_0_STAGES_getD 0 (by omega)]; norm_num
        have hOK : ∀ k, base.val + width.val * k + width.val ≤ 48 →
            tq1_0QsOK raw b (pos.val + 5 * width.val * k) (base.val + width.val * k) width.val := by
          intro k hk
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0
          simpa [hb, hp, hwv] using tq1_0QsOK_zero32 raw b
        have hpos256 : ∀ k, base.val + width.val * k + width.val ≤ 48 →
            pos.val + 5 * width.val * k + 5 * width.val ≤ 256 := by
          intro k hk
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0; rw [hp, hwv]; norm_num
        have hprod : ∀ k, base.val + width.val * k + width.val ≤ 48 → ∀ p, p < 5 →
            ∀ l, l < width.val →
            I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ∧
            scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max := by
          intro k hk p hp' l hl
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0
          have h := hprod0 p hp' l (by rw [hwv] at hl; omega)
          simpa [hb, hwv] using h
        obtain ⟨out1, pos1, base1, stage1, hbody, hemit1, hb1le, hb1gt, hkq, hstage1⟩ :=
          tq1_0Stage_step raw b scale qs take stage start out pos base width hlt hwidthread hsc hqs
            hOK hpos256 hprod (by rw [hwv]; norm_num) (by rw [hwv]; norm_num) (by rw [hb]; norm_num)
            hcapOut ⟨rfl, rfl, hrec⟩
        obtain ⟨k, hbk, hpk⟩ := hkq
        rw [hwv] at hbk hpk hb1gt
        have hk1 : k = 1 := by omega
        have hb1 : base1.val = 32 := by omega
        have hp1 : pos1.val = 160 := by omega
        rw [hbody]
        simp only [WP.spec_ok]
        refine ⟨⟨hemit1, ?_⟩, ?_⟩
        · right; left
          exact ⟨by omega, hb1, hp1⟩
        · show 2 - stage1.val < 2 - stage.val
          omega
      · -- stage 1: width 16, base 32, pos 160
        have hwv : width.val = 16 := by
          rw [hwvg, hs]; rw [TQ1_0_STAGES_getD 1 (by omega)]; norm_num
        have hOK : ∀ k, base.val + width.val * k + width.val ≤ 48 →
            tq1_0QsOK raw b (pos.val + 5 * width.val * k) (base.val + width.val * k) width.val := by
          intro k hk
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0
          simpa [hb, hp, hwv] using tq1_0QsOK_lane16 raw b
        have hpos256 : ∀ k, base.val + width.val * k + width.val ≤ 48 →
            pos.val + 5 * width.val * k + 5 * width.val ≤ 256 := by
          intro k hk
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0; rw [hp, hwv]; norm_num
        have hprod : ∀ k, base.val + width.val * k + width.val ≤ 48 → ∀ p, p < 5 →
            ∀ l, l < width.val →
            I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ∧
            scale.val * (tritRef (qs.val.getD (base.val + width.val * k + l) 0#u8) p - 1) ≤ I64.max := by
          intro k hk p hp' l hl
          have hk0 : k = 0 := by rw [hwv] at hk; omega
          subst hk0
          have h := hprod1 p hp' l (by rw [hwv] at hl; omega)
          simpa [hb, hp, hwv] using h
        obtain ⟨out1, pos1, base1, stage1, hbody, hemit1, hb1le, hb1gt, hkq, hstage1⟩ :=
          tq1_0Stage_step raw b scale qs take stage start out pos base width hlt hwidthread hsc hqs
            hOK hpos256 hprod (by rw [hwv]; norm_num) (by rw [hwv]; norm_num) (by rw [hb]; norm_num)
            hcapOut ⟨rfl, rfl, hrec⟩
        obtain ⟨k, hbk, hpk⟩ := hkq
        rw [hwv] at hbk hpk hb1gt
        have hk1 : k = 1 := by omega
        have hb1 : base1.val = 48 := by omega
        have hp1 : pos1.val = 240 := by omega
        rw [hbody]
        simp only [WP.spec_ok]
        refine ⟨⟨hemit1, ?_⟩, ?_⟩
        · right; right
          exact ⟨by omega, hb1, hp1⟩
        · show 2 - stage1.val < 2 - stage.val
          omega
      · omega
    · have hnot : ¬ (stage < 2#usize) := fun hc => hlt ((UScalar.lt_equiv stage 2#usize).1 hc)
      have hstage2 : stage.val = 2 := by
        rcases hphase with ⟨hs, _, _⟩ | ⟨hs, _, _⟩ | ⟨hs, _, _⟩ <;> omega
      have hpos : pos.val = 240 := by
        rcases hphase with ⟨_, _, hp⟩ | ⟨_, _, hp⟩ | ⟨_, _, hp⟩ <;> omega
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop0.body scale qs take out true none
          pos base stage = ok (ControlFlow.done (out, true, none, pos)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨out, pos, rfl, hpos, rfl, rfl, hrec⟩
  · exact ⟨hinv0, Or.inl ⟨rfl, rfl, rfl⟩⟩

/-! ## `TQ1_0` `qh` tail -/

/-- Tail-loop invariant: `emitInv` and `pos = pos0 + tail`, `tail ≤ 4`. -/
def tq1_0TailInv (raw : Slice U8) (b : Nat) (take : Usize) (pos0 : Nat)
    (start out : alloc.vec.Vec I64) (pos tail : Usize) (ok1 : Bool)
    (err : Option error.EngineError) : Prop :=
  tq1_0EmitInv raw b take start out pos ok1 err ∧ pos.val = pos0 + tail.val ∧ tail.val ≤ 4

/-- One `qh`-tail lane-loop body step: decode `qh[tail]` at plane `plane`, multiply and push. -/
theorem tq1_0Tail_body_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qh : Array U8 4#usize) (take : Usize) (plane pos0 : Usize)
    (start out : alloc.vec.Vec I64) (pos tail : Usize)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqh : ∀ j, j < 4 → (qh.val.getD j 0#u8).val = (raw.val.getD (b * 54 + 48 + j) 0#u8).val)
    (hplane4 : plane.val < 4)
    (hbyte : tq1_0Byte pos.val = 48 + tail.val)
    (hplane : tq1_0Plane pos.val = plane.val)
    (htail : tail.val < 4)
    (hpos256 : pos.val < 256)
    (hmul : I64.min ≤ scale.val * (tritRef (qh.val.getD tail.val 0#u8) plane.val - 1) ∧
      scale.val * (tritRef (qh.val.getD tail.val 0#u8) plane.val - 1) ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : tq1_0TailInv raw b take pos0.val start out pos tail true none) :
    ∃ (out2 : alloc.vec.Vec I64) (pos2 tail2 : Usize),
      dequant.lowbit.dequant_tq1_0_row_loop0_loop1_loop0.body scale qh take plane out true none pos tail
        = ok (ControlFlow.cont (out2, true, none, pos2, tail2)) ∧
      tq1_0TailInv raw b take pos0.val start out2 pos2 tail2 true none ∧
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
      tritRef (raw.val.getD (b * 54 + tq1_0Byte pos.val) 0#u8) (tq1_0Plane pos.val) := by
    rw [hplane]
    apply tritRef_congr
    rw [hqh tail.val htail, Nat.add_assoc, ← hbyte]
  have hvval : v.val = (tq1_0BlockVals raw b)[pos.val]'(by
      rw [tq1_0BlockVals_length]; exact hpos256) := by
    rw [hvv, hi3val, hsc, hcong]
    rw [list_getD_getElem (tq1_0BlockVals raw b) pos.val
      (by rw [tq1_0BlockVals_length]; exact hpos256) (0 : ℤ), tq1_0BlockVals_getD raw b pos.val hpos256]
    unfold tq1_0Weight
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
    · unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop1_loop0.body
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
        List.take_succ_eq_append_getElem (by rw [tq1_0BlockVals_length]; exact hpos256),
        List.append_assoc]
  · obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨tail2, haddl, htail2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := tail) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have htail2 : tail2.val = tail.val + 1 := by rw [htail2v]; simp
    refine ⟨out, pos2, tail2, ?_, ?_, hpos2, htail2⟩
    · unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop1_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact htail), if_pos rfl]
      simp only [hi1, htrit, hsub, hmulv, bind_tc_ok]
      rw [if_neg (fun hc => hlt ((UScalar.lt_equiv pos take).1 hc))]
      simp only [haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, by rw [hpos2, hposrel, htail2]; omega, by rw [htail2]; omega⟩
      have h1 : (pos.val + 1).min take.val = take.val := Nat.min_eq_right (by omega)
      have h2 : pos.val.min take.val = take.val := Nat.min_eq_right (by omega)
      rw [hrec, hpos2, h1, h2]

/-- **`TQ1_0` `qh` tail loop.** From `tail = 0`, the four `qh` lanes emit `4` staged values. -/
theorem tq1_0Tail_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qh : Array U8 4#usize) (take : Usize) (plane pos0 : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqh : ∀ j, j < 4 → (qh.val.getD j 0#u8).val = (raw.val.getD (b * 54 + 48 + j) 0#u8).val)
    (hplane4 : plane.val < 4)
    (hrun : ∀ lane : Usize, lane.val < 4 →
      tq1_0Byte (pos0.val + lane.val) = 48 + lane.val ∧
      tq1_0Plane (pos0.val + lane.val) = plane.val)
    (hpos256 : pos0.val + 4 ≤ 256)
    (hmul : ∀ lane : Usize, lane.val < 4 →
      I64.min ≤ scale.val * (tritRef (qh.val.getD lane.val 0#u8) plane.val - 1) ∧
      scale.val * (tritRef (qh.val.getD lane.val 0#u8) plane.val - 1) ≤ I64.max)
    (hcap : out0.val.length + 256 < Usize.max)
    (hinv0 : tq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_tq1_0_row_loop0_loop1_loop0 out0 true none scale qh take pos0
        plane 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none, pos') ∧ pos'.val = pos0.val + 4 ∧
        tq1_0EmitInv raw b take start out pos' true none) := by
  have hinit : tq1_0TailInv raw b take pos0.val start out0 pos0 0#usize true none :=
    ⟨hinv0, by simp, by simp⟩
  unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop1_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      4 - s.2.2.2.2.val)
    (inv := fun s => tq1_0TailInv raw b take pos0.val start s.1 s.2.2.2.1 s.2.2.2.2 s.2.1 s.2.2.1)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none, pos') ∧ pos'.val = pos0.val + 4 ∧
      tq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, tail⟩ ⟨hemit, hposrel, htaille⟩
    dsimp only at hemit hposrel htaille ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : tail.val < 4
    · have hpos256' : pos.val < 256 := by omega
      have hbyte : tq1_0Byte pos.val = 48 + tail.val := by rw [hposrel]; exact (hrun tail hlt).1
      have hplane : tq1_0Plane pos.val = plane.val := by rw [hposrel]; exact (hrun tail hlt).2
      have hcap' : out.val.length < Usize.max := by
        have h1 := tq1_0EmitInv_length (⟨hok, herr, hrec⟩ :
          tq1_0EmitInv raw b take start out pos ok1 err)
        have h2 := tq1_0EmitInv_start_le hinv0
        omega
      obtain ⟨out2, pos2, tail2, hbody, hinv2, hpos2, htail2⟩ :=
        tq1_0Tail_body_step raw b scale qh take plane pos0 start out pos tail hsc hqh hplane4 hbyte
          hplane hlt hpos256' (hmul tail hlt) hcap' ⟨⟨rfl, rfl, hrec⟩, hposrel, htaille⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨?_, ?_⟩
      · exact hinv2
      · show 4 - tail2.val < 4 - tail.val
        omega
    · have hnot : ¬ (tail < 4#usize) := fun hc => hlt ((UScalar.lt_equiv tail 4#usize).1 hc)
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop1_loop0.body scale qh take plane out
          true none pos tail = ok (ControlFlow.done (out, true, none, pos)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop1_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have htail4 : tail.val = 4 := by omega
      refine ⟨out, pos, rfl, by rw [hposrel, htail4], rfl, rfl, hrec⟩
  · exact hinit

/-- **`TQ1_0` `qh` tail plane loop.** The four tail planes emit `16` staged values. Capacity pad
`512`. -/
theorem tq1_0TailPlane_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qh : Array U8 4#usize) (take : Usize) (pos0 : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqh : ∀ j, j < 4 → (qh.val.getD j 0#u8).val = (raw.val.getD (b * 54 + 48 + j) 0#u8).val)
    (hOK : ∀ p, p < 4 → ∀ l, l < 4 →
      tq1_0Byte (pos0.val + p * 4 + l) = 48 + l ∧ tq1_0Plane (pos0.val + p * 4 + l) = p)
    (hpos256 : pos0.val + 16 ≤ 256)
    (hprod : ∀ p, p < 4 → ∀ l, l < 4 →
      I64.min ≤ scale.val * (tritRef (qh.val.getD l 0#u8) p - 1) ∧
      scale.val * (tritRef (qh.val.getD l 0#u8) p - 1) ≤ I64.max)
    (hcap : out0.val.length + 512 < Usize.max)
    (hinv0 : tq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_tq1_0_row_loop0_loop1 out0 true none scale qh take pos0 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none) ∧ pos'.val = pos0.val + 16 ∧
        tq1_0EmitInv raw b take start out pos' true none) := by
  unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop1
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      4 - s.2.2.2.2.val)
    (inv := fun s => tq1_0EmitInv raw b take start s.1 s.2.2.2.1 s.2.1 s.2.2.1 ∧
      s.2.2.2.1.val = pos0.val + s.2.2.2.2.val * 4 ∧ s.2.2.2.2.val ≤ 4)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none) ∧ pos'.val = pos0.val + 16 ∧
      tq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, plane⟩ ⟨hemit, hposrel, hplanele⟩
    dsimp only at hemit hposrel hplanele ⊢
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : plane.val < 4
    · have hpos256' : pos.val + 4 ≤ 256 := by
        have he : pos.val + 4 = pos0.val + (plane.val + 1) * 4 := by rw [hposrel]; ring
        rw [he]
        have hpl : plane.val + 1 ≤ 4 := by omega
        calc pos0.val + (plane.val + 1) * 4 ≤ pos0.val + 4 * 4 := by
              exact Nat.add_le_add_left (Nat.mul_le_mul_right _ hpl) _
          _ ≤ 256 := hpos256
      have hrun : ∀ lane : Usize, lane.val < 4 →
          tq1_0Byte (pos.val + lane.val) = 48 + lane.val ∧
          tq1_0Plane (pos.val + lane.val) = plane.val := by
        intro lane hlane
        have h := hOK plane.val hlt lane.val hlane
        have hidx : pos.val + lane.val = pos0.val + plane.val * 4 + lane.val := by rw [hposrel]
        rw [hidx]
        exact h
      have hmul' : ∀ lane : Usize, lane.val < 4 →
          I64.min ≤ scale.val * (tritRef (qh.val.getD lane.val 0#u8) plane.val - 1) ∧
          scale.val * (tritRef (qh.val.getD lane.val 0#u8) plane.val - 1) ≤ I64.max :=
        fun lane hlane => hprod plane.val hlt lane.val hlane
      have hlen : out.val.length ≤ out0.val.length + 256 := by
        have h1 := tq1_0EmitInv_length (⟨hok, herr, hrec⟩ :
          tq1_0EmitInv raw b take start out pos ok1 err)
        have h2 := tq1_0EmitInv_start_le hinv0
        omega
      obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
        (tq1_0Tail_loop_spec raw b scale qh take plane pos start out hsc hqh hlt hrun hpos256'
          hmul' (by omega) ⟨rfl, rfl, hrec⟩)
      obtain ⟨out2, pos2, hp_eq, hpos2, hemit2⟩ := hp
      have hloop' : dequant.lowbit.dequant_tq1_0_row_loop0_loop1_loop0 out true none scale qh take
          pos plane 0#usize = ok (out2, true, none, pos2) := by rw [hloop, hp_eq]
      obtain ⟨plane1, haddp, hplane1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := plane) (y := 1#usize) (by scalar_tac))
      have hplane1 : plane1.val = plane.val + 1 := by rw [hplane1v]; simp
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop1.body scale qh take out true none pos
          plane = ok (ControlFlow.cont (out2, true, none, pos2, plane1)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop1.body
        rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), if_pos rfl]
        simp only [hloop', haddp, bind_tc_ok]
        rfl
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨hemit2, by rw [hpos2, hposrel, hplane1]; ring, by omega⟩, ?_⟩
      show 4 - plane1.val < 4 - plane.val
      omega
    · have hnot : ¬ (plane < 4#usize) := fun hc => hlt ((UScalar.lt_equiv plane 4#usize).1 hc)
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop1.body scale qh take out true none pos
          plane = ok (ControlFlow.done (out, true, none)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop1.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hplane4 : plane.val = 4 := by omega
      refine ⟨out, pos, rfl, by rw [hposrel, hplane4], rfl, rfl, hrec⟩
  · exact ⟨hinv0, by simp, by simp⟩

/-! ## `TQ1_0` row composition -/

/-- The `TQ1_0` block success domain: the 54 bytes are present, `d` is finite, and every one of the
256 staged weights is inside `i64`. -/
def tq1_0BlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  b.val * 54 + 54 ≤ raw.val.length ∧
  f16Exp (f16LeHalf raw (b.val * 54 + 52)) ≠ 31 ∧
  (∀ r, r < 256 → I64.min ≤ tq1_0Weight raw b.val r ∧ tq1_0Weight raw b.val r ≤ I64.max)

/-- The first `take ≤ 256` entries of a staged block stream. -/
lemma tq1_0BlockVals_take (raw : Slice U8) (b take : Nat) (htake : take ≤ 256) :
    (tq1_0BlockVals raw b).take take = (List.range take).map (fun j => tq1_0Weight raw b j) := by
  unfold tq1_0BlockVals
  rw [← List.map_take, List.take_range, Nat.min_eq_left htake]

/-- Appending the next `take` row weights advances the row prefix by `take`. -/
lemma tq1_0RowValues_append (raw : Slice U8) (m take : Nat) :
    tq1_0RowValues raw m ++
      (List.range take).map (fun j => tq1_0Weight raw ((m + j) / 256) ((m + j) % 256)) =
    tq1_0RowValues raw (m + take) := by
  unfold tq1_0RowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

lemma div256_mul_add (g l : Nat) (hl : l < 256) : (256 * g + l) / 256 = g := by omega
lemma mod256_mul_add (g l : Nat) (hl : l < 256) : (256 * g + l) % 256 = l := by omega

lemma mul_block_add_min_256 (b n : Nat) (h : 256 * b ≤ n) :
    256 * b + Nat.min 256 (n - 256 * b) = Nat.min (256 * (b + 1)) n := by
  change 256 * b + min 256 (n - 256 * b) = min (256 * (b + 1)) n
  by_cases h1 : n - 256 * b ≤ 256
  · rw [Nat.min_eq_right h1, Nat.min_eq_right (by omega)]
    omega
  · have h1' : 256 < n - 256 * b := not_le.mp h1
    rw [Nat.min_eq_left (le_of_lt h1'), Nat.min_eq_left (by omega)]
    omega

/-- The extracted `div_ceil n TQ1_0_QK` succeeds with `(n + 255)/256`. -/
theorem dequant_div_ceil_tq1_0_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.TQ1_0_QK = ok blocks ∧
      blocks.val = (n.val + 255) / 256 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.TQ1_0_QK) (by rw [quant_TQ1_0_QK_eq]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.TQ1_0_QK) (by rw [quant_TQ1_0_QK_eq]; decide))
  have hqv' : q.val = n.val / 256 := by rw [hqv, quant_TQ1_0_QK_eq]; simp
  have hrv' : r.val = n.val % 256 := by rw [hrv, quant_TQ1_0_QK_eq]; simp
  have hq' : UScalar.div n quant.TQ1_0_QK = ok q := hq
  have hr' : UScalar.rem n quant.TQ1_0_QK = ok r := hr
  unfold core.num.Usize.div_ceil
  rw [hq']
  simp only [bind_tc_ok]
  rw [hr']
  simp only [bind_tc_ok]
  by_cases hr0 : r.val = 0
  · refine ⟨q, ?_, ?_⟩
    · rw [if_pos hr0]
    · rw [hqv']
      have hr0' : n.val % 256 = 0 := by rw [← hrv']; exact hr0
      omega
  · obtain ⟨q1, hq1, hq1v⟩ := WP.spec_imp_exists
      (UScalar.add_spec (x := q) (y := (UScalar.one : Usize)) (by rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]
      exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 256 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (256·block) n` raw weights. -/
def tq1_0RowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ tq1_0RowValues raw (Nat.min (256 * block.val) n.val)

/-- One `TQ1_0` block-loop body step: decode the block, run the `qs` stage loop and the `qh` tail
plane loop, append `take = min 256 (n - 256·block)` values and advance. -/
theorem tq1_0Block_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : tq1_0BlockDomain raw block)
    (hcap : out.val.length + 1280 < Usize.max)
    (hinv : tq1_0RowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.lowbit.dequant_tq1_0_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      tq1_0RowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨hlen, hfin, hbound⟩ := hdom
  obtain ⟨d, qs, qh, hq, hdv, hqsv, hqhv⟩ := tq1_0_block_ok raw block hlen hfin
  have hltmul : 256 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hblock)
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.TQ1_0_QK) (by rw [quant_TQ1_0_QK_eq]; scalar_tac))
  have hivv' : i.val = 256 * block.val := by rw [hivv, quant_TQ1_0_QK_eq]; scalar_tac
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 256 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.TQ1_0_QK i1
  have htakev' : take.val = Nat.min 256 (n.val - 256 * block.val) := by
    rw [htakev, hi1v', quant_TQ1_0_QK_eq]; simp
  have htake_le : take.val ≤ 256 := by rw [htakev']; exact min_le_left _ _
  have hinv0 : tq1_0EmitInv raw block.val take out out 0#usize true none := ⟨rfl, rfl, by simp⟩
  obtain ⟨p1, hstage_call, hp1⟩ := WP.spec_imp_exists
    (tq1_0Stage_loop_spec raw block.val d qs take out out hdv hqsv hbound (by omega) hinv0)
  obtain ⟨out1, pos1, hp1_eq, hpos1, hemit1⟩ := hp1
  have hstage' : dequant.lowbit.dequant_tq1_0_row_loop0_loop0 out true none d qs take 0#usize
      0#usize 0#usize = ok (out1, true, none, pos1) := by rw [hstage_call, hp1_eq]
  have hOKtail : ∀ p, p < 4 → ∀ l, l < 4 →
      tq1_0Byte (pos1.val + p * 4 + l) = 48 + l ∧ tq1_0Plane (pos1.val + p * 4 + l) = p := by
    intro p hp l hl
    rw [hpos1]
    exact ⟨tq1_0Byte_tail p l hp hl, tq1_0Plane_tail p l hp hl⟩
  have hprodtail : ∀ p, p < 4 → ∀ l, l < 4 →
      I64.min ≤ d.val * (tritRef (qh.val.getD l 0#u8) p - 1) ∧
      d.val * (tritRef (qh.val.getD l 0#u8) p - 1) ≤ I64.max := by
    intro p hp l hl
    have hr : 240 + p * 4 + l < 256 := by omega
    have hb := hbound (240 + p * 4 + l) hr
    have hbyte : tq1_0Byte (240 + p * 4 + l) = 48 + l := tq1_0Byte_tail p l hp hl
    have hplane : tq1_0Plane (240 + p * 4 + l) = p := tq1_0Plane_tail p l hp hl
    have hbyteEq : (qh.val.getD l 0#u8).val =
        (raw.val.getD (block.val * 54 + tq1_0Byte (240 + p * 4 + l)) 0#u8).val := by
      rw [hbyte]
      rw [show block.val * 54 + (48 + l) = block.val * 54 + 48 + l by omega]
      exact hqhv l hl
    have htrit : tritRef (qh.val.getD l 0#u8) p =
        tritRef (raw.val.getD (block.val * 54 + tq1_0Byte (240 + p * 4 + l)) 0#u8)
          (tq1_0Plane (240 + p * 4 + l)) := by
      have h1 := tritRef_congr hbyteEq p
      rw [hplane]
      exact h1
    have hW : tq1_0Weight raw block.val (240 + p * 4 + l) =
        tq1_0ScaleRef raw block.val * (tritRef (qh.val.getD l 0#u8) p - 1) := by
      unfold tq1_0Weight
      rw [htrit]
    have hb' := hb
    rw [hW, ← hdv] at hb'
    exact hb'
  have hlen1 : out1.val.length ≤ out.val.length + 256 :=
    tq1_0EmitInv_length hemit1
  obtain ⟨p2, htail_call, hp2⟩ := WP.spec_imp_exists
    (tq1_0TailPlane_loop_spec raw block.val d qh take pos1 out out1 hdv hqhv hOKtail
      (by rw [hpos1]) hprodtail (by omega) hemit1)
  obtain ⟨out2, pos2, hp2_eq, hpos2, hemit2⟩ := hp2
  have htail' : dequant.lowbit.dequant_tq1_0_row_loop0_loop1 out1 true none d qh take pos1 0#usize
      = ok (out2, true, none) := by rw [htail_call, hp2_eq]
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1 : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 256 * block.val + take.val = Nat.min (256 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min_256 block.val n.val (le_of_lt hltmul)
  have hinv_new : dequantVals out2.val =
      dequantVals out0.val ++ tq1_0RowValues raw (Nat.min (256 * block1.val) n.val) := by
    obtain ⟨-, -, hrec2⟩ := hemit2
    have hminpos : Nat.min pos2.val take.val = take.val := by
      rw [hpos2, hpos1]; exact Nat.min_eq_right htake_le
    rw [hrec2, hminpos, hinv.2.2.2, hmin, List.append_assoc,
      tq1_0BlockVals_take raw block.val take.val htake_le]
    have hmap : (List.range take.val).map (fun j => tq1_0Weight raw block.val j) =
        (List.range take.val).map
          (fun j => tq1_0Weight raw ((256 * block.val + j) / 256) ((256 * block.val + j) % 256)) := by
      apply List.map_congr_left
      intro j hj
      have hj256 : j < 256 := by have := List.mem_range.mp hj; omega
      rw [div256_mul_add block.val j hj256, mod256_mul_add block.val j hj256]
    rw [hmap, tq1_0RowValues_append, hsum, hblock1]
  refine ⟨out2, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1⟩
  rw [dequant.lowbit.dequant_tq1_0_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq]
  simp only [bind_tc_ok]
  change (do
    let i ← block * quant.TQ1_0_QK
    let i1 ← n - i
    let take ← core.cmp.Ord.min.trait_default core.cmp.OrdUsize quant.TQ1_0_QK i1
    let (out1, ok2, err1, pos) ←
      dequant.lowbit.dequant_tq1_0_row_loop0_loop0 out true none d qs take 0#usize 0#usize 0#usize
    let (out2, ok3, err2) ← dequant.lowbit.dequant_tq1_0_row_loop0_loop1 out1 ok2 err1 d qh take pos 0#usize
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
      dequant.lowbit.dequant_tq1_0_row_loop0_loop1 out1 true none d qh take pos1 0#usize
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out2, block1, ok3, err2)))
    = ok (ControlFlow.cont (out2, block1, true, none))
  rw [htail']
  simp only [bind_tc_ok]
  rw [hblock1a]
  simp only [bind_tc_ok]
  rfl

/-- The `TQ1_0` block loop: `loop.spec_decr_nat` with `tq1_0RowInv` and measure `blocks - block`. -/
theorem tq1_0Row_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : ∀ b : Usize, b.val < blocks.val → tq1_0BlockDomain raw b)
    (hcap : out0.val.length + n.val + 1280 < Usize.max) :
    WP.spec (dequant.lowbit.dequant_tq1_0_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ tq1_0RowValues raw n.val) := by
  unfold dequant.lowbit.dequant_tq1_0_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => tq1_0RowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, block, ok1, err⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : block.val < blocks.val
    · have hltmul : 256 * block.val < n.val :=
        lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hlt)
      have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
        Nat.min_eq_left (le_of_lt hltmul)
      have hlen_out : out.val.length = out0.val.length + 256 * block.val := by
        have h := congrArg List.length hrec
        rw [hmin] at h
        simpa [dequantVals, tq1_0RowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + 1280 < Usize.max := by rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        tq1_0Block_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.lowbit.dequant_tq1_0_row_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨trivial, ?_⟩
      have hblock_eq : block.val = blocks.val := by omega
      rw [hrec, hblock_eq, hblocks]
      have hminend : Nat.min (256 * ((n.val + 255) / 256)) n.val = n.val :=
        Nat.min_eq_right (le_mul_div_ceil_256 n.val)
      rw [hminend]
  · refine ⟨rfl, rfl, by simp, ?_⟩
    have h0 : Nat.min (256 * (0#usize).val) n.val = 0 := by simp
    rw [h0]
    simp [tq1_0RowValues]

/-- **`TQ1_0` row law.** On the block success domain, the extracted `dequant_tq1_0_row` succeeds
with `Ok ()` and appends exactly the first `n` raw weights `tq1_0RowValues raw n`. -/
theorem dequant_tq1_0_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 255) / 256 → tq1_0BlockDomain raw b)
    (hcap : out.val.length + n.val + 1280 < Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_tq1_0_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ tq1_0RowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_tq1_0_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (tq1_0Row_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.lowbit.dequant_tq1_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl



/-- The 54-byte `TQ1_0` witness block: `d = 1.0` (`0x3C00` at bytes 52–53), all code bytes `0`. -/
def tq1_0WitBytes : List U8 := List.replicate 52 0#u8 ++ [0#u8, 60#u8]

theorem tq1_0WitBytes_length : tq1_0WitBytes.length = 54 := by
  simp [tq1_0WitBytes]

def tq1_0WitRaw : Slice U8 := Slice.from tq1_0WitBytes (by rw [tq1_0WitBytes_length]; scalar_tac)

theorem tq1_0WitRaw_val : tq1_0WitRaw.val = tq1_0WitBytes := Slice.from_val _ _

/-- `getD` of the witness bytes before byte 52 is `0`. -/
lemma tq1_0Wit_getD_zero (k : Nat) (hk : k < 52) :
    (tq1_0WitBytes.getD k 0#u8).val = 0 := by
  unfold tq1_0WitBytes
  rw [List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_left (by rw [List.length_replicate]; exact hk)]
  rw [List.getElem?_replicate, if_pos hk]
  rfl

/-- The witness byte list's option at `52` is `0`. -/
lemma tq1_0Wit_getElem?_52 : tq1_0WitBytes[52]? = some 0#u8 := by
  unfold tq1_0WitBytes
  rw [List.getElem?_append_right (by rw [List.length_replicate])]
  rw [List.length_replicate, show 52 - 52 = 0 by omega]
  rfl

/-- The witness byte list's option at `53` is `60`. -/
lemma tq1_0Wit_getElem?_53 : tq1_0WitBytes[53]? = some 60#u8 := by
  unfold tq1_0WitBytes
  rw [List.getElem?_append_right (by rw [List.length_replicate]; omega)]
  rw [List.length_replicate, show 53 - 52 = 1 by omega]
  rfl

/-- Byte 52 of the witness is `0`. -/
lemma tq1_0Wit_getD_52 : (tq1_0WitBytes.getD 52 0#u8).val = 0 := by
  rw [List.getD_eq_getElem?_getD, tq1_0Wit_getElem?_52]
  rfl

/-- Byte 53 of the witness is `60`. -/
lemma tq1_0Wit_getD_53 : (tq1_0WitBytes.getD 53 0#u8).val = 60 := by
  rw [List.getD_eq_getElem?_getD, tq1_0Wit_getElem?_53]
  rfl

/-- The witness half-word is the binary16 pattern `0x3C00 = 15360` (`1.0`). -/
lemma tq1_0Wit_f16LeHalf : f16LeHalf tq1_0WitRaw 52 =
    core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) := by
  unfold f16LeHalf
  congr 1

/-- The half-word `[0x00, 0x3C]` is `15360`. -/
lemma tq1_0_u16_le_0_60 :
    (core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8])).val = 15360 := by decide

lemma tq1_0Wit_f16_val : (f16LeHalf tq1_0WitRaw 52).val = 15360 := by
  rw [tq1_0Wit_f16LeHalf, tq1_0_u16_le_0_60]

/-- The witness scale is `1.0 = 2^32` in Q32.32. -/
lemma tq1_0Wit_scale : tq1_0ScaleRef tq1_0WitRaw 0 = 2 ^ 32 := by
  have hv := tq1_0Wit_f16_val
  unfold tq1_0ScaleRef
  rw [show 0 * 54 + 52 = 52 by norm_num]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

/-- Every within-block byte index read by the walk is `< 52`, hence `0`. -/
lemma tq1_0Byte_lt52 (r : Nat) : tq1_0Byte r < 52 := by
  unfold tq1_0Byte
  split
  · have := Nat.mod_lt r (by norm_num : 0 < 32); omega
  · split
    · have := Nat.mod_lt (r - 160) (by norm_num : 0 < 16); omega
    · have := Nat.mod_lt (r - 240) (by norm_num : 0 < 4); omega

/-- The trit recovered from byte `0` is `0`. -/
lemma tritRef_zero (n : Nat) : tritRef 0#u8 n = 0 := by
  unfold tritRef
  norm_num

/-- Every staged weight of the witness block is `-2^32`. -/
lemma tq1_0Wit_weight (r : Nat) : tq1_0Weight tq1_0WitRaw 0 r = -(2 ^ 32) := by
  unfold tq1_0Weight
  rw [tq1_0Wit_scale]
  have hbyte : (tq1_0WitRaw.val.getD (0 * 54 + tq1_0Byte r) 0#u8).val = 0 := by
    rw [tq1_0WitRaw_val]
    have hlt : 0 * 54 + tq1_0Byte r < 52 := by
      have := tq1_0Byte_lt52 r; omega
    exact tq1_0Wit_getD_zero _ hlt
  have htrit : tritRef (tq1_0WitRaw.val.getD (0 * 54 + tq1_0Byte r) 0#u8) (tq1_0Plane r) = 0 := by
    have heq : tq1_0WitRaw.val.getD (0 * 54 + tq1_0Byte r) 0#u8 = 0#u8 :=
      UScalar.eq_of_val_eq (by simpa using hbyte)
    rw [heq, tritRef_zero]
  rw [htrit]
  norm_num

/-- The witness block satisfies the `TQ1_0` success domain. -/
theorem tq1_0BlockDomain_witness : tq1_0BlockDomain tq1_0WitRaw 0#usize := by
  have hb : (0#usize).val = 0 := by decide
  refine ⟨?_, ?_, ?_⟩
  · simp [hb, tq1_0WitRaw_val, tq1_0WitBytes_length]
  · have hv := tq1_0Wit_f16_val
    unfold f16Exp
    rw [hb, show 0 * 54 + 52 = 52 by norm_num, hv]
    norm_num
  · intro r hr
    rw [hb, tq1_0Wit_weight r]
    norm_num [I64.min_eq, I64.max_eq]

/-- The head of the `256`-element row list. -/
lemma tq1_0RowValues_head (raw : Slice U8) :
    (tq1_0RowValues raw 256).getD 0 0 = tq1_0Weight raw 0 0 := by
  unfold tq1_0RowValues
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 256)[0]? = some 0 := by decide
  rw [h]; rfl

/-- `getD` of the `dequantVals` map. -/
lemma tq1_0_dequantVals_getD (l : List I64) (i : Nat) :
    (dequantVals l).getD i 0 = (l.getD i 0#i64).val := by
  unfold dequantVals
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases h : l[i]? with
  | none => simp
  | some v => simp

lemma tq1_0_vec_new_val_i64 : (alloc.vec.Vec.new I64).val = ([] : List I64) := by
  simp [alloc.vec.Vec.new, alloc.vec.Vec.from_val]

/-- **`TQ1_0` one-block witness at `n = 256`.** The extracted row succeeds on the empty vector,
appends exactly the `256` weights of the witness block, and element `0` is `-2^32`. -/
theorem tq1_0Witness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_tq1_0_row tq1_0WitRaw 256#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = tq1_0RowValues tq1_0WitRaw 256 ∧
      (out1.val.getD 0 0#i64).val = -(2 ^ 32) := by
  have hdom : ∀ b : Usize, b.val < ((256#usize).val + 255) / 256 → tq1_0BlockDomain tq1_0WitRaw b := by
    intro b hb
    have hb1 : b.val < 1 := by
      have hR : (((256#usize).val + 255) / 256) = 1 := by decide
      rwa [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact tq1_0BlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_tq1_0_row_spec tq1_0WitRaw 256#usize (alloc.vec.Vec.new I64) hdom (by
      rw [tq1_0_vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = tq1_0RowValues tq1_0WitRaw 256 := by
    rw [tq1_0_vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
  rw [tq1_0_dequantVals_getD] at h0
  rw [h0, tq1_0RowValues_head, tq1_0Wit_weight]


end

end apeiron_word
