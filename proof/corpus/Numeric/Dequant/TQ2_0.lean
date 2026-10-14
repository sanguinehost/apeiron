/-
Refinement of the extracted `TQ2_0` dequantizer (`APEIRON-TERNARY-W3-T3`).

The Rust bodies are `dequant::lowbit::tq2_0_block` and `dequant::lowbit::dequant_tq2_0_row` in
`src/engine/dequant_lowbit.rs`. The decode is exact integer arithmetic over Q32.32: the block's `d`
at byte 64 of the 66-byte block is a finite `f16` decoded exactly by `f16_to_fixed_exact`, and each
weight is the checked `i64` product `(code − 1)·d` of the scale and one of the four-level codes.

The block width is 256: four levels per code byte, two 32-byte `qs` groups, four planes and 32
lanes, so weight `i` (of `n`) uses block `i/256` and position `i%256`, with out-position `p` decoded
as group `(p/128)·32`, plane `(p%128)/32` and lane `p%32`:

  `tq2_0Weight raw i = (code − 1)·d`, `code = (qs[(p/128)·32 + p%32] >> 2·((p%128)/32)) & 3`.

The block-level reader is `tq2_0_block` (the 64-byte `qs` fill plus `read_scale`); the row nest is
lane → plane → group, assembled with `loop.spec_decr_nat` against the emission invariant
`tq2_0EmitInv` (`out = start ++ (blockVals).take (min pos take)`).

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.Dequant.Q8_0

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-- Destructuring a successful 4-tuple bind reduces to the direct application. -/
theorem tq2_0_let_quad {α β γ δ ε : Type} (x : α) (y : β) (z : γ) (w : δ)
    (f : α → β → γ → δ → Result ε) :
    (do let (a, b, c, d) ← (ok (x, y, z, w) : Result (α × β × γ × δ)); f a b c d)
      = f x y z w := by
  simp only [bind_tc_ok]
  rfl

/-! ## TQ2_0 references -/

/-- The exact Q32.32 scale `d` of block `b`, at byte `b·66 + 64`. -/
def tq2_0ScaleRef (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (b * 66 + 64))

/-- The four-level code of out-position `p` of block `b`. -/
def tq2_0CodeRef (raw : Slice U8) (b p : Nat) : Nat :=
  ((raw.val.getD (b * 66 + ((p / 128) * 32 + p % 32)) 0#u8).val
    >>> (2 * ((p % 128) / 32))) &&& 3

/-- The exact weight `i` of a TQ2_0 row: `(code − 1)·d` at block `i/256`, position `i%256`. -/
def tq2_0Weight (raw : Slice U8) (i : Nat) : ℤ :=
  ((tq2_0CodeRef raw (i / 256) (i % 256) : ℤ) - 1) * tq2_0ScaleRef raw (i / 256)

/-- The first `n` exact Q32.32 integers of a TQ2_0 row. -/
def tq2_0RowValues (raw : Slice U8) (n : Nat) : List ℤ :=
  (List.range n).map (tq2_0Weight raw)

/-- The 256 exact weights of block `b`, indexed by out-position. -/
def tq2_0BlockVals (raw : Slice U8) (b : Nat) : List ℤ :=
  (List.range 256).map (fun p => tq2_0Weight raw (256 * b + p))

/-- The four-level code at out-position `p` over a `qs` byte image. -/
def tq2_0PosCode (q : List U8) (p : Nat) : Nat :=
  ((q.getD ((p / 128) * 32 + p % 32) 0#u8).val >>> (2 * ((p % 128) / 32))) &&& 3

/-- The code at `(group, plane, lane)` of a `qs` byte image. -/
def tq2_0LaneCode (q : List U8) (group plane lane : Nat) : Nat :=
  ((q.getD (group + lane) 0#u8).val >>> (2 * plane)) &&& 3

/-- The exact weight at out-position `p` of a block with scale `scale` and byte image `q`. -/
def tq2_0PosVal (scale : ℤ) (q : List U8) (p : Nat) : ℤ :=
  ((tq2_0PosCode q p : ℤ) - 1) * scale

/-! ## Block values and position arithmetic -/

lemma tq2_0BlockVals_length (raw : Slice U8) (b : Nat) :
    (tq2_0BlockVals raw b).length = 256 := by
  simp [tq2_0BlockVals]

lemma tq2_0BlockVals_getD (raw : Slice U8) (b p : Nat) (hp : p < 256) :
    (tq2_0BlockVals raw b).getD p 0 = tq2_0Weight raw (256 * b + p) := by
  unfold tq2_0BlockVals
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 256)[p]? = some p := by
    rw [List.getElem?_eq_getElem (by simpa using hp)]; simp
  rw [h]; rfl

/-- Appending the next `take` raw weights advances the TQ2_0 row prefix by `take`. -/
lemma tq2_0RowValues_append (raw : Slice U8) (m take : Nat) :
    tq2_0RowValues raw m ++ (List.range take).map (fun j => tq2_0Weight raw (m + j)) =
      tq2_0RowValues raw (m + take) := by
  unfold tq2_0RowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-- The `(p/128)·32 + p%32` index and `(p%128)/32` plane of `p = 4·group + 32·plane + lane`. -/
lemma tq2_0_pos_index (group plane lane : Nat) (hg : group = 0 ∨ group = 32)
    (hp : plane < 4) (hl : lane < 32) :
    ((4 * group + 32 * plane + lane) / 128) * 32 + (4 * group + 32 * plane + lane) % 32
        = group + lane ∧
      ((4 * group + 32 * plane + lane) % 128) / 32 = plane ∧
      4 * group + 32 * plane + lane < 256 := by
  rcases hg with rfl | rfl <;> omega

/-- `tq2_0PosVal` at `4·group + 32·plane + lane` is the lane code times the scale. -/
lemma tq2_0PosVal_lane (scale : ℤ) (q : List U8) (group plane lane : Nat)
    (hg : group = 0 ∨ group = 32) (hp : plane < 4) (hl : lane < 32) :
    tq2_0PosVal scale q (4 * group + 32 * plane + lane) =
      ((tq2_0LaneCode q group plane lane : ℤ) - 1) * scale := by
  obtain ⟨h1, h2, -⟩ := tq2_0_pos_index group plane lane hg hp hl
  unfold tq2_0PosVal tq2_0PosCode tq2_0LaneCode
  rw [h1, h2]

/-- The `qs` byte image of a whole block reads the block's bytes. -/
lemma tq2_0PosCode_eq_codeRef (raw : Slice U8) (b : Nat) (q : List U8) (p : Nat)
    (hq : ∀ k, k < 64 → (q.getD k 0#u8).val = (raw.val.getD (b * 66 + k) 0#u8).val)
    (hp : p < 256) :
    tq2_0PosCode q p = tq2_0CodeRef raw b p := by
  unfold tq2_0PosCode tq2_0CodeRef
  have hidx : (p / 128) * 32 + p % 32 < 64 := by omega
  rw [hq _ hidx]

/-- `tq2_0PosVal` of the block byte image is the block weight at out-position `p`. -/
lemma tq2_0PosVal_weight (raw : Slice U8) (b : Nat) (q : List U8) (scale : ℤ) (p : Nat)
    (hq : ∀ k, k < 64 → (q.getD k 0#u8).val = (raw.val.getD (b * 66 + k) 0#u8).val)
    (hscale : scale = tq2_0ScaleRef raw b) (hp : p < 256) :
    tq2_0PosVal scale q p = tq2_0Weight raw (256 * b + p) := by
  unfold tq2_0PosVal tq2_0Weight
  rw [tq2_0PosCode_eq_codeRef raw b q p hq hp,
    show (256 * b + p) / 256 = b by omega,
    show (256 * b + p) % 256 = p by omega, hscale]

/-! ## `quant::TQ2_0_BLOCK_BYTES` and the block reader -/

theorem quant_TQ2_0_BLOCK_BYTES_eq : quant.TQ2_0_BLOCK_BYTES = ok (66#usize) := by
  have hQK : quant.TQ2_0_QK = 256#usize := by with_unfolding_all rfl
  have h : quant.TQ2_0_BLOCK_BYTES ⦃ z => z = (66#usize) ⦄ := by
    unfold quant.TQ2_0_BLOCK_BYTES
    rw [hQK]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- After `k` steps the `qs` image holds the block's first `k` bytes and zeros elsewhere. -/
def tq2_0QsInv (blk : Slice U8) (qs : Array U8 64#usize) (k : Nat) : Prop :=
  k ≤ 64 ∧ ∀ i, i < 64 → (qs.val.getD i 0#u8).val =
    if i < k then (blk.val.getD i 0#u8).val else 0

/-- One `tq2_0_block` copy-loop body step: read `blk[k]`, write entry `k`. -/
theorem tq2_0Qs_body_step (blk : Slice U8) (qs : Array U8 64#usize) (k : Usize)
    (hblk : 64 ≤ blk.val.length) (hk : k.val < 64) (hinv : tq2_0QsInv blk qs k.val) :
    ∃ (qs2 : Array U8 64#usize) (k2 : Usize),
      dequant.lowbit.tq2_0_block_loop.body blk qs k = ok (ControlFlow.cont (qs2, k2)) ∧
      tq2_0QsInv blk qs2 (k.val + 1) ∧ k2.val = k.val + 1 := by
  obtain ⟨i, hidx, hiv⟩ := WP.spec_imp_exists
    (Slice.index_usize_spec blk k (by show k.val < blk.val.length; omega))
  have hival : i.val = (blk.val.getD k.val 0#u8).val := by
    rw [hiv]
    exact congrArg (fun x : U8 => x.val)
      (getD_of_lt (l := blk.val) (by omega)).symm
  obtain ⟨qs2, hupd, hqsv⟩ := WP.spec_imp_exists
    (Array.update_spec qs k i (by show k.val < qs.val.length; rw [Array.property qs]; exact hk))
  obtain ⟨k2, hadd, hk2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by scalar_tac))
  have hk2 : k2.val = k.val + 1 := by rw [hk2v]; simp
  refine ⟨qs2, k2, ?_, ?_, hk2⟩
  · rw [dequant.lowbit.tq2_0_block_loop.body]
    rw [if_pos (by scalar_tac : k < 64#usize)]
    simp only [hidx, hupd, hadd, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    intro i2 hi2
    have hqprop : qs.val.length = 64 := Array.property qs
    rw [hqsv, Array.set_val_eq, getD_set]
    by_cases hik : i2 = k.val
    · subst hik
      rw [if_pos ⟨rfl, by rw [hqprop]; exact hk⟩, if_pos (by omega)]
      rw [hival]
    · rw [if_neg (fun h => hik h.1), hinv.2 i2 hi2]
      by_cases hlt : i2 < k.val
      · rw [if_pos hlt, if_pos (by omega)]
      · rw [if_neg hlt, if_neg (by omega)]

/-- The `qs` fill loop: `loop.spec_decr_nat` with `tq2_0QsInv` and measure `64 - k`. -/
theorem tq2_0Qs_loop_spec (blk : Slice U8) (qs0 : Array U8 64#usize)
    (hblk : 64 ≤ blk.val.length) (hinv0 : tq2_0QsInv blk qs0 0) :
    WP.spec (dequant.lowbit.tq2_0_block_loop blk qs0 0#usize)
      (fun qs => tq2_0QsInv blk qs 64) := by
  unfold dequant.lowbit.tq2_0_block_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array U8 64#usize × Usize => 64 - s.2.val)
    (inv := fun s => tq2_0QsInv blk s.1 s.2.val)
  · rintro ⟨qs, k⟩ hinv'
    dsimp only at hinv' ⊢
    by_cases hlt : k.val < 64
    · obtain ⟨qs2, k2, hbody, hinv2, hk2⟩ := tq2_0Qs_body_step blk qs k hblk hlt hinv'
      simp only [hbody]
      refine ⟨by simpa [hk2] using hinv2, ?_⟩
      show 64 - k2.val < 64 - k.val
      omega
    · have hnot : ¬ (k < 64#usize) := fun hc => hlt ((UScalar.lt_equiv k 64#usize).1 hc)
      have hbody : dequant.lowbit.tq2_0_block_loop.body blk qs k = ok (ControlFlow.done qs) := by
        rw [dequant.lowbit.tq2_0_block_loop.body, if_neg hnot]
      simp only [hbody]
      have hk64 : k.val = 64 := by have := hinv'.1; omega
      simpa [hk64] using hinv'
  · simpa using hinv0

/-- The slice `getD` at an in-window index is the source list's `getD`. -/
lemma tq2_0_slice_getD {α : Type} (l : List α) (a b k : Nat) (d : α) (hk : k < b - a) :
    (l.slice a b).getD k d = l.getD (a + k) d := by
  unfold List.slice
  exact getD_drop_take l a (b - a) k d hk

/-- `(c : ℤ) - 1` fits `i64` for a four-level code `c < 4`. -/
lemma tq2_0_sub_one_bounds (c : Nat) (hc : c < 4) :
    IScalar.min IScalarTy.I64 ≤ (c : ℤ) - 1 ∧ (c : ℤ) - 1 ≤ IScalar.max IScalarTy.I64 := by
  have h3 : (c : ℤ) ≤ 3 := by exact_mod_cast (by omega : c ≤ 3)
  constructor
  · simp only [IScalar.min_IScalarTy_I64_eq, I64.min_eq]; omega
  · simp only [IScalar.max_IScalarTy_I64_eq, I64.max_eq]; omega

/-- **`TQ2_0` block law.** On a whole 66-byte block with finite `d`, `tq2_0_block` returns the
exact Q32.32 scale and the 64 `qs` bytes. -/
theorem tq2_0_block_ok (raw : Slice U8) (block : Usize)
    (hlen : block.val * 66 + 66 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw (block.val * 66 + 64)) ≠ 31) :
    ∃ (d : I64) (qs : Array U8 64#usize),
      dequant.lowbit.tq2_0_block raw block = ok (.Ok (d, qs)) ∧
      d.val = tq2_0ScaleRef raw block.val ∧
      (∀ k, k < 64 → (qs.val.getD k 0#u8).val =
        (raw.val.getD (block.val * 66 + k) 0#u8).val) := by
  obtain ⟨blk, hbs, hblkv⟩ := block_slice_ok raw block 66#usize (by simpa using hlen)
  have h66 : (66#usize).val = 66 := rfl
  have hblkv' : blk.val = raw.val.slice (block.val * 66) (block.val * 66 + 66) := by
    rw [hblkv, h66]
  have hlenblk : blk.val.length = 66 := by
    rw [hblkv', List.slice, List.length_take, List.length_drop]
    omega
  have hget : ∀ k, k < 66 → blk.val.getD k 0#u8 = raw.val.getD (block.val * 66 + k) 0#u8 := by
    intro k hk
    rw [hblkv', List.slice, show block.val * 66 + 66 - block.val * 66 = 66 by omega]
    exact getD_drop_take raw.val (block.val * 66) 66 k 0#u8 (by simpa using hk)
  have hf16 : f16LeHalf blk 64 = f16LeHalf raw (block.val * 66 + 64) := by
    unfold f16LeHalf
    rw [hget 64 (by norm_num)]
    rw [show (64 : Nat) + 1 = 65 by norm_num, hget 65 (by norm_num)]
  have hf : f16Exp (f16LeHalf blk 64) ≠ 31 := by rw [hf16]; exact hd
  obtain ⟨d, hscale_eq, hdv⟩ := read_scale_ok blk 64#usize (by rw [hlenblk]; decide) hf
  have hdv' : d.val = tq2_0ScaleRef raw block.val := by
    rw [hdv, show (64#usize).val = 64 from rfl, hf16]
    rfl
  have hqsinv0 : tq2_0QsInv blk (Array.repeat 64#usize 0#u8) 0 := by
    refine ⟨by simp, ?_⟩
    intro i hi
    rw [Array.repeat_val]
    simp only [List.getD, List.getElem?_replicate]
    rw [if_pos (by simpa using hi), if_neg (by omega)]
    rfl
  obtain ⟨qs, hloop, hqsinv⟩ := WP.spec_imp_exists
    (tq2_0Qs_loop_spec blk (Array.repeat 64#usize 0#u8) (by rw [hlenblk]; decide) hqsinv0)
  refine ⟨d, qs, ?_, hdv', ?_⟩
  · rw [dequant.lowbit.tq2_0_block, quant_TQ2_0_BLOCK_BYTES_eq]
    simp only [hbs, hscale_eq, hloop, bind_tc_ok,
      core.result.Result.Insts.CoreOpsTry.branch]
  · intro k hk
    have h1 : (qs.val.getD k 0#u8).val = (blk.val.getD k 0#u8).val := by
      have h := hqsinv.2 k hk
      rw [if_pos hk] at h
      exact h
    rw [h1, hget k (by omega)]

/-! ## Emission invariants -/

/-- `dequantVals` of an append. -/
lemma tq2_0_dequantVals_append (l m : List I64) :
    dequantVals (l ++ m) = dequantVals l ++ dequantVals m := by
  unfold dequantVals; rw [List.map_append]

/-- `dequantVals` of a singleton. -/
lemma tq2_0_dequantVals_singleton (v : I64) : dequantVals [v] = [v.val] := by
  simp [dequantVals]

/-- Emission invariant: the output is `start` followed by the first `min pos take` block values. -/
def tq2_0EmitInv (raw : Slice U8) (b : Nat) (take : Usize) (start out : alloc.vec.Vec I64)
    (pos : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧
  dequantVals out.val = dequantVals start.val ++
    (tq2_0BlockVals raw b).take (Nat.min pos.val take.val)

/-- Lane-loop invariant: `tq2_0EmitInv` plus `pos = pos0 + lane`. -/
def tq2_0LaneInv (raw : Slice U8) (b : Nat) (take : Usize) (pos0 : Nat)
    (start out : alloc.vec.Vec I64) (pos lane : Usize) (ok1 : Bool)
    (err : Option error.EngineError) : Prop :=
  tq2_0EmitInv raw b take start out pos ok1 err ∧ pos.val = pos0 + lane.val ∧ lane.val ≤ 32

/-- **Innermost lane body step.** Emit `blockVals[pos]`, push while `pos < take`, advance. -/
theorem tq2_0Lane_body_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 64#usize) (take : Usize) (group plane : Usize) (pos0 : Nat)
    (start out : alloc.vec.Vec I64) (pos lane : Usize)
    (hgroup : group.val = 0 ∨ group.val = 32)
    (hplane4 : plane.val < 4)
    (hlane : lane.val < 32)
    (hpos256 : pos.val < 256)
    (hval : ((tq2_0LaneCode qs.val group.val plane.val lane.val : ℤ) - 1) * scale.val =
      (tq2_0BlockVals raw b).getD pos.val 0)
    (hmul : I64.min ≤ ((tq2_0LaneCode qs.val group.val plane.val lane.val : ℤ) - 1) * scale.val ∧
      ((tq2_0LaneCode qs.val group.val plane.val lane.val : ℤ) - 1) * scale.val ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : tq2_0LaneInv raw b take pos0 start out pos lane true none) :
    ∃ (out2 : alloc.vec.Vec I64) (pos2 lane2 : Usize),
      dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0_loop0.body scale qs take group plane
          out true none pos lane
        = ok (ControlFlow.cont (out2, true, none, pos2, lane2)) ∧
      tq2_0LaneInv raw b take pos0 start out2 pos2 lane2 true none ∧
      pos2.val = pos.val + 1 ∧ lane2.val = lane.val + 1 := by
  obtain ⟨hemit, hposrel, hlanele⟩ := hinv
  obtain ⟨hok, herr, hrec⟩ := hemit
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := group) (y := lane) (by scalar_tac))
  have hbnd : i.val < qs.val.length := by
    rw [hiv, Array.property qs, show (64#usize).val = 64 from rfl]
    rcases hgroup with h | h <;> omega
  obtain ⟨i1, hi1, hi1eq⟩ := WP.spec_imp_exists (Array.index_usize_spec qs i hbnd)
  have hi1val : i1.val = (qs.val.getD (group.val + lane.val) 0#u8).val := by
    have h' : i1.val = (qs.val.getD i.val 0#u8).val := by
      rw [hi1eq]
      exact congrArg (fun x : U8 => x.val) (getD_of_lt (l := qs.val) hbnd).symm
    rw [h', hiv]
  obtain ⟨i2, hmul2, hi2v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 2#usize) (y := plane) (by scalar_tac))
  have hi2 : i2.val = 2 * plane.val := by rw [hi2v]; simp
  obtain ⟨i3, hshr, hi3v, -⟩ := WP.spec_imp_exists
    (UScalar.ShiftRight_spec (x := i1) (y := i2)
      (by rw [hi2]; simp only [UScalarTy.U8_numBits_eq]; omega))
  have hcodeval : (i3 &&& 3#u8).val = tq2_0LaneCode qs.val group.val plane.val lane.val := by
    rw [UScalar.val_and, show (3#u8).val = 3 from rfl, hi3v, hi2, hi1val]
    rfl
  let i4 : I64 := UScalar.hcast .I64 (i3 &&& 3#u8)
  have hcast : I64.Insts.CoreConvertFromU8.from (i3 &&& 3#u8) = ok i4 := rfl
  have hi4val : i4.val = (tq2_0LaneCode qs.val group.val plane.val lane.val : ℤ) := by
    dsimp only [i4]
    rw [hcast_u8_val, hcodeval]
  have hcode_lt : tq2_0LaneCode qs.val group.val plane.val lane.val < 4 :=
    Nat.and_lt_two_pow _ (by norm_num : (3 : Nat) < 2 ^ 2)
  obtain ⟨i5, hsub, hi5v⟩ := WP.spec_imp_exists
    (IScalar.sub_spec (x := i4) (y := 1#i64)
      (by rw [hi4val, show (1#i64).val = 1 from rfl]
          exact (tq2_0_sub_one_bounds _ hcode_lt).1)
      (by rw [hi4val, show (1#i64).val = 1 from rfl]
          exact (tq2_0_sub_one_bounds _ hcode_lt).2))
  have hi5val : i5.val = (tq2_0LaneCode qs.val group.val plane.val lane.val : ℤ) - 1 := by
    rw [hi5v, hi4val, show (1#i64).val = 1 from rfl]
  obtain ⟨v, hmulv, hvv⟩ := dequant_mul_ok scale i5
    (by rw [hi5val, mul_comm]; exact hmul.1)
    (by rw [hi5val, mul_comm]; exact hmul.2)
  have hvval : v.val =
      (tq2_0BlockVals raw b)[pos.val]'(by rw [tq2_0BlockVals_length]; exact hpos256) := by
    rw [hvv, hi5val, mul_comm, hval]
    rw [list_getD_getElem (tq2_0BlockVals raw b) pos.val
      (by rw [tq2_0BlockVals_length]; exact hpos256) (0 : ℤ)]
  by_cases hlt : pos.val < take.val
  · obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out v hcap)
    obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
    refine ⟨out2, pos2, lane2, ?_, ?_, hpos2, hlane2⟩
    · unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane), hok, if_pos rfl]
      simp only [hi, hi1, hmul2, hshr, lift, bind_tc_ok, hcast, hsub, hmulv]
      rw [if_pos ((UScalar.lt_equiv pos take).2 hlt)]
      simp only [hpush, haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, ?_, by rw [hlane2]; omega⟩
      · have hmin1 : Nat.min (pos.val + 1) take.val = pos.val + 1 :=
          Nat.min_eq_left (by omega)
        have hmin2 : Nat.min pos.val take.val = pos.val := Nat.min_eq_left (le_of_lt hlt)
        rw [hpushv, tq2_0_dequantVals_append, tq2_0_dequantVals_singleton, hrec, hvval, hpos2,
          hmin2, hmin1, List.take_succ_eq_append_getElem
            (by rw [tq2_0BlockVals_length]; exact hpos256), List.append_assoc]
      · rw [hpos2, hlane2, hposrel]; omega
  · obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
    refine ⟨out, pos2, lane2, ?_, ?_, hpos2, hlane2⟩
    · unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane), hok, if_pos rfl]
      simp only [hi, hi1, hmul2, hshr, lift, bind_tc_ok, hcast, hsub, hmulv]
      rw [if_neg (fun hc => hlt ((UScalar.lt_equiv pos take).1 hc))]
      simp only [haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, ?_, by rw [hlane2]; omega⟩
      · have h1 : Nat.min (pos.val + 1) take.val = take.val := Nat.min_eq_right (by omega)
        have h2 : Nat.min pos.val take.val = take.val := Nat.min_eq_right (by omega)
        rw [hrec, hpos2, h1, h2]
      · rw [hpos2, hlane2, hposrel]; omega

/-- **Innermost lane loop.** From `lane = 0`, `pos = pos0` the loop emits 32 staged values
(truncated by `take`) and leaves `pos = pos0 + 32`. -/
theorem tq2_0Lane_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 64#usize) (take : Usize) (group plane : Usize) (pos0 : Nat)
    (start out0 : alloc.vec.Vec I64) (pos : Usize)
    (hgroup : group.val = 0 ∨ group.val = 32)
    (hplane4 : plane.val < 4)
    (hval : ∀ lane, lane < 32 →
      ((tq2_0LaneCode qs.val group.val plane.val lane : ℤ) - 1) * scale.val =
        (tq2_0BlockVals raw b).getD (pos0 + lane) 0)
    (hmul : ∀ lane, lane < 32 →
      I64.min ≤ ((tq2_0LaneCode qs.val group.val plane.val lane : ℤ) - 1) * scale.val ∧
      ((tq2_0LaneCode qs.val group.val plane.val lane : ℤ) - 1) * scale.val ≤ I64.max)
    (hpos256 : pos0 + 32 ≤ 256)
    (hcap : start.val.length + take.val < Usize.max)
    (hpos : pos.val = pos0)
    (hinv0 : tq2_0EmitInv raw b take start out0 pos true none) :
    WP.spec (dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0_loop0 out0 true none scale qs take
        pos group plane 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pe : Usize),
        p = (out, true, none, pe) ∧ pe.val = pos0 + 32 ∧
        tq2_0EmitInv raw b take start out pe true none) := by
  have hinit : tq2_0LaneInv raw b take pos0 start out0 pos 0#usize true none :=
    ⟨hinv0, by rw [hpos]; simp, by simp⟩
  unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      32 - s.2.2.2.2.val)
    (inv := fun s => tq2_0LaneInv raw b take pos0 start s.1 s.2.2.2.1 s.2.2.2.2 s.2.1 s.2.2.1)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pe : Usize),
      p = (out, true, none, pe) ∧ pe.val = pos0 + 32 ∧
      tq2_0EmitInv raw b take start out pe true none)
  · rintro ⟨out, ok1, err, posX, lane⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hemit, hposrel, hlanele⟩ := hinv'
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : lane.val < 32
    · have hpos256' : posX.val < 256 := by
        have : posX.val = pos0 + lane.val := hposrel
        omega
      have hcap' : out.val.length < Usize.max := by
        have hlen : out.val.length ≤ start.val.length + take.val := by
          have h := congrArg List.length hrec
          simp only [dequantVals, List.length_append, List.length_map, List.length_take,
            tq2_0BlockVals_length] at h
          have h1 : Nat.min posX.val take.val ≤ take.val := Nat.min_le_right _ _
          omega
        omega
      obtain ⟨out2, pos2, lane2, hbody, hinv2, hpos2, hlane2⟩ :=
        tq2_0Lane_body_step raw b scale qs take group plane pos0 start out posX lane hgroup
          hplane4 hlt hpos256'
          (by rw [hposrel]; exact hval lane hlt)
          (hmul lane hlt) hcap' ⟨⟨rfl, rfl, hrec⟩, hposrel, hlanele⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show 32 - lane2.val < 32 - lane.val
      omega
    · have hnot : ¬ (lane < 32#usize) := fun hc => hlt ((UScalar.lt_equiv lane 32#usize).1 hc)
      have hbody : dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0_loop0.body scale qs take
          group plane out true none posX lane = ok (ControlFlow.done (out, true, none, posX)) := by
        unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hl32 : lane.val = 32 := by omega
      refine ⟨out, posX, rfl, by rw [hposrel, hl32], ?_⟩
      exact ⟨rfl, rfl, hrec⟩
  · exact hinit

/-! ## Plane loop -/

/-- Plane-loop invariant: `plane` planes complete and `pos` positions emitted. -/
def tq2_0PlaneInv (raw : Slice U8) (b : Nat) (take : Usize) (start out : alloc.vec.Vec I64)
    (gpos : Nat) (pos plane : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ plane.val ≤ 4 ∧ pos.val = gpos + 32 * plane.val ∧
  tq2_0EmitInv raw b take start out pos ok1 err

/-- One plane-loop body step: run the 32-lane loop, advance the plane. -/
theorem tq2_0Plane_body_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 64#usize) (take : Usize) (start : alloc.vec.Vec I64)
    (group pos plane : Usize) (out : alloc.vec.Vec I64)
    (hgroup : group.val = 0 ∨ group.val = 32)
    (hbridge : ∀ p, p < 256 →
      tq2_0PosVal scale.val qs.val p = (tq2_0BlockVals raw b).getD p 0 ∧
      I64.min ≤ tq2_0PosVal scale.val qs.val p ∧ tq2_0PosVal scale.val qs.val p ≤ I64.max)
    (hcap : start.val.length + take.val < Usize.max)
    (hplane : plane.val < 4)
    (hinv : tq2_0PlaneInv raw b take start out (4 * group.val) pos plane true none) :
    ∃ (out1 : alloc.vec.Vec I64) (pos1 plane1 : Usize),
      dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0.body scale qs take group out true none pos plane
        = ok (ControlFlow.cont (out1, true, none, pos1, plane1)) ∧
      plane1.val = plane.val + 1 ∧
      tq2_0PlaneInv raw b take start out1 (4 * group.val) pos1 plane1 true none ∧
      pos1.val = pos.val + 32 := by
  obtain ⟨hok, herr, hplanele, hposg, hemit⟩ := hinv
  have hVall : ∀ lane, lane < 32 →
      ((tq2_0LaneCode qs.val group.val plane.val lane : ℤ) - 1) * scale.val =
        (tq2_0BlockVals raw b).getD (4 * group.val + 32 * plane.val + lane) 0 := by
    intro lane hl
    obtain ⟨h1, h2, h3⟩ := tq2_0_pos_index group.val plane.val lane hgroup hplane hl
    have hpl := tq2_0PosVal_lane scale.val qs.val group.val plane.val lane hgroup hplane hl
    rw [← hpl]
    exact (hbridge _ h3).1
  have hmulall : ∀ lane, lane < 32 →
      I64.min ≤ ((tq2_0LaneCode qs.val group.val plane.val lane : ℤ) - 1) * scale.val ∧
      ((tq2_0LaneCode qs.val group.val plane.val lane : ℤ) - 1) * scale.val ≤ I64.max := by
    intro lane hl
    obtain ⟨h1, h2, h3⟩ := tq2_0_pos_index group.val plane.val lane hgroup hplane hl
    have hpl := tq2_0PosVal_lane scale.val qs.val group.val plane.val lane hgroup hplane hl
    rw [← hpl]
    exact (hbridge _ h3).2
  have hpos256 : 4 * group.val + 32 * plane.val + 32 ≤ 256 := by
    rcases hgroup with h | h <;> omega
  obtain ⟨r, hlane, hpost⟩ := WP.spec_imp_exists
    (tq2_0Lane_loop_spec raw b scale qs take group plane (4 * group.val + 32 * plane.val) start out
      pos hgroup hplane hVall hmulall hpos256 hcap hposg hemit)
  obtain ⟨out1, pe1, hr, hpe1, hemit1⟩ := hpost
  subst hr
  obtain ⟨plane1, hplane1a, hplane1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := plane) (y := 1#usize) (by scalar_tac))
  have hplane1 : plane1.val = plane.val + 1 := by rw [hplane1v]; simp
  have hpos1 : pe1.val = pos.val + 32 := by rw [hpe1, hposg]
  refine ⟨out1, pe1, plane1, ?_, hplane1, ?_, hpos1⟩
  · unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hplane), hok, if_pos rfl]
    rw [hlane, tq2_0_let_quad, hplane1a]
    simp only [bind_tc_ok]
  · refine ⟨rfl, rfl, by omega, ?_, hemit1⟩
    rw [hpos1, hplane1, hposg]
    ring

/-- The plane loop: `loop.spec_decr_nat` with `tq2_0PlaneInv` and measure `4 - plane`. -/
theorem tq2_0Plane_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 64#usize) (take : Usize) (start : alloc.vec.Vec I64)
    (group pos : Usize) (out0 : alloc.vec.Vec I64)
    (hgroup : group.val = 0 ∨ group.val = 32)
    (hbridge : ∀ p, p < 256 →
      tq2_0PosVal scale.val qs.val p = (tq2_0BlockVals raw b).getD p 0 ∧
      I64.min ≤ tq2_0PosVal scale.val qs.val p ∧ tq2_0PosVal scale.val qs.val p ≤ I64.max)
    (hcap : start.val.length + take.val < Usize.max)
    (hpos : pos.val = 4 * group.val)
    (hinv0 : tq2_0EmitInv raw b take start out0 pos true none) :
    WP.spec (dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0 out0 true none scale qs take pos
        group 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pe : Usize),
        p = (out, true, none, pe) ∧ pe.val = 4 * group.val + 128 ∧
        tq2_0EmitInv raw b take start out pe true none) := by
  have hinit : tq2_0PlaneInv raw b take start out0 (4 * group.val) pos 0#usize true none :=
    ⟨rfl, rfl, by simp, by rw [hpos]; simp, hinv0⟩
  unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      4 - s.2.2.2.2.val)
    (inv := fun s => tq2_0PlaneInv raw b take start s.1 (4 * group.val) s.2.2.2.1
      s.2.2.2.2 s.2.1 s.2.2.1)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pe : Usize),
      p = (out, true, none, pe) ∧ pe.val = 4 * group.val + 128 ∧
      tq2_0EmitInv raw b take start out pe true none)
  · rintro ⟨out, ok1, err, posX, planeX⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hok, herr, hplanele, hposX, hemit⟩ := hinv'
    rw [hok, herr]
    by_cases hlt : planeX.val < 4
    · obtain ⟨out1, pos1, plane1, hbody, hplane1, hinv2, hpos1⟩ :=
        tq2_0Plane_body_step raw b scale qs take start group posX planeX out hgroup hbridge
          hcap hlt ⟨rfl, rfl, hplanele, hposX, ⟨rfl, rfl, hemit.2.2⟩⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show 4 - plane1.val < 4 - planeX.val
      omega
    · have hnot : ¬ (planeX < 4#usize) := fun hc => hlt ((UScalar.lt_equiv planeX 4#usize).1 hc)
      have hbody : dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0.body scale qs take group out
          true none posX planeX = ok (ControlFlow.done (out, true, none, posX)) := by
        unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hp4 : planeX.val = 4 := by omega
      refine ⟨out, posX, rfl, by omega, ?_⟩
      exact ⟨rfl, rfl, hemit.2.2⟩
  · exact hinit

/-! ## Group loop -/

/-- Group-loop invariant: `group` groups complete and `pos` positions emitted. -/
def tq2_0GroupInv (raw : Slice U8) (b : Nat) (take : Usize) (start out : alloc.vec.Vec I64)
    (pos group : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ group.val ≤ 64 ∧
  (group.val = 0 ∨ group.val = 32 ∨ group.val = 64) ∧ pos.val = 4 * group.val ∧
  tq2_0EmitInv raw b take start out pos ok1 err

/-- One group-loop body step: run the plane nest, advance the group by 32. -/
theorem tq2_0Group_body_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 64#usize) (take : Usize) (start : alloc.vec.Vec I64)
    (pos group : Usize) (out : alloc.vec.Vec I64)
    (hbridge : ∀ p, p < 256 →
      tq2_0PosVal scale.val qs.val p = (tq2_0BlockVals raw b).getD p 0 ∧
      I64.min ≤ tq2_0PosVal scale.val qs.val p ∧ tq2_0PosVal scale.val qs.val p ≤ I64.max)
    (hcap : start.val.length + take.val < Usize.max)
    (hlt : group.val < 64)
    (hinv : tq2_0GroupInv raw b take start out pos group true none) :
    ∃ (out1 : alloc.vec.Vec I64) (pos1 group1 : Usize),
      dequant.lowbit.dequant_tq2_0_row_loop0_loop0.body scale qs take out true none pos group
        = ok (ControlFlow.cont (out1, true, none, pos1, group1)) ∧
      group1.val = group.val + 32 ∧
      tq2_0GroupInv raw b take start out1 pos1 group1 true none ∧
      pos1.val = pos.val + 128 := by
  obtain ⟨hok, herr, hgle, hgdisj, hposg, hemit⟩ := hinv
  have hgor : group.val = 0 ∨ group.val = 32 := by
    rcases hgdisj with h | h | h
    · exact Or.inl h
    · exact Or.inr h
    · omega
  have hpos0 : pos.val = 4 * group.val := hposg
  obtain ⟨r, hplane, hpost⟩ := WP.spec_imp_exists
    (tq2_0Plane_loop_spec raw b scale qs take start group pos out hgor hbridge hcap hpos0 hemit)
  obtain ⟨out1, pe1, hr, hpe1, hemit1⟩ := hpost
  subst hr
  obtain ⟨group1, hgroup1a, hgroup1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := group) (y := 32#usize) (by scalar_tac))
  have hgroup1 : group1.val = group.val + 32 := by rw [hgroup1v]; simp
  have hpos1 : pe1.val = pos.val + 128 := by rw [hpe1, hpos0]
  refine ⟨out1, pe1, group1, ?_, hgroup1, ?_, hpos1⟩
  · unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), hok, if_pos rfl]
    rw [hplane, tq2_0_let_quad, hgroup1a]
    simp only [bind_tc_ok]
  · refine ⟨rfl, rfl, by omega, ?_, ?_, hemit1⟩
    · rw [hgroup1]; omega
    · rw [hpos1, hgroup1, hpos0]; ring

/-- The group loop: `loop.spec_decr_nat` with `tq2_0GroupInv` and measure `64 - group`. -/
theorem tq2_0Group_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 64#usize) (take : Usize) (start out0 : alloc.vec.Vec I64)
    (hbridge : ∀ p, p < 256 →
      tq2_0PosVal scale.val qs.val p = (tq2_0BlockVals raw b).getD p 0 ∧
      I64.min ≤ tq2_0PosVal scale.val qs.val p ∧ tq2_0PosVal scale.val qs.val p ≤ I64.max)
    (hcap : start.val.length + take.val < Usize.max)
    (hinv0 : tq2_0EmitInv raw b take start out0 0#usize true none) :
    WP.spec (dequant.lowbit.dequant_tq2_0_row_loop0_loop0 out0 true none scale qs take 0#usize
        0#usize)
      (fun p => ∃ out : alloc.vec.Vec I64, p = (out, true, none) ∧
        tq2_0EmitInv raw b take start out 256#usize true none) := by
  have hinit : tq2_0GroupInv raw b take start out0 0#usize 0#usize true none :=
    ⟨rfl, rfl, by simp, by simp, by simp, hinv0⟩
  unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      64 - s.2.2.2.2.val)
    (inv := fun s => tq2_0GroupInv raw b take start s.1 s.2.2.2.1 s.2.2.2.2 s.2.1 s.2.2.1)
    (post := fun p => ∃ out : alloc.vec.Vec I64, p = (out, true, none) ∧
      tq2_0EmitInv raw b take start out 256#usize true none)
  · rintro ⟨out, ok1, err, posX, groupX⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hok, herr, hgle, hgdisj, hposX, hemit⟩ := hinv'
    rw [hok, herr]
    by_cases hlt : groupX.val < 64
    · obtain ⟨out1, pos1, group1, hbody, hgroup1, hinv2, hpos1⟩ :=
        tq2_0Group_body_step raw b scale qs take start posX groupX out hbridge hcap
          hlt ⟨rfl, rfl, hgle, hgdisj, hposX, ⟨rfl, rfl, hemit.2.2⟩⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show 64 - group1.val < 64 - groupX.val
      omega
    · have hnot : ¬ (groupX < 64#usize) :=
        fun hc => hlt ((UScalar.lt_equiv groupX 64#usize).1 hc)
      have hbody : dequant.lowbit.dequant_tq2_0_row_loop0_loop0.body scale qs take out true none
          posX groupX = ok (ControlFlow.done (out, true, none)) := by
        unfold dequant.lowbit.dequant_tq2_0_row_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hg64 : groupX.val = 64 := by omega
      refine ⟨out, rfl, ?_⟩
      have hpos2 : posX = 256#usize := UScalar.eq_of_val_eq (by rw [hposX, hg64]; rfl)
      rw [← hpos2]
      exact ⟨rfl, rfl, hemit.2.2⟩
  · exact hinit

/-! ## Block loop and the row law -/

/-- The success domain of one TQ2_0 block `b`. -/
def tq2_0BlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  b.val * 66 + 66 ≤ raw.val.length ∧
  f16Exp (f16LeHalf raw (b.val * 66 + 64)) ≠ 31 ∧
  (∀ p, p < 256 → I64.min ≤ tq2_0Weight raw (256 * b.val + p) ∧
    tq2_0Weight raw (256 * b.val + p) ≤ I64.max)

/-- Block-loop invariant: after `block` blocks the output is the prefix of `min (256·block) n`. -/
def tq2_0RowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ tq2_0RowValues raw (Nat.min (256 * block.val) n.val)

/-- The extracted `div_ceil n TQ2_0_QK` succeeds and its value is `(n + 255)/256`. -/
theorem tq2_0_div_ceil_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.TQ2_0_QK = ok blocks ∧
      blocks.val = (n.val + 255) / 256 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.TQ2_0_QK) (by simp only [quant.TQ2_0_QK]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.TQ2_0_QK) (by simp only [quant.TQ2_0_QK]; decide))
  have hqv' : q.val = n.val / 256 := by rw [hqv]; simp [quant.TQ2_0_QK]
  have hrv' : r.val = n.val % 256 := by rw [hrv]; simp [quant.TQ2_0_QK]
  have hq' : UScalar.div n quant.TQ2_0_QK = ok q := hq
  have hr' : UScalar.rem n quant.TQ2_0_QK = ok r := hr
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

/-- `256·b + min 256 (n − 256·b) = min (256·(b+1)) n` when `256·b ≤ n`. -/
lemma mul_block_add_min_256 (b n : Nat) (h : 256 * b ≤ n) :
    256 * b + Nat.min 256 (n - 256 * b) = Nat.min (256 * (b + 1)) n := by
  change 256 * b + min 256 (n - 256 * b) = min (256 * (b + 1)) n
  by_cases h1 : n - 256 * b ≤ 256
  · rw [Nat.min_eq_right h1, Nat.min_eq_right (by omega)]
    omega
  · have h1' : 256 < n - 256 * b := not_le.mp h1
    rw [Nat.min_eq_left (le_of_lt h1'), Nat.min_eq_left (by omega)]
    omega

/-- One block-loop body step: decode the block, run the group nest, append the block prefix. -/
theorem tq2_0Block_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : tq2_0BlockDomain raw block)
    (hcap : out.val.length + (n.val - 256 * block.val) < Usize.max)
    (hinv : tq2_0RowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.lowbit.dequant_tq2_0_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      tq2_0RowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨hlen, hfin, hprod⟩ := hdom
  obtain ⟨d, qs, hq2, hdv, hqs⟩ := tq2_0_block_ok raw block hlen hfin
  have hltmul : 256 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hblock)
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.TQ2_0_QK) (by simp only [quant.TQ2_0_QK]; scalar_tac))
  have hivv' : i.val = 256 * block.val := by
    rw [hivv]; simp only [quant.TQ2_0_QK]; scalar_tac
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 256 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.TQ2_0_QK i1
  have htakev' : take.val = Nat.min 256 (n.val - 256 * block.val) := by
    rw [htakev, hi1v']; simp [quant.TQ2_0_QK]
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 256 := by rw [htakev']; exact min_le_left _ _
  have hcap' : out.val.length + take.val < Usize.max := by
    rw [htakev']
    have : Nat.min 256 (n.val - 256 * block.val) ≤ n.val - 256 * block.val := min_le_right _ _
    omega
  have hbridge : ∀ p, p < 256 →
      tq2_0PosVal d.val qs.val p = (tq2_0BlockVals raw block.val).getD p 0 ∧
      I64.min ≤ tq2_0PosVal d.val qs.val p ∧ tq2_0PosVal d.val qs.val p ≤ I64.max := by
    intro p hp
    have hw := tq2_0PosVal_weight raw block.val qs.val d.val p hqs hdv hp
    refine ⟨?_, ?_⟩
    · rw [hw, tq2_0BlockVals_getD raw block.val p hp]
    · rw [hw]
      exact hprod p hp
  have hem0 : tq2_0EmitInv raw block.val take out out 0#usize true none :=
    ⟨rfl, rfl, by simp⟩
  obtain ⟨r, hloop, hpost⟩ := WP.spec_imp_exists
    (tq2_0Group_loop_spec raw block.val d qs take out out hbridge hcap' hem0)
  obtain ⟨vals1, hr, hemit⟩ := hpost
  subst hr
  have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 256 * block.val + take.val = Nat.min (256 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min_256 block.val n.val (le_of_lt hltmul)
  have hmap : (tq2_0BlockVals raw block.val).take (Nat.min 256 take.val) =
      (List.range take.val).map (fun j => tq2_0Weight raw (256 * block.val + j)) := by
    have hm : Nat.min 256 take.val = take.val := Nat.min_eq_right htake_le
    rw [hm]
    unfold tq2_0BlockVals
    rw [← List.map_take]
    have htr : (List.range 256).take take.val = List.range take.val := by
      rw [List.take_range]
      congr 1
      exact Nat.min_eq_left htake_le
    rw [htr]
  have hemit' : dequantVals vals1.val = dequantVals out.val ++
      (List.range take.val).map (fun j => tq2_0Weight raw (256 * block.val + j)) := by
    have h := hemit.2.2
    rw [show (256#usize).val = 256 from rfl] at h
    rw [hmap] at h
    exact h
  have hinv_new : dequantVals vals1.val =
      dequantVals out0.val ++ tq2_0RowValues raw (Nat.min (256 * block1.val) n.val) := by
    rw [hemit', hinv.2.2.2, hmin, List.append_assoc, tq2_0RowValues_append, hsum, hblock1v']
  refine ⟨vals1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.lowbit.dequant_tq2_0_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq2]
  simp only [bind_tc_ok]
  change (do
    let i' ← block * quant.TQ2_0_QK
    let i1' ← n - i'
    let take' ← core.cmp.Ord.min.trait_default core.cmp.OrdUsize quant.TQ2_0_QK i1'
    let (out1, ok2, err1) ←
      dequant.lowbit.dequant_tq2_0_row_loop0_loop0 out true none d qs take' 0#usize 0#usize
    let block1' ← block + 1#usize
    ok (ControlFlow.cont (out1, block1', ok2, err1)))
    = ok (ControlFlow.cont (vals1, block1, true, none))
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rw [hblock1a]
  simp only [bind_tc_ok]
  rfl

/-- The block loop: `loop.spec_decr_nat` with `tq2_0RowInv` and measure `blocks - block`. -/
theorem tq2_0Row_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : ∀ b : Usize, b.val < blocks.val → tq2_0BlockDomain raw b)
    (hcap : out0.val.length + n.val < Usize.max) :
    WP.spec (dequant.lowbit.dequant_tq2_0_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ tq2_0RowValues raw n.val) := by
  unfold dequant.lowbit.dequant_tq2_0_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => tq2_0RowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
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
        simpa [dequantVals, tq2_0RowValues] using h
      have hcap' : out.val.length + (n.val - 256 * block.val) < Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        tq2_0Block_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.lowbit.dequant_tq2_0_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.lowbit.dequant_tq2_0_row_loop0.body, if_neg hnot]
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
    simp [tq2_0RowValues]

/-- **`TQ2_0` row law.** On the block success domain, the extracted `dequant_tq2_0_row` succeeds
with `Ok ()` and appends exactly the first `n` raw weights `tq2_0RowValues raw n`. -/
theorem dequant_tq2_0_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 255) / 256 → tq2_0BlockDomain raw b)
    (hcap : out.val.length + n.val < Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_tq2_0_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ tq2_0RowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := tq2_0_div_ceil_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (tq2_0Row_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.lowbit.dequant_tq2_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- `dequant_row_fixed` at `ggml_type = 35` (`GGML_TYPE_TQ2_0`) is `dequant_tq2_0_row`. -/
theorem dequant_row_fixed_tq2_0 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 35#u32 raw n out = dequant.lowbit.dequant_tq2_0_row raw n out := rfl

/-! ## One-block joint domain witness -/

/-- The 66-byte witness block: 64 zero `qs` bytes and `d = 1.0` (`0x3C00` little-endian). -/
def tq2_0WitBytes : List U8 := List.replicate 64 0#u8 ++ [0#u8, 60#u8]

theorem tq2_0WitBytes_length : tq2_0WitBytes.length = 66 := by simp [tq2_0WitBytes]

def tq2_0WitRaw : Slice U8 := Slice.from tq2_0WitBytes (by rw [tq2_0WitBytes_length]; scalar_tac)

theorem tq2_0WitRaw_val : tq2_0WitRaw.val = tq2_0WitBytes := Slice.from_val _ _

/-- Every `qs` byte of the witness block is `0`. -/
lemma tq2_0Wit_getD_small (k : Nat) (hk : k < 64) : tq2_0WitBytes.getD k 0#u8 = 0#u8 := by
  unfold tq2_0WitBytes
  rw [List.getD_eq_getElem?_getD, List.getElem?_append]
  rw [if_pos (by simpa using hk)]
  rw [List.getElem?_replicate]
  simp [hk]

lemma tq2_0Wit_getD_64 : tq2_0WitBytes.getD 64 0#u8 = 0#u8 := by
  unfold tq2_0WitBytes
  rw [List.getD_eq_getElem?_getD, List.getElem?_append]
  rw [if_neg (by simp)]
  simp

lemma tq2_0Wit_getD_65 : tq2_0WitBytes.getD 65 0#u8 = 60#u8 := by
  unfold tq2_0WitBytes
  rw [List.getD_eq_getElem?_getD, List.getElem?_append]
  rw [if_neg (by simp)]
  simp

/-- The witness scale bits `[0x00, 0x3C]` are `0x3C00 = 15360` (`1.0`). -/
lemma tq2_0Wit_f16LeHalf_64 : (f16LeHalf tq2_0WitRaw 64).val = 15360 := by
  have hb0 : tq2_0WitRaw.val.getD 64 0#u8 = 0#u8 := by
    rw [tq2_0WitRaw_val]; exact tq2_0Wit_getD_64
  have hb1 : tq2_0WitRaw.val.getD 65 0#u8 = 60#u8 := by
    rw [tq2_0WitRaw_val]; exact tq2_0Wit_getD_65
  have heq : f16LeHalf tq2_0WitRaw 64 =
      core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) := by
    unfold f16LeHalf
    rw [hb0, hb1]
  rw [heq]
  decide

/-- `d = 1.0 = 2^32` in Q32.32 at block `0`. -/
lemma tq2_0Wit_scale : tq2_0ScaleRef tq2_0WitRaw 0 = 2 ^ 32 := by
  have hv := tq2_0Wit_f16LeHalf_64
  unfold tq2_0ScaleRef
  rw [show 0 * 66 + 64 = 64 by ring]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

/-- Every code of the witness block is `0`. -/
lemma tq2_0Wit_code (p : Nat) (hp : p < 256) : tq2_0CodeRef tq2_0WitRaw 0 p = 0 := by
  unfold tq2_0CodeRef
  have hidx : (p / 128) * 32 + p % 32 < 64 := by omega
  rw [tq2_0WitRaw_val,
    show 0 * 66 + ((p / 128) * 32 + p % 32) = (p / 128) * 32 + p % 32 by ring,
    tq2_0Wit_getD_small _ hidx]
  simp

/-- Every weight of the witness block is `(0 − 1)·2^32 = −2^32`. -/
lemma tq2_0Wit_weight (p : Nat) (hp : p < 256) :
    tq2_0Weight tq2_0WitRaw p = -(2 ^ 32) := by
  unfold tq2_0Weight
  rw [show p / 256 = 0 by omega, show p % 256 = p by omega,
    tq2_0Wit_scale, tq2_0Wit_code p hp]
  norm_num

/-- **Joint non-vacuity of `tq2_0BlockDomain` at block `0`.** -/
theorem tq2_0BlockDomain_witness : tq2_0BlockDomain tq2_0WitRaw 0#usize := by
  have hb : (0#usize).val = 0 := by decide
  refine ⟨?_, ?_, ?_⟩
  · simp [hb, tq2_0WitRaw_val, tq2_0WitBytes_length]
  · have hv := tq2_0Wit_f16LeHalf_64
    unfold f16Exp
    rw [hb, show 0 * 66 + 64 = 64 by ring, hv]
    norm_num
  · intro p hp
    rw [show 256 * (0#usize).val + p = p by rw [hb]; ring, tq2_0Wit_weight p hp]
    norm_num [I64.min_eq, I64.max_eq]

/-- The empty vector's payload is `[]`. -/
lemma tq2_0_vec_new_val_i64 : (alloc.vec.Vec.new I64).val = ([] : List I64) := by
  simp [alloc.vec.Vec.new, alloc.vec.Vec.from_val]

/-- `getD` of a `Vec` value is the `getD` of the underlying `I64` list. -/
lemma tq2_0_dequantVals_getD (l : List I64) (i : Nat) :
    (dequantVals l).getD i 0 = (l.getD i 0#i64).val := by
  unfold dequantVals
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases h : l[i]? with
  | none => simp
  | some v => simp

/-- **`TQ2_0` row witness at `n = 256`.** A single concrete block: the extracted row succeeds on the
empty vector, appends exactly the `tq2_0RowValues`, and element `0` is `−2^32`. -/
theorem tq2_0Witness_row :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_tq2_0_row tq2_0WitRaw 256#usize (alloc.vec.Vec.new I64) =
          ok (.Ok (), out1) ∧
      dequantVals out1.val = tq2_0RowValues tq2_0WitRaw 256 ∧
      (out1.val.getD 0 0#i64).val = -(2 ^ 32) := by
  have hdom : ∀ b : Usize, b.val < ((256#usize).val + 255) / 256 →
      tq2_0BlockDomain tq2_0WitRaw b := by
    intro b hb
    have hb1 : b.val < 1 := by
      have hR : (((256#usize).val + 255) / 256) = 1 := by decide
      rwa [hR] at hb
    have hb0 : b.val = 0 := by omega
    have hbeq : b = 0#usize := UScalar.eq_of_val_eq (by rw [hb0]; rfl)
    rw [hbeq]
    exact tq2_0BlockDomain_witness
  obtain ⟨out1, hout, hvals⟩ :=
    dequant_tq2_0_row_spec tq2_0WitRaw 256#usize (alloc.vec.Vec.new I64) hdom (by
      rw [tq2_0_vec_new_val_i64]; scalar_tac)
  have hvals' : dequantVals out1.val = tq2_0RowValues tq2_0WitRaw 256 := by
    rw [tq2_0_vec_new_val_i64] at hvals
    simpa [dequantVals] using hvals
  refine ⟨out1, hout, hvals', ?_⟩
  · have h0 := congrArg (fun l : List ℤ => l.getD 0 0) hvals'
    rw [tq2_0_dequantVals_getD] at h0
    rw [h0]
    unfold tq2_0RowValues
    rw [List.getD_eq_getElem?_getD, List.getElem?_map]
    have h : (List.range 256)[0]? = some 0 := by decide
    rw [h]
    exact tq2_0Wit_weight 0 (by norm_num)

end

end apeiron_word
