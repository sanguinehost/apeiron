/-
  `Serve/Ffn.lean` — reference semantics for the integer/index kernels of
  `src/engine/chart/ffn.rs`, group `chart.ffn`.

  The extracted bodies are in `ApeironWord`: `chart.ffn.copy_i64_at(_loop)` and
  `chart.ffn.copy_i8_at(_loop)`. They are the layout layer of the Q8-activation FFN path
  (`quant_act_pooled` → `matvec_q8act_chunked` → `projections_q8act` → `ffn_q8act`): one pool task's
  block run of `scales` (i64) or `codes` (i8) is written into the shared activation allocation.

  This file carries, for each of the two kernels, a plain-Lean reference — write a list into another
  at an index offset, stopping at the first end — and the refinement theorem that the extracted body
  returns `ok` and its result denotes the reference, for every destination, source and offset, with no
  hypothesis bundle. Both kernels are total: the loop guard bounds every write, and every checked
  addition stays inside `Usize`, because the offset is a `Usize` (at most `Usize.max`) and while a
  write happens `off + i < dst.length ≤ Usize.max`.

  Skipped in this group, with the reason:
  * `f32_to_q32_checked`, `swiglu_one_f32_scalar`, `swiglu_one_carrier`, `swiglu_f32_avx2_8_into` —
    `F32` carrier; the float model and its rounding are a separate refinement, not index arithmetic.
  * the AVX2 lane — intrinsic/lane semantics; no lane model here.
  * `ActPtr::write_scales` / `ActPtr::write_codes` — `unsafe` raw-pointer `copy_nonoverlapping`; the
    model has no raw-pointer memory.
  * `any_delegate` — a `Bool` fold selecting between the pooled dispatch and the sequential delegating
    one; a dispatch selector, not an index/layout kernel.
  * `lut_at` — a table lookup over loaded codes whose `Err` branch carries `MatvecError`.
  `quant_act_pooled`, `projections_q8act`, `ffn_q8act` cross into the `pool` and `matvec` groups.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, or the `pool`/`matvec` groups. The statements are about the Aeneas-extracted bodies over
  the model `Slice` and the model `Usize` word bounds.
-/

import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow

/-! ## §1 · The reference: a clamped strided write

`copyPart dst src off j` is `dst` after writing `src[0..j]` at `off`: position `off + i` holds
`src[i]` for every `i < j`, and every other position still holds `dst`'s. `copyRef` is the whole
operation: the loop stops at the first end, so exactly `min src.length (dst.length - off)` writes
happen and a position at or past `dst.length` is never written. `List.set` past the length is the
identity, so the reference is total. -/

/-- `dst` after the first `j` writes of `src` at `off`. -/
def copyPart {α : Type} [Inhabited α] (dst src : List α) (off : Nat) : Nat → List α
  | 0 => dst
  | (j + 1) => (copyPart dst src off j).set (off + j) (src.getD j default)

/-- The reference result: write `src` into `dst` at offset `off`, stopping at the first end. -/
def copyRef {α : Type} [Inhabited α] (dst src : List α) (off : Nat) : List α :=
  copyPart dst src off (min src.length (dst.length - off))

theorem copyPart_zero {α : Type} [Inhabited α] (dst src : List α) (off : Nat) :
    copyPart dst src off 0 = dst := rfl

theorem copyPart_succ {α : Type} [Inhabited α] (dst src : List α) (off j : Nat) :
    copyPart dst src off (j + 1)
      = (copyPart dst src off j).set (off + j) (src.getD j default) := rfl

/-- The write leaves the destination length alone: both `List.set` and the copy preserve `dst`. -/
theorem copyPart_length {α : Type} [Inhabited α] (dst src : List α) (off j : Nat) :
    (copyPart dst src off j).length = dst.length := by
  induction j with
  | zero => rfl
  | succ j ih => rw [copyPart_succ, List.length_set, ih]

theorem copyRef_length {α : Type} [Inhabited α] (dst src : List α) (off : Nat) :
    (copyRef dst src off).length = dst.length := copyPart_length dst src off _

/-- Every model `Usize` value is at most `Usize.max`: the offset argument of a copy is bounded by
the platform word maximum, which is what keeps the checked `off + i` additions from overflowing. -/
theorem usize_val_le_max (x : Std.Usize) : x.val ≤ Usize.max := by
  have h := x.hrBounds
  simpa [UScalar.rMax, Usize.rMax, Usize.max, Usize.numBits] using h

/-! ## §2 · The extracted bodies, step by step

`chart.ffn.copy_i64_at_loop.body` (`ApeironWord.lean`, from `src/engine/chart/ffn.rs:68-74`) and
`chart.ffn.copy_i8_at_loop.body` (`:77-83`) are the same loop over the same two model operations
(`Slice.len`, `Slice.index_usize`, `Slice.update` and the checked `Usize.add`), at `Std.I64` and
`Std.I8`; the two proofs are the same argument at those two carriers.

`copy_*_body_step` is one turn of the loop below the stopping bound: the body writes one word and
increments the counter. `copy_*_body_done` is a turn at or past the bound: the body returns `done`
with the accumulator unchanged. The no-overflow hypothesis of `copy_*_body_done` is exactly what the
loop invariant supplies — see `copy_*_loop_spec`, where the counter's bound gives `off + i <
dst.length` whenever a write happens and `off + i ≤ Usize.max` in the returning case. -/

/-- **One `copy_i64_at` body step.** Below the stopping bound the body succeeds, writes
`src[i]` at `off + i` into the accumulator, and returns the counter `i + 1`. -/
theorem copy_i64_body_step (dst src : Slice Std.I64) (off i : Std.Usize)
    (hi : i.val < min src.val.length (dst.val.length - off.val)) :
    ∃ s : Slice Std.I64, ∃ i' : Std.Usize,
      chart.ffn.copy_i64_at_loop.body off src dst i = ok (ControlFlow.cont (s, i')) ∧
      i'.val = i.val + 1 ∧
      s.val = dst.val.set (off.val + i.val) (src.val.getD i.val default) := by
  unfold chart.ffn.copy_i64_at_loop.body
  have hilt : i.val < src.val.length := lt_of_lt_of_le hi (Nat.min_le_left _ _)
  have hlen_dst : dst.val.length ≤ Usize.max := Slice.length_ineq dst
  have hlen_src : src.val.length ≤ Usize.max := Slice.length_ineq src
  have hlt_dst : off.val + i.val < dst.val.length := by
    have := Nat.min_le_right src.val.length (dst.val.length - off.val)
    have := lt_of_lt_of_le hi this
    omega
  have hsum : off.val + i.val ≤ Usize.max := by omega
  have h1 : (1#usize : Std.Usize).val = 1 := rfl
  have h1max : i.val + (1#usize : Std.Usize).val ≤ Usize.max := by omega
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hilt)]
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := off) (y := i) hsum)
  rw [hi2]
  simp only [bind_tc_ok]
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val, hi2v]; exact hlt_dst)]
  obtain ⟨i4, hi4, hi4v⟩ := WP.spec_imp_exists (Slice.index_usize_spec src i hilt)
  rw [hi4]
  simp only [bind_tc_ok]
  obtain ⟨s, hs, hsv⟩ :=
    WP.spec_imp_exists (Slice.update_spec dst i2 i4 (by rw [hi2v]; exact hlt_dst))
  rw [hs]
  simp only [bind_tc_ok]
  obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) h1max)
  rw [hi5]
  simp only [bind_tc_ok]
  refine ⟨s, i5, rfl, by rw [hi5v, h1], ?_⟩
  rw [hsv, Slice.set_val_eq, hi2v, hi4v, ← List.getD_eq_getElem (l := src.val) (d := default) hilt]

/-- **One `copy_i64_at` body turn at or past the bound.** With no room left — the counter reached the
stopping bound, or `off + i` is past the destination — the body returns `done` with the accumulator
unchanged, provided the checked addition does not overflow. -/
theorem copy_i64_body_done (dst src : Slice Std.I64) (off i : Std.Usize)
    (hi : ¬ i.val < min src.val.length (dst.val.length - off.val))
    (hov : off.val + i.val ≤ Usize.max) :
    chart.ffn.copy_i64_at_loop.body off src dst i = ok (ControlFlow.done dst) := by
  unfold chart.ffn.copy_i64_at_loop.body
  by_cases hs : i.val < src.val.length
  · rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hs)]
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := off) (y := i) hov)
    rw [hi2]
    simp only [bind_tc_ok]
    have hge : ¬ i2.val < dst.val.length := by
      rw [hi2v]
      have hKle : min src.val.length (dst.val.length - off.val) ≤ i.val := le_of_not_gt hi
      rcases lt_or_ge (dst.val.length - off.val) src.val.length with h | h
      · rw [Nat.min_eq_right (le_of_lt h)] at hKle
        omega
      · rw [Nat.min_eq_left h] at hKle
        omega
    rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hge)]
  · rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hs)]

/-- **One `copy_i8_at` body step.** `copy_i64_body_step` at `Std.I8`. -/
theorem copy_i8_body_step (dst src : Slice Std.I8) (off i : Std.Usize)
    (hi : i.val < min src.val.length (dst.val.length - off.val)) :
    ∃ s : Slice Std.I8, ∃ i' : Std.Usize,
      chart.ffn.copy_i8_at_loop.body off src dst i = ok (ControlFlow.cont (s, i')) ∧
      i'.val = i.val + 1 ∧
      s.val = dst.val.set (off.val + i.val) (src.val.getD i.val default) := by
  unfold chart.ffn.copy_i8_at_loop.body
  have hilt : i.val < src.val.length := lt_of_lt_of_le hi (Nat.min_le_left _ _)
  have hlen_dst : dst.val.length ≤ Usize.max := Slice.length_ineq dst
  have hlen_src : src.val.length ≤ Usize.max := Slice.length_ineq src
  have hlt_dst : off.val + i.val < dst.val.length := by
    have := Nat.min_le_right src.val.length (dst.val.length - off.val)
    have := lt_of_lt_of_le hi this
    omega
  have hsum : off.val + i.val ≤ Usize.max := by omega
  have h1 : (1#usize : Std.Usize).val = 1 := rfl
  have h1max : i.val + (1#usize : Std.Usize).val ≤ Usize.max := by omega
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hilt)]
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := off) (y := i) hsum)
  rw [hi2]
  simp only [bind_tc_ok]
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val, hi2v]; exact hlt_dst)]
  obtain ⟨i4, hi4, hi4v⟩ := WP.spec_imp_exists (Slice.index_usize_spec src i hilt)
  rw [hi4]
  simp only [bind_tc_ok]
  obtain ⟨s, hs, hsv⟩ :=
    WP.spec_imp_exists (Slice.update_spec dst i2 i4 (by rw [hi2v]; exact hlt_dst))
  rw [hs]
  simp only [bind_tc_ok]
  obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) h1max)
  rw [hi5]
  simp only [bind_tc_ok]
  refine ⟨s, i5, rfl, by rw [hi5v, h1], ?_⟩
  rw [hsv, Slice.set_val_eq, hi2v, hi4v, ← List.getD_eq_getElem (l := src.val) (d := default) hilt]

/-- **One `copy_i8_at` body turn at or past the bound.** `copy_i64_body_done` at `Std.I8`. -/
theorem copy_i8_body_done (dst src : Slice Std.I8) (off i : Std.Usize)
    (hi : ¬ i.val < min src.val.length (dst.val.length - off.val))
    (hov : off.val + i.val ≤ Usize.max) :
    chart.ffn.copy_i8_at_loop.body off src dst i = ok (ControlFlow.done dst) := by
  unfold chart.ffn.copy_i8_at_loop.body
  by_cases hs : i.val < src.val.length
  · rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hs)]
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := off) (y := i) hov)
    rw [hi2]
    simp only [bind_tc_ok]
    have hge : ¬ i2.val < dst.val.length := by
      rw [hi2v]
      have hKle : min src.val.length (dst.val.length - off.val) ≤ i.val := le_of_not_gt hi
      rcases lt_or_ge (dst.val.length - off.val) src.val.length with h | h
      · rw [Nat.min_eq_right (le_of_lt h)] at hKle
        omega
      · rw [Nat.min_eq_left h] at hKle
        omega
    rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hge)]
  · rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hs)]

/-! ## §3 · The loops and the headlines

The invariant is the write prefix: at counter `j` the accumulator is `copyPart dst src off j`, and
`j` never passes `K = min src.length (dst.length - off)`, where the loop stops. The measure is
`K - j`. The loop is entered at counter zero, where the accumulator is `dst` itself, so no
hypothesis about the incoming accumulator is needed. -/

/-- **The `copy_i64_at` loop, from counter zero.** Its result denotes the reference. -/
theorem copy_i64_loop_spec (dst src : Slice Std.I64) (off : Std.Usize) :
    chart.ffn.copy_i64_at_loop dst off src 0#usize ⦃ out =>
      out.val = copyRef dst.val src.val off.val ⦄ := by
  unfold chart.ffn.copy_i64_at_loop
  apply loop.spec_decr_nat
    (measure := fun st : Slice Std.I64 × Std.Usize =>
      min src.val.length (dst.val.length - off.val) - st.2.val)
    (inv := fun st => st.2.val ≤ min src.val.length (dst.val.length - off.val) ∧
      st.1.val = copyPart dst.val src.val off.val st.2.val)
    (post := fun out : Slice Std.I64 => out.val = copyRef dst.val src.val off.val)
  · rintro ⟨d, j⟩ ⟨hjK, hd⟩
    have hdlen : d.val.length = dst.val.length := by rw [hd, copyPart_length]
    by_cases hj : j.val < min src.val.length (dst.val.length - off.val)
    · obtain ⟨s, i', hb, hi'v, hsv⟩ := copy_i64_body_step d src off j (by rw [hdlen]; exact hj)
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      show (i'.val ≤ min src.val.length (dst.val.length - off.val) ∧
          s.val = copyPart dst.val src.val off.val i'.val) ∧
        (min src.val.length (dst.val.length - off.val) - i'.val
          < min src.val.length (dst.val.length - off.val) - j.val)
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [hi'v]; omega
      · rw [hi'v, copyPart_succ, ← hd, hsv]
      · rw [hi'v]; omega
    · have hjeq : j.val = min src.val.length (dst.val.length - off.val) :=
        le_antisymm hjK (le_of_not_gt hj)
      have hov : off.val + j.val ≤ Usize.max := by
        have h1 := Nat.min_le_right src.val.length (dst.val.length - off.val)
        have h2 := Slice.length_ineq dst
        have h3 : j.val ≤ dst.val.length - off.val := le_trans hjK h1
        have h4 := usize_val_le_max off
        omega
      have hb := copy_i64_body_done d src off j (by rw [hdlen]; exact hj) hov
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      show d.val = copyRef dst.val src.val off.val
      rw [copyRef, hd, hjeq]
  · refine ⟨by simp, ?_⟩
    show dst.val = copyPart dst.val src.val off.val (0#usize).val
    rw [show (0#usize : Std.Usize).val = 0 from rfl, copyPart_zero]

/-- **The `copy_i8_at` loop, from counter zero.** `copy_i64_loop_spec` at `Std.I8`. -/
theorem copy_i8_loop_spec (dst src : Slice Std.I8) (off : Std.Usize) :
    chart.ffn.copy_i8_at_loop dst off src 0#usize ⦃ out =>
      out.val = copyRef dst.val src.val off.val ⦄ := by
  unfold chart.ffn.copy_i8_at_loop
  apply loop.spec_decr_nat
    (measure := fun st : Slice Std.I8 × Std.Usize =>
      min src.val.length (dst.val.length - off.val) - st.2.val)
    (inv := fun st => st.2.val ≤ min src.val.length (dst.val.length - off.val) ∧
      st.1.val = copyPart dst.val src.val off.val st.2.val)
    (post := fun out : Slice Std.I8 => out.val = copyRef dst.val src.val off.val)
  · rintro ⟨d, j⟩ ⟨hjK, hd⟩
    have hdlen : d.val.length = dst.val.length := by rw [hd, copyPart_length]
    by_cases hj : j.val < min src.val.length (dst.val.length - off.val)
    · obtain ⟨s, i', hb, hi'v, hsv⟩ := copy_i8_body_step d src off j (by rw [hdlen]; exact hj)
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      show (i'.val ≤ min src.val.length (dst.val.length - off.val) ∧
          s.val = copyPart dst.val src.val off.val i'.val) ∧
        (min src.val.length (dst.val.length - off.val) - i'.val
          < min src.val.length (dst.val.length - off.val) - j.val)
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [hi'v]; omega
      · rw [hi'v, copyPart_succ, ← hd, hsv]
      · rw [hi'v]; omega
    · have hjeq : j.val = min src.val.length (dst.val.length - off.val) :=
        le_antisymm hjK (le_of_not_gt hj)
      have hov : off.val + j.val ≤ Usize.max := by
        have h1 := Nat.min_le_right src.val.length (dst.val.length - off.val)
        have h2 := Slice.length_ineq dst
        have h3 : j.val ≤ dst.val.length - off.val := le_trans hjK h1
        have h4 := usize_val_le_max off
        omega
      have hb := copy_i8_body_done d src off j (by rw [hdlen]; exact hj) hov
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      show d.val = copyRef dst.val src.val off.val
      rw [copyRef, hd, hjeq]
  · refine ⟨by simp, ?_⟩
    show dst.val = copyPart dst.val src.val off.val (0#usize).val
    rw [show (0#usize : Std.Usize).val = 0 from rfl, copyPart_zero]

/-! ## §4 · Headlines -/

/-- **`copy_i64_at` is total and denotes the reference.** For every destination, source and offset,
the extracted `chart.ffn.copy_i64_at` returns `ok out` with `out` the clamped strided write of `src`
into `dst` at `off`. There is no refusal path: the loop guards the write and both checked additions
stay in `Usize`, so the `Result` always carries `ok`. `proved`. -/
theorem word_copy_i64_at_ok (dst src : Slice Std.I64) (off : Std.Usize) :
    ∃ out : Slice Std.I64,
      chart.ffn.copy_i64_at dst off src = ok out ∧
      out.val = copyRef dst.val src.val off.val := by
  have hspec := copy_i64_loop_spec dst src off
  rw [chart.ffn.copy_i64_at]
  exact WP.spec_imp_exists hspec

/-- **`copy_i8_at` is total and denotes the reference.** `word_copy_i64_at_ok` at `Std.I8`. `proved`. -/
theorem word_copy_i8_at_ok (dst src : Slice Std.I8) (off : Std.Usize) :
    ∃ out : Slice Std.I8,
      chart.ffn.copy_i8_at dst off src = ok out ∧
      out.val = copyRef dst.val src.val off.val := by
  have hspec := copy_i8_loop_spec dst src off
  rw [chart.ffn.copy_i8_at]
  exact WP.spec_imp_exists hspec

/-- The `copy_i64_at` copy does not resize the destination: the pool path writes each task's block run
into the shared activation allocation in place. `proved`. -/
theorem word_copy_i64_at_length (dst src : Slice Std.I64) (off : Std.Usize) (out : Slice Std.I64)
    (h : chart.ffn.copy_i64_at dst off src = ok out) : out.val.length = dst.val.length := by
  obtain ⟨out', hout', hval⟩ := word_copy_i64_at_ok dst src off
  rw [h] at hout'
  have hout : out = out' := Result.ok_injective hout'
  rw [hout, hval, copyRef, copyPart_length]

/-- The `copy_i8_at` copy does not resize the destination. `proved`. -/
theorem word_copy_i8_at_length (dst src : Slice Std.I8) (off : Std.Usize) (out : Slice Std.I8)
    (h : chart.ffn.copy_i8_at dst off src = ok out) : out.val.length = dst.val.length := by
  obtain ⟨out', hout', hval⟩ := word_copy_i8_at_ok dst src off
  rw [h] at hout'
  have hout : out = out' := Result.ok_injective hout'
  rw [hout, hval, copyRef, copyPart_length]

/-! ## §5 · Non-vacuity

The headlines carry no hypothesis, so the witness is a concrete instance of the reference. The three
i64 instances separate the three cases the clamp distinguishes: a run clamped by the destination end,
a run exactly filling the room, and an offset past the end, where nothing is written. -/

/-- Clamped by the destination: two words are written and the rest of `dst` stands. -/
example : copyRef [1#i64, 2#i64, 3#i64, 4#i64] [9#i64, 8#i64] 1
    = [1#i64, 9#i64, 8#i64, 4#i64] := by decide

/-- Exact fit: the source fills the room at the offset and is copied whole. -/
example : copyRef [1#i64, 2#i64, 3#i64] [7#i64, 8#i64] 1 = [1#i64, 7#i64, 8#i64] := by decide

/-- Offset past the end: no write happens and the destination is unchanged. -/
example : copyRef [1#i64, 2#i64] [9#i64] 5 = [1#i64, 2#i64] := by decide

/-- The same reference at `Std.I8`, one word written at offset zero. -/
example : copyPart [1#i8, 2#i8, 3#i8] [9#i8] 0 1 = [9#i8, 2#i8, 3#i8] := by decide

/-- The length statement of §4 on a definite instance. -/
example : (copyRef ([1#i8, 2#i8] : List Std.I8) [9#i8] 0).length = 2 := by decide

/-! ## §6 · Axiom cone

Each headline is checked here to depend on nothing outside
`{propext, Classical.choice, Quot.sound}`. -/

#print axioms word_copy_i64_at_ok
#print axioms word_copy_i8_at_ok
#print axioms word_copy_i64_at_length
#print axioms word_copy_i8_at_length

end apeiron_word
