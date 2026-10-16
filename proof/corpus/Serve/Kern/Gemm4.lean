/-
Integer, index and layout refinement of the extracted `kern::gemm` bodies, fourth group.

Serving path: `Engine::next_token_state` -> `step::cols` -> `kern::gemm::gemm_q8_0_group`. This
module refines the remaining portable (non-AVX2, non-`unsafe`) bodies that `Serve/Gemm3.lean` left
open:

* `pack_group_loop1_loop0` (the pad-block loop) and `pack_group_loop1` (the pad-lane loop).

`pack_group_loop0_loop1_loop0` (the four-code quad packer) is not contributed here. It was
attempted; its body symbolically executes under `step*`, but closing the extraction exposed a
missing library `step` rule for `core.convert.num.FromU32U8.from`, 15 index/overflow side
conditions, and the final `Slice.set` equality, none of which were discharged, and the attempt hit
heartbeat/recursion timeouts. No statement is weakened to pass, so the attempt is absent from this
file rather than carried open; the residual goals and the failed approaches are in the task return.

`column_ptrs` / `column_ptrs_loop` are blocked at this layer: their extracted bodies dereference
`core.slice.Slice.as_mut_ptr` (`proof/ApeironWord.lean:659`) and `alloc.vec.Vec.spare_capacity_mut`
(`proof/ApeironWord.lean:1136`), both emitted as `axiom : Result _`, so no `wp` spec and no raw
pointer value are derivable. `fast_columns_ok` / `fast_columns_ok_loop` need the `chart` group's
references and `step` rules (`chart.QuantAct.len`, `chart.QuantAct.magnitude`,
`chart.q8act_fast_bound_ok`, the `impl scales`/`codes` accessors); they are a different kernel group
and are left to that group.

Each reference is plain Lean over `Nat`/`List`: a fold of `Slice.set`/`Slice.update` writes, an
index arithmetic, a `Bool` scan. It states WHAT the body computes, not how the loop runs. The
pad loops do not refuse (they are total after their bounds), so `gemm4_*_spec` is the whole
contract; the quad packer's refusal condition is `Slice.index_usize`/`Slice.update` out of bounds,
which its premise bundle excludes.

NOT refined here, with the reason stated rather than skirted: the tile value itself. Every output
word is written by `kern.gemm.x86.rows`, emitted by the extraction as `axiom`; `gemm_q8_0_group`
also calls the `std_detect` `axiom`s and the `axiom` pool driver
`chart.pool.chunks.for_pbands`. No `wp` spec is derivable for an `axiom : Result _`, so neither the
dot-product equality nor `ok`-totality of the composed body is provable at this layer.
-/
import Serve.Kern.Gemm3

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Shared helpers -/

/-- `kern.gemm.LANES` is 8, the pad loops' index literal. -/
private lemma gemm4_LANES_val : (kern.gemm.LANES : Std.Usize).val = 8 := by
  unfold kern.gemm.LANES
  rfl

/-- `gemm3_padWords` preserves the list length, so the `Slice.update` bounds stay stated over the
loop's initial slice. -/
private lemma gemm4_padWords_length (w : List Std.U32) (base q : Nat) :
    (gemm3_padWords w base q).length = w.length := by
  unfold gemm3_padWords
  induction q with
  | zero => simp
  | succ n ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, List.length_set]
      exact ih

/-- The words after the pad-block loop has run for blocks `b0 .. b0 + n - 1` at pad lane
`pad_lane`: each block pads its eight quads. -/
def gemm4_wordsPadLoop (w0 : List Std.U32) (pad_lane b0 n : Nat) : List Std.U32 :=
  (List.range n).foldl
    (fun w k => gemm3_padWords w (((b0 + k) * 8) * 8 + pad_lane) 8) w0

/-- The shifts after the pad-block loop has run for blocks `b0 .. b0 + n - 1`: block `b`'s own lane
is zeroed. -/
def gemm4_shiftBlocks (s0 : List Std.I64) (b0 pad_lane n : Nat) : List Std.I64 :=
  (List.range n).foldl (fun s k => s.set ((b0 + k) * 8 + pad_lane) 0#i64) s0

private lemma gemm4_wordsPadLoop_succ (w0 : List Std.U32) (pad_lane b0 n : Nat) :
    gemm4_wordsPadLoop w0 pad_lane b0 (n + 1) =
      gemm3_padWords (gemm4_wordsPadLoop w0 pad_lane b0 n)
        (((b0 + n) * 8) * 8 + pad_lane) 8 := by
  unfold gemm4_wordsPadLoop
  rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

private lemma gemm4_shiftBlocks_succ (s0 : List Std.I64) (b0 pad_lane n : Nat) :
    gemm4_shiftBlocks s0 b0 pad_lane (n + 1) =
      (gemm4_shiftBlocks s0 b0 pad_lane n).set ((b0 + n) * 8 + pad_lane) 0#i64 := by
  unfold gemm4_shiftBlocks
  rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

private lemma gemm4_wordsPadLoop_length (w0 : List Std.U32) (pl b0 n : Nat) :
    (gemm4_wordsPadLoop w0 pl b0 n).length = w0.length := by
  unfold gemm4_wordsPadLoop
  induction n with
  | zero => simp
  | succ m ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
        gemm4_padWords_length, ih]

private lemma gemm4_shiftBlocks_length (s0 : List Std.I64) (b0 pl n : Nat) :
    (gemm4_shiftBlocks s0 b0 pl n).length = s0.length := by
  unfold gemm4_shiftBlocks
  induction n with
  | zero => simp
  | succ m ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
        List.length_set, ih]

/-- The pad-quad refinement of `Serve/Gemm3.lean`, restated as a `step` rule. -/
@[step]
theorem gemm4_loop1_loop0_loop0_spec (pad_lane b : Std.Usize) (words0 : Slice Std.U32)
    (hno : (b.val * 8 + 8) * 8 + pad_lane.val ≤ Usize.max)
    (hbound : ∀ (q : Nat) (_ : q < 8), (b.val * 8 + q) * 8 + pad_lane.val < words0.val.length) :
    kern.gemm.pack_group_loop1_loop0_loop0 words0 pad_lane b 0#usize ⦃ r =>
      r.val = gemm3_padWords words0.val (b.val * 8 * 8 + pad_lane.val) 8 ⦄ :=
  gemm3_pack_group_loop1_loop0_loop0_spec pad_lane b words0 hno hbound

/-! ## `pack_group_loop1_loop0`: the pad-block loop -/

@[step]
theorem gemm4_pack_group_loop1_loop0_spec (blocks : Std.Usize) (words0 : Slice Std.U32)
    (shift0 : Slice Std.I64) (pad_lane b0 : Std.Usize)
    (hb : b0.val ≤ blocks.val) (hpad : pad_lane.val < 8)
    (hno : ∀ (b' : Nat), b0.val ≤ b' → b' < blocks.val →
      (b' * 8 + 8) * 8 + pad_lane.val ≤ Usize.max)
    (hwb : ∀ (b' : Nat), b0.val ≤ b' → b' < blocks.val → ∀ (q : Nat), q < 8 →
      (b' * 8 + q) * 8 + pad_lane.val < words0.val.length)
    (hsb : ∀ (b' : Nat), b0.val ≤ b' → b' < blocks.val →
      b' * 8 + pad_lane.val < shift0.val.length) :
    kern.gemm.pack_group_loop1_loop0 blocks words0 shift0 pad_lane b0 ⦃ r =>
      r.1.val = gemm4_wordsPadLoop words0.val pad_lane.val b0.val (blocks.val - b0.val) ∧
      r.2.val = gemm4_shiftBlocks shift0.val b0.val pad_lane.val (blocks.val - b0.val) ⦄ := by
  unfold kern.gemm.pack_group_loop1_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Slice Std.U32 × Slice Std.I64 × Std.Usize =>
      blocks.val - st.2.2.val)
    (inv := fun st : Slice Std.U32 × Slice Std.I64 × Std.Usize =>
      b0.val ≤ st.2.2.val ∧ st.2.2.val ≤ blocks.val ∧
      st.1.val.length = words0.val.length ∧
      st.2.1.val.length = shift0.val.length ∧
      st.1.val = gemm4_wordsPadLoop words0.val pad_lane.val b0.val (st.2.2.val - b0.val) ∧
      st.2.1.val = gemm4_shiftBlocks shift0.val b0.val pad_lane.val (st.2.2.val - b0.val))
    (post := fun r : Slice Std.U32 × Slice Std.I64 =>
      r.1.val = gemm4_wordsPadLoop words0.val pad_lane.val b0.val (blocks.val - b0.val) ∧
      r.2.val = gemm4_shiftBlocks shift0.val b0.val pad_lane.val (blocks.val - b0.val))
  · rintro ⟨words, shift, b⟩ ⟨hb0b, hb', hwlen, hslen, hinvw, hinvs⟩
    dsimp only at *
    unfold kern.gemm.pack_group_loop1_loop0.body
    by_cases hlt : b < blocks
    · rw [if_pos hlt]
      have hbLt : b.val < blocks.val := by simpa [UScalar.lt_equiv] using hlt
      have hslen' : shift.length = shift0.val.length := hslen
      have hidx2 : ((b0.val + (b.val - b0.val)) * 8) * 8 + pad_lane.val
          = b.val * 8 * 8 + pad_lane.val := by
        rw [Nat.add_sub_cancel' hb0b]
      have hidx3 : (b0.val + (b.val - b0.val)) * 8 + pad_lane.val
          = b.val * 8 + pad_lane.val := by
        rw [Nat.add_sub_cancel' hb0b]
      step*
      · rw [gemm4_LANES_val]; have h := hno b.val hb0b hbLt; omega
      · rw [i_post, gemm4_LANES_val]; have h := hno b.val hb0b hbLt; omega
      · rw [i1_post, i_post, gemm4_LANES_val, hslen']
        exact hsb b.val hb0b hbLt
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · rw [b1_post]; omega
        · rw [b1_post]; omega
        · rw [words1_post, gemm4_padWords_length, hwlen]
        · rw [s_post]
          simp only [Slice.set_val_eq, List.length_set, hslen]
        · have hidx : b1.val - b0.val = (b.val - b0.val) + 1 := by rw [b1_post]; omega
          rw [words1_post, hinvw, hidx, gemm4_wordsPadLoop_succ, hidx2]
        · have hidx : b1.val - b0.val = (b.val - b0.val) + 1 := by rw [b1_post]; omega
          rw [s_post, Slice.set_val_eq, hidx, gemm4_shiftBlocks_succ, ← hinvs, hidx3,
            i1_post, i_post, gemm4_LANES_val]
        · rw [b1_post]; omega
    · rw [if_neg hlt]
      have hge : blocks.val ≤ b.val := by
        have h : ¬ (b.val < blocks.val) := by simpa [UScalar.lt_equiv] using hlt
        omega
      have hbk : b.val = blocks.val := le_antisymm hb' hge
      simp only [WP.spec_ok]
      rw [← hbk]
      exact ⟨hinvw, hinvs⟩
  · exact ⟨le_refl _, hb, rfl, rfl, by simp [gemm4_wordsPadLoop],
      by simp [gemm4_shiftBlocks]⟩

/-! ## `pack_group_loop1`: the pad-lane loop -/

/-- The pad-lane loop's reference: the pad-block loop run for every pad lane `p0 .. p0 + n - 1`. -/
def gemm4_padLanes (w0 : List Std.U32) (s0 : List Std.I64) (blocks p0 n : Nat) :
    List Std.U32 × List Std.I64 :=
  (List.range n).foldl
    (fun acc k => (gemm4_wordsPadLoop acc.1 (p0 + k) 0 blocks,
      gemm4_shiftBlocks acc.2 0 (p0 + k) blocks)) (w0, s0)

private lemma gemm4_padLanes_succ (w0 : List Std.U32) (s0 : List Std.I64) (blocks p0 n : Nat) :
    gemm4_padLanes w0 s0 blocks p0 (n + 1) =
      (gemm4_wordsPadLoop (gemm4_padLanes w0 s0 blocks p0 n).1 (p0 + n) 0 blocks,
       gemm4_shiftBlocks (gemm4_padLanes w0 s0 blocks p0 n).2 0 (p0 + n) blocks) := by
  unfold gemm4_padLanes
  rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem gemm4_pack_group_loop1_spec (blocks : Std.Usize) (words0 : Slice Std.U32)
    (shift0 : Slice Std.I64) (pad_lane0 : Std.Usize)
    (hpad : pad_lane0.val ≤ 8)
    (hno : ∀ (p b : Nat), pad_lane0.val ≤ p → p < 8 → b < blocks.val →
      (b * 8 + 8) * 8 + p ≤ Usize.max)
    (hwb : ∀ (p b q : Nat), pad_lane0.val ≤ p → p < 8 → b < blocks.val → q < 8 →
      (b * 8 + q) * 8 + p < words0.val.length)
    (hsb : ∀ (p b : Nat), pad_lane0.val ≤ p → p < 8 → b < blocks.val →
      b * 8 + p < shift0.val.length) :
    kern.gemm.pack_group_loop1 blocks words0 shift0 pad_lane0 ⦃ r =>
      r.1.val = (gemm4_padLanes words0.val shift0.val blocks.val pad_lane0.val
        (8 - pad_lane0.val)).1 ∧
      r.2.val = (gemm4_padLanes words0.val shift0.val blocks.val pad_lane0.val
        (8 - pad_lane0.val)).2 ⦄ := by
  unfold kern.gemm.pack_group_loop1
  apply loop.spec_decr_nat
    (measure := fun st : Slice Std.U32 × Slice Std.I64 × Std.Usize =>
      8 - st.2.2.val)
    (inv := fun st : Slice Std.U32 × Slice Std.I64 × Std.Usize =>
      pad_lane0.val ≤ st.2.2.val ∧ st.2.2.val ≤ 8 ∧
      st.1.val.length = words0.val.length ∧
      st.2.1.val.length = shift0.val.length ∧
      st.1.val = (gemm4_padLanes words0.val shift0.val blocks.val pad_lane0.val
        (st.2.2.val - pad_lane0.val)).1 ∧
      st.2.1.val = (gemm4_padLanes words0.val shift0.val blocks.val pad_lane0.val
        (st.2.2.val - pad_lane0.val)).2)
    (post := fun r : Slice Std.U32 × Slice Std.I64 =>
      r.1.val = (gemm4_padLanes words0.val shift0.val blocks.val pad_lane0.val
        (8 - pad_lane0.val)).1 ∧
      r.2.val = (gemm4_padLanes words0.val shift0.val blocks.val pad_lane0.val
        (8 - pad_lane0.val)).2)
  · rintro ⟨words, shift, pad_lane⟩ ⟨hpl0, hpl, hwlen, hslen, hinvw, hinvs⟩
    dsimp only at *
    unfold kern.gemm.pack_group_loop1.body
    by_cases hlt : pad_lane < kern.gemm.LANES
    · rw [if_pos hlt]
      have hlt8 : pad_lane.val < 8 := by
        simpa [UScalar.lt_equiv, gemm4_LANES_val] using hlt
      have hidx2 : pad_lane0.val + (pad_lane.val - pad_lane0.val) = pad_lane.val :=
        Nat.add_sub_cancel' hpl0
      step*
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · rw [pad_lane1_post]; omega
        · rw [pad_lane1_post]; omega
        · rw [words1_post, gemm4_wordsPadLoop_length, hwlen]
        · rw [words1_post1, gemm4_shiftBlocks_length, hslen]
        · have hidx : pad_lane1.val - pad_lane0.val = (pad_lane.val - pad_lane0.val) + 1 := by
            rw [pad_lane1_post]; omega
          rw [words1_post, hidx, gemm4_padLanes_succ, ← hinvw, hidx2, Nat.sub_zero]
        · have hidx : pad_lane1.val - pad_lane0.val = (pad_lane.val - pad_lane0.val) + 1 := by
            rw [pad_lane1_post]; omega
          rw [words1_post1, hidx, gemm4_padLanes_succ, ← hinvs, hidx2, Nat.sub_zero]
        · rw [pad_lane1_post]; omega
    · rw [if_neg hlt]
      have hge : 8 ≤ pad_lane.val := by
        have h : ¬ (pad_lane.val < 8) := by
          simpa [UScalar.lt_equiv, gemm4_LANES_val] using hlt
        omega
      have hpk : pad_lane.val = 8 := le_antisymm hpl hge
      simp only [WP.spec_ok]
      rw [← hpk]
      exact ⟨hinvw, hinvs⟩
  · exact ⟨le_refl _, hpad, rfl, rfl, by simp [gemm4_padLanes], by simp [gemm4_padLanes]⟩

/-! ## Non-vacuity -/

/-- The word-pad reference on a length-8 list changes exactly the addressed word. -/
example : gemm4_wordsPadLoop (List.replicate 8 (5#u32)) 0 0 1
    = (List.replicate 8 (5#u32)).set 0 2155905152#u32 := by decide

/-- The shift reference zeroes exactly the addressed lane. -/
example : gemm4_shiftBlocks (List.replicate 9 (5#i64)) 0 3 1
    = (List.replicate 9 (5#i64)).set 3 0#i64 := by decide

/-! ## Axiom cone -/


end apeiron_word
