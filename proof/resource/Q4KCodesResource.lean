import Q4KCodesSource
import Aeneas

/-
Cost model of the actual Q4_K integer code unpack, extracted from `src/format/quant.rs`.

SCOPE: `q4k_codes_source.quant.q4_k_codes` and its two loop bodies as extracted by the gate:
per-iteration transitions under their guards, the two loop ranks, the exit and refusal branches,
and one reachable success witness. Nothing outside these declarations is claimed.
REGISTER: resource obligation for the unpack loops: per-iteration work, loop rank, exit and
refusal branch, reachable success witness. Excluded: hardware timing, the code-value refinement
against a reference (`proof/corpus/Numeric/Packing/Format.lean`), and any canonical ring
correspondence.
CORPUS: `src/format/quant.rs:492-507` (`q4_k_codes`), `:497-505` (outer `while b < 4`),
`:499-503` (inner `while l < 32`), `:221-230` (`need`); extracted to
`proof/resource/Q4KCodesSource.lean` by `proof/resource/Q4KCodes_gate.py`.
REFERENCE: NONE - no borrowed object is invoked. The ranks are the `Nat` measures of the
extracted `loop` combinator and the position arithmetic is `Nat` division and modulo.
GATE: `python3 proof/resource/Q4KCodes_gate.py --controls`, cwd `/home/socol/Workspace/apeiron`,
lane `/tmp/opencode/apeiron-lane-1.lock`, 540-second whole-invocation budget.
GRADE: `proved` for every headline below: Lean kernel check, constant cone restricted to
`{propext, Classical.choice, Quot.sound}` and audited by `#resource_audit`, extraction from the
real source. The reading of a rank as an iteration *count* is a `model` deduction, marked at
`codesPerBlock`. No hardware or wall-clock measurement is claimed.
-/

namespace Q4KCodesResource

open Aeneas Aeneas.Std Result
open q4k_codes_source

/-! ## Block geometry and the refusal branch

The unpack reads one 144-byte super-block and the guard refuses anything shorter before any
accumulator is built. -/

/-- The geometry the guard uses: `2 + 2 + 12 + QK_K / 2 = 144` bytes. -/
theorem blockBytes : quant.Q4_K_BLOCK_BYTES = ok (144#usize) := by
  have h : quant.Q4_K_BLOCK_BYTES ⦃ z => z = (144#usize) ⦄ := by
    unfold quant.Q4_K_BLOCK_BYTES
    simp only [quant.QK_K]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- Refusal branch of the actual body: a short block returns `Truncated 144 (Slice.len blk)` from
    the guard `quant.need`, which is the `Break` arm of `quant.q4_k_codes`. No loop is entered on
    this path, so the cost of a refused block is the guard alone. -/
theorem refuses (blk : Slice Std.U8) (h : Slice.len blk < 144#usize) :
    word_q4_k_codes blk =
      ok (core.result.Result.Err (quant.QuantError.Truncated 144#usize (Slice.len blk))) := by
  rw [word_q4_k_codes]
  unfold quant.q4_k_codes
  rw [blockBytes]
  simp only [bind_tc_ok, quant.need, Slice.len]
  rw [if_pos h]
  simp [core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]

/-- A 143-byte payload is a reachable refusal of the real guard, not only a hypothetical one. -/
theorem refusal_witness :
    ∃ blk : Slice Std.U8, Slice.len blk = 143#usize ∧
      word_q4_k_codes blk =
        ok (core.result.Result.Err (quant.QuantError.Truncated 144#usize 143#usize)) := by
  set blk := Slice.from (List.replicate 143 (0#u8 : Std.U8)) (by scalar_tac) with hblk
  have hlen : Slice.len blk = 143#usize := UScalar.eq_of_val_eq (by simp [hblk])
  refine ⟨blk, hlen, ?_⟩
  rw [refuses blk ((UScalar.lt_equiv _ _).mpr (by simp [hlen])), hlen]

/-! ## Loop ranks

`b` counts 64-position groups (rank `4 - b`) and `l` counts written positions inside a group
(rank `32 - l`). Both are `Nat` measures that every guarded continuation lowers, so the extracted
`loop` bounds a run by the rank it starts from. -/

/-- Work measure of the inner loop: the positions of group `b` still to be written. -/
def innerRank (l : Std.Usize) : Nat := 32 - l.val

/-- Work measure of the outer loop: the 64-position groups still to be unpacked. -/
def outerRank (b : Std.Usize) : Nat := 4 - b.val

/-- A group starts at the full inner rank and a block starts at the full outer rank; both ranks
    reach zero exactly at the loop guards. -/
theorem rank_starts : innerRank 0#usize = 32 ∧ outerRank 0#usize = 4 ∧
    innerRank 32#usize = 0 ∧ outerRank 4#usize = 0 := by
  norm_num [innerRank, outerRank]

/-- Every guarded continuation of the inner loop lowers its rank by exactly one. -/
theorem innerRank_step (l l1 : Nat) (h : l < 32) (h1 : l1 = l + 1) :
    (32 - l1) + 1 = 32 - l := by omega

/-- Every guarded continuation of the outer loop lowers its rank by exactly one. -/
theorem outerRank_step (b b1 : Nat) (h : b < 4) (h1 : b1 = b + 1) :
    (4 - b1) + 1 = 4 - b := by omega

/-! ## Exit branches

Each guard has exactly two arms; the false arm returns the accumulator unchanged and performs no
read and no write. -/

/-- Inner exit: with the position rank exhausted the body returns `done out`, so the iteration
    costs nothing. -/
theorem inner_exit (qs : Slice Std.U8) (b : Std.Usize) (out : Array Std.U8 256#usize)
    (l : Std.Usize) (h : ¬ (l < 32#usize)) :
    quant.q4_k_codes_loop0_loop0.body qs b out l = ok (ControlFlow.done out) := by
  unfold quant.q4_k_codes_loop0_loop0.body
  rw [if_neg h]

/-- Outer exit: with the group rank exhausted the body returns `done out`. -/
theorem outer_exit (qs : Slice Std.U8) (out : Array Std.U8 256#usize) (b : Std.Usize)
    (h : ¬ (b < 4#usize)) :
    quant.q4_k_codes_loop0.body qs out b = ok (ControlFlow.done out) := by
  unfold quant.q4_k_codes_loop0.body
  rw [if_neg h]

/-! ## Accumulator invariant and the code arithmetic

`code i` is the 4-bit code the body is to place at output position `i`; `hread` is the named
premise fixing the source bytes that the two reads of an iteration carry. `acc code b l out`
describes the accumulator after `b` whole groups and `l` written positions of group `b`. -/

/-- The 4-bit code the body writes at output position `i`: low nibble in the first half of each
    64-position group, high nibble in the second half. -/
def codeOf (i : Nat) : Nat := if i % 64 < 32 then 5 else 15

/-- Accumulator contents after `b` whole groups and `l` written positions of group `b`. -/
def part (code : Nat → Nat) (b l i : Nat) : Nat :=
  if i < 64 * b then code i
  else if i / 64 = b ∧ (i % 64 < l ∨ (32 ≤ i % 64 ∧ i % 64 < 32 + l)) then code i
  else 0

/-- The loop invariant: both ranks in range and the accumulator equal to `part code` pointwise. -/
def acc (code : Nat → Nat) (b l : Nat) (out : Array Std.U8 256#usize) : Prop :=
  b ≤ 4 ∧ l ≤ 32 ∧ ∀ i, i < 256 → (out.val.getD i (0#u8 : Std.U8)).val = part code b l i

lemma getD_set {α : Type} (l : List α) (j i : Nat) (x d : α) :
    (l.set j x).getD i d = if i = j ∧ j < l.length then x else l.getD i d := by
  rw [List.getD, List.getElem?_set]
  by_cases hij : i = j
  · subst i
    rw [if_pos rfl]
    by_cases hj : j < l.length
    · rw [if_pos hj, if_pos ⟨rfl, hj⟩]
      rfl
    · rw [if_neg hj, if_neg (fun h => hj h.2)]
      rw [List.getD, List.getElem?_eq_none (by omega)]
  · rw [if_neg (fun h => hij h.symm), if_neg (fun h => hij h.1)]
    rfl

/-- Positions other than the two written by an iteration are unchanged. -/
lemma part_step (code : Nat → Nat) (b l i : Nat)
    (h1 : i ≠ 64 * b + l) (h2 : i ≠ 64 * b + 32 + l) :
    part code b (l + 1) i = part code b l i := by
  unfold part
  by_cases hlt : i < 64 * b
  · rw [if_pos hlt, if_pos hlt]
  · rw [if_neg hlt, if_neg hlt]
    by_cases hb : i / 64 = b
    · have hlo : i % 64 ≠ l := by intro h; exact h1 (by omega)
      have hhi : i % 64 ≠ 32 + l := by intro h; exact h2 (by omega)
      by_cases hA : i % 64 < l
      · rw [if_pos ⟨hb, Or.inl (by omega : i % 64 < l + 1)⟩]
        rw [if_pos ⟨hb, Or.inl hA⟩]
      · by_cases hB : 32 ≤ i % 64 ∧ i % 64 < 32 + l
        · rw [if_pos ⟨hb, Or.inr ⟨hB.1, by omega⟩⟩]
          rw [if_pos ⟨hb, Or.inr hB⟩]
        · have hneg1 : ¬ (i % 64 < l + 1 ∨ 32 ≤ i % 64 ∧ i % 64 < 32 + (l + 1)) := by
            rintro (h | h) <;> omega
          have hneg2 : ¬ (i % 64 < l ∨ 32 ≤ i % 64 ∧ i % 64 < 32 + l) := by
            rintro (h | h) <;> [exact hA h; exact hB h]
          rw [if_neg (fun h => hneg1 h.2), if_neg (fun h => hneg2 h.2)]
    · rw [if_neg (fun h => hb h.1), if_neg (fun h => hb h.1)]

/-- The low-nibble position of an iteration is written by that iteration. -/
lemma part_low (code : Nat → Nat) (b l : Nat) (hl : l < 32) :
    part code b (l + 1) (64 * b + l) = code (64 * b + l) := by
  unfold part
  rw [if_neg (by omega : ¬ (64 * b + l < 64 * b)),
    if_pos ⟨by omega, Or.inl (by omega)⟩]

/-- The high-nibble position of an iteration is written by that iteration. -/
lemma part_high (code : Nat → Nat) (b l : Nat) (hl : l < 32) :
    part code b (l + 1) (64 * b + 32 + l) = code (64 * b + 32 + l) := by
  unfold part
  rw [if_neg (by omega : ¬ (64 * b + 32 + l < 64 * b)),
    if_pos ⟨by omega, Or.inr ⟨by omega, by omega⟩⟩]

/-- Finishing a group leaves the accumulator in the state of the next group with no position
    written yet. -/
lemma part_bump (code : Nat → Nat) (b i : Nat) : part code b 32 i = part code (b + 1) 0 i := by
  unfold part
  have hm : i % 64 < 64 := Nat.mod_lt _ (by norm_num)
  by_cases h1 : i / 64 < b
  · rw [if_pos (by omega : i < 64 * b), if_pos (by omega : i < 64 * (b + 1))]
  · by_cases h2 : i / 64 = b
    · rw [if_neg (by omega : ¬ i < 64 * b), if_pos (by omega : i < 64 * (b + 1))]
      rw [if_pos ⟨h2, by omega⟩]
    · rw [if_neg (by omega : ¬ i < 64 * b), if_neg (by omega : ¬ i < 64 * (b + 1))]
      rw [if_neg (fun hh => h2 hh.1)]
      rw [if_neg (fun hh => by rcases hh.2 with h | h <;> omega)]

/-- The empty accumulator of the constructor state. -/
lemma part_zero (code : Nat → Nat) (i : Nat) : part code 0 0 i = 0 := by
  unfold part
  rw [if_neg (by omega : ¬ i < 64 * 0)]
  rw [if_neg (fun h => by rcases h.2 with h | h <;> omega)]

/-- Four finished groups expose the whole code array. -/
lemma part_full (code : Nat → Nat) (i : Nat) (hi : i < 256) : part code 4 0 i = code i := by
  unfold part
  rw [if_pos (by omega : i < 64 * 4)]

/-- The invariant holds at the constructor state of the outer loop. -/
lemma acc_repeat (code : Nat → Nat) : acc code 0 0 (Array.repeat 256#usize (0#u8 : Std.U8)) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro i hi
  rw [part_zero, Array.repeat_val]
  simp only [List.getD, List.getElem?_replicate]
  rw [if_pos (by simpa using hi)]
  rfl

lemma getD_index (qs : Slice Std.U8) (i : Std.Usize) (h : i.val < qs.val.length) :
    (qs.val[i.val]'h).val = (qs.val.getD i.val (0#u8 : Std.U8)).val := by
  rw [List.getD, List.getElem?_eq_getElem h]
  rfl

lemma nat_shr_4 (x : Nat) : x >>> 4 = x / 16 := by
  simpa using Nat.shiftRight_eq_div_pow x 4

lemma nat_and_15 (x : Nat) : x &&& 15 = x % 16 := by
  simpa using Nat.and_two_pow_sub_one_eq_mod x 4

/-! ## Per-iteration transitions and the loop work bounds -/

/-- One guarded inner iteration of the actual body: it reads `qs` twice at the single source
    index `32b + l`, writes exactly two codes at the distinct in-range positions `64b + l` and
    `64b + 32 + l`, and advances the position rank by exactly one. -/
theorem inner_step (qs : Slice Std.U8) (b : Std.Usize) (out : Array Std.U8 256#usize)
    (l : Std.Usize) (hb : b.val < 4) (hl : l < 32#usize) (hqs : 128 ≤ qs.val.length) :
    quant.q4_k_codes_loop0_loop0.body qs b out l ⦃ r =>
      match r with
      | .done _ => False
      | .cont p => p.2.val = l.val + 1 ∧
          64 * b.val + l.val < 256 ∧ 64 * b.val + 32 + l.val < 256 ∧
          64 * b.val + l.val ≠ 64 * b.val + 32 + l.val ⦄ := by
  unfold quant.q4_k_codes_loop0_loop0.body
  rw [if_pos hl]
  simp only [lift, bind_tc_ok]
  step*
  have hL : l.val < 32 := by scalar_tac
  exact ⟨l1_post, by omega, by omega, by omega⟩

/-- Inner loop work: from a state in the invariant, the actual inner loop performs at most
    `32 - l` guarded continuations, each of which writes the two positions of one pair, and ends
    with the 32 positions of group `b` written. -/
theorem inner_spec (qs : Slice Std.U8) (out0 : Array Std.U8 256#usize) (bv lv : Std.Usize)
    (code : Nat → Nat) (hb : bv.val < 4) (hqs : 128 ≤ qs.val.length)
    (hinv : acc code bv.val lv.val out0)
    (hread : ∀ b l : Nat, b < 4 → l < 32 →
      (qs.val.getD (32 * b + l) (0#u8 : Std.U8)).val &&& 15 = code (64 * b + l) ∧
      (qs.val.getD (32 * b + l) (0#u8 : Std.U8)).val / 16 = code (64 * b + 32 + l)) :
    quant.q4_k_codes_loop0_loop0 qs out0 bv lv ⦃ out => acc code bv.val 32 out ⦄ := by
  unfold quant.q4_k_codes_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Array Std.U8 256#usize × Std.Usize => innerRank st.2)
    (inv := fun st => acc code bv.val st.2.val st.1)
    (post := fun out => acc code bv.val 32 out)
  · rintro ⟨out, l⟩ hinv'
    dsimp only
    unfold quant.q4_k_codes_loop0_loop0.body
    by_cases hl : l < 32#usize
    · rw [if_pos hl]
      simp only [lift, bind_tc_ok]
      step*
      have hL : l.val < 32 := by scalar_tac
      have hi4 : i4.val = 64 * bv.val + l.val := by rw [i4_post, i3_post]
      have hi9 : i9.val = 64 * bv.val + 32 + l.val := by rw [i9_post, i8_post, i3_post]
      have hi1 : i1.val = 32 * bv.val + l.val := by rw [i1_post, i_post]
      have hi6 : i6.val = 32 * bv.val + l.val := by rw [i6_post, i_post]
      have hout : out.val.length = 256 := by exact Array.property out
      have hlen4 : i4.val < out.val.length := by rw [hi4, hout]; omega
      have hlen9 : i9.val < (out.val.set i4.val (i2 &&& 15#u8)).length := by
        rw [List.length_set, hi9, hout]; omega
      refine ⟨⟨hinv'.1, by omega, ?_⟩, ?_⟩
      · intro j hj
        have ha : a.val = (out.val.set i4.val (i2 &&& 15#u8)).set i9.val i10 := by
          rw [a_post, out1_post, Array.set_val_eq, Array.set_val_eq]
        rw [ha, getD_set]
        by_cases h9 : j = i9.val
        · subst j
          rw [if_pos ⟨rfl, hlen9⟩, hi9, l1_post, part_high code bv.val l.val (by omega), i10_post]
          have h7 : i7.val = (qs.val.getD (32 * bv.val + l.val) (0#u8 : Std.U8)).val := by
            have hb7 : i6.val < qs.val.length := by rw [hi6]; omega
            rw [i7_post, ← hi6]
            exact getD_index qs i6 hb7
          rw [h7, nat_shr_4]
          exact (hread bv.val l.val hb hL).2
        · rw [if_neg (fun h => h9 h.1), getD_set]
          by_cases h4 : j = i4.val
          · subst j
            rw [if_pos ⟨rfl, hlen4⟩, hi4, l1_post, part_low code bv.val l.val (by omega)]
            have h2 : i2.val = (qs.val.getD (32 * bv.val + l.val) (0#u8 : Std.U8)).val := by
              have hb2 : i1.val < qs.val.length := by rw [hi1]; omega
              rw [i2_post, ← hi1]
              exact getD_index qs i1 hb2
            rw [UScalar.val_and, h2]
            exact (hread bv.val l.val hb hL).1
          · rw [if_neg (fun h => h4 h.1), l1_post,
              part_step code bv.val l.val j
                (by intro h; exact h4 (by rw [hi4]; exact h))
                (by intro h; exact h9 (by rw [hi9]; exact h))]
            exact hinv'.2.2 j hj
      · simp only [innerRank]
        omega
    · rw [if_neg hl]
      have hl32 : l.val = 32 := by have := hinv'.2.1; scalar_tac
      simpa [hl32] using hinv'
  · exact hinv

/-- One guarded outer iteration: exactly one inner run of the 32 positions of group `b`, then the
    group rank advances by one. -/
theorem outer_step (qs : Slice Std.U8) (out : Array Std.U8 256#usize) (bv : Std.Usize)
    (code : Nat → Nat) (hb : bv.val < 4) (hqs : 128 ≤ qs.val.length)
    (hinv : acc code bv.val 0 out)
    (hread : ∀ b l : Nat, b < 4 → l < 32 →
      (qs.val.getD (32 * b + l) (0#u8 : Std.U8)).val &&& 15 = code (64 * b + l) ∧
      (qs.val.getD (32 * b + l) (0#u8 : Std.U8)).val / 16 = code (64 * b + 32 + l)) :
    quant.q4_k_codes_loop0.body qs out bv ⦃ r =>
      match r with
      | .done _ => False
      | .cont p => p.2.val = bv.val + 1 ∧ acc code bv.val 32 p.1 ∧ acc code (bv.val + 1) 0 p.1 ⦄ := by
  unfold quant.q4_k_codes_loop0.body
  rw [if_pos (show bv < 4#usize by scalar_tac)]
  refine WP.spec_bind (inner_spec qs out bv 0#usize code hb hqs hinv hread) ?_
  intro out1 hout1
  step*
  refine ⟨b1_post, hout1, ?_⟩
  refine ⟨by omega, by norm_num, ?_⟩
  intro i hi
  rw [hout1.2.2 i hi, part_bump]

/-- Outer loop work: at most `4 - b` guarded iterations, each one inner run of at most 32
    two-code writes, and the four groups are all written at the end. -/
theorem outer_spec (qs : Slice Std.U8) (out0 : Array Std.U8 256#usize) (bv : Std.Usize)
    (code : Nat → Nat) (hqs : 128 ≤ qs.val.length) (hinv : acc code bv.val 0 out0)
    (hread : ∀ b l : Nat, b < 4 → l < 32 →
      (qs.val.getD (32 * b + l) (0#u8 : Std.U8)).val &&& 15 = code (64 * b + l) ∧
      (qs.val.getD (32 * b + l) (0#u8 : Std.U8)).val / 16 = code (64 * b + 32 + l)) :
    quant.q4_k_codes_loop0 qs out0 bv ⦃ out => acc code 4 0 out ⦄ := by
  unfold quant.q4_k_codes_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Array Std.U8 256#usize × Std.Usize => outerRank st.2)
    (inv := fun st => acc code st.2.val 0 st.1)
    (post := fun out => acc code 4 0 out)
  · rintro ⟨out, b⟩ hinv'
    dsimp only
    unfold quant.q4_k_codes_loop0.body
    by_cases hb4 : b < 4#usize
    · rw [if_pos hb4]
      have hbs : b.val < 4 := by scalar_tac
      refine WP.spec_bind (inner_spec qs out b 0#usize code hbs hqs hinv' hread) ?_
      intro out1 hout1
      step*
      refine ⟨?_, ?_⟩
      · rw [b1_post]
        refine ⟨by omega, by norm_num, ?_⟩
        intro i hi
        rw [hout1.2.2 i hi, part_bump]
      · simp only [outerRank]
        omega
    · rw [if_neg hb4]
      have hb4' : b.val = 4 := by have := hinv'.1; scalar_tac
      simpa [hb4'] using hinv'
  · exact hinv

/-! ## Code slots and the work of one block

The two writes of an iteration land on `(64b + l, 64b + 32 + l)`. Over `b < 4`, `l < 32` those pairs
cover the 256 slots of the code array exactly once, so an accepted block costs one accumulator
write per slot. -/

/-- Every one of the 256 output positions is the low or the high slot of a group/position pair. -/
theorem positions_cover (i : Nat) (hi : i < 256) :
    ∃ b l : Nat, b < 4 ∧ l < 32 ∧ (i = 64 * b + l ∨ i = 64 * b + 32 + l) := by
  have hb : i / 64 < 4 := by omega
  by_cases h : i % 64 < 32
  · exact ⟨i / 64, i % 64, hb, h, Or.inl (by omega)⟩
  · refine ⟨i / 64, i % 64 - 32, hb, ?_, Or.inr (by omega)⟩
    omega

/-- Distinct group/position pairs write distinct output positions. -/
theorem positions_unique (i : Nat) (b l b' l' : Nat)
    (h1 : b < 4 ∧ l < 32 ∧ (i = 64 * b + l ∨ i = 64 * b + 32 + l))
    (h2 : b' < 4 ∧ l' < 32 ∧ (i = 64 * b' + l' ∨ i = 64 * b' + 32 + l')) :
    b = b' ∧ l = l' := by
  rcases h1 with ⟨hb, hl, h1a | h1b⟩
  · rcases h2 with ⟨hb', hl', h2a | h2b⟩ <;> omega
  · rcases h2 with ⟨hb', hl', h2a | h2b⟩ <;> omega

/-- Work of one accepted Q4_K block: `4` groups of `32` positions, two codes per position, which
    is the whole code array. The arithmetic is checked here. The reading of the two ranks as an
    iteration *count* is a `model` deduction from `innerRank_step`, `outerRank_step` and the two
    exit theorems, not a kernel-counted iteration; `positions_cover` and `positions_unique` make
    the consequence exact, one accumulator write per slot. -/
theorem codesPerBlock : 4 * 32 * 2 = quant.QK_K.val := by
  norm_num [quant.QK_K]

/-- The accumulator writes charged to one call of the real body: the refusal branch charges
    nothing, the accepting branch charges one write per code slot. This is the `model` reading;
    the `proved` content is `refuses` for the zero branch and `codesPerBlock` with
    `positions_cover`/`positions_unique` for the 256. -/
def blockWork (h : Nat) : Nat := if h < 144 then 0 else 256

/-- Refused call: no accumulator write is charged. -/
theorem blockWork_refuses (h : Nat) (hh : h < 144) : blockWork h = 0 := by
  have hlt : h < 144 := hh
  simp [blockWork, hlt]

/-- Accepted call: one write per code slot, the whole array. -/
theorem blockWork_accepts (h : Nat) (hh : 144 ≤ h) :
    blockWork h = quant.QK_K.val := by
  have hnlt : ¬ (h < 144) := by omega
  simp [blockWork, hnlt, quant.QK_K]

/-! ## Reachable success witness

A 144-byte block whose `qs` window is all `0xF5` is a reachable input to the real body, and its
codes are the low nibbles `5` on the first half of each group and the high nibbles `15` on the
second. Every slot is written, so the witness is not vacuous. -/

lemma getD_rep_drop (j : Nat) (h : 16 + j < 144) :
    ((List.replicate 144 (0xF5#u8 : Std.U8)).drop 16).getD j (0#u8 : Std.U8) = 0xF5#u8 := by
  simp only [List.getD, List.getElem?_drop]
  rw [List.getElem?_replicate, if_pos h]
  rfl

lemma codeOf_low (b l : Nat) (hl : l < 32) : codeOf (64 * b + l) = 5 := by
  unfold codeOf
  rw [show (64 * b + l) % 64 = l by omega, if_pos hl]

lemma codeOf_high (b l : Nat) (hl : l < 32) : codeOf (64 * b + 32 + l) = 15 := by
  unfold codeOf
  rw [show (64 * b + 32 + l) % 64 = 32 + l by omega, if_neg (by omega : ¬ (32 + l < 32))]

/-- The codes of a reachable block are 4-bit values. -/
theorem codes_4bit (i : Nat) : codeOf i < 16 := by
  unfold codeOf
  split <;> norm_num

/-- Reachable success: on a 144-byte block of `0xF5` the real body returns `Ok` with all 256 code
    slots written, low nibbles first. Each slot is nonzero, so no slot is left at the
    constructor's zero. -/
theorem success_witness :
    ∃ blk : Slice Std.U8, ∃ v : Array Std.U8 256#usize,
      word_q4_k_codes blk = ok (core.result.Result.Ok v) ∧
      (∀ i, i < 256 → (v.val.getD i (0#u8 : Std.U8)).val = codeOf i) := by
  set blk := Slice.from (List.replicate 144 (0xF5#u8 : Std.U8)) (by scalar_tac) with hblk
  have hlen : 144#usize ≤ Slice.len blk := by norm_num [hblk]
  have hblklen : blk.length = 144 := by simp [hblk]
  have hspec : word_q4_k_codes blk ⦃ r =>
      ∃ v : Array Std.U8 256#usize, r = (core.result.Result.Ok v) ∧
        (∀ i, i < 256 → (v.val.getD i (0#u8 : Std.U8)).val = codeOf i) ⦄ := by
    rw [word_q4_k_codes]
    unfold quant.q4_k_codes
    rw [blockBytes]
    simp only [bind_tc_ok, quant.need]
    rw [if_neg (by scalar_tac : ¬ (Slice.len blk < 144#usize))]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    simp only [quant.Q4_K_SCALE_SIZE]
    step*
    have hread : ∀ b l : Nat, b < 4 → l < 32 →
        (qs.val.getD (32 * b + l) (0#u8 : Std.U8)).val &&& 15 = codeOf (64 * b + l) ∧
        (qs.val.getD (32 * b + l) (0#u8 : Std.U8)).val / 16 = codeOf (64 * b + 32 + l) := by
      intro b l hb hl
      have hq : qs.val = (List.replicate 144 (0xF5#u8 : Std.U8)).drop 16 := by
        rw [qs_post, i1_post, hblk, Slice.from_val]
      have hv : (0xF5#u8 : Std.U8).val = 245 := by rfl
      constructor
      · rw [hq, getD_rep_drop _ (by omega), hv, nat_and_15, codeOf_low b l hl]
      · rw [hq, getD_rep_drop _ (by omega), hv, codeOf_high b l hl]
    refine WP.spec_bind
      (outer_spec qs (Array.repeat 256#usize (0#u8 : Std.U8)) 0#usize codeOf ?_ ?_ hread) ?_
    · change 128 ≤ qs.length
      rw [qs_post1]
      have hlen' : 144 ≤ blk.length := by omega
      omega
    · exact acc_repeat codeOf
    · intro out1 hout1
      refine ⟨out1, rfl, ?_⟩
      intro i hi
      rw [hout1.2.2 i hi, part_full codeOf i hi]
  obtain ⟨r, hr, hv⟩ := WP.spec_imp_exists hspec
  obtain ⟨v, hrv, hpt⟩ := hv
  refine ⟨blk, v, ?_, hpt⟩
  rw [hr, hrv]

/-- Every code slot of the reachable witness carries a nonzero 4-bit code, so the 256 writes are
    not vacuous. -/
theorem codes_written (v : Array Std.U8 256#usize)
    (h : ∀ i, i < 256 → (v.val.getD i (0#u8 : Std.U8)).val = codeOf i) :
    ∀ i, i < 256 → 0 < (v.val.getD i (0#u8 : Std.U8)).val ∧
      (v.val.getD i (0#u8 : Std.U8)).val < 16 := by
  intro i hi
  rw [h i hi]
  unfold codeOf
  split <;> norm_num

end Q4KCodesResource

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit Q4KCodesResource.blockBytes
#resource_audit Q4KCodesResource.refuses
#resource_audit Q4KCodesResource.refusal_witness
#resource_audit Q4KCodesResource.rank_starts
#resource_audit Q4KCodesResource.innerRank_step
#resource_audit Q4KCodesResource.outerRank_step
#resource_audit Q4KCodesResource.inner_exit
#resource_audit Q4KCodesResource.outer_exit
#resource_audit Q4KCodesResource.inner_step
#resource_audit Q4KCodesResource.inner_spec
#resource_audit Q4KCodesResource.outer_step
#resource_audit Q4KCodesResource.outer_spec
#resource_audit Q4KCodesResource.positions_cover
#resource_audit Q4KCodesResource.positions_unique
#resource_audit Q4KCodesResource.codesPerBlock
#resource_audit Q4KCodesResource.blockWork_refuses
#resource_audit Q4KCodesResource.blockWork_accepts
#resource_audit Q4KCodesResource.codes_4bit
#resource_audit Q4KCodesResource.success_witness
#resource_audit Q4KCodesResource.codes_written

run_cmd Lean.logInfo "Q4K_CODES_KERNEL_OK"
