/-
Integer refinement of the extracted Q4_K and Q6_K code unpackers
(`APEIRON-INT-FORMAT-W3-T3`; spec `docs/epics/inference-parity.md`).

The Rust bodies are `src/format/quant.rs`: `q4_k_codes` (block layout `f16 d; f16 dmin;
u8 scales[12]; u8 qs[128]`, 144 bytes) and `q6_k_codes` (`u8 ql[128]; u8 qh[64];
i8 scales[16]; f16 d`, 210 bytes). Both are extracted into `ApeironWord` by `proof/regen.sh` as
`quant.q4_k_codes` / `quant.q6_k_codes` and rooted as `word_q4_k_codes` / `word_q6_k_codes`.

This module states the reference integer unpack over the block bytes and proves, exactly:
the refusal correspondence (a block shorter than the documented length returns
`QuantError.Truncated`) and, for Q4_K, the positive byte-exact equality
(`word_q4_k_codes_ok`: success yields an array equal to `q4CodesRef` pointwise). The Q4_K equality
is proved through the `loop.spec_decr_nat` invariants `q4_inner_spec`/`q4_outer_spec`. The Q6_K
positive equality is the remaining residual; its reference and refusal are landed.

No floats are involved: the code unpack is the integer layer that the float scaling
(`dequant_*_block`) consumes.
-/
import ApeironWord

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Reference integer unpack

The layouts are the reference's own; the offsets name the packed fields rather than restating
the arithmetic. `q4CodesRef` reads the 4-bit `qs` nibbles: for output index `i`, block counter
`b = i / 64`, in-block offset `r = i % 64`, the low 32 outputs take `qs[32·b + r] & 0xF` and the
high 32 take `qs[32·b + (r - 32)] >> 4`, where `qs` begins at byte 16. `q6CodesRef` folds the
6-bit codes from the low plane `ql` (bytes `0..128`) and the high plane `qh` (bytes `128..192`):
the three-segment split `seg = (i % 128) / 32` selects nibble, `ql` sub-word and the two high
bits of `qh`, and the result is offset by `-32` into the signed range `[-32, 31]`. -/

/-- Q4_K 4-bit reference code `i` as a natural (values `< 16`). -/
def q4CodesRef (blk : List Std.U8) (i : Nat) : Nat :=
  let b := i / 64
  let r := i % 64
  let q := (blk.getD (16 + 32 * b + r % 32) 0#u8).val
  if r < 32 then q % 16 else q / 16

/-- Q6_K signed 6-bit reference code `i` as an integer (values in `[-32, 31]`).

Stated with the same `&&&`/`|||`/`<<<`/`>>>` operations the extracted body uses, so the write
value and the reference meet syntactically. -/
def q6CodesRef (blk : List Std.U8) (i : Nat) : Int :=
  (((if i % 128 / 32 < 2
      then (blk.getD (64 * (i / 128) + i % 128 % 32
              + (if i % 128 / 32 % 2 = 1 then 32 else 0)) 0#u8).val &&& 15
      else (blk.getD (64 * (i / 128) + i % 128 % 32
              + (if i % 128 / 32 % 2 = 1 then 32 else 0)) 0#u8).val >>> 4)
    ||| (((blk.getD (128 + 32 * (i / 128) + i % 128 % 32) 0#u8).val
            >>> (2 * (i % 128 / 32)) &&& 3) <<< 4)
    : Nat) : Int) - 32

/-! ## Refusal correspondence

`quant.need` is the length gate: `Slice.len blk < n` returns
`Err (Truncated n (Slice.len blk))`. Both unpackers pass `Q4_K_BLOCK_BYTES` / `Q6_K_BLOCK_BYTES`,
so a short block refuses with that exact error; nothing is read or truncated silently.

Those two constants are extracted as `Result Std.Usize` (their Rust bodies are checked
additions), so their evaluated values are established first. -/

/-- The Q4_K block length evaluates to 144 bytes. -/
theorem quant_Q4_K_BLOCK_BYTES_eq : quant.Q4_K_BLOCK_BYTES = ok (144#usize) := by
  have h : quant.Q4_K_BLOCK_BYTES ⦃ z => z = (144#usize) ⦄ := by
    unfold quant.Q4_K_BLOCK_BYTES
    simp only [quant.QK_K]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The Q6_K block length evaluates to 210 bytes. -/
theorem quant_Q6_K_BLOCK_BYTES_eq : quant.Q6_K_BLOCK_BYTES = ok (210#usize) := by
  have h : quant.Q6_K_BLOCK_BYTES ⦃ z => z = (210#usize) ⦄ := by
    unfold quant.Q6_K_BLOCK_BYTES
    simp only [quant.QK_K]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- A short Q4_K block refuses with `Truncated` carrying the required and available lengths. -/
theorem word_q4_k_codes_refuses (blk : Slice Std.U8)
    (h : Slice.len blk < 144#usize) :
    word_q4_k_codes blk =
      ok (core.result.Result.Err
        (quant.QuantError.Truncated 144#usize (Slice.len blk))) := by
  rw [word_q4_k_codes]
  unfold quant.q4_k_codes
  rw [quant_Q4_K_BLOCK_BYTES_eq]
  simp only [bind_tc_ok, quant.need, Slice.len]
  rw [if_pos h]
  simp [core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]

/-- A short Q6_K block refuses with `Truncated` carrying the required and available lengths. -/
theorem word_q6_k_codes_refuses (blk : Slice Std.U8)
    (h : Slice.len blk < 210#usize) :
    word_q6_k_codes blk =
      ok (core.result.Result.Err
        (quant.QuantError.Truncated 210#usize (Slice.len blk))) := by
  rw [word_q6_k_codes]
  unfold quant.q6_k_codes
  rw [quant_Q6_K_BLOCK_BYTES_eq]
  simp only [bind_tc_ok, quant.need, Slice.len]
  rw [if_pos h]
  simp [core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]


/-! ## Q4_K loop invariant scaffolding (APEIRON-INT-FORMAT-W3-T6)

Proven building blocks for the positive equality, not the theorem itself. `q4QsRef` is the
reference on the post-offset `qs` bytes; `q4Val qs b l i` is the partial reference after blocks
`0..b-1` are written and the first `l` low/high entries of block `b` are written; `q4Inv` states
the accumulator equals it pointwise. The lemmas below discharge the `step*` obligation for the
first written index (`q4Val_low`), the second (`q4Val_high`), and every unwritten index
(`q4Val_step`), plus the `List.getD`/`set` interaction (`getD_set`) and the index/bit arithmetic
(`q4QsRef_low`, `q4QsRef_high`). `q4_inner_spec`/`q4_outer_spec` wrap them with
`loop.spec_decr_nat` and `word_q4_k_codes_ok` assembles the top-level equality.
-/

def q4QsRef (qs : List Std.U8) (i : Nat) : Nat :=
  if i % 64 < 32 then (qs.getD (32 * (i / 64) + (i % 64) % 32) 0#u8).val % 16
  else (qs.getD (32 * (i / 64) + (i % 64) % 32) 0#u8).val / 16

def q4Val (qs : List Std.U8) (b l i : Nat) : Nat :=
  if i < 64 * b then q4QsRef qs i
  else if i / 64 = b ∧ (i % 64 < l ∨ (32 ≤ i % 64 ∧ i % 64 < 32 + l)) then q4QsRef qs i
  else 0

def q4Inv (qs : List Std.U8) (b l : Nat) (out : Array Std.U8 256#usize) : Prop :=
  b ≤ 4 ∧ l ≤ 32 ∧ ∀ i, i < 256 → (out.val.getD i 0#u8).val = q4Val qs b l i

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

lemma q4div (b x : Nat) (hx : x < 64) : (64 * b + x) / 64 = b := by omega
lemma q4mod (b x : Nat) (hx : x < 64) : (64 * b + x) % 64 = x := by omega
lemma q4mod32 (x : Nat) (hx : x < 32) : x % 32 = x := by omega

lemma q4QsRef_low (qs : List Std.U8) (b l : Nat) (hl : l < 32) :
    q4QsRef qs (64 * b + l) = (qs.getD (32 * b + l) 0#u8).val % 16 := by
  unfold q4QsRef
  rw [q4div b l (by omega), q4mod b l (by omega), q4mod32 l hl, if_pos hl]

lemma q4QsRef_high (qs : List Std.U8) (b l : Nat) (hl : l < 32) :
    q4QsRef qs (64 * b + 32 + l) = (qs.getD (32 * b + l) 0#u8).val / 16 := by
  unfold q4QsRef
  have hd : (64 * b + 32 + l) / 64 = b := by omega
  have h32 : (64 * b + 32 + l) % 64 = 32 + l := by omega
  rw [hd, h32]
  have hmod : (32 + l) % 32 = l := by omega
  rw [hmod, if_neg (by omega : ¬ (32 + l < 32))]

lemma q4Val_step (qs : List Std.U8) (b l : Nat) (i : Nat)
    (h1 : i ≠ 64 * b + l) (h2 : i ≠ 64 * b + 32 + l) :
    q4Val qs b (l + 1) i = q4Val qs b l i := by
  unfold q4Val
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

lemma q4Val_low (qs : List Std.U8) (b l : Nat) (hl : l < 32) :
    q4Val qs b (l + 1) (64 * b + l) = q4QsRef qs (64 * b + l) := by
  unfold q4Val
  rw [if_neg (by omega : ¬ (64 * b + l < 64 * b)), if_pos ⟨by omega, Or.inl (by omega)⟩]

lemma q4Val_high (qs : List Std.U8) (b l : Nat) (hl : l < 32) :
    q4Val qs b (l + 1) (64 * b + 32 + l) = q4QsRef qs (64 * b + 32 + l) := by
  unfold q4Val
  rw [if_neg (by omega : ¬ (64 * b + 32 + l < 64 * b)),
    if_pos ⟨by omega, Or.inr ⟨by omega, by omega⟩⟩]

lemma q4_getD_getElem (qs : Slice Std.U8) (i : Std.Usize) (h : i.val < qs.val.length) :
    (qs.val[i.val]'h).val = (qs.val.getD i.val 0#u8).val := by
  rw [List.getD, List.getElem?_eq_getElem h]
  rfl

lemma nat_and_15 (x : Nat) : x &&& 15 = x % 16 := by
  simpa using Nat.and_two_pow_sub_one_eq_mod x 4

lemma nat_shr_4 (x : Nat) : x >>> 4 = x / 16 := by
  simpa using Nat.shiftRight_eq_div_pow x 4

theorem q4_inner_spec (qs : Slice Std.U8) (out0 : Array Std.U8 256#usize) (bv lv : Std.Usize)
    (hb : bv.val < 4) (hqs : 128 ≤ qs.val.length) (hinv : q4Inv qs.val bv.val lv.val out0) :
    quant.q4_k_codes_loop0_loop0 qs out0 bv lv ⦃ out => q4Inv qs.val bv.val 32 out ⦄ := by
  unfold quant.q4_k_codes_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Array Std.U8 256#usize × Std.Usize => 32 - st.2.val)
    (inv := fun st => q4Inv qs.val bv.val st.2.val st.1)
    (post := fun out => q4Inv qs.val bv.val 32 out)
  · rintro ⟨out, l⟩ hinv'
    dsimp only
    unfold quant.q4_k_codes_loop0_loop0.body
    by_cases hl : l < 32#usize
    · rw [if_pos hl]
      simp only [lift, bind_tc_ok]
      step*
      have hL : l.val < 32 := by scalar_tac
      have hB : bv.val < 4 := hb
      have hi4 : i4.val = 64 * bv.val + l.val := by rw [i4_post, i3_post]
      have hi9 : i9.val = 64 * bv.val + 32 + l.val := by rw [i9_post, i8_post, i3_post]
      have hi1 : i1.val = 32 * bv.val + l.val := by rw [i1_post, i_post]
      have hi6 : i6.val = 32 * bv.val + l.val := by rw [i6_post, i_post]
      have hout : out.val.length = 256 := by exact Array.property out
      have hlen4 : i4.val < out.val.length := by rw [hi4, hout]; omega
      have hlen9 : i9.val < (out.val.set i4.val (i2 &&& 15#u8)).length := by
        rw [List.length_set, hi9, hout]; omega
      refine ⟨⟨hinv'.1, by omega, ?_⟩, by omega⟩
      intro j hj
      have ha : a.val = (out.val.set i4.val (i2 &&& 15#u8)).set i9.val i10 := by
        rw [a_post, out1_post, Array.set_val_eq, Array.set_val_eq]
      rw [ha, getD_set]
      by_cases h9 : j = i9.val
      · subst j
        rw [if_pos ⟨rfl, hlen9⟩, hi9, l1_post, q4Val_high qs.val bv.val l.val hL,
          q4QsRef_high qs.val bv.val l.val hL, i10_post]
        have h7 : i7.val = (qs.val.getD (32 * bv.val + l.val) 0#u8).val := by
          have hb7 : i6.val < qs.val.length := by rw [hi6]; omega
          rw [i7_post, ← hi6]
          exact q4_getD_getElem qs i6 hb7
        rw [h7, nat_shr_4]
      · rw [if_neg (fun h => h9 h.1), getD_set]
        by_cases h4 : j = i4.val
        · subst j
          rw [if_pos ⟨rfl, hlen4⟩, hi4, l1_post, q4Val_low qs.val bv.val l.val hL,
            q4QsRef_low qs.val bv.val l.val hL]
          have h2 : i2.val = (qs.val.getD (32 * bv.val + l.val) 0#u8).val := by
            have hb2 : i1.val < qs.val.length := by rw [hi1]; omega
            rw [i2_post, ← hi1]
            exact q4_getD_getElem qs i1 hb2
          rw [UScalar.val_and, h2]
          show (qs.val.getD (32 * bv.val + l.val) 0#u8).val &&& 15
            = (qs.val.getD (32 * bv.val + l.val) 0#u8).val % 16
          exact nat_and_15 _
        · rw [if_neg (fun h => h4 h.1), l1_post,
            q4Val_step qs.val bv.val l.val j
              (by intro h; exact h4 (by rw [hi4]; exact h))
              (by intro h; exact h9 (by rw [hi9]; exact h))]
          exact hinv'.2.2 j hj
    · rw [if_neg hl]
      have hl32 : l.val = 32 := by have := hinv'.2.1; scalar_tac
      simpa [hl32] using hinv'
  · exact hinv

lemma q4Val_succ (qs : List Std.U8) (b : Nat) (i : Nat) :
    q4Val qs b 32 i = q4Val qs (b + 1) 0 i := by
  unfold q4Val
  have hm : i % 64 < 64 := Nat.mod_lt _ (by norm_num)
  by_cases h1 : i / 64 < b
  · rw [if_pos (by omega : i < 64 * b), if_pos (by omega : i < 64 * (b + 1))]
  · by_cases h2 : i / 64 = b
    · rw [if_neg (by omega : ¬ i < 64 * b), if_pos (by omega : i < 64 * (b + 1))]
      rw [if_pos ⟨h2, by omega⟩]
    · rw [if_neg (by omega : ¬ i < 64 * b), if_neg (by omega : ¬ i < 64 * (b + 1))]
      rw [if_neg (fun hh => h2 hh.1)]
      rw [if_neg (fun hh => by rcases hh.2 with h | h <;> omega)]

lemma q4Inv_succ (qs : List Std.U8) (b : Nat) (out : Array Std.U8 256#usize)
    (hb : b < 4) (h : q4Inv qs b 32 out) : q4Inv qs (b + 1) 0 out := by
  refine ⟨by omega, by omega, ?_⟩
  intro i hi
  rw [h.2.2 i hi]
  exact q4Val_succ qs b i

lemma q4QsRef_eq_q4CodesRef (blk : List Std.U8) (i : Nat) :
    q4CodesRef blk i = q4QsRef (blk.drop 16) i := by
  have hidx : 16 + 32 * (i / 64) + (i % 64) % 32
      = 16 + (32 * (i / 64) + (i % 64) % 32) := by omega
  simp only [q4CodesRef, q4QsRef, List.getD, List.getElem?_drop]
  rw [hidx]

theorem q4_outer_spec (qs : Slice Std.U8) (out0 : Array Std.U8 256#usize) (bv : Std.Usize)
    (hqs : 128 ≤ qs.val.length) (hinv : q4Inv qs.val bv.val 0 out0) :
    quant.q4_k_codes_loop0 qs out0 bv ⦃ out => q4Inv qs.val 4 0 out ⦄ := by
  unfold quant.q4_k_codes_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Array Std.U8 256#usize × Std.Usize => 4 - st.2.val)
    (inv := fun st => q4Inv qs.val st.2.val 0 st.1)
    (post := fun out => q4Inv qs.val 4 0 out)
  · rintro ⟨out, b⟩ hinv'
    dsimp only
    unfold quant.q4_k_codes_loop0.body
    by_cases hb4 : b < 4#usize
    · rw [if_pos hb4]
      have hbs : b.val < 4 := by scalar_tac
      refine WP.spec_bind (q4_inner_spec qs out b 0#usize hbs hqs hinv') ?_
      intro out1 hout1
      step*
      refine ⟨?_, by scalar_tac⟩
      rw [b1_post]
      exact q4Inv_succ qs b.val out1 hbs hout1
    · rw [if_neg hb4]
      have hb4' : b.val = 4 := by have := hinv'.1; scalar_tac
      simpa [hb4'] using hinv'
  · exact hinv

lemma q4Val_zero (qs : List Std.U8) (i : Nat) : q4Val qs 0 0 i = 0 := by
  unfold q4Val
  rw [if_neg (by omega : ¬ i < 64 * 0)]
  rw [if_neg (fun h => by rcases h.2 with h | h <;> omega)]

lemma q4Val_four (qs : List Std.U8) (i : Nat) (hi : i < 256) : q4Val qs 4 0 i = q4QsRef qs i := by
  unfold q4Val
  rw [if_pos (by omega : i < 64 * 4)]

lemma q4Inv_repeat (qs : List Std.U8) :
    q4Inv qs 0 0 (Array.repeat 256#usize 0#u8) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro i hi
  rw [q4Val_zero, Array.repeat_val]
  simp only [List.getD, List.getElem?_replicate]
  rw [if_pos (by simpa using hi)]
  rfl

theorem word_q4_k_codes_ok (blk : Slice Std.U8) (hlen : 144#usize ≤ Slice.len blk) :
    ∃ v : Array Std.U8 256#usize,
      word_q4_k_codes blk = ok (core.result.Result.Ok v) ∧
      ∀ i, i < 256 → (v.val.getD i 0#u8).val = q4CodesRef blk.val i := by
  have hspec : word_q4_k_codes blk ⦃ r => ∃ v : Array Std.U8 256#usize,
      r = (core.result.Result.Ok v : core.result.Result (Array Std.U8 256#usize)
        quant.QuantError) ∧
      ∀ i, i < 256 → (v.val.getD i 0#u8).val = q4QsRef (blk.val.drop 16) i ⦄ := by
    rw [word_q4_k_codes]
    unfold quant.q4_k_codes
    rw [quant_Q4_K_BLOCK_BYTES_eq]
    simp only [bind_tc_ok, quant.need]
    rw [if_neg (by scalar_tac : ¬ (Slice.len blk < 144#usize))]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    simp only [quant.Q4_K_SCALE_SIZE]
    step*
    refine WP.spec_bind (q4_outer_spec qs (Array.repeat 256#usize 0#u8) 0#usize ?_ ?_) ?_
    · change 128 ≤ qs.length
      rw [qs_post1, i1_post]
      have hlen' : 144 ≤ blk.length := by have := hlen; scalar_tac
      omega
    · exact q4Inv_repeat qs.val
    · intro out1 hout1
      refine ⟨out1, rfl, ?_⟩
      intro i hi
      rw [hout1.2.2 i hi, q4Val_four qs.val i hi, qs_post, i1_post]
  rcases WP.spec_imp_exists hspec with ⟨r, hr, hv⟩
  rcases hv with ⟨v, hrv, hpt⟩
  refine ⟨v, ?_, ?_⟩
  · rw [hr, hrv]
  · intro i hi
    rw [hpt i hi]
    exact (q4QsRef_eq_q4CodesRef blk.val i).symm

/-! ## Q6_K loop invariant scaffolding

`q6QsRef ql qh i` is the reference code on the post-offset planes (`ql` = bytes `0..128`,
`qh` = bytes `128..192`); `q6Val ql qh n l i` adds the partial-write state after `n` full
128-weight halves and the first `l` positions of half `n`; `q6Inv` states the `Array Std.I8`
accumulator equals it pointwise. -/

/-- Q6_K reference code on the post-offset `ql`/`qh` planes, using the extracted operations. -/
def q6QsRef (ql qh : List Std.U8) (i : Nat) : Int :=
  (((if i % 128 / 32 < 2
      then (ql.getD (64 * (i / 128) + i % 128 % 32
              + (if i % 128 / 32 % 2 = 1 then 32 else 0)) 0#u8).val &&& 15
      else (ql.getD (64 * (i / 128) + i % 128 % 32
              + (if i % 128 / 32 % 2 = 1 then 32 else 0)) 0#u8).val >>> 4)
    ||| (((qh.getD (32 * (i / 128) + i % 128 % 32) 0#u8).val
            >>> (2 * (i % 128 / 32)) &&& 3) <<< 4)
    : Nat) : Int) - 32

/-- Partial Q6_K reference after `n` full halves and `l` written positions of half `n`. -/
def q6Val (ql qh : List Std.U8) (n l i : Nat) : Int :=
  if i < 128 * n then q6QsRef ql qh i
  else if i / 128 = n ∧ (i % 128 < l ∨ (32 ≤ i % 128 ∧ i % 128 < 32 + l)
      ∨ (64 ≤ i % 128 ∧ i % 128 < 64 + l)
      ∨ (96 ≤ i % 128 ∧ i % 128 < 96 + l)) then q6QsRef ql qh i
  else 0

/-- The Q6_K accumulator invariant. -/
def q6Inv (ql qh : List Std.U8) (n l : Nat) (out : Array Std.I8 256#usize) : Prop :=
  n ≤ 2 ∧ l ≤ 32 ∧ ∀ i, i < 256 → (out.val.getD i 0#i8).val = q6Val ql qh n l i

lemma getD_take {α : Type} (l : List α) (b k : Nat) (d : α) (hk : k < b) :
    (l.take b).getD k d = l.getD k d := by
  simp only [List.getD, List.getElem?_take, if_pos hk]

lemma getD_drop_take {α : Type} (l : List α) (a b k : Nat) (d : α) (hk : k < b) :
    ((l.drop a).take b).getD k d = l.getD (a + k) d := by
  simp only [List.getD, List.getElem?_take, if_pos hk, List.getElem?_drop]

lemma q6QsRef_eq_q6CodesRef (blk : List Std.U8) (i : Nat) (hi : i < 256) :
    q6CodesRef blk i = q6QsRef (blk.take 128) ((blk.drop 128).take 64) i := by
  have hql : 64 * (i / 128) + i % 128 % 32
      + (if i % 128 / 32 % 2 = 1 then 32 else 0) < 128 := by
    have h1 : i / 128 ≤ 1 := by omega
    have h2 : i % 128 % 32 < 32 := Nat.mod_lt _ (by norm_num)
    split <;> omega
  have hqh : 32 * (i / 128) + i % 128 % 32 < 64 := by
    have h1 : i / 128 ≤ 1 := by omega
    have h2 : i % 128 % 32 < 32 := Nat.mod_lt _ (by norm_num)
    omega
  have hidx : 128 + 32 * (i / 128) + i % 128 % 32
      = 128 + (32 * (i / 128) + i % 128 % 32) := by omega
  unfold q6CodesRef q6QsRef
  rw [getD_take blk 128 _ 0#u8 hql, getD_drop_take blk 128 64 _ 0#u8 hqh, hidx]


lemma hcast_u8_I32_val (x : Std.U8) :
    (UScalar.hcast IScalarTy.I32 x).val = (x.val : ℤ) := by
  have hx : (x.val : ℤ) < (2 ^ 32 : ℕ) := by
    have := x.hBounds
    norm_num [IScalar.size, UScalarTy.numBits] at *
    omega
  have h0 : (0 : ℤ) ≤ (x.val : ℤ) := Int.natCast_nonneg _
  rw [UScalar.hcast_val_eq]
  simp only [IScalarTy.numBits]
  rw [Int.bmod_eq_emod_of_lt (by
      rw [Int.emod_eq_of_lt h0 hx]
      have hb := x.hBounds
      norm_num [IScalar.size, UScalarTy.numBits] at hb
      omega),
    Int.emod_eq_of_lt h0 hx]

lemma hcast_sub32_lower (x : Std.U8) :
    I32.min ≤ (UScalar.hcast IScalarTy.I32 x).val - 32 := by
  rw [hcast_u8_I32_val]
  have h0 : (0 : ℤ) ≤ (x.val : ℤ) := Int.natCast_nonneg _
  norm_num [IScalar.min, IScalar.max, I32.min, I32.max, I32.numBits]
  omega

/-- The assembled `nib ||| (hi <<< 4)` value is a valid 6-bit code (`< 64`). -/
lemma q6_combine_lt (a h : Nat) : ((a &&& 15) ||| (((h >>> 0) &&& 3) <<< 4)) < 64 := by
  rw [Nat.shiftLeft_eq]
  have ha : a &&& 15 < 2 ^ 6 := by
    have h := Nat.and_lt_two_pow a (n := 4) (by norm_num : (15 : Nat) < 2 ^ 4)
    omega
  have hb : (h >>> 0 &&& 3) * 2 ^ 4 < 2 ^ 6 := by
    have h := Nat.and_lt_two_pow (h >>> 0) (n := 2) (by norm_num : (3 : Nat) < 2 ^ 2)
    nlinarith
  exact Nat.or_lt_two_pow ha hb

/-- The signed 6-bit code `n - 32` survives the `I8` balanced-mod representation exactly. -/
lemma bmod_sub32 (n : Nat) (hn : n < 64) : Int.bmod ((n : Int) - 32) 256 = (n : Int) - 32 := by
  rw [Int.bmod_eq_emod]
  split <;> omega


lemma q6Val_zero (ql qh : List Std.U8) (i : Nat) : q6Val ql qh 0 0 i = 0 := by
  unfold q6Val
  rw [if_neg (by omega : ¬ i < 128 * 0)]
  rw [if_neg (fun h => by rcases h.2 with h | h | h | h <;> omega)]

lemma q6Val_four (ql qh : List Std.U8) (i : Nat) (hi : i < 256) :
    q6Val ql qh 2 0 i = q6QsRef ql qh i := by
  unfold q6Val
  rw [if_pos (by omega : i < 128 * 2)]

lemma q6Val_step (ql qh : List Std.U8) (n l : Nat) (i : Nat)
    (h1 : i ≠ 128 * n + l) (h2 : i ≠ 128 * n + l + 32)
    (h3 : i ≠ 128 * n + l + 64) (h4 : i ≠ 128 * n + l + 96) :
    q6Val ql qh n (l + 1) i = q6Val ql qh n l i := by
  unfold q6Val
  by_cases hlt : i < 128 * n
  · rw [if_pos hlt, if_pos hlt]
  · rw [if_neg hlt, if_neg hlt]
    by_cases hb : i / 128 = n
    · have hlo : i % 128 ≠ l := by intro h; exact h1 (by omega)
      have h32 : i % 128 ≠ 32 + l := by intro h; exact h2 (by omega)
      have h64 : i % 128 ≠ 64 + l := by intro h; exact h3 (by omega)
      have h96 : i % 128 ≠ 96 + l := by intro h; exact h4 (by omega)
      by_cases hA : i % 128 < l
      · rw [if_pos ⟨hb, Or.inl (by omega : i % 128 < l + 1)⟩]
        rw [if_pos ⟨hb, Or.inl hA⟩]
      · by_cases hB : 32 ≤ i % 128 ∧ i % 128 < 32 + l
        · rw [if_pos ⟨hb, Or.inr (Or.inl ⟨hB.1, by omega⟩)⟩]
          rw [if_pos ⟨hb, Or.inr (Or.inl hB)⟩]
        · by_cases hC : 64 ≤ i % 128 ∧ i % 128 < 64 + l
          · rw [if_pos ⟨hb, Or.inr (Or.inr (Or.inl ⟨hC.1, by omega⟩))⟩]
            rw [if_pos ⟨hb, Or.inr (Or.inr (Or.inl hC))⟩]
          · by_cases hD : 96 ≤ i % 128 ∧ i % 128 < 96 + l
            · rw [if_pos ⟨hb, Or.inr (Or.inr (Or.inr ⟨hD.1, by omega⟩))⟩]
              rw [if_pos ⟨hb, Or.inr (Or.inr (Or.inr hD))⟩]
            · rw [if_neg (fun h => by rcases h.2 with h | h | h | h <;> omega)]
              rw [if_neg (fun h => by
                rcases h.2 with h | h | h | h
                · exact hA h
                · exact hB h
                · exact hC h
                · exact hD h)]
    · rw [if_neg (fun h => hb h.1), if_neg (fun h => hb h.1)]

lemma q6Val_1 (ql qh : List Std.U8) (n l : Nat) (hl : l < 32) :
    q6Val ql qh n (l + 1) (128 * n + l) = q6QsRef ql qh (128 * n + l) := by
  unfold q6Val
  rw [if_neg (by omega : ¬ (128 * n + l < 128 * n)), if_pos ⟨by omega, Or.inl (by omega)⟩]

lemma q6Val_2 (ql qh : List Std.U8) (n l : Nat) (hl : l < 32) :
    q6Val ql qh n (l + 1) (128 * n + l + 32) = q6QsRef ql qh (128 * n + l + 32) := by
  unfold q6Val
  rw [if_neg (by omega : ¬ (128 * n + l + 32 < 128 * n)),
    if_pos ⟨by omega, Or.inr (Or.inl ⟨by omega, by omega⟩)⟩]

lemma q6Val_3 (ql qh : List Std.U8) (n l : Nat) (hl : l < 32) :
    q6Val ql qh n (l + 1) (128 * n + l + 64) = q6QsRef ql qh (128 * n + l + 64) := by
  unfold q6Val
  rw [if_neg (by omega : ¬ (128 * n + l + 64 < 128 * n)),
    if_pos ⟨by omega, Or.inr (Or.inr (Or.inl ⟨by omega, by omega⟩))⟩]

lemma q6Val_4 (ql qh : List Std.U8) (n l : Nat) (hl : l < 32) :
    q6Val ql qh n (l + 1) (128 * n + l + 96) = q6QsRef ql qh (128 * n + l + 96) := by
  unfold q6Val
  rw [if_neg (by omega : ¬ (128 * n + l + 96 < 128 * n)),
    if_pos ⟨by omega, Or.inr (Or.inr (Or.inr ⟨by omega, by omega⟩))⟩]

lemma q6Val_succ (ql qh : List Std.U8) (n : Nat) (i : Nat) :
    q6Val ql qh n 32 i = q6Val ql qh (n + 1) 0 i := by
  unfold q6Val
  by_cases h1 : i / 128 < n
  · rw [if_pos (by omega : i < 128 * n), if_pos (by omega : i < 128 * (n + 1))]
  · by_cases h2 : i / 128 = n
    · rw [if_neg (by omega : ¬ i < 128 * n), if_pos (by omega : i < 128 * (n + 1))]
      rw [if_pos ⟨h2, by omega⟩]
    · rw [if_neg (by omega : ¬ i < 128 * n), if_neg (by omega : ¬ i < 128 * (n + 1))]
      rw [if_neg (fun h => h2 h.1)]
      rw [if_neg (fun h => by rcases h.2 with h | h | h | h <;> omega)]

lemma q6Inv_succ (ql qh : List Std.U8) (n : Nat) (out : Array Std.I8 256#usize)
    (hn : n < 2) (h : q6Inv ql qh n 32 out) : q6Inv ql qh (n + 1) 0 out := by
  refine ⟨by omega, by omega, ?_⟩
  intro i hi
  rw [h.2.2 i hi]
  exact q6Val_succ ql qh n i

lemma q6Inv_repeat (ql qh : List Std.U8) :
    q6Inv ql qh 0 0 (Array.repeat 256#usize 0#i8) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro i hi
  rw [q6Val_zero, Array.repeat_val]
  simp only [List.getD, List.getElem?_replicate]
  rw [if_pos (by simpa using hi)]
  rfl


lemma shift4_mod (x : Nat) (hx : x < 4) : (x <<< 4) % 256 = x <<< 4 := by
  apply Nat.mod_eq_of_lt
  rw [Nat.shiftLeft_eq]
  nlinarith [Nat.two_pow_pos 4]

lemma q6QsRef_idx0 (ql qh : List Std.U8) (n l : Nat) (hl : l < 32) :
    q6QsRef ql qh (128 * n + l)
      = ((((ql.getD (64 * n + l) 0#u8).val &&& 15)
          ||| (((qh.getD (32 * n + l) 0#u8).val >>> 0 &&& 3) <<< 4) : Nat) : Int) - 32 := by
  unfold q6QsRef
  have h1 : (128 * n + l) / 128 = n := by omega
  have h2 : (128 * n + l) % 128 = l := by omega
  have h3 : l / 32 = 0 := by omega
  have h4 : l % 32 = l := by omega
  rw [h1, h2, h3, h4]
  norm_num

lemma q6QsRef_idx1 (ql qh : List Std.U8) (n l : Nat) (hl : l < 32) :
    q6QsRef ql qh (128 * n + l + 32)
      = ((((ql.getD (64 * n + l + 32) 0#u8).val &&& 15)
          ||| (((qh.getD (32 * n + l) 0#u8).val >>> 2 &&& 3) <<< 4) : Nat) : Int) - 32 := by
  unfold q6QsRef
  have h1 : (128 * n + l + 32) / 128 = n := by omega
  have h2 : (128 * n + l + 32) % 128 = l + 32 := by omega
  have h3 : (l + 32) / 32 = 1 := by omega
  have h4 : (l + 32) % 32 = l := by omega
  rw [h1, h2, h3, h4]
  norm_num

lemma q6QsRef_idx2 (ql qh : List Std.U8) (n l : Nat) (hl : l < 32) :
    q6QsRef ql qh (128 * n + l + 64)
      = ((((ql.getD (64 * n + l) 0#u8).val >>> 4)
          ||| (((qh.getD (32 * n + l) 0#u8).val >>> 4 &&& 3) <<< 4) : Nat) : Int) - 32 := by
  unfold q6QsRef
  have h1 : (128 * n + l + 64) / 128 = n := by omega
  have h2 : (128 * n + l + 64) % 128 = l + 64 := by omega
  have h3 : (l + 64) / 32 = 2 := by omega
  have h4 : (l + 64) % 32 = l := by omega
  rw [h1, h2, h3, h4]
  norm_num

lemma q6QsRef_idx3 (ql qh : List Std.U8) (n l : Nat) (hl : l < 32) :
    q6QsRef ql qh (128 * n + l + 96)
      = ((((ql.getD (64 * n + l + 32) 0#u8).val >>> 4)
          ||| (((qh.getD (32 * n + l) 0#u8).val >>> 6 &&& 3) <<< 4) : Nat) : Int) - 32 := by
  unfold q6QsRef
  have h1 : (128 * n + l + 96) / 128 = n := by omega
  have h2 : (128 * n + l + 96) % 128 = l + 96 := by omega
  have h3 : (l + 96) / 32 = 3 := by omega
  have h4 : (l + 96) % 32 = l := by omega
  rw [h1, h2, h3, h4]
  norm_num


lemma q6_combine_lt' (a h k : Nat) :
    ((a &&& 15) ||| (((h >>> k) &&& 3) <<< 4)) < 64 := by
  rw [Nat.shiftLeft_eq]
  have ha : a &&& 15 < 2 ^ 6 := by
    have h := Nat.and_lt_two_pow a (n := 4) (by norm_num : (15 : Nat) < 2 ^ 4)
    omega
  have hb : (h >>> k &&& 3) * 2 ^ 4 < 2 ^ 6 := by
    have h := Nat.and_lt_two_pow (h >>> k) (n := 2) (by norm_num : (3 : Nat) < 2 ^ 2)
    nlinarith
  exact Nat.or_lt_two_pow ha hb

lemma q6_high_lt (a : Nat) (ha : a < 256) (h k : Nat) :
    ((a >>> 4) ||| (((h >>> k) &&& 3) <<< 4)) < 64 := by
  rw [Nat.shiftLeft_eq]
  have ha' : a >>> 4 < 2 ^ 4 := by rw [Nat.shiftRight_eq_div_pow]; omega
  have hb : (h >>> k &&& 3) * 2 ^ 4 < 2 ^ 6 := by
    have h := Nat.and_lt_two_pow (h >>> k) (n := 2) (by norm_num : (3 : Nat) < 2 ^ 2)
    nlinarith
  exact Nat.or_lt_two_pow (by omega) hb

lemma q6_cast_value (c : Nat) (hc : c < 64) (q : Std.I32) (x : Std.I8)
    (hq : q.val = (c : ℤ) - 32) (hx : x = IScalar.cast IScalarTy.I8 q) :
    x.val = (c : ℤ) - 32 := by
  rw [hx, IScalar.cast_val_eq]
  rw [show (2 : Nat) ^ (min IScalarTy.I8.numBits IScalarTy.I32.numBits) = 256 by
    norm_num [IScalarTy.numBits]]
  rw [hq, bmod_sub32 c hc]

lemma q6_write_val_gen (ql qh : List Std.U8) (idx : Nat) (qhil : Nat) (hqc : qhil < 64)
    (i7 : Std.U8) (q1 : Std.I32) (x : Std.I8)
    (h7 : i7.val = qhil) (hq : q1.val = (i7.val : ℤ) - 32)
    (hx : x = IScalar.cast IScalarTy.I8 q1)
    (hr : q6QsRef ql qh idx = ((qhil : ℤ) - 32)) :
    x.val = q6QsRef ql qh idx := by
  have hc : i7.val < 64 := by rw [h7]; exact hqc
  rw [q6_cast_value i7.val hc q1 x hq hx, h7, hr]


theorem q6_val_1_ident (ql qh : List Std.U8) (n l : Nat) (hl : l < 32)
    (i2 hi i3 i6 i5 i4 i7 : Std.U8) (q1 : Std.I32) (x : Std.I8)
    (h2 : i2.val = (ql.getD (64 * n + l) 0#u8).val)
    (hh : hi.val = (qh.getD (32 * n + l) 0#u8).val)
    (h3 : i3.val = (i2 &&& 15#u8).val) (h5 : i5.val = (i4 &&& 3#u8).val)
    (h4 : i4.val = hi.val >>> 0) (h6 : i6.val = i5.val <<< 4 % U8.size)
    (h7 : i7.val = (i3 ||| i6).val)
    (hq : q1.val = (i7.val : ℤ) - 32) (hx : x = IScalar.cast IScalarTy.I8 q1) :
    x.val = q6QsRef ql qh (128 * n + l) := by
  have h7c : i7.val = ((i2.val &&& 15) ||| (((hi.val >>> 0) &&& 3) <<< 4)) := by
    rw [h7, UScalar.val_or, h3, UScalar.val_and, h6, h5, UScalar.val_and, h4]
    simp only [Nat.shiftRight_zero, show (15#u8).val = 15 by rfl,
      show (3#u8).val = 3 by rfl, U8.size_eq]
    rw [shift4_mod (hi.val &&& 3) (by
      have := Nat.and_lt_two_pow hi.val (n := 2) (by norm_num : (3 : Nat) < 2 ^ 2)
      omega)]
  have hr : q6QsRef ql qh (128 * n + l)
      = ((((i2.val &&& 15) ||| (((hi.val >>> 0) &&& 3) <<< 4)) : Nat) : ℤ) - 32 := by
    rw [q6QsRef_idx0 ql qh n l hl, h2, hh]
  exact q6_write_val_gen ql qh (128 * n + l)
    (((i2.val &&& 15) ||| (((hi.val >>> 0) &&& 3) <<< 4)))
    (q6_combine_lt' i2.val hi.val 0) i7 q1 x h7c hq hx hr

theorem q6_val_2_ident (ql qh : List Std.U8) (n l : Nat) (hl : l < 32)
    (i10 hi i11 i14 i13 i12 i15 : Std.U8) (q2 : Std.I32) (x : Std.I8)
    (h2 : i10.val = (ql.getD (64 * n + l + 32) 0#u8).val)
    (hh : hi.val = (qh.getD (32 * n + l) 0#u8).val)
    (h3 : i11.val = (i10 &&& 15#u8).val) (h5 : i13.val = (i12 &&& 3#u8).val)
    (h4 : i12.val = hi.val >>> 2) (h6 : i14.val = i13.val <<< 4 % U8.size)
    (h7 : i15.val = (i11 ||| i14).val)
    (hq : q2.val = (i15.val : ℤ) - 32) (hx : x = IScalar.cast IScalarTy.I8 q2) :
    x.val = q6QsRef ql qh (128 * n + l + 32) := by
  have h7c : i15.val = ((i10.val &&& 15) ||| (((hi.val >>> 2) &&& 3) <<< 4)) := by
    rw [h7, UScalar.val_or, h3, UScalar.val_and, h6, h5, UScalar.val_and, h4]
    simp only [show (15#u8).val = 15 by rfl, show (3#u8).val = 3 by rfl, U8.size_eq]
    rw [shift4_mod (hi.val >>> 2 &&& 3) (by
      have := Nat.and_lt_two_pow (hi.val >>> 2) (n := 2) (by norm_num : (3 : Nat) < 2 ^ 2)
      omega)]
  have hr : q6QsRef ql qh (128 * n + l + 32)
      = ((((i10.val &&& 15) ||| (((hi.val >>> 2) &&& 3) <<< 4)) : Nat) : ℤ) - 32 := by
    rw [q6QsRef_idx1 ql qh n l hl, h2, hh]
  exact q6_write_val_gen ql qh (128 * n + l + 32)
    (((i10.val &&& 15) ||| (((hi.val >>> 2) &&& 3) <<< 4)))
    (q6_combine_lt' i10.val hi.val 2) i15 q2 x h7c hq hx hr

theorem q6_val_3_ident (ql qh : List Std.U8) (n l : Nat) (hl : l < 32)
    (i17 hi i18 i21 i20 i19 i22 : Std.U8) (q3 : Std.I32) (x : Std.I8)
    (h2 : i17.val = (ql.getD (64 * n + l) 0#u8).val)
    (hh : hi.val = (qh.getD (32 * n + l) 0#u8).val)
    (h3 : i18.val = i17.val >>> 4) (h5 : i20.val = (i19 &&& 3#u8).val)
    (h4 : i19.val = hi.val >>> 4) (h6 : i21.val = i20.val <<< 4 % U8.size)
    (h7 : i22.val = (i18 ||| i21).val)
    (hq : q3.val = (i22.val : ℤ) - 32) (hx : x = IScalar.cast IScalarTy.I8 q3) :
    x.val = q6QsRef ql qh (128 * n + l + 64) := by
  have h7c : i22.val = ((i17.val >>> 4) ||| (((hi.val >>> 4) &&& 3) <<< 4)) := by
    rw [h7, UScalar.val_or, h3, h6, h5, UScalar.val_and, h4]
    simp only [show (3#u8).val = 3 by rfl, U8.size_eq]
    rw [shift4_mod (hi.val >>> 4 &&& 3) (by
      have := Nat.and_lt_two_pow (hi.val >>> 4) (n := 2) (by norm_num : (3 : Nat) < 2 ^ 2)
      omega)]
  have hr : q6QsRef ql qh (128 * n + l + 64)
      = ((((i17.val >>> 4) ||| (((hi.val >>> 4) &&& 3) <<< 4)) : Nat) : ℤ) - 32 := by
    rw [q6QsRef_idx2 ql qh n l hl, h2, hh]
  exact q6_write_val_gen ql qh (128 * n + l + 64)
    (((i17.val >>> 4) ||| (((hi.val >>> 4) &&& 3) <<< 4)))
    (q6_high_lt i17.val i17.hBounds hi.val 4) i22 q3 x h7c hq hx hr

theorem q6_val_4_ident (ql qh : List Std.U8) (n l : Nat) (hl : l < 32)
    (i25 hi i26 i29 i28 i27 i30 : Std.U8) (q4 : Std.I32) (x : Std.I8)
    (h2 : i25.val = (ql.getD (64 * n + l + 32) 0#u8).val)
    (hh : hi.val = (qh.getD (32 * n + l) 0#u8).val)
    (h3 : i26.val = i25.val >>> 4) (h5 : i28.val = (i27 &&& 3#u8).val)
    (h4 : i27.val = hi.val >>> 6) (h6 : i29.val = i28.val <<< 4 % U8.size)
    (h7 : i30.val = (i26 ||| i29).val)
    (hq : q4.val = (i30.val : ℤ) - 32) (hx : x = IScalar.cast IScalarTy.I8 q4) :
    x.val = q6QsRef ql qh (128 * n + l + 96) := by
  have h7c : i30.val = ((i25.val >>> 4) ||| (((hi.val >>> 6) &&& 3) <<< 4)) := by
    rw [h7, UScalar.val_or, h3, h6, h5, UScalar.val_and, h4]
    simp only [show (3#u8).val = 3 by rfl, U8.size_eq]
    rw [shift4_mod (hi.val >>> 6 &&& 3) (by
      have := Nat.and_lt_two_pow (hi.val >>> 6) (n := 2) (by norm_num : (3 : Nat) < 2 ^ 2)
      omega)]
  have hr : q6QsRef ql qh (128 * n + l + 96)
      = ((((i25.val >>> 4) ||| (((hi.val >>> 6) &&& 3) <<< 4)) : Nat) : ℤ) - 32 := by
    rw [q6QsRef_idx3 ql qh n l hl, h2, hh]
  exact q6_write_val_gen ql qh (128 * n + l + 96)
    (((i25.val >>> 4) ||| (((hi.val >>> 6) &&& 3) <<< 4)))
    (q6_high_lt i25.val i25.hBounds hi.val 6) i30 q4 x h7c hq hx hr


theorem q6_pointwise (ql qh : List Std.U8) (n l : Nat) (out a : Array Std.I8 256#usize)
    (i33 i36 i39 i42 : Std.Usize) (i34 i37 i40 i43 : Std.I8) (hl : l < 32)
    (hset : a.val = (((out.val.set i33.val i34).set i36.val i37).set i39.val i40).set i42.val i43)
    (hi33 : i33.val = 128 * n + l) (hi36 : i36.val = 128 * n + l + 32)
    (hi39 : i39.val = 128 * n + l + 64) (hi42 : i42.val = 128 * n + l + 96)
    (h33 : i33.val < 256) (h36 : i36.val < 256) (h39 : i39.val < 256) (h42 : i42.val < 256)
    (hv1 : i34.val = q6QsRef ql qh (128 * n + l))
    (hv2 : i37.val = q6QsRef ql qh (128 * n + l + 32))
    (hv3 : i40.val = q6QsRef ql qh (128 * n + l + 64))
    (hv4 : i43.val = q6QsRef ql qh (128 * n + l + 96))
    (hinv : q6Inv ql qh n l out) :
    ∀ j, j < 256 → (a.val.getD j 0#i8).val = q6Val ql qh n (l + 1) j := by
  intro j hj
  have e42 : i42.val < (((out.val.set i33.val i34).set i36.val i37).set i39.val i40).length := by
    rw [List.length_set, List.length_set, List.length_set]; simpa using h42
  have e39 : i39.val < ((out.val.set i33.val i34).set i36.val i37).length := by
    rw [List.length_set, List.length_set]; simpa using h39
  have e36 : i36.val < (out.val.set i33.val i34).length := by
    rw [List.length_set]; simpa using h36
  have e33 : i33.val < out.val.length := by simpa using h33
  rw [hset, getD_set]
  by_cases h4 : j = i42.val
  · rw [if_pos ⟨h4, e42⟩, h4, hi42, q6Val_4 ql qh n l hl]
    exact hv4
  · rw [if_neg (fun h => h4 h.1), getD_set]
    by_cases h3 : j = i39.val
    · rw [if_pos ⟨h3, e39⟩, h3, hi39, q6Val_3 ql qh n l hl]
      exact hv3
    · rw [if_neg (fun h => h3 h.1), getD_set]
      by_cases h2 : j = i36.val
      · rw [if_pos ⟨h2, e36⟩, h2, hi36, q6Val_2 ql qh n l hl]
        exact hv2
      · rw [if_neg (fun h => h2 h.1), getD_set]
        by_cases h1 : j = i33.val
        · rw [if_pos ⟨h1, e33⟩, h1, hi33, q6Val_1 ql qh n l hl]
          exact hv1
        · rw [if_neg (fun h => h1 h.1),
            q6Val_step ql qh n l j
              (by intro h; exact h1 (by rw [hi33]; exact h))
              (by intro h; exact h2 (by rw [hi36]; exact h))
              (by intro h; exact h3 (by rw [hi39]; exact h))
              (by intro h; exact h4 (by rw [hi42]; exact h))]
          exact hinv.2.2 j hj

lemma q6_measure (l1 l : Nat) (h : l1 = l + 1) (hl : l < 32) : 32 - l1 < 32 - l := by omega
lemma q6_succ_le (l : Nat) (hl : l < 32) : l + 1 ≤ 32 := by omega
lemma q6_succ_le' (l1 l : Nat) (h : l1 = l + 1) (hl : l < 32) : l1 ≤ 32 := by omega


lemma q6_measure2 (b : Nat) (hb : b < 2) : 2 - (b + 1) < 2 - b := by omega
lemma q6_idx_lt_64 (i n l c L : Nat) (hn : n < 2) (hl : l < 32)
    (hi : i = 64 * n + l + c) (hc : c ≤ 32) (hL : 128 ≤ L) : i < L := by omega
lemma q6_idx_lt_32 (i n l L : Nat) (hn : n < 2) (hl : l < 32)
    (hi : i = 32 * n + l) (hL : 64 ≤ L) : i < L := by omega
lemma q6_idx_lt_128 (i n l c L : Nat) (hn : n < 2) (hl : l < 32)
    (hi : i = 128 * n + l + c) (hc : c ≤ 96) (hL : 256 ≤ L) : i < L := by omega


theorem q6_body_vals (ql qh : List Std.U8) (n l : Nat) (hl : l < 32)
    (i2 hi i3 i6 i5 i4 i7 i10 i11 i14 i13 i12 i15 i17 i18 i21 i20 i19 i22
      i25 i26 i29 i28 i27 i30 : Std.U8) (q1 q2 q3 q4 : Std.I32) (i34 i37 i40 i43 : Std.I8)
    (h2a : i2.val = (ql.getD (64 * n + l) 0#u8).val)
    (hha : hi.val = (qh.getD (32 * n + l) 0#u8).val)
    (h2b : i10.val = (ql.getD (64 * n + l + 32) 0#u8).val)
    (h2c : i17.val = (ql.getD (64 * n + l) 0#u8).val)
    (h2d : i25.val = (ql.getD (64 * n + l + 32) 0#u8).val)
    (h3a : i3.val = (i2 &&& 15#u8).val) (h5a : i5.val = (i4 &&& 3#u8).val)
    (h4a : i4.val = hi.val >>> 0) (h6a : i6.val = i5.val <<< 4 % U8.size)
    (h7a : i7.val = (i3 ||| i6).val)
    (h3b : i11.val = (i10 &&& 15#u8).val) (h5b : i13.val = (i12 &&& 3#u8).val)
    (h4b : i12.val = hi.val >>> 2) (h6b : i14.val = i13.val <<< 4 % U8.size)
    (h7b : i15.val = (i11 ||| i14).val)
    (h3c : i18.val = i17.val >>> 4) (h5c : i20.val = (i19 &&& 3#u8).val)
    (h4c : i19.val = hi.val >>> 4) (h6c : i21.val = i20.val <<< 4 % U8.size)
    (h7c : i22.val = (i18 ||| i21).val)
    (h3d : i26.val = i25.val >>> 4) (h5d : i28.val = (i27 &&& 3#u8).val)
    (h4d : i27.val = hi.val >>> 6) (h6d : i29.val = i28.val <<< 4 % U8.size)
    (h7d : i30.val = (i26 ||| i29).val)
    (hq1 : q1.val = (i7.val : ℤ) - 32) (hq2 : q2.val = (i15.val : ℤ) - 32)
    (hq3 : q3.val = (i22.val : ℤ) - 32) (hq4 : q4.val = (i30.val : ℤ) - 32)
    (hx1 : i34 = IScalar.cast IScalarTy.I8 q1) (hx2 : i37 = IScalar.cast IScalarTy.I8 q2)
    (hx3 : i40 = IScalar.cast IScalarTy.I8 q3) (hx4 : i43 = IScalar.cast IScalarTy.I8 q4) :
    i34.val = q6QsRef ql qh (128 * n + l) ∧ i37.val = q6QsRef ql qh (128 * n + l + 32) ∧
      i40.val = q6QsRef ql qh (128 * n + l + 64) ∧
      i43.val = q6QsRef ql qh (128 * n + l + 96) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact q6_val_1_ident ql qh n l hl i2 hi i3 i6 i5 i4 i7 q1 i34
      h2a hha h3a h5a h4a h6a h7a hq1 hx1
  · exact q6_val_2_ident ql qh n l hl i10 hi i11 i14 i13 i12 i15 q2 i37
      h2b hha h3b h5b h4b h6b h7b hq2 hx2
  · exact q6_val_3_ident ql qh n l hl i17 hi i18 i21 i20 i19 i22 q3 i40
      h2c hha h3c h5c h4c h6c h7c hq3 hx3
  · exact q6_val_4_ident ql qh n l hl i25 hi i26 i29 i28 i27 i30 q4 i43
      h2d hha h3d h5d h4d h6d h7d hq4 hx4

theorem q6_inner_body (ql qh : Slice Std.U8) (n qb64 qb32 l : Std.Usize)
    (out : Array Std.I8 256#usize)
    (hn : n.val < 2) (hq64 : qb64.val = 64 * n.val) (hq32 : qb32.val = 32 * n.val)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) (hl : l.val < 32)
    (hinv : q6Inv ql.val qh.val n.val l.val out) :
    quant.q6_k_codes_loop0_loop0.body ql qh n qb64 qb32 out l ⦃ r =>
      match r with
      | .done o => q6Inv ql.val qh.val n.val 32 o
      | .cont p => q6Inv ql.val qh.val n.val p.2.val p.1 ∧ 32 - p.2.val < 32 - l.val ⦄ := by
  unfold quant.q6_k_codes_loop0_loop0.body
  rw [if_pos (by scalar_tac : l < 32#usize)]
  step*
  · rw [i8_post]; exact hcast_sub32_lower i7
  · rw [i16_post]; exact hcast_sub32_lower i15
  · rw [i23_post]; exact hcast_sub32_lower i22
  · rw [i31_post]; exact hcast_sub32_lower i30
  · have hl' : l.val < 32 := hl
    have hi1 : i1.val = 64 * n.val + l.val := by rw [i1_post, hq64]
    have hidx : i.val = 32 * n.val + l.val := by rw [i_post, hq32]
    have hi9 : i9.val = 64 * n.val + l.val + 32 := by rw [i9_post, hi1]
    have hi24 : i24.val = 64 * n.val + l.val + 32 := by rw [i24_post, hi1]
    have hi32 : i32.val = 128 * n.val := by rw [i32_post]
    have hi33 : i33.val = 128 * n.val + l.val := by rw [i33_post, hi32]
    have hi35 : i35.val = 128 * n.val + l.val := by rw [i35_post, hi32]
    have hi36 : i36.val = 128 * n.val + l.val + 32 := by rw [i36_post, hi35]
    have hi38 : i38.val = 128 * n.val + l.val := by rw [i38_post, hi32]
    have hi39 : i39.val = 128 * n.val + l.val + 64 := by rw [i39_post, hi38]
    have hi41 : i41.val = 128 * n.val + l.val := by rw [i41_post, hi32]
    have hi42 : i42.val = 128 * n.val + l.val + 96 := by rw [i42_post, hi41]
    have hb1 : i1.val < ql.val.length :=
      q6_idx_lt_64 i1.val n.val l.val 0 ql.val.length hn hl' hi1 (by norm_num) hql
    have hbq : i.val < qh.val.length :=
      q6_idx_lt_32 i.val n.val l.val qh.val.length hn hl' hidx hqh
    have hb9 : i9.val < ql.val.length :=
      q6_idx_lt_64 i9.val n.val l.val 32 ql.val.length hn hl' hi9 (by norm_num) hql
    have hb24 : i24.val < ql.val.length :=
      q6_idx_lt_64 i24.val n.val l.val 32 ql.val.length hn hl' hi24 (by norm_num) hql
    have h2a : i2.val = (ql.val.getD (64 * n.val + l.val) 0#u8).val := by
      rw [i2_post, ← hi1]; exact q4_getD_getElem ql i1 hb1
    have hha : hi.val = (qh.val.getD (32 * n.val + l.val) 0#u8).val := by
      rw [hi_post, ← hidx]; exact q4_getD_getElem qh i hbq
    have h2b : i10.val = (ql.val.getD (64 * n.val + l.val + 32) 0#u8).val := by
      rw [i10_post, ← hi9]; exact q4_getD_getElem ql i9 hb9
    have h2c : i17.val = (ql.val.getD (64 * n.val + l.val) 0#u8).val := by
      rw [i17_post, ← hi1]; exact q4_getD_getElem ql i1 hb1
    have h2d : i25.val = (ql.val.getD (64 * n.val + l.val + 32) 0#u8).val := by
      rw [i25_post, ← hi24]; exact q4_getD_getElem ql i24 hb24
    have h33 : i33.val < 256 :=
      q6_idx_lt_128 i33.val n.val l.val 0 256 hn hl' hi33 (by norm_num) (by norm_num)
    have h36 : i36.val < 256 :=
      q6_idx_lt_128 i36.val n.val l.val 32 256 hn hl' hi36 (by norm_num) (by norm_num)
    have h39 : i39.val < 256 :=
      q6_idx_lt_128 i39.val n.val l.val 64 256 hn hl' hi39 (by norm_num) (by norm_num)
    have h42 : i42.val < 256 :=
      q6_idx_lt_128 i42.val n.val l.val 96 256 hn hl' hi42 (by norm_num) (by norm_num)
    have hq1 : q1.val = (i7.val : ℤ) - 32 := by rw [q1_post, i8_post, hcast_u8_I32_val]
    have hq2 : q2.val = (i15.val : ℤ) - 32 := by rw [q2_post, i16_post, hcast_u8_I32_val]
    have hq3 : q3.val = (i22.val : ℤ) - 32 := by rw [q3_post, i23_post, hcast_u8_I32_val]
    have hq4 : q4.val = (i30.val : ℤ) - 32 := by rw [q4_post, i31_post, hcast_u8_I32_val]
    obtain ⟨hv1, hv2, hv3, hv4⟩ := q6_body_vals ql.val qh.val n.val l.val hl'
      i2 hi i3 i6 i5 i4 i7 i10 i11 i14 i13 i12 i15 i17 i18 i21 i20 i19 i22
      i25 i26 i29 i28 i27 i30 q1 q2 q3 q4 i34 i37 i40 i43
      h2a hha h2b h2c h2d i3_post i5_post i4_post i6_post i7_post
      i11_post i13_post i12_post i14_post i15_post
      i18_post i20_post i19_post i21_post i22_post
      i26_post i28_post i27_post i29_post i30_post
      hq1 hq2 hq3 hq4 i34_post i37_post i40_post i43_post
    have ha' : a.val
        = (((out.val.set i33.val i34).set i36.val i37).set i39.val i40).set i42.val i43 := by
      rw [a_post, out3_post, out2_post, out1_post, Array.set_val_eq, Array.set_val_eq,
        Array.set_val_eq, Array.set_val_eq]
    have hout : out.val.length = 256 := by exact Array.property out
    have hpw : ∀ j, j < 256 → (a.val.getD j 0#i8).val = q6Val ql.val qh.val n.val l1.val j := by
      rw [l1_post]
      exact q6_pointwise ql.val qh.val n.val l.val out a i33 i36 i39 i42
        i34 i37 i40 i43 hl' ha' hi33 hi36 hi39 hi42 h33 h36 h39 h42 hv1 hv2 hv3 hv4 hinv
    exact ⟨⟨hinv.1, q6_succ_le' l1.val l.val l1_post hl', hpw⟩,
      q6_measure l1.val l.val l1_post hl'⟩

theorem q6_inner_spec (ql qh : Slice Std.U8) (out0 : Array Std.I8 256#usize)
    (n qb64 qb32 : Std.Usize) (hn : n.val < 2) (hq64 : qb64.val = 64 * n.val)
    (hq32 : qb32.val = 32 * n.val) (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length)
    (hinv : q6Inv ql.val qh.val n.val 0 out0) :
    quant.q6_k_codes_loop0_loop0 ql qh out0 n qb64 qb32 0#usize ⦃ out =>
      q6Inv ql.val qh.val n.val 32 out ⦄ := by
  unfold quant.q6_k_codes_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Array Std.I8 256#usize × Std.Usize => 32 - st.2.val)
    (inv := fun st => q6Inv ql.val qh.val n.val st.2.val st.1)
    (post := fun out => q6Inv ql.val qh.val n.val 32 out)
  · intro x hx
    rcases x with ⟨out, l⟩
    by_cases hl : l < 32#usize
    · convert q6_inner_body ql qh n qb64 qb32 l out hn hq64 hq32 hql hqh (by scalar_tac) hx using 2
      split <;> simp
    · unfold quant.q6_k_codes_loop0_loop0.body
      simp only [if_neg hl]
      have hl32 : l.val = 32 := by have := hx.2.1; scalar_tac
      simpa [hl32] using hx
  · exact hinv

theorem q6_outer_spec (ql qh : Slice Std.U8) (out0 : Array Std.I8 256#usize) (n : Std.Usize)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length)
    (hinv : q6Inv ql.val qh.val n.val 0 out0) :
    quant.q6_k_codes_loop0 ql qh out0 n ⦃ out => q6Inv ql.val qh.val 2 0 out ⦄ := by
  unfold quant.q6_k_codes_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Array Std.I8 256#usize × Std.Usize => 2 - st.2.val)
    (inv := fun st => q6Inv ql.val qh.val st.2.val 0 st.1)
    (post := fun out => q6Inv ql.val qh.val 2 0 out)
  · rintro ⟨out, b⟩ hb_inv
    by_cases hb : b < 2#usize
    · unfold quant.q6_k_codes_loop0.body
      dsimp only
      rw [if_pos hb]
      have hbv : b.val < 2 := by scalar_tac
      step*
      · refine WP.spec_bind (q6_inner_spec ql qh out b qb64 qb32 hbv qb64_post qb32_post hql hqh ?_) ?_
        · simpa using hb_inv
        · intro out1 hout1
          step*
          rw [n1_post]
          exact ⟨q6Inv_succ ql.val qh.val b.val out1 hbv hout1,
            q6_measure2 b.val hbv⟩
    · unfold quant.q6_k_codes_loop0.body
      simp only [if_neg hb]
      have hb2 : b.val = 2 := by have := hb_inv.1; scalar_tac
      simpa [hb2] using hb_inv
  · exact hinv

theorem word_q6_k_codes_ok (blk : Slice Std.U8) (hlen : 210#usize ≤ Slice.len blk) :
    ∃ v : Array Std.I8 256#usize,
      word_q6_k_codes blk = ok (core.result.Result.Ok v) ∧
      ∀ i, i < 256 → (v.val.getD i 0#i8).val = q6CodesRef blk.val i := by
  have hblen : 210 ≤ blk.val.length := by have := hlen; scalar_tac
  have hspec : word_q6_k_codes blk ⦃ r => ∃ v : Array Std.I8 256#usize,
      r = (core.result.Result.Ok v : core.result.Result (Array Std.I8 256#usize)
        quant.QuantError) ∧
      ∀ i, i < 256 →
        (v.val.getD i 0#i8).val = q6QsRef (blk.val.take 128) ((blk.val.drop 128).take 64) i ⦄ := by
    rw [word_q6_k_codes]
    unfold quant.q6_k_codes
    rw [quant_Q6_K_BLOCK_BYTES_eq]
    simp only [bind_tc_ok, quant.need]
    rw [if_neg (by scalar_tac : ¬ (Slice.len blk < 210#usize))]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    step*
    · rw [i1_post]; simp only [quant.QK_K]; scalar_tac
    · rw [i3_post, i1_post, i2_post]; simp only [quant.QK_K]; scalar_tac
    · have hi1 : i1.val = 128 := by rw [i1_post]; simp only [quant.QK_K]; norm_num
      have hi3 : i3.val = 192 := by
        rw [i3_post, hi1, i2_post]
        simp only [quant.QK_K]
        norm_num
      have hql' : ql.val = blk.val.take 128 := by
        rw [ql_post]; simp only [List.slice]; rw [hi1]; simp
      have hqh' : qh.val = (blk.val.drop 128).take 64 := by
        rw [qh_post]; simp only [List.slice]; rw [hi1, hi3]
      have hql : 128 ≤ ql.val.length := by rw [hql']; simp [List.length_take]; omega
      have hqh : 64 ≤ qh.val.length := by
        rw [hqh']; simp [List.length_take, List.length_drop]; omega
      refine WP.spec_bind (q6_outer_spec ql qh (Array.repeat 256#usize 0#i8) 0#usize hql hqh
        (q6Inv_repeat ql.val qh.val)) ?_
      intro out1 hout1
      refine ⟨out1, rfl, ?_⟩
      intro i hi
      rw [hout1.2.2 i hi, q6Val_four ql.val qh.val i hi, hql', hqh']
  rcases WP.spec_imp_exists hspec with ⟨r, hr, hv⟩
  rcases hv with ⟨v, hrv, hpt⟩
  refine ⟨v, ?_, ?_⟩
  · rw [hr, hrv]
  · intro i hi
    rw [hpt i hi]
    exact (q6QsRef_eq_q6CodesRef blk.val i hi).symm

/-! ## GGUF parse invariants (integer spec layer)

The specification predicates `word_gguf_parse_header` must establish. They are stated over the
parsed structure and the input bytes, with no floats: ggml block geometry, checked element/byte
sizing (`LengthOverflow` / `UnknownType` gates), tensor-directory range containment (`OutOfBounds`
gate), alignment, and the header/data ordering. -/

/-- The four magic bytes `GGUF`. -/
def GGUF_MAGIC_BYTES : List Std.U8 := [71#u8, 71#u8, 85#u8, 70#u8]

/-- The input's first four bytes are the GGUF magic. -/
def magic_ok (buf : Slice Std.U8) : Prop := buf.val.take 4 = GGUF_MAGIC_BYTES

/-- The admitted GGUF versions (2 and 3). -/
def version_ok (v : Std.U32) : Prop := v = 2#u32 ∨ v = 3#u32

/-- ggml block geometry `(weights per block, bytes per block)` for the implemented types. -/
def block_geom (t : Std.U32) : Option (Nat × Nat) :=
  match t.val with
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 8 => some (32, 34)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 14 => some (256, 210)
  | _ => none

/-- Storage blocks for `n` elements at `qk` weights per block (ceil). -/
def block_count (qk n : Nat) : Nat := n / qk + if n % qk = 0 then 0 else 1

/-- Checked element count: the product of the dims, `none` on overflow. -/
def dims_prod (dims : List Std.U64) : Option Nat :=
  dims.foldl (fun acc d => acc.bind (fun a => some (a * d.val))) (some 1)

/-- Checked storage bytes: the `LengthOverflow` / `UnknownType` gate. -/
def required_bytes (t : Std.U32) (n : Nat) : Option Nat :=
  match block_geom t with
  | some (qk, bb) => some (block_count qk n * bb)
  | none => none

/-- A tensor range: known type, non-overflowing element count, and the byte range inside the
payload (the `OutOfBounds` gate). -/
def TensorRangeWF (payload_len : Nat) (t : gguf.TensorDesc) : Prop :=
  ∃ n k, dims_prod t.dims.val = some n ∧ required_bytes t.ggml_type n = some k ∧
    t.offset.val + k ≤ payload_len

/-- The parsed header is well-formed: admitted version, positive alignment, data section after the
header and inside the file, and every tensor range contained in the payload. -/
def HeaderWF (payload_len : Nat) (h : gguf.GgufHeader) : Prop :=
  version_ok h.version ∧ 0 < h.alignment.val ∧
    h.header_end.val ≤ h.data0.val ∧ h.data0.val ≤ payload_len ∧
    ∀ t, t ∈ h.tensors.val → TensorRangeWF payload_len t

lemma gguf_cur_u8_some (s : gguf.Cur) (v : Std.U8) (h : s.b[s.i]? = some v) :
    gguf.Cur.u8 s ⦃ r => r.1 = some v ∧ r.2.b = s.b ∧ r.2.i.val = s.i.val + 1 ⦄ := by
  obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h
  have hmax : s.i.val + 1 ≤ Usize.max := by
    have := Slice.length_ineq s.b
    omega
  unfold gguf.Cur.u8
  simp only [core.slice.Slice.get, core.slice.index.Usize.get]
  rw [h]
  simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch,
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]
  step*

lemma gguf_cur_u8_none (s : gguf.Cur) (h : s.b[s.i]? = none) :
    gguf.Cur.u8 s ⦃ r => r.1 = none ∧ r.2 = s ⦄ := by
  unfold gguf.Cur.u8
  simp only [core.slice.Slice.get, core.slice.index.Usize.get]
  rw [h]
  simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch,
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]
  step*

lemma option_branch_some {α : Type} (v : α) :
    core.option.Option.Insts.CoreOpsTry_traitTry.branch (some v) =
      ok (core.ops.control_flow.ControlFlow.Continue v) := rfl

lemma block_geom_zero : block_geom 0#u32 = some (1, 4) := rfl
lemma block_geom_q4k : block_geom 12#u32 = some (256, 144) := rfl
lemma block_geom_bad : block_geom 77#u32 = none := rfl
lemma dims_prod_nil : dims_prod [] = some 1 := rfl
lemma required_bytes_none_iff (t : Std.U32) (n : Nat) :
    required_bytes t n = none ↔ block_geom t = none := by
  unfold required_bytes
  split <;> simp_all
lemma version_2_ok : version_ok 2#u32 := Or.inl rfl
lemma version_3_ok : version_ok 3#u32 := Or.inr rfl


/-! ## GGUF cursor chain: equality forms

`gguf.Cur.u8`/`u16` are pinned to whole equations over `gguf.Cur.adv s n`, the cursor advanced `n`
bytes in the same buffer. The composite readers are proved as `rw` chains on these equations; the
extracted pair destructures are resolved by defeq substitution (`show`), so no reduction runs over
the cursor's `i + 1#usize` literal. A read that runs past the buffer returns `none` at the first
missing byte (`gguf_cur_u16_short`). -/

/-- Cursor advanced `n` bytes in the same buffer. -/
def gguf.Cur.adv (s : gguf.Cur) (n : Nat) (h : s.i.val + n ≤ Usize.max) : gguf.Cur :=
  ⟨s.b, UScalar.ofNatCore (s.i.val + n) (by
    have h1 := Usize.cMax_bound
    show s.i.val + n < 2^System.Platform.numBits
    omega)⟩

@[simp] lemma gguf.Cur.adv_b (s : gguf.Cur) (n : Nat) (h : s.i.val + n ≤ Usize.max) :
    (gguf.Cur.adv s n h).b = s.b := rfl

@[simp] lemma gguf.Cur.adv_i (s : gguf.Cur) (n : Nat) (h : s.i.val + n ≤ Usize.max) :
    (gguf.Cur.adv s n h).i.val = s.i.val + n := rfl

/-! ## scalar add step -/

lemma gguf_usize_add_one (i : Std.Usize) (h : i.val + 1 ≤ Usize.max) :
    i + 1#usize = ok (UScalar.ofNatCore (i.val + 1) (by
      have h1 := Usize.cMax_bound
      show i.val + 1 < 2^System.Platform.numBits
      omega)) := by
  show UScalar.add i (1#usize) = _
  simp only [UScalar.add]
  show UScalar.tryMk .Usize (i.val + 1) = _
  have hin : UScalar.inBounds .Usize (i.val + 1) := by
    show i.val + 1 < 2^System.Platform.numBits
    have h1 := Usize.cMax_bound
    omega
  have hck : UScalar.check_bounds .Usize (i.val + 1) = true :=
    (UScalar.check_bounds_eq_inBounds _ _).mpr hin
  simp only [UScalar.tryMk, UScalar.tryMkOpt]
  rw [dif_pos hck]
  simp only [Result.ofOption]

lemma option_branch_none {α : Type} :
    core.option.Option.Insts.CoreOpsTry_traitTry.branch (none : Option α) =
      ok (core.ops.control_flow.ControlFlow.Break none) := rfl

lemma from_residual_none {α : Type} :
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual α none
      = ok none := rfl

/-! ## u8 equality forms -/

lemma gguf_cur_u8_eq_some (s : gguf.Cur) (v : Std.U8) (h : s.b.val[s.i.val]? = some v)
    (b : s.i.val + 1 ≤ Usize.max) :
    gguf.Cur.u8 s = ok (some v, gguf.Cur.adv s 1 b) := by
  have h' : s.b[s.i]? = some v := by rwa [Slice.getElem?_Usize_eq]
  unfold gguf.Cur.u8
  simp only [core.slice.Slice.get, core.slice.index.Usize.get]
  rw [h']
  simp only [bind_tc_ok]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [gguf_usize_add_one s.i b]
  simp only [bind_tc_ok]
  try rfl

lemma gguf_cur_u8_eq_none (s : gguf.Cur) (h : s.b.val[s.i.val]? = none) :
    gguf.Cur.u8 s = ok (none, s) := by
  have h' : s.b[s.i]? = none := by rwa [Slice.getElem?_Usize_eq]
  unfold gguf.Cur.u8
  simp only [core.slice.Slice.get, core.slice.index.Usize.get]
  rw [h']
  simp only [bind_tc_ok]
  rw [option_branch_none]
  simp only [bind_tc_ok, from_residual_none]
  try rfl

/-! ## u16 -/

lemma gguf_cur_u16_eq_some (s : gguf.Cur) (v0 v1 : Std.U8)
    (h0 : s.b.val[s.i.val]? = some v0) (h1 : s.b.val[s.i.val + 1]? = some v1)
    (b2 : s.i.val + 2 ≤ Usize.max) :
    gguf.Cur.u16 s = ok (some (core.num.U16.from_le_bytes (Array.make 2#usize [v0, v1])),
      gguf.Cur.adv s 2 b2) := by
  have b1 : s.i.val + 1 ≤ Usize.max := by omega
  have b1' : (gguf.Cur.adv s 1 b1).i.val + 1 ≤ Usize.max := by
    simp only [gguf.Cur.adv_i]; omega
  have e0 := gguf_cur_u8_eq_some s v0 h0 b1
  have e1 := gguf_cur_u8_eq_some (gguf.Cur.adv s 1 b1) v1 h1 b1'
  unfold gguf.Cur.u16
  rw [e0]
  simp only [bind_tc_ok]
  show
    (do let cf ← core.option.Option.Insts.CoreOpsTry_traitTry.branch (some v0)
        match cf with
        | core.ops.control_flow.ControlFlow.Continue val =>
          let (o1, self2) ← gguf.Cur.u8 (gguf.Cur.adv s 1 b1)
          let cf1 ← core.option.Option.Insts.CoreOpsTry_traitTry.branch o1
          match cf1 with
          | core.ops.control_flow.ControlFlow.Continue val1 =>
            let i ← lift (core.num.U16.from_le_bytes (Array.make 2#usize [val, val1]))
            ok (some i, self2)
          | core.ops.control_flow.ControlFlow.Break residual =>
            let o2 ←
              core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
                Std.U16 residual
            ok (o2, self2)
        | core.ops.control_flow.ControlFlow.Break residual =>
          let o1 ←
            core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
              Std.U16 residual
          ok (o1, gguf.Cur.adv s 1 b1)) =
    ok (some (core.num.U16.from_le_bytes (Array.make 2#usize [v0, v1])),
      gguf.Cur.adv s 2 b2)
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [e1]
  simp only [bind_tc_ok]
  show
    (do let cf1 ← core.option.Option.Insts.CoreOpsTry_traitTry.branch (some v1)
        match cf1 with
        | core.ops.control_flow.ControlFlow.Continue val1 =>
          let i ← lift (core.num.U16.from_le_bytes (Array.make 2#usize [v0, val1]))
          ok (some i, gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 b1')
        | core.ops.control_flow.ControlFlow.Break residual =>
          let o2 ←
            core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
              Std.U16 residual
          ok (o2, gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 b1')) =
    ok (some (core.num.U16.from_le_bytes (Array.make 2#usize [v0, v1])),
      gguf.Cur.adv s 2 b2)
  rw [option_branch_some]
  simp only [bind_tc_ok, lift]
  try rfl

lemma gguf_cur_u16_short (s : gguf.Cur) (h : s.b.val.length < s.i.val + 2) :
    ∃ s', gguf.Cur.u16 s = ok (none, s') := by
  cases h0 : s.b.val[s.i.val]? with
  | none =>
    refine ⟨s, ?_⟩
    have e0 := gguf_cur_u8_eq_none s h0
    unfold gguf.Cur.u16
    rw [e0]
    simp only [bind_tc_ok]
    show
      (do let cf ← core.option.Option.Insts.CoreOpsTry_traitTry.branch (none : Option Std.U8)
          match cf with
          | core.ops.control_flow.ControlFlow.Continue val =>
            let (o1, self2) ← gguf.Cur.u8 s
            let cf1 ← core.option.Option.Insts.CoreOpsTry_traitTry.branch o1
            match cf1 with
            | core.ops.control_flow.ControlFlow.Continue val1 =>
              let i ← lift (core.num.U16.from_le_bytes (Array.make 2#usize [val, val1]))
              ok (some i, self2)
            | core.ops.control_flow.ControlFlow.Break residual =>
              let o2 ←
                core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
                  Std.U16 residual
              ok (o2, self2)
          | core.ops.control_flow.ControlFlow.Break residual =>
            let o1 ←
              core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
                Std.U16 residual
            ok (o1, s)) =
      ok (none, s)
    rw [option_branch_none]
    simp only [bind_tc_ok, from_residual_none]
    try rfl
  | some v0 =>
    have b1 : s.i.val + 1 ≤ Usize.max := by
      obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h0
      have := Slice.length_ineq s.b
      omega
    have h1 : s.b.val[s.i.val + 1]? = none :=
      List.getElem?_eq_none (by have := Slice.length_ineq s.b; omega)
    have e0 := gguf_cur_u8_eq_some s v0 h0 b1
    have e1 := gguf_cur_u8_eq_none (gguf.Cur.adv s 1 b1) h1
    refine ⟨gguf.Cur.adv s 1 b1, ?_⟩
    unfold gguf.Cur.u16
    rw [e0]
    simp only [bind_tc_ok]
    show
      (do let cf ← core.option.Option.Insts.CoreOpsTry_traitTry.branch (some v0)
          match cf with
          | core.ops.control_flow.ControlFlow.Continue val =>
            let (o1, self2) ← gguf.Cur.u8 (gguf.Cur.adv s 1 b1)
            let cf1 ← core.option.Option.Insts.CoreOpsTry_traitTry.branch o1
            match cf1 with
            | core.ops.control_flow.ControlFlow.Continue val1 =>
              let i ← lift (core.num.U16.from_le_bytes (Array.make 2#usize [val, val1]))
              ok (some i, self2)
            | core.ops.control_flow.ControlFlow.Break residual =>
              let o2 ←
                core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
                  Std.U16 residual
              ok (o2, self2)
          | core.ops.control_flow.ControlFlow.Break residual =>
            let o1 ←
              core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
                Std.U16 residual
            ok (o1, gguf.Cur.adv s 1 b1)) =
      ok (none, gguf.Cur.adv s 1 b1)
    rw [option_branch_some]
    simp only [bind_tc_ok]
    rw [e1]
    simp only [bind_tc_ok]
    show
      (do let cf1 ← core.option.Option.Insts.CoreOpsTry_traitTry.branch (none : Option Std.U8)
          match cf1 with
          | core.ops.control_flow.ControlFlow.Continue val1 =>
            let i ← lift (core.num.U16.from_le_bytes (Array.make 2#usize [v0, val1]))
            ok (some i, gguf.Cur.adv s 1 b1)
          | core.ops.control_flow.ControlFlow.Break residual =>
            let o2 ←
              core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
                Std.U16 residual
            ok (o2, gguf.Cur.adv s 1 b1)) =
      ok (none, gguf.Cur.adv s 1 b1)
    rw [option_branch_none]
    simp only [bind_tc_ok, from_residual_none]
    try rfl


/-! ## GGUF cursor chain: `u32`, `u64`, `four`, and the first refusal

Same equation route as the `u8`/`u16` chain: whole-equation `rw` on `gguf_cur_u8_eq_*` /
`gguf_cur_u32_eq_some`, then `simp only [bind_tc_ok, uncurry_apply_pair]` (Aeneas's
`uncurry_apply_pair` resolves the extracted pair destructures first-order), then
`option_branch_some`/`option_branch_none`/`from_residual_none`. Short buffers truncate at the first
missing byte: `gguf_cur_u32_short`/`gguf_cur_u64_short`/`gguf_cur_four_short` return the cursor at
that byte. `word_gguf_parse_header_truncated_magic` is the first refusal theorem: a buffer shorter
than four bytes parses to `Err (Truncated "magic")`. -/

/-! ## u32 -/

lemma gguf_cur_u32_eq_some (s : gguf.Cur) (v0 v1 v2 v3 : Std.U8)
    (h0 : s.b.val[s.i.val]? = some v0) (h1 : s.b.val[s.i.val + 1]? = some v1)
    (h2 : s.b.val[s.i.val + 2]? = some v2) (h3 : s.b.val[s.i.val + 3]? = some v3)
    (b4 : s.i.val + 4 ≤ Usize.max) :
    gguf.Cur.u32 s = ok (some (core.num.U32.from_le_bytes (Array.make 4#usize [v0, v1, v2, v3])),
      gguf.Cur.adv s 4 b4) := by
  have b1 : s.i.val + 1 ≤ Usize.max := by omega
  have c1 : (gguf.Cur.adv s 1 b1).i.val + 1 ≤ Usize.max := by
    simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h1; have := Slice.length_ineq s.b; omega
  have c2 : (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1).i.val + 1 ≤ Usize.max := by
    simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h2; have := Slice.length_ineq s.b; omega
  have c3 : (gguf.Cur.adv (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) 1 c2).i.val + 1
      ≤ Usize.max := by
    simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h3; have := Slice.length_ineq s.b; omega
  have e0 := gguf_cur_u8_eq_some s v0 h0 b1
  have e1 := gguf_cur_u8_eq_some (gguf.Cur.adv s 1 b1) v1 h1 c1
  have e2 := gguf_cur_u8_eq_some (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) v2 h2 c2
  have e3 := gguf_cur_u8_eq_some
    (gguf.Cur.adv (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) 1 c2) v3 h3 c3
  unfold gguf.Cur.u32
  rw [e0]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [e1]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [e2]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [e3]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok, lift]
  try rfl

lemma gguf_cur_u32_short (s : gguf.Cur) (h : s.b.val.length < s.i.val + 4) :
    ∃ s', gguf.Cur.u32 s = ok (none, s') := by
  have h3 : s.b.val[s.i.val + 3]? = none :=
    List.getElem?_eq_none (by have := Slice.length_ineq s.b; omega)
  cases h0 : s.b.val[s.i.val]? with
  | none =>
    refine ⟨s, ?_⟩
    have e0 := gguf_cur_u8_eq_none s h0
    unfold gguf.Cur.u32
    rw [e0]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [option_branch_none]
    simp only [bind_tc_ok, from_residual_none]
    try rfl
  | some v0 =>
    have b1 : s.i.val + 1 ≤ Usize.max := by
      obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h0
      have := Slice.length_ineq s.b
      omega
    cases h1 : s.b.val[s.i.val + 1]? with
    | none =>
      have e0 := gguf_cur_u8_eq_some s v0 h0 b1
      have e1 := gguf_cur_u8_eq_none (gguf.Cur.adv s 1 b1) h1
      refine ⟨gguf.Cur.adv s 1 b1, ?_⟩
      unfold gguf.Cur.u32
      rw [e0]
      simp only [bind_tc_ok, uncurry_apply_pair]
      rw [option_branch_some]
      simp only [bind_tc_ok]
      rw [e1]
      simp only [bind_tc_ok, uncurry_apply_pair]
      rw [option_branch_none]
      simp only [bind_tc_ok, from_residual_none]
      try rfl
    | some v1 =>
      have c1 : (gguf.Cur.adv s 1 b1).i.val + 1 ≤ Usize.max := by
        simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h1; have := Slice.length_ineq s.b; omega
      cases h2 : s.b.val[s.i.val + 2]? with
      | none =>
        have e0 := gguf_cur_u8_eq_some s v0 h0 b1
        have e1 := gguf_cur_u8_eq_some (gguf.Cur.adv s 1 b1) v1 h1 c1
        have e2 := gguf_cur_u8_eq_none (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) h2
        refine ⟨gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1, ?_⟩
        unfold gguf.Cur.u32
        rw [e0]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e1]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e2]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_none]
        simp only [bind_tc_ok, from_residual_none]
        try rfl
      | some v2 =>
        have c2 : (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1).i.val + 1 ≤ Usize.max := by
          simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h2; have := Slice.length_ineq s.b; omega
        have e0 := gguf_cur_u8_eq_some s v0 h0 b1
        have e1 := gguf_cur_u8_eq_some (gguf.Cur.adv s 1 b1) v1 h1 c1
        have e2 := gguf_cur_u8_eq_some (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) v2 h2 c2
        have e3 := gguf_cur_u8_eq_none
          (gguf.Cur.adv (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) 1 c2) h3
        refine ⟨gguf.Cur.adv (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) 1 c2, ?_⟩
        unfold gguf.Cur.u32
        rw [e0]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e1]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e2]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e3]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_none]
        simp only [bind_tc_ok, from_residual_none]
        try rfl

/-! ## four -/

lemma gguf_cur_four_eq_some (s : gguf.Cur) (v0 v1 v2 v3 : Std.U8)
    (h0 : s.b.val[s.i.val]? = some v0) (h1 : s.b.val[s.i.val + 1]? = some v1)
    (h2 : s.b.val[s.i.val + 2]? = some v2) (h3 : s.b.val[s.i.val + 3]? = some v3)
    (b4 : s.i.val + 4 ≤ Usize.max) :
    gguf.Cur.four s = ok (some (Array.make 4#usize [v0, v1, v2, v3]),
      gguf.Cur.adv s 4 b4) := by
  have b1 : s.i.val + 1 ≤ Usize.max := by omega
  have c1 : (gguf.Cur.adv s 1 b1).i.val + 1 ≤ Usize.max := by
    simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h1; have := Slice.length_ineq s.b; omega
  have c2 : (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1).i.val + 1 ≤ Usize.max := by
    simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h2; have := Slice.length_ineq s.b; omega
  have c3 : (gguf.Cur.adv (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) 1 c2).i.val + 1
      ≤ Usize.max := by
    simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h3; have := Slice.length_ineq s.b; omega
  have e0 := gguf_cur_u8_eq_some s v0 h0 b1
  have e1 := gguf_cur_u8_eq_some (gguf.Cur.adv s 1 b1) v1 h1 c1
  have e2 := gguf_cur_u8_eq_some (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) v2 h2 c2
  have e3 := gguf_cur_u8_eq_some
    (gguf.Cur.adv (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) 1 c2) v3 h3 c3
  unfold gguf.Cur.four
  rw [e0]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [e1]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [e2]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [e3]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  try rfl

lemma gguf_cur_four_short (s : gguf.Cur) (h : s.b.val.length < s.i.val + 4) :
    ∃ s', gguf.Cur.four s = ok (none, s') := by
  have h3 : s.b.val[s.i.val + 3]? = none :=
    List.getElem?_eq_none (by have := Slice.length_ineq s.b; omega)
  cases h0 : s.b.val[s.i.val]? with
  | none =>
    refine ⟨s, ?_⟩
    have e0 := gguf_cur_u8_eq_none s h0
    unfold gguf.Cur.four
    rw [e0]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [option_branch_none]
    simp only [bind_tc_ok, from_residual_none]
    try rfl
  | some v0 =>
    have b1 : s.i.val + 1 ≤ Usize.max := by
      obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h0
      have := Slice.length_ineq s.b
      omega
    cases h1 : s.b.val[s.i.val + 1]? with
    | none =>
      have e0 := gguf_cur_u8_eq_some s v0 h0 b1
      have e1 := gguf_cur_u8_eq_none (gguf.Cur.adv s 1 b1) h1
      refine ⟨gguf.Cur.adv s 1 b1, ?_⟩
      unfold gguf.Cur.four
      rw [e0]
      simp only [bind_tc_ok, uncurry_apply_pair]
      rw [option_branch_some]
      simp only [bind_tc_ok]
      rw [e1]
      simp only [bind_tc_ok, uncurry_apply_pair]
      rw [option_branch_none]
      simp only [bind_tc_ok, from_residual_none]
      try rfl
    | some v1 =>
      have c1 : (gguf.Cur.adv s 1 b1).i.val + 1 ≤ Usize.max := by
        simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h1; have := Slice.length_ineq s.b; omega
      cases h2 : s.b.val[s.i.val + 2]? with
      | none =>
        have e0 := gguf_cur_u8_eq_some s v0 h0 b1
        have e1 := gguf_cur_u8_eq_some (gguf.Cur.adv s 1 b1) v1 h1 c1
        have e2 := gguf_cur_u8_eq_none (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) h2
        refine ⟨gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1, ?_⟩
        unfold gguf.Cur.four
        rw [e0]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e1]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e2]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_none]
        simp only [bind_tc_ok, from_residual_none]
        try rfl
      | some v2 =>
        have c2 : (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1).i.val + 1 ≤ Usize.max := by
          simp only [gguf.Cur.adv_i]; obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp h2; have := Slice.length_ineq s.b; omega
        have e0 := gguf_cur_u8_eq_some s v0 h0 b1
        have e1 := gguf_cur_u8_eq_some (gguf.Cur.adv s 1 b1) v1 h1 c1
        have e2 := gguf_cur_u8_eq_some (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) v2 h2 c2
        have e3 := gguf_cur_u8_eq_none
          (gguf.Cur.adv (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) 1 c2) h3
        refine ⟨gguf.Cur.adv (gguf.Cur.adv (gguf.Cur.adv s 1 b1) 1 c1) 1 c2, ?_⟩
        unfold gguf.Cur.four
        rw [e0]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e1]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e2]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_some]
        simp only [bind_tc_ok]
        rw [e3]
        simp only [bind_tc_ok, uncurry_apply_pair]
        rw [option_branch_none]
        simp only [bind_tc_ok, from_residual_none]
        try rfl

/-! ## u64 -/

lemma gguf_cur_u64_eq_some (s : gguf.Cur) (v0 v1 v2 v3 v4 v5 v6 v7 : Std.U8)
    (h0 : s.b.val[s.i.val]? = some v0) (h1 : s.b.val[s.i.val + 1]? = some v1)
    (h2 : s.b.val[s.i.val + 2]? = some v2) (h3 : s.b.val[s.i.val + 3]? = some v3)
    (h4 : s.b.val[s.i.val + 4]? = some v4) (h5 : s.b.val[s.i.val + 5]? = some v5)
    (h6 : s.b.val[s.i.val + 6]? = some v6) (h7 : s.b.val[s.i.val + 7]? = some v7)
    (b8 : s.i.val + 8 ≤ Usize.max) :
    gguf.Cur.u64 s =
      (do let i ← lift (core.convert.num.FromU64U32.from
            (core.num.U32.from_le_bytes (Array.make 4#usize [v0, v1, v2, v3])))
          let i1 ← lift (core.convert.num.FromU64U32.from
            (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7])))
          let i2 ← i1 <<< 32#i32
          let i3 ← lift (i ||| i2)
          ok (some i3, gguf.Cur.adv s 8 b8)) := by
  have b4 : s.i.val + 4 ≤ Usize.max := by omega
  have b4' : (gguf.Cur.adv s 4 b4).i.val + 4 ≤ Usize.max := by
    simp only [gguf.Cur.adv_i]; omega
  have e0 := gguf_cur_u32_eq_some s v0 v1 v2 v3 h0 h1 h2 h3 b4
  have e1 := gguf_cur_u32_eq_some (gguf.Cur.adv s 4 b4) v4 v5 v6 v7 h4 h5 h6 h7 b4'
  unfold gguf.Cur.u64
  rw [e0]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  rw [e1]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [option_branch_some]
  simp only [bind_tc_ok]
  try rfl

lemma gguf_cur_u64_short (s : gguf.Cur) (h : s.b.val.length < s.i.val + 8) :
    ∃ s', gguf.Cur.u64 s = ok (none, s') := by
  by_cases h4 : s.b.val.length < s.i.val + 4
  · obtain ⟨s', hs'⟩ := gguf_cur_u32_short s h4
    exact ⟨s', by
      unfold gguf.Cur.u64
      rw [hs']
      simp only [bind_tc_ok, uncurry_apply_pair]
      rw [option_branch_none]
      simp only [bind_tc_ok, from_residual_none]
      try rfl⟩
  · have b4 : s.i.val + 4 ≤ Usize.max := by
      have := Slice.length_ineq s.b
      omega
    have hex : ∀ j < 4, s.i.val + j < s.b.val.length := by
      intro j hj
      omega
    have e0 := gguf_cur_u32_eq_some s
      (s.b.val[s.i.val + 0]'(hex 0 (by omega))) (s.b.val[s.i.val + 1]'(hex 1 (by omega)))
      (s.b.val[s.i.val + 2]'(hex 2 (by omega))) (s.b.val[s.i.val + 3]'(hex 3 (by omega)))
      (List.getElem?_eq_some_iff.mpr ⟨hex 0 (by omega), rfl⟩)
      (List.getElem?_eq_some_iff.mpr ⟨hex 1 (by omega), rfl⟩)
      (List.getElem?_eq_some_iff.mpr ⟨hex 2 (by omega), rfl⟩)
      (List.getElem?_eq_some_iff.mpr ⟨hex 3 (by omega), rfl⟩) b4
    have h2 : (gguf.Cur.adv s 4 b4).b.val.length < (gguf.Cur.adv s 4 b4).i.val + 4 := by
      simp only [gguf.Cur.adv_b, gguf.Cur.adv_i]
      omega
    obtain ⟨s', hs'⟩ := gguf_cur_u32_short (gguf.Cur.adv s 4 b4) h2
    exact ⟨s', by
      unfold gguf.Cur.u64
      rw [e0]
      simp only [bind_tc_ok, uncurry_apply_pair]
      rw [option_branch_some]
      simp only [bind_tc_ok]
      rw [hs']
      simp only [bind_tc_ok, uncurry_apply_pair]
      rw [option_branch_none]
      simp only [bind_tc_ok, from_residual_none]
      try rfl⟩

/-! ## refusal: Truncated "magic" -/

lemma core_result_branch_err {T E : Type} (e : E) :
    core.result.Result.Insts.CoreOpsTry.branch (core.result.Result.Err e : core.result.Result T E) =
      ok (core.ops.control_flow.ControlFlow.Break (core.result.Result.Err e)) := rfl

lemma core_result_from_residual_err {T E : Type} (e : E) :
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual T
      (core.convert.FromSame E) (core.result.Result.Err e) = ok (core.result.Result.Err e) := by
  simp [core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]

theorem word_gguf_parse_header_truncated_magic (buf : Slice Std.U8) (hbuf : buf.val.length < 4) :
    word_gguf_parse_header buf =
      ok (core.result.Result.Err (gguf.GgufError.Truncated (toStr "magic"))) := by
  have hz : ((0#usize) : Std.Usize).val = 0 := rfl
  have hfour := gguf_cur_four_short { b := buf, i := 0#usize }
    (by show buf.val.length < (0#usize).val + 4; omega)
  obtain ⟨s', hs'⟩ := hfour
  unfold word_gguf_parse_header
  simp only [lift, bind_tc_ok]
  unfold gguf.parse_header
  rw [hs']
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_none]
  simp only [bind_tc_ok]
  rw [core_result_branch_err]
  simp only [bind_tc_ok, core_result_from_residual_err]
  try rfl


/-! ## refusal: BadMagic -/

lemma core_result_branch_ok {T E : Type} (v : T) :
    core.result.Result.Insts.CoreOpsTry.branch (core.result.Result.Ok v : core.result.Result T E) =
      ok (core.ops.control_flow.ControlFlow.Continue v) := rfl

theorem word_gguf_parse_header_bad_magic (buf : Slice Std.U8) (v0 v1 v2 v3 : Std.U8)
    (h0 : buf.val[0]? = some v0) (h1 : buf.val[1]? = some v1) (h2 : buf.val[2]? = some v2)
    (h3 : buf.val[3]? = some v3)
    (hne : core.array.equality.PartialEqArray.ne core.cmp.PartialEqU8
      (Array.make 4#usize [v0, v1, v2, v3]) gguf.GGUF_MAGIC = ok true) :
    word_gguf_parse_header buf = ok (core.result.Result.Err
      (gguf.GgufError.BadMagic (Array.make 4#usize [v0, v1, v2, v3]))) := by
  have b4 : ((0#usize) : Std.Usize).val + 4 ≤ Usize.max := by
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    obtain ⟨hl, _⟩ := List.getElem?_eq_some_iff.mp h3
    have := Slice.length_ineq buf
    omega
  have hfour := gguf_cur_four_eq_some { b := buf, i := 0#usize } v0 v1 v2 v3 h0 h1 h2 h3 b4
  unfold word_gguf_parse_header
  simp only [lift, bind_tc_ok]
  unfold gguf.parse_header
  rw [hfour]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hne]
  simp only [bind_tc_ok]
  try rfl


/-! ## Refusal theorems: version and count guards

`Truncated "version"` (the `u32` version read runs past the buffer), `UnsupportedVersion` (the
version is neither `2#u32` nor `3#u32`, the `!=` guards as explicit Bool equations), and
`Truncated "tensor count"` / `"kv count"` (the two `u64` count reads). The tensor-count `u64`
success for the `kv count` refusal is an explicit hypothesis (`hu1`); its reduction is
`gguf_cur_u64_eq_some`. -/

theorem word_gguf_parse_header_truncated_version (buf : Slice Std.U8) (v0 v1 v2 v3 : Std.U8)
    (h0 : buf.val[0]? = some v0) (h1 : buf.val[1]? = some v1) (h2 : buf.val[2]? = some v2)
    (h3 : buf.val[3]? = some v3)
    (hmag : core.array.equality.PartialEqArray.ne core.cmp.PartialEqU8
      (Array.make 4#usize [v0, v1, v2, v3]) gguf.GGUF_MAGIC = ok false)
    (hlen : buf.val.length < 8) :
    word_gguf_parse_header buf =
      ok (core.result.Result.Err (gguf.GgufError.Truncated (toStr "version"))) := by
  have b4 : ((0#usize) : Std.Usize).val + 4 ≤ Usize.max := by
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    obtain ⟨hl, _⟩ := List.getElem?_eq_some_iff.mp h3
    have := Slice.length_ineq buf
    omega
  have hfour := gguf_cur_four_eq_some { b := buf, i := 0#usize } v0 v1 v2 v3 h0 h1 h2 h3 b4
  obtain ⟨s'', hs''⟩ := gguf_cur_u32_short (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4) (by
    show buf.val.length < (0#usize).val + 4 + 4
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    omega)
  unfold word_gguf_parse_header
  simp only [lift, bind_tc_ok]
  unfold gguf.parse_header
  rw [hfour]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hmag]
  simp only [bind_tc_ok]
  rw [hs'']
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_none]
  simp only [bind_tc_ok]
  rw [core_result_branch_err]
  simp only [bind_tc_ok, core_result_from_residual_err]
  try rfl

theorem word_gguf_parse_header_unsupported_version (buf : Slice Std.U8)
    (v0 v1 v2 v3 v4 v5 v6 v7 : Std.U8)
    (h0 : buf.val[0]? = some v0) (h1 : buf.val[1]? = some v1) (h2 : buf.val[2]? = some v2)
    (h3 : buf.val[3]? = some v3) (h4 : buf.val[4]? = some v4) (h5 : buf.val[5]? = some v5)
    (h6 : buf.val[6]? = some v6) (h7 : buf.val[7]? = some v7)
    (hmag : core.array.equality.PartialEqArray.ne core.cmp.PartialEqU8
      (Array.make 4#usize [v0, v1, v2, v3]) gguf.GGUF_MAGIC = ok false)
    (hne2 : (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7]) != 2#u32) = true)
    (hne3 : (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7]) != 3#u32) = true) :
    word_gguf_parse_header buf = ok (core.result.Result.Err
      (gguf.GgufError.UnsupportedVersion
        (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7])))) := by
  have b4 : ((0#usize) : Std.Usize).val + 4 ≤ Usize.max := by
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    obtain ⟨hl, _⟩ := List.getElem?_eq_some_iff.mp h3
    have := Slice.length_ineq buf
    omega
  have b8 : ((0#usize) : Std.Usize).val + 8 ≤ Usize.max := by
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    obtain ⟨hl, _⟩ := List.getElem?_eq_some_iff.mp h7
    have := Slice.length_ineq buf
    omega
  have hfour := gguf_cur_four_eq_some { b := buf, i := 0#usize } v0 v1 v2 v3 h0 h1 h2 h3 b4
  have hu32 := gguf_cur_u32_eq_some (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4)
    v4 v5 v6 v7 h4 h5 h6 h7 (by
      show ((0#usize) : Std.Usize).val + 4 + 4 ≤ Usize.max
      have hz : ((0#usize) : Std.Usize).val = 0 := rfl
      omega)
  unfold word_gguf_parse_header
  simp only [lift, bind_tc_ok]
  unfold gguf.parse_header
  rw [hfour]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hmag]
  simp only [bind_tc_ok]
  rw [hu32]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hne2, hne3]
  try rfl

theorem word_gguf_parse_header_truncated_tensor_count (buf : Slice Std.U8)
    (v0 v1 v2 v3 v4 v5 v6 v7 : Std.U8)
    (h0 : buf.val[0]? = some v0) (h1 : buf.val[1]? = some v1) (h2 : buf.val[2]? = some v2)
    (h3 : buf.val[3]? = some v3) (h4 : buf.val[4]? = some v4) (h5 : buf.val[5]? = some v5)
    (h6 : buf.val[6]? = some v6) (h7 : buf.val[7]? = some v7)
    (hmag : core.array.equality.PartialEqArray.ne core.cmp.PartialEqU8
      (Array.make 4#usize [v0, v1, v2, v3]) gguf.GGUF_MAGIC = ok false)
    (hne2 : (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7]) != 2#u32) = true)
    (hne3 : (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7]) != 3#u32) = false)
    (hlen : buf.val.length < 16) :
    word_gguf_parse_header buf =
      ok (core.result.Result.Err (gguf.GgufError.Truncated (toStr "tensor count"))) := by
  have b4 : ((0#usize) : Std.Usize).val + 4 ≤ Usize.max := by
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    obtain ⟨hl, _⟩ := List.getElem?_eq_some_iff.mp h3
    have := Slice.length_ineq buf
    omega
  have b8 : ((0#usize) : Std.Usize).val + 8 ≤ Usize.max := by
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    obtain ⟨hl, _⟩ := List.getElem?_eq_some_iff.mp h7
    have := Slice.length_ineq buf
    omega
  have hfour := gguf_cur_four_eq_some { b := buf, i := 0#usize } v0 v1 v2 v3 h0 h1 h2 h3 b4
  have b8' : (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4).i.val + 4 ≤ Usize.max := by
    show ((0#usize) : Std.Usize).val + 4 + 4 ≤ Usize.max
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    omega
  have hu32 := gguf_cur_u32_eq_some (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4)
    v4 v5 v6 v7 h4 h5 h6 h7 b8'
  obtain ⟨s'', hs''⟩ := gguf_cur_u64_short
    (gguf.Cur.adv (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4) 4 b8') (by
    show buf.val.length < (0#usize).val + 4 + 4 + 8
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    omega)
  unfold word_gguf_parse_header
  simp only [lift, bind_tc_ok]
  unfold gguf.parse_header
  rw [hfour]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hmag]
  simp only [bind_tc_ok]
  rw [hu32]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hne2, hne3]
  rw [hs'']
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_none]
  simp only [bind_tc_ok]
  rw [core_result_branch_err]
  simp only [bind_tc_ok, core_result_from_residual_err]
  try rfl

theorem word_gguf_parse_header_truncated_kv_count (buf : Slice Std.U8)
    (v0 v1 v2 v3 v4 v5 v6 v7 : Std.U8)
    (h0 : buf.val[0]? = some v0) (h1 : buf.val[1]? = some v1) (h2 : buf.val[2]? = some v2)
    (h3 : buf.val[3]? = some v3) (h4 : buf.val[4]? = some v4) (h5 : buf.val[5]? = some v5)
    (h6 : buf.val[6]? = some v6) (h7 : buf.val[7]? = some v7)
    (hmag : core.array.equality.PartialEqArray.ne core.cmp.PartialEqU8
      (Array.make 4#usize [v0, v1, v2, v3]) gguf.GGUF_MAGIC = ok false)
    (hne2 : (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7]) != 2#u32) = true)
    (hne3 : (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7]) != 3#u32) = false)
    (b4 : ((0#usize) : Std.Usize).val + 4 ≤ Usize.max)
    (b8 : ((0#usize) : Std.Usize).val + 8 ≤ Usize.max)
    (b16 : ((0#usize) : Std.Usize).val + 16 ≤ Usize.max)
    (t : Std.U64)
    (hu1 : gguf.Cur.u64 (gguf.Cur.adv (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4) 4 b8)
        = ok (some t, gguf.Cur.adv { b := buf, i := 0#usize } 16 b16))
    (hlen : buf.val.length < 24) :
    word_gguf_parse_header buf =
      ok (core.result.Result.Err (gguf.GgufError.Truncated (toStr "kv count"))) := by
  have hfour := gguf_cur_four_eq_some { b := buf, i := 0#usize } v0 v1 v2 v3 h0 h1 h2 h3 b4
  have hu32 := gguf_cur_u32_eq_some (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4)
    v4 v5 v6 v7 h4 h5 h6 h7 b8
  obtain ⟨s'', hs''⟩ := gguf_cur_u64_short
    (gguf.Cur.adv { b := buf, i := 0#usize } 16 b16) (by
    show buf.val.length < (0#usize).val + 16 + 8
    have hz : ((0#usize) : Std.Usize).val = 0 := rfl
    omega)
  unfold word_gguf_parse_header
  simp only [lift, bind_tc_ok]
  unfold gguf.parse_header
  rw [hfour]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hmag]
  simp only [bind_tc_ok]
  rw [hu32]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hne2, hne3]
  rw [hu1]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hs'']
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_none]
  simp only [bind_tc_ok]
  rw [core_result_branch_err]
  simp only [bind_tc_ok, core_result_from_residual_err]
  try rfl


/-! ## The count guard and its cone status

`gguf_counts_guard_tail` is the count-guard reduction: message- and continuation-parametric
(`e : gguf.GgufError`, `K`), proved entirely at the `Result` level, so its axiom cone is exactly
`{propext, Classical.choice, Quot.sound}`. `word_gguf_parse_header_truncated_kv_tensor_count` is
the `Truncated "kv/tensor count"` refusal built on it.

Cone status of the refusal headline: **unregistrable in the closed cone**. Two independent causes,
both named here since `AxiomAudit.lean` is lead-owned:

* structural: `apeiron_word.gguf.parse_header` and its callees (`parse_kv`, `parse_tensor`,
  `parse_tensor_loop.body`) embed Aeneas-native decidability proofs (`toStr`'s auto-bound argument
  is `by decide +native`, `Aeneas/Std/String.lean:28`, with its own TODO against `+native`), and
  the dependency is def-transitive: even `word_gguf_parse_header buf = word_gguf_parse_header buf`
  by `rfl` inherits `*_native.decide.ax_*`. Every statement mentioning `word_gguf_parse_header` is
  affected until the model patch de-nativizes that auto-bound and the extraction is regenerated.
* local: each `toStr "…"` occurrence in a statement or proof mints a fresh
  `<decl>._native.decide.ax_*`, because `String.toByteArray` is extern-opaque and the bound
  `s.toByteArray.size ≤ U32.max` has no kernel-`decide` proof.

Cursor-side and `Result`-side declarations are unaffected: `gguf_cur_*` and `gguf_counts_guard_tail`
audit at `{propext, Classical.choice, Quot.sound}`. -/

/-- The `parse_header` count-guard tail: once the reads have produced `t`/`k` and the cursor `c3i`,
the guard chain terminates in `K (Err e)` whenever the `counts_ok` computation is `ok false`.
Message-parametric (`e`), continuation-parametric (`K`): no `toStr`, no `decide`. -/
lemma gguf_counts_guard_tail {α : Type} (buf : Slice Std.U8) (c3i : Std.Usize) (t k : Std.U64)
    (e : gguf.GgufError)
    (K : core.result.Result Unit gguf.GgufError → Std.Result α)
    (hcounts : (if k <= (UScalar.cast .U64 (core.num.Usize.saturating_sub buf.len c3i))
        then ok (decide (t <= (UScalar.cast .U64 (core.num.Usize.saturating_sub buf.len c3i))))
        else ok false) = ok false) :
    (do let i1 ← lift (core.num.Usize.saturating_sub buf.len c3i)
        let remaining ← lift (UScalar.cast .U64 i1)
        let counts_ok ← if k <= remaining then ok (decide (t <= remaining)) else ok false
        let counts_guard ← if counts_ok = true then ok (some ()) else ok none
        Bind.bind (core.option.Option.ok_or counts_guard e) K) = K (core.result.Result.Err e) := by
  simp only [lift, bind_tc_ok]
  rw [hcounts]
  simp only [bind_tc_ok]
  rw [if_neg (by decide : ¬((false : Bool) = true))]
  simp only [bind_tc_ok, core.option.Option.ok_or_none]
  try rfl

theorem word_gguf_parse_header_truncated_kv_tensor_count (buf : Slice Std.U8)
    (v0 v1 v2 v3 v4 v5 v6 v7 : Std.U8)
    (h0 : buf.val[0]? = some v0) (h1 : buf.val[1]? = some v1) (h2 : buf.val[2]? = some v2)
    (h3 : buf.val[3]? = some v3) (h4 : buf.val[4]? = some v4) (h5 : buf.val[5]? = some v5)
    (h6 : buf.val[6]? = some v6) (h7 : buf.val[7]? = some v7)
    (hmag : core.array.equality.PartialEqArray.ne core.cmp.PartialEqU8
      (Array.make 4#usize [v0, v1, v2, v3]) gguf.GGUF_MAGIC = ok false)
    (hne2 : (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7]) != 2#u32) = true)
    (hne3 : (core.num.U32.from_le_bytes (Array.make 4#usize [v4, v5, v6, v7]) != 3#u32) = false)
    (b4 : ((0#usize) : Std.Usize).val + 4 ≤ Usize.max)
    (b8 : ((0#usize) : Std.Usize).val + 8 ≤ Usize.max)
    (b16 : ((0#usize) : Std.Usize).val + 16 ≤ Usize.max)
    (b24 : ((0#usize) : Std.Usize).val + 24 ≤ Usize.max)
    (t k : Std.U64)
    (hu1 : gguf.Cur.u64 (gguf.Cur.adv (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4) 4 b8)
        = ok (some t, gguf.Cur.adv { b := buf, i := 0#usize } 16 b16))
    (hu2 : gguf.Cur.u64 (gguf.Cur.adv { b := buf, i := 0#usize } 16 b16)
        = ok (some k, gguf.Cur.adv { b := buf, i := 0#usize } 24 b24))
    (hcounts : (if k <= (UScalar.cast .U64 (core.num.Usize.saturating_sub buf.len
          (gguf.Cur.adv { b := buf, i := 0#usize } 24 b24).i))
        then ok (decide (t <= (UScalar.cast .U64 (core.num.Usize.saturating_sub buf.len
          (gguf.Cur.adv { b := buf, i := 0#usize } 24 b24).i))))
        else ok false) = ok false) :
    word_gguf_parse_header buf =
      ok (core.result.Result.Err (gguf.GgufError.Truncated (toStr "kv/tensor count"))) := by
  have hfour := gguf_cur_four_eq_some { b := buf, i := 0#usize } v0 v1 v2 v3 h0 h1 h2 h3 b4
  have hu32 := gguf_cur_u32_eq_some (gguf.Cur.adv { b := buf, i := 0#usize } 4 b4)
    v4 v5 v6 v7 h4 h5 h6 h7 b8
  unfold word_gguf_parse_header
  simp only [lift, bind_tc_ok]
  unfold gguf.parse_header
  rw [hfour]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hmag]
  simp only [bind_tc_ok]
  rw [hu32]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hne2, hne3]
  rw [hu1]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [hu2]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [core.option.Option.ok_or_some]
  simp only [bind_tc_ok]
  rw [core_result_branch_ok]
  simp only [bind_tc_ok]
  rw [gguf_counts_guard_tail buf ((gguf.Cur.adv { b := buf, i := 0#usize } 24 b24).i) t k
    (gguf.GgufError.Truncated (toStr "kv/tensor count")) _ hcounts]
  simp only [bind_tc_ok, core_result_branch_err, core_result_from_residual_err]
  try rfl


end apeiron_word
