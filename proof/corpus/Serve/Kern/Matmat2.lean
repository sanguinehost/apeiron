import Serve.Kern.Matmat

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-!
# Serving kernel group `kern.matmat` (part 2): the `Q8_0` and `Q5_0` block dots

`src/engine/kern/matmat.rs` projects `k` quantised activation columns against one weight block. The
portable integer core has three leaves and their drivers:

* `portable_u8_i8_dot` — the `Q8_0` 32-lane window dot `Σ_l (codes[l] as i8)·xq[l]`.
* `q8_0_block_kdots_portable` / `q8_0_block_kdots` — one `Q8_0` window dotted against each column.
* `q5_0_block_kdots_portable` / `q5_0_block_kdots` — one `Q5_0` block decoded and dotted (its leaf
  `portable_q5_0_dot` is refined in `Serve/Matmat.lean`, whose `q5DotRef`/`q5Word4` this file reuses).

Each extracted body is refined against a plain-`Int`/`List` reference that says *what* is computed;
the loop bodies are not restated. `i8reinterpret` is the native-endian one-byte reinterpretation the
extracted `i8::from_ne_bytes` performs.

Trust boundary: rustc/LLVM codegen, the `unsafe` AVX2 cores (`q5_0_block_kdots_avx2`,
`q8_0_block_kdots_avx2`, `hsum128`/`hsum256`), the `F32`/`F64` scale carriers and the `I64`/`I128`
accumulation in `row_accs`/`matmat_q8act` (whose checked arithmetic and `f16_to_fixed` scale read are
a separate layer) are not covered.

The portable leaf `core.num.I8.from_ne_bytes` is an `axiom` of the extraction, listed in
`proof/boundary-axioms.txt` ("std operation ... absent from the pinned Aeneas model"). It is carried
as the named hypothesis `I8FromNeBytesSpec`, which keeps every *statement* free of it — but the
body-calling headlines that consume the hypothesis still print that boundary axiom in their
`#print axioms` cone, because their proofs transitively mention the constant (the `--opaque`-external
pattern documented in `Serve/Cols.lean`). The two selectors `word_q8_0_block_kdots_ok` /
`word_q5_0_block_kdots_ok` and their refusals additionally carry the AVX2/x86 externals
(`_mm256_*`, `_mm_*`) and `kern.matmat.q5_0_block_kdots_avx2`, because the `avx = true` branch is in
the extracted body and the proof `unfold`s it. The single headline whose cone is exactly
`{propext, Classical.choice, Quot.sound}` is `word_q5_0_block_kdots_portable_refuses`.

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`. `decide` is used only on closed literal facts
about the reference, never on an extracted body.
-/

/-! ## The `i8::from_ne_bytes` boundary

`core.num.I8.from_ne_bytes` is opaque (an extraction `axiom`, `ApeironWord.lean:451`): its Rust
semantics is the one-byte little/native-endian reinterpretation `[b] ↦ b as i8`. It cannot be given a
defining equation here, so its behaviour enters as this named hypothesis; the refinement theorems
that call it are stated relative to it, the boundary the proof does not reach. -/

/-- The extracted `core.num.I8.from_ne_bytes` reinterprets its one-byte argument. Carried as this
lane's single boundary hypothesis (the opaque extraction axiom). -/
def I8FromNeBytesSpec : Prop :=
  ∀ x : U8, core.num.I8.from_ne_bytes (Array.make 1#usize [x]) = ok ⟨x.bv⟩

/-- Two `Array.make`s with the same data differ only in their length proof. -/
theorem array_make_congr {α : Type} (n : Usize) (init : List α)
    (h1 h2 : init.length = n.val) : Array.make n init h1 = Array.make n init h2 := by
  apply Aeneas.Std.Array.ext
  simp only [Array.make_val]

/-- `I8FromNeBytesSpec` applied at a specific length proof. -/
theorem I8FromNeBytesSpec_apply (hspec : I8FromNeBytesSpec) (x : U8)
    (hl : ([x] : List U8).length = (1#usize).val) :
    core.num.I8.from_ne_bytes (Array.make 1#usize [x] hl) = ok ⟨x.bv⟩ := by
  have he : Array.make 1#usize [x] hl = Array.make 1#usize [x] :=
    array_make_congr 1#usize [x] hl _
  rw [he]
  exact hspec x

/-- The native-endian reinterpretation of a byte as a signed `Int` (`b as i8`). -/
def i8reinterpret (b : U8) : Int := b.bv.toInt

theorem i8reinterpret_bounds (b : U8) : -128 ≤ i8reinterpret b ∧ i8reinterpret b ≤ 127 := by
  unfold i8reinterpret
  have h1 := BitVec.le_toInt b.bv
  have h2 := BitVec.toInt_lt (x := b.bv)
  have e : (2 : ℤ) ^ (8 - 1) = 128 := by norm_num
  rw [e] at h1 h2
  omega

/-- `i8reinterpret` of a byte is the value of the model `I8` with the same bit vector. -/
theorem i8reinterpret_eq_val (b : U8) : i8reinterpret b = ((⟨b.bv⟩ : I8)).val := rfl

/-! ## Reference: the `Q8_0` code-window dot -/

/-- The `Q8_0` reference dot `Σ_{l<n} (codes[l] as i8)·xq[l]`. -/
def u8i8DotRef (codes : List U8) (xq : List I8) : Nat → Int
  | 0 => 0
  | n + 1 => u8i8DotRef codes xq n + i8reinterpret (codes.getD n 0#u8) * (xq.getD n 0#i8).val

theorem u8i8DotRef_succ (codes : List U8) (xq : List I8) (n : Nat) :
    u8i8DotRef codes xq (n + 1) =
      u8i8DotRef codes xq n + i8reinterpret (codes.getD n 0#u8) * (xq.getD n 0#i8).val := rfl

/-- `u8i8DotRef` respects equal lane lists and counts. -/
theorem u8i8DotRef_congr {codes : List U8} {x1 x2 : List I8} {n1 n2 : Nat}
    (hx : x1 = x2) (hn : n1 = n2) : u8i8DotRef codes x1 n1 = u8i8DotRef codes x2 n2 := by
  subst hx; subst hn; rfl

theorem u8i8Term_bounds (codes : List U8) (xq : List I8) (l : Nat) :
    -16384 ≤ i8reinterpret (codes.getD l 0#u8) * (xq.getD l 0#i8).val ∧
      i8reinterpret (codes.getD l 0#u8) * (xq.getD l 0#i8).val ≤ 16384 := by
  have ha := i8reinterpret_bounds (codes.getD l 0#u8)
  have hb := i8_val_bounds (xq.getD l 0#i8)
  constructor <;> nlinarith [ha.1, ha.2, hb.1, hb.2]

theorem u8i8DotRef_bounds (codes : List U8) (xq : List I8) (n : Nat) :
    -16384 * (n : Int) ≤ u8i8DotRef codes xq n ∧
      u8i8DotRef codes xq n ≤ 16384 * (n : Int) := by
  induction n with
  | zero => simp only [u8i8DotRef]; norm_num
  | succ k ih =>
    rw [u8i8DotRef_succ]
    have ht := u8i8Term_bounds codes xq k
    constructor <;> push_cast <;> nlinarith [ih.1, ih.2, ht.1, ht.2]

/-! ## The `portable_u8_i8_dot` loop -/

/-- The `portable_u8_i8_dot` loop state: lane counter `l` and an accumulator that is the reference
prefix sum. The `codes/length ≤ 131071` bound keeps every accumulator inside `2^31`. -/
def u8i8LoopInv (codes : Slice U8) (xq : Slice I8) (st : I32 × Usize) : Prop :=
  st.2.val ≤ codes.val.length ∧ st.1.val = u8i8DotRef codes.val xq.val st.2.val

/-- The byte read of an in-bounds `Slice.getElem` is its `getD`. -/
theorem u8_getD_getElem_eq (xs : Slice U8) (i : Usize) (h : i.val < xs.val.length) :
    xs.val[i.val]'h = xs.val.getD i.val 0#u8 := by
  apply UScalar.eq_of_val_eq
  exact q4_getD_getElem xs i h

theorem u8i8_dot_loop_body_spec (codes : Slice U8) (xq : Slice I8)
    (hspec : I8FromNeBytesSpec) (hlen : codes.val.length ≤ xq.val.length)
    (hb : codes.val.length ≤ 131071) (acc : I32) (l : Usize)
    (hinv : u8i8LoopInv codes xq (acc, l)) :
    kern.matmat.portable_u8_i8_dot_loop.body codes xq acc l ⦃ r =>
      match r with
      | ControlFlow.done y => y.val = u8i8DotRef codes.val xq.val codes.val.length
      | ControlFlow.cont st' => u8i8LoopInv codes xq st' ∧
          codes.val.length - st'.2.val < codes.val.length - l.val ⦄ := by
  simp only [u8i8LoopInv] at hinv
  obtain ⟨hle, hacc⟩ := hinv
  by_cases hlt : l < Slice.len codes
  · have hl : l.val < codes.val.length := by
      have h := (UScalar.lt_equiv l (Slice.len codes)).1 hlt
      rw [Slice.len_val] at h
      exact h
    unfold kern.matmat.portable_u8_i8_dot_loop.body
    rw [if_pos hlt]
    simp only [lift, bind_tc_ok, I8FromNeBytesSpec_apply hspec]
    step*
    · -- `i3 * i5` lower bound
      rw [core.convert.num.FromI32I8.from_val_eq, core.convert.num.FromI32I8.from_val_eq, I32.min_eq]
      have h2 := i8_val_bounds (i1.bv#iscalar)
      have h3 := i8_val_bounds i4
      nlinarith [h2.1, h2.2, h3.1, h3.2]
    · rw [core.convert.num.FromI32I8.from_val_eq, core.convert.num.FromI32I8.from_val_eq, I32.max_eq]
      have h2 := i8_val_bounds (i1.bv#iscalar)
      have h3 := i8_val_bounds i4
      nlinarith [h2.1, h2.2, h3.1, h3.2]
    · -- `acc + i6` lower bound
      rw [hacc, i6_post, core.convert.num.FromI32I8.from_val_eq, core.convert.num.FromI32I8.from_val_eq, I32.min_eq]
      have h2 := i8_val_bounds (i1.bv#iscalar)
      have h3 := i8_val_bounds i4
      have hq := (u8i8DotRef_bounds codes.val xq.val (↑l)).1
      nlinarith [h2.1, h2.2, h3.1, h3.2, hq, Nat.le_of_lt hl, hb]
    · rw [hacc, i6_post, core.convert.num.FromI32I8.from_val_eq, core.convert.num.FromI32I8.from_val_eq, I32.max_eq]
      have h2 := i8_val_bounds (i1.bv#iscalar)
      have h3 := i8_val_bounds i4
      have hq := (u8i8DotRef_bounds codes.val xq.val (↑l)).2
      nlinarith [h2.1, h2.2, h3.1, h3.2, hq, Nat.le_of_lt hl, hb]
    · -- continuation
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [l1_post]; have := Nat.le_of_lt hl; omega
      · have hf1 : (↑(core.convert.num.FromI32I8.from (i1.bv#iscalar)) : ℤ) =
            i8reinterpret (codes.val.getD (↑l) 0#u8) := by
          rw [core.convert.num.FromI32I8.from_val_eq, i1_post, u8_getD_getElem_eq codes l hl]
          rfl
        have hf2 : (↑(core.convert.num.FromI32I8.from i4) : ℤ) =
            (xq.val.getD (↑l) 0#i8).val := by
          rw [core.convert.num.FromI32I8.from_val_eq, i4_post]
          exact i8_getD_getElem xq l (by omega)
        rw [l1_post, u8i8DotRef_succ, ← hacc, acc1_post, i6_post]
        congr 1
        rw [hf1, hf2]
      · rw [l1_post]; omega
  · unfold kern.matmat.portable_u8_i8_dot_loop.body
    rw [if_neg hlt]
    simp only [WP.spec_ok]
    have hge : codes.val.length ≤ (↑l : Nat) := by
      have h := (UScalar.lt_equiv l (Slice.len codes)).not.mp hlt
      rw [Slice.len_val] at h
      exact Nat.le_of_not_lt h
    have hl : (↑l : Nat) = codes.val.length := by omega
    rw [hacc, hl]

/-! ## The `portable_u8_i8_dot` loop and entry point -/

theorem word_portable_u8_i8_dot_loop_ok (codes : Slice U8) (xq : Slice I8)
    (hspec : I8FromNeBytesSpec) (hlen : codes.val.length ≤ xq.val.length)
    (hb : codes.val.length ≤ 131071) (acc : I32) (l : Usize)
    (hinv : u8i8LoopInv codes xq (acc, l)) :
    kern.matmat.portable_u8_i8_dot_loop codes xq acc l ⦃ r =>
      r.val = u8i8DotRef codes.val xq.val codes.val.length ⦄ := by
  unfold kern.matmat.portable_u8_i8_dot_loop
  apply loop.spec_decr_nat
    (measure := fun s : I32 × Usize => codes.val.length - s.2.val)
    (inv := fun s => u8i8LoopInv codes xq s)
    (post := fun y : I32 => y.val = u8i8DotRef codes.val xq.val codes.val.length)
  · rintro ⟨acc', l'⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
      (u8i8_dot_loop_body_spec codes xq hspec hlen hb acc' l' hinv')
    rw [hr]
    rcases r with y | s'
    · simpa only [WP.spec_ok] using hpost
    · simpa only [WP.spec_ok] using hpost
  · exact hinv

theorem word_portable_u8_i8_dot_ok (codes : Slice U8) (xq : Slice I8)
    (hspec : I8FromNeBytesSpec) (hlen : codes.val.length ≤ xq.val.length)
    (hb : codes.val.length ≤ 131071) :
    kern.matmat.portable_u8_i8_dot codes xq ⦃ r =>
      r.val = u8i8DotRef codes.val xq.val codes.val.length ⦄ := by
  unfold kern.matmat.portable_u8_i8_dot
  apply word_portable_u8_i8_dot_loop_ok codes xq hspec hlen hb 0#i32 0#usize
  refine ⟨?_, ?_⟩
  · simp
  · simp [u8i8DotRef]

/-! ## The `Q8_0` block-kdots driver -/

theorem ACT_QK_eq : kern.matmat.ACT_QK = 32#usize := by with_unfolding_all rfl

/-- The default activation block used to totalise `List.getD` in the references; never reached on
the stated domain (`d < k ≤ acts.length`). -/
def qaDefault : chart.QuantAct :=
  { scales := alloc.vec.Vec.from ([] : List I64) (by simp)
    codes := alloc.vec.Vec.from ([] : List I8) (by simp) }

/-- Column `c`'s reference `Q8_0` dot: the 32-byte window dotted with the column's lanes at
`xbase`. -/
def q8ColRef (window : List U8) (acts : List chart.QuantAct) (xbase k c : Nat) : Int :=
  if c < k then
    u8i8DotRef window ((acts.getD c qaDefault).codes.val.slice xbase (xbase + 32)) 32
  else 0

/-- The partially-filled array: columns before `c` hold their reference dot, the rest `0`. -/
def q8PartRef (window : List U8) (acts : List chart.QuantAct) (xbase k c i : Nat) : Int :=
  if i < c then q8ColRef window acts xbase k i else 0

/-- The `q8_0_block_kdots_portable` loop state: counter `c` and an output array that matches the
partial reference. -/
def q8KdotsInv (window : Slice U8) (acts : Slice chart.QuantAct) (xbase k : Usize)
    (out : Array I32 16#usize) (c : Usize) : Prop :=
  c.val ≤ k.val ∧ ∀ i, i < 16 → (out[i]!).val = q8PartRef window.val acts.val xbase.val k.val c.val i

/-- `getD` at an in-bounds index is the `getElem`. -/
theorem list_getD_eq_getElem {α : Type} (l : List α) (d : α) {n : Nat} (h : n < l.length) :
    l.getD n d = l[n] := List.getD_eq_getElem l d h

theorem q8_kdots_loop_body_spec (window : Slice U8) (acts : Slice chart.QuantAct)
    (hspec : I8FromNeBytesSpec) (k xbase : Usize) (hwin : window.val.length = 32)
    (hk : k.val ≤ acts.val.length) (hk16 : k.val ≤ 16) (hxb : xbase.val + 32 ≤ Usize.max)
    (hcodes : ∀ d, d < k.val → xbase.val + 32 ≤ (acts.val.getD d qaDefault).codes.val.length)
    (out : Array I32 16#usize) (c : Usize) (hinv : q8KdotsInv window acts xbase k out c) :
    kern.matmat.q8_0_block_kdots_portable_loop.body window acts k xbase out c ⦃ r =>
      match r with
      | ControlFlow.done y => ∀ i, i < 16 → (y[i]!).val = q8ColRef window.val acts.val xbase.val k.val i
      | ControlFlow.cont st' => q8KdotsInv window acts xbase k st'.1 st'.2 ∧
          k.val - st'.2.val < k.val - c.val ⦄ := by
  simp only [q8KdotsInv] at hinv
  obtain ⟨hck, hout⟩ := hinv
  by_cases hlt : c < k
  · have hc : c.val < k.val := (UScalar.lt_equiv c k).1 hlt
    have hca : c.val < acts.val.length := by omega
    unfold kern.matmat.q8_0_block_kdots_portable_loop.body
    rw [if_pos hlt]
    simp only [chart.QuantAct.impl.codes, lift, bind_tc_ok]
    step*
    · -- `xbase + ACT_QK` does not overflow
      rw [ACT_QK_eq]
      exact hxb
    · -- the range end lies inside the column's lanes
      have hAQ : (↑kern.matmat.ACT_QK : Nat) = 32 := by rw [ACT_QK_eq]; norm_num
      have hlen : qa.codes.deref.val.length = (acts.val.getD c.val qaDefault).codes.val.length := by
        rw [qa_post, ← list_getD_eq_getElem acts.val qaDefault hca]
        simp only [alloc.vec.Vec.deref, Slice.from_val]
      have hi : (↑i : Nat) = ↑xbase + 32 := by rw [i_post, hAQ]
      simp only [Slice.length]
      rw [hi, hlen]
      exact hcodes c.val hc
    · -- the portable dot
      refine WP.spec_bind (word_portable_u8_i8_dot_ok window s1 hspec ?_ ?_) ?_
      · -- `window ≤ s1` by `s1.length = 32`
        have s1len : s1.length = 32 := by
          rw [s1_post1, i_post]
          have hAQ : (↑kern.matmat.ACT_QK : Nat) = 32 := by rw [ACT_QK_eq]; norm_num
          rw [hAQ]; omega
        have hs1 : s1.val.length = 32 := by simpa using s1len
        show window.val.length ≤ s1.val.length
        omega
      · rw [hwin]; norm_num
      intro i1 hi1
      step*
      · -- invariant after the write
        have hs : qa.codes.deref.val = (acts.val.getD c.val qaDefault).codes.val := by
          rw [qa_post, list_getD_eq_getElem acts.val qaDefault hca]
          simp only [alloc.vec.Vec.deref, Slice.from_val, Slice.v]
        have s1val : s1.val =
            (acts.val.getD c.val qaDefault).codes.val.slice xbase.val (xbase.val + 32) := by
          rw [s1_post, hs, i_post]
          have hAQ : (↑kern.matmat.ACT_QK : Nat) = 32 := by rw [ACT_QK_eq]; norm_num
          rw [hAQ]
        have ho : out.length = 16 := by simpa using Array.property out
        refine ⟨⟨?_, ?_⟩, ?_⟩
        · rw [c1_post]; omega
        · intro i hi
          rw [a_post, c1_post]
          by_cases hic : i = c.val
          · subst hic
            rw [Array.getElem!_Nat_set_eq (h := ⟨rfl, by omega⟩)]
            rw [hi1]
            unfold q8PartRef q8ColRef
            rw [if_pos (Nat.lt_succ_self _), if_pos hc]
            exact u8i8DotRef_congr s1val hwin
          · rw [Array.getElem!_Nat_set_ne (h := fun hc_eq => hic hc_eq.symm)]
            rw [hout i hi]
            unfold q8PartRef
            by_cases hltc : i < c.val
            · rw [if_pos hltc, if_pos (by omega)]
            · rw [if_neg hltc, if_neg (by omega)]
        · rw [c1_post]; omega
  · -- `c ≥ k`: the loop finished; the done post is the full reference
    unfold kern.matmat.q8_0_block_kdots_portable_loop.body
    rw [if_neg hlt]
    simp only [WP.spec_ok]
    have hck' : c.val = k.val := le_antisymm hck (by
      by_contra h
      exact hlt ((UScalar.lt_equiv c k).2 (Nat.lt_of_not_le h)))
    intro i hi
    rw [hout i hi, hck']
    show q8PartRef window.val acts.val xbase.val k.val k.val i =
      q8ColRef window.val acts.val xbase.val k.val i
    unfold q8PartRef q8ColRef
    by_cases hik : i < k.val <;> simp only [hik, if_true, if_false]

/-- The `Q8_0` block output reference: column `i`'s dot against the window at base `xbase`. -/
abbrev q8BlockRef (window : List U8) (acts : List chart.QuantAct) (xbase k : Nat) (i : Nat) : Int :=
  q8ColRef window acts xbase k i

/-- The `Q8_0` block-kdots driver: the loop from its invariant returns the whole column array. -/
theorem word_q8_0_block_kdots_portable_loop_ok (window : Slice U8) (acts : Slice chart.QuantAct)
    (hspec : I8FromNeBytesSpec) (k xbase : Usize) (hwin : window.val.length = 32)
    (hk : k.val ≤ acts.val.length) (hk16 : k.val ≤ 16) (hxb : xbase.val + 32 ≤ Usize.max)
    (hcodes : ∀ d, d < k.val → xbase.val + 32 ≤ (acts.val.getD d qaDefault).codes.val.length)
    (out : Array I32 16#usize) (c : Usize) (hinv : q8KdotsInv window acts xbase k out c) :
    kern.matmat.q8_0_block_kdots_portable_loop window acts k xbase out c ⦃ r =>
      ∀ i, i < 16 → (r[i]!).val = q8BlockRef window.val acts.val xbase.val k.val i ⦄ := by
  unfold kern.matmat.q8_0_block_kdots_portable_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array I32 16#usize × Usize => k.val - s.2.val)
    (inv := fun s => q8KdotsInv window acts xbase k s.1 s.2)
    (post := fun y : Array I32 16#usize =>
      ∀ i, i < 16 → (y[i]!).val = q8BlockRef window.val acts.val xbase.val k.val i)
  · rintro ⟨out', c'⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
      (q8_kdots_loop_body_spec window acts hspec k xbase hwin hk hk16 hxb hcodes out' c' hinv')
    rw [hr]
    rcases r with y | s'
    · simpa only [WP.spec_ok, q8BlockRef] using hpost
    · simpa only [WP.spec_ok] using hpost
  · exact hinv

/-- The `Q8_0` block-kdots entry: zeroed `16`-lane output, loop over the columns. -/
theorem word_q8_0_block_kdots_portable_ok (window : Slice U8) (acts : Slice chart.QuantAct)
    (blk k : Usize) (hspec : I8FromNeBytesSpec) (hwin : window.val.length = 32)
    (hk : k.val ≤ acts.val.length) (hk16 : k.val ≤ 16) (hblk : blk.val * 32 + 32 ≤ Usize.max)
    (hcodes : ∀ d, d < k.val → blk.val * 32 + 32 ≤ (acts.val.getD d qaDefault).codes.val.length) :
    kern.matmat.q8_0_block_kdots_portable window acts blk k ⦃ r =>
      ∀ i, i < 16 → (r[i]!).val = q8BlockRef window.val acts.val (blk.val * 32) k.val i ⦄ := by
  unfold kern.matmat.q8_0_block_kdots_portable
  simp only [lift, bind_tc_ok]
  step*
  · have h32 : (32#usize).val = 32 := by norm_num
    rw [ACT_QK_eq, h32]; omega
  · have hxbval : xbase.val = blk.val * 32 := by rw [xbase_post, ACT_QK_eq]; norm_num
    refine WP.spec_mono (word_q8_0_block_kdots_portable_loop_ok window acts hspec k xbase hwin
      hk hk16 ?_ ?_ (Array.repeat 16#usize 0#i32) 0#usize ?_) ?_
    · rw [hxbval]; exact hblk
    · intro d hd; rw [hxbval]; exact hcodes d hd
    · refine ⟨by norm_num, ?_⟩
      intro i hi
      have h0 : (0#usize).val = 0 := by norm_num
      unfold q8PartRef
      rw [if_neg (by rw [h0]; omega), Array.getElem!_Nat_eq, Array.repeat_val]
      have h16 : (16#usize).val = 16 := by norm_num
      rw [h16, List.getElem!_replicate 0#i32 hi]
      rfl
    · intro r hr i hi
      rw [← hxbval]
      exact hr i hi

/-- `quant.Q8_0_QK` is the block width `32`, exposed for the driver's window. -/
theorem Q8_0_QK_eq : quant.Q8_0_QK = 32#usize := by with_unfolding_all rfl

/-- A `List.slice` that ends in bounds has length `end - start`. -/
theorem slice_len_eq {α : Type} (l : List α) (s e : Nat) (h : e ≤ l.length) :
    (l.slice s e).length = e - s := by
  rw [List.slice_length]; omega

/-- `getD` into an in-bounds `List.slice` reads the source at `s + j`. -/
theorem slice_getD {α : Type} (l : List α) (d : α) (s e j : Nat) (he : e ≤ l.length)
    (hj : s + j < e) : (l.slice s e).getD j d = l.getD (s + j) d := by
  rw [list_getD_eq_getElem (l.slice s e) d (by rw [List.slice_length]; omega),
      list_getD_eq_getElem l d (by omega)]
  exact List.getElem_slice s e j l ⟨he, hj⟩

/-- The `Q8_0` block-kdots selector: on the portable (`avx = false`) face it slices the row window
and refines the block driver. The `avx = true` face calls the `unsafe` AVX2 axiom and is not
covered. -/
theorem word_q8_0_block_kdots_ok (row : Slice U8) (weight_off : Usize) (acts : Slice chart.QuantAct)
    (blk k : Usize) (hspec : I8FromNeBytesSpec)
    (hrow : weight_off.val + 32 ≤ row.val.length) (hwov : weight_off.val + 32 ≤ Usize.max)
    (hk : k.val ≤ acts.val.length) (hk16 : k.val ≤ 16) (hblk : blk.val * 32 + 32 ≤ Usize.max)
    (hcodes : ∀ d, d < k.val → blk.val * 32 + 32 ≤ (acts.val.getD d qaDefault).codes.val.length) :
    kern.matmat.q8_0_block_kdots false row weight_off acts blk k ⦃ r =>
      ∀ i, i < 16 → (r[i]!).val =
        q8BlockRef (row.val.slice weight_off.val (weight_off.val + 32)) acts.val
          (blk.val * 32) k.val i ⦄ := by
  unfold kern.matmat.q8_0_block_kdots
  simp only [Bool.false_eq_true, if_false, lift, bind_tc_ok]
  step*
  · have h32 : (32#usize).val = 32 := by norm_num
    rw [Q8_0_QK_eq, h32]; omega
  · have h32 : (32#usize).val = 32 := by norm_num
    rw [i_post, Q8_0_QK_eq, h32]; exact hrow
  · have hsval : s.val = row.val.slice weight_off.val (weight_off.val + 32) := by
      rw [s_post, i_post, Q8_0_QK_eq]
      have h32 : (32#usize).val = 32 := by norm_num
      rw [h32]
    refine WP.spec_mono (word_q8_0_block_kdots_portable_ok s acts blk k hspec ?_ hk hk16 hblk ?_) ?_
    · rw [hsval, slice_len_eq row.val weight_off.val (weight_off.val + 32) hrow]; omega
    · exact hcodes
    · intro r hr i hi
      rw [← hsval]
      exact hr i hi

/-! ## The `Q5_0` block-kdots driver

`q5_0_block_kdots_portable` slices the packed `qh` word and the `16` low-nibble bytes `qs` from one
`22`-byte block, then runs the same column loop as `Q8_0` with the `Q5_0` leaf `portable_q5_0_dot`
(refined in `Serve/Matmat.lean`). The reference `q5ColRef` carries the decoded column dot; `q5BlockRef`
is the whole-block view used by the selector. -/

/-- The `Q5_0` block width; equal to the `Q8_0` width. -/
theorem Q5_0_QK_eq : quant.Q5_0_QK = 32#usize := by with_unfolding_all rfl

/-- Column `c`'s `Q5_0` reference dot: the packed word `w` and low-nibble bytes `qs` against the
column's lanes at `xbase`. -/
def q5ColRef (w : Nat) (qs : List U8) (acts : List chart.QuantAct) (xbase k c : Nat) : Int :=
  if c < k then q5DotRef w qs ((acts.getD c qaDefault).codes.val.slice xbase (xbase + 32)) else 0

/-- The partially-filled `Q5_0` output array. -/
def q5PartRef (w : Nat) (qs : List U8) (acts : List chart.QuantAct) (xbase k c i : Nat) : Int :=
  if i < c then q5ColRef w qs acts xbase k i else 0

/-- The `Q5_0` block loop state. -/
def q5KdotsInv (w : Nat) (qs : Slice U8) (acts : Slice chart.QuantAct) (xbase k : Usize)
    (out : Array I32 16#usize) (c : Usize) : Prop :=
  c.val ≤ k.val ∧ ∀ i, i < 16 → (out[i]!).val = q5PartRef w qs.val acts.val xbase.val k.val c.val i

/-- The whole-block `Q5_0` reference: the packed word is the four word bytes of `block`, the
low-nibble bytes are `block[6..22]`. -/
abbrev q5BlockRef (block : List U8) (acts : List chart.QuantAct) (xbase k : Nat) (i : Nat) : Int :=
  q5ColRef (q5Word4 (block.getD 2 0#u8) (block.getD 3 0#u8) (block.getD 4 0#u8) (block.getD 5 0#u8))
    (block.slice 6 22) acts xbase k i

theorem q5_kdots_loop_body_spec (w : Nat) (qh qs : Slice U8) (acts : Slice chart.QuantAct)
    (hspec : I8FromNeBytesSpec) (hqh : 4 ≤ qh.val.length) (hqs : 16 ≤ qs.val.length)
    (hw : w = q5Word4 (qh.val.getD 0 0#u8) (qh.val.getD 1 0#u8) (qh.val.getD 2 0#u8)
      (qh.val.getD 3 0#u8))
    (k xbase : Usize) (hk : k.val ≤ acts.val.length) (hk16 : k.val ≤ 16)
    (hxb : xbase.val + 32 ≤ Usize.max)
    (hcodes : ∀ d, d < k.val → xbase.val + 32 ≤ (acts.val.getD d qaDefault).codes.val.length)
    (out : Array I32 16#usize) (c : Usize) (hinv : q5KdotsInv w qs acts xbase k out c) :
    kern.matmat.q5_0_block_kdots_portable_loop.body acts k qh qs xbase out c ⦃ r =>
      match r with
      | ControlFlow.done y => ∀ i, i < 16 → (y[i]!).val = q5ColRef w qs.val acts.val xbase.val k.val i
      | ControlFlow.cont st' => q5KdotsInv w qs acts xbase k st'.1 st'.2 ∧
          k.val - st'.2.val < k.val - c.val ⦄ := by
  simp only [q5KdotsInv] at hinv
  obtain ⟨hck, hout⟩ := hinv
  by_cases hlt : c < k
  · have hc : c.val < k.val := (UScalar.lt_equiv c k).1 hlt
    have hca : c.val < acts.val.length := by omega
    unfold kern.matmat.q5_0_block_kdots_portable_loop.body
    rw [if_pos hlt]
    simp only [chart.QuantAct.impl.codes, lift, bind_tc_ok]
    step*
    · -- `xbase + ACT_QK` does not overflow
      have h32 : (32#usize).val = 32 := by norm_num
      rw [ACT_QK_eq, h32]; omega
    · -- the range end lies inside the column's lanes
      have hAQ : (↑kern.matmat.ACT_QK : Nat) = 32 := by rw [ACT_QK_eq]; norm_num
      have hlen : qa.codes.deref.val.length = (acts.val.getD c.val qaDefault).codes.val.length := by
        rw [qa_post, ← list_getD_eq_getElem acts.val qaDefault hca]
        simp only [alloc.vec.Vec.deref, Slice.from_val]
      have hi : (↑i : Nat) = ↑xbase + 32 := by rw [i_post, hAQ]
      simp only [Slice.length]
      rw [hi, hlen]
      exact hcodes c.val hc
    · -- the portable q5 dot
      refine WP.spec_bind (word_portable_q5_0_dot_ok qh qs s1 hqh hqs ?_) ?_
      · -- `s1` holds the `32` lanes
        have hs : qa.codes.deref.val = (acts.val.getD c.val qaDefault).codes.val := by
          rw [qa_post, list_getD_eq_getElem acts.val qaDefault hca]
          simp only [alloc.vec.Vec.deref, Slice.from_val, Slice.v]
        have s1val : s1.val =
            (acts.val.getD c.val qaDefault).codes.val.slice xbase.val (xbase.val + 32) := by
          rw [s1_post, hs, i_post]
          have hAQ : (↑kern.matmat.ACT_QK : Nat) = 32 := by rw [ACT_QK_eq]; norm_num
          rw [hAQ]
        rw [s1val, slice_len_eq _ _ _ (hcodes c.val hc)]
        omega
      intro i1 hi1
      step*
      · -- invariant after the write
        have hs : qa.codes.deref.val = (acts.val.getD c.val qaDefault).codes.val := by
          rw [qa_post, list_getD_eq_getElem acts.val qaDefault hca]
          simp only [alloc.vec.Vec.deref, Slice.from_val, Slice.v]
        have s1val : s1.val =
            (acts.val.getD c.val qaDefault).codes.val.slice xbase.val (xbase.val + 32) := by
          rw [s1_post, hs, i_post]
          have hAQ : (↑kern.matmat.ACT_QK : Nat) = 32 := by rw [ACT_QK_eq]; norm_num
          rw [hAQ]
        have ho : out.length = 16 := by simpa using Array.property out
        refine ⟨⟨?_, ?_⟩, ?_⟩
        · rw [c1_post]; omega
        · intro i hi
          rw [a_post, c1_post]
          by_cases hic : i = c.val
          · subst hic
            rw [Array.getElem!_Nat_set_eq (h := ⟨rfl, by omega⟩)]
            rw [hi1]
            unfold q5PartRef q5ColRef
            rw [if_pos (Nat.lt_succ_self _), if_pos hc, ← hw, s1val]
          · rw [Array.getElem!_Nat_set_ne (h := fun hc_eq => hic hc_eq.symm)]
            rw [hout i hi]
            unfold q5PartRef
            by_cases hltc : i < c.val
            · rw [if_pos hltc, if_pos (by omega)]
            · rw [if_neg hltc, if_neg (by omega)]
        · rw [c1_post]; omega
  · unfold kern.matmat.q5_0_block_kdots_portable_loop.body
    rw [if_neg hlt]
    simp only [WP.spec_ok]
    have hck' : c.val = k.val := le_antisymm hck (by
      by_contra h
      exact hlt ((UScalar.lt_equiv c k).2 (Nat.lt_of_not_le h)))
    intro i hi
    rw [hout i hi, hck']
    show q5PartRef w qs.val acts.val xbase.val k.val k.val i =
      q5ColRef w qs.val acts.val xbase.val k.val i
    unfold q5PartRef q5ColRef
    by_cases hik : i < k.val <;> simp only [hik, if_true, if_false]

theorem word_q5_0_block_kdots_portable_loop_ok (w : Nat) (qh qs : Slice U8)
    (acts : Slice chart.QuantAct) (hspec : I8FromNeBytesSpec) (hqh : 4 ≤ qh.val.length)
    (hqs : 16 ≤ qs.val.length)
    (hw : w = q5Word4 (qh.val.getD 0 0#u8) (qh.val.getD 1 0#u8) (qh.val.getD 2 0#u8)
      (qh.val.getD 3 0#u8))
    (k xbase : Usize) (hk : k.val ≤ acts.val.length) (hk16 : k.val ≤ 16)
    (hxb : xbase.val + 32 ≤ Usize.max)
    (hcodes : ∀ d, d < k.val → xbase.val + 32 ≤ (acts.val.getD d qaDefault).codes.val.length)
    (out : Array I32 16#usize) (c : Usize) (hinv : q5KdotsInv w qs acts xbase k out c) :
    kern.matmat.q5_0_block_kdots_portable_loop acts k qh qs xbase out c ⦃ r =>
      ∀ i, i < 16 → (r[i]!).val = q5ColRef w qs.val acts.val xbase.val k.val i ⦄ := by
  unfold kern.matmat.q5_0_block_kdots_portable_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array I32 16#usize × Usize => k.val - s.2.val)
    (inv := fun s => q5KdotsInv w qs acts xbase k s.1 s.2)
    (post := fun y : Array I32 16#usize =>
      ∀ i, i < 16 → (y[i]!).val = q5ColRef w qs.val acts.val xbase.val k.val i)
  · rintro ⟨out', c'⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
      (q5_kdots_loop_body_spec w qh qs acts hspec hqh hqs hw k xbase hk hk16 hxb hcodes out' c' hinv')
    rw [hr]
    rcases r with y | s'
    · simpa only [WP.spec_ok] using hpost
    · simpa only [WP.spec_ok] using hpost
  · exact hinv

/-- The `Q5_0` slice `block[2..6]` used for the packed word. -/
theorem word_q5_0_block_kdots_portable_ok (block : Slice U8) (acts : Slice chart.QuantAct)
    (blk k : Usize) (hspec : I8FromNeBytesSpec)
    (hlen : 22 ≤ block.val.length) (hk : k.val ≤ acts.val.length) (hk16 : k.val ≤ 16)
    (hblk : blk.val * 32 + 32 ≤ Usize.max)
    (hcodes : ∀ d, d < k.val → blk.val * 32 + 32 ≤ (acts.val.getD d qaDefault).codes.val.length) :
    kern.matmat.q5_0_block_kdots_portable block acts blk k ⦃ r =>
      ∀ i, i < 16 → (r[i]!).val = q5BlockRef block.val acts.val (blk.val * 32) k.val i ⦄ := by
  unfold kern.matmat.q5_0_block_kdots_portable
  simp only [lift, bind_tc_ok]
  step*
  · -- `block[6..i1]` in bounds
    show i1.val ≤ block.val.length
    have h32 : (32#usize).val = 32 := by norm_num
    rw [i1_post, i_post, Q5_0_QK_eq, h32]
    norm_num
    omega
  · -- `blk * ACT_QK` no overflow
    have h32 : (32#usize).val = 32 := by norm_num
    rw [ACT_QK_eq, h32]; omega
  · -- the loop
    have hxbval : xbase.val = blk.val * 32 := by rw [xbase_post, ACT_QK_eq]; norm_num
    have hqhval : qh.val = block.val.slice 2 6 := by
      rw [qh_post]
    have hqsval : qs.val = block.val.slice 6 22 := by
      rw [qs_post, i1_post, i_post, Q5_0_QK_eq]
      norm_num
    have hqh4 : 4 ≤ qh.val.length := by
      rw [hqhval, slice_len_eq block.val 2 6 (by omega)]
    have hqs16 : 16 ≤ qs.val.length := by
      rw [hqsval, slice_len_eq block.val 6 22 (by omega)]
    have hw : q5Word4 (block.val.getD 2 0#u8) (block.val.getD 3 0#u8) (block.val.getD 4 0#u8)
        (block.val.getD 5 0#u8)
        = q5Word4 (qh.val.getD 0 0#u8) (qh.val.getD 1 0#u8) (qh.val.getD 2 0#u8)
          (qh.val.getD 3 0#u8) := by
      rw [hqhval]
      have e0 := (slice_getD block.val 0#u8 2 6 0 (by omega) (by omega)).symm
      have e1 := (slice_getD block.val 0#u8 2 6 1 (by omega) (by omega)).symm
      have e2 := (slice_getD block.val 0#u8 2 6 2 (by omega) (by omega)).symm
      have e3 := (slice_getD block.val 0#u8 2 6 3 (by omega) (by omega)).symm
      rw [e0, e1, e2, e3]
    refine WP.spec_mono (word_q5_0_block_kdots_portable_loop_ok
      (q5Word4 (block.val.getD 2 0#u8) (block.val.getD 3 0#u8) (block.val.getD 4 0#u8)
        (block.val.getD 5 0#u8))
      qh qs acts hspec hqh4 hqs16 hw k xbase hk hk16 ?_ ?_ (Array.repeat 16#usize 0#i32)
      0#usize ?_) ?_
    · rw [hxbval]; exact hblk
    · intro d hd; rw [hxbval]; exact hcodes d hd
    · refine ⟨by norm_num, ?_⟩
      intro i hi
      have h0 : (0#usize).val = 0 := by norm_num
      unfold q5PartRef
      rw [if_neg (by rw [h0]; omega), Array.getElem!_Nat_eq, Array.repeat_val]
      have h16 : (16#usize).val = 16 := by norm_num
      rw [h16, List.getElem!_replicate 0#i32 hi]
      rfl
    · intro r hr i hi
      dsimp only [q5BlockRef]
      rw [← hqsval, ← hxbval]
      exact hr i hi

/-- The `Q5_0` block-kdots selector: on the portable (`avx = false`) face it slices the `22`-byte
block window and refines the block driver. The `avx = true` face calls the `unsafe` AVX2 axiom and is
not covered. -/
theorem word_q5_0_block_kdots_ok (row : Slice U8) (block_off : Usize) (acts : Slice chart.QuantAct)
    (blk k : Usize) (hspec : I8FromNeBytesSpec)
    (hrow : block_off.val + 22 ≤ row.val.length) (hwov : block_off.val + 22 ≤ Usize.max)
    (hk : k.val ≤ acts.val.length) (hk16 : k.val ≤ 16) (hblk : blk.val * 32 + 32 ≤ Usize.max)
    (hcodes : ∀ d, d < k.val → blk.val * 32 + 32 ≤ (acts.val.getD d qaDefault).codes.val.length) :
    kern.matmat.q5_0_block_kdots false row block_off acts blk k ⦃ r =>
      ∀ i, i < 16 → (r[i]!).val =
        q5BlockRef (row.val.slice block_off.val (block_off.val + 22)) acts.val
          (blk.val * 32) k.val i ⦄ := by
  unfold kern.matmat.q5_0_block_kdots
  simp only [Bool.false_eq_true, if_false, lift, bind_tc_ok]
  rw [quant_Q5_0_BLOCK_BYTES_eq]
  step*
  · have hsval : s.val = row.val.slice block_off.val (block_off.val + 22) := by
      rw [s_post, i1_post]
    refine WP.spec_mono (word_q5_0_block_kdots_portable_ok s acts blk k hspec ?_ hk hk16 hblk ?_) ?_
    · rw [hsval, slice_len_eq row.val block_off.val (block_off.val + 22) hrow]
      omega
    · exact hcodes
    · intro r hr i hi
      dsimp only [q5BlockRef]
      rw [← hsval]
      exact hr i hi

/-! ## Refusals: the failing branches

Each covered body can also `Err`. The checked `usize` arithmetic refuses with `integerOverflow` and the
slice range index refuses with `panic`; the `_ok` theorems carry those as hypotheses, so the partial
domain they exclude is not empty. These theorems exhibit a reachable failing input for each, which is
the refusing-branch witness the `_ok` refinements otherwise omit. -/

/-- `usize::MAX` is at least `1`. -/
theorem usize_max_ge_one : (1 : Nat) ≤ Usize.max := by
  rcases Usize.bounds_eq with h | h <;> rw [h] <;> norm_num [U32.max_eq, U64.max_eq]

/-- `2 ^ usize::BITS = usize::MAX + 1`, used to put the overflow tests in `Nat`. -/
theorem pow_numBits_eq_max_succ : (2 : Nat) ^ (UScalarTy.Usize.numBits) = Usize.max + 1 := by
  rcases System.Platform.numBits_eq with h | h <;>
    simp [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq, h]

/-- A checked `usize` addition past `usize::MAX` refuses with `integerOverflow`. -/
theorem usize_add_overflow (x y : Usize) (h : Usize.max < x.val + y.val) :
    x + y = fail .integerOverflow := by
  change UScalar.add x y = fail .integerOverflow
  unfold UScalar.add UScalar.tryMk UScalar.tryMkOpt
  rw [dif_neg (by
    rw [UScalar.check_bounds_eq_inBounds]; intro hb
    rw [UScalar.inBounds, pow_numBits_eq_max_succ] at hb
    omega)]
  rfl

/-- A checked `usize` multiplication past `usize::MAX` refuses with `integerOverflow`. -/
theorem usize_mul_overflow (x y : Usize) (h : Usize.max < x.val * y.val) :
    x * y = fail .integerOverflow := by
  change UScalar.mul x y = fail .integerOverflow
  unfold UScalar.mul UScalar.tryMk UScalar.tryMkOpt
  rw [dif_neg (by
    rw [UScalar.check_bounds_eq_inBounds]; intro hb
    rw [UScalar.inBounds, pow_numBits_eq_max_succ] at hb
    omega)]
  rfl

/-- A range slice index outside `start ≤ end ≤ len` refuses with `panic`. -/
theorem range_index_fail {T : Type} (r : core.ops.range.Range Usize) (s : Slice T)
    (h : ¬ (r.start ≤ r.end ∧ r.end ≤ Slice.length s)) :
    core.slice.index.Slice.index (core.slice.index.SliceIndexRangeUsizeSlice T) s r = fail .panic := by
  show core.slice.index.SliceIndexRangeUsizeSlice.index r s = fail .panic
  unfold core.slice.index.SliceIndexRangeUsizeSlice.index
  rw [if_neg h]

/-- `portable_u8_i8_dot_loop`'s body refuses on one code byte and no activation lane: the first
out-of-range `xq` read gives `arrayOutOfBounds`. -/
theorem u8_i8_dot_loop_body_refuses (hspec : I8FromNeBytesSpec) (b : U8) :
    kern.matmat.portable_u8_i8_dot_loop.body
      (Slice.from [b] (by simpa using usize_max_ge_one)) (Slice.from [] (by simp))
      0#i32 0#usize = fail .arrayOutOfBounds := by
  unfold kern.matmat.portable_u8_i8_dot_loop.body
  have hlen : (Slice.from [b] (by simpa using usize_max_ge_one) : Slice U8).len = 1#usize := rfl
  rw [hlen]
  rw [if_pos (by rw [UScalar.lt_equiv]; norm_num)]
  rw [slice_index_usize_ok (Slice.from [b] (by simpa using usize_max_ge_one)) 0#usize
        (by rw [Slice.from_val]; simp)]
  simp only [bind_tc_ok]
  rw [I8FromNeBytesSpec_apply hspec _ _]
  simp only [bind_tc_ok]
  simp only [lift, bind_tc_ok]
  rw [slice_index_usize_fail (Slice.from [] (by simp)) 0#usize (by rw [Slice.from_val]; simp)]
  simp only [bind_tc_fail]

/-- `portable_u8_i8_dot` refuses when the activation slice is shorter than the code slice: the loop
reaches the first lane past `xq` and fails there. Witness: one code byte, no activation lane. -/
theorem word_portable_u8_i8_dot_refuses (hspec : I8FromNeBytesSpec) (b : U8) :
    kern.matmat.portable_u8_i8_dot
      (Slice.from [b] (by simpa using usize_max_ge_one)) (Slice.from [] (by simp))
      = fail .arrayOutOfBounds := by
  unfold kern.matmat.portable_u8_i8_dot kern.matmat.portable_u8_i8_dot_loop
  rw [loop.eq_def]
  dsimp only []
  rw [u8_i8_dot_loop_body_refuses hspec b]
  simp only [bind_fail]

/-- `q8_0_block_kdots_portable` refuses when the column base `blk * 32` overflows. -/
theorem word_q8_0_block_kdots_portable_refuses (window : Slice U8) (acts : Slice chart.QuantAct)
    (blk k : Usize) (h : Usize.max < blk.val * 32) :
    kern.matmat.q8_0_block_kdots_portable window acts blk k = fail .integerOverflow := by
  unfold kern.matmat.q8_0_block_kdots_portable
  simp only [lift, bind_tc_ok]
  have h' : Usize.max < blk.val * (kern.matmat.ACT_QK).val := by rw [ACT_QK_eq]; exact h
  rw [usize_mul_overflow blk kern.matmat.ACT_QK h']
  simp only [bind_tc_fail]

/-- `q8_0_block_kdots` (portable face) refuses when the row window `weight_off + 32` overflows. -/
theorem word_q8_0_block_kdots_refuses (row : Slice U8) (weight_off : Usize)
    (acts : Slice chart.QuantAct) (blk k : Usize) (h : Usize.max < weight_off.val + 32) :
    kern.matmat.q8_0_block_kdots false row weight_off acts blk k = fail .integerOverflow := by
  unfold kern.matmat.q8_0_block_kdots
  simp only [Bool.false_eq_true, if_false, lift, bind_tc_ok]
  have h' : Usize.max < weight_off.val + (quant.Q8_0_QK).val := by rw [Q8_0_QK_eq]; exact h
  rw [usize_add_overflow weight_off quant.Q8_0_QK h']
  simp only [bind_tc_fail]

/-- `q5_0_block_kdots` (portable face) refuses when the block end `block_off + 22` overflows. -/
theorem word_q5_0_block_kdots_refuses (row : Slice U8) (block_off : Usize)
    (acts : Slice chart.QuantAct) (blk k : Usize) (h : Usize.max < block_off.val + 22) :
    kern.matmat.q5_0_block_kdots false row block_off acts blk k = fail .integerOverflow := by
  unfold kern.matmat.q5_0_block_kdots
  simp only [Bool.false_eq_true, if_false, lift, bind_tc_ok]
  rw [quant_Q5_0_BLOCK_BYTES_eq]
  simp only [lift, bind_tc_ok]
  have h22 : (22#usize).val = 22 := by norm_num
  have h' : Usize.max < block_off.val + (22#usize).val := by rw [h22]; exact h
  rw [usize_add_overflow block_off 22#usize h']
  simp only [bind_tc_fail]

/-- `q5_0_block_kdots_portable` refuses when the `22`-byte block is shorter than the `6`-byte
packed-word prefix: its first `qh` range read is out of bounds. -/
theorem word_q5_0_block_kdots_portable_refuses (block : Slice U8) (acts : Slice chart.QuantAct)
    (blk k : Usize) (h : block.val.length < 6) :
    kern.matmat.q5_0_block_kdots_portable block acts blk k = fail .panic := by
  unfold kern.matmat.q5_0_block_kdots_portable
  simp only [lift, bind_tc_ok]
  have hr : ¬ ((2#usize) ≤ (6#usize) ∧ (6#usize) ≤ Slice.length block) := by
    rintro ⟨-, h6⟩
    have h6' : (6 : Nat) ≤ block.val.length := by simpa [Slice.length] using h6
    omega
  rw [range_index_fail ⟨2#usize, 6#usize⟩ block hr]
  simp only [bind_tc_fail]

/-! ## Non-vacuity -/

/-- The `Q8_0` reference dot on two constant lanes of `1`. -/
example : u8i8DotRef (List.replicate 2 1#u8) (List.replicate 2 1#i8) 2 = 2 := by decide

/-- A `-1` activation negates the byte sum. -/
example : u8i8DotRef (List.replicate 2 1#u8) (List.replicate 2 (-1)#i8) 2 = -2 := by decide

/-- A column past the projection width `k` has reference dot `0`. -/
example : q8ColRef (List.replicate 32 1#u8) [] 0 3 7 = 0 := by decide

/-- A `Q5_0` column past the projection width `k` has reference dot `0`. -/
example : q5BlockRef (List.replicate 22 1#u8) [] 0 3 7 = 0 := by decide

/-- The leaf refusal has a witness: one code byte and no activation lane. -/
example (hspec : I8FromNeBytesSpec) (b : U8) :
    kern.matmat.portable_u8_i8_dot (Slice.from [b] (by simpa using usize_max_ge_one))
      (Slice.from [] (by simp)) = fail .arrayOutOfBounds :=
  word_portable_u8_i8_dot_refuses hspec b

/-! ## Axiom cone -/

#print axioms word_portable_u8_i8_dot_ok
#print axioms word_q8_0_block_kdots_portable_ok
#print axioms word_q8_0_block_kdots_ok
#print axioms word_q5_0_block_kdots_portable_ok
#print axioms word_q5_0_block_kdots_ok
#print axioms word_portable_u8_i8_dot_refuses
#print axioms word_q8_0_block_kdots_portable_refuses
#print axioms word_q8_0_block_kdots_refuses
#print axioms word_q5_0_block_kdots_refuses
#print axioms word_q5_0_block_kdots_portable_refuses

end apeiron_word
