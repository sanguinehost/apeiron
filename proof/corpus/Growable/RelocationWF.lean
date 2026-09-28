import Growable.Relocation

/-!
# Locator traversal totality (`hstep` chain)

Totality of the generated `Directory.walk` read chain, from the well-formedness predicates of
`Growable.Relocation`, supplying the `hstep` premise of the rebuild traversal.  Bodies read from
`proof/ApeironWord.lean` (digest
`8e76b917bc89cd0b3ea188950d7957d4296a69a9833b4485abda7c7cea8d9b50`).

Three obligations carry the loop chain and are discharged at the explicit premises stated here:

1. **Width.** The generated checked arithmetic (`RESIDUE_STRIDE * i`, `EDGE_STRIDE * i`, and the
   digit/edge-offset adds) must not overflow `usize`.  A node value is a `U32`, so the largest
   derived index is `4 * u32.max + 2` (a residue child read with the largest `U8` digit truncated
   below `ARITY ≤ 3`); the premise `4 * UScalar.max .U32 + 3 ≤ UScalar.max .Usize` carries this and
   is what the 64-bit target satisfies (`System.Platform.numBits = 64`).  On a 32-bit `usize`
   target the premise is false and the generated code genuinely fails with `integerOverflow`; this
   module's theorems then do not apply (stated, not hidden).
2. **Cast values.** `UScalar.cast .Usize e` for `e : U32` is value-preserving; discharged with
   `UScalar.cast_inBounds_spec` (`cast_usize_val` below), not by unfolding `UScalar.cast`.
3. **Loop-state projections.** The `loop.spec_decr_nat` bodies quantify over the pair state; the
   pair projections are normalized (`dsimp only`) before the arithmetic is handed to `omega`, so
   `(e, f).1` and `e` are one atom.
-/

namespace Apeiron.GrowableRelocationWF

open Aeneas Aeneas.Std Result
open apeiron_word
open Apeiron.GrowableRelocation

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- Value preservation of the `U32 → Usize` cast in the generated index arithmetic.  Discharged
from `UScalar.cast_inBounds_spec`; no bit-vector unfolding. -/
theorem cast_usize_val (e : Std.U32) : (UScalar.cast UScalarTy.Usize e).val = e.val := by
  obtain ⟨y, hy, hyval⟩ := WP.spec_imp_exists
    (UScalar.cast_inBounds_spec UScalarTy.Usize e (by scalar_tac))
  have hyok : (Result.ok (UScalar.cast UScalarTy.Usize e) : Result Std.Usize) = Result.ok y := hy
  have heq : UScalar.cast UScalarTy.Usize e = y := Result.ok_injective hyok
  rw [heq, hyval]

/-- Numeric values of the generated layout constants. -/
theorem locator_word_const_vals :
    (coord.locator.locator_core.EDGE_STRIDE : Std.Usize).val = 3 ∧
      (coord.locator.locator_core.EDGE_LABEL : Std.Usize).val = 0 ∧
      (coord.locator.locator_core.EDGE_CHILD : Std.Usize).val = 1 ∧
      (coord.locator.locator_core.EDGE_NEXT : Std.Usize).val = 2 ∧
      (coord.locator.locator_core.IDENTITY_STRIDE : Std.Usize).val = 2 ∧
      (coord.locator.locator_core.IDENTITY_FIRST_EDGE : Std.Usize).val = 1 ∧
      (coord.locator.locator_core.RESIDUE_STRIDE : Std.Usize).val = 4 := by
  simp [coord.locator.locator_core.EDGE_STRIDE, coord.locator.locator_core.EDGE_LABEL,
    coord.locator.locator_core.EDGE_CHILD, coord.locator.locator_core.EDGE_NEXT,
    coord.locator.locator_core.IDENTITY_STRIDE,
    coord.locator.locator_core.IDENTITY_FIRST_EDGE,
    coord.locator.locator_core.RESIDUE_STRIDE]

/-- `usize` leaves room for the `+ 1` loop-counter step below the digit-buffer bound `64`. -/
theorem usize_max_room : 65 ≤ Usize.max := by
  rcases Usize.bounds_eq with h | h <;>
    norm_num [h, U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]

/-- `find_edge_loop` is total under `DirectoryWF` and the width premise: the edge chain strictly
decreases at `EDGE_NEXT`, so the loop closes on the `edge` measure; the result is `NO_CHILD` or an
identity node value (`EDGE_CHILD` class). -/
theorem find_edge_loop_succeeds (d : coord.locator.locator_core.Directory)
    (edge_words : Slice Std.U32) (hwords : edge_words.val = d.edge_words.val)
    (wf : DirectoryWF d) (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (label : Std.U8) (edge found : Std.U32)
    (hedge : EdgeIndexWF d edge)
    (hfound : found = coord.locator.locator_core.NO_CHILD ∨ IdentityNodeWF d found) :
    ∃ r, coord.locator.locator_core.find_edge_loop edge_words label edge found = ok r ∧
      (r = coord.locator.locator_core.NO_CHILD ∨ IdentityNodeWF d r) := by
  unfold coord.locator.locator_core.find_edge_loop
  refine WP.spec_imp_exists (loop.spec_decr_nat
    (fun p : Std.U32 × Std.U32 =>
      if p.1 = coord.locator.locator_core.NO_CHILD then 0 else p.1.val + 1)
    (fun p : Std.U32 × Std.U32 =>
      EdgeIndexWF d p.1 ∧ (p.2 = coord.locator.locator_core.NO_CHILD ∨ IdentityNodeWF d p.2))
    (fun r => r = coord.locator.locator_core.NO_CHILD ∨ IdentityNodeWF d r)
    (fun p => coord.locator.locator_core.find_edge_loop.body edge_words label p.1 p.2)
    (edge, found) ?_ ⟨hedge, hfound⟩)
  rintro ⟨e, f⟩ ⟨he, hf⟩
  dsimp only at he hf ⊢
  unfold coord.locator.locator_core.find_edge_loop.body
  by_cases houter : ((e != coord.locator.locator_core.NO_CHILD) = true)
  · rw [if_pos houter]
    by_cases hinner : f = coord.locator.locator_core.NO_CHILD
    · rw [if_pos hinner]
      have hno : e ≠ coord.locator.locator_core.NO_CHILD := by
        intro h
        simp [h] at houter
      obtain ⟨hbound, hdec⟩ := he.resolve_left hno
      have hcast : (UScalar.cast UScalarTy.Usize e).val = e.val := cast_usize_val e
      have heu : e.val ≤ UScalar.max UScalarTy.U32 := by scalar_tac
      obtain ⟨cStride, cLabel, cChild, cNext, cIStride, cIFirst, cRStride⟩ :=
        locator_word_const_vals
      obtain ⟨atE, hatE, hatEval, _⟩ := WP.spec_imp_exists
        (UScalar.mul_bv_spec (x := coord.locator.locator_core.EDGE_STRIDE)
          (y := UScalar.cast UScalarTy.Usize e)
          (by rw [cStride, hcast]; omega))
      obtain ⟨w0, hw0, hw0val, _⟩ := WP.spec_imp_exists
        (UScalar.add_bv_spec (x := atE) (y := coord.locator.locator_core.EDGE_LABEL)
          (by rw [cLabel, hatEval, cStride, hcast]; omega))
      obtain ⟨w1, hw1, hw1val, _⟩ := WP.spec_imp_exists
        (UScalar.add_bv_spec (x := atE) (y := coord.locator.locator_core.EDGE_CHILD)
          (by rw [cChild, hatEval, cStride, hcast]; omega))
      obtain ⟨w2, hw2, hw2val, _⟩ := WP.spec_imp_exists
        (UScalar.add_bv_spec (x := atE) (y := coord.locator.locator_core.EDGE_NEXT)
          (by rw [cNext, hatEval, cStride, hcast]; omega))
      have hp0 : (↑w0 : Nat) = 3 * ↑e := by rw [hw0val, hatEval, cStride, cLabel, hcast]; omega
      have hp1 : (↑w1 : Nat) = 3 * ↑e + 1 := by rw [hw1val, hatEval, cStride, cChild, hcast]
      have hp2 : (↑w2 : Nat) = 3 * ↑e + 2 := by rw [hw2val, hatEval, cStride, cNext, hcast]
      have hlen : edge_words.val.length = d.edge_words.val.length := by rw [hwords]
      have hb0 : w0.val < edge_words.val.length := by omega
      have hb1 : w1.val < edge_words.val.length := by omega
      have hb2 : w2.val < edge_words.val.length := by omega
      have hread0 : coord.locator.locator_core.read_word edge_words w0 =
          ok edge_words.val[w0.val] := read_word_ok edge_words w0 hb0
      have hread1 : coord.locator.locator_core.read_word edge_words w1 =
          ok edge_words.val[w1.val] := read_word_ok edge_words w1 hb1
      have hread2 : coord.locator.locator_core.read_word edge_words w2 =
          ok edge_words.val[w2.val] := read_word_ok edge_words w2 hb2
      have hget1 : d.edge_words.val[3 * e.val + 1]? = some edge_words.val[w1.val] := by
        rw [← hwords, ← hp1]
        exact List.getElem?_eq_getElem hb1
      have hget2 : d.edge_words.val[3 * e.val + 2]? = some edge_words.val[w2.val] := by
        rw [← hwords, ← hp2]
        exact List.getElem?_eq_getElem hb2
      have hpc1 : (3 * e.val + 1) % 3 = 1 := by omega
      have hpc2 : (3 * e.val + 2) % 3 = 2 := by omega
      have hedge2 : EdgeIndexWF d edge_words.val[w2.val] := wf.edge_next_bound _ _ hget2 hpc2
      simp only [lift, bind_tc_ok]
      rw [hatE]
      simp only [bind_tc_ok]
      rw [hw0]
      simp only [bind_tc_ok]
      rw [hread0]
      simp only [bind_tc_ok]
      rw [hw2]
      simp only [bind_tc_ok]
      rw [hread2]
      simp only [bind_tc_ok]
      by_cases hmatch : edge_words.val[w0.val] =
          core.convert.num.FromU32U8.from label
      · rw [if_pos hmatch, hw1]
        simp only [bind_tc_ok]
        rw [hread1]
        simp only [bind_tc_ok, WP.spec_ok]
        refine ⟨⟨hedge2, ?_⟩, ?_⟩
        · by_cases hc : edge_words.val[w1.val] = coord.locator.locator_core.NO_CHILD
          · exact Or.inl hc
          · exact Or.inr (wf.edge_child_bound _ _ hget1 hpc1 hc)
        · by_cases hn : edge_words.val[w2.val] = coord.locator.locator_core.NO_CHILD
          · rw [if_pos hn, if_neg hno]
            omega
          · rw [if_neg hn, if_neg hno]
            rcases hdec edge_words.val[w2.val] hget2 with h | h
            · exact absurd h hn
            · omega
      · rw [if_neg hmatch]
        simp only [bind_tc_ok, WP.spec_ok]
        refine ⟨⟨hedge2, hf⟩, ?_⟩
        by_cases hn : edge_words.val[w2.val] = coord.locator.locator_core.NO_CHILD
        · rw [if_pos hn, if_neg hno]
          omega
        · rw [if_neg hn, if_neg hno]
          rcases hdec edge_words.val[w2.val] hget2 with h | h
          · exact absurd h hn
          · omega
    · rw [if_neg hinner]
      simp only [WP.spec_ok]
      exact Or.inr (hf.resolve_left hinner)
  · rw [if_neg houter]
    simp only [WP.spec_ok]
    exact hf

/-- `find_edge` is total under `DirectoryWF` and the width premise: the first-edge read is in
bounds and the edge loop terminates; the result is `NO_CHILD` or an identity node value. -/
theorem find_edge_succeeds (d : coord.locator.locator_core.Directory)
    (identity_words edge_words : Slice Std.U32)
    (hI : identity_words.val = d.identity_words.val) (hE : edge_words.val = d.edge_words.val)
    (wf : DirectoryWF d)
    (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (node : Std.U32) (label : Std.U8) (hnode : IdentityNodeWF d node) :
    ∃ r, coord.locator.locator_core.find_edge identity_words edge_words node label = ok r ∧
      (r = coord.locator.locator_core.NO_CHILD ∨ IdentityNodeWF d r) := by
  unfold coord.locator.locator_core.find_edge
  have hcast : (UScalar.cast UScalarTy.Usize node).val = node.val := cast_usize_val node
  have heu : node.val ≤ UScalar.max UScalarTy.U32 := by scalar_tac
  obtain ⟨cStride, cLabel, cChild, cNext, cIStride, cIFirst, cRStride⟩ :=
    locator_word_const_vals
  obtain ⟨iN, hiN, hiNval, _⟩ := WP.spec_imp_exists
    (UScalar.mul_bv_spec (x := coord.locator.locator_core.IDENTITY_STRIDE)
      (y := UScalar.cast UScalarTy.Usize node)
      (by rw [cIStride, hcast]; omega))
  obtain ⟨wE, hwE, hwEval, _⟩ := WP.spec_imp_exists
    (UScalar.add_bv_spec (x := iN) (y := coord.locator.locator_core.IDENTITY_FIRST_EDGE)
      (by rw [cIFirst, hiNval, cIStride, hcast]; omega))
  have hpE : (↑wE : Nat) = 2 * ↑node + 1 := by
    rw [hwEval, hiNval, cIStride, cIFirst, hcast]
  have hbE : wE.val < identity_words.val.length := by
    rw [hpE, hI]
    exact hnode
  have hreadE : coord.locator.locator_core.read_word identity_words wE =
      ok identity_words.val[wE.val] := read_word_ok identity_words wE hbE
  have hgetE : d.identity_words.val[2 * node.val + 1]? = some identity_words.val[wE.val] := by
    rw [← hI, ← hpE]
    exact List.getElem?_eq_getElem hbE
  simp only [lift, bind_tc_ok]
  rw [hiN]
  simp only [bind_tc_ok]
  rw [hwE]
  simp only [bind_tc_ok]
  rw [hreadE]
  simp only [bind_tc_ok]
  exact find_edge_loop_succeeds d edge_words hE wf hwide label identity_words.val[wE.val]
    coord.locator.locator_core.NO_CHILD
    (DirectoryWF.edgeIndexWF_of_first_edge wf hgetE (by omega))
    (Or.inl rfl)

/-- `walk_residue_loop` is total under `DirectoryWF`, the width premise, and the digit bound
(`massert (i < arity)`), and preserves the residue-node bound; the loop closes on the
`levels - depth` measure. -/
theorem walk_residue_loop_succeeds (d : coord.locator.locator_core.Directory)
    (residue_words : Slice Std.U32) (hwords : residue_words.val = d.residue_words.val)
    (wf : DirectoryWF d) (digits : Array Std.U8 64#usize) (levels arity : Std.Usize)
    (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (hlevels : levels.val ≤ 64) (har3 : arity.val ≤ 3)
    (hdigits : ∀ k < levels.val, ∀ dv, digits.val[k]? = some dv →
      core.convert.num.FromUsizeU8.from dv < arity)
    (node : Std.U32) (depth : Std.Usize)
    (hnode : ResidueNodeWF d node) (hdepth : depth.val ≤ levels.val) :
    ∃ r, coord.locator.locator_core.walk_residue_loop residue_words digits levels arity node depth =
      ok r ∧ ResidueNodeWF d r.1 := by
  have h64 : ((64#usize : Std.Usize).val : Nat) = 64 := by simp
  have hlenD : digits.val.length = (64#usize : Std.Usize).val := digits.property
  unfold coord.locator.locator_core.walk_residue_loop
  refine WP.spec_imp_exists (loop.spec_decr_nat
    (fun p : Std.U32 × Std.Usize => levels.val - p.2.val)
    (fun p : Std.U32 × Std.Usize => ResidueNodeWF d p.1 ∧ p.2.val ≤ levels.val)
    (fun r => ResidueNodeWF d r.1)
    (fun p => coord.locator.locator_core.walk_residue_loop.body residue_words digits levels arity
      p.1 p.2)
    (node, depth) ?_ ⟨hnode, hdepth⟩)
  rintro ⟨e, dp⟩ ⟨he, hdp⟩
  dsimp only at he hdp ⊢
  unfold coord.locator.locator_core.walk_residue_loop.body
  by_cases houter : (dp < levels)
  · rw [if_pos houter]
    have hdpv : dp.val < levels.val := (UScalar.lt_equiv dp levels).mp houter
    by_cases hinner : ((e != coord.locator.locator_core.NO_CHILD) = true)
    · rw [if_pos hinner]
      have hcast : (UScalar.cast UScalarTy.Usize e).val = e.val := cast_usize_val e
      have heu : e.val ≤ UScalar.max UScalarTy.U32 := by scalar_tac
      obtain ⟨cStride, cLabel, cChild, cNext, cIStride, cIFirst, cRStride⟩ :=
        locator_word_const_vals
      have hboundD : dp.val < digits.length := by simp [Array.length, hlenD]; omega
      have hreadD : Array.index_usize digits dp = ok digits.val[dp.val] := by
        obtain ⟨y, hy, rfl⟩ := WP.spec_imp_exists (Array.index_usize_spec digits dp hboundD)
        exact hy
      have hdigitGet : digits.val[dp.val]? = some digits.val[dp.val] :=
        List.getElem?_eq_getElem (by omega)
      have hdigBound : core.convert.num.FromUsizeU8.from digits.val[dp.val] < arity :=
        hdigits dp.val (by omega) digits.val[dp.val] hdigitGet
      have hfv : (core.convert.num.FromUsizeU8.from digits.val[dp.val]).val =
          digits.val[dp.val].val := core.convert.num.FromUsizeU8.from_val_eq _
      have hlt3 : digits.val[dp.val].val < 3 := by
        have hlt := (UScalar.lt_equiv _ _).mp hdigBound
        omega
      have hmassert : Aeneas.Std.massert
          (core.convert.num.FromUsizeU8.from digits.val[dp.val] < arity) = ok () := by
        simp only [Aeneas.Std.massert]
        rw [if_pos hdigBound]
      obtain ⟨prod, hprod, hprodval, _⟩ := WP.spec_imp_exists
        (UScalar.mul_bv_spec (x := coord.locator.locator_core.RESIDUE_STRIDE)
          (y := UScalar.cast UScalarTy.Usize e)
          (by rw [cRStride, hcast]; omega))
      obtain ⟨sum, hsum, hsumval, _⟩ := WP.spec_imp_exists
        (UScalar.add_bv_spec (x := prod)
          (y := core.convert.num.FromUsizeU8.from digits.val[dp.val])
          (by rw [hprodval, cRStride, hcast, hfv]; omega))
      have h1val : ((1#usize : Std.Usize).val : Nat) = 1 := by simp
      obtain ⟨d1, hd1, hd1val, _⟩ := WP.spec_imp_exists
        (UScalar.add_bv_spec (x := dp) (y := 1#usize)
          (by
            have hroom : (65 : Nat) ≤ UScalar.max UScalarTy.Usize := by
              rw [UScalar.max_USize_eq]
              exact usize_max_room
            rw [h1val]
            omega))
      have hpSum : (↑sum : Nat) = 4 * ↑e + digits.val[dp.val].val := by
        rw [hsumval, hprodval, cRStride, hcast, hfv]
      have hpMod : sum.val % 4 ≠ 3 := by omega
      rw [hreadD]
      simp only [lift, bind_tc_ok]
      rw [hmassert]
      simp only [bind_tc_ok]
      rw [hprod]
      simp only [bind_tc_ok]
      rw [hsum]
      simp only [bind_tc_ok]
      rw [hd1]
      simp only [bind_tc_ok]
      by_cases hguard : (sum < Slice.len residue_words)
      · rw [if_pos hguard]
        have hbA : sum.val < residue_words.val.length := by
          have hlt := (UScalar.lt_equiv sum (Slice.len residue_words)).mp hguard
          simpa [Slice.len] using hlt
        rw [show coord.locator.locator_core.read_word residue_words sum =
            ok residue_words.val[sum.val] from read_word_ok residue_words sum hbA]
        simp only [bind_tc_ok, WP.spec_ok]
        have hget : d.residue_words.val[sum.val]? = some residue_words.val[sum.val] := by
          rw [← hwords]
          exact List.getElem?_eq_getElem hbA
        refine ⟨⟨DirectoryWF.residueNodeWF_of_child wf hget hpMod, ?_⟩, ?_⟩
        · omega
        · omega
      · rw [if_neg hguard]
        simp only [bind_tc_ok, WP.spec_ok]
        refine ⟨⟨Or.inl rfl, ?_⟩, ?_⟩
        · omega
        · omega
    · rw [if_neg hinner]
      simp only [WP.spec_ok]
      exact he
  · rw [if_neg houter]
    simp only [WP.spec_ok]
    exact he

/-- `walk_residue` is total under `DirectoryWF`, the width premise, and the digit bound; its
returned node is a residue node value. -/
theorem walk_residue_succeeds (d : coord.locator.locator_core.Directory)
    (residue_words : Slice Std.U32) (hwords : residue_words.val = d.residue_words.val)
    (wf : DirectoryWF d) (digits : Array Std.U8 64#usize) (levels arity : Std.Usize)
    (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (hlevels : levels.val ≤ 64) (har3 : arity.val ≤ 3)
    (hdigits : ∀ k < levels.val, ∀ dv, digits.val[k]? = some dv →
      core.convert.num.FromUsizeU8.from dv < arity)
    (root : Std.U32) (hroot : ResidueNodeWF d root) :
    ∃ r, coord.locator.locator_core.walk_residue residue_words digits levels arity root = ok r ∧
      ResidueNodeWF d r.1 := by
  unfold coord.locator.locator_core.walk_residue
  obtain ⟨r, hr, hrpost⟩ := walk_residue_loop_succeeds d residue_words hwords wf digits levels
    arity hwide hlevels har3 hdigits root 0#usize hroot (by scalar_tac)
  rcases r with ⟨rn, rd⟩
  rw [hr]
  simp only [bind_tc_ok]
  refine ⟨(rn, core.num.Usize.saturating_sub rd 1#usize), ?_, hrpost⟩
  simp [lift]

/-- Numeric value of `IDENTITY_TERMINAL` (the terminal word sits at `2v + 0`). -/
theorem identity_terminal_val :
    (coord.locator.locator_core.IDENTITY_TERMINAL : Std.Usize).val = 0 := by
  simp [coord.locator.locator_core.IDENTITY_TERMINAL]

/-- Numeric value of `RESIDUE_IDENTITY` (the identity root sits at `4v + 3`). -/
theorem residue_identity_val :
    (coord.locator.locator_core.RESIDUE_IDENTITY : Std.Usize).val = 3 := by
  simp [coord.locator.locator_core.RESIDUE_IDENTITY]

/-- `alloc.vec.Vec.deref` presents the vector's stored list unchanged. -/
theorem vec_deref_val {T : Type} (v : alloc.vec.Vec T) :
    (alloc.vec.Vec.deref v).val = v.val := by
  simp [alloc.vec.Vec.deref]

/-- `walk_identity_loop` is total under `DirectoryWF`, `ExactIdentityWF`, the width premise and a
successful `LEVELS`: the loop closes on the `encoding_len − index` measure (every continued state
advances `index` by one below `len`), and its post `IdentityNodeWF d r.2.1` bounds the terminal read
`identity_words[2v]` performed by `walk_identity`; the loop returns its input directory unchanged
(`r.1 = d`). -/
theorem walk_identity_loop_succeeds {I K Ret Res : Type}
    (resIdx : coord.locator.locator_core.ResidueIndex I)
    (exId : coord.locator.locator_core.ExactIdentity K Ret Res)
    (d : coord.locator.locator_core.Directory) (wf : DirectoryWF d) (ei : ExactIdentityWF exId)
    (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (hlev : ∃ l, resIdx.LEVELS = ok l)
    (key : K) (len : Std.Usize) (root : Std.U32) (hroot : IdentityNodeWF d root)
    (miss : Option coord.locator.locator_core.Walk) (index : Std.Usize) :
    ∃ r, coord.locator.locator_core.Directory.walk_identity_loop resIdx exId d key len root miss
        index = ok r ∧ (IdentityNodeWF d r.2.1 ∧ r.1 = d) := by
  unfold coord.locator.locator_core.Directory.walk_identity_loop
  apply WP.spec_imp_exists
  apply loop.spec_decr_nat
    (measure := fun p : Std.U32 × Option coord.locator.locator_core.Walk × Std.Usize =>
      len.val - p.2.2.val)
    (inv := fun p : Std.U32 × Option coord.locator.locator_core.Walk × Std.Usize =>
      IdentityNodeWF d p.1)
    (post := fun r : coord.locator.locator_core.Directory × Std.U32 ×
      Option coord.locator.locator_core.Walk => IdentityNodeWF d r.2.1 ∧ r.1 = d)
  · rintro ⟨nd, ms, ix⟩ hnd
    dsimp only at hnd ⊢
    unfold coord.locator.locator_core.Directory.walk_identity_loop.body
    by_cases houter : (ix < len)
    · rw [if_pos houter]
      cases hms : ms with
      | none =>
        simp only [core.option.Option.is_none, Option.isNone_none, if_true]
        obtain ⟨b, hb⟩ := ei.encoding_byte_total key ix
        rw [hb]
        simp only [bind_tc_ok]
        have hI : (alloc.vec.Vec.deref d.identity_words).val = d.identity_words.val := by
          simp [alloc.vec.Vec.deref]
        have hE : (alloc.vec.Vec.deref d.edge_words).val = d.edge_words.val := by
          simp [alloc.vec.Vec.deref]
        obtain ⟨child, hchild, hchildpost⟩ := find_edge_succeeds d
          (alloc.vec.Vec.deref d.identity_words) (alloc.vec.Vec.deref d.edge_words)
          hI hE wf hwide nd b hnd
        rw [hchild]
        simp only [bind_tc_ok]
        have hcap : ix.val + 1 ≤ Usize.max := by
          have hlt := (UScalar.lt_equiv ix len).mp houter
          have hlenBounds := len.hBounds
          simp only [UScalarTy.Usize_numBits_eq] at hlenBounds
          have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
            simp [Usize.max, Usize.numBits]
          rw [hmax]
          omega
        by_cases hno : child = coord.locator.locator_core.NO_CHILD
        · rw [if_pos hno]
          obtain ⟨l, hl⟩ := hlev
          rw [hl]
          simp only [bind_tc_ok]
          obtain ⟨ix1, hix1, hix1val⟩ := WP.spec_imp_exists
            (Usize.add_spec (x := ix) (y := 1#usize) hcap)
          rw [hix1]
          simp only [bind_tc_ok]
          refine ⟨hnd, ?_⟩
          dsimp only
          have hix1v : ix1.val = ix.val + 1 := by simpa using hix1val
          have hlt := (UScalar.lt_equiv ix len).mp houter
          omega
        · rw [if_neg hno]
          simp only [bind_tc_ok]
          obtain ⟨ix1, hix1, hix1val⟩ := WP.spec_imp_exists
            (Usize.add_spec (x := ix) (y := 1#usize) hcap)
          rw [hix1]
          simp only [bind_tc_ok]
          refine ⟨hchildpost.resolve_left hno, ?_⟩
          dsimp only
          have hix1v : ix1.val = ix.val + 1 := by simpa using hix1val
          have hlt := (UScalar.lt_equiv ix len).mp houter
          omega
      | some m =>
        simp only [core.option.Option.is_none, Option.isNone_some, Bool.false_eq_true,
          if_false, WP.spec_ok]
        exact ⟨hnd, by simp⟩
    · rw [if_neg houter]
      simp only [WP.spec_ok]
      exact ⟨hnd, by simp⟩
  · exact hroot

/-- `walk_identity` is total under `DirectoryWF`, `ExactIdentityWF`, the width premise and a
successful `LEVELS`: the loop closes on `encoding_len − index`, and its post bounds the terminal
read `identity_words[2v]`; both terminal outcomes (`Walk.Miss`, `Walk.Terminal`) are returned `ok`. -/
theorem walk_identity_succeeds {I K Ret Res : Type}
    (resIdx : coord.locator.locator_core.ResidueIndex I)
    (exId : coord.locator.locator_core.ExactIdentity K Ret Res)
    (d : coord.locator.locator_core.Directory) (wf : DirectoryWF d) (ei : ExactIdentityWF exId)
    (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (hlev : ∃ l, resIdx.LEVELS = ok l)
    (key : K) (root : Std.U32) (hroot : IdentityNodeWF d root) :
    ∃ w, coord.locator.locator_core.Directory.walk_identity resIdx exId d root key = ok w := by
  unfold coord.locator.locator_core.Directory.walk_identity
  obtain ⟨len, hlen⟩ := ei.encoding_len_total key
  rw [hlen]
  simp only [bind_tc_ok]
  obtain ⟨r, hr, hrpost⟩ := walk_identity_loop_succeeds resIdx exId d wf ei hwide hlev key len root
    hroot none 0#usize
  rcases r with ⟨d1, idn, ms⟩
  rcases hrpost with ⟨hidn, hd1⟩
  have hd1e : d = d1 := (hd1).symm
  subst hd1e
  rw [hr]
  simp only [bind_tc_ok]
  cases hms : ms with
  | some m => exact ⟨m, rfl⟩
  | none =>
    change ∃ w, (do
        let i ← lift (UScalar.cast UScalarTy.Usize idn)
        let i1 ← coord.locator.locator_core.IDENTITY_STRIDE * i
        let i2 ← i1 + coord.locator.locator_core.IDENTITY_TERMINAL
        let terminal ← coord.locator.locator_core.read_word (alloc.vec.Vec.deref d.identity_words) i2
        if terminal = coord.locator.locator_core.NO_CHILD then
          do
            let i3 ← resIdx.LEVELS
            ok (coord.locator.locator_core.Walk.Miss i3 len)
        else ok (coord.locator.locator_core.Walk.Terminal terminal)) = ok w
    have hcast : (UScalar.cast UScalarTy.Usize idn).val = idn.val := cast_usize_val idn
    have heu : idn.val ≤ UScalar.max UScalarTy.U32 := by scalar_tac
    obtain ⟨cStride, cLabel, cChild, cNext, cIStride, cIFirst, cRStride⟩ :=
      locator_word_const_vals
    have cITerm := identity_terminal_val
    obtain ⟨i1, hi1, hi1val, _⟩ := WP.spec_imp_exists
      (UScalar.mul_bv_spec (x := coord.locator.locator_core.IDENTITY_STRIDE)
        (y := UScalar.cast UScalarTy.Usize idn)
        (by rw [cIStride, hcast]; omega))
    obtain ⟨i2, hi2, hi2val, _⟩ := WP.spec_imp_exists
      (UScalar.add_bv_spec (x := i1) (y := coord.locator.locator_core.IDENTITY_TERMINAL)
        (by rw [cITerm, hi1val, cIStride, hcast]; omega))
    have hp2 : (↑i2 : Nat) = 2 * ↑idn := by
      rw [hi2val, hi1val, cIStride, cITerm, hcast]; omega
    have hI : (alloc.vec.Vec.deref d.identity_words).val = d.identity_words.val := by
      simp [alloc.vec.Vec.deref]
    have hbE : i2.val < (alloc.vec.Vec.deref d.identity_words).val.length := by
      rw [hI]
      have hbn : 2 * idn.val + 1 < d.identity_words.val.length := hidn
      omega
    have hread : coord.locator.locator_core.read_word (alloc.vec.Vec.deref d.identity_words) i2 =
        ok (alloc.vec.Vec.deref d.identity_words).val[i2.val] :=
      read_word_ok (alloc.vec.Vec.deref d.identity_words) i2 hbE
    simp only [lift, bind_tc_ok]
    rw [hi1]
    simp only [bind_tc_ok]
    rw [hi2]
    simp only [bind_tc_ok]
    rw [hread]
    simp only [bind_tc_ok]
    by_cases hterm : (alloc.vec.Vec.deref d.identity_words).val[i2.val] =
        coord.locator.locator_core.NO_CHILD
    · rw [if_pos hterm]
      obtain ⟨l, hl⟩ := hlev
      rw [hl]
      simp only [bind_tc_ok]
      exact ⟨_, rfl⟩
    · rw [if_neg hterm]
      exact ⟨_, rfl⟩

/-- `Directory.walk` is total under `ResidueIndexWF`, `DirectoryWF`, `ExactIdentityWF` and the width
premise, given the arena-root bound `harena`.  The walk prelude's `Array.index_mut` readback
(`(back s').val = a.val.setSlice! 0 s'.val`) transports `ResidueIndexWF.digits_bounded` across the
writeback into the `walk_residue` digit bound; `ResidueNodeWF d node` bounds the
`residue_identity_root` read `residue_words[4 * node + 3]`, and `walk_identity_succeeds` closes the
identity chain.  `harena` is supplied by `ResidueIndexWF.root_bound` when `levels = 0`; with
`levels > 0` it follows from `DirectoryWF` after one residue step, and discharging it from
`ResidueIndexWF` alone needs a weakened `walk_residue_loop` entry invariant (recorded residual). -/
theorem directory_walk_succeeds {I K Ret Res : Type}
    (resIdx : coord.locator.locator_core.ResidueIndex I)
    (exId : coord.locator.locator_core.ExactIdentity K Ret Res)
    (d : coord.locator.locator_core.Directory) (levels arity : Std.Usize) (arena : Std.U32)
    (rwf : ResidueIndexWF resIdx d levels arity arena) (wf : DirectoryWF d)
    (ei : ExactIdentityWF exId)
    (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (harena : ResidueNodeWF d arena) (index : I) (key : K) :
    ∃ w, coord.locator.locator_core.Directory.walk resIdx exId d index key = ok w := by
  unfold coord.locator.locator_core.Directory.walk
  rw [rwf.levels_eq]
  simp only [bind_tc_ok]
  have h64 : ((64#usize : Std.Usize).val : Nat) = 64 := by simp
  have hle : (levels : Std.Usize) ≤ 64#usize := (UScalar.le_equiv levels 64#usize).mpr
    (by have hlevels_le := rwf.levels_le; omega)
  obtain ⟨⟨s, back⟩, hidx, hp⟩ := WP.spec_imp_exists
    (Array.index_mut_SliceIndexRangeToUsizeSlice (Array.repeat 64#usize 0#u8)
      { «end» := levels } hle)
  simp only [WP.uncurry'_eq] at hp
  rw [hidx]
  simp only [bind_tc_ok]
  change ∃ w, (do
      let s1 ← resIdx.digits index s
      let arn ← resIdx.ARENA
      let ary ← resIdx.ARITY
      let (node, edge_depth) ← coord.locator.locator_core.walk_residue
        (alloc.vec.Vec.deref d.residue_words) (back s1) levels ary arn
      if node = coord.locator.locator_core.NO_CHILD then
        ok (coord.locator.locator_core.Walk.Miss edge_depth 0#usize)
      else do
        let identity_root ← coord.locator.locator_core.Directory.residue_identity_root d node
        if identity_root = coord.locator.locator_core.NO_CHILD then
          ok (coord.locator.locator_core.Walk.Miss levels 0#usize)
        else coord.locator.locator_core.Directory.walk_identity resIdx exId d identity_root key) =
      ok w
  have hslen : s.val.length = levels.val := by simpa using hp.2.1
  obtain ⟨digitsOut, hdig, hdiglen, hdigbound⟩ := rwf.digits_bounded index s hslen
  rw [hdig]
  simp only [bind_tc_ok]
  rw [rwf.arena_eq]
  simp only [bind_tc_ok]
  rw [rwf.arity_eq]
  simp only [bind_tc_ok]
  have hdig1 : ∀ k < levels.val, ∀ dv, (back digitsOut).val[k]? = some dv →
      core.convert.num.FromUsizeU8.from dv < arity := by
    intro k hk dv hdv
    have hset : (back digitsOut).val = (Array.repeat 64#usize 0#u8).val.setSlice! 0 digitsOut.val :=
      hp.2.2 digitsOut
    rw [hset] at hdv
    have hmid := List.setSlice!_getElem?_middle (Array.repeat 64#usize 0#u8).val digitsOut.val 0 k
      ⟨by omega, by rw [hdiglen, hslen]; omega,
        by have hlevels_le := rwf.levels_le; simp [Array.repeat_val]; omega⟩
    rw [hmid] at hdv
    exact hdigbound k dv hk hdv
  have hwordsS : (alloc.vec.Vec.deref d.residue_words).val = d.residue_words.val := by
    simp [alloc.vec.Vec.deref]
  obtain ⟨r, hr, hrpost⟩ := walk_residue_succeeds d (alloc.vec.Vec.deref d.residue_words)
    hwordsS wf (back digitsOut) levels arity hwide rwf.levels_le rwf.arity_le hdig1 arena harena
  rcases r with ⟨node, edepth⟩
  rw [hr]
  simp only [bind_tc_ok]
  change ∃ w, (if node = coord.locator.locator_core.NO_CHILD then
      ok (coord.locator.locator_core.Walk.Miss edepth 0#usize)
    else do
      let identity_root ← coord.locator.locator_core.Directory.residue_identity_root d node
      if identity_root = coord.locator.locator_core.NO_CHILD then
        ok (coord.locator.locator_core.Walk.Miss levels 0#usize)
      else coord.locator.locator_core.Directory.walk_identity resIdx exId d identity_root key) = ok w
  by_cases hn : node = coord.locator.locator_core.NO_CHILD
  · rw [if_pos hn]
    exact ⟨_, rfl⟩
  · rw [if_neg hn]
    have hcast : (UScalar.cast UScalarTy.Usize node).val = node.val := cast_usize_val node
    have heu : node.val ≤ UScalar.max UScalarTy.U32 := by scalar_tac
    have cRId := residue_identity_val
    obtain ⟨cStride, cLabel, cChild, cNext, cIStride, cIFirst, cRStride⟩ :=
      locator_word_const_vals
    obtain ⟨i1, hi1, hi1val, _⟩ := WP.spec_imp_exists
      (UScalar.mul_bv_spec (x := coord.locator.locator_core.RESIDUE_STRIDE)
        (y := UScalar.cast UScalarTy.Usize node)
        (by rw [cRStride, hcast]; omega))
    obtain ⟨i2, hi2, hi2val, _⟩ := WP.spec_imp_exists
      (UScalar.add_bv_spec (x := i1) (y := coord.locator.locator_core.RESIDUE_IDENTITY)
        (by rw [cRId, hi1val, cRStride, hcast]; omega))
    have hp2 : (↑i2 : Nat) = 4 * ↑node + 3 := by
      rw [hi2val, hi1val, cRStride, cRId, hcast]
    have hbA : i2.val < d.residue_words.val.length := by
      have hbound : 4 * node.val + 3 < d.residue_words.val.length := Or.resolve_left hrpost hn
      omega
    have hbA' : i2.val < (alloc.vec.Vec.deref d.residue_words).val.length := by
      rw [hwordsS]
      exact hbA
    have hread : coord.locator.locator_core.Directory.residue_identity_root d node =
        ok (alloc.vec.Vec.deref d.residue_words).val[i2.val] := by
      unfold coord.locator.locator_core.Directory.residue_identity_root
      simp only [lift, bind_tc_ok]
      rw [hi1]
      simp only [bind_tc_ok]
      rw [hi2]
      simp only [bind_tc_ok]
      rw [read_word_ok (alloc.vec.Vec.deref d.residue_words) i2 hbA']
    rw [hread]
    simp only [bind_tc_ok]
    by_cases hird : (alloc.vec.Vec.deref d.residue_words).val[i2.val] =
        coord.locator.locator_core.NO_CHILD
    · rw [if_pos hird]
      exact ⟨_, rfl⟩
    · rw [if_neg hird]
      have hident : IdentityNodeWF d (alloc.vec.Vec.deref d.residue_words).val[i2.val] := by
        apply DirectoryWF.identityNodeWF_of_identity_root wf (p := i2.val)
        · simp only [hwordsS]
          exact List.getElem?_eq_getElem hbA
        · have := hp2
          omega
        · exact hird
      exact walk_identity_succeeds resIdx exId d wf ei hwide ⟨levels, rwf.levels_eq⟩ key
        ((alloc.vec.Vec.deref d.residue_words).val[i2.val]) hident

/-- `table.TableLocator.linked_handle` succeeds under directory well-formedness, exact identity
well-formedness, width bounds, totality of key conversion and coordinate projections, and residue
index well-formedness for active directory branches. -/
theorem linked_handle_succeeds {K H R Faces : Type}
    (cloneInst : core.clone.Clone K) (homeInst : table.Home H K Faces)
    (keyRep : table.keyrep.KeyRep R K H) (self : table.TableLocator)
    (home : H) (sigma : cell.Sigma) (held : R)
    (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (hei : ExactIdentityWF (table.QueryIdentity.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityTupleTuple keyRep))
    (hdir : ∀ d, self.directory = some d → DirectoryWF d)
    (hkey : ∃ k, keyRep.key_owned sigma held = ok k)
    (hdual : ∀ k, ∃ idx, homeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceResidueCapabilityInst.parabolic_dual home k = ok idx)
    (hscalar : ∀ prof k, ∃ idx, homeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceResidueCapabilityInst.hyperbolic_scalar home prof k = ok idx)
    (hpar : ∀ d, self.directory = some d → sigma = cell.Sigma.Parabolic →
      ∃ levels arity arena, ResidueIndexWF coord.address.ParabolicDual.Insts.Apeiron_wordCoordLocatorLocator_coreResidueIndex d levels arity arena ∧ ResidueNodeWF d arena)
    (hhyp : ∀ d prof, self.directory = some d → sigma = cell.Sigma.Hyperbolic → self.profile = some prof →
      ∃ levels arity arena, ResidueIndexWF coord.address.Residue.Insts.Apeiron_wordCoordLocatorLocator_coreResidueIndex d levels arity arena ∧ ResidueNodeWF d arena) :
    ∃ h, table.TableLocator.linked_handle cloneInst homeInst keyRep self home sigma held = ok h := by
  unfold table.TableLocator.linked_handle
  cases hdir_case : self.directory with
  | none =>
    simp only [bind_tc_ok, core.option.Option.as_ref,
      core.option.Option.Insts.CoreOpsTry_traitTry.branch,
      core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]
    exact ⟨none, rfl⟩
  | some d =>
    simp only [bind_tc_ok, core.option.Option.as_ref,
      core.option.Option.Insts.CoreOpsTry_traitTry.branch]
    cases sigma with
    | Elliptic =>
      simp only [bind_tc_ok]
      exact ⟨none, rfl⟩
    | Parabolic =>
      simp only [bind_tc_ok]
      obtain ⟨k, hk⟩ := hkey
      rw [hk]
      simp only [bind_tc_ok]
      obtain ⟨idx, hidx⟩ := hdual k
      rw [hidx]
      simp only [bind_tc_ok]
      obtain ⟨levels, arity, arena, rwf, harena⟩ := hpar d hdir_case rfl
      have hwf := hdir d hdir_case
      obtain ⟨w, hw⟩ := directory_walk_succeeds
        coord.address.ParabolicDual.Insts.Apeiron_wordCoordLocatorLocator_coreResidueIndex
        (table.QueryIdentity.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityTupleTuple keyRep)
        d levels arity arena rwf hwf hei hwide harena idx { key := k, _carrier := () }
      rw [hw]
      simp only [bind_tc_ok]
      cases w with
      | Terminal handle => exact ⟨some handle, rfl⟩
      | Miss _ _ => exact ⟨none, rfl⟩
    | Hyperbolic =>
      simp only [bind_tc_ok]
      obtain ⟨k, hk⟩ := hkey
      rw [hk]
      simp only [bind_tc_ok]
      cases hprof_case : self.profile with
      | none =>
        dsimp only
        exact ⟨none, rfl⟩
      | some prof =>
        dsimp only
        obtain ⟨idx, hidx⟩ := hscalar prof k
        rw [hidx]
        simp only [bind_tc_ok]
        obtain ⟨levels, arity, arena, rwf, harena⟩ := hhyp d prof hdir_case rfl hprof_case
        have hwf := hdir d hdir_case
        obtain ⟨w, hw⟩ := directory_walk_succeeds
          coord.address.Residue.Insts.Apeiron_wordCoordLocatorLocator_coreResidueIndex
          (table.QueryIdentity.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityTupleTuple keyRep)
          d levels arity arena rwf hwf hei hwide harena idx { key := k, _carrier := () }
        rw [hw]
        simp only [bind_tc_ok]
        cases w with
        | Terminal handle => exact ⟨some handle, rfl⟩
        | Miss _ _ => exact ⟨none, rfl⟩

/-- The locator retarget step is total under well-formedness, supplying the `hstep` premise for
relocation traversal induction. -/
theorem retarget_held_hstep_of_wf {K H R Faces : Type}
    (cloneInst : core.clone.Clone K) (homeInst : table.Home H K Faces)
    (keyRep : table.keyrep.KeyRep R K H) (self : table.TableLocator)
    (home : H) (sigma : cell.Sigma) (held : R)
    (old new : table.PayloadLocation)
    (hwide : 4 * UScalar.max UScalarTy.U32 + 3 ≤ UScalar.max UScalarTy.Usize)
    (hei : ExactIdentityWF (table.QueryIdentity.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityTupleTuple keyRep))
    (hdir : ∀ d, self.directory = some d → DirectoryWF d)
    (hkey : ∃ k, keyRep.key_owned sigma held = ok k)
    (hdual : ∀ k, ∃ idx, homeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceResidueCapabilityInst.parabolic_dual home k = ok idx)
    (hscalar : ∀ prof k, ∃ idx, homeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceResidueCapabilityInst.hyperbolic_scalar home prof k = ok idx)
    (hpar : ∀ d, self.directory = some d → sigma = cell.Sigma.Parabolic →
      ∃ levels arity arena, ResidueIndexWF coord.address.ParabolicDual.Insts.Apeiron_wordCoordLocatorLocator_coreResidueIndex d levels arity arena ∧ ResidueNodeWF d arena)
    (hhyp : ∀ d prof, self.directory = some d → sigma = cell.Sigma.Hyperbolic → self.profile = some prof →
      ∃ levels arity arena, ResidueIndexWF coord.address.Residue.Insts.Apeiron_wordCoordLocatorLocator_coreResidueIndex d levels arity arena ∧ ResidueNodeWF d arena) :
    ∃ nextLocator, table.TableLocator.retarget_held cloneInst homeInst keyRep self home sigma held old new = ok nextLocator := by
  have ⟨h, hh⟩ := linked_handle_succeeds cloneInst homeInst keyRep self home sigma held
    hwide hei hdir hkey hdual hscalar hpar hhyp
  exact (retarget_held_succeeds_iff cloneInst homeInst keyRep self home sigma held old new).mpr (Or.inr ⟨h, hh⟩)

end Apeiron.GrowableRelocationWF
