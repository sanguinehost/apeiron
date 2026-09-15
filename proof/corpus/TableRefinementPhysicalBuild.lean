import TableRefinementPhysical
import TableRefinementAdmissionGeneric

/-! # Physical build postcondition at general `N`

Composes the universal constructor `emptyWordTableN_physicalInv` and the one-step
`physicalInv_insert_with_all` through the generated generic-`N` build loop, so the real generated
build returns the canonical ordered map together with `PhysicalInv` whenever `BuildAdmissibleN`
holds.  The positive-capacity theorem is `table_build_physicalInv_of_buildAdmissibleN`.  The
zero-capacity build is characterized separately in `wordBuildN_zero_*` because `BuildAdmissibleN`
is not equivalent to success at `N = 0`. -/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-- At every positive `N`, exact per-step admission makes the real generated build succeed and
    return a table that both refines the canonical ordered `build` and satisfies `PhysicalInv`. -/
theorem table_build_physicalInv_of_buildAdmissibleN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) (hadmissible : BuildAdmissibleN N default kvs) :
    ∃ final, wordBuildN (N := N) default kvs = ok (core.result.Result.Ok final) ∧
      PhysicalInv final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) := by
  have hinitial : PhysicalInv (emptyWordTableN (N := N) default) (orderedFold default []) := by
    have hconst : orderedFold default ([] : List (I64 × I64)) = (fun _ => default) := rfl
    rw [hconst]
    exact emptyWordTableN_physicalInv default
  unfold wordBuildN table.Table.build
  rw [table_new_eq_emptyN]
  simp only [bind_tc_ok]
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (emptyWordTableN (N := N) default, 0#usize)
      ⦃ r => ∃ final, r = core.result.Result.Ok final ∧
        PhysicalInv final (orderedFold default (sliceEntries kvs)) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTableN N × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
        PhysicalInv p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)) ∧
        ReachablePrefixExactN kvs p.2.val p.1)
      (post := fun r => ∃ final, r = core.result.Result.Ok final ∧
        PhysicalInv final (orderedFold default (sliceEntries kvs)))
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hphys, hexact⟩ := hp
      dsimp only at hi hphys hexact ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        step
        have hiList : i.val < (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact hlt
        have hindex : Slice.index_usize kvs i = ok (key, value) := by
          have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
          unfold Slice.index_usize
          rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]
        obtain ⟨next, hinsert, _, hexactNext⟩ :=
          reachablePrefixExact_insertN hN default kvs i state key value hiList hindex hphys.2
            hexact (fun routed hrouted => hadmissible i key value routed hiList hindex hrouted)
        obtain ⟨result, hres, hphysRes⟩ := Aeneas.Std.WP.spec_imp_exists
          (physicalInv_insert_with_all (N := N) state
            (orderedFold default ((sliceEntries kvs).take i.val)) key value hphys)
        have hpair : result = (core.result.Result.Ok (), next) :=
          Aeneas.Std.Result.ok_injective (hres.symm.trans hinsert)
        subst result
        have hphysOk : PhysicalInv next (insertWith core.num.I64.wrapping_add key value
            (orderedFold default ((sliceEntries kvs).take i.val))) := hphysRes
        rw [hinsert]
        simp only [bind_tc_ok]
        have htake : (sliceEntries kvs).take (i.val + 1) =
            (sliceEntries kvs).take i.val ++ [(key, value)] := by
          rw [List.take_succ_eq_append_getElem hiList]
          congr
          exact key_post.symm
        step
        refine ⟨?_, ?_, ?_, ?_⟩
        · simpa [r_post] using Nat.succ_le_iff.mpr hiList
        · simpa [r_post, orderedFold, htake] using hphysOk
        · simpa [r_post] using hexactNext
        · simp only [r_post]
          omega
      · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        exact ⟨state, rfl, by simpa [hieq] using hphys⟩
    · exact ⟨Nat.zero_le _, by simpa using hinitial,
        emptyWordTableN_reachablePrefixExactN default kvs⟩
  obtain ⟨result, hresult, final, hfinal, hphys⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, by simpa [orderedFold_eq_build] using hphys⟩

/-- The zero-capacity build of the empty stream returns the generated empty table. -/
theorem wordBuildN_zero_of_empty (default : I64) (kvs : Slice (I64 × I64))
    (hempty : (sliceEntries kvs).length = 0) :
    wordBuildN (N := 0#usize) default kvs =
      ok (core.result.Result.Ok (emptyWordTableN (N := 0#usize) default)) := by
  unfold wordBuildN table.Table.build
  rw [table_new_eq_emptyN]
  simp only [bind_tc_ok]
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (emptyWordTableN (N := 0#usize) default, 0#usize)
      ⦃ r => r = core.result.Result.Ok (emptyWordTableN (N := 0#usize) default) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTableN 0#usize × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.1 = emptyWordTableN (N := 0#usize) default ∧ p.2.val = 0)
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hstate, hi⟩ := hp
      dsimp only at hstate hi ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · exfalso
        have hsz : (Slice.len kvs).val = 0 := by
          rw [← sliceEntries_length]
          exact hempty
        have hlt' : i.val < (Slice.len kvs).val := hlt
        rw [hsz] at hlt'
        omega
      · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
        subst hstate
        rfl
    · exact ⟨rfl, rfl⟩
  obtain ⟨result, hresult, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hpost] at hresult
  exact hresult

/-- The zero-capacity build of any non-empty stream reports the explicit `ShardFull` refusal. -/
theorem wordBuildN_zero_refuses_of_nonempty (default : I64) (kvs : Slice (I64 × I64))
    (hne : 0 < (sliceEntries kvs).length) :
    wordBuildN (N := 0#usize) default kvs =
      ok (core.result.Result.Err table.StorageError.ShardFull) := by
  unfold wordBuildN table.Table.build
  rw [table_new_eq_emptyN]
  simp only [bind_tc_ok]
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (emptyWordTableN (N := 0#usize) default, 0#usize)
      ⦃ r => r = core.result.Result.Err table.StorageError.ShardFull ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTableN 0#usize × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val < (sliceEntries kvs).length)
    · intro p hp
      rcases p with ⟨state, i⟩
      dsimp only at hp ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        step
        rw [insert_with_zero_exact state key value]
        simp
      · exfalso
        rw [sliceEntries_length] at hp
        exact absurd hp hlt
    · exact hne
  obtain ⟨result, hresult, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hpost] at hresult
  exact hresult

/-- Exact zero-capacity build characterization: the real generated build returns `Ok` on exactly
    the empty stream and the explicit `ShardFull` refusal otherwise. -/
theorem wordBuildN_zero_ok_iff_empty (default : I64) (kvs : Slice (I64 × I64)) :
    (∃ final, wordBuildN (N := 0#usize) default kvs = ok (core.result.Result.Ok final)) ↔
      (sliceEntries kvs).length = 0 := by
  constructor
  · rintro ⟨final, hfinal⟩
    by_contra hne
    have hpos : 0 < (sliceEntries kvs).length := Nat.pos_of_ne_zero hne
    rw [wordBuildN_zero_refuses_of_nonempty default kvs hpos] at hfinal
    have hbad := Aeneas.Std.Result.ok_injective hfinal
    cases hbad
  · intro hempty
    exact ⟨_, wordBuildN_zero_of_empty default kvs hempty⟩

end Apeiron.TableRefinement
