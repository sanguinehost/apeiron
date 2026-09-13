import TableRefinementReachablePrefixExactN

/-! # Stream-only admission at general `N`

The admission/refusal layer: the tight per-step `BuildAdmissibleN` condition, its sufficiency for
the real generated build, and the exact `ShardFull` refusal characterization. The reachable-prefix
exact coverage layer is `TableRefinementReachablePrefixExactN`. -/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-- Exact stream condition needed at each generated prefix at general `N`. -/
def BuildAdmissibleN (N : Usize) (_default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (key value : I64) (shard : Usize),
    i.val < (sliceEntries kvs).length →
    Slice.index_usize kvs i = ok (key, value) →
    table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok shard →
    key ∈ routedStreamKeys kvs i.val shard ∨
      (routedStreamKeys kvs i.val shard).card < N.val

/-- The strict-prefix condition is sufficient for exact admission at general `N`. -/
theorem buildAdmissibleN_of_distinctKeysPerShardN {N : Usize} (default : I64)
    (kvs : Slice (I64 × I64)) (hdistinct : DistinctKeysPerShardN N kvs) :
    BuildAdmissibleN N default kvs := by
  intro i key value shard hi _ _
  exact Or.inr (hdistinct i.val shard hi)

/-- Exact per-step admission makes the real generic-`N` generated build succeed and gives the
    canonical ordered read representation. -/
theorem table_build_readRepN_of_buildAdmissibleN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) (hadmissible : BuildAdmissibleN N default kvs) :
    ∃ final, wordBuildN (N := N) default kvs = ok (core.result.Result.Ok final) ∧
      ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) := by
  have hinitial : ReadRep (emptyWordTableN (N := N) default) (orderedFold default []) := by
    have hconst : orderedFold default ([] : List (I64 × I64)) = (fun _ => default) := rfl
    rw [hconst]
    exact emptyWordTableN_readRep default
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
        ReadRep final (orderedFold default (sliceEntries kvs)) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTableN N × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
        ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)) ∧
        ReachablePrefixExactN kvs p.2.val p.1)
      (post := fun r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)))
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hrep, hexact⟩ := hp
      dsimp only at hi hrep hexact ⊢
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
        obtain ⟨next, hinsert, hrepNext, hexactNext⟩ :=
          reachablePrefixExact_insertN hN default kvs i state key value hiList hindex hrep
            hexact (fun routed hrouted => hadmissible i key value routed hiList hindex hrouted)
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
        · simpa [r_post, orderedFold, htake] using hrepNext
        · simpa [r_post] using hexactNext
        · simp only [r_post]
          omega
      · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        exact ⟨state, rfl, by simpa [hieq] using hrep⟩
    · exact ⟨Nat.zero_le _, by simpa using hinitial,
        emptyWordTableN_reachablePrefixExactN default kvs⟩
  obtain ⟨result, hresult, final, hfinal, hrep⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, by simpa [orderedFold_eq_build] using hrep⟩

private def PrefixAdmissibleN (N : Usize) (kvs : Slice (I64 × I64)) (limit : Nat) : Prop :=
  ∀ (i : Usize) (key value : I64) (shard : Usize),
    i.val < limit →
    Slice.index_usize kvs i = ok (key, value) →
    table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok shard →
    key ∈ routedStreamKeys kvs i.val shard ∨
      (routedStreamKeys kvs i.val shard).card < N.val

/-- Any successful result from the real generated generic-`N` word build satisfies every tight
    per-step admission condition. -/
theorem table_build_success_implies_buildAdmissibleN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) (final : WordTableN N)
    (hsuccess : wordBuildN (N := N) default kvs = ok (core.result.Result.Ok final)) :
    BuildAdmissibleN N default kvs := by
  have hinitial : ReadRep (emptyWordTableN (N := N) default) (orderedFold default []) := by
    have hconst : orderedFold default ([] : List (I64 × I64)) = (fun _ => default) := rfl
    rw [hconst]
    exact emptyWordTableN_readRep default
  have hloopSuccess : table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs
      (emptyWordTableN (N := N) default) 0#usize = ok (core.result.Result.Ok final) := by
    unfold wordBuildN table.Table.build at hsuccess
    rw [table_new_eq_emptyN] at hsuccess
    simpa only [bind_tc_ok] using hsuccess
  unfold table.Table.build_loop at hloopSuccess
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (emptyWordTableN (N := N) default, 0#usize)
      ⦃ result => ∀ final, result = core.result.Result.Ok final →
        BuildAdmissibleN N default kvs ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTableN N × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
        ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)) ∧
        ReachablePrefixExactN kvs p.2.val p.1 ∧ PrefixAdmissibleN N kvs p.2.val)
      (post := fun result => ∀ final, result = core.result.Result.Ok final →
        BuildAdmissibleN N default kvs)
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hrep, hexact, hprefix⟩ := hp
      dsimp only at hi hrep hexact hprefix ⊢
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
        obtain ⟨result, hinsert, hresult⟩ := Aeneas.Std.WP.spec_imp_exists
          (buildInsertResult_of_readRepN hN state
            (orderedFold default ((sliceEntries kvs).take i.val)) key value hrep)
        rcases result with ⟨result, next⟩
        cases result with
        | Ok resultUnit =>
            cases resultUnit
            have hadmit := reachablePrefixExact_admissible_of_insert_successN hN default kvs i
              state next key value hexact hinsert
            obtain ⟨advanced, hadvancedInsert, hrepNext, hexactNext⟩ :=
              reachablePrefixExact_insertN hN default kvs i state key value hiList hindex hrep
                hexact hadmit
            have hadvanced : advanced = next := by
              rw [hinsert] at hadvancedInsert
              exact (congrArg Prod.snd
                (Aeneas.Std.Result.ok_injective hadvancedInsert)).symm
            subst advanced
            rw [hinsert]
            simp only [bind_tc_ok]
            have htake : (sliceEntries kvs).take (i.val + 1) =
                (sliceEntries kvs).take i.val ++ [(key, value)] := by
              rw [List.take_succ_eq_append_getElem hiList]
              congr
              exact key_post.symm
            step
            refine ⟨?_, ?_, ?_, ?_, ?_⟩
            · simpa [r_post] using Nat.succ_le_iff.mpr hiList
            · simpa [r_post, orderedFold, htake] using hrepNext
            · simpa [r_post] using hexactNext
            · intro j stored storedValue shard hj hentry hroute
              by_cases hji : j.val < i.val
              · exact hprefix j stored storedValue shard hji hentry hroute
              · have hjiEq : j.val = i.val := by omega
                have hjiUsize : j = i := UScalar.eq_of_val_eq hjiEq
                subst j
                have hpair : (stored, storedValue) = (key, value) :=
                  Aeneas.Std.Result.ok_injective (hentry.symm.trans hindex)
                have hstored : stored = key := congrArg Prod.fst hpair
                rw [hstored] at hroute ⊢
                exact hadmit shard hroute
            · simp only [r_post]
              omega
        | Err error =>
            cases error
            rw [hinsert]
            simp
      · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
        intro result _
        have hieq : i.val = (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        intro j key value shard hj hindex hroute
        exact hprefix j key value shard (by simpa [hieq] using hj) hindex hroute
    · exact ⟨Nat.zero_le _, by simpa using hinitial,
        emptyWordTableN_reachablePrefixExactN default kvs, by
          intro i key value shard hi _ _
          change i.val < 0 at hi
          omega⟩
  obtain ⟨result, hloop, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
  exact hpost final (Aeneas.Std.Result.ok_injective (hloop.symm.trans hloopSuccess))

/-- Failure of the tight per-step admission condition forces the real generated generic-`N` build
    to report `ShardFull`. -/
theorem table_build_refusesN_of_not_buildAdmissibleN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) (h : ¬ BuildAdmissibleN N default kvs) :
    wordBuildN (N := N) default kvs =
      ok (core.result.Result.Err table.StorageError.ShardFull) := by
  rcases table_build_result_readRepN hN default kvs with hrefusal | ⟨final, hsuccess, _⟩
  · exact hrefusal
  · exact (h (table_build_success_implies_buildAdmissibleN hN default kvs final hsuccess)).elim

/-- The real generated generic-`N` word build reports `ShardFull` exactly on streams outside the
    tight per-step admission condition. -/
theorem table_build_refusesN_iff_not_buildAdmissibleN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) :
    wordBuildN (N := N) default kvs =
        ok (core.result.Result.Err table.StorageError.ShardFull) ↔
      ¬ BuildAdmissibleN N default kvs :=
  ⟨fun hrefusal hadmissible => by
      obtain ⟨final, hsuccess, _⟩ :=
        table_build_readRepN_of_buildAdmissibleN hN default kvs hadmissible
      rw [hrefusal] at hsuccess
      simp at hsuccess,
    table_build_refusesN_of_not_buildAdmissibleN hN default kvs⟩

end Apeiron.TableRefinement
