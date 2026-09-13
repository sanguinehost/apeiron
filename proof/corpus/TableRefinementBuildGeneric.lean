import TableRefinementBuild

/-! # Generic-`N`, callback-parameterized ordered build refinement

`TableRefinementBuild` proves the ordered build refinement only at the fixed `N = 16` word
instance.  This file lifts it in two directions.

* The `…G` theorems are parameterized by arbitrary `K V H F`, `N`, the `cmp`/`clone`/`home`/
  `merge` callbacks and an explicit one-step insertion contract.  They mention no `I64`, no
  `WordHome`, no `WordMerge` and no `16`.
* The `…N` theorems instantiate the contracts for the concrete word callbacks at arbitrary `N`,
  using the generic-`N` insertion lemmas of `TableRefinementGeneric`.

The one-step contracts are hypotheses, never axioms: `BuildInsertResultG` (a success refines the
canonical write, a capacity refusal is permitted) and `BuildInsertOkG` (success only).  The
concrete word instantiation discharges them with `buildInsertResult_of_readRepN` and
`build_insert_premises_successN`.

The stream-only *refusal* direction at general `N` — that a non-`BuildAdmissibleN` stream is
refused — is the residual: it needs the generic-`N` `TableInv2`/`ReachablePrefixExact` invariant
chain of `TableRefinementBuild`/`TableRefinementAdmission` ported to `N`. -/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-! ## The empty generic-`N` constructor -/

/-- The generated all-empty state at array length `N`.  The shard table is the same at every `N`;
    only the per-shard slot arrays carry `N`. -/
def emptyWordTableN {N : Usize} (default : I64) : WordTableN N where
  default := default
  home := ()
  shards := Array.repeat 256#usize none

/-- The real generated constructor at array length `N`. -/
theorem table_new_eq_emptyN {N : Usize} (default : I64) :
    table.Table.new N table.WordHome.Insts.Apeiron_wordTableHomeI64 default () =
      ok (emptyWordTableN (N := N) default) := rfl

/-- The generated all-empty table returns its default at every word key, for every `N`. -/
theorem emptyWordTableN_read {N : Usize} (default key : I64) :
    table.Table.get core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTableN (N := N) default) key ⦃ value => value = default ⦄ := by
  set_option maxRecDepth 4096 in
  unfold table.Table.get table.Table.shard_index
  simp only [table.WordHome.Insts.Apeiron_wordTableHomeI64,
    table.WordHome.Insts.Apeiron_wordTableHomeI64.home]
  step
  simp only [table.Table.find]
  step
  · have hs : shard.val < 256 := by rw [shard_post]; omega
    have hfit : shard.val < 2 ^ UScalarTy.Usize.numBits := by scalar_tac
    change (UScalar.cast .Usize shard).val < 256
    rw [UScalar.cast_val_mod_pow_of_inBounds_eq .Usize shard hfit]
    exact hs
  · have hs : shard.val < 256 := by rw [shard_post]; omega
    have hfit : shard.val < 2 ^ UScalarTy.Usize.numBits := by scalar_tac
    have hidx : (UScalar.cast .Usize shard).val < 256 := by
      rw [UScalar.cast_val_mod_pow_of_inBounds_eq .Usize shard hfit]
      exact hs
    have hslot :
        (emptyWordTableN (N := N) default).shards.val[(UScalar.cast .Usize shard).val] = none := by
      change (List.replicate 256 none)[(UScalar.cast .Usize shard).val] = none
      rw [List.getElem_replicate]
    have ho : o = none := o_post.trans hslot
    rw [ho]
    simp [emptyWordTableN]

/-- The generated all-empty table represents the constant-default map, for every `N`. -/
theorem emptyWordTableN_readRep {N : Usize} (default : I64) :
    ReadRep (emptyWordTableN (N := N) default) (fun _ => default) := by
  intro key
  obtain ⟨value, hget, hvalue⟩ := Aeneas.Std.WP.spec_imp_exists (emptyWordTableN_read default key)
  rw [hvalue] at hget
  exact hget

/-! ## Generic-`N` one-step physical premises -/

/-- The generic-`N` analogue of `BuildInsertPremises`: a present key, a first vacant slot in an
    existing shard, or slot zero in an untouched shard. -/
def BuildInsertPremisesN {N : Usize} (t : WordTableN N) (h : HMap I64 I64)
    (key value : I64) : Prop :=
  ∃ locate, FunctionalRep locate t h ∧ Routes t locate ∧
    ((∃ shard i slots old,
        locate key = some (shard, i) ∧
        table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard ∧
        table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
          t shard key = ok (some i) ∧
        Array.index_usize t.shards shard = ok (some slots) ∧
        Array.index_usize slots i = ok (some (key, old))) ∨
      (∃ shard first slots,
        locate key = none ∧
        table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard ∧
        table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
          t shard key = ok none ∧
        Array.index_usize t.shards shard = ok (some slots) ∧
        first < N ∧ NoEmptyIn slots.val 0 first.val ∧
        NoMatchIn key slots.val 0 first.val ∧ Array.index_usize slots first = ok none) ∨
      (∃ shard,
        locate key = none ∧
        table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard ∧
        table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
          t shard key = ok none ∧
        Array.index_usize t.shards shard = ok none ∧ shard < 256#usize ∧
        Routes (insertVacant t shard (Array.repeat N none) 0#usize key
          (core.num.I64.wrapping_add value t.default))
          (insertLocator locate key shard 0#usize)))

private theorem getElem?_eq_some_of_index_usize_eq_ok {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    a.val[i.val]? = some x := by
  unfold Array.index_usize at h
  rw [Array.getElem?_Usize_eq] at h
  split at h <;> simp_all

private theorem index_usize_eq_ok_of_getElem?_eq_some {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : a.val[i.val]? = some x) :
    Array.index_usize a i = ok x := by
  unfold Array.index_usize
  rw [Array.getElem?_Usize_eq, h]

private theorem index_usize_repeat_zero {N : Usize} (hN : 0#usize < N) :
    Array.index_usize (Array.repeat N (none : Option (I64 × I64))) 0#usize = ok none := by
  apply index_usize_eq_ok_of_getElem?_eq_some
  rw [Array.repeat_val]
  have hb : (0#usize).val < (List.replicate N.val (none : Option (I64 × I64))).length := by
    simpa using hN
  rw [List.getElem?_eq_getElem hb, List.getElem_replicate]

/-- The generic-`N` bounded search on an all-vacant shard returns `none`. -/
private theorem repeatN_find_none {N : Usize} (key : I64) :
    table.Table.find_slots core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      (Array.repeat N (none : Option (I64 × I64))) key = ok none := by
  obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
    (find_slots_spec_generic (Array.repeat N (none : Option (I64 × I64))) key
      0#usize (Nat.zero_le _))
  cases found with
  | none => exact hfound
  | some i =>
      obtain ⟨hi, _, ⟨value, hmatch⟩, _⟩ := hpost
      rw [Array.repeat_val] at hmatch
      have hiLen : i.val < (List.replicate N.val (none : Option (I64 × I64))).length := by
        simpa [Array.repeat_val] using hi
      rw [List.getElem?_eq_getElem hiLen, List.getElem_replicate] at hmatch
      simp at hmatch

/-- The generic-`N` empty-shard branch preserves the generated routes under the extended
    locator.  This is `routes_insertEmptyShard` at arbitrary `N`. -/
private theorem routes_insertEmptyShardN {N : Usize} (hN : 0#usize < N)
    (locate : I64 → Option (Usize × Usize)) (t : WordTableN N) (key value : I64)
    (shard : Usize) (hroutes : Routes t locate)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (houter : Array.index_usize t.shards shard = ok none) (hshardBound : shard < 256#usize) :
    Routes (insertVacant t shard (Array.repeat N none) 0#usize key value)
      (insertLocator locate key shard 0#usize) := by
  intro x
  let next := insertVacant t shard (Array.repeat N none) 0#usize key value
  have hnewOuter : Array.index_usize next.shards shard =
      ok (some ((Array.repeat N none).set 0#usize (some (key, value)))) := by
    have hget : next.shards.val[shard.val]? =
        some (some ((Array.repeat N none).set 0#usize (some (key, value)))) := by
      dsimp [next, insertVacant]
      rw [Array.set_val_eq]
      have hsLen : shard.val < t.shards.val.length := by
        rw [t.shards.property]
        exact hshardBound
      exact List.getElem?_set_eq_of_lt _ hsLen
    exact index_usize_eq_ok_of_getElem?_eq_some next.shards shard _ hget
  have hrepeatSlot : Array.index_usize (Array.repeat N
      (none : Option (I64 × I64))) 0#usize = ok none := index_usize_repeat_zero hN
  by_cases hx : x = key
  · subst x
    simp only [insertLocator]
    apply generatedRoute_some_of_find next key shard 0#usize
    · rw [← word_shard_index_state_independent t next key]
      exact hshard
    · unfold table.Table.find
      rw [hnewOuter]
      simp only [bind_tc_ok]
      exact find_slots_insertVacant_self (Array.repeat N none) 0#usize key value
        hN (fun j hj hjlt _ => (Nat.not_lt_of_ge hj) hjlt)
  · simp only [insertLocator, if_neg hx]
    have hroute := hroutes x
    cases hloc : locate x with
    | none =>
        rw [hloc] at hroute
        obtain ⟨s, hs, hf⟩ := hroute
        apply generatedRoute_none_of_find next x s
        · rw [← word_shard_index_state_independent t next x]
          exact hs
        · by_cases hsame : s = shard
          · subst s
            unfold table.Table.find
            rw [hnewOuter]
            simp only [bind_tc_ok]
            rw [find_slots_insertVacant_ne (Array.repeat N none) 0#usize key value x
              hrepeatSlot hx, repeatN_find_none]
          · have hne : shard.val ≠ s.val := fun h => hsame (UScalar.eq_of_val_eq h.symm)
            simpa [next, insertVacant, table.Table.find, Array.index_usize, Array.set_val_eq,
              hne] using hf
    | some p =>
        rcases p with ⟨s, i⟩
        rw [hloc] at hroute
        obtain ⟨oldSlots, oldValue, hs, hf, ho, hi⟩ := hroute
        have hsame : s ≠ shard := by
          intro heq
          subst s
          rw [houter] at ho
          simp at ho
        apply generatedRoute_some_of_find next x s i
        · rw [← word_shard_index_state_independent t next x]
          exact hs
        · have hne : shard.val ≠ s.val := fun h => hsame (UScalar.eq_of_val_eq h.symm)
          simpa [next, insertVacant, table.Table.find, Array.index_usize, Array.set_val_eq,
            hne] using hf

/-- The generic-`N` one-step physical premises discharge the real generated insertion and
    preserve `ReadRep`. -/
theorem build_insert_premises_successN {N : Usize} (hN : 0#usize < N)
    (t : WordTableN N) (h : HMap I64 I64) (key value : I64)
    (hp : BuildInsertPremisesN t h key value) :
    ∃ t', table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') ∧
      ReadRep t' (insertWith core.num.I64.wrapping_add key value h) := by
  obtain ⟨locate, hfunctional, hroutes, hpresent | hvacant | hempty⟩ := hp
  · obtain ⟨shard, i, slots, old, hlocate, hshard, hfind, houter, hslot⟩ := hpresent
    let t' := replacePresent t shard slots i key
      (core.num.I64.wrapping_add value old)
    have hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') :=
      insert_with_present_exact t key value shard i slots key old hshard hfind houter hslot
    exact ⟨t', hinsert, insert_with_present_readRep_preservation locate t t' h key value
      shard i slots old hfunctional hroutes hlocate hshard hfind houter hslot hinsert⟩
  · obtain ⟨shard, first, slots, hlocate, hshard, hfind, houter, hfirst, hempty,
      hnomatch, hslot⟩ := hvacant
    let t' := insertVacant t shard slots first key
      (core.num.I64.wrapping_add value t.default)
    have hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') :=
      insert_with_vacant_exact t key value shard first slots hshard hfind houter hfirst
        hempty hnomatch hslot
    exact ⟨t', hinsert, insert_with_vacant_readRep_preservation_routed locate t t' h key
      value shard first slots hfunctional hroutes hlocate hshard hfind houter hfirst
      hempty hnomatch hslot hinsert⟩
  · obtain ⟨shard, hlocate, hshard, hfind, houter, hshardBound, hroutes'⟩ := hempty
    let t' := insertVacant t shard (Array.repeat N none) 0#usize key
      (core.num.I64.wrapping_add value t.default)
    have hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') :=
      insert_with_empty_shard_exact t key value shard hshard hfind houter hN hshardBound
    refine ⟨t', hinsert, (functionalRep_iff_readRep_of_routes _ t' _ hroutes').mp ?_⟩
    unfold FunctionalRep at hfunctional ⊢
    rw [stateRead_insertEmptyShard locate t key value shard hlocate houter hN,
      hfunctional]

/-- The generic-`N` absent key in a physically full shard returns `ShardFull` unchanged. -/
theorem insert_with_full_exactN {N : Usize} (hN : 0#usize < N) (t : WordTableN N)
    (key value : I64) (shard : Usize)
    (slots : Array (Option (I64 × I64)) N)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfull : NoEmptyIn slots.val 0 N.val) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Err table.StorageError.ShardFull, t) := by
  have hm : Array.index_mut_usize t.shards shard = ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  obtain ⟨r, hr, hp⟩ := Aeneas.Std.WP.spec_imp_exists
    (insert_with_loop0_full_spec_generic (t.shards.set shard) slots t.default key value
      0#usize (Nat.zero_le _) hfull)
  rw [hp] at hr
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp only [bind_tc_ok]
  rw [if_neg (ne_of_gt hN), houter]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_neg (by simp), hm]
  simp only [bind_tc_ok]
  change (do
    let (result, shards) ← table.Table.insert_with_loop0 core.clone.CloneI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 (t.shards.set shard)
      t.default () key value slots 0#usize
    ok (result, { t with shards := shards })) = _
  rw [hr]
  have houter' := getElem?_eq_some_of_index_usize_eq_ok t.shards shard (some slots) houter
  have hsLen : shard.val < t.shards.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
    simp at houter'
  have hget : t.shards.val[shard.val] = some slots := by
    rw [List.getElem?_eq_getElem hsLen] at houter'
    exact Option.some.inj houter'
  have hset : t.shards.set shard (some slots) = t.shards := by
    apply Aeneas.Std.Array.ext
    simp only [Array.set_val_eq]
    apply List.ext_getElem?
    intro n
    rw [List.getElem?_set_of_lt' (some slots) t.shards.val hsLen]
    split
    · rename_i hn
      subst n
      exact houter'.symm
    · rfl
  simp [hset]

/-- The generic-`N` read-boundary behavior of one insertion: success refines the canonical write,
    capacity refusal is permitted. -/
def BuildInsertResultN {N : Usize} (t : WordTableN N) (h : HMap I64 I64)
    (key value : I64) : Prop :=
  table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
    table.WordHome.Insts.Apeiron_wordTableHomeI64
    table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
    ⦃ result => match result.1 with
      | core.result.Result.Ok _ =>
          ReadRep result.2 (insertWith core.num.I64.wrapping_add key value h)
      | core.result.Result.Err table.StorageError.ShardFull => True ⦄

/-- Every generic-`N` generated insertion either preserves the canonical read representation or
    reports `ShardFull`; no physical invariant is needed to expose this disjunction. -/
theorem buildInsertResult_of_readRepN {N : Usize} (hN : 0#usize < N)
    (t : WordTableN N) (h : HMap I64 I64) (key value : I64)
    (hrep : ReadRep t h) : BuildInsertResultN t h key value := by
  let locate := generatedLocator t
  have hroutes : Routes t locate := generatedLocator_routes t
  have hfunctional : FunctionalRep locate t h :=
    (generatedFunctionalRep_iff_readRep t h).mpr hrep
  have hroute := hroutes key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, i⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      obtain ⟨result, hinsert, hrep'⟩ := build_insert_premises_successN hN t h key value
        ⟨locate, hfunctional, hroutes, Or.inl
          ⟨shard, i, slots, old, hlocate, hshard, hfind, houter, hslot⟩⟩
      unfold BuildInsertResultN
      rw [hinsert]
      simpa using hrep'
  | none =>
      rw [hlocate] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      obtain ⟨routedShard, hroutedShard, hshardBound⟩ := word_shard_index_spec t key
      have hshardEq : routedShard = shard :=
        Aeneas.Std.Result.ok_injective (hroutedShard.symm.trans hshard)
      subst routedShard
      have hshardLen : shard.val < t.shards.val.length := by
        rw [t.shards.property]
        exact hshardBound
      obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
        (Array.index_usize_spec t.shards shard hshardLen)
      cases outer with
      | none =>
          let t' := insertVacant t shard (Array.repeat N none) 0#usize key
            (core.num.I64.wrapping_add value t.default)
          have hroutes' : Routes t' (insertLocator locate key shard 0#usize) :=
            routes_insertEmptyShardN hN locate t key
              (core.num.I64.wrapping_add value t.default) shard hroutes hshard houter
              hshardBound
          obtain ⟨result, hinsert, hrep'⟩ := build_insert_premises_successN hN t h key value
            ⟨locate, hfunctional, hroutes, Or.inr (Or.inr
              ⟨shard, hlocate, hshard, hfind, houter, hshardBound, hroutes'⟩)⟩
          unfold BuildInsertResultN
          rw [hinsert]
          simpa using hrep'
      | some slots =>
          have hfindSlots : table.Table.find_slots core.cmp.EqI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64 slots key = ok none := by
            unfold table.Table.find at hfind
            rw [houter] at hfind
            simpa only [bind_tc_ok] using hfind
          obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
            (find_slots_spec_generic slots key 0#usize (Nat.zero_le _))
          unfold table.Table.find_slots at hfindSlots
          rw [hfindSlots] at hfound
          have hfoundNone : found = none := Aeneas.Std.Result.ok_injective hfound.symm
          subst found
          have hnomatch : NoMatchIn key slots.val 0 N.val := by
            simpa [FindPostN] using hpost
          by_cases hvacant : ∃ j, j < N.val ∧ slots.val[j]? = some none
          · let k := Nat.find hvacant
            have hk : k < N.val := (Nat.find_spec hvacant).1
            have hslotGet : slots.val[k]? = some none := (Nat.find_spec hvacant).2
            have hnoempty : NoEmptyIn slots.val 0 k := by
              intro j _ hj hempty
              have hmin := Nat.find_min' hvacant
                ⟨hj.trans hk, by simpa [EmptyAt] using hempty⟩
              exact (Nat.not_lt_of_ge hmin hj).elim
            let first : Usize := ⟨k, by scalar_tac⟩
            have hfirst : first < N := hk
            have hslot : Array.index_usize slots first = ok none :=
              index_usize_eq_ok_of_getElem?_eq_some slots first none hslotGet
            obtain ⟨result, hinsert, hrep'⟩ := build_insert_premises_successN hN t h key value
              ⟨locate, hfunctional, hroutes, Or.inr (Or.inl
                ⟨shard, first, slots, hlocate, hshard, hfind, houter, hfirst,
                  hnoempty, fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hk), hslot⟩)⟩
            unfold BuildInsertResultN
            rw [hinsert]
            simpa using hrep'
          · have hfull : NoEmptyIn slots.val 0 N.val := by
              intro j _ hj hempty
              exact hvacant ⟨j, hj, by simpa [EmptyAt] using hempty⟩
            unfold BuildInsertResultN
            rw [insert_with_full_exactN hN t key value shard slots hshard hfind houter hfull]
            simp

/-- Every possible generic-`N` prefix state has the capacity-aware insertion result required by
    the generated build loop. -/
def AllBuildInsertResultsN {N : Usize} (default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (state : WordTableN N) (key value : I64),
    i.val < (sliceEntries kvs).length →
    Slice.index_usize kvs i = ok (key, value) →
    ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)) →
    BuildInsertResultN state
      (orderedFold default ((sliceEntries kvs).take i.val)) key value

/-- At every `N ≥ 1` the capacity-aware insertion result holds unconditionally. -/
theorem allBuildInsertResultsN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) :
    AllBuildInsertResultsN (N := N) default kvs := by
  intro i state key value _ _ hrep
  exact buildInsertResult_of_readRepN hN state
    (orderedFold default ((sliceEntries kvs).take i.val)) key value hrep

/-- The real generated generic-`N` loop either stops at `ShardFull` or returns the full ordered
    fold. -/
theorem table_build_loop_result_readRepN {N : Usize} (default : I64)
    (kvs : Slice (I64 × I64)) (initial : WordTableN N) (start : Usize)
    (hstart : start.val ≤ (sliceEntries kvs).length)
    (hinitial : ReadRep initial (orderedFold default ((sliceEntries kvs).take start.val)))
    (hresults : AllBuildInsertResultsN (N := N) default kvs) :
    table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs
      initial start
      ⦃ result => result = core.result.Result.Err table.StorageError.ShardFull ∨
        ∃ final, result = core.result.Result.Ok final ∧
          ReadRep final (orderedFold default (sliceEntries kvs)) ⦄ := by
  unfold table.Table.build_loop
  apply loop.spec_decr_nat
    (measure := fun p : WordTableN N × Usize => (sliceEntries kvs).length - p.2.val)
    (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
      ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)))
    (post := fun result => result = core.result.Result.Err table.StorageError.ShardFull ∨
      ∃ final, result = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)))
  · intro p hp
    rcases p with ⟨state, i⟩
    obtain ⟨hi, hrep⟩ := hp
    dsimp only at hi hrep ⊢
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
      have hinsert := hresults i state key value hiList hindex hrep
      unfold BuildInsertResultN at hinsert
      step
      cases r with
      | Ok resultUnit =>
          have hrep' := r_post
          simp only at hrep'
          have htake : (sliceEntries kvs).take (i.val + 1) =
              (sliceEntries kvs).take i.val ++ [(key, value)] := by
            rw [List.take_succ_eq_append_getElem hiList]
            congr
            exact key_post.symm
          step
          refine ⟨?_, ?_, ?_⟩
          · simpa [r_post] using Nat.succ_le_iff.mpr hiList
          · simpa [r_post, orderedFold, htake] using hrep'
          · simp only [r_post]
            omega
      | Err error =>
          cases error
          simp
    · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
      right
      refine ⟨state, rfl, ?_⟩
      have hieq : i.val = (sliceEntries kvs).length := by
        rw [sliceEntries_length]
        exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
      simpa [hieq] using hrep
  · exact ⟨hstart, hinitial⟩

/-- The generated generic-`N` word build, at the concrete word callbacks. -/
def wordBuildN {N : Usize} (default : I64) (kvs : Slice (I64 × I64)) :
    Result (core.result.Result (WordTableN N) table.StorageError) :=
  table.Table.build N core.cmp.EqI64 core.clone.CloneI64
    table.WordHome.Insts.Apeiron_wordTableHomeI64
    table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 ()
    default kvs ()

/-- The real generated generic-`N` word build is unconditional: it either reports physical capacity
    refusal or returns a table whose reads are the full canonical ordered fold. -/
theorem table_build_result_readRepN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) :
    wordBuildN (N := N) default kvs =
        ok (core.result.Result.Err table.StorageError.ShardFull) ∨
      ∃ final, wordBuildN (N := N) default kvs = ok (core.result.Result.Ok final) ∧
        ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
          (sliceEntries kvs)) := by
  have hinitial : ReadRep (emptyWordTableN (N := N) default) (orderedFold default []) := by
    have hconst : orderedFold default ([] : List (I64 × I64)) = (fun _ => default) := rfl
    rw [hconst]
    exact emptyWordTableN_readRep default
  have hs := table_build_loop_result_readRepN default kvs (emptyWordTableN (N := N) default)
    0#usize (Nat.zero_le _) (by simpa using hinitial) (allBuildInsertResultsN hN default kvs)
  obtain ⟨result, hloop, hresult⟩ := Aeneas.Std.WP.spec_imp_exists hs
  unfold wordBuildN table.Table.build
  rw [table_new_eq_emptyN]
  simp only [bind_tc_ok]
  rcases hresult with hfull | ⟨final, hok, hrep⟩
  · left
    rw [hloop, hfull]
  · right
    refine ⟨final, ?_, ?_⟩
    · rw [hloop, hok]
    · simpa [orderedFold_eq_build] using hrep

/-- The generic-`N` analogue of `AllBuildInsertsAdmitted`: every canonical prefix representative
    supplies the physical facts for its next generated insertion. -/
def AllBuildInsertsAdmittedN {N : Usize} (default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (state : WordTableN N) (key value : I64),
    i.val < (sliceEntries kvs).length →
    Slice.index_usize kvs i = ok (key, value) →
    ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)) →
    BuildInsertPremisesN state
      (orderedFold default ((sliceEntries kvs).take i.val)) key value

/-- The real generated generic-`N` loop consumes the slice in order and returns the canonical
    prefix fold once every prefix insertion is admitted. -/
theorem table_build_loop_readRepN {N : Usize} (hN : 0#usize < N) (default : I64)
    (kvs : Slice (I64 × I64)) (initial : WordTableN N) (start : Usize)
    (hstart : start.val ≤ (sliceEntries kvs).length)
    (hinitial : ReadRep initial (orderedFold default ((sliceEntries kvs).take start.val)))
    (hadmitted : AllBuildInsertsAdmittedN (N := N) default kvs) :
    ∃ final,
      table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs
        initial start = ok (core.result.Result.Ok final) ∧
      ReadRep final (orderedFold default (sliceEntries kvs)) := by
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (initial, start)
      ⦃ r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTableN N × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
        ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)))
      (post := fun r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)))
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hrep⟩ := hp
      dsimp only at hi hrep ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        step
        have hiList : i.val < (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact hlt
        obtain ⟨state', hinsert, hrep'⟩ := build_insert_premises_successN hN state
          (orderedFold default ((sliceEntries kvs).take i.val)) key value
          (hadmitted i state key value hiList (by
            have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
            unfold Slice.index_usize
            rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]) hrep)
        rw [hinsert]
        simp only [bind_tc_ok]
        have htake : (sliceEntries kvs).take (i.val + 1) =
            (sliceEntries kvs).take i.val ++ [(key, value)] := by
          rw [List.take_succ_eq_append_getElem hiList]
          congr
          exact key_post.symm
        step
        refine ⟨?_, ?_, ?_⟩
        · simpa [r_post] using Nat.succ_le_iff.mpr hiList
        · simpa [r_post, orderedFold, htake] using hrep'
        · simp only [r_post]
          omega
      · rw [if_neg hlt]
        simp only [Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        refine ⟨state, rfl, ?_⟩
        simpa [hieq] using hrep
    · exact ⟨hstart, hinitial⟩
  obtain ⟨result, hresult, final, hfinal, hrep⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, hrep⟩

/-- The generated generic-`N` word build from its real empty constructor refines canonical ordered
    `build` once every prefix insertion is admitted. -/
theorem table_build_readRepN {N : Usize} (hN : 0#usize < N) (default : I64)
    (kvs : Slice (I64 × I64))
    (hadmitted : AllBuildInsertsAdmittedN (N := N) default kvs) :
    ∃ final, wordBuildN (N := N) default kvs = ok (core.result.Result.Ok final) ∧
      ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) := by
  have hinitial : ReadRep (emptyWordTableN (N := N) default) (orderedFold default []) := by
    have hconst : orderedFold default ([] : List (I64 × I64)) = (fun _ => default) := rfl
    rw [hconst]
    exact emptyWordTableN_readRep default
  obtain ⟨final, hloop, hrep⟩ := table_build_loop_readRepN hN default kvs
    (emptyWordTableN (N := N) default) 0#usize (Nat.zero_le _) (by simpa using hinitial)
    hadmitted
  refine ⟨final, ?_, ?_⟩
  · unfold wordBuildN table.Table.build
    rw [table_new_eq_emptyN]
    simpa using hloop
  · simpa [orderedFold_eq_build] using hrep

/-! ## Abstract-callback loop refinement

These theorems are stated for arbitrary `K V H F`, `N`, `cmp`/`clone`/`home`/`merge` callbacks and
take the one-step insertion contract as an explicit hypothesis.  The only callback contracts the
loop itself needs are the two clone identities; equality and merge behavior is absorbed by the
one-step contract. -/

/-- The ordinary list carried by a generated borrowed slice, at arbitrary key and value types. -/
def sliceEntriesG {K V : Type} (kvs : Slice (K × V)) : List (K × V) := kvs.val

/-- The generic slice and its ordinary list have the same length. -/
theorem sliceEntriesG_length {K V : Type} (kvs : Slice (K × V)) :
    (sliceEntriesG kvs).length = (Slice.len kvs).val := by
  simp [sliceEntriesG, Slice.len]

/-- The canonical ordered fold for an arbitrary merge function. -/
def orderedFoldG {K V : Type} [DecidableEq K] (mergeFn : V → V → V) (default : V)
    (kvs : List (K × V)) : HMap K V :=
  kvs.foldl (fun state kv => insertWith mergeFn kv.1 kv.2 state) (fun _ => default)

/-- Executing read representation for arbitrary callbacks. -/
def ReadRepG {K V H : Type} {N : Usize} (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (t : table.Table K V H N) (h : HMap K V) : Prop :=
  ∀ key, table.Table.get cmp HomeInst t key = ok (h key)

/-- One generated insertion either refines the canonical write or reports `ShardFull`. -/
def BuildInsertResultG {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (merge : F) (mergeFn : V → V → V)
    (t : table.Table K V H N) (h : HMap K V) (key : K) (value : V) : Prop :=
  table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value
    ⦃ result => match result.1 with
      | core.result.Result.Ok _ =>
          ReadRepG cmp HomeInst result.2 (insertWith mergeFn key value h)
      | core.result.Result.Err table.StorageError.ShardFull => True ⦄

/-- One generated insertion succeeds and refines the canonical write. -/
def BuildInsertOkG {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (merge : F) (mergeFn : V → V → V)
    (t : table.Table K V H N) (h : HMap K V) (key : K) (value : V) : Prop :=
  ∃ t', table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value =
      ok (core.result.Result.Ok (), t') ∧
    ReadRepG cmp HomeInst t' (insertWith mergeFn key value h)

/-- The generated loop with arbitrary callbacks either stops at `ShardFull` or returns the full
    canonical ordered fold, given the one-step disjunction contract and clone identities. -/
theorem table_build_loop_result_readRepG {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (cloneK : core.clone.Clone K) (merge : F)
    (mergeFn : V → V → V) (default : V) (kvs : Slice (K × V))
    (initial : table.Table K V H N) (start : Usize)
    (hcloneK : ∀ x, cloneK.clone x = ok x) (hcloneV : ∀ x, cloneV.clone x = ok x)
    (hstart : start.val ≤ (sliceEntriesG kvs).length)
    (hinitial : ReadRepG cmp HomeInst initial
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take start.val)))
    (hresults : ∀ (i : Usize) (state : table.Table K V H N) (key : K) (value : V),
      i.val < (sliceEntriesG kvs).length →
      Slice.index_usize kvs i = ok (key, value) →
      ReadRepG cmp HomeInst state
        (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) →
      BuildInsertResultG cmp cloneV HomeInst MergeInst merge mergeFn state
        (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) key value) :
    table.Table.build_loop cmp cloneV HomeInst MergeInst cloneK merge kvs initial start
      ⦃ result => result = core.result.Result.Err table.StorageError.ShardFull ∨
        ∃ final, result = core.result.Result.Ok final ∧
          ReadRepG cmp HomeInst final
            (orderedFoldG mergeFn default (sliceEntriesG kvs)) ⦄ := by
  unfold table.Table.build_loop
  apply loop.spec_decr_nat
    (measure := fun p : table.Table K V H N × Usize => (sliceEntriesG kvs).length - p.2.val)
    (inv := fun p => p.2.val ≤ (sliceEntriesG kvs).length ∧
      ReadRepG cmp HomeInst p.1 (orderedFoldG mergeFn default ((sliceEntriesG kvs).take p.2.val)))
    (post := fun result => result = core.result.Result.Err table.StorageError.ShardFull ∨
      ∃ final, result = core.result.Result.Ok final ∧
        ReadRepG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)))
  · intro p hp
    rcases p with ⟨state, i⟩
    obtain ⟨hi, hrep⟩ := hp
    dsimp only at hi hrep ⊢
    unfold table.Table.build_loop.body
    by_cases hlt : i < Slice.len kvs
    · rw [if_pos hlt]
      simp only [hcloneK, hcloneV]
      step
      have hiList : i.val < (sliceEntriesG kvs).length := by
        rw [sliceEntriesG_length]
        exact hlt
      have hindex : Slice.index_usize kvs i = ok (key, value) := by
        have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
        unfold Slice.index_usize
        rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]
      have hinsert := hresults i state key value hiList hindex hrep
      unfold BuildInsertResultG at hinsert
      step
      cases r with
      | Ok resultUnit =>
          have hrep' := r_post
          simp only at hrep'
          have htake : (sliceEntriesG kvs).take (i.val + 1) =
              (sliceEntriesG kvs).take i.val ++ [(key, value)] := by
            rw [List.take_succ_eq_append_getElem hiList]
            congr
            exact key_post.symm
          step
          refine ⟨?_, ?_, ?_⟩
          · simpa [r_post] using Nat.succ_le_iff.mpr hiList
          · simpa [r_post, orderedFoldG, htake] using hrep'
          · simp only [r_post]
            omega
      | Err error =>
          cases error
          simp
    · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
      right
      refine ⟨state, rfl, ?_⟩
      have hieq : i.val = (sliceEntriesG kvs).length := by
        rw [sliceEntriesG_length]
        exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
      simpa [hieq] using hrep
  · exact ⟨hstart, hinitial⟩

/-- The generated loop with arbitrary callbacks returns the full canonical ordered fold, given the
    one-step success contract and clone identities. -/
theorem table_build_loop_readRepG {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (cloneK : core.clone.Clone K) (merge : F)
    (mergeFn : V → V → V) (default : V) (kvs : Slice (K × V))
    (initial : table.Table K V H N) (start : Usize)
    (hcloneK : ∀ x, cloneK.clone x = ok x) (hcloneV : ∀ x, cloneV.clone x = ok x)
    (hstart : start.val ≤ (sliceEntriesG kvs).length)
    (hinitial : ReadRepG cmp HomeInst initial
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take start.val)))
    (hok : ∀ (i : Usize) (state : table.Table K V H N) (key : K) (value : V),
      i.val < (sliceEntriesG kvs).length →
      Slice.index_usize kvs i = ok (key, value) →
      ReadRepG cmp HomeInst state
        (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) →
      BuildInsertOkG cmp cloneV HomeInst MergeInst merge mergeFn state
        (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) key value) :
    ∃ final,
      table.Table.build_loop cmp cloneV HomeInst MergeInst cloneK merge kvs initial start =
        ok (core.result.Result.Ok final) ∧
      ReadRepG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)) := by
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body cmp cloneV HomeInst MergeInst cloneK merge
        kvs state i)
      (initial, start)
      ⦃ r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRepG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : table.Table K V H N × Usize => (sliceEntriesG kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntriesG kvs).length ∧
        ReadRepG cmp HomeInst p.1
          (orderedFoldG mergeFn default ((sliceEntriesG kvs).take p.2.val)))
      (post := fun r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRepG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)))
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hrep⟩ := hp
      dsimp only at hi hrep ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        simp only [hcloneK, hcloneV]
        step
        have hiList : i.val < (sliceEntriesG kvs).length := by
          rw [sliceEntriesG_length]
          exact hlt
        obtain ⟨state', hinsert, hrep'⟩ := hok i state key value hiList (by
          have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
          unfold Slice.index_usize
          rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]) hrep
        rw [hinsert]
        simp only [bind_tc_ok]
        have htake : (sliceEntriesG kvs).take (i.val + 1) =
            (sliceEntriesG kvs).take i.val ++ [(key, value)] := by
          rw [List.take_succ_eq_append_getElem hiList]
          congr
          exact key_post.symm
        step
        refine ⟨?_, ?_, ?_⟩
        · simpa [r_post] using Nat.succ_le_iff.mpr hiList
        · simpa [r_post, orderedFoldG, htake] using hrep'
        · simp only [r_post]
          omega
      · rw [if_neg hlt]
        simp only [Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntriesG kvs).length := by
          rw [sliceEntriesG_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        refine ⟨state, rfl, ?_⟩
        simpa [hieq] using hrep
    · exact ⟨hstart, hinitial⟩
  obtain ⟨result, hresult, final, hfinal, hrep⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, hrep⟩

/-! ## Concrete callback discharge and the minimal loop-step contract -/

/-- The generated word clone callback is the identity computation. -/
theorem cloneI64_identity (x : I64) : core.clone.CloneI64.clone x = ok x := by
  rfl

/-- The abstract capacity-aware insertion contract is exactly the existing concrete word contract. -/
theorem buildInsertResultG_word_iff {N : Usize} (t : WordTableN N) (h : HMap I64 I64)
    (key value : I64) :
    BuildInsertResultG core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 () core.num.I64.wrapping_add
        t h key value ↔ BuildInsertResultN t h key value := by
  rfl

/-- The real `WordHome`/`WordMerge`/`CloneI64` callbacks satisfy the abstract one-step result
    contract at every positive `N`. -/
theorem buildInsertResultG_word {N : Usize} (hN : 0#usize < N) (t : WordTableN N)
    (h : HMap I64 I64) (key value : I64) (hrep : ReadRep t h) :
    BuildInsertResultG core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 () core.num.I64.wrapping_add
      t h key value := by
  rw [buildInsertResultG_word_iff]
  exact buildInsertResult_of_readRepN hN t h key value hrep

/-- Under the existing concrete physical insertion premises, the real callbacks satisfy the
    abstract success-only one-step contract. -/
theorem buildInsertOkG_word {N : Usize} (hN : 0#usize < N) (t : WordTableN N)
    (h : HMap I64 I64) (key value : I64) (hp : BuildInsertPremisesN t h key value) :
    BuildInsertOkG core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 () core.num.I64.wrapping_add
      t h key value := by
  simpa [BuildInsertOkG, ReadRepG, ReadRep] using
    build_insert_premises_successN hN t h key value hp

/-- Explicit instantiation of `table_build_loop_result_readRepG` by the real word callbacks.
    It is unconditional apart from `N > 0` and the initial read representation: physical capacity
    refusal is one of its two conclusions. -/
theorem table_build_loop_result_readRepG_word {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) (initial : WordTableN N) (start : Usize)
    (hstart : start.val ≤ (sliceEntriesG kvs).length)
    (hinitial : ReadRepG core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 initial
      (orderedFoldG core.num.I64.wrapping_add default ((sliceEntriesG kvs).take start.val))) :
    table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs initial start
      ⦃ result => result = core.result.Result.Err table.StorageError.ShardFull ∨
        ∃ final, result = core.result.Result.Ok final ∧
          ReadRepG core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64 final
            (orderedFoldG core.num.I64.wrapping_add default (sliceEntriesG kvs)) ⦄ := by
  simpa [ReadRepG, ReadRep, orderedFoldG, orderedFold, sliceEntriesG, sliceEntries] using
    table_build_loop_result_readRepN default kvs initial start hstart hinitial
      (allBuildInsertResultsN hN default kvs)

/-- Explicit success-only instantiation of `table_build_loop_readRepG`; the concrete physical
    premises are necessary here because a full shard is a real refusal branch. -/
theorem table_build_loop_readRepG_word {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64)) (initial : WordTableN N) (start : Usize)
    (hstart : start.val ≤ (sliceEntriesG kvs).length)
    (hinitial : ReadRepG core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 initial
      (orderedFoldG core.num.I64.wrapping_add default ((sliceEntriesG kvs).take start.val)))
    (hadmitted : AllBuildInsertsAdmittedN (N := N) default kvs) :
    ∃ final, table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs initial start =
        ok (core.result.Result.Ok final) ∧
      ReadRepG core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64 final
        (orderedFoldG core.num.I64.wrapping_add default (sliceEntriesG kvs)) := by
  simpa [ReadRepG, ReadRep, orderedFoldG, orderedFold, sliceEntriesG, sliceEntries] using
    table_build_loop_readRepN hN default kvs initial start hstart hinitial hadmitted

/-- Minimal semantic contract at the loop boundary: every represented state has a capacity-aware
    one-step result.  Equality, routing, cloning, and merging need no stronger algebraic laws. -/
structure WordBuildOneStepContractN (N : Usize) : Prop where
  result : ∀ (t : WordTableN N) (h : HMap I64 I64) (key value : I64),
    ReadRep t h →
    BuildInsertResultG core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 () core.num.I64.wrapping_add
      t h key value

/-- The real callbacks satisfy the minimal contract at every positive `N`. -/
theorem wordBuildOneStepContractN {N : Usize} (hN : 0#usize < N) :
    WordBuildOneStepContractN N :=
  ⟨fun t h key value hrep => buildInsertResultG_word hN t h key value hrep⟩

 /-- The named abstract contract is neither stronger nor weaker than the existing concrete
    capacity-aware one-step result. -/
theorem wordBuildOneStepContractN_iff_concrete {N : Usize} :
    WordBuildOneStepContractN N ↔
      ∀ (t : WordTableN N) (h : HMap I64 I64) (key value : I64),
        ReadRep t h → BuildInsertResultN t h key value := by
  constructor
  · intro contract t h key value hrep
    rw [← buildInsertResultG_word_iff]
    exact contract.result t h key value hrep
  · intro concrete
    refine ⟨?_⟩
    intro t h key value hrep
    rw [buildInsertResultG_word_iff]
    exact concrete t h key value hrep

/-- Sufficiency of the named minimal contract for the abstract loop theorem. -/
theorem wordBuildOneStepContractN_sufficient {N : Usize}
    (contract : WordBuildOneStepContractN N) (default : I64) (kvs : Slice (I64 × I64))
    (initial : WordTableN N) (start : Usize)
    (hstart : start.val ≤ (sliceEntriesG kvs).length)
    (hinitial : ReadRepG core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 initial
      (orderedFoldG core.num.I64.wrapping_add default ((sliceEntriesG kvs).take start.val))) :
    table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs initial start
      ⦃ result => result = core.result.Result.Err table.StorageError.ShardFull ∨
        ∃ final, result = core.result.Result.Ok final ∧
          ReadRepG core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64 final
            (orderedFoldG core.num.I64.wrapping_add default (sliceEntriesG kvs)) ⦄ := by
  apply table_build_loop_result_readRepG
    (cmp := core.cmp.EqI64) (cloneV := core.clone.CloneI64)
    (HomeInst := table.WordHome.Insts.Apeiron_wordTableHomeI64)
    (MergeInst := table.WordMerge.Insts.Apeiron_wordTableMergeI64)
    (cloneK := core.clone.CloneI64) (merge := ())
    (mergeFn := core.num.I64.wrapping_add) (default := default) (kvs := kvs)
    (initial := initial) (start := start) cloneI64_identity cloneI64_identity hstart hinitial
  intro i state key value _ _ hrep
  exact contract.result state _ key value hrep

/-! ## Necessity of the abstract one-step contract -/

/-- `WP.spec` distributes over the actual Aeneas interaction-tree bind in both directions. -/
theorem wpSpec_bind_iff {α β : Type} (m : Result α) (k : α → Result β)
    (Q : Aeneas.Std.WP.Post β) :
    Aeneas.Std.WP.spec (m >>= k) Q ↔
      Aeneas.Std.WP.spec m (fun x => Aeneas.Std.WP.spec (k x) Q) := by
  constructor
  · intro h
    obtain ⟨y, hbind, hQ⟩ := Aeneas.Std.WP.spec_imp_exists h
    cases m with
    | ret x =>
        rw [Aeneas.Std.WP.spec_ok]
        apply (Aeneas.Std.WP.spec_equiv_exists (k x) Q).mpr
        have hbind' : k x = ok y := by simpa only [bind_tc_ok] using hbind
        exact ⟨y, hbind', hQ⟩
    | vis e next =>
        rw [bind_tc_vis] at hbind
        exact (vis_not_ok hbind).elim
    | div =>
        rw [bind_tc_div] at hbind
        exact (div_not_ok hbind).elim
  · intro h
    obtain ⟨x, hm, hk⟩ := Aeneas.Std.WP.spec_imp_exists h
    rw [hm]
    simpa only [bind_tc_ok] using hk

/-- Canonical ordered fold from an arbitrary initial map, rather than a constant default. -/
def orderedFoldFromG {K V : Type} [DecidableEq K] (mergeFn : V → V → V)
    (initial : HMap K V) (kvs : List (K × V)) : HMap K V :=
  kvs.foldl (fun state kv => insertWith mergeFn kv.1 kv.2 state) initial

/-- One insertion specification parameterized by whether physical capacity failure is admitted. -/
def BuildInsertSpecG (allowFailure : Prop) {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V) (merge : F)
    (mergeFn : V → V → V) (t : table.Table K V H N) (h : HMap K V)
    (key : K) (value : V) : Prop :=
  table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value
    ⦃ result => match result.1 with
      | core.result.Result.Ok _ =>
          ReadRepG cmp HomeInst result.2 (insertWith mergeFn key value h)
      | core.result.Result.Err table.StorageError.ShardFull => allowFailure ⦄

/-- Universal one-step insertion specification at every represented state. -/
def AllBuildInsertSpecsG (allowFailure : Prop) {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V) (merge : F)
    (mergeFn : V → V → V) : Prop :=
  ∀ (t : table.Table K V H N) (h : HMap K V) (key : K) (value : V),
    ReadRepG cmp HomeInst t h →
    BuildInsertSpecG allowFailure cmp cloneV HomeInst MergeInst merge mergeFn t h key value

/-- Universal generated-loop refinement from an arbitrary represented initial map. -/
def BuildLoopSpecG (allowFailure : Prop) {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (cloneK : core.clone.Clone K) (merge : F) (mergeFn : V → V → V) : Prop :=
  ∀ (kvs : Slice (K × V)) (initial : table.Table K V H N) (h : HMap K V),
    ReadRepG cmp HomeInst initial h →
    table.Table.build_loop cmp cloneV HomeInst MergeInst cloneK merge kvs initial 0#usize
      ⦃ result => match result with
        | core.result.Result.Ok final =>
            ReadRepG cmp HomeInst final (orderedFoldFromG mergeFn h (sliceEntriesG kvs))
        | core.result.Result.Err table.StorageError.ShardFull => allowFailure ⦄

/-- Universal one-step specifications are sufficient for the arbitrary-initial-map loop contract. -/
theorem buildLoopSpecG_of_allBuildInsertSpecsG (allowFailure : Prop)
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (cloneK : core.clone.Clone K) (merge : F)
    (mergeFn : V → V → V) (hcloneK : ∀ x, cloneK.clone x = ok x)
    (hcloneV : ∀ x, cloneV.clone x = ok x)
    (hsteps : AllBuildInsertSpecsG (N := N) allowFailure cmp cloneV HomeInst MergeInst merge mergeFn) :
    BuildLoopSpecG (N := N) allowFailure cmp cloneV HomeInst MergeInst cloneK merge mergeFn := by
  intro kvs initial h hinitial
  unfold table.Table.build_loop
  apply loop.spec_decr_nat
    (measure := fun p : table.Table K V H N × Usize => (sliceEntriesG kvs).length - p.2.val)
    (inv := fun p => p.2.val ≤ (sliceEntriesG kvs).length ∧
      ReadRepG cmp HomeInst p.1
        (orderedFoldFromG mergeFn h ((sliceEntriesG kvs).take p.2.val)))
    (post := fun result : core.result.Result (table.Table K V H N) table.StorageError => match result with
      | core.result.Result.Ok final =>
          ReadRepG cmp HomeInst final (orderedFoldFromG mergeFn h (sliceEntriesG kvs))
      | core.result.Result.Err table.StorageError.ShardFull => allowFailure)
  · rintro ⟨state, i⟩ ⟨hi, hrep⟩
    dsimp only at hi hrep ⊢
    unfold table.Table.build_loop.body
    by_cases hlt : i < Slice.len kvs
    · rw [if_pos hlt]
      simp only [hcloneK, hcloneV]
      step
      have hiList : i.val < (sliceEntriesG kvs).length := by
        rw [sliceEntriesG_length]
        exact hlt
      have hinsert := hsteps state
        (orderedFoldFromG mergeFn h ((sliceEntriesG kvs).take i.val)) key value hrep
      unfold BuildInsertSpecG at hinsert
      step
      cases r with
      | Ok resultUnit =>
          have hrep' := r_post
          simp only at hrep'
          have htake : (sliceEntriesG kvs).take (i.val + 1) =
              (sliceEntriesG kvs).take i.val ++ [(key, value)] := by
            rw [List.take_succ_eq_append_getElem hiList]
            congr
            exact key_post.symm
          step
          refine ⟨?_, ?_, ?_⟩
          · simpa [r_post] using Nat.succ_le_iff.mpr hiList
          · simpa [r_post, orderedFoldFromG, htake] using hrep'
          · simp only [r_post]
            omega
      | Err error =>
          cases error
          simpa using r_post
    · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
      have hieq : i.val = (sliceEntriesG kvs).length := by
        rw [sliceEntriesG_length]
        exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
      simpa [hieq] using hrep
  · exact ⟨Nat.zero_le _, by simpa [orderedFoldFromG] using hinitial⟩

/-- Universal loop refinement forces the one-step contract; a singleton stream exposes insertion. -/
theorem allBuildInsertSpecsG_of_buildLoopSpecG (allowFailure : Prop)
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (cloneK : core.clone.Clone K) (merge : F)
    (mergeFn : V → V → V) (hcloneK : ∀ x, cloneK.clone x = ok x)
    (hcloneV : ∀ x, cloneV.clone x = ok x)
    (hloopSpec : BuildLoopSpecG (N := N) allowFailure cmp cloneV HomeInst MergeInst cloneK merge mergeFn) :
    AllBuildInsertSpecsG (N := N) allowFailure cmp cloneV HomeInst MergeInst merge mergeFn := by
  intro t h key value hrep
  let kvs : Slice (K × V) := Slice.from [(key, value)] (by scalar_tac)
  have hs := hloopSpec kvs t h hrep
  unfold table.Table.build_loop at hs
  rw [loop.eq_def] at hs
  have hbody := (wpSpec_bind_iff _ _ _).mp hs
  unfold table.Table.build_loop.body at hbody
  simp [kvs, Slice.len, Slice.index_usize, hcloneK, hcloneV] at hbody
  have hinsert := (wpSpec_bind_iff _ _ _).mp hbody
  unfold BuildInsertSpecG
  apply Aeneas.Std.WP.spec_mono hinsert
  rintro ⟨r, state⟩ hnext
  cases r with
  | Ok resultUnit =>
      cases resultUnit
      have hadd := (wpSpec_bind_iff _ _ _).mp hnext
      obtain ⟨i, hi, hrest⟩ := Aeneas.Std.WP.spec_imp_exists hadd
      have hiSpec := Usize.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac)
      obtain ⟨j, hj, hjval⟩ := Aeneas.Std.WP.spec_imp_exists hiSpec
      have hij : i = j := Result.ok_injective (hi.symm.trans hj)
      subst j
      have hiOne : i = 1#usize := UScalar.eq_of_val_eq (by simpa using hjval)
      subst i
      simp only [Aeneas.Std.WP.spec_ok] at hrest
      rw [loop.eq_def] at hrest
      have hterminal := (wpSpec_bind_iff _ _ _).mp hrest
      simp at hterminal
      simpa [kvs, sliceEntriesG, orderedFoldFromG] using hterminal
  | Err error =>
      cases error
      simpa using hnext

/-- Arbitrary-initial-map loop refinement is equivalent to its universal one-step contract. -/
theorem buildLoopSpecG_iff_allBuildInsertSpecsG (allowFailure : Prop)
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (cloneK : core.clone.Clone K) (merge : F)
    (mergeFn : V → V → V) (hcloneK : ∀ x, cloneK.clone x = ok x)
    (hcloneV : ∀ x, cloneV.clone x = ok x) :
    BuildLoopSpecG (N := N) allowFailure cmp cloneV HomeInst MergeInst cloneK merge mergeFn ↔
      AllBuildInsertSpecsG (N := N) allowFailure cmp cloneV HomeInst MergeInst merge mergeFn :=
  ⟨allBuildInsertSpecsG_of_buildLoopSpecG allowFailure cmp cloneV HomeInst MergeInst cloneK
      merge mergeFn hcloneK hcloneV,
    buildLoopSpecG_of_allBuildInsertSpecsG allowFailure cmp cloneV HomeInst MergeInst cloneK
      merge mergeFn hcloneK hcloneV⟩

/-- Allowing capacity failure recovers exactly `BuildInsertResultG`. -/
theorem buildInsertSpecG_true_iff_resultG {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (merge : F) (mergeFn : V → V → V)
    (t : table.Table K V H N) (h : HMap K V) (key : K) (value : V) :
    BuildInsertSpecG True cmp cloneV HomeInst MergeInst merge mergeFn t h key value ↔
      BuildInsertResultG cmp cloneV HomeInst MergeInst merge mergeFn t h key value := by
  rfl

/-- Forbidding capacity failure recovers exactly the existential `BuildInsertOkG` contract. -/
theorem buildInsertSpecG_false_iff_okG {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (merge : F) (mergeFn : V → V → V)
    (t : table.Table K V H N) (h : HMap K V) (key : K) (value : V) :
    BuildInsertSpecG False cmp cloneV HomeInst MergeInst merge mergeFn t h key value ↔
      BuildInsertOkG cmp cloneV HomeInst MergeInst merge mergeFn t h key value := by
  constructor
  · intro hs
    obtain ⟨result, hresult, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
    rcases result with ⟨result, state⟩
    cases result with
    | Ok resultUnit =>
        cases resultUnit
        exact ⟨state, hresult, hpost⟩
    | Err error =>
        cases error
        exact hpost.elim
  · rintro ⟨state, hresult, hrep⟩
    unfold BuildInsertSpecG
    rw [hresult, Aeneas.Std.WP.spec_ok]
    exact hrep

/-- Capacity-aware loop refinement is equivalent to universal `BuildInsertResultG`. -/
theorem buildLoopSpecG_true_iff_buildInsertResultG {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (cloneK : core.clone.Clone K) (merge : F) (mergeFn : V → V → V)
    (hcloneK : ∀ x, cloneK.clone x = ok x) (hcloneV : ∀ x, cloneV.clone x = ok x) :
    BuildLoopSpecG (N := N) True cmp cloneV HomeInst MergeInst cloneK merge mergeFn ↔
      ∀ (t : table.Table K V H N) (h : HMap K V) (key : K) (value : V),
        ReadRepG cmp HomeInst t h →
        BuildInsertResultG cmp cloneV HomeInst MergeInst merge mergeFn t h key value := by
  rw [buildLoopSpecG_iff_allBuildInsertSpecsG True cmp cloneV HomeInst MergeInst cloneK
    merge mergeFn hcloneK hcloneV]
  unfold AllBuildInsertSpecsG
  exact forall_congr' fun t => forall_congr' fun h => forall_congr' fun key =>
    forall_congr' fun value => forall_congr' fun _ =>
      buildInsertSpecG_true_iff_resultG cmp cloneV HomeInst MergeInst merge mergeFn
        t h key value

/-- Success-only loop refinement is equivalent to universal `BuildInsertOkG`. -/
theorem buildLoopSpecG_false_iff_buildInsertOkG {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (cloneK : core.clone.Clone K) (merge : F) (mergeFn : V → V → V)
    (hcloneK : ∀ x, cloneK.clone x = ok x) (hcloneV : ∀ x, cloneV.clone x = ok x) :
    BuildLoopSpecG (N := N) False cmp cloneV HomeInst MergeInst cloneK merge mergeFn ↔
      ∀ (t : table.Table K V H N) (h : HMap K V) (key : K) (value : V),
        ReadRepG cmp HomeInst t h →
        BuildInsertOkG cmp cloneV HomeInst MergeInst merge mergeFn t h key value := by
  rw [buildLoopSpecG_iff_allBuildInsertSpecsG False cmp cloneV HomeInst MergeInst cloneK
    merge mergeFn hcloneK hcloneV]
  unfold AllBuildInsertSpecsG
  exact forall_congr' fun t => forall_congr' fun h => forall_congr' fun key =>
    forall_congr' fun value => forall_congr' fun _ =>
      buildInsertSpecG_false_iff_okG cmp cloneV HomeInst MergeInst merge mergeFn t h key value

/-! ## Generic-`N` occupied-prefix capacity

These are the generic-`N` analogues of `OccupiedPrefix`, `TableInv`, `ShardKeysDistinct`,
`TableInv2`, `PrefixShardCapacity`, `DistinctKeysPerShard` and `PrefixShardKeyCoverage`.  They
carry the stream-only distinct-routed-key bound to strict physical capacity and hence to
`AllBuildInsertsAdmittedN`. -/

/-- A populated generic-`N` shard consists of occupied slots followed by vacant slots. -/
def OccupiedPrefixN (N : Usize) (slots : List (Option (I64 × I64))) : Prop :=
  ∃ k, k ≤ N.val ∧ NoEmptyIn slots 0 k ∧
    ∀ j, k ≤ j → j < N.val → slots[j]? = some none

/-- Every populated generic-`N` shard has an occupied prefix and vacant suffix. -/
def TableInvN {N : Usize} (t : WordTableN N) : Prop :=
  ∀ (shard : Nat) (slots : Array (Option (I64 × I64)) N),
    t.shards.val[shard]? = some (some slots) → OccupiedPrefixN N slots.val

/-- No key occurs at two physical positions in one populated generic-`N` shard. -/
def ShardKeysDistinctN (N : Usize) (slots : List (Option (I64 × I64))) : Prop :=
  ∀ (key : I64) (i j : Nat) (left right : I64),
    i < N.val → j < N.val →
    slots[i]? = some (some (key, left)) →
    slots[j]? = some (some (key, right)) → i = j

/-- The occupied-prefix invariant with physical key uniqueness in every populated shard. -/
def TableInv2N {N : Usize} (t : WordTableN N) : Prop :=
  TableInvN t ∧ ∀ (shard : Nat) (slots : Array (Option (I64 × I64)) N),
    t.shards.val[shard]? = some (some slots) → ShardKeysDistinctN N slots.val

/-- The generated constructor establishes the generic-`N` occupied-prefix invariant. -/
theorem emptyWordTableN_tableInv {N : Usize} (default : I64) :
    TableInvN (emptyWordTableN (N := N) default) := by
  set_option maxRecDepth 4096 in
  unfold TableInvN
  intro shard slots h
  change (List.replicate 256
    (none : Option (Array (Option (I64 × I64)) N)))[shard]? = some (some slots) at h
  by_cases hs : shard < 256
  · rw [List.getElem?_eq_getElem (by simpa using hs), List.getElem_replicate] at h
    simp at h
  · rw [List.getElem?_eq_none (by simpa using Nat.not_lt.mp hs)] at h
    simp at h

/-- The generated constructor establishes generic-`N` shape and physical key uniqueness. -/
theorem emptyWordTableN_tableInv2 {N : Usize} (default : I64) :
    TableInv2N (emptyWordTableN (N := N) default) := by
  set_option maxRecDepth 4096 in
  refine ⟨emptyWordTableN_tableInv default, ?_⟩
  intro shard slots h
  change (List.replicate 256
    (none : Option (Array (Option (I64 × I64)) N)))[shard]? = some (some slots) at h
  by_cases hs : shard < 256
  · rw [List.getElem?_eq_getElem (by simpa using hs), List.getElem_replicate] at h
    simp at h
  · rw [List.getElem?_eq_none (by simpa using Nat.not_lt.mp hs)] at h
    simp at h

/-- An occupied prefix of length `k` with physical uniqueness has exactly `k` distinct keys,
    represented by an injective enumeration indexed by `Fin k`, at arbitrary `N`. -/
theorem occupiedPrefix_distinct_key_countN {N : Usize}
    (slots : Array (Option (I64 × I64)) N) (k : Nat)
    (hk : k ≤ N.val) (hoccupied : NoEmptyIn slots.val 0 k)
    (hdistinct : ShardKeysDistinctN N slots.val) :
    ∃ keys : Fin k → I64, Function.Injective keys ∧
      ∀ i : Fin k, ∃ value, slots.val[i.val]? = some (some (keys i, value)) := by
  have hexists : ∀ i : Fin k, ∃ key value,
      slots.val[i.val]? = some (some (key, value)) := by
    intro i
    have hiN : i.val < N.val := Nat.lt_of_lt_of_le i.isLt hk
    have hiLen : i.val < slots.val.length := by simpa [slots.property] using hiN
    have hsome : ∃ entry, slots.val[i.val]? = some entry :=
      ⟨slots.val[i.val], List.getElem?_eq_getElem hiLen⟩
    obtain ⟨entry, hentry⟩ := hsome
    cases entry with
    | none =>
        exact (hoccupied i.val (Nat.zero_le _) i.isLt (by simpa [EmptyAt] using hentry)).elim
    | some pair =>
        rcases pair with ⟨key, value⟩
        exact ⟨key, value, hentry⟩
  classical
  let keys : Fin k → I64 := fun i => Classical.choose (hexists i)
  have hvalues : ∀ i : Fin k, ∃ value,
      slots.val[i.val]? = some (some (keys i, value)) := by
    intro i
    exact ⟨Classical.choose (Classical.choose_spec (hexists i)),
      Classical.choose_spec (Classical.choose_spec (hexists i))⟩
  refine ⟨keys, ?_, hvalues⟩
  intro i j hij
  obtain ⟨vi, hi⟩ := hvalues i
  obtain ⟨vj, hj⟩ := hvalues j
  apply Fin.ext
  exact hdistinct (keys i) i.val j.val vi vj
    (Nat.lt_of_lt_of_le i.isLt hk) (Nat.lt_of_lt_of_le j.isLt hk) hi (by simpa [hij] using hj)

/-- Fallback capacity contract on physical representatives of each canonical prefix fold at
    arbitrary `N`: every populated shard has an occupied-prefix witness strictly shorter than `N`. -/
def PrefixShardCapacityN {N : Usize} (default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (state : WordTableN N),
    i.val < (sliceEntries kvs).length →
    ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)) →
    TableInvN state ∧
      ∀ (shard : Nat) (slots : Array (Option (I64 × I64)) N) (k : Nat),
        state.shards.val[shard]? = some (some slots) →
        k ≤ N.val → NoEmptyIn slots.val 0 k →
        (∀ j, k ≤ j → j < N.val → slots.val[j]? = some none) → k < N.val

/-- Stream-only admission contract at arbitrary `N`: every proper prefix has fewer than `N`
    distinct keys routed to each physical shard. -/
def DistinctKeysPerShardN (N : Usize) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Nat) (shard : Usize), i < (sliceEntries kvs).length →
    (routedStreamKeys kvs i shard).card < N.val

/-- Physical/stream coverage at arbitrary `N`: shape and uniqueness, and every occupied-prefix key
    is covered by a distinct routed key from the corresponding stream prefix. -/
def PrefixShardKeyCoverageN {N : Usize} (default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (state : WordTableN N),
    i.val < (sliceEntries kvs).length →
    ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)) →
    TableInv2N state ∧
      ∀ (shard : Usize) (slots : Array (Option (I64 × I64)) N) (k : Nat),
        Array.index_usize state.shards shard = ok (some slots) →
        k ≤ N.val → NoEmptyIn slots.val 0 k →
        ∀ (j : Nat) (key value : I64), j < k →
          slots.val[j]? = some (some (key, value)) →
          key ∈ routedStreamKeys kvs i.val shard

/-- Distinct routed stream keys imply strict physical prefix capacity at arbitrary `N` once the
    explicit physical/stream coverage invariant is supplied. -/
theorem prefixShardCapacity_of_distinctKeysPerShardN {N : Usize} (default : I64)
    (kvs : Slice (I64 × I64)) (hdistinct : DistinctKeysPerShardN N kvs)
    (hcoverage : PrefixShardKeyCoverageN (N := N) default kvs) :
    PrefixShardCapacityN (N := N) default kvs := by
  intro i state hi hrep
  obtain ⟨hinv, hcovered⟩ := hcoverage i state hi hrep
  refine ⟨hinv.1, ?_⟩
  intro shardNat slots k houter hk hoccupied _hvacant
  have hshardBound : shardNat < 256 := by
    have hlen : shardNat < state.shards.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter
      simp at houter
    simpa [state.shards.property] using hlen
  let shard : Usize := ⟨shardNat, by scalar_tac⟩
  have houterIndex : Array.index_usize state.shards shard = ok (some slots) :=
    index_usize_eq_ok_of_getElem?_eq_some state.shards shard (some slots) houter
  obtain ⟨keys, hkeysInjective, hkeysSlots⟩ :=
    occupiedPrefix_distinct_key_countN slots k hk hoccupied
      (hinv.2 shardNat slots houter)
  let intoStream : Fin k → { key // key ∈ routedStreamKeys kvs i.val shard } := fun j =>
    ⟨keys j, by
      obtain ⟨value, hslot⟩ := hkeysSlots j
      exact hcovered shard slots k houterIndex hk hoccupied j.val (keys j) value j.isLt hslot⟩
  have hIntoInjective : Function.Injective intoStream := by
    intro a b hab
    apply hkeysInjective
    exact Subtype.ext_iff.mp hab
  have hcard : k ≤ (routedStreamKeys kvs i.val shard).card := by
    simpa [intoStream] using Fintype.card_le_of_injective intoStream hIntoInjective
  exact Nat.lt_of_le_of_lt hcard (hdistinct i.val shard hi)

/-- Strict occupied-prefix capacity at every canonical prefix supplies all one-step admission
    premises required by the generated build loop at arbitrary `N`. -/
theorem allBuildInsertsAdmitted_of_prefixShardCapacityN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64))
    (hcapacity : PrefixShardCapacityN (N := N) default kvs) :
    AllBuildInsertsAdmittedN (N := N) default kvs := by
  intro i state key value hi _hindex hrep
  obtain ⟨hinv, hload⟩ := hcapacity i state hi hrep
  let locate := generatedLocator state
  have hroutes : Routes state locate := generatedLocator_routes state
  have hfunctional : FunctionalRep locate state
      (orderedFold default ((sliceEntries kvs).take i.val)) :=
    (generatedFunctionalRep_iff_readRep state
      (orderedFold default ((sliceEntries kvs).take i.val))).mpr hrep
  have hroute := hroutes key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, slot⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      exact ⟨locate, hfunctional, hroutes, Or.inl
        ⟨shard, slot, slots, old, hlocate, hshard, hfind, houter, hslot⟩⟩
  | none =>
      rw [hlocate] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      obtain ⟨routedShard, hroutedShard, hshardBound⟩ := word_shard_index_spec state key
      have hshardEq : routedShard = shard :=
        Aeneas.Std.Result.ok_injective (hroutedShard.symm.trans hshard)
      subst routedShard
      have hshardLen : shard.val < state.shards.val.length := by
        rw [state.shards.property]
        exact hshardBound
      obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
        (Array.index_usize_spec state.shards shard hshardLen)
      cases outer with
      | none =>
          let next := insertVacant state shard (Array.repeat N none) 0#usize key
            (core.num.I64.wrapping_add value state.default)
          have hroutes' : Routes next (insertLocator locate key shard 0#usize) :=
            routes_insertEmptyShardN hN locate state key
              (core.num.I64.wrapping_add value state.default) shard hroutes hshard houter
              hshardBound
          exact ⟨locate, hfunctional, hroutes, Or.inr (Or.inr
            ⟨shard, hlocate, hshard, hfind, houter, hshardBound, hroutes'⟩)⟩
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
            (some slots) houter
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hinv shard.val slots houterGet
          have hkLt : k < N.val := hload shard.val slots k houterGet hk hoccupied hvacant
          have hfindSlots : table.Table.find_slots core.cmp.EqI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64 slots key = ok none := by
            unfold table.Table.find at hfind
            rw [houter] at hfind
            simpa only [bind_tc_ok] using hfind
          obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
            (find_slots_spec_generic slots key 0#usize (Nat.zero_le _))
          unfold table.Table.find_slots at hfindSlots
          rw [hfindSlots] at hfound
          have hfoundNone : found = none := Aeneas.Std.Result.ok_injective hfound.symm
          subst found
          have hnomatch : NoMatchIn key slots.val 0 N.val := by
            simpa [FindPostN] using hpost
          let first : Usize := ⟨k, by scalar_tac⟩
          have hfirst : first < N := hkLt
          have hslotGet : slots.val[first.val]? = some none := hvacant k le_rfl hkLt
          have hslot : Array.index_usize slots first = ok none :=
            index_usize_eq_ok_of_getElem?_eq_some slots first none hslotGet
          exact ⟨locate, hfunctional, hroutes, Or.inr (Or.inl
            ⟨shard, first, slots, hlocate, hshard, hfind, houter, hfirst, hoccupied,
              fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hkLt), hslot⟩)⟩

/-- The stream-only distinct-key bound yields successful generic-`N` generated build refinement
    under the explicit physical/stream coverage invariant. -/
theorem table_build_readRepN_of_distinctKeysPerShardN {N : Usize} (hN : 0#usize < N)
    (default : I64) (kvs : Slice (I64 × I64))
    (hdistinct : DistinctKeysPerShardN N kvs)
    (hcoverage : PrefixShardKeyCoverageN (N := N) default kvs) :
    ∃ final, wordBuildN (N := N) default kvs = ok (core.result.Result.Ok final) ∧
      ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) :=
  table_build_readRepN hN default kvs
    (allBuildInsertsAdmitted_of_prefixShardCapacityN hN default kvs
      (prefixShardCapacity_of_distinctKeysPerShardN default kvs hdistinct hcoverage))

end Apeiron.TableRefinement
