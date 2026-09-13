import ApeironWord
import Apeiron.Table.Basic
import Apeiron.Table.Adjacency
import Apeiron.Table.Home

/-! # Refinement of the executing fixed-shard word table -/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

abbrev WordTable := table.Table I64 I64 table.WordHome 16#usize

/-- The generated state returned by the real constructor. -/
def emptyWordTable (default : I64) : WordTable where
  default := default
  home := ()
  shards := Array.repeat 256#usize none

/-- Extensional representation at the executing table's public read boundary. -/
def Rep (t : WordTable) (h : HMap I64 I64) : Prop :=
  ∀ key, word_table_get t key ⦃ value => value = h key ⦄

theorem cast_u64_zero_usize : UScalar.cast .Usize 0#u64 = 0#usize := by
  apply UScalar.eq_of_val_eq
  rw [UScalar.cast_val_mod_pow_of_inBounds_eq .Usize 0#u64 (by scalar_tac)]
  rfl

theorem u64_zero_mod_256 : (Mod.mod 0#64 256#64#uscalar : U64) = 0#u64 := by
  apply UScalar.eq_of_val_eq
  decide

theorem u64_4096_mod_256 : (Mod.mod 4096#64 256#64#uscalar : U64) = 0#u64 := by
  apply UScalar.eq_of_val_eq
  decide

/-- The real extracted constructor is the all-empty fixed-shard state. -/
theorem word_table_new_eq_empty (default : I64) :
    word_table_new default = ok (emptyWordTable default) := by
  rfl

/-- The generated all-empty table returns its default at every word key. -/
theorem emptyWordTable_read (default key : I64) :
    word_table_get (emptyWordTable default) key ⦃ value => value = default ⦄ := by
  set_option maxRecDepth 4096 in
  unfold word_table_get table.Table.get table.Table.shard_index
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
        (emptyWordTable default).shards.val[(UScalar.cast .Usize shard).val] = none := by
      change (List.replicate 256 none)[(UScalar.cast .Usize shard).val] = none
      rw [List.getElem_replicate]
    have ho : o = none := o_post.trans hslot
    rw [ho]
    simp [emptyWordTable]

/-- Constructor initialization: the real constructor represents the constant-default `HMap`. -/
theorem word_table_new_rep (default : I64) :
    ∃ t, word_table_new default = ok t ∧ Rep t (fun _ => default) := by
  refine ⟨emptyWordTable default, word_table_new_eq_empty default, ?_⟩
  intro key
  exact emptyWordTable_read default key

/-- Joint hypotheses are inhabited by the state returned from the real constructor. -/
theorem word_table_new_joint_witness (default key : I64) :
    ∃ t h, word_table_new default = ok t ∧ Rep t h ∧
      Apeiron.Table.get h key = default := by
  refine ⟨emptyWordTable default, fun _ => default, word_table_new_eq_empty default, ?_, rfl⟩
  intro x
  exact emptyWordTable_read default x

/-- The concrete shard produced by the first write at word key zero. -/
def oneSlots (default value : I64) : Array (Option (I64 × I64)) 16#usize :=
  (Array.repeat 16#usize none).set 0#usize
    (some (0#i64, core.num.I64.wrapping_add value default))

/-- The concrete state produced by the first write at word key zero. -/
def oneWordTable (default value : I64) : WordTable where
  default := default
  home := ()
  shards := (Array.repeat 256#usize none).set 0#usize
    (some (oneSlots default value))

/-- The real insertion body is new-first: its first write merges `value` with `default`. -/
theorem table_insert_empty_zero (default value : I64) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64
      (emptyWordTable default) () 0#i64 value =
      ok (core.result.Result.Ok (), oneWordTable default value) := by
  have hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (emptyWordTable default) 0#i64 =
      ok 0#usize := by
    simp [table.Table.shard_index,
      table.WordHome.Insts.Apeiron_wordTableHomeI64.home, HMod.hMod,
      UScalar.rem, IScalar.hcast]
    rw [u64_zero_mod_256, cast_u64_zero_usize]
  have hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (emptyWordTable default) 0#usize 0#i64 =
      ok none := by
    set_option maxRecDepth 4096 in
    simp [table.Table.find, emptyWordTable, Array.index_usize, Array.repeat_val]
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  set_option maxRecDepth 4096 in
  simp [emptyWordTable, oneWordTable,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
    Array.index_usize, Array.index_mut_usize, Array.update, Array.repeat_val]
  apply Aeneas.Std.Array.ext
  simp [oneSlots, Array.repeat_val, Array.set_val_eq]
  apply Aeneas.Std.Array.ext
  simp [Array.repeat_val, Array.set_val_eq]

theorem word_table_insert_new_first (default value : I64) :
    word_table_insert (emptyWordTable default) 0#i64 value =
      ok (oneWordTable default value, core.result.Result.Ok ()) := by
  simp [word_table_insert, table_insert_empty_zero]

/-- The generated bounded search finds the key in the first slot of `oneSlots`. -/
theorem oneSlots_find (default value : I64) :
    table.Table.find_slots core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (oneSlots default value) 0#i64 =
      ok (some 0#usize) := by
  unfold table.Table.find_slots table.Table.find_slots_loop
  have hs : loop
      (fun i => table.Table.find_slots_loop.body core.cmp.EqI64
        (oneSlots default value) 0#i64 i) 0#usize
      ⦃ r => r = some 0#usize ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun (i : Usize) => 16 - i.val)
      (inv := fun i => i = 0#usize)
      (post := fun r => r = some 0#usize)
    · intro i hi
      subst i
      simp [table.Table.find_slots_loop.body, oneSlots, Array.index_usize,
        Array.repeat_val, core.cmp.impls.PartialEqI64.eq, liftFun2]
    · rfl
  obtain ⟨r, hr, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hpost] at hr
  exact hr

theorem oneWordTable_find (default value : I64) :
    table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (oneWordTable default value) 0#usize 0#i64 =
      ok (some 0#usize) := by
  unfold table.Table.find
  set_option maxRecDepth 4096 in
  simp [oneWordTable, Array.index_usize, Array.repeat_val, oneSlots_find]

/-- Reading the first successful write is the corpus `insertWith` self equation. -/
theorem word_table_get_new_first (default value : I64) :
    word_table_get (oneWordTable default value) 0#i64 =
      ok (Apeiron.Table.get
        (insertWith core.num.I64.wrapping_add 0#i64 value (fun _ => default)) 0#i64) := by
  have hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (oneWordTable default value) 0#i64 =
      ok 0#usize := by
    simp [table.Table.shard_index,
      table.WordHome.Insts.Apeiron_wordTableHomeI64.home, HMod.hMod,
      UScalar.rem, IScalar.hcast]
    rw [u64_zero_mod_256, cast_u64_zero_usize]
  unfold word_table_get table.Table.get
  rw [hshard]
  simp only [bind_tc_ok]
  rw [oneWordTable_find]
  set_option maxRecDepth 4096 in
  simp [oneWordTable, oneSlots, Array.index_usize, Array.repeat_val,
    Apeiron.Table.get_insertWith_self]

/-- The concrete state after replacing the present key zero. -/
def replacedWordTable (default first second : I64) : WordTable where
  default := default
  home := ()
  shards := (Array.repeat 256#usize none).set 0#usize
    (some (oneSlots (core.num.I64.wrapping_add first default) second))

/-- A present key updates even though insertion uses the same bounded storage slot. -/
theorem table_insert_present_zero (default first second : I64) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64
      (oneWordTable default first) () 0#i64 second =
      ok (core.result.Result.Ok (), replacedWordTable default first second) := by
  have hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (oneWordTable default first) 0#i64 =
      ok 0#usize := by
    simp [table.Table.shard_index,
      table.WordHome.Insts.Apeiron_wordTableHomeI64.home, HMod.hMod,
      UScalar.rem, IScalar.hcast]
    rw [u64_zero_mod_256, cast_u64_zero_usize]
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [oneWordTable_find]
  set_option maxRecDepth 4096 in
  simp [oneWordTable, oneSlots, replacedWordTable,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
    Array.index_usize, Array.index_mut_usize, Array.repeat_val]
  apply Aeneas.Std.Array.ext
  simp [Array.repeat_val, Array.set_val_eq]
  apply Aeneas.Std.Array.ext
  simp [Array.repeat_val, Array.set_val_eq]

theorem word_table_insert_replaces_present (default first second : I64) :
    word_table_insert (oneWordTable default first) 0#i64 second =
      ok (replacedWordTable default first second, core.result.Result.Ok ()) := by
  simp [word_table_insert, table_insert_present_zero]

theorem replacedWordTable_find (default first second : I64) :
    table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      (replacedWordTable default first second) 0#usize 0#i64 = ok (some 0#usize) := by
  unfold table.Table.find
  set_option maxRecDepth 4096 in
  simp [replacedWordTable, Array.index_usize, Array.repeat_val,
    oneSlots_find (core.num.I64.wrapping_add first default) second]

/-- The replacement read is the second ordered corpus insertion at the present key. -/
theorem word_table_get_replaced (default first second : I64) :
    word_table_get (replacedWordTable default first second) 0#i64 =
      ok (Apeiron.Table.get
        (insertWith core.num.I64.wrapping_add 0#i64 second
          (insertWith core.num.I64.wrapping_add 0#i64 first (fun _ => default))) 0#i64) := by
  rw [Apeiron.Table.get_insertWith_self]
  change word_table_get (replacedWordTable default first second) 0#i64 =
    ok (core.num.I64.wrapping_add second (Apeiron.Table.get
      (insertWith core.num.I64.wrapping_add 0#i64 first (fun _ => default)) 0#i64))
  rw [Apeiron.Table.get_insertWith_self]
  have hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      (replacedWordTable default first second) 0#i64 = ok 0#usize := by
    simp [table.Table.shard_index,
      table.WordHome.Insts.Apeiron_wordTableHomeI64.home, HMod.hMod,
      UScalar.rem, IScalar.hcast]
    rw [u64_zero_mod_256, cast_u64_zero_usize]
  unfold word_table_get table.Table.get
  rw [hshard]
  simp only [bind_tc_ok]
  rw [replacedWordTable_find]
  set_option maxRecDepth 4096 in
  simp [replacedWordTable, oneSlots, Array.index_usize, Array.repeat_val]

/-- The two-write stream used to check build order at one key. -/
def twoWrites (first second : I64) : Slice (I64 × I64) :=
  Slice.from [(0#i64, first), (0#i64, second)] (by scalar_tac)

theorem table_build_loop_two (default first second : I64) :
    table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 ()
      (twoWrites first second) (emptyWordTable default) 0#usize =
      ok (core.result.Result.Ok (replacedWordTable default first second)) := by
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 ()
        (twoWrites first second) state i)
      (emptyWordTable default, 0#usize)
      ⦃ r => r = core.result.Result.Ok (replacedWordTable default first second) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTable × Usize => 2 - p.2.val)
      (inv := fun p =>
        (p.1 = emptyWordTable default ∧ p.2 = 0#usize) ∨
        (p.1 = oneWordTable default first ∧ p.2 = 1#usize) ∨
        (p.1 = replacedWordTable default first second ∧ p.2 = 2#usize))
      (post := fun r => r = core.result.Result.Ok (replacedWordTable default first second))
    · rintro ⟨state, i⟩ hi
      rcases hi with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · simp [table.Table.build_loop.body, twoWrites, Slice.index_usize,
          table_insert_empty_zero]
        step
        scalar_tac
      · simp [table.Table.build_loop.body, twoWrites, Slice.index_usize,
          table_insert_present_zero]
        step
        scalar_tac
      · simp [table.Table.build_loop.body, twoWrites]
    · exact Or.inl ⟨rfl, rfl⟩
  obtain ⟨r, hr, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hpost] at hr
  exact hr

/-- The real build body applies same-key writes in stream order. -/
theorem word_table_build_ordered (default first second : I64) :
    word_table_build default (twoWrites first second) =
      ok (core.result.Result.Ok (replacedWordTable default first second)) := by
  unfold word_table_build table.Table.build
  rw [show table.Table.new 16#usize table.WordHome.Insts.Apeiron_wordTableHomeI64 default () =
      ok (emptyWordTable default) by rfl]
  simp [table_build_loop_two]

/-- The ordered build read is the corpus `get_build_acc` fold for the same stream. -/
theorem word_table_build_get_acc (default first second : I64) :
    word_table_get (replacedWordTable default first second) 0#i64 =
      ok (Apeiron.Table.get
        (Apeiron.Table.build core.num.I64.wrapping_add default
          [(0#i64, first), (0#i64, second)]) 0#i64) := by
  rw [word_table_get_replaced]
  unfold Apeiron.Table.build
  rw [Apeiron.Table.get_build_acc]
  congr 1

/-- A shard position holds `key`. -/
def MatchesAt (key : I64) (slots : List (Option (I64 × I64))) (i : Nat) : Prop :=
  ∃ value, slots[i]? = some (some (key, value))

def NoMatchIn (key : I64) (slots : List (Option (I64 × I64))) (a b : Nat) : Prop :=
  ∀ i, a ≤ i → i < b → ¬ MatchesAt key slots i

def FindInv (key : I64) (slots : List (Option (I64 × I64))) (start : Nat)
    (i : Usize) : Prop :=
  i.val ≤ 16 ∧ start ≤ i.val ∧ NoMatchIn key slots start i.val

def FindPost (key : I64) (slots : List (Option (I64 × I64))) (start : Nat) :
    Option Usize → Prop
  | some i => i.val < 16 ∧ start ≤ i.val ∧ MatchesAt key slots i.val ∧
      NoMatchIn key slots start i.val
  | none => NoMatchIn key slots start 16

@[step] theorem eqI64_spec (a b : I64) :
    core.cmp.impls.PartialEqShared.eq core.cmp.EqI64.partialEqInst a b
      ⦃ r => r = decide (a = b) ⦄ := by
  change Aeneas.Std.WP.spec (ok (decide (a = b))) (fun r => r = decide (a = b))
  exact Aeneas.Std.WP.spec.ret rfl

theorem noMatchIn_succ {key : I64} {slots : List (Option (I64 × I64))} {a i : Nat}
    (h : NoMatchIn key slots a i) (hnot : ¬ MatchesAt key slots i) :
    NoMatchIn key slots a (i + 1) := by
  intro j ha hj
  rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hlt | rfl
  · exact h j ha hlt
  · exact hnot

theorem matchesAt_of_get {key stored value i} {slots : List (Option (I64 × I64))}
    (hi : i < slots.length) (h : slots[i] = some (stored, value)) (heq : stored = key) :
    MatchesAt key slots i := by
  exact ⟨value, by rw [List.getElem?_eq_getElem hi, h, heq]⟩

theorem not_matchesAt_of_none {key i} {slots : List (Option (I64 × I64))}
    (hi : i < slots.length) (h : slots[i] = none) : ¬ MatchesAt key slots i := by
  rintro ⟨value, hv⟩
  rw [List.getElem?_eq_getElem hi, h] at hv
  simp at hv

theorem not_matchesAt_of_ne {key stored value i} {slots : List (Option (I64 × I64))}
    (hi : i < slots.length) (h : slots[i] = some (stored, value)) (hne : stored ≠ key) :
    ¬ MatchesAt key slots i := by
  rintro ⟨v, hv⟩
  rw [List.getElem?_eq_getElem hi, h] at hv
  exact hne (Prod.mk.inj (Option.some.inj (Option.some.inj hv))).1

/-- The generated shard search is total and returns the first matching slot. -/
theorem find_slots_spec (slots : Array (Option (I64 × I64)) 16#usize) (key : I64)
    (start : Usize) (hstart : start.val ≤ 16) :
    table.Table.find_slots_loop core.cmp.EqI64 slots key start
      ⦃ r => FindPost key slots.val start.val r ⦄ := by
  unfold table.Table.find_slots_loop
  apply loop.spec_decr_nat
    (measure := fun i : Usize => 16 - i.val)
    (inv := fun i => FindInv key slots.val start.val i)
    (post := fun r => FindPost key slots.val start.val r)
  · intro i hi
    obtain ⟨hile, hstartle, hno⟩ := hi
    unfold table.Table.find_slots_loop.body
    by_cases hlt : i < 16#usize
    · rw [if_pos hlt]
      have hin : i.val < 16 := hlt
      have hib : i.val < slots.val.length := by rw [Array.length_eq]; exact hin
      step
      rcases o with _ | ⟨stored, value⟩
      · have ho : slots.val[i.val] = none := o_post.symm
        step
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · simpa [r_post] using noMatchIn_succ hno (not_matchesAt_of_none hib ho)
        · omega
      · have ho : slots.val[i.val] = some (stored, value) := o_post.symm
        step
        split
        · rename_i heq
          have hk : stored = key := by
            rw [heq] at r_post
            exact of_decide_eq_true r_post.symm
          simp only [Aeneas.Std.WP.spec_ok]
          exact ⟨hin, hstartle, matchesAt_of_get hib ho hk, hno⟩
        · rename_i hne
          have hk : stored ≠ key := by
            intro hk
            exact hne (by simp [r_post, hk])
          step
          refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
          · omega
          · omega
          · simpa [i1_post] using noMatchIn_succ hno (not_matchesAt_of_ne hib ho hk)
          · omega
    · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
      have hi16 : i.val = 16 := by
        have hge : 16 ≤ i.val := not_lt.mp hlt
        omega
      simpa [FindPost, hi16] using hno
  · exact ⟨hstart, le_rfl, fun j hj hjlt => by omega⟩

/-- A concrete full shard: sixteen distinct keys with the same home residue zero. -/
def fullSlots : Array (Option (I64 × I64)) 16#usize :=
  Array.make 16#usize
    [some (0#i64, 0#i64), some (256#i64, 0#i64), some (512#i64, 0#i64),
      some (768#i64, 0#i64), some (1024#i64, 0#i64), some (1280#i64, 0#i64),
      some (1536#i64, 0#i64), some (1792#i64, 0#i64), some (2048#i64, 0#i64),
      some (2304#i64, 0#i64), some (2560#i64, 0#i64), some (2816#i64, 0#i64),
      some (3072#i64, 0#i64), some (3328#i64, 0#i64), some (3584#i64, 0#i64),
      some (3840#i64, 0#i64)] (by simp)

def fullWordTable : WordTable where
  default := 0#i64
  home := ()
  shards := (Array.repeat 256#usize none).set 0#usize (some fullSlots)

theorem fullSlots_noMatch_4096 : NoMatchIn 4096#i64 fullSlots.val 0 16 := by
  intro i _ hi
  rintro ⟨value, hv⟩
  interval_cases i <;> simp [fullSlots, Array.make_val] at hv

theorem fullSlots_find_4096 :
    table.Table.find_slots core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 fullSlots 4096#i64 = ok none := by
  unfold table.Table.find_slots
  obtain ⟨r, hr, hp⟩ := Aeneas.Std.WP.spec_imp_exists
    (find_slots_spec fullSlots 4096#i64 0#usize (by decide))
  rcases r with _ | i
  · exact hr
  · exact (fullSlots_noMatch_4096 i.val (by omega) hp.1 hp.2.2.1).elim

theorem fullWordTable_find_4096 :
    table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
      fullWordTable 0#usize 4096#i64 = ok none := by
  unfold table.Table.find
  set_option maxRecDepth 4096 in
  simp [fullWordTable, Array.index_usize, Array.repeat_val, fullSlots_find_4096]

def EmptyAt (slots : List (Option (I64 × I64))) (i : Nat) : Prop :=
  slots[i]? = some none

def NoEmptyIn (slots : List (Option (I64 × I64))) (a b : Nat) : Prop :=
  ∀ i, a ≤ i → i < b → ¬ EmptyAt slots i

def InsertInv (slots : List (Option (I64 × I64))) (start : Nat) (i : Usize) : Prop :=
  i.val ≤ 16 ∧ start ≤ i.val ∧ NoEmptyIn slots start i.val

theorem noEmptyIn_succ {slots : List (Option (I64 × I64))} {a i : Nat}
    (h : NoEmptyIn slots a i) (hnot : ¬ EmptyAt slots i) : NoEmptyIn slots a (i + 1) := by
  intro j ha hj
  rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hlt | rfl
  · exact h j ha hlt
  · exact hnot

theorem emptyAt_of_none {slots : List (Option (I64 × I64))} {i : Nat}
    (hi : i < slots.length) (h : slots[i] = none) : EmptyAt slots i := by
  rw [EmptyAt, List.getElem?_eq_getElem hi, h]

/-- A full shard makes the generated vacant-slot loop return `ShardFull` unchanged. -/
theorem insert_full_loop_spec
    (back : Option (Array (Option (I64 × I64)) 16#usize) →
      Array (Option (Array (Option (I64 × I64)) 16#usize)) 256#usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (default key value : I64)
    (start : Usize) (hstart : start.val ≤ 16)
    (hfull : NoEmptyIn slots.val start.val 16) :
    table.Table.insert_with_loop0 core.clone.CloneI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 back default () key value slots start
      ⦃ r => r = (core.result.Result.Err table.StorageError.ShardFull, back (some slots)) ⦄ := by
  unfold table.Table.insert_with_loop0
  apply loop.spec_decr_nat
    (measure := fun i : Usize => 16 - i.val)
    (inv := fun i => InsertInv slots.val start.val i)
    (post := fun r => r = (core.result.Result.Err table.StorageError.ShardFull, back (some slots)))
  · intro i hi
    obtain ⟨hile, hstartle, hno⟩ := hi
    unfold table.Table.insert_with_loop0.body
    by_cases hlt : i < 16#usize
    · rw [if_pos hlt]
      have hin : i.val < 16 := hlt
      have hib : i.val < slots.val.length := by rw [Array.length_eq]; exact hin
      step
      rcases o with _ | ⟨stored, storedValue⟩
      · have ho : slots.val[i.val] = none := o_post.symm
        exact (hfull i.val (by omega) hin (emptyAt_of_none hib ho)).elim
      · simp
        step
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · simpa [i1_post] using noEmptyIn_succ hno
            (hfull i.val (by omega) hin)
        · omega
    · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
  · exact ⟨hstart, le_rfl, fun j hj hjlt => by omega⟩

theorem fullSlots_noEmpty : NoEmptyIn fullSlots.val 0 16 := by
  intro i _ hi
  interval_cases i <;> simp [EmptyAt, fullSlots, Array.make_val]

def fullBack (shard : Option (Array (Option (I64 × I64)) 16#usize)) :
    Array (Option (Array (Option (I64 × I64)) 16#usize)) 256#usize :=
  (Array.repeat 256#usize none).set 0#usize shard

theorem insert_full_loop_eq :
    table.Table.insert_with_loop0 core.clone.CloneI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 fullBack 0#i64 () 4096#i64
      1#i64 fullSlots 0#usize =
      ok (core.result.Result.Err table.StorageError.ShardFull, fullBack (some fullSlots)) := by
  obtain ⟨r, hr, hp⟩ := Aeneas.Std.WP.spec_imp_exists
    (insert_full_loop_spec fullBack fullSlots 0#i64 4096#i64 1#i64 0#usize
      (by decide) fullSlots_noEmpty)
  rw [hp] at hr
  exact hr

/-- The actual insertion refuses an absent key in a full home shard and returns the same state. -/
theorem table_insert_shardFull :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 fullWordTable () 4096#i64 1#i64 =
      ok (core.result.Result.Err table.StorageError.ShardFull, fullWordTable) := by
  have hback : Aeneas.Std.Array.set fullWordTable.shards 0#usize = fullBack := by
    funext shard
    apply Aeneas.Std.Array.ext
    simp only [fullWordTable, fullBack, Array.set_val_eq]
    rw [List.set_set]
  have hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 fullWordTable 4096#i64 = ok 0#usize := by
    simp [table.Table.shard_index,
      table.WordHome.Insts.Apeiron_wordTableHomeI64.home, HMod.hMod,
      UScalar.rem, IScalar.hcast]
    rw [u64_4096_mod_256, cast_u64_zero_usize]
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [fullWordTable_find_4096]
  set_option maxRecDepth 4096 in
  simp [fullWordTable, Array.index_usize, Array.index_mut_usize,
    Array.repeat_val]
  have hback' :
      ((Array.repeat 256#usize none).set 0#usize (some fullSlots)).set 0#usize = fullBack := by
    simpa only [fullWordTable] using hback
  rw [hback', insert_full_loop_eq]
  simp [fullBack]

/-- Refusal witness: `ShardFull`, absent key, full shard, and byte-for-byte unchanged generated state. -/
theorem word_table_shardFull_refusal :
    word_table_insert fullWordTable 4096#i64 1#i64 =
        ok (fullWordTable, core.result.Result.Err table.StorageError.ShardFull) ∧
      NoMatchIn 4096#i64 fullSlots.val 0 16 ∧ NoEmptyIn fullSlots.val 0 16 := by
  refine ⟨?_, fullSlots_noMatch_4096, fullSlots_noEmpty⟩
  simp [word_table_insert, table_insert_shardFull]

/-- One checked term jointly inhabits constructor success, write success, and resource refusal. -/
theorem word_table_joint_branch_witness :
    word_table_new 0#i64 = ok (emptyWordTable 0#i64) ∧
    word_table_insert (emptyWordTable 0#i64) 0#i64 1#i64 =
      ok (oneWordTable 0#i64 1#i64, core.result.Result.Ok ()) ∧
    word_table_insert fullWordTable 4096#i64 1#i64 =
      ok (fullWordTable, core.result.Result.Err table.StorageError.ShardFull) := by
  exact ⟨word_table_new_eq_empty 0#i64, word_table_insert_new_first 0#i64 1#i64,
    word_table_shardFull_refusal.1⟩

end Apeiron.TableRefinement
