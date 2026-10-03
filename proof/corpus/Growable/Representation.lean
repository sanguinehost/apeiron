import Growable.Geometry
import Growable.Refinement
import Growable.Scan

/-!
# Whole-table growable representation

This ratifies the whole-table contract required by the storage-refinement note, extending the
selected-entry relations in `Growable.Relocation` to the complete generated flat/spill table.  The
represented semantic object is exactly `Apeiron.Table.{HMap,get,insertWith,build}` on
`(Sigma × I64)` keys.  An explicit logical live set is retained because a live override may equal the
default value; the support of a total function cannot recover that physical fact.

`ReservedRepR` denotes materialized relocation storage before the pending `insertWith` is applied.
`PendingCommittedR` states the exact old-map/new-map boundary.  Constructing it from the generated
commit roots remains a separate theorem requiring `SearchDecisionCorrect` and callback laws.
-/

namespace Apeiron.GrowableRepresentation

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

abbrev LogicalKey := cell.Sigma × I64
abbrev LogicalEntry := LogicalKey × I64
abbrev WordTable (R : Type) := table.Table I64 I64 table.WordHome R 16#usize
abbrev Location := Apeiron.GrowableSearch.Location

noncomputable local instance : DecidableEq LogicalKey := Classical.decEq _

/-- A flat live bit or nonzero spill slot tag is the sole occupancy authority. -/
def Occupied {R : Type} (state : WordTable R) : Location → Prop
  | .flat index => ∃ slot,
      state.resident.slots.val[index.val]? = some slot ∧ slot.live = true
  | .spill shard index => ∃ region slot,
      state.shards.val[shard.val]? = some region ∧
      region.slots.val[index.val]? = some slot ∧ slot.tag ≠ 0#u8

/-- The logical keys represented by live physical entries are exactly the declared live set. -/
def LiveSetEqual {R : Type} (state : WordTable R) (decode : cell.Sigma → R → Option I64)
    (live : LogicalKey → Prop) : Prop :=
  ∀ key, live key ↔ ∃ location value,
    Apeiron.GrowableSearch.entryAt state decode location = some (key, value)

/-- Every live observation denotes the canonical total map; every non-live key reads the default. -/
def ObservationCorrect {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64) : Prop :=
  (∀ location key value,
      Apeiron.GrowableSearch.entryAt state decode location = some (key, value) →
      Apeiron.Table.get logical key = value) ∧
  (∀ key, ¬live key → Apeiron.Table.get logical key = state.default)

/-- No logical key has two physical representatives in the composite flat/spill layout. -/
def UniqueRepresentation {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) : Prop :=
  ∀ location₁ location₂ key value₁ value₂,
    Apeiron.GrowableSearch.entryAt state decode location₁ = some (key, value₁) →
    Apeiron.GrowableSearch.entryAt state decode location₂ = some (key, value₂) →
    location₁ = location₂ ∧ value₁ = value₂

/-- Every occupancy-authoritative live slot decodes, and only such slots decode as live entries. -/
def PhysicalCoverage {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) : Prop :=
  ∀ location, Occupied state location ↔ ∃ entry,
    Apeiron.GrowableSearch.entryAt state decode location = some entry

/-- The direct-index function is exactly the generated `WordHome` address and shard-major index. -/
def DirectIndexCorrect {R : Type} (state : WordTable R)
    (direct : LogicalKey → Option Location) : Prop :=
  ∀ key location, direct key = some location ↔ ∃ address index,
    Apeiron.GrowableIdentity.wordHomeInst.address state.home key.1 key.2 = ok address ∧
    Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val =
      some index.val ∧
    location = .flat index

/-- Each live representative agrees with generated addressing at its flat index or spill shard. -/
def AddressCorrect {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) : Prop :=
  ∀ location key value,
    Apeiron.GrowableSearch.entryAt state decode location = some (key, value) →
    match location with
    | .flat index => ∃ address,
        Apeiron.GrowableIdentity.wordHomeInst.address state.home key.1 key.2 = ok address ∧
        Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val
    | .spill shard _ => ∃ address,
        Apeiron.GrowableIdentity.wordHomeInst.address state.home key.1 key.2 = ok address ∧
        table.shard address = ok shard

/-- Flat extent and fixed spill directory have the checked lengths. -/
def SpillDirectoryConsistent {R : Type} (state : WordTable R) : Prop :=
  state.resident.slots.length =
      Apeiron.GrowableGeometry.shardCount * state.resident.lanes.val ∧
  state.shards.length = Apeiron.GrowableGeometry.shardCount

/-- `len` counts the finite logical live set, including live entries whose value equals the default. -/
def LengthCorrect {R : Type} (state : WordTable R) (live : LogicalKey → Prop) : Prop :=
  ∃ keys : List LogicalKey, keys.Nodup ∧
    (∀ key, key ∈ keys ↔ live key) ∧ keys.length = state.len.val

/-- A carrier dictionary reconstructs exactly the generated address of every decoded held key. -/
def CarrierAddressRespecting {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (keyRep : table.keyrep.KeyRep R I64 table.WordHome) :
    Prop :=
  ∀ sigma held key, decode sigma held = some key →
    keyRep.address_held state.home sigma held =
      Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key

/-- Successful evaluation of the generated tuple write exposes the old element and the write
function as a `Vec.set`. -/
private theorem vec_index_mut_usize_ok {α : Type} {v : alloc.vec.Vec α} {i : Usize}
    {x : α} {back : α → alloc.vec.Vec α}
    (h : alloc.vec.Vec.index_mut_usize v i = ok (x, back)) :
    v.val[i.val]? = some x ∧ back = alloc.vec.Vec.set v i := by
  unfold alloc.vec.Vec.index_mut_usize alloc.vec.Vec.index_usize at h
  cases hv : v[i.val]? with
  | none => simp [hv] at h
  | some y =>
      simp only [hv, bind_tc_ok] at h
      have hpair := Result.ok_injective h
      injection hpair with hyx hback
      exact ⟨congrArg some hyx, hback.symm⟩

/-- A successful `publish_live` reads an existing slot at the published index. -/
private theorem publish_live_bound {R V : Type} {before after : table.Region R V}
    {index : Usize} {payload : cell.Sigma × R × V} {place : U64}
    (hpub : table.Region.publish_live before index payload place = ok after) :
    index.val < before.slots.length := by
  by_contra h
  have hnone : before.slots.val[index.val]? = none :=
    List.getElem?_eq_none (by simpa using Nat.le_of_not_gt h)
  unfold table.Region.publish_live at hpub
  dsimp only at hpub
  simp only [alloc.vec.Vec.index_mut_slice_index] at hpub
  cases h0 : alloc.vec.Vec.index_mut_usize before.slots index with
  | vis eff k =>
      rw [h0] at hpub
      simp only [bind_tc_vis] at hpub
      exact (vis_not_ok hpub).elim
  | div =>
      rw [h0] at hpub
      simp only [bind_tc_div] at hpub
      exact (div_not_ok hpub).elim
  | ret r =>
      obtain ⟨x, _⟩ := r
      have hsome := (vec_index_mut_usize_ok h0).1
      rw [hnone] at hsome
      simp at hsome

/-- Successful generated publication writes the generated tag and the full payload at its selected
slot, leaving every other slot of the region unchanged. -/
theorem publish_live_slots_exact {R V : Type}
    (before after : table.Region R V) (index : Usize)
    (payload : cell.Sigma × R × V) (place : U64)
    (hpub : table.Region.publish_live before index payload place = ok after) :
    ∃ (tag : U8),
      table.tag place = ok tag ∧ tag ≠ 0#u8 ∧
      after.slots.val = before.slots.val.set index.val ⟨tag, payload.1, payload.2.1, payload.2.2⟩ := by
  have hbound : index.val < before.slots.length := publish_live_bound hpub
  rcases payload with ⟨s, t, t1⟩
  unfold table.Region.publish_live at hpub
  dsimp only at hpub
  simp only [alloc.vec.Vec.index_mut_slice_index] at hpub
  obtain ⟨p0, hp0, _⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_mut_usize_spec before.slots index hbound)
  rcases p0 with ⟨ss0, back0⟩
  obtain ⟨hget0, hback0⟩ := vec_index_mut_usize_ok hp0
  subst back0
  rw [hp0] at hpub
  simp only [bind_tc_ok, uncurry_apply_pair] at hpub
  have hbound1 : index.val < (before.slots.set index { ss0 with sigma := s }).length := by
    simpa [alloc.vec.Vec.set_length] using hbound
  obtain ⟨p1, hp1, _⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_mut_usize_spec
      (before.slots.set index { ss0 with sigma := s }) index hbound1)
  rcases p1 with ⟨ss1, back1⟩
  obtain ⟨hget1, hback1⟩ := vec_index_mut_usize_ok hp1
  subst back1
  rw [hp1] at hpub
  simp only [bind_tc_ok, uncurry_apply_pair] at hpub
  have hss1' : ss1 = { ss0 with sigma := s } := by
    have hx : (before.slots.set index { ss0 with sigma := s }).val[index.val]? =
        some { ss0 with sigma := s } := by
      rw [alloc.vec.Vec.set_val_eq]
      exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp hget0).1
    exact Option.some.inj (hget1.symm.trans hx)
  have hbound2 : index.val <
      ((before.slots.set index { ss0 with sigma := s }).set index
        { ss1 with held := t }).length := by
    simpa [alloc.vec.Vec.set_length] using hbound
  obtain ⟨p2, hp2, _⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_mut_usize_spec
      ((before.slots.set index { ss0 with sigma := s }).set index
        { ss1 with held := t }) index hbound2)
  rcases p2 with ⟨ss2, back2⟩
  obtain ⟨hget2, hback2⟩ := vec_index_mut_usize_ok hp2
  subst back2
  rw [hp2] at hpub
  simp only [bind_tc_ok, uncurry_apply_pair] at hpub
  have hss2' : ss2 = { ss1 with held := t } := by
    have hx : (((before.slots.set index { ss0 with sigma := s }).set index
        { ss1 with held := t })).val[index.val]? =
        some { ss1 with held := t } := by
      rw [alloc.vec.Vec.set_val_eq]
      exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp hget1).1
    exact Option.some.inj (hget2.symm.trans hx)
  cases htag : table.tag place with
  | vis eff k => simp [htag] at hpub
  | div => simp [htag] at hpub
  | ret tag =>
      rw [htag] at hpub
      simp only [bind_tc_ok] at hpub
      have hbound3 : index.val <
          (((before.slots.set index { ss0 with sigma := s }).set index
            { ss1 with held := t }).set index
            { ss2 with value := t1 }).length := by
        simpa [alloc.vec.Vec.set_length] using hbound
      obtain ⟨p3, hp3, _⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.index_mut_usize_spec
          (((before.slots.set index { ss0 with sigma := s }).set index
            { ss1 with held := t }).set index
            { ss2 with value := t1 }) index hbound3)
      rcases p3 with ⟨ss3, back3⟩
      obtain ⟨hget3, hback3⟩ := vec_index_mut_usize_ok hp3
      subst back3
      rw [hp3] at hpub
      simp only [bind_tc_ok, uncurry_apply_pair] at hpub
      have hss3' : ss3 = { ss2 with value := t1 } := by
        have hx : ((((before.slots.set index { ss0 with sigma := s }).set index
            { ss1 with held := t }).set index
            { ss2 with value := t1 })).val[index.val]? =
            some { ss2 with value := t1 } := by
          rw [alloc.vec.Vec.set_val_eq]
          exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp hget2).1
        exact Option.some.inj (hget3.symm.trans hx)
      cases hlive : before.live + 1#usize with
      | vis eff k => simp [hlive] at hpub
      | div => simp [hlive] at hpub
      | ret live1 =>
          rw [hlive] at hpub
          simp only [bind_tc_ok] at hpub
          have hafter := (Result.ok_injective hpub).symm
          subst after
          refine ⟨tag, rfl, Apeiron.GrowableScan.tag_ne_zero place tag htag, ?_⟩
          subst ss3
          subst ss2
          subst ss1
          simp [alloc.vec.Vec.set_val_eq, List.set_set]

/-- A successful generated publication changes only its selected slot. -/
theorem publish_live_slot_frame {R V : Type}
    (before after : table.Region R V) (index read : Usize)
    (payload : cell.Sigma × R × V) (place : U64)
    (hpub : table.Region.publish_live before index payload place = ok after)
    (hne : read.val ≠ index.val) :
    after.slots.val[read.val]? = before.slots.val[read.val]? := by
  obtain ⟨tag, _, _, hslots⟩ := publish_live_slots_exact before after index payload place hpub
  rw [hslots]
  exact List.getElem?_set_ne (by simpa using hne.symm)

/--
The whole generated table represents one canonical `Apeiron.Table.HMap (Sigma × I64) I64`.
`DirectCover` is the composite direct-empty certificate from `Growable.Search`.
-/
structure RepR {R : Type} (state : WordTable R) (decode : cell.Sigma → R → Option I64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64) : Prop where
  liveSet : LiveSetEqual state decode live
  observations : ObservationCorrect state decode live logical
  unique : UniqueRepresentation state decode
  coverage : PhysicalCoverage state decode
  address : AddressCorrect state decode
  directIndex : DirectIndexCorrect state direct
  directCover : Apeiron.GrowableSearch.DirectCover state decode direct
  directory : SpillDirectoryConsistent state
  length : LengthCorrect state live

private def EmptyRegion {R : Type} (region : table.Region R I64) : Prop :=
  region.slots.val = []

private theorem table_new_loop_empty {R : Type} (shards : alloc.vec.Vec (table.Region R I64))
    (shardAt : Usize) (hAt : shardAt.val ≤ Apeiron.GrowableGeometry.shardCount)
    (hLength : shards.length = shardAt.val)
    (hEmpty : ∀ region ∈ shards.val, EmptyRegion region) :
    table.Table.new_loop shards shardAt ⦃ result =>
      result.length = Apeiron.GrowableGeometry.shardCount ∧
      ∀ region ∈ result.val, EmptyRegion region ⦄ := by
  unfold table.Table.new_loop
  refine loop.spec_decr_nat
    (fun state : alloc.vec.Vec (table.Region R I64) × Usize =>
      Apeiron.GrowableGeometry.shardCount - state.2.val)
    (fun state : alloc.vec.Vec (table.Region R I64) × Usize =>
      state.2.val ≤ Apeiron.GrowableGeometry.shardCount ∧
      state.1.length = state.2.val ∧ ∀ region ∈ state.1.val, EmptyRegion region)
    (fun result => result.length = Apeiron.GrowableGeometry.shardCount ∧
      ∀ region ∈ result.val, EmptyRegion region)
    (fun state => table.Table.new_loop.body state.1 state.2) (shards, shardAt) ?_ ?_
  · rintro ⟨current, cursor⟩ ⟨hcursor, hlength, hempty⟩
    change cursor.val ≤ Apeiron.GrowableGeometry.shardCount at hcursor
    change current.length = cursor.val at hlength
    change ∀ region ∈ current.val, EmptyRegion region at hempty
    unfold table.Table.new_loop.body
    simp only [table.SHARDS, table.partition.SHARDS, table.partition.SHARD_BITS]
    obtain ⟨shifted, hshards, hshifted, _⟩ := WP.spec_imp_exists
      (Aeneas.Std.Usize.ShiftLeft_spec 1#usize 8#u32 (by
        rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num))
    have hshiftedEq : shifted = 256#usize := by
      apply UScalar.eq_of_val_eq
      rw [hshifted]
      rcases System.Platform.numBits_eq with h | h <;>
        norm_num [Usize.size, Usize.numBits, h, ShiftLeft.shiftLeft, Nat.shiftLeft_eq]
    subst shifted
    rw [hshards]
    by_cases hlt : cursor < 256#usize
    · simp only [hlt, ↓reduceIte, table.Region.empty, bind_tc_ok]
      have hltVal : cursor.val < 256 := by simpa using hlt
      have hlengthVal : current.val.length = cursor.val := by exact hlength
      have hbound : current.val.length < Usize.max := by
        rw [hlengthVal]
        have hmax : 256 ≤ Usize.max := by
          rcases Usize.bounds_eq with h | h <;>
            norm_num [h, U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
        omega
      step
      step
      refine ⟨?_, ?_, ?_, ?_⟩
      · simp only [Apeiron.GrowableGeometry.shardCount]
        simp only [Apeiron.GrowableGeometry.shardCount] at hcursor
        rw [shard_at1_post]
        omega
      · simp only [alloc.vec.Vec.length]
        rw [shards1_post, List.length_append, List.length_singleton]
        rw [hlengthVal, shard_at1_post]
      · intro region hregion
        rw [shards1_post] at hregion
        simp only [List.mem_append, List.mem_singleton] at hregion
        rcases hregion with hregion | rfl
        · exact hempty region hregion
        · simp [EmptyRegion, alloc.vec.Vec.new]
      · simp [Apeiron.GrowableGeometry.shardCount]
        simp only [Apeiron.GrowableGeometry.shardCount] at hcursor
        rw [shard_at1_post]
        omega
    ·
      have hnotLt : ¬cursor.val < 256 := by
        intro hval
        apply hlt
        exact hval
      have hcursorEq : cursor.val = 256 := by
        simp only [Apeiron.GrowableGeometry.shardCount] at hcursor
        omega
      simp only [bind_tc_ok]
      rw [if_neg hlt]
      simp only [WP.spec_ok]
      refine ⟨?_, hempty⟩
      rw [hlength, hcursorEq]
      rfl
  · exact ⟨hAt, hLength, hEmpty⟩

private theorem table_new_empty_repR {R : Type}
    (keyRep : table.keyrep.KeyRep R I64 table.WordHome)
    (decode : cell.Sigma → R → Option I64)
    (_haddress : ∀ sigma held key, decode sigma held = some key →
      keyRep.address_held () sigma held =
        Apeiron.GrowableIdentity.wordHomeInst.address () sigma key)
    (default : I64) :
    table.Table.new 16#usize Apeiron.GrowableIdentity.wordHomeInst keyRep default () ⦃ state =>
      RepR state decode (fun _ => none) (fun _ => False) (fun _ => default) ⦄ := by
  unfold table.Table.new
  obtain ⟨shards, hshards, hshardsLength, hshardsEmpty⟩ := WP.spec_imp_exists
    (table_new_loop_empty (alloc.vec.Vec.new (table.Region R I64)) 0#usize
      (by simp [Apeiron.GrowableGeometry.shardCount]) (by rfl)
      (by simp [alloc.vec.Vec.new]))
  rw [hshards]
  simp only [bind_tc_ok, table.Flat.empty, table.TableLocator.new, WP.spec_ok]
  let state : WordTable R :=
    { default := default
      home := ()
      resident := { lanes := 0#usize, slots := alloc.vec.Vec.new (table.FlatSlot R I64) }
      shards := shards
      len := 0#usize
      key := ()
      locator :=
        { profile := none, directory := none
          links := alloc.vec.Vec.new (U32 × table.PayloadLocation), staged := none } }
  change RepR state decode (fun _ => none) (fun _ => False) (fun _ => default)
  have regionEmpty {shard : Usize} {region : table.Region R I64}
      (hregion : state.shards.val[shard.val]? = some region) : region.slots.val = [] := by
    apply hshardsEmpty region
    exact List.mem_of_getElem? hregion
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro key
    constructor
    · intro h; contradiction
    · rintro ⟨location, value, hentry⟩
      cases location with
      | flat index => simp [Apeiron.GrowableSearch.entryAt, state, alloc.vec.Vec.new] at hentry
      | spill shard index =>
          cases hregion : state.shards.val[shard.val]? with
          | none => simp [Apeiron.GrowableSearch.entryAt, hregion] at hentry
          | some region =>
              have hempty := regionEmpty hregion
              simp [Apeiron.GrowableSearch.entryAt, hregion, hempty] at hentry
  · constructor
    · intro location key value hentry
      cases location with
      | flat index => simp [Apeiron.GrowableSearch.entryAt, state, alloc.vec.Vec.new] at hentry
      | spill shard index =>
          simp only [Apeiron.GrowableSearch.entryAt] at hentry
          cases hregion : state.shards.val[shard.val]? with
          | none => simp [hregion] at hentry
          | some region =>
              have hempty := regionEmpty hregion
              simp [hregion, hempty] at hentry
    · simp [Apeiron.Table.get, state]
  · intro location₁ location₂ key value₁ value₂ hentry
    cases location₁ with
    | flat index => simp [Apeiron.GrowableSearch.entryAt, state, alloc.vec.Vec.new] at hentry
    | spill shard index =>
        simp only [Apeiron.GrowableSearch.entryAt] at hentry
        cases hregion : state.shards.val[shard.val]? with
        | none => simp [hregion] at hentry
        | some region =>
            have hempty := regionEmpty hregion
            simp [hregion, hempty] at hentry
  · intro location
    cases location with
    | flat index => simp [Occupied, Apeiron.GrowableSearch.entryAt, state, alloc.vec.Vec.new]
    | spill shard index =>
        cases hregion : state.shards.val[shard.val]? with
        | none => simp [Occupied, Apeiron.GrowableSearch.entryAt, hregion]
        | some region =>
            have hempty := regionEmpty hregion
            simp [Occupied, Apeiron.GrowableSearch.entryAt, hregion, hempty]
  · intro location key value hentry
    cases location with
    | flat index => simp [Apeiron.GrowableSearch.entryAt, state, alloc.vec.Vec.new] at hentry
    | spill shard index =>
        simp only [Apeiron.GrowableSearch.entryAt] at hentry
        cases hregion : state.shards.val[shard.val]? with
        | none => simp [hregion] at hentry
        | some region =>
            have hempty := regionEmpty hregion
            simp [hregion, hempty] at hentry
  · intro key location
    constructor
    · intro h; simp at h
    · rintro ⟨address, index, _haddress, hindex, rfl⟩
      simp only [Apeiron.GrowableGeometry.preparedFlatIndex, state] at hindex
      cases hword : address.direct_word <;> simp [hword] at hindex
  · intro query location hdirect
    simp at hdirect
  · refine ⟨?_, hshardsLength⟩
    · simp [state, Apeiron.GrowableGeometry.shardCount, alloc.vec.Vec.new]
  · exact ⟨[], by simp, by simp [state]⟩

/-- Generated Stored-carrier construction establishes the full empty canonical-map relation. -/
theorem word_stored_table_new_repR (default : I64) :
    table.Table.new 16#usize Apeiron.GrowableIdentity.wordHomeInst
      Apeiron.GrowableIdentity.storedKeyRep default () ⦃ state =>
      RepR state Apeiron.GrowableSearch.decodeStoredOption (fun _ => none) (fun _ => False)
        (fun _ => default) ⦄ := by
  apply table_new_empty_repR
  intro sigma held key hdecode
  simp [Apeiron.GrowableSearch.decodeStoredOption] at hdecode
  subst key
  exact Apeiron.GrowableIdentity.stored_address_held_agrees () sigma held

/-- Generated Regen-carrier construction establishes the full empty canonical-map relation. -/
theorem word_regen_table_new_repR (default : I64) :
    table.Table.new 16#usize Apeiron.GrowableIdentity.wordHomeInst
      Apeiron.GrowableIdentity.regenKeyRep default () ⦃ state =>
      RepR state Apeiron.GrowableSearch.decodeRegenOption (fun _ => none) (fun _ => False)
        (fun _ => default) ⦄ := by
  apply table_new_empty_repR
  intro sigma held key hdecode
  unfold Apeiron.GrowableSearch.decodeRegenOption at hdecode
  cases hresult : Apeiron.GrowableIdentity.decodeRegen sigma held with
  | vis eff k => simp [hresult] at hdecode
  | div => simp [hresult] at hdecode
  | ret decoded =>
      simp [hresult] at hdecode
      subst decoded
      change (do
        let decoded ← Apeiron.GrowableIdentity.decodeRegen sigma held
        Apeiron.GrowableIdentity.wordHomeInst.address () sigma decoded) = _
      rw [hresult]
      simp

/-- Materialized relocation storage represents the old map before the pending write is committed. -/
def ReservedRepR {R : Type} (state : WordTable R) (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (direct : LogicalKey → Option Location)
    (live : LogicalKey → Prop) (logical : Apeiron.Table.HMap LogicalKey I64) : Prop :=
  RepR (Apeiron.GrowableRefinement.rebuiltView state reserved) decode direct live logical

/--
The pending commit links the materialized old map to the canonical new-first successor.  The run
equation is explicit; this definition does not assert that an unextracted decision is correct.
-/
structure PendingCommittedR {R : Type} (state : WordTable R)
    (reserved : table.RebuildReserved R I64) (committed : WordTable R)
    (decode : cell.Sigma → R → Option I64)
    (directBefore directAfter : LogicalKey → Option Location)
    (liveBefore liveAfter : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64) (merge : I64 → I64 → I64)
    (pending : LogicalKey) (value : I64) (commitRun : Result (WordTable R)) : Prop where
  run : commitRun = ok committed
  reservedRep : ReservedRepR state reserved decode directBefore liveBefore logical
  committedRep : RepR committed decode directAfter liveAfter
    (Apeiron.Table.insertWith merge pending value logical)

/-- Whole-table live-set projection used by rebuild preservation. -/
theorem repR_liveSet {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) : LiveSetEqual state decode live := h.liveSet

/-- Whole-table decoded uniqueness projection used by rebuild preservation. -/
theorem repR_unique {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) : UniqueRepresentation state decode := h.unique

/-- Whole-table occupancy coverage projection used by rebuild preservation. -/
theorem repR_coverage {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) : PhysicalCoverage state decode := h.coverage

/-- Composite `DirectCover` projection used by search after relocation. -/
theorem repR_directCover {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) :
    Apeiron.GrowableSearch.DirectCover state decode direct := h.directCover

/-- Spill-directory shape projection used to frame disjoint relocation writes. -/
theorem repR_directory {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) : SpillDirectoryConsistent state := h.directory

/-- Construct the reserved whole-table relation from certificates about the target rebuilt geometry. -/
theorem reservedRepR_of_components {R state reserved decode direct live logical}
    (hliveSet : LiveSetEqual (Apeiron.GrowableRefinement.rebuiltView state reserved) decode live)
    (hobservations : ObservationCorrect
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode live logical)
    (hunique : UniqueRepresentation
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode)
    (hcoverage : PhysicalCoverage
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode)
    (haddress : AddressCorrect
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode)
    (hdirectIndex : DirectIndexCorrect
      (Apeiron.GrowableRefinement.rebuiltView state reserved) direct)
    (hdirectCover : Apeiron.GrowableSearch.DirectCover
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode direct)
    (hdirectory : SpillDirectoryConsistent
      (Apeiron.GrowableRefinement.rebuiltView state reserved))
    (hlength : LengthCorrect
      (Apeiron.GrowableRefinement.rebuiltView state reserved) live)
    (keyRep : table.keyrep.KeyRep R I64 table.WordHome)
    (_hcarrierAddress : CarrierAddressRespecting
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode keyRep) :
    @ReservedRepR R state reserved decode direct live logical :=
  ⟨hliveSet, hobservations, hunique, hcoverage, haddress, hdirectIndex, hdirectCover, hdirectory,
    hlength⟩

/-- The rebuilt target's live set is the one named by its reserved relation. -/
theorem reservedRepR_liveSet {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    LiveSetEqual (Apeiron.GrowableRefinement.rebuiltView state reserved) decode live := h.liveSet

/-- Decoded keys are unique in the rebuilt target geometry. -/
theorem reservedRepR_unique {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    UniqueRepresentation (Apeiron.GrowableRefinement.rebuiltView state reserved) decode := h.unique

/-- Occupancy-authoritative slots exactly cover the rebuilt target's decoded entries. -/
theorem reservedRepR_coverage {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    PhysicalCoverage (Apeiron.GrowableRefinement.rebuiltView state reserved) decode := h.coverage

/-- Direct-empty absence uses the direct relation for the rebuilt target geometry. -/
theorem reservedRepR_directCover {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    Apeiron.GrowableSearch.DirectCover
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode direct := h.directCover

/-- The rebuilt target has the checked flat extent and fixed spill directory. -/
theorem reservedRepR_directory {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    SpillDirectoryConsistent (Apeiron.GrowableRefinement.rebuiltView state reserved) := h.directory

/-- The rebuilt target's committed length counts its declared live set. -/
theorem reservedRepR_length {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    LengthCorrect (Apeiron.GrowableRefinement.rebuiltView state reserved) live := h.length

/-- Reserved storage is precisely the whole-table relation on its pre-pending table view. -/
theorem reservedRepR_iff_rebuiltView {R state reserved decode direct live logical} :
    @ReservedRepR R state reserved decode direct live logical ↔
      RepR (Apeiron.GrowableRefinement.rebuiltView state reserved) decode direct live logical :=
  Iff.rfl

/-- Project the old-map relation from the pending/committed boundary. -/
theorem pendingCommitted_reserved {R state reserved committed decode directBefore directAfter
    liveBefore liveAfter logical merge pending value commitRun}
    (h : @PendingCommittedR R state reserved committed decode directBefore directAfter liveBefore
      liveAfter logical merge pending value commitRun) :
    ReservedRepR state reserved decode directBefore liveBefore logical := h.reservedRep

/-- Project the canonical `insertWith` successor from the pending/committed boundary. -/
theorem pendingCommitted_committed {R state reserved committed decode directBefore directAfter
    liveBefore liveAfter logical merge pending value commitRun}
    (h : @PendingCommittedR R state reserved committed decode directBefore directAfter liveBefore
      liveAfter logical merge pending value commitRun) :
    RepR committed decode directAfter liveAfter
      (Apeiron.Table.insertWith merge pending value logical) := h.committedRep

/-- A rebuild source and the search location with the same generated indexes observe the same slot. -/
def sourceLocation : table.RebuildSource → Location
  | .Resident index => .flat index
  | .Spill shard index => .spill shard index

theorem sourceObservation_eq_entryAt {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (source : table.RebuildSource) :
    Apeiron.GrowableRelocation.sourceObservation state decode source =
      Apeiron.GrowableSearch.entryAt state decode (sourceLocation source) := by
  cases source with
  | Resident index =>
      cases hslot : state.resident.slots.val[index.val]? with
      | none => simp [Apeiron.GrowableRelocation.sourceObservation,
          Apeiron.GrowableSearch.entryAt, sourceLocation, hslot]
      | some slot =>
          cases hlive : slot.live <;>
            cases hdecode : decode slot.sigma slot.held <;>
            simp [Apeiron.GrowableRelocation.sourceObservation,
              Apeiron.GrowableSearch.entryAt, sourceLocation, hslot, hlive, hdecode]
  | Spill shard index =>
      cases hregion : state.shards.val[shard.val]? with
      | none => simp [Apeiron.GrowableRelocation.sourceObservation,
          Apeiron.GrowableSearch.entryAt, sourceLocation, hregion]
      | some region =>
          cases hslot : region.slots.val[index.val]? with
          | none => simp [Apeiron.GrowableRelocation.sourceObservation,
              Apeiron.GrowableSearch.entryAt, sourceLocation, hregion, hslot]
          | some slot =>
              rcases slot with ⟨tag, sigma, held, value⟩
              by_cases hzero : tag = 0#u8 <;>
                cases hdecode : decode sigma held <;>
                simp [Apeiron.GrowableRelocation.sourceObservation,
                  Apeiron.GrowableSearch.entryAt, sourceLocation, hregion, hslot,
                  hzero, hdecode]

/-- Whole-table representation agrees with the existing selected-source `RepR`. -/
theorem repR_singleSource {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) (source : table.RebuildSource)
    (entry : LogicalEntry)
    (hentry : Apeiron.GrowableRelocation.sourceObservation state decode source = some entry) :
    Apeiron.GrowableRelocation.RepR state decode source logical := by
  refine ⟨entry, hentry, ?_⟩
  have hobserve := h.observations.1 (sourceLocation source) entry.1 entry.2
  apply hobserve
  rw [← sourceObservation_eq_entryAt]
  exact hentry

/-- A reserved destination and the rebuilt table view observe the same materialized slot. -/
def destinationLocation : Apeiron.GrowableRelocation.Destination → Location
  | .flat index => .flat index
  | .spill shard index => .spill shard index

theorem reservedObservation_eq_entryAt {R : Type} (state : WordTable R)
    (reserved : table.RebuildReserved R I64) (decode : cell.Sigma → R → Option I64)
    (destination : Apeiron.GrowableRelocation.Destination) :
    Apeiron.GrowableRelocation.reservedObservation reserved decode destination =
      Apeiron.GrowableSearch.entryAt (Apeiron.GrowableRefinement.rebuiltView state reserved) decode
        (destinationLocation destination) := by
  cases destination with
  | flat index =>
      cases hslot : reserved.resident.slots.val[index.val]? with
      | none => simp [Apeiron.GrowableRelocation.reservedObservation,
          Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
          destinationLocation, hslot]
      | some slot =>
          cases hlive : slot.live <;>
            cases hdecode : decode slot.sigma slot.held <;>
            simp [Apeiron.GrowableRelocation.reservedObservation,
              Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
              destinationLocation, hslot, hlive, hdecode]
  | spill shard index =>
      cases hregion : reserved.regions.val[shard.val]? with
      | none => simp [Apeiron.GrowableRelocation.reservedObservation,
          Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
          destinationLocation, hregion]
      | some region =>
          cases hslot : region.slots.val[index.val]? with
          | none => simp [Apeiron.GrowableRelocation.reservedObservation,
              Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
              destinationLocation, hregion, hslot]
          | some slot =>
              rcases slot with ⟨tag, sigma, held, value⟩
              by_cases hzero : tag = 0#u8 <;>
                cases hdecode : decode sigma held <;>
                simp [Apeiron.GrowableRelocation.reservedObservation,
                  Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
                  destinationLocation, hregion, hslot, hzero, hdecode]

/-- Whole reserved representation agrees with the existing selected-destination relation. -/
theorem reservedRepR_singleDestination {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical)
    (destination : Apeiron.GrowableRelocation.Destination) (entry : LogicalEntry)
    (hentry : Apeiron.GrowableRelocation.reservedObservation reserved decode destination =
      some entry) :
    Apeiron.GrowableRelocation.ReservedRepR reserved decode destination logical := by
  refine ⟨entry, hentry, ?_⟩
  have hobserve := h.observations.1 (destinationLocation destination) entry.1 entry.2
  apply hobserve
  rw [← reservedObservation_eq_entryAt]
  exact hentry

/-- The successful generated vacant-spill apply returns exactly the shard
vector produced by the selected `Region.publish_live` call. The input address
and shard are not certified here as query routing: they must be source-linked
before this equality is used as a whole-table representation theorem. -/
theorem word_stored_vacant_spill_return_shards
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress)
    (shard index : Usize)
    (region published : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (nextLocator : table.TableLocator)
    (hlocator : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      state.locator state.home sigma key (.Spill shard index) = ok nextLocator)
    (hpublish : table.Region.publish_live region index
      (sigma, { key := key }, core.num.I64.wrapping_add value state.default)
      address.place = ok published)
    (hlen : state.len.val + 1 < Usize.max) :
    ∃ next : WordTable Apeiron.GrowableIdentity.StoredI64,
      word_stored_apply_insert state
        { «class» := .Vacant, location := .Spill shard index }
        sigma key value address = ok (next, none) ∧
      next.home = state.home ∧ next.shards = state.shards.set shard published := by
  have hshardBound : shard.val < state.shards.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  unfold word_stored_apply_insert table.Table.apply_insert
  simp_all [table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize]
  have hlenCap : state.len.val + (1#usize).val ≤ Usize.max := by
    norm_num at *
    omega
  obtain ⟨nextLength, hnextLength, _⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := state.len) (y := 1#usize) hlenCap)
  rw [hnextLength, bind_tc_ok]
  exact ⟨_, rfl, rfl⟩

/-- The generated vacant-flat apply leaves both the home dictionary and the
entire spill directory untouched in its returned table. -/
theorem word_stored_vacant_flat_return_frame
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress)
    (index : Usize) (old : table.FlatSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some old)
    (nextLocator : table.TableLocator)
    (hlocator : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      state.locator state.home sigma key (.Resident index) = ok nextLocator)
    (hlen : state.len.val + 1 < Usize.max) :
    ∃ next : WordTable Apeiron.GrowableIdentity.StoredI64,
      word_stored_apply_insert state
        { «class» := .Vacant, location := .Flat index }
        sigma key value address = ok (next, none) ∧
      next.home = state.home ∧ next.shards = state.shards := by
  have hbound : index.val < state.resident.slots.length :=
    (List.getElem?_eq_some_iff.mp hslot).1
  unfold word_stored_apply_insert table.Table.apply_insert
  simp_all [table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize]
  have hlenCap : state.len.val + (1#usize).val ≤ Usize.max := by
    norm_num at *
    omega
  obtain ⟨nextLength, hnextLength, _⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := state.len) (y := 1#usize) hlenCap)
  rw [hnextLength, bind_tc_ok]
  exact ⟨_, rfl, rfl⟩

/-!
## Rebuild relocation step equations and destination-traversal induction

The equations below pin the *actual generated* `word_stored_relocate_entry` post-state at its
planned spill or flat destination.  They are the summary-aware write facts for the destination
traversal: spill publication commits the generated tag of the computed address and the region
summary word, so a summary-blind symbolic write is not this body's post-state.
-/

/-- Exact successful post-state of the generated stored-carrier *spill* relocation step.  The
source table keeps every payload field and changes only its locator; the reserved storage replaces
exactly the planned shard with the `publish_live` result at the located vacancy.  The computed
address and the routing to the planned shard remain explicit input-side obligations. -/
theorem word_stored_relocate_entry_spill_run
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (entryAt shard index : Usize)
    (entry : table.RebuildPlanEntry)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (address : table.TableAddress)
    (region published : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (nextLocator : table.TableLocator)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hsource : Apeiron.GrowableRelocation.sourcePayload state entry.source =
      some (sigma, held, value))
    (hflat : entry.flat = none)
    (hspill : entry.spill = shard)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma held.key =
      ok address)
    (hregion : current.regions.val[shard.val]? = some region)
    (hvacant : table.probe.spill_locate_held Apeiron.GrowableIdentity.storedKeyRep region sigma
      address { key := held.key } 16#usize = ok (.Vacant index))
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
      Apeiron.GrowableIdentity.storedKeyRep state.locator state.home sigma { key := held.key }
      (Apeiron.GrowableRelocation.sourceLocation entry.source)
      (Apeiron.GrowableRelocation.destinationLocation (.spill shard index)) = ok nextLocator)
    (hpublish : table.Region.publish_live region index (sigma, { key := held.key }, value)
      address.place = ok published) :
    word_stored_relocate_entry state plan current entryAt =
      ok ({ state with locator := nextLocator },
        { current with regions := current.regions.set shard published }) := by
  have hshardBound : shard.val < current.regions.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  have haddress' :
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.address_held
        Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma state.home sigma
        { key := held.key } = ok address :=
    haddress
  have hvacant' :
      table.probe.spill_locate_held
        (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
          Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
        region sigma address { key := held.key } 16#usize = ok (.Vacant index) := by
    simpa [Apeiron.GrowableIdentity.storedKeyRep] using hvacant
  cases hsrc : entry.source with
  | Resident sourceIndex =>
      simp only [Apeiron.GrowableRelocation.sourcePayload, hsrc] at hsource
      cases hslot : state.resident.slots.val[sourceIndex.val]? with
      | none => simp [hslot] at hsource
      | some sourceSlot =>
          simp only [hslot] at hsource
          by_cases hlive : sourceSlot.live = true
          · simp only [hlive, ↓reduceIte, Option.some.injEq] at hsource
            obtain ⟨hsigma, htail⟩ := Prod.mk.inj hsource
            obtain ⟨hheld, hvalue⟩ := Prod.mk.inj htail
            subst hsigma
            subst hheld
            subst hvalue
            have hretarget' :
                table.TableLocator.retarget_held core.clone.CloneI64
                  table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
                  (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
                    Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
                    table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
                  state.locator state.home sourceSlot.sigma { key := sourceSlot.held.key }
                  (.Resident sourceIndex) (.Spill shard index) = ok nextLocator := by
              rw [hsrc] at hretarget
              simpa [Apeiron.GrowableIdentity.storedKeyRep, Apeiron.GrowableIdentity.wordHomeInst,
                Apeiron.GrowableRelocation.sourceLocation,
                Apeiron.GrowableRelocation.destinationLocation, hspill] using hretarget
            unfold word_stored_relocate_entry table.Table.relocate_entry
            simp [hentry, hsrc, hslot, hflat, hspill, haddress', hvacant', hretarget',
              hregion, hpublish, table.keyrep.Stored.Insts.CoreCloneClone.clone,
              alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
              alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]
          · simp [hlive] at hsource
  | Spill sourceShard sourceIndex =>
      simp only [Apeiron.GrowableRelocation.sourcePayload, hsrc] at hsource
      cases hsourceRegion : state.shards.val[sourceShard.val]? with
      | none => simp [hsourceRegion] at hsource
      | some sourceRegion =>
          cases hsourceSlot : sourceRegion.slots.val[sourceIndex.val]? with
          | none => simp [hsourceRegion, hsourceSlot] at hsource
          | some sourceSlot =>
              rcases sourceSlot with ⟨sTag, sSigma, sHeld, sValue⟩
              by_cases hzero : sTag = 0#u8
              · simp [hsourceRegion, hsourceSlot, hzero] at hsource
              · simp only [hsourceRegion, hsourceSlot, hzero, ↓reduceIte,
                  Option.some.injEq] at hsource
                obtain ⟨hsigma, htail⟩ := Prod.mk.inj hsource
                obtain ⟨hheld, hvalue⟩ := Prod.mk.inj htail
                subst hsigma
                subst hheld
                subst hvalue
                have hretarget' :
                    table.TableLocator.retarget_held core.clone.CloneI64
                      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
                      (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
                        Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
                        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
                      state.locator state.home sSigma { key := sHeld.key }
                      (.Spill sourceShard sourceIndex) (.Spill shard index) = ok nextLocator := by
                  rw [hsrc] at hretarget
                  simpa [Apeiron.GrowableIdentity.storedKeyRep,
                    Apeiron.GrowableIdentity.wordHomeInst,
                    Apeiron.GrowableRelocation.sourceLocation,
                    Apeiron.GrowableRelocation.destinationLocation, hspill] using hretarget
                unfold word_stored_relocate_entry table.Table.relocate_entry
                simp [hentry, hsrc, hsourceRegion, hsourceSlot, hflat, hspill,
                  haddress', hvacant', hretarget', hregion, hpublish,
                  table.keyrep.Stored.Insts.CoreCloneClone.clone,
                  alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
                  alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]

/-- Exact successful post-state of the generated stored-carrier *flat* relocation step.  The
reserved storage replaces exactly the planned resident slot; no spill region is touched. -/
theorem word_stored_relocate_entry_flat_run
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (entryAt index : Usize)
    (entry : table.RebuildPlanEntry)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (nextLocator : table.TableLocator)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hsource : Apeiron.GrowableRelocation.sourcePayload state entry.source =
      some (sigma, held, value))
    (hflat : entry.flat = some index)
    (old : table.FlatSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (hdst : current.resident.slots.val[index.val]? = some old)
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
      Apeiron.GrowableIdentity.storedKeyRep state.locator state.home sigma { key := held.key }
      (Apeiron.GrowableRelocation.sourceLocation entry.source)
      (Apeiron.GrowableRelocation.destinationLocation (.flat index)) = ok nextLocator) :
    word_stored_relocate_entry state plan current entryAt =
      ok ({ state with locator := nextLocator },
        { current with resident := { current.resident with
            slots := current.resident.slots.set index
              { live := true, sigma := sigma, held := { key := held.key }, value := value } } }) := by
  cases hsrc : entry.source with
  | Resident sourceIndex =>
      simp only [Apeiron.GrowableRelocation.sourcePayload, hsrc] at hsource
      cases hslot : state.resident.slots.val[sourceIndex.val]? with
      | none => simp [hslot] at hsource
      | some sourceSlot =>
          simp only [hslot] at hsource
          by_cases hlive : sourceSlot.live = true
          · simp only [hlive, ↓reduceIte, Option.some.injEq] at hsource
            obtain ⟨hsigma, htail⟩ := Prod.mk.inj hsource
            obtain ⟨hheld, hvalue⟩ := Prod.mk.inj htail
            subst hsigma
            subst hheld
            subst hvalue
            have hretarget' :
                table.TableLocator.retarget_held core.clone.CloneI64
                  table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
                  (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
                    Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
                    table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
                  state.locator state.home sourceSlot.sigma { key := sourceSlot.held.key }
                  (.Resident sourceIndex) (.Resident index) = ok nextLocator := by
              rw [hsrc] at hretarget
              simpa [Apeiron.GrowableIdentity.storedKeyRep, Apeiron.GrowableIdentity.wordHomeInst,
                Apeiron.GrowableRelocation.sourceLocation,
                Apeiron.GrowableRelocation.destinationLocation] using hretarget
            unfold word_stored_relocate_entry table.Table.relocate_entry
            simp [hentry, hsrc, hslot, hflat, hdst, hretarget',
              table.keyrep.Stored.Insts.CoreCloneClone.clone,
              alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
              alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]
          · simp [hlive] at hsource
  | Spill sourceShard sourceIndex =>
      simp only [Apeiron.GrowableRelocation.sourcePayload, hsrc] at hsource
      cases hsourceRegion : state.shards.val[sourceShard.val]? with
      | none => simp [hsourceRegion] at hsource
      | some sourceRegion =>
          cases hsourceSlot : sourceRegion.slots.val[sourceIndex.val]? with
          | none => simp [hsourceRegion, hsourceSlot] at hsource
          | some sourceSlot =>
              rcases sourceSlot with ⟨sTag, sSigma, sHeld, sValue⟩
              by_cases hzero : sTag = 0#u8
              · simp [hsourceRegion, hsourceSlot, hzero] at hsource
              · simp only [hsourceRegion, hsourceSlot, hzero, ↓reduceIte,
                  Option.some.injEq] at hsource
                obtain ⟨hsigma, htail⟩ := Prod.mk.inj hsource
                obtain ⟨hheld, hvalue⟩ := Prod.mk.inj htail
                subst hsigma
                subst hheld
                subst hvalue
                have hretarget' :
                    table.TableLocator.retarget_held core.clone.CloneI64
                      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
                      (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
                        Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
                        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
                      state.locator state.home sSigma { key := sHeld.key }
                      (.Spill sourceShard sourceIndex) (.Resident index) = ok nextLocator := by
                  rw [hsrc] at hretarget
                  simpa [Apeiron.GrowableIdentity.storedKeyRep,
                    Apeiron.GrowableIdentity.wordHomeInst,
                    Apeiron.GrowableRelocation.sourceLocation,
                    Apeiron.GrowableRelocation.destinationLocation] using hretarget
                unfold word_stored_relocate_entry table.Table.relocate_entry
                simp [hentry, hsrc, hsourceRegion, hsourceSlot, hflat, hdst, hretarget',
                  table.keyrep.Stored.Insts.CoreCloneClone.clone,
                  alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
                  alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]

/-- Per-step relocation: the generated spill relocation/materialize step publishes its selected
destination slot with the generated tag of the computed address for the held key, keeps the
computed address routed to the planned shard, and leaves every other destination slot unchanged.
The routing equation `table.shard address = ok shard` is an explicit input-side premise carried
into the conclusion; the body itself writes at the plan's `entry.spill` shard. -/
theorem word_stored_relocate_entry_spill_publishes_tag_address
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (entryAt shard index : Usize)
    (entry : table.RebuildPlanEntry)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (address : table.TableAddress)
    (region published : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (nextLocator : table.TableLocator)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hsource : Apeiron.GrowableRelocation.sourcePayload state entry.source =
      some (sigma, held, value))
    (hflat : entry.flat = none)
    (hspill : entry.spill = shard)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma held.key =
      ok address)
    (hregion : current.regions.val[shard.val]? = some region)
    (hvacant : table.probe.spill_locate_held Apeiron.GrowableIdentity.storedKeyRep region sigma
      address { key := held.key } 16#usize = ok (.Vacant index))
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
      Apeiron.GrowableIdentity.storedKeyRep state.locator state.home sigma { key := held.key }
      (Apeiron.GrowableRelocation.sourceLocation entry.source)
      (Apeiron.GrowableRelocation.destinationLocation (.spill shard index)) = ok nextLocator)
    (hpublish : table.Region.publish_live region index (sigma, { key := held.key }, value)
      address.place = ok published)
    (hshard : table.shard address = ok shard) :
    word_stored_relocate_entry state plan current entryAt ⦃ result =>
      result.2.regions.val[shard.val]? = some published ∧
      (∃ tag : U8, published.slots.val[index.val]? =
          some ⟨tag, sigma, { key := held.key }, value⟩ ∧
        table.tag address.place = ok tag ∧ tag ≠ 0#u8) ∧
      table.shard address = ok shard ∧
      Apeiron.GrowableIdentity.wordHomeInst.address result.1.home sigma held.key = ok address ∧
      (∀ other : Usize, other.val ≠ shard.val →
        result.2.regions.val[other.val]? = current.regions.val[other.val]?) ∧
      (∀ other : Usize, other.val ≠ index.val →
        published.slots.val[other.val]? = region.slots.val[other.val]?) ⦄ := by
  rw [word_stored_relocate_entry_spill_run state plan current entryAt shard index entry sigma held
    value address region published nextLocator hentry hsource hflat hspill haddress hregion hvacant
    hretarget hpublish, WP.spec_ok]
  have hshardBound : shard.val < current.regions.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  have hbound : index.val < region.slots.val.length := by
    simpa [alloc.vec.Vec.length] using publish_live_bound hpublish
  obtain ⟨tag, htagGen, htagNz, hslots⟩ := publish_live_slots_exact region published index
    (sigma, { key := held.key }, value) address.place hpublish
  have htagAt : published.slots.val[index.val]? =
      some ⟨tag, sigma, { key := held.key }, value⟩ := by
    rw [hslots]
    exact List.getElem?_set_self hbound
  refine ⟨?_, ⟨tag, htagAt, htagGen, htagNz⟩, hshard, haddress, ?_, ?_⟩
  · simp [alloc.vec.Vec.set_val_eq, List.getElem?_set_self hshardBound]
  · intro other hne
    simp [alloc.vec.Vec.set_val_eq, hne]
  · intro other hne
    exact publish_live_slot_frame region published index other
      (sigma, { key := held.key }, value) address.place hpublish hne

/-- The published destination reads back through the rebuilt view's `entryAt` at the reserved
destination location; the identity of the destination read is the landed reserved-observation
bridge (`reservedObservation_eq_entryAt` along `destinationLocation`). -/
theorem word_stored_relocate_entry_spill_entryAt
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (entryAt shard index : Usize)
    (entry : table.RebuildPlanEntry)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (address : table.TableAddress)
    (region published : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (nextLocator : table.TableLocator)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hsource : Apeiron.GrowableRelocation.sourcePayload state entry.source =
      some (sigma, held, value))
    (hflat : entry.flat = none)
    (hspill : entry.spill = shard)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma held.key =
      ok address)
    (hregion : current.regions.val[shard.val]? = some region)
    (hvacant : table.probe.spill_locate_held Apeiron.GrowableIdentity.storedKeyRep region sigma
      address { key := held.key } 16#usize = ok (.Vacant index))
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
      Apeiron.GrowableIdentity.storedKeyRep state.locator state.home sigma { key := held.key }
      (Apeiron.GrowableRelocation.sourceLocation entry.source)
      (Apeiron.GrowableRelocation.destinationLocation (.spill shard index)) = ok nextLocator)
    (hpublish : table.Region.publish_live region index (sigma, { key := held.key }, value)
      address.place = ok published)
    (_hshard : table.shard address = ok shard) :
    word_stored_relocate_entry state plan current entryAt ⦃ result =>
      Apeiron.GrowableRelocation.reservedObservation result.2
          Apeiron.GrowableSearch.decodeStoredOption (.spill shard index) =
        some ((sigma, held.key), value) ∧
      Apeiron.GrowableSearch.entryAt (Apeiron.GrowableRefinement.rebuiltView result.1 result.2)
          Apeiron.GrowableSearch.decodeStoredOption
          (destinationLocation (.spill shard index)) =
        some ((sigma, held.key), value) ⦄ := by
  rw [word_stored_relocate_entry_spill_run state plan current entryAt shard index entry sigma held
    value address region published nextLocator hentry hsource hflat hspill haddress hregion hvacant
    hretarget hpublish, WP.spec_ok]
  have hshardBound : shard.val < current.regions.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  have hbound : index.val < region.slots.val.length := by
    simpa [alloc.vec.Vec.length] using publish_live_bound hpublish
  obtain ⟨tag, _, htagNz, hslots⟩ := publish_live_slots_exact region published index
    (sigma, { key := held.key }, value) address.place hpublish
  have hslotAt : published.slots.val[index.val]? =
      some ⟨tag, sigma, { key := held.key }, value⟩ := by
    rw [hslots]
    exact List.getElem?_set_self hbound
  have hread :
      Apeiron.GrowableRelocation.reservedObservation
          { current with regions := current.regions.set shard published }
          Apeiron.GrowableSearch.decodeStoredOption (.spill shard index) =
        some ((sigma, held.key), value) := by
    have h := Apeiron.GrowableRelocation.reservedObservation_writeSpill_same current
      Apeiron.GrowableSearch.decodeStoredOption shard index published sigma { key := held.key }
      value tag hshardBound hslotAt htagNz
    simpa [Apeiron.GrowableRelocation.writeSpill,
      Apeiron.GrowableSearch.decodeStoredOption] using h
  refine ⟨hread, ?_⟩
  rw [← reservedObservation_eq_entryAt]
  exact hread

/-- When the rebuilt target carries the whole reserved representation, the relocated destination
is the selected-destination canonical representation of the same logical entry, by the landed
`reservedRepR_singleDestination` bridge. -/
theorem reservedRepR_singleDestination_of_relocated
    (nextState : WordTable Apeiron.GrowableIdentity.StoredI64)
    (nextStorage : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (shard index : Usize)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (hwhole : @ReservedRepR Apeiron.GrowableIdentity.StoredI64 nextState nextStorage
      Apeiron.GrowableSearch.decodeStoredOption direct live logical)
    (hread : Apeiron.GrowableRelocation.reservedObservation nextStorage
      Apeiron.GrowableSearch.decodeStoredOption (.spill shard index) =
      some ((sigma, held.key), value)) :
    Apeiron.GrowableRelocation.ReservedRepR nextStorage
        Apeiron.GrowableSearch.decodeStoredOption (.spill shard index) logical ∧
      logical (sigma, held.key) = value := by
  have hsingle := reservedRepR_singleDestination hwhole (.spill shard index)
    ((sigma, held.key), value) hread
  refine ⟨hsingle, ?_⟩
  obtain ⟨entry, hobs, hlog⟩ := hsingle
  have heq : entry = ((sigma, held.key), value) := Option.some.inj (hobs.symm.trans hread)
  cases heq
  exact hlog

/-- Source-side composition of the landed reserved helpers: a whole-table `RepR` and the selected
source observation (`repR_singleSource` along `sourceObservation_eq_entryAt`/`sourceLocation`)
agree that the source-selected payload represents its canonical value. -/
theorem repR_singleSource_logical {R : Type} {state : WordTable R}
    {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : @RepR R state decode direct live logical)
    (source : table.RebuildSource) (entry : Apeiron.GrowableRelocation.LogicalEntry)
    (hobs : Apeiron.GrowableRelocation.sourceObservation state decode source = some entry) :
    logical entry.1 = entry.2 ∧
      Apeiron.GrowableRelocation.sourceObservation state decode source =
        Apeiron.GrowableSearch.entryAt state decode (sourceLocation source) := by
  obtain ⟨observed, hobserved, hlogical⟩ := repR_singleSource hrep source entry hobs
  have heq : observed = entry := Option.some.inj (hobserved.symm.trans hobs)
  subst observed
  exact ⟨hlogical, sourceObservation_eq_entryAt state decode source⟩

/-- The stored-carrier destination traversal preserves every two-state invariant preserved by
each rooted step, threading the source table through its locator-only mutation.  This is the
`PayloadStateEq`-tolerant form of `word_stored_relocate_all_preserves`, whose `result.1 = state`
step contract cannot hold for a step that retargets the locator. -/
theorem word_stored_relocate_all_preserves_reserved
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (reserved : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (inv : WordTable Apeiron.GrowableIdentity.StoredI64 →
      table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64 → Prop)
    (hinitial : inv state reserved)
    (hstep : ∀ (s : WordTable Apeiron.GrowableIdentity.StoredI64)
      (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64) (entryAt : Usize),
      entryAt.val < plan.entries.length → inv s current →
      word_stored_relocate_entry s plan current entryAt ⦃ result => inv result.1 result.2 ⦄) :
    word_stored_relocate_all state plan reserved ⦃ result => inv result.1 result.2 ⦄ := by
  unfold word_stored_relocate_all table.Table.relocate_all
  unfold table.Table.relocate_all_loop
  apply loop.spec_decr_nat
    (measure := fun p :
      WordTable Apeiron.GrowableIdentity.StoredI64 ×
        table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64 × Usize =>
      plan.entries.length - p.2.2.val)
    (inv := fun p =>
      p.2.2.val ≤ plan.entries.length ∧ inv p.1 p.2.1)
    (post := fun result :
      WordTable Apeiron.GrowableIdentity.StoredI64 ×
        table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64 => inv result.1 result.2)
  · intro current hcurrent
    rcases current with ⟨currentState, currentReserved, entryAt⟩
    rcases hcurrent with ⟨hbound, hinv⟩
    dsimp only at hbound hinv ⊢
    unfold table.Table.relocate_all_loop.body
    by_cases hlt : entryAt < alloc.vec.Vec.len plan.entries
    · rw [if_pos hlt]
      have hlt' : entryAt.val < plan.entries.length := by simpa [alloc.vec.Vec.len] using hlt
      have hrelocate := hstep currentState currentReserved entryAt hlt' hinv
      unfold word_stored_relocate_entry at hrelocate
      step with hrelocate
      step
      refine ⟨?_, ?_, ?_⟩
      · omega
      · assumption
      · omega
    · rw [if_neg hlt, WP.spec_ok]
      exact hinv
  · exact ⟨Nat.zero_le _, hinitial⟩

end Apeiron.GrowableRepresentation
