import Growable.Geometry
import Growable.Refinement
import Growable.Scan

/-!
# Whole-table growable representation

This ratifies the whole-table contract required by `docs/epics/storage-refinement.md`, extending the
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

/-- A flat live bit or nonzero spill tag is the sole occupancy authority. -/
def Occupied {R : Type} (state : WordTable R) : Location → Prop
  | .flat index => ∃ slot,
      state.resident.slots.val[index.val]? = some slot ∧ slot.live = true
  | .spill shard index => ∃ region tag payload,
      state.shards.val[shard.val]? = some region ∧
      region.tags.val[index.val]? = some tag ∧
      region.slots.val[index.val]? = some payload ∧ tag ≠ 0#u8

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

/-- Flat extent, fixed spill directory, and each spill tag/payload directory have matching lengths. -/
def SpillDirectoryConsistent {R : Type} (state : WordTable R) : Prop :=
  state.resident.slots.length =
      Apeiron.GrowableGeometry.shardCount * state.resident.lanes.val ∧
  state.shards.length = Apeiron.GrowableGeometry.shardCount ∧
  ∀ region ∈ state.shards.val, region.tags.length = region.slots.length

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

/-- Every search-visible spill region has the scalar summary-vector shape used by `summary_maybe`. -/
def SpillSummaryShapes {R : Type} (state : WordTable R) : Prop :=
  ∀ (shard : Usize) (region : table.Region R I64),
    state.shards.val[shard.val]? = some region →
    Apeiron.GrowableScan.SpillSummaryShape region

/-- Every live spill payload records the bit selected through its carrier-reconstructed address. -/
def SpillSummaryRecorded {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (keyRep : table.keyrep.KeyRep R I64 table.WordHome) :
    Prop :=
  ∀ (shard index : Usize) (region : table.Region R I64) (tag : U8)
      (sigma : cell.Sigma) (held : R) (value key : I64) (address : table.TableAddress),
    state.shards.val[shard.val]? = some region →
    region.tags.val[index.val]? = some tag → tag ≠ 0#u8 →
    region.slots.val[index.val]? = some (sigma, held, value) →
    decode sigma held = some key →
    keyRep.address_held state.home sigma held = ok address →
    Apeiron.GrowableScan.SummaryBitRecorded region address.place

/-- Fingerprint equality is exact at equal held identities in every search-visible spill region. -/
def SpillFingerprintExactRegions {R : Type} (state : WordTable R) : Prop :=
  ∀ (shard : Usize) (region : table.Region R I64),
    state.shards.val[shard.val]? = some region →
    Apeiron.GrowableScan.SpillFingerprintExact
      (fun sigma key => Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key)

/-- A negative generated summary query excludes the queried key from its routed spill region. -/
structure SpillSummaryMissSound {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (query : LogicalKey)
    (shard : Usize) (region : table.Region R I64) : Prop where
  shape : Apeiron.GrowableScan.SpillSummaryShape region
  excludes : ∀ index value,
    Apeiron.GrowableSearch.entryAt state decode (.spill shard index) ≠ some (query, value)

/-- The three search-visible facts established at one completed spill publication point. -/
structure LivePublicationRecorded {R V : Type} (region : table.Region R V) (index : Usize)
    (payload : cell.Sigma × R × V) (place : U64) : Prop where
  payloadAt : region.slots.val[index.val]? = some payload
  occupancy : ∃ tag, region.tags.val[index.val]? = some tag ∧ tag ≠ 0#u8
  summary : table.Region.summary_maybe region place = ok true

private theorem tag_ne_zero (place : U64) (tag : U8) (htag : table.tag place = ok tag) :
    tag ≠ 0#u8 := by
  unfold table.tag at htag
  cases hshift : place >>> table.TAG_FINGERPRINT_SHIFT with
  | vis eff k => simp [hshift] at htag
  | div => simp [hshift] at htag
  | ret shifted =>
      simp only [hshift, bind_tc_ok, lift] at htag
      have hbits := congrArg UScalar.bv (Result.ok_injective htag)
      intro hzero
      have hzeroBits := congrArg UScalar.bv hzero
      simp only [UScalar.bv_or] at hbits
      rw [hzeroBits] at hbits
      have hlive := (BitVec.or_eq_zero_iff.mp hbits).1
      have hliveScalar : table.TAG_LIVE = 0#u8 :=
        (UScalar.eq_equiv_bv_eq table.TAG_LIVE 0#u8).mpr hlive
      have hvalue := congrArg UScalar.val hliveScalar
      norm_num [table.TAG_LIVE] at hvalue

theorem summary_set_frames {R V : Type} (before after : table.Region R V) (place : U64)
    (hset : table.Region.summary_set before place = ok after) :
    after.slots = before.slots ∧ after.tags = before.tags := by
  unfold table.Region.summary_set at hset
  cases hindex : table.Region.summary_index R V place (alloc.vec.Vec.len before.summary) with
  | vis eff k => simp [hindex] at hset
  | div => simp [hindex] at hset
  | ret result =>
      simp only [hindex, bind_tc_ok] at hset
      cases result with
      | none => simp at hset; subst after; exact ⟨rfl, rfl⟩
      | some index =>
          cases hrem : index % 64#usize with
          | vis eff k => simp [hrem] at hset
          | div => simp [hrem] at hset
          | ret shift =>
              simp only [hrem, bind_tc_ok] at hset
              cases hshift : 1#u64 <<< shift with
              | vis eff k => rw [hshift] at hset; simp at hset
              | div => rw [hshift] at hset; simp at hset
              | ret mask =>
                  rw [hshift] at hset
                  simp only [bind_tc_ok] at hset
                  cases hdiv : index / 64#usize with
                  | vis eff k => rw [hdiv] at hset; simp at hset
                  | div => rw [hdiv] at hset; simp at hset
                  | ret wordIndex =>
                      rw [hdiv] at hset
                      simp only [bind_tc_ok] at hset
                      cases hmut : alloc.vec.Vec.index_mut
                          (core.slice.index.SliceIndexUsizeSlice U64) before.summary wordIndex with
                      | vis eff k => rw [hmut] at hset; simp at hset
                      | div => rw [hmut] at hset; simp at hset
                      | ret pair =>
                          rcases pair with ⟨word, back⟩
                          rw [hmut] at hset
                          simp [lift] at hset
                          subst after
                          exact ⟨rfl, rfl⟩

/-- A successful generated publication changes only its selected payload
and tag lane; the intervening summary update frames both vectors. -/
theorem publish_live_frames {R V : Type}
    (before after : table.Region R V) (index read : Usize)
    (payload : cell.Sigma × R × V) (place : U64)
    (hpub : table.Region.publish_live before index payload place = ok after)
    (hne : read.val ≠ index.val) :
    after.tags.val[read.val]? = before.tags.val[read.val]? ∧
      after.slots.val[read.val]? = before.slots.val[read.val]? := by
  unfold table.Region.publish_live at hpub
  cases hslot : alloc.vec.Vec.index_mut
      (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) before.slots index with
  | vis eff k => simp [hslot] at hpub
  | div => simp [hslot] at hpub
  | ret slotResult =>
      rcases slotResult with ⟨_, slotBack⟩
      simp only [hslot, bind_tc_ok] at hpub
      generalize hset : table.Region.summary_set
          { before with slots := slotBack payload } place = summaryResult at hpub
      cases summaryResult with
      | vis eff k => simp [hset] at hpub
      | div => simp [hset] at hpub
      | ret interim =>
          simp at hpub
          generalize htag : table.tag place = tagResult at hpub
          cases tagResult with
          | vis eff k => simp [hset] at hpub
          | div => simp [hset] at hpub
          | ret tag =>
              simp [hset] at hpub
              generalize htagSlot : alloc.vec.Vec.index_mut_usize interim.tags index =
                  tagSlotResult at hpub
              cases tagSlotResult with
              | vis eff k => simp at hpub
              | div => simp at hpub
              | ret tagResult =>
                  rcases tagResult with ⟨_, tagBack⟩
                  simp at hpub
                  subst after
                  have hslotBound : index.val < before.slots.length := by
                    cases hget : before.slots.val[index.val]? with
                    | none =>
                        rw [alloc.vec.Vec.index_mut_slice_index] at hslot
                        simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize,
                          hget] at hslot
                    | some old => exact (List.getElem?_eq_some_iff.mp hget).1
                  have htagBound : index.val < interim.tags.length := by
                    cases hget : interim.tags.val[index.val]? with
                    | none =>
                        simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize,
                          hget] at htagSlot
                    | some old => exact (List.getElem?_eq_some_iff.mp hget).1
                  have hslotSpec := alloc.vec.Vec.index_mut_usize_spec
                    before.slots index hslotBound
                  rw [alloc.vec.Vec.index_mut_slice_index] at hslot
                  rw [hslot, WP.spec_ok] at hslotSpec
                  have htagSpec := alloc.vec.Vec.index_mut_usize_spec
                    interim.tags index htagBound
                  rw [htagSlot, WP.spec_ok] at htagSpec
                  have hframes := summary_set_frames
                    { before with slots := slotBack payload } interim place hset
                  constructor
                  · change (tagBack tag).val[read.val]? = before.tags.val[read.val]?
                    rw [htagSpec.2, alloc.vec.Vec.set_val_eq, hframes.2]
                    simp [hne]
                  · change interim.slots.val[read.val]? = before.slots.val[read.val]?
                    rw [hframes.1, hslotSpec.2, alloc.vec.Vec.set_val_eq]
                    simp [hne]

/-- Successful generated payload-summary-tag publication establishes the local recordedness fact. -/
theorem publish_live_recorded {R V : Type} (before after : table.Region R V) (index : Usize)
    (payload : cell.Sigma × R × V) (place : U64)
    (hpub : table.Region.publish_live before index payload place = ok after) :
    LivePublicationRecorded after index payload place := by
  unfold table.Region.publish_live at hpub
  cases hslot : alloc.vec.Vec.index_mut
      (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) before.slots index with
  | vis eff k => simp [hslot] at hpub
  | div => simp [hslot] at hpub
  | ret slotResult =>
      rcases slotResult with ⟨oldPayload, slotBack⟩
      simp only [hslot, bind_tc_ok] at hpub
      simp at hpub
      generalize hset : table.Region.summary_set
          { before with slots := slotBack payload } place = summaryResult at hpub
      cases summaryResult with
      | vis eff k => simp at hpub
      | div => simp at hpub
      | ret interim =>
          simp at hpub
          generalize htag : table.tag place = tagResult at hpub
          cases tagResult with
          | vis eff k => simp at hpub
          | div => simp at hpub
          | ret tag =>
              simp at hpub
              generalize htagSlot : alloc.vec.Vec.index_mut_usize
                  interim.tags index = tagSlotResult at hpub
              cases tagSlotResult with
              | vis eff k => simp at hpub
              | div => simp at hpub
              | ret tagResult =>
                  rcases tagResult with ⟨oldTag, tagBack⟩
                  simp at hpub
                  subst after
                  have hslotBound : index.val < before.slots.length := by
                    cases hget : before.slots.val[index.val]? with
                    | none =>
                        rw [alloc.vec.Vec.index_mut_slice_index] at hslot
                        simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, hget] at hslot
                    | some old => exact (List.getElem?_eq_some_iff.mp hget).1
                  have htagBound : index.val < interim.tags.length := by
                    cases hget : interim.tags.val[index.val]? with
                    | none =>
                        simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, hget] at htagSlot
                    | some old => exact (List.getElem?_eq_some_iff.mp hget).1
                  have hslotSpec :=
                    alloc.vec.Vec.index_mut_usize_spec before.slots index hslotBound
                  rw [alloc.vec.Vec.index_mut_slice_index] at hslot
                  rw [hslot, WP.spec_ok] at hslotSpec
                  have htagSpec :=
                    alloc.vec.Vec.index_mut_usize_spec interim.tags index htagBound
                  rw [htagSlot, WP.spec_ok] at htagSpec
                  have hslotBack : slotBack = before.slots.set index := hslotSpec.2
                  have htagBack : tagBack = interim.tags.set index := htagSpec.2
                  have hframes := summary_set_frames
                    { before with slots := slotBack payload } interim place hset
                  have hinterimSlots : interim.slots = slotBack payload := hframes.1
                  refine ⟨?_, ⟨tag, ?_, tag_ne_zero place tag htag⟩, ?_⟩
                  · change interim.slots.val[index.val]? = some payload
                    rw [hinterimSlots, hslotBack]
                    rw [alloc.vec.Vec.set_val_eq]
                    exact List.getElem?_set_self hslotBound
                  · change (tagBack tag).val[index.val]? = some tag
                    rw [htagBack]
                    rw [alloc.vec.Vec.set_val_eq]
                    exact List.getElem?_set_self htagBound
                  · change table.Region.summary_maybe interim place = ok true
                    exact Apeiron.GrowableScan.summary_set_implies_maybe
                      { before with slots := slotBack payload } interim place hset

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
  summaryShape : SpillSummaryShapes state
  summaryRecorded : ∃ keyRep : table.keyrep.KeyRep R I64 table.WordHome,
    CarrierAddressRespecting state decode keyRep ∧ SpillSummaryRecorded state decode keyRep
  fingerprintExact : SpillFingerprintExactRegions state

private def EmptyRegion {R : Type} (region : table.Region R I64) : Prop :=
  region.tags.val = [] ∧ region.slots.val = [] ∧ region.summary.val = []

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
    (haddress : ∀ sigma held key, decode sigma held = some key →
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
      (hregion : state.shards.val[shard.val]? = some region) : EmptyRegion region := by
    apply hshardsEmpty region
    exact List.mem_of_getElem? hregion
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
              simp [Apeiron.GrowableSearch.entryAt, hregion, hempty.1] at hentry
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
              simp [hregion, hempty.1] at hentry
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
            simp [hregion, hempty.1] at hentry
  · intro location
    cases location with
    | flat index => simp [Occupied, Apeiron.GrowableSearch.entryAt, state, alloc.vec.Vec.new]
    | spill shard index =>
        cases hregion : state.shards.val[shard.val]? with
        | none => simp [Occupied, Apeiron.GrowableSearch.entryAt, hregion]
        | some region =>
            have hempty := regionEmpty hregion
            simp [Occupied, Apeiron.GrowableSearch.entryAt, hregion, hempty.1, hempty.2.1]
  · intro location key value hentry
    cases location with
    | flat index => simp [Apeiron.GrowableSearch.entryAt, state, alloc.vec.Vec.new] at hentry
    | spill shard index =>
        simp only [Apeiron.GrowableSearch.entryAt] at hentry
        cases hregion : state.shards.val[shard.val]? with
        | none => simp [hregion] at hentry
        | some region =>
            have hempty := regionEmpty hregion
            simp [hregion, hempty.1] at hentry
  · intro key location
    constructor
    · intro h; simp at h
    · rintro ⟨address, index, _haddress, hindex, rfl⟩
      simp only [Apeiron.GrowableGeometry.preparedFlatIndex, state] at hindex
      cases hword : address.direct_word <;> simp [hword] at hindex
  · intro query location hdirect
    simp at hdirect
  · refine ⟨?_, hshardsLength, ?_⟩
    · simp [state, Apeiron.GrowableGeometry.shardCount, alloc.vec.Vec.new]
    · intro region hregion
      have hempty := hshardsEmpty region hregion
      simp [hempty.1, hempty.2.1]
  · exact ⟨[], by simp, by simp [state]⟩
  · intro shard region hregion
    exact Apeiron.GrowableScan.spillSummaryShape region
  · refine ⟨keyRep, ?_, ?_⟩
    · intro sigma held key hdecode
      exact haddress sigma held key hdecode
    · intro shard index region tag sigma held value key address hregion htag
        _htagLive hpayload
      have hempty := regionEmpty hregion
      simp [hempty.1] at htag
  · intro shard region hregion
    exact Apeiron.GrowableScan.spillFingerprintExact _

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

/-- A generated summary miss contradicts every live representative of the same routed key. -/
theorem repR_spillSummaryMissSound {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) (query : LogicalKey)
    (address : table.TableAddress) (shard : Usize) (region : table.Region R I64)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address
      state.home query.1 query.2 = ok address)
    (_hshard : table.shard address = ok shard)
    (hregion : state.shards.val[shard.val]? = some region)
    (hmiss : table.Region.summary_maybe region address.place = ok false) :
    SpillSummaryMissSound state decode query shard region := by
  rcases query with ⟨querySigma, queryKey⟩
  refine ⟨h.summaryShape shard region hregion, ?_⟩
  intro index value hentry
  have hentryAddress := h.address (.spill shard index) (querySigma, queryKey) value hentry
  rcases hentryAddress with ⟨entryAddress, hentryAddress, _⟩
  have heqAddress : entryAddress = address := by
    rw [haddress] at hentryAddress
    exact (Result.ok_injective hentryAddress).symm
  subst entryAddress
  simp only [Apeiron.GrowableSearch.entryAt] at hentry
  rw [hregion] at hentry
  cases htag : region.tags.val[index.val]? with
  | none => simp [htag] at hentry
  | some tag =>
      cases hpayload : region.slots.val[index.val]? with
      | none => simp [htag, hpayload] at hentry
      | some payload =>
          rcases payload with ⟨sigma, held, storedValue⟩
          by_cases hzero : tag = 0#u8
          · simp [htag, hpayload, hzero] at hentry
          · cases hdecode : decode sigma held with
            | none => simp [htag, hpayload, hzero, hdecode] at hentry
            | some key =>
                simp only [htag, hpayload, hzero, hdecode, ↓reduceIte,
                  Option.some.injEq, Prod.mk.injEq] at hentry
                rcases hentry with ⟨⟨hsigma, hkey⟩, hvalue⟩
                subst sigma
                subst key
                subst storedValue
                rcases h.summaryRecorded with ⟨keyRep, hcarrierAddress, hsummaryRecorded⟩
                have hheldAddress : keyRep.address_held state.home querySigma held = ok address := by
                  rw [hcarrierAddress querySigma held queryKey hdecode, haddress]
                have hrecorded := hsummaryRecorded shard index region tag querySigma held value
                  queryKey address hregion htag hzero hpayload hdecode hheldAddress
                exact (Apeiron.GrowableScan.summary_maybe_false_excludes_recorded
                  region address.place hmiss) hrecorded

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
    (hcarrierAddress : CarrierAddressRespecting
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode keyRep)
    (hsummaryShape : SpillSummaryShapes
      (Apeiron.GrowableRefinement.rebuiltView state reserved))
    (hsummaryRecorded : SpillSummaryRecorded
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode keyRep)
    (hfingerprintExact : SpillFingerprintExactRegions
      (Apeiron.GrowableRefinement.rebuiltView state reserved)) :
    @ReservedRepR R state reserved decode direct live logical :=
  ⟨hliveSet, hobservations, hunique, hcoverage, haddress, hdirectIndex, hdirectCover, hdirectory,
    hlength, hsummaryShape, ⟨keyRep, hcarrierAddress, hsummaryRecorded⟩, hfingerprintExact⟩

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

/-- Every rebuilt search-visible spill region has valid generated summary shape. -/
theorem reservedRepR_summaryShape {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    SpillSummaryShapes (Apeiron.GrowableRefinement.rebuiltView state reserved) := h.summaryShape

/-- Every rebuilt live spill representative has its routed summary bit recorded. -/
theorem reservedRepR_summaryRecorded {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    ∃ keyRep : table.keyrep.KeyRep R I64 table.WordHome,
      CarrierAddressRespecting (Apeiron.GrowableRefinement.rebuiltView state reserved) decode keyRep ∧
      SpillSummaryRecorded (Apeiron.GrowableRefinement.rebuiltView state reserved) decode keyRep :=
  h.summaryRecorded

/-- Every rebuilt search-visible spill region uses an equality-respecting fingerprint dictionary. -/
theorem reservedRepR_fingerprintExact {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    SpillFingerprintExactRegions (Apeiron.GrowableRefinement.rebuiltView state reserved) :=
  h.fingerprintExact

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
          cases htag : region.tags.val[index.val]? with
          | none => simp [Apeiron.GrowableRelocation.sourceObservation,
              Apeiron.GrowableSearch.entryAt, sourceLocation, hregion, htag]
          | some tag =>
              cases hslot : region.slots.val[index.val]? with
              | none => simp [Apeiron.GrowableRelocation.sourceObservation,
                  Apeiron.GrowableSearch.entryAt, sourceLocation, hregion, htag, hslot]
              | some payload =>
                  rcases payload with ⟨sigma, held, value⟩
                  by_cases hzero : tag = 0#u8 <;>
                    cases hdecode : decode sigma held <;>
                    simp [Apeiron.GrowableRelocation.sourceObservation,
                      Apeiron.GrowableSearch.entryAt, sourceLocation, hregion, htag, hslot,
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
          cases htag : region.tags.val[index.val]? with
          | none => simp [Apeiron.GrowableRelocation.reservedObservation,
              Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
              destinationLocation, hregion, htag]
          | some tag =>
              cases hslot : region.slots.val[index.val]? with
              | none => simp [Apeiron.GrowableRelocation.reservedObservation,
                  Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
                  destinationLocation, hregion, htag, hslot]
              | some payload =>
                  rcases payload with ⟨sigma, held, value⟩
                  by_cases hzero : tag = 0#u8 <;>
                    cases hdecode : decode sigma held <;>
                    simp [Apeiron.GrowableRelocation.reservedObservation,
                      Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
                      destinationLocation, hregion, htag, hslot, hzero, hdecode]

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
          cases hsourceTag : sourceRegion.tags.val[sourceIndex.val]? with
          | none => simp [hsourceRegion, hsourceTag] at hsource
          | some sourceTag =>
              cases hsourceSlot : sourceRegion.slots.val[sourceIndex.val]? with
              | none => simp [hsourceRegion, hsourceTag, hsourceSlot] at hsource
              | some sourcePayloadVal =>
                  rcases sourcePayloadVal with ⟨sSigma, sHeld, sValue⟩
                  by_cases hzero : sourceTag = 0#u8
                  · simp [hsourceRegion, hsourceTag, hsourceSlot, hzero] at hsource
                  · simp only [hsourceRegion, hsourceTag, hsourceSlot, hzero, ↓reduceIte,
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
          cases hsourceTag : sourceRegion.tags.val[sourceIndex.val]? with
          | none => simp [hsourceRegion, hsourceTag] at hsource
          | some sourceTag =>
              cases hsourceSlot : sourceRegion.slots.val[sourceIndex.val]? with
              | none => simp [hsourceRegion, hsourceTag, hsourceSlot] at hsource
              | some sourcePayloadVal =>
                  rcases sourcePayloadVal with ⟨sSigma, sHeld, sValue⟩
                  by_cases hzero : sourceTag = 0#u8
                  · simp [hsourceRegion, hsourceTag, hsourceSlot, hzero] at hsource
                  · simp only [hsourceRegion, hsourceTag, hsourceSlot, hzero, ↓reduceIte,
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
      published.slots.val[index.val]? = some (sigma, { key := held.key }, value) ∧
      (∃ tag : U8, published.tags.val[index.val]? = some tag ∧
        table.tag address.place = ok tag ∧ tag ≠ 0#u8) ∧
      table.shard address = ok shard ∧
      Apeiron.GrowableIdentity.wordHomeInst.address result.1.home sigma held.key = ok address ∧
      (∀ other : Usize, other.val ≠ shard.val →
        result.2.regions.val[other.val]? = current.regions.val[other.val]?) ∧
      (∀ other : Usize, other.val ≠ index.val →
        published.tags.val[other.val]? = region.tags.val[other.val]? ∧
        published.slots.val[other.val]? = region.slots.val[other.val]?) ⦄ := by
  rw [word_stored_relocate_entry_spill_run state plan current entryAt shard index entry sigma held
    value address region published nextLocator hentry hsource hflat hspill haddress hregion hvacant
    hretarget hpublish, WP.spec_ok]
  have hshardBound : shard.val < current.regions.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  obtain ⟨tag, htagGen, htagNz, htagAt⟩ := Apeiron.GrowableScan.publish_live_tag_readback
    region published index (sigma, { key := held.key }, value) address.place hpublish
  have hrecorded := publish_live_recorded region published index
    (sigma, { key := held.key }, value) address.place hpublish
  refine ⟨?_, ?_, ⟨tag, htagAt, htagGen, htagNz⟩, hshard, haddress, ?_, ?_⟩
  · simp [alloc.vec.Vec.set_val_eq, List.getElem?_set_self hshardBound]
  · exact hrecorded.payloadAt
  · intro other hne
    simp [alloc.vec.Vec.set_val_eq, hne]
  · intro other hne
    exact publish_live_frames region published index other
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
  obtain ⟨tag, _, htagNz, htagAt⟩ := Apeiron.GrowableScan.publish_live_tag_readback
    region published index (sigma, { key := held.key }, value) address.place hpublish
  have hrecorded := publish_live_recorded region published index
    (sigma, { key := held.key }, value) address.place hpublish
  have hread :
      Apeiron.GrowableRelocation.reservedObservation
          { current with regions := current.regions.set shard published }
          Apeiron.GrowableSearch.decodeStoredOption (.spill shard index) =
        some ((sigma, held.key), value) := by
    simp [Apeiron.GrowableRelocation.reservedObservation, alloc.vec.Vec.set_val_eq,
      List.getElem?_set_self hshardBound, hrecorded.payloadAt, htagAt, htagNz,
      Apeiron.GrowableSearch.decodeStoredOption]
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
