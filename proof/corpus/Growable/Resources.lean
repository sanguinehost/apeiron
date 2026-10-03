import Growable.Representation

/-!
# Prepared rebuild resources

The certificates in this module are input-side data over the generated plan, source table, and
materialized flat/spill storage.  Publication is the generated production `publish_prepared` body.
Planning, capacity observation, materialization, and allocator execution remain outside extraction;
`ResourceAdapterSimulates` is the explicit premise connecting those operations to these values.
-/

namespace Apeiron.GrowableResources

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedVariables false

abbrev WordTable (R : Type) := table.Table I64 I64 table.WordHome R 16#usize
abbrev AdapterResult (R : Type) :=
  core.result.Result
    (table.RebuildPlan × table.RebuildReserved R I64 × table.resources.PreparedOutcome R)
    table.StorageError
abbrev PublishedResult (R : Type) := table.resources.PreparedResult R I64
abbrev Destination := Apeiron.GrowableRelocation.Destination

/-- The exact generated `KeyRep` dictionary passed to stored-carrier relocation roots. -/
def storedRelocationKeyRep :
    table.keyrep.KeyRep (table.keyrep.Stored I64) I64 table.WordHome :=
  Apeiron.GrowableIdentity.storedKeyRep

/-- The exact generated `KeyRep` dictionary passed to regenerating-carrier relocation roots. -/
def regenRelocationKeyRep :
    table.keyrep.KeyRep (table.keyrep.Regen table.keyrep.I64WordCodec) I64 table.WordHome :=
  Apeiron.GrowableIdentity.regenKeyRep

/-- The generated relocation dictionary selected for one concrete carrier. -/
class ProductionRelocationKeyRep (R : Type) where
  keyRep : table.keyrep.KeyRep R I64 table.WordHome

instance storedProductionRelocationKeyRep :
    ProductionRelocationKeyRep (table.keyrep.Stored I64) where
  keyRep := storedRelocationKeyRep

instance regenProductionRelocationKeyRep :
    ProductionRelocationKeyRep (table.keyrep.Regen table.keyrep.I64WordCodec) where
  keyRep := regenRelocationKeyRep

/-- The unchanged pending operation supplied to the resource adapter. -/
structure PendingInput where
  sigma : cell.Sigma
  key : I64
  value : I64
  address : table.TableAddress

/-- The callback-free relocation call trace required by the prepared-core boundary. -/
structure CoreCallTrace where
  allocationCalls : Nat
  initializationCalls : Nat
  mergeCalls : Nat

/-- Relocation has no allocator, initializer, or merge effect. -/
def NoCoreEffects (trace : CoreCallTrace) : Prop :=
  trace.allocationCalls = 0 ∧ trace.initializationCalls = 0 ∧ trace.mergeCalls = 0

/-- The generated destination index is inside the materialized flat or spill vectors.  A spill
    region carries one slot array, so the destination bound is the single slot bound. -/
def DestinationInBounds {R : Type} (storage : table.RebuildReserved R I64) :
    Destination → Prop
  | .flat index => index.val < storage.resident.slots.length
  | .spill shard index => ∃ region,
      storage.regions.val[shard.val]? = some region ∧
      index.val < region.slots.length

/-- Flat live bits and the slots' own `tag` markers show that materialized relocation storage is
    initially vacant.  The retired separate tag array is replaced by the marker each slot carries. -/
def MaterializedVacant {R : Type} (storage : table.RebuildReserved R I64) : Prop :=
  (∀ slot ∈ storage.resident.slots.val, slot.live = false) ∧
  (∀ region ∈ storage.regions.val, ∀ slot ∈ region.slots.val, slot.tag = 0#u8)

/-- Receipt targets and materialized lengths agree with the complete generated target geometry. -/
structure ReceiptMatchesStorage {R : Type} (plan : table.RebuildPlan)
    (storage : table.RebuildReserved R I64)
    (receipt : table.resources.ReservationReceipt) : Prop where
  lanes : storage.resident.lanes = plan.lanes
  flatReceipt : receipt.flat.required_len.val = storage.resident.slots.length
  flatExtent : storage.resident.slots.length =
    Apeiron.GrowableGeometry.shardCount * plan.lanes.val
  directoryReceipt : receipt.directory.required_len.val = storage.regions.length
  spillDirectory : storage.regions.length = Apeiron.GrowableGeometry.shardCount
  receiptDirectory : receipt.regions.val.length = Apeiron.GrowableGeometry.shardCount
  regionLengths : ∀ (shard : Nat) (region : table.Region R I64)
      (regionReceipt : table.resources.RegionReservation),
    storage.regions.val[shard]? = some region →
    receipt.regions.val[shard]? = some regionReceipt →
    regionReceipt.slots.required_len.val = region.slots.length

/-- The materialized flat vector has the checked target extent `256 × plan.lanes`. -/
theorem receiptMatches_flatExtent {R : Type} {plan : table.RebuildPlan}
    {storage : table.RebuildReserved R I64} {receipt : table.resources.ReservationReceipt}
    (h : ReceiptMatchesStorage plan storage receipt) :
    storage.resident.slots.length =
      Apeiron.GrowableGeometry.shardCount * plan.lanes.val := h.flatExtent

/-- The materialized storage and its fixed receipt both have the 256-entry spill directory. -/
theorem receiptMatches_spillDirectory {R : Type} {plan : table.RebuildPlan}
    {storage : table.RebuildReserved R I64} {receipt : table.resources.ReservationReceipt}
    (h : ReceiptMatchesStorage plan storage receipt) :
    storage.regions.length = Apeiron.GrowableGeometry.shardCount ∧
      receipt.regions.val.length = Apeiron.GrowableGeometry.shardCount :=
  ⟨h.spillDirectory, h.receiptDirectory⟩

/-- Every receipt entry carries the slot-array length of its target spill region. -/
theorem receiptMatches_regionLengths {R : Type} {plan : table.RebuildPlan}
    {storage : table.RebuildReserved R I64} {receipt : table.resources.ReservationReceipt}
    (h : ReceiptMatchesStorage plan storage receipt) (shard : Nat)
    (region : table.Region R I64) (regionReceipt : table.resources.RegionReservation)
    (hregion : storage.regions.val[shard]? = some region)
    (hreceipt : receipt.regions.val[shard]? = some regionReceipt) :
    regionReceipt.slots.required_len.val = region.slots.length :=
  h.regionLengths shard region regionReceipt hregion hreceipt

/-- One observed capacity satisfies the production scalar comparison. -/
def CapacityBound_R (bound : table.resources.CapacityBound) : Prop :=
  word_capacity_bound_holds bound.required_len bound.observed_capacity = ok true

/-- Every scalar observation in a reservation receipt satisfies the production comparison. -/
def EveryScalarCapacityCheck_R (receipt : table.resources.ReservationReceipt) : Prop :=
  CapacityBound_R receipt.flat ∧
  CapacityBound_R receipt.directory ∧
  ∀ region ∈ receipt.regions.val,
    CapacityBound_R region.slots

/--
Concrete prepared-resource evidence for one generated success envelope.  In particular, this is not
a family of caller-chosen propositions: its data are the exact table, plan, storage, receipt, held
carrier, generated carrier operations, materialized vacancy proof, and relocation call trace.
-/
structure PreparedResourceClauses (R : Type) where
  state : WordTable R
  pending : PendingInput
  plan : table.RebuildPlan
  storage : table.RebuildReserved R I64
  receipt : table.resources.ReservationReceipt
  locatorReceipt : table.resources.LocatorReservationReceipt
  held : R
  decode : cell.Sigma → R → Option I64
  addressHeld : cell.Sigma → R → Result table.TableAddress
  receiptMatches : ReceiptMatchesStorage plan storage receipt
  materializedVacant : MaterializedVacant storage
  pendingIdentity : decode pending.sigma held = some pending.key
  pendingAddress : addressHeld pending.sigma held = ok pending.address
  relocationTrace : CoreCallTrace
  noRelocationEffects : NoCoreEffects relocationTrace

/-- The exact prepared-resource envelope, indexed by the actual generated publication result. -/
def PreparedResources_R {R : Type} (certificate : PreparedResourceClauses R)
    (state : WordTable R) (pending : PendingInput) (adapterResult : AdapterResult R)
    (prepared : PublishedResult R) : Prop :=
  certificate.state = state ∧ certificate.pending = pending ∧
  adapterResult = .Ok (certificate.plan, certificate.storage,
    { reservation := certificate.receipt, locator := certificate.locatorReceipt,
      pending_held := certificate.held }) ∧
  table.resources.publish_prepared adapterResult = ok prepared ∧
  prepared = .Prepared certificate.plan certificate.storage
    { reservation := certificate.receipt, locator := certificate.locatorReceipt,
      pending_held := certificate.held } ∧
  EveryScalarCapacityCheck_R certificate.receipt

/-- A plan entry is a live, in-bounds generated source, and every live source occurs exactly once. -/
structure SourceEnumerationCertificate {R : Type} (state : WordTable R)
    (plan : table.RebuildPlan) where
  plannedLive : ∀ (entryAt : Usize),
    entryAt.val < plan.entries.length → ∃ entry payload,
      plan.entries.val[entryAt.val]? = some entry ∧
      Apeiron.GrowableRelocation.sourcePayload state entry.source = some payload
  sourceBijection : ∀ (source : table.RebuildSource) (payload : cell.Sigma × R × I64),
    Apeiron.GrowableRelocation.sourcePayload state source = some payload →
    ∃! entryAt : Usize, entryAt.val < plan.entries.length ∧ ∃ entry,
      plan.entries.val[entryAt.val]? = some entry ∧ entry.source = source
  sourceInjective : ∀ (left right : Usize)
      (leftEntry rightEntry : table.RebuildPlanEntry),
    plan.entries.val[left.val]? = some leftEntry →
    plan.entries.val[right.val]? = some rightEntry →
    leftEntry.source = rightEntry.source → left = right

/-- One planned write preserves decoded identity, value, carrier, address, and destination geometry. -/
structure EntryObservationCertificate {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (cloneHeld : R → R)
    (addressHeld : cell.Sigma → R → Result table.TableAddress)
    (plan : table.RebuildPlan) (entryAt : Usize) (destination : Destination) where
  entry : table.RebuildPlanEntry
  sigma : cell.Sigma
  sourceHeld : R
  writtenHeld : R
  sourceValue : I64
  writtenValue : I64
  key : I64
  address : table.TableAddress
  tag : U8
  entry_eq : plan.entries.val[entryAt.val]? = some entry
  source_eq : Apeiron.GrowableRelocation.sourcePayload state entry.source =
    some (sigma, sourceHeld, sourceValue)
  carrier_eq : writtenHeld = cloneHeld sourceHeld
  value_eq : writtenValue = sourceValue
  sourceIdentity : decode sigma sourceHeld = some key
  writtenIdentity : decode sigma writtenHeld = some key
  address_eq : addressHeld sigma writtenHeld = ok address
  tag_eq : table.tag address.place = ok tag
  tag_live : tag ≠ 0#u8
  destination_eq :
    match entry.flat, destination with
    | some planned, .flat actual =>
        planned = actual ∧
        Apeiron.GrowableGeometry.preparedFlatIndex address plan.lanes.val = some actual.val
    | none, .spill shard _ => entry.spill = shard ∧ table.shard address = ok shard
    | _, _ => False

/-- Apply one symbolic plan write to the generated flat/spill storage. -/
def applyPlannedWrite {R : Type} {state : WordTable R} {decode : cell.Sigma → R → Option I64}
    {cloneHeld : R → R} {addressHeld : cell.Sigma → R → Result table.TableAddress}
    {plan : table.RebuildPlan} {entryAt : Usize} {destination : Destination}
    (current : table.RebuildReserved R I64)
    (entry : EntryObservationCertificate state decode cloneHeld addressHeld plan entryAt destination) :
    table.RebuildReserved R I64 :=
  match destination with
  | .flat index => Apeiron.GrowableRelocation.writeFlat current index
      { live := true, sigma := entry.sigma, held := entry.writtenHeld, value := entry.writtenValue }
  | .spill shard index =>
      match current.regions.val[shard.val]? with
      | some region => Apeiron.GrowableRelocation.writeSpill current shard index region entry.tag
          (entry.sigma, entry.writtenHeld, entry.writtenValue)
      | none => current

/-- The pending operation's unique symbolic found-or-vacant classification. -/
inductive PendingSymbolicDecision where
  | found (entryAt : Usize) (destination : Destination)
  | vacant (destination : Destination)

/-- Concrete validity of a symbolic pending decision against the source enumeration and assignment. -/
def PendingDecisionValid {R : Type} (state : WordTable R) (decode : cell.Sigma → R → Option I64)
    (pending : PendingInput) (plan : table.RebuildPlan)
    (storage : table.RebuildReserved R I64) (assignment : Usize → Destination) :
    PendingSymbolicDecision → Prop
  | .found entryAt destination =>
      entryAt.val < plan.entries.length ∧ assignment entryAt = destination ∧
      ∃ entry sigma held value,
        plan.entries.val[entryAt.val]? = some entry ∧
        Apeiron.GrowableRelocation.sourcePayload state entry.source = some (sigma, held, value) ∧
        sigma = pending.sigma ∧ decode sigma held = some pending.key
  | .vacant destination =>
      DestinationInBounds storage destination ∧
      (∀ entryAt, entryAt.val < plan.entries.length → assignment entryAt ≠ destination) ∧
      ∀ source sigma held value,
        Apeiron.GrowableRelocation.sourcePayload state source = some (sigma, held, value) →
         sigma = pending.sigma → decode sigma held ≠ some pending.key

/-- Every live source has one planned destination and no destination is extraneous. -/
def PlannedCoverage {R : Type} (state : WordTable R) (plan : table.RebuildPlan)
    (storage : table.RebuildReserved R I64) (assignment : Usize → Destination) : Prop :=
  (∀ (source : table.RebuildSource) (payload : cell.Sigma × R × I64),
    Apeiron.GrowableRelocation.sourcePayload state source = some payload →
    ∃ (entryAt : Usize) (entry : table.RebuildPlanEntry),
      entryAt.val < plan.entries.length ∧
      plan.entries.val[entryAt.val]? = some entry ∧ entry.source = source) ∧
  ∀ (entryAt : Usize), entryAt.val < plan.entries.length →
    ∃ (entry : table.RebuildPlanEntry) (payload : cell.Sigma × R × I64),
    plan.entries.val[entryAt.val]? = some entry ∧
    Apeiron.GrowableRelocation.sourcePayload state entry.source = some payload ∧
    DestinationInBounds storage (assignment entryAt)

/-- Every planned certified direct coordinate is assigned to that exact flat destination. -/
def PlannedDirectCover {R : Type} (state : WordTable R) (decode : cell.Sigma → R → Option I64)
    (cloneHeld : R → R) (addressHeld : cell.Sigma → R → Result table.TableAddress)
    (plan : table.RebuildPlan) (assignment : Usize → Destination) : Prop :=
  ∀ (entryAt : Usize),
    ∀ certificate : EntryObservationCertificate state decode cloneHeld addressHeld plan entryAt
      (assignment entryAt),
    ∀ flatIndex, certificate.entry.flat = some flatIndex →
      assignment entryAt = .flat flatIndex

/--
Input-side physical certificates for the plan's fully written target.  `targetStorage` is symbolic
plan data, not the state returned by the extracted traversal; its direct relation therefore uses
the target lane extent rather than the pre-state geometry.
-/
structure TargetGeometryCertificate {R : Type} (state : WordTable R) (plan : table.RebuildPlan)
    (targetStorage : table.RebuildReserved R I64) (decode : cell.Sigma → R → Option I64)
    (directAfter : Apeiron.GrowableRepresentation.LogicalKey → Option
      Apeiron.GrowableRepresentation.Location)
    (committedLive : Apeiron.GrowableRepresentation.LogicalKey → Prop) : Prop where
  lanes : targetStorage.resident.lanes = plan.lanes
  liveSet : Apeiron.GrowableRepresentation.LiveSetEqual
    (Apeiron.GrowableRefinement.rebuiltView state targetStorage) decode committedLive
  unique : Apeiron.GrowableRepresentation.UniqueRepresentation
    (Apeiron.GrowableRefinement.rebuiltView state targetStorage) decode
  coverage : Apeiron.GrowableRepresentation.PhysicalCoverage
    (Apeiron.GrowableRefinement.rebuiltView state targetStorage) decode
  address : Apeiron.GrowableRepresentation.AddressCorrect
    (Apeiron.GrowableRefinement.rebuiltView state targetStorage) decode
  directIndex : Apeiron.GrowableRepresentation.DirectIndexCorrect
    (Apeiron.GrowableRefinement.rebuiltView state targetStorage) directAfter
  directCover : Apeiron.GrowableSearch.DirectCover
    (Apeiron.GrowableRefinement.rebuiltView state targetStorage) decode directAfter
  directory : Apeiron.GrowableRepresentation.SpillDirectoryConsistent
    (Apeiron.GrowableRefinement.rebuiltView state targetStorage)
  length : Apeiron.GrowableRepresentation.LengthCorrect
    (Apeiron.GrowableRefinement.rebuiltView state targetStorage) committedLive

/--
Concrete plan certificate.  The prefix function is symbolic input data: `prefixStep` fixes it to the
actual generated flat/spill writes, and `prefixVacancy` checks each spill write with the generated
`spill_locate_held`-shaped operation before traversal executes.
-/
structure RebuildPlanClauses (R : Type) where
  state : WordTable R
  pending : PendingInput
  plan : table.RebuildPlan
  storage : table.RebuildReserved R I64
  decode : cell.Sigma → R → Option I64
  cloneHeld : R → R
  addressHeld : cell.Sigma → R → Result table.TableAddress
  locateHeld : table.Region R I64 → cell.Sigma → table.TableAddress → R → Result table.SpillSite
  keyRep : table.keyrep.KeyRep R I64 table.WordHome
  cloneHeld_eq : ∀ held, keyRep.corecloneCloneInst.clone held = ok (cloneHeld held)
  addressHeld_eq : addressHeld = fun sigma held => keyRep.address_held state.home sigma held
  locateHeld_eq : locateHeld = fun region sigma address held =>
    table.probe.spill_locate_held keyRep region sigma address held 16#usize
  sourceEnumeration : SourceEnumerationCertificate state plan
  assignment : Usize → Destination
  destinationInBounds : ∀ (entryAt : Usize),
    entryAt.val < plan.entries.length → DestinationInBounds storage (assignment entryAt)
  destinationDisjoint : ∀ (left right : Usize),
    left.val < plan.entries.length → right.val < plan.entries.length → left ≠ right →
    assignment left ≠ assignment right
  entryObservation : (entryAt : Usize) → (bound : entryAt.val < plan.entries.length) →
    EntryObservationCertificate state decode cloneHeld addressHeld plan entryAt (assignment entryAt)
  logicalKeyInjective : ∀ (left right : Usize)
      (leftBound : left.val < plan.entries.length)
      (rightBound : right.val < plan.entries.length),
    ((entryObservation left leftBound).sigma, (entryObservation left leftBound).key) =
      ((entryObservation right rightBound).sigma, (entryObservation right rightBound).key) →
    left = right
  prefixStorage : Nat → table.RebuildReserved R I64
  prefixInitial : prefixStorage 0 = storage
  prefixStep : ∀ (entryAt : Usize) (bound : entryAt.val < plan.entries.length),
    prefixStorage (entryAt.val + 1) =
      applyPlannedWrite (prefixStorage entryAt.val) (entryObservation entryAt bound)
  prefixVacancy : ∀ (entryAt : Usize) (bound : entryAt.val < plan.entries.length),
    match assignment entryAt with
    | .flat _ => True
    | .spill shard index => ∃ region,
        (prefixStorage entryAt.val).regions.val[shard.val]? = some region ∧
        locateHeld region (entryObservation entryAt bound).sigma
          (entryObservation entryAt bound).address
          (entryObservation entryAt bound).writtenHeld = ok (.Vacant index)
  directAfter : Apeiron.GrowableRepresentation.LogicalKey → Option
    Apeiron.GrowableRepresentation.Location
  committedLive : Apeiron.GrowableRepresentation.LogicalKey → Prop
  targetGeometry : TargetGeometryCertificate state plan (prefixStorage plan.entries.length)
    decode directAfter committedLive
  pendingDecision : PendingSymbolicDecision
  pendingDecisionValid : PendingDecisionValid state decode pending plan storage assignment pendingDecision
  pendingDecisionUnique : ∀ (decision : PendingSymbolicDecision),
    PendingDecisionValid state decode pending plan storage assignment decision →
    decision = pendingDecision
  coverage : PlannedCoverage state plan storage assignment
  directCover : PlannedDirectCover state decode cloneHeld addressHeld plan assignment

/-- Convert a symbolic rebuild destination to the exact generated runtime location. -/
def runtimeLocation (destination : Destination) : table.SearchLocation :=
  match destination with
  | .flat index => .Flat index
  | .spill shard index => .Spill shard index

/--
The runtime decision supplied by unextracted `locate_rebuilt` has the class and complete generated
location of the plan's unique symbolic decision.  In particular, spill correspondence retains both
the shard and the in-shard index.  Establishing this premise from execution requires extracting or
otherwise source-binding `locate_rebuilt`; no generated definition for that operation exists here.
-/
structure DecisionSymbolicCorrespondence {R : Type}
    (certificate : RebuildPlanClauses R) (runtimeDecision : table.SearchDecision) : Prop where
  corresponds :
    match certificate.pendingDecision with
    | .found _ destination =>
        runtimeDecision.class = .Found ∧ runtimeDecision.location = runtimeLocation destination
    | .vacant destination =>
        runtimeDecision.class = .Vacant ∧ runtimeDecision.location = runtimeLocation destination

/-- The exact input-side rebuild-plan predicate; no execution post-state occurs in its definition. -/
def RebuildPlanCorrect_R {R : Type} [ProductionRelocationKeyRep R]
    (resources : PreparedResourceClauses R)
    (planCertificate : RebuildPlanClauses R) (state : WordTable R) (pending : PendingInput)
    (adapterResult : AdapterResult R) (prepared : PublishedResult R) : Prop :=
  PreparedResources_R resources state pending adapterResult prepared ∧
  planCertificate.state = state ∧ planCertificate.pending = pending ∧
  planCertificate.plan = resources.plan ∧ planCertificate.storage = resources.storage ∧
  planCertificate.decode = resources.decode ∧
  planCertificate.addressHeld = resources.addressHeld ∧
  planCertificate.keyRep = ProductionRelocationKeyRep.keyRep

/-- Eliminate correspondence for a symbolic found decision, retaining the full runtime location. -/
theorem decisionSymbolic_found {R : Type} {certificate : RebuildPlanClauses R}
    {runtimeDecision : table.SearchDecision} (h : DecisionSymbolicCorrespondence certificate runtimeDecision)
    (entryAt : Usize) (destination : Destination)
    (hsymbolic : certificate.pendingDecision = .found entryAt destination) :
    runtimeDecision.class = .Found ∧ runtimeDecision.location = runtimeLocation destination := by
  have hc := h.corresponds
  rw [hsymbolic] at hc
  exact hc

/-- Eliminate correspondence for a symbolic vacancy, retaining flat index or spill shard/index. -/
theorem decisionSymbolic_vacant {R : Type} {certificate : RebuildPlanClauses R}
    {runtimeDecision : table.SearchDecision} (h : DecisionSymbolicCorrespondence certificate runtimeDecision)
    (destination : Destination) (hsymbolic : certificate.pendingDecision = .vacant destination) :
    runtimeDecision.class = .Vacant ∧ runtimeDecision.location = runtimeLocation destination := by
  have hc := h.corresponds
  rw [hsymbolic] at hc
  exact hc

/-- A corresponded symbolic vacancy also carries its input-side bounds and source absence. -/
theorem decisionSymbolic_vacant_valid {R : Type} {certificate : RebuildPlanClauses R}
    {runtimeDecision : table.SearchDecision} (h : DecisionSymbolicCorrespondence certificate runtimeDecision)
    (destination : Destination) (hsymbolic : certificate.pendingDecision = .vacant destination) :
    runtimeDecision.class = .Vacant ∧
      runtimeDecision.location = runtimeLocation destination ∧
      DestinationInBounds certificate.storage destination ∧
      (∀ entryAt, entryAt.val < certificate.plan.entries.length →
        certificate.assignment entryAt ≠ destination) ∧
      ∀ source sigma held value,
        Apeiron.GrowableRelocation.sourcePayload certificate.state source =
          some (sigma, held, value) →
        sigma = certificate.pending.sigma →
        certificate.decode sigma held ≠ some certificate.pending.key := by
  have hruntime := decisionSymbolic_vacant h destination hsymbolic
  have hvalid := certificate.pendingDecisionValid
  rw [hsymbolic] at hvalid
  exact ⟨hruntime.1, hruntime.2, hvalid⟩

/-- The receipt inequalities before transport through the extracted scalar capacity root. -/
def EveryObservedCapacitySufficient (receipt : table.resources.ReservationReceipt) : Prop :=
  receipt.flat.required_len.val ≤ receipt.flat.observed_capacity.val ∧
  receipt.directory.required_len.val ≤ receipt.directory.observed_capacity.val ∧
  ∀ region ∈ receipt.regions.val,
    region.slots.required_len.val ≤ region.slots.observed_capacity.val

/-- The concrete destination assigned to any plan entry. -/
theorem rebuildPlan_destination {R : Type} (certificate : RebuildPlanClauses R)
    (entryAt : Usize) (bound : entryAt.val < certificate.plan.entries.length) :
    DestinationInBounds certificate.storage (certificate.assignment entryAt) :=
  certificate.destinationInBounds entryAt bound

/-- Distinct source entries have pairwise-disjoint destinations. -/
theorem rebuildPlan_destinations_disjoint {R : Type} (certificate : RebuildPlanClauses R)
    (left right : Usize) (leftBound : left.val < certificate.plan.entries.length)
    (rightBound : right.val < certificate.plan.entries.length) (hne : left ≠ right) :
    certificate.assignment left ≠ certificate.assignment right :=
  certificate.destinationDisjoint left right leftBound rightBound hne

/-- Spill entries expose the generated symbolic prefix-vacancy equation. -/
theorem rebuildPlan_prefix_vacancy {R : Type} (certificate : RebuildPlanClauses R)
    (entryAt : Usize) (bound : entryAt.val < certificate.plan.entries.length)
    (shard index : Usize) (hdestination : certificate.assignment entryAt = .spill shard index) :
    ∃ region, (certificate.prefixStorage entryAt.val).regions.val[shard.val]? = some region ∧
      certificate.locateHeld region (certificate.entryObservation entryAt bound).sigma
        (certificate.entryObservation entryAt bound).address
        (certificate.entryObservation entryAt bound).writtenHeld = ok (.Vacant index) := by
  simpa [hdestination] using certificate.prefixVacancy entryAt bound

/-- The symbolic clone is exactly the successful clone used by extracted relocation. -/
theorem rebuildPlan_cloneHeld_binding {R : Type} (certificate : RebuildPlanClauses R)
    (held : R) :
    certificate.keyRep.corecloneCloneInst.clone held = ok (certificate.cloneHeld held) :=
  certificate.cloneHeld_eq held

/-- The symbolic address operation is the generated carrier address used by relocation. -/
theorem rebuildPlan_addressHeld_binding {R : Type} (certificate : RebuildPlanClauses R) :
    certificate.addressHeld = fun sigma held =>
      certificate.keyRep.address_held certificate.state.home sigma held :=
  certificate.addressHeld_eq

/-- The symbolic spill search is the generated `spill_locate_held` call at width 16. -/
theorem rebuildPlan_locateHeld_binding {R : Type} (certificate : RebuildPlanClauses R) :
    certificate.locateHeld = fun region sigma address held =>
      table.probe.spill_locate_held certificate.keyRep region sigma address held 16#usize :=
  certificate.locateHeld_eq

/-- A correct plan exposes the production-dictionary restriction used by the extracted roots. -/
theorem rebuildPlanCorrect_productionKeyRep {R : Type}
    [ProductionRelocationKeyRep R]
    {resources : PreparedResourceClauses R} {certificate : RebuildPlanClauses R}
    {state : WordTable R} {pending : PendingInput} {adapterResult : AdapterResult R}
    {prepared : PublishedResult R}
    (h : RebuildPlanCorrect_R resources certificate state pending adapterResult prepared) :
    certificate.keyRep = ProductionRelocationKeyRep.keyRep := h.2.2.2.2.2.2.2

/-- Stored plans use the exact dictionary instantiated by `word_stored_relocate_entry`. -/
theorem rebuildPlan_stored_keyRep
    {resources : PreparedResourceClauses (table.keyrep.Stored I64)}
    {certificate : RebuildPlanClauses (table.keyrep.Stored I64)}
    {state : WordTable (table.keyrep.Stored I64)} {pending : PendingInput}
    {adapterResult : AdapterResult (table.keyrep.Stored I64)}
    {prepared : PublishedResult (table.keyrep.Stored I64)}
    (h : RebuildPlanCorrect_R resources certificate state pending adapterResult prepared) :
    certificate.keyRep = storedRelocationKeyRep :=
  rebuildPlanCorrect_productionKeyRep h

/-- Regenerating plans use the exact dictionary instantiated by `word_regen_relocate_entry`. -/
theorem rebuildPlan_regen_keyRep
    {resources : PreparedResourceClauses
      (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {certificate : RebuildPlanClauses
      (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {state : WordTable (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {pending : PendingInput}
    {adapterResult : AdapterResult (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {prepared : PublishedResult (table.keyrep.Regen table.keyrep.I64WordCodec)}
    (h : RebuildPlanCorrect_R resources certificate state pending adapterResult prepared) :
    certificate.keyRep = regenRelocationKeyRep :=
  rebuildPlanCorrect_productionKeyRep h

/-- Stored prefix vacancy is computed by the actual extracted spill locator. -/
theorem rebuildPlan_stored_locateHeld_binding
    {resources : PreparedResourceClauses (table.keyrep.Stored I64)}
    {certificate : RebuildPlanClauses (table.keyrep.Stored I64)}
    {state : WordTable (table.keyrep.Stored I64)} {pending : PendingInput}
    {adapterResult : AdapterResult (table.keyrep.Stored I64)}
    {prepared : PublishedResult (table.keyrep.Stored I64)}
    (h : RebuildPlanCorrect_R resources certificate state pending adapterResult prepared) :
    certificate.locateHeld = fun region sigma address held =>
      table.probe.spill_locate_held storedRelocationKeyRep region sigma address held 16#usize := by
  rw [rebuildPlan_locateHeld_binding certificate, rebuildPlan_stored_keyRep h]

/-- Regenerating prefix vacancy is computed by the actual extracted spill locator. -/
theorem rebuildPlan_regen_locateHeld_binding
    {resources : PreparedResourceClauses
      (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {certificate : RebuildPlanClauses
      (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {state : WordTable (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {pending : PendingInput}
    {adapterResult : AdapterResult (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {prepared : PublishedResult (table.keyrep.Regen table.keyrep.I64WordCodec)}
    (h : RebuildPlanCorrect_R resources certificate state pending adapterResult prepared) :
    certificate.locateHeld = fun region sigma address held =>
      table.probe.spill_locate_held regenRelocationKeyRep region sigma address held 16#usize := by
  rw [rebuildPlan_locateHeld_binding certificate, rebuildPlan_regen_keyRep h]

/-- All symbolic Stored operations are the operations called by the extracted relocation root. -/
theorem rebuildPlan_stored_operations_binding
    {resources : PreparedResourceClauses (table.keyrep.Stored I64)}
    {certificate : RebuildPlanClauses (table.keyrep.Stored I64)}
    {state : WordTable (table.keyrep.Stored I64)} {pending : PendingInput}
    {adapterResult : AdapterResult (table.keyrep.Stored I64)}
    {prepared : PublishedResult (table.keyrep.Stored I64)}
    (h : RebuildPlanCorrect_R resources certificate state pending adapterResult prepared) :
    (∀ held, storedRelocationKeyRep.corecloneCloneInst.clone held =
      ok (certificate.cloneHeld held)) ∧
    (certificate.addressHeld = fun sigma held =>
      storedRelocationKeyRep.address_held certificate.state.home sigma held) ∧
    certificate.locateHeld = fun region sigma address held =>
      table.probe.spill_locate_held storedRelocationKeyRep region sigma address held 16#usize := by
  have hkeyRep := rebuildPlan_stored_keyRep h
  refine ⟨?_, ?_, rebuildPlan_stored_locateHeld_binding h⟩
  · intro held
    rw [← hkeyRep]
    exact rebuildPlan_cloneHeld_binding certificate held
  · rw [rebuildPlan_addressHeld_binding certificate, hkeyRep]

/-- All symbolic Regen operations are the operations called by the extracted relocation root. -/
theorem rebuildPlan_regen_operations_binding
    {resources : PreparedResourceClauses
      (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {certificate : RebuildPlanClauses
      (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {state : WordTable (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {pending : PendingInput}
    {adapterResult : AdapterResult (table.keyrep.Regen table.keyrep.I64WordCodec)}
    {prepared : PublishedResult (table.keyrep.Regen table.keyrep.I64WordCodec)}
    (h : RebuildPlanCorrect_R resources certificate state pending adapterResult prepared) :
    (∀ held, regenRelocationKeyRep.corecloneCloneInst.clone held =
      ok (certificate.cloneHeld held)) ∧
    (certificate.addressHeld = fun sigma held =>
      regenRelocationKeyRep.address_held certificate.state.home sigma held) ∧
    certificate.locateHeld = fun region sigma address held =>
      table.probe.spill_locate_held regenRelocationKeyRep region sigma address held 16#usize := by
  have hkeyRep := rebuildPlan_regen_keyRep h
  refine ⟨?_, ?_, rebuildPlan_regen_locateHeld_binding h⟩
  · intro held
    rw [← hkeyRep]
    exact rebuildPlan_cloneHeld_binding certificate held
  · rw [rebuildPlan_addressHeld_binding certificate, hkeyRep]

/-- Each plan index exposes its identity/value/codec/address observation certificate. -/
def rebuildPlan_entry_observation {R : Type} (certificate : RebuildPlanClauses R)
    (entryAt : Usize) (bound : entryAt.val < certificate.plan.entries.length) :
    EntryObservationCertificate certificate.state certificate.decode certificate.cloneHeld
      certificate.addressHeld certificate.plan entryAt (certificate.assignment entryAt) :=
  certificate.entryObservation entryAt bound

/-- Distinct plan entries cannot name the same decoded `(Sigma, key)` logical key. -/
theorem rebuildPlan_logicalKey_injective {R : Type} (certificate : RebuildPlanClauses R)
    (left right : Usize) (leftBound : left.val < certificate.plan.entries.length)
    (rightBound : right.val < certificate.plan.entries.length)
    (heq : ((certificate.entryObservation left leftBound).sigma,
        (certificate.entryObservation left leftBound).key) =
      ((certificate.entryObservation right rightBound).sigma,
        (certificate.entryObservation right rightBound).key)) : left = right :=
  certificate.logicalKeyInjective left right leftBound rightBound heq

/-- Source/destination coverage is available without assuming a relocation post-state. -/
theorem rebuildPlan_coverage {R : Type} (certificate : RebuildPlanClauses R) :
    PlannedCoverage certificate.state certificate.plan certificate.storage certificate.assignment :=
  certificate.coverage

/-- Planned direct-coordinate coverage is available without assuming resulting `DirectCover`. -/
theorem rebuildPlan_directCover {R : Type} (certificate : RebuildPlanClauses R) :
    PlannedDirectCover certificate.state certificate.decode certificate.cloneHeld
      certificate.addressHeld certificate.plan certificate.assignment := certificate.directCover

/-- The committed live set is interpreted on the plan's target lane geometry. -/
theorem rebuildPlan_targetLiveSet {R : Type} (certificate : RebuildPlanClauses R) :
    Apeiron.GrowableRepresentation.LiveSetEqual
      (Apeiron.GrowableRefinement.rebuiltView certificate.state
        (certificate.prefixStorage certificate.plan.entries.length))
      certificate.decode certificate.committedLive := certificate.targetGeometry.liveSet

/-- The symbolic target carries the lane extent selected by the rebuild plan. -/
theorem rebuildPlan_targetLanes {R : Type} (certificate : RebuildPlanClauses R) :
    (certificate.prefixStorage certificate.plan.entries.length).resident.lanes =
      certificate.plan.lanes := certificate.targetGeometry.lanes

/-- Decoded logical keys are unique in the fully written symbolic target. -/
theorem rebuildPlan_targetUnique {R : Type} (certificate : RebuildPlanClauses R) :
    Apeiron.GrowableRepresentation.UniqueRepresentation
      (Apeiron.GrowableRefinement.rebuiltView certificate.state
        (certificate.prefixStorage certificate.plan.entries.length)) certificate.decode :=
  certificate.targetGeometry.unique

/-- Occupancy-authoritative target slots exactly cover decoded entries. -/
theorem rebuildPlan_targetCoverage {R : Type} (certificate : RebuildPlanClauses R) :
    Apeiron.GrowableRepresentation.PhysicalCoverage
      (Apeiron.GrowableRefinement.rebuiltView certificate.state
        (certificate.prefixStorage certificate.plan.entries.length)) certificate.decode :=
  certificate.targetGeometry.coverage

/-- Every symbolic target entry agrees with generated target addressing. -/
theorem rebuildPlan_targetAddress {R : Type} (certificate : RebuildPlanClauses R) :
    Apeiron.GrowableRepresentation.AddressCorrect
      (Apeiron.GrowableRefinement.rebuiltView certificate.state
        (certificate.prefixStorage certificate.plan.entries.length)) certificate.decode :=
  certificate.targetGeometry.address

/-- The post-rebuild direct function is indexed with the plan's target lane extent. -/
theorem rebuildPlan_targetDirectIndex {R : Type} (certificate : RebuildPlanClauses R) :
    Apeiron.GrowableRepresentation.DirectIndexCorrect
      (Apeiron.GrowableRefinement.rebuiltView certificate.state
        (certificate.prefixStorage certificate.plan.entries.length))
      certificate.directAfter := certificate.targetGeometry.directIndex

/-- The symbolic target supplies direct-empty coverage for its own geometry. -/
theorem rebuildPlan_targetDirectCover {R : Type} (certificate : RebuildPlanClauses R) :
    Apeiron.GrowableSearch.DirectCover
      (Apeiron.GrowableRefinement.rebuiltView certificate.state
        (certificate.prefixStorage certificate.plan.entries.length))
      certificate.decode certificate.directAfter := certificate.targetGeometry.directCover

/-- The symbolic target has the flat extent and spill-directory shape required by `RepR`. -/
theorem rebuildPlan_targetDirectory {R : Type} (certificate : RebuildPlanClauses R) :
    Apeiron.GrowableRepresentation.SpillDirectoryConsistent
      (Apeiron.GrowableRefinement.rebuiltView certificate.state
        (certificate.prefixStorage certificate.plan.entries.length)) :=
  certificate.targetGeometry.directory

/-- The symbolic target length counts the committed live set at the target geometry. -/
theorem rebuildPlan_targetLength {R : Type} (certificate : RebuildPlanClauses R) :
    Apeiron.GrowableRepresentation.LengthCorrect
      (Apeiron.GrowableRefinement.rebuiltView certificate.state
        (certificate.prefixStorage certificate.plan.entries.length)) certificate.committedLive :=
  certificate.targetGeometry.length

/-- Prepared storage exposes its concrete initial vacancy certificate. -/
theorem preparedResources_materializedVacant {R : Type} (certificate : PreparedResourceClauses R) :
    MaterializedVacant certificate.storage := certificate.materializedVacant

/-- The relocation root's explicit side condition excludes allocation and merge calls. -/
theorem preparedResources_zero_allocation_merge {R : Type}
    (certificate : PreparedResourceClauses R) :
    certificate.relocationTrace.allocationCalls = 0 ∧
      certificate.relocationTrace.mergeCalls = 0 := by
  exact ⟨certificate.noRelocationEffects.1, certificate.noRelocationEffects.2.2⟩

/-!
`ResourceAdapterSimulates` remains an explicit trust-boundary premise.  Its prepared constructor now
carries concrete certificates whose indexed generated values must equal the published observation.
The refusal constructor records unchanged committed state and zero clone, merge, and relocation calls.
-/

inductive ResourceAdapterSimulates {R : Type} [ProductionRelocationKeyRep R]
    (resources : PreparedResourceClauses R) (planCertificate : RebuildPlanClauses R)
    (publish : AdapterResult R → Result (PublishedResult R))
    (state : WordTable R) (pending : PendingInput) (committedAfter : WordTable R)
    (cloneCalls mergeCalls relocationCalls : Nat) :
    AdapterResult R → PublishedResult R → Prop
  | prepared
      (hstate : resources.state = state) (hpending : resources.pending = pending)
      (hplanState : planCertificate.state = state)
      (hplanPending : planCertificate.pending = pending)
      (hplan : planCertificate.plan = resources.plan)
      (hstorage : planCertificate.storage = resources.storage)
      (hdecode : planCertificate.decode = resources.decode)
      (haddress : planCertificate.addressHeld = resources.addressHeld)
      (hkeyRep : planCertificate.keyRep = ProductionRelocationKeyRep.keyRep)
      (hcapacity : EveryObservedCapacitySufficient resources.receipt)
      (hpublish : publish (.Ok (resources.plan, resources.storage,
        { reservation := resources.receipt, locator := resources.locatorReceipt,
          pending_held := resources.held })) =
        ok (.Prepared resources.plan resources.storage
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held })) :
      ResourceAdapterSimulates resources planCertificate publish state pending committedAfter
        cloneCalls mergeCalls relocationCalls
        (.Ok (resources.plan, resources.storage,
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }))
        (.Prepared resources.plan resources.storage
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held })
  | refused (error : table.StorageError)
      (hpublish : publish (.Err error) = ok (.Refused error))
      (hstutter : committedAfter = state)
      (hclone : cloneCalls = 0) (hmerge : mergeCalls = 0)
      (hrelocate : relocationCalls = 0) :
      ResourceAdapterSimulates resources planCertificate publish state pending committedAfter
        cloneCalls mergeCalls relocationCalls (.Err error) (.Refused error)

/--
Refusal carries transactional stutter and zero clone/merge/relocation counters.  This eliminator is
the no-clone-on-`Err` obligation; it applies only to the refused branch and supplies no fact about a
prepared-success execution.
-/
theorem resourceAdapter_refusal_stutter {R : Type} [ProductionRelocationKeyRep R]
    {resources : PreparedResourceClauses R} {planCertificate : RebuildPlanClauses R}
    {publish : AdapterResult R → Result (PublishedResult R)} {state committedAfter : WordTable R}
    {pending : PendingInput} {cloneCalls mergeCalls relocationCalls : Nat}
    {error : table.StorageError}
    (h : ResourceAdapterSimulates resources planCertificate publish state pending committedAfter
      cloneCalls mergeCalls relocationCalls (.Err error) (.Refused error)) :
    committedAfter = state ∧ cloneCalls = 0 ∧ mergeCalls = 0 ∧ relocationCalls = 0 := by
  cases h with
  | refused _ _ hstutter hclone hmerge hrelocate =>
      exact ⟨hstutter, hclone, hmerge, hrelocate⟩

/-- The extracted scalar capacity check accepts exactly `required_len ≤ observed_capacity`. -/
theorem word_capacity_bound_holds_iff (required observed : Usize) :
    word_capacity_bound_holds required observed = ok true ↔ required.val ≤ observed.val := by
  simp [word_capacity_bound_holds, table.resources.capacity_bound_holds]

/-- Every receipt inequality is observed by the actual extracted scalar comparison. -/
theorem everyScalarCapacityCheck_of_sufficient
    (receipt : table.resources.ReservationReceipt)
    (h : EveryObservedCapacitySufficient receipt) :
    EveryScalarCapacityCheck_R receipt := by
  rcases h with ⟨hflat, hdirectory, hregions⟩
  refine ⟨?_, ?_, ?_⟩
  · exact (word_capacity_bound_holds_iff _ _).2 hflat
  · exact (word_capacity_bound_holds_iff _ _).2 hdirectory
  · intro region hregion
    exact (word_capacity_bound_holds_iff _ _).2 (hregions region hregion)

/-- Stored publication moves the exact receipt and pending carrier into the published envelope. -/
theorem word_stored_publish_prepared_carries
    (plan : table.RebuildPlan)
    (storage : table.RebuildReserved (table.keyrep.Stored I64) I64)
    (receipt : table.resources.ReservationReceipt)
    (locator : table.resources.LocatorReservationReceipt) (held : table.keyrep.Stored I64) :
    word_stored_publish_prepared
        (.Ok (plan, storage, { reservation := receipt, locator, pending_held := held })) =
      ok (.Prepared plan storage { reservation := receipt, locator, pending_held := held }) := by
  rfl

/-- Regenerating publication moves the exact receipt and pending carrier into the envelope. -/
theorem word_regen_publish_prepared_carries
    (plan : table.RebuildPlan)
    (storage : table.RebuildReserved (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (receipt : table.resources.ReservationReceipt)
    (locator : table.resources.LocatorReservationReceipt)
    (held : table.keyrep.Regen table.keyrep.I64WordCodec) :
    word_regen_publish_prepared
        (.Ok (plan, storage, { reservation := receipt, locator, pending_held := held })) =
      ok (.Prepared plan storage { reservation := receipt, locator, pending_held := held }) := by
  rfl

/-- Faithful publication transports one concrete prepared-resource certificate. -/
theorem preparedResources_of_publication {R : Type} (certificate : PreparedResourceClauses R)
    (hcapacity : EveryScalarCapacityCheck_R certificate.receipt) :
    PreparedResources_R certificate certificate.state certificate.pending
      (.Ok (certificate.plan, certificate.storage,
        { reservation := certificate.receipt, locator := certificate.locatorReceipt,
          pending_held := certificate.held }))
      (.Prepared certificate.plan certificate.storage
        { reservation := certificate.receipt, locator := certificate.locatorReceipt,
          pending_held := certificate.held }) := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, hcapacity⟩

/-- The Stored publication has concrete resource and plan witnesses under adapter simulation. -/
theorem word_stored_reachable_publication_witness
    (resources : PreparedResourceClauses (table.keyrep.Stored I64))
    (planCertificate : RebuildPlanClauses (table.keyrep.Stored I64))
    (state committedAfter : WordTable (table.keyrep.Stored I64)) (pending : PendingInput)
    (cloneCalls mergeCalls relocationCalls : Nat)
    (hsim : ResourceAdapterSimulates resources planCertificate word_stored_publish_prepared
      state pending committedAfter cloneCalls mergeCalls relocationCalls
      (.Ok (resources.plan, resources.storage,
        { reservation := resources.receipt, locator := resources.locatorReceipt,
          pending_held := resources.held }))
      (.Prepared resources.plan resources.storage
        { reservation := resources.receipt, locator := resources.locatorReceipt,
          pending_held := resources.held })) :
    PreparedResources_R resources state pending
        (.Ok (resources.plan, resources.storage,
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }))
        (.Prepared resources.plan resources.storage
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }) ∧
      RebuildPlanCorrect_R resources planCertificate state pending
        (.Ok (resources.plan, resources.storage,
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }))
        (.Prepared resources.plan resources.storage
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }) := by
  cases hsim with
  | prepared hstate hpending hplanState hplanPending hplan hstorage hdecode haddress hkeyRep hcapacity
      hpublish =>
    have hchecks := everyScalarCapacityCheck_of_sufficient resources.receipt hcapacity
    have hpublication :
        table.resources.publish_prepared
            (.Ok (resources.plan, resources.storage,
              { reservation := resources.receipt, locator := resources.locatorReceipt,
                pending_held := resources.held })) =
          ok (.Prepared resources.plan resources.storage
            { reservation := resources.receipt, locator := resources.locatorReceipt,
              pending_held := resources.held }) := by
      simpa [word_stored_publish_prepared] using hpublish
    have hresources : PreparedResources_R resources state pending
        (.Ok (resources.plan, resources.storage,
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }))
        (.Prepared resources.plan resources.storage
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }) :=
      ⟨hstate, hpending, rfl, hpublication, rfl, hchecks⟩
    exact ⟨hresources, hresources, hplanState, hplanPending, hplan, hstorage, hdecode, haddress,
      hkeyRep⟩

/-- The regenerating production publication has the analogous conditional reachable witness. -/
theorem word_regen_reachable_publication_witness
    (resources : PreparedResourceClauses
      (table.keyrep.Regen table.keyrep.I64WordCodec))
    (planCertificate : RebuildPlanClauses
      (table.keyrep.Regen table.keyrep.I64WordCodec))
    (state committedAfter : WordTable
      (table.keyrep.Regen table.keyrep.I64WordCodec)) (pending : PendingInput)
    (cloneCalls mergeCalls relocationCalls : Nat)
    (hsim : ResourceAdapterSimulates resources planCertificate word_regen_publish_prepared
      state pending committedAfter cloneCalls mergeCalls relocationCalls
      (.Ok (resources.plan, resources.storage,
        { reservation := resources.receipt, locator := resources.locatorReceipt,
          pending_held := resources.held }))
      (.Prepared resources.plan resources.storage
        { reservation := resources.receipt, locator := resources.locatorReceipt,
          pending_held := resources.held })) :
    PreparedResources_R resources state pending
        (.Ok (resources.plan, resources.storage,
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }))
        (.Prepared resources.plan resources.storage
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }) ∧
      RebuildPlanCorrect_R resources planCertificate state pending
        (.Ok (resources.plan, resources.storage,
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }))
        (.Prepared resources.plan resources.storage
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }) := by
  cases hsim with
  | prepared hstate hpending hplanState hplanPending hplan hstorage hdecode haddress hkeyRep hcapacity
      hpublish =>
    have hchecks := everyScalarCapacityCheck_of_sufficient resources.receipt hcapacity
    have hpublication :
        table.resources.publish_prepared
            (.Ok (resources.plan, resources.storage,
              { reservation := resources.receipt, locator := resources.locatorReceipt,
                pending_held := resources.held })) =
          ok (.Prepared resources.plan resources.storage
            { reservation := resources.receipt, locator := resources.locatorReceipt,
              pending_held := resources.held }) := by
      simpa [word_regen_publish_prepared] using hpublish
    have hresources : PreparedResources_R resources state pending
        (.Ok (resources.plan, resources.storage,
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }))
        (.Prepared resources.plan resources.storage
          { reservation := resources.receipt, locator := resources.locatorReceipt,
            pending_held := resources.held }) :=
      ⟨hstate, hpending, rfl, hpublication, rfl, hchecks⟩
    exact ⟨hresources, hresources, hplanState, hplanPending, hplan, hstorage, hdecode, haddress,
      hkeyRep⟩

end Apeiron.GrowableResources
