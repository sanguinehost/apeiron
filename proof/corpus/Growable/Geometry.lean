import ApeironWord
import Apeiron.Table.Basic

/-!
# Geometry of the extracted prepared core

The prepared core receives storage whose allocation and sizing have already succeeded.  This module
states that resource premise with natural-number arithmetic and proves the bounds used by the
generated `TableAddress`, `Region`, and prepared shard bodies.  The planner, reservation calls, and
allocator are outside this module.

The flat area is shard-major.  For a certified `U32` word, its low byte selects one of 256 shards
and the remaining high bits select a lane.  Spill storage consists of power-of-two groups of `N`
lanes.  These facts are physical premises for representing the total canonical
`Apeiron.Table.HMap`; no table operation is refined here.
-/

namespace Apeiron.GrowableGeometry

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- The number of flat shards in the extracted table layout. -/
def shardCount : Nat := 256

/-- The largest resident-lane exponent admitted by the source resource adapter. -/
def maxLaneBits : Nat := 24

/-- The unsigned natural represented by a certified extracted direct word. -/
def directWordNat (word : U32) : Nat := word.val

/-- The low byte of a direct word is its flat shard. -/
def directShard (word : U32) : Nat :=
  (UScalar.cast .Usize word &&& 255#usize).val

/-- The high 24 bits of a direct word are its flat lane. -/
def directLane (word : U32) : Nat := directWordNat word / shardCount

/-- The shard-major flat index is `shard * lanes + lane`. -/
def flatIndex (word : U32) (lanes : Nat) : Nat :=
  directShard word * lanes + directLane word

/-- A generated address admits a flat coordinate only with its exact word and an in-range lane. -/
def DirectAdmission (address : table.TableAddress) (word : U32) (lanes : Nat) : Prop :=
  address.direct_word = some word ∧ directLane word < lanes

/-- The prepared index operation rejects missing and out-of-extent direct words. -/
def preparedFlatIndex (address : table.TableAddress) (lanes : Nat) : Option Nat :=
  match address.direct_word with
  | none => none
  | some word => if directLane word < lanes then some (flatIndex word lanes) else none

/-- The direct-word branch of the real extracted shard function returns the certified low byte. -/
theorem extracted_shard_of_direct_word (address : table.TableAddress) (word : U32)
    (hword : address.direct_word = some word) :
    table.shard address ⦃ shard => shard.val = directShard word ⦄ := by
  unfold table.shard
  rw [hword]
  step
  simp only [directShard]
  rw [shard_post]

/-- A certified in-range word determines the prepared flat index independently of placement. -/
theorem preparedFlatIndex_of_admission (address : table.TableAddress) (word : U32) (lanes : Nat)
    (h : DirectAdmission address word lanes) :
    preparedFlatIndex address lanes = some (flatIndex word lanes) := by
  rcases h with ⟨hword, hlane⟩
  simp [preparedFlatIndex, hword, hlane]

/-- Every admitted flat index is below the checked `256 * lanes` allocation. -/
theorem flatIndex_lt (word : U32) (lanes : Nat) (hlane : directLane word < lanes) :
    flatIndex word lanes < shardCount * lanes := by
  have hshard : directShard word < shardCount := by
    simp only [directShard, shardCount]
    simp only [UScalar.cast, UScalar.val]
    refine lt_of_le_of_lt Nat.and_le_right ?_
    cases System.Platform.numBits_eq <;> simp_all
  calc
    flatIndex word lanes < directShard word * lanes + lanes := by
      exact Nat.add_lt_add_left hlane _
    _ = (directShard word + 1) * lanes := by rw [Nat.add_mul, one_mul]
    _ ≤ shardCount * lanes := Nat.mul_le_mul_right lanes (Nat.succ_le_iff.mpr hshard)

/-- Every `U32` lane is below `2^24`. -/
theorem directLane_lt_two_pow_24 (word : U32) : directLane word < 2 ^ maxLaneBits := by
  have hword : word.val < 2 ^ 32 := by
    simpa [UScalar.size, UScalarTy.numBits] using word.hBounds
  simp only [directLane, directWordNat, shardCount, maxLaneBits]
  norm_num at hword ⊢
  omega

/-- A resident extent `2^laneBits`, with `laneBits ≤ 24`, fits the extracted 64-bit `usize`. -/
theorem laneExtent_representable (laneBits : Nat) (hbits : laneBits ≤ maxLaneBits) :
    2 ^ laneBits ≤ 2 ^ maxLaneBits ∧ 2 ^ laneBits < 2 ^ 64 := by
  have hmono : 2 ^ laneBits ≤ (2 : Nat) ^ maxLaneBits :=
    pow_le_pow_right' (by omega) hbits
  constructor
  · exact hmono
  · exact lt_of_le_of_lt hmono (by norm_num [maxLaneBits])

/-- Ceiling division used by grouped spill sizing; positive divisors are an explicit premise. -/
def ceilDiv (numerator width : Nat) : Nat := (numerator + width - 1) / width

/-- Prepared spill widths are nonzero powers of two no larger than the source limit 64. -/
def AdmissibleGroupWidth (width : Nat) : Prop :=
  0 < width ∧ width ≤ 64 ∧ ∃ exponent, width = 2 ^ exponent

/-- Bounded doubling implements the checked power-of-two ceiling. -/
def checkedPow2CeilLoop : Nat → Nat → Nat → Option Nat
  | 0, current, target => if target ≤ current then some current else none
  | fuel + 1, current, target =>
      if target ≤ current then some current else checkedPow2CeilLoop fuel (2 * current) target

/-- The adapter permits at most 64 checked doublings from one. -/
def checkedPow2Ceil (target : Nat) : Option Nat := checkedPow2CeilLoop 64 1 target

/-- A successful checked power-of-two ceiling reaches its target. -/
theorem checkedPow2CeilLoop_ge (fuel current target groups : Nat)
    (h : checkedPow2CeilLoop fuel current target = some groups) : target ≤ groups := by
  induction fuel generalizing current with
  | zero =>
      simp only [checkedPow2CeilLoop] at h
      split at h <;> simp_all
  | succ fuel ih =>
      simp only [checkedPow2CeilLoop] at h
      split at h
      · simp_all
      · exact ih (2 * current) h

/-- A successful bounded doubling result is the initial value times at most `2^fuel`. -/
theorem checkedPow2CeilLoop_power (fuel current target groups : Nat)
    (h : checkedPow2CeilLoop fuel current target = some groups) :
    ∃ exponent, exponent ≤ fuel ∧ groups = current * 2 ^ exponent := by
  induction fuel generalizing current with
  | zero =>
      simp only [checkedPow2CeilLoop] at h
      split at h
      · simp_all
      · simp_all
  | succ fuel ih =>
      simp only [checkedPow2CeilLoop] at h
      split at h
      · refine ⟨0, Nat.zero_le _, ?_⟩
        simpa using (Option.some.inj h).symm
      · obtain ⟨exponent, hexponent, hgroups⟩ := ih (2 * current) h
        refine ⟨exponent + 1, Nat.add_le_add_right hexponent 1, ?_⟩
        rw [hgroups, pow_succ]
        ac_rfl

/-- A successful checked ceiling is a power of two at least as large as its target. -/
theorem checkedPow2Ceil_spec (target groups : Nat) (h : checkedPow2Ceil target = some groups) :
    target ≤ groups ∧ ∃ exponent, exponent ≤ 64 ∧ groups = 2 ^ exponent := by
  constructor
  · exact checkedPow2CeilLoop_ge 64 1 target groups h
  · obtain ⟨exponent, hexponent, hgroups⟩ :=
      checkedPow2CeilLoop_power 64 1 target groups h
    exact ⟨exponent, hexponent, by simpa using hgroups⟩

/-- Checked multiplication at the extracted 64-bit `usize` ceiling. -/
def checkedMul (left right : Nat) : Option Nat :=
  if left * right < 2 ^ 64 then some (left * right) else none

/-- Successful checked multiplication returns the mathematical product and its representability. -/
theorem checkedMul_eq_some (left right result : Nat) :
    checkedMul left right = some result ↔ left * right < 2 ^ 64 ∧ result = left * right := by
  simp [checkedMul, eq_comm]

/-- The successful grouped-region sizing result supplied to the prepared core. -/
structure CheckedSizing where
  groups : Nat
  slots : Nat
  deriving DecidableEq

/-- Empty loads allocate no tag or payload slots; positive loads use the checked sizing formula. -/
def checkedSizing (load width : Nat) : Option CheckedSizing :=
  if load = 0 then
    some { groups := 0, slots := 0 }
  else if width = 0 then
    none
  else do
    let twiceLoad ← checkedMul 2 load
    let groups ← checkedPow2Ceil (max 2 (ceilDiv twiceLoad width))
    let slots ← checkedMul groups width
    some { groups, slots }

/-- A successful nonempty sizing is exactly the checked formula supplied to the prepared core. -/
theorem checkedSizing_positive_spec (load width : Nat) (plan : CheckedSizing)
    (hload : 0 < load) (h : checkedSizing load width = some plan) :
    0 < width ∧ ∃ twiceLoad groups,
      checkedMul 2 load = some twiceLoad ∧
      checkedPow2Ceil (max 2 (ceilDiv twiceLoad width)) = some groups ∧
      checkedMul groups width = some plan.slots ∧ plan.groups = groups := by
  unfold checkedSizing at h
  simp only [if_neg (Nat.ne_of_gt hload)] at h
  split at h
  · simp_all
  · rename_i hwidth
    have hwidthPos : 0 < width := Nat.pos_of_ne_zero hwidth
    constructor
    · exact hwidthPos
    cases htwice : checkedMul 2 load with
    | none => simp [htwice] at h
    | some twiceLoad =>
        cases hgroups : checkedPow2Ceil (max 2 (ceilDiv twiceLoad width)) with
        | none => simp [htwice, hgroups] at h
        | some groups =>
            cases hslots : checkedMul groups width with
            | none => simp [htwice, hgroups, hslots] at h
            | some slots =>
                simp [htwice, hgroups, hslots] at h
                subst plan
                exact ⟨twiceLoad, groups, rfl, hgroups, hslots, rfl⟩

/-- Positive admitted loads use `pow2Ceil(max 2 (ceilDiv (2 * load) width))` groups. -/
theorem checkedSizing_matches_formula (load width : Nat) (plan : CheckedSizing)
    (hload : 0 < load) (h : checkedSizing load width = some plan) :
    0 < width ∧ ∃ groups,
      checkedPow2Ceil (max 2 (ceilDiv (2 * load) width)) = some groups ∧
      plan.groups = groups ∧ plan.slots = groups * width ∧
      2 * load < 2 ^ 64 ∧ groups * width < 2 ^ 64 := by
  obtain ⟨hwidth, twiceLoad, groups, htwice, hgroups, hslots, hplanGroups⟩ :=
    checkedSizing_positive_spec load width plan hload h
  have htwiceSpec := (checkedMul_eq_some 2 load twiceLoad).mp htwice
  have hslotsSpec := (checkedMul_eq_some groups width plan.slots).mp hslots
  rw [htwiceSpec.2] at hgroups
  refine ⟨hwidth, groups, hgroups, hplanGroups, hslotsSpec.2, htwiceSpec.1, ?_⟩
  exact hslotsSpec.1

/-- Admitted widths and successful sizing give divisible slots and a power-of-two group count. -/
theorem checkedSizing_group_geometry (load width : Nat) (plan : CheckedSizing)
    (hload : 0 < load) (hwidth : AdmissibleGroupWidth width)
    (h : checkedSizing load width = some plan) :
    AdmissibleGroupWidth width ∧ 2 ≤ plan.groups ∧
      (∃ exponent, exponent ≤ 64 ∧ plan.groups = 2 ^ exponent) ∧
        plan.slots = plan.groups * width := by
  obtain ⟨_, groups, hgroups, hplanGroups, hslots, _, _⟩ :=
    checkedSizing_matches_formula load width plan hload h
  obtain ⟨htarget, hpower⟩ :=
    checkedPow2Ceil_spec (max 2 (ceilDiv (2 * load) width)) groups hgroups
  subst groups
  exact ⟨hwidth, (le_max_left 2 _).trans htarget, hpower, hslots⟩

/-- Checked sizing of an empty load allocates no grouped region. -/
theorem empty_region_allocates_nothing (width : Nat) :
    checkedSizing 0 width = some { groups := 0, slots := 0 } := by
  simp [checkedSizing]

/-- The source density test compares capacity with twice the distinct admitted-word count. -/
def densityAdmits (capacity distinctWords : Nat) : Bool :=
  decide (capacity ≤ max shardCount (2 * distinctWords))

/-- The Boolean density test is exact. -/
theorem densityAdmits_eq_true (capacity distinctWords : Nat) :
    densityAdmits capacity distinctWords = true ↔
      capacity ≤ max shardCount (2 * distinctWords) := by
  simp [densityAdmits]

/-- Increasing the distinct admitted-word count cannot turn an admitted density into a rejection. -/
theorem densityAdmits_mono (capacity lower upper : Nat) (hcount : lower ≤ upper)
    (h : densityAdmits capacity lower = true) :
    densityAdmits capacity upper = true := by
  rw [densityAdmits_eq_true] at h ⊢
  exact h.trans (max_le_max (le_refl shardCount) (Nat.mul_le_mul_left 2 hcount))

/-- A prepared destination is either flat or one lane of a grouped spill region. -/
inductive Destination where
  | flat (word : U32)
  | spill (group lane : Nat)

/-- Capacity and extent premises certified by the unextracted resource adapter. -/
structure PreparedGeometry where
  lanes : Nat
  laneBits : Nat
  groups : Nat
  width : Nat
  flatCapacity : Nat
  spillCapacity : Nat
  lanes_eq : lanes = 2 ^ laneBits
  laneBits_le : laneBits ≤ maxLaneBits
  width_admissible : AdmissibleGroupWidth width
  groups_min : 2 ≤ groups
  groups_power : ∃ exponent, groups = 2 ^ exponent
  flatCapacity_eq : flatCapacity = shardCount * lanes
  spillCapacity_eq : spillCapacity = groups * width

/-- Admission of a destination records the exact flat word or grouped coordinates. -/
def DestinationAdmitted (plan : PreparedGeometry) : Destination → Prop
  | .flat word => directLane word < plan.lanes
  | .spill group lane => group < plan.groups ∧ lane < plan.width

/-- Prepared destinations use the same shard-major or group-major arithmetic as their allocation. -/
def destinationIndex (plan : PreparedGeometry) : Destination → Nat
  | .flat word => flatIndex word plan.lanes
  | .spill group lane => group * plan.width + lane

/-- The allocated bound corresponding to a prepared destination. -/
def destinationCapacity (plan : PreparedGeometry) : Destination → Nat
  | .flat _ => plan.flatCapacity
  | .spill _ _ => plan.spillCapacity

/-- Every destination certified by the prepared resource plan is in its allocated storage. -/
theorem prepared_destination_in_bounds (plan : PreparedGeometry) (destination : Destination)
    (hadmitted : DestinationAdmitted plan destination) :
    destinationIndex plan destination < destinationCapacity plan destination := by
  cases destination with
  | flat word =>
      simp only [destinationIndex, destinationCapacity]
      rw [plan.flatCapacity_eq]
      apply flatIndex_lt word plan.lanes
      exact hadmitted
  | spill group lane =>
      simp only [DestinationAdmitted] at hadmitted
      simp only [destinationIndex, destinationCapacity]
      rw [plan.spillCapacity_eq]
      calc
        group * plan.width + lane < group * plan.width + plan.width := by
          exact Nat.add_lt_add_left hadmitted.2 _
        _ = (group + 1) * plan.width := by rw [Nat.add_mul, one_mul]
        _ ≤ plan.groups * plan.width :=
          Nat.mul_le_mul_right plan.width (Nat.succ_le_iff.mpr hadmitted.1)

/-- The geometry contract packages the physical premises required before an `HMap` refinement. -/
structure GeometryContract : Prop where
  extractedShard : ∀ address word, address.direct_word = some word →
    table.shard address ⦃ shard => shard.val = directShard word ⦄
  admittedIndex : ∀ address word lanes, DirectAdmission address word lanes →
    preparedFlatIndex address lanes = some (flatIndex word lanes)
  flatBound : ∀ word lanes, directLane word < lanes →
    flatIndex word lanes < shardCount * lanes
  extentBound : ∀ laneBits, laneBits ≤ maxLaneBits →
    2 ^ laneBits ≤ 2 ^ maxLaneBits ∧ 2 ^ laneBits < 2 ^ 64
  positiveSizing : ∀ load width plan, 0 < load → checkedSizing load width = some plan →
    0 < width ∧ ∃ groups,
      checkedPow2Ceil (max 2 (ceilDiv (2 * load) width)) = some groups ∧
      plan.groups = groups ∧ plan.slots = groups * width ∧
      2 * load < 2 ^ 64 ∧ groups * width < 2 ^ 64
  groupSizing : ∀ load width plan, 0 < load → AdmissibleGroupWidth width →
    checkedSizing load width = some plan →
      AdmissibleGroupWidth width ∧ 2 ≤ plan.groups ∧
        (∃ exponent, exponent ≤ 64 ∧ plan.groups = 2 ^ exponent) ∧
          plan.slots = plan.groups * width
  densityExact : ∀ capacity distinctWords,
    densityAdmits capacity distinctWords = true ↔
      capacity ≤ max shardCount (2 * distinctWords)
  densityMonotone : ∀ capacity lower upper, lower ≤ upper →
    densityAdmits capacity lower = true → densityAdmits capacity upper = true
  emptyAllocation : ∀ width, checkedSizing 0 width = some { groups := 0, slots := 0 }
  destinationBound : ∀ plan destination, DestinationAdmitted plan destination →
    destinationIndex plan destination < destinationCapacity plan destination

/-- Checked flat and spill geometry supplies the physical bounds premise for canonical `HMap`. -/
theorem geometry_wellFormed : GeometryContract := by
  exact {
    extractedShard := extracted_shard_of_direct_word
    admittedIndex := preparedFlatIndex_of_admission
    flatBound := flatIndex_lt
    extentBound := laneExtent_representable
    positiveSizing := checkedSizing_matches_formula
    groupSizing := checkedSizing_group_geometry
    densityExact := densityAdmits_eq_true
    densityMonotone := densityAdmits_mono
    emptyAllocation := empty_region_allocates_nothing
    destinationBound := prepared_destination_in_bounds
  }

end Apeiron.GrowableGeometry
