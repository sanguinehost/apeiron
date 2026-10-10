/-
# Zero-axiom audit for face-correct addressing

The audit population of `Iota.Address.Faces`: the address and shard types, the
refusal type, the packing bound and its arithmetic helpers, the two coordinate
formulas, the face dispatcher, the home shard, the admission check, the
hyperbolic observation domain and the two headline links. Every declaration is
registered in `tools/iota/zero_axioms.json` under the `address` stage and must
report an empty `Lean.collectAxioms` set.

The rejecting fixtures are:

* `checkInput_elliptic_out_of_domain`: a cell whose first coordinate is `2 ^ 32`
  is outside `FaceDomain .elliptic`, so `checkInput` returns
  `.error .ellipticOutOfDomain`;
* `hyperbolic_collision_not_interchangeable`: two hyperbolic keys with the same
  coordinate but different `limits` are unequal under `MemoKey.equal`, and a
  lookup keyed on the full key misses. The mechanism is that `MemoKey.equal`
  compares every data field including `limits`, so the address coordinate alone
  does not identify a key.
-/
import Iota.Address.Faces
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.unusedVariables false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Address.Audit

/-- A positive fixture: `rfl` uses no axiom. The runner's mutation control replaces this proof with
a `Classical.choice` term, so this declaration is the red/restored-green target. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

/-- The elliptic domain refuses a cell whose first coordinate is `2 ^ 32`: the coordinate is not
below `2 ^ 32`, so the admission check returns the refusal. -/
theorem checkInput_elliptic_out_of_domain {sources : Iota.Source.SourceArena}
    {stateCapacity : Nat} {states : Iota.Arena.StateArena stateCapacity}
    (sigma : Iota.Word) (input : Iota.Memo.MemoInput sources states)
    (cell_eq : input.addressCell =
      (Iota.Word.ofNat (Nat.pow 2 32), Iota.Word.ofNat 0)) :
    checkInput sigma .elliptic input = .error .ellipticOutOfDomain := by
  have hout : ¬ (input.addressCell.1.val < packBase ∧ input.addressCell.2.val < packBase) := by
    intro h
    rw [cell_eq] at h
    have hvalue : (Iota.Word.ofNat (Nat.pow 2 32)).val = Nat.pow 2 32 :=
      Iota.Refinement.modEqOfLt packBase_lt_wordModulus
    rw [hvalue] at h
    exact Nat.lt_irrefl _ h.1
  unfold checkInput
  rw [dif_neg hout]

/-- A lookup shape: the table's entries are keys and a probe answers with the first entry whose
`MemoKey.equal` is `true`. This is the identity comparison the table performs; the address
coordinate is not the comparison. -/
def lookupShape {sources : Iota.Source.SourceArena} {stateCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    (entries : List (Iota.Memo.MemoKey sources states))
    (key : Iota.Memo.MemoKey sources states) : Bool :=
  entries.any (fun entry => Iota.Memo.MemoKey.equal entry key)

/-- A one-entry lookup answers exactly the `MemoKey.equal` of its entry and the probe. -/
theorem lookupShape_singleton {sources : Iota.Source.SourceArena} {stateCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    (entry key : Iota.Memo.MemoKey sources states) :
    lookupShape [entry] key = Iota.Memo.MemoKey.equal entry key := by
  unfold lookupShape
  rw [List.any_cons, List.any_nil, Bool.or_false]

/-- Changing only the `limits` field of a hyperbolic key leaves its address coordinate unchanged:
the coordinate reads the face, `σ` and the address cell, and the update touches none of them. -/
theorem hyperbolic_collision_same_coord {sources : Iota.Source.SourceArena}
    {stateCapacity : Nat} {states : Iota.Arena.StateArena stateCapacity}
    (base : Iota.Memo.MemoKey sources states) (limits : Iota.Memo.Limits)
    (face : base.face = .hyperbolic) :
    coord base = coord { base with limits := limits } := by
  have hbase : coord base = hyperbolicCoord base.sigma base.addressCell := by
    unfold coord
    rw [face]
  have hother : coord { base with limits := limits } =
      hyperbolicCoord base.sigma base.addressCell := by
    unfold coord
    dsimp only
    rw [face]
  rw [hbase, hother]

/-- Changing only the `limits` field changes the key, so `MemoKey.equal` is `false`. -/
theorem hyperbolic_collision_equal_false {sources : Iota.Source.SourceArena}
    {stateCapacity : Nat} {states : Iota.Arena.StateArena stateCapacity}
    (base : Iota.Memo.MemoKey sources states) (limits : Iota.Memo.Limits)
    (different : limits ≠ base.limits) :
    Iota.Memo.MemoKey.equal base { base with limits := limits } = false := by
  cases h : Iota.Memo.MemoKey.equal base { base with limits := limits } with
  | true =>
      have equal : base = { base with limits := limits } :=
        (Iota.Memo.MemoKey.equal_iff base _).mp h
      exact absurd (congrArg Iota.Memo.MemoKey.limits equal).symm different
  | false => rfl

/-- A lookup keyed on the full key misses a key that shares the address coordinate but not the
`limits`, because `MemoKey.equal` compares `limits`. -/
theorem hyperbolic_collision_lookup_miss {sources : Iota.Source.SourceArena}
    {stateCapacity : Nat} {states : Iota.Arena.StateArena stateCapacity}
    (base : Iota.Memo.MemoKey sources states) (limits : Iota.Memo.Limits)
    (different : limits ≠ base.limits) :
    lookupShape [base] { base with limits := limits } = false := by
  rw [lookupShape_singleton, hyperbolic_collision_equal_false base limits different]

/-- The collision, jointly: the two hyperbolic keys have the same address coordinate and the
key-keyed lookup does not answer for the other. The mechanism is that `MemoKey.equal` compares
`limits`, so equal coordinates are not interchangeable in a lookup keyed on the full key. -/
theorem hyperbolic_collision_not_interchangeable {sources : Iota.Source.SourceArena}
    {stateCapacity : Nat} {states : Iota.Arena.StateArena stateCapacity}
    (base : Iota.Memo.MemoKey sources states) (limits : Iota.Memo.Limits)
    (face : base.face = .hyperbolic) (different : limits ≠ base.limits) :
    coord base = coord { base with limits := limits } ∧
      lookupShape [base] { base with limits := limits } = false :=
  ⟨hyperbolic_collision_same_coord base limits face,
    hyperbolic_collision_lookup_miss base limits different⟩

end Iota.Address.Audit

-- The registered population, mirrored in tools/iota/zero_axioms.json. Each line binds the
-- elaborated declaration to its registered type; the command's log lines are the coverage report.
#audit_empty Iota.Address.Coord : type_of% @Iota.Address.Coord
#audit_empty Iota.Address.Shard : type_of% @Iota.Address.Shard
#audit_empty Iota.Address.AddressRefusal : type_of% @Iota.Address.AddressRefusal
#audit_empty Iota.Address.packBase : type_of% @Iota.Address.packBase
#audit_empty Iota.Address.packBase_pos : type_of% @Iota.Address.packBase_pos
#audit_empty Iota.Address.packBase_lt_wordModulus : type_of% @Iota.Address.packBase_lt_wordModulus
#audit_empty Iota.Address.packLt : type_of% @Iota.Address.packLt
#audit_empty Iota.Address.divGoFuelCongr : type_of% @Iota.Address.divGoFuelCongr
#audit_empty Iota.Address.divEqSubDiv : type_of% @Iota.Address.divEqSubDiv
#audit_empty Iota.Address.divSmall : type_of% @Iota.Address.divSmall
#audit_empty Iota.Address.divMulAdd : type_of% @Iota.Address.divMulAdd
#audit_empty Iota.Address.modMulAdd : type_of% @Iota.Address.modMulAdd
#audit_empty Iota.Address.ellipticPack : type_of% @Iota.Address.ellipticPack
#audit_empty Iota.Address.ellipticUnpack : type_of% @Iota.Address.ellipticUnpack
#audit_empty Iota.Address.ellipticUnpack_pack : type_of% @Iota.Address.ellipticUnpack_pack
#audit_empty Iota.Address.parabolicCoord : type_of% @Iota.Address.parabolicCoord
#audit_empty Iota.Address.hyperbolicCoord : type_of% @Iota.Address.hyperbolicCoord
#audit_empty Iota.Address.coord : type_of% @Iota.Address.coord
#audit_empty Iota.Address.homeShard : type_of% @Iota.Address.homeShard
#audit_empty Iota.Address.checkInput : type_of% @Iota.Address.checkInput
#audit_empty Iota.Address.HyperbolicObservationDomain : type_of% @Iota.Address.HyperbolicObservationDomain
#audit_empty Iota.Address.hyperbolic_observation_sound : type_of% @Iota.Address.hyperbolic_observation_sound
#audit_empty Iota.Address.hyperbolicCoord_denotes_ringNormForm : type_of% @Iota.Address.hyperbolicCoord_denotes_ringNormForm
#audit_empty Iota.Address.Audit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Address.Audit.checkInput_elliptic_out_of_domain : type_of% @Iota.Address.Audit.checkInput_elliptic_out_of_domain
#audit_empty Iota.Address.Audit.lookupShape : type_of% @Iota.Address.Audit.lookupShape
#audit_empty Iota.Address.Audit.lookupShape_singleton : type_of% @Iota.Address.Audit.lookupShape_singleton
#audit_empty Iota.Address.Audit.hyperbolic_collision_same_coord : type_of% @Iota.Address.Audit.hyperbolic_collision_same_coord
#audit_empty Iota.Address.Audit.hyperbolic_collision_equal_false : type_of% @Iota.Address.Audit.hyperbolic_collision_equal_false
#audit_empty Iota.Address.Audit.hyperbolic_collision_lookup_miss : type_of% @Iota.Address.Audit.hyperbolic_collision_lookup_miss
#audit_empty Iota.Address.Audit.hyperbolic_collision_not_interchangeable : type_of% @Iota.Address.Audit.hyperbolic_collision_not_interchangeable
