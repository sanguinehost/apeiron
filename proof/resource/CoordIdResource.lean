import CoordIdSource
import Aeneas

/-
SCOPE: the extracted transitions, work measure and reachable witness of the production
  `word_coord_id`, the identity-word projection of `coord::Addr`, and the reachability of its
  input domain from the real constructor `coord::spread`.
REGISTER: scalar resource obligation (transition relation, bounded work, joint reachable witness)
  over the extracted bodies only. Excluded: injectivity of the identity word (a ring fact
  registered elsewhere), the placement-word security argument, canonical ring correspondence,
  instruction counts and any hardware timing.
CORPUS: `src/coord.rs` — the `Addr` declaration (lines 184-193) and `spread` (lines 366-374) — and
  the extracted `coord.Addr`, `coord.spread`, `word_coord_id`, `word_coord_spread`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean:24180 (`word_coord_id`).
GATE: `python3 proof/resource/CoordId_gate.py [--controls]`, from the repository root; 540-second
  whole-invocation budget serialized on /tmp/opencode/apeiron-lane-2.lock, systemd-run
  MemoryMax=12G MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: `proved` for every theorem below when the gate exits 0 with COORD_ID_KERNEL_OK and, under
  --controls, COORD_ID_REAL_SOURCE_CONTROL_RED followed by _RESTORED. No measurement is claimed:
  the work measure below is a proved count of transitions, not a timing.
-/

namespace CoordIdResource
open Aeneas Aeneas.Std Result

/-- The real `coord::Addr`: the placement word beside the identity word. -/
abbrev Addr := coord_id_source.coord.Addr

/-- The odd multiplier the production `spread` multiplies by. -/
abbrev Phi : Std.U64 := coord_id_source.coord.PHI

/-- The 128-bit product the extracted `spread` body computes, named as that body names it. -/
abbrev spreadProduct (x : Std.U64) : Std.U128 :=
  core.num.U128.wrapping_mul (UScalar.cast .U128 x) (UScalar.cast .U128 Phi)

/-! ## Transitions

The extracted body has no loop, no branch and no early return: one field projection wrapped in
one `ok`. The guard of the transition is therefore vacuous and the relation holds at every
address. -/

/-- The single production transition, from an address to the identity word it carries. -/
theorem body_step (address : Addr) :
    coord_id_source.word_coord_id address = ok address.id := by
  unfold coord_id_source.word_coord_id
  rfl

/-- Every input takes that one step and no other: the body has no error branch, so no refusal,
    no retry and no second attempt is reachable from any address. -/
theorem body_never_err (address : Addr) (r : Result Std.U64)
    (h : coord_id_source.word_coord_id address = r) :
    ∃ w : Std.U64, r = ok w ∧ w = address.id := by
  rw [body_step] at h
  exact ⟨address.id, h.symm, rfl⟩

/-- Every input reaches a returned word. Together with [`body_never_err`] that closes the work
    measure: the temporal cost is the constant one transition, with no rank to decrease because
    there is no loop to iterate. -/
theorem work_one_step (address : Addr) :
    ∃ w : Std.U64, coord_id_source.word_coord_id address = ok w ∧ w = address.id :=
  ⟨address.id, body_step address, rfl⟩

/-! ## Work measure in closed form

The result is a function of the `id` field alone, and the closed form is that field. -/

/-- Closed form: the transition returns the identity field of its argument, whatever the
    placement word is. -/
theorem work_closed_form (place id : Std.U64) :
    coord_id_source.word_coord_id (⟨place, id⟩ : Addr) = ok id :=
  body_step _

/-- The placement word is never read: two addresses differing only in it cost the same single
    transition and return the same word. -/
theorem work_independent_of_place (p q i : Std.U64) :
    coord_id_source.word_coord_id (⟨p, i⟩ : Addr) = coord_id_source.word_coord_id (⟨q, i⟩ : Addr) ∧
      coord_id_source.word_coord_id (⟨p, i⟩ : Addr) = ok i :=
  ⟨work_closed_form p i ▸ work_closed_form q i, work_closed_form p i⟩

/-! ## Reachable success witness

`Addr` has one production constructor, `spread`, so the witness below runs the real constructor
and then the real projection rather than a model literal. -/

/-- The production constructor returns: for every word it yields an `ok`, so the domain of the
    projection -- an `Addr` -- is reachable from the constructor. -/
theorem spread_succeeds (x : Std.U64) :
    ∃ a : Addr, coord_id_source.word_coord_spread x = ok a := by
  unfold coord_id_source.word_coord_spread coord_id_source.coord.spread
  simp only [bind_tc_ok, lift]
  obtain ⟨i2, hi2, _⟩ := WP.spec_imp_exists
    (Std.U128.ShiftRight_IScalar_spec (spreadProduct x) 64#i32 (by norm_num) (by norm_num))
  refine ⟨{ place := core.num.U64.rotate_left (UScalar.cast .U64 i2) 32#u32 ^^^
                  UScalar.cast .U64 (spreadProduct x),
            id := UScalar.cast .U64 (spreadProduct x) }, ?_⟩
  rw [hi2]
  simp only [bind_tc_ok]

/-- Joint reachable success witness: the real constructor produces an address and the real
    projection returns that address's identity word. -/
theorem reachable_success (x : Std.U64) :
    ∃ a : Addr, ∃ w : Std.U64,
      coord_id_source.word_coord_spread x = ok a ∧
      coord_id_source.word_coord_id a = ok w ∧ w = a.id := by
  obtain ⟨a, ha⟩ := spread_succeeds x
  exact ⟨a, a.id, ha, body_step a, rfl⟩

/-- The identity field of a constructed address is the truncated widening product: the `id` slot
    is the low half, and the `place` slot is where the high half goes. -/
theorem spread_id_field (x : Std.U64) (a : Addr)
    (h : coord_id_source.word_coord_spread x = ok a) :
    a.id = UScalar.cast .U64 (spreadProduct x) := by
  unfold coord_id_source.word_coord_spread coord_id_source.coord.spread at h
  simp only [bind_tc_ok, lift] at h
  obtain ⟨i2, hi2, _⟩ := WP.spec_imp_exists
    (Std.U128.ShiftRight_IScalar_spec (spreadProduct x) 64#i32 (by norm_num) (by norm_num))
  rw [hi2] at h
  simp only [bind_tc_ok] at h
  have ha : { place := core.num.U64.rotate_left (UScalar.cast .U64 i2) 32#u32 ^^^
                      UScalar.cast .U64 (spreadProduct x),
                 id := UScalar.cast .U64 (spreadProduct x) } = a :=
    Result.ok_injective h
  rw [← ha]

/-- That field's value is `x * PHI` in `Z/2^64`: the low half of the 128-bit product, the
    placement word having taken the high half. -/
theorem identity_value (x : Std.U64) (a : Addr)
    (h : coord_id_source.word_coord_spread x = ok a) :
    a.id.val = (x.val * Phi.val) % 2 ^ 64 := by
  rw [spread_id_field x a h]
  simp only [spreadProduct, UScalar.cast_val_eq, core.num.U128.wrapping_mul_val_eq,
    UScalar.size, UScalarTy.numBits]
  have hx : x.val < 2 ^ 128 := by
    have h := U64.le_max x
    norm_num at h ⊢
    omega
  have hphi : Phi.val < 2 ^ 128 := by
    have h := U64.le_max Phi
    norm_num at h ⊢
    omega
  rw [Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hphi]
  exact Nat.mod_mod_of_dvd (a := x.val * Phi.val) (by norm_num)

/-- On an address the production constructor produced, the projection returns the identity word
    `x * PHI` in `Z/2^64`. This is the statement the mutation control attacks. -/
theorem identity_is_low_half (x : Std.U64) (a : Addr)
    (h : coord_id_source.word_coord_spread x = ok a) :
    ∃ w : Std.U64, coord_id_source.word_coord_id a = ok w ∧
      w.val = (x.val * Phi.val) % 2 ^ 64 :=
  ⟨a.id, body_step a, identity_value x a h⟩

end CoordIdResource

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit CoordIdResource.body_step
#resource_audit CoordIdResource.body_never_err
#resource_audit CoordIdResource.work_one_step
#resource_audit CoordIdResource.work_closed_form
#resource_audit CoordIdResource.work_independent_of_place
#resource_audit CoordIdResource.spread_succeeds
#resource_audit CoordIdResource.reachable_success
#resource_audit CoordIdResource.spread_id_field
#resource_audit CoordIdResource.identity_value
#resource_audit CoordIdResource.identity_is_low_half

run_cmd Lean.logInfo "COORD_ID_KERNEL_OK"
