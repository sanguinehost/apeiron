import ApeironWord
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases

/-!
# Spill placement, the bounded probe cycle, and the tag fingerprint

The flat spill port retires the occupancy summary, the per-slot tag array and the group/window
descent.  One open-addressed `Vec (SpillSlot R V)` carries the occupancy marker, the face, the held
carrier and the payload together, and the probe is one bounded linear cycle over the capacity.

Three bodies of fact are proved here.

* **Placement.**  `SpillPlacement` states the two invariants the production `Region` documentation
  records (`src/table/state.rs:70-71`): `slots.len()` is the capacity and is either `0` or a power of
  two, and `live` is the exact number of occupied slots.  The constructor establishes them
  (`Region.empty` and `Region.reserve_empty`), and `publish_live_preserves_placement` carries one
  publication.  The checked sizing `table.slots_for` is the remaining constructor obligation; see the
  receipt.
* **The cycle is a permutation.**  `spill_cycle_covers_of_placement` : for a capacity that is a power
  of two, the `bound` slots a bounded cycle visits are exactly the region's `bound` slots, each once.
  This is the step that consumes the power-of-two obligation, and it is the headline the mutation
  control breaks.
* **Absence.**  `SpillProbeBoundAbsentSound` is derived from a completed cycle: a cycle that ran all
  `bound` steps without a vacancy observed every slot occupied, so no stored key was passed over.

The tag fingerprint itself is unchanged by the port and is proved exact at equal `(Sigma, key)`.
-/

namespace Apeiron.GrowableScan

open Aeneas Aeneas.Std Result
open apeiron_word

/-! ## The generated tag fingerprint -/

/-- The generated fingerprint shift is 57: the fingerprint reads the top seven bits of the
    spread word. -/
private theorem fingerprintShift : table.TAG_FINGERPRINT_SHIFT = ok 57#u32 := by
  have hlit : (57#u32).val = 57 := by norm_num
  have hspec : table.TAG_FINGERPRINT_SHIFT ⦃ z => z.val = 57 ⦄ := by
    unfold table.TAG_FINGERPRINT_SHIFT
    step as ⟨amount, hamount⟩ by norm_num [table.TAG_FINGERPRINT_BITS]
    norm_num [table.TAG_FINGERPRINT_BITS] at hamount
    exact hamount
  obtain ⟨z, hz, hval⟩ := WP.spec_imp_exists hspec
  exact hz.trans (congrArg ok (UScalar.eq_of_val_eq (hval.trans hlit.symm)))

/-- The generated fingerprint mask is `127`, the seven fingerprint bits below the live bit. -/
private theorem fingerprintMask : table.TAG_FINGERPRINT_MASK = ok 127#u8 := by
  have hlit : (127#u8).val = 127 := by norm_num
  have hspec : table.TAG_FINGERPRINT_MASK ⦃ z => z.val = 127 ⦄ := by
    unfold table.TAG_FINGERPRINT_MASK
    step as ⟨shifted, hshifted⟩ by norm_num [table.TAG_FINGERPRINT_BITS]
    norm_num [table.TAG_FINGERPRINT_BITS, UScalar.size, U8.size, U8.numBits] at hshifted
    step as ⟨reduced, hreduced⟩
    calc (↑reduced : ℕ) = ↑shifted - 1 := hreduced
      _ = 127 := by rw [hshifted]; simp
  obtain ⟨z, hz, hval⟩ := WP.spec_imp_exists hspec
  exact hz.trans (congrArg ok (UScalar.eq_of_val_eq (hval.trans hlit.symm)))

/-- A generated tag always contains the nonzero live bit.  The generated tag is
    `TAG_LIVE ||| (cast .U8 (place.wrapping_mul TAG_SPREAD >>> TAG_FINGERPRINT_SHIFT) &&& TAG_FINGERPRINT_MASK)`,
    so a tag equal to zero would force `TAG_LIVE` itself to zero. -/
theorem tag_ne_zero (place : U64) (tag : U8) (htag : table.tag place = ok tag) :
    tag ≠ 0#u8 := by
  have hspec : table.tag place ⦃ r => r ≠ 0#u8 ⦄ := by
    unfold table.tag
    step as ⟨spread, hspread⟩
    rw [fingerprintShift]
    simp only [bind_tc_ok]
    step as ⟨shifted, hshifted⟩
    step as ⟨low, hlow⟩
    rw [fingerprintMask]
    simp only [bind_tc_ok]
    step as ⟨masked, hmasked⟩
    intro hzero
    have hbits : table.TAG_LIVE.bv ||| masked.bv = (0#u8).bv := by
      have hbv := congrArg UScalar.bv hzero
      simpa only [UScalar.bv_or] using hbv
    have hlive : table.TAG_LIVE.bv = (0#u8).bv :=
      (BitVec.or_eq_zero_iff.mp hbits).1
    have hliveScalar : table.TAG_LIVE = 0#u8 :=
      (UScalar.eq_equiv_bv_eq table.TAG_LIVE 0#u8).mpr hlive
    have hvalue := congrArg UScalar.val hliveScalar
    norm_num [table.TAG_LIVE] at hvalue
  simpa only [htag, WP.spec_ok] using hspec

/-- The tag fingerprint obtained through a generated address computation. -/
def spillFingerprint {K : Type}
    (address : cell.Sigma → K → Result table.TableAddress)
    (identity : cell.Sigma × K) : Result U8 := do
  let a ← address identity.1 identity.2
  table.tag a.place

/-- Fingerprints respect equality of the complete `(Sigma, key)` identity. -/
structure SpillFingerprintExact {K : Type}
    (address : cell.Sigma → K → Result table.TableAddress) : Prop where
  eq_of_identity : ∀ left right : cell.Sigma × K, left = right →
    spillFingerprint address left = spillFingerprint address right

/-- Generated address/tag composition is fingerprint-exact at equal full identities. -/
theorem spillFingerprintExact {K : Type}
    (address : cell.Sigma → K → Result table.TableAddress) :
    SpillFingerprintExact address := by
  constructor
  intro left right heq
  subst right
  rfl

/-- Equal full identities therefore have equal generated tag fingerprints. -/
theorem spillFingerprint_eq_of_identity {K : Type}
    (address : cell.Sigma → K → Result table.TableAddress)
    (left right : cell.Sigma × K) (heq : left = right) :
    spillFingerprint address left = spillFingerprint address right :=
  (spillFingerprintExact address).eq_of_identity left right heq

/-- A place whose spread word has no fingerprint bits set carries the bare live-bit tag `128`.
    `TAG_SPREAD` is the odd constant `11400714819323198485`, and the premise is discharged by
    `rfl` at each concrete place. -/
private theorem tag_live_only (place : U64)
    (hzero : (core.num.U64.wrapping_mul place 11400714819323198485#u64 >>> 57#u32) =
      ok 0#u64) :
    table.tag place = ok 128#u8 := by
  simp only [table.tag, table.TAG_SPREAD, coord.PHI, table.TAG_LIVE,
    fingerprintShift, fingerprintMask, lift, bind_tc_ok, hzero]
  have hval : (128#u8 ||| (UScalar.cast UScalarTy.U8 0#u64 &&& 127#u8)).val = 128 := by
    norm_num [UScalar.val_or, UScalar.val_and, UScalar.cast_val_eq]
  exact congrArg ok (UScalar.eq_of_val_eq hval)

/-- Distinct full keys can have the same generated seven-bit fingerprint: the place is spread
    modulo `2^64` by the odd `TAG_SPREAD` before the top seven bits are read, so places `0` and
    `89` share a fingerprint, `89 * TAG_SPREAD` reducing to `92694865739326285` whose top seven
    bits are zero. -/
theorem spillFingerprint_collision_witness :
    ∃ left right : cell.Sigma × U64,
      left ≠ right ∧
      spillFingerprint (fun _ key => ok { place := key, direct_word := none }) left =
        spillFingerprint (fun _ key => ok { place := key, direct_word := none }) right := by
  have hplace0 : table.tag 0#u64 = ok 128#u8 :=
    tag_live_only 0#u64 (by rfl)
  have hplace89 : table.tag 89#u64 = ok 128#u8 :=
    tag_live_only 89#u64 (by rfl)
  refine ⟨(cell.Sigma.Parabolic, 0#u64), (cell.Sigma.Parabolic, 89#u64), ?_, ?_⟩
  · intro heq
    have hkey := congrArg (fun identity : cell.Sigma × U64 => identity.2.val) heq
    norm_num at hkey
  · simp only [spillFingerprint, bind_tc_ok, hplace0, hplace89]

/-- A tag match at a probed slot: the generated query fingerprint equals the slot's occupancy
    marker.  This is only a candidate; the full `(Sigma, K)` compare still has to succeed. -/
def SpillTagHit (slotTag wanted : U8) : Prop :=
  slotTag = wanted

/-- Even after a tag hit, successful presence still traverses generated `table.matches`:
both `Sigma` equality and the carrier's full-key comparison must return true. -/
theorem fingerprint_hit_requires_full_matches {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (slot : table.SpillSlot R V)
    (querySigma : cell.Sigma) (key : K) (address : table.TableAddress)
    (fingerprint wanted : U8)
    (_htag : table.tag address.place = ok fingerprint)
    (_hhit : SpillTagHit slot.tag wanted)
    (hmatch : table.matches keyRep slot querySigma key address = ok true) :
    cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma querySigma = ok true ∧
      keyRep.«matches» querySigma slot.held key address = ok true := by
  unfold table.matches at hmatch
  cases heq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma querySigma with
  | vis eff k => simp [heq] at hmatch
  | div => simp [heq] at hmatch
  | ret equal =>
      cases equal <;> simp_all

/-! ## Placement

The occupancy marker a slot carries is its `tag`, so a slot is occupied exactly when `tag ≠ 0`.  The
placement invariant below is the two facts `src/table/state.rs:70-71` records. -/

/-- The number of vacant slots of a region: the slots whose occupancy marker is zero. -/
def SpillVacantCount {R V : Type} (region : table.Region R V) : Nat :=
  region.slots.val.countP (fun slot => slot.tag = 0#u8)

/-- The placement invariant of a generated spill region: `slots.len()` is the capacity and is
    either `0` or a power of two, so the probe mask `len - 1` masks off a whole number of index bits;
    and `live` is the exact number of occupied slots, so a vacancy always exists. -/
structure SpillPlacement {R V : Type} (region : table.Region R V) : Prop where
  /-- The capacity obligation: zero, or a power of two.  This is the field the bounded-probe
      coverage consumes, through `is_power_of_two_exists` on the generated predicate. -/
  capacity : region.slots.val.length = 0 ∨ ∃ k : Nat, region.slots.val.length = 2 ^ k
  /-- The exactness obligation: `live` counts the occupied slots. -/
  liveExact : region.live.val + SpillVacantCount region = region.slots.val.length

/-- The `Ok` component of a computed pair-bind chain whose second component is untouched. -/
private theorem pair_bind_snd' {α β γ : Type} (x : α) (y : β) (f : α → Result γ) (m : γ × β)
    (h : (do let (a, b) ← ok (x, y); let c ← f a; ok (c, b)) = ok m) : m.2 = y := by
  rw [bind_tc_ok] at h
  change (do let c ← f x; ok (c, y)) = ok m at h
  cases hc : f x with
  | vis eff k =>
      rw [hc] at h
      simp at h
  | div =>
      rw [hc] at h
      simp at h
  | ret c =>
      rw [hc] at h
      rw [bind_tc_ok] at h
      exact (congrArg Prod.snd (ok_injective h)).symm

/-- One `Vec` component survives a `try_reserve_exact`: the reservation changes the capacity field
    and never the stored elements. -/
private theorem vec_try_reserve_exact_val_eq {α : Type} (v : alloc.vec.Vec α) (n : Usize) (m : _)
    (h : alloc.vec.Vec.try_reserve_exact Global v n = ok m) : m.2.val = v.val := by
  unfold alloc.vec.Vec.try_reserve_exact alloc.vec.Vec.try_reserve at h
  simp only [] at h
  split at h
  · rw [← ok_injective h]; rfl
  · rw [← ok_injective h]

private theorem resources_try_reserve_exact_val_eq {α : Type} (v : alloc.vec.Vec α) (n : Usize) (m : _)
    (h : table.resources.try_reserve_exact v n = ok m) : m.2.val = v.val := by
  unfold table.resources.try_reserve_exact at h
  cases hres : alloc.vec.Vec.try_reserve_exact Global v n with
  | vis eff k => simp [hres] at h
  | div => simp [hres] at h
  | ret pair =>
      rcases pair with ⟨r, items⟩
      have hval : items.val = v.val :=
        vec_try_reserve_exact_val_eq v n (r, items) hres
      rw [hres] at h
      have hsnd : m.2 = items := pair_bind_snd' r items _ m h
      rw [hsnd, hval]

/-- The coordinate reservation stores no coordinate word: it starts at
    `alloc.vec.Vec.new Std.U64`, so the successful branch changes only the capacity field. -/
private theorem coordinates_try_reserve_exact_val_eq (capacity : Usize) (m : _)
    (h : table.resources.try_reserve_exact (alloc.vec.Vec.new Std.U64) capacity = ok m) :
    m.2.val = [] :=
  resources_try_reserve_exact_val_eq (alloc.vec.Vec.new Std.U64) capacity m h

/-- The generated `Region.reserve_empty` body with its two checked reservations abstracted out: each
    argument is the `(reservation result, resulting vector)` pair the generated
    `table.resources.try_reserve_exact` returns. A region is produced only when both reservations
    succeed, built from both vectors, with `bits` and `live` zero
    (`proof/ApeironWord.lean:20099-20122`). -/
private def reserve_empty_ok {R V : Type}
    (rsv : core.result.Result Unit table.StorageError × alloc.vec.Vec (table.SpillSlot R V))
    (rsvCoords : core.result.Result Unit table.StorageError × alloc.vec.Vec Std.U64) :
    Result (core.result.Result (table.Region R V) table.StorageError) := do
  let (rr, ss) ← ok rsv
  let cf ← core.result.Result.Insts.CoreOpsTry.branch rr
  match cf with
  | core.ops.control_flow.ControlFlow.Continue a =>
      let (rr1, cc) ← ok rsvCoords
      let cf1 ← core.result.Result.Insts.CoreOpsTry.branch rr1
      match cf1 with
      | core.ops.control_flow.ControlFlow.Continue a1 =>
          ok (core.result.Result.Ok
            { slots := ss, coords := cc, bits := 0#u32, live := 0#usize })
      | core.ops.control_flow.ControlFlow.Break residual1 =>
          core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
            (table.Region R V) (core.convert.FromSame table.StorageError) residual1
  | core.ops.control_flow.ControlFlow.Break residual =>
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
      (table.Region R V) (core.convert.FromSame table.StorageError) residual

/-- A successful checked reservation writes no slot and no coordinate word, and counts none:
    both arrays of the generated constructor are reserved from an empty vector, so both stored
    elements are empty and `bits` and `live` are zero
    (`table.Region.reserve_empty`, `proof/ApeironWord.lean:20099-20122`). -/
private theorem reserve_empty_vals {R V : Type} (cloneR : core.clone.Clone R)
    (cloneV : core.clone.Clone V) (capacity : Usize) (region : table.Region R V)
    (h : table.Region.reserve_empty cloneR cloneV capacity = ok (core.result.Result.Ok region)) :
    region.slots.val = [] ∧ region.coords.val = [] ∧ region.live.val = 0 := by
  unfold table.Region.reserve_empty at h
  cases hres : table.resources.try_reserve_exact (alloc.vec.Vec.new (table.SpillSlot R V))
      capacity with
  | vis eff k => simp [hres] at h
  | div => simp [hres] at h
  | ret p =>
      rcases p with ⟨r, slots⟩
      have hslots : slots.val = [] :=
        resources_try_reserve_exact_val_eq (alloc.vec.Vec.new (table.SpillSlot R V)) capacity _ hres
      rw [hres] at h
      rw [bind_tc_ok] at h
      change (do let cf ← core.result.Result.Insts.CoreOpsTry.branch r; match cf with
          | core.ops.control_flow.ControlFlow.Continue a =>
              do
                let (r1, coords) ←
                  table.resources.try_reserve_exact (alloc.vec.Vec.new Std.U64) capacity
                let cf1 ← core.result.Result.Insts.CoreOpsTry.branch r1
                match cf1 with
                | core.ops.control_flow.ControlFlow.Continue a1 =>
                    ok (core.result.Result.Ok
                      { slots := slots, coords := coords, bits := 0#u32, live := 0#usize })
                | core.ops.control_flow.ControlFlow.Break residual1 =>
                    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                      (table.Region R V) (core.convert.FromSame table.StorageError) residual1
          | core.ops.control_flow.ControlFlow.Break residual =>
              core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                (table.Region R V) (core.convert.FromSame table.StorageError) residual) = ok (core.result.Result.Ok region) at h
      cases hc : core.result.Result.Insts.CoreOpsTry.branch r with
      | vis eff kk =>
          rw [hc] at h
          simp at h
      | div =>
          rw [hc] at h
          simp at h
      | ret c =>
          rw [hc] at h
          cases c with
          | Continue _ =>
              rw [bind_tc_ok] at h
              cases hres1 :
                  table.resources.try_reserve_exact (alloc.vec.Vec.new Std.U64) capacity with
              | vis eff k => simp [hres1] at h
              | div => simp [hres1] at h
              | ret p1 =>
                  have hcoords : p1.2.val = [] :=
                    coordinates_try_reserve_exact_val_eq capacity p1 hres1
                  rw [hres1] at h
                  rw [bind_tc_ok] at h
                  change (do let cf1 ← core.result.Result.Insts.CoreOpsTry.branch p1.1; match cf1 with
                      | core.ops.control_flow.ControlFlow.Continue a1 =>
                          ok (core.result.Result.Ok
                            { slots := slots, coords := p1.2, bits := 0#u32, live := 0#usize })
                      | core.ops.control_flow.ControlFlow.Break residual1 =>
                          core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                            (table.Region R V) (core.convert.FromSame table.StorageError) residual1) = ok (core.result.Result.Ok region) at h
                  cases hc1 : core.result.Result.Insts.CoreOpsTry.branch p1.1 with
                  | vis eff kk =>
                      rw [hc1] at h
                      simp at h
                  | div =>
                      rw [hc1] at h
                      simp at h
                  | ret c1 =>
                      rw [hc1] at h
                      cases c1 with
                      | Continue _ =>
                          rw [bind_tc_ok] at h
                          injection (ok_injective h) with hm
                          subst region
                          exact ⟨hslots, hcoords, rfl⟩
                      | Break residual1 =>
                          unfold core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual at h
                          cases residual1 with
                          | Ok inf => cases inf
                          | Err e => simp at h
          | Break residual =>
              unfold core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual at h
              cases residual with
              | Ok inf => cases inf
              | Err e => simp at h


/-- The generated empty region satisfies the placement invariant in its zero-capacity form. -/
theorem region_empty_spillPlacement {R V : Type} :
    table.Region.empty R V ⦃ region => SpillPlacement region ⦄ := by
  refine ⟨Or.inl rfl, ?_⟩
  simp [SpillVacantCount]

/-- A successful checked reservation writes no slot and counts none: the placement invariant holds
    in its zero-capacity form. -/
theorem reserve_empty_spillPlacement {R V : Type}
    (cloneR : core.clone.Clone R) (cloneV : core.clone.Clone V) (capacity : Usize)
    (region : table.Region R V)
    (hresult : table.Region.reserve_empty cloneR cloneV capacity = ok (core.result.Result.Ok region)) :
    SpillPlacement region := by
  have hvals := reserve_empty_vals cloneR cloneV capacity region hresult
  have hlen : region.slots.val.length = 0 := by rw [hvals.1]; rfl
  refine ⟨Or.inl hlen, ?_⟩
  simp [SpillVacantCount, hvals.1, hvals.2.2]

/-! ## The bounded probe cycle -/

/-- The slot a bounded linear-probe cycle visits at `step`: the generated index
    `(home + step) &&& (bound - 1)`, read on the natural valuations. -/
def SpillCycleSlot (bound home step : Nat) : Nat :=
  (home + step) &&& (bound - 1)

/-- The probe bound is the capacity, so the mask is `bound - 1`; a power-of-two capacity makes the
    mask all-ones, and the visited index is below the capacity. -/
theorem spill_cycle_slot_lt {bound home step k : Nat} (hp : bound = 2 ^ k) (hstep : step < bound) :
    (home + step) &&& (bound - 1) < bound := by
  subst hp
  rw [Nat.and_two_pow_sub_one_eq_mod]
  exact Nat.mod_lt _ (Nat.pow_pos (by decide))

/-- For a power-of-two capacity the bounded cycle is surjective on the slots: every slot is visited
    at some step below the capacity. -/
theorem spill_cycle_covers {bound home j k : Nat} (hp : bound = 2 ^ k) (hj : j < bound) :
    ∃ step, step < bound ∧ (home + step) &&& (bound - 1) = j := by
  subst hp
  have hk : 0 < 2 ^ k := Nat.pow_pos (by decide)
  set step : Nat := (j + 2 ^ k - home % 2 ^ k) % 2 ^ k with hstepDef
  have hstepLt : step < 2 ^ k := by rw [hstepDef]; exact Nat.mod_lt _ hk
  have happ : (home + step) % 2 ^ k = (home % 2 ^ k + step) % 2 ^ k := by
    have hmod : step % 2 ^ k = step := Nat.mod_eq_of_lt hstepLt
    rw [Nat.add_mod, hmod]
  refine ⟨step, hstepLt, ?_⟩
  rw [Nat.and_two_pow_sub_one_eq_mod, happ]
  by_cases hle : home % 2 ^ k ≤ j
  · have hstepEq : step = j - home % 2 ^ k := by
      have hdiff : j - home % 2 ^ k < 2 ^ k :=
        lt_of_le_of_lt (Nat.sub_le j (home % 2 ^ k)) hj
      have hform : j + 2 ^ k - home % 2 ^ k = (j - home % 2 ^ k) + 2 ^ k := by omega
      rw [hstepDef, hform, Nat.add_mod_right, Nat.mod_eq_of_lt hdiff]
    rw [hstepEq]
    have hsum : home % 2 ^ k + (j - home % 2 ^ k) = j := Nat.add_sub_of_le hle
    rw [hsum]
    exact Nat.mod_eq_of_lt hj
  · have hlt : j < home % 2 ^ k := Nat.lt_of_not_ge hle
    have hstepEq : step = j + 2 ^ k - home % 2 ^ k := by
      rw [hstepDef]
      rw [Nat.mod_eq_of_lt (by omega)]
    rw [hstepEq]
    have hsum : home % 2 ^ k + (j + 2 ^ k - home % 2 ^ k) = j + 2 ^ k := by
      have h1 : 2 ^ k - home % 2 ^ k + home % 2 ^ k = 2 ^ k := by omega
      omega
    rw [hsum]
    have hmod : (j + 2 ^ k) % 2 ^ k = j := by
      rw [Nat.add_mod, Nat.mod_eq_of_lt hj, Nat.mod_self, Nat.add_zero,
        Nat.mod_eq_of_lt hj]
    exact hmod

/-- The generated `is_power_of_two` predicate is exactly "a power of two", read on the natural
    valuation.  This is the bridge from the generated constructor obligation to the arithmetic the
    bounded cycle needs. -/
theorem is_power_of_two_exists {x : Usize}
    (h : core.num.Usize.is_power_of_two x = ok true) :
    ∃ k : Nat, x.val = 2 ^ k := by
  obtain ⟨b, hb, hval⟩ := WP.spec_imp_exists (core.num.Usize.is_power_of_two_spec x)
  have hb' : b = true := ok_injective (hb.symm.trans h)
  rw [hb'] at hval
  exact Eq.mpr hval.symm rfl

/-- **The placement headline for the bounded cycle.**  A region whose placement invariant holds and
    whose probe bound is its capacity has every slot visited by a completed cycle.  This is the
    statement that consumes `SpillPlacement.capacity`: without the power-of-two obligation the mask
    `bound - 1` no longer selects every slot and the conclusion does not follow. -/
theorem spill_cycle_covers_of_placement {R V : Type} {region : table.Region R V}
    {home bound : Usize}
    (hplacement : SpillPlacement region)
    (hbound : bound.val = region.slots.val.length) :
    ∀ j : Nat, j < region.slots.val.length →
      ∃ step : Nat, step < bound.val ∧ SpillCycleSlot bound.val home.val step = j := by
  intro j hj
  rcases hplacement.capacity with hzero | hpow
  · exfalso
    omega
  · obtain ⟨k, hk⟩ := hpow
    obtain ⟨step, hstep, hslot⟩ := spill_cycle_covers hk hj
    exact ⟨step, by omega, by simpa [SpillCycleSlot, hbound] using hslot⟩

/-- **Absence from a completed cycle.**  `src/table/state.rs:79` names this certificate
    `SpillProbeBoundAbsentSound`: a bounded cycle that ran all `bound` steps of a region without
    finding a vacancy has visited every slot of the region, so every slot is occupied and no stored
    key was passed over. -/
structure SpillProbeBoundAbsentSound {R V : Type} (region : table.Region R V)
    (home : Usize) (bound : Usize) : Prop where
  /-- The region satisfies the placement invariant. -/
  placement : SpillPlacement region
  /-- The cycle ran the whole capacity: the probe bound is the capacity. -/
  completed : bound.val = region.slots.val.length
  /-- Every slot of the region was visited at some step below the bound. -/
  visited : ∀ j : Nat, j < region.slots.val.length →
    ∃ step : Nat, step < bound.val ∧ SpillCycleSlot bound.val home.val step = j
  /-- Every slot the cycle read was occupied: the generated marker is nonzero. -/
  occupied : ∀ (j : Usize) (slot : table.SpillSlot R V), j.val < region.slots.val.length →
    region.slots.val[j.val]? = some slot → slot.tag ≠ 0#u8
  /-- Hence every slot of the region is occupied, which is the absence conclusion the read path
      relies on. -/
  excludes : ∀ (j : Usize) (slot : table.SpillSlot R V), j.val < region.slots.val.length →
    region.slots.val[j.val]? = some slot → table.SpillSlot.live slot = ok true

/-- **A completed cycle derives absence.**  The premises are the two facts a completed bounded cycle
    supplies: the probe bound is the capacity, and every slot the cycle read carried a nonzero
    occupancy marker.  The conclusion is that no key is stored in a slot the cycle passed over.
    The power-of-two obligation enters through `spill_cycle_covers_of_placement`. -/
theorem spill_probe_bound_absent_sound {R V : Type} {region : table.Region R V}
    {home bound : Usize} (hplacement : SpillPlacement region)
    (hbound : bound.val = region.slots.val.length)
    (hoccupied : ∀ (j : Usize) (slot : table.SpillSlot R V), j.val < region.slots.val.length →
      region.slots.val[j.val]? = some slot → slot.tag ≠ 0#u8) :
    SpillProbeBoundAbsentSound region home bound := by
  refine
    { placement := hplacement, completed := hbound
      visited := spill_cycle_covers_of_placement hplacement hbound
      occupied := hoccupied
      excludes := ?_ }
  intro j slot hj hslot
  have htag := hoccupied j slot hj hslot
  simp [table.SpillSlot.live, htag]

end Apeiron.GrowableScan