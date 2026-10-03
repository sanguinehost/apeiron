import ApeironWord
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases

/-!
# Spill-slot tag fingerprints and publication readback

The executing spill representation is a vector of `table.SpillSlot`s read by the linear probe in
`table.probe.spill_read`. This file proves the tag facts the probe depends on: every generated tag
carries the live bit, `table.matches` is exactly the conjunction of `Sigma` equality and the
carrier key comparison, and successful publication writes the generated tag at its index.
-/

namespace Apeiron.GrowableScan

open Aeneas Aeneas.Std Result
open apeiron_word

/-- Every generated tag carries the live bit and is therefore nonzero. -/
theorem tag_ne_zero (place : U64) (tag : U8) (htag : table.tag place = ok tag) :
    tag ≠ 0#u8 := by
  unfold table.tag at htag
  simp only [lift, bind_tc_ok] at htag
  have hex : ∃ rest : U8, tag = table.TAG_LIVE ||| rest := by
    cases h1 : table.TAG_FINGERPRINT_SHIFT with
    | vis eff k => simp [h1] at htag
    | div => simp [h1] at htag
    | ret i =>
        simp only [h1, bind_tc_ok] at htag
        cases h2 : core.num.U64.wrapping_mul place table.TAG_SPREAD >>> i with
        | vis eff k => simp [h2] at htag
        | div => simp [h2] at htag
        | ret shifted =>
            simp only [h2, bind_tc_ok] at htag
            cases h3 : table.TAG_FINGERPRINT_MASK with
            | vis eff k => simp [h3] at htag
            | div => simp [h3] at htag
            | ret mask =>
                simp only [h3, bind_tc_ok] at htag
                exact ⟨_, (Result.ok_injective htag).symm⟩
  obtain ⟨rest, hrest⟩ := hex
  intro hzero
  have hbits : table.TAG_LIVE.bv ||| rest.bv = 0#8 := by
    have h := congrArg UScalar.bv (hrest.symm.trans hzero)
    simpa [UScalar.bv_or] using h
  have hlive := (BitVec.or_eq_zero_iff.mp hbits).1
  unfold table.TAG_LIVE at hlive
  exact (by decide : (128#8 : BitVec 8) ≠ 0#8) hlive

/-- The generated tag mask is the seven-bit low mask. -/
private theorem mask_value : table.TAG_FINGERPRINT_MASK = ok 127#u8 := by
  unfold table.TAG_FINGERPRINT_MASK table.TAG_FINGERPRINT_BITS
  obtain ⟨shifted, hshifted, hsp⟩ :=
    WP.spec_imp_exists (U8.ShiftLeft_spec 1#u8 7#u32 (by norm_num))
  rw [hshifted]
  simp only [bind_tc_ok]
  have h1 : (1#u8 : U8).val = 1 := by decide
  have h7 : (7#u32 : U32).val = 7 := by decide
  have hval : shifted.val = 128 := by
    rw [hsp.1, h1, h7]
    simp [U8.size, U8.numBits]
  have hle : (1#u8 : U8).val ≤ shifted.val := by rw [hval]; norm_num
  obtain ⟨sub, hsub, hsp2⟩ := WP.spec_imp_exists (U8.sub_spec (x := shifted) (y := 1#u8) hle)
  rw [hsub]
  apply congrArg ok
  apply UScalar.eq_of_val_eq
  have h127 : (127#u8 : U8).val = 127 := by decide
  rw [hsp2.1, hval, h1, h127]

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

/-- Distinct full keys can have the same generated seven-bit fingerprint: place zero and
`PHI_INV` both map to the zero fingerprint because `PHI_INV` inverts the spread multiplier. -/
theorem spillFingerprint_collision_witness :
    ∃ left right : cell.Sigma × U64,
      left ≠ right ∧
      spillFingerprint (fun _ key => ok { place := key, direct_word := none }) left =
        spillFingerprint (fun _ key => ok { place := key, direct_word := none }) right := by
  refine ⟨(cell.Sigma.Parabolic, 0#u64),
    (cell.Sigma.Parabolic, 17428512612931826493#u64), ?_, ?_⟩
  · intro heq
    have := congrArg (fun identity : cell.Sigma × U64 => identity.2.val) heq
    norm_num at this
  · unfold spillFingerprint table.tag
    simp only [lift, bind_tc_ok]
    have hshift : table.TAG_FINGERPRINT_SHIFT = ok 57#u32 := by
      unfold table.TAG_FINGERPRINT_SHIFT table.TAG_FINGERPRINT_BITS
      rfl
    rw [hshift, mask_value]
    simp only [bind_tc_ok]
    obtain ⟨s0, hs0, hsp0⟩ := WP.spec_imp_exists
      (U64.ShiftRight_spec (core.num.U64.wrapping_mul 0#u64 table.TAG_SPREAD)
        57#u32 (by norm_num))
    obtain ⟨s1, hs1, hsp1⟩ := WP.spec_imp_exists
      (U64.ShiftRight_spec (core.num.U64.wrapping_mul 17428512612931826493#u64 table.TAG_SPREAD)
        57#u32 (by norm_num))
    have h0v : (0#u64 : U64).val = 0 := rfl
    have h57 : (57#u32 : U32).val = 57 := by decide
    have hx : (17428512612931826493#u64 : U64).val = 17428512612931826493 := by decide
    have hphi : (11400714819323198485#u64 : U64).val = 11400714819323198485 := by decide
    have hs0z : s0 = 0#u64 := by
      apply UScalar.eq_of_val_eq
      rw [hsp0.1, h57, h0v]
      unfold table.TAG_SPREAD coord.PHI
      rw [core.num.U64.wrapping_mul_val_eq, h0v, hphi]
      simp [UScalar.size_UScalarTyU64, U64.size, U64.numBits]
    have hs1z : s1 = 0#u64 := by
      apply UScalar.eq_of_val_eq
      rw [hsp1.1, h57, h0v]
      unfold table.TAG_SPREAD coord.PHI
      rw [core.num.U64.wrapping_mul_val_eq, hx, hphi]
      simp [UScalar.size_UScalarTyU64, U64.size, U64.numBits]
    rw [hs0, hs1, hs0z, hs1z]

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

/-- Successful publication writes the generated tag at the published index. -/
theorem publish_live_tag_readback {R V : Type}
    (before after : table.Region R V) (index : Usize)
    (payload : cell.Sigma × R × V) (place : U64)
    (hpub : table.Region.publish_live before index payload place = ok after) :
    ∃ (tag : U8) (slot : table.SpillSlot R V),
      table.tag place = ok tag ∧ tag ≠ 0#u8 ∧
      after.slots.val[index.val]? = some slot ∧ slot.tag = tag := by
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
      cases hlive : before.live + 1#usize with
      | vis eff k => simp [hlive] at hpub
      | div => simp [hlive] at hpub
      | ret live1 =>
          rw [hlive] at hpub
          simp only [bind_tc_ok] at hpub
          have hafter := (Result.ok_injective hpub).symm
          subst after
          refine ⟨tag, { ss3 with tag := tag }, rfl, tag_ne_zero place tag htag, ?_, rfl⟩
          rw [alloc.vec.Vec.set_val_eq]
          exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp hget3).1

/-- Successful `table.matches` entails both generated comparisons. -/
theorem fingerprint_hit_requires_full_matches {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (slot : table.SpillSlot R V)
    (querySigma : cell.Sigma) (key : K)
    (address : table.TableAddress) (fingerprint : U8)
    (_htag : table.tag address.place = ok fingerprint)
    (hmatch : table.matches keyRep slot querySigma key address = ok true) :
    cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma querySigma = ok true ∧
      keyRep.«matches» querySigma slot.held key address = ok true := by
  unfold table.matches at hmatch
  cases heq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma querySigma with
  | vis eff k => simp [heq] at hmatch
  | div => simp [heq] at hmatch
  | ret equal =>
      cases equal <;> simp_all

end Apeiron.GrowableScan
