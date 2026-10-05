import Growable.GaplessRead

/-!
# `OPS-1`: the fixed `Region.home` returns a slot index

`Region.home` (`src/table/state.rs`) now reduces both of its branches through the probe cycle mask
`bound - 1`, where `bound = Region.bound r = r.slots.len()`.  The value is therefore in
`[0, bound)` for every nonempty region, independent of any relation between the `coords` and `slots`
lengths: the direct branch's `coord_of` value and the spread branch's top-`bits` value are both `&&&`
the mask, and `x &&& m ≤ m < bound`.

This is the fact `OPS-1` named as missing.  It is proved here on the regenerated body
(`proof/ApeironWord.lean`, `table.Region.home`), so the read's resident lane is a real slot index and
the tail sweep `(home + step) &&& mask` for `step = 1 .. bound - 1` visits every other lane.  The
proof only reads the generated body: it assumes no well-formedness of the region beyond `bound ≠ 0`.
-/

namespace Apeiron.GrowableHomeBound

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedVariables false
set_option linter.dupNamespace false

/-- Generic successful-bind inversion on the generated `do` form. -/
private theorem bind_ok_inv {α β : Type} (x : Result α) (f : α → Result β) (r : β)
    (h : (do let y ← x; f y) = ok r) : ∃ y, x = ok y ∧ f y = ok r := by
  cases hx : x with
  | ret y => rw [hx, bind_tc_ok] at h; exact ⟨y, rfl, h⟩
  | vis e k => rw [hx, bind_tc_vis] at h; exact (vis_not_ok h).elim
  | div => rw [hx, bind_tc_div] at h; exact (div_not_ok h).elim

/-- `m &&& mask ≤ mask` at the value level, for `Usize`. -/
private theorem and_val_le (x mask : Std.Usize) : (x &&& mask).val ≤ mask.val := by
  rw [UScalar.val_and]
  exact Nat.and_le_right

/-- **`OPS-1`, resolved by the source fix.**  Every value the fixed `table.Region.home` returns is
    strictly below `table.Region.bound`, for every nonempty region and every placement. -/
theorem home_lt_bound {R V : Type} (r : table.Region R V) (place : Std.U64)
    (home bound : Std.Usize)
    (hhome : table.Region.home r place = ok home)
    (hbound : table.Region.bound r = ok bound) (hne : bound.val ≠ 0) :
    home.val < bound.val := by
  have hb : r.slots.val.length = bound.val := by
    have h1 : alloc.vec.Vec.len r.slots = bound := by
      have := Result.ok_injective (by simpa only [table.Region.bound] using hbound)
      simpa using this
    simpa [alloc.vec.Vec.len_val] using congrArg (fun x : Std.Usize => x.val) h1
  have hbne : alloc.vec.Vec.len r.slots ≠ 0#usize := by
    intro h
    apply hne
    rw [← hb]
    have := congrArg (fun x : Std.Usize => x.val) h
    simpa [alloc.vec.Vec.len_val] using this
  have hbpos : 1 ≤ bound.val := by omega
  unfold table.Region.home at hhome
  dsimp only at hhome
  rw [if_neg hbne] at hhome
  obtain ⟨mask, hmask, hhome⟩ := bind_ok_inv (alloc.vec.Vec.len r.slots - 1#usize)
    (fun mask => do
      let c ← table.Region.coord_of R V place
      let len := alloc.vec.Vec.len r.coords
      if len = 0#usize then ok 0#usize
      else
        let i ← lift (UScalar.cast .U64 len)
        if c < i then
          let i1 ← lift (UScalar.cast .Usize c)
          ok (i1 &&& mask)
        else
          let i1 ← lift (core.num.U64.wrapping_mul c table.HOME_SPREAD)
          let i2 ← 64#u32 - r.bits
          let i3 ← i1 >>> i2
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask)) home hhome
  have hmaskval : mask.val = bound.val - 1 := by
    have h := @UScalar.sub_equiv .Usize (alloc.vec.Vec.len r.slots) 1#usize
    rw [hmask, Result.match.ok] at h
    simp at h
    omega
  obtain ⟨c, hc, hhome⟩ := bind_ok_inv (table.Region.coord_of R V place)
    (fun c => do
      let len := alloc.vec.Vec.len r.coords
      if len = 0#usize then ok 0#usize
      else
        let i ← lift (UScalar.cast .U64 len)
        if c < i then
          let i1 ← lift (UScalar.cast .Usize c)
          ok (i1 &&& mask)
        else
          let i1 ← lift (core.num.U64.wrapping_mul c table.HOME_SPREAD)
          let i2 ← 64#u32 - r.bits
          let i3 ← i1 >>> i2
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask)) home hhome
  by_cases hlen : alloc.vec.Vec.len r.coords = 0#usize
  · dsimp only at hhome
    rw [if_pos hlen] at hhome
    have hz : home = 0#usize := (Result.ok_injective hhome).symm
    rw [hz]
    exact Nat.pos_of_ne_zero hne
  · dsimp only at hhome
    rw [if_neg hlen] at hhome
    obtain ⟨i, hi, hhome⟩ := bind_ok_inv
      (lift (UScalar.cast .U64 (alloc.vec.Vec.len r.coords)))
      (fun i => if c < i then
        (do let i1 ← lift (UScalar.cast .Usize c); ok (i1 &&& mask))
      else
        (do
          let i1 ← lift (core.num.U64.wrapping_mul c table.HOME_SPREAD)
          let i2 ← 64#u32 - r.bits
          let i3 ← i1 >>> i2
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask))) home hhome
    by_cases hci : c < i
    · rw [if_pos hci] at hhome
      obtain ⟨i1, hi1, hhome⟩ := bind_ok_inv (lift (UScalar.cast .Usize c))
        (fun i1 => ok (i1 &&& mask)) home hhome
      have he : home = i1 &&& mask := (Result.ok_injective hhome).symm
      rw [he]
      exact lt_of_le_of_lt (and_val_le i1 mask) (by omega)
    · rw [if_neg hci] at hhome
      obtain ⟨w, hw, hhome⟩ := bind_ok_inv
        (lift (core.num.U64.wrapping_mul c table.HOME_SPREAD))
        (fun w => do
          let i2 ← 64#u32 - r.bits
          let i3 ← w >>> i2
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask)) home hhome
      obtain ⟨sh, hsh, hhome⟩ := bind_ok_inv (64#u32 - r.bits)
        (fun sh => do
          let i3 ← w >>> sh
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask)) home hhome
      obtain ⟨z, hz, hhome⟩ := bind_ok_inv (w >>> sh)
        (fun z => do
          let i4 ← lift (UScalar.cast .Usize z)
          ok (i4 &&& mask)) home hhome
      obtain ⟨i4, hi4, hhome⟩ := bind_ok_inv (lift (UScalar.cast .Usize z))
        (fun i4 => ok (i4 &&& mask)) home hhome
      have he : home = i4 &&& mask := (Result.ok_injective hhome).symm
      rw [he]
      exact lt_of_le_of_lt (and_val_le i4 mask) (by omega)

end Apeiron.GrowableHomeBound

#print axioms Apeiron.GrowableHomeBound.home_lt_bound
