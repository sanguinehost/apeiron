-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): the AVX2 premise of the read is discharged.
import Map.ExecutedRead
import Map.PrefixMasks
open Aeneas Aeneas.Std Result

/-!
# The executed read without the AVX2 premise

`Avx2Spec` (`Map/ExecutedProbe.lean`) is the contract of `prefix_masks_avx2` on a readable window.
`Apeiron.PrefixMasks.avxBranch_spec` (`Map/PrefixMasks.lean`) proves it from the modelled `_mm256_*`
intrinsics: `avx2_spec_holds` restates that result as `MaskSpec`.  `get_refines_uncond` and
`coord_get_refines_uncond` are `get_refines` and `coord_get_refines` with the premise removed.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- The AVX2 contract holds: `prefix_masks_avx2` over a readable window returns `MaskSpec`. -/
theorem avx2_spec_holds : Avx2Spec := by
  intro coords home coord p p1 r hlen hp hp1 hr
  have hav : Apeiron.PrefixMasks.avxBranch coords home coord = ok r := by
    unfold Apeiron.PrefixMasks.avxBranch
    rw [hp]; simp only [bind_tc_ok]
    rw [hp1]; simp only [bind_tc_ok]
    exact hr
  obtain ⟨r', hr', hs⟩ := Apeiron.PrefixMasks.avxBranch_spec coords home coord hlen
  rw [hav] at hr'
  have hrr : r = r' := (Result.ok_injective hr')
  subst hrr
  obtain ⟨h1, h2, hj⟩ := hs
  intro k
  by_cases hk : k < 8
  · obtain ⟨x, hx, hb1, hb2⟩ := hj k hk
    refine ⟨?_, ?_⟩
    · rw [hb1]
      simp only [hk, true_and, hx, Option.some.injEq, decide_eq_true_eq]
    · rw [hb2]
      simp only [hk, true_and, hx, Option.some.injEq, decide_eq_true_eq]
      constructor
      · intro h; exact ⟨x, rfl, (EMPTY_eq_iff x).mp h⟩
      · rintro ⟨y, hy, hy2⟩
        subst hy
        exact (EMPTY_eq_iff x).mpr hy2
  · refine ⟨?_, ?_⟩
    · have h8 : 256 ≤ 2 ^ k := by
        calc 256 = 2 ^ 8 := by norm_num
          _ ≤ 2 ^ k := Nat.pow_le_pow_right (by norm_num) (by omega)
      have : r.1.val.testBit k = false := Nat.testBit_lt_two_pow (by omega)
      simp [this, hk]
    · have h8 : 256 ≤ 2 ^ k := by
        calc 256 = 2 ^ 8 := by norm_num
          _ ≤ 2 ^ k := Nat.pow_le_pow_right (by norm_num) (by omega)
      have : r.2.val.testBit k = false := Nat.testBit_lt_two_pow (by omega)
      simp [this, hk]

/-- **`get_refines`, unconditional** (`Stored<u64>`): no AVX2 premise. -/
theorem get_refines_uncond
    (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : TableRep s m live) (k : U64) (r : Option I64)
    (h : word_hashtrinity_get s k = ok r) :
    (∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k) ∧ (¬ live k → r = none) :=
  get_refines (fun _ => avx2_spec_holds) s m live hR k r h

/-- **`coord_get_refines`, unconditional** (`Regen<u64>`): no AVX2 premise. -/
theorem coord_get_refines_uncond
    (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : CoordRep s m live) (k : U64) (r : Option I64)
    (h : word_coordmap_get s k = ok r) :
    (∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k) ∧ (¬ live k → r = none) :=
  coord_get_refines (fun _ => avx2_spec_holds) s m live hR k r h

end Apeiron.MapExecuted
