/-
Refinement of the extracted binary16 splitter `word_f16_split`: every finite pattern
(`f16Exp h ≠ 31`) splits into an exact signed Q32.32 integer product, and every non-finite
pattern is refused with `none`.

Depends on: `Numeric.DequantRefinement` (`f16_to_fixed_exact`, `f16_to_fixed_nonfinite`).
Excludes: `act_scale`, `finish`, `ChartDot`, the audit registry, tool pins and all other
extracted source. Audit registration and a real-source mutation/restoration control are
separate and are not established here.
Grade: finite exactness and non-finite refusal are `proved` by a direct Lean check of this
module.
-/
import Numeric.DequantRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-- The extracted splitter returns `none` for every binary16 non-finite exponent. -/
theorem word_f16_split_nonfinite (h : U16) (he : f16Exp h = 31) :
    word_f16_split h = ok none := by
  obtain ⟨sign, hs, -⟩ := f16_shr15 h
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hiv' : i.val = h.val >>> 10 := by rw [hiv]; rfl
  have hE := f16_exp_eq h i hiv'
  have he31 : (i &&& 31#u16) = 31#u16 := by
    apply UScalar.eq_of_val_eq
    rw [hE, he]
    simp
  have heconv : core.convert.num.FromU32U16.from (i &&& 31#u16) = 31#u32 := by
    apply UScalar.eq_of_val_eq
    rw [core.convert.num.FromU32U16.from_val_eq, hE, he]
    simp
  rw [word_f16_split, pow2.f16_split]
  simp only [hs, hi, lift, bind_tc_ok, if_pos heconv]

/-- Every finite binary16 pattern splits into an exact signed Q32.32 integer product. -/
theorem word_f16_split_exact (h : U16) (he : f16Exp h ≠ 31) :
    ∃ m : I32, ∃ k : U32, word_f16_split h = ok (some (m, k)) ∧
      m.val * (2 : ℤ) ^ k.val = f16Value h := by
  obtain ⟨sign, hs, hsv⟩ := f16_shr15 h
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hiv' : i.val = h.val >>> 10 := by rw [hiv]; rfl
  have hE := f16_exp_eq h i hiv'
  have hM := f16_man_eq h
  have hEl := f16_exp_lt h
  have hMl := f16_man_lt h
  let e := core.convert.num.FromU32U16.from (i &&& 31#u16)
  let f := UScalar.hcast .I32 (h &&& 1023#u16)
  have hev : e.val = f16Exp h := by
    rw [core.convert.num.FromU32U16.from_val_eq, hE]
  have hfv : f.val = (f16Man h : ℤ) := by
    dsimp [f]
    rw [UScalar.hcast_val_eq, hM]
    apply Int.bmod_eq_of_le
    all_goals norm_num
    all_goals omega
  have he31 : e ≠ 31#u32 := by
    intro hc
    have := congrArg UScalar.val hc
    simp only [hev] at this
    apply he
    simpa using this
  have hpair : ∃ m : I32, ∃ k : U32,
      (if e = 0#u32 then ok (f, 8#u32) else do
        let m ← f + 1024#i32
        let k ← e + 7#u32
        ok (m, k)) = ok (m, k) ∧
      0 ≤ m.val ∧ m.val * (2 : ℤ) ^ k.val = f16Mag h := by
    by_cases he0 : e = 0#u32
    · have hE0 : f16Exp h = 0 := by rw [← hev, he0]; simp
      refine ⟨f, 8#u32, ?_, ?_, ?_⟩
      · simp only [if_pos he0]
      · rw [hfv]; positivity
      · rw [hfv, f16Mag, if_pos hE0]; rfl
    · have hE0 : f16Exp h ≠ 0 := by
        intro hc
        apply he0
        apply UScalar.eq_of_val_eq
        rw [hev, hc]
        simp
      obtain ⟨m, hm, hmv⟩ := WP.spec_imp_exists
        (I32.add_spec (x := f) (y := 1024#i32)
          (by rw [hfv, I32.min_eq]; simp; omega)
          (by rw [hfv, I32.max_eq]; simp; omega))
      obtain ⟨k, hk, hkv⟩ := WP.spec_imp_exists
        (U32.add_spec (x := e) (y := 7#u32)
          (by rw [hev, U32.max_eq]; simp; omega))
      have hmv' : m.val = ((1024 + f16Man h : ℕ) : ℤ) := by
        rw [hmv, hfv]; push_cast; simp; ring
      have hkv' : k.val = f16Exp h + 7 := by rw [hkv, hev]; simp
      refine ⟨m, k, ?_, ?_, ?_⟩
      · simp only [if_neg he0, hm, hk, bind_tc_ok]
      · rw [hmv']; positivity
      · rw [hmv', hkv', f16Mag, if_neg hE0]
  obtain ⟨m, k, hpair, hm0, hmag⟩ := hpair
  rw [word_f16_split, pow2.f16_split]
  simp only [hi, lift, bind_tc_ok]
  change ∃ m' : I32, ∃ k' : U32,
    (if e = 31#u32 then ok none else do
      let (m, k) ← if e = 0#u32 then ok (f, 8#u32) else do
        let m ← f + 1024#i32
        let k ← e + 7#u32
        ok (m, k)
      let sign ← h >>> 15#i32
      if sign = 1#u16 then do
        let m ← -. m
        ok (some (m, k))
      else ok (some (m, k))) = ok (some (m', k')) ∧
      m'.val * (2 : ℤ) ^ k'.val = f16Value h
  simp only [if_neg he31, hpair, bind_tc_ok, hs]
  by_cases hs1 : sign = 1#u16
  · have hS : f16Sign h = 1 := by rw [← hsv, hs1]; simp
    obtain ⟨neg, hneg, hnegv⟩ := WP.spec_imp_exists
      (HNeg.hNeg.step m (by
        intro hc
        rw [IScalar.min_IScalarTy_I32_eq, I32.min_eq] at hc
        omega))
    simp only [if_pos hs1]
    refine ⟨neg, k, ?_, ?_⟩
    · change (do let z ← -.m; ok (some (z, k))) = ok (some (neg, k))
      rw [hneg]
      simp only [bind_tc_ok]
    · rw [hnegv, neg_mul, hmag, f16Value, if_pos hS]
  · have hS : f16Sign h ≠ 1 := by
      intro hc
      apply hs1
      apply UScalar.eq_of_val_eq
      rw [hsv, hc]
      simp
    simp only [if_neg hs1]
    exact ⟨m, k, rfl, by rw [hmag, f16Value, if_neg hS]⟩

/-- A reachable finite normal input jointly satisfies the domain and exact successful result. -/
theorem word_f16_split_finite_witness :
    ∃ h : U16, ∃ m : I32, ∃ k : U32,
      f16Exp h = 15 ∧ word_f16_split h = ok (some (m, k)) ∧
      m.val * (2 : ℤ) ^ k.val = 2 ^ 32 := by
  have he : f16Exp (15360#u16) ≠ 31 := by norm_num [f16Exp]
  obtain ⟨m, k, hr, hv⟩ := word_f16_split_exact (15360#u16) he
  refine ⟨15360#u16, m, k, ?_, hr, ?_⟩
  · norm_num [f16Exp]
  · rw [hv]
    norm_num [f16Value, f16Sign, f16Mag, f16Exp, f16Man]

/-- Positive infinity jointly satisfies the non-finite domain and extracted refusal result. -/
theorem word_f16_split_nonfinite_witness :
    ∃ h : U16, f16Exp h = 31 ∧ word_f16_split h = ok none := by
  have he : f16Exp (31744#u16) = 31 := by norm_num [f16Exp]
  exact ⟨31744#u16, he, word_f16_split_nonfinite _ he⟩

end apeiron_word
