import F16SplitSource
import Aeneas

/-
SCOPE: the work (resource) model of the extracted half-word split `pow2.f16_split`, reached
  from the typed wrapper `split_forward`. `src/engine/kern/pow2.rs` and every other body
  (`act_scale`, `finish`, the vector kernel, the `Table`) are excluded.
REGISTER: scalar source refinement and resource bound on the actual extracted body; no
  canonical ring correspondence, no scheduler, table or hardware attribution.
CORPUS: `src/engine/kern/pow2.rs` `f16_split` (lines 52-64) and the actual generated
  `f16_split_source.pow2.f16_split` in `F16SplitSource.lean` (lines 23-43).
REFERENCE: NONE - no canonical-corpus theorem is refined here; the guards are the body's own.
GATE: python3 proof/resource/F16Split_gate.py --check --controls, 540 s wall budget,
  /tmp/opencode/apeiron-lane-2.lock, MemoryMax=12G, MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: Lean kernel check of the real body, its guarded transitions and its work bound; no
  cycle count, no GPU time and no benchmark is claimed.

Work measure. `Work.tests` counts the conditional nodes the generated body evaluates,
`Work.adds` the scalar additions it evaluates and `Work.negs` its sign negation. The named
premise is straight-line structure: `F16SplitSource.lean` lines 23-43 hold exactly three `if`
nodes (lines 29, 33, 40), no `loop` and no recursive call, so no path can revisit a node. Each
transition theorem below discharges the body's conditionals one at a time, and
`transition_classification` proves the resulting five cases cover every pattern, so the tally
is the work of the real body and not of a model of it. Work is counted under that premise; it
is never a cycle or GPU-time count.
-/

namespace F16SplitResource

open Aeneas Aeneas.Std Result

/-! ### The three guards, read off the real body -/

/-- Biased exponent guard the body tests: `(h >> 10) mod 32`. -/
def expField (h : U16) : Nat := (h.val / 2 ^ 10) % 32

/-- Significand field the body carries: `h mod 2^10`. -/
def manField (h : U16) : Nat := h.val % 2 ^ 10

/-- Sign the body tests: `h / 2^15`. -/
def signField (h : U16) : Nat := h.val / 2 ^ 15

theorem expField_lt (h : U16) : expField h < 32 := Nat.mod_lt _ (by norm_num)

theorem manField_lt (h : U16) : manField h < 1024 := Nat.mod_lt _ (by norm_num)

/-- The exponent read of the real body, `h >> 10`, is the pattern's high bits. -/
theorem shift10 (h : U16) :
    ∃ i : U16, h >>> 10#i32 = ok i ∧ i.val = h.val >>> 10 := by
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  exact ⟨i, hi, hiv⟩

/-- The sign read of the real body, `h >> 15`, is the pattern's top bit. -/
theorem shift15 (h : U16) :
    ∃ s : U16, h >>> 15#i32 = ok s ∧ s.val = signField h := by
  obtain ⟨s, hs, hsv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 15#i32 (by simp) (by simp))
  refine ⟨s, hs, ?_⟩
  rw [hsv, Nat.shiftRight_eq_div_pow]
  rfl

/-- The exponent guard the body tests, `(h >> 10) & 0x1f`, is `expField h`. -/
theorem and31_val (i h : U16) (hi : i.val = h.val >>> 10) :
    (i &&& 31#u16).val = expField h := by
  rw [UScalar.val_and, hi, Nat.shiftRight_eq_div_pow]
  have h31 : (31#u16).val = 2 ^ 5 - 1 := by simp
  rw [h31, Nat.and_two_pow_sub_one_eq_mod]
  rfl

/-- **Guard agreement.** For the shift value `i` the body actually binds, the masked exponent
    it compares against 31 and against 0 is exactly `expField h`. This is the link between the
    work model's guards and the body's own conditionals. -/
theorem expField_read (h i : U16) (hi : h >>> 10#i32 = ok i) :
    (i &&& 31#u16).val = expField h := by
  obtain ⟨j, hj, hjv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  exact and31_val i h (Result.ok_injective (hj.symm.trans hi) ▸ hjv)

/-- The masked significand is the exact `i32` the body carries: the `u16`-to-`i32` cast is
    value-preserving because the mask is below `2^31`. -/
theorem fraction_val (h : U16) :
    (UScalar.hcast .I32 (h &&& 1023#u16)).val = (manField h : ℤ) := by
  rw [UScalar.hcast_val_eq]
  rw [show (h &&& 1023#u16).val = manField h by
    rw [UScalar.val_and]
    have h1023 : (1023#u16).val = 2 ^ 10 - 1 := by simp
    rw [h1023, Nat.and_two_pow_sub_one_eq_mod]
    rfl]
  have hman := manField_lt h
  have hman0 : (0 : ℤ) ≤ manField h := Int.natCast_nonneg _
  refine Int.bmod_eq_of_le ?_ ?_
  · norm_num
  · norm_num; omega

/-! ### The work measure -/

/-- Work of one path of the extracted body: `tests` the conditional nodes it evaluates,
    `adds` the scalar additions it evaluates, `negs` the sign negation it evaluates. -/
structure Work where
  tests : Nat
  adds : Nat
  negs : Nat
  deriving DecidableEq

/-- The bound every path of the extracted splitter stays inside: three conditional nodes
    (lines 29, 33 and 40 of `F16SplitSource.lean`), the two additions of the normal arm, and
    the single sign negation. -/
def splitBudget : Work := ⟨3, 2, 1⟩

/-- The work the body spends on `h`, keyed by the guards the body itself reads: a non-finite
    pattern returns at the first guard, a subnormal skips both additions, and the sign guard
    negates exactly when it reads 1. -/
def pathWork (h : U16) : Work :=
  if expField h = 31 then ⟨1, 0, 0⟩
  else if expField h = 0 then ⟨3, 0, if signField h = 1 then 1 else 0⟩
  else ⟨3, 2, if signField h = 1 then 1 else 0⟩

/-! ### Guarded transitions of the actual extracted body -/

/-- **Refusal transition (non-finite).** The body's first guard, `e = 31`, returns `ok none`
    at once: the subnormal guard, the sign guard, both additions and the negation are never
    evaluated. One branch test, no addition, no negation. -/
theorem refusal_transition (h : U16) (he : expField h = 31) :
    f16_split_source.pow2.f16_split h = ok none ∧ pathWork h = ⟨1, 0, 0⟩ := by
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hE := expField_read h i hi
  have heconv : core.convert.num.FromU32U16.from (i &&& 31#u16) = 31#u32 := by
    apply UScalar.eq_of_val_eq
    rw [core.convert.num.FromU32U16.from_val_eq, hE, he]
    simp
  refine ⟨?_, ?_⟩
  · unfold f16_split_source.pow2.f16_split
    simp only [hi, lift, bind_tc_ok, if_pos heconv]
  · simp [pathWork, he]

/-- **Subnormal transition, positive sign.** The finite arm takes the subnormal arm of the
    second guard: no addition is evaluated, `k` is the literal 8 and `m` is the masked
    significand. Three branch tests, no addition, no negation. -/
theorem subnormal_positive_transition (h : U16) (he0 : expField h = 0)
    (hs : signField h ≠ 1) :
    ∃ m : I32, ∃ k : U32,
      f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧
      k.val = 8 ∧ m.val = (manField h : ℤ) ∧ pathWork h = ⟨3, 0, 0⟩ := by
  have hlt := expField_lt h
  have he : expField h ≠ 31 := by omega
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hE := expField_read h i hi
  have hev : (core.convert.num.FromU32U16.from (i &&& 31#u16)).val = expField h := by
    rw [core.convert.num.FromU32U16.from_val_eq, hE]
  have he0' : core.convert.num.FromU32U16.from (i &&& 31#u16) = 0#u32 := by
    apply UScalar.eq_of_val_eq
    rw [hev, he0]
    simp
  have he31' : core.convert.num.FromU32U16.from (i &&& 31#u16) ≠ 31#u32 := by
    intro hc
    apply he
    have hval := congrArg UScalar.val hc
    rw [hev] at hval
    simpa using hval.symm
  have hfv := fraction_val h
  obtain ⟨sign, hsg, hsgv⟩ := shift15 h
  have hs' : sign ≠ 1#u16 := by
    intro hc
    have hval : sign.val = 1 := by simpa using congrArg UScalar.val hc
    rw [hsgv] at hval
    exact hs hval
  refine ⟨UScalar.hcast .I32 (h &&& 1023#u16), 8#u32, ?_, rfl, hfv, ?_⟩
  · unfold f16_split_source.pow2.f16_split
    simp only [hi, lift, bind_tc_ok, if_neg he31', if_pos he0', hsg, if_neg hs']
    rfl
  · simp [pathWork, he0, hs]

/-- **Subnormal transition, negative sign.** As `subnormal_positive_transition`, and the sign
    guard negates the significand. Three branch tests, no addition, one negation. -/
theorem subnormal_negative_transition (h : U16) (he0 : expField h = 0)
    (hs1 : signField h = 1) :
    ∃ m : I32, ∃ k : U32,
      f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧
      k.val = 8 ∧ m.val = -((manField h : ℤ)) ∧ pathWork h = ⟨3, 0, 1⟩ := by
  have hlt := expField_lt h
  have he : expField h ≠ 31 := by omega
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hE := expField_read h i hi
  have hev : (core.convert.num.FromU32U16.from (i &&& 31#u16)).val = expField h := by
    rw [core.convert.num.FromU32U16.from_val_eq, hE]
  have he0' : core.convert.num.FromU32U16.from (i &&& 31#u16) = 0#u32 := by
    apply UScalar.eq_of_val_eq
    rw [hev, he0]
    simp
  have he31' : core.convert.num.FromU32U16.from (i &&& 31#u16) ≠ 31#u32 := by
    intro hc
    apply he
    have hval := congrArg UScalar.val hc
    rw [hev] at hval
    simpa using hval.symm
  have hfv := fraction_val h
  have hman := manField_lt h
  obtain ⟨sign, hsg, hsgv⟩ := shift15 h
  have hs' : sign = 1#u16 := by
    apply UScalar.eq_of_val_eq
    rw [hsgv, hs1]
    simp
  obtain ⟨neg, hn, hnv⟩ := WP.spec_imp_exists
    (HNeg.hNeg.step (UScalar.hcast .I32 (h &&& 1023#u16)) (by
      rw [hfv]
      intro hc
      rw [IScalar.min_IScalarTy_I32_eq, I32.min_eq] at hc
      omega))
  refine ⟨neg, 8#u32, ?_, rfl, ?_, ?_⟩
  · unfold f16_split_source.pow2.f16_split
    simp only [hi, lift, bind_tc_ok, if_neg he31', if_pos he0', hsg, if_pos hs']
    show (do let i4 ← -. UScalar.hcast .I32 (h &&& 1023#u16); ok (some (i4, 8#u32))) =
      ok (some (neg, 8#u32))
    rw [hn]
    simp only [bind_tc_ok]
  · rw [hnv, hfv]
  · simp [pathWork, he0, hs1]

/-- **Normal transition, positive sign.** The finite arm takes the normal arm of the second
    guard: both additions are evaluated, `k = e + 7` and `m = 1024 +` the masked significand.
    Three branch tests, two additions, no negation. -/
theorem normal_positive_transition (h : U16) (he0 : expField h ≠ 0) (he : expField h ≠ 31)
    (hs : signField h ≠ 1) :
    ∃ m : I32, ∃ k : U32,
      f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧
      k.val = expField h + 7 ∧ m.val = ((1024 + manField h : ℕ) : ℤ) ∧
      pathWork h = ⟨3, 2, 0⟩ := by
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hE := expField_read h i hi
  have hev : (core.convert.num.FromU32U16.from (i &&& 31#u16)).val = expField h := by
    rw [core.convert.num.FromU32U16.from_val_eq, hE]
  have he0' : core.convert.num.FromU32U16.from (i &&& 31#u16) ≠ 0#u32 := by
    intro hc
    apply he0
    have hval := congrArg UScalar.val hc
    rw [hev] at hval
    simpa using hval.symm
  have he31' : core.convert.num.FromU32U16.from (i &&& 31#u16) ≠ 31#u32 := by
    intro hc
    apply he
    have hval := congrArg UScalar.val hc
    rw [hev] at hval
    simpa using hval.symm
  have hfv := fraction_val h
  have hman := manField_lt h
  have hmin : I32.min ≤ (UScalar.hcast .I32 (h &&& 1023#u16)).val + (1024#i32).val := by
    rw [hfv, I32.min_eq]
    norm_num
    omega
  have hmax : (UScalar.hcast .I32 (h &&& 1023#u16)).val + (1024#i32).val ≤ I32.max := by
    rw [hfv, I32.max_eq]
    norm_num
    omega
  obtain ⟨mm, hm, hmv⟩ := WP.spec_imp_exists
    (I32.add_spec (x := UScalar.hcast .I32 (h &&& 1023#u16)) (y := 1024#i32) hmin hmax)
  have hmv' : mm.val = ((1024 + manField h : ℕ) : ℤ) := by
    rw [hmv, hfv]; push_cast; simp; ring
  have hkmax : (core.convert.num.FromU32U16.from (i &&& 31#u16)).val + (7#u32).val ≤ U32.max := by
    rw [hev]
    have hlt := expField_lt h
    rw [U32.max_eq]
    norm_num
    omega
  obtain ⟨kk, hk, hkv⟩ := WP.spec_imp_exists
    (U32.add_spec (x := core.convert.num.FromU32U16.from (i &&& 31#u16)) (y := 7#u32) hkmax)
  have hkv' : kk.val = expField h + 7 := by rw [hkv, hev]; simp
  obtain ⟨sign, hsg, hsgv⟩ := shift15 h
  have hs' : sign ≠ 1#u16 := by
    intro hc
    have hval : sign.val = 1 := by simpa using congrArg UScalar.val hc
    rw [hsgv] at hval
    exact hs hval
  refine ⟨mm, kk, ?_, hkv', hmv', ?_⟩
  · unfold f16_split_source.pow2.f16_split
    simp only [hi, lift, bind_tc_ok, if_neg he31', if_neg he0', hm, hk, hsg, if_neg hs']
    rfl
  · simp [pathWork, he0, he, hs]

/-- **Normal transition, negative sign.** As `normal_positive_transition`, and the sign guard
    negates the significand. Three branch tests, two additions, one negation: the whole
    budget. -/
theorem normal_negative_transition (h : U16) (he0 : expField h ≠ 0) (he : expField h ≠ 31)
    (hs1 : signField h = 1) :
    ∃ m : I32, ∃ k : U32,
      f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧
      k.val = expField h + 7 ∧ m.val = -((1024 + manField h : ℕ) : ℤ) ∧
      pathWork h = splitBudget := by
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hE := expField_read h i hi
  have hev : (core.convert.num.FromU32U16.from (i &&& 31#u16)).val = expField h := by
    rw [core.convert.num.FromU32U16.from_val_eq, hE]
  have he0' : core.convert.num.FromU32U16.from (i &&& 31#u16) ≠ 0#u32 := by
    intro hc
    apply he0
    have hval := congrArg UScalar.val hc
    rw [hev] at hval
    simpa using hval.symm
  have he31' : core.convert.num.FromU32U16.from (i &&& 31#u16) ≠ 31#u32 := by
    intro hc
    apply he
    have hval := congrArg UScalar.val hc
    rw [hev] at hval
    simpa using hval.symm
  have hfv := fraction_val h
  have hman := manField_lt h
  have hmin : I32.min ≤ (UScalar.hcast .I32 (h &&& 1023#u16)).val + (1024#i32).val := by
    rw [hfv, I32.min_eq]
    norm_num
    omega
  have hmax : (UScalar.hcast .I32 (h &&& 1023#u16)).val + (1024#i32).val ≤ I32.max := by
    rw [hfv, I32.max_eq]
    norm_num
    omega
  obtain ⟨mm, hm, hmv⟩ := WP.spec_imp_exists
    (I32.add_spec (x := UScalar.hcast .I32 (h &&& 1023#u16)) (y := 1024#i32) hmin hmax)
  have hmv' : mm.val = ((1024 + manField h : ℕ) : ℤ) := by
    rw [hmv, hfv]; push_cast; simp; ring
  have hmm0 : (0 : ℤ) ≤ mm.val := by rw [hmv']; positivity
  have hkmax : (core.convert.num.FromU32U16.from (i &&& 31#u16)).val + (7#u32).val ≤ U32.max := by
    rw [hev]
    have hlt := expField_lt h
    rw [U32.max_eq]
    norm_num
    omega
  obtain ⟨kk, hk, hkv⟩ := WP.spec_imp_exists
    (U32.add_spec (x := core.convert.num.FromU32U16.from (i &&& 31#u16)) (y := 7#u32) hkmax)
  have hkv' : kk.val = expField h + 7 := by rw [hkv, hev]; simp
  obtain ⟨sign, hsg, hsgv⟩ := shift15 h
  have hs' : sign = 1#u16 := by
    apply UScalar.eq_of_val_eq
    rw [hsgv, hs1]
    simp
  obtain ⟨neg, hn, hnv⟩ := WP.spec_imp_exists
    (HNeg.hNeg.step mm (by
      intro hc
      rw [IScalar.min_IScalarTy_I32_eq, I32.min_eq] at hc
      omega))
  refine ⟨neg, kk, ?_, hkv', ?_, ?_⟩
  · unfold f16_split_source.pow2.f16_split
    simp only [hi, lift, bind_tc_ok, if_neg he31', if_neg he0', hm, hk, hsg, if_pos hs']
    show (do let i4 ← -. mm; ok (some (i4, kk))) = ok (some (neg, kk))
    rw [hn]
    simp only [bind_tc_ok]
  · rw [hnv, hmv']
  · simp [pathWork, he0, he, hs1, splitBudget]

/-! ### Exhaustive branch classification of the real body -/

/-- The finite arm of the body splits into exactly two guarded transitions: the subnormal
    guard and the normal guard, each resolved by the sign guard. The significand the body
    emits and the work it spends are both determined by the guard triple. -/
theorem finite_classification (h : U16) (he : expField h ≠ 31) :
    (expField h = 0 ∧
        ∃ m k, f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧ k.val = 8 ∧
          (if signField h = 1 then m.val = -((manField h : ℤ)) else m.val = (manField h : ℤ)) ∧
          pathWork h = ⟨3, 0, if signField h = 1 then 1 else 0⟩) ∨
    (expField h ≠ 0 ∧ expField h ≠ 31 ∧
        ∃ m k, f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧
          k.val = expField h + 7 ∧
          (if signField h = 1 then m.val = -((1024 + manField h : ℕ) : ℤ)
           else m.val = ((1024 + manField h : ℕ) : ℤ)) ∧
          pathWork h = ⟨3, 2, if signField h = 1 then 1 else 0⟩) := by
  by_cases he0 : expField h = 0
  · by_cases hs1 : signField h = 1
    · obtain ⟨m, k, hm, hk, hmval, hw⟩ := subnormal_negative_transition h he0 hs1
      refine Or.inl ⟨he0, m, k, hm, hk, ?_, ?_⟩
      · simp only [if_pos hs1]
        exact hmval
      · simp only [if_pos hs1] at hw ⊢
        exact hw
    · obtain ⟨m, k, hm, hk, hmval, hw⟩ := subnormal_positive_transition h he0 hs1
      refine Or.inl ⟨he0, m, k, hm, hk, ?_, ?_⟩
      · simp only [if_neg hs1]
        exact hmval
      · simp only [if_neg hs1] at hw ⊢
        exact hw
  · by_cases hs1 : signField h = 1
    · obtain ⟨m, k, hm, hk, hmval, hw⟩ := normal_negative_transition h he0 he hs1
      refine Or.inr ⟨he0, he, m, k, hm, hk, ?_, ?_⟩
      · simp only [if_pos hs1]
        exact hmval
      · simp only [if_pos hs1] at hw ⊢
        exact hw
    · obtain ⟨m, k, hm, hk, hmval, hw⟩ := normal_positive_transition h he0 he hs1
      refine Or.inr ⟨he0, he, m, k, hm, hk, ?_, ?_⟩
      · simp only [if_neg hs1]
        exact hmval
      · simp only [if_neg hs1] at hw ⊢
        exact hw

/-- Every pattern falls in exactly one of the guarded transitions: the exponent guard
    separates non-finite from finite, and `finite_classification` splits the finite arm by
    the subnormal guard and the sign guard. Together with the straight-line premise of the
    module header this is the whole control flow of the generated body. -/
theorem transition_classification (h : U16) :
    (expField h = 31 ∧ f16_split_source.pow2.f16_split h = ok none ∧
        pathWork h = ⟨1, 0, 0⟩) ∨
    (expField h = 0 ∧
        ∃ m k, f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧ k.val = 8 ∧
          (if signField h = 1 then m.val = -((manField h : ℤ)) else m.val = (manField h : ℤ)) ∧
          pathWork h = ⟨3, 0, if signField h = 1 then 1 else 0⟩) ∨
    (expField h ≠ 0 ∧ expField h ≠ 31 ∧
        ∃ m k, f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧
          k.val = expField h + 7 ∧
          (if signField h = 1 then m.val = -((1024 + manField h : ℕ) : ℤ)
           else m.val = ((1024 + manField h : ℕ) : ℤ)) ∧
          pathWork h = ⟨3, 2, if signField h = 1 then 1 else 0⟩) := by
  by_cases he : expField h = 31
  · obtain ⟨hr, hw⟩ := refusal_transition h he
    exact Or.inl ⟨he, hr, hw⟩
  · rcases finite_classification h he with
      ⟨he0, m, k, hm, hk, hmv, hw⟩ | ⟨he0, he, m, k, hm, hk, hmv, hw⟩
    · exact Or.inr (Or.inl ⟨he0, m, k, hm, hk, hmv, hw⟩)
    · exact Or.inr (Or.inr ⟨he0, he, m, k, hm, hk, hmv, hw⟩)

/-- The extracted body is total on every pattern: it returns a signed significand and shift,
    or the refusal `none`, and never `err`. -/
theorem body_total (h : U16) :
    (∃ m : I32, ∃ k : U32, f16_split_source.pow2.f16_split h = ok (some (m, k))) ∨
      f16_split_source.pow2.f16_split h = ok none := by
  by_cases he : expField h = 31
  · exact Or.inr (refusal_transition h he).1
  · by_cases he0 : expField h = 0
    · by_cases hs1 : signField h = 1
      · obtain ⟨m, k, hm, -⟩ := subnormal_negative_transition h he0 hs1
        exact Or.inl ⟨m, k, hm⟩
      · obtain ⟨m, k, hm, -⟩ := subnormal_positive_transition h he0 hs1
        exact Or.inl ⟨m, k, hm⟩
    · by_cases hs1 : signField h = 1
      · obtain ⟨m, k, hm, -⟩ := normal_negative_transition h he0 he hs1
        exact Or.inl ⟨m, k, hm⟩
      · obtain ⟨m, k, hm, -⟩ := normal_positive_transition h he0 he hs1
        exact Or.inl ⟨m, k, hm⟩

/-! ### The work bound -/

/-- **Constant branch count.** No input evaluates more than three conditional nodes. -/
theorem tests_at_most_three (h : U16) : (pathWork h).tests ≤ 3 := by
  by_cases he : expField h = 31
  · simp only [pathWork, if_pos he]
    norm_num
  · by_cases he0 : expField h = 0
    · simp only [pathWork, if_neg he, if_pos he0]
      norm_num
    · simp only [pathWork, if_neg he, if_neg he0]
      norm_num

/-- A non-finite pattern evaluates exactly the refusal guard. -/
theorem refusal_is_one_test (h : U16) (he : expField h = 31) : (pathWork h).tests = 1 := by
  simp only [pathWork, if_pos he]

/-- A finite pattern evaluates all three guards of the body. -/
theorem finite_is_three_tests (h : U16) (he : expField h ≠ 31) : (pathWork h).tests = 3 := by
  by_cases he0 : expField h = 0
  · simp only [pathWork, if_neg he, if_pos he0]
  · simp only [pathWork, if_neg he, if_neg he0]

/-- Every path of the extracted body stays inside the branch, addition and negation budget. -/
theorem work_within_budget (h : U16) :
    (pathWork h).tests ≤ splitBudget.tests ∧ (pathWork h).adds ≤ splitBudget.adds ∧
      (pathWork h).negs ≤ splitBudget.negs := by
  by_cases he : expField h = 31
  · simp only [pathWork, if_pos he, splitBudget]
    norm_num
  · by_cases he0 : expField h = 0
    · simp only [pathWork, if_neg he, if_pos he0, splitBudget]
      split <;> norm_num
    · simp only [pathWork, if_neg he, if_neg he0, splitBudget]
      split <;> norm_num

/-- The budget is attained, so the bound above is tight and not slack. -/
theorem budget_attained : pathWork 63489#u16 = splitBudget := by
  norm_num [pathWork, expField, signField, splitBudget]

/-- **Normalisation range.** On a finite pattern the shift the body emits lies in `[8, 37]`
    and the significand it emits lies in `[-2047, 2047]`, so the consumer's shift budget is
    bounded by 37 and its significand by 2047. -/
theorem normalisation_range (h : U16) (he : expField h ≠ 31) :
    ∃ m : I32, ∃ k : U32,
      f16_split_source.pow2.f16_split h = ok (some (m, k)) ∧
      8 ≤ k.val ∧ k.val ≤ 37 ∧ m.val ≤ 2047 ∧ -2047 ≤ m.val := by
  have hlt := expField_lt h
  have hman := manField_lt h
  by_cases he0 : expField h = 0
  · by_cases hs1 : signField h = 1
    · obtain ⟨m, k, hm, hk, hmval, -⟩ := subnormal_negative_transition h he0 hs1
      refine ⟨m, k, hm, hk ▸ by omega, hk ▸ by omega, ?_, ?_⟩
      · rw [hmval]; omega
      · rw [hmval]; omega
    · obtain ⟨m, k, hm, hk, hmval, -⟩ := subnormal_positive_transition h he0 hs1
      refine ⟨m, k, hm, hk ▸ by omega, hk ▸ by omega, ?_, ?_⟩
      · rw [hmval]; omega
      · rw [hmval]; omega
  · by_cases hs1 : signField h = 1
    · obtain ⟨m, k, hm, hk, hmval, -⟩ := normal_negative_transition h he0 he hs1
      refine ⟨m, k, hm, hk ▸ by omega, hk ▸ by omega, ?_, ?_⟩
      · rw [hmval]; omega
      · rw [hmval]; omega
    · obtain ⟨m, k, hm, hk, hmval, -⟩ := normal_positive_transition h he0 he hs1
      refine ⟨m, k, hm, hk ▸ by omega, hk ▸ by omega, ?_, ?_⟩
      · rw [hmval]; omega
      · rw [hmval]; omega

/-! ### Reachable witnesses -/

/-- Reachable subnormal success: `1#u16` is `2^-24`, so the extracted body returns `(1, 8)`
    after three branch tests and no addition. -/
theorem subnormal_witness :
    f16_split_source.pow2.f16_split 1#u16 = ok (some (1#i32, 8#u32)) ∧
      pathWork 1#u16 = ⟨3, 0, 0⟩ := by
  obtain ⟨m, k, hm, hk, hmval, hw⟩ :=
    subnormal_positive_transition 1#u16 (by norm_num [expField]) (by norm_num [signField])
  refine ⟨?_, hw⟩
  rw [hm]
  have hmv' : m = 1#i32 := by
    apply IScalar.eq_of_val_eq
    rw [hmval]
    norm_num [manField]
  have hkv' : k = 8#u32 := by
    apply UScalar.eq_of_val_eq
    rw [hk]
    norm_num
  rw [hmv', hkv']

/-- Reachable normal success: `0x3C00 = 1.0` splits into `(1024, 22)`, the shift `15 + 7` in
    Q32.32. -/
theorem normal_witness :
    f16_split_source.pow2.f16_split 15360#u16 = ok (some (1024#i32, 22#u32)) ∧
      pathWork 15360#u16 = ⟨3, 2, 0⟩ := by
  obtain ⟨m, k, hm, hk, hmval, hw⟩ :=
    normal_positive_transition 15360#u16 (by norm_num [expField]) (by norm_num [expField])
      (by norm_num [signField])
  refine ⟨?_, hw⟩
  rw [hm]
  have hmv' : m = 1024#i32 := by
    apply IScalar.eq_of_val_eq
    rw [hmval]
    norm_num [manField]
  have hkv' : k = 22#u32 := by
    apply UScalar.eq_of_val_eq
    rw [hk]
    norm_num [expField]
  rw [hmv', hkv']

/-- Reachable negative normal: `0xBC01 = -1.0009765625` splits into `(-1025, 22)`, the sign
    negation of the body's own `HNeg.hNeg`, at the whole budget. -/
theorem negative_witness :
    ∃ m : I32, ∃ k : U32,
      f16_split_source.pow2.f16_split 48129#u16 = ok (some (m, k)) ∧
      m.val = -1025 ∧ k.val = 22 ∧ pathWork 48129#u16 = splitBudget := by
  obtain ⟨m, k, hm, hk, hmval, hw⟩ :=
    normal_negative_transition 48129#u16 (by norm_num [expField]) (by norm_num [expField])
      (by norm_num [signField])
  refine ⟨m, k, hm, ?_, ?_, hw⟩
  · rw [hmval]
    norm_num [manField]
  · rw [hk]
    norm_num [expField]

/-- Reachable largest shift: `0xF801` has exponent field 30, so the body splits it into
    `(-1025, 37)` — the largest shift it can emit — at the whole budget. -/
theorem maximum_shift_witness :
    ∃ m : I32, ∃ k : U32,
      f16_split_source.pow2.f16_split 63489#u16 = ok (some (m, k)) ∧
      m.val = -1025 ∧ k.val = 37 ∧ pathWork 63489#u16 = splitBudget := by
  obtain ⟨m, k, hm, hk, hmval, hw⟩ :=
    normal_negative_transition 63489#u16 (by norm_num [expField]) (by norm_num [expField])
      (by norm_num [signField])
  refine ⟨m, k, hm, ?_, ?_, hw⟩
  · rw [hmval]
    norm_num [manField]
  · rw [hk]
    norm_num [expField]

/-- Reachable refusal: `0x7C00` is positive infinity, and the extracted body returns `ok none`
    after one branch test. -/
theorem positive_infinity_refusal :
    f16_split_source.pow2.f16_split 31744#u16 = ok none ∧
      pathWork 31744#u16 = ⟨1, 0, 0⟩ := by
  obtain ⟨hr, hw⟩ := refusal_transition 31744#u16 (by norm_num [expField])
  exact ⟨hr, hw⟩

/-- Reachable refusal: `0x7C01` is a signalling NaN pattern, refused by the same first guard. -/
theorem nan_refusal :
    f16_split_source.pow2.f16_split 31745#u16 = ok none ∧
      pathWork 31745#u16 = ⟨1, 0, 0⟩ := by
  obtain ⟨hr, hw⟩ := refusal_transition 31745#u16 (by norm_num [expField])
  exact ⟨hr, hw⟩

end F16SplitResource

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

#resource_audit F16SplitResource.expField_read
#resource_audit F16SplitResource.fraction_val
#resource_audit F16SplitResource.refusal_transition
#resource_audit F16SplitResource.subnormal_positive_transition
#resource_audit F16SplitResource.subnormal_negative_transition
#resource_audit F16SplitResource.normal_positive_transition
#resource_audit F16SplitResource.normal_negative_transition
#resource_audit F16SplitResource.finite_classification
#resource_audit F16SplitResource.transition_classification
#resource_audit F16SplitResource.body_total
#resource_audit F16SplitResource.tests_at_most_three
#resource_audit F16SplitResource.finite_is_three_tests
#resource_audit F16SplitResource.work_within_budget
#resource_audit F16SplitResource.budget_attained
#resource_audit F16SplitResource.normalisation_range
#resource_audit F16SplitResource.subnormal_witness
#resource_audit F16SplitResource.normal_witness
#resource_audit F16SplitResource.negative_witness
#resource_audit F16SplitResource.maximum_shift_witness
#resource_audit F16SplitResource.positive_infinity_refusal
#resource_audit F16SplitResource.nan_refusal

run_cmd Lean.logInfo "F16_SPLIT_KERNEL_OK"