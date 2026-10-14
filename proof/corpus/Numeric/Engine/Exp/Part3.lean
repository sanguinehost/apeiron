import Numeric.Engine.Exp.Part2

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-- Shift-count side condition for the `k ≥ 0` scaling branch: on `0 ≤ k.val < 64` the extracted
`IScalar.hcast .U32 k` is `k.val`, so the left-shift amount is below 128. -/
theorem fixed_exp_scale_shift_count_lt (k : I64) (h0 : 0 ≤ k.val) (h64 : k.val < 64) :
    (IScalar.hcast UScalarTy.U32 k).val < 128 := by
  rw [IScalar.hcast_val_eq]
  norm_num [UScalarTy.numBits]
  have h : k.val % 4294967296 = k.val := Int.emod_eq_of_lt h0 (by omega)
  rw [h]
  omega

/-- `k ≥ 0` scaling branch: on `0 ≤ k.val < 64` the extracted checked left shift returns either
`Err OutOfRange` (when the narrowed result leaves the `i64` range) or `Ok v` with
`v.val = (acc6.val <<< k.val) mod 2^128`. The two `i64`-range guards put `scaled` in `[-2^63, 2^63-1]`,
where the `i128→i64` cast and `bmod 2^64` are the identity. -/
theorem fixed_exp_scale_nonneg (k : I64) (acc6 : I128) (h0 : 0 ≤ k.val) (h64 : k.val < 64) :
    (do
        let n ← lift (IScalar.hcast .U32 k)
        let scaled ← acc6 <<< n
        let i26 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if scaled > i26
        then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
        else
          let i27 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if scaled < i27
          then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let i28 ← lift (IScalar.cast .I64 scaled)
            ok (core.result.Result.Ok i28)) ⦃ z =>
      z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ∨
        ∃ v : I64, z = core.result.Result.Ok v ∧
          (v.val : ℤ) = (acc6.val <<< k.val.toNat).bmod (2^128) ⦄ := by
  have hMIN : (IScalar.cast IScalarTy.I128 core.num.I64.MIN).val = -2^63 := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    apply fixed_exp_bmod_self_64 <;> norm_num [I64.min, I64.rMin, I64.numBits]
  have hMAX : (IScalar.cast IScalarTy.I128 core.num.I64.MAX).val = 2^63 - 1 := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    apply fixed_exp_bmod_self_64 <;> norm_num [I64.max, I64.rMax, I64.numBits]
  have hn : (IScalar.hcast UScalarTy.U32 k).val = k.val.toNat := by
    rw [IScalar.hcast_val_eq]
    norm_num [UScalarTy.numBits]
    have h : k.val % 4294967296 = k.val := Int.emod_eq_of_lt h0 (by omega)
    rw [h]
  simp only [lift, bind_tc_ok]
  step*
  · exact fixed_exp_scale_shift_count_lt k h0 h64
  · refine Or.inr ⟨IScalar.cast IScalarTy.I64 scaled, rfl, ?_⟩
    rename_i hgt hlt
    have hle_max : scaled.val ≤ (IScalar.cast IScalarTy.I128 core.num.I64.MAX).val := by
      rw [gt_iff_lt] at hgt
      rw [IScalar.lt_equiv] at hgt
      exact not_lt.mp hgt
    have hge_min : (IScalar.cast IScalarTy.I128 core.num.I64.MIN).val ≤ scaled.val := by
      rw [IScalar.lt_equiv] at hlt
      exact not_lt.mp hlt
    have hslo : -(2:ℤ)^63 ≤ scaled.val := by rw [hMIN] at hge_min; exact hge_min
    have hshi : scaled.val ≤ 2^63 - 1 := by rw [hMAX] at hle_max; exact hle_max
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    rw [show (18446744073709551616 : ℕ) = 2^64 by norm_num]
    rw [fixed_exp_bmod_self_64 scaled.val hslo hshi]
    rw [show (340282366920938463463374607431768211456 : ℕ) = 2^128 by norm_num]
    rw [scaled_post]
    norm_num [I128.size, I128.numBits]
    have hmod : k.val % 4294967296 = k.val := Int.emod_eq_of_lt h0 (by omega)
    rw [hmod]

/-- `k < 0` scaling core: given `n = |k.val|` with `1 ≤ n.val < 64` and `|acc6.val| ≤ 2^40`, the
extracted round-to-nearest right shift returns either `Err OutOfRange` (when the rounded value leaves
the `i64` range) or `Ok v` with `v.val = ((acc6.val + 2^(n.val-1)) >>> n.val) mod 2^64`. The `half`
addend is `1 <<< (n-1) = 2^(n-1)`, the `>>>` is the arithmetic right shift, and the cast to `i64`
reduces mod `2^64`. -/
theorem fixed_exp_scale_neg_core (n : U64) (acc6 : I128) (hn : n.val < 64) (hn1 : 1 ≤ n.val)
    (ha : |acc6.val| ≤ 2^40) :
    (do
        if n >= 64#u64 then ok (core.result.Result.Ok 0#i64)
        else
          let i26 ← n - 1#u64
          let half ← 1#i128 <<< i26
          let i27 ← acc6 + half
          let rounded ← i27 >>> n
          let i28 ← lift (IScalar.cast .I128 core.num.I64.MAX)
          if rounded > i28
          then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let i29 ← lift (IScalar.cast .I128 core.num.I64.MIN)
            if rounded < i29
            then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
            else
              let i30 ← lift (IScalar.cast .I64 rounded)
              ok (core.result.Result.Ok i30)) ⦃ z =>
      z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ∨
        ∃ v : I64, z = core.result.Result.Ok v ∧
          (v.val : ℤ) = ((acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val).bmod (2^64) ⦄ := by
  simp only [lift, bind_tc_ok]
  step*
  · -- add lower bound
    have hhalf : (↑half : ℤ) = (2:ℤ)^(i26.val) := by
      rw [half_post, Int.shiftLeft_eq, one_mul]
      norm_num [I128.size, I128.numBits]
      have hp : (2:ℤ)^i26.val ≤ 2^62 := by
        have hb : i26.val ≤ 62 := by rw [i26_post]; omega
        exact pow_le_pow_right₀ (by norm_num : (1:ℤ) ≤ 2) hb
      have hlt : (2:ℤ)^i26.val < 2^128 := by
        calc (2:ℤ)^i26.val ≤ 2^62 := hp
          _ < 2^128 := by norm_num
      have hcond : (2:ℤ)^i26.val % 2^128 < (2^128 + 1) / 2 := by
        rw [Int.emod_eq_of_lt (by positivity) hlt]
        norm_num
        omega
      norm_num at hcond
      rw [Int.bmod_eq_emod_of_lt hcond]
      norm_num at hlt ⊢
      exact Int.emod_eq_of_lt (by positivity) hlt
    rw [hhalf]
    norm_num [I128.min, I128.numBits]
    have ha' := abs_le.mp ha
    have hp : (0:ℤ) ≤ (2:ℤ)^i26.val := by positivity
    nlinarith [ha'.1, ha'.2, hp]
  · -- add upper bound
    have hhalf : (↑half : ℤ) = (2:ℤ)^(i26.val) := by
      rw [half_post, Int.shiftLeft_eq, one_mul]
      norm_num [I128.size, I128.numBits]
      have hp : (2:ℤ)^i26.val ≤ 2^62 := by
        have hb : i26.val ≤ 62 := by rw [i26_post]; omega
        exact pow_le_pow_right₀ (by norm_num : (1:ℤ) ≤ 2) hb
      have hlt : (2:ℤ)^i26.val < 2^128 := by
        calc (2:ℤ)^i26.val ≤ 2^62 := hp
          _ < 2^128 := by norm_num
      have hcond : (2:ℤ)^i26.val % 2^128 < (2^128 + 1) / 2 := by
        rw [Int.emod_eq_of_lt (by positivity) hlt]
        norm_num
        omega
      norm_num at hcond
      rw [Int.bmod_eq_emod_of_lt hcond]
      norm_num at hlt ⊢
      exact Int.emod_eq_of_lt (by positivity) hlt
    rw [hhalf]
    norm_num [I128.max, I128.numBits]
    have ha' := abs_le.mp ha
    have hp : (2:ℤ)^i26.val ≤ 2^62 := by
      have hb : i26.val ≤ 62 := by rw [i26_post]; omega
      exact pow_le_pow_right₀ (by norm_num : (1:ℤ) ≤ 2) hb
    nlinarith [ha'.1, ha'.2, hp]
  · -- value
    have hhalf : (↑half : ℤ) = (2:ℤ)^(i26.val) := by
      rw [half_post, Int.shiftLeft_eq, one_mul]
      norm_num [I128.size, I128.numBits]
      have hp : (2:ℤ)^i26.val ≤ 2^62 := by
        have hb : i26.val ≤ 62 := by rw [i26_post]; omega
        exact pow_le_pow_right₀ (by norm_num : (1:ℤ) ≤ 2) hb
      have hlt : (2:ℤ)^i26.val < 2^128 := by
        calc (2:ℤ)^i26.val ≤ 2^62 := hp
          _ < 2^128 := by norm_num
      have hcond : (2:ℤ)^i26.val % 2^128 < (2^128 + 1) / 2 := by
        rw [Int.emod_eq_of_lt (by positivity) hlt]
        norm_num
        omega
      norm_num at hcond
      rw [Int.bmod_eq_emod_of_lt hcond]
      norm_num at hlt ⊢
      exact Int.emod_eq_of_lt (by positivity) hlt
    refine Or.inr ⟨IScalar.cast IScalarTy.I64 rounded, rfl, ?_⟩
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    rw [rounded_post, i27_post, hhalf, i26_post]

/-- Flush sub-case of the `k < 0` scaling branch: for `64 ≤ n.val` the extracted core returns
`Ok 0` (the extracted body flushes below half a unit). -/
theorem fixed_exp_scale_flush (n : U64) (acc6 : I128) (h : 64 ≤ n.val) :
    (do
        if n >= 64#u64 then ok (core.result.Result.Ok 0#i64)
        else
          let i26 ← n - 1#u64
          let half ← 1#i128 <<< i26
          let i27 ← acc6 + half
          let rounded ← i27 >>> n
          let i28 ← lift (IScalar.cast .I128 core.num.I64.MAX)
          if rounded > i28
          then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let i29 ← lift (IScalar.cast .I128 core.num.I64.MIN)
            if rounded < i29
            then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
            else
              let i30 ← lift (IScalar.cast .I64 rounded)
              ok (core.result.Result.Ok i30)) = ok (core.result.Result.Ok 0#i64) := by
  have hge : (64#u64 : U64) ≤ n := (UScalar.le_equiv _ _).2 (by simpa using h)
  rw [if_pos hge]

theorem fixed_exp_scale_neg (n : U64) (acc6 : I128) (hn1 : 1 ≤ n.val) (ha : |acc6.val| ≤ 2^40) :
    (do
        if n >= 64#u64 then ok (core.result.Result.Ok 0#i64)
        else
          let i26 ← n - 1#u64
          let half ← 1#i128 <<< i26
          let i27 ← acc6 + half
          let rounded ← i27 >>> n
          let i28 ← lift (IScalar.cast .I128 core.num.I64.MAX)
          if rounded > i28
          then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let i29 ← lift (IScalar.cast .I128 core.num.I64.MIN)
            if rounded < i29
            then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
            else
              let i30 ← lift (IScalar.cast .I64 rounded)
              ok (core.result.Result.Ok i30)) ⦃ z =>
      (64 ≤ n.val → z = core.result.Result.Ok 0#i64) ∧
      (n.val < 64 → z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ∨
        ∃ v : I64, z = core.result.Result.Ok v ∧
          (v.val : ℤ) = ((acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val).bmod (2^64)) ⦄ := by
  by_cases h64 : 64 ≤ n.val
  · rw [fixed_exp_scale_flush n acc6 h64]
    refine ⟨?_, ?_⟩
    · intro _; rfl
    · intro hlt; exfalso; omega
  · have hlt : n.val < 64 := by omega
    exact Aeneas.Std.WP.spec_mono (fixed_exp_scale_neg_core n acc6 hlt hn1 ha)
      (fun z hz => ⟨fun hc => absurd hc (by omega), fun _ => hz⟩)

/-- `n`-extraction for the `k < 0` scaling branch: on `-2^62 ≤ k.val < 0` the extracted
`if k < 0 then (do i26 ← -.k; ok (hcast .U64 i26)) else …` yields `n.val = k.val.natAbs`. -/
theorem fixed_exp_scale_abs (k : I64) (hk : k.val < 0) (hlo : -2^62 ≤ k.val) :
    (do let n ← (if k < 0#i64 then (do let i26 ← -.k; ok (IScalar.hcast .U64 i26))
                     else ok (IScalar.hcast .U64 k)); ok n)
      ⦃ n => n.val = k.val.natAbs ⦄ := by
  have hlt : k < 0#i64 := (IScalar.lt_equiv _ _).2 (by simpa using hk)
  simp only [if_pos hlt]
  step*
  rw [IScalar.hcast_val_eq, i26_post]
  norm_num [UScalarTy.numBits]
  have h0 : 0 ≤ -k.val := by omega
  have hlt2 : -k.val < 2^64 := by omega
  norm_num at hlt2
  rw [Int.emod_eq_of_lt h0 hlt2]
  omega

theorem fixed_exp_scale_nonneg_ge64 (k : I64) (acc6 : I128) (hk : 64 ≤ k.val) :
    (do
        if k >= 64#i64
        then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
        else
          let n ← lift (IScalar.hcast .U32 k)
          let scaled ← acc6 <<< n
          let i26 ← lift (IScalar.cast .I128 core.num.I64.MAX)
          if scaled > i26
          then
            ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let i27 ← lift (IScalar.cast .I128 core.num.I64.MIN)
            if scaled < i27
            then
              ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
            else
              let i28 ← lift (IScalar.cast .I64 scaled)
              ok (core.result.Result.Ok i28)) = ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange) := by
  have h64 : (64#i64 : I64) ≤ k := (IScalar.le_equiv _ _).2 (by simpa using hk)
  rw [if_pos h64]

/-- Raw-`if` form of `fixed_exp_scale_abs` (the `n`-extraction spec `spec_bind` needs): on
`-2^62 ≤ k.val < 0` the extracted `if k < 0 then (do i26 ← -.k; ok (hcast .U64 i26)) else …`
yields `n.val = k.val.natAbs`. -/
theorem fixed_exp_scale_abs_raw (k : I64) (hk : k.val < 0) (hlo : -2^62 ≤ k.val) :
    (if k < 0#i64 then (do let i26 ← -.k; ok (IScalar.hcast .U64 i26))
      else ok (IScalar.hcast .U64 k)) ⦃ n => n.val = k.val.natAbs ⦄ := by
  have hlt : k < 0#i64 := (IScalar.lt_equiv _ _).2 (by simpa using hk)
  rw [if_pos hlt]
  step*
  rw [IScalar.hcast_val_eq, i26_post]
  norm_num [UScalarTy.numBits]
  have h0 : 0 ≤ -k.val := by omega
  have hlt2 : -k.val < 2^64 := by omega
  norm_num at hlt2
  rw [Int.emod_eq_of_lt h0 hlt2]
  omega

/-- The full `fixed_exp` scaling block on `-2^62 ≤ k.val` and `|acc6.val| ≤ 2^40`: the `k ≥ 0`
checked left shift and the `k < 0` round-to-nearest right shift (flush to `0` above 64), with the
`i64`-range refusal. Post gives the exact `i64` output value as a function of `k.val`/`acc6.val`, or
`Err`. -/
theorem word_fixed_exp_scale (k : I64) (acc6 : I128) (hlo : -2^62 ≤ k.val) (hak : |acc6.val| ≤ 2^40) :
    (do
                if k >= 0#i64
                then
                  if k >= 64#i64
                  then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
                  else
                    let n ← lift (IScalar.hcast .U32 k)
                    let scaled ← acc6 <<< n
                    let i26 ← lift (IScalar.cast .I128 core.num.I64.MAX)
                    if scaled > i26
                    then
                      ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
                    else
                      let i27 ← lift (IScalar.cast .I128 core.num.I64.MIN)
                      if scaled < i27
                      then
                        ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
                      else
                        let i28 ← lift (IScalar.cast .I64 scaled)
                        ok (core.result.Result.Ok i28)
                else
                  let n ←
                    if k < 0#i64
                    then do
                         let i26 ← -. k
                         ok (IScalar.hcast .U64 i26)
                    else ok (IScalar.hcast .U64 k)
                  if n >= 64#u64
                  then ok (core.result.Result.Ok 0#i64)
                  else
                    let i26 ← n - 1#u64
                    let half ← 1#i128 <<< i26
                    let i27 ← acc6 + half
                    let rounded ← i27 >>> n
                    let i28 ← lift (IScalar.cast .I128 core.num.I64.MAX)
                    if rounded > i28
                    then
                      ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
                    else
                      let i29 ← lift (IScalar.cast .I128 core.num.I64.MIN)
                      if rounded < i29
                      then
                        ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
                      else
                        let i30 ← lift (IScalar.cast .I64 rounded)
                        ok (core.result.Result.Ok i30)
    ) ⦃ z =>
      (0 ≤ k.val → z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ∨
        ∃ v : I64, z = core.result.Result.Ok v ∧
          (v.val : ℤ) = (acc6.val <<< k.val.toNat).bmod (2^128)) ∧
      (k.val < 0 → z = core.result.Result.Ok 0#i64 ∨
        z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ∨
        ∃ v : I64, z = core.result.Result.Ok v ∧
          (v.val : ℤ) = ((acc6.val + (2:ℤ)^(k.val.natAbs - 1)) >>> k.val.natAbs).bmod (2^64)) ⦄ := by
  by_cases hk0 : 0 ≤ k.val
  · have hge : (0#i64 : I64) ≤ k := (IScalar.le_equiv _ _).2 (by simpa using hk0)
    rw [if_pos hge]
    by_cases hk64 : k.val < 64
    · have hge64 : ¬ ((64#i64 : I64) ≤ k) := by
        intro h
        have h' : (64:ℤ) ≤ k.val := by simpa using (IScalar.le_equiv _ _).1 h
        omega
      rw [if_neg hge64]
      refine Aeneas.Std.WP.spec_mono (fixed_exp_scale_nonneg k acc6 hk0 hk64) ?_
      intro z hz
      exact ⟨fun _ => hz, fun hlt => by exfalso; omega⟩
    · have hge64 : (64#i64 : I64) ≤ k := (IScalar.le_equiv _ _).2 (by simpa using (by omega : 64 ≤ k.val))
      rw [if_pos hge64]
      refine ⟨?_, ?_⟩
      · intro _; exact Or.inl rfl
      · intro hlt; exfalso; omega
  · have hnf : ¬ ((0#i64 : I64) ≤ k) := by
      intro h
      have h' : (0:ℤ) ≤ k.val := by simpa using (IScalar.le_equiv _ _).1 h
      omega
    rw [if_neg hnf]
    have hkn : k.val < 0 := lt_of_not_ge hk0
    refine Aeneas.Std.WP.spec_bind (fixed_exp_scale_abs_raw k hkn hlo) ?_
    intro n hn
    refine Aeneas.Std.WP.spec_mono (fixed_exp_scale_neg n acc6 ?hn1 hak) ?_
    · rw [hn]; omega
    · intro z hz
      refine ⟨fun h0 => absurd h0 hk0, ?_⟩
      intro _
      by_cases h64 : 64 ≤ n.val
      · exact Or.inl (hz.1 h64)
      · rcases hz.2 (by omega) with h | ⟨v, hv, hexpr⟩
        · exact Or.inr (Or.inl h)
        · refine Or.inr (Or.inr ⟨v, hv, ?_⟩)
          rw [← hn]
          exact hexpr

/-- Pair-returning reduction chunk (W1-T1W): the reduction operations verbatim with the success value
`(k, r)` (inner `Result` kept), so the scaling continuation can be composed with both `k` and `r` in
scope. Post: `k.val = (x.val·6196328019 + 2^63) >>> 64`, `r.val = x.val − k.val·2977044472`. -/
theorem word_fixed_exp_reduction_value_pair (x : I64) (hx : window x) :
    (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let i3 ←
          lift (core.convert.num.FromI128I64.from fixed_exp.LN2_RECIP_Q32)
        let product ← i2 * i3
        let i4 ← 1#i128 <<< 63#i32
        let i5 ← product + i4
        let shifted ← i5 >>> 64#i32
        let i6 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if shifted > i6
        then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
        else
          let i7 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if shifted < i7
          then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let k ← lift (IScalar.cast .I64 shifted)
            let i8 ← lift (core.convert.num.FromI128I64.from k)
            let i9 ← lift (core.convert.num.FromI128I64.from fixed_exp.LN2_Q32)
            let k_ln2_wide ← i8 * i9
            let i10 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if k_ln2_wide > i10
            then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
            else
              let i11 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if k_ln2_wide < i11
              then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
              else
                let k_ln2 ← lift (IScalar.cast .I64 k_ln2_wide)
                let r ← x - k_ln2
                ok (core.result.Result.Ok (k, r))) ⦃ z =>
      ∃ k r : I64, z = core.result.Result.Ok (k, r) ∧
        k.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 ∧
        r.val = x.val - k.val * 2977044472 ⦄ := by
  simp only [lift, bind_tc_ok]
  step*
  all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq, fixed_exp.LN2_RECIP_Q32,
    fixed_exp.LN2_Q32, I128.min, I128.max, I64.min, I64.max] at *))
  all_goals (try norm_num at *)
  all_goals (try scalar_tac)
  all_goals (try (simp only [Int.shiftRight_eq_div_pow] at *))
  · -- product + i4
    have hi4 : (↑i4 : ℤ) = 2 ^ 63 := by
      rw [i4_post]; norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
    rw [product_post, hi4]
    norm_num [I128.numBits]
    nlinarith [hx.1, hx.2]
  · -- guard: shifted <= I64.max
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_exp_shift_recip_shift_bound x.val shifted.val hx.1 hx.2 hsv
    have hle : (shifted : ℤ) ≤ 30 := hsb.2
    norm_num [I64.rMax] at *
    exfalso; omega
  · -- guard: shifted >= I64.min
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_exp_shift_recip_shift_bound x.val shifted.val hx.1 hx.2 hsv
    have hge : -24 ≤ (shifted : ℤ) := hsb.1
    norm_num [I64.rMin] at *
    exfalso; omega
  · -- shifted.bmod lower
    have h := fixed_exp_bmod_64_lower (shifted : ℤ)
    norm_num [I128.numBits] at h ⊢
    nlinarith [h]
  · -- shifted.bmod upper
    have h := fixed_exp_bmod_64_upper (shifted : ℤ)
    norm_num [I128.numBits] at h ⊢
    nlinarith [h]
  · -- guard: k_ln2_wide <= I64.max
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    have hlink : (↑k_ln2_wide : ℤ) = shifted.val * 2977044472 := hkval
    have hle : (↑k_ln2_wide : ℤ) ≤ 2 ^ 63 - 1 := by rw [hlink]; exact hsr.2
    norm_num [I64.rMax] at *
    exfalso; omega
  · -- guard: k_ln2_wide >= I64.min
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    have hlink : (↑k_ln2_wide : ℤ) = shifted.val * 2977044472 := hkval
    have hge : -2 ^ 63 ≤ (↑k_ln2_wide : ℤ) := by rw [hlink]; exact hsr.1
    norm_num [I64.rMin] at *
    exfalso; omega
  · -- k_ln2_wide.bmod lower
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    rw [show (18446744073709551616 : ℕ) = 2 ^ 64 by norm_num]
    rw [hb, hkval]
    norm_num [I64.numBits]
    have hr := (fixed_exp_reduction_remainder_range x.val shifted.val hsv hsr).1
    omega
  · -- k_ln2_wide.bmod upper
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    rw [show (18446744073709551616 : ℕ) = 2 ^ 64 by norm_num]
    rw [hb, hkval]
    norm_num [I64.numBits]
    have hr := (fixed_exp_reduction_remainder_range x.val shifted.val hsv hsr).2
    omega
  · -- post: value characterization with k = cast .I64 shifted
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    have hcast : (↑(IScalar.cast IScalarTy.I64 shifted) : ℤ) = shifted.val := by
      rw [IScalar.cast_val_eq]
      norm_num
      exact fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)
    refine ⟨?_, ?_⟩
    · have hbmod : ((shifted : ℤ)).bmod (18446744073709551616 : ℕ) = (shifted : ℤ) := by
        rw [← hcast]
        rw [IScalar.cast_val_eq]
        norm_num
      rw [hbmod]
      rw [Int.shiftRight_eq_div_pow] at hsv
      norm_num at hsv
      exact hsv
    · norm_num at hb
      rw [r_post, hb, hkval]
      have hbmod : ((shifted : ℤ)).bmod (18446744073709551616 : ℕ) = (shifted : ℤ) := by
        rw [← hcast]
        rw [IScalar.cast_val_eq]
        norm_num
      rw [hbmod]

/-- Success-path-only reduction pair chunk (W1-T1AC): the reduction operations without the four inner
`i64`-range guard `if`s, returning the pair `(k, r)` at the outer level (`Result (I64 × I64)`), with
the same `k`/`r` characterization. This is the head-composable form: its value carries `k` so the
scaling continuation can be composed with `spec_bind`. -/
theorem word_fixed_exp_reduction_pair_succ (x : I64) (hx : window x) :
    (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let i3 ← lift (core.convert.num.FromI128I64.from fixed_exp.LN2_RECIP_Q32)
        let product ← i2 * i3
        let i4 ← 1#i128 <<< 63#i32
        let i5 ← product + i4
        let shifted ← i5 >>> 64#i32
        let k ← lift (IScalar.cast .I64 shifted)
        let i8 ← lift (core.convert.num.FromI128I64.from k)
        let i9 ← lift (core.convert.num.FromI128I64.from fixed_exp.LN2_Q32)
        let k_ln2_wide ← i8 * i9
        let k_ln2 ← lift (IScalar.cast .I64 k_ln2_wide)
        let r ← x - k_ln2
        ok (k, r)) ⦃ z =>
      z.1.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 ∧
        z.2.val = x.val - z.1.val * 2977044472 ⦄ := by
  simp only [lift, bind_tc_ok]
  step*
  all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq, fixed_exp.LN2_RECIP_Q32,
    fixed_exp.LN2_Q32] at *))
  all_goals (try (simp only [Int.shiftRight_eq_div_pow] at *))
  · -- product lower
    norm_num [I128.min, I128.numBits]
    nlinarith [hx.1]
  · -- product upper
    norm_num [I128.max, I128.numBits]
    nlinarith [hx.2]
  · -- product + i4
    have hi4 : (↑i4 : ℤ) = 2 ^ 63 := by
      rw [i4_post]; norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
    rw [product_post, hi4]
    norm_num [I128.max, I128.numBits]
    nlinarith [hx.2]
  · -- k_ln2_wide lower
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hcast : (↑(IScalar.cast IScalarTy.I64 shifted) : ℤ) = shifted.val := by
      rw [IScalar.cast_val_eq]; norm_num
      exact fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)
    rw [hcast]
    norm_num [I128.min, I128.numBits]
    omega
  · -- k_ln2_wide upper
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hcast : (↑(IScalar.cast IScalarTy.I64 shifted) : ℤ) = shifted.val := by
      rw [IScalar.cast_val_eq]; norm_num
      exact fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)
    rw [hcast]
    norm_num [I128.max, I128.numBits]
    omega
  · -- r lower
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hcast2 : (↑(IScalar.cast IScalarTy.I64 k_ln2_wide) : ℤ) = k_ln2_wide.val := by
      rw [IScalar.cast_val_eq]; norm_num
      exact fixed_exp_bmod_self_64 k_ln2_wide.val (by omega) (by omega)
    rw [hcast2, hkval]
    norm_num [I64.min, I64.numBits]
    have hr := (fixed_exp_reduction_remainder_range x.val shifted.val hsv hsr).1
    omega
  · -- r upper
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hcast2 : (↑(IScalar.cast IScalarTy.I64 k_ln2_wide) : ℤ) = k_ln2_wide.val := by
      rw [IScalar.cast_val_eq]; norm_num
      exact fixed_exp_bmod_self_64 k_ln2_wide.val (by omega) (by omega)
    rw [hcast2, hkval]
    norm_num [I64.max, I64.numBits]
    have hr := (fixed_exp_reduction_remainder_range x.val shifted.val hsv hsr).2
    omega
  · -- post
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    have hcast : (↑(IScalar.cast IScalarTy.I64 shifted) : ℤ) = shifted.val := by
      rw [IScalar.cast_val_eq]; norm_num
      exact fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)
    refine ⟨?_, ?_⟩
    · rw [hcast]
      rw [Int.shiftRight_eq_div_pow] at hsv
      norm_num at hsv ⊢
      exact hsv
    · have hcast2 : (↑(IScalar.cast IScalarTy.I64 k_ln2_wide) : ℤ) = k_ln2_wide.val := by
        rw [IScalar.cast_val_eq]; norm_num
        exact fixed_exp_bmod_self_64 k_ln2_wide.val (by omega) (by omega)
      rw [r_post, hcast2, hkval, hcast]
/-! ## Re-anchored function specs (regenerated chunked extraction) -/

/-- On the window the evaluated lower endpoint comparison is false. -/
theorem fixed_exp_lower_test_false (x : I64) (hx : window x) :
    ¬ x < ((-68719476736)#i64) := by
  intro h
  rw [IScalar.lt_equiv] at h
  exact not_lt.mpr hx.1 h

/-- On the window the evaluated upper endpoint comparison is false. -/
theorem fixed_exp_upper_test_false (x : I64) (hx : window x) :
    ¬ x > (90194313216#i64) := by
  intro h
  rw [gt_iff_lt] at h
  rw [IScalar.lt_equiv] at h
  exact not_lt.mpr hx.2 h

/-- `exp_reduce` spec: the value `(k, r)` characterization or the window refusal. -/
theorem fixed_exp_exp_reduce_spec (x : I64) (hx : window x) :
    fixed_exp.exp_reduce x ⦃ z =>
      (∃ k r : I64, z = core.result.Result.Ok (k, r) ∧
        k.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 ∧
        r.val = x.val - k.val * 2977044472) ∨
      z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ⦄ := by
  unfold fixed_exp.exp_reduce
  rw [fixed_exp_LOWER_eq, fixed_exp_UPPER_eq]
  simp only [bind_tc_ok]
  rw [if_neg (fixed_exp_lower_test_false x hx), if_neg (fixed_exp_upper_test_false x hx)]
  apply Aeneas.Std.WP.spec_mono (word_fixed_exp_reduction_value_pair x hx)
  intro z hz
  exact Or.inl hz

/-! ### Pending: `exp_horner` spec

`fixed_exp_exp_horner_spec` (`exp_horner r ⦃ z => z.val = fixed_exp.horner_acc r.val ⦄`) is not yet
stated here: the body of `exp_horner` is the same Horner chain as the retained inline
`word_fixed_exp_horner_acc`, and restating it by `unfold …; exact …` doubles the 8M-heartbeat defeq
cost (the module build exceeded 540 s). The next slice must either prove the spec directly against
`exp_horner`'s body (adapting the `word_fixed_exp_horner_acc` proof) and retire the inline copy, or
expose a bound on `horner_acc` so the composition can proceed. -/

/-- `exp_scale` spec: the scaled output value or the `i64` refusal. -/
theorem fixed_exp_exp_scale_spec (k : I64) (acc6 : I128)
    (hlo : -2^62 ≤ k.val) (hak : |acc6.val| ≤ 2^40) :
    fixed_exp.exp_scale k acc6 ⦃ z =>
      (0 ≤ k.val → z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ∨
        ∃ v : I64, z = core.result.Result.Ok v ∧
          (v.val : ℤ) = (acc6.val <<< k.val.toNat).bmod (2^128)) ∧
      (k.val < 0 → z = core.result.Result.Ok 0#i64 ∨
        z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ∨
        ∃ v : I64, z = core.result.Result.Ok v ∧
          (v.val : ℤ) = ((acc6.val + (2:ℤ)^(k.val.natAbs - 1)) >>> k.val.natAbs).bmod (2^64)) ⦄ := by
  unfold fixed_exp.exp_scale
  exact word_fixed_exp_scale k acc6 hlo hak


/-- Tight remainder bound: the reduction remainder `x - k·LN2_Q32` is within `2^32` in absolute value
(the `i64`-range statement is `fixed_exp_reduction_remainder_range`). -/
theorem fixed_exp_reduction_remainder_abs_le (x s : ℤ)
    (hshift : s = (x * 6196328019 + 2 ^ 63) >>> 64)
    (hk : -2 ^ 63 ≤ s * 2977044472 ∧ s * 2977044472 ≤ 2 ^ 63 - 1) :
    |x - s * 2977044472| ≤ 2 ^ 32 := by
  have hshift' : (x * 6196328019 + 2 ^ 63) >>> 64 = s := hshift.symm
  have hdiv : (x * 6196328019 + 2 ^ 63) / 2 ^ 64 = s := by
    rw [Int.shiftRight_eq_div_pow] at hshift'
    norm_num at hshift'
    exact hshift'
  have hdecomp : 2 ^ 64 * s + (x * 6196328019 + 2 ^ 63) % 2 ^ 64
      = x * 6196328019 + 2 ^ 63 := by
    have := Int.mul_ediv_add_emod (x * 6196328019 + 2 ^ 63) (2 ^ 64)
    rw [hdiv] at this
    exact this
  have ht0 : 0 ≤ (x * 6196328019 + 2 ^ 63) % 2 ^ 64 := Int.emod_nonneg _ (by norm_num)
  have htlt : (x * 6196328019 + 2 ^ 63) % 2 ^ 64 < 2 ^ 64 := Int.emod_lt_of_pos _ (by norm_num)
  have hmR : (2977044472 : ℤ) * 6196328019 = 2 ^ 64 + 1953109352 := by norm_num
  have hDle : (1953109352 : ℤ) ≤ 2977044472 := by norm_num
  have hDnn : (0 : ℤ) ≤ 1953109352 := by norm_num
  have hsD_ub : s * 1953109352 ≤ 2 ^ 63 := by nlinarith [hk.1, hk.2, hDle, hDnn]
  have hsD_lb : -2 ^ 63 ≤ s * 1953109352 := by nlinarith [hk.1, hk.2, hDle, hDnn]
  have hrR : (x - s * 2977044472) * 6196328019
      = (x * 6196328019 + 2 ^ 63) % 2 ^ 64 - 2 ^ 63 - s * 1953109352 := by
    nlinarith [hdecomp, hmR]
  have hlow : -2 ^ 64 ≤ (x - s * 2977044472) * 6196328019 := by
    rw [hrR]; nlinarith [ht0, htlt, hsD_ub, hsD_lb]
  have hhigh : (x - s * 2977044472) * 6196328019 < 2 ^ 64 := by
    rw [hrR]; nlinarith [ht0, htlt, hsD_ub, hsD_lb]
  have hub : x - s * 2977044472 ≤ 2 ^ 32 - 1 := by
    by_contra hc
    have hrge : (2 : ℤ) ^ 32 ≤ x - s * 2977044472 := by omega
    have hmul : (2 : ℤ) ^ 32 * 6196328019 ≤ (x - s * 2977044472) * 6196328019 :=
      mul_le_mul_of_nonneg_right hrge (by norm_num)
    nlinarith [hhigh, hmul]
  have hlb : -(2 : ℤ) ^ 32 ≤ x - s * 2977044472 := by
    by_contra hc
    have hrle : x - s * 2977044472 ≤ -(2 : ℤ) ^ 32 := by omega
    have hmul : (x - s * 2977044472) * 6196328019 ≤ (-(2 : ℤ) ^ 32) * 6196328019 :=
      mul_le_mul_of_nonneg_right hrle (by norm_num)
    nlinarith [hlow, hmul]
  exact abs_le.mpr ⟨hlb, by omega⟩

/-- Sharpened reduction remainder: `|x - s·2977044472| ≤ 2^31` (halving the landed `2^32`), from
`k ∈ [-24, 30]` and `R·D = 2^64 + 1953109352`. -/
theorem fixed_exp_reduction_remainder_abs_le_2p31 (x s : ℤ)
    (hshift : s = (x * 6196328019 + 2 ^ 63) >>> 64)
    (hlo : -68719476736 ≤ x) (hhi : x ≤ 90194313216) :
    |x - s * 2977044472| ≤ 2 ^ 31 := by
  have hshift' : (x * 6196328019 + 2 ^ 63) >>> 64 = s := hshift.symm
  have hdiv : (x * 6196328019 + 2 ^ 63) / 2 ^ 64 = s := by
    rw [Int.shiftRight_eq_div_pow] at hshift'
    norm_num at hshift'
    exact hshift'
  have hdecomp : 2 ^ 64 * s + (x * 6196328019 + 2 ^ 63) % 2 ^ 64
      = x * 6196328019 + 2 ^ 63 := by
    have := Int.mul_ediv_add_emod (x * 6196328019 + 2 ^ 63) (2 ^ 64)
    rw [hdiv] at this
    exact this
  have ht0 : 0 ≤ (x * 6196328019 + 2 ^ 63) % 2 ^ 64 := Int.emod_nonneg _ (by norm_num)
  have htlt : (x * 6196328019 + 2 ^ 63) % 2 ^ 64 < 2 ^ 64 := Int.emod_lt_of_pos _ (by norm_num)
  have hmR : (2977044472 : ℤ) * 6196328019 = 2 ^ 64 + 1953109352 := by norm_num
  have hsb := fixed_exp_shift_recip_shift_bound x s hlo hhi hshift
  have hrR : (x - s * 2977044472) * 6196328019
      = (x * 6196328019 + 2 ^ 63) % 2 ^ 64 - 2 ^ 63 - s * 1953109352 := by
    nlinarith [hdecomp, hmR]
  have hlow : -(2 ^ 63) - 30 * 1953109352 ≤ (x - s * 2977044472) * 6196328019 := by
    rw [hrR]; nlinarith [ht0, hsb.2]
  have hhigh : (x - s * 2977044472) * 6196328019 < 2 ^ 63 + 30 * 1953109352 := by
    rw [hrR]; nlinarith [htlt, hsb.2]
  have hub : x - s * 2977044472 ≤ 2 ^ 31 := by
    by_contra hc
    have hrge : (2 : ℤ) ^ 31 < x - s * 2977044472 := by omega
    have hmul : (2 : ℤ) ^ 31 * 6196328019 < (x - s * 2977044472) * 6196328019 :=
      mul_lt_mul_of_pos_right hrge (by norm_num)
    nlinarith [hhigh, hmul]
  have hlb : -(2 : ℤ) ^ 31 ≤ x - s * 2977044472 := by
    by_contra hc
    have hrle : x - s * 2977044472 < -(2 : ℤ) ^ 31 := by omega
    have hmul : (x - s * 2977044472) * 6196328019 < (-(2 : ℤ) ^ 31) * 6196328019 :=
      mul_lt_mul_of_pos_right hrle (by norm_num)
    nlinarith [hlow, hmul]
  exact abs_le.mpr ⟨hlb, hub⟩



/-- Real form of the sharpened remainder: on the window `|toReal r| ≤ 1/2`, the hypothesis
`horner_acc_error` needs. -/
theorem fixed_exp_reduction_remainder_toReal_half (x k r : I64) (hx : window x)
    (hk : k.val = (x.val * 6196328019 + 2 ^ 63) >>> 64)
    (hr : r.val = x.val - k.val * 2977044472) :
    |toReal r| ≤ 1/2 := by
  have h31 : |r.val| ≤ 2 ^ 31 := by
    rw [hr]
    exact fixed_exp_reduction_remainder_abs_le_2p31 x.val k.val hk hx.1 hx.2
  unfold toReal
  rw [abs_div, abs_of_pos q32_pos, ← Int.cast_abs]
  have hle : ((|r.val| : ℤ) : ℝ) ≤ ((2 ^ 31 : ℤ) : ℝ) := by exact_mod_cast h31
  calc ((|r.val| : ℤ) : ℝ) / q32 ≤ ((2 ^ 31 : ℤ) : ℝ) / q32 :=
        div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
    _ = 1/2 := by unfold q32; norm_num

/-- Whole-body `fixed_exp` value (W1-T1AJ): the extracted call chain `fixed_exp.fixed_exp` returns
`Err OutOfRange` or `Ok v` with the exact integer value, using the re-anchored `exp_reduce` /
`exp_horner` / `exp_scale` specs. -/
theorem word_fixed_exp_body (x : I64) (hx : window x) :
    fixed_exp.fixed_exp x ⦃ z =>
      z = core.result.Result.Err fixed_exp.ExpError.OutOfRange ∨
      ∃ k r : I64, ∃ acc : I128, ∃ v : I64,
        z = core.result.Result.Ok v ∧
        k.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 ∧
        r.val = x.val - k.val * 2977044472 ∧
        acc.val = fixed_exp.horner_acc r.val ∧
        (0 ≤ k.val → (v.val : ℤ) = (acc.val <<< k.val.toNat).bmod (2^128)) ∧
        (k.val < 0 → v.val = 0#i64 ∨
          (v.val : ℤ) = ((acc.val + (2:ℤ)^(k.val.natAbs - 1)) >>> k.val.natAbs).bmod (2^64)) ⦄ := by
  unfold fixed_exp.fixed_exp
  apply Aeneas.Std.WP.spec_bind (fixed_exp_exp_reduce_spec x hx)
  intro kv hkv
  rcases hkv with ⟨k, r, rfl, hk, hr⟩ | hErr
  · dsimp only
    have hsr := fixed_exp_shift_recip_bound_window x.val k.val hx.1 hx.2 hk
    have hr32 : |r.val| ≤ 2^32 := by
      rw [hr]; exact fixed_exp_reduction_remainder_abs_le x.val k.val hk hsr
    apply Aeneas.Std.WP.spec_bind (word_fixed_exp_horner_acc r hr32)
    intro acc hacc
    have haccv : acc.val = fixed_exp.horner_acc r.val := hacc.1
    have haccb : |acc.val| ≤ 2^40 := hacc.2
    have hklo : -2^62 ≤ k.val := by
      have hsb := fixed_exp_shift_recip_shift_bound x.val k.val hx.1 hx.2 hk
      omega
    apply Aeneas.Std.WP.spec_mono (fixed_exp_exp_scale_spec k acc hklo haccb)
    intro z hz
    by_cases hk0 : 0 ≤ k.val
    · rcases (hz.1 hk0) with hE | ⟨v, hv, hval⟩
      · exact Or.inl hE
      · exact Or.inr ⟨k, r, acc, v, hv, hk, hr, haccv,
          (fun _ => hval), (fun hlt => absurd hk0 (not_le.mpr hlt))⟩
    · rcases (hz.2 (lt_of_not_ge hk0)) with h0 | hE | ⟨v, hv, hval⟩
      · exact Or.inr ⟨k, r, acc, 0#i64, h0, hk, hr, haccv,
          (fun hge => absurd hge (not_le.mpr (lt_of_not_ge hk0))),
          (fun _ => Or.inl rfl)⟩
      · exact Or.inl hE
      · exact Or.inr ⟨k, r, acc, v, hv, hk, hr, haccv,
          (fun hge => absurd hge (not_le.mpr (lt_of_not_ge hk0))),
          (fun _ => Or.inr hval)⟩
  · rw [hErr]
    rw [Aeneas.Std.WP.spec_ok]
    exact Or.inl rfl

end

end apeiron_word
