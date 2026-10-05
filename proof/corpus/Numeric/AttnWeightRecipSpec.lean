/-
Reciprocal-arm specification of the extracted single-token attention weight stage
(`APEIRON-T2F2E`).

`Numeric.EngineAttnRefinement` proves `weight_at_spec` for the `recip = none` fallback of the
weight stage but leaves the `some r` arm unstated (`Numeric/EngineAttnRefinement.lean`, residual
item 4): `attn.weight_at_recip` (`src/engine/attn.rs:279-315`) rounds each numerator
`n = e·2^32 + total/2` by the classical round-up `u64` reciprocal `r = 2^64/total + 1`
(`src/engine/attn.rs:252-258`), one `u128` multiply-shift and a single subtract fixup. This module
supplies that missing specification, stated against the generated bodies in `proof/ApeironWord.lean`
(`apeiron_word.attn.weight_at_recip`, `apeiron_word.attn.recip_u64`).

Landed:

* `recip_fixup_R`, `recip_roundup_fixup`: the pure arithmetic of the round-up reciprocal — for
  `r = 2^64/T + 1` and numerator `n < 2^64`, the fixup `(n·r/2^64)·T > n ? n·r/2^64 - 1 : n·r/2^64`
  is exactly the half-biased quotient `n/T`. Grade `proved` (kernel-checked, no extraction);
* `weight_at_recip_fast`: the fast path, `n = e·2^32 + T/2 ≤ 2^64 - 1`, gives
  `Ok w` with `w.val = divRoundNearest (e·2^32) total.val`;
* `weight_at_recip_spec`: the full statement on the whole admitted numerator window
  (`e·2^32 + total/2 ≤ I128.max`, matching `AttnWeightWindow`), covering both the reciprocal fast
  path and the `weight_at` fallback taken when the numerator does not fit `u64`;
* `recip_u64_some_spec`: `attn.recip_u64 total = ok (some r)` pins the reciprocal
  `r.val = 2^64/total.val.toNat + 1` and the window `2 ≤ total.val ≤ 2^63 - 1`, so the full
  specification can be instantiated from the extracted `word_attn_head` reciprocal call.

## Trust boundary

Reaches the extracted `attn.weight_at_recip`, `attn.weight_at` and `attn.recip_u64` of
`proof/ApeironWord.lean` at the digest `Numeric.EngineAttnRefinement` binds to. Does not cover
rustc/LLVM codegen of `src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe` container.
The headline is not yet registered in `AxiomAudit/Core.lean`: that module imports the currently-red
`Numeric.AttnHeadRefines`, so it cannot be imported here without reddening this gate; registration is
a separate task. The cone is checked in the receipt with `#print axioms`.
-/

import Numeric.EngineAttnRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Scratch: round-up reciprocal fixup arithmetic -/

/-- Core round-up-reciprocal fixup: with `R = 2^64 / T` (so `R + 1` is the round-up reciprocal of
`T`) and `n < 2^64`, the extracted fixup
`if (n·(R+1)/2^64)·T > n then (n·(R+1)/2^64) - 1 else n·(R+1)/2^64`
is exactly `n / T`. -/
theorem recip_fixup_R (n T R : ℕ) (hT0 : 0 < T) (hn : n < 2 ^ 64)
    (hR : R = 2 ^ 64 / T) :
    (if (n * (R + 1) / 2 ^ 64) * T > n then (n * (R + 1) / 2 ^ 64) - 1
      else n * (R + 1) / 2 ^ 64) = n / T := by
  set q := n / T with hq
  set qc := n * (R + 1) / 2 ^ 64 with hqc
  have hTpos : 0 < T := hT0
  have hq_lo : q * T ≤ n := by rw [hq]; exact Nat.div_mul_le_self n T
  have hn_lt : n < (q + 1) * T := by
    have hdm : T * q + n % T = n := by rw [hq]; exact Nat.div_add_mod n T
    have hmod : n % T < T := Nat.mod_lt n hTpos
    calc n = T * q + n % T := hdm.symm
      _ < T * q + T := Nat.add_lt_add_left hmod _
      _ = (q + 1) * T := by ring
  have hR_hi : R * T ≤ 2 ^ 64 := by rw [hR]; exact Nat.div_mul_le_self (2 ^ 64) T
  have hR_lo : 2 ^ 64 < T * (R + 1) := by
    have hdm : T * (2 ^ 64 / T) + 2 ^ 64 % T = 2 ^ 64 := Nat.div_add_mod (2 ^ 64) T
    have hmod : 2 ^ 64 % T < T := Nat.mod_lt (2 ^ 64) hTpos
    rw [hR]
    calc 2 ^ 64 = T * (2 ^ 64 / T) + 2 ^ 64 % T := hdm.symm
      _ < T * (2 ^ 64 / T) + T := Nat.add_lt_add_left hmod _
      _ = T * (2 ^ 64 / T + 1) := by ring
  have hqc_ge : q ≤ qc := by
    rw [hqc, Nat.le_div_iff_mul_le (by norm_num : 0 < 2 ^ 64)]
    calc q * 2 ^ 64 ≤ q * (T * (R + 1)) := Nat.mul_le_mul_left q (le_of_lt hR_lo)
      _ = q * T * (R + 1) := by ring
      _ ≤ n * (R + 1) := Nat.mul_le_mul_right (R + 1) hq_lo
  have hup : n * (R + 1) < (q + 2) * 2 ^ 64 := by
    have hC : (0 : ℚ) < T := by exact_mod_cast hT0
    have hnlt : (n : ℚ) < ((q : ℚ) + 1) * T := by exact_mod_cast hn_lt
    have hn64 : (n : ℚ) < (2 : ℚ) ^ 64 := by
      have hle : n ≤ 2 ^ 64 - 1 := by omega
      calc (n : ℚ) ≤ ((2 ^ 64 - 1 : ℕ) : ℚ) := by exact_mod_cast hle
        _ < (2 : ℚ) ^ 64 := by norm_num
    have hRq : ((R : ℚ) + 1) ≤ (2 : ℚ) ^ 64 / T + 1 := by
      have hRT : (R : ℚ) * T ≤ (2 : ℚ) ^ 64 := by
        have := hR_hi; exact_mod_cast this
      have hle : (R : ℚ) ≤ (2 : ℚ) ^ 64 / T := by
        rw [le_div_iff₀ hC]; linarith
      linarith
    have hupQ : (n : ℚ) * ((R : ℚ) + 1) < ((q : ℚ) + 2) * (2 : ℚ) ^ 64 := by
      calc (n : ℚ) * ((R : ℚ) + 1)
          ≤ (n : ℚ) * ((2 : ℚ) ^ 64 / T + 1) := by
            apply mul_le_mul_of_nonneg_left hRq; positivity
        _ = (n : ℚ) * (2 : ℚ) ^ 64 / T + n := by ring
        _ < ((q : ℚ) + 1) * T * (2 : ℚ) ^ 64 / T + n := by
            have h : (n : ℚ) * (2 : ℚ) ^ 64 / T < ((q : ℚ) + 1) * T * (2 : ℚ) ^ 64 / T := by
              apply div_lt_div_of_pos_right _ hC
              nlinarith [hnlt]
            linarith
        _ = ((q : ℚ) + 1) * (2 : ℚ) ^ 64 + n := by field_simp
        _ < ((q : ℚ) + 1) * (2 : ℚ) ^ 64 + (2 : ℚ) ^ 64 := by linarith
        _ = ((q : ℚ) + 2) * (2 : ℚ) ^ 64 := by ring
    exact_mod_cast hupQ
  have hqc_lt : qc < q + 2 := by
    rw [hqc]
    exact (Nat.div_lt_iff_lt_mul (by norm_num : 0 < 2 ^ 64)).mpr hup
  have hqc_le : qc ≤ q + 1 := by omega
  by_cases hgt : qc * T > n
  · have hqcq : qc = q + 1 := by
      have h3 : qc ≠ q := by
        intro h; rw [h] at hgt; exact (not_lt.mpr hq_lo) hgt
      omega
    rw [if_pos hgt, hqcq]
    simp
  · have hqcq : qc = q := by
      have hle : qc ≤ q := by
        by_contra h
        have hge : q + 1 ≤ qc := by omega
        have hmul : (q + 1) * T ≤ qc * T := Nat.mul_le_mul_right T hge
        have h2 : qc * T ≤ n := le_of_not_gt hgt
        omega
      omega
    rw [if_neg hgt, hqcq]

/-- Wrapper taking the extracted reciprocal value `c = 2^64 / T + 1`. -/
theorem recip_roundup_fixup (n T c : ℕ) (hT0 : 0 < T) (hn : n < 2 ^ 64)
    (hc : c = 2 ^ 64 / T + 1) :
    (if (n * c / 2 ^ 64) * T > n then (n * c / 2 ^ 64) - 1 else n * c / 2 ^ 64) = n / T := by
  have h := recip_fixup_R n T (2 ^ 64 / T) hT0 hn rfl
  rw [hc]; exact h

/-- The `U64` cast of a `U128` value below `2^64` is the value itself. -/
theorem uscalar_cast_u64_val (x : U128) (hx : x.val < 2 ^ 64) :
    (UScalar.cast UScalarTy.U64 x).val = x.val :=
  UScalar.cast_val_mod_pow_of_inBounds_eq .U64 x (by simpa using hx)

/-- **Reciprocal-arm weight specification.** For the round-up `u64` reciprocal
`recip = 2^64 / T + 1` of the positive total `T = total.val`, the extracted
`attn.weight_at_recip e total recip` returns `Ok w` with `w.val` the half-biased division
`divRoundNearest (e·2^32) total.val`, on the admitted window (nonnegative numerator, `T ≥ 2`,
`T ≤ 2^63 - 1`, and the `u64` narrowing `e·2^32 + T/2 ≤ 2^64 - 1`). -/
theorem weight_at_recip_fast (e : I64) (total : I128) (recip : U64)
    (htot : 0 < total.val)
    (hnum : 0 ≤ e.val)
    (hreclo : 2 ≤ total.val.toNat)
    (hrechi : total.val.toNat ≤ 2 ^ 63 - 1)
    (hrecip : recip.val = 2 ^ 64 / total.val.toNat + 1)
    (hadd : e.val * (2 : ℤ) ^ 32 + total.val / 2 ≤ (2 : ℤ) ^ 64 - 1)
    (hfitlo : -((2 : ℤ) ^ 63) ≤ divRoundNearest (e.val * (2 : ℤ) ^ 32) total.val)
    (hfithi : divRoundNearest (e.val * (2 : ℤ) ^ 32) total.val ≤ (2 : ℤ) ^ 63 - 1) :
    ∃ w : I64, attn.weight_at_recip e total recip = ok (core.result.Result.Ok w) ∧
      w.val = divRoundNearest (e.val * (2 : ℤ) ^ 32) total.val := by
  set T : ℕ := total.val.toNat with hTdef
  have hTval : (T : ℤ) = total.val := by rw [hTdef]; exact Int.toNat_of_nonneg (le_of_lt htot)
  have hTpos : 0 < T := by omega
  have hTle : T ≤ 2 ^ 64 - 1 := by omega
  have hTlt : T < 2 ^ 64 := by omega
  -- numerator product
  obtain ⟨s, hs_eq, hsv⟩ := I128_mul_q32_some (core.convert.num.FromI128I64.from e) (by
    have hb := e.hBounds
    rw [IScalarTy.I64_numBits_eq] at hb
    rw [core.convert.num.FromI128I64.from_val_eq, abs_le]
    constructor <;> omega)
  have hsv' : s.val = e.val * (2 : ℤ) ^ 32 := by
    rw [hsv, core.convert.num.FromI128I64.from_val_eq]; norm_num
  -- half of the total
  obtain ⟨i3, hi3_eq, hi3v⟩ := WP.spec_imp_exists
    (I128.div_spec (x := total) (y := 2#i128) (by norm_num)
      (by intro h; exact absurd h.2 (by norm_num)))
  have hi3v' : i3.val = ((T / 2 : ℕ) : ℤ) := by
    rw [Int.tdiv_eq_ediv_of_nonneg (le_of_lt htot)] at hi3v
    have h2 : total.val / (2 : ℤ) = ((T / 2 : ℕ) : ℤ) := by
      rw [← hTval]; exact (Int.natCast_ediv T 2).symm
    have hi3n : (i3.val : ℤ) = total.val / (2 : ℤ) := by simpa using hi3v
    rw [h2] at hi3n
    exact_mod_cast hi3n
  have haddZ : e.val * (2 : ℤ) ^ 32 + (T : ℤ) / 2 ≤ (2 : ℤ) ^ 64 - 1 := by
    rw [hTval]; exact hadd
  obtain ⟨nI, hn_eq, hnv⟩ := I128_checked_add_some s i3
    (by
      have h0 : 0 ≤ e.val * (2 : ℤ) ^ 32 := mul_nonneg hnum (by norm_num)
      have h0' : 0 ≤ ((T / 2 : ℕ) : ℤ) := by positivity
      rw [hsv', hi3v']
      have hmin : I128.min = -((2 : ℤ) ^ 127) := by norm_num [I128.min, IScalar.min, I128.numBits]
      rw [hmin]; linarith)
    (by
      rw [hsv', hi3v']
      have hmax : I128.max = (2 : ℤ) ^ 127 - 1 := by
        norm_num [I128.max, IScalar.max, I128.numBits]
      rw [hmax]
      have hnat : ((T / 2 : ℕ) : ℤ) = (T : ℤ) / 2 := by
        rw [Int.natCast_ediv]; norm_num
      rw [hnat]
      linarith [haddZ])
  have hnvZ : nI.val = e.val * (2 : ℤ) ^ 32 + (T : ℤ) / 2 := by
    rw [hnv, hsv', hi3v']
    rw [Int.natCast_ediv]; norm_num
  -- the u64 numerator
  let n : ℕ := nI.val.toNat
  have hnInn : 0 ≤ nI.val := by rw [hnvZ]; positivity
  have hnIle : nI.val ≤ (2 : ℤ) ^ 64 - 1 := by
    rw [hnvZ]; exact haddZ
  have hnval : (n : ℤ) = e.val * (2 : ℤ) ^ 32 + (T : ℤ) / 2 := by
    show ((nI.val.toNat : ℕ) : ℤ) = _
    rw [Int.toNat_of_nonneg hnInn]; exact hnvZ
  have hn : n < 2 ^ 64 := by
    have : (n : ℤ) < 2 ^ 64 := by rw [hnval]; linarith
    exact_mod_cast this
  -- the division relation
  have hqr : ((n / T : ℕ) : ℤ) = divRoundNearest (e.val * (2 : ℤ) ^ 32) total.val := by
    unfold divRoundNearest
    rw [← hTval, ← hnval]
    exact (Int.natCast_ediv n T).symm
  -- hcast of the numerator to U64
  have hidx : (IScalar.hcast UScalarTy.U64 nI).val = n := by
    rw [IScalar.hcast_val_eq, UScalarTy.U64_numBits_eq]
    have hmod : nI.val % (2 : ℤ) ^ 64 = nI.val :=
      Int.emod_eq_of_lt hnInn (by rw [hnvZ]; linarith)
    rw [hmod]
  -- i5 = (fromU128 n2) * (fromU128 recip)
  have hrec_le : recip.val ≤ 2 ^ 63 + 1 := by
    rw [hrecip]
    have hdiv : 2 ^ 64 / T ≤ 2 ^ 63 := by
      rw [Nat.div_le_iff_le_mul_add_pred hTpos]
      nlinarith [hreclo]
    omega
  have hmul5 : (core.convert.num.FromU128U64.from (IScalar.hcast UScalarTy.U64 nI)).val *
      (core.convert.num.FromU128U64.from recip).val ≤ U128.max := by
    rw [core.convert.num.FromU128U64.from_val_eq, core.convert.num.FromU128U64.from_val_eq, hidx]
    have hnle : n ≤ 2 ^ 64 - 1 := by omega
    calc n * recip.val ≤ (2 ^ 64 - 1) * (2 ^ 63 + 1) := Nat.mul_le_mul hnle hrec_le
      _ ≤ U128.max := by norm_num [U128.max, U128.numBits]
  obtain ⟨i5, hi5_eq, hi5v⟩ := WP.spec_imp_exists
    (U128.mul_spec (x := core.convert.num.FromU128U64.from (IScalar.hcast UScalarTy.U64 nI))
      (y := core.convert.num.FromU128U64.from recip) hmul5)
  have hi5val : i5.val = n * recip.val := by
    rw [hi5v, core.convert.num.FromU128U64.from_val_eq,
      core.convert.num.FromU128U64.from_val_eq, hidx]
  -- i6 = i5 >>> 64
  obtain ⟨i6, hi6_eq, hi6v, -⟩ := WP.spec_imp_exists
    (UScalar.ShiftRight_IScalar_spec (x := i5) (y := 64#i32) (by norm_num) (by norm_num))
  have hi6val : i6.val = n * recip.val / 2 ^ 64 := by
    rw [hi6v]
    have h64 : (64#i32).toNat = 64 := rfl
    rw [h64, Nat.shiftRight_eq_div_pow, hi5val]
  have hi6lt : i6.val < 2 ^ 64 := by
    rw [hi6val]
    rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 2 ^ 64)]
    calc n * recip.val ≤ (2 ^ 64 - 1) * (2 ^ 63 + 1) :=
          Nat.mul_le_mul (by omega) hrec_le
      _ < 2 ^ 64 * 2 ^ 64 := by norm_num
  have hqqval : (UScalar.cast UScalarTy.U64 i6).val = n * recip.val / 2 ^ 64 := by
    rw [uscalar_cast_u64_val i6 hi6lt, hi6val]
  -- d = hcast total, and i9 = qq * d
  have hdval : (IScalar.hcast UScalarTy.U64 total).val = T := by
    rw [IScalar.hcast_val_eq, UScalarTy.U64_numBits_eq]
    have hmod : total.val % (2 : ℤ) ^ 64 = total.val :=
      Int.emod_eq_of_lt (le_of_lt htot) (by rw [← hTval]; exact_mod_cast hTlt)
    rw [hmod]
  have hmul9 : (core.convert.num.FromU128U64.from (UScalar.cast UScalarTy.U64 i6)).val *
      (core.convert.num.FromU128U64.from (IScalar.hcast UScalarTy.U64 total)).val ≤ U128.max := by
    rw [core.convert.num.FromU128U64.from_val_eq,
      core.convert.num.FromU128U64.from_val_eq, hqqval, hdval]
    have hqlt : n * recip.val / 2 ^ 64 < 2 ^ 64 := by rw [← hi6val]; exact hi6lt
    have hprod : (n * recip.val / 2 ^ 64) * T < 2 ^ 128 := by
      nlinarith [hqlt, hTlt]
    have hmax : U128.max = 2 ^ 128 - 1 := by norm_num [U128.max, U128.numBits]
    rw [hmax]; omega
  obtain ⟨i9, hi9_eq, hi9v⟩ := WP.spec_imp_exists
    (U128.mul_spec (x := core.convert.num.FromU128U64.from (UScalar.cast UScalarTy.U64 i6))
      (y := core.convert.num.FromU128U64.from (IScalar.hcast UScalarTy.U64 total)) hmul9)
  have hi9val : i9.val = (n * recip.val / 2 ^ 64) * T := by
    rw [hi9v, core.convert.num.FromU128U64.from_val_eq,
      core.convert.num.FromU128U64.from_val_eq, hqqval, hdval]
  -- the fixup conditional
  have hfix := recip_roundup_fixup n T recip.val hTpos hn hrecip
  have hq1spec : (if i9 > core.convert.num.FromU128U64.from (IScalar.hcast UScalarTy.U64 nI)
      then UScalar.cast UScalarTy.U64 i6 - 1#u64 else ok (UScalar.cast UScalarTy.U64 i6))
      ⦃ q => q.val = n / T ⦄ := by
    by_cases hgt : i9 > core.convert.num.FromU128U64.from (IScalar.hcast UScalarTy.U64 nI)
    · rw [if_pos hgt]
      have hgtv : (n * recip.val / 2 ^ 64) * T > n := by
        have h := (UScalar.lt_equiv
          (core.convert.num.FromU128U64.from (IScalar.hcast UScalarTy.U64 nI)) i9).1 hgt
        rw [hi9val, core.convert.num.FromU128U64.from_val_eq, hidx] at h
        exact h
      have hqcpos : 1 ≤ n * recip.val / 2 ^ 64 := by
        by_contra h
        have h0 : n * recip.val / 2 ^ 64 = 0 := by omega
        rw [h0, Nat.zero_mul] at hgtv
        omega
      have hfix' : n * recip.val / 2 ^ 64 - 1 = n / T := by
        rw [← hfix, if_pos hgtv]
      refine WP.spec_mono (UScalar.sub_spec (x := UScalar.cast UScalarTy.U64 i6)
        (y := 1#u64) (by rw [hqqval]; exact hqcpos)) ?_
      intro q hq
      rw [hq.1]
      simpa [hqqval] using hfix'
    · rw [if_neg hgt, WP.spec_ok]
      have hnt : ¬ (n * recip.val / 2 ^ 64) * T > n := by
        intro hc
        apply hgt
        apply (UScalar.lt_equiv
          (core.convert.num.FromU128U64.from (IScalar.hcast UScalarTy.U64 nI)) i9).2
        rw [hi9val, core.convert.num.FromU128U64.from_val_eq, hidx]; exact hc
      have hfix' : n * recip.val / 2 ^ 64 = n / T := by
        rw [← hfix, if_neg hnt]
      rw [hqqval]; exact hfix'
  obtain ⟨q1, hq1_eq, hq1v⟩ := WP.spec_imp_exists hq1spec
  -- p = hcast q1 to I128
  have hpval : (UScalar.hcast IScalarTy.I128 q1).val = (q1.val : ℤ) := by
    rw [UScalar.hcast_val_eq, IScalarTy.I128_numBits_eq]
    have hbmod : Int.bmod (q1.val : ℤ) (2 ^ 128) = (q1.val : ℤ) := by
      apply Int.bmod_eq_of_le
      · have h0 : (0 : ℤ) ≤ (q1.val : ℤ) := by positivity
        have hd : (↑(2 ^ 128 : ℕ) : ℤ) / 2 = (2 : ℤ) ^ 127 := by norm_num
        rw [hd]; linarith
      · have hq1n : (q1.val : ℤ) = ((n / T : ℕ) : ℤ) := by rw [hq1v]
        have hd : (↑(2 ^ 128 : ℕ) + 1) / 2 = (2 : ℤ) ^ 127 := by norm_num
        rw [hd, hq1n, hqr]; linarith [hfithi]
    rw [hbmod]
  have hpv : (UScalar.hcast IScalarTy.I128 q1).val = ((n / T : ℕ) : ℤ) := by
    rw [hpval, hq1v]
  -- the epilogue narrowing ifs
  have hmaxv : (core.convert.num.FromI128I64.from core.num.I64.MAX).val = (2 : ℤ) ^ 63 - 1 := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ 63 - 1 := by decide
    rw [h]
  have hminv : (core.convert.num.FromI128I64.from core.num.I64.MIN).val = -((2 : ℤ) ^ 63) := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MIN.val = -((2 : ℤ) ^ 63) := by decide
    rw [h]
  have hmaxnot : ¬ ((UScalar.hcast IScalarTy.I128 q1) >
      core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro h
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv, hpv, hqr] at h
    omega
  have hminnot : ¬ ((UScalar.hcast IScalarTy.I128 q1) <
      core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro h
    rw [IScalar.lt_equiv, hminv, hpv, hqr] at h
    omega
  have hwval : (IScalar.cast IScalarTy.I64 (UScalar.hcast IScalarTy.I128 q1)).val =
      ((n / T : ℕ) : ℤ) := by
    rw [IScalar.cast_val_eq, IScalarTy.I64_numBits_eq, IScalarTy.I128_numBits_eq,
      Nat.min_eq_left (by norm_num)]
    have hbmod : Int.bmod (UScalar.hcast IScalarTy.I128 q1).val (2 ^ 64) =
        (UScalar.hcast IScalarTy.I128 q1).val := by
      apply Int.bmod_eq_of_le
      · have h0 : (0 : ℤ) ≤ (UScalar.hcast IScalarTy.I128 q1).val := by
          rw [hpval]; positivity
        have hd : (↑(2 ^ 64 : ℕ) : ℤ) / 2 = (2 : ℤ) ^ 63 := by norm_num
        rw [hd]; linarith
      · have hd : (↑(2 ^ 64 : ℕ) + 1) / 2 = (2 : ℤ) ^ 63 := by norm_num
        rw [hd, hpv, hqr]; linarith [hfithi]
    rw [hbmod, hpv]
  have he_ge : (0#i64 : I64) ≤ e := by rw [IScalar.le_equiv]; simpa using hnum
  have hi4v : (UScalar.hcast IScalarTy.I128 core.num.U64.MAX).val = (2 : ℤ) ^ 64 - 1 := by
    rw [UScalar.hcast_val_eq, IScalarTy.I128_numBits_eq]
    have h : core.num.U64.MAX.val = 2 ^ 64 - 1 := by decide
    rw [h, Int.bmod_eq_of_le] <;> norm_num
  have hnle : nI ≤ UScalar.hcast IScalarTy.I128 core.num.U64.MAX := by
    rw [IScalar.le_equiv, hi4v]; exact hnIle
  refine ⟨IScalar.cast IScalarTy.I64 (UScalar.hcast IScalarTy.I128 q1), ?_, ?_⟩
  · rw [attn.weight_at_recip]
    simp only [lift, bind_tc_ok, engine_ONE_eq, hs_eq, hi3_eq, hn_eq, if_pos he_ge,
      I128.Insts.CoreConvertFromU64.from, if_pos hnle, hi5_eq, hi6_eq, hi9_eq, hq1_eq,
      if_neg hmaxnot, if_neg hminnot]
  · rw [hwval, hqr]

/-- **Reciprocal-arm weight specification (full).** On the whole admitted numerator window
(`e·2^32 + total/2 ≤ I128.max`, matching `AttnWeightWindow`), `attn.weight_at_recip e total recip`
returns `Ok w` with `w.val = divRoundNearest (e·2^32) total.val`, whichever internal arm the
numerator narrowing selects: the round-up-reciprocal fast path when the numerator fits `u64`, and
the `weight_at` fallback otherwise. -/
theorem weight_at_recip_spec (e : I64) (total : I128) (recip : U64)
    (htot : 0 < total.val)
    (hnum : 0 ≤ e.val)
    (hreclo : 2 ≤ total.val.toNat)
    (hrechi : total.val.toNat ≤ 2 ^ 63 - 1)
    (hrecip : recip.val = 2 ^ 64 / total.val.toNat + 1)
    (haddI : e.val * (2 : ℤ) ^ 32 + total.val / 2 ≤ I128.max)
    (hfit : -((2 : ℤ) ^ 63) ≤ divRoundNearest (e.val * (2 : ℤ) ^ 32) total.val ∧
      divRoundNearest (e.val * (2 : ℤ) ^ 32) total.val ≤ (2 : ℤ) ^ 63 - 1) :
    ∃ w : I64, attn.weight_at_recip e total recip = ok (core.result.Result.Ok w) ∧
      w.val = divRoundNearest (e.val * (2 : ℤ) ^ 32) total.val := by
  by_cases hfast : e.val * (2 : ℤ) ^ 32 + total.val / 2 ≤ (2 : ℤ) ^ 64 - 1
  · exact weight_at_recip_fast e total recip htot hnum hreclo hrechi hrecip hfast hfit.1 hfit.2
  · set T : ℕ := total.val.toNat with hTdef
    have hTval : (T : ℤ) = total.val := by rw [hTdef]; exact Int.toNat_of_nonneg (le_of_lt htot)
    have hbig : (2 : ℤ) ^ 64 - 1 < e.val * (2 : ℤ) ^ 32 + (T : ℤ) / 2 := by
      rw [hTval]; exact not_le.mp hfast
    have hTlt : T < 2 ^ 64 := by omega
    obtain ⟨s, hs_eq, hsv⟩ := I128_mul_q32_some (core.convert.num.FromI128I64.from e) (by
      have hb := e.hBounds
      rw [IScalarTy.I64_numBits_eq] at hb
      rw [core.convert.num.FromI128I64.from_val_eq, abs_le]
      constructor <;> omega)
    have hsv' : s.val = e.val * (2 : ℤ) ^ 32 := by
      rw [hsv, core.convert.num.FromI128I64.from_val_eq]; norm_num
    obtain ⟨i3, hi3_eq, hi3v⟩ := WP.spec_imp_exists
      (I128.div_spec (x := total) (y := 2#i128) (by norm_num)
        (by intro h; exact absurd h.2 (by norm_num)))
    have hi3v' : i3.val = ((T / 2 : ℕ) : ℤ) := by
      rw [Int.tdiv_eq_ediv_of_nonneg (le_of_lt htot)] at hi3v
      have h2 : total.val / (2 : ℤ) = ((T / 2 : ℕ) : ℤ) := by
        rw [← hTval]; exact (Int.natCast_ediv T 2).symm
      have hi3n : (i3.val : ℤ) = total.val / (2 : ℤ) := by simpa using hi3v
      rw [h2] at hi3n
      exact_mod_cast hi3n
    have haddZ : e.val * (2 : ℤ) ^ 32 + (T : ℤ) / 2 ≤ I128.max := by
      rw [hTval]; exact haddI
    obtain ⟨nI, hn_eq, hnv⟩ := I128_checked_add_some s i3
      (by
        have h0 : 0 ≤ e.val * (2 : ℤ) ^ 32 := mul_nonneg hnum (by norm_num)
        have h0' : 0 ≤ ((T / 2 : ℕ) : ℤ) := by positivity
        rw [hsv', hi3v']
        have hmin : I128.min = -((2 : ℤ) ^ 127) := by norm_num [I128.min, IScalar.min, I128.numBits]
        rw [hmin]; linarith)
      (by
        rw [hsv', hi3v']
        have hnat : ((T / 2 : ℕ) : ℤ) = (T : ℤ) / 2 := by
          rw [Int.natCast_ediv]; norm_num
        rw [hnat]; exact haddZ)
    have hnvZ : nI.val = e.val * (2 : ℤ) ^ 32 + (T : ℤ) / 2 := by
      rw [hnv, hsv', hi3v']
      rw [Int.natCast_ediv]; norm_num
    have hbigN : (2 : ℤ) ^ 64 - 1 < nI.val := by rw [hnvZ]; exact hbig
    have hi4v : (UScalar.hcast IScalarTy.I128 core.num.U64.MAX).val = (2 : ℤ) ^ 64 - 1 := by
      rw [UScalar.hcast_val_eq, IScalarTy.I128_numBits_eq]
      have h : core.num.U64.MAX.val = 2 ^ 64 - 1 := by decide
      rw [h, Int.bmod_eq_of_le] <;> norm_num
    have he_ge : (0#i64 : I64) ≤ e := by rw [IScalar.le_equiv]; simpa using hnum
    have hnle_not : ¬ (nI ≤ UScalar.hcast IScalarTy.I128 core.num.U64.MAX) := by
      rw [IScalar.le_equiv, hi4v]; omega
    have hfall : attn.weight_at_recip e total recip = attn.weight_at e total := by
      rw [attn.weight_at_recip]
      simp only [lift, bind_tc_ok, engine_ONE_eq, hs_eq, hi3_eq, hn_eq, if_pos he_ge,
        I128.Insts.CoreConvertFromU64.from, if_neg hnle_not]
    rw [hfall]
    exact weight_at_spec e total hnum htot haddI hfit

theorem recip_u64_some_spec (total : I128) (r : U64)
    (h : attn.recip_u64 total = ok (some r)) :
    2 ≤ total.val ∧ total.val ≤ (2 : ℤ) ^ 63 - 1 ∧
      r.val = 2 ^ 64 / total.val.toNat + 1 := by
  rw [attn.recip_u64] at h
  simp only [lift, bind_tc_ok] at h
  have hge : ¬ total < 2#i128 := by
    intro hlt; rw [if_pos hlt] at h; simp at h
  have h2 : 2 ≤ total.val := by
    have := (IScalar.le_equiv (2#i128) total).1 (le_of_not_gt hge)
    simpa using this
  have htot : 0 < total.val := by linarith [h2]
  rw [if_neg hge] at h
  obtain ⟨i, hi_eq, hiv⟩ := WP.spec_imp_exists
    (U64.div_spec (x := core.num.U64.MAX) (y := 2#u64) (by norm_num))
  rw [hi_eq] at h; simp only [bind_tc_ok] at h
  rw [show I128.Insts.CoreConvertFromU64.from i = ok (UScalar.hcast IScalarTy.I128 i)
        from rfl] at h
  simp only [bind_tc_ok] at h
  have hiv' : i.val = 2 ^ 63 - 1 := by
    rw [hiv]
    have hmax : core.num.U64.MAX.val = 2 ^ 64 - 1 := by decide
    rw [hmax]; norm_num
  have hi1v : (UScalar.hcast IScalarTy.I128 i).val = (2 : ℤ) ^ 63 - 1 := by
    rw [UScalar.hcast_val_eq, IScalarTy.I128_numBits_eq, hiv', Int.bmod_eq_of_le]
    · norm_num
    · norm_num
    · norm_num
  have hle : ¬ total > UScalar.hcast IScalarTy.I128 i := by
    intro hgt; rw [if_pos hgt] at h; simp at h
  have h3 : total.val ≤ (2 : ℤ) ^ 63 - 1 := by
    have := (IScalar.le_equiv total (UScalar.hcast IScalarTy.I128 i)).1 (le_of_not_gt hle)
    rwa [hi1v] at this
  rw [if_neg hle] at h
  obtain ⟨i2, hi2_eq, hi2v, -⟩ := WP.spec_imp_exists
    (U128.ShiftLeft_IScalar_spec (x := 1#u128) (y := 64#i32) (by norm_num) (by norm_num))
  rw [hi2_eq] at h; simp only [bind_tc_ok] at h
  have hi2val : i2.val = 2 ^ 64 := by
    rw [hi2v]
    have h64 : (64#i32).toNat = 64 := rfl
    rw [h64, Nat.shiftLeft_eq]; norm_num [U128.size, U128.numBits, UScalarTy.U128_numBits_eq]
  have hi3val : (IScalar.hcast UScalarTy.U128 total).val = total.val.toNat := by
    rw [IScalar.hcast_val_eq, UScalarTy.U128_numBits_eq]
    have hmod : total.val % (2 : ℤ) ^ 128 = total.val :=
      Int.emod_eq_of_lt (le_of_lt htot) (by
        have : (total.val.toNat : ℤ) = total.val := Int.toNat_of_nonneg (le_of_lt htot)
        omega)
    rw [hmod]
  have hi3pos : (IScalar.hcast UScalarTy.U128 total).val ≠ 0 := by
    rw [hi3val]; omega
  obtain ⟨i4, hi4_eq, hi4v⟩ := WP.spec_imp_exists
    (U128.div_spec (x := i2) (y := IScalar.hcast UScalarTy.U128 total) hi3pos)
  rw [hi4_eq] at h; simp only [bind_tc_ok] at h
  have hi4val : i4.val = 2 ^ 64 / total.val.toNat := by
    rw [hi4v, hi2val, hi3val]
  have hi4lt : i4.val < 2 ^ 64 := by
    rw [hi4val]
    have ht2 : 2 ≤ total.val.toNat := by
      have : (total.val.toNat : ℤ) = total.val := Int.toNat_of_nonneg (le_of_lt htot)
      omega
    have hle64 : 2 ^ 64 / total.val.toNat ≤ 2 ^ 63 := by
      rw [Nat.div_le_iff_le_mul_add_pred (by omega : 0 < total.val.toNat)]
      nlinarith [ht2]
    omega
  have hbaseval : (UScalar.cast UScalarTy.U64 i4).val = 2 ^ 64 / total.val.toNat := by
    rw [uscalar_cast_u64_val i4 hi4lt, hi4val]
  have hc : U64.checked_add (UScalar.cast UScalarTy.U64 i4) 1#u64 = some r := by
    simpa using h
  have hrval : r.val = 2 ^ 64 / total.val.toNat + 1 := by
    have hs := U64.checked_add_bv_spec (UScalar.cast UScalarTy.U64 i4) 1#u64
    rw [hc] at hs
    rw [hs.2.1, hbaseval]
    simp
  exact ⟨h2, h3, hrval⟩

end

end apeiron_word
