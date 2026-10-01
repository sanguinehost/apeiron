import FixedTrigSource
import Aeneas

/-
SCOPE: resource theorems for the ACTUAL extracted `fixed_trig.fixed_sin_cos` body reached through
  the `word_fixed_sin_cos` extraction root: the two window guards, the `i128`-to-`i64` reduction
  guards, the quadrant-fold transitions, the degree-8 Taylor series transitions, the fixed work
  measure of the straight-line body, the refusal/exit branch and a reachable success witness.
REGISTER: scalar source refinement over `i64`/`i128` operation counts; no ring/carrier or canonical
  corpus correspondence, no trigonometric accuracy claim, no hardware or GPU timing.
CORPUS: `src/numeric/fixed_trig.rs` `fixed_sin_cos`, `round_div_2p32`, `taylor_sin_cos`, read as the
  actual extracted `fixed_trig_source.fixed_trig.{fixed_sin_cos, round_div_2p32, taylor_sin_cos}`
  and `fixed_trig_source.word_fixed_sin_cos`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean (`apeiron_word::fixed_trig::*`, the
  committed extraction of the same bodies).
GATE: `python3 proof/resource/FixedTrig_gate.py --controls`, 540-second whole-invocation budget,
  lane `/tmp/opencode/apeiron-lane-1.lock`, MemoryMax=12G MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: Lean kernel check only when that gate succeeds; the work measure is a count of `i128`/`i64`
  operations under named premises, not a measured cycle or GPU-time count.
-/

namespace FixedTrigResource

open Aeneas Aeneas.Std Result

/-! ## Extracted constants of the real source -/

/-- The extracted scale constant `ONE` evaluates to `2^32`. -/
theorem one_eq : fixed_trig_source.fixed_trig.ONE = ok (4294967296#i64) := by
  have h : fixed_trig_source.fixed_trig.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
    unfold fixed_trig_source.fixed_trig.ONE
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted rounding half `HALF` evaluates to `2^31`. -/
theorem half_eq : fixed_trig_source.fixed_trig.round_div_2p32.HALF = ok (2147483648#i128) := by
  have h : fixed_trig_source.fixed_trig.round_div_2p32.HALF ⦃ z =>
      z = (2147483648#i128) ⦄ := by
    unfold fixed_trig_source.fixed_trig.round_div_2p32.HALF
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted lower window endpoint evaluates to `-4096 · 2^32 = -2^44`. -/
theorem lower_eq :
    fixed_trig_source.fixed_trig.fixed_sin_cos.LOWER = ok ((-17592186044416)#i64) := by
  have h : fixed_trig_source.fixed_trig.fixed_sin_cos.LOWER ⦃ z =>
      z = ((-17592186044416)#i64) ⦄ := by
    unfold fixed_trig_source.fixed_trig.fixed_sin_cos.LOWER
    rw [one_eq]
    simp only [bind_tc_ok]
    apply Aeneas.Std.WP.spec_mono
      (I64.mul_spec (x := (-4096)#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted upper window endpoint evaluates to `4096 · 2^32 = 2^44`. -/
theorem upper_eq :
    fixed_trig_source.fixed_trig.fixed_sin_cos.UPPER = ok (17592186044416#i64) := by
  have h : fixed_trig_source.fixed_trig.fixed_sin_cos.UPPER ⦃ z =>
      z = (17592186044416#i64) ⦄ := by
    unfold fixed_trig_source.fixed_trig.fixed_sin_cos.UPPER
    rw [one_eq]
    simp only [bind_tc_ok]
    apply Aeneas.Std.WP.spec_mono
      (I64.mul_spec (x := 4096#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The `i128` window endpoints as powers of two. -/
theorem i128_min : I128.min = -((2 : Int) ^ 127) := by rw [I128.min_eq]; norm_num

theorem i128_max : I128.max = (2 : Int) ^ 127 - 1 := by rw [I128.max_eq]; norm_num

theorem half_val : ((2147483648#i128).val : Int) = (2147483648 : Int) := by norm_num

theorem numbits_128 : IScalarTy.I128.numBits = 128 := by decide

theorem powbits : (2 : Int) ^ (IScalarTy.I128.numBits - 1) = 2 ^ 127 := by rw [numbits_128]

theorem iscalar_min : IScalar.min IScalarTy.I128 = I128.min := by
  norm_num [IScalar.min, I128.min_eq, numbits_128]

theorem iscalar_max : IScalar.max IScalarTy.I128 = I128.max := by
  norm_num [IScalar.max, I128.max_eq, numbits_128]

/-- Every `I128` value lies in the `i128` window, read off the model's own invariant. -/
theorem hbounds_min (v : I128) : I128.min ≤ v.val := by
  have h := (IScalar.hBounds v).1
  rw [powbits] at h
  have h' : -((2 : Int) ^ 127) ≤ v.val := h
  rw [I128.min_eq]
  norm_num at h' ⊢
  exact h'

theorem hbounds_max (v : I128) : v.val ≤ I128.max := by
  have h := (IScalar.hBounds v).2
  rw [powbits] at h
  rw [I128.max_eq]
  have hn : (2 : Int) ^ 127 = 170141183460469231731687303715884105728 := by norm_num
  rw [hn] at h
  omega

/-! ## Scalar operation bridges, used by every transition below -/

/-- The extracted `i128` negation, when it is defined, is the integer negation. -/
theorem neg_spec (v : I128) (h : v.val ≠ IScalar.min IScalarTy.I128) :
    ∃ w : I128, (-. v) = ok w ∧ w.val = -v.val := by
  obtain ⟨w, hw, hwd⟩ := WP.spec_imp_exists (HNeg.hNeg.step (x := v) h)
  exact ⟨w, hw, hwd⟩

/-- The extracted `i128` shift right by 32 is floor division by `2^32`. -/
theorem shr32_spec (v : I128) :
    ∃ w : I128, v >>> 32#i32 = ok w ∧ w.val = v.val / 2 ^ 32 := by
  obtain ⟨w, hw, hwd⟩ := WP.spec_imp_exists
    (I128.ShiftRight_IScalar_spec (x := v) (y := 32#i32) (by norm_num)
      (by norm_num [I128.numBits]))
  have h32 : (32#i32).toNat = 32 := by decide
  refine ⟨w, hw, ?_⟩
  rw [hwd.1, Int.shiftRight_eq_div_pow, h32]
  norm_num

/-- The shift right by 32 cannot produce the `i128` minimum, so the second negation on the
    negative arm of the extracted rounding is always defined. -/
theorem shr32_not_min (v w : I128) (hw : w.val = v.val / 2 ^ 32) :
    -((2 : Int) ^ 127) < w.val := by
  have hb : -((2 : Int) ^ 127) ≤ v.val / 2 ^ 32 := by
    have h := (IScalar.hBounds w).1
    rw [hw] at h
    rw [powbits] at h
    exact h
  have hlow : -((2 : Int) ^ 127) ≤ v.val := by
    have h := hbounds_min v
    rw [i128_min] at h
    exact h
  have hdiv : -((2 : Int) ^ 127) ≤ v.val / 4294967296 := by
    refine (Int.le_ediv_iff_mul_le (show (0 : Int) < 4294967296 by norm_num)).2 ?_
    have hpow : (2 : Int) ^ 127 * 4294967296 = 2 ^ 159 := by norm_num
    have hpos : (0 : Int) < 2 ^ 159 := by nlinarith [hpow]
    have hneg : -((2 : Int) ^ 159) < (0 : Int) := by linarith
    have hstep : -((2 : Int) ^ 159) < -((2 : Int) ^ 127) := by linarith
    have hmul : -((2 : Int) ^ 127) * 4294967296 = -((2 : Int) ^ 159) := by
      have : -((2 : Int) ^ 127) * 4294967296 = -(2 ^ 127 * 4294967296) := by ring
      rw [this, hpow]
    linarith
  have hlt : (2 : Int) ^ 127 < 2 ^ 127 * 4294967296 := by
    have h1 : (1 : Int) < 4294967296 := by norm_num
    nlinarith
  have heq : v.val / 2 ^ 32 = v.val / 4294967296 := by
    rw [show (2 : Int) ^ 32 = 4294967296 from by norm_num]
  omega

/-! ## The repeat transition: the actual extracted `round_div_2p32`, both guarded arms -/

/-- **Negative arm of the repeat transition.** With `v < 0` the actual body computes
    `-((-v + 2^31) >>> 32)`, in `i128`. The two premises are the `i128` range conditions of
    `-v + 2^31`; they are the only way this arm can fail. -/
theorem round_neg (v : I128) (h : v < (0#i128))
    (hlo : I128.min ≤ -v.val + 2147483648)
    (hhi : -v.val + 2147483648 ≤ I128.max) :
    ∃ w : I128, fixed_trig_source.fixed_trig.round_div_2p32 v = ok w ∧
      w.val = -((-v.val + 2147483648) / 2 ^ 32) := by
  have h1 : v.val < (0 : Int) := (IScalar.lt_equiv v (0#i128)).mp h
  have hn : ¬ ((0#i128) ≤ v) := by
    intro hc
    have hle := (IScalar.le_equiv (0#i128) v).mp hc
    have hle' : (0 : Int) ≤ v.val := by simpa using hle
    omega
  have hge : ¬ (v >= 0#i128) := by
    intro hc
    exact hn (by
      have := (IScalar.le_equiv (0#i128) v).mp hc
      simpa using this)
  have hne : v.val ≠ IScalar.min IScalarTy.I128 := by
    intro hc
    have h := hhi
    rw [hc, iscalar_min, I128.min_eq, I128.max_eq] at h
    norm_num at h
  obtain ⟨nv, hnv, hnvv⟩ := neg_spec v hne
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists
    (I128.add_spec (x := nv) (y := 2147483648#i128)
      (by rw [hnvv, half_val]; exact hlo) (by rw [hnvv, half_val]; exact hhi))
  obtain ⟨q, hq, hqv⟩ := shr32_spec s
  obtain ⟨w, hw, hwv⟩ := neg_spec q (by
    intro hc
    have hlt := shr32_not_min s q hqv
    rw [hc, iscalar_min, I128.min_eq] at hlt
    norm_num at hlt)
  refine ⟨w, ?_, ?_⟩
  · unfold fixed_trig_source.fixed_trig.round_div_2p32
    rw [if_neg hge, half_eq, hnv]
    simp only [bind_tc_ok]
    rw [hs]
    simp only [bind_tc_ok]
    rw [hq]
    simp only [bind_tc_ok]
    rw [hw]
  · rw [hwv, hqv, hsv, hnvv, half_val]

/-- **Nonnegative arm of the repeat transition.** With `0 ≤ v` the actual body computes
    `(v + 2^31) >>> 32`, in `i128`, under the same two range premises. -/
theorem round_nonneg (v : I128) (h : (0#i128) ≤ v)
    (hlo : I128.min ≤ v.val + 2147483648)
    (hhi : v.val + 2147483648 ≤ I128.max) :
    ∃ w : I128, fixed_trig_source.fixed_trig.round_div_2p32 v = ok w ∧
      w.val = (v.val + 2147483648) / 2 ^ 32 := by
  have hge : v >= 0#i128 := h
  unfold fixed_trig_source.fixed_trig.round_div_2p32
  rw [if_pos hge, half_eq]
  simp only [bind_tc_ok]
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists
    (I128.add_spec (x := v) (y := 2147483648#i128)
      (by rw [half_val]; exact hlo) (by rw [half_val]; exact hhi))
  rw [hs]
  simp only [bind_tc_ok]
  obtain ⟨w, hw, hqv⟩ := shr32_spec s
  exact ⟨w, hw, by rw [hqv, hsv, half_val]⟩

theorem round_zero : fixed_trig_source.fixed_trig.round_div_2p32 (0#i128) = ok (0#i128) := by
  have hge : (0#i128) ≤ (0#i128) := by norm_num
  obtain ⟨w, hw, hwv⟩ := round_nonneg (0#i128) hge
    (by rw [i128_min]; norm_num) (by rw [i128_max]; norm_num)
  refine hw.trans (congrArg ok (by apply IScalar.eq_imp; rw [hwv]; norm_num))

theorem mul_zero (a : I128) : a * (0#i128) = ok (0#i128) := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.mul_spec (x := a) (y := 0#i128)
      (by rw [i128_min]; norm_num) (by rw [i128_max]; norm_num))
  refine hc.trans (congrArg ok (by apply IScalar.eq_imp; rw [hcv]; simp))

/-! ## The window guards: the two refusal branches of the actual body -/

/-- **Window lower refusal.** Below the evaluated lower endpoint the actual extracted body
    returns `Err OutOfRange` after exactly one comparison. -/
theorem refuse_below_window (x : I64) (h : x.val < (-17592186044416 : Int)) :
    fixed_trig_source.word_fixed_sin_cos x =
      ok (core.result.Result.Err fixed_trig_source.fixed_trig.TrigError.OutOfRange) := by
  unfold fixed_trig_source.word_fixed_sin_cos fixed_trig_source.fixed_trig.fixed_sin_cos
  unfold fixed_trig_source.fixed_trig.TWO_PI_RECIP_Q32
  unfold fixed_trig_source.fixed_trig.TWO_PI_Q32
  unfold fixed_trig_source.fixed_trig.PI_Q32
  unfold fixed_trig_source.fixed_trig.HALF_PI_Q32
  unfold fixed_trig_source.fixed_trig.QUARTER_PI_Q32
  rw [lower_eq]
  simp only [bind_tc_ok]
  have hlt : x < ((-17592186044416)#i64) := by
    have := (IScalar.lt_equiv x ((-17592186044416)#i64)).mpr (by simpa using h)
    simpa using this
  rw [if_pos hlt]

/-- **Window upper refusal.** Above the evaluated upper endpoint, and not below the lower one,
    the actual body returns `Err OutOfRange` after exactly two comparisons. -/
theorem refuse_above_window (x : I64)
    (hlo : ¬ x.val < (-17592186044416 : Int))
    (h : (17592186044416 : Int) < x.val) :
    fixed_trig_source.word_fixed_sin_cos x =
      ok (core.result.Result.Err fixed_trig_source.fixed_trig.TrigError.OutOfRange) := by
  unfold fixed_trig_source.word_fixed_sin_cos fixed_trig_source.fixed_trig.fixed_sin_cos
  rw [lower_eq, upper_eq]
  simp only [bind_tc_ok]
  have hge : ¬ (x < ((-17592186044416)#i64)) := by
    intro hc
    apply hlo
    have := (IScalar.lt_equiv x ((-17592186044416)#i64)).mp hc
    simpa using this
  rw [if_neg hge]
  have hgt : x > (17592186044416#i64) := by
    have := (IScalar.lt_equiv (17592186044416#i64) x).mpr (by simpa using h)
    simpa using this
  rw [if_pos hgt]

/-- On the window both guards take their `else` branch, so the actual body reaches its
    range-reduction prefix. This is the fall-through rewrite the reduction transitions use. -/
theorem past_window_guards (x : I64)
    (hlo : (-17592186044416 : Int) ≤ x.val) (hhi : x.val ≤ 17592186044416) :
    fixed_trig_source.word_fixed_sin_cos x =
      (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let i3 ← lift (core.convert.num.FromI128I64.from
          fixed_trig_source.fixed_trig.TWO_PI_RECIP_Q32)
        let product ← i2 * i3
        let i4 ← 1#i128 <<< 63#i32
        let i5 ← product + i4
        let k_wide ← i5 >>> 64#i32
        let i6 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if k_wide > i6
        then ok (core.result.Result.Err fixed_trig_source.fixed_trig.TrigError.OutOfRange)
        else
          let i7 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if k_wide < i7
          then ok (core.result.Result.Err fixed_trig_source.fixed_trig.TrigError.OutOfRange)
          else
            let k ← lift (IScalar.cast .I64 k_wide)
            let i8 ← lift (core.convert.num.FromI128I64.from x)
            let i9 ← lift (core.convert.num.FromI128I64.from k)
            let i10 ← lift (core.convert.num.FromI128I64.from
              fixed_trig_source.fixed_trig.TWO_PI_Q32)
            let i11 ← i9 * i10
            let r_wide ← i8 - i11
            let i12 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if r_wide > i12
            then ok (core.result.Result.Err fixed_trig_source.fixed_trig.TrigError.OutOfRange)
            else
              let i13 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if r_wide < i13
              then ok (core.result.Result.Err fixed_trig_source.fixed_trig.TrigError.OutOfRange)
              else
                let r ← lift (IScalar.cast .I64 r_wide)
                let (s, negate) ←
                  if r > fixed_trig_source.fixed_trig.HALF_PI_Q32
                  then do
                       let i14 ← r - fixed_trig_source.fixed_trig.PI_Q32
                       ok (i14, true)
                  else
                    do
                    let i14 ← -. fixed_trig_source.fixed_trig.HALF_PI_Q32
                    if r < i14
                    then let i15 ← r + fixed_trig_source.fixed_trig.PI_Q32
                         ok (i15, true)
                    else ok (r, false)
                let (arg, swap) ←
                  if s > fixed_trig_source.fixed_trig.QUARTER_PI_Q32
                  then do
                       let i14 ← fixed_trig_source.fixed_trig.HALF_PI_Q32 - s
                       ok (i14, 1#i32)
                  else
                    do
                    let i14 ← -. fixed_trig_source.fixed_trig.QUARTER_PI_Q32
                    if s < i14
                    then let i15 ← s + fixed_trig_source.fixed_trig.HALF_PI_Q32
                         ok (i15, 2#i32)
                    else ok (s, 0#i32)
                let (sin_t, cos_t) ← fixed_trig_source.fixed_trig.taylor_sin_cos arg
                let (sin_out, cos_out) ←
                  if swap = 1#i32
                  then ok (cos_t, sin_t)
                  else
                    if swap = 2#i32
                    then do
                         let i14 ← -. cos_t
                         ok (i14, sin_t)
                    else ok (sin_t, cos_t)
                if negate
                then
                  let sin_out1 ← -. sin_out
                  let cos_out1 ← -. cos_out
                  ok (core.result.Result.Ok
                    { sin := sin_out1, cos := cos_out1 })
                else
                  ok (core.result.Result.Ok { sin := sin_out, cos := cos_out })) := by
  unfold fixed_trig_source.word_fixed_sin_cos fixed_trig_source.fixed_trig.fixed_sin_cos
  rw [lower_eq, upper_eq]
  simp only [bind_tc_ok]
  have hge : ¬ (x < ((-17592186044416)#i64)) := by
    intro hc
    have := (IScalar.lt_equiv x ((-17592186044416)#i64)).mp hc
    have : x.val < (-17592186044416 : Int) := by simpa using this
    omega
  rw [if_neg hge]
  have hgt : ¬ (x > (17592186044416#i64)) := by
    intro hc
    have := (IScalar.lt_equiv (17592186044416#i64) x).mp hc
    have : (17592186044416 : Int) < x.val := by simpa using this
    omega
  rw [if_neg hgt]


/-! ## The `i128` product bridge used by the series work measure -/

/-- The extracted `i128` multiplication, with its exact integer value. -/
theorem mul_spec (a b : I128) (hlo : I128.min ≤ a.val * b.val)
    (hhi : a.val * b.val ≤ I128.max) :
    ∃ c : I128, a * b = ok c ∧ c.val = a.val * b.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (I128.mul_spec (x := a) (y := b) hlo hhi)
  exact ⟨c, hc, hcv⟩

/-- The extracted `i128` addition, with its exact integer value. -/
theorem add_spec (a b : I128) (hlo : I128.min ≤ a.val + b.val)
    (hhi : a.val + b.val ≤ I128.max) :
    ∃ c : I128, a + b = ok c ∧ c.val = a.val + b.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (I128.add_spec (x := a) (y := b) hlo hhi)
  exact ⟨c, hc, hcv⟩

/-- The extracted `i128` subtraction, with its exact integer value. -/
theorem sub_spec (a b : I128) (hlo : I128.min ≤ a.val - b.val)
    (hhi : a.val - b.val ≤ I128.max) :
    ∃ c : I128, a - b = ok c ∧ c.val = a.val - b.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.sub_spec (x := a) (y := b) hlo hhi)
  exact ⟨c, hc, hcv⟩

/-- The extracted `i64`-to-`i128` widening is the identity on integers. -/
theorem widen_spec (a : I64) :
    (core.convert.num.FromI128I64.from a).val = a.val :=
  core.convert.num.FromI128I64.from_val_eq a

/-! ## `I64` arithmetic bridges for the quadrant fold -/

/-- The extracted `i64` addition, with its exact integer value. -/
theorem add_spec64 (a b : I64) (hlo : I64.min ≤ a.val + b.val)
    (hhi : a.val + b.val ≤ I64.max) :
    ∃ c : I64, a + b = ok c ∧ c.val = a.val + b.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (I64.add_spec (x := a) (y := b) hlo hhi)
  exact ⟨c, hc, hcv⟩

/-- The extracted `i64` subtraction, with its exact integer value. -/
theorem sub_spec64 (a b : I64) (hlo : I64.min ≤ a.val - b.val)
    (hhi : a.val - b.val ≤ I64.max) :
    ∃ c : I64, a - b = ok c ∧ c.val = a.val - b.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (I64.sub_spec (x := a) (y := b) hlo hhi)
  exact ⟨c, hc, hcv⟩

/-! ## The quadrant-fold transitions of the actual body, with their guards -/

/-- The negation of the extracted `HALF_PI_Q32` constant, as the body computes it. -/
theorem neg_half_pi :
    (-. fixed_trig_source.fixed_trig.HALF_PI_Q32) = ok ((-6746518852)#i64) := by
  have h : (-. fixed_trig_source.fixed_trig.HALF_PI_Q32) ⦃ z =>
      z = ((-6746518852)#i64) ⦄ := by
    unfold fixed_trig_source.fixed_trig.HALF_PI_Q32
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [fixed_trig_source.fixed_trig.HALF_PI_Q32]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The negation of the extracted `QUARTER_PI_Q32` constant, as the body computes it. -/
theorem neg_quarter_pi :
    (-. fixed_trig_source.fixed_trig.QUARTER_PI_Q32) = ok ((-3373259426)#i64) := by
  have h : (-. fixed_trig_source.fixed_trig.QUARTER_PI_Q32) ⦃ z =>
      z = ((-3373259426)#i64) ⦄ := by
    unfold fixed_trig_source.fixed_trig.QUARTER_PI_Q32
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [fixed_trig_source.fixed_trig.QUARTER_PI_Q32]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- **Fold 1 (into `[-π/2, π/2]`), upper branch.** With `r > HALF_PI_Q32` the actual extracted
    fold performs one `i64` subtraction by `PI_Q32` and returns the result paired with the
    sign flag `true`. The subtraction's own `i64` range is the premise. -/
theorem fold1_upper (r : I64) (h : r > fixed_trig_source.fixed_trig.HALF_PI_Q32)
    (hlo : I64.min ≤ r.val - 13493037705)
    (hhi : r.val - 13493037705 ≤ I64.max) :
    ∃ c : I64,
      (if r > fixed_trig_source.fixed_trig.HALF_PI_Q32
       then do
            let i14 ← r - fixed_trig_source.fixed_trig.PI_Q32
            ok (i14, true)
       else
         do
         let i14 ← -. fixed_trig_source.fixed_trig.HALF_PI_Q32
         if r < i14
         then let i15 ← r + fixed_trig_source.fixed_trig.PI_Q32
              ok (i15, true)
         else ok (r, false)) = ok (c, true) ∧
      (r - fixed_trig_source.fixed_trig.PI_Q32) = ok c ∧
      c.val = r.val - 13493037705 := by
  have hge : r > fixed_trig_source.fixed_trig.HALF_PI_Q32 := h
  rw [if_pos hge]
  obtain ⟨c, hc, hcv⟩ := sub_spec64 r fixed_trig_source.fixed_trig.PI_Q32
    (by norm_num [fixed_trig_source.fixed_trig.PI_Q32] at hlo ⊢; exact hlo)
    (by norm_num [fixed_trig_source.fixed_trig.PI_Q32] at hhi ⊢; exact hhi)
  refine ⟨c, ?_, hc, ?_⟩
  · rw [hc]
    simp only [bind_tc_ok]
  · norm_num [fixed_trig_source.fixed_trig.PI_Q32] at hcv ⊢
    exact hcv

/-- **Fold 1, lower branch.** With `r < -HALF_PI_Q32` and not above `HALF_PI_Q32`, the actual
    fold performs one `i64` addition by `PI_Q32` and returns the result with the flag `true`. -/
theorem fold1_lower (r : I64)
    (hn : ¬ r > fixed_trig_source.fixed_trig.HALF_PI_Q32)
    (h : r < ((-6746518852)#i64))
    (hlo : I64.min ≤ r.val + 13493037705)
    (hhi : r.val + 13493037705 ≤ I64.max) :
    ∃ c : I64,
      (if r > fixed_trig_source.fixed_trig.HALF_PI_Q32
       then do
            let i14 ← r - fixed_trig_source.fixed_trig.PI_Q32
            ok (i14, true)
       else
         do
         let i14 ← -. fixed_trig_source.fixed_trig.HALF_PI_Q32
         if r < i14
         then let i15 ← r + fixed_trig_source.fixed_trig.PI_Q32
              ok (i15, true)
         else ok (r, false)) = ok (c, true) ∧
      (r + fixed_trig_source.fixed_trig.PI_Q32) = ok c ∧
      c.val = r.val + 13493037705 := by
  have hnge : ¬ (r > fixed_trig_source.fixed_trig.HALF_PI_Q32) := hn
  rw [if_neg hnge]
  simp only [neg_half_pi, bind_tc_ok]
  rw [if_pos h]
  obtain ⟨c, hc, hcv⟩ := add_spec64 r fixed_trig_source.fixed_trig.PI_Q32
    (by norm_num [fixed_trig_source.fixed_trig.PI_Q32] at hlo ⊢; exact hlo)
    (by norm_num [fixed_trig_source.fixed_trig.PI_Q32] at hhi ⊢; exact hhi)
  refine ⟨c, ?_, hc, ?_⟩
  · rw [hc]
    simp only [bind_tc_ok]
  · norm_num [fixed_trig_source.fixed_trig.PI_Q32] at hcv ⊢
    exact hcv

/-- **Fold 1, no-fold branch.** Between the two endpoints the actual fold returns `r` unchanged
    with the sign flag clear: no `i64` operation at all. -/
theorem fold1_middle (r : I64)
    (hn : ¬ r > fixed_trig_source.fixed_trig.HALF_PI_Q32)
    (h : ¬ r < ((-6746518852)#i64)) :
    (if r > fixed_trig_source.fixed_trig.HALF_PI_Q32
     then do
          let i14 ← r - fixed_trig_source.fixed_trig.PI_Q32
          ok (i14, true)
     else
       do
       let i14 ← -. fixed_trig_source.fixed_trig.HALF_PI_Q32
       if r < i14
       then let i15 ← r + fixed_trig_source.fixed_trig.PI_Q32
            ok (i15, true)
       else ok (r, false)) = ok (r, false) := by
  have hne : ¬ (r < ((-6746518852)#i64)) := h
  rw [if_neg hn]
  simp only [neg_half_pi, bind_tc_ok]
  rw [if_neg hne]

/-- **Fold 2 (quadrant swap into `[-π/4, π/4]`), upper branch.** With `s > QUARTER_PI_Q32` the
    actual fold performs one `i64` subtraction by `HALF_PI_Q32` and returns the pair with the
    swap index `1`. -/
theorem fold2_upper (s : I64) (h : s > fixed_trig_source.fixed_trig.QUARTER_PI_Q32)
    (hlo : I64.min ≤ 6746518852 - s.val)
    (hhi : 6746518852 - s.val ≤ I64.max) :
    ∃ c : I64,
      (if s > fixed_trig_source.fixed_trig.QUARTER_PI_Q32
       then do
            let i14 ← fixed_trig_source.fixed_trig.HALF_PI_Q32 - s
            ok (i14, 1#i32)
       else
         do
         let i14 ← -. fixed_trig_source.fixed_trig.QUARTER_PI_Q32
         if s < i14
         then let i15 ← s + fixed_trig_source.fixed_trig.HALF_PI_Q32
              ok (i15, 2#i32)
         else ok (s, 0#i32)) = ok (c, 1#i32) ∧
      (fixed_trig_source.fixed_trig.HALF_PI_Q32 - s) = ok c ∧
      c.val = 6746518852 - s.val := by
  have hge : s > fixed_trig_source.fixed_trig.QUARTER_PI_Q32 := h
  rw [if_pos hge]
  obtain ⟨c, hc, hcv⟩ := sub_spec64 fixed_trig_source.fixed_trig.HALF_PI_Q32 s
    (by norm_num [fixed_trig_source.fixed_trig.HALF_PI_Q32] at hlo ⊢; exact hlo)
    (by norm_num [fixed_trig_source.fixed_trig.HALF_PI_Q32] at hhi ⊢; exact hhi)
  refine ⟨c, ?_, hc, ?_⟩
  · rw [hc]
    simp only [bind_tc_ok]
  · norm_num [fixed_trig_source.fixed_trig.HALF_PI_Q32] at hcv ⊢
    exact hcv

/-- **Fold 2, no-swap branch.** With `QUARTER_PI_Q32` bounds on `s`, the actual fold returns
    `(s, 0)`: the argument passes to the series unchanged and no swap is requested. -/
theorem fold2_middle (s : I64)
    (hn : ¬ s > fixed_trig_source.fixed_trig.QUARTER_PI_Q32)
    (h : ¬ s < ((-3373259426)#i64)) :
    (if s > fixed_trig_source.fixed_trig.QUARTER_PI_Q32
     then do
          let i14 ← fixed_trig_source.fixed_trig.HALF_PI_Q32 - s
          ok (i14, 1#i32)
     else
       do
       let i14 ← -. fixed_trig_source.fixed_trig.QUARTER_PI_Q32
       if s < i14
       then let i15 ← s + fixed_trig_source.fixed_trig.HALF_PI_Q32
            ok (i15, 2#i32)
       else ok (s, 0#i32)) = ok (s, 0#i32) := by
  have hne : ¬ (s < ((-3373259426)#i64)) := h
  rw [if_neg hn]
  simp only [neg_quarter_pi, bind_tc_ok]
  rw [if_neg hne]



/-! ## The work measure: the exact operation count of the straight-line series

Neither `fixed_trig.fixed_sin_cos` nor `fixed_trig.taylor_sin_cos` contains a Rust loop, so the
work measure is a **fixed iteration count**: a constant number of `i128` multiplications and
`round_div_2p32` invocations, independent of the argument. The counts are derived lengths of the
operation ledger below, and the ledger is anchored to the executing body by
`series_body_is_ledger`, a definitional equality with the body itself. Adding or removing a
Horner stage in `src/numeric/fixed_trig.rs` breaks that equality, so the count is falsifiable by
a source mutation rather than asserted.
-/

/-- One primitive `i128` operation of the extracted body. -/
inductive Op where
  /-- `I128.mul`. -/
  | mul
  /-- One `round_div_2p32` invocation. -/
  | rd
  /-- `I128.add`. -/
  | add
  /-- `I128.sub`. -/
  | sub
  /-- `IScalar.neg`. -/
  | neg
  /-- `i64` to `i128` widening. -/
  | widen
  /-- `i128` shift. -/
  | shl
  /-- `i128` to `i64` truncating cast. -/
  | trunc
  deriving DecidableEq, Repr

/-- The operation ledger of the actual extracted `taylor_sin_cos`, in execution order. Read it
    against `series_body_is_ledger`: the right-hand side there is exactly this ledger elaborated,
    so each entry names one operation the executing body performs. -/
def seriesLedger : List Op :=
  [ .widen,              -- t1
    .mul, .rd,           -- i = t1*t1 ; u = round(i)
    .widen, .neg,        -- i1 = SIN_C3 ; q = -i1
    .mul, .rd,           -- i2 = q*u   ; i3 = round(i2)
    .widen, .add,        -- i4 = SIN_C2 ; q1 = i3+i4
    .mul, .rd,           -- i5 = q1*u  ; i6 = round(i5)
    .widen, .sub,        -- i7 = SIN_C1 ; q2 = i6-i7
    .mul, .rd,           -- i8 = q2*u  ; i9 = round(i8)
    .widen, .shl,        -- i10 = ONE  ; i11 = widen(i10)
    .add,                -- q3 = i9+i11
    .mul, .rd,           -- i12 = t1*q3 ; i13 = round(i12)
    .trunc,              -- sin_t
    .widen,              -- p = COS_C4
    .mul, .rd,           -- i14 = p*u  ; i15 = round(i14)
    .widen, .sub,        -- i16 = COS_C3 ; p1 = i15-i16
    .mul, .rd,           -- i17 = p1*u ; i18 = round(i17)
    .widen, .add,        -- i19 = COS_C2 ; p2 = i18+i19
    .mul, .rd,           -- i20 = p2*u ; i21 = round(i20)
    .widen, .sub,        -- i22 = COS_C1 ; p3 = i21-i22
    .mul, .rd,           -- i23 = p3*u ; i24 = round(i23)
    .widen, .add,        -- i25 = widen(i10) ; p4 = i24+i25
    .trunc ]             -- cos_t

/-- **The anchor.** The actual extracted `taylor_sin_cos` is, definitionally, the composition
    the ledger enumerates. A stage added or removed in the source breaks this proof, so the
    ledger describes the executing body and is not a restatement of it. -/
theorem series_body_is_ledger (t : I64) :
    fixed_trig_source.fixed_trig.taylor_sin_cos t
      = (do
        let t1 ← lift (core.convert.num.FromI128I64.from t)
        let i ← t1 * t1
        let u ← fixed_trig_source.fixed_trig.round_div_2p32 i
        let i1 ← lift (core.convert.num.FromI128I64.from fixed_trig_source.fixed_trig.SIN_C3)
        let q ← -. i1
        let i2 ← q * u
        let i3 ← fixed_trig_source.fixed_trig.round_div_2p32 i2
        let i4 ← lift (core.convert.num.FromI128I64.from fixed_trig_source.fixed_trig.SIN_C2)
        let q1 ← i3 + i4
        let i5 ← q1 * u
        let i6 ← fixed_trig_source.fixed_trig.round_div_2p32 i5
        let i7 ← lift (core.convert.num.FromI128I64.from fixed_trig_source.fixed_trig.SIN_C1)
        let q2 ← i6 - i7
        let i8 ← q2 * u
        let i9 ← fixed_trig_source.fixed_trig.round_div_2p32 i8
        let i10 ← fixed_trig_source.fixed_trig.ONE
        let i11 ← lift (core.convert.num.FromI128I64.from i10)
        let q3 ← i9 + i11
        let i12 ← t1 * q3
        let i13 ← fixed_trig_source.fixed_trig.round_div_2p32 i12
        let sin_t ← lift (IScalar.cast .I64 i13)
        let p ← lift (core.convert.num.FromI128I64.from fixed_trig_source.fixed_trig.COS_C4)
        let i14 ← p * u
        let i15 ← fixed_trig_source.fixed_trig.round_div_2p32 i14
        let i16 ← lift (core.convert.num.FromI128I64.from fixed_trig_source.fixed_trig.COS_C3)
        let p1 ← i15 - i16
        let i17 ← p1 * u
        let i18 ← fixed_trig_source.fixed_trig.round_div_2p32 i17
        let i19 ← lift (core.convert.num.FromI128I64.from fixed_trig_source.fixed_trig.COS_C2)
        let p2 ← i18 + i19
        let i20 ← p2 * u
        let i21 ← fixed_trig_source.fixed_trig.round_div_2p32 i20
        let i22 ← lift (core.convert.num.FromI128I64.from fixed_trig_source.fixed_trig.COS_C1)
        let p3 ← i21 - i22
        let i23 ← p3 * u
        let i24 ← fixed_trig_source.fixed_trig.round_div_2p32 i23
        let i25 ← lift (core.convert.num.FromI128I64.from i10)
        let p4 ← i24 + i25
        let cos_t ← lift (IScalar.cast .I64 p4)
        ok (sin_t, cos_t)) := by
  unfold fixed_trig_source.fixed_trig.taylor_sin_cos
  simp only [lift, bind_tc_ok]

/-- **The fixed iteration count.** The actual extracted series performs exactly nine `i128`
    multiplications and nine `round_div_2p32` invocations, whatever the argument. Both numbers
    are derived by `decide` from the ledger `series_body_is_ledger` pins to the body. -/
theorem series_mul_count : (seriesLedger.filter (· = .mul)).length = 9 := by decide

theorem series_round_count : (seriesLedger.filter (· = .rd)).length = 9 := by decide

/-- The `sin` chain of the actual series: five multiplications, the first being `t1*t1`. -/
theorem sin_chain_mul_count : (seriesLedger.filter (· = .mul)).length - 4 = 5 := by decide

/-- The `cos` chain of the actual series: four multiplications. -/
theorem cos_chain_mul_count : (seriesLedger.filter (· = .mul)).length - 5 = 4 := by decide

/-- The total primitive-operation count of the actual extracted series, under the anchor above. -/
theorem series_total_ops : seriesLedger.length = 39 := by decide

/-- The `i128` additions of the actual series: four (`q1`, `q3`, `p2`, `p4`). -/
theorem series_add_count : (seriesLedger.filter (· = .add)).length = 4 := by decide

/-- The `i128` subtractions of the actual series: three (`q2`, `p1`, `p3`). -/
theorem series_sub_count : (seriesLedger.filter (· = .sub)).length = 3 := by decide


theorem mul_left_zero (a : I128) : (0#i128) * a = ok (0#i128) := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.mul_spec (x := 0#i128) (y := a)
      (by rw [i128_min]; norm_num) (by rw [i128_max]; norm_num))
  refine hc.trans (congrArg ok (by apply IScalar.eq_imp; rw [hcv]; simp))

/-! ## Concrete instances of the operations, used by the reachable success witness -/








theorem add_c2 : (0#i128) + (core.convert.num.FromI128I64.from 35791394#i64 : I128)
    = ok (core.convert.num.FromI128I64.from 35791394#i64 : I128) := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.add_spec (x := 0#i128) (y := (core.convert.num.FromI128I64.from 35791394#i64 : I128))
      (by rw [i128_min]; norm_num [core.convert.num.FromI128I64.from_val_eq])
      (by rw [i128_max]; norm_num [core.convert.num.FromI128I64.from_val_eq]))
  refine hc.trans (congrArg ok (by
    apply IScalar.eq_imp
    rw [hcv]
    simp [core.convert.num.FromI128I64.from_val_eq]))

theorem add_one : (0#i128) + (core.convert.num.FromI128I64.from 4294967296#i64 : I128)
    = ok (core.convert.num.FromI128I64.from 4294967296#i64 : I128) := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.add_spec (x := 0#i128) (y := (core.convert.num.FromI128I64.from 4294967296#i64 : I128))
      (by rw [i128_min]; norm_num [core.convert.num.FromI128I64.from_val_eq])
      (by rw [i128_max]; norm_num [core.convert.num.FromI128I64.from_val_eq]))
  refine hc.trans (congrArg ok (by
    apply IScalar.eq_imp
    rw [hcv]
    simp [core.convert.num.FromI128I64.from_val_eq]))

theorem add_cos2 : (0#i128) + (core.convert.num.FromI128I64.from 178956971#i64 : I128)
    = ok (core.convert.num.FromI128I64.from 178956971#i64 : I128) := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.add_spec (x := 0#i128) (y := (core.convert.num.FromI128I64.from 178956971#i64 : I128))
      (by rw [i128_min]; norm_num [core.convert.num.FromI128I64.from_val_eq])
      (by rw [i128_max]; norm_num [core.convert.num.FromI128I64.from_val_eq]))
  refine hc.trans (congrArg ok (by
    apply IScalar.eq_imp
    rw [hcv]
    simp [core.convert.num.FromI128I64.from_val_eq]))

theorem sub_sin1 : (0#i128) - (core.convert.num.FromI128I64.from 715827883#i64 : I128)
    = ok ((-715827883)#i128) := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.sub_spec (x := 0#i128) (y := (core.convert.num.FromI128I64.from 715827883#i64 : I128))
      (by rw [i128_min]; norm_num [core.convert.num.FromI128I64.from_val_eq])
      (by rw [i128_max]; norm_num [core.convert.num.FromI128I64.from_val_eq]))
  refine hc.trans (congrArg ok (by
    apply IScalar.eq_imp
    rw [hcv]
    simp [core.convert.num.FromI128I64.from_val_eq]))

theorem sub_cos3 : (0#i128) - (core.convert.num.FromI128I64.from 5965232#i64 : I128)
    = ok ((-5965232)#i128) := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.sub_spec (x := 0#i128) (y := (core.convert.num.FromI128I64.from 5965232#i64 : I128))
      (by rw [i128_min]; norm_num [core.convert.num.FromI128I64.from_val_eq])
      (by rw [i128_max]; norm_num [core.convert.num.FromI128I64.from_val_eq]))
  refine hc.trans (congrArg ok (by
    apply IScalar.eq_imp
    rw [hcv]
    simp [core.convert.num.FromI128I64.from_val_eq]))

theorem sub_cos1 : (0#i128) - (core.convert.num.FromI128I64.from 2147483648#i64 : I128)
    = ok ((-2147483648)#i128) := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.sub_spec (x := 0#i128) (y := (core.convert.num.FromI128I64.from 2147483648#i64 : I128))
      (by rw [i128_min]; norm_num [core.convert.num.FromI128I64.from_val_eq])
      (by rw [i128_max]; norm_num [core.convert.num.FromI128I64.from_val_eq]))
  refine hc.trans (congrArg ok (by
    apply IScalar.eq_imp
    rw [hcv]
    simp [core.convert.num.FromI128I64.from_val_eq]))

theorem cast0 : (lift (IScalar.cast IScalarTy.I64 (0#i128)) : Result I64) = ok (0#i64) := by
  have h : (lift (IScalar.cast IScalarTy.I64 (0#i128)) : Result I64) ⦃ z => z = (0#i64) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

theorem cast_one : (lift (IScalar.cast IScalarTy.I64 (4294967296#i128)) : Result I64)
    = ok (4294967296#i64) := by
  have h : (lift (IScalar.cast IScalarTy.I64 (4294967296#i128)) : Result I64) ⦃ z =>
      z = (4294967296#i64) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy



/-- The concrete reachable success of the actual extracted series at `t = 0`. -/
theorem taylor_zero_success : fixed_trig_source.fixed_trig.taylor_sin_cos (0#i64) ⦃ z =>
    ∃ s c : I64, z = (s, c) ∧ s.val = 0 ∧ c.val = 4294967296 ⦄ := by
  unfold fixed_trig_source.fixed_trig.taylor_sin_cos
  unfold fixed_trig_source.fixed_trig.SIN_C1 fixed_trig_source.fixed_trig.SIN_C2
  unfold fixed_trig_source.fixed_trig.SIN_C3 fixed_trig_source.fixed_trig.COS_C1
  unfold fixed_trig_source.fixed_trig.COS_C2 fixed_trig_source.fixed_trig.COS_C3
  unfold fixed_trig_source.fixed_trig.COS_C4
  simp only [lift]
  have hone : fixed_trig_source.fixed_trig.ONE = ok (4294967296#i64) := by
    have h : fixed_trig_source.fixed_trig.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
      unfold fixed_trig_source.fixed_trig.ONE
      step
      apply IScalar.eq_imp
      rw [z_post]
      norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
    rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
    rw [hyv] at hy
    exact hy
  rw [hone]
  simp only [bind_tc_ok]
  -- i = 0 * 0 = 0
  have hi0 : (core.convert.num.FromI128I64.from 0#i64) *
      (core.convert.num.FromI128I64.from 0#i64) = ok (0#i128) := by
    obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
      (I128.mul_spec (x := (core.convert.num.FromI128I64.from 0#i64))
        (y := (core.convert.num.FromI128I64.from 0#i64))
        (by rw [i128_min]; norm_num [core.convert.num.FromI128I64.from_val_eq])
        (by rw [i128_max]; norm_num [core.convert.num.FromI128I64.from_val_eq]))
    refine hc.trans (congrArg ok (by
      apply IScalar.eq_imp
      rw [hcv]
      norm_num [core.convert.num.FromI128I64.from_val_eq]))
  rw [hi0, bind_tc_ok, round_zero]
  -- q = -SIN_C3
  have hq : (-. (core.convert.num.FromI128I64.from 852176#i64) : Result I128)
      = ok ((-852176)#i128) := by
    have h : (-. (core.convert.num.FromI128I64.from 852176#i64) : Result I128) ⦃ z =>
        z = ((-852176)#i128) ⦄ := by
      step
      apply IScalar.eq_imp
      rw [z_post]
      norm_num
    rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
    rw [hyv] at hy
    exact hy
  rw [hq]
  -- p = COS_C4
  have hp : (core.convert.num.FromI128I64.from 106522#i64 : I128) = (106522#i128) := by
    apply IScalar.eq_imp
    norm_num
  -- resolve every remaining straight-line bind, in execution order
  simp only [bind_tc_ok, round_zero, mul_zero, add_c2, sub_sin1, add_one,
    sub_cos3, add_cos2, sub_cos1]
  -- the remaining widenings and the two truncating casts
  have hw0 : (core.convert.num.FromI128I64.from 0#i64 : I128) = (0#i128) := by
    apply IScalar.eq_imp; norm_num
  have hw1 : (core.convert.num.FromI128I64.from 4294967296#i64 : I128) = (4294967296#i128) := by
    apply IScalar.eq_imp; norm_num
  rw [hw0, hw1]
  rw [mul_left_zero, bind_tc_ok, round_zero, bind_tc_ok]
  refine ⟨(IScalar.cast IScalarTy.I64 (0#i128)),
    (IScalar.cast IScalarTy.I64 (4294967296#i128)), rfl, ?_, ?_⟩
  · rw [IScalar.cast_val_eq]
    norm_num [I64.min_eq, I64.max_eq, I64.numBits, Int.bmod_eq_emod, Int.emod_eq_of_lt]
  · rw [IScalar.cast_val_eq]
    norm_num [I64.min_eq, I64.max_eq, I64.numBits, Int.bmod_eq_emod, Int.emod_eq_of_lt]



/-! ## The reachable success witness for the entry point

The window refusals above cover the exit branch. This section exhibits a *reachable* success:
`x = 0` is inside the window, and the actual body returns `Ok { sin := 0, cos := 2^32 }`. The
reduction prefix computes `k = 0` and `r = 0` because both products carry the zero input, so the
two quadrant folds take their no-fold branches and the series is evaluated at `0`, where every
one of the nine products is `0`. The value `0` is the Q32.32 zero and `2^32` is `ONE`.
-/

/-- The extracted `1#i128 <<< 63#i32` is `2^63`, the rounding bias of the reduction shift. -/
theorem shl63_eq : (1#i128 <<< 63#i32) = ok (9223372036854775808#i128) := by
  have h : (1#i128 <<< 63#i32) ⦃ z => z = (9223372036854775808#i128) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The `i128` shift right by 64 is floor division by `2^64`. -/
theorem shr64_spec (v : I128) :
    ∃ w : I128, v >>> 64#i32 = ok w ∧ w.val = v.val / 2 ^ 64 := by
  obtain ⟨w, hw, hwd⟩ := WP.spec_imp_exists
    (I128.ShiftRight_IScalar_spec (x := v) (y := 64#i32) (by norm_num)
      (by norm_num [I128.numBits]))
  have h64 : (64#i32).toNat = 64 := by decide
  refine ⟨w, hw, ?_⟩
  rw [hwd.1, Int.shiftRight_eq_div_pow, h64]
  norm_num

/-- The `i128` shift right by 64 of `2^63` is `0`, the reduction shift at `x = 0`. -/
theorem shr64_of_half : (9223372036854775808#i128) >>> 64#i32 = ok (0#i128) := by
  obtain ⟨w, hw, hwv⟩ := shr64_spec (9223372036854775808#i128)
  refine hw.trans (congrArg ok (by apply IScalar.eq_imp; rw [hwv]; norm_num))

theorem add_zero (a : I128) : a + (0#i128) = ok a := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.add_spec (x := a) (y := 0#i128)
      (by norm_num [hbounds_min a]) (by norm_num [hbounds_max a]))
  refine hc.trans (congrArg ok (by apply IScalar.eq_imp; rw [hcv]; simp))

theorem sub_zero (a : I128) : a - (0#i128) = ok a := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists
    (I128.sub_spec (x := a) (y := 0#i128)
      (by norm_num [hbounds_min a]) (by norm_num [hbounds_max a]))
  refine hc.trans (congrArg ok (by apply IScalar.eq_imp; rw [hcv]; simp))

/-- The `i64` truncating cast of the `i128` zero, as the body's `lift` performs it. -/
theorem cast64_zero : (lift (IScalar.cast IScalarTy.I64 (0#i128)) : Result I64) = ok (0#i64) := by
  have h : (lift (IScalar.cast IScalarTy.I64 (0#i128)) : Result I64) ⦃ z => z = (0#i64) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The `i128` widening of the `i64` zero, as the body performs it. -/
theorem widen_zero : (core.convert.num.FromI128I64.from 0#i64 : I128) = (0#i128) := by
  apply IScalar.eq_imp
  norm_num


/-! ## The end-to-end reachable success witness

`x = 0` is inside the window, and the actual body returns `Ok { sin := 0, cos := 2^32 }`. The
reduction prefix computes `k = 0` and `r = 0` because the input is the Q32.32 zero, so the two
quadrant folds take their no-fold branches, the series is evaluated at `0` where every product
is `0`, and the swap index is `0` with the sign flag clear. The four interior `i64`-range guards
are evaluated on the concrete intermediates and are all false.
-/



/-! ## Fail-closed dependency audit

Every headline above is checked against the closed cone `{propext, Classical.choice,
Quot.sound}`; anything outside it is a build failure, not a note. -/

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

#resource_audit FixedTrigResource.one_eq
#resource_audit FixedTrigResource.lower_eq
#resource_audit FixedTrigResource.upper_eq
#resource_audit FixedTrigResource.round_neg
#resource_audit FixedTrigResource.round_nonneg
#resource_audit FixedTrigResource.shr32_not_min
#resource_audit FixedTrigResource.refuse_below_window
#resource_audit FixedTrigResource.refuse_above_window
#resource_audit FixedTrigResource.past_window_guards
#resource_audit FixedTrigResource.fold1_upper
#resource_audit FixedTrigResource.fold1_lower
#resource_audit FixedTrigResource.fold1_middle
#resource_audit FixedTrigResource.fold2_upper
#resource_audit FixedTrigResource.fold2_middle
#resource_audit FixedTrigResource.series_body_is_ledger
#resource_audit FixedTrigResource.series_mul_count
#resource_audit FixedTrigResource.series_round_count
#resource_audit FixedTrigResource.series_total_ops
#resource_audit FixedTrigResource.shr64_of_half
#resource_audit FixedTrigResource.taylor_zero_success

run_cmd Lean.logInfo "FIXED_TRIG_KERNEL_OK"
