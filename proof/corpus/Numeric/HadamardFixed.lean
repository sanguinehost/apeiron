/-
Normalized exact-integer Walsh–Hadamard refinement (spec
the ternary-hadamard note, contract "normalized FWHT").

The Rust body is `src/engine/hadamard.rs`; the exact-integer core (`fwht_int_spec`,
`fwht_int_sylvester`) is in `Numeric.HadamardRefinement`. This module lands:

* `round_q32_signed_spec`, the checked sign-magnitude `>> 32` rounding law of
  `hadamard.round_q32_signed`;
* `hadamard_C_val`, `C_val`, `C_even` and `C_close`, the `C` table entry values and their
  half-ulp distance from `2^32/√(2^k)`;
* the validation-refusal equalities of `hadamard.fwht_fixed` (NotPowerOfTwo, both
  LengthMismatch kinds, BadSign), each leaving `x` untouched.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom` or `decide`
over an extracted body appears.
-/
import Numeric.HadamardRefinement
import Numeric.Scale
import Numeric.EngineNormRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Item 1 · the checked sign-magnitude rounding

`hadamard.round_q32_signed` (`src/engine/hadamard.rs:83`) rounds `num / 2^32` to nearest,
ties away from zero: it rounds `|num|` with `value.div_round_nearest` and reapplies the sign.
The law is `div_round_nearest_abs_bound` (`Numeric.EngineSiluRefinement`) on the magnitude. -/

/-- `I128.min` as an integer. -/
theorem i128_min_eq : I128.min = -(2 : ℤ) ^ 127 := by
  norm_num [I128.min, IScalar.min, I128.numBits]

/-- `I128.max` as an integer. -/
theorem i128_max_eq : I128.max = (2 : ℤ) ^ 127 - 1 := by
  norm_num [I128.max, IScalar.max, I128.numBits]

/-- **Positive half-biased quotient.** For `n ≥ 0` with `n + 2^31` still inside `I128`,
`value.div_round_nearest n 2^32` is the exact half-biased quotient `(n + 2^31)/2^32`. -/
theorem div_round_nearest_q32_nonneg (n : I128) (hnn : 0 ≤ n.val)
    (hub : n.val + 2 ^ 31 ≤ (2 : ℤ) ^ 127 - 1) :
    value.div_round_nearest n (4294967296#i128) ⦃ q =>
      q.val = divRoundNearest n.val ((2 : ℤ) ^ 32) ∧ 0 ≤ q.val ⦄ := by
  have hden : (4294967296#i128).val = (2 : ℤ) ^ 32 := by decide
  have h2 : (2#i128).val = (2 : ℤ) := by decide
  unfold value.div_round_nearest
  refine WP.spec_bind (I128.div_spec (x := 4294967296#i128) (y := 2#i128)
    (by norm_num) (by intro h; exact absurd h.2 (by norm_num))) ?_
  intro i hiv
  have hival : i.val = (2 : ℤ) ^ 31 := by
    rw [hiv, hden, h2]
    norm_num [Int.tdiv_eq_ediv]
  refine WP.spec_bind (I128.add_spec (x := n) (y := i)
    (by rw [i128_min_eq, hival]; omega)
    (by rw [i128_max_eq, hival]; omega)) ?_
  intro i1 hi1v
  refine WP.spec_mono (I128.div_spec (x := i1) (y := 4294967296#i128)
    (by rw [hden]; norm_num)
    (by intro h; rw [hden] at h; norm_num at h)) ?_
  intro q hqv
  have hqval : q.val = divRoundNearest n.val ((2 : ℤ) ^ 32) := by
    rw [hqv, hi1v, hival, hden, divRoundNearest]
    have h1 : (0 : ℤ) ≤ n.val + 2 ^ 31 := by omega
    rw [Int.tdiv_eq_ediv_of_nonneg h1]
    norm_num
  refine ⟨hqval, ?_⟩
  rw [hqval, divRoundNearest]
  exact Int.ediv_nonneg (by omega) (by norm_num)

/-- **Item 1 · the checked rounding law.** For `|num| < 2^126`, `hadamard.round_q32_signed num`
returns `r` with `r.val = sign(num) · divRoundNearest |num| 2^32`, and the rounding error scaled
back by `2^32` is at most `2^31` — half a Q32.32 unit. -/
theorem round_q32_signed_spec (num : I128) (hnum : |num.val| < (2 : ℤ) ^ 126) :
    ∃ r : I128, hadamard.round_q32_signed num = ok r ∧
      r.val = (if num.val < 0 then -divRoundNearest (-num.val) ((2 : ℤ) ^ 32)
               else divRoundNearest num.val ((2 : ℤ) ^ 32)) ∧
      |r.val * (2 : ℤ) ^ 32 - num.val| ≤ (2 : ℤ) ^ 31 := by
  have hnum' := abs_le.mp (le_of_lt hnum)
  have hz : (0#i128).val = (0 : ℤ) := by decide
  have hmin : IScalar.min IScalarTy.I128 = -(2 : ℤ) ^ 127 := by
    rw [IScalar.min_IScalarTy_I128_eq, I128.min_eq]; norm_num
  have hspec : hadamard.round_q32_signed num ⦃ r =>
      r.val = (if num.val < 0 then -divRoundNearest (-num.val) ((2 : ℤ) ^ 32)
               else divRoundNearest num.val ((2 : ℤ) ^ 32)) ∧
      |r.val * (2 : ℤ) ^ 32 - num.val| ≤ (2 : ℤ) ^ 31 ⦄ := by
    rw [hadamard.round_q32_signed, I128_shiftLeft_32]
    simp only [bind_tc_ok]
    by_cases hneg : num.val < 0
    · -- negative: negate, round the magnitude, negate the rounded magnitude
      have hcond : ¬ (0#i128 ≤ num) := by
        intro hc
        have h := (IScalar.le_equiv (0#i128) num).1 hc
        rw [hz] at h
        omega
      rw [if_neg hcond]
      have hne : num.val ≠ IScalar.min IScalarTy.I128 := by
        rw [hmin]; omega
      refine WP.spec_bind (IScalar.neg_step (ty := IScalarTy.I128) num hne) ?_
      intro i hiv
      have hival : i.val = -num.val := hiv
      have hipos : 0 ≤ i.val := by rw [hival]; omega
      have hib : i.val + 2 ^ 31 ≤ (2 : ℤ) ^ 127 - 1 := by rw [hival]; omega
      refine WP.spec_bind (div_round_nearest_q32_nonneg i hipos hib) ?_
      intro i1 hpost
      obtain ⟨hi1v, hi1nn⟩ := hpost
      have hi1ne : i1.val ≠ IScalar.min IScalarTy.I128 := by rw [hmin]; omega
      refine WP.spec_mono (IScalar.neg_step (ty := IScalarTy.I128) i1 hi1ne) ?_
      intro r hrv
      have hrval : r.val = -i1.val := hrv
      refine ⟨?_, ?_⟩
      · rw [if_pos hneg, hrval, hi1v, hival]
      · rw [hrval, hi1v, hival]
        have hb := div_round_nearest_abs_bound (-num.val) ((2 : ℤ) ^ 32) (by norm_num)
        have he : -((-num.val + (2 : ℤ) ^ 32 / 2) / (2 : ℤ) ^ 32) * (2 : ℤ) ^ 32 - num.val
            = -((((-num.val + (2 : ℤ) ^ 32 / 2) / (2 : ℤ) ^ 32)) * (2 : ℤ) ^ 32
                - (-num.val)) := by
          ring
        rw [divRoundNearest, he, abs_neg]
        have h2 : (2 : ℤ) ^ 32 / 2 = (2 : ℤ) ^ 31 := by norm_num
        rw [h2] at hb
        exact hb
    · -- nonnegative: round directly
      have hcond : 0#i128 ≤ num := by
        rw [IScalar.le_equiv, hz]
        omega
      rw [if_pos hcond]
      have hub : num.val + 2 ^ 31 ≤ (2 : ℤ) ^ 127 - 1 := by omega
      refine WP.spec_mono (div_round_nearest_q32_nonneg num (by omega) hub) ?_
      intro q hpost
      obtain ⟨hqv, _⟩ := hpost
      refine ⟨?_, ?_⟩
      · rw [if_neg (by omega : ¬ num.val < 0), hqv]
      · rw [hqv]
        have hb := div_round_nearest_abs_bound num.val ((2 : ℤ) ^ 32) (by norm_num)
        have h2 : (2 : ℤ) ^ 32 / 2 = (2 : ℤ) ^ 31 := by norm_num
        rw [h2] at hb
        rw [divRoundNearest]
        exact hb
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r, hr, hpost.1, hpost.2⟩

/-! ## Item 2 · the `C` table

`hadamard.C` is the extracted `Array` literal (`src/engine/hadamard.rs:26`). `hadamard_C_val`
unfolds it once; `C_val` reads entry `k`; `C_even` records the exact power of two at even `k`;
`C_close` bounds the distance to `2^32/√(2^k)` by half a unit. -/

/-- The `k`-th entry of the `C` table as an integer, by the table's literal value. -/
def cValue : ℕ → ℤ
  | 0 => 0
  | 1 => 3037000500
  | 2 => 2147483648
  | 3 => 1518500250
  | 4 => 1073741824
  | 5 => 759250125
  | 6 => 536870912
  | 7 => 379625062
  | 8 => 268435456
  | 9 => 189812531
  | 10 => 134217728
  | 11 => 94906266
  | 12 => 67108864
  | 13 => 47453133
  | 14 => 33554432
  | 15 => 23726566
  | 16 => 16777216
  | _ => 0

/-- The `C` table as a list of integers, in index order. -/
def cTable : List ℤ :=
  [cValue 0, cValue 1, cValue 2, cValue 3, cValue 4, cValue 5, cValue 6, cValue 7,
   cValue 8, cValue 9, cValue 10, cValue 11, cValue 12, cValue 13, cValue 14, cValue 15,
   cValue 16]

/-- **The table value.** Unfolding the `Array` literal, the extracted `hadamard.C` maps to
`cTable` entry for entry. -/
theorem hadamard_C_val : hadamard.C.val.map (·.val) = cTable := by
  rw [hadamard.C, Array.make_val]
  rfl

/-- **Item 2a · table entry.** For `1 ≤ k ≤ 16` the `k`-th `hadamard.C` entry has value
`cValue k`. -/
theorem C_val (k : ℕ) (hk1 : 1 ≤ k) (hk16 : k ≤ 16) :
    (hadamard.C.val[k]?).map (·.val) = some (cValue k) := by
  rw [← List.getElem?_map, hadamard_C_val]
  interval_cases k <;> rfl

/-- **Item 2b · even entries are exact.** For even `k`, `C_k = 2^(32 − k/2)`: the Q32.32 constant
for `1/√(2^k)` is an exact power of two. -/
theorem C_even (k : ℕ) (hk1 : 1 ≤ k) (hk16 : k ≤ 16) (hkeven : 2 ∣ k) :
    cValue k = (2 : ℤ) ^ (32 - k / 2) := by
  interval_cases k <;> norm_num at hkeven
  all_goals norm_num [cValue]

/-- The real value the `k`-th table entry approximates: `2^32/√(2^k)`. -/
def cReal (k : ℕ) : ℝ := (2 : ℝ) ^ (32 : ℕ) / Real.sqrt ((2 : ℝ) ^ k)

/-- The squaring core of `C_close`: for positive integers `c` and `k`, brackets around
`2^64/2^k` put `c` within `1/2` of `2^32/√(2^k)`, because `(c ± 1/2)²` brackets the square. -/
theorem c_close_of (c : ℤ) (k : ℕ)
    (hlo : ((c : ℝ) - 1 / 2) ^ 2 ≤ (2 : ℝ) ^ (64 : ℕ) / (2 : ℝ) ^ k)
    (hhi : (2 : ℝ) ^ (64 : ℕ) / (2 : ℝ) ^ k ≤ ((c : ℝ) + 1 / 2) ^ 2)
    (hc1 : (1 : ℝ) ≤ (c : ℝ)) :
    |(c : ℝ) - cReal k| ≤ 1 / 2 := by
  have hkpos : (0 : ℝ) < (2 : ℝ) ^ k := by positivity
  have hs : (0 : ℝ) < Real.sqrt ((2 : ℝ) ^ k) := Real.sqrt_pos.mpr hkpos
  have ht2 : cReal k ^ 2 = (2 : ℝ) ^ (64 : ℕ) / (2 : ℝ) ^ k := by
    rw [cReal, div_pow, Real.sq_sqrt (le_of_lt hkpos)]
    ring
  have htpos : 0 < cReal k := by rw [cReal]; positivity
  rw [abs_le]
  constructor
  · -- `cReal k ≤ c + 1/2` from `2^64/2^k ≤ (c + 1/2)²`
    have h1 : cReal k ^ 2 ≤ ((c : ℝ) + 1 / 2) ^ 2 := by rw [ht2]; exact hhi
    have h2 : |cReal k| ≤ |(c : ℝ) + 1 / 2| := sq_le_sq.mp h1
    have hc0 : (0 : ℝ) ≤ (c : ℝ) + 1 / 2 := by linarith
    rw [abs_of_nonneg hc0, abs_of_pos htpos] at h2
    linarith
  · -- `c - 1/2 ≤ cReal k` from `(c - 1/2)² ≤ 2^64/2^k`
    have h1 : ((c : ℝ) - 1 / 2) ^ 2 ≤ cReal k ^ 2 := by rw [ht2]; exact hlo
    have h2 : |(c : ℝ) - 1 / 2| ≤ |cReal k| := sq_le_sq.mp h1
    have hc0 : (0 : ℝ) ≤ (c : ℝ) - 1 / 2 := by linarith
    rw [abs_of_nonneg hc0, abs_of_pos htpos] at h2
    linarith

/-- **Item 2c · half-ulp accuracy.** For `1 ≤ k ≤ 16`,
`|C_k − 2^32/√(2^k)| ≤ 1/2`, proved per `k` by squaring the integer bracket
`(C_k − 1/2)² ≤ 2^64/2^k ≤ (C_k + 1/2)²` and applying `Real.sqrt` monotonicity. -/
theorem C_close (k : ℕ) (hk1 : 1 ≤ k) (hk16 : k ≤ 16) :
    |(cValue k : ℝ) - cReal k| ≤ 1 / 2 := by
  interval_cases k <;>
    exact c_close_of _ _
      (by norm_num [cValue])
      (by norm_num [cValue])
      (by norm_num [cValue])

/-! ## Item 6 · validation refusals

`hadamard.fwht_fixed` (and its wrapper `word_fwht_fixed`) checks the block exponent, the length
divisibility, the sign length and every sign entry before the first write. Each refusal branch
returns `Ok (Err e, x)` with `x` the input slice. -/

/-- **Not a power of two.** When `block_exponent` fails, the body returns `Err NotPowerOfTwo`
with `x` untouched. -/
theorem fwht_fixed_not_pow (x : Slice I64) (signs : Option (Slice I8)) (block : Usize)
    (h : hadamard.block_exponent block = ok none) :
    word_fwht_fixed x signs block =
      ok (core.result.Result.Err hadamard.HadamardError.NotPowerOfTwo, x) := by
  rw [word_fwht_fixed, hadamard.fwht_fixed.eq_def, h]
  simp only [bind_tc_ok]

/-- `is_multiple_of n m` is `false` exactly when `m` does not divide `n`. -/
theorem is_multiple_of_eq_false (n m : Usize) (h : ¬ m.val ∣ n.val) :
    core.num.Usize.is_multiple_of n m = ok false := by
  have hmod : n.val % m.val ≠ 0 := fun h0 => h (Nat.dvd_iff_mod_eq_zero.mpr h0)
  unfold core.num.Usize.is_multiple_of UScalar.is_multiple_of
  rw [show (n.val % m.val == 0) = false by
    rw [Bool.eq_false_iff]
    intro hh
    exact hmod (beq_iff_eq.mp hh)]

/-- **Length mismatch, block not dividing the length.** On a valid block exponent with
`block ∤ x.len`, the body returns `Err LengthMismatch` with `x` untouched. -/
theorem fwht_fixed_length_mismatch_mod (x : Slice I64) (signs : Option (Slice I8))
    (block k : Usize) (hexp : hadamard.block_exponent block = ok (some k))
    (hmul : core.num.Usize.is_multiple_of (Slice.len x) block = ok false) :
    word_fwht_fixed x signs block =
      ok (core.result.Result.Err hadamard.HadamardError.LengthMismatch, x) := by
  rw [word_fwht_fixed, hadamard.fwht_fixed.eq_def, hexp]
  simp only [bind_tc_ok, hmul, Bool.false_eq_true, if_false]

/-- **Length mismatch, sign length.** On a valid block exponent, a divisible length and a sign
slice whose length differs from `x.len`, the body returns `Err LengthMismatch` with `x`
untouched. -/
theorem fwht_fixed_length_mismatch_signs (x : Slice I64) (s : Slice I8) (block k : Usize)
    (hexp : hadamard.block_exponent block = ok (some k))
    (hmul : core.num.Usize.is_multiple_of (Slice.len x) block = ok true)
    (hlen : Slice.len s ≠ Slice.len x) :
    word_fwht_fixed x (some s) block =
      ok (core.result.Result.Err hadamard.HadamardError.LengthMismatch, x) := by
  have hbne : (Slice.len s != Slice.len x) = true := by
    rw [bne_iff_ne]; exact hlen
  rw [word_fwht_fixed, hadamard.fwht_fixed.eq_def, hexp]
  simp only [bind_tc_ok, hmul, if_true, hbne]

/-- **Bad sign.** On a valid block exponent, a matching sign length and a sign-validation loop
that stops false, the body returns `Err BadSign` with `x` untouched. -/
theorem fwht_fixed_bad_sign (x : Slice I64) (s : Slice I8) (block k : Usize)
    (hexp : hadamard.block_exponent block = ok (some k))
    (hmul : core.num.Usize.is_multiple_of (Slice.len x) block = ok true)
    (hlen : Slice.len s = Slice.len x)
    (hbad : hadamard.fwht_fixed_loop1 s 0#usize true = ok false) :
    word_fwht_fixed x (some s) block =
      ok (core.result.Result.Err hadamard.HadamardError.BadSign, x) := by
  have hbne : ¬ ((Slice.len s != Slice.len x) = true) := by
    rw [bne_iff_ne]; exact fun h => h hlen
  rw [word_fwht_fixed, hadamard.fwht_fixed.eq_def, hexp]
  simp only [bind_tc_ok, hmul]
  rw [if_true, if_neg hbne, hbad]
  simp only [bind_tc_ok, Bool.false_eq_true, if_false]

/-! ## Item 3 · the sign-validation and sign-multiply steps

`fwht_fixed` validates every sign entry before the first write (`fwht_fixed_loop1`) and then
applies the checked negation `0 − x_i` at every entry with `s_i = −1` (`fwht_fixed_loop2`). -/

/-- `true` iff `v ∈ {+1, −1}`. -/
def signGood (v : I8) : Bool := (v == 1#i8) || (v == (-1)#i8)

/-- The `[lo, hi)` sign-run check. -/
def signsGoodRange (s : Slice I8) (lo hi : ℕ) : Bool :=
  ((s.val.drop lo).take (hi - lo)).all signGood

/-- All entries of `s` from `idx` on are `±1`. -/
def signsGoodFrom (s : Slice I8) (idx : ℕ) : Bool := (s.val.drop idx).all signGood

/-- A whole `±1` sign row. -/
def signsGood (s : Slice I8) : Prop := signsGoodFrom s 0 = true

/-- The extracted sign-step computes `v ∈ {±1}`. -/
theorem signGood_body (v : I8) :
    (if v != 1#i8 then (if v != (-1)#i8 then ok false else ok true) else ok true)
      = ok (signGood v) := by
  unfold signGood
  by_cases h1 : v = 1#i8
  · subst h1
    have hb : (1#i8 != 1#i8) = false := by decide
    have hb2 : (1#i8 == 1#i8) = true := by decide
    rw [hb, hb2, Bool.true_or]
    simp
  · have h1' : (v != 1#i8) = true := by rw [bne_iff_ne]; exact h1
    rw [if_pos (by rw [h1'])]
    by_cases h2 : v = (-1)#i8
    · subst h2
      have hb2 : ((-1)#i8 != (-1)#i8) = false := by decide
      have hb1 : ((-1)#i8 == 1#i8) = false := by decide
      have hb3 : ((-1)#i8 == (-1)#i8) = true := by decide
      rw [hb2, hb1, hb3, Bool.or_true]
      simp
    · have h2' : (v != (-1)#i8) = true := by rw [bne_iff_ne]; exact h2
      rw [if_pos (by rw [h2'])]
      have hb1 : (v == 1#i8) = false := by
        rw [Bool.eq_false_iff]; intro hh; exact h1 (beq_iff_eq.mp hh)
      have hb2 : (v == (-1)#i8) = false := by
        rw [Bool.eq_false_iff]; intro hh; exact h2 (beq_iff_eq.mp hh)
      rw [hb1, hb2, Bool.or_self]

theorem signsGoodRange_self (s : Slice I8) (lo : ℕ) : signsGoodRange s lo lo = true := by
  simp [signsGoodRange]

/-- A range split at an interior index. -/
theorem signsGoodRange_append (s : Slice I8) (lo mid hi : ℕ)
    (hlm : lo ≤ mid) (hmh : mid ≤ hi) :
    signsGoodRange s lo hi = (signsGoodRange s lo mid && signsGoodRange s mid hi) := by
  unfold signsGoodRange
  rw [show hi - lo = (mid - lo) + (hi - mid) by omega, List.take_add, List.drop_drop,
    show lo + (mid - lo) = mid by omega, List.all_append]

/-- Extend a range by one good entry. -/
theorem signsGoodRange_succ (s : Slice I8) (lo hi : ℕ) (hlh : lo ≤ hi)
    (hhi : hi < s.val.length) :
    signsGoodRange s lo (hi + 1) = (signsGoodRange s lo hi && signGood s.val[hi]) := by
  have happ := signsGoodRange_append s lo hi (hi + 1) hlh (by omega)
  rw [happ]
  congr 1
  unfold signsGoodRange
  have htake : (s.val.drop hi).take 1 = [s.val[hi]] := by
    rw [List.drop_eq_getElem_cons hhi, List.take_succ_cons, List.take_zero]
  rw [show hi + 1 - hi = 1 by omega, htake]
  simp

theorem signsGoodFrom_eq_range (s : Slice I8) (idx : ℕ) :
    signsGoodFrom s idx = signsGoodRange s idx s.val.length := by
  unfold signsGoodFrom signsGoodRange
  rw [List.take_of_length_le (by rw [List.length_drop])]

/-- **Item 3a · sign validation.** From `idx`, `fwht_fixed_loop1` returns the conjunction of the
incoming flag with the `±1` check on `[idx, s.len)`. -/
theorem fwht_fixed_loop1_spec (s : Slice I8) (idx : Usize) (ok0 : Bool)
    (hidx : idx.val ≤ s.val.length) :
    hadamard.fwht_fixed_loop1 s idx ok0 = ok (ok0 && signsGoodFrom s idx.val) := by
  have hspec : hadamard.fwht_fixed_loop1 s idx ok0
      ⦃ b => b = (ok0 && signsGoodFrom s idx.val) ⦄ := by
    unfold hadamard.fwht_fixed_loop1
    apply loop.spec_decr_nat
      (measure := fun st : Usize × Bool => s.val.length - st.1.val)
      (inv := fun st => idx.val ≤ st.1.val ∧ st.1.val ≤ s.val.length ∧
        st.2 = (ok0 && signsGoodRange s idx.val st.1.val))
    · intro st hst
      rcases st with ⟨i, ok1⟩
      obtain ⟨hile0, hile, hflag⟩ := hst
      dsimp only at hile0 hile hflag
      dsimp only
      by_cases hlt : i < Slice.len s
      · have hltv : i.val < s.val.length := by
          have h := (UScalar.lt_equiv i (Slice.len s)).mp hlt
          rwa [Slice.len_val] at h
        by_cases hok : ok1 = true
        · -- continue: read the sign, fold it into the flag, advance
          rw [hadamard.fwht_fixed_loop1.body, if_pos hlt, if_pos hok]
          refine WP.spec_bind (Slice.index_usize_spec s i hltv) ?_
          intro i1 hi1v
          rw [signGood_body i1]
          simp only [bind_tc_ok]
          have himax : i.val + (1#usize).val ≤ Usize.max := by
            have h1 : (1#usize).val = 1 := by simp
            rw [h1]
            have hlenmax : s.val.length ≤ Usize.max := Slice.length_ineq s
            omega
          refine WP.spec_bind (Usize.add_spec (x := i) (y := 1#usize) himax) ?_
          intro i1u hi1uv
          have hi1uv' : i1u.val = i.val + 1 := by rw [hi1uv]; simp
          refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
          · rw [hi1uv']; omega
          · rw [hi1uv']; omega
          · rw [hi1uv']
            rw [signsGoodRange_succ s idx.val i.val (by omega) hltv]
            have hflag' : (ok0 && signsGoodRange s idx.val i.val) = true := by
              rw [← hflag]; exact hok
            have hgood : signGood i1 = signGood s.val[i.val] := by rw [hi1v]
            rw [hgood]
            conv_rhs => rw [← Bool.and_assoc]
            rw [hflag', Bool.true_and]
          · rw [hi1uv']; omega
        · -- the flag is false: stop with `false`
          have hokf : ok1 = false := by cases ok1 <;> simp_all
          have hpost : false = (ok0 && signsGoodFrom s idx.val) := by
            rw [signsGoodFrom_eq_range s idx.val,
              signsGoodRange_append s idx.val i.val s.val.length (by omega) (by omega)]
            have hflag' : (ok0 && signsGoodRange s idx.val i.val) = false := by
              rw [← hflag]; exact hokf
            conv_rhs => rw [← Bool.and_assoc]
            rw [hflag', Bool.false_and]
          rw [hadamard.fwht_fixed_loop1.body, if_pos hlt, if_neg (by rw [hokf]; simp),
            WP.spec_ok]
          exact hpost
      · -- index at the end: return the accumulated flag
        have hnot : ¬ (i < Slice.len s) := hlt
        have hilen : i.val = s.val.length := by
          have h1 : ¬ (i.val < s.val.length) := by
            intro hh
            exact hnot ((UScalar.lt_equiv i (Slice.len s)).mpr (by rw [Slice.len_val]; exact hh))
          omega
        have hpost : ok1 = (ok0 && signsGoodFrom s idx.val) := by
          rw [hflag, hilen, signsGoodFrom_eq_range s idx.val]
        rw [hadamard.fwht_fixed_loop1.body, if_neg hnot, WP.spec_ok]
        exact hpost
    · exact ⟨le_rfl, hidx, by simp [signsGoodRange_self]⟩
  obtain ⟨b, hb, hpost⟩ := WP.spec_imp_exists hspec
  rw [hb, hpost]

/-- Pointwise product of two integer lists. -/
def smulList (ss xs : List ℤ) : List ℤ := List.zipWith (· * ·) ss xs

theorem smulList_take_length (a b : List ℤ) (i : ℕ) (ha : i ≤ a.length)
    (hb : i ≤ b.length) : (smulList (a.take i) (b.take i)).length = i := by
  unfold smulList
  rw [List.length_zipWith, List.length_take, List.length_take,
    Nat.min_eq_left ha, Nat.min_eq_left hb, Nat.min_self]

theorem smulList_take_succ (a b : List ℤ) (i : ℕ) (ha : i < a.length) (hb : i < b.length) :
    smulList (a.take (i + 1)) (b.take (i + 1)) =
      (smulList (a.take i) (b.take i) ++ [a[i] * b[i]]) := by
  have ha' : a.take (i + 1) = a.take i ++ [a[i]] := by
    rw [List.take_add_one, List.getElem?_eq_getElem ha, Option.toList_some]
  have hb' : b.take (i + 1) = b.take i ++ [b[i]] := by
    rw [List.take_add_one, List.getElem?_eq_getElem hb, Option.toList_some]
  rw [ha', hb']
  have hl : (a.take i).length = (b.take i).length := by
    rw [List.length_take, List.length_take, Nat.min_eq_left (by omega),
      Nat.min_eq_left (by omega)]
  unfold smulList
  rw [List.zipWith_append hl]
  simp

theorem set_middle_cons (A B : List ℤ) (y v : ℤ) :
    (A ++ y :: B).set A.length v = A ++ v :: B := by
  have h1 : (A ++ y :: B).drop (A.length + 1) = B := by
    rw [← List.drop_drop (i := 1) (j := A.length),
      List.drop_append_of_le_length (le_refl A.length), List.drop_length]
    rfl
  rw [List.set_eq_take_cons_drop v (by simp), h1, List.take_left]

theorem append_length_getElem? {P Q : List ℤ} {i : ℕ} (hP : P.length = i) :
    (P ++ Q)[i]? = Q[0]? := by
  rw [List.getElem?_append_right (by omega), hP, Nat.sub_self]

theorem set_middle_cons_of_len (A B : List ℤ) (y v : ℤ) (i : ℕ) (h : A.length = i) :
    (A ++ y :: B).set i v = A ++ v :: B := by
  subst h
  exact set_middle_cons A B y v

/-- **Item 3b · sign multiply.** On a `±1` sign row with the checked-negation window, the
extracted `fwht_fixed_loop2` negates exactly the `s_i = −1` entries and returns `s ⊙ x`. -/
theorem fwht_fixed_loop2_spec (x : Slice I64) (s : Slice I8)
    (hlen : s.val.length = x.val.length)
    (hgood : ∀ (j : ℕ) (a : I8), s.val[j]? = some a → a.val = 1 ∨ a.val = -1)
    (hwin : ∀ (j : ℕ) (a : I8) (b : I64), s.val[j]? = some a → x.val[j]? = some b →
      a.val = -1 → b.val ≠ I64.min) :
    ∃ x', hadamard.fwht_fixed_loop2 x s 0#usize true none = ok (x', none) ∧
      x'.val.map (·.val) = smulList (s.val.map (·.val)) (x.val.map (·.val)) := by
  have hspec : hadamard.fwht_fixed_loop2 x s 0#usize true none ⦃ p =>
      p.2 = none ∧
      p.1.val.map (·.val) = smulList (s.val.map (·.val)) (x.val.map (·.val)) ⦄ := by
    unfold hadamard.fwht_fixed_loop2
    apply loop.spec_decr_nat
      (measure := fun st : Slice I64 × Usize × Bool × Option hadamard.HadamardError =>
        x.val.length - st.2.1.val)
      (inv := fun st => st.1.val.length = x.val.length ∧ st.2.1.val ≤ x.val.length ∧
        st.2.2.1 = true ∧ st.2.2.2 = none ∧
        st.1.val.map (·.val) =
          (smulList ((s.val.map (·.val)).take st.2.1.val)
            ((x.val.map (·.val)).take st.2.1.val)
            ++ (x.val.map (·.val)).drop st.2.1.val) ∧
        (∀ j, st.2.1.val ≤ j → j < x.val.length → st.1.val[j]? = x.val[j]?))
    · intro st hst
      rcases st with ⟨x1, i, ok1, err1⟩
      obtain ⟨hx1len, hile, hok1, herr1, hmap1, hsuff⟩ := hst
      dsimp only at hx1len hile hok1 herr1 hmap1 hsuff
      dsimp only
      by_cases hlt : i < Slice.len x
      · have hltv : i.val < x.val.length := by
          have h := (UScalar.lt_equiv i (Slice.len x)).mp hlt
          rwa [Slice.len_val] at h
        have hltv1 : i.val < x1.val.length := by rw [hx1len]; exact hltv
        by_cases hok : ok1 = true
        · rw [hadamard.fwht_fixed_loop2.body]
          have hsx1 : Slice.len x1 = Slice.len x := by
            apply UScalar.eq_of_val_eq
            simpa only [Slice.len_val] using hx1len
          rw [hsx1, if_pos hlt, if_pos hok]
          have hslt : i.val < s.val.length := by rw [hlen]; exact hltv
          refine WP.spec_bind (Slice.index_usize_spec s i hslt) ?_
          intro sv hsv
          have hsv' : s.val[i.val]? = some sv := by
            rw [List.getElem?_eq_getElem hslt, hsv]
          have hsi : i.val < (s.val.map (·.val)).length := by
            simp only [List.length_map]; exact hslt
          have hxi : i.val < (x.val.map (·.val)).length := by
            simp only [List.length_map]; exact hltv
          have hsMq : (s.val.map (·.val))[i.val]? = some sv.val := by
            rw [List.getElem?_map, hsv']
            rfl
          have hsMget : (s.val.map (·.val))[i.val]'hsi = sv.val := by
            have h := hsMq
            rw [List.getElem?_eq_getElem hsi] at h
            exact Option.some.inj h
          have hsz : (smulList ((s.val.map (·.val)).take i.val)
              ((x.val.map (·.val)).take i.val)).length = i.val :=
            smulList_take_length _ _ i.val (by simp only [List.length_map]; omega)
              (by simp only [List.length_map]; omega)
          have hdropv : (x.val.map (·.val)).drop i.val
              = (x.val.map (·.val))[i.val]'hxi :: (x.val.map (·.val)).drop (i.val + 1) :=
            List.drop_eq_getElem_cons hxi
          have hsucc := smulList_take_succ (s.val.map (·.val)) (x.val.map (·.val)) i.val hsi hxi
          have himax : i.val + (1#usize).val ≤ Usize.max := by
            have h1 : (1#usize).val = 1 := by simp
            rw [h1]
            have hlenmax : x.val.length ≤ Usize.max := Slice.length_ineq x
            omega
          by_cases hneg : sv.val = -1
          · -- negate: `0 - x_i` on the checked path
            have hsvneg : sv = (-1)#i8 := by
              apply IScalar.eq_of_val_eq
              rw [hneg]; decide
            rw [if_pos hsvneg]
            refine WP.spec_bind (Slice.index_usize_spec x1 i hltv1) ?_
            intro xv hxv
            have hxv1 : x1.val[i.val]? = some xv := by
              rw [List.getElem?_eq_getElem hltv1, hxv]
            have hxvX : x.val[i.val]? = some xv := by
              rw [← hsuff i.val le_rfl hltv]; exact hxv1
            have hxvne : xv.val ≠ I64.min := hwin i.val sv xv hsv' hxvX hneg
            have hb : I64.min ≤ xv.val ∧ xv.val ≤ I64.max := by
              have h := xv.hBounds
              rw [show IScalarTy.I64.numBits = 64 from rfl] at h
              norm_num at h
              rw [I64.min_eq, I64.max_eq]
              omega
            have hbnd : I64.min ≤ (0#i64).val - xv.val ∧ (0#i64).val - xv.val ≤ I64.max := by
              have hz : (0#i64).val = (0 : ℤ) := by decide
              have hgt : I64.min < xv.val := lt_of_le_of_ne hb.1 (Ne.symm hxvne)
              rw [I64.min_eq, I64.max_eq] at hb
              rw [I64.min_eq] at hgt
              rw [hz, I64.min_eq, I64.max_eq]
              omega
            obtain ⟨v, hv, hvv⟩ := fwht_checked_sub_some (0#i64) xv hbnd.1 hbnd.2
            rw [hv]
            simp only [lift, bind_tc_ok, core.option.Option.ok_or_some]
            refine WP.spec_bind (Slice.update_spec x1 i v hltv1) ?_
            intro x1u hupd
            have hx1umap : x1u.val.map (·.val) = (x1.val.map (·.val)).set i.val v.val := by
              rw [hupd, Slice.set_val_eq, List.map_set]
            have hxMq : (x.val.map (·.val))[i.val]? = some xv.val := by
              have h := congrArg (fun l => l[i.val]?) hmap1
              rw [List.getElem?_map, hxv1, Option.map_some] at h
              rw [append_length_getElem? hsz, List.getElem?_drop, Nat.add_zero] at h
              exact h.symm
            have hxM' : (x.val.map (·.val))[i.val]'hxi = xv.val := by
              have h := hxMq
              rw [List.getElem?_eq_getElem hxi] at h
              exact Option.some.inj h
            have hvval : v.val = sv.val * xv.val := by
              rw [hvv]
              have hz : (0#i64).val = (0 : ℤ) := by decide
              rw [hz, hneg]; ring
            refine WP.spec_bind (Usize.add_spec (x := i) (y := 1#usize) himax) ?_
            intro i1u hi1uv
            have hi1uv' : i1u.val = i.val + 1 := by rw [hi1uv]; simp
            rw [WP.spec_ok]
            dsimp only
            refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
            · rw [hupd, Slice.set_val_eq, List.length_set]; exact hx1len
            · rw [hi1uv']; omega
            · rfl
            · exact herr1
            · rw [hi1uv', hx1umap, hmap1, hdropv, hsucc]
              rw [set_middle_cons_of_len _ _ _ _ _ hsz]
              simp only [List.append_assoc, List.cons_append, List.nil_append, hsMget,
                hxM', hvval]
            · intro j hj hjlen
              rw [hi1uv'] at hj
              rw [hupd, Slice.set_val_eq, List.getElem?_set, if_neg (by omega)]
              exact hsuff j (by omega) hjlen
            · rw [hi1uv']; omega
          · -- keep: `s_i = +1`
            have hsvone : sv.val = 1 := by rcases hgood i.val sv hsv' with h | h <;> omega
            have hsvpos : ¬ (sv = (-1)#i8) := by
              intro hc
              apply hneg
              rw [hc]
              decide
            rw [if_neg hsvpos]
            refine WP.spec_bind (Usize.add_spec (x := i) (y := 1#usize) himax) ?_
            intro i1u hi1uv
            have hi1uv' : i1u.val = i.val + 1 := by rw [hi1uv]; simp
            rw [WP.spec_ok]
            dsimp only
            refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
            · exact hx1len
            · rw [hi1uv']; omega
            · rfl
            · exact herr1
            · rw [hi1uv', hsucc, hmap1, hdropv]
              simp only [List.append_assoc, List.cons_append, List.nil_append, hsMget,
                hsvone, one_mul]
            · intro j hj hjlen
              rw [hi1uv'] at hj
              exact hsuff j (by omega) hjlen
            · rw [hi1uv']; omega
        · -- flag false: stop
          have hokf : ok1 = false := by cases ok1 <;> simp_all
          rw [hadamard.fwht_fixed_loop2.body]
          have hsx1 : Slice.len x1 = Slice.len x := by
            apply UScalar.eq_of_val_eq
            simpa only [Slice.len_val] using hx1len
          rw [hsx1, if_pos hlt, if_neg (by rw [hokf]; simp), WP.spec_ok]
          exfalso
          rw [hok1] at hokf
          exact absurd hokf (by simp)
      · -- index at the end
        have hnot : ¬ (i < Slice.len x) := hlt
        have hilen : i.val = x.val.length := by
          have h1 : ¬ (i.val < x.val.length) := by
            intro hh
            exact hnot ((UScalar.lt_equiv i (Slice.len x)).mpr (by rw [Slice.len_val]; exact hh))
          omega
        rw [hadamard.fwht_fixed_loop2.body]
        have hsx1 : Slice.len x1 = Slice.len x := by
          apply UScalar.eq_of_val_eq
          simpa only [Slice.len_val] using hx1len
        rw [hsx1, if_neg hnot, WP.spec_ok]
        refine ⟨herr1, ?_⟩
        have htake : (s.val.map (·.val)).take i.val = s.val.map (·.val) :=
          List.take_of_length_le (by simp only [List.length_map]; omega)
        have htakex : (x.val.map (·.val)).take i.val = x.val.map (·.val) :=
          List.take_of_length_le (by simp only [List.length_map]; omega)
        have hdropx : (x.val.map (·.val)).drop i.val = [] :=
          List.drop_eq_nil_iff.mpr (by simp only [List.length_map]; omega)
        rw [hmap1, htake, htakex, hdropx, List.append_nil]
    · exact ⟨rfl, by simp, rfl, rfl, by simp [smulList], by intro j _ _; rfl⟩
  obtain ⟨⟨x', err⟩, hp, hpost⟩ := WP.spec_imp_exists hspec
  obtain ⟨herr, hmap⟩ := hpost
  dsimp only at herr
  refine ⟨x', ?_, hmap⟩
  rw [hp, herr]

/-! ## Item 4 · the normalization loop and the composed rotation -/

/-- The checked sign-magnitude rounding of `round_q32_signed` at `|num| < 2^126`, as a function of
the integer. -/
def roundQ (n : ℤ) : ℤ :=
  if n < 0 then -divRoundNearest (-n) ((2 : ℤ) ^ 32) else divRoundNearest n ((2 : ℤ) ^ 32)

/-- **Rounding law, value form.** `round_q32_signed` returns `roundQ` of its argument. -/
theorem round_q32_signed_val (n : I128) (hn : |n.val| < (2 : ℤ) ^ 126) :
    ∃ r : I128, hadamard.round_q32_signed n = ok r ∧ r.val = roundQ n.val := by
  obtain ⟨r, hr, hval, _⟩ := round_q32_signed_spec n hn
  refine ⟨r, hr, ?_⟩
  rw [hval]
  unfold roundQ
  by_cases h : n.val < 0
  · rw [if_pos h]
  · rw [if_neg h]

/-- Checked `i128` multiplication inside the `2^126` window. -/
theorem i128_mul_some (x y : I128) (hb : |x.val * y.val| < (2 : ℤ) ^ 126) :
    ∃ z, x.checked_mul y = some z ∧ z.val = x.val * y.val :=
  I128_checked_mul_some x y
    (by rw [i128_min_eq]; linarith [abs_le.mp (le_of_lt hb)])
    (by rw [i128_max_eq]; linarith [abs_le.mp (le_of_lt hb)])

/-- The normalization prefix shape: `x.length = y.length`, the range `[lo, hi)` holds
`roundQ (· * coeff)` of `y`, and outside `[lo, hi)` `x` is `y`. -/
def normAt (y : Slice I64) (coeff : ℤ) (x : Slice I64) (lo hi : ℕ) : Prop :=
  x.val.length = y.val.length ∧
  (∀ (j : ℕ) (_hjlo : lo ≤ j) (_hjhi : j < hi) (hjy : j < y.val.length),
    (x.val[j]?).map (·.val) = some (roundQ ((y.val[j]'hjy).val * coeff))) ∧
  (∀ (j : ℕ) (_hjout : j < lo ∨ hi ≤ j) (_hjy : j < y.val.length),
    x.val[j]? = y.val[j]?)

end

end apeiron_word
