import Mathlib.Tactic
import Mathlib.Data.Int.Log
import Mathlib.Data.Rat.Floor
import Perf.Identity

/-!
# The identity floor of the bit-identical coopmat epilogue under a relative-precision rounding

`Perf/Identity.lean` proves a floor of `3` roundings for the five-node epilogue
(`src/spirv/coopmat_gemm.comp:263-268`) under a fixed-grid integer rounding, and records that the
fixed grid cannot see nodes 2 and 3 (an on-grid value times an integer is on-grid). That is a model
artefact: an fp32 rounding is relative (a `p`-bit significand, the exponent moves with the value), so
`fd * qf` with an arbitrary fp32 `fd` is not exact. This module replaces the fixed grid with a
relative-precision rounding and states the floor on the same five-node tree.

## The rounding model

`fl p : ℚ → ℚ` rounds to the nearest rational with a `p`-bit significand, ties to even:

* `rne q` is the nearest integer to `q`, a tie going to the even integer;
* for `x > 0`, with `e = ⌊log₂ x⌋` (`Int.log 2 x`), `ulp = 2^(e-p+1)` and `flPos p x = rne(x/ulp) · ulp`;
* `fl p` is odd (`fl_neg`) and `fl p 0 = 0`.

Proved: `fl_mono` (monotone), `fl_idem` (idempotent), both for `p ≥ 1`.

Gap to IEEE binary32 (`p = 24`), stated, not hidden: the carrier is `ℚ`, so there is no exponent
range. Not modelled: overflow to infinity, subnormals (the result of a rounding is never flushed or
gradually underflowed), NaN, signed zero (`fl p 0 = 0`, one zero), the rounding-mode state
(round-to-nearest-even only), and the `intBitsToFloat` reinterpret (the model's leaf `q` is the
float value, not a bit pattern). An fp32 value with a normal exponent is a model value; the model
rounds identically to binary32 on every operand and result whose binary exponent lies in `[-126, 127]`.

## Tree and deletion family

`Perf.E`, `Perf.rcount`, `Perf.N2 .. N5` are reused from `Perf/Identity.lean`. `N1M` replaces its `N1`:
the constant is the shipped `MAGIC_F = 12582912` (`coopmat_gemm.comp:103`), not a grid unit.
`evalF p` evaluates a tree with `fl p` at each `rnd` node. `variantM d1 .. d5` deletes the `rnd` of
node `i` when `dᵢ = true`; `epiM` keeps all five.

| node | shader op | tree |
|---|---|---|
| 1 | `intBitsToFloat(q[e]) - MAGIC_F` | `rnd(var 3 + (-12582912))` |
| 2 | `fd[e] * qf` | `rnd(node1 * var 4)` |
| 3 | `* fsx[j][e]` | `rnd(node2 * var 5)` |
| 4 | `fma(flo[e], fxs[j][e], dqs)` | `rnd(var 1 * var 2 + node3)`; product exact, one rounding |
| 5 | `a[e] = a[e] + s2` | `rnd(var 0 + node4)` |

## Result

All five nodes are deletion-necessary under the relative model, for every setting of the other four
deletion bits (`V1_ne .. V5_ne`, concrete `p = 24` witnesses in `ℚ`). Hence `identity_floor`: any
variant that agrees with `epiM` on every environment keeps all five roundings, floor `5`. Nodes 2
and 3, redundant under the fixed grid (`Perf.redundant_two`, `Perf.redundant_three`), are necessary
here: a significand product `n₁ · fd` needs `log₂ n₁ + log₂ fd` bits, more than `p`.

## Mapping caveats

* The floor quantifies over environments `ρ : Nat → ℚ` with arbitrary fp32 values at every leaf. The
  shipped `q` is `MAGIC_BITS + n` with `n` a bounded integer (`sBias = s*128 + MAGIC_BITS`,
  `coopmat_gemm.comp:177`), so `q` is an fp32 in `[2^23, 2^24)` and `q - MAGIC_F` is an integer of
  small magnitude, exact in fp32. The node-1 witness `V1` uses `q = -16777215`, outside that domain.
  On the shipped domain node 1 is exact; that is **not proved here** (inference from the bit layout),
  and the floor on the shipped input domain is open.
* The family is deletion of `rnd` nodes from this tree. Reassociation (`(fd*qf)*fsx` as `qf*(fd*fsx)`),
  precomputing `fd*fsx`, or changing the sub-expression shapes is not covered; those change which values
  are rounded, so they are not deletions.
* The model's node 4 is the fused `fma` (one rounding). The SPIR-V `FMul`/`FAdd` pair the compiler may
  emit instead has two roundings (`coopmat_gemm.comp:259-262` states the driver fuses it).
* The witnesses prove a change on `ℚ` at `p = 24`. A hardware run on the same operands is the
  falsifier for the model gap (subnormals, flush-to-zero modes); none is run here.

Axiom cone: `{propext, Classical.choice, Quot.sound}` (audited in `AxiomAudit.lean`).
-/

namespace Perf.Float

/-! ## 1 · Nearest-even integer rounding -/

/-- Nearest integer to `q`; a tie goes to the even neighbour. -/
def rne (q : ℚ) : ℤ :=
  if q - ((⌊q⌋ : ℤ) : ℚ) < 1 / 2 then ⌊q⌋
  else if 1 / 2 < q - ((⌊q⌋ : ℤ) : ℚ) then ⌊q⌋ + 1
  else if ⌊q⌋ % 2 = 0 then ⌊q⌋ else ⌊q⌋ + 1

/-- `rne q` is within one half of `q`. -/
theorem rne_near (q : ℚ) : ((rne q : ℤ) : ℚ) - 1 / 2 ≤ q ∧ q ≤ ((rne q : ℤ) : ℚ) + 1 / 2 := by
  have h1 : ((⌊q⌋ : ℤ) : ℚ) ≤ q := Int.floor_le q
  have h2 : q < ((⌊q⌋ : ℤ) : ℚ) + 1 := Int.lt_floor_add_one q
  unfold rne
  split_ifs <;> push_cast <;> constructor <;> linarith

/-- `rne` is the identity on an integer. -/
theorem rne_of_eq (q : ℚ) (m : ℤ) (h : q = (m : ℚ)) : rne q = m := by
  subst h
  norm_num [rne]

/-- An integer within an open half of `q` is `rne q`. -/
theorem rne_of_near (q : ℚ) (m : ℤ) (h1 : (m : ℚ) - 1 / 2 < q) (h2 : q < (m : ℚ) + 1 / 2) :
    rne q = m := by
  obtain ⟨a, b⟩ := rne_near q
  have c1 : ((rne q : ℤ) : ℚ) < (m : ℚ) + 1 := by linarith
  have c2 : (m : ℚ) < ((rne q : ℤ) : ℚ) + 1 := by linarith
  have d1 : rne q < m + 1 := by exact_mod_cast c1
  have d2 : m < rne q + 1 := by exact_mod_cast c2
  omega

/-- A tie goes to the even neighbour `m`. -/
theorem rne_tie (q : ℚ) (m : ℤ) (h : q = (m : ℚ) - 1 / 2 ∨ q = (m : ℚ) + 1 / 2)
    (hm : m % 2 = 0) : rne q = m := by
  rcases h with h | h
  · have hf : ⌊q⌋ = m - 1 := by
      rw [Int.floor_eq_iff]; push_cast; constructor <;> linarith
    unfold rne
    rw [hf]
    split_ifs <;> first | omega | (exfalso; try push_cast at *; linarith)
  · have hf : ⌊q⌋ = m := by
      rw [Int.floor_eq_iff]; constructor <;> linarith
    unfold rne
    rw [hf]
    split_ifs <;> first | omega | (exfalso; linarith)

/-- **`rne` is monotone.** -/
theorem rne_mono : Monotone rne := by
  intro q q' hqq
  by_contra hc
  have hlt : rne q' < rne q := lt_of_not_ge hc
  have h1 : rne q' + 1 ≤ rne q := Int.add_one_le_iff.mpr hlt
  have h1' : ((rne q' : ℤ) : ℚ) + 1 ≤ ((rne q : ℤ) : ℚ) := by exact_mod_cast h1
  obtain ⟨a, b⟩ := rne_near q
  obtain ⟨c, d⟩ := rne_near q'
  have : q = q' := le_antisymm hqq (by linarith)
  subst this
  exact lt_irrefl _ hlt

/-! ## 2 · Relative-precision rounding to a `p`-bit significand -/

/-- The unit in the last place at binary exponent `e` for a `p`-bit significand: `2^(e-p+1)`. -/
def ulp (p : ℕ) (e : ℤ) : ℚ := (2 : ℚ) ^ (e - (p : ℤ) + 1)

/-- Round `x` to a multiple of `ulp p e`, nearest-even. -/
def flE (p : ℕ) (e : ℤ) (x : ℚ) : ℚ := ((rne (x / ulp p e) : ℤ) : ℚ) * ulp p e

/-- Round a positive `x` at its own binary exponent `⌊log₂ x⌋`. -/
def flPos (p : ℕ) (x : ℚ) : ℚ := flE p (Int.log 2 x) x

/-- **The relative-precision rounding**: round-to-nearest-even to a `p`-bit significand, odd in `x`. -/
def fl (p : ℕ) (x : ℚ) : ℚ := if 0 < x then flPos p x else if x < 0 then -flPos p (-x) else 0

theorem ulp_pos (p : ℕ) (e : ℤ) : 0 < ulp p e := by unfold ulp; positivity

theorem ulp_mul (p : ℕ) (e : ℤ) : ulp p e * (2 : ℚ) ^ p = (2 : ℚ) ^ (e + 1) := by
  unfold ulp
  rw [← zpow_natCast, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
  congr 1
  ring

theorem ulp_mul_half (p : ℕ) (hp : 1 ≤ p) (e : ℤ) :
    ulp p e * (2 : ℚ) ^ (p - 1) = (2 : ℚ) ^ e := by
  unfold ulp
  rw [← zpow_natCast, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
  congr 1
  have : (((p - 1 : ℕ)) : ℤ) = (p : ℤ) - 1 := by omega
  rw [this]
  ring

/-- Inside the binade `[2^e, 2^(e+1))` the quotient `x / ulp` lies in `[2^(p-1), 2^p)`. -/
theorem quot_bounds (p : ℕ) (hp : 1 ≤ p) (e : ℤ) (x : ℚ) (h1 : (2 : ℚ) ^ e ≤ x)
    (h2 : x < (2 : ℚ) ^ (e + 1)) :
    (2 : ℚ) ^ (p - 1) ≤ x / ulp p e ∧ x / ulp p e < (2 : ℚ) ^ p := by
  have hu := ulp_pos p e
  constructor
  · rw [le_div_iff₀ hu, mul_comm, ulp_mul_half p hp]
    exact h1
  · rw [div_lt_iff₀ hu, mul_comm, ulp_mul]
    exact h2

/-- The rounded significand lies in `[2^(p-1), 2^p]`; the top value `2^p` is the carry. -/
theorem rne_bounds (p : ℕ) (hp : 1 ≤ p) (e : ℤ) (x : ℚ) (h1 : (2 : ℚ) ^ e ≤ x)
    (h2 : x < (2 : ℚ) ^ (e + 1)) :
    (2 : ℚ) ^ (p - 1) ≤ ((rne (x / ulp p e) : ℤ) : ℚ) ∧
      ((rne (x / ulp p e) : ℤ) : ℚ) ≤ (2 : ℚ) ^ p := by
  obtain ⟨q1, q2⟩ := quot_bounds p hp e x h1 h2
  obtain ⟨n1, n2⟩ := rne_near (x / ulp p e)
  constructor
  · by_contra hc
    have hc' : ((rne (x / ulp p e) : ℤ) : ℚ) < (2 : ℚ) ^ (p - 1) := not_le.mp hc
    have hz : rne (x / ulp p e) < (2 : ℤ) ^ (p - 1) := by exact_mod_cast hc'
    have hz' : rne (x / ulp p e) + 1 ≤ (2 : ℤ) ^ (p - 1) := Int.add_one_le_iff.mpr hz
    have hq : ((rne (x / ulp p e) : ℤ) : ℚ) + 1 ≤ (2 : ℚ) ^ (p - 1) := by exact_mod_cast hz'
    linarith
  · by_contra hc
    have hc' : (2 : ℚ) ^ p < ((rne (x / ulp p e) : ℤ) : ℚ) := not_le.mp hc
    have hz : (2 : ℤ) ^ p < rne (x / ulp p e) := by exact_mod_cast hc'
    have hz' : (2 : ℤ) ^ p + 1 ≤ rne (x / ulp p e) := Int.add_one_le_iff.mpr hz
    have hq : (2 : ℚ) ^ p + 1 ≤ ((rne (x / ulp p e) : ℤ) : ℚ) := by exact_mod_cast hz'
    linarith

/-- A value rounded inside the binade `[2^e, 2^(e+1))` lands in `[2^e, 2^(e+1)]`. -/
theorem flE_bounds (p : ℕ) (hp : 1 ≤ p) (e : ℤ) (x : ℚ) (h1 : (2 : ℚ) ^ e ≤ x)
    (h2 : x < (2 : ℚ) ^ (e + 1)) :
    (2 : ℚ) ^ e ≤ flE p e x ∧ flE p e x ≤ (2 : ℚ) ^ (e + 1) := by
  obtain ⟨b1, b2⟩ := rne_bounds p hp e x h1 h2
  have hu := ulp_pos p e
  unfold flE
  constructor
  · calc (2 : ℚ) ^ e = ulp p e * 2 ^ (p - 1) := (ulp_mul_half p hp e).symm
      _ = 2 ^ (p - 1) * ulp p e := mul_comm _ _
      _ ≤ _ := mul_le_mul_of_nonneg_right b1 hu.le
  · calc _ ≤ (2 : ℚ) ^ p * ulp p e := mul_le_mul_of_nonneg_right b2 hu.le
      _ = ulp p e * 2 ^ p := mul_comm _ _
      _ = _ := ulp_mul p e

/-- The binary exponent of a positive rational: `2^e ≤ x < 2^(e+1)` for `e = Int.log 2 x`. -/
theorem log_bounds {x : ℚ} (hx : 0 < x) :
    (2 : ℚ) ^ (Int.log 2 x) ≤ x ∧ x < (2 : ℚ) ^ (Int.log 2 x + 1) := by
  constructor
  · have := Int.zpow_log_le_self (b := 2) (by norm_num) hx
    exact_mod_cast this
  · have := Int.lt_zpow_succ_log_self (b := 2) (by norm_num) x
    exact_mod_cast this

/-- The exponent is determined by the binade. -/
theorem log_eq (e : ℤ) {x : ℚ} (hx : 0 < x) (h1 : (2 : ℚ) ^ e ≤ x) (h2 : x < (2 : ℚ) ^ (e + 1)) :
    Int.log 2 x = e := by
  have a : e ≤ Int.log 2 x :=
    (Int.zpow_le_iff_le_log (b := 2) (by norm_num) hx).mp (by exact_mod_cast h1)
  have b : Int.log 2 x < e + 1 :=
    (Int.lt_zpow_iff_log_lt (b := 2) (by norm_num) hx).mp (by exact_mod_cast h2)
  omega

theorem flPos_bounds (p : ℕ) (hp : 1 ≤ p) {x : ℚ} (hx : 0 < x) :
    (2 : ℚ) ^ (Int.log 2 x) ≤ flPos p x ∧ flPos p x ≤ (2 : ℚ) ^ (Int.log 2 x + 1) :=
  flE_bounds p hp _ x (log_bounds hx).1 (log_bounds hx).2

theorem flPos_pos (p : ℕ) (hp : 1 ≤ p) {x : ℚ} (hx : 0 < x) : 0 < flPos p x :=
  lt_of_lt_of_le (by positivity) (flPos_bounds p hp hx).1

/-- **`flPos` is monotone on the positives**: across a binade boundary the rounded value stays at or
below `2^(e+1)`, which is at or below every rounded value of the next binade. -/
theorem flPos_mono (p : ℕ) (hp : 1 ≤ p) {x y : ℚ} (hx : 0 < x) (hxy : x ≤ y) :
    flPos p x ≤ flPos p y := by
  have hy : 0 < y := lt_of_lt_of_le hx hxy
  have hle : Int.log 2 x ≤ Int.log 2 y := Int.log_mono_right hx hxy
  rcases hle.eq_or_lt with heq | hlt
  · unfold flPos
    rw [← heq]
    unfold flE
    have hu := ulp_pos p (Int.log 2 x)
    have hq : x / ulp p (Int.log 2 x) ≤ y / ulp p (Int.log 2 x) := by
      rw [div_eq_mul_inv, div_eq_mul_inv]
      exact mul_le_mul_of_nonneg_right hxy (inv_nonneg.mpr hu.le)
    have h3 : rne (x / ulp p (Int.log 2 x)) ≤ rne (y / ulp p (Int.log 2 x)) := rne_mono hq
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast h3) hu.le
  · have hb1 := (flPos_bounds p hp hx).2
    have hb2 := (flPos_bounds p hp hy).1
    calc flPos p x ≤ (2 : ℚ) ^ (Int.log 2 x + 1) := hb1
      _ ≤ (2 : ℚ) ^ (Int.log 2 y) := zpow_le_zpow_right₀ (by norm_num) (by omega)
      _ ≤ flPos p y := hb2

theorem fl_of_pos (p : ℕ) {x : ℚ} (hx : 0 < x) : fl p x = flPos p x := by
  simp [fl, hx]

theorem fl_of_neg (p : ℕ) {x : ℚ} (hx : x < 0) : fl p x = -flPos p (-x) := by
  have h : ¬ (0 < x) := not_lt.mpr hx.le
  simp [fl, h, hx]

theorem fl_zero (p : ℕ) : fl p 0 = 0 := by simp [fl]

/-- `fl` is odd. -/
theorem fl_neg (p : ℕ) (x : ℚ) : fl p (-x) = -fl p x := by
  rcases lt_trichotomy x 0 with h | h | h
  · have h' : 0 < -x := neg_pos.mpr h
    rw [fl_of_pos p h', fl_of_neg p h]
    simp
  · subst h
    simp [fl_zero]
  · have h' : -x < 0 := neg_lt_zero.mpr h
    rw [fl_of_neg p h', fl_of_pos p h]
    simp

theorem fl_nonneg (p : ℕ) (hp : 1 ≤ p) {x : ℚ} (hx : 0 ≤ x) : 0 ≤ fl p x := by
  rcases hx.eq_or_lt with h | h
  · subst h
    simp [fl_zero]
  · rw [fl_of_pos p h]
    exact (flPos_pos p hp h).le

theorem fl_nonpos (p : ℕ) (hp : 1 ≤ p) {x : ℚ} (hx : x ≤ 0) : fl p x ≤ 0 := by
  rcases hx.eq_or_lt with h | h
  · subst h
    simp [fl_zero]
  · rw [fl_of_neg p h]
    have := flPos_pos p hp (neg_pos.mpr h)
    linarith

/-- **`fl` is monotone.** -/
theorem fl_mono (p : ℕ) (hp : 1 ≤ p) : Monotone (fl p) := by
  intro x y hxy
  rcases lt_trichotomy 0 x with hx | hx | hx
  · have hy : 0 < y := lt_of_lt_of_le hx hxy
    rw [fl_of_pos p hx, fl_of_pos p hy]
    exact flPos_mono p hp hx hxy
  · subst hx
    rw [fl_zero]
    exact fl_nonneg p hp hxy
  · rcases lt_trichotomy 0 y with hy | hy | hy
    · exact le_trans (fl_nonpos p hp hx.le) (fl_nonneg p hp hy.le)
    · subst hy
      rw [fl_zero]
      exact fl_nonpos p hp hx.le
    · rw [fl_of_neg p hx, fl_of_neg p hy]
      have := flPos_mono p hp (neg_pos.mpr hy) (neg_le_neg hxy)
      linarith

/-- A multiple `n · ulp` is a fixed point of `flE` at its own exponent. -/
theorem flE_mul_ulp (p : ℕ) (e : ℤ) (n : ℤ) : flE p e ((n : ℚ) * ulp p e) = (n : ℚ) * ulp p e := by
  unfold flE
  rw [mul_div_assoc, div_self (ne_of_gt (ulp_pos p e)), mul_one, rne_of_eq (n : ℚ) n rfl]

theorem flPos_idem_aux (p : ℕ) (hp : 1 ≤ p) (e : ℤ) {x : ℚ} (hx : 0 < x) (h1 : (2 : ℚ) ^ e ≤ x)
    (h2 : x < (2 : ℚ) ^ (e + 1)) : flPos p (flPos p x) = flPos p x := by
  have hy : flPos p x = flE p e x := by
    unfold flPos
    rw [log_eq e hx h1 h2]
  obtain ⟨b1, b2⟩ := rne_bounds p hp e x h1 h2
  obtain ⟨m, hm⟩ : ∃ m : ℤ, rne (x / ulp p e) = m := ⟨_, rfl⟩
  have hfe : flE p e x = (m : ℚ) * ulp p e := by
    unfold flE
    rw [hm]
  rw [hm] at b1 b2
  have hmu := ulp_pos p e
  rw [hy, hfe]
  rcases b2.eq_or_lt with heq | hlt
  · have hy2 : (m : ℚ) * ulp p e = (2 : ℚ) ^ (e + 1) := by
      rw [heq, mul_comm]
      exact ulp_mul p e
    have hy3 : (2 : ℚ) ^ (e + 1) = (((2 : ℤ) ^ (p - 1) : ℤ) : ℚ) * ulp p (e + 1) := by
      push_cast
      rw [mul_comm]
      exact (ulp_mul_half p hp (e + 1)).symm
    have hpos : (0 : ℚ) < (2 : ℚ) ^ (e + 1) := by positivity
    rw [hy2]
    unfold flPos
    rw [log_eq (e + 1) hpos le_rfl (zpow_lt_zpow_right₀ (by norm_num) (by omega))]
    rw [hy3]
    exact flE_mul_ulp p (e + 1) _
  · have hmpos : 0 < (m : ℚ) * ulp p e :=
      mul_pos (lt_of_lt_of_le (by positivity) b1) hmu
    have c1 : (2 : ℚ) ^ e ≤ (m : ℚ) * ulp p e := by
      calc (2 : ℚ) ^ e = ulp p e * 2 ^ (p - 1) := (ulp_mul_half p hp e).symm
        _ = 2 ^ (p - 1) * ulp p e := mul_comm _ _
        _ ≤ _ := mul_le_mul_of_nonneg_right b1 hmu.le
    have c2 : (m : ℚ) * ulp p e < (2 : ℚ) ^ (e + 1) := by
      calc (m : ℚ) * ulp p e < 2 ^ p * ulp p e := mul_lt_mul_of_pos_right hlt hmu
        _ = ulp p e * 2 ^ p := mul_comm _ _
        _ = _ := ulp_mul p e
    unfold flPos
    rw [log_eq e hmpos c1 c2]
    exact flE_mul_ulp p e m

theorem flPos_idem (p : ℕ) (hp : 1 ≤ p) {x : ℚ} (hx : 0 < x) :
    flPos p (flPos p x) = flPos p x :=
  flPos_idem_aux p hp (Int.log 2 x) hx (log_bounds hx).1 (log_bounds hx).2

/-- **`fl` is idempotent.** -/
theorem fl_idem (p : ℕ) (hp : 1 ≤ p) (x : ℚ) : fl p (fl p x) = fl p x := by
  rcases lt_trichotomy x 0 with h | h | h
  · rw [fl_of_neg p h]
    have hp' := flPos_pos p hp (neg_pos.mpr h)
    have h3 : -flPos p (-x) < 0 := by linarith
    rw [fl_of_neg p h3, neg_neg, flPos_idem p hp (neg_pos.mpr h)]
  · subst h
    simp [fl_zero]
  · rw [fl_of_pos p h, fl_of_pos p (flPos_pos p hp h), flPos_idem p hp h]

/-- Evaluation of `fl` on a value inside a stated binade, for concrete witnesses. -/
theorem fl_eval (p : ℕ) (e m : ℤ) (x w : ℚ) (h1 : (2 : ℚ) ^ e ≤ x) (h2 : x < (2 : ℚ) ^ (e + 1))
    (hr : rne (x / ulp p e) = m) (hw : (m : ℚ) * ulp p e = w) : fl p x = w := by
  have hx : 0 < x := lt_of_lt_of_le (by positivity) h1
  rw [fl_of_pos p hx, ← hw]
  unfold flPos
  rw [log_eq e hx h1 h2]
  unfold flE
  rw [hr]

/-! ## 3 · The five-node tree with the shipped `MAGIC_F` -/

/-- Evaluate a tree with `fl p` at each `rnd` node, over `ℚ`. -/
def evalF (p : ℕ) : E → (Nat → ℚ) → ℚ
  | .var n, ρ => ρ n
  | .const c, _ => (c : ℚ)
  | .add a b, ρ => evalF p a ρ + evalF p b ρ
  | .mul a b, ρ => evalF p a ρ * evalF p b ρ
  | .rnd a, ρ => fl p (evalF p a ρ)

/-- Node 1 with the shipped constant `MAGIC_F = 12582912`
(`src/spirv/coopmat_gemm.comp:103`): `rnd(var 3 + (-12582912))`; `true` deletes its `rnd`. -/
def N1M : Bool → E
  | true => E.add (E.var 3) (E.const (-12582912))
  | false => E.rnd (E.add (E.var 3) (E.const (-12582912)))

/-- The tree with the `rnd` of node `i` deleted exactly when `dᵢ = true`; nodes 2 to 5 are
`Perf.N2 .. Perf.N5`. -/
def variantM (d1 d2 d3 d4 d5 : Bool) : E := N5 d5 (N4 d4 (N3 d3 (N2 d2 (N1M d1))))

/-- The identity-preserving epilogue tree: all five roundings present. -/
def epiM : E := variantM false false false false false

/-! ## 4 · Evaluation of the 18 rounding facts used by the witnesses -/

theorem fl24_p12582910 : fl 24 12582910 = 12582910 :=
  fl_eval 24 23 12582910 12582910 12582910 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n12582910 : fl 24 (-12582910) = -12582910 := by
  rw [fl_neg, fl24_p12582910]

theorem fl24_p12582911 : fl 24 12582911 = 12582911 :=
  fl_eval 24 23 12582911 12582911 12582911 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n12582911 : fl 24 (-12582911) = -12582911 := by
  rw [fl_neg, fl24_p12582911]

theorem fl24_p20971520 : fl 24 20971520 = 20971520 :=
  fl_eval 24 24 10485760 20971520 20971520 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n20971520 : fl 24 (-20971520) = -20971520 := by
  rw [fl_neg, fl24_p20971520]

theorem fl24_p20971521 : fl 24 20971521 = 20971520 :=
  fl_eval 24 24 10485760 20971521 20971520 (by norm_num) (by norm_num)
    (rne_tie _ _ (Or.inr (by norm_num [ulp])) (by norm_num)) (by norm_num [ulp])

theorem fl24_n20971521 : fl 24 (-20971521) = -20971520 := by
  rw [fl_neg, fl24_p20971521]

theorem fl24_p20971522 : fl 24 20971522 = 20971522 :=
  fl_eval 24 24 10485761 20971522 20971522 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n20971522 : fl 24 (-20971522) = -20971522 := by
  rw [fl_neg, fl24_p20971522]

theorem fl24_p29360127 : fl 24 29360127 = 29360128 :=
  fl_eval 24 24 14680064 29360127 29360128 (by norm_num) (by norm_num)
    (rne_tie _ _ (Or.inl (by norm_num [ulp])) (by norm_num)) (by norm_num [ulp])

theorem fl24_n29360127 : fl 24 (-29360127) = -29360128 := by
  rw [fl_neg, fl24_p29360127]

theorem fl24_p29360128 : fl 24 29360128 = 29360128 :=
  fl_eval 24 24 14680064 29360128 29360128 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n29360128 : fl 24 (-29360128) = -29360128 := by
  rw [fl_neg, fl24_p29360128]

theorem fl24_p62914548 : fl 24 62914548 = 62914548 :=
  fl_eval 24 25 15728637 62914548 62914548 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n62914548 : fl 24 (-62914548) = -62914548 := by
  rw [fl_neg, fl24_p62914548]

theorem fl24_p62914550 : fl 24 62914550 = 62914552 :=
  fl_eval 24 25 15728638 62914550 62914552 (by norm_num) (by norm_num)
    (rne_tie _ _ (Or.inl (by norm_num [ulp])) (by norm_num)) (by norm_num [ulp])

theorem fl24_n62914550 : fl 24 (-62914550) = -62914552 := by
  rw [fl_neg, fl24_p62914550]

theorem fl24_p62914552 : fl 24 62914552 = 62914552 :=
  fl_eval 24 25 15728638 62914552 62914552 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n62914552 : fl 24 (-62914552) = -62914552 := by
  rw [fl_neg, fl24_p62914552]

theorem fl24_p105553120460799 : fl 24 105553120460799 = 105553116266496 :=
  fl_eval 24 46 12582912 105553120460799 105553116266496 (by norm_num) (by norm_num)
    (rne_of_near _ _ (by norm_num [ulp]) (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n105553120460799 : fl 24 (-105553120460799) = -105553116266496 := by
  rw [fl_neg, fl24_p105553120460799]

theorem fl24_p246290621399040 : fl 24 246290621399040 = 246290621399040 :=
  fl_eval 24 47 14680065 246290621399040 246290621399040 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n246290621399040 : fl 24 (-246290621399040) = -246290621399040 := by
  rw [fl_neg, fl24_p246290621399040]

theorem fl24_p246290625593343 : fl 24 246290625593343 = 246290621399040 :=
  fl_eval 24 47 14680065 246290625593343 246290621399040 (by norm_num) (by norm_num)
    (rne_of_near _ _ (by norm_num [ulp]) (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n246290625593343 : fl 24 (-246290625593343) = -246290621399040 := by
  rw [fl_neg, fl24_p246290625593343]

theorem fl24_p246290633981952 : fl 24 246290633981952 = 246290638176256 :=
  fl_eval 24 47 14680066 246290633981952 246290638176256 (by norm_num) (by norm_num)
    (rne_of_near _ _ (by norm_num [ulp]) (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n246290633981952 : fl 24 (-246290633981952) = -246290638176256 := by
  rw [fl_neg, fl24_p246290633981952]

theorem fl24_p246290638176256 : fl 24 246290638176256 = 246290638176256 :=
  fl_eval 24 47 14680066 246290638176256 246290638176256 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n246290638176256 : fl 24 (-246290638176256) = -246290638176256 := by
  rw [fl_neg, fl24_p246290638176256]

theorem fl24_p527765581332480 : fl 24 527765581332480 = 527765581332480 :=
  fl_eval 24 48 15728640 527765581332480 527765581332480 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n527765581332480 : fl 24 (-527765581332480) = -527765581332480 := by
  rw [fl_neg, fl24_p527765581332480]

theorem fl24_p527765602303995 : fl 24 527765602303995 = 527765614886912 :=
  fl_eval 24 48 15728641 527765602303995 527765614886912 (by norm_num) (by norm_num)
    (rne_of_near _ _ (by norm_num [ulp]) (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n527765602303995 : fl 24 (-527765602303995) = -527765614886912 := by
  rw [fl_neg, fl24_p527765602303995]

theorem fl24_p527765614886912 : fl 24 527765614886912 = 527765614886912 :=
  fl_eval 24 48 15728641 527765614886912 527765614886912 (by norm_num) (by norm_num)
    (rne_of_eq _ _ (by norm_num [ulp])) (by norm_num [ulp])

theorem fl24_n527765614886912 : fl 24 (-527765614886912) = -527765614886912 := by
  rw [fl_neg, fl24_p527765614886912]

/-! ## 5 · Witnesses -/

/-- Witness for node 1 (MAGIC subtract `intBitsToFloat(q) - MAGIC_F`): `a = 0`, `lo = 0`, `xs = 0`, `q = -16777215`, `fd = 8388609`, `fsx = 1`. -/
def V1 : Nat → ℚ
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => -16777215
  | 4 => 8388609
  | 5 => 1
  | _ => 0

/-- At `V1`, deleting node 1 changes the value, whatever the other deletion bits are. -/
theorem V1_ne (b2 b3 b4 b5 : Bool) :
    evalF 24 (variantM true b2 b3 b4 b5) V1 ≠ evalF 24 epiM V1 := by
  rcases b2 <;> rcases b3 <;> rcases b4 <;> rcases b5 <;>
    norm_num [epiM, variantM, N1M, N2, N3, N4, N5, evalF, V1, fl_zero, fl24_p12582910, fl24_n12582910, fl24_p12582911, fl24_n12582911, fl24_p20971520, fl24_n20971520, fl24_p20971521, fl24_n20971521, fl24_p20971522, fl24_n20971522, fl24_p29360127, fl24_n29360127, fl24_p29360128, fl24_n29360128, fl24_p62914548, fl24_n62914548, fl24_p62914550, fl24_n62914550, fl24_p62914552, fl24_n62914552, fl24_p105553120460799, fl24_n105553120460799, fl24_p246290621399040, fl24_n246290621399040, fl24_p246290625593343, fl24_n246290625593343, fl24_p246290633981952, fl24_n246290633981952, fl24_p246290638176256, fl24_n246290638176256, fl24_p527765581332480, fl24_n527765581332480, fl24_p527765602303995, fl24_n527765602303995, fl24_p527765614886912, fl24_n527765614886912]

/-- **`drop_changes` for node 1**: deleting the MAGIC subtract `intBitsToFloat(q) - MAGIC_F` rounding changes the value. -/
theorem drop1_changes :
    ∃ ρ, evalF 24 (variantM true false false false false) ρ ≠ evalF 24 epiM ρ :=
  ⟨V1, V1_ne false false false false⟩

/-- Witness for node 2 (scale product `fd * qf`): `a = 0`, `lo = 0`, `xs = 0`, `q = 1`, `fd = 8388609`, `fsx = 5`. -/
def V2 : Nat → ℚ
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 1
  | 4 => 8388609
  | 5 => 5
  | _ => 0

/-- At `V2`, deleting node 2 changes the value, whatever the other deletion bits are. -/
theorem V2_ne (b1 b3 b4 b5 : Bool) :
    evalF 24 (variantM b1 true b3 b4 b5) V2 ≠ evalF 24 epiM V2 := by
  rcases b1 <;> rcases b3 <;> rcases b4 <;> rcases b5 <;>
    norm_num [epiM, variantM, N1M, N2, N3, N4, N5, evalF, V2, fl_zero, fl24_p12582910, fl24_n12582910, fl24_p12582911, fl24_n12582911, fl24_p20971520, fl24_n20971520, fl24_p20971521, fl24_n20971521, fl24_p20971522, fl24_n20971522, fl24_p29360127, fl24_n29360127, fl24_p29360128, fl24_n29360128, fl24_p62914548, fl24_n62914548, fl24_p62914550, fl24_n62914550, fl24_p62914552, fl24_n62914552, fl24_p105553120460799, fl24_n105553120460799, fl24_p246290621399040, fl24_n246290621399040, fl24_p246290625593343, fl24_n246290625593343, fl24_p246290633981952, fl24_n246290633981952, fl24_p246290638176256, fl24_n246290638176256, fl24_p527765581332480, fl24_n527765581332480, fl24_p527765602303995, fl24_n527765602303995, fl24_p527765614886912, fl24_n527765614886912]

/-- **`drop_changes` for node 2**: deleting the scale product `fd * qf` rounding changes the value. -/
theorem drop2_changes :
    ∃ ρ, evalF 24 (variantM false true false false false) ρ ≠ evalF 24 epiM ρ :=
  ⟨V2, V2_ne false false false false⟩

/-- Witness for node 3 (scale product `* fsx`): `a = 0`, `lo = 2`, `xs = 1`, `q = 2`, `fd = 1`, `fsx = 5`. -/
def V3 : Nat → ℚ
  | 0 => 0
  | 1 => 2
  | 2 => 1
  | 3 => 2
  | 4 => 1
  | 5 => 5
  | _ => 0

/-- At `V3`, deleting node 3 changes the value, whatever the other deletion bits are. -/
theorem V3_ne (b1 b2 b4 b5 : Bool) :
    evalF 24 (variantM b1 b2 true b4 b5) V3 ≠ evalF 24 epiM V3 := by
  rcases b1 <;> rcases b2 <;> rcases b4 <;> rcases b5 <;>
    norm_num [epiM, variantM, N1M, N2, N3, N4, N5, evalF, V3, fl_zero, fl24_p12582910, fl24_n12582910, fl24_p12582911, fl24_n12582911, fl24_p20971520, fl24_n20971520, fl24_p20971521, fl24_n20971521, fl24_p20971522, fl24_n20971522, fl24_p29360127, fl24_n29360127, fl24_p29360128, fl24_n29360128, fl24_p62914548, fl24_n62914548, fl24_p62914550, fl24_n62914550, fl24_p62914552, fl24_n62914552, fl24_p105553120460799, fl24_n105553120460799, fl24_p246290621399040, fl24_n246290621399040, fl24_p246290625593343, fl24_n246290625593343, fl24_p246290633981952, fl24_n246290633981952, fl24_p246290638176256, fl24_n246290638176256, fl24_p527765581332480, fl24_n527765581332480, fl24_p527765602303995, fl24_n527765602303995, fl24_p527765614886912, fl24_n527765614886912]

/-- **`drop_changes` for node 3**: deleting the scale product `* fsx` rounding changes the value. -/
theorem drop3_changes :
    ∃ ρ, evalF 24 (variantM false false true false false) ρ ≠ evalF 24 epiM ρ :=
  ⟨V3, V3_ne false false false false⟩

/-- Witness for node 4 (fma `fma(flo, fxs, dqs)`): `a = 1`, `lo = 1`, `xs = 1`, `q = 33554432`, `fd = 1`, `fsx = 1`. -/
def V4 : Nat → ℚ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | 3 => 33554432
  | 4 => 1
  | 5 => 1
  | _ => 0

/-- At `V4`, deleting node 4 changes the value, whatever the other deletion bits are. -/
theorem V4_ne (b1 b2 b3 b5 : Bool) :
    evalF 24 (variantM b1 b2 b3 true b5) V4 ≠ evalF 24 epiM V4 := by
  rcases b1 <;> rcases b2 <;> rcases b3 <;> rcases b5 <;>
    norm_num [epiM, variantM, N1M, N2, N3, N4, N5, evalF, V4, fl_zero, fl24_p12582910, fl24_n12582910, fl24_p12582911, fl24_n12582911, fl24_p20971520, fl24_n20971520, fl24_p20971521, fl24_n20971521, fl24_p20971522, fl24_n20971522, fl24_p29360127, fl24_n29360127, fl24_p29360128, fl24_n29360128, fl24_p62914548, fl24_n62914548, fl24_p62914550, fl24_n62914550, fl24_p62914552, fl24_n62914552, fl24_p105553120460799, fl24_n105553120460799, fl24_p246290621399040, fl24_n246290621399040, fl24_p246290625593343, fl24_n246290625593343, fl24_p246290633981952, fl24_n246290633981952, fl24_p246290638176256, fl24_n246290638176256, fl24_p527765581332480, fl24_n527765581332480, fl24_p527765602303995, fl24_n527765602303995, fl24_p527765614886912, fl24_n527765614886912]

/-- **`drop_changes` for node 4**: deleting the fma `fma(flo, fxs, dqs)` rounding changes the value. -/
theorem drop4_changes :
    ∃ ρ, evalF 24 (variantM false false false true false) ρ ≠ evalF 24 epiM ρ :=
  ⟨V4, V4_ne false false false false⟩

/-- Witness for node 5 (accumulate `a + s2`): `a = 1`, `lo = 0`, `xs = 0`, `q = -16777215`, `fd = 1`, `fsx = 1`. -/
def V5 : Nat → ℚ
  | 0 => 1
  | 1 => 0
  | 2 => 0
  | 3 => -16777215
  | 4 => 1
  | 5 => 1
  | _ => 0

/-- At `V5`, deleting node 5 changes the value, whatever the other deletion bits are. -/
theorem V5_ne (b1 b2 b3 b4 : Bool) :
    evalF 24 (variantM b1 b2 b3 b4 true) V5 ≠ evalF 24 epiM V5 := by
  rcases b1 <;> rcases b2 <;> rcases b3 <;> rcases b4 <;>
    norm_num [epiM, variantM, N1M, N2, N3, N4, N5, evalF, V5, fl_zero, fl24_p12582910, fl24_n12582910, fl24_p12582911, fl24_n12582911, fl24_p20971520, fl24_n20971520, fl24_p20971521, fl24_n20971521, fl24_p20971522, fl24_n20971522, fl24_p29360127, fl24_n29360127, fl24_p29360128, fl24_n29360128, fl24_p62914548, fl24_n62914548, fl24_p62914550, fl24_n62914550, fl24_p62914552, fl24_n62914552, fl24_p105553120460799, fl24_n105553120460799, fl24_p246290621399040, fl24_n246290621399040, fl24_p246290625593343, fl24_n246290625593343, fl24_p246290633981952, fl24_n246290633981952, fl24_p246290638176256, fl24_n246290638176256, fl24_p527765581332480, fl24_n527765581332480, fl24_p527765602303995, fl24_n527765602303995, fl24_p527765614886912, fl24_n527765614886912]

/-- **`drop_changes` for node 5**: deleting the accumulate `a + s2` rounding changes the value. -/
theorem drop5_changes :
    ∃ ρ, evalF 24 (variantM false false false false true) ρ ≠ evalF 24 epiM ρ :=
  ⟨V5, V5_ne false false false false⟩

/-! ## 6 · The identity floor under the relative model -/

/-- The rounding count of a deletion variant is `5` minus the number of deleted nodes. -/
theorem rcount_variantM (d1 d2 d3 d4 d5 : Bool) :
    rcount (variantM d1 d2 d3 d4 d5) = 5 - (b2n d1 + b2n d2 + b2n d3 + b2n d4 + b2n d5) := by
  cases d1 <;> cases d2 <;> cases d3 <;> cases d4 <;> cases d5 <;> rfl

/-- **The identity floor under the relative model.** Any tree in the node-deletion family that agrees
with the five-rounding epilogue on every `ℚ`-environment at `p = 24` keeps all five roundings. Every
node is forced present by `V1_ne .. V5_ne`; the floor is over node deletions only (see the caveats in
the module header). -/
theorem identity_floor (d1 d2 d3 d4 d5 : Bool)
    (h : ∀ ρ, evalF 24 (variantM d1 d2 d3 d4 d5) ρ = evalF 24 epiM ρ) :
    5 ≤ rcount (variantM d1 d2 d3 d4 d5) := by
  have hd1 : d1 = false := by
    cases d1 with
    | false => rfl
    | true => exact absurd (h V1) (V1_ne d2 d3 d4 d5)
  have hd2 : d2 = false := by
    cases d2 with
    | false => rfl
    | true => exact absurd (h V2) (V2_ne d1 d3 d4 d5)
  have hd3 : d3 = false := by
    cases d3 with
    | false => rfl
    | true => exact absurd (h V3) (V3_ne d1 d2 d4 d5)
  have hd4 : d4 = false := by
    cases d4 with
    | false => rfl
    | true => exact absurd (h V4) (V4_ne d1 d2 d3 d5)
  have hd5 : d5 = false := by
    cases d5 with
    | false => rfl
    | true => exact absurd (h V5) (V5_ne d1 d2 d3 d4)
  subst hd1 hd2 hd3 hd4 hd5
  rw [rcount_variantM]
  decide

/-- The floor is attained: the unmodified epilogue has exactly `5` roundings. -/
theorem epiM_rcount : rcount epiM = 5 := rfl

end Perf.Float
