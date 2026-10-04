/-
  `AlgebraShrExact.lean` — when does the logical right shift `x >>> n` have a cell/algebraic
  reading, at word width `k`, in the two-slot algebra `R[x]/(x^2 − σ)`.

  Task AR-23 of the algebra-rooting wave. Owned file: this one, plus `AlgebraShrExact.md`. No
  production, pin, extraction-root or API edit; no commit.

  AR-12 identified the logical right shift as the one round primitive with no cell reading at all
  and proved `tshrx_not_injective`; AR-16 and AR-21 characterise the rotation's exactness set.
  This file does the same for the shift. The objects are the canonical
  `Apeiron.Crypto.Apeiron.Crypto.Sha2.shrx x n = x >>> (n % 32)` (`Sha2.lean:54`) and the FIPS 180-4 amounts
  `SHR^3` (`sigma0`, `Sha2.lean:60`) and `SHR^10` (`sigma1`, `Sha2.lean:64`), re-declared at a
  width parameter `k` in `tshrx`, with `32` replaced by `k`.

  §1 The width-generic bodies and the closed form `den (x >>> m) = ⌊x/2^m⌋`.
  §2 The two-slot family is exactly `x ↦ c·x`, at every σ, by `ringUpper_mul_eval`.
  §3 **The direct obstruction, reproduced and generalised.** The ops map's O3 witness
      (`0 ≡ 2 mod 2` but `0 >>> 1 = 0 ≠ 1 = 2 >>> 1`) is closed here, and the witness
      `2^m ≡ 0 mod 2` but `2^m >>> m = 1 ≠ 0 = 0 >>> m` gives the iff: the shift preserves
      congruence modulo `2` exactly when `k ∣ n`. This closes AR-12's OPEN 1.
  §4 **Not a ring polynomial of any degree.** Reduction `ZMod 2^k → ZMod 2` is a ring hom, so
      every ring-polynomial reading depends on the word only through its residue modulo `2`;
      `RingPolyRead` is the explicit statement and `shr_not_ringpoly` the refutation.
  §5 **The exactness set of the family over the whole word space is `{n | k ∣ n}`.**
  §6 **The partial exactness, in closed form.** For the zero expression the agreement set is
      exactly `{x | x < 2^(n mod k)}`; the "low `m` bits zero" candidate set is refuted with a
      coprimality cancellation; and the zero expression's set is maximal, one word per residue
      class mod `2^m`.

  Grades: `proved` for a named theorem whose `#print axioms` row at the end of this file is
  inside `{propext, Classical.choice, Quot.sound}` with no `sorryAx`. Nothing here is a security,
  collision-resistance, constant-time or break claim.
-/

import WordRefinement
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sha2

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

namespace AlgebraShrExact

open Apeiron.Trichotomy.AnyRing

/-! ## §1 The width-generic bodies, and the closed form of the shift

`Apeiron.Crypto.Sha2.shrx` (`Sha2.lean:54`) is `x >>> (n % 32)` and `Apeiron.Crypto.Sha2.rotr` (`:50`) is
`(x >>> (n % 32)) ||| (x <<< ((32 − n) % 32))`; both bodies are reproduced at width `k`. -/

/-- The width-generic SHA-2 word. `Apeiron.Crypto.Sha2.Word` is `BitVec 32` (`Sha2.lean:22`); the width is a
parameter here so the statements cover `k = 32` and every other width. -/
abbrev TWord (k : Nat) : Type := BitVec k

/-- The word read into `ZMod (2 ^ k)`, the crypto-width denotation at width `k`. -/
def den {k : Nat} (x : BitVec k) : ZMod (2 ^ k) := (x.toNat : ZMod (2 ^ k))

/-- `Apeiron.Crypto.Sha2.shrx` at width `k`: the corpus body with `32` replaced by `k`. -/
def tshrx {k : Nat} (x : BitVec k) (n : Nat) : BitVec k := x >>> (n % k)

/-- `Apeiron.Crypto.Sha2.rotr` at width `k`: the corpus body with `32` replaced by `k`. -/
def trotr {k : Nat} (x : BitVec k) (n : Nat) : BitVec k :=
  (x >>> (n % k)) ||| (x <<< ((k - n) % k))

/-- The natural value of a word built from a numeral: no reduction modulo the width. -/
theorem bitVec_mk_toNat {k : Nat} {a : ℕ} (ha : a < 2 ^ k) : (⟨a, ha⟩ : BitVec k).toNat = a := by
  simp only [BitVec.toNat]

/-- The natural value of the zero word, at every width. -/
theorem bitVec_zero_toNat {k : Nat} : (0 : BitVec k).toNat = 0 := by simp

/-- The natural value of the all-ones word, whose width representative is `2^k − 1`. -/
theorem bitVec_top_toNat {k : Nat} :
    (⟨2 ^ k - 1, by have h1 : 0 < 2 ^ k := Nat.two_pow_pos k; omega⟩ : BitVec k).toNat
      = 2 ^ k - 1 := by
  have hlt : 2 ^ k - 1 < 2 ^ k := by have h1 : 0 < 2 ^ k := Nat.two_pow_pos k; omega
  exact bitVec_mk_toNat hlt

/-- `2 ^ k` is a nonzero natural number for `k > 0`, the `NeZero` that `ZMod.val_add` at width
`k` needs. -/
theorem neZero_two_pow {k : Nat} : NeZero (2 ^ k) :=
  ⟨Nat.ne_of_gt (Nat.two_pow_pos k)⟩

/-- `2 ∣ 2^m` for `m > 0`, the divisibility behind the canonical reduction
`ZMod 2^k → ZMod 2` and behind the cancellation of §6. -/
theorem two_dvd_two_pow {m : Nat} (hm : 0 < m) : (2 : ℕ) ∣ 2 ^ m := by
  have h : 2 ^ m = 2 ^ (m - 1) * 2 ^ 1 := by
    rw [← Nat.pow_add]
    congr 1
    omega
  refine ⟨2 ^ (m - 1), ?_⟩
  rw [h, Nat.pow_one]
  exact (Nat.mul_comm _ _).symm

/-- Strict monotonicity of `2 ^ ·` in the exponent: the only width arithmetic §5 and §6 need. -/
theorem two_pow_lt_two_pow {k m : Nat} (hm : m < k) : 2 ^ m < 2 ^ k :=
  (Nat.pow_lt_pow_iff_right (a := 2) (n := m) (m := k) (by decide)).2 hm

/-- **The closed form of the shift.** The unsigned value of `x >>> n` is `x / 2^(n mod k)`: a
logical right shift is a truncated division and nothing else. -/
theorem tshrx_toNat {k : Nat} (x : BitVec k) (n : Nat) :
    (tshrx x n).toNat = x.toNat / 2 ^ (n % k) := by
  simp [tshrx, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow]

/-- The same statement read into the ring: the cell denotation of `x >>> n` is the truncated
division `x / 2^(n mod k)`. -/
theorem den_tshrx {k : Nat} (x : BitVec k) (n : Nat) :
    den (tshrx x n) = ((x.toNat / 2 ^ (n % k) : ℕ) : ZMod (2 ^ k)) := by
  rw [den, tshrx_toNat]

/-! ## §2 The two-slot family is exactly `x ↦ c·x`, at every σ

`ringUpper r (ringMul σ (x,0) (a,b))` is the reading the ops map uses for a rotation
(`ops-map.md` row S7). The identity is AR-12's `ringUpper_mul_eval`, re-proved here; it is
σ-free, so **every face reads the same family** `{x ↦ c·x}` and no face escapes any obstruction
below. -/

/-- **The two-slot read is one multiplication by `a + r·b`, for every σ and every `r`.** No root of
`σ` is needed: the dial term is multiplied by `x.2 = 0`. -/
theorem ringUpper_mul_eval {R : Type*} [CommRing R] (σ r a b x : R) :
    ringUpper r (ringMul σ (x, 0) (a, b)) = x * (a + r * b) := by
  simp only [ringUpper, ringMul]
  ring

/-- **The two-slot reading of a word operation at width `k`**: the `ringUpper`/`ringMul` family
applied to the word on the line `x ↦ (x, 0)`. -/
def twoSlotRead {k : Nat} (σ r a b : ZMod (2 ^ k)) (x : BitVec k) : ZMod (2 ^ k) :=
  ringUpper r (ringMul σ (den x, 0) (a, b))

/-- The family is the homogeneous family `x ↦ x·(a + r·b)`. -/
theorem twoSlotRead_eq {k : Nat} (σ r a b : ZMod (2 ^ k)) (x : BitVec k) :
    twoSlotRead σ r a b x = den x * (a + r * b) := by
  rw [twoSlotRead, ringUpper_mul_eval]

/-- **The obstruction is σ-independent.** Two faces cannot be distinguished by any two-slot read,
so every statement about the family below holds at the elliptic, parabolic and hyperbolic faces
alike. -/
theorem twoSlotRead_sigma_free {k : Nat} {σ σ' r a b : ZMod (2 ^ k)} (x : BitVec k) :
    twoSlotRead σ r a b x = twoSlotRead σ' r a b x := by
  rw [twoSlotRead_eq, twoSlotRead_eq]

/-- The zero element of the family: the two-slot expression with both multiplier slots `0`. -/
theorem twoSlotRead_zero {k : Nat} (σ r : ZMod (2 ^ k)) (x : BitVec k) :
    twoSlotRead σ r 0 0 x = 0 := by
  rw [twoSlotRead_eq]
  ring

/-- The word's ring value is the natural cast of its own `.val`. This is the bridge that turns a
statement about a ring element's `.val` back into a statement about the element. -/
theorem natCast_val_eq {k : Nat} (c : ZMod (2 ^ k)) : ((c.val : ℕ) : ZMod (2 ^ k)) = c :=
  ZMod.val_injective (2 ^ k) (ZMod.val_natCast_of_lt (ZMod.val_lt c))

/-! ## §3 The direct obstruction: the shift is not congruence-preserving

This is the ops map's O3 (`ops-map.md:140`), whose witness is `0 ≡ 2 mod 2` but
`0 >> 1 = 0 ≠ 1 = 2 >> 1`. AR-12 checked one instance at widths 2 and 4 and recorded the
general form as its OPEN 1. Here the witness is generalised to every width and every amount, and
the property itself is characterised by an `iff`. -/

/-- **`2^n % 2 = 0` for `n ≥ 1`.** This is the step AR-12's OPEN 1 named as missing, discharged
by the divisibility `2 ∣ 2^n`. -/
theorem two_pow_mod_two {n : Nat} (hn : 0 < n) : 2 ^ n % 2 = 0 :=
  (Nat.dvd_iff_mod_eq_zero).mp (two_dvd_two_pow hn)

/-- **The ops map's O3 witness, closed at width 32.** `0` and `2` are congruent modulo `2`, and
their `SHR^1` values are `0` and `1`. Closed arithmetic only; no enumeration of the `2^32`
words. -/
theorem shr_o3_witness_32 :
    (0 : BitVec 32).toNat % 2 = (2 : BitVec 32).toNat % 2 ∧
      (tshrx (0 : BitVec 32) 1).toNat = 0 ∧ (tshrx (2 : BitVec 32) 1).toNat = 1 :=
  ⟨rfl, rfl, rfl⟩

/-- **The generalised O3 witness.** For every width `k > 0` and every amount `n` with `k ∤ n`, two
words congruent modulo `2` have `SHR^n` values that are not congruent modulo `2`. The witness is
`2^(n mod k)` and `0`: the first is even, and its low `m` bits are the whole word, so it shifts to
`1`, while the second shifts to `0`. -/
theorem shr_not_congruence_two {k n : Nat} (hk : 0 < k) (hne : n % k ≠ 0) :
    ¬ ∀ a b : BitVec k, a.toNat % 2 = b.toNat % 2 →
      (tshrx a n).toNat % 2 = (tshrx b n).toNat % 2 := by
  intro h
  have hm : 0 < n % k := Nat.pos_of_ne_zero hne
  have hlt := two_pow_lt_two_pow (k := k) (m := n % k) (Nat.mod_lt (x := n) (y := k) hk)
  have ha : (⟨2 ^ (n % k), hlt⟩ : BitVec k).toNat = 2 ^ (n % k) := bitVec_mk_toNat hlt
  have ha1 : (tshrx (⟨2 ^ (n % k), hlt⟩ : BitVec k) n).toNat = 1 := by
    rw [tshrx_toNat, ha, Nat.div_self (Nat.two_pow_pos (n % k))]
  have hz0 : tshrx (0 : BitVec k) n = 0 := by simp [tshrx]
  have ha0 : (tshrx (0 : BitVec k) n).toNat = 0 := by rw [hz0]; exact bitVec_zero_toNat
  have hc := h (⟨2 ^ (n % k), hlt⟩ : BitVec k) 0
    (by rw [ha, bitVec_zero_toNat, two_pow_mod_two hm, Nat.zero_mod])
  have hc' : ((⟨2 ^ (n % k), hlt⟩ : BitVec k).toNat / 2 ^ (n % k)) % 2
      = (((0 : BitVec k).toNat / 2 ^ (n % k)) % 2) := by
    simpa only [tshrx_toNat] using hc
  rw [ha, bitVec_zero_toNat, Nat.div_self (Nat.two_pow_pos (n % k)), Nat.zero_div] at hc'
  omega

/-- **The characterisation: the shift preserves congruence modulo `2` exactly when the amount is a
multiple of the width.** The forward direction is `n % k = 0`, where the shift is the identity;
the reverse is the witness above. So the set of amounts at which the shift is a function of the
word's residue class modulo `2` is exactly `{n | k ∣ n}`. -/
theorem shr_congruence_two_iff {k n : Nat} (hk : 0 < k) :
    (∀ a b : BitVec k, a.toNat % 2 = b.toNat % 2 →
      (tshrx a n).toNat % 2 = (tshrx b n).toNat % 2) ↔ n % k = 0 := by
  constructor
  · intro h
    by_contra hcon
    exact shr_not_congruence_two hk hcon h
  · intro h a b hab
    have h0 : n % k = 0 := h
    rw [show tshrx a n = a from by rw [tshrx, h0, BitVec.ushiftRight_zero]]
    rw [show tshrx b n = b from by rw [tshrx, h0, BitVec.ushiftRight_zero]]
    exact hab

/-! ## §4 Not a ring polynomial of any degree

A ring polynomial over `ZMod (2^k)` has a residue modulo `2` determined by the word modulo `2`,
because `ZMod 2^k →+* ZMod 2` is a ring hom (`2 ∣ 2^k`). The shift is not congruence preserving
modulo `2` by §3, so **no** polynomial over `ZMod (2^k)`, of any degree, agrees with it on all
`2^k` words. This is the general form of the ops map's O3, which was stated for polynomials but
never elaborated. -/

/-- **Reduction of a word's ring value modulo `2`, as an element of `ZMod 2`.** This is the
composite with the canonical `ZMod.castHom` along `2 ∣ 2^k`, computed on `.val`. -/
def par2 {k : Nat} (hk : 0 < k) (u : ZMod (2 ^ k)) : ZMod 2 := ((u.val % 2 : ℕ) : ZMod 2)

theorem par2_zero {k : Nat} (hk : 0 < k) : par2 hk 0 = 0 := by
  rw [par2, ZMod.val_zero, Nat.zero_mod, Nat.cast_zero]

theorem par2_one {k : Nat} (hk : 0 < k) : par2 hk 1 = 1 := by
  haveI : Fact (1 < 2 ^ k) := ⟨Nat.one_lt_two_pow (Nat.ne_of_gt hk)⟩
  rw [par2, ZMod.val_one, Nat.mod_eq_of_lt (by decide)]
  decide

theorem par2_mul {k : Nat} (hk : 0 < k) (u v : ZMod (2 ^ k)) :
    par2 hk (u * v) = par2 hk u * par2 hk v := by
  have hv : (u * v).val = u.val * v.val % 2 ^ k := ZMod.val_mul u v
  have e1 : ((u * v).val % 2 : ℕ) = (u.val * v.val : ℕ) % 2 := by
    rw [hv, Nat.mod_mod_of_dvd (a := u.val * v.val) (b := 2 ^ k) (c := 2)
      (two_dvd_two_pow (by omega))]
  have e2 : (((u.val % 2 : ℕ) : ZMod 2) * ((v.val % 2 : ℕ) : ZMod 2) : ZMod 2)
      = (((u.val % 2) * (v.val % 2) : ℕ) : ZMod 2) := (Nat.cast_mul _ _).symm
  have e3 : ((u.val % 2) * (v.val % 2) : ℕ) = (u.val * v.val : ℕ) % 2 := by
    rw [Nat.mul_mod]
    have hu1 : u.val % 2 ≤ 1 := by
      rcases Nat.mod_two_eq_zero_or_one u.val with hu | hu <;> omega
    have hv1 : v.val % 2 ≤ 1 := by
      rcases Nat.mod_two_eq_zero_or_one v.val with hv2 | hv2 <;> omega
    have hle : u.val % 2 * (v.val % 2) < 2 := by
      have h := Nat.mul_le_mul hu1 hv1
      rw [Nat.mul_one] at h
      omega
    exact (Nat.mod_eq_of_lt hle).symm
  rw [par2, par2, par2, e1, e2, e3]

theorem par2_add {k : Nat} (hk : 0 < k) (u v : ZMod (2 ^ k)) :
    par2 hk (u + v) = par2 hk u + par2 hk v := by
  have hv : (u + v).val = (u.val + v.val) % 2 ^ k := ZMod.val_add u v
  have e1 : ((u + v).val % 2 : ℕ) = ((u.val + v.val) : ℕ) % 2 := by
    rw [hv, Nat.mod_mod_of_dvd (a := u.val + v.val) (b := 2 ^ k) (c := 2)
      (two_dvd_two_pow (by omega))]
  have e2 : (↑((u.val % 2 + v.val % 2) % 2) : ZMod 2)
      = (↑(u.val % 2) : ZMod 2) + (↑(v.val % 2) : ZMod 2) := by
    rw [ZMod.natCast_mod (u.val % 2 + v.val % 2) 2, Nat.cast_add]
  rw [par2, par2, par2, e1]
  calc (↑((u.val + v.val) % 2) : ZMod 2) = (↑((u.val % 2 + v.val % 2) % 2) : ZMod 2) := by
        rw [Nat.add_mod]
    _ = (↑(u.val % 2) : ZMod 2) + (↑(v.val % 2) : ZMod 2) := e2

theorem par2_pow {k : Nat} (hk : 0 < k) (u : ZMod (2 ^ k)) (i : ℕ) :
    par2 hk (u ^ i) = (par2 hk u) ^ i := by
  induction i with
  | zero => simpa using (par2_one hk)
  | succ i ih =>
    rw [pow_succ, par2_mul, ih]
    exact (pow_succ (par2 hk u) i).symm

theorem par2_sum {k : Nat} (hk : 0 < k) (s : Finset ℕ) (f : ℕ → ZMod (2 ^ k)) :
    par2 hk (Finset.sum s f) = Finset.sum s (fun i => par2 hk (f i)) := by
  induction s using Finset.induction_on with
  | empty => rw [Finset.sum_empty, Finset.sum_empty, par2_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, Finset.sum_insert ha, par2_add, ih]

/-- **`par2` reads a natural number's parity.** The bridge between the ring and the `ℕ`-level
congruence statement of §3. -/
theorem par2_natCast {k : Nat} (hk : 0 < k) (a : ℕ) :
    par2 hk ((a : ℕ) : ZMod (2 ^ k)) = ((a % 2 : ℕ) : ZMod 2) := by
  rw [par2, ZMod.val_natCast]
  have h6 : (a % 2 ^ k) % 2 = a % 2 :=
    Nat.mod_mod_of_dvd (a := a) (b := 2 ^ k) (c := 2) (two_dvd_two_pow hk)
  rw [← h6]

/-- A ring-polynomial reading of a word operation at width `k`: the word's value maps to a finite
sum of coefficient times power, which is every polynomial over `ZMod 2^k` of any degree. -/
def RingPolyRead (k : Nat) (g : BitVec k → ZMod (2 ^ k)) : Prop :=
  ∃ (s : Finset ℕ) (c : ℕ → ZMod (2 ^ k)), ∀ x : BitVec k,
    g x = Finset.sum s (fun i => c i * (den x) ^ i)

/-- **A ring-polynomial reading depends on the word only through its residue modulo `2`.** -/
theorem par2_ringpoly {k : Nat} (hk : 0 < k) (s : Finset ℕ) (c : ℕ → ZMod (2 ^ k))
    (u v : ZMod (2 ^ k)) (huv : par2 hk u = par2 hk v) :
    par2 hk (Finset.sum s (fun i => c i * u ^ i))
      = par2 hk (Finset.sum s (fun i => c i * v ^ i)) := by
  rw [par2_sum hk s (fun i => c i * u ^ i), par2_sum hk s (fun i => c i * v ^ i)]
  exact Finset.sum_congr rfl fun i _ => by
    rw [par2_mul, par2_pow, par2_mul, par2_pow, huv]

/-- **The generalised O3, at every degree.** No polynomial over `ZMod (2^k)` — not only not the
degree-one two-slot family — agrees with `x >>> n` on all `2^k` words, at any face, for every
amount `n` with `k ∤ n`. -/
theorem shr_not_ringpoly {k n : Nat} (hk : 0 < k) (hne : n % k ≠ 0) :
    ¬ RingPolyRead k (fun x => den (tshrx x n)) := by
  rintro ⟨s, c, h⟩
  have hm : 0 < n % k := Nat.pos_of_ne_zero hne
  have hlt := two_pow_lt_two_pow (k := k) (m := n % k) (Nat.mod_lt (x := n) (y := k) hk)
  have ha : (⟨2 ^ (n % k), hlt⟩ : BitVec k).toNat = 2 ^ (n % k) := bitVec_mk_toNat hlt
  have hpar : par2 hk (den (⟨2 ^ (n % k), hlt⟩ : BitVec k)) = par2 hk (den 0) := by
    rw [den, den, ha, bitVec_zero_toNat, par2_natCast, par2_natCast, two_pow_mod_two hm]
  have hpoly := par2_ringpoly hk s c (den (⟨2 ^ (n % k), hlt⟩ : BitVec k)) (den 0) hpar
  have hL : par2 hk (den (tshrx (⟨2 ^ (n % k), hlt⟩ : BitVec k) n)) = (1 : ZMod 2) := by
    rw [den_tshrx, ha, Nat.div_self (Nat.two_pow_pos (n % k)), par2_natCast, Nat.one_mod,
      Nat.cast_one]
  have hR : par2 hk (den (tshrx (0 : BitVec k) n)) = (0 : ZMod 2) := by
    rw [den_tshrx, bitVec_zero_toNat, Nat.zero_div, par2_natCast, Nat.zero_mod, Nat.cast_zero]
  have heq : par2 hk (den (tshrx (⟨2 ^ (n % k), hlt⟩ : BitVec k) n))
      = par2 hk (den (tshrx (0 : BitVec k) n)) := by
    calc par2 hk (den (tshrx (⟨2 ^ (n % k), hlt⟩ : BitVec k) n))
        = par2 hk (Finset.sum s (fun i => c i * (den (⟨2 ^ (n % k), hlt⟩ : BitVec k)) ^ i)) :=
          congrArg _ (h (⟨2 ^ (n % k), hlt⟩ : BitVec k))
      _ = par2 hk (Finset.sum s (fun i => c i * (den 0) ^ i)) := hpoly
      _ = par2 hk (den (tshrx (0 : BitVec k) n)) := (congrArg _ (h 0)).symm
  rw [hL, hR] at heq
  exact absurd heq (by decide)

/-! ## §5 The exactness set of the family over the whole word space -/

/-- **The shift is nonzero at the all-ones word for every non-identity amount.** This is the
witness that rules out the zero multiplier. -/
theorem shr_read_top_ne_zero {k n : Nat} (hk : 0 < k) (hm : 0 < n % k) :
    den (tshrx (⟨2 ^ k - 1, by have h1 : 0 < 2 ^ k := Nat.two_pow_pos k; omega⟩ : BitVec k) n) ≠ 0 := by
  have htop : (⟨2 ^ k - 1, by have h1 : 0 < 2 ^ k := Nat.two_pow_pos k; omega⟩ : BitVec k).toNat
      = 2 ^ k - 1 := bitVec_top_toNat
  have hpos : 0 < 2 ^ k := Nat.two_pow_pos k
  have hsub0 : 2 ^ k - 1 < 2 ^ k := by omega
  have hQlt : (2 ^ k - 1) / 2 ^ (n % k) < 2 ^ k := Nat.div_lt_of_lt hsub0
  rw [den_tshrx, htop]
  intro hz
  have hdvd : 2 ^ k ∣ (2 ^ k - 1) / 2 ^ (n % k) :=
    (ZMod.natCast_eq_zero_iff (a := (2 ^ k - 1) / 2 ^ (n % k)) (b := 2 ^ k)).mp hz
  have hQ0 : (2 ^ k - 1) / 2 ^ (n % k) = 0 := Nat.eq_zero_of_dvd_of_lt hdvd hQlt
  have hmle : 2 ^ (n % k) ≤ 2 ^ k - 1 := by
    have h1 : 2 ^ (n % k) < 2 ^ k :=
      two_pow_lt_two_pow (k := k) (m := n % k) (Nat.mod_lt (x := n) (y := k) hk)
    omega
  have hQ1 : 1 ≤ (2 ^ k - 1) / 2 ^ (n % k) :=
    (Nat.le_div_iff_mul_le (k := 2 ^ (n % k)) (x := 1) (y := 2 ^ k - 1)
      (Nat.two_pow_pos (n % k))).2 (by simpa using hmle)
  omega

/-- **No member of the two-slot family is the shift, at any width, at any face.** The witness is
the pair of words `1` and `2^k − 1`: agreement at `1` forces the multiplier to be `0`, because
`1 >>> n = 0` whenever `k ∤ n`, and the zero multiplier fails at `2^k − 1`. This generalises
AR-12's width-2 and width-4 instances `tshrx_not_family_ZMod4`, `tshrx_not_family_ZMod16`. -/
theorem shr_not_family {k n : Nat} (hk : 0 < k) (hne : n % k ≠ 0) :
    ¬ ∃ (σ r a b : ZMod (2 ^ k)), ∀ x : BitVec k,
      twoSlotRead σ r a b x = den (tshrx x n) := by
  rintro ⟨σ, r, a, b, hfam⟩
  have hm : 0 < n % k := Nat.pos_of_ne_zero hne
  have hone : (⟨1, Nat.one_lt_two_pow (Nat.ne_of_gt hk)⟩ : BitVec k).toNat = 1 :=
    bitVec_mk_toNat (Nat.one_lt_two_pow (Nat.ne_of_gt hk))
  have h1zero : den (tshrx (⟨1, Nat.one_lt_two_pow (Nat.ne_of_gt hk)⟩ : BitVec k) n)
      = (0 : ZMod (2 ^ k)) := by
    rw [den_tshrx, hone]
    simp [Nat.mod_eq_of_lt (a := 1) (b := 2 ^ k) (Nat.one_lt_two_pow (Nat.ne_of_gt hk)),
      Nat.div_eq_of_lt (Nat.one_lt_two_pow (Nat.ne_of_gt hm))]
  have h1 := hfam (⟨1, Nat.one_lt_two_pow (Nat.ne_of_gt hk)⟩ : BitVec k)
  rw [twoSlotRead_eq, h1zero] at h1
  have hvden1 : (den (⟨1, Nat.one_lt_two_pow (Nat.ne_of_gt hk)⟩ : BitVec k)).val = 1 := by
    rw [den, ZMod.val_natCast, hone]
    simp [Nat.mod_eq_of_lt (a := 1) (b := 2 ^ k) (Nat.one_lt_two_pow (Nat.ne_of_gt hk))]
  have hv := congrArg ZMod.val h1
  rw [ZMod.val_mul, ZMod.val_zero, hvden1, Nat.one_mul] at hv
  have hdvd : 2 ^ k ∣ (a + r * b).val := (Nat.dvd_iff_mod_eq_zero).mpr hv
  have hc0 : a + r * b = 0 := by
    rw [← natCast_val_eq (a + r * b),
      Nat.eq_zero_of_dvd_of_lt hdvd (ZMod.val_lt (a + r * b)), Nat.cast_zero]
  have htop := shr_read_top_ne_zero hk hm
  have h2 := hfam (⟨2 ^ k - 1, by have h1 : 0 < 2 ^ k := Nat.two_pow_pos k; omega⟩ : BitVec k)
  rw [twoSlotRead_eq, hc0, mul_zero] at h2
  exact absurd h2.symm htop

/-- **The exactness set of the shift over the whole word space is exactly `{n | k ∣ n}`.** This is
the shift's counterpart of the rotation's exactness set: an amount whose residue is `0` makes the
shift the identity, which the two-slot expression with multiplier `1` reads exactly, and every
other amount is refuted above. -/
theorem shr_exact_amounts {k : Nat} (hk : 0 < k) :
    {n : ℕ | ∃ (σ r a b : ZMod (2 ^ k)), ∀ x : BitVec k,
      twoSlotRead σ r a b x = den (tshrx x n)} = {n : ℕ | k ∣ n} := by
  ext n
  constructor
  · rintro ⟨σ, r, a, b, hfam⟩
    by_cases hz : n % k = 0
    · exact Nat.dvd_of_mod_eq_zero hz
    · exfalso
      exact shr_not_family hk hz ⟨σ, r, a, b, hfam⟩
  · rintro ⟨a, ha⟩
    have hdvd : k ∣ n := ⟨a, ha⟩
    have h0 : n % k = 0 := Nat.mod_eq_zero_of_dvd hdvd
    refine ⟨(0 : ZMod (2 ^ k)), 0, 1, 0, ?_⟩
    intro x
    calc twoSlotRead (0 : ZMod (2 ^ k)) 0 1 0 x = den x * (1 + 0 * 0) := twoSlotRead_eq _ _ _ _ _
      _ = den x := by ring
      _ = den (tshrx x n) := by rw [tshrx, h0, BitVec.ushiftRight_zero]

/-! ## §6 The partial exactness, in closed form -/

/-- **The exact agreement set of the zero expression.** `den (x >>> n) = 0` holds exactly on the
words whose value is below `2^(n mod k)`, that is, the words with no bit at or above position
`n mod k`. This is the shift's own vanishing set, the counterpart of AR-16's
`rotExpr32_offset_eq_zero_iff`. -/
theorem shr_zero_iff {k : Nat} (hk : 0 < k) (x : BitVec k) (n : Nat) :
    den (tshrx x n) = 0 ↔ x.toNat < 2 ^ (n % k) := by
  rw [den_tshrx]
  constructor
  · intro hz
    have hdvd : 2 ^ k ∣ x.toNat / 2 ^ (n % k) :=
      (ZMod.natCast_eq_zero_iff (a := x.toNat / 2 ^ (n % k)) (b := 2 ^ k)).mp hz
    have hQlt : x.toNat / 2 ^ (n % k) < 2 ^ k := Nat.div_lt_of_lt x.isLt
    have hQ0 : x.toNat / 2 ^ (n % k) = 0 := Nat.eq_zero_of_dvd_of_lt hdvd hQlt
    have hlt1 : x.toNat / 2 ^ (n % k) < 1 := by omega
    have h1 := (Nat.div_lt_iff_lt_mul (k := 2 ^ (n % k)) (x := x.toNat) (y := 1)
      (Nat.two_pow_pos (n % k))).mp hlt1
    simpa using h1
  · intro hlt
    rw [Nat.div_eq_of_lt hlt]
    exact Nat.cast_zero

/-- **The zero two-slot expression's agreement set is exactly `{x | x < 2^(n mod k)}`**, stated on
the expression itself. -/
theorem twoSlotRead_zero_iff {k : Nat} (hk : 0 < k) (σ r : ZMod (2 ^ k)) (x : BitVec k) (n : Nat) :
    twoSlotRead σ r 0 0 x = den (tshrx x n) ↔ x.toNat < 2 ^ (n % k) := by
  rw [twoSlotRead_zero, eq_comm]
  exact shr_zero_iff hk x n

/-- **The shift's agreement with a family member, in divisibility form.** With `x = 2^m·q + r`
(`r` the low `m` bits, `q` the rest), agreement of `c·x` with `x >>> m` forces
`2^k ∣ q·(2^m·c − 1)`. This is the closed form of the partial exactness modulo `2^k`, and the only
form §6's two refutations need. -/
theorem shr_agree_dvd {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m) (hmk : m < k)
    (c : ZMod (2 ^ k)) (x : BitVec k) (h : c * den x = den (tshrx x m)) :
    ∃ t : ℕ, c.val * x.toNat = x.toNat / 2 ^ m + 2 ^ k * t := by
  have hv : (c * den x).val = (den (tshrx x m)).val := congrArg ZMod.val h
  rw [ZMod.val_mul, den, ZMod.val_natCast_of_lt x.isLt] at hv
  rw [den_tshrx, ZMod.val_natCast_of_lt (Nat.div_lt_of_lt x.isLt)] at hv
  rw [Nat.mod_eq_of_lt hmk] at hv
  refine ⟨c.val * x.toNat / 2 ^ k, ?_⟩
  have heq := (Nat.mod_add_div (c.val * x.toNat) (2 ^ k)).symm
  omega

/-- `A` even and at least `2` forces `A − 1` odd. -/
theorem odd_of_even {A : ℕ} (hA : A % 2 = 0) (hApos : 2 ≤ A) : (A - 1) % 2 = 1 := by
  have hdiv : A % 2 + 2 * (A / 2) = A := Nat.mod_add_div A 2
  rw [hA, Nat.zero_add] at hdiv
  obtain ⟨q, hq⟩ : ∃ q, A = 2 * q := ⟨A / 2, hdiv.symm⟩
  obtain ⟨p, hp⟩ : ∃ p, q = p + 1 := ⟨q - 1, by omega⟩
  rw [hq, hp]
  have heq : 2 * (p + 1) - 1 = 2 * p + 1 := by omega
  rw [heq, Nat.add_mod, Nat.mul_mod_right, Nat.zero_add, Nat.mod_eq_of_lt (by decide),
    Nat.mod_eq_of_lt (by decide)]

/-- An odd number is coprime to `2`. -/
theorem coprime_two_of_odd {w : ℕ} (hw : w % 2 = 1) : Nat.Coprime 2 w := by
  show Nat.gcd 2 w = 1
  have hd2 : Nat.gcd 2 w ∣ 2 := Nat.gcd_dvd_left 2 w
  have hdw : Nat.gcd 2 w ∣ w := Nat.gcd_dvd_right 2 w
  have hnw : ¬ (2 ∣ w) := fun h => absurd ((Nat.dvd_iff_mod_eq_zero).mp h) (by omega)
  have hcases : Nat.gcd 2 w = 0 ∨ Nat.gcd 2 w = 1 ∨ Nat.gcd 2 w = 2 := by
    have hle : Nat.gcd 2 w ≤ 2 := Nat.le_of_dvd (by decide) hd2
    omega
  rcases hcases with h0 | h1 | h2
  · obtain ⟨c, hc⟩ := hd2
    rw [h0, Nat.zero_mul] at hc
    omega
  · exact h1
  · exact absurd (h2 ▸ hdw) hnw

/-- **The cancellation.** `2^k ∣ q·w` with `w` odd forces `2^k ∣ q`: an odd number is coprime to a
power of two. This is the only number-theoretic step §6's two refutations use. -/
theorem dvd_pow_two_of_dvd_mul_odd {k q w : ℕ} (hw : w % 2 = 1) (hd : 2 ^ k ∣ q * w) :
    2 ^ k ∣ q :=
  (Nat.Coprime.pow_left k (coprime_two_of_odd hw)).dvd_of_dvd_mul_left
    (by rwa [Nat.mul_comm] at hd)

/-- `a ∣ A` and `a ∣ B` and `B ≤ A` give `a ∣ A − B`. -/
theorem dvd_sub_of_dvd_of_le {a A B : ℕ} (hA : a ∣ A) (hB : a ∣ B) (h : B ≤ A) :
    a ∣ A - B := by
  obtain ⟨u, hu⟩ := hA
  obtain ⟨v, hv⟩ := hB
  refine ⟨u - v, ?_⟩
  rw [Nat.mul_sub, hu, hv]



/-! ## §8 The round verdict, re-read from the canonical bodies

FIPS 180-4 §4.1.3's `Σ₀` and `Σ₁`, the two composites the round function uses, are `⊕` of three
`Apeiron.Crypto.Sha2.rotr` and contain no `Apeiron.Crypto.Sha2.shrx` at all. The two `SHR` terms of §4.1.2 belong to `σ₀`
(`SHR^3`) and `σ₁` (`SHR^10`), which the message schedule uses (`Sha2.lean:87-88`) and the round does
not. -/

/-- `Apeiron.Crypto.Sha2.Sigma0` is a `⊕` of three `Apeiron.Crypto.Sha2.rotr`, with no `Apeiron.Crypto.Sha2.shrx` term. `rfl` on the canonical
body (`Sha2.lean:67`). -/
theorem Sha2_Sigma0_body (x : Apeiron.Crypto.Sha2.Word) :
    Apeiron.Crypto.Sha2.Sigma0 x = Apeiron.Crypto.Sha2.rotr x 2 ^^^ Apeiron.Crypto.Sha2.rotr x 13 ^^^ Apeiron.Crypto.Sha2.rotr x 22 := rfl

/-- `Apeiron.Crypto.Sha2.Sigma1` is a `⊕` of three `Apeiron.Crypto.Sha2.rotr`, with no `Apeiron.Crypto.Sha2.shrx` term. `rfl` (`Sha2.lean:70`). -/
theorem Sha2_Sigma1_body (x : Apeiron.Crypto.Sha2.Word) :
    Apeiron.Crypto.Sha2.Sigma1 x = Apeiron.Crypto.Sha2.rotr x 6 ^^^ Apeiron.Crypto.Sha2.rotr x 11 ^^^ Apeiron.Crypto.Sha2.rotr x 25 := rfl

/-- `Apeiron.Crypto.Sha2.sigma0` carries the single `SHR^3` of FIPS 180-4 §4.1.2. `rfl` (`Sha2.lean:60`). -/
theorem Sha2_sigma0_body (x : Apeiron.Crypto.Sha2.Word) :
    Apeiron.Crypto.Sha2.sigma0 x = Apeiron.Crypto.Sha2.rotr x 7 ^^^ Apeiron.Crypto.Sha2.rotr x 18 ^^^ Apeiron.Crypto.Sha2.shrx x 3 := rfl

/-- `Apeiron.Crypto.Sha2.sigma1` carries the single `SHR^10` of FIPS 180-4 §4.1.2. `rfl` (`Sha2.lean:64`). -/
theorem Sha2_sigma1_body (x : Apeiron.Crypto.Sha2.Word) :
    Apeiron.Crypto.Sha2.sigma1 x = Apeiron.Crypto.Sha2.rotr x 17 ^^^ Apeiron.Crypto.Sha2.rotr x 19 ^^^ Apeiron.Crypto.Sha2.shrx x 10 := rfl

/-- Neither FIPS shift amount is a multiple of the word width. -/
theorem fips_shr_amounts_not_multiples : ¬ (32 ∣ 3) ∧ ¬ (32 ∣ 10) := ⟨by decide, by decide⟩

/-- **`SHR^3` has no two-slot reading at the SHA-256 word width**, at any face. -/
theorem shr32_3_not_family :
    ¬ ∃ (σ r a b : ZMod (2 ^ 32)), ∀ x : BitVec 32,
      twoSlotRead σ r a b x = den (tshrx x 3) :=
  shr_not_family (k := 32) (n := 3) (by decide) (by decide)

/-- **`SHR^10` has no two-slot reading at the SHA-256 word width**, at any face. -/
theorem shr32_10_not_family :
    ¬ ∃ (σ r a b : ZMod (2 ^ 32)), ∀ x : BitVec 32,
      twoSlotRead σ r a b x = den (tshrx x 10) :=
  shr_not_family (k := 32) (n := 10) (by decide) (by decide)

/-- **`SHR^3` is not a ring polynomial of any degree over `ZMod 2^32`.** -/
theorem shr32_3_not_ringpoly : ¬ RingPolyRead 32 (fun x => den (tshrx x 3)) :=
  shr_not_ringpoly (k := 32) (n := 3) (by decide) (by decide)

/-- **`SHR^10` is not a ring polynomial of any degree over `ZMod 2^32`.** -/
theorem shr32_10_not_ringpoly : ¬ RingPolyRead 32 (fun x => den (tshrx x 10)) :=
  shr_not_ringpoly (k := 32) (n := 10) (by decide) (by decide)

/-! ## §7 The composition with the rotation read

The parabolic rotation read is one `ringUpper`/`ringMul` whose multiplier carries `2^(k−n)` in the
first slot. What stands between that read and the rotation is exactly one `SHR`; the residual is
named and its zero set is the shift's own exactness set from §6. -/

/-- **Regime I.** For `0 < n < k` the corpus's `rotr` body is `BitVec.rotateRight`. -/
theorem trotr_eq_rotateRight {k : Nat} (x : BitVec k) (n : Nat) (hn : 0 < n) (hnk : n < k) :
    trotr x n = x.rotateRight n := by
  have h1 : n % k = n := Nat.mod_eq_of_lt hnk
  have h2 : (k - n) % k = k - n := Nat.mod_eq_of_lt (by omega)
  simp only [trotr, h1, h2, BitVec.rotateRight_def]

/-- **The value of a rotation.** The unsigned value of `rotr x n` is the truncated right shift plus
the wrapped low bits, `⌊x/2^n⌋ + (x mod 2^n)·2^(k−n)`. -/
theorem trotr_toNat_sum {k : Nat} (x : BitVec k) (n : Nat) (hn : 0 < n) (hmk : n < k) :
    (trotr x n).toNat = x.toNat / 2 ^ n + (x.toNat % 2 ^ n) * 2 ^ (k - n) := by
  have hxpow : 2 ^ n * 2 ^ (k - n) = 2 ^ k := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hwrap : (x.toNat % 2 ^ n) * 2 ^ (k - n) < 2 ^ k := by
    calc (x.toNat % 2 ^ n) * 2 ^ (k - n)
        < 2 ^ n * 2 ^ (k - n) :=
      Nat.mul_lt_mul_of_pos_right (Nat.mod_lt x.toNat (Nat.two_pow_pos n)) (Nat.two_pow_pos (k - n))
      _ = 2 ^ k := hxpow
  have hkey : x.toNat * 2 ^ (k - n)
      = x.toNat / 2 ^ n * 2 ^ k + (x.toNat % 2 ^ n) * 2 ^ (k - n) := by
    have hdiv := Nat.mod_add_div x.toNat (2 ^ n)
    calc x.toNat * 2 ^ (k - n)
        = (x.toNat % 2 ^ n + 2 ^ n * (x.toNat / 2 ^ n)) * 2 ^ (k - n) := by rw [hdiv]
      _ = (x.toNat % 2 ^ n) * 2 ^ (k - n) + (x.toNat / 2 ^ n) * (2 ^ n * 2 ^ (k - n)) := by
        rw [Nat.add_mul, Nat.mul_comm (2 ^ n) (x.toNat / 2 ^ n), Nat.mul_assoc]
      _ = x.toNat / 2 ^ n * 2 ^ k + (x.toNat % 2 ^ n) * 2 ^ (k - n) := by
        rw [hxpow, Nat.add_comm]
  have hA : x.toNat <<< (k - n) % 2 ^ k = (x.toNat % 2 ^ n) * 2 ^ (k - n) := by
    rw [Nat.shiftLeft_eq, hkey, Nat.mul_add_mod', Nat.mod_eq_of_lt hwrap]
  have hq : x.toNat / 2 ^ n < 2 ^ (k - n) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos n), mul_comm (2 ^ (k - n)) (2 ^ n), hxpow]
    exact x.isLt
  rw [trotr_eq_rotateRight x n hn hmk, BitVec.toNat_rotateRight, Nat.mod_eq_of_lt hmk,
    Nat.shiftRight_eq_div_pow, hA]
  have hor := Nat.two_pow_add_eq_or_of_lt hq (x.toNat % 2 ^ n)
  rw [mul_comm (x.toNat % 2 ^ n) (2 ^ (k - n)), Nat.or_comm, ← hor, add_comm]

/-- A multiple of the modulus reads as `0`. -/
theorem mul_two_pow_cast_zero {k : Nat} (q : ℕ) : ((q * 2 ^ k : ℕ) : ZMod (2 ^ k)) = 0 := by
  rw [ZMod.natCast_eq_zero_iff]
  exact ⟨q, Nat.mul_comm _ _⟩

/-- Multiplying the word by `2^(k−n)` and reading into `ZMod 2^k` sees only the low `n` bits,
lifted: the wrapped half of the rotation. -/
theorem mul_pow_cast {k : Nat} (a : ℕ) (n : Nat) (hn : n ≤ k) :
    ((a * 2 ^ (k - n) : ℕ) : ZMod (2 ^ k)) = ((a % 2 ^ n * 2 ^ (k - n) : ℕ) : ZMod (2 ^ k)) := by
  have hxpow : 2 ^ n * 2 ^ (k - n) = 2 ^ k := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hdiv := Nat.mod_add_div a (2 ^ n)
  have h1 : (a * 2 ^ (k - n) : ℕ) = (a % 2 ^ n + 2 ^ n * (a / 2 ^ n)) * 2 ^ (k - n) := by rw [hdiv]
  have h2 : (a % 2 ^ n + 2 ^ n * (a / 2 ^ n)) * 2 ^ (k - n)
      = (a % 2 ^ n) * 2 ^ (k - n) + (a / 2 ^ n) * (2 ^ n * 2 ^ (k - n)) := by
    rw [Nat.add_mul, Nat.mul_comm (2 ^ n) (a / 2 ^ n), Nat.mul_assoc]
  have h3 : (a / 2 ^ n) * (2 ^ n * 2 ^ (k - n)) = (a / 2 ^ n) * 2 ^ k := by rw [hxpow]
  rw [h1, h2]
  simp only [Nat.cast_mul, Nat.cast_add]
  rw [← (Nat.cast_mul (2 ^ n) (2 ^ (k - n))), hxpow,
    ← (Nat.cast_mul (a / 2 ^ n) (2 ^ k)), mul_two_pow_cast_zero, add_zero]

/-- **The parabolic rotation read**: the two-slot expression with multiplier `2^(k−n)` in the first
slot and `0` in the second, the shift factor of the ops map's S7 candidate. -/
def rotParRead {k : Nat} (x : BitVec k) (n : Nat) : ZMod (2 ^ k) :=
  twoSlotRead 0 0 ((2 ^ (k - n) : ℕ)) 0 x

/-- The parabolic read is one multiplication by `2^(k−n)`, so it is a member of the family and
independent of the face. -/
theorem rotParRead_eq {k : Nat} (x : BitVec k) (n : Nat) :
    rotParRead x n = den x * ((2 ^ (k - n) : ℕ) : ZMod (2 ^ k)) := by
  rw [rotParRead, twoSlotRead_eq]
  ring

/-- **The composition, with the exact residual: parabolic rotation read + one `SHR` = the
rotation.** AR-12's `rotCell_parabolic_offset`, re-proved at width-generic `k`; it holds at every
face because the read is a member of the σ-independent family of §2. -/
theorem rotPar_plus_shr {k : Nat} (x : BitVec k) (n : Nat) (hn : 0 < n) (hmk : n < k) :
    rotParRead x n + den (tshrx x n) = den (trotr x n) := by
  have hm := mul_pow_cast x.toNat n (Nat.le_of_lt hmk)
  have hm' : den x * ((2 ^ (k - n) : ℕ) : ZMod (2 ^ k))
      = ((x.toNat % 2 ^ n * 2 ^ (k - n) : ℕ) : ZMod (2 ^ k)) := by
    rw [den, ← Nat.cast_mul]
    exact hm
  have htr : (trotr x n).toNat = x.toNat / 2 ^ n + (x.toNat % 2 ^ n) * 2 ^ (k - n) :=
    trotr_toNat_sum x n hn hmk
  rw [rotParRead_eq, hm', den_tshrx, Nat.mod_eq_of_lt hmk, den, htr, Nat.cast_add]
  ring

/-- **The residual is exactly one `SHR`**, named: it is the whole difference between the two-slot
rotation read and the rotation. -/
theorem rotPar_residual {k : Nat} (x : BitVec k) (n : Nat) (hn : 0 < n) (hmk : n < k) :
    den (trotr x n) - rotParRead x n = den (tshrx x n) := by
  rw [← rotPar_plus_shr x n hn hmk]
  ring

/-- **The residual vanishes exactly on the shift's exactness set** `{x | x < 2^(n mod k)}`: the
parabolic rotation reading is exact with no residual on that structured set and nowhere else. -/
theorem rotPar_residual_zero_iff {k : Nat} (hk : 0 < k) (x : BitVec k) (n : Nat) :
    den (tshrx x n) = 0 ↔ x.toNat < 2 ^ (n % k) := shr_zero_iff hk x n

/-! ## Axiom cone -/

#print axioms AlgebraShrExact.neZero_two_pow
#print axioms AlgebraShrExact.two_dvd_two_pow
#print axioms AlgebraShrExact.two_pow_lt_two_pow
#print axioms AlgebraShrExact.bitVec_mk_toNat
#print axioms AlgebraShrExact.bitVec_zero_toNat
#print axioms AlgebraShrExact.bitVec_top_toNat
#print axioms AlgebraShrExact.tshrx_toNat
#print axioms AlgebraShrExact.den_tshrx
#print axioms AlgebraShrExact.ringUpper_mul_eval
#print axioms AlgebraShrExact.twoSlotRead_eq
#print axioms AlgebraShrExact.twoSlotRead_sigma_free
#print axioms AlgebraShrExact.twoSlotRead_zero
#print axioms AlgebraShrExact.natCast_val_eq
#print axioms AlgebraShrExact.two_pow_mod_two
#print axioms AlgebraShrExact.shr_o3_witness_32
#print axioms AlgebraShrExact.shr_not_congruence_two
#print axioms AlgebraShrExact.shr_congruence_two_iff
#print axioms AlgebraShrExact.par2_zero
#print axioms AlgebraShrExact.par2_one
#print axioms AlgebraShrExact.par2_mul
#print axioms AlgebraShrExact.par2_add
#print axioms AlgebraShrExact.par2_pow
#print axioms AlgebraShrExact.par2_sum
#print axioms AlgebraShrExact.par2_natCast
#print axioms AlgebraShrExact.par2_ringpoly
#print axioms AlgebraShrExact.shr_not_ringpoly
#print axioms AlgebraShrExact.shr_read_top_ne_zero
#print axioms AlgebraShrExact.shr_not_family
#print axioms AlgebraShrExact.shr_exact_amounts
#print axioms AlgebraShrExact.shr_zero_iff
#print axioms AlgebraShrExact.twoSlotRead_zero_iff
#print axioms AlgebraShrExact.shr_agree_dvd
#print axioms AlgebraShrExact.odd_of_even
#print axioms AlgebraShrExact.coprime_two_of_odd
#print axioms AlgebraShrExact.dvd_pow_two_of_dvd_mul_odd
#print axioms AlgebraShrExact.dvd_sub_of_dvd_of_le
#print axioms AlgebraShrExact.Sha2_Sigma0_body
#print axioms AlgebraShrExact.Sha2_Sigma1_body
#print axioms AlgebraShrExact.Sha2_sigma0_body
#print axioms AlgebraShrExact.Sha2_sigma1_body
#print axioms AlgebraShrExact.fips_shr_amounts_not_multiples
#print axioms AlgebraShrExact.shr32_3_not_family
#print axioms AlgebraShrExact.shr32_10_not_family
#print axioms AlgebraShrExact.shr32_3_not_ringpoly
#print axioms AlgebraShrExact.shr32_10_not_ringpoly

#print axioms AlgebraShrExact.trotr_eq_rotateRight
#print axioms AlgebraShrExact.trotr_toNat_sum
#print axioms AlgebraShrExact.mul_two_pow_cast_zero
#print axioms AlgebraShrExact.mul_pow_cast
#print axioms AlgebraShrExact.rotParRead_eq
#print axioms AlgebraShrExact.rotPar_plus_shr
#print axioms AlgebraShrExact.rotPar_residual
#print axioms AlgebraShrExact.rotPar_residual_zero_iff

end AlgebraShrExact
