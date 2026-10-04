/-
  `AlgebraRotExact.lean` — the set of 32-bit words on which the two-slot algebraic rotation
  reading is EXACT, in closed form, at every amount `0 < n < 32`.

  Task AR-21 of the algebra-rooting wave. Owned files: this one and `AlgebraRotExact.md`. No
  production, pin, extraction-root or API edit; no commit.

  Nothing here is imported from `AlgebraRotate32.lean` (AR-16): that file is in flight and its last
  recorded run is red (`/tmp/opencode/alg-rot32-run3.log`, 30 errors, `sorryAx` in the offset
  chain). Every step of the offset is re-proved here from the corpus primitives. The AR-16 file
  read was sha256 `b638f3c606c294bb8917952d6104848b48eabc6c46035f32f1798aee946f0f62`, 726 lines.

  Proved here, in order.

  §1 The 32-bit carrier, the ops-map S7 two-slot expression `rotExpr32`, and what it computes.
  §2 The rotation value at width 32, the offset, and the exactness predicate `Exact32`.
  §3 The closed form: `Exact32 n x` iff the base-`2^(64−2n)` digits of `x` repeat, for every
      `0 < n < 32`.
  §4 Two regimes with `Fin`-indexed parametrisations: `n ≤ 16` (low `n` bits zero) and
      `17 ≤ n ≤ 21` (the digit family `b·2^(64−2n) + b`), plus the triviality boundary.
  §5 SHA-2: the ten rotation amounts and the six named amounts, with sizes and concrete members.
  §6 Invariance: entry, exit and confinement at the amount `rotr 22`, at one, two and four round
      steps; the general case is marked open, not claimed.
  §7 The joint statement with the attack lab's algebraic-failure set.
  §8 `#print axioms` for every headline.

  Grades: `proved` for a named theorem whose `#print axioms` row is inside
  `{propext, Classical.choice, Quot.sound}`. Nothing here is a security, collision-resistance,
  constant-time or break claim.

  The rotation amounts are written as the *left* rotation amount `n`. FIPS 180-4's `rotr k` is
  `rotl (32 − k)`, so `rotateRight_eq_rotateLeft32` converts and the tables name both.
-/

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Mathlib.Data.BitVec
import Mathlib.Data.ZMod.Basic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace AlgebraRotExact

open Apeiron.Trichotomy.AnyRing

/-! ## §1 The 32-bit carrier and the two-slot expression -/

/-- **Monotonicity of `2^·`**, used everywhere below in place of `omega`, which does not reason
about powers. -/
theorem two_pow_mono {a b : ℕ} (h : a ≤ b) : (2 : ℕ) ^ a ≤ 2 ^ b := by
  induction b, h using Nat.le_induction with
  | base => exact Nat.le_refl _
  | succ b _ ih =>
      rw [pow_succ, Nat.mul_two]
      exact Nat.le_trans ih (Nat.le_add_right _ _)

/-- `(2 ^ e : ℕ) * 2 ^ e = 2 ^ (2 * e)`. -/
theorem two_pow_sq (e : ℕ) : (2 ^ e : ℕ) * 2 ^ e = 2 ^ (2 * e) := by
  rw [← Nat.pow_add]; congr 1; omega

/-- `2 ^ (32 − n) * 2 ^ n = 2 ^ 32`, for `n < 32`. -/
theorem two_pow_add_32 (n : ℕ) (hn32 : n < 32) : (2 ^ (32 - n) : ℕ) * 2 ^ n = 2 ^ 32 := by
  rw [← Nat.pow_add]; congr 1; omega

/-- `2 ^ n * 2 ^ (32 − n) = 2 ^ 32`, for `n < 32`: the same identity read in the other order. -/
theorem two_pow_add_32' (n : ℕ) (hn32 : n < 32) : (2 ^ n : ℕ) * 2 ^ (32 - n) = 2 ^ 32 := by
  rw [← Nat.pow_add]; congr 1; omega

/-- The hyperbolic face's σ is `1` at the width SHA-2 uses; the root premise the two-slot reading
carries is then the arithmetic identity `1 ^ 2 = 1`, so no number theory is needed. -/
theorem ringUpper_mul_root32 : (1 : ZMod (2 ^ 32)) ^ 2 = 1 := rfl

/-- The reading of a product along the `x ↦ (x, 0)` line, in every commutative ring and so at
width 32. σ cancels before it is used. -/
theorem ringUpper_mul_eval {R : Type*} [CommRing R] (σ r a b x : R) :
    ringUpper r (ringMul σ (x, 0) (a, b)) = x * (a + r * b) := by
  simp only [ringUpper, ringMul]
  ring

/-- **The 32-bit carrier denotation**: the SHA-2 word read into `ZMod (2 ^ 32)`. -/
def bvDenote32 (x : BitVec 32) : ZMod (2 ^ 32) := (x.toNat : ZMod (2 ^ 32))

theorem bvDenote32_def (x : BitVec 32) : bvDenote32 x = (x.toNat : ZMod (2 ^ 32)) := rfl

/-- **The ops-map S7 candidate expression at the SHA-256 word width**:
`ringUpper 1 (ringMul 1 (x, 0) (2^n, 2^(32−n)))`. -/
def rotExpr32 (x n : Nat) : ZMod (2 ^ 32) :=
  ringUpper (1 : ZMod (2 ^ 32))
    (ringMul (1 : ZMod (2 ^ 32)) (((x : Nat) : ZMod (2 ^ 32)), (0 : ZMod (2 ^ 32)))
      (((2 ^ n : Nat) : ZMod (2 ^ 32)), ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))))

/-- **What the candidate is at width 32**: one ring multiplication by `2^n + 2^(32−n)`, for every
word value and every amount, with no hypothesis on either. The second slot cannot supply the
wrapped high bits. -/
theorem rotExpr32_eq (x n : Nat) : rotExpr32 x n
    = (x : ZMod (2 ^ 32))
      * (((2 ^ n : Nat) : ZMod (2 ^ 32)) + ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))) := by
  have h := ringUpper_mul_eval (1 : ZMod (2 ^ 32)) (1 : ZMod (2 ^ 32))
    ((2 ^ n : Nat) : ZMod (2 ^ 32)) ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))
    ((x : Nat) : ZMod (2 ^ 32))
  calc rotExpr32 x n
      = ringUpper (1 : ZMod (2 ^ 32))
          (ringMul (1 : ZMod (2 ^ 32)) (((x : Nat) : ZMod (2 ^ 32)), (0 : ZMod (2 ^ 32)))
            (((2 ^ n : Nat) : ZMod (2 ^ 32)), ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32)))) := rfl
    _ = (x : ZMod (2 ^ 32))
        * (((2 ^ n : Nat) : ZMod (2 ^ 32)) + (1 : ZMod (2 ^ 32))
          * ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))) := h
    _ = (x : ZMod (2 ^ 32))
        * (((2 ^ n : Nat) : ZMod (2 ^ 32)) + ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))) := by ring

/-! ## §2 The rotation value, the offset, and the exactness predicate -/

/-- **The rotation identity in `ℕ`, at width 32**: the rotate-left value is the truncated left
shift plus the wrapped high bits, for every `n < 32`. -/
theorem toNat_rotateLeft_eq_sum32 (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    (x.rotateLeft n).toNat = (x.toNat * 2 ^ n) % 2 ^ 32 + x.toNat / 2 ^ (32 - n) := by
  have hxpow := two_pow_add_32 n hn32
  have hA : (x.toNat * 2 ^ n) % 2 ^ 32 = 2 ^ n * (x.toNat % 2 ^ (32 - n)) := by
    have hmm := Nat.mul_mod_mul_left (2 ^ n) x.toNat (2 ^ (32 - n))
    rw [mul_comm (2 ^ n) x.toNat, mul_comm (2 ^ n) (2 ^ (32 - n)), hxpow] at hmm
    exact hmm
  have hq : x.toNat / 2 ^ (32 - n) < 2 ^ n := by
    refine (Nat.div_lt_iff_lt_mul (by positivity)).mpr ?_
    rw [mul_comm (2 ^ n) (2 ^ (32 - n)), two_pow_add_32 n hn32]
    exact x.isLt
  rw [BitVec.toNat_rotateLeft, Nat.mod_eq_of_lt hn32, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow,
    hA]
  exact (Nat.two_pow_add_eq_or_of_lt hq (x.toNat % 2 ^ (32 - n))).symm

/-- **The rotation identity at the SHA-256 word width.** -/
theorem bvDenote32_rotateLeft_eq (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    bvDenote32 (x.rotateLeft n)
      = bvDenote32 x * ((2 ^ n : ℕ) : ZMod (2 ^ 32))
        + ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) := by
  have h := congrArg (fun t : ℕ => (t : ZMod (2 ^ 32))) (toNat_rotateLeft_eq_sum32 x n hn32)
  rw [Nat.cast_add, ZMod.natCast_mod, Nat.cast_mul] at h
  simpa only [bvDenote32] using h

/-- **The offset at the word width**: what the two-slot expression computes less what the rotation
computes is `x·2^(32−n) − ⌊x / 2^(32−n)⌋`. -/
theorem rotExpr32_offset_eq (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    rotExpr32 x.toNat n - bvDenote32 (x.rotateLeft n)
      = bvDenote32 x * ((2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32))
        - ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) := by
  have h1 := rotExpr32_eq x.toNat n
  have h2 := bvDenote32_rotateLeft_eq x n hn32
  simp only [bvDenote32] at h1 h2 ⊢
  rw [h1, h2]
  ring

/-- **No power of two strictly below the modulus is zero at width 32.** -/
theorem two_pow_ne_zero32 (n : ℕ) (hn0 : 0 < n) (hn32 : n < 32) :
    ((2 ^ n : ℕ) : ZMod (2 ^ 32)) ≠ 0 := by
  intro h
  have hd : 2 ^ 32 ∣ 2 ^ n := (ZMod.natCast_eq_zero_iff (a := 2 ^ n) (b := 2 ^ 32)).mp h
  have hle : 2 ^ 32 ≤ 2 ^ n := Nat.le_of_dvd (by positivity) hd
  have hlt : 2 ^ n < 2 ^ 32 :=
    lt_of_le_of_lt (two_pow_mono (a := n) (b := 31) (by omega)) (by decide)
  exact absurd hle (not_le_of_gt hlt)

/-- **The exactness predicate at the word width**: the two-slot expression computes the rotation
exactly at `(n, x)`. This is the set the file characterises. -/
def Exact32 (n : ℕ) (x : BitVec 32) : Prop := rotExpr32 x.toNat n - bvDenote32 (x.rotateLeft n) = 0

/-- **The algebraic-failure predicate**: the two-slot expression does not compute the rotation. -/
def Fail32 (n : ℕ) (x : BitVec 32) : Prop := ¬ Exact32 n x

/-- **The exactness set is the complement of the failure set.** A state is either read exactly or
misread; never both, never neither. This is the identity §7's joint statement rests on. -/
theorem exact32_iff_not_fail32 (n : ℕ) (x : BitVec 32) : Exact32 n x ↔ ¬ Fail32 n x := by
  simp only [Fail32, Classical.not_not]

/-- **The decidable criterion at width 32**, for any word and any `n < 32`: the offset vanishes iff
the second slot `x·2^(32−n)` reduced modulo `2^32` equals the wrapped high bits `⌊x / 2^(32−n)⌋`. -/
theorem exact32_iff_offset_iff (x : BitVec 32) (n : ℕ) (hn32 : n < 32) :
    Exact32 n x ↔ (x.toNat * 2 ^ (32 - n)) % 2 ^ 32 = x.toNat / 2 ^ (32 - n) := by
  have hqlt : x.toNat / 2 ^ (32 - n) < 2 ^ 32 := by
    have h1 : x.toNat / 2 ^ (32 - n) ≤ x.toNat := Nat.div_le_self _ _
    have h2 : x.toNat < 2 ^ 32 := x.isLt
    have h3 : x.toNat / 2 ^ (32 - n) ≤ 2 ^ 32 - 1 := by omega
    exact lt_of_le_of_lt h3 (by decide)
  rw [Exact32, rotExpr32_offset_eq x n hn32]
  simp only [bvDenote32]
  constructor
  · intro h
    have h' : (x.toNat : ZMod (2 ^ 32)) * ((2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32))
        = (x.toNat / 2 ^ (32 - n) : ℕ) := sub_eq_zero.mp h
    rw [← Nat.cast_mul] at h'
    have h'' := (ZMod.natCast_eq_natCast_iff' _ _ _).mp h'
    rw [Nat.mod_eq_of_lt hqlt] at h''
    exact h''
  · intro h
    have h' : (((x.toNat * 2 ^ (32 - n)) % 2 ^ 32 : ℕ) : ZMod (2 ^ 32))
        = ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) :=
      congrArg (fun t : ℕ => (t : ZMod (2 ^ 32))) h
    rw [← h', ← Nat.cast_mul, ZMod.natCast_mod]
    exact sub_self _

/-- A concrete exactness instance from the criterion. -/
theorem exact32_ofNat {n w : ℕ} (hn : n < 32) (hw : w < 2 ^ 32)
    (h : (w * 2 ^ (32 - n)) % 2 ^ 32 = w / 2 ^ (32 - n)) :
    Exact32 n (BitVec.ofNat 32 w) := by
  rw [exact32_iff_offset_iff _ _ hn, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hw]
  exact h

/-- A concrete failure from the criterion. -/
theorem notExact32_ofNat {n w : ℕ} (hn : n < 32) (hw : w < 2 ^ 32)
    (h : (w * 2 ^ (32 - n)) % 2 ^ 32 ≠ w / 2 ^ (32 - n)) :
    ¬ Exact32 n (BitVec.ofNat 32 w) := by
  rw [exact32_iff_offset_iff _ _ hn, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hw]
  exact h

/-! ## §3 The closed form, for every amount `0 < n < 32` -/

/-- **The vanishing set in closed form at width 32, for every `0 < n < 32`.**

The two-slot reading is exact at `(n, x)` iff the base-`2^(64−2n)` digit split of the word
repeats: the *high* part `⌊x / 2^(64−2n)⌋` must equal the low `n` bits of the word, and the
remaining low part `x mod 2^(64−2n)` must be below `2^(32−n)`.

Derivation: with `e = 32 − n` the offset vanishes iff `2^e · (x mod 2^n) = ⌊x / 2^e⌋`, i.e. iff
`x = 2^(2e)·(x mod 2^n) + (x mod 2^e)` — the second slot is a *second shift*, and it equals the
wrapped high bits only when the two digit systems line up. -/
theorem exact32_iff_digit (x : BitVec 32) (n : ℕ) (hn0 : 0 < n) (hn32 : n < 32) :
    Exact32 n x
      ↔ x.toNat / 2 ^ (64 - 2 * n) = x.toNat % 2 ^ n
        ∧ x.toNat % 2 ^ (64 - 2 * n) < 2 ^ (32 - n) := by
  have he : 0 < 32 - n := by omega
  have he2 : 0 < 2 ^ (32 - n) := by positivity
  have hexp : 2 * (32 - n) = 64 - 2 * n := by omega
  have hxpow := two_pow_add_32 n hn32
  have hmod : (x.toNat * 2 ^ (32 - n)) % 2 ^ 32 = 2 ^ (32 - n) * (x.toNat % 2 ^ n) := by
    have hmm := Nat.mul_mod_mul_left (2 ^ (32 - n)) x.toNat (2 ^ n)
    rw [mul_comm (2 ^ (32 - n)) x.toNat, hxpow] at hmm
    exact hmm
  rw [exact32_iff_offset_iff x n hn32, hmod]
  constructor
  · intro h
    have key0 : x.toNat % 2 ^ (32 - n) + 2 ^ (32 - n) * (x.toNat / 2 ^ (32 - n)) = x.toNat :=
      Nat.mod_add_div _ _
    have key : x.toNat % 2 ^ (32 - n) + 2 ^ (64 - 2 * n) * (x.toNat % 2 ^ n) = x.toNat := by
      calc x.toNat % 2 ^ (32 - n) + 2 ^ (64 - 2 * n) * (x.toNat % 2 ^ n)
          = x.toNat % 2 ^ (32 - n)
              + 2 ^ (32 - n) * (2 ^ (32 - n) * (x.toNat % 2 ^ n)) := by
            rw [← Nat.mul_assoc, two_pow_sq (32 - n), hexp]
        _ = x.toNat % 2 ^ (32 - n) + 2 ^ (32 - n) * (x.toNat / 2 ^ (32 - n)) := by rw [h]
        _ = x.toNat := key0
    have hbig : x.toNat % 2 ^ (32 - n) < 2 ^ (64 - 2 * n) :=
      lt_of_lt_of_le (Nat.mod_lt _ he2) (two_pow_mono (a := 32 - n) (b := 64 - 2 * n) (by omega))
    obtain ⟨hdiv, hrem⟩ :=
      (Nat.div_mod_unique (by omega : 0 < 2 ^ (64 - 2 * n))).mpr ⟨key, hbig⟩
    exact ⟨hdiv, hrem ▸ Nat.mod_lt _ he2⟩
  · rintro ⟨h1, h2⟩
    have hd := Nat.mod_add_div x.toNat (2 ^ (64 - 2 * n))
    rw [h1] at hd
    have key : x.toNat % 2 ^ (64 - 2 * n)
        + 2 ^ (32 - n) * (2 ^ (32 - n) * (x.toNat % 2 ^ n)) = x.toNat := by
      calc x.toNat % 2 ^ (64 - 2 * n) + 2 ^ (32 - n) * (2 ^ (32 - n) * (x.toNat % 2 ^ n))
          = x.toNat % 2 ^ (64 - 2 * n) + 2 ^ (64 - 2 * n) * (x.toNat % 2 ^ n) := by
            rw [← Nat.mul_assoc, two_pow_sq (32 - n), hexp]
        _ = x.toNat := hd
    obtain ⟨h3, _⟩ := (Nat.div_mod_unique he2).mpr ⟨key, h2⟩
    exact h3.symm

/-! ## §4 The two regimes, with `Fin`-indexed parametrisations -/

/-- **Regime `0 < n ≤ 16`: the exactness set is exactly the words with the low `n` bits zero,
below `2^(32−n)`.** With `e = 32−n ≥ n` the second slot `2^e·(x mod 2^n)` is either `0` or at
least `2^e ≥ 2^n`, while the wrapped high bits are below `2^n`; so the two agree only when both
are zero, and then `⌊x/2^e⌋ = 0` forces `x < 2^e`. -/
theorem exact32_iff_lowbits (x : BitVec 32) (n : ℕ) (hn0 : 0 < n) (hn16 : n ≤ 16) :
    Exact32 n x ↔ x.toNat < 2 ^ (32 - n) ∧ x.toNat % 2 ^ n = 0 := by
  rw [exact32_iff_offset_iff x n (by omega : n < 32)]
  have hxpow := two_pow_add_32 n (by omega : n < 32)
  have hmod : (x.toNat * 2 ^ (32 - n)) % 2 ^ 32 = 2 ^ (32 - n) * (x.toNat % 2 ^ n) := by
    have hmm := Nat.mul_mod_mul_left (2 ^ (32 - n)) x.toNat (2 ^ n)
    rw [mul_comm (2 ^ (32 - n)) x.toNat, hxpow] at hmm
    exact hmm
  have hqlt : x.toNat / 2 ^ (32 - n) < 2 ^ n := by
    refine (Nat.div_lt_iff_lt_mul (by positivity)).mpr ?_
    rw [mul_comm (2 ^ n) (2 ^ (32 - n)), two_pow_add_32 n (by omega)]
    exact x.isLt
  have hele : 2 ^ n ≤ 2 ^ (32 - n) := two_pow_mono (a := n) (b := 32 - n) (by omega)
  rw [hmod]
  constructor
  · intro h
    by_cases hr0 : x.toNat % 2 ^ n = 0
    · have hz : x.toNat / 2 ^ (32 - n) = 0 := by rw [← h, hr0, Nat.mul_zero]
      have hd := Nat.mod_add_div x.toNat (2 ^ (32 - n))
      rw [hz, Nat.mul_zero, Nat.add_zero] at hd
      exact ⟨hd ▸ Nat.mod_lt _ (by positivity), hr0⟩
    · have hr1 : 1 ≤ x.toNat % 2 ^ n := Nat.succ_le_of_lt (Nat.pos_of_ne_zero hr0)
      have hmul : (2 ^ (32 - n)) * 1 ≤ 2 ^ (32 - n) * (x.toNat % 2 ^ n) :=
        Nat.mul_le_mul_left _ hr1
      have hbig : 2 ^ (32 - n) ≤ 2 ^ (32 - n) * (x.toNat % 2 ^ n) := by
        simpa only [Nat.mul_one] using hmul
      have hlt : x.toNat / 2 ^ (32 - n) < 2 ^ (32 - n) * (x.toNat % 2 ^ n) :=
        lt_of_lt_of_le hqlt (le_trans hele hbig)
      omega
  · rintro ⟨hx, hr0⟩
    have hz : x.toNat / 2 ^ (32 - n) = 0 := Nat.div_eq_of_lt hx
    rw [hz, hr0, Nat.mul_zero]

/-- **Regime `0 < n ≤ 16`, as a parametrisation by `Fin (2^(32−2n))`.** The exactness set has
exactly `2^(32−2n)` elements: the multiples of `2^n` below `2^(32−n)`. For `n = 1` that is every
even word below `2^31`, a quarter of the word space; for `n = 16` it is the single word `0`. -/
theorem exact32_iff_fin_lowbits (x : BitVec 32) (n : ℕ) (hn0 : 0 < n) (hn16 : n ≤ 16) :
    Exact32 n x ↔ ∃ k : Fin (2 ^ (32 - 2 * n)), x.toNat = (k : ℕ) * 2 ^ n := by
  rw [exact32_iff_lowbits x n hn0 hn16]
  constructor
  · rintro ⟨hx, hr0⟩
    have hexp : (32 - 2 * n) + n = 32 - n := by omega
    have hrlt : x.toNat / 2 ^ n < 2 ^ (32 - 2 * n) := by
      refine (Nat.div_lt_iff_lt_mul (by positivity)).mpr ?_
      rw [← Nat.pow_add, hexp]
      exact hx
    refine ⟨⟨x.toNat / 2 ^ n, hrlt⟩, ?_⟩
    calc x.toNat = x.toNat % 2 ^ n + 2 ^ n * (x.toNat / 2 ^ n) :=
          (Nat.mod_add_div x.toNat (2 ^ n)).symm
      _ = x.toNat / 2 ^ n * 2 ^ n := by rw [hr0, Nat.zero_add, Nat.mul_comm]
  · rintro ⟨k, hk⟩
    have hexp : (32 - 2 * n) + n = 32 - n := by omega
    refine ⟨?_, ?_⟩
    · calc x.toNat = (k : ℕ) * 2 ^ n := hk
        _ < 2 ^ (32 - 2 * n) * 2 ^ n := Nat.mul_lt_mul_of_pos_right k.isLt (by positivity)
        _ = 2 ^ (32 - n) := by rw [← Nat.pow_add, hexp]
    · rw [hk, Nat.mul_comm (k : ℕ) (2 ^ n), Nat.mul_mod_right]

/-- **The triviality boundary on the lower side.** At `n = 16` the exactness set is the single word
`0`. So for `0 < n ≤ 16` it is trivial exactly at `n = 16`. -/
theorem exact32_16_iff_zero (x : BitVec 32) : Exact32 16 x ↔ x.toNat = 0 := by
  rw [exact32_iff_fin_lowbits x 16 (by decide) (by decide)]
  constructor
  · rintro ⟨k, hk⟩
    have hklt : (k : ℕ) < 1 := by
      exact Nat.lt_of_lt_of_le k.isLt (by decide : 2 ^ (32 - 2 * 16) ≤ 1)
    have hk0 : (k : ℕ) = 0 := by omega
    rw [hk0] at hk
    simpa using hk
  · intro h
    rw [h]
    exact ⟨⟨0, by decide⟩, by decide⟩

/-- The zero word is in the exactness set at every amount below the width. -/
theorem exact32_zero (n : ℕ) (hn : n < 32) : Exact32 n (BitVec.ofNat 32 0) := by
  refine exact32_ofNat hn (by decide) ?_
  rw [Nat.zero_mul, Nat.zero_div, Nat.zero_mod]

/-- **Regime `17 ≤ n ≤ 21`: the exactness set is the digit family `b · 2^(64−2n) + b`, over
`Fin (2^(2n−32))`.** Here `64 − 2n ≥ n`, so reducing the word modulo `2^n` cannot see the high
digit, and the digit condition of §3 forces the low part to equal the high part. The set is *not*
the low-`n`-bits-zero set: at `n = 17` it is `{0, 2^30+1, 2^31+2, 2^31+2^30+3}`, four words of
which only `0` has its low 17 bits zero. -/
theorem exact32_iff_fin_digit (x : BitVec 32) (n : ℕ) (h17 : 17 ≤ n) (h21 : n ≤ 21) :
    Exact32 n x
      ↔ ∃ b : Fin (2 ^ (2 * n - 32)), x.toNat = (b : ℕ) * 2 ^ (64 - 2 * n) + (b : ℕ) := by
  have hn32 : n < 32 := by omega
  have hdvd : 2 ^ n ∣ 2 ^ (64 - 2 * n) := by
    refine ⟨2 ^ (64 - 3 * n), ?_⟩
    rw [← Nat.pow_add]
    congr 2
    omega
  constructor
  · intro h
    rw [exact32_iff_digit x n (by omega) hn32] at h
    obtain ⟨hb, hr⟩ := h
    have hrltn : x.toNat % 2 ^ (64 - 2 * n) < 2 ^ n :=
      lt_of_lt_of_le hr (two_pow_mono (by omega : 32 - n ≤ n))
    have hkey : x.toNat % 2 ^ (64 - 2 * n) = x.toNat % 2 ^ n := by
      rw [← Nat.mod_mod_of_dvd x.toNat hdvd]
      exact (Nat.mod_eq_of_lt hrltn).symm
    have hrd : x.toNat % 2 ^ (64 - 2 * n) = x.toNat / 2 ^ (64 - 2 * n) := hkey.trans hb.symm
    have hexp : (2 * n - 32) + (64 - 2 * n) = 32 := by omega
    have hbrlt : x.toNat / 2 ^ (64 - 2 * n) < 2 ^ (2 * n - 32) := by
      refine (Nat.div_lt_iff_lt_mul (by positivity)).mpr ?_
      rw [← Nat.pow_add, hexp]
      exact x.isLt
    refine ⟨⟨x.toNat / 2 ^ (64 - 2 * n), hbrlt⟩, ?_⟩
    calc x.toNat
        = x.toNat % 2 ^ (64 - 2 * n)
            + 2 ^ (64 - 2 * n) * (x.toNat / 2 ^ (64 - 2 * n)) := (Nat.mod_add_div _ _).symm
      _ = x.toNat / 2 ^ (64 - 2 * n) * 2 ^ (64 - 2 * n)
            + (x.toNat / 2 ^ (64 - 2 * n)) := by
        rw [hrd, Nat.mul_comm (2 ^ (64 - 2 * n)) (x.toNat / 2 ^ (64 - 2 * n)), Nat.add_comm]
  · rintro ⟨b, hk⟩
    have hbd : (b : ℕ) < 2 ^ (64 - 2 * n) :=
      lt_of_lt_of_le b.isLt (two_pow_mono (by omega : 2 * n - 32 ≤ 64 - 2 * n))
    have hbn : (b : ℕ) < 2 ^ n :=
      lt_of_lt_of_le b.isLt (two_pow_mono (by omega : 2 * n - 32 ≤ n))
    have hbe : (b : ℕ) < 2 ^ (32 - n) :=
      lt_of_lt_of_le b.isLt (two_pow_mono (by omega : 2 * n - 32 ≤ 32 - n))
    have hxd : x.toNat = (b : ℕ) + 2 ^ (64 - 2 * n) * (b : ℕ) := by
      rw [hk, Nat.mul_comm (b : ℕ) (2 ^ (64 - 2 * n)), Nat.add_comm]
    have hmodD : x.toNat % 2 ^ (64 - 2 * n) = (b : ℕ) := by
      rw [hxd, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hbd]
    have hdivb : 2 ^ n ∣ 2 ^ (64 - 2 * n) * (b : ℕ) := dvd_mul_of_dvd_left hdvd _
    have hz : (2 ^ (64 - 2 * n) * (b : ℕ)) % 2 ^ n = 0 := Nat.mod_eq_zero_of_dvd hdivb
    have hmodn : x.toNat % 2 ^ n = (b : ℕ) := by
      rw [hxd, Nat.add_mod, hz, Nat.add_zero]
      simp only [Nat.mod_eq_of_lt hbn]
    obtain ⟨hdivD, _⟩ :=
      (Nat.div_mod_unique (by positivity : 0 < 2 ^ (64 - 2 * n))).mpr ⟨hxd.symm, hbd⟩
    rw [exact32_iff_digit x n (by omega) hn32]
    exact ⟨hdivD.trans hmodn.symm, hmodD ▸ hbe⟩

/-! ## §5 SHA-2: the rotation amounts, with sizes and concrete members -/

/-- A rotate *right* by `0 < n < 32` is the rotate *left* by `32 − n`, so the exactness set of
FIPS 180-4's `rotr k` is the set at left amount `32 − k`. The hypothesis `0 < n` is needed: for
`n = 0` the reduction `(32 − n) mod 32 = 32 − n` fails. -/
theorem rotateRight_eq_rotateLeft32 (x : BitVec 32) (n : Nat) (hn0 : 0 < n) (hn32 : n < 32) :
    x.rotateRight n = x.rotateLeft (32 - n) := by
  have h3 : 32 - n < 32 := by omega
  rw [BitVec.rotateRight_def, BitVec.rotateLeft_def, Nat.mod_eq_of_lt hn32,
    Nat.mod_eq_of_lt h3, show 32 - (32 - n) = n from by omega, BitVec.or_comm]

/-! ### §5a The six named amounts `n ∈ {3, 7, 10, 17, 18, 19}` -/

theorem size3 : 2 ^ (32 - 2 * 3) = 2 ^ 26 := by decide
theorem size7 : 2 ^ (32 - 2 * 7) = 2 ^ 18 := by decide
theorem size10 : 2 ^ (32 - 2 * 10) = 2 ^ 12 := by decide
theorem size17 : 2 ^ (2 * 17 - 32) = 2 ^ 2 := by decide
theorem size18 : 2 ^ (2 * 18 - 32) = 2 ^ 4 := by decide
theorem size19 : 2 ^ (2 * 19 - 32) = 2 ^ 6 := by decide

/-- `n = 3`: the exactness set is `{k · 2^3 : k < 2^26}` (`size3`), and `8` is a member. -/
theorem exact32_3_set (x : BitVec 32) :
    Exact32 3 x ↔ ∃ k : Fin (2 ^ (32 - 2 * 3)), x.toNat = (k : ℕ) * 2 ^ 3 :=
  exact32_iff_fin_lowbits x 3 (by decide) (by decide)

theorem exact32_3_member : Exact32 3 (BitVec.ofNat 32 8) := by
  refine exact32_ofNat (by decide) (by decide) ?_
  decide

theorem notExact32_3_member : ¬ Exact32 3 (BitVec.ofNat 32 9) := by
  refine notExact32_ofNat (by decide) (by decide) ?_
  decide

/-- `n = 7`: the exactness set is `{k · 2^7 : k < 2^18}` (`size7`), and `128` is a member. -/
theorem exact32_7_set (x : BitVec 32) :
    Exact32 7 x ↔ ∃ k : Fin (2 ^ (32 - 2 * 7)), x.toNat = (k : ℕ) * 2 ^ 7 :=
  exact32_iff_fin_lowbits x 7 (by decide) (by decide)

theorem exact32_7_member : Exact32 7 (BitVec.ofNat 32 128) := by
  refine exact32_ofNat (by decide) (by decide) ?_
  decide

theorem notExact32_7_member : ¬ Exact32 7 (BitVec.ofNat 32 129) := by
  refine notExact32_ofNat (by decide) (by decide) ?_
  decide

/-- `n = 10`: the exactness set is `{k · 2^10 : k < 2^12}` (`size10`), and `1024` is a member. -/
theorem exact32_10_set (x : BitVec 32) :
    Exact32 10 x ↔ ∃ k : Fin (2 ^ (32 - 2 * 10)), x.toNat = (k : ℕ) * 2 ^ 10 :=
  exact32_iff_fin_lowbits x 10 (by decide) (by decide)

theorem exact32_10_member : Exact32 10 (BitVec.ofNat 32 1024) := by
  refine exact32_ofNat (by decide) (by decide) ?_
  decide

/-- `n = 17`: the exactness set is the four words `{b · 2^30 + b : b < 4}` (`size17`), and
`2^30 + 1` is a member that does *not* have its low 17 bits zero. -/
theorem exact32_17_set (x : BitVec 32) :
    Exact32 17 x
      ↔ ∃ b : Fin (2 ^ (2 * 17 - 32)), x.toNat = (b : ℕ) * 2 ^ (64 - 2 * 17) + (b : ℕ) :=
  exact32_iff_fin_digit x 17 (by decide) (by decide)

theorem exact32_17_member : Exact32 17 (BitVec.ofNat 32 (2 ^ 30 + 1)) := by
  refine exact32_ofNat (by decide) (by decide) ?_
  decide

theorem exact32_17_member_not_lowbits : (2 ^ 30 + 1 : ℕ) % 2 ^ 17 ≠ 0 := by decide

/-- `n = 18`: the exactness set is `{b · 2^28 + b : b < 2^4}` (`size18`), and `2^28 + 1` is a
member. -/
theorem exact32_18_set (x : BitVec 32) :
    Exact32 18 x
      ↔ ∃ b : Fin (2 ^ (2 * 18 - 32)), x.toNat = (b : ℕ) * 2 ^ (64 - 2 * 18) + (b : ℕ) :=
  exact32_iff_fin_digit x 18 (by decide) (by decide)

theorem exact32_18_member : Exact32 18 (BitVec.ofNat 32 (2 ^ 28 + 1)) := by
  refine exact32_ofNat (by decide) (by decide) ?_
  decide

theorem notExact32_18_member : ¬ Exact32 18 (BitVec.ofNat 32 (2 ^ 28)) := by
  refine notExact32_ofNat (by decide) (by decide) ?_
  decide

/-- `n = 19`: the exactness set is `{b · 2^26 + b : b < 2^6}` (`size19`), and `2^26 + 1` is a
member. -/
theorem exact32_19_set (x : BitVec 32) :
    Exact32 19 x
      ↔ ∃ b : Fin (2 ^ (2 * 19 - 32)), x.toNat = (b : ℕ) * 2 ^ (64 - 2 * 19) + (b : ℕ) :=
  exact32_iff_fin_digit x 19 (by decide) (by decide)

theorem exact32_19_member : Exact32 19 (BitVec.ofNat 32 (2 ^ 26 + 1)) := by
  refine exact32_ofNat (by decide) (by decide) ?_
  decide

theorem notExact32_19_member : ¬ Exact32 19 (BitVec.ofNat 32 (2 ^ 26)) := by
  refine notExact32_ofNat (by decide) (by decide) ?_
  decide

/-! ### §5b The ten rotation amounts of FIPS 180-4 -/

/-- **The ten `ROTR` amounts of FIPS 180-4, in the left-rotation spelling this file uses.** Each
conjunct is the existence of a concrete non-trivial member of that amount's exactness set; the
structure and size are §4's `exact32_iff_fin_lowbits` (left amount `≤ 16`),
`exact32_iff_fin_digit` (left amount in `[17, 21]`) and the closed digit criterion
`exact32_iff_digit` for the three rows above `21`. `rotr 2` (left amount `30`) is the only row
whose set is trivial; see `exact32_30_iff_zero`. -/
theorem sha2_rotation_amounts :
    -- left 7 = rotr 25, a term of Sigma1
    (Exact32 7 (BitVec.ofNat 32 128))
    ∧ -- left 10 = rotr 22, a term of Sigma0
      (Exact32 10 (BitVec.ofNat 32 1024))
    ∧ -- left 13 = rotr 19, a term of sigma1
      (Exact32 13 (BitVec.ofNat 32 8192))
    ∧ -- left 14 = rotr 18, a term of sigma0
      (Exact32 14 (BitVec.ofNat 32 16384))
    ∧ -- left 15 = rotr 17, a term of sigma1
      (Exact32 15 (BitVec.ofNat 32 32768))
    ∧ -- left 19 = rotr 13, a term of Sigma0
      (Exact32 19 (BitVec.ofNat 32 (2 ^ 26 + 1)))
    ∧ -- left 21 = rotr 11, a term of Sigma1
      (Exact32 21 (BitVec.ofNat 32 (2 ^ 22 + 1)))
    ∧ -- left 25 = rotr 7, a term of sigma0
      (Exact32 25 (BitVec.ofNat 32 (2 ^ 28 + 2 ^ 14 + 1)))
    ∧ -- left 26 = rotr 6, a term of Sigma1
      (Exact32 26 (BitVec.ofNat 32 (2 ^ 26 + 2 ^ 14 + 4)))
    ∧ -- left 30 = rotr 2, a term of Sigma0: the exactness set is {0}
      (Exact32 30 (BitVec.ofNat 32 0)) := by
  refine ⟨exact32_ofNat (by decide) (by decide) (by decide),
    exact32_ofNat (by decide) (by decide) (by decide),
    exact32_ofNat (by decide) (by decide) (by decide),
    exact32_ofNat (by decide) (by decide) (by decide),
    exact32_ofNat (by decide) (by decide) (by decide),
    exact32_ofNat (by decide) (by decide) (by decide),
    exact32_ofNat (by decide) (by decide) (by decide),
    exact32_ofNat (by decide) (by decide) (by decide),
    exact32_ofNat (by decide) (by decide) (by decide),
    exact32_zero 30 (by decide)⟩

/-- **The `rotr 2` term of `Sigma0` is the only SHA-2 rotation amount whose exactness set is
trivial.** It is the left amount `30`, and the digit criterion forces `x = 0`.

The obstruction is arithmetic, not a digit argument: with `q = ⌊x / 2^4⌋`, `s = x mod 2^4 ≤ 3` and
`u = ⌊x / 2^30⌋ ≤ 3` the two decompositions of `x` give `15q + s = 2^30·u`, so reducing modulo 15
gives `s ≡ 4u (mod 15)`; with `s ≤ 3` that forces `u = 0, s = 0, q = 0, x = 0`. -/
theorem exact32_30_iff_zero (x : BitVec 32) : Exact32 30 x ↔ x.toNat = 0 := by
  rw [exact32_iff_digit x 30 (by decide) (by decide)]
  constructor
  · rintro ⟨h1, h2⟩
    have hs3 : x.toNat % 2 ^ 4 ≤ 3 := by omega
    have hs15 : x.toNat % 2 ^ 4 < 15 := by omega
    have hu3 : x.toNat / 2 ^ 30 ≤ 3 := by
      refine (Nat.div_le_iff_le_mul (by decide : 0 < 2 ^ 30)).mpr ?_
      have hx : x.toNat ≤ 2 ^ 32 - 1 := by omega
      omega
    have hA : x.toNat % 2 ^ 4 + 2 ^ 4 * (x.toNat / 2 ^ 4) = x.toNat := Nat.mod_add_div _ _
    have hB : x.toNat % 2 ^ 30 + 2 ^ 30 * (x.toNat / 2 ^ 30) = x.toNat := Nat.mod_add_div _ _
    have hkey : 15 * (x.toNat / 2 ^ 4) + (x.toNat % 2 ^ 4)
        = 2 ^ 30 * (x.toNat / 2 ^ 30) := by
      rw [← h1] at hB
      omega
    have hmodr : (2 ^ 30 * (x.toNat / 2 ^ 30) : ℕ) % 15 = x.toNat % 2 ^ 4 := by
      have hz := congrArg (fun t : ℕ => t % 15) hkey
      rw [Nat.add_mod, Nat.mul_mod_right, Nat.zero_add] at hz
      have hz' : (x.toNat % 2 ^ 4) % 15 % 15 = x.toNat % 2 ^ 4 := by
        simp only [Nat.mod_eq_of_lt hs15]
      exact hz.symm.trans hz'
    have hu0 : x.toNat / 2 ^ 30 = 0 ∨ x.toNat / 2 ^ 30 = 1 ∨ x.toNat / 2 ^ 30 = 2
        ∨ x.toNat / 2 ^ 30 = 3 := by omega
    rcases hu0 with hu | hu | hu | hu
    · rw [hu] at hmodr hkey
      have e0 : (2 ^ 30 * 0 : ℕ) % 15 = 0 := by decide
      rw [e0] at hmodr
      omega
    · rw [hu] at hmodr
      have e1 : (2 ^ 30 * 1 : ℕ) % 15 = 4 := by decide
      rw [e1] at hmodr
      omega
    · rw [hu] at hmodr
      have e2 : (2 ^ 30 * 2 : ℕ) % 15 = 8 := by decide
      rw [e2] at hmodr
      omega
    · rw [hu] at hmodr
      have e3 : (2 ^ 30 * 3 : ℕ) % 15 = 12 := by decide
      rw [e3] at hmodr
      omega
  · intro h
    rw [h]
    exact ⟨by decide, by decide⟩

theorem notExact32_30_one : ¬ Exact32 30 (BitVec.ofNat 32 1) := by
  refine notExact32_ofNat (by decide) (by decide) ?_
  decide

/-! ## §6 Invariance: can a state enter, leave, or be confined to the exactness set? -/

/-! The question, stated as a hypothesis and **not** as a claim. Fix an amount `n` with
`0 < n < 32` and a word map `S` (a SHA-2 round step). Three properties are distinguishable:

* **confinement in** — `∀ x, Exact32 n x → Exact32 n (S x)`; the exactness set is forward
  invariant;
* **confinement out** — `∀ x, Fail32 n x → Fail32 n (S x)`; the failure set is forward invariant,
  equivalently the exactness set is backward invariant;
* **no confinement either way** — both fail, so one round step moves states across the boundary
  in both orientations.

What would have to be proved for any invariance claim is a *structural* statement about `S`: that
`S` respects the digit condition of `exact32_iff_digit`, i.e. that it sends
`{x : ⌊x / 2^(64−2n)⌋ = x mod 2^n ∧ x mod 2^(64−2n) < 2^(32−n)}` into itself. SHA-2's `S` is a
composition of XOR, modular additions and the four σ functions, none of which is a digit shift in
`ZMod (2^32)`, so no such statement is proved here and the general case is **open**.

What is proved below are small decidable instances: entry and exit at one round step of `Sigma0`
at the amount `rotr 22`, and exit at two and at four round steps, each by an explicit word. They
refute both confinements at that amount. They establish nothing about other amounts, other maps,
or a general `S`. -/

/-- `Sigma0` at width 32: `ROTR^2 ⊕ ROTR^13 ⊕ ROTR^22` (FIPS 180-4 §4.1.3, `Sha2.lean:67`). -/
def Sigma0_32 (x : BitVec 32) : BitVec 32 :=
  x.rotateRight 2 ^^^ x.rotateRight 13 ^^^ x.rotateRight 22

/-- `Sigma1` at width 32: `ROTR^6 ⊕ ROTR^11 ⊕ ROTR^25` (`Sha2.lean:70`). -/
def Sigma1_32 (x : BitVec 32) : BitVec 32 :=
  x.rotateRight 6 ^^^ x.rotateRight 11 ^^^ x.rotateRight 25

theorem Sigma0_1024_eq : Sigma0_32 (BitVec.ofNat 32 1024) = BitVec.ofNat 32 537919744 := by
  decide

theorem Sigma0_537919744_eq :
    Sigma0_32 (BitVec.ofNat 32 537919744) = BitVec.ofNat 32 1073807424 := by decide

theorem Sigma0_1073807424_eq :
    Sigma0_32 (BitVec.ofNat 32 1073807424) = BitVec.ofNat 32 369312024 := by decide

theorem Sigma0_369312024_eq :
    Sigma0_32 (BitVec.ofNat 32 369312024) = BitVec.ofNat 32 4456452 := by decide

theorem Sigma0_1397419_eq :
    Sigma0_32 (BitVec.ofNat 32 1397419) = BitVec.ofNat 32 1570816 := by decide

theorem Sigma0_1024_toNat : (Sigma0_32 (BitVec.ofNat 32 1024)).toNat = 537919744 := by decide

theorem Sigma0_1024_two_toNat :
    (Sigma0_32 (Sigma0_32 (BitVec.ofNat 32 1024))).toNat = 1073807424 := by
  rw [Sigma0_1024_eq, Sigma0_537919744_eq]
  decide

theorem Sigma0_1024_four_toNat :
    (Sigma0_32 (Sigma0_32 (Sigma0_32 (Sigma0_32 (BitVec.ofNat 32 1024))))).toNat = 4456452 := by
  rw [Sigma0_1024_eq, Sigma0_537919744_eq, Sigma0_1073807424_eq, Sigma0_369312024_eq]
  decide

/-- **Exit, at the amount `rotr 22` (left amount 10), one round step of `Sigma0`.** The word `1024`
is read exactly and its `Sigma0` image `537919744` is not. So the exactness set is not forward
invariant. -/
theorem Sigma0_leaves_exact10 :
    Exact32 10 (BitVec.ofNat 32 1024) ∧ Fail32 10 (Sigma0_32 (BitVec.ofNat 32 1024)) :=
  ⟨exact32_10_member, by
    rw [Fail32, exact32_iff_offset_iff _ _ (by decide), Sigma0_1024_toNat]
    decide⟩

/-- **Entry, at the amount `rotr 22`, one round step of `Sigma0`.** The word `1397419` is *not*
read exactly and its `Sigma0` image `1570816` *is*. So the exactness set is not backward invariant. -/
theorem Sigma0_enters_exact10 :
    Fail32 10 (BitVec.ofNat 32 1397419) ∧ Exact32 10 (Sigma0_32 (BitVec.ofNat 32 1397419)) := by
  refine ⟨by
    refine notExact32_ofNat (by decide) (by decide) ?_
    decide, by
    rw [exact32_iff_offset_iff _ _ (by decide), Sigma0_1397419_eq]
    decide⟩

/-- **Exit at two round steps** of `Sigma0`, same amount. -/
theorem Sigma0_leaves_exact10_two :
    Exact32 10 (BitVec.ofNat 32 1024)
      ∧ Fail32 10 (Sigma0_32 (Sigma0_32 (BitVec.ofNat 32 1024))) :=
  ⟨exact32_10_member, by
    rw [Fail32, exact32_iff_offset_iff _ _ (by decide), Sigma0_1024_two_toNat]
    decide⟩

/-- **Exit at four round steps** of `Sigma0`, same amount. -/
theorem Sigma0_leaves_exact10_four :
    Exact32 10 (BitVec.ofNat 32 1024)
      ∧ Fail32 10
          (Sigma0_32 (Sigma0_32 (Sigma0_32 (Sigma0_32 (BitVec.ofNat 32 1024))))) :=
  ⟨exact32_10_member, by
    rw [Fail32, exact32_iff_offset_iff _ _ (by decide), Sigma0_1024_four_toNat]
    decide⟩

/-- **Neither confinement holds at the amount `rotr 22` under one round step of `Sigma0`.** This is
the refutation of the invariance hypothesis in that one instance. -/
theorem exact10_neither_confinement :
    (¬ (∀ x : BitVec 32, Exact32 10 x → Exact32 10 (Sigma0_32 x)))
      ∧ (¬ (∀ x : BitVec 32, Fail32 10 x → Fail32 10 (Sigma0_32 x))) := by
  constructor
  · intro h
    exact Sigma0_leaves_exact10.2 (h _ exact32_10_member)
  · intro h
    exact h _ Sigma0_enters_exact10.1 Sigma0_enters_exact10.2

theorem Sigma1_128_eq : Sigma1_32 (BitVec.ofNat 32 128) = BitVec.ofNat 32 268451842 := by decide

theorem Sigma1_268451842_eq :
    Sigma1_32 (BitVec.ofNat 32 268451842) = BitVec.ofNat 32 136445952 := by decide

theorem Sigma1_136445952_eq :
    Sigma1_32 (BitVec.ofNat 32 136445952) = BitVec.ofNat 32 287411268 := by decide

theorem Sigma1_287411268_eq :
    Sigma1_32 (BitVec.ofNat 32 287411268) = BitVec.ofNat 32 134250504 := by decide

theorem Sigma1_8456_eq : Sigma1_32 (BitVec.ofNat 32 8456) = BitVec.ofNat 32 17859712 := by
  decide

/-- **Exit, at the amount `rotr 25` (left amount 7), one round step of `Sigma1`.** The word `128`
is read exactly and its `Sigma1` image `268451842` is not. -/
theorem Sigma1_leaves_exact7 :
    Exact32 7 (BitVec.ofNat 32 128) ∧ Fail32 7 (Sigma1_32 (BitVec.ofNat 32 128)) := by
  refine ⟨exact32_7_member, ?_⟩
  rw [Fail32, exact32_iff_offset_iff _ _ (by decide), Sigma1_128_eq]
  decide

/-- **Entry, at the amount `rotr 25`, one round step of `Sigma1`.** The word `8456` is *not* read
exactly and its `Sigma1` image `17859712` *is*. So both confinements fail at a second, independent
amount and a second, independent round map. -/
theorem Sigma1_enters_exact7 :
    Fail32 7 (BitVec.ofNat 32 8456) ∧ Exact32 7 (Sigma1_32 (BitVec.ofNat 32 8456)) := by
  refine ⟨by
    refine notExact32_ofNat (by decide) (by decide) ?_
    decide, by
    rw [exact32_iff_offset_iff _ _ (by decide), Sigma1_8456_eq]
    decide⟩

/-- **Exit at two and at four round steps** of `Sigma1`, amount `rotr 25`. -/
theorem Sigma1_leaves_exact7_two :
    Exact32 7 (BitVec.ofNat 32 128)
      ∧ Fail32 7 (Sigma1_32 (Sigma1_32 (BitVec.ofNat 32 128))) := by
  refine ⟨exact32_7_member, ?_⟩
  rw [Fail32, exact32_iff_offset_iff _ _ (by decide), Sigma1_128_eq, Sigma1_268451842_eq]
  decide

theorem Sigma1_leaves_exact7_four :
    Exact32 7 (BitVec.ofNat 32 128)
      ∧ Fail32 7
          (Sigma1_32 (Sigma1_32 (Sigma1_32 (Sigma1_32 (BitVec.ofNat 32 128))))) := by
  refine ⟨exact32_7_member, ?_⟩
  rw [Fail32, exact32_iff_offset_iff _ _ (by decide), Sigma1_128_eq, Sigma1_268451842_eq,
    Sigma1_136445952_eq, Sigma1_287411268_eq]
  decide

/-- **Neither confinement holds at the amount `rotr 25` under one round step of `Sigma1`.** -/
theorem exact7_neither_confinement :
    (¬ (∀ x : BitVec 32, Exact32 7 x → Exact32 7 (Sigma1_32 x)))
      ∧ (¬ (∀ x : BitVec 32, Fail32 7 x → Fail32 7 (Sigma1_32 x))) := by
  constructor
  · intro h
    exact Sigma1_leaves_exact7.2 (h _ exact32_7_member)
  · intro h
    exact h _ Sigma1_enters_exact7.1 Sigma1_enters_exact7.2

/-! ## §7 The joint statement with the attack lab's algebraic-failure set -/

/-- **The algebraic-failure predicate at width 32**: multiplication by the key cell `κ` collapses
two distinct states of the two-slot algebra. `AlgebraAttackLab.lean` characterises this as
`¬ IsUnit (ringNormForm σ κ)` (`mul_collides_iff_norm_notUnit`, read-only here, sha256
`9b73fb922beb45d0efb2a94e694835e3e191aac5e859c5f85059668be7a63ed6`); it is not imported, so the
concrete rows below are re-proved from the corpus primitives `ringMul` and `ringNormForm`. -/
def Collapses32 (κ : RingCell (ZMod (2 ^ 32))) : Prop :=
  ∃ s₁ s₂ : RingCell (ZMod (2 ^ 32)), s₁ ≠ s₂ ∧
    ringMul (1 : ZMod (2 ^ 32)) κ s₁ = ringMul (1 : ZMod (2 ^ 32)) κ s₂

/-- Multiplication by the key `(1, 0)` is the identity on the two-slot algebra. -/
theorem ringMul_one32_id (s : RingCell (ZMod (2 ^ 32))) :
    ringMul (1 : ZMod (2 ^ 32)) (1, 0) s = s := by
  obtain ⟨a, b⟩ := s
  apply Prod.ext <;> ring

/-- The key `(1, 0)` does not collapse: the algebraic-failure condition fails there. -/
theorem notCollapses32_one : ¬ Collapses32 ((1, 0) : RingCell (ZMod (2 ^ 32))) := by
  rintro ⟨s₁, s₂, hne, heq⟩
  have h1 := ringMul_one32_id s₁
  have h2 := ringMul_one32_id s₂
  rw [h1, h2] at heq
  exact hne heq

/-- The key `(2, 2)` has norm form `0`: `N(2,2) = 2² − 1·2² = 0`. -/
theorem norm_key22_32 :
    ringNormForm (1 : ZMod (2 ^ 32)) ((2, 2) : RingCell (ZMod (2 ^ 32))) = 0 := by
  ring

/-- The key `(2, 2)` collapses the two distinct states `(1,0)` and `(0,1)` onto the same cell. -/
theorem collapses32_key22 : Collapses32 ((2, 2) : RingCell (ZMod (2 ^ 32))) := by
  refine ⟨(1, 0), (0, 1), ?_, ?_⟩
  · decide
  · decide

/-- **Reading exactly does not certify injectivity.** At the amount `rotr 25` (left amount 7) the
word `128` is read exactly, yet at the key `(2, 2)` the round map still collapses two distinct
states. -/
theorem joint_exact_not_injective :
    Exact32 7 (BitVec.ofNat 32 128) ∧ Collapses32 ((2, 2) : RingCell (ZMod (2 ^ 32))) ∧
      ringNormForm (1 : ZMod (2 ^ 32)) ((2, 2) : RingCell (ZMod (2 ^ 32))) = 0 :=
  ⟨exact32_7_member, collapses32_key22, norm_key22_32⟩

/-- **Non-collapse does not certify exact reading.** At the key `(1, 0)` the round map is
injective, yet at the amount `rotr 25` the word `1` is still misread. So the two conditions are
independent, and the trustworthy region for the whole two-slot reading is the *product* of the two
complement sets, not either one alone. -/
theorem joint_not_injective_not_exact :
    (¬ Collapses32 ((1, 0) : RingCell (ZMod (2 ^ 32)))) ∧
      Fail32 7 (BitVec.ofNat 32 1) := by
  refine ⟨notCollapses32_one, ?_⟩
  refine notExact32_ofNat (by decide) (by decide) ?_
  decide

/-- **The joint trustworthy-region statement.** A state is read exactly, and the round built on the
key cell is free of the algebraic failure, exactly on
`{x : ¬ Fail32 n x} × {κ : ¬ Collapses32 κ}`; the two factors are independent
(`joint_exact_not_injective`, `joint_not_injective_not_exact`), so neither one implies the other
and the two-slot reading cannot be certified from injectivity or from exactness alone. -/
theorem joint_trustworthy_region :
    ∀ (n : ℕ) (x : BitVec 32) (κ : RingCell (ZMod (2 ^ 32))),
      (Exact32 n x ∧ ¬ Collapses32 κ) ↔ (¬ Fail32 n x ∧ ¬ Collapses32 κ) := by
  intro n x κ
  constructor
  · rintro ⟨h, h2⟩
    exact ⟨exact32_iff_not_fail32 n x |>.mp h, h2⟩
  · rintro ⟨h, h2⟩
    exact ⟨exact32_iff_not_fail32 n x |>.mpr h, h2⟩

/-! ## §8 Axiom cone -/

#print axioms AlgebraRotExact.two_pow_mono
#print axioms AlgebraRotExact.two_pow_sq
#print axioms AlgebraRotExact.two_pow_add_32
#print axioms AlgebraRotExact.two_pow_add_32'
#print axioms AlgebraRotExact.ringUpper_mul_root32
#print axioms AlgebraRotExact.ringUpper_mul_eval
#print axioms AlgebraRotExact.bvDenote32_def
#print axioms AlgebraRotExact.rotExpr32_eq
#print axioms AlgebraRotExact.toNat_rotateLeft_eq_sum32
#print axioms AlgebraRotExact.bvDenote32_rotateLeft_eq
#print axioms AlgebraRotExact.rotExpr32_offset_eq
#print axioms AlgebraRotExact.two_pow_ne_zero32
#print axioms AlgebraRotExact.exact32_iff_not_fail32
#print axioms AlgebraRotExact.exact32_iff_offset_iff
#print axioms AlgebraRotExact.exact32_ofNat
#print axioms AlgebraRotExact.notExact32_ofNat
#print axioms AlgebraRotExact.exact32_iff_digit
#print axioms AlgebraRotExact.exact32_iff_lowbits
#print axioms AlgebraRotExact.exact32_iff_fin_lowbits
#print axioms AlgebraRotExact.exact32_16_iff_zero
#print axioms AlgebraRotExact.exact32_zero
#print axioms AlgebraRotExact.exact32_iff_fin_digit
#print axioms AlgebraRotExact.rotateRight_eq_rotateLeft32
#print axioms AlgebraRotExact.size3
#print axioms AlgebraRotExact.size7
#print axioms AlgebraRotExact.size10
#print axioms AlgebraRotExact.size17
#print axioms AlgebraRotExact.size18
#print axioms AlgebraRotExact.size19
#print axioms AlgebraRotExact.exact32_3_set
#print axioms AlgebraRotExact.exact32_3_member
#print axioms AlgebraRotExact.notExact32_3_member
#print axioms AlgebraRotExact.exact32_7_set
#print axioms AlgebraRotExact.exact32_7_member
#print axioms AlgebraRotExact.notExact32_7_member
#print axioms AlgebraRotExact.exact32_10_set
#print axioms AlgebraRotExact.exact32_10_member
#print axioms AlgebraRotExact.exact32_17_set
#print axioms AlgebraRotExact.exact32_17_member
#print axioms AlgebraRotExact.exact32_17_member_not_lowbits
#print axioms AlgebraRotExact.exact32_18_set
#print axioms AlgebraRotExact.exact32_18_member
#print axioms AlgebraRotExact.notExact32_18_member
#print axioms AlgebraRotExact.exact32_19_set
#print axioms AlgebraRotExact.exact32_19_member
#print axioms AlgebraRotExact.notExact32_19_member
#print axioms AlgebraRotExact.sha2_rotation_amounts
#print axioms AlgebraRotExact.exact32_30_iff_zero
#print axioms AlgebraRotExact.notExact32_30_one
#print axioms AlgebraRotExact.Sigma0_leaves_exact10
#print axioms AlgebraRotExact.Sigma0_enters_exact10
#print axioms AlgebraRotExact.Sigma0_leaves_exact10_two
#print axioms AlgebraRotExact.Sigma0_leaves_exact10_four
#print axioms AlgebraRotExact.exact10_neither_confinement
#print axioms AlgebraRotExact.Sigma1_leaves_exact7
#print axioms AlgebraRotExact.Sigma1_enters_exact7
#print axioms AlgebraRotExact.Sigma1_leaves_exact7_two
#print axioms AlgebraRotExact.Sigma1_leaves_exact7_four
#print axioms AlgebraRotExact.exact7_neither_confinement
#print axioms AlgebraRotExact.notCollapses32_one
#print axioms AlgebraRotExact.norm_key22_32
#print axioms AlgebraRotExact.collapses32_key22
#print axioms AlgebraRotExact.joint_exact_not_injective
#print axioms AlgebraRotExact.joint_not_injective_not_exact
#print axioms AlgebraRotExact.joint_trustworthy_region

end AlgebraRotExact