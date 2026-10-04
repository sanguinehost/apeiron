/-
  `AlgebraRotBridge32.lean` — the width-32 rotation representation bridge: the bounded word
  domain, the read refinement, and the action verdict for `ringMul (-1)` on the bit-halves
  encoding.

  Task AR-36 of the algebra-rooting wave, from `receipts/sol-face-adjudication.md` §5 row 1.
  Owned files: this one and `AlgebraRotBridge32.md`. No production, pin, extraction-root or API
  edit; no commit. Formal base `549d00ca62df50802db47f3e58ec4ee11c40c4b6`.

  AR-17 (`work/proofs/AlgebraRotorSha.lean`, read-only reference) proves the read refinement
  `(ringMul (-1) (halves n a) (halvesMultiplier n)).1 = denNat (rotrVal n a)` for every width
  and amount. It does not prove the action property, and its raw product is not the re-encoded
  rotation. This file settles the action property on the bounded word domain `a < 2^k`,
  `0 < n < k`, parametrically and at `k = 32`:

  §1 Domain, lossless recovery, the read refinement through the actual `ringMul (-1)`, and the
     `BitVec.rotateRight` link, with a joint reachable witness at `k = 32`.
  §2 The action property fails. `product_snd` is the exact coordinate form of the raw product;
     `action_gap_snd` is the exact second-coordinate difference as a function of `a` and `n`;
     `mismatch_at_one` refutes the equality at `a = 1` for every `k ≥ 2`, `0 < n < k`;
     `no_equivariant_multiplier` proves that no multiplier at all makes the halves pairing
     equivariant, so no corrected `u` exists for this pairing.
  §3 Composition. `cellIter_eq_ringMul_pow` is what raw iteration computes;
     `composition_breaks` is the one-step refutation of rotation iteration;
     `reencode_composes` is the composition that does hold, with the encoded state re-derived
     from the read value after each step.
  §4 A multiplicative injective re-encoding cannot repair it: pullback would equip the halves
     pairing with an equivariant multiplier, which §2 excludes.

  Trust boundary. Statements are about `ZMod (2^k)`, `ZMod (2^k) × ZMod (2^k)` and `BitVec k`.
  They do not reach `rustc`/LLVM, the kernel ABI, the fixed-point extraction, or any security,
  collision-resistance or constant-time claim. No `sorry`, `admit`, `axiom`, `native_decide`,
  raised `maxRecDepth` or `maxHeartbeats 0`; the only numerical steps are ground `decide` calls
  at width 2 and `omega`/`ring` steps over `ℕ` and `ZMod (2^k)`. No norm-one premise is imposed
  on the read: `ringMul_halves_read` holds for every `n`, including the non-unitary amounts.

  Reference: `work/proofs/AlgebraRotorSha.lean` (read in full), whose declarations `denNat`,
  `halves`, `halvesMultiplier`, `rotrVal`, `rotrBv`, `rotrVal_eq_rotrBv`, `denNat_div_add_mod`,
  `halves_recover` are re-derived here because `work/proofs` is not a lake package and nothing
  is imported from it. `ringMul_halves_read` is that file's positive result, restated through
  the raw `ringMul (-1)`.
-/

import Apeiron.Trichotomy.AnyRing

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace AlgebraRotBridge32

open Apeiron.Trichotomy.AnyRing

/-! ## §0 The two-slot operations in projection form -/

/-- First slot of `ringMul`, componentwise. -/
theorem ringMul_fst_eq {R : Type*} [CommRing R] (σ : R) (z w : R × R) :
    (ringMul σ z w).1 = z.1 * w.1 + σ * z.2 * w.2 := rfl

/-- Second slot of `ringMul`, componentwise. -/
theorem ringMul_snd_eq {R : Type*} [CommRing R] (σ : R) (z w : R × R) :
    (ringMul σ z w).2 = z.1 * w.2 + z.2 * w.1 := rfl

/-! ## §1 Bounded domain, recovery and the read refinement -/

/-- The `k`-bit word of value `a`, read into `ZMod (2^k)`. -/
def denNat {k : Nat} (a : Nat) : ZMod (2 ^ k) := (a : Nat)

/-- **The bounded word domain**: `a` fits in `k` bits. An abbreviation so `decide` sees the
    underlying comparison. -/
abbrev InWord (k a : Nat) : Prop := a < 2 ^ k

/-- **The bounded amount domain**: `n` is a positive right-rotation amount below the width.
    An abbreviation so `decide` sees the underlying conjunction. -/
abbrev InAmount (k n : Nat) : Prop := 0 < n ∧ n < k

/-- **The halves encoding of `a` at amount `n`**: low `n` bits in slot 1, high `k - n` in slot 2. -/
def halves {k : Nat} (n a : Nat) : ZMod (2 ^ k) × ZMod (2 ^ k) :=
  (denNat (a % 2 ^ n), denNat (a / 2 ^ n))

theorem halves_fst {k : Nat} (n a : Nat) :
    (halves (k := k) n a).1 = denNat (a % 2 ^ n) := rfl

theorem halves_snd {k : Nat} (n a : Nat) :
    (halves (k := k) n a).2 = denNat (a / 2 ^ n) := rfl

/-- **The multiplier the read uses**: `(2^(k-n), -1)`. -/
def halvesMultiplier {k : Nat} (n : Nat) : ZMod (2 ^ k) × ZMod (2 ^ k) :=
  (denNat (2 ^ (k - n)), -1)

theorem halvesMultiplier_fst {k : Nat} (n : Nat) :
    (halvesMultiplier (k := k) n).1 = denNat (2 ^ (k - n)) := rfl

theorem halvesMultiplier_snd {k : Nat} (n : Nat) :
    (halvesMultiplier (k := k) n).2 = -1 := rfl

/-- Right rotation of the value `a` by `n` at width `k`, on `ℕ`. -/
def rotrVal {k : Nat} (n a : Nat) : Nat := a / 2 ^ n + (a % 2 ^ n) * 2 ^ (k - n)

/-- The rotation as `BitVec.rotateRight` of the word `a`, the extracted `rotr` body. -/
def rotrBv {k : Nat} (a n : Nat) : BitVec k := (BitVec.ofNat k a).rotateRight n

/-- Division theorem `a = (a / 2^n) · 2^n + a % 2^n`, read into `ZMod (2^k)`. -/
theorem denNat_div_add_mod (k n a : Nat) :
    denNat (k := k) (a / 2 ^ n) * denNat (k := k) (2 ^ n) + denNat (k := k) (a % 2 ^ n)
      = denNat (k := k) a := by
  have h := Nat.div_add_mod' a (2 ^ n)
  simpa only [denNat, Nat.cast_mul, Nat.cast_add] using
    congrArg (fun t : Nat => (t : ZMod (2 ^ k))) h

/-- **Lossless recovery.** The two slots recombine to the word. -/
theorem halves_recover {k n a : Nat} :
    (halves (k := k) n a).1 + (halves (k := k) n a).2 * denNat (2 ^ n) = denNat a := by
  rw [halves_fst, halves_snd]
  exact (add_comm _ _).trans (denNat_div_add_mod k n a)

/-- **The encoding is injective on bounded words.** Recovery gives `denNat a = denNat b`, and
    both words are below `2^k`. -/
theorem halves_injective_bounded {k n a b : Nat} (ha : a < 2 ^ k) (hb : b < 2 ^ k)
    (h : halves (k := k) n a = halves (k := k) n b) : a = b := by
  have h1 := halves_recover (k := k) (n := n) (a := a)
  have h2 := halves_recover (k := k) (n := n) (a := b)
  rw [h] at h1
  have hc : ((a : Nat) : ZMod (2 ^ k)) = ((b : Nat) : ZMod (2 ^ k)) := h1.symm.trans h2
  have hm : a % 2 ^ k = b % 2 ^ k := (ZMod.natCast_eq_natCast_iff' a b (2 ^ k)).mp hc
  rwa [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at hm

/-- The `rotrVal` formula read into the ring, in the order the read produces it. -/
theorem denNat_rotrVal {k n a : Nat} :
    denNat (k := k) (rotrVal (k := k) n a)
      = denNat (k := k) (a / 2 ^ n)
        + denNat (k := k) (a % 2 ^ n) * denNat (k := k) (2 ^ (k - n)) := by
  simp only [rotrVal, denNat, Nat.cast_add, Nat.cast_mul]

/-- **The read refinement, through the actual `ringMul (-1)`.** The first slot of the raw
    product is the rotated word's value, for every `k`, `n`, `a`; no domain bound and no
    norm premise. This is AR-17's positive result in `ringMul` form. -/
theorem ringMul_halves_read {k n a : Nat} :
    (ringMul (-1) (halves (k := k) n a) (halvesMultiplier (k := k) n)).1
      = denNat (k := k) (rotrVal (k := k) n a) := by
  rw [ringMul_fst_eq, halves_fst, halves_snd, halvesMultiplier_fst, halvesMultiplier_snd,
    denNat_rotrVal]
  ring

/-- **`rotrVal` is the value of `BitVec.rotateRight`** on the bounded domain `a < 2^k`,
    `n < k`. Re-derived from `BitVec.toNat_rotateRight`. -/
theorem rotrVal_eq_rotrBv {k : Nat} (a n : Nat) (ha : a < 2 ^ k) (hnk : n < k) :
    (rotrBv (k := k) a n).toNat = rotrVal (k := k) n a := by
  have hxpow : 2 ^ n * 2 ^ (k - n) = 2 ^ k := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hA : a <<< (k - n) % 2 ^ k = (a % 2 ^ n) * 2 ^ (k - n) := by
    have hkey : a * 2 ^ (k - n) = a / 2 ^ n * 2 ^ k + (a % 2 ^ n) * 2 ^ (k - n) := by
      have hdiv := Nat.div_add_mod' a (2 ^ n)
      have hstep1 : a * 2 ^ (k - n)
          = (a / 2 ^ n * 2 ^ n + a % 2 ^ n) * 2 ^ (k - n) := by rw [hdiv]
      have hstep2 : (a / 2 ^ n * 2 ^ n + a % 2 ^ n) * 2 ^ (k - n)
          = a / 2 ^ n * (2 ^ n * 2 ^ (k - n)) + (a % 2 ^ n) * 2 ^ (k - n) := by
        rw [Nat.add_mul, mul_assoc]
      have hstep3 : a / 2 ^ n * (2 ^ n * 2 ^ (k - n)) + (a % 2 ^ n) * 2 ^ (k - n)
          = a / 2 ^ n * 2 ^ k + (a % 2 ^ n) * 2 ^ (k - n) := by
        rw [show 2 ^ n * 2 ^ (k - n) = 2 ^ k by exact hxpow]
      exact hstep1.trans (hstep2.trans hstep3)
    have hz0 : a / 2 ^ n * 2 ^ k % 2 ^ k = 0 := by
      rw [mul_comm]
      exact Nat.mul_mod_right (2 ^ k) (a / 2 ^ n)
    have hBlt : (a % 2 ^ n) * 2 ^ (k - n) < 2 ^ k := by
      have hlt1 : (a % 2 ^ n) * 2 ^ (k - n) < 2 ^ n * 2 ^ (k - n) :=
        Nat.mul_lt_mul_of_pos_right (Nat.mod_lt _ (Nat.two_pow_pos n)) (Nat.two_pow_pos _)
      have hlt2 : 2 ^ n * 2 ^ (k - n) = 2 ^ k := hxpow
      rw [hlt2] at hlt1
      exact hlt1
    have hmod : (a / 2 ^ n * 2 ^ k + (a % 2 ^ n) * 2 ^ (k - n)) % 2 ^ k
        = a % 2 ^ n * 2 ^ (k - n) := by
      rw [Nat.add_mod, hz0, Nat.zero_add, Nat.mod_eq_of_lt hBlt, Nat.mod_eq_of_lt hBlt]
    rw [Nat.shiftLeft_eq, hkey, hmod]
  have hq : a / 2 ^ n < 2 ^ (k - n) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos n), mul_comm (2 ^ (k - n)) (2 ^ n), hxpow]
    exact ha
  have hofnat : (BitVec.ofNat k a).toNat = a := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt ha]
  rw [rotrBv, BitVec.toNat_rotateRight, hofnat, Nat.mod_eq_of_lt hnk,
    Nat.shiftRight_eq_div_pow, hA]
  have hor := Nat.two_pow_add_eq_or_of_lt hq (a % 2 ^ n)
  rw [mul_comm (a % 2 ^ n) (2 ^ (k - n)), Nat.or_comm, ← hor, add_comm, rotrVal, mul_comm]

/-! ## §1.1 The width-32 instantiation (SHA-256 word width) -/

/-- Read refinement at width 32, every amount, no norm premise. -/
theorem read_refinement32 (n a : Nat) :
    (ringMul (-1) (halves (k := 32) n a) (halvesMultiplier (k := 32) n)).1
      = denNat (k := 32) (rotrVal (k := 32) n a) :=
  ringMul_halves_read

/-- Lossless recovery at width 32. -/
theorem recovery32 (n a : Nat) :
    (halves (k := 32) n a).1 + (halves (k := 32) n a).2 * denNat (k := 32) (2 ^ n)
      = denNat (k := 32) a :=
  halves_recover

/-- The width-32 rotation value is `BitVec.rotateRight`'s value on the `rotr_step` bounds. -/
theorem rotr32_eq_rotateRight (a n : Nat) (ha : a < 2 ^ 32) (hnk : n < 32) :
    (rotrBv (k := 32) a n).toNat = rotrVal (k := 32) n a :=
  rotrVal_eq_rotrBv a n ha hnk

/-- **Joint reachable witness at width 32**: word `1`, amount `7` (the σ₀ rotation amount),
    with the word bound, the amount bound, recovery, the read refinement and the `BitVec`
    link all at once. -/
theorem width32_witness :
    InWord 32 1 ∧ InAmount 32 7 ∧
      (ringMul (-1) (halves (k := 32) 7 1) (halvesMultiplier (k := 32) 7)).1
        = denNat (k := 32) (rotrVal (k := 32) 7 1) ∧
      (halves (k := 32) 7 1).1 + (halves (k := 32) 7 1).2 * denNat (k := 32) (2 ^ 7)
        = denNat (k := 32) 1 ∧
      (rotrBv (k := 32) 1 7).toNat = rotrVal (k := 32) 7 1 :=
  ⟨by decide, by decide, ringMul_halves_read, halves_recover,
    rotrVal_eq_rotrBv 1 7 (by decide) (by decide)⟩

/-! ## §2 The action property: exact failure -/

/-- **Exact coordinate form of the raw product**: first slot the rotated word's value (the
    read), second slot `high · 2^(k-n) - low`. -/
theorem product_snd (k n a : Nat) :
    (ringMul (-1) (halves (k := k) n a) (halvesMultiplier (k := k) n)).2
      = denNat (k := k) (a / 2 ^ n) * denNat (k := k) (2 ^ (k - n))
        - denNat (k := k) (a % 2 ^ n) := by
  rw [ringMul_snd_eq, halves_fst, halves_snd, halvesMultiplier_fst, halvesMultiplier_snd]
  ring

/-- **The exact second-coordinate difference** between the raw product and the re-encoded
    rotated word, as a function of `a` and `n`. -/
theorem action_gap_snd (k n a : Nat) :
    (ringMul (-1) (halves (k := k) n a) (halvesMultiplier (k := k) n)).2
      - (halves (k := k) n (rotrVal (k := k) n a)).2
    = denNat (k := k) (a / 2 ^ n) * denNat (k := k) (2 ^ (k - n))
      - denNat (k := k) (a % 2 ^ n)
      - denNat (k := k) (rotrVal (k := k) n a / 2 ^ n) := by
  rw [product_snd, halves_snd]

/-- First-coordinate difference: the product carries the whole rotated value, the re-encoding
    carries only its low `n` bits. -/
theorem action_gap_fst (k n a : Nat) :
    (ringMul (-1) (halves (k := k) n a) (halvesMultiplier (k := k) n)).1
      - (halves (k := k) n (rotrVal (k := k) n a)).1
    = denNat (k := k) (rotrVal (k := k) n a)
      - denNat (k := k) (rotrVal (k := k) n a % 2 ^ n) := by
  rw [ringMul_halves_read, halves_fst]

theorem rotrVal_one {k n : Nat} (hn : 0 < n) :
    rotrVal (k := k) n 1 = 2 ^ (k - n) := by
  have h1 : (1 : Nat) < 2 ^ n := Nat.one_lt_two_pow (by omega)
  simp only [rotrVal, Nat.div_eq_of_lt h1, Nat.mod_eq_of_lt h1, one_mul, zero_add]

theorem rotrVal_two_pow (k n : Nat) : rotrVal (k := k) n (2 ^ n) = 1 := by
  simp only [rotrVal, Nat.mod_self, Nat.div_self (Nat.two_pow_pos n), zero_mul, add_zero]

/-- The raw product's second slot at `a = 1` is `-1`, independent of `k` and `n`. -/
theorem product_snd_one (k n : Nat) (hn : 0 < n) :
    (ringMul (-1) (halves (k := k) n 1) (halvesMultiplier (k := k) n)).2 = -1 := by
  rw [product_snd]
  rw [Nat.div_eq_of_lt (Nat.one_lt_two_pow (by omega)),
    Nat.mod_eq_of_lt (Nat.one_lt_two_pow (by omega))]
  simp only [denNat, Nat.cast_zero, Nat.cast_one, zero_mul, zero_sub]

/-- The value `2^(k-n) / 2^n` is never `-1` in `ZMod (2^k)` on the bounded amount domain: it
    is below `2^k - 1`, and `-1` is `2^k - 1`. -/
theorem denNat_div_two_pow_ne_neg_one (k n : Nat) (hn : 0 < n) (hnk : n < k) :
    denNat (k := k) (2 ^ (k - n) / 2 ^ n) ≠ -1 := by
  intro h
  have hk : 2 ≤ k := by omega
  have hraw : ((2 ^ (k - n) / 2 ^ n : Nat) : ZMod (2 ^ k)) = -1 := by
    simpa only [denNat] using h
  have hzero : (((2 ^ (k - n) / 2 ^ n) + 1 : Nat) : ZMod (2 ^ k)) = 0 := by
    rw [Nat.cast_add, hraw, Nat.cast_one]
    ring
  have hdvd : 2 ^ k ∣ (2 ^ (k - n) / 2 ^ n) + 1 :=
    (ZMod.natCast_eq_zero_iff ((2 ^ (k - n) / 2 ^ n) + 1) (2 ^ k)).mp hzero
  have hdivle : 2 ^ (k - n) / 2 ^ n ≤ 2 ^ (k - 1) := by
    have h1 : 2 ^ (k - n) ≤ 2 ^ (k - 1) :=
      (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).mpr (by omega)
    exact le_trans (Nat.div_le_self _ _) h1
  have hpk : 2 ^ k = 2 * 2 ^ (k - 1) := by
    have h : (2 : Nat) ^ k = 2 ^ ((k - 1) + 1) :=
      congrArg (fun t => 2 ^ t) (by omega : k = (k - 1) + 1)
    rw [h, Nat.pow_succ, mul_comm]
  have h1lt : 1 < 2 ^ (k - 1) := Nat.one_lt_two_pow (by omega : k - 1 ≠ 0)
  have hlt : 2 ^ (k - n) / 2 ^ n + 1 < 2 ^ k := by
    calc 2 ^ (k - n) / 2 ^ n + 1 ≤ 2 ^ (k - 1) + 1 := Nat.add_le_add_right hdivle 1
      _ < 2 ^ (k - 1) + 2 ^ (k - 1) := Nat.add_lt_add_left h1lt (2 ^ (k - 1))
      _ = 2 * 2 ^ (k - 1) := by ring
      _ = 2 ^ k := hpk.symm
  have hle := Nat.le_of_dvd (Nat.succ_pos _) hdvd
  exact (not_le.mpr hlt) hle

/-- **The action property fails at `a = 1`**, for every `k ≥ 2` and `0 < n < k`: the product's
    second slot is `-1` while the re-encoded rotation's second slot is `2^(k-n) / 2^n`. -/
theorem mismatch_at_one (k n : Nat) (hn : 0 < n) (hnk : n < k) :
    ringMul (-1) (halves (k := k) n 1) (halvesMultiplier (k := k) n)
      ≠ halves (k := k) n (rotrVal (k := k) n 1) := by
  intro h
  have hs := congrArg Prod.snd h
  rw [product_snd_one k n hn, halves_snd, rotrVal_one hn] at hs
  exact denNat_div_two_pow_ne_neg_one k n hn hnk hs.symm

/-- The bounded-domain action equality is refuted at `a = 1`. -/
theorem action_fails (k n : Nat) (hn : 0 < n) (hnk : n < k) :
    ¬ (∀ a : Nat, a < 2 ^ k →
        ringMul (-1) (halves (k := k) n a) (halvesMultiplier (k := k) n)
          = halves (k := k) n (rotrVal (k := k) n a)) := by
  intro h
  exact mismatch_at_one k n hn hnk (h 1 (Nat.one_lt_two_pow (by omega)))

/-- **No multiplier repairs the halves pairing.** The two bounded witnesses `a = 1` and
    `a = 2^n` force `u.2 = 2^(k-n) / 2^n` and `u.2 = -1` at once; `denNat_div_two_pow_ne_neg_one`
    makes those incompatible. -/
theorem no_equivariant_multiplier (k n : Nat) (hn : 0 < n) (hnk : n < k) :
    ¬ ∃ u : ZMod (2 ^ k) × ZMod (2 ^ k), ∀ a : Nat, a < 2 ^ k →
      ringMul (-1) (halves (k := k) n a) u
        = halves (k := k) n (rotrVal (k := k) n a) := by
  rintro ⟨u, hu⟩
  have h1 : (1 : Nat) < 2 ^ k := Nat.one_lt_two_pow (by omega)
  have hp : 2 ^ n < 2 ^ k := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mpr hnk
  have hmod1 : (1 : Nat) % 2 ^ n = 1 := Nat.mod_eq_of_lt (Nat.one_lt_two_pow (by omega))
  have hdiv1 : (1 : Nat) / 2 ^ n = 0 := Nat.div_eq_of_lt (Nat.one_lt_two_pow (by omega))
  have hu2 : u.2 = denNat (k := k) (2 ^ (k - n) / 2 ^ n) := by
    have hs := congrArg Prod.snd (hu 1 h1)
    have hs' : denNat (k := k) (1 % 2 ^ n) * u.2 + denNat (k := k) (1 / 2 ^ n) * u.1
        = denNat (k := k) (rotrVal (k := k) n 1 / 2 ^ n) := hs
    rw [hmod1, hdiv1, rotrVal_one hn] at hs'
    simpa only [denNat, Nat.cast_one, Nat.cast_zero, one_mul, zero_mul, add_zero] using hs'
  have hu2neg : u.2 = -1 := by
    have hs := congrArg Prod.fst (hu (2 ^ n) hp)
    have hs' : denNat (k := k) (2 ^ n % 2 ^ n) * u.1
          + (-1) * denNat (k := k) (2 ^ n / 2 ^ n) * u.2
        = denNat (k := k) (rotrVal (k := k) n (2 ^ n) % 2 ^ n) := hs
    rw [Nat.mod_self, Nat.div_self (Nat.two_pow_pos n), rotrVal_two_pow, hmod1] at hs'
    have h : -u.2 = 1 := by
      simpa only [denNat, Nat.cast_one, Nat.cast_zero, zero_mul, one_mul, mul_one,
        neg_one_mul, zero_add] using hs'
    simpa using congrArg Neg.neg h
  exact denNat_div_two_pow_ne_neg_one k n hn hnk (hu2.symm.trans hu2neg)

/-- **Ground check at width 2, amount 1**: the product is `(2, -1)` while the re-encoded
    rotated word is `(0, 1)`, the smallest instance of the failure. -/
theorem width2_counterexample :
    ringMul (-1) (halves (k := 2) 1 1) (halvesMultiplier (k := 2) 1) = (2, -1)
      ∧ halves (k := 2) 1 (rotrVal (k := 2) 1 1) = (0, 1)
      ∧ ringMul (-1) (halves (k := 2) 1 1) (halvesMultiplier (k := 2) 1)
          ≠ halves (k := 2) 1 (rotrVal (k := 2) 1 1) := by
  decide

/-- The action failure at width 32, every amount `0 < n < 32`. -/
theorem mismatch_at_one_32 (n : Nat) (hn : 0 < n) (hnk : n < 32) :
    ringMul (-1) (halves (k := 32) n 1) (halvesMultiplier (k := 32) n)
      ≠ halves (k := 32) n (rotrVal (k := 32) n 1) :=
  mismatch_at_one 32 n hn hnk

/-- The four SHA-256 rotation amounts all fail the action equality at `a = 1`, including the
    three (`17`, `18`, `19`) whose multipliers are not norm-one. No norm-one premise is used. -/
theorem sha256_amounts_action_fail :
    (ringMul (-1) (halves (k := 32) 7 1) (halvesMultiplier (k := 32) 7)
        ≠ halves (k := 32) 7 (rotrVal (k := 32) 7 1))
      ∧ (ringMul (-1) (halves (k := 32) 17 1) (halvesMultiplier (k := 32) 17)
        ≠ halves (k := 32) 17 (rotrVal (k := 32) 17 1))
      ∧ (ringMul (-1) (halves (k := 32) 18 1) (halvesMultiplier (k := 32) 18)
        ≠ halves (k := 32) 18 (rotrVal (k := 32) 18 1))
      ∧ (ringMul (-1) (halves (k := 32) 19 1) (halvesMultiplier (k := 32) 19)
        ≠ halves (k := 32) 19 (rotrVal (k := 32) 19 1)) :=
  ⟨mismatch_at_one_32 7 (by norm_num) (by norm_num),
   mismatch_at_one_32 17 (by norm_num) (by norm_num),
   mismatch_at_one_32 18 (by norm_num) (by norm_num),
   mismatch_at_one_32 19 (by norm_num) (by norm_num)⟩

/-- No multiplier repairs the halves pairing at width 32 either. -/
theorem no_equivariant_multiplier32 (n : Nat) (hn : 0 < n) (hnk : n < 32) :
    ¬ ∃ u : ZMod (2 ^ 32) × ZMod (2 ^ 32), ∀ a : Nat, a < 2 ^ 32 →
      ringMul (-1) (halves (k := 32) n a) u
        = halves (k := 32) n (rotrVal (k := 32) n a) :=
  no_equivariant_multiplier 32 n hn hnk

/-! ## §3 Composition: what iterates -/

/-- One cell step: multiply the encoded state by the read multiplier. -/
def cellStep {k : Nat} (n : Nat) (z : ZMod (2 ^ k) × ZMod (2 ^ k)) :
    ZMod (2 ^ k) × ZMod (2 ^ k) :=
  ringMul (-1) z (halvesMultiplier (k := k) n)

/-- The `m`-fold raw cell iteration. -/
def cellIter {k : Nat} (n : Nat) :
    Nat → (ZMod (2 ^ k) × ZMod (2 ^ k)) → (ZMod (2 ^ k) × ZMod (2 ^ k))
  | 0, z => z
  | m + 1, z => cellStep (k := k) n (cellIter n m z)

/-- The `m`-fold multiplier under `ringMul (-1)`. -/
def ringPow {k : Nat} (z : ZMod (2 ^ k) × ZMod (2 ^ k)) :
    Nat → ZMod (2 ^ k) × ZMod (2 ^ k)
  | 0 => ringOne
  | m + 1 => ringMul (-1) (ringPow z m) z

/-- **Raw iteration is multiplication by the multiplier power.** -/
theorem cellIter_eq_ringMul_pow {k n m : Nat} (z : ZMod (2 ^ k) × ZMod (2 ^ k)) :
    cellIter (k := k) n m z
      = ringMul (-1) z (ringPow (k := k) (halvesMultiplier (k := k) n) m) := by
  induction m generalizing z with
  | zero =>
      show z = ringMul (-1) z ringOne
      rw [ringMul_one]
  | succ m ih =>
      show cellStep (k := k) n (cellIter (k := k) n m z)
        = ringMul (-1) z
            (ringMul (-1) (ringPow (k := k) (halvesMultiplier (k := k) n) m)
              (halvesMultiplier (k := k) n))
      rw [ih]
      show ringMul (-1)
            (ringMul (-1) z (ringPow (k := k) (halvesMultiplier (k := k) n) m))
            (halvesMultiplier (k := k) n)
        = ringMul (-1) z
            (ringMul (-1) (ringPow (k := k) (halvesMultiplier (k := k) n) m)
              (halvesMultiplier (k := k) n))
      rw [ringMul_assoc]

/-- The word-level `m`-fold right rotation by `n`. `k` is explicit: it occurs only in the
    body, not in the type. -/
def rotrIter (k : Nat) (n : Nat) : Nat → Nat → Nat
  | 0, a => a
  | m + 1, a => rotrVal (k := k) n (rotrIter k n m a)

/-- **Composition breaks at the first step.** No `m`-fold raw cell iteration of the encoding
    equals the encoding of the `m`-fold rotation: the step `m = 1` already fails. -/
theorem composition_breaks (k n : Nat) (hn : 0 < n) (hnk : n < k) :
    ¬ ∀ (m a : Nat), a < 2 ^ k →
      cellIter (k := k) n m (halves (k := k) n a)
        = halves (k := k) n (rotrIter k n m a) := by
  intro h
  exact action_fails k n hn hnk (fun a ha => by
    have h1 := h 1 a ha
    simpa only [cellIter, rotrIter, cellStep] using h1)

/-- **The composition that holds**: read, re-encode from the read value, multiply again. -/
theorem reencode_composes {k n m a : Nat} :
    (cellStep (k := k) n (halves (k := k) n (rotrIter k n m a))).1
      = denNat (k := k) (rotrIter k n (m + 1) a) := by
  simp only [cellStep, ringMul_halves_read, rotrIter]

/-! ## §4 Multiplicative re-encodings -/

/-- **No multiplicative injective re-encoding repairs the action.** If `A` were multiplicative
    and injective, equivariance of `A ∘ halves` would pull back along `A` to equivariance of
    the halves pairing itself, which `no_equivariant_multiplier` excludes. -/
theorem no_equivariant_after_multiplicative_reencoding (k n : Nat) (hn : 0 < n)
    (hnk : n < k)
    (A : ZMod (2 ^ k) × ZMod (2 ^ k) → ZMod (2 ^ k) × ZMod (2 ^ k))
    (hmul : ∀ z w, A (ringMul (-1) z w) = ringMul (-1) (A z) (A w))
    (hinj : Function.Injective A) :
    ¬ ∃ u : ZMod (2 ^ k) × ZMod (2 ^ k), ∀ a : Nat, a < 2 ^ k →
      ringMul (-1) (A (halves (k := k) n a)) u
        = A (halves (k := k) n (rotrVal (k := k) n a)) := by
  rintro ⟨u, hu⟩
  haveI : NeZero (2 ^ k) := ⟨(Nat.two_pow_pos k).ne'⟩
  obtain ⟨u', hu'⟩ := Finite.surjective_of_injective hinj u
  exact no_equivariant_multiplier k n hn hnk ⟨u', fun a ha => by
    have h := hu a ha
    rw [← hu'] at h
    rw [← hmul (halves (k := k) n a) u'] at h
    exact hinj h⟩

/-- **The bridge verdict at the bounded domain.** The read is faithful for every word; the raw
    product is not the re-encoded rotation; and no multiplier makes the halves pairing
    equivariant. -/
theorem bridge_verdict (k n : Nat) (hn : 0 < n) (hnk : n < k) :
    (∀ a : Nat, (ringMul (-1) (halves (k := k) n a) (halvesMultiplier (k := k) n)).1
        = denNat (k := k) (rotrVal (k := k) n a))
      ∧ (¬ ∀ a : Nat, a < 2 ^ k →
          ringMul (-1) (halves (k := k) n a) (halvesMultiplier (k := k) n)
            = halves (k := k) n (rotrVal (k := k) n a))
      ∧ (¬ ∃ u : ZMod (2 ^ k) × ZMod (2 ^ k), ∀ a : Nat, a < 2 ^ k →
          ringMul (-1) (halves (k := k) n a) u
            = halves (k := k) n (rotrVal (k := k) n a)) :=
  ⟨fun _ => ringMul_halves_read, action_fails k n hn hnk, no_equivariant_multiplier k n hn hnk⟩

/-! ## §5 Axiom cone -/

#print axioms ringMul_fst_eq
#print axioms ringMul_snd_eq
#print axioms halves_fst
#print axioms halves_snd
#print axioms halvesMultiplier_fst
#print axioms halvesMultiplier_snd
#print axioms denNat_div_add_mod
#print axioms halves_recover
#print axioms halves_injective_bounded
#print axioms denNat_rotrVal
#print axioms ringMul_halves_read
#print axioms rotrVal_eq_rotrBv
#print axioms read_refinement32
#print axioms recovery32
#print axioms rotr32_eq_rotateRight
#print axioms width32_witness
#print axioms product_snd
#print axioms action_gap_snd
#print axioms action_gap_fst
#print axioms rotrVal_one
#print axioms rotrVal_two_pow
#print axioms product_snd_one
#print axioms denNat_div_two_pow_ne_neg_one
#print axioms mismatch_at_one
#print axioms action_fails
#print axioms width2_counterexample
#print axioms mismatch_at_one_32
#print axioms sha256_amounts_action_fail
#print axioms no_equivariant_multiplier32
#print axioms no_equivariant_multiplier
#print axioms cellIter_eq_ringMul_pow
#print axioms composition_breaks
#print axioms reencode_composes
#print axioms no_equivariant_after_multiplicative_reencoding
#print axioms bridge_verdict

end AlgebraRotBridge32
