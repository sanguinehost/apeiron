/-
  `AlgebraRotate32.lean` — the rotation identity at the SHA-256 word width, and the exact place the
  two-slot ring reading fails there.

  Task AR-16 of the algebra-rooting wave. Owned file: this one, plus `AlgebraRotate32.md`. No
  production, pin, extraction-root or API edit; no commit.

  Every statement in `work/proofs/AlgebraRotate.lean` is at `ZMod (2 ^ 64)` / `BitVec 64`; SHA-2's
  word is 32-bit (`Apeiron.Crypto.Sha2.Word` is `BitVec 32`, `Sha2.lean:22`). Nothing is imported
  from that file: each step below is re-proved at width 32, the width SHA-2 actually uses.

  Proved here, in order.

  §1 The dial, transported to `ZMod (2 ^ 32)`: hyperbolic `σ = 1`, parabolic `0`, elliptic `−1`, and
      the root premise `r ^ 2 = σ` at `(σ, r) = (1, 1)`.
  §2 The 32-bit carrier `bvDenote32`, the evaluation identity for the two-slot expression, the two
      slots of the product, and what the expression is: one multiplication by `2^n + 2^(32−n)`.
  §3 The corrected value statement at width 32, in `ℕ` and in the ring, with both spellings of the
      shift factor identified.
  §4 The offset at width 32: its exact value, a non-zero witness for every `n < 32`, the decidable
      criterion for any `n < 32`, and the closed-form set of words at which it vanishes for
      `0 < n ≤ 16`.
  §5 The obstruction at width 32: no `A + B·x` reads a rotation by *any* amount `0 < n < 32`; AND by
      `1` is not affine; XOR by `1` is not affine; the `ringUpper`/`ringMul` two-slot read family is
      additive in the word, so no member of it is a rotation.
  §6 The consequence for `sigma0`, `sigma1`, `Sigma0`, `Sigma1` at width 32: which term has an exact
      algebraic reading, and the residual.

  Every shift amount is written as a natural number read into the ring, `((2 ^ n : ℕ) : ZMod
  (2 ^ 32))`, which is the element `Nat.cast_pow` produces. Concrete bitwise values are discharged by
  `decide` on small `Fin (2 ^ 32)` arithmetic, never by `norm_num` over rotation literals.

  Grades: `proved` for a named theorem whose `#print axioms` row at the end of this file is inside
  `{propext, Classical.choice, Quot.sound}`. Nothing here is a security, collision-resistance,
  constant-time or break claim.
-/

import WordRefinement
import Apeiron.Trichotomy.ChannelsAnyRing

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

namespace AlgebraRotate32

open apeiron_word
open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing

/-! ## §1 The dial, and the face the width-32 reading uses -/

/-- The hyperbolic face's σ in the dial's own ring, `ZMod (2 ^ 64)`, by `rfl` (`WordRefinement.lean:35`,
`denoteSigma s = denote (sigmaI64 s)`, `sigmaI64 cell.Sigma.Hyperbolic = 1#i64`). Re-proved here
rather than cited: `AlgebraRotate.lean` is a read-only reference and is not imported. -/
theorem denoteSigma_hyperbolic : denoteSigma cell.Sigma.Hyperbolic = (1 : ZMod (2 ^ 64)) := rfl

/-- The parabolic face's σ is `0`. -/
theorem denoteSigma_parabolic : denoteSigma cell.Sigma.Parabolic = (0 : ZMod (2 ^ 64)) := rfl

/-- The elliptic face's σ is `−1`. -/
theorem denoteSigma_elliptic : denoteSigma cell.Sigma.Elliptic = (-1 : ZMod (2 ^ 64)) := rfl

/-- **The face values the width-32 reading uses.** The dial is `denoteSigma : cell.Sigma → ZMod (2 ^ 64)`
by definition (`WordRefinement.lean:35`), and the three rows above give its values: hyperbolic `1`,
parabolic `0`, elliptic `−1`. The two-slot reading at width 32 instantiates `σ` with the hyperbolic
face, and the element it uses is the ring element `1` of `ZMod (2 ^ 32)` — the same number the
dial's hyperbolic value is, so `denoteSigma_hyperbolic` is the link between the executing dial and
the width-32 reading. (`AlgebraRotate.lean` carried a `denoteSigma32` transport of the dial value
through `ℕ`; run 1 of this file showed there is no `ℕ` coercion out of `ZMod n`, since `ZMod n` is
definitionally `Fin (n+1)` and instance search does not unfold it, so the transport is not needed
here and is not claimed.)

The premise `ChannelsAnyRing.ringUpper_mul` carries is `r ^ 2 = σ`. At the width-32 hyperbolic
instance it is the arithmetic identity `1 ^ 2 = 1`, so no number-theory hypothesis is needed. -/
theorem ringUpper_mul_root32 : (1 : ZMod (2 ^ 32)) ^ 2 = 1 := rfl

/-- The width-32 face value `1` is also available in the `ℕ`-cast spelling, so a statement may be
written with either and they mean the same element. -/
theorem cast_two_32 : ((2 : ℕ) : ZMod (2 ^ 32)) = (2 : ZMod (2 ^ 32)) := by rw [Nat.cast_ofNat]

/-! ## §2 The 32-bit carrier and what the two-slot expression computes -/

/-- **The 32-bit carrier denotation.** The SHA-2 word as an element of `ZMod (2 ^ 32)`: its unsigned
value read into the ring. This is the width-32 counterpart of `Denotation.lean:21`'s `denote`, which
covers `I64` only, and the `BitVec 32` half of ops-map row A1's second denotation. -/
def bvDenote32 (x : BitVec 32) : ZMod (2 ^ 32) := (x.toNat : ZMod (2 ^ 32))

/-- The carrier reads a word's unsigned value into the ring. -/
theorem bvDenote32_def (x : BitVec 32) : bvDenote32 x = (x.toNat : ZMod (2 ^ 32)) := rfl

/-- The reading of a product along the `x ↦ (x, 0)` line, in every commutative ring and so at width
32. σ cancels before it is used, so the identity needs no root at all. Re-proved here. -/
theorem ringUpper_mul_eval {R : Type*} [CommRing R] (σ r a b x : R) :
    ringUpper r (ringMul σ (x, 0) (a, b)) = x * (a + r * b) := by
  simp only [ringUpper, ringMul]
  ring

/-- The two slots of the ops-map S7 product at width 32, for every amount, with no hypothesis on `n`:
`2^n` in the first slot and `2^(32−n)` in the second. -/
theorem ringMul_slots32 (x : ZMod (2 ^ 32)) (n : Nat) :
    ringMul (1 : ZMod (2 ^ 32)) (x, (0 : ZMod (2 ^ 32)))
        ((2 ^ n : ZMod (2 ^ 32)), (2 ^ (32 - n) : ZMod (2 ^ 32)))
      = (x * (2 ^ n : ZMod (2 ^ 32)), x * (2 ^ (32 - n) : ZMod (2 ^ 32))) := by
  simp only [ringMul]
  ring

/-- The ops-map S7 candidate expression at the SHA-256 word width:
`ringUpper 1 (ringMul 1 (x, 0) (2^n, 2^(32−n)))`. -/
def rotExpr32 (x n : Nat) : ZMod (2 ^ 32) :=
  ringUpper (1 : ZMod (2 ^ 32))
    (ringMul (1 : ZMod (2 ^ 32)) (((x : Nat) : ZMod (2 ^ 32)), (0 : ZMod (2 ^ 32)))
      (((2 ^ n : Nat) : ZMod (2 ^ 32)), ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))))

/-- **What the candidate actually is at width 32**: one ring multiplication by `2^n + 2^(32−n)`, for
every word value and every amount, with no hypothesis on either. The second slot cannot supply the
wrapped high bits. -/
theorem rotExpr32_eq (x n : Nat) : rotExpr32 x n
    = (x : ZMod (2 ^ 32))
      * (((2 ^ n : Nat) : ZMod (2 ^ 32)) + ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))) := by
  have h := ringUpper_mul_eval (1 : ZMod (2 ^ 32)) (1 : ZMod (2 ^ 32))
    ((2 ^ n : Nat) : ZMod (2 ^ 32)) ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32)) (x : ZMod (2 ^ 32))
  calc rotExpr32 x n
      = ringUpper (1 : ZMod (2 ^ 32))
          (ringMul (1 : ZMod (2 ^ 32)) (((x : Nat) : ZMod (2 ^ 32)), (0 : ZMod (2 ^ 32)))
            (((2 ^ n : Nat) : ZMod (2 ^ 32)), ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32)))) := rfl
    _ = (x : ZMod (2 ^ 32))
        * (((2 ^ n : Nat) : ZMod (2 ^ 32)) + (1 : ZMod (2 ^ 32))
          * ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))) := h
    _ = (x : ZMod (2 ^ 32))
        * (((2 ^ n : Nat) : ZMod (2 ^ 32)) + ((2 ^ (32 - n) : Nat) : ZMod (2 ^ 32))) := by ring

/-! ## §3 The corrected value statement at width 32 -/

/-- **The rotation identity in `ℕ`, at width 32.** The rotate-left value of a 32-bit word is the
truncated left shift plus the wrapped high bits, `(x·2^n mod 2^32) + ⌊x / 2^(32−n)⌋`, for every
`n < 32`. `BitVec.toNat_rotateLeft` supplies the bitwise `|||` of those two operands and
`Nat.two_pow_add_eq_or_of_lt` turns that into their sum, because the wrapped part is below `2^n`
while the shift is a multiple of `2^n`. The only width-dependent step is `hxpow`. -/
theorem toNat_rotateLeft_eq_sum32 (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    (x.rotateLeft n).toNat = (x.toNat * 2 ^ n) % 2 ^ 32 + x.toNat / 2 ^ (32 - n) := by
  have hxpow : 2 ^ (32 - n) * 2 ^ n = 2 ^ 32 := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hA : (x.toNat * 2 ^ n) % 2 ^ 32 = 2 ^ n * (x.toNat % 2 ^ (32 - n)) := by
    have hmm := Nat.mul_mod_mul_left (2 ^ n) x.toNat (2 ^ (32 - n))
    rw [mul_comm (2 ^ n) x.toNat, mul_comm (2 ^ n) (2 ^ (32 - n)), hxpow] at hmm
    exact hmm
  have hq : x.toNat / 2 ^ (32 - n) < 2 ^ n := by
    refine (Nat.div_lt_iff_lt_mul (by positivity)).mpr ?_
    rw [mul_comm (2 ^ n) (2 ^ (32 - n)), hxpow]
    exact x.isLt
  rw [BitVec.toNat_rotateLeft, Nat.mod_eq_of_lt hn32, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow,
    hA]
  exact (Nat.two_pow_add_eq_or_of_lt hq (x.toNat % 2 ^ (32 - n))).symm

/-- The two spellings of the shift factor are the same element at width 32. -/
theorem two_pow_cast_eq32 (n : ℕ) : (2 : ZMod (2 ^ 32)) ^ n = ((2 ^ n : ℕ) : ZMod (2 ^ 32)) := by
  rw [Nat.cast_pow, Nat.cast_ofNat]

/-- **The rotation identity at the SHA-256 word width.**
`bvDenote32 (x.rotateLeft n) = bvDenote32 x * 2^n + ⌊x.toNat / 2^(32−n)⌋` for every `n < 32`. The
transfer is `congrArg` of the `ℕ` statement into the ring followed by `Nat.cast_add`,
`ZMod.natCast_mod` (which rewrites *out of* the modulus) and `Nat.cast_mul`: the `2^n` factor is an
`ℕ`-typed cast and the truncation is removed by `natCast_mod`, so no numeral is ever evaluated in the
ring. -/
theorem bvDenote32_rotateLeft_eq (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    bvDenote32 (x.rotateLeft n)
      = bvDenote32 x * ((2 ^ n : ℕ) : ZMod (2 ^ 32))
        + ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) := by
  have h := congrArg (fun t : ℕ => (t : ZMod (2 ^ 32))) (toNat_rotateLeft_eq_sum32 x n hn32)
  rw [Nat.cast_add, ZMod.natCast_mod, Nat.cast_mul] at h
  simpa only [bvDenote32] using h

/-- The same statement with the shift factor written as the ring power `(2 : ZMod (2 ^ 32)) ^ n`. -/
theorem bvDenote32_rotateLeft_eq_pow (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    bvDenote32 (x.rotateLeft n)
      = bvDenote32 x * (2 : ZMod (2 ^ 32)) ^ n
        + ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) := by
  rw [bvDenote32_rotateLeft_eq x n hn32, two_pow_cast_eq32]

/-- **The ops map's predicted value at width 32 is the rotation value.** So the candidate lemma's
right-hand side is right at width 32; only the left-hand side, the two-slot expression, is wrong. -/
theorem rotPredicted32_eq (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    ((x.toNat * 2 ^ n) % 2 ^ 32 + x.toNat / 2 ^ (32 - n) : ℕ)
      = bvDenote32 (x.rotateLeft n) := by
  have h := congrArg (fun t : ℕ => (t : ZMod (2 ^ 32))) (toNat_rotateLeft_eq_sum32 x n hn32)
  symm
  simpa only [bvDenote32] using h

/-- `bvDenote32` at a `BitVec` literal is the literal read into the ring: `BitVec.ofNat` stores
`k % 2^32` and `ZMod.natCast_mod` removes the modulus. -/
theorem bvDenote32_ofNat (k : ℕ) : bvDenote32 (BitVec.ofNat 32 k) = ((k : ℕ) : ZMod (2 ^ 32)) := by
  rw [bvDenote32_def, BitVec.toNat_ofNat, ZMod.natCast_mod]

/-- `2^32` is `0` in its own modulus: the truncation every transfer step needs. -/
theorem two_pow_32_cast_eq_zero : ((2 ^ 32 : ℕ) : ZMod (2 ^ 32)) = 0 := by
  have h := ZMod.natCast_mod (2 ^ 32) (2 ^ 32)
  rw [Nat.mod_self] at h
  exact h.symm

/-- The same truncation in the numeral spelling used below. -/
theorem two_pow_32_zmod_eq_zero : (2 ^ 32 : ZMod (2 ^ 32)) = 0 := by
  rw [two_pow_cast_eq32]
  exact two_pow_32_cast_eq_zero

/-- **No power of two strictly below the modulus is zero at width 32.** For every `0 < n < 32` the
element `2^n` of `ZMod (2 ^ 32)` is non-zero: a divisor of `2^32` cannot exceed `2^32`. -/
theorem two_pow_ne_zero32 (n : ℕ) (hn0 : 0 < n) (hn32 : n < 32) :
    ((2 ^ n : ℕ) : ZMod (2 ^ 32)) ≠ 0 := by
  intro h
  have hd : 2 ^ 32 ∣ 2 ^ n := (ZMod.natCast_eq_zero_iff (a := 2 ^ n) (b := 2 ^ 32)).mp h
  have hle : (2 ^ 32 : ℕ) ≤ 2 ^ n := Nat.le_of_dvd (Nat.two_pow_pos n) hd
  have hlt : (2 : ℕ) ^ n < 2 ^ 32 :=
    (Nat.pow_lt_pow_iff_right (a := 2) (n := n) (m := 32) (by decide)).mpr hn32
  omega

/-! ## §4 The offset at width 32: exactly where the two-slot expression fails -/

/-- **The offset at the word width.** The difference between what the ops-map S7 two-slot expression
computes and what the rotation computes is `x·2^(32−n) − ⌊x / 2^(32−n)⌋`, for every word and every
`n < 32`. The first slot of the product is the correct truncated shift; the second slot is a second
shift, not the wrapped high bits. -/
theorem rotExpr32_offset_eq (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    rotExpr32 x.toNat n - bvDenote32 (x.rotateLeft n)
      = bvDenote32 x * ((2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32))
        - ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) := by
  have h1 := rotExpr32_eq x.toNat n
  have h2 := bvDenote32_rotateLeft_eq x n hn32
  simp only [bvDenote32] at h1 h2 ⊢
  rw [h1, h2]
  ring

theorem one_bitvec32_toNat : ((1 : BitVec 32) : BitVec 32).toNat = 1 := by decide

/-- **The offset at the word `1`, for every amount below the width**, reads `2^(32−n)`. This is the
width-32 counterpart of the width-64 value `2^63` at `(x = 1, n = 1)`. -/
theorem rotExpr32_offset_one (n : Nat) (hn32 : n < 32) :
    rotExpr32 (1 : BitVec 32).toNat n - bvDenote32 ((1 : BitVec 32).rotateLeft n)
      = ((2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) := by
  rw [rotExpr32_offset_eq (1 : BitVec 32) n hn32]
  have hb : bvDenote32 (1 : BitVec 32) = 1 := by
    rw [bvDenote32_def, one_bitvec32_toNat, Nat.cast_one]
  have hq : ((1 : BitVec 32).toNat / 2 ^ (32 - n) : ℕ) = 0 := by
    rw [one_bitvec32_toNat]
    refine Nat.div_eq_of_lt ?_
    have h2 : 2 ≤ 2 ^ (32 - n) := by
      have h : 2 ^ (32 - n) = 2 ^ (31 - n) * 2 := by
        rw [← show (31 - n) + 1 = 32 - n by omega, Nat.pow_succ]
      rw [h]
      exact Nat.le_mul_of_pos_left 2 (Nat.two_pow_pos (31 - n))
    omega
  rw [hb, hq, Nat.cast_zero]
  ring

/-- **The offset is non-zero at the word `1` for every `n < 32`**, so the two-slot expression and the
rotation differ there by a non-zero element rather than agreeing up to a hidden summand. -/
theorem rotExpr32_offset_one_ne (n : Nat) (hn0 : 0 < n) (hn32 : n < 32) :
    rotExpr32 (1 : BitVec 32).toNat n - bvDenote32 ((1 : BitVec 32).rotateLeft n) ≠ 0 := by
  rw [rotExpr32_offset_one n hn32]
  exact two_pow_ne_zero32 (32 - n) (by omega) (by omega)

/-- **The exact decidable criterion at width 32**, for any word and any `n < 32`: the offset vanishes
if and only if the second slot `x·2^(32−n)` and the wrapped high bits `⌊x/2^(32−n)⌋` agree modulo
`2^32`. This is the precise statement of where the two-slot expression fails. -/
theorem rotExpr32_offset_eq_zero_iff (x : BitVec 32) (n : Nat) (hn32 : n < 32) :
    rotExpr32 x.toNat n - bvDenote32 (x.rotateLeft n) = 0
      ↔ (x.toNat * 2 ^ (32 - n)) % 2 ^ 32 = x.toNat / 2 ^ (32 - n) := by
  have hqlt : x.toNat / 2 ^ (32 - n) < 2 ^ 32 := by
    refine (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos (32 - n))).mpr ?_
    rw [mul_comm (2 ^ 32) (2 ^ (32 - n))]
    exact Nat.lt_of_lt_of_le x.isLt
      (Nat.le_mul_of_pos_left (2 ^ 32) (Nat.two_pow_pos (32 - n)))
  have hoff := rotExpr32_offset_eq x n hn32
  constructor
  · intro h
    have h0 : bvDenote32 x * ((2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32))
        - ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) = 0 := by
      rw [← hoff]
      exact h
    have h' : ((x.toNat * 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32))
        = ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) := by
      rw [Nat.cast_mul]
      simpa only [bvDenote32_def] using sub_eq_zero.mp h0
    have hiff := (ZMod.natCast_eq_natCast_iff' _ _ _).mp h'
    rw [Nat.mod_eq_of_lt hqlt] at hiff
    exact hiff
  · intro h
    have h' : ((x.toNat * 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32))
        = ((x.toNat / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) :=
      (ZMod.natCast_eq_natCast_iff' _ _ _).mpr (by rw [h, Nat.mod_eq_of_lt hqlt])
    rw [hoff, bvDenote32_def, ← Nat.cast_mul, h', sub_self]

/-- **Where the two-slot expression is exactly right at width 32.** The offset vanishes at every
word `x` that is a multiple of `2^n` and below `2^(32−n)`, for `0 < n ≤ 16`. This is the
substantive difference from width 64, where the corresponding vanishing set is empty apart from the
zero word: at width 32 the second slot `x·2^(32−n)` is congruent to `0` whenever `2^n ∣ x`, because
`2^n·2^(32−n) = 2^32 ≡ 0`. For `n = 1` that is every even word below `2^31`, a quarter of the word
space; the non-vanishing side is `rotExpr32_offset_one_ne` and the exact criterion for every word and
every `n < 32` is `rotExpr32_offset_eq_zero_iff`.

Only the containment direction is proved here. The full converse is proved independently, and with a
different route, in `work/proofs/AlgebraRotExact.lean` (`exact32_iff_lowbits` for `0 < n ≤ 16`,
`exact32_iff_digit` for every `0 < n < 32`): that file is a read-only reference, is not imported here,
and its closed form is not re-proved. -/
theorem rotExpr32_offset_eq_zero_of_dvd (x : BitVec 32) (n : Nat) (hn0 : 0 < n) (hn16 : n ≤ 16)
    (hlt : x.toNat < 2 ^ (32 - n)) (hdvd : 2 ^ n ∣ x.toNat) :
    rotExpr32 x.toNat n - bvDenote32 (x.rotateLeft n) = 0 := by
  have hn32 : n < 32 := by omega
  obtain ⟨j, hj⟩ := hdvd
  have hq0 : (x.toNat / 2 ^ (32 - n) : ℕ) = 0 := Nat.div_eq_of_lt hlt
  have hre : 2 ^ n * j * 2 ^ (32 - n) = j * 2 ^ 32 := by
    calc 2 ^ n * j * 2 ^ (32 - n) = j * (2 ^ n * 2 ^ (32 - n)) := by ring
      _ = j * 2 ^ (n + (32 - n)) := by rw [Nat.pow_add]
      _ = j * 2 ^ 32 := by rw [show (n + (32 - n) : ℕ) = 32 by omega]
  have hprod : ((x.toNat * 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) = 0 := by
    rw [hj, hre, Nat.cast_mul, two_pow_32_cast_eq_zero, mul_zero]
  rw [rotExpr32_offset_eq x n hn32]
  rw [bvDenote32_def, hq0, Nat.cast_zero, ← Nat.cast_mul]
  rw [hprod, sub_self]

/-! ## §5 The obstruction at the SHA-256 word width -/

/-- The denotation of rotating the word whose bits are `k` by `n`, at width 32. -/
def rotDenoteNat32 (k n : ℕ) : ZMod (2 ^ 32) :=
  bvDenote32 ((BitVec.ofNat 32 k).rotateLeft n)

/-- The width-32 rotation identity at a natural word index, for `k < 2^32` and `n < 32`. No rotation
literal is evaluated: the value comes from the cast of the `ℕ` formula. -/
theorem rotDenoteNat32_eq (k n : ℕ) (hk : k < 2 ^ 32) (hn32 : n < 32) :
    rotDenoteNat32 k n
      = (k : ZMod (2 ^ 32)) * ((2 ^ n : ℕ) : ZMod (2 ^ 32))
        + ((k / 2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) := by
  have h := bvDenote32_rotateLeft_eq (BitVec.ofNat 32 k) n hn32
  rw [bvDenote32_ofNat, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hk] at h
  exact h

/-- **Rotating the word with the single top bit set, by any `0 < n < 32`, gives `2^(n−1)`**: the top
bit wraps down to position `n−1`, and `2^31·2^n + 2^31/2^(32−n) = 2^(n−1)·2^32 + 2^(n−1)`, which
reduces to `2^(n−1)` modulo `2^32`. -/
theorem rotDenoteNat32_top (n : ℕ) (hn0 : 0 < n) (hn32 : n < 32) :
    rotDenoteNat32 (2 ^ 31) n = ((2 ^ (n - 1) : ℕ) : ZMod (2 ^ 32)) := by
  have h := rotDenoteNat32_eq (2 ^ 31) n (by decide) hn32
  have hq : 2 ^ 31 / 2 ^ (32 - n) = 2 ^ (n - 1) := by
    have hd : 2 ^ 31 = 2 ^ (31 - (32 - n)) * 2 ^ (32 - n) := by
      rw [← Nat.pow_add]
      congr 1
      omega
    have hdd : 2 ^ 31 / 2 ^ (32 - n) = 2 ^ (31 - (32 - n)) := by
      rw [hd]
      exact Nat.mul_div_left _ (Nat.two_pow_pos (32 - n))
    simpa [show 31 - (32 - n) = n - 1 by omega] using hdd
  have hpow : 2 ^ 31 * 2 ^ n = 2 ^ (n - 1) * 2 ^ 32 := by
    have hexp : (31 + n : ℕ) = (n - 1 + 32 : ℕ) := by omega
    rw [← Nat.pow_add, ← Nat.pow_add, hexp]
  rw [h, ← Nat.cast_mul, hq]
  rw [hpow]
  have hmul : ((2 ^ (n - 1) * 2 ^ 32 : ℕ) : ZMod (2 ^ 32)) = 0 := by
    rw [Nat.cast_mul, two_pow_32_cast_eq_zero, mul_zero]
  rw [hmul, zero_add]

/-- **No `A + B·x` reading computes a rotation, at width 32, for any amount below the width.** The
witness is the pair `x = y = 2^31`. As 32-bit words `2^31 + 2^31 = 0`, so a linear reading must send
their sum to `A + B·0 = A`; reading the zero word forces `A = 0`; and reading `2^31` then gives
`2·2^(n−1) = 2·(B·2^31) = B·2^32 = 0`, contradicting `2^n ≠ 0`.

Rotation is *not* parity-preserving (its low bit is bit `32−n` of the word), so parity cannot supply
this obstruction; the `2^31 + 2^31 = 0` witness can. -/
theorem no_affine_reads_rotation32 (n : Nat) (hn0 : 0 < n) (hn32 : n < 32) :
    ¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateLeft n) = A + B * bvDenote32 x := by
  rintro ⟨A, B, h⟩
  have hz : bvDenote32 ((0 : BitVec 32).rotateLeft n) = 0 := by
    have ht : ((0 : BitVec 32).rotateLeft n).toNat = 0 := by
      rw [toNat_rotateLeft_eq_sum32 (0 : BitVec 32) n hn32]
      simp
    rw [bvDenote32_def, ht, Nat.cast_zero]
  have hb0 : bvDenote32 (0 : BitVec 32) = 0 := by
    rw [bvDenote32_def]
    decide
  have hAz : bvDenote32 ((0 : BitVec 32).rotateLeft n) = A + B * bvDenote32 (0 : BitVec 32) :=
    h (0 : BitVec 32)
  rw [hz, hb0, mul_zero, add_zero] at hAz
  have htop : bvDenote32 ((BitVec.ofNat 32 (2 ^ 31)).rotateLeft n)
      = ((2 ^ (n - 1) : ℕ) : ZMod (2 ^ 32)) := by
    simpa only [rotDenoteNat32, bvDenote32] using rotDenoteNat32_top n hn0 hn32
  have hb31 : bvDenote32 (BitVec.ofNat 32 (2 ^ 31)) = ((2 ^ 31 : ℕ) : ZMod (2 ^ 32)) := by
    rw [bvDenote32_def, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by decide)]
  have hAb : bvDenote32 ((BitVec.ofNat 32 (2 ^ 31)).rotateLeft n)
      = A + B * bvDenote32 (BitVec.ofNat 32 (2 ^ 31)) := h (BitVec.ofNat 32 (2 ^ 31))
  rw [htop, hb31] at hAb
  have hmul : (2 : ZMod (2 ^ 32)) * ((2 ^ 31 : ℕ) : ZMod (2 ^ 32)) = 0 := by
    rw [← cast_two_32, ← Nat.cast_mul]
    rw [show (2 * 2 ^ 31 : ℕ) = 2 ^ 32 by decide]
    rw [two_pow_32_cast_eq_zero]
  have key : ((2 ^ n : ℕ) : ZMod (2 ^ 32)) = 0 := by
    calc ((2 ^ n : ℕ) : ZMod (2 ^ 32))
        = (2 : ZMod (2 ^ 32)) * ((2 ^ (n - 1) : ℕ) : ZMod (2 ^ 32)) := by
          have hp : 2 * 2 ^ (n - 1) = 2 ^ n := by
            calc 2 * 2 ^ (n - 1) = 2 ^ (n - 1) * 2 := by ring
              _ = 2 ^ (n - 1 + 1) := (Nat.pow_succ 2 (n - 1)).symm
              _ = 2 ^ n := by rw [show n - 1 + 1 = n by omega]
          symm
          rw [← cast_two_32, ← Nat.cast_mul, hp]
      _ = (2 : ZMod (2 ^ 32)) * (A + B * ((2 ^ 31 : ℕ) : ZMod (2 ^ 32))) := by
          rw [hAb]
      _ = 0 := by
          rw [← hAz, zero_add]
          calc (2 : ZMod (2 ^ 32)) * (B * ((2 ^ 31 : ℕ) : ZMod (2 ^ 32)))
              = B * ((2 : ZMod (2 ^ 32)) * ((2 ^ 31 : ℕ) : ZMod (2 ^ 32))) := by ring
            _ = B * 0 := by rw [hmul]
            _ = 0 := by ring
  exact two_pow_ne_zero32 n hn0 hn32 key

/-- Rotating the word `1` by one gives `2`. -/
theorem rotDenoteNat32_one_one : rotDenoteNat32 1 1 = (2 : ZMod (2 ^ 32)) := by
  have h := rotDenoteNat32_eq 1 1 (by decide) (by decide)
  rw [h, Nat.cast_one]
  have hd : ((1 : ℕ) / 2 ^ (32 - 1)) = 0 := Nat.div_eq_of_lt (by decide)
  rw [hd, Nat.cast_zero]
  ring

/-- Rotating the word `2^31 − 1` by one gives `2^32 − 2`. -/
theorem rotDenoteNat32_top_sub_one_one :
    rotDenoteNat32 (2 ^ 31 - 1) 1 = (2 ^ 32 - 2 : ZMod (2 ^ 32)) := by
  have h := rotDenoteNat32_eq (2 ^ 31 - 1) 1 (by decide) (by decide)
  rw [h, show (32 - 1 : ℕ) = 31 by decide, ← Nat.cast_mul,
    show (2 ^ 31 - 1) * 2 ^ 1 = 2 ^ 32 - 2 by decide,
    Nat.div_eq_of_lt (by decide), Nat.cast_zero]
  ring

/-- Rotating `2^31` by one gives `1`, which is `rotDenoteNat32_top` at `n = 1`. -/
theorem rotDenoteNat32_top_one : rotDenoteNat32 (2 ^ 31) 1 = (1 : ZMod (2 ^ 32)) := by
  rw [rotDenoteNat32_top 1 (by decide) (by decide)]
  rfl

/-- **The rotation by one is not additive at width 32**: at the three words `2^31 − 1`, `1`, `2^31`,
whose images are `↑(2^32 − 2)`, `↑2`, `↑1`, the first two do not sum to the third, because
`2^32 − 2 + 2 = 2^32 ≡ 0 ≠ 1`. -/
theorem rotDenoteNat32_add_defect :
    rotDenoteNat32 (2 ^ 31 - 1) 1 + rotDenoteNat32 1 1 ≠ rotDenoteNat32 (2 ^ 31) 1 := by
  rw [rotDenoteNat32_top_sub_one_one, rotDenoteNat32_one_one, rotDenoteNat32_top_one]
  have hsum : (2 ^ 32 - 2 : ZMod (2 ^ 32)) + (2 : ZMod (2 ^ 32)) = (2 ^ 32 : ZMod (2 ^ 32)) := by
    ring
  rw [hsum, two_pow_32_zmod_eq_zero]
  decide

/-- **The reason the family fails at width 32.** `ringUpper r (ringMul σ (·, 0) (a, b))` is additive in
the word: every member is `x ↦ x·(a + r·b)` (§2), so it maps a sum to a sum. A rotation does not. -/
theorem read_family_additive32 (σ r a b x y : ZMod (2 ^ 32)) :
    ringUpper r (ringMul σ (x + y, 0) (a, b))
      = ringUpper r (ringMul σ (x, 0) (a, b)) + ringUpper r (ringMul σ (y, 0) (a, b)) := by
  rw [ringUpper_mul_eval, ringUpper_mul_eval, ringUpper_mul_eval]
  ring

/-- **No `σ`, no channel root and no second slot in `ZMod (2 ^ 32)` make the two-slot read a rotation
by one**, on any word. The quantification is over all `k : ℕ`, which is stronger than the word
domain, and the three witnesses all satisfy `k < 2^32`. -/
theorem no_read_computes_rotation32 :
    ¬ ∃ (σ r a b : ZMod (2 ^ 32)), ∀ k : ℕ,
        ringUpper r (ringMul σ ((k : ZMod (2 ^ 32)), 0) (a, b)) = rotDenoteNat32 k 1 := by
  rintro ⟨σ, r, a, b, h⟩
  have key : ringUpper r (ringMul σ (((2 ^ 31 - 1 + 1 : ℕ) : ZMod (2 ^ 32)), 0) (a, b))
      = ringUpper r (ringMul σ (((2 ^ 31 - 1 : ℕ) : ZMod (2 ^ 32)), 0) (a, b))
        + ringUpper r (ringMul σ (((1 : ℕ) : ZMod (2 ^ 32)), 0) (a, b)) := by
    have e := ringUpper_mul_eval (σ := σ) (r := r) a b
    rw [e, e, e, Nat.cast_add]
    ring
  have c₁ := h (2 ^ 31 - 1)
  have c₂ := h (1 : ℕ)
  have c₃ := h (2 ^ 31)
  rw [show (2 ^ 31 - 1 + 1 : ℕ) = 2 ^ 31 by decide] at key
  rw [c₁, c₂, c₃] at key
  exact rotDenoteNat32_add_defect key.symm

/-! ### §5b AND and XOR against an affine reading, at width 32 -/

theorem fin32_and_0_1 : ((0 : Fin (2 ^ 32)) &&& (1 : Fin (2 ^ 32))) = 0 := by decide
theorem fin32_and_1_1 : ((1 : Fin (2 ^ 32)) &&& (1 : Fin (2 ^ 32))) = 1 := by decide
theorem fin32_and_3_1 : ((3 : Fin (2 ^ 32)) &&& (1 : Fin (2 ^ 32))) = 1 := by decide

/-- **AND is not affine at the word width**, in the corpus's carrier `ZMod (2 ^ 32) = Fin (2 ^ 32)`.
`x ↦ x &&& 1` takes `0, 1, 3` to `0, 1, 1`, while `A + B·x` takes them to `A, A + B, A + 3B`, so
`A = 0` and `B = 1` are forced and the third point gives `3 ≠ 1`. This is ops-map obstruction O2 at
the crypto width. -/
theorem and_one_not_affine_Fin32 :
    ¬ ∃ A B : Fin (2 ^ 32),
      (fun x : Fin (2 ^ 32) => A + B * x) = fun x => x &&& (1 : Fin (2 ^ 32)) := by
  rintro ⟨A, B, h⟩
  have h0 := congrFun h (0 : Fin (2 ^ 32))
  have h1 := congrFun h (1 : Fin (2 ^ 32))
  have h3 := congrFun h (3 : Fin (2 ^ 32))
  simp only [fin32_and_0_1, fin32_and_1_1, fin32_and_3_1] at h0 h1 h3
  simp only [mul_zero, add_zero] at h0
  subst h0
  simp only [zero_add, mul_one] at h1
  subst h1
  simp only [zero_add, mul_one] at h3
  have h3ne : ((3 : Fin (2 ^ 32))) ≠ 1 := by decide
  exact h3ne h3

/-- The correspondence the ops map's `ZMod` spelling uses: `ZMod (2 ^ 32)` and `Fin (2 ^ 32)` are the
same type definitionally (`Mathlib/Data/ZMod/Defs.lean:149-151`), so `Equiv.refl` identifies them. -/
def zmod32_fin32_equiv : Fin (2 ^ 32) ≃ ZMod (2 ^ 32) := Equiv.refl _

/-- **AND is not affine, in the ops map's `ZMod (2 ^ 32)` spelling**, with the coefficients read
through the identification. -/
theorem and_one_not_affine_ZMod32 :
    ¬ ∃ A B : ZMod (2 ^ 32),
      (fun x : Fin (2 ^ 32) => zmod32_fin32_equiv.symm A + zmod32_fin32_equiv.symm B * x)
        = fun x => x &&& (1 : Fin (2 ^ 32)) := by
  rintro ⟨A, B, h⟩
  exact and_one_not_affine_Fin32 ⟨zmod32_fin32_equiv.symm A, zmod32_fin32_equiv.symm B, h⟩

theorem fin32_xor_0_1 : ((0 : Fin (2 ^ 32)) ^^^ (1 : Fin (2 ^ 32))) = 1 := by decide
theorem fin32_xor_1_1 : ((1 : Fin (2 ^ 32)) ^^^ (1 : Fin (2 ^ 32))) = 0 := by decide
theorem fin32_xor_2_1 : ((2 : Fin (2 ^ 32)) ^^^ (1 : Fin (2 ^ 32))) = 3 := by decide

/-- **XOR by a constant is not affine at the word width**, in the corpus's carrier. `x ↦ x ^^^ 1`
takes `0, 1, 2` to `1, 0, 3`, so `A = f(0) = 1` and `B = f(1) − f(0) = −1` are forced, and the third
point gives `1 + (−1)·2 = 2^32 − 1 ≠ 3`. This is ops-map obstruction O4 at the crypto width, the
reading the `Ch` and `Maj` composites of `Sha2.lean:73,76` would need. -/
theorem xor_one_not_affine_Fin32 :
    ¬ ∃ A B : Fin (2 ^ 32),
      (fun x : Fin (2 ^ 32) => A + B * x) = fun x => x ^^^ (1 : Fin (2 ^ 32)) := by
  rintro ⟨A, B, h⟩
  have h0 := congrFun h (0 : Fin (2 ^ 32))
  have h1 := congrFun h (1 : Fin (2 ^ 32))
  have h2 := congrFun h (2 : Fin (2 ^ 32))
  simp only [fin32_xor_0_1, fin32_xor_1_1, fin32_xor_2_1] at h0 h1 h2
  simp only [mul_zero, add_zero] at h0
  subst h0
  simp only [zero_add, mul_one] at h1
  rw [add_comm] at h1
  have hB : B = -(1 : Fin (2 ^ 32)) := eq_neg_of_add_eq_zero_left h1
  have hBmax : (-(1 : Fin (2 ^ 32))) = 2 ^ 32 - 1 := by decide
  rw [hB, hBmax] at h2
  have h2ne : ((1 : Fin (2 ^ 32)) + (2 ^ 32 - 1 : Fin (2 ^ 32)) * 2) ≠ 3 := by decide
  exact h2ne h2

/-- **XOR by a constant is not affine, in the ops map's `ZMod (2 ^ 32)` spelling.** -/
theorem xor_one_not_affine_ZMod32 :
    ¬ ∃ A B : ZMod (2 ^ 32),
      (fun x : Fin (2 ^ 32) => zmod32_fin32_equiv.symm A + zmod32_fin32_equiv.symm B * x)
        = fun x => x ^^^ (1 : Fin (2 ^ 32)) := by
  rintro ⟨A, B, h⟩
  exact xor_one_not_affine_Fin32 ⟨zmod32_fin32_equiv.symm A, zmod32_fin32_equiv.symm B, h⟩

/-! ## §6 The consequence for `sigma0`, `sigma1`, `Sigma0`, `Sigma1` at width 32

FIPS 180-4 §4.1.2-4.1.3 at the width SHA-2 uses, with `Sha2.rotr` (`Sha2.lean:50`) and `Sha2.shrx`
(`:54`):

    sigma0 x = ROTR^7  x ⊕ ROTR^18 x ⊕ SHR^3  x      (Sha2.lean:60)
    sigma1 x = ROTR^17 x ⊕ ROTR^19 x ⊕ SHR^10 x      (Sha2.lean:64)
    Sigma0 x = ROTR^2  x ⊕ ROTR^13 x ⊕ ROTR^22 x     (Sha2.lean:67)
    Sigma1 x = ROTR^6  x ⊕ ROTR^11 x ⊕ ROTR^25 x     (Sha2.lean:70)

Each σ function has three terms. The two rotation terms are rotations at width 32 and therefore have
the exact algebraic reading of §3; the third is a truncation, not a rotation; and the `⊕` that combines
them is not the ring addition. -/

/-- A rotate *right* by `0 < n < 32` is the rotate *left* by `32 − n`, the form §3 reads: both
bodies are the same `|||` of the two shifts, in opposite order. -/
theorem rotateRight_eq_rotateLeft32 (x : BitVec 32) (n : Nat) (hn0 : 0 < n) (hn32 : n < 32) :
    x.rotateRight n = x.rotateLeft (32 - n) := by
  have h1 : n % 32 = n := Nat.mod_eq_of_lt hn32
  have h2 : (32 - n) % 32 = 32 - n := Nat.mod_eq_of_lt (by omega)
  have h3 : 32 - (32 - n) = n := by omega
  rw [BitVec.rotateRight_def, BitVec.rotateLeft_def, h1, h2, h3, BitVec.or_comm]

/-- No `A + B·x` reading computes a rotate *right* either, at width 32, for any `0 < n < 32`. -/
theorem no_affine_reads_rotateRight32 (n : Nat) (hn0 : 0 < n) (hn32 : n < 32) :
    ¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight n) = A + B * bvDenote32 x := by
  rintro ⟨A, B, h⟩
  refine no_affine_reads_rotation32 (32 - n) (by omega) (by omega) ⟨A, B, fun x => ?_⟩
  rw [← rotateRight_eq_rotateLeft32 x n hn0 hn32]
  exact h x

/-- `sigma0` at width 32. -/
def sigma0_32 (x : BitVec 32) : BitVec 32 := x.rotateRight 7 ^^^ x.rotateRight 18 ^^^ (x >>> 3)

/-- `sigma1` at width 32. -/
def sigma1_32 (x : BitVec 32) : BitVec 32 := x.rotateRight 17 ^^^ x.rotateRight 19 ^^^ (x >>> 10)

/-- `Sigma0` at width 32. -/
def Sigma0_32 (x : BitVec 32) : BitVec 32 :=
  x.rotateRight 2 ^^^ x.rotateRight 13 ^^^ x.rotateRight 22

/-- `Sigma1` at width 32. -/
def Sigma1_32 (x : BitVec 32) : BitVec 32 :=
  x.rotateRight 6 ^^^ x.rotateRight 11 ^^^ x.rotateRight 25

/-- **The two rotation terms of `sigma0` have no `A + B·x` reading** at width 32, amounts 7 and 18. -/
theorem sigma0_32_rotation_terms_not_affine :
    (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 7) = A + B * bvDenote32 x)
      ∧ (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 18) = A + B * bvDenote32 x) :=
  ⟨no_affine_reads_rotateRight32 7 (by decide) (by decide),
    no_affine_reads_rotateRight32 18 (by decide) (by decide)⟩

/-- **The two rotation terms of `sigma1` have no `A + B·x` reading** at width 32, amounts 17 and 19. -/
theorem sigma1_32_rotation_terms_not_affine :
    (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 17) = A + B * bvDenote32 x)
      ∧ (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 19) = A + B * bvDenote32 x) :=
  ⟨no_affine_reads_rotateRight32 17 (by decide) (by decide),
    no_affine_reads_rotateRight32 19 (by decide) (by decide)⟩

/-- **All three terms of `Sigma0` have no `A + B·x` reading** at width 32, amounts 2, 13 and 22: here
every term is a rotation. -/
theorem Sigma0_32_terms_not_affine :
    (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 2) = A + B * bvDenote32 x)
      ∧ (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 13) = A + B * bvDenote32 x)
      ∧ (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 22) = A + B * bvDenote32 x) :=
  ⟨no_affine_reads_rotateRight32 2 (by decide) (by decide),
    no_affine_reads_rotateRight32 13 (by decide) (by decide),
    no_affine_reads_rotateRight32 22 (by decide) (by decide)⟩

/-- **All three terms of `Sigma1` have no `A + B·x` reading** at width 32, amounts 6, 11 and 25. -/
theorem Sigma1_32_terms_not_affine :
    (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 6) = A + B * bvDenote32 x)
      ∧ (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 11) = A + B * bvDenote32 x)
      ∧ (¬ ∃ A B : ZMod (2 ^ 32), ∀ x : BitVec 32,
        bvDenote32 (x.rotateRight 25) = A + B * bvDenote32 x) :=
  ⟨no_affine_reads_rotateRight32 6 (by decide) (by decide),
    no_affine_reads_rotateRight32 11 (by decide) (by decide),
    no_affine_reads_rotateRight32 25 (by decide) (by decide)⟩

/-! ### §6b The residual: `⊕` is not the ring addition

The residual after the two rotation terms of each σ function have their exact readings is the `⊕`
that combines the three terms, plus the third term itself, which for `sigma0`/`sigma1` is a
truncation (`SHR^3`, `SHR^10`) and not a rotation. Two facts about that residual, both `decide`d on
closed `Fin (2 ^ 32)` arithmetic so no rotation literal is ever evaluated: XOR is not the ring
addition, and it already differs from it on a single carry. -/

/-- XOR is not the ring addition at the word width: one carry separates them, `3 ^^^ 1 = 2` against
`3 + 1 = 4`. This is the step that no `A + B·x` sum of term values can reproduce, independently of
which terms are chosen. -/
theorem xor_ne_add_32 : ((3 : Fin (2 ^ 32)) ^^^ (1 : Fin (2 ^ 32))) ≠ 3 + 1 := by decide

/-- Three XORed terms differ from the three summed terms at the word width:
`(3 ^^^ 1) ^^^ 2 = 0` against `3 + 1 + 2 = 6`. This is the shape of the σ residual: three exact
term readings, combined by `⊕`, do not add up in the ring. -/
theorem xor_three_ne_add_32 :
    (((3 : Fin (2 ^ 32)) ^^^ (1 : Fin (2 ^ 32))) ^^^ (2 : Fin (2 ^ 32))) ≠ 3 + 1 + 2 := by decide

end AlgebraRotate32

#print axioms AlgebraRotate32.denoteSigma_hyperbolic
#print axioms AlgebraRotate32.denoteSigma_parabolic
#print axioms AlgebraRotate32.denoteSigma_elliptic
#print axioms AlgebraRotate32.ringUpper_mul_root32
#print axioms AlgebraRotate32.cast_two_32
#print axioms AlgebraRotate32.bvDenote32_def
#print axioms AlgebraRotate32.ringUpper_mul_eval
#print axioms AlgebraRotate32.ringMul_slots32
#print axioms AlgebraRotate32.rotExpr32_eq
#print axioms AlgebraRotate32.toNat_rotateLeft_eq_sum32
#print axioms AlgebraRotate32.two_pow_cast_eq32
#print axioms AlgebraRotate32.bvDenote32_rotateLeft_eq
#print axioms AlgebraRotate32.bvDenote32_rotateLeft_eq_pow
#print axioms AlgebraRotate32.rotPredicted32_eq
#print axioms AlgebraRotate32.bvDenote32_ofNat
#print axioms AlgebraRotate32.two_pow_32_cast_eq_zero
#print axioms AlgebraRotate32.two_pow_32_zmod_eq_zero
#print axioms AlgebraRotate32.two_pow_ne_zero32
#print axioms AlgebraRotate32.rotExpr32_offset_eq
#print axioms AlgebraRotate32.one_bitvec32_toNat
#print axioms AlgebraRotate32.rotExpr32_offset_one
#print axioms AlgebraRotate32.rotExpr32_offset_one_ne
#print axioms AlgebraRotate32.rotExpr32_offset_eq_zero_iff
#print axioms AlgebraRotate32.rotExpr32_offset_eq_zero_of_dvd
#print axioms AlgebraRotate32.no_affine_reads_rotation32
#print axioms AlgebraRotate32.rotDenoteNat32_eq
#print axioms AlgebraRotate32.rotDenoteNat32_top
#print axioms AlgebraRotate32.rotDenoteNat32_one_one
#print axioms AlgebraRotate32.rotDenoteNat32_top_sub_one_one
#print axioms AlgebraRotate32.rotDenoteNat32_top_one
#print axioms AlgebraRotate32.rotDenoteNat32_add_defect
#print axioms AlgebraRotate32.read_family_additive32
#print axioms AlgebraRotate32.no_read_computes_rotation32
#print axioms AlgebraRotate32.fin32_and_0_1
#print axioms AlgebraRotate32.fin32_and_1_1
#print axioms AlgebraRotate32.fin32_and_3_1
#print axioms AlgebraRotate32.and_one_not_affine_Fin32
#print axioms AlgebraRotate32.and_one_not_affine_ZMod32
#print axioms AlgebraRotate32.fin32_xor_0_1
#print axioms AlgebraRotate32.fin32_xor_1_1
#print axioms AlgebraRotate32.fin32_xor_2_1
#print axioms AlgebraRotate32.xor_one_not_affine_Fin32
#print axioms AlgebraRotate32.xor_one_not_affine_ZMod32
#print axioms AlgebraRotate32.rotateRight_eq_rotateLeft32
#print axioms AlgebraRotate32.no_affine_reads_rotateRight32
#print axioms AlgebraRotate32.sigma0_32_rotation_terms_not_affine
#print axioms AlgebraRotate32.sigma1_32_rotation_terms_not_affine
#print axioms AlgebraRotate32.Sigma0_32_terms_not_affine
#print axioms AlgebraRotate32.Sigma1_32_terms_not_affine
#print axioms AlgebraRotate32.xor_ne_add_32
#print axioms AlgebraRotate32.xor_three_ne_add_32
